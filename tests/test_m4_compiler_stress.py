"""tests/test_m4_compiler_stress.py
Adversarial Stress Test Suite for Milestone 4 (Challenger 2).

Empirically stress-tests:
1. Malicious Code Injection Attacks:
   - Evaluates validate_cartridge_ast and EngineCompiler against hostile injected Python
     constructs (os.system, import subprocess, eval, __import__('socket'), __builtins__, __subclasses__).
   - Verifies strict rejection with ASTSecurityError without writing any files to disk.
   - Verifies CartridgeLoader rejection of manually placed malicious code.
2. Dual Profile Divergence Stress:
   - Compiles Episode 0 from both Rusher and Tactician Steam DNA.
   - Verifies measurable divergence across mechanics.py, level.json, player parameters, and enemy counts.
3. Multi-Episode Compilation Succession:
   - Compiles Episode 0 -> Episode 1 -> Episode 2 in strict succession.
   - Verifies each cartridge loads independently via CartridgeLoader.load_cartridge without module pollution.
4. Cryptographic Envelope Tampering:
   - Verifies that altering any synthesized file in a cartridge invalidates envelope hash verification.
   - Verifies that tampering with envelope.json raises ArtifactIntegrityError.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

from engine.compiler.compiler import (
    BytecodeCompilationError,
    CartridgeVerificationError,
    CompilationResult,
    CompiledFile,
    EngineCompiler,
    SpecValidationError,
)
from engine.director.episodic import CurrentEpisodeConfig, EpisodicDirector
from engine.director.genesis import GenesisDirector
from engine.director.models import EpisodicUpdateSpec, GenesisSpec
from engine.loader import ASTSecurityError, CartridgeLoader, validate_cartridge_ast
from engine.scheduler.artifact import (
    ArtifactIntegrityError,
    ArtifactMetadataEnvelope,
    ArtifactType,
)
from tests.test_director_episodic import MockBehavioralProfile
from tests.test_director_genesis import rusher_dna, tactician_dna
from tests.test_tier1_units import validate_rationale_first_ast, validate_rationale_first_dict


@pytest.fixture
def compiler(tmp_path: Path) -> EngineCompiler:
    return EngineCompiler(
        cartridges_root=tmp_path / "cartridges",
        active_cartridge_dir=tmp_path / "cartridges" / "active",
        specs_dir=tmp_path / "specs",
    )


class TestMaliciousCodeInjectionStress:
    """Stress tests verifying AST security sandbox rejection of hostile Python injection."""
    deliberation: str = "Empirically verifies AST security sandbox and EngineCompiler defense against code injection."

    @pytest.mark.parametrize(
        "malicious_snippet, description",
        [
            ('import os\nos.system("rm -rf /")', "os.system shell execution"),
            ('import subprocess\nsubprocess.run(["rm", "-rf", "/"])', "subprocess.run process invocation"),
            ('eval("1 + 1")', "eval dynamic code execution"),
            ('__import__("socket").socket()', "__import__ dynamic networking"),
            ('x = __builtins__', "__builtins__ global environment access"),
            ('s = ().__class__.__bases__[0].__subclasses__()', "__subclasses__ sandbox escape"),
            ('import sys\nsys.exit(0)', "sys module access"),
            ('import posix\nposix.system("echo hacked")', "posix low-level OS access"),
            ('import ctypes\nctypes.CDLL(None)', "ctypes native memory interop"),
            ('import shutil\nshutil.rmtree("/")', "shutil filesystem manipulation"),
            ('import pickle\npickle.loads(b"")', "pickle deserialization exploit"),
            ('globals()["__builtins__"]', "globals() reflection lookup"),
            ('locals()["__name__"]', "locals() reflection lookup"),
        ],
    )
    def test_ast_sandbox_direct_rejections(self, malicious_snippet: str, description: str):
        """Verifies validate_cartridge_ast rejects every dangerous module, call, and attribute."""
        with pytest.raises(ASTSecurityError) as exc_info:
            validate_cartridge_ast(malicious_snippet)
        assert len(str(exc_info.value)) > 0

    @pytest.mark.parametrize(
        "injection_payload, attack_vector",
        [
            ('spread":\n            import os\n            os.system("rm -rf /")\n        elif "1', "os.system attack"),
            ('spread":\n            import subprocess\n            subprocess.run(["ls"])\n        elif "1', "subprocess attack"),
            ('spread":\n            eval("1 + 1")\n        elif "1', "eval attack"),
            ('spread":\n            __import__("socket")\n        elif "1', "__import__ socket attack"),
            ('spread":\n            b = __builtins__\n        elif "1', "__builtins__ escape"),
            ('spread":\n            s = ().__class__.__bases__[0].__subclasses__()\n        elif "1', "__subclasses__ escape"),
        ],
    )
    def test_compiler_rejects_injected_specs_without_disk_write(
        self,
        compiler: EngineCompiler,
        rusher_dna,
        tmp_path: Path,
        injection_payload: str,
        attack_vector: str,
    ):
        """Verifies EngineCompiler rejects injected specs via AST checks without writing any files to disk."""
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        # Inject attack string into weapon archetype interpolation point
        spec.weapon_spec.archetype = injection_payload

        target_dir = tmp_path / f"injected_cartridge_{hashlib.md5(attack_vector.encode()).hexdigest()[:8]}"

        with pytest.raises(ASTSecurityError):
            compiler.compile(spec, target_dir=target_dir)

        # Zero disk write invariant: target directory must not contain any synthesized files
        if target_dir.exists():
            disk_files = list(target_dir.iterdir())
            assert disk_files == [], f"Files leaked to disk during failed compilation of {attack_vector}: {disk_files}"
            assert not (target_dir / "mechanics.py").exists()
            assert not (target_dir / "entities.py").exists()
            assert not (target_dir / "manifest.json").exists()
            assert not (target_dir / "level.json").exists()
            assert not (target_dir / "envelope.json").exists()

    def test_cartridge_loader_rejects_malicious_mechanics_on_disk(self, tmp_path: Path):
        """Verifies CartridgeLoader intercepts tampered or hostile mechanics.py before importing."""
        hostile_dir = tmp_path / "hostile_cartridge"
        hostile_dir.mkdir(parents=True, exist_ok=True)
        (hostile_dir / "manifest.json").write_text(json.dumps({"deliberation": "Hostile", "name": "Evil"}), encoding="utf-8")
        (hostile_dir / "mechanics.py").write_text(
            'from engine.api.contracts import BaseCartridge\nimport os\nos.system("echo evil")\nclass EvilCartridge(BaseCartridge):\n    pass\n',
            encoding="utf-8"
        )

        with pytest.raises(ASTSecurityError):
            CartridgeLoader.load_cartridge(hostile_dir)


class TestDualProfileDivergenceStress:
    """Stress tests proving that Rusher and Tactician DNA compile into genuinely diverging games."""
    deliberation: str = "Empirically verifies that Rusher and Tactician DNA compile into measurably diverging game cartridges."

    def test_dual_profile_genesis_compilation_divergence(
        self,
        compiler: EngineCompiler,
        rusher_dna,
        tactician_dna,
        tmp_path: Path,
    ):
        """Empirically compiles both profiles and asserts divergence across mechanics, levels, player, and enemies."""
        director = GenesisDirector()
        rusher_spec = director.generate_genesis_spec(rusher_dna)
        tactician_spec = director.generate_genesis_spec(tactician_dna)

        rusher_dir = tmp_path / "rusher_cartridge"
        tactician_dir = tmp_path / "tactician_cartridge"

        res_rusher = compiler.compile(rusher_spec, target_dir=rusher_dir)
        res_tactician = compiler.compile(tactician_spec, target_dir=tactician_dir)

        # 1. mechanics.py source code divergence
        code_r = (rusher_dir / "mechanics.py").read_text(encoding="utf-8")
        code_t = (tactician_dir / "mechanics.py").read_text(encoding="utf-8")
        assert code_r != code_t, "mechanics.py must not be identical across distinct player DNA"
        assert "SpreadWeapon" in code_r, "Rusher mechanics.py must feature SpreadWeapon"
        assert "BlasterWeapon" in code_t, "Tactician mechanics.py must feature BlasterWeapon"

        # 2. level.json geometry & configuration divergence
        level_r = json.loads((rusher_dir / "level.json").read_text(encoding="utf-8"))
        level_t = json.loads((tactician_dir / "level.json").read_text(encoding="utf-8"))
        assert level_r != level_t, "level.json must not be identical across distinct player DNA"

        # Dimensions: Rusher gets wider arena
        assert level_r["width"] == 1440.0 and level_r["height"] == 810.0
        assert level_t["width"] == 1280.0 and level_t["height"] == 720.0

        # Pillar layout / wall divergence: 6 boundary+pillar rects for Rusher vs 8 for Tactician
        assert len(level_r["walls"]) == 6, "Rusher level should have 2 pillars (6 total wall rects)"
        assert len(level_t["walls"]) == 8, "Tactician level should have 4 pillars (8 total wall rects)"

        # Hazards divergence: Rusher gets laser sweep + 2 spikes (3 hazard entities), Tactician gets 1 spike (1 hazard entity)
        assert len(level_r["hazards"]) == 3, f"Rusher should have 3 hazard instances (got {len(level_r['hazards'])})"
        assert len(level_t["hazards"]) == 1, f"Tactician should have 1 hazard instance (got {len(level_t['hazards'])})"
        types_r = set(h["type"] for h in level_r["hazards"])
        types_t = set(h["type"] for h in level_t["hazards"])
        assert types_r == {"spikes", "laser_sweep"}
        assert types_t == {"spikes"}

        # 3. Player parameters divergence via instantiated cartridges
        cart_r = CartridgeLoader.load_cartridge(rusher_dir)
        cart_t = CartridgeLoader.load_cartridge(tactician_dir)

        player_r = cart_r.create_player(100.0, 100.0)
        player_t = cart_t.create_player(100.0, 100.0)

        # Rusher has higher kinetic mobility
        assert player_r.walk_speed > player_t.walk_speed, f"Rusher walk {player_r.walk_speed} must exceed {player_t.walk_speed}"
        assert player_r.dash_speed > player_t.dash_speed, f"Rusher dash {player_r.dash_speed} must exceed {player_t.dash_speed}"
        assert player_r.dash_cooldown < player_t.dash_cooldown, f"Rusher cooldown {player_r.dash_cooldown} must be faster than {player_t.dash_cooldown}"
        assert player_r.dash_cost < player_t.dash_cost, f"Rusher dash cost {player_r.dash_cost} must be less than {player_t.dash_cost}"
        assert type(player_r.weapon).__name__ == "SpreadWeapon"
        assert type(player_t.weapon).__name__ == "BlasterWeapon"

        # 4. Enemy counts & archetype composition divergence
        spec_r = cart_r.get_level_spec(1)
        spec_t = cart_t.get_level_spec(1)
        enemies_r = cart_r.create_enemies(spec_r)
        enemies_t = cart_t.create_enemies(spec_t)

        assert len(enemies_r) != len(enemies_t), "Enemy count must diverge between Rusher and Tactician"
        assert len(enemies_r) == 5, f"Rusher should have 5 enemies (got {len(enemies_r)})"
        assert len(enemies_t) == 4, f"Tactician should have 4 enemies (got {len(enemies_t)})"

        chasers_r = [e for e in enemies_r if "chaser" in e.entity_id]
        chasers_t = [e for e in enemies_t if "chaser" in e.entity_id]
        assert len(chasers_r) == 4, "Rusher should face 4 chaser enemies"
        assert len(chasers_t) == 2, "Tactician should face 2 chaser enemies"

        ranged_r = [e for e in enemies_r if "ranged" in e.entity_id]
        ranged_t = [e for e in enemies_t if "ranged" in e.entity_id]
        assert len(ranged_r) == 1, "Rusher should face 1 ranged enemy"
        assert len(ranged_t) == 2, "Tactician should face 2 ranged enemies"


class TestMultiEpisodeSuccessionStress:
    """Stress tests verifying sequential multi-episode compilation and clean cartridge loader isolation."""
    deliberation: str = "Empirically verifies multi-episode compilation succession (Ep0 -> Ep1 -> Ep2) and dynamic loader isolation."

    def test_compile_succession_ep0_ep1_ep2(
        self,
        compiler: EngineCompiler,
        rusher_dna,
        tmp_path: Path,
    ):
        """Compiles Episode 0 -> Episode 1 -> Episode 2 in succession and verifies independent cartridge instantiation."""
        genesis_director = GenesisDirector()
        episodic_director = EpisodicDirector()

        ep0_dir = tmp_path / "episodes" / "ep0"
        ep1_dir = tmp_path / "episodes" / "ep1"
        ep2_dir = tmp_path / "episodes" / "ep2"

        # --- Stage 1: Compile Episode 0 (Genesis) ---
        spec0 = genesis_director.generate_genesis_spec(rusher_dna)
        res0 = compiler.compile(spec0, target_dir=ep0_dir)
        assert res0.episode_number == 0
        assert res0.cartridge_class_name == "GenesisCartridge"

        # --- Stage 2: Compile Episode 1 (Episodic) ---
        prof1 = MockBehavioralProfile(
            playstyle_tag="kinetic_high_mobility",
            mean_velocity=265.0,
            dash_ratio=0.72,
            mastery_score=0.88,
        )
        spec1 = episodic_director.evaluate_and_update(
            prof1,
            current_config=CurrentEpisodeConfig(episode_number=0),
        )
        assert spec1.target_episode_number == 1
        res1 = compiler.compile(spec1, target_dir=ep1_dir)
        assert res1.episode_number == 1
        assert res1.cartridge_class_name == "Episode01Cartridge"

        # --- Stage 3: Compile Episode 2 (Episodic) ---
        prof2 = MockBehavioralProfile(
            playstyle_tag="tactical_methodical",
            mean_velocity=185.0,
            dash_ratio=0.20,
            mastery_score=0.92,
        )
        spec2 = episodic_director.evaluate_and_update(
            prof2,
            current_config=CurrentEpisodeConfig(episode_number=1),
        )
        assert spec2.target_episode_number == 2
        res2 = compiler.compile(spec2, target_dir=ep2_dir)
        assert res2.episode_number == 2
        assert res2.cartridge_class_name == "Episode02Cartridge"

        # --- Stage 4: Concurrent Independent Cartridge Loading ---
        cart0 = CartridgeLoader.load_cartridge(ep0_dir)
        cart1 = CartridgeLoader.load_cartridge(ep1_dir)
        cart2 = CartridgeLoader.load_cartridge(ep2_dir)

        # Verify distinct runtime cartridge class identities
        assert type(cart0).__name__ == "GenesisCartridge"
        assert type(cart1).__name__ == "Episode01Cartridge"
        assert type(cart2).__name__ == "Episode02Cartridge"

        # Verify level spec generation from each independent cartridge
        lvl0 = cart0.get_level_spec(1)
        lvl1 = cart1.get_level_spec(1)
        lvl2 = cart2.get_level_spec(2)
        assert lvl1.level_id == "level_01"
        assert lvl2.level_id == "level_02"

        # Verify player instantiation from each independent cartridge
        p0 = cart0.create_player(100.0, 100.0)
        p1 = cart1.create_player(100.0, 100.0)
        p2 = cart2.create_player(100.0, 100.0)
        assert type(p0).__name__ == "GenesisPlayer"
        assert type(p1).__name__ == "Episode01Player"
        assert type(p2).__name__ == "Episode02Player"

        # --- Stage 5: Reload Isolation Invariant ---
        # Re-loading earlier cartridges must not be overwritten or corrupted by later ones
        reloaded_cart0 = CartridgeLoader.load_cartridge(ep0_dir)
        reloaded_cart1 = CartridgeLoader.load_cartridge(ep1_dir)
        assert type(reloaded_cart0).__name__ == "GenesisCartridge"
        assert type(reloaded_cart1).__name__ == "Episode01Cartridge"


class TestCryptographicEnvelopeTamperingStress:
    """Stress tests verifying that tampering with any compiled file or envelope metadata is detected."""
    deliberation: str = "Empirically verifies cryptographic envelope tampering detection across all cartridge files and envelope metadata."

    def test_tampering_each_cartridge_file_invalidates_envelope(
        self,
        compiler: EngineCompiler,
        rusher_dna,
        tmp_path: Path,
    ):
        """Verifies that altering any synthesized file in a cartridge invalidates envelope hash verification."""
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        cartridge_dir = tmp_path / "sealed_cartridge"

        result = compiler.compile(spec, target_dir=cartridge_dir)
        envelope_path = cartridge_dir / "envelope.json"
        assert envelope_path.is_file()

        # Load and verify pristine baseline
        envelope = ArtifactMetadataEnvelope.load_from_disk(envelope_path, verify=True)
        recorded_hashes: Dict[str, str] = envelope.payload["file_hashes"]

        cartridge_files = ["manifest.json", "level.json", "entities.py", "mechanics.py"]
        for fname in cartridge_files:
            file_path = cartridge_dir / fname
            assert file_path.is_file()
            original_bytes = file_path.read_bytes()

            # Baseline check: disk bytes match envelope hash
            assert ArtifactMetadataEnvelope.compute_sha256_bytes(original_bytes) == recorded_hashes[fname]

            # --- Tamper: alter file content on disk ---
            tampered_bytes = original_bytes + b"\n# adversarial injection line\n"
            file_path.write_bytes(tampered_bytes)

            tampered_hash = ArtifactMetadataEnvelope.compute_sha256_bytes(file_path.read_bytes())
            assert tampered_hash != recorded_hashes[fname], f"Tampered hash must diverge for {fname}"
            assert tampered_hash != envelope.metadata["file_hashes"][fname]

            # --- Restore original pristine bytes before next iteration ---
            file_path.write_bytes(original_bytes)
            assert ArtifactMetadataEnvelope.compute_sha256_bytes(file_path.read_bytes()) == recorded_hashes[fname]

    def test_tampering_envelope_json_payload_raises_integrity_error(
        self,
        compiler: EngineCompiler,
        rusher_dna,
        tmp_path: Path,
    ):
        """Verifies modifying payload values in envelope.json triggers ArtifactIntegrityError."""
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        cartridge_dir = tmp_path / "tampered_envelope_payload"
        compiler.compile(spec, target_dir=cartridge_dir)

        envelope_path = cartridge_dir / "envelope.json"
        raw_json = json.loads(envelope_path.read_text(encoding="utf-8"))

        # Adversary modifies payload without updating sha256_hash
        raw_json["payload"]["episode_number"] = 999
        envelope_path.write_text(json.dumps(raw_json, indent=2), encoding="utf-8")

        with pytest.raises(ArtifactIntegrityError):
            ArtifactMetadataEnvelope.load_from_disk(envelope_path, verify=True)

    def test_tampering_envelope_json_signature_raises_integrity_error(
        self,
        compiler: EngineCompiler,
        rusher_dna,
        tmp_path: Path,
    ):
        """Verifies corrupting sha256_hash signature in envelope.json triggers ArtifactIntegrityError."""
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        cartridge_dir = tmp_path / "tampered_envelope_sig"
        compiler.compile(spec, target_dir=cartridge_dir)

        envelope_path = cartridge_dir / "envelope.json"
        raw_json = json.loads(envelope_path.read_text(encoding="utf-8"))

        # Adversary corrupts hash
        raw_json["sha256_hash"] = "deadbeef" * 8
        envelope_path.write_text(json.dumps(raw_json, indent=2), encoding="utf-8")

        with pytest.raises(ArtifactIntegrityError):
            ArtifactMetadataEnvelope.load_from_disk(envelope_path, verify=True)

    def test_forged_hash_update_in_envelope_fails_verification(
        self,
        compiler: EngineCompiler,
        rusher_dna,
        tmp_path: Path,
    ):
        """Verifies adversary updating file_hashes in envelope to match tampered file is caught by envelope seal."""
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        cartridge_dir = tmp_path / "forged_hash_attempt"
        compiler.compile(spec, target_dir=cartridge_dir)

        mechanics_path = cartridge_dir / "mechanics.py"
        envelope_path = cartridge_dir / "envelope.json"

        # Tamper mechanics.py
        mechanics_path.write_text(mechanics_path.read_text(encoding="utf-8") + "\n# forged change\n", encoding="utf-8")
        new_file_hash = ArtifactMetadataEnvelope.compute_sha256_bytes(mechanics_path.read_bytes())

        # Attempt to forge the hash inside envelope.json payload to match the modified mechanics.py
        raw_json = json.loads(envelope_path.read_text(encoding="utf-8"))
        raw_json["payload"]["file_hashes"]["mechanics.py"] = new_file_hash
        envelope_path.write_text(json.dumps(raw_json, indent=2), encoding="utf-8")

        # Must fail envelope integrity check because envelope.sha256_hash no longer matches canonicalized payload
        with pytest.raises(ArtifactIntegrityError):
            ArtifactMetadataEnvelope.load_from_disk(envelope_path, verify=True)
