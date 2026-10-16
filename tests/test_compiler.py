"""tests/test_compiler.py
Comprehensive unit test suite for Autonomous SWE Engine Compiler (Milestone 4, Feature 12).
Verifies:
  1. GenesisSpec (Episode 0) compilation, file generation, active promotion, and cartridge loading.
  2. EpisodicUpdateSpec (Episode 1+) compilation with clamped parameters.
  3. AST Security Invariant enforcement (banned modules and dangerous calls strictly rejected).
  4. Rationale-First Epistemic Architecture (deliberation first in all AST classes and JSON schemas).
  5. Bytecode compilation verification.
  6. Cryptographic SHA-256 sealing in ArtifactMetadataEnvelope.
  7. Gameplay parameter divergence between Rusher and Tactician Genesis compilations.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import pytest

from engine.compiler.compiler import (
    BytecodeCompilationError,
    CartridgeVerificationError,
    CompilationResult,
    CompiledFile,
    EngineCompiler,
    SpecValidationError,
)
from engine.director.episodic import EpisodicDirector
from engine.director.genesis import GenesisDirector
from engine.director.models import EpisodicUpdateSpec, GenesisSpec
from engine.loader import ASTSecurityError, CartridgeLoader, validate_cartridge_ast
from engine.scheduler.artifact import ArtifactMetadataEnvelope, ArtifactType
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


class TestEngineCompilerGenesis:
    """Verifies Episode 0 compilation from Steam Genesis specifications."""

    def test_compile_genesis_spec(self, compiler: EngineCompiler, rusher_dna, tmp_path: Path):
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        target = tmp_path / "genesis_cartridge"

        result = compiler.compile(spec, target_dir=target, promote_to_active=True)

        assert isinstance(result, CompilationResult)
        assert result.ast_validated is True
        assert result.bytecode_compiled is True
        assert result.cartridge_loaded is True
        assert result.cartridge_class_name == "GenesisCartridge"

        # Check files on disk
        assert (target / "manifest.json").is_file()
        assert (target / "level.json").is_file()
        assert (target / "entities.py").is_file()
        assert (target / "mechanics.py").is_file()
        assert (target / "envelope.json").is_file()

        # Check active promotion
        active = compiler.active_cartridge_dir
        assert (active / "mechanics.py").is_file()

        # Check cartridge loader instantiation
        cartridge = CartridgeLoader.load_cartridge(target)
        level_spec = cartridge.get_level_spec(1)
        assert level_spec.width == spec.level_dimensions.width
        assert len(level_spec.walls) > 0

        player = cartridge.create_player(120.0, 360.0)
        assert player.walk_speed == spec.player_params.walk_speed
        assert player.dash_speed == spec.player_params.dash_speed

        enemies = cartridge.create_enemies(level_spec)
        assert len(enemies) > 0

        hazards = cartridge.create_hazards(level_spec)
        assert len(hazards) > 0


class TestEngineCompilerEpisodic:
    """Verifies Episode 1+ compilation from live telemetry episodic specifications."""

    def test_compile_episodic_update_spec(self, compiler: EngineCompiler, tmp_path: Path):
        director = EpisodicDirector()
        profile = MockBehavioralProfile(
            playstyle_tag="kinetic_high_mobility",
            mean_velocity=260.0,
            dash_ratio=0.65,
            mastery_score=0.90,
        )
        spec = director.evaluate_and_update(profile)
        target = tmp_path / "episode_02_cartridge"

        result = compiler.compile(spec, target_dir=target)

        assert result.episode_number == spec.target_episode_number
        assert result.cartridge_class_name == f"Episode{spec.target_episode_number:02d}Cartridge"

        cartridge = CartridgeLoader.load_cartridge(target)
        level_spec = cartridge.get_level_spec(spec.target_episode_number)
        assert level_spec.level_id == spec.target_episode_id

        enemies = cartridge.create_enemies(level_spec)
        assert len(enemies) > 0


class TestCompilerRationaleFirstAndSecurity:
    """Verifies strict AST security rejection and Rationale-First ordering invariants."""

    def test_generated_python_ast_rationale_first(self, compiler: EngineCompiler, rusher_dna, tmp_path: Path):
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        target = tmp_path / "genesis_test_ast"
        result = compiler.compile(spec, target_dir=target)

        # Validate AST of generated Python files
        for rel_name in ("mechanics.py", "entities.py"):
            code = (target / rel_name).read_text(encoding="utf-8")
            assert validate_rationale_first_ast(code) is True, f"Failed rationale AST check in {rel_name}"

        # Validate JSON dictionary ordering
        for rel_name in ("manifest.json", "level.json"):
            with open(target / rel_name, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert validate_rationale_first_dict(data) is True, f"Failed dict order check in {rel_name}"

        # Validate CompilationResult serialization order
        res_dict = result.to_dict()
        assert validate_rationale_first_dict(res_dict) is True

    def test_compiler_classes_ast_rationale_first(self):
        """Validates that compiler source files conform to Rationale-First AST rules."""
        compiler_path = Path(__file__).resolve().parent.parent / "engine" / "compiler" / "compiler.py"
        codegen_path = Path(__file__).resolve().parent.parent / "engine" / "compiler" / "codegen.py"
        assert validate_rationale_first_ast(compiler_path.read_text("utf-8")) is True
        assert validate_rationale_first_ast(codegen_path.read_text("utf-8")) is True

    def test_ast_security_rejection_of_malicious_code(self):
        unsafe_snippets = [
            "import os\nos.system('echo hacked')",
            "import sys\nsys.exit(1)",
            "import subprocess\nsubprocess.run(['ls'])",
            "import socket\ns = socket.socket()",
            "eval('1 + 1')",
            "exec('x = 2')",
        ]
        for snippet in unsafe_snippets:
            with pytest.raises(ASTSecurityError):
                validate_cartridge_ast(snippet)

    def test_spec_missing_deliberation_rejected(self, compiler: EngineCompiler, rusher_dna):
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        spec.deliberation = ""
        with pytest.raises(SpecValidationError):
            compiler.compile(spec)


class TestCryptographicSealing:
    """Verifies cryptographic SHA-256 sealing and envelope integrity."""

    def test_envelope_sealing_and_verification(self, compiler: EngineCompiler, rusher_dna, tmp_path: Path):
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        target = tmp_path / "genesis_envelope"
        result = compiler.compile(spec, target_dir=target)

        envelope = result.envelope
        assert envelope is not None
        assert envelope.artifact_type == ArtifactType.GAME_MODULE
        assert envelope.source_agent == "SWECompiler"
        assert envelope.target_agent == "BuildSupervisor"
        assert envelope.verify_integrity(raise_on_error=True) is True

        # Recompute hash of each file directly and compare
        for fname, cfile in result.files.items():
            disk_bytes = (target / fname).read_bytes()
            computed_h = ArtifactMetadataEnvelope.compute_sha256_bytes(disk_bytes)
            assert computed_h == cfile.sha256_hash


class TestProfileDivergenceCompilation:
    """Proves two distinct Steam profiles compile into two genuinely distinct games."""

    def test_rusher_vs_tactician_game_divergence(self, compiler: EngineCompiler, rusher_dna, tactician_dna, tmp_path: Path):
        director = GenesisDirector()
        rusher_spec = director.generate_genesis_spec(rusher_dna)
        tactician_spec = director.generate_genesis_spec(tactician_dna)

        rusher_dir = tmp_path / "cartridge_rusher"
        tactician_dir = tmp_path / "cartridge_tactician"

        res_rusher = compiler.compile(rusher_spec, target_dir=rusher_dir)
        res_tactician = compiler.compile(tactician_spec, target_dir=tactician_dir)

        cart_rusher = CartridgeLoader.load_cartridge(rusher_dir)
        cart_tactician = CartridgeLoader.load_cartridge(tactician_dir)

        p_rusher = cart_rusher.create_player(100.0, 100.0)
        p_tactician = cart_tactician.create_player(100.0, 100.0)

        # Mobility parameter divergence
        assert p_rusher.walk_speed > p_tactician.walk_speed
        assert p_rusher.dash_speed > p_tactician.dash_speed
