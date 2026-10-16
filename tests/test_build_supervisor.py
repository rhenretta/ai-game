"""tests/test_build_supervisor.py
Authoritative unit and integration test suite for BuildSupervisor 5-Gate pipeline (Feature 24).
Verifies Gates A, E, D, B, and C in computational dependency order, Rationale-First field ordering,
and hermetic promotion to cartridges/active/.
Strictly 0 occurrences of forbidden legacy branding tokens.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import shutil
import struct
import wave
import pytest
from PIL import Image

from engine.assets.palette import STYLE_BIBLE_16_PALETTE
from engine.compiler.compiler import EngineCompiler
from engine.loader import CartridgeLoader
from engine.supervisor.models import (
    AssetValidationReport,
    BuildEvaluationReport,
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    RemedyType,
    SupervisorEvaluationReport,
    SupervisorGateVerdict,
)
from engine.supervisor.supervisor import BuildSupervisor
from tests.test_tier1_units import validate_rationale_first_ast


@pytest.fixture
def supervisor_env(tmp_path: Path) -> BuildSupervisor:
    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    compiler = EngineCompiler(
        cartridges_root=cartridges_root,
        active_cartridge_dir=active_dir,
        specs_dir=tmp_path / "specs",
    )
    return BuildSupervisor(
        cartridges_root=cartridges_root,
        active_cartridge_dir=active_dir,
        compiler=compiler,
        tile_size=32.0,
        max_retries=3,
        early_death_tick_threshold=180,
    )


def create_mock_candidate(dest: Path) -> Path:
    """Creates a self-contained candidate cartridge directory with valid multimodal assets."""
    dest.mkdir(parents=True, exist_ok=True)

    (dest / "manifest.json").write_text(json.dumps({
        "cartridge_id": "cand_open_genre_01",
        "title": "Aether Void",
        "genre": "tactical_deckbuilder",
        "version": "1.0.0",
        "entrypoint": "mechanics.py",
        "author": "Autonomous Studio Swarm",
    }, indent=2), encoding="utf-8")

    (dest / "level.json").write_text(json.dumps({
        "level_id": "level_01",
        "name": "Level 1",
        "spawn_point": [120.0, 360.0],
        "exit_point": [1160.0, 360.0],
        "walls": [
            [0.0, 0.0, 1280.0, 32.0],
            [0.0, 688.0, 1280.0, 32.0],
            [0.0, 0.0, 32.0, 720.0],
            [1248.0, 0.0, 32.0, 720.0],
        ],
        "enemies": [
            {"id": "enemy_1", "type": "chaser", "x": 600.0, "y": 360.0}
        ],
        "hazards": [
            {"id": "hazard_1", "type": "spike", "x": 400.0, "y": 360.0, "w": 32.0, "h": 32.0}
        ]
    }, indent=2), encoding="utf-8")

    (dest / "entities.py").write_text('''"""entities.py"""
from typing import Any
from engine.api.contracts import BasePlayer, BaseEnemy, BaseHazard

class MockPlayer(BasePlayer):
    deliberation: str = "Mock player entity for testing."
    def __init__(self, x: float = 120.0, y: float = 360.0):
        super().__init__(x=x, y=y, width=32.0, height=32.0, speed=200.0, max_health=100.0)

class MockEnemy(BaseEnemy):
    deliberation: str = "Mock enemy entity for testing."
    def __init__(self, enemy_id: str = "e1", x: float = 600.0, y: float = 360.0):
        super().__init__(enemy_id=enemy_id, enemy_type="chaser", x=x, y=y, width=32.0, height=32.0, speed=100.0, health=50.0, damage=10.0)

class MockHazard(BaseHazard):
    deliberation: str = "Mock hazard entity for testing."
    def __init__(self, hazard_id: str = "h1", x: float = 400.0, y: float = 360.0):
        super().__init__(hazard_id=hazard_id, hazard_type="spike", x=x, y=y, width=32.0, height=32.0, damage=10.0)
''', encoding="utf-8")

    (dest / "mechanics.py").write_text('''"""mechanics.py"""
from typing import Any, Dict, List, Optional
from engine.api.contracts import BaseCartridge, BaseScene, BaseLevelSpec, BasePlayer, BaseEnemy, BaseHazard
from engine.telemetry.probes import AgentTelemetryProbe
from entities import MockPlayer, MockEnemy, MockHazard

class CustomScene(BaseScene):
    deliberation: str = "Custom scene implementing open-genre game loop."
    def __init__(self, engine: Any):
        super().__init__(engine)
        self.state_counter = 0
        self.probe = AgentTelemetryProbe(
            recorder=getattr(engine, "telemetry_recorder", None),
            probe_name="turn_resolved",
            category="combat",
        )
    def on_enter(self, context: Optional[Dict[str, Any]] = None) -> None:
        self.is_active = True
    def on_exit(self) -> Dict[str, Any]:
        self.is_active = False
        return {}
    def handle_event(self, event: Any) -> None:
        self.state_counter += 1
    def update(self, dt: float) -> None:
        self.state_counter += 1
        if self.state_counter % 10 == 0:
            self.probe.emit(pos=(100.0, 100.0), data={"turn": self.state_counter // 10})
    def render(self, surface: Any) -> None:
        pass

class CandidateCartridge(BaseCartridge):
    deliberation: str = "Candidate cartridge implementing dual-mode scene and spatial contracts."
    def __init__(self, manifest: Optional[Dict[str, Any]] = None, cartridge_dir: Optional[Any] = None, narrative: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest, cartridge_dir=cartridge_dir, narrative=narrative)
        self.scenes = {}
        self.level_spec = BaseLevelSpec(
            level_id="level_01",
            name="Level 1",
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            walls=[
                (0.0, 0.0, 1280.0, 32.0),
                (0.0, 688.0, 1280.0, 32.0),
                (0.0, 0.0, 32.0, 720.0),
                (1248.0, 0.0, 32.0, 720.0),
            ],
            enemies=[{"id": "enemy_1", "type": "chaser", "x": 600.0, "y": 360.0}],
            hazards=[{"id": "hazard_1", "type": "spike", "x": 400.0, "y": 360.0, "w": 32.0, "h": 32.0}],
        )

    def get_scenes(self, engine: Any) -> Dict[str, BaseScene]:
        if not self.scenes:
            self.scenes = {"main": CustomScene(engine)}
        return self.scenes

    def get_initial_scene(self, engine: Any) -> BaseScene:
        return self.get_scenes(engine)["main"]

    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return self.level_spec

    def create_player(self, x: float = 120.0, y: float = 360.0, telemetry_recorder: Optional[Any] = None) -> BasePlayer:
        p = MockPlayer(x, y)
        p.telemetry_recorder = telemetry_recorder
        return p

    def create_enemies(self, spec: Optional[BaseLevelSpec] = None) -> List[BaseEnemy]:
        return [MockEnemy("enemy_1", 600.0, 360.0)]

    def create_hazards(self, spec: Optional[BaseLevelSpec] = None) -> List[BaseHazard]:
        return [MockHazard("hazard_1", 400.0, 360.0)]
''', encoding="utf-8")

    # Add valid audio asset (16-bit PCM WAV)
    sample_rate = 22050
    duration = 0.5
    total_samples = int(sample_rate * duration)
    wav_path = dest / "bark.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        raw_data = bytearray()
        for i in range(total_samples):
            raw_data.extend(struct.pack("<h", 12000 if (i // 100) % 2 == 0 else -12000))
        wf.writeframes(raw_data)

    # Add valid 16-color palette sprite
    png_path = dest / "hero.png"
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(8, 24):
        for x in range(8, 24):
            img.putpixel((x, y), (41, 173, 255, 255))  # Cyan in Style Bible
    img.save(png_path, format="PNG")

    # Add valid narrative.json
    narrative_path = dest / "narrative.json"
    narrative_data = {
        "deliberation": "Valid test narrative bible.",
        "characters": [
            {
                "character_id": "hero",
                "name": "Operator",
                "portrait_file": "hero.png",
                "voice_bark_file": "bark.wav",
            }
        ],
        "dialogue_tree": {
            "root": {
                "speaker": "hero",
                "text": "Mission commencing.",
                "choices": [{"text": "Proceed", "next_node": "exit"}],
            },
            "exit": {
                "speaker": "hero",
                "text": "Sector cleared.",
                "choices": [],
            },
        },
    }
    narrative_path.write_text(json.dumps(narrative_data, indent=2), encoding="utf-8")

    # Add valid telemetry_manifest.json
    (dest / "telemetry_manifest.json").write_text(json.dumps({
        "deliberation": "Custom telemetry probe manifest.",
        "game_id": "cand_open_genre_01",
        "probes": [{"probe_name": "turn_resolved", "category": "combat", "description": "Emitted when tactical turn resolves."}],
    }, indent=2), encoding="utf-8")

    return dest


class TestSupervisorRationaleFirst:
    """Verifies that all supervisor models adhere strictly to Rationale-First architecture."""
    deliberation: str = "Verifying Rationale-First field ordering on BuildSupervisor models."

    def test_models_deliberation_field_index_zero(self):
        models_path = Path("engine/supervisor/models.py")
        code = models_path.read_text(encoding="utf-8")
        assert validate_rationale_first_ast(code) is True

    def test_serialization_preserves_deliberation_first(self):
        result = QualityGateResult(
            deliberation="Verified AST security checks without violations.",
            gate_name="Gate A: Syntax & AST",
            passed=True,
        )
        assert list(result.to_dict().keys())[0] == "deliberation"
        assert list(json.loads(result.model_dump_json()).keys())[0] == "deliberation"

        trace = RejectionDiagnosticTrace(
            deliberation="Wall blocked critical path at x=640.",
            candidate_id="spec_01",
            episode_number=1,
            failure_gate="Gate B: Traversability",
            failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
            failure_reason="A* found no open path.",
        )
        assert list(trace.to_dict().keys())[0] == "deliberation"
        assert list(json.loads(trace.model_dump_json()).keys())[0] == "deliberation"

        report = BuildEvaluationReport(
            deliberation="Promotion approved.",
            candidate_id="spec_01",
            episode_number=1,
            status=SupervisorGateVerdict.PROMOTED,
            promoted_to_active=True,
        )
        assert list(report.to_dict().keys())[0] == "deliberation"
        assert list(json.loads(report.model_dump_json()).keys())[0] == "deliberation"

        asset_report = AssetValidationReport(
            deliberation="Valid audio verified.",
            asset_path="bark.wav",
            asset_type="audio",
            passed=True,
        )
        assert list(asset_report.to_dict().keys())[0] == "deliberation"

    def test_empty_deliberation_raises_value_error(self):
        with pytest.raises(ValueError):
            QualityGateResult(
                deliberation="",
                gate_name="Gate A",
                passed=True,
            )
        with pytest.raises(ValueError):
            QualityGateResult(
                deliberation="   ",
                gate_name="Gate A",
                passed=True,
            )


class TestQualityGateA_AST:
    """Verifies Gate A AST Security and Sandboxing."""
    deliberation: str = "Verifying Gate A AST static security gate."

    def test_gate_a_valid_cartridge_passes(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_a_pass")
        res = supervisor_env.verify_gate_a_syntax_and_ast(cart_dir)
        assert res.passed is True
        assert res.failure_type == GateFailureType.NONE

    def test_gate_a_blocks_prohibited_os_import(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_a_os")
        # Injected forbidden import in mechanics.py
        mech_file = cart_dir / "mechanics.py"
        content = "import os\n" + mech_file.read_text(encoding="utf-8")
        mech_file.write_text(content, encoding="utf-8")

        res = supervisor_env.verify_gate_a_syntax_and_ast(cart_dir)
        assert res.passed is False
        assert res.failure_type == GateFailureType.SYNTAX_AST_ERROR

    def test_gate_a_blocks_reflection_exploit(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_a_exploit")
        mech_file = cart_dir / "mechanics.py"
        content = mech_file.read_text(encoding="utf-8") + "\nx = ().__class__.__subclasses__()\n"
        mech_file.write_text(content, encoding="utf-8")

        res = supervisor_env.verify_gate_a_syntax_and_ast(cart_dir)
        assert res.passed is False
        assert res.failure_type == GateFailureType.SYNTAX_AST_ERROR

    def test_gate_a_enforces_class_deliberation(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        from engine.studio.swe import LeadGameplaySWEAgent
        swe = LeadGameplaySWEAgent()
        content = "from engine.api.contracts import BaseCartridge\nclass RogueClass(BaseCartridge):\n    pass\n"
        valid, err = swe.verify_code_module("mechanics.py", content)
        assert valid is False
        assert "deliberation" in err


class TestQualityGateE_MultimodalAssets:
    """Verifies Gate E Multimodal Asset Integrity."""
    deliberation: str = "Verifying Gate E audio, visual, and narrative asset checks."

    def test_gate_e_valid_riff_wave_audio_passes(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_e_pass")
        res = supervisor_env.verify_gate_e_asset_integrity(cart_dir)
        assert res.passed is True

    def test_gate_e_corrupt_wave_header_rejected(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_e_corrupt_wav")
        (cart_dir / "bark.wav").write_bytes(b"CORRUPT_NOT_A_WAV_HEADER_DATA_123456789012345678901234567890")
        res = supervisor_env.verify_gate_e_asset_integrity(cart_dir)
        assert res.passed is False
        assert res.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION

    def test_gate_e_silent_audio_rejected(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_e_silent_wav")
        wav_path = cart_dir / "bark.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(bytearray(22050 * 2))  # All zeros
        res = supervisor_env.verify_gate_e_asset_integrity(cart_dir)
        assert res.passed is False
        assert res.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION

    def test_gate_e_valid_16_color_png_passes(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_e_png_pass")
        res = supervisor_env.verify_gate_e_asset_integrity(cart_dir)
        assert res.passed is True

    def test_gate_e_unquantized_png_rejected(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        supervisor_env.asset_validator.enforce_palette = True
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_e_unquantized_png")
        png_path = cart_dir / "hero.png"
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        # Use color outside 16-color Style Bible
        img.putpixel((10, 10), (123, 234, 56, 255))
        img.save(png_path, format="PNG")

        res = supervisor_env.verify_gate_e_asset_integrity(cart_dir)
        assert res.passed is False
        assert res.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION

    def test_gate_e_narrative_missing_portrait_rejected(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_e_missing_portrait")
        (cart_dir / "hero.png").unlink()

        res = supervisor_env.verify_gate_e_asset_integrity(cart_dir)
        assert res.passed is False
        assert res.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION


class TestQualityGateD_ZeroExceptions:
    """Verifies Gate D Zero Runtime Exceptions during cartridge load and boot ticks."""
    deliberation: str = "Verifying Gate D clean boot and runtime exception capture."

    def test_gate_d_clean_cartridge_load_and_boot(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_d_clean")
        cart = CartridgeLoader.load_cartridge(cart_dir)
        res = supervisor_env.verify_gate_d_exceptions(None)
        assert res.passed is True
        assert res.failure_type == GateFailureType.NONE

    def test_gate_d_unhandled_runtime_crash_rejected(self, supervisor_env: BuildSupervisor):
        exc = RuntimeError("Simulated unhandled hardware fault during boot tick")
        res = supervisor_env.verify_gate_d_exceptions(exc)
        assert res.passed is False
        assert res.failure_type == GateFailureType.UNHANDLED_EXCEPTION
        assert "Simulated unhandled hardware fault" in res.details


class TestQualityGateB_NavMeshTraversability:
    """Verifies Gate B NavMesh A* Critical-Path Traversability."""
    deliberation: str = "Verifying Gate B spatial pathfinding and obstacle detection."

    def test_gate_b_open_level_traversable(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_b_open")
        cart = CartridgeLoader.load_cartridge(cart_dir)
        spec = cart.get_level_spec(1)
        res, blocked = supervisor_env.verify_gate_b_traversability(cart, spec)
        assert res.passed is True
        assert res.failure_type == GateFailureType.NONE
        assert len(blocked) == 0

    def test_gate_b_sealed_exit_rejected(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_b_blocked")
        cart = CartridgeLoader.load_cartridge(cart_dir)
        spec = cart.get_level_spec(1)
        # Seal room by putting a massive vertical wall spanning entire height
        spec.walls.append((640.0, 0.0, 64.0, 720.0))
        res, blocked = supervisor_env.verify_gate_b_traversability(cart, spec)
        assert res.passed is False
        assert res.failure_type == GateFailureType.CRITICAL_PATH_BLOCKED
        assert len(blocked) > 0

    def test_gate_b_spawn_inside_obstacle_rejected(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_b_spawn_in_wall")
        cart = CartridgeLoader.load_cartridge(cart_dir)
        spec = cart.get_level_spec(1)
        # Move spawn point inside perimeter wall
        spec.spawn_point = (16.0, 16.0)
        res, blocked = supervisor_env.verify_gate_b_traversability(cart, spec)
        assert res.passed is False
        assert res.failure_type == GateFailureType.CRITICAL_PATH_BLOCKED


class TestQualityGateC_Survivability:
    """Verifies Gate C Playability and Survivability."""
    deliberation: str = "Verifying Gate C spawn clearance, early death, and playability bounds."

    def test_gate_c_safe_spawn_clearance_passes(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_c_safe")
        cart = CartridgeLoader.load_cartridge(cart_dir)
        spec = cart.get_level_spec(1)
        # Hazard is at x=400, spawn is at x=120: safe clearance > 64px
        res = supervisor_env.verify_gate_c_survivability(None, spec)
        assert res.passed is True

    def test_gate_c_spawn_hazard_overlap_rejected(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_c_overlap")
        cart = CartridgeLoader.load_cartridge(cart_dir)
        spec = cart.get_level_spec(1)
        # Place hazard directly at player spawn (120, 360)
        spec.hazards.append({
            "id": "lethal_spawn_spike",
            "x": 120.0,
            "y": 360.0,
            "w": 32.0,
            "h": 32.0,
            "damage": 50.0,
        })
        res = supervisor_env.verify_gate_c_survivability(None, spec)
        assert res.passed is False
        assert res.failure_type == GateFailureType.SURVIVABILITY_VIOLATION

    def test_gate_c_premature_death_rejected(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_gate_c_early_death")
        cart = CartridgeLoader.load_cartridge(cart_dir)
        spec = cart.get_level_spec(1)
        sim_result = {
            "first_death_tick": 45,
            "failure_coordinates": (120.0, 360.0),
        }
        res = supervisor_env.verify_gate_c_survivability(sim_result, spec)
        assert res.passed is False
        assert res.failure_type == GateFailureType.SURVIVABILITY_VIOLATION


class TestPromotionLogic:
    """Verifies that candidates passing 100% of gates are hermetically promoted to cartridges/active/."""
    deliberation: str = "Verifying atomic promotion upon 100% gate pass."

    def test_successful_promotion_populates_active_directory(self, supervisor_env: BuildSupervisor, tmp_path: Path):
        cart_dir = create_mock_candidate(tmp_path / "candidate_promoted")
        gate_results, trace = supervisor_env.evaluate_candidate_cartridge(cart_dir)
        assert trace is None
        for gate_name, result in gate_results.items():
            assert result.passed is True

        supervisor_env.promote_candidate(cart_dir)
        active_dir = supervisor_env.active_cartridge_dir
        assert active_dir.is_dir()
        assert (active_dir / "mechanics.py").is_file()
        assert (active_dir / "level.json").is_file()
        assert (active_dir / "manifest.json").is_file()
