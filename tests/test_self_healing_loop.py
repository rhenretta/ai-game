"""tests/test_self_healing_loop.py
Authoritative unit and integration test suite for Closed-Loop Self-Healing & Build Supervisor (Milestone M4, Features 24-27).
Verifies:
  1. Gate A Python syntax repair via LeadGameplaySWEAgent.
  2. Gate B traversability geometry repair via bottleneck wall removal.
  3. Gate C survivability repair via spawn hazard relocation and softening.
  4. Gate E asset repair via 16-color palette re-quantization.
  5. Max retries exhaustion halting with FAILED_MAX_RETRIES.
  6. Hermetic staging isolation and zero leakage to active cartridge directory.
  7. StudioOrchestrator self-healing state transitions (VERIFYING -> SYNTHESIZING -> VERIFYING -> PROMOTED).
  8. Rationale-First field ordering on all diagnostic traces and evaluation reports.
Strictly 0 occurrences of forbidden legacy branding tokens.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any, Dict, List, Optional
import wave
from PIL import Image
import pytest

from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.studio.models import StudioBuildState
from engine.studio.orchestrator import InvalidStateTransitionError, StudioOrchestrator
from engine.studio.swe import LeadGameplaySWEAgent
from engine.supervisor.models import (
    BuildEvaluationReport,
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    RemedyType,
    SupervisorGateVerdict,
)
from engine.supervisor.supervisor import BuildSupervisor


def create_minimal_valid_cartridge(dest_dir: Path) -> None:
    """Helper creating a minimal valid cartridge directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest_data = {
        "cartridge_id": "test_cartridge",
        "title": "Test Cartridge",
        "version": "1.0.0",
        "initial_level": 1,
        "author": "Autonomous Studio",
        "description": "Minimal valid cartridge for self-healing verification.",
    }
    (dest_dir / "manifest.json").write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    level_data = {
        "level_id": "chamber_01",
        "width": 1280.0,
        "height": 720.0,
        "spawn_point": [120.0, 360.0],
        "exit_point": [1160.0, 360.0],
        "walls": [
            [0.0, 0.0, 1280.0, 32.0],
            [0.0, 688.0, 1280.0, 32.0],
            [0.0, 0.0, 32.0, 720.0],
            [1248.0, 0.0, 32.0, 720.0],
        ],
        "hazards": [],
        "enemies": [],
    }
    (dest_dir / "level.json").write_text(json.dumps(level_data, indent=2), encoding="utf-8")

    abilities_src = (
        '"""abilities.py"""\n'
        'from engine.combat.damage import DamagePacket\n\n'
        'class DashAbility:\n'
        '    deliberation: str = "Dash mobility ability."\n'
        '    def __init__(self):\n'
        '        self.cooldown = 1.0\n'
    )
    (dest_dir / "abilities.py").write_text(abilities_src, encoding="utf-8")

    entities_src = (
        '"""entities.py"""\n'
        'from engine.entities.base import LivingEntity\n\n'
        'class CustomEnemy(LivingEntity):\n'
        '    deliberation: str = "Custom enemy unit."\n'
        '    def __init__(self, entity_id: str, x: float, y: float):\n'
        '        super().__init__(entity_id, x, y, 32.0, 32.0, 50.0)\n'
    )
    (dest_dir / "entities.py").write_text(entities_src, encoding="utf-8")

    mechanics_src = (
        '"""mechanics.py"""\n'
        'import json\n'
        'from pathlib import Path\n'
        'from engine.api.contracts import BaseCartridge, BaseLevelSpec\n\n'
        'class CartridgeMechanics(BaseCartridge):\n'
        '    deliberation: str = "Cartridge mechanics implementation."\n'
        '    def __init__(self, manifest=None, cartridge_dir=None, narrative=None):\n'
        '        super().__init__(manifest=manifest, cartridge_dir=cartridge_dir, narrative=narrative)\n'
        '        self.score = 0\n'
        '    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:\n'
        '        cdir = Path(self.cartridge_dir) if self.cartridge_dir else Path(".")\n'
        '        level_file = cdir / "level.json"\n'
        '        if level_file.exists():\n'
        '            return BaseLevelSpec.from_dict(json.loads(level_file.read_text(encoding="utf-8")))\n'
        '        return BaseLevelSpec(level_id="chamber_01", spawn_point=(120.0, 360.0), exit_point=(1160.0, 360.0))\n'
    )
    (dest_dir / "mechanics.py").write_text(mechanics_src, encoding="utf-8")

    # Generate a valid 1-second 22050Hz mono WAV
    sample_rate = 22050
    duration_sec = 0.5
    num_frames = int(sample_rate * duration_sec)
    audio_path = dest_dir / "test_sfx.wav"
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Sine wave with audible amplitude
        import math
        frames = bytearray()
        for i in range(num_frames):
            val = int(8000.0 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(bytes(frames))

    # Generate a valid indexed PNG with Style Bible 16-color palette
    from engine.assets.palette import STYLE_BIBLE_16_PALETTE
    img = Image.new("P", (64, 64), color=0)
    flat_palette = []
    for c in STYLE_BIBLE_16_PALETTE:
        flat_palette.extend([c[0], c[1], c[2]])
    while len(flat_palette) < 768:
        flat_palette.extend([0, 0, 0])
    img.putpalette(flat_palette)
    img.save(dest_dir / "test_sprite.png", format="PNG")


class TestGateACodeSelfHealing:
    """Verifies closed-loop repair of Python syntax and AST defects."""
    deliberation: str = "Verifying closed-loop repair of Gate A syntax and reflection defects."

    def test_repair_syntax_error_in_mechanics(self, tmp_path: Path):
        cand_dir = tmp_path / "candidate_gate_a"
        create_minimal_valid_cartridge(cand_dir)

        # Inject syntax error
        (cand_dir / "mechanics.py").write_text(
            '"""mechanics.py"""\n'
            'from engine.api.contracts import BaseCartridge\n\n'
            'class CartridgeMechanics(BaseCartridge):\n'
            '    def __init__(self)\n'  # Missing colon
            '        self.score = 0\n',
            encoding="utf-8"
        )

        trace = RejectionDiagnosticTrace(
            deliberation="Syntax error detected in mechanics.py.",
            candidate_id="candidate_gate_a",
            failure_gate="Gate A: Syntax & AST Security",
            failure_type=GateFailureType.SYNTAX_AST_ERROR,
            failure_reason="SyntaxError in mechanics.py: expected ':'",
            file_path="mechanics.py",
            line_number=4,
            recommended_remedy=RemedyType.FIX_SYNTAX,
        )

        class MockModels:
            def generate_content(self, model, contents):
                class MockResponse:
                    text = (
                        '"""mechanics.py"""\n'
                        'from engine.api.contracts import BaseCartridge, BaseLevelSpec\n\n'
                        'class CartridgeMechanics(BaseCartridge):\n'
                        '    deliberation: str = "Repaired mechanics."\n'
                        '    def __init__(self, manifest=None, cartridge_dir=None, narrative=None):\n'
                        '        super().__init__(manifest=manifest, cartridge_dir=cartridge_dir, narrative=narrative)\n'
                        '        self.score = 0\n'
                        '    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:\n'
                        '        return BaseLevelSpec(level_id="chamber_01", spawn_point=(120.0, 360.0), exit_point=(1160.0, 360.0))\n'
                    )
                return MockResponse()

        class MockGeminiClient:
            models = MockModels()

        swe = LeadGameplaySWEAgent(client=MockGeminiClient())
        results = swe.repair_cartridge_code(cand_dir, trace)

        assert "mechanics.py" in results
        repaired_text = (cand_dir / "mechanics.py").read_text(encoding="utf-8")
        import ast
        tree = ast.parse(repaired_text)
        assert tree is not None
        assert "def __init__" in repaired_text


class TestGateBTraversabilitySelfHealing:
    """Verifies closed-loop repair of blocked critical corridors in level.json."""
    deliberation: str = "Verifying closed-loop removal of blocking partition walls in level.json."

    def test_repair_blocking_wall_via_bottleneck_nodes(self, tmp_path: Path):
        cand_dir = tmp_path / "candidate_gate_b"
        create_minimal_valid_cartridge(cand_dir)

        # Add blocking divider wall at x=640 (col 20)
        level_data = json.loads((cand_dir / "level.json").read_text(encoding="utf-8"))
        level_data["walls"].append([640.0, 0.0, 32.0, 720.0])
        (cand_dir / "level.json").write_text(json.dumps(level_data, indent=2), encoding="utf-8")

        trace = RejectionDiagnosticTrace(
            deliberation="Interior wall blocking path from spawn to exit.",
            candidate_id="candidate_gate_b",
            failure_gate="Gate B: Traversability",
            failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
            failure_reason="Critical path obstructed by solid partition at column 20.",
            file_path="level.json",
            failure_coordinates=(640.0, 360.0),
            blocked_nodes=[(20, 11)],
            recommended_remedy=RemedyType.REMOVE_OBSTACLE,
        )

        swe = LeadGameplaySWEAgent()
        results = swe.repair_cartridge_code(cand_dir, trace)

        assert "level.json" in results
        repaired_level = json.loads((cand_dir / "level.json").read_text(encoding="utf-8"))
        # Verify partition wall was removed
        has_partition = any(w[0] == 640.0 and w[2] == 32.0 and w[3] == 720.0 for w in repaired_level["walls"])
        assert has_partition is False


class TestGateCSurvivabilitySelfHealing:
    """Verifies closed-loop relocation and damage softening of spawn hazards."""
    deliberation: str = "Verifying closed-loop relocation and softening of lethal spawn hazards."

    def test_repair_hazard_overlapping_spawn(self, tmp_path: Path):
        cand_dir = tmp_path / "candidate_gate_c"
        create_minimal_valid_cartridge(cand_dir)

        # Add lethal acid directly on player spawn (120, 360)
        level_data = json.loads((cand_dir / "level.json").read_text(encoding="utf-8"))
        level_data["hazards"].append({
            "id": "spawn_trap",
            "type": "floor_acid",
            "x": 120.0,
            "y": 360.0,
            "w": 64.0,
            "h": 64.0,
            "damage": 50.0,
        })
        (cand_dir / "level.json").write_text(json.dumps(level_data, indent=2), encoding="utf-8")

        trace = RejectionDiagnosticTrace(
            deliberation="Lethal hazard overlaps spawn clearance margin.",
            candidate_id="candidate_gate_c",
            failure_gate="Gate C: Survivability",
            failure_type=GateFailureType.SURVIVABILITY_VIOLATION,
            failure_reason="Hazard 'spawn_trap' overlaps spawn margin.",
            file_path="level.json",
            failure_coordinates=(120.0, 360.0),
            recommended_remedy=RemedyType.WIDEN_MARGIN,
        )

        swe = LeadGameplaySWEAgent()
        results = swe.repair_cartridge_code(cand_dir, trace)

        assert "level.json" in results
        repaired_level = json.loads((cand_dir / "level.json").read_text(encoding="utf-8"))
        repaired_hazard = repaired_level["hazards"][0]
        # Hazard relocated away from spawn (dist > 120) and damage softened
        assert repaired_hazard["x"] >= 360.0
        assert repaired_hazard["damage"] < 50.0


class TestGateEAssetIntegritySelfHealing:
    """Verifies closed-loop palette quantization of RGB image assets."""
    deliberation: str = "Verifying closed-loop palette quantization for Gate E compliance."

    def test_repair_unquantized_png_asset(self, tmp_path: Path):
        cand_dir = tmp_path / "candidate_gate_e"
        create_minimal_valid_cartridge(cand_dir)

        # Overwrite test_sprite.png with an unquantized 24-bit RGB gradient
        rgb_img = Image.new("RGB", (64, 64))
        for x in range(64):
            for y in range(64):
                rgb_img.putpixel((x, y), (x * 4, y * 4, (x + y) * 2))
        rgb_img.save(cand_dir / "test_sprite.png", format="PNG")

        trace = RejectionDiagnosticTrace(
            deliberation="Unquantized 24-bit RGB image violates 16-color Style Bible palette.",
            candidate_id="candidate_gate_e",
            failure_gate="Gate E: Asset Integrity",
            failure_type=GateFailureType.ASSET_INTEGRITY_VIOLATION,
            failure_reason="Image mode RGB with out-of-palette colors.",
            file_path="test_sprite.png",
            recommended_remedy=RemedyType.REGENERATE_ASSET,
        )

        swe = LeadGameplaySWEAgent()
        results = swe.repair_cartridge_code(cand_dir, trace)

        assert "test_sprite.png" in results
        repaired_img = Image.open(cand_dir / "test_sprite.png")
        assert repaired_img.mode == "P"


class TestFullSelfHealingLoopPromotion:
    """Verifies BuildSupervisor driving evaluate_and_heal to successful promotion."""
    deliberation: str = "Verifying end-to-end evaluate_and_heal loop promoting on healed attempt."

    def test_supervisor_heals_blocked_corridor_and_promotes(self, tmp_path: Path):
        active_dir = tmp_path / "active"
        active_dir.mkdir(parents=True, exist_ok=True)
        cand_dir = tmp_path / "candidate_heal"
        create_minimal_valid_cartridge(cand_dir)

        # Inject blocking wall
        level_data = json.loads((cand_dir / "level.json").read_text(encoding="utf-8"))
        level_data["walls"].append([640.0, 0.0, 32.0, 720.0])
        (cand_dir / "level.json").write_text(json.dumps(level_data, indent=2), encoding="utf-8")

        swe = LeadGameplaySWEAgent()
        supervisor = BuildSupervisor(
            cartridges_root=tmp_path,
            active_cartridge_dir=active_dir,
            max_retries=3,
            swe_agent=swe,
        )

        report = supervisor.evaluate_and_heal(cand_dir, max_retries=3)

        assert report.status == SupervisorGateVerdict.PROMOTED
        assert report.promoted_to_active is True
        assert report.attempt_count >= 2
        assert len(report.repair_history) >= 1
        assert (active_dir / "manifest.json").exists()


class TestMaxRetriesExhaustionAndHermeticStaging:
    """Verifies that unrepairable candidates halt after max_retries without polluting active directory."""
    deliberation: str = "Verifying max retries exhaustion and hermetic active directory preservation."

    def test_unrepairable_defect_exhausts_retries(self, tmp_path: Path):
        active_dir = tmp_path / "active"
        active_dir.mkdir(parents=True, exist_ok=True)
        # Seed active with sentinel file
        (active_dir / "sentinel.txt").write_text("PRISTINE_ACTIVE", encoding="utf-8")

        cand_dir = tmp_path / "candidate_doomed"
        cand_dir.mkdir(parents=True, exist_ok=True)
        # Empty candidate without manifest.json
        (cand_dir / "broken.txt").write_text("garbage", encoding="utf-8")

        supervisor = BuildSupervisor(
            cartridges_root=tmp_path,
            active_cartridge_dir=active_dir,
            max_retries=2,
        )

        report = supervisor.evaluate_and_heal(cand_dir, max_retries=2)

        assert report.status == SupervisorGateVerdict.FAILED_MAX_RETRIES
        assert report.promoted_to_active is False
        assert report.attempt_count == 2
        # Active directory remains strictly pristine
        assert (active_dir / "sentinel.txt").read_text(encoding="utf-8") == "PRISTINE_ACTIVE"
        assert not (active_dir / "broken.txt").exists()


class TestStudioOrchestratorStateTransitions:
    """Verifies StudioOrchestrator self-healing state transitions."""
    deliberation: str = "Verifying StudioOrchestrator transitions through VERIFYING and SYNTHESIZING."

    def test_state_machine_self_healing_cycle(self):
        orch = StudioOrchestrator()
        assert orch.state == StudioBuildState.IDLE

        # IDLE -> ANALYZING -> SYNTHESIZING -> VERIFYING
        orch.transition_to(StudioBuildState.ANALYZING, "Starting analysis")
        orch.transition_to(StudioBuildState.SYNTHESIZING, "Generating cartridge")
        orch.transition_to(StudioBuildState.VERIFYING, "Running 5-gate supervisor")

        # Defect found: VERIFYING -> SYNTHESIZING (repair)
        orch.transition_to(StudioBuildState.SYNTHESIZING, "Self-healing repair triggered")

        # Repaired: SYNTHESIZING -> VERIFYING
        orch.transition_to(StudioBuildState.VERIFYING, "Re-running 5-gate supervisor")

        # Passed: VERIFYING -> PROMOTED
        orch.transition_to(StudioBuildState.PROMOTED, "All quality gates passed")
        assert orch.state == StudioBuildState.PROMOTED

    def test_invalid_state_transition_rejected(self):
        orch = StudioOrchestrator()
        with pytest.raises(InvalidStateTransitionError):
            orch.transition_to(StudioBuildState.PROMOTED, "Illegal direct jump to PROMOTED")


class TestEpistemicRationaleFirstContracts:
    """Verifies that all models emitted during the self-healing loop declare deliberation strictly first."""
    deliberation: str = "Verifying Rationale-First field ordering on BuildEvaluationReport."

    def test_build_evaluation_report_deliberation_first(self):
        report = BuildEvaluationReport(
            deliberation="Candidate passed all verification gates.",
            candidate_id="cand_01",
            status=SupervisorGateVerdict.PROMOTED,
            promoted_to_active=True,
            attempt_count=1,
            max_retries=3,
        )
        assert list(report.to_dict().keys())[0] == "deliberation"

    def test_rejection_trace_deliberation_first(self):
        trace = RejectionDiagnosticTrace(
            deliberation="Trace diagnostic rationale.",
            candidate_id="cand_01",
            failure_gate="Gate B",
            failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
            failure_reason="Obstacle detected.",
        )
        assert list(trace.to_dict().keys())[0] == "deliberation"
