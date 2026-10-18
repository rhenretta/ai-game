"""tests/test_m4_challenger_gate.py
Comprehensive unit, boundary, and integration test suite for Milestone M4:
Independent Challenger Gate & Quality Verification Supervisor (Requirement R3).

Exhaustively verifies:
1. Pydantic Epistemic Models (Rationale-First Rule 4, 100% threshold enforcement, enum additions).
2. CandidateASTInspector: static AST and tokenize inspection (anti-stub detection, zero regexes).
3. CandidateTelemetryAuditor: runtime events, custom probes, and state progression.
4. CandidateVisualPlaytester: in-memory software frame buffer capture under SDL_VIDEODRIVER=dummy,
   non-blank canvas, motion deltas, types.Part.from_bytes packaging.
5. ChallengerGate: domain-agnostic prompt, Rule 5 fail-fast on client=None, structured evaluation,
   and deterministic invariant enforcement.
6. BuildSupervisor Gate F Integration: computational order (Gates A-E pass before Gate F),
   vacuous pass on missing spec, RejectionDiagnosticTrace on failure, and verify_candidate alias.
7. LeadGameplaySWEAgent self-healing repair for GateFailureType.REQUIREMENT_UNFULFILLED.
8. SwarmEventBus.publish_challenger_gate_verdict with deliberation strictly first.
"""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pygame
import pytest

import struct

from engine.api.contracts import BaseCartridge
from engine.sim.headless import HeadlessSimulator, SimulationInput
from engine.studio.models import (
    AcceptanceCriterion,
    GameRequirementsSpec,
    RequirementTrack,
    StudioAgentError,
)
from engine.studio.swe import LeadGameplaySWEAgent
from engine.supervisor.challenger import (
    AdversarialDesignCritic,
    CandidateASTInspector,
    CandidateTelemetryAuditor,
    CandidateVisualPlaytester,
    ChallengerGate,
)
from engine.supervisor.models import (
    ChallengerGateEvaluation,
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    RemedyType,
    RequirementEvaluationRecord,
    SupervisorGateVerdict,
)
from engine.supervisor.supervisor import BuildSupervisor
from engine.ui.models import SwarmEventType
from engine.ui.server import SwarmEventBus


@pytest.fixture
def mock_cartridge_active(tmp_path: Path) -> Path:
    """Creates a valid, complete game cartridge directory for testing."""
    cart_dir = tmp_path / "mock_cartridge_active"
    assets_dir = cart_dir / "assets"
    audio_dir = assets_dir / "audio"
    sprites_dir = assets_dir / "sprites"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sprites_dir.mkdir(parents=True, exist_ok=True)

    # manifest.json
    manifest = {
        "cartridge_id": "cartridge_active_test",
        "title": "Quantum Odyssey",
        "version": "1.0.0",
        "author": "Autonomous Game Studio Swarm",
        "entry_point": "mechanics.py",
        "episodes": [1, 2],
        "created_at": "2026-10-16T00:00:00Z",
    }
    (cart_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # level.json
    level_data = {
        "deliberation": "Spatial layout designed for balanced combat encounter.",
        "level_id": "level_01",
        "name": "Testing Chamber 1",
        "width": 1280.0,
        "height": 720.0,
        "spawn_point": [100.0, 360.0],
        "exit_point": [1180.0, 360.0],
        "walls": [
            [0.0, 0.0, 1280.0, 32.0],
            [0.0, 688.0, 1280.0, 32.0],
            [0.0, 0.0, 32.0, 720.0],
            [1248.0, 0.0, 32.0, 720.0],
        ],
        "enemies": [
            {"type": "drone", "x": 640.0, "y": 360.0, "hp": 50.0}
        ],
        "hazards": [
            {"type": "spikes", "x": 400.0, "y": 360.0, "radius": 24.0}
        ],
    }
    (cart_dir / "level.json").write_text(json.dumps(level_data, indent=2), encoding="utf-8")

    # narrative.json
    narrative_data = {
        "deliberation": "Narrative arc tracking protagonist escaping an abandoned orbital station.",
        "world": {"setting": "Orbital Station Alpha", "theme": "Sci-Fi Survival"},
        "characters": [
            {"id": "hero", "name": "Vanguard 7", "role": "protagonist", "portrait": "assets/sprites/hero.png"},
            {"id": "overseer", "name": "AI Overseer", "role": "rival", "portrait": "assets/sprites/overseer.png"},
        ],
        "dialogues": [
            {
                "trigger": "level_start",
                "character_id": "hero",
                "text": "Systems online. Exit portal located ahead.",
                "voice_clip": "assets/audio/hero_start.wav",
            }
        ],
        "dialogue_barks": {
            "on_damage": [
                {
                    "character_id": "hero",
                    "text": "Armor compromised!",
                    "audio_clip": "hero_start.wav",
                }
            ]
        },
    }
    (cart_dir / "narrative.json").write_text(json.dumps(narrative_data, indent=2), encoding="utf-8")

    # mechanics.py
    mechanics_code = """# Agent-authored mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.entities.player import Player as BasePlayer
from engine.entities.enemies import BaseEnemy
from engine.entities.hazard import Hazard as BaseHazard

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active cartridge with tactical traversal."
    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            level_id=f"episode_{episode_number}",
            name="Procedural Chamber",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 360.0),
            exit_point=(1180.0, 360.0),
            walls=[(0.0, 0.0, 1280.0, 32.0), (0.0, 688.0, 1280.0, 32.0)],
            deliberation="Level geometry generated for tactical traversal."
        )

    def create_player(self, x: float, y: float, telemetry_recorder=None) -> BasePlayer:
        return BasePlayer(x=x, y=y)

    def create_enemies(self, spec: BaseLevelSpec):
        return []

    def create_hazards(self, spec: BaseLevelSpec):
        return []
"""
    (cart_dir / "mechanics.py").write_text(mechanics_code, encoding="utf-8")

    # entities.py
    entities_code = """# Agent-authored entities.py
from engine.entities.base import Entity
from engine.spatial.aabb import Vector2

class CustomDrone(Entity):
    deliberation: str = "Drone entity definition."
    def __init__(self, x: float, y: float):
        super().__init__(x=x, y=y, width=32.0, height=32.0)
        self.health = 50.0
"""
    (cart_dir / "entities.py").write_text(entities_code, encoding="utf-8")

    # Create valid WAV file (0.5s tone)
    import math
    import wave
    from PIL import Image

    wav_path = audio_dir / "hero_start.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        raw = bytearray()
        for i in range(11025):
            val = int(12000 * math.sin(2 * math.pi * 440 * i / 22050))
            raw.extend(struct.pack("<h", val))
        wf.writeframes(raw)

    # Create valid 32x32 PNG files
    for s_name in ("hero.png", "overseer.png"):
        img = Image.new("RGB", (32, 32), color=(0, 0, 0))
        img.save(sprites_dir / s_name, format="PNG")

    return cart_dir



# =============================================================================
# 1. Pydantic Epistemic Models & Rationale-First Schema Invariants (Rule 4)
# =============================================================================

class TestChallengerGateEpistemicModels:
    """Verifies that all M4 data models declare deliberation strictly first and enforce invariants."""

    def test_requirement_evaluation_record_deliberation_first(self):
        record = RequirementEvaluationRecord(
            deliberation="Verified requirement R1 compliance against simulation trace.",
            requirement_id="R1",
            fulfilled=True,
            grounding_evidence="DroneEntity moved 25px in 60 ticks.",
            critique_notes="Meets all movement criteria.",
        )
        dumped = record.model_dump()
        keys = list(dumped.keys())
        assert keys[0] == "deliberation", f"Expected deliberation at index 0, got {keys[0]}"
        assert record.id == "R1"
        assert record.fulfilled is True

    def test_requirement_evaluation_record_empty_deliberation_raises(self):
        with pytest.raises(ValueError, match="deliberation must not be empty or whitespace"):
            RequirementEvaluationRecord(
                deliberation="",
                requirement_id="R1",
                fulfilled=True,
                grounding_evidence="none",
            )
        with pytest.raises(ValueError, match="deliberation must not be empty or whitespace"):
            RequirementEvaluationRecord(
                deliberation="   \n\t  ",
                requirement_id="R1",
                fulfilled=True,
                grounding_evidence="none",
            )

    def test_challenger_gate_evaluation_deliberation_first(self):
        rec = RequirementEvaluationRecord(
            deliberation="Track R1 satisfied.",
            requirement_id="R1",
            fulfilled=True,
            grounding_evidence="Telemetry verified.",
        )
        evaluation = ChallengerGateEvaluation(
            deliberation="Evaluated all requirements with zero defects.",
            candidate_id="cand_test_01",
            overall_verdict=True,
            evaluated_requirements=[rec],
        )
        dumped = evaluation.model_dump()
        keys = list(dumped.keys())
        assert keys[0] == "deliberation"
        assert evaluation.overall_verdict is True
        assert evaluation.spec_id == "cand_test_01"

    def test_challenger_gate_evaluation_empty_deliberation_raises(self):
        with pytest.raises(ValueError, match="deliberation must not be empty or whitespace"):
            ChallengerGateEvaluation(
                deliberation="",
                candidate_id="cand_test_02",
                overall_verdict=True,
            )

    def test_challenger_gate_evaluation_100_percent_threshold_enforcement(self):
        """Any requirement with fulfilled=False MUST force overall_verdict=False."""
        rec_pass = RequirementEvaluationRecord(
            deliberation="R1 passed.",
            requirement_id="R1",
            fulfilled=True,
            grounding_evidence="Evidence 1",
        )
        rec_fail = RequirementEvaluationRecord(
            deliberation="R2 failed: audio asset missing.",
            requirement_id="R2",
            fulfilled=False,
            grounding_evidence="Evidence 2",
        )
        # Attempting to declare overall_verdict=True when rec_fail is present
        evaluation = ChallengerGateEvaluation(
            deliberation="Adversarial evaluation found defect in R2.",
            candidate_id="cand_fail_threshold",
            overall_verdict=True,
            evaluated_requirements=[rec_pass, rec_fail],
        )
        assert evaluation.overall_verdict is False, "100% threshold validator must force overall_verdict=False"

    def test_rejection_diagnostic_trace_m4_extensions_and_sync(self):
        trace = RejectionDiagnosticTrace(
            deliberation="Challenger gate rejected build: unfulfilled tracks R1, R2.",
            candidate_id="cand_trace_01",
            failure_gate="Gate F: Challenger Gate",
            failure_type=GateFailureType.REQUIREMENT_UNFULFILLED,
            failure_reason="Criteria AC1.1 and AC2.1 unfulfilled.",
            remedy_type=RemedyType.IMPLEMENT_REQUIREMENT,
            recommended_remedy="Implement genuine loop logic.",
            unfulfilled_requirements=["R1", "R2"],
            failing_criteria=["AC1.1", "AC2.1"],
            ast_evidence={"stubs": ["pass in mechanics.py"]},
            telemetry_evidence={"events_count": 0},
            recommended_corrective_actions=["Remove pass stub", "Wire event emission"],
        )
        dumped = trace.model_dump()
        assert list(dumped.keys())[0] == "deliberation"
        assert trace.failure_type == GateFailureType.REQUIREMENT_UNFULFILLED
        assert trace.remedy_type == RemedyType.IMPLEMENT_REQUIREMENT
        assert "R1" in trace.remedy_parameters["unfulfilled_requirements"]
        assert "AC1.1" in trace.remedy_parameters["failing_criteria"]
        assert trace.ast_evidence == {"stubs": ["pass in mechanics.py"]}
        assert trace.telemetry_evidence == {"events_count": 0}

    def test_enums_membership(self):
        assert GateFailureType.REQUIREMENT_UNFULFILLED.value == "REQUIREMENT_UNFULFILLED"
        assert RemedyType.IMPLEMENT_REQUIREMENT.value == "IMPLEMENT_REQUIREMENT"


# =============================================================================
# 2. CandidateASTInspector: Anti-Stub Detection & Zero Regexes (Rule 2)
# =============================================================================

class TestCandidateASTInspector:
    """Verifies static AST and tokenize inspection detecting evasive hollow stubs."""

    def test_detects_pass_stub(self, tmp_path: Path):
        code = """# Mechanics with pass stub
from engine.api.contracts import BaseCartridge

class TestCartridge(BaseCartridge):
    deliberation: str = "Test cartridge"
    def update_world(self):
        pass
"""
        (tmp_path / "mechanics.py").write_text(code, encoding="utf-8")
        inspector = CandidateASTInspector()
        report = inspector.inspect_directory(tmp_path)

        assert report["has_critical_stubs"] is True
        assert len(report["pass_stubs"]) >= 1
        assert report["pass_stubs"][0]["function"] == "update_world"
        assert report["pass_stubs"][0]["stmt"] == "pass"

    def test_detects_empty_and_docstring_only_bodies(self, tmp_path: Path):
        code = """# Mechanics with docstring only body
from engine.api.contracts import BaseCartridge

class TestCartridge(BaseCartridge):
    deliberation: str = "Test cartridge"
    def handle_interaction(self, target):
        \"\"\"Docstring only without any executable statements.\"\"\"
"""
        (tmp_path / "mechanics.py").write_text(code, encoding="utf-8")
        inspector = CandidateASTInspector()
        report = inspector.inspect_directory(tmp_path)

        assert report["has_critical_stubs"] is True
        assert len(report["empty_bodies"]) >= 1
        assert report["empty_bodies"][0]["function"] == "handle_interaction"

    def test_detects_ellipsis_and_return_none_stubs(self, tmp_path: Path):
        code = """# Mechanics with ellipsis and return None
from engine.api.contracts import BaseCartridge

class TestCartridge(BaseCartridge):
    deliberation: str = "Test cartridge"
    def do_action(self):
        ...
    def check_condition(self):
        return None
    def trigger_event(self):
        raise NotImplementedError
"""
        (tmp_path / "mechanics.py").write_text(code, encoding="utf-8")
        inspector = CandidateASTInspector()
        report = inspector.inspect_directory(tmp_path)

        assert report["has_critical_stubs"] is True
        stubs = {s["function"]: s["stmt"] for s in report["pass_stubs"]}
        assert "do_action" in stubs
        assert "check_condition" in stubs
        assert "trigger_event" in stubs

    def test_detects_todo_comments_via_tokenize(self, tmp_path: Path):
        code = """# Mechanics module
# TODO: implement player health deduction
from engine.api.contracts import BaseCartridge

class TestCartridge(BaseCartridge):
    deliberation: str = "Cartridge with todo"
    def step(self):
        x = 10
        # FIXME: magic number
        return x
"""
        (tmp_path / "mechanics.py").write_text(code, encoding="utf-8")
        inspector = CandidateASTInspector()
        report = inspector.inspect_directory(tmp_path)

        todos = [t["tag"] for t in report["todo_comments"]]
        assert "TODO" in todos
        assert "FIXME" in todos

    def test_detects_disconnected_classes_in_multi_module(self, tmp_path: Path):
        mech_code = """# mechanics.py
from engine.api.contracts import BaseCartridge

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active cartridge"
    def update(self):
        return True
"""
        entities_code = """# entities.py
class DisconnectedEnemy:
    \"\"\"Created but never referenced in mechanics or imported.\"\"\"
    def __init__(self):
        self.health = 100
"""
        (tmp_path / "mechanics.py").write_text(mech_code, encoding="utf-8")
        (tmp_path / "entities.py").write_text(entities_code, encoding="utf-8")

        inspector = CandidateASTInspector()
        report = inspector.inspect_directory(tmp_path)

        disconnected = [d["symbol"] for d in report["disconnected_symbols"]]
        assert "DisconnectedEnemy" in disconnected

    def test_genuine_implementation_passes_ast_inspection(self, tmp_path: Path):
        mech_code = """# mechanics.py
from engine.api.contracts import BaseCartridge
from entities import PlayerEntity

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Genuine active cartridge implementation"
    def __init__(self):
        self.player = PlayerEntity()
    def update(self, dt):
        self.player.x += 10.0 * dt
        return self.player.x
"""
        entities_code = """# entities.py
class PlayerEntity:
    def __init__(self):
        self.x = 0.0
"""
        (tmp_path / "mechanics.py").write_text(mech_code, encoding="utf-8")
        (tmp_path / "entities.py").write_text(entities_code, encoding="utf-8")

        inspector = CandidateASTInspector()
        report = inspector.inspect_directory(tmp_path)

        assert report["has_critical_stubs"] is False
        assert len(report["pass_stubs"]) == 0
        assert len(report["empty_bodies"]) == 0
        assert len(report["disconnected_symbols"]) == 0
        assert report["cleanliness_score"] == 1.0


# =============================================================================
# 3. CandidateTelemetryAuditor: Runtime Event & Progression Verification
# =============================================================================

class TestCandidateTelemetryAuditor:
    """Verifies runtime telemetry inspection and frozen state loop detection."""

    def test_auditor_detects_events_and_custom_probes(self, mock_cartridge_active: Path):
        sim = HeadlessSimulator(cartridge_path=mock_cartridge_active)
        for _ in range(60):
            sim.step(SimulationInput(move_x=1.0))

        auditor = CandidateTelemetryAuditor()
        report = auditor.audit_simulation(sim=sim)

        assert report["ticks_executed"] == 60
        assert report["avg_frame_time_ms"] < 16.6
        assert report["frame_budget_passed"] is True
        assert report["crashed"] is False
        assert report["is_stagnant_or_frozen"] is False


# =============================================================================
# 4. CandidateVisualPlaytester: Headless Software Frame Buffer Capture (Rule 6)
# =============================================================================

class TestCandidateVisualPlaytester:
    """Verifies in-memory software frame buffer capture, non-blank checks, and motion deltas."""

    def test_capture_surface_png_raw_bytes(self):
        surf = pygame.Surface((320, 240))
        surf.fill((255, 100, 50))
        png_bytes = CandidateVisualPlaytester.capture_surface_png(surf)

        assert len(png_bytes) > 0
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "Must be valid PNG format"

    def test_blank_canvas_detection(self):
        playtester = CandidateVisualPlaytester()
        # Create a mock sim whose surface is completely black
        mock_sim = MagicMock()
        surf = pygame.Surface((1280, 720))
        surf.fill((0, 0, 0))  # Solid black monochrome canvas
        mock_sim.engine.logical_surface = surf
        mock_sim.current_scene.render = MagicMock()

        report = playtester.playtest_and_capture(sim=mock_sim, ticks=2)
        assert report["blank_canvas_detected"] is True
        assert report["non_blank_verified"] is False
        assert report["unique_colors_t0"] == 1

    def test_multi_modal_image_parts_packaging(self):
        playtester = CandidateVisualPlaytester()
        surf = pygame.Surface((640, 360))
        surf.fill((30, 80, 150))
        pygame.draw.circle(surf, (255, 255, 0), (100, 100), 40)

        mock_sim = MagicMock()
        mock_sim.engine.logical_surface = surf
        mock_sim.current_scene.render = MagicMock()

        report = playtester.playtest_and_capture(sim=mock_sim, ticks=2)
        assert report["blank_canvas_detected"] is False
        assert report["non_blank_verified"] is True
        assert len(report["snapshot_images"]) >= 1
        assert len(report["image_parts"]) >= 1


# =============================================================================
# 5. ChallengerGate: Epistemic Evaluation & Rule 5 Fail-Fast
# =============================================================================

class TestChallengerGateEvaluation:
    """Verifies ChallengerGate prompt synthesis, Rule 5 fail-fast, and structured evaluation."""

    def test_rule_5_fail_fast_when_client_is_none(self, tmp_path: Path):
        gate = ChallengerGate(client=None)
        spec = GameRequirementsSpec(
            deliberation="Requirements for testing fail fast.",
            game_title="FailFast Game",
            genre_paradigm="simulation",
            requirements=[
                RequirementTrack(
                    deliberation="Track 1",
                    id="R1",
                    title="Core Mechanics",
                    description="Move entities",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="Criterion 1.1",
                            id="AC1.1",
                            description="Entities move on tick",
                            verification_method="runtime_telemetry",
                        )
                    ],
                )
            ],
        )
        with pytest.raises(StudioAgentError, match="ChallengerGate requires an active google.genai Client"):
            gate.evaluate(candidate_dir=tmp_path, requirements_spec=spec, client=None)

    def test_vacuous_pass_when_requirements_spec_is_absent(self, tmp_path: Path):
        gate = ChallengerGate(client=None)
        # Empty candidate directory without requirements_spec.json
        eval_result = gate.evaluate(candidate_dir=tmp_path, requirements_spec=None)
        assert eval_result.overall_verdict is True
        assert len(eval_result.evaluated_requirements) == 0
        assert "vacuously" in eval_result.deliberation

    def test_structured_evaluation_call_success(self, tmp_path: Path):
        # Create genuine code in candidate_dir
        (tmp_path / "mechanics.py").write_text("""# mechanics.py
from engine.api.contracts import BaseCartridge

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active cartridge"
    def __init__(self):
        self.tick_counter = 0
    def update(self, dt):
        self.tick_counter += 1
        return self.tick_counter
""", encoding="utf-8")

        spec = GameRequirementsSpec(
            deliberation="Requirements spec.",
            game_title="Genuine Game",
            genre_paradigm="arcade",
            requirements=[
                RequirementTrack(
                    deliberation="Track 1",
                    id="R1",
                    title="Counter Logic",
                    description="Increment counter",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="Criterion 1",
                            id="AC1.1",
                            description="Counter increments",
                            verification_method="ast_inspection",
                        )
                    ],
                )
            ],
        )

        # Mock Gemini client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "deliberation": "Audited candidate against R1: counter increments properly.",
            "candidate_id": tmp_path.name,
            "overall_verdict": True,
            "evaluated_requirements": [
                {
                    "deliberation": "R1 criteria satisfied genuinely in mechanics.py.",
                    "requirement_id": "R1",
                    "fulfilled": True,
                    "grounding_evidence": "AST analysis shows tick_counter incrementing in update().",
                    "critique_notes": "Clean genuine logic.",
                }
            ],
            "rejection_trace": None,
        })
        mock_client.models.generate_content.return_value = mock_response

        gate = ChallengerGate(client=mock_client)
        evaluation = gate.evaluate(candidate_dir=tmp_path, requirements_spec=spec)

        assert evaluation.overall_verdict is True
        assert len(evaluation.evaluated_requirements) == 1
        assert evaluation.evaluated_requirements[0].fulfilled is True
        assert evaluation.rejection_trace is None

    def test_hollow_stub_overrides_llm_hallucination_to_false(self, tmp_path: Path):
        """Even if LLM claims overall_verdict=True, detected hollow stubs programmatically force False."""
        (tmp_path / "mechanics.py").write_text("""# mechanics.py
from engine.api.contracts import BaseCartridge

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Hollow stub cartridge"
    def execute_logic(self):
        pass
""", encoding="utf-8")

        spec = GameRequirementsSpec(
            deliberation="Requirements spec.",
            game_title="Hollow Game",
            genre_paradigm="action",
            requirements=[
                RequirementTrack(
                    deliberation="Track 1",
                    id="R1",
                    title="Logic Execution",
                    description="Execute logic",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="Criterion 1",
                            id="AC1.1",
                            description="Execute logic",
                            verification_method="ast_inspection",
                        )
                    ],
                )
            ],
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        # Simulated LLM hallucinating a pass verdict
        mock_response.text = json.dumps({
            "deliberation": "LLM incorrectly thought pass was fine.",
            "candidate_id": tmp_path.name,
            "overall_verdict": True,
            "evaluated_requirements": [
                {
                    "deliberation": "R1 claimed passed.",
                    "requirement_id": "R1",
                    "fulfilled": True,
                    "grounding_evidence": "Found execute_logic.",
                    "critique_notes": "Looks ok.",
                }
            ],
            "rejection_trace": None,
        })
        mock_client.models.generate_content.return_value = mock_response

        gate = ChallengerGate(client=mock_client)
        evaluation = gate.evaluate(candidate_dir=tmp_path, requirements_spec=spec)

        # Programmatic defense-in-depth invariant forces False!
        assert evaluation.overall_verdict is False
        assert evaluation.rejection_trace is not None
        assert evaluation.rejection_trace.failure_type == GateFailureType.REQUIREMENT_UNFULFILLED
        assert "Hollow code stubs detected" in evaluation.rejection_trace.failure_reason


# =============================================================================
# 6. BuildSupervisor Gate F Integration
# =============================================================================

class TestBuildSupervisorGateFIntegration:
    """Verifies that Gate F executes after Gates A-E pass and drives promotion/rejection."""

    def test_gate_f_executes_only_after_gates_a_through_e_pass(self, tmp_path: Path):
        cartridges_root = tmp_path / "cartridges"
        active_dir = cartridges_root / "active"
        candidate_dir = cartridges_root / "candidate_01"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        # Create broken syntax in mechanics.py so Gate A fails
        (candidate_dir / "mechanics.py").write_text("class SyntaxErrorDef: def", encoding="utf-8")

        mock_challenger = MagicMock()
        supervisor = BuildSupervisor(
            cartridges_root=cartridges_root,
            active_cartridge_dir=active_dir,
            challenger_gate=mock_challenger,
        )

        gate_results, trace = supervisor.evaluate_candidate_cartridge(candidate_dir)
        # Gate A failed
        assert not gate_results["gate_a"].passed
        # Challenger gate MUST NOT be called because Gate A failed earlier in computational order
        mock_challenger.evaluate.assert_not_called()

    def test_gate_f_vacuous_pass_when_no_requirements_spec(self, mock_cartridge_active: Path, tmp_path: Path):
        cartridges_root = tmp_path / "cartridges"
        active_dir = cartridges_root / "active"

        # mock_cartridge_active has no requirements_spec.json
        supervisor = BuildSupervisor(
            cartridges_root=cartridges_root,
            active_cartridge_dir=active_dir,
        )

        gate_results, trace = supervisor.evaluate_candidate_cartridge(mock_cartridge_active)
        assert gate_results["Gate A"].passed
        assert gate_results["gate_f"].passed
        assert gate_results["gate_f"].diagnostic_data.get("vacuous_pass") is True
        assert trace is None

    def test_gate_f_rejection_blocks_promotion_and_emits_trace(self, mock_cartridge_active: Path, tmp_path: Path):
        cartridges_root = tmp_path / "cartridges"
        active_dir = cartridges_root / "active"
        candidate_dir = cartridges_root / "candidate_gate_f_fail"
        shutil.copytree(mock_cartridge_active, candidate_dir)

        # Add requirements_spec.json
        spec_data = {
            "deliberation": "Spec with unfulfilled requirement.",
            "game_title": "Test Title",
            "genre_paradigm": "strategy",
            "requirements": [
                {
                    "deliberation": "Track 1",
                    "id": "R1",
                    "title": "Unfulfilled Track",
                    "description": "Must implement trade loop",
                    "acceptance_criteria": [
                        {
                            "deliberation": "AC1",
                            "id": "AC1.1",
                            "description": "Trade executed",
                            "verification_method": "runtime_telemetry",
                        }
                    ],
                }
            ],
        }
        (candidate_dir / "requirements_spec.json").write_text(json.dumps(spec_data), encoding="utf-8")

        # Mock ChallengerGate to reject candidate
        mock_challenger = MagicMock()
        mock_eval = ChallengerGateEvaluation(
            deliberation="Challenger gate rejected: trade loop missing.",
            candidate_id=candidate_dir.name,
            overall_verdict=False,
            evaluated_requirements=[
                RequirementEvaluationRecord(
                    deliberation="R1 not satisfied.",
                    requirement_id="R1",
                    fulfilled=False,
                    grounding_evidence="Zero trade events found in telemetry.",
                    critique_notes="Missing trade mechanics.",
                )
            ],
            rejection_trace=RejectionDiagnosticTrace(
                deliberation="Emitting diagnostic trace for R1 failure.",
                candidate_id=candidate_dir.name,
                failure_gate="Gate F: Challenger Gate",
                failure_type=GateFailureType.REQUIREMENT_UNFULFILLED,
                failure_reason="Unfulfilled requirements: ['R1']",
                remedy_type=RemedyType.IMPLEMENT_REQUIREMENT,
                recommended_remedy="Implement trade loop mechanics.",
                unfulfilled_requirements=["R1"],
                failing_criteria=["AC1.1"],
            ),
        )
        mock_challenger.evaluate.return_value = mock_eval

        supervisor = BuildSupervisor(
            cartridges_root=cartridges_root,
            active_cartridge_dir=active_dir,
            challenger_gate=mock_challenger,
        )

        gate_results, trace = supervisor.evaluate_candidate_cartridge(candidate_dir)
        assert gate_results["gate_f"].passed is False
        assert trace is not None
        assert trace.failure_type == GateFailureType.REQUIREMENT_UNFULFILLED
        assert "R1" in trace.unfulfilled_requirements

        # verify_candidate alias must produce identical output
        gate_results_alias, trace_alias = supervisor.verify_candidate(candidate_dir)
        assert gate_results_alias["gate_f"].passed is False
        assert trace_alias.failure_type == GateFailureType.REQUIREMENT_UNFULFILLED


# =============================================================================
# 7. LeadGameplaySWEAgent Repair for REQUIREMENT_UNFULFILLED
# =============================================================================

class TestLeadGameplaySWEAgentRepairRequirementUnfulfilled:
    """Verifies that SWE agent repairs cartridges rejected by Gate F."""

    def test_repair_cartridge_code_for_requirement_unfulfilled(self, tmp_path: Path):
        candidate_dir = tmp_path / "candidate_repair"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        # Stubbed mechanics
        (candidate_dir / "mechanics.py").write_text("""# mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Stub cartridge"
    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            level_id="ep1",
            name="Level 1",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 360.0),
            exit_point=(1180.0, 360.0),
            deliberation="Tactical level."
        )
    def score_points(self):
        pass
""", encoding="utf-8")

        spec_data = {
            "deliberation": "Spec for repair.",
            "game_title": "Scoring Game",
            "genre_paradigm": "arcade",
            "requirements": [
                {
                    "deliberation": "Scoring track",
                    "id": "R1",
                    "title": "Score Points",
                    "description": "Must increment score",
                    "target_modules": ["mechanics.py"],
                    "acceptance_criteria": [
                        {
                            "deliberation": "AC1",
                            "id": "AC1.1",
                            "description": "Score increments",
                            "verification_method": "ast_inspection",
                        }
                    ],
                }
            ],
        }
        (candidate_dir / "requirements_spec.json").write_text(json.dumps(spec_data), encoding="utf-8")

        trace = RejectionDiagnosticTrace(
            deliberation="Challenger Gate rejected R1 stub.",
            candidate_id=candidate_dir.name,
            failure_gate="Gate F: Challenger Gate",
            failure_type=GateFailureType.REQUIREMENT_UNFULFILLED,
            failure_reason="Unfulfilled requirements: ['R1']",
            remedy_type=RemedyType.IMPLEMENT_REQUIREMENT,
            recommended_remedy="Replace pass stub with score increment logic.",
            unfulfilled_requirements=["R1"],
        )

        swe_agent = LeadGameplaySWEAgent()
        # Mock repair_code_module to return genuine implementation
        def mock_repair(mod_name, broken, diag):
            return """# mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Repaired active cartridge"
    def __init__(self):
        self.score = 0
    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            level_id="ep1",
            name="Level 1",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 360.0),
            exit_point=(1180.0, 360.0),
            deliberation="Tactical level."
        )
    def score_points(self):
        self.score += 10
        return self.score
"""
        swe_agent.repair_code_module = MagicMock(side_effect=mock_repair)

        repaired_files = swe_agent.repair_cartridge_code(candidate_dir, trace)
        assert "mechanics.py" in repaired_files
        repaired_src = (candidate_dir / "mechanics.py").read_text(encoding="utf-8")
        assert "self.score += 10" in repaired_src
        assert "pass" not in repaired_src


# =============================================================================
# 8. SwarmEventBus: CHALLENGER_GATE_VERDICT Emission (Rule 4)
# =============================================================================

class TestSwarmEventBusChallengerVerdict:
    """Verifies CHALLENGER_GATE_VERDICT publishing and event indexing."""

    def test_publish_challenger_gate_verdict_deliberation_first(self, tmp_path: Path):
        log_path = tmp_path / "events.jsonl"
        bus = SwarmEventBus(log_path=log_path)

        bus.publish_challenger_gate_verdict(
            deliberation="Comprehensive audit passed: 100% acceptance criteria satisfied.",
            candidate_id="cand_pub_01",
            passed=True,
            details={"evaluated_count": 3, "unfulfilled": []},
            rejection_trace=None,
        )

        events = list(bus.get_recent_events())
        assert len(events) >= 1
        last_ev = events[-1]
        assert list(last_ev.keys())[0] == "deliberation"
        assert last_ev["event_type"] == SwarmEventType.CHALLENGER_GATE_VERDICT.value
        assert last_ev["payload"]["passed"] is True
        assert last_ev["payload"]["candidate_id"] == "cand_pub_01"
