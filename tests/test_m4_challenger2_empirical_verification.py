"""tests/test_m4_challenger2_empirical_verification.py
Empirical stress-testing and verification harness for Milestone M4 Challenger 2.
Exhaustively challenges:
1. Headless playtest invariance (Rule 6: SDL_VIDEODRIVER=dummy, blank surface detection, motion deltas).
2. BuildSupervisor Gate F Integration:
   - Gate F execution strictly after Gates A-E
   - Vacuous pass on missing requirements_spec.json and empty requirements
   - RejectionDiagnosticTrace on failure (failure_type=GateFailureType.REQUIREMENT_UNFULFILLED, unfulfilled_requirements)
   - verify_candidate alias equivalence
3. Closed-loop repair via repair_cartridge_code:
   - Targeted repair of unfulfilled requirement tracks
   - AST validation of repaired files
   - Rule 5 fail-fast enforcement (zero silent fallback to deterministic templates)
4. Telemetry auditing and stagnation detection.
5. Deterministic invariant enforcement (anti-stub defense-in-depth).
"""
from __future__ import annotations
import ast
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch
import wave

from PIL import Image
import pygame
import pytest

from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.sim.headless import HeadlessSimulator, SimulationInput
from engine.studio.models import (
    AcceptanceCriterion,
    GameRequirementsSpec,
    RequirementTrack,
    StudioAgentError,
)
from engine.studio.swe import CodeVerificationError, LeadGameplaySWEAgent
from engine.supervisor.challenger import (
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
)
from engine.supervisor.supervisor import BuildSupervisor


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def clean_cartridge(tmp_path: Path) -> Path:
    """Creates a hermetic candidate cartridge that passes Gates A-E cleanly."""
    c_dir = tmp_path / "candidate_clean"
    assets_dir = c_dir / "assets"
    audio_dir = assets_dir / "audio"
    sprites_dir = assets_dir / "sprites"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sprites_dir.mkdir(parents=True, exist_ok=True)

    # manifest.json
    (c_dir / "manifest.json").write_text(json.dumps({
        "cartridge_id": "cand_clean_01",
        "title": "Quantum Odyssey",
        "version": "1.0.0",
        "author": "Autonomous Studio Swarm",
        "entry_point": "mechanics.py",
        "episodes": [1],
    }, indent=2), encoding="utf-8")

    # level.json
    (c_dir / "level.json").write_text(json.dumps({
        "deliberation": "Clean level geometry.",
        "level_id": "ep1",
        "name": "Level 1",
        "width": 1280.0,
        "height": 720.0,
        "spawn_point": [100.0, 360.0],
        "exit_point": [1180.0, 360.0],
        "walls": [
            [0.0, 0.0, 1280.0, 32.0],
            [0.0, 688.0, 1280.0, 32.0],
        ],
        "enemies": [],
        "hazards": [],
    }, indent=2), encoding="utf-8")

    # narrative.json
    (c_dir / "narrative.json").write_text(json.dumps({
        "deliberation": "Narrative specification.",
        "world": {"setting": "Orbital Station", "theme": "Sci-Fi"},
        "characters": [
            {"id": "hero", "name": "Vanguard 7", "role": "protagonist", "portrait": "assets/sprites/hero.png"},
            {"id": "overseer", "name": "AI Overseer", "role": "rival", "portrait": "assets/sprites/overseer.png"},
        ],
        "dialogues": [
            {
                "trigger": "level_start",
                "character_id": "hero",
                "text": "All systems operational.",
                "voice_clip": "assets/audio/hero_start.wav",
            }
        ],
        "dialogue_barks": {
            "on_damage": [
                {
                    "character_id": "hero",
                    "text": "Shields depleted!",
                    "audio_clip": "hero_start.wav",
                }
            ]
        },
    }, indent=2), encoding="utf-8")

    # mechanics.py
    mechanics_code = """# mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.entities.player import Player as BasePlayer

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active cartridge with kinetic mechanics."
    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            level_id=f"episode_{episode_number}",
            name="Test Level",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 360.0),
            exit_point=(1180.0, 360.0),
            walls=[(0.0, 0.0, 1280.0, 32.0), (0.0, 688.0, 1280.0, 32.0)],
            deliberation="Level layout."
        )

    def create_player(self, x: float, y: float, telemetry_recorder=None) -> BasePlayer:
        return BasePlayer(x=x, y=y)

    def create_enemies(self, spec: BaseLevelSpec):
        return []

    def create_hazards(self, spec: BaseLevelSpec):
        return []
"""
    (c_dir / "mechanics.py").write_text(mechanics_code, encoding="utf-8")

    # Valid WAV audio (0.5s tone)
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

    # Valid PNG sprites
    for s_name in ("hero.png", "overseer.png"):
        img = Image.new("RGB", (32, 32), color=(10, 20, 30))
        img.save(sprites_dir / s_name, format="PNG")

    return c_dir


# =============================================================================
# 1. Headless Playtest Invariance & Canvas Detection (Rule 6)
# =============================================================================

class TestHeadlessPlaytestInvarianceAndCanvasDetection:
    """Empirical verification of headless surface capture, blank detection, and motion deltas."""

    def test_headless_invariance_preflight_env(self):
        """Verifies SDL dummy drivers are active for container invariance."""
        assert os.environ.get("SDL_VIDEODRIVER") == "dummy"
        assert os.environ.get("SDL_AUDIODRIVER") == "dummy"

    def test_capture_surface_png_exact_png_header(self):
        """Verifies capture_surface_png produces authentic 8-byte PNG signature."""
        playtester = CandidateVisualPlaytester()
        surface = pygame.Surface((640, 360))
        surface.fill((45, 90, 135))

        png_bytes = playtester.capture_surface_png(surface)
        assert len(png_bytes) > 64
        # Verify 8-byte PNG magic header: \x89PNG\r\n\x1a\n
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

        # Verify Pillow can parse captured image without error
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (640, 360)
        assert img.format == "PNG"

    def test_blank_surface_detection_all_black(self):
        """Verifies completely black (0 entropy) surface is flagged as blank."""
        playtester = CandidateVisualPlaytester()

        # Mock simulation with authentic Pygame surface
        mock_sim = MagicMock()
        mock_sim.engine.logical_surface = pygame.Surface((1280, 720))
        mock_sim.screen = mock_sim.engine.logical_surface
        mock_scene = MagicMock()
        mock_sim.current_scene = mock_scene

        def render_black(surf):
            surf.fill((0, 0, 0))

        mock_scene.render = MagicMock(side_effect=render_black)

        res = playtester.playtest_and_capture(sim=mock_sim, ticks=30)
        assert res["blank_canvas_detected"] is True
        assert res["non_blank_verified"] is False
        assert res["unique_colors_t0"] == 1
        assert res["unique_colors_t_end"] == 1

    def test_blank_surface_detection_monochrome_white_and_gray(self):
        """Verifies any single-color monochromatic surface (e.g. solid white or gray) is flagged as blank."""
        playtester = CandidateVisualPlaytester()

        for color in [(255, 255, 255), (128, 128, 128), (0, 0, 255)]:
            mock_sim = MagicMock()
            mock_sim.engine.logical_surface = pygame.Surface((1280, 720))
            mock_sim.screen = mock_sim.engine.logical_surface
            mock_scene = MagicMock()
            mock_sim.current_scene = mock_scene
            mock_scene.render = MagicMock(side_effect=lambda surf, c=color: surf.fill(c))

            res = playtester.playtest_and_capture(sim=mock_sim, ticks=30)
            assert res["blank_canvas_detected"] is True, f"Failed for color {color}"
            assert res["non_blank_verified"] is False

    def test_dynamic_motion_delta_calculation(self):
        """Verifies motion delta computes nonzero pixel diff when rendered elements move."""
        playtester = CandidateVisualPlaytester()

        mock_sim = MagicMock()
        mock_sim.engine.logical_surface = pygame.Surface((1280, 720))
        mock_sim.screen = mock_sim.engine.logical_surface
        mock_scene = MagicMock()
        mock_sim.current_scene = mock_scene

        # Dynamic state: offset shifts on every render
        state = {"x": 100}

        def render_dynamic(surf):
            surf.fill((20, 20, 20))
            # Draw moving rectangle
            pygame.draw.rect(surf, (200, 100, 50), (state["x"], 150, 60, 60))
            state["x"] += 50

        mock_scene.render = MagicMock(side_effect=render_dynamic)
        mock_sim.step = MagicMock()

        res = playtester.playtest_and_capture(sim=mock_sim, ticks=30)
        assert res["blank_canvas_detected"] is False
        assert res["non_blank_verified"] is True
        assert res["pixel_diff_t0_to_tend"] > 5
        assert res["visual_motion_verified"] is True

    def test_static_frozen_canvas_fails_motion_verification(self):
        """Verifies a non-blank but completely static scene has zero motion delta."""
        playtester = CandidateVisualPlaytester()

        mock_sim = MagicMock()
        mock_sim.engine.logical_surface = pygame.Surface((1280, 720))
        mock_sim.screen = mock_sim.engine.logical_surface
        mock_scene = MagicMock()
        mock_sim.current_scene = mock_scene

        # Renders static pattern that never moves
        def render_static(surf):
            surf.fill((20, 20, 20))
            pygame.draw.rect(surf, (200, 100, 50), (100, 150, 60, 60))

        mock_scene.render = MagicMock(side_effect=render_static)
        mock_sim.step = MagicMock()

        res = playtester.playtest_and_capture(sim=mock_sim, ticks=30)
        assert res["blank_canvas_detected"] is False
        assert res["non_blank_verified"] is True
        assert res["pixel_diff_t0_to_tend"] == 0
        assert res["visual_motion_verified"] is False


# =============================================================================
# 2. BuildSupervisor Gate F Integration
# =============================================================================

class TestBuildSupervisorGateFIntegrationEmpirical:
    """Exhaustive empirical checks for Gate F sequencing, vacuous pass, and diagnostic trace."""

    def test_gate_f_computational_order_skipped_if_earlier_gates_fail(self, clean_cartridge: Path, tmp_path: Path):
        """Gate F must NOT be called if Gate A, B, C, D, or E fails."""
        cartridges_root = tmp_path / "cartridges"
        active_dir = cartridges_root / "active"
        cartridges_root.mkdir(parents=True, exist_ok=True)
        active_dir.mkdir(parents=True, exist_ok=True)

        mock_challenger = MagicMock()
        supervisor = BuildSupervisor(
            cartridges_root=cartridges_root,
            active_cartridge_dir=active_dir,
            challenger_gate=mock_challenger,
        )

        # 1. Gate A Failure (prohibited os import)
        candidate_gate_a = tmp_path / "cand_gate_a"
        shutil.copytree(clean_cartridge, candidate_gate_a)
        (candidate_gate_a / "mechanics.py").write_text("import os\nclass ActiveCartridge: pass", encoding="utf-8")
        res, trace = supervisor.evaluate_candidate_cartridge(candidate_gate_a)
        assert not res["gate_a"].passed
        mock_challenger.evaluate.assert_not_called()

        # 2. Asset Gate Failure (corrupt audio WAV)
        candidate_gate_asset = tmp_path / "cand_gate_asset"
        shutil.copytree(clean_cartridge, candidate_gate_asset)
        (candidate_gate_asset / "assets" / "audio" / "hero_start.wav").write_bytes(b"CORRUPT_NOT_A_WAV")
        res, trace = supervisor.evaluate_candidate_cartridge(candidate_gate_asset)
        asset_gate = res.get("gate_b") or res.get("gate_e")
        assert asset_gate is not None and not asset_gate.passed
        mock_challenger.evaluate.assert_not_called()

    def test_gate_f_vacuous_pass_missing_requirements_spec(self, clean_cartridge: Path, tmp_path: Path):
        """Cartridge without requirements_spec.json passes Gate F vacuously."""
        cartridges_root = tmp_path / "cartridges"
        active_dir = cartridges_root / "active"

        # clean_cartridge has no requirements_spec.json
        supervisor = BuildSupervisor(
            cartridges_root=cartridges_root,
            active_cartridge_dir=active_dir,
        )

        gate_results, trace = supervisor.evaluate_candidate_cartridge(clean_cartridge)
        assert gate_results["gate_f"].passed is True
        assert gate_results["gate_f"].diagnostic_data.get("vacuous_pass") is True
        assert trace is None

    def test_gate_f_vacuous_pass_empty_requirements_list(self, clean_cartridge: Path, tmp_path: Path):
        """Cartridge with empty requirements: [] passes Gate F vacuously."""
        cartridges_root = tmp_path / "cartridges"
        active_dir = cartridges_root / "active"
        candidate = tmp_path / "cand_empty_reqs"
        shutil.copytree(clean_cartridge, candidate)

        spec = {
            "deliberation": "Empty spec without feature tracks.",
            "game_title": "Empty Spec Game",
            "genre_paradigm": "sandbox",
            "requirements": [],
        }
        (candidate / "requirements_spec.json").write_text(json.dumps(spec), encoding="utf-8")

        supervisor = BuildSupervisor(
            cartridges_root=cartridges_root,
            active_cartridge_dir=active_dir,
        )

        gate_results, trace = supervisor.evaluate_candidate_cartridge(candidate)
        assert gate_results["gate_f"].passed is True
        assert gate_results["gate_f"].diagnostic_data.get("vacuous_pass") is True
        assert trace is None

    def test_gate_f_rejection_trace_structure_and_promotion_block(self, clean_cartridge: Path, tmp_path: Path):
        """Rejection at Gate F yields RejectionDiagnosticTrace with REQUIREMENT_UNFULFILLED and blocks promotion."""
        cartridges_root = tmp_path / "cartridges"
        active_dir = cartridges_root / "active"
        candidate = tmp_path / "cand_gate_f_reject"
        shutil.copytree(clean_cartridge, candidate)

        spec = {
            "deliberation": "Spec with trade requirement.",
            "game_title": "Trade Economy Sim",
            "genre_paradigm": "economy_sim",
            "requirements": [
                {
                    "deliberation": "Track R1: Trading System",
                    "id": "R1",
                    "title": "Trading Loop",
                    "description": "Must implement buy and sell trade offers",
                    "acceptance_criteria": [
                        {
                            "deliberation": "AC1.1",
                            "id": "AC1.1",
                            "description": "Trade offers generated and executed",
                            "verification_method": "runtime_telemetry",
                        }
                    ],
                }
            ],
        }
        (candidate / "requirements_spec.json").write_text(json.dumps(spec), encoding="utf-8")

        mock_challenger = MagicMock()
        mock_eval = ChallengerGateEvaluation(
            deliberation="Challenger gate critique: trade system not implemented.",
            candidate_id=candidate.name,
            overall_verdict=False,
            evaluated_requirements=[
                RequirementEvaluationRecord(
                    deliberation="R1 trade loop unfulfilled.",
                    requirement_id="R1",
                    fulfilled=False,
                    grounding_evidence="No trade events in telemetry.",
                    critique_notes="Missing trade subsystem.",
                )
            ],
            rejection_trace=RejectionDiagnosticTrace(
                deliberation="Rejecting candidate build: R1 unfulfilled.",
                candidate_id=candidate.name,
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

        gate_results, trace = supervisor.evaluate_candidate_cartridge(candidate)
        assert gate_results["gate_f"].passed is False
        assert trace is not None
        assert trace.failure_type == GateFailureType.REQUIREMENT_UNFULFILLED
        assert trace.remedy_type == RemedyType.IMPLEMENT_REQUIREMENT
        assert "R1" in trace.unfulfilled_requirements
        assert "AC1.1" in trace.failing_criteria

        # Verify candidate is NOT promoted to active_dir
        assert not (active_dir / "manifest.json").exists()

        # Verify verify_candidate alias produces identical results
        gate_results_alias, trace_alias = supervisor.verify_candidate(candidate)
        assert gate_results_alias["gate_f"].passed is False
        assert trace_alias.failure_type == GateFailureType.REQUIREMENT_UNFULFILLED
        assert "R1" in trace_alias.unfulfilled_requirements


# =============================================================================
# 3. Closed-Loop Repair via LeadGameplaySWEAgent
# =============================================================================

class TestClosedLoopRepairEmpirical:
    """Empirical verification of SWE agent self-healing on REQUIREMENT_UNFULFILLED."""

    def test_repair_targets_unfulfilled_requirement_modules(self, tmp_path: Path):
        """Verifies repair_cartridge_code extracts target_modules for unfulfilled tracks and repairs them."""
        cand_dir = tmp_path / "cand_swe_repair"
        cand_dir.mkdir(parents=True, exist_ok=True)

        # Module with stub
        (cand_dir / "mechanics.py").write_text("""# mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Stub mechanics"
    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            level_id="ep1",
            name="Test",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 360.0),
            exit_point=(1180.0, 360.0),
            deliberation="Level"
        )
    def trigger_trade(self):
        pass
""", encoding="utf-8")

        # requirements_spec.json specifying target_modules
        spec = {
            "deliberation": "Spec for trade repair.",
            "game_title": "Economy Sim",
            "genre_paradigm": "economy",
            "requirements": [
                {
                    "deliberation": "R1 trade track",
                    "id": "R1",
                    "title": "Trade Functionality",
                    "description": "Implement trigger_trade",
                    "target_modules": ["mechanics.py"],
                    "acceptance_criteria": [
                        {"deliberation": "AC1", "id": "AC1.1", "description": "Trade works", "verification_method": "ast_inspection"}
                    ],
                }
            ],
        }
        (cand_dir / "requirements_spec.json").write_text(json.dumps(spec), encoding="utf-8")

        trace = RejectionDiagnosticTrace(
            deliberation="Gate F rejected R1.",
            candidate_id=cand_dir.name,
            failure_gate="Gate F: Challenger Gate",
            failure_type=GateFailureType.REQUIREMENT_UNFULFILLED,
            failure_reason="Unfulfilled requirements: ['R1']",
            remedy_type=RemedyType.IMPLEMENT_REQUIREMENT,
            recommended_remedy="Implement trigger_trade with genuine state change.",
            unfulfilled_requirements=["R1"],
        )

        swe_agent = LeadGameplaySWEAgent()

        def mock_repair(mod_name, broken, diag):
            assert mod_name == "mechanics.py"
            assert "Unfulfilled requirements: ['R1']" in diag
            return """# mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Repaired mechanics"
    def __init__(self):
        self.trades_executed = 0
    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            level_id="ep1",
            name="Test",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 360.0),
            exit_point=(1180.0, 360.0),
            deliberation="Level"
        )
    def trigger_trade(self):
        self.trades_executed += 1
        return self.trades_executed
"""
        swe_agent.repair_code_module = MagicMock(side_effect=mock_repair)

        repaired_map = swe_agent.repair_cartridge_code(cand_dir, trace)
        assert "mechanics.py" in repaired_map
        content = (cand_dir / "mechanics.py").read_text(encoding="utf-8")
        assert "self.trades_executed += 1" in content
        assert "pass" not in content

    def test_rule_5_fail_fast_on_repair_without_client(self, tmp_path: Path):
        """Verifies repair_code_module raises CodeVerificationError if client is None (Rule 5)."""
        swe_agent = LeadGameplaySWEAgent(client=None)
        with pytest.raises(CodeVerificationError, match="requires an active google.genai Client"):
            swe_agent.repair_code_module("mechanics.py", "def foo(): pass", "fix pass")

    def test_rule_5_prohibition_on_deterministic_code_remedy(self):
        """Verifies _deterministic_code_remedy raises CodeVerificationError under Rule 5."""
        swe_agent = LeadGameplaySWEAgent()
        with pytest.raises(CodeVerificationError, match="strictly prohibited under Rule 5"):
            swe_agent._deterministic_code_remedy("mechanics.py", "broken code")


# =============================================================================
# 4. Telemetry Auditing & Stagnation Detection
# =============================================================================

class TestTelemetryAuditingEmpirical:
    """Empirical verification of CandidateTelemetryAuditor."""

    def test_auditor_extracts_custom_probes(self):
        """Verifies auditor extracts CUSTOM_PROBE events and their payloads."""
        auditor = CandidateTelemetryAuditor()

        mock_sim = MagicMock()
        mock_rec = MagicMock()
        mock_sim.telemetry_recorder = mock_rec
        mock_sim.step_count = 60
        mock_sim.avg_frame_time_ms = 8.5
        mock_sim.frame_budget_passed = True
        mock_sim.elapsed_time = 1.0

        mock_rec.events = [
            {"event_type": "TICK", "frame_index": 1},
            {
                "event_type": "CUSTOM_PROBE",
                "probe_name": "trade_deal_closed",
                "frame_index": 12,
                "payload": {"probe_name": "trade_deal_closed", "volume": 500, "profit": 50},
            },
            {
                "event_type": "CUSTOM_PROBE",
                "probe_name": "inventory_updated",
                "frame_index": 13,
                "payload": {"probe_name": "inventory_updated", "item_count": 5},
            },
        ]

        res = auditor.audit_simulation(sim=mock_sim)
        assert res["total_events_recorded"] == 3
        assert res["event_type_counts"]["CUSTOM_PROBE"] == 2
        assert len(res["custom_probes_fired"]) == 2
        probe_names = [p["probe_name"] for p in res["custom_probes_fired"]]
        assert "trade_deal_closed" in probe_names
        assert "inventory_updated" in probe_names

    def test_auditor_detects_frozen_stagnant_loop(self):
        """Verifies auditor flags frozen scene loop (zero movement, zero events, stagnant counter)."""
        auditor = CandidateTelemetryAuditor()

        mock_sim = MagicMock()
        mock_scene = MagicMock()
        mock_scene.player_pos = (120.0, 360.0)
        mock_scene.initial_state_counter = 0
        mock_scene.state_counter = 0
        mock_sim.current_scene = mock_scene
        mock_sim.player = None
        mock_sim.step_count = 30
        mock_sim.avg_frame_time_ms = 10.0
        mock_sim.frame_budget_passed = True
        mock_sim.elapsed_time = 0.5
        mock_sim.telemetry_recorder.events = []

        res = auditor.audit_simulation(sim=mock_sim)
        assert res["is_stagnant_or_frozen"] is True
        assert res["state_progression_verified"] is False


# =============================================================================
# 5. AST Anti-Stub Defense-in-Depth
# =============================================================================

class TestASTAntiStubDefenseInDepth:
    """Verifies all hollow stub patterns are detected and programmatically override LLM verdicts."""

    @pytest.mark.parametrize("stub_body,expected_stmt", [
        ("pass", "pass"),
        ("...", "..."),
        ("return None", "return None"),
        ("raise NotImplementedError", "raise NotImplementedError"),
    ])
    def test_hollow_stub_patterns_detected(self, tmp_path: Path, stub_body: str, expected_stmt: str):
        inspector = CandidateASTInspector()
        cand = tmp_path / f"cand_stub_{expected_stmt.replace(' ', '_')}"
        cand.mkdir(parents=True, exist_ok=True)
        code = textwrap.dedent(f"""
        class CombatSystem:
            def calculate(self):
                {stub_body}
        """)
        (cand / "mechanics.py").write_text(code, encoding="utf-8")

        res = inspector.inspect_directory(cand)
        assert res["has_critical_stubs"] is True
        assert len(res["pass_stubs"]) >= 1
        found_stmts = [s["stmt"] for s in res["pass_stubs"]]
        assert expected_stmt in found_stmts

    def test_docstring_only_body_detected(self, tmp_path: Path):
        inspector = CandidateASTInspector()
        cand = tmp_path / "cand_docstring"
        cand.mkdir(parents=True, exist_ok=True)
        code = textwrap.dedent("""
        class CombatSystem:
            def execute_move(self):
                \"\"\"Executes the move.\"\"\"
        """)
        (cand / "mechanics.py").write_text(code, encoding="utf-8")

        res = inspector.inspect_directory(cand)
        assert res["has_critical_stubs"] is True
        assert len(res["empty_bodies"]) >= 1

    def test_todo_comment_detected_via_tokenize(self, tmp_path: Path):
        inspector = CandidateASTInspector()
        cand = tmp_path / "cand_todo"
        cand.mkdir(parents=True, exist_ok=True)
        code = textwrap.dedent("""
        # TODO: Implement complete physics calculation
        class Physics:
            def update(self):
                x = 1 + 1
        """)
        (cand / "mechanics.py").write_text(code, encoding="utf-8")

        res = inspector.inspect_directory(cand)
        assert len(res["todo_comments"]) >= 1
        assert res["todo_comments"][0]["tag"] == "TODO"

    def test_disconnected_class_detected(self, tmp_path: Path):
        inspector = CandidateASTInspector()
        cand = tmp_path / "cand_disconnected"
        cand.mkdir(parents=True, exist_ok=True)
        (cand / "mechanics.py").write_text("class ActiveCartridge:\n    def run(self): pass\n", encoding="utf-8")
        (cand / "entities.py").write_text("class ZombieEnemy:\n    def __init__(self):\n        self.hp = 100\n", encoding="utf-8")

        res = inspector.inspect_directory(cand)
        disc = res["disconnected_symbols"]
        assert len(disc) >= 1
        assert any(d["symbol"] == "ZombieEnemy" for d in disc)

    def test_defense_in_depth_overrides_llm_hallucination(self, tmp_path: Path):
        """Even if Gemini claims overall_verdict=True, hollow stubs programmatically force overall_verdict=False."""
        cand = tmp_path / "cand_hallucination"
        cand.mkdir(parents=True, exist_ok=True)
        (cand / "mechanics.py").write_text("class Cartridge:\n    def attack(self):\n        pass\n", encoding="utf-8")

        spec = GameRequirementsSpec(
            deliberation="Spec for defense test.",
            game_title="Combat Game",
            genre_paradigm="action",
            requirements=[
                RequirementTrack(
                    deliberation="R1 track",
                    id="R1",
                    title="Attack system",
                    description="Implement attack",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="AC1",
                            id="AC1.1",
                            description="Attack deals damage",
                            verification_method="ast_inspection",
                        )
                    ],
                )
            ],
        )

        # Mock client returning overall_verdict=True despite pass stub
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "deliberation": "LLM erroneously believes pass stub is acceptable.",
            "candidate_id": cand.name,
            "overall_verdict": True,
            "evaluated_requirements": [
                {
                    "deliberation": "R1 evaluated pass.",
                    "requirement_id": "R1",
                    "fulfilled": True,
                    "grounding_evidence": "Found attack method.",
                    "critique_notes": "Looks good.",
                }
            ],
        })
        mock_client.models.generate_content.return_value = mock_response

        gate = ChallengerGate(client=mock_client)
        eval_result = gate.evaluate(candidate_dir=cand, requirements_spec=spec)

        # Programmatic invariant MUST override LLM hallucination
        assert eval_result.overall_verdict is False
        assert eval_result.rejection_trace is not None
        assert eval_result.rejection_trace.failure_type == GateFailureType.REQUIREMENT_UNFULFILLED
        assert "mechanics.py:attack" in eval_result.rejection_trace.failure_reason
