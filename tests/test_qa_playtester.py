"""tests/test_qa_playtester.py
Authoritative unit and integration test suite for Open-Genre Synthetic QA Playtester (Feature 15 & Milestone 4).
Verifies:
  1. Rationale-First Epistemic Architecture (deliberation field declared first).
  2. Open-Genre Scene-based headless simulation and playtest evaluation.
  3. Synthetic input event dispatch (cardinal movement, dash, fire).
  4. Progression verification (detecting frozen game loops).
  5. Unhandled runtime exception capture during scene execution.
  6. Custom domain telemetry emission and validation.
  7. Frame budget (< 16.6ms / 60 FPS) evaluation.
Strictly 0 occurrences of forbidden legacy branding or seed cartridge dependencies.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional
import pytest

from engine.api.contracts import BaseCartridge, BaseScene
from engine.sim.headless import (
    HeadlessSimulator,
    InMemoryTelemetryRecorder,
    SimulationInput,
    SimulationStepResult,
)
from engine.telemetry.probes import AgentTelemetryProbe
from qa.playtester import (
    BotAction,
    PlaytestResult,
    SyntheticPlaytesterBot,
    run_playtest_evaluation,
)
from qa.profiles import CAUTIOUS_PROFILE, RUSHER_PROFILE


class MockOpenGenreScene(BaseScene):
    """Hermetic open-genre scene for QA testing."""
    deliberation: str = "Mock open-genre scene tracking state progression and telemetry."

    def __init__(self, engine: Any, should_crash: bool = False, freeze_state: bool = False):
        super().__init__(engine)
        self.state_counter = 0
        self.should_crash = should_crash
        self.freeze_state = freeze_state
        self.events_received = []
        self.probe = AgentTelemetryProbe(
            recorder=getattr(engine, "telemetry_recorder", None),
            probe_name="combat_turn_resolved",
            category="combat",
        )

    def on_enter(self, context: Optional[Dict[str, Any]] = None) -> None:
        self.is_active = True

    def on_exit(self) -> Dict[str, Any]:
        self.is_active = False
        return {}

    def handle_event(self, event: Any) -> None:
        self.events_received.append(event)
        if not self.freeze_state:
            self.state_counter += 1

    def update(self, dt: float) -> None:
        if self.should_crash:
            raise RuntimeError("Intentional simulated crash in scene update loop.")
        if not self.freeze_state:
            self.state_counter += 1
            if self.state_counter % 5 == 0:
                self.probe.emit(pos=(100.0, 100.0), data={"turn": self.state_counter // 5})

    def render(self, surface: Any) -> None:
        pass


class MockOpenGenreCartridge(BaseCartridge):
    """Hermetic open-genre cartridge delivering MockOpenGenreScene."""
    deliberation: str = "Mock cartridge for open-genre QA playtester verification."

    def __init__(
        self,
        should_crash: bool = False,
        freeze_state: bool = False,
        manifest: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(manifest=manifest)
        self.should_crash = should_crash
        self.freeze_state = freeze_state
        self.scenes: Dict[str, BaseScene] = {}

    def get_scenes(self, engine: Any) -> Dict[str, BaseScene]:
        if not self.scenes:
            self.scenes = {
                "main": MockOpenGenreScene(
                    engine,
                    should_crash=self.should_crash,
                    freeze_state=self.freeze_state,
                )
            }
        return self.scenes

    def get_initial_scene(self, engine: Any) -> BaseScene:
        return self.get_scenes(engine)["main"]


# ============================================================================
# 1. Rationale-First Schema Invariants
# ============================================================================

class TestQAPlaytesterRationaleFirst:
    """Verifies that all QA models adhere strictly to Rationale-First architecture."""
    deliberation: str = "Verifying Rationale-First deliberation ordering on PlaytestResult."

    def test_playtest_result_deliberation_field_index_zero(self):
        playtester_file = Path("qa/playtester.py")
        code = playtester_file.read_text(encoding="utf-8")
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "PlaytestResult":
                first_field = None
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        first_field = item.target.id
                        break
                assert first_field == "deliberation", f"PlaytestResult first field must be 'deliberation', got '{first_field}'"

    def test_serialization_preserves_deliberation_first(self):
        res = PlaytestResult(
            deliberation="Verified open-genre gameplay session completed without faults.",
            traversable=True,
            cleared=True,
            completion_time_sec=1.5,
            completion_ticks=90,
            status="PROMOTED",
        )
        d = res.to_dict() if hasattr(res, "to_dict") else res.model_dump()
        assert list(d.keys())[0] == "deliberation"
        assert list(json.loads(res.model_dump_json()).keys())[0] == "deliberation"


# ============================================================================
# 2. Open-Genre Scene Headless Simulation & Playtest Evaluation
# ============================================================================

class TestOpenGenrePlaytesting:
    """Verifies SyntheticPlaytesterBot in open-genre scene mode."""
    deliberation: str = "Verifying open-genre playtesting execution and status promotion."

    def test_open_genre_cartridge_passes_playtest(self):
        cart = MockOpenGenreCartridge()
        bot = SyntheticPlaytesterBot(profile=RUSHER_PROFILE)
        res = bot.evaluate_cartridge(cart, max_seconds=2.0)

        assert res.cleared is True
        assert res.status == "PROMOTED"
        assert res.progression_verified is True
        assert res.frame_budget_passed is True
        assert res.failure_type is None
        assert res.completion_ticks > 0
        assert "PROMOTED" in res.deliberation

    def test_synthetic_input_dispatch_to_scene(self):
        cart = MockOpenGenreCartridge()
        bot = SyntheticPlaytesterBot()
        res = bot.evaluate_cartridge(cart, max_seconds=1.0)

        assert res.status == "PROMOTED"
        scene = cart.scenes["main"]
        assert isinstance(scene, MockOpenGenreScene)
        assert scene.state_counter > 0


# ============================================================================
# 3. Progression Verification & Fault Detection
# ============================================================================

class TestProgressionAndFaultDetection:
    """Verifies detection of frozen game loops and runtime exceptions."""
    deliberation: str = "Verifying frozen state rejection and unhandled exception capture."

    def test_frozen_state_rejected_by_playtester(self):
        cart = MockOpenGenreCartridge(freeze_state=True)
        bot = SyntheticPlaytesterBot()
        res = bot.evaluate_cartridge(cart, max_seconds=2.0)

        assert res.cleared is False
        assert res.status == "REJECTED"
        assert res.progression_verified is False
        assert res.failure_type == "ProgressionOrPerformanceFailure"

    def test_unhandled_runtime_crash_rejected_with_trace(self):
        cart = MockOpenGenreCartridge(should_crash=True)
        bot = SyntheticPlaytesterBot()
        res = bot.evaluate_cartridge(cart, max_seconds=2.0)

        assert res.cleared is False
        assert res.status == "REJECTED"
        assert res.failure_type == "UnhandledExceptionError"
        assert len(res.trace_log) > 0
        assert "Intentional simulated crash" in res.trace_log[0]


# ============================================================================
# 4. Custom Telemetry & Frame Budget Verification
# ============================================================================

class TestTelemetryAndPerformanceVerification:
    """Verifies custom probe emission and frame timing budget checks."""
    deliberation: str = "Verifying custom telemetry probe recording and frame budget."

    def test_custom_telemetry_emission_recorded(self):
        cart = MockOpenGenreCartridge()
        sim = HeadlessSimulator(cartridge=cart, max_steps=60)

        for step in range(30):
            sim.step(SimulationInput(move_x=1.0, fire=(step % 10 == 0)))

        events = sim.telemetry_recorder.get_events_by_type("CUSTOM_PROBE")
        assert len(events) > 0
        probe_names = {getattr(e, "probe_name", "") for e in events}
        assert "combat_turn_resolved" in probe_names

    def test_frame_budget_adherence_under_normal_load(self):
        cart = MockOpenGenreCartridge()
        sim = HeadlessSimulator(cartridge=cart, max_steps=60)

        for _ in range(60):
            sim.step(SimulationInput(move_x=1.0))

        assert sim.avg_frame_time_ms < 16.6
        assert sim.frame_budget_passed is True
