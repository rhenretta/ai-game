"""tests/test_m4_challenger2_stress.py
Adversarial empirical stress-test suite for Milestone M4 Gate Verification (Challenger 2).
Exhaustively challenges:
1. Playtesting Bot Behavioral Divergence:
   - Measures dash count, average standoff distance to enemies, and clearance time on identical levels.
   - Empirically verifies kinetic divergence (Delta dashes >= 3, standoff distance difference >= 100px).
2. Gate C Playability & Survivability Invariants:
   - Lethal hazards overlapping player spawn within 64px clearance margin.
   - Premature player death (< 180 ticks / 3.0s) causing SURVIVABILITY_VIOLATION.
   - Immediate clearance on exit placed 10px from spawn (evaluating minimum time-to-clear threshold).
3. Headless Simulator Throughput:
   - 10,000 continuous simulation ticks across multiple bot sessions.
   - Throughput > 1,000 ticks/sec with zero memory leaks and zero Cocoa window allocations.
4. Rationale-First Schema Invariance:
   - Verifies all Pydantic models in engine/supervisor/models.py, qa/bots.py, qa/profiles.py, and qa/playtester.py
     produce 'deliberation' as key index 0 in .model_dump() and .model_dump_json().
Strictly 0 occurrences of forbidden legacy branding tokens.
"""
from __future__ import annotations
import gc
import json
import os
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

import pygame
import pytest

from engine.api.contracts import BaseLevelSpec
from engine.sim.simulator import HeadlessSimulator, SimulationInput
from engine.spatial.aabb import Vector2
from engine.supervisor.models import (
    AssetValidationReport,
    BuildEvaluationReport,
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    SupervisorGateVerdict,
)
from engine.supervisor.supervisor import BuildSupervisor
from qa.bots import CautiousBot, RusherBot
from qa.playtester import BotAction, DualPlaytestReport, PlaytestResult
from qa.profiles import CAUTIOUS_PROFILE, RUSHER_PROFILE, PlaytesterProfile


# =============================================================================
# 1. Playtesting Bot Behavioral Divergence
# =============================================================================

class TestPlaytestingBotBehavioralDivergence:
    """Empirically evaluates behavioral and kinetic divergence between RusherBot and CautiousBot."""
    deliberation: str = "Empirically testing kinetic divergence, standoff distance, and dash cadence between bot archetypes."

    @pytest.fixture
    def arena_spec(self) -> BaseLevelSpec:
        return BaseLevelSpec(
            level_id="arena_divergence_test",
            name="Open Divergence Arena",
            width=1280.0,
            height=720.0,
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            walls=[
                (0, 0, 1280, 32),
                (0, 688, 1280, 32),
                (0, 0, 32, 720),
                (1248, 0, 32, 720),
            ],
            enemies=[
                {"id": "chaser_1", "type": "chaser", "x": 500.0, "y": 360.0},
                {"id": "chaser_2", "type": "chaser", "x": 700.0, "y": 360.0},
            ],
            hazards=[],
        )

    def _run_bot_session(self, bot_cls: Any, spec: BaseLevelSpec, max_ticks: int = 1800) -> Dict[str, Any]:
        sim = HeadlessSimulator(cartridge=None, level_spec=spec, max_steps=max_ticks)
        bot = bot_cls()
        exit_vec = Vector2(spec.exit_point[0], spec.exit_point[1])
        navmesh = bot.build_navmesh(spec)
        bot.plan_path(sim.player.center, exit_vec, navmesh)

        standoffs: List[float] = []
        ticks = 0
        while not sim.is_completed:
            ticks += 1
            alive = [e for e in sim.enemies if getattr(e, "is_alive", True)]
            if alive:
                standoffs.append(min(sim.player.center.distance_to(e.center) for e in alive))
            action = bot.decide_action(sim.player, sim.enemies, sim.hazards, exit_vec, sim.walls)
            sim.step(SimulationInput(
                deliberation=action.deliberation,
                move_x=action.move_x,
                move_y=action.move_y,
                dash=action.dash_pressed,
                fire=action.fire_pressed,
                aim_x=action.aim_x,
                aim_y=action.aim_y,
            ))
        summary = sim.get_summary()
        avg_standoff = (sum(standoffs) / len(standoffs)) if standoffs else 0.0
        return {
            "bot": bot_cls.__name__,
            "cleared": summary.cleared,
            "ticks": ticks,
            "time_sec": ticks / 60.0,
            "dashes": sim.stats.get("dashes", 0),
            "avg_standoff": avg_standoff,
            "enemies_killed": summary.enemies_killed,
            "final_health": summary.final_player_health,
        }

    def test_bot_kinetic_divergence_metrics(self, arena_spec: BaseLevelSpec):
        rusher = self._run_bot_session(RusherBot, arena_spec)
        cautious = self._run_bot_session(CautiousBot, arena_spec)

        delta_dashes = rusher["dashes"] - cautious["dashes"]
        standoff_diff = cautious["avg_standoff"] - rusher["avg_standoff"]

        # 1. RusherBot dashes significantly more than CautiousBot (Delta dashes >= 3)
        assert delta_dashes >= 3, f"Delta dashes {delta_dashes} must be >= 3 (rusher={rusher['dashes']}, cautious={cautious['dashes']})"

        # 2. CautiousBot maintains a significantly wider standoff distance (diff >= 100px)
        assert standoff_diff >= 100.0, f"Standoff difference {standoff_diff:.2f}px must be >= 100px (cautious={cautious['avg_standoff']:.2f}, rusher={rusher['avg_standoff']:.2f})"

        # 3. Both bots completed the encounter safely
        assert rusher["cleared"] is True
        assert cautious["cleared"] is True

        # 4. RusherBot finishes faster than CautiousBot due to aggressive dashes
        assert rusher["time_sec"] < cautious["time_sec"]


# =============================================================================
# 2. Gate C Playability & Survivability Invariants
# =============================================================================

class TestGateCSurvivabilityInvariants:
    """Verifies Gate C spawn clearance, premature player death, and clearance time bounds."""
    deliberation: str = "Stress testing Gate C invariants: spawn clearance margin, early death threshold, and rapid clearance."

    @pytest.fixture
    def supervisor(self) -> BuildSupervisor:
        return BuildSupervisor(
            spawn_clearance_margin=64.0,
            early_death_tick_threshold=180,
            max_simulation_ticks=3600,
        )

    def test_spawn_hazard_overlap_within_64px_margin_rejected(self, supervisor: BuildSupervisor):
        sx, sy = 120.0, 360.0
        # Test 1: Directly at spawn (distance = 0)
        spec_exact = BaseLevelSpec(
            level_id="exact_spawn",
            spawn_point=(sx, sy),
            exit_point=(1160.0, 360.0),
            hazards=[{"id": "spike_direct", "x": sx, "y": sy, "w": 32.0, "h": 32.0}],
        )
        res_exact = supervisor.verify_gate_c_survivability(None, spec_exact)
        assert res_exact.passed is False
        assert res_exact.failure_type == GateFailureType.SURVIVABILITY_VIOLATION
        assert "overlaps spawn clearance margin" in res_exact.details

        # Test 2: Within 64px clearance margin (distance = 32px)
        spec_close = BaseLevelSpec(
            level_id="close_spawn",
            spawn_point=(sx, sy),
            exit_point=(1160.0, 360.0),
            hazards=[{"id": "spike_close", "x": sx + 32.0, "y": sy, "w": 32.0, "h": 32.0}],
        )
        res_close = supervisor.verify_gate_c_survivability(None, spec_close)
        assert res_close.passed is False
        assert res_close.failure_type == GateFailureType.SURVIVABILITY_VIOLATION

        # Test 3: Right at boundary (distance = 63px < 64px)
        spec_boundary = BaseLevelSpec(
            level_id="boundary_spawn",
            spawn_point=(sx, sy),
            exit_point=(1160.0, 360.0),
            hazards=[{"id": "spike_bound", "x": sx + 63.0, "y": sy, "w": 32.0, "h": 32.0}],
        )
        res_boundary = supervisor.verify_gate_c_survivability(None, spec_boundary)
        assert res_boundary.passed is False
        assert res_boundary.failure_type == GateFailureType.SURVIVABILITY_VIOLATION

        # Test 4: Safely outside margin (distance = 65px > 64px)
        spec_safe = BaseLevelSpec(
            level_id="safe_spawn",
            spawn_point=(sx, sy),
            exit_point=(1160.0, 360.0),
            hazards=[{"id": "spike_safe", "x": sx + 65.0, "y": sy, "w": 32.0, "h": 32.0}],
        )
        res_safe = supervisor.verify_gate_c_survivability(None, spec_safe)
        assert res_safe.passed is True
        assert res_safe.failure_type == GateFailureType.NONE

    def test_premature_death_under_180_ticks_rejected(self, supervisor: BuildSupervisor):
        spec = BaseLevelSpec(
            level_id="early_death_spec",
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            hazards=[],
        )
        # Death at tick 45 (< 180 ticks / 3.0s)
        sim_result_45 = {
            "first_death_tick": 45,
            "completion_time_sec": 0.75,
            "failure_coordinates": (120.0, 360.0),
        }
        res_45 = supervisor.verify_gate_c_survivability(sim_result_45, spec)
        assert res_45.passed is False
        assert res_45.failure_type == GateFailureType.SURVIVABILITY_VIOLATION
        assert "died prematurely at tick 45" in res_45.deliberation

        # Death at tick 179 (< 180 ticks)
        sim_result_179 = {
            "first_death_tick": 179,
            "completion_time_sec": 2.98,
            "failure_coordinates": (200.0, 360.0),
        }
        res_179 = supervisor.verify_gate_c_survivability(sim_result_179, spec)
        assert res_179.passed is False
        assert res_179.failure_type == GateFailureType.SURVIVABILITY_VIOLATION

        # Death at tick 180 (boundary: not an early-death violation)
        sim_result_180 = {
            "first_death_tick": 180,
            "completion_time_sec": 3.00,
            "failure_coordinates": (250.0, 360.0),
        }
        res_180 = supervisor.verify_gate_c_survivability(sim_result_180, spec)
        assert res_180.passed is True

    def test_exit_placed_10px_from_spawn_empirical_analysis(self, supervisor: BuildSupervisor):
        """Empirically analyzes level where exit portal is placed 10px from spawn.
        Reveals whether a minimum time-to-clear threshold exists in Gate C.
        """
        spec = BaseLevelSpec(
            level_id="near_exit_spec",
            spawn_point=(120.0, 360.0),
            exit_point=(130.0, 360.0),
            hazards=[],
            enemies=[],
        )
        sim_result_rapid = {
            "cleared": True,
            "completion_time_sec": 0.05,
            "completion_ticks": 3,
            "first_death_tick": None,
        }
        gate_c = supervisor.verify_gate_c_survivability(sim_result_rapid, spec)
        # Note: Gate C currently only enforces an upper bound (max_simulation_ticks),
        # lacking a lower-bound minimum time-to-clear threshold or minimum spawn-to-exit distance.
        assert gate_c.passed is True, "Gate C has no lower bound on clearance time; 0.05s clearance passes"


# =============================================================================
# 3. Headless Simulator Throughput & Resource Invariants
# =============================================================================

class TestHeadlessSimulatorThroughput:
    """Verifies HeadlessSimulator executes 10,000 continuous ticks at > 1,000 ticks/sec with zero leaks."""
    deliberation: str = "Benchmarking HeadlessSimulator tick throughput, memory stability, and display isolation."

    def test_10000_ticks_throughput_and_leak_isolation(self):
        # 1. Verify headless driver environment and ensure no display surface is allocated
        if pygame.display.get_init():
            pygame.display.quit()
        assert os.environ.get("SDL_VIDEODRIVER") == "dummy"
        assert os.environ.get("SDL_AUDIODRIVER") == "dummy"
        assert pygame.display.get_surface() is None, "No Cocoa window / display surface may be allocated"

        spec = BaseLevelSpec(
            level_id="bench_level",
            name="Throughput Benchmark Level",
            width=1280.0,
            height=720.0,
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            walls=[(0, 0, 1280, 32), (0, 688, 1280, 32), (0, 0, 32, 720), (1248, 0, 32, 720)],
            enemies=[
                {"id": "e1", "type": "chaser", "x": 500.0, "y": 360.0},
                {"id": "e2", "type": "chaser", "x": 700.0, "y": 360.0},
            ],
            hazards=[],
        )

        gc.collect()
        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()

        target_ticks = 10000
        ticks_executed = 0
        sessions_completed = 0
        bot_types = [RusherBot, CautiousBot]

        start_time = time.perf_counter()

        while ticks_executed < target_ticks:
            bot_cls = bot_types[sessions_completed % len(bot_types)]
            sim = HeadlessSimulator(cartridge=None, level_spec=spec, max_steps=500)
            bot = bot_cls()
            exit_vec = Vector2(spec.exit_point[0], spec.exit_point[1])
            bot.plan_path(sim.player.center, exit_vec, bot.build_navmesh(spec))

            while not sim.is_completed and ticks_executed < target_ticks:
                action = bot.decide_action(sim.player, sim.enemies, sim.hazards, exit_vec, sim.walls)
                sim.step(SimulationInput(
                    deliberation=action.deliberation,
                    move_x=action.move_x,
                    move_y=action.move_y,
                    dash=action.dash_pressed,
                    fire=action.fire_pressed,
                    aim_x=action.aim_x,
                    aim_y=action.aim_y,
                ))
                ticks_executed += 1
            sessions_completed += 1

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        throughput = ticks_executed / elapsed

        gc.collect()
        snap_after = tracemalloc.take_snapshot()
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 2. Verify display isolation preserved throughout all 10k ticks
        assert pygame.display.get_surface() is None, "Display surface must remain None throughout simulation"

        # 3. Verify throughput exceeds 1,000 ticks/sec
        assert throughput > 1000.0, f"Headless simulator throughput {throughput:.1f} ticks/s was below 1000 ticks/s requirement"

        # 4. Verify memory bounds (zero memory leaks: peak < 5MB, net delta < 2MB)
        top_diff = sum(stat.size_diff for stat in snap_after.compare_to(snap_before, "lineno"))
        assert (mem_peak / (1024 * 1024)) < 5.0, f"Peak memory {mem_peak / (1024*1024):.2f}MB exceeded 5MB"
        assert (top_diff / (1024 * 1024)) < 2.0, f"Net memory delta {top_diff / (1024*1024):.2f}MB exceeded 2MB"


# =============================================================================
# 4. Rationale-First Schema Invariance
# =============================================================================

class TestRationaleFirstSchemaInvariance:
    """Verifies that every Pydantic model produces deliberation as key index 0 in dict and JSON dumps."""
    deliberation: str = "Verifying mandatory deliberation field at index 0 across all supervisor and QA models."

    def test_all_models_produce_deliberation_at_key_index_zero(self):
        playtest_r = PlaytestResult(deliberation="Playtest result", bot_name="RusherBot")
        playtest_c = PlaytestResult(deliberation="Playtest result", bot_name="CautiousBot")

        models = [
            ("QualityGateResult", QualityGateResult(
                deliberation="Verified AST security checks.",
                gate_name="Gate A",
                passed=True,
            )),
            ("RejectionDiagnosticTrace", RejectionDiagnosticTrace(
                deliberation="Wall blocked critical path.",
                candidate_id="cand_01",
                failure_gate="Gate B",
                failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
                failure_reason="No open path.",
            )),
            ("BuildEvaluationReport", BuildEvaluationReport(
                deliberation="Build evaluation complete.",
                candidate_id="cand_01",
                status=SupervisorGateVerdict.PROMOTED,
            )),
            ("AssetValidationReport", AssetValidationReport(
                deliberation="Audio asset validated.",
                asset_path="bark.wav",
                asset_type="audio",
                passed=True,
            )),
            ("PlaytesterProfile", PlaytesterProfile(
                deliberation="Rusher behavioral profile.",
                name="rusher",
            )),
            ("BotAction", BotAction(
                deliberation="Steering toward waypoint.",
                move_x=1.0,
                move_y=0.0,
            )),
            ("PlaytestResult", playtest_r),
            ("DualPlaytestReport", DualPlaytestReport(
                deliberation="Dual playtest analysis.",
                rusher_result=playtest_r,
                cautious_result=playtest_c,
                both_cleared=True,
            )),
        ]

        for model_name, model_instance in models:
            # 1. Test model_dump() key ordering
            dumped_dict = model_instance.model_dump()
            keys_dict = list(dumped_dict.keys())
            assert keys_dict[0] == "deliberation", f"{model_name}.model_dump() has first key '{keys_dict[0]}', expected 'deliberation'"

            # 2. Test model_dump_json() key ordering
            dumped_json = model_instance.model_dump_json()
            keys_json = list(json.loads(dumped_json).keys())
            assert keys_json[0] == "deliberation", f"{model_name}.model_dump_json() has first key '{keys_json[0]}', expected 'deliberation'"
