"""tests/test_playtester.py
Comprehensive unit and integration test suite for Feature 15: Synthetic Playtesting Bot (`qa/playtester.py`).
"""
from __future__ import annotations
import ast
from pathlib import Path
import pytest

from engine.api.contracts import BaseLevelSpec
from engine.entities.enemies import ChaserEnemy, RangedEnemy
from engine.entities.hazard import FloorAcidHazard, LaserSweepHazard, SpikeHazard, SpikeState
from engine.entities.player import Player
from engine.loader import GenericCartridge
from engine.sim.headless import HeadlessSimulator
from engine.spatial.aabb import AABB, Vector2
from engine.spatial.navmesh import NavMesh
from qa.playtester import (
    BotAction,
    PlaytestResult,
    SyntheticPlaytester,
    SyntheticPlaytesterBot,
    run_playtest_evaluation,
)
from qa.profiles import CAUTIOUS_PROFILE, RUSHER_PROFILE, PlaytesterProfile


# ============================================================================
# 1. Rationale-First Schema Invariants
# ============================================================================

class TestRationaleFirstSchemaInvariants:
    """Verifies that deliberation is strictly the first attribute across models and AST."""
    deliberation: str = "Schema invariant verification suite."

    def test_playtest_result_ast_deliberation_first(self):
        playtester_file = Path(__file__).resolve().parent.parent / "qa" / "playtester.py"
        with open(playtester_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        found_result = False
        found_action = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == "PlaytestResult":
                    found_result = True
                    first_field = None
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            first_field = item.target.id
                            break
                    assert first_field == "deliberation", f"PlaytestResult first field must be 'deliberation', got '{first_field}'"
                elif node.name == "BotAction":
                    found_action = True
                    first_field = None
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            first_field = item.target.id
                            break
                    assert first_field == "deliberation", f"BotAction first field must be 'deliberation', got '{first_field}'"

        assert found_result, "PlaytestResult class not found in AST"
        assert found_action, "BotAction class not found in AST"

    def test_playtest_result_to_dict_deliberation_first(self):
        res = PlaytestResult(
            deliberation="Verified critical path traversability.",
            traversable=True,
            cleared=True,
            completion_time_sec=4.2,
        )
        d = res.to_dict()
        assert list(d.keys())[0] == "deliberation"
        assert d["deliberation"] == "Verified critical path traversability."
        assert d["traversable"] is True
        assert d["is_traversable"] is True  # Compatibility key

    def test_bot_action_to_dict_deliberation_first(self):
        action = BotAction(
            deliberation="Aiming at closest chaser brawler.",
            move_x=1.0,
            move_y=0.0,
            aim_x=500.0,
            aim_y=300.0,
            fire_pressed=True,
        )
        d = action.to_dict()
        assert list(d.keys())[0] == "deliberation"
        assert d["fire_pressed"] is True


# ============================================================================
# 2. NavMesh & Spatial Traversability Evaluation
# ============================================================================

class TestNavMeshAndSpatialTraversability:
    """Verifies NavMesh rasterization, A* path generation, and blocked path rejection."""
    deliberation: str = "NavMesh traversability verification suite."

    def test_build_navmesh_from_level_spec(self):
        bot = SyntheticPlaytesterBot(tile_size=32.0)
        spec = BaseLevelSpec(
            level_id="test_arena",
            width=640.0,
            height=480.0,
            walls=[
                (0, 0, 640, 32),
                (0, 448, 640, 32),
                (0, 0, 32, 480),
                (608, 0, 32, 480),
                (200, 100, 32, 280),
            ]
        )
        navmesh = bot.build_navmesh(spec)
        assert navmesh.width == 20
        assert navmesh.height == 15
        assert not navmesh.is_walkable(0, 0)
        assert navmesh.is_walkable(2, 2)
        # Check interior wall at x=200 (tile col 6)
        wall_col = int(200 // 32)
        assert not navmesh.is_walkable(wall_col, 5)

    def test_evaluate_traversability_open_corridor(self):
        bot = SyntheticPlaytesterBot(tile_size=32.0)
        spec = BaseLevelSpec(
            level_id="clear_corridor",
            width=640.0,
            height=320.0,
            spawn_point=(64.0, 160.0),
            exit_point=(576.0, 160.0),
            walls=[
                (0, 0, 640, 32),
                (0, 288, 640, 32),
                (0, 0, 32, 320),
                (608, 0, 32, 320),
            ]
        )
        res = bot.evaluate_traversability(spec)
        assert res.traversable is True
        assert res.cleared is True
        assert res.status == "PROMOTED"
        assert res.path_length > 0
        assert res.failure_type is None
        assert "traversable via A* NavMesh" in res.deliberation

    def test_evaluate_traversability_blocked_corridor_rejection(self):
        bot = SyntheticPlaytesterBot(tile_size=32.0)
        # Solid wall completely dividing left and right halves
        spec = BaseLevelSpec(
            level_id="blocked_chamber",
            width=640.0,
            height=320.0,
            spawn_point=(64.0, 160.0),
            exit_point=(576.0, 160.0),
            walls=[
                (0, 0, 640, 32),
                (0, 288, 640, 32),
                (0, 0, 32, 320),
                (608, 0, 32, 320),
                (300, 0, 32, 320),  # Full vertical seal
            ]
        )
        res = bot.evaluate_traversability(spec)
        assert res.traversable is False
        assert res.cleared is False
        assert res.status == "REJECTED"
        assert res.failure_type == "CriticalPathBlockedError"
        assert "critical path blocked" in res.deliberation.lower()


# ============================================================================
# 3. Dynamic Waypoint Steering & Progress Advancement
# ============================================================================

class TestWaypointSteering:
    """Verifies steering direction towards waypoints and node advancement."""
    deliberation: str = "Waypoint steering verification suite."

    def test_waypoint_steering_vector_points_to_target(self):
        bot = SyntheticPlaytesterBot(tile_size=32.0)
        bot.world_waypoints = [Vector2(200.0, 116.0), Vector2(300.0, 116.0)]
        bot.waypoint_index = 0

        player = Player(100.0, 100.0)  # center is (116.0, 116.0)
        action = bot.decide_action(player, [], [], Vector2(300.0, 116.0), [])

        assert action.move_x > 0.9
        assert abs(action.move_y) < 0.1

    def test_waypoint_advances_when_reached(self):
        bot = SyntheticPlaytesterBot(tile_size=32.0, waypoint_tolerance=20.0)
        bot.world_waypoints = [Vector2(110.0, 100.0), Vector2(300.0, 100.0)]
        bot.waypoint_index = 0

        player = Player(100.0, 100.0)  # center is (116, 116), dist to (110, 100) is ~17px < 20px
        action = bot.decide_action(player, [], [], Vector2(300.0, 100.0), [])

        assert bot.waypoint_index == 1


# ============================================================================
# 4. Combat AI: Targeting, Firing & Kiting
# ============================================================================

class TestCombatAI:
    """Verifies weapon aiming, firing activation in range, and melee kiting."""
    deliberation: str = "Combat AI verification suite."

    def test_combat_aims_at_closest_enemy(self):
        bot = SyntheticPlaytesterBot()
        player = Player(100.0, 100.0)
        enemy1 = ChaserEnemy("chaser_1", 300.0, 100.0)
        enemy2 = ChaserEnemy("chaser_2", 500.0, 100.0)

        action = bot.decide_action(player, [enemy1, enemy2], [], Vector2(1000.0, 100.0), [])
        assert action.aim_x == pytest.approx(enemy1.center.x, abs=1e-2)
        assert action.aim_y == pytest.approx(enemy1.center.y, abs=1e-2)
        assert action.fire_pressed is True

    def test_combat_does_not_fire_when_out_of_range(self):
        bot = SyntheticPlaytesterBot()
        player = Player(100.0, 100.0)
        far_enemy = ChaserEnemy("far_enemy", 900.0, 100.0)

        action = bot.decide_action(player, [far_enemy], [], Vector2(1000.0, 100.0), [])
        assert action.fire_pressed is False

    def test_combat_kites_and_dashes_from_closing_chaser(self):
        bot = SyntheticPlaytesterBot(profile=RUSHER_PROFILE)
        player = Player(100.0, 100.0)
        player.dash_cooldown = 0.0
        player.stamina = 100.0
        closing_enemy = ChaserEnemy("chaser_urgent", 140.0, 100.0)  # dist ~40px < 80px melee threat

        action = bot.decide_action(player, [closing_enemy], [], Vector2(500.0, 100.0), [])
        assert action.dash_pressed is True


# ============================================================================
# 5. Hazard Avoidance & Boundary Integrity
# ============================================================================

class TestHazardAvoidanceAndBoundaryIntegrity:
    """Verifies hazard retreat triggers and zero wall-clipping enforcement."""
    deliberation: str = "Hazard and boundary integrity verification suite."

    def test_hazard_evasion_active_spikes(self):
        bot = SyntheticPlaytesterBot()
        player = Player(100.0, 100.0)
        player.dash_cooldown = 0.0
        player.stamina = 100.0
        spikes = SpikeHazard("spikes_1", 120.0, 100.0)
        spikes.state = SpikeState.EXTENDED

        action = bot.decide_action(player, [], [spikes], Vector2(500.0, 100.0), [])
        assert action.dash_pressed is True

    def test_zero_wall_clipping_clean_boundary(self):
        bot = SyntheticPlaytesterBot()
        player_aabb = AABB(100.0, 100.0, 32.0, 32.0)
        walls = [AABB(200.0, 100.0, 32.0, 32.0)]
        assert bot.check_collision_boundary_integrity(player_aabb, walls) is True
        assert bot.clipping_incidents == 0

    def test_wall_clipping_detector_flags_penetration(self):
        bot = SyntheticPlaytesterBot()
        # Penetrate wall by 10 pixels
        player_aabb = AABB(190.0, 100.0, 32.0, 32.0)
        walls = [AABB(200.0, 100.0, 32.0, 32.0)]
        assert bot.check_collision_boundary_integrity(player_aabb, walls) is False
        assert bot.clipping_incidents == 1


# ============================================================================
# 6. Stuck & Oscillation Detection
# ============================================================================

class TestStuckDetection:
    """Verifies detection of progress stalls and stuck loops."""
    deliberation: str = "Stuck detection verification suite."

    def test_stuck_detection_on_immobility(self):
        bot = SyntheticPlaytesterBot(stuck_threshold_ticks=10, stuck_displacement_min=16.0)
        pos = Vector2(100.0, 100.0)
        for _ in range(9):
            assert bot.check_stuck_condition(pos) is False
        # 10th tick with 0 displacement
        assert bot.check_stuck_condition(pos) is True

    def test_no_false_positive_when_moving(self):
        bot = SyntheticPlaytesterBot(stuck_threshold_ticks=10, stuck_displacement_min=16.0)
        for i in range(20):
            pos = Vector2(100.0 + i * 5.0, 100.0)
            stuck = bot.check_stuck_condition(pos)
            assert stuck is False


# ============================================================================
# 7. End-to-End Cartridge Simulation & Legacy Parity
# ============================================================================

class TestEndToEndCartridgeEvaluation:
    """Verifies complete headless playtesting run on SeedCartridge."""
    deliberation: str = "End-to-end cartridge evaluation verification suite."

    def test_seed_cartridge_playtest_promoted(self):
        from engine.api.contracts import BaseCartridge
        class DynamicCartridge(BaseCartridge):
            deliberation: str = "Dynamic cartridge for playtester test."
            def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
                return BaseLevelSpec(
                    level_id="test_01",
                    width=1280.0,
                    height=720.0,
                    spawn_point=(100.0, 360.0),
                    exit_point=(1180.0, 360.0),
                    walls=[(0, 0, 1280, 32), (0, 688, 1280, 32), (0, 0, 32, 720), (1248, 0, 32, 720)],
                    enemies=[],
                    hazards=[],
                    deliberation="Dynamic level spec",
                )
        cartridge = DynamicCartridge()
        bot = SyntheticPlaytesterBot()

        res = bot.evaluate_cartridge(cartridge, max_seconds=30.0)
        assert res.traversable is True
        assert res.status == "PROMOTED"
        assert res.completion_time_sec > 0.0
        assert res.wall_clipping_detected is False
        assert res.stuck is False
        assert "PROMOTED" in res.deliberation

    def test_legacy_evaluate_level_dictionary_parity(self, sample_level_grid_16x16):
        data = sample_level_grid_16x16
        playtester = SyntheticPlaytester(tile_size=32.0)
        report = playtester.evaluate_level(data["grid"], data["spawn"], data["exit"])

        assert report["is_traversable"] is True
        assert report["status"] == "PROMOTED"
        assert report["failure_type"] is None
        assert report["path_length"] > 0
        assert report["estimated_clear_time_sec"] > 0.0
        assert "deliberation" in report

    def test_run_playtest_with_headless_simulator_seed_cartridge(self):
        """Verifies SyntheticPlaytesterBot.run_playtest executes cleanly to completion against HeadlessSimulator."""
        sim = HeadlessSimulator()
        bot = SyntheticPlaytesterBot(profile=RUSHER_PROFILE)

        res = bot.run_playtest(sim)
        assert isinstance(res, PlaytestResult)
        assert res.deliberation, "PlaytestResult must include a non-empty deliberation rationale."
        assert res.traversable is True
        assert res.cleared is True
        assert res.status == "PROMOTED"
        assert res.completion_ticks > 0
        assert res.completion_time_sec > 0.0
        assert res.wall_clipping_detected is False
        assert res.stuck is False
        assert res.failure_type is None

        # Verify Rationale-First dictionary serialization
        d = res.to_dict()
        assert list(d.keys())[0] == "deliberation"

    def test_run_playtest_with_custom_chamber(self):
        """Verifies run_playtest succeeds on custom BaseLevelSpec without pre-existing cartridge."""
        spec = BaseLevelSpec(
            level_id="test_open_chamber",
            width=800.0,
            height=600.0,
            spawn_point=(100.0, 300.0),
            exit_point=(700.0, 300.0),
            walls=[
                (0, 0, 800, 32),
                (0, 568, 800, 32),
                (0, 0, 32, 600),
                (768, 0, 32, 600),
            ],
            enemies=[],
            hazards=[],
            deliberation="Open straight chamber test spec",
        )
        sim = HeadlessSimulator(level_spec=spec, ignore_combat_lock=True)
        bot = SyntheticPlaytesterBot()

        res = bot.run_playtest(sim)
        assert res.cleared is True
        assert res.status == "PROMOTED"
        assert res.completion_ticks > 0
        assert res.wall_clipping_detected is False

    def test_run_playtest_detects_wall_clipping_rejection(self):
        """Verifies run_playtest detects wall penetration and returns a clean REJECTED result."""
        spec = BaseLevelSpec(
            level_id="test_clip_chamber",
            width=800.0,
            height=600.0,
            spawn_point=(16.0, 300.0),  # inside left boundary wall at (0, 0, 32, 600)
            exit_point=(700.0, 300.0),
            walls=[
                (0, 0, 800, 32),
                (0, 568, 800, 32),
                (0, 0, 32, 600),
                (768, 0, 32, 600),
            ],
            enemies=[],
            hazards=[],
            deliberation="Clipping test spec",
        )
        sim = HeadlessSimulator(level_spec=spec, ignore_combat_lock=True)
        bot = SyntheticPlaytesterBot()

        res = bot.run_playtest(sim)
        assert res.status == "REJECTED"
        assert res.failure_type == "WallClippingError"
        assert res.wall_clipping_detected is True


# ============================================================================
# 8. Calibrated Stuck Detection & Bounded Memory
# ============================================================================

class TestStuckDetectionCalibrated:
    """Verifies calibrated stuck detection and bounded memory."""
    deliberation: str = "Targeted tests for calibrated stuck detection and bounded deque."

    def test_wedged_bot_detected_within_ten_seconds(self):
        """Verifies that a wedged bot behind a solid wall is detected as stuck
        within a 10s playtest (at ~4.5s), avoiding the previous 45s timeout failure.
        """
        spec = BaseLevelSpec(
            level_id="wedged_barrier_test",
            width=640.0,
            height=480.0,
            spawn_point=(80.0, 80.0),
            exit_point=(560.0, 240.0),
            walls=[
                (0, 0, 640, 32),
                (0, 448, 640, 32),
                (0, 0, 32, 480),
                (608, 0, 32, 480),
                (300, 32, 32, 384),
            ],
            enemies=[],
            hazards=[],
            deliberation="Chokepoint wedging verification level",
        )

        cart = GenericCartridge(Path("."))
        cart.level_specs[1] = spec

        bot = SyntheticPlaytesterBot(profile=CAUTIOUS_PROFILE, stuck_threshold_ticks=180)
        res = bot.evaluate_cartridge(cart, max_seconds=10.0)

        assert res.cleared is False
        assert res.stuck is True
        assert res.status == "REJECTED"
        assert res.failure_type == "StuckBotError"
        assert res.completion_time_sec <= 5.5
        assert res.completion_ticks <= 330
        assert res.stuck_position is not None

    def test_position_history_bounded_memory(self):
        """Verifies that position_history is strictly bounded to maxlen elements."""
        threshold = 180
        bot = SyntheticPlaytesterBot(stuck_threshold_ticks=threshold)
        assert hasattr(bot.position_history, "maxlen")
        assert bot.position_history.maxlen == threshold

        for i in range(5000):
            pos = Vector2(float(i), float(i))
            bot.check_stuck_condition(pos)

        assert len(bot.position_history) == threshold
        assert bot.position_history[0] == (4820.0, 4820.0)
        assert bot.position_history[-1] == (4999.0, 4999.0)

    def test_run_playtest_detects_stuck_with_simulation_input(self):
        """Verifies that run_playtest detects stuck without NameError."""
        spec = BaseLevelSpec(
            level_id="wedged_sim_test",
            width=640.0,
            height=480.0,
            spawn_point=(80.0, 80.0),
            exit_point=(560.0, 240.0),
            walls=[
                (0, 0, 640, 32),
                (0, 448, 640, 32),
                (0, 0, 32, 480),
                (608, 0, 32, 480),
                (300, 32, 32, 384),
            ],
            enemies=[],
            hazards=[],
            deliberation="Simulator stuck detection level",
        )

        sim = HeadlessSimulator(level_spec=spec, max_steps=600, ignore_combat_lock=True)
        bot = SyntheticPlaytesterBot(profile=CAUTIOUS_PROFILE, stuck_threshold_ticks=180)
        res = bot.run_playtest(sim)
        assert res.stuck is True
        assert res.failure_type == "StuckBotError"


# ============================================================================
# 9. Portal Combat Lock Integrity & Bypass Enforcement
# ============================================================================

class TestPortalCombatLockIntegrity:
    """Verifies that exit portals strictly enforce combat lock unless ignore_combat_lock=True."""
    deliberation: str = "Portal combat lock enforcement and bypass validation suite."

    @pytest.fixture
    def locked_encounter_cartridge(self, tmp_path) -> GenericCartridge:
        """Fixture providing a level spec with walled-off living enemies."""
        spec = BaseLevelSpec(
            level_id="combat_lock_chamber",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 360.0),
            exit_point=(1180.0, 360.0),
            walls=[
                (0, 0, 1280, 32), (0, 688, 1280, 32), (0, 0, 32, 720), (1248, 0, 32, 720),
                # Wall off enemies in separate isolation chamber
                (100, 200, 300, 32),
            ],
            enemies=[
                {"id": f"chaser_{i}", "type": "chaser", "x": 200.0, "y": 100.0}
                for i in range(3)
            ],
            hazards=[],
            deliberation="Combat lock chamber with isolated threats.",
        )
        cart = GenericCartridge(tmp_path)
        cart.level_specs[1] = spec
        return cart

    @pytest.fixture
    def non_combat_cartridge(self, tmp_path) -> GenericCartridge:
        """Fixture providing a pure navigation level with zero enemies."""
        spec = BaseLevelSpec(
            level_id="navigation_corridor",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 360.0),
            exit_point=(1180.0, 360.0),
            walls=[(0, 0, 1280, 32), (0, 688, 1280, 32), (0, 0, 32, 720), (1248, 0, 32, 720)],
            enemies=[],
            hazards=[],
            deliberation="Pure navigation corridor.",
        )
        cart = GenericCartridge(tmp_path)
        cart.level_specs[1] = spec
        return cart

    def test_combat_lock_enforced_by_default(self, locked_encounter_cartridge):
        """Living enemies must prevent portal clearance by default (ignore_combat_lock=False)."""
        bot = SyntheticPlaytesterBot()
        res = bot.evaluate_cartridge(locked_encounter_cartridge, max_seconds=5.0)

        assert res.cleared is False
        assert res.status == "REJECTED"
        assert res.failure_type in ("SimulationTimeoutError", "StuckBotError")
        assert res.enemies_killed == 0

    def test_combat_lock_bypassed_when_flag_enabled(self, locked_encounter_cartridge):
        """Setting ignore_combat_lock=True must allow clearance on portal touch even with enemies alive."""
        bot = SyntheticPlaytesterBot()
        res = bot.evaluate_cartridge(locked_encounter_cartridge, max_seconds=5.0, ignore_combat_lock=True)

        assert res.cleared is True
        assert res.status == "PROMOTED"
        assert res.failure_type is None
        assert res.enemies_killed == 0

    def test_non_combat_level_clears_with_default_flag(self, non_combat_cartridge):
        """Non-combat rooms must clear cleanly on portal contact with default ignore_combat_lock=False."""
        bot = SyntheticPlaytesterBot()
        res = bot.evaluate_cartridge(non_combat_cartridge, max_seconds=5.0)

        assert res.cleared is True
        assert res.status == "PROMOTED"
        assert res.failure_type is None
        assert res.completion_time_sec < 4.0

    def test_evaluate_level_signature_backward_compatibility(self, sample_level_grid_16x16):
        """evaluate_level must accept ignore_combat_lock without error."""
        bot = SyntheticPlaytesterBot()
        data = sample_level_grid_16x16

        # Legacy 3-arg positional call
        r1 = bot.evaluate_level(data["grid"], data["spawn"], data["exit"])
        assert r1["is_traversable"] is True

        # Explicit keyword call
        r2 = bot.evaluate_level(data["grid"], data["spawn"], data["exit"], ignore_combat_lock=True)
        assert r2["is_traversable"] is True

        r3 = bot.evaluate_level(data["grid"], data["spawn"], data["exit"], ignore_combat_lock=False)
        assert r3["is_traversable"] is True

    def test_run_playtest_evaluation_forwards_ignore_combat_lock(self, locked_encounter_cartridge):
        """run_playtest_evaluation must forward ignore_combat_lock to evaluate_cartridge."""
        res_locked = run_playtest_evaluation(locked_encounter_cartridge, max_ticks=300, ignore_combat_lock=False)
        assert res_locked.cleared is False

        res_bypassed = run_playtest_evaluation(locked_encounter_cartridge, max_ticks=300, ignore_combat_lock=True)
        assert res_bypassed.cleared is True
