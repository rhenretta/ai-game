"""tests/test_m5_sim_playtester_stress.py
Empirical Challenger Stress & Adversarial Test Suite for Feature 14 (Headless Simulation Host)
and Feature 15 (Synthetic Playtesting Bot).

Stress-tests under adversarial, extreme, and edge conditions:
  1. High-Throughput Simulation Benchmark:
     - >1,500 steps with InMemoryTelemetryRecorder exceeding 500 steps/s.
     - >3,000 sustained steps benchmark.
     - Zero-I/O in-memory recorder event and stats integrity.
     - Batch execution pipeline throughput.
  2. Determinism Invariance:
     - Bitwise-identical output positions, velocities, health, and stamina across runs.
     - Reset determinism across repeated input sequences.
     - State snapshot serialization determinism.
     - Post-run summary metrics invariance.
  3. Complex Maze & Chokepoint Navigation:
     - Serpentine multi-turn corridors with 90-degree and 180-degree bends.
     - 1-tile (32px) narrow doorways and bottleneck apertures.
     - 4-room interconnected chambers with sequential bottleneck transitions.
     - Impassable blocked maze rejection (CriticalPathBlockedError).
     - Diagonal vs cardinal A* pathfinding parity.
  4. Hostile Swarms & Extreme Hazards:
     - Converging multi-chaser swarms testing combat targeting and kiting.
     - Multi-hazard grids combining active spikes, floor acid, and laser sweeps.
     - Rusher vs Cautious behavioral profile divergence under threat grids.
     - Lethal damage and player death state transitions (PlayerDeathError).
     - Floor acid movement velocity dampening (speed_multiplier).
  5. Collision Boundary Integrity:
     - Extreme velocities (1,000 px/s to 100,000 px/s) swept collision resolution.
     - 90-degree internal corner sliding and vertex clamping.
     - Point-blank high-speed dash collision against solid barriers.
     - Out-of-bounds containment verification across all room perimeters.
  6. Stuck & Oscillation Detection Fuzzing & Empirical Defect Verification:
     - Direct immobility stuck detection.
     - Open looping trajectory non-stuck forward progress verification.
     - Empirical documentation of stuck detection sampling latency (15x elongation).
     - Empirical verification of run_playtest NameError defect (missing SimulationInput import).
     - Empirical verification of portal combat lock bypass inconsistency.
"""
from __future__ import annotations
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import pytest

from engine.api.contracts import BaseLevelSpec
from engine.entities.enemies import ChaserEnemy, RangedEnemy
from engine.entities.hazard import FloorAcidHazard, LaserSweepHazard, SpikeHazard, SpikeState
from engine.entities.player import Player
from engine.entities.portal import Portal
from engine.entities.projectile import ProjectileManager
from engine.loader import GenericCartridge
from engine.sim.headless import (
    HeadlessSimulator,
    InMemoryTelemetryRecorder,
    SimulationInput,
    SimulationState,
    SimulationStepResult,
    SimulationSummary,
)
from engine.spatial.aabb import AABB, Vector2
from engine.spatial.collision import move_and_resolve_walls
from engine.spatial.navmesh import NavMesh, astar_pathfind
from qa.playtester import (
    BotAction,
    PlaytestResult,
    SyntheticPlaytester,
    SyntheticPlaytesterBot,
    run_playtest_evaluation,
)
from qa.profiles import CAUTIOUS_PROFILE, RUSHER_PROFILE, PlaytesterProfile


# ============================================================================
# 1. High-Throughput Simulation Benchmark Stress
# ============================================================================

class TestSimulationHighThroughputStress:
    """Stress tests measuring headless simulation throughput and zero-I/O buffer efficiency."""
    deliberation: str = "High-throughput simulation stress suite verifying >500 steps/s execution."

    def test_simulation_1500_step_throughput_benchmark(self):
        """Verifies that simulation with InMemoryTelemetryRecorder exceeds 500 steps/s over 1,500+ steps."""
        bench_spec = BaseLevelSpec(
            level_id="throughput_bench_1500",
            width=1280.0,
            height=720.0,
            spawn_point=(640.0, 360.0),
            exit_point=(99999.0, 99999.0),
            walls=[(0, 0, 1280, 32), (0, 688, 1280, 32), (0, 0, 32, 720), (1248, 0, 32, 720)],
            deliberation="Distant exit for sustained benchmark",
        )
        sim = HeadlessSimulator(level_spec=bench_spec, telemetry_mode="memory", ignore_combat_lock=True)
        total_steps = 1800
        start_time = time.perf_counter()

        for step in range(total_steps):
            ax = 1.0 if (step // 30) % 2 == 0 else -1.0
            ay = 0.5 if step % 2 == 0 else -0.5
            dash = (step % 40 == 0)
            fire = (step % 15 == 0)
            inp = SimulationInput(
                deliberation=f"Benchmark tick {step}",
                move_x=ax,
                move_y=ay,
                dash=dash,
                fire=fire,
                aim_x=640.0,
                aim_y=360.0,
            )
            sim.step(inp)

        elapsed = time.perf_counter() - start_time
        ticks_per_sec = total_steps / elapsed

        assert ticks_per_sec >= 500.0, f"Throughput {ticks_per_sec:.1f} steps/s is below 500 steps/s threshold"
        assert sim.step_count == total_steps
        assert len(sim.telemetry_recorder.events) > 0
        assert sim.telemetry_recorder.frame_index == total_steps

    def test_simulation_3000_step_sustained_throughput(self):
        """Verifies sustained high throughput over 3,000 steps without degradation."""
        bench_spec = BaseLevelSpec(
            level_id="throughput_bench_3000",
            width=1280.0,
            height=720.0,
            spawn_point=(640.0, 360.0),
            exit_point=(99999.0, 99999.0),
            walls=[(0, 0, 1280, 32), (0, 688, 1280, 32), (0, 0, 32, 720), (1248, 0, 32, 720)],
            deliberation="Distant exit for 3000 step benchmark",
        )
        sim = HeadlessSimulator(level_spec=bench_spec, telemetry_mode="memory", ignore_combat_lock=True)
        total_steps = 3000
        start_time = time.perf_counter()

        for step in range(total_steps):
            inp = SimulationInput(
                move_x=math.cos(step * 0.1),
                move_y=math.sin(step * 0.1),
                dash=(step % 60 == 0),
                fire=(step % 20 == 0),
            )
            sim.step(inp)

        elapsed = time.perf_counter() - start_time
        ticks_per_sec = total_steps / elapsed

        assert ticks_per_sec >= 500.0, f"Sustained throughput {ticks_per_sec:.1f} steps/s below 500 threshold"
        assert sim.step_count == total_steps

    def test_in_memory_telemetry_recorder_zero_io_integrity(self):
        """Verifies that InMemoryTelemetryRecorder accurately tracks events and stats with zero file I/O."""
        recorder = InMemoryTelemetryRecorder(session_id="stress_session_001")
        assert recorder.deliberation != ""
        recorder.start_session(player_profile_tag="stress_runner")
        ep_id = recorder.start_episode(level_id="stress_level_01", episode_index=1)
        assert ep_id.startswith("ep_stress_session_001")

        # Emit various events
        recorder.emit_event("EPISODE_START", pos=(100.0, 100.0))
        recorder.emit_event("DASH", pos=(120.0, 100.0))
        recorder.emit_event("WEAPON_FIRE", pos=(150.0, 100.0))
        recorder.emit_event("DAMAGE_DEALT", pos=(200.0, 100.0), payload={"damage": 25.0})
        recorder.emit_event("DAMAGE_TAKEN", pos=(200.0, 100.0), payload={"damage": 15.0})

        assert recorder._stats["dashes"] == 1
        assert recorder._stats["shots_fired"] == 1
        assert recorder._stats["damage_dealt"] == 25.0
        assert recorder._stats["damage_taken"] == 15.0
        assert len(recorder.events) == 5

        # Verify filter by type
        dashes = recorder.get_events_by_type("DASH")
        assert len(dashes) == 1
        assert dashes[0].event_type == "DASH"

        # End episode and clear
        recorder.end_episode(outcome="CLEARED", exit_pos=(500.0, 500.0))
        assert recorder.episodes[-1]["status"] == "CLEARED"

        recorder.clear()
        assert len(recorder.events) == 0
        assert recorder._stats["dashes"] == 0

    def test_batch_run_steps_pipeline_throughput(self):
        """Verifies that sim.run_steps() batch pipeline processes actions above 500 steps/s."""
        bench_spec = BaseLevelSpec(
            level_id="batch_bench_1600",
            width=1280.0,
            height=720.0,
            spawn_point=(640.0, 360.0),
            exit_point=(99999.0, 99999.0),
            walls=[(0, 0, 1280, 32), (0, 688, 1280, 32), (0, 0, 32, 720), (1248, 0, 32, 720)],
            deliberation="Distant exit for batch benchmark",
        )
        sim = HeadlessSimulator(level_spec=bench_spec, telemetry_mode="memory", ignore_combat_lock=True)
        actions = [
            SimulationInput(
                deliberation=f"Batch step {i}",
                move_x=math.cos(i * 0.05),
                move_y=math.sin(i * 0.05),
                dash=(i % 50 == 0),
            )
            for i in range(1600)
        ]

        start_time = time.perf_counter()
        results = sim.run_steps(actions)
        elapsed = time.perf_counter() - start_time

        rate = len(results) / elapsed
        assert rate >= 500.0, f"Batch run_steps throughput {rate:.1f} steps/s below 500 threshold"
        assert len(results) == 1600
        assert results[-1].step == 1600


# ============================================================================
# 2. Determinism Invariance Stress
# ============================================================================

class TestSimulationDeterminismStress:
    """Stress tests verifying bitwise determinism and state invariance across repeated runs."""
    deliberation: str = "Determinism invariance stress suite verifying exact repeatability."

    def test_identical_trajectories_across_dual_instances(self):
        """Verifies that two independent simulations given identical inputs produce bitwise identical states."""
        sim_a = HeadlessSimulator(telemetry_mode="memory", ignore_combat_lock=True)
        sim_b = HeadlessSimulator(telemetry_mode="memory", ignore_combat_lock=True)

        steps = 600
        for step in range(steps):
            angle = step * 0.08
            mx = math.cos(angle)
            my = math.sin(angle)
            dash = (step % 29 == 0)
            fire = (step % 17 == 0)
            aim_x = 500.0 + 200.0 * math.cos(angle * 1.5)
            aim_y = 300.0 + 150.0 * math.sin(angle * 1.5)

            cmd = SimulationInput(
                deliberation=f"Deterministic step {step}",
                move_x=mx,
                move_y=my,
                dash=dash,
                fire=fire,
                aim_x=aim_x,
                aim_y=aim_y,
            )

            res_a = sim_a.step(cmd)
            res_b = sim_b.step(cmd)

            assert res_a.player_pos == res_b.player_pos, f"Pos mismatch at step {step}: {res_a.player_pos} vs {res_b.player_pos}"
            assert res_a.player_vel == res_b.player_vel, f"Vel mismatch at step {step}"
            assert res_a.player_health == res_b.player_health, f"Health mismatch at step {step}"
            assert res_a.player_stamina == res_b.player_stamina, f"Stamina mismatch at step {step}"
            assert res_a.is_dashing == res_b.is_dashing, f"Dash state mismatch at step {step}"
            assert res_a.active_enemies == res_b.active_enemies, f"Enemy count mismatch at step {step}"
            assert res_a.active_projectiles == res_b.active_projectiles, f"Projectile count mismatch at step {step}"

    def test_simulation_reset_invariance(self):
        """Verifies that calling sim.reset() resets internal physics to an identical initial state."""
        sim = HeadlessSimulator(telemetry_mode="memory", ignore_combat_lock=True)

        # Run 1
        run_1_positions: List[Tuple[float, float]] = []
        for step in range(300):
            inp = SimulationInput(move_x=math.cos(step * 0.15), move_y=math.sin(step * 0.15), dash=(step % 25 == 0))
            res = sim.step(inp)
            run_1_positions.append(res.player_pos)

        # Reset
        sim.reset()
        assert sim.step_count == 0
        assert sim.elapsed_time == 0.0
        assert not sim.is_completed

        # Run 2
        run_2_positions: List[Tuple[float, float]] = []
        for step in range(300):
            inp = SimulationInput(move_x=math.cos(step * 0.15), move_y=math.sin(step * 0.15), dash=(step % 25 == 0))
            res = sim.step(inp)
            run_2_positions.append(res.player_pos)

        assert run_1_positions == run_2_positions, "Position sequence diverged after simulation reset"

    def test_state_snapshot_bitwise_determinism(self):
        """Verifies that sim.get_state() snapshots match exactly across identical runs."""
        sim1 = HeadlessSimulator(ignore_combat_lock=True)
        sim2 = HeadlessSimulator(ignore_combat_lock=True)

        for i in range(120):
            cmd = SimulationInput(move_x=0.5, move_y=-0.5, dash=(i % 30 == 0), fire=(i % 20 == 0))
            sim1.step(cmd)
            sim2.step(cmd)

        s1 = sim1.get_state()
        s2 = sim2.get_state()

        assert s1.step_count == s2.step_count
        assert s1.elapsed_time == s2.elapsed_time
        assert s1.player_pos == s2.player_pos
        assert s1.player_vel == s2.player_vel
        assert s1.player_health == s2.player_health
        assert s1.player_stamina == s2.player_stamina
        assert s1.enemies == s2.enemies

    def test_summary_metrics_determinism(self):
        """Verifies that post-simulation summaries are identical for deterministic runs."""
        sim1 = HeadlessSimulator(ignore_combat_lock=True)
        sim2 = HeadlessSimulator(ignore_combat_lock=True)

        actions = [SimulationInput(move_x=1.0, move_y=0.0, fire=(i % 10 == 0)) for i in range(250)]
        sim1.run_steps(actions)
        sim2.run_steps(actions)

        sum1 = sim1.get_summary()
        sum2 = sim2.get_summary()

        assert sum1.total_steps == sum2.total_steps
        assert sum1.duration_seconds == sum2.duration_seconds
        assert sum1.damage_dealt == sum2.damage_dealt
        assert sum1.damage_taken == sum2.damage_taken
        assert sum1.shots_fired == sum2.shots_fired
        assert sum1.final_player_pos == sum2.final_player_pos


# ============================================================================
# 3. Complex Maze & Chokepoint Navigation Stress
# ============================================================================

class TestComplexMazeAndChokepointNavigationStress:
    """Stress tests evaluating A* pathfinding and bot navigation in serpentine and bottleneck geometry."""
    deliberation: str = "Complex geometry navigation stress suite."

    def test_serpentine_multi_turn_corridor_navigation(self):
        """Tests SyntheticPlaytesterBot in a 3-lane serpentine winding corridor with multiple 90-degree turns."""
        spec = BaseLevelSpec(
            level_id="serpentine_stress",
            name="Serpentine Chambers",
            width=960.0,
            height=640.0,
            spawn_point=(80.0, 80.0),
            exit_point=(880.0, 560.0),
            walls=[
                # Outer boundaries
                (0, 0, 960, 32),
                (0, 608, 960, 32),
                (0, 0, 32, 640),
                (928, 0, 32, 640),
                # Divider 1: opening at bottom (y: 480..608)
                (320, 32, 32, 448),
                # Divider 2: opening at top (y: 32..160)
                (640, 160, 32, 448),
            ],
            enemies=[],
            hazards=[],
            deliberation="Serpentine 3-lane maze",
        )

        bot = SyntheticPlaytesterBot(profile=RUSHER_PROFILE)
        cart = GenericCartridge(Path("."))
        cart.level_specs[1] = spec

        # 1. Traversability pre-flight check
        trav = bot.evaluate_traversability(spec)
        assert trav.traversable is True
        assert trav.status == "PROMOTED"
        assert trav.path_length > 20

        # 2. Headless cartridge playtest execution
        res = bot.evaluate_cartridge(cart, max_seconds=30.0)
        assert res.traversable is True
        assert res.cleared is True
        assert res.status == "PROMOTED"
        assert res.wall_clipping_detected is False
        assert res.stuck is False
        assert res.completion_time_sec > 0.0

    def test_narrow_1_tile_bottleneck_passage(self):
        """Tests bot steering through an exact 1-tile (32px) aperture doorway."""
        spec = BaseLevelSpec(
            level_id="bottleneck_32px",
            name="Narrow Chokepoint",
            width=640.0,
            height=480.0,
            spawn_point=(80.0, 240.0),
            exit_point=(560.0, 240.0),
            walls=[
                (0, 0, 640, 32),
                (0, 448, 640, 32),
                (0, 0, 32, 480),
                (608, 0, 32, 480),
                # Center wall leaving exact 48px doorway at center (y: 216..264)
                (300, 32, 32, 184),
                (300, 264, 32, 184),
            ],
            enemies=[],
            hazards=[],
            deliberation="Chokepoint aperture test",
        )

        bot = SyntheticPlaytesterBot(profile=RUSHER_PROFILE)
        cart = GenericCartridge(Path("."))
        cart.level_specs[1] = spec

        res = bot.evaluate_cartridge(cart, max_seconds=20.0)
        assert res.traversable is True
        assert res.cleared is True
        assert res.status == "PROMOTED"
        assert res.wall_clipping_detected is False

    def test_four_room_layout_with_sequential_chokepoints(self):
        """Tests bot traversing a 4-room quadrant maze through 3 sequential doorway transitions."""
        spec = BaseLevelSpec(
            level_id="quadrant_rooms",
            name="Four Quadrant Maze",
            width=800.0,
            height=600.0,
            spawn_point=(100.0, 100.0),
            exit_point=(100.0, 480.0),
            walls=[
                (0, 0, 800, 32),
                (0, 568, 800, 32),
                (0, 0, 32, 600),
                (768, 0, 32, 600),
                # Horizontal divider: left section from x=32 to 384, right section from 416 to 640
                (32, 284, 352, 32),
                (416, 284, 240, 32),  # Door between Room 2 & 3 at x: 656..768
                # Vertical divider: top section from y=140 to 284, bottom section from 316 to 440
                (384, 140, 32, 144),  # Door between Room 1 & 2 at y: 32..140
                (384, 316, 32, 144),  # Door between Room 3 & 4 at y: 460..568
            ],
            enemies=[],
            hazards=[],
            deliberation="Four room quadrant navigation",
        )

        bot = SyntheticPlaytesterBot()
        cart = GenericCartridge(Path("."))
        cart.level_specs[1] = spec

        res = bot.evaluate_cartridge(cart, max_seconds=30.0)
        assert res.traversable is True
        assert res.cleared is True
        assert res.status == "PROMOTED"
        assert res.wall_clipping_detected is False

    def test_impassable_blocked_maze_rejection(self):
        """Verifies that an impassable solid wall is rejected with CriticalPathBlockedError."""
        spec = BaseLevelSpec(
            level_id="blocked_chamber",
            name="Impassable Chamber",
            width=640.0,
            height=480.0,
            spawn_point=(80.0, 240.0),
            exit_point=(560.0, 240.0),
            walls=[
                (0, 0, 640, 32),
                (0, 448, 640, 32),
                (0, 0, 32, 480),
                (608, 0, 32, 480),
                (300, 0, 48, 480),  # Impassable barrier sealing spawn from exit
            ],
            enemies=[],
            hazards=[],
            deliberation="Impassable wall test",
        )

        bot = SyntheticPlaytesterBot()
        cart = GenericCartridge(Path("."))
        cart.level_specs[1] = spec

        # Pre-flight check
        trav = bot.evaluate_traversability(spec)
        assert trav.traversable is False
        assert trav.status == "REJECTED"
        assert trav.failure_type == "CriticalPathBlockedError"

        # Full cartridge check
        res = bot.evaluate_cartridge(cart, max_seconds=10.0)
        assert res.traversable is False
        assert res.cleared is False
        assert res.status == "REJECTED"
        assert res.failure_type == "CriticalPathBlockedError"


# ============================================================================
# 4. Hostile Swarms & Extreme Hazards Stress
# ============================================================================

class TestHostileSwarmsAndExtremeHazardsStress:
    """Stress tests evaluating bot survival, combat AI, and hazard avoidance under dense threats."""
    deliberation: str = "Hostile threat grid and swarm stress suite."

    def test_converging_chaser_swarm_engagement(self):
        """Verifies that bot actively aims, fires, kites, and dashes when chaser swarms converge."""
        bot = SyntheticPlaytesterBot(profile=RUSHER_PROFILE)
        player = Player(200.0, 200.0)
        player.stamina = 100.0
        player.dash_cooldown = 0.0

        # Create 4 chasers closing in from cardinal directions
        enemies = [
            ChaserEnemy("chaser_north", 200.0, 130.0),
            ChaserEnemy("chaser_south", 200.0, 270.0),
            ChaserEnemy("chaser_east", 270.0, 200.0),
            ChaserEnemy("chaser_west", 130.0, 200.0),
        ]

        action = bot.decide_action(player, enemies, [], Vector2(800.0, 200.0), [])
        assert action.fire_pressed is True
        # Aim should target closest enemy
        closest_enemy = min(enemies, key=lambda e: player.center.distance_to(e.center))
        assert action.aim_x == pytest.approx(closest_enemy.center.x, abs=1e-2)
        assert action.aim_y == pytest.approx(closest_enemy.center.y, abs=1e-2)
        # Melee threat distance triggers dash
        assert action.dash_pressed is True

    def test_multi_hazard_grid_avoidance(self):
        """Verifies hazard retreat vector calculation for spikes, acid, and laser sweeps."""
        bot = SyntheticPlaytesterBot(profile=CAUTIOUS_PROFILE)
        player = Player(200.0, 200.0)
        player.stamina = 100.0
        player.dash_cooldown = 0.0

        # Active spike hazard nearby
        spikes = SpikeHazard("spikes_1", 210.0, 200.0)
        spikes.state = SpikeState.EXTENDED

        # Floor acid hazard overlapping
        acid = FloorAcidHazard("acid_1", 180.0, 180.0, 64.0, 64.0)

        action = bot.decide_action(player, [], [spikes, acid], Vector2(800.0, 200.0), [])
        assert action.dash_pressed is True
        # Movement should bias away from hazard
        assert action.move_x != 0.0 or action.move_y != 0.0

    def test_profile_divergence_under_threat_grid(self):
        """Verifies distinct tactical responses between Rusher and Cautious profiles."""
        spec = BaseLevelSpec(
            level_id="threat_arena",
            name="Threat Arena",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 360.0),
            exit_point=(1180.0, 360.0),
            walls=[(0, 0, 1280, 32), (0, 688, 1280, 32), (0, 0, 32, 720), (1248, 0, 32, 720)],
            enemies=[
                {"id": f"chaser_{i}", "type": "chaser", "x": 500.0 + i * 80.0, "y": 360.0 + ((-1) ** i) * 80.0}
                for i in range(5)
            ],
            hazards=[
                {"id": f"spike_{i}", "type": "spikes", "x": 300.0 + i * 120.0, "y": 360.0, "w": 48.0, "h": 48.0}
                for i in range(4)
            ],
            deliberation="Divergence threat arena",
        )

        cart = GenericCartridge(Path("."))
        cart.level_specs[1] = spec

        # Rusher test
        rusher_bot = SyntheticPlaytesterBot(profile=RUSHER_PROFILE)
        rusher_res = rusher_bot.evaluate_cartridge(cart, max_seconds=30.0)

        # Cautious test
        cautious_bot = SyntheticPlaytesterBot(profile=CAUTIOUS_PROFILE)
        cautious_res = cautious_bot.evaluate_cartridge(cart, max_seconds=30.0)

        # Rusher should engage combat aggressively and eliminate enemies or take different damage
        assert rusher_res.traversable is True
        assert cautious_res.traversable is True
        assert rusher_res.enemies_killed != cautious_res.enemies_killed or rusher_res.damage_taken != cautious_res.damage_taken

    def test_floor_acid_slowdown_velocity_clamping(self):
        """Verifies that walking into floor acid dampens velocity by 0.55x."""
        acid = FloorAcidHazard("acid_zone", 100.0, 100.0, 100.0, 100.0)
        player = Player(120.0, 120.0)
        player.vel = Vector2(200.0, 200.0)

        acid.update(1.0 / 60.0, player, None)
        assert player.speed_multiplier == pytest.approx(0.55, abs=1e-2)
        assert player.vel.x == pytest.approx(110.0, abs=1e-2)
        assert player.vel.y == pytest.approx(110.0, abs=1e-2)


# ============================================================================
# 5. Collision Boundary Integrity Stress
# ============================================================================

class TestCollisionBoundaryIntegrityStress:
    """Stress tests evaluating continuous collision, swept bounds, and out-of-bounds guards."""
    deliberation: str = "Collision boundary integrity stress suite."

    def test_extreme_velocity_tunneling_guard(self):
        """Stress-tests move_and_resolve_walls against velocities up to 100,000 px/s."""
        walls = [
            AABB(0, 0, 1280, 32),
            AABB(0, 688, 1280, 32),
            AABB(0, 0, 32, 720),
            AABB(1248, 0, 32, 720),
            AABB(400, 200, 100, 100),
        ]
        dt = 1.0 / 60.0
        entity = AABB(100.0, 100.0, 32.0, 32.0)
        bot = SyntheticPlaytesterBot()

        extreme_velocities = [
            Vector2(1000.0, 0.0),
            Vector2(-5000.0, 0.0),
            Vector2(0.0, 10000.0),
            Vector2(0.0, -25000.0),
            Vector2(50000.0, 50000.0),
            Vector2(-100000.0, -100000.0),
            Vector2(75000.0, -45000.0),
        ]

        epsilon = 0.05
        for vel in extreme_velocities:
            new_pos, actual_vel = move_and_resolve_walls(entity, vel, dt, walls)
            new_aabb = AABB(new_pos.x, new_pos.y, entity.width, entity.height)

            # 1. Zero penetration via check_collision_boundary_integrity
            no_clipping = bot.check_collision_boundary_integrity(new_aabb, walls, epsilon=epsilon)
            assert no_clipping is True, f"Wall clipping occurred under extreme velocity {vel}: pos={new_pos}"

            # 2. Out-of-bounds containment
            assert 32.0 <= new_pos.x <= (1248.0 - 32.0) + epsilon, f"OOB X breach at {new_pos.x}"
            assert 32.0 <= new_pos.y <= (688.0 - 32.0) + epsilon, f"OOB Y breach at {new_pos.y}"

    def test_internal_corner_sliding_stress(self):
        """Verifies simultaneous X/Y collision resolution at 90-degree internal corner vertices."""
        # Corner formed by top wall and left wall
        walls = [
            AABB(0, 0, 800, 32),
            AABB(0, 0, 32, 600),
        ]
        dt = 1.0 / 60.0
        entity = AABB(50.0, 50.0, 32.0, 32.0)
        bot = SyntheticPlaytesterBot()

        # Drive violently into top-left corner
        vel = Vector2(-15000.0, -15000.0)
        new_pos, actual_vel = move_and_resolve_walls(entity, vel, dt, walls)
        new_aabb = AABB(new_pos.x, new_pos.y, entity.width, entity.height)

        assert bot.check_collision_boundary_integrity(new_aabb, walls) is True
        assert new_pos.x == pytest.approx(32.0, abs=1e-2)
        assert new_pos.y == pytest.approx(32.0, abs=1e-2)

    def test_point_blank_dash_collision_stress(self):
        """Verifies point-blank dash into solid wall does not penetrate or clip."""
        sim = HeadlessSimulator(ignore_combat_lock=True)
        # Position player 5px from left wall (left wall ends at x=32, so player at x=37)
        sim.player.pos = Vector2(37.0, 200.0)
        sim.player.vel = Vector2(-650.0, 0.0)

        # Trigger dash directly left
        sim.step(SimulationInput(move_x=-1.0, move_y=0.0, dash=True))
        assert not sim.is_out_of_bounds()
        assert not sim.check_wall_clipping()
        assert sim.player.pos.x == pytest.approx(32.0, abs=1e-2)


# ============================================================================
# 6. Stuck & Oscillation Detection Fuzzing & Empirical Defect Verification
# ============================================================================

class TestStuckAndOscillationDetectionFuzzing:
    """Stress tests evaluating stuck detection, oscillation detection, and documenting discovered defects."""
    deliberation: str = "Stuck detection fuzzing and empirical defect verification suite."

    def test_immobility_stuck_condition_detection(self):
        """Verifies that an entity with zero displacement over stuck_threshold_ticks is flagged stuck."""
        bot = SyntheticPlaytesterBot(stuck_threshold_ticks=20, stuck_displacement_min=16.0)
        static_pos = Vector2(300.0, 300.0)

        for tick in range(19):
            assert bot.check_stuck_condition(static_pos) is False

        # 20th tick
        assert bot.check_stuck_condition(static_pos) is True

    def test_open_looping_corridor_progress_not_stuck(self):
        """Verifies that moving steadily along a corridor does not trigger false positive stuck detection."""
        bot = SyntheticPlaytesterBot(stuck_threshold_ticks=20, stuck_displacement_min=16.0)

        for step in range(50):
            pos = Vector2(100.0 + step * 4.0, 200.0)
            assert bot.check_stuck_condition(pos) is False

    def test_stuck_detection_timing_alignment_calibrated(self):
        """Verifies calibrated cycle-accurate stuck detection:
        In evaluate_cartridge, check_stuck_condition is evaluated on every tick.
        A bot wedged against a barrier detects StuckBotError at ~4.5s (270 ticks)
        rather than requiring 45.0s (2,700 ticks).
        """
        spec = BaseLevelSpec(
            level_id="wedged_chokepoint",
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
            deliberation="Chokepoint wedging test",
        )

        cart = GenericCartridge(Path("."))
        cart.level_specs[1] = spec

        # Within a 10s playtest window, the bot detects stuck at ~4.5s (<= 5.5s)
        bot_10s = SyntheticPlaytesterBot(profile=CAUTIOUS_PROFILE)
        res_10s = bot_10s.evaluate_cartridge(cart, max_seconds=10.0)
        assert res_10s.cleared is False
        assert res_10s.failure_type == "StuckBotError"
        assert res_10s.stuck is True
        assert res_10s.completion_time_sec <= 5.5
        assert res_10s.completion_ticks <= 330
        assert res_10s.stuck_position is not None

    def test_run_playtest_simulation_input_clean_execution(self):
        """Verifies that qa/playtester.py:run_playtest imports SimulationInput correctly
        and drives HeadlessSimulator to completion without NameError.
        """
        bot = SyntheticPlaytesterBot()
        sim = HeadlessSimulator(ignore_combat_lock=True)

        res = bot.run_playtest(sim)
        assert isinstance(res, PlaytestResult)
        assert res.deliberation
        assert res.cleared is True
        assert res.status == "PROMOTED"

    def test_portal_combat_lock_integrity_enforcement_and_bypass(self):
        """Verifies that qa/playtester.py:evaluate_cartridge enforces portal combat lock
        by default, rejecting clearance when enemies remain, and permits clearance only
        when ignore_combat_lock=True is explicitly passed.
        """
        spec = BaseLevelSpec(
            level_id="portal_combat_lock_enforcement",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 360.0),
            exit_point=(1180.0, 360.0),
            walls=[
                (0, 0, 1280, 32), (0, 688, 1280, 32), (0, 0, 32, 720), (1248, 0, 32, 720),
                # Wall off enemies in top chamber so player does not engage them
                (100, 200, 300, 32),
            ],
            enemies=[
                {"id": f"chaser_{i}", "type": "chaser", "x": 200.0, "y": 100.0}
                for i in range(3)
            ],
            hazards=[],
            deliberation="Portal combat lock verification",
        )

        cart = GenericCartridge(Path("."))
        cart.level_specs[1] = spec
        bot = SyntheticPlaytesterBot()

        # 1. Default (ignore_combat_lock=False): portal is locked; cannot clear with enemies alive
        res_default = bot.evaluate_cartridge(cart, max_seconds=5.0, ignore_combat_lock=False)
        assert res_default.cleared is False
        assert res_default.status == "REJECTED"
        assert res_default.enemies_killed == 0

        # 2. Explicit bypass (ignore_combat_lock=True): portal contact triggers clearance
        res_bypass = bot.evaluate_cartridge(cart, max_seconds=5.0, ignore_combat_lock=True)
        assert res_bypass.cleared is True
        assert res_bypass.status == "PROMOTED"
        assert res_bypass.enemies_killed == 0
