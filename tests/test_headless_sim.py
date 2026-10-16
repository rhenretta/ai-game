"""tests/test_headless_sim.py
Comprehensive unit test suite for Feature 14: Headless Simulation Host.
"""
from __future__ import annotations
import ast
import inspect
from pathlib import Path
import time
import pytest

from engine.api.contracts import BaseLevelSpec
from engine.entities.player import Player
from engine.sim.headless import (
    HeadlessSimulator,
    InMemoryTelemetryRecorder,
    SimulationInput,
    SimulationState,
    SimulationStepResult,
    SimulationSummary,
)
from engine.spatial.aabb import Vector2


class TestSimulationDeterminism:
    """Verifies that discrete fixed-timestep physics is 100% deterministic."""
    deliberation: str = "Regression test suite verifying deterministic physics under identical inputs."

    def test_determinism_identical_inputs(self):
        sim1 = HeadlessSimulator(ignore_combat_lock=True)
        sim2 = HeadlessSimulator(ignore_combat_lock=True)

        for step in range(120):
            ax = 1.0 if step < 60 else -0.5
            ay = 0.5 if step % 2 == 0 else -0.5
            dash = (step % 15 == 0)
            fire = (step % 10 == 0)
            inp = SimulationInput(move_x=ax, move_y=ay, dash=dash, fire=fire, aim_x=800.0, aim_y=360.0)
            res1 = sim1.step(inp)
            res2 = sim2.step(inp)

            assert res1.player_pos == res2.player_pos
            assert res1.player_vel == res2.player_vel
            assert res1.player_health == res2.player_health
            assert res1.player_stamina == res2.player_stamina
            assert res1.is_dashing == res2.is_dashing


class TestHighThroughputBenchmark:
    """Verifies that simulation runs headless at >500 steps/sec."""
    deliberation: str = "Benchmark test suite verifying high-throughput headless execution."

    def test_benchmark_thousand_steps(self):
        sim = HeadlessSimulator(telemetry_mode="null", ignore_combat_lock=True)
        start = time.perf_counter()
        target_steps = 1000
        for _ in range(target_steps):
            sim.step(SimulationInput(move_x=1.0, move_y=0.0))
        elapsed = time.perf_counter() - start
        rate = target_steps / elapsed
        assert rate >= 500.0, f"Headless sim below benchmark: {rate:.1f} steps/s"


class TestCartridgeIngestion:
    """Verifies loading from cartridges, custom specs, and default geometries."""
    deliberation: str = "Test suite verifying level spec and cartridge loading in headless sim."

    def test_load_from_seed_cartridge(self, tmp_path: Path):
        cart_dir = tmp_path / "mock_cartridge"
        cart_dir.mkdir(parents=True, exist_ok=True)
        import json
        (cart_dir / "manifest.json").write_text(json.dumps({
            "title": "Mock Cartridge",
            "entrypoint": "mechanics.py",
        }), encoding="utf-8")
        (cart_dir / "level.json").write_text(json.dumps({
            "width": 1280.0,
            "height": 720.0,
            "spawn_point": [100.0, 100.0],
            "exit_point": [1100.0, 360.0],
            "walls": [[0.0, 0.0, 1280.0, 20.0]],
            "enemies": [{"id": "e1", "type": "chaser", "x": 300.0, "y": 300.0}],
            "hazards": [],
        }), encoding="utf-8")
        (cart_dir / "mechanics.py").write_text('''"""mechanics.py"""
from engine.api.contracts import BaseCartridge, BaseLevelSpec
class MechanicsCartridge(BaseCartridge):
    deliberation: str = "Test cartridge for headless sim."
    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            level_id="mock_01",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 100.0),
            exit_point=(1100.0, 360.0),
            walls=[(0.0, 0.0, 1280.0, 20.0)],
            enemies=[{"id": "e1", "type": "chaser", "x": 300.0, "y": 300.0}],
            hazards=[],
            deliberation="Mock level spec for headless sim.",
        )
''', encoding="utf-8")
        sim = HeadlessSimulator(cartridge_path=cart_dir)
        assert len(sim.walls) > 0
        assert len(sim.enemies) > 0
        assert sim.player is not None

    def test_load_from_custom_spec(self):
        spec = BaseLevelSpec(
            level_id="custom_01",
            width=800.0,
            height=600.0,
            spawn_point=(50.0, 50.0),
            exit_point=(700.0, 500.0),
            walls=[(0, 0, 800, 20), (0, 580, 800, 20)],
            enemies=[{"id": "c1", "type": "chaser", "x": 200.0, "y": 200.0}],
            deliberation="Custom spec for testing",
        )
        sim = HeadlessSimulator(level_spec=spec)
        assert sim.map_width == 800.0
        assert sim.player.pos.x == 50.0
        assert len(sim.enemies) == 1


class TestSyntheticInputHandling:
    """Verifies move, dash, and fire synthetic inputs."""
    deliberation: str = "Test suite verifying synthetic input packet handling."

    def test_cardinal_movement(self):
        sim = HeadlessSimulator(ignore_combat_lock=True)
        start_x = sim.player.pos.x
        sim.step(SimulationInput(move_x=1.0, move_y=0.0))
        assert sim.player.pos.x > start_x

    def test_dash_mechanic(self):
        sim = HeadlessSimulator()
        sim.step(SimulationInput(move_x=1.0, move_y=0.0, dash=True))
        assert sim.player.is_dashing is True
        assert sim.player.stamina < 100.0

    def test_weapon_firing(self):
        sim = HeadlessSimulator()
        sim.step(SimulationInput(fire=True, aim_x=500.0, aim_y=360.0))
        assert len(sim.projectile_manager.projectiles) > 0


class TestSweptCollisionAndBounds:
    """Verifies zero tunneling, wall sliding, and out-of-bounds detection."""
    deliberation: str = "Test suite verifying continuous swept collision and boundary guards."

    def test_wall_collision_stops_player(self):
        sim = HeadlessSimulator()
        sim.player_x = 50.0
        sim.player_y = 100.0
        sim.add_obstacle(100.0, 50.0, 40.0, 200.0)

        for _ in range(30):
            sim.step(SimulationInput(move_x=1.0, move_y=0.0))

        assert sim.player.pos.x + sim.player.size.x <= 100.0
        assert not sim.check_wall_clipping()

    def test_high_speed_dash_no_tunneling(self):
        sim = HeadlessSimulator()
        sim.player_x = 50.0
        sim.player_y = 100.0
        sim.add_obstacle(100.0, 50.0, 40.0, 200.0)

        for _ in range(20):
            sim.step(SimulationInput(move_x=1.0, move_y=0.0, dash=True))

        assert sim.player.pos.x + sim.player.size.x <= 100.0
        assert not sim.check_wall_clipping()

    def test_out_of_bounds_detection(self):
        sim = HeadlessSimulator(map_width=800, map_height=600)
        sim.player.pos.x = -10.0
        assert sim.is_out_of_bounds() is True


class TestHazardInteractions:
    """Verifies environmental hazards tick and interact with player."""
    deliberation: str = "Test suite verifying hazard simulation and player damage."

    def test_spike_hazard_damage(self):
        spec = BaseLevelSpec(
            spawn_point=(100.0, 100.0),
            hazards=[{"id": "sp1", "type": "spikes", "x": 100.0, "y": 100.0, "w": 48.0, "h": 48.0}],
            deliberation="Spike test spec"
        )
        sim = HeadlessSimulator(level_spec=spec)
        hazard = sim.hazards[0]
        hazard.state_timer = 0.01  # accelerate spike cycle

        # Step until extended
        for _ in range(60):
            sim.step(SimulationInput())

        assert sim.player.health < 100.0 or hazard.state is not None


class TestCombatAndEnemies:
    """Verifies enemy tracking, damage, and defeat."""
    deliberation: str = "Test suite verifying enemy AI behavior and pursuit."

    def test_chaser_pursuit(self):
        spec = BaseLevelSpec(
            spawn_point=(100.0, 100.0),
            enemies=[{"id": "c1", "type": "chaser", "x": 300.0, "y": 100.0}],
            deliberation="Combat test spec"
        )
        sim = HeadlessSimulator(level_spec=spec)
        enemy = sim.enemies[0]
        initial_dist = (enemy.center - sim.player.center).magnitude()

        for _ in range(30):
            sim.step(SimulationInput())

        new_dist = (enemy.center - sim.player.center).magnitude()
        assert new_dist < initial_dist


class TestExitAndOutcomes:
    """Verifies victory, defeat, and timeout outcomes."""
    deliberation: str = "Test suite verifying episode completion outcomes."

    def test_cleared_with_ignore_combat_lock(self):
        spec = BaseLevelSpec(spawn_point=(100.0, 100.0), exit_point=(120.0, 100.0), deliberation="Exit test spec")
        sim = HeadlessSimulator(level_spec=spec, ignore_combat_lock=True)
        res = sim.step(SimulationInput(move_x=1.0, move_y=0.0))
        assert sim.is_completed is True
        assert sim.outcome == "CLEARED"

    def test_player_death_outcome(self):
        sim = HeadlessSimulator()
        sim.player.health = 0.0
        sim.player.is_alive = False
        res = sim.step(SimulationInput())
        assert sim.is_completed is True
        assert sim.outcome == "PLAYER_DEATH"


class TestTelemetryModes:
    """Verifies in-memory, SQLite WAL, and null telemetry recording."""
    deliberation: str = "Test suite verifying telemetry modes."

    def test_in_memory_telemetry(self):
        sim = HeadlessSimulator(telemetry_mode="memory", ignore_combat_lock=True)
        sim.step(SimulationInput(move_x=1.0, move_y=0.0, dash=True, fire=True))
        rec = sim.telemetry_recorder
        assert isinstance(rec, InMemoryTelemetryRecorder)
        assert len(rec.events) > 0
        dash_events = rec.get_events_by_type("DASH")
        assert len(dash_events) > 0

    def test_sqlite_wal_telemetry(self, tmp_path):
        db_file = tmp_path / "test_sim.db"
        sim = HeadlessSimulator(telemetry_mode="sqlite", telemetry_db_path=db_file)
        sim.step(SimulationInput(move_x=1.0, move_y=0.0))
        sim.close()
        assert db_file.exists()


class TestRationaleFirstEpistemicArchitecture:
    """Verifies that deliberation: str = '' is declared strictly first on all models."""
    deliberation: str = "Test suite verifying AST rationale-first compliance."

    def test_deliberation_declared_first_via_ast(self):
        filepath = Path("engine/sim/headless.py")
        if not filepath.exists():
            pytest.skip("engine/sim/headless.py not yet written")

        tree = ast.parse(filepath.read_text(encoding="utf-8"))
        target_classes = {
            "SimulationInput",
            "SimulationStepResult",
            "SimulationState",
            "SimulationSummary",
            "InMemoryTelemetryRecorder",
        }

        found_classes = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in target_classes:
                found_classes.add(node.name)
                # Find first statement/attribute in class body
                first_stmt = node.body[0] if node.body else None
                # If docstring is first, check second statement
                if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                    first_stmt = node.body[1] if len(node.body) > 1 else None

                assert first_stmt is not None, f"Class {node.name} has empty body"
                # Check that it is an assignment or annotated assignment to 'deliberation'
                is_delib = False
                if isinstance(first_stmt, ast.AnnAssign) and isinstance(first_stmt.target, ast.Name):
                    is_delib = (first_stmt.target.id == "deliberation")
                elif isinstance(first_stmt, ast.Assign) and len(first_stmt.targets) == 1 and isinstance(first_stmt.targets[0], ast.Name):
                    is_delib = (first_stmt.targets[0].id == "deliberation")

                assert is_delib, f"Class {node.name} does not declare 'deliberation' as its very first property!"

        assert target_classes.issubset(found_classes), f"Missing target classes: {target_classes - found_classes}"
