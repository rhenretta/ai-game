"""tests/test_navmesh_traversability.py
Authoritative unit and integration test suite for Gate B NavMesh Traversability
and Gate C Survivability Bounds (Features 24, 25, 26).
Verifies rasterized grid NavMesh generation, A* pathfinding, Dual-Frontier BFS bottleneck extraction,
spawn clearance margins, and premature player death rejection.
Strictly 0 occurrences of forbidden legacy branding tokens.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple
import pytest

from engine.api.contracts import BaseLevelSpec
from engine.spatial.navmesh import NavMesh, astar_pathfind
from engine.supervisor.models import GateFailureType, QualityGateResult
from engine.supervisor.supervisor import BuildSupervisor


class MockLevelSpec(BaseLevelSpec):
    deliberation: str = "Mock level spec for traversability and survivability verification."
    level_id: str = "test_chamber"
    width: float = 1280.0
    height: float = 720.0
    spawn_point: Tuple[float, float] = (120.0, 360.0)
    exit_point: Tuple[float, float] = (1160.0, 360.0)
    walls: List[Tuple[float, float, float, float]] = []
    hazards: List[Dict[str, Any]] = []
    enemies: List[Dict[str, Any]] = []


class MockCartridge:
    deliberation: str = "Mock cartridge container for unit testing."
    def get_level_spec(self, episode: int = 1) -> MockLevelSpec:
        return MockLevelSpec()


class TestNavMeshExtraction:
    """Verifies rasterization of level walls into 2D grid NavMesh."""
    deliberation: str = "Verifying rasterization from continuous level spec coordinates to grid tiles."

    def test_empty_level_navmesh_is_all_walkable(self):
        spec = MockLevelSpec()
        tile_size = 32.0
        cols = int(spec.width // tile_size)
        rows = int(spec.height // tile_size)
        grid = [[0 for _ in range(cols)] for _ in range(rows)]
        navmesh = NavMesh(grid, tile_size=tile_size)

        assert navmesh.width == cols
        assert navmesh.height == rows
        assert navmesh.is_walkable(0, 0) is True
        assert navmesh.is_walkable(cols - 1, rows - 1) is True

    def test_wall_rasterization_marks_tiles_solid(self):
        spec = MockLevelSpec()
        tile_size = 32.0
        cols = int(spec.width // tile_size)
        rows = int(spec.height // tile_size)
        grid = [[0 for _ in range(cols)] for _ in range(rows)]

        # Add a wall at (640, 0) with width 32, height 720
        spec.walls = [(640.0, 0.0, 32.0, 720.0)]
        for wx, wy, ww, wh in spec.walls:
            min_c = max(0, int(wx // tile_size))
            max_c = min(cols - 1, int((wx + ww - 1e-4) // tile_size))
            min_r = max(0, int(wy // tile_size))
            max_r = min(rows - 1, int((wy + wh - 1e-4) // tile_size))
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    grid[r][c] = 1

        navmesh = NavMesh(grid, tile_size=tile_size)
        wall_col = int(640.0 // tile_size)
        for r in range(rows):
            assert navmesh.is_walkable(wall_col, r) is False
        assert navmesh.is_walkable(wall_col - 1, 10) is True

    def test_coordinate_conversion(self):
        grid = [[0 for _ in range(10)] for _ in range(10)]
        navmesh = NavMesh(grid, tile_size=32.0)

        tx, ty = navmesh.world_to_tile(48.0, 80.0)
        assert tx == 1
        assert ty == 2

        wx, wy = navmesh.tile_to_world(1, 2)
        assert wx == 48.0
        assert wy == 80.0


class TestAStarPathfinding:
    """Verifies A* pathfinding behavior across open and obstructed topologies."""
    deliberation: str = "Verifying A* pathfinding algorithm correctness across topologies."

    def test_open_grid_finds_path(self):
        grid = [[0 for _ in range(10)] for _ in range(10)]
        path = astar_pathfind(grid, (0, 0), (9, 9), allow_diagonal=False)
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (9, 9)
        # Manhattan distance is 18, so path length should be 19 nodes
        assert len(path) == 19

    def test_sealed_wall_returns_none(self):
        grid = [[0 for _ in range(10)] for _ in range(10)]
        # Complete vertical divider at column 5
        for r in range(10):
            grid[r][5] = 1

        path = astar_pathfind(grid, (1, 5), (8, 5), allow_diagonal=True)
        assert path is None

    def test_corridor_with_gap_finds_detour(self):
        grid = [[0 for _ in range(10)] for _ in range(10)]
        # Wall at col 5, but row 2 is open
        for r in range(10):
            if r != 2:
                grid[r][5] = 1

        path = astar_pathfind(grid, (1, 5), (8, 5), allow_diagonal=False)
        assert path is not None
        assert (5, 2) in path


class TestSpawnAndExitObstructions:
    """Verifies Gate B rejection when spawn or exit overlap solid obstacles."""
    deliberation: str = "Verifying Gate B rejection on obstructed spawn or exit portals."

    def test_spawn_inside_obstacle_rejected(self):
        spec = MockLevelSpec()
        spec.walls = [(100.0, 340.0, 64.0, 64.0)]  # overlaps spawn (120, 360)
        supervisor = BuildSupervisor(tile_size=32.0)
        gate_res, blocked = supervisor.verify_gate_b_traversability(MockCartridge(), spec)

        assert gate_res.passed is False
        assert gate_res.failure_type == GateFailureType.CRITICAL_PATH_BLOCKED
        assert "spawn" in gate_res.details.lower()
        assert len(blocked) > 0

    def test_exit_inside_obstacle_rejected(self):
        spec = MockLevelSpec()
        spec.walls = [(1140.0, 340.0, 64.0, 64.0)]  # overlaps exit (1160, 360)
        supervisor = BuildSupervisor(tile_size=32.0)
        gate_res, blocked = supervisor.verify_gate_b_traversability(MockCartridge(), spec)

        assert gate_res.passed is False
        assert gate_res.failure_type == GateFailureType.CRITICAL_PATH_BLOCKED
        assert "exit" in gate_res.details.lower()
        assert len(blocked) > 0


class TestDualFrontierBFSBottleneckExtraction:
    """Verifies Dual-Frontier BFS pinpointing the obstacle bottleneck partition."""
    deliberation: str = "Verifying Dual-Frontier BFS identifying critical obstacle partition chokepoints."

    def test_bottleneck_extraction_finds_partition_wall(self):
        spec = MockLevelSpec()
        # Vertical wall dividing the arena at col 20 (x=640)
        spec.walls = [(640.0, 0.0, 32.0, 720.0)]
        supervisor = BuildSupervisor(tile_size=32.0)
        gate_res, blocked = supervisor.verify_gate_b_traversability(MockCartridge(), spec)

        assert gate_res.passed is False
        assert gate_res.failure_type == GateFailureType.CRITICAL_PATH_BLOCKED
        assert len(blocked) > 0
        # The blocking node column should be tile 20 (640 // 32)
        for bx, by in blocked:
            assert bx == 20

    def test_direct_find_bottleneck_chokepoints(self):
        grid = [[0 for _ in range(15)] for _ in range(10)]
        # Vertical divider at col 7
        for r in range(10):
            grid[r][7] = 1

        supervisor = BuildSupervisor(tile_size=32.0)
        bottlenecks = supervisor.find_bottleneck_chokepoints(grid, (2, 5), (12, 5))
        assert len(bottlenecks) > 0
        for bx, by in bottlenecks:
            assert bx == 7


class TestGateCSurvivabilityAndSpawnMargin:
    """Verifies Gate C spawn clearance margin and premature death thresholds."""
    deliberation: str = "Verifying Gate C survivability verification against hazards and rapid death."

    def test_safe_spawn_clearance_passes(self):
        spec = MockLevelSpec()
        # Hazard well away from spawn (120, 360)
        spec.hazards = [{"id": "spike_far", "x": 500.0, "y": 360.0, "w": 48.0, "h": 48.0}]
        supervisor = BuildSupervisor()
        gate_res = supervisor.verify_gate_c_survivability(None, spec)
        assert gate_res.passed is True
        assert gate_res.failure_type == GateFailureType.NONE

    def test_hazard_overlapping_spawn_margin_rejected(self):
        spec = MockLevelSpec()
        # Hazard at (130, 360), overlapping 64px spawn margin
        spec.hazards = [{"id": "lethal_acid", "x": 130.0, "y": 360.0, "w": 48.0, "h": 48.0}]
        supervisor = BuildSupervisor(spawn_clearance_margin=64.0)
        gate_res = supervisor.verify_gate_c_survivability(None, spec)
        assert gate_res.passed is False
        assert gate_res.failure_type == GateFailureType.SPAWN_HAZARD_OVERLAP
        assert "lethal_acid" in gate_res.details

    def test_premature_player_death_rejected(self):
        spec = MockLevelSpec()
        supervisor = BuildSupervisor()
        # Simulation where player died at tick 60 (< 180 ticks / 3.0s threshold)
        sim_result = {
            "cleared": False,
            "outcome": "PLAYER_DEATH",
            "completion_ticks": 60,
            "first_death_tick": 60,
            "damage_taken": 100.0,
            "failure_coordinates": (140.0, 360.0),
        }
        gate_res = supervisor.verify_gate_c_survivability(sim_result, spec)
        assert gate_res.passed is False
        assert gate_res.failure_type == GateFailureType.PLAYER_DEATH
        assert "died prematurely" in gate_res.deliberation.lower()


class TestEpistemicRationaleFirstContracts:
    """Verifies Rationale-First field ordering on models returned by Gate B & C."""
    deliberation: str = "Verifying deliberation field serialization index 0 on QualityGateResult."

    def test_quality_gate_result_deliberation_first(self):
        res = QualityGateResult(
            deliberation="Verified Gate B path traversability.",
            gate_name="Gate B: Traversability",
            passed=True,
            failure_type=GateFailureType.NONE,
        )
        assert list(res.to_dict().keys())[0] == "deliberation"
