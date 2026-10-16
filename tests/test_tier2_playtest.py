"""Tier 2 Headless Playtest & Spatial Simulation Tests.

Verifies:
  1. Headless engine simulation determinism (fixed dt=1/60s).
  2. A* pathfinding correctness, obstacle avoidance, and blocked-path detection.
  3. Collision bounds resolution and wall-clipping prevention.
  4. Out-of-bounds detection across map boundaries.
  5. Synthetic Playtesting critical-path traversability verification.
"""

import heapq
import math
import time
import pytest
from typing import List, Tuple, Dict, Optional, Set


# ============================================================================
# Reference Engine Simulation & Navigation Components
# ============================================================================

class Node:
    """A* graph node for spatial navigation."""
    def __init__(self, x: int, y: int, cost: float, heuristic: float, parent: Optional["Node"] = None):
        self.x = x
        self.y = y
        self.cost = cost
        self.heuristic = heuristic
        self.total = cost + heuristic
        self.parent = parent

    def __lt__(self, other: "Node") -> bool:
        return self.total < other.total


def astar_pathfind(
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    allow_diagonal: bool = False
) -> Optional[List[Tuple[int, int]]]:
    """Authoritative A* pathfinding algorithm over a 2D occupancy grid.
    
    0 = traversable
    1 = solid obstacle
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    sx, sy = start
    gx, gy = goal

    if not (0 <= sx < width and 0 <= sy < height):
        return None
    if not (0 <= gx < width and 0 <= gy < height):
        return None
    if grid[sy][sx] == 1 or grid[gy][gx] == 1:
        return None
    if start == goal:
        return [start]

    def heuristic(x: int, y: int) -> float:
        # Manhattan or Euclidean distance
        dx = abs(x - gx)
        dy = abs(y - gy)
        return math.sqrt(dx * dx + dy * dy) if allow_diagonal else (dx + dy)

    open_set: List[Node] = []
    start_node = Node(sx, sy, 0.0, heuristic(sx, sy))
    heapq.heappush(open_set, start_node)
    
    closed_set: Set[Tuple[int, int]] = set()
    best_cost: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}

    # Movement vectors
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if allow_diagonal:
        directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

    while open_set:
        current = heapq.heappop(open_set)
        curr_pos = (current.x, current.y)

        if curr_pos == goal:
            # Reconstruct path
            path = []
            node: Optional[Node] = current
            while node:
                path.append((node.x, node.y))
                node = node.parent
            path.reverse()
            return path

        if curr_pos in closed_set:
            continue
        closed_set.add(curr_pos)

        for dx, dy in directions:
            nx, ny = current.x + dx, current.y + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == 0:
                step_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                new_cost = current.cost + step_cost
                if (nx, ny) not in best_cost or new_cost < best_cost[(nx, ny)]:
                    best_cost[(nx, ny)] = new_cost
                    neighbor = Node(nx, ny, new_cost, heuristic(nx, ny), parent=current)
                    heapq.heappush(open_set, neighbor)

    return None  # No traversable path exists


class HeadlessSimulator:
    """Discrete fixed-timestep physics & state simulation (dt=1/60s)."""
    def __init__(self, map_width: float = 1280.0, map_height: float = 720.0, dt: float = 1.0 / 60.0):
        self.map_width = map_width
        self.map_height = map_height
        self.dt = dt
        self.step_count = 0
        
        # Player state
        self.player_x = 100.0
        self.player_y = 100.0
        self.player_vx = 0.0
        self.player_vy = 0.0
        self.player_width = 32.0
        self.player_height = 32.0
        self.player_health = 100
        
        # Solid obstacles: list of (x, y, width, height)
        self.obstacles: List[Tuple[float, float, float, float]] = []

    def add_obstacle(self, x: float, y: float, w: float, h: float):
        self.obstacles.append((x, y, w, h))

    def step(self, input_ax: float, input_ay: float, is_dash: bool = False):
        """Simulate one discrete fixed-time tick."""
        speed = 350.0 if is_dash else 180.0
        self.player_vx = input_ax * speed
        self.player_vy = input_ay * speed
        
        # Predict tentative position
        new_x = self.player_x + self.player_vx * self.dt
        new_y = self.player_y + self.player_vy * self.dt
        
        # Resolve horizontal collision
        for ox, oy, ow, oh in self.obstacles:
            if not (new_x + self.player_width <= ox or new_x >= ox + ow or
                    self.player_y + self.player_height <= oy or self.player_y >= oy + oh):
                if self.player_vx > 0:
                    new_x = ox - self.player_width
                elif self.player_vx < 0:
                    new_x = ox + ow
                self.player_vx = 0.0

        # Resolve vertical collision
        for ox, oy, ow, oh in self.obstacles:
            if not (new_x + self.player_width <= ox or new_x >= ox + ow or
                    new_y + self.player_height <= oy or new_y >= oy + oh):
                if self.player_vy > 0:
                    new_y = oy - self.player_height
                elif self.player_vy < 0:
                    new_y = oy + oh
                self.player_vy = 0.0

        self.player_x = new_x
        self.player_y = new_y
        self.step_count += 1

    def is_out_of_bounds(self) -> bool:
        return (
            self.player_x < 0.0 or
            self.player_x + self.player_width > self.map_width or
            self.player_y < 0.0 or
            self.player_y + self.player_height > self.map_height
        )


class SyntheticPlaytester:
    """Headless automated playtesting bot verifying collision and critical-path traversability."""
    def __init__(self, tile_size: float = 32.0):
        self.tile_size = tile_size

    def evaluate_level(
        self,
        grid: List[List[int]],
        spawn_tile: Tuple[int, int],
        exit_tile: Tuple[int, int]
    ) -> Dict[str, any]:
        """Runs A* pathfinding and traversability verification."""
        path = astar_pathfind(grid, spawn_tile, exit_tile)
        if path is None:
            return {
                "is_traversable": False,
                "status": "REJECTED",
                "failure_type": "CriticalPathBlockedError",
                "spawn": spawn_tile,
                "exit": exit_tile,
                "path_length": 0,
                "estimated_clear_time_sec": None
            }
        
        # Calculate time to clear based on nominal speed of 180 px/s
        distance_px = len(path) * self.tile_size
        clear_time = distance_px / 180.0
        
        return {
            "is_traversable": True,
            "status": "PROMOTED",
            "failure_type": None,
            "spawn": spawn_tile,
            "exit": exit_tile,
            "path_length": len(path),
            "estimated_clear_time_sec": clear_time,
            "path": path
        }


# ============================================================================
# Test Cases
# ============================================================================

class TestHeadlessSimulationEngine:
    """Verifies fixed-timestep simulation determinism and throughput."""

    def test_simulation_determinism(self):
        """Two identical simulators receiving identical inputs produce bit-exact states."""
        sim1 = HeadlessSimulator(dt=1.0/60.0)
        sim2 = HeadlessSimulator(dt=1.0/60.0)
        
        # Add identical obstacle
        sim1.add_obstacle(200.0, 50.0, 40.0, 200.0)
        sim2.add_obstacle(200.0, 50.0, 40.0, 200.0)
        
        # Execute 120 steps with varied directional inputs
        for step in range(120):
            ax = 1.0 if step < 60 else -0.5
            ay = 0.5 if step % 2 == 0 else -0.5
            dash = (step % 15 == 0)
            sim1.step(ax, ay, is_dash=dash)
            sim2.step(ax, ay, is_dash=dash)
            
        assert sim1.player_x == sim2.player_x
        assert sim1.player_y == sim2.player_y
        assert sim1.player_vx == sim2.player_vx
        assert sim1.player_vy == sim2.player_vy
        assert sim1.step_count == 120

    def test_simulation_high_throughput_benchmark(self):
        """Simulation must execute at least 500 steps/sec in headless mode."""
        sim = HeadlessSimulator()
        start_time = time.perf_counter()
        
        target_steps = 1000
        for i in range(target_steps):
            sim.step(1.0, 0.0)
            
        elapsed = time.perf_counter() - start_time
        steps_per_sec = target_steps / elapsed
        assert steps_per_sec >= 500.0, f"Headless sim too slow: {steps_per_sec:.1f} steps/s"


class TestAStarPathfinding:
    """Verifies A* spatial navigation algorithm across obstacle layouts."""

    def test_astar_straight_clear_path(self):
        """A* discovers straight horizontal path on an empty grid."""
        grid = [[0 for _ in range(10)] for _ in range(5)]
        start = (1, 2)
        goal = (8, 2)
        
        path = astar_pathfind(grid, start, goal)
        assert path is not None
        assert path[0] == start
        assert path[-1] == goal
        assert len(path) == 8  # 8 steps: (1,2) to (8,2)

    def test_astar_obstacle_avoidance_corner(self):
        """A* navigates around an obstacle wall without penetrating solid nodes."""
        grid = [[0 for _ in range(10)] for _ in range(10)]
        # Place vertical wall at x=4 from y=1 to y=8, leaving openings at y=0 and y=9
        for y in range(1, 9):
            grid[y][4] = 1
            
        start = (2, 5)
        goal = (6, 5)
        path = astar_pathfind(grid, start, goal)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == goal
        
        # Verify no node in path crosses a solid wall
        for px, py in path:
            assert grid[py][px] == 0, f"Path stepped on solid obstacle at ({px}, {py})"

    def test_astar_complex_16x16_maze(self, sample_level_grid_16x16):
        """A* finds traversable critical path through 16x16 maze."""
        data = sample_level_grid_16x16
        path = astar_pathfind(data["grid"], data["spawn"], data["exit"])
        
        assert path is not None
        assert path[0] == data["spawn"]
        assert path[-1] == data["exit"]
        assert len(path) > 20  # Winding path

    def test_astar_unreachable_destination(self, blocked_level_grid_16x16):
        """A* returns None when exit is completely enclosed by solid walls."""
        data = blocked_level_grid_16x16
        path = astar_pathfind(data["grid"], data["spawn"], data["exit"])
        assert path is None, "A* should return None for blocked destination"

    def test_astar_start_equals_goal(self):
        """A* handles trivial start == goal case."""
        grid = [[0, 0], [0, 0]]
        path = astar_pathfind(grid, (1, 1), (1, 1))
        assert path == [(1, 1)]


class TestCollisionBoundsAndWallClipping:
    """Verifies bounding box collision resolution and clipping prevention."""

    def test_wall_collision_stops_movement(self):
        """Player moving toward a wall stops at the wall surface."""
        sim = HeadlessSimulator()
        sim.player_x = 50.0
        sim.player_y = 100.0
        # Place solid wall at x=100 with width 40
        sim.add_obstacle(100.0, 50.0, 40.0, 200.0)
        
        # Step forward horizontally
        for _ in range(30):
            sim.step(1.0, 0.0)
            
        # Player width is 32, so right edge cannot exceed wall left edge (100)
        assert sim.player_x + sim.player_width <= 100.0
        assert sim.player_x == pytest.approx(100.0 - 32.0, abs=1e-3)
        assert sim.player_vx == 0.0

    def test_high_velocity_clipping_guard(self):
        """Extreme velocity dash must not clip through solid geometry."""
        sim = HeadlessSimulator()
        sim.player_x = 50.0
        sim.player_y = 100.0
        sim.add_obstacle(100.0, 50.0, 50.0, 200.0)
        
        # Massive dash velocity (350 px/s with is_dash=True)
        for _ in range(50):
            sim.step(1.0, 0.0, is_dash=True)
            
        assert sim.player_x + sim.player_width <= 100.0

    def test_wall_sliding_resolution(self):
        """Diagonal movement against a vertical wall slides along Y axis."""
        sim = HeadlessSimulator()
        sim.player_x = 68.0  # Right at the wall boundary (x=100)
        sim.player_y = 100.0
        sim.add_obstacle(100.0, 0.0, 40.0, 400.0)
        
        initial_y = sim.player_y
        # Move diagonally (into wall in X, downward in Y)
        sim.step(1.0, 1.0)
        
        # X is clamped at wall
        assert sim.player_x + sim.player_width <= 100.0
        # Y advanced downward
        assert sim.player_y > initial_y


class TestOutOfBoundsDetection:
    """Verifies detection of entities exiting level coordinate boundaries."""

    def test_in_bounds_normal_state(self):
        sim = HeadlessSimulator(map_width=800, map_height=600)
        sim.player_x = 400
        sim.player_y = 300
        assert sim.is_out_of_bounds() is False

    def test_out_of_bounds_negative_x(self):
        sim = HeadlessSimulator(map_width=800, map_height=600)
        sim.player_x = -5.0
        assert sim.is_out_of_bounds() is True

    def test_out_of_bounds_excessive_y(self):
        sim = HeadlessSimulator(map_width=800, map_height=600)
        sim.player_y = 601.0
        assert sim.is_out_of_bounds() is True


class TestSyntheticPlaytesterTraversability:
    """Verifies automated playtester QA guardrail on valid vs blocked levels."""

    def test_playtester_promotes_traversable_level(self, sample_level_grid_16x16):
        """Playtester verifies clear critical path and estimates clear time."""
        playtester = SyntheticPlaytester(tile_size=32.0)
        data = sample_level_grid_16x16
        report = playtester.evaluate_level(data["grid"], data["spawn"], data["exit"])
        
        assert report["is_traversable"] is True
        assert report["status"] == "PROMOTED"
        assert report["failure_type"] is None
        assert report["estimated_clear_time_sec"] > 0.0
        assert len(report["path"]) > 0

    def test_playtester_rejects_blocked_level(self, blocked_level_grid_16x16):
        """Playtester flags blocked level with CriticalPathBlockedError."""
        playtester = SyntheticPlaytester(tile_size=32.0)
        data = blocked_level_grid_16x16
        report = playtester.evaluate_level(data["grid"], data["spawn"], data["exit"])
        
        assert report["is_traversable"] is False
        assert report["status"] == "REJECTED"
        assert report["failure_type"] == "CriticalPathBlockedError"
        assert report["path_length"] == 0
