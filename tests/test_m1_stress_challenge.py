"""tests/test_m1_stress_challenge.py
Milestone 1 Empirical Stress Tests:
1. High-velocity swept collision and tunneling prevention (v > 2000 px/s, large dt).
2. SpatialHashGrid scaling (1000+ entities inserted and queried in < 5ms) and coordinate math.
3. Complex maze A* pathfinding correctness and zero diagonal wall cuts.
4. Wall sliding momentum along orthogonal and 45-degree angled / stepped wall collisions.
"""

import math
import random
import time
import pytest
from typing import List, Tuple

from engine.spatial.aabb import AABB, Vector2
from engine.spatial.collision import move_and_resolve_walls
from engine.spatial.grid import SpatialHashGrid, math_floor
from engine.spatial.navmesh import NavMesh, astar_pathfind
from engine.entities.player import Player


# ============================================================================
# 1. High-Velocity Tunneling Tests
# ============================================================================

class TestHighVelocityTunneling:
    """Stress-tests swept bounding box collision resolution against extreme velocities and dt."""

    @pytest.mark.parametrize("speed", [2000.0, 5000.0, 10000.0])
    @pytest.mark.parametrize("dt", [0.1, 0.25, 0.5])
    def test_orthogonal_tunneling_prevention_x_axis(self, speed: float, dt: float):
        """Orthogonal high-speed movement along +X and -X must clamp at wall surface without tunneling."""
        wall = AABB(300.0, 0.0, 32.0, 500.0)

        # Moving +X towards wall from x=50
        entity_forward = AABB(50.0, 100.0, 32.0, 32.0)
        vel_fwd = Vector2(speed, 0.0)
        new_pos_fwd, actual_vel_fwd = move_and_resolve_walls(entity_forward, vel_fwd, dt, [wall])
        assert new_pos_fwd.x + entity_forward.width <= wall.left, (
            f"Tunneled past wall: new_x={new_pos_fwd.x}, wall_left={wall.left}"
        )
        if entity_forward.x + speed * dt >= wall.left - entity_forward.width:
            assert pytest.approx(new_pos_fwd.x, abs=1e-3) == wall.left - entity_forward.width
        assert actual_vel_fwd.x >= 0.0

        # Moving -X towards wall from x=600
        entity_backward = AABB(600.0, 100.0, 32.0, 32.0)
        vel_bwd = Vector2(-speed, 0.0)
        new_pos_bwd, actual_vel_bwd = move_and_resolve_walls(entity_backward, vel_bwd, dt, [wall])
        assert new_pos_bwd.x >= wall.right, (
            f"Tunneled past wall backwards: new_x={new_pos_bwd.x}, wall_right={wall.right}"
        )
        if entity_backward.x - speed * dt <= wall.right:
            assert pytest.approx(new_pos_bwd.x, abs=1e-3) == wall.right

    @pytest.mark.parametrize("speed", [2000.0, 5000.0, 10000.0])
    @pytest.mark.parametrize("dt", [0.1, 0.25, 0.5])
    def test_orthogonal_tunneling_prevention_y_axis(self, speed: float, dt: float):
        """Orthogonal high-speed movement along +Y and -Y must clamp at wall surface without tunneling."""
        wall = AABB(0.0, 300.0, 500.0, 32.0)

        # Moving +Y towards wall from y=50
        entity_down = AABB(100.0, 50.0, 32.0, 32.0)
        vel_down = Vector2(0.0, speed)
        new_pos_down, actual_vel_down = move_and_resolve_walls(entity_down, vel_down, dt, [wall])
        assert new_pos_down.y + entity_down.height <= wall.top
        if entity_down.y + speed * dt >= wall.top - entity_down.height:
            assert pytest.approx(new_pos_down.y, abs=1e-3) == wall.top - entity_down.height

        # Moving -Y towards wall from y=600
        entity_up = AABB(100.0, 600.0, 32.0, 32.0)
        vel_up = Vector2(0.0, -speed)
        new_pos_up, actual_vel_up = move_and_resolve_walls(entity_up, vel_up, dt, [wall])
        assert new_pos_up.y >= wall.bottom
        if entity_up.y - speed * dt <= wall.bottom:
            assert pytest.approx(new_pos_up.y, abs=1e-3) == wall.bottom

    def test_high_velocity_thin_wall_tunneling(self):
        """High-velocity entity (v=3000 px/s, dt=0.1) must not tunnel through razor-thin wall (width=2px)."""
        thin_wall = AABB(200.0, 0.0, 2.0, 400.0)
        entity = AABB(50.0, 100.0, 32.0, 32.0)
        vel = Vector2(3000.0, 0.0)
        new_pos, _ = move_and_resolve_walls(entity, vel, 0.1, [thin_wall])
        assert new_pos.x + 32.0 <= thin_wall.left, f"Tunneled through 2px wall to x={new_pos.x}"

    def test_diagonal_swept_bounding_box_prevents_tunneling(self):
        """
        Adversarial test: moving diagonally at extreme velocity (v > 2000 px/s) across an isolated
        obstacle directly on the diagonal path must not tunnel through the obstacle.
        """
        # Entity starts at (0, 0), moving diagonally towards (600, 600)
        entity = AABB(0.0, 0.0, 32.0, 32.0)
        obstacle = AABB(200.0, 200.0, 64.0, 64.0)
        vel = Vector2(2400.0, 2400.0)
        dt = 0.25  # Unresolved displacement is (600, 600)

        new_pos, actual_vel = move_and_resolve_walls(entity, vel, dt, [obstacle])

        # If tunneling occurred, entity jumped over obstacle and landed at or near (600, 600)
        tunneled = (new_pos.x >= obstacle.right and new_pos.y >= obstacle.bottom)
        assert not tunneled, (
            f"CRITICAL DEFECT: Entity tunneled through diagonal obstacle! "
            f"Start=(0,0), TargetObstacle=(200,200,64,64), NewPos=({new_pos.x}, {new_pos.y})"
        )


# ============================================================================
# 2. SpatialHashGrid Scaling & Coordinate Tests
# ============================================================================

class TestSpatialHashGridScaling:
    """Stress-tests SpatialHashGrid performance with 1000+ entities and coordinate edge cases."""

    def test_spatial_hash_grid_1000_entities_under_5ms(self):
        """1000+ entities inserted and queried must complete in under 5ms total."""
        grid = SpatialHashGrid(cell_size=64.0)
        random.seed(1337)
        num_entities = 1200
        entities = []
        for i in range(num_entities):
            x = random.uniform(0.0, 1200.0)
            y = random.uniform(0.0, 650.0)
            w = random.uniform(16.0, 48.0)
            h = random.uniform(16.0, 48.0)
            entities.append((f"ent_{i}", AABB(x, y, w, h)))

        # 1. Insertion benchmark
        t_start = time.perf_counter()
        for name, box in entities:
            grid.insert(name, box)
        t_insert = time.perf_counter() - t_start
        insert_ms = t_insert * 1000.0

        # 2. Query benchmark (single player-sized bounding box query)
        player_box = AABB(600.0, 340.0, 32.0, 32.0)
        t_qstart = time.perf_counter()
        results = grid.query(player_box)
        t_query = time.perf_counter() - t_qstart
        query_ms = t_query * 1000.0

        total_ms = insert_ms + query_ms
        assert total_ms < 5.0, (
            f"SpatialHashGrid scaling budget exceeded: {total_ms:.3f}ms (Insert: {insert_ms:.3f}ms, Query: {query_ms:.3f}ms)"
        )
        assert len(results) > 0, "Query should find nearby entities in populated grid"

    def test_spatial_hash_grid_negative_coordinate_math_floor_defect(self):
        """
        Adversarial test: tests math_floor behavior on exact negative integers.
        math.floor(-1.0) must be -1, not -2.
        """
        for val in [-1.0, -2.0, -5.0]:
            mf = math_floor(val)
            expected = math.floor(val)
            assert mf == expected, (
                f"DEFECT in math_floor: math_floor({val}) returned {mf}, expected {expected} "
                f"(causes off-by-one cell indexing on negative boundaries)"
            )


# ============================================================================
# 3. Complex Maze A* Pathfinding & Diagonal Wall Cutting Tests
# ============================================================================

class TestComplexMazeAStarPathfinding:
    """Stress-tests A* pathfinding on complex mazes and verifies zero diagonal wall cuts."""

    def test_complex_winding_maze_optimality_and_soundness(self):
        """A* navigates a complex winding 12x12 maze with dead ends and narrow channels."""
        # 12x12 maze layout
        # 1 = solid, 0 = traversable
        maze = [
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        ]
        start = (0, 0)
        goal = (11, 11)

        path = astar_pathfind(maze, start, goal, allow_diagonal=False)
        assert path is not None, "A* failed to find path in solvable complex maze"
        assert path[0] == start
        assert path[-1] == goal

        # Verify no step lands on solid obstacle
        for x, y in path:
            assert maze[y][x] == 0, f"A* path stepped on solid obstacle at ({x}, {y})"

        # Verify consecutive steps are orthogonal
        for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
            step_dist = abs(x2 - x1) + abs(y2 - y1)
            assert step_dist == 1, f"Non-orthogonal step taken: ({x1},{y1}) -> ({x2},{y2})"

    def test_zero_diagonal_wall_cuts_touching_walls(self):
        """
        Adversarial test: when allow_diagonal=True, A* MUST NOT cut diagonally between
        two diagonally touching solid walls.
        Grid:
          (0, 0)=0   (1, 0)=1 (WALL)
          (0, 1)=1   (1, 1)=0
        Moving from (0, 0) to (1, 1) cuts through the touching wall corners!
        """
        grid = [
            [0, 1],
            [1, 0],
        ]
        path = astar_pathfind(grid, (0, 0), (1, 1), allow_diagonal=True)
        # Squeezing between (1, 0) and (0, 1) must be forbidden!
        # Since no alternate path exists, path should be None.
        assert path is None, (
            f"CRITICAL DEFECT: A* cut diagonally through touching solid wall corners! Path={path}"
        )

    def test_zero_diagonal_wall_cuts_single_corner(self):
        """
        Adversarial test: when allow_diagonal=True, moving diagonally around a single wall corner
        must avoid diagonal cutting across the corner of the solid tile.
        Grid:
          (0,0)=0  (1,0)=1 (WALL)  (2,0)=0
          (0,1)=0  (1,1)=0         (2,1)=0
        """
        grid = [
            [0, 1, 0],
            [0, 0, 0],
        ]
        path = astar_pathfind(grid, (0, 0), (2, 0), allow_diagonal=True)
        assert path is not None

        # Verify no diagonal step cuts through an orthogonal wall neighbor
        for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
            dx = x2 - x1
            dy = y2 - y1
            if dx != 0 and dy != 0:
                # Diagonal step: both orthogonal tiles (x1 + dx, y1) and (x1, y1 + dy) must be free
                ortho1_wall = grid[y1][x1 + dx] == 1
                ortho2_wall = grid[y1 + dy][x1] == 1
                assert not (ortho1_wall or ortho2_wall), (
                    f"CRITICAL DEFECT: A* cut across corner of solid wall! "
                    f"Step ({x1},{y1}) -> ({x2},{y2}), "
                    f"blocked neighbors: ({x1+dx},{y1})={ortho1_wall}, ({x1},{y1+dy})={ortho2_wall}"
                )


# ============================================================================
# 4. Wall Sliding Momentum Tests
# ============================================================================

class TestWallSlidingMomentum:
    """Verifies that forward sliding momentum is preserved along orthogonal and angled wall collisions."""

    def test_orthogonal_vertical_wall_sliding(self):
        """Diagonal movement into a vertical wall maintains full perpendicular (Y) sliding velocity."""
        player_box = AABB(68.0, 100.0, 32.0, 32.0)
        wall = AABB(100.0, 0.0, 32.0, 400.0)

        # 45-degree movement: vel = (200, 200) into wall at x=100
        dt = 0.01667
        new_pos, actual_vel = move_and_resolve_walls(player_box, Vector2(200.0, 200.0), dt, [wall])

        # X is clamped at wall surface
        assert pytest.approx(new_pos.x, abs=1e-3) == 68.0
        assert actual_vel.x == 0.0
        # Y maintains forward sliding velocity
        assert pytest.approx(actual_vel.y, abs=1e-2) == 200.0
        assert new_pos.y > player_box.y

    def test_orthogonal_horizontal_wall_sliding(self):
        """Diagonal movement into a horizontal wall maintains full perpendicular (X) sliding velocity."""
        player_box = AABB(100.0, 68.0, 32.0, 32.0)
        wall = AABB(0.0, 100.0, 400.0, 32.0)

        dt = 0.01667
        new_pos, actual_vel = move_and_resolve_walls(player_box, Vector2(200.0, 200.0), dt, [wall])

        # Y is clamped at wall surface
        assert pytest.approx(new_pos.y, abs=1e-3) == 68.0
        assert actual_vel.y == 0.0
        # X maintains forward sliding velocity
        assert pytest.approx(actual_vel.x, abs=1e-2) == 200.0
        assert new_pos.x > player_box.x

    def test_stepped_45_degree_diagonal_wall_sliding(self):
        """Player sliding along a 45-degree stepped tile wall maintains forward momentum without locking."""
        stepped_tiles = [
            AABB(100.0, 100.0, 32.0, 32.0),
            AABB(132.0, 132.0, 32.0, 32.0),
            AABB(164.0, 164.0, 32.0, 32.0)
        ]
        cur_box = AABB(68.0, 70.0, 32.0, 32.0)
        dt = 0.01667
        # Move southeast along slope
        for step in range(50):
            new_pos, actual_vel = move_and_resolve_walls(cur_box, Vector2(100.0, 100.0), dt, stepped_tiles)
            # Velocity should not drop to 0,0 (player should not get permanently jammed)
            assert actual_vel.length() > 0.0, f"Player jammed at step {step}: pos=({new_pos.x}, {new_pos.y})"
            cur_box = AABB(new_pos.x, new_pos.y, 32.0, 32.0)

        # After 50 steps, player made substantial forward progress
        assert cur_box.y > 120.0
