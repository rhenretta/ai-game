"""engine/spatial/navmesh.py
2D Occupancy Grid NavMesh and A* Pathfinding for spatial reasoning and playtesting.
"""
from __future__ import annotations
import heapq
import math
from typing import Dict, List, Optional, Set, Tuple
from engine.spatial.aabb import AABB, Vector2


class Node:
    """A* graph node for spatial navigation."""

    def __init__(self, x: int, y: int, cost: float, heuristic: float, parent: Optional[Node] = None):
        self.x = x
        self.y = y
        self.cost = cost
        self.heuristic = heuristic
        self.total = cost + heuristic
        self.parent = parent

    def __lt__(self, other: Node) -> bool:
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
        dx = abs(x - gx)
        dy = abs(y - gy)
        return math.hypot(dx, dy) if allow_diagonal else float(dx + dy)

    open_set: List[Node] = []
    start_node = Node(sx, sy, 0.0, heuristic(sx, sy))
    heapq.heappush(open_set, start_node)

    closed_set: Set[Tuple[int, int]] = set()
    best_cost: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if allow_diagonal:
        directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

    while open_set:
        current = heapq.heappop(open_set)
        curr_pos = (current.x, current.y)

        if curr_pos == goal:
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
                if dx != 0 and dy != 0:
                    # Diagonal step: forbid cutting across corners of adjacent solid obstacles
                    # Both orthogonal neighbors (current.x + dx, current.y) and (current.x, current.y + dy) must be free
                    if grid[current.y][current.x + dx] != 0 or grid[current.y + dy][current.x] != 0:
                        continue
                step_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                new_cost = current.cost + step_cost
                if (nx, ny) not in best_cost or new_cost < best_cost[(nx, ny)]:
                    best_cost[(nx, ny)] = new_cost
                    neighbor = Node(nx, ny, new_cost, heuristic(nx, ny), parent=current)
                    heapq.heappush(open_set, neighbor)

    return None


class NavMesh:
    """Manages occupancy grid, world-tile conversions, and clearance validation."""

    def __init__(self, grid: List[List[int]], tile_size: float = 32.0):
        self.grid = [row[:] for row in grid]
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        self.tile_size = float(tile_size)

    def is_walkable(self, tx: int, ty: int) -> bool:
        if 0 <= tx < self.width and 0 <= ty < self.height:
            return self.grid[ty][tx] == 0
        return False

    def world_to_tile(self, x: float, y: float) -> Tuple[int, int]:
        return int(x // self.tile_size), int(y // self.tile_size)

    def tile_to_world(self, tx: int, ty: int) -> Tuple[float, float]:
        return (tx + 0.5) * self.tile_size, (ty + 0.5) * self.tile_size

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], allow_diagonal: bool = False) -> Optional[List[Tuple[int, int]]]:
        return astar_pathfind(self.grid, start, goal, allow_diagonal)

    def check_clearance(self, aabb: AABB) -> bool:
        """Returns True if aabb does not intersect any solid tile."""
        min_tx = max(0, int(aabb.left // self.tile_size))
        max_tx = min(self.width - 1, int(aabb.right // self.tile_size))
        min_ty = max(0, int(aabb.top // self.tile_size))
        max_ty = min(self.height - 1, int(aabb.bottom // self.tile_size))

        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                if self.grid[ty][tx] == 1:
                    tile_box = AABB(tx * self.tile_size, ty * self.tile_size, self.tile_size, self.tile_size)
                    if aabb.intersects(tile_box):
                        return False
        return True
