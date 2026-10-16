"""engine/spatial/grid.py
Uniform Spatial Hash Grid (64x64) for accelerated broadphase spatial queries.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Set, Tuple
from engine.spatial.aabb import AABB, Vector2


class SpatialHashGrid:
    """Partitions 2D space into uniform grid buckets for fast candidate query."""

    def __init__(self, cell_size: float = 64.0):
        self.cell_size = float(cell_size)
        self.grid: Dict[Tuple[int, int], List[Any]] = {}

    def clear(self) -> None:
        self.grid.clear()

    def _hash_coord(self, x: float, y: float) -> Tuple[int, int]:
        return int(math_floor(x / self.cell_size)), int(math_floor(y / self.cell_size))

    def insert(self, item: Any, aabb: AABB) -> None:
        min_cx, min_cy = self._hash_coord(aabb.left, aabb.top)
        max_cx, max_cy = self._hash_coord(aabb.right, aabb.bottom)
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                self.grid.setdefault((cx, cy), []).append(item)

    def query(self, aabb: AABB) -> Set[Any]:
        min_cx, min_cy = self._hash_coord(aabb.left, aabb.top)
        max_cx, max_cy = self._hash_coord(aabb.right, aabb.bottom)
        results: Set[Any] = set()
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                cell = self.grid.get((cx, cy))
                if cell:
                    results.update(cell)
        return results

    def query_radius(self, center: Vector2, radius: float) -> Set[Any]:
        query_box = AABB(center.x - radius, center.y - radius, radius * 2.0, radius * 2.0)
        return self.query(query_box)


def math_floor(val: float) -> int:
    """Returns mathematical floor of val as integer."""
    return math.floor(val)
