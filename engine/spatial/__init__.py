"""engine/spatial package - 2D math, spatial partitioning, collisions, and NavMesh."""
from engine.spatial.aabb import AABB, Circle, Vector2, Vector2D
from engine.spatial.collision import move_and_resolve_walls
from engine.spatial.grid import SpatialHashGrid
from engine.spatial.navmesh import NavMesh, astar_pathfind

__all__ = [
    "Vector2",
    "Vector2D",
    "AABB",
    "Circle",
    "SpatialHashGrid",
    "move_and_resolve_walls",
    "NavMesh",
    "astar_pathfind",
]
