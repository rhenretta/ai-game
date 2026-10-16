"""engine/spatial/aabb.py
2D spatial primitives: Vector2, AABB, and Circle with robust collision math.
"""
from __future__ import annotations
import math
from typing import Tuple


class Vector2:
    """2D Vector with arithmetic operations, safe normalization, and distance queries."""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalize(self) -> Vector2:
        mag = self.magnitude()
        if mag == 0.0:
            return Vector2(0.0, 0.0)
        return Vector2(self.x / mag, self.y / mag)

    def normalized(self) -> Vector2:
        return self.normalize()

    def dot(self, other: Vector2) -> float:
        return self.x * other.x + self.y * other.y

    def distance_to(self, other: Vector2) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def __add__(self, other: Vector2) -> Vector2:
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2) -> Vector2:
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector2:
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vector2:
        return Vector2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Vector2:
        if scalar == 0.0:
            return Vector2(0.0, 0.0)
        return Vector2(self.x / scalar, self.y / scalar)

    def __repr__(self) -> str:
        return f"Vector2({self.x:.3f}, {self.y:.3f})"


Vector2D = Vector2


class AABB:
    """Axis-Aligned Bounding Box with edge queries, overlap tests, and containment checks."""

    def __init__(self, x: Any, y: Any = 0.0, width: float = 0.0, height: float = 0.0):
        if isinstance(x, Vector2) and isinstance(y, Vector2):
            self.x = float(x.x)
            self.y = float(x.y)
            self.width = float(y.x)
            self.height = float(y.y)
        else:
            self.x = float(x)
            self.y = float(y)
            self.width = float(width)
            self.height = float(height)

    @property
    def w(self) -> float:
        return self.width

    @w.setter
    def w(self, val: float) -> None:
        self.width = float(val)

    @property
    def h(self) -> float:
        return self.height

    @h.setter
    def h(self, val: float) -> None:
        self.height = float(val)

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width * 0.5

    @property
    def center_y(self) -> float:
        return self.y + self.height * 0.5

    def intersects(self, other: AABB) -> bool:
        """Returns True if this bounding box overlaps with another AABB (touching edges = False)."""
        return not (
            self.right <= other.left or
            self.left >= other.right or
            self.bottom <= other.top or
            self.top >= other.bottom
        )

    def overlaps_aabb(self, other: AABB) -> bool:
        """Alias for intersects."""
        return self.intersects(other)

    def contains_point(self, px: float, py: float) -> bool:
        """Returns True if (px, py) is strictly inside or on boundaries."""
        return (self.left <= px <= self.right) and (self.top <= py <= self.bottom)

    def get_overlap_vector(self, other: AABB) -> Vector2:
        """Computes Minimum Translation Vector (MTV) to push self out of other."""
        dx1 = other.right - self.left
        dx2 = self.right - other.left
        dy1 = other.bottom - self.top
        dy2 = self.bottom - other.top

        min_x = dx1 if dx1 < dx2 else -dx2
        min_y = dy1 if dy1 < dy2 else -dy2

        if abs(min_x) < abs(min_y):
            return Vector2(min_x, 0.0)
        else:
            return Vector2(0.0, min_y)

    def __repr__(self) -> str:
        return f"AABB(x={self.x:.1f}, y={self.y:.1f}, w={self.width:.1f}, h={self.height:.1f})"


class Circle:
    """Circle bounding primitive for radial queries and projectiles."""

    def __init__(self, x: float, y: float, r: float):
        self.x = float(x)
        self.y = float(y)
        self.r = float(r)

    @property
    def radius(self) -> float:
        return self.r

    def overlaps_circle(self, other: Circle) -> bool:
        dx = self.x - other.x
        dy = self.y - other.y
        rad_sum = self.r + other.r
        return (dx * dx + dy * dy) < (rad_sum * rad_sum)

    def overlaps_aabb(self, box: AABB) -> bool:
        """Finds closest point on AABB to circle center; tests distance."""
        closest_x = min(max(self.x, box.left), box.right)
        closest_y = min(max(self.y, box.top), box.bottom)
        dx = self.x - closest_x
        dy = self.y - closest_y
        return (dx * dx + dy * dy) < (self.r * self.r)

    def contains_point(self, px: float, py: float) -> bool:
        dx = self.x - px
        dy = self.y - py
        return (dx * dx + dy * dy) <= (self.r * self.r)

    def __repr__(self) -> str:
        return f"Circle(x={self.x:.1f}, y={self.y:.1f}, r={self.r:.1f})"
