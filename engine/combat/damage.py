"""engine/combat/damage.py
Damage calculation, faction definitions, and knockback dynamics.
"""
from __future__ import annotations
from enum import Enum, auto
import math
from typing import Optional
from engine.spatial.aabb import Vector2


class Faction(Enum):
    PLAYER = auto()
    ENEMY = auto()
    NEUTRAL = auto()


class DamagePacket:
    """Carries damage, knockback, faction, and iFrame duration data."""

    def __init__(
        self,
        amount: float,
        source_faction: Faction,
        knockback_impulse: Optional[Vector2] = None,
        iframes_granted: float = 0.40,
        source_id: str = ""
    ):
        self.amount = float(amount)
        self.source_faction = source_faction
        self.knockback_impulse = knockback_impulse if knockback_impulse is not None else Vector2(0.0, 0.0)
        self.iframes_granted = float(iframes_granted)
        self.source_id = str(source_id)

    def __repr__(self) -> str:
        return f"DamagePacket(amount={self.amount:.1f}, faction={self.source_faction.name}, source={self.source_id})"


class KnockbackState:
    """Manages exponential decay of knockback velocity impulses."""

    def __init__(self, decay_rate: float = 10.0):
        self.vel = Vector2(0.0, 0.0)
        self.decay_rate = float(decay_rate)

    def apply_impulse(self, impulse: Vector2) -> None:
        self.vel = self.vel + impulse

    def update(self, dt: float) -> None:
        decay = math.exp(-self.decay_rate * dt)
        self.vel = self.vel * decay

    def reset(self) -> None:
        self.vel = Vector2(0.0, 0.0)


def calculate_damage(raw_damage: float, defense: float = 0.0) -> float:
    """Calculates effective damage with minimum 1.0 floor."""
    effective = max(1.0, float(raw_damage) - float(defense))
    return effective
