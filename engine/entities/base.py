"""engine/entities/base.py
Entity and LivingEntity base classes for spatial actors and combat participants.
"""
from __future__ import annotations
import math
from typing import Any, Tuple
from engine.combat.damage import DamagePacket, Faction
from engine.spatial.aabb import AABB, Vector2


class Entity:
    """Base class for all interactive objects, hazards, and characters."""

    def __init__(self, entity_id: str, x: float, y: float, w: float, h: float):
        self.entity_id = str(entity_id)
        self.pos = Vector2(x, y)
        self.size = Vector2(w, h)
        self.is_alive = True

    @property
    def aabb(self) -> AABB:
        return AABB(self.pos.x, self.pos.y, self.size.x, self.size.y)

    @property
    def center(self) -> Vector2:
        return Vector2(self.pos.x + self.size.x * 0.5, self.pos.y + self.size.y * 0.5)

    @property
    def x(self) -> float:
        return self.pos.x

    @x.setter
    def x(self, val: float) -> None:
        self.pos.x = float(val)

    @property
    def y(self) -> float:
        return self.pos.y

    @y.setter
    def y(self, val: float) -> None:
        self.pos.y = float(val)

    def update(self, dt: float) -> None:
        pass

    def render(self, surface: Any, camera_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        pass


class LivingEntity(Entity):
    """Actors with health, factions, damage vulnerability, and knockback physics."""

    def __init__(
        self,
        entity_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        max_health: float,
        faction: Faction
    ):
        super().__init__(entity_id, x, y, w, h)
        self.max_health = float(max_health)
        self.health = float(max_health)
        self.faction = faction
        self.vel = Vector2(0.0, 0.0)
        self.knockback_vel = Vector2(0.0, 0.0)
        self.iframe_timer = 0.0

    @property
    def vx(self) -> float:
        return self.vel.x

    @vx.setter
    def vx(self, val: float) -> None:
        self.vel.x = float(val)

    @property
    def vy(self) -> float:
        return self.vel.y

    @vy.setter
    def vy(self, val: float) -> None:
        self.vel.y = float(val)

    def take_damage(self, packet: DamagePacket) -> bool:
        """Applies damage if not in iframes. Returns True if damage was applied."""
        if not self.is_alive or self.iframe_timer > 0.0:
            return False

        self.health = max(0.0, self.health - packet.amount)
        self.iframe_timer = packet.iframes_granted
        self.knockback_vel = self.knockback_vel + packet.knockback_impulse

        if self.health <= 0.0:
            self.is_alive = False
            self.on_death()
        return True

    def on_death(self) -> None:
        pass

    def update_combat_state(self, dt: float) -> None:
        if self.iframe_timer > 0.0:
            self.iframe_timer = max(0.0, self.iframe_timer - dt)
        decay = math.exp(-10.0 * dt)
        self.knockback_vel = self.knockback_vel * decay
