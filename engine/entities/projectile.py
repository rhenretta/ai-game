"""engine/entities/projectile.py
Fast kinetic projectile simulation, faction-based collision, and manager.
"""
from __future__ import annotations
from typing import Any, List, Optional, Tuple, Union
import config
from engine.combat.damage import DamagePacket, Faction
from engine.entities.base import Entity
from engine.spatial.aabb import AABB, Circle, Vector2


class Projectile(Entity):
    """Linear moving projectile with lifetime decay and obstacle collision."""

    def __init__(
        self,
        entity_id: str,
        owner_id: str,
        faction: Faction,
        pos: Vector2,
        vel: Vector2,
        damage: float,
        radius: float = 4.0,
        lifetime: float = 2.0
    ):
        super().__init__(entity_id, pos.x - radius, pos.y - radius, radius * 2.0, radius * 2.0)
        self.owner_id = owner_id
        self.faction = faction
        self.vel = vel
        self.damage = float(damage)
        self.radius = float(radius)
        self.lifetime = float(lifetime)

    @property
    def circle(self) -> Circle:
        return Circle(self.pos.x + self.radius, self.pos.y + self.radius, self.radius)

    def update(self, dt: float, walls: Optional[List[Union[AABB, Tuple[float, float, float, float]]]] = None) -> bool:
        self.lifetime -= dt
        if self.lifetime <= 0.0:
            self.is_alive = False
            return False

        # Move position
        self.pos = self.pos + self.vel * dt

        # Wall collision
        if walls:
            c = self.circle
            for w in walls:
                box = w if isinstance(w, AABB) else AABB(w[0], w[1], w[2], w[3])
                if c.overlaps_aabb(box):
                    self.is_alive = False
                    return False

        return True

    def render(self, surface: Any, camera_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        try:
            import pygame
            cx = int(self.pos.x + self.radius - camera_offset[0])
            cy = int(self.pos.y + self.radius - camera_offset[1])
            r = int(self.radius)
            color = config.PALETTE["cyan"] if self.faction == Faction.PLAYER else config.PALETTE["red"]
            pygame.draw.circle(surface, color, (cx, cy), r)
            pygame.draw.circle(surface, config.PALETTE["white"], (cx, cy), max(1, r - 1))
        except Exception:
            pass


class ProjectileManager:
    """Manages spawning, updating, and faction collision resolution for active projectiles."""

    def __init__(self, telemetry_recorder: Optional[Any] = None):
        self.projectiles: List[Projectile] = []
        self.counter: int = 0
        self.telemetry = telemetry_recorder

    def set_telemetry_recorder(self, recorder: Optional[Any]) -> None:
        self.telemetry = recorder

    def clear(self) -> None:
        self.projectiles.clear()

    def spawn_projectile(
        self,
        owner_id: str,
        faction: Faction,
        pos: Vector2,
        vel: Vector2,
        damage: float,
        radius: float = 4.0,
        range_lifetime: float = 2.0
    ) -> Projectile:
        self.counter += 1
        pid = f"proj_{self.counter}"
        p = Projectile(
            entity_id=pid,
            owner_id=owner_id,
            faction=faction,
            pos=pos,
            vel=vel,
            damage=damage,
            radius=radius,
            lifetime=range_lifetime
        )
        self.projectiles.append(p)
        return p

    def update(
        self,
        dt: float,
        walls: Optional[List[Union[AABB, Tuple[float, float, float, float]]]],
        player: Any,
        enemies: List[Any]
    ) -> None:
        alive_projectiles: List[Projectile] = []
        wall_list = walls or []

        for p in self.projectiles:
            if not p.update(dt, wall_list):
                continue

            hit = False
            if p.faction == Faction.PLAYER:
                for enemy in enemies:
                    if enemy.is_alive and p.circle.overlaps_aabb(enemy.aabb):
                        kb = p.vel.normalized() * 180.0
                        packet = DamagePacket(
                            amount=p.damage,
                            source_faction=Faction.PLAYER,
                            knockback_impulse=kb,
                            iframes_granted=0.25,
                            source_id=p.owner_id
                        )
                        if enemy.take_damage(packet):
                            hit = True
                            if self.telemetry:
                                self.telemetry.emit_event(
                                    event_type="DAMAGE_DEALT",
                                    pos=(enemy.center.x, enemy.center.y),
                                    vel=(0.0, 0.0),
                                    health=enemy.health,
                                    payload={"target_id": enemy.entity_id, "damage": p.damage}
                                )
                        break

            elif p.faction == Faction.ENEMY:
                if player.is_alive and p.circle.overlaps_aabb(player.aabb):
                    kb = p.vel.normalized() * 220.0
                    packet = DamagePacket(
                        amount=p.damage,
                        source_faction=Faction.ENEMY,
                        knockback_impulse=kb,
                        iframes_granted=0.40,
                        source_id=p.owner_id
                    )
                    if player.take_damage(packet):
                        hit = True
                        if self.telemetry:
                            self.telemetry.emit_event(
                                event_type="DAMAGE_TAKEN",
                                pos=(player.center.x, player.center.y),
                                vel=(player.vel.x, player.vel.y),
                                health=player.health,
                                payload={"source": "enemy_projectile", "damage": p.damage, "attacker_id": p.owner_id}
                            )

            if not hit and p.is_alive:
                alive_projectiles.append(p)

        self.projectiles = alive_projectiles

    def render(self, surface: Any, camera_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        for p in self.projectiles:
            p.render(surface, camera_offset)
