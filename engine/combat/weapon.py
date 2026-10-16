"""engine/combat/weapon.py
Weapons and firing mechanics for player and enemies.
"""
from __future__ import annotations
import math
from typing import Any, List, Optional
from engine.combat.damage import Faction
from engine.spatial.aabb import Vector2


class Weapon:
    """Base weapon class managing fire rate cooldowns and projectile spawning."""

    def __init__(
        self,
        name: str = "base_weapon",
        fire_rate: float = 0.25,
        damage: float = 20.0,
        projectile_speed: float = 600.0,
        projectile_radius: float = 4.0,
        projectile_lifetime: float = 1.8
    ):
        self.name = name
        self.fire_rate = float(fire_rate)
        self.damage = float(damage)
        self.projectile_speed = float(projectile_speed)
        self.projectile_radius = float(projectile_radius)
        self.projectile_lifetime = float(projectile_lifetime)
        self.cooldown = 0.0

    def update(self, dt: float) -> None:
        if self.cooldown > 0.0:
            self.cooldown = max(0.0, self.cooldown - dt)

    def can_fire(self) -> bool:
        return self.cooldown <= 0.0

    def fire(
        self,
        muzzle_pos: Vector2,
        aim_angle: float,
        faction: Faction,
        owner_id: str,
        projectile_manager: Any
    ) -> List[Any]:
        if not self.can_fire():
            return []
        self.cooldown = self.fire_rate
        p_vel = Vector2(math.cos(aim_angle) * self.projectile_speed, math.sin(aim_angle) * self.projectile_speed)
        proj = projectile_manager.spawn_projectile(
            owner_id=owner_id,
            faction=faction,
            pos=muzzle_pos,
            vel=p_vel,
            damage=self.damage,
            radius=self.projectile_radius,
            range_lifetime=self.projectile_lifetime
        )
        return [proj]


class BlasterWeapon(Weapon):
    """Standard rapid-fire single beam weapon."""

    def __init__(self):
        super().__init__(
            name="blaster",
            fire_rate=0.22,
            damage=25.0,
            projectile_speed=750.0,
            projectile_radius=4.0,
            projectile_lifetime=1.8
        )


class SpreadWeapon(Weapon):
    """Tri-shot spread weapon firing 3 divergent projectiles."""

    def __init__(self):
        super().__init__(
            name="spread_cannon",
            fire_rate=0.45,
            damage=14.0,
            projectile_speed=650.0,
            projectile_radius=3.5,
            projectile_lifetime=1.5
        )

    def fire(
        self,
        muzzle_pos: Vector2,
        aim_angle: float,
        faction: Faction,
        owner_id: str,
        projectile_manager: Any
    ) -> List[Any]:
        if not self.can_fire():
            return []
        self.cooldown = self.fire_rate
        projectiles = []
        spread_offsets = [-0.18, 0.0, 0.18]
        for offset in spread_offsets:
            angle = aim_angle + offset
            p_vel = Vector2(math.cos(angle) * self.projectile_speed, math.sin(angle) * self.projectile_speed)
            proj = projectile_manager.spawn_projectile(
                owner_id=owner_id,
                faction=faction,
                pos=muzzle_pos,
                vel=p_vel,
                damage=self.damage,
                radius=self.projectile_radius,
                range_lifetime=self.projectile_lifetime
            )
            projectiles.append(proj)
        return projectiles
