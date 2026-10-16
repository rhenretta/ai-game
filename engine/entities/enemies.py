"""engine/entities/enemies.py
Enemy archetypes: ChaserEnemy (melee tracking) and RangedEnemy (standoff kiting sniper).
"""
from __future__ import annotations
from enum import Enum, auto
import math
from typing import Any, List, Optional, Tuple, Union
import config
from engine.combat.damage import DamagePacket, Faction
from engine.entities.base import LivingEntity
from engine.spatial.aabb import AABB, Vector2
from engine.spatial.collision import move_and_resolve_walls


class EnemyState(Enum):
    IDLE = auto()
    CHASE = auto()
    ATTACK = auto()
    DEAD = auto()


class BaseEnemy(LivingEntity):
    """Base class for autonomous enemy entities with health bar rendering."""

    def __init__(
        self,
        entity_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        max_health: float,
        speed: float
    ):
        super().__init__(entity_id, x, y, w, h, max_health=max_health, faction=Faction.ENEMY)
        self.speed = float(speed)
        self.state = EnemyState.CHASE
        self.attack_cooldown = 0.0

    def update_ai(
        self,
        dt: float,
        player: Any,
        walls: Optional[List[Union[AABB, Tuple[float, float, float, float]]]] = None,
        projectile_manager: Optional[Any] = None,
        telemetry_recorder: Optional[Any] = None,
    ) -> None:
        """Polymorphic AI tick hook for enemy behavior.
        Subclasses should override this method to implement autonomous behaviors.
        """
        pass

    def render_health_bar(self, surface: Any, camera_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        try:
            import pygame
            bar_w = self.size.x
            bar_h = 4.0
            bar_x = self.pos.x - camera_offset[0]
            bar_y = self.pos.y - bar_h - 6.0 - camera_offset[1]
            pct = max(0.0, self.health / self.max_health)

            bg_rect = pygame.Rect(int(bar_x), int(bar_y), int(bar_w), int(bar_h))
            fg_rect = pygame.Rect(int(bar_x), int(bar_y), int(bar_w * pct), int(bar_h))
            pygame.draw.rect(surface, config.PALETTE["red"], bg_rect)
            pygame.draw.rect(surface, config.PALETTE["green"], fg_rect)
            pygame.draw.rect(surface, config.PALETTE["black"], bg_rect, width=1)
        except Exception:
            pass


class ChaserEnemy(BaseEnemy):
    """Aggressive melee brawler tracking the player directly."""

    def __init__(self, entity_id: str = "", x: float = 0.0, y: float = 0.0, enemy_id: Optional[str] = None):
        eid = enemy_id if enemy_id is not None else entity_id
        super().__init__(eid, x, y, w=32.0, h=32.0, max_health=60.0, speed=185.0)
        self.contact_damage = 15.0
        self.attack_rate = 0.80

    def update_ai(
        self,
        dt: float,
        player: Any,
        walls: Optional[List[Union[AABB, Tuple[float, float, float, float]]]] = None,
        projectile_manager: Optional[Any] = None,
        telemetry_recorder: Optional[Any] = None
    ) -> None:
        self.update_combat_state(dt)
        if self.attack_cooldown > 0.0:
            self.attack_cooldown = max(0.0, self.attack_cooldown - dt)

        if not self.is_alive or not player.is_alive:
            return

        c_enemy = self.center
        c_player = player.center
        dist = c_enemy.distance_to(c_player)

        # 1. Track player direction
        if dist > 8.0:
            dir_vec = (c_player - c_enemy).normalized()
            self.vel = dir_vec * self.speed
        else:
            self.vel = Vector2(0.0, 0.0)

        # 2. Wall resolution
        wall_list = walls or []
        total_vel = self.vel + self.knockback_vel
        self.pos, _ = move_and_resolve_walls(self.aabb, total_vel, dt, wall_list)

        # 3. Melee contact attack
        if self.aabb.intersects(player.aabb) and self.attack_cooldown <= 0.0:
            self.attack_cooldown = self.attack_rate
            push_dir = (player.center - self.center).normalized()
            packet = DamagePacket(
                amount=self.contact_damage,
                source_faction=Faction.ENEMY,
                knockback_impulse=push_dir * 380.0,
                iframes_granted=0.40,
                source_id=self.entity_id
            )
            applied = player.take_damage(packet)
            if applied and telemetry_recorder:
                telemetry_recorder.emit_event(
                    event_type="DAMAGE_TAKEN",
                    pos=(player.center.x, player.center.y),
                    vel=(player.vel.x, player.vel.y),
                    health=player.health,
                    payload={"source": "chaser_contact", "damage": self.contact_damage, "attacker_id": self.entity_id}
                )

    def render(self, surface: Any, camera_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        try:
            import pygame
            px = int(self.pos.x - camera_offset[0])
            py = int(self.pos.y - camera_offset[1])
            pw = int(self.size.x)
            ph = int(self.size.y)

            color = config.PALETTE["orange"] if self.iframe_timer <= 0 else config.PALETTE["white"]
            rect = pygame.Rect(px, py, pw, ph)
            pygame.draw.rect(surface, color, rect, border_radius=4)
            pygame.draw.rect(surface, config.PALETTE["red"], rect, width=2, border_radius=4)
            self.render_health_bar(surface, camera_offset)
        except Exception:
            pass


class RangedEnemy(BaseEnemy):
    """Methodical sniper maintaining standoff distance and firing projectiles."""

    def __init__(self, entity_id: str, x: float, y: float):
        super().__init__(entity_id, x, y, w=28.0, h=28.0, max_health=45.0, speed=135.0)
        self.fire_rate = 1.40
        self.strafe_dir = 1.0
        self.strafe_timer = 2.0
        self.projectile_damage = 12.0
        self.projectile_speed = 450.0

    def update_ai(
        self,
        dt: float,
        player: Any,
        walls: Optional[List[Union[AABB, Tuple[float, float, float, float]]]] = None,
        projectile_manager: Optional[Any] = None,
        telemetry_recorder: Optional[Any] = None
    ) -> None:
        self.update_combat_state(dt)
        if self.attack_cooldown > 0.0:
            self.attack_cooldown = max(0.0, self.attack_cooldown - dt)

        if not self.is_alive or not player.is_alive:
            return

        c_enemy = self.center
        c_player = player.center
        dist = c_enemy.distance_to(c_player)
        to_player = (c_player - c_enemy).normalized()

        # Standoff positioning
        if dist < 220.0:
            self.vel = to_player * -self.speed
        elif dist > 360.0:
            self.vel = to_player * self.speed
        else:
            self.strafe_timer -= dt
            if self.strafe_timer <= 0.0:
                self.strafe_dir *= -1.0
                self.strafe_timer = 2.0
            lateral = Vector2(-to_player.y, to_player.x) * self.strafe_dir
            self.vel = lateral * self.speed

        wall_list = walls or []
        total_vel = self.vel + self.knockback_vel
        self.pos, _ = move_and_resolve_walls(self.aabb, total_vel, dt, wall_list)

        # Firing projectile towards player
        if self.attack_cooldown <= 0.0 and dist <= 550.0 and projectile_manager:
            self.attack_cooldown = self.fire_rate
            aim_angle = math.atan2(c_player.y - c_enemy.y, c_player.x - c_enemy.x)
            p_vel = Vector2(math.cos(aim_angle) * self.projectile_speed, math.sin(aim_angle) * self.projectile_speed)
            projectile_manager.spawn_projectile(
                owner_id=self.entity_id,
                faction=Faction.ENEMY,
                pos=c_enemy,
                vel=p_vel,
                damage=self.projectile_damage,
                radius=5.0,
                range_lifetime=2.5
            )

    def render(self, surface: Any, camera_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        try:
            import pygame
            px = int(self.pos.x - camera_offset[0])
            py = int(self.pos.y - camera_offset[1])
            pw = int(self.size.x)
            ph = int(self.size.y)

            color = config.PALETTE["purple"] if self.iframe_timer <= 0 else config.PALETTE["white"]
            rect = pygame.Rect(px, py, pw, ph)
            pygame.draw.rect(surface, color, rect, border_radius=4)
            pygame.draw.rect(surface, config.PALETTE["pink"], rect, width=2, border_radius=4)
            self.render_health_bar(surface, camera_offset)
        except Exception:
            pass
