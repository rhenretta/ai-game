"""engine/entities/player.py
Player entity with 8-directional WASD movement, high-mobility dash, aiming, and telemetry hooks.
"""
from __future__ import annotations
import math
from typing import Any, List, Optional, Tuple, Union
import config
from engine.combat.damage import Faction
from engine.combat.weapon import BlasterWeapon, Weapon
from engine.entities.base import LivingEntity
from engine.spatial.aabb import AABB, Vector2
from engine.spatial.collision import move_and_resolve_walls


class Player(LivingEntity):
    """Controllable player character with combat and movement mechanics."""

    def __init__(self, x: float = 100.0, y: float = 100.0, telemetry_recorder: Optional[Any] = None):
        super().__init__("player", x, y, 32.0, 32.0, max_health=100.0, faction=Faction.PLAYER)
        self.telemetry = telemetry_recorder

        # Movement attributes
        self.walk_speed = 260.0
        self.dash_speed = 650.0
        self.speed_multiplier = 1.0

        # Stamina system
        self.stamina = 100.0
        self.max_stamina = 100.0
        self.dash_cost = 25.0
        self.stamina_regen_rate = 35.0
        self.stamina_regen_timer = 0.0

        # Dash state
        self.is_dashing = False
        self.dash_timer = 0.0
        self.dash_cooldown = 0.0
        self.dash_dir = Vector2(1.0, 0.0)

        # Combat & Aiming
        self.aim_angle = 0.0
        self.weapon: Weapon = BlasterWeapon()
        self.telemetry_tick_timer = 0.0

    @property
    def speed_modifier(self) -> float:
        return self.speed_multiplier

    @speed_modifier.setter
    def speed_modifier(self, value: float) -> None:
        self.speed_multiplier = float(value)

    def set_telemetry_recorder(self, recorder: Optional[Any]) -> None:
        self.telemetry = recorder

    def handle_input(
        self,
        move_input: Vector2,
        mouse_canvas_pos: Vector2,
        dash_pressed: bool,
        fire_pressed: bool,
        projectile_manager: Any
    ) -> None:
        c = self.center
        self.aim_angle = math.atan2(mouse_canvas_pos.y - c.y, mouse_canvas_pos.x - c.x)

        # 1. Dash trigger
        if dash_pressed and not self.is_dashing and self.dash_cooldown <= 0.0:
            if self.stamina >= self.dash_cost:
                self.is_dashing = True
                self.dash_timer = 0.20
                self.dash_cooldown = 0.60
                self.stamina -= self.dash_cost
                self.stamina_regen_timer = 0.50

                if move_input.length_squared() > 0.001:
                    self.dash_dir = move_input.normalized()
                else:
                    self.dash_dir = Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle))

                # Invulnerability window during dash
                self.iframe_timer = 0.20

                if self.telemetry:
                    self.telemetry.emit_event(
                        event_type="DASH",
                        pos=(c.x, c.y),
                        vel=(self.dash_dir.x * self.dash_speed, self.dash_dir.y * self.dash_speed),
                        health=self.health,
                        payload={"stamina_remaining": self.stamina}
                    )

        # 2. Weapon firing
        if fire_pressed and not self.is_dashing and self.weapon.can_fire():
            muzzle_x = c.x + math.cos(self.aim_angle) * 20.0
            muzzle_y = c.y + math.sin(self.aim_angle) * 20.0
            muzzle_pos = Vector2(muzzle_x, muzzle_y)
            spawned = self.weapon.fire(
                muzzle_pos=muzzle_pos,
                aim_angle=self.aim_angle,
                faction=Faction.PLAYER,
                owner_id=self.entity_id,
                projectile_manager=projectile_manager
            )
            if spawned and self.telemetry:
                self.telemetry.emit_event(
                    event_type="WEAPON_FIRE",
                    pos=(muzzle_x, muzzle_y),
                    vel=(math.cos(self.aim_angle) * self.weapon.projectile_speed, math.sin(self.aim_angle) * self.weapon.projectile_speed),
                    health=self.health,
                    payload={"weapon": self.weapon.name, "aim_angle": self.aim_angle}
                )

        # 3. Base velocity computation
        if self.is_dashing:
            self.vel = self.dash_dir * self.dash_speed
        else:
            norm_input = move_input.normalized()
            effective_speed = self.walk_speed * getattr(self, "speed_multiplier", 1.0)
            self.vel = norm_input * effective_speed

    def update(self, dt: float, walls: Optional[List[Union[AABB, Tuple[float, float, float, float]]]] = None) -> None:
        self.update_combat_state(dt)
        self.weapon.update(dt)

        # Cooldowns
        if self.dash_cooldown > 0.0:
            self.dash_cooldown = max(0.0, self.dash_cooldown - dt)

        # Dash progress
        if self.is_dashing:
            self.dash_timer -= dt
            if self.dash_timer <= 0.0:
                self.is_dashing = False

        # Stamina regen
        if self.stamina_regen_timer > 0.0:
            self.stamina_regen_timer = max(0.0, self.stamina_regen_timer - dt)
        elif self.stamina < self.max_stamina:
            self.stamina = min(self.max_stamina, self.stamina + self.stamina_regen_rate * dt)

        # Spatial integration & wall resolution
        wall_list = walls or []
        total_vel = self.vel + self.knockback_vel
        new_pos, actual_vel = move_and_resolve_walls(self.aabb, total_vel, dt, wall_list)
        self.pos = new_pos

        # Periodic telemetry tick (100ms)
        self.telemetry_tick_timer += dt
        if self.telemetry_tick_timer >= 0.10:
            self.telemetry_tick_timer = 0.0
            if self.telemetry:
                c = self.center
                self.telemetry.emit_event(
                    event_type="MOVEMENT_TICK",
                    pos=(c.x, c.y),
                    vel=(actual_vel.x, actual_vel.y),
                    health=self.health,
                    payload={"is_dashing": self.is_dashing, "stamina": self.stamina}
                )

    def on_death(self) -> None:
        if self.telemetry:
            c = self.center
            self.telemetry.emit_event(
                event_type="PLAYER_DEATH",
                pos=(c.x, c.y),
                vel=(0.0, 0.0),
                health=0.0,
                payload={"cause": "combat"}
            )

    def render(self, surface: Any, camera_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        """Renders player sprite and directional indicator."""
        try:
            import pygame
            px = int(self.pos.x - camera_offset[0])
            py = int(self.pos.y - camera_offset[1])
            pw = int(self.size.x)
            ph = int(self.size.y)

            # iFrame blink
            if self.iframe_timer > 0.0 and int(self.iframe_timer * 30) % 2 == 0:
                body_color = config.PALETTE["white"]
            else:
                body_color = config.PALETTE["cyan"] if not self.is_dashing else config.PALETTE["yellow"]

            rect = pygame.Rect(px, py, pw, ph)
            pygame.draw.rect(surface, body_color, rect, border_radius=4)
            pygame.draw.rect(surface, config.PALETTE["white"], rect, width=2, border_radius=4)

            # Muzzle aim pointer
            cx = px + pw // 2
            cy = py + ph // 2
            gun_x = cx + int(math.cos(self.aim_angle) * 18.0)
            gun_y = cy + int(math.sin(self.aim_angle) * 18.0)
            pygame.draw.line(surface, config.PALETTE["yellow"], (cx, cy), (gun_x, gun_y), width=3)
        except Exception:
            pass
