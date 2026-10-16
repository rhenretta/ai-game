"""engine/entities/hazard.py
Environmental hazards: SpikeHazard, LaserSweepHazard, and FloorAcidHazard.
"""
from __future__ import annotations
from enum import Enum, auto
import math
from typing import Any, Optional, Tuple
import config
from engine.combat.damage import DamagePacket, Faction
from engine.entities.base import Entity
from engine.spatial.aabb import Vector2


class Hazard(Entity):
    """Base class for lethal or impairing environmental hazards."""

    def __init__(
        self,
        entity_id: str = "hazard",
        x: float = 0.0,
        y: float = 0.0,
        w: float = 32.0,
        h: float = 32.0,
        hazard_id: Optional[str] = None,
    ):
        actual_id = hazard_id if hazard_id is not None else entity_id
        super().__init__(actual_id, x, y, w, h)


class SpikeState(Enum):
    RETRACTED = auto()
    WARNING = auto()
    EXTENDED = auto()


class SpikeHazard(Hazard):
    """Cyclic ground trap cycling between Retracted, Warning, and Extended spikes."""

    def __init__(self, hazard_id: str, x: float, y: float, w: float = 48.0, h: float = 48.0):
        super().__init__(hazard_id, x, y, w, h)
        self.state = SpikeState.RETRACTED
        self.state_timer = 1.6
        self.damage = 20.0

    def update(self, dt: float, player: Any, telemetry_recorder: Optional[Any] = None) -> None:
        self.state_timer -= dt
        if self.state_timer <= 0.0:
            if self.state == SpikeState.RETRACTED:
                self.state = SpikeState.WARNING
                self.state_timer = 0.5
            elif self.state == SpikeState.WARNING:
                self.state = SpikeState.EXTENDED
                self.state_timer = 1.0
            elif self.state == SpikeState.EXTENDED:
                self.state = SpikeState.RETRACTED
                self.state_timer = 1.6

        if self.state == SpikeState.EXTENDED and player.is_alive:
            if self.aabb.intersects(player.aabb):
                c_diff = player.center - self.center
                kb = c_diff.normalized() * 300.0 if c_diff.length_squared() > 0.01 else Vector2(0.0, -300.0)
                packet = DamagePacket(
                    amount=self.damage,
                    source_faction=Faction.NEUTRAL,
                    knockback_impulse=kb,
                    iframes_granted=0.50,
                    source_id=self.entity_id
                )
                if player.take_damage(packet) and telemetry_recorder:
                    telemetry_recorder.emit_event(
                        event_type="HAZARD_HIT",
                        pos=(player.center.x, player.center.y),
                        vel=(player.vel.x, player.vel.y),
                        health=player.health,
                        payload={"hazard_type": "spikes", "damage": self.damage}
                    )

    def render(self, surface: Any, camera_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        try:
            import pygame
            rx = int(self.pos.x - camera_offset[0])
            ry = int(self.pos.y - camera_offset[1])
            rw = int(self.size.x)
            rh = int(self.size.y)
            rect = pygame.Rect(rx, ry, rw, rh)

            if self.state == SpikeState.RETRACTED:
                pygame.draw.rect(surface, config.PALETTE["dark_gray"], rect)
                pygame.draw.rect(surface, config.PALETTE["muted_gray"], rect, width=2)
            elif self.state == SpikeState.WARNING:
                pygame.draw.rect(surface, config.PALETTE["rust_brown"], rect)
                pygame.draw.rect(surface, config.PALETTE["yellow"], rect, width=2)
            else:  # EXTENDED
                pygame.draw.rect(surface, config.PALETTE["red"], rect)
                pygame.draw.rect(surface, config.PALETTE["white"], rect, width=2)
                # Draw spike teeth
                for i in range(4):
                    sx = rx + i * 12 + 6
                    pygame.draw.polygon(surface, config.PALETTE["white"], [
                        (sx - 4, ry + rh - 4), (sx, ry + 4), (sx + 4, ry + rh - 4)
                    ])
        except Exception:
            pass


class LaserSweepHazard(Hazard):
    """Sweeping laser emitter that sweeps an angular arc across the room."""

    def __init__(
        self,
        hazard_id: str,
        origin_x: float,
        origin_y: float,
        length: float = 380.0,
        mid_angle: float = 0.0,
        sweep_amplitude: float = math.pi * 0.4,
        speed: float = 1.2
    ):
        super().__init__(hazard_id, origin_x - 10.0, origin_y - 10.0, 20.0, 20.0)
        self.origin = Vector2(origin_x, origin_y)
        self.length = float(length)
        self.mid_angle = float(mid_angle)
        self.amplitude = float(sweep_amplitude)
        self.speed = float(speed)
        self.time_accum = 0.0
        self.current_angle = float(mid_angle)
        self.damage = 12.0
        self.beam_width = 8.0

    def get_endpoints(self) -> Tuple[Vector2, Vector2]:
        p1 = self.origin
        p2 = Vector2(
            p1.x + math.cos(self.current_angle) * self.length,
            p1.y + math.sin(self.current_angle) * self.length
        )
        return p1, p2

    def update(self, dt: float, player: Any, telemetry_recorder: Optional[Any] = None) -> None:
        self.time_accum += dt
        self.current_angle = self.mid_angle + self.amplitude * math.sin(self.speed * self.time_accum)

        if not player.is_alive:
            return

        p1, p2 = self.get_endpoints()
        c = player.center
        r = player.size.x * 0.5

        seg_vec = p2 - p1
        seg_len_sq = seg_vec.length_squared()
        if seg_len_sq == 0.0:
            return

        t = max(0.0, min(1.0, (c - p1).dot(seg_vec) / seg_len_sq))
        closest = p1 + seg_vec * t
        dist = c.distance_to(closest)

        if dist <= (r + self.beam_width * 0.5):
            c_diff = c - closest
            kb = c_diff.normalized() * 250.0 if c_diff.length_squared() > 0.01 else Vector2(0.0, -250.0)
            packet = DamagePacket(
                amount=self.damage,
                source_faction=Faction.NEUTRAL,
                knockback_impulse=kb,
                iframes_granted=0.35,
                source_id=self.entity_id
            )
            if player.take_damage(packet) and telemetry_recorder:
                telemetry_recorder.emit_event(
                    event_type="HAZARD_HIT",
                    pos=(c.x, c.y),
                    vel=(player.vel.x, player.vel.y),
                    health=player.health,
                    payload={"hazard_type": "laser_sweep", "damage": self.damage}
                )

    def render(self, surface: Any, camera_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        try:
            import pygame
            p1, p2 = self.get_endpoints()
            x1 = int(p1.x - camera_offset[0])
            y1 = int(p1.y - camera_offset[1])
            x2 = int(p2.x - camera_offset[0])
            y2 = int(p2.y - camera_offset[1])

            # Draw emitter base
            pygame.draw.circle(surface, config.PALETTE["rust_brown"], (x1, y1), 10)
            pygame.draw.circle(surface, config.PALETTE["red"], (x1, y1), 6)

            # Draw laser beam with core glow
            pygame.draw.line(surface, config.PALETTE["red"], (x1, y1), (x2, y2), width=int(self.beam_width))
            pygame.draw.line(surface, config.PALETTE["white"], (x1, y1), (x2, y2), width=2)
        except Exception:
            pass


class FloorAcidHazard(Hazard):
    """Corrosive floor zone applying continuous movement speed reduction and DoT."""

    def __init__(self, hazard_id: str, x: float, y: float, w: float = 96.0, h: float = 96.0):
        super().__init__(hazard_id, x, y, w, h)
        self.dot_tick_rate = 0.50
        self.dot_timer = 0.0
        self.dot_damage = 6.0

    def update(self, dt: float, player: Any, telemetry_recorder: Optional[Any] = None) -> None:
        if not player.is_alive:
            return

        if self.aabb.intersects(player.aabb):
            # Slow player movement velocity while wading through acid
            if hasattr(player, "speed_multiplier"):
                player.speed_multiplier = min(getattr(player, "speed_multiplier", 1.0), 0.55)
            player.vel = player.vel * 0.55

            self.dot_timer += dt
            if self.dot_timer >= self.dot_tick_rate:
                self.dot_timer = 0.0
                packet = DamagePacket(
                    amount=self.dot_damage,
                    source_faction=Faction.NEUTRAL,
                    knockback_impulse=Vector2(0.0, 0.0),
                    iframes_granted=0.20,
                    source_id=self.entity_id
                )
                if player.take_damage(packet) and telemetry_recorder:
                    c = player.center
                    telemetry_recorder.emit_event(
                        event_type="HAZARD_HIT",
                        pos=(c.x, c.y),
                        vel=(player.vel.x, player.vel.y),
                        health=player.health,
                        payload={"hazard_type": "floor_acid", "damage": self.dot_damage}
                    )
        else:
            self.dot_timer = 0.0

    def render(self, surface: Any, camera_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        try:
            import pygame
            rx = int(self.pos.x - camera_offset[0])
            ry = int(self.pos.y - camera_offset[1])
            rw = int(self.size.x)
            rh = int(self.size.y)
            rect = pygame.Rect(rx, ry, rw, rh)

            # Bubbling acid puddle representation
            pygame.draw.rect(surface, config.PALETTE["dark_green"], rect, border_radius=8)
            pygame.draw.rect(surface, config.PALETTE["green"], rect, width=2, border_radius=8)
        except Exception:
            pass
