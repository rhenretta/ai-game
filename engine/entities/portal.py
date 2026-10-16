"""engine/entities/portal.py
Level exit portal triggering episodic victory upon player entry.
"""
from __future__ import annotations
import math
from typing import Any, List, Optional, Tuple
import config
from engine.entities.base import Entity
from engine.spatial.aabb import Circle


class Portal(Entity):
    """Exit portal that unlocks when level combat objectives are met."""

    def __init__(self, x: float, y: float, radius: float = 36.0):
        super().__init__("level_portal", x - radius, y - radius, radius * 2.0, radius * 2.0)
        self.radius = float(radius)
        self.is_unlocked = False
        self.pulse_timer = 0.0

    @property
    def circle(self) -> Circle:
        return Circle(self.pos.x + self.radius, self.pos.y + self.radius, self.radius)

    def update(
        self,
        dt: float,
        enemies: List[Any],
        player: Any,
        telemetry_recorder: Optional[Any] = None
    ) -> bool:
        """
        Updates portal lock state. Returns True if player entered unlocked portal.
        """
        self.pulse_timer += dt * 4.0

        # Unlocks when all enemies are defeated
        active_enemies = any(e.is_alive for e in enemies)
        self.is_unlocked = not active_enemies

        if self.is_unlocked and player.is_alive:
            if self.circle.overlaps_aabb(player.aabb):
                if telemetry_recorder:
                    c = player.center
                    telemetry_recorder.emit_event(
                        event_type="LEVEL_EXIT_REACHED",
                        pos=(c.x, c.y),
                        vel=(player.vel.x, player.vel.y),
                        health=player.health,
                        payload={"portal_id": self.entity_id}
                    )
                return True
        return False

    def render(self, surface: Any, camera_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        try:
            import pygame
            cx = int(self.pos.x + self.radius - camera_offset[0])
            cy = int(self.pos.y + self.radius - camera_offset[1])
            r = int(self.radius)

            if self.is_unlocked:
                pulse_r = max(2, int(r + 4.0 * math.sin(self.pulse_timer)))
                pygame.draw.circle(surface, config.PALETTE["dark_navy"], (cx, cy), pulse_r)
                pygame.draw.circle(surface, config.PALETTE["cyan"], (cx, cy), pulse_r, width=3)
                pygame.draw.circle(surface, config.PALETTE["white"], (cx, cy), max(2, pulse_r - 8), width=2)
            else:
                # Locked state
                pygame.draw.circle(surface, config.PALETTE["black"], (cx, cy), r)
                pygame.draw.circle(surface, config.PALETTE["dark_gray"], (cx, cy), r, width=3)
                pygame.draw.line(surface, config.PALETTE["red"], (cx - 12, cy - 12), (cx + 12, cy + 12), width=3)
                pygame.draw.line(surface, config.PALETTE["red"], (cx - 12, cy + 12), (cx + 12, cy - 12), width=3)
        except Exception:
            pass
