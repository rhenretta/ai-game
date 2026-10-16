"""engine/core/scenes/title_scene.py
Splash & Title scene displaying game title and controls.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import uuid
import pygame
import config
from engine.core.scene import Scene


class TitleScene(Scene):
    """Title screen displaying control instructions and starting episodes."""

    def __init__(self, engine: Any):
        super().__init__(engine)
        self.font_title: Optional[pygame.font.Font] = None
        self.font_sub: Optional[pygame.font.Font] = None
        self.font_body: Optional[pygame.font.Font] = None
        self.pulse_timer: float = 0.0
        self.session_id: str = str(uuid.uuid4())

    def on_enter(self, context: Dict[str, Any]) -> None:
        try:
            self.font_title = pygame.font.SysFont("Helvetica", 46, bold=True)
            self.font_sub = pygame.font.SysFont("Helvetica", 22, bold=True)
            self.font_body = pygame.font.SysFont("Courier", 16)
        except Exception:
            self.font_title = pygame.font.Font(None, 46)
            self.font_sub = pygame.font.Font(None, 24)
            self.font_body = pygame.font.Font(None, 18)
        self.pulse_timer = 0.0

    def on_exit(self) -> Dict[str, Any]:
        return {"session_id": self.session_id}

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                self.start_game()
            elif event.key == pygame.K_q:
                self.engine.is_running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.start_game()

    def start_game(self) -> None:
        """Transitions into PlayingScene for Episode 1."""
        self.engine.scene_manager.switch_to("playing", episode_number=1, session_id=self.session_id)

    def update(self, dt: float) -> None:
        self.pulse_timer += dt * 3.0

    def render(self, surface: pygame.Surface) -> None:
        if not self.font_title or not self.font_sub or not self.font_body:
            return

        # Title Banner
        title_surf = self.font_title.render("AUTONOMOUS GAME ENGINE & SHOWRUNNER", True, config.PALETTE["cyan"])
        surface.blit(title_surf, (config.CANVAS_WIDTH // 2 - title_surf.get_width() // 2, 140))

        sub_surf = self.font_sub.render("Continuous Rationale-First Showrunner Studio", True, config.PALETTE["light_gray"])
        surface.blit(sub_surf, (config.CANVAS_WIDTH // 2 - sub_surf.get_width() // 2, 205))

        # Controls Overview Card
        controls_card = pygame.Rect(config.CANVAS_WIDTH // 2 - 270, 270, 540, 180)
        pygame.draw.rect(surface, config.PALETTE["dark_navy"], controls_card, border_radius=8)
        pygame.draw.rect(surface, config.PALETTE["cyan"], controls_card, width=2, border_radius=8)

        lines = [
            "CONTROLS & KINETIC MECHANICS",
            "  [WASD / Arrows]  - 8-Directional Movement",
            "  [SPACEBAR]       - High-Mobility Dodge Roll (iFrames)",
            "  [MOUSE / Track]  - Aim Crosshair & Left-Click Fire",
            "  [J]              - Keyboard Fire Alternative",
            "  [ESC]            - Pause Game",
        ]
        y_offset = 285
        for i, line in enumerate(lines):
            col = config.PALETTE["yellow"] if i == 0 else config.PALETTE["white"]
            txt = self.font_body.render(line, True, col)
            surface.blit(txt, (controls_card.x + 24, y_offset))
            y_offset += 24

        # Pulsing prompt
        prompt_surf = self.font_sub.render("PRESS SPACEBAR OR CLICK TO START", True, config.PALETTE["yellow"])
        surface.blit(prompt_surf, (config.CANVAS_WIDTH // 2 - prompt_surf.get_width() // 2, 500))

        # Telemetry Connection Indicator
        telemetry_status = "TELEMETRY: CONNECTED [SQLITE WAL]" if self.engine.telemetry_recorder else "TELEMETRY: STANDALONE"
        status_surf = self.font_body.render(telemetry_status, True, config.PALETTE["green"])
        surface.blit(status_surf, (24, config.CANVAS_HEIGHT - 35))
