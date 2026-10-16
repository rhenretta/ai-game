"""engine/core/scenes/game_over_scene.py
Defeat scene presenting telemetry autopsy and retry options.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import pygame
import config
from engine.core.scene import Scene


class GameOverScene(Scene):
    """Defeat autopsy scene recording metrics and providing retry flow."""

    def __init__(self, engine: Any):
        super().__init__(engine)
        self.stats: Dict[str, Any] = {}
        self.episode_number: int = 1
        self.session_id: str = ""

    def on_enter(self, context: Dict[str, Any]) -> None:
        self.stats = context.get("stats", {})
        self.episode_number = context.get("episode_number", 1)
        self.session_id = context.get("session_id", "")

    def on_exit(self) -> Dict[str, Any]:
        return {"session_id": self.session_id, "episode_number": self.episode_number}

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_r, pygame.K_SPACE):
                self.engine.scene_manager.switch_to("playing", episode_number=self.episode_number, session_id=self.session_id)
            elif event.key == pygame.K_t:
                self.engine.scene_manager.switch_to("title")

    def update(self, dt: float) -> None:
        pass

    def render(self, surface: pygame.Surface) -> None:
        try:
            font_large = pygame.font.SysFont("Helvetica", 50, bold=True)
            font_sub = pygame.font.SysFont("Courier", 18)
        except Exception:
            font_large = pygame.font.Font(None, 52)
            font_sub = pygame.font.Font(None, 20)

        # Defeat Banner
        banner = font_large.render("EPISODE FAILED", True, config.PALETTE["red"])
        surface.blit(banner, (config.CANVAS_WIDTH // 2 - banner.get_width() // 2, 170))

        # Autopsy Card
        stats_lines = [
            f"Episode Attempt:    {self.episode_number}",
            f"Total Dashes:       {self.stats.get('dashes', 0)}",
            f"Weapons Discharged: {self.stats.get('shots_fired', 0)}",
            f"Enemies Defeated:   {self.stats.get('enemies_killed', 0)}",
            "Showrunner Status:  Autopsy recorded to SQLite WAL.",
            "                    Adaptive balancing queued.",
        ]
        card_rect = pygame.Rect(config.CANVAS_WIDTH // 2 - 260, 260, 520, 180)
        pygame.draw.rect(surface, config.PALETTE["dark_navy"], card_rect, border_radius=8)
        pygame.draw.rect(surface, config.PALETTE["red"], card_rect, width=2, border_radius=8)

        y = 280
        for line in stats_lines:
            txt = font_sub.render(line, True, config.PALETTE["white"])
            surface.blit(txt, (card_rect.x + 28, y))
            y += 24

        prompt = font_sub.render("[R / SPACE] Retry Episode      [T] Title Screen", True, config.PALETTE["yellow"])
        surface.blit(prompt, (config.CANVAS_WIDTH // 2 - prompt.get_width() // 2, 480))
