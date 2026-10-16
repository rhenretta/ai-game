"""engine/core/engine.py
Core Engine managing Pygame subsystems, High-DPI scaling, and scene dispatch.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional, Tuple
import pygame
import config
from engine.core.input import InputManager
from engine.core.loop import GameLoop
from engine.core.scene import SceneManager


class Engine:
    """The central game engine orchestrating windowing, scaling, and subsystems."""

    def __init__(self, headless: bool = False, title: str = config.WINDOW_TITLE):
        self.headless = headless or (os.environ.get("SDL_VIDEODRIVER") == "dummy")
        self.title = title
        self.is_running: bool = False
        self.telemetry_recorder: Optional[Any] = None

        # Display surfaces
        self.window_surface: Optional[pygame.Surface] = None
        self.logical_surface: Optional[pygame.Surface] = None
        self.window_size: Tuple[int, int] = config.WINDOW_SIZE
        self.canvas_size: Tuple[int, int] = config.CANVAS_SIZE

        # Subsystems
        self.input_manager: InputManager = InputManager(self)
        self.scene_manager: SceneManager = SceneManager(self)
        self.game_loop: GameLoop = GameLoop(self, target_fps=config.TARGET_FPS)

        self.init_subsystems()

    def init_subsystems(self) -> None:
        """Initializes Pygame, font, audio, and display surfaces."""
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"

        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        pygame.init()
        pygame.font.init()

        # Audio initialization with graceful degradation
        if config.AUDIO_ENABLED and not self.headless:
            try:
                pygame.mixer.init(frequency=config.SAMPLE_RATE, channels=config.AUDIO_CHANNELS, buffer=config.AUDIO_BUFFER_SIZE)
                pygame.mixer.set_reserved(1)
            except pygame.error:
                pass

        if not self.headless:
            pygame.display.set_caption(self.title)
            flags = pygame.DOUBLEBUF | pygame.RESIZABLE
            try:
                self.window_surface = pygame.display.set_mode(self.window_size, flags | pygame.SCALED, vsync=1)
            except pygame.error:
                self.window_surface = pygame.display.set_mode(self.window_size, flags)
        else:
            try:
                self.window_surface = pygame.display.set_mode(config.CANVAS_SIZE)
            except Exception:
                self.window_surface = pygame.Surface(config.CANVAS_SIZE)

        # Create the 1280x720 logical canvas
        try:
            self.logical_surface = pygame.Surface(config.CANVAS_SIZE, flags=pygame.SRCALPHA).convert_alpha()
        except pygame.error:
            self.logical_surface = pygame.Surface(config.CANVAS_SIZE, flags=pygame.SRCALPHA)

        self.active_cartridge: Optional[Any] = None
        self.booted_cartridge_path: Optional[Path] = None
        self.steam_profile_tag: Optional[str] = None
        self.steam_dna: Optional[Any] = None

    def set_telemetry_recorder(self, recorder: Any) -> None:
        """Injects the continuous SQLite WAL telemetry recorder."""
        self.telemetry_recorder = recorder

    def load_cartridge(self, cartridge_or_path: Any) -> Any:
        """Dynamically loads and hot-swaps game cartridge into engine."""
        from engine.loader import BaseCartridge, CartridgeLoader
        if isinstance(cartridge_or_path, BaseCartridge):
            self.active_cartridge = cartridge_or_path
            self.booted_cartridge_path = getattr(cartridge_or_path, "cartridge_dir", None)
        else:
            self.active_cartridge = CartridgeLoader.load_cartridge(cartridge_or_path)
            self.booted_cartridge_path = Path(cartridge_or_path).resolve()
        return self.active_cartridge

    def screen_to_canvas_coords(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Normalizes physical screen/window coordinates to 1280x720 logical canvas space."""
        if self.headless or self.window_surface is None:
            return float(screen_x), float(screen_y)

        win_w, win_h = self.window_surface.get_size()
        can_w, can_h = config.CANVAS_WIDTH, config.CANVAS_HEIGHT

        scale = min(win_w / can_w, win_h / can_h)
        if scale <= 0.0:
            return 0.0, 0.0

        viewport_w = can_w * scale
        viewport_h = can_h * scale
        offset_x = (win_w - viewport_w) / 2.0
        offset_y = (win_h - viewport_h) / 2.0

        canvas_x = (screen_x - offset_x) / scale
        canvas_y = (screen_y - offset_y) / scale

        # Clamp to canvas boundaries
        canvas_x = max(0.0, min(float(can_w), canvas_x))
        canvas_y = max(0.0, min(float(can_h), canvas_y))

        return canvas_x, canvas_y

    def run(self) -> None:
        """Runs the main 60 FPS game loop."""
        self.is_running = True
        self.game_loop.run()

    def quit(self) -> None:
        """Clean shutdown of engine and Pygame."""
        self.is_running = False
        if self.telemetry_recorder and hasattr(self.telemetry_recorder, "shutdown"):
            self.telemetry_recorder.shutdown()
        pygame.quit()


# Canonical alias
GameEngine = Engine
