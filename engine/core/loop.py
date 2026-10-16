"""engine/core/loop.py
60 FPS deterministic GameLoop with Cocoa main-thread event pump and headless stepping.
"""
from __future__ import annotations
from typing import Any, Optional
import pygame
import config


class GameLoop:
    """Manages clock ticks, fixed delta time simulation, Cocoa event pumping, and presentation."""

    def __init__(self, engine: Any, target_fps: int = config.TARGET_FPS):
        self.engine = engine
        self.target_fps = int(target_fps)
        self.fixed_dt: float = config.FIXED_DT
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.frame_count: int = 0
        self.running_fps: float = float(target_fps)
        self.is_active: bool = False

    def run(self) -> None:
        """Main 60 FPS loop running on macOS Thread 0."""
        self.is_active = True
        while self.engine.is_running and self.is_active:
            # 1. Clock tick regulation
            self.clock.tick(self.target_fps)
            self.running_fps = self.clock.get_fps()

            # 2. Step one frame with fixed delta time
            keep_running = self.step(self.fixed_dt)
            if not keep_running:
                break

        self.engine.quit()

    def step(self, dt: Optional[float] = None) -> bool:
        """
        Executes a single discrete simulation and render step.
        Returns False if the engine received a QUIT event.
        Can be called repeatedly in headless tests at >1000 FPS without real-time delays.
        """
        step_dt = dt if dt is not None else self.fixed_dt
        self.frame_count += 1

        # 1. Cocoa Event Pump on Main OS Thread
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.engine.is_running = False
                return False
            elif event.type == pygame.VIDEORESIZE and not self.engine.headless:
                self.engine.window_size = (event.w, event.h)

            # Dispatch to input manager and active scene
            self.engine.input_manager.process_event(event)
            self.engine.scene_manager.handle_event(event)

        # 2. Fixed Timestep Simulation Update
        self.engine.scene_manager.update(step_dt)

        # 3. Canvas Rendering & Presentation
        self.render_and_present()

        # 4. Clear single-frame input edges
        self.engine.input_manager.clear_frame_edges()

        # 5. Tick telemetry frame if recorder present
        if self.engine.telemetry_recorder and hasattr(self.engine.telemetry_recorder, "tick_frame"):
            self.engine.telemetry_recorder.tick_frame()

        return True

    def render_and_present(self) -> None:
        """Renders current scene to 1280x720 canvas and presents it to the display."""
        canvas = self.engine.logical_surface
        if canvas is None:
            return

        # Clear canvas with dark base color
        canvas.fill(config.COLOR_BG)

        # Render active scene
        self.engine.scene_manager.render(canvas)

        # If running headlessly, skip physical window blit and flip
        if self.engine.headless:
            return

        # Blit/scale to native window surface
        win = self.engine.window_surface
        if win is not None:
            win_w, win_h = win.get_size()
            can_w, can_h = config.CANVAS_WIDTH, config.CANVAS_HEIGHT

            if win_w == can_w and win_h == can_h:
                win.blit(canvas, (0, 0))
            else:
                scale = min(win_w / can_w, win_h / can_h)
                scaled_w = int(can_w * scale)
                scaled_h = int(can_h * scale)
                offset_x = (win_w - scaled_w) // 2
                offset_y = (win_h - scaled_h) // 2

                scaled_surface = pygame.transform.smoothscale(canvas, (scaled_w, scaled_h))
                win.fill((0, 0, 0))
                win.blit(scaled_surface, (offset_x, offset_y))

            pygame.display.flip()
