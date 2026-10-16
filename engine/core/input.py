"""engine/core/input.py
Keyboard and Mouse Input Manager with High-DPI Retina coordinate normalization.
"""
from __future__ import annotations
import math
from typing import Any, Set, Tuple
import pygame
import config


class InputManager:
    """Tracks keyboard and mouse states, edge detections, and normalized canvas coordinates."""

    def __init__(self, engine: Any):
        self.engine = engine
        self.keys_down: Set[int] = set()
        self.keys_just_pressed: Set[int] = set()
        self.keys_just_released: Set[int] = set()

        self.mouse_buttons_down: Set[int] = set()
        self.mouse_just_pressed: Set[int] = set()
        self.mouse_just_released: Set[int] = set()

        self.raw_mouse_pos: Tuple[int, int] = (0, 0)
        self.canvas_mouse_pos: Tuple[float, float] = (config.CANVAS_WIDTH / 2, config.CANVAS_HEIGHT / 2)

    def process_event(self, event: pygame.event.Event) -> None:
        """Processes a single SDL event."""
        if event.type == pygame.KEYDOWN:
            self.keys_down.add(event.key)
            self.keys_just_pressed.add(event.key)
        elif event.type == pygame.KEYUP:
            self.keys_down.discard(event.key)
            self.keys_just_released.add(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.mouse_buttons_down.add(event.button)
            self.mouse_just_pressed.add(event.button)
            self.update_mouse_pos(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.mouse_buttons_down.discard(event.button)
            self.mouse_just_released.add(event.button)
            self.update_mouse_pos(event.pos)
        elif event.type == pygame.MOUSEMOTION:
            self.update_mouse_pos(event.pos)

    def update_mouse_pos(self, raw_pos: Tuple[int, int]) -> None:
        """Normalizes physical screen/window coordinates to 1280x720 canvas coordinates."""
        self.raw_mouse_pos = raw_pos
        if hasattr(self.engine, "screen_to_canvas_coords"):
            self.canvas_mouse_pos = self.engine.screen_to_canvas_coords(raw_pos[0], raw_pos[1])
        else:
            self.canvas_mouse_pos = (float(raw_pos[0]), float(raw_pos[1]))

    def clear_frame_edges(self) -> None:
        """Clears single-frame edge triggers. Call at the end of each frame."""
        self.keys_just_pressed.clear()
        self.keys_just_released.clear()
        self.mouse_just_pressed.clear()
        self.mouse_just_released.clear()

    def is_key_down(self, key: int) -> bool:
        return key in self.keys_down

    def is_key_just_pressed(self, key: int) -> bool:
        return key in self.keys_just_pressed

    def is_key_just_released(self, key: int) -> bool:
        return key in self.keys_just_released

    def is_mouse_down(self, button: int = 1) -> bool:
        return button in self.mouse_buttons_down

    def is_mouse_just_pressed(self, button: int = 1) -> bool:
        return button in self.mouse_just_pressed

    def get_movement_vector(self) -> Tuple[float, float]:
        """Returns normalized 8-directional WASD / Arrow movement vector (dx, dy)."""
        dx, dy = 0.0, 0.0
        if self.is_key_down(pygame.K_w) or self.is_key_down(pygame.K_UP):
            dy -= 1.0
        if self.is_key_down(pygame.K_s) or self.is_key_down(pygame.K_DOWN):
            dy += 1.0
        if self.is_key_down(pygame.K_a) or self.is_key_down(pygame.K_LEFT):
            dx -= 1.0
        if self.is_key_down(pygame.K_d) or self.is_key_down(pygame.K_RIGHT):
            dx += 1.0

        length = math.hypot(dx, dy)
        if length > 0.0:
            return (dx / length, dy / length)
        return (0.0, 0.0)

    def is_dash_triggered(self) -> bool:
        """True if Spacebar was just pressed this frame."""
        return self.is_key_just_pressed(pygame.K_SPACE)

    def is_fire_active(self) -> bool:
        """True if Left Mouse Button or J key is held down."""
        return self.is_mouse_down(1) or self.is_key_down(pygame.K_j)
