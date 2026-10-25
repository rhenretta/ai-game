"""engine/core/scene.py
Abstract Scene base class and SceneManager finite state machine.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import pygame


class Scene(ABC):
    """Abstract base class for all game scenes."""

    def __init__(self, engine: Any = None):
        self.engine = engine
        self.is_active: bool = False

    def on_enter(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Invoked when scene becomes active."""
        pass

    def on_exit(self) -> Dict[str, Any]:
        """Invoked when scene is transitioning out."""
        return {}

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle an SDL event."""
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        """Update simulation by fixed delta time dt."""
        pass

    @abstractmethod
    def render(self, surface: pygame.Surface) -> None:
        """Render scene onto 1280x720 canvas surface."""
        pass


class SceneManager:
    """Manages scene registration, transitions, and lifecycle dispatch."""

    def __init__(self, engine: Any):
        self.engine = engine
        self.scenes: Dict[str, Scene] = {}
        self.current_scene_name: Optional[str] = None
        self.current_scene: Optional[Scene] = None

    def register_scene(self, name: str, scene: Scene) -> None:
        """Register a scene instance under a unique name."""
        self.scenes[name] = scene

    def switch_to(self, name: str, **context) -> None:
        """Transitions from current scene to the named scene."""
        if name not in self.scenes:
            raise KeyError(f"Scene '{name}' is not registered.")

        exit_data = {}
        if self.current_scene is not None and hasattr(self.current_scene, "on_exit"):
            try:
                exit_data = self.current_scene.on_exit() or {}
            except TypeError:
                exit_data = {}
            self.current_scene.is_active = False

        combined_context = {**exit_data, **context}
        self.current_scene_name = name
        self.current_scene = self.scenes[name]
        self.current_scene.is_active = True
        if hasattr(self.current_scene, "on_enter"):
            try:
                self.current_scene.on_enter(combined_context)
            except TypeError:
                self.current_scene.on_enter()

    def handle_event(self, event: pygame.event.Event) -> None:
        """Forward event to current active scene."""
        if self.current_scene is not None and hasattr(self.current_scene, "handle_event"):
            self.current_scene.handle_event(event)

    def update(self, dt: float) -> None:
        """Forward fixed timestep update to current active scene."""
        if self.current_scene is not None:
            self.current_scene.update(dt)

    def render(self, surface: pygame.Surface) -> None:
        """Forward render call to current active scene."""
        if self.current_scene is not None:
            self.current_scene.render(surface)
