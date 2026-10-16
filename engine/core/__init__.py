"""engine/core package - engine lifecycle, loop, input, and scenes."""
from engine.core.engine import Engine
from engine.core.input import InputManager
from engine.core.loop import GameLoop
from engine.core.scene import Scene, SceneManager

__all__ = [
    "Engine",
    "GameLoop",
    "InputManager",
    "Scene",
    "SceneManager",
]
