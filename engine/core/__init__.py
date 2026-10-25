"""engine/core package - engine lifecycle, loop, input, and scenes."""
from typing import Any
from engine.core.scene import Scene, SceneManager

__all__ = [
    "Engine",
    "GameLoop",
    "InputManager",
    "Scene",
    "SceneManager",
]


def __getattr__(name: str) -> Any:
    if name == "Engine":
        from engine.core.engine import Engine
        return Engine
    if name == "InputManager":
        from engine.core.input import InputManager
        return InputManager
    if name == "GameLoop":
        from engine.core.loop import GameLoop
        return GameLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
