"""engine/api/base_level.py
Canonical specification defining episodic spatial geometry, entity spawns, and hazards.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


class BaseLevelSpec:
    """Canonical specification defining episodic spatial geometry, entity spawns, and hazards."""
    deliberation: str = ""

    def __init__(
        self,
        level_id: str = "level_01",
        name: str = "Default Chamber",
        width: float = 1280.0,
        height: float = 720.0,
        spawn_point: Tuple[float, float] = (120.0, 360.0),
        exit_point: Tuple[float, float] = (1160.0, 360.0),
        walls: Optional[List[Tuple[float, float, float, float]]] = None,
        enemies: Optional[List[Dict[str, Any]]] = None,
        hazards: Optional[List[Dict[str, Any]]] = None,
        deliberation: str = "",
        episode_number: int = 1,
        **kwargs: Any,
    ):
        self.deliberation = deliberation
        self.level_id = level_id
        self.name = name
        self.width = float(width)
        self.height = float(height)
        self.spawn_point = spawn_point
        self.exit_point = exit_point
        self.walls = walls or []
        self.enemies = enemies or []
        self.hazards = hazards or []
        self.episode_number = int(episode_number)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.deliberation:
            result["deliberation"] = self.deliberation
        result.update({
            "level_id": self.level_id,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "spawn_point": list(self.spawn_point),
            "exit_point": list(self.exit_point),
            "walls": [list(w) for w in self.walls],
            "enemies": self.enemies,
            "hazards": self.hazards,
            "episode_number": self.episode_number,
        })
        return result
