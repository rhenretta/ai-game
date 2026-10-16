"""engine/api/contracts.py
Version-pinned SDK contracts and base classes for autonomous game cartridges.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from engine.combat.damage import DamagePacket, Faction
from engine.combat.weapon import Weapon as BaseWeapon
from engine.core.scene import Scene as BaseScene
from engine.entities.base import Entity as BaseEntity, LivingEntity
from engine.entities.enemies import BaseEnemy
from engine.entities.hazard import Hazard as BaseHazard
from engine.entities.player import Player as BasePlayer
from engine.spatial.aabb import AABB, Vector2

ENGINE_API_VERSION = "1.1.0"


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
        deliberation: str = ""
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
            "hazards": self.hazards
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaseLevelSpec:
        delib = data.get("deliberation", "")
        return cls(
            level_id=data.get("level_id", "level_01"),
            name=data.get("name", "Generated Level"),
            width=float(data.get("width", 1280.0)),
            height=float(data.get("height", 720.0)),
            spawn_point=tuple(data.get("spawn_point", (120.0, 360.0))),
            exit_point=tuple(data.get("exit_point", (1160.0, 360.0))),
            walls=[tuple(w) for w in data.get("walls", [])],
            enemies=data.get("enemies", []),
            hazards=data.get("hazards", []),
            deliberation=delib
        )


class BaseAbility(ABC):
    """Abstract interface for character abilities, active skills, and gameplay mechanics."""
    deliberation: str = ""

    def __init__(
        self,
        name: str = "base_ability",
        cooldown: float = 1.0,
        stamina_cost: float = 10.0,
        deliberation: str = "",
    ):
        self.deliberation = deliberation
        self.name = name
        self.cooldown = float(cooldown)
        self.stamina_cost = float(stamina_cost)
        self.cooldown_timer = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes ability specification with deliberation first."""
        result: Dict[str, Any] = {}
        if self.deliberation:
            result["deliberation"] = self.deliberation
        result.update({
            "name": self.name,
            "cooldown": self.cooldown,
            "stamina_cost": self.stamina_cost,
        })
        return result

    def update(self, dt: float) -> None:
        if self.cooldown_timer > 0.0:
            self.cooldown_timer = max(0.0, self.cooldown_timer - dt)

    def can_activate(self, current_stamina: float) -> bool:
        return self.cooldown_timer <= 0.0 and current_stamina >= self.stamina_cost

    def activate(self, user: Any, target_pos: Optional[Tuple[float, float]] = None, **kwargs) -> bool:
        """Executes ability effect. Returns True if successfully triggered."""
        if not self.can_activate(getattr(user, "stamina", 0.0)):
            return False
        self.cooldown_timer = self.cooldown
        if hasattr(user, "stamina"):
            user.stamina = max(0.0, user.stamina - self.stamina_cost)
        return True


class BaseCartridge(ABC):
    """Abstract interface implemented by modular, hot-swappable game cartridges."""

    def __init__(
        self,
        manifest: Optional[Dict[str, Any]] = None,
        cartridge_dir: Optional[Union[str, Path]] = None,
        narrative: Optional[Dict[str, Any]] = None,
    ):
        self.manifest: Dict[str, Any] = manifest or {}
        self.cartridge_dir: Optional[Path] = Path(cartridge_dir).resolve() if cartridge_dir else None
        self.narrative: Dict[str, Any] = narrative or {}
        self.modules: Dict[str, Any] = {}
        self.assets_dir: Optional[Path] = (self.cartridge_dir / "assets") if self.cartridge_dir else None

    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        """Returns spatial geometry and spawn specifications if cartridge uses 2D spatial levels."""
        return BaseLevelSpec(level_id=f"level_{episode_number:02d}")

    def create_player(self, x: float, y: float, telemetry_recorder: Optional[Any] = None) -> Optional[BasePlayer]:
        """Instantiates the player entity if cartridge uses the standard player."""
        return None

    def create_enemies(self, spec: Optional[BaseLevelSpec] = None) -> List[BaseEnemy]:
        """Instantiates enemies configured in the level spec. Returns empty list by default for open-genre cartridges."""
        return []

    def create_hazards(self, spec: Optional[BaseLevelSpec] = None) -> List[BaseHazard]:
        """Instantiates hazards configured in the level spec. Returns empty list by default for open-genre cartridges."""
        return []

    # -------------------------------------------------------------------------
    # Open-Genre Scene & Lifecycle Contracts
    # -------------------------------------------------------------------------

    def get_scenes(self, engine: Any) -> Dict[str, BaseScene]:
        """Returns mapping of scene name to Scene instance for open-genre cartridges."""
        return {}

    def get_initial_scene(self, engine: Any) -> Optional[BaseScene]:
        """Returns primary initial scene to mount, or None to let engine determine initial scene."""
        return None

    def setup(self, engine: Any) -> None:
        """Optional lifecycle hook called when cartridge is loaded into engine."""
        pass

    def teardown(self) -> None:
        """Optional lifecycle hook called when cartridge is unloaded."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Returns cartridge metadata dictionary."""
        return self.manifest


    # -------------------------------------------------------------------------
    # Expanded Multi-Module & Multimodal Asset Resolution Contracts
    # -------------------------------------------------------------------------

    def get_narrative(self) -> Dict[str, Any]:
        """Returns the loaded narrative.json dictionary if available."""
        return self.narrative

    def get_characters(self) -> Dict[str, Dict[str, Any]]:
        """Returns dictionary of character profiles defined in narrative.json."""
        return self.narrative.get("characters", {})

    def get_dialogue_barks(self, trigger_event: str) -> List[Dict[str, Any]]:
        """Returns list of dialogue barks configured for the specified trigger event."""
        barks = self.narrative.get("dialogue_barks", {})
        if isinstance(barks, dict):
            return barks.get(trigger_event, [])
        elif isinstance(barks, list):
            return [b for b in barks if b.get("trigger") == trigger_event]
        return []

    def get_abilities(self) -> Dict[str, Any]:
        """Returns custom ability definitions or abilities module exports."""
        if "abilities" in self.modules:
            mod = self.modules["abilities"]
            return {name: getattr(mod, name) for name in dir(mod) if not name.startswith("_")}
        return {}

    def resolve_asset_path(self, relative_path: str) -> Optional[Path]:
        """Safely resolves an asset file path within the cartridge directory or assets subtree."""
        if not self.cartridge_dir:
            return None
        clean_rel = relative_path.lstrip("/")
        candidates = [
            self.cartridge_dir / clean_rel,
            self.cartridge_dir / "assets" / clean_rel,
            self.cartridge_dir / "assets" / "sprites" / clean_rel,
            self.cartridge_dir / "assets" / "audio" / clean_rel,
            self.cartridge_dir / "assets" / "tiles" / clean_rel,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        return None

    def get_character_portrait(self, character_id: str) -> Optional[Path]:
        """Resolves portrait image path for character from narrative.json."""
        char_info = self.get_characters().get(character_id, {})
        portrait_rel = char_info.get("portrait") or char_info.get("portrait_file")
        if portrait_rel:
            resolved = self.resolve_asset_path(portrait_rel)
            if resolved:
                return resolved
        return self.resolve_asset_path(f"portrait_{character_id}.png")

    def get_voice_audio(self, audio_key_or_path: str) -> Optional[Path]:
        """Resolves voice acting audio file path (.wav or .mp3)."""
        return self.resolve_asset_path(audio_key_or_path)
