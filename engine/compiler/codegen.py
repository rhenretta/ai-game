"""engine/compiler/codegen.py
Programmatic code synthesis engine for Autonomous Game Engine & Showrunner SWE Compiler.
Generates valid, executable Python and JSON source files conforming to Engine SDK contracts
and strict Rationale-First Epistemic Architecture.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from engine.director.models import (
    EnemyArchetypeSpec,
    EpisodicUpdateSpec,
    GenesisSpec,
    HazardArchetypeSpec,
    LevelDimensionsSpec,
    PlayerParametersSpec,
    WeaponArchetypeSpec,
)


def generate_manifest_json(spec: Union[GenesisSpec, EpisodicUpdateSpec]) -> str:
    """Generates manifest.json text with deliberation strictly as the first key."""
    if isinstance(spec, GenesisSpec):
        data: Dict[str, Any] = {
            "deliberation": spec.deliberation,
            "name": spec.game_title,
            "version": "1.0.0",
            "author": "Autonomous Game Engine & Showrunner",
            "description": f"Episode 0 Genesis Cartridge autonomously compiled for {spec.core_genre} ({spec.palette_mood}).",
            "spec_id": spec.spec_id,
            "target_episode": 0,
            "core_genre": spec.core_genre,
            "palette_mood": spec.palette_mood,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    else:
        data = {
            "deliberation": spec.deliberation,
            "name": f"Episode {spec.target_episode_number:02d} Cartridge",
            "version": f"1.{spec.target_episode_number}.0",
            "author": "Autonomous Game Engine & Showrunner",
            "description": f"Episode {spec.target_episode_number} Cartridge adapted for {spec.player_archetype} mastery={spec.mastery_score:.2f}.",
            "spec_id": spec.spec_id,
            "target_episode": spec.target_episode_number,
            "player_archetype": spec.player_archetype,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    return json.dumps(data, indent=2)


def generate_level_json(spec: Union[GenesisSpec, EpisodicUpdateSpec]) -> str:
    """Generates level.json text conforming to BaseLevelSpec with deliberation first."""
    if isinstance(spec, GenesisSpec):
        width = float(spec.level_dimensions.width)
        height = float(spec.level_dimensions.height)
        wt = float(spec.level_dimensions.wall_thickness)

        # 1. Boundary walls
        walls: List[List[float]] = [
            [0.0, 0.0, width, wt],                    # Top
            [0.0, height - wt, width, wt],             # Bottom
            [0.0, 0.0, wt, height],                    # Left
            [width - wt, 0.0, wt, height],             # Right
        ]

        # 2. Interior pillars based on pillar_count and arena_layout_type
        if spec.level_dimensions.pillar_count >= 4:
            walls.extend([
                [360.0, 160.0, 48.0, 160.0],
                [360.0, 400.0, 48.0, 160.0],
                [780.0, 160.0, 48.0, 160.0],
                [780.0, 400.0, 48.0, 160.0],
            ])
        elif spec.level_dimensions.pillar_count >= 2:
            walls.extend([
                [500.0, 200.0, 48.0, 320.0],
                [732.0, 200.0, 48.0, 320.0],
            ])

        # 3. Enemies placement
        enemies: List[Dict[str, Any]] = []
        spawn_x, spawn_y = spec.level_dimensions.spawn_point
        region_coords = {
            "mid_right": (width * 0.65, height * 0.50),
            "top_right": (width * 0.75, height * 0.25),
            "bottom_right": (width * 0.75, height * 0.75),
            "center": (width * 0.50, height * 0.50),
            "left_flank": (width * 0.35, height * 0.30),
        }
        for arch in spec.enemy_specs:
            for i in range(max(1, arch.count)):
                reg = arch.spawn_regions[i % len(arch.spawn_regions)] if arch.spawn_regions else "mid_right"
                base_x, base_y = region_coords.get(reg, (width * 0.70, height * 0.50))
                offset_y = (i - (arch.count - 1) / 2.0) * 80.0
                enemies.append({
                    "id": f"{arch.enemy_id}_{i+1}",
                    "type": arch.archetype,
                    "x": round(base_x, 1),
                    "y": round(max(wt + 60.0, min(height - wt - 60.0, base_y + offset_y)), 1),
                    "speed": round(arch.speed, 1),
                    "max_health": round(arch.max_health, 1),
                    "contact_damage": round(arch.contact_damage, 1),
                })

        # 4. Hazards placement
        hazards: List[Dict[str, Any]] = []
        for h_idx, h_spec in enumerate(spec.hazard_specs):
            for j in range(max(1, h_spec.count)):
                hid = f"{h_spec.hazard_id}_{j+1}"
                if h_spec.hazard_type == "spikes":
                    hazards.append({
                        "id": hid,
                        "type": "spikes",
                        "x": 450.0 + j * 120.0,
                        "y": 336.0,
                        "w": 48.0,
                        "h": 48.0,
                        "damage": round(h_spec.damage, 1),
                    })
                elif h_spec.hazard_type == "floor_acid":
                    hazards.append({
                        "id": hid,
                        "type": "floor_acid",
                        "x": 600.0 + j * 140.0,
                        "y": 312.0,
                        "w": 96.0,
                        "h": 96.0,
                        "damage": round(h_spec.damage, 1),
                    })
                elif h_spec.hazard_type == "laser_sweep":
                    hazards.append({
                        "id": hid,
                        "type": "laser_sweep",
                        "x": 780.0,
                        "y": 360.0,
                        "length": 280.0,
                        "mid_angle": 3.14159,
                        "sweep_amplitude": 1.1,
                        "speed": 1.2,
                        "damage": round(h_spec.damage, 1),
                    })

        data: Dict[str, Any] = {
            "deliberation": spec.level_dimensions.deliberation or spec.deliberation,
            "level_id": "level_00",
            "name": f"{spec.game_title} - Genesis Chamber",
            "width": width,
            "height": height,
            "spawn_point": list(spec.level_dimensions.spawn_point),
            "exit_point": list(spec.level_dimensions.exit_point),
            "walls": walls,
            "enemies": enemies,
            "hazards": hazards,
        }
    else:
        # EpisodicUpdateSpec
        cfg = spec.updated_episode_config
        walls = [list(w) for w in cfg.walls]
        if not walls:
            walls = [
                [0.0, 0.0, 1280.0, 32.0],
                [0.0, 688.0, 1280.0, 32.0],
                [0.0, 0.0, 32.0, 720.0],
                [1248.0, 0.0, 32.0, 720.0],
                [360.0, 160.0, 48.0, 160.0],
                [360.0, 400.0, 48.0, 160.0],
                [780.0, 160.0, 48.0, 160.0],
                [780.0, 400.0, 48.0, 160.0],
            ]

        enemies = list(cfg.enemies)
        if not enemies:
            enemies = [
                {
                    "id": "chaser_1",
                    "type": "chaser",
                    "x": 550.0,
                    "y": 240.0,
                    "speed": round(cfg.enemy_speed, 1),
                    "max_health": round(cfg.enemy_health, 1),
                    "contact_damage": round(cfg.enemy_damage, 1),
                },
                {
                    "id": "chaser_2",
                    "type": "chaser",
                    "x": 550.0,
                    "y": 480.0,
                    "speed": round(cfg.enemy_speed, 1),
                    "max_health": round(cfg.enemy_health, 1),
                    "contact_damage": round(cfg.enemy_damage, 1),
                },
                {
                    "id": "ranged_1",
                    "type": "ranged",
                    "x": 950.0,
                    "y": 360.0,
                    "speed": round(cfg.ranged_enemy_speed, 1),
                    "max_health": 50.0,
                },
            ]

        hazards = list(cfg.hazards)
        if not hazards:
            hazards = [
                {
                    "id": "spikes_1",
                    "type": "spikes",
                    "x": 450.0,
                    "y": 336.0,
                    "w": 48.0,
                    "h": 48.0,
                    "damage": round(cfg.hazard_damage, 1),
                },
            ]

        data = {
            "deliberation": spec.deliberation,
            "level_id": spec.target_episode_id,
            "name": f"Chamber {spec.target_episode_number:02d} - Adapted",
            "width": 1280.0,
            "height": 720.0,
            "spawn_point": [120.0, 360.0],
            "exit_point": [1160.0, 360.0],
            "walls": walls,
            "enemies": enemies,
            "hazards": hazards,
        }
    return json.dumps(data, indent=2)


def generate_entities_code(spec: Union[GenesisSpec, EpisodicUpdateSpec]) -> str:
    """Synthesizes executable entities.py Python source code inheriting from Base contracts."""
    if isinstance(spec, GenesisSpec):
        pp = spec.player_params
        wp = spec.weapon_spec
        chaser = next((e for e in spec.enemy_specs if e.archetype == "chaser"), None)
        ranged = next((e for e in spec.enemy_specs if e.archetype == "ranged"), None)
        c_speed = chaser.speed if chaser else 185.0
        c_hp = chaser.max_health if chaser else 60.0
        c_dmg = chaser.contact_damage if chaser else 15.0
        r_speed = ranged.speed if ranged else 135.0
        r_hp = ranged.max_health if ranged else 50.0
        p_walk = pp.walk_speed
        p_dash = pp.dash_speed
        p_dash_cd = pp.dash_cooldown
        p_dash_cost = pp.dash_cost
        p_max_hp = pp.max_health
        p_max_stm = pp.max_stamina
        p_stm_regen = pp.stamina_regen_rate
        wp_archetype = wp.archetype
        wp_rate = wp.fire_rate
        wp_dmg = wp.damage
        wp_spd = wp.projectile_speed
        class_prefix = "Genesis"
    else:
        cfg = spec.updated_episode_config
        c_speed = cfg.enemy_speed
        c_hp = cfg.enemy_health
        c_dmg = cfg.enemy_damage
        r_speed = cfg.ranged_enemy_speed
        r_hp = 50.0
        p_walk = 260.0
        p_dash = 650.0
        p_dash_cd = 0.60
        p_dash_cost = 25.0
        p_max_hp = 100.0
        p_max_stm = 100.0
        p_stm_regen = 35.0
        wp_archetype = "blaster"
        wp_rate = 0.22
        wp_dmg = cfg.weapon_damage
        wp_spd = 750.0
        class_prefix = f"Episode{spec.target_episode_number:02d}"

    weapon_setup = f'''
        if "{wp_archetype}" == "spread":
            self.weapon = SpreadWeapon()
            self.weapon.damage = {wp_dmg}
            self.weapon.fire_rate = {wp_rate}
        else:
            self.weapon = BlasterWeapon()
            self.weapon.damage = {wp_dmg}
            self.weapon.fire_rate = {wp_rate}
            self.weapon.projectile_speed = {wp_spd}'''

    return f'''"""cartridges/generated/entities.py
Dynamically compiled entity declarations synthesized by Autonomous SWE Engine Compiler.
Inherits authentically from BasePlayer, BaseEnemy, and BaseHazard.
"""
from __future__ import annotations
import math
from typing import Any, List, Optional, Tuple, Union
from engine.api.contracts import BaseEnemy, BaseHazard, BasePlayer
from engine.combat.damage import DamagePacket, Faction
from engine.combat.weapon import Weapon, BlasterWeapon, SpreadWeapon
from engine.entities.enemies import ChaserEnemy, RangedEnemy
from engine.entities.hazard import FloorAcidHazard, LaserSweepHazard, SpikeHazard
from engine.entities.player import Player
from engine.spatial.aabb import AABB, Vector2
from engine.spatial.collision import move_and_resolve_walls


class {class_prefix}Player(Player):
    deliberation: str = "Player entity tuned to kinetic and combat parameters."

    def __init__(self, x: float = 120.0, y: float = 360.0, telemetry_recorder: Optional[Any] = None):
        super().__init__(x=x, y=y, telemetry_recorder=telemetry_recorder)
        self.walk_speed = {p_walk}
        self.dash_speed = {p_dash}
        self.dash_cooldown = {p_dash_cd}
        self.dash_cost = {p_dash_cost}
        self.max_health = {p_max_hp}
        self.health = {p_max_hp}
        self.max_stamina = {p_max_stm}
        self.stamina = {p_max_stm}
        self.stamina_regen_rate = {p_stm_regen}{weapon_setup}


class {class_prefix}ChaserEnemy(ChaserEnemy):
    deliberation: str = "Chaser enemy tuned to aggressive tracking and contact pressure."

    def __init__(
        self,
        entity_id: str,
        x: float,
        y: float,
        speed: float = {c_speed},
        max_health: float = {c_hp},
        contact_damage: float = {c_dmg}
    ):
        super().__init__(entity_id=entity_id, x=x, y=y)
        self.speed = float(speed)
        self.max_health = float(max_health)
        self.health = float(max_health)
        self.contact_damage = float(contact_damage)


class {class_prefix}RangedEnemy(RangedEnemy):
    deliberation: str = "Ranged enemy maintaining standoff perimeter."

    def __init__(
        self,
        entity_id: str,
        x: float,
        y: float,
        speed: float = {r_speed},
        max_health: float = {r_hp}
    ):
        super().__init__(entity_id=entity_id, x=x, y=y)
        self.speed = float(speed)
        self.max_health = float(max_health)
        self.health = float(max_health)


class {class_prefix}SpikeHazard(SpikeHazard):
    deliberation: str = "Ground spike hazard cycling between warning and extension."

    def __init__(self, hazard_id: str, x: float, y: float, w: float = 48.0, h: float = 48.0, damage: float = 15.0):
        super().__init__(hazard_id=hazard_id, x=x, y=y, w=w, h=h)
        self.damage = float(damage)


class {class_prefix}FloorAcidHazard(FloorAcidHazard):
    deliberation: str = "Acid hazard applying damage ticks to occupying actors."

    def __init__(self, hazard_id: str, x: float, y: float, w: float = 96.0, h: float = 96.0, damage: float = 12.0):
        super().__init__(hazard_id=hazard_id, x=x, y=y, w=w, h=h)
        self.damage = float(damage)


class {class_prefix}LaserSweepHazard(LaserSweepHazard):
    deliberation: str = "Angular sweeping laser beam dividing arena lanes."

    def __init__(
        self,
        hazard_id: str,
        x: float,
        y: float,
        length: float = 280.0,
        mid_angle: float = 3.14159,
        sweep_amplitude: float = 1.1,
        speed: float = 1.2,
        damage: float = 25.0
    ):
        super().__init__(
            hazard_id=hazard_id,
            origin_x=x,
            origin_y=y,
            length=length,
            mid_angle=mid_angle,
            sweep_amplitude=sweep_amplitude,
            speed=speed
        )
        self.damage = float(damage)


__all__ = [
    "{class_prefix}Player",
    "{class_prefix}ChaserEnemy",
    "{class_prefix}RangedEnemy",
    "{class_prefix}SpikeHazard",
    "{class_prefix}FloorAcidHazard",
    "{class_prefix}LaserSweepHazard",
]
'''


def generate_mechanics_code(spec: Union[GenesisSpec, EpisodicUpdateSpec]) -> str:
    """Synthesizes executable mechanics.py source code implementing BaseCartridge."""
    class_prefix = "Genesis" if isinstance(spec, GenesisSpec) else f"Episode{spec.target_episode_number:02d}"
    cartridge_name = f"{class_prefix}Cartridge"
    deliberation_json = json.dumps(spec.deliberation)

    if isinstance(spec, GenesisSpec):
        pp = spec.player_params
        wp = spec.weapon_spec
        chaser = next((e for e in spec.enemy_specs if e.archetype == "chaser"), None)
        ranged = next((e for e in spec.enemy_specs if e.archetype == "ranged"), None)
        c_speed = chaser.speed if chaser else 185.0
        c_hp = chaser.max_health if chaser else 60.0
        c_dmg = chaser.contact_damage if chaser else 15.0
        r_speed = ranged.speed if ranged else 135.0
        r_hp = ranged.max_health if ranged else 50.0
        p_walk = pp.walk_speed
        p_dash = pp.dash_speed
        p_dash_cd = pp.dash_cooldown
        p_dash_cost = pp.dash_cost
        p_max_hp = pp.max_health
        p_max_stm = pp.max_stamina
        p_stm_regen = pp.stamina_regen_rate
        wp_archetype = wp.archetype
        wp_rate = wp.fire_rate
        wp_dmg = wp.damage
        wp_spd = wp.projectile_speed
    else:
        cfg = spec.updated_episode_config
        c_speed = cfg.enemy_speed
        c_hp = cfg.enemy_health
        c_dmg = cfg.enemy_damage
        r_speed = cfg.ranged_enemy_speed
        r_hp = 50.0
        p_walk = 260.0
        p_dash = 650.0
        p_dash_cd = 0.60
        p_dash_cost = 25.0
        p_max_hp = 100.0
        p_max_stm = 100.0
        p_stm_regen = 35.0
        wp_archetype = "blaster"
        wp_rate = 0.22
        wp_dmg = cfg.weapon_damage
        wp_spd = 750.0

    weapon_setup = f'''
        if "{wp_archetype}" == "spread":
            self.weapon = SpreadWeapon()
            self.weapon.damage = {wp_dmg}
            self.weapon.fire_rate = {wp_rate}
        else:
            self.weapon = BlasterWeapon()
            self.weapon.damage = {wp_dmg}
            self.weapon.fire_rate = {wp_rate}
            self.weapon.projectile_speed = {wp_spd}'''

    return f'''"""cartridges/generated/mechanics.py
Dynamically compiled cartridge implementation synthesized by Autonomous SWE Engine Compiler.
Inherits from BaseCartridge and dynamically loads level geometry and entity factories.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from engine.api.contracts import BaseCartridge, BaseEnemy, BaseHazard, BaseLevelSpec, BasePlayer
from engine.combat.damage import DamagePacket, Faction
from engine.combat.weapon import Weapon, BlasterWeapon, SpreadWeapon
from engine.entities.enemies import ChaserEnemy, RangedEnemy
from engine.entities.hazard import FloorAcidHazard, LaserSweepHazard, SpikeHazard
from engine.entities.player import Player
from engine.spatial.aabb import AABB, Vector2
from engine.spatial.collision import move_and_resolve_walls


class {class_prefix}Player(Player):
    deliberation: str = "Player entity tuned to kinetic and combat parameters."

    def __init__(self, x: float = 120.0, y: float = 360.0, telemetry_recorder: Optional[Any] = None):
        super().__init__(x=x, y=y, telemetry_recorder=telemetry_recorder)
        self.walk_speed = {p_walk}
        self.dash_speed = {p_dash}
        self.dash_cooldown = {p_dash_cd}
        self.dash_cost = {p_dash_cost}
        self.max_health = {p_max_hp}
        self.health = {p_max_hp}
        self.max_stamina = {p_max_stm}
        self.stamina = {p_max_stm}
        self.stamina_regen_rate = {p_stm_regen}{weapon_setup}


class {class_prefix}ChaserEnemy(ChaserEnemy):
    deliberation: str = "Chaser enemy tuned to aggressive tracking and contact pressure."

    def __init__(
        self,
        entity_id: str,
        x: float,
        y: float,
        speed: float = {c_speed},
        max_health: float = {c_hp},
        contact_damage: float = {c_dmg}
    ):
        super().__init__(entity_id=entity_id, x=x, y=y)
        self.speed = float(speed)
        self.max_health = float(max_health)
        self.health = float(max_health)
        self.contact_damage = float(contact_damage)


class {class_prefix}RangedEnemy(RangedEnemy):
    deliberation: str = "Ranged enemy maintaining standoff perimeter."

    def __init__(
        self,
        entity_id: str,
        x: float,
        y: float,
        speed: float = {r_speed},
        max_health: float = {r_hp}
    ):
        super().__init__(entity_id=entity_id, x=x, y=y)
        self.speed = float(speed)
        self.max_health = float(max_health)
        self.health = float(max_health)


class {class_prefix}SpikeHazard(SpikeHazard):
    deliberation: str = "Ground spike hazard cycling between warning and extension."

    def __init__(self, hazard_id: str, x: float, y: float, w: float = 48.0, h: float = 48.0, damage: float = 15.0):
        super().__init__(hazard_id=hazard_id, x=x, y=y, w=w, h=h)
        self.damage = float(damage)


class {class_prefix}FloorAcidHazard(FloorAcidHazard):
    deliberation: str = "Acid hazard applying damage ticks to occupying actors."

    def __init__(self, hazard_id: str, x: float, y: float, w: float = 96.0, h: float = 96.0, damage: float = 12.0):
        super().__init__(hazard_id=hazard_id, x=x, y=y, w=w, h=h)
        self.damage = float(damage)


class {class_prefix}LaserSweepHazard(LaserSweepHazard):
    deliberation: str = "Angular sweeping laser beam dividing arena lanes."

    def __init__(
        self,
        hazard_id: str,
        x: float,
        y: float,
        length: float = 280.0,
        mid_angle: float = 3.14159,
        sweep_amplitude: float = 1.1,
        speed: float = 1.2,
        damage: float = 25.0
    ):
        super().__init__(
            hazard_id=hazard_id,
            origin_x=x,
            origin_y=y,
            length=length,
            mid_angle=mid_angle,
            sweep_amplitude=sweep_amplitude,
            speed=speed
        )
        self.damage = float(damage)


class {cartridge_name}(BaseCartridge):
    deliberation: str = {deliberation_json}

    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        super().__init__(manifest)
        self.level_spec: Optional[BaseLevelSpec] = None
        self._load_level_json()

    def _load_level_json(self) -> None:
        level_file = Path(__file__).resolve().parent / "level.json"
        if level_file.exists():
            with open(level_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.level_spec = BaseLevelSpec.from_dict(data)

    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        if self.level_spec is not None:
            return self.level_spec
        return BaseLevelSpec(
            level_id="level_01",
            name="Default Generated Chamber",
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            deliberation=self.deliberation
        )

    def create_player(self, x: float, y: float, telemetry_recorder: Optional[Any] = None) -> BasePlayer:
        return {class_prefix}Player(x, y, telemetry_recorder)

    def create_enemies(self, spec: BaseLevelSpec) -> List[BaseEnemy]:
        enemies: List[BaseEnemy] = []
        for idx, e_def in enumerate(spec.enemies):
            e_type = str(e_def.get("type", "chaser")).lower()
            ex = float(e_def.get("x", 400.0))
            ey = float(e_def.get("y", 300.0))
            eid = str(e_def.get("id", f"{{e_type}}_{{idx + 1}}"))
            speed = float(e_def.get("speed", 185.0))
            max_hp = float(e_def.get("max_health", 60.0))
            dmg = float(e_def.get("contact_damage", 15.0))
            if e_type == "chaser":
                enemies.append({class_prefix}ChaserEnemy(eid, ex, ey, speed=speed, max_health=max_hp, contact_damage=dmg))
            elif e_type == "ranged":
                enemies.append({class_prefix}RangedEnemy(eid, ex, ey, speed=speed, max_health=max_hp))
        return enemies

    def create_hazards(self, spec: BaseLevelSpec) -> List[BaseHazard]:
        hazards: List[BaseHazard] = []
        for idx, h_def in enumerate(spec.hazards):
            h_type = str(h_def.get("type", "spikes")).lower()
            hx = float(h_def.get("x", 400.0))
            hy = float(h_def.get("y", 300.0))
            hid = str(h_def.get("id", f"{{h_type}}_{{idx + 1}}"))
            dmg = float(h_def.get("damage", 15.0))
            if h_type == "spikes":
                hazards.append({class_prefix}SpikeHazard(hid, hx, hy, float(h_def.get("w", 48.0)), float(h_def.get("h", 48.0)), damage=dmg))
            elif h_type == "floor_acid":
                hazards.append({class_prefix}FloorAcidHazard(hid, hx, hy, float(h_def.get("w", 96.0)), float(h_def.get("h", 96.0)), damage=dmg))
            elif h_type == "laser_sweep":
                hazards.append({class_prefix}LaserSweepHazard(
                    hid, hx, hy,
                    length=float(h_def.get("length", 280.0)),
                    mid_angle=float(h_def.get("mid_angle", 3.14159)),
                    sweep_amplitude=float(h_def.get("sweep_amplitude", 1.1)),
                    speed=float(h_def.get("speed", 1.2)),
                    damage=dmg
                ))
        return hazards
'''
