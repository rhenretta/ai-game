"""engine/director/models.py
Rationale-First data models for Steam Genesis and Episodic Showrunner Directors.
In accordance with Architectural Integrity Rule 4 and AGENTS.md, every model declares
deliberation as its very first property.
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from pydantic import BaseModel, Field


# ============================================================================
# Steam Genesis Spec Models (Feature 10 Part 1)
# ============================================================================

class LevelDimensionsSpec(BaseModel):
    """Spatial arena dimensions, boundary thickness, and layout structure."""
    deliberation: str = Field(
        default="",
        description="Epistemic rationale explaining spatial arena sizing and obstacle density."
    )
    width: float = Field(default=1280.0, description="Total canvas width in pixels")
    height: float = Field(default=720.0, description="Total canvas height in pixels")
    spawn_point: Tuple[float, float] = Field(default=(120.0, 360.0), description="Player initial spawn coordinates")
    exit_point: Tuple[float, float] = Field(default=(1160.0, 360.0), description="Exit portal coordinates")
    wall_thickness: float = Field(default=32.0, description="Thickness of boundary walls")
    pillar_count: int = Field(default=4, description="Count of interior defensive pillars")
    arena_layout_type: str = Field(default="open_arena", description="'open_arena' | 'tactical_corridors' | 'wide_expanse'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "width": self.width,
            "height": self.height,
            "spawn_point": list(self.spawn_point),
            "exit_point": list(self.exit_point),
            "wall_thickness": self.wall_thickness,
            "pillar_count": self.pillar_count,
            "arena_layout_type": self.arena_layout_type,
        }


class WeaponArchetypeSpec(BaseModel):
    """Specification for the player's initial starter weapon."""
    deliberation: str = Field(
        default="",
        description="Epistemic rationale explaining weapon archetype selection based on mechanical biases."
    )
    name: str = Field(default="blaster", description="Display name of the weapon")
    archetype: str = Field(default="blaster", description="'blaster' | 'spread' | 'sniper'")
    fire_rate: float = Field(default=0.22, description="Cooldown seconds between shots")
    damage: float = Field(default=25.0, description="Base damage per projectile")
    projectile_speed: float = Field(default=750.0, description="Speed of projectile in pixels per second")
    projectile_radius: float = Field(default=4.0, description="Hitbox radius of projectile")
    projectile_lifetime: float = Field(default=1.8, description="Maximum lifetime in seconds")
    spread_count: int = Field(default=1, description="Number of projectiles spawned per shot")
    range_distance: float = Field(default=600.0, description="Effective weapon range in pixels")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "name": self.name,
            "archetype": self.archetype,
            "fire_rate": round(self.fire_rate, 3),
            "damage": round(self.damage, 2),
            "projectile_speed": round(self.projectile_speed, 1),
            "projectile_radius": round(self.projectile_radius, 1),
            "projectile_lifetime": round(self.projectile_lifetime, 2),
            "spread_count": self.spread_count,
            "range_distance": round(self.range_distance, 1),
        }


class EnemyArchetypeSpec(BaseModel):
    """Specification for an initial enemy archetype spawn configuration."""
    deliberation: str = Field(
        default="",
        description="Epistemic rationale explaining enemy threat calibration and attack patterns."
    )
    enemy_id: str = Field(default="chaser_1", description="Unique entity identifier prefix")
    archetype: str = Field(default="chaser", description="'chaser' | 'ranged' | 'swarmer'")
    count: int = Field(default=2, description="Number of instances to spawn in Episode 0")
    max_health: float = Field(default=60.0, description="Maximum health pool")
    speed: float = Field(default=185.0, description="Movement speed in pixels per second")
    contact_damage: float = Field(default=15.0, description="Damage inflicted on physical contact")
    attack_rate: float = Field(default=0.80, description="Attack or fire interval in seconds")
    preferred_distance: float = Field(default=0.0, description="Standoff distance in pixels (0 for melee)")
    spawn_regions: List[str] = Field(
        default_factory=lambda: ["mid_right", "top_right"],
        description="Logical spatial regions for initial placement"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "enemy_id": self.enemy_id,
            "archetype": self.archetype,
            "count": self.count,
            "max_health": round(self.max_health, 1),
            "speed": round(self.speed, 1),
            "contact_damage": round(self.contact_damage, 1),
            "attack_rate": round(self.attack_rate, 2),
            "preferred_distance": round(self.preferred_distance, 1),
            "spawn_regions": self.spawn_regions,
        }


class HazardArchetypeSpec(BaseModel):
    """Specification for environmental hazards in the genesis chamber."""
    deliberation: str = Field(
        default="",
        description="Epistemic rationale explaining environmental hazard placement."
    )
    hazard_id: str = Field(default="hazard_1", description="Unique hazard identifier prefix")
    hazard_type: str = Field(default="spikes", description="'spikes' | 'floor_acid' | 'laser_sweep'")
    count: int = Field(default=1, description="Number of hazard instances to spawn")
    damage: float = Field(default=10.0, description="Damage applied per tick or contact")
    placement_hint: str = Field(default="center_lane", description="Spatial placement hint")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "hazard_id": self.hazard_id,
            "hazard_type": self.hazard_type,
            "count": self.count,
            "damage": round(self.damage, 1),
            "placement_hint": self.placement_hint,
        }


class PlayerParametersSpec(BaseModel):
    """Kinetic and physical parameters for the player entity."""
    deliberation: str = Field(
        default="",
        description="Epistemic rationale explaining player mobility tuning and stamina mechanics."
    )
    walk_speed: float = Field(default=260.0, description="Base 8-directional movement speed in px/s")
    dash_speed: float = Field(default=650.0, description="Impulse dash speed in px/s")
    dash_cooldown: float = Field(default=0.60, description="Cooldown in seconds between dashes")
    dash_duration: float = Field(default=0.20, description="Duration of dash state in seconds")
    dash_cost: float = Field(default=25.0, description="Stamina consumed per dash")
    jump_force: float = Field(default=420.0, description="Vertical impulse velocity for platforming extensions")
    max_health: float = Field(default=100.0, description="Starting player health")
    max_stamina: float = Field(default=100.0, description="Starting player stamina pool")
    stamina_regen_rate: float = Field(default=35.0, description="Stamina recovered per second")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "walk_speed": round(self.walk_speed, 1),
            "dash_speed": round(self.dash_speed, 1),
            "dash_cooldown": round(self.dash_cooldown, 3),
            "dash_duration": round(self.dash_duration, 3),
            "dash_cost": round(self.dash_cost, 1),
            "jump_force": round(self.jump_force, 1),
            "max_health": round(self.max_health, 1),
            "max_stamina": round(self.max_stamina, 1),
            "stamina_regen_rate": round(self.stamina_regen_rate, 1),
        }


class GenesisSpec(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation MUST be declared first!
    Complete initial game specification (Episode 0) synthesized from SteamPlayerDNA.
    """
    deliberation: str = Field(
        default="",
        description="Multi-sentence rationale-first epistemic deliberation explaining why the player's Steam library history dictates gameplay rules, physics, and themes."
    )
    spec_id: str = Field(default="genesis_spec_default", description="Unique specification identifier")
    player_id: str = Field(default="local_player", description="SteamID64 or local player profile identifier")
    source_profile_tag: str = Field(default="genesis_default", description="Source DNA profile tag")
    target_episode: int = Field(default=0, description="Always Episode 0 for Genesis")
    game_title: str = Field(default="Genesis: Episode 0", description="Generated title for the initial cartridge")
    core_genre: str = Field(default="Action Roguelike", description="Derived primary genre blend")
    palette_mood: str = Field(default="retro_arcade", description="'neon_cyberpunk' | 'grim_dark' | 'retro_arcade' | 'vibrant_fantasy'")
    palette_colors: Dict[str, str] = Field(default_factory=dict, description="Hex color mapping for asset synthesis")
    player_params: PlayerParametersSpec = Field(default_factory=PlayerParametersSpec)
    weapon_spec: WeaponArchetypeSpec = Field(default_factory=WeaponArchetypeSpec)
    enemy_specs: List[EnemyArchetypeSpec] = Field(default_factory=list)
    hazard_specs: List[HazardArchetypeSpec] = Field(default_factory=list)
    level_dimensions: LevelDimensionsSpec = Field(default_factory=LevelDimensionsSpec)
    target_difficulty: float = Field(default=1.0, description="Normalized initial difficulty baseline [0.5, 1.5]")
    camera_style: str = Field(default="arena_framing", description="'arena_framing' | 'tight_follow'")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "spec_id": self.spec_id,
            "player_id": self.player_id,
            "source_profile_tag": self.source_profile_tag,
            "target_episode": self.target_episode,
            "game_title": self.game_title,
            "core_genre": self.core_genre,
            "palette_mood": self.palette_mood,
            "palette_colors": self.palette_colors,
            "player_params": self.player_params.to_dict(),
            "weapon_spec": self.weapon_spec.to_dict(),
            "enemy_specs": [e.to_dict() for e in self.enemy_specs],
            "hazard_specs": [h.to_dict() for h in self.hazard_specs],
            "level_dimensions": self.level_dimensions.to_dict(),
            "target_difficulty": round(self.target_difficulty, 2),
            "camera_style": self.camera_style,
            "created_at_utc": self.created_at_utc,
        }


# ============================================================================
# Episodic Director Models (Feature 10 Part 2)
# ============================================================================

class DirectiveAction(str, Enum):
    """Categorical action types for episodic directives."""
    INTRODUCE_BEHAVIOR = "INTRODUCE_BEHAVIOR"
    TUNE_PARAMETER = "TUNE_PARAMETER"
    ADD_HAZARD_TYPE = "ADD_HAZARD_TYPE"
    BALANCE_WEAPON = "BALANCE_WEAPON"
    ADD_OBSTACLE = "ADD_OBSTACLE"
    ADD_PLATFORM = "ADD_PLATFORM"
    RELOCATE_SPAWN = "RELOCATE_SPAWN"
    EXPAND_CHOKEPOINT = "EXPAND_CHOKEPOINT"
    ADD_HAZARD_ZONE = "ADD_HAZARD_ZONE"


class DifficultyClampRecord(BaseModel):
    """Tracks between-run adaptation clamping for an individual numerical parameter."""
    deliberation: str = Field(
        default="",
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining the clamping decision."
    )
    parameter_name: str = Field(..., description="Name of the clamped parameter (e.g., enemy_speed).")
    previous_value: float = Field(..., description="Parameter value from the previous episode (p_old).")
    raw_proposed_value: float = Field(..., description="Unclamped proposed value from adaptive heuristics (p_raw).")
    clamped_value: float = Field(..., description="Final value enforced by the +/-15% adaptation clamp (p_new).")
    min_bound: float = Field(..., description="Lower bound (0.85 * p_old).")
    max_bound: float = Field(..., description="Upper bound (1.15 * p_old).")
    clamp_triggered: bool = Field(default=False, description="True if raw value violated the +/-15% boundary.")
    delta_percentage: float = Field(..., description="Effective percentage change ((p_new - p_old) / p_old) * 100.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "parameter_name": self.parameter_name,
            "previous_value": round(self.previous_value, 3),
            "raw_proposed_value": round(self.raw_proposed_value, 3),
            "clamped_value": round(self.clamped_value, 3),
            "min_bound": round(self.min_bound, 3),
            "max_bound": round(self.max_bound, 3),
            "clamp_triggered": self.clamp_triggered,
            "delta_percentage": round(self.delta_percentage, 2),
        }


class DifficultyClampState(BaseModel):
    """Enforces and records the Bounded Between-Run Adaptation Clamp (+/-15% max)."""
    deliberation: str = Field(
        default="",
        description="MANDATORY FIRST PROPERTY: Epistemic justification for aggregate difficulty adjustment."
    )
    max_adaptation_ratio: float = Field(
        default=0.15,
        description="Authoritative maximum between-run shift (0.15 = +/-15%)."
    )
    default_min_floor: float = Field(
        default=0.10,
        description="Minimum positive baseline floor preventing asymptotic collapse to zero."
    )
    default_max_cap: float = Field(
        default=1_000_000.0,
        description="Maximum sanity cap preventing runaway upward parameter scaling."
    )
    clamps: Dict[str, DifficultyClampRecord] = Field(
        default_factory=dict,
        description="Dictionary mapping parameter_name to DifficultyClampRecord."
    )

    def apply_clamp(
        self,
        param_name: str,
        p_old: float,
        p_raw: float,
        rationale: str = "",
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> float:
        """Applies strict +/-15% bounds on p_raw relative to p_old and records clamp state."""
        p_old_float = float(p_old)
        p_raw_float = float(p_raw)

        if math.isnan(p_old_float) or math.isinf(p_old_float) or p_old_float <= 0.0:
            clamped_val = p_raw_float
            min_b = p_raw_float
            max_b = p_raw_float
            clamp_active = False
            pct = 0.0
        elif math.isnan(p_raw_float):
            raw_min = p_old_float - (p_old_float * self.max_adaptation_ratio)
            raw_max = p_old_float + (p_old_float * self.max_adaptation_ratio)
            effective_min = min_val if min_val is not None else self.default_min_floor
            effective_max = max_val if max_val is not None else self.default_max_cap
            min_b = max(raw_min, min(effective_min, p_old_float))
            max_b = min(raw_max, max(effective_max, p_old_float))
            clamped_val = p_old_float
            clamp_active = True
            pct = 0.0
        else:
            raw_min = p_old_float - (p_old_float * self.max_adaptation_ratio)
            raw_max = p_old_float + (p_old_float * self.max_adaptation_ratio)
            effective_min = min_val if min_val is not None else self.default_min_floor
            effective_max = max_val if max_val is not None else self.default_max_cap
            min_b = max(raw_min, min(effective_min, p_old_float))
            max_b = min(raw_max, max(effective_max, p_old_float))
            if min_b > max_b:
                min_b = max_b
            clamp_active = (p_raw_float > max_b + 1e-9) or (p_raw_float < min_b - 1e-9)
            clamped_val = max(min_b, min(max_b, p_raw_float))
            pct = ((clamped_val - p_old_float) / p_old_float) * 100.0

        record = DifficultyClampRecord(
            deliberation=rationale or f"Clamped {param_name} to [{min_b:.2f}, {max_b:.2f}]",
            parameter_name=param_name,
            previous_value=p_old_float,
            raw_proposed_value=p_raw_float,
            clamped_value=clamped_val,
            min_bound=min_b,
            max_bound=max_b,
            clamp_triggered=clamp_active,
            delta_percentage=pct,
        )
        self.clamps[param_name] = record
        return clamped_val

    def get_summary_table_markdown(self) -> str:
        """Renders Markdown summary table of clamped parameters."""
        lines = [
            "| Parameter | Previous Value | Proposed Value | Clamped Value | Effective Delta | Clamp Active |",
            "|---|---|---|---|---|---|"
        ]
        for name, rec in self.clamps.items():
            clamp_str = "YES (CLAMPED)" if rec.clamp_triggered else "NO"
            lines.append(
                f"| `{name}` | {rec.previous_value:.2f} | {rec.raw_proposed_value:.2f} "
                f"| {rec.clamped_value:.2f} | {rec.delta_percentage:+.1f}% | {clamp_str} |"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "max_adaptation_ratio": self.max_adaptation_ratio,
            "clamps": {k: v.to_dict() for k, v in self.clamps.items()},
        }


class MechanicalUpdateDirective(BaseModel):
    """Directives modifying entity AI behaviors, weapon properties, or hazard parameters."""
    deliberation: str = Field(
        default="",
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining why this mechanical change is required."
    )
    directive_id: str = Field(
        default_factory=lambda: f"mech_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for the directive."
    )
    action: DirectiveAction = Field(
        default=DirectiveAction.TUNE_PARAMETER,
        description="Action type."
    )
    target_entity: str = Field(
        ...,
        description="Target entity archetype ('chaser', 'ranged', 'hazard_spikes', 'player_weapon', etc.)."
    )
    behavior_name: Optional[str] = Field(
        default=None,
        description="New behavioral routine to attach ('telegraphed_lunge', 'flanking_rush', 'mortar_lob')."
    )
    parameter_modifications: Dict[str, float] = Field(
        default_factory=dict,
        description="Clamped parameter updates (e.g., {'speed': 212.75, 'contact_damage': 17.25})."
    )
    priority: int = Field(default=1, description="Application order priority.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "directive_id": self.directive_id,
            "action": self.action.value if hasattr(self.action, "value") else str(self.action),
            "target_entity": self.target_entity,
            "behavior_name": self.behavior_name,
            "parameter_modifications": self.parameter_modifications,
            "priority": self.priority,
        }


class SpatialUpdateDirective(BaseModel):
    """Directives modifying room geometry, obstacles, platforms, or spawn coordinates."""
    deliberation: str = Field(
        default="",
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining geometry modification."
    )
    directive_id: str = Field(
        default_factory=lambda: f"spatial_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for the directive."
    )
    action: DirectiveAction = Field(
        default=DirectiveAction.ADD_OBSTACLE,
        description="Spatial action type."
    )
    geometry_type: str = Field(
        default="pillar",
        description="'pillar', 'wall', 'platform', 'hazard_zone', 'enemy_spawn_node'."
    )
    bounds: Tuple[float, float, float, float] = Field(
        ...,
        description="Axis-aligned bounding box coordinates (x, y, width, height)."
    )
    target_cluster_id: Optional[str] = Field(
        default=None,
        description="Reference to death cluster ID being addressed."
    )
    cluster_centroid: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Centroid (x, y) of player death cluster."
    )
    death_count_addressed: int = Field(
        default=0,
        description="Number of player casualties in the target cluster."
    )
    navmesh_verified: bool = Field(
        default=False,
        description="True if NavMesh A* validation confirmed critical path remains clear."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "directive_id": self.directive_id,
            "action": self.action.value if hasattr(self.action, "value") else str(self.action),
            "geometry_type": self.geometry_type,
            "bounds": list(self.bounds),
            "target_cluster_id": self.target_cluster_id,
            "cluster_centroid": list(self.cluster_centroid) if self.cluster_centroid else None,
            "death_count_addressed": self.death_count_addressed,
            "navmesh_verified": self.navmesh_verified,
        }


class CurrentEpisodeConfig(BaseModel):
    """Snapshot of baseline parameters for the current episode prior to adaptation."""
    deliberation: str = Field(
        default="",
        description="MANDATORY FIRST PROPERTY: Epistemic summary of episode parameters."
    )
    episode_id: str = Field(default="level_01", description="Identifier of the current episode.")
    episode_number: int = Field(default=1, description="1-indexed episode counter.")
    enemy_speed: float = Field(default=185.0, description="Baseline chaser enemy movement speed.")
    enemy_health: float = Field(default=60.0, description="Baseline chaser enemy health.")
    enemy_damage: float = Field(default=15.0, description="Baseline chaser contact damage.")
    ranged_enemy_speed: float = Field(default=135.0, description="Baseline ranged enemy speed.")
    ranged_enemy_damage: float = Field(default=12.0, description="Baseline ranged projectile damage.")
    hazard_cycle_rate: float = Field(default=1.2, description="Baseline hazard cycle speed / frequency.")
    hazard_damage: float = Field(default=12.0, description="Baseline hazard damage.")
    spawn_rate: float = Field(default=1.0, description="Enemy spawn density / rate multiplier.")
    weapon_damage: float = Field(default=25.0, description="Player primary weapon damage.")
    spawn_point: Tuple[float, float] = Field(
        default=(120.0, 360.0),
        description="Player initial spawn coordinates."
    )
    exit_point: Tuple[float, float] = Field(
        default=(1160.0, 360.0),
        description="Episode goal/exit portal coordinates."
    )
    walls: List[Tuple[float, float, float, float]] = Field(
        default_factory=list,
        description="List of solid wall rectangles (x, y, w, h)."
    )
    enemies: List[Dict[str, Any]] = Field(default_factory=list, description="Enemy spawn definitions.")
    hazards: List[Dict[str, Any]] = Field(default_factory=list, description="Hazard definitions.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "episode_id": self.episode_id,
            "episode_number": self.episode_number,
            "spawn_point": list(self.spawn_point),
            "exit_point": list(self.exit_point),
            "enemy_speed": self.enemy_speed,
            "enemy_health": self.enemy_health,
            "enemy_damage": self.enemy_damage,
            "ranged_enemy_speed": self.ranged_enemy_speed,
            "ranged_enemy_damage": self.ranged_enemy_damage,
            "hazard_cycle_rate": self.hazard_cycle_rate,
            "hazard_damage": self.hazard_damage,
            "spawn_rate": self.spawn_rate,
            "weapon_damage": self.weapon_damage,
            "walls": [list(w) for w in self.walls],
            "enemies": self.enemies,
            "hazards": self.hazards,
        }


class EpisodicUpdateSpec(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation MUST be declared first!
    Complete structured output emitted by EpisodicDirector for SWE Compiler consumption.
    """
    deliberation: str = Field(
        default="",
        description="MANDATORY FIRST PROPERTY: Multi-sentence deliberation reflecting on player profile, bottlenecks, and adaptation strategy."
    )
    spec_id: str = Field(
        default_factory=lambda: f"spec_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this episodic update specification."
    )
    target_tier: str = Field(
        default="episodic_update",
        description="Tier identifier for routing."
    )
    current_episode_id: str = Field(..., description="Episode ID that was just analyzed.")
    target_episode_id: str = Field(..., description="New Episode ID being generated.")
    target_episode_number: int = Field(..., description="Episode sequence counter.")
    player_archetype: str = Field(..., description="Player archetype classified by TelemetryAnalyst.")
    mastery_score: float = Field(..., description="Calculated player mastery score in [0.0, 1.0].")
    bottleneck_detected: bool = Field(default=False, description="Whether a bottleneck or death cluster was found.")
    bottleneck_summary: str = Field(default="", description="Summary description of bottleneck.")
    clamp_state: DifficultyClampState = Field(..., description="Enforced difficulty clamp records.")
    mechanical_directives: List[MechanicalUpdateDirective] = Field(
        default_factory=list,
        description="Ordered list of mechanical adaptations."
    )
    spatial_directives: List[SpatialUpdateDirective] = Field(
        default_factory=list,
        description="Ordered list of spatial geometry adaptations."
    )
    updated_episode_config: CurrentEpisodeConfig = Field(
        ...,
        description="Full updated episode parameters ready for SWE cartridge compilation."
    )
    recommended_features: List[str] = Field(
        default_factory=list,
        description="High-level feature tags."
    )
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Strictly serializes with 'deliberation' as the first dictionary key."""
        return {
            "deliberation": self.deliberation,
            "spec_id": self.spec_id,
            "target_tier": self.target_tier,
            "current_episode_id": self.current_episode_id,
            "target_episode_id": self.target_episode_id,
            "target_episode_number": self.target_episode_number,
            "player_archetype": self.player_archetype,
            "mastery_score": round(self.mastery_score, 3),
            "bottleneck_detected": self.bottleneck_detected,
            "bottleneck_summary": self.bottleneck_summary,
            "clamp_state": self.clamp_state.to_dict(),
            "mechanical_directives": [m.to_dict() for m in self.mechanical_directives],
            "spatial_directives": [s.to_dict() for s in self.spatial_directives],
            "updated_episode_config": self.updated_episode_config.to_dict(),
            "recommended_features": self.recommended_features,
            "created_at_utc": self.created_at_utc,
        }

    def to_markdown(self) -> str:
        """Renders update_spec.md strictly starting with ## Deliberation."""
        lines = [
            f"# Episodic Update Specification — Episode {self.target_episode_number}",
            "",
            "## Deliberation",
            self.deliberation.strip(),
            "",
            "## Adaptive Profile Summary",
            f"- **Target Profile Archetype**: `{self.player_archetype}`",
            f"- **Mastery Score**: `{self.mastery_score:.2f}`",
            f"- **Bottleneck Detected**: `{'YES' if self.bottleneck_detected else 'NO'}`",
        ]
        if self.bottleneck_summary:
            lines.append(f"- **Bottleneck Diagnostics**: {self.bottleneck_summary}")
        lines.append(f"- **Recommended Features**: {', '.join(self.recommended_features)}")
        lines.extend([
            "",
            "## Bounded Difficulty Clamping State",
            self.clamp_state.get_summary_table_markdown(),
            "",
            "## Mechanical Directives",
        ])
        if not self.mechanical_directives:
            lines.append("- No mechanical changes required.")
        else:
            for idx, md in enumerate(self.mechanical_directives, 1):
                param_str = ", ".join(f"{k}={v:.2f}" for k, v in md.parameter_modifications.items())
                beh_str = f" [Behavior: {md.behavior_name}]" if md.behavior_name else ""
                lines.append(f"{idx}. **[{md.action.value}] {md.target_entity}**{beh_str} — {param_str}")
                lines.append(f"   *Rationale*: {md.deliberation}")

        lines.extend([
            "",
            "## Spatial Directives",
        ])
        if not self.spatial_directives:
            lines.append("- No spatial alterations required.")
        else:
            for idx, sd in enumerate(self.spatial_directives, 1):
                x, y, w, h = sd.bounds
                nav_str = "Verified Traversable" if sd.navmesh_verified else "Unverified"
                lines.append(f"{idx}. **[{sd.action.value}] {sd.geometry_type.capitalize()}** at `({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f})` [{nav_str}]")
                lines.append(f"   *Rationale*: {sd.deliberation}")

        lines.extend([
            "",
            "## Compiled Episode Configuration",
            "```json",
            json.dumps(self.updated_episode_config.to_dict(), indent=2),
            "```",
            ""
        ])
        return "\n".join(lines)
