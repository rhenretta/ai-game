"""engine/director/genesis.py
Rationale-First Steam Genesis Spec Generator & Director Engine.
Consumes SteamPlayerDNA and generates GenesisSpec and genesis_spec.md.
Strictly enforces deliberation declared first across all classes, schemas, and markdown outputs.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import config
from config import SPECS_DIR, DB_PATH
from engine.director.models import (
    EnemyArchetypeSpec,
    GenesisSpec,
    HazardArchetypeSpec,
    LevelDimensionsSpec,
    PlayerParametersSpec,
    WeaponArchetypeSpec,
)
from engine.steam.models import SteamPlayerDNA

# Canonical palette definitions aligned with Style Bible 16-color palette
PALETTE_THEMES: Dict[str, Dict[str, str]] = {
    "neon_cyberpunk": {
        "primary": "#29ADFF",      # Cyan
        "secondary": "#FF004D",    # Neon Red/Pink
        "accent": "#FFEC27",       # Bright Yellow
        "background": "#000000",   # Pure Black
        "surface": "#1D2B53",      # Dark Navy
    },
    "grim_dark": {
        "primary": "#7E2553",      # Deep Crimson
        "secondary": "#AB5236",    # Rust Brown
        "accent": "#5F574F",       # Muted Slate Gray
        "background": "#000000",   # Pure Black
        "surface": "#1D2B53",      # Dark Navy
    },
    "retro_arcade": {
        "primary": "#00E436",      # Electric Green
        "secondary": "#FFA300",    # Amber Orange
        "accent": "#FFEC27",       # Bright Yellow
        "background": "#000000",   # Pure Black
        "surface": "#5F574F",      # Muted Gray
    },
    "vibrant_fantasy": {
        "primary": "#83769C",      # Lavender Purple
        "secondary": "#FF77A8",    # Pastel Rose Pink
        "accent": "#00E436",       # Electric Green
        "background": "#1D2B53",   # Dark Navy
        "surface": "#C2C3C7",      # Light Gray
    },
}


class GenesisDirector:
    """Cognitive Director that analyzes SteamPlayerDNA to synthesize the Episode 0 GenesisSpec."""
    deliberation: str = "Synthesizing Episode 0 GenesisSpec from player Steam DNA engagement profile."

    def __init__(
        self,
        specs_dir: Optional[Union[Path, str]] = None,
        db_path: Optional[Union[Path, str]] = None,
        deliberation: str = "Synthesizing Episode 0 GenesisSpec from player Steam DNA engagement profile."
    ):
        self.deliberation = deliberation
        self.specs_dir = Path(specs_dir or SPECS_DIR)
        self.db_path = Path(db_path or DB_PATH)
        self.specs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_float(
        val: Any,
        default: float = 0.5,
        min_val: Optional[float] = 0.0,
        max_val: Optional[float] = 1.0,
    ) -> float:
        """Sanitizes raw numeric inputs against NaN, Infinity, None, type errors, and boundary limits."""
        if val is None:
            return default
        try:
            f_val = float(val)
        except (ValueError, TypeError):
            return default
        if math.isnan(f_val):
            return default
        if math.isinf(f_val):
            if f_val > 0:
                return max_val if max_val is not None else default
            else:
                return min_val if min_val is not None else default
        if min_val is not None and f_val < min_val:
            return min_val
        if max_val is not None and f_val > max_val:
            return max_val
        return f_val

    def _sanitize_dna(self, dna: SteamPlayerDNA) -> SteamPlayerDNA:
        """Creates a sanitized copy of SteamPlayerDNA with NaN/Inf values replaced and bounds enforced."""
        delib = getattr(dna, "deliberation", "") or "Sanitized SteamPlayerDNA for Genesis Spec Generation."
        safe_c_tol = self._sanitize_float(
            getattr(dna, "challenge_tolerance", 0.5), default=0.5, min_val=0.0, max_val=1.0
        )
        safe_p_score = self._sanitize_float(
            getattr(dna, "pacing_score", 0.0), default=0.0, min_val=-1.0, max_val=1.0
        )
        safe_playtime = self._sanitize_float(
            getattr(dna, "total_playtime_hours", 0.0), default=0.0, min_val=0.0, max_val=None
        )
        raw_biases = getattr(dna, "mechanical_biases", {}) or {}
        safe_biases = {
            k: self._sanitize_float(v, default=0.5, min_val=0.0, max_val=1.0)
            for k, v in raw_biases.items()
        }
        return SteamPlayerDNA(
            deliberation=delib,
            player_id=getattr(dna, "player_id", "local_player"),
            profile_tag=getattr(dna, "profile_tag", "genesis_default"),
            total_playtime_hours=safe_playtime,
            total_games_analyzed=max(0, getattr(dna, "total_games_analyzed", 0)),
            genre_weights=getattr(dna, "genre_weights", {}) or {},
            pacing_preference=getattr(dna, "pacing_preference", "balanced") or "balanced",
            pacing_score=safe_p_score,
            challenge_tolerance=safe_c_tol,
            mechanical_biases=safe_biases,
            preferred_palette_mood=getattr(dna, "preferred_palette_mood", "retro_arcade") or "retro_arcade",
            palette_weights=getattr(dna, "palette_weights", {}) or {},
            top_genres=getattr(dna, "top_genres", []) or [],
        )

    def generate_genesis_spec(self, dna: SteamPlayerDNA) -> GenesisSpec:
        """Consumes SteamPlayerDNA and outputs a fully articulated GenesisSpec with deliberation first."""
        spec_id = f"genesis_spec_{uuid.uuid4().hex[:12]}"

        # Robust input sanitization (defense-in-depth)
        clean_dna = self._sanitize_dna(dna)

        # 1. Epistemic Deliberation (Reasoning tokens FIRST)
        deliberation = self._formulate_deliberation(clean_dna)

        # 2. Derive Game Title & Core Genre
        game_title, core_genre = self._derive_title_and_genre(clean_dna)

        # 3. Derive Player Parameters
        player_params = self._compute_player_parameters(clean_dna)

        # 4. Derive Starter Weapon Archetype
        weapon_spec = self._compute_weapon_archetype(clean_dna)

        # 5. Derive Enemy Archetypes
        enemy_specs = self._compute_enemy_archetypes(clean_dna)

        # 6. Derive Environmental Hazards
        hazard_specs = self._compute_hazard_archetypes(clean_dna)

        # 7. Derive Spatial Level Dimensions & Layout
        level_dims = self._compute_level_dimensions(clean_dna)

        # 8. Derive Palette Mood & Colors
        palette_mood = clean_dna.preferred_palette_mood or "retro_arcade"
        if palette_mood not in PALETTE_THEMES:
            palette_mood = "retro_arcade"
        palette_colors = PALETTE_THEMES.get(palette_mood, PALETTE_THEMES["retro_arcade"])

        # 9. Derive Initial Difficulty Baseline
        safe_c_tol = self._sanitize_float(clean_dna.challenge_tolerance, default=0.5, min_val=0.0, max_val=1.0)
        target_diff = round(0.70 + (safe_c_tol * 0.60), 2)  # [0.70, 1.30]

        # 10. Camera Style
        safe_p_score = self._sanitize_float(clean_dna.pacing_score, default=0.0, min_val=-1.0, max_val=1.0)
        camera_style = "wide_expanse" if safe_p_score >= 0.30 else "arena_framing"

        return GenesisSpec(
            deliberation=deliberation,
            spec_id=spec_id,
            player_id=clean_dna.player_id,
            source_profile_tag=clean_dna.profile_tag,
            target_episode=0,
            game_title=game_title,
            core_genre=core_genre,
            palette_mood=palette_mood,
            palette_colors=palette_colors,
            player_params=player_params,
            weapon_spec=weapon_spec,
            enemy_specs=enemy_specs,
            hazard_specs=hazard_specs,
            level_dimensions=level_dims,
            target_difficulty=target_diff,
            camera_style=camera_style,
        )

    def _formulate_deliberation(self, dna: SteamPlayerDNA) -> str:
        """Generates comprehensive multi-sentence rationale explaining how Steam library history dictates design."""
        top_genres_str = ", ".join(dna.top_genres) if dna.top_genres else "Action, Adventure"
        total_hrs = self._sanitize_float(getattr(dna, "total_playtime_hours", 0.0), default=0.0, min_val=0.0, max_val=None)
        pacing_score = self._sanitize_float(getattr(dna, "pacing_score", 0.0), default=0.0, min_val=-1.0, max_val=1.0)
        pacing_pref = getattr(dna, "pacing_preference", "balanced") or "balanced"
        challenge_tol = self._sanitize_float(getattr(dna, "challenge_tolerance", 0.5), default=0.5, min_val=0.0, max_val=1.0)
        biases = getattr(dna, "mechanical_biases", {}) or {}
        dodge_b = self._sanitize_float(biases.get("dodge_roll", 0.5), default=0.5, min_val=0.0, max_val=1.0)
        aim_b = self._sanitize_float(biases.get("ranged_aim", 0.5), default=0.5, min_val=0.0, max_val=1.0)
        melee_b = self._sanitize_float(biases.get("close_combat", 0.5), default=0.5, min_val=0.0, max_val=1.0)
        speed_b = self._sanitize_float(biases.get("speedrun", 0.5), default=0.5, min_val=0.0, max_val=1.0)

        # Sentence 1: Historical grounding
        s1 = (
            f"Evaluated player '{dna.player_id}' across {dna.total_games_analyzed} Steam titles "
            f"with {total_hrs:.1f} hours of logged gameplay, identifying primary genre affinity for {top_genres_str}."
        )

        # Sentence 2: Pacing & Mobility rationale
        if pacing_score >= 0.20:
            s2 = (
                f"The kinetic pacing score of {pacing_score:+.2f} ({pacing_pref}) signals high demand for rapid twitch reflexes; "
                f"consequently, the player entity is calibrated with elevated walk speed, high-velocity burst dash, and reduced dash cooldown "
                f"to facilitate aggressive momentum preservation."
            )
        elif pacing_score <= -0.20:
            s2 = (
                f"The tactical pacing score of {pacing_score:+.2f} ({pacing_pref}) demonstrates preference for deliberate spatial positioning "
                f"and strategic risk calculation; therefore, player movement is tuned with moderate walk velocity and deliberate cooldowns "
                f"to emphasize precision footwork over reckless rushing."
            )
        else:
            s2 = (
                f"The balanced pacing index of {pacing_score:+.2f} reflects an adaptable playstyle; player dynamics are configured with "
                f"standard baseline walk speed and responsive evasive dash to support versatile skirmishing."
            )

        # Sentence 3: Challenge tolerance & encounter intensity
        if challenge_tol >= 0.65:
            s3 = (
                f"High challenge tolerance of {challenge_tol:.2f} (derived from deep achievement completion and hardcore genre exposure) "
                f"justifies spawning a denser initial mob cluster with elevated tracking speeds and aggressive contact damage to avoid trivialization."
            )
        elif challenge_tol <= 0.40:
            s3 = (
                f"Moderate challenge tolerance of {challenge_tol:.2f} dictates an accessible introductory encounter, featuring fewer simultaneous "
                f"enemies, wider attack telegraph windows, and forgiving contact damage."
            )
        else:
            s3 = (
                f"Calibrated challenge tolerance of {challenge_tol:.2f} dictates a balanced combat threshold with moderate enemy health and readable attack patterns."
            )

        # Sentence 4: Mechanical biases & Weapon Archetype selection
        if aim_b >= 0.65 or (aim_b > dodge_b and aim_b > melee_b):
            s4 = (
                f"Prominent ranged aiming bias ({aim_b:.2f}) dictates the allocation of a high-velocity precision blaster reward structure, "
                f"granting direct projectile velocity and high single-target impact."
            )
        elif dodge_b >= 0.60 or melee_b >= 0.60:
            s4 = (
                f"Pronounced evasive and close-quarters engagement biases (dodge_roll={dodge_b:.2f}, close_combat={melee_b:.2f}) mandate a "
                f"tri-shot spread cannon archetype, rewarding aggressive close-range point-blank engagement."
            )
        else:
            s4 = (
                f"Balanced mechanical biases dictate an all-around rapid blaster archetype with steady fire rate and reliable projectile velocity."
            )

        # Sentence 5: Spatial geometry & aesthetic theme
        s5 = (
            f"Arena geometry is configured with a '{dna.preferred_palette_mood}' aesthetic mood and tailored pillar distribution "
            f"to guarantee clean navigational sightlines and strict A* traversability from spawn to exit."
        )

        return f"{s1} {s2} {s3} {s4} {s5}"

    def _derive_title_and_genre(self, dna: SteamPlayerDNA) -> Tuple[str, str]:
        top = dna.top_genres[0] if (dna.top_genres and dna.top_genres[0]) else "Action"
        pref = getattr(dna, "pacing_preference", "balanced")
        if pref == "kinetic_high_mobility":
            return f"Hyper Kinetic: {top} Genesis", f"Fast-Paced {top} Roguelike"
        elif pref == "tactical_methodical":
            return f"Tactical Breach: {top} Genesis", f"Strategic {top} Arena"
        return f"Crucible: {top} Genesis", f"Balanced {top} Roguelike"

    def _compute_player_parameters(self, dna: SteamPlayerDNA) -> PlayerParametersSpec:
        p_score = self._sanitize_float(getattr(dna, "pacing_score", 0.0), default=0.0, min_val=-1.0, max_val=1.0)
        biases = getattr(dna, "mechanical_biases", {}) or {}
        speed_b = self._sanitize_float(biases.get("speedrun", 0.5), default=0.5, min_val=0.0, max_val=1.0)
        dodge_b = self._sanitize_float(biases.get("dodge_roll", 0.5), default=0.5, min_val=0.0, max_val=1.0)

        # Continuous linear mappings anchored at neutral baseline
        walk_speed = max(200.0, min(360.0, 260.0 + (p_score * 60.0) + ((speed_b - 0.5) * 40.0)))
        dash_speed = max(450.0, min(850.0, 600.0 + (p_score * 150.0) + ((dodge_b - 0.5) * 100.0)))
        dash_cooldown = max(0.35, min(1.00, 0.60 - (p_score * 0.20) - ((dodge_b - 0.5) * 0.20)))
        dash_cost = max(16.0, min(25.0, 25.0 - ((dodge_b - 0.5) * 10.0)))
        jump_force = max(350.0, min(560.0, 420.0 + (p_score * 80.0) + ((speed_b - 0.5) * 80.0)))
        stamina_regen = max(20.0, min(55.0, 35.0 + (p_score * 10.0) + ((dodge_b - 0.5) * 20.0)))

        delib = (
            f"Configured player walk speed at {walk_speed:.1f} px/s, dash impulse at {dash_speed:.1f} px/s, "
            f"and dash cooldown at {dash_cooldown:.2f}s based on pacing score ({p_score:+.2f}) and mobility biases."
        )

        return PlayerParametersSpec(
            deliberation=delib,
            walk_speed=walk_speed,
            dash_speed=dash_speed,
            dash_cooldown=dash_cooldown,
            dash_duration=0.20,
            dash_cost=dash_cost,
            jump_force=jump_force,
            max_health=100.0,
            max_stamina=100.0,
            stamina_regen_rate=stamina_regen,
        )

    def _compute_weapon_archetype(self, dna: SteamPlayerDNA) -> WeaponArchetypeSpec:
        biases = getattr(dna, "mechanical_biases", {}) or {}
        aim_b = self._sanitize_float(biases.get("ranged_aim", 0.5), default=0.5, min_val=0.0, max_val=1.0)
        dodge_b = self._sanitize_float(biases.get("dodge_roll", 0.5), default=0.5, min_val=0.0, max_val=1.0)
        melee_b = self._sanitize_float(biases.get("close_combat", 0.5), default=0.5, min_val=0.0, max_val=1.0)

        if aim_b >= 0.65 or (aim_b > dodge_b and aim_b > melee_b):
            delib = f"Selected precision tactical sniper weapon to match player's high ranged aiming affinity ({aim_b:.2f})."
            return WeaponArchetypeSpec(
                deliberation=delib,
                name="tactical_sniper",
                archetype="sniper",
                fire_rate=0.35,
                damage=36.0,
                projectile_speed=880.0,
                projectile_radius=4.5,
                projectile_lifetime=1.8,
                spread_count=1,
                range_distance=700.0,
            )
        elif dodge_b >= 0.60 or melee_b >= 0.60:
            delib = f"Selected spread shotgun cannon to reward high-mobility dash-ins (dodge={dodge_b:.2f}, melee={melee_b:.2f})."
            return WeaponArchetypeSpec(
                deliberation=delib,
                name="spread_cannon",
                archetype="spread",
                fire_rate=0.45,
                damage=16.0,
                projectile_speed=650.0,
                projectile_radius=3.5,
                projectile_lifetime=1.5,
                spread_count=3,
                range_distance=520.0,
            )
        else:
            delib = "Selected balanced rapid blaster for steady kinetic output across standard engagement distances."
            return WeaponArchetypeSpec(
                deliberation=delib,
                name="rapid_blaster",
                archetype="blaster",
                fire_rate=0.22,
                damage=25.0,
                projectile_speed=750.0,
                projectile_radius=4.0,
                projectile_lifetime=1.8,
                spread_count=1,
                range_distance=600.0,
            )

    def _compute_enemy_archetypes(self, dna: SteamPlayerDNA) -> List[EnemyArchetypeSpec]:
        c_tol = self._sanitize_float(getattr(dna, "challenge_tolerance", 0.5), default=0.5, min_val=0.0, max_val=1.0)
        p_score = self._sanitize_float(getattr(dna, "pacing_score", 0.0), default=0.0, min_val=-1.0, max_val=1.0)

        # Total enemy count in [3, 6]
        raw_count = round(3.0 + (3.0 * c_tol))
        total_count = max(3, min(6, int(raw_count)))

        if p_score >= 0.10:
            chaser_count = max(2, int(round(total_count * 0.70)))
            ranged_count = max(1, total_count - chaser_count)
        else:
            ranged_count = max(2, int(round(total_count * 0.60)))
            chaser_count = max(1, total_count - ranged_count)

        chaser_speed = max(150.0, min(220.0, 150.0 + (c_tol * 60.0) + (p_score * 25.0)))
        chaser_hp = max(45.0, min(80.0, 50.0 + (c_tol * 30.0)))
        chaser_dmg = max(10.0, min(22.0, 12.0 + (c_tol * 10.0)))

        ranged_speed = max(110.0, min(160.0, 120.0 + (c_tol * 35.0)))
        ranged_hp = max(35.0, min(60.0, 40.0 + (c_tol * 20.0)))
        ranged_rate = max(1.10, min(1.80, 1.60 - (c_tol * 0.40)))

        enemies: List[EnemyArchetypeSpec] = [
            EnemyArchetypeSpec(
                deliberation=f"Calibrated {chaser_count} melee chasers with speed {chaser_speed:.1f} px/s for kinetic pressure.",
                enemy_id="chaser",
                archetype="chaser",
                count=chaser_count,
                max_health=chaser_hp,
                speed=chaser_speed,
                contact_damage=chaser_dmg,
                attack_rate=0.80,
                preferred_distance=0.0,
                spawn_regions=["mid_right", "top_right"],
            ),
            EnemyArchetypeSpec(
                deliberation=f"Calibrated {ranged_count} ranged snipers with fire interval {ranged_rate:.2f}s for standoff denial.",
                enemy_id="ranged",
                archetype="ranged",
                count=ranged_count,
                max_health=ranged_hp,
                speed=ranged_speed,
                contact_damage=8.0,
                attack_rate=ranged_rate,
                preferred_distance=280.0,
                spawn_regions=["far_right", "bottom_right"],
            ),
        ]
        return enemies

    def _compute_hazard_archetypes(self, dna: SteamPlayerDNA) -> List[HazardArchetypeSpec]:
        c_tol = self._sanitize_float(getattr(dna, "challenge_tolerance", 0.5), default=0.5, min_val=0.0, max_val=1.0)
        hazards = [
            HazardArchetypeSpec(
                deliberation="Placed spike hazard grid in central combat lane to punish static standing.",
                hazard_id="spikes_1",
                hazard_type="spikes",
                count=2 if c_tol >= 0.50 else 1,
                damage=15.0,
                placement_hint="center_lane",
            )
        ]
        if c_tol >= 0.60:
            hazards.append(
                HazardArchetypeSpec(
                    deliberation="Introduced perimeter laser sweep to challenge high-tolerance player spatial awareness.",
                    hazard_id="laser_1",
                    hazard_type="laser_sweep",
                    count=1,
                    damage=20.0,
                    placement_hint="perimeter_upper",
                )
            )
        return hazards

    def _compute_level_dimensions(self, dna: SteamPlayerDNA) -> LevelDimensionsSpec:
        p_score = self._sanitize_float(getattr(dna, "pacing_score", 0.0), default=0.0, min_val=-1.0, max_val=1.0)
        biases = getattr(dna, "mechanical_biases", {}) or {}
        speed_b = self._sanitize_float(biases.get("speedrun", 0.5), default=0.5, min_val=0.0, max_val=1.0)

        if p_score >= 0.30 or speed_b >= 0.70:
            w, h = 1440.0, 810.0
            pillars = 2
            layout = "wide_expanse"
            delib = f"Expanded arena dimensions to {w:.0f}x{h:.0f} with minimal pillars ({pillars}) to provide wide running lanes."
        elif p_score <= -0.20:
            w, h = 1280.0, 720.0
            pillars = 6
            layout = "tactical_corridors"
            delib = f"Structured standard {w:.0f}x{h:.0f} arena with {pillars} defensive pillars creating cover and choke points."
        else:
            w, h = 1280.0, 720.0
            pillars = 4
            layout = "open_arena"
            delib = f"Configured balanced {w:.0f}x{h:.0f} arena with {pillars} symmetrical pillars."

        spawn_pt = (120.0, h / 2.0)
        exit_pt = (w - 120.0, h / 2.0)

        return LevelDimensionsSpec(
            deliberation=delib,
            width=w,
            height=h,
            spawn_point=spawn_pt,
            exit_point=exit_pt,
            wall_thickness=32.0,
            pillar_count=pillars,
            arena_layout_type=layout,
        )

    def render_genesis_markdown(self, spec: GenesisSpec) -> str:
        """Renders the authoritative genesis_spec.md strictly starting with ## Deliberation."""
        p = spec.player_params
        w = spec.weapon_spec
        l = spec.level_dimensions

        enemies_md = ""
        for e in spec.enemy_specs:
            enemies_md += (
                f"- **{e.archetype.capitalize()} Enemy** (`{e.enemy_id}`):\n"
                f"  - Count: {e.count}\n"
                f"  - Health: {e.max_health:.1f}\n"
                f"  - Speed: {e.speed:.1f} px/s\n"
                f"  - Contact Damage: {e.contact_damage:.1f}\n"
                f"  - Attack Rate: {e.attack_rate:.2f} s\n"
                f"  - Regions: {', '.join(e.spawn_regions)}\n"
            )

        hazards_md = ""
        for h in spec.hazard_specs:
            hazards_md += (
                f"- **{h.hazard_type.capitalize()}** (`{h.hazard_id}`): Count {h.count}, "
                f"Damage {h.damage:.1f}, Placement `{h.placement_hint}`\n"
            )

        colors_md = ""
        for k, v in spec.palette_colors.items():
            colors_md += f"  - `{k}`: {v}\n"

        spec_json = json.dumps(spec.to_dict(), indent=2)

        md = f"""# Genesis Specification: Episode 0

## Deliberation
{spec.deliberation}

## Genesis Metadata
- **Spec ID**: `{spec.spec_id}`
- **Player ID**: `{spec.player_id}`
- **Profile Tag**: `{spec.source_profile_tag}`
- **Target Episode**: {spec.target_episode}
- **Game Title**: {spec.game_title}
- **Core Genre**: {spec.core_genre}
- **Target Difficulty**: {spec.target_difficulty:.2f}
- **Created At UTC**: {spec.created_at_utc}

## Player Dynamics & Physics
- **Walk Speed**: {p.walk_speed:.1f} px/s
- **Dash Speed**: {p.dash_speed:.1f} px/s
- **Dash Cooldown**: {p.dash_cooldown:.3f} s
- **Dash Duration**: {p.dash_duration:.3f} s
- **Dash Stamina Cost**: {p.dash_cost:.1f}
- **Jump Force**: {p.jump_force:.1f} px/s
- **Max Health**: {p.max_health:.1f}
- **Stamina Regen Rate**: {p.stamina_regen_rate:.1f} /s

## Starter Weapon Archetype
- **Name**: `{w.name}`
- **Archetype**: `{w.archetype}`
- **Fire Rate**: {w.fire_rate:.3f} s
- **Damage**: {w.damage:.1f}
- **Projectile Speed**: {w.projectile_speed:.1f} px/s
- **Spread Count**: {w.spread_count}
- **Projectile Lifetime**: {w.projectile_lifetime:.2f} s
- **Range Distance**: {w.range_distance:.1f} px

## Enemy Archetypes & Spawns
{enemies_md}
## Environmental Hazards
{hazards_md if hazards_md else "- None configured for initial run."}
## Spatial Geometry & Level Dimensions
- **Arena Size**: {l.width:.0f} x {l.height:.0f}
- **Layout Type**: `{l.arena_layout_type}`
- **Pillar Count**: {l.pillar_count}
- **Player Spawn**: ({l.spawn_point[0]:.1f}, {l.spawn_point[1]:.1f})
- **Exit Portal**: ({l.exit_point[0]:.1f}, {l.exit_point[1]:.1f})
- **Wall Thickness**: {l.wall_thickness:.1f} px

## Aesthetic Palette & Mood
- **Palette Mood**: `{spec.palette_mood}`
- **Color Palette Tokens**:
{colors_md}- **Camera Style**: `{spec.camera_style}`

## Structured Spec (JSON)
```json
{spec_json}
```
"""
        return md

    def write_genesis_spec(
        self,
        spec: GenesisSpec,
        output_path: Optional[Union[Path, str]] = None
    ) -> Path:
        """Writes genesis_spec.md and optional genesis_spec.json to disk."""
        target_file = Path(output_path) if output_path else self.specs_dir / "genesis_spec.md"
        target_file.parent.mkdir(parents=True, exist_ok=True)

        md_content = self.render_genesis_markdown(spec)
        target_file.write_text(md_content, encoding="utf-8")

        json_file = target_file.with_suffix(".json")
        json_file.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

        return target_file

    def save_spec_to_db(self, spec: GenesisSpec) -> str:
        """Persists the GenesisSpec into the SQLite database for audit tracing."""
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS director_genesis_specs (
                spec_id TEXT PRIMARY KEY,
                player_id TEXT NOT NULL,
                profile_tag TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                deliberation TEXT NOT NULL,
                palette_mood TEXT NOT NULL,
                target_difficulty REAL NOT NULL,
                raw_spec_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_gen_player ON director_genesis_specs(player_id);
            CREATE INDEX IF NOT EXISTS idx_gen_created ON director_genesis_specs(created_at_utc);
            """)
            conn.execute(
                """
                INSERT OR REPLACE INTO director_genesis_specs (
                    spec_id, player_id, profile_tag, created_at_utc,
                    deliberation, palette_mood, target_difficulty, raw_spec_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    spec.spec_id,
                    spec.player_id,
                    spec.source_profile_tag,
                    spec.created_at_utc,
                    spec.deliberation,
                    spec.palette_mood,
                    spec.target_difficulty,
                    json.dumps(spec.to_dict()),
                )
            )
            conn.commit()
            return spec.spec_id
        finally:
            conn.close()
