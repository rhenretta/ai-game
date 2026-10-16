"""engine/director/episodic.py
Rationale-First Episodic Director with Bounded Between-Run Adaptation Clamp (+/-15%).
Analyzes BehavioralProfile, reflects on bottlenecks, applies spatial/mechanical adaptations,
and emits cryptographically sealed EpisodicUpdateSpec and update_spec.md artifacts.
Strictly adheres to Rationale-First Epistemic Architecture.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import config
from config import SPECS_DIR
from engine.director.models import (
    CurrentEpisodeConfig,
    DifficultyClampState,
    DirectiveAction,
    EpisodicUpdateSpec,
    MechanicalUpdateDirective,
    SpatialUpdateDirective,
)
from engine.scheduler.artifact import ArtifactMetadataEnvelope, ArtifactType
from engine.spatial.navmesh import NavMesh, astar_pathfind


class EpisodicDirector:
    """Showrunner Director analyzing player telemetry profiles to author episodic updates."""
    deliberation: str = "Showrunner Director analyzing player telemetry profiles to author episodic updates."

    def __init__(
        self,
        max_adaptation_ratio: float = 0.15,
        tile_size: float = 32.0,
        clearance_margin: float = 48.0,
        specs_dir: Optional[Union[Path, str]] = None,
    ):
        self.max_adaptation_ratio = float(max_adaptation_ratio)
        self.tile_size = float(tile_size)
        self.clearance_margin = float(clearance_margin)
        self.specs_dir = Path(specs_dir or SPECS_DIR)
        self.specs_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_and_update(
        self,
        profile: Any,
        current_config: Optional[CurrentEpisodeConfig] = None,
        world_bounds: Tuple[float, float] = (1280.0, 720.0),
        spawn_point: Tuple[float, float] = (120.0, 360.0),
        exit_point: Tuple[float, float] = (1160.0, 360.0),
    ) -> EpisodicUpdateSpec:
        """Main entry point: evaluates profile, deliberates, clamps difficulty, and generates spec."""
        cfg = current_config or CurrentEpisodeConfig(
            deliberation="Default baseline parameters",
            walls=[
                (0.0, 0.0, 1280.0, 32.0),
                (0.0, 688.0, 1280.0, 32.0),
                (0.0, 0.0, 32.0, 720.0),
                (1248.0, 0.0, 32.0, 720.0),
                (360.0, 160.0, 48.0, 160.0),
                (360.0, 400.0, 48.0, 160.0),
                (780.0, 160.0, 48.0, 160.0),
                (780.0, 400.0, 48.0, 160.0),
            ]
        )

        target_episode_num = cfg.episode_number + 1
        target_episode_id = f"level_{target_episode_num:02d}"

        # 1. Extract Profile Dimensions
        archetype = getattr(profile, "playstyle_tag", getattr(profile, "playstyle_archetype", "balanced_skirmisher"))
        mean_v = float(getattr(profile, "mean_velocity", 180.0))
        dash_ratio = float(getattr(profile, "dash_ratio", 0.25))

        # Mastery computation (0.0 to 1.0)
        mastery = self._calculate_mastery(profile)

        # Bottleneck detection
        death_clusters = getattr(profile, "death_clusters", [])
        deaths_total = int(getattr(profile, "deaths", 0))
        bottleneck_detected = bool(death_clusters or deaths_total >= 3)
        bottleneck_summary = ""
        if death_clusters:
            c = death_clusters[0]
            cx = c.get("x", c.get("center_x", 0.0))
            cy = c.get("y", c.get("center_y", 0.0))
            count = c.get("count", c.get("death_count", 2))
            bottleneck_summary = f"Identified death cluster of {count} casualties near centroid ({cx:.1f}, {cy:.1f})."
        elif deaths_total >= 3:
            bottleneck_summary = f"Player suffered {deaths_total} casualties across the episode."

        # 2. Formulate Rationale-First Epistemic Deliberation
        deliberation = self._formulate_deliberation(
            archetype=archetype,
            mean_v=mean_v,
            dash_ratio=dash_ratio,
            mastery=mastery,
            bottleneck_detected=bottleneck_detected,
            bottleneck_summary=bottleneck_summary,
            current_episode=cfg.episode_number,
            target_episode=target_episode_num,
        )

        # 3. Bounded Between-Run Adaptation Clamp (+/-15%)
        clamp_state = DifficultyClampState(
            deliberation=f"Enforcing strict +/-{self.max_adaptation_ratio * 100:.0f}% adaptation clamp between Episode {cfg.episode_number} and {target_episode_num}.",
            max_adaptation_ratio=self.max_adaptation_ratio,
        )

        # Calculate raw deltas based on mastery & bottleneck
        raw_enemy_delta, raw_player_delta = self._compute_raw_deltas(mastery, bottleneck_detected)

        # Clamp numerical parameters
        new_enemy_speed = clamp_state.apply_clamp(
            "enemy_speed",
            cfg.enemy_speed,
            cfg.enemy_speed * (1.0 + raw_enemy_delta),
            rationale=f"Enemy speed adapted from {cfg.enemy_speed:.1f} based on raw delta {raw_enemy_delta:+.1%}",
        )
        new_enemy_damage = clamp_state.apply_clamp(
            "enemy_damage",
            cfg.enemy_damage,
            cfg.enemy_damage * (1.0 + raw_enemy_delta),
            rationale=f"Enemy damage adapted from {cfg.enemy_damage:.1f}",
        )
        new_enemy_health = clamp_state.apply_clamp(
            "enemy_health",
            cfg.enemy_health,
            cfg.enemy_health * (1.0 + raw_enemy_delta),
            rationale=f"Enemy health adapted from {cfg.enemy_health:.1f}",
        )
        new_ranged_speed = clamp_state.apply_clamp(
            "ranged_enemy_speed",
            cfg.ranged_enemy_speed,
            cfg.ranged_enemy_speed * (1.0 + raw_enemy_delta),
            rationale=f"Ranged enemy speed adapted from {cfg.ranged_enemy_speed:.1f}",
        )
        new_ranged_damage = clamp_state.apply_clamp(
            "ranged_enemy_damage",
            cfg.ranged_enemy_damage,
            cfg.ranged_enemy_damage * (1.0 + raw_enemy_delta),
            rationale=f"Ranged enemy damage adapted from {cfg.ranged_enemy_damage:.1f}",
        )
        new_hazard_cycle = clamp_state.apply_clamp(
            "hazard_cycle_rate",
            cfg.hazard_cycle_rate,
            cfg.hazard_cycle_rate * (1.0 + raw_enemy_delta),
            rationale=f"Hazard cycle frequency adapted from {cfg.hazard_cycle_rate:.2f}",
        )
        new_hazard_damage = clamp_state.apply_clamp(
            "hazard_damage",
            cfg.hazard_damage,
            cfg.hazard_damage * (1.0 + raw_enemy_delta),
            rationale=f"Hazard damage adapted from {cfg.hazard_damage:.1f}",
        )
        new_spawn_rate = clamp_state.apply_clamp(
            "spawn_rate",
            cfg.spawn_rate,
            cfg.spawn_rate * (1.0 + raw_enemy_delta),
            rationale=f"Spawn rate multiplier adapted from {cfg.spawn_rate:.2f}",
        )
        new_weapon_damage = clamp_state.apply_clamp(
            "weapon_damage",
            cfg.weapon_damage,
            cfg.weapon_damage * (1.0 + raw_player_delta),
            rationale=f"Player weapon damage adapted from {cfg.weapon_damage:.1f}",
        )

        # 4. Mechanical Adaptation Directives
        mechanical_directives = self._generate_mechanical_directives(
            archetype=archetype,
            mastery=mastery,
            dash_ratio=dash_ratio,
            new_enemy_speed=new_enemy_speed,
            new_enemy_damage=new_enemy_damage,
            new_hazard_cycle=new_hazard_cycle,
            new_weapon_damage=new_weapon_damage,
        )

        # 5. Spatial Adaptation Directives with NavMesh Verification
        spatial_directives, updated_walls = self._generate_spatial_directives(
            profile=profile,
            cfg=cfg,
            world_bounds=world_bounds,
            spawn_point=spawn_point,
            exit_point=exit_point,
        )

        # 6. Assemble Updated Episode Config
        updated_config = CurrentEpisodeConfig(
            deliberation=f"Compiled parameters for Episode {target_episode_num} reflecting bounded adaptation.",
            episode_id=target_episode_id,
            episode_number=target_episode_num,
            enemy_speed=new_enemy_speed,
            enemy_health=new_enemy_health,
            enemy_damage=new_enemy_damage,
            ranged_enemy_speed=new_ranged_speed,
            ranged_enemy_damage=new_ranged_damage,
            hazard_cycle_rate=new_hazard_cycle,
            hazard_damage=new_hazard_damage,
            spawn_rate=new_spawn_rate,
            weapon_damage=new_weapon_damage,
            walls=updated_walls,
            enemies=cfg.enemies,
            hazards=cfg.hazards,
        )

        recommended_features = self._recommend_features(archetype, bottleneck_detected, mastery)

        spec = EpisodicUpdateSpec(
            deliberation=deliberation,
            current_episode_id=cfg.episode_id,
            target_episode_id=target_episode_id,
            target_episode_number=target_episode_num,
            player_archetype=archetype,
            mastery_score=mastery,
            bottleneck_detected=bottleneck_detected,
            bottleneck_summary=bottleneck_summary,
            clamp_state=clamp_state,
            mechanical_directives=mechanical_directives,
            spatial_directives=spatial_directives,
            updated_episode_config=updated_config,
            recommended_features=recommended_features,
        )
        return spec

    def _calculate_mastery(self, profile: Any) -> float:
        """Derives a normalized [0.0, 1.0] mastery score from telemetry metrics."""
        if hasattr(profile, "mastery_score") and profile.mastery_score is not None:
            return float(profile.mastery_score)

        deaths = int(getattr(profile, "deaths", 0))
        cleared = bool(getattr(profile, "cleared", True))
        dmg_taken = float(getattr(profile, "damage_taken_total", 0.0))

        score = 0.50
        if cleared:
            score += 0.20
        if deaths == 0:
            score += 0.15
        elif deaths >= 3:
            score -= 0.30

        if dmg_taken < 30.0:
            score += 0.15
        elif dmg_taken > 100.0:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _compute_raw_deltas(self, mastery: float, bottleneck_detected: bool) -> Tuple[float, float]:
        """Calculates raw proposed adjustments prior to clamp enforcement.

        Returns (raw_enemy_delta, raw_player_delta).
        """
        if bottleneck_detected or mastery <= 0.35:
            deficit = max(0.0, 0.40 - mastery)
            raw_enemy = -0.15 - deficit * 0.50
            raw_player = +0.10 + deficit * 0.30
        elif mastery >= 0.75:
            surplus = mastery - 0.70
            raw_enemy = +0.15 + surplus * 0.50
            raw_player = 0.0
        else:
            raw_enemy = (mastery - 0.55) * 0.25
            raw_player = 0.0
        return raw_enemy, raw_player

    def _formulate_deliberation(
        self,
        archetype: str,
        mean_v: float,
        dash_ratio: float,
        mastery: float,
        bottleneck_detected: bool,
        bottleneck_summary: str,
        current_episode: int,
        target_episode: int,
    ) -> str:
        """Constructs multi-sentence epistemic deliberation preceding all decision tokens."""
        sentences = [
            f"Episodic analysis of Episode {current_episode} demonstrates a '{archetype}' playstyle profile "
            f"operating with mean kinetic velocity {mean_v:.1f} px/s and dash ratio {dash_ratio:.1%}."
        ]
        if bottleneck_detected:
            sentences.append(
                f"A gameplay bottleneck was identified: {bottleneck_summary} "
                f"Player mastery dropped to {mastery:.2f}, indicating acute encounter fatigue."
            )
            sentences.append(
                f"To relieve this failure cluster without eliminating mechanical tension, we inject defensive cover geometry "
                f"and clamp downward enemy scaling strictly within the safety envelope of [-15.0%, 0.0%]."
            )
        elif mastery >= 0.75:
            sentences.append(
                f"Player exhibited commanding mastery ({mastery:.2f}) across all combat phases, navigating corridors cleanly. "
                f"To prevent encounter trivialization, enemy tracking velocity and hazard tempos are raised, "
                f"strictly bounded by the +15% adaptation ceiling to preserve rhythmic fairness."
            )
        else:
            sentences.append(
                f"Player maintains balanced mechanical engagement with steady progression (mastery {mastery:.2f}). "
                f"Difficulty parameters are fine-tuned within +/-5% while introducing varied enemy positioning."
            )

        sentences.append(
            f"All structural and mechanical changes for Episode {target_episode} strictly enforce "
            f"the Bounded Between-Run Adaptation Clamp and verify NavMesh path traversability."
        )
        return " ".join(sentences)

    def _generate_mechanical_directives(
        self,
        archetype: str,
        mastery: float,
        dash_ratio: float,
        new_enemy_speed: float,
        new_enemy_damage: float,
        new_hazard_cycle: float,
        new_weapon_damage: float,
    ) -> List[MechanicalUpdateDirective]:
        directives: List[MechanicalUpdateDirective] = []

        chaser_beh = "telegraphed_lunge" if dash_ratio > 0.40 else "flanking_rush"
        directives.append(
            MechanicalUpdateDirective(
                deliberation=f"Equip Chaser enemies with {chaser_beh} to counter high dash mobility.",
                action=DirectiveAction.INTRODUCE_BEHAVIOR,
                target_entity="chaser",
                behavior_name=chaser_beh,
                parameter_modifications={
                    "speed": new_enemy_speed,
                    "contact_damage": new_enemy_damage,
                },
                priority=1,
            )
        )

        directives.append(
            MechanicalUpdateDirective(
                deliberation=f"Tune hazard cycle frequency to {new_hazard_cycle:.2f} Hz based on clamped difficulty.",
                action=DirectiveAction.TUNE_PARAMETER,
                target_entity="hazard_spikes",
                parameter_modifications={"cycle_rate": new_hazard_cycle},
                priority=2,
            )
        )

        if mastery < 0.40:
            directives.append(
                MechanicalUpdateDirective(
                    deliberation=f"Buff player weapon damage to {new_weapon_damage:.1f} to assist struggling player.",
                    action=DirectiveAction.BALANCE_WEAPON,
                    target_entity="player_weapon",
                    parameter_modifications={"damage": new_weapon_damage},
                    priority=3,
                )
            )

        return directives

    def _is_clear_of_point(
        self,
        box: Tuple[float, float, float, float],
        point: Tuple[float, float],
        margin: float = 48.0,
    ) -> bool:
        """Returns True if the box does not overlap the clearance bounding box around point."""
        bx, by, bw, bh = box
        px, py = point
        overlaps_x = (bx < px + margin) and (bx + bw > px - margin)
        overlaps_y = (by < py + margin) and (by + bh > py - margin)
        return not (overlaps_x and overlaps_y)

    def _generate_spatial_directives(
        self,
        profile: Any,
        cfg: CurrentEpisodeConfig,
        world_bounds: Tuple[float, float],
        spawn_point: Tuple[float, float],
        exit_point: Tuple[float, float],
    ) -> Tuple[List[SpatialUpdateDirective], List[Tuple[float, float, float, float]]]:
        """Generates geometry modifications validated against NavMesh A* traversability."""
        spatial_directives: List[SpatialUpdateDirective] = []
        updated_walls = [tuple(w) for w in cfg.walls]

        # 1. Check for death clusters to address
        death_clusters = getattr(profile, "death_clusters", [])
        if death_clusters:
            cluster = death_clusters[0]
            cx = float(cluster.get("x", cluster.get("center_x", 640.0)))
            cy = float(cluster.get("y", cluster.get("center_y", 360.0)))
            count = int(cluster.get("count", cluster.get("death_count", 3)))

            pillar_w, pillar_h = 48.0, 48.0
            pillar_x = max(64.0, min(world_bounds[0] - 120.0, cx - 40.0))
            pillar_y = max(64.0, min(world_bounds[1] - 120.0, cy - 40.0))
            proposed_box = (pillar_x, pillar_y, pillar_w, pillar_h)

            margin = self.clearance_margin
            clear_of_spawn = self._is_clear_of_point(proposed_box, spawn_point, margin)
            clear_of_exit = self._is_clear_of_point(proposed_box, exit_point, margin)

            if not clear_of_spawn or not clear_of_exit:
                candidates: List[Tuple[float, float, float, float]] = []
                if not clear_of_spawn:
                    cand_x_east = max(64.0, min(world_bounds[0] - 120.0, spawn_point[0] + margin + 16.0))
                    cand_y_north = max(64.0, min(world_bounds[1] - 120.0, spawn_point[1] - margin - pillar_h - 16.0))
                    cand_y_south = max(64.0, min(world_bounds[1] - 120.0, spawn_point[1] + margin + 16.0))
                    candidates.extend([
                        (cand_x_east, pillar_y, pillar_w, pillar_h),
                        (pillar_x, cand_y_north, pillar_w, pillar_h),
                        (pillar_x, cand_y_south, pillar_w, pillar_h),
                        (cand_x_east, cand_y_north, pillar_w, pillar_h),
                        (cand_x_east, cand_y_south, pillar_w, pillar_h),
                    ])
                if not clear_of_exit:
                    cand_x_west = max(64.0, min(world_bounds[0] - 120.0, exit_point[0] - margin - pillar_w - 16.0))
                    cand_y_north = max(64.0, min(world_bounds[1] - 120.0, exit_point[1] - margin - pillar_h - 16.0))
                    cand_y_south = max(64.0, min(world_bounds[1] - 120.0, exit_point[1] + margin + 16.0))
                    candidates.extend([
                        (cand_x_west, pillar_y, pillar_w, pillar_h),
                        (pillar_x, cand_y_north, pillar_w, pillar_h),
                        (pillar_x, cand_y_south, pillar_w, pillar_h),
                        (cand_x_west, cand_y_north, pillar_w, pillar_h),
                        (cand_x_west, cand_y_south, pillar_w, pillar_h),
                    ])

                relocated_box: Optional[Tuple[float, float, float, float]] = None
                for cand in candidates:
                    if self._is_clear_of_point(cand, spawn_point, margin) and self._is_clear_of_point(cand, exit_point, margin):
                        if self._verify_navmesh_traversability(updated_walls + [cand], world_bounds, spawn_point, exit_point):
                            relocated_box = cand
                            break
                proposed_box = relocated_box

            if proposed_box is not None:
                test_walls = updated_walls + [proposed_box]
                if self._verify_navmesh_traversability(test_walls, world_bounds, spawn_point, exit_point):
                    updated_walls.append(proposed_box)
                    px, py, pw, ph = proposed_box
                    reloc_note = " relocated outside spawn/exit clearance margin" if (not clear_of_spawn or not clear_of_exit) else ""
                    spatial_directives.append(
                        SpatialUpdateDirective(
                            deliberation=f"Construct defensive cover pillar at ({px:.0f}, {py:.0f}){reloc_note} mitigating death cluster of {count} casualties. Critical path verified.",
                            action=DirectiveAction.ADD_OBSTACLE,
                            geometry_type="pillar",
                            bounds=proposed_box,
                            target_cluster_id=str(cluster.get("id", "cluster_01")),
                            cluster_centroid=(cx, cy),
                            death_count_addressed=count,
                            navmesh_verified=True,
                        )
                    )

        # 2. Check for camping behavior
        camping_index = float(getattr(profile, "camping_index", 0.0))
        if camping_index > 0.55 and not spatial_directives:
            plat_box = (580.0, 320.0, 120.0, 40.0)
            if self._is_clear_of_point(plat_box, spawn_point, self.clearance_margin) and self._is_clear_of_point(plat_box, exit_point, self.clearance_margin):
                test_walls = updated_walls + [plat_box]
                if self._verify_navmesh_traversability(test_walls, world_bounds, spawn_point, exit_point):
                    updated_walls.append(plat_box)
                    spatial_directives.append(
                        SpatialUpdateDirective(
                            deliberation="Add mid-arena tactical platform at (580, 320) to incentivize player movement out of defensive corners.",
                            action=DirectiveAction.ADD_PLATFORM,
                            geometry_type="platform",
                            bounds=plat_box,
                            navmesh_verified=True,
                        )
                    )

        return spatial_directives, updated_walls

    def _verify_navmesh_traversability(
        self,
        walls: List[Tuple[float, float, float, float]],
        world_bounds: Tuple[float, float],
        spawn: Tuple[float, float],
        exit_pt: Tuple[float, float],
    ) -> bool:
        """Constructs a tile grid and verifies A* pathfinder connects spawn to exit."""
        cols = int(world_bounds[0] // self.tile_size)
        rows = int(world_bounds[1] // self.tile_size)
        grid = [[0 for _ in range(cols)] for _ in range(rows)]

        for wx, wy, ww, wh in walls:
            min_c = max(0, int(wx // self.tile_size))
            max_c = min(cols - 1, int((wx + ww) // self.tile_size))
            min_r = max(0, int(wy // self.tile_size))
            max_r = min(rows - 1, int((wy + wh) // self.tile_size))
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    grid[r][c] = 1

        sx = max(0, min(cols - 1, int(spawn[0] // self.tile_size)))
        sy = max(0, min(rows - 1, int(spawn[1] // self.tile_size)))
        ex = max(0, min(cols - 1, int(exit_pt[0] // self.tile_size)))
        ey = max(0, min(rows - 1, int(exit_pt[1] // self.tile_size)))

        if grid[sy][sx] == 1 or grid[ey][ex] == 1:
            return False

        path = astar_pathfind(grid, (sx, sy), (ex, ey), allow_diagonal=True)
        return path is not None

    def _recommend_features(self, archetype: str, bottleneck: bool, mastery: float) -> List[str]:
        features = []
        if bottleneck:
            features.extend(["defensive_cover_barricades", "staggered_enemy_waves", "expanded_escape_routes"])
        elif archetype in ("high_mobility_rusher", "kinetic_high_mobility"):
            features.extend(["flanking_chaser_lunges", "dynamic_traversal_hazards", "wide_combat_arenas"])
        elif archetype in ("methodical_cautious", "cautious_ranged", "methodical_tactical"):
            features.extend(["tactical_line_of_sight_barriers", "mortar_ranged_pressure", "anti_camping_zones"])
        else:
            features.extend(["balanced_encounter_mix", "rhythmic_hazard_timing"])
        return features

    def write_update_spec(
        self,
        spec: EpisodicUpdateSpec,
        output_path: Optional[Union[Path, str]] = None,
    ) -> Path:
        """Writes update_spec.md and optional update_spec.json to disk."""
        target_file = Path(output_path) if output_path else self.specs_dir / "update_spec.md"
        target_file.parent.mkdir(parents=True, exist_ok=True)

        md_content = spec.to_markdown()
        target_file.write_text(md_content, encoding="utf-8")

        json_file = target_file.with_suffix(".json")
        json_file.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

        return target_file

    def seal_update_artifact(
        self,
        spec: EpisodicUpdateSpec,
        source_agent: str = "EpisodicDirector",
        target_agent: str = "SWEWorkerCompiler",
        run_id: Optional[str] = None,
    ) -> ArtifactMetadataEnvelope:
        """Packs EpisodicUpdateSpec into a cryptographically sealed ArtifactMetadataEnvelope."""
        return ArtifactMetadataEnvelope.seal(
            deliberation=spec.deliberation,
            artifact_type=ArtifactType.UPDATE_SPEC,
            payload=spec.to_dict(),
            source_agent=source_agent,
            target_agent=target_agent,
            run_id=run_id,
            metadata={
                "target_episode_id": spec.target_episode_id,
                "player_archetype": spec.player_archetype,
                "mastery_score": spec.mastery_score,
            },
        )
