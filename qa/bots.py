"""qa/bots.py
Feature 26: Dual Synthetic Playtesting Bots (RusherBot & CautiousBot).
Strictly Rationale-First with deliberation declared strictly first on all models and bot classes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
import math
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.entities.hazard import FloorAcidHazard, LaserSweepHazard, SpikeHazard, SpikeState
from engine.entities.portal import Portal
from engine.entities.projectile import ProjectileManager
from engine.sim.simulator import HeadlessSimulator, SimulationInput
from engine.spatial.aabb import AABB, Vector2
from engine.spatial.navmesh import NavMesh, astar_pathfind
from qa.profiles import CAUTIOUS_PROFILE, RUSHER_PROFILE, PlaytesterProfile


class BasePlaytestingBot(ABC):
    """Abstract base class for synthetic playtesting bots driving HeadlessSimulator."""
    deliberation: str = "Abstract synthetic playtesting bot base class."

    def __init__(
        self,
        profile: PlaytesterProfile,
        tile_size: float = 32.0,
        stuck_threshold_ticks: int = 180,
        stuck_displacement_min: float = 16.0,
        waypoint_tolerance: float = 24.0,
    ):
        self.profile = profile
        self.tile_size = float(tile_size)
        self.stuck_threshold_ticks = int(stuck_threshold_ticks)
        self.stuck_displacement_min = float(stuck_displacement_min)
        self.waypoint_tolerance = float(waypoint_tolerance)

        self.navmesh: Optional[NavMesh] = None
        self.world_waypoints: List[Vector2] = []
        self.waypoint_index: int = 0
        self.position_history: Deque[Tuple[float, float]] = deque(maxlen=max(1, self.stuck_threshold_ticks))
        self.trace_log: List[str] = []
        self.clipping_incidents: int = 0
        self.clipping_locations: List[Tuple[float, float]] = []

    def build_navmesh(self, spec: BaseLevelSpec) -> NavMesh:
        """Rasterizes level spec boundary walls and interior obstacles into a 2D NavMesh."""
        cols = max(1, int(math.ceil(spec.width / self.tile_size)))
        rows = max(1, int(math.ceil(spec.height / self.tile_size)))
        grid = [[0 for _ in range(cols)] for _ in range(rows)]

        for wx, wy, ww, wh in spec.walls:
            min_c = max(0, int(wx // self.tile_size))
            max_c = min(cols - 1, int((wx + ww - 1e-4) // self.tile_size))
            min_r = max(0, int(wy // self.tile_size))
            max_r = min(rows - 1, int((wy + wh - 1e-4) // self.tile_size))
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    grid[r][c] = 1

        self.navmesh = NavMesh(grid, tile_size=self.tile_size)
        return self.navmesh

    def plan_path(self, current_pos: Vector2, target_pos: Vector2, navmesh: NavMesh) -> bool:
        """Plans or re-plans the waypoint route from current_pos to target_pos."""
        st = navmesh.world_to_tile(current_pos.x, current_pos.y)
        gt = navmesh.world_to_tile(target_pos.x, target_pos.y)

        # Snap to adjacent valid neighbor if start tile is occupied
        if not navmesh.is_walkable(st[0], st[1]):
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                if navmesh.is_walkable(st[0] + dx, st[1] + dy):
                    st = (st[0] + dx, st[1] + dy)
                    break

        path = navmesh.find_path(st, gt, allow_diagonal=self.profile.allow_diagonal)
        if not path:
            return False

        self.world_waypoints = [Vector2(*navmesh.tile_to_world(tx, ty)) for tx, ty in path]
        if not self.world_waypoints or self.world_waypoints[-1].distance_to(target_pos) > 16.0:
            self.world_waypoints.append(target_pos)
        self.waypoint_index = 0
        return True

    def check_collision_boundary_integrity(
        self,
        player_aabb: Union[AABB, Any],
        walls: List[Union[AABB, Tuple[float, float, float, float]]],
        epsilon: float = 0.05,
    ) -> bool:
        """Verifies that the player bounding box has not penetrated inside any solid wall."""
        pa = player_aabb.aabb if hasattr(player_aabb, "aabb") else player_aabb
        for w in walls:
            wb = w if isinstance(w, AABB) else AABB(w[0], w[1], w[2], w[3])
            overlap_x = min(pa.right, wb.right) - max(pa.left, wb.left)
            overlap_y = min(pa.bottom, wb.bottom) - max(pa.top, wb.top)
            if overlap_x > epsilon and overlap_y > epsilon:
                self.clipping_incidents += 1
                self.clipping_locations.append((pa.x, pa.y))
                return False
        return True

    def check_stuck_condition(self, current_pos: Vector2) -> bool:
        """Detects whether forward progress has stalled or looped over evaluation window."""
        self.position_history.append((current_pos.x, current_pos.y))
        if len(self.position_history) < self.stuck_threshold_ticks:
            return False

        past_pos = self.position_history[0]
        displacement = math.hypot(current_pos.x - past_pos[0], current_pos.y - past_pos[1])
        return displacement < self.stuck_displacement_min

    @abstractmethod
    def decide_action(
        self,
        player: Any,
        enemies: List[Any],
        hazards: List[Any],
        exit_pos: Vector2,
        walls: List[AABB],
    ) -> Any:
        """Computes the per-tick synthetic controller command."""
        pass

    def run_playtest(self, simulator: HeadlessSimulator, max_ticks: Optional[int] = None) -> Any:
        """Runs the bot through a HeadlessSimulator instance until completion or max ticks."""
        from qa.playtester import PlaytestResult, BotAction

        if max_ticks is not None:
            simulator.max_steps = max_ticks

        walls = simulator.walls
        spec = simulator.level_spec
        navmesh = self.build_navmesh(spec)
        exit_vec = Vector2(spec.exit_point[0], spec.exit_point[1])
        self.plan_path(simulator.player.center, exit_vec, navmesh)
        self.position_history.clear()
        self.clipping_incidents = 0
        self.clipping_locations.clear()
        self.trace_log.clear()

        while not simulator.is_completed:
            # 1. Boundary integrity check
            if not self.check_collision_boundary_integrity(simulator.player.aabb, walls):
                return PlaytestResult(
                    deliberation=f"{self.profile.name.capitalize()}Bot detected wall penetration at {simulator.player.pos.to_tuple()}.",
                    traversable=True,
                    cleared=False,
                    wall_clipping_detected=True,
                    clipping_count=self.clipping_incidents,
                    bot_name=f"{self.profile.name.capitalize()}Bot",
                    dashes_used=simulator.stats.get("dashes", 0),
                    status="REJECTED",
                    failure_type="WallClippingError",
                    failure_coordinates=simulator.player.pos.to_tuple(),
                )

            # 2. Stuck check
            if self.check_stuck_condition(simulator.player.center):
                stuck_coords = (simulator.player.center.x, simulator.player.center.y)
                return PlaytestResult(
                    deliberation=f"{self.profile.name.capitalize()}Bot forward progress stalled at {stuck_coords}.",
                    traversable=True,
                    cleared=False,
                    stuck=True,
                    stuck_position=stuck_coords,
                    bot_name=f"{self.profile.name.capitalize()}Bot",
                    dashes_used=simulator.stats.get("dashes", 0),
                    status="REJECTED",
                    failure_type="StuckBotError",
                    failure_coordinates=stuck_coords,
                )

            # 3. Decide action & step simulator
            bot_action = self.decide_action(
                simulator.player,
                simulator.enemies,
                simulator.hazards,
                exit_vec,
                walls,
            )
            sim_input = SimulationInput(
                deliberation=bot_action.deliberation,
                move_x=bot_action.move_x,
                move_y=bot_action.move_y,
                dash=bot_action.dash_pressed,
                fire=bot_action.fire_pressed,
                aim_x=bot_action.aim_x,
                aim_y=bot_action.aim_y,
            )
            simulator.step(sim_input)

        summary = simulator.get_summary()
        status = "PROMOTED" if summary.cleared else "REJECTED"
        failure_type = None if summary.cleared else (
            "PlayerDeathError" if summary.outcome == "PLAYER_DEATH" else "SimulationTimeoutError"
        )
        return PlaytestResult(
            deliberation=summary.deliberation or f"{self.profile.name.capitalize()}Bot run complete. Outcome: {summary.outcome}.",
            traversable=True,
            cleared=summary.cleared,
            completion_time_sec=summary.duration_seconds,
            completion_ticks=summary.total_steps,
            damage_taken=summary.damage_taken,
            final_health=summary.final_player_health,
            enemies_killed=summary.enemies_killed,
            bot_name=f"{self.profile.name.capitalize()}Bot",
            dashes_used=getattr(summary, "dashes_used", simulator.stats.get("dashes", 0)),
            status=status,
            failure_type=failure_type,
            failure_coordinates=summary.final_player_pos if not summary.cleared else None,
        )


class RusherBot(BasePlaytestingBot):
    """High kinetic pacing, frequent dash usage, aggressive melee approach, and boundary stress testing."""
    deliberation: str = "High-mobility aggressive rusher testing kinetic limits and collision boundaries."

    def __init__(
        self,
        profile: Optional[PlaytesterProfile] = None,
        tile_size: float = 32.0,
        stuck_threshold_ticks: int = 180,
        stuck_displacement_min: float = 16.0,
        waypoint_tolerance: float = 24.0,
    ):
        super().__init__(
            profile=profile or RUSHER_PROFILE,
            tile_size=tile_size,
            stuck_threshold_ticks=stuck_threshold_ticks,
            stuck_displacement_min=stuck_displacement_min,
            waypoint_tolerance=waypoint_tolerance,
        )

    def decide_action(
        self,
        player: Any,
        enemies: List[Any],
        hazards: List[Any],
        exit_pos: Vector2,
        walls: List[AABB],
    ) -> Any:
        from qa.playtester import BotAction

        p_center = player.center

        # 1. Waypoint steering
        if not self.world_waypoints or self.waypoint_index >= len(self.world_waypoints):
            if self.navmesh:
                self.plan_path(p_center, exit_pos, self.navmesh)

        if self.world_waypoints and self.waypoint_index < len(self.world_waypoints):
            target_wp = self.world_waypoints[self.waypoint_index]
            if p_center.distance_to(target_wp) < self.waypoint_tolerance and self.waypoint_index < len(self.world_waypoints) - 1:
                self.waypoint_index += 1
                target_wp = self.world_waypoints[self.waypoint_index]
            to_target = target_wp - p_center
            move_dir = to_target.normalized() if to_target.length_squared() > 0.01 else Vector2(0.0, 0.0)
        else:
            to_exit = exit_pos - p_center
            move_dir = to_exit.normalized() if to_exit.length_squared() > 0.01 else Vector2(0.0, 0.0)

        # 2. Hazard proximity
        hazard_threat = False
        hazard_escape_vec = Vector2(0.0, 0.0)
        for h in hazards:
            if isinstance(h, SpikeHazard) and h.state in (SpikeState.WARNING, SpikeState.EXTENDED):
                if p_center.distance_to(h.center) < (h.size.x * 0.5 + self.profile.hazard_caution_distance_px):
                    diff = p_center - h.center
                    hazard_escape_vec += (diff.normalized() if diff.length_squared() > 0.01 else Vector2(0.0, -1.0))
                    hazard_threat = True
            elif isinstance(h, FloorAcidHazard) and h.aabb.intersects(player.aabb):
                diff = p_center - h.center
                hazard_escape_vec += (diff.normalized() if diff.length_squared() > 0.01 else Vector2(1.0, 0.0))
                hazard_threat = True

        # 3. Aggressive Combat Engagement
        alive_enemies = [e for e in enemies if getattr(e, "is_alive", True)]
        aim_target = exit_pos
        fire_active = False
        closing_in_melee = False

        if alive_enemies:
            closest_enemy = min(alive_enemies, key=lambda e: p_center.distance_to(e.center))
            dist = p_center.distance_to(closest_enemy.center)
            aim_target = closest_enemy.center
            if dist <= self.profile.weapon_range_px:
                fire_active = True
            if dist > self.profile.standoff_distance_px:
                to_enemy = (closest_enemy.center - p_center).normalized()
                move_dir = (move_dir * 0.3 + to_enemy * 0.7).normalized()
                closing_in_melee = True

        # 4. Frequent Dash Cadence along path & collision boundaries
        dash_requested = False
        can_dash = getattr(player, "dash_cooldown", 0.0) <= 0.0 and getattr(player, "stamina", 0.0) >= getattr(player, "dash_cost", 25.0)

        if can_dash:
            if hazard_threat:
                dash_requested = True
                move_dir = (move_dir * 0.3 + hazard_escape_vec.normalized() * 0.7).normalized()
            elif closing_in_melee or (self.profile.dash_frequency_bias > 0.5 and move_dir.length_squared() > 0.1):
                dash_requested = True

        delib = (
            f"RusherBot tick: waypoint {self.waypoint_index}/{len(self.world_waypoints)}, "
            f"closing_melee={closing_in_melee}, fire={fire_active}, dash={dash_requested}."
        )
        return BotAction(
            deliberation=delib,
            move_x=move_dir.x,
            move_y=move_dir.y,
            aim_x=aim_target.x,
            aim_y=aim_target.y,
            dash_pressed=dash_requested,
            fire_pressed=fire_active,
        )


class CautiousBot(BasePlaytestingBot):
    """Methodical pacing, standoff distance maintenance, kiting behavior, and high survivability."""
    deliberation: str = "Methodical cautious player maintaining standoff distance, kiting threats, and avoiding hazards."

    def __init__(
        self,
        profile: Optional[PlaytesterProfile] = None,
        tile_size: float = 32.0,
        stuck_threshold_ticks: int = 180,
        stuck_displacement_min: float = 16.0,
        waypoint_tolerance: float = 24.0,
    ):
        super().__init__(
            profile=profile or CAUTIOUS_PROFILE,
            tile_size=tile_size,
            stuck_threshold_ticks=stuck_threshold_ticks,
            stuck_displacement_min=stuck_displacement_min,
            waypoint_tolerance=waypoint_tolerance,
        )

    def decide_action(
        self,
        player: Any,
        enemies: List[Any],
        hazards: List[Any],
        exit_pos: Vector2,
        walls: List[AABB],
    ) -> Any:
        from qa.playtester import BotAction

        p_center = player.center

        # 1. Waypoint steering
        if not self.world_waypoints or self.waypoint_index >= len(self.world_waypoints):
            if self.navmesh:
                self.plan_path(p_center, exit_pos, self.navmesh)

        if self.world_waypoints and self.waypoint_index < len(self.world_waypoints):
            target_wp = self.world_waypoints[self.waypoint_index]
            if p_center.distance_to(target_wp) < self.waypoint_tolerance and self.waypoint_index < len(self.world_waypoints) - 1:
                self.waypoint_index += 1
                target_wp = self.world_waypoints[self.waypoint_index]
            to_target = target_wp - p_center
            move_dir = to_target.normalized() if to_target.length_squared() > 0.01 else Vector2(0.0, 0.0)
        else:
            to_exit = exit_pos - p_center
            move_dir = to_exit.normalized() if to_exit.length_squared() > 0.01 else Vector2(0.0, 0.0)

        # 2. Wide Hazard Safety Margin (75px)
        hazard_threat = False
        hazard_escape_vec = Vector2(0.0, 0.0)
        for h in hazards:
            if isinstance(h, SpikeHazard):
                if h.state in (SpikeState.WARNING, SpikeState.EXTENDED):
                    d = p_center.distance_to(h.center)
                    if d < (h.size.x * 0.5 + self.profile.hazard_caution_distance_px):
                        diff = p_center - h.center
                        hazard_escape_vec += (diff.normalized() if diff.length_squared() > 0.01 else Vector2(0.0, -1.0))
                        hazard_threat = True
            elif isinstance(h, FloorAcidHazard):
                caution_box = AABB(
                    h.aabb.x - self.profile.hazard_caution_distance_px,
                    h.aabb.y - self.profile.hazard_caution_distance_px,
                    h.aabb.width + 2 * self.profile.hazard_caution_distance_px,
                    h.aabb.height + 2 * self.profile.hazard_caution_distance_px,
                )
                if caution_box.intersects(player.aabb):
                    diff = p_center - h.center
                    hazard_escape_vec += (diff.normalized() if diff.length_squared() > 0.01 else Vector2(1.0, 0.0))
                    hazard_threat = True
            elif isinstance(h, LaserSweepHazard):
                p1, p2 = h.get_endpoints()
                seg = p2 - p1
                seg_len_sq = seg.length_squared()
                if seg_len_sq > 0.0:
                    t = max(0.0, min(1.0, (p_center - p1).dot(seg) / seg_len_sq))
                    closest = p1 + seg * t
                    if p_center.distance_to(closest) < self.profile.hazard_caution_distance_px:
                        diff = p_center - closest
                        hazard_escape_vec += (diff.normalized() if diff.length_squared() > 0.01 else Vector2(0.0, -1.0))
                        hazard_threat = True

        # 3. Kiting & Standoff Distance Maintenance (260px)
        alive_enemies = [e for e in enemies if getattr(e, "is_alive", True)]
        aim_target = exit_pos
        fire_active = False
        kiting_vec = Vector2(0.0, 0.0)
        needs_kiting = False

        if alive_enemies:
            closest_enemy = min(alive_enemies, key=lambda e: p_center.distance_to(e.center))
            dist = p_center.distance_to(closest_enemy.center)
            aim_target = closest_enemy.center
            if dist <= self.profile.weapon_range_px:
                fire_active = True

            if dist < self.profile.standoff_distance_px:
                away = (p_center - closest_enemy.center).normalized()
                kiting_vec += away
                needs_kiting = True

        # 4. Methodical Movement & Emergency Dash Only
        dash_requested = False
        can_dash = getattr(player, "dash_cooldown", 0.0) <= 0.0 and getattr(player, "stamina", 0.0) >= getattr(player, "dash_cost", 25.0)

        final_move = move_dir
        if hazard_threat:
            final_move = (move_dir * 0.2 + hazard_escape_vec.normalized() * 0.8).normalized()
            if can_dash:
                dash_requested = True
        elif needs_kiting:
            final_move = (move_dir * 0.3 + kiting_vec.normalized() * 0.7).normalized()
            if can_dash and alive_enemies and min(p_center.distance_to(e.center) for e in alive_enemies) < 70.0:
                dash_requested = True

        delib = (
            f"CautiousBot tick: waypoint {self.waypoint_index}/{len(self.world_waypoints)}, "
            f"kiting={needs_kiting}, hazard_threat={hazard_threat}, fire={fire_active}, dash={dash_requested}."
        )
        return BotAction(
            deliberation=delib,
            move_x=final_move.x,
            move_y=final_move.y,
            aim_x=aim_target.x,
            aim_y=aim_target.y,
            dash_pressed=dash_requested,
            fire_pressed=fire_active,
        )
