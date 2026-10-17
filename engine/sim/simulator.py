"""engine/sim/simulator.py
Decoupled Headless Simulation Host for discrete fixed-timestep cartridge execution.
"""
from __future__ import annotations
import os
# Force SDL headless drivers before pygame initialization
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import copy
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import config
from engine.api.contracts import (
    BaseCartridge,
    BaseEnemy,
    BaseHazard,
    BaseLevelSpec,
    BasePlayer,
)
from engine.entities.enemies import BaseEnemy, ChaserEnemy, RangedEnemy
from engine.entities.hazard import FloorAcidHazard, Hazard, LaserSweepHazard, SpikeHazard
from engine.entities.player import Player
from engine.entities.portal import Portal
from engine.entities.projectile import ProjectileManager
from engine.loader import CartridgeLoader, GenericCartridge
from engine.spatial.aabb import AABB, Vector2
from engine.spatial.collision import move_and_resolve_walls
from engine.spatial.grid import SpatialHashGrid
from engine.telemetry.db import TelemetryDB
from engine.telemetry.models import TelemetryEventRecord, TelemetryEventType
from engine.telemetry.recorder import NullTelemetryRecorder, TelemetryRecorder

logger = logging.getLogger("HeadlessSimulator")


from engine.sim.headless import (
    InMemoryTelemetryRecorder,
    SimulationInput,
    SimulationState,
    SimulationStepResult,
    SimulationSummary,
)


class HeadlessSimulator:
    """Discrete fixed-timestep headless simulation host for dynamic cartridges and playtesting."""
    deliberation: str = "Discrete fixed-timestep headless simulation host for dynamic cartridges and playtesting."

    def __init__(
        self,
        cartridge: Optional[BaseCartridge] = None,
        level_spec: Optional[BaseLevelSpec] = None,
        cartridge_path: Optional[str | Path] = None,
        episode_number: int = 1,
        dt: float = 1.0 / 60.0,
        map_width: float = 1280.0,
        map_height: float = 720.0,
        telemetry_mode: str = "memory",  # "memory" | "sqlite" | "null"
        telemetry_db_path: Optional[str | Path] = None,
        ignore_combat_lock: bool = False,
        max_steps: Optional[int] = None,
    ):
        self.dt = float(dt)
        self.episode_number = int(episode_number)
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.ignore_combat_lock = bool(ignore_combat_lock)
        self.max_steps = max_steps

        self.step_count = 0
        self.elapsed_time = 0.0
        self.is_completed = False
        self.outcome: Optional[str] = None

        # Statistics
        self.stats = {
            "dashes": 0,
            "shots_fired": 0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "enemies_killed": 0,
        }

        # Telemetry Initialization
        self.telemetry_mode = telemetry_mode.lower()
        if self.telemetry_mode == "sqlite":
            db_path = telemetry_db_path or config.DB_PATH
            self.telemetry_recorder: Any = TelemetryRecorder(db_path=db_path)
        elif self.telemetry_mode == "null":
            self.telemetry_recorder = NullTelemetryRecorder()
        else:
            self.telemetry_recorder = InMemoryTelemetryRecorder()
        self.frame_times: List[float] = []

        # Load Cartridge & Spec
        self.cartridge: Optional[BaseCartridge] = None
        if cartridge is not None:
            self.cartridge = cartridge
        elif cartridge_path is not None:
            cpath = Path(cartridge_path)
            if not cpath.exists():
                raise FileNotFoundError(f"Cartridge directory not found: {cpath}")
            self.cartridge = CartridgeLoader.load_cartridge(cpath)

        # Determine mode: Scene Mode (Open-Genre) vs Spatial Mode (Backward Compatibility)
        has_custom_scenes = False
        overrides_get_scenes = (
            self.cartridge is not None
            and hasattr(self.cartridge, "get_scenes")
            and getattr(type(self.cartridge), "get_scenes", None) is not BaseCartridge.get_scenes
        )
        if self.cartridge is not None and hasattr(self.cartridge, "get_scenes"):
            try:
                from engine.core.engine import Engine
                self.engine = Engine(headless=True)
                self.engine.set_telemetry_recorder(self.telemetry_recorder)
                self.engine.telemetry_recorder = self.telemetry_recorder
                scenes = self.cartridge.get_scenes(self.engine)
                if scenes:
                    has_custom_scenes = True
                    self.scenes = scenes
                    for sc_name, sc in scenes.items():
                        if getattr(sc, "engine", None) is None:
                            setattr(sc, "engine", self.engine)
                        if getattr(sc, "telemetry_recorder", None) is None:
                            setattr(sc, "telemetry_recorder", self.telemetry_recorder)
                        self.engine.scene_manager.register_scene(sc_name, sc)
                    initial = self.cartridge.get_initial_scene(self.engine)
                    initial_key = "main"
                    if isinstance(initial, str) and initial in scenes:
                        self.current_scene = scenes[initial]
                        initial_key = initial
                    elif initial is not None and hasattr(initial, "update"):
                        self.current_scene = initial
                        initial_key = "cartridge_initial"
                        self.engine.scene_manager.register_scene(initial_key, initial)
                    else:
                        first_key = next(iter(scenes.keys()))
                        self.current_scene = scenes[first_key]
                        initial_key = first_key
                    if getattr(self.current_scene, "engine", None) is None:
                        setattr(self.current_scene, "engine", self.engine)
                    if getattr(self.current_scene, "telemetry_recorder", None) is None:
                        setattr(self.current_scene, "telemetry_recorder", self.telemetry_recorder)
                    self.current_scene_name = getattr(self.current_scene, "scene_name", initial_key)
                    self.scene_history = [self.current_scene_name]
                    if hasattr(self.cartridge, "setup"):
                        self.cartridge.setup(self.engine)
                    self.engine.scene_manager.switch_to(initial_key)
            except Exception as e:
                if overrides_get_scenes:
                    raise
                logger.warning("Could not initialize scene mode on cartridge: %s", e)
                has_custom_scenes = False

        if has_custom_scenes:
            self.mode = "scene"
            self._init_dummy_spatial_properties()
        else:
            self.mode = "spatial"
            if level_spec is not None:
                self.level_spec = level_spec
            elif self.cartridge is not None and hasattr(self.cartridge, "get_level_spec"):
                self.level_spec = self.cartridge.get_level_spec(self.episode_number)
            else:
                self.level_spec = BaseLevelSpec(
                    level_id=f"level_{self.episode_number:02d}",
                    name="Default Chamber",
                    width=self.map_width,
                    height=self.map_height,
                    spawn_point=(120.0, 360.0),
                    exit_point=(1160.0, 360.0),
                    walls=[
                        (0, 0, self.map_width, 32),
                        (0, self.map_height - 32, self.map_width, 32),
                        (0, 0, 32, self.map_height),
                        (self.map_width - 32, 0, 32, self.map_height),
                    ]
                )
            self._build_environment()

    def _init_dummy_spatial_properties(self) -> None:
        """Initializes safe fallback spatial properties for open-genre cartridges in scene mode."""
        self.level_spec = BaseLevelSpec(
            level_id=f"scene_level_{self.episode_number:02d}",
            name="Scene Level",
            width=self.map_width,
            height=self.map_height,
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            walls=[],
            enemies=[],
            hazards=[],
        )
        self.walls: List[AABB] = []
        self.spatial_grid = SpatialHashGrid(cell_size=64.0)
        self.player = Player(120.0, 360.0, self.telemetry_recorder)
        self.enemies: List[BaseEnemy] = []
        self.hazards: List[Hazard] = []
        self.portal = Portal(1160.0, 360.0)
        self.projectile_manager = ProjectileManager(self.telemetry_recorder)

        # Telemetry: start session & episode
        self.telemetry_recorder.start_session(player_profile_tag="headless_scene_sim")
        self.telemetry_recorder.start_episode(
            level_id=self.level_spec.level_id,
            episode_index=self.episode_number,
            player_character="hero",
            metadata={"map_width": self.map_width, "map_height": self.map_height, "mode": "scene"}
        )

    def _build_environment(self) -> None:
        """Instantiates all entities, walls, hazards, and portals from level specification."""
        spec = self.level_spec
        self.map_width = spec.width
        self.map_height = spec.height

        # Build walls
        self.walls: List[AABB] = [
            w if isinstance(w, AABB) else AABB(w[0], w[1], w[2], w[3]) for w in spec.walls
        ]

        # Spatial hash grid
        self.spatial_grid = SpatialHashGrid(cell_size=64.0)
        for w in self.walls:
            self.spatial_grid.insert(w, w)

        # Instantiate entities
        player = None
        enemies = None
        hazards = None
        if self.cartridge is not None:
            try:
                player = self.cartridge.create_player(spec.spawn_point[0], spec.spawn_point[1], self.telemetry_recorder)
            except Exception:
                player = None
            try:
                enemies = self.cartridge.create_enemies(spec)
            except Exception:
                enemies = None
            try:
                hazards = self.cartridge.create_hazards(spec)
            except Exception:
                hazards = None

        if player is None:
            self.player = Player(spec.spawn_point[0], spec.spawn_point[1], self.telemetry_recorder)
        else:
            self.player = player

        if enemies is None or (len(enemies) == 0 and spec.enemies):
            self.enemies = []
            for idx, e_def in enumerate(spec.enemies):
                e_type = e_def.get("type", "chaser").lower()
                ex = float(e_def.get("x", 400.0))
                ey = float(e_def.get("y", 300.0))
                eid = e_def.get("id", f"{e_type}_{idx + 1}")
                if e_type == "chaser":
                    self.enemies.append(ChaserEnemy(eid, ex, ey))
                elif e_type == "ranged":
                    self.enemies.append(RangedEnemy(eid, ex, ey))
        else:
            self.enemies = enemies

        if hazards is None or (len(hazards) == 0 and spec.hazards):
            self.hazards = []
            for idx, h_def in enumerate(spec.hazards):
                h_type = h_def.get("type", "spikes").lower()
                hx = float(h_def.get("x", 400.0))
                hy = float(h_def.get("y", 300.0))
                hid = h_def.get("id", f"{h_type}_{idx + 1}")
                if h_type == "spikes":
                    self.hazards.append(SpikeHazard(hid, hx, hy, float(h_def.get("w", 48.0)), float(h_def.get("h", 48.0))))
                elif h_type == "floor_acid":
                    self.hazards.append(FloorAcidHazard(hid, hx, hy, float(h_def.get("w", 96.0)), float(h_def.get("h", 96.0))))
                elif h_type == "laser_sweep":
                    self.hazards.append(LaserSweepHazard(
                        hid, hx, hy,
                        length=float(h_def.get("length", 280.0)),
                        mid_angle=float(h_def.get("mid_angle", 0.0)),
                        sweep_amplitude=float(h_def.get("sweep_amplitude", 1.0)),
                        speed=float(h_def.get("speed", 1.2))
                    ))
        else:
            self.hazards = hazards

        self.portal = Portal(spec.exit_point[0], spec.exit_point[1])
        self.projectile_manager = ProjectileManager(self.telemetry_recorder)

        # Telemetry: start session & episode
        self.telemetry_recorder.start_session(player_profile_tag="headless_sim")
        self.telemetry_recorder.start_episode(
            level_id=spec.level_id,
            episode_index=self.episode_number,
            player_character="hero",
            metadata={"map_width": self.map_width, "map_height": self.map_height}
        )
        self.telemetry_recorder.emit_event(
            event_type="EPISODE_START",
            pos=spec.spawn_point,
            vel=(0.0, 0.0),
            health=self.player.health,
            payload={"episode_number": self.episode_number, "level_id": spec.level_id}
        )

    # ------------------------------------------------------------------------
    # Backward compatibility accessors for test_tier2_playtest.py
    # ------------------------------------------------------------------------
    @property
    def player_x(self) -> float:
        return self.player.pos.x

    @player_x.setter
    def player_x(self, val: float) -> None:
        self.player.pos.x = float(val)

    @property
    def player_y(self) -> float:
        return self.player.pos.y

    @player_y.setter
    def player_y(self, val: float) -> None:
        self.player.pos.y = float(val)

    @property
    def player_vx(self) -> float:
        return self.player.vel.x

    @player_vx.setter
    def player_vx(self, val: float) -> None:
        self.player.vel.x = float(val)

    @property
    def player_vy(self) -> float:
        return self.player.vel.y

    @player_vy.setter
    def player_vy(self, val: float) -> None:
        self.player.vel.y = float(val)

    @property
    def player_width(self) -> float:
        return self.player.size.x

    @property
    def player_height(self) -> float:
        return self.player.size.y

    @property
    def player_health(self) -> float:
        return self.player.health

    @player_health.setter
    def player_health(self, val: float) -> None:
        self.player.health = float(val)

    @property
    def obstacles(self) -> List[Tuple[float, float, float, float]]:
        return [(w.x, w.y, w.width, w.height) for w in self.walls]

    def add_obstacle(self, x: float, y: float, w: float, h: float) -> None:
        box = AABB(x, y, w, h)
        self.walls.append(box)
        self.spatial_grid.insert(box, box)

    # ------------------------------------------------------------------------
    # Core Computational Step
    # ------------------------------------------------------------------------
    def step(
        self,
        action: Optional[Union[SimulationInput, float]] = None,
        input_ay: float = 0.0,
        is_dash: bool = False,
        is_fire: bool = False,
        aim_x: Optional[float] = None,
        aim_y: Optional[float] = None,
        deliberation: str = "",
    ) -> SimulationStepResult:
        """Executes one discrete fixed-timestep tick (dt=1/60s)."""
        t0 = time.perf_counter()
        if self.is_completed:
            duration = time.perf_counter() - t0
            self.frame_times.append(duration)
            return self._make_step_result(deliberation or "Simulation already completed.")

        # Harmonize input parameters
        if isinstance(action, SimulationInput):
            cmd = action
        elif isinstance(action, (int, float)):
            cmd = SimulationInput(
                deliberation=deliberation,
                move_x=float(action),
                move_y=float(input_ay),
                dash=is_dash,
                fire=is_fire,
                aim_x=float(aim_x if aim_x is not None else 0.0),
                aim_y=float(aim_y if aim_y is not None else 0.0),
            )
        else:
            cmd = SimulationInput(deliberation=deliberation)

        dt = self.dt

        # Scene Mode Execution
        if getattr(self, "mode", "spatial") == "scene":
            import pygame
            if hasattr(self.current_scene, "handle_event"):
                if cmd.fire:
                    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
                elif cmd.dash:
                    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LSHIFT)
                elif cmd.move_x > 0:
                    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d)
                elif cmd.move_x < 0:
                    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
                elif cmd.move_y > 0:
                    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s)
                elif cmd.move_y < 0:
                    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w)
                else:
                    ev = pygame.event.Event(pygame.NOEVENT)
                self.current_scene.handle_event(ev)

            if hasattr(self.current_scene, "update"):
                self.current_scene.update(dt)

            if hasattr(self.current_scene, "render"):
                surf = getattr(getattr(self, "engine", None), "logical_surface", None)
                if surf is None:
                    surf = pygame.Surface((int(self.map_width), int(self.map_height)))
                self.current_scene.render(surf)

            self.step_count += 1
            self.elapsed_time += dt
            if hasattr(self.telemetry_recorder, "tick_frame"):
                self.telemetry_recorder.tick_frame()

            if self.max_steps is not None and self.step_count >= self.max_steps:
                self.is_completed = True
                self.outcome = "COMPLETED"

            duration = time.perf_counter() - t0
            self.frame_times.append(duration)

            return SimulationStepResult(
                deliberation=cmd.deliberation or "Scene step executed.",
                step=self.step_count,
                elapsed_time=self.elapsed_time,
                player_pos=getattr(self.current_scene, "player_pos", (120.0, 360.0)),
                player_vel=getattr(self.current_scene, "player_vel", (0.0, 0.0)),
                player_health=getattr(self.current_scene, "player_health", 100.0),
                player_stamina=getattr(self.current_scene, "player_stamina", 100.0),
                is_dashing=cmd.dash,
                active_enemies=len(getattr(self.current_scene, "enemies", [])),
                active_projectiles=len(getattr(self.current_scene, "projectiles", [])),
                is_completed=self.is_completed or getattr(self.current_scene, "is_completed", False),
                outcome=self.outcome or getattr(self.current_scene, "outcome", None),
            )

        # Spatial Mode Execution
        # 1. Update Hazards & Environmental Status Effects
        self.player.speed_multiplier = 1.0
        for hazard in self.hazards:
            hazard.update(dt, self.player, self.telemetry_recorder)

        # 2. Update Player Input & Movement
        move_vec = Vector2(cmd.move_x, cmd.move_y)
        c = self.player.center
        if cmd.aim_x != 0.0 or cmd.aim_y != 0.0:
            aim_vec = Vector2(cmd.aim_x, cmd.aim_y)
        elif move_vec.length_squared() > 0.01:
            aim_vec = c + move_vec * 100.0
        else:
            aim_vec = Vector2(c.x + 100.0, c.y)

        self.player.handle_input(
            move_input=move_vec,
            mouse_canvas_pos=aim_vec,
            dash_pressed=cmd.dash,
            fire_pressed=cmd.fire,
            projectile_manager=self.projectile_manager
        )
        self.player.update(dt, self.walls)

        if cmd.dash and self.player.is_dashing:
            self.stats["dashes"] += 1
        if cmd.fire and self.player.weapon.cooldown > (self.player.weapon.fire_rate - dt):
            self.stats["shots_fired"] += 1

        # 3. Update Enemies
        for enemy in self.enemies:
            if hasattr(enemy, "update_ai"):
                try:
                    enemy.update_ai(dt, self.player, self.walls, self.projectile_manager, self.telemetry_recorder)
                except TypeError:
                    try:
                        enemy.update_ai(dt, self.player, self.walls, self.telemetry_recorder)
                    except TypeError:
                        enemy.update_ai(dt, self.player, self.walls)
            elif hasattr(enemy, "update"):
                try:
                    enemy.update(dt, self.player, self.walls, self.projectile_manager, self.telemetry_recorder)
                except TypeError:
                    try:
                        enemy.update(dt, self.player, self.walls)
                    except TypeError:
                        enemy.update(dt)

        self.stats["enemies_killed"] = sum(1 for e in self.enemies if not e.is_alive)

        # 4. Update Projectiles
        self.projectile_manager.update(dt, self.walls, self.player, self.enemies)

        # 5. Update Exit Portal
        portal_enemies = [] if self.ignore_combat_lock else self.enemies
        if self.ignore_combat_lock:
            self.portal.is_unlocked = True
        cleared = self.portal.update(dt, portal_enemies, self.player, self.telemetry_recorder)
        if cleared or (self.ignore_combat_lock and self.player.is_alive and self.portal.circle.overlaps_aabb(self.player.aabb)):
            self.is_completed = True
            self.outcome = "CLEARED"
            self.telemetry_recorder.emit_event(
                event_type="LEVEL_EXIT_REACHED",
                pos=(self.portal.center.x, self.portal.center.y),
                vel=(self.player.vel.x, self.player.vel.y),
                health=self.player.health,
                payload={"time": self.elapsed_time}
            )
            self.telemetry_recorder.end_episode(outcome="CLEARED", exit_pos=self.portal.center.to_tuple())
            duration = time.perf_counter() - t0
            self.frame_times.append(duration)
            return self._make_step_result(f"Level cleared at step {self.step_count}.")

        # 6. Check Player Death
        if not self.player.is_alive:
            self.is_completed = True
            self.outcome = "PLAYER_DEATH"
            self.telemetry_recorder.emit_event(
                event_type="PLAYER_DEATH",
                pos=(self.player.center.x, self.player.center.y),
                vel=(0.0, 0.0),
                health=0.0,
                payload={"time": self.elapsed_time}
            )
            self.telemetry_recorder.end_episode(outcome="PLAYER_DEATH", exit_pos=self.player.center.to_tuple())
            duration = time.perf_counter() - t0
            self.frame_times.append(duration)
            return self._make_step_result("Player died.")

        # 7. Advance Step Count & Check Max Steps
        self.step_count += 1
        self.elapsed_time += dt
        self.telemetry_recorder.tick_frame()

        if self.max_steps is not None and self.step_count >= self.max_steps:
            self.is_completed = True
            self.outcome = "TIMEOUT"
            self.telemetry_recorder.end_episode(outcome="TIMEOUT", exit_pos=self.player.center.to_tuple())

        duration = time.perf_counter() - t0
        self.frame_times.append(duration)

        return self._make_step_result(cmd.deliberation)

    def _make_step_result(self, deliberation: str) -> SimulationStepResult:
        return SimulationStepResult(
            deliberation=deliberation,
            step=self.step_count,
            elapsed_time=self.elapsed_time,
            player_pos=(self.player.pos.x, self.player.pos.y),
            player_vel=(self.player.vel.x, self.player.vel.y),
            player_health=self.player.health,
            player_stamina=self.player.stamina,
            is_dashing=self.player.is_dashing,
            active_enemies=sum(1 for e in self.enemies if e.is_alive),
            active_projectiles=len(self.projectile_manager.projectiles),
            is_completed=self.is_completed,
            outcome=self.outcome,
        )

    @property
    def avg_frame_time_ms(self) -> float:
        if not self.frame_times:
            return 0.0
        return (sum(self.frame_times) / len(self.frame_times)) * 1000.0

    @property
    def max_frame_time_ms(self) -> float:
        if not self.frame_times:
            return 0.0
        return max(self.frame_times) * 1000.0

    @property
    def frame_budget_passed(self) -> bool:
        if not self.frame_times:
            return True
        return self.avg_frame_time_ms < 16.6 and self.max_frame_time_ms < 33.3

    # ------------------------------------------------------------------------
    # Batch & Policy Execution
    # ------------------------------------------------------------------------
    def run_steps(self, actions: List[SimulationInput]) -> List[SimulationStepResult]:
        """Executes a series of pre-scripted inputs."""
        results = []
        for act in actions:
            res = self.step(act)
            results.append(res)
            if self.is_completed:
                break
        return results

    def run_until_complete(
        self,
        policy_fn: Optional[Callable[[SimulationState], SimulationInput]] = None,
        max_steps: int = 3600,
    ) -> SimulationSummary:
        """Runs the simulation loop continuously until completion or timeout."""
        limit = min(max_steps, self.max_steps or max_steps)
        while not self.is_completed and self.step_count < limit:
            if policy_fn is not None:
                action = policy_fn(self.get_state())
            else:
                action = SimulationInput()
            self.step(action)

        if not self.is_completed and self.step_count >= limit:
            self.is_completed = True
            self.outcome = "TIMEOUT"
            self.telemetry_recorder.end_episode(outcome="TIMEOUT", exit_pos=self.player.center.to_tuple())

        return self.get_summary()

    # ------------------------------------------------------------------------
    # State Inspection & Spatial Queries
    # ------------------------------------------------------------------------
    def get_state(self) -> SimulationState:
        """Returns snapshot of current simulation state."""
        return SimulationState(
            deliberation="Simulation state snapshot.",
            step_count=self.step_count,
            elapsed_time=self.elapsed_time,
            player_pos=(self.player.pos.x, self.player.pos.y),
            player_vel=(self.player.vel.x, self.player.vel.y),
            player_health=self.player.health,
            player_stamina=self.player.stamina,
            is_dashing=self.player.is_dashing,
            enemies=[
                {"id": e.entity_id, "pos": (e.pos.x, e.pos.y), "health": e.health, "is_alive": e.is_alive}
                for e in self.enemies
            ],
            active_projectiles=len(self.projectile_manager.projectiles),
            hazards_count=len(self.hazards),
            portal_unlocked=self.portal.is_unlocked,
            portal_pos=(self.portal.pos.x, self.portal.pos.y),
            is_completed=self.is_completed,
            outcome=self.outcome,
        )

    def get_summary(self) -> SimulationSummary:
        """Generates comprehensive run summary."""
        remaining_enemies = sum(1 for e in self.enemies if e.is_alive)
        event_count = (
            len(self.telemetry_recorder.events)
            if isinstance(self.telemetry_recorder, InMemoryTelemetryRecorder)
            else 0
        )
        return SimulationSummary(
            deliberation="Post-simulation evaluation summary.",
            total_steps=self.step_count,
            duration_seconds=self.elapsed_time,
            outcome=self.outcome or "IN_PROGRESS",
            cleared=(self.outcome == "CLEARED"),
            damage_dealt=self.stats["damage_dealt"],
            damage_taken=self.stats["damage_taken"],
            dashes_used=self.stats["dashes"],
            shots_fired=self.stats["shots_fired"],
            enemies_killed=self.stats["enemies_killed"],
            remaining_enemies=remaining_enemies,
            final_player_health=self.player.health,
            final_player_pos=(self.player.pos.x, self.player.pos.y),
            exit_pos=(self.portal.pos.x, self.portal.pos.y),
            telemetry_event_count=event_count,
            metadata={"episode_number": self.episode_number, "telemetry_mode": self.telemetry_mode},
        )

    def is_out_of_bounds(self) -> bool:
        """Returns True if player is outside logical map dimensions."""
        px = self.player.pos.x
        py = self.player.pos.y
        pw = self.player.size.x
        ph = self.player.size.y
        return px < 0.0 or (px + pw) > self.map_width or py < 0.0 or (py + ph) > self.map_height

    def check_wall_clipping(self) -> bool:
        """Returns True if player intersects any solid wall bounding box."""
        p_aabb = self.player.aabb
        for w in self.walls:
            if p_aabb.intersects(w):
                return True
        return False

    def reset(self) -> None:
        """Resets the simulation to initial spawn state."""
        self.step_count = 0
        self.elapsed_time = 0.0
        self.is_completed = False
        self.outcome = None
        self.stats = {
            "dashes": 0,
            "shots_fired": 0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "enemies_killed": 0,
        }
        if isinstance(self.telemetry_recorder, InMemoryTelemetryRecorder):
            self.telemetry_recorder.clear()
        self._build_environment()

    def close(self) -> None:
        """Flushes and closes telemetry recorder."""
        if hasattr(self.telemetry_recorder, "close"):
            self.telemetry_recorder.close()

    def __enter__(self) -> HeadlessSimulator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
