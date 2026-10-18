"""engine/sim/headless.py
Unified Headless Simulation Host and Epistemic Data Models.
Provides discrete fixed-timestep execution in Scene Mode (open-genre) and Spatial Mode.
All models strictly declare deliberation first.
"""
from __future__ import annotations
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from engine.telemetry.models import TelemetryEventRecord, TelemetryEventType


class InMemoryTelemetryRecorder:
    """In-memory telemetry event recorder for headless simulations."""
    deliberation: str = "Capturing telemetry events in-memory for testing and simulation verification."

    def __init__(self, session_id: Optional[str] = None):
        self.session_id: str = session_id or str(uuid.uuid4())
        self.episode_index: int = 0
        self.current_episode_id: Optional[str] = None
        self.frame_index: int = 0
        self.events: List[TelemetryEventRecord] = []
        self.sessions: List[Dict[str, Any]] = []
        self.episodes: List[Dict[str, Any]] = []
        self._stats = {
            "dashes": 0,
            "shots_fired": 0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
        }

    def start_session(
        self,
        player_profile_tag: str = "headless",
        platform: str = "macOS",
        resolution: str = "1280x720",
        target_fps: int = 60,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.sessions.append({
            "session_id": self.session_id,
            "player_profile_tag": player_profile_tag,
            "platform": platform,
            "resolution": resolution,
            "target_fps": target_fps,
            "metadata": metadata or {},
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        return self.session_id

    def end_session(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        pass

    def start_episode(
        self,
        level_id: str = "level_01",
        episode_index: Optional[int] = None,
        player_character: str = "hero",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if episode_index is not None:
            self.episode_index = episode_index
        else:
            self.episode_index += 1

        self.current_episode_id = f"ep_{self.session_id}_{self.episode_index:03d}"
        self.frame_index = 0
        self.episodes.append({
            "episode_id": self.current_episode_id,
            "level_id": level_id,
            "episode_index": self.episode_index,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "ACTIVE",
            "metadata": metadata or {},
        })
        return self.current_episode_id

    def end_episode(
        self,
        outcome: str = "CLEARED",
        score: Optional[int] = None,
        exit_pos: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.episodes:
            self.episodes[-1]["status"] = outcome
            self.episodes[-1]["exit_pos"] = exit_pos
            self.episodes[-1]["ended_at"] = datetime.now(timezone.utc).isoformat()
            if metadata:
                self.episodes[-1]["metadata"].update(metadata)

    def emit_custom_probe(
        self,
        probe_name: str,
        pos: Tuple[float, float] = (0.0, 0.0),
        vel: Tuple[float, float] = (0.0, 0.0),
        health: Optional[float] = None,
        category: str = "custom",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {"category": category, "probe_name": probe_name, **(data or {})}
        self.emit_event(
            event_type="CUSTOM_PROBE",
            pos=pos,
            vel=vel,
            health=health,
            payload=payload,
            probe_name=probe_name,
        )

    def emit_event(
        self,
        event_type: str | TelemetryEventType,
        pos: Tuple[float, float] = (0.0, 0.0),
        vel: Tuple[float, float] = (0.0, 0.0),
        health: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
        probe_name: Optional[str] = None,
    ) -> None:
        ev_type_str = event_type.value if isinstance(event_type, TelemetryEventType) else str(event_type)

        if ev_type_str == TelemetryEventType.DASH:
            self._stats["dashes"] += 1
        elif ev_type_str == TelemetryEventType.WEAPON_FIRE:
            self._stats["shots_fired"] += 1
        elif ev_type_str == TelemetryEventType.DAMAGE_TAKEN and payload:
            self._stats["damage_taken"] += float(payload.get("damage", 0.0))
        elif ev_type_str == TelemetryEventType.DAMAGE_DEALT and payload:
            self._stats["damage_dealt"] += float(payload.get("damage", 0.0))

        record = TelemetryEventRecord(
            session_id=self.session_id,
            episode_id=self.current_episode_id or f"ep_{self.session_id}_001",
            timestamp=time.time(),
            event_type=ev_type_str,
            pos_x=pos[0],
            pos_y=pos[1],
            vel_x=vel[0],
            vel_y=vel[1],
            health=health,
            payload=payload or {},
            frame_index=self.frame_index,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            deliberation="Captured in-memory event",
        )
        p_name = probe_name or (payload.get("probe_name") if payload else None)
        if p_name:
            setattr(record, "probe_name", str(p_name))
        self.events.append(record)

    def tick_frame(self) -> None:
        self.frame_index += 1

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        self.events.clear()
        self.sessions.clear()
        self.episodes.clear()
        self.frame_index = 0
        self._stats = {"dashes": 0, "shots_fired": 0, "damage_dealt": 0.0, "damage_taken": 0.0}

    def get_events_by_type(self, event_type: str | TelemetryEventType) -> List[TelemetryEventRecord]:
        ev_type_str = event_type.value if isinstance(event_type, TelemetryEventType) else str(event_type)
        return [e for e in self.events if e.event_type == ev_type_str]

    def get_events(self, event_type: str | TelemetryEventType) -> List[TelemetryEventRecord]:
        return self.get_events_by_type(event_type)


class SimulationInput:
    """Synthetic input packet applied per discrete simulation tick."""
    deliberation: str = "Synthetic input packet applied per discrete simulation tick."

    def __init__(
        self,
        deliberation: str = "Synthetic input packet applied per discrete simulation tick.",
        move_x: float = 0.0,
        move_y: float = 0.0,
        dash: bool = False,
        fire: bool = False,
        aim_x: float = 0.0,
        aim_y: float = 0.0,
        **kwargs: Any,
    ):
        self.deliberation = deliberation
        if kwargs.get("right"):
            move_x = 1.0
        elif kwargs.get("left"):
            move_x = -1.0
        if kwargs.get("down"):
            move_y = 1.0
        elif kwargs.get("up"):
            move_y = -1.0
        if kwargs.get("action_1"):
            fire = True
        if kwargs.get("space"):
            dash = True

        self.move_x = float(move_x)
        self.move_y = float(move_y)
        self.dash = bool(dash)
        self.fire = bool(fire)
        self.aim_x = float(aim_x)
        self.aim_y = float(aim_y)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "deliberation": self.deliberation,
            "move_x": self.move_x,
            "move_y": self.move_y,
            "dash": self.dash,
            "fire": self.fire,
            "aim_x": self.aim_x,
            "aim_y": self.aim_y,
        }
        for k, v in self.__dict__.items():
            if k not in d:
                d[k] = v
        return d


class SimulationStepResult:
    """Outcome payload generated after executing one simulation step."""
    deliberation: str = "Outcome payload generated after executing one simulation step."

    def __init__(
        self,
        deliberation: str = "Outcome payload generated after executing one simulation step.",
        step: int = 0,
        elapsed_time: float = 0.0,
        player_pos: Tuple[float, float] = (0.0, 0.0),
        player_vel: Tuple[float, float] = (0.0, 0.0),
        player_health: float = 100.0,
        player_stamina: float = 100.0,
        is_dashing: bool = False,
        active_enemies: int = 0,
        active_projectiles: int = 0,
        is_completed: bool = False,
        outcome: Optional[str] = None,
    ):
        self.deliberation = deliberation
        self.step = step
        self.elapsed_time = elapsed_time
        self.player_pos = player_pos
        self.player_vel = player_vel
        self.player_health = player_health
        self.player_stamina = player_stamina
        self.is_dashing = is_dashing
        self.active_enemies = active_enemies
        self.active_projectiles = active_projectiles
        self.is_completed = is_completed
        self.outcome = outcome


class SimulationState:
    """Serializable snapshot of the headless simulation."""
    deliberation: str = "Serializable snapshot of the headless simulation."

    def __init__(
        self,
        deliberation: str = "Serializable snapshot of the headless simulation.",
        step_count: int = 0,
        elapsed_time: float = 0.0,
        player_pos: Tuple[float, float] = (0.0, 0.0),
        player_vel: Tuple[float, float] = (0.0, 0.0),
        player_health: float = 100.0,
        player_stamina: float = 100.0,
        is_dashing: bool = False,
        enemies: Optional[List[Dict[str, Any]]] = None,
        active_projectiles: int = 0,
        hazards_count: int = 0,
        portal_unlocked: bool = False,
        portal_pos: Tuple[float, float] = (0.0, 0.0),
        is_completed: bool = False,
        outcome: Optional[str] = None,
    ):
        self.deliberation = deliberation
        self.step_count = step_count
        self.elapsed_time = elapsed_time
        self.player_pos = player_pos
        self.player_vel = player_vel
        self.player_health = player_health
        self.player_stamina = player_stamina
        self.is_dashing = is_dashing
        self.enemies = enemies or []
        self.active_projectiles = active_projectiles
        self.hazards_count = hazards_count
        self.portal_unlocked = portal_unlocked
        self.portal_pos = portal_pos
        self.is_completed = is_completed
        self.outcome = outcome


class SimulationSummary:
    """Comprehensive post-run evaluation metrics."""
    deliberation: str = "Comprehensive post-run evaluation metrics."

    def __init__(
        self,
        deliberation: str = "Comprehensive post-run evaluation metrics.",
        total_steps: int = 0,
        duration_seconds: float = 0.0,
        outcome: str = "IN_PROGRESS",
        cleared: bool = False,
        damage_dealt: float = 0.0,
        damage_taken: float = 0.0,
        dashes_used: int = 0,
        shots_fired: int = 0,
        enemies_killed: int = 0,
        remaining_enemies: int = 0,
        final_player_health: float = 0.0,
        final_player_pos: Tuple[float, float] = (0.0, 0.0),
        exit_pos: Tuple[float, float] = (0.0, 0.0),
        telemetry_event_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.deliberation = deliberation
        self.total_steps = total_steps
        self.duration_seconds = duration_seconds
        self.outcome = outcome
        self.cleared = cleared
        self.damage_dealt = damage_dealt
        self.damage_taken = damage_taken
        self.dashes_used = dashes_used
        self.shots_fired = shots_fired
        self.enemies_killed = enemies_killed
        self.remaining_enemies = remaining_enemies
        self.final_player_health = final_player_health
        self.final_player_pos = final_player_pos
        self.exit_pos = exit_pos
        self.telemetry_event_count = telemetry_event_count
        self.metadata = metadata or {}


def __getattr__(name: str) -> Any:
    if name == "HeadlessSimulator":
        from engine.sim.simulator import HeadlessSimulator
        return HeadlessSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HeadlessSimulator",
    "InMemoryTelemetryRecorder",
    "SimulationInput",
    "SimulationStepResult",
    "SimulationState",
    "SimulationSummary",
]
