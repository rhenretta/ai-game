"""engine/telemetry/models.py
Telemetry event models, records, and schema envelopes.
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
import json
import math
import time
from typing import Any, Dict, Optional, Tuple


def _safe_float(val: Any, default: float = 0.0, allow_none: bool = False, allow_inf: bool = True) -> Optional[float]:
    """Safely coerces val to float, converting NaN or un-coercible values to default/None."""
    if val is None:
        return None if allow_none else default
    try:
        f = float(val)
        if math.isnan(f):
            return None if allow_none else default
        if not allow_inf and math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return None if allow_none else default


def _safe_round(val: Optional[float], default: float = 0.0, allow_none: bool = False, allow_inf: bool = True, ndigits: int = 3) -> Optional[float]:
    """Safely rounds float values, preserving IEEE infinity for SQLite, and coercing NaN/None."""
    f = _safe_float(val, default=default, allow_none=allow_none, allow_inf=allow_inf)
    if f is None:
        return None
    if math.isinf(f):
        return f
    return round(f, ndigits)


def _safe_json_dumps(payload: Any, default_fallback: str = "{}") -> str:
    """Robust JSON serializer that handles arbitrary types, functions, and non-serializable objects."""
    if not payload:
        return default_fallback
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, default=str, separators=(',', ':'))
    except Exception as ser_err:
        try:
            fallback = {
                "_serialization_error": type(ser_err).__name__,
                "_error_msg": str(ser_err),
                "_safe_repr": repr(payload)[:500] if hasattr(payload, "__repr__") else "<unrepresentable>",
            }
            return json.dumps(fallback, default=str, separators=(',', ':'))
        except Exception:
            return '{"_serialization_error":"unserializable_payload"}'


class TelemetryEventType(str, Enum):
    """Authoritative event types for telemetry stream."""
    PLAYER_SPAWN = "PLAYER_SPAWN"
    MOVEMENT_TICK = "MOVEMENT_TICK"
    DASH = "DASH"
    WEAPON_FIRE = "WEAPON_FIRE"
    DAMAGE_TAKEN = "DAMAGE_TAKEN"
    DAMAGE_DEALT = "DAMAGE_DEALT"
    HAZARD_HIT = "HAZARD_HIT"
    PLAYER_DEATH = "PLAYER_DEATH"
    LEVEL_EXIT_REACHED = "LEVEL_EXIT_REACHED"
    EPISODE_START = "EPISODE_START"
    EPISODE_END = "EPISODE_END"
    CUSTOM_PROBE = "CUSTOM_PROBE"


class ProbeCatalogRecord:
    """Metadata catalog entry for an agent-defined dynamic probe."""
    deliberation: str = ""

    def __init__(
        self,
        probe_id: str,
        probe_name: str,
        category: str,
        description: str,
        hypothesis_target: str,
        schema_json: Optional[Dict[str, Any] | str] = None,
        created_at: Optional[str] = None,
        deliberation: str = ""
    ):
        self.deliberation = deliberation
        self.probe_id = str(probe_id or "")
        self.probe_name = str(probe_name or "")
        self.category = str(category or "general")
        self.description = str(description or "")
        self.hypothesis_target = str(hypothesis_target or "")
        if isinstance(schema_json, dict):
            self.schema_json = _safe_json_dumps(schema_json)
        elif isinstance(schema_json, str):
            self.schema_json = schema_json
        else:
            self.schema_json = "{}"
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    def to_db_tuple(self) -> Tuple[Any, ...]:
        return (
            self.probe_id,
            self.probe_name,
            self.category,
            self.description,
            self.hypothesis_target,
            self.schema_json,
            self.created_at,
        )


class TelemetryEventRecord:
    """Represents a single atomic telemetry event matching SQLite schema."""
    deliberation: str = ""

    def __init__(
        self,
        session_id: str,
        episode_id: str,
        timestamp: float,
        event_type: str,
        pos_x: float,
        pos_y: float,
        vel_x: float = 0.0,
        vel_y: float = 0.0,
        health: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
        frame_index: int = 0,
        timestamp_utc: str = "",
        probe_name: Optional[str] = None,
        deliberation: str = ""
    ):
        self.deliberation = deliberation
        self.session_id = str(session_id)
        self.episode_id = str(episode_id)
        self.timestamp = float(timestamp)
        self.event_type = str(event_type)
        self.probe_name = str(probe_name) if probe_name is not None else None
        self.pos_x = _safe_float(pos_x, default=0.0, allow_none=False)
        self.pos_y = _safe_float(pos_y, default=0.0, allow_none=False)
        self.vel_x = _safe_float(vel_x, default=0.0, allow_none=False)
        self.vel_y = _safe_float(vel_y, default=0.0, allow_none=False)
        self.health = _safe_float(health, default=0.0, allow_none=True)
        self.payload = payload or {}
        self.frame_index = int(frame_index)
        self.timestamp_utc = str(timestamp_utc)

    def to_db_tuple(self) -> Tuple[Any, ...]:
        payload_str = _safe_json_dumps(self.payload)
        safe_pos_x = _safe_round(self.pos_x, default=0.0, allow_none=False)
        safe_pos_y = _safe_round(self.pos_y, default=0.0, allow_none=False)
        safe_vel_x = _safe_round(self.vel_x, default=0.0, allow_none=False)
        safe_vel_y = _safe_round(self.vel_y, default=0.0, allow_none=False)
        safe_health = _safe_round(self.health, default=0.0, allow_none=True)
        return (
            self.session_id,
            self.episode_id,
            self.timestamp,
            self.event_type,
            self.probe_name,
            safe_pos_x,
            safe_pos_y,
            safe_vel_x,
            safe_vel_y,
            safe_health,
            payload_str,
            self.frame_index,
            self.timestamp_utc
        )


class SessionStartRecord:
    deliberation: str = ""

    def __init__(
        self,
        session_id: str,
        started_at: str,
        player_profile_tag: str = "default",
        platform: str = "macOS",
        resolution: str = "1280x720",
        target_fps: int = 60,
        metadata: Optional[Dict[str, Any]] = None,
        deliberation: str = ""
    ):
        self.deliberation = deliberation
        self.session_id = session_id
        self.started_at = started_at
        self.player_profile_tag = player_profile_tag
        self.platform = platform
        self.resolution = resolution
        self.target_fps = target_fps
        self.metadata = metadata or {}

    def to_db_tuple(self) -> Tuple[Any, ...]:
        return (
            self.session_id,
            self.started_at,
            self.started_at,
            None,
            self.player_profile_tag,
            self.platform,
            self.resolution,
            self.target_fps,
            0,
            _safe_json_dumps(self.metadata)
        )


class SessionEndRecord:
    deliberation: str = ""

    def __init__(
        self,
        session_id: str,
        ended_at: str,
        total_episodes: int,
        metadata: Optional[Dict[str, Any]] = None,
        deliberation: str = ""
    ):
        self.deliberation = deliberation
        self.session_id = session_id
        self.ended_at = ended_at
        self.total_episodes = total_episodes
        self.metadata = metadata or {}


class EpisodeStartRecord:
    deliberation: str = ""

    def __init__(
        self,
        episode_id: str,
        session_id: str,
        episode_number: int,
        status: str = "ACTIVE",
        started_at: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        deliberation: str = ""
    ):
        self.deliberation = deliberation
        self.episode_id = episode_id
        self.session_id = session_id
        self.episode_number = int(episode_number)
        self.status = status
        self.started_at = started_at
        self.metadata = metadata or {}

    def to_db_tuple(self) -> Tuple[Any, ...]:
        return (
            self.episode_id,
            self.session_id,
            self.episode_number,
            self.status,
            self.started_at,
            None,
            0.0,
            "IN_PROGRESS",
            "hero",
            0,
            0.0,
            0.0,
            0,
            0,
            None,
            None,
            _safe_json_dumps(self.metadata)
        )


class EpisodeEndRecord:
    deliberation: str = ""

    def __init__(
        self,
        episode_id: str,
        ended_at: str,
        duration_seconds: float,
        outcome: str = "CLEARED",
        score: int = 0,
        damage_dealt_total: float = 0.0,
        damage_taken_total: float = 0.0,
        dashes_total: int = 0,
        shots_fired_total: int = 0,
        exit_pos: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        deliberation: str = ""
    ):
        self.deliberation = deliberation
        self.episode_id = episode_id
        self.ended_at = ended_at
        self.duration_seconds = float(duration_seconds)
        self.outcome = outcome
        self.score = score
        self.damage_dealt_total = float(damage_dealt_total)
        self.damage_taken_total = float(damage_taken_total)
        self.dashes_total = int(dashes_total)
        self.shots_fired_total = int(shots_fired_total)
        self.exit_pos = exit_pos
        self.metadata = metadata or {}
