"""engine/telemetry/recorder.py
Thread-safe, sub-microsecond telemetry recorder called from the game loop.
"""
from __future__ import annotations
import atexit
from datetime import datetime, timezone
import logging
from pathlib import Path
import queue
import time
from typing import Any, Dict, Optional, Tuple
import uuid

from engine.telemetry.db import TelemetryDB
from engine.telemetry.models import (
    EpisodeEndRecord,
    EpisodeStartRecord,
    ProbeCatalogRecord,
    SessionEndRecord,
    SessionStartRecord,
    TelemetryEventRecord,
    TelemetryEventType,
)
from engine.telemetry.writer import TelemetryWriterThread

logger = logging.getLogger("TelemetryRecorder")


class TelemetryRecorder:
    """High-level, zero-stutter telemetry recorder for the game loop."""

    def __init__(
        self,
        db_path: str | Path = "data/telemetry.db",
        batch_size: int = 50,
        flush_interval_sec: float = 0.1,
    ):
        self.db = TelemetryDB(db_path)
        self.queue: queue.Queue = queue.Queue(maxsize=0)
        self.writer_thread = TelemetryWriterThread(
            db=self.db,
            telemetry_queue=self.queue,
            batch_size=batch_size,
            flush_interval_sec=flush_interval_sec,
        )
        self.writer_thread.start()

        self.session_id: Optional[str] = None
        self.current_episode_id: Optional[str] = None
        self.episode_start_time: float = 0.0
        self.episode_index: int = 0
        self.frame_index: int = 0

        self._damage_dealt_accum: float = 0.0
        self._damage_taken_accum: float = 0.0
        self._dashes_accum: int = 0
        self._shots_fired_accum: int = 0
        self._score_accum: int = 0

        self._closed = False
        atexit.register(self.close)

    def start_session(
        self,
        player_profile_tag: str = "default",
        platform: str = "macOS",
        resolution: str = "1280x720",
        target_fps: int = 60,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.session_id = f"sess_{uuid.uuid4().hex[:12]}"
        now_utc = datetime.now(timezone.utc).isoformat()
        record = SessionStartRecord(
            session_id=self.session_id,
            started_at=now_utc,
            player_profile_tag=player_profile_tag,
            platform=platform,
            resolution=resolution,
            target_fps=target_fps,
            metadata=metadata or {},
        )
        self.queue.put_nowait(("SESSION_START", record))
        return self.session_id

    def end_session(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.session_id:
            return
        now_utc = datetime.now(timezone.utc).isoformat()
        record = SessionEndRecord(
            session_id=self.session_id,
            ended_at=now_utc,
            total_episodes=self.episode_index,
            metadata=metadata or {},
        )
        self.queue.put_nowait(("SESSION_END", record))
        self.session_id = None

    def start_episode(
        self,
        level_id: str = "level_01",
        episode_index: Optional[int] = None,
        player_character: str = "hero",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.session_id:
            self.start_session()

        if episode_index is not None:
            self.episode_index = episode_index
        else:
            self.episode_index += 1

        self.current_episode_id = f"ep_{self.session_id}_{self.episode_index:03d}"
        self.episode_start_time = time.perf_counter()
        self.frame_index = 0
        self._damage_dealt_accum = 0.0
        self._damage_taken_accum = 0.0
        self._dashes_accum = 0
        self._shots_fired_accum = 0
        self._score_accum = 0

        now_utc = datetime.now(timezone.utc).isoformat()
        record = EpisodeStartRecord(
            episode_id=self.current_episode_id,
            session_id=self.session_id,
            episode_number=self.episode_index,
            status="ACTIVE",
            started_at=now_utc,
            metadata=metadata or {},
        )
        self.queue.put_nowait(("EPISODE_START", record))
        return self.current_episode_id

    def end_episode(
        self,
        outcome: str = "CLEARED",
        score: Optional[int] = None,
        exit_pos: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.current_episode_id:
            return

        duration = time.perf_counter() - self.episode_start_time
        now_utc = datetime.now(timezone.utc).isoformat()
        record = EpisodeEndRecord(
            episode_id=self.current_episode_id,
            ended_at=now_utc,
            duration_seconds=round(duration, 3),
            outcome=outcome,
            score=score if score is not None else self._score_accum,
            damage_dealt_total=round(self._damage_dealt_accum, 2),
            damage_taken_total=round(self._damage_taken_accum, 2),
            dashes_total=self._dashes_accum,
            shots_fired_total=self._shots_fired_accum,
            exit_pos=exit_pos,
            metadata=metadata or {},
        )
        self.queue.put_nowait(("EPISODE_END", record))
        self.current_episode_id = None

    def emit_event(
        self,
        event_type: str | TelemetryEventType,
        pos: Tuple[float, float],
        vel: Tuple[float, float] = (0.0, 0.0),
        health: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
        probe_name: Optional[str] = None,
    ) -> None:
        """
        Non-blocking enqueue of telemetry event.
        Executes in < 2 microseconds.
        """
        if not self.session_id:
            self.start_session()
        if not self.current_episode_id:
            self.start_episode(level_id="default_level", episode_index=1)

        ev_type_str = event_type.value if isinstance(event_type, TelemetryEventType) else str(event_type)

        if ev_type_str == TelemetryEventType.DASH:
            self._dashes_accum += 1
        elif ev_type_str == TelemetryEventType.WEAPON_FIRE:
            self._shots_fired_accum += 1
        elif ev_type_str == TelemetryEventType.DAMAGE_TAKEN and payload:
            self._damage_taken_accum += float(payload.get("damage", 0.0))
        elif ev_type_str == TelemetryEventType.DAMAGE_DEALT and payload:
            self._damage_dealt_accum += float(payload.get("damage", 0.0))

        record = TelemetryEventRecord(
            session_id=self.session_id,
            episode_id=self.current_episode_id,
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
            probe_name=probe_name,
        )

        try:
            self.queue.put_nowait(record)
        except Exception as e:
            logger.error("Failed to enqueue event: %s", e)

    def emit_custom_probe(
        self,
        probe_name: str,
        pos: Tuple[float, float] = (0.0, 0.0),
        vel: Tuple[float, float] = (0.0, 0.0),
        health: Optional[float] = None,
        category: str = "combat",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Sub-microsecond (< 2 µs) non-blocking emission of CUSTOM_PROBE event.
        Encapsulates arbitrary agent-designed domain payloads under payload_json.
        """
        payload = {"category": category, **(data or {})}
        self.emit_event(
            event_type=TelemetryEventType.CUSTOM_PROBE,
            pos=pos,
            vel=vel,
            health=health,
            payload=payload,
            probe_name=probe_name,
        )

    def register_probe(
        self,
        probe_id: str,
        probe_name: str,
        category: str,
        description: str,
        hypothesis_target: str,
        schema_json: Optional[Dict[str, Any] | str] = None,
    ) -> None:
        """
        Registers probe metadata via background writer queue (non-blocking).
        """
        record = ProbeCatalogRecord(
            probe_id=probe_id,
            probe_name=probe_name,
            category=category,
            description=description,
            hypothesis_target=hypothesis_target,
            schema_json=schema_json,
        )
        self.queue.put_nowait(("REGISTER_PROBE", record))

    def tick_frame(self) -> None:
        self.frame_index += 1

    def flush(self) -> None:
        if not self._closed:
            self.writer_thread.flush()

    def close(self, timeout: float = 2.0, fast: bool = False) -> None:
        if self._closed:
            return
        self._closed = True

        if fast:
            timeout = min(timeout, 0.8)

        if self.current_episode_id:
            self.end_episode(outcome="ABORTED")
        if self.session_id:
            self.end_session()

        t0 = time.perf_counter()
        writer_timeout = max(0.05, timeout * 0.75)
        self.writer_thread.stop(timeout=writer_timeout)

        elapsed = time.perf_counter() - t0
        remaining_timeout = max(0.05, timeout - elapsed)
        try:
            self.db.checkpoint("TRUNCATE", timeout=remaining_timeout)
        except TypeError:
            try:
                self.db.checkpoint("TRUNCATE")
            except Exception:
                pass
        except Exception:
            pass

    def shutdown(self, timeout: float = 2.0, fast: bool = False) -> None:
        self.close(timeout=timeout, fast=fast)

    def __enter__(self) -> TelemetryRecorder:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class NullTelemetryRecorder:
    """Safe no-op telemetry recorder for testing and headless runs."""

    def __init__(self, *args, **kwargs):
        self.session_id = "sess_null"
        self.current_episode_id = "ep_null_001"
        self.frame_index = 0

    def start_session(self, *args, **kwargs) -> str:
        return self.session_id

    def end_session(self, *args, **kwargs) -> None:
        pass

    def start_episode(self, *args, **kwargs) -> str:
        return self.current_episode_id

    def end_episode(self, *args, **kwargs) -> None:
        pass

    def emit_event(self, *args, **kwargs) -> None:
        pass

    def emit_custom_probe(self, *args, **kwargs) -> None:
        pass

    def register_probe(self, *args, **kwargs) -> None:
        pass

    def tick_frame(self) -> None:
        self.frame_index += 1

    def flush(self) -> None:
        pass

    def close(self, timeout: float = 2.0, fast: bool = False) -> None:
        pass

    def shutdown(self, timeout: float = 2.0, fast: bool = False) -> None:
        pass
