"""engine/telemetry/writer.py
Background daemon thread for batched, non-blocking SQLite WAL telemetry writes.
"""
from __future__ import annotations
from datetime import datetime, timezone
import logging
import queue
import sqlite3
import threading
import time
from typing import Any, List, Optional, Tuple, Union
from engine.telemetry.db import TelemetryDB
from engine.telemetry.models import (
    EpisodeEndRecord,
    EpisodeStartRecord,
    ProbeCatalogRecord,
    SessionEndRecord,
    SessionStartRecord,
    TelemetryEventRecord,
    _safe_json_dumps,
)

logger = logging.getLogger("TelemetryWriterThread")


class TelemetryWriterThread(threading.Thread):
    """
    Dedicated daemon thread draining the in-memory telemetry queue in batched transactions.
    Flushes when batch size reaches 50 events OR after 100ms has elapsed.
    """

    def __init__(
        self,
        db: TelemetryDB,
        telemetry_queue: queue.Queue,
        batch_size: int = 50,
        flush_interval_sec: float = 0.1,
        auto_register_probes: bool = False,
    ):
        super().__init__(name="TelemetryWriterThread", daemon=True)
        self.db = db
        self.queue = telemetry_queue
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        self.auto_register_probes = auto_register_probes

        self._stop_event = threading.Event()
        self._flush_event = threading.Event()
        self.discarded_row_count: int = 0
        self.discarded_errors: List[str] = []
        self._conn: Optional[sqlite3.Connection] = None
        self._event_insert_sql = """
            INSERT INTO telemetry_events (
                session_id, episode_id, timestamp, event_type, probe_name, pos_x, pos_y,
                vel_x, vel_y, health, payload_json, frame_index, timestamp_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._custom_event_insert_sql = """
            INSERT INTO custom_telemetry_events (
                session_id, episode_id, timestamp, probe_name, category, payload_json, frame_index, timestamp_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

    def run(self) -> None:
        self._conn = self.db.get_writer_connection()
        batch_events: List[Tuple[Any, ...]] = []
        custom_batch_events: List[Tuple[Any, ...]] = []
        last_flush_time = time.perf_counter()

        try:
            while not self._stop_event.is_set():
                timeout = min(0.05, max(0.005, self.flush_interval_sec - (time.perf_counter() - last_flush_time)))
                try:
                    item = self.queue.get(timeout=timeout)
                except queue.Empty:
                    item = None

                if item is not None:
                    try:
                        if isinstance(item, tuple) and item[0] == "FLUSH":
                            if batch_events or custom_batch_events:
                                self._commit_batch(batch_events, custom_batch_events)
                                batch_events.clear()
                                custom_batch_events.clear()
                                last_flush_time = time.perf_counter()
                            item[1].set()
                            continue
                        elif isinstance(item, tuple) and item[0] == "STOP":
                            # Fast-drain and commit all remaining queue items on STOP
                            self._drain_and_commit(batch_events, custom_batch_events)
                            break
                        elif isinstance(item, tuple) and item[0] == "REGISTER_PROBE":
                            self._process_register_probe_command(item[1])
                            if batch_events or custom_batch_events:
                                self._commit_batch(batch_events, custom_batch_events)
                                batch_events.clear()
                                custom_batch_events.clear()
                                last_flush_time = time.perf_counter()
                        elif isinstance(item, tuple) and item[0] in ("SESSION_START", "SESSION_END", "EPISODE_START", "EPISODE_END"):
                            if batch_events or custom_batch_events:
                                self._commit_batch(batch_events, custom_batch_events)
                                batch_events.clear()
                                custom_batch_events.clear()
                                last_flush_time = time.perf_counter()
                            self._process_lifecycle_command(item)
                        elif isinstance(item, TelemetryEventRecord):
                            try:
                                batch_events.append(item.to_db_tuple())
                                if item.event_type == "CUSTOM_PROBE" or item.probe_name:
                                    cat = "combat"
                                    if isinstance(item.payload, dict):
                                        cat = str(item.payload.get("category", "combat") or "combat")
                                    p_str = _safe_json_dumps(item.payload)
                                    custom_batch_events.append((
                                        item.session_id,
                                        item.episode_id,
                                        item.timestamp,
                                        item.probe_name or "custom_probe",
                                        cat,
                                        p_str,
                                        item.frame_index,
                                        item.timestamp_utc,
                                    ))
                            except Exception as item_err:
                                logger.error("Failed to serialize TelemetryEventRecord: %s", item_err)
                    except Exception as loop_err:
                        logger.error("Failed to process queue item %s: %s", type(item), loop_err)
                    finally:
                        self.queue.task_done()

                now = time.perf_counter()
                total_pending = len(batch_events) + len(custom_batch_events)
                should_flush = (
                    total_pending >= self.batch_size
                    or (total_pending > 0 and (now - last_flush_time) >= self.flush_interval_sec)
                    or self._flush_event.is_set()
                )

                if should_flush and (batch_events or custom_batch_events):
                    self._commit_batch(batch_events, custom_batch_events)
                    batch_events.clear()
                    custom_batch_events.clear()
                    last_flush_time = now
                    self._flush_event.clear()

            # Drain and commit remaining queue items if _stop_event was set externally
            self._drain_and_commit(batch_events, custom_batch_events)

        except Exception as e:
            logger.error("Error in TelemetryWriterThread: %s", e)
        finally:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def _commit_batch(self, batch: List[Tuple[Any, ...]], custom_batch: Optional[List[Tuple[Any, ...]]] = None) -> None:
        if not self._conn or (not batch and not custom_batch):
            return
        try:
            self._conn.execute("BEGIN TRANSACTION;")
            if custom_batch:
                # Ensure all probe_names exist in telemetry_probes to satisfy foreign key constraint if auto_register enabled
                if self.auto_register_probes:
                    probes_to_ensure = {}
                    for crow in custom_batch:
                        p_name = crow[3]
                        p_cat = crow[4]
                        p_created = crow[7] if len(crow) > 7 and crow[7] else datetime.now(timezone.utc).isoformat()
                        if p_name and p_name not in probes_to_ensure:
                            probes_to_ensure[p_name] = (
                                f"prb_{p_name}",
                                p_name,
                                p_cat or "gameplay",
                                f"Auto-registered probe {p_name}",
                                "H_AUTO_REGISTERED",
                                p_created,
                            )
                    if probes_to_ensure:
                        self._conn.executemany(
                            """
                            INSERT OR IGNORE INTO telemetry_probes (
                                probe_id, probe_name, category, description, hypothesis_target, schema_json, created_at
                            ) VALUES (?, ?, ?, ?, ?, '{}', ?)
                            """,
                            list(probes_to_ensure.values()),
                        )
            if batch:
                self._conn.executemany(self._event_insert_sql, batch)
            if custom_batch:
                self._conn.executemany(self._custom_event_insert_sql, custom_batch)
            self._conn.execute("COMMIT;")
        except Exception as err:
            logger.error("Failed to commit telemetry batch: %s. Attempting row-by-row rescue.", err)
            try:
                self._conn.execute("ROLLBACK;")
            except Exception:
                pass
            # Row-by-row rescue fallback
            try:
                self._conn.execute("BEGIN TRANSACTION;")
                if batch:
                    for row in batch:
                        try:
                            self._conn.execute(self._event_insert_sql, row)
                        except Exception as row_err:
                            self.discarded_row_count += 1
                            self.discarded_errors.append(f"telemetry_events: {row_err}")
                            logger.error("Discarding single corrupted telemetry row: %s (error: %s)", row, row_err)
                if custom_batch:
                    for crow in custom_batch:
                        try:
                            if self.auto_register_probes:
                                p_name = crow[3]
                                p_cat = crow[4]
                                p_created = crow[7] if len(crow) > 7 and crow[7] else datetime.now(timezone.utc).isoformat()
                                self._conn.execute(
                                    """
                                    INSERT OR IGNORE INTO telemetry_probes (
                                        probe_id, probe_name, category, description, hypothesis_target, schema_json, created_at
                                    ) VALUES (?, ?, ?, ?, ?, '{}', ?)
                                    """,
                                    (
                                        f"prb_{p_name}",
                                        p_name,
                                        p_cat or "gameplay",
                                        f"Auto-registered probe {p_name}",
                                        "H_AUTO_REGISTERED",
                                        p_created,
                                    ),
                                )
                            self._conn.execute(self._custom_event_insert_sql, crow)
                        except Exception as crow_err:
                            self.discarded_row_count += 1
                            self.discarded_errors.append(f"custom_telemetry_events: {crow_err}")
                            logger.error("Discarding single corrupted custom telemetry row: %s (error: %s)", crow, crow_err)
                self._conn.execute("COMMIT;")
            except Exception as rescue_err:
                logger.error("Failed rescue transaction: %s", rescue_err)
                try:
                    self._conn.execute("ROLLBACK;")
                except Exception:
                    pass

    def _process_lifecycle_command(self, cmd_tuple: Tuple[str, Any]) -> None:
        cmd, record = cmd_tuple
        if not self._conn:
            return
        try:
            self._conn.execute("BEGIN TRANSACTION;")
            if cmd == "SESSION_START" and isinstance(record, SessionStartRecord):
                self._conn.execute(
                    """
                    INSERT INTO telemetry_sessions (
                        session_id, created_at, started_at, ended_at, player_profile_tag,
                        platform, resolution, target_fps, total_episodes, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET started_at=excluded.started_at
                    """,
                    record.to_db_tuple(),
                )
            elif cmd == "SESSION_END" and isinstance(record, SessionEndRecord):
                self._conn.execute(
                    """
                    UPDATE telemetry_sessions
                    SET ended_at = ?, total_episodes = ?
                    WHERE session_id = ?
                    """,
                    (record.ended_at, record.total_episodes, record.session_id),
                )
            elif cmd == "EPISODE_START" and isinstance(record, EpisodeStartRecord):
                self._conn.execute(
                    """
                    INSERT INTO telemetry_episodes (
                        episode_id, session_id, episode_number, status, started_at, ended_at,
                        duration_seconds, outcome, player_character, score, damage_dealt_total,
                        damage_taken_total, dashes_total, shots_fired_total, exit_x, exit_y, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(episode_id) DO NOTHING
                    """,
                    record.to_db_tuple(),
                )
            elif cmd == "EPISODE_END" and isinstance(record, EpisodeEndRecord):
                exit_x = record.exit_pos[0] if record.exit_pos else None
                exit_y = record.exit_pos[1] if record.exit_pos else None
                self._conn.execute(
                    """
                    UPDATE telemetry_episodes
                    SET ended_at = ?, duration_seconds = ?, outcome = ?, score = ?,
                        damage_dealt_total = ?, damage_taken_total = ?, dashes_total = ?,
                        shots_fired_total = ?, exit_x = ?, exit_y = ?
                    WHERE episode_id = ?
                    """,
                    (
                        record.ended_at,
                        record.duration_seconds,
                        record.outcome,
                        record.score,
                        record.damage_dealt_total,
                        record.damage_taken_total,
                        record.dashes_total,
                        record.shots_fired_total,
                        exit_x,
                        exit_y,
                        record.episode_id,
                    ),
                )
            self._conn.execute("COMMIT;")
        except Exception as err:
            logger.error("Failed to execute lifecycle command %s: %s", cmd, err)
            try:
                self._conn.execute("ROLLBACK;")
            except Exception:
                pass

    def _process_register_probe_command(self, record: ProbeCatalogRecord) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute("BEGIN TRANSACTION;")
            self._conn.execute(
                """
                INSERT INTO telemetry_probes (
                    probe_id, probe_name, category, description, hypothesis_target, schema_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(probe_id) DO UPDATE SET
                    probe_name=excluded.probe_name,
                    category=excluded.category,
                    description=excluded.description,
                    hypothesis_target=excluded.hypothesis_target,
                    schema_json=excluded.schema_json
                ON CONFLICT(probe_name) DO UPDATE SET
                    category=excluded.category,
                    description=excluded.description,
                    hypothesis_target=excluded.hypothesis_target,
                    schema_json=excluded.schema_json
                """,
                record.to_db_tuple(),
            )
            self._conn.execute("COMMIT;")
        except Exception as err:
            logger.error("Failed to execute probe registration command: %s", err)
            try:
                self._conn.execute("ROLLBACK;")
            except Exception:
                pass

    def flush(self, timeout: float = 2.0) -> None:
        done_event = threading.Event()
        self.queue.put_nowait(("FLUSH", done_event))
        done_event.wait(timeout=timeout)

    def _drain_and_commit(
        self,
        batch_events: List[Tuple[Any, ...]],
        custom_batch_events: Optional[List[Tuple[Any, ...]]] = None,
    ) -> None:
        """Fast-drain all remaining queue items and commit in bulk on shutdown."""
        if custom_batch_events is None:
            custom_batch_events = []
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                if isinstance(item, tuple):
                    cmd = item[0]
                    if cmd == "FLUSH":
                        if batch_events or custom_batch_events:
                            self._commit_batch(batch_events, custom_batch_events)
                            batch_events.clear()
                            custom_batch_events.clear()
                        item[1].set()
                    elif cmd == "STOP":
                        pass
                    elif cmd == "REGISTER_PROBE":
                        self._process_register_probe_command(item[1])
                        if batch_events or custom_batch_events:
                            self._commit_batch(batch_events, custom_batch_events)
                            batch_events.clear()
                            custom_batch_events.clear()
                    elif cmd in ("SESSION_START", "SESSION_END", "EPISODE_START", "EPISODE_END"):
                        if batch_events or custom_batch_events:
                            self._commit_batch(batch_events, custom_batch_events)
                            batch_events.clear()
                            custom_batch_events.clear()
                        self._process_lifecycle_command(item)
                elif isinstance(item, TelemetryEventRecord):
                    try:
                        batch_events.append(item.to_db_tuple())
                        if item.event_type == "CUSTOM_PROBE" or item.probe_name:
                            cat = "combat"
                            if isinstance(item.payload, dict):
                                cat = str(item.payload.get("category", "combat") or "combat")
                            p_str = _safe_json_dumps(item.payload)
                            custom_batch_events.append((
                                item.session_id,
                                item.episode_id,
                                item.timestamp,
                                item.probe_name or "custom_probe",
                                cat,
                                p_str,
                                item.frame_index,
                                item.timestamp_utc,
                            ))
                    except Exception as item_err:
                        logger.error("Failed to serialize TelemetryEventRecord: %s", item_err)
            except Exception as loop_err:
                logger.error("Failed to process queue item %s: %s", type(item), loop_err)
            finally:
                self.queue.task_done()

        if batch_events or custom_batch_events:
            self._commit_batch(batch_events, custom_batch_events)
            batch_events.clear()
            custom_batch_events.clear()

    def abort(self) -> None:
        """Emergency abort: signals stop, interrupts active SQLite operations, and drains queue."""
        self._stop_event.set()
        conn = self._conn
        if conn is not None:
            try:
                conn.interrupt()
            except Exception:
                pass
        try:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break
        except Exception:
            pass
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def stop(self, timeout: float = 0.5) -> None:
        """Gracefully stops writer thread within timeout, falling back to emergency abort.
        Ensures self._conn is closed and set to None before returning.
        """
        self._stop_event.set()
        try:
            self.queue.put_nowait(("STOP", None))
        except Exception:
            pass
        self.join(timeout=timeout)
        if self.is_alive():
            logger.warning("TelemetryWriterThread did not exit within %.2fs timeout. Invoking abort.", timeout)
            self.abort()
            self.join(timeout=0.1)
        # Defensive safeguard: ensure connection is closed if thread hung or connection remains open
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
