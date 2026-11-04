"""engine/telemetry/db.py
SQLite connection manager, WAL mode configuration, and pragmas.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger("TelemetryDB")

_PRODUCTION_TELEMETRY_DB = (Path(__file__).resolve().parent.parent.parent / "data" / "telemetry.db").resolve()
_INIT_DB_LOCK = threading.Lock()
_INITIALIZED_DB_KEYS: set[Tuple[str, int, int]] = set()


def compact_game_bible_event_payload(
    payload: Dict[str, Any],
    fallback_deliberation: Optional[str] = None,
) -> Dict[str, Any]:
    """Compacts multi-megabyte GameBibleSpec snapshots inside event-feed payloads into lightweight metadata."""
    tx = payload.get("writers_room_transcript")
    evo = payload.get("concept_evolution")
    sb = payload.get("experience_storyboard")
    mocks = payload.get("visual_mockups")
    pillars = payload.get("gameplay_pillars")
    return {
        "deliberation": str(payload.get("deliberation") or fallback_deliberation or "Living Game Bible updated."),
        "bible_id": payload.get("bible_id"),
        "cartridge_id": payload.get("cartridge_id"),
        "target_episode": payload.get("target_episode", 0),
        "status": payload.get("status"),
        "game_title": payload.get("game_title"),
        "core_genre": payload.get("core_genre"),
        "elevator_pitch": payload.get("elevator_pitch"),
        "creative_vision_summary": payload.get("creative_vision_summary"),
        "updated_at_utc": payload.get("updated_at_utc"),
        "turn_count": payload.get("turn_count") if payload.get("turn_count") is not None else (len(tx) if isinstance(tx, list) else 0),
        "evolution_count": payload.get("evolution_count") if payload.get("evolution_count") is not None else (len(evo) if isinstance(evo, list) else 0),
        "storyboard_count": payload.get("storyboard_count") if payload.get("storyboard_count") is not None else (len(sb) if isinstance(sb, list) else 0),
        "mockup_count": payload.get("mockup_count") if payload.get("mockup_count") is not None else (len(mocks) if isinstance(mocks, list) else 0),
        "pillar_count": len(pillars) if isinstance(pillars, list) else 0,
    }


class TelemetryDB:
    """Manages SQLite database connections, schemas, and WAL pragmas."""

    def __init__(self, db_path: str | Path = "data/telemetry.db"):
        resolved_candidate = Path(db_path).resolve()
        if resolved_candidate == _PRODUCTION_TELEMETRY_DB:
            env_override = os.environ.get("STUDIO_DB_PATH")
            if env_override:
                db_path = Path(env_override)
            elif "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
                db_path = Path(tempfile.gettempdir()) / "pytest_isolated_telemetry.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_initialized()

    def _db_cache_key(self) -> Optional[Tuple[str, int, int]]:
        try:
            resolved = self.db_path.resolve()
            st = resolved.stat()
            if st.st_size <= 0:
                return None
            return (str(resolved), int(st.st_dev), int(st.st_ino))
        except OSError:
            return None

    def _ensure_initialized(self) -> None:
        key = self._db_cache_key()
        if key is not None and key in _INITIALIZED_DB_KEYS:
            return
        with _INIT_DB_LOCK:
            key = self._db_cache_key()
            if key is not None and key in _INITIALIZED_DB_KEYS:
                return
            self.initialize_database()

    def initialize_database(self) -> None:
        """Applies WAL pragmas and creates table schemas."""
        schema_path = Path(__file__).resolve().parent / "schema.sql"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()
        else:
            schema_sql = ""

        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA synchronous = NORMAL;")
            cursor.execute("PRAGMA busy_timeout = 5000;")
            cursor.execute("PRAGMA foreign_keys = ON;")

            # Safe auto-migration for existing telemetry_events tables before schema_sql index creation
            cursor.execute("PRAGMA table_info(telemetry_events);")
            cols = [r[1] for r in cursor.fetchall()]
            if cols and "probe_name" not in cols:
                cursor.execute("ALTER TABLE telemetry_events ADD COLUMN probe_name TEXT;")

            if schema_sql:
                conn.executescript(schema_sql)

            # Ensure indices exist
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_probe ON telemetry_events(probe_name);")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS custom_telemetry_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    probe_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    payload_json TEXT DEFAULT '{}',
                    frame_index INTEGER DEFAULT 0,
                    timestamp_utc TEXT,
                    FOREIGN KEY (session_id) REFERENCES telemetry_sessions(session_id),
                    FOREIGN KEY (episode_id) REFERENCES telemetry_episodes(episode_id),
                    FOREIGN KEY (probe_name) REFERENCES telemetry_probes(probe_name)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_custom_events_probe ON custom_telemetry_events(probe_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_custom_events_episode ON custom_telemetry_events(episode_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_custom_events_category ON custom_telemetry_events(category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_custom_events_timestamp ON custom_telemetry_events(timestamp);")

            # Standard built-in engine probes seeded for foreign key referential integrity on game telemetry DB
            if self.db_path.name == "telemetry.db":
                cursor.execute("""
                    INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target, schema_json)
                    VALUES 
                        ('probe_dialogue_bark_played', 'dialogue_bark_played', 'dialogue', 'Reactive dialogue bark played in PlayingScene', 'H_DIALOGUE_ENGAGEMENT', '{}'),
                        ('probe_cartridge_hot_reload_triggered', 'cartridge_hot_reload_triggered', 'system', 'In-game cartridge hot-reload event', 'H_HOT_RELOAD', '{}')
                    ON CONFLICT(probe_name) DO NOTHING;
                """)

            # Initialize relational studio schema
            try:
                from engine.telemetry.schema import init_studio_schema
                init_studio_schema(conn)
            except Exception as e:
                logger.debug("Studio schema init error (handled): %s", e)

            conn.commit()
        finally:
            conn.close()
        key = self._db_cache_key()
        if key is not None:
            _INITIALIZED_DB_KEYS.add(key)

    def get_writer_connection(self) -> sqlite3.Connection:
        """Isolated connection for background writer thread with manual transaction control."""
        conn = sqlite3.connect(str(self.db_path), timeout=5.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def get_write_connection(self, timeout: float = 5.0) -> sqlite3.Connection:
        """Isolated connection for write operations enforcing WAL pragmas, busy_timeout, and foreign keys."""
        conn = sqlite3.connect(str(self.db_path), timeout=timeout)
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def get_reader_connection(self) -> sqlite3.Connection:
        """Isolated read-only connection for Analyst and MCP server."""
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA query_only = ON;")
        return conn

    def checkpoint(self, mode: str = "PASSIVE", timeout: float = 5.0) -> None:
        """Triggers WAL checkpoint (PASSIVE, FULL, RESTART, TRUNCATE)."""
        timeout_sec = max(0.05, float(timeout))
        conn = sqlite3.connect(str(self.db_path), timeout=timeout_sec)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(f"PRAGMA busy_timeout = {int(timeout_sec * 1000)};")
            conn.execute(f"PRAGMA wal_checkpoint({mode});")
            conn.commit()
        finally:
            conn.close()

    def register_probe(
        self,
        probe_id: str,
        probe_name: str,
        category: str,
        description: str,
        hypothesis_target: str,
        schema_json: Optional[Dict[str, Any] | str] = None,
    ) -> None:
        """Synchronously registers or updates a probe metadata entry."""
        probe_id_str = str(probe_id or "")
        probe_name_str = str(probe_name or "")
        category_str = str(category or "general")
        description_str = str(description or "")
        hypothesis_str = str(hypothesis_target or "")

        if isinstance(schema_json, dict):
            try:
                schema_str = json.dumps(schema_json, default=str, separators=(',', ':'))
            except Exception:
                schema_str = "{}"
        elif isinstance(schema_json, str):
            schema_str = schema_json
        else:
            schema_str = "{}"
        now_utc = datetime.now(timezone.utc).isoformat()
        conn = self.get_write_connection()
        try:
            conn.execute(
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
                (probe_id_str, probe_name_str, category_str, description_str, hypothesis_str, schema_str, now_utc),
            )
            conn.commit()
        finally:
            conn.close()

    def get_probe(self, probe_name: str) -> Optional[Dict[str, Any]]:
        """Queries probe metadata by probe_name."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM telemetry_probes WHERE probe_name = ?;", (probe_name,))
            row = cur.fetchone()
            if not row:
                return None
            return dict(row)
        finally:
            conn.close()

    def list_probes(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists all registered probes, optionally filtered by category."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            if category:
                cur.execute("SELECT * FROM telemetry_probes WHERE category = ? ORDER BY created_at ASC;", (category,))
            else:
                cur.execute("SELECT * FROM telemetry_probes ORDER BY created_at ASC;")
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def query_custom_events(
        self,
        session_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        probe_name: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Queries custom probe events from custom_telemetry_events table."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            clauses = []
            params: List[Any] = []
            if session_id:
                clauses.append("session_id = ?")
                params.append(session_id)
            if episode_id:
                clauses.append("episode_id = ?")
                params.append(episode_id)
            if probe_name:
                clauses.append("probe_name = ?")
                params.append(probe_name)
            if category:
                clauses.append("category = ?")
                params.append(category)

            where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            query = f"SELECT * FROM custom_telemetry_events {where_str} ORDER BY timestamp DESC LIMIT ? OFFSET ?;"
            params.extend([max(1, limit), max(0, offset)])
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def count_custom_events(
        self,
        probe_name: Optional[str] = None,
        category: Optional[str] = None,
    ) -> int:
        """Counts matching custom probe events in custom_telemetry_events table."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            clauses = []
            params: List[Any] = []
            if probe_name:
                clauses.append("probe_name = ?")
                params.append(probe_name)
            if category:
                clauses.append("category = ?")
                params.append(category)
            where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            cur.execute(f"SELECT COUNT(*) FROM custom_telemetry_events {where_str};", params)
            row = cur.fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Relational Studio Observability Tables (Milestone 7 Godot 4 Transition)
    # -------------------------------------------------------------------------

    def record_studio_build(
        self,
        build_id: str,
        candidate_id: Optional[str] = None,
        build_type: str = "genesis",
        engine_type: str = "godot4",
        game_title: Optional[str] = None,
        genre_paradigm: Optional[str] = None,
        status: str = "STARTED",
        started_at_utc: Optional[str] = None,
        deliberation: Optional[str] = None,
        completed_at_utc: Optional[str] = None,
    ) -> None:
        """Records or updates a studio build session."""
        started_iso = started_at_utc or datetime.now(timezone.utc).isoformat()
        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT INTO studio_builds (
                    build_id, candidate_id, build_type, engine_type, game_title,
                    genre_paradigm, status, started_at_utc, completed_at_utc, deliberation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(build_id) DO UPDATE SET
                    candidate_id=coalesce(excluded.candidate_id, studio_builds.candidate_id),
                    build_type=coalesce(excluded.build_type, studio_builds.build_type),
                    engine_type=coalesce(excluded.engine_type, studio_builds.engine_type),
                    game_title=coalesce(excluded.game_title, studio_builds.game_title),
                    genre_paradigm=coalesce(excluded.genre_paradigm, studio_builds.genre_paradigm),
                    status=case when studio_builds.status = 'PROMOTED' then studio_builds.status else excluded.status end,
                    completed_at_utc=coalesce(excluded.completed_at_utc, studio_builds.completed_at_utc),
                    deliberation=coalesce(excluded.deliberation, studio_builds.deliberation)
                """,
                (
                    build_id,
                    candidate_id,
                    build_type,
                    engine_type,
                    game_title,
                    genre_paradigm,
                    status,
                    started_iso,
                    completed_at_utc,
                    deliberation,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def update_studio_build(
        self,
        build_id: str,
        status: Optional[str] = None,
        completed_at_utc: Optional[str] = None,
        deliberation: Optional[str] = None,
    ) -> None:
        """Updates status, completion timestamp, or deliberation of a studio build."""
        conn = self.get_write_connection()
        try:
            updates = []
            params = []
            if status is not None:
                updates.append("status = ?")
                params.append(status)
            if completed_at_utc is not None:
                updates.append("completed_at_utc = ?")
                params.append(completed_at_utc)
            if deliberation is not None:
                updates.append("deliberation = ?")
                params.append(deliberation)
            if updates:
                params.append(build_id)
                conn.execute(f"UPDATE studio_builds SET {', '.join(updates)} WHERE build_id = ?", params)
                conn.commit()
        finally:
            conn.close()

    def get_latest_build(self, status: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Returns the most recent studio build record."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            if status:
                cur.execute("SELECT * FROM studio_builds WHERE status = ? ORDER BY started_at_utc DESC LIMIT 1;", (status,))
            else:
                cur.execute("SELECT * FROM studio_builds ORDER BY started_at_utc DESC LIMIT 1;")
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_studio_build(self, build_id: str) -> Optional[Dict[str, Any]]:
        """Returns the studio build record for a specific build_id."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM studio_builds WHERE build_id = ?;", (build_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def record_studio_event(
        self,
        event_id: str,
        build_id: str,
        event_type: str,
        agent_name: str,
        timestamp_utc: str,
        deliberation: Optional[str] = None,
        payload: Optional[Dict[str, Any] | str] = None,
    ) -> None:
        """Records a studio swarm event in the SQLite WAL studio_events table."""
        if isinstance(payload, dict):
            try:
                payload_str = json.dumps(payload, default=str)
            except Exception:
                payload_str = "{}"
        elif isinstance(payload, str):
            payload_str = payload
        else:
            payload_str = "{}"

        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT INTO studio_events (
                    event_id, build_id, event_type, agent_name, timestamp_utc, deliberation, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    deliberation=excluded.deliberation,
                    payload_json=excluded.payload_json
                """,
                (event_id, build_id, event_type, agent_name, timestamp_utc, deliberation, payload_str),
            )
            conn.commit()
        finally:
            conn.close()

    def get_studio_events(
        self,
        build_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Returns studio swarm events in chronological order, returning the most recent `limit` events when `limit` is set."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            params: List[Any] = []
            if limit and limit > 0:
                inner = "SELECT rowid, * FROM studio_events"
                if build_id:
                    inner += " WHERE build_id = ?"
                    params.append(build_id)
                inner += " ORDER BY rowid DESC LIMIT ?"
                params.append(limit)
                query = f"SELECT * FROM ({inner}) ORDER BY rowid ASC"
            else:
                query = "SELECT * FROM studio_events"
                if build_id:
                    query += " WHERE build_id = ?"
                    params.append(build_id)
                query += " ORDER BY rowid ASC"
            cur.execute(query, params)
            out_rows: List[Dict[str, Any]] = []
            for raw_row in cur.fetchall():
                r = dict(raw_row)
                r.pop("rowid", None)
                raw_payload = r.pop("payload_json", None)
                pl: Any = {}
                if raw_payload:
                    try:
                        pl = json.loads(raw_payload)
                    except Exception:
                        pl = {}
                if (
                    r.get("event_type") in ("game_bible_updated", "GAME_BIBLE_UPDATED")
                    and isinstance(pl, dict)
                    and any(
                        heavy_k in pl
                        for heavy_k in (
                            "writers_room_transcript",
                            "experience_storyboard",
                            "concept_evolution",
                            "visual_mockups",
                            "world_spec",
                            "narrative_bible",
                        )
                    )
                ):
                    pl = compact_game_bible_event_payload(pl, r.get("deliberation"))
                delib_val = r.pop("deliberation", None)
                out_rows.append(
                    {
                        "deliberation": "" if delib_val is None else str(delib_val),
                        **r,
                        "payload": pl,
                    }
                )
            return out_rows
        finally:
            conn.close()

    def record_studio_task(
        self,
        task_data: Dict[str, Any],
        build_id: str,
        seq_order: int = 0,
    ) -> None:
        """Records an agent task record into studio_tasks and indexes output artifacts."""
        tid = task_data.get("task_id") or f"task_{uuid.uuid4().hex[:8]}"
        role = task_data.get("agent_role") or task_data.get("agent_name", "StudioAgent")
        name = task_data.get("agent_name", role)
        req_id = task_data.get("requirement_id")
        title = task_data.get("track_title", "")
        ttype = task_data.get("task_type", "GENERAL")
        status = task_data.get("status", "COMPLETED")
        start_utc = task_data.get("start_time_utc") or datetime.now(timezone.utc).isoformat()
        end_utc = task_data.get("end_time_utc") or start_utc
        dur = float(task_data.get("duration_ms", 0.0) or 0.0)
        delib = task_data.get("deliberation", "Task execution.")
        verdict = task_data.get("verification_verdict") or {}
        trace = task_data.get("diagnostic_trace") or {}
        summary = task_data.get("summary", "")

        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT INTO studio_tasks (
                    task_id, build_id, seq_order, agent_role, agent_name, requirement_id,
                    track_title, task_type, status, start_time_utc, end_time_utc, duration_ms,
                    deliberation, verification_verdict_json, diagnostic_trace_json, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status=excluded.status,
                    end_time_utc=excluded.end_time_utc,
                    duration_ms=excluded.duration_ms,
                    deliberation=excluded.deliberation,
                    verification_verdict_json=excluded.verification_verdict_json,
                    diagnostic_trace_json=excluded.diagnostic_trace_json,
                    summary=excluded.summary
                """,
                (
                    tid, build_id, seq_order, role, name, req_id, title, ttype, status,
                    start_utc, end_utc, dur, delib,
                    json.dumps(verdict, default=str),
                    json.dumps(trace, default=str),
                    summary,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        # Also index any input and output artifacts from this task
        for art in task_data.get("input_artifacts", []):
            if isinstance(art, dict):
                art_with_agents = dict(art)
                art_with_agents.setdefault("target_agent", role)
                art_with_agents["direction"] = "INPUT"
                self.record_studio_artifact(art_with_agents, build_id=build_id, task_id=tid)
        for art in task_data.get("output_artifacts", []):
            if isinstance(art, dict):
                art_with_agents = dict(art)
                art_with_agents.setdefault("source_agent", role)
                art_with_agents["direction"] = "OUTPUT"
                self.record_studio_artifact(art_with_agents, build_id=build_id, task_id=tid)

    def get_studio_tasks(
        self,
        build_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Returns recorded studio tasks, optionally filtered by build_id or task_id."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            clauses = []
            params = []
            if build_id:
                clauses.append("build_id = ?")
                params.append(build_id)
            if task_id:
                clauses.append("task_id = ?")
                params.append(task_id)
            where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            cur.execute(f"SELECT * FROM studio_tasks {where_str} ORDER BY seq_order ASC, start_time_utc ASC;", params)
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                if "verification_verdict_json" in r and r["verification_verdict_json"]:
                    try:
                        r["verification_verdict"] = json.loads(r["verification_verdict_json"])
                    except Exception:
                        r["verification_verdict"] = {}
                else:
                    r["verification_verdict"] = {}
                if "diagnostic_trace_json" in r and r["diagnostic_trace_json"]:
                    try:
                        r["diagnostic_trace"] = json.loads(r["diagnostic_trace_json"])
                    except Exception:
                        r["diagnostic_trace"] = {}
                else:
                    r["diagnostic_trace"] = {}

                # Attach input_artifacts and output_artifacts from studio_artifacts
                cur.execute(
                    "SELECT * FROM studio_artifacts WHERE task_id = ? ORDER BY created_at_utc ASC;",
                    (r.get("task_id", ""),),
                )
                art_rows = [dict(ar) for ar in cur.fetchall()]
                if not art_rows and r.get("build_id") and r.get("agent_role"):
                    cur.execute(
                        "SELECT * FROM studio_artifacts WHERE build_id = ? AND source_agent = ? ORDER BY created_at_utc ASC;",
                        (r["build_id"], r["agent_role"]),
                    )
                    seen_names = set()
                    for ar in cur.fetchall():
                        ard = dict(ar)
                        aname = ard.get("artifact_name", "")
                        if aname not in seen_names:
                            seen_names.add(aname)
                            art_rows.append(ard)

                in_arts: List[Dict[str, Any]] = []
                out_arts: List[Dict[str, Any]] = []
                for ar in art_rows:
                    meta = {}
                    if ar.get("metadata_json"):
                        try:
                            meta = json.loads(ar["metadata_json"])
                        except Exception:
                            meta = {}
                    art_dict = {
                        "deliberation": ar.get("deliberation") or f"Artifact {ar.get('artifact_name', '')}",
                        "artifact_id": ar.get("artifact_id", ""),
                        "artifact_name": ar.get("artifact_name", ""),
                        "artifact_type": ar.get("artifact_type", "FILE"),
                        "file_path": ar.get("file_path", ""),
                        "sha256_hash": ar.get("sha256_hash", ""),
                        "byte_size": int(ar.get("byte_size", 0) or 0),
                        "created_at_utc": ar.get("created_at_utc", ""),
                        "source_agent": ar.get("source_agent", ""),
                        "target_agent": ar.get("target_agent", ""),
                        "metadata": meta,
                        "metadata_summary": meta if isinstance(meta, dict) else {},
                    }
                    if str(ar.get("direction", "OUTPUT")).upper() == "INPUT":
                        in_arts.append(art_dict)
                    else:
                        out_arts.append(art_dict)
                r["input_artifacts"] = in_arts
                r["output_artifacts"] = out_arts
            return rows
        finally:
            conn.close()

    def record_studio_artifact(
        self,
        artifact_data: Dict[str, Any],
        build_id: str,
        task_id: Optional[str] = None,
    ) -> None:
        """Records an artifact record in studio_artifacts."""
        aid = artifact_data.get("artifact_id") or f"art_{uuid.uuid4().hex[:8]}"
        direction = artifact_data.get("direction", "OUTPUT")
        name = artifact_data.get("artifact_name") or artifact_data.get("filename", "")
        atype = artifact_data.get("artifact_type", "FILE")
        fpath = str(artifact_data.get("file_path", ""))
        sha = artifact_data.get("sha256_hash", "")
        size = int(artifact_data.get("byte_size", 0) or 0)
        src = artifact_data.get("source_agent", "")
        dst = artifact_data.get("target_agent", "")
        delib = artifact_data.get("deliberation", "Artifact metadata.")
        created_utc = artifact_data.get("created_at_utc") or datetime.now(timezone.utc).isoformat()
        meta = artifact_data.get("metadata", {})

        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT INTO studio_artifacts (
                    artifact_id, task_id, build_id, direction, artifact_name, artifact_type,
                    file_path, sha256_hash, byte_size, source_agent, target_agent,
                    deliberation, created_at_utc, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    sha256_hash=coalesce(nullif(excluded.sha256_hash, ''), studio_artifacts.sha256_hash),
                    byte_size=case when excluded.byte_size > 0 then excluded.byte_size else studio_artifacts.byte_size end,
                    metadata_json=excluded.metadata_json
                """,
                (
                    aid, task_id, build_id, direction, name, atype, fpath, sha, size,
                    src, dst, delib, created_utc, json.dumps(meta, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_studio_artifacts(
        self,
        build_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Returns artifacts, optionally filtered by build_id or task_id."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            clauses = []
            params = []
            if build_id:
                clauses.append("build_id = ?")
                params.append(build_id)
            if task_id:
                clauses.append("task_id = ?")
                params.append(task_id)
            where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            cur.execute(f"SELECT * FROM studio_artifacts {where_str} ORDER BY created_at_utc ASC;", params)
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                if "metadata_json" in r and r["metadata_json"]:
                    try:
                        r["metadata"] = json.loads(r["metadata_json"])
                    except Exception:
                        r["metadata"] = {}
                else:
                    r["metadata"] = {}
            return rows
        finally:
            conn.close()

    def record_studio_requirements(
        self,
        build_id: str,
        spec: Any,
    ) -> None:
        """Records requirement tracks and full GameRequirementsSpec (including asset_manifest) into SQLite."""
        if hasattr(spec, "model_dump"):
            spec_data = spec.model_dump()
        elif isinstance(spec, dict):
            spec_data = dict(spec)
        else:
            return

        requirements = spec_data.get("requirements", [])
        now_utc = datetime.now(timezone.utc).isoformat()
        conn = self.get_write_connection()
        try:
            # 0. Ensure parent studio_builds row exists so foreign keys on studio_requirement_specs and studio_requirements succeed
            conn.execute(
                """
                INSERT OR IGNORE INTO studio_builds (
                    build_id, status, started_at_utc, deliberation
                ) VALUES (?, 'RUNNING', ?, 'Auto-registered build record for studio requirements.')
                """,
                (build_id, now_utc),
            )

            # 1. Persist full GameRequirementsSpec JSON (preserving asset_manifest, high_concept_summary, etc.)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS studio_requirement_specs (
                        build_id TEXT PRIMARY KEY,
                        spec_id TEXT,
                        game_title TEXT,
                        genre_paradigm TEXT,
                        spec_json TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    INSERT INTO studio_requirement_specs (
                        build_id, spec_id, game_title, genre_paradigm, spec_json, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(build_id) DO UPDATE SET
                        spec_id=excluded.spec_id,
                        game_title=excluded.game_title,
                        genre_paradigm=excluded.genre_paradigm,
                        spec_json=excluded.spec_json,
                        updated_at_utc=excluded.updated_at_utc
                    """,
                    (
                        build_id,
                        str(spec_data.get("spec_id") or ""),
                        str(spec_data.get("game_title") or ""),
                        str(spec_data.get("genre_paradigm") or spec_data.get("genre_verdict") or ""),
                        json.dumps(spec_data, default=str),
                        now_utc,
                    ),
                )
            except Exception as e:
                logger.debug("studio_requirement_specs upsert error: %s", e)

            # 2. Persist normalized track rows into studio_requirements
            for idx, req in enumerate(requirements):
                if hasattr(req, "model_dump"):
                    req = req.model_dump()
                rid = req.get("id") or req.get("track_id") or f"R{idx+1}"
                title = req.get("title", f"Requirement {rid}")
                desc = req.get("description", "")
                status = req.get("status", "pending")
                agents = req.get("responsible_agents") or req.get("responsible_agent", ["LeadGameplaySWEAgent"])
                owner = agents[0] if isinstance(agents, list) and agents else str(agents)
                targets = req.get("target_modules", ["mechanics.gd"])
                criteria = req.get("acceptance_criteria", [])
                delib = req.get("deliberation", "Requirement track.")

                conn.execute(
                    """
                    INSERT INTO studio_requirements (
                        build_id, requirement_id, seq_order, title, description, status,
                        specialist_owner, target_modules_json, acceptance_criteria_json,
                        deliberation, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(build_id, requirement_id) DO UPDATE SET
                        title=excluded.title,
                        description=excluded.description,
                        status=excluded.status,
                        specialist_owner=excluded.specialist_owner,
                        target_modules_json=excluded.target_modules_json,
                        acceptance_criteria_json=excluded.acceptance_criteria_json,
                        deliberation=excluded.deliberation,
                        updated_at_utc=excluded.updated_at_utc
                    """,
                    (
                        build_id, rid, idx, title, desc, status, owner,
                        json.dumps(targets, default=str),
                        json.dumps(criteria, default=str),
                        delib, now_utc,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def update_studio_requirement_status(
        self,
        build_id: str,
        requirement_id: str,
        status: str,
        updated_at_utc: Optional[str] = None,
    ) -> None:
        """Updates the status of a specific requirement in studio_requirements and studio_requirement_specs."""
        now_utc = updated_at_utc or datetime.now(timezone.utc).isoformat()
        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                UPDATE studio_requirements
                SET status = ?, updated_at_utc = ?
                WHERE build_id = ? AND requirement_id = ?
                """,
                (status, now_utc, build_id, requirement_id),
            )
            try:
                cur = conn.cursor()
                cur.execute("SELECT spec_json FROM studio_requirement_specs WHERE build_id = ?;", (build_id,))
                row = cur.fetchone()
                if row and row[0]:
                    spec_dict = json.loads(row[0])
                    for r in spec_dict.get("requirements", []):
                        if isinstance(r, dict) and (r.get("id") == requirement_id or r.get("track_id") == requirement_id):
                            r["status"] = status
                    conn.execute(
                        "UPDATE studio_requirement_specs SET spec_json = ?, updated_at_utc = ? WHERE build_id = ?;",
                        (json.dumps(spec_dict, default=str), now_utc, build_id),
                    )
            except Exception:
                pass
            conn.commit()
        finally:
            conn.close()

    def get_studio_requirements(self, build_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns requirement tracks from studio_requirements, ordered by sequence."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            query = "SELECT * FROM studio_requirements"
            params = []
            if build_id:
                query += " WHERE build_id = ?"
                params.append(build_id)
            query += " ORDER BY seq_order ASC;"
            cur.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                if "target_modules_json" in r and r["target_modules_json"]:
                    try:
                        r["target_modules"] = json.loads(r["target_modules_json"])
                    except Exception:
                        r["target_modules"] = []
                else:
                    r["target_modules"] = []
                if "acceptance_criteria_json" in r and r["acceptance_criteria_json"]:
                    try:
                        r["acceptance_criteria"] = json.loads(r["acceptance_criteria_json"])
                    except Exception:
                        r["acceptance_criteria"] = []
                else:
                    r["acceptance_criteria"] = []
                if "requirement_id" in r:
                    r["id"] = r["requirement_id"]
            return rows
        finally:
            conn.close()

    def get_studio_requirements_spec(self, build_id: str) -> Optional[Dict[str, Any]]:
        """Returns the full GameRequirementsSpec dictionary (including asset_manifest) merged with latest track statuses."""
        conn = self.get_reader_connection()
        spec_dict: Optional[Dict[str, Any]] = None
        try:
            cur = conn.cursor()
            cur.execute("SELECT spec_json FROM studio_requirement_specs WHERE build_id = ?;", (build_id,))
            row = cur.fetchone()
            if row and row["spec_json"]:
                spec_dict = json.loads(row["spec_json"])
        except Exception:
            spec_dict = None
        finally:
            conn.close()

        tracks = self.get_studio_requirements(build_id=build_id)
        if spec_dict and isinstance(spec_dict, dict):
            if tracks:
                status_by_id = {str(t.get("id") or t.get("requirement_id")): t.get("status", "pending") for t in tracks}
                for req in spec_dict.get("requirements", []):
                    if isinstance(req, dict):
                        rid = str(req.get("id") or req.get("track_id") or "")
                        if rid in status_by_id:
                            req["status"] = status_by_id[rid]
            return spec_dict
        return None

    def record_studio_checkpoint(
        self,
        checkpoint_id: str,
        build_id: str,
        candidate_id: Optional[str] = None,
        requirement_id: Optional[str] = None,
        track_title: Optional[str] = None,
        engine_type: str = "godot4",
        checkpoint_path: Optional[str] = None,
        verified_tracks: Optional[List[str]] = None,
        deliberation: Optional[str] = None,
        created_at_utc: Optional[str] = None,
    ) -> None:
        """Records a verified incremental checkpoint in studio_checkpoints."""
        created_iso = created_at_utc or datetime.now(timezone.utc).isoformat()
        tracks_json = json.dumps(verified_tracks or [], default=str)
        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT INTO studio_checkpoints (
                    checkpoint_id, build_id, candidate_id, requirement_id, track_title,
                    engine_type, checkpoint_path, verified_tracks_json, created_at_utc, deliberation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(checkpoint_id) DO UPDATE SET
                    checkpoint_path=excluded.checkpoint_path,
                    verified_tracks_json=excluded.verified_tracks_json,
                    deliberation=excluded.deliberation
                """,
                (
                    checkpoint_id, build_id, candidate_id, requirement_id, track_title,
                    engine_type, checkpoint_path, tracks_json, created_iso, deliberation,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_studio_checkpoints(self, build_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns verified checkpoints from studio_checkpoints."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            query = "SELECT * FROM studio_checkpoints"
            params = []
            if build_id:
                query += " WHERE build_id = ?"
                params.append(build_id)
            query += " ORDER BY created_at_utc ASC;"
            cur.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                if "verified_tracks_json" in r and r["verified_tracks_json"]:
                    try:
                        r["verified_tracks"] = json.loads(r["verified_tracks_json"])
                    except Exception:
                        r["verified_tracks"] = []
                else:
                    r["verified_tracks"] = []
            return rows
        finally:
            conn.close()

    def upsert_game_bible(
        self,
        bible: Any = None,
        *,
        bible_id: Optional[str] = None,
        cartridge_id: Optional[str] = None,
        status: Optional[str] = None,
        bible_dict: Optional[Dict[str, Any]] = None,
        build_id: Optional[str] = None,
    ) -> None:
        """Upserts a GameBibleSpec (or dict) into studio_game_bibles."""
        target_bible = bible if bible is not None else bible_dict
        if hasattr(target_bible, "model_dump"):
            b_dict = target_bible.model_dump(mode="json")
            bible_json = target_bible.model_dump_json()
        elif isinstance(target_bible, dict):
            b_dict = dict(target_bible)
            bible_json = json.dumps(b_dict, default=str)
        else:
            raise ValueError(f"Unsupported game bible type: {type(target_bible)}")

        now_iso = datetime.now(timezone.utc).isoformat()
        resolved_bible_id = str(bible_id or b_dict.get("bible_id") or f"game_bible_{uuid.uuid4().hex[:8]}")
        resolved_cartridge_id = str(cartridge_id or build_id or b_dict.get("cartridge_id") or "active")
        target_ep = int(b_dict.get("target_episode", 0) or 0)
        resolved_status = str(b_dict.get("status") or status or "brainstorming").lower()
        game_title = str(b_dict.get("game_title") or "Untitled Concept")
        created_at = str(b_dict.get("created_at_utc") or now_iso)
        updated_at = str(b_dict.get("updated_at_utc") or now_iso)

        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT INTO studio_game_bibles (
                    bible_id, cartridge_id, target_episode, status,
                    game_title, bible_json, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(bible_id) DO UPDATE SET
                    cartridge_id=excluded.cartridge_id,
                    target_episode=excluded.target_episode,
                    status=excluded.status,
                    game_title=excluded.game_title,
                    bible_json=excluded.bible_json,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (
                    resolved_bible_id,
                    resolved_cartridge_id,
                    target_ep,
                    resolved_status,
                    game_title,
                    bible_json,
                    created_at,
                    updated_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_latest_game_bible(
        self,
        cartridge_id: Optional[str] = None,
        build_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Returns the latest GameBibleSpec dictionary from studio_game_bibles."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            if cartridge_id:
                cur.execute(
                    "SELECT * FROM studio_game_bibles WHERE cartridge_id = ? ORDER BY updated_at_utc DESC LIMIT 1;",
                    (cartridge_id,),
                )
            else:
                cur.execute(
                    "SELECT * FROM studio_game_bibles ORDER BY updated_at_utc DESC LIMIT 1;"
                )
            row = cur.fetchone()
            if not row:
                return None
            row_dict = dict(row)
            raw_json = row_dict.get("bible_json", "{}")
            try:
                parsed = json.loads(raw_json)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return row_dict
        except Exception:
            return None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Blind Playtester Persistence (Sessions, Turns, Compactions, Wishes)
    # ------------------------------------------------------------------

    def upsert_playtest_session(self, record: Dict[str, Any]) -> None:
        """Inserts or updates a Blind Playtester session row in `playtest_sessions`."""
        now_iso = datetime.now(timezone.utc).isoformat()
        session_id = str(record["session_id"])
        checkpoint_id = str(record.get("checkpoint_id") or "active")
        candidate_id = record.get("candidate_id")
        slice_id = record.get("slice_id")
        initiated_by = str(record.get("initiated_by") or "operator")
        game_title = str(record.get("game_title") or checkpoint_id)
        game_description = str(record.get("game_description") or "")
        status = str(record.get("status") or "running")
        end_outcome = record.get("end_outcome")
        end_summary = record.get("end_summary")
        error_message = record.get("error_message")
        turn_count = int(record.get("turn_count", 0) or 0)
        sim_ticks_total = int(record.get("sim_ticks_total", 0) or 0)
        sim_duration_sec = float(record.get("sim_duration_sec", 0.0) or 0.0)
        compaction_count = int(record.get("compaction_count", 0) or 0)

        rb = record.get("rulebook") if "rulebook" in record else record.get("rulebook_json", {})
        rulebook_json = rb if isinstance(rb, str) else json.dumps(rb or {}, default=str)

        fn = record.get("field_notes") if "field_notes" in record else record.get("field_notes_json", {})
        field_notes_json = fn if isinstance(fn, str) else json.dumps(fn or {}, default=str)

        qs = record.get("questions") if "questions" in record else record.get("questions_json", {})
        questions_json = qs if isinstance(qs, str) else json.dumps(qs or {}, default=str)

        bf = record.get("bugs_found") if "bugs_found" in record else record.get("bugs_found_json", {})
        bugs_found_json = bf if isinstance(bf, str) else json.dumps(bf or {}, default=str)

        imp = record.get("impressions") if "impressions" in record else record.get("impressions_json", {})
        impressions_json = imp if isinstance(imp, str) else json.dumps(imp or {}, default=str)

        sr = record.get("session_report") if "session_report" in record else record.get("session_report_json", {})
        if hasattr(sr, "model_dump"):
            session_report_json = sr.model_dump_json()
        elif isinstance(sr, str):
            session_report_json = sr
        else:
            session_report_json = json.dumps(sr or {}, default=str)

        cj = record.get("compacted_journal") if "compacted_journal" in record else record.get("compacted_journal_json", {})
        if hasattr(cj, "model_dump"):
            compacted_journal_json = cj.model_dump_json()
        elif isinstance(cj, str):
            compacted_journal_json = cj
        else:
            compacted_journal_json = json.dumps(cj or {}, default=str)

        current_plan = str(record.get("current_plan") or "")
        current_mood = str(record.get("current_mood") or "")
        prompt_tokens = int(record.get("prompt_tokens", 0) or 0)
        cached_tokens = int(record.get("cached_tokens", 0) or 0)
        output_tokens = int(record.get("output_tokens", 0) or 0)
        thought_tokens = int(record.get("thought_tokens", 0) or 0)
        video_path = record.get("video_path")
        latest_frame_path = record.get("latest_frame_path")
        started_at_utc = str(record.get("started_at_utc") or now_iso)
        updated_at_utc = str(record.get("updated_at_utc") or now_iso)
        ended_at_utc = record.get("ended_at_utc")
        deliberation = str(record.get("deliberation") or f"Playtest session {session_id} state={status}.")

        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT INTO playtest_sessions (
                    session_id, checkpoint_id, candidate_id, slice_id, initiated_by,
                    game_title, game_description, status, end_outcome,
                    end_summary, error_message, turn_count, sim_ticks_total,
                    sim_duration_sec, compaction_count, rulebook_json,
                    field_notes_json, questions_json, bugs_found_json,
                    impressions_json, session_report_json,
                    compacted_journal_json, current_plan, current_mood,
                    prompt_tokens, cached_tokens, output_tokens, thought_tokens,
                    video_path, latest_frame_path, started_at_utc, updated_at_utc,
                    ended_at_utc, deliberation
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(session_id) DO UPDATE SET
                    initiated_by=COALESCE(excluded.initiated_by, playtest_sessions.initiated_by),
                    status=excluded.status,
                    end_outcome=excluded.end_outcome,
                    end_summary=excluded.end_summary,
                    error_message=excluded.error_message,
                    turn_count=excluded.turn_count,
                    sim_ticks_total=excluded.sim_ticks_total,
                    sim_duration_sec=excluded.sim_duration_sec,
                    compaction_count=excluded.compaction_count,
                    rulebook_json=excluded.rulebook_json,
                    field_notes_json=excluded.field_notes_json,
                    questions_json=excluded.questions_json,
                    bugs_found_json=excluded.bugs_found_json,
                    impressions_json=excluded.impressions_json,
                    session_report_json=CASE
                        WHEN excluded.session_report_json IS NOT NULL AND excluded.session_report_json != '{}'
                        THEN excluded.session_report_json
                        ELSE playtest_sessions.session_report_json
                    END,
                    compacted_journal_json=excluded.compacted_journal_json,
                    current_plan=excluded.current_plan,
                    current_mood=excluded.current_mood,
                    prompt_tokens=excluded.prompt_tokens,
                    cached_tokens=excluded.cached_tokens,
                    output_tokens=excluded.output_tokens,
                    thought_tokens=excluded.thought_tokens,
                    video_path=COALESCE(excluded.video_path, playtest_sessions.video_path),
                    latest_frame_path=COALESCE(excluded.latest_frame_path, playtest_sessions.latest_frame_path),
                    updated_at_utc=excluded.updated_at_utc,
                    ended_at_utc=COALESCE(excluded.ended_at_utc, playtest_sessions.ended_at_utc),
                    deliberation=excluded.deliberation
                """,
                (
                    session_id, checkpoint_id, candidate_id, slice_id, initiated_by,
                    game_title, game_description, status, end_outcome,
                    end_summary, error_message, turn_count, sim_ticks_total,
                    sim_duration_sec, compaction_count, rulebook_json,
                    field_notes_json, questions_json, bugs_found_json,
                    impressions_json, session_report_json,
                    compacted_journal_json, current_plan, current_mood,
                    prompt_tokens, cached_tokens, output_tokens, thought_tokens,
                    video_path, latest_frame_path, started_at_utc, updated_at_utc,
                    ended_at_utc, deliberation,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _hydrate_playtest_session_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
        for json_col, out_col, fallback in (
            ("rulebook_json", "rulebook", {}),
            ("field_notes_json", "field_notes", {}),
            ("questions_json", "questions", {}),
            ("bugs_found_json", "bugs_found", {}),
            ("impressions_json", "impressions", {}),
            ("session_report_json", "session_report", {}),
            ("compacted_journal_json", "compacted_journal", {}),
        ):
            raw = row_dict.get(json_col)
            if isinstance(raw, str) and raw.strip():
                try:
                    row_dict[out_col] = json.loads(raw)
                except Exception:
                    row_dict[out_col] = fallback
            else:
                row_dict[out_col] = fallback
        return row_dict

    def get_playtest_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM playtest_sessions WHERE session_id = ?;", (session_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._hydrate_playtest_session_row(dict(row))
        finally:
            conn.close()

    def list_playtest_sessions(
        self,
        checkpoint_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            clauses: List[str] = []
            params: List[Any] = []
            if checkpoint_id:
                clauses.append("checkpoint_id = ?")
                params.append(checkpoint_id)
            if candidate_id:
                clauses.append("candidate_id = ?")
                params.append(candidate_id)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            params.append(max(1, int(limit)))
            cur.execute(
                f"SELECT * FROM playtest_sessions {where_sql} ORDER BY started_at_utc DESC LIMIT ?;",
                params,
            )
            return [self._hydrate_playtest_session_row(dict(r)) for r in cur.fetchall()]
        finally:
            conn.close()

    def insert_playtest_turn(self, turn: Dict[str, Any]) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        turn_id = str(turn.get("turn_id") or f"pt_turn_{uuid.uuid4().hex[:12]}")
        act = turn.get("action") if "action" in turn else turn.get("action_json", {})
        action_json = act if isinstance(act, str) else json.dumps(act or {}, default=str)
        fnu = turn.get("field_note_updates") if "field_note_updates" in turn else turn.get("field_note_updates_json", [])
        fnu_json = fnu if isinstance(fnu, str) else json.dumps(fnu or [], default=str)
        qu = turn.get("question_updates") if "question_updates" in turn else turn.get("question_updates_json", [])
        qu_json = qu if isinstance(qu, str) else json.dumps(qu or [], default=str)
        rbu = turn.get("rulebook_updates") if "rulebook_updates" in turn else turn.get("rulebook_updates_json", [])
        rbu_json = rbu if isinstance(rbu, str) else json.dumps(rbu or [], default=str)
        bru = turn.get("bug_reports") if "bug_reports" in turn else turn.get("bug_reports_json", [])
        bru_json = bru if isinstance(bru, str) else json.dumps(bru or [], default=str)
        impu = turn.get("impression_updates") if "impression_updates" in turn else turn.get("impression_updates_json", [])
        impu_json = impu if isinstance(impu, str) else json.dumps(impu or [], default=str)

        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO playtest_turns (
                    turn_id, session_id, turn_index, sim_tick_start, sim_tick_end,
                    video_timestamp_sec, cursor_start_x, cursor_start_y,
                    cursor_end_x, cursor_end_y, current_plan, deliberation,
                    action_json, field_note_updates_json, question_updates_json,
                    rulebook_updates_json, bug_reports_json, impression_updates_json, append_log_line,
                    mood_state, mood_note, screenshot_path, audio_clip_path,
                    prompt_tokens, cached_tokens, output_tokens, thought_tokens,
                    created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    str(turn["session_id"]),
                    int(turn["turn_index"]),
                    int(turn.get("sim_tick_start", 0) or 0),
                    int(turn.get("sim_tick_end", 0) or 0),
                    float(turn.get("video_timestamp_sec", 0.0) or 0.0),
                    float(turn.get("cursor_start_x", 640.0)),
                    float(turn.get("cursor_start_y", 360.0)),
                    float(turn.get("cursor_end_x", 640.0)),
                    float(turn.get("cursor_end_y", 360.0)),
                    str(turn.get("current_plan") or ""),
                    str(turn.get("deliberation") or ""),
                    action_json,
                    fnu_json,
                    qu_json,
                    rbu_json,
                    bru_json,
                    impu_json,
                    str(turn.get("append_log_line") or ""),
                    turn.get("mood_state"),
                    turn.get("mood_note"),
                    turn.get("screenshot_path"),
                    turn.get("audio_clip_path"),
                    int(turn.get("prompt_tokens", 0) or 0),
                    int(turn.get("cached_tokens", 0) or 0),
                    int(turn.get("output_tokens", 0) or 0),
                    int(turn.get("thought_tokens", 0) or 0),
                    str(turn.get("created_at_utc") or now_iso),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def list_playtest_turns(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM playtest_turns WHERE session_id = ? ORDER BY turn_index ASC;",
                (session_id,),
            )
            rows: List[Dict[str, Any]] = []
            for r in cur.fetchall():
                d = dict(r)
                for jcol, ocol, fb in (
                    ("action_json", "action", {}),
                    ("field_note_updates_json", "field_note_updates", []),
                    ("question_updates_json", "question_updates", []),
                    ("rulebook_updates_json", "rulebook_updates", []),
                    ("bug_reports_json", "bug_reports", []),
                    ("impression_updates_json", "impression_updates", []),
                ):
                    raw = d.get(jcol)
                    try:
                        d[ocol] = json.loads(raw) if isinstance(raw, str) and raw else fb
                    except Exception:
                        d[ocol] = fb
                rows.append(d)
            return rows
        finally:
            conn.close()

    def insert_playtest_compaction(self, rec: Dict[str, Any]) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        comp_id = str(rec.get("compaction_id") or f"pt_comp_{uuid.uuid4().hex[:12]}")
        rb = rec.get("rulebook_snapshot") if "rulebook_snapshot" in rec else rec.get("rulebook_snapshot_json", {})
        rb_json = rb if isinstance(rb, str) else json.dumps(rb or {}, default=str)
        jr = rec.get("journal") if "journal" in rec else rec.get("journal_json", {})
        jr_json = jr.model_dump_json() if hasattr(jr, "model_dump") else (jr if isinstance(jr, str) else json.dumps(jr or {}, default=str))

        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO playtest_compactions (
                    compaction_id, session_id, compaction_index, covered_up_to_turn,
                    carried_tail_turns, deliberation, rulebook_snapshot_json,
                    journal_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comp_id,
                    str(rec["session_id"]),
                    int(rec["compaction_index"]),
                    int(rec["covered_up_to_turn"]),
                    int(rec.get("carried_tail_turns", 5)),
                    str(rec.get("deliberation") or ""),
                    rb_json,
                    jr_json,
                    str(rec.get("created_at_utc") or now_iso),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def list_playtest_compactions(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM playtest_compactions WHERE session_id = ? ORDER BY compaction_index ASC;",
                (session_id,),
            )
            out: List[Dict[str, Any]] = []
            for r in cur.fetchall():
                d = dict(r)
                for jcol, ocol in (("rulebook_snapshot_json", "rulebook_snapshot"), ("journal_json", "journal")):
                    raw = d.get(jcol)
                    try:
                        d[ocol] = json.loads(raw) if isinstance(raw, str) and raw else {}
                    except Exception:
                        d[ocol] = {}
                out.append(d)
            return out
        finally:
            conn.close()

    def insert_playtest_wish(self, wish: Dict[str, Any]) -> str:
        now_iso = datetime.now(timezone.utc).isoformat()
        wish_id = str(wish.get("wish_id") or f"wish_{uuid.uuid4().hex[:10]}")
        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO playtest_wishes (
                    wish_id, session_id, turn_index, candidate_id, checkpoint_id,
                    observed_on_slice_id, status, category, summary, rationale,
                    mood_at_time, screenshot_path, video_timestamp_sec,
                    producer_note, sent_at_utc, created_at_utc, deliberation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wish_id,
                    str(wish["session_id"]),
                    int(wish.get("turn_index", 0) or 0),
                    wish.get("candidate_id"),
                    str(wish.get("checkpoint_id") or "active"),
                    wish.get("observed_on_slice_id"),
                    str(wish.get("status") or "pending"),
                    str(wish.get("category") or "gameplay_feel"),
                    str(wish.get("summary") or ""),
                    str(wish.get("rationale") or ""),
                    wish.get("mood_at_time"),
                    wish.get("screenshot_path"),
                    float(wish.get("video_timestamp_sec", 0.0) or 0.0),
                    wish.get("producer_note"),
                    wish.get("sent_at_utc"),
                    str(wish.get("created_at_utc") or now_iso),
                    str(wish.get("deliberation") or ""),
                ),
            )
            conn.commit()
            return wish_id
        finally:
            conn.close()

    def update_playtest_wish_status(
        self,
        wish_id: str,
        status: str,
        producer_note: Optional[str] = None,
        sent_at_utc: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                UPDATE playtest_wishes
                SET status = ?,
                    producer_note = COALESCE(?, producer_note),
                    sent_at_utc = COALESCE(?, sent_at_utc)
                WHERE wish_id = ?
                """,
                (status, producer_note, sent_at_utc, wish_id),
            )
            conn.commit()
        finally:
            conn.close()
        matches = self.get_playtest_wishes_by_ids([wish_id])
        return matches[0] if matches else None

    def get_playtest_wishes_by_ids(self, wish_ids: List[str]) -> List[Dict[str, Any]]:
        clean_ids = [str(w).strip() for w in (wish_ids or []) if str(w).strip()]
        if not clean_ids:
            return []
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            placeholders = ",".join("?" for _ in clean_ids)
            cur.execute(
                f"SELECT * FROM playtest_wishes WHERE wish_id IN ({placeholders}) ORDER BY created_at_utc ASC;",
                clean_ids,
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def list_playtest_wishes(
        self,
        session_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            clauses: List[str] = []
            params: List[Any] = []
            if session_id:
                clauses.append("session_id = ?")
                params.append(session_id)
            if candidate_id:
                clauses.append("candidate_id = ?")
                params.append(candidate_id)
            if status:
                clauses.append("status = ?")
                params.append(status)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            params.append(max(1, int(limit)))
            cur.execute(
                f"SELECT * FROM playtest_wishes {where_sql} ORDER BY created_at_utc DESC LIMIT ?;",
                params,
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def reset_playtest_session_data(
        self,
        session_id: str,
        deliberation: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Wipes all turns, compactions, wishes, rulebook, impressions, report, and media references
        for `session_id` while leaving the session row in `status='idle'` without starting it.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        delib_text = deliberation or (
            f"Operator reset playtest session {session_id} data to a clean idle state without starting."
        )
        conn = self.get_write_connection()
        try:
            conn.execute("DELETE FROM playtest_turns WHERE session_id = ?;", (session_id,))
            conn.execute("DELETE FROM playtest_compactions WHERE session_id = ?;", (session_id,))
            conn.execute("DELETE FROM playtest_wishes WHERE session_id = ?;", (session_id,))
            conn.execute(
                """
                UPDATE playtest_sessions
                SET status = 'idle',
                    turn_count = 0,
                    sim_ticks_total = 0,
                    sim_duration_sec = 0.0,
                    compaction_count = 0,
                    prompt_tokens = 0,
                    cached_tokens = 0,
                    output_tokens = 0,
                    thought_tokens = 0,
                    rulebook_json = '{}',
                    field_notes_json = '{}',
                    questions_json = '{}',
                    bugs_found_json = '{}',
                    impressions_json = '{}',
                    session_report_json = '{}',
                    compacted_journal_json = '{}',
                    current_plan = 'Session data reset — click Run Playtester to start.',
                    current_mood = 'idle',
                    video_path = NULL,
                    latest_frame_path = NULL,
                    end_outcome = NULL,
                    end_summary = NULL,
                    error_message = NULL,
                    ended_at_utc = NULL,
                    updated_at_utc = ?,
                    deliberation = ?
                WHERE session_id = ?;
                """,
                (now_iso, delib_text, session_id),
            )
            conn.commit()
        finally:
            conn.close()
        return self.get_playtest_session(session_id)

    def delete_playtest_session(self, session_id: str) -> None:
        """Atomically deletes a playtest session and all of its child turns, compactions, and wishes."""
        conn = self.get_write_connection()
        try:
            conn.execute("DELETE FROM playtest_turns WHERE session_id = ?;", (session_id,))
            conn.execute("DELETE FROM playtest_compactions WHERE session_id = ?;", (session_id,))
            conn.execute("DELETE FROM playtest_wishes WHERE session_id = ?;", (session_id,))
            conn.execute("DELETE FROM playtest_sessions WHERE session_id = ?;", (session_id,))
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Per-Game Multi-Bank Long-Term Memory Persistence
    # -------------------------------------------------------------------------

    def upsert_game_memory(
        self,
        cartridge_id: str,
        entry: Dict[str, Any],
        game_title: str = "",
    ) -> None:
        """Upserts a single long-term game memory entry into studio_game_memories."""
        cid = (cartridge_id or "").strip()
        mid = str(entry.get("memory_id") or "").strip()
        if not cid or not mid:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        roles = entry.get("applicable_roles")
        if isinstance(roles, list):
            roles_json = json.dumps([str(r) for r in roles if r])
        else:
            roles_json = "[]"

        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT INTO studio_game_memories (
                    cartridge_id, memory_id, deliberation, game_title, memory_bank,
                    author_role, kind, status, title, content, applicable_roles_json,
                    slice_id, item_id, resolution_note, superseded_by_id,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cartridge_id, memory_id) DO UPDATE SET
                    deliberation = excluded.deliberation,
                    game_title = CASE WHEN excluded.game_title != '' THEN excluded.game_title ELSE studio_game_memories.game_title END,
                    memory_bank = excluded.memory_bank,
                    author_role = excluded.author_role,
                    kind = excluded.kind,
                    status = excluded.status,
                    title = excluded.title,
                    content = excluded.content,
                    applicable_roles_json = excluded.applicable_roles_json,
                    slice_id = excluded.slice_id,
                    item_id = excluded.item_id,
                    resolution_note = excluded.resolution_note,
                    superseded_by_id = excluded.superseded_by_id,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    cid,
                    mid,
                    str(entry.get("deliberation") or "Durable game memory persisted."),
                    str(game_title or ""),
                    str(entry.get("memory_bank") or "council"),
                    str(entry.get("author_role") or "council"),
                    str(entry.get("kind") or "lesson_learned"),
                    str(entry.get("status") or "active"),
                    str(entry.get("title") or ""),
                    str(entry.get("content") or ""),
                    roles_json,
                    str(entry.get("slice_id") or ""),
                    str(entry.get("item_id") or ""),
                    str(entry.get("resolution_note") or ""),
                    str(entry.get("superseded_by_id") or ""),
                    str(entry.get("created_at_utc") or now_iso),
                    str(entry.get("updated_at_utc") or now_iso),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def sync_game_memories(
        self,
        cartridge_id: str,
        entries: List[Dict[str, Any]],
        game_title: str = "",
    ) -> None:
        """Atomically mirrors the authoritative `game_memory.json` entries for `cartridge_id` into SQLite WAL."""
        cid = (cartridge_id or "").strip()
        if not cid:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = self.get_write_connection()
        try:
            keep_ids: List[str] = []
            for entry in entries:
                mid = str(entry.get("memory_id") or "").strip()
                if not mid:
                    continue
                keep_ids.append(mid)
                roles = entry.get("applicable_roles")
                roles_json = json.dumps([str(r) for r in roles if r]) if isinstance(roles, list) else "[]"
                conn.execute(
                    """
                    INSERT INTO studio_game_memories (
                        cartridge_id, memory_id, deliberation, game_title, memory_bank,
                        author_role, kind, status, title, content, applicable_roles_json,
                        slice_id, item_id, resolution_note, superseded_by_id,
                        created_at_utc, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cartridge_id, memory_id) DO UPDATE SET
                        deliberation = excluded.deliberation,
                        game_title = CASE WHEN excluded.game_title != '' THEN excluded.game_title ELSE studio_game_memories.game_title END,
                        memory_bank = excluded.memory_bank,
                        author_role = excluded.author_role,
                        kind = excluded.kind,
                        status = excluded.status,
                        title = excluded.title,
                        content = excluded.content,
                        applicable_roles_json = excluded.applicable_roles_json,
                        slice_id = excluded.slice_id,
                        item_id = excluded.item_id,
                        resolution_note = excluded.resolution_note,
                        superseded_by_id = excluded.superseded_by_id,
                        updated_at_utc = excluded.updated_at_utc
                    """,
                    (
                        cid,
                        mid,
                        str(entry.get("deliberation") or "Durable game memory persisted."),
                        str(game_title or ""),
                        str(entry.get("memory_bank") or "council"),
                        str(entry.get("author_role") or "council"),
                        str(entry.get("kind") or "lesson_learned"),
                        str(entry.get("status") or "active"),
                        str(entry.get("title") or ""),
                        str(entry.get("content") or ""),
                        roles_json,
                        str(entry.get("slice_id") or ""),
                        str(entry.get("item_id") or ""),
                        str(entry.get("resolution_note") or ""),
                        str(entry.get("superseded_by_id") or ""),
                        str(entry.get("created_at_utc") or now_iso),
                        str(entry.get("updated_at_utc") or now_iso),
                    ),
                )
            if keep_ids:
                placeholders = ",".join("?" for _ in keep_ids)
                conn.execute(
                    f"DELETE FROM studio_game_memories WHERE cartridge_id = ? AND memory_id NOT IN ({placeholders});",
                    [cid, *keep_ids],
                )
            else:
                conn.execute("DELETE FROM studio_game_memories WHERE cartridge_id = ?;", (cid,))
            conn.commit()
        finally:
            conn.close()

    def list_game_memories(
        self,
        cartridge_id: str,
        memory_bank: Optional[str] = None,
        kind: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lists long-term game memory entries for `cartridge_id` ordered rationale-first."""
        cid = (cartridge_id or "").strip()
        if not cid:
            return []
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            clauses = ["cartridge_id = ?"]
            params: List[Any] = [cid]
            if memory_bank and memory_bank != "all":
                clauses.append("memory_bank = ?")
                params.append(memory_bank)
            if kind and kind != "all":
                clauses.append("kind = ?")
                params.append(kind)
            if status and status != "all":
                clauses.append("status = ?")
                params.append(status)
            where_sql = " AND ".join(clauses)
            cur.execute(
                f"SELECT * FROM studio_game_memories WHERE {where_sql} ORDER BY updated_at_utc DESC;",
                params,
            )
            rows: List[Dict[str, Any]] = []
            for raw in cur.fetchall():
                d = dict(raw)
                roles_raw = d.pop("applicable_roles_json", "[]")
                try:
                    roles = json.loads(roles_raw) if roles_raw else []
                except Exception:
                    roles = []
                rows.append(
                    {
                        "deliberation": d.get("deliberation") or "Durable game memory persisted.",
                        "memory_id": d.get("memory_id", ""),
                        "memory_bank": d.get("memory_bank", "council"),
                        "author_role": d.get("author_role", "council"),
                        "kind": d.get("kind", "lesson_learned"),
                        "status": d.get("status", "active"),
                        "title": d.get("title", ""),
                        "content": d.get("content", ""),
                        "applicable_roles": roles if isinstance(roles, list) else [],
                        "slice_id": d.get("slice_id") or "",
                        "item_id": d.get("item_id") or "",
                        "resolution_note": d.get("resolution_note") or "",
                        "superseded_by_id": d.get("superseded_by_id") or "",
                        "created_at_utc": d.get("created_at_utc", ""),
                        "updated_at_utc": d.get("updated_at_utc", ""),
                    }
                )
            return rows
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Interactive Studio Co-Producer Thread & Turn Persistence
    # -------------------------------------------------------------------------

    @staticmethod
    def _decode_json_field(raw: Any, fallback: Any) -> Any:
        if isinstance(raw, (dict, list)):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                return json.loads(raw)
            except Exception:
                return fallback
        return fallback

    def _row_to_coproducer_thread(self, raw: sqlite3.Row | Dict[str, Any], message_count: Optional[int] = None) -> Dict[str, Any]:
        d = dict(raw)
        brief = self._decode_json_field(d.get("concept_brief_json"), {})
        if not isinstance(brief, dict):
            brief = {}
        out: Dict[str, Any] = {
            "deliberation": str(d.get("deliberation") or "Co-Producer conversation thread metadata loaded."),
            "thread_id": str(d.get("thread_id") or ""),
            "scope_kind": str(d.get("scope_kind") or "cartridge"),
            "cartridge_id": d.get("cartridge_id") or None,
            "title": str(d.get("title") or "New Game Concept"),
            "graduated_to_cartridge_id": str(d.get("graduated_to_cartridge_id") or ""),
            "concept_brief": brief,
            "created_at_utc": str(d.get("created_at_utc") or ""),
            "updated_at_utc": str(d.get("updated_at_utc") or ""),
        }
        if message_count is not None:
            out["message_count"] = int(message_count)
        elif "message_count" in d and d["message_count"] is not None:
            out["message_count"] = int(d["message_count"])
        return out

    def _row_to_coproducer_message(self, raw: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
        d = dict(raw)
        media_cards = self._decode_json_field(d.get("media_cards_json"), [])
        consultation_cards = self._decode_json_field(d.get("consultation_cards_json"), [])
        action_proposals = self._decode_json_field(d.get("action_proposals_json"), [])
        tool_trace = self._decode_json_field(d.get("tool_trace_json"), [])
        prompt_trace = self._decode_json_field(d.get("prompt_trace_json"), [])
        token_usage = self._decode_json_field(d.get("token_usage_json"), {})
        return {
            "deliberation": str(d.get("deliberation") or "Co-Producer conversation message loaded."),
            "message_id": str(d.get("message_id") or ""),
            "thread_id": str(d.get("thread_id") or ""),
            "turn_index": int(d.get("turn_index") or 0),
            "sender_role": str(d.get("sender_role") or "coproducer"),
            "content_markdown": str(d.get("content_markdown") or ""),
            "media_cards": media_cards if isinstance(media_cards, list) else [],
            "consultation_cards": consultation_cards if isinstance(consultation_cards, list) else [],
            "action_proposals": action_proposals if isinstance(action_proposals, list) else [],
            "tool_trace": tool_trace if isinstance(tool_trace, list) else [],
            "prompt_trace": prompt_trace if isinstance(prompt_trace, list) else [],
            "token_usage": token_usage if isinstance(token_usage, dict) else {},
            "created_at_utc": str(d.get("created_at_utc") or ""),
        }

    def upsert_coproducer_thread(self, thread: Dict[str, Any]) -> None:
        """Upserts a Co-Producer conversation thread into studio_coproducer_threads."""
        tid = str(thread.get("thread_id") or "").strip()
        if not tid:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        brief = thread.get("concept_brief")
        if isinstance(brief, dict):
            brief_json = json.dumps(brief, default=str)
        elif isinstance(thread.get("concept_brief_json"), str):
            brief_json = str(thread.get("concept_brief_json"))
        else:
            brief_json = "{}"
        cartridge_id = thread.get("cartridge_id")
        cid_val = str(cartridge_id).strip() if cartridge_id else None

        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT INTO studio_coproducer_threads (
                    thread_id, deliberation, scope_kind, cartridge_id, title,
                    graduated_to_cartridge_id, concept_brief_json, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    deliberation = excluded.deliberation,
                    scope_kind = excluded.scope_kind,
                    cartridge_id = excluded.cartridge_id,
                    title = excluded.title,
                    graduated_to_cartridge_id = excluded.graduated_to_cartridge_id,
                    concept_brief_json = excluded.concept_brief_json,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    tid,
                    str(thread.get("deliberation") or "Persisting Co-Producer conversation thread state."),
                    str(thread.get("scope_kind") or ("cartridge" if cid_val else "draft")),
                    cid_val,
                    str(thread.get("title") or "New Game Concept"),
                    str(thread.get("graduated_to_cartridge_id") or ""),
                    brief_json,
                    str(thread.get("created_at_utc") or now_iso),
                    str(thread.get("updated_at_utc") or now_iso),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_coproducer_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a single Co-Producer thread record by thread_id."""
        tid = (thread_id or "").strip()
        if not tid:
            return None
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT t.*, (
                    SELECT COUNT(*) FROM studio_coproducer_messages m WHERE m.thread_id = t.thread_id
                ) AS message_count
                FROM studio_coproducer_threads t
                WHERE t.thread_id = ?;
                """,
                (tid,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_coproducer_thread(row)
        finally:
            conn.close()

    def list_coproducer_threads(
        self,
        cartridge_id: Optional[str] = None,
        scope_kind: Optional[str] = None,
        include_graduated_drafts: bool = False,
    ) -> List[Dict[str, Any]]:
        """Lists Co-Producer threads ordered by updated_at_utc DESC."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            clauses: List[str] = []
            params: List[Any] = []
            if cartridge_id:
                clauses.append("t.cartridge_id = ?")
                params.append(cartridge_id.strip())
            if scope_kind:
                clauses.append("t.scope_kind = ?")
                params.append(scope_kind.strip())
            if not include_graduated_drafts:
                clauses.append("NOT (t.scope_kind = 'draft' AND COALESCE(t.graduated_to_cartridge_id, '') != '')")
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            cur.execute(
                f"""
                SELECT t.*, (
                    SELECT COUNT(*) FROM studio_coproducer_messages m WHERE m.thread_id = t.thread_id
                ) AS message_count
                FROM studio_coproducer_threads t
                {where_sql}
                ORDER BY t.updated_at_utc DESC;
                """,
                params,
            )
            return [self._row_to_coproducer_thread(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def bind_draft_thread_to_cartridge(
        self,
        draft_thread_id: str,
        cartridge_id: str,
        title: Optional[str] = None,
        concept_brief: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Graduates a pre-genesis draft thread directly onto a commissioned cartridge_id."""
        tid = (draft_thread_id or "").strip()
        cid = (cartridge_id or "").strip()
        if not tid or not cid:
            return None
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = self.get_write_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM studio_coproducer_threads WHERE thread_id = ?;", (tid,))
            existing = cur.fetchone()
            if not existing:
                return None
            cols = [desc[0] for desc in cur.description]
            row_dict = dict(zip(cols, existing))
            next_title = str(title or row_dict.get("title") or cid)
            if isinstance(concept_brief, dict):
                brief_json = json.dumps(concept_brief, default=str)
            else:
                brief_json = str(row_dict.get("concept_brief_json") or "{}")
            conn.execute(
                """
                UPDATE studio_coproducer_threads
                SET scope_kind = 'cartridge',
                    cartridge_id = ?,
                    graduated_to_cartridge_id = ?,
                    title = ?,
                    concept_brief_json = ?,
                    deliberation = ?,
                    updated_at_utc = ?
                WHERE thread_id = ?;
                """,
                (
                    cid,
                    cid,
                    next_title,
                    brief_json,
                    f"Graduated draft thread {tid} into active cartridge {cid}.",
                    now_iso,
                    tid,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return self.get_coproducer_thread(tid)

    def insert_coproducer_message(self, message: Dict[str, Any]) -> None:
        """Inserts or updates a Co-Producer conversation message and bumps parent thread updated_at_utc."""
        mid = str(message.get("message_id") or "").strip()
        tid = str(message.get("thread_id") or "").strip()
        if not mid or not tid:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        created_iso = str(message.get("created_at_utc") or now_iso)
        media_json = json.dumps(message.get("media_cards") or [], default=str)
        consult_json = json.dumps(message.get("consultation_cards") or [], default=str)
        actions_json = json.dumps(message.get("action_proposals") or [], default=str)
        tools_json = json.dumps(message.get("tool_trace") or [], default=str)
        prompts_json = json.dumps(message.get("prompt_trace") or [], default=str)
        tokens_json = json.dumps(message.get("token_usage") or {}, default=str)

        conn = self.get_write_connection()
        try:
            conn.execute(
                """
                INSERT INTO studio_coproducer_messages (
                    message_id, thread_id, turn_index, sender_role, deliberation,
                    content_markdown, media_cards_json, consultation_cards_json,
                    action_proposals_json, tool_trace_json, prompt_trace_json,
                    token_usage_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    turn_index = excluded.turn_index,
                    sender_role = excluded.sender_role,
                    deliberation = excluded.deliberation,
                    content_markdown = excluded.content_markdown,
                    media_cards_json = excluded.media_cards_json,
                    consultation_cards_json = excluded.consultation_cards_json,
                    action_proposals_json = excluded.action_proposals_json,
                    tool_trace_json = excluded.tool_trace_json,
                    prompt_trace_json = excluded.prompt_trace_json,
                    token_usage_json = excluded.token_usage_json
                """,
                (
                    mid,
                    tid,
                    int(message.get("turn_index") or 0),
                    str(message.get("sender_role") or "coproducer"),
                    str(message.get("deliberation") or "Persisting Co-Producer message."),
                    str(message.get("content_markdown") or ""),
                    media_json,
                    consult_json,
                    actions_json,
                    tools_json,
                    prompts_json,
                    tokens_json,
                    created_iso,
                ),
            )
            conn.execute(
                "UPDATE studio_coproducer_threads SET updated_at_utc = ? WHERE thread_id = ?;",
                (now_iso, tid),
            )
            conn.commit()
        finally:
            conn.close()

    def update_coproducer_message_action_state(
        self,
        thread_id: str,
        message_id: str,
        proposal_id: str,
        status: str,
        result_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Updates the status and execution metadata of an action proposal inside a persisted message."""
        tid = (thread_id or "").strip()
        mid = (message_id or "").strip()
        pid = (proposal_id or "").strip()
        if not tid or not mid or not pid:
            return None
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = self.get_write_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM studio_coproducer_messages WHERE thread_id = ? AND message_id = ?;",
                (tid, mid),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            row_dict = dict(zip(cols, row))
            proposals = self._decode_json_field(row_dict.get("action_proposals_json"), [])
            if not isinstance(proposals, list):
                proposals = []
            updated_proposal: Optional[Dict[str, Any]] = None
            for item in proposals:
                if isinstance(item, dict) and str(item.get("proposal_id") or "") == pid:
                    item["status"] = status
                    item["executed_at_utc"] = now_iso
                    if isinstance(result_meta, dict):
                        item["execution_result"] = result_meta
                    updated_proposal = item
                    break
            if updated_proposal is None:
                return None
            conn.execute(
                "UPDATE studio_coproducer_messages SET action_proposals_json = ? WHERE message_id = ?;",
                (json.dumps(proposals, default=str), mid),
            )
            conn.execute(
                "UPDATE studio_coproducer_threads SET updated_at_utc = ? WHERE thread_id = ?;",
                (now_iso, tid),
            )
            conn.commit()
            return updated_proposal
        finally:
            conn.close()

    def list_coproducer_messages(self, thread_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Lists messages in a Co-Producer thread in chronological turn order."""
        tid = (thread_id or "").strip()
        if not tid:
            return []
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT * FROM studio_coproducer_messages
                WHERE thread_id = ?
                ORDER BY turn_index ASC, created_at_utc ASC
                LIMIT ?;
                """,
                (tid, max(1, int(limit))),
            )
            return [self._row_to_coproducer_message(r) for r in cur.fetchall()]
        finally:
            conn.close()
