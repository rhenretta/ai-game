"""engine/telemetry/db.py
SQLite connection manager, WAL mode configuration, and pragmas.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger("TelemetryDB")


class TelemetryDB:
    """Manages SQLite database connections, schemas, and WAL pragmas."""

    def __init__(self, db_path: str | Path = "data/telemetry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
        """Returns studio swarm events, optionally filtered by build_id."""
        conn = self.get_reader_connection()
        try:
            cur = conn.cursor()
            query = "SELECT * FROM studio_events"
            params: List[Any] = []
            if build_id:
                query += " WHERE build_id = ?"
                params.append(build_id)
            query += " ORDER BY timestamp_utc ASC"
            if limit and limit > 0:
                query += " LIMIT ?"
                params.append(limit)
            cur.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                if "payload_json" in r and r["payload_json"]:
                    try:
                        r["payload"] = json.loads(r["payload_json"])
                    except Exception:
                        r["payload"] = {}
                else:
                    r["payload"] = {}
            return rows
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
        """Records requirement tracks from a GameRequirementsSpec into studio_requirements."""
        if hasattr(spec, "model_dump"):
            spec_data = spec.model_dump()
        elif isinstance(spec, dict):
            spec_data = spec
        else:
            return

        requirements = spec_data.get("requirements", [])
        now_utc = datetime.now(timezone.utc).isoformat()
        conn = self.get_write_connection()
        try:
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
        """Updates the status of a specific requirement in studio_requirements."""
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


