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

    def get_reader_connection(self) -> sqlite3.Connection:
        """Isolated read-only connection for Analyst and MCP server."""
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA query_only = ON;")
        return conn

    def checkpoint(self, mode: str = "PASSIVE", timeout: float = 5.0) -> None:
        """Triggers WAL checkpoint (PASSIVE, FULL, RESTART, TRUNCATE)."""
        timeout_sec = max(0.05, float(timeout))
        conn = sqlite3.connect(str(self.db_path), timeout=timeout_sec)
        try:
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
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        try:
            conn.execute("PRAGMA busy_timeout = 5000;")
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

