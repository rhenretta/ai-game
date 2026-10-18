"""engine/telemetry/schema.py
Relational SQLite WAL schema and DDL definitions for telemetry and studio observability.
Replaces flat JSON/JSONL studio event and task logs with normalized WAL tables:
- studio_builds
- studio_events
- studio_tasks
- studio_artifacts
- studio_requirements
- studio_checkpoints
"""
from __future__ import annotations
import sqlite3
from typing import Optional

STUDIO_BUILDS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS studio_builds (
    build_id TEXT PRIMARY KEY,
    candidate_id TEXT,
    build_type TEXT DEFAULT 'genesis',
    engine_type TEXT DEFAULT 'godot4',
    game_title TEXT,
    genre_paradigm TEXT,
    status TEXT DEFAULT 'STARTED',
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    deliberation TEXT
);
"""

STUDIO_EVENTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS studio_events (
    event_id TEXT PRIMARY KEY,
    build_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    deliberation TEXT,
    payload_json TEXT DEFAULT '{}',
    FOREIGN KEY (build_id) REFERENCES studio_builds(build_id)
);
"""

STUDIO_TASKS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS studio_tasks (
    task_id TEXT PRIMARY KEY,
    build_id TEXT NOT NULL,
    seq_order INTEGER DEFAULT 0,
    agent_role TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    requirement_id TEXT,
    track_title TEXT,
    task_type TEXT,
    status TEXT,
    start_time_utc TEXT,
    end_time_utc TEXT,
    duration_ms REAL DEFAULT 0.0,
    deliberation TEXT,
    verification_verdict_json TEXT DEFAULT '{}',
    diagnostic_trace_json TEXT DEFAULT '{}',
    summary TEXT,
    FOREIGN KEY (build_id) REFERENCES studio_builds(build_id)
);
"""

STUDIO_ARTIFACTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS studio_artifacts (
    artifact_id TEXT PRIMARY KEY,
    task_id TEXT,
    build_id TEXT NOT NULL,
    direction TEXT,
    artifact_name TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    sha256_hash TEXT,
    byte_size INTEGER DEFAULT 0,
    source_agent TEXT,
    target_agent TEXT,
    deliberation TEXT,
    created_at_utc TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    FOREIGN KEY (build_id) REFERENCES studio_builds(build_id)
);
"""

STUDIO_REQUIREMENTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS studio_requirements (
    build_id TEXT NOT NULL,
    requirement_id TEXT NOT NULL,
    seq_order INTEGER DEFAULT 0,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'pending',
    specialist_owner TEXT,
    target_modules_json TEXT DEFAULT '[]',
    acceptance_criteria_json TEXT DEFAULT '[]',
    deliberation TEXT,
    updated_at_utc TEXT,
    PRIMARY KEY (build_id, requirement_id),
    FOREIGN KEY (build_id) REFERENCES studio_builds(build_id)
);
"""

STUDIO_CHECKPOINTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS studio_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    build_id TEXT NOT NULL,
    candidate_id TEXT,
    requirement_id TEXT,
    track_title TEXT,
    engine_type TEXT DEFAULT 'godot4',
    checkpoint_path TEXT,
    verified_tracks_json TEXT DEFAULT '[]',
    created_at_utc TEXT NOT NULL,
    deliberation TEXT,
    FOREIGN KEY (build_id) REFERENCES studio_builds(build_id)
);
"""

STUDIO_REQUIREMENT_SPECS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS studio_requirement_specs (
    build_id TEXT PRIMARY KEY,
    spec_id TEXT,
    game_title TEXT,
    genre_paradigm TEXT,
    spec_json TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    FOREIGN KEY (build_id) REFERENCES studio_builds(build_id)
);
"""

STUDIO_GAME_BIBLES_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS studio_game_bibles (
    bible_id TEXT PRIMARY KEY,
    cartridge_id TEXT NOT NULL,
    target_episode INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'brainstorming',
    game_title TEXT NOT NULL,
    bible_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);
"""

STUDIO_INDICES_DDL = """
CREATE INDEX IF NOT EXISTS idx_studio_events_build_id ON studio_events(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_events_type ON studio_events(event_type);
CREATE INDEX IF NOT EXISTS idx_studio_tasks_build_id ON studio_tasks(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_tasks_agent ON studio_tasks(agent_role);
CREATE INDEX IF NOT EXISTS idx_studio_artifacts_build_id ON studio_artifacts(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_artifacts_task_id ON studio_artifacts(task_id);
CREATE INDEX IF NOT EXISTS idx_studio_requirements_build_id ON studio_requirements(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_checkpoints_build_id ON studio_checkpoints(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_game_bibles_updated ON studio_game_bibles(updated_at_utc);
"""

ALL_STUDIO_DDL = [
    STUDIO_BUILDS_TABLE_DDL,
    STUDIO_EVENTS_TABLE_DDL,
    STUDIO_TASKS_TABLE_DDL,
    STUDIO_ARTIFACTS_TABLE_DDL,
    STUDIO_REQUIREMENTS_TABLE_DDL,
    STUDIO_REQUIREMENT_SPECS_TABLE_DDL,
    STUDIO_CHECKPOINTS_TABLE_DDL,
    STUDIO_GAME_BIBLES_TABLE_DDL,
    STUDIO_INDICES_DDL,
]


def init_studio_schema(conn: sqlite3.Connection) -> None:
    """Initializes the normalized relational studio tables and indices."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA foreign_keys = ON;")
    for ddl in ALL_STUDIO_DDL:
        cursor.executescript(ddl)
    conn.commit()
