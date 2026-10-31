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

PLAYTEST_SESSIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS playtest_sessions (
    session_id TEXT PRIMARY KEY,
    checkpoint_id TEXT NOT NULL,
    candidate_id TEXT,
    slice_id TEXT,
    initiated_by TEXT NOT NULL DEFAULT 'operator',
    game_title TEXT NOT NULL,
    game_description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    end_outcome TEXT,
    end_summary TEXT,
    error_message TEXT,
    turn_count INTEGER DEFAULT 0,
    sim_ticks_total INTEGER DEFAULT 0,
    sim_duration_sec REAL DEFAULT 0.0,
    compaction_count INTEGER DEFAULT 0,
    rulebook_json TEXT DEFAULT '{}',
    field_notes_json TEXT DEFAULT '{}',
    questions_json TEXT DEFAULT '{}',
    bugs_found_json TEXT DEFAULT '{}',
    impressions_json TEXT DEFAULT '{}',
    session_report_json TEXT DEFAULT '{}',
    compacted_journal_json TEXT DEFAULT '{}',
    current_plan TEXT DEFAULT '',
    current_mood TEXT DEFAULT '',
    prompt_tokens INTEGER DEFAULT 0,
    cached_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    thought_tokens INTEGER DEFAULT 0,
    video_path TEXT,
    latest_frame_path TEXT,
    started_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    ended_at_utc TEXT,
    deliberation TEXT
);
"""

PLAYTEST_TURNS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS playtest_turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    sim_tick_start INTEGER DEFAULT 0,
    sim_tick_end INTEGER DEFAULT 0,
    video_timestamp_sec REAL DEFAULT 0.0,
    cursor_start_x REAL DEFAULT 640.0,
    cursor_start_y REAL DEFAULT 360.0,
    cursor_end_x REAL DEFAULT 640.0,
    cursor_end_y REAL DEFAULT 360.0,
    current_plan TEXT DEFAULT '',
    deliberation TEXT NOT NULL,
    action_json TEXT NOT NULL DEFAULT '{}',
    field_note_updates_json TEXT DEFAULT '[]',
    question_updates_json TEXT DEFAULT '[]',
    rulebook_updates_json TEXT DEFAULT '[]',
    bug_reports_json TEXT DEFAULT '[]',
    impression_updates_json TEXT DEFAULT '[]',
    append_log_line TEXT NOT NULL DEFAULT '',
    mood_state TEXT,
    mood_note TEXT,
    screenshot_path TEXT,
    audio_clip_path TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    cached_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    thought_tokens INTEGER DEFAULT 0,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES playtest_sessions(session_id)
);
"""

PLAYTEST_COMPACTIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS playtest_compactions (
    compaction_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    compaction_index INTEGER NOT NULL,
    covered_up_to_turn INTEGER NOT NULL,
    carried_tail_turns INTEGER DEFAULT 5,
    deliberation TEXT NOT NULL,
    rulebook_snapshot_json TEXT NOT NULL DEFAULT '{}',
    journal_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES playtest_sessions(session_id)
);
"""

PLAYTEST_WISHES_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS playtest_wishes (
    wish_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    candidate_id TEXT,
    checkpoint_id TEXT NOT NULL,
    observed_on_slice_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    category TEXT NOT NULL,
    summary TEXT NOT NULL,
    rationale TEXT NOT NULL,
    mood_at_time TEXT,
    screenshot_path TEXT,
    video_timestamp_sec REAL DEFAULT 0.0,
    producer_note TEXT,
    sent_at_utc TEXT,
    created_at_utc TEXT NOT NULL,
    deliberation TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES playtest_sessions(session_id)
);
"""

PLAYTEST_INDICES_DDL = """
CREATE INDEX IF NOT EXISTS idx_playtest_sessions_checkpoint ON playtest_sessions(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_playtest_sessions_candidate ON playtest_sessions(candidate_id);
CREATE INDEX IF NOT EXISTS idx_playtest_sessions_started ON playtest_sessions(started_at_utc);
CREATE INDEX IF NOT EXISTS idx_playtest_turns_session ON playtest_turns(session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_playtest_compactions_session ON playtest_compactions(session_id, compaction_index);
CREATE INDEX IF NOT EXISTS idx_playtest_wishes_session ON playtest_wishes(session_id);
CREATE INDEX IF NOT EXISTS idx_playtest_wishes_candidate ON playtest_wishes(candidate_id, status);
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
    PLAYTEST_SESSIONS_TABLE_DDL,
    PLAYTEST_TURNS_TABLE_DDL,
    PLAYTEST_COMPACTIONS_TABLE_DDL,
    PLAYTEST_WISHES_TABLE_DDL,
    PLAYTEST_INDICES_DDL,
]


def init_studio_schema(conn: sqlite3.Connection) -> None:
    """Initializes the normalized relational studio tables and indices."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA foreign_keys = ON;")
    for ddl in ALL_STUDIO_DDL:
        cursor.executescript(ddl)

    # Idempotent migrations for existing playtest_sessions & playtest_turns tables
    cursor.execute("PRAGMA table_info(playtest_sessions);")
    sess_cols = {r[1] for r in cursor.fetchall()}
    if sess_cols:
        if "initiated_by" not in sess_cols:
            cursor.execute("ALTER TABLE playtest_sessions ADD COLUMN initiated_by TEXT NOT NULL DEFAULT 'operator';")
        if "impressions_json" not in sess_cols:
            cursor.execute("ALTER TABLE playtest_sessions ADD COLUMN impressions_json TEXT DEFAULT '{}';")
        if "field_notes_json" not in sess_cols:
            cursor.execute("ALTER TABLE playtest_sessions ADD COLUMN field_notes_json TEXT DEFAULT '{}';")
        if "questions_json" not in sess_cols:
            cursor.execute("ALTER TABLE playtest_sessions ADD COLUMN questions_json TEXT DEFAULT '{}';")
        if "bugs_found_json" not in sess_cols:
            cursor.execute("ALTER TABLE playtest_sessions ADD COLUMN bugs_found_json TEXT DEFAULT '{}';")
        if "session_report_json" not in sess_cols:
            cursor.execute("ALTER TABLE playtest_sessions ADD COLUMN session_report_json TEXT DEFAULT '{}';")

    cursor.execute("PRAGMA table_info(playtest_turns);")
    turn_cols = {r[1] for r in cursor.fetchall()}
    if turn_cols:
        if "impression_updates_json" not in turn_cols:
            cursor.execute("ALTER TABLE playtest_turns ADD COLUMN impression_updates_json TEXT DEFAULT '[]';")
        if "field_note_updates_json" not in turn_cols:
            cursor.execute("ALTER TABLE playtest_turns ADD COLUMN field_note_updates_json TEXT DEFAULT '[]';")
        if "question_updates_json" not in turn_cols:
            cursor.execute("ALTER TABLE playtest_turns ADD COLUMN question_updates_json TEXT DEFAULT '[]';")
        if "bug_reports_json" not in turn_cols:
            cursor.execute("ALTER TABLE playtest_turns ADD COLUMN bug_reports_json TEXT DEFAULT '[]';")

    conn.commit()
