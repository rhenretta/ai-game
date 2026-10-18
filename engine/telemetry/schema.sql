-- engine/telemetry/schema.sql
-- SQLite WAL schema for continuous zero-stutter telemetry streaming and dynamic probe catalog.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS telemetry_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    ended_at TEXT,
    player_profile_tag TEXT,
    platform TEXT DEFAULT 'macOS',
    resolution TEXT DEFAULT '1280x720',
    target_fps INTEGER DEFAULT 60,
    total_episodes INTEGER DEFAULT 0,
    metadata_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS telemetry_episodes (
    episode_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    episode_number INTEGER,
    status TEXT DEFAULT 'ACTIVE',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    duration_seconds REAL DEFAULT 0.0,
    outcome TEXT DEFAULT 'IN_PROGRESS',
    player_character TEXT DEFAULT 'hero',
    score INTEGER DEFAULT 0,
    damage_dealt_total REAL DEFAULT 0.0,
    damage_taken_total REAL DEFAULT 0.0,
    dashes_total INTEGER DEFAULT 0,
    shots_fired_total INTEGER DEFAULT 0,
    exit_x REAL,
    exit_y REAL,
    metadata_json TEXT DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES telemetry_sessions(session_id)
);

-- Dynamic Probe Metadata Catalog Table
CREATE TABLE IF NOT EXISTS telemetry_probes (
    probe_id TEXT PRIMARY KEY,
    probe_name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL, -- 'combat', 'movement', 'dialogue', 'ability', 'risk_reward', 'exploration'
    description TEXT NOT NULL,
    hypothesis_target TEXT NOT NULL,
    schema_json TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_probes_category ON telemetry_probes(category);
CREATE INDEX IF NOT EXISTS idx_probes_name ON telemetry_probes(probe_name);

-- Core Telemetry Events Table
CREATE TABLE IF NOT EXISTS telemetry_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL, -- Baseline enum OR 'CUSTOM_PROBE'
    probe_name TEXT,          -- Non-null when event_type == 'CUSTOM_PROBE'
    pos_x REAL NOT NULL,
    pos_y REAL NOT NULL,
    vel_x REAL DEFAULT 0.0,
    vel_y REAL DEFAULT 0.0,
    health REAL,
    payload_json TEXT DEFAULT '{}',
    frame_index INTEGER DEFAULT 0,
    timestamp_utc TEXT,
    FOREIGN KEY (session_id) REFERENCES telemetry_sessions(session_id),
    FOREIGN KEY (episode_id) REFERENCES telemetry_episodes(episode_id)
);

CREATE INDEX IF NOT EXISTS idx_events_session ON telemetry_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_episode ON telemetry_events(episode_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON telemetry_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_probe ON telemetry_events(probe_name);
CREATE INDEX IF NOT EXISTS idx_events_coords ON telemetry_events(pos_x, pos_y);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON telemetry_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON telemetry_episodes(session_id);

-- Dynamic Custom Telemetry Events Stream
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

CREATE INDEX IF NOT EXISTS idx_custom_events_probe ON custom_telemetry_events(probe_name);
CREATE INDEX IF NOT EXISTS idx_custom_events_episode ON custom_telemetry_events(episode_id);
CREATE INDEX IF NOT EXISTS idx_custom_events_category ON custom_telemetry_events(category);
CREATE INDEX IF NOT EXISTS idx_custom_events_timestamp ON custom_telemetry_events(timestamp);

-- Normalized Relational Studio Observability Tables (Milestone 7 Godot 4 Transition)
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

CREATE INDEX IF NOT EXISTS idx_studio_events_build_id ON studio_events(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_events_type ON studio_events(event_type);
CREATE INDEX IF NOT EXISTS idx_studio_tasks_build_id ON studio_tasks(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_tasks_agent ON studio_tasks(agent_role);
CREATE INDEX IF NOT EXISTS idx_studio_artifacts_build_id ON studio_artifacts(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_artifacts_task_id ON studio_artifacts(task_id);
CREATE INDEX IF NOT EXISTS idx_studio_requirements_build_id ON studio_requirements(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_checkpoints_build_id ON studio_checkpoints(build_id);


