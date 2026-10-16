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

