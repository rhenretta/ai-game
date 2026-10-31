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

CREATE INDEX IF NOT EXISTS idx_studio_events_build_id ON studio_events(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_events_type ON studio_events(event_type);
CREATE INDEX IF NOT EXISTS idx_studio_tasks_build_id ON studio_tasks(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_tasks_agent ON studio_tasks(agent_role);
CREATE INDEX IF NOT EXISTS idx_studio_artifacts_build_id ON studio_artifacts(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_artifacts_task_id ON studio_artifacts(task_id);
CREATE INDEX IF NOT EXISTS idx_studio_requirements_build_id ON studio_requirements(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_checkpoints_build_id ON studio_checkpoints(build_id);
CREATE INDEX IF NOT EXISTS idx_studio_game_bibles_updated ON studio_game_bibles(updated_at_utc);

-- Decoupled Blind Playtester Sessions, Turns, Compactions & Wishes
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
    cursor_end_y REAL DEFAULT 640.0,
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

CREATE INDEX IF NOT EXISTS idx_playtest_sessions_checkpoint ON playtest_sessions(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_playtest_sessions_candidate ON playtest_sessions(candidate_id);
CREATE INDEX IF NOT EXISTS idx_playtest_sessions_started ON playtest_sessions(started_at_utc);
CREATE INDEX IF NOT EXISTS idx_playtest_turns_session ON playtest_turns(session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_playtest_compactions_session ON playtest_compactions(session_id, compaction_index);
CREATE INDEX IF NOT EXISTS idx_playtest_wishes_session ON playtest_wishes(session_id);
CREATE INDEX IF NOT EXISTS idx_playtest_wishes_candidate ON playtest_wishes(candidate_id, status);



