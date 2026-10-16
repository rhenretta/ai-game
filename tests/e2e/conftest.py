"""tests/e2e/conftest.py
Shared test fixtures, mock environments, and configuration for E2E opaque-box testing.
Enforces headless SDL mode, SQLite WAL mode, and isolated filesystem sandboxes.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
from typing import Any, Dict, Generator, List, Tuple
import pytest

# Enforce headless mode for macOS & CI environments
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# Canonical 16-color Style Bible palette (RGB)
STYLE_BIBLE_16_PALETTE = [
    (0, 0, 0),        # 0: Pure Black
    (29, 43, 83),     # 1: Dark Navy
    (126, 37, 83),    # 2: Deep Crimson / Violet
    (0, 135, 81),     # 3: Dark Green
    (171, 82, 54),    # 4: Rust Brown
    (95, 87, 79),     # 5: Muted Slate Gray
    (194, 195, 199),  # 6: Light Gray
    (255, 241, 232),  # 7: Off-White / Pearl
    (255, 0, 77),     # 8: Neon Red
    (255, 163, 0),    # 9: Amber Orange
    (255, 236, 39),   # 10: Bright Yellow
    (0, 228, 54),     # 11: Electric Green
    (41, 173, 255),   # 12: Cyan Sky Blue
    (131, 118, 156),  # 13: Lavender Purple
    (255, 119, 168),  # 14: Pastel Rose Pink
    (255, 204, 170),  # 15: Peach Flesh Tint
]


@pytest.fixture
def temp_e2e_db(tmp_path: Path) -> Generator[str, None, None]:
    """Creates a temporary SQLite database in WAL mode with the complete schema.
    Includes telemetry_sessions, telemetry_episodes, telemetry_events, and telemetry_probes.
    """
    db_file = tmp_path / "telemetry_e2e.db"
    conn = sqlite3.connect(str(db_file), timeout=5.0)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
    cursor.execute("PRAGMA busy_timeout = 5000;")
    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.execute("""
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
    """)

    cursor.execute("""
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
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_probes (
            probe_id TEXT PRIMARY KEY,
            probe_name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            hypothesis_target TEXT NOT NULL,
            schema_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            episode_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            probe_name TEXT,
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
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON telemetry_events(session_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_episode ON telemetry_events(episode_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON telemetry_events(event_type);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_probe ON telemetry_events(probe_name);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_coords ON telemetry_events(pos_x, pos_y);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON telemetry_events(timestamp);")

    conn.commit()
    conn.close()

    yield str(db_file)


@pytest.fixture
def mock_steam_installation(tmp_path: Path) -> Path:
    """Creates a mock macOS Steam directory structure with realistic VDF and ACF data."""
    steam_root = tmp_path / "mock_steam"
    config_dir = steam_root / "config"
    steamapps_dir = steam_root / "steamapps"
    userdata_dir = steam_root / "userdata" / "24938259" / "config"

    config_dir.mkdir(parents=True, exist_ok=True)
    steamapps_dir.mkdir(parents=True, exist_ok=True)
    userdata_dir.mkdir(parents=True, exist_ok=True)

    # loginusers.vdf
    (config_dir / "loginusers.vdf").write_text(
        '"users"\n{\n\t"76561197985203987"\n\t{\n\t\t"AccountName"\t\t"Gnaget"\n\t\t"PersonaName"\t\t"Gnaget"\n\t\t"RememberPassword"\t\t"1"\n\t\t"MostRecent"\t\t"1"\n\t\t"Timestamp"\t\t"1729000000"\n\t}\n}\n',
        encoding="utf-8"
    )

    # libraryfolders.vdf
    (steamapps_dir / "libraryfolders.vdf").write_text(
        f'"libraryfolders"\n{{\n\t"0"\n\t{{\n\t\t"path"\t\t"{steam_root}"\n\t\t"label"\t\t""\n\t\t"apps"\n\t\t{{\n\t\t\t"212680"\t\t"298283921"\n\t\t\t"1062090"\t\t"2281928312"\n\t\t\t"620"\t\t"9128381283"\n\t\t}}\n\t}}\n}}\n',
        encoding="utf-8"
    )

    # Installed app manifests (ACF)
    (steamapps_dir / "appmanifest_212680.acf").write_text(
        '"AppState"\n{\n\t"appid"\t\t"212680"\n\t"name"\t\t"FTL: Faster Than Light"\n\t"installdir"\t\t"FTL"\n\t"SizeOnDisk"\t\t"298283921"\n}\n',
        encoding="utf-8"
    )
    (steamapps_dir / "appmanifest_1062090.acf").write_text(
        '"AppState"\n{\n\t"appid"\t\t"1062090"\n\t"name"\t\t"Timberborn"\n\t"installdir"\t\t"Timberborn"\n\t"SizeOnDisk"\t\t"2281928312"\n}\n',
        encoding="utf-8"
    )

    # localconfig.vdf with playtime records for 250 apps
    apps_entries = []
    # 250 apps with various playtimes
    for i in range(1, 251):
        appid = 1000 + i
        playtime = (251 - i) * 15  # Descending playtime minutes
        last_played = 1729000000 - (i * 3600)
        apps_entries.append(
            f'\t\t\t\t"{appid}"\n\t\t\t\t{{\n\t\t\t\t\t"Playtime"\t\t"{playtime}"\n\t\t\t\t\t"LastPlayed"\t\t"{last_played}"\n\t\t\t\t}}'
        )

    apps_section = "\n".join(apps_entries)
    localconfig_text = f'"UserLocalConfigStore"\n{{\n\t"Software"\n\t{{\n\t\t"Valve"\n\t\t{{\n\t\t\t"Steam"\n\t\t\t{{\n\t\t\t\t"Apps"\n\t\t\t\t{{\n{apps_section}\n\t\t\t\t}}\n\t\t\t}}\n\t\t}}\n\t}}\n}}\n'
    (userdata_dir / "localconfig.vdf").write_text(localconfig_text, encoding="utf-8")

    return steam_root


@pytest.fixture
def mock_cartridge_active(tmp_path: Path) -> Path:
    """Creates a valid, complete game cartridge directory for testing."""
    cart_dir = tmp_path / "mock_cartridge_active"
    assets_dir = cart_dir / "assets"
    audio_dir = assets_dir / "audio"
    sprites_dir = assets_dir / "sprites"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sprites_dir.mkdir(parents=True, exist_ok=True)

    # manifest.json
    manifest = {
        "cartridge_id": "cartridge_active_test",
        "title": "Quantum Odyssey",
        "version": "1.0.0",
        "author": "Autonomous Game Studio Swarm",
        "entry_point": "mechanics.py",
        "episodes": [1, 2],
        "created_at": "2026-10-16T00:00:00Z"
    }
    (cart_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # level.json
    level_data = {
        "deliberation": "Spatial layout designed for balanced combat encounter.",
        "level_id": "level_01",
        "name": "Testing Chamber 1",
        "width": 1280.0,
        "height": 720.0,
        "spawn_point": [100.0, 360.0],
        "exit_point": [1180.0, 360.0],
        "walls": [
            [0.0, 0.0, 1280.0, 32.0],
            [0.0, 688.0, 1280.0, 32.0],
            [0.0, 0.0, 32.0, 720.0],
            [1248.0, 0.0, 32.0, 720.0]
        ],
        "enemies": [
            {"type": "drone", "x": 640.0, "y": 360.0, "hp": 50.0}
        ],
        "hazards": [
            {"type": "spikes", "x": 400.0, "y": 360.0, "radius": 24.0}
        ]
    }
    (cart_dir / "level.json").write_text(json.dumps(level_data, indent=2), encoding="utf-8")

    # narrative.json
    narrative_data = {
        "deliberation": "Narrative arc tracking protagonist escaping an abandoned orbital station.",
        "world": {"setting": "Orbital Station Alpha", "theme": "Sci-Fi Survival"},
        "characters": [
            {"id": "hero", "name": "Vanguard 7", "role": "protagonist", "portrait": "assets/sprites/hero.png"},
            {"id": "overseer", "name": "AI Overseer", "role": "rival", "portrait": "assets/sprites/overseer.png"}
        ],
        "dialogues": [
            {
                "trigger": "level_start",
                "character_id": "hero",
                "text": "Systems online. Exit portal located ahead.",
                "voice_clip": "assets/audio/hero_start.wav"
            }
        ],
        "dialogue_barks": {
            "on_damage": [
                {
                    "character_id": "hero",
                    "text": "Armor compromised!",
                    "audio_clip": "hero_start.wav"
                }
            ]
        }
    }
    (cart_dir / "narrative.json").write_text(json.dumps(narrative_data, indent=2), encoding="utf-8")

    # mechanics.py
    mechanics_code = """# Agent-authored mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.entities.player import Player as BasePlayer
from engine.entities.enemies import BaseEnemy
from engine.entities.hazard import Hazard as BaseHazard

class ActiveCartridge(BaseCartridge):
    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            level_id=f"episode_{episode_number}",
            name="Procedural Chamber",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 360.0),
            exit_point=(1180.0, 360.0),
            walls=[(0.0, 0.0, 1280.0, 32.0), (0.0, 688.0, 1280.0, 32.0)],
            deliberation="Level geometry generated for tactical traversal."
        )

    def create_player(self, x: float, y: float, telemetry_recorder=None) -> BasePlayer:
        return BasePlayer(x=x, y=y)

    def create_enemies(self, spec: BaseLevelSpec):
        return []

    def create_hazards(self, spec: BaseLevelSpec):
        return []
"""
    (cart_dir / "mechanics.py").write_text(mechanics_code, encoding="utf-8")

    # entities.py
    entities_code = """# Agent-authored entities.py
from engine.entities.base import Entity
from engine.spatial.aabb import Vector2

class CustomDrone(Entity):
    def __init__(self, x: float, y: float):
        super().__init__(x=x, y=y, width=32.0, height=32.0)
        self.health = 50.0
"""
    (cart_dir / "entities.py").write_text(entities_code, encoding="utf-8")

    # Create dummy valid 44-byte WAV file (RIFF header)
    wav_path = audio_dir / "hero_start.wav"
    riff_header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36, b"WAVE", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16, b"data", 0
    )
    wav_path.write_bytes(riff_header)

    # Create dummy 1x1 PNG file
    png_path = sprites_dir / "hero.png"
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    png_path.write_bytes(png_bytes)

    return cart_dir
