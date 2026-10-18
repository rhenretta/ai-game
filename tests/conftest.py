"""Pytest configuration and shared fixtures for Autonomous Game Engine test suites."""

import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
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
def style_bible_palette():
    """Returns canonical 16-color palette."""
    return STYLE_BIBLE_16_PALETTE


@pytest.fixture
def temp_sqlite_path(tmp_path):
    """Provides a clean temporary database path on disk for SQLite WAL tests."""
    db_file = tmp_path / "telemetry_test.db"
    return str(db_file)


@pytest.fixture
def temp_sqlite_wal_db(temp_sqlite_path):
    """Initializes a temporary SQLite database in WAL mode with the authoritative schema.
    
    Tables from PROJECT.md:
      - telemetry_sessions
      - telemetry_episodes
      - telemetry_events
    """
    conn = sqlite3.connect(temp_sqlite_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            player_profile_tag TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_episodes (
            episode_id TEXT PRIMARY KEY,
            session_id TEXT,
            episode_number INTEGER,
            status TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES telemetry_sessions(session_id)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            episode_id TEXT,
            timestamp REAL,
            event_type TEXT NOT NULL,
            pos_x REAL,
            pos_y REAL,
            vel_x REAL,
            vel_y REAL,
            health INTEGER,
            payload_json TEXT,
            FOREIGN KEY (session_id) REFERENCES telemetry_sessions(session_id),
            FOREIGN KEY (episode_id) REFERENCES telemetry_episodes(episode_id)
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON telemetry_events(session_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_episode ON telemetry_events(episode_id);")
    conn.commit()
    conn.close()
    
    yield temp_sqlite_path


@pytest.fixture
def sample_rusher_telemetry_events():
    """Generates synthetic telemetry event sequence for Aggressive High-Mobility Rusher.
    
    Characteristics:
      - Constant high velocity (> 250 px/s)
      - High dash frequency (> 60% of action ticks)
      - Direct path to exit
      - Fast completion (< 30s)
    """
    events = []
    # 20 ticks: 13 dashes, 7 movement ticks at high velocity
    for i in range(20):
        t = i * 0.1
        is_dash = (i % 3 != 0)
        event_type = "DASH" if is_dash else "MOVEMENT_TICK"
        vx = 280.0 if is_dash else 220.0
        vy = 280.0 if is_dash else 220.0
        events.append({
            "event_type": event_type,
            "timestamp": t,
            "pos": (64.0 + i * 35.0, 64.0 + i * 35.0),
            "vel": (vx, vy),
            "health": 100 - (i * 2),
            "payload": {"stamina_cost": 20 if is_dash else 0}
        })
    return events


@pytest.fixture
def sample_cautious_telemetry_events():
    """Generates synthetic telemetry event sequence for Methodical Ranged/Cautious Player.
    
    Characteristics:
      - Low velocity (< 120 px/s)
      - Low dash frequency (< 10%)
      - Prolonged stationary cover-peeking
      - High health preservation
    """
    events = []
    # 20 ticks: 1 dash, 10 slow movements, 9 stationary pauses
    for i in range(20):
        t = i * 0.1
        if i == 5:
            event_type = "DASH"
            vx, vy = 150.0, 0.0
        elif i < 10:
            event_type = "MOVEMENT_TICK"
            vx, vy = 80.0, 60.0
        else:
            event_type = "MOVEMENT_TICK"
            vx, vy = 2.0, 1.0  # Stationary / idle noise
        events.append({
            "event_type": event_type,
            "timestamp": t,
            "pos": (64.0 + min(i, 8) * 10.0, 64.0 + min(i, 8) * 8.0),
            "vel": (vx, vy),
            "health": 100,
            "payload": {"in_cover": True if i >= 10 else False}
        })
    return events


@pytest.fixture
def sample_level_grid_16x16():
    """16x16 grid with perimeter walls and an open winding path.
    0 = walkable, 1 = solid obstacle.
    Spawn: (1, 1), Exit: (14, 14)
    """
    grid = [[0 for _ in range(16)] for _ in range(16)]
    # Perimeter walls
    for x in range(16):
        grid[0][x] = 1
        grid[15][x] = 1
        grid[x][0] = 1
        grid[x][15] = 1
    # Interior maze walls leaving clear path
    for y in range(2, 12):
        grid[y][5] = 1
    for y in range(4, 14):
        grid[y][10] = 1
    return {
        "grid": grid,
        "width": 16,
        "height": 16,
        "spawn": (1, 1),
        "exit": (14, 14)
    }


@pytest.fixture
def blocked_level_grid_16x16(sample_level_grid_16x16):
    """16x16 grid where the exit is completely surrounded by solid obstacles."""
    data = sample_level_grid_16x16
    grid = [row[:] for row in data["grid"]]
    ex, ey = data["exit"]
    # Seal exit (14, 14) with walls at (13, 14) and (14, 13)
    grid[ey - 1][ex] = 1
    grid[ey][ex - 1] = 1
    grid[ey - 1][ex - 1] = 1
    return {
        "grid": grid,
        "width": 16,
        "height": 16,
        "spawn": data["spawn"],
        "exit": data["exit"],
        "blocked_nodes": [(ex - 1, ey), (ex, ey - 1)]
    }


class MockGeminiResponse:
    def __init__(self, text: str = "", audio_bytes: Optional[bytes] = None, image_bytes: Optional[bytes] = None):
        self.text = text
        self.audio_bytes = audio_bytes
        self.image_bytes = image_bytes

        from unittest.mock import MagicMock
        candidate = MagicMock()
        content = MagicMock()
        part = MagicMock()
        part.text = text
        if audio_bytes is not None:
            part.inline_data = MagicMock()
            part.inline_data.data = audio_bytes
            part.inline_data.mime_type = "audio/wav"
        if image_bytes is not None:
            part.inline_data = MagicMock()
            part.inline_data.data = image_bytes
            part.inline_data.mime_type = "image/png"
        content.parts = [part]
        candidate.content = content
        self.candidates = [candidate]


class MockGeminiClient:
    """Mock google.genai Client supporting set_json_response, set_audio_response,
    set_image_response, and set_error for zero-fallback testing."""
    def __init__(self):
        self._responses = []
        self._error = None
        self.models = self

    def set_json_response(self, data: Any) -> "MockGeminiClient":
        import json
        if isinstance(data, str):
            text = data
        else:
            text = json.dumps(data)
        self._responses.append(MockGeminiResponse(text=text))
        return self

    def set_audio_response(self, audio_bytes: bytes) -> "MockGeminiClient":
        self._responses.append(MockGeminiResponse(audio_bytes=audio_bytes))
        return self

    def set_image_response(self, image_bytes: bytes) -> "MockGeminiClient":
        self._responses.append(MockGeminiResponse(image_bytes=image_bytes))
        return self

    def set_error(self, exc: Exception) -> "MockGeminiClient":
        self._error = exc
        return self

    def generate_content(self, model: str, contents: Any, config: Optional[Any] = None) -> MockGeminiResponse:
        if self._error:
            raise self._error
        if self._responses:
            return self._responses.pop(0)
        return MockGeminiResponse(text="")


@pytest.fixture
def anyio_backend():
    """Restricts anyio backend to asyncio only."""
    return "asyncio"


@pytest.fixture
def mock_gemini_client():
    """Provides a fresh MockGeminiClient for specialist agents."""
    return MockGeminiClient()


def create_open_genre_cartridge(dest: Any) -> Any:
    """Populates a directory with a fully valid open-genre cartridge adhering to all 5 quality gates."""
    import json
    import math
    from pathlib import Path
    import struct
    import wave
    from PIL import Image

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    (dest / "manifest.json").write_text(json.dumps({
        "cartridge_id": "test_open_genre",
        "title": "Test Open Genre Game",
        "version": "1.0.0",
        "entrypoint": "mechanics.py",
        "author": "Autonomous Studio Swarm",
    }), encoding="utf-8")

    (dest / "mechanics.py").write_text('''"""mechanics.py"""
from typing import Any, Dict, Optional
from engine.api.contracts import BaseCartridge, BaseScene
from engine.telemetry.probes import AgentTelemetryProbe

class TestScene(BaseScene):
    deliberation: str = "Test open-genre scene."
    def __init__(self, engine: Any):
        super().__init__(engine)
        self.state_counter = 0
        self.probe = AgentTelemetryProbe(
            recorder=getattr(engine, "telemetry_recorder", None),
            probe_name="test_probe",
            category="test",
        )
    def on_enter(self, context=None): self.is_active = True
    def on_exit(self): self.is_active = False; return {}
    def handle_event(self, event): self.state_counter += 1
    def update(self, dt):
        self.state_counter += 1
        if self.probe and self.state_counter % 5 == 0:
            self.probe.emit(pos=(100.0, 100.0), data={"tick": self.state_counter})
    def render(self, surface): pass

class OpenGenreCartridge(BaseCartridge):
    deliberation: str = "Test open-genre cartridge."
    def get_scenes(self, engine: Any): return {"main": TestScene(engine)}
    def get_initial_scene(self, engine: Any): return self.get_scenes(engine)["main"]
''', encoding="utf-8")

    audio_dir = dest / "assets" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio_dir / "voice.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        raw = bytearray()
        for i in range(11025):
            val = int(12000 * math.sin(2 * math.pi * 440 * i / 22050))
            raw.extend(struct.pack("<h", val))
        wf.writeframes(raw)

    sprite_dir = dest / "assets" / "sprites"
    sprite_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=(0, 0, 0))
    img.save(sprite_dir / "sprite.png", format="PNG")

    (dest / "narrative.json").write_text(json.dumps({
        "world": {"world_title": "Test World"},
        "characters": [{"id": "hero", "name": "Hero", "portrait": "assets/sprites/sprite.png"}],
        "dialogues": [{"bark_id": "b1", "text": "Ready.", "voice_clip": "assets/audio/voice.wav"}],
        "dialogue_trees": [{"root_node_id": "n1", "nodes": {"n1": {"text": "Hi", "choices": []}}}],
    }), encoding="utf-8")

    (dest / "telemetry_manifest.json").write_text(json.dumps({
        "probes": [{"probe_name": "test_probe", "category": "test"}]
    }), encoding="utf-8")

    return dest


@pytest.fixture
def sample_open_genre_cartridge(tmp_path_factory):
    """Creates a temporary, fully valid open-genre cartridge adhering to all 5 quality gates."""
    dest = tmp_path_factory.mktemp("cartridges") / "active"
    return create_open_genre_cartridge(dest)


@pytest.fixture(scope="session", autouse=True)
def clean_stale_telemetry_db():
    """Removes stale telemetry database state without unlinking open inodes.

    Preserves open file handles and inodes for live processes (e.g. run_studio.py
    server PID 34772) to prevent POSIX lock divergence, WAL mapping corruption,
    and sqlite3.OperationalError('disk I/O error').
    """
    import sqlite3
    import config

    db_path = config.DB_PATH
    if not db_path.exists():
        return

    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            conn.execute("PRAGMA busy_timeout = 5000;")
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cur.fetchall()]
            if tables:
                conn.execute("PRAGMA foreign_keys = OFF;")
                for tbl in ["custom_telemetry_events", "telemetry_events", "telemetry_episodes", "telemetry_sessions", "telemetry_probes"]:
                    if tbl in tables:
                        conn.execute(f"DELETE FROM {tbl};")
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence';")
                if cur.fetchone():
                    conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('telemetry_events', 'custom_telemetry_events');")
                conn.commit()
                conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

            # Re-seed built-in engine probes for referential integrity
            try:
                conn.execute("""
                    INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target, schema_json)
                    VALUES 
                        ('probe_dialogue_bark_played', 'dialogue_bark_played', 'dialogue', 'Reactive dialogue bark played in PlayingScene', 'H_DIALOGUE_ENGAGEMENT', '{}'),
                        ('probe_cartridge_hot_reload_triggered', 'cartridge_hot_reload_triggered', 'system', 'In-game cartridge hot-reload event', 'H_HOT_RELOAD', '{}')
                    ON CONFLICT(probe_name) DO NOTHING;
                """)
            except Exception:
                pass

            # Re-seed active cartridge probes from telemetry_manifest.json
            active_manifest = config.PROJECT_ROOT / "cartridges" / "active" / "telemetry_manifest.json"
            if active_manifest.is_file():
                try:
                    import json
                    m_data = json.loads(active_manifest.read_text(encoding="utf-8"))
                    for p in m_data.get("probes", []):
                        if isinstance(p, dict) and p.get("probe_name"):
                            conn.execute("""
                                INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target, schema_json)
                                VALUES (?, ?, ?, ?, ?, ?)
                                ON CONFLICT(probe_name) DO NOTHING;
                            """, (
                                str(p.get("probe_id") or f"prb_{p['probe_name']}"),
                                str(p["probe_name"]),
                                str(p.get("category", "general")),
                                str(p.get("description", "")),
                                str(p.get("hypothesis_target", "")),
                                json.dumps(p.get("payload_schema", {})),
                            ))
                except Exception:
                    pass

            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def isolate_studio_daemon_in_run_studio_tests(request, monkeypatch):
    """Ensures test_genesis_cycle_promotes_cartridge exercises mock_supervisor without unmocked live Gemini API calls."""
    if "test_run_studio" in request.node.nodeid and "test_genesis_cycle_promotes_cartridge" in request.node.nodeid:
        from engine.studio.daemon import StudioDaemon
        orig_init = StudioDaemon.__init__
        def mocked_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.orchestrator = None
        monkeypatch.setattr(StudioDaemon, "__init__", mocked_init)


@pytest.fixture
def stash_active_cartridge():
    """Hermetically stashes cartridges/active for tests verifying absent active cartridge behavior.

    Moves cartridges/active to cartridges/.active_stash during the test, ensuring
    subprocesses and in-process code observe no active cartridge. Touches
    cartridges/.no_cold_boot to prevent background studio daemon cold boots.
    Guarantees atomic restoration of the original active cartridge upon test completion.
    """
    import shutil
    from pathlib import Path
    import config

    cartridges_dir = config.PROJECT_ROOT / "cartridges"
    active_dir = cartridges_dir / "active"
    stash_dir = cartridges_dir / ".active_stash"
    checkpoints_dir = cartridges_dir / "checkpoints"
    cp_stash_dir = cartridges_dir / ".checkpoints_stash"
    no_cold_boot = cartridges_dir / ".no_cold_boot"
    created_no_cold_boot = False
    stashed = False
    stashed_cp = False

    # Prevent studio daemon cold boot while stashed
    if not no_cold_boot.exists():
        try:
            no_cold_boot.touch(exist_ok=True)
            created_no_cold_boot = True
        except OSError:
            pass

    if active_dir.exists():
        if stash_dir.exists():
            shutil.rmtree(stash_dir, ignore_errors=True)
        try:
            active_dir.rename(stash_dir)
            stashed = True
        except OSError:
            shutil.move(str(active_dir), str(stash_dir))
            stashed = True

    if checkpoints_dir.exists():
        if cp_stash_dir.exists():
            shutil.rmtree(cp_stash_dir, ignore_errors=True)
        try:
            checkpoints_dir.rename(cp_stash_dir)
            stashed_cp = True
        except OSError:
            shutil.move(str(checkpoints_dir), str(cp_stash_dir))
            stashed_cp = True

    try:
        yield active_dir
    finally:
        if stashed:
            if active_dir.exists():
                shutil.rmtree(active_dir, ignore_errors=True)
            if stash_dir.exists():
                try:
                    stash_dir.rename(active_dir)
                except OSError:
                    shutil.move(str(stash_dir), str(active_dir))
        elif stash_dir.exists() and not active_dir.exists():
            try:
                stash_dir.rename(active_dir)
            except OSError:
                shutil.move(str(stash_dir), str(active_dir))

        if stashed_cp:
            if checkpoints_dir.exists():
                shutil.rmtree(checkpoints_dir, ignore_errors=True)
            if cp_stash_dir.exists():
                try:
                    cp_stash_dir.rename(checkpoints_dir)
                except OSError:
                    shutil.move(str(cp_stash_dir), str(checkpoints_dir))
        elif cp_stash_dir.exists() and not checkpoints_dir.exists():
            try:
                cp_stash_dir.rename(checkpoints_dir)
            except OSError:
                shutil.move(str(cp_stash_dir), str(checkpoints_dir))

        if created_no_cold_boot and no_cold_boot.exists():
            try:
                no_cold_boot.unlink()
            except OSError:
                pass


