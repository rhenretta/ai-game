"""Pytest configuration and shared fixtures for Autonomous Game Engine test suites."""

import atexit
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import pytest

# Enforce headless mode for macOS & CI environments
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
# Keep Writers' Room LLM call ordering deterministic for scripted fakes (parallelism is tested explicitly).
os.environ["STUDIO_ROOM_PARALLELISM"] = "1"

# Hermetically isolate all pytest database, cartridges & studio data paths away from production directories
_PYTEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="pytest_studio_data_"))
_PYTEST_DB_PATH = _PYTEST_DATA_DIR / "telemetry.db"
_PYTEST_CARTRIDGES_DIR = _PYTEST_DATA_DIR / "cartridges"
os.environ["STUDIO_DATA_DIR"] = str(_PYTEST_DATA_DIR)
os.environ["STUDIO_DB_PATH"] = str(_PYTEST_DB_PATH)
os.environ["STUDIO_CARTRIDGES_DIR"] = str(_PYTEST_CARTRIDGES_DIR)

import config as _root_config  # noqa: E402
_root_config.DATA_DIR = _PYTEST_DATA_DIR
_root_config.DB_PATH = _PYTEST_DB_PATH

atexit.register(lambda: shutil.rmtree(_PYTEST_DATA_DIR, ignore_errors=True))


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


def _seed_pytest_cartridges_dir(root: Path) -> None:
    """Seeds per-worker isolated active cartridge and minimal Godot 4.3 checkpoint."""
    create_open_genre_cartridge(root / "active")
    cp_dir = root / "checkpoints" / "cp_001_hermetic"
    cp_dir.mkdir(parents=True, exist_ok=True)
    (cp_dir / "project.godot").write_text(
        '; Engine configuration file.\n'
        'config_version=5\n\n'
        '[application]\n'
        'config/name="HermeticPytestCheckpoint"\n'
        'run/main_scene="res://main.tscn"\n'
        'config/features=PackedStringArray("4.3")\n',
        encoding="utf-8",
    )
    (cp_dir / "main.tscn").write_text(
        '[gd_scene load_steps=2 format=3]\n'
        '[ext_resource type="Script" path="res://main.gd" id="1"]\n'
        '[node name="Main" type="Node2D"]\n'
        'script = ExtResource("1")\n',
        encoding="utf-8",
    )
    (cp_dir / "main.gd").write_text(
        'extends Node2D\n'
        'var tick_count: int = 0\n'
        'func _physics_process(_delta: float) -> void:\n'
        '\ttick_count += 1\n',
        encoding="utf-8",
    )
    (cp_dir / "manifest.json").write_text(
        '{"deliberation": "Hermetic checkpoint.", "name": "HermeticPytestCheckpoint", "version": "1.0.0"}',
        encoding="utf-8",
    )


_seed_pytest_cartridges_dir(_PYTEST_CARTRIDGES_DIR)


@pytest.fixture
def sample_open_genre_cartridge(tmp_path_factory):
    """Creates a temporary, fully valid open-genre cartridge adhering to all 5 quality gates."""
    dest = tmp_path_factory.mktemp("cartridges") / "active"
    return create_open_genre_cartridge(dest)


@pytest.fixture(scope="session", autouse=True)
def clean_stale_telemetry_db():
    """Initializes the isolated pytest temporary telemetry database.

    Never touches the production <PROJECT_ROOT>/data/telemetry.db database.
    """
    import sqlite3
    import config
    from engine.telemetry.db import TelemetryDB

    prod_db = (config.PROJECT_ROOT / "data" / "telemetry.db").resolve()
    db_path = _PYTEST_DB_PATH
    assert db_path.resolve() != prod_db, "Pytest DB_PATH must never point to production data/telemetry.db"

    try:
        # Ensure schema exists in the isolated pytest DB
        TelemetryDB(db_path)
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            conn.execute("PRAGMA busy_timeout = 5000;")
            # Seed built-in engine probes for referential integrity in the isolated test DB
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
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def isolate_llm_and_db_in_tests(request, monkeypatch):
    """Guarantees every test uses the isolated pytest DB and never makes live network Gemini API calls."""
    import config

    monkeypatch.setenv("STUDIO_DATA_DIR", str(_PYTEST_DATA_DIR))
    monkeypatch.setenv("STUDIO_DB_PATH", str(_PYTEST_DB_PATH))
    monkeypatch.setenv("STUDIO_CARTRIDGES_DIR", str(_PYTEST_CARTRIDGES_DIR))
    monkeypatch.setattr(config, "DATA_DIR", _PYTEST_DATA_DIR, raising=False)
    monkeypatch.setattr(config, "DB_PATH", _PYTEST_DB_PATH, raising=False)
    try:
        from engine.studio.models import TOKEN_USAGE_TRACKER
        TOKEN_USAGE_TRACKER.reset()
    except Exception:
        pass

    # Block any unmocked google.genai.Client network instantiation
    try:
        from google import genai

        class _OfflineTestModels:
            def generate_content(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Unmocked live Gemini API call (generate_content) blocked during pytest.")

            def generate_images(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Unmocked live Gemini API call (generate_images) blocked during pytest.")

        class _OfflineTestGenaiClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.api_key = kwargs.get("api_key", "mock-pytest-key")
                self.models = _OfflineTestModels()

        monkeypatch.setattr(genai, "Client", _OfflineTestGenaiClient)
    except ImportError:
        pass

    if "test_run_studio" in request.node.nodeid and "test_genesis_cycle_promotes_cartridge" in request.node.nodeid:
        from engine.studio.daemon import StudioDaemon
        orig_init = StudioDaemon.__init__
        def mocked_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.orchestrator = None
        monkeypatch.setattr(StudioDaemon, "__init__", mocked_init)


@pytest.fixture
def stash_active_cartridge(tmp_path, monkeypatch):
    """Hermetically provides an empty cartridges directory for tests verifying absent active cartridge behavior.

    Uses STUDIO_CARTRIDGES_DIR pointing to an isolated tmp_path directory so parallel xdist workers
    and live Studio builds never touch <PROJECT_ROOT>/cartridges/active.
    """
    empty_cartridges_dir = tmp_path / "empty_cartridges"
    empty_cartridges_dir.mkdir(parents=True, exist_ok=True)
    (empty_cartridges_dir / ".no_cold_boot").touch(exist_ok=True)
    monkeypatch.setenv("STUDIO_CARTRIDGES_DIR", str(empty_cartridges_dir))
    yield empty_cartridges_dir / "active"




# ---------------------------------------------------------------------------
# Retry-backoff sleeps in agent code (Gemini retry loops) must never slow tests down.
# Only the `time` / `asyncio` references inside these engine modules are proxied, so tests that
# rely on real timing elsewhere (launcher signal sweeps, WAL concurrency) are unaffected.
# ---------------------------------------------------------------------------
class _NoSleepTime:
    def __init__(self, real):
        self._real = real

    def sleep(self, _seconds=0):
        return None

    def __getattr__(self, name):
        return getattr(self._real, name)


class _NoSleepAsyncio:
    def __init__(self, real):
        self._real = real

    async def sleep(self, _seconds=0, result=None):
        return result

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture(autouse=True)
def _no_retry_backoff_sleeps(monkeypatch):
    import importlib

    models_mod = importlib.import_module("engine.studio.models")
    monkeypatch.setattr(models_mod, "_retry_sleep", lambda _seconds=0: None)

    for mod_name, attr, proxy in (
        ("engine.studio.swe", "time", _NoSleepTime),
        ("engine.studio.visual", "time", _NoSleepTime),
        ("engine.studio.audio", "time", _NoSleepTime),
        ("engine.studio.audio_evaluator", "time", _NoSleepTime),
        ("engine.playtest.agent", "time", _NoSleepTime),
        ("engine.steam.store_api", "time", _NoSleepTime),
        ("engine.studio.orchestrator", "asyncio", _NoSleepAsyncio),
    ):
        mod = importlib.import_module(mod_name)
        monkeypatch.setattr(mod, attr, proxy(getattr(mod, attr)))
    yield
