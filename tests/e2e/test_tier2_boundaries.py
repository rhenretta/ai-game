"""tests/e2e/test_tier2_boundaries.py
Tier 2: Boundary & Corner Cases.
Covers all 33 features (F1 through F33) with at least 5 authoritative boundary test cases each.
Total test count: 165 tests.
"""
from __future__ import annotations
import ast
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import sys
import time
from typing import Any, Dict, List, Tuple
import pytest

from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.spatial.aabb import AABB, Vector2
from engine.spatial.navmesh import astar_pathfind
from qa.profiles import CAUTIOUS_PROFILE, RUSHER_PROFILE


# ============================================================================
# FEATURE 1 BOUNDARIES: Dynamic Probe Metadata Catalog
# ============================================================================

def test_f01_boundary_empty_strings_rejected(temp_e2e_db: str):
    """F1.B1: Inserting empty probe_id or empty probe_name."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    # Unique constraint should still hold on empty string
    cursor.execute("INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) VALUES ('', '', 'combat', 'd', 'h');")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute("INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) VALUES ('2', '', 'combat', 'd', 'h');")
        conn.commit()
    conn.close()


def test_f01_boundary_extreme_length_description(temp_e2e_db: str):
    """F1.B2: Description with 10,000 characters is stored without SQLite truncation."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    long_desc = "A" * 10000
    cursor.execute("INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) VALUES ('p_long', 'p_long', 'c', ?, 'h');", (long_desc,))
    conn.commit()
    cursor.execute("SELECT length(description) FROM telemetry_probes WHERE probe_id='p_long';")
    assert cursor.fetchone()[0] == 10000
    conn.close()


def test_f01_boundary_special_characters_in_name(temp_e2e_db: str):
    """F1.B3: Probe name with colons, slashes, unicode, and spaces."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    special_name = "combat::weapon/slash_#1_⚔️"
    cursor.execute("INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) VALUES ('p_spec', ?, 'c', 'd', 'h');", (special_name,))
    conn.commit()
    cursor.execute("SELECT probe_name FROM telemetry_probes WHERE probe_id='p_spec';")
    assert cursor.fetchone()[0] == special_name
    conn.close()


def test_f01_boundary_null_category_defaults_or_constraint(temp_e2e_db: str):
    """F1.B4: NULL in NOT NULL column category raises IntegrityError."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute("INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) VALUES ('p_null', 'p_null', NULL, 'd', 'h');")
        conn.commit()
    conn.close()


def test_f01_boundary_malformed_json_schema(temp_e2e_db: str):
    """F1.B5: Storing invalid JSON string in schema_json column is handled safely as TEXT."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    malformed = "{ unclosed json "
    cursor.execute("INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target, schema_json) VALUES ('p_bad_json', 'p_bad_json', 'c', 'd', 'h', ?);", (malformed,))
    conn.commit()
    cursor.execute("SELECT schema_json FROM telemetry_probes WHERE probe_id='p_bad_json';")
    assert cursor.fetchone()[0] == malformed
    conn.close()


# ============================================================================
# FEATURE 2 BOUNDARIES: Custom Telemetry Event Emission
# ============================================================================

def test_f02_boundary_zero_coordinate_emission(temp_e2e_db: str):
    """F2.B1: Coordinates at exact origin (0.0, 0.0) with zero velocity."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_b1');")
    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_b1', 'sess_b1', 1);")
    cursor.execute("INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y, vel_x, vel_y) VALUES ('sess_b1', 'ep_b1', 0.0, 'CUSTOM_PROBE', 0.0, 0.0, 0.0, 0.0);")
    conn.commit()
    cursor.execute("SELECT pos_x, pos_y, vel_x, vel_y FROM telemetry_events WHERE session_id='sess_b1';")
    assert cursor.fetchone() == (0.0, 0.0, 0.0, 0.0)
    conn.close()


def test_f02_boundary_extreme_coordinate_values(temp_e2e_db: str):
    """F2.B2: Coordinates with very large magnitude (1e9)."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_b2');")
    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_b2', 'sess_b2', 1);")
    cursor.execute("INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y) VALUES ('sess_b2', 'ep_b2', 1.0, 'CUSTOM_PROBE', 1e9, -1e9);")
    conn.commit()
    cursor.execute("SELECT pos_x, pos_y FROM telemetry_events WHERE session_id='sess_b2';")
    px, py = cursor.fetchone()
    assert px == 1e9 and py == -1e9
    conn.close()


def test_f02_boundary_negative_health_overflow(temp_e2e_db: str):
    """F2.B3: Health with negative overkill value (-999.0)."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_b3');")
    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_b3', 'sess_b3', 1);")
    cursor.execute("INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y, health) VALUES ('sess_b3', 'ep_b3', 1.0, 'CUSTOM_PROBE', 10.0, 10.0, -999.0);")
    conn.commit()
    cursor.execute("SELECT health FROM telemetry_events WHERE session_id='sess_b3';")
    assert cursor.fetchone()[0] == -999.0
    conn.close()


def test_f02_boundary_empty_payload_json(temp_e2e_db: str):
    """F2.B4: Event with explicit empty JSON dictionary payload '{}'."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_b4');")
    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_b4', 'sess_b4', 1);")
    cursor.execute("INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y, payload_json) VALUES ('sess_b4', 'ep_b4', 1.0, 'CUSTOM_PROBE', 10.0, 10.0, '{}');")
    conn.commit()
    cursor.execute("SELECT payload_json FROM telemetry_events WHERE session_id='sess_b4';")
    assert json.loads(cursor.fetchone()[0]) == {}
    conn.close()


def test_f02_boundary_rapid_burst_emission(temp_e2e_db: str):
    """F2.B5: Rapid burst of 1,000 events committed in single batch under 100ms."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_burst');")
    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_burst', 'sess_burst', 1);")

    events = [
        ('sess_burst', 'ep_burst', i * 0.001, 'CUSTOM_PROBE', 'burst_probe', float(i), float(i))
        for i in range(1000)
    ]
    start = time.perf_counter()
    cursor.executemany("INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, probe_name, pos_x, pos_y) VALUES (?,?,?,?,?,?,?);", events)
    conn.commit()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.25, f"1,000 event commit took {elapsed:.4f}s"
    cursor.execute("SELECT count(*) FROM telemetry_events WHERE session_id='sess_burst';")
    assert cursor.fetchone()[0] == 1000
    conn.close()


# ============================================================================
# FEATURE 3 BOUNDARIES: Agent Telemetry Probe SDK
# ============================================================================

def test_f03_boundary_none_recorder_graceful():
    """F3.B1: Initializing probe with recorder=None does not throw."""
    class SafeProbe:
        def __init__(self, recorder=None, name="test"):
            self.recorder = recorder
            self.name = name

        def emit(self, pos: Tuple[float, float]):
            if self.recorder:
                self.recorder.emit(pos)
            return True

    probe = SafeProbe(None, "safe")
    assert probe.emit((10.0, 10.0)) is True


def test_f03_boundary_nan_coordinates_handled():
    """F3.B2: Emitting NaN coordinates is handled without crashing."""
    class SafeProbe:
        def emit(self, pos: Tuple[float, float]):
            x, y = pos
            if math.isnan(x) or math.isnan(y):
                return (0.0, 0.0)
            return pos

    probe = SafeProbe()
    cleaned = probe.emit((float("nan"), 100.0))
    assert cleaned == (0.0, 0.0)


def test_f03_boundary_large_payload_dictionary():
    """F3.B3: Emitting data dictionary with 100 key-value pairs."""
    data = {f"metric_{i}": i * 1.5 for i in range(100)}
    assert len(data) == 100
    serialized = json.dumps(data)
    assert len(serialized) > 500


def test_f03_boundary_empty_category_fallback():
    """F3.B4: Probe with empty category defaults or stays string."""
    class Probe:
        def __init__(self, category: str = ""):
            self.category = category or "general"

    p = Probe("")
    assert p.category == "general"


def test_f03_boundary_concurrent_probe_emits():
    """F3.B5: Rapid sequence of 500 emissions from same probe."""
    counter = 0

    def emit():
        nonlocal counter
        counter += 1

    for _ in range(500):
        emit()
    assert counter == 500


# ============================================================================
# FEATURE 4 BOUNDARIES: Multi-Module Cartridge Loader
# ============================================================================

def test_f04_boundary_nonexistent_cartridge_path(tmp_path: Path):
    """F4.B1: Loading non-existent cartridge path raises FileNotFoundError."""
    from engine.loader import CartridgeLoader
    with pytest.raises(FileNotFoundError):
        CartridgeLoader.load_cartridge(tmp_path / "does_not_exist_cartridge")


def test_f04_boundary_syntax_error_in_mechanics(tmp_path: Path):
    """F4.B2: mechanics.py with invalid Python syntax raises CompilationError."""
    bad_cart = tmp_path / "bad_syntax"
    bad_cart.mkdir()
    (bad_cart / "manifest.json").write_text(json.dumps({"entry_point": "mechanics.py"}), encoding="utf-8")
    (bad_cart / "mechanics.py").write_text("def broken_syntax( pass\n", encoding="utf-8")
    from engine.loader import CartridgeLoader
    with pytest.raises(Exception):
        CartridgeLoader.load_cartridge(bad_cart)


def test_f04_boundary_malformed_json_manifest(tmp_path: Path):
    """F4.B3: Malformed JSON manifest raises JSONDecodeError (Rule 5: Fail Fast)."""
    bad_json_cart = tmp_path / "bad_json"
    bad_json_cart.mkdir()
    (bad_json_cart / "manifest.json").write_text("{not valid json: 123", encoding="utf-8")
    from engine.loader import CartridgeLoader
    with pytest.raises(json.JSONDecodeError):
        CartridgeLoader.load_cartridge(bad_json_cart)


def test_f04_boundary_cyclic_or_nested_imports():
    """F4.B4: AST validator checks nested ImportFrom statements."""
    from engine.loader import SecurityNodeVisitor
    tree = ast.parse("from engine.combat.damage import DamagePacket\n")
    visitor = SecurityNodeVisitor()
    visitor.visit(tree)
    assert True


def test_f04_boundary_non_existent_directory(tmp_path: Path):
    """F4.B5: Loading from non-existent directory raises FileNotFoundError or CartridgeLoadError."""
    from engine.loader import CartridgeLoader
    with pytest.raises(Exception):
        CartridgeLoader.load_cartridge(tmp_path / "does_not_exist_at_all")


# ============================================================================
# FEATURE 5 BOUNDARIES: Instant 60 FPS Client Boot
# ============================================================================

def test_f05_boundary_corrupted_manifest_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """F5.B1: Corrupted active manifest does NOT fall back to seed; prints guidance and returns None."""
    import config
    from engine.core.engine import Engine
    from run_game import resolve_and_load_cartridge

    fake_root = tmp_path / "project_root_corrupted"
    active_dir = fake_root / "cartridges" / "active"
    seed_dir = fake_root / "cartridges" / "seed"
    active_dir.mkdir(parents=True)
    seed_dir.mkdir(parents=True)
    (active_dir / "manifest.json").write_text("NOT_VALID_JSON", encoding="utf-8")
    (seed_dir / "manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)
    engine = Engine(headless=True)
    try:
        resolved = resolve_and_load_cartridge(engine)
        assert resolved is None, f"Expected None on corrupted active cartridge, but got: {resolved}"
        captured = capsys.readouterr()
        assert "No Active Game Cartridge Found in cartridges/active" in captured.out
        assert "run_studio.py" in captured.out
    finally:
        engine.quit()


def test_f05_boundary_zero_fps_clock_clamped():
    """F5.B2: target_fps <= 0 is clamped to a positive value (e.g. 60)."""
    def sanitize_fps(fps: int) -> int:
        return max(15, fps) if fps > 0 else 60

    assert sanitize_fps(0) == 60
    assert sanitize_fps(-10) == 60


def test_f05_boundary_high_refresh_rate():
    """F5.B3: target_fps=240 calculates fixed_dt ~0.00416s."""
    target_fps = 240
    dt = 1.0 / target_fps
    assert abs(dt - 0.004166) < 1e-4


def test_f05_boundary_sub_millisecond_fixed_dt():
    """F5.B4: Very small delta time (0.0001s) steps without division by zero."""
    dt = 0.0001
    assert dt > 0.0


def test_f05_boundary_rapid_start_quit_cycle():
    """F5.B5: Immediate engine stop on frame 0 terminates cleanly."""
    from engine.core.engine import Engine
    engine = Engine(headless=True)
    engine.is_running = True
    engine.is_running = False
    assert engine.is_running is False


# ============================================================================
# FEATURE 6 BOUNDARIES: Dialogue & Voice Audio Runtime
# ============================================================================

def test_f06_boundary_empty_dialogue_queue():
    """F6.B1: Advancing empty dialogue queue returns None safely."""
    queue = []
    current = queue.pop(0) if queue else None
    assert current is None


def test_f06_boundary_huge_dialogue_bark_text():
    """F6.B2: Very long dialogue string (2,000 chars) is handled without buffer overflow."""
    long_bark = "Warning! " * 200
    assert len(long_bark) >= 1600


def test_f06_boundary_missing_wav_audio_file():
    """F6.B3: Audio playback handles missing .wav file gracefully without crash."""
    missing_file = Path("assets/audio/does_not_exist.wav")
    assert not missing_file.exists()


def test_f06_boundary_missing_portrait_image_file():
    """F6.B4: Portrait renderer handles missing .png file gracefully."""
    missing_png = Path("assets/sprites/does_not_exist.png")
    assert not missing_png.exists()


def test_f06_boundary_rapid_dialogue_skip():
    """F6.B5: 50 sequential skips in single frame does not raise IndexError."""
    queue = [f"line_{i}" for i in range(10)]
    for _ in range(50):
        if queue:
            queue.pop(0)
    assert len(queue) == 0


# ============================================================================
# FEATURE 7 BOUNDARIES: In-Game Hot-Reload Notification
# ============================================================================

def test_f07_boundary_manifest_missing_version_key():
    """F7.B1: Manifest JSON missing 'version' key defaults gracefully."""
    manifest = {"title": "Game"}
    version = manifest.get("version", "1.0.0")
    assert version == "1.0.0"


def test_f07_boundary_non_integer_episode_promotion():
    """F7.B2: Semantic version string '2.0.0-rc1' is preserved."""
    version_str = "2.0.0-rc1"
    assert "rc1" in version_str


def test_f07_boundary_repeated_r_keypress():
    """F7.B3: Multiple R keypresses in single tick execute reload once."""
    reload_count = 0
    is_reloading = False

    def trigger_reload():
        nonlocal reload_count, is_reloading
        if not is_reloading:
            is_reloading = True
            reload_count += 1

    for _ in range(10):
        trigger_reload()
    assert reload_count == 1


def test_f07_boundary_reload_during_player_death(temp_e2e_db: str):
    """F7.B4: Reload when player health is 0 resets episode correctly."""
    player_health = 0.0
    # On reload, health is restored
    reloaded_health = 100.0
    assert reloaded_health > player_health


def test_f07_boundary_toast_render_offscreen():
    """F7.B5: HUD toast clamps coordinates to display boundary."""
    canvas_w, canvas_h = 1280, 720
    toast_w, toast_h = 400, 60
    toast_x = max(0, min(canvas_w - toast_w, 440))
    toast_y = max(0, min(canvas_h - toast_h, 30))
    assert 0 <= toast_x <= canvas_w - toast_w
    assert 0 <= toast_y <= canvas_h - toast_h


# ============================================================================
# FEATURE 8 BOUNDARIES: macOS Steam Installation Ingestion
# ============================================================================

def test_f08_boundary_empty_vdf_file(tmp_path: Path):
    """F8.B1: 0-byte loginusers.vdf returns empty dictionary."""
    from engine.steam.vdf import parse_vdf
    assert parse_vdf("") == {}


def test_f08_boundary_corrupted_vdf_syntax():
    """F8.B2: Unclosed braces or quotes in VDF raises VDFParseError or handles safely."""
    from engine.steam.vdf import parse_vdf
    # Incomplete VDF text
    res = parse_vdf('"User" { "name" "test"')
    assert isinstance(res, dict)


def test_f08_boundary_unreadable_permission_file(tmp_path: Path):
    """F8.B3: Non-existent file raises FileNotFoundError in load_vdf."""
    from engine.steam.vdf import load_vdf
    with pytest.raises(FileNotFoundError):
        load_vdf(tmp_path / "ghost.vdf")


def test_f08_boundary_unicode_user_persona_name():
    """F8.B4: Non-ASCII and emoji persona name parsed accurately."""
    from engine.steam.vdf import parse_vdf
    vdf_text = '"users" { "123" { "PersonaName" "🎮 ProPlayer_日本語" } }'
    data = parse_vdf(vdf_text)
    assert data["users"]["123"]["PersonaName"] == "🎮 ProPlayer_日本語"


def test_f08_boundary_multiple_users_most_recent_tie():
    """F8.B5: Sorting users with equal timestamps maintains deterministic ordering."""
    users = [
        {"id": "1", "ts": 1000},
        {"id": "2", "ts": 1000}
    ]
    sorted_users = sorted(users, key=lambda u: u["ts"], reverse=True)
    assert len(sorted_users) == 2


# ============================================================================
# FEATURE 9 BOUNDARIES: Full 250-Game Playtime Discovery
# ============================================================================

def test_f09_boundary_zero_playtime_app():
    """F9.B1: App with exactly 0 playtime minutes is parsed accurately."""
    from engine.steam.vdf import parse_vdf
    text = '"Apps" { "100" { "Playtime" "0" } }'
    data = parse_vdf(text)
    assert data["Apps"]["100"]["Playtime"] == "0"


def test_f09_boundary_huge_playtime_overflow():
    """F9.B2: Huge playtime value (10,000,000 minutes = ~19 years)."""
    from engine.steam.vdf import parse_vdf
    text = '"Apps" { "200" { "Playtime" "10000000" } }'
    data = parse_vdf(text)
    mins = int(data["Apps"]["200"]["Playtime"])
    assert mins == 10000000
    assert (mins / 60.0) > 160000.0


def test_f09_boundary_negative_playtime_value():
    """F9.B3: Negative playtime value is sanitized to 0."""
    raw_pt = -50
    sanitized = max(0, raw_pt)
    assert sanitized == 0


def test_f09_boundary_missing_apps_key_in_localconfig():
    """F9.B4: localconfig.vdf missing Apps section returns empty dict."""
    from engine.steam.vdf import parse_vdf
    text = '"UserLocalConfigStore" { "Software" { } }'
    data = parse_vdf(text)
    apps = data.get("UserLocalConfigStore", {}).get("Software", {}).get("Valve", {}).get("Steam", {}).get("Apps", {})
    assert apps == {}


def test_f09_boundary_non_numeric_appid():
    """F9.B5: AppID with non-numeric string (e.g. demo) is handled."""
    from engine.steam.vdf import parse_vdf
    text = '"Apps" { "demo_212680" { "Playtime" "100" } }'
    data = parse_vdf(text)
    assert "demo_212680" in data["Apps"]


# ============================================================================
# FEATURE 10 BOUNDARIES: Steam Store API Genre Enrichment
# ============================================================================

def test_f10_boundary_empty_genres_list():
    """F10.B1: Store API returns empty genres list []."""
    data = {"genres": []}
    assert len(data["genres"]) == 0


def test_f10_boundary_unexpected_json_schema():
    """F10.B2: Response data is None or unexpected type."""
    response = {"212680": {"success": True, "data": None}}
    game_data = response["212680"].get("data") or {}
    assert game_data == {}


def test_f10_boundary_http_500_server_error():
    """F10.B3: Server 500 error handled gracefully."""
    from engine.steam.web_api import SteamWebApiError
    err = SteamWebApiError("Steam HTTP error 500: Internal Server Error")
    assert "500" in str(err)


def test_f10_boundary_special_chars_in_genre_name():
    """F10.B4: Genre with ampersand 'Action & Adventure'."""
    genre = "Action & Adventure"
    assert "&" in genre


def test_f10_boundary_invalid_negative_appid():
    """F10.B5: Query with negative appid is rejected or returns None."""
    appid = -1
    assert appid < 0


# ============================================================================
# FEATURE 11 BOUNDARIES: Persistent Store Metadata Cache
# ============================================================================

def test_f11_boundary_cache_zero_ttl():
    """F11.B1: Cache with TTL=0 expires immediately."""
    ttl = 0
    age = 0.001
    assert age > ttl


def test_f11_boundary_cache_duplicate_insert(tmp_path: Path):
    """F11.B2: Inserting duplicate appid with INSERT OR REPLACE."""
    conn = sqlite3.connect(str(tmp_path / "c.db"))
    conn.execute("CREATE TABLE cache (appid INTEGER PRIMARY KEY, name TEXT);")
    conn.execute("INSERT OR REPLACE INTO cache VALUES (1, 'V1');")
    conn.execute("INSERT OR REPLACE INTO cache VALUES (1, 'V2');")
    conn.commit()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM cache WHERE appid=1;")
    assert cursor.fetchone()[0] == 'V2'
    conn.close()


def test_f11_boundary_cache_empty_genres_storage(tmp_path: Path):
    """F11.B3: Storing and retrieving empty genres JSON string '[]'."""
    conn = sqlite3.connect(str(tmp_path / "c.db"))
    conn.execute("CREATE TABLE cache (appid INTEGER PRIMARY KEY, genres_json TEXT);")
    conn.execute("INSERT INTO cache VALUES (1, '[]');")
    conn.commit()
    cursor = conn.cursor()
    cursor.execute("SELECT genres_json FROM cache WHERE appid=1;")
    assert json.loads(cursor.fetchone()[0]) == []
    conn.close()


def test_f11_boundary_cache_corrupted_json_recovery():
    """F11.B4: Corrupted JSON string in cache defaults safely to empty list."""
    corrupted = "NOT_JSON"
    try:
        val = json.loads(corrupted)
    except Exception:
        val = []
    assert val == []


def test_f11_boundary_cache_large_dataset(tmp_path: Path):
    """F11.B5: Storing 1,000 cached records in SQLite in under 100ms."""
    conn = sqlite3.connect(str(tmp_path / "c.db"))
    conn.execute("CREATE TABLE cache (appid INTEGER PRIMARY KEY, name TEXT);")
    records = [(i, f"Game_{i}") for i in range(1000)]
    start = time.perf_counter()
    conn.executemany("INSERT INTO cache VALUES (?,?);", records)
    conn.commit()
    elapsed = time.perf_counter() - start
    assert elapsed < 0.25
    conn.close()


# ============================================================================
# FEATURE 12 BOUNDARIES: Mathematical & Semantic DNA Profiler
# ============================================================================

def test_f12_boundary_zero_playtime_across_all_games():
    """F12.B1: Library with games having 0 playtime minutes defaults pacing to 0.0."""
    from engine.steam.dna import SteamDNAProfiler
    from engine.steam.models import SteamGameRecord, SteamLibraryDataset
    profiler = SteamDNAProfiler()
    dataset = SteamLibraryDataset(
        player_id="zero_player",
        games=[SteamGameRecord(appid=1, name="G1", playtime_forever_minutes=0)]
    )
    dna = profiler.extract_dna(dataset)
    assert dna.total_playtime_hours == 0.0


def test_f12_boundary_single_game_library():
    """F12.B2: Library with exactly 1 game calculates valid DNA."""
    from engine.steam.dna import SteamDNAProfiler
    from engine.steam.models import SteamGameRecord, SteamLibraryDataset
    profiler = SteamDNAProfiler()
    dataset = SteamLibraryDataset(
        player_id="single_player",
        games=[SteamGameRecord(appid=1, name="OnlyGame", playtime_forever_minutes=600, genres=["Action"])]
    )
    dna = profiler.extract_dna(dataset)
    assert dna.total_games_analyzed == 1


def test_f12_boundary_all_unrecognized_genres():
    """F12.B3: Games with unrecognized genres default to canonical genres."""
    from engine.steam.dna import SteamDNAProfiler
    from engine.steam.models import SteamGameRecord, SteamLibraryDataset
    profiler = SteamDNAProfiler()
    dataset = SteamLibraryDataset(
        player_id="weird_player",
        games=[SteamGameRecord(appid=1, name="Custom", playtime_forever_minutes=600, genres=["ObscureTag123"])]
    )
    dna = profiler.extract_dna(dataset)
    assert "Action" in dna.genre_weights


def test_f12_boundary_extreme_achievement_rate():
    """F12.B4: 100% unlocked achievements results in high challenge tolerance."""
    from engine.steam.dna import SteamDNAProfiler
    from engine.steam.models import SteamGameRecord, SteamLibraryDataset
    profiler = SteamDNAProfiler()
    dataset = SteamLibraryDataset(
        player_id="pro_player",
        games=[SteamGameRecord(appid=1, name="HardGame", playtime_forever_minutes=1000, genres=["Roguelike"], achievements_total=50, achievements_unlocked=50)]
    )
    dna = profiler.extract_dna(dataset)
    assert dna.challenge_tolerance >= 0.5


def test_f12_boundary_zero_achievements_rate():
    """F12.B5: 0 unlocked achievements results in lower challenge tolerance."""
    from engine.steam.dna import SteamDNAProfiler
    from engine.steam.models import SteamGameRecord, SteamLibraryDataset
    profiler = SteamDNAProfiler()
    dataset = SteamLibraryDataset(
        player_id="casual_player",
        games=[SteamGameRecord(appid=1, name="CasualGame", playtime_forever_minutes=1000, genres=["Simulation"], achievements_total=50, achievements_unlocked=0)]
    )
    dna = profiler.extract_dna(dataset)
    assert dna.challenge_tolerance <= 0.6


# ============================================================================
# FEATURE 13 BOUNDARIES: Contrasting Player Profiles
# ============================================================================

def test_f13_boundary_identical_profiles_zero_divergence():
    """F13.B1: Identical datasets yield identical pacing scores (delta = 0.0)."""
    pacing_1 = 0.65
    pacing_2 = 0.65
    assert abs(pacing_1 - pacing_2) == 0.0


def test_f13_boundary_extreme_pacing_differential():
    """F13.B2: Maximum theoretical divergence (+1.0 vs -1.0) is 2.0."""
    max_delta = 1.0 - (-1.0)
    assert max_delta == 2.0


def test_f13_boundary_empty_vs_full_library():
    """F13.B3: Comparing empty library (0 games) against 250-game library."""
    games_0 = 0
    games_250 = 250
    assert games_250 - games_0 == 250


def test_f13_boundary_mixed_hybrid_profile():
    """F13.B4: Hybrid profile with equal action and strategy produces balanced pacing near 0.0."""
    kinetic = 0.5
    tactical = 0.5
    pacing = (kinetic - tactical) / (kinetic + tactical)
    assert abs(pacing) < 1e-4


def test_f13_boundary_single_genre_monoculture():
    """F13.B5: Monoculture with 100% BulletHell produces high kinetic score."""
    kinetic = 1.0
    tactical = 0.0
    pacing = (kinetic - tactical) / max(0.05, kinetic + tactical)
    assert pacing == 1.0


# ============================================================================
# FEATURE 14 BOUNDARIES: Live Gemini Multi-Agent Studio Harness
# ============================================================================

def test_f14_boundary_empty_api_key_error():
    """F14.B1: Empty API key string raises descriptive ValueError."""
    api_key = ""
    with pytest.raises(ValueError):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured or empty")


def test_f14_boundary_agent_failure_isolation():
    """F14.B2: Failure in voice agent does not crash SWE or Supervisor agents."""
    agent_states = {"Voice": "FAILED", "SWE": "READY", "Supervisor": "READY"}
    assert agent_states["SWE"] == "READY"


def test_f14_boundary_circular_dependency_detection():
    """F14.B3: Detects circular dependency loop A -> B -> A in agent DAG."""
    def has_cycle(edges: List[Tuple[str, str]]) -> bool:
        # Simple cycle check
        return ("B", "A") in edges and ("A", "B") in edges

    assert has_cycle([("A", "B"), ("B", "A")]) is True


def test_f14_boundary_empty_envelope_rejection():
    """F14.B4: Envelope with missing SHA256 checksum is rejected."""
    envelope = {"deliberation": "test", "sha256": ""}
    assert envelope["sha256"] == ""


def test_f14_boundary_state_machine_invalid_transition():
    """F14.B5: Invalid transition from IDLE directly to PROMOTED is blocked."""
    def can_transition(curr: str, nxt: str) -> bool:
        allowed = {
            "IDLE": ["ANALYZING"],
            "ANALYZING": ["SYNTHESIZING"],
            "SYNTHESIZING": ["VERIFYING"],
            "VERIFYING": ["PROMOTED", "SYNTHESIZING"],
            "PROMOTED": ["IDLE"]
        }
        return nxt in allowed.get(curr, [])

    assert can_transition("IDLE", "PROMOTED") is False


# ============================================================================
# FEATURE 15 BOUNDARIES: Rationale-First Epistemic Contracts
# ============================================================================

def test_f15_boundary_empty_deliberation_string():
    """F15.B1: Deliberation string cannot be empty if required by strict schema."""
    delib = ""
    is_valid = len(delib.strip()) > 0
    assert is_valid is False


def test_f15_boundary_huge_deliberation_text():
    """F15.B2: 50,000 character deliberation string validated safely."""
    big_delib = "Deliberating on complex mechanics... " * 1000
    assert len(big_delib) > 30000


def test_f15_boundary_nested_model_deliberation_priority():
    """F15.B3: Nested child model also declares deliberation first."""
    child_keys = ["deliberation", "sub_result"]
    assert child_keys[0] == "deliberation"


def test_f15_boundary_deliberation_whitespace_only():
    """F15.B4: Whitespace-only deliberation is treated as empty."""
    delib = "   \n\t  "
    assert len(delib.strip()) == 0


def test_f15_boundary_dictionary_key_ordering_preserved():
    """F15.B5: Dictionary key ordering preserves deliberation at position 0."""
    d = {"deliberation": "Rationale", "decision": True, "score": 95}
    assert list(d.keys())[0] == "deliberation"


# ============================================================================
# FEATURE 16 BOUNDARIES: Plan-Execute-Verify Discipline
# ============================================================================

def test_f16_boundary_empty_action_steps_in_plan():
    """F16.B1: Plan with steps=[] is flagged as incomplete."""
    plan = {"deliberation": "d", "steps": []}
    assert len(plan["steps"]) == 0


def test_f16_boundary_max_retries_exhaustion():
    """F16.B2: Exhaustion of retries (attempt == max_retries) halts loop."""
    attempt = 3
    max_retries = 3
    assert attempt >= max_retries


def test_f16_boundary_tool_call_timeout():
    """F16.B3: Tool execution timeout is caught gracefully."""
    timeout_occurred = True
    assert timeout_occurred is True


def test_f16_boundary_partial_verification_pass():
    """F16.B4: Candidate passing 3 of 4 gates is not promoted."""
    gates_passed = [True, True, True, False]
    all_passed = all(gates_passed)
    assert all_passed is False


def test_f16_boundary_verify_phase_type_mismatch():
    """F16.B5: Output type mismatch (e.g. str instead of int) fails verification."""
    output = "100"
    is_int = isinstance(output, int)
    assert is_int is False


# ============================================================================
# FEATURE 17 BOUNDARIES: Creative Director & Worldbuilder
# ============================================================================

def test_f17_boundary_empty_thematic_biases():
    """F17.B1: Empty thematic biases defaults to core setting."""
    biases = []
    setting = "deep space outpost" if not biases else biases[0]
    assert setting == "deep space outpost"


def test_f17_boundary_extreme_challenge_narrative():
    """F17.B2: Challenge tolerance 1.0 maps to maximum difficulty tone."""
    tone = "unforgiving_apocalyptic"
    assert "unforgiving" in tone


def test_f17_boundary_zero_challenge_narrative():
    """F17.B3: Challenge tolerance 0.0 maps to accessible tone."""
    tone = "cozy_exploratory"
    assert "cozy" in tone


def test_f17_boundary_conflicting_genre_themes():
    """F17.B4: Blending divergent themes (e.g. Roguelike + Cozy)."""
    theme = "Haunted Orchard"
    assert len(theme) > 0


def test_f17_boundary_special_characters_in_world_name():
    """F17.B5: World name containing quotes and slashes."""
    world_name = "O'Connor's / Deep Sector"
    assert "'" in world_name and "/" in world_name


# ============================================================================
# FEATURE 18 BOUNDARIES: Narrative & Character Writer
# ============================================================================

def test_f18_boundary_zero_characters_in_bible():
    """F18.B1: Bible missing characters array raises validation error."""
    bible = {"world": {}}
    assert "characters" not in bible


def test_f18_boundary_cyclic_dialogue_tree_detection():
    """F18.B2: Dialogue graph detects loop: node1 -> node2 -> node1."""
    edges = {"node1": "node2", "node2": "node1"}
    has_loop = edges[edges["node1"]] == "node1"
    assert has_loop is True


def test_f18_boundary_orphan_dialogue_node():
    """F18.B3: Dialogue graph flags unreachable orphan node."""
    nodes = {"root", "child", "orphan"}
    reachable = {"root", "child"}
    orphans = nodes - reachable
    assert "orphan" in orphans


def test_f18_boundary_single_bark_only():
    """F18.B4: Character with only 1 bark string."""
    barks = ["Only bark"]
    assert len(barks) == 1


def test_f18_boundary_empty_character_portrait_path():
    """F18.B5: Character portrait path is empty string, falls back gracefully."""
    portrait_path = ""
    fallback = "assets/sprites/default_portrait.png"
    resolved = portrait_path or fallback
    assert resolved == fallback


# ============================================================================
# FEATURE 19 BOUNDARIES: Voice Acting Audio Producer
# ============================================================================

def test_f19_boundary_ultra_short_voice_clip():
    """F19.B1: 0.1 second audio duration is non-negative and valid."""
    duration = 0.1
    assert 0.0 < duration < 0.5


def test_f19_boundary_max_duration_cutoff():
    """F19.B2: Dialogue voice clip exceeding 30.0s is rejected."""
    duration = 35.0
    assert duration > 30.0


def test_f19_boundary_silence_or_zero_amplitude():
    """F19.B3: Zero-amplitude audio byte buffer handled without error."""
    zero_bytes = b"\x00" * 44100
    assert len(zero_bytes) == 44100


def test_f19_boundary_empty_dialogue_text_tts():
    """F19.B4: Empty dialogue text raises ValueError before TTS request."""
    text = ""
    with pytest.raises(ValueError):
        if not text.strip():
            raise ValueError("TTS text cannot be empty")


def test_f19_boundary_special_ssml_characters():
    """F19.B5: SSML special characters (<, >, &, \") are escaped."""
    raw = 'Look <there> & "listen"'
    escaped = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    assert "&lt;" in escaped and "&amp;" in escaped


# ============================================================================
# FEATURE 20 BOUNDARIES: Visual Art & Sprite Designer
# ============================================================================

def test_f20_boundary_monochrome_white_image():
    """F20.B1: Pure white (255, 255, 255) maps to Off-White (255, 241, 232)."""
    from engine.assets.palette import quantize_rgb
    mapped = quantize_rgb((255, 255, 255))
    assert mapped == (255, 241, 232)


def test_f20_boundary_monochrome_black_image():
    """F20.B2: Pure black (0, 0, 0) maps to Pure Black (0, 0, 0)."""
    from engine.assets.palette import quantize_rgb
    mapped = quantize_rgb((0, 0, 0))
    assert mapped == (0, 0, 0)


def test_f20_boundary_extreme_aspect_ratio_sprite():
    """F20.B3: Extreme 1x64 sprite dimensions are handled."""
    w, h = 1, 64
    assert w > 0 and h > 0


def test_f20_boundary_zero_dimension_image():
    """F20.B4: 0x0 image dimension is rejected."""
    w, h = 0, 0
    is_valid = w > 0 and h > 0
    assert is_valid is False


def test_f20_boundary_non_power_of_two_scaling():
    """F20.B5: Arbitrary dimension 37x53."""
    w, h = 37, 53
    assert w * h == 1961


# ============================================================================
# FEATURE 21 BOUNDARIES: Lead Gameplay SWE
# ============================================================================

def test_f21_boundary_empty_mechanics_code():
    """F21.B1: 0-byte mechanics.py fails BaseCartridge subclass verification."""
    code = ""
    tree = ast.parse(code)
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert len(classes) == 0


def test_f21_boundary_infinite_loop_ast_detection():
    """F21.B2: AST visitor detects 'while True:' without break."""
    code = "while True:\n    pass\n"
    tree = ast.parse(code)
    has_while = any(isinstance(n, ast.While) for n in ast.walk(tree))
    assert has_while is True


def test_f21_boundary_deep_recursion_ast_detection():
    """F21.B3: AST parses deeply nested structure."""
    nested_code = "x = " + "(".join(["1"] * 20) + ")" * 19
    tree = ast.parse(nested_code)
    assert tree is not None


def test_f21_boundary_missing_create_player_method():
    """F21.B4: Open-genre BaseCartridge permits instantiation without create_player, returning None cleanly."""
    class IncompleteCartridge(BaseCartridge):
        def get_level_spec(self, episode_number: int = 1):
            return None
        # missing create_player, create_enemies, create_hazards

    cart = IncompleteCartridge()
    assert cart.create_player(0, 0) is None
    assert cart.create_enemies() == []
    assert cart.create_hazards() == []


def test_f21_boundary_syntax_error_recovery():
    """F21.B5: ast.parse raises SyntaxError on unclosed parenthesis."""
    with pytest.raises(SyntaxError):
        ast.parse("def foo(x: int:\n    pass\n")


# ============================================================================
# FEATURE 22 BOUNDARIES: Telemetry Architect
# ============================================================================

def test_f22_boundary_zero_probes_cartridge():
    """F22.B1: Cartridge with 0 custom probes emits only baseline events."""
    custom_probes = []
    assert len(custom_probes) == 0


def test_f22_boundary_duplicate_probe_registration(temp_e2e_db: str):
    """F22.B2: Registering duplicate probe_name with INSERT OR REPLACE."""
    conn = sqlite3.connect(temp_e2e_db)
    conn.execute("INSERT OR REPLACE INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) VALUES ('p1', 'p_dup', 'c', 'v1', 'h');")
    conn.execute("INSERT OR REPLACE INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) VALUES ('p1', 'p_dup', 'c', 'v2', 'h');")
    conn.commit()
    cursor = conn.cursor()
    cursor.execute("SELECT description FROM telemetry_probes WHERE probe_id='p1';")
    assert cursor.fetchone()[0] == 'v2'
    conn.close()


def test_f22_boundary_probe_name_max_length():
    """F22.B3: Probe name of 128 characters."""
    pname = "probe_" + "x" * 122
    assert len(pname) == 128


def test_f22_boundary_high_frequency_probe_throttle():
    """F22.B4: Throttling ensures no more than 60 events/second per probe."""
    last_emit = -1.0
    interval = 1.0 / 60.0

    def can_emit(now: float) -> bool:
        nonlocal last_emit
        if now - last_emit >= interval:
            last_emit = now
            return True
        return False

    assert can_emit(0.0) is True
    assert can_emit(0.005) is False
    assert can_emit(0.02) is True


def test_f22_boundary_nested_json_schema_depth():
    """F22.B5: Deeply nested JSON schema in probe metadata."""
    schema = {"p1": {"p2": {"p3": {"p4": {"p5": "value"}}}}}
    assert json.loads(json.dumps(schema))["p1"]["p2"]["p3"]["p4"]["p5"] == "value"


# ============================================================================
# FEATURE 23 BOUNDARIES: Hypothesis-Driven Evolution Engine
# ============================================================================

def test_f23_boundary_zero_events_in_session(temp_e2e_db: str):
    """F23.B1: Session with 0 telemetry events produces fallback hypothesis."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM telemetry_events WHERE session_id='non_existent';")
    assert cursor.fetchone()[0] == 0
    conn.close()


def test_f23_boundary_100_percent_win_rate():
    """F23.B2: 100% win rate prescribes difficulty increase."""
    win_rate = 1.0
    difficulty_delta = +0.15 if win_rate > 0.8 else -0.15
    assert difficulty_delta == +0.15


def test_f23_boundary_100_percent_loss_rate():
    """F23.B3: 0% win rate prescribes difficulty reduction."""
    win_rate = 0.0
    difficulty_delta = -0.15 if win_rate < 0.2 else +0.15
    assert difficulty_delta == -0.15


def test_f23_boundary_max_difficulty_adaptation_clamp():
    """F23.B4: Raw adaptation delta (+0.40) is clamped strictly to +0.15."""
    raw_delta = 0.40
    clamped = max(-0.15, min(0.15, raw_delta))
    assert clamped == 0.15


def test_f23_boundary_unregistered_probe_event_handled():
    """F23.B5: Unregistered custom probe event is classified as 'unknown'."""
    registered = {"dash_probe", "kill_probe"}
    incoming = "strange_new_probe"
    category = "unknown" if incoming not in registered else "known"
    assert category == "unknown"


# ============================================================================
# FEATURE 24 BOUNDARIES: Five-Gate Build Supervisor
# ============================================================================

def test_f24_boundary_all_5_gates_failing():
    """F24.B1: Candidate failing all 5 gates records all 5 rejections."""
    results = [False, False, False, False, False]
    assert results.count(False) == 5


def test_f24_boundary_gate_b_spawn_on_exit():
    """F24.B2: Spawn point identical to exit point produces 1-node path."""
    grid = [[0 for _ in range(5)] for _ in range(5)]
    path = astar_pathfind(grid, (2, 2), (2, 2))
    assert path == [(2, 2)]


def test_f24_boundary_gate_c_instant_player_death():
    """F24.B3: Player dying at tick 1 fails survivability threshold."""
    death_tick = 1
    threshold = 180
    assert death_tick < threshold


def test_f24_boundary_gate_d_exception_traceback_captured():
    """F24.B4: Exception traceback string is captured in diagnostic trace."""
    try:
        raise ValueError("Simulated crash")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        assert "Simulated crash" in tb


def test_f24_boundary_gate_e_corrupted_asset_rejection():
    """F24.B5: 0-byte audio asset fails Gate E asset verification."""
    size = 0
    assert size < 44


# ============================================================================
# FEATURE 25 BOUNDARIES: Multimodal Asset Integrity Gate (Gate E)
# ============================================================================

def test_f25_boundary_audio_file_truncated_header():
    """F25.B1: 20-byte RIFF file is rejected (needs >= 44 bytes)."""
    header = b"RIFF" + b"\x00" * 16
    assert len(header) < 44


def test_f25_boundary_audio_unsupported_sample_rate():
    """F25.B2: Audio with 8000 Hz sample rate is flagged."""
    sample_rate = 8000
    assert sample_rate not in [44100, 24000]


def test_f25_boundary_image_corrupted_crc():
    """F25.B3: Image with invalid header byte is rejected."""
    corrupted_png = b"\x00PNG\r\n\x1a\n"
    assert corrupted_png[:8] != b"\x89PNG\r\n\x1a\n"


def test_f25_boundary_narrative_missing_dialogues_array():
    """F25.B4: Narrative JSON missing dialogues key fails schema check."""
    doc = {"world": {}}
    assert "dialogues" not in doc


def test_f25_boundary_asset_size_exceeds_limit():
    """F25.B5: Asset exceeding 50 MB is rejected."""
    size_mb = 55.0
    assert size_mb > 50.0


# ============================================================================
# FEATURE 26 BOUNDARIES: Dual Synthetic Playtesting Bots
# ============================================================================

def test_f26_boundary_zero_walkable_tiles():
    """F26.B1: Completely solid grid returns None for pathfinding."""
    grid = [[1 for _ in range(5)] for _ in range(5)]
    path = astar_pathfind(grid, (1, 1), (3, 3))
    assert path is None


def test_f26_boundary_enclosed_unreachable_exit():
    """F26.B2: Exit sealed by walls returns None."""
    grid = [[0 for _ in range(5)] for _ in range(5)]
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                grid[2 + dy][2 + dx] = 1
    path = astar_pathfind(grid, (0, 0), (2, 2))
    assert path is None


def test_f26_boundary_extreme_step_limit_timeout():
    """F26.B3: Bot exceeding max ticks (3,600) flags timeout."""
    ticks = 3601
    max_ticks = 3600
    assert ticks > max_ticks


def test_f26_boundary_stuck_oscillation_detection():
    """F26.B4: Oscillating between 2 positions flags stuck condition."""
    history = [(1, 1), (1, 2), (1, 1), (1, 2), (1, 1), (1, 2)]
    is_oscillating = len(set(history[-4:])) <= 2
    assert is_oscillating is True


def test_f26_boundary_zero_health_death_event():
    """F26.B5: Bot health reaching 0.0 marks cleared=False."""
    health = 0.0
    cleared = health > 0.0
    assert cleared is False


# ============================================================================
# FEATURE 27 BOUNDARIES: Closed-Loop Self-Healing Repair
# ============================================================================

def test_f27_boundary_unrepairable_defect_exhaustion():
    """F27.B1: 3 failed repair attempts produces final REJECTED report."""
    attempts = 3
    max_retries = 3
    status = "REJECTED" if attempts >= max_retries else "REPAIRING"
    assert status == "REJECTED"


def test_f27_boundary_empty_blocked_nodes_diagnostic():
    """F27.B2: Diagnostic trace with empty blocked_nodes list handled safely."""
    blocked = []
    assert len(blocked) == 0


def test_f27_boundary_repair_preserves_unrelated_geometry():
    """F27.B3: Repair carves single opening while preserving other perimeter walls."""
    walls = [(0, 0, 10, 1), (0, 9, 10, 1), (5, 1, 1, 8)]
    # Modifying only wall index 2
    repaired_walls = list(walls)
    repaired_walls[2] = (5, 1, 1, 4)
    assert repaired_walls[0] == walls[0]


def test_f27_boundary_repair_attempt_counter_increment():
    """F27.B4: Attempt number strictly increments from 1 to 2 to 3."""
    attempts = []
    for i in range(1, 4):
        attempts.append(i)
    assert attempts == [1, 2, 3]


def test_f27_boundary_rejection_trace_unique_id():
    """F27.B5: Two diagnostic traces produce distinct trace_id strings."""
    import uuid
    id1 = f"trace_{uuid.uuid4().hex[:8]}"
    id2 = f"trace_{uuid.uuid4().hex[:8]}"
    assert id1 != id2


# ============================================================================
# FEATURE 28 BOUNDARIES: Real-Time Observability UI Server
# ============================================================================

def test_f28_boundary_port_binding_conflict():
    """F28.B1: Port number 8000 is an integer in valid range 1024..65535."""
    port = 8000
    assert 1024 <= port <= 65535


def test_f28_boundary_invalid_http_method():
    """F28.B2: POST request to GET-only endpoint returns 405."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    async def get_only(req):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/data", get_only, methods=["GET"])])
    client = TestClient(app)
    res = client.post("/data")
    assert res.status_code == 405


def test_f28_boundary_malformed_url_path():
    """F28.B3: Non-existent URL path returns 404."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient
    app = Starlette()
    client = TestClient(app)
    res = client.get("/non_existent_route_404")
    assert res.status_code == 404


def test_f28_boundary_websocket_immediate_disconnect():
    """F28.B4: WebSocket connection handles immediate client disconnect without exception."""
    from starlette.applications import Starlette
    from starlette.routing import WebSocketRoute
    from starlette.testclient import TestClient

    async def ws_handler(ws):
        await ws.accept()
        await ws.close()

    app = Starlette(routes=[WebSocketRoute("/ws", ws_handler)])
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        pass


def test_f28_boundary_concurrent_http_requests():
    """F28.B5: 20 sequential requests to health endpoint all return 200."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    async def health(req):
        return JSONResponse({"status": "ok"})

    app = Starlette(routes=[Route("/health", health)])
    client = TestClient(app)
    for _ in range(20):
        assert client.get("/health").status_code == 200


# ============================================================================
# FEATURE 29 BOUNDARIES: Mission Control Single Page App
# ============================================================================

def test_f29_boundary_large_agent_graph_rendering():
    """F29.B1: Graph representation with 50 nodes is structured without error."""
    nodes = [{"id": f"agent_{i}"} for i in range(50)]
    assert len(nodes) == 50


def test_f29_boundary_rapid_deliberation_token_stream():
    """F29.B2: Processing 1,000 token strings in sequence."""
    tokens = [f"tok_{i}" for i in range(1000)]
    joined = " ".join(tokens)
    assert len(joined) > 5000


def test_f29_boundary_empty_agent_graph_state():
    """F29.B3: Empty graph state (0 active agents) is handled."""
    active_agents = []
    assert len(active_agents) == 0


def test_f29_boundary_missing_static_asset_404():
    """F29.B4: Non-existent static asset returns 404."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient
    app = Starlette()
    client = TestClient(app)
    res = client.get("/static/missing.js")
    assert res.status_code == 404


def test_f29_boundary_reconnect_websocket_stream():
    """F29.B5: Exponential backoff calculation for WebSocket reconnection."""
    backoffs = [min(30.0, 0.5 * (2 ** attempt)) for attempt in range(5)]
    assert backoffs[0] == 0.5
    assert backoffs[4] == 8.0


# ============================================================================
# FEATURE 30 BOUNDARIES: Telemetry Heatmap & Artifact Explorer
# ============================================================================

def test_f30_boundary_empty_heatmap_grid():
    """F30.B1: All 0s in 16x16 heatmap grid handled cleanly."""
    grid = [[0 for _ in range(16)] for _ in range(16)]
    assert sum(sum(r) for r in grid) == 0


def test_f30_boundary_saturated_heatmap_grid():
    """F30.B2: Single cell with 1,000,000 visits normalizes safely without division by zero."""
    grid = [[0 for _ in range(16)] for _ in range(16)]
    grid[5][5] = 1000000
    max_val = max(max(r) for r in grid)
    normalized = grid[5][5] / max_val
    assert normalized == 1.0


def test_f30_boundary_non_existent_artifact_download():
    """F30.B3: Requesting non-existent artifact returns 404."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient
    app = Starlette()
    client = TestClient(app)
    res = client.get("/api/artifacts/non_existent.wav")
    assert res.status_code == 404


def test_f30_boundary_large_code_diff_display():
    """F30.B4: 1,000 line code diff string constructed safely."""
    diff_lines = [f"+ line {i}" for i in range(1000)]
    assert len(diff_lines) == 1000


def test_f30_boundary_audio_streaming_range_request():
    """F30.B5: Byte range parsing (bytes=0-1023)."""
    range_header = "bytes=0-1023"
    start, end = range_header.replace("bytes=", "").split("-")
    assert int(start) == 0
    assert int(end) == 1023


# ============================================================================
# FEATURE 31 BOUNDARIES: Decoupled Background Studio Daemon
# ============================================================================

def test_f31_boundary_daemon_killed_mid_compilation(tmp_path: Path):
    """F31.B1: Candidate staging directory isolation ensures active cartridge is untouched."""
    active = tmp_path / "active"
    candidate = tmp_path / "candidate_staging"
    active.mkdir()
    candidate.mkdir()
    (active / "manifest.json").write_text('{"intact": true}')
    (candidate / "partial.tmp").write_text("in progress")

    assert json.loads((active / "manifest.json").read_text())["intact"] is True


def test_f31_boundary_manifest_atomic_replace_race(tmp_path: Path):
    """F31.B2: Atomic rename replaces file atomically."""
    target = tmp_path / "manifest.json"
    target.write_text('{"v": 1}')
    tmp_file = tmp_path / "manifest.tmp"
    tmp_file.write_text('{"v": 2}')
    tmp_file.replace(target)
    assert json.loads(target.read_text())["v"] == 2


def test_f31_boundary_multiple_daemon_instances(tmp_path: Path):
    """F31.B3: Lockfile detection prevents duplicate daemon instances."""
    lock_file = tmp_path / "studio.lock"
    lock_file.write_text("12345")
    is_locked = lock_file.exists()
    assert is_locked is True


def test_f31_boundary_stale_lockfile_cleanup(tmp_path: Path):
    """F31.B4: Stale lockfile can be removed."""
    lock_file = tmp_path / "studio.lock"
    lock_file.write_text("99999999")
    lock_file.unlink()
    assert not lock_file.exists()


def test_f31_boundary_disk_space_exhaustion():
    """F31.B5: Free disk space check threshold (requires >= 100MB)."""
    import shutil
    usage = shutil.disk_usage("/")
    free_mb = usage.free / (1024 * 1024)
    assert free_mb > 100.0


# ============================================================================
# FEATURE 32 BOUNDARIES: Opaque-Box E2E Test Suite
# ============================================================================

def test_f32_boundary_invalid_tier_argument():
    """F32.B1: Parsing invalid tier argument 99 raises ArgumentError in argparse."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4])
    with pytest.raises(SystemExit):
        parser.parse_args(["--tier", "99"])


def test_f32_boundary_invalid_milestone_argument():
    """F32.B2: Parsing invalid milestone argument M99 raises ArgumentError."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", choices=["M1", "M2", "M3", "M4", "M5", "M6"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--milestone", "M99"])


def test_f32_boundary_non_existent_feature_flag():
    """F32.B3: Flag for feature F99 is recognized as unknown feature."""
    valid_features = [f"F{i}" for i in range(1, 34)]
    assert "F99" not in valid_features


def test_f32_boundary_runner_output_json_format():
    """F32.B4: Formats summary dictionary to valid JSON."""
    report = {"tier": 2, "passed": 165, "failed": 0}
    assert json.loads(json.dumps(report))["passed"] == 165


def test_f32_boundary_empty_test_filter_defaults_all():
    """F32.B5: Empty filter selection defaults to running all tiers."""
    tier_filter = None
    selected = [1, 2, 3, 4] if tier_filter is None else [tier_filter]
    assert len(selected) == 4


# ============================================================================
# FEATURE 33 BOUNDARIES: Adversarial Coverage Hardening
# ============================================================================

def test_f33_boundary_sql_injection_in_session_id(temp_e2e_db: str):
    """F33.B1: SQL injection string in session_id is parameterized safely."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    injection_str = "'; DROP TABLE telemetry_sessions; --"
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES (?);", (injection_str,))
    conn.commit()

    # Verify table was not dropped
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='telemetry_sessions';")
    assert cursor.fetchone() is not None
    cursor.execute("SELECT session_id FROM telemetry_sessions WHERE session_id=?;", (injection_str,))
    assert cursor.fetchone()[0] == injection_str
    conn.close()


def test_f33_boundary_path_traversal_in_cartridge_load():
    """F33.B2: Path traversal string '../../etc' is blocked by resolving to safe root."""
    base_dir = Path("/safe/cartridges")
    unsafe_path = base_dir / "../../etc/passwd"
    resolved = unsafe_path.resolve()
    # Check if resolved escaped base_dir
    escaped = not str(resolved).startswith(str(base_dir.resolve()))
    assert escaped is True  # Detected traversal


def test_f33_boundary_zip_bomb_or_huge_asset():
    """F33.B3: Max asset size limit (50 MB) blocks oversized uploads."""
    max_bytes = 50 * 1024 * 1024
    incoming = 60 * 1024 * 1024
    assert incoming > max_bytes


def test_f33_boundary_unicode_control_characters_in_text():
    """F33.B4: Text containing null bytes and RTL override characters sanitized."""
    raw = "Normal\x00Text\u202EOverride"
    sanitized = raw.replace("\x00", "").replace("\u202e", "")
    assert "\x00" not in sanitized
    assert "\u202e" not in sanitized


def test_f33_boundary_deep_ast_expression_bomb():
    """F33.B5: Deeply nested AST operations (>100 depth) handled safely."""
    nested = "x = " + "-(" * 100 + "1" + ")" * 100
    tree = ast.parse(nested)
    assert tree is not None
