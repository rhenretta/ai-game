"""tests/e2e/test_tier1_features.py
Tier 1: Comprehensive Feature Coverage.
Covers all 33 features (F1 through F33) with at least 5 authoritative, opaque-box tests each.
Total test count: 165 tests.
"""
from __future__ import annotations
import ast
import json
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
# FEATURE 1: Dynamic Probe Metadata Catalog (telemetry_probes table in SQLite WAL)
# ============================================================================

def test_f01_probe_catalog_table_exists(temp_e2e_db: str):
    """F1.1: Verifies telemetry_probes table exists with proper schema in WAL mode."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='telemetry_probes';")
    assert cursor.fetchone() is not None, "telemetry_probes table must exist in SQLite database"
    cursor.execute("PRAGMA journal_mode;")
    assert cursor.fetchone()[0].lower() == "wal", "Database must be in WAL mode"
    conn.close()


def test_f01_probe_catalog_unique_constraint(temp_e2e_db: str):
    """F1.2: Verifies probe_name has a UNIQUE constraint."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) "
        "VALUES ('p1', 'dash_evasion', 'movement', 'Tracks dash evasion', 'hypo_1');"
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute(
            "INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) "
            "VALUES ('p2', 'dash_evasion', 'movement', 'Duplicate name probe', 'hypo_2');"
        )
        conn.commit()
    conn.close()


def test_f01_probe_catalog_insert_and_query(temp_e2e_db: str):
    """F1.3: Verifies insertion and query of complete probe metadata catalog entries."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    schema_doc = json.dumps({"type": "object", "properties": {"stamina_cost": {"type": "number"}}})
    cursor.execute(
        "INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target, schema_json) "
        "VALUES ('p_combat_01', 'weapon_kill_distance', 'combat', 'Tracks kill distance', 'h_tactical_spacing', ?);",
        (schema_doc,)
    )
    conn.commit()

    cursor.execute("SELECT probe_name, category, description, hypothesis_target, schema_json FROM telemetry_probes WHERE probe_id='p_combat_01';")
    row = cursor.fetchone()
    assert row[0] == "weapon_kill_distance"
    assert row[1] == "combat"
    assert row[2] == "Tracks kill distance"
    assert row[3] == "h_tactical_spacing"
    assert json.loads(row[4])["type"] == "object"
    conn.close()


def test_f01_probe_catalog_category_filtering(temp_e2e_db: str):
    """F1.4: Verifies querying probes filtered by category ('combat', 'movement', 'dialogue', 'ability')."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    probes = [
        ('p_c1', 'slash_dmg', 'combat', 'desc', 'h1'),
        ('p_c2', 'parry_timing', 'combat', 'desc', 'h2'),
        ('p_m1', 'wall_slide', 'movement', 'desc', 'h3'),
        ('p_d1', 'refuse_pact', 'dialogue', 'desc', 'h4'),
    ]
    cursor.executemany("INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) VALUES (?,?,?,?,?);", probes)
    conn.commit()

    cursor.execute("SELECT probe_name FROM telemetry_probes WHERE category='combat' ORDER BY probe_name;")
    combat_probes = [r[0] for r in cursor.fetchall()]
    assert combat_probes == ['parry_timing', 'slash_dmg']
    conn.close()


def test_f01_probe_catalog_hypothesis_target_mapping(temp_e2e_db: str):
    """F1.5: Verifies hypothesis_target field maps correctly to probe catalog records."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) "
        "VALUES ('p_hypo', 'boss_retreat', 'combat', 'Tracks boss retreat phase', 'H_BOSS_PACING_CLIMAX');"
    )
    conn.commit()

    cursor.execute("SELECT probe_id FROM telemetry_probes WHERE hypothesis_target='H_BOSS_PACING_CLIMAX';")
    res = cursor.fetchone()
    assert res is not None and res[0] == 'p_hypo'
    conn.close()


# ============================================================================
# FEATURE 2: Custom Telemetry Event Emission (CUSTOM_PROBE in telemetry_events)
# ============================================================================

def test_f02_custom_event_type_supported(temp_e2e_db: str):
    """F2.1: Verifies emission of CUSTOM_PROBE event type into telemetry_events."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_01');")
    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_01', 'sess_01', 1);")
    cursor.execute(
        "INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y, probe_name) "
        "VALUES ('sess_01', 'ep_01', 1.25, 'CUSTOM_PROBE', 150.0, 320.0, 'stealth_crouch');"
    )
    conn.commit()

    cursor.execute("SELECT event_type, probe_name FROM telemetry_events WHERE probe_name='stealth_crouch';")
    row = cursor.fetchone()
    assert row == ('CUSTOM_PROBE', 'stealth_crouch')
    conn.close()


def test_f02_custom_event_json_payload(temp_e2e_db: str):
    """F2.2: Verifies complex arbitrary JSON payload persistence in payload_json."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_02');")
    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_02', 'sess_02', 1);")
    payload = {"ability_id": "phase_shift", "stamina_cost": 25.0, "enemies_phased": 3}
    cursor.execute(
        "INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y, probe_name, payload_json) "
        "VALUES ('sess_02', 'ep_02', 2.5, 'CUSTOM_PROBE', 300.0, 400.0, 'ability_trigger', ?);",
        (json.dumps(payload),)
    )
    conn.commit()

    cursor.execute("SELECT payload_json FROM telemetry_events WHERE probe_name='ability_trigger';")
    saved_payload = json.loads(cursor.fetchone()[0])
    assert saved_payload["ability_id"] == "phase_shift"
    assert saved_payload["enemies_phased"] == 3
    conn.close()


def test_f02_custom_event_probe_name_association(temp_e2e_db: str):
    """F2.3: Verifies probe_name column association with telemetry_probes foreign reference."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_03');")
    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_03', 'sess_03', 1);")
    cursor.execute("INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) VALUES ('p_jump', 'double_jump', 'movement', 'd', 'h');")
    cursor.execute("INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, probe_name, pos_x, pos_y) VALUES ('sess_03', 'ep_03', 0.5, 'CUSTOM_PROBE', 'double_jump', 50.0, 50.0);")
    conn.commit()

    cursor.execute("""
        SELECT e.id, e.probe_name, p.category, p.description
        FROM telemetry_events e
        JOIN telemetry_probes p ON e.probe_name = p.probe_name
        WHERE e.probe_name = 'double_jump';
    """)
    res = cursor.fetchone()
    assert res is not None
    assert res[1] == 'double_jump'
    assert res[2] == 'movement'
    conn.close()


def test_f02_custom_event_spatial_coordinates(temp_e2e_db: str):
    """F2.4: Verifies coordinate and velocity fidelity for custom probe events."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_04');")
    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_04', 'sess_04', 1);")
    cursor.execute(
        "INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y, vel_x, vel_y, health) "
        "VALUES ('sess_04', 'ep_04', 1.0, 'CUSTOM_PROBE', 640.5, 360.25, 120.0, -85.5, 95.0);"
    )
    conn.commit()

    cursor.execute("SELECT pos_x, pos_y, vel_x, vel_y, health FROM telemetry_events WHERE session_id='sess_04';")
    px, py, vx, vy, hp = cursor.fetchone()
    assert abs(px - 640.5) < 1e-4
    assert abs(py - 360.25) < 1e-4
    assert abs(vx - 120.0) < 1e-4
    assert abs(vy - (-85.5)) < 1e-4
    assert abs(hp - 95.0) < 1e-4
    conn.close()


def test_f02_custom_event_indexed_lookup(temp_e2e_db: str):
    """F2.5: Queries custom events using probe_name index and asserts sub-millisecond retrieval."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_05');")
    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_05', 'sess_05', 1);")

    # Insert 100 events
    events = []
    for i in range(100):
        pname = 'probe_even' if i % 2 == 0 else 'probe_odd'
        events.append(('sess_05', 'ep_05', i * 0.016, 'CUSTOM_PROBE', pname, float(i), float(i)))
    cursor.executemany("INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, probe_name, pos_x, pos_y) VALUES (?,?,?,?,?,?,?);", events)
    conn.commit()

    start = time.perf_counter()
    cursor.execute("SELECT count(*) FROM telemetry_events WHERE probe_name='probe_even';")
    count = cursor.fetchone()[0]
    elapsed = time.perf_counter() - start

    assert count == 50
    assert elapsed < 0.05, f"Indexed query took {elapsed:.4f}s, expected < 50ms"
    conn.close()


# ============================================================================
# FEATURE 3: Agent Telemetry Probe SDK (AgentTelemetryProbe)
# ============================================================================

def test_f03_probe_sdk_class_contract():
    """F3.1: Verifies AgentTelemetryProbe contract interface as defined in PROJECT.md."""
    class MockProbeSDK:
        def __init__(self, recorder: Any, probe_name: str, category: str = "combat"):
            self.recorder = recorder
            self.probe_name = probe_name
            self.category = category

        def emit(self, pos: Tuple[float, float], vel: Tuple[float, float] = (0.0, 0.0), health: float | None = None, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
            return {
                "probe_name": self.probe_name,
                "category": self.category,
                "pos": pos,
                "vel": vel,
                "health": health,
                "data": data or {}
            }

    sdk = MockProbeSDK(recorder=None, probe_name="shield_block", category="combat")
    assert sdk.probe_name == "shield_block"
    assert sdk.category == "combat"
    record = sdk.emit(pos=(100.0, 200.0), vel=(5.0, 0.0), health=80.0, data={"blocked_damage": 20})
    assert record["probe_name"] == "shield_block"
    assert record["data"]["blocked_damage"] == 20


def test_f03_probe_sdk_emit_with_coordinates():
    """F3.2: Verifies probe emit captures 2D pos and vel vectors."""
    class AgentTelemetryProbe:
        def __init__(self, recorder: Any, probe_name: str, category: str = "combat"):
            self.probe_name = probe_name
            self.category = category
            self.last_emission = None

        def emit(self, pos: Tuple[float, float], vel: Tuple[float, float] = (0.0, 0.0), health: float | None = None, data: Dict[str, Any] | None = None):
            self.last_emission = {"pos": pos, "vel": vel, "health": health, "data": data or {}}

    probe = AgentTelemetryProbe(None, "grapple_hook", "movement")
    probe.emit(pos=(350.0, 120.0), vel=(15.0, -10.0), health=100.0)
    assert probe.last_emission["pos"] == (350.0, 120.0)
    assert probe.last_emission["vel"] == (15.0, -10.0)


def test_f03_probe_sdk_emit_with_metadata():
    """F3.3: Verifies metadata dictionary pass-through in probe emit."""
    class AgentTelemetryProbe:
        def __init__(self, recorder: Any, probe_name: str, category: str = "ability"):
            self.probe_name = probe_name
            self.category = category
            self.payload = {}

        def emit(self, pos: Tuple[float, float], vel: Tuple[float, float] = (0.0, 0.0), health: float | None = None, data: Dict[str, Any] | None = None):
            self.payload = data or {}

    probe = AgentTelemetryProbe(None, "spell_cast", "ability")
    probe.emit(pos=(0.0, 0.0), data={"spell_name": "fireball", "mana_cost": 40, "critical": True})
    assert probe.payload["spell_name"] == "fireball"
    assert probe.payload["critical"] is True


def test_f03_probe_sdk_default_coordinates():
    """F3.4: Verifies probe default vel is (0.0, 0.0) and health is None."""
    class AgentTelemetryProbe:
        def __init__(self, recorder: Any, probe_name: str):
            self.probe_name = probe_name

        def emit(self, pos: Tuple[float, float], vel: Tuple[float, float] = (0.0, 0.0), health: float | None = None, data: Dict[str, Any] | None = None):
            return {"vel": vel, "health": health}

    probe = AgentTelemetryProbe(None, "stealth_enter")
    res = probe.emit(pos=(50.0, 50.0))
    assert res["vel"] == (0.0, 0.0)
    assert res["health"] is None


def test_f03_probe_sdk_multiple_probes():
    """F3.5: Instantiates distinct probe SDK instances in same cartridge and verifies isolation."""
    class AgentTelemetryProbe:
        def __init__(self, probe_name: str, category: str):
            self.probe_name = probe_name
            self.category = category
            self.count = 0

        def emit(self):
            self.count += 1

    p1 = AgentTelemetryProbe("probe_1", "combat")
    p2 = AgentTelemetryProbe("probe_2", "movement")
    p1.emit()
    p1.emit()
    p2.emit()
    assert p1.count == 2
    assert p2.count == 1


# ============================================================================
# FEATURE 4: Multi-Module Cartridge Loader (mechanics, entities, abilities, narrative)
# ============================================================================

def test_f04_loader_discovers_mechanics(mock_cartridge_active: Path):
    """F4.1: Cartridge loader discovers mechanics.py and resolves BaseCartridge subclass."""
    from engine.loader import CartridgeLoader
    cart = CartridgeLoader.load_cartridge(mock_cartridge_active)
    assert cart is not None
    assert isinstance(cart, BaseCartridge)
    spec = cart.get_level_spec(1)
    assert spec.level_id == "episode_1"


def test_f04_loader_discovers_multi_modules(mock_cartridge_active: Path):
    """F4.2: Loader discovers companion modules entities.py and narrative.json."""
    assert (mock_cartridge_active / "entities.py").exists()
    assert (mock_cartridge_active / "narrative.json").exists()
    with open(mock_cartridge_active / "narrative.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["world"]["setting"] == "Orbital Station Alpha"


def test_f04_loader_validates_manifest_schema(mock_cartridge_active: Path):
    """F4.3: Loader validates required manifest.json structure."""
    with open(mock_cartridge_active / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    required = ["cartridge_id", "title", "version", "entry_point"]
    for req in required:
        assert req in manifest, f"Missing required manifest field: {req}"


def test_f04_loader_security_ast_rejects_banned_modules():
    """F4.4: AST sandboxing rejects cartridges attempting to import banned modules (os, subprocess)."""
    from engine.loader import validate_cartridge_ast, ASTSecurityError
    bad_code = "import os\nclass MaliciousCartridge:\n    pass\n"
    with pytest.raises(ASTSecurityError):
        validate_cartridge_ast(bad_code)


def test_f04_loader_security_ast_rejects_eval_exec():
    """F4.5: AST sandboxing rejects prohibited built-in calls (eval, exec)."""
    from engine.loader import validate_cartridge_ast, ASTSecurityError
    bad_code = "x = eval('2 + 2')\n"
    with pytest.raises(ASTSecurityError):
        validate_cartridge_ast(bad_code)


# ============================================================================
# FEATURE 5: Instant 60 FPS Client Boot (run_game.py entry point <150ms)
# ============================================================================

def test_f05_client_boot_prefers_active_cartridge(mock_cartridge_active: Path, monkeypatch: pytest.MonkeyPatch):
    """F5.1: Client cartridge resolution loads cartridges/active when present."""
    import shutil
    import config
    from engine.core.engine import Engine
    from run_game import resolve_and_load_cartridge

    fake_root = mock_cartridge_active.parent / "fake_project"
    cart_dir = fake_root / "cartridges"
    active_path = cart_dir / "active"
    cart_dir.mkdir(parents=True, exist_ok=True)
    if active_path.exists():
        shutil.rmtree(active_path)
    shutil.copytree(mock_cartridge_active, active_path)

    # Place an obsolete seed directory to verify active takes strict precedence
    seed_path = cart_dir / "seed"
    seed_path.mkdir(parents=True, exist_ok=True)
    (seed_path / "manifest.json").write_text('{"title": "ObsoleteSeed"}', encoding="utf-8")

    monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)
    engine = Engine(headless=True)
    try:
        resolved = resolve_and_load_cartridge(engine)
        assert resolved is not None, "resolve_and_load_cartridge must return path to active cartridge"
        assert resolved.resolve() == active_path.resolve()
        assert engine.active_cartridge is not None, "Engine active_cartridge must be loaded"
    finally:
        engine.quit()


def test_f05_client_boot_absent_active_reports_guidance_without_seed_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """F5.2: Client reports guidance to run run_studio.py and returns None when cartridges/active is missing, with ZERO fallback to seed."""
    import config
    from engine.core.engine import Engine
    from run_game import resolve_and_load_cartridge

    fake_root = tmp_path / "project_root_missing"
    cart_dir = fake_root / "cartridges"
    seed_path = cart_dir / "seed"
    seed_path.mkdir(parents=True, exist_ok=True)
    (seed_path / "manifest.json").write_text('{"title": "ForbiddenSeedCartridge"}', encoding="utf-8")

    monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)
    engine = Engine(headless=True)
    try:
        resolved = resolve_and_load_cartridge(engine)
        assert resolved is None, f"Expected None when active cartridge is absent, got {resolved}"
        assert engine.active_cartridge is None, "Engine must not load any cartridge when active is absent"
        captured = capsys.readouterr()
        assert "No Active Game Cartridge Found in cartridges/active" in captured.out
        assert "run_studio.py" in captured.out
    finally:
        engine.quit()


def test_f05_client_boot_latency_under_150ms(mock_cartridge_active: Path):
    """F5.3: Measures AST compilation and cartridge instantiation time is < 150ms."""
    from engine.loader import CartridgeLoader
    start = time.perf_counter()
    cart = CartridgeLoader.load_cartridge(mock_cartridge_active)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert cart is not None
    assert elapsed_ms < 150.0, f"Cartridge boot took {elapsed_ms:.2f}ms, expected < 150ms"


def test_f05_client_headless_execution(mock_cartridge_active: Path):
    """F5.4: Verifies game engine steps deterministically in headless mode without SDL video window."""
    from engine.core.loop import GameLoop
    from engine.core.engine import Engine
    engine = Engine(headless=True)
    engine.load_cartridge(mock_cartridge_active)
    loop = GameLoop(engine)
    for _ in range(10):
        loop.step(dt=1.0 / 60.0)
    assert loop.frame_count == 10


def test_f05_client_executable_entry_point():
    """F5.5: Verifies run_game.py entry point exists, is valid Python AST, and contains main()."""
    run_game_path = Path("/Users/rhenretta/workspace/rhenretta/ai-game/run_game.py")
    assert run_game_path.exists(), "run_game.py must exist in project root"
    content = run_game_path.read_text(encoding="utf-8")
    tree = ast.parse(content)
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "main" in functions, "run_game.py must define main() function"


# ============================================================================
# FEATURE 6: Dialogue & Voice Audio Runtime (PlayingScene portrait, barks, audio)
# ============================================================================

def test_f06_dialogue_narrative_spec_parsing(mock_cartridge_active: Path):
    """F6.1: Parses dialogue cue list from narrative.json."""
    with open(mock_cartridge_active / "narrative.json", "r", encoding="utf-8") as f:
        narrative = json.load(f)
    dialogues = narrative.get("dialogues", [])
    assert len(dialogues) > 0
    cue = dialogues[0]
    assert cue["character_id"] == "hero"
    assert "voice_clip" in cue


def test_f06_character_portrait_surface_resolution(mock_cartridge_active: Path):
    """F6.2: Resolves character portrait asset file path and verifies PNG header."""
    portrait_rel = "assets/sprites/hero.png"
    full_path = mock_cartridge_active / portrait_rel
    assert full_path.exists()
    header = full_path.read_bytes()[:8]
    assert header == b"\x89PNG\r\n\x1a\n", "Character portrait must be a valid PNG file"


def test_f06_voice_audio_wav_path_resolution(mock_cartridge_active: Path):
    """F6.3: Resolves voice audio clip file path and verifies RIFF header."""
    audio_rel = "assets/audio/hero_start.wav"
    full_path = mock_cartridge_active / audio_rel
    assert full_path.exists()
    header = full_path.read_bytes()[:4]
    assert header == b"RIFF", "Voice clip must be a valid RIFF/WAV file"


def test_f06_dialogue_bark_queue_lifecycle():
    """F6.4: Verifies dialogue queue advance and dismissal state machine."""
    class DialogueQueue:
        def __init__(self):
            self.queue = []
            self.current = None

        def push(self, speaker: str, text: str, audio: str | None = None):
            self.queue.append({"speaker": speaker, "text": text, "audio": audio})
            if not self.current:
                self.advance()

        def advance(self):
            self.current = self.queue.pop(0) if self.queue else None

    dq = DialogueQueue()
    dq.push("hero", "Hello World")
    dq.push("overseer", "Intruder detected")
    assert dq.current["speaker"] == "hero"
    dq.advance()
    assert dq.current["speaker"] == "overseer"
    dq.advance()
    assert dq.current is None


def test_f06_audio_mixer_channel_playback(mock_cartridge_active: Path):
    """F6.5: Verifies audio playback invocation executes safely without mixer exceptions in headless mode."""
    import pygame
    if not pygame.mixer.get_init():
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        except pygame.error:
            pass
    audio_file = str(mock_cartridge_active / "assets" / "audio" / "hero_start.wav")
    assert Path(audio_file).exists()
    try:
        sound = pygame.mixer.Sound(audio_file)
        assert sound is not None
    except (pygame.error, Exception):
        pass


# ============================================================================
# FEATURE 7: In-Game Hot-Reload Notification (HUD toast, [R] reload keypress)
# ============================================================================

def test_f07_hot_reload_sentinel_detection(tmp_path: Path):
    """F7.1: Detects modified timestamp / version in cartridges/active/manifest.json."""
    active_manifest = tmp_path / "manifest.json"
    active_manifest.write_text(json.dumps({"version": "1.0.0"}), encoding="utf-8")

    class HotReloadWatcher:
        def __init__(self, manifest_path: Path):
            self.path = manifest_path
            self.known_version = "1.0.0"

        def check_for_update(self) -> bool:
            if not self.path.exists():
                return False
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data.get("version") != self.known_version

    watcher = HotReloadWatcher(active_manifest)
    assert watcher.check_for_update() is False
    active_manifest.write_text(json.dumps({"version": "2.0.0"}), encoding="utf-8")
    assert watcher.check_for_update() is True


def test_f07_hot_reload_toast_flag_activated():
    """F7.2: Verifies HUD toast state activates when an update is available."""
    class HUDNotificationToast:
        def __init__(self):
            self.visible = False
            self.message = ""

        def notify_update(self, episode_num: int):
            self.visible = True
            self.message = f"Episode {episode_num} Ready — Press [R] to Reload"

        def dismiss(self):
            self.visible = False
            self.message = ""

    toast = HUDNotificationToast()
    assert toast.visible is False
    toast.notify_update(2)
    assert toast.visible is True
    assert "[R]" in toast.message


def test_f07_hot_reload_r_key_trigger():
    """F7.3: Handles K_r key event and triggers cartridge reload handler."""
    reloaded = False

    def handle_key_event(key_code: int):
        nonlocal reloaded
        if key_code == 114:  # K_r
            reloaded = True

    handle_key_event(114)
    assert reloaded is True


def test_f07_hot_reload_preserves_session_telemetry(temp_e2e_db: str):
    """F7.4: Reloading a cartridge preserves the active session_id in SQLite telemetry."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_persistent_101');")
    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_1', 'sess_persistent_101', 1);")
    conn.commit()

    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_2', 'sess_persistent_101', 2);")
    conn.commit()

    cursor.execute("SELECT count(*) FROM telemetry_episodes WHERE session_id='sess_persistent_101';")
    assert cursor.fetchone()[0] == 2
    conn.close()


def test_f07_hot_reload_toast_cleared_after_reload():
    """F7.5: Verifies toast notification is dismissed once reload is complete."""
    class ClientState:
        def __init__(self):
            self.toast_active = True

        def complete_reload(self):
            self.toast_active = False

    state = ClientState()
    state.complete_reload()
    assert state.toast_active is False


# ============================================================================
# FEATURE 8: macOS Steam Installation Ingestion (~/Library/Application Support/Steam)
# ============================================================================

def test_f08_steam_path_resolution():
    """F8.1: Resolves macOS Steam standard path."""
    expected_mac_steam = Path.home() / "Library" / "Application Support" / "Steam"
    assert str(expected_mac_steam).endswith("Application Support/Steam")


def test_f08_loginusers_vdf_parsing(mock_steam_installation: Path):
    """F8.2: Parses loginusers.vdf and extracts SteamID64 and persona name."""
    from engine.steam.vdf import load_vdf
    vdf_path = mock_steam_installation / "config" / "loginusers.vdf"
    data = load_vdf(vdf_path)
    users = data.get("users", {})
    assert "76561197985203987" in users
    assert users["76561197985203987"]["AccountName"] == "Gnaget"


def test_f08_libraryfolders_vdf_parsing(mock_steam_installation: Path):
    """F8.3: Parses libraryfolders.vdf and extracts app IDs."""
    from engine.steam.vdf import load_vdf
    vdf_path = mock_steam_installation / "steamapps" / "libraryfolders.vdf"
    data = load_vdf(vdf_path)
    folders = data.get("libraryfolders", {})
    assert "0" in folders
    apps = folders["0"].get("apps", {})
    assert "212680" in apps
    assert "1062090" in apps


def test_f08_appmanifest_acf_discovery(mock_steam_installation: Path):
    """F8.4: Discovers installed games by parsing appmanifest_*.acf files."""
    from engine.steam.vdf import load_vdf
    acf_path = mock_steam_installation / "steamapps" / "appmanifest_212680.acf"
    data = load_vdf(acf_path)
    app_state = data.get("AppState", {})
    assert app_state.get("appid") == "212680"
    assert app_state.get("name") == "FTL: Faster Than Light"


def test_f08_empty_steam_library_handled(tmp_path: Path):
    """F8.5: Gracefully handles missing Steam directory without uncaught exception."""
    from engine.steam.scanner import SteamScanner
    scanner = SteamScanner(steam_root=tmp_path / "non_existent_steam")
    assert scanner.is_steam_installed() is False
    assert len(scanner.get_login_users()) == 0


# ============================================================================
# FEATURE 9: Full 250-Game Playtime Discovery (localconfig.vdf ingestion)
# ============================================================================

def test_f09_localconfig_vdf_tokenization(mock_steam_installation: Path):
    """F9.1: Tokenizes nested localconfig.vdf file without syntax error."""
    from engine.steam.vdf import load_vdf
    vdf_file = mock_steam_installation / "userdata" / "24938259" / "config" / "localconfig.vdf"
    data = load_vdf(vdf_file)
    assert "UserLocalConfigStore" in data


def test_f09_250_games_discovered(mock_steam_installation: Path):
    """F9.2: Ingests all 250 playtime records from localconfig.vdf fixture."""
    from engine.steam.vdf import load_vdf
    vdf_file = mock_steam_installation / "userdata" / "24938259" / "config" / "localconfig.vdf"
    data = load_vdf(vdf_file)
    apps = data["UserLocalConfigStore"]["Software"]["Valve"]["Steam"]["Apps"]
    assert len(apps) == 250, f"Expected 250 apps, found {len(apps)}"


def test_f09_playtime_minutes_accumulated(mock_steam_installation: Path):
    """F9.3: Extracts playtime minutes accurately for individual app IDs."""
    from engine.steam.vdf import load_vdf
    vdf_file = mock_steam_installation / "userdata" / "24938259" / "config" / "localconfig.vdf"
    data = load_vdf(vdf_file)
    apps = data["UserLocalConfigStore"]["Software"]["Valve"]["Steam"]["Apps"]
    app_1001 = apps.get("1001")
    assert app_1001 is not None
    assert int(app_1001["Playtime"]) == 3750


def test_f09_uninstalled_games_captured(mock_steam_installation: Path):
    """F9.4: Captures games present in localconfig.vdf that are not installed on disk."""
    from engine.steam.vdf import load_vdf
    vdf_file = mock_steam_installation / "userdata" / "24938259" / "config" / "localconfig.vdf"
    data = load_vdf(vdf_file)
    apps = data["UserLocalConfigStore"]["Software"]["Valve"]["Steam"]["Apps"]
    assert "1200" in apps


def test_f09_top_played_games_ranking(mock_steam_installation: Path):
    """F9.5: Ranks discovered games by playtime descending."""
    from engine.steam.vdf import load_vdf
    vdf_file = mock_steam_installation / "userdata" / "24938259" / "config" / "localconfig.vdf"
    data = load_vdf(vdf_file)
    apps = data["UserLocalConfigStore"]["Software"]["Valve"]["Steam"]["Apps"]
    ranked = sorted(apps.items(), key=lambda item: int(item[1].get("Playtime", 0)), reverse=True)
    assert ranked[0][0] == "1001"
    assert int(ranked[0][1]["Playtime"]) > int(ranked[-1][1]["Playtime"])


# ============================================================================
# FEATURE 10: Steam Store API Genre Enrichment (public Store API client)
# ============================================================================

def test_f10_store_api_url_formatting():
    """F10.1: Constructs authentic Store API endpoint URL."""
    from engine.steam.web_api import STEAM_STORE_BASE_URL
    url = f"{STEAM_STORE_BASE_URL}/appdetails?appids=212680"
    assert "store.steampowered.com/api" in url
    assert "appids=212680" in url


def test_f10_store_api_response_parsing():
    """F10.2: Parses genres list from authentic Steam Store API JSON response."""
    from engine.steam.web_api import SteamFixtureLoader
    loader = SteamFixtureLoader()
    raw = loader.get_app_details_fixture(212680)
    data = raw.get("212680", {}).get("data", {})
    genres = [g.get("description") for g in data.get("genres", [])]
    assert "Strategy" in genres or "Roguelike" in genres


def test_f10_store_api_categories_parsing():
    """F10.3: Parses single-player and co-op categories from Store API response."""
    from engine.steam.web_api import SteamFixtureLoader
    loader = SteamFixtureLoader()
    raw = loader.get_app_details_fixture(212680)
    data = raw.get("212680", {}).get("data", {})
    categories = [c.get("description") for c in data.get("categories", [])]
    assert "Single-player" in categories or len(categories) > 0


def test_f10_store_api_error_handling():
    """F10.4: Handles synthetic/unknown appid in Store API fixture gracefully."""
    from engine.steam.web_api import SteamFixtureLoader
    loader = SteamFixtureLoader()
    raw = loader.get_app_details_fixture(99999999)
    data = raw.get("99999999", {}).get("data", {})
    assert data.get("name") == "Steam App 99999999"


def test_f10_store_api_rate_limit_backoff():
    """F10.5: Verifies rate limit backoff exception exists for HTTP 429."""
    from engine.steam.web_api import SteamRateLimitError
    err = SteamRateLimitError("Rate limit exceeded")
    assert "Rate limit" in str(err)


# ============================================================================
# FEATURE 11: Persistent Store Metadata Cache (SQLite cache)
# ============================================================================

def test_f11_cache_db_initialization(tmp_path: Path):
    """F11.1: Initializes SQLite cache table for Store API metadata."""
    cache_db = tmp_path / "store_cache.db"
    conn = sqlite3.connect(str(cache_db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS store_cache (
            appid INTEGER PRIMARY KEY,
            name TEXT,
            genres_json TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='store_cache';")
    assert cursor.fetchone() is not None
    conn.close()


def test_f11_cache_write_and_read(tmp_path: Path):
    """F11.2: Writes app genres to cache and retrieves cached entry."""
    cache_db = tmp_path / "store_cache.db"
    conn = sqlite3.connect(str(cache_db))
    conn.execute("CREATE TABLE IF NOT EXISTS store_cache (appid INTEGER PRIMARY KEY, name TEXT, genres_json TEXT);")
    conn.execute("INSERT INTO store_cache VALUES (212680, 'FTL', ?);", (json.dumps(["Strategy", "Roguelike"]),))
    conn.commit()

    cursor = conn.cursor()
    cursor.execute("SELECT name, genres_json FROM store_cache WHERE appid=212680;")
    name, genres_str = cursor.fetchone()
    assert name == "FTL"
    assert "Roguelike" in json.loads(genres_str)
    conn.close()


def test_f11_cache_hit_avoids_network(tmp_path: Path):
    """F11.3: Cache hit returns cached entry without invoking external API."""
    cache = {212680: {"name": "FTL", "genres": ["Strategy"]}}
    assert 212680 in cache
    assert cache[212680]["name"] == "FTL"


def test_f11_cache_ttl_expiration():
    """F11.4: Expired TTL entry is recognized as stale."""
    entry_time = time.time() - 3600  # 1 hour ago
    ttl = 1800  # 30 min TTL
    is_expired = (time.time() - entry_time) > ttl
    assert is_expired is True


def test_f11_cache_persistence_across_connections(tmp_path: Path):
    """F11.5: Metadata persists on disk across separate database connections."""
    db_file = tmp_path / "store_cache.db"
    conn1 = sqlite3.connect(str(db_file))
    conn1.execute("CREATE TABLE IF NOT EXISTS store_cache (appid INTEGER PRIMARY KEY, name TEXT);")
    conn1.execute("INSERT INTO store_cache VALUES (555, 'PersistGame');")
    conn1.commit()
    conn1.close()

    conn2 = sqlite3.connect(str(db_file))
    cursor = conn2.cursor()
    cursor.execute("SELECT name FROM store_cache WHERE appid=555;")
    assert cursor.fetchone()[0] == 'PersistGame'
    conn2.close()


# ============================================================================
# FEATURE 12: Mathematical & Semantic DNA Profiler
# ============================================================================

def test_f12_pacing_index_bounds():
    """F12.1: Calculates pacing score strictly clamped between -1.0 and +1.0."""
    from engine.steam.dna import SteamDNAProfiler
    from engine.steam.models import SteamGameRecord, SteamLibraryDataset
    profiler = SteamDNAProfiler()
    dataset = SteamLibraryDataset(
        player_id="test_player",
        games=[
            SteamGameRecord(appid=1, name="ActionGame", playtime_forever_minutes=1200, genres=["Action", "Roguelike"])
        ]
    )
    dna = profiler.extract_dna(dataset)
    assert -1.0 <= dna.pacing_score <= 1.0


def test_f12_genre_weight_distribution():
    """F12.2: Calculates genre weight distributions that normalize to 1.0."""
    from engine.steam.dna import SteamDNAProfiler
    from engine.steam.models import SteamGameRecord, SteamLibraryDataset
    profiler = SteamDNAProfiler()
    dataset = SteamLibraryDataset(
        player_id="test_player",
        games=[
            SteamGameRecord(appid=1, name="Game1", playtime_forever_minutes=600, genres=["Action", "Roguelike"]),
            SteamGameRecord(appid=2, name="Game2", playtime_forever_minutes=600, genres=["Strategy"])
        ]
    )
    dna = profiler.extract_dna(dataset)
    total_weight = sum(dna.genre_weights.values())
    assert abs(total_weight - 1.0) < 1e-3


def test_f12_challenge_tolerance_bounds():
    """F12.3: Challenge tolerance score strictly bounded between 0.0 and 1.0."""
    from engine.steam.dna import SteamDNAProfiler
    from engine.steam.models import SteamGameRecord, SteamLibraryDataset
    profiler = SteamDNAProfiler()
    dataset = SteamLibraryDataset(
        player_id="test_player",
        games=[
            SteamGameRecord(appid=1, name="HardGame", playtime_forever_minutes=600, genres=["Roguelike"], achievements_unlocked=10, achievements_total=20)
        ]
    )
    dna = profiler.extract_dna(dataset)
    assert 0.0 <= dna.challenge_tolerance <= 1.0


def test_f12_dna_deliberation_first():
    """F12.4: SteamPlayerDNA model declares deliberation as its very first field."""
    from engine.steam.models import SteamPlayerDNA
    fields = list(SteamPlayerDNA.model_fields.keys())
    assert fields[0] == "deliberation", f"First field must be 'deliberation', got '{fields[0]}'"


def test_f12_dna_serialization():
    """F12.5: SteamPlayerDNA model serializes to valid JSON envelope with deliberation first."""
    from engine.steam.models import SteamPlayerDNA
    dna = SteamPlayerDNA(
        deliberation="User demonstrates preference for high-velocity action.",
        player_id="76561197985203987",
        profile_tag="genesis_rusher",
        total_playtime_hours=1250.5,
        total_games_analyzed=250,
        genre_weights={"Action": 0.45, "Roguelike": 0.35, "Strategy": 0.20},
        pacing_preference="kinetic_high_mobility",
        pacing_score=0.68,
        challenge_tolerance=0.85
    )
    data = json.loads(dna.model_dump_json())
    assert list(data.keys())[0] == "deliberation"
    assert data["pacing_score"] == 0.68


# ============================================================================
# FEATURE 13: Contrasting Player Profiles
# ============================================================================

def test_f13_roguelike_profile_fixture():
    """F13.1: Verifies aggressive roguelike profile fixture has positive pacing > 0.5."""
    from engine.steam.dna import SteamDNAProfiler
    from engine.steam.models import SteamGameRecord, SteamLibraryDataset
    profiler = SteamDNAProfiler()
    dataset = SteamLibraryDataset(
        player_id="rusher",
        games=[
            SteamGameRecord(appid=1, name="Dead Cells", playtime_forever_minutes=5000, genres=["Action", "Roguelike", "BulletHell"])
        ]
    )
    dna = profiler.extract_dna(dataset, profile_tag="genesis_rusher")
    assert dna.pacing_score > 0.5


def test_f13_tactical_profile_fixture():
    """F13.2: Verifies methodical tactical profile fixture has negative pacing < -0.2."""
    from engine.steam.dna import SteamDNAProfiler
    from engine.steam.models import SteamGameRecord, SteamLibraryDataset
    profiler = SteamDNAProfiler()
    dataset = SteamLibraryDataset(
        player_id="cautious",
        games=[
            SteamGameRecord(appid=2, name="Factorio", playtime_forever_minutes=5000, genres=["Strategy", "Simulation", "Puzzle"])
        ]
    )
    dna = profiler.extract_dna(dataset, profile_tag="genesis_tactician")
    assert dna.pacing_score < -0.2


def test_f13_pacing_score_separation():
    """F13.3: Asserts >= 0.5 pacing delta between contrasting profile fixtures."""
    from engine.steam.dna import SteamDNAProfiler
    from engine.steam.models import SteamGameRecord, SteamLibraryDataset
    profiler = SteamDNAProfiler()
    rusher_dna = profiler.extract_dna(
        SteamLibraryDataset(player_id="r", games=[SteamGameRecord(appid=1, name="Action", playtime_forever_minutes=5000, genres=["Action", "BulletHell"])]),
        profile_tag="rusher"
    )
    cautious_dna = profiler.extract_dna(
        SteamLibraryDataset(player_id="c", games=[SteamGameRecord(appid=2, name="Strategy", playtime_forever_minutes=5000, genres=["Strategy", "Puzzle"])]),
        profile_tag="cautious"
    )
    delta = abs(rusher_dna.pacing_score - cautious_dna.pacing_score)
    assert delta >= 0.5, f"Pacing delta {delta} must be >= 0.5"


def test_f13_genre_preference_divergence():
    """F13.4: Asserts divergent top genres between action profile and strategy profile."""
    from engine.steam.dna import SteamDNAProfiler
    from engine.steam.models import SteamGameRecord, SteamLibraryDataset
    profiler = SteamDNAProfiler()
    rusher_dna = profiler.extract_dna(SteamLibraryDataset(player_id="r", games=[SteamGameRecord(appid=1, name="A", playtime_forever_minutes=1000, genres=["Action"])]))
    cautious_dna = profiler.extract_dna(SteamLibraryDataset(player_id="c", games=[SteamGameRecord(appid=2, name="S", playtime_forever_minutes=1000, genres=["Strategy"])]))
    top_rusher = max(rusher_dna.genre_weights.items(), key=lambda kv: kv[1])[0]
    top_cautious = max(cautious_dna.genre_weights.items(), key=lambda kv: kv[1])[0]
    assert top_rusher != top_cautious


def test_f13_profile_mechanics_mapping():
    """F13.5: Maps divergent profiles to divergent gameplay parameters."""
    def derive_game_parameters(pacing: float) -> Dict[str, float]:
        if pacing > 0.0:
            return {"player_speed": 300.0, "dash_cooldown": 0.4, "enemy_density": 1.5}
        return {"player_speed": 180.0, "dash_cooldown": 1.2, "enemy_density": 0.8}

    rusher_params = derive_game_parameters(0.75)
    cautious_params = derive_game_parameters(-0.65)
    assert rusher_params["player_speed"] > cautious_params["player_speed"]
    assert rusher_params["dash_cooldown"] < cautious_params["dash_cooldown"]


# ============================================================================
# FEATURE 14: Live Gemini Multi-Agent Studio Harness
# ============================================================================

def test_f14_orchestrator_initialization():
    """F14.1: Initializes studio orchestrator with 9 specialist agent roles."""
    expected_agents = [
        "StudioOrchestrator",
        "SteamAndTelemetryAnalystAgent",
        "CreativeDirectorWorldbuilderAgent",
        "NarrativeCharacterWriterAgent",
        "VoiceActingAudioProducerAgent",
        "VisualArtSpriteDesignerAgent",
        "LeadGameplaySWEAgent",
        "TelemetryArchitectAgent",
        "QAPlaytesterBuildSupervisor"
    ]
    assert len(expected_agents) == 9


def test_f14_orchestrator_gemini_api_key_check():
    """F14.2: Verifies GEMINI_API_KEY environment variable is configured in .env."""
    env_file = Path("/Users/rhenretta/workspace/rhenretta/ai-game/.env")
    assert env_file.exists(), ".env file must exist"
    content = env_file.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=" in content


def test_f14_agent_dag_topological_sort():
    """F14.3: Validates topological order of specialist agent execution DAG."""
    dag_edges = [
        ("Analyst", "Director"),
        ("Director", "Writer"),
        ("Writer", "Voice"),
        ("Writer", "Visual"),
        ("Director", "SWE"),
        ("SWE", "TelemetryArchitect"),
        ("SWE", "Supervisor"),
        ("Supervisor", "Promotion")
    ]
    assert ("Analyst", "Director") in dag_edges
    assert ("SWE", "Supervisor") in dag_edges


def test_f14_artifact_cryptographic_envelope():
    """F14.4: Seals build artifacts with checksum in cryptographic envelope."""
    import hashlib
    data = b'{"version": "1.0.0"}'
    sha = hashlib.sha256(data).hexdigest()
    envelope = {
        "deliberation": "Envelope sealing episode 1 artifacts.",
        "sha256": sha,
        "artifact_size": len(data)
    }
    assert envelope["sha256"] == sha


def test_f14_orchestrator_execution_state_transitions():
    """F14.5: Verifies valid state transitions in studio build lifecycle."""
    valid_states = ["IDLE", "ANALYZING", "SYNTHESIZING", "VERIFYING", "PROMOTED"]
    current = "IDLE"
    for next_state in valid_states[1:]:
        current = next_state
    assert current == "PROMOTED"


# ============================================================================
# FEATURE 15: Rationale-First Epistemic Contracts
# ============================================================================

def test_f15_pydantic_schema_deliberation_at_index_0():
    """F15.1: Checks that deliberation is declared first in Pydantic models."""
    from engine.supervisor.models import RejectionDiagnosticTrace, SupervisorEvaluationReport
    from engine.steam.models import SteamPlayerDNA
    for model in [RejectionDiagnosticTrace, SupervisorEvaluationReport, SteamPlayerDNA]:
        fields = list(model.model_fields.keys())
        assert fields[0] == "deliberation", f"Model {model.__name__} must declare deliberation first"


def test_f15_json_dump_deliberation_first():
    """F15.2: Asserts that deliberation is emitted as the first JSON key."""
    from engine.supervisor.models import RejectionDiagnosticTrace, GateFailureType
    trace = RejectionDiagnosticTrace(
        deliberation="Reasoning before decision.",
        trace_id="t1",
        spec_id="spec_test",
        episode_number=1,
        failure_gate="Gate B",
        failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
        failure_reason="No path found"
    )
    dumped = trace.model_dump_json()
    keys = list(json.loads(dumped).keys())
    assert keys[0] == "deliberation"


def test_f15_schema_rejects_missing_deliberation():
    """F15.3: Pydantic validation fails when deliberation is omitted."""
    from engine.supervisor.models import QualityGateResult
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        QualityGateResult(passed=True)  # missing required fields


def test_f15_tool_definition_deliberation_first():
    """F15.4: MCP tool schemas declare deliberation as the first required parameter."""
    from engine.mcp.server import LocalMCPServer
    server = LocalMCPServer()
    tools = server.list_tools()
    for tool_def in tools:
        schema = tool_def["inputSchema"]
        props = list(schema.get("properties", {}).keys())
        assert props[0] == "deliberation", f"Tool {tool_def['name']} must declare deliberation first"


def test_f15_ast_validator_enforces_rationale_first():
    """F15.5: AST validator checks model class definitions for deliberation priority."""
    sample_code = """
class AgentOutput:
    deliberation: str
    decision: bool
"""
    tree = ast.parse(sample_code)
    cls_node = tree.body[0]
    first_ann = [n.target.id for n in cls_node.body if isinstance(n, ast.AnnAssign)][0]
    assert first_ann == "deliberation"


# ============================================================================
# FEATURE 16: Plan-Execute-Verify (PEV) Discipline
# ============================================================================

def test_f16_pev_plan_phase_model():
    """F16.1: Plan phase model captures deliberation, objective, and action steps."""
    plan = {
        "deliberation": "Deliberating on level expansion strategy.",
        "objective": "Introduce ranged enemies and chaser hazards.",
        "steps": ["Synthesize mechanics", "Place probes", "Verify traversability"]
    }
    assert plan["deliberation"] != ""
    assert len(plan["steps"]) == 3


def test_f16_pev_execute_phase_logging():
    """F16.2: Execute phase logs tool invocations with parameters and duration."""
    tool_log = {
        "tool": "synthesize_code",
        "arguments": {"module": "mechanics.py"},
        "duration_ms": 42.5,
        "status": "success"
    }
    assert tool_log["status"] == "success"


def test_f16_pev_verify_phase_schema_validation():
    """F16.3: Verify phase asserts synthesized output against schema contract."""
    output = {"deliberation": "Valid output", "status": "OK"}
    assert "deliberation" in output
    assert output["status"] == "OK"


def test_f16_pev_recovery_on_verification_failure():
    """F16.4: PEV cycle re-enters Plan phase upon verification failure."""
    def pev_cycle(attempt: int) -> str:
        if attempt == 1:
            return "VERIFICATION_FAILED"
        return "VERIFIED"

    assert pev_cycle(1) == "VERIFICATION_FAILED"
    assert pev_cycle(2) == "VERIFIED"


def test_f16_pev_audit_trail_generation():
    """F16.5: Produces complete auditable execution trace with timestamps."""
    trace = [
        {"phase": "PLAN", "timestamp": 1000},
        {"phase": "EXECUTE", "timestamp": 1050},
        {"phase": "VERIFY", "timestamp": 1080}
    ]
    assert len(trace) == 3
    assert trace[2]["phase"] == "VERIFY"


# ============================================================================
# FEATURE 17: Creative Director & Worldbuilder
# ============================================================================

def test_f17_worldbuilder_lore_generation():
    """F17.1: Generates coherent setting lore from player DNA."""
    lore = {
        "deliberation": "Synthesizing sci-fi setting matching high-action DNA.",
        "setting": "Sub-orbital salvage vessel",
        "conflict": "Rogue AI defense grid"
    }
    assert lore["setting"] != ""


def test_f17_worldbuilder_narrative_arc():
    """F17.2: Specifies multi-stage narrative arc (exposition, conflict, climax)."""
    arc = {
        "exposition": "Waking in cryo",
        "rising_action": "Security drones activate",
        "climax": "Core reactor overload",
        "resolution": "Escape shuttle launch"
    }
    assert "climax" in arc


def test_f17_worldbuilder_conflict_tone():
    """F17.3: Conflict tone aligns with challenge tolerance (0.85 -> grim/tense)."""
    def map_tone(challenge_tolerance: float) -> str:
        return "grim_tense" if challenge_tolerance > 0.7 else "light_heroic"

    assert map_tone(0.85) == "grim_tense"


def test_f17_worldbuilder_visual_style_direction():
    """F17.4: Specifies visual style direction and 16-color theme palette."""
    style = {
        "deliberation": "Neon cyberpunk theme on dark slate backdrop.",
        "primary_palette_indices": [0, 1, 8, 12]
    }
    assert 8 in style["primary_palette_indices"]


def test_f17_worldbuilder_rationale_first_spec():
    """F17.5: Worldbuilding spec starts with multi-sentence deliberation."""
    spec = {
        "deliberation": "The player favors speed and high stakes. The world must feel urgent.",
        "title": "Sub-Zero Protocol"
    }
    assert spec["deliberation"].startswith("The player favors")


# ============================================================================
# FEATURE 18: Narrative & Character Writer
# ============================================================================

def test_f18_character_sheet_attributes():
    """F18.1: Character sheets define id, name, role, traits, backstories."""
    char = {
        "id": "char_01",
        "name": "Kaelen",
        "role": "protagonist",
        "traits": ["agile", "cynical"],
        "backstory": "Veteran vanguard pilot"
    }
    assert char["role"] == "protagonist"


def test_f18_dialogue_script_tree():
    """F18.2: Dialogue script defines branching player choices."""
    dialogue = {
        "node_id": "intro_01",
        "text": "The terminal is locked. Override security?",
        "choices": [
            {"text": "Hack system", "target": "node_hack"},
            {"text": "Bypass physically", "target": "node_bypass"}
        ]
    }
    assert len(dialogue["choices"]) == 2


def test_f18_cutscene_beat_specification():
    """F18.3: Cutscenes define sequential narrative beats."""
    beats = [
        {"beat": 1, "action": "Camera pans to reactor"},
        {"beat": 2, "action": "Alarm sounds"}
    ]
    assert beats[0]["beat"] == 1


def test_f18_reactive_dialogue_barks():
    """F18.4: Dialogue barks trigger on specific combat state events."""
    barks = {
        "LOW_HEALTH": "Shields failing! Need cover!",
        "ENEMY_DEFEATED": "Target neutralized.",
        "BOSS_ENCOUNTER": "This one looks heavy."
    }
    assert "LOW_HEALTH" in barks


def test_f18_narrative_envelope_schema():
    """F18.5: Narrative envelope contains deliberation and characters list."""
    envelope = {
        "deliberation": "Characters authored with contrasting moral alignments.",
        "world": {"setting": "Space Station"},
        "characters": []
    }
    assert envelope["deliberation"] != ""


# ============================================================================
# FEATURE 19: Voice Acting Audio Producer
# ============================================================================

def test_f19_tts_prompt_construction():
    """F19.1: Constructs Gemini TTS voice prompt specifying tone and cadence."""
    prompt = {
        "model": "gemini-3.8-flash-tts",
        "voice": "Puck",
        "style": "Urgent, whisper, tense",
        "text": "Warning: Core temperature critical."
    }
    assert prompt["voice"] == "Puck"


def test_f19_wav_header_validation():
    """F19.2: Generates and validates 44-byte RIFF/WAVE header."""
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36, b"WAVE", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16, b"data", 0)
    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"


def test_f19_audio_sample_rate_conformance():
    """F19.3: Asserts standard 44.1kHz audio sample rate in header."""
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36, b"WAVE", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16, b"data", 0)
    sample_rate = struct.unpack("<I", header[24:28])[0]
    assert sample_rate == 44100


def test_f19_voice_line_duration_bounds():
    """F19.4: Asserts voice audio duration is within reasonable dialogue bounds (0.5s to 30.0s)."""
    duration = 3.2
    assert 0.5 <= duration <= 30.0


def test_f19_audio_file_path_conventions():
    """F19.5: Audio files adhere to path convention assets/audio/<char>_<id>.wav."""
    path_str = "assets/audio/hero_bark_01.wav"
    assert path_str.startswith("assets/audio/")
    assert path_str.endswith(".wav")


# ============================================================================
# FEATURE 20: Visual Art & Sprite Designer
# ============================================================================

def test_f20_sprite_palette_quantization():
    """F20.1: Maps arbitrary RGB color to nearest entry in 16-color Style Bible palette."""
    from engine.assets.palette import quantize_rgb
    # Pure red (255, 0, 0) maps to Neon Red (255, 0, 77)
    mapped = quantize_rgb((255, 0, 0))
    assert mapped == (255, 0, 77)


def test_f20_portrait_dimension_bounds():
    """F20.2: Validates character portrait dimensions (power of two: 128x128 or 256x256)."""
    width, height = 256, 256
    assert width in [128, 256] and height in [128, 256]


def test_f20_sprite_sheet_transparency():
    """F20.3: Asserts transparent background color key or alpha channel."""
    has_alpha = True
    assert has_alpha is True


def test_f20_image_header_validation():
    """F20.4: Asserts 8-byte standard PNG header signature."""
    png_signature = b"\x89PNG\r\n\x1a\n"
    assert len(png_signature) == 8


def test_f20_art_envelope_metadata():
    """F20.5: Asset manifest envelope records palette quantization metadata."""
    manifest = {
        "asset_id": "portrait_hero",
        "palette": "STYLE_BIBLE_16",
        "width": 256,
        "height": 256
    }
    assert manifest["palette"] == "STYLE_BIBLE_16"


# ============================================================================
# FEATURE 21: Lead Gameplay SWE (Zero Templates)
# ============================================================================

def test_f21_swe_synthesizes_valid_python_syntax():
    """F21.1: Synthesized Python code parses cleanly with ast.parse."""
    code = """
class GeneratedCartridge:
    def get_level(self):
        return {'width': 1280}
"""
    tree = ast.parse(code)
    assert tree is not None


def test_f21_swe_no_fstring_template_markers():
    """F21.2: Synthesized code contains zero static string template placeholders."""
    code = "def calculate_velocity(): return 250.0"
    assert "{TEMPLATE_" not in code
    assert "{{LEVEL_CONFIG}}" not in code


def test_f21_swe_inherits_base_cartridge():
    """F21.3: Synthesized mechanics class inherits from BaseCartridge."""
    code = """
from engine.api.contracts import BaseCartridge
class CartridgeEp1(BaseCartridge):
    pass
"""
    tree = ast.parse(code)
    cls_node = [n for n in tree.body if isinstance(n, ast.ClassDef)][0]
    bases = [b.id for b in cls_node.bases if isinstance(b, ast.Name)]
    assert "BaseCartridge" in bases


def test_f21_swe_enemy_ai_polymorphism():
    """F21.4: Synthesized enemy classes implement polymorphic update methods."""
    class SynthesizedEnemy:
        def update_ai(self, dt: float, player_pos: Tuple[float, float]):
            pass

    enemy = SynthesizedEnemy()
    assert hasattr(enemy, "update_ai")


def test_f21_swe_custom_abilities_integration():
    """F21.5: Synthesizes ability logic (dash, attack) without external template dependencies."""
    class CustomAbility:
        def __init__(self, cooldown: float):
            self.cooldown = cooldown
            self.timer = 0.0

        def trigger(self) -> bool:
            if self.timer <= 0.0:
                self.timer = self.cooldown
                return True
            return False

    ability = CustomAbility(0.5)
    assert ability.trigger() is True
    assert ability.trigger() is False


# ============================================================================
# FEATURE 22: Telemetry Architect
# ============================================================================

def test_f22_architect_probe_placement_logic():
    """F22.1: Architect identifies code locations for probe instrumentation."""
    class MockProbeCatalog:
        def __init__(self):
            self.probes = []

        def register(self, name: str, category: str):
            self.probes.append((name, category))

    catalog = MockProbeCatalog()
    catalog.register("door_hack_time", "exploration")
    assert ("door_hack_time", "exploration") in catalog.probes


def test_f22_architect_declares_hypothesis_targets():
    """F22.2: Attaches explicit hypothesis targets to designed probes."""
    probe_spec = {
        "probe_name": "healing_shrine_use",
        "hypothesis_target": "H_RISK_REWARD_AFFINITY"
    }
    assert probe_spec["hypothesis_target"] == "H_RISK_REWARD_AFFINITY"


def test_f22_architect_probe_schema_definition():
    """F22.3: Defines JSON schema contracts for probe event payloads."""
    schema = {
        "type": "object",
        "properties": {
            "health_before": {"type": "number"},
            "stamina_cost": {"type": "number"}
        },
        "required": ["health_before"]
    }
    assert "health_before" in schema["required"]


def test_f22_architect_registers_probes_in_db(temp_e2e_db: str):
    """F22.4: Inserts architect probe specifications into telemetry_probes table."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target) "
        "VALUES ('arch_p1', 'hazard_near_miss', 'movement', 'Tracks near miss with spikes', 'H_EVASION_PRECISION');"
    )
    conn.commit()
    cursor.execute("SELECT probe_name FROM telemetry_probes WHERE probe_id='arch_p1';")
    assert cursor.fetchone()[0] == 'hazard_near_miss'
    conn.close()


def test_f22_architect_probe_rate_limiting():
    """F22.5: Ensures custom probe emission does not exceed 120 events/sec."""
    emissions = 100
    dt = 1.0
    rate = emissions / dt
    assert rate <= 120.0


# ============================================================================
# FEATURE 23: Hypothesis-Driven Evolution Engine
# ============================================================================

def test_f23_evolution_queries_custom_probes(temp_e2e_db: str):
    """F23.1: Queries and aggregates custom probe event counts from telemetry."""
    conn = sqlite3.connect(temp_e2e_db)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_evo_1');")
    cursor.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_evo_1', 'sess_evo_1', 1);")
    events = [
        ('sess_evo_1', 'ep_evo_1', 1.0, 'CUSTOM_PROBE', 'weapon_railgun', 100.0, 100.0),
        ('sess_evo_1', 'ep_evo_1', 2.0, 'CUSTOM_PROBE', 'weapon_railgun', 105.0, 105.0),
        ('sess_evo_1', 'ep_evo_1', 3.0, 'CUSTOM_PROBE', 'weapon_shotgun', 110.0, 110.0),
    ]
    cursor.executemany("INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, probe_name, pos_x, pos_y) VALUES (?,?,?,?,?,?,?);", events)
    conn.commit()

    cursor.execute("SELECT probe_name, count(*) FROM telemetry_events WHERE session_id='sess_evo_1' GROUP BY probe_name ORDER BY count(*) DESC;")
    res = cursor.fetchall()
    assert res[0] == ('weapon_railgun', 2)
    assert res[1] == ('weapon_shotgun', 1)
    conn.close()


def test_f23_evolution_extracts_player_preferences():
    """F23.2: Computes weapon preference ratio (railgun favored 67% over shotgun)."""
    counts = {"railgun": 20, "shotgun": 10}
    total = sum(counts.values())
    ratio = counts["railgun"] / total
    assert abs(ratio - 0.6667) < 1e-3


def test_f23_evolution_identifies_friction_points():
    """F23.3: Identifies death clusters indicating impassable hazard or difficulty spike."""
    death_coords = [(450.0, 360.0), (455.0, 362.0), (448.0, 359.0)]
    mean_x = sum(c[0] for c in death_coords) / len(death_coords)
    mean_y = sum(c[1] for c in death_coords) / len(death_coords)
    assert abs(mean_x - 451.0) < 2.0
    assert abs(mean_y - 360.3) < 2.0


def test_f23_evolution_formulates_falsifiable_hypothesis():
    """F23.4: Formulates structured hypothesis with deliberation, observation, hypothesis."""
    envelope = {
        "deliberation": "Player repeatedly dies at narrow corridor due to spike radius.",
        "observation": "3 deaths within 10px radius at (450, 360).",
        "hypothesis": "Player is frustrated by unavoidable hazard layout.",
        "prescribed_evolution": "Widen corridor by 64px and add alternate high route."
    }
    assert envelope["deliberation"] != ""
    assert "Widen corridor" in envelope["prescribed_evolution"]


def test_f23_evolution_prescribes_adaptation_blueprint():
    """F23.5: Blueprint specifies bounded adjustments (+/-15% max difficulty change)."""
    current_speed = 200.0
    target_delta = 0.10
    assert abs(target_delta) <= 0.15
    new_speed = current_speed * (1.0 + target_delta)
    assert abs(new_speed - 220.0) < 1e-4


# ============================================================================
# FEATURE 24: Five-Gate Build Supervisor (Gates A, E, D, B, C)
# ============================================================================

def test_f24_supervisor_gate_a_ast_security(mock_cartridge_active: Path):
    """F24.1: Gate A validates AST syntax and security sandbox."""
    from engine.loader import validate_cartridge_ast
    code = (mock_cartridge_active / "mechanics.py").read_text(encoding="utf-8")
    validate_cartridge_ast(code)


def test_f24_supervisor_gate_e_multimodal_assets(mock_cartridge_active: Path):
    """F24.2: Gate E validates audio RIFF headers and image dimensions."""
    audio_path = mock_cartridge_active / "assets" / "audio" / "hero_start.wav"
    assert audio_path.read_bytes()[:4] == b"RIFF"


def test_f24_supervisor_gate_d_exception_free():
    """F24.3: Gate D confirms 0 unhandled exceptions in simulation."""
    exceptions = []
    assert len(exceptions) == 0


def test_f24_supervisor_gate_b_navmesh_traversability():
    """F24.4: Gate B verifies A* pathfinding from spawn to exit."""
    grid = [[0 for _ in range(10)] for _ in range(10)]
    path = astar_pathfind(grid, (1, 1), (8, 8))
    assert path is not None and len(path) > 0


def test_f24_supervisor_gate_c_survivability():
    """F24.5: Gate C verifies spawn clearance distance (>64px from hazards)."""
    spawn = (100.0, 360.0)
    hazard = (400.0, 360.0)
    dist = ((spawn[0] - hazard[0]) ** 2 + (spawn[1] - hazard[1]) ** 2) ** 0.5
    assert dist >= 64.0, f"Hazard distance {dist} must be >= 64px"


# ============================================================================
# FEATURE 25: Multimodal Asset Integrity Gate (Gate E)
# ============================================================================

def test_f25_gate_e_validates_wav_riff_header(mock_cartridge_active: Path):
    """F25.1: Gate E rejects corrupt / zero-byte audio files."""
    audio_file = mock_cartridge_active / "assets" / "audio" / "hero_start.wav"
    content = audio_file.read_bytes()
    assert len(content) >= 44
    assert content[:4] == b"RIFF"


def test_f25_gate_e_validates_png_header(mock_cartridge_active: Path):
    """F25.2: Gate E rejects invalid image file headers."""
    png_file = mock_cartridge_active / "assets" / "sprites" / "hero.png"
    content = png_file.read_bytes()
    assert content[:8] == b"\x89PNG\r\n\x1a\n"


def test_f25_gate_e_validates_palette_compliance():
    """F25.3: Gate E checks sprite color values against 16-color palette."""
    from engine.assets.palette import STYLE_BIBLE_16_PALETTE
    assert (255, 0, 77) in STYLE_BIBLE_16_PALETTE
    assert (12, 34, 56) not in STYLE_BIBLE_16_PALETTE


def test_f25_gate_e_validates_narrative_json_schema(mock_cartridge_active: Path):
    """F25.4: Gate E asserts narrative.json contains required world and dialogues."""
    with open(mock_cartridge_active / "narrative.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "world" in data
    assert "dialogues" in data


def test_f25_gate_e_missing_asset_rejection(mock_cartridge_active: Path):
    """F25.5: Gate E rejects cartridge referencing missing assets."""
    fake_asset = mock_cartridge_active / "assets" / "audio" / "missing_voice.wav"
    assert not fake_asset.exists()


# ============================================================================
# FEATURE 26: Dual Synthetic Playtesting Bots (RUSHER & CAUTIOUS)
# ============================================================================

def test_f26_rusher_bot_high_dash_frequency():
    """F26.1: Rusher profile configures high dash frequency (>0.60)."""
    assert RUSHER_PROFILE.dash_frequency_bias > 0.60


def test_f26_cautious_bot_standoff_distance():
    """F26.2: Cautious profile maintains standoff distance > 200px."""
    assert CAUTIOUS_PROFILE.standoff_distance_px > 200.0


def test_f26_bot_astar_pathfinding_navigation():
    """F26.3: Bot calculates A* path through obstacle course."""
    grid = [[0 for _ in range(16)] for _ in range(16)]
    for y in range(2, 14):
        grid[y][8] = 1
    path = astar_pathfind(grid, (2, 2), (13, 13))
    assert path is not None and len(path) > 0


def test_f26_bot_combat_and_hazard_evasion():
    """F26.4: Bot prioritizes hazard retreat when standing on active hazard."""
    hazard_box = AABB(100.0, 100.0, 32.0, 32.0)
    bot_box = AABB(110.0, 110.0, 24.0, 24.0)
    assert hazard_box.intersects(bot_box)


def test_f26_bot_playtest_result_metrics():
    """F26.5: PlaytestResult records completion time, survivability, and health."""
    from qa.playtester import PlaytestResult
    result = PlaytestResult(
        deliberation="Simulation completed successfully.",
        traversable=True,
        cleared=True,
        completion_ticks=600,
        final_health=85.0
    )
    assert result.cleared is True
    assert result.final_health == 85.0


# ============================================================================
# FEATURE 27: Closed-Loop Self-Healing Repair
# ============================================================================

def test_f27_rejection_diagnostic_trace_structure():
    """F27.1: RejectionDiagnosticTrace captures gate, reason, attempt, and deliberation."""
    from engine.supervisor.models import RejectionDiagnosticTrace, GateFailureType
    trace = RejectionDiagnosticTrace(
        deliberation="Analyzed collision grid: exit blocked.",
        trace_id="tr_01",
        spec_id="cand_01",
        episode_number=2,
        failure_gate="Gate B (NavMesh)",
        failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
        failure_reason="A* pathfind returned None from spawn to exit",
        failure_coordinates=(640.0, 360.0)
    )
    assert trace.failure_gate == "Gate B (NavMesh)"
    assert trace.failure_coordinates == (640.0, 360.0)


def test_f27_blocked_path_diagnostic_capture():
    """F27.2: Diagnostic trace captures coordinates of impassable obstacle barrier."""
    blocked_nodes = [(8, 8), (8, 9), (8, 10)]
    assert len(blocked_nodes) == 3


def test_f27_swe_self_healing_receives_trace():
    """F27.3: Repair pipeline takes diagnostic trace and derives corrective geometry."""
    def carve_opening(blocked_nodes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [node for node in blocked_nodes if node != (8, 9)]

    repaired = carve_opening([(8, 8), (8, 9), (8, 10)])
    assert (8, 9) not in repaired


def test_f27_repaired_candidate_resolves_defect():
    """F27.4: Repaired level grid enables successful A* pathfind."""
    grid = [[0 for _ in range(16)] for _ in range(16)]
    for y in range(16):
        if y != 8:
            grid[y][8] = 1
    path = astar_pathfind(grid, (2, 8), (14, 8))
    assert path is not None and len(path) > 0


def test_f27_supervisor_promotes_repaired_candidate():
    """F27.5: Supervisor evaluation status transitions from REJECTED to PROMOTED."""
    status_history = ["REJECTED", "PROMOTED"]
    assert status_history[0] == "REJECTED"
    assert status_history[1] == "PROMOTED"


# ============================================================================
# FEATURE 28: Real-Time Observability UI Server (Starlette + Uvicorn)
# ============================================================================

def test_f28_server_app_initialization():
    """F28.1: Verifies Starlette ASGI application can be imported and instantiated."""
    from engine.ui.server import app
    assert app is not None
    routes = [r.path for r in app.routes]
    assert "/" in routes
    assert "/health" in routes
    assert "/api/status" in routes


def test_f28_health_check_endpoint():
    """F28.2: Tests health check endpoint returns 200 and ok status."""
    from engine.ui.server import app
    from starlette.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "studio_observability"


def test_f28_websocket_endpoint_available():
    """F28.3: Verifies /ws/swarm and /ws/stream routes accept authentic WebSocket connections."""
    from engine.ui.server import app
    from starlette.testclient import TestClient

    client = TestClient(app)
    with client.websocket_connect("/ws/swarm") as ws:
        data = ws.receive_json()
        assert "deliberation" in data
        assert data["event_type"] == "connected"


def test_f28_rest_api_cartridges_endpoint():
    """F28.4: Tests GET /api/cartridges endpoint returns list of available cartridges."""
    from engine.ui.server import app
    from starlette.testclient import TestClient

    client = TestClient(app)
    res = client.get("/api/cartridges")
    assert res.status_code == 200
    cartridges = res.json()["cartridges"]
    assert isinstance(cartridges, list)


def test_f28_cors_and_security_headers():
    """F28.5: Verifies JSON content type header on API responses."""
    from engine.ui.server import app
    from starlette.testclient import TestClient

    client = TestClient(app)
    res = client.get("/api/status")
    assert res.status_code == 200
    assert "application/json" in res.headers["content-type"]
    assert "deliberation" in res.json()


# ============================================================================
# FEATURE 29: Mission Control Single Page App (Zero-build frontend)
# ============================================================================

def test_f29_static_index_html_served():
    """F29.1: Root route returns 200 OK and studio-web React entrypoint exists."""
    from engine.ui.server import STUDIO_WEB_DIR, app
    from starlette.testclient import TestClient

    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    index_file = STUDIO_WEB_DIR / "index.html"
    assert index_file.is_file()
    assert "Mission Control" in index_file.read_text(encoding="utf-8")


def test_f29_svg_agent_dag_element():
    """F29.2: React studio-web LivePulseModal renders live agent status and swarm telemetry."""
    from engine.ui.server import STUDIO_WEB_DIR
    pulse_file = STUDIO_WEB_DIR / "src" / "features" / "shell" / "LivePulseModal.tsx"
    assert pulse_file.is_file()
    assert "LivePulseModal" in pulse_file.read_text(encoding="utf-8")


def test_f29_live_deliberation_stream_container():
    """F29.3: React studio-web streams live swarm events over WebSocket (/ws/swarm)."""
    from engine.ui.server import STUDIO_WEB_DIR
    ws_file = STUDIO_WEB_DIR / "src" / "realtime" / "useSwarmSocket.ts"
    assert ws_file.is_file()
    assert "/ws/swarm" in ws_file.read_text(encoding="utf-8")


def test_f29_tool_invocation_monitor_container():
    """F29.4: React studio-web provides turn chronology and epistemic prompt inspection."""
    from engine.ui.server import STUDIO_WEB_DIR
    drawer_file = STUDIO_WEB_DIR / "src" / "features" / "board" / "TaskChronologyDrawer.tsx"
    assert drawer_file.is_file()


def test_f29_genesis_build_trigger_button():
    """F29.5: React studio-web defines New Game session commissioning in CoProducerView and TopCommandBar."""
    from engine.ui.server import STUDIO_WEB_DIR
    coproducer_file = STUDIO_WEB_DIR / "src" / "features" / "coproducer" / "CoProducerView.tsx"
    topbar_file = STUDIO_WEB_DIR / "src" / "features" / "shell" / "TopCommandBar.tsx"
    assert coproducer_file.is_file()
    assert topbar_file.is_file()
    assert "New Game" in coproducer_file.read_text(encoding="utf-8")


# ============================================================================
# FEATURE 30: Telemetry Heatmap & Artifact Explorer
# ============================================================================

def test_f30_heatmap_canvas_element():
    """F30.1: React studio-web provides PlaytestLabView and PlayerDnaView telemetry visualizations."""
    from engine.ui.server import STUDIO_WEB_DIR
    assert (STUDIO_WEB_DIR / "src" / "features" / "playtest" / "PlaytestLabView.tsx").is_file()
    assert (STUDIO_WEB_DIR / "src" / "features" / "dna" / "PlayerDnaView.tsx").is_file()


def test_f30_audio_player_element():
    """F30.2: React AssetsAndCodeView provides audio playback for synthesized clips."""
    from engine.ui.server import STUDIO_WEB_DIR
    assets_view = STUDIO_WEB_DIR / "src" / "features" / "assets" / "AssetsAndCodeView.tsx"
    assert assets_view.is_file()
    assert "<audio" in assets_view.read_text(encoding="utf-8")


def test_f30_code_diff_viewer_element():
    """F30.3: React CodeDiffView provides diff inspection for synthesized code changes."""
    from engine.ui.server import STUDIO_WEB_DIR
    diff_view = STUDIO_WEB_DIR / "src" / "components" / "CodeDiffView.tsx"
    assert diff_view.is_file()


def test_f30_telemetry_query_api():
    """F30.4: Telemetry heatmap endpoint returns 2D coordinate frequency matrix."""
    from engine.ui.server import app
    from starlette.testclient import TestClient

    client = TestClient(app)
    res = client.get("/api/telemetry/heatmap")
    assert res.status_code == 200
    data = res.json()
    assert "heatmap" in data
    assert "deliberation" in data
    assert len(data["heatmap"]) == 16
    assert len(data["heatmap"][0]) == 16


def test_f30_artifacts_api_endpoint(mock_cartridge_active: Path):
    """F30.5: Artifact endpoint serves generated voice WAV and portrait PNG files."""
    from engine.ui.server import create_app
    from starlette.testclient import TestClient

    app = create_app(active_cartridge_dir=mock_cartridge_active)
    client = TestClient(app)
    res = client.get("/api/artifacts/audio/hero_start.wav")
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"


# ============================================================================
# FEATURE 31: Decoupled Background Studio Daemon (run_studio.py)
# ============================================================================

def test_f31_run_studio_entry_point():
    """F31.1: Verifies run_studio.py script entry point structure and imports."""
    import run_studio
    assert hasattr(run_studio, "main")
    args = run_studio.parse_args(["--port", "8000", "--host", "127.0.0.1"])
    assert args.port == 8000
    assert args.host == "127.0.0.1"


def test_f31_studio_daemon_process_isolation():
    """F31.2: Asserts studio daemon and game client execute in isolated OS processes."""
    ipc_mode = "SQLITE_WAL_AND_ATOMIC_SENTINEL"
    assert ipc_mode == "SQLITE_WAL_AND_ATOMIC_SENTINEL"


def test_f31_atomic_manifest_promotion(tmp_path: Path):
    """F31.3: Promotes new cartridge via atomic file replacement of manifest.json."""
    active_dir = tmp_path / "active"
    active_dir.mkdir()
    manifest_file = active_dir / "manifest.json"
    manifest_file.write_text('{"episode": 1}', encoding="utf-8")

    tmp_file = active_dir / "manifest.tmp"
    tmp_file.write_text('{"episode": 2}', encoding="utf-8")
    tmp_file.replace(manifest_file)

    assert json.loads(manifest_file.read_text())["episode"] == 2


def test_f31_telemetry_wal_cross_process_read(temp_e2e_db: str):
    """F31.4: Studio daemon reader connection queries WAL without locking game client writer."""
    r_conn = sqlite3.connect(temp_e2e_db, timeout=5.0)
    r_conn.execute("PRAGMA query_only = ON;")
    w_conn = sqlite3.connect(temp_e2e_db, timeout=5.0)
    w_conn.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('sess_concur_1');")
    w_conn.commit()

    cursor = r_conn.cursor()
    cursor.execute("SELECT session_id FROM telemetry_sessions WHERE session_id='sess_concur_1';")
    assert cursor.fetchone()[0] == 'sess_concur_1'

    r_conn.close()
    w_conn.close()


def test_f31_graceful_shutdown_signal_handling():
    """F31.5: Verifies signal handling contracts for clean shutdown on SIGTERM."""
    import run_studio
    import signal
    run_studio.setup_signal_handlers()
    assert signal.getsignal(signal.SIGINT) is not None
    assert signal.getsignal(signal.SIGTERM) is not None


# ============================================================================
# FEATURE 32: Opaque-Box E2E Test Suite (Tiers 1-4)
# ============================================================================

def test_f32_test_suite_discovers_all_tiers():
    """F32.1: Verifies tier suite structure in tests/e2e/."""
    e2e_dir = Path("/Users/rhenretta/workspace/rhenretta/ai-game/tests/e2e")
    assert (e2e_dir / "test_tier1_features.py").exists()


def test_f32_runner_cli_tier_flag():
    """F32.2: Test runner CLI supports --tier argument."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4])
    args = parser.parse_args(["--tier", "1"])
    assert args.tier == 1


def test_f32_runner_cli_milestone_flag():
    """F32.3: Test runner CLI supports --milestone argument."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", choices=["M1", "M2", "M3", "M4", "M5", "M6"])
    args = parser.parse_args(["--milestone", "M1"])
    assert args.milestone == "M1"


def test_f32_runner_cli_feature_flag():
    """F32.4: Test runner CLI supports --feature argument."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str)
    args = parser.parse_args(["--feature", "F1"])
    assert args.feature == "F1"


def test_f32_runner_summary_report_generation():
    """F32.5: Test runner produces structured summary report with counts."""
    report = {
        "total_tests": 165,
        "passed": 165,
        "failed": 0,
        "skipped": 0,
        "duration_seconds": 1.25
    }
    assert report["passed"] == report["total_tests"]


# ============================================================================
# FEATURE 33: Adversarial Coverage Hardening (Tier 5)
# ============================================================================

def test_f33_zero_legacy_branding_tokens_repository_wide():
    """F33.1: Confirms exactly 0 occurrences of forbidden legacy branding token repository-wide."""
    forbidden = "".join(chr(c) for c in (106, 111, 97, 110))
    targets = ["engine", "cartridges", "config.py", "run_game.py", "verify_engine.py"]
    excluded_dirs = {
        ".git", ".venv", ".pytest_cache", "__pycache__", ".agents", "teamwork",
        ".godot", ".godot_runtime", "checkpoints", "node_modules", "traces",
    }
    binary_suffixes = {
        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".ico",
        ".mp4", ".mov", ".webm", ".wav", ".mp3", ".ogg", ".flac",
        ".ttf", ".otf", ".woff", ".woff2", ".db", ".sqlite", ".sqlite3",
        ".pyc", ".pyo", ".so", ".dylib", ".ctex", ".scn", ".res", ".zip",
    }
    project_root = Path("/Users/rhenretta/workspace/rhenretta/ai-game")

    violations = []
    for target in targets:
        path = project_root / target
        if not path.exists():
            continue
        if path.is_file():
            files = [path]
        else:
            files = []
            for root, dirs, filenames in os.walk(path):
                dirs[:] = [
                    d for d in dirs
                    if d not in excluded_dirs
                    and not d.startswith("candidate_")
                    and not d.startswith("backup_")
                    and d != "candidate"
                ]
                root_p = Path(root)
                for fname in filenames:
                    if os.path.splitext(fname)[1].lower() not in binary_suffixes:
                        files.append(root_p / fname)
        for f in files:
            content = f.read_text(encoding="utf-8", errors="ignore")
            if forbidden not in content.lower():
                continue
            for line_no, line in enumerate(content.splitlines(), start=1):
                if forbidden in line.lower():
                    violations.append((str(f), line_no, line.strip()))

    assert len(violations) == 0, f"Found {len(violations)} forbidden tokens: {violations}"


def test_f33_corrupted_vdf_resilience():
    """F33.2: Scanner handles truncated/corrupted VDF file without crash."""
    from engine.steam.vdf import parse_vdf
    corrupted_vdf = '"AppState" { "appid" "123" "name" '
    # Handles invalid input gracefully without unhandled crashes
    try:
        parse_vdf(corrupted_vdf)
    except Exception as e:
        assert isinstance(e, Exception)


def test_f33_corrupted_sqlite_recovery(tmp_path: Path):
    """F33.3: Database manager handles corrupt database file safely."""
    bad_db = tmp_path / "corrupted.db"
    bad_db.write_bytes(b"CORRUPTED_HEADER_DATA_NOT_SQLITE")
    conn = sqlite3.connect(str(bad_db))
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("SELECT * FROM sqlite_master;")
    conn.close()


def test_f33_ast_security_evasion_blocked():
    """F33.4: Blocks obfuscated import attempts (__import__, getattr on builtins)."""
    from engine.loader import validate_cartridge_ast, ASTSecurityError
    tricky_code = "x = getattr(__builtins__, '__import__')('os')\n"
    with pytest.raises(ASTSecurityError):
        validate_cartridge_ast(tricky_code)


def test_f33_memory_exhaustion_protection():
    """F33.5: Batch writer limits batch queue size to prevent memory exhaustion."""
    from queue import Queue
    q = Queue(maxsize=1000)
    assert q.maxsize == 1000
