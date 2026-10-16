"""tests/test_m1_challenger_stress.py
Empirical stress tests for Milestone 1 Autonomous Engine:
- High-throughput SQLite WAL telemetry flood (5,000 events) under concurrent reader queries.
- Telemetry flush durability to disk.
- CartridgeLoader AST security validation against malicious code (eval, os.system, subprocess, socket).
- Dynamic cartridge hot-swapping and state cleanup between episodes.
"""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
from typing import List
import pytest

from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.combat.weapon import SpreadWeapon
from engine.core.engine import Engine
from engine.core.scenes.playing_scene import PlayingScene
from engine.core.scenes.victory_scene import EpisodeVictoryScene
from engine.entities.enemies import ChaserEnemy, RangedEnemy
from engine.entities.hazard import SpikeHazard
from engine.entities.player import Player
from engine.loader import ASTSecurityError, CartridgeLoader, validate_cartridge_ast
from engine.telemetry.recorder import TelemetryRecorder


class TestTelemetryWALStress:
    """Stress tests for SQLite WAL queue, high throughput, and concurrency."""

    def test_high_throughput_telemetry_flood_concurrent_reader(self, tmp_path: Path):
        """
        Emits 5,000 events in rapid succession while a reader concurrently queries
        `SELECT count(*) FROM telemetry_events`.
        Verifies zero database locked errors and zero dropped events.
        """
        db_path = tmp_path / "telemetry_flood.db"
        recorder = TelemetryRecorder(db_path=db_path, batch_size=50, flush_interval_sec=0.05)
        recorder.start_session(player_profile_tag="flood_test")
        recorder.start_episode(level_id="stress_chamber", episode_index=1)

        reader_errors: List[str] = []
        queries_executed = [0]
        stop_reader = threading.Event()

        def reader_loop():
            conn = recorder.db.get_reader_connection()
            cur = conn.cursor()
            while not stop_reader.is_set():
                try:
                    cur.execute("SELECT count(*) FROM telemetry_events;")
                    cur.fetchone()
                    queries_executed[0] += 1
                except Exception as exc:
                    reader_errors.append(str(exc))
            conn.close()

        reader_thread = threading.Thread(target=reader_loop, daemon=True)
        reader_thread.start()

        # Emit 5,000 events in rapid succession
        t_start = time.perf_counter()
        for i in range(5000):
            recorder.emit_event(
                event_type="MOVEMENT_TICK",
                pos=(float(i % 1280), float(i % 720)),
                vel=(150.0, 50.0),
                health=100.0,
                payload={"frame": i, "stress_index": i}
            )
        emit_elapsed = time.perf_counter() - t_start

        # Flush to disk
        recorder.flush()
        stop_reader.set()
        reader_thread.join(timeout=3.0)

        # Assert no reader locked errors occurred
        assert len(reader_errors) == 0, f"Reader encountered SQLite errors: {reader_errors}"
        assert queries_executed[0] > 0, "Reader queries did not execute during flood"

        # Verify zero dropped events (exactly 5,000 events in database)
        verify_conn = recorder.db.get_reader_connection()
        cur = verify_conn.cursor()
        cur.execute("SELECT count(*) FROM telemetry_events;")
        final_count = cur.fetchone()[0]
        verify_conn.close()

        assert final_count == 5000, f"Dropped events detected: expected 5000, found {final_count}"

        # Clean shutdown
        recorder.end_episode(outcome="CLEARED")
        recorder.end_session()
        recorder.close()

    def test_flush_durability_immediate_persistence(self, tmp_path: Path):
        """
        Verifies that recorder.flush() synchronously persists all rows to disk before returning,
        verified by opening an independent raw SQLite connection directly against the database file.
        """
        db_path = tmp_path / "telemetry_durability.db"
        recorder = TelemetryRecorder(db_path=db_path, batch_size=33, flush_interval_sec=1.0)
        recorder.start_session(player_profile_tag="durability_test")
        recorder.start_episode(level_id="chamber_durable", episode_index=1)

        event_count = 377  # Not a multiple of batch_size
        for i in range(event_count):
            recorder.emit_event(
                event_type="DASH",
                pos=(float(i), 150.0),
                vel=(250.0, 0.0),
                health=100.0,
                payload={"i": i}
            )

        # Trigger flush
        recorder.flush()

        # Connect independently from outside the recorder directly to the SQLite file
        direct_conn = sqlite3.connect(str(db_path), timeout=5.0)
        cur = direct_conn.cursor()
        cur.execute("SELECT count(*) FROM telemetry_events;")
        persisted_count = cur.fetchone()[0]
        direct_conn.close()

        recorder.close()

        assert persisted_count == event_count, (
            f"Flush durability failure: expected {event_count} persisted rows, found {persisted_count}"
        )


class TestCartridgeLoaderSecurityStress:
    """Stress tests for CartridgeLoader security and AST validation."""

    def test_ast_security_rejects_eval(self):
        """Verifies rejection of eval() dynamic execution."""
        code = 'def bad(): eval("1 + 1")'
        with pytest.raises(ASTSecurityError):
            validate_cartridge_ast(code)

    def test_ast_security_rejects_exec(self):
        """Verifies rejection of exec() dynamic execution."""
        code = 'def bad(): exec("x = 1")'
        with pytest.raises(ASTSecurityError):
            validate_cartridge_ast(code)

    def test_ast_security_rejects_os_system(self):
        """Verifies rejection of os.system() system calls."""
        code = """
import os
def bad():
    os.system("echo compromised")
"""
        with pytest.raises(ASTSecurityError):
            validate_cartridge_ast(code)

    def test_ast_security_rejects_from_os_system(self):
        """Verifies rejection of from os import system and direct system() call."""
        code = """
from os import system
def bad():
    system("echo compromised")
"""
        with pytest.raises(ASTSecurityError):
            validate_cartridge_ast(code)

    def test_ast_security_rejects_subprocess(self):
        """Verifies rejection of subprocess imports and execution calls."""
        code = """
import subprocess
def bad():
    subprocess.run(["echo", "compromised"])
"""
        with pytest.raises(ASTSecurityError):
            validate_cartridge_ast(code)

    def test_ast_security_rejects_socket(self):
        """Verifies rejection of socket imports and network socket creation."""
        code = """
import socket
def bad():
    s = socket.socket()
    s.connect(("127.0.0.1", 80))
"""
        with pytest.raises(ASTSecurityError):
            validate_cartridge_ast(code)

    def test_cartridge_loader_rejects_malicious_cartridge(self, tmp_path: Path):
        """Verifies that CartridgeLoader.load_cartridge rejects cartridges with malicious code."""
        cart_dir = tmp_path / "exploit_cartridge"
        cart_dir.mkdir()
        (cart_dir / "manifest.json").write_text(json.dumps({"name": "ExploitCartridge"}))
        (cart_dir / "mechanics.py").write_text("""
import subprocess
from engine.api.contracts import BaseCartridge, BaseLevelSpec

class ExploitCartridge(BaseCartridge):
    def get_level_spec(self, episode_number=1):
        return BaseLevelSpec(level_id="lvl1", name="exploit")
    def create_player(self, x, y, recorder=None):
        return None
    def create_enemies(self, spec):
        return []
    def create_hazards(self, spec):
        return []
    def __init__(self, manifest=None):
        super().__init__(manifest)
        subprocess.run(["echo", "injected"])
""")
        with pytest.raises(ASTSecurityError):
            CartridgeLoader.load_cartridge(cart_dir)


class TestDynamicHotSwapStress:
    """Stress tests for Cartridge dynamic loading, hot-swapping, and scene state cleanup."""

    def test_dynamic_hot_swap_and_state_cleanup(self, tmp_path: Path):
        """
        Loads Cartridge A, executes gameplay ticks, switches scenes, hot-swaps to Cartridge B,
        and verifies that:
        1. All episode state from Cartridge A is purged (health, stats, position).
        2. Cartridge B's distinct level spec, player config, weapon, and entities are active.
        3. Engine steps cleanly under the hot-swapped cartridge.
        """
        dir_a = tmp_path / "cartridge_a"
        dir_a.mkdir()
        (dir_a / "manifest.json").write_text(json.dumps({"name": "CartridgeA"}))
        (dir_a / "mechanics.py").write_text("""
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.entities.player import Player
from engine.entities.enemies import ChaserEnemy

class CartridgeA(BaseCartridge):
    def get_level_spec(self, episode_number=1):
        return BaseLevelSpec(
            level_id="lvl_a",
            name="Chamber A",
            spawn_point=(80.0, 80.0),
            exit_point=(1100.0, 360.0),
            walls=[(0, 0, 1280, 32), (0, 0, 32, 720)],
            enemies=[{"type": "chaser", "x": 300.0, "y": 300.0}]
        )
    def create_player(self, x, y, recorder=None):
        p = Player(x, y, recorder)
        p.max_health = 250.0
        p.health = 250.0
        return p
    def create_enemies(self, spec):
        return [ChaserEnemy("e1", 300.0, 300.0)]
    def create_hazards(self, spec):
        return []
""")

        dir_b = tmp_path / "cartridge_b"
        dir_b.mkdir()
        (dir_b / "manifest.json").write_text(json.dumps({"name": "CartridgeB"}))
        (dir_b / "mechanics.py").write_text("""
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.entities.player import Player
from engine.combat.weapon import SpreadWeapon
from engine.entities.enemies import RangedEnemy
from engine.entities.hazard import SpikeHazard

class CartridgeB(BaseCartridge):
    def get_level_spec(self, episode_number=1):
        return BaseLevelSpec(
            level_id="lvl_b",
            name="Chamber B",
            spawn_point=(350.0, 450.0),
            exit_point=(900.0, 200.0),
            walls=[(0, 0, 1280, 32), (0, 688, 1280, 32), (500, 200, 40, 200)],
            enemies=[{"type": "ranged", "x": 700.0, "y": 450.0}, {"type": "ranged", "x": 800.0, "y": 200.0}],
            hazards=[{"type": "spikes", "x": 400.0, "y": 300.0, "w": 48.0, "h": 48.0}]
        )
    def create_player(self, x, y, recorder=None):
        p = Player(x, y, recorder)
        p.max_health = 75.0
        p.health = 75.0
        p.weapon = SpreadWeapon()
        return p
    def create_enemies(self, spec):
        return [RangedEnemy("r1", 700.0, 450.0), RangedEnemy("r2", 800.0, 200.0)]
    def create_hazards(self, spec):
        return [SpikeHazard("s1", 400.0, 300.0, 48.0, 48.0)]
""")

        engine = Engine(headless=True)
        engine.scene_manager.register_scene("playing", PlayingScene(engine))
        engine.scene_manager.register_scene("victory", EpisodeVictoryScene(engine))

        # 1. Load Cartridge A
        engine.load_cartridge(dir_a)
        engine.scene_manager.switch_to("playing", episode_number=1)
        scene = engine.scene_manager.current_scene

        assert scene.player.health == 250.0
        assert scene.player.max_health == 250.0
        assert scene.player.pos.x == 80.0
        assert scene.player.pos.y == 80.0
        assert len(scene.enemies) == 1
        assert len(scene.hazards) == 0

        # Simulate game ticks and state mutations in Episode 1
        scene.player.health = 180.0
        scene.stats["dashes"] = 7
        scene.stats["damage_taken"] = 70.0
        for _ in range(10):
            engine.game_loop.step()

        # Switch to victory
        engine.scene_manager.switch_to("victory")

        # 2. Hot-swap to Cartridge B
        engine.load_cartridge(dir_b)

        # 3. Transition into Episode 2
        engine.scene_manager.switch_to("playing", episode_number=2)
        scene2 = engine.scene_manager.current_scene

        # Verify state cleanup
        assert scene2.stats["dashes"] == 0, "Dashes stat was not cleaned up across hot-swap"
        assert scene2.stats["damage_taken"] == 0.0, "Damage taken was not reset across hot-swap"
        assert scene2.elapsed_time == 0.0, "Elapsed time was not reset across hot-swap"

        # Verify Cartridge B mechanics and entities active
        assert scene2.player.max_health == 75.0, "Player max_health does not reflect Cartridge B"
        assert scene2.player.health == 75.0, "Player health was not reset to Cartridge B baseline"
        assert scene2.player.pos.x == 350.0 and scene2.player.pos.y == 450.0, "Spawn position does not match Cartridge B"
        assert isinstance(scene2.player.weapon, SpreadWeapon), "Player weapon does not reflect Cartridge B"
        assert len(scene2.enemies) == 2, "Enemy count does not match Cartridge B"
        assert len(scene2.hazards) == 1, "Hazard count does not match Cartridge B"

        # Verify engine steps cleanly under new cartridge
        for _ in range(10):
            engine.game_loop.step()

        engine.quit()
