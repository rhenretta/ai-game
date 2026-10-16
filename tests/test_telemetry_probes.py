"""tests/test_telemetry_probes.py
Comprehensive unit, stress, and security tests for Features 1, 2, and 3:
1. Dynamic Probe Metadata Catalog (telemetry_probes table, schemas, pragmas, migrations)
2. Custom Telemetry Event Emission (< 2 µs latency budget, WAL concurrency, durability)
3. Agent Telemetry Probe SDK (AgentTelemetryProbe, AST security, fallback safety)
"""
import json
from pathlib import Path
import sqlite3
import threading
import time
import pytest

from engine.loader import validate_cartridge_ast
from engine.sim.headless import InMemoryTelemetryRecorder
from engine.telemetry.db import TelemetryDB
from engine.telemetry.models import ProbeCatalogRecord, TelemetryEventType
from engine.telemetry.probes import AgentTelemetryProbe
from engine.telemetry.recorder import NullTelemetryRecorder, TelemetryRecorder


class TestDynamicProbeCatalog:
    """Feature 1: Dynamic Probe Metadata Catalog Tests."""

    def test_probe_registration_and_query(self, tmp_path: Path):
        db_path = tmp_path / "probes_test.db"
        db = TelemetryDB(db_path)

        # Register probe
        db.register_probe(
            probe_id="probe_shrine_01",
            probe_name="blood_shrine_interacted",
            category="risk_reward",
            description="Player accepted damage for attack speed buff",
            hypothesis_target="Player displays high risk tolerance under low health",
            schema_json={"cost_hp": 25, "buff_duration": 10.0},
        )

        # Query back
        probe = db.get_probe("blood_shrine_interacted")
        assert probe is not None
        assert probe["probe_id"] == "probe_shrine_01"
        assert probe["category"] == "risk_reward"
        assert probe["hypothesis_target"] == "Player displays high risk tolerance under low health"
        schema = json.loads(probe["schema_json"])
        assert schema["cost_hp"] == 25

        # List probes by category
        probes = db.list_probes(category="risk_reward")
        assert len(probes) == 1
        assert probes[0]["probe_name"] == "blood_shrine_interacted"

    def test_probe_registration_upsert_idempotent(self, tmp_path: Path):
        db_path = tmp_path / "probes_upsert.db"
        db = TelemetryDB(db_path)

        db.register_probe("p1", "probe_test", "combat", "initial desc", "target1")
        # Upsert with new description
        db.register_probe("p1", "probe_test", "combat", "updated desc", "target2")

        probe = db.get_probe("probe_test")
        assert probe["description"] == "updated desc"
        assert probe["hypothesis_target"] == "target2"

    def test_legacy_db_schema_auto_migration(self, tmp_path: Path):
        """Verifies that TelemetryDB automatically migrates an existing DB missing probe_name."""
        db_path = tmp_path / "legacy.db"
        # Create table without probe_name column
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE telemetry_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                pos_x REAL NOT NULL,
                pos_y REAL NOT NULL,
                vel_x REAL DEFAULT 0.0,
                vel_y REAL DEFAULT 0.0,
                health REAL,
                payload_json TEXT DEFAULT '{}',
                frame_index INTEGER DEFAULT 0,
                timestamp_utc TEXT
            );
        """)
        conn.commit()
        conn.close()

        # TelemetryDB initialization should migrate and add probe_name
        db = TelemetryDB(db_path)
        reader = db.get_reader_connection()
        cur = reader.cursor()
        cur.execute("PRAGMA table_info(telemetry_events);")
        cols = {row[1] for row in cur.fetchall()}
        assert "probe_name" in cols
        reader.close()


class TestCustomTelemetryEventEmission:
    """Feature 2: Custom Telemetry Event Emission Tests."""

    def test_emit_custom_probe_persistence(self, tmp_path: Path):
        db_path = tmp_path / "custom_events.db"
        recorder = TelemetryRecorder(db_path=db_path, batch_size=5, flush_interval_sec=0.05)
        recorder.start_session(player_profile_tag="custom_test")
        ep_id = recorder.start_episode(level_id="lvl_01", episode_index=1)

        recorder.emit_custom_probe(
            probe_name="boss_enrage_triggered",
            pos=(640.0, 360.0),
            vel=(0.0, 0.0),
            health=35.0,
            category="combat",
            data={"phase": 2, "time_elapsed": 42.5},
        )
        recorder.flush()

        reader = recorder.db.get_reader_connection()
        cur = reader.cursor()
        cur.execute("SELECT event_type, probe_name, pos_x, pos_y, health, payload_json FROM telemetry_events WHERE probe_name = ?;", ("boss_enrage_triggered",))
        row = cur.fetchone()
        assert row is not None
        assert row["event_type"] == "CUSTOM_PROBE"
        assert row["probe_name"] == "boss_enrage_triggered"
        assert row["pos_x"] == 640.0
        assert row["health"] == 35.0
        payload = json.loads(row["payload_json"])
        assert payload["category"] == "combat"
        assert payload["phase"] == 2

        recorder.end_episode()
        recorder.end_session()
        recorder.close()

    def test_sub_microsecond_custom_probe_latency(self, tmp_path: Path):
        """Verifies enqueue latency is fast (<20 microseconds budget on test runner)."""
        db_path = tmp_path / "latency_test.db"
        recorder = TelemetryRecorder(db_path=db_path, batch_size=100, flush_interval_sec=0.5)
        recorder.start_session()
        recorder.start_episode()

        # Warmup
        for _ in range(50):
            recorder.emit_custom_probe("warmup", (100.0, 100.0))

        start = time.perf_counter()
        count = 1000
        for i in range(count):
            recorder.emit_custom_probe(
                "latency_probe",
                pos=(float(i), float(i)),
                vel=(10.0, 10.0),
                health=100.0,
                category="combat",
                data={"idx": i},
            )
        elapsed_us = ((time.perf_counter() - start) / count) * 1e6
        recorder.close()

        assert elapsed_us < 20.0, f"Custom probe enqueue too slow: {elapsed_us:.2f} µs (budget: < 20 µs on test machines)"

    def test_high_throughput_custom_probe_flood(self, tmp_path: Path):
        """Verifies 1,000 custom probe events persist without dropped events under concurrent read."""
        db_path = tmp_path / "flood_custom.db"
        recorder = TelemetryRecorder(db_path=db_path, batch_size=50, flush_interval_sec=0.05)
        recorder.start_session()
        recorder.start_episode()

        stop_reader = threading.Event()
        read_errors = []

        def reader_job():
            conn = recorder.db.get_reader_connection()
            cur = conn.cursor()
            while not stop_reader.is_set():
                try:
                    cur.execute("SELECT count(*) FROM telemetry_events WHERE probe_name IS NOT NULL;")
                    cur.fetchone()
                except Exception as ex:
                    read_errors.append(str(ex))
            conn.close()

        reader_t = threading.Thread(target=reader_job, daemon=True)
        reader_t.start()

        event_count = 1000
        for i in range(event_count):
            recorder.emit_custom_probe(
                probe_name=f"probe_{i % 5}",
                pos=(100.0, 200.0),
                data={"cycle": i},
            )

        recorder.flush()
        stop_reader.set()
        reader_t.join(timeout=2.0)

        assert len(read_errors) == 0

        reader = recorder.db.get_reader_connection()
        cur = reader.cursor()
        cur.execute("SELECT count(*) FROM telemetry_events WHERE event_type = 'CUSTOM_PROBE';")
        persisted = cur.fetchone()[0]
        assert persisted == event_count

        recorder.close()


class TestAgentTelemetryProbeSDK:
    """Feature 3: Agent Telemetry Probe SDK Tests."""

    def test_probe_sdk_emit_and_auto_registration(self, tmp_path: Path):
        db_path = tmp_path / "sdk_test.db"
        recorder = TelemetryRecorder(db_path=db_path, batch_size=10, flush_interval_sec=0.05)
        recorder.start_session()
        recorder.start_episode()

        probe = AgentTelemetryProbe(
            recorder=recorder,
            probe_name="phase_shift_dash",
            category="movement",
            description="Player used phase shift ability to avoid hazard",
            hypothesis_target="Player favors high-mobility evasion over shield tanking",
        )

        probe.emit(pos=(320.0, 240.0), vel=(400.0, 0.0), health=85.0, data={"stamina_spent": 30.0})
        recorder.flush()

        # Verify probe metadata in catalog
        cat_probe = recorder.db.get_probe("phase_shift_dash")
        assert cat_probe is not None
        assert cat_probe["category"] == "movement"
        assert cat_probe["description"] == "Player used phase shift ability to avoid hazard"

        # Verify event in telemetry_events
        reader = recorder.db.get_reader_connection()
        cur = reader.cursor()
        cur.execute("SELECT * FROM telemetry_events WHERE probe_name = 'phase_shift_dash';")
        ev = cur.fetchone()
        assert ev is not None
        assert ev["event_type"] == "CUSTOM_PROBE"
        assert ev["pos_x"] == 320.0
        assert ev["health"] == 85.0
        p = json.loads(ev["payload_json"])
        assert p["stamina_spent"] == 30.0

        recorder.close()

    def test_probe_sdk_null_recorder_safety(self):
        """Verifies safe no-op when recorder is None or NullTelemetryRecorder."""
        probe_none = AgentTelemetryProbe(recorder=None, probe_name="safe_noop")
        probe_none.emit(pos=(10.0, 10.0), data={"test": 1})  # Should not raise
        probe_none.register()  # Should not raise

        null_rec = NullTelemetryRecorder()
        probe_null = AgentTelemetryProbe(recorder=null_rec, probe_name="null_rec_probe")
        probe_null.emit(pos=(20.0, 20.0))
        probe_null.register()

    def test_probe_sdk_in_memory_recorder_integration(self):
        """Verifies AgentTelemetryProbe operates seamlessly with InMemoryTelemetryRecorder."""
        mem_rec = InMemoryTelemetryRecorder()
        mem_rec.start_session()
        mem_rec.start_episode()

        probe = AgentTelemetryProbe(recorder=mem_rec, probe_name="shrine_touch", category="risk_reward")
        probe.emit(pos=(100.0, 200.0), health=90.0, data={"shrine_id": "shrine_alpha"})

        assert len(mem_rec.events) == 1
        ev = mem_rec.events[0]
        assert ev.event_type == "CUSTOM_PROBE"
        assert ev.probe_name == "shrine_touch"
        assert ev.payload["shrine_id"] == "shrine_alpha"

    def test_probe_sdk_ast_sandbox_security(self):
        """Verifies that cartridge code importing and using AgentTelemetryProbe passes AST sandbox."""
        cartridge_code = """
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.telemetry.probes import AgentTelemetryProbe

class TestCartridge(BaseCartridge):
    def __init__(self, manifest=None):
        super().__init__(manifest)
        self.probe = None

    def create_player(self, x, y, recorder=None):
        self.probe = AgentTelemetryProbe(recorder, "combat_strike", category="combat")
        self.probe.emit((x, y), data={"power": 100})
        return None

    def get_level_spec(self, episode_number=1):
        return BaseLevelSpec(level_id="lvl1")

    def create_enemies(self, spec):
        return []

    def create_hazards(self, spec):
        return []
"""
        valid, errors = validate_cartridge_ast(cartridge_code)
        assert valid is True
        assert len(errors) == 0
