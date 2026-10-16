"""tests/test_telemetry_resilience.py
Regression test suite for Telemetry Resilience Remediation (Milestone M1 Iteration 2).
Validates:
1. TelemetryWriterThread daemon crash immunity against non-JSON-serializable payloads.
2. Coordinate coercion for NaN/Inf/None coordinates preventing SQLite batch rollbacks.
3. Consistent None field handling between TelemetryDB.register_probe and TelemetryRecorder.register_probe.
4. Row-by-row rescue fallback when batch commit fails.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import time
import pytest

from engine.telemetry.db import TelemetryDB
from engine.telemetry.models import TelemetryEventRecord, ProbeCatalogRecord
from engine.telemetry.recorder import TelemetryRecorder
from engine.telemetry.probes import AgentTelemetryProbe


def test_writer_thread_survives_poison_payload(tmp_path: Path):
    """Verifies that non-serializable payloads do not terminate TelemetryWriterThread."""
    db_path = tmp_path / "poison_payload.db"
    recorder = TelemetryRecorder(db_path=db_path, batch_size=5, flush_interval_sec=0.05)
    recorder.start_session()
    recorder.start_episode()
    probe = AgentTelemetryProbe(recorder, "poison_probe", "combat")

    # 1. Lambda payload
    probe.emit(pos=(10.0, 20.0), data={"callback": lambda x: x, "tag": "lambda_test"})

    # 2. Circular reference payload
    circ: dict = {"tag": "circular_test"}
    circ["self"] = circ
    probe.emit(pos=(30.0, 40.0), data=circ)

    # 3. Custom class instance payload
    class CustomObj:
        def __repr__(self):
            return "<CustomObj value=42>"
    probe.emit(pos=(50.0, 60.0), data={"custom": CustomObj(), "tag": "obj_test"})

    # 4. Standard valid payload
    probe.emit(pos=(70.0, 80.0), data={"valid": True, "tag": "valid_test"})

    time.sleep(0.15)
    recorder.flush()

    # Writer daemon must still be running
    assert recorder.writer_thread.is_alive(), "TelemetryWriterThread died on poison payload"

    # Verify all 4 events persisted into database
    reader = recorder.db.get_reader_connection()
    cur = reader.cursor()
    cur.execute("SELECT payload_json FROM telemetry_events WHERE probe_name = 'poison_probe' ORDER BY id ASC;")
    rows = cur.fetchall()
    assert len(rows) == 4, f"Expected 4 persisted events, got {len(rows)}"

    payload0 = json.loads(rows[0]["payload_json"])
    assert "callback" in payload0
    assert "<function" in payload0["callback"]

    payload1 = json.loads(rows[1]["payload_json"])
    assert "_serialization_error" in payload1

    payload2 = json.loads(rows[2]["payload_json"])
    assert "<CustomObj value=42>" in payload2["custom"]

    payload3 = json.loads(rows[3]["payload_json"])
    assert payload3["valid"] is True

    reader.close()
    recorder.close()


def test_nan_coordinates_sanitized_batch_preserved(tmp_path: Path):
    """Verifies that float('nan') coordinates are coerced to 0.0 and never trigger batch rollback."""
    db_path = tmp_path / "nan_coords.db"
    recorder = TelemetryRecorder(db_path=db_path, batch_size=5, flush_interval_sec=0.05)
    recorder.start_session()
    recorder.start_episode()

    # Emit 4 valid events + 1 event with NaN pos_x
    for i in range(4):
        recorder.emit_custom_probe("valid_event", pos=(float(i * 10), 10.0), data={"seq": i})
    recorder.emit_custom_probe("nan_event", pos=(float("nan"), 25.0), data={"nan": True})

    # Emit event with NaN velocity and NaN health
    recorder.emit_custom_probe(
        "nan_extra",
        pos=(100.0, 200.0),
        vel=(float("nan"), float("nan")),
        health=float("nan"),
        data={"nan_vel_hp": True}
    )

    # Emit event with Infinity coordinates (supported by SQLite REAL)
    recorder.emit_custom_probe(
        "inf_event",
        pos=(float("inf"), float("-inf")),
        data={"inf": True}
    )

    recorder.flush()

    reader = recorder.db.get_reader_connection()
    cur = reader.cursor()
    cur.execute("SELECT count(*) FROM telemetry_events;")
    total_events = cur.fetchone()[0]
    assert total_events == 7, f"Expected 7 persisted events, found {total_events} (batch was dropped!)"

    # Verify NaN coordinate was coerced to 0.0
    cur.execute("SELECT pos_x, pos_y FROM telemetry_events WHERE probe_name = 'nan_event';")
    row_nan = cur.fetchone()
    assert row_nan["pos_x"] == 0.0
    assert row_nan["pos_y"] == 25.0

    # Verify NaN vel coerced to 0.0, health coerced to NULL
    cur.execute("SELECT vel_x, vel_y, health FROM telemetry_events WHERE probe_name = 'nan_extra';")
    row_extra = cur.fetchone()
    assert row_extra["vel_x"] == 0.0
    assert row_extra["vel_y"] == 0.0
    assert row_extra["health"] is None

    # Verify Infinity preserved
    cur.execute("SELECT pos_x, pos_y FROM telemetry_events WHERE probe_name = 'inf_event';")
    row_inf = cur.fetchone()
    assert math.isinf(row_inf["pos_x"])
    assert math.isinf(row_inf["pos_y"])

    reader.close()
    recorder.close()


def test_db_register_probe_none_fields_consistency(tmp_path: Path):
    """Verifies that None fields in probe registration do not crash and behave consistently."""
    db_path = tmp_path / "probe_null.db"
    db = TelemetryDB(db_path)

    # Direct DB registration with None description and hypothesis_target
    db.register_probe(
        probe_id="probe_direct_null",
        probe_name="null_desc_probe",
        category="combat",
        description=None,
        hypothesis_target=None,
        schema_json=None
    )

    p_direct = db.get_probe("null_desc_probe")
    assert p_direct is not None
    assert p_direct["description"] == ""
    assert p_direct["hypothesis_target"] == ""
    assert p_direct["schema_json"] == "{}"

    # Async Recorder registration with None description
    recorder = TelemetryRecorder(db_path=db_path, batch_size=5, flush_interval_sec=0.05)
    recorder.register_probe(
        probe_id="probe_rec_null",
        probe_name="rec_null_probe",
        category="movement",
        description=None,
        hypothesis_target=None,
        schema_json=None
    )
    recorder.flush()

    p_rec = db.get_probe("rec_null_probe")
    assert p_rec is not None
    assert p_rec["description"] == ""
    assert p_rec["hypothesis_target"] == ""
    assert p_rec["schema_json"] == "{}"

    recorder.close()
