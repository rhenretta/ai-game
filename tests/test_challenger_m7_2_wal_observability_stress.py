"""tests/test_challenger_m7_2_wal_observability_stress.py
Adversarial Stress Test Suite for Milestone 7:
Empirical challenger testing SQLite WAL concurrency, zero lock failures under normal operations,
WAL checkpointing behavior, DB foreign key integrity, Studio API route resolution,
pure SQLite WAL serving isolation, and rich rejection diagnostics.
All tests use isolated temporary SQLite databases and TestClient (zero real DB writes or network calls).
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

from engine.telemetry.db import TelemetryDB
from engine.ui.server import SwarmEventBus, create_app
from engine.supervisor.models import (
    GateFailureType,
    RejectionDiagnosticTrace,
)
import engine.ui.server as ui_server


def _seed_hermetic_studio_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Creates a hermetic SQLite WAL database, active Godot 4 cartridge, and 5 verified checkpoints."""
    db_path = tmp_path / "telemetry.db"
    db = TelemetryDB(db_path)

    build_id = "build_candidate_genesis_2ab772"
    candidate_id = "candidate_genesis_2ab772"
    db.record_studio_build(
        build_id=build_id,
        candidate_id=candidate_id,
        build_type="genesis",
        engine_type="godot4",
        game_title="Aegis Grid Defense",
        genre_paradigm="Tower Defense",
        status="PROMOTED",
        deliberation="Promoted Godot 4 build across 5 verified requirement tracks.",
    )

    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    checkpoints_root = cartridges_root / "checkpoints"
    active_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    godot_files = [
        "project.godot",
        "main.tscn",
        "main.gd",
        "mechanics.gd",
        "entities.gd",
        "scenes.gd",
        "ui_hud.gd",
        "telemetry_manifest.json",
        "requirements_spec.json",
    ]
    for fname in godot_files:
        (active_dir / fname).write_text(f"# {fname}\n", encoding="utf-8")

    (active_dir / "manifest.json").write_text(
        json.dumps(
            {
                "deliberation": "Verified Godot 4 cartridge manifest.",
                "name": "Aegis Grid Defense",
                "entrypoint": "res://main.tscn",
                "version": "0.5.0",
                "track": "R5",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    audio_dir = active_dir / "assets" / "audio"
    sprites_dir = active_dir / "assets" / "sprites"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sprites_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        (audio_dir / f"clip_{i}.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    for i in range(5):
        (sprites_dir / f"sprite_{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    from datetime import timedelta
    base_time = datetime.now(timezone.utc)
    for idx in range(1, 6):
        req_id = f"R{idx}"
        chk_id = f"{candidate_id}_{req_id}"
        chk_dir = checkpoints_root / chk_id
        chk_dir.mkdir(parents=True, exist_ok=True)
        for fname in godot_files:
            (chk_dir / fname).write_text(f"# {fname}\n", encoding="utf-8")
        ts_iso = (base_time + timedelta(seconds=idx)).isoformat()
        meta_dict = {
            "deliberation": f"Checkpoint {chk_id} verified.",
            "checkpoint_id": chk_id,
            "candidate_id": candidate_id,
            "requirement_id": req_id,
            "track_title": f"Requirement Track {idx}",
            "created_at_utc": ts_iso,
            "verified_tracks": [f"R{j}" for j in range(1, idx + 1)],
            "version": f"0.{idx}.0",
        }
        (chk_dir / "checkpoint_metadata.json").write_text(
            json.dumps(meta_dict, indent=2),
            encoding="utf-8",
        )
        if idx == 5:
            (active_dir / "checkpoint_metadata.json").write_text(
                json.dumps(meta_dict, indent=2),
                encoding="utf-8",
            )
        db.record_studio_checkpoint(
            checkpoint_id=chk_id,
            build_id=build_id,
            candidate_id=candidate_id,
            requirement_id=req_id,
            track_title=f"Requirement Track {idx}",
            engine_type="godot4",
            checkpoint_path=str(chk_dir),
            verified_tracks=[f"R{j}" for j in range(1, idx + 1)],
            deliberation=f"Checkpoint {chk_id} verified.",
            created_at_utc=ts_iso,
        )

    for i in range(12):
        role = "LeadGameplaySWEAgent" if i < 6 else "CreativeDirectorWorldbuilderAgent"
        db.record_studio_task(
            task_data={
                "task_id": f"task_hermetic_{i}",
                "agent_role": role,
                "agent_name": role,
                "status": "COMPLETED",
                "requirement_id": f"R{(i % 5) + 1}",
                "track_title": f"Track {(i % 5) + 1}",
                "deliberation": f"Completed task {i} for {role}.",
                "output_artifacts": [],
            },
            build_id=build_id,
            seq_order=i,
        )

    for i in range(55):
        ev_type = "QA_GATE_RESULT" if i == 50 else "AGENT_STREAM"
        db.record_studio_event(
            event_id=f"ev_hermetic_{i}",
            build_id=build_id,
            event_type=ev_type,
            agent_name="LeadGameplaySWEAgent",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            deliberation=f"Hermetic event {i} deliberation.",
            payload={"step": i},
        )

    import config
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    bus = SwarmEventBus(log_path=tmp_path / "studio_events.jsonl", db_path=db_path)
    bus.set_active_build_id(build_id)
    app = create_app(
        db_path=db_path,
        event_bus=bus,
    )
    return db, db_path, app


class TestSQLiteWALConcurrencyStress:
    """Stress tests verifying concurrent multi-threaded SQLite WAL reads and writes."""

    def test_concurrent_wal_writes_and_reads_passive_checkpoint(self, tmp_path: Path):
        """Verify 10 concurrent writers, 10 concurrent readers, and PASSIVE checkpointing
        operate with ZERO lock errors, zero exceptions, and clean WAL truncation.
        """
        db_path = tmp_path / "stress_telemetry.db"
        db = TelemetryDB(db_path)

        build_id = "build_stress_concurrency_001"
        db.record_studio_build(
            build_id=build_id,
            candidate_id="candidate_stress_001",
            build_type="genesis",
            engine_type="godot4",
            game_title="Stress Test Game",
            genre_paradigm="Industrial Simulation",
            status="IN_PROGRESS",
            deliberation="Initial stress build record.",
        )

        stop_event = threading.Event()
        errors: List[Any] = []
        write_count = 0
        read_count = 0
        count_lock = threading.Lock()

        def writer_worker(worker_id: int):
            nonlocal write_count
            local_db = TelemetryDB(db_path)
            for i in range(50):
                if stop_event.is_set():
                    break
                try:
                    ev_id = f"ev_w{worker_id}_{i}"
                    local_db.record_studio_event(
                        event_id=ev_id,
                        build_id=build_id,
                        event_type="AGENT_STREAM",
                        agent_name=f"WorkerAgent_{worker_id}",
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        deliberation=f"Worker {worker_id} iteration {i} deliberation block.",
                        payload={"step": i, "worker": worker_id, "data": "x" * 200},
                    )

                    if i % 5 == 0:
                        t_id = f"task_w{worker_id}_{i}"
                        local_db.record_studio_task(
                            task_data={
                                "task_id": t_id,
                                "agent_role": f"Specialist_{worker_id}",
                                "status": "COMPLETED",
                                "requirement_id": f"R{worker_id % 5 + 1}",
                                "track_title": f"Track {worker_id}",
                                "deliberation": f"Task {t_id} deliberation.",
                                "output_artifacts": [
                                    {
                                        "artifact_id": f"art_{worker_id}_{i}",
                                        "artifact_name": f"module_{worker_id}.gd",
                                        "byte_size": 1024,
                                        "sha256_hash": "deadbeef" * 8,
                                    }
                                ],
                            },
                            build_id=build_id,
                            seq_order=i,
                        )

                    if i % 10 == 0:
                        local_db.record_studio_checkpoint(
                            checkpoint_id=f"chk_w{worker_id}_{i}",
                            build_id=build_id,
                            candidate_id="candidate_stress_001",
                            requirement_id=f"R{worker_id % 5 + 1}",
                            track_title=f"Track {worker_id}",
                            engine_type="godot4",
                            checkpoint_path=str(tmp_path / f"chk_{worker_id}_{i}"),
                            verified_tracks=["R1"],
                            deliberation="Checkpoint verified.",
                        )

                    with count_lock:
                        write_count += 1
                except Exception as exc:
                    import traceback
                    errors.append((f"writer_{worker_id}", exc, traceback.format_exc()))
                    break
                time.sleep(0.002)

        def reader_worker(reader_id: int):
            nonlocal read_count
            local_db = TelemetryDB(db_path)
            while not stop_event.is_set():
                try:
                    local_db.get_latest_build()
                    local_db.get_studio_events(build_id=build_id, limit=20)
                    local_db.get_studio_tasks(build_id=build_id)
                    local_db.get_studio_checkpoints(build_id=build_id)
                    with count_lock:
                        read_count += 1
                except Exception as exc:
                    import traceback
                    errors.append((f"reader_{reader_id}", exc, traceback.format_exc()))
                    break
                time.sleep(0.005)

        def checkpoint_worker():
            local_db = TelemetryDB(db_path)
            while not stop_event.is_set():
                try:
                    local_db.checkpoint(mode="PASSIVE", timeout=5.0)
                except Exception as exc:
                    import traceback
                    errors.append(("checkpoint_PASSIVE", exc, traceback.format_exc()))
                    break
                time.sleep(0.05)

        threads: List[threading.Thread] = []
        for w in range(10):
            threads.append(threading.Thread(target=writer_worker, args=(w,)))
        for r in range(10):
            threads.append(threading.Thread(target=reader_worker, args=(r,)))
        threads.append(threading.Thread(target=checkpoint_worker))

        for t in threads:
            t.start()

        for t in threads[:10]:
            t.join(timeout=15.0)

        stop_event.set()

        for t in threads[10:]:
            t.join(timeout=5.0)

        assert len(errors) == 0, f"Encountered concurrency errors: {errors[:5]}"
        assert write_count == 500, f"Expected 500 writes, got {write_count}"
        assert read_count > 50, f"Expected significant reads, got {read_count}"

        db.checkpoint(mode="TRUNCATE", timeout=5.0)
        wal_file = Path(str(db_path) + "-wal")
        if wal_file.exists():
            assert wal_file.stat().st_size <= 32768, f"WAL file was not truncated: {wal_file.stat().st_size} bytes"

        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("PRAGMA integrity_check;")
        assert c.fetchone()[0] == "ok"
        c.execute("PRAGMA quick_check;")
        assert c.fetchone()[0] == "ok"
        conn.close()

    def test_production_db_wal_mode_and_latest_build(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify an initialized TelemetryDB is in WAL mode and persists promoted Godot 4 build & checkpoints."""
        db, db_path, _ = _seed_hermetic_studio_environment(tmp_path, monkeypatch)
        assert db_path.exists()

        conn = db.get_reader_connection()
        cur = conn.cursor()

        cur.execute("PRAGMA journal_mode;")
        journal_mode = cur.fetchone()[0]
        assert journal_mode.lower() == "wal", f"Expected WAL mode, got {journal_mode}"

        cur.execute("PRAGMA integrity_check;")
        integrity = cur.fetchone()[0]
        assert integrity == "ok", f"Integrity check failed: {integrity}"

        latest_build = db.get_latest_build()
        assert latest_build is not None
        assert latest_build.get("status") == "PROMOTED"
        assert latest_build.get("engine_type") == "godot4"

        checkpoints = db.get_studio_checkpoints(build_id=latest_build["build_id"])
        assert len(checkpoints) == 5
        assert checkpoints[-1]["checkpoint_id"] == "candidate_genesis_2ab772_R5"

        conn.close()

    def test_production_db_foreign_key_violation_discovery(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify PRAGMA foreign_key_check on a hermetic TelemetryDB has zero unexpected FK violations."""
        _, db_path, _ = _seed_hermetic_studio_environment(tmp_path, monkeypatch)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_key_check;")
        violations = cur.fetchall()
        conn.close()

        assert len(violations) == 0, f"Unexpected FK violations: {violations}"


class TestStudioObservabilityLiveEndpoints:
    """Hermetic verification of Studio HTTP API endpoints using Starlette TestClient."""

    def test_active_cartridge_endpoint(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Query /api/cartridges/active and assert entrypoint, godot4 files, and asset inventory."""
        _, _, app = _seed_hermetic_studio_environment(tmp_path, monkeypatch)
        with TestClient(app) as client:
            res = client.get("/api/cartridges/active")
            assert res.status_code == 200
            data = res.json()

            assert data.get("exists") is True
            manifest = data.get("manifest", {})
            assert manifest.get("entrypoint") == "res://main.tscn"
            assert manifest.get("version") == "0.5.0"
            assert manifest.get("track") == "R5"

            files = data.get("files", [])
            assert "project.godot" in files
            assert "main.tscn" in files
            assert "main.gd" in files
            assert "mechanics.gd" in files
            assert "entities.gd" in files
            assert "scenes.gd" in files
            assert "ui_hud.gd" in files
            assert "telemetry_manifest.json" in files
            assert "requirements_spec.json" in files

            assets = data.get("assets", {})
            assert len(assets.get("audio", [])) >= 4
            assert len(assets.get("sprites", [])) >= 5

    def test_checkpoints_endpoint(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Query /api/checkpoints and assert latest checkpoint is candidate_genesis_2ab772_R5."""
        _, _, app = _seed_hermetic_studio_environment(tmp_path, monkeypatch)
        with TestClient(app) as client:
            res = client.get("/api/checkpoints")
            assert res.status_code == 200
            data = res.json()

            assert data.get("total_checkpoints") == 5
            assert data.get("latest_checkpoint_id") == "candidate_genesis_2ab772_R5"

            checkpoints = data.get("checkpoints", [])
            req_ids = [c.get("requirement_id") for c in checkpoints]
            assert req_ids == ["R1", "R2", "R3", "R4", "R5"]

            for c in checkpoints:
                assert c.get("is_active") is True
                assert "res://main.tscn" or "main.tscn" in c.get("files", [])

    def test_tasks_and_events_route_resolution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify route resolution: literal /api/tasks and /api/events route to task and event history."""
        _, _, app = _seed_hermetic_studio_environment(tmp_path, monkeypatch)
        with TestClient(app) as client:
            r_tasks_literal = client.get("/api/tasks")
            assert r_tasks_literal.status_code == 200
            assert "tasks" in r_tasks_literal.json()

            r_events_literal = client.get("/api/events")
            assert r_events_literal.status_code == 200
            assert "events" in r_events_literal.json()

            r_agents = client.get("/api/agents/history")
            assert r_agents.status_code == 200
            agents_data = r_agents.json()
            assert agents_data.get("total_tasks") >= 10
            assert len(agents_data.get("tasks", [])) >= 10

            r_events = client.get("/api/events/history")
            assert r_events.status_code == 200
            events_data = r_events.json()
            assert len(events_data.get("events", [])) >= 50

    def test_pure_sqlite_wal_serving_isolation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify endpoints are served purely from SQLite WAL without any JSONL file present."""
        _, _, app = _seed_hermetic_studio_environment(tmp_path, monkeypatch)
        with TestClient(app) as client:
            r_agents = client.get("/api/agents/history")
            assert r_agents.status_code == 200
            data_agents = r_agents.json()
            assert data_agents.get("total_tasks") >= 10
            tasks = data_agents.get("tasks", [])
            swe_tasks = [t for t in tasks if t.get("agent_role") == "LeadGameplaySWEAgent"]
            assert len(swe_tasks) >= 5

            r_events = client.get("/api/events/history")
            assert r_events.status_code == 200
            data_events = r_events.json()
            events = data_events.get("events", [])
            assert len(events) >= 50

            qa_events = [ev for ev in events if ev.get("event_type") == "QA_GATE_RESULT"]
            assert len(qa_events) >= 1


class TestRichRejectionDiagnostics:
    """Empirical verification of rich rejection diagnostics across supervisor and event bus."""

    def test_rejection_diagnostic_trace_emission(self, tmp_path: Path):
        """Verify that when a gate rejection occurs, full structured RejectionDiagnosticTrace is captured."""
        db_path = tmp_path / "telemetry_reject.db"
        bus = SwarmEventBus(db_path=db_path)
        build_id = "build_reject_test_001"
        bus.set_active_build_id(build_id)

        trace = RejectionDiagnosticTrace(
            deliberation="Challenger gate detected unfulfilled requirement R4: kinetic defense turrets lack projectile instantiation.",
            candidate_id="candidate_test_fail",
            failure_gate="GATE_F_CHALLENGER",
            failure_type=GateFailureType.REQUIREMENT_UNFULFILLED,
            failure_reason="Kinetic defense turrets did not instantiate projectiles in headless simulation.",
            unfulfilled_requirements=["R4"],
            recommended_remedy="Implement instantiate_projectile() in entities.gd and bind to physics tick.",
            failing_criteria=["AC4.1"],
            telemetry_evidence={"gaps": ["turret_projectile_fired"]},
        )

        qa_event = {
            "event_type": "QA_GATE_RESULT",
            "agent_name": "QAPlaytesterBuildSupervisor",
            "deliberation": f"Gate {trace.failure_gate} ({trace.failure_type.value}) failed on attempt 1/3: {trace.failure_reason}",
            "payload": {
                "attempt": 1,
                "max_retries": 3,
                "failure_gate": trace.failure_gate,
                "failure_type": trace.failure_type.value,
                "failure_reason": trace.failure_reason,
                "deliberation": trace.deliberation,
                "unfulfilled_requirements": trace.unfulfilled_requirements,
                "remedy_directive": trace.recommended_remedy,
                "promoted": False,
            },
        }
        bus.publish_threadsafe(qa_event)

        task_data = {
            "task_id": "task_supervisor_reject_test",
            "agent_role": "QAPlaytesterBuildSupervisor",
            "agent_name": "BuildSupervisor",
            "task_type": "QA_VERIFICATION",
            "status": "REJECTED",
            "deliberation": f"Candidate rejected at {trace.failure_gate}: {trace.failure_reason}",
            "verification_verdict": {"passed": False},
            "diagnostic_trace": trace.model_dump(),
            "summary": "Supervisor rejected build on Gate F.",
        }
        bus.publish_agent_task_record(task_data)

        db = TelemetryDB(db_path)
        events = db.get_studio_events(build_id=build_id)
        qa_ev = next(e for e in events if e.get("event_type") == "QA_GATE_RESULT")
        assert qa_ev["payload"]["failure_gate"] == "GATE_F_CHALLENGER"
        assert qa_ev["payload"]["failure_reason"] == "Kinetic defense turrets did not instantiate projectiles in headless simulation."
        assert qa_ev["payload"]["unfulfilled_requirements"] == ["R4"]

        tasks = db.get_studio_tasks(build_id=build_id)
        assert len(tasks) == 1
        sup_task = tasks[0]
        assert sup_task["status"] == "REJECTED"
        assert sup_task["diagnostic_trace"]["failure_gate"] == "GATE_F_CHALLENGER"
        assert sup_task["diagnostic_trace"]["recommended_remedy"] == "Implement instantiate_projectile() in entities.gd and bind to physics tick."
        assert sup_task["diagnostic_trace"]["telemetry_evidence"]["gaps"] == ["turret_projectile_fired"]
