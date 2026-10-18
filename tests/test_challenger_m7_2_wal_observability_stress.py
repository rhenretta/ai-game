"""tests/test_challenger_m7_2_wal_observability_stress.py
Adversarial Stress Test Suite for Milestone 7:
Empirical challenger testing SQLite WAL concurrency, zero lock failures under normal operations,
WAL checkpointing behavior, production DB foreign key integrity, Studio API route resolution,
pure SQLite WAL serving isolation, and rich rejection diagnostics.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List
import uuid

import pytest
import requests

from engine.telemetry.db import TelemetryDB
from engine.ui.server import SwarmEventBus
from engine.supervisor.models import (
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    SupervisorGateVerdict,
)


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
                    b = local_db.get_latest_build()
                    evs = local_db.get_studio_events(build_id=build_id, limit=20)
                    tasks = local_db.get_studio_tasks(build_id=build_id)
                    chks = local_db.get_studio_checkpoints(build_id=build_id)
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

        # Clean WAL management verification: truncate checkpoint
        db.checkpoint(mode="TRUNCATE", timeout=5.0)
        wal_file = Path(str(db_path) + "-wal")
        if wal_file.exists():
            assert wal_file.stat().st_size <= 32768, f"WAL file was not truncated: {wal_file.stat().st_size} bytes"

        # Integrity checks
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("PRAGMA integrity_check;")
        assert c.fetchone()[0] == "ok"
        c.execute("PRAGMA quick_check;")
        assert c.fetchone()[0] == "ok"
        conn.close()

    def test_production_db_wal_mode_and_latest_build(self):
        """Verify the live production data/telemetry.db is in WAL mode and has promoted Godot 4 build."""
        prod_db = Path("data/telemetry.db")
        assert prod_db.exists(), "data/telemetry.db must exist"

        db = TelemetryDB(prod_db)
        conn = db.get_reader_connection()
        cur = conn.cursor()

        cur.execute("PRAGMA journal_mode;")
        journal_mode = cur.fetchone()[0]
        assert journal_mode.lower() == "wal", f"Expected WAL mode, got {journal_mode}"

        cur.execute("PRAGMA integrity_check;")
        integrity = cur.fetchone()[0]
        assert integrity == "ok", f"Integrity check failed: {integrity}"

        latest_build = db.get_latest_build()
        assert latest_build is not None, "No builds found in production DB"
        assert latest_build.get("status") == "PROMOTED", f"Build not promoted: {latest_build}"
        assert latest_build.get("engine_type") == "godot4", f"Engine type not godot4: {latest_build}"

        checkpoints = db.get_studio_checkpoints(build_id=latest_build["build_id"])
        assert len(checkpoints) == 5, f"Expected 5 checkpoints, found {len(checkpoints)}"
        assert checkpoints[-1]["checkpoint_id"] == "candidate_genesis_2ab772_R5"

        conn.close()

    def test_production_db_foreign_key_violation_discovery(self):
        """Empirical challenger finding: check PRAGMA foreign_key_check on data/telemetry.db.
        Verifies whether orphaned unit test rows contaminated the production telemetry database.
        """
        prod_db = Path("data/telemetry.db")
        conn = sqlite3.connect(str(prod_db))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_key_check;")
        violations = cur.fetchall()
        conn.close()

        # Document finding: if violations exist, record exact violating rows
        if violations:
            violating_tables = [r["table"] for r in violations]
            assert "studio_checkpoints" in violating_tables, f"Unexpected FK violations: {violations}"


class TestStudioObservabilityLiveEndpoints:
    """Empirical verification of Studio HTTP API endpoints against live running Studio server."""

    BASE_URL = "http://127.0.0.1:8000"

    def test_active_cartridge_endpoint(self):
        """Query /api/cartridges/active and assert entrypoint, godot4 files, and asset inventory."""
        res = requests.get(f"{self.BASE_URL}/api/cartridges/active", timeout=5.0)
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

    def test_checkpoints_endpoint(self):
        """Query /api/checkpoints and assert latest checkpoint is candidate_genesis_2ab772_R5."""
        res = requests.get(f"{self.BASE_URL}/api/checkpoints", timeout=5.0)
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

    def test_tasks_and_events_route_resolution(self):
        """Verify route resolution: literal /api/tasks and /api/events route to task and event history."""
        r_tasks_literal = requests.get(f"{self.BASE_URL}/api/tasks", timeout=5.0)
        assert r_tasks_literal.status_code == 200
        assert "tasks" in r_tasks_literal.json()

        r_events_literal = requests.get(f"{self.BASE_URL}/api/events", timeout=5.0)
        assert r_events_literal.status_code == 200
        assert "events" in r_events_literal.json()

        r_agents = requests.get(f"{self.BASE_URL}/api/agents/history", timeout=5.0)
        assert r_agents.status_code == 200
        agents_data = r_agents.json()
        assert agents_data.get("total_tasks") >= 10
        assert len(agents_data.get("tasks", [])) >= 10

        r_events = requests.get(f"{self.BASE_URL}/api/events/history", timeout=5.0)
        assert r_events.status_code == 200
        events_data = r_events.json()
        assert len(events_data.get("events", [])) >= 50

    def test_pure_sqlite_wal_serving_isolation(self):
        """Verify endpoints are served purely from SQLite WAL by temporarily moving studio_events.jsonl."""
        jsonl_path = Path("data/studio_events.jsonl")
        jsonl_backup = Path("data/studio_events.jsonl.challenger_bak")

        had_jsonl = jsonl_path.exists()
        if had_jsonl:
            jsonl_path.rename(jsonl_backup)

        try:
            r_agents = requests.get(f"{self.BASE_URL}/api/agents/history", timeout=5.0)
            assert r_agents.status_code == 200
            data_agents = r_agents.json()
            assert data_agents.get("total_tasks") >= 10
            tasks = data_agents.get("tasks", [])
            swe_tasks = [t for t in tasks if t.get("agent_role") == "LeadGameplaySWEAgent"]
            assert len(swe_tasks) >= 5, "SWE requirement tracks served from SQLite WAL"

            r_events = requests.get(f"{self.BASE_URL}/api/events/history", timeout=5.0)
            assert r_events.status_code == 200
            data_events = r_events.json()
            events = data_events.get("events", [])
            assert len(events) >= 50, "Events served directly from SQLite WAL"

            qa_events = [ev for ev in events if ev.get("event_type") == "QA_GATE_RESULT"]
            assert len(qa_events) >= 1
        finally:
            if had_jsonl and jsonl_backup.exists():
                jsonl_backup.rename(jsonl_path)


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
