"""tests/test_studio_telemetry_wal.py
Unit and integration tests for normalized relational SQLite WAL studio tables:
- studio_builds
- studio_events
- studio_tasks
- studio_artifacts
- studio_requirements
- studio_checkpoints
Verifies end-to-end integration across TelemetryDB, SwarmEventBus, and Starlette REST APIs.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import pytest
from starlette.testclient import TestClient

from engine.telemetry.db import TelemetryDB
from engine.telemetry.schema import init_studio_schema
from engine.ui.models import AgentTaskRecord, SwarmEventType
from engine.ui.server import SwarmEventBus, create_app


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    db_file = tmp_path / "telemetry_test.db"
    return db_file


class TestStudioTelemetryWAL:
    """Verifies DDL, WAL pragmas, and CRUD operations on studio relational tables."""

    def test_schema_initialization_and_wal_mode(self, test_db_path: Path):
        db = TelemetryDB(test_db_path)
        conn = db.get_reader_connection()
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode;")
            journal_mode = cur.fetchone()[0].lower()
            assert journal_mode == "wal"

            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {r[0] for r in cur.fetchall()}
            expected_tables = {
                "studio_builds",
                "studio_events",
                "studio_tasks",
                "studio_artifacts",
                "studio_requirements",
                "studio_checkpoints",
            }
            assert expected_tables.issubset(tables)
        finally:
            conn.close()

    def test_studio_build_lifecycle(self, test_db_path: Path):
        db = TelemetryDB(test_db_path)
        build_id = "build_test_001"
        db.record_studio_build(
            build_id=build_id,
            candidate_id="candidate_001",
            build_type="genesis",
            engine_type="godot4",
            game_title="Void Drifter",
            genre_paradigm="action_roguelike",
            status="STARTED",
            deliberation="Autonomous Godot 4 build initialized.",
        )

        b = db.get_studio_build(build_id)
        assert b is not None
        assert b["build_id"] == build_id
        assert b["engine_type"] == "godot4"
        assert b["status"] == "STARTED"

        latest = db.get_latest_build()
        assert latest is not None
        assert latest["build_id"] == build_id

        db.update_studio_build(
            build_id=build_id,
            status="PROMOTED",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            deliberation="Candidate passed all gates and promoted.",
        )
        b_updated = db.get_studio_build(build_id)
        assert b_updated["status"] == "PROMOTED"
        assert b_updated["completed_at_utc"] is not None

    def test_studio_events_and_filtering(self, test_db_path: Path):
        db = TelemetryDB(test_db_path)
        b1 = "build_run_1"
        b2 = "build_run_2"
        db.record_studio_build(build_id=b1, game_title="Game 1")
        db.record_studio_build(build_id=b2, game_title="Game 2")

        db.record_studio_event(
            event_id="ev_1",
            build_id=b1,
            event_type="state_change",
            agent_name="StudioOrchestrator",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            deliberation="Starting synthesis",
            payload={"state": "SYNTHESIZING"},
        )
        db.record_studio_event(
            event_id="ev_2",
            build_id=b2,
            event_type="state_change",
            agent_name="StudioOrchestrator",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            deliberation="Starting second build",
            payload={"state": "ANALYZING"},
        )

        events_b1 = db.get_studio_events(build_id=b1)
        assert len(events_b1) == 1
        assert events_b1[0]["event_id"] == "ev_1"
        assert events_b1[0]["payload"]["state"] == "SYNTHESIZING"

        events_b2 = db.get_studio_events(build_id=b2)
        assert len(events_b2) == 1
        assert events_b2[0]["event_id"] == "ev_2"

    def test_studio_tasks_and_artifacts(self, test_db_path: Path):
        db = TelemetryDB(test_db_path)
        build_id = "build_task_test"
        db.record_studio_build(build_id=build_id)

        task_payload = {
            "task_id": "task_swe_001",
            "agent_role": "LeadGameplaySWEAgent",
            "agent_name": "LeadGameplaySWEAgent",
            "requirement_id": "R1",
            "track_title": "Core Mechanics",
            "task_type": "CODE_SYNTHESIS",
            "status": "COMPLETED",
            "start_time_utc": datetime.now(timezone.utc).isoformat(),
            "end_time_utc": datetime.now(timezone.utc).isoformat(),
            "duration_ms": 1250.0,
            "deliberation": "Synthesized main.gd and mechanics.gd",
            "verification_verdict": {"passed": True},
            "output_artifacts": [
                {
                    "artifact_id": "art_main_gd",
                    "artifact_name": "main.gd",
                    "artifact_type": "GDSCRIPT",
                    "file_path": "main.gd",
                    "sha256_hash": "abcdef123456",
                    "byte_size": 2048,
                    "deliberation": "Root SceneTree entrypoint script.",
                }
            ],
            "summary": "Completed R1 mechanics.",
        }

        db.record_studio_task(task_payload, build_id=build_id, seq_order=1)

        tasks = db.get_studio_tasks(build_id=build_id)
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "task_swe_001"
        assert tasks[0]["verification_verdict"]["passed"] is True

        artifacts = db.get_studio_artifacts(build_id=build_id, task_id="task_swe_001")
        assert len(artifacts) == 1
        assert artifacts[0]["artifact_id"] == "art_main_gd"
        assert artifacts[0]["artifact_name"] == "main.gd"

    def test_studio_requirements_and_checkpoints(self, test_db_path: Path):
        db = TelemetryDB(test_db_path)
        build_id = "build_req_test"
        db.record_studio_build(build_id=build_id, game_title="Aether Rift")

        spec = {
            "deliberation": "Backlog for Aether Rift",
            "requirements": [
                {
                    "id": "R1",
                    "title": "Player Controller",
                    "description": "Kinetic CharacterBody2D controls",
                    "status": "in_progress",
                    "responsible_agents": ["LeadGameplaySWEAgent"],
                    "target_modules": ["main.gd", "mechanics.gd"],
                    "acceptance_criteria": [],
                },
                {
                    "id": "R2",
                    "title": "Hazard Grid",
                    "description": "Area2D procedural hazards",
                    "status": "pending",
                    "responsible_agents": ["LeadGameplaySWEAgent"],
                    "target_modules": ["entities.gd"],
                    "acceptance_criteria": [],
                },
            ]
        }

        db.record_studio_requirements(build_id=build_id, spec=spec)
        reqs = db.get_studio_requirements(build_id=build_id)
        assert len(reqs) == 2
        assert reqs[0]["requirement_id"] == "R1"
        assert reqs[0]["target_modules"] == ["main.gd", "mechanics.gd"]

        db.update_studio_requirement_status(build_id=build_id, requirement_id="R1", status="completed")
        updated_reqs = db.get_studio_requirements(build_id=build_id)
        assert updated_reqs[0]["status"] == "completed"

        db.record_studio_checkpoint(
            checkpoint_id="cand_001_R1",
            build_id=build_id,
            candidate_id="cand_001",
            requirement_id="R1",
            track_title="Player Controller",
            engine_type="godot4",
            checkpoint_path="/fake/cartridges/checkpoints/cand_001_R1",
            verified_tracks=["R1"],
            deliberation="Verified checkpoint for R1.",
        )

        cps = db.get_studio_checkpoints(build_id=build_id)
        assert len(cps) == 1
        assert cps[0]["checkpoint_id"] == "cand_001_R1"
        assert cps[0]["verified_tracks"] == ["R1"]


class TestSwarmEventBusAndServerIntegration:
    """Verifies that SwarmEventBus writes directly to SQLite WAL and server endpoints read from it."""

    def test_event_bus_and_api_endpoints_backed_by_sqlite(self, tmp_path: Path):
        db_path = tmp_path / "telemetry.db"
        cartridges_root = tmp_path / "cartridges"
        cartridges_root.mkdir()
        active_dir = cartridges_root / "active"
        active_dir.mkdir()

        bus = SwarmEventBus(db_path=db_path)
        build_id = "build_live_wal_001"
        bus.set_active_build_id(build_id)

        # Record build session in DB
        db = TelemetryDB(db_path)
        db.record_studio_build(
            build_id=build_id,
            game_title="Godot 4 Starship",
            genre_paradigm="space_sim",
            status="SYNTHESIZING",
            deliberation="Live active Godot 4 build session.",
        )

        # Publish requirement spec via bus
        spec = {
            "deliberation": "Godot 4 Starship backlog.",
            "game_title": "Godot 4 Starship",
            "genre_paradigm": "space_sim",
            "requirements": [
                {
                    "id": "R1",
                    "title": "Flight Physics",
                    "description": "6DOF movement in GDScript 4",
                    "status": "completed",
                    "target_modules": ["main.gd", "mechanics.gd"],
                }
            ]
        }
        bus.publish_requirements_spec_generated(
            agent_name="CreativeDirectorWorldbuilderAgent",
            deliberation="Authored GDScript requirements spec.",
            spec=spec,
        )

        # Publish task record via bus
        task = AgentTaskRecord(
            deliberation="Synthesized space flight mechanics in GDScript 4.",
            task_id="task_flight_01",
            agent_role="LeadGameplaySWEAgent",
            agent_name="LeadGameplaySWEAgent",
            requirement_id="R1",
            track_title="Flight Physics",
            task_type="CODE_SYNTHESIS",
            status="COMPLETED",
            start_time_utc=datetime.now(timezone.utc).isoformat(),
            end_time_utc=datetime.now(timezone.utc).isoformat(),
            duration_ms=800.0,
            summary="Flight physics implemented.",
        )
        bus.publish_agent_task_record(task)

        # Record checkpoint
        db.record_studio_checkpoint(
            checkpoint_id="chk_flight_R1",
            build_id=build_id,
            requirement_id="R1",
            track_title="Flight Physics",
            engine_type="godot4",
            checkpoint_path=str(active_dir),
            verified_tracks=["R1"],
            deliberation="Verified flight checkpoint.",
        )

        # Create app backed by this DB and bus
        app = create_app(
            db_path=db_path,
            cartridges_root=cartridges_root,
            active_cartridge_dir=active_dir,
            event_bus=bus,
        )
        client = TestClient(app)

        # 1. Test /api/requirements reads from SQLite WAL
        res_req = client.get("/api/requirements")
        assert res_req.status_code == 200
        req_json = res_req.json()
        assert req_json["game_title"] == "Godot 4 Starship"
        assert len(req_json["requirements"]) == 1
        assert req_json["requirements"][0]["id"] == "R1"
        assert req_json["requirements"][0]["status"] == "completed"

        # 2. Test /api/agents/history reads tasks from SQLite WAL
        res_agents = client.get("/api/agents/history")
        assert res_agents.status_code == 200
        agents_json = res_agents.json()
        assert len(agents_json["tasks"]) >= 1
        assert agents_json["tasks"][0]["task_id"] == "task_flight_01"

        # 3. Test /api/checkpoints reads checkpoints from SQLite WAL
        res_cps = client.get("/api/checkpoints")
        assert res_cps.status_code == 200
        cps_json = res_cps.json()
        assert len(cps_json["checkpoints"]) >= 1
        assert cps_json["checkpoints"][0]["checkpoint_id"] == "chk_flight_R1"

        # 4. Test /api/events/history reads events from SQLite WAL
        res_evs = client.get("/api/events/history")
        assert res_evs.status_code == 200
        evs_json = res_evs.json()
        assert len(evs_json["events"]) >= 1

    def test_events_history_bounds_and_game_bible_compaction(self, tmp_path: Path):
        """Verify /api/events/history bounds row counts, omits duplicate payload_json, and compacts heavy game_bible_updated snapshots while preserving full Game Bible storage."""
        db_path = tmp_path / "bounded_telemetry.db"
        cartridges_root = tmp_path / "cartridges"
        active_dir = cartridges_root / "active"
        cartridges_root.mkdir(parents=True, exist_ok=True)

        bus = SwarmEventBus(history_size=300, db_path=db_path)
        build_id = "build_candidate_genesis_perf01"
        bus.set_active_build_id(build_id)

        heavy_bible = {
            "deliberation": "Initial concept locked.",
            "bible_id": "bible_perf01",
            "cartridge_id": "candidate_genesis_perf01",
            "status": "locked",
            "game_title": "Synthetic Test Project",
            "core_genre": "Simulation",
            "elevator_pitch": "A generic systems test project.",
            "writers_room_transcript": [
                {"utterance_id": f"u_{idx}", "speaker_role": "Lead", "spoken_dialogue": "x" * 2000}
                for idx in range(40)
            ],
            "experience_storyboard": [{"beat": "intro"}],
            "concept_evolution": [{"step": 1}],
            "visual_mockups": [{"path": "mockup.png"}],
            "gameplay_pillars": ["pillar_a", "pillar_b"],
        }
        bus.publish_game_bible_update(heavy_bible)

        for idx in range(350):
            bus.publish_agent_stream(
                agent_name="LeadGameplaySWEAgent",
                deliberation=f"Progress step {idx}",
                token=f"tok_{idx}",
            )

        app = create_app(
            db_path=db_path,
            cartridges_root=cartridges_root,
            active_cartridge_dir=active_dir,
            event_bus=bus,
        )
        client = TestClient(app)

        # Default /api/events/history caps at 200 events and explicit limit=25 returns 25
        res_default = client.get("/api/events/history")
        assert res_default.status_code == 200
        default_events = res_default.json()["events"]
        assert len(default_events) == 200
        for ev in default_events:
            assert "payload_json" not in ev
            assert list(ev.keys())[0] == "deliberation"

        res_limited = client.get("/api/events/history?limit=25&cartridge=candidate_genesis_perf01")
        assert res_limited.status_code == 200
        assert len(res_limited.json()["events"]) == 25

        # Requesting limit=400 includes the first game_bible_updated event in compacted form
        res_all = client.get("/api/events/history?limit=400")
        assert res_all.status_code == 200
        bible_events = [
            ev for ev in res_all.json()["events"]
            if ev.get("event_type") in ("game_bible_updated", "GAME_BIBLE_UPDATED")
        ]
        assert len(bible_events) == 1
        b_payload = bible_events[0]["payload"]
        assert "writers_room_transcript" not in b_payload
        assert b_payload["turn_count"] == 40
        assert b_payload["pillar_count"] == 2
        assert list(b_payload.keys())[0] == "deliberation"

        # Full Game Bible endpoint still serves the complete writers_room_transcript intact
        full_bible = bus.get_game_bible(build_id=build_id)
        assert full_bible is not None
        assert len(full_bible["writers_room_transcript"]) == 40
