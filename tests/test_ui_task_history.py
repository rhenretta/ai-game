"""tests/test_ui_task_history.py
Comprehensive test suite for Requirement R5:
Agent Task History & Handoff Observability UI across models, server endpoints,
cartridge hydration, cryptographic integrity, and static frontend assets.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, List
import uuid

import pytest
from starlette.testclient import TestClient

import config
from engine.studio.models import (
    AcceptanceCriterion,
    GameRequirementsSpec,
    RequirementTrack,
)
from engine.ui.models import (
    AgentTaskDetailResponse,
    AgentTaskHistoryResponse,
    AgentTaskRecord,
    ArtifactMetadataEnvelope,
    RequirementsBacklogResponse,
    SwarmEventType,
)
from engine.ui.server import SwarmEventBus, create_app


@pytest.fixture
def test_setup(tmp_path: Path):
    """Hermetic test fixture provisioning temporary DB, event bus, cartridge directories, and app."""
    cartridges_root = tmp_path / "cartridges"
    cartridges_root.mkdir(parents=True, exist_ok=True)
    active_dir = cartridges_root / "active"
    active_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "telemetry_test.db"
    log_path = tmp_path / "studio_events.jsonl"

    bus = SwarmEventBus(log_path=log_path)
    app = create_app(
        db_path=db_path,
        cartridges_root=cartridges_root,
        active_cartridge_dir=active_dir,
        event_bus=bus,
    )
    client = TestClient(app)

    return {
        "client": client,
        "bus": bus,
        "app": app,
        "cartridges_root": cartridges_root,
        "active_dir": active_dir,
        "tmp_path": tmp_path,
        "log_path": log_path,
    }


# =============================================================================
# Test Class 1: TestAgentTaskHistoryAPI
# =============================================================================
class TestAgentTaskHistoryAPI:
    """Verifies REST endpoints for agent task history querying, filtering, and detail inspection."""

    def test_get_agents_history_empty(self, test_setup):
        client: TestClient = test_setup["client"]
        res = client.get("/api/agents/history")
        assert res.status_code == 200

        data = res.json()
        assert list(data.keys())[0] == "deliberation"
        assert len(data["deliberation"].strip()) > 0
        assert data.get("total_tasks", 0) == 0
        assert data["tasks"] == []

    def test_get_agents_history_populated(self, test_setup):
        client: TestClient = test_setup["client"]
        bus: SwarmEventBus = test_setup["bus"]

        task1 = AgentTaskRecord(
            deliberation="Initial synthesis of core scene loop.",
            task_id="task_swe_001",
            agent_role="LeadGameplaySWEAgent",
            agent_name="GameplaySWE",
            requirement_id="R1",
            track_title="Core Game Loop",
            task_type="CODE_SYNTHESIS",
            status="COMPLETED",
            start_time_utc=datetime.now(timezone.utc).isoformat(),
            end_time_utc=datetime.now(timezone.utc).isoformat(),
            duration_ms=450.0,
            input_artifacts=[],
            output_artifacts=["mechanics.py"],
            verification_verdict={"ast_valid": True},
            summary="Synthesized mechanics.py",
        )
        task2 = AgentTaskRecord(
            deliberation="World spec generation grounded in Steam DNA.",
            task_id="task_director_002",
            agent_role="CreativeDirectorWorldbuilderAgent",
            agent_name="Director",
            requirement_id=None,
            track_title="Genesis World Spec",
            task_type="SPEC_GENERATION",
            status="COMPLETED",
            start_time_utc=datetime.now(timezone.utc).isoformat(),
            end_time_utc=datetime.now(timezone.utc).isoformat(),
            duration_ms=1200.0,
            input_artifacts=[],
            output_artifacts=["world_spec.json"],
            verification_verdict={"pacing_pass": True},
            summary="Generated world spec",
        )

        bus.publish_agent_task_record(task1)
        bus.publish_agent_task_record(task2)

        res = client.get("/api/agents/history")
        assert res.status_code == 200
        data = res.json()
        assert list(data.keys())[0] == "deliberation"
        assert len(data["tasks"]) == 2
        for task_dict in data["tasks"]:
            assert list(task_dict.keys())[0] == "deliberation"

    def test_get_agents_history_filter_by_agent(self, test_setup):
        client: TestClient = test_setup["client"]
        bus: SwarmEventBus = test_setup["bus"]

        t1 = AgentTaskRecord(
            deliberation="Deliberation for SWE task.",
            task_id="task_swe_1",
            agent_role="LeadGameplaySWEAgent",
            agent_name="GameplaySWE",
            requirement_id="R1",
            track_title="Track R1",
            task_type="CODE_SYNTHESIS",
            status="COMPLETED",
            start_time_utc=datetime.now(timezone.utc).isoformat(),
        )
        t2 = AgentTaskRecord(
            deliberation="Deliberation for Analyst task.",
            task_id="task_analyst_2",
            agent_role="SteamAndTelemetryAnalystAgent",
            agent_name="Analyst",
            requirement_id=None,
            track_title="Player DNA",
            task_type="DNA_ANALYSIS",
            status="COMPLETED",
            start_time_utc=datetime.now(timezone.utc).isoformat(),
        )
        bus.publish_agent_task_record(t1)
        bus.publish_agent_task_record(t2)

        res = client.get("/api/agents/history?agent=LeadGameplaySWEAgent")
        assert res.status_code == 200
        data = res.json()
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["task_id"] == "task_swe_1"

    def test_get_agents_history_filter_by_requirement(self, test_setup):
        client: TestClient = test_setup["client"]
        bus: SwarmEventBus = test_setup["bus"]

        t1 = AgentTaskRecord(
            deliberation="Requirement R1 implementation task.",
            task_id="task_r1",
            agent_role="LeadGameplaySWEAgent",
            agent_name="GameplaySWE",
            requirement_id="R1",
            track_title="Track R1",
            task_type="CODE_SYNTHESIS",
            status="COMPLETED",
            start_time_utc=datetime.now(timezone.utc).isoformat(),
        )
        t2 = AgentTaskRecord(
            deliberation="Requirement R2 implementation task.",
            task_id="task_r2",
            agent_role="LeadGameplaySWEAgent",
            agent_name="GameplaySWE",
            requirement_id="R2",
            track_title="Track R2",
            task_type="CODE_SYNTHESIS",
            status="COMPLETED",
            start_time_utc=datetime.now(timezone.utc).isoformat(),
        )
        bus.publish_agent_task_record(t1)
        bus.publish_agent_task_record(t2)

        res = client.get("/api/agents/history?requirement_id=R2")
        assert res.status_code == 200
        data = res.json()
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["task_id"] == "task_r2"

    def test_get_agents_history_filter_by_status(self, test_setup):
        client: TestClient = test_setup["client"]
        bus: SwarmEventBus = test_setup["bus"]

        t1 = AgentTaskRecord(
            deliberation="Completed task deliberation.",
            task_id="task_pass",
            agent_role="LeadGameplaySWEAgent",
            agent_name="GameplaySWE",
            requirement_id="R1",
            track_title="Track R1",
            task_type="CODE_SYNTHESIS",
            status="COMPLETED",
            start_time_utc=datetime.now(timezone.utc).isoformat(),
        )
        t2 = AgentTaskRecord(
            deliberation="Rejected task deliberation on smoke check failure.",
            task_id="task_fail",
            agent_role="LeadGameplaySWEAgent",
            agent_name="GameplaySWE",
            requirement_id="R1",
            track_title="Track R1",
            task_type="CODE_SYNTHESIS",
            status="REJECTED",
            start_time_utc=datetime.now(timezone.utc).isoformat(),
        )
        bus.publish_agent_task_record(t1)
        bus.publish_agent_task_record(t2)

        res = client.get("/api/agents/history?status=rejected")
        assert res.status_code == 200
        data = res.json()
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["task_id"] == "task_fail"

    def test_get_agents_history_pagination_and_bounds_clamping(self, test_setup):
        client: TestClient = test_setup["client"]
        bus: SwarmEventBus = test_setup["bus"]

        for i in range(15):
            t = AgentTaskRecord(
                deliberation=f"Task {i} deliberation.",
                task_id=f"task_{i}",
                agent_role="LeadGameplaySWEAgent",
                agent_name="GameplaySWE",
                requirement_id="R1",
                track_title="Batch",
                task_type="TEST",
                status="COMPLETED",
                start_time_utc=datetime.now(timezone.utc).isoformat(),
            )
            bus.publish_agent_task_record(t)

        # Negative bounds are clamped safely
        res = client.get("/api/agents/history?limit=-10&offset=-5")
        assert res.status_code == 200
        data = res.json()
        assert len(data["tasks"]) > 0

        # Oversized limit is clamped
        res2 = client.get("/api/agents/history?limit=1000")
        assert res2.status_code == 200
        data2 = res2.json()
        assert len(data2["tasks"]) == 15

    def test_get_task_detail_by_id_success(self, test_setup):
        client: TestClient = test_setup["client"]
        bus: SwarmEventBus = test_setup["bus"]

        task = AgentTaskRecord(
            deliberation="Unique deliberation for single task retrieval.",
            task_id="task_detail_target_999",
            agent_role="VisualArtSpriteDesignerAgent",
            agent_name="VisualArtist",
            requirement_id="R2",
            track_title="Sprite Assets",
            task_type="SPRITE_SYNTHESIS",
            status="COMPLETED",
            start_time_utc=datetime.now(timezone.utc).isoformat(),
            output_artifacts=["player.png"],
            verification_verdict={"dimensions": [16, 16]},
        )
        bus.publish_agent_task_record(task)

        res = client.get("/api/agents/history/task_detail_target_999")
        assert res.status_code == 200
        data = res.json()
        assert list(data.keys())[0] == "deliberation"
        assert data["task"]["task_id"] == "task_detail_target_999"
        assert data["task"]["agent_role"] == "VisualArtSpriteDesignerAgent"

    def test_get_task_detail_not_found(self, test_setup):
        client: TestClient = test_setup["client"]
        res = client.get("/api/agents/history/task_missing_does_not_exist")
        assert res.status_code == 404
        data = res.json()
        assert "not found" in data["detail"].lower()

    def test_get_task_detail_path_traversal_and_null_bytes(self, test_setup):
        client: TestClient = test_setup["client"]
        res1 = client.get("/api/agents/history/..%2F..%2Fetc%2Fpasswd")
        assert res1.status_code in (400, 404)

        res2 = client.get("/api/agents/history/task_123%00injection")
        assert res2.status_code in (400, 404)


# =============================================================================
# Test Class 2: TestRequirementsBacklogAPI
# =============================================================================
class TestRequirementsBacklogAPI:
    """Verifies live requirements specification and backlog status streaming."""

    def test_get_requirements_empty(self, test_setup):
        client: TestClient = test_setup["client"]
        res = client.get("/api/requirements")
        assert res.status_code == 200
        data = res.json()
        assert list(data.keys())[0] == "deliberation"
        assert (data.get("has_active_spec") is False) or (data.get("has_spec") is False)
        assert data.get("requirements_spec") is None

    def test_get_requirements_spec_populated(self, test_setup):
        client: TestClient = test_setup["client"]
        bus: SwarmEventBus = test_setup["bus"]

        spec = GameRequirementsSpec(
            deliberation="Backlog decomposition derived from player telemetry.",
            spec_id="req_spec_test_001",
            game_title="Quantum Grid Run",
            genre_paradigm="Action Roguelike",
            target_audience_summary="High-speed kinematic players.",
            requirements=[
                RequirementTrack(
                    deliberation="Player velocity and arena collision loop.",
                    id="R1",
                    title="Core Kinematic Movement",
                    description="Continuous 2D player velocity and kinematic collision loop.",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="Verify player movement update dt.",
                            id="AC1.1",
                            description="Player moves with deterministic dt.",
                            verification_method="headless_playtest",
                            passed=False,
                        )
                    ],
                    status="pending",
                )
            ],
            asset_manifest=[],
        )
        bus.publish_requirements_spec(spec)

        res = client.get("/api/requirements")
        assert res.status_code == 200
        data = res.json()
        assert list(data.keys())[0] == "deliberation"
        assert (data.get("has_active_spec") is True) or (data.get("has_spec") is True)
        assert data["requirements_spec"]["spec_id"] == "req_spec_test_001"
        assert data["requirements_spec"]["game_title"] == "Quantum Grid Run"
        assert len(data["requirements_spec"]["requirements"]) == 1

    def test_get_requirements_status_update(self, test_setup):
        client: TestClient = test_setup["client"]
        bus: SwarmEventBus = test_setup["bus"]

        spec = GameRequirementsSpec(
            deliberation="Backlog initial state.",
            spec_id="req_spec_update_test",
            game_title="Update Simulation",
            genre_paradigm="Strategy",
            target_audience_summary="Tactical players.",
            requirements=[
                RequirementTrack(
                    deliberation="2D Grid rendering.",
                    id="R1",
                    title="Grid Map",
                    description="2D Grid map generation and tile rendering.",
                    acceptance_criteria=[],
                    status="pending",
                )
            ],
            asset_manifest=[],
        )
        bus.publish_requirements_spec(spec)

        bus.publish_requirement_status_update(
            agent_name="LeadGameplaySWEAgent",
            deliberation="Track R1 completed headless smoke test.",
            requirement_id="R1",
            status="completed",
            track_title="Grid Map",
        )

        res = client.get("/api/requirements")
        assert res.status_code == 200
        data = res.json()
        track = data["requirements_spec"]["requirements"][0]
        assert track["id"] == "R1"
        assert track["status"] == "completed"


# =============================================================================
# Test Class 3: TestTaskHistoryPersistenceAcrossReloads
# =============================================================================
class TestTaskHistoryPersistenceAcrossReloads:
    """Verifies cartridge task history persistence, fallback hydration, and pytest isolation."""

    def test_persistence_from_task_history_json(self, test_setup):
        bus: SwarmEventBus = test_setup["bus"]
        cartridge_dir = test_setup["tmp_path"] / "active_cartridge_persistence"
        cartridge_dir.mkdir(parents=True, exist_ok=True)

        history_file = cartridge_dir / "task_history.json"
        task_payload = {
            "deliberation": "Persisted task execution history.",
            "plan_id": "plan_persisted_001",
            "game_title": "Persisted Game",
            "tasks": [
                {
                    "deliberation": "Synthesized mechanics from persistent disk.",
                    "task_id": "task_disk_001",
                    "agent_role": "LeadGameplaySWEAgent",
                    "agent_name": "GameplaySWE",
                    "requirement_id": "R1",
                    "track_title": "Core Mechanics",
                    "task_type": "CODE_SYNTHESIS",
                    "status": "COMPLETED",
                    "start_time_utc": datetime.now(timezone.utc).isoformat(),
                    "end_time_utc": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": 320.0,
                    "input_artifacts": [],
                    "output_artifacts": ["mechanics.py"],
                    "verification_verdict": {"ast_check": True},
                    "summary": "Mechanics loaded from task_history.json",
                }
            ],
        }
        history_file.write_text(json.dumps(task_payload, indent=2), encoding="utf-8")

        # Hydrate task records from cartridge directory
        records = bus.get_task_records(cartridge_dir=cartridge_dir)
        assert len(records) >= 1
        matching = [r for r in records if r.get("task_id") == "task_disk_001"]
        assert len(matching) == 1
        assert matching[0].get("status") == "COMPLETED"
        assert matching[0].get("deliberation") == "Synthesized mechanics from persistent disk."

    def test_fallback_cartridge_hydration_when_task_history_missing(self, test_setup):
        bus: SwarmEventBus = test_setup["bus"]
        cartridge_dir = test_setup["tmp_path"] / "cartridge_missing_history"
        cartridge_dir.mkdir(parents=True, exist_ok=True)

        manifest = {"title": "Fallback Cartridge", "genre": "Arcade", "version": "1.0.0"}
        (cartridge_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (cartridge_dir / "mechanics.py").write_text("# mechanics code\nclass Scene: pass\n", encoding="utf-8")

        # Fallback hydration synthesizes baseline records
        records = bus._hydrate_from_cartridge_artifacts(cartridge_dir)
        assert len(records) >= 1
        assert any(r.get("agent_role") == "LeadGameplaySWEAgent" for r in records)
        for r in records:
            assert len(r.get("deliberation", "")) > 0
            if r.get("output_artifacts"):
                assert any(isinstance(art, dict) and "sha256" in art for art in r["output_artifacts"])

    def test_pytest_log_path_isolation(self, tmp_path: Path):
        test_log = tmp_path / "custom_test_events.jsonl"
        event_dict = {
            "event_type": "AGENT_TASK_RECORD",
            "agent_name": "LeadGameplaySWEAgent",
            "deliberation": "Isolated test event deliberation.",
            "payload": {
                "deliberation": "Isolated test event deliberation.",
                "task_id": "task_isolated_pytest_001",
                "agent_role": "LeadGameplaySWEAgent",
                "agent_name": "GameplaySWE",
                "requirement_id": "R1",
                "track_title": "Isolation",
                "task_type": "ISOLATION_TEST",
                "status": "COMPLETED",
                "start_time_utc": datetime.now(timezone.utc).isoformat(),
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        test_log.write_text(json.dumps(event_dict) + "\n", encoding="utf-8")

        isolated_bus = SwarmEventBus(log_path=test_log)
        task = isolated_bus.get_task_record_by_id("task_isolated_pytest_001")
        assert task is not None
        assert task.get("deliberation") == "Isolated test event deliberation."


# =============================================================================
# Test Class 4: TestArtifactMetadataEnvelopeIntegrity
# =============================================================================
class TestArtifactMetadataEnvelopeIntegrity:
    """Verifies ArtifactMetadataEnvelope cryptographic hashing, sealing, and tampering detection."""

    def test_compute_file_sha256(self, tmp_path: Path):
        sample_file = tmp_path / "sample_code.py"
        code = "print('Authentic SHA256 Verification Test')\n"
        sample_file.write_text(code, encoding="utf-8")

        expected_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        computed_hash = ArtifactMetadataEnvelope.compute_file_sha256(sample_file)
        assert computed_hash == expected_hash

    def test_seal_and_verify_integrity(self, tmp_path: Path):
        code_file = tmp_path / "mechanics.py"
        code_file.write_text("class TestCartridge: pass\n", encoding="utf-8")

        envelope = ArtifactMetadataEnvelope(
            deliberation="Sealing mechanics module following code synthesis.",
            artifact_id="art_env_001",
            artifact_type="CODE_MODULE",
            source_agent="LeadGameplaySWEAgent",
            target_agent="QAPlaytesterBuildSupervisor",
            payload_path=str(code_file),
        )

        envelope.seal(code_file)
        assert envelope.output_sha256 is not None
        assert len(envelope.output_sha256) == 64
        assert envelope.sealed_at_utc is not None
        assert envelope.verify_integrity(code_file) is True

    def test_tampered_file_fails_integrity(self, tmp_path: Path):
        code_file = tmp_path / "tamper_target.py"
        code_file.write_text("original_content = True\n", encoding="utf-8")

        envelope = ArtifactMetadataEnvelope(
            deliberation="Sealing before simulated adversarial tampering.",
            artifact_id="art_env_tamper",
            artifact_type="CODE_MODULE",
            source_agent="LeadGameplaySWEAgent",
        )
        envelope.seal(code_file)
        assert envelope.verify_integrity(code_file) is True

        # Adversarially modify file
        code_file.write_text("original_content = False # TAMPERED\n", encoding="utf-8")
        assert envelope.verify_integrity(code_file) is False

    def test_envelope_deliberation_first(self):
        assert list(ArtifactMetadataEnvelope.model_fields.keys())[0] == "deliberation"

        env = ArtifactMetadataEnvelope(
            deliberation="Envelope deliberation test.",
            artifact_id="art_1",
            artifact_type="CODE_MODULE",
            source_agent="LeadGameplaySWEAgent",
        )
        data = json.loads(env.model_dump_json())
        assert list(data.keys())[0] == "deliberation"


# =============================================================================
# Test Class 5: TestFrontendStaticAssetIntegrity
# =============================================================================
class TestFrontendStaticAssetIntegrity:
    """Verifies engine/ui/static/index.html structure, mandatory DOM elements, and JS syntax."""

    @pytest.fixture
    def index_content(self) -> str:
        index_path = (
            Path(__file__).resolve().parent.parent
            / "engine"
            / "ui"
            / "static"
            / "index.html"
        )
        assert index_path.exists(), f"index.html missing at {index_path}"
        return index_path.read_text(encoding="utf-8")

    def test_html5_structure_and_title(self, index_content: str):
        assert index_content.strip().startswith("<!DOCTYPE html>")
        assert "<title>Mission Control</title>" in index_content

    def test_r5_mandatory_dom_elements(self, index_content: str):
        mandatory_r5_ids = [
            "task-history-drawer",
            "agent-inspector",
            "requirements-backlog-container",
            "task-cards-list",
            "task-agent-filter",
            "task-status-filter",
            "task-search-input",
            "handoff-viewer",
        ]
        for dom_id in mandatory_r5_ids:
            assert (
                f'id="{dom_id}"' in index_content
            ), f"Mandatory R5 DOM ID '{dom_id}' missing in index.html"

    def test_zero_build_and_zero_npm_invariance(self, index_content: str):
        assert "node_modules" not in index_content
        assert "webpack" not in index_content
        assert "<script src=" not in index_content, "External script tags are forbidden; UI must be zero-build"

    def test_javascript_syntax_validity(self, index_content: str):
        scripts = re.findall(r"<script>(.*?)</script>", index_content, re.DOTALL)
        assert len(scripts) >= 1, "At least one inline <script> block expected in index.html"

        if shutil.which("node"):
            for i, script_body in enumerate(scripts):
                proc = subprocess.run(
                    ["node", "--check"],
                    input=script_body,
                    capture_output=True,
                    text=True,
                )
                assert proc.returncode == 0, f"Inline JS syntax error in script block {i}:\n{proc.stderr}"
        else:
            for script_body in scripts:
                open_braces = script_body.count("{") - script_body.count("}")
                open_parens = script_body.count("(") - script_body.count(")")
                open_brackets = script_body.count("[") - script_body.count("]")
                assert open_braces == 0, f"Unbalanced braces in JS: diff={open_braces}"
                assert open_parens == 0, f"Unbalanced parentheses in JS: diff={open_parens}"
                assert open_brackets == 0, f"Unbalanced square brackets in JS: diff={open_brackets}"

    def test_localstorage_persistence_contracts(self, index_content: str):
        assert "localStorage.getItem" in index_content
        assert "localStorage.setItem" in index_content
        assert "studio_observability_state_v1" in index_content
