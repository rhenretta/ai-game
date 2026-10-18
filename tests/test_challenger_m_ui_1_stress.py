"""tests/test_challenger_m_ui_1_stress.py
Adversarial empirical stress tests authored by Challenger 1 for Milestone M_UI.
Evaluates REST endpoints (/api/agents/history, /api/agents/history/{task_id}, /api/requirements),
epistemic contracts (Rule 4 deliberation-first ordering), and ArtifactMetadataEnvelope integrity.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import uuid

import pytest
from starlette.testclient import TestClient

from engine.studio.models import AcceptanceCriterion, GameRequirementsSpec, RequirementTrack
from engine.ui.models import (
    AgentTaskDetailResponse,
    AgentTaskHistoryResponse,
    AgentTaskRecord,
    ArtifactMetadataEnvelope,
    RequirementsBacklogResponse,
)
from engine.ui.server import SwarmEventBus, create_app


@pytest.fixture
def stress_setup(tmp_path: Path):
    """Hermetic test fixture provisioning isolated bus, log, cartridges, and test client."""
    cartridges_root = tmp_path / "cartridges"
    cartridges_root.mkdir(parents=True, exist_ok=True)
    active_dir = cartridges_root / "active"
    active_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "telemetry_stress.db"
    log_path = tmp_path / "studio_events_stress.jsonl"

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
# Suite 1: GET /api/agents/history Stress & Edge Cases
# =============================================================================
class TestAgentHistoryEndpointStress:
    """Stress tests limits, offsets, bounds clamping, special characters, and high volume."""

    def test_extreme_limits_and_offsets(self, stress_setup):
        client: TestClient = stress_setup["client"]
        bus: SwarmEventBus = stress_setup["bus"]

        # Populate with 10 records
        for i in range(10):
            bus.publish_agent_task_record(
                AgentTaskRecord(
                    deliberation=f"Deliberation {i}",
                    task_id=f"task_{i:02d}",
                    agent_role="LeadGameplaySWEAgent",
                    agent_name="SWE",
                    start_time_utc=datetime.now(timezone.utc).isoformat(),
                )
            )

        # 1. Negative limits clamped to 1
        res = client.get("/api/agents/history?limit=-50")
        assert res.status_code == 200
        d = res.json()
        assert list(d.keys())[0] == "deliberation"
        assert len(d["tasks"]) == 1

        # 2. Limit 0 clamped to 1
        res = client.get("/api/agents/history?limit=0")
        assert res.status_code == 200
        assert len(res.json()["tasks"]) == 1

        # 3. Very large limits clamped to 500
        res = client.get("/api/agents/history?limit=999999999")
        assert res.status_code == 200
        assert len(res.json()["tasks"]) == 10

        # 4. Non-integer limits default to 50
        for bad_limit in ["abc", "3.14", "true", "null"]:
            res = client.get(f"/api/agents/history?limit={bad_limit}")
            assert res.status_code == 200
            assert list(res.json().keys())[0] == "deliberation"

        # 5. Negative offsets clamped to 0
        res = client.get("/api/agents/history?offset=-10")
        assert res.status_code == 200
        assert len(res.json()["tasks"]) == 10

        # 6. Out-of-bounds offsets return empty tasks list
        res = client.get("/api/agents/history?offset=1000")
        assert res.status_code == 200
        d = res.json()
        assert d["tasks"] == []
        assert d["total_tasks"] == 10

    def test_query_filters_special_characters_and_security(self, stress_setup):
        client: TestClient = stress_setup["client"]
        bus: SwarmEventBus = stress_setup["bus"]

        bus.publish_agent_task_record(
            AgentTaskRecord(
                deliberation="Task for filter test.",
                task_id="task_filter_01",
                agent_role="LeadGameplaySWEAgent",
                agent_name="SWE",
                requirement_id="R1",
                status="COMPLETED",
                start_time_utc=datetime.now(timezone.utc).isoformat(),
            )
        )

        # 1. Malicious null bytes in query parameters rejected with 400 and deliberation first
        for param in ["agent_role", "agent", "status", "requirement_id"]:
            res = client.get(f"/api/agents/history?{param}=malicious%00payload")
            assert res.status_code == 400
            d = res.json()
            assert list(d.keys())[0] == "deliberation"
            assert "error" in d

        # 2. Oversized parameter strings (> 255 chars) rejected with 400 and deliberation first
        oversized = "A" * 256
        for param in ["agent_role", "status", "requirement_id"]:
            res = client.get(f"/api/agents/history?{param}={oversized}")
            assert res.status_code == 400
            d = res.json()
            assert list(d.keys())[0] == "deliberation"

        # 3. Special characters (SQL injection, XSS, unicode, emojis) handled cleanly
        for weird in ["' OR 1=1 --", "<script>alert(1)</script>", "🔥🎮✨", "R1; DROP TABLE tasks;"]:
            res = client.get(f"/api/agents/history?agent={weird}")
            assert res.status_code == 200
            d = res.json()
            assert list(d.keys())[0] == "deliberation"
            assert d["tasks"] == []

        # 4. Non-existent agent returns 200 with empty task list
        res = client.get("/api/agents/history?agent=NonExistentPhantomAgent")
        assert res.status_code == 200
        d = res.json()
        assert d["tasks"] == []
        assert list(d.keys())[0] == "deliberation"

    def test_high_volume_pagination(self, stress_setup):
        client: TestClient = stress_setup["client"]
        bus: SwarmEventBus = stress_setup["bus"]

        total_records = 500
        for i in range(total_records):
            bus.publish_agent_task_record(
                AgentTaskRecord(
                    deliberation=f"High-volume task {i:03d} deliberation.",
                    task_id=f"hv_task_{i:03d}",
                    agent_role="LeadGameplaySWEAgent" if i % 2 == 0 else "VoiceActingAudioProducerAgent",
                    agent_name="SWE" if i % 2 == 0 else "Voice",
                    requirement_id=f"R{(i % 4) + 1}",
                    status="COMPLETED" if i % 3 != 0 else "REJECTED",
                    start_time_utc=datetime.now(timezone.utc).isoformat(),
                )
            )

        # Page 1: 200 items
        res = client.get("/api/agents/history?limit=200&offset=0")
        assert res.status_code == 200
        d = res.json()
        assert len(d["tasks"]) == 200
        assert d["total_tasks"] == total_records
        assert d["tasks"][0]["task_id"] == "hv_task_000"
        assert d["tasks"][-1]["task_id"] == "hv_task_199"

        # Page 2: 200 items
        res = client.get("/api/agents/history?limit=200&offset=200")
        assert res.status_code == 200
        d = res.json()
        assert len(d["tasks"]) == 200
        assert d["tasks"][0]["task_id"] == "hv_task_200"

        # Page 3: 100 items (remainder)
        res = client.get("/api/agents/history?limit=200&offset=400")
        assert res.status_code == 200
        d = res.json()
        assert len(d["tasks"]) == 100
        assert d["tasks"][-1]["task_id"] == f"hv_task_{total_records - 1:03d}"


# =============================================================================
# Suite 2: GET /api/agents/history/{task_id} Path Traversal & Attacks
# =============================================================================
class TestAgentHistoryDetailEndpointStress:
    """Stress tests path traversal payloads, null bytes, oversized IDs, and non-existent IDs."""

    def test_path_traversal_detection(self, stress_setup):
        client: TestClient = stress_setup["client"]

        # 1. Traversal with double dots inside path parameter
        traversal_ids = [
            "task..traversal",
            "..traversal",
            "task_id..",
            "task_id%00inject",
        ]
        for tid in traversal_ids:
            res = client.get(f"/api/agents/history/{tid}")
            assert res.status_code in (400, 404)
            d = res.json()
            assert list(d.keys())[0] == "deliberation"

    def test_oversized_task_ids(self, stress_setup):
        client: TestClient = stress_setup["client"]

        # > 255 chars rejected with 400 and deliberation first
        oversized = "T" * 256
        res = client.get(f"/api/agents/history/{oversized}")
        assert res.status_code == 400
        d = res.json()
        assert list(d.keys())[0] == "deliberation"
        assert "error" in d

    def test_nonexistent_task_id(self, stress_setup):
        client: TestClient = stress_setup["client"]

        res = client.get("/api/agents/history/task_nonexistent_random_uuid")
        assert res.status_code == 404
        d = res.json()
        assert list(d.keys())[0] == "deliberation"
        assert "not found" in d["detail"].lower()


# =============================================================================
# Suite 3: GET /api/requirements Backlog & Spec States
# =============================================================================
class TestRequirementsEndpointStress:
    """Stress tests uninitialized cartridges, partially completed specs, rejected tracks,
    corrupted JSON payloads, and parameter traversal.
    """

    def test_uninitialized_cartridge(self, stress_setup):
        client: TestClient = stress_setup["client"]
        active_dir: Path = stress_setup["active_dir"]

        # Remove active directory to simulate completely uninitialized environment
        import shutil
        shutil.rmtree(active_dir, ignore_errors=True)

        res = client.get("/api/requirements")
        assert res.status_code == 200
        d = res.json()
        assert list(d.keys())[0] == "deliberation"
        assert d["has_active_spec"] is False
        assert d["total_tracks"] == 0
        assert d["requirements"] == []

    def test_partially_completed_and_rejected_tracks(self, stress_setup):
        client: TestClient = stress_setup["client"]
        active_dir: Path = stress_setup["active_dir"]

        spec = {
            "deliberation": "Decomposed space game requirements spec.",
            "spec_id": "req_spec_multi_state",
            "game_title": "Orbit Pioneer",
            "genre_paradigm": "Physics Sim",
            "target_episode": 2,
            "requirements": [
                {"id": "R1", "title": "Flight Physics", "status": "completed"},
                {"id": "R2", "title": "Orbital Docking", "status": "in_progress"},
                {"id": "R3", "title": "Solar Storm Hazards", "status": "pending"},
                {"id": "R4", "title": "Hyperdrive Navigation", "status": "rejected"},
            ],
            "asset_manifest": [],
        }
        (active_dir / "requirements_spec.json").write_text(json.dumps(spec), encoding="utf-8")

        res = client.get("/api/requirements")
        assert res.status_code == 200
        d = res.json()
        assert list(d.keys())[0] == "deliberation"
        assert d["has_active_spec"] is True
        assert d["total_tracks"] == 4
        assert d["completed_tracks"] == 1
        assert d["in_progress_tracks"] == 1
        assert d["pending_tracks"] == 1
        assert d["rejected_tracks"] == 1
        assert d["progress_percentage"] == 25.0

    def test_corrupted_json_syntax(self, stress_setup):
        client: TestClient = stress_setup["client"]
        active_dir: Path = stress_setup["active_dir"]

        # Malformed JSON syntax
        (active_dir / "requirements_spec.json").write_text("{broken json syntax: [", encoding="utf-8")

        res = client.get("/api/requirements")
        assert res.status_code == 200
        d = res.json()
        assert list(d.keys())[0] == "deliberation"
        assert d["has_active_spec"] is False

    def test_rule4_violation_on_cartridge_parameter_escape(self, stress_setup):
        """Verifies that GET /api/requirements?cartridge=.. returns 400 Bad Request
        with 'deliberation' strictly at key index 0, adhering to Rule 4.
        """
        client: TestClient = stress_setup["client"]

        res = client.get("/api/requirements?cartridge=..")
        assert res.status_code == 400
        d = res.json()
        keys = list(d.keys())
        has_deliberation_first = (len(keys) > 0 and keys[0] == "deliberation")
        assert has_deliberation_first, "Rule 4 mandates deliberation-first on cartridge override error"
        assert list(d.keys())[0] == "deliberation"
        assert "error" in d


# =============================================================================
# Suite 4: Cryptographic SHA-256 Integrity Stress on ArtifactMetadataEnvelope
# =============================================================================
class TestArtifactMetadataEnvelopeIntegrityStress:
    """Stress tests hashing, sealing, tampering, and missing file edge cases."""

    def test_sha256_exactness_across_file_types(self, tmp_path: Path):
        # 1. Empty file
        empty_f = tmp_path / "empty.txt"
        empty_f.write_bytes(b"")
        assert ArtifactMetadataEnvelope.compute_file_sha256(empty_f) == hashlib.sha256(b"").hexdigest()

        # 2. Binary random bytes (100KB)
        bin_f = tmp_path / "random.bin"
        rand_bytes = b"SPRITE_DATA_" * 8000
        bin_f.write_bytes(rand_bytes)
        assert ArtifactMetadataEnvelope.compute_file_sha256(bin_f) == hashlib.sha256(rand_bytes).hexdigest()

        # 3. Python code module
        code_f = tmp_path / "mechanics.py"
        code_text = "class SceneManager:\n    def update(self): pass\n"
        code_f.write_text(code_text, encoding="utf-8")
        assert ArtifactMetadataEnvelope.compute_file_sha256(code_f) == hashlib.sha256(code_text.encode("utf-8")).hexdigest()

    def test_dual_seal_class_and_instance(self, tmp_path: Path):
        f = tmp_path / "test_module.py"
        f.write_text("x = 42\n", encoding="utf-8")

        # Class seal
        env_cls = ArtifactMetadataEnvelope.seal(
            deliberation="Class sealing deliberation.",
            artifact_name="test_module.py",
            file_path=f,
            source_agent="LeadGameplaySWEAgent",
        )
        assert list(json.loads(env_cls.model_dump_json()).keys())[0] == "deliberation"
        assert env_cls.output_sha256 == hashlib.sha256(b"x = 42\n").hexdigest()
        assert env_cls.verify_integrity(f) is True

        # Instance seal
        env_inst = ArtifactMetadataEnvelope(
            deliberation="Instance sealing deliberation.",
            artifact_name="test_module.py",
            file_path=str(f),
        )
        env_inst.seal(f)
        assert env_inst.output_sha256 == hashlib.sha256(b"x = 42\n").hexdigest()
        assert env_inst.verify_integrity(f) is True

    def test_tampering_bit_flip_detection(self, tmp_path: Path):
        f = tmp_path / "secure_code.py"
        f.write_text("secure_flag = True\n", encoding="utf-8")

        env = ArtifactMetadataEnvelope.seal(
            deliberation="Sealing authentic code.",
            file_path=f,
        )
        assert env.verify_integrity(f) is True

        # Bit flip / modification
        f.write_text("secure_flag = False\n", encoding="utf-8")
        assert env.verify_integrity(f) is False

    def test_missing_file_false_positive_behavior(self, tmp_path: Path):
        """Verifies that verify_integrity() returns False when the referenced target
        file is completely missing from disk.
        """
        missing_path = tmp_path / "does_not_exist.py"
        env = ArtifactMetadataEnvelope(
            deliberation="Envelope referencing missing file.",
            file_path=str(missing_path),
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            output_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        # Note: missing_path does not exist
        assert not missing_path.exists()
        result = env.verify_integrity(missing_path)
        assert result is False, "Missing file must fail verification"
        assert env.verify_integrity() is False
