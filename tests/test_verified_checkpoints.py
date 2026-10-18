"""tests/test_verified_checkpoints.py
Verification suite for Feature 20: Verified Checkpoints & Live Playable Progression.
Validates:
1. Incremental verified checkpoint persistence in StudioOrchestrator._persist_verified_checkpoint.
2. Rationale-First model integrity (deliberation at index 0) on VerifiedCheckpointRecord and CheckpointsResponse.
3. REST endpoints GET /api/checkpoints and POST /api/checkpoints/{checkpoint_id}/activate in engine/ui/server.py.
4. CLI launcher support in run_game.py (--checkpoint, --latest-checkpoint, and automatic progression fallback).
"""
import json
from pathlib import Path
import shutil
import tempfile
import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

import config
from engine.studio.models import GameRequirementsSpec, RequirementTrack, AcceptanceCriterion
from engine.studio.orchestrator import StudioOrchestrator
from engine.ui.models import VerifiedCheckpointRecord, CheckpointsResponse
from engine.ui.server import create_app
from run_game import resolve_and_load_cartridge, parse_args
from engine.core.engine import Engine


def test_verified_checkpoint_record_epistemic_ordering():
    """Rule 4: Asserts deliberation is declared at index 0 and validated strictly."""
    fields = list(VerifiedCheckpointRecord.model_fields.keys())
    assert fields[0] == "deliberation", f"Expected 'deliberation' at index 0, got {fields[0]}"

    # Blank deliberation must raise ValidationError
    with pytest.raises(ValidationError):
        VerifiedCheckpointRecord(
            deliberation="",
            checkpoint_id="cp_test",
            candidate_id="cand_test",
            requirement_id="R1",
            created_at_utc="2026-10-17T00:00:00Z",
            cartridge_path="/tmp/fake",
        )


def test_checkpoints_response_epistemic_ordering():
    """Rule 4: Asserts deliberation is declared at index 0 on CheckpointsResponse."""
    fields = list(CheckpointsResponse.model_fields.keys())
    assert fields[0] == "deliberation", f"Expected 'deliberation' at index 0, got {fields[0]}"


def test_persist_verified_checkpoint_creates_immutable_snapshot(tmp_path, monkeypatch):
    """Verifies StudioOrchestrator._persist_verified_checkpoint creates checkpoint directory,
    writes checkpoint_metadata.json with sha256 manifest, and mirrors to cartridges/active.
    """
    fake_project_root = tmp_path / "project"
    fake_project_root.mkdir()
    fake_cartridges = fake_project_root / "cartridges"
    fake_cartridges.mkdir()
    staging_dir = fake_cartridges / "candidate_test_01"
    staging_dir.mkdir()

    # Create dummy cartridge files in staging
    (staging_dir / "mechanics.py").write_text("class GameMechanics:\n    pass\n", encoding="utf-8")
    (staging_dir / "entities.py").write_text("# entities\n", encoding="utf-8")
    (staging_dir / "scenes.py").write_text("# scenes\n", encoding="utf-8")
    (staging_dir / "manifest.json").write_text(json.dumps({"title": "Test Game"}), encoding="utf-8")
    (staging_dir / "requirements_spec.json").write_text(json.dumps({"spec_id": "REQ-01"}), encoding="utf-8")

    monkeypatch.setattr("engine.studio.orchestrator.PROJECT_ROOT", fake_project_root)

    orch = StudioOrchestrator(api_key="mock_api_key_test")
    track = RequirementTrack(
        deliberation="Testing track R1 checkpoint",
        id="R1",
        title="Mobile Crawler Kinematics",
        description="Crawler physics",
        responsible_agents=["LeadGameplaySWEAgent"],
        acceptance_criteria=[],
        status="completed",
    )

    checkpoint_dir = orch._persist_verified_checkpoint(staging_dir, track, ["R1"])
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "mechanics.py").exists()
    assert (checkpoint_dir / "checkpoint_metadata.json").exists()

    meta = json.loads((checkpoint_dir / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    assert meta["checkpoint_id"] == "candidate_test_01_R1"
    assert meta["requirement_id"] == "R1"
    assert "mechanics.py" in meta["sha256_manifest"]

    # Verify mirrored to active
    active_dir = fake_cartridges / "active"
    assert active_dir.exists()
    assert (active_dir / "mechanics.py").exists()


def test_api_checkpoints_get_and_activate(tmp_path, monkeypatch):
    """Verifies GET /api/checkpoints and POST /api/checkpoints/{id}/activate."""
    fake_cartridges = tmp_path / "cartridges"
    fake_cartridges.mkdir()
    checkpoints_dir = fake_cartridges / "checkpoints"
    checkpoints_dir.mkdir()

    # Create a dummy checkpoint
    cp_dir = checkpoints_dir / "cand1_R1"
    cp_dir.mkdir()
    (cp_dir / "mechanics.py").write_text("# code\n", encoding="utf-8")
    (cp_dir / "checkpoint_metadata.json").write_text(
        json.dumps({
            "checkpoint_id": "cand1_R1",
            "candidate_id": "cand1",
            "requirement_id": "R1",
            "track_title": "Track 1",
            "created_at_utc": "2026-10-17T12:00:00Z",
            "verified_tracks": ["R1"],
            "sha256_manifest": {"mechanics.py": "abc123hash"},
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr("engine.ui.server.config.PROJECT_ROOT", tmp_path)

    app = create_app(cartridges_root=fake_cartridges, active_cartridge_dir=fake_cartridges / "active")
    client = TestClient(app)

    # 1. GET /api/checkpoints
    resp = client.get("/api/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert "deliberation" in data
    assert data["total_checkpoints"] == 1
    assert data["checkpoints"][0]["checkpoint_id"] == "cand1_R1"
    assert data["checkpoints"][0]["is_active"] is False

    # 2. POST /api/checkpoints/cand1_R1/activate
    post_resp = client.post("/api/checkpoints/cand1_R1/activate")
    assert post_resp.status_code == 200
    post_data = post_resp.json()
    assert post_data["status"] == "activated"

    active_dir = fake_cartridges / "active"
    assert active_dir.exists()
    assert (active_dir / "mechanics.py").exists()

    # 3. Verify GET /api/checkpoints now marks it as active
    resp2 = client.get("/api/checkpoints")
    assert resp2.status_code == 200
    assert resp2.json()["checkpoints"][0]["is_active"] is True


def test_run_game_cli_checkpoint_flags(tmp_path, monkeypatch):
    """Verifies run_game.py flags: --checkpoint, --latest-checkpoint, and progression fallback."""
    args = parse_args(["--checkpoint", "cand1_R1", "--latest-checkpoint"])
    assert args.checkpoint == "cand1_R1"
    assert args.latest_checkpoint is True

    fake_cartridges = tmp_path / "cartridges"
    fake_cartridges.mkdir()
    checkpoints_dir = fake_cartridges / "checkpoints"
    checkpoints_dir.mkdir()

    cp_dir = checkpoints_dir / "cand_alpha_R1"
    cp_dir.mkdir()
    (cp_dir / "mechanics.py").write_text("class GameMechanics:\n    pass\n", encoding="utf-8")
    (cp_dir / "manifest.json").write_text(json.dumps({"title": "Alpha"}), encoding="utf-8")

    monkeypatch.setattr("run_game.config.PROJECT_ROOT", tmp_path)

    engine = Engine()

    # 1. Boot explicit checkpoint
    loaded = resolve_and_load_cartridge(engine, requested_checkpoint="cand_alpha_R1")
    assert loaded == cp_dir

    # 2. Boot latest checkpoint
    loaded_latest = resolve_and_load_cartridge(engine, use_latest_checkpoint=True)
    assert loaded_latest == cp_dir

    # 3. Boot with neither active nor explicit, but checkpoint exists -> automatic progression fallback
    loaded_fallback = resolve_and_load_cartridge(engine)
    assert loaded_fallback == cp_dir


def test_checkpoint_custom_probe_emission_zero_fk_errors(tmp_path):
    """Regression test: verifies that running a checkpoint emitting shock_coil_triggered
    and structure_deployed writes cleanly to SQLite WAL with zero FK errors and zero discarded rows.
    """
    db_path = tmp_path / "test_telemetry.db"

    # Create cartridge emitting custom probes shock_coil_triggered and structure_deployed
    cart_dir = tmp_path / "cartridge_probes_test"
    cart_dir.mkdir()

    mech_code = '''
from engine.api.contracts import BaseCartridge
import pygame

class CustomProbeScene:
    def __init__(self, engine):
        self.engine = engine
        self.state_counter = 0

    def update(self, dt: float):
        self.state_counter += 1
        if hasattr(self.engine, "telemetry_recorder") and self.engine.telemetry_recorder:
            self.engine.telemetry_recorder.emit_custom_probe(
                probe_name="shock_coil_triggered",
                category="combat",
                data={"voltage": 120.0, "tick": self.state_counter},
            )
            self.engine.telemetry_recorder.emit_custom_probe(
                probe_name="structure_deployed",
                category="risk_reward",
                data={"structure_type": "pneumatic_junction"},
            )

    def render(self, surface: pygame.Surface):
        surface.fill((10, 10, 10))

class TestProbeCartridge(BaseCartridge):
    deliberation: str = "Test cartridge emitting custom probes"

    def get_scenes(self, engine):
        return {"main": CustomProbeScene(engine)}

    def get_initial_scene(self, engine):
        return "main"
'''
    (cart_dir / "mechanics.py").write_text(mech_code, encoding="utf-8")
    (cart_dir / "manifest.json").write_text(json.dumps({"title": "Probe Test"}), encoding="utf-8")

    from engine.sim.headless import HeadlessSimulator, SimulationInput
    sim = HeadlessSimulator(
        cartridge_path=cart_dir,
        telemetry_mode="sqlite",
        telemetry_db_path=db_path,
    )

    try:
        sim.telemetry_recorder.start_session(player_profile_tag="probe_tester")
        sim.telemetry_recorder.start_episode(level_id="lvl_probe_test", episode_index=1)

        # Step with active inputs
        inp = SimulationInput(right=True, action_1=True, space=True)
        for _ in range(15):
            sim.step(inp)

        sim.telemetry_recorder.end_episode(outcome="CLEARED")
        sim.telemetry_recorder.end_session()
        sim.telemetry_recorder.flush()
    finally:
        sim.telemetry_recorder.close(timeout=1.0)

    assert sim.telemetry_recorder.discarded_row_count == 0, (
        f"Expected 0 discarded rows, got {sim.telemetry_recorder.discarded_row_count}: "
        f"{getattr(sim.telemetry_recorder.writer_thread, 'discarded_errors', [])}"
    )

    # Verify in SQLite database that custom events and probes are properly recorded
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT probe_name, category FROM telemetry_probes;")
        probes = dict(cur.fetchall())
        assert "shock_coil_triggered" in probes
        assert "structure_deployed" in probes

        cur.execute("SELECT count(*) FROM custom_telemetry_events WHERE probe_name='shock_coil_triggered';")
        count_shock = cur.fetchone()[0]
        assert count_shock >= 10

        cur.execute("SELECT count(*) FROM custom_telemetry_events WHERE probe_name='structure_deployed';")
        count_struct = cur.fetchone()[0]
        assert count_struct >= 10
    finally:
        conn.close()

