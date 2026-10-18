"""tests/test_godot_launcher_and_ui.py
Unit tests for Godot 4 Launcher and Studio UI Integration (Milestone 7 Track 4).
Verifies cartridge resolution with project.godot, checkpoint loading,
UI artifact inventory indexing of GDScript files, and MIME type handling.
"""
import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from engine.studio.models import ModuleFileSpec, TrackCodeSynthesisPayload
from engine.ui.server import SwarmEventBus, create_app
import run_game


@pytest.fixture
def godot_cartridge_and_checkpoints(tmp_path: Path):
    """Provisions cartridges/ directory with active cartridge and verified checkpoint."""
    cartridges_root = tmp_path / "cartridges"
    cartridges_root.mkdir(parents=True, exist_ok=True)
    active_dir = cartridges_root / "active"
    active_dir.mkdir(parents=True, exist_ok=True)

    payload = TrackCodeSynthesisPayload(
        deliberation="Active Godot 4 cartridge for UI and launcher testing.",
        manifest_data={"title": "Godot Launcher Cartridge", "genre": "SciFi", "version": "1.0.0"},
        files=[
            ModuleFileSpec(
                deliberation="Entrypoint script.",
                filename="main.gd",
                content="extends Node2D\nfunc _ready():\n\tprint('Active Godot')\n",
            ),
            ModuleFileSpec(
                deliberation="Mechanics script.",
                filename="mechanics.gd",
                content="extends Node2D\nclass_name Mechanics\n",
            ),
        ],
    )
    payload.assemble_cartridge_directory(active_dir)

    # Create a verified checkpoint
    cp_dir = cartridges_root / "checkpoints" / "candidate_godot_test_R1"
    cp_dir.mkdir(parents=True, exist_ok=True)
    payload.assemble_cartridge_directory(cp_dir)
    cp_meta = {
        "checkpoint_id": "candidate_godot_test_R1",
        "candidate_id": "candidate_godot_test",
        "requirement_id": "R1",
        "track_title": "Core Locomotion",
        "created_at_utc": "2026-10-17T12:00:00Z",
        "verified_tracks": ["R1"],
        "cartridge_path": str(cp_dir),
    }
    (cp_dir / "checkpoint_metadata.json").write_text(json.dumps(cp_meta, indent=2), encoding="utf-8")

    return {
        "cartridges_root": cartridges_root,
        "active_dir": active_dir,
        "checkpoint_dir": cp_dir,
        "tmp_path": tmp_path,
    }


class TestGodotLauncherAndUI:
    """Verifies Launcher Godot detection and UI artifact indexing."""

    def test_launcher_resolves_godot_active_cartridge(self, godot_cartridge_and_checkpoints):
        """Verifies run_game.resolve_and_load_cartridge resolves directory with project.godot."""
        active_dir = godot_cartridge_and_checkpoints["active_dir"]

        booted_path = run_game.resolve_and_load_cartridge(
            engine=None,
            requested_cartridge=str(active_dir),
        )

        assert booted_path == active_dir.resolve()
        assert (booted_path / "project.godot").is_file()

    def test_launcher_resolves_godot_checkpoint(self, godot_cartridge_and_checkpoints):
        """Verifies run_game.resolve_and_load_cartridge resolves a specific Godot checkpoint."""
        cp_dir = godot_cartridge_and_checkpoints["checkpoint_dir"]

        booted_path = run_game.resolve_and_load_cartridge(
            engine=None,
            requested_checkpoint=str(cp_dir),
        )

        assert booted_path == cp_dir.resolve()
        assert (booted_path / "project.godot").is_file()

    def test_ui_artifacts_inventory_indexes_godot_files(self, godot_cartridge_and_checkpoints):
        """Verifies GET /api/artifacts includes .gd, .tscn, and project.godot in code_modules."""
        active_dir = godot_cartridge_and_checkpoints["active_dir"]
        c_root = godot_cartridge_and_checkpoints["cartridges_root"]
        tmp_path = godot_cartridge_and_checkpoints["tmp_path"]

        bus = SwarmEventBus(log_path=tmp_path / "events.jsonl")
        app = create_app(
            db_path=tmp_path / "telemetry.db",
            cartridges_root=c_root,
            active_cartridge_dir=active_dir,
            event_bus=bus,
        )
        client = TestClient(app)

        res = client.get("/api/artifacts")
        assert res.status_code == 200
        data = res.json()

        assert data["exists"] is True
        mods = data.get("code_modules", [])
        assert "main.gd" in mods
        assert "mechanics.gd" in mods
        assert "main.tscn" in mods
        assert "project.godot" in mods

    def test_ui_artifacts_content_serves_gdscript_as_text(self, godot_cartridge_and_checkpoints):
        """Verifies GET /api/artifacts/content reads .gd and .godot files as text."""
        active_dir = godot_cartridge_and_checkpoints["active_dir"]
        c_root = godot_cartridge_and_checkpoints["cartridges_root"]
        tmp_path = godot_cartridge_and_checkpoints["tmp_path"]

        bus = SwarmEventBus(log_path=tmp_path / "events.jsonl")
        app = create_app(
            db_path=tmp_path / "telemetry.db",
            cartridges_root=c_root,
            active_cartridge_dir=active_dir,
            event_bus=bus,
        )
        client = TestClient(app)

        res = client.get("/api/artifacts/content?filename=main.gd")
        assert res.status_code == 200
        data = res.json()
        assert "content" in data
        assert "extends Node2D" in data["content"]
        assert data["mime_type"] == "text/plain"
