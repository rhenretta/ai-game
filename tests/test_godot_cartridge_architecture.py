"""tests/test_godot_cartridge_architecture.py
Unit tests for Godot 4 Code-First Cartridge Architecture (Milestone 7 Track 2).
Verifies Rule 4 epistemic schema layout, code-first GDScript module synthesis,
project.godot / main.tscn generation, and deprecated syntax detection.
"""
from pathlib import Path
import pytest

from engine.studio.models import (
    CodeRepairSpec,
    ModuleFileSpec,
    TrackCodeSynthesisPayload,
)
from engine.sim.godot_verifier import (
    parse_godot_stderr,
    verify_godot_cartridge_syntax,
)


class TestGodotCartridgeArchitecture:
    """Verifies Godot 4 code-first cartridge data models and file structure."""

    def test_epistemic_schema_field_ordering_rule_4(self):
        """Rule 4: Mandatory Rationale-First Epistemic Architecture.
        deliberation: str MUST be strictly at field index 0.
        """
        for model_cls in (ModuleFileSpec, TrackCodeSynthesisPayload, CodeRepairSpec):
            fields = list(model_cls.model_fields.keys())
            assert fields[0] == "deliberation", f"{model_cls.__name__} field 0 is '{fields[0]}', expected 'deliberation'"

    def test_assemble_cartridge_directory(self, tmp_path: Path):
        """Verifies TrackCodeSynthesisPayload.assemble_cartridge_directory generates valid project.godot and main.tscn."""
        target_dir = tmp_path / "test_cartridge"

        payload = TrackCodeSynthesisPayload(
            deliberation="Architected test Godot 4 modular cartridge.",
            manifest_data={"title": "Atomic Expedition", "genre": "Industrial Action", "version": "1.0.0"},
            files=[
                ModuleFileSpec(
                    deliberation="Root entrypoint script coordinating main scene flow.",
                    filename="main.gd",
                    content="""extends Node2D

const Mechanics = preload("res://mechanics.gd")
var mechanics: Mechanics

func _ready() -> void:
	mechanics = Mechanics.new()
	add_child(mechanics)
	print("GODOT_READY_SUCCESS")
""",
                ),
                ModuleFileSpec(
                    deliberation="Core mechanics and physics coordinator.",
                    filename="mechanics.gd",
                    content="""extends Node2D
class_name CoreMechanics

var player_pos: Vector2 = Vector2.ZERO

func _ready() -> void:
	player_pos = Vector2(100.0, 200.0)

func get_state_snapshot() -> Dictionary:
	return {
		"player_pos": [player_pos.x, player_pos.y],
		"player_vel": [0.0, 0.0],
		"player_health": 100.0,
	}
""",
                ),
            ],
        )

        assembled_files = payload.assemble_cartridge_directory(target_dir)

        # Verify mandatory files exist
        assert (target_dir / "project.godot").is_file()
        assert (target_dir / "main.tscn").is_file()
        assert (target_dir / "main.gd").is_file()
        assert (target_dir / "mechanics.gd").is_file()
        assert (target_dir / "manifest.json").is_file()

        # Check project.godot content
        godot_cfg = (target_dir / "project.godot").read_text(encoding="utf-8")
        assert "config_version=5" in godot_cfg
        assert 'run/main_scene="res://main.tscn"' in godot_cfg

        # Check main.tscn content
        main_tscn = (target_dir / "main.tscn").read_text(encoding="utf-8")
        assert 'format=3' in main_tscn
        assert 'path="res://main.gd"' in main_tscn

        # Verify syntax with real Godot compiler
        is_valid, diagnostics = verify_godot_cartridge_syntax(target_dir)
        assert is_valid, f"Cartridge failed syntax check: {[d.error_message for d in diagnostics]}"

    def test_deprecated_godot_3_syntax_detected(self, tmp_path: Path):
        """Verifies compiler diagnostics catch deprecated Godot 3 syntax."""
        target_dir = tmp_path / "broken_cartridge"

        payload = TrackCodeSynthesisPayload(
            deliberation="Testing deprecated syntax detection.",
            manifest_data={"title": "Broken Cartridge", "genre": "Test", "version": "1.0.0"},
            files=[
                ModuleFileSpec(
                    deliberation="Root script using deprecated yield().",
                    filename="main.gd",
                    content="""extends Node2D

func _ready() -> void:
	yield(get_tree(), "idle_frame")
""",
                ),
            ],
        )

        payload.assemble_cartridge_directory(target_dir)

        is_valid, diagnostics = verify_godot_cartridge_syntax(target_dir)
        assert not is_valid
        assert len(diagnostics) > 0
        assert any("yield" in d.error_message.lower() or "await" in d.error_message.lower() for d in diagnostics)
