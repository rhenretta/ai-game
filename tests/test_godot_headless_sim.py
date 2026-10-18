"""tests/test_godot_headless_sim.py
Unit tests for Godot 4 Headless Simulator, Telemetry Bridge, and Error Parsing (Milestone 7 Track 3).
Verifies headless simulation execution, frame budget accounting, state snapshots,
telemetry probe capture, and structured stderr diagnostic extraction.
"""
from pathlib import Path
import pytest

from engine.studio.models import ModuleFileSpec, TrackCodeSynthesisPayload
from engine.sim.godot_verifier import (
    GodotHeadlessSimulator,
    parse_godot_stderr,
    verify_godot_cartridge_syntax,
)


@pytest.fixture
def sample_godot_cartridge(tmp_path: Path) -> Path:
    """Provisions a hermetic, fully functional Godot 4 cartridge with telemetry probes."""
    cart_dir = tmp_path / "valid_sim_cartridge"
    payload = TrackCodeSynthesisPayload(
        deliberation="Architected hermetic Godot 4 cartridge for headless sim testing.",
        manifest_data={"title": "Test Sim Cartridge", "genre": "Action", "version": "1.0.0"},
        files=[
            ModuleFileSpec(
                deliberation="Root entrypoint script instantiating player and emitting telemetry.",
                filename="main.gd",
                content="""extends Node2D

var tick_count: int = 0
var player: CharacterBody2D

func _ready() -> void:
	player = CharacterBody2D.new()
	player.name = "Player"
	player.position = Vector2(100.0, 150.0)
	add_child(player)
	print("[TELEMETRY_EVENT] " + JSON.stringify({
		"probe_name": "player_spawn",
		"category": "lifecycle",
		"payload": {"spawn_x": 100.0, "spawn_y": 150.0}
	}))

func _process(delta: float) -> void:
	tick_count += 1
	if player:
		player.position.x += 2.0 * delta * 60.0
	if tick_count % 5 == 0:
		print("[TELEMETRY_EVENT] " + JSON.stringify({
			"probe_name": "locomotion_tick",
			"category": "kinematics",
			"payload": {"tick": tick_count, "x": player.position.x if player else 0.0}
		}))

func get_state_snapshot() -> Dictionary:
	return {
		"tick": tick_count,
		"player_pos": [player.position.x if player else 0.0, player.position.y if player else 0.0],
		"player_vel": [2.0, 0.0],
		"player_health": 100.0,
		"is_completed": false
	}
""",
            ),
        ],
    )
    payload.assemble_cartridge_directory(cart_dir)
    return cart_dir


class TestGodotHeadlessSimulation:
    """Verifies Godot 4 headless playtesting, performance monitoring, and telemetry capture."""

    def test_headless_sim_execution_and_frame_budget(self, sample_godot_cartridge: Path):
        """Verifies running headless simulator produces clean execution and passes frame budget."""
        sim = GodotHeadlessSimulator(
            cartridge_path=sample_godot_cartridge,
            telemetry_mode="in_memory",
            max_ticks=20,
        )
        result = sim.run()

        assert result.success is True, f"Simulation failed: {result.error_message}"
        assert result.frame_budget_passed is True
        assert result.avg_frame_time_ms < 16.6
        assert sim.ticks_executed >= 15
        assert sim.outcome == "COMPLETED"

        # State reflection
        state = result.state
        assert state.step_count >= 15
        assert state.player_pos[0] > 100.0, "Player position did not advance"
        assert state.player_health == 100.0

        sim.close()

    def test_headless_sim_telemetry_streaming(self, sample_godot_cartridge: Path, tmp_path: Path):
        """Verifies telemetry events printed by Godot are intercepted and persisted without errors."""
        db_path = tmp_path / "telemetry_test.db"
        sim = GodotHeadlessSimulator(
            cartridge_path=sample_godot_cartridge,
            telemetry_mode="sqlite",
            telemetry_db_path=db_path,
            max_ticks=20,
        )
        result = sim.run()
        assert result.success is True
        sim.close()

        # Query SQLite WAL to verify probes and custom events
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        cur.execute("SELECT probe_name, category FROM telemetry_probes;")
        probes = {row[0]: row[1] for row in cur.fetchall()}
        assert "player_spawn" in probes
        assert "locomotion_tick" in probes

        cur.execute("SELECT count(*) FROM custom_telemetry_events;")
        count = cur.fetchone()[0]
        assert count >= 2, f"Expected at least 2 telemetry events, found: {count}"
        conn.close()

    def test_parse_godot_stderr_compiler_diagnostics(self, tmp_path: Path):
        """Verifies parsing varied Godot 4 compiler and runtime error strings."""
        sample_stderr = """
Godot Engine v4.3.stable.official.77dcf97d8 - https://godotengine.org
SCRIPT ERROR: Parse Error: Expected expression for variable initial value after "=".
          at: GDScript::reload (res://main.gd:14)
ERROR: Failed to load script "res://main.gd" with error "Parse error".
   at: load (modules/gdscript/gdscript.cpp:2936)
SCRIPT ERROR: Parse Error: Identifier "KinematicBody2D" not declared in the current scope.
          at: GDScript::reload (res://entities.gd:22)
"""
        diagnostics = parse_godot_stderr(tmp_path, sample_stderr)
        assert len(diagnostics) >= 2

        d0 = diagnostics[0]
        assert d0.file_path == tmp_path / "main.gd"
        assert d0.line_number == 14
        assert "Expected expression" in d0.error_message

        d1 = diagnostics[1]
        assert d1.file_path == tmp_path / "entities.gd"
        assert d1.line_number == 22
        assert "KinematicBody2D" in d1.error_message
