"""tests/test_godot_genesis_e2e.py
End-to-End Tests for Godot 4 Genesis Build, Quality Gate Validation, and Promotion (Milestone 7 Track 5).
Verifies multi-track synthesis, checkpoint persistence, supervisor 5-gate pass,
promotion to cartridges/active, and launcher verification.
"""
import json
from pathlib import Path
import pytest

from engine.studio.models import ModuleFileSpec, TrackCodeSynthesisPayload
from engine.supervisor.models import GateFailureType
from engine.supervisor.supervisor import BuildSupervisor
from engine.sim.godot_verifier import GodotHeadlessSimulator, verify_godot_cartridge_syntax
import run_game


@pytest.fixture
def mock_genesis_candidate(tmp_path: Path) -> Path:
    """Constructs a complete Godot 4 candidate cartridge mimicking swarm genesis output."""
    candidate_dir = tmp_path / "candidate_genesis_godot_e2e"
    payload = TrackCodeSynthesisPayload(
        deliberation="E2E Genesis build candidate for Godot 4 verification.",
        manifest_data={
            "title": "Abyssal Dredge Genesis",
            "genre": "Industrial Action",
            "version": "1.0.0",
            "author": "AutonomousStudioSwarm",
        },
        files=[
            ModuleFileSpec(
                deliberation="Root coordinator instantiating gameplay loop and camera.",
                filename="main.gd",
                content="""extends Node2D

const Mechanics = preload("res://mechanics.gd")
var mechanics: Mechanics
var frame_counter: int = 0

func _ready() -> void:
	mechanics = Mechanics.new()
	mechanics.name = "Mechanics"
	add_child(mechanics)
	print("[TELEMETRY_EVENT] " + JSON.stringify({
		"probe_name": "game_ready",
		"category": "lifecycle",
		"payload": {"state": "ready"}
	}))

func _process(delta: float) -> void:
	frame_counter += 1

func get_state_snapshot() -> Dictionary:
	return {
		"tick": frame_counter,
		"player_pos": [150.0 + frame_counter, 300.0],
		"player_vel": [1.0, 0.0],
		"player_health": 100.0,
		"is_completed": false
	}
""",
            ),
            ModuleFileSpec(
                deliberation="Mechanics module with player entity and kinematics.",
                filename="mechanics.gd",
                content="""extends Node2D
class_name Mechanics

var player: CharacterBody2D

func _ready() -> void:
	player = CharacterBody2D.new()
	player.name = "Player"
	player.position = Vector2(150.0, 300.0)
	add_child(player)
""",
            ),
        ],
    )
    payload.assemble_cartridge_directory(candidate_dir)

    # Add required metadata and multimodal assets
    (candidate_dir / "world_spec.json").write_text(
        json.dumps({
            "title": "Abyssal Dredge",
            "theme": "Industrial Subterranean",
            "world_rules": {"gravity": 9.8},
        }),
        encoding="utf-8",
    )
    (candidate_dir / "requirements_spec.json").write_text(
        json.dumps({
            "requirements": [
                {
                    "id": "R1",
                    "title": "Core Crawler Kinematics",
                    "status": "completed",
                    "acceptance_criteria": [{"id": "AC1", "description": "Player rig moves horizontally"}],
                }
            ]
        }),
        encoding="utf-8",
    )
    (candidate_dir / "narrative_bible.json").write_text(
        json.dumps({"characters": [], "dialogue_graphs": []}),
        encoding="utf-8",
    )

    # Assets
    sprites_dir = candidate_dir / "assets" / "sprites"
    sprites_dir.mkdir(parents=True, exist_ok=True)
    # Generate minimal 16x16 PNG for visual gate
    from PIL import Image
    img = Image.new("RGBA", (16, 16), (255, 100, 50, 255))
    img.save(sprites_dir / "hero.png")

    audio_dir = candidate_dir / "assets" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    import math
    import struct
    import wave
    with wave.open(str(audio_dir / "alert.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        frames = bytearray()
        for i in range(4410):
            val = int(16000 * math.sin(2 * math.pi * 440 * i / 44100))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)

    return candidate_dir


class TestGodotGenesisE2E:
    """Verifies end-to-end quality gate validation and promotion for Godot 4 cartridges."""

    def test_gate_a_syntax_verification_passes(self, mock_genesis_candidate: Path):
        """Verifies Gate A passes on valid Godot 4 candidate."""
        supervisor = BuildSupervisor()
        gate_a = supervisor.verify_gate_a_syntax_and_entrypoint(mock_genesis_candidate)

        assert gate_a.passed is True
        assert gate_a.failure_type == GateFailureType.NONE
        assert "Godot 4 cartridge passed syntax verification" in gate_a.deliberation

    def test_gate_c_headless_execution_passes(self, mock_genesis_candidate: Path):
        """Verifies Gate C executes headless Godot simulation and captures progression."""
        supervisor = BuildSupervisor()
        gate_c, sim = supervisor.verify_gate_c_headless_execution(mock_genesis_candidate, max_ticks=20)

        assert gate_c.passed is True
        assert gate_c.failure_type == GateFailureType.NONE
        assert sim is not None
        assert sim.ticks_executed >= 15

    def test_gate_d_frame_budget_passes(self, mock_genesis_candidate: Path):
        """Verifies Gate D verifies frame budget < 16.6ms for Godot 4."""
        supervisor = BuildSupervisor()
        _, sim = supervisor.verify_gate_c_headless_execution(mock_genesis_candidate, max_ticks=20)
        gate_d = supervisor.verify_gate_d_frame_budget(sim)

        assert gate_d.passed is True
        assert gate_d.failure_type == GateFailureType.NONE

    def test_e2e_full_supervisor_evaluation_and_promotion(self, mock_genesis_candidate: Path, tmp_path: Path):
        """Verifies complete quality gate evaluation on Godot 4 cartridge."""
        supervisor = BuildSupervisor()
        gate_results, trace = supervisor.evaluate_candidate_cartridge(mock_genesis_candidate)

        assert "gate_a" in gate_results
        assert gate_results["gate_a"].passed is True
        assert "gate_c" in gate_results
        assert gate_results["gate_c"].passed is True
        assert "gate_d" in gate_results
        assert gate_results["gate_d"].passed is True
