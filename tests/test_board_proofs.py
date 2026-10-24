"""Proof renderers (level map, filmstrip) and the level-map worker."""
from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from engine.studio.models import (
    LevelEntitySpec, LevelLayoutSpec, LevelRegionSpec, LevelTriggerSpec, StudioAgentError, WorkItem,
)
from engine.studio.proofs import capture_filmstrip, render_level_map
from engine.studio.workers import WorkerContext, level_map_handler


def make_layout(**over) -> LevelLayoutSpec:
    ents = [
        LevelEntitySpec(deliberation="d", entity_id="spawn", entity_kind="spawn", position=(100.0, 100.0), size=(0.0, 0.0), initial_state="idle"),
        LevelEntitySpec(deliberation="d", entity_id="gate", entity_kind="goal", position=(900.0, 300.0), size=(120.0, 160.0), initial_state="locked"),
    ]
    base = dict(
        deliberation="d", viewport_size=(400.0, 200.0), intended_world_scale_in_viewports=6.0,
        world_bounds=(0.0, 0.0, 1200.0, 400.0), camera_framing="follow",
        world_regions=[LevelRegionSpec(deliberation="d", region_id="start_area", bounds=(0, 0, 600, 400), progression_order=0),
                       LevelRegionSpec(deliberation="d", region_id="far_area", bounds=(600, 0, 600, 400), progression_order=1)],
        entities=ents,
        triggers=[LevelTriggerSpec(deliberation="d", trigger_id="t", source_entity_id="spawn", condition="c",
                                   target_entity_id="gate", resulting_state="open")],
    )
    base.update(over)
    return LevelLayoutSpec(**base)


def test_level_map_renders_regions_entities_and_triggers(tmp_path):
    out = render_level_map(make_layout(), tmp_path / "m" / "map.png", width=1000)
    with Image.open(out) as im:
        arr = np.asarray(im.convert("RGB")).astype(int)
    assert arr.shape[1] == 1000 and arr.shape[0] > 300
    bg = np.array([14, 16, 24])
    assert (np.abs(arr - bg).sum(axis=2) > 30).mean() > 0.05            # drawn content, not a blank canvas
    red = ((arr[:, :, 0] > 200) & (arr[:, :, 1] < 150) & (arr[:, :, 2] < 150)).sum()
    assert red > 50                                                     # the trigger arrow is visible
    yellow = ((arr[:, :, 0] > 240) & (arr[:, :, 1] > 200) & (arr[:, :, 2] < 120)).sum()
    assert yellow > 20                                                  # the point entity (spawn) marker


def test_level_map_rejects_degenerate_world(tmp_path):
    with pytest.raises(StudioAgentError):
        render_level_map(make_layout(world_bounds=(0.0, 0.0, 0.0, 10.0)), tmp_path / "x.png")


def test_level_map_worker_passes_council_direction_and_feedback_and_returns_map(tmp_path):
    seen = {}

    class FakeSwe:
        def formulate_level_layout(self, c_dir, force_refresh=False, guidance=""):
            seen.update(force=force_refresh, guidance=guidance)
            return make_layout()

    from engine.studio.models import RevisionRecord
    ctx = WorkerContext(cartridge_dir=tmp_path, swe_agent=FakeSwe())
    item = WorkItem(deliberation="d", item_id="lvl2", kind="level_map", title="Level 2", spec="a vertical shaft",
                    proof_spec="top-down map", state="in_progress",
                    revisions=[RevisionRecord(revision_index=0, feedback="too flat", verdict="request_changes")])
    result = level_map_handler(ctx)(item)
    assert seen["force"] is True and "a vertical shaft" in seen["guidance"] and "too flat" in seen["guidance"]
    assert Path(result.proof_images[0]).is_file() and "2 regions" in result.notes
    with pytest.raises(StudioAgentError):
        level_map_handler(WorkerContext(cartridge_dir=tmp_path))(item)


# ------------------------------------------------------------------ real Godot filmstrip with scripted input
_HAS_DISPLAY = (
    platform.system() == "Darwin" or bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))
    or (platform.system() == "Linux" and shutil.which("xvfb-run") is not None)
)

PROJECT = '''config_version=5
[application]
config/name="strip"
run/main_scene="res://main.tscn"
config/features=PackedStringArray("4.3", "GL Compatibility")
[display]
window/size/viewport_width=1280
window/size/viewport_height=720
[rendering]
renderer/rendering_method="gl_compatibility"
'''
SCENE = '''[gd_scene load_steps=2 format=3]
[ext_resource type="Script" path="res://main.gd" id="1"]
[node name="Main" type="Node2D"]
script = ExtResource("1")
'''
SCRIPT = '''extends Node2D
var mover: ColorRect
func _ready() -> void:
	var bg := ColorRect.new()
	bg.size = Vector2(1280, 720)
	bg.color = Color(0.1, 0.1, 0.15)
	add_child(bg)
	mover = ColorRect.new()
	mover.size = Vector2(60, 60)
	mover.position = Vector2(100, 300)
	mover.color = Color(1, 0, 0)
	add_child(mover)
func _process(_d: float) -> void:
	if Input.is_action_pressed("go_right"):
		mover.position.x += 12.0
'''


def _red_centroid_x(strip: Image.Image, col: int, cell_w: int = 426) -> float:
    arr = np.asarray(strip.convert("RGB").crop((col * cell_w, 18, (col + 1) * cell_w, strip.height))).astype(int)
    mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 60) & (arr[:, :, 2] < 60)
    xs = np.nonzero(mask)[1]
    assert xs.size > 20, "red mover not visible in this frame"
    return float(xs.mean())


@pytest.mark.skipif(not _HAS_DISPLAY, reason="viewport capture needs a display or xvfb")
def test_filmstrip_shows_scripted_input_moving_the_game_and_idle_script_does_not(tmp_path):
    probe = Path(__file__).parent / "_filmstrip_probe"
    shutil.rmtree(probe, ignore_errors=True)
    probe.mkdir(parents=True)
    try:
        (probe / "project.godot").write_text(PROJECT)
        (probe / "main.tscn").write_text(SCENE)
        (probe / "main.gd").write_text(SCRIPT)
        moving = capture_filmstrip(
            probe, ticks=[5, 25, 45], out_path=tmp_path / "moving.png",
            input_script=[{"from": 1, "to": 45, "actions": ["go_right"]}],
        )
        idle = capture_filmstrip(probe, ticks=[5, 25, 45], out_path=tmp_path / "idle.png", input_script=[])
        mv, idl = Image.open(moving), Image.open(idle)
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    xs = [_red_centroid_x(mv, i) for i in range(3)]
    assert xs[0] + 20 < xs[1] < xs[2] - 20                                # the held action visibly moves the actor
    ix = [_red_centroid_x(idl, i) for i in range(3)]
    assert max(ix) - min(ix) < 5                                          # an empty script means no input at all
    assert mv.width == 3 * 426


def test_filmstrip_requires_ticks_and_never_fabricates_missing_frames(tmp_path, monkeypatch):
    with pytest.raises(StudioAgentError):
        capture_filmstrip(tmp_path, ticks=[], out_path=tmp_path / "x.png")

    class NoCapture:
        def __init__(self, *a, **k): ...
        def run_until_complete(self): return None

    import engine.sim.godot_verifier as gv
    monkeypatch.setattr(gv, "GodotHeadlessSimulator", NoCapture)
    with pytest.raises(StudioAgentError) as ei:
        capture_filmstrip(tmp_path, ticks=[3], out_path=tmp_path / "x.png")
    assert "refusing to fabricate proof" in str(ei.value)


@pytest.mark.skipif(not _HAS_DISPLAY, reason="viewport capture needs a display or xvfb")
def test_filmstrip_injects_key_events_and_setup_calls_and_catches_self_playing_demos(tmp_path):
    probe = Path(__file__).parent / "_filmstrip_qa_probe"
    shutil.rmtree(probe, ignore_errors=True)
    probe.mkdir(parents=True)
    qa_script = '''extends Node2D
var mover: ColorRect
var hazard_spawned: bool = false
var bark_fired: bool = false
func _ready() -> void:
	var bg = ColorRect.new()
	bg.size = Vector2(1280, 720)
	bg.color = Color(0.08, 0.08, 0.12)
	add_child(bg)
	mover = ColorRect.new()
	mover.size = Vector2(80, 80)
	mover.position = Vector2(100, 300)
	mover.color = Color(1, 0, 0)
	add_child(mover)
func spawn_hazard(offset_x: float) -> void:
	hazard_spawned = true
	mover.position.x = offset_x
func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and event.keycode == KEY_SPACE and hazard_spawned:
		bark_fired = true
		mover.position.x = 900.0
func get_state_snapshot() -> Dictionary:
	return {"player_pos": [mover.position.x, mover.position.y], "bark_fired": bark_fired, "hazard_spawned": hazard_spawned}
'''
    try:
        (probe / "project.godot").write_text(PROJECT)
        (probe / "main.tscn").write_text(SCENE)
        (probe / "main.gd").write_text(qa_script)
        strip_path = capture_filmstrip(
            probe,
            ticks=[2, 12, 25],
            out_path=tmp_path / "qa_staged.png",
            input_script=[{"from": 15, "to": 20, "actions": ["KEY_SPACE"]}],
            setup_script=[{"at_tick": 8, "target_node_or_property": "", "method_name": "spawn_hazard", "args_json": "[450.0]"}],
            verify_counterfactual=True,
        )
        im = Image.open(strip_path)
        xs = [_red_centroid_x(im, i) for i in range(3)]
        assert xs[0] < 80 < xs[1] < 220 < xs[2]
        assert im.height == 240 + 18 + 18  # includes external QA telemetry header + footer

        # Now replace main.gd with a self-playing sim_tick demo loop that ignores input
        fake_demo = '''extends Node2D
var sim_tick: int = 0
var stage: String = "idle"
func _ready() -> void:
	var bg = ColorRect.new()
	bg.size = Vector2(1280, 720)
	bg.color = Color(0.1, 0.2, 0.3)
	add_child(bg)
func _process(_d: float) -> void:
	sim_tick += 1
	if sim_tick >= 10:
		stage = "auto_completed"
func get_state_snapshot() -> Dictionary:
	return {"player_pos": [0.0, 0.0], "stage": stage}
'''
        (probe / "main.gd").write_text(fake_demo)
        with pytest.raises(StudioAgentError) as ei:
            capture_filmstrip(
                probe,
                ticks=[2, 15],
                out_path=tmp_path / "fake_demo.png",
                input_script=[{"from": 5, "to": 12, "actions": ["KEY_SPACE"]}],
                verify_counterfactual=True,
            )
        assert "Counterfactual QA check failed" in str(ei.value)
    finally:
        shutil.rmtree(probe, ignore_errors=True)

