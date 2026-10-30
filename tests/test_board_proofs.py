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


def _red_centroid_x(strip: Image.Image, idx: int, cell_w: int = 640, cell_h: int = 360, cols: int = 2) -> float:
    row_h = strip.height // 2 if strip.height > (cell_h + 40) else strip.height
    col, row = idx % cols, idx // cols
    x0, y0 = col * cell_w, row * row_h + 18
    arr = np.asarray(strip.convert("RGB").crop((x0, y0, x0 + cell_w, y0 + cell_h))).astype(int)
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
    assert mv.width == 2 * 640


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
        assert xs[0] < 120 < xs[1] < 330 < xs[2]
        assert im.height == 2 * (360 + 18 + 18)  # 2 rows of 640x360 cells with external QA telemetry header + footer

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


def test_board_test_target_guard_and_qa_production_defect_report_tool(tmp_path):
    """Verifies (1) ProductionBoard rejects tasks targeting `tests/...`, (2) `FilmstripPlan.production_defect_report`
    lets QA report broken/deadlocked production code cleanly without faking state mutations, and (3) `integration`
    tasks permit legitimate scenario `setup_calls` while rejecting empty passive runs."""
    from engine.studio.models import FilmstripPlan, InputSegment, QASetupCall, WorkItemProposal
    from engine.studio.production_board import ProductionBoard
    from engine.studio.workers import run_qa_test_engineer_specialist, validate_filmstrip_plan

    board = ProductionBoard(tmp_path / "backlog.json")
    bad_prop = WorkItemProposal(
        deliberation="Attempt to target QA harness instead of production script.",
        item_id="fix_qa_harness",
        kind="integration",
        title="Patch QA wrapper",
        spec="Modify test script",
        proof_spec="Filmstrip",
        target_modules=["tests/test_slice_playtest.gd"],
    )
    issues = board.validate_proposals([bad_prop])
    assert any("targets QA test file(s)" in msg for msg in issues)

    # 1. QA can submit a structured `production_defect_report` when production code is broken/deadlocked
    defect_plan = FilmstripPlan(
        deliberation="Production input handler ignores clicks on world resource nodes at t=0.",
        production_defect_report="main.gd._unhandled_input never dispatches left-click to world nodes; resource_count stays 0.",
    )
    assert validate_filmstrip_plan(defect_plan, ["KEY_SPACE"], item_kind="integration") == []

    # 2. `run_qa_test_engineer_specialist` surfaces `QA Diagnostic Production Defect Report:` to TaskLead
    import engine.studio.workers as workers_mod

    orig_plan_fn = workers_mod.plan_filmstrip
    try:
        workers_mod.plan_filmstrip = lambda ctx, it, **kw: defect_plan
        ctx = WorkerContext(cartridge_dir=tmp_path)
        it = WorkItem(
            deliberation="d",
            item_id="slice_loop",
            kind="integration",
            title="Slice loop",
            spec="Play opening loop",
            proof_spec="Filmstrip",
        )
        with pytest.raises(StudioAgentError) as qa_ei:
            run_qa_test_engineer_specialist(ctx, it, brief="Verify opening loop")
        assert "QA Diagnostic Production Defect Report:" in str(qa_ei.value)
        assert "resource_count stays 0" in str(qa_ei.value)
    finally:
        workers_mod.plan_filmstrip = orig_plan_fn

    # 3. `integration` tasks allow legitimate Slice #2+ staging `setup_calls` with real `input_segments`,
    #    while rejecting completely empty passive plans.
    staged_integ_plan = FilmstripPlan(
        deliberation="Stage mid-game starting state for Slice #2 and play with real inputs.",
        council_test_scene_summary="Stages tier-2 unlock and exercises player input.",
        test_scene_gd_content="extends Node2D\n",
        capture_ticks=[5, 30],
        input_segments=[InputSegment(deliberation="press", from_tick=10, to_tick=15, actions=["KEY_SPACE"])],
        setup_calls=[QASetupCall(deliberation="setup", at_tick=2, method_name="configure_slice_two_start", args_json="[]")],
    )
    assert validate_filmstrip_plan(staged_integ_plan, ["KEY_SPACE"], item_kind="integration") == []

    passive_integ_plan = FilmstripPlan(
        deliberation="Empty passive integration plan.",
        council_test_scene_summary="Passive wrapper",
        test_scene_gd_content="extends Node2D\n",
        capture_ticks=[5, 30],
        input_segments=[],
        setup_calls=[],
    )
    passive_issues = validate_filmstrip_plan(passive_integ_plan, ["KEY_SPACE"], item_kind="integration")
    assert any("input_segments" in msg for msg in passive_issues)


@pytest.mark.skipif(not _HAS_DISPLAY, reason="viewport capture needs a display or xvfb")
def test_entrypoint_parity_and_runtime_ui_audio_observability(tmp_path):
    """Verifies `capture_filmstrip`:
    1. Rejects mid-run `sim_tick` state injection inside `tests/test_*.gd` when `t=0` matches `res://main.tscn`.
    2. Captures `visible_ui_texts`, `audio_players_detail`, and `main_scene_parity` into `.telemetry.json`
       and `format_filmstrip_telemetry_progression`.
    3. Permits distinct `t=0` Slice #2+ staging while surfacing `staged_distinct_t0: True` in `main_scene_parity`."""
    from engine.studio.models import QAStateAssertion
    from engine.studio.proofs import format_filmstrip_telemetry_progression

    probe = Path(__file__).parent / "_filmstrip_parity_probe"
    shutil.rmtree(probe, ignore_errors=True)
    (probe / "tests").mkdir(parents=True)
    deadlocked_main = '''extends Node2D
var resource_count: int = 0
func _ready() -> void:
	var bg = ColorRect.new()
	bg.size = Vector2(1280, 720)
	bg.color = Color(0.1, 0.14, 0.2)
	add_child(bg)
	var hud = CanvasLayer.new()
	var lbl = Label.new()
	lbl.text = "Resources: 0/50"
	hud.add_child(lbl)
	add_child(hud)
	var aud = AudioStreamPlayer.new()
	aud.name = "AmbientWave"
	var wav = AudioStreamWAV.new()
	wav.loop_mode = AudioStreamWAV.LOOP_FORWARD
	aud.stream = wav
	add_child(aud)
func get_state_snapshot() -> Dictionary:
	return {"player_pos": [100.0, 100.0], "resource_count": resource_count}
'''
    cheating_wrapper_gd = '''extends Node2D
var main_inst: Node2D
var sim_tick: int = 0
func _ready() -> void:
	main_inst = preload("res://main.tscn").instantiate()
	add_child(main_inst)
func _physics_process(_d: float) -> void:
	sim_tick += 1
	if sim_tick == 10:
		main_inst.resource_count = 15
func get_state_snapshot() -> Dictionary:
	return main_inst.get_state_snapshot()
'''
    wrapper_tscn = '''[gd_scene load_steps=2 format=3]
[ext_resource type="Script" path="res://tests/test_wrapper.gd" id="1"]
[node name="TestScene" type="Node2D"]
script = ExtResource("1")
'''
    try:
        (probe / "project.godot").write_text(PROJECT, encoding="utf-8")
        (probe / "main.tscn").write_text(SCENE, encoding="utf-8")
        (probe / "main.gd").write_text(deadlocked_main, encoding="utf-8")
        (probe / "tests" / "test_wrapper.gd").write_text(cheating_wrapper_gd, encoding="utf-8")
        (probe / "tests" / "test_wrapper.tscn").write_text(wrapper_tscn, encoding="utf-8")

        assertions = [
            QAStateAssertion(
                deliberation="Verify player harvested resources.",
                at_tick=20,
                snapshot_key="resource_count",
                expected_value_or_change=">= 10",
            )
        ]
        with pytest.raises(StudioAgentError) as ei:
            capture_filmstrip(
                probe,
                ticks=[2, 20],
                out_path=tmp_path / "cheating_wrapper.png",
                input_script=[{"from": 5, "to": 12, "actions": ["KEY_SPACE"]}],
                state_assertions=assertions,
                target_scene="res://tests/test_wrapper.tscn",
                verify_entrypoint_parity=True,
            )
        assert "Entrypoint Parity Gate failed" in str(ei.value)

        # Now wire real input handling in main.gd and make test_wrapper.gd a pure read-only observer
        wired_main = deadlocked_main.replace(
            "func get_state_snapshot()",
            "func _input(event: InputEvent) -> void:\n\tif event is InputEventKey and event.pressed and event.keycode == KEY_SPACE:\n\t\tresource_count += 15\nfunc get_state_snapshot()",
        )
        honest_wrapper_gd = '''extends Node2D
var main_inst: Node2D
func _ready() -> void:
	main_inst = preload("res://main.tscn").instantiate()
	add_child(main_inst)
func get_state_snapshot() -> Dictionary:
	var snap = main_inst.get_state_snapshot().duplicate()
	snap["diag_overlay_active"] = true
	return snap
'''
        (probe / "main.gd").write_text(wired_main, encoding="utf-8")
        (probe / "tests" / "test_wrapper.gd").write_text(honest_wrapper_gd, encoding="utf-8")

        honest_strip = capture_filmstrip(
            probe,
            ticks=[2, 20],
            out_path=tmp_path / "honest_wrapper.png",
            input_script=[{"from": 5, "to": 12, "actions": ["KEY_SPACE"]}],
            state_assertions=assertions,
            target_scene="res://tests/test_wrapper.tscn",
            verify_entrypoint_parity=True,
        )
        assert honest_strip.is_file()
        telem_str = format_filmstrip_telemetry_progression(honest_strip)
        assert "Resources: 0/50" in telem_str
        assert "AmbientWave" in telem_str
        assert "Live Main Scene Parity:" in telem_str

        # Verify distinct t=0 Slice #2+ staging is permitted and transparently surfaced in `main_scene_parity`
        staged_slice2_wrapper = '''extends Node2D
var main_inst: Node2D
func _ready() -> void:
	main_inst = preload("res://main.tscn").instantiate()
	main_inst.resource_count = 100
	add_child(main_inst)
func get_state_snapshot() -> Dictionary:
	return main_inst.get_state_snapshot()
'''
        (probe / "tests" / "test_wrapper.gd").write_text(staged_slice2_wrapper, encoding="utf-8")
        staged_strip = capture_filmstrip(
            probe,
            ticks=[2, 20],
            out_path=tmp_path / "staged_slice2.png",
            input_script=[{"from": 5, "to": 12, "actions": ["KEY_SPACE"]}],
            target_scene="res://tests/test_wrapper.tscn",
            verify_entrypoint_parity=True,
        )
        assert staged_strip.is_file()
        staged_telem = format_filmstrip_telemetry_progression(staged_strip)
        assert "'staged_distinct_t0': True" in staged_telem
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def test_unread_module_guard_verified_disk_diff_and_council_test_script_visibility(tmp_path):
    """Verifies:
    1. `LeadGameplaySWEAgent._validate_no_large_file_clobber` blocks `files` overwrites ONLY on unread
       `summarized_only_mods` (prompting `self_question`), while allowing full-file edits on loaded modules.
    2. `run_swe_engineer_specialist` records ONLY files actually modified on disk and flags untouched `target_modules`.
    3. `CreativeWritersRoom._format_item_for_review` shows Council reviewers the actual `tests/test_*.gd` script on disk."""
    from engine.studio.models import ModuleFileSpec, RevisionRecord, TrackCodeSynthesisPayload
    from engine.studio.swe import CodeVerificationError, LeadGameplaySWEAgent
    from engine.studio.workers import run_swe_engineer_specialist
    from engine.studio.writers_room import CreativeWritersRoom

    (tmp_path / "project.godot").write_text(PROJECT, encoding="utf-8")
    (tmp_path / "main.gd").write_text("extends Node2D\n", encoding="utf-8")
    large_existing = "\n".join(["extends Node2D"] + [f"var v_{i}: int = {i}" for i in range(150)]) + "\n"
    (tmp_path / "scenes" / "mechanics").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scenes" / "mechanics" / "tool_controller.gd").write_text(large_existing, encoding="utf-8")

    swe_agent = LeadGameplaySWEAgent(client=object())
    rewrite_payload = TrackCodeSynthesisPayload(
        deliberation="Refactor module.",
        files=[
            ModuleFileSpec(
                deliberation="Overwrite tool_controller.gd.",
                filename="scenes/mechanics/tool_controller.gd",
                content="extends Node2D\n",
            )
        ],
    )
    # 1a. Rejects overwrite when `tool_controller.gd` was only summarized (unread) in SWE's prompt
    with pytest.raises(CodeVerificationError) as clobber_ei:
        swe_agent._validate_no_large_file_clobber(
            rewrite_payload,
            staging_dir=tmp_path,
            summarized_only_mods={"scenes/mechanics/tool_controller.gd"},
        )
    assert "self_question" in str(clobber_ei.value)

    # 1b. Allows full-file refactor when `tool_controller.gd` WAS loaded in full in SWE's prompt (even > 150 lines!)
    swe_agent._validate_no_large_file_clobber(
        rewrite_payload,
        staging_dir=tmp_path,
        summarized_only_mods=set(),
    )

    # 2. Ground-truth disk diff when SWE only creates `scenes/entities/harvest_node.gd` and leaves `main.gd` untouched
    class PartialWriteSwe:
        def implement_requirement_track(self, track, staging_dir, requirements_spec=None, **kw):
            out_f = Path(staging_dir) / "scenes" / "entities" / "harvest_node.gd"
            out_f.parent.mkdir(parents=True, exist_ok=True)
            out_f.write_text("extends Node2D\nfunc harvest() -> int:\n\treturn 1\n", encoding="utf-8")
            return {"scenes/entities/harvest_node.gd": out_f}

    ctx = WorkerContext(cartridge_dir=tmp_path, swe_agent=PartialWriteSwe())
    item = WorkItem(
        deliberation="d",
        item_id="mech_harvest",
        kind="mechanic_code",
        title="Harvest mechanic",
        spec="Create harvest_node.gd and wire click dispatch in main.gd",
        proof_spec="Filmstrip",
        target_modules=["scenes/entities/harvest_node.gd", "main.gd"],
    )
    rep = run_swe_engineer_specialist(
        ctx,
        item,
        brief="Implement harvest node",
        target_paths=["scenes/entities/harvest_node.gd", "main.gd"],
    )
    assert rep.touched_files == ["scenes/entities/harvest_node.gd"]
    assert "UNTOUCHED requested target_modules: ['main.gd']" in rep.summary

    # 3. Council review dossier includes the actual `tests/test_mech_harvest.gd` source code
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_mech_harvest.gd").write_text(
        "extends Node2D\nfunc _ready() -> void:\n\tvar m = preload('res://main.tscn').instantiate()\n\tadd_child(m)\n",
        encoding="utf-8",
    )
    item.revisions.append(
        RevisionRecord(
            deliberation="rev",
            revision_index=0,
            worker_notes="Claimed done",
            proof_images=[],
        )
    )
    review_txt = CreativeWritersRoom._format_item_for_review(item, candidate_dir=tmp_path)
    assert "qa_test_script_source (`tests/test_mech_harvest.gd`" in review_txt
    assert "preload('res://main.tscn').instantiate()" in review_txt


