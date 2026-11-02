"""Unit and integration tests for Recursive Atomic Decomposition, Executable Godot Unit Testing,
and 4-Tier Visual Verification (Asset Inspection Plates, Animation Contact Sheets, Scene + Collision Overlays,
and Challenger Gate Causal/Spatial Invariant Enforcement).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from PIL import Image, ImageDraw

from engine.sim.atomic_runner import GodotAtomicTestRunner
from engine.studio.models import (
    AnimationValidationVerdict,
    AssetVisualValidationVerdict,
    AtomicInvariantSpec,
    AtomicTestSynthesisPayload,
    AtomicWorkUnit,
    GameRequirementsSpec,
    RequirementTrack,
    SceneUnitVisualVerdict,
    TrackAtomicDecompositionPlan,
)
from engine.studio.visual import VisualArtSpriteDesignerAgent
from engine.supervisor.challenger import (
    ChallengerGate,
    MultiAgentSignOffCouncil,
)
from engine.supervisor.models import (
    ChallengerGateEvaluation,
    RequirementEvaluationRecord,
)


def test_rule_4_deliberation_first_on_new_atomic_and_visual_models() -> None:
    """Verify that every new Pydantic schema places 'deliberation' strictly at index 0 (Rule 4)."""
    schemas = [
        AssetVisualValidationVerdict,
        AnimationValidationVerdict,
        SceneUnitVisualVerdict,
        AtomicInvariantSpec,
        AtomicWorkUnit,
        TrackAtomicDecompositionPlan,
        AtomicTestSynthesisPayload,
    ]
    for model_cls in schemas:
        field_names = list(model_cls.model_fields.keys())
        assert field_names[0] == "deliberation", (
            f"{model_cls.__name__} violates Rule 4: first field is '{field_names[0]}', expected 'deliberation'"
        )


def test_tier1_asset_inspection_plate_cutout_and_tiled_grid(tmp_path: Path) -> None:
    """Verify Tier 1 per-asset render plate generates dual-checkerboard cutout plates and 3x3 tiled seam grids."""
    sprite_dir = tmp_path / "assets" / "sprites"
    sprite_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = tmp_path / "artifacts" / "renders"

    # 1. Cutout sprite with transparent border
    cutout_path = sprite_dir / "player_avatar.png"
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([16, 12, 48, 52], fill=(60, 180, 220, 255))
    img.save(cutout_path, format="PNG")

    plate_cutout = renders_dir / "asset_player_avatar_plate.png"
    plate_img_cutout = VisualArtSpriteDesignerAgent.render_asset_inspection_plate(
        img=cutout_path,
        asset_type="sprite",
        output_path=plate_cutout,
    )
    assert plate_cutout.is_file()
    assert plate_img_cutout.size[0] > plate_img_cutout.size[1]  # Side-by-side dual checkerboard

    agent = VisualArtSpriteDesignerAgent(client=None)
    verdict_cutout = agent.verify_rendered_asset(
        asset_req={
            "deliberation": "Cutout avatar check",
            "asset_id": "player_avatar",
            "asset_type": "sprite",
            "fabrication_mode": "static_sprite",
            "description": "Hero avatar",
        },
        rendered_asset_path=cutout_path,
        inspection_plate_path=plate_cutout,
    )
    assert verdict_cutout.accepted is True
    assert verdict_cutout.alpha_cleanliness_passed is True

    # 2. Tileable surface texture (3x3 seamless grid plate)
    tile_path = sprite_dir / "terrain_surface.png"
    tile_img = Image.new("RGBA", (64, 64), (48, 128, 72, 255))
    tile_img.save(tile_path, format="PNG")

    plate_tile = renders_dir / "asset_terrain_surface_plate.png"
    plate_img_tile = VisualArtSpriteDesignerAgent.render_asset_inspection_plate(
        img=tile_path,
        asset_type="tile_surface",
        output_path=plate_tile,
    )
    assert plate_tile.is_file()
    assert plate_img_tile.size == (192, 192)  # 3x3 of 64x64


def test_tier2_animation_contact_sheet_and_interframe_delta(tmp_path: Path) -> None:
    """Verify Tier 2 animation contact sheet computes deterministic inter-frame L1 pixel deltas
    and rejects static duplicate frames while accepting genuine pose articulation.
    """
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = tmp_path / "artifacts" / "renders"

    # Create a 3-frame horizontal sprite sheet with distinct articulated walk frames
    strip = Image.new("RGBA", (64 * 3, 64), (0, 0, 0, 0))
    for idx, x_offset in enumerate((12, 22, 32)):
        im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle([x_offset, 16, x_offset + 20, 52], fill=(240, 160, 60, 255))
        strip.paste(im, (idx * 64, 0))
    sheet_src_path = frames_dir / "walk_strip.png"
    strip.save(sheet_src_path, format="PNG")

    sheet_path = renders_dir / "anim_hero_walk_sheet.png"
    _, metrics = VisualArtSpriteDesignerAgent.render_animation_contact_sheet(
        sheet_img=sheet_src_path,
        states_meta={"walk": {"row": 0, "frame_count": 3, "fps": 8.0, "loop": True}},
        cell_size=(64, 64),
        output_path=sheet_path,
    )
    assert sheet_path.is_file()
    assert metrics["all_frames_non_empty"] is True
    assert metrics["is_truly_animated"] is True
    assert metrics["min_interframe_l1_delta"] > 0.5

    # Also test articulated rig contact sheet
    base_frame_path = frames_dir / "base_part.png"
    strip.crop((0, 0, 64, 64)).save(base_frame_path, format="PNG")
    rig_sheet_path = renders_dir / "anim_hero_rig_sheet.png"
    _, rig_metrics = VisualArtSpriteDesignerAgent.render_articulated_rig_contact_sheet(
        composite_path=base_frame_path,
        articulated_parts=[],
        cartridge_dir=tmp_path,
        cell_size=(64, 64),
        output_path=rig_sheet_path,
    )
    assert rig_sheet_path.is_file()
    assert rig_metrics["is_truly_animated"] is True
    assert rig_metrics["min_interframe_l1_delta"] > 0.1

    # Verify static duplicate frames produce 0.0 delta and fail deterministic check
    dup_strip = Image.new("RGBA", (64 * 2, 64), (0, 0, 0, 0))
    frame0 = strip.crop((0, 0, 64, 64))
    dup_strip.paste(frame0, (0, 0))
    dup_strip.paste(frame0, (64, 0))
    dup_sheet_path = renders_dir / "anim_static_dup_sheet.png"
    _, dup_metrics = VisualArtSpriteDesignerAgent.render_animation_contact_sheet(
        sheet_img=dup_strip,
        states_meta={"static_dup": {"row": 0, "frame_count": 2, "fps": 8.0, "loop": True}},
        cell_size=(64, 64),
        output_path=dup_sheet_path,
    )
    assert dup_metrics["min_interframe_l1_delta"] == 0.0
    assert dup_metrics["is_truly_animated"] is False

    agent = VisualArtSpriteDesignerAgent(client=None)
    verdict = agent.verify_rendered_animation(
        target_id="static_dup",
        contact_sheet_path=dup_sheet_path,
        anim_telemetry=dup_metrics,
    )
    assert verdict.accepted is False
    assert verdict.motion_progression_verified is False


def test_challenger_gate_rejects_passive_auto_win_and_unbounded_sky() -> None:
    """Verify ChallengerGate and MultiAgentSignOffCouncil deterministically reject builds when
    passive_auto_win_detected or unbounded_sky_or_void_detected or atomic_test_suite failures occur.
    """
    spec = GameRequirementsSpec(
        deliberation="Test specification for spatial and causal invariants.",
        game_title="Terrace Test",
        genre_paradigm="2d_spatial_simulation",
        requirements=[
            RequirementTrack(
                deliberation="Track 1",
                id="REQ-01",
                title="World & Mechanics",
                description="Terrace world with boundaries and causal mechanics.",
                target_modules=["scenes.gd", "mechanics.gd"],
                acceptance_criteria=[],
            )
        ],
    )

    telemetry_grounding: Dict[str, Any] = {
        "ticks_executed": 180,
        "is_stagnant_or_frozen": False,
        "scene_audit": {
            "audio_players_found": 2,
            "audio_streams_loaded": 2,
            "audio_mix_rates": [24000, 24000],
            "passive_auto_win_detected": True,
            "passive_win_reason": "Game engine_status transitioned to 'active' at tick 5 during zero-input window.",
            "unbounded_sky_or_void_detected": True,
            "boundary_violation_reason": "Player walked into sky backdrop at y=16.6 above walkable_min_y=260.0.",
        },
        "atomic_test_suite": {
            "total_tests": 2,
            "passed_tests": 1,
            "failed_tests": 1,
            "all_passed": False,
            "failed_details": [
                {
                    "test_script": "res://tests/test_unit_01.gd",
                    "failed_assertions": ["Player crossed north sky boundary"],
                    "error_trace": "",
                }
            ],
        },
    }
    visual_grounding: Dict[str, Any] = {
        "blank_canvas_detected": False,
        "captured_frame_paths": [
            "assets/playtest_captures/playtest_t0.png",
            "assets/playtest_captures/playtest_tend.png",
            "assets/playtest_captures/playtest_collision_overlay.png",
        ],
    }

    from engine.supervisor.challenger import SPECIALIST_COUNCIL_ROLES
    from engine.supervisor.models import MultiAgentSignOffReport, SpecialistSignOffVerdict

    council = MultiAgentSignOffCouncil(client=None)
    raw_council_report = MultiAgentSignOffReport(
        deliberation="Council specialists reviewed scene audit and unit test failures.",
        candidate_id="candidate_test",
        unanimous_approval=False,
        verdicts=[
            SpecialistSignOffVerdict(
                deliberation="Gameplay / boundary defects detected.",
                specialist_role=role,
                approved=role not in ("GameplayDesigner", "VisualArtDirector"),
                blocking_defects=(
                    ["Passive auto-win", "Unbounded sky", "Atomic Godot unit test failure"]
                    if role == "GameplayDesigner"
                    else (["walk-on-sky defect"] if role == "VisualArtDirector" else [])
                ),
            )
            for role in SPECIALIST_COUNCIL_ROLES
        ],
    )
    council_report = council._normalize_sign_off_report(
        report=raw_council_report,
        candidate_id="candidate_test",
        captured_paths=visual_grounding["captured_frame_paths"],
    )
    assert council_report.unanimous_approval is False
    by_role = {v.specialist_role: v for v in council_report.verdicts}
    assert by_role["GameplayDesigner"].approved is False
    assert by_role["VisualArtDirector"].approved is False

    gate = ChallengerGate(client=None)
    initial_eval = ChallengerGateEvaluation(
        deliberation="Initial evaluation prior to structural finalization.",
        candidate_id="candidate_test",
        overall_verdict=True,
        evaluated_requirements=[
            RequirementEvaluationRecord(
                deliberation="Track 1 check",
                requirement_id="REQ-01",
                fulfilled=True,
                evidence_summary="Checked",
                failing_criteria=[],
            )
        ],
        specialist_sign_off=council_report,
    )
    final_eval = gate._finalize_structural_evaluation(
        evaluation=initial_eval,
        spec=spec,
        ast_grounding={"hollow_stubs": []},
        telemetry_grounding=telemetry_grounding,
        visual_grounding=visual_grounding,
        candidate_id="candidate_test",
        episode_number=1,
        attempt_number=1,
    )
    assert final_eval.overall_verdict is False
    assert final_eval.rejection_trace is not None
    assert "Specialist Sign-Off Vetoes" in final_eval.rejection_trace.failure_reason
    assert "Atomic Unit Test Failure" in final_eval.rejection_trace.failure_reason


def test_godot_atomic_runner_executes_unit_tests_and_renders_collision_overlay(tmp_path: Path) -> None:
    """Verify GodotAtomicTestRunner executes headless GDScript unit tests and renders Tier 3
    clean + physics collision debug overlays on a real minimal Godot 4 project.
    """
    from engine.godot_runtime import ensure_godot_binary
    try:
        ensure_godot_binary()
    except Exception:
        pytest.skip("Godot binary not available on host")

    # Create a minimal Godot 4 project in tmp_path
    (tmp_path / "project.godot").write_text(
        '; Engine configuration file.\n'
        'config_version=5\n\n'
        '[application]\n'
        'config/name="AtomicRunnerTest"\n'
        'run/main_scene="res://main.tscn"\n\n'
        '[display]\n'
        'window/size/viewport_width=1280\n'
        'window/size/viewport_height=720\n',
        encoding="utf-8",
    )
    (tmp_path / "main.tscn").write_text(
        '[gd_scene load_steps=2 format=3]\n'
        '[ext_resource type="Script" path="res://main.gd" id="1"]\n'
        '[node name="Main" type="Node2D"]\n'
        'script = ExtResource("1")\n',
        encoding="utf-8",
    )
    (tmp_path / "main.gd").write_text(
        'extends Node2D\n'
        'var is_solved: bool = false\n'
        'func _ready() -> void:\n'
        '    var wall := StaticBody2D.new()\n'
        '    wall.name = "NorthHorizonWall"\n'
        '    var poly := CollisionPolygon2D.new()\n'
        '    poly.polygon = PackedVector2Array([Vector2(0, 240), Vector2(1280, 240), Vector2(1280, 260), Vector2(0, 260)])\n'
        '    wall.add_child(poly)\n'
        '    add_child(wall)\n'
        '    var player := CharacterBody2D.new()\n'
        '    player.name = "Player"\n'
        '    player.position = Vector2(400, 320)\n'
        '    var cshape := CollisionShape2D.new()\n'
        '    var rect := RectangleShape2D.new()\n'
        '    rect.size = Vector2(28, 28)\n'
        '    cshape.shape = rect\n'
        '    player.add_child(cshape)\n'
        '    add_child(player)\n'
        'func _draw() -> void:\n'
        '    draw_rect(Rect2(0, 0, 1280, 260), Color(0.12, 0.16, 0.28, 1.0), true)\n'
        '    draw_rect(Rect2(0, 260, 1280, 460), Color(0.22, 0.52, 0.30, 1.0), true)\n',
        encoding="utf-8",
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_script = tests_dir / "test_unit_horizon_barrier.gd"
    test_script.write_text(
        'extends SceneTree\n'
        'func _init() -> void:\n'
        '    var main_scr = load("res://main.gd")\n'
        '    var root_node = main_scr.new()\n'
        '    root.add_child(root_node)\n'
        '    await process_frame\n'
        '    var baseline_ok: bool = (root_node.is_solved == false)\n'
        '    var wall_ok: bool = (root_node.get_node_or_null("NorthHorizonWall") != null)\n'
        '    if baseline_ok:\n'
        '        print("[ATOMIC_ASSERT_PASS] INV-01|unsolved_baseline|Game starts unsolved at t=0")\n'
        '    else:\n'
        '        print("[ATOMIC_ASSERT_FAIL] INV-01|unsolved_baseline|Game was pre-solved at t=0")\n'
        '    if wall_ok:\n'
        '        print("[ATOMIC_ASSERT_PASS] INV-02|spatial_boundary|NorthHorizonWall StaticBody2D exists")\n'
        '    else:\n'
        '        print("[ATOMIC_ASSERT_FAIL] INV-02|spatial_boundary|NorthHorizonWall StaticBody2D missing")\n'
        '    quit(0 if (baseline_ok and wall_ok) else 1)\n',
        encoding="utf-8",
    )

    res = GodotAtomicTestRunner.run_atomic_test(tmp_path, "res://tests/test_unit_horizon_barrier.gd")
    assert res.passed is True
    assert len(res.assertions) == 2
    assert len(res.failed_assertions) == 0

    suite_res = GodotAtomicTestRunner.run_all_atomic_tests(tmp_path)
    assert suite_res.total_tests == 1
    assert suite_res.passed_tests == 1
    assert suite_res.all_passed is True

    # Also test Tier 3 scene + collision overlay rendering
    render_res = GodotAtomicTestRunner.render_scene_and_collision_overlay(
        cartridge_dir=tmp_path,
        unit_id="unit_horizon_barrier",
        ticks=24,
    )
    assert render_res["clean_render_path"] is not None
    assert Path(render_res["clean_render_path"]).is_file()
    assert render_res["collision_overlay_path"] is not None
    assert Path(render_res["collision_overlay_path"]).is_file()


def test_non_mirrored_toroidal_tile_and_rorschach_rejection(tmp_path: Path) -> None:
    """Verify ensure_full_bleed_tile synthesizes seamless non-mirrored tiles via toroidal offset
    cross-blending and that verify_rendered_asset rejects Rorschach/kaleidoscope mirrored tiles.
    """
    import numpy as np
    from engine.studio.models import KeyframeTransformSpec, PartKeyframeTrackSpec

    for cls in (KeyframeTransformSpec, PartKeyframeTrackSpec):
        assert list(cls.model_fields.keys())[0] == "deliberation"

    # 1. Create an asymmetric natural stone/moss patch with transparent corners
    rng = np.random.default_rng(42)
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    for y in range(8, 56):
        for x in range(8, 56):
            arr[y, x, 0] = int(40 + (x * 1.7 + y * 0.9) % 70 + rng.integers(0, 25))
            arr[y, x, 1] = int(95 + (x * 1.1 + y * 2.1) % 80 + rng.integers(0, 25))
            arr[y, x, 2] = int(50 + (x * 0.7 + y * 1.3) % 60 + rng.integers(0, 20))
            arr[y, x, 3] = 255

    raw_tile = Image.fromarray(arr, mode="RGBA")
    healed_tile = VisualArtSpriteDesignerAgent.ensure_full_bleed_tile(raw_tile, target_size=(64, 64))
    healed_arr = np.array(healed_tile)

    # Full bleed: all corners opaque
    assert int(healed_arr[:, :, 3].min()) == 255
    sym_score = VisualArtSpriteDesignerAgent.compute_tile_mirror_symmetry(healed_tile)
    assert sym_score < 0.62, f"Expected non-mirrored toroidal tile symmetry < 0.62, got {sym_score:.3f}"

    # 2. Create a synthetic Rorschach 4-way mirrored tile and verify critic rejects it
    q = healed_arr[:32, :32, :3]
    top_half = np.concatenate([q, q[:, ::-1, :]], axis=1)
    rorschach_rgb = np.concatenate([top_half, top_half[::-1, :, :]], axis=0)
    rorschach_rgba = np.dstack([rorschach_rgb, np.full((64, 64), 255, dtype=np.uint8)])
    rorschach_path = tmp_path / "rorschach_tile.png"
    Image.fromarray(rorschach_rgba, mode="RGBA").save(rorschach_path, format="PNG")
    plate_path = tmp_path / "rorschach_plate.png"
    VisualArtSpriteDesignerAgent.render_asset_inspection_plate(
        rorschach_path, asset_type="tile_surface", output_path=plate_path
    )

    agent = VisualArtSpriteDesignerAgent(client=None)
    verdict = agent.verify_rendered_asset(
        asset_req={
            "deliberation": "Check tile mirror symmetry",
            "asset_id": "rorschach_tile",
            "asset_type": "tile_surface",
            "fabrication_mode": "tile_surface",
            "description": "Stone wall tile",
        },
        rendered_asset_path=rorschach_path,
        inspection_plate_path=plate_path,
    )
    assert verdict.mirror_symmetry_score > 0.90
    assert verdict.tile_seam_continuity_passed is False
    assert verdict.accepted is False


def test_neural_alpha_matting_and_continuous_uv_keyframe_articulation(tmp_path: Path) -> None:
    """Verify neural/chroma-key alpha matting (including enclosed #FF00FF chroma pockets),
    continuous topology-preserving UV keyframe articulation, and StudioRigHelper GDScript export.
    """
    from engine.studio.asset_library import AssetLibraryManager

    # Create a character sprite on a #FF00FF chroma-key background with an enclosed #FF00FF pocket between arm and torso
    raw = Image.new("RGBA", (96, 96), (255, 0, 255, 255))
    d = ImageDraw.Draw(raw)
    # Character body
    d.ellipse([36, 12, 60, 34], fill=(40, 80, 140, 255))  # Head
    # White eye sclera inside head (must NOT be punched out as a transparent hole!)
    d.rectangle([42, 20, 46, 24], fill=(255, 255, 255, 255))
    d.rectangle([34, 34, 62, 66], fill=(35, 110, 160, 255))  # Torso
    # Left arm forming an enclosed loop with torso (leaving #FF00FF chroma pocket inside [26..33, 42..54])
    d.rectangle([20, 36, 34, 42], fill=(35, 110, 160, 255))
    d.rectangle([20, 42, 26, 56], fill=(35, 110, 160, 255))
    d.rectangle([20, 54, 34, 60], fill=(35, 110, 160, 255))
    # Legs
    d.rectangle([38, 66, 46, 84], fill=(30, 60, 90, 255))
    d.rectangle([50, 66, 58, 84], fill=(30, 60, 90, 255))

    cleaned = VisualArtSpriteDesignerAgent.extract_clean_alpha_sprite(
        raw, size=(96, 96), use_neural_matting=False
    )
    # Verify enclosed #FF00FF chroma pocket (mapped to ~(35, 48) after tight-bbox centering) was made transparent
    hole_px = cleaned.getpixel((35, 48))
    assert hole_px[3] == 0, f"Expected enclosed chroma pocket to be transparent, got alpha={hole_px[3]}"
    # Verify white eye sclera (mapped to ~(51, 18)) was NOT punched out as a hole
    eye_px = cleaned.getpixel((51, 18))
    assert eye_px[3] > 200, f"Expected white eye sclera to remain opaque, got alpha={eye_px[3]}"

    # Continuous UV keyframe articulation across 4 frames (preserves topology without severing limbs/props)
    sheet = Image.new("RGBA", (96 * 4, 96), (0, 0, 0, 0))
    tracks = VisualArtSpriteDesignerAgent.default_keyframe_tracks_for_profile("locomotion")
    for f_idx in range(4):
        posed = VisualArtSpriteDesignerAgent.fabricate_pose_frame_from_base(
            cleaned,
            frame_idx=f_idx,
            frame_count=4,
            motion_profile="locomotion",
            target_size=(96, 96),
            keyframe_tracks=tracks,
        )
        sheet.paste(posed, (f_idx * 96, 0))

    sheet_out = tmp_path / "hero_walk_contact.png"
    _, metrics = VisualArtSpriteDesignerAgent.render_animation_contact_sheet(
        sheet_img=sheet,
        states_meta={"walk": {"row": 0, "frame_count": 4, "fps": 10.0, "loop": True, "motion_profile": "locomotion"}},
        cell_size=(96, 96),
        output_path=sheet_out,
    )
    assert metrics["is_truly_animated"] is True
    assert metrics["min_interframe_l1_delta"] > 0.35
    assert metrics["optical_flow_magnitude_mean"] > 0.05

    # Verify StudioRigHelper export
    mgr = AssetLibraryManager(cartridge_dir=tmp_path, client=None)
    helper_path = mgr.export_godot_studio_helper()
    assert helper_path.is_file()
    helper_src = helper_path.read_text(encoding="utf-8")
    assert "class_name StudioRigHelper" in helper_src
    assert "opaque_to_polygons" in helper_src
    assert "AnimationPlayer" in helper_src


def test_custom_non_humanoid_part_tracks_and_hierarchical_articulated_rig(tmp_path: Path) -> None:
    """Verify non-humanoid custom part_id keyframe tracks (e.g. outrigger_arm_0) deform frames across
    both state rows and columns, and hierarchical articulated rigs with pivot_normalized + negative
    z_index_offset assemble and scale cleanly without burying child parts inside the root body.
    """
    import numpy as np
    from engine.studio.models import KeyframeTransformSpec, PartKeyframeTrackSpec

    base = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d = ImageDraw.Draw(base)
    # Central disc + 6 radial arms
    d.ellipse([44, 44, 84, 84], fill=(210, 160, 60, 255))
    for angle_deg in range(0, 360, 60):
        import math
        rad = math.radians(angle_deg)
        x_end = int(64 + math.cos(rad) * 52)
        y_end = int(64 + math.sin(rad) * 52)
        d.line([(64, 64), (x_end, y_end)], fill=(180, 110, 45, 255), width=6)
        d.ellipse([x_end - 6, y_end - 6, x_end + 6, y_end + 6], fill=(160, 180, 200, 255))

    deployed_track = PartKeyframeTrackSpec(
        deliberation="Extended outriggers",
        part_id="outrigger_arm_0",
        keyframes=[
            KeyframeTransformSpec(deliberation="k0", time_normalized=0.0, position_offset_px=(50.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
            KeyframeTransformSpec(deliberation="k1", time_normalized=0.5, position_offset_px=(54.0, 0.0), rotation_degrees=-4.0, scale_xy=(1.04, 1.04)),
            KeyframeTransformSpec(deliberation="k2", time_normalized=1.0, position_offset_px=(50.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
        ],
    )
    retracted_track = PartKeyframeTrackSpec(
        deliberation="Retracted outriggers",
        part_id="outrigger_arm_0",
        keyframes=[
            KeyframeTransformSpec(deliberation="k0", time_normalized=0.0, position_offset_px=(21.0, 0.0), rotation_degrees=78.0, scale_xy=(0.75, 0.75)),
            KeyframeTransformSpec(deliberation="k1", time_normalized=0.5, position_offset_px=(23.0, 0.0), rotation_degrees=72.0, scale_xy=(0.78, 0.78)),
            KeyframeTransformSpec(deliberation="k2", time_normalized=1.0, position_offset_px=(21.0, 0.0), rotation_degrees=78.0, scale_xy=(0.75, 0.75)),
        ],
    )

    sheet = Image.new("RGBA", (128 * 4, 128 * 2), (0, 0, 0, 0))
    for c_idx in range(4):
        f_dep = VisualArtSpriteDesignerAgent.fabricate_pose_frame_from_base(
            base, state_name="outriggers_deployed", frame_idx=c_idx, frame_count=4,
            cell_size=(128, 128), motion_profile="radial_outward_extension", keyframe_tracks=[deployed_track],
        )
        f_ret = VisualArtSpriteDesignerAgent.fabricate_pose_frame_from_base(
            base, state_name="outriggers_retracted", frame_idx=c_idx, frame_count=4,
            cell_size=(128, 128), motion_profile="compact_radial_inward_fold", keyframe_tracks=[retracted_track],
        )
        sheet.paste(f_dep, (c_idx * 128, 0))
        sheet.paste(f_ret, (c_idx * 128, 128))

    _, metrics = VisualArtSpriteDesignerAgent.render_animation_contact_sheet(
        sheet_img=sheet,
        states_meta={
            "outriggers_deployed": {"row": 0, "frame_count": 4, "fps": 8.0, "loop": True, "motion_profile": "action"},
            "outriggers_retracted": {"row": 1, "frame_count": 4, "fps": 12.0, "loop": True, "motion_profile": "action"},
        },
        cell_size=(128, 128),
        output_path=tmp_path / "custom_tracks_contact.png",
    )
    assert metrics["is_truly_animated"] is True
    assert metrics["min_interframe_l1_delta"] > 0.25
    # Verify deployed frame 0 and retracted frame 0 are visually distinct
    arr_dep = np.array(sheet.crop((0, 0, 128, 128)), dtype=np.float32)
    arr_ret = np.array(sheet.crop((0, 128, 128, 256)), dtype=np.float32)
    assert float(np.mean(np.abs(arr_dep - arr_ret))) > 2.0

    # Verify hierarchical articulated rig assembly with pivot_normalized + negative z_index_offset
    eye_img = Image.new("RGBA", (64, 64), (220, 170, 50, 255))
    arm_img = Image.new("RGBA", (32, 12), (40, 210, 220, 255))
    claw_img = Image.new("RGBA", (16, 16), (240, 90, 60, 255))
    parts_with_images = [
        ({"part_id": "chassis_eye", "parent_part_id": None, "pivot_normalized": [0.5, 0.5], "attach_offset_px": [0.0, 0.0], "z_index_offset": 0}, eye_img),
        ({"part_id": "outrigger_arm_0", "parent_part_id": "chassis_eye", "pivot_normalized": [0.1, 0.5], "attach_offset_px": [24.0, 0.0], "z_index_offset": -1}, arm_img),
        ({"part_id": "caliper_claw_0", "parent_part_id": "outrigger_arm_0", "pivot_normalized": [0.15, 0.5], "attach_offset_px": [28.0, 0.0], "z_index_offset": 1}, claw_img),
    ]
    assembled = VisualArtSpriteDesignerAgent._assemble_articulated_rig_frame(parts_with_images, (512, 512), phase=0.0)
    bbox = assembled.getbbox()
    assert bbox is not None
    # Assembled rig auto-fits to fill the 512x512 canvas instead of staying a tiny 64x64 dot
    assert (bbox[2] - bbox[0]) > 250



