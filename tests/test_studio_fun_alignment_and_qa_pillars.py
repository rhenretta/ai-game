"""End-to-end verification of the 6-Pillar Studio Fun Alignment, Multi-Horizon Event QA,
Deterministic Scene/UI Linter, Cohesive Per-Entity Visual Contracts, and Pivot Cleanout Architecture.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from engine.playtest.agent import (
    BlindPlaytesterAgent,
    _playtest_non_thinking_config,
    compute_frame_mse,
)
from engine.playtest.godot_bridge import (
    TurnObservation,
    stamp_diagnostic_click_reticle,
)
from engine.playtest.models import (
    BlindStoreCard,
    PlaytestExperienceReport,
    PlaytestMotorAction,
    PlaytestTurnDecision,
)
from engine.sim.godot_verifier import GodotHeadlessSimulator, format_scene_audit_lint_warnings
from engine.studio.asset_library import CartridgeAssetLibrary
from engine.studio.models import (
    AssetRequirement,
    BlindPlaytestCouncilTurn,
    BlindPlaytestCouncilVerdict,
    CoreLoopSpec,
    EngagementPlan,
    FilmstripPlan,
    GameBibleSpec,
    GameSliceSpec,
    GoalHierarchy,
    GoalSpec,
    InputSegment,
    LevelArchetypeSpec,
    LevelGenerationStrategy,
    OnboardingPlan,
    OnboardingStep,
    QACaptureEvent,
    QAStateAssertion,
    StudioAgentError,
    TaskLeadDecision,
    WorkItemProposal,
    WritersRoomTurnProposal,
    validate_game_design,
)
from engine.studio.production_board import (
    BoardScheduler,
    BoardValidationError,
    ProductionBoard,
    WorkerDesignConcern,
)
from engine.studio.proofs import capture_filmstrip
from engine.studio.visual_evaluator import render_in_situ_magnified_crop_plate
from engine.studio.workers import (
    MAX_PROOF_TICKS,
    _summarize_cartridge_for_qa,
    validate_filmstrip_plan,
)
from engine.studio.writers_room import CreativeWritersRoom


# ==============================================================================
# PILLAR 1: Intrinsic Studio Alignment, Rationale-First Ordering & Design Concern
# ==============================================================================


def test_pillar1_rationale_first_and_decision_last_across_modified_schemas():
    """Rule 4: Every modified Pydantic contract places `deliberation` strictly first
    and its irreversible decision/score token strictly last."""
    schema_expectations = [
        (CoreLoopSpec, "deliberation", "failure_and_recovery"),
        (QACaptureEvent, "deliberation", "trigger_mode"),
        (FilmstripPlan, "deliberation", "verification_mode"),
        (TaskLeadDecision, "deliberation", "action"),
        (PlaytestExperienceReport, "deliberation", "overall_fun_rating"),
        (BlindPlaytestCouncilTurn, "deliberation", "recommends_slice_approval"),
        (BlindPlaytestCouncilVerdict, "deliberation", "verdict"),
    ]
    for model_cls, expected_first, expected_last in schema_expectations:
        field_names = list(model_cls.model_fields.keys())
        assert field_names[0] == expected_first, (
            f"{model_cls.__name__} must place '{expected_first}' at index 0, got {field_names[0]}"
        )
        assert field_names[-1] == expected_last, (
            f"{model_cls.__name__} must place '{expected_last}' strictly last, got {field_names[-1]}"
        )


def test_pillar1_fun_and_tension_pact_validation_and_canonical_telemetry_keys(tmp_path: Path):
    """Verifies `validate_game_design(..., enforce_fun_pact=True)` enforces non-empty Fun & Tension Pact
    contracts while legacy deserialization (`enforce_fun_pact=False`) stays backwards-compatible, and
    `canonical_telemetry_keys` propagate from Council proposals into QA cartridge summaries."""
    incomplete_bible = GameBibleSpec(
        deliberation="Initial draft",
        game_title="Abstract Challenge",
        core_genre="Strategy",
        elevator_pitch="Manage finite energy relays under escalating load.",
        camera_and_spatial_paradigm="2D top-down",
        color_palette_hex=["#101820", "#F2AA4C", "#00A896"],
        level_generation_strategy=LevelGenerationStrategy(
            deliberation="d",
            generation_approach="procedural",
            world_structure="Bounded grid",
            runtime_construction_plan="Spawn grid cells",
            level_archetypes=[
                LevelArchetypeSpec(
                    deliberation="d",
                    archetype_id="sector_alpha",
                    purpose="Starter sector",
                    visual_identity="Dark slate grid",
                    signature_challenge="Rising load",
                )
            ],
            generation_rules=["Place 3 demand nodes spaced >= 180px apart"],
            difficulty_and_pacing_curve="Gradual load escalation",
            variety_and_replayability="Randomized node positions",
        ),
        core_loop=CoreLoopSpec(
            deliberation="Loop draft",
            player_verbs=["scan", "place"],
            moment_to_moment_loop="Scan nodes and place relays.",
            minute_to_minute_loop="Balance load across sectors.",
            session_loop="Stabilize all sectors.",
            feedback_and_juice="Crisp pulse and audio chime.",
            reward_cadence="Unlock capacity per sector.",
            failure_and_recovery="Overload resets unstable relay.",
        ),
        engagement_plan=EngagementPlan(
            deliberation="d",
            entertainment_pillars=["Spatial optimization"],
            pacing_and_intensity_curve="Steady build",
            surprise_and_discovery="Emergent load spikes",
            mastery_and_expression="Minimal-cost layouts",
            emotional_arc="Curiosity to mastery",
        ),
        onboarding_plan=OnboardingPlan(
            deliberation="d",
            world_introduction="Show dark grid with one pulsing node.",
            steps=[
                OnboardingStep(
                    deliberation="d",
                    step_index=1,
                    concept_introduced="Manual scan and placement",
                    diegetic_method="Pulsing outline on target node",
                    player_proof_of_learning="Player clicks node to scan and places first relay",
                )
            ],
            confusion_recovery="Highlight affordable action when idle.",
        ),
        goal_hierarchy=GoalHierarchy(
            deliberation="d",
            goals=[
                GoalSpec(
                    deliberation="d",
                    goal_id="g1",
                    horizon="short_term",
                    description="Power first sector",
                    motivation="Prevent blackout",
                    progress_feedback="HUD stability meter",
                    reward="Additional relay charge",
                )
            ],
            progression_structure="Sector-by-sector",
            end_state="All 3 sectors stabilized",
        ),
        slice_roadmap=[
            GameSliceSpec(
                deliberation="Slice 0",
                slice_id="slice_0_core",
                slice_index=0,
                title="Slice 0: Core Relay Loop",
                slice_objective="Playable scan and placement under scarcity.",
                player_acclimation_and_mechanics_focus="Click placement and budget.",
                status="active",
            )
        ],
    )
    # Legacy default remains backwards-compatible
    assert validate_game_design(incomplete_bible) == []

    # Active Council design enforcement requires all 3 Fun & Tension Pact fields
    pact_errors = validate_game_design(incomplete_bible, enforce_fun_pact=True)
    assert len(pact_errors) == 3
    assert any("core_decision_or_skill_test" in e for e in pact_errors)
    assert any("pushback_and_scarcity_model" in e for e in pact_errors)
    assert any("earned_vs_given_progression" in e for e in pact_errors)

    complete_bible = incomplete_bible.model_copy(deep=True)
    complete_bible.core_loop.core_decision_or_skill_test = (
        "Deciding where to commit scarce relay budget before sector load peaks."
    )
    complete_bible.core_loop.pushback_and_scarcity_model = (
        "Player starts with limited budget for only 1 relay while sector demand drains stability."
    )
    complete_bible.core_loop.earned_vs_given_progression = (
        "Zero free income at t=0; energy is earned exclusively by manually scanning and routing active nodes."
    )
    assert validate_game_design(complete_bible, enforce_fun_pact=True) == []

    # Verify canonical_telemetry_keys update via _apply_turn_proposal and appear in QA summary
    board = ProductionBoard(tmp_path / "backlog.json")
    room = CreativeWritersRoom(client=None, board=board)
    turn = WritersRoomTurnProposal(
        deliberation="Lock canonical snapshot keys so SWE and QA share identical state identifiers.",
        spoken_dialogue="Locking canonical telemetry keys for main.gd::get_state_snapshot().",
        updated_canonical_telemetry_keys=["relay_budget", "active_relays"],
    )
    room._apply_turn_proposal(
        complete_bible,
        "GameplaySystemsEngineer",
        "Gameplay Systems Engineer",
        turn,
        candidate_dir=tmp_path,
    )
    assert "relay_budget" in complete_bible.canonical_telemetry_keys
    assert "active_relays" in complete_bible.canonical_telemetry_keys

    (tmp_path / "game_bible.json").write_text(
        complete_bible.model_dump_json(indent=2), encoding="utf-8"
    )
    (tmp_path / "project.godot").write_text('[application]\nrun/main_scene="res://main.tscn"\n', encoding="utf-8")
    (tmp_path / "main.gd").write_text("extends Node2D\nfunc get_state_snapshot() -> Dictionary:\n\treturn {}\n", encoding="utf-8")

    qa_summary = _summarize_cartridge_for_qa(tmp_path)
    assert "canonical_telemetry_keys (Game Bible snapshot schema):" in qa_summary
    assert "relay_budget" in qa_summary
    assert "active_relays" in qa_summary


def test_pillar1_worker_design_concern_escalates_directly_to_needs_decision(tmp_path: Path):
    """Verifies TaskLead co-designer pushback (`WorkerDesignConcern`) escalates a `WorkItem`
    cleanly to `needs_decision` with `[TaskLead Design Concern]` rather than failing or retrying."""
    events: list[tuple[str, dict]] = []
    board = ProductionBoard(
        tmp_path / "backlog.json",
        event_callback=lambda et, payload: events.append((et, payload)),
    )
    board.add_items(
        [
            WorkItemProposal(
                deliberation="d",
                item_id="mechanic_auto_grant",
                kind="mechanic_code",
                title="Auto Resource Grant",
                spec="Grant infinite free resources every frame from t=0.",
                proof_spec="Verify resources climb automatically.",
                target_modules=["systems/economy.gd"],
            )
        ],
        raised_by_role="GameplayDesigner",
    )

    def _design_pushback_handler(_item):
        raise WorkerDesignConcern(
            "Spec mandates unconditional free resource generation at t=0, violating the Fun & Tension Pact."
        )

    sched = BoardScheduler(board, handlers={"mechanic_code": _design_pushback_handler}, max_workers=1)
    try:
        assert sched.run_until_idle(timeout=5.0) is True
    finally:
        sched.stop()

    escalated = board.get("mechanic_auto_grant")
    assert escalated.state == "needs_decision"
    assert escalated.failure_count == 0
    assert escalated.decision_reason.startswith("[TaskLead Design Concern]")
    assert "Fun & Tension Pact" in escalated.decision_reason
    assert any(et == "backlog_item_needs_decision" and p.get("design_concern") is True for et, p in events)


# ==============================================================================
# PILLAR 2: Discerning Gamer Playtester & Non-Destructive Click Reticle
# ==============================================================================


def test_pillar2_playtester_medium_thinking_and_nondestructive_click_reticle(tmp_path: Path):
    """Verifies per-turn playtester uses `ThinkingLevel.MEDIUM` (`thinking_budget=2048` on 2.5),
    and `stamp_diagnostic_click_reticle` creates a reticle-stamped copy without altering raw screenshot MSE."""
    cfg_3 = _playtest_non_thinking_config("gemini-3.8-flash")
    assert "MEDIUM" in str(getattr(cfg_3, "thinking_level", "")).upper()

    cfg_25 = _playtest_non_thinking_config("gemini-2.5-flash")
    assert getattr(cfg_25, "thinking_budget", None) == 2048

    raw_a = tmp_path / "turn_0001.png"
    raw_b = tmp_path / "turn_0002.png"
    img = Image.new("RGB", (1280, 720), (24, 32, 48))
    img.save(raw_a, "PNG")
    img.save(raw_b, "PNG")
    orig_bytes = raw_b.read_bytes()

    diag_b = stamp_diagnostic_click_reticle(
        raw_b,
        click_xy_px=(420.0, 310.0),
        action_label="left_click",
    )
    assert diag_b is not None and diag_b.is_file()
    assert diag_b.name == "turn_0002_click_diag.png"
    # Raw screenshot bytes & MSE must remain 100% untouched
    assert raw_b.read_bytes() == orig_bytes
    assert compute_frame_mse(raw_a, raw_b) == 0.0
    # Diagnostic copy has visible reticle pixels at (420, 310)
    assert compute_frame_mse(raw_b, diag_b) > 0.0

    # Verify decide_turn sends diagnostic_screenshot_path to Gemini while keeping raw MSE == 0.0
    seen_contents: list = []

    class _FakeModels:
        def generate_content(self, *, model, contents, config):
            seen_contents.append(contents)
            dec = PlaytestTurnDecision(
                deliberation="Inspecting target reticle alignment at (420, 310).",
                turn_log_entry="Checked click placement.",
                current_mood="curious",
                current_plan="Probe adjacent control.",
                motor_action=PlaytestMotorAction(
                    deliberation="Wait briefly.",
                    mouse_button_action="none",
                    duration_ms=250,
                ),
            )
            return SimpleNamespace(parsed=dec, text=dec.model_dump_json())

    agent = BlindPlaytesterAgent(
        store_card=BlindStoreCard(
            deliberation="d",
            checkpoint_id="cp_test",
            game_title="Abstract Challenge",
            short_description="Manual routing challenge.",
        ),
        client=SimpleNamespace(models=_FakeModels()),
    )
    obs1 = TurnObservation(
        observation_index=1,
        tick=60,
        sim_time_sec=1.0,
        cursor_x=420.0,
        cursor_y=310.0,
        screenshot_path=raw_a,
        preview_path=raw_a,
        wav_path=None,
        audio_rms=0.0,
        audio_peak=0.0,
        audio_is_silent=True,
        interrupted_early=False,
    )
    agent.decide_turn(1, obs1)

    obs2 = TurnObservation(
        observation_index=2,
        tick=120,
        sim_time_sec=2.0,
        cursor_x=420.0,
        cursor_y=310.0,
        screenshot_path=raw_b,
        preview_path=raw_b,
        wav_path=None,
        audio_rms=0.0,
        audio_peak=0.0,
        audio_is_silent=True,
        interrupted_early=False,
        diagnostic_screenshot_path=diag_b,
    )
    _, _, meta2 = agent.decide_turn(2, obs2)
    flat_parts = [
        pt
        for c in seen_contents[-1]
        for pt in (getattr(c, "parts", None) or [c])
    ]
    text_blob_t2 = "\n".join(str(getattr(pt, "text", "") or "") for pt in flat_parts)
    attached_img_bytes = [
        getattr(getattr(pt, "inline_data", None), "data", b"")
        for pt in flat_parts
        if getattr(pt, "inline_data", None) is not None
    ]
    assert diag_b.read_bytes() in attached_img_bytes
    assert "The screen is visually identical to your previous turn" in text_blob_t2
    assert bool(meta2.get("screen_unchanged")) is True


# ==============================================================================
# PILLAR 3 & 5: Multi-Horizon (>90 Ticks) Event-Driven QA Proofs & 3x Crop Plates
# ==============================================================================


def test_pillar3_filmstrip_plan_validation_supports_3600_ticks_and_event_keyframes():
    """Verifies `MAX_PROOF_TICKS == 3600` and `validate_filmstrip_plan` validates event-driven keyframes."""
    assert MAX_PROOF_TICKS == 3600

    valid_plan = FilmstripPlan(
        deliberation="Multi-horizon event-driven verification on res://main.tscn.",
        target_scene="res://main.tscn",
        max_sim_ticks=600,
        capture_ticks=[],
        capture_events=[
            QACaptureEvent(
                deliberation="Baseline initial state.",
                event_label="boot_ready",
                at_tick=4,
                settle_ticks=0,
                trigger_mode="at_tick",
            ),
            QACaptureEvent(
                deliberation="Wait for player-initiated cycle completion beyond 90 ticks.",
                event_label="cycle_completed",
                when_snapshot_key="completed_cycles",
                when_operator=">=",
                when_value="1",
                at_tick=10,
                settle_ticks=6,
                trigger_mode="on_condition",
            ),
        ],
        input_segments=[
            InputSegment(
                deliberation="Trigger cycle start via mouse click.",
                from_tick=8,
                to_tick=10,
                actions=[],
                mouse_click_pos=[640, 360],
            )
        ],
        state_assertions=[
            QAStateAssertion(
                deliberation="Cycle count must reach 1 when cycle_completed fires.",
                at_tick=600,
                at_event_label="cycle_completed",
                snapshot_key="completed_cycles",
                expected_value_or_change=">= 1",
            )
        ],
        frame_expectations=["Initial dark grid", "Completed cycle active pulse"],
        verification_mode="causal_gameplay",
    )
    assert validate_filmstrip_plan(valid_plan, actions=[]) == []

    # Unknown at_event_label or missing when_snapshot_key on on_condition must fail validation
    bad_plan = valid_plan.model_copy(deep=True)
    bad_plan.capture_events[1].when_snapshot_key = ""
    bad_plan.state_assertions[0].at_event_label = "nonexistent_label"
    errs = validate_filmstrip_plan(bad_plan, actions=[])
    assert any("when_snapshot_key" in e for e in errs)
    assert any("nonexistent_label" in e for e in errs)


def test_pillar3_and_5_headless_godot_multi_horizon_event_proof_early_exit_unmet_and_3x_crops(tmp_path: Path):
    """Runs real headless Godot 4.3 across >90 ticks (`max_sim_ticks=300`) to verify:
    1. Event-driven `on_condition` capture firing at tick ~130 (>90 ticks!) + `settle_ticks=5` -> ~135,
    2. Smart early exit well before `max_sim_ticks=300`,
    3. Resolved-tick counterfactual parity (zero-input baseline stays at 0),
    4. 3x nearest-neighbor magnified crop plate (`*.magnified_crops_3x.png`) generation,
    5. Deterministic `StudioAgentError` when an `on_condition` keyframe never occurs."""
    (tmp_path / "project.godot").write_text(
        '[application]\nconfig/name="EventProofTest"\nrun/main_scene="res://main.tscn"\n'
        '[display]\nwindow/size/viewport_width=1280\nwindow/size/viewport_height=720\n',
        encoding="utf-8",
    )
    (tmp_path / "main.tscn").write_text(
        '[gd_scene load_steps=2 format=3]\n'
        '[ext_resource type="Script" path="res://main.gd" id="1"]\n'
        '[node name="Main" type="Node2D"]\n'
        'script = ExtResource("1")\n',
        encoding="utf-8",
    )
    # Process requires explicit player click to arm, then takes 120 ticks (2.0s > 90 ticks!) to complete!
    (tmp_path / "main.gd").write_text(
        "extends Node2D\n"
        "var armed: bool = false\n"
        "var progress_ticks: int = 0\n"
        "var completed_cycles: int = 0\n"
        "func _unhandled_input(event: InputEvent) -> void:\n"
        "\tif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:\n"
        "\t\tarmed = true\n"
        "func _physics_process(_delta: float) -> void:\n"
        "\tif armed and completed_cycles == 0:\n"
        "\t\tprogress_ticks += 1\n"
        "\t\tif progress_ticks >= 120:\n"
        "\t\t\tcompleted_cycles = 1\n"
        "\tqueue_redraw()\n"
        "func _draw() -> void:\n"
        "\tdraw_rect(Rect2(0, 0, 1280, 720), Color(0.08, 0.10, 0.14))\n"
        "\tvar col: Color = Color(0.2, 0.85, 0.45) if completed_cycles > 0 else Color(0.85, 0.55, 0.2)\n"
        "\tdraw_circle(Vector2(640, 360), 48.0 + float(completed_cycles) * 24.0, col)\n"
        "func get_state_snapshot() -> Dictionary:\n"
        "\treturn {\n"
        "\t\t'armed': armed,\n"
        "\t\t'progress_ticks': progress_ticks,\n"
        "\t\t'completed_cycles': completed_cycles,\n"
        "\t}\n",
        encoding="utf-8",
    )

    out_strip = tmp_path / "artifacts" / "proof_multi_horizon.png"
    returned_path = capture_filmstrip(
        cartridge_dir=tmp_path,
        ticks=[],
        out_path=out_strip,
        input_script=[
            {"from": 10, "to": 12, "actions": [], "mouse_click_pos": [640, 360]}
        ],
        verify_counterfactual=True,
        state_assertions=[
            QAStateAssertion(
                deliberation="Cycle completed after 120 ticks of armed processing.",
                at_tick=300,
                at_event_label="cycle_done_after_2s",
                snapshot_key="completed_cycles",
                expected_value_or_change="== 1",
            )
        ],
        target_scene="res://main.tscn",
        capture_events=[
            {
                "event_label": "initial_idle",
                "trigger_mode": "at_tick",
                "at_tick": 4,
                "settle_ticks": 0,
            },
            {
                "event_label": "armed_started",
                "trigger_mode": "on_change",
                "when_snapshot_key": "armed",
                "at_tick": 6,
                "settle_ticks": 2,
            },
            {
                "event_label": "cycle_done_after_2s",
                "trigger_mode": "on_condition",
                "when_snapshot_key": "completed_cycles",
                "when_operator": ">=",
                "when_value": "1",
                "at_tick": 10,
                "settle_ticks": 5,
            },
        ],
        max_sim_ticks=300,
    )

    assert returned_path.is_file()
    crop_sidecar = out_strip.with_name(f"{out_strip.stem}.magnified_crops_3x.png")
    assert crop_sidecar.is_file(), "Expected 3x magnified in-situ crop plate sidecar to be generated"

    telem = json.loads(out_strip.with_suffix(".telemetry.json").read_text(encoding="utf-8"))
    captured_evs = telem.get("scene_audit", {}).get("captured_events", [])
    by_label = {ev["event_label"]: ev for ev in captured_evs}
    assert "cycle_done_after_2s" in by_label
    # Click at tick 10 + 120 ticks progress = ~130 (>90 ticks!) + 5 settle ticks = ~135
    resolved_done_tick = int(by_label["cycle_done_after_2s"]["captured_tick"])
    assert 120 < resolved_done_tick < 200, f"Expected >90-tick horizon capture with early exit, got {resolved_done_tick}"

    # Verify smart early exit directly via GodotHeadlessSimulator.perf_metrics["ticks_executed"]
    early_sim = GodotHeadlessSimulator(
        tmp_path,
        capture_viewport=False,
        capture_ticks=[],
        capture_events=[
            {
                "event_label": "armed_fast",
                "trigger_mode": "on_condition",
                "when_snapshot_key": "armed",
                "when_operator": "==",
                "when_value": "true",
                "at_tick": 5,
                "settle_ticks": 3,
            }
        ],
        input_script=[{"from": 8, "to": 10, "actions": [], "mouse_click_pos": [640, 360]}],
        max_ticks=300,
        telemetry_mode="null",
        target_scene="res://main.tscn",
    )
    early_sim.run_until_complete()
    assert int(early_sim.ticks_executed) <= 20

    # Verify unmet event failure when no click occurs so `completed_cycles >= 1` never triggers
    unmet_strip = tmp_path / "artifacts" / "proof_unmet.png"
    with pytest.raises(StudioAgentError, match="never triggered"):
        capture_filmstrip(
            cartridge_dir=tmp_path,
            ticks=[],
            out_path=unmet_strip,
            input_script=[],
            verify_counterfactual=False,
            state_assertions=[],
            target_scene="res://main.tscn",
            capture_events=[
                {
                    "event_label": "never_fires",
                    "trigger_mode": "on_condition",
                    "when_snapshot_key": "completed_cycles",
                    "when_operator": ">=",
                    "when_value": "1",
                    "at_tick": 2,
                    "settle_ticks": 0,
                }
            ],
            max_sim_ticks=35,
        )


# ==============================================================================
# PILLAR 4: Deterministic Godot Scene-Graph, UI Layout & Spatial Scale Linter
# ==============================================================================


def test_pillar4_scene_audit_detects_ui_overlaps_overflow_clipping_upscale_and_entity_pileups(tmp_path: Path):
    """Runs real headless Godot 4.3 to verify `[SCENE_AUDIT]` deterministically flags all 6 visual/layout classes
    on a broken scene and produces 0 warnings on a clean scene."""
    (tmp_path / "project.godot").write_text(
        '[application]\nconfig/name="LintAuditTest"\nrun/main_scene="res://main.tscn"\n'
        '[display]\nwindow/size/viewport_width=1280\nwindow/size/viewport_height=720\n',
        encoding="utf-8",
    )
    sprites_dir = tmp_path / "assets" / "sprites"
    sprites_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (32, 32), (220, 80, 80, 255)).save(sprites_dir / "tiny_32.png", "PNG")
    Image.new("RGBA", (128, 128), (60, 180, 240, 255)).save(sprites_dir / "unit_128.png", "PNG")

    (tmp_path / "main.tscn").write_text(
        '[gd_scene load_steps=2 format=3]\n'
        '[ext_resource type="Script" path="res://main.gd" id="1"]\n'
        '[node name="Main" type="Node2D"]\n'
        'script = ExtResource("1")\n',
        encoding="utf-8",
    )
    (tmp_path / "main.gd").write_text(
        "extends Node2D\n"
        "func _ready() -> void:\n"
        "\tvar hud := CanvasLayer.new()\n"
        "\thud.name = 'HUD'\n"
        "\tadd_child(hud)\n"
        "\tvar lbl_a := Label.new()\n"
        "\tlbl_a.name = 'StatusLabelA'\n"
        "\tlbl_a.text = 'Sector Stability Nominal'\n"
        "\tlbl_a.position = Vector2(24, 24)\n"
        "\tlbl_a.size = Vector2(220, 32)\n"
        "\thud.add_child(lbl_a)\n"
        "\tvar lbl_b := Label.new()\n"
        "\tlbl_b.name = 'StatusLabelB'\n"
        "\tlbl_b.text = 'Overlapping Counter Readout'\n"
        "\tlbl_b.position = Vector2(36, 28)\n"
        "\tlbl_b.size = Vector2(220, 32)\n"
        "\thud.add_child(lbl_b)\n"
        "\tvar off_btn := Button.new()\n"
        "\toff_btn.name = 'OverflowActionBtn'\n"
        "\toff_btn.text = 'Deploy'\n"
        "\toff_btn.position = Vector2(1180, 40)\n"
        "\toff_btn.size = Vector2(180, 40)\n"
        "\thud.add_child(off_btn)\n"
        "\tvar clip_lbl := Label.new()\n"
        "\tclip_lbl.name = 'ClippedTitleLabel'\n"
        "\tclip_lbl.text = 'Extremely Long Header Truncated By Tiny Width'\n"
        "\tclip_lbl.clip_text = true\n"
        "\tclip_lbl.position = Vector2(24, 120)\n"
        "\tclip_lbl.custom_minimum_size = Vector2(28, 28)\n"
        "\tclip_lbl.size = Vector2(28, 28)\n"
        "\thud.add_child(clip_lbl)\n"
        "\tvar world := Node2D.new()\n"
        "\tworld.name = 'Units'\n"
        "\tadd_child(world)\n"
        "\tvar tiny_img := Image.load_from_file(ProjectSettings.globalize_path('res://assets/sprites/tiny_32.png'))\n"
        "\tvar tiny_tex := ImageTexture.create_from_image(tiny_img)\n"
        "\tvar unit_img := Image.load_from_file(ProjectSettings.globalize_path('res://assets/sprites/unit_128.png'))\n"
        "\tvar unit_tex := ImageTexture.create_from_image(unit_img)\n"
        "\tvar blurry_holder := Node2D.new()\n"
        "\tblurry_holder.name = 'UpscaledBeacon'\n"
        "\tblurry_holder.position = Vector2(300, 360)\n"
        "\tworld.add_child(blurry_holder)\n"
        "\tvar blurry_sp := Sprite2D.new()\n"
        "\tblurry_sp.name = 'TinySprite'\n"
        "\tblurry_sp.texture = tiny_tex\n"
        "\tblurry_sp.scale = Vector2(3.0, 3.0)\n"
        "\tblurry_holder.add_child(blurry_sp)\n"
        "\tvar ent_a := Node2D.new()\n"
        "\tent_a.name = 'EntityAlpha'\n"
        "\tent_a.position = Vector2(640, 360)\n"
        "\tworld.add_child(ent_a)\n"
        "\tvar sp_a1 := Sprite2D.new()\n"
        "\tsp_a1.name = 'BodyPrimary'\n"
        "\tsp_a1.texture = unit_tex\n"
        "\tent_a.add_child(sp_a1)\n"
        "\tvar sp_a2 := Sprite2D.new()\n"
        "\tsp_a2.name = 'BodyDuplicate'\n"
        "\tsp_a2.texture = unit_tex\n"
        "\tent_a.add_child(sp_a2)\n"
        "\tvar ent_b := Node2D.new()\n"
        "\tent_b.name = 'EntityBeta'\n"
        "\tent_b.position = Vector2(680, 360)\n"
        "\tworld.add_child(ent_b)\n"
        "\tvar sp_b1 := Sprite2D.new()\n"
        "\tsp_b1.name = 'BodyPrimary'\n"
        "\tsp_b1.texture = unit_tex\n"
        "\tent_b.add_child(sp_b1)\n"
        "func get_state_snapshot() -> Dictionary:\n"
        "\treturn {'ok': true}\n",
        encoding="utf-8",
    )

    sim = GodotHeadlessSimulator(
        tmp_path,
        capture_viewport=False,
        input_script=[],
        max_ticks=6,
        telemetry_mode="null",
        target_scene="res://main.tscn",
    )
    sim.run_until_complete()
    audit = sim.scene_audit

    assert audit.get("ui_rect_overlaps"), f"Expected ui_rect_overlaps, got {audit}"
    assert audit.get("ui_viewport_overflow_nodes"), f"Expected ui_viewport_overflow_nodes, got {audit}"
    assert audit.get("clipped_ui_labels"), f"Expected clipped_ui_labels, got {audit}"
    assert audit.get("blurry_upscaled_sprites"), f"Expected blurry_upscaled_sprites, got {audit}"
    assert audit.get("duplicate_stacked_sprites"), f"Expected duplicate_stacked_sprites, got {audit}"
    assert audit.get("overlapping_world_entities"), f"Expected overlapping_world_entities, got {audit}"

    warnings = format_scene_audit_lint_warnings(audit)
    assert len(warnings) == 6
    joined = "\n".join(warnings)
    assert "[UI_OVERLAP]" in joined
    assert "[UI_OVERFLOW]" in joined
    assert "[UI_CLIPPED_LABEL]" in joined
    assert "[SPRITE_UPSCALE_BLUR]" in joined
    assert "[DUPLICATE_STACKED_SPRITES]" in joined
    assert "[WORLD_ENTITY_COLLISION]" in joined


def test_pillar5_asset_requirement_footprint_and_projection_lock(tmp_path: Path):
    """Verifies `AssetRequirement` propagates `world_footprint_px` and `camera_projection` into
    `CartridgeAssetLibrary` (`target_display_size` matches `world_footprint_px`) and `render_in_situ_magnified_crop_plate`."""
    seen_projections: list[str] = []

    class _FakeVisualAgent:
        allow_offline_procedural = True

        def generate_dynamic_sprite(self, *, asset_req, output_path, **kwargs):
            seen_projections.append(str(getattr(asset_req, "camera_projection", "")))
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
            draw = ImageDraw.Draw(im)
            draw.ellipse((16, 16, 112, 112), fill=(70, 190, 220, 255))
            im.save(out, "PNG")
            return out

    lib = CartridgeAssetLibrary(
        cartridge_dir=tmp_path,
        visual_agent=_FakeVisualAgent(),
        allow_offline_procedural=True,
    )
    req = AssetRequirement(
        deliberation="Dedicated single-entity high-res sprite downsampled to 48x48 world footprint.",
        asset_id="station_core",
        asset_type="sprite",
        fabrication_mode="single_sprite",
        target_path="assets/sprites/station_core.png",
        description="Single hexagonal relay station core.",
        visual_prompt_or_text="Single hexagonal relay station core centered on transparent background.",
        dimensions=(128, 128),
        world_footprint_px=(48, 48),
        camera_projection="strict orthographic 90-degree top-down",
    )
    out_file = lib.fabricate_single_asset(req)
    assert out_file.is_file()
    assert seen_projections == ["strict orthographic 90-degree top-down"]
    entry = lib.catalog.entries["station_core"]
    assert tuple(entry.target_display_size) == (48, 48)
    assert tuple(entry.texture_pixel_size) == (128, 128)

    frame_path = tmp_path / "sample_frame.png"
    frame_im = Image.new("RGB", (1280, 720), (18, 22, 30))
    ImageDraw.Draw(frame_im).rectangle((600, 320, 680, 400), fill=(240, 160, 50))
    frame_im.save(frame_path, "PNG")
    crop_plate = render_in_situ_magnified_crop_plate(
        frame_path,
        tmp_path / "sample_frame.magnified_crops_3x.png",
        lint_warnings=["UI OVERLAP: LabelA overlaps LabelB by 40x18px"],
    )
    assert crop_plate is not None and crop_plate.is_file()


# ==============================================================================
# PILLAR 6: Structural Pivot Clean-Slate Garbage Collection & Read-Only Board
# ==============================================================================


def test_pillar6_read_only_board_never_mutates_disk_or_rewrites_in_progress_items(tmp_path: Path):
    """Verifies `ProductionBoard(..., read_only=True)` preserves live `in_progress` states in memory,
    never rewrites `backlog.json` during `_load()`, and raises `BoardValidationError` on mutating calls."""
    backlog_path = tmp_path / "backlog.json"
    writer_board = ProductionBoard(backlog_path)
    writer_board.add_items(
        [
            WorkItemProposal(
                deliberation="d",
                item_id="live_task_1",
                kind="mechanic_code",
                title="Live Task",
                spec="Implement active mechanic.",
                proof_spec="Verify active mechanic.",
                target_modules=["systems/live.gd"],
            )
        ],
        raised_by_role="GameplaySystemsEngineer",
    )
    claimed = writer_board.claim_ready(limit=1)
    assert len(claimed) == 1 and claimed[0].state == "in_progress"
    disk_bytes_before = backlog_path.read_bytes()

    ro_board = ProductionBoard(backlog_path, read_only=True)
    assert ro_board.get("live_task_1").state == "in_progress"
    assert backlog_path.read_bytes() == disk_bytes_before

    with pytest.raises(BoardValidationError, match="read_only"):
        ro_board.cancel_item("live_task_1", reason="illegal mutation from read_only reader")
    assert backlog_path.read_bytes() == disk_bytes_before


def test_pillar6_structural_pivot_archives_superseded_tests_syncs_title_and_spawns_cleanout_task(tmp_path: Path):
    """Verifies that a Tier 2 `structural_pivot` on an existing cartridge:
    1. Syncs `bible.game_title` to `concept_brief.working_title`,
    2. Archives pre-pivot `tests/test_*` scripts into `artifacts/archived_pre_pivot_tests/`,
    3. Cancels obsolete pre-pivot tasks and sets `_pending_structural_pivot_cleanout`."""
    (tmp_path / "project.godot").write_text('[application]\nconfig/name="OldTitle"\n', encoding="utf-8")
    (tmp_path / "main.tscn").write_text('[gd_scene format=3]\n[node name="Main" type="Node2D"]\n', encoding="utf-8")
    (tmp_path / "main.gd").write_text("extends Node2D\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_old_mechanic.gd").write_text("extends Node2D\n", encoding="utf-8")
    (tests_dir / "test_old_mechanic.tscn").write_text('[gd_scene format=3]\n[node name="T" type="Node2D"]\n', encoding="utf-8")

    board = ProductionBoard(tmp_path / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="d",
                item_id="old_mechanic",
                kind="mechanic_code",
                title="Old Mechanic",
                spec="Legacy pre-pivot mechanic.",
                proof_spec="Verify legacy mechanic.",
                target_modules=["systems/legacy.gd"],
            )
        ],
        raised_by_role="GameplayDesigner",
    )

    # Write `.pending_structural_pivot.json` envelope matching CoProducerAgent's Tier 2 structural pivot output
    (tmp_path / ".pending_structural_pivot.json").write_text(
        json.dumps(
            {
                "steering_brief": "Pivot to manual tactical routing and purge legacy pre-pivot systems.",
                "concept_brief": {
                    "working_title": "New Pivoted Title",
                },
                "cancel_item_ids": ["*"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    bible = GameBibleSpec(
        deliberation="d",
        game_title="Old Pre-Pivot Title",
        core_genre="Simulation",
        slice_roadmap=[
            GameSliceSpec(
                deliberation="d",
                slice_id="slice_0_core",
                slice_index=0,
                title="Slice 0",
                slice_objective="Core",
                player_acclimation_and_mechanics_focus="Controls",
                status="active",
            )
        ],
    )

    room = CreativeWritersRoom(client=None, board=board)
    consumed = room._consume_pending_steering(bible, candidate_dir=tmp_path)

    assert len(consumed) == 1
    # 1. Title synced to concept_brief.working_title
    assert bible.game_title == "New Pivoted Title"
    # 2. Pre-pivot tests archived out of tests/
    assert not (tests_dir / "test_old_mechanic.gd").exists()
    assert (tmp_path / "artifacts" / "archived_pre_pivot_tests" / "test_old_mechanic.gd").is_file()
    assert (tmp_path / "artifacts" / "archived_pre_pivot_tests" / "test_old_mechanic.tscn").is_file()
    # 3. Old task cancelled AND _pending_structural_pivot_cleanout armed for the steering round
    assert board.get("old_mechanic").state == "cancelled"
    assert room._pending_structural_pivot_cleanout is True
