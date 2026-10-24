"""Code workers, orchestrator wiring and an end-to-end council <-> board <-> workers run (all fakes, no network)."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import tests.test_board_workers as bw
import tests.test_council_backlog as cb
from engine.studio.models import (
    FilmstripPlan, InputSegment, RequirementTrack, StudioAgentError, StudioOrchestratorConfig, WorkItem,
    WorkItemProposal,
)
from engine.studio.orchestrator import StudioOrchestrator
from engine.studio.production_board import BoardScheduler, ProductionBoard
from engine.studio.workers import (
    WorkerContext, available_input_actions, code_item_handler, default_handlers, plan_filmstrip, validate_filmstrip_plan,
)


class FakeSwe:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def implement_requirement_track(self, track, staging_dir, requirements_spec=None, **kw):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            self.calls.append(track)
            time.sleep(0.03)
            for m in track.target_modules:
                (Path(staging_dir) / m).write_text(f"# implemented {track.id}\n")
            if self.fail:
                raise RuntimeError("compile error in generated code")
            return {}
        finally:
            with self.lock:
                self.active -= 1


def plan_json(ticks=(5, 20), segs=(), expectations=("before", "after")):
    return FilmstripPlan(
        deliberation="show before/after", capture_ticks=list(ticks),
        input_segments=[InputSegment(deliberation="d", from_tick=a, to_tick=b, actions=acts) for a, b, acts in segs],
        frame_expectations=list(expectations),
    ).model_dump_json()


def code_ctx(tmp_path, responder, swe=None, verify=None, strip_fn=None):
    (tmp_path / "game_bible.json").write_text(json.dumps({"game_title": "Test Game"}))
    (tmp_path / "project.godot").write_text('[input]\ndive={"events":[]}\nsoar={"events":[]}\n[rendering]\nx=1\n')
    calls = {"strip": []}

    def fake_strip(cartridge_dir, ticks, out_path, input_script=None):
        calls["strip"].append((list(ticks), input_script))
        Image.new("RGB", (30, 10), (5, 5, 5)).save(out_path)
        return Path(out_path)

    ctx = WorkerContext(cartridge_dir=tmp_path, client=bw.make_client(responder), swe_agent=swe or FakeSwe(),
                        code_verify=verify, filmstrip_fn=strip_fn or fake_strip, library_factory=bw.FakeLibrary)
    return ctx, calls


def code_item(item_id="dive", modules=("mechanics.gd",), revisions=None):
    return WorkItem(deliberation="d", item_id=item_id, kind="mechanic_code", title="Dive mechanic", spec="dive faster",
                    proof_spec="skiff dives then recovers", target_modules=list(modules), state="in_progress",
                    revisions=revisions or [])


# ----------------------------------------------------------------------------- code worker
def test_input_actions_come_from_project_godot_and_plans_are_validated_structurally(tmp_path):
    ctx, _ = code_ctx(tmp_path, lambda c: plan_json())
    actions = available_input_actions(tmp_path)
    assert {"dive", "soar", "ui_right"} <= set(actions) and "x" not in actions
    bad = FilmstripPlan(deliberation="d", capture_ticks=[0, 5000], input_segments=[
        InputSegment(deliberation="d", from_tick=10, to_tick=5, actions=["teleport"])])
    issues = validate_filmstrip_plan(bad, actions)
    assert len(issues) >= 3 and any("teleport" in i for i in issues)
    assert validate_filmstrip_plan(FilmstripPlan(deliberation="d", capture_ticks=[]), actions)


def test_plan_filmstrip_retries_with_feedback(tmp_path):
    answers = iter([plan_json(segs=[(1, 10, ["teleport"])]), plan_json(segs=[(1, 10, ["dive"])])])
    ctx, _ = code_ctx(tmp_path, lambda c: next(answers))
    plan = plan_filmstrip(ctx, code_item())
    assert plan.input_segments[0].actions == ["dive"]
    assert "FAILED VALIDATION" in ctx.client.models.prompts[1] and "teleport" in ctx.client.models.prompts[1]


def test_code_worker_builds_verifies_and_proves_with_scripted_filmstrip(tmp_path):
    verified = []
    swe = FakeSwe()
    ctx, calls = code_ctx(tmp_path, lambda c: plan_json(ticks=(5, 25, 45), segs=[(1, 45, ["dive"])]), swe=swe,
                          verify=lambda d, t: verified.append(t.id))
    result = code_item_handler(ctx)(code_item())
    track = swe.calls[0]
    assert isinstance(track, RequirementTrack) and track.id == "dive" and track.target_modules == ["mechanics.gd"]
    assert "skiff dives then recovers" in track.description and track.acceptance_criteria[0].description == "skiff dives then recovers"
    assert verified == ["dive"]
    assert calls["strip"] == [([5, 25, 45], [{"from": 1, "to": 45, "actions": ["dive"]}])]
    assert Path(result.proof_images[0]).is_file() and "Filmstrip ticks [5, 25, 45]" in result.notes


def test_code_worker_restores_touched_modules_on_failure_and_on_failed_verification(tmp_path):
    (tmp_path / "mechanics.gd").write_text("# original\n")
    ctx, _ = code_ctx(tmp_path, lambda c: plan_json(), swe=FakeSwe(fail=True))
    with pytest.raises(RuntimeError):
        code_item_handler(ctx)(code_item())
    assert (tmp_path / "mechanics.gd").read_text() == "# original\n"

    def failing_verify(d, t):
        raise StudioAgentError("smoke check failed: parse error")

    ctx, _ = code_ctx(tmp_path, lambda c: plan_json(), swe=FakeSwe(), verify=failing_verify)
    with pytest.raises(StudioAgentError):
        code_item_handler(ctx)(code_item(modules=["mechanics.gd", "brand_new.gd"]))
    assert (tmp_path / "mechanics.gd").read_text() == "# original\n"
    assert not (tmp_path / "brand_new.gd").exists()                        # files created by the failed attempt are removed


def test_code_worker_passes_council_feedback_history_to_the_swe_agent(tmp_path):
    from engine.studio.models import RevisionRecord
    swe = FakeSwe()
    ctx, _ = code_ctx(tmp_path, lambda c: plan_json(), swe=swe)
    code_item_handler(ctx)(code_item(revisions=[RevisionRecord(revision_index=0, feedback="dive is too slow", verdict="request_changes")]))
    assert "dive is too slow" in swe.calls[0].description


def test_code_workers_are_serialised_but_asset_workers_stay_parallel(tmp_path):
    swe = FakeSwe()
    ctx, _ = code_ctx(tmp_path, lambda c: plan_json(), swe=swe)
    handlers = default_handlers(ctx)
    board = ProductionBoard()
    props = [WorkItemProposal(deliberation="d", item_id=f"code{i}", kind="mechanic_code", title="t", spec="s", proof_spec="p",
                              target_modules=[f"m{i}.gd"]) for i in range(3)]
    board.add_items(props, "d")
    sched = BoardScheduler(board, handlers, max_workers=3, poll_interval=0.005)
    try:
        assert sched.run_until_idle(20)
    finally:
        sched.stop()
    assert swe.peak == 1                                                    # whole-project verification gates never overlap
    assert all(board.get(f"code{i}").state == "awaiting_review" for i in range(3))


# ----------------------------------------------------------------------------- orchestrator wiring
def make_orchestrator(**cfg):
    return StudioOrchestrator(api_key="test-key", client=SimpleNamespace(models=SimpleNamespace()),
                              config_obj=StudioOrchestratorConfig(gemini_api_key="test-key", **cfg))


def test_board_is_disabled_for_non_live_clients_unless_forced(tmp_path):
    orch = make_orchestrator()
    assert orch._continuous_production_enabled() is False and orch._build_production_board(tmp_path) is None
    orch = make_orchestrator(continuous_production=True, board_workers=2, max_item_revisions=3)
    board = orch._build_production_board(tmp_path)
    assert board is not None and board.default_max_revisions == 3 and orch._board_scheduler.max_workers == 2
    assert set(orch._board_scheduler.handlers) == {
        "visual_asset", "animation", "audio_asset", "level_map", "mechanic_code", "ui", "integration"}
    assert orch.production_board is board


def test_board_events_are_published_through_the_orchestrator(tmp_path):
    orch = make_orchestrator(continuous_production=True)
    published = []
    orch._publish_event = lambda et, agent, delib, payload=None: published.append((et, agent))
    board = orch._build_production_board(tmp_path)
    board.add_items([WorkItemProposal(deliberation="d", item_id="a", kind="visual_asset", title="a", spec="s", proof_spec="p")], "Visual")
    assert ("backlog_item_created", "ProductionBoard") in published


def test_requirements_are_reconciled_from_the_board_and_assets_not_rebuilt(tmp_path):
    from engine.studio.models import GameRequirementsSpec, ReviewDecision
    from engine.studio.production_board import WorkerResult
    orch = make_orchestrator(continuous_production=True)
    board = orch._build_production_board(tmp_path)
    orch._continuous_active = True
    board.add_items([
        WorkItemProposal(deliberation="d", item_id="dive", kind="mechanic_code", title="Dive", spec="s", proof_spec="p", target_modules=["mechanics.gd"]),
        WorkItemProposal(deliberation="d", item_id="sky", kind="visual_asset", title="Sky", spec="s", proof_spec="p"),
        WorkItemProposal(deliberation="d", item_id="hud", kind="ui", title="HUD", spec="s", proof_spec="p"),
    ], "Gameplay")
    board.claim_ready(limit=5)
    for i in ("dive", "sky", "hud"):
        board.submit_result(i, WorkerResult())
        board.apply_review(ReviewDecision(deliberation="ok", item_id=i, verdict="approve"))
    legacy = GameRequirementsSpec(deliberation="legacy", game_title="G", requirements=[
        RequirementTrack(deliberation="d", id="R1", title="legacy", description="legacy", status="pending")],
        asset_manifest=[])
    out = orch._reconcile_requirements_with_board(legacy)
    assert [t.id for t in out.requirements] == ["dive", "hud"]            # legacy tracks replaced; assets are not code tracks
    assert all(t.status == "completed" for t in out.requirements)
    assert out.asset_manifest == []
    orch._continuous_active = False
    assert orch._reconcile_requirements_with_board(legacy) is legacy       # untouched when continuous mode is off


# ----------------------------------------------------------------------------- end to end
def test_end_to_end_council_designs_workers_build_council_reviews_and_game_locks(tmp_path):
    adds = {2: [
        WorkItemProposal(deliberation="d", item_id="skiff", kind="visual_asset", title="Skiff", spec="the hero skiff", proof_spec="side view"),
        WorkItemProposal(deliberation="d", item_id="world", kind="level_map", title="World", spec="three regions", proof_spec="top-down map"),
        WorkItemProposal(deliberation="d", item_id="flight", kind="mechanic_code", title="Flight", spec="dive and soar", proof_spec="dive filmstrip",
                         depends_on=["skiff"], target_modules=["mechanics.gd"]),
    ]}
    first_review = {"done": set()}

    def policy(item_id, awaiting, text):
        from engine.studio.models import ReviewDecision
        if item_id == "flight" and "flight" not in first_review["done"]:
            first_review["done"].add("flight")
            return ReviewDecision(deliberation="dive too weak", item_id=item_id, feedback="make the dive more dramatic", verdict="request_changes")
        return ReviewDecision(deliberation="good", item_id=item_id, verdict="approve")

    def responder(contents):
        text = contents if isinstance(contents, str) else "\n".join(c for c in contents if isinstance(c, str))
        if "AssetRequirement" in text or "Visual/Audio Production Lead" in text:
            return bw.req_json("skiff")
        return plan_json(ticks=(5, 30), segs=[(1, 30, ["dive"])])

    worker_client = bw.make_client(responder)

    class FakeLayoutSwe(FakeSwe):
        def formulate_level_layout(self, c_dir, force_refresh=False, guidance=""):
            import tests.test_board_proofs as bp
            return bp.make_layout()

    room, client, board, sched, candidate, order = cb.build(tmp_path, adds, policy)
    (candidate / "game_bible.json").write_text(json.dumps({"game_title": "E2E"}))
    (candidate / "project.godot").write_text("[input]\ndive={}\n")
    ctx = WorkerContext(cartridge_dir=candidate, client=worker_client, swe_agent=FakeLayoutSwe(), library_factory=bw.FakeLibrary,
                        filmstrip_fn=lambda d, ticks, out_path, input_script=None: (Image.new("RGB", (20, 8)).save(out_path), Path(out_path))[1])
    sched.handlers.update(default_handlers(ctx))
    try:
        bible = room.run_writers_room(dna=cb.base._sample_dna(), candidate_dir=candidate)
    finally:
        sched.stop()
    assert bible.status == "locked" and board.is_drained()
    flight = board.get("flight")
    assert len(flight.revisions) == 2 and flight.revisions[0].feedback == "make the dive more dramatic"
    assert "make the dive more dramatic" in ctx.swe_agent.calls[1].description        # the revision saw the council's feedback
    assert board.get("world").revisions and Path(board.get("world").revisions[0].proof_paths[0]).name.startswith("level_map")
    kinds = {i.item_id: i.kind for i in board.items()}
    assert kinds == {"skiff": "visual_asset", "world": "level_map", "flight": "mechanic_code"}


def test_available_input_actions_discovers_gdscript_actions_and_key_constants(tmp_path):
    (tmp_path / "project.godot").write_text("[input]\nbark={}\n")
    (tmp_path / "main.gd").write_text(
        'extends Node2D\nfunc _input(event):\n\tif event.keycode == KEY_SPACE or event.keycode == KEY_3:\n\t\tpass\n'
        'func _process(_d):\n\tif Input.is_action_pressed("tether_attach"):\n\t\tpass\n'
    )
    acts = available_input_actions(tmp_path)
    assert {"bark", "KEY_SPACE", "KEY_3", "tether_attach", "ui_accept"} <= set(acts)


def test_validate_filmstrip_plan_checks_setup_calls(tmp_path):
    from engine.studio.models import QASetupCall
    acts = ["ui_right", "KEY_SPACE"]
    bad_plan = FilmstripPlan(
        deliberation="d",
        capture_ticks=[5, 20],
        setup_calls=[
            QASetupCall(deliberation="d", at_tick=0, method_name="", args_json="not-json"),
        ],
    )
    issues = validate_filmstrip_plan(bad_plan, acts)
    assert len(issues) == 3
    good_plan = FilmstripPlan(
        deliberation="d",
        capture_ticks=[5, 20],
        setup_calls=[
            QASetupCall(deliberation="d", at_tick=5, target_node_or_property="world", method_name="spawn_hazard", args_json="[1]"),
        ],
    )
    assert validate_filmstrip_plan(good_plan, acts) == []


def test_godot_gameplay_and_asset_gates_reject_demo_loops_debug_huds_and_unbound_sprites(tmp_path):
    track = RequirementTrack(deliberation="d", id="t1", title="T1", description="desc", target_modules=["main.gd", "entities.gd", "ui_hud.gd"])

    # 1. Self-playing sim_tick demo loop in main.gd is rejected
    (tmp_path / "main.gd").write_text(
        "extends Node2D\nvar sim_tick = 0\nfunc _physics_process(_d):\n"
        "\tsim_tick += 1\n\tif sim_tick == 62:\n\t\tpass\n\telif sim_tick == 130:\n\t\tpass\n"
    )
    with pytest.raises(RuntimeError, match="Self-playing demo loop detected"):
        StudioOrchestrator._check_godot_gameplay_and_asset_gates(tmp_path, track, {})
    (tmp_path / "main.gd").write_text("extends Node2D\n")

    # 2. Debug telemetry dashboard in ui_hud.gd is rejected
    (tmp_path / "ui_hud.gd").write_text(
        'extends CanvasLayer\nvar label = "HEADLESS MECHANICS & TELEMETRY VERIFICATION: ALL PASS"\n'
    )
    with pytest.raises(RuntimeError, match="Debug telemetry dashboard detected in ui_hud.gd"):
        StudioOrchestrator._check_godot_gameplay_and_asset_gates(tmp_path, track, {})
    (tmp_path / "ui_hud.gd").write_text("extends CanvasLayer\n")

    # 3. Unbound sprite assets (PNGs exist in assets/sprites/ but textures_loaded_count == 0) is rejected
    sprites_dir = tmp_path / "assets" / "sprites"
    sprites_dir.mkdir(parents=True)
    Image.new("RGBA", (32, 32), (255, 0, 0, 255)).save(sprites_dir / "hero.png")
    (tmp_path / "entities.gd").write_text("extends Node2D\n")
    with pytest.raises(RuntimeError, match="Unbound sprite assets"):
        StudioOrchestrator._check_godot_gameplay_and_asset_gates(tmp_path, track, {"textures_loaded_count": 0})

    # Passes once textures_loaded_count > 0
    StudioOrchestrator._check_godot_gameplay_and_asset_gates(tmp_path, track, {"textures_loaded_count": 2})

    # 4. playtest_timer / handle_automated_playtest_cadence in main.gd is rejected
    (tmp_path / "main.gd").write_text(
        "extends Node2D\nvar playtest_timer: float = 0.0\nfunc handle_automated_playtest_cadence(delta: float):\n"
        "\tif playtest_timer >= 1.0:\n\t\tpass\n\telif playtest_timer >= 2.8:\n\t\tpass\n"
    )
    with pytest.raises(RuntimeError, match="Self-playing demo loop detected"):
        StudioOrchestrator._check_godot_gameplay_and_asset_gates(tmp_path, track, {"textures_loaded_count": 2})
    (tmp_path / "main.gd").write_text("extends Node2D\n")

    # 5. Tiny 1-screen hardcoded Rect2 placeholder in scenes.gd when level_layout.json has multi-screen scale is rejected
    scene_track = RequirementTrack(deliberation="d", id="t2", title="T2", description="desc", target_modules=["scenes.gd"])
    (tmp_path / "level_layout.json").write_text(
        json.dumps(
            {
                "deliberation": "d",
                "viewport_size": [1920.0, 1080.0],
                "intended_world_scale_in_viewports": 12.0,
                "world_bounds": [0.0, 0.0, 11520.0, 2160.0],
                "world_regions": [
                    {"deliberation": "d", "region_id": "r0", "bounds": [0.0, 0.0, 3840.0, 1080.0], "progression_order": 0},
                    {"deliberation": "d", "region_id": "r1", "bounds": [3840.0, 0.0, 7680.0, 2160.0], "progression_order": 1},
                ],
                "camera_framing": "tracking",
                "entities": [],
                "triggers": [],
            }
        )
    )
    (tmp_path / "scenes.gd").write_text(
        "extends Node2D\nfunc build():\n\tvar r = [Rect2(100, 520, 480, 60), Rect2(40, 680, 1200, 40)]\n"
    )
    with pytest.raises(RuntimeError, match="Tiny placeholder level detected in scenes.gd"):
        StudioOrchestrator._check_godot_gameplay_and_asset_gates(tmp_path, scene_track, {"textures_loaded_count": 2})


def test_level_generation_strategy_accepts_bespoke_open_ended_strategies():
    from engine.studio.models import (
        LevelArchetypeSpec,
        LevelGenerationStrategy,
        LevelLayoutSpec,
        LevelRegionSpec,
        validate_level_layout,
    )

    # 1. Bespoke single-screen arena strategy with open-ended generation_approach
    arena_strategy = LevelGenerationStrategy(
        deliberation="Fighting game arena requires a single-screen stage with reactive hazards.",
        generation_approach="symmetrical_combat_stage_with_reactive_hazards",
        world_structure="Single 1280x720 combat stage.",
        runtime_construction_plan="Direct instantiation of stage floor, walls, and breakable pillars in scenes.gd.",
        level_archetypes=[
            LevelArchetypeSpec(
                deliberation="d",
                archetype_id="dojo_arena",
                purpose="Combat",
                visual_identity="Neon dojo",
                signature_challenge="Footsies",
            )
        ],
        generation_rules=["Symmetrical spawn points"],
        difficulty_and_pacing_curve="Round-based",
    )
    arena_layout = LevelLayoutSpec(
        deliberation="d",
        viewport_size=(1280.0, 720.0),
        intended_world_scale_in_viewports=1.0,
        world_bounds=(0.0, 0.0, 1280.0, 720.0),
        world_regions=[
            LevelRegionSpec(deliberation="d", region_id="dojo", bounds=(0.0, 0.0, 1280.0, 720.0), progression_order=0)
        ],
        camera_framing="Fixed 16:9 stage camera",
    )
    assert validate_level_layout(arena_layout, strategy=arena_strategy) == []

    # 2. Bespoke procedural grammar strategy with per-region chunk grammar
    world_strategy = LevelGenerationStrategy(
        deliberation="Osteon-cluster grammar with kinematic trajectory validation.",
        generation_approach="osteon_cluster_grammar_with_trajectory_validation",
        world_structure="3 descending strata stitched by procedural connective tissue.",
        runtime_construction_plan="Assemble osteon chunks in scenes.gd using procedural_chunk_grammar per region.",
        level_archetypes=[
            LevelArchetypeSpec(deliberation="d", archetype_id="biome_a", purpose="p", visual_identity="v", signature_challenge="c"),
        ],
        generation_rules=["Reachable anchors"],
        difficulty_and_pacing_curve="Escalating",
    )
    multi_layout = LevelLayoutSpec(
        deliberation="d",
        viewport_size=(1280.0, 720.0),
        intended_world_scale_in_viewports=4.0,
        world_bounds=(0.0, 0.0, 2560.0, 1440.0),
        world_regions=[
            LevelRegionSpec(
                deliberation="d",
                region_id="strata_0",
                bounds=(0.0, 0.0, 2560.0, 1440.0),
                progression_order=0,
                archetype_id="biome_a",
                procedural_chunk_grammar="Generate 8 connected calcium shelves with <= 1.85x tether spacing",
                target_chunk_or_feature_count=8,
            )
        ],
        camera_framing="Scrolling",
    )
    assert validate_level_layout(multi_layout, strategy=world_strategy) == []


def test_level_map_handler_realizes_scenes_gd_and_captures_viewport_filmstrip(tmp_path):
    import tests.test_board_proofs as bp
    from engine.studio.models import WorkItem
    from engine.studio.workers import level_map_handler

    class FakeLayoutSwe(FakeSwe):
        def formulate_level_layout(self, c_dir, force_refresh=False, guidance=""):
            return bp.make_layout()

    plan = FilmstripPlan(deliberation="tour regions", capture_ticks=[1, 30, 60], frame_expectations=["r0", "r1", "r2"])
    swe = FakeLayoutSwe()
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=bw.make_client(lambda _c: plan.model_dump_json()),
        swe_agent=swe,
        filmstrip_fn=lambda d, ticks, out_path, input_script=None: (Image.new("RGB", (20, 8)).save(out_path), Path(out_path))[1],
    )
    wi = WorkItem(
        deliberation="d",
        item_id="level_strata_0",
        kind="level_map",
        title="Strata 0",
        spec="Build multi-region descent in scenes.gd",
        proof_spec="Show regions in Godot",
        target_modules=["scenes.gd", "level_layout.json"],
    )
    res = level_map_handler(ctx)(wi)
    assert len(swe.calls) == 1
    assert "scenes.gd" in swe.calls[0].target_modules
    assert len(res.proof_images) == 2
    assert Path(res.proof_images[0]).name.startswith("level_map_")
    assert Path(res.proof_images[1]).name.startswith("level_filmstrip_")


def _lead_json(
    action: str,
    strategy: str = "Execute multi-specialist plan",
    specialist_agent: str = "",
    brief: str = "",
    target_paths=None,
    submission_summary: str = "",
    selected_proof_paths=None,
) -> str:
    from engine.studio.models import TaskLeadDecision
    return TaskLeadDecision(
        deliberation=f"Deliberating for action={action} specialist={specialist_agent}",
        current_strategy=strategy,
        specialist_agent=specialist_agent,
        brief=brief,
        target_paths=list(target_paths or []),
        submission_summary=submission_summary,
        selected_proof_paths=list(selected_proof_paths or []),
        action=action,
    ).model_dump_json()


def test_task_lead_iteratively_delegates_to_visual_swe_and_qa_before_submitting(tmp_path):
    from engine.studio.workers import task_lead_handler

    (tmp_path / "game_bible.json").write_text(json.dumps({"game_title": "GrappleQuest"}))
    (tmp_path / "project.godot").write_text("[input]\ndive={}\n")

    lead_turns = iter([
        _lead_json(
            "delegate",
            strategy="Fabricate animated grapple claw sprite sheet first, then wire into mechanics.gd, then run QA filmstrip.",
            specialist_agent="visual_artist",
            brief="Create an animated 4-frame grapple claw sprite sheet.",
            target_paths=["assets/sprites/grapple_claw.png"],
        ),
        _lead_json(
            "delegate",
            strategy="Now that grapple_claw.png is in the catalog, implement grapple physics and bind the sprite in mechanics.gd.",
            specialist_agent="swe_engineer",
            brief="Implement grapple tether launch and bind res://assets/sprites/grapple_claw.png in mechanics.gd.",
            target_paths=["mechanics.gd"],
        ),
        _lead_json(
            "delegate",
            strategy="Verify grapple launch in headless Godot across ticks 5, 25, 45.",
            specialist_agent="qa_test_engineer",
            brief="Trigger dive/grapple from tick 1 to 45 and capture frames at 5, 25, 45.",
        ),
        _lead_json(
            "submit_to_council",
            strategy="All visual, code, and QA steps verified; submitting combined proofs to the Council.",
            submission_summary="Fabricated animated grapple_claw sprite sheet, bound it in mechanics.gd, and verified trajectory in QA filmstrip.",
        ),
    ])

    def responder(contents):
        text = contents if isinstance(contents, str) else "\n".join(c for c in contents if isinstance(c, str))
        if "Task Lead Strategist" in text:
            return next(lead_turns)
        if "Visual/Audio Production Lead" in text:
            # Visual artist chooses an animated sprite_sheet mode even though the board item is mechanic_code!
            return bw.req_json("grapple_claw", mode="sprite_sheet", asset_type="sprite_sheet", target="assets/sprites/grapple_claw.png")
        return plan_json(ticks=(5, 25, 45), segs=[(1, 45, ["dive"])])

    swe = FakeSwe()
    events = []
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=bw.make_client(responder),
        swe_agent=swe,
        library_factory=bw.FakeLibrary,
        filmstrip_fn=lambda d, ticks, out_path, input_script=None: (Image.new("RGB", (24, 8)).save(out_path), Path(out_path))[1],
        event_callback=lambda et, payload: events.append((et, payload)),
    )
    wi = code_item(item_id="grapple_tether", modules=["mechanics.gd"])
    result = task_lead_handler(ctx)(wi)

    # 1. Visual artist created the animated sprite sheet and contact plate in the catalog
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert "grapple_claw" in catalog and catalog["grapple_claw"]["mode"] == "sprite_sheet"
    # 2. SWE engineer received the Task Lead's brief
    assert len(swe.calls) == 1
    assert "res://assets/sprites/grapple_claw.png" in swe.calls[0].description
    # 3. Combined proofs include both the visual plates and the gameplay filmstrip
    proof_names = {Path(p).name for p in result.proof_images}
    assert "anim_sheet_grapple_claw.png" in proof_names
    assert "filmstrip_grapple_tether_rev0.png" in proof_names
    # 4. Trace file and telemetry events recorded all 4 Task Lead turns
    trace_file = tmp_path / "artifacts" / "work_items" / "grapple_tether" / "lead_trace_rev0.json"
    assert trace_file.is_file()
    trace = json.loads(trace_file.read_text())
    assert [t["specialist_agent"] for t in trace["turns"]] == ["visual_artist", "swe_engineer", "qa_test_engineer"]
    assert len([e for e in events if e[0] == "backlog_item_lead_turn"]) == 4


def test_task_lead_self_corrects_within_same_revision_after_inspecting_specialist_report(tmp_path):
    from engine.studio.workers import task_lead_handler

    (tmp_path / "game_bible.json").write_text(json.dumps({"game_title": "SelfCorrectGame"}))
    (tmp_path / "project.godot").write_text("[input]\ndive={}\n")

    lead_turns = iter([
        _lead_json("delegate", strategy="Initial implementation pass", specialist_agent="swe_engineer", brief="Implement initial dive impulse", target_paths=["mechanics.gd"]),
        _lead_json("delegate", strategy="Run initial QA test", specialist_agent="qa_test_engineer", brief="Test dive at ticks 5, 20"),
        _lead_json("delegate", strategy="QA report inspected; refine impulse in mechanics.gd before bothering Council", specialist_agent="swe_engineer", brief="Double dive velocity so trajectory clears the gap", target_paths=["mechanics.gd"]),
        _lead_json("delegate", strategy="Re-run QA filmstrip after velocity fix", specialist_agent="qa_test_engineer", brief="Re-verify dive trajectory across ticks 5, 25, 45"),
        _lead_json("submit_to_council", strategy="Self-correction verified; submit to Council", submission_summary="Refined dive velocity after initial QA check."),
    ])

    def responder(contents):
        text = contents if isinstance(contents, str) else "\n".join(c for c in contents if isinstance(c, str))
        if "Task Lead Strategist" in text:
            return next(lead_turns)
        return plan_json(ticks=(5, 25, 45), segs=[(1, 45, ["dive"])])

    swe = FakeSwe()
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=bw.make_client(responder),
        swe_agent=swe,
        filmstrip_fn=lambda d, ticks, out_path, input_script=None: (Image.new("RGB", (24, 8)).save(out_path), Path(out_path))[1],
    )
    result = task_lead_handler(ctx)(code_item())
    assert len(swe.calls) == 2
    assert "Double dive velocity" in swe.calls[1].description
    assert "Refined dive velocity" in result.notes


def test_task_lead_restrategizes_and_pivots_on_council_request_changes(tmp_path):
    from engine.studio.models import RevisionRecord
    from engine.studio.workers import task_lead_handler

    (tmp_path / "game_bible.json").write_text(json.dumps({"game_title": "PivotGame"}))
    (tmp_path / "project.godot").write_text("[input]\ndive={}\n")

    # Seed revision 0 trace file as if rev 0 only used swe_engineer + qa_test_engineer
    item_dir = tmp_path / "artifacts" / "work_items" / "dive"
    item_dir.mkdir(parents=True, exist_ok=True)
    (item_dir / "lead_trace_rev0.json").write_text(
        json.dumps({"item_id": "dive", "revision": 0, "current_strategy": "Code-only implementation", "turns": []})
    )

    lead_turns = iter([
        _lead_json(
            "delegate",
            strategy="Pivoting per Council feedback: outsource dive whoosh SFX to audio_designer and custom skiff sprite to visual_artist before updating mechanics.gd.",
            specialist_agent="audio_designer",
            brief="Synthesize a sharp aerodynamic dive whoosh SFX.",
            target_paths=["assets/audio/dive_whoosh.wav"],
        ),
        _lead_json(
            "delegate",
            strategy="Now fabricate the custom diving skiff sprite.",
            specialist_agent="visual_artist",
            brief="Create a sleek diving skiff sprite.",
            target_paths=["assets/sprites/diving_skiff.png"],
        ),
        _lead_json(
            "delegate",
            strategy="Bind the new sprite and SFX in mechanics.gd.",
            specialist_agent="swe_engineer",
            brief="Bind diving_skiff.png and dive_whoosh.wav in mechanics.gd.",
            target_paths=["mechanics.gd"],
        ),
        _lead_json(
            "delegate",
            strategy="Capture updated gameplay filmstrip.",
            specialist_agent="qa_test_engineer",
            brief="Capture dive with new sprite and audio trigger.",
        ),
        _lead_json(
            "submit_to_council",
            strategy="Pivot complete: audio, visual, code, and QA proofs ready.",
            submission_summary="Pivoted to fabricate dive_whoosh.wav and diving_skiff.png and bound them in mechanics.gd.",
        ),
    ])

    lead_prompts = []

    def responder(contents):
        text = contents if isinstance(contents, str) else "\n".join(c for c in contents if isinstance(c, str))
        if "Task Lead Strategist" in text:
            lead_prompts.append(text)
            return next(lead_turns)
        if "Visual/Audio Production Lead" in text:
            if "audio_asset" in text:
                return bw.req_json("dive_whoosh", mode="sfx", asset_type="sfx", target="assets/audio/dive_whoosh.wav")
            return bw.req_json("diving_skiff", mode="static_sprite", target="assets/sprites/diving_skiff.png")
        return plan_json(ticks=(5, 30), segs=[(1, 30, ["dive"])])

    swe = FakeSwe()
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=bw.make_client(responder),
        voice_agent=object(),
        swe_agent=swe,
        library_factory=bw.FakeLibrary,
        filmstrip_fn=lambda d, ticks, out_path, input_script=None: (Image.new("RGB", (24, 8)).save(out_path), Path(out_path))[1],
    )
    rev0 = RevisionRecord(
        revision_index=0,
        worker_notes="Strategy: Code-only implementation",
        feedback="[VisualArtDirector]: Missing custom diving skiff sprite | [SoundDesigner]: Add aerodynamic dive whoosh SFX",
        verdict="request_changes",
    )
    wi = code_item(revisions=[rev0])
    result = task_lead_handler(ctx)(wi)

    # Verify the Task Lead saw both the Council feedback and the prior revision trace on Turn 0
    assert "Missing custom diving skiff sprite" in lead_prompts[0]
    assert "PREVIOUS REVISION STRATEGY & SPECIALIST TRACE" in lead_prompts[0]
    # Verify all specialists contributed proofs
    proof_names = {Path(p).name for p in result.proof_images}
    assert any("waveform" in n for n in proof_names)
    assert "plate_diving_skiff.png" in proof_names
    assert "filmstrip_dive_rev1.png" in proof_names


def test_validate_task_lead_decision_blocks_premature_submit_and_unknown_specialists(tmp_path):
    from engine.studio.workers import plan_task_lead_turn, validate_task_lead_decision

    # 1. Direct structural validation checks
    premature = json.loads(_lead_json("submit_to_council", strategy="Submit immediately"))
    from engine.studio.models import TaskLeadDecision
    issues = validate_task_lead_decision(TaskLeadDecision(**premature), ["visual_artist", "swe_engineer"], [])
    assert any("Cannot submit_to_council before any proof images" in i for i in issues)

    bad_spec = TaskLeadDecision(**json.loads(_lead_json("delegate", strategy="s", specialist_agent="nonexistent_agent", brief="b")))
    issues2 = validate_task_lead_decision(bad_spec, ["visual_artist"], [])
    assert any("not available" in i for i in issues2)

    # 2. Retry loop feeds validation error back to the Task Lead and succeeds on valid retry
    answers = iter([
        _lead_json("submit_to_council", strategy="Try submitting without proof"),
        _lead_json("delegate", strategy="Delegate to visual_artist first", specialist_agent="visual_artist", brief="Make hero sprite"),
    ])
    ctx = bw.ctx_for(tmp_path, lambda _c: next(answers))
    decision = plan_task_lead_turn(ctx, bw.item("hero"), turn_history=[], collected_proofs=[])
    assert decision.action == "delegate" and decision.specialist_agent == "visual_artist"
    assert "FAILED VALIDATION" in ctx.client.models.prompts[1]


def test_open_ended_strategies_and_dynamic_council_reviewers():
    from engine.studio.models import (
        AnimationStateSpec,
        GoalSpec,
        RenderingStrategySpec,
        RevisionRecord,
        SubjectRenderTreatment,
        WorkItem,
    )
    from engine.studio.writers_room import CreativeWritersRoom
    import inspect
    from engine.studio import swe

    # 1. Bespoke rendering paradigms and subject techniques are accepted without pre-baked Literal menus
    rs = RenderingStrategySpec(
        deliberation="Inventing a bespoke radial radar CRT phosphor vector strategy.",
        rendering_paradigm="radial_sonar_crt_phosphor_sweep",
        renderer="compatibility",
        treatments=[
            SubjectRenderTreatment(
                deliberation="Phosphor blips match sonar command deck.",
                subject_id="sonar_contacts",
                technique="phosphor_persistence_blip_shader",
                target_resolution=(128, 128),
                notes="Decays over 3 seconds after sweep.",
            )
        ],
    )
    assert rs.rendering_paradigm == "radial_sonar_crt_phosphor_sweep"
    assert rs.treatments[0].technique == "phosphor_persistence_blip_shader"

    # 2. Bespoke animation motion profiles and goal horizons are accepted without pre-baked Literal menus
    anim = AnimationStateSpec(
        deliberation="Turret unfolds via hydraulic struts.",
        state_name="transform_turret",
        frame_count=6,
        fps=12.0,
        loop=False,
        motion_profile="hydraulic_deployment_unfold",
    )
    assert anim.motion_profile == "hydraulic_deployment_unfold"

    goal = GoalSpec(
        deliberation="Long-term orbital federation objective.",
        goal_id="unite_habitats",
        horizon="epochal_dynasty_arc",
        description="Unite the orbital habitats.",
    )
    assert goal.horizon == "epochal_dynasty_arc"

    # 3. Council review dynamically includes reviewers for all specialists engaged by the Task Lead
    wi = WorkItem(
        deliberation="Task requires code, sprite, and SFX.",
        item_id="task_1",
        title="Implement thruster ignition & sound",
        kind="mechanic_code",
        spec="Add thruster mechanics, custom exhaust plume sprite, and ignition SFX.",
        raised_by_role="GameplaySystemsEngineer",
        revisions=[
            RevisionRecord(
                deliberation="Executed multi-specialist strategy.",
                revision_index=0,
                worker_notes=(
                    "Task Lead Strategy: Full audiovisual + code pass\n"
                    "[Turn 1 - visual_artist]: Fabricated plume sprite\n"
                    "[Turn 2 - audio_designer]: Synthesized ignition SFX\n"
                    "[Turn 3 - swe_engineer]: Implemented thruster physics"
                ),
            )
        ],
    )
    reviewer_roles = CreativeWritersRoom._reviewers_for(wi)
    assert "VisualArtDirector" in reviewer_roles
    assert "AudioAtmosphereDesigner" in reviewer_roles
    assert "GameplayDesigner" in reviewer_roles

    # 4. SWE prompt is free of overfitted past-game references (Rule 1)
    swe_src = inspect.getsource(swe.LeadGameplaySWEAgent.implement_requirement_track)
    assert "canal_grid" not in swe_src
    assert "shelves: Array[Rect2]" not in swe_src


def test_new_game_never_leaks_previous_candidate_board_or_assets(tmp_path):
    from starlette.testclient import TestClient
    from engine.ui.server import SwarmEventBus, create_app

    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    old_cand = cartridges_root / "candidate_genesis_old111"
    new_cand = cartridges_root / "candidate_genesis_new222"

    # Populate old candidate & active cartridge with previous game's board, sign-off report, and assets
    (active_dir / "assets" / "sprites").mkdir(parents=True)
    (active_dir / "assets" / "sprites" / "old_hero.png").write_bytes(b"png")
    (active_dir / "manifest.json").write_text(json.dumps({"title": "Old Game"}), encoding="utf-8")
    (active_dir / "world_spec.json").write_text(json.dumps({"game_title": "Old Game"}), encoding="utf-8")

    old_cand.mkdir(parents=True)
    (old_cand / "backlog.json").write_text(
        json.dumps(
            {
                "design_revision": 2,
                "items": [
                    {
                        "deliberation": "Old game item",
                        "item_id": "old_item_1",
                        "title": "Old Game Tether Solver",
                        "kind": "mechanic_code",
                        "spec": "Old spec",
                        "state": "approved",
                        "revisions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (old_cand / "sign_off_report.json").write_text(
        json.dumps({"candidate_id": "candidate_genesis_old111", "unanimous_approval": True, "verdicts": []}),
        encoding="utf-8",
    )

    # Start a new game in candidate_genesis_new222 (still brainstorming, no backlog.json yet)
    new_cand.mkdir(parents=True)
    new_bible = {
        "deliberation": "Starting new game",
        "bible_id": "bible_new222",
        "cartridge_id": "candidate_genesis_new222",
        "status": "brainstorming",
        "game_title": "Untitled Concept",
        "core_genre": "Simulation",
        "writers_room_transcript": [],
    }
    (new_cand / "game_bible.json").write_text(json.dumps(new_bible), encoding="utf-8")

    bus = SwarmEventBus(db_path=tmp_path / "telemetry.db")
    bus.reset_for_new_genesis_build(build_id="build_new222")
    bus.publish_threadsafe({"event_type": "game_bible_updated", "payload": new_bible})

    app = create_app(
        db_path=tmp_path / "telemetry.db",
        cartridges_root=cartridges_root,
        active_cartridge_dir=active_dir,
        event_bus=bus,
    )
    client = TestClient(app)

    # 1. /api/studio/game-bible must return an empty production_board for candidate_genesis_new222 (NOT old_item_1)
    gb_resp = client.get("/api/studio/game-bible").json()
    assert gb_resp["cartridge_id"] == "candidate_genesis_new222"
    assert gb_resp["active_candidate_id"] == "candidate_genesis_new222"
    assert gb_resp["production_board"]["cartridge_id"] == "candidate_genesis_new222"
    assert gb_resp["production_board"]["items"] == []
    assert gb_resp["sign_off_report"] is None

    # 2. /api/requirements must return 0 tracks (NOT old_item_1)
    req_resp = client.get("/api/requirements").json()
    assert req_resp["total_tracks"] == 0
    assert req_resp["requirements"] == []

    # 3. /api/artifacts must return 0 sprites (NOT old_hero.png from active)
    art_resp = client.get("/api/artifacts").json()
    assert art_resp["sprites"] == []

    # 4. /api/agents/history must return 0 tasks (NOT hydrated from active)
    hist_resp = client.get("/api/agents/history").json()
    assert hist_resp["total_tasks"] == 0
    assert hist_resp["tasks"] == []


def test_live_executive_producer_steering_is_preserved_and_consumed(tmp_path):
    from engine.studio.models import GameBibleSpec
    from engine.studio.writers_room import CreativeWritersRoom

    room = CreativeWritersRoom(client=None)
    cand_dir = tmp_path / "candidate_genesis_steer1"
    cand_dir.mkdir(parents=True)

    bible = GameBibleSpec(
        deliberation="Initial high-action concept.",
        bible_id="bible_steer1",
        cartridge_id="candidate_genesis_steer1",
        game_title="Action Prototype",
        core_genre="Action",
    )
    room._persist_bible(bible, candidate_dir=cand_dir)

    # Simulate Mission Control writing steering to disk while a specialist turn is in flight
    steering_msg = "I don't want too much action. I like a good casual game where I can let it run while I get a coffee if I want"
    room.enqueue_user_steering(steering_msg, candidate_dir=cand_dir)
    assert (cand_dir / ".pending_steering.json").is_file()

    # Consume steering and verify it records the ExecutiveProducer utterance, ConceptEvolutionEntry, and high-priority snapshot banner
    consumed = room._consume_pending_steering(bible, candidate_dir=cand_dir)
    assert consumed == [steering_msg]
    assert not (cand_dir / ".pending_steering.json").is_file()
    assert any(u.speaker_role == "ExecutiveProducer" and steering_msg in u.spoken_dialogue for u in bible.writers_room_transcript)
    assert any(e.author_role == "ExecutiveProducer" and steering_msg in e.pivot_reason for e in bible.concept_evolution)

    snapshot = room._format_bible_snapshot(bible)
    assert "MANDATORY EXECUTIVE PRODUCER CREATIVE STEERING" in snapshot
    assert steering_msg in snapshot


def test_asset_tasks_cannot_clobber_shared_scene_and_code_lock_spans_swe_and_qa_turns(tmp_path):
    import threading
    from engine.studio.models import RevisionRecord
    from engine.studio.workers import available_specialists, task_lead_handler, validate_task_lead_decision

    (tmp_path / "game_bible.json").write_text(json.dumps({"game_title": "LockIsolationGame"}))
    (tmp_path / "project.godot").write_text("[input]\ndive={}\n")

    swe = FakeSwe()
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=bw.make_client(lambda _c: "{}"),
        voice_agent=object(),
        swe_agent=swe,
        library_factory=bw.FakeLibrary,
        filmstrip_fn=lambda d, ticks, out_path, input_script=None: (Image.new("RGB", (24, 8)).save(out_path), Path(out_path))[1],
    )

    # 1. Standalone asset tasks only expose their asset specialist and reject swe_engineer / qa_test_engineer
    vis_item = bw.item("tile_atlas", kind="visual_asset")
    anim_item = bw.item("hunter_sheet", kind="animation")
    aud_item = bw.item("clang_sfx", kind="audio_asset")
    mech_item = code_item(item_id="hunter_controller", modules=["mechanics.gd"])

    assert available_specialists(ctx, vis_item) == ["visual_artist"]
    assert available_specialists(ctx, anim_item) == ["visual_artist"]
    assert available_specialists(ctx, aud_item) == ["audio_designer"]
    assert set(available_specialists(ctx, mech_item)) == {
        "visual_artist",
        "audio_designer",
        "swe_engineer",
        "qa_test_engineer",
    }

    from engine.studio.models import TaskLeadDecision
    illegal_swe_from_visual = TaskLeadDecision.model_validate_json(
        _lead_json("delegate", strategy="Try hijacking scenes.gd", specialist_agent="swe_engineer", brief="Overwrite scenes.gd")
    )
    issues = validate_task_lead_decision(illegal_swe_from_visual, available_specialists(ctx, vis_item), [])
    assert any("swe_engineer" in i and "not available" in i for i in issues)

    # 2. Verify ctx.code_lock is held continuously from swe_engineer through qa_test_engineer
    lock_held_between_turns = {"held_during_turn_1_planning": False}
    lead_turns = iter([
        _lead_json("delegate", strategy="Run SWE", specialist_agent="swe_engineer", brief="Write mechanics.gd", target_paths=["mechanics.gd"]),
        _lead_json("delegate", strategy="Run QA", specialist_agent="qa_test_engineer", brief="Capture filmstrip"),
        _lead_json("submit_to_council", strategy="Done", submission_summary="Completed atomically"),
    ])
    turn_counter = {"n": 0}

    def responder(contents):
        text = contents if isinstance(contents, str) else "\n".join(c for c in contents if isinstance(c, str))
        if "Task Lead Strategist" in text:
            idx = turn_counter["n"]
            turn_counter["n"] += 1
            if idx == 1:
                # Between Turn 0 (swe_engineer) and Turn 1 (qa_test_engineer), code_lock MUST still be held!
                acquired = ctx.code_lock.acquire(blocking=False)
                if acquired:
                    ctx.code_lock.release()
                lock_held_between_turns["held_during_turn_1_planning"] = not acquired
            return next(lead_turns)
        return plan_json(ticks=(5, 25), segs=[(1, 25, ["dive"])])

    ctx.client = bw.make_client(responder)

    # Also seed a prior revision plate to verify it is preserved across revisions
    renders_dir = tmp_path / "artifacts" / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    prior_plate = renders_dir / "plate_hunter_sprite.png"
    Image.new("RGB", (16, 16)).save(prior_plate)
    rev0 = RevisionRecord(
        revision_index=0,
        worker_notes="Rev 0 fabricated sprite",
        proof_paths=[str(prior_plate)],
        verdict="request_changes",
        feedback="Fix velocity",
    )
    res = task_lead_handler(ctx)(code_item(item_id="hunter_controller", modules=["mechanics.gd"], revisions=[rev0]))
    assert lock_held_between_turns["held_during_turn_1_planning"] is True
    # And after run_task_lead finishes, code_lock is released
    assert ctx.code_lock.acquire(blocking=False) is True
    ctx.code_lock.release()
    # Prior revision plate was preserved alongside the new filmstrip_hunter_controller_rev1.png
    proof_names = [Path(p).name for p in res.proof_images]
    assert "plate_hunter_sprite.png" in proof_names
    assert "filmstrip_hunter_controller_rev1.png" in proof_names

    # 3. Verify plan_asset_requirement disambiguates target_path with item_id when another task already uses that path
    from engine.studio.workers import plan_asset_requirement
    (tmp_path / "catalog.json").write_text(
        json.dumps({"other_task": {"file": "assets/sprites/shared_name.png", "mode": "static_sprite"}})
    )
    ctx_asset = bw.ctx_for(tmp_path, lambda _c: bw.req_json("ignored", target="assets/sprites/shared_name.png"))
    req = plan_asset_requirement(ctx_asset, bw.item("task_two"), ("static_sprite",))
    assert req.asset_id == "task_two"
    assert req.target_path == "assets/sprites/task_two_shared_name.png"







