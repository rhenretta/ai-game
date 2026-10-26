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
        "script_writer",
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


def test_task_lead_yields_on_busy_code_lock_persists_pending_decision_and_resumes_exact_turn(tmp_path):
    """Verifies that a `run_task_lead` worker running inside `BoardScheduler`:
    1. Can complete non-code turns (e.g. `visual_artist` on Turn 0) even while `code_lock` is held.
    2. When it reaches a `swe_engineer` turn (Turn 1) and `code_lock` is busy, persists Turn 0 + Turn 1's
       `pending_decision` (`status="yielded"`) to `lead_trace_rev0.json` and yields the worker slot.
    3. Allows the freed worker slot to build another ready task (`vis_prop`) while `code_lock` remains busy.
    4. Once `code_lock` is released, resumes at Turn 1 from `pending_decision` without re-running Turn 0
       (`visual_artist`) or repeating the Turn 1 Task Lead planning LLM call."""
    import threading
    from engine.studio.workers import default_handlers

    (tmp_path / "game_bible.json").write_text(json.dumps({"game_title": "YieldResumeGame"}))
    (tmp_path / "project.godot").write_text("[input]\nui_right={}\n")

    swe = FakeSwe()
    code_lock = threading.Lock()
    lead_planning_calls = {"code_enemy_ai": 0, "vis_prop": 0}
    visual_fab_calls = []

    enemy_lead_turns = iter([
        # Call 1 (Turn 0): fabricate enemy sprite (does NOT need code_lock)
        _lead_json(
            "delegate",
            strategy="Turn 0: Fabricate enemy sprite first",
            specialist_agent="visual_artist",
            brief="Draw enemy sprite",
            target_paths=["assets/sprites/enemy.png"],
        ),
        # Call 2 (Turn 1): implement enemy AI in mechanics.gd (needs code_lock -> will yield if lock held, then resume without a 3rd call for Turn 1)
        _lead_json(
            "delegate",
            strategy="Turn 1: Implement enemy AI in mechanics.gd",
            specialist_agent="swe_engineer",
            brief="Write enemy AI in mechanics.gd",
            target_paths=["mechanics.gd"],
        ),
        # Call 3 (Turn 2): capture QA filmstrip
        _lead_json(
            "delegate",
            strategy="Turn 2: Capture QA filmstrip",
            specialist_agent="qa_test_engineer",
            brief="Capture enemy movement filmstrip",
        ),
        # Call 4 (Turn 3): submit to council
        _lead_json(
            "submit_to_council",
            strategy="All turns verified",
            submission_summary="Enemy sprite, code, and QA filmstrip complete",
        ),
    ])

    vis_prop_lead_turns = iter([
        _lead_json(
            "delegate",
            strategy="Fabricate prop sprite",
            specialist_agent="visual_artist",
            brief="Draw crate prop",
            target_paths=["assets/sprites/crate.png"],
        ),
        _lead_json(
            "submit_to_council",
            strategy="Prop ready",
            submission_summary="Crate sprite plate ready",
        ),
    ])

    yielded_event = threading.Event()
    vis_prop_done_event = threading.Event()

    def responder(contents):
        text = contents if isinstance(contents, str) else "\n".join(c for c in contents if isinstance(c, str))
        if "Task Lead Strategist" in text:
            if "[code_enemy_ai]" in text:
                lead_planning_calls["code_enemy_ai"] += 1
                return next(enemy_lead_turns)
            if "[vis_prop]" in text:
                lead_planning_calls["vis_prop"] += 1
                return next(vis_prop_lead_turns)
        if "Visual/Audio Production Lead" in text or "autonomous Visual Artist specialist" in text:
            if "[enemy]" in text or "[code_enemy_ai]" in text:
                visual_fab_calls.append("code_enemy_ai")
                return bw.req_json("enemy", mode="static_sprite", target="assets/sprites/enemy.png")
            if "[vis_prop]" in text:
                visual_fab_calls.append("vis_prop")
                return bw.req_json("vis_prop", mode="static_sprite", target="assets/sprites/crate.png")
        return plan_json(ticks=(5, 20), segs=[(1, 20, ["ui_right"])])

    events = []

    def on_event(et, payload):
        events.append((et, payload.get("item_id"), dict(payload)))
        if et == "backlog_item_yielded" and payload.get("item_id") == "code_enemy_ai":
            yielded_event.set()
        if et == "backlog_item_proof_ready" and payload.get("item_id") == "vis_prop":
            vis_prop_done_event.set()

    board = ProductionBoard(path=tmp_path / "backlog.json", event_callback=on_event)
    board.add_items(
        [
            WorkItemProposal(
                deliberation="d",
                item_id="code_enemy_ai",
                kind="mechanic_code",
                title="Enemy AI",
                spec="Build enemy sprite and AI",
                proof_spec="Show enemy moving",
                target_modules=["mechanics.gd"],
            )
        ],
        "GameplaySystemsDesigner",
    )

    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=bw.make_client(responder),
        swe_agent=swe,
        library_factory=bw.FakeLibrary,
        filmstrip_fn=lambda d, ticks, out_path, input_script=None: (Image.new("RGB", (24, 8)).save(out_path), Path(out_path))[1],
        code_lock=code_lock,
        event_callback=on_event,
    )

    # Acquire code_lock externally to simulate Worker 1 busy in SWE/QA on another code task
    assert code_lock.acquire(blocking=False)
    sch = BoardScheduler(board, default_handlers(ctx), max_workers=1, poll_interval=0.01, code_lock=code_lock)
    sch.start()
    try:
        # Wait for code_enemy_ai to finish Turn 0 (visual_artist), plan Turn 1 (swe_engineer), see code_lock busy, and yield
        assert yielded_event.wait(timeout=5.0), "code_enemy_ai did not yield when code_lock was held"

        # Inspect persisted trace right after yield: Turn 0 is saved and Turn 1 is saved as pending_decision
        trace_file = tmp_path / "artifacts" / "work_items" / "code_enemy_ai" / "lead_trace_rev0.json"
        assert trace_file.is_file()
        trace_after_yield = json.loads(trace_file.read_text(encoding="utf-8"))
        assert trace_after_yield["status"] == "yielded"
        assert len(trace_after_yield["turns"]) == 1
        assert trace_after_yield["turns"][0]["specialist_agent"] == "visual_artist"
        assert trace_after_yield["pending_decision"]["specialist_agent"] == "swe_engineer"
        assert lead_planning_calls["code_enemy_ai"] == 2

        # Add a parallel visual_asset item while code_lock is STILL held; the single worker slot must build vis_prop immediately
        board.add_items(
            [
                WorkItemProposal(
                    deliberation="d",
                    item_id="vis_prop",
                    kind="visual_asset",
                    title="Crate Prop",
                    spec="Draw crate",
                    proof_spec="Show crate plate",
                )
            ],
            "VisualArtDirectionLead",
        )
        assert vis_prop_done_event.wait(timeout=5.0), "Worker did not move on to vis_prop while code_enemy_ai was paused"
        assert board.get("vis_prop").state == "awaiting_review"
        assert board.get("code_enemy_ai").state == "ready"

        # Now release code_lock so code_enemy_ai can resume from Turn 1
        code_lock.release()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and board.get("code_enemy_ai").state != "awaiting_review":
            time.sleep(0.02)

        assert board.get("code_enemy_ai").state == "awaiting_review"
        # Turn 0 visual_artist was NOT re-run on resume, and Turn 1 planning call was NOT repeated (4 calls total: Turn 0, Turn 1, Turn 2, Submit)
        assert visual_fab_calls.count("code_enemy_ai") == 1
        assert lead_planning_calls["code_enemy_ai"] == 4

        final_trace = json.loads(trace_file.read_text(encoding="utf-8"))
        assert final_trace["status"] == "submitted"
        assert "pending_decision" not in final_trace
        assert [t["specialist_agent"] for t in final_trace["turns"]] == [
            "visual_artist",
            "swe_engineer",
            "qa_test_engineer",
        ]
    finally:
        if code_lock.locked():
            try:
                code_lock.release()
            except RuntimeError:
                pass
        sch.stop()


def test_council_filmstrip_contact_sheet_guidance_and_task_lead_multimodal_proof_inspection(tmp_path):
    """Verifies that:
    1. Every Council reviewer (`GameplayDesigner`, `VisualArtDirector`, `UXExperienceDesigner`, `CreativeDirector`)
       receives the explicit `HOW TO READ filmstrip_*.png QA CONTACT SHEETS` instructions and `[QA CONTACT SHEET: ...]`
       part labels so external QA margin strips (`[QA] tick ...`, `[QA] pos=...`) are never mistaken for in-game debug text.
    2. `_build_task_lead_multimodal_parts` attaches the reference 16:9 Concept Art Mockup and the latest `collected_proofs`
       image bytes so the Task Lead Strategist visually inspects captured proofs before submitting to the Council."""
    from engine.studio.models import GameBibleSpec, ReviewBatch, ReviewDecision, VisualMockupArtifact
    from engine.studio.production_board import WorkerResult
    from engine.studio.workers import _build_task_lead_multimodal_parts
    from engine.studio.writers_room import CreativeWritersRoom

    mockup_path = tmp_path / "artifacts" / "mockups" / "concept.png"
    mockup_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 36), (20, 40, 80)).save(mockup_path)

    filmstrip_path = tmp_path / "artifacts" / "work_items" / "code_item_1" / "filmstrip_rev0.png"
    filmstrip_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 72), (10, 20, 30)).save(filmstrip_path)

    bible = GameBibleSpec(
        deliberation="d",
        bible_id="b1",
        cartridge_id="c1",
        game_title="ContactSheetTest",
        core_genre="Action",
        visual_mockups=[
            VisualMockupArtifact(
                deliberation="d",
                mockup_id="m1",
                title="Concept",
                scene_description="desc",
                ui_callout_notes=["notes"],
                image_path=str(mockup_path),
            )
        ],
    )
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    ctx = WorkerContext(cartridge_dir=tmp_path)
    mm_parts = _build_task_lead_multimodal_parts(ctx, [str(filmstrip_path)])
    str_labels = [p for p in mm_parts if isinstance(p, str)]
    assert any("REFERENCE 16:9 CONCEPT ART MOCKUP" in s for s in str_labels)
    assert any("CAPTURED PROOF IMAGE (filmstrip_rev0.png)" in s for s in str_labels)
    assert len(mm_parts) == 4  # 2 text labels + 2 image Part objects

    board = ProductionBoard()
    board.add_items(
        [
            WorkItemProposal(
                deliberation="d",
                item_id="code_item_1",
                kind="mechanic_code",
                title="Controller",
                spec="Build controller",
                proof_spec="Show filmstrip",
                target_modules=["mechanics.gd"],
            )
        ],
        "GameplayDesigner",
    )
    claimed = board.claim_ready(1)
    assert len(claimed) == 1
    board.submit_result("code_item_1", WorkerResult(proof_images=[str(filmstrip_path)], notes="Built controller"))

    captured_prompts = []

    def fake_call(prompt, schema_cls, temperature=0.7, parts=None):
        captured_prompts.append((prompt, parts))
        return ReviewBatch(
            deliberation="Reviewed with QA contact sheet telemetry.",
            review_decisions=[
                ReviewDecision(deliberation="ok", item_id="code_item_1", feedback="", verdict="approve")
            ],
            cap_decisions=[],
        )

    room = CreativeWritersRoom(client=None, board=board)
    room._call_structured_llm = fake_call

    for role in ("GameplayDesigner", "VisualArtDirector", "UXExperienceDesigner", "CreativeDirector"):
        captured_prompts.clear()
        room._review_as(role, board.awaiting_review(), [], bible, "dna", tmp_path)
        assert len(captured_prompts) == 1
        prompt_text, parts_list = captured_prompts[0]
        assert "HOW TO READ `filmstrip_*.png` QA CONTACT SHEETS" in prompt_text
        assert "[QA] pos=(...)" in prompt_text
        part_strings = [p for p in (parts_list or []) if isinstance(p, str)]
        assert any("QA CONTACT SHEET:" in s for s in part_strings)


def test_context_curator_semantic_search_and_role_tailored_curation(tmp_path):
    """Verifies that:
    1. `CartridgeSemanticIndex` extracts discrete chunks across `asset` (concept art mockups & catalog sprites),
       `council_utterance` (`writers_room_transcript`), `design_spec` (`game_bible.json` & `rendering_strategy.json`
       per-subject treatments), `board_item`, and `code_module` (`bound_assets`).
    2. `semantic_search` embeds texts via `client.models.embed_content`, caches vectors by SHA-256 in
       `artifacts/context_index.json`, and ranks semantically matching chunks highest by cosine similarity.
    3. `ContextCuratorAgent.curate_context` enforces Rule 4 (`deliberation` at field index 0 on `ContextCuratorDecision`),
       assembles `CuratedContextPack` with selected text chunks + multimodal `Part` image attachments, and writes
       `artifacts/work_items/<item_id>/context_pack_<role>.json`.
    4. `_build_task_lead_multimodal_parts` attaches the 16:9 Concept Art Mockup even on Turn 0 (`collected_proofs=[]`),
       and Council `_review_as` attaches the 16:9 Concept Art Mockup when reviewing `level_map` items."""
    from types import SimpleNamespace
    from engine.studio.context_curator import (
        CartridgeSemanticIndex,
        ContextChunk,
        ContextCuratorAgent,
        ContextCuratorDecision,
    )
    from engine.studio.models import (
        GameBibleSpec,
        ReviewBatch,
        ReviewDecision,
        VisualMockupArtifact,
        WritersRoomUtterance,
    )
    from engine.studio.workers import _build_task_lead_multimodal_parts
    from engine.studio.writers_room import CreativeWritersRoom

    # Rule 4 check: deliberation must be at index 0
    assert list(ContextChunk.model_fields.keys())[0] == "deliberation"
    assert list(ContextCuratorDecision.model_fields.keys())[0] == "deliberation"

    # Set up cartridge files on disk
    mockup_path = tmp_path / "assets" / "mockups" / "mockup_spindle.png"
    mockup_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 36), (180, 120, 40)).save(mockup_path)

    sprite_path = tmp_path / "assets" / "sprites" / "surveyor_platform_rig.png"
    sprite_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (32, 32), (100, 110, 120, 255)).save(sprite_path)

    bible = GameBibleSpec(
        deliberation="d",
        bible_id="b_spindle",
        cartridge_id="c_spindle",
        game_title="Omega Spindle",
        core_genre="Precision Platformer",
        visual_style_guide="Ornate astronomical brass clockwork with vitreous enamel spokes and crumbling masonry.",
        visual_mockups=[
            VisualMockupArtifact(
                deliberation="d",
                mockup_id="m_spindle",
                title="Spindle Ascent Mockup",
                scene_description="Three interlocking brass clockwork gears above a crumbling stone surveyor platform.",
                ui_callout_notes=["Diegetic astrolabe dial"],
                image_path="assets/mockups/mockup_spindle.png",
            )
        ],
        writers_room_transcript=[
            WritersRoomUtterance(
                deliberation="d",
                utterance_id="u1",
                turn_index=1,
                speaker_role="VisualArtDirector",
                speaker_display_name="Visual Art Director",
                spoken_dialogue="The Sector Primus gears must use layered ornate brass rim textures with enamel cutouts, never flat primitive circles.",
                addressed_to=["GameplayDesigner"],
                bible_updates_summary=["Defined brass clockwork gear visual direction"],
            )
        ],
    )
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    (tmp_path / "rendering_strategy.json").write_text(
        json.dumps(
            {
                "deliberation": "d",
                "rendering_paradigm": "2.5D painterly clockwork",
                "renderer": "forward_plus",
                "treatments": [
                    {
                        "deliberation": "d",
                        "subject_id": "interactive_gear_trains",
                        "technique": "Layered Sprite2D nodes rotated procedurally with custom thermal shader.",
                        "target_resolution": [1024, 1024],
                        "notes": "Outer cycloidal tooth ring and enamel spokes.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    (tmp_path / "assets" / "asset_library.json").write_text(
        json.dumps(
            {
                "entries": {
                    "primus_crumbling_platform_prop": {
                        "asset_id": "primus_crumbling_platform_prop",
                        "asset_type": "prop",
                        "fabrication_mode": "static_sprite",
                        "res_path": "res://assets/sprites/surveyor_platform_rig.png",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    (tmp_path / "scenes.gd").write_text(
        "extends Node2D\nclass ClockworkGear:\n\tpass\nfunc build_arena():\n\tvar p = 'res://assets/sprites/surveyor_platform_rig.png'\n",
        encoding="utf-8",
    )

    board = ProductionBoard()
    board.add_items(
        [
            WorkItemProposal(
                deliberation="d",
                item_id="sector_primus_gear_arena",
                kind="level_map",
                title="Sector Primus 3-Gear Clockwork Arena",
                spec="Construct the 3-gear clockwork arena and lower surveyor platform.",
                proof_spec="Filmstrip showing rotating gears and platform.",
                target_modules=["scenes.gd"],
            )
        ],
        "GameplayDesigner",
    )

    # Verify Turn 0 Task Lead multimodal parts include the 16:9 Concept Art Mockup even when collected_proofs is empty
    ctx_turn0 = WorkerContext(cartridge_dir=tmp_path)
    turn0_parts = _build_task_lead_multimodal_parts(ctx_turn0, [])
    assert len(turn0_parts) == 2
    assert "REFERENCE 16:9 CONCEPT ART MOCKUP" in turn0_parts[0]

    # Create a mock live google.genai client with deterministic keyword-sensitive embeddings
    embed_calls = []

    class _LiveGenAIClientStub:
        __module__ = "google.genai.client"

        def __init__(self):
            self.models = self

        def embed_content(self, *, model, contents, config=None):
            embed_calls.append(list(contents))
            vecs = []
            for text in contents:
                tl = text.lower()
                v = [0.1] * 256
                if "gear" in tl or "clockwork" in tl or "enamel" in tl:
                    v[0] = 0.95
                if "platform" in tl or "surveyor" in tl:
                    v[1] = 0.90
                if "mockup" in tl or "concept" in tl:
                    v[2] = 0.85
                vecs.append(SimpleNamespace(values=v))
            return SimpleNamespace(embeddings=vecs)

        def generate_content(self, *, model, contents, config=None):
            decision = ContextCuratorDecision(
                deliberation="Selecting the concept art mockup, gear rendering treatment, VisualArtDirector utterance, and surveyor platform sprite for the Task Lead.",
                search_queries_used=["clockwork gear arena", "brass enamel gear rendering"],
                role_tailored_synthesis="Use the ornate brass/enamel gear aesthetic from the Concept Art Mockup and bind `surveyor_platform_rig.png` for the surveyor platform.",
                selected_chunk_ids=[
                    "asset:concept_mockup:0",
                    "design:rendering:treatment:interactive_gear_trains",
                    "council:utterance:turn_1_VisualArtDirector",
                    "asset:catalog:primus_crumbling_platform_prop",
                ],
                selected_image_paths=[str(mockup_path)],
            )
            return SimpleNamespace(text=decision.model_dump_json())

    live_client = _LiveGenAIClientStub()
    curator_events = []
    curator = ContextCuratorAgent(
        cartridge_dir=tmp_path,
        client=live_client,
        board=board,
        game_bible=bible,
        event_callback=lambda et, pl: curator_events.append((et, pl)),
    )

    chunks = curator.semantic_index.extract_corpus_chunks(board=board, game_bible=bible)
    chunk_ids = {c.chunk_id for c in chunks}
    assert "asset:concept_mockup:0" in chunk_ids
    assert "asset:catalog:primus_crumbling_platform_prop" in chunk_ids
    assert "council:utterance:turn_1_VisualArtDirector" in chunk_ids
    assert "design:rendering:treatment:interactive_gear_trains" in chunk_ids
    assert "board:item:sector_primus_gear_arena" in chunk_ids
    assert "code:module:scenes.gd" in chunk_ids

    pack = curator.curate_context(
        target_agent_role="task_lead",
        task_request="Sector Primus 3-Gear Clockwork Arena: Construct the 3-gear clockwork arena and lower surveyor platform.",
        item_id="sector_primus_gear_arena",
        item_kind="level_map",
    )
    assert "SEMANTICALLY CURATED STUDIO CONTEXT" in pack.prompt_block
    assert "interactive_gear_trains" in pack.prompt_block
    assert "council:utterance:turn_1_VisualArtDirector" in pack.prompt_block
    assert "surveyor_platform_rig.png" in pack.prompt_block
    assert len(pack.multimodal_parts) == 2
    assert "mockup_spindle.png" in pack.multimodal_parts[0]
    assert (tmp_path / "artifacts" / "work_items" / "sector_primus_gear_arena" / "context_pack_task_lead.json").is_file()
    assert any(et == "context_curator_packed" for et, _ in curator_events)

    # Verify SHA-256 embedding cache prevents re-embedding unchanged corpus chunks on a second call with the same query
    calls_before = len(embed_calls)
    curator.curate_context(
        target_agent_role="task_lead",
        task_request="Sector Primus 3-Gear Clockwork Arena: Construct the 3-gear clockwork arena and lower surveyor platform.",
        item_id="sector_primus_gear_arena",
        item_kind="level_map",
    )
    assert len(embed_calls) == calls_before  # 100% cache hit in artifacts/context_index.json!


def test_world_bg_color_rect_not_flagged_as_fullscreen_occlusion_and_snapshot_restores_all_gd(tmp_path):
    """Verifies that:
    1. `godot_harness.gd` does not falsely classify a world-space background `ColorRect` (`z_index <= 0` outside a
       foreground `CanvasLayer`) or a background `CanvasLayer` (`layer < 0`) as `current_in_ui = true` /
       `fullscreen_occlusion_detected = true`.
    2. `_snapshot_modules` and `_restore_modules` snapshot and restore all `.gd` files in `cartridge_dir` (even those
       not listed in `target_modules`), preventing a failed SWE verification from leaving side-effect `.gd` edits on disk."""
    from pathlib import Path
    from engine.studio.workers import _restore_modules, _snapshot_modules

    harness_src = (Path(__file__).resolve().parent.parent / "engine" / "sim" / "godot_harness.gd").read_text(encoding="utf-8")
    assert "var is_world_bg_color_rect = (node is ColorRect) and (not in_ui) and (node.z_index <= 0)" in harness_src
    assert "var is_bg_canvas_layer = (node is CanvasLayer) and (node.layer < 0)" in harness_src

    (tmp_path / "entities.gd").write_text("extends Node2D\n# original entities\n", encoding="utf-8")
    (tmp_path / "scenes.gd").write_text("extends Node2D\n# original scenes\n", encoding="utf-8")

    # Even if `modules` only lists `["entities.gd"]`, `_snapshot_modules` captures `scenes.gd` and removes newly created `.gd` files
    snap = _snapshot_modules(tmp_path, ["entities.gd"])
    (tmp_path / "entities.gd").write_text("extends Node2D\n# modified entities\n", encoding="utf-8")
    (tmp_path / "scenes.gd").write_text("extends Node2D\n# modified scenes side-effect\n", encoding="utf-8")
    (tmp_path / "extra_untracked.gd").write_text("extends Node2D\n", encoding="utf-8")

    _restore_modules(tmp_path, snap)
    assert (tmp_path / "entities.gd").read_text(encoding="utf-8") == "extends Node2D\n# original entities\n"
    assert (tmp_path / "scenes.gd").read_text(encoding="utf-8") == "extends Node2D\n# original scenes\n"
    assert not (tmp_path / "extra_untracked.gd").exists()


def test_curated_multimodal_context_and_anti_workaround_gates(tmp_path):
    """Verifies the 6 coordinated fixes that prevent agent context loss and superficial workarounds:
    1. `ContextCuratorAgent` loads `backlog.json` when `board` is None, includes `hframes`/`vframes`/`tex_size` in
       `asset:catalog:*` titles, and guarantees `board:item:{item_id}` + its prior proof image are included.
    2. `CartridgeAssetLibrary.build_prompt_summary_for_swe` reports `hframes`, `vframes`, and `tex_size` for every mode
       including `articulated_parts` (`hframes=1, vframes=1`).
    3. `StudioOrchestrator._check_godot_gameplay_and_asset_gates` distinguishes `early_failure_state_detected` from
       `fullscreen_occlusion_detected` and rejects `empty_sprite_frame_nodes` (`hframes > 1` on a single-frame texture).
    4. `run_swe_engineer_specialist` passes prior revision `proof_paths` (`extra_image_paths`) and `board` to
       `implement_requirement_track`.
    5. `CreativeWritersRoom._format_item_for_review` marks `worker_notes` as `UNVERIFIED_WORKER_CLAIMS`.
    """
    from engine.studio.asset_library import CartridgeAssetLibrary
    from engine.studio.context_curator import ContextCuratorAgent
    from engine.studio.models import AssetLibraryCatalog, AssetLibraryEntry, RequirementTrack, RevisionRecord, WorkItem
    from engine.studio.orchestrator import StudioOrchestrator
    from engine.studio.workers import run_swe_engineer_specialist
    from engine.studio.writers_room import CreativeWritersRoom

    # 1. ContextCuratorAgent reads backlog.json when board=None and includes asset hframes/vframes
    proof_img = tmp_path / "artifacts" / "work_items" / "mc_tether" / "filmstrip_mc_tether_rev0.png"
    proof_img.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 36), (40, 50, 60)).save(proof_img)

    backlog_payload = {
        "items": [
            {
                "deliberation": "d",
                "item_id": "mc_tether",
                "kind": "mechanic_code",
                "title": "Player Tether Controller",
                "spec": "Bind player rig and anchor.",
                "proof_spec": "4-frame filmstrip.",
                "state": "in_progress",
                "revisions": [
                    {
                        "revision_index": 0,
                        "proof_paths": [str(proof_img)],
                        "worker_notes": "Claimed fixed",
                        "verdict": "request_changes",
                        "feedback": "Fix clipped sprite.",
                    }
                ],
            }
        ]
    }
    (tmp_path / "backlog.json").write_text(json.dumps(backlog_payload), encoding="utf-8")

    assets_dir = tmp_path / "assets"
    sprites_dir = assets_dir / "sprites"
    sprites_dir.mkdir(parents=True, exist_ok=True)
    rig_png = sprites_dir / "va_kaelen_rig.png"
    Image.new("RGBA", (352, 512), (200, 120, 80, 255)).save(rig_png)

    cat = AssetLibraryCatalog(
        deliberation="d",
        cartridge_id="test_cart",
        entries={
            "va_kaelen_rig": AssetLibraryEntry(
                deliberation="d",
                asset_id="va_kaelen_rig",
                asset_type="sprite_sheet",
                fabrication_mode="articulated_parts",
                file_path="assets/sprites/va_kaelen_rig.png",
                res_path="res://assets/sprites/va_kaelen_rig.png",
                texture_pixel_size=[352, 512],
                hframes=1,
                vframes=1,
            )
        },
    )
    (assets_dir / "asset_library.json").write_text(cat.model_dump_json(indent=2), encoding="utf-8")

    curator = ContextCuratorAgent(cartridge_dir=tmp_path, client=None, board=None)
    chunks = curator.semantic_index.extract_corpus_chunks(board=None)
    chunk_ids = {c.chunk_id for c in chunks}
    assert "board:item:mc_tether" in chunk_ids
    asset_chunk = next(c for c in chunks if c.chunk_id == "asset:catalog:va_kaelen_rig")
    assert "hframes=1, vframes=1" in asset_chunk.title
    assert "mode=articulated_parts" in asset_chunk.title

    # 2. CartridgeAssetLibrary.build_prompt_summary_for_swe explicitly warns articulated_parts has hframes=1, vframes=1
    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path)
    swe_summary = lib.build_prompt_summary_for_swe()
    assert "hframes=1, vframes=1" in swe_summary
    assert "NEVER set hframes>1 or vframes>1 on this composite!" in swe_summary

    # 3. StudioOrchestrator gates: early_failure_state_detected vs fullscreen_occlusion_detected + empty_sprite_frame_nodes
    orch = StudioOrchestrator.__new__(StudioOrchestrator)
    track = RequirementTrack(
        deliberation="d",
        track_id="mc_tether",
        title="Player Tether",
        category="Core Mechanics",
        description="Spec",
        target_modules=["entities.gd"],
    )
    with pytest.raises(RuntimeError, match="Unrecoverable early failure state"):
        orch._check_godot_gameplay_and_asset_gates(
            tmp_path,
            track,
            {"fullscreen_occlusion_detected": False, "early_failure_state_detected": True},
        )
    with pytest.raises(RuntimeError, match="Invisible / mis-sliced Sprite2D frame detected"):
        orch._check_godot_gameplay_and_asset_gates(
            tmp_path,
            track,
            {
                "fullscreen_occlusion_detected": False,
                "early_failure_state_detected": False,
                "empty_sprite_frame_nodes": ["Silt Skiff(res://assets/sprites/va_kaelen_rig.png, frame=0, hframes=4, vframes=1)"],
            },
        )
    with pytest.raises(RuntimeError, match="Severe GPU SubViewport frame-rate thrashing detected"):
        orch._check_godot_gameplay_and_asset_gates(
            tmp_path,
            track,
            {
                "fullscreen_occlusion_detected": False,
                "early_failure_state_detected": False,
                "subviewport_thrashing_nodes": ["CompactionMaskViewport (size=(1920, 1080), UPDATE_ALWAYS + CLEAR_MODE_NEVER)"],
            },
        )

    # 4. run_swe_engineer_specialist forwards prior revision proof_paths and board to implement_requirement_track
    captured_swe_kwargs = {}

    class RecordingSwe:
        def implement_requirement_track(self, track, staging_dir, requirements_spec=None, prior_error=None, **kwargs):
            captured_swe_kwargs.update(kwargs)
            for m in track.target_modules:
                (Path(staging_dir) / m).write_text("extends Node2D\n", encoding="utf-8")

    item_with_rev = WorkItem.model_validate(backlog_payload["items"][0])
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=bw.make_client(lambda _c: "{}"),
        swe_agent=RecordingSwe(),
        board="sentinel_board",
    )
    run_swe_engineer_specialist(ctx, item_with_rev, "Fix sprite slicing", ["entities.gd"])
    assert captured_swe_kwargs.get("extra_image_paths") == [str(proof_img)]
    assert captured_swe_kwargs.get("board") == "sentinel_board"

    # 5. CreativeWritersRoom labels worker_notes as UNVERIFIED_WORKER_CLAIMS
    formatted = CreativeWritersRoom._format_item_for_review(item_with_rev)
    assert "UNVERIFIED_WORKER_CLAIMS='Claimed fixed'" in formatted


def test_multi_region_layout_validation_specialist_participation_and_level_filmstrip_gates(tmp_path):
    """Verifies the 4 architectural guardrails against tiny single-screen programmer-art games:
    1. `validate_level_layout` rejects a cramped 1-region / 1-screen layout when `LevelGenerationStrategy`
       defines multiple `level_archetypes`.
    2. `CreativeWritersRoom` seats un-spoken essential domain specialists (`UXExperienceDesigner`,
       `AudioAtmosphereDesigner`, `LevelAndProgressionDesigner`, `GameplaySystemsEngineer`) and ensures
       every required board specialist contributes before locking a full Production Board build.
    3. `validate_task_lead_decision` blocks `submit_to_council` on a `level_map` task when only a 2D wireframe
       `level_map_*.png` has been collected and `qa_test_engineer` is available, requiring a live Godot
       `filmstrip_*.png` / `level_filmstrip_*.png`, and `VisualArtDirector` reviews `level_map` items.
    4. `GodotHeadlessSimulator.run_until_complete` includes `--fixed-fps 60` in the Godot CLI command.
    """
    import inspect
    from engine.sim.godot_verifier import GodotHeadlessSimulator
    from engine.studio.models import (
        LevelArchetypeSpec,
        LevelEntitySpec,
        LevelGenerationStrategy,
        LevelLayoutSpec,
        LevelRegionSpec,
        LevelTriggerSpec,
        TaskLeadDecision,
        WorkItem,
        validate_level_layout,
    )
    from engine.studio.production_board import ProductionBoard
    from engine.studio.workers import validate_task_lead_decision
    from engine.studio.writers_room import CreativeWritersRoom

    # 1. Multi-archetype LevelGenerationStrategy rejects cramped 1-screen / 1-region LevelLayoutSpec
    strategy = LevelGenerationStrategy(
        deliberation="Three-tier vertical world.",
        generation_approach="hybrid",
        world_structure="Three linked vertical strata with scrolling camera.",
        runtime_construction_plan="Procedural strata with textured polygons and multiple nodes.",
        level_archetypes=[
            LevelArchetypeSpec(
                deliberation="Upper tier.",
                archetype_id="tier_1_crust",
                purpose="Acclimation",
                visual_identity="Rime crust",
                signature_challenge="First conduit",
            ),
            LevelArchetypeSpec(
                deliberation="Mid tier.",
                archetype_id="tier_2_cavern",
                purpose="Pressure routing",
                visual_identity="Deep cavern",
                signature_challenge="Multi-junction split",
            ),
        ],
        generation_rules=["Connect all tiers via vertical shafts."],
        difficulty_and_pacing_curve="Escalates with depth.",
    )
    cramped_layout = LevelLayoutSpec(
        deliberation="Shrunk to one 1280x720 screen.",
        world_bounds=(0.0, 0.0, 1280.0, 720.0),
        viewport_size=(1280, 720),
        intended_world_scale_in_viewports=1.0,
        camera_framing="fixed",
        world_regions=[
            LevelRegionSpec(
                deliberation="Single screen.",
                region_id="r0",
                label="R0",
                purpose="All in one box",
                bounds=(0.0, 0.0, 1280.0, 720.0),
                progression_order=0,
            )
        ],
        entities=[
            LevelEntitySpec(
                deliberation="Spawn.",
                entity_id="player",
                label="Player",
                entity_kind="spawn",
                region_id="r0",
                position=(200.0, 360.0),
                size=(48.0, 48.0),
            ),
            LevelEntitySpec(
                deliberation="Pipe.",
                entity_id="pipe_1",
                label="Pipe",
                entity_kind="mechanism",
                region_id="r0",
                position=(640.0, 360.0),
                size=(120.0, 48.0),
                initial_state="broken",
            ),
            LevelEntitySpec(
                deliberation="Well.",
                entity_id="well_1",
                label="Well",
                entity_kind="goal",
                region_id="r0",
                position=(960.0, 360.0),
                size=(96.0, 96.0),
                initial_state="dormant",
            ),
        ],
        triggers=[
            LevelTriggerSpec(
                deliberation="Pipe feeds well.",
                trigger_id="t1",
                source_entity_id="pipe_1",
                target_entity_id="well_1",
                condition="repaired",
                resulting_state="active",
            )
        ],
        progression_path=["r0"],
    )
    layout_issues = validate_level_layout(cramped_layout, strategy=strategy)
    assert any("world_regions only defines 1 region" in i for i in layout_issues)
    assert any("intended_world_scale_in_viewports" in i for i in layout_issues)
    assert any(">= 4 entities" in i for i in layout_issues)

    # 2. Level map review panel includes VisualArtDirector, and Task Lead cannot submit wireframe-only proof
    lvl_item = WorkItem(
        deliberation="Build world map",
        item_id="world_layout",
        title="World Layout",
        kind="level_map",
        spec="Multi-region world",
    )
    assert "VisualArtDirector" in CreativeWritersRoom._reviewers_for(lvl_item)

    submit_decision = TaskLeadDecision.model_validate_json(
        _lead_json("submit_to_council", strategy="Submit wireframe only", submission_summary="Done")
    )
    specialists = ["visual_artist", "swe_engineer", "qa_test_engineer"]
    wireframe_only = [str(tmp_path / "level_map_world_layout_rev0.png")]
    Path(wireframe_only[0]).write_bytes(b"png")
    lead_issues = validate_task_lead_decision(
        submit_decision,
        specialists,
        wireframe_only,
        item_kind="level_map",
    )
    assert any("Cannot submit_to_council for a 'level_map' task with only a 2D wireframe diagram" in i for i in lead_issues)

    with_filmstrip = wireframe_only + [str(tmp_path / "level_filmstrip_world_layout_rev0.png")]
    Path(with_filmstrip[1]).write_bytes(b"png")
    assert validate_task_lead_decision(submit_decision, specialists, with_filmstrip, item_kind="level_map") == []

    # 3. Essential domain specialists are polled once conversation is underway, and required on full board builds
    client = cb.base._FakeGeminiClient()
    board = ProductionBoard(tmp_path / "backlog.json")
    room = CreativeWritersRoom(client=client, max_turns=6, min_turns_before_convergence=2, board=board)
    bible = room.run_writers_room(dna=cb.base._sample_dna(), candidate_dir=tmp_path)
    spoken_roles = {u.speaker_role for u in bible.writers_room_transcript}
    for required_role in (
        "GameplayDesigner",
        "LevelAndProgressionDesigner",
        "GameplaySystemsEngineer",
        "VisualArtDirector",
        "UXExperienceDesigner",
        "AudioAtmosphereDesigner",
    ):
        assert required_role in spoken_roles, f"Expected {required_role} to speak before locking a full board build"

    # 4. GodotHeadlessSimulator enforces --fixed-fps 60
    sim_src = inspect.getsource(GodotHeadlessSimulator.run_until_complete)
    assert '"--fixed-fps", "60"' in sim_src


def test_qa_error_recovery_in_task_lead_and_failed_item_escalation(tmp_path):
    """Verifies that:
    1. When `qa_test_engineer` raises a `StudioAgentError` on an early turn inside `run_task_lead`,
       the error is caught and reported back to the Task Lead in `turn_history` (`verification_passed=False`)
       so the Task Lead can delegate a fix to `swe_engineer` and re-run QA without crashing the item to `failed`.
    2. `ProductionBoard.escalate_failed_to_decision` transitions a `failed` item to `needs_decision` with the
       worker failure reason, and `apply_cap_decision(action='rewrite_spec')` resets `failure_count=0` and `failure=''`.
    """
    from engine.studio.models import CapDecision, StudioAgentError, WorkItemProposal
    from engine.studio.production_board import ProductionBoard
    from engine.studio.workers import task_lead_handler
    from engine.studio.writers_room import CreativeWritersRoom

    (tmp_path / "game_bible.json").write_text(json.dumps({"game_title": "RecoveryGame"}))
    (tmp_path / "project.godot").write_text("[input]\ndive={}\n")

    lead_turns = iter([
        _lead_json("delegate", strategy="Turn 0: Implement spawner", specialist_agent="swe_engineer", brief="Implement spawner", target_paths=["mechanics.gd"]),
        _lead_json("delegate", strategy="Turn 1: Run QA filmstrip", specialist_agent="qa_test_engineer", brief="Test spawner"),
        _lead_json("delegate", strategy="Turn 2: QA failed; wire spawner into main.gd", specialist_agent="swe_engineer", brief="Wire spawner into main.gd", target_paths=["main.gd"]),
        _lead_json("delegate", strategy="Turn 3: Re-run QA filmstrip", specialist_agent="qa_test_engineer", brief="Re-test spawner"),
        _lead_json("submit_to_council", strategy="Turn 4: Submit after self-correcting QA error", submission_summary="Fixed wiring after QA error"),
    ])
    qa_attempts = {"n": 0}

    def flaky_filmstrip(d, ticks, out_path, input_script=None):
        qa_attempts["n"] += 1
        if qa_attempts["n"] == 1:
            raise StudioAgentError("QA filmstrip capture failed: spawner node not active")
        Image.new("RGB", (24, 8)).save(out_path)
        return Path(out_path)

    def responder(contents):
        text = contents if isinstance(contents, str) else "\n".join(c for c in contents if isinstance(c, str))
        if "Task Lead Strategist" in text:
            return next(lead_turns)
        return plan_json(ticks=(5, 25), segs=[(1, 25, ["dive"])])

    swe = FakeSwe()
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=bw.make_client(responder),
        swe_agent=swe,
        filmstrip_fn=flaky_filmstrip,
    )
    res = task_lead_handler(ctx)(code_item(item_id="mechanic_airspace_generator", modules=["mechanics.gd"]))
    assert qa_attempts["n"] == 2
    assert len(swe.calls) == 2
    assert "Fixed wiring after QA error" in res.notes

    # Verify board escalation of failed items -> needs_decision -> rewrite_spec resets failure_count
    board = ProductionBoard(tmp_path / "esc_backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="d",
                item_id="shader_caustic",
                kind="mechanic_code",
                title="Caustic Shader",
                spec="Old spec",
                proof_spec="Show shader",
                target_modules=["scenes.gd"],
            )
        ],
        "VisualArtDirector",
    )
    board.claim_ready(limit=1)
    board.fail_item("shader_caustic", "Error 1")
    board.requeue_failed("shader_caustic")
    board.claim_ready(limit=1)
    board.fail_item("shader_caustic", "Error 2: shader uniform mismatch")
    assert board.get("shader_caustic").failure_count == 2

    escalated = board.escalate_failed_to_decision("shader_caustic")
    assert escalated.state == "needs_decision"
    assert "worker failed 2 times" in escalated.decision_reason
    assert "last_worker_failure" in CreativeWritersRoom._format_item_for_review(escalated)

    rewritten = board.apply_cap_decision(
        CapDecision(
            deliberation="Simplify shader spec",
            item_id="shader_caustic",
            action="rewrite_spec",
            replacement_spec="Simplified CanvasItem shader spec",
        )
    )
    assert rewritten.state == "ready"
    assert rewritten.failure_count == 0
    assert rewritten.failure == ""


def test_prefix_cache_ordering_curator_reuse_and_token_usage_telemetry(tmp_path):
    """Verifies that:
    1. `call_gemini_with_transient_retry` records `usage_metadata` (`prompt_token_count`, `cached_content_token_count`,
       `candidates_token_count`, `thoughts_token_count`) in `TOKEN_USAGE_TRACKER`.
    2. `ContextCuratorAgent.curate_context` reuses the persisted pack for the same `(item_id, revision, normalized_role)`
       across multi-turn `task_lead` calls, `swe_engineer` retries, and parallel `council_reviewer:*` panelists without
       making redundant `ContextCuratorDecision` LLM calls.
    3. `CreativeWritersRoom._evaluate_specialist_listen` places the shared `PLAYER DNA & STUDIO CONTEXT` + `bible_snapshot`
       at byte 0 so parallel listener evaluations share an identical prefix.
    4. `plan_task_lead_turn` places static instructions + curated context + prior revision history before turn-volatile
       state and validation retries so Turns 0..N share an identical prefix at byte 0.
    """
    from types import SimpleNamespace
    from engine.studio.context_curator import ContextCuratorAgent, ContextCuratorDecision
    from engine.studio.models import (
        GameBibleSpec,
        TOKEN_USAGE_TRACKER,
        WorkItem,
        WorkItemProposal,
        WritersRoomListenCheck,
        call_gemini_with_transient_retry,
    )
    from engine.studio.production_board import ProductionBoard
    from engine.studio.workers import WorkerContext, plan_task_lead_turn
    from engine.studio.writers_room import CreativeWritersRoom, SPECIALIST_ROSTER

    # 1. Verify TOKEN_USAGE_TRACKER captures usage_metadata
    TOKEN_USAGE_TRACKER.reset()
    fake_resp = SimpleNamespace(
        text="ok",
        usage_metadata=SimpleNamespace(
            prompt_token_count=4000,
            cached_content_token_count=3200,
            candidates_token_count=300,
            thoughts_token_count=200,
            total_token_count=4500,
        ),
    )
    call_gemini_with_transient_retry(lambda: fake_resp, label="plan_task_lead_turn[item_1:t1]")
    summary = TOKEN_USAGE_TRACKER.get_summary()
    assert summary["total_calls"] == 1
    assert summary["calls_with_cache_hit"] == 1
    assert summary["total_prompt_tokens"] == 4000
    assert summary["total_cached_tokens"] == 3200
    assert summary["cached_token_share_pct"] == 80.0
    assert summary["by_label"]["plan_task_lead_turn"]["cached_tokens"] == 3200

    # 2. Verify ContextCuratorAgent reuses cached pack across turns and across parallel council_reviewer:* roles
    board = ProductionBoard(tmp_path / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="d",
                item_id="mech_valve",
                kind="mechanic_code",
                title="Thermal Valve",
                spec="Build interactive valve.",
                proof_spec="Filmstrip of valve rotation.",
                target_modules=["mechanics.gd"],
            )
        ],
        "GameplayDesigner",
    )
    curator_llm_calls = []

    class _CuratorLiveStub:
        __module__ = "google.genai.client"

        def __init__(self):
            self.models = self

        def embed_content(self, *, model, contents, config=None):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.2] * 256) for _ in contents])

        def generate_content(self, *, model, contents, config=None):
            curator_llm_calls.append(contents)
            dec = ContextCuratorDecision(
                deliberation="Selected valve context.",
                search_queries_used=["thermal valve"],
                role_tailored_synthesis="Synthesized valve context.",
                selected_chunk_ids=["board:item:mech_valve"],
                selected_image_paths=[],
            )
            return SimpleNamespace(text=dec.model_dump_json())

    curator = ContextCuratorAgent(cartridge_dir=tmp_path, client=_CuratorLiveStub(), board=board)
    pack_r1 = curator.curate_context("council_reviewer:VisualArtDirector", "Review valve", item_id="mech_valve", item_kind="mechanic_code")
    pack_r2 = curator.curate_context("council_reviewer:GameplayDesigner", "Review valve", item_id="mech_valve", item_kind="mechanic_code")
    pack_r3 = curator.curate_context("council_reviewer:UXExperienceDesigner", "Review valve", item_id="mech_valve", item_kind="mechanic_code")
    assert len(curator_llm_calls) == 1  # All 3 reviewers shared 1 curation call!
    assert pack_r1.prompt_block == pack_r2.prompt_block == pack_r3.prompt_block

    # 3. Verify _evaluate_specialist_listen puts shared DNA + Bible snapshot at byte 0 across specialists
    listen_prompts = []
    bible = GameBibleSpec(deliberation="d", game_title="PrefixCacheGame", elevator_pitch="A deep systemic world.")
    room = CreativeWritersRoom(client=cb.base._FakeGeminiClient())
    room._call_structured_llm = lambda prompt, schema_cls, **kw: (
        listen_prompts.append(prompt),
        WritersRoomListenCheck(deliberation="d", should_interject=False, urgency_score=0.1, intended_contribution_summary=""),
    )[1]
    for sp in SPECIALIST_ROSTER[:4]:
        room._evaluate_specialist_listen(sp, bible, "Shared DNA summary block")
    assert len(listen_prompts) == 4
    shared_prefix_len = len("PLAYER DNA & STUDIO CONTEXT:\nShared DNA summary block\n\n" + room._format_bible_snapshot(bible))
    assert shared_prefix_len > 200
    first_prefix = listen_prompts[0][:shared_prefix_len]
    for p in listen_prompts[1:]:
        assert p.startswith(first_prefix)

    # 4. Verify plan_task_lead_turn shares byte-0 prefix across Turn 0 and Turn 1
    lead_prompts = []

    def lead_responder(contents):
        text = contents if isinstance(contents, str) else "\n".join(c for c in contents if isinstance(c, str))
        lead_prompts.append(text)
        return _lead_json("delegate", strategy="Delegate to SWE", specialist_agent="swe_engineer", brief="Implement valve", target_paths=["mechanics.gd"])

    ctx = WorkerContext(cartridge_dir=tmp_path, client=bw.make_client(lead_responder), swe_agent=FakeSwe(), board=board)
    item = board.get("mech_valve")
    plan_task_lead_turn(ctx, item, turn_history=[], collected_proofs=[])
    from engine.studio.models import SpecialistReport
    plan_task_lead_turn(
        ctx,
        item,
        turn_history=[
            SpecialistReport(
                deliberation="d",
                turn_index=0,
                specialist_agent="swe_engineer",
                brief="Implement valve",
                summary="Implemented valve",
                touched_files=["mechanics.gd"],
                verification_passed=True,
            )
        ],
        collected_proofs=[],
    )
    assert len(lead_prompts) == 2
    common_instr_idx = lead_prompts[0].index("CARTRIDGE SYMBOLS & FILES:")
    assert common_instr_idx > 2500
    assert lead_prompts[0][:common_instr_idx] == lead_prompts[1][:common_instr_idx]


def test_ai_cost_optimizations_without_quality_loss(tmp_path):
    """Verifies the 6 AI token & call optimizations:
    1. Standalone single-specialist asset tasks (`visual_asset`, `animation`, `audio_asset`) in `run_task_lead`
       skip the redundant 3-call Task Lead wrapper in live mode (`is_real_genai_client`) and execute directly
       while writing `lead_trace_rev0.json`.
    2. `LeadGameplaySWEAgent._summarize_non_target_gd_module` condenses large non-target `.gd` modules to their
       cross-module API signature + `get_state_snapshot` contract instead of dumping 20KB+ of implementation body.
    3. `plan_filmstrip` reuses `context_pack_task_lead.json` when present on disk (0 extra curation calls), and
       `_build_task_lead_multimodal_parts` attaches only the latest filmstrip when multiple filmstrips exist.
    4. `CreativeWritersRoom._select_next_speaker` in live mode narrows `WritersRoomListenCheck` polling to
       `addressed_to` specialists (plus at most 1 unheard essential specialist).
    5. `ProductionBoard.snapshot_text` compacts terminal (`approved` / `cancelled`) items when board size > 8.
    """
    from types import SimpleNamespace
    from engine.studio.context_curator import ContextCuratorDecision
    from engine.studio.models import GameBibleSpec, ReviewDecision, WorkItemProposal, WritersRoomListenCheck
    from engine.studio.production_board import ProductionBoard, WorkerResult
    from engine.studio.swe import LeadGameplaySWEAgent
    from engine.studio.workers import _build_task_lead_multimodal_parts, plan_filmstrip, run_task_lead
    from engine.studio.writers_room import CreativeWritersRoom

    (tmp_path / "game_bible.json").write_text(
        GameBibleSpec(deliberation="d", bible_id="b1", cartridge_id="c1", game_title="OptGame", core_genre="Sim").model_dump_json()
    )
    (tmp_path / "project.godot").write_text("[input]\ndive={}\n")

    # 1. Task Lead Strategist runs for visual_asset in live mode and reuses context_pack_task_lead.json across turns
    llm_schemas_called = []
    lead_live_turns = iter([
        _lead_json(
            "delegate",
            strategy="Delegate crate_prop to visual_artist",
            specialist_agent="visual_artist",
            brief="Fabricate crate_prop sprite",
            target_paths=["assets/sprites/crate_prop.png"],
        ),
        _lead_json(
            "submit_to_council",
            strategy="Submit verified crate_prop plate",
            submission_summary="Fabricated and verified crate_prop.",
        ),
    ])

    class _LiveOptClient:
        __module__ = "google.genai.client"

        def __init__(self):
            self.models = self

        def embed_content(self, *, model, contents, config=None):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.2] * 256) for _ in contents])

        def generate_content(self, *, model, contents, config=None):
            schema = getattr(config, "response_schema", None)
            sname = getattr(schema, "__name__", str(schema))
            llm_schemas_called.append(sname)
            if sname == "ContextCuratorDecision":
                return SimpleNamespace(
                    text=ContextCuratorDecision(
                        deliberation="d",
                        search_queries_used=["q"],
                        role_tailored_synthesis="synth",
                        selected_chunk_ids=[],
                        selected_image_paths=[],
                    ).model_dump_json()
                )
            if sname == "TaskLeadDecision":
                return SimpleNamespace(text=next(lead_live_turns))
            if sname == "AssetRequirement":
                return SimpleNamespace(text=bw.req_json("crate_prop"))
            if sname == "FilmstripPlan":
                return SimpleNamespace(text=plan_json(ticks=(5, 20), segs=[(1, 20, ["dive"])]))
            if sname == "WritersRoomListenCheck":
                return SimpleNamespace(
                    text=WritersRoomListenCheck(
                        deliberation="d",
                        should_interject=True,
                        urgency_score=0.8,
                        intended_contribution_summary="s",
                    ).model_dump_json()
                )
            raise AssertionError(f"Unexpected schema {sname}")

    live_client = _LiveOptClient()
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=live_client,
        library_factory=bw.FakeLibrary,
    )
    vis_item = bw.item("crate_prop", kind="visual_asset")
    res = run_task_lead(ctx, vis_item)
    assert llm_schemas_called.count("TaskLeadDecision") == 2
    assert "AssetRequirement" in llm_schemas_called
    trace_file = tmp_path / "artifacts" / "work_items" / "crate_prop" / "lead_trace_rev0.json"
    assert trace_file.is_file()
    assert json.loads(trace_file.read_text(encoding="utf-8"))["status"] == "submitted"
    assert res.proof_images

    # 2. _summarize_non_target_gd_module extracts API signatures + get_state_snapshot without dumping method bodies
    long_gd = (
        "extends Node2D\nclass_name BigEntities\nsignal valve_opened(pressure: float)\nconst MAX_PRESSURE: float = 100.0\n"
        "var current_pressure: float = 12.5\n"
        "func apply_impulse_step(delta: float, force: Vector2) -> void:\n"
        + "".join(f"\tvar local_calc_{i} = delta * {i}\n" for i in range(120))
        + "func get_state_snapshot() -> Dictionary:\n\treturn {'pressure': current_pressure, 'max': MAX_PRESSURE}\n"
    )
    assert len(long_gd) > 2500
    sig = LeadGameplaySWEAgent._summarize_non_target_gd_module("entities.gd", long_gd)
    assert len(sig) < 500
    assert "signal valve_opened" in sig
    assert "func apply_impulse_step(delta: float, force: Vector2) -> void:" in sig
    assert "local_calc_50" not in sig
    assert "func get_state_snapshot() -> Dictionary:" in sig
    assert "'pressure': current_pressure" in sig

    # 3. plan_filmstrip reuses context_pack_task_lead.json when present (0 extra ContextCuratorDecision calls)
    mech = code_item(item_id="mech_reuse", modules=["mechanics.gd"])
    from engine.studio.context_curator import ContextCuratorAgent
    curator = ContextCuratorAgent(cartridge_dir=tmp_path, client=live_client)
    curator.curate_context(
        target_agent_role="task_lead",
        task_request="Build mech_reuse",
        item_id="mech_reuse",
        item_kind="mechanic_code",
    )
    llm_schemas_called.clear()
    fplan = plan_filmstrip(ctx, mech)
    assert fplan.capture_ticks == [5, 20]
    assert "ContextCuratorDecision" not in llm_schemas_called
    assert llm_schemas_called == ["FilmstripPlan"]

    # Deduplicating multiple filmstrips in _build_task_lead_multimodal_parts keeps only the latest filmstrip
    f1 = tmp_path / "filmstrip_turn1.png"
    f2 = tmp_path / "filmstrip_turn3.png"
    plate = tmp_path / "plate_hero.png"
    for p in (f1, f2, plate):
        Image.new("RGB", (16, 16), (10, 20, 30)).save(p)
    mm_parts = _build_task_lead_multimodal_parts(ctx, [str(plate), str(f1), str(f2)])
    mm_labels = [p for p in mm_parts if isinstance(p, str)]
    assert any("plate_hero.png" in s for s in mm_labels)
    assert any("filmstrip_turn3.png" in s for s in mm_labels)
    assert not any("filmstrip_turn1.png" in s for s in mm_labels)

    # 4. CreativeWritersRoom._select_next_speaker narrows polling to invited specialists (+ 1 unheard essential if invited already spoke)
    llm_schemas_called.clear()
    room = CreativeWritersRoom(client=live_client)
    bible = GameBibleSpec(deliberation="d", bible_id="b1", cartridge_id="c1", game_title="OptGame", core_genre="Sim")
    room._select_next_speaker(
        last_speaker_role="CreativeDirector",
        addressed_to=["GameplayDesigner"],
        game_bible=bible,
        dna_summary="dna",
        spoken_counts={"CreativeDirector": 1, "VisualArtDirector": 1},
    )
    # GameplayDesigner is unheard and explicitly invited -> only 1 call instead of 7
    assert len(llm_schemas_called) == 1

    llm_schemas_called.clear()
    room._select_next_speaker(
        last_speaker_role="CreativeDirector",
        addressed_to=["GameplayDesigner"],
        game_bible=bible,
        dna_summary="dna",
        spoken_counts={"CreativeDirector": 1, "GameplayDesigner": 1},
    )
    # GameplayDesigner already spoke -> polls GameplayDesigner + 1 unheard essential specialist (2 calls instead of 7)
    assert len(llm_schemas_called) == 2

    # 5. ProductionBoard.snapshot_text compacts approved items when board has > 8 items
    board = ProductionBoard()
    proposals = [
        WorkItemProposal(
            deliberation="d",
            item_id=f"item_{i}",
            kind="visual_asset" if i < 7 else "mechanic_code",
            title=f"Task Number {i}",
            spec="Spec",
            proof_spec="Proof",
        )
        for i in range(10)
    ]
    board.add_items(proposals, "GameplayDesigner")
    board.claim_ready(limit=6)
    for i in range(6):
        board.submit_result(f"item_{i}", WorkerResult(notes="ok"))
        board.apply_review(ReviewDecision(deliberation="ok", item_id=f"item_{i}", verdict="approve"))
    snap = board.snapshot_text()
    assert "- APPROVED ITEMS (6):" in snap
    assert "[item_0] (visual_asset)" in snap
    assert "- [item_8] mechanic_code 'Task Number 8' state=ready" in snap


def test_harness_shift_left_reactive_qa_and_asset_revision_escalation(tmp_path):
    """Verifies the shift-left harness improvements:
    1. Standalone asset tasks (`visual_asset`, `animation`, `audio_asset`) use the fast single-specialist path on Rev 0,
       but once rejected by the Council (`len(item.revisions) > 0`), `available_specialists` unlocks `swe_engineer` /
       `qa_test_engineer` and `run_task_lead` runs the full `TaskLeadDecision` strategist loop so the Task Lead can
       re-strategize, split multi-asset packs, or composite in-engine proofs instead of repeating a failed 1-shot pipe.
    2. `validate_filmstrip_plan` rejects `setup_calls` starting with `simulate_*` (which bypass real player input/physics)
       while `_summarize_cartridge_for_qa` hides `simulate_*` shortcuts, and `_serialize_input_segments` preserves reactive
       state triggers (`when_snapshot_key`, `when_operator`, `when_value`, `hold_ticks`).
    3. `_evaluate_qa_state_assertions` enforces `QAStateAssertion`s against live `tick_snapshots`.
    4. `_remove_magenta_background` despills light/white translucent boundary pixels blended with `#FF00FF`.
    5. `LeadGameplaySWEAgent` indexes, slices, and surgically patches `tests/test_authored_beats.gd` alongside root `.gd` files.
    6. `OnboardingPlan` exposes open-ended strategic fields (`purpose_and_stakes_framing` and `causal_chain_and_feedback_strategy`).
    """
    from types import SimpleNamespace
    from engine.studio.context_curator import ContextCuratorDecision
    from engine.studio.models import (
        CodeSliceRequest,
        FilmstripPlan,
        InputSegment,
        OnboardingPlan,
        QASetupCall,
        QAStateAssertion,
        RevisionRecord,
        StudioAgentError,
        SurgicalCodePatch,
        TaskLeadDecision,
    )
    from engine.studio.proofs import _evaluate_qa_state_assertions
    from engine.studio.swe import LeadGameplaySWEAgent
    from engine.studio.visual import VisualArtSpriteDesignerAgent
    from engine.studio.workers import (
        _serialize_input_segments,
        _summarize_cartridge_for_qa,
        available_specialists,
        run_task_lead,
        validate_filmstrip_plan,
    )

    # 1. Asset task on Revision 1 unlocks full specialists and TaskLeadDecision loop
    (tmp_path / "project.godot").write_text("[input]\nact={}\n", encoding="utf-8")
    (tmp_path / "main.gd").write_text(
        "extends Node2D\nvar pressure: float = 0.0\nfunc real_action() -> void:\n\tpass\nfunc simulate_full_solve() -> void:\n\tpass\n",
        encoding="utf-8",
    )
    swe = FakeSwe()
    vis_rev0 = bw.item("multi_prop_pack", kind="visual_asset")
    vis_rev1 = vis_rev0.model_copy(
        update={
            "revisions": [
                RevisionRecord(
                    revision_index=0,
                    proof_paths=[],
                    worker_notes="Generated single sprite",
                    verdict="request_changes",
                    feedback="Proof only shows 1 prop; composite all props in-engine or split task.",
                )
            ]
        }
    )
    ctx_check = WorkerContext(
        cartridge_dir=tmp_path,
        client=bw.make_client(lambda _: ""),
        swe_agent=swe,
        library_factory=bw.FakeLibrary,
    )
    assert available_specialists(ctx_check, vis_rev0) == ["visual_artist"]
    rev1_specs = available_specialists(ctx_check, vis_rev1)
    assert "visual_artist" in rev1_specs
    assert "swe_engineer" in rev1_specs
    assert "qa_test_engineer" in rev1_specs

    lead_called = {"n": 0}
    lead_turns = iter([
        TaskLeadDecision(
            deliberation="Council rejected single-sprite proof; delegate to visual_artist with updated multi-panel brief.",
            current_strategy="Fabricate composite prop sheet addressing Council feedback.",
            specialist_agent="visual_artist",
            brief="Fabricate full prop sheet.",
            action="delegate",
        ).model_dump_json(),
        TaskLeadDecision(
            deliberation="Composite proof ready; submit to Council.",
            current_strategy="Submit updated proof.",
            submission_summary="Updated prop pack per Council feedback.",
            action="submit_to_council",
        ).model_dump_json(),
    ])

    class _Rev1LiveClient:
        __module__ = "google.genai.client"

        def __init__(self):
            self.models = self

        def embed_content(self, *, model, contents, config=None):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.2] * 256) for _ in contents])

        def generate_content(self, *, model, contents, config=None):
            schema = getattr(config, "response_schema", None)
            sname = getattr(schema, "__name__", "")
            if sname == "ContextCuratorDecision":
                return SimpleNamespace(
                    text=ContextCuratorDecision(
                        deliberation="d",
                        search_queries_used=["q"],
                        role_tailored_synthesis="synth",
                        selected_chunk_ids=[],
                        selected_image_paths=[],
                    ).model_dump_json()
                )
            if sname == "TaskLeadDecision":
                lead_called["n"] += 1
                return SimpleNamespace(text=next(lead_turns))
            if sname == "AssetRequirement":
                return SimpleNamespace(text=bw.req_json("multi_prop_pack"))
            raise AssertionError(f"Unexpected schema {sname}")

    ctx_live = WorkerContext(
        cartridge_dir=tmp_path,
        client=_Rev1LiveClient(),
        swe_agent=swe,
        library_factory=bw.FakeLibrary,
    )
    res_rev1 = run_task_lead(ctx_live, vis_rev1)
    assert lead_called["n"] == 2
    assert res_rev1.proof_images

    # 2. QA summary filters simulate_* and validate_filmstrip_plan rejects simulate_* setup_calls
    qa_summary = _summarize_cartridge_for_qa(tmp_path)
    assert "real_action()" in qa_summary
    assert "simulate_full_solve" not in qa_summary

    bad_sim_plan = FilmstripPlan(
        deliberation="d",
        capture_ticks=[5, 20],
        input_segments=[InputSegment(deliberation="d", from_tick=1, to_tick=10, actions=["act"])],
        setup_calls=[QASetupCall(deliberation="d", at_tick=1, method_name="simulate_full_solve", args_json="[]")],
        frame_expectations=["f1", "f2"],
    )
    issues = validate_filmstrip_plan(bad_sim_plan, ["act"])
    assert any("simulate_full_solve" in i and "simulate_*" in i for i in issues)

    reactive_plan = FilmstripPlan(
        deliberation="d",
        capture_ticks=[5, 40],
        input_segments=[
            InputSegment(
                deliberation="d",
                from_tick=1,
                to_tick=60,
                actions=["act"],
                when_snapshot_key="socket_dist",
                when_operator="<=",
                when_value="18.0",
                hold_ticks=12,
            )
        ],
        state_assertions=[
            QAStateAssertion(deliberation="d", at_tick=40, snapshot_key="pressure", expected_value_or_change="> 10")
        ],
        frame_expectations=["f1", "f2"],
    )
    assert validate_filmstrip_plan(reactive_plan, ["act"]) == []
    serialized_segs = _serialize_input_segments(reactive_plan)
    assert serialized_segs[0]["when_snapshot_key"] == "socket_dist"
    assert serialized_segs[0]["when_operator"] == "<="
    assert serialized_segs[0]["when_value"] == "18.0"
    assert serialized_segs[0]["hold_ticks"] == 12

    # 3. _evaluate_qa_state_assertions verifies state_assertions against live tick snapshots
    snaps_pass = [
        {"tick": 1, "pressure": 0.0, "locked": True},
        {"tick": 40, "pressure": 25.0, "locked": False},
    ]
    _evaluate_qa_state_assertions(
        snaps_pass,
        [
            QAStateAssertion(deliberation="d", at_tick=40, snapshot_key="pressure", expected_value_or_change=">= 20.0"),
            QAStateAssertion(deliberation="d", at_tick=40, snapshot_key="locked", expected_value_or_change="false"),
        ],
    )
    snaps_fail = [
        {"tick": 1, "pressure": 0.0, "locked": True},
        {"tick": 40, "pressure": 5.0, "locked": True},
    ]
    with pytest.raises(StudioAgentError, match="pressure"):
        _evaluate_qa_state_assertions(
            snaps_fail,
            [QAStateAssertion(deliberation="d", at_tick=40, snapshot_key="pressure", expected_value_or_change=">= 20.0")],
        )

    # 4. Chroma-key despill cleans white/light translucent boundary pixels blended with #FF00FF
    img = Image.new("RGBA", (64, 64), (255, 0, 255, 255))
    # Interior opaque sprite block (warm white/steam)
    for y in range(16, 48):
        for x in range(16, 48):
            img.putpixel((x, y), (235, 235, 235, 255))
    # Translucent white puff pixels blended with #FF00FF at the boundary (R=250, G=130, B=250 -> g >= 110, magenta_excess = 120 > 45)
    for x in range(16, 48):
        img.putpixel((x, 15), (250, 130, 250, 255))
    cleaned = VisualArtSpriteDesignerAgent.extract_clean_alpha_sprite(img, target_size=(64, 64))
    for y in range(cleaned.height):
        for x in range(cleaned.width):
            r_px, g_px, b_px, a_px = cleaned.getpixel((x, y))
            if a_px > 0:
                assert min(r_px, b_px) - g_px <= 45, f"Residual pink/magenta fringe at {(x, y)}: {(r_px, g_px, b_px, a_px)}"

    # 5. SWE symbol index, slice extractor, and surgical patcher support tests/test_authored_beats.gd
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(exist_ok=True)
    beat_file = tests_dir / "test_authored_beats.gd"
    beat_file.write_text(
        "extends RefCounted\nfunc run(root: Node) -> Dictionary:\n\tvar ok: bool = false\n\treturn {'passed': ok}\n",
        encoding="utf-8",
    )
    swe_agent = LeadGameplaySWEAgent(client=None)
    sym_idx = swe_agent._build_cartridge_symbol_index(tmp_path)
    assert "tests/test_authored_beats.gd" in sym_idx
    slice_txt = swe_agent._extract_code_slice(
        tmp_path,
        CodeSliceRequest(
            deliberation="Inspect beat runner",
            filename="tests/test_authored_beats.gd",
            symbol_query="func run",
        ),
    )
    assert "tests/test_authored_beats.gd" in slice_txt
    assert "var ok: bool = false" in slice_txt
    patched = swe_agent._apply_surgical_patches(
        tmp_path,
        [
            SurgicalCodePatch(
                deliberation="Fix beat check",
                filename="tests/test_authored_beats.gd",
                start_line=3,
                end_line=3,
                replacement_code="\tvar ok: bool = true",
            )
        ],
    )
    assert "tests/test_authored_beats.gd" in patched
    assert "var ok: bool = true" in beat_file.read_text(encoding="utf-8")

    # 6. OnboardingPlan supports open-ended empathetic strategy fields and WritersRoomTurnProposal.for_role bounds schema complexity
    from engine.studio.models import WritersRoomTurnProposal
    from engine.studio.writers_room import SPECIALIST_ROSTER

    onb = OnboardingPlan(
        deliberation="Strategizing how a cold player understands purpose and causality.",
        world_introduction="A storm-tossed signal relay station.",
        player_role_and_stakes="Keep the beacon aligned before the fleet hits the shoals.",
        purpose_and_stakes_framing="Visual fleet silhouettes on the horizon drift toward the reef when the beam wanders.",
        causal_chain_and_feedback_strategy="Turning the crank visibly rotates the mirror gear and sweeps the beam; locked gears spark amber with a clank.",
    )
    assert "fleet silhouettes" in onb.purpose_and_stakes_framing
    assert "locked gears spark" in onb.causal_chain_and_feedback_strategy

    for sp in SPECIALIST_ROSTER:
        scoped_cls = WritersRoomTurnProposal.for_role(sp["role"])
        assert scoped_cls.__name__ == "WritersRoomTurnProposal"
        js = scoped_cls.model_json_schema()
        defs = js.get("$defs", {})
        total_props = len(js.get("properties", {})) + sum(len(v.get("properties", {})) for v in defs.values())
        assert len(defs) <= 12, f"Role {sp['role']} has too many $defs ({len(defs)}) for Gemini structured output"
        assert total_props <= 95, f"Role {sp['role']} has too many properties ({total_props}) for Gemini structured output"


def test_gateway_pause_and_runaway_loop_circuit_breakers(tmp_path: Path, monkeypatch):
    """Verifies the 4 hard safeguards against runaway/unpaused LLM calls:
    1. Gateway-level pause enforcement blocks both `call_gemini_with_transient_retry` and direct
       `client.models.generate_content` calls when `.studio_paused.json` is active.
    2. `TokenUsageTracker.check_circuit_breaker` trips when consecutive SWE/PEV calls exceed the ceiling,
       automatically writing `.studio_paused.json` and raising `StudioPausedError`.
    3. `StudioOrchestrator._execute_iterative_swe_loop` never invokes legacy post-board loops
       (`formulate_level_layout`, `formulate_production_strategy`, `execute_scene_strategy_and_refine_loop`)
       when tracks were already completed by the Continuous Production Board (`backlog.json`).
    """
    import asyncio
    from types import SimpleNamespace
    import pytest
    from engine.studio.models import (
        GameRequirementsSpec,
        RequirementTrack,
        StudioPausedError,
        TOKEN_USAGE_TRACKER,
        call_gemini_with_transient_retry,
        install_studio_gemini_guardrails,
        is_studio_paused,
    )
    from engine.studio.orchestrator import StudioOrchestrator

    cand_dir = tmp_path / "cartridges" / "candidate_genesis_test"
    cand_dir.mkdir(parents=True, exist_ok=True)
    pause_file = cand_dir.parent / ".studio_paused.json"
    pause_file.write_text('{"paused": true}', encoding="utf-8")
    monkeypatch.setenv("STUDIO_FAIL_FAST_ON_PAUSE", "1")

    assert is_studio_paused(cand_dir) is True

    # 1a. call_gemini_with_transient_retry is blocked before invoking call_fn
    called = []
    with pytest.raises(StudioPausedError):
        call_gemini_with_transient_retry(
            lambda: called.append(1),
            label="LeadGameplaySWEAgent.pev_phase1",
            candidate_dir=cand_dir,
        )
    assert called == []

    # 1b. Direct client.models.generate_content on a guarded live client is also blocked
    class _LiveClientStub:
        __module__ = "google.genai.client"

        def __init__(self):
            self.models = self

        def generate_content(self, *args, **kwargs):
            called.append(2)
            return SimpleNamespace(text="ok")

    guarded_client = install_studio_gemini_guardrails(_LiveClientStub(), candidate_dir=cand_dir)
    with pytest.raises(StudioPausedError):
        guarded_client.models.generate_content(model="gemini-3.8-flash", contents="hi")
    assert called == []

    # 2. Unpause and verify Runaway Loop Circuit Breaker trips after STUDIO_MAX_CONSECUTIVE_SWE_CALLS
    pause_file.unlink()
    assert is_studio_paused(cand_dir) is False
    monkeypatch.setenv("STUDIO_MAX_CONSECUTIVE_SWE_CALLS", "3")
    TOKEN_USAGE_TRACKER.reset()

    for _ in range(3):
        call_gemini_with_transient_retry(
            lambda: SimpleNamespace(text="ok"),
            label="LeadGameplaySWEAgent.pev_phase1",
            candidate_dir=cand_dir,
        )
    # 4th consecutive SWE PEV call trips circuit breaker and auto-writes .studio_paused.json
    with pytest.raises(StudioPausedError):
        call_gemini_with_transient_retry(
            lambda: SimpleNamespace(text="ok"),
            label="LeadGameplaySWEAgent.pev_phase2",
            candidate_dir=cand_dir,
        )
    assert pause_file.is_file()
    assert is_studio_paused(cand_dir) is True
    pause_file.unlink()
    TOKEN_USAGE_TRACKER.reset()

    # 3. Verify StudioOrchestrator._execute_iterative_swe_loop skips legacy post-board loop when backlog.json exists
    (cand_dir / "project.godot").write_text("[application]\nconfig/name=\"Test\"\n", encoding="utf-8")
    (cand_dir / "game_bible.json").write_text("{\"game_title\": \"Test\"}\n", encoding="utf-8")
    (cand_dir / "backlog.json").write_text("{\"items\": []}\n", encoding="utf-8")

    orch = StudioOrchestrator(api_key="test_key", client=SimpleNamespace())
    orch._is_live_genai_client = lambda: True

    legacy_calls = []
    swe_stub = SimpleNamespace(
        _rendering_strategy_block=lambda d: legacy_calls.append("rendering_strategy"),
        formulate_level_layout=lambda d: legacy_calls.append("level_layout"),
        formulate_production_strategy=lambda d: legacy_calls.append("production_strategy"),
        execute_scene_strategy_and_refine_loop=lambda d, **kw: legacy_calls.append("scene_loop"),
    )
    req_spec = GameRequirementsSpec(
        deliberation="Board verified spec.",
        spec_id="req_test",
        game_title="Test",
        genre_paradigm="simulation",
        high_concept_summary="Test summary",
        requirements=[
            RequirementTrack(
                deliberation="Completed on board.",
                id="mechanic_1",
                title="Track 1",
                description="Desc",
                responsible_agents=["LeadGameplaySWEAgent"],
                target_modules=["main.gd"],
                status="completed",
                acceptance_criteria=[],
            )
        ],
    )
    asyncio.run(
        orch._execute_iterative_swe_loop(
            agent=swe_stub,
            world_spec=None,
            narrative_bible=None,
            req_spec=req_spec,
            staging_dir=cand_dir,
            context=None,
        )
    )
    assert legacy_calls == []


def test_qa_plate_labels_pose_frame_and_causal_filmstrip_invariants(tmp_path: Path):
    """Verifies:
    1. `CreativeWritersRoom._proof_parts` and `_build_task_lead_multimodal_parts` explicitly label
       `plate_*.png` and `anim_sheet_*.png` with `[QA ALPHA INSPECTION PLATE: ...]` and
       `[QA ANIMATION CONTACT SHEET: ...]` so reviewers never confuse external QA checkerboards,
       cyan borders, or yellow baseline lines with in-asset defects.
    2. `VisualArtSpriteDesignerAgent.fabricate_pose_frame_from_base` does not inject unwanted
       rotational twist when `rotation_deg == 0` on custom keyframe tracks and keeps pixels bounded.
    3. `validate_filmstrip_plan` rejects setup-call shortcuts (`set_onboarding_step`, `force_*`,
       `complete_*`) and requires at least one `input_segments` entry for `mechanic_code` / `integration`.
    """
    from PIL import Image
    from engine.studio.models import (
        FilmstripPlan,
        InputSegment,
        KeyframeTransformSpec,
        PartKeyframeTrackSpec,
        QASetupCall,
        RevisionRecord,
        WorkItem,
    )
    from engine.studio.visual import VisualArtSpriteDesignerAgent
    from engine.studio.workers import WorkerContext, _build_task_lead_multimodal_parts, validate_filmstrip_plan
    from engine.studio.writers_room import CreativeWritersRoom

    # 1. Verify QA plate and animation contact sheet captions
    plate_file = tmp_path / "plate_test_asset_rev0.png"
    anim_file = tmp_path / "anim_sheet_test_asset_rev0.png"
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(plate_file)
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(anim_file)

    item = WorkItem(
        deliberation="Test item",
        item_id="visual_test_item",
        kind="visual_asset",
        title="Visual Test",
        spec="Spec",
        proof_spec="Proof",
        revisions=[
            RevisionRecord(
                revision_index=0,
                worker_notes="Done",
                proof_paths=[str(plate_file), str(anim_file)],
            )
        ],
    )
    wr = CreativeWritersRoom(client=None)
    wr_parts = wr._proof_parts(item, candidate_dir=tmp_path)
    wr_labels = [p for p in wr_parts if isinstance(p, str)]
    assert any("QA ALPHA INSPECTION PLATE" in lbl for lbl in wr_labels)
    assert any("QA ANIMATION CONTACT SHEET" in lbl and "yellow/gold baseline" in lbl for lbl in wr_labels)

    ctx = WorkerContext(cartridge_dir=tmp_path, client=None)
    tl_parts = _build_task_lead_multimodal_parts(ctx, [str(plate_file), str(anim_file)])
    tl_labels = [p for p in tl_parts if isinstance(p, str)]
    assert any("QA ALPHA INSPECTION PLATE" in lbl or "QA ANIMATION CONTACT SHEET" in lbl for lbl in tl_labels)

    # 2. Verify fabricate_pose_frame_from_base with custom track & rotation_degrees == 0
    vis_agent = VisualArtSpriteDesignerAgent(client=None)
    base_img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    for y in range(24, 104):
        for x in range(24, 104):
            base_img.putpixel((x, y), (220, 140, 40, 255))
    custom_track = PartKeyframeTrackSpec(
        deliberation="Radial expansion without rotation.",
        part_id="core",
        keyframes=[
            KeyframeTransformSpec(
                deliberation="f0",
                time_normalized=0.0,
                position_offset_px=(0.0, 0.0),
                rotation_degrees=0.0,
                scale_xy=(1.0, 1.0),
            ),
            KeyframeTransformSpec(
                deliberation="f1",
                time_normalized=1.0,
                position_offset_px=(0.0, 0.0),
                rotation_degrees=0.0,
                scale_xy=(1.15, 1.15),
            ),
        ],
    )
    frame_out = vis_agent.fabricate_pose_frame_from_base(
        base_sprite=base_img,
        state_name="pulse",
        frame_idx=1,
        frame_count=4,
        cell_size=(64, 64),
        motion_profile="custom",
        keyframe_tracks=[custom_track],
    )
    assert frame_out.size == (64, 64)

    # 3. Verify validate_filmstrip_plan blocks onboarding/completion setup_calls and empty mechanic inputs
    bad_plan = FilmstripPlan(
        deliberation="Shortcut plan",
        capture_ticks=[20, 60],
        setup_calls=[
            QASetupCall(
                deliberation="Force onboarding step",
                at_tick=10,
                target_node_or_property="main",
                method_name="set_onboarding_step",
                args_json="[2]",
            )
        ],
        input_segments=[],
        frame_expectations=["f1", "f2"],
    )
    issues = validate_filmstrip_plan(bad_plan, ["KEY_1", "KEY_SPACE"], item_kind="mechanic_code")
    assert any("set_onboarding_step" in iss for iss in issues)
    assert any("must include at least one `input_segments`" in iss for iss in issues)

    good_plan = FilmstripPlan(
        deliberation="Authentic causal plan",
        capture_ticks=[15, 60],
        setup_calls=[],
        input_segments=[
            InputSegment(
                deliberation="Click node A then node B",
                from_tick=25,
                to_tick=35,
                actions=[],
                mouse_click_pos=[320.0, 400.0],
            )
        ],
        frame_expectations=["Unsolved baseline at t=15", "Connected state at t=60"],
    )
    assert validate_filmstrip_plan(good_plan, ["KEY_1", "KEY_SPACE"], item_kind="mechanic_code") == []


def test_target_player_dna_persistence_and_agency_depth(tmp_path: Path):
    """Verifies that SteamPlayerDNA is formatted with playtime hours, genre weights, and mastery distribution,
    persisted into GameBibleSpec, and indexed by ContextCuratorAgent."""
    from engine.studio.context_curator import CartridgeSemanticIndex
    from engine.studio.models import CoreLoopSpec, GameBibleSpec
    from engine.studio.writers_room import CreativeWritersRoom

    wr = CreativeWritersRoom(client=object())
    sample_dna = {
        "total_games": 250,
        "total_playtime_hours": 5891.27,
        "pacing_preference": "tactical_methodical",
        "pacing_score": -0.411,
        "challenge_tolerance": 0.4167,
        "playtime_distribution": {"mastered_over_100h": 0.6992, "core_20_to_100h": 0.2319},
        "genre_weights": {"Systems Engineering": 0.68, "Tactical Strategy": 0.32},
        "games": [
            {"name": "Game Alpha", "playtime_hours": 724.4, "genres": ["Simulation"]},
            {"name": "Game Beta", "playtime_hours": 542.6, "genres": ["Strategy"]},
            {"name": "Game Gamma", "playtime_hours": 424.2, "genres": ["Simulation"]},
        ],
    }
    dna_text = wr._format_dna_summary(sample_dna)
    assert "Game Alpha (724.4h" in dna_text
    assert "5891.3 total hours" in dna_text
    assert "tactical_methodical" in dna_text
    assert "TARGET PLAYER OPTIMIZATION MANDATE" in dna_text

    bible = GameBibleSpec(
        deliberation="Test bible",
        game_title="Test Sim",
        elevator_pitch="A systemic builder.",
        core_fantasy="Build and optimize networks.",
        target_player_dna_summary=dna_text,
        core_loop=CoreLoopSpec(
            deliberation="Loop",
            moment_to_moment_loop="Place and route components",
            minute_to_minute_loop="Expand throughput and clear bottlenecks",
            session_loop="Scale multi-stage networks",
            player_verbs=["place", "route", "tune"],
            feedback_channels=["Throughput gauge", "Chime pitch"],
            player_agency_and_optimization_depth="Open-ended network construction with continuous throughput optimization.",
        ),
    )
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    idx = CartridgeSemanticIndex(tmp_path, client=object())
    chunks = idx.extract_corpus_chunks(ensure_rendering=False)
    chunk_ids = {c.chunk_id for c in chunks}
    assert "design:target_player_dna_summary" in chunk_ids
    assert "design:core_loop" in chunk_ids
    core_chunk = next(c for c in chunks if c.chunk_id == "design:core_loop")
    assert "Open-ended network construction" in core_chunk.content


def test_gemini_dataset_labels_and_store_injection(tmp_path):
    from google.genai import types
    from engine.studio.models import (
        build_gemini_request_labels,
        inject_gemini_dataset_config,
        sanitize_gemini_label_value,
        set_active_gemini_dataset_context,
    )

    cand_dir = tmp_path / "candidate_genesis_dda57b"
    cand_dir.mkdir(parents=True)
    set_active_gemini_dataset_context(
        dataset_name="Run #1: Lithic Harmonic!!",
        candidate_dir=cand_dir,
        store_logs=True,
    )

    assert sanitize_gemini_label_value("Run #1: Lithic Harmonic!!") == "run_1_lithic_harmonic"
    labels = build_gemini_request_labels(
        label="ContextCurator.curate_context[mechanic_photoelastic_stress_shader]",
        candidate_dir=cand_dir,
    )
    assert labels["dataset"] == "run_1_lithic_harmonic"
    assert labels["run_id"] == "run_1_lithic_harmonic"
    assert labels["cartridge"] == "candidate_genesis_dda57b"
    assert labels["agent"] == "contextcurator"
    assert labels["operation"].startswith("contextcurator_curate_context")
    assert all(len(v) <= 63 for v in labels.values())

    # 1. Inject into GenerateContentConfig (text/code reasoning call) -> sets labels AND store=True
    base_cfg = types.GenerateContentConfig(temperature=0.3, labels={"custom_tag": "my_val"})
    kw = inject_gemini_dataset_config(
        {"model": "gemini-3.8-flash", "contents": "Hello", "config": base_cfg},
        label="LeadGameplaySWEAgent.pev_phase2_code_synthesis",
        candidate_dir=cand_dir,
    )
    injected_cfg = kw["config"]
    assert injected_cfg.labels["dataset"] == "run_1_lithic_harmonic"
    assert injected_cfg.labels["agent"] == "leadgameplaysweagent"
    assert injected_cfg.labels["custom_tag"] == "my_val"
    assert injected_cfg.http_options is not None
    assert injected_cfg.http_options.extra_body == {"store": True}

    # 2. Image / TTS media calls receive labels but do NOT set store=True (unsupported by AI Studio logging)
    img_cfg = types.GenerateContentConfig(response_modalities=["IMAGE"])
    img_kw = inject_gemini_dataset_config(
        {"model": "gemini-3.1-flash-lite-image", "contents": "Sprite", "config": img_cfg},
        label="VisualArtSpriteDesignerAgent.generate_sprite",
        candidate_dir=cand_dir,
    )
    assert img_kw["config"].labels["dataset"] == "run_1_lithic_harmonic"
    assert img_kw["config"].http_options is None


def test_agent_self_inquiry_question_research_and_answer_loop(tmp_path):
    from engine.studio.context_curator import (
        ContextCuratorAgent,
        ContextCuratorDecision,
        QuestionContextResearchDecision,
    )
    from engine.studio.models import (
        GameRequirementsSpec,
        ModuleFileSpec,
        RequirementTrack,
        TrackCodeSynthesisPayload,
    )
    from engine.studio.swe import LeadGameplaySWEAgent

    # Set up a cartridge with a helper script in scripts/pneumatic_bus.gd that main.gd does NOT yet import
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "pneumatic_bus.gd").write_text(
        "extends Node\nclass_name PneumaticBus\nsignal pressure_stabilized(node_id: String, kpa: float)\nfunc lock_aperture(ratio: float) -> void:\n\tpass\n",
        encoding="utf-8",
    )
    (tmp_path / "main.gd").write_text("extends Node2D\nfunc _ready() -> void:\n\tpass\n", encoding="utf-8")

    # Fake live genai client that simulates:
    # 1) Initial ContextCuratorDecision (does NOT include scripts/pneumatic_bus.gd)
    # 2) SWE Turn 0: SWE asks a self_question about PneumaticBus instead of guessing (files=[])
    # 3) QuestionContextResearchDecision: ContextCuratorAgent selects code:module:scripts/pneumatic_bus.gd (without answering the question itself)
    # 4) SWE Turn 1: SWE sees the full source of scripts/pneumatic_bus.gd, answers its own question in self_question_answer, and outputs files=[...]
    prompts_seen = []

    class _FakeModels:
        def embed_content(self, model, contents, config=None):
            class _Emb:
                def __init__(self, text):
                    t = str(text).lower()
                    self.values = [
                        1.0 if "pneumatic_bus" in t or "pneumaticbus" in t or "lock_aperture" in t else 0.1,
                        1.0 if "main.gd" in t else 0.1,
                        0.5,
                    ]
            return type("Resp", (), {"embeddings": [_Emb(c) for c in contents]})()

        def generate_content(self, model, contents, config=None):
            prompt_str = str(contents[0] if isinstance(contents, list) else contents)
            prompts_seen.append(prompt_str)
            schema = getattr(config, "response_schema", None)
            if schema is ContextCuratorDecision:
                payload = ContextCuratorDecision(
                    deliberation="Select main.gd initially.",
                    search_queries_used=["main.gd"],
                    role_tailored_synthesis="Initial context pack.",
                    selected_chunk_ids=["code:module:main.gd"],
                    selected_image_paths=[],
                ).model_dump_json()
                return type("Resp", (), {"text": payload})()
            if schema is QuestionContextResearchDecision:
                payload = QuestionContextResearchDecision(
                    deliberation="Select code:module:scripts/pneumatic_bus.gd so SWE can inspect PneumaticBus signals and methods.",
                    selected_chunk_ids=["code:module:scripts/pneumatic_bus.gd"],
                ).model_dump_json()
                return type("Resp", (), {"text": payload})()
            if schema is TrackCodeSynthesisPayload:
                if "CONTEXT RESEARCHED FOR YOUR SELF-QUESTION" not in prompt_str:
                    # Turn 0: SWE realizes it needs PneumaticBus's exact signal & method signature
                    payload = TrackCodeSynthesisPayload(
                        deliberation="I need to check the exact signal and method signatures in scripts/pneumatic_bus.gd before implementing.",
                        self_question="What signal and method signatures does scripts/pneumatic_bus.gd define on PneumaticBus?",
                        files=[],
                    ).model_dump_json()
                    return type("Resp", (), {"text": payload})()
                else:
                    # Turn 1: Verify the full source of scripts/pneumatic_bus.gd was brought in by ContextCuratorAgent
                    assert "signal pressure_stabilized(node_id: String, kpa: float)" in prompt_str
                    assert "func lock_aperture(ratio: float)" in prompt_str
                    payload = TrackCodeSynthesisPayload(
                        deliberation="Now that I have inspected scripts/pneumatic_bus.gd, I can answer my question and implement main.gd.",
                        self_question_answer="PneumaticBus defines `signal pressure_stabilized(node_id: String, kpa: float)` and `func lock_aperture(ratio: float) -> void`.",
                        self_question="",
                        files=[
                            ModuleFileSpec(
                                deliberation="Wire PneumaticBus.lock_aperture in main.gd",
                                filename="main.gd",
                                content="extends Node2D\nfunc _ready() -> void:\n\tvar bus = PneumaticBus.new()\n\tbus.lock_aperture(0.5)\n",
                            )
                        ],
                    ).model_dump_json()
                    return type("Resp", (), {"text": payload})()
            raise AssertionError(f"Unexpected schema: {schema}")

    fake_live_client = type("Client", (), {"models": _FakeModels()})()
    fake_live_client.__class__.__module__ = "google.genai.client"

    swe = LeadGameplaySWEAgent(client=fake_live_client)
    swe._verify_and_heal_godot_syntax_and_probes = lambda c_dir: {}
    track = RequirementTrack(
        deliberation="Test self-inquiry",
        id="mechanic_pneumatic_tuner",
        title="Pneumatic Tuner",
        description="Connect main.gd to PneumaticBus.",
        target_modules=["main.gd"],
    )
    req_spec = GameRequirementsSpec(
        deliberation="Spec",
        game_title="Test",
        genre_paradigm="simulation",
        engine_type="godot4",
        requirements=[track],
    )

    written = swe.implement_requirement_track(track=track, staging_dir=tmp_path, requirements_spec=req_spec)
    assert "main.gd" in written
    assert "bus.lock_aperture(0.5)" in (tmp_path / "main.gd").read_text(encoding="utf-8")

    # Verify the Q->A pair was saved to persistent self-inquiry memory and formats cleanly for subsequent turns
    curator = ContextCuratorAgent(cartridge_dir=tmp_path, client=fake_live_client)
    saved_qa = curator.load_self_inquiry_memory("mechanic_pneumatic_tuner", "swe_engineer")
    assert len(saved_qa) == 1
    assert "What signal and method signatures does scripts/pneumatic_bus.gd define" in saved_qa[0].question
    assert "pressure_stabilized" in saved_qa[0].answer
    assert "code:module:scripts/pneumatic_bus.gd" in saved_qa[0].researched_chunk_ids

    mem_section = curator.format_self_inquiry_prompt_section(saved_qa, pending_pack=None)
    assert "SELF-INQUIRY MEMORY (QUESTIONS YOU ALREADY ASKED & ANSWERED)" in mem_section
    assert "lock_aperture" in mem_section


def test_board_item_chronology_endpoint_and_turn_code_diffs(tmp_path):
    import json
    from starlette.testclient import TestClient
    from engine.studio.workers import _compute_turn_code_diffs, _snapshot_modules
    from engine.ui.server import create_app

    cand_dir = tmp_path / "candidate_chron_test"
    cand_dir.mkdir(parents=True)
    (cand_dir / "main.gd").write_text("extends Node2D\nfunc _ready() -> void:\n\tpass\n", encoding="utf-8")
    snap = _snapshot_modules(cand_dir, ["main.gd"])
    (cand_dir / "main.gd").write_text(
        "extends Node2D\nfunc _ready() -> void:\n\tvar x = 42\n",
        encoding="utf-8",
    )
    diffs = _compute_turn_code_diffs(cand_dir, snap)
    assert "main.gd" in diffs
    assert "+	var x = 42" in diffs["main.gd"]

    item_dir = cand_dir / "artifacts" / "work_items" / "mechanic_tuner"
    item_dir.mkdir(parents=True)
    proof_png = item_dir / "filmstrip_mechanic_tuner_rev0.png"
    proof_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    (item_dir / "lead_trace_rev0.json").write_text(
        json.dumps(
            {
                "item_id": "mechanic_tuner",
                "revision": 0,
                "status": "submitted",
                "current_strategy": "Implement and verify tuner in main.gd",
                "submission_summary": "Verified tuner",
                "collected_proofs": [str(proof_png)],
                "turns": [
                    {
                        "deliberation": "SWE Turn",
                        "turn_index": 0,
                        "specialist_agent": "swe_engineer",
                        "brief": "Add var x = 42 in main.gd",
                        "summary": "Implemented main.gd",
                        "touched_files": ["main.gd"],
                        "proof_images": [],
                        "verification_passed": True,
                        "code_diffs": diffs,
                    },
                    {
                        "deliberation": "QA Turn",
                        "turn_index": 1,
                        "specialist_agent": "qa_test_engineer",
                        "brief": "Capture filmstrip",
                        "summary": "Captured 5 frames",
                        "touched_files": [],
                        "proof_images": [str(proof_png)],
                        "verification_passed": True,
                        "code_diffs": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (cand_dir / "backlog.json").write_text(
        json.dumps(
            {
                "design_revision": 1,
                "items": [
                    {
                        "item_id": "mechanic_tuner",
                        "title": "Acoustic Tuner",
                        "kind": "mechanic_code",
                        "state": "changes_requested",
                        "spec": "1) Hover conduit. 2) Drag dial.",
                        "proof_spec": "Verify dial rotation.",
                        "target_modules": ["main.gd"],
                        "depends_on": [],
                        "revisions": [
                            {
                                "revision_index": 0,
                                "verdict": "request_changes",
                                "approved": False,
                                "feedback": "[GameplayDesigner]: 1. Add physical conduit. 2. Fix text clarity. | [VisualArtDirector]: 1. Render brass dial.",
                                "proof_paths": [str(proof_png)],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    app = create_app(cartridges_root=tmp_path, active_cartridge_dir=cand_dir, db_path=tmp_path / "telemetry.db")
    with TestClient(app) as client:
        res = client.get("/api/studio/board-item/mechanic_tuner/chronology?cartridge=candidate_chron_test")
        assert res.status_code == 200
        payload = res.json()
        assert payload["item_id"] == "mechanic_tuner"
        assert len(payload["attempts"]) == 1
        att0 = payload["attempts"][0]
        assert len(att0["turns"]) == 2
        assert "+	var x = 42" in att0["turns"][0]["code_diffs"]["main.gd"]
        assert len(att0["council_review"]["reviewer_breakdown"]) == 2
        assert att0["council_review"]["reviewer_breakdown"][0]["reviewer_role"] == "GameplayDesigner"
        assert "main.gd" in payload["file_contents"]


def test_active_build_process_inspection_and_pause_state(tmp_path: Path):
    """Server actively inspects process/threads/locks and persists paused state across page refreshes."""
    from starlette.testclient import TestClient
    from engine.ui.server import create_app

    cand_dir = tmp_path / "candidate_proc_test"
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "game_bible.json").write_text(
        json.dumps({"game_title": "Process Inspection Test", "status": "building"}),
        encoding="utf-8",
    )

    app = create_app(cartridges_root=tmp_path, active_cartridge_dir=cand_dir, db_path=tmp_path / "telemetry.db")
    with TestClient(app) as client:
        # Initially stopped and not paused
        s0 = client.get("/api/studio/settings").json()
        assert s0["paused"] is False
        assert "process_status" in s0
        assert s0["process_status"]["is_process_alive"] is False
        assert s0["process_status"]["build_actively_running"] is False
        assert s0["process_status"]["process_state"] == "stopped"

        # Pause build via POST /api/studio/settings
        s1 = client.post(
            "/api/studio/settings",
            json={
                "deliberation": "Operator paused build",
                "pause_for_greenlight": True,
                "paused": True,
            },
        ).json()
        assert s1["paused"] is True
        assert s1["process_status"]["is_paused"] is True
        assert s1["process_status"]["process_state"] == "paused"
        assert s1["process_status"]["build_actively_running"] is False

        # Refreshing /api/events/history and /api/studio/game-bible preserves paused process_status
        hist = client.get("/api/events/history").json()
        assert hist["process_status"]["is_paused"] is True
        assert hist["process_status"]["build_actively_running"] is False

        bible = client.get("/api/studio/game-bible?cartridge=candidate_proc_test").json()
        assert bible["build_paused"] is True
        assert bible["build_actively_running"] is False
        assert bible["process_status"]["is_paused"] is True


def test_task_lead_generic_multi_cell_asset_and_proof_strategy(tmp_path: Path):
    """Verifies that:
    1. Every task (including Revision 0 `visual_asset` tasks) runs through the Task Lead Strategist loop
       (`plan_task_lead_turn`), formulating `current_strategy` and `proof_strategy`, delegating to `visual_artist`,
       inspecting the resulting `plate_*.png` inspection plate, and submitting to the Council.
    2. When the Task Lead's strategy delegates a multi-cell grid/sheet (`grid_cols=2, grid_rows=2, cell_prompts=[...]`),
       `CartridgeAssetLibrary.fabricate_single_asset` synthesizes all 4 cells into a `2x2` sheet with zero cross-cell
       bleed, records `hframes=2, vframes=2` in `AssetLibraryEntry`, renders a `plate_*.png` inspection plate with
       2x2 grid lines, and preserves all 4 cells during `verify_rendered_asset` and `verify_all_catalog_assets_and_animations`.
    """
    import numpy as np
    from engine.studio.asset_library import CartridgeAssetLibrary
    from engine.studio.workers import WorkerContext, task_lead_handler

    (tmp_path / "game_bible.json").write_text(json.dumps({"game_title": "MultiCellStrategyGame"}), encoding="utf-8")

    lead_turns = iter([
        json.dumps(
            {
                "deliberation": "Task requires a 4-state indicator sheet; delegating to visual_artist with a 2x2 grid strategy.",
                "current_strategy": "Delegate 2x2 multi-cell sheet fabrication to visual_artist, then inspect plate.",
                "proof_strategy": "Render a 2x2 multi-cell alpha inspection plate with cell dividers.",
                "specialist_agent": "visual_artist",
                "brief": "Fabricate a 2x2 multi-cell indicator sheet (grid_cols=2, grid_rows=2) with 4 distinct states in cell_prompts.",
                "target_paths": ["assets/sprites/multi_state_indicator.png"],
                "action": "delegate",
            }
        ),
        json.dumps(
            {
                "deliberation": "Inspected plate_multi_state_indicator.png; all 4 cells are populated and cleanly separated.",
                "current_strategy": "Submit verified 2x2 multi-cell indicator plate to Council.",
                "proof_strategy": "Submit plate_multi_state_indicator.png showing all 4 grid cells.",
                "submission_summary": "Fabricated and verified 2x2 multi-state indicator sheet with hframes=2, vframes=2.",
                "action": "submit_to_council",
            }
        ),
    ])

    lead_prompts = []

    def responder(contents):
        text = contents if isinstance(contents, str) else "\n".join(c for c in contents if isinstance(c, str))
        if "Task Lead Strategist" in text:
            lead_prompts.append(contents)
            return next(lead_turns)
        return bw.req_json(
            "multi_state_indicator",
            mode="diegetic_ui",
            asset_type="diegetic_ui",
            target="assets/sprites/multi_state_indicator.png",
            dimensions=[256, 256],
            grid_cols=2,
            grid_rows=2,
            cell_prompts=[
                "State 1: circular neutral indicator ring",
                "State 2: diamond active lock bracket",
                "State 3: hexagonal warning indicator",
                "State 4: octagonal overload emblem",
            ],
        )

    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=bw.make_client(responder),
        library_factory=lambda c: CartridgeAssetLibrary(cartridge_dir=c.cartridge_dir, client=None),
    )
    item = bw.item("multi_state_indicator", kind="visual_asset")
    res = task_lead_handler(ctx)(item)

    # 1. Task Lead ran 2 turns (Turn 0 delegate -> Turn 1 inspect plate & submit) and persisted proof_strategy
    assert len(lead_prompts) == 2
    trace_data = json.loads(
        (tmp_path / "artifacts" / "work_items" / "multi_state_indicator" / "lead_trace_rev0.json").read_text(encoding="utf-8")
    )
    assert trace_data["status"] == "submitted"
    assert "plate_multi_state_indicator.png" in trace_data["proof_strategy"]

    # 2. Turn 1 Task Lead prompt and multimodal parts include plate_multi_state_indicator.png (not just the raw sprite)
    from engine.studio.workers import _build_task_lead_multimodal_parts
    assert "plate_multi_state_indicator.png" in str(lead_prompts[1])
    mm_parts = _build_task_lead_multimodal_parts(ctx, res.proof_images)
    mm_strs = [p for p in mm_parts if isinstance(p, str)]
    assert any("CAPTURED PROOF IMAGE (plate_multi_state_indicator.png)" in s for s in mm_strs)

    # 3. Fabricated 256x256 sheet has non-empty pixels in all 4 quadrants and records hframes=2, vframes=2
    sheet_path = tmp_path / "assets" / "sprites" / "multi_state_indicator.png"
    assert sheet_path.is_file()
    with Image.open(sheet_path) as im:
        arr = np.asarray(im.convert("RGBA"))
    assert arr.shape == (256, 256, 4)
    for r_i in range(2):
        for c_i in range(2):
            quad_alpha = arr[r_i * 128:(r_i + 1) * 128, c_i * 128:(c_i + 1) * 128, 3]
            assert int(np.count_nonzero(quad_alpha > 24)) > 50, f"Quadrant ({r_i}, {c_i}) was unexpectedly empty"

    cat_data = json.loads((tmp_path / "assets" / "asset_library.json").read_text(encoding="utf-8"))
    entry = cat_data["entries"]["multi_state_indicator"]
    assert entry["hframes"] == 2
    assert entry["vframes"] == 2
    assert "hframes=2, vframes=2" in res.notes

    # 4. Running verify_all_catalog_assets_and_animations preserves all 4 quadrants without single-subject isolation
    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path, client=None)
    audit = lib.verify_all_catalog_assets_and_animations(auto_remediate=True)
    assert audit["all_passed"] is True
    with Image.open(sheet_path) as im_after:
        arr_after = np.asarray(im_after.convert("RGBA"))
    for r_i in range(2):
        for c_i in range(2):
            quad_alpha = arr_after[r_i * 128:(r_i + 1) * 128, c_i * 128:(c_i + 1) * 128, 3]
            assert int(np.count_nonzero(quad_alpha > 24)) > 50, f"Quadrant ({r_i}, {c_i}) erased during catalog verification"

















