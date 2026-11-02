"""tests/test_agent_routing_and_gate_removal.py
Unit and integration tests verifying:
1. Elimination of hardcoded 5-Gate QA blocking & `qa_regen_*` task stamping on Continuous Production Board builds
   (`CreativeWritersRoom._run_slice_qa_gate_check` disabled by default; `StudioOrchestrator` promotes completed
   board cartridges via boot syntax check without calling `evaluate_and_heal` or `build_qa_gate_work_item_proposals`).
2. Dynamic LLM-driven peer re-routing across all 5 Execution Specialist Personas (`visual_artist`, `audio_designer`,
   `script_writer`, `swe_engineer`, `qa_test_engineer`), including cross-domain escalation on rev-0 `visual_asset`
   items, post-inspection re-routing via `VisualEvaluationReport`, and cycle-safe hop guards.
3. Dynamic Council Steering routing (`CouncilSteeringRoutingPlan` + multimodal asset contact sheet attachment),
   `WritersRoomTurnProposal.route_to_council_role` peer handoffs in both `_execute_steering_pivot_turns` and the
   main Writers' Room discussion loop, and read-only specialist rerouting inside `_run_council_diagnostic_probe`.
4. Active-vs-unreferenced visual asset visibility (`_collect_observational_asset_metadata`,
   `render_cartridge_asset_contact_sheet`, `_build_asset_integrity_dossier_and_sheet`) and sandboxed orphan file
   cleanup (`apply_sandboxed_cartridge_deletions`, `visual_artist` `operation="cleanup"`, `swe_engineer` `delete_paths`).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from engine.studio.models import (
    CouncilDiagnosticProbe,
    CouncilSteeringRoutingPlan,
    DirectorConvergenceCheck,
    ExperienceStoryboardBeat,
    FilmstripPlan,
    GameBibleSpec,
    GameSliceSpec,
    ReviewBatch,
    ReviewDecision,
    StudioAgentError,
    SurgicalCodePatch,
    TaskLeadDecision,
    TrackCodeSynthesisPayload,
    VisualArtistPlan,
    VisualEvaluationReport,
    WorkItem,
    WorkItemProposal,
    WritersRoomListenCheck,
    WritersRoomTurnProposal,
    apply_sandboxed_cartridge_deletions,
)
from engine.studio.production_board import BoardScheduler, ProductionBoard, WorkerResult
from engine.studio.swe import LeadGameplaySWEAgent
from engine.studio.workers import (
    WorkerContext,
    _dispatch_to_specialist,
    available_specialists,
    run_task_lead,
)
from engine.studio.writers_room import CreativeWritersRoom
from engine.supervisor.asset_validator import (
    _collect_observational_asset_metadata,
    render_cartridge_asset_contact_sheet,
)


def _make_rgba_png(
    path: Path,
    size: tuple[int, int] = (64, 64),
    color: tuple[int, int, int, int] = (60, 150, 220, 255),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle((4, 4, size[0] - 5, size[1] - 5), fill=color, outline=(255, 255, 255, 255), width=2)
    img.save(path, format="PNG")
    return path


def _deterministic_embed(text: str, dim: int = 768) -> List[float]:
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9_]+", str(text).lower()):
        h = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:4], "big")
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class _QueuedFakeClient:
    """Fast deterministic fake Gemini client supporting queued structured JSON responses and ContextCurator."""

    __module__ = "google.genai.fake"
    supports_specialist_plans = True
    supports_steering_routing = False
    enable_post_capture_audit = False

    def __init__(self, responses: List[str], *, steering_routing: bool = False):
        self._queue = list(responses)
        self.supports_steering_routing = steering_routing
        self.calls: List[Dict[str, Any]] = []
        self.models = SimpleNamespace(
            generate_content=self._generate_content,
            embed_content=self._embed_content,
        )

    def _embed_content(self, model: str, contents: Any, config: Any = None) -> Any:
        items = list(contents) if isinstance(contents, (list, tuple)) else [str(contents)]
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=_deterministic_embed(str(t))) for t in items]
        )

    def _generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
        schema_obj = getattr(config, "response_schema", None)
        schema_name = getattr(schema_obj, "__name__", str(schema_obj or ""))
        if schema_name == "ContextCuratorDecision" or "role_tailored_synthesis" in str(schema_obj):
            cur_json = json.dumps(
                {
                    "deliberation": "Curating focused cartridge context.",
                    "search_queries_used": ["entrypoint"],
                    "role_tailored_synthesis": "Curated entrypoint and asset context.",
                    "selected_chunk_ids": [],
                    "selected_image_paths": [],
                    "selected_media_paths": [],
                    "sufficient_context": True,
                }
            )
            return SimpleNamespace(text=cur_json, parsed=None, candidates=[])
        assert self._queue, f"Unexpected extra LLM call for schema {schema_name}"
        raw = self._queue.pop(0)
        self.calls.append({"model": model, "contents": contents, "schema": schema_name})
        return SimpleNamespace(text=raw, parsed=None, candidates=[])


def _seed_minimal_cartridge(c_dir: Path) -> None:
    c_dir.mkdir(parents=True, exist_ok=True)
    (c_dir / "project.godot").write_text(
        '[application]\nconfig/name="SyntheticStudioGame"\nrun/main_scene="res://main.tscn"\n',
        encoding="utf-8",
    )
    (c_dir / "main.tscn").write_text(
        '[gd_scene load_steps=2 format=3]\n'
        '[ext_resource type="Script" path="res://main.gd" id="1"]\n'
        '[node name="Main" type="Node2D"]\n'
        'script = ExtResource("1")\n',
        encoding="utf-8",
    )
    (c_dir / "main.gd").write_text(
        "extends Node2D\n"
        "var sprite_hframes: int = 1\n"
        "func _ready() -> void:\n"
        "\tvar _tex = load('res://assets/sprites/z_active_hero.png')\n",
        encoding="utf-8",
    )


def test_slice_drain_and_board_promotion_skip_hardcoded_qa_gate(tmp_path: Path):
    """Verifies that:
    1. `_drain_backlog` does NOT invoke hardcoded 5-Gate `_run_slice_qa_gate_check` / `build_qa_gate_work_item_proposals`
       by default even when `is_real_genai_client` is True.
    2. `StudioOrchestrator` promotes a Continuous Production Board cartridge (`backlog.json` present) via boot
       entrypoint sanity check without calling `BuildSupervisor.evaluate_and_heal` or `build_qa_gate_work_item_proposals`.
    """
    cand_dir = tmp_path / "cartridges" / "candidate_genesis_board01"
    _seed_minimal_cartridge(cand_dir)

    board = ProductionBoard(cand_dir / "backlog.json")
    seeded = board.add_items(
        [
            WorkItemProposal(
                deliberation="Seed initial gameplay task.",
                item_id="core_loop_01",
                kind="mechanic_code",
                title="Core Loop",
                spec="Implement core loop.",
                proof_spec="Verify tick.",
                target_modules=["main.gd"],
                slice_id="slice_0",
            )
        ],
        raised_by_role="GameplaySystemsEngineer",
        slice_id="slice_0",
    )[0]
    claimed = board.claim_ready()
    proof_img = _make_rgba_png(cand_dir / "artifacts" / "proofs" / "core_loop_01.png")
    board.submit_result(claimed[0].item_id, WorkerResult(proof_images=[str(proof_img)], notes="Built core loop"))
    board.apply_review(
        ReviewDecision(
            deliberation="Approved.",
            validation_criteria_for_my_role="Playable.",
            item_id=seeded.item_id,
            verdict="approve",
            feedback="Looks great.",
        )
    )

    scheduler = BoardScheduler(board, {"mechanic_code": lambda _it: WorkerResult(proof_images=[], notes="ok")}, max_workers=1)
    room = CreativeWritersRoom(client=object(), board=board, scheduler=scheduler)
    room.max_assembly_passes = 0
    room.max_blind_playtest_passes = 0

    bible = GameBibleSpec(
        deliberation="Synthetic bible.",
        cartridge_id=cand_dir.name,
        target_episode=0,
        status="in_production",
        game_title="Synthetic Studio Game",
        core_genre="action",
        slice_roadmap=[
            GameSliceSpec(
                deliberation="Slice 0.",
                slice_id="slice_0",
                slice_index=0,
                title="Slice 0",
                slice_objective="Core loop.",
                player_acclimation_and_mechanics_focus="Controls.",
                status="active",
            )
        ],
        active_slice_index=0,
    )
    (cand_dir / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    # Even when is_real_genai_client returns True, default _run_slice_qa_gate_check returns 0 without running BuildSupervisor
    with (
        patch("engine.studio.writers_room.is_real_genai_client", return_value=True),
        patch("engine.supervisor.supervisor.BuildSupervisor.evaluate_candidate_cartridge") as spy_eval_cand,
        patch("engine.supervisor.supervisor.build_qa_gate_work_item_proposals") as spy_qa_props,
    ):
        try:
            room._drain_backlog(bible, "Synthetic DNA", cand_dir)
        finally:
            scheduler.stop()
        assert spy_eval_cand.call_count == 0
        assert spy_qa_props.call_count == 0
        assert bible.slice_roadmap[0].status == "completed"

    # Verify BuildSupervisor boot check + promotion path used by StudioOrchestrator on board cartridges
    from engine.supervisor.supervisor import BuildSupervisor

    sup = BuildSupervisor(
        cartridges_root=cand_dir.parent,
        active_cartridge_dir=cand_dir.parent / "active",
    )
    with (
        patch.object(sup, "evaluate_and_heal", side_effect=AssertionError("Hardcoded 5-Gate evaluate_and_heal must not run!")),
    ):
        gate_a = sup.verify_gate_a_syntax_and_entrypoint(cand_dir)
        assert gate_a.passed is True
        sup.promote_candidate(cand_dir)
        assert (cand_dir.parent / "active" / "main.gd").is_file()


def test_specialist_peer_rerouting_across_domains_and_cycle_guard(tmp_path: Path):
    """Verifies that:
    1. A rev-0 `visual_asset` task dispatched to `visual_artist` can dynamically re-route (`operation='reroute'`
       or post-`inspect` `VisualEvaluationReport.reroute_to_specialist='swe_engineer'`) to `swe_engineer`,
       executing the GDScript fix cleanly, emitting `backlog_item_specialist_rerouted`, recording
       `rerouted_from=['visual_artist']`, and unlocking `swe_engineer` + `qa_test_engineer` in `available_specialists`.
    2. `qa_test_engineer` (`verification_mode='reroute'`) and `swe_engineer` (`operation_mode='reroute'`)
       re-route cleanly, and a cyclic `A -> B -> A` re-route stops safely without infinite recursion.
    """
    _seed_minimal_cartridge(tmp_path)
    hero_png = _make_rgba_png(tmp_path / "assets" / "sprites" / "z_active_hero.png", size=(128, 64))

    # Case 1: Post-inspection re-route on a rev-0 `visual_asset` item:
    # Turn 0 Task Lead -> `visual_artist` (`inspect`) -> `VisualEvaluationReport` discovers PNG is already a clean
    # 2-frame strip (`128x64`) and re-routes to `swe_engineer` -> `swe_engineer` patches `sprite_hframes: int = 2` ->
    # Turn 1 Task Lead sees `swe_engineer` unlocked and finishes!
    tl_turn_0 = TaskLeadDecision(
        deliberation="Inspect hero sprite first before regenerating.",
        current_strategy="Delegate 1:1 visual inspection to visual_artist before regenerating pixels.",
        action="delegate",
        specialist_agent="visual_artist",
        brief="Inspect assets/sprites/z_active_hero.png at 1:1 resolution.",
        target_paths=["assets/sprites/z_active_hero.png"],
    ).model_dump_json()
    vis_inspect_plan = VisualArtistPlan(
        deliberation="Inspect existing hero PNG before touching pixels.",
        target_asset_paths=["assets/sprites/z_active_hero.png"],
        grid_px=64,
        operation="inspect",
    ).model_dump_json()
    vis_eval_reroute = VisualEvaluationReport(
        deliberation="The PNG on disk is already a clean 2-frame 128x64 horizontal strip; the bug is main.gd using sprite_hframes = 1.",
        observed_visual_properties="Two crisp 64x64 character frames side-by-side on transparent alpha.",
        reroute_to_specialist="swe_engineer",
        reroute_brief="Update `sprite_hframes` from 1 to 2 in `main.gd` so the 2-frame strip renders cleanly.",
        reroute_target_paths=["main.gd"],
        answers_brief_satisfactorily=True,
    )
    swe_fix_payload = TrackCodeSynthesisPayload(
        deliberation="Applying 1-line surgical patch to set sprite_hframes = 2 in main.gd.",
        patches=[
            SurgicalCodePatch(
                deliberation="Update sprite_hframes from 1 to 2.",
                filename="main.gd",
                start_line=2,
                end_line=2,
                replacement_code="var sprite_hframes: int = 2",
            )
        ],
        implementation_summary="Updated `sprite_hframes` to 2 in `main.gd`.",
        operation_mode="edit",
    ).model_dump_json()
    tl_turn_1 = TaskLeadDecision(
        deliberation="Visual Artist inspected the sheet, routed to SWE Engineer who fixed `sprite_hframes = 2`, and the 1:1 inspection plate proves clean frames.",
        current_strategy="Submit verified 2-frame strip inspection plate and GDScript hframes fix to Council.",
        action="submit_to_council",
        submission_summary="Verified 2-frame hero strip and updated `main.gd` `sprite_hframes = 2` via peer reroute.",
    ).model_dump_json()

    from engine.studio.visual_evaluator import VisualEvaluatorAgent

    client = _QueuedFakeClient(
        [tl_turn_0, vis_inspect_plan, vis_eval_reroute.model_dump_json(), swe_fix_payload, tl_turn_1]
    )
    swe_agent = LeadGameplaySWEAgent(client=client)
    events: List[tuple[str, Dict[str, Any]]] = []

    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=client,
        swe_agent=swe_agent,
        visual_evaluator=VisualEvaluatorAgent(client=client),
        code_verify=lambda _p, _t: None,
        filmstrip_fn=lambda *_a, **_k: hero_png,
        event_callback=lambda et, pl: events.append((et, pl)),
    )
    vis_item = WorkItem(
        deliberation="Hero sprite appears squished/clipped in-game.",
        item_id="fix_hero_sprite",
        kind="visual_asset",
        title="Fix Hero Sprite Rendering",
        spec="Ensure z_active_hero.png renders without clipping.",
        proof_spec="1:1 sprite plate and updated hframes.",
    )
    assert available_specialists(ctx, vis_item) == ["visual_artist"]

    res = run_task_lead(ctx, vis_item)
    assert "sprite_hframes: int = 2" in (tmp_path / "main.gd").read_text(encoding="utf-8")
    assert hero_png.is_file()
    reroute_events = [pl for et, pl in events if et == "backlog_item_specialist_rerouted"]
    assert len(reroute_events) == 1
    assert reroute_events[0]["from_specialist"] == "visual_artist"
    assert reroute_events[0]["to_specialist"] == "swe_engineer"
    assert reroute_events[0]["reasoning_rationale"]
    assert res.proof_images, "Inspection plate from visual_artist hop should be preserved across reroute!"

    # Verify turn_history with rerouted_from unlocks cross-domain specialists on rev-0 visual_asset
    unlocked = available_specialists(
        ctx,
        vis_item,
        turn_history=[SimpleNamespace(turn_index=0, specialist_agent="swe_engineer", rerouted_from=["visual_artist"])],
    )
    assert "swe_engineer" in unlocked
    assert "qa_test_engineer" in unlocked

    # Case 2: Cycle guard (`qa_test_engineer` -> `swe_engineer` -> `qa_test_engineer` stops cleanly)
    qa_reroute = FilmstripPlan(
        deliberation="Need SWE Engineer to expose state before QA capture; rerouting.",
        rationale="Rerouting to swe_engineer.",
        reroute_to_specialist="swe_engineer",
        reroute_brief="Check main.gd state.",
        reroute_target_paths=["main.gd"],
        verification_mode="reroute",
    ).model_dump_json()
    swe_reroute_back = TrackCodeSynthesisPayload(
        deliberation="Attempting to bounce back to already-visited qa_test_engineer.",
        implementation_summary="Rerouting back to qa_test_engineer.",
        reroute_to_specialist="qa_test_engineer",
        reroute_brief="Bounce back.",
        operation_mode="reroute",
    ).model_dump_json()

    cycle_client = _QueuedFakeClient([qa_reroute, swe_reroute_back])
    cycle_ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=cycle_client,
        swe_agent=LeadGameplaySWEAgent(client=cycle_client),
        code_verify=lambda _p, _t: None,
        filmstrip_fn=lambda *_a, **_k: hero_png,
    )
    mech_item = WorkItem(
        deliberation="Cycle guard check.",
        item_id="mech_cycle",
        kind="mechanic_code",
        title="Cycle Check",
        spec="Verify cycle guard.",
        proof_spec="Proof.",
        target_modules=["main.gd"],
    )
    decision_qa = TaskLeadDecision(
        deliberation="Start with QA.",
        current_strategy="Delegate initial proof check to qa_test_engineer.",
        action="delegate",
        specialist_agent="qa_test_engineer",
        brief="Capture proof.",
    )
    cycle_report = _dispatch_to_specialist(cycle_ctx, mech_item, decision_qa, turn_index=0)
    assert cycle_report.specialist_agent == "swe_engineer"
    assert cycle_report.rerouted_from == ["qa_test_engineer"]
    assert len(cycle_client.calls) == 2, "Cycle back to visited `qa_test_engineer` must terminate immediately!"


def test_council_steering_routing_peer_handoffs_and_diagnostic_probe_reroute(tmp_path: Path):
    """Verifies that:
    1. `_execute_steering_pivot_turns` uses `CouncilSteeringRoutingPlan` + multimodal asset contact sheet to pick
       `VisualArtDirector` first (instead of hardcoded `GameplaySystemsEngineer`), and when `VisualArtDirector` sets
       `route_to_council_role='GameplaySystemsEngineer: update sprite_hframes'`, dynamically inserts
       `GameplaySystemsEngineer` next in `pivot_queue` and raises the board task.
    2. `_run_council_diagnostic_probe` follows a read-only specialist reroute (`swe_engineer` -> `visual_artist`).
    3. Main `CreativeWritersRoom` discussion loop honors `route_to_council_role` directly on the next turn without
       polling `_select_next_speaker`.
    """
    _seed_minimal_cartridge(tmp_path)
    _make_rgba_png(tmp_path / "assets" / "sprites" / "z_active_hero.png", size=(128, 64))

    board = ProductionBoard(tmp_path / "backlog.json")
    bible = GameBibleSpec(
        deliberation="Synthetic bible.",
        cartridge_id=tmp_path.name,
        target_episode=0,
        status="in_production",
        game_title="Synthetic Studio Game",
        core_genre="action",
        slice_roadmap=[
            GameSliceSpec(
                deliberation="Slice 0.",
                slice_id="slice_0",
                slice_index=0,
                title="Slice 0",
                slice_objective="Core loop.",
                player_acclimation_and_mechanics_focus="Controls.",
                status="active",
            )
        ],
        active_slice_index=0,
    )

    # 1. Read-only diagnostic probe reroute: swe_engineer (inspect_only) -> visual_artist (_read_only_probe=True)
    swe_probe_reroute = TrackCodeSynthesisPayload(
        deliberation="main.gd loads z_active_hero.png; visual_artist should inspect pixel dimensions at 1:1.",
        implementation_summary="Found load('res://assets/sprites/z_active_hero.png') in main.gd.",
        inspection_report="main.gd loads z_active_hero.png with sprite_hframes = 1.",
        reroute_to_specialist="visual_artist",
        reroute_brief="Render 1:1 diagnostic contact sheet of assets/sprites/z_active_hero.png.",
        reroute_target_paths=["assets/sprites/z_active_hero.png"],
        operation_mode="reroute",
    ).model_dump_json()
    vis_probe_sheet = VisualArtistPlan(
        deliberation="Rendering 1:1 plate for Council diagnostic probe.",
        target_asset_paths=["assets/sprites/z_active_hero.png"],
        grid_px=64,
        operation="render_contact_sheet",
    ).model_dump_json()

    probe_events: List[tuple[str, Dict[str, Any]]] = []
    probe_client = _QueuedFakeClient([swe_probe_reroute, vis_probe_sheet])
    wr_probe = CreativeWritersRoom(
        client=probe_client,
        board=board,
        event_callback=lambda et, pl: probe_events.append((et, pl)),
    )
    probe_txt, _ = wr_probe._run_council_diagnostic_probe(
        candidate_dir=tmp_path,
        diag_probe=CouncilDiagnosticProbe(
            deliberation="Check hero sprite wiring.",
            specialist_agent="swe_engineer",
            diagnostic_brief="Inspect hero sprite usage in main.gd.",
            target_paths=["main.gd"],
            needs_runtime_debug_scene=True,
        ),
        title_hint="Hero sprite framing",
    )
    assert "Probe Leg #1 (`swe_engineer`)" in probe_txt
    assert "Probe Leg #2 (`visual_artist`)" in probe_txt
    assert any(et == "council_diagnostic_probe_rerouted" for et, _ in probe_events)

    # 2. Steering routing (`CouncilSteeringRoutingPlan`) -> `VisualArtDirector` -> handoff to `GameplaySystemsEngineer`
    steering_plan = CouncilSteeringRoutingPlan(
        deliberation="Visual framing report touches sprite layout; VisualArtDirector should triage first.",
        lead_council_role="VisualArtDirector",
        supporting_council_roles=[],
        needs_specialist_probe=False,
    ).model_dump_json()
    vad_turn = WritersRoomTurnProposal(
        deliberation="The contact sheet shows z_active_hero.png is intact; GDScript hframes needs GameplaySystemsEngineer.",
        spoken_dialogue="The hero sheet pixels on the contact sheet are clean; routing to GameplaySystemsEngineer to wire hframes.",
        visual_style_guide_update="Crisp 2-frame character strips.",
        route_to_council_role="GameplaySystemsEngineer: raise mechanic_code task to wire sprite_hframes in main.gd",
    ).model_dump_json()
    gse_turn = WritersRoomTurnProposal(
        deliberation="Raising focused mechanic_code task on main.gd.",
        spoken_dialogue="Raising `wire_hero_hframes` on `main.gd` to match the 2-frame strip.",
        backlog_additions=[
            WorkItemProposal(
                deliberation="Align sprite_hframes with 2-frame strip.",
                item_id="wire_hero_hframes",
                kind="mechanic_code",
                title="Wire Hero Sprite HFrames",
                spec="Set sprite_hframes = 2 in main.gd.",
                proof_spec="Verify sprite renders single frame cleanly.",
                target_modules=["main.gd"],
                slice_id="slice_0",
            )
        ],
    ).model_dump_json()

    steer_events: List[tuple[str, Dict[str, Any]]] = []
    steer_client = _QueuedFakeClient([steering_plan, vad_turn, gse_turn], steering_routing=True)
    wr_steer = CreativeWritersRoom(
        client=steer_client,
        board=board,
        event_callback=lambda et, pl: steer_events.append((et, pl)),
    )
    wr_steer._execute_steering_pivot_turns(
        game_bible=bible,
        dna_summary="Synthetic DNA",
        candidate_dir=tmp_path,
        steered_notes=["The character sprite shows two frames side-by-side."],
    )
    speakers = [u.speaker_role for u in bible.writers_room_transcript]
    assert speakers == ["VisualArtDirector", "GameplaySystemsEngineer"]
    assert any(it.item_id == "wire_hero_hframes" for it in board.items())
    assert any(et == "council_steering_routed" and pl["lead_council_role"] == "VisualArtDirector" for et, pl in steer_events)
    assert any(
        et == "council_role_rerouted"
        and pl["from_role"] == "VisualArtDirector"
        and pl["to_role"] == "GameplaySystemsEngineer"
        for et, pl in steer_events
    )

    # 3. Main discussion loop honors `route_to_council_role` without calling `_select_next_speaker` on the routed turn
    gd_turn = WritersRoomTurnProposal(
        deliberation="Core loop defined; routing directly to AudioAtmosphereDesigner for reactive acoustic cues.",
        spoken_dialogue="Core loop is set; handing directly to AudioAtmosphereDesigner for sound cues.",
        updated_first_60_seconds_hook="Immediate tactile launch.",
        updated_diegetic_ui_philosophy="In-world gauges.",
        added_or_updated_storyboard_beats=[
            ExperienceStoryboardBeat(
                deliberation="Beat 0.",
                beat_index=0,
                phase_title="Opening",
                player_emotional_state="Curiosity.",
                sensory_and_camera_staging="Wide 2D world view.",
                diegetic_teaching_moment="In-world prompt.",
                narrative_integration="Environmental stake.",
            )
        ],
        route_to_council_role="AudioAtmosphereDesigner",
        routing_handoff_reason="Define acoustic feedback for the opening loop.",
    ).model_dump_json()
    audio_turn = WritersRoomTurnProposal(
        deliberation="Adding acoustic direction following GameplayDesigner handoff.",
        spoken_dialogue="Layering crisp mechanical transients on player actions.",
        audio_direction_update="Crisp tactile foley and reactive ambient bed.",
    ).model_dump_json()
    conv_done = DirectorConvergenceCheck(
        deliberation="Ready.",
        experience_cohesion_score=0.95,
        is_ready_to_lock_bible=True,
    ).model_dump_json()

    loop_client = _QueuedFakeClient([gd_turn, audio_turn, conv_done])
    wr_loop = CreativeWritersRoom(client=loop_client, max_turns=2, min_turns_before_convergence=2)
    select_calls: List[str] = []

    def _fake_select_next_speaker(**kw):
        select_calls.append(kw["last_speaker_role"])
        from engine.studio.writers_room import SPECIALIST_ROSTER

        gd_spec = next(s for s in SPECIALIST_ROSTER if s["role"] == "GameplayDesigner")
        return gd_spec, WritersRoomListenCheck(
            deliberation="GameplayDesigner opens.",
            should_interject=True,
            urgency_score=0.9,
            intended_contribution_summary="Define opening loop.",
        )

    wr_loop._select_next_speaker = _fake_select_next_speaker  # type: ignore[method-assign]
    wr_loop._complete_missing_design_sections = lambda *a, **kw: None  # type: ignore[method-assign]
    wr_loop._synthesize_final_world_spec = lambda *a, **kw: None  # type: ignore[method-assign]
    wr_loop._missing_required_board_specialists = lambda *_a, **_kw: []  # type: ignore[method-assign]
    wr_loop.run_slice_expansion = lambda *a, **kw: setattr(seed_bible.slice_roadmap[0], "status", "completed")  # type: ignore[method-assign]

    from engine.studio.models import WritersRoomUtterance

    seed_bible = GameBibleSpec(
        deliberation="Seed.",
        cartridge_id="seed_01",
        target_episode=0,
        status="brainstorming",
        game_title="Acoustic Loop",
        core_genre="puzzle",
        writers_room_transcript=[
            WritersRoomUtterance(
                deliberation="Opening seed.",
                turn_index=0,
                speaker_role="CreativeDirector",
                speaker_display_name="Creative Director",
                spoken_dialogue="Let us shape the opening loop.",
            )
        ],
    )
    wr_loop.run_writers_room(dna=SimpleNamespace(model_dump_json=lambda **_k: "{}"), existing_bible=seed_bible)
    loop_speakers = [u.speaker_role for u in seed_bible.writers_room_transcript if u.speaker_role != "CreativeDirector"]
    assert loop_speakers == ["GameplayDesigner", "AudioAtmosphereDesigner"]
    assert len(select_calls) == 1, "Turn 2 must route directly to AudioAtmosphereDesigner without polling _select_next_speaker!"


def test_active_asset_visibility_and_sandboxed_orphan_cleanup(tmp_path: Path):
    """Verifies that:
    1. `_collect_observational_asset_metadata` and `_build_asset_integrity_dossier_and_sheet` surface 100% of
       actively referenced in-game PNGs (`z_active_*.png`) even when 35 alphabetically earlier unreferenced
       orphan PNGs (`a_orphan_00.png` .. `a_orphan_34.png`) sit in `assets/sprites/` and `assets/asset_library.json`.
    2. `apply_sandboxed_cartridge_deletions` removes obsolete files under `assets/` (plus `.import` sidecars and
       `assets/asset_library.json` entries) while strictly blocking path traversal (`../`) and protected files
       (`project.godot`, `main.gd`, `game_bible.json`, `assets/asset_library.json`).
    """
    _seed_minimal_cartridge(tmp_path)
    (tmp_path / "scenes.gd").write_text(
        "extends Node2D\n"
        "const PROP_TEX := preload('res://assets/sprites/z_active_prop.png')\n",
        encoding="utf-8",
    )

    entries: Dict[str, Any] = {}
    for i in range(35):
        rel_orph = f"assets/sprites/a_orphan_{i:02d}.png"
        _make_rgba_png(tmp_path / rel_orph, size=(32, 32))
        entries[f"a_orphan_{i:02d}"] = {
            "asset_id": f"a_orphan_{i:02d}",
            "file_path": rel_orph,
            "hframes": 1,
            "vframes": 1,
        }
        # Create a Godot `.import` sidecar for orphan 00
        if i == 0:
            (tmp_path / f"{rel_orph}.import").write_text("[remap]\nimporter=\"texture\"\n", encoding="utf-8")

    for active_id in ("z_active_hero", "z_active_prop"):
        rel_act = f"assets/sprites/{active_id}.png"
        _make_rgba_png(tmp_path / rel_act, size=(64, 64))
        entries[active_id] = {
            "asset_id": active_id,
            "file_path": rel_act,
            "hframes": 2 if active_id == "z_active_hero" else 1,
            "vframes": 1,
        }

    lib_path = tmp_path / "assets" / "asset_library.json"
    lib_path.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")

    visual_files = sorted((tmp_path / "assets").rglob("*.png"))
    meta = _collect_observational_asset_metadata(tmp_path, visual_files)
    assert meta["assets/sprites/z_active_hero.png"]["referenced"] is True
    assert meta["assets/sprites/z_active_prop.png"]["referenced"] is True
    assert meta["assets/sprites/a_orphan_00.png"]["referenced"] is False

    wr = CreativeWritersRoom(client=None)
    dossier, sheet_path, _ = wr._build_asset_integrity_dossier_and_sheet(tmp_path, "visibility_check")
    assert sheet_path is not None and sheet_path.is_file()
    assert "active_in_game=2, unreferenced_on_disk=35" in dossier
    # Both z_active_* assets appear in the ACTIVE section despite 35 alphabetically earlier a_orphan_* files!
    active_section = dossier.split("Unreferenced / Orphaned PNG Files on Disk")[0]
    assert "`assets/sprites/z_active_hero.png` [ACTIVE]" in active_section
    assert "`assets/sprites/z_active_prop.png` [ACTIVE]" in active_section

    # Verify sandboxed deletion deletes obsolete orphan PNG + .import sidecar and prunes asset_library.json
    deleted = apply_sandboxed_cartridge_deletions(
        tmp_path,
        ["assets/sprites/a_orphan_00.png", "res://assets/sprites/a_orphan_01.png"],
        caller_role="visual_artist",
    )
    assert deleted == ["assets/sprites/a_orphan_00.png", "assets/sprites/a_orphan_01.png"]
    assert not (tmp_path / "assets" / "sprites" / "a_orphan_00.png").exists()
    assert not (tmp_path / "assets" / "sprites" / "a_orphan_00.png.import").exists()
    pruned_lib = json.loads(lib_path.read_text(encoding="utf-8"))
    assert "a_orphan_00" not in pruned_lib["entries"]
    assert "z_active_hero" in pruned_lib["entries"]

    # Verify sandbox strictly blocks path traversal and protected cartridge root/control files
    for forbidden in ("../outside.png", "project.godot", "main.gd", "assets/asset_library.json", "game_bible.json"):
        with pytest.raises(StudioAgentError):
            apply_sandboxed_cartridge_deletions(tmp_path, [forbidden], caller_role="visual_artist")
