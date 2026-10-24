"""Council <-> Production Board integration: backlog raising, concurrent building, proof review, revision cap, draining."""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
from PIL import Image

from engine.studio.models import (
    CapDecision,
    ReviewBatch,
    ReviewDecision,
    StudioAgentError,
    WorkItemProposal,
    WritersRoomTurnProposal,
)
from engine.studio.production_board import BoardScheduler, ProductionBoard, WorkerResult
from engine.studio.visual import VisualArtisanAgent
from engine.studio.writers_room import CreativeWritersRoom

import tests.test_writers_room_and_game_bible as base


def P(item_id, deps=(), kind="visual_asset", modules=(), max_rev=None):
    return WorkItemProposal(
        deliberation="needed for the design", item_id=item_id, kind=kind, title=item_id, spec=f"build {item_id}",
        proof_spec="show a clear render", depends_on=list(deps), target_modules=list(modules), max_revisions=max_rev,
    )


class CouncilFakeModels(base._SchemaDispatchGeminiModels):
    """Scripted council: raises backlog items on its first turns and reviews proofs via a pluggable policy."""

    def __init__(self, additions_by_turn, review_policy):
        super().__init__()
        self.additions_by_turn = additions_by_turn
        self.review_policy = review_policy
        self.review_contents: List[Any] = []
        self.turn_prompts: List[str] = []
        self.review_calls = 0

    def generate_content(self, model, contents, config=None):
        schema = getattr(config, "response_schema", None) if config else None
        name = getattr(schema, "__name__", "")
        if name == "ReviewBatch":
            self.review_calls += 1
            self.review_contents.append(contents)
            text = "\n".join(c for c in (contents if isinstance(contents, list) else [contents]) if isinstance(c, str))
            awaiting = text.split("ITEMS AWAITING YOUR REVIEW")[1].split("ITEMS THAT HIT THEIR REVISION CAP")[0]
            deciding = text.split("ITEMS THAT HIT THEIR REVISION CAP")[1]
            decisions = [self.review_policy(i, awaiting, text) for i in re.findall(r"ITEM \[([^\]]+)\]", awaiting)]
            caps = [CapDecision(deliberation="read the whole history", item_id=i, action="accept_as_is")
                    for i in re.findall(r"ITEM \[([^\]]+)\]", deciding)]
            return base._FakeGeminiResponse(ReviewBatch(deliberation="assessed the proofs", review_decisions=decisions, cap_decisions=caps))
        resp = super().generate_content(model, contents, config)
        if name == "WritersRoomTurnProposal":
            self.turn_prompts.append(contents if isinstance(contents, str) else str(contents))
            adds = self.additions_by_turn.get(self.turn_call_count) or self.additions_by_turn.get("all")
            if adds:
                proposal = WritersRoomTurnProposal.model_validate_json(resp.text)
                proposal.backlog_additions = adds
                return base._FakeGeminiResponse(proposal)
        return resp


def build(tmp_path, additions_by_turn, review_policy, handlers=None, max_workers=3, **board_kw):
    client = base._FakeGeminiClient()
    client.models = CouncilFakeModels(additions_by_turn, review_policy)
    candidate = tmp_path / "cartridges" / "candidate_new"
    candidate.mkdir(parents=True)
    board = ProductionBoard(candidate / "backlog.json", **board_kw)
    order: List[str] = []
    lock = threading.Lock()

    def make_handler(kind):
        def handler(item):
            with lock:
                order.append(item.item_id)
            png = candidate / f"{item.item_id}_rev{len(item.revisions)}.png"
            Image.new("RGB", (8, 8), (len(item.revisions) * 40 % 255, 20, 90)).save(png)
            time.sleep(0.02)
            return WorkerResult([str(png)], f"built {item.item_id} (rev {len(item.revisions)})")
        return handler

    handlers = handlers or {k: make_handler(k) for k in ("visual_asset", "level_map", "mechanic_code", "integration")}
    scheduler = BoardScheduler(board, handlers, max_workers=max_workers, poll_interval=0.01)
    room = CreativeWritersRoom(
        client=client, visual_agent=VisualArtisanAgent(client=client), max_turns=3, min_turns_before_convergence=2,
        board=board, scheduler=scheduler, drain_timeout_s=20, drain_poll_s=0.01,
    )
    return room, client, board, scheduler, candidate, order


def approve_all(item_id, awaiting, text):
    return ReviewDecision(deliberation="matches the design", item_id=item_id, verdict="approve")


def test_council_raises_items_workers_build_in_dependency_order_and_room_locks_after_drain(tmp_path):
    adds = {2: [P("sky"), P("terrain", ["sky"]), P("hero"), P("level1", ["terrain", "hero"], kind="level_map")]}
    room, client, board, sched, candidate, order = build(tmp_path, adds, approve_all)
    try:
        bible = room.run_writers_room(dna=base._sample_dna(), candidate_dir=candidate)
    finally:
        sched.stop()
    assert bible.status == "locked"
    assert board.is_drained() and all(i.state == "approved" for i in board.items())
    assert order.index("sky") < order.index("terrain") < order.index("level1")
    assert order.index("hero") < order.index("level1")
    assert any("Raised 4 backlog item" in u for t in bible.writers_room_transcript for u in t.bible_updates_summary)
    persisted = json.loads((candidate / "backlog.json").read_text())
    assert {i["item_id"] for i in persisted["items"]} == {"sky", "terrain", "hero", "level1"}


def test_reviewer_receives_proof_images_and_level_items_go_to_the_director(tmp_path):
    adds = {2: [P("lvl", kind="level_map"), P("art")]}
    room, client, board, sched, candidate, _ = build(tmp_path, adds, approve_all)
    try:
        room.run_writers_room(dna=base._sample_dna(), candidate_dir=candidate)
    finally:
        sched.stop()
    flat = [c for contents in client.models.review_contents for c in (contents if isinstance(contents, list) else [])]
    image_parts = [c for c in flat if not isinstance(c, str)]
    assert len(image_parts) >= 2                                   # actual proof images were attached as parts
    texts = [c for c in flat if isinstance(c, str) and "PROOF IMAGE for item" in c]
    assert any("'lvl'" in t for t in texts) and any("'art'" in t for t in texts)
    prompts = [c for c in flat if isinstance(c, str) and c.startswith("You are")]
    # the level map was reviewed by the Director, the asset by the specialist who raised it
    assert any("Creative Director" in p and "ITEM [lvl]" in p for p in prompts)
    assert not any("ITEM [art]" in p and "Creative Director" in p.split("\n")[0] for p in prompts)


def test_request_changes_roundtrip_gives_worker_the_feedback_history(tmp_path):
    seen_revisions = {}

    def handler(item):
        seen_revisions.setdefault(item.item_id, []).append([(r.feedback, r.verdict) for r in item.revisions])
        png = candidate / f"{item.item_id}_{len(item.revisions)}.png"
        Image.new("RGB", (8, 8), (len(item.revisions) * 60, 10, 10)).save(png)
        return WorkerResult([str(png)], "ok")

    def policy(item_id, awaiting, text):
        first = "rev 1:" not in awaiting.split(f"ITEM [{item_id}]")[1].split("ITEM [")[0] or "verdict=(pending)" in awaiting
        has_history = "feedback='make the silhouette bolder'" in text
        if not has_history:
            return ReviewDecision(deliberation="too timid", item_id=item_id, feedback="make the silhouette bolder", verdict="request_changes")
        return ReviewDecision(deliberation="now it reads", item_id=item_id, verdict="approve")

    adds = {2: [P("hero")]}
    room, client, board, sched, candidate, _ = build(tmp_path, adds, policy, handlers={"visual_asset": handler})
    try:
        room.run_writers_room(dna=base._sample_dna(), candidate_dir=candidate)
    finally:
        sched.stop()
    hero = board.get("hero")
    assert hero.state == "approved" and len(hero.revisions) == 2
    assert hero.revisions[0].feedback == "make the silhouette bolder" and hero.revisions[0].verdict == "request_changes"
    assert seen_revisions["hero"][0] == [] and seen_revisions["hero"][1][0][0] == "make the silhouette bolder"


def test_revision_cap_forces_explicit_council_decision_recorded_as_compromise(tmp_path):
    def always_change(item_id, awaiting, text):
        n = text.count(f"ITEM [{item_id}]")
        return ReviewDecision(deliberation="still not right", item_id=item_id, feedback=f"attempt feedback {time.time_ns()}",
                              verdict="request_changes")

    adds = {2: [P("stubborn")]}
    room, client, board, sched, candidate, _ = build(tmp_path, adds, always_change, default_max_revisions=2)
    try:
        bible = room.run_writers_room(dna=base._sample_dna(), candidate_dir=candidate)
    finally:
        sched.stop()
    item = board.get("stubborn")
    assert item.state == "approved" and "Accepted as-is" in item.accepted_compromise
    assert len(item.revisions) == 2 and "cap reached" in item.accepted_compromise
    assert bible.status == "locked"


def test_invalid_backlog_additions_are_fed_back_and_corrected_on_retry(tmp_path):
    adds = {2: [P("bad", ["does_not_exist"])], 3: [P("good")]}      # 2nd attempt of the same turn fixes the batch
    room, client, board, sched, candidate, _ = build(tmp_path, adds, approve_all)
    try:
        room.run_writers_room(dna=base._sample_dna(), candidate_dir=candidate)
    finally:
        sched.stop()
    assert [i.item_id for i in board.items()] == ["good"]
    retry_prompts = [p for p in client.models.turn_prompts if "FAILED VALIDATION" in p]
    assert retry_prompts and "unknown item 'does_not_exist'" in retry_prompts[0]


def test_persistently_invalid_backlog_additions_fail_loudly_and_add_nothing(tmp_path):
    room, client, board, sched, candidate, _ = build(tmp_path, {"all": [P("bad", ["does_not_exist"])]}, approve_all)
    try:
        with pytest.raises(StudioAgentError) as ei:
            room.run_writers_room(dna=base._sample_dna(), candidate_dir=candidate)
    finally:
        sched.stop()
    assert "unknown item 'does_not_exist'" in str(ei.value)
    assert board.items() == []


def test_worker_failures_are_requeued_once_then_surface_the_real_error(tmp_path):
    calls = {"n": 0}

    def flaky(item):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient provider error")
        png = candidate / "ok.png"
        Image.new("RGB", (8, 8)).save(png)
        return WorkerResult([str(png)])

    room, client, board, sched, candidate, _ = build(tmp_path, {2: [P("flaky")]}, approve_all, handlers={"visual_asset": flaky})
    try:
        room.run_writers_room(dna=base._sample_dna(), candidate_dir=candidate)
    finally:
        sched.stop()
    assert board.get("flaky").state == "approved" and board.get("flaky").failure_count == 1

    def broken(item):
        raise RuntimeError("permanent defect XYZ")

    room, client, board, sched, candidate, _ = build(tmp_path / "second", {2: [P("broken")]}, approve_all, handlers={"visual_asset": broken})
    try:
        with pytest.raises(StudioAgentError) as ei:
            room.run_writers_room(dna=base._sample_dna(), candidate_dir=candidate)
    finally:
        sched.stop()
    assert "permanent defect XYZ" in str(ei.value) and "failed 2 times" in str(ei.value)


def test_room_without_board_is_unchanged(tmp_path):
    room, client = base._make_room(tmp_path)
    room.max_turns, room.min_turns_before_convergence = 3, 2
    bible = room.run_writers_room(dna=base._sample_dna(), candidate_dir=tmp_path / "cartridges" / "candidate_new")
    assert bible.status == "locked"
    assert not any("BUILD BACKLOG" in str(c) for c in [])  # no board -> no backlog instructions or review calls
    assert all((x["schema"].__name__ if x["schema"] else "") != "ReviewBatch" for x in client.models.calls)


def test_multi_specialist_panel_vetoes_when_ux_or_art_rejects_and_attaches_concept_mockup(tmp_path):
    vetoed_once = {"ux": False, "art": False}

    def panel_policy(item_id, awaiting, text):
        if item_id == "hud_filter":
            if "(UXExperienceDesigner) reviewing work" in text and not vetoed_once["ux"]:
                vetoed_once["ux"] = True
                return ReviewDecision(
                    deliberation="giant debug panel",
                    item_id=item_id,
                    feedback="Remove HEADLESS MECHANICS debug panel",
                    verdict="request_changes",
                )
            if "(VisualArtDirector) reviewing work" in text and not vetoed_once["art"]:
                vetoed_once["art"] = True
                return ReviewDecision(
                    deliberation="programmer art boxes",
                    item_id=item_id,
                    feedback="Bind kalon_stances.png instead of draw_rect",
                    verdict="request_changes",
                )
        return ReviewDecision(deliberation="looks good", item_id=item_id, verdict="approve")

    adds = {2: [P("hud_filter", kind="mechanic_code", modules=("ui_hud.gd",))]}
    room, client, board, sched, candidate, _ = build(tmp_path, adds, panel_policy)
    mockup_dir = candidate / "assets" / "mockups"
    mockup_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 36), (40, 80, 120)).save(mockup_dir / "mockup_1_mockup_test.png")
    try:
        bible = room.run_writers_room(dna=base._sample_dna(), candidate_dir=candidate)
    finally:
        sched.stop()
    assert bible.status == "locked"
    hud_item = board.get("hud_filter")
    assert hud_item.state == "approved" and len(hud_item.revisions) == 2
    rev0_fb = hud_item.revisions[0].feedback
    assert "[UXExperienceDesigner]: Remove HEADLESS MECHANICS debug panel" in rev0_fb
    assert "[VisualArtDirector]: Bind kalon_stances.png instead of draw_rect" in rev0_fb
    flat = [c for contents in client.models.review_contents for c in (contents if isinstance(contents, list) else [])]
    assert any(isinstance(c, str) and "REFERENCE 16:9 CONCEPT ART MOCKUP" in c for c in flat)

