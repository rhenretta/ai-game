"""Production Board + scheduler: dependency tree, concurrency, review loop, revision cap, stalls, persistence."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from engine.studio.models import CapDecision, ReviewDecision, StudioAgentError, WorkItemProposal
from engine.studio.production_board import (
    BoardScheduler,
    BoardValidationError,
    ProductionBoard,
    WorkerResult,
    WorkerYielded,
    is_in_board_scheduler,
)


def P(item_id, deps=(), kind="visual_asset", modules=(), max_rev=None, slice_id=""):
    return WorkItemProposal(
        deliberation="d", item_id=item_id, kind=kind, title=item_id, spec=f"build {item_id}",
        proof_spec="show it", depends_on=list(deps), target_modules=list(modules),
        slice_id=slice_id, max_revisions=max_rev,
    )


def review(item_id, verdict, feedback=""):
    return ReviewDecision(deliberation="looked closely", item_id=item_id, feedback=feedback, verdict=verdict)


def proof(tmp_path: Path, name: str, content: bytes = b"x") -> str:
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


# ----------------------------------------------------------------------------- structure
def test_models_are_rationale_first():
    from engine.studio.models import RevisionRecord, WorkItem, WorkItem
    for cls in (WorkItemProposal, ReviewDecision, CapDecision, WorkItem, RevisionRecord):
        assert list(cls.model_fields)[0] == "deliberation", cls.__name__
    assert list(ReviewDecision.model_fields)[-1] == "verdict"
    assert list(CapDecision.model_fields)[-1] == "action"


def test_dependency_states_and_ordering():
    b = ProductionBoard()
    b.add_items([P("a"), P("b", ["a"]), P("c", ["a", "b"])], "Designer")
    assert [b.get(i).state for i in "abc"] == ["ready", "blocked", "blocked"]
    claimed = b.claim_ready(limit=5)
    assert [i.item_id for i in claimed] == ["a"]            # b and c are not claimable yet
    b.submit_result("a", WorkerResult())
    assert b.get("b").state == "blocked"                    # approval, not completion, unblocks
    b.apply_review(review("a", "approve"))
    assert b.get("b").state == "ready" and b.get("c").state == "blocked"
    b.claim_ready(); b.submit_result("b", WorkerResult()); b.apply_review(review("b", "approve"))
    assert b.get("c").state == "ready"


def test_rejects_unknown_deps_duplicates_self_and_cycles_atomically():
    b = ProductionBoard()
    b.add_items([P("a")], "d")
    for bad, needle in [
        ([P("x", ["nope"])], "unknown item"),
        ([P("a")], "already exists"),
        ([P("y"), P("y")], "Duplicate"),
        ([P("z", ["z"])], "itself"),
        ([P("m", ["n"]), P("n", ["m"])], "cycle"),
    ]:
        with pytest.raises(BoardValidationError) as ei:
            b.add_items(bad, "d")
        assert needle in str(ei.value)
    assert [i.item_id for i in b.items()] == ["a"]           # nothing from rejected batches was added


def test_module_lock_serialises_code_items_but_not_assets():
    b = ProductionBoard()
    b.add_items([P("c1", kind="mechanic_code", modules=["main.gd"]), P("c2", kind="mechanic_code", modules=["main.gd"]),
                 P("c3", kind="mechanic_code", modules=["scenes.gd"]), P("a1"), P("a2")], "d")
    first = {i.item_id for i in b.claim_ready(limit=10)}
    assert first == {"c1", "c3", "a1", "a2"}                 # c2 waits for main.gd
    b.submit_result("c1", WorkerResult()); b.apply_review(review("c1", "approve"))
    assert [i.item_id for i in b.claim_ready(limit=10)] == ["c2"]


# ----------------------------------------------------------------------------- review loop
def test_review_roundtrip_changes_then_approve(tmp_path):
    b = ProductionBoard()
    b.add_items([P("a")], "Visual")
    b.claim_ready(); b.submit_result("a", WorkerResult([proof(tmp_path, "p1.png", b"1")], "v1"))
    assert [i.item_id for i in b.awaiting_review("Visual")] == ["a"]
    assert b.awaiting_review("Other") == []
    b.apply_review(review("a", "request_changes", "make the sails larger"))
    assert b.get("a").state == "changes_requested"
    assert b.get("a").revisions[0].feedback == "make the sails larger"
    claimed = b.claim_ready()
    assert [i.item_id for i in claimed] == ["a"] and len(claimed[0].revisions) == 1   # worker sees history
    b.submit_result("a", WorkerResult([proof(tmp_path, "p2.png", b"2")], "v2"))
    b.apply_review(review("a", "approve"))
    assert b.get("a").state == "approved" and b.is_drained()


def test_request_changes_requires_feedback_and_state_unchanged_on_error():
    b = ProductionBoard()
    b.add_items([P("a")], "d")
    b.claim_ready(); b.submit_result("a", WorkerResult())
    with pytest.raises(BoardValidationError):
        b.apply_review(review("a", "request_changes", "  "))
    assert b.get("a").state == "awaiting_review" and b.get("a").revisions[0].verdict == ""


def test_revision_cap_escalates_to_needs_decision(tmp_path):
    b = ProductionBoard(default_max_revisions=3)
    b.add_items([P("a")], "d")
    for n in range(3):
        b.claim_ready()
        b.submit_result("a", WorkerResult([proof(tmp_path, f"p{n}.png", bytes([n]))]))
        b.apply_review(review("a", "request_changes", f"different feedback {n}"))
    assert b.get("a").state == "needs_decision"
    assert "cap reached" in b.get("a").decision_reason
    assert not b.is_drained() and [i.item_id for i in b.needing_decision()] == ["a"]
    assert b.claim_ready() == []


def test_per_item_cap_override():
    b = ProductionBoard(default_max_revisions=5)
    b.add_items([P("a", max_rev=1)], "d")
    b.claim_ready(); b.submit_result("a", WorkerResult())
    b.apply_review(review("a", "request_changes", "x"))
    assert b.get("a").state == "needs_decision"


def test_stall_on_repeated_feedback_and_on_identical_proof(tmp_path):
    b = ProductionBoard()
    b.add_items([P("a"), P("b")], "d")
    b.claim_ready(limit=2)
    for item, fb2 in (("a", "Make it   BIGGER"), ("b", "other words")):
        b.submit_result(item, WorkerResult([proof(tmp_path, f"{item}.png", b"same")]))
        b.apply_review(review(item, "request_changes", "make it bigger" if item == "a" else "first words"))
        b.claim_ready()
        b.submit_result(item, WorkerResult([proof(tmp_path, f"{item}.png", b"same")]))
        b.apply_review(review(item, "request_changes", fb2))
    assert "identical feedback" in b.get("a").decision_reason       # normalised-equal feedback
    assert "proof is unchanged" in b.get("b").decision_reason       # same proof hash, different feedback


@pytest.mark.parametrize("action", ["accept_as_is", "rewrite_spec", "split", "cancel"])
def test_cap_decisions(action, tmp_path):
    b = ProductionBoard(default_max_revisions=1)
    b.add_items([P("a"), P("child", ["a"])], "d")
    b.claim_ready(); b.submit_result("a", WorkerResult([proof(tmp_path, "p.png")]))
    b.apply_review(review("a", "request_changes", "no"))
    assert b.get("a").state == "needs_decision"
    kwargs = {}
    if action == "rewrite_spec":
        kwargs["replacement_spec"] = "simpler spec"
    if action == "split":
        kwargs["split_items"] = [P("a1"), P("a2")]
    b.apply_cap_decision(CapDecision(deliberation="after reading history", item_id="a", action=action, **kwargs))
    a = b.get("a")
    if action == "accept_as_is":
        assert a.state == "approved" and "Accepted as-is" in a.accepted_compromise
        assert b.get("child").state == "ready"                      # compromise still unblocks dependents
    elif action == "rewrite_spec":
        assert a.state == "ready" and a.spec == "simpler spec" and a.spec_resets == 1 and a.cap_base_index == 1
    elif action == "split":
        assert a.state == "cancelled"
        assert set(b.get("child").depends_on) == {"a1", "a2"}       # dependents wait on every piece
        assert b.get("a1").state == "ready"
    else:
        assert a.state == "cancelled"
        assert b.get("child").state == "blocked" and [i.item_id for i in b.blocked_by_cancelled()] == ["child"]
        assert not b.is_drained()                                   # a stranded dependent keeps the backlog open


def test_rewrite_resets_counter_but_max_two_resets():
    b = ProductionBoard(default_max_revisions=1)
    b.add_items([P("a")], "d")
    for n in range(3):
        b.claim_ready(); b.submit_result("a", WorkerResult())
        b.apply_review(review("a", "request_changes", f"fb{n}"))
        assert b.get("a").state == "needs_decision"
        if n < 2:
            b.apply_cap_decision(CapDecision(deliberation="d", item_id="a", action="rewrite_spec", replacement_spec=f"s{n}"))
            assert b.get("a").state == "ready"
        else:
            with pytest.raises(BoardValidationError):
                b.apply_cap_decision(CapDecision(deliberation="d", item_id="a", action="rewrite_spec", replacement_spec="s"))
    assert b.get("a").state == "needs_decision"                     # unchanged; council must pick another action


def test_update_item_repoints_dependencies_and_blocks_cycles():
    b = ProductionBoard()
    b.add_items([P("a"), P("b", ["a"]), P("c")], "d")
    b.cancel_item("a")
    assert b.get("b").state == "blocked"
    b.update_item("b", depends_on=["c"])
    assert b.get("b").depends_on == ["c"] and b.blocked_by_cancelled() == []
    with pytest.raises(BoardValidationError):
        b.update_item("c", depends_on=["b"])                        # b -> c -> b
    with pytest.raises(BoardValidationError):
        b.update_item("b", depends_on=["ghost"])


# ----------------------------------------------------------------------------- persistence
def test_persistence_roundtrip_and_resume_requeues_in_progress(tmp_path):
    path = tmp_path / "backlog.json"
    b = ProductionBoard(path)
    b.add_items([P("a"), P("b", ["a"])], "Visual")
    b.claim_ready()                                                  # a in_progress when "the process dies"
    b2 = ProductionBoard(path)
    assert b2.get("a").state == "ready" and b2.get("b").state == "blocked"
    b2.claim_ready(); b2.submit_result("a", WorkerResult()); b2.apply_review(review("a", "request_changes", "again"))
    b3 = ProductionBoard(path)
    assert b3.get("a").state == "changes_requested" and b3.get("a").revisions[0].feedback == "again"


def test_events_emitted_for_lifecycle():
    events = []
    b = ProductionBoard(event_callback=lambda t, p: events.append(t))
    b.add_items([P("a")], "d"); b.claim_ready(); b.submit_result("a", WorkerResult())
    b.apply_review(review("a", "approve"))
    assert events == ["backlog_item_created", "backlog_item_started", "backlog_item_proof_ready", "backlog_item_approved"]


def test_snapshot_text_lists_items_and_decisions():
    b = ProductionBoard(default_max_revisions=1)
    b.add_items([P("a")], "Visual")
    b.claim_ready(); b.submit_result("a", WorkerResult()); b.apply_review(review("a", "request_changes", "x"))
    text = b.snapshot_text()
    assert "needs_decision" in text and "[a]" in text and "REASON=" in text


# ----------------------------------------------------------------------------- scheduler
def test_scheduler_runs_dependency_tree_concurrently_with_bounded_workers(tmp_path):
    b = ProductionBoard()
    b.add_items([P("root"), P("l1", ["root"]), P("l2", ["root"]), P("l3", ["root"]), P("l4", ["root"]),
                 P("l5", ["root"]), P("join", ["l1", "l2", "l3", "l4", "l5"])], "d")
    order, lock = [], threading.Lock()

    def handler(item):
        with lock:
            order.append(("start", item.item_id))
        time.sleep(0.05)
        with lock:
            order.append(("end", item.item_id))
        return WorkerResult([proof(tmp_path, f"{item.item_id}.png", item.item_id.encode())], "ok")

    sch = BoardScheduler(b, {"visual_asset": handler}, max_workers=3)
    # council approves each proof as it arrives (simulated reviewer loop)
    for _ in range(10):
        assert sch.run_until_idle(10)
        waiting = b.awaiting_review()
        if not waiting:
            break
        for it in waiting:
            b.apply_review(review(it.item_id, "approve"))
    assert b.is_drained()
    assert 2 <= sch.peak_running <= 3                                # genuinely concurrent, never above the bound
    starts = {i: n for n, (kind, i) in enumerate(order) if kind == "start"}
    ends = {i: n for n, (kind, i) in enumerate(order) if kind == "end"}
    assert ends["root"] < min(starts[f"l{k}"] for k in range(1, 6))  # nothing starts before its dependency finished
    assert ends and max(ends[f"l{k}"] for k in range(1, 6)) < starts["join"]


def test_scheduler_surfaces_worker_exception_with_traceback_and_missing_handler():
    b = ProductionBoard()
    b.add_items([P("boom"), P("orphan", kind="narrative")], "d")

    def handler(item):
        raise RuntimeError("provider exploded")

    sch = BoardScheduler(b, {"visual_asset": handler}, max_workers=2)
    assert sch.run_until_idle(10)
    boom, orphan = b.get("boom"), b.get("orphan")
    assert boom.state == "failed" and "provider exploded" in boom.failure and "Traceback" in boom.failure
    assert orphan.state == "failed" and "No worker is registered" in orphan.failure
    assert not b.is_drained()
    b.requeue_failed("boom")
    assert b.get("boom").state == "ready"


def test_scheduler_background_mode_picks_up_items_added_later(tmp_path):
    b = ProductionBoard()
    done = threading.Event()

    def handler(item):
        done.set()
        return WorkerResult()

    sch = BoardScheduler(b, {"visual_asset": handler}, max_workers=2)
    sch.start()
    try:
        b.add_items([P("late")], "d")                                # added while the scheduler is already running
        assert done.wait(5)
        deadline = time.time() + 5
        while b.get("late").state != "awaiting_review" and time.time() < deadline:
            time.sleep(0.02)
        assert b.get("late").state == "awaiting_review"
    finally:
        sch.stop()


def test_worker_yields_on_busy_code_lock_and_runs_other_tasks_then_resumes(tmp_path):
    """When Worker 1 holds `code_lock` on a code task, Worker 2 must NOT sit blocked waiting for `code_lock`:
    it pauses/yields any code task that contends on `code_lock`, immediately executes ready parallel tasks
    (such as `visual_asset` items), and resumes the paused code task once `code_lock` is released."""
    events = []
    b = ProductionBoard(event_callback=lambda et, p: events.append((et, p["item_id"], p.get("state"))))
    b.add_items(
        [
            P("c1", kind="mechanic_code", modules=["a.gd"]),
            P("c2", kind="mechanic_code", modules=["b.gd"]),
        ],
        "GameplaySystemsDesigner",
    )

    code_lock = threading.Lock()
    c1_entered_lock = threading.Event()
    release_c1 = threading.Event()
    v_completed_while_c1_held_lock = []

    def code_handler(item):
        assert is_in_board_scheduler()
        if item.item_id == "c1":
            assert code_lock.acquire(blocking=False)
            try:
                c1_entered_lock.set()
                # Wait until both visual asset tasks finish on Worker 2 while Worker 1 holds code_lock
                assert release_c1.wait(timeout=5.0), "Worker 2 did not complete visual tasks while c1 held code_lock"
                return WorkerResult(proof_images=[proof(tmp_path, "c1.png")], notes="c1 done")
            finally:
                code_lock.release()
        else:
            if not code_lock.acquire(blocking=False):
                raise WorkerYielded("code_lock held by c1")
            try:
                return WorkerResult(proof_images=[proof(tmp_path, f"{item.item_id}.png")], notes="c2 done")
            finally:
                code_lock.release()

    def visual_handler(item):
        # Verify c1 is still holding code_lock while Worker 2 builds this visual asset
        assert code_lock.locked()
        res = WorkerResult(proof_images=[proof(tmp_path, f"{item.item_id}.png")], notes="asset done")
        v_completed_while_c1_held_lock.append(item.item_id)
        if len(v_completed_while_c1_held_lock) >= 2:
            release_c1.set()
        return res

    sch = BoardScheduler(
        b,
        {"mechanic_code": code_handler, "visual_asset": visual_handler},
        max_workers=2,
        poll_interval=0.01,
        code_lock=code_lock,
    )
    sch.start()
    try:
        assert c1_entered_lock.wait(timeout=5.0)
        # Add two visual assets after c1 & c2 were initially dispatched so c2 has to yield its slot
        b.add_items([P("v1", kind="visual_asset"), P("v2", kind="visual_asset")], "VisualArtDirectionLead")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            states = {i.item_id: i.state for i in b.items()}
            if all(st == "awaiting_review" for st in states.values()):
                break
            time.sleep(0.02)
        states = {i.item_id: i.state for i in b.items()}
        assert states == {
            "c1": "awaiting_review",
            "c2": "awaiting_review",
            "v1": "awaiting_review",
            "v2": "awaiting_review",
        }
        assert set(v_completed_while_c1_held_lock) == {"v1", "v2"}
        assert any(et == "backlog_item_yielded" and iid == "c2" for et, iid, _ in events)
    finally:
        sch.stop()


def test_slice_roadmap_and_task_lead_models_enforce_rationale_first():
    """Rule 4: Verify GameSliceSpec and TaskLeadDecision place deliberation at index 0 and status/action last."""
    from engine.studio.models import GameSliceSpec, TaskLeadDecision

    slice_fields = list(GameSliceSpec.model_fields.keys())
    assert slice_fields[0] == "deliberation"
    assert slice_fields[-1] == "status"

    lead_fields = list(TaskLeadDecision.model_fields.keys())
    assert lead_fields[0] == "deliberation"
    assert lead_fields[-1] == "action"


def test_emergent_task_spawning_tasks_lead_to_more_tasks_and_parent_resumes(tmp_path):
    """When a Task Lead executing a slice task discovers it needs specific assets/animations (e.g., player sprite,
    walk animation, jump animation), it chooses `action='spawn_tasks'` with `wait_for_spawned_tasks=True`.
    The ProductionBoard dynamically creates those child tasks, adds them to the parent's `depends_on`, blocks the parent,
    builds and Council-approves the spawned tasks, and then automatically unblocks and resumes the parent task."""
    from engine.studio.models import SpecialistReport, TaskLeadDecision
    from engine.studio.workers import WorkerContext, default_handlers

    events = []
    b = ProductionBoard(
        tmp_path / "backlog.json",
        event_callback=lambda et, p: events.append((et, p["item_id"], p.get("state"), p)),
    )
    b.add_items(
        [
            P(
                "player_acclimation_locomotion",
                kind="mechanic_code",
                modules=["entities.gd", "scenes.gd"],
                slice_id="slice_0_acclimation",
            )
        ],
        raised_by_role="GameplayDesigner",
        slice_id="slice_0_acclimation",
    )

    class _FakeLiveModels:
        def __init__(self):
            self.parent_calls = 0

        def generate_content(self, model, contents, config=None):
            prompt_text = contents[0] if isinstance(contents, list) else str(contents)
            if "TASK [player_acclimation_locomotion]" in prompt_text:
                self.parent_calls += 1
                if self.parent_calls == 1:
                    # Turn 0: Task Lead reasons out that it needs a player sprite, walk animation, and jump animation
                    decision = TaskLeadDecision(
                        deliberation=(
                            "To implement the Slice 0 player acclimation locomotion without upfront asset bloat, "
                            "we need a player character sprite, a walk cycle animation, and a jump arc animation. "
                            "Spawning these three tasks onto the ProductionBoard and waiting for Council approval."
                        ),
                        current_strategy="Spawn prerequisite player sprite, walk animation, and jump animation tasks first.",
                        spawned_tasks=[
                            P("player_base_sprite", kind="visual_asset"),
                            P("player_walk_anim", kind="animation", deps=["player_base_sprite"]),
                            P("player_jump_anim", kind="animation", deps=["player_base_sprite"]),
                        ],
                        wait_for_spawned_tasks=True,
                        action="spawn_tasks",
                    )
                elif self.parent_calls == 2:
                    # Resumed after spawned tasks are approved: delegate to swe_engineer + qa_test_engineer
                    decision = TaskLeadDecision(
                        deliberation="All three spawned player sprite/animation tasks are now approved; wiring GDScript and capturing proof.",
                        current_strategy="Wire approved player sprite, walk, and jump animations into entities.gd and capture QA filmstrip.",
                        specialist_agent="swe_engineer",
                        brief="Bind player_base_sprite, player_walk_anim, and player_jump_anim in entities.gd.",
                        target_paths=["entities.gd"],
                        action="delegate",
                    )
                else:
                    decision = TaskLeadDecision(
                        deliberation="Proof filmstrip captured and verified; submitting to Council.",
                        current_strategy="Submit completed Slice 0 locomotion proof to Council.",
                        submission_summary="Wired spawned player sprite, walk, and jump animations and verified gameplay.",
                        selected_proof_paths=[str(tmp_path / "proof_locomotion.png")],
                        action="submit_to_council",
                    )
            else:
                # Child tasks (player_base_sprite, player_walk_anim, player_jump_anim)
                if "Turn 0 — no specialists invoked yet" in prompt_text:
                    decision = TaskLeadDecision(
                        deliberation="Fabricating spawned visual/animation asset via visual_artist.",
                        current_strategy="Delegate asset fabrication to visual_artist.",
                        specialist_agent="visual_artist",
                        brief="Fabricate sprite/animation sheet and contact plate.",
                        action="delegate",
                    )
                else:
                    decision = TaskLeadDecision(
                        deliberation="Asset inspection plate ready; submitting to Council.",
                        current_strategy="Submit asset plate to Council.",
                        submission_summary="Fabricated spawned asset plate.",
                        action="submit_to_council",
                    )

            class _Resp:
                text = decision.model_dump_json()

            return _Resp()

    class _FakeLiveClient:
        __module__ = "google.genai.fake"

        def __init__(self):
            self.models = _FakeLiveModels()

    class _FakeSweAgent:
        def implement_requirement_track(self, *args, **kwargs):
            return []

    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=_FakeLiveClient(),
        visual_agent=object(),
        swe_agent=_FakeSweAgent(),
        board=b,
    )

    import engine.studio.workers as workers_mod

    orig_dispatch = workers_mod._dispatch_to_specialist

    def _fake_dispatch(w_ctx, item, decision, turn_index, _lock_already_held=False):
        if item.item_id == "player_acclimation_locomotion":
            p_file = proof(tmp_path, "proof_locomotion.png")
        else:
            p_file = proof(tmp_path, f"plate_{item.item_id}.png")
        return SpecialistReport(
            turn_index=turn_index,
            specialist_agent=decision.specialist_agent,
            brief=decision.brief,
            summary=f"Completed {decision.specialist_agent} for {item.item_id}",
            touched_files=list(decision.target_paths),
            proof_images=[p_file],
            verification_passed=True,
        )

    workers_mod._dispatch_to_specialist = _fake_dispatch
    try:
        sch = BoardScheduler(b, default_handlers(ctx), max_workers=2, poll_interval=0.01)
        # Round 1: parent runs Turn 0, spawns 3 tasks, and blocks itself; player_base_sprite immediately runs and reaches awaiting_review
        assert sch.run_until_idle(5.0)
        parent = b.get("player_acclimation_locomotion")
        assert parent.state == "blocked"
        assert set(parent.depends_on) == {"player_base_sprite", "player_walk_anim", "player_jump_anim"}
        for child_id in ("player_base_sprite", "player_walk_anim", "player_jump_anim"):
            child = b.get(child_id)
            assert child.spawned_by_item_id == "player_acclimation_locomotion"
            assert child.slice_id == "slice_0_acclimation"

        assert b.get("player_base_sprite").state == "awaiting_review"
        assert b.get("player_walk_anim").state == "blocked"
        assert b.get("player_jump_anim").state == "blocked"

        # Council approves player_base_sprite -> unblocks walk & jump animations
        b.apply_review(review("player_base_sprite", "approve"))
        assert sch.run_until_idle(5.0)
        assert b.get("player_walk_anim").state == "awaiting_review"
        assert b.get("player_jump_anim").state == "awaiting_review"
        assert b.get("player_acclimation_locomotion").state == "blocked"

        # Council approves walk & jump animations -> automatically unblocks parent task!
        b.apply_review(review("player_walk_anim", "approve"))
        b.apply_review(review("player_jump_anim", "approve"))
        assert b.get("player_acclimation_locomotion").state == "ready"

        # Parent resumes from its saved lead trace, runs Turn 1 & Turn 2, and reaches awaiting_review
        assert sch.run_until_idle(5.0)
        assert b.get("player_acclimation_locomotion").state == "awaiting_review"
        b.apply_review(review("player_acclimation_locomotion", "approve"))
        assert b.is_drained()
        assert b.is_slice_drained("slice_0_acclimation")
        assert any(et == "backlog_item_spawned_tasks" for et, _, _, _ in events)
    finally:
        workers_mod._dispatch_to_specialist = orig_dispatch


def test_writers_room_does_not_generate_upfront_portraits_or_barks_and_expands_slices(tmp_path):
    """Verify Writers' Room brainstorming stores character sheets and dialogue barks as lightweight metadata
    without calling generate_portrait or synthesize_voice_line upfront, initializes Slice 0 (player acclimation
    & mechanics focus), and advances to Slice 1 via run_slice_expansion."""
    from engine.studio.models import CharacterSheet, DialogueBarkSpec, GameBibleSpec, GameSliceSpec, WritersRoomTurnProposal
    from engine.studio.writers_room import CreativeWritersRoom

    portrait_calls = []
    voice_calls = []

    class _SpyVisualAgent:
        client = object()

        def generate_portrait(self, *args, **kwargs):
            portrait_calls.append((args, kwargs))

    class _SpyAudioAgent:
        client = object()

        def synthesize_voice_line(self, *args, **kwargs):
            voice_calls.append((args, kwargs))

    b = ProductionBoard(tmp_path / "backlog.json")
    room = CreativeWritersRoom(
        client=object(),
        visual_agent=_SpyVisualAgent(),
        audio_agent=_SpyAudioAgent(),
        board=b,
    )
    bible = GameBibleSpec(deliberation="Testing iterative slices and zero upfront asset bloat.", game_title="Echoes of the Reed")

    proposal = WritersRoomTurnProposal(
        deliberation="Strategizing Slice 0 and Slice 1 while adding character and bark metadata without upfront asset synthesis.",
        spoken_dialogue="Let's focus Slice 0 on getting the player acclimated to wind-reed movement and resonance, then expand in Slice 1.",
        added_or_updated_slices=[
            GameSliceSpec(
                deliberation="Slice 0 acclimates the player to the reed marsh and core wind-gliding mechanics.",
                slice_id="slice_0_marsh_acclimation",
                slice_index=0,
                title="Slice 0: Marsh Acclimation & Wind-Reed Mechanics",
                slice_objective="Get the player acclimated to the marsh world and master the core wind-reed movement verb.",
                player_acclimation_and_mechanics_focus="Diegetic wind ripples teach reed-bending and glide timing in a calm opening basin.",
                expansion_from_prior_slice="Initial playable slice.",
                scope_boundaries=["Only opening marsh basin and core reed-bend mechanic"],
                status="active",
            ),
            GameSliceSpec(
                deliberation="Slice 1 expands Slice 0 by introducing tidal currents and multi-reed harmonic chains.",
                slice_id="slice_1_tidal_harmonics",
                slice_index=1,
                title="Slice 1: Tidal Currents & Harmonic Chains",
                slice_objective="Expand the marsh basin into tidal channels where reed-bending chains across moving water.",
                player_acclimation_and_mechanics_focus="Combines learned reed-bending with rising/falling tidal gates.",
                expansion_from_prior_slice="Expands Slice 0's single-reed glide into multi-node harmonic chaining.",
                scope_boundaries=["Adds tidal channel zone and harmonic chain mechanic"],
                status="planned",
            ),
        ],
        added_or_updated_characters=[
            CharacterSheet(
                deliberation="Protagonist metadata only during brainstorming.",
                character_id="reed_walker",
                name="Sora",
                role="protagonist",
            )
        ],
        added_or_updated_barks=[
            DialogueBarkSpec(
                deliberation="Bark metadata only during brainstorming.",
                bark_id="bark_wind_01",
                speaker_id="reed_walker",
                speaker_name="Sora",
                trigger_type="environmental",
                text="The reeds are humming.",
                voice_clip="assets/audio/bark_wind_01.wav",
            )
        ],
        backlog_additions=[
            P("marsh_basin_slice0_mechanic", kind="mechanic_code", modules=["mechanics.gd"]),
        ],
    )

    room._apply_turn_proposal(
        game_bible=bible,
        speaker_role="LevelAndProgressionDesigner",
        speaker_display_name="Level, Progression & Goals Designer",
        proposal=proposal,
        candidate_dir=tmp_path,
    )

    # Zero upfront portrait or voice synthesis during brainstorming!
    assert portrait_calls == []
    assert voice_calls == []
    assert len(bible.slice_roadmap) == 2
    active = bible.get_active_slice()
    assert active is not None and active.slice_id == "slice_0_marsh_acclimation"
    assert b.get("marsh_basin_slice0_mechanic").slice_id == "slice_0_marsh_acclimation"


def test_blocks_items_and_link_existing_prerequisite_blocks_mechanic_until_sprite_approved():
    b = ProductionBoard()
    # Turn 1: GameplayDesigner proposes mechanic_grip_pump_controller (before VisualArtDirector has spoken)
    b.add_items([P("mechanic_grip_pump_controller", kind="mechanic_code", modules=["entities.gd"])], "GameplayDesigner")
    assert b.get("mechanic_grip_pump_controller").state == "ready"

    # Turn 4: VisualArtDirector proposes visual_asset_steward_rig with blocks_items=["mechanic_grip_pump_controller"]
    steward_proposal = P("visual_asset_steward_rig", kind="visual_asset").model_copy(
        update={"blocks_items": ["mechanic_grip_pump_controller"]}
    )
    b.add_items([steward_proposal], "VisualArtDirector")

    # mechanic_grip_pump_controller is now automatically blocked by visual_asset_steward_rig!
    assert "visual_asset_steward_rig" in b.get("mechanic_grip_pump_controller").depends_on
    assert b.get("mechanic_grip_pump_controller").state == "blocked"
    assert b.get("visual_asset_steward_rig").state == "ready"

    # Only visual_asset_steward_rig can be claimed until it is approved
    claimed = b.claim_ready(limit=5)
    assert [i.item_id for i in claimed] == ["visual_asset_steward_rig"]
    b.submit_result("visual_asset_steward_rig", WorkerResult())
    assert b.get("mechanic_grip_pump_controller").state == "blocked"
    b.apply_review(review("visual_asset_steward_rig", "approve"))
    assert b.get("mechanic_grip_pump_controller").state == "ready"


def test_persistence_roundtrip_resets_failed_items_on_resume(tmp_path):
    """When a persisted backlog.json is reloaded on resume, any item left in state='failed' with failure_count>=2
    is re-queued to ready/blocked with failure_count=0 so _handle_failed_items does not abort the resumed build."""
    path = tmp_path / "backlog.json"
    b = ProductionBoard(path)
    b.add_items([P("dep_item"), P("failed_item", ["dep_item"])], "GameplayDesigner")
    b.claim_ready()
    b.fail_item("dep_item", "Error 1")
    b.requeue_failed("dep_item")
    b.claim_ready()
    b.fail_item("dep_item", "Error 2")
    assert b.get("dep_item").state == "failed"
    assert b.get("dep_item").failure_count == 2

    # Reload board from disk as happens during Resume
    b_resumed = ProductionBoard(path)
    reloaded = b_resumed.get("dep_item")
    assert reloaded.state == "ready"
    assert reloaded.failure_count == 0
    assert "Error 2" in reloaded.failure


def test_final_turn_qa_assertion_failure_preserves_verified_swe_code_and_rendered_filmstrip(tmp_path):
    """When swe_engineer passes verification on Turn 0 and qa_test_engineer renders the filmstrip on the final turn
    (Turn 1 of max_turns=2) before failing a QA state assertion, run_task_lead must NOT roll back the verified GDScript
    changes to task_baseline_snap, must preserve the rendered filmstrip proof, and must surface the QA assertion error
    in the result notes."""
    from PIL import Image
    from engine.studio.models import (
        ModuleFileSpec,
        SpecialistReport,
        StudioAgentError,
        TaskLeadDecision,
        TrackCodeSynthesisPayload,
        WorkItem,
    )
    from engine.studio.workers import WorkerContext, run_task_lead
    import engine.studio.workers as workers_mod

    (tmp_path / "mechanics.gd").write_text("# baseline mechanics\n", encoding="utf-8")
    (tmp_path / "scenes.gd").write_text("# baseline scenes\n", encoding="utf-8")
    (tmp_path / "main.gd").write_text("extends Node2D\n", encoding="utf-8")
    (tmp_path / "project.godot").write_text("config_version=5\n", encoding="utf-8")
    (tmp_path / "main.tscn").write_text("[gd_scene format=3]\n", encoding="utf-8")

    # Also verify TrackCodeSynthesisPayload normalizes canonical root modules prefixed with 'scripts/'
    payload = TrackCodeSynthesisPayload(
        deliberation="Normalize canonical root module paths.",
        files=[
            ModuleFileSpec(
                deliberation="Write mechanics at root.",
                filename="scripts/mechanics.gd",
                content="# verified mechanics v1\n",
            )
        ],
    )
    payload.assemble_cartridge_directory(tmp_path, swe_agent=None)
    assert (tmp_path / "mechanics.gd").read_text(encoding="utf-8") == "# verified mechanics v1\n"
    assert not (tmp_path / "scripts" / "mechanics.gd").exists()
    # Reset back to baseline for run_task_lead test
    (tmp_path / "mechanics.gd").write_text("# baseline mechanics\n", encoding="utf-8")

    class _FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, model, contents, config=None):
            self.calls += 1
            if self.calls == 1:
                dec = TaskLeadDecision(
                    deliberation="Turn 0: Delegate to swe_engineer.",
                    current_strategy="Implement mechanic in mechanics.gd.",
                    proof_strategy="Capture filmstrip in qa_test_engineer.",
                    specialist_agent="swe_engineer",
                    brief="Implement mechanic.",
                    target_paths=["scripts/mechanics.gd"],
                    action="delegate",
                )
            else:
                dec = TaskLeadDecision(
                    deliberation="Turn 1: Delegate to qa_test_engineer.",
                    current_strategy="Verify mechanic with filmstrip.",
                    proof_strategy="Capture filmstrip across ticks.",
                    specialist_agent="qa_test_engineer",
                    brief="Capture filmstrip.",
                    action="delegate",
                )

            class _Resp:
                text = dec.model_dump_json()

            return _Resp()

    class _FakeClient:
        __module__ = "google.genai.fake"

        def __init__(self):
            self.models = _FakeModels()

    class _FakeSwe:
        def implement_requirement_track(self, *args, **kwargs):
            return {}

    item = WorkItem(
        deliberation="d",
        item_id="mech_test_item",
        kind="mechanic_code",
        title="Mechanic Test",
        spec="Implement mechanic",
        proof_spec="Filmstrip proof",
        target_modules=["scripts/mechanics.gd"],
        state="in_progress",
    )
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=_FakeClient(),
        swe_agent=_FakeSwe(),
        code_verify=lambda c_dir, track: None,
    )

    orig_dispatch = workers_mod._dispatch_to_specialist

    def _fake_dispatch(w_ctx, w_item, decision, turn_index, _lock_already_held=False):
        if decision.specialist_agent == "swe_engineer":
            (w_ctx.cartridge_dir / "mechanics.gd").write_text("# verified swe code\n", encoding="utf-8")
            return SpecialistReport(
                turn_index=turn_index,
                specialist_agent="swe_engineer",
                brief=decision.brief,
                summary="Implemented mechanics.gd; verification passed.",
                touched_files=["mechanics.gd"],
                proof_images=[],
                verification_passed=True,
            )
        # Final turn qa_test_engineer: saves filmstrip PNG first, then raises state assertion StudioAgentError
        item_dir = w_ctx.cartridge_dir / "artifacts" / "work_items" / w_item.item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        strip_p = item_dir / f"filmstrip_{w_item.item_id}_rev0.png"
        Image.new("RGB", (64, 36), (20, 40, 60)).save(strip_p)
        raise StudioAgentError("QA filmstrip state assertions failed against live engine snapshots: tick 60: expected flow 0.3, got 1")

    workers_mod._dispatch_to_specialist = _fake_dispatch
    try:
        res = run_task_lead(ctx, item, max_turns=2)
    finally:
        workers_mod._dispatch_to_specialist = orig_dispatch

    # Verified SWE code must NOT be rolled back to '# baseline mechanics\n'!
    assert (tmp_path / "mechanics.gd").read_text(encoding="utf-8") == "# verified swe code\n"
    assert len(res.proof_images) == 1
    assert Path(res.proof_images[0]).name == "filmstrip_mech_test_item_rev0.png"
    assert "QA filmstrip state assertions failed" in res.notes


def test_council_refinement_enrich_and_split_pre_execution(tmp_path):
    from engine.studio.models import TaskShapingBatch, WorkItemRefinement

    # Rule 4: deliberation strictly first, action strictly last on WorkItemRefinement
    ref_fields = list(WorkItemRefinement.model_fields.keys())
    assert ref_fields[0] == "deliberation"
    assert ref_fields[-1] == "action"
    batch_fields = list(TaskShapingBatch.model_fields.keys())
    assert batch_fields[0] == "deliberation"

    events = []
    board = ProductionBoard(tmp_path / "backlog.json", event_callback=lambda t, p: events.append((t, p)))
    board.add_items(
        [
            P("rig_asset", kind="visual_asset"),
            P("core_system", kind="mechanic_code", modules=("mechanics.gd",)),
            P("final_integration", deps=("core_system",), kind="integration", modules=("main.gd",)),
        ],
        raised_by_role="GameplayDesigner",
    )

    # 1. UXExperienceDesigner and AudioAtmosphereDesigner enrich `core_system` before any worker claims it
    shaped = board.apply_refinements(
        [
            WorkItemRefinement(
                deliberation="Needs discoverable placement toolbar and non-overlapping HUD layout.",
                item_id="core_system",
                spec_addition_or_rewrite="Expose interactive placement tools on the unified CanvasLayer HUD without overlapping panels.",
                proof_spec_update="Show placement toolbar and non-overlapping HUD in filmstrip.",
                add_depends_on=["rig_asset"],
                add_target_modules=["ui_hud.gd"],
                action="enrich",
            )
        ],
        by_role="UXExperienceDesigner",
    )
    assert len(shaped) == 1
    core = board.get("core_system")
    assert core.state == "blocked"  # now waits on rig_asset
    assert "rig_asset" in core.depends_on
    assert "ui_hud.gd" in core.target_modules
    assert len(core.shaping_notes) == 1
    assert "[UXExperienceDesigner Direction]: Expose interactive placement tools" in core.shaping_notes[0]
    assert "[UXExperienceDesigner Direction]: Expose interactive placement tools" in core.spec
    assert "[UXExperienceDesigner]: Show placement toolbar" in core.proof_spec

    # 2. GameplaySystemsEngineer splits `core_system` into `sim_engine` and `placement_tools` before execution
    split_out = board.apply_refinements(
        [
            WorkItemRefinement(
                deliberation="Separate pure simulation physics from interactive placement tool UI.",
                item_id="core_system",
                split_items=[
                    P("sim_engine", kind="mechanic_code", modules=("mechanics.gd",)),
                    P("placement_tools", deps=("sim_engine",), kind="ui", modules=("ui_hud.gd",)),
                ],
                action="split",
            )
        ],
        by_role="GameplaySystemsEngineer",
    )
    assert {i.item_id for i in split_out} == {"sim_engine", "placement_tools"}
    assert board.get("core_system").state == "cancelled"
    # Both split items inherit the parent's `rig_asset` dependency and prior shaping notes
    assert "rig_asset" in board.get("sim_engine").depends_on
    assert any("UXExperienceDesigner Direction" in n for n in board.get("sim_engine").shaping_notes)
    assert any("Split from core_system by GameplaySystemsEngineer" in n for n in board.get("sim_engine").shaping_notes)
    # Downstream `final_integration` automatically re-pointed from `core_system` to `sim_engine` and `placement_tools`
    assert set(board.get("final_integration").depends_on) == {"sim_engine", "placement_tools"}
    event_types = [t for t, _ in events]
    assert "backlog_item_shaped" in event_types
    assert "backlog_item_split" in event_types

    # 3. Deliberation-only enrich (when spec_addition_or_rewrite is empty) and enriching an already-split parent
    board.add_items(
        [P("mechanic_tug_flight_controller", kind="mechanic_code", modules=("mechanics.gd",))],
        raised_by_role="GameplayDesigner",
    )
    shaped_delib = board.apply_refinements(
        [
            WorkItemRefinement(
                deliberation="Tug flight controller must expose velocity vector and tether tension in get_state_snapshot().",
                item_id="mechanic_tug_flight_controller",
                spec_addition_or_rewrite="",
                action="enrich",
            ),
            WorkItemRefinement(
                deliberation="Audio cues must bind to both split tasks.",
                item_id="core_system",
                spec_addition_or_rewrite="Trigger acoustic feedback on state change.",
                action="enrich",
            ),
        ],
        by_role="Director",
    )
    tug = board.get("mechanic_tug_flight_controller")
    assert any("velocity vector and tether tension" in n for n in tug.shaping_notes)
    assert "velocity vector and tether tension" in tug.spec
    assert any("Trigger acoustic feedback" in n for n in board.get("sim_engine").shaping_notes)
    assert any("Trigger acoustic feedback" in n for n in board.get("placement_tools").shaping_notes)
    assert len(shaped_delib) == 3


def test_audio_proof_wav_never_wrapped_as_image_part(tmp_path):
    from PIL import Image
    from engine.studio.context_curator import ContextCuratorAgent
    from engine.studio.models import RevisionRecord, WorkItem
    from engine.studio.workers import WorkerContext, _build_task_lead_multimodal_parts
    from engine.studio.writers_room import CreativeWritersRoom

    waveform_png = tmp_path / "waveform_audio_cockpit_rev0.png"
    Image.new("RGB", (900, 220), (16, 18, 28)).save(waveform_png, "PNG")
    wav_file = tmp_path / "cockpit_kinesthetic_sfx.wav"
    wav_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

    ctx = WorkerContext(cartridge_dir=tmp_path, client=None)
    parts = _build_task_lead_multimodal_parts(ctx, [str(waveform_png), str(wav_file)])
    assert len(parts) == 2
    assert "waveform_audio_cockpit_rev0.png" in parts[0]

    item = WorkItem(
        deliberation="d",
        item_id="audio_cockpit_kinesthetic_sfx",
        kind="audio_asset",
        title="Cockpit SFX",
        spec="Generate SFX",
        proof_spec="Waveform",
        state="awaiting_review",
        revisions=[
            RevisionRecord(
                revision_index=0,
                proof_paths=[str(waveform_png), str(wav_file)],
            )
        ],
    )
    room = CreativeWritersRoom(client=None)
    review_parts = room._proof_parts(item, tmp_path)
    assert len(review_parts) == 2
    assert "waveform_audio_cockpit_rev0.png" in review_parts[0]

    board = ProductionBoard()
    board._items[item.item_id] = item
    curator = ContextCuratorAgent(cartridge_dir=tmp_path, client=None, board=board)
    chunks = curator.semantic_index.extract_corpus_chunks(board=board)
    board_chunk = next(c for c in chunks if c.chunk_id == "board:item:audio_cockpit_kinesthetic_sfx")
    assert board_chunk.associated_image_path == str(waveform_png)


def test_two_stage_full_council_task_direction_and_deduplication(tmp_path):
    from engine.studio.models import GameBibleSpec, TaskShapingBatch, WorkItemRefinement
    from engine.studio.writers_room import CreativeWritersRoom

    board = ProductionBoard(tmp_path / "backlog.json")
    board.add_items(
        [
            P("mechanic_tug_flight_controller", kind="mechanic_code", modules=("mechanics.gd",)),
            P("asset_visual_crucible_pod", kind="visual_asset"),
        ],
        raised_by_role="GameplaySystemsEngineer",
    )

    # 1. Same-role deduplication: if GameplayDesigner enriches mechanic_tug_flight_controller twice,
    # it updates its single `[GameplayDesigner Direction]` note in-place instead of creating duplicates.
    board.apply_refinements(
        [
            WorkItemRefinement(
                deliberation="Initial gameplay note.",
                item_id="mechanic_tug_flight_controller",
                spec_addition_or_rewrite="Include flight assist toggle.",
                action="enrich",
            )
        ],
        by_role="GameplayDesigner",
    )
    board.apply_refinements(
        [
            WorkItemRefinement(
                deliberation="Updated gameplay note with torque compensation.",
                item_id="mechanic_tug_flight_controller",
                spec_addition_or_rewrite="Include flight assist toggle and asymmetric torque counter-compensation.",
                action="enrich",
            )
        ],
        by_role="GameplayDesigner",
    )
    tug = board.get("mechanic_tug_flight_controller")
    gd_notes = [n for n in tug.shaping_notes if n.startswith("[GameplayDesigner Direction]:")]
    assert len(gd_notes) == 1
    assert "asymmetric torque counter-compensation" in gd_notes[0]

    # 2. Verify Stage 2 `_items_needing_domain_direction` and `_run_full_council_direction_pass`
    room = CreativeWritersRoom(client=None, board=board)
    # VisualArtDirector must review both `mechanic_tug_flight_controller` and `asset_visual_crucible_pod` (raised by GameplaySystemsEngineer)
    vad_targets = [i.item_id for i in room._items_needing_domain_direction("VisualArtDirector")]
    assert set(vad_targets) == {"mechanic_tug_flight_controller", "asset_visual_crucible_pod"}
    # GameplayDesigner already has `[GameplayDesigner Direction]` on `mechanic_tug_flight_controller`, so it is not re-requested
    gd_targets = [i.item_id for i in room._items_needing_domain_direction("GameplayDesigner")]
    assert "mechanic_tug_flight_controller" not in gd_targets

    def _fake_structured(prompt, schema, validate_fn, **kwargs):
        # Extract role and required task IDs from prompt
        tasks_section = prompt.split("TASKS REQUIRING YOUR DOMAIN REVIEW", 1)[-1]
        refs = []
        for iid in ("mechanic_tug_flight_controller", "asset_visual_crucible_pod"):
            if f"[{iid}]" in tasks_section:
                if "(AudioAtmosphereDesigner)" in prompt and iid == "asset_visual_crucible_pod":
                    refs.append(
                        WorkItemRefinement(
                            deliberation="Pure visual sprite; no audio requirement.",
                            item_id=iid,
                            action="no_input",
                        )
                    )
                else:
                    refs.append(
                        WorkItemRefinement(
                            deliberation=f"Domain direction for {iid}.",
                            item_id=iid,
                            spec_addition_or_rewrite=f"Ensure domain contract for {iid}.",
                            action="enrich",
                        )
                    )
        batch = TaskShapingBatch(
            deliberation="Reviewed all required tasks.",
            backlog_additions=[],
            backlog_refinements=refs,
        )
        assert validate_fn(batch) == []
        return batch

    room._structured_with_validation = _fake_structured
    bible = GameBibleSpec(deliberation="d", game_title="Sky Foundry", core_genre="Simulation")
    room._run_full_council_direction_pass(bible, "dna", "snapshot")

    # `mechanic_tug_flight_controller` now has input from GameplayDesigner, LevelAndProgressionDesigner, UXExperienceDesigner, VisualArtDirector, and AudioAtmosphereDesigner
    tug_after = board.get("mechanic_tug_flight_controller")
    assert len(tug_after.shaping_notes) == 5
    # `asset_visual_crucible_pod` now has input from VisualArtDirector and UXExperienceDesigner
    pod_after = board.get("asset_visual_crucible_pod")
    assert any(n.startswith("[VisualArtDirector Direction]:") for n in pod_after.shaping_notes)
    assert any(n.startswith("[UXExperienceDesigner Direction]:") for n in pod_after.shaping_notes)


def test_council_review_cross_task_triage_spawns_prerequisite_and_prunes_scope(tmp_path):
    from PIL import Image

    proof_img = tmp_path / "proof.png"
    Image.new("RGB", (64, 36), (10, 20, 30)).save(proof_img)

    board = ProductionBoard(tmp_path / "backlog.json", default_max_revisions=3)
    board.add_items(
        [
            P("camera_rig", kind="mechanic_code", modules=("scenes/camera/train_camera.gd",)),
            P("mech_a", kind="mechanic_code", modules=("scenes/mechanics/mech_a.gd", "scenes/camera/train_camera.gd", "scenes.gd")),
            P("mech_b", kind="mechanic_code", modules=("scenes/mechanics/mech_b.gd", "scenes/camera/train_camera.gd", "scenes.gd")),
        ],
        raised_by_role="GameplaySystemsEngineer",
    )

    # Approve camera_rig initially
    board.claim_ready(limit=1)
    board.submit_result("camera_rig", WorkerResult(proof_images=[str(proof_img)], notes="camera built"))
    board.apply_review(ReviewDecision(deliberation="ok", item_id="camera_rig", verdict="approve"))

    # Claim and submit mech_a
    board.claim_ready(limit=1)
    board.submit_result("mech_a", WorkerResult(proof_images=[str(proof_img)], notes="mech_a attempt 1"))

    # Council reviews mech_a: notices the defect is in shared train_camera.gd and scenes.gd, spawns a dedicated fix task
    reviewed_a = board.apply_review(
        ReviewDecision(
            deliberation="The off-screen framing and backdrop seam belong to shared camera/scene scripts, not mech_a.",
            item_id="mech_a",
            feedback="Once the shared camera/backdrop fix is approved, re-capture the mech_a filmstrip.",
            spawned_prerequisite_tasks=[
                WorkItemProposal(
                    deliberation="Fix shared camera bounds and backdrop mirroring.",
                    item_id="fix_shared_camera_and_backdrop",
                    kind="mechanic_code",
                    title="Fix Shared Camera & Backdrop",
                    spec="Only include coupled bodies in camera bounds; fix backdrop mirroring.",
                    proof_spec="Filmstrip showing tug centered and seamless backdrop.",
                    target_modules=["scenes/camera/train_camera.gd", "scenes.gd"],
                    blocks_items=["mech_a", "mech_b"],
                )
            ],
            out_of_scope_modules_to_prune=["scenes/camera/train_camera.gd", "scenes.gd"],
            verdict="request_changes",
        )
    )

    # 1. mech_a is now blocked on fix_shared_camera_and_backdrop AND its revision cap was NOT penalized!
    assert reviewed_a.state == "blocked"
    assert "fix_shared_camera_and_backdrop" in reviewed_a.depends_on
    assert reviewed_a.cap_base_index == 1
    assert board._counted_revisions(reviewed_a) == 0
    assert reviewed_a.target_modules == ["scenes/mechanics/mech_a.gd"]
    assert set(reviewed_a.excluded_modules) == {"scenes/camera/train_camera.gd", "scenes.gd"}
    assert any("[Council Scope Boundary]" in n for n in reviewed_a.shaping_notes)

    # 2. Sibling mech_b was also automatically linked to fix_shared_camera_and_backdrop and had shared modules pruned!
    mech_b = board.get("mech_b")
    assert mech_b.state == "blocked"
    assert "fix_shared_camera_and_backdrop" in mech_b.depends_on
    assert mech_b.target_modules == ["scenes/mechanics/mech_b.gd"]
    assert set(mech_b.excluded_modules) == {"scenes/camera/train_camera.gd", "scenes.gd"}

    # 3. Expand target modules refuses to re-add excluded modules
    board.expand_item_target_modules("mech_a", ["scenes.gd", "scenes/camera/train_camera.gd"])
    assert board.get("mech_a").target_modules == ["scenes/mechanics/mech_a.gd"]

    # 4. Once fix_shared_camera_and_backdrop is built and approved, mech_a transitions back to changes_requested
    # and mech_b transitions to ready!
    claimed_fix = board.claim_ready(limit=2)
    assert [c.item_id for c in claimed_fix] == ["fix_shared_camera_and_backdrop"]
    board.submit_result("fix_shared_camera_and_backdrop", WorkerResult(proof_images=[str(proof_img)], notes="fixed"))
    board.apply_review(ReviewDecision(deliberation="fixed", item_id="fix_shared_camera_and_backdrop", verdict="approve"))

    assert board.get("mech_a").state == "changes_requested"
    assert board.get("mech_b").state == "ready"


def test_yield_on_file_lock_sets_waiting_for_lock_status_and_resumes_when_released(tmp_path):
    board = ProductionBoard(tmp_path / "backlog.json")
    board.add_items(
        [
            P("level_corridor", kind="level_map", modules=("scripts/world/corridor.gd", "scenes.gd")),
            P("mech_transfer", kind="mechanic_code", modules=("scripts/mechanics/transfer.gd", "scenes.gd")),
        ],
        raised_by_role="LevelAndProgressionDesigner",
    )

    # Claim level_corridor on rev 0, submit, and request changes so it enters rev 1
    board.claim_ready(limit=1, locked_modules=set())
    board.submit_result("level_corridor", WorkerResult(proof_images=[], notes="rev0"))
    board.apply_review(review("level_corridor", "request_changes", "Fix beacon visibility"))
    assert board.get("level_corridor").state == "changes_requested"

    # Now both level_corridor and mech_transfer are claimed concurrently (distinct primary modules)
    claimed = board.claim_ready(limit=2, locked_modules=set())
    assert {c.item_id for c in claimed} == {"level_corridor", "mech_transfer"}

    # level_corridor tries to lock scenes.gd while mech_transfer holds it -> yields with file_lock_busy:scenes.gd
    yielded = board.yield_item("level_corridor", reason="file_lock_busy:scenes.gd,level_map.gd")
    assert yielded.state == "waiting_for_lock"
    assert yielded.waiting_on_locks == ["level_map.gd", "scenes.gd"]
    assert "waiting_locks=['level_map.gd', 'scenes.gd']" in board.snapshot_text()

    # While scenes.gd is still locked, level_corridor stays in waiting_for_lock and cannot be claimed
    assert board.claim_ready(limit=1, locked_modules={"scenes.gd"}) == []
    assert board.get("level_corridor").state == "waiting_for_lock"

    # Once scenes.gd is released, level_corridor is claimed and waiting_on_locks is cleared
    resumed = board.claim_ready(limit=1, locked_modules=set())
    assert [c.item_id for c in resumed] == ["level_corridor"]
    assert resumed[0].state == "in_progress"
    assert resumed[0].waiting_on_locks == []


def test_slice_assembly_load_image_part_and_narrative_character_spec_alias(tmp_path):
    """Verifies that CreativeWritersRoom._load_image_part loads PNG filmstrips into multimodal Parts
    during _run_slice_assembly_check without raising AttributeError, and NarrativeCharacterSpec is importable."""
    from PIL import Image
    from engine.studio.models import (
        CharacterSheet,
        GameBibleSpec,
        GameSliceSpec,
        NarrativeCharacterSpec,
        NextSliceStrategyPlan,
        SliceAssemblyDecision,
        SliceStateReflectionTurn,
    )
    from engine.studio.writers_room import CreativeWritersRoom

    assert NarrativeCharacterSpec is CharacterSheet

    # Rule 4: Rationale-First field order invariants on SliceStateReflectionTurn & NextSliceStrategyPlan
    refl_fields = list(SliceStateReflectionTurn.model_fields.keys())
    assert refl_fields[0] == "deliberation"
    assert refl_fields[-1] == "current_slice_quality_score"
    strat_fields = list(NextSliceStrategyPlan.model_fields.keys())
    assert strat_fields[0] == "deliberation"

    (tmp_path / "project.godot").write_text('[application]\nrun/main_scene="res://main.tscn"\n', encoding="utf-8")
    (tmp_path / "main.tscn").write_text('[gd_scene format=3]\n[node name="Main" type="Node2D"]\n', encoding="utf-8")
    (tmp_path / "main.gd").write_text("extends Node2D\n", encoding="utf-8")

    board = ProductionBoard(tmp_path / "backlog.json")
    room = CreativeWritersRoom(client=object(), board=board)
    room.enable_slice_assembly_check = True

    def _fake_filmstrip(c_dir, ticks, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 90), (30, 50, 70)).save(out_path, "PNG")

    room.assembly_filmstrip_fn = _fake_filmstrip

    captured_parts = []

    def _fake_structured(prompt, schema, validate_fn, **kwargs):
        if schema is SliceAssemblyDecision:
            captured_parts.extend(kwargs.get("parts") or [])
            dec = SliceAssemblyDecision(
                deliberation="Inspected rendered filmstrip; slice is assembled.",
                unmounted_or_disconnected_systems=[],
                playability_and_fun_evaluation="Responsive and clear.",
                spoken_dialogue="Slice assembly verified with attached filmstrip.",
                backlog_additions=[],
                is_slice_assembled=True,
            )
            assert validate_fn(dec) == []
            return dec
        raise AssertionError(f"Unexpected schema {schema}")

    room._structured_with_validation = _fake_structured

    bible = GameBibleSpec(
        deliberation="d",
        game_title="Test Game",
        core_genre="Simulation",
        slice_roadmap=[
            GameSliceSpec(
                deliberation="d",
                slice_id="slice_0_core",
                slice_index=0,
                title="Slice 0",
                slice_objective="Core loop",
                player_acclimation_and_mechanics_focus="Controls",
                status="active",
            )
        ],
        active_slice_index=0,
    )
    spawned = room._run_slice_assembly_check(bible, "Player DNA", tmp_path, pass_index=0)
    assert spawned == 0
    assert len(captured_parts) == 1
    assert hasattr(captured_parts[0], "inline_data")


def test_slice_reflection_roundtable_and_next_slice_strategy_in_slice_expansion(tmp_path):
    """Verifies that run_slice_expansion empirically inspects the current slice, runs the 5-specialist
    SliceStateReflectionTurn roundtable, synthesizes the Creative Director's NextSliceStrategyPlan across
    cohesion, how-to-play, mechanical depth, short/long-term goals, and sense of accomplishment, updates
    the Living Game Bible, and drains the Slice 1 backlog."""
    from PIL import Image
    from engine.studio.models import (
        CoreLoopSpec,
        EngagementPlan,
        GameBibleSpec,
        GameSliceSpec,
        GoalHierarchy,
        GoalSpec,
        NextSliceStrategyPlan,
        OnboardingPlan,
        OnboardingStep,
        ReviewBatch,
        ReviewDecision,
        SliceStateReflectionTurn,
        WritersRoomTurnProposal,
    )
    from engine.studio.writers_room import CreativeWritersRoom

    (tmp_path / "project.godot").write_text('[application]\nrun/main_scene="res://main.tscn"\n', encoding="utf-8")
    (tmp_path / "main.tscn").write_text('[gd_scene format=3]\n[node name="Main" type="Node2D"]\n', encoding="utf-8")
    (tmp_path / "main.gd").write_text("extends Node2D\n", encoding="utf-8")

    proof_img = tmp_path / "proof_slice1.png"
    Image.new("RGB", (64, 36), (40, 80, 120)).save(proof_img, "PNG")

    board = ProductionBoard(tmp_path / "backlog.json")
    # Seed and approve a Slice 0 item so the board starts with Slice 0 completed
    board.add_items([P("slice0_core", kind="mechanic_code", modules=("mechanics.gd",))], "GameplayDesigner", slice_id="slice_0_core")
    board.claim_ready()
    board.submit_result("slice0_core", WorkerResult(proof_images=[str(proof_img)], notes="done"))
    board.apply_review(ReviewDecision(deliberation="ok", item_id="slice0_core", verdict="approve"))
    assert board.is_drained()

    def _worker(item):
        return WorkerResult(proof_images=[str(proof_img)], notes=f"built {item.item_id}")

    sched = BoardScheduler(board, {"mechanic_code": _worker, "ui": _worker, "integration": _worker}, max_workers=2)
    events = []
    checkpoints = []
    room = CreativeWritersRoom(
        client=object(),
        board=board,
        scheduler=sched,
        drain_poll_s=0.02,
        event_callback=lambda t, p: events.append((t, p)),
        slice_checkpoint_callback=lambda gb, sl, cd: checkpoints.append((sl.slice_id, sl.status)),
    )
    room.enable_slice_reflection_roundtable = True
    room.assembly_filmstrip_fn = lambda c_dir, ticks, out_path: Image.new("RGB", (128, 72), (20, 30, 40)).save(out_path, "PNG")

    reflection_roles_called = []

    def _fake_structured(prompt, schema, validate_fn, **kwargs):
        if schema is SliceStateReflectionTurn:
            # Verify multimodal filmstrip part was attached to the reflection turn
            assert any(hasattr(p, "inline_data") for p in (kwargs.get("parts") or []))
            for r in ("UXExperienceDesigner", "GameplayDesigner", "LevelAndProgressionDesigner", "VisualArtDirector", "GameplaySystemsEngineer"):
                if f"({r}). Your specialty focus:" in prompt:
                    reflection_roles_called.append(r)
                    break
            turn = SliceStateReflectionTurn(
                deliberation="Evaluating current slice from first principles.",
                honest_current_state_critique="Center of viewport is occluded by a static control panel; world feels disconnected.",
                how_to_play_and_intuitiveness_gaps="Player lacks interactive onboarding prompts and clear causal feedback.",
                depth_and_fun_gaps="Sliders have no dynamic world hazards or meaningful resource trade-offs.",
                accomplishment_and_goals_gaps="Short-term and long-term goals are not visibly celebrated on completion.",
                strategy_for_next_slice="Dock compact controls to peripheral HUD, add interactive 60s tutorial + hazard depth + milestone rewards.",
                spoken_dialogue="We need to open up the center viewport, teach the controls diegetically, and add real hazard trade-offs and milestone payoffs.",
                addressed_to=["CreativeDirector"],
                current_slice_quality_score=0.35,
            )
            assert validate_fn(turn) == []
            return turn
        if schema is NextSliceStrategyPlan:
            plan = NextSliceStrategyPlan(
                deliberation="Synthesizing Council reflection into Slice 1 strategy.",
                spoken_dialogue="Council, for Slice 1 we are opening the viewport, adding guided onboarding, hazard depth, and milestone rewards.",
                current_slice_honest_diagnosis="Slice 0 built raw controls in a center box without cohesive world framing, onboarding, or rewarding goals.",
                cohesion_and_world_integration_strategy="Move instruments to a peripheral dock so the living 2D world fills 80% of the viewport.",
                how_to_play_and_onboarding_strategy="Add step-by-step diegetic control callouts and immediate world reaction in the first 60 seconds.",
                depth_and_interesting_decisions_strategy="Introduce dynamic environmental pressure anomalies requiring trade-offs between speed and stability.",
                short_term_and_long_term_goals_strategy="Display active short-term objective tracker and long-term shift progression bar.",
                sense_of_accomplishment_and_reward_strategy="Trigger visual/audio milestone celebration and unlock upgraded capability on each completed goal.",
                next_slice_spec=GameSliceSpec(
                    deliberation="Slice 1 transforms the prototype into a cohesive, intuitive, deep game.",
                    slice_id="slice_1_cohesive_depth_and_goals",
                    slice_index=1,
                    title="Slice 1: Cohesive World, Onboarding & Strategic Depth",
                    slice_objective="Deliver an unobstructed living world, intuitive onboarding, dynamic hazard trade-offs, and rewarding short/long-term goals.",
                    player_acclimation_and_mechanics_focus="Interactive 60s onboarding + dynamic hazard trade-offs.",
                    expansion_from_prior_slice="Replaces center-blocking panel with peripheral HUD, adds interactive tutorial, hazard depth, and goal rewards.",
                    scope_boundaries=["Peripheral HUD & world view", "Interactive onboarding & goal celebrations"],
                    status="active",
                ),
                updated_core_loop=CoreLoopSpec(
                    deliberation="Updated loop with clear risk/reward depth.",
                    player_verbs=["Throttle", "Stabilize", "Deploy Countermeasure"],
                    player_agency_and_optimization_model="Balance speed vs stability across shifting strata.",
                    moment_to_moment_loop="Read world hazard -> adjust control -> observe immediate physical response.",
                    minute_to_minute_loop="Complete short-term milestone while balancing stability vs. speed.",
                    session_loop="Advance through multi-stage progression tiers and unlock upgrades.",
                ),
                updated_onboarding_plan=OnboardingPlan(
                    deliberation="Updated onboarding plan.",
                    world_introduction="Camera frames the living world with peripheral instruments.",
                    purpose_and_stakes_framing="Highlight first short-term objective with diegetic prompt.",
                    causal_chain_and_feedback_strategy="Controls visually drive world machinery.",
                    steps=[
                        OnboardingStep(
                            deliberation="s0",
                            step_index=0,
                            concept_introduced="Primary throttle",
                            diegetic_method="Highlighted prompt",
                        ),
                        OnboardingStep(
                            deliberation="s1",
                            step_index=1,
                            concept_introduced="Stability margin",
                            diegetic_method="Peripheral gauge pulse",
                        ),
                        OnboardingStep(
                            deliberation="s2",
                            step_index=2,
                            concept_introduced="Milestone completion",
                            diegetic_method="Celebration banner",
                        ),
                    ],
                ),
                updated_goal_hierarchy=GoalHierarchy(
                    deliberation="Updated multi-horizon goals.",
                    goals=[
                        GoalSpec(
                            deliberation="g1",
                            goal_id="goal_short_1",
                            horizon="short_term",
                            description="Reach Milestone 1 while surviving pressure surge",
                            reward="Unlock stabilizer upgrade",
                        ),
                        GoalSpec(
                            deliberation="g2",
                            goal_id="goal_long_1",
                            horizon="long_term",
                            description="Complete full expedition shift across all strata",
                            reward="Mastery rank",
                        ),
                    ],
                    progression_structure="Top-right objective tracker with milestone celebration banner.",
                ),
                updated_engagement_plan=EngagementPlan(
                    deliberation="Updated engagement plan.",
                    entertainment_pillars=["Tactile mastery", "Risk-reward tension"],
                    variety_injectors=["Dynamic pressure anomalies"],
                    pacing_and_intensity_curve="Escalates across strata with relief at milestone checkpoints.",
                ),
                updated_diegetic_ui_philosophy="Compact peripheral dock keeping 80%+ of the 16:9 viewport unobstructed.",
                backlog_additions=[],
            )
            assert validate_fn(plan) == []
            return plan
        if schema is WritersRoomTurnProposal:
            # Verify the Master Strategy context block is injected into specialist decomposition prompts
            assert "MASTER STRATEGY FOR SLICE #1" in prompt
            if "You are the Lead Gameplay & Godot Systems Engineer (GameplaySystemsEngineer)" in prompt:
                prop = WritersRoomTurnProposal(
                    deliberation="Seed Slice 1 implementation task.",
                    spoken_dialogue="Seeding `slice1_world_onboarding_and_depth` for Slice 1.",
                    addressed_to=["CreativeDirector"],
                    backlog_additions=[
                        P("slice1_world_onboarding_and_depth", kind="mechanic_code", modules=("mechanics.gd", "ui_hud.gd"))
                    ],
                )
            else:
                prop = WritersRoomTurnProposal(
                    deliberation="Specialist Slice 1 design contribution.",
                    spoken_dialogue="Contributing Slice 1 design updates.",
                    addressed_to=["GameplaySystemsEngineer"],
                    backlog_additions=[],
                )
            assert validate_fn(prop) == []
            return prop
        if schema is ReviewBatch:
            awaiting = [i.item_id for i in board.awaiting_review()]
            batch = ReviewBatch(
                deliberation="Approve Slice 1 task.",
                spoken_dialogue="Slice 1 task verified.",
                review_decisions=[ReviewDecision(deliberation="ok", item_id=iid, verdict="approve") for iid in awaiting],
            )
            assert validate_fn(batch) == []
            return batch
        raise AssertionError(f"Unexpected schema {schema}")

    room._structured_with_validation = _fake_structured

    bible = GameBibleSpec(
        deliberation="d",
        game_title="Deep Shift",
        core_genre="Simulation",
        slice_roadmap=[
            GameSliceSpec(
                deliberation="d",
                slice_id="slice_0_core",
                slice_index=0,
                title="Slice 0",
                slice_objective="Initial prototype",
                player_acclimation_and_mechanics_focus="Basic controls",
                status="active",
            )
        ],
        active_slice_index=0,
    )

    try:
        updated_bible = room.run_slice_expansion(bible, candidate_dir=tmp_path)
    finally:
        sched.stop()

    assert reflection_roles_called == [
        "UXExperienceDesigner",
        "GameplayDesigner",
        "LevelAndProgressionDesigner",
        "VisualArtDirector",
        "GameplaySystemsEngineer",
    ]
    assert len(updated_bible.slice_roadmap) == 2
    assert updated_bible.slice_roadmap[0].status == "completed"
    assert updated_bible.slice_roadmap[1].slice_id == "slice_1_cohesive_depth_and_goals"
    assert updated_bible.slice_roadmap[1].status == "completed"
    assert updated_bible.onboarding_plan is not None
    assert "diegetic prompt" in updated_bible.onboarding_plan.purpose_and_stakes_framing
    assert updated_bible.goal_hierarchy is not None
    assert "Milestone 1" in updated_bible.goal_hierarchy.goals[0].description
    assert board.get("slice1_world_onboarding_and_depth").state == "approved"
    assert board.get("slice1_world_onboarding_and_depth").slice_id == "slice_1_cohesive_depth_and_goals"
    assert ("slice_1_cohesive_depth_and_goals", "completed") in checkpoints
    event_types = [t for t, _ in events]
    assert "slice_strategy_synthesized" in event_types
    assert "slice_completed" in event_types


def test_resumed_drained_slice_with_optional_user_feedback(tmp_path: Path):
    """Verifies that resuming a drained Slice 0 cartridge in `run_writers_room` advances to Slice 1
    both when optional user feedback is provided (via `creative_prompt` or `.pending_steering.json`)
    and passes that feedback into the Slice State Reflection Roundtable, `NextSliceStrategyPlan`,
    and specialist task decomposition."""
    from PIL import Image
    from engine.studio.models import (
        CoreLoopSpec,
        EngagementPlan,
        ExperienceStoryboardBeat,
        GameBibleSpec,
        GameSliceSpec,
        GenesisWorldSpec,
        GoalHierarchy,
        GoalSpec,
        LevelArchetypeSpec,
        LevelGenerationStrategy,
        NextSliceStrategyPlan,
        OnboardingPlan,
        OnboardingStep,
        ReviewBatch,
        ReviewDecision,
        SliceStateReflectionTurn,
        WritersRoomTurnProposal,
        WritersRoomUtterance,
    )
    from engine.studio.writers_room import CreativeWritersRoom

    proof_img = tmp_path / "proof0.png"
    Image.new("RGB", (64, 36), (30, 60, 90)).save(proof_img, "PNG")

    board = ProductionBoard(tmp_path / "backlog.json")
    board.add_items(
        [P("slice0_item", kind="mechanic_code", modules=("main.gd",))],
        "GameplaySystemsEngineer",
        slice_id="slice_0_full_core_loop",
    )
    board.claim_ready()
    board.submit_result("slice0_item", WorkerResult(proof_images=[str(proof_img)], notes="done"))
    board.apply_review(ReviewDecision(deliberation="ok", item_id="slice0_item", verdict="approve"))
    assert board.is_drained()

    sched = BoardScheduler(
        board,
        {
            "mechanic_code": lambda it: WorkerResult(
                proof_images=[str(proof_img)],
                notes=f"Implemented {it.item_id}",
            )
        },
        max_workers=2,
    )

    room = CreativeWritersRoom(
        client=object(),
        board=board,
        scheduler=sched,
        drain_poll_s=0.01,
        drain_timeout_s=5.0,
        max_parallel_calls=1,
    )
    room.enable_slice_reflection_roundtable = True
    room.max_assembly_passes = 0

    user_feedback = "Make the onboarding intuitive in the first 30 seconds and give the player clear short-term and long-term goals."
    reflection_prompts_seen: list[str] = []
    strategy_prompts_seen: list[str] = []
    decomposition_prompts_seen: list[str] = []

    def _fake_structured(prompt, schema, validate_fn, **kwargs):
        if schema is SliceStateReflectionTurn:
            reflection_prompts_seen.append(prompt)
            turn = SliceStateReflectionTurn(
                deliberation="Reflecting on Slice 0 with optional Executive Producer feedback.",
                honest_current_state_critique="Slice 0 is a disconnected control box without clear goals.",
                how_to_play_and_intuitiveness_gaps="Needs 30-second interactive onboarding.",
                depth_and_fun_gaps="Needs meaningful trade-offs.",
                accomplishment_and_goals_gaps="Needs explicit short-term and long-term goals.",
                strategy_for_next_slice="Implement interactive 30s tutorial and multi-horizon goal HUD.",
                spoken_dialogue="Incorporating user feedback and our diagnosis for Slice 1.",
                addressed_to=["CreativeDirector"],
                current_slice_quality_score=0.4,
            )
            assert validate_fn(turn) == []
            return turn
        if schema is NextSliceStrategyPlan:
            strategy_prompts_seen.append(prompt)
            plan = NextSliceStrategyPlan(
                deliberation="Synthesizing Council reflection and Executive Producer feedback into Slice 1 strategy.",
                spoken_dialogue="Locking Slice 1 strategy.",
                current_slice_honest_diagnosis="Slice 0 lacked onboarding and clear goals.",
                cohesion_and_world_integration_strategy="Integrate controls into peripheral HUD around the world.",
                how_to_play_and_onboarding_strategy="Interactive 30-second diegetic tutorial.",
                depth_and_interesting_decisions_strategy="Add dynamic hazard trade-offs.",
                short_term_and_long_term_goals_strategy="Add short-term milestone and long-term shift tracker.",
                sense_of_accomplishment_and_reward_strategy="Celebrate completed milestones with upgrades.",
                next_slice_spec=GameSliceSpec(
                    deliberation="Slice 1 spec.",
                    slice_id="slice_1_feedback_driven",
                    slice_index=1,
                    title="Slice 1: Guided Onboarding & Goal Depth",
                    slice_objective="Deliver 30s onboarding and multi-horizon goals.",
                    player_acclimation_and_mechanics_focus="Guided 30s onboarding + hazard trade-offs.",
                    expansion_from_prior_slice="Adds 30s tutorial and goal hierarchy.",
                    scope_boundaries=["Onboarding + Goal HUD"],
                    status="active",
                ),
                backlog_additions=[],
            )
            assert validate_fn(plan) == []
            return plan
        if schema is WritersRoomTurnProposal:
            decomposition_prompts_seen.append(prompt)
            if "You are the Lead Gameplay & Godot Systems Engineer (GameplaySystemsEngineer)" in prompt:
                prop = WritersRoomTurnProposal(
                    deliberation="Seed Slice 1 task.",
                    spoken_dialogue="Seeding `slice1_feedback_task`.",
                    addressed_to=["CreativeDirector"],
                    backlog_additions=[
                        P("slice1_feedback_task", kind="mechanic_code", modules=("main.gd",))
                    ],
                )
            else:
                prop = WritersRoomTurnProposal(
                    deliberation="Design contribution.",
                    spoken_dialogue="Contributing Slice 1 design.",
                    addressed_to=["GameplaySystemsEngineer"],
                    backlog_additions=[],
                )
            assert validate_fn(prop) == []
            return prop
        if schema is ReviewBatch:
            awaiting = [i.item_id for i in board.awaiting_review()]
            batch = ReviewBatch(
                deliberation="Approve Slice 1 task.",
                spoken_dialogue="Approved.",
                review_decisions=[ReviewDecision(deliberation="ok", item_id=iid, verdict="approve") for iid in awaiting],
            )
            assert validate_fn(batch) == []
            return batch
        raise AssertionError(f"Unexpected schema {schema}")

    room._structured_with_validation = _fake_structured

    existing_bible = GameBibleSpec(
        deliberation="d",
        status="locked",
        game_title="Caisson Shift",
        core_genre="Simulation",
        elevator_pitch="Operate a subterranean caisson through hazardous strata.",
        core_fantasy="Master heavy subterranean machinery under pressure.",
        first_60_seconds_hook="Drop into the caisson and engage the primary auger.",
        diegetic_ui_philosophy="Peripheral industrial gauges.",
        visual_style_guide="Crisp 2D cross-section.",
        audio_direction_guide="Heavy hydraulic hums.",
        gameplay_pillars=["Pressure management", "Strata excavation"],
        level_generation_strategy=LevelGenerationStrategy(
            deliberation="d",
            generation_approach="Vertical strata cross-section",
            world_structure="Multi-layer subterranean shaft",
            level_archetypes=[
                LevelArchetypeSpec(
                    deliberation="a1",
                    archetype_id="alluvial_silt",
                    purpose="Onboarding layer",
                    visual_identity="Warm silt",
                    signature_challenge="Steady excavation",
                )
            ],
            generation_rules=["Always place milestone markers every 10m"],
            difficulty_and_pacing_curve="Escalates with depth.",
        ),
        core_loop=CoreLoopSpec(
            deliberation="d",
            player_verbs=["Excavate", "Stabilize", "Vent"],
            player_agency_and_optimization_depth="Balance excavation speed against pore pressure.",
            moment_to_moment_loop="Adjust auger and slurry pressure -> watch caisson sink.",
            minute_to_minute_loop="Clear a 10m stratum ring and lock segments.",
            session_loop="Complete full shift down to bedrock.",
        ),
        onboarding_plan=OnboardingPlan(
            deliberation="d",
            world_introduction="Cross-section view of the caisson shaft.",
            purpose_and_stakes_framing="Reach target depth before shift timer expires.",
            causal_chain_and_feedback_strategy="Gauges and shaft vibrations reflect pressure.",
            steps=[
                OnboardingStep(deliberation="s0", step_index=0, concept_introduced="Auger", diegetic_method="Prompt"),
                OnboardingStep(deliberation="s1", step_index=1, concept_introduced="Slurry", diegetic_method="Gauge"),
                OnboardingStep(deliberation="s2", step_index=2, concept_introduced="Stroke", diegetic_method="Marker"),
            ],
        ),
        goal_hierarchy=GoalHierarchy(
            deliberation="d",
            goals=[
                GoalSpec(deliberation="g1", goal_id="g_short", horizon="short_term", description="Sink 5m", reward="Seal"),
                GoalSpec(deliberation="g2", goal_id="g_mid", horizon="mid_term", description="Clear clay layer", reward="Upgrade"),
                GoalSpec(deliberation="g3", goal_id="g_long", horizon="long_term", description="Reach bedrock", reward="Victory"),
            ],
            progression_structure="Depth milestones.",
        ),
        engagement_plan=EngagementPlan(
            deliberation="d",
            entertainment_pillars=["Tactile machinery", "Depth tension"],
            variety_injectors=["Gas pockets", "Quick-clay surges"],
            pacing_and_intensity_curve="Escalating pressure with depth.",
        ),
        experience_storyboard=[
            ExperienceStoryboardBeat(
                deliberation="b1",
                beat_index=0,
                phase_title="Opening",
                player_emotional_state="Weight and power",
                sensory_and_camera_staging="Caisson shaft cross-section",
                diegetic_teaching_moment="Prompt to engage auger",
                narrative_integration="Shift 800 begins",
            ),
        ],
        slice_roadmap=[
            GameSliceSpec(
                deliberation="d",
                slice_id="slice_0_full_core_loop",
                slice_index=0,
                title="Slice 0",
                slice_objective="Initial build",
                player_acclimation_and_mechanics_focus="Core controls",
                status="completed",
            )
        ],
        active_slice_index=0,
        writers_room_transcript=[
            WritersRoomUtterance(
                deliberation="d",
                turn_index=0,
                speaker_role="CreativeDirector",
                speaker_display_name="Creative Director",
                spoken_dialogue="Locked Living Game Bible for production",
                bible_updates_summary=["Locked Living Game Bible for production"],
            )
        ],
        world_spec=GenesisWorldSpec(
            deliberation="d",
            game_title="Caisson Shift",
            core_genre="Simulation",
            thematic_premise="Subterranean caisson operation.",
            aesthetic_style="Industrial 2D",
            palette_mood="grim_dark",
            world_lore="Deep excavation.",
            conflict_tone="High-pressure survival",
            conflict_narrative="Survive pore pressure surges while sinking the caisson.",
            gameplay_pacing="methodical",
            pacing_score=0.5,
            target_difficulty=0.6,
            mechanics_guidelines=["Balance pressure and excavation."],
        ),
    )

    try:
        result_bible = room.run_writers_room(
            existing_bible=existing_bible,
            candidate_dir=tmp_path,
            creative_prompt=user_feedback,
        )
    finally:
        sched.stop()

    assert len(result_bible.slice_roadmap) == 2
    assert result_bible.slice_roadmap[1].slice_id == "slice_1_feedback_driven"
    assert result_bible.slice_roadmap[1].status == "completed"
    assert any(user_feedback in p for p in reflection_prompts_seen)
    assert any(user_feedback in p for p in strategy_prompts_seen)
    assert any(user_feedback in p for p in decomposition_prompts_seen)
    assert any(
        u.speaker_role == "ExecutiveProducer" and user_feedback in u.spoken_dialogue
        for u in result_bible.writers_room_transcript
    )


def test_new_game_preempts_paused_old_candidate_build(tmp_path: Path):
    """Verifies that starting a new game from scratch while an older candidate build is paused
    marks the older candidate aborted BEFORE unpausing, so the old build raises StudioPausedError
    instead of waking up and resuming."""
    import json
    from engine.studio.models import (
        StudioPausedError,
        clear_candidate_build_aborted,
        is_candidate_build_aborted,
        set_active_gemini_dataset_context,
        wait_if_studio_paused,
    )
    from engine.ui.server import SwarmEventBus

    cand_old = tmp_path / "candidate_genesis_old123"
    cand_old.mkdir(parents=True)
    pause_sentinel = tmp_path / ".studio_paused.json"
    pause_sentinel.write_text(json.dumps({"paused": True}), encoding="utf-8")

    set_active_gemini_dataset_context(dataset_name="old_run", candidate_dir=cand_old)
    assert not is_candidate_build_aborted(cand_old)

    bus = SwarmEventBus()
    bus._active_build_id = "build_candidate_genesis_old123"

    caught_errors: List[Exception] = []

    def _paused_old_worker() -> None:
        try:
            wait_if_studio_paused(candidate_dir=cand_old, label="CreativeWritersRoom[SliceStateReflectionTurn]")
        except Exception as exc:
            caught_errors.append(exc)

    worker_thread = threading.Thread(target=_paused_old_worker, daemon=True)
    worker_thread.start()
    time.sleep(0.05)

    # Simulate commissioning a New Game from Scratch: reset_for_new_genesis_build FIRST, then unlink pause sentinel
    bus.reset_for_new_genesis_build(
        build_id="build_gen_new999",
        creative_prompt="A bright, fun factory automation game",
    )
    assert is_candidate_build_aborted(cand_old)
    pause_sentinel.unlink(missing_ok=True)

    worker_thread.join(timeout=2.0)
    assert not worker_thread.is_alive()
    assert len(caught_errors) == 1
    assert isinstance(caught_errors[0], StudioPausedError)
    clear_candidate_build_aborted(cand_old)


def test_drain_backlog_handles_concurrent_worker_failure_during_review_and_cancelled_dependents(tmp_path):
    """Regression test:
    1. When a background worker transitions an item (`task_b`) to `state='failed'` (`failure_count=1`) while
       `_run_review_round` is synchronously reviewing `task_a` (with `task_c` blocked on `task_b` and
       `scheduler.running_count() == 0`), `_drain_backlog` must NOT throw 'Backlog cannot drain: remaining items
       are blocked with nothing in flight'—it must re-queue `task_b` and drain all items cleanly.
    2. When the Council issues `CapDecision(action='cancel')` on a `needs_decision` item, `_drain_backlog`
       unblocks any `blocked_by_cancelled()` downstream dependents instead of deadlocking.
    3. `_format_worker_failure_excerpt` preserves the head of long `StudioAgentError` QA gate messages.
    """
    from engine.studio.models import CapDecision, GameBibleSpec, ReviewBatch, ReviewDecision, WorkItemProposal
    from engine.studio.workers import _format_worker_failure_excerpt
    from engine.studio.writers_room import CreativeWritersRoom

    long_err = (
        "Traceback (most recent call last):\n"
        "  File \"workers.py\", line 3910, in run_task_lead\n"
        "engine.studio.models.StudioAgentError: Task Lead for 'task_b' exhausted 6 turns with failing QA gate: "
        "ERROR from qa_test_engineer: Counterfactual QA check failed: specific root-cause headline here. "
        + ("frame_telemetry_padding " * 60)
    )
    excerpt = _format_worker_failure_excerpt(long_err, max_chars=400)
    assert excerpt.startswith("engine.studio.models.StudioAgentError: Task Lead for 'task_b'")
    assert "specific root-cause headline here" in excerpt

    def _mk_prop(iid: str, deps: Optional[List[str]] = None, mods: Optional[List[str]] = None) -> WorkItemProposal:
        return WorkItemProposal(
            deliberation=f"Proposal for {iid}",
            item_id=iid,
            kind="mechanic_code",
            title=f"Title {iid}",
            spec=f"Spec for {iid}",
            proof_spec="Filmstrip",
            depends_on=list(deps or []),
            target_modules=list(mods or []),
        )

    board = ProductionBoard(tmp_path / "backlog.json", default_max_revisions=1)
    board.add_items(
        [
            _mk_prop("task_a", mods=["a.gd"]),
            _mk_prop("task_b", mods=["b.gd"]),
            _mk_prop("task_c", deps=["task_b"], mods=["c.gd"]),
            _mk_prop("opt_prereq", mods=["opt.gd"]),
            _mk_prop("dep_after_opt", deps=["opt_prereq"], mods=["dep.gd"]),
        ],
        "GameplaySystemsEngineer",
    )

    attempts: Dict[str, int] = {"task_b": 0, "opt_prereq": 0}
    review_entered = threading.Event()
    allow_review_finish = threading.Event()

    def _mechanic_handler(item: WorkItem) -> WorkerResult:
        if item.item_id == "task_b":
            attempts["task_b"] += 1
            if attempts["task_b"] == 1:
                review_entered.wait(timeout=3.0)
                raise StudioAgentError("Transient first-attempt QA failure on task_b")
        return WorkerResult(notes=f"completed {item.item_id}")

    scheduler = BoardScheduler(board, {"mechanic_code": _mechanic_handler}, max_workers=2, poll_interval=0.01)

    class _ReviewMockClient:
        pass

    room = CreativeWritersRoom(
        client=_ReviewMockClient(),
        board=board,
        scheduler=scheduler,
        drain_poll_s=0.02,
        drain_timeout_s=10.0,
    )
    room.max_assembly_passes = 0
    room.max_blind_playtest_passes = 0

    def _fake_review_as(role, awaiting, deciding, *args, **kwargs):
        if any(i.item_id == "task_a" for i in awaiting) and attempts["task_b"] <= 1:
            review_entered.set()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if board.get("task_b").state == "failed" and scheduler.running_count() == 0:
                    break
                time.sleep(0.01)
            allow_review_finish.set()
        rev_decs = []
        for it in awaiting:
            if it.item_id == "opt_prereq":
                rev_decs.append(
                    ReviewDecision(
                        deliberation="Reject opt_prereq so it hits revision cap and escalates.",
                        item_id=it.item_id,
                        feedback="Unneeded optional module.",
                        verdict="request_changes",
                    )
                )
            else:
                rev_decs.append(
                    ReviewDecision(
                        deliberation="Looks solid.",
                        item_id=it.item_id,
                        feedback="",
                        verdict="approve",
                    )
                )
        cap_decs = [
            CapDecision(
                deliberation="Cancel optional prerequisite; downstream item can proceed independently.",
                item_id=it.item_id,
                action="cancel",
            )
            for it in deciding
        ]
        return ReviewBatch(
            deliberation="Review round completed.",
            spoken_dialogue="Reviewed.",
            review_decisions=rev_decs,
            cap_decisions=cap_decs,
        )

    room._review_as = _fake_review_as  # type: ignore[method-assign]
    bible = GameBibleSpec(
        deliberation="Test bible",
        game_title="Test Drain Resilience",
        elevator_pitch="Resilience test",
        core_genre="Simulation",
        status="in_production",
    )

    try:
        room._drain_backlog(bible, "Test DNA", tmp_path)
    finally:
        scheduler.stop()

    assert board.is_drained()
    assert attempts["task_b"] == 2
    assert board.get("task_a").state == "approved"
    assert board.get("task_b").state == "approved"
    assert board.get("task_c").state == "approved"
    assert board.get("opt_prereq").state == "cancelled"
    assert board.get("dep_after_opt").state == "approved"


def test_blind_playtest_mode_skips_scripted_qa_playthrough_wrapper(tmp_path):
    """Verifies that when the end-of-slice Blind AI Playtester is enabled, `_run_slice_assembly_check`
    runs fast static entrypoint reachability + zero-input boot sanity (`input_script=[]`), skips generating
    any `tests/test_slice_playtest_*.gd` wrapper script or `SlicePlaytestRoundtableTurn` LLM calls, and
    returns 0 so `_drain_backlog` hands straight off to `_run_end_of_slice_blind_playtest`."""
    from types import SimpleNamespace
    from engine.studio.models import GameBibleSpec, GameSliceSpec
    from engine.studio.writers_room import CreativeWritersRoom

    (tmp_path / "project.godot").write_text(
        '[application]\nconfig/name="SanityTest"\nrun/main_scene="res://main.tscn"\n',
        encoding="utf-8",
    )
    (tmp_path / "main.tscn").write_text(
        '[gd_scene load_steps=2 format=3]\n'
        '[ext_resource type="Script" path="res://main.gd" id="1"]\n'
        '[node name="Main" type="Node2D"]\n'
        'script = ExtResource("1")\n',
        encoding="utf-8",
    )
    (tmp_path / "main.gd").write_text("extends Node2D\nfunc get_state_snapshot() -> Dictionary:\n\treturn {'ok': true}\n", encoding="utf-8")

    llm_calls: list[str] = []

    def _forbid_llm(*_a, **_kw):
        llm_calls.append("called")
        raise AssertionError("Scripted QA playtest planner / roundtable LLM must not be called when Blind Playtester takes over")

    live_like_client = SimpleNamespace(models=SimpleNamespace(generate_content=_forbid_llm))
    board = ProductionBoard(tmp_path / "backlog.json")
    events: list[tuple[str, dict]] = []
    room = CreativeWritersRoom(
        client=live_like_client,
        board=board,
        event_callback=lambda et, payload: events.append((et, payload)),
    )
    room.enable_end_of_slice_blind_playtest = True

    bible = GameBibleSpec(
        deliberation="d",
        game_title="Sanity Game",
        core_genre="Simulation",
        slice_roadmap=[
            GameSliceSpec(
                deliberation="d",
                slice_id="slice_0_core",
                slice_index=0,
                title="Slice 0",
                slice_objective="Core loop",
                player_acclimation_and_mechanics_focus="Controls",
                status="active",
            )
        ],
        active_slice_index=0,
    )

    spawned = room._run_slice_assembly_check(bible, "Player DNA", tmp_path, pass_index=0)
    assert spawned == 0
    assert llm_calls == []
    assert not list((tmp_path / "tests").glob("test_slice_playtest_*.gd")) if (tmp_path / "tests").exists() else True
    completed_ev = [p for et, p in events if et == "slice_assembly_check_completed"]
    assert len(completed_ev) == 1
    assert completed_ev[0]["is_slice_assembled"] is True
    assert completed_ev[0]["mode"] == "static_boot_sanity"


def test_task_lead_archives_prior_failed_trace_and_timeline_exposes_retries(tmp_path):
    """Verifies that when a prior Task Lead attempt leaves `lead_trace_rev0.json` with `status='failed'`,
    the auto-retried `run_task_lead` archives it as `lead_trace_rev0_failed1.json` instead of clobbering it,
    and `/api/studio/board-item/{item_id}/timeline` returns both Attempt #1 (`is_failed_retry=True`) and
    Attempt #2 (`is_failed_retry=False`)."""
    import json
    from starlette.testclient import TestClient
    from engine.studio.models import SpecialistReport, TaskLeadDecision, WorkItemProposal
    from engine.studio.workers import WorkerContext, run_task_lead
    from engine.ui.server import create_app

    cand_dir = tmp_path / "cartridges" / "candidate_genesis_retry"
    work_dir = cand_dir / "artifacts" / "work_items" / "item_retry_demo"
    work_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "game_bible.json").write_text(
        json.dumps({"deliberation": "d", "game_title": "Retry Demo", "core_genre": "Puzzle"}),
        encoding="utf-8",
    )
    proof_img = work_dir / "proof_ok.png"
    from PIL import Image
    Image.new("RGB", (32, 32), (10, 120, 80)).save(proof_img, "PNG")

    board = ProductionBoard(cand_dir / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="d",
                item_id="item_retry_demo",
                kind="mechanic_code",
                title="Retry Demo Task",
                spec="Implement mechanic",
                proof_spec="Verify mechanic",
                target_modules=["mechanics.gd"],
            )
        ],
        raised_by_role="GameplaySystemsEngineer",
    )
    item = board.claim_ready()[0]

    # Seed Attempt #1's failed trace on disk
    (work_dir / "lead_trace_rev0.json").write_text(
        json.dumps(
            {
                "item_id": "item_retry_demo",
                "revision": 0,
                "status": "failed",
                "current_strategy": "Attempt 1 strategy",
                "turns": [
                    SpecialistReport(
                        specialist_agent="qa_test_engineer",
                        turn_index=0,
                        brief="Run QA",
                        summary="ERROR from qa_test_engineer: Entrypoint Parity Gate failed",
                        verification_passed=False,
                    ).model_dump()
                ],
            }
        ),
        encoding="utf-8",
    )

    # Attempt #2 succeeds on turn 0 -> submits on turn 1
    decisions = iter(
        [
            TaskLeadDecision(
                deliberation="Fix mechanic",
                current_strategy="Attempt 2 strategy",
                proof_strategy="Capture clean proof",
                specialist_agent="swe_engineer",
                brief="Fix wiring",
                target_paths=["mechanics.gd"],
                action="delegate",
            ),
            TaskLeadDecision(
                deliberation="Submit verified result",
                current_strategy="Attempt 2 strategy",
                proof_strategy="Capture clean proof",
                submission_summary="Fixed and verified",
                selected_proof_images=[str(proof_img)],
                action="submit_to_council",
            ),
        ]
    )

    import engine.studio.workers as workers_mod
    orig_decide = workers_mod.plan_task_lead_turn
    orig_dispatch = workers_mod._dispatch_to_specialist
    try:
        workers_mod.plan_task_lead_turn = lambda *_a, **_kw: next(decisions)
        workers_mod._dispatch_to_specialist = lambda *_a, **_kw: SpecialistReport(
            specialist_agent="swe_engineer",
            turn_index=0,
            brief="Fix wiring",
            summary="Implemented fix cleanly",
            touched_files=["mechanics.gd"],
            proof_images=[str(proof_img)],
            verification_passed=True,
        )
        ctx = WorkerContext(cartridge_dir=cand_dir, client=object(), model_name="test-model", board=board)
        res = run_task_lead(ctx, item, max_turns=3)
    finally:
        workers_mod.plan_task_lead_turn = orig_decide
        workers_mod._dispatch_to_specialist = orig_dispatch

    assert res.proof_images == [str(proof_img)]
    assert (work_dir / "lead_trace_rev0_failed1.json").is_file()
    assert (work_dir / "lead_trace_rev0.json").is_file()

    app = create_app(cartridges_root=tmp_path / "cartridges", db_path=tmp_path / "telemetry.db")
    with TestClient(app) as tc:
        resp = tc.get("/api/studio/board-item/item_retry_demo/chronology?cartridge=candidate_genesis_retry")
        assert resp.status_code == 200
        payload = resp.json()
        assert len(payload["attempts"]) == 2
        assert payload["attempts"][0]["attempt_number"] == 1
        assert payload["attempts"][0]["is_failed_retry"] is True
        assert payload["attempts"][0]["status"] == "failed"
        assert "Entrypoint Parity Gate failed" in payload["attempts"][0]["turns"][0]["summary"]
        assert payload["attempts"][1]["attempt_number"] == 2
        assert payload["attempts"][1]["is_failed_retry"] is False
        assert payload["attempts"][1]["status"] == "submitted"


def test_paused_resumed_item_skips_council_reshaping_and_reports_resumed_turn_index(tmp_path: Path) -> None:
    """When an item yields mid-execution (e.g., on Pause after Turn #1) back to state='ready', resuming must:
    1) NOT treat it as an unstarted item for Council Task Shaping or `action='split'` decomposition, and
    2) emit `resumed=True` and `resumed_turn_index=1` on `backlog_item_started`."""
    from engine.studio import writers_room as wr_mod
    from engine.studio.models import SpecialistReport, WorkItemProposal, WorkItemRefinement
    import json

    cand_dir = tmp_path / "cartridges" / "candidate_resume_shape"
    cand_dir.mkdir(parents=True, exist_ok=True)
    events: list[tuple[str, dict]] = []
    board = ProductionBoard(
        path=cand_dir / "backlog.json",
        event_callback=lambda ev_type, payload: events.append((ev_type, payload)),
    )
    board.add_items(
        [
            WorkItemProposal(
                deliberation="Spawned follow-up integration task.",
                item_id="integration_followup_fix",
                kind="integration",
                title="Follow-up Integration Fix",
                spec="Wire subsystem interaction cleanly.",
                proof_spec="Filmstrip verifying end-to-end flow.",
                target_modules=["main.gd"],
            )
        ],
        raised_by_role="GameplaySystemsEngineer",
    )

    # Simulate Turn #1 completed and saved to lead_trace_rev0.json before operator paused the build
    work_dir = cand_dir / "artifacts" / "work_items" / "integration_followup_fix"
    work_dir.mkdir(parents=True, exist_ok=True)
    turn0 = SpecialistReport(
        deliberation="SWE completed Turn #1.",
        turn_index=0,
        specialist_agent="swe_engineer",
        brief="Implement fix in main.gd.",
        summary="Verified modified main.gd.",
        touched_files=["main.gd"],
        verification_passed=True,
    )
    (work_dir / "lead_trace_rev0.json").write_text(
        json.dumps(
            {
                "item_id": "integration_followup_fix",
                "revision": 0,
                "status": "yielded",
                "turns": [turn0.model_dump()],
                "collected_proofs": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert board.has_prior_execution_activity("integration_followup_fix") is True

    # Splitting a task that already has completed worker turns must be rejected
    split_issues = board.validate_refinements(
        [
            WorkItemRefinement(
                deliberation="Attempt to split an already-started task.",
                item_id="integration_followup_fix",
                split_items=[
                    WorkItemProposal(
                        deliberation="Subtask A",
                        item_id="sub_a",
                        kind="mechanic_code",
                        title="Sub A",
                        spec="Spec A",
                        proof_spec="Proof A",
                        target_modules=["scenes/mechanics/a.gd"],
                    )
                ],
                action="split",
            )
        ]
    )
    assert any("already started worker execution" in msg for msg in split_issues)

    # CreativeWritersRoom must not select this item for pre-execution domain shaping
    room = wr_mod.CreativeWritersRoom(client=None, board=board)
    item_obj = board.get("integration_followup_fix")
    assert room._has_started_execution(item_obj, candidate_dir=cand_dir) is True
    assert room._items_needing_domain_direction("UXExperienceDesigner", candidate_dir=cand_dir) == []

    # Claiming the item again emits resumed=True and resumed_turn_index=1 (Turn #2)
    claimed = board.claim_ready(limit=1)
    assert len(claimed) == 1
    started_evs = [p for et, p in events if et == "backlog_item_started" and p.get("item_id") == "integration_followup_fix"]
    assert len(started_evs) == 1
    assert started_evs[0]["resumed"] is True
    assert started_evs[0]["resumed_turn_index"] == 1


def test_qa_specialist_rejects_self_recursive_test_script(tmp_path: Path) -> None:
    """Rejects QA test scripts that call `load()` or `preload()` on their own `.gd`/`.tscn` path (preventing infinite `_ready()` recursion)."""
    from types import SimpleNamespace
    from engine.studio import workers as workers_mod
    from engine.studio.models import FilmstripPlan, StudioAgentError, WorkItemProposal

    cand_dir = tmp_path / "cartridge_recursive_qa"
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "project.godot").write_text("[application]\nrun/main_scene=\"res://main.tscn\"\n", encoding="utf-8")
    (cand_dir / "main.tscn").write_text("[gd_scene format=3]\n[node name=\"Main\" type=\"Node2D\"]\n", encoding="utf-8")

    board = ProductionBoard()
    [item] = board.add_items(
        [
            WorkItemProposal(
                deliberation="Integration verification item.",
                item_id="integration_demo_self_ref",
                kind="integration",
                title="Integration Demo",
                spec="Verify integration.",
                proof_spec="Capture filmstrip.",
                target_modules=["main.gd"],
            )
        ],
        raised_by_role="GameplaySystemsEngineer",
    )

    recursive_plan = FilmstripPlan(
        deliberation="Authoring QA wrapper scene.",
        target_scene="res://tests/test_integration_demo_self_ref.tscn",
        test_scene_gd_content=(
            "extends Node2D\n"
            "func _ready() -> void:\n"
            "\tvar overlay = Node2D.new()\n"
            "\toverlay.set_script(load(\"res://tests/test_integration_demo_self_ref.gd\"))\n"
            "\tadd_child(overlay)\n"
        ),
        capture_ticks=[1, 30, 60],
        input_segments=[],
    )

    ctx = workers_mod.WorkerContext(
        cartridge_dir=cand_dir,
        client=SimpleNamespace(),
        model_name="test-model",
        board=board,
    )
    orig_plan_fn = workers_mod.plan_filmstrip
    workers_mod.plan_filmstrip = lambda *_a, **_kw: recursive_plan
    try:
        with pytest.raises(StudioAgentError, match="Self-recursive QA test script rejected"):
            workers_mod.run_qa_test_engineer_specialist(ctx, item, brief="Verify integration")
    finally:
        workers_mod.plan_filmstrip = orig_plan_fn


def test_snapshot_text_preserves_active_items_after_many_approved_slices_and_surfaces_valid_ids():
    from engine.studio.models import WorkItemProposal, WorkItemRefinement

    board = ProductionBoard()
    approved_batch = [
        WorkItemProposal(
            deliberation="Historical slice work item.",
            item_id=f"slice_hist_item_{idx:02d}",
            kind="mechanic_code",
            title=f"Historical Slice Item {idx}",
            spec="Historical spec content.",
            proof_spec="Historical proof.",
            target_modules=[
                f"scenes/mechanics/system_{idx}_a.gd",
                f"scenes/entities/entity_{idx}_b.gd",
                f"scenes/ui/panel_{idx}_c.gd",
            ],
        )
        for idx in range(55)
    ]
    board.add_items(approved_batch, "GameplayDesigner", slice_id="slice_0_foundation")
    while True:
        claimed = board.claim_ready(limit=20)
        if not claimed:
            break
        for c in claimed:
            board.submit_result(c.item_id, WorkerResult(notes="done"))
            board.apply_review(review(c.item_id, "approve"))

    next_slice_batch = [
        WorkItemProposal(
            deliberation="Active slice work item.",
            item_id=f"slice_next_active_{idx:02d}",
            kind="mechanic_code",
            title=f"Next Slice Active Item {idx}",
            spec="Active slice implementation spec.",
            proof_spec="Active slice proof.",
            target_modules=[f"scenes/mechanics/next_{idx}.gd"],
        )
        for idx in range(12)
    ]
    board.add_items(next_slice_batch, "CreativeDirector", slice_id="slice_1_expansion")

    snap = board.snapshot_text()
    for idx in range(12):
        assert f"[slice_next_active_{idx:02d}]" in snap
    assert "- APPROVED ITEMS (55):" in snap

    prop_issues = board.validate_proposals(
        [
            WorkItemProposal(
                deliberation="Companion visual item.",
                item_id="companion_asset_pack",
                kind="visual_asset",
                title="Companion Asset Pack",
                spec="Sprites.",
                proof_spec="Plate.",
                blocks_items=["nonexistent_slug_id"],
            )
        ]
    )
    assert any(
        "blocks_items references unknown item 'nonexistent_slug_id'" in msg
        and "slice_next_active_00" in msg
        for msg in prop_issues
    )

    ref_issues = board.validate_refinements(
        [
            WorkItemRefinement(
                deliberation="Attempted enrichment with wrong slug.",
                item_id="nonexistent_slug_id",
                action="enrich",
                spec_addition_or_rewrite="Extra detail.",
            )
        ]
    )
    assert any(
        "Cannot refine unknown item 'nonexistent_slug_id'" in msg
        and "slice_next_active_00" in msg
        for msg in ref_issues
    )


def test_resuming_drained_board_with_unfinished_slice_runs_blind_playtest_instead_of_next_slice(tmp_path: Path):
    """Verifies that if all ProductionBoard items are approved and Slice Assembly already passed
    (`slice_assembly:verified`), but the active slice still has `status='active'` (`bible.status='in_production'`)
    because the end-of-slice Council Blind Playtest was interrupted, `run_writers_room` resumes directly into
    `_run_end_of_slice_blind_playtest` for the unfinished slice instead of skipping it and expanding the next slice."""
    from PIL import Image
    from engine.studio.models import (
        CoreLoopSpec,
        EngagementPlan,
        ExperienceStoryboardBeat,
        GameBibleSpec,
        GameSliceSpec,
        GenesisWorldSpec,
        GoalHierarchy,
        GoalSpec,
        LevelArchetypeSpec,
        LevelGenerationStrategy,
        OnboardingPlan,
        OnboardingStep,
        ReviewDecision,
        WritersRoomUtterance,
    )
    from engine.studio.writers_room import CreativeWritersRoom

    proof_img = tmp_path / "proof0.png"
    Image.new("RGB", (64, 36), (30, 60, 90)).save(proof_img, "PNG")

    board = ProductionBoard(tmp_path / "backlog.json")
    board.add_items(
        [P("slice1_item", kind="mechanic_code", modules=("main.gd",))],
        "GameplaySystemsEngineer",
        slice_id="slice_1_active_gate",
    )
    board.claim_ready()
    board.submit_result("slice1_item", WorkerResult(proof_images=[str(proof_img)], notes="done"))
    board.apply_review(ReviewDecision(deliberation="ok", item_id="slice1_item", verdict="approve"))
    assert board.is_drained()

    sched = BoardScheduler(board, {}, max_workers=1)
    room = CreativeWritersRoom(
        client=object(),
        board=board,
        scheduler=sched,
        drain_poll_s=0.01,
        drain_timeout_s=5.0,
        max_parallel_calls=1,
    )
    room.max_assembly_passes = 2
    room.max_blind_playtest_passes = 1

    assembly_calls: list[int] = []
    playtest_calls: list[tuple[str, int]] = []
    expansion_calls: list[str] = []

    def _spy_assembly(game_bible, dna_summary, candidate_dir, pass_index=0):
        assembly_calls.append(pass_index)
        return 0

    def _spy_blind_playtest(game_bible, dna_summary, candidate_dir, pass_index=0):
        active_s = game_bible.get_active_slice()
        playtest_calls.append((active_s.slice_id if active_s else "", pass_index))
        return 0

    def _spy_expansion(*args, **kwargs):
        expansion_calls.append("expanded")

    room._run_slice_assembly_check = _spy_assembly
    room._run_end_of_slice_blind_playtest = _spy_blind_playtest
    room.run_slice_expansion = _spy_expansion

    existing_bible = GameBibleSpec(
        deliberation="d",
        status="in_production",
        game_title="Synthetic Shift",
        core_genre="Simulation",
        elevator_pitch="Operate a kinetic relay station.",
        core_fantasy="Master high-throughput signal routing.",
        first_60_seconds_hook="Route the opening pulse through node A.",
        diegetic_ui_philosophy="Peripheral console readouts.",
        visual_style_guide="Clean 2D schematics.",
        audio_direction_guide="Crisp relay clicks.",
        gameplay_pillars=["Signal routing", "Load balancing"],
        level_generation_strategy=LevelGenerationStrategy(
            deliberation="d",
            generation_approach="Procedural sector grid",
            world_structure="Connected relay sectors",
            level_archetypes=[
                LevelArchetypeSpec(
                    deliberation="a1",
                    archetype_id="sector_alpha",
                    purpose="Core hub",
                    visual_identity="Slate grid",
                    signature_challenge="Steady throughput",
                )
            ],
            generation_rules=["Spawn hub in sector 0"],
            difficulty_and_pacing_curve="Scales with active relays.",
        ),
        core_loop=CoreLoopSpec(
            deliberation="d",
            player_verbs=["Place", "Link", "Tune"],
            player_agency_and_optimization_depth="Trade latency against bandwidth.",
            moment_to_moment_loop="Connect relay -> observe pulse.",
            minute_to_minute_loop="Stabilize sector load.",
            session_loop="Complete multi-sector network.",
        ),
        onboarding_plan=OnboardingPlan(
            deliberation="d",
            world_introduction="Central relay hub view.",
            purpose_and_stakes_framing="Power the beacon before overload.",
            causal_chain_and_feedback_strategy="Line glow reflects load.",
            steps=[
                OnboardingStep(deliberation="s0", step_index=0, concept_introduced="Place", diegetic_method="Ghost"),
                OnboardingStep(deliberation="s1", step_index=1, concept_introduced="Link", diegetic_method="Trace"),
                OnboardingStep(deliberation="s2", step_index=2, concept_introduced="Tune", diegetic_method="Dial"),
            ],
        ),
        goal_hierarchy=GoalHierarchy(
            deliberation="d",
            goals=[
                GoalSpec(deliberation="g1", goal_id="g_short", horizon="short_term", description="Link 1 node", reward="Power"),
                GoalSpec(deliberation="g2", goal_id="g_mid", horizon="mid_term", description="Stabilize hub", reward="Tier 2"),
                GoalSpec(deliberation="g3", goal_id="g_long", horizon="long_term", description="Full grid", reward="Mastery"),
            ],
            progression_structure="Sector milestones.",
        ),
        engagement_plan=EngagementPlan(
            deliberation="d",
            entertainment_pillars=["Tactile routing", "Throughput puzzles"],
            variety_injectors=["Surge events"],
            pacing_and_intensity_curve="Steady escalation.",
        ),
        experience_storyboard=[
            ExperienceStoryboardBeat(
                deliberation="b1",
                beat_index=0,
                phase_title="Boot",
                player_emotional_state="Curiosity",
                sensory_and_camera_staging="Hub overview",
                diegetic_teaching_moment="Pulse indicator",
                narrative_integration="Station online",
            ),
        ],
        slice_roadmap=[
            GameSliceSpec(
                deliberation="d",
                slice_id="slice_0_boot",
                slice_index=0,
                title="Slice 0",
                slice_objective="Initial boot",
                player_acclimation_and_mechanics_focus="Basic link",
                status="completed",
            ),
            GameSliceSpec(
                deliberation="d",
                slice_id="slice_1_active_gate",
                slice_index=1,
                title="Slice 1",
                slice_objective="Throughput expansion",
                player_acclimation_and_mechanics_focus="Load balancing",
                status="active",
            ),
        ],
        active_slice_index=1,
        writers_room_transcript=[
            WritersRoomUtterance(
                deliberation="d",
                turn_index=0,
                speaker_role="CreativeDirector",
                speaker_display_name="Creative Director",
                spoken_dialogue="Locked Living Game Bible for production",
                bible_updates_summary=["Locked Living Game Bible for production"],
            ),
            WritersRoomUtterance(
                deliberation="Assembly verified.",
                turn_index=1,
                speaker_role="GameplaySystemsEngineer",
                speaker_display_name="Lead Gameplay & Godot Systems Engineer",
                spoken_dialogue="Slice assembly verified cleanly.",
                bible_updates_summary=["slice_assembly:verified"],
            ),
        ],
        world_spec=GenesisWorldSpec(
            deliberation="d",
            game_title="Synthetic Shift",
            core_genre="Simulation",
            thematic_premise="Kinetic relay operation.",
            aesthetic_style="Clean 2D",
            palette_mood="vibrant_fantasy",
            world_lore="Signal grid.",
            conflict_tone="tactical_methodical",
            conflict_narrative="Prevent grid overload.",
            gameplay_pacing="methodical",
            pacing_score=0.5,
            target_difficulty=0.5,
            mechanics_guidelines=["Balance throughput."],
        ),
    )

    try:
        res_bible = room.run_writers_room(existing_bible=existing_bible, candidate_dir=tmp_path)
    finally:
        sched.stop()

    assert assembly_calls == []
    assert playtest_calls == [("slice_1_active_gate", 0)]
    assert expansion_calls == []
    assert len(res_bible.slice_roadmap) == 2
    assert res_bible.slice_roadmap[1].status == "completed"
    assert res_bible.status == "locked"

















