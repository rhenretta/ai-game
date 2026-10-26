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
    from engine.studio.models import RevisionRecord, WorkItem
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





