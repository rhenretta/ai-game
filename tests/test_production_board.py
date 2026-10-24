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
)


def P(item_id, deps=(), kind="visual_asset", modules=(), max_rev=None):
    return WorkItemProposal(
        deliberation="d", item_id=item_id, kind=kind, title=item_id, spec=f"build {item_id}",
        proof_spec="show it", depends_on=list(deps), target_modules=list(modules), max_revisions=max_rev,
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
