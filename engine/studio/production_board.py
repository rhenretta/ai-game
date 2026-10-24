"""Production Board: the council's backlog and the scheduler that builds it.

The council (Writers' Room) raises dependency-linked work items while designing; workers build every item
whose dependencies are approved, concurrently; each result returns with proof images that the council
reviews (approve / request changes). Items that hit the revision cap or stall escalate to an explicit
council decision. Nothing is ever approved silently (Rule 5).

Readiness, caps and stall detection are purely structural (graph state, counters, content hashes); every
judgement about quality is made by the council via structured LLM contracts (Rules 2 & 4).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from engine.studio.models import (
    CapDecision,
    ReviewDecision,
    RevisionRecord,
    StudioAgentError,
    WorkItem,
    WorkItemProposal,
)

logger = logging.getLogger("ProductionBoard")

DEFAULT_MAX_REVISIONS = 5
MAX_SPEC_RESETS = 2
BACKLOG_FILENAME = "backlog.json"
RUNNABLE_STATES = ("ready", "changes_requested")
TERMINAL_STATES = ("approved", "cancelled")
CODE_LOCK_KINDS = ("mechanic_code", "level_map", "ui", "integration")
ASSET_KINDS = ("visual_asset", "animation", "audio_asset")

_SCHEDULER_TLS = threading.local()


def is_in_board_scheduler() -> bool:
    """Returns True when the current thread is executing inside `BoardScheduler._run_item`."""
    return bool(getattr(_SCHEDULER_TLS, "active", False))


class WorkerYielded(Exception):
    """Raised by a worker when an exclusive resource (e.g. `code_lock`) is held by another worker,
    allowing `BoardScheduler` to pause this task and immediately assign the worker to another runnable task."""

    def __init__(self, reason: str = "code_lock_busy") -> None:
        super().__init__(reason)
        self.reason = reason


class BoardValidationError(StudioAgentError):
    """Raised when a proposal or decision violates a board invariant (the proposing specialist is told why)."""
    deliberation: str = "Board invariant violated; the proposal was rejected and nothing changed."


@dataclass
class WorkerResult:
    """What a worker returns for an item: proof image paths (absolute or cartridge-relative) and notes."""
    proof_images: List[str] = field(default_factory=list)
    notes: str = ""


def _hash_file(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        return f"missing:{path}"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


class ProductionBoard:
    """Thread-safe, persisted backlog with dependency tracking and a review/revision state machine."""

    def __init__(
        self,
        path: Optional[Path] = None,
        default_max_revisions: int = DEFAULT_MAX_REVISIONS,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.default_max_revisions = max(1, int(default_max_revisions))
        self.event_callback = event_callback
        self._lock = threading.RLock()
        self._items: Dict[str, WorkItem] = {}
        self._design_revision = 0
        self._yielded_on_code_lock: Set[str] = set()
        self._active_resumed_code_lock_item: Optional[str] = None
        if self.path is not None and self.path.is_file():
            self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._design_revision = int(data.get("design_revision", 0))
        for raw in data.get("items", []):
            item = WorkItem.model_validate(raw)
            if item.state == "in_progress":  # worker died with the process: re-queue
                item.state = "changes_requested" if item.revisions and item.revisions[-1].feedback else "ready"
            self._items[item.item_id] = item
        self._refresh()

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "design_revision": self._design_revision,
            "items": [i.model_dump(mode="json") for i in self._items.values()],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def _emit(self, event_type: str, item: WorkItem, **extra: Any) -> None:
        if self.event_callback is None:
            return
        payload = {
            "deliberation": f"{event_type}: {item.item_id} ({item.title})",
            "item_id": item.item_id,
            "kind": item.kind,
            "state": item.state,
            **extra,
        }
        try:
            self.event_callback(event_type, payload)
        except Exception:  # an observer must never corrupt board state, but its failure must be visible
            logger.exception("Board event callback failed for %s", event_type)

    # ------------------------------------------------------------------ queries
    def get(self, item_id: str) -> WorkItem:
        with self._lock:
            if item_id not in self._items:
                raise BoardValidationError(f"Unknown item '{item_id}'.")
            return self._items[item_id].model_copy(deep=True)

    def items(self) -> List[WorkItem]:
        with self._lock:
            return [i.model_copy(deep=True) for i in self._items.values()]

    def note_design_revision(self) -> None:
        with self._lock:
            self._design_revision += 1

    def _deps_satisfied(self, item: WorkItem) -> bool:
        return all(d in self._items and self._items[d].state == "approved" for d in item.depends_on)

    def _refresh(self) -> None:
        """Promotes blocked items whose dependencies are all approved, and demotes ready items whose dependencies changed."""
        for item in self._items.values():
            if item.state == "blocked" and self._deps_satisfied(item):
                item.state = "ready"
            elif item.state == "ready" and not self._deps_satisfied(item):
                item.state = "blocked"

    def is_drained(self) -> bool:
        """True only when every item is approved or cancelled (so no needs_decision, blocked, failed or pending work)."""
        with self._lock:
            return all(i.state in TERMINAL_STATES for i in self._items.values())

    def unresolved(self) -> List[WorkItem]:
        with self._lock:
            return [i.model_copy(deep=True) for i in self._items.values() if i.state not in TERMINAL_STATES]

    def awaiting_review(self, raised_by_role: Optional[str] = None) -> List[WorkItem]:
        with self._lock:
            return [
                i.model_copy(deep=True)
                for i in self._items.values()
                if i.state == "awaiting_review" and (raised_by_role is None or i.raised_by_role == raised_by_role)
            ]

    def needing_decision(self, raised_by_role: Optional[str] = None) -> List[WorkItem]:
        with self._lock:
            return [
                i.model_copy(deep=True)
                for i in self._items.values()
                if i.state == "needs_decision" and (raised_by_role is None or i.raised_by_role == raised_by_role)
            ]

    def blocked_by_cancelled(self) -> List[WorkItem]:
        """Items that can never run because a dependency was cancelled; the council must re-point or cancel them."""
        with self._lock:
            return [
                i.model_copy(deep=True)
                for i in self._items.values()
                if i.state == "blocked" and any(d in self._items and self._items[d].state == "cancelled" for d in i.depends_on)
            ]

    def snapshot_text(self, limit_chars: int = 6000) -> str:
        """Compact board summary for council prompts."""
        with self._lock:
            if not self._items:
                return "PRODUCTION BOARD: (empty)"
            counts: Dict[str, int] = {}
            for i in self._items.values():
                counts[i.state] = counts.get(i.state, 0) + 1
            lines = [f"PRODUCTION BOARD ({len(self._items)} items; states={counts}; drained={self.is_drained()}):"]
            for i in self._items.values():
                extra = ""
                if i.state == "needs_decision":
                    extra = f" REASON={i.decision_reason}"
                if i.accepted_compromise:
                    extra += " [accepted compromise]"
                lines.append(
                    f"- [{i.item_id}] {i.kind} '{i.title}' state={i.state} deps={i.depends_on} "
                    f"rev={len(i.revisions)}/{i.max_revisions} by={i.raised_by_role}{extra}"
                )
            text = "\n".join(lines)
            return text[:limit_chars] + ("\n...(truncated)" if len(text) > limit_chars else "")

    # ------------------------------------------------------------------ validation
    def validate_proposals(self, proposals: List[WorkItemProposal]) -> List[str]:
        """Structural validation of a batch against the board and itself (unknown deps, duplicates, cycles)."""
        with self._lock:
            return self._validate_batch(proposals, existing=set(self._items))

    def _validate_batch(self, proposals: List[WorkItemProposal], existing: Set[str]) -> List[str]:
        issues: List[str] = []
        batch_ids = [p.item_id for p in proposals]
        for dup in {i for i in batch_ids if batch_ids.count(i) > 1}:
            issues.append(f"Duplicate item_id '{dup}' in the proposal batch.")
        for p in proposals:
            if p.item_id in existing:
                issues.append(f"item_id '{p.item_id}' already exists on the board.")
            if not p.item_id.strip():
                issues.append("item_id must not be empty.")
        known = existing | set(batch_ids)
        for p in proposals:
            for d in p.depends_on:
                if d not in known:
                    issues.append(f"Item '{p.item_id}' depends on unknown item '{d}'.")
            if p.item_id in p.depends_on:
                issues.append(f"Item '{p.item_id}' depends on itself.")
        # cycle detection over (existing + batch)
        graph: Dict[str, List[str]] = {i.item_id: list(i.depends_on) for i in self._items.values()}
        for p in proposals:
            graph[p.item_id] = list(p.depends_on)
        indegree = {n: 0 for n in graph}
        children: Dict[str, List[str]] = {n: [] for n in graph}
        for n, deps in graph.items():
            for d in deps:
                if d in graph:
                    indegree[n] += 1
                    children[d].append(n)
        queue = [n for n, deg in indegree.items() if deg == 0]
        visited = 0
        while queue:
            n = queue.pop()
            visited += 1
            for c in children[n]:
                indegree[c] -= 1
                if indegree[c] == 0:
                    queue.append(c)
        if visited != len(graph):
            cyclic = sorted(n for n, deg in indegree.items() if deg > 0)
            issues.append(f"Dependency cycle among items: {cyclic}")
        return issues

    # ------------------------------------------------------------------ mutations
    def add_items(self, proposals: List[WorkItemProposal], raised_by_role: str = "") -> List[WorkItem]:
        """Adds a validated batch atomically; raises BoardValidationError (with every issue) otherwise."""
        with self._lock:
            issues = self._validate_batch(proposals, existing=set(self._items))
            if issues:
                raise BoardValidationError("; ".join(issues))
            added: List[WorkItem] = []
            for p in proposals:
                item = WorkItem(
                    deliberation=p.deliberation,
                    item_id=p.item_id,
                    kind=p.kind,
                    title=p.title,
                    spec=p.spec,
                    proof_spec=p.proof_spec,
                    depends_on=list(p.depends_on),
                    target_modules=list(p.target_modules),
                    raised_by_role=raised_by_role,
                    design_revision=self._design_revision,
                    state="blocked",
                    max_revisions=p.max_revisions or self.default_max_revisions,
                )
                self._items[item.item_id] = item
                added.append(item)
            self._refresh()
            self._persist()
            for item in added:
                self._emit("backlog_item_created", self._items[item.item_id], depends_on=item.depends_on)
            return [self._items[i.item_id].model_copy(deep=True) for i in added]

    def update_item(
        self,
        item_id: str,
        spec: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
    ) -> WorkItem:
        """Lets the council edit a not-yet-started item (e.g. re-point dependencies after a cancellation)."""
        with self._lock:
            item = self._require(item_id)
            if item.state in ("in_progress", "approved"):
                raise BoardValidationError(f"Item '{item_id}' is {item.state} and cannot be edited.")
            if depends_on is not None:
                issues: List[str] = []
                unknown = [d for d in depends_on if d not in self._items]
                if unknown:
                    issues.append(f"Unknown dependencies: {unknown}")
                if item_id in depends_on:
                    issues.append(f"Item '{item_id}' cannot depend on itself.")
                cyc = self._would_cycle(item_id, list(depends_on))
                if cyc and item_id not in depends_on:
                    issues.append(cyc)
                if issues:
                    raise BoardValidationError("; ".join(issues))
                item.depends_on = list(depends_on)
            if spec is not None:
                item.spec = spec
            self._refresh()
            self._persist()
            return item.model_copy(deep=True)

    def _would_cycle(self, item_id: str, new_deps: List[str]) -> str:
        stack = list(new_deps)
        seen: Set[str] = set()
        while stack:
            n = stack.pop()
            if n == item_id:
                return f"Re-pointing '{item_id}' would create a dependency cycle."
            if n in seen or n not in self._items:
                continue
            seen.add(n)
            stack.extend(self._items[n].depends_on)
        return ""

    def cancel_item(self, item_id: str, reason: str = "") -> WorkItem:
        with self._lock:
            item = self._require(item_id)
            if item.state == "approved":
                raise BoardValidationError(f"Item '{item_id}' is already approved and cannot be cancelled.")
            item.state = "cancelled"
            item.decision_reason = reason or item.decision_reason
            self._yielded_on_code_lock.discard(item_id)
            if self._active_resumed_code_lock_item == item_id:
                self._active_resumed_code_lock_item = None
            self._refresh()
            self._persist()
            self._emit("backlog_item_cancelled", item, reason=reason)
            return item.model_copy(deep=True)

    def requeue_failed(self, item_id: str) -> WorkItem:
        """Council/operator re-queues a failed item after reading its error."""
        with self._lock:
            item = self._require(item_id)
            if item.state != "failed":
                raise BoardValidationError(f"Item '{item_id}' is {item.state}, not failed.")
            item.state = "ready" if self._deps_satisfied(item) else "blocked"
            self._persist()
            return item.model_copy(deep=True)

    def _require(self, item_id: str) -> WorkItem:
        if item_id not in self._items:
            raise BoardValidationError(f"Unknown item '{item_id}'.")
        return self._items[item_id]

    # ------------------------------------------------------------------ worker side
    def claim_ready(
        self,
        limit: int = 1,
        locked_modules: Optional[Set[str]] = None,
        code_lock_busy: bool = False,
        prefer_parallel_assets: bool = False,
    ) -> List[WorkItem]:
        """Atomically claims up to `limit` runnable items, never two that share a target module, and none that
        touch a module in `locked_modules` (held by in-progress work).

        When `prefer_parallel_assets` or `_yielded_on_code_lock` is active:
        - Items parked in `_yielded_on_code_lock` are skipped while `code_lock_busy` is True.
        - Once `code_lock_busy` is False, at most one item from `_yielded_on_code_lock` is resumed first.
        - When `prefer_parallel_assets` is True and a code item is already in progress (or `code_lock_busy`),
          runnable asset tasks (`ASSET_KINDS`) are claimed ahead of additional code tasks so free worker slots
          stay productive instead of contending on `code_lock`.
        """
        with self._lock:
            self._refresh()
            if (
                self._active_resumed_code_lock_item is not None
                and (
                    self._active_resumed_code_lock_item not in self._items
                    or self._items[self._active_resumed_code_lock_item].state != "in_progress"
                )
            ):
                self._active_resumed_code_lock_item = None

            busy: Set[str] = set(locked_modules or set())
            in_progress_code_count = 0
            for i in self._items.values():
                if i.state == "in_progress":
                    busy.update(i.target_modules)
                    if i.kind in CODE_LOCK_KINDS:
                        in_progress_code_count += 1

            effective_code_busy = bool(code_lock_busy or self._active_resumed_code_lock_item is not None)
            claimed: List[WorkItem] = []
            order_index = {item_id: idx for idx, item_id in enumerate(self._items)}
            runnable = [i for i in self._items.values() if i.state in RUNNABLE_STATES and self._deps_satisfied(i)]
            runnable.sort(key=lambda i: (-i.priority, order_index[i.item_id]))

            if not prefer_parallel_assets and not self._yielded_on_code_lock and not effective_code_busy:
                for item in runnable:
                    if len(claimed) >= limit:
                        break
                    mods = set(item.target_modules)
                    if mods & busy:
                        continue
                    item.state = "in_progress"
                    busy.update(mods)
                    claimed.append(item.model_copy(deep=True))
                    self._emit("backlog_item_started", item, revision=len(item.revisions))
                if claimed:
                    self._persist()
                return claimed

            code_claimed_in_call = 0
            resumed_code_claimed_in_call = False
            remaining = list(runnable)

            while len(claimed) < limit and remaining:
                eligible: List[WorkItem] = []
                for cand in remaining:
                    if set(cand.target_modules) & busy:
                        continue
                    if cand.item_id in self._yielded_on_code_lock and (
                        effective_code_busy or resumed_code_claimed_in_call
                    ):
                        continue
                    eligible.append(cand)
                if not eligible:
                    break

                chosen: Optional[WorkItem] = None
                if not effective_code_busy and not resumed_code_claimed_in_call:
                    for cand in eligible:
                        if cand.item_id in self._yielded_on_code_lock:
                            chosen = cand
                            break

                if chosen is None:
                    code_active = effective_code_busy or (in_progress_code_count + code_claimed_in_call) >= 1
                    if prefer_parallel_assets and code_active:
                        for cand in eligible:
                            if cand.kind in ASSET_KINDS:
                                chosen = cand
                                break
                    if chosen is None:
                        chosen = eligible[0]

                remaining.remove(chosen)
                was_yielded = chosen.item_id in self._yielded_on_code_lock
                self._yielded_on_code_lock.discard(chosen.item_id)
                if was_yielded:
                    self._active_resumed_code_lock_item = chosen.item_id
                    resumed_code_claimed_in_call = True
                    effective_code_busy = True
                if chosen.kind in CODE_LOCK_KINDS:
                    code_claimed_in_call += 1

                chosen.state = "in_progress"
                busy.update(chosen.target_modules)
                claimed.append(chosen.model_copy(deep=True))
                self._emit(
                    "backlog_item_started",
                    chosen,
                    revision=len(chosen.revisions),
                    resumed=was_yielded,
                )

            if claimed:
                self._persist()
            return claimed

    def yield_item(self, item_id: str, reason: str = "") -> WorkItem:
        """Pauses an in-progress item back to its runnable state so the worker slot can execute another item."""
        with self._lock:
            item = self._require(item_id)
            if self._active_resumed_code_lock_item == item_id:
                self._active_resumed_code_lock_item = None
            if item.state == "in_progress":
                item.state = "changes_requested" if item.revisions and item.revisions[-1].feedback else "ready"
            self._yielded_on_code_lock.add(item_id)
            self._refresh()
            self._persist()
            self._emit(
                "backlog_item_yielded",
                item,
                reason=reason or "code_lock_busy",
                revision=len(item.revisions),
            )
            return item.model_copy(deep=True)

    def submit_result(self, item_id: str, result: WorkerResult) -> WorkItem:
        with self._lock:
            item = self._require(item_id)
            if item.state != "in_progress":
                raise BoardValidationError(f"Item '{item_id}' is {item.state}; cannot accept a worker result.")
            missing = [p for p in result.proof_images if not Path(p).is_file()]
            if missing:
                raise BoardValidationError(f"Worker for '{item_id}' reported proof images that do not exist: {missing}")
            self._yielded_on_code_lock.discard(item_id)
            if self._active_resumed_code_lock_item == item_id:
                self._active_resumed_code_lock_item = None
            item.revisions.append(
                RevisionRecord(
                    revision_index=len(item.revisions),
                    worker_notes=result.notes,
                    proof_paths=list(result.proof_images),
                    proof_hashes=[_hash_file(p) for p in result.proof_images],
                )
            )
            item.state = "awaiting_review"
            item.failure = ""
            self._persist()
            self._emit("backlog_item_proof_ready", item, proof_paths=list(result.proof_images), revision=len(item.revisions))
            return item.model_copy(deep=True)

    def fail_item(self, item_id: str, error_text: str) -> WorkItem:
        """Records a worker failure with its full error text; the item stays visible until re-queued (Rule 5)."""
        with self._lock:
            item = self._require(item_id)
            self._yielded_on_code_lock.discard(item_id)
            if self._active_resumed_code_lock_item == item_id:
                self._active_resumed_code_lock_item = None
            item.state = "failed"
            item.failure = error_text
            item.failure_count += 1
            self._persist()
            self._emit("backlog_item_failed", item, error=error_text)
            return item.model_copy(deep=True)

    # ------------------------------------------------------------------ council side
    def _counted_revisions(self, item: WorkItem) -> int:
        return len(item.revisions) - item.cap_base_index

    def apply_review(self, decision: ReviewDecision) -> WorkItem:
        with self._lock:
            item = self._require(decision.item_id)
            if item.state != "awaiting_review":
                raise BoardValidationError(f"Item '{decision.item_id}' is {item.state}, not awaiting_review.")
            if decision.verdict == "request_changes" and not decision.feedback.strip():
                raise BoardValidationError("request_changes requires concrete feedback.")
            latest = item.revisions[-1]
            latest.verdict = decision.verdict
            latest.feedback = decision.feedback
            if decision.verdict == "approve":
                item.state = "approved"
                self._refresh()
                self._persist()
                self._emit("backlog_item_approved", item, revision=len(item.revisions))
                return item.model_copy(deep=True)

            reason = self._escalation_reason(item)
            if reason:
                item.state = "needs_decision"
                item.decision_reason = reason
                self._persist()
                self._emit("backlog_item_needs_decision", item, reason=reason)
            else:
                item.state = "changes_requested"
                self._persist()
                self._emit("backlog_item_changes_requested", item, feedback=decision.feedback, revision=len(item.revisions))
            return item.model_copy(deep=True)

    def _escalation_reason(self, item: WorkItem) -> str:
        counted = self._counted_revisions(item)
        if counted >= item.max_revisions:
            return f"revision cap reached ({counted}/{item.max_revisions})"
        window = item.revisions[item.cap_base_index:]
        if len(window) >= 2:
            prev, cur = window[-2], window[-1]
            if _normalize_text(prev.feedback) and _normalize_text(prev.feedback) == _normalize_text(cur.feedback):
                return "stalled: identical feedback on consecutive revisions"
            if cur.proof_hashes and cur.proof_hashes == prev.proof_hashes:
                return "stalled: proof is unchanged between revisions"
        return ""

    def apply_cap_decision(self, decision: CapDecision) -> WorkItem:
        with self._lock:
            item = self._require(decision.item_id)
            if item.state != "needs_decision":
                raise BoardValidationError(f"Item '{decision.item_id}' is {item.state}, not needs_decision.")
            if decision.action == "accept_as_is":
                item.state = "approved"
                item.accepted_compromise = f"Accepted as-is after {item.decision_reason}: {decision.deliberation}"
                self._refresh()
                self._persist()
                self._emit("backlog_item_approved", item, accepted_compromise=True)
            elif decision.action == "rewrite_spec":
                if item.spec_resets >= MAX_SPEC_RESETS:
                    raise BoardValidationError(
                        f"Item '{item.item_id}' has already had {MAX_SPEC_RESETS} spec rewrites; choose accept_as_is, split or cancel."
                    )
                if not decision.replacement_spec.strip():
                    raise BoardValidationError("rewrite_spec requires a non-empty replacement_spec.")
                item.spec = decision.replacement_spec
                item.spec_resets += 1
                item.cap_base_index = len(item.revisions)
                item.decision_reason = ""
                item.state = "ready" if self._deps_satisfied(item) else "blocked"
                self._persist()
                self._emit("backlog_item_spec_rewritten", item, spec_resets=item.spec_resets)
            elif decision.action == "split":
                if not decision.split_items:
                    raise BoardValidationError("split requires at least one split item.")
                issues = self._validate_batch(decision.split_items, existing=set(self._items))
                if issues:
                    raise BoardValidationError("; ".join(issues))
                new_ids = [p.item_id for p in decision.split_items]
                for p in decision.split_items:
                    deps = list(dict.fromkeys(list(p.depends_on) + [d for d in item.depends_on if d not in p.depends_on]))
                    self._items[p.item_id] = WorkItem(
                        deliberation=p.deliberation, item_id=p.item_id, kind=p.kind, title=p.title, spec=p.spec,
                        proof_spec=p.proof_spec, depends_on=deps, target_modules=list(p.target_modules),
                        raised_by_role=item.raised_by_role, design_revision=self._design_revision, state="blocked",
                        max_revisions=p.max_revisions or self.default_max_revisions,
                    )
                for other in self._items.values():  # dependents now wait for every split piece
                    if item.item_id in other.depends_on and other.item_id not in new_ids:
                        other.depends_on = [d for d in other.depends_on if d != item.item_id] + new_ids
                item.state = "cancelled"
                item.decision_reason = f"split into {new_ids}"
                self._refresh()
                self._persist()
                for nid in new_ids:
                    self._emit("backlog_item_created", self._items[nid], split_from=item.item_id)
            else:  # cancel
                item.state = "cancelled"
                item.decision_reason = decision.deliberation
                self._refresh()
                self._persist()
                self._emit("backlog_item_cancelled", item, reason=decision.deliberation)
            return self._items[item.item_id].model_copy(deep=True)


Handler = Callable[[WorkItem], WorkerResult]


class BoardScheduler:
    """Runs workers concurrently for every claimable board item (bounded pool, exclusive module access)."""

    def __init__(
        self,
        board: ProductionBoard,
        handlers: Dict[str, Handler],
        max_workers: int = 2,
        poll_interval: float = 0.05,
        code_lock: Optional[Any] = None,
    ) -> None:
        self.board = board
        self.handlers = handlers
        self.max_workers = max(1, int(max_workers))
        self.poll_interval = poll_interval
        self.code_lock = code_lock
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="board-worker")
        self._running = 0
        self._cv = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self.peak_running = 0

    def _get_code_lock(self) -> Optional[Any]:
        if self.code_lock is not None:
            return self.code_lock
        for handler in self.handlers.values():
            lock = getattr(handler, "_code_lock", None)
            if lock is not None:
                self.code_lock = lock
                return lock
        return None

    def is_code_lock_busy(self) -> bool:
        lock = self._get_code_lock()
        if lock is None:
            return False
        locked_fn = getattr(lock, "locked", None)
        if callable(locked_fn):
            try:
                return bool(locked_fn())
            except Exception:
                pass
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
            return False
        return True

    def pause(self) -> None:
        """Pauses new worker claims while allowing in-flight worker tasks to finish."""
        self._paused.set()
        with self._cv:
            self._cv.notify_all()

    def resume(self) -> None:
        """Resumes worker claims on the ProductionBoard."""
        self._paused.clear()
        with self._cv:
            self._cv.notify_all()

    def is_paused(self) -> bool:
        if self._paused.is_set():
            return True
        if self.board.path is not None:
            pause_file = self.board.path.parent.parent / ".studio_paused.json"
            if pause_file.is_file():
                try:
                    data = json.loads(pause_file.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and data.get("paused") is True:
                        return True
                except Exception:
                    pass
        return False

    # -- worker execution
    def _run_item(self, item: WorkItem) -> None:
        _SCHEDULER_TLS.active = True
        try:
            handler = self.handlers.get(item.kind)
            if handler is None:
                raise StudioAgentError(f"No worker is registered for item kind '{item.kind}' (item '{item.item_id}').")
            result = handler(item)
            if not isinstance(result, WorkerResult):
                raise StudioAgentError(f"Worker for '{item.kind}' returned {type(result).__name__}, expected WorkerResult.")
            self.board.submit_result(item.item_id, result)
        except WorkerYielded as wy:
            logger.info("Worker yielded item %s: %s", item.item_id, wy.reason)
            self.board.yield_item(item.item_id, reason=wy.reason)
        except Exception:
            err = traceback.format_exc()
            logger.error("Worker failed for %s:\n%s", item.item_id, err)
            self.board.fail_item(item.item_id, err)
        finally:
            _SCHEDULER_TLS.active = False
            with self._cv:
                self._running -= 1
                self._cv.notify_all()

    def dispatch_once(self) -> int:
        """Claims as many runnable items as there are free worker slots and starts them; returns the number started."""
        if self.is_paused():
            return 0
        with self._cv:
            free = self.max_workers - self._running
        if free <= 0:
            return 0
        claimed = self.board.claim_ready(
            limit=free,
            code_lock_busy=self.is_code_lock_busy(),
            prefer_parallel_assets=True,
        )
        for item in claimed:
            with self._cv:
                self._running += 1
                self.peak_running = max(self.peak_running, self._running)
            self._pool.submit(self._run_item, item)
        return len(claimed)

    def running_count(self) -> int:
        with self._cv:
            return self._running

    def run_until_idle(self, timeout: float = 60.0) -> bool:
        """Dispatches until nothing is running and nothing more is claimable (items may remain awaiting review/blocked).
        Returns False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            started = self.dispatch_once()
            with self._cv:
                if self._running > 0:
                    self._cv.wait(timeout=self.poll_interval)
                    continue
            if started == 0:
                return True
        return False

    # -- background mode (used while the council keeps designing)
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    self.dispatch_once()
                except Exception:
                    logger.exception("Scheduler dispatch failed")
                with self._cv:
                    self._cv.wait(timeout=self.poll_interval)

        self._thread = threading.Thread(target=_loop, name="board-scheduler", daemon=True)
        self._thread.start()

    def stop(self, wait: bool = True) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._pool.shutdown(wait=wait)
