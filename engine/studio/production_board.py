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
    StudioPausedError,
    WorkItem,
    WorkItemProposal,
    WorkItemRefinement,
)

logger = logging.getLogger("ProductionBoard")

DEFAULT_MAX_REVISIONS = 5
MAX_SPEC_RESETS = 2
BACKLOG_FILENAME = "backlog.json"
RUNNABLE_STATES = ("ready", "changes_requested", "waiting_for_lock")
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
        self._yielded_fifo: List[str] = []
        self._yielded_file_locks: Dict[str, Set[str]] = {}
        self._active_resumed_code_lock_item: Optional[str] = None
        if self.path is not None and self.path.is_file():
            self._load()

    def _discard_yielded(self, item_id: str) -> None:
        self._yielded_on_code_lock.discard(item_id)
        self._yielded_file_locks.pop(item_id, None)
        while item_id in self._yielded_fifo:
            self._yielded_fifo.remove(item_id)
        if item_id in self._items:
            self._items[item_id].waiting_on_locks = []

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._design_revision = int(data.get("design_revision", 0))
        dirty = False
        for raw in data.get("items", []):
            item = WorkItem.model_validate(raw)
            if item.state in ("in_progress", "failed"):  # worker died with the process or build was resumed: re-queue
                item.state = "changes_requested" if item.revisions and item.revisions[-1].feedback else "ready"
                item.failure_count = 0
                item.waiting_on_locks = []
                dirty = True
            elif item.state == "waiting_for_lock" and item.waiting_on_locks:
                file_locks = {p for p in item.waiting_on_locks if p and p != "code_lock"}
                if file_locks:
                    self._yielded_file_locks[item.item_id] = file_locks
            self._items[item.item_id] = item
        self._refresh()
        if dirty:
            self._persist()

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
        """Promotes blocked items whose dependencies are all approved, and demotes runnable items whose dependencies became unsatisfied."""
        for item in self._items.values():
            if item.state == "blocked" and self._deps_satisfied(item):
                item.state = "changes_requested" if item.revisions and item.revisions[-1].feedback else "ready"
            elif item.state in ("ready", "changes_requested", "waiting_for_lock") and not self._deps_satisfied(item):
                self._discard_yielded(item.item_id)
                item.state = "blocked"

    def is_drained(self) -> bool:
        """True only when every item is approved or cancelled (so no needs_decision, blocked, failed or pending work)."""
        with self._lock:
            return all(i.state in TERMINAL_STATES for i in self._items.values())

    def items_for_slice(self, slice_id: str) -> List[WorkItem]:
        """Returns all board items belonging to `slice_id` (or all items if `slice_id` is empty)."""
        with self._lock:
            if not slice_id:
                return [i.model_copy(deep=True) for i in self._items.values()]
            return [i.model_copy(deep=True) for i in self._items.values() if i.slice_id == slice_id]

    def is_slice_drained(self, slice_id: str) -> bool:
        """True when every item belonging to `slice_id` is approved or cancelled."""
        with self._lock:
            target = [i for i in self._items.values() if not slice_id or i.slice_id == slice_id]
            return all(i.state in TERMINAL_STATES for i in target)

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
            approved_items = [i for i in self._items.values() if i.state == "approved"]
            cancelled_items = [i for i in self._items.values() if i.state == "cancelled"]
            compact_terminals = len(self._items) > 8 and (len(approved_items) + len(cancelled_items)) >= 4
            if compact_terminals:
                if approved_items:
                    lines.append(
                        f"- APPROVED ITEMS ({len(approved_items)}): "
                        + ", ".join(
                            f"[{i.item_id}] ({i.kind}, mods={i.target_modules})"
                            if i.target_modules
                            else f"[{i.item_id}] ({i.kind})"
                            for i in approved_items
                        )
                    )
                if cancelled_items:
                    lines.append(
                        f"- CANCELLED ITEMS ({len(cancelled_items)}): "
                        + ", ".join(f"[{i.item_id}] ({i.kind})" for i in cancelled_items)
                    )
            for i in self._items.values():
                if compact_terminals and i.state in ("approved", "cancelled"):
                    continue
                extra = ""
                if i.target_modules:
                    extra += f" mods={i.target_modules}"
                if getattr(i, "excluded_modules", None):
                    extra += f" excluded_mods={i.excluded_modules}"
                if getattr(i, "waiting_on_locks", None):
                    extra += f" waiting_locks={i.waiting_on_locks}"
                if i.slice_id:
                    extra += f" slice={i.slice_id}"
                if i.spawned_by_item_id:
                    extra += f" spawned_by={i.spawned_by_item_id}"
                if i.state in ("needs_decision", "cancelled") and i.decision_reason:
                    extra += f" REASON={i.decision_reason}"
                if i.accepted_compromise:
                    extra += " [accepted compromise]"
                if getattr(i, "shaping_notes", None):
                    extra += f" shaped={len(i.shaping_notes)}"
                spec_preview = ""
                if i.state in ("ready", "blocked") and i.spec:
                    clean_s = " ".join(i.spec.split())
                    spec_preview = f" | spec: {clean_s[:220]}{'...' if len(clean_s) > 220 else ''}"
                elif i.state in ("in_progress", "waiting_for_lock", "changes_requested", "awaiting_review") and i.revisions and i.revisions[-1].feedback:
                    clean_fb = " ".join(i.revisions[-1].feedback.split())
                    spec_preview = f" | last_feedback: {clean_fb[:180]}{'...' if len(clean_fb) > 180 else ''}"
                lines.append(
                    f"- [{i.item_id}] {i.kind} '{i.title}' state={i.state} deps={i.depends_on} "
                    f"rev={len(i.revisions)}/{i.max_revisions} by={i.raised_by_role}{extra}{spec_preview}"
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
            bad_test_mods = [
                m for m in (getattr(p, "target_modules", None) or [])
                if str(m or "").strip().replace("\\", "/").removeprefix("res://").lstrip("/").startswith("tests/")
            ]
            if bad_test_mods:
                issues.append(
                    f"Item '{p.item_id}' targets QA test file(s) {bad_test_mods} in `target_modules`. "
                    "Production Board tasks must target production game scripts (never `tests/test_*`); QA test harnesses are authored exclusively by `qa_test_engineer` during proof verification."
                )
        known = existing | set(batch_ids)
        for p in proposals:
            for d in p.depends_on:
                if d not in known:
                    issues.append(f"Item '{p.item_id}' depends on unknown item '{d}'.")
            if p.item_id in p.depends_on:
                issues.append(f"Item '{p.item_id}' depends on itself.")
            for b in getattr(p, "blocks_items", None) or []:
                if b not in known:
                    issues.append(f"Item '{p.item_id}' blocks_items references unknown item '{b}'.")
                if b == p.item_id:
                    issues.append(f"Item '{p.item_id}' cannot block itself in blocks_items.")
        # cycle detection over (existing + batch + blocks_items edges)
        graph: Dict[str, List[str]] = {i.item_id: list(i.depends_on) for i in self._items.values()}
        for p in proposals:
            graph[p.item_id] = list(p.depends_on)
        for p in proposals:
            for b in getattr(p, "blocks_items", None) or []:
                if b in graph and p.item_id not in graph[b]:
                    graph[b].append(p.item_id)
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
    def add_items(
        self,
        proposals: List[WorkItemProposal],
        raised_by_role: str = "",
        slice_id: str = "",
    ) -> List[WorkItem]:
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
                    slice_id=p.slice_id or slice_id,
                    design_revision=self._design_revision,
                    state="blocked",
                    max_revisions=p.max_revisions or self.default_max_revisions,
                )
                self._items[item.item_id] = item
                added.append(item)
            for p in proposals:
                for b_id in getattr(p, "blocks_items", None) or []:
                    tgt = self._items.get(b_id)
                    if tgt is not None and tgt.state not in TERMINAL_STATES and p.item_id not in tgt.depends_on:
                        tgt.depends_on.append(p.item_id)
            self._refresh()
            self._persist()
            for item in added:
                self._emit(
                    "backlog_item_created",
                    self._items[item.item_id],
                    depends_on=item.depends_on,
                    slice_id=item.slice_id,
                )
            return [self._items[i.item_id].model_copy(deep=True) for i in added]

    def spawn_tasks_from_item(
        self,
        parent_item_id: str,
        proposals: List[WorkItemProposal],
        wait_for_spawned: bool = True,
    ) -> List[WorkItem]:
        """Atomically spawns new tasks discovered by an executing task onto the ProductionBoard.

        When `wait_for_spawned` is True, the newly spawned task IDs are added to `parent.depends_on`
        and `parent` transitions back to `blocked` until those spawned tasks are built and approved,
        enabling emergent just-in-time asset & sub-mechanic discovery (`Tasks -> More Tasks`).
        """
        with self._lock:
            parent = self._require(parent_item_id)
            if not proposals:
                raise BoardValidationError("spawn_tasks_from_item requires at least one WorkItemProposal.")
            sanitized_proposals: List[WorkItemProposal] = []
            existing_wait_ids: List[str] = []
            for raw_p in proposals:
                p = raw_p.model_copy(deep=True)
                if wait_for_spawned and parent_item_id in p.depends_on:
                    p.depends_on = [d for d in p.depends_on if d != parent_item_id]
                if wait_for_spawned and p.item_id in self._items and p.item_id != parent_item_id:
                    if p.item_id not in existing_wait_ids:
                        existing_wait_ids.append(p.item_id)
                    continue
                if p.kind == "level_map" and "level_layout.json" not in p.target_modules:
                    p.target_modules.append("level_layout.json")
                elif p.kind in ASSET_KINDS:
                    p.target_modules = [m for m in p.target_modules if not str(m).endswith(".gd")]
                sanitized_proposals.append(p)

            issues = self._validate_batch(sanitized_proposals, existing=set(self._items)) if sanitized_proposals else []
            if wait_for_spawned and not issues:
                batch_map = {p.item_id: list(p.depends_on) for p in sanitized_proposals}
                check_roots = [(p.item_id, list(p.depends_on)) for p in sanitized_proposals] + [
                    (eid, list(self._items[eid].depends_on)) for eid in existing_wait_ids
                ]
                for root_id, root_deps in check_roots:
                    stack = list(root_deps)
                    seen: Set[str] = set()
                    while stack:
                        n = stack.pop()
                        if n == parent_item_id:
                            issues.append(
                                f"Spawned/linked prerequisite task '{root_id}' cannot depend (directly or transitively) on parent '{parent_item_id}' when wait_for_spawned=True."
                            )
                            break
                        if n in seen:
                            continue
                        seen.add(n)
                        if n in batch_map:
                            stack.extend(batch_map[n])
                        elif n in self._items:
                            stack.extend(self._items[n].depends_on)
            if issues:
                raise BoardValidationError("; ".join(issues))

            added: List[WorkItem] = []
            for p in sanitized_proposals:
                item = WorkItem(
                    deliberation=p.deliberation,
                    item_id=p.item_id,
                    kind=p.kind,
                    title=p.title,
                    spec=p.spec,
                    proof_spec=p.proof_spec,
                    depends_on=list(p.depends_on),
                    target_modules=list(p.target_modules),
                    raised_by_role=parent.raised_by_role or "TaskLead",
                    slice_id=p.slice_id or parent.slice_id,
                    spawned_by_item_id=parent.item_id,
                    design_revision=self._design_revision,
                    state="blocked",
                    max_revisions=p.max_revisions or self.default_max_revisions,
                    priority=(parent.priority + 1) if wait_for_spawned else parent.priority,
                    shaping_notes=list(parent.shaping_notes) + [f"[Spawned from {parent.item_id}]: {p.deliberation}"],
                )
                self._items[item.item_id] = item
                added.append(item)
            for p in sanitized_proposals:
                for b_id in getattr(p, "blocks_items", None) or []:
                    tgt = self._items.get(b_id)
                    if tgt is not None and tgt.state not in TERMINAL_STATES and p.item_id not in tgt.depends_on:
                        tgt.depends_on.append(p.item_id)

            if wait_for_spawned:
                for wait_id in [i.item_id for i in added] + existing_wait_ids:
                    if wait_id not in parent.depends_on:
                        parent.depends_on.append(wait_id)
                self._yielded_on_code_lock.discard(parent.item_id)
                if self._active_resumed_code_lock_item == parent.item_id:
                    self._active_resumed_code_lock_item = None
                parent.state = "blocked"

            self._refresh()
            self._persist()
            for item in added:
                self._emit(
                    "backlog_item_created",
                    self._items[item.item_id],
                    depends_on=item.depends_on,
                    slice_id=item.slice_id,
                    spawned_by=parent.item_id,
                )
            self._emit(
                "backlog_item_spawned_tasks",
                parent,
                spawned_item_ids=[i.item_id for i in added],
                wait_for_spawned=wait_for_spawned,
            )
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

    def validate_refinements(
        self,
        refinements: List[WorkItemRefinement],
        new_proposals: Optional[List[WorkItemProposal]] = None,
    ) -> List[str]:
        """Validates pre-execution Council refinements ('enrich' or 'split') against the board and any companion proposals."""
        with self._lock:
            issues: List[str] = []
            new_prop_map: Dict[str, WorkItemProposal] = {p.item_id: p for p in (new_proposals or [])}
            known_ids: Set[str] = set(self._items) | set(new_prop_map)
            split_proposals: List[WorkItemProposal] = []
            seen_splits: Set[str] = set()
            for r in refinements:
                if (
                    r.action == "split"
                    and not r.split_items
                    and (
                        r.spec_addition_or_rewrite.strip()
                        or r.proof_spec_update.strip()
                        or r.add_depends_on
                        or r.add_target_modules
                    )
                ):
                    r.action = "enrich"
                if r.item_id in self._items:
                    target = self._items[r.item_id]
                    if target.state not in ("blocked", "ready"):
                        split_children = [
                            c for c in self._items.values()
                            if c.spawned_by_item_id == r.item_id and c.state in ("blocked", "ready")
                        ]
                        if target.state == "cancelled" and split_children and r.action in ("enrich", "no_input"):
                            base_deps = list(target.depends_on)
                        elif r.action == "no_input":
                            continue
                        else:
                            issues.append(
                                f"Item '{r.item_id}' is in state '{target.state}' ({target.decision_reason or 'already started/finished'}) and can no longer be pre-shaped or split (only 'blocked' or 'ready' unstarted items can be refined)."
                            )
                            continue
                    else:
                        base_deps = list(target.depends_on)
                elif r.item_id in new_prop_map:
                    base_deps = list(new_prop_map[r.item_id].depends_on)
                else:
                    issues.append(f"Cannot refine unknown item '{r.item_id}'.")
                    continue
                if r.action == "no_input":
                    continue
                if r.action == "split":
                    if r.item_id in seen_splits:
                        issues.append(f"Duplicate split refinement for item '{r.item_id}' in the same turn.")
                        continue
                    seen_splits.add(r.item_id)
                    if not r.split_items:
                        issues.append(
                            f"Refinement action='split' on '{r.item_id}' has empty split_items=[]. Because split_items appears BEFORE action in JSON schema order, populate split_items with WorkItemProposal objects BEFORE setting action='split', OR set action='enrich'."
                        )
                    else:
                        split_proposals.extend(r.split_items)
                        for sp in r.split_items:
                            known_ids.add(sp.item_id)
                else:
                    if not (
                        r.spec_addition_or_rewrite.strip()
                        or r.deliberation.strip()
                        or r.proof_spec_update.strip()
                        or r.add_depends_on
                        or r.add_target_modules
                    ):
                        issues.append(
                            f"Refinement action='enrich' on '{r.item_id}' must provide spec_addition_or_rewrite, proof_spec_update, add_depends_on, or add_target_modules."
                        )
                    unknown_deps = [d for d in r.add_depends_on if d not in known_ids]
                    if unknown_deps:
                        issues.append(f"Refinement on '{r.item_id}' references unknown add_depends_on {unknown_deps}.")
                    if r.item_id in r.add_depends_on:
                        issues.append(f"Item '{r.item_id}' cannot depend on itself.")
                    merged_deps = list(dict.fromkeys(base_deps + list(r.add_depends_on)))
                    cyc = self._would_cycle(r.item_id, merged_deps)
                    if cyc:
                        issues.append(cyc)
            if split_proposals:
                existing_for_split = set(self._items) | set(new_prop_map)
                issues.extend(self._validate_batch(split_proposals, existing=existing_for_split))
            return issues

    def apply_refinements(
        self,
        refinements: List[WorkItemRefinement],
        by_role: str = "",
    ) -> List[WorkItem]:
        """Applies Council specialist refinements ('enrich', 'split', or 'no_input') to unstarted items on the Production Board."""
        active_refs = [r for r in (refinements or []) if r.action != "no_input"]
        if not active_refs:
            return []
        with self._lock:
            issues = self.validate_refinements(active_refs)
            if issues:
                raise BoardValidationError("; ".join(issues))
            touched: List[WorkItem] = []
            role_tag = by_role or "Council"
            role_prefix = f"[{role_tag} Direction]:"
            ordered_refinements = [r for r in active_refs if r.action != "split"] + [
                r for r in active_refs if r.action == "split"
            ]
            for r in ordered_refinements:
                item = self._require(r.item_id)
                if r.action == "enrich":
                    targets = [item]
                    if item.state == "cancelled":
                        targets = [
                            c for c in self._items.values()
                            if c.spawned_by_item_id == item.item_id and c.state in ("blocked", "ready")
                        ] or [item]
                    direction_text = r.spec_addition_or_rewrite.strip() or (
                        r.deliberation.strip()
                        if not (r.proof_spec_update.strip() or r.add_depends_on or r.add_target_modules)
                        else ""
                    )
                    for tgt_item in targets:
                        if direction_text:
                            note = f"{role_prefix} {direction_text}"
                            existing_idx = next(
                                (idx for idx, n in enumerate(tgt_item.shaping_notes) if n.startswith(role_prefix)),
                                None,
                            )
                            if existing_idx is not None:
                                old_note = tgt_item.shaping_notes[existing_idx]
                                if old_note != note:
                                    tgt_item.shaping_notes[existing_idx] = note
                                    if old_note in tgt_item.spec:
                                        tgt_item.spec = tgt_item.spec.replace(old_note, note)
                                    elif direction_text not in tgt_item.spec:
                                        tgt_item.spec = f"{tgt_item.spec.rstrip()}\n\n{note}"
                            else:
                                tgt_item.shaping_notes.append(note)
                                if direction_text not in tgt_item.spec:
                                    tgt_item.spec = f"{tgt_item.spec.rstrip()}\n\n{note}"
                        if r.proof_spec_update.strip() and r.proof_spec_update.strip() not in tgt_item.proof_spec:
                            tgt_item.proof_spec = (
                                f"{tgt_item.proof_spec.rstrip()} | [{role_tag}]: {r.proof_spec_update.strip()}"
                                if tgt_item.proof_spec.strip()
                                else r.proof_spec_update.strip()
                            )
                        for d in r.add_depends_on:
                            if d not in tgt_item.depends_on and d != tgt_item.item_id:
                                tgt_item.depends_on.append(d)
                        if tgt_item.kind not in ASSET_KINDS:
                            for m in r.add_target_modules:
                                if m not in tgt_item.target_modules:
                                    tgt_item.target_modules.append(m)
                        touched.append(tgt_item)
                elif r.action == "split":
                    new_ids = [p.item_id for p in r.split_items]
                    for p in r.split_items:
                        if p.kind == "level_map" and "level_layout.json" not in p.target_modules:
                            p.target_modules.append("level_layout.json")
                        elif p.kind in ASSET_KINDS:
                            p.target_modules = [m for m in p.target_modules if not str(m).endswith(".gd")]
                        deps = list(dict.fromkeys(list(p.depends_on) + [d for d in item.depends_on if d not in p.depends_on]))
                        split_item = WorkItem(
                            deliberation=p.deliberation,
                            item_id=p.item_id,
                            kind=p.kind,
                            title=p.title,
                            spec=p.spec,
                            proof_spec=p.proof_spec,
                            depends_on=deps,
                            target_modules=list(p.target_modules),
                            raised_by_role=by_role or item.raised_by_role,
                            slice_id=p.slice_id or item.slice_id,
                            spawned_by_item_id=item.item_id,
                            design_revision=self._design_revision,
                            state="blocked",
                            max_revisions=p.max_revisions or self.default_max_revisions,
                            shaping_notes=list(item.shaping_notes) + [f"[Split from {item.item_id} by {role_tag}]: {r.deliberation}"],
                        )
                        self._items[p.item_id] = split_item
                        touched.append(split_item)
                    for p in r.split_items:
                        for b_id in getattr(p, "blocks_items", None) or []:
                            tgt = self._items.get(b_id)
                            if tgt is not None and tgt.state not in TERMINAL_STATES and p.item_id not in tgt.depends_on:
                                tgt.depends_on.append(p.item_id)
                    for other in self._items.values():
                        if item.item_id in other.depends_on and other.item_id not in new_ids:
                            other.depends_on = [d for d in other.depends_on if d != item.item_id] + new_ids
                    item.state = "cancelled"
                    item.decision_reason = f"split by {role_tag} into {new_ids}"
                    self._yielded_on_code_lock.discard(item.item_id)
                    if self._active_resumed_code_lock_item == item.item_id:
                        self._active_resumed_code_lock_item = None

            self._refresh()
            self._persist()
            for r in active_refs:
                orig = self._items[r.item_id]
                if r.action == "enrich":
                    self._emit(
                        "backlog_item_shaped",
                        orig,
                        by_role=role_tag,
                        depends_on=orig.depends_on,
                    )
                else:
                    new_ids = [p.item_id for p in r.split_items]
                    self._emit(
                        "backlog_item_split",
                        orig,
                        by_role=role_tag,
                        split_into=new_ids,
                    )
                    for nid in new_ids:
                        self._emit(
                            "backlog_item_created",
                            self._items[nid],
                            split_from=orig.item_id,
                            by_role=role_tag,
                        )
            return [self._items[i.item_id].model_copy(deep=True) for i in touched]

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

    def escalate_failed_to_decision(self, item_id: str) -> WorkItem:
        """Escalates a repeatedly failed item (`failure_count >= 2`) to `needs_decision` so the Council can
        rewrite its spec, accept as-is, or cancel & unblock dependents instead of aborting the whole build."""
        with self._lock:
            item = self._require(item_id)
            if item.state != "failed":
                raise BoardValidationError(f"Item '{item_id}' is {item.state}, not failed.")
            short_err = (item.failure.strip().splitlines()[-1] if item.failure.strip() else "worker error")[:240]
            reason = f"worker failed {item.failure_count} times ({short_err})"
            item.state = "needs_decision"
            item.decision_reason = reason
            self._persist()
            self._emit("backlog_item_needs_decision", item, reason=reason)
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

            use_live_file_locks = locked_modules is not None
            busy: Set[str] = set(locked_modules or set())
            in_progress_code_count = 0
            for i in self._items.values():
                if i.state == "in_progress":
                    if use_live_file_locks and i.kind in CODE_LOCK_KINDS:
                        busy.update(i.target_modules[:1])
                    else:
                        busy.update(i.target_modules)
                    if i.kind in CODE_LOCK_KINDS:
                        in_progress_code_count += 1

            def _cand_mods(cand: WorkItem) -> Set[str]:
                if use_live_file_locks and cand.kind in CODE_LOCK_KINDS:
                    yielded_req = self._yielded_file_locks.get(cand.item_id)
                    if yielded_req:
                        return set(yielded_req) | set(cand.target_modules[:1])
                    return set(cand.target_modules[:1])
                return set(cand.target_modules)

            effective_code_busy = bool(code_lock_busy or self._active_resumed_code_lock_item is not None)
            claimed: List[WorkItem] = []
            order_index = {item_id: idx for idx, item_id in enumerate(self._items)}
            runnable = [i for i in self._items.values() if i.state in RUNNABLE_STATES and self._deps_satisfied(i)]
            runnable.sort(
                key=lambda i: (
                    -i.priority,
                    0 if i.state in ("waiting_for_lock", "changes_requested") else 1,
                    0 if i.state == "waiting_for_lock" else 1,
                    max(0, len(i.revisions) - i.cap_base_index),
                    order_index[i.item_id],
                )
            )

            if not prefer_parallel_assets and not self._yielded_on_code_lock and not effective_code_busy:
                for item in runnable:
                    if len(claimed) >= limit:
                        break
                    mods = _cand_mods(item)
                    if mods & busy:
                        continue
                    self._discard_yielded(item.item_id)
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
                    if _cand_mods(cand) & busy:
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
                    eligible_by_id = {c.item_id: c for c in eligible}
                    for y_id in self._yielded_fifo:
                        if y_id in eligible_by_id:
                            chosen = eligible_by_id[y_id]
                            break
                    if chosen is None:
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
                chosen_mods = _cand_mods(chosen)
                was_yielded = chosen.item_id in self._yielded_on_code_lock
                self._discard_yielded(chosen.item_id)
                if was_yielded:
                    self._active_resumed_code_lock_item = chosen.item_id
                    resumed_code_claimed_in_call = True
                    effective_code_busy = True
                if chosen.kind in CODE_LOCK_KINDS:
                    code_claimed_in_call += 1

                chosen.state = "in_progress"
                busy.update(chosen_mods)
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

    def expand_item_target_modules(self, item_id: str, extra_modules: List[str]) -> List[str]:
        """Atomically appends newly discovered target module paths to `item.target_modules` (deduplicated, skipping `excluded_modules`)."""
        with self._lock:
            item = self._require(item_id)
            excluded = set(getattr(item, "excluded_modules", None) or [])
            changed = False
            for raw_mod in extra_modules or []:
                mod = str(raw_mod or "").strip().replace("\\", "/")
                if mod.startswith("res://"):
                    mod = mod[len("res://"):]
                elif mod.startswith("res:/"):
                    mod = mod[len("res:/"):]
                mod = mod.lstrip("/")
                if mod and mod not in excluded and mod not in item.target_modules:
                    item.target_modules.append(mod)
                    changed = True
            if changed:
                self._persist()
            return list(item.target_modules)

    def _prune_item_modules_locked(
        self,
        item: WorkItem,
        modules_to_prune: List[str],
        owner_item_ids: List[str],
    ) -> List[str]:
        """Removes out-of-scope modules from `item.target_modules` (preserving at least one primary module if all would be removed),
        records them in `item.excluded_modules`, and appends a `[Council Scope Boundary]` note to `item.shaping_notes`."""
        pruned: List[str] = []
        primary_mod = item.target_modules[0] if len(item.target_modules) == 1 else None
        for raw_m in modules_to_prune or []:
            mod = str(raw_m or "").strip().replace("\\", "/")
            if mod.startswith("res://"):
                mod = mod[len("res://"):]
            elif mod.startswith("res:/"):
                mod = mod[len("res:/"):]
            mod = mod.lstrip("/")
            if not mod:
                continue
            if primary_mod and mod == primary_mod:
                continue
            if mod in item.target_modules and len(item.target_modules) > 1:
                item.target_modules = [m for m in item.target_modules if m != mod]
            if mod not in item.excluded_modules:
                item.excluded_modules.append(mod)
            if mod not in pruned:
                pruned.append(mod)
        if pruned:
            note = (
                f"[Council Scope Boundary]: Modules {pruned} are owned by prerequisite task(s) {owner_item_ids}. "
                f"Do NOT modify {pruned} inside '{item.item_id}'—restrict your edits strictly to {item.target_modules}."
            )
            if note not in item.shaping_notes:
                item.shaping_notes.append(note)
        return pruned

    def yield_item(self, item_id: str, reason: str = "") -> WorkItem:
        """Pauses an in-progress item back to its runnable, waiting_for_lock, or blocked state so the worker slot can execute another item."""
        with self._lock:
            item = self._require(item_id)
            if self._active_resumed_code_lock_item == item_id:
                self._active_resumed_code_lock_item = None
            deps_ok = self._deps_satisfied(item)
            is_lock_yield = (
                reason.startswith("file_lock_busy")
                or "code_lock" in reason
                or reason == ""
            ) and reason not in ("spawned_prerequisite_tasks", "studio_paused")
            if item.state == "in_progress":
                if not deps_ok:
                    item.state = "blocked"
                elif is_lock_yield:
                    item.state = "waiting_for_lock"
                else:
                    item.state = "changes_requested" if item.revisions and item.revisions[-1].feedback else "ready"
            if reason in ("spawned_prerequisite_tasks", "studio_paused") or reason.startswith("file_lock_busy") or not deps_ok:
                self._discard_yielded(item_id)
                if reason.startswith("file_lock_busy:") and deps_ok:
                    conflicts = {
                        p.strip()
                        for p in reason.split(":", 1)[1].split(",")
                        if p.strip()
                    }
                    if conflicts:
                        self._yielded_file_locks[item_id] = conflicts
                        item.waiting_on_locks = sorted(conflicts)
            else:
                self._yielded_on_code_lock.add(item_id)
                if item_id not in self._yielded_fifo:
                    self._yielded_fifo.append(item_id)
                if deps_ok:
                    item.waiting_on_locks = ["code_lock"]
            self._refresh()
            self._persist()
            self._emit(
                "backlog_item_yielded",
                item,
                reason=reason or "code_lock_busy",
                waiting_on_locks=list(item.waiting_on_locks),
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
            self._discard_yielded(item_id)
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
            self._discard_yielded(item_id)
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

            # Handle cross-task root-cause routing (spawned prerequisite bugfix tasks, linked blockers, and scope pruning)
            spawned_items: List[WorkItem] = []
            linked_blocker_ids: List[str] = []
            for raw_p in getattr(decision, "spawned_prerequisite_tasks", None) or []:
                p = raw_p.model_copy(deep=True)
                p.depends_on = [d for d in p.depends_on if d in self._items and d != item.item_id]
                if p.kind == "level_map" and "level_layout.json" not in p.target_modules:
                    p.target_modules.append("level_layout.json")
                elif p.kind in ASSET_KINDS:
                    p.target_modules = [m for m in p.target_modules if not str(m).endswith(".gd")]

                # Deduplicate if an open council-spawned fix task already exists with the same ID or exact target_modules
                existing_match: Optional[WorkItem] = None
                if p.item_id in self._items:
                    existing_match = self._items[p.item_id]
                elif p.target_modules:
                    p_mods = set(p.target_modules)
                    for cand in self._items.values():
                        if (
                            cand.item_id != item.item_id
                            and cand.state not in TERMINAL_STATES
                            and set(cand.target_modules) == p_mods
                            and (cand.priority >= 2 or cand.spawned_by_item_id)
                        ):
                            existing_match = cand
                            break
                if existing_match is not None:
                    if existing_match.state not in TERMINAL_STATES and existing_match.item_id != item.item_id:
                        if existing_match.item_id not in linked_blocker_ids:
                            linked_blocker_ids.append(existing_match.item_id)
                        for b_id in getattr(p, "blocks_items", None) or []:
                            tgt = self._items.get(b_id)
                            if (
                                tgt is not None
                                and tgt.state not in TERMINAL_STATES
                                and tgt.item_id != existing_match.item_id
                                and existing_match.item_id not in tgt.depends_on
                                and not self._would_cycle(tgt.item_id, list(tgt.depends_on) + [existing_match.item_id])
                            ):
                                tgt.depends_on.append(existing_match.item_id)
                    continue

                if not self._validate_batch([p], existing=set(self._items)):
                    new_item = WorkItem(
                        deliberation=p.deliberation,
                        item_id=p.item_id,
                        kind=p.kind,
                        title=p.title,
                        spec=p.spec,
                        proof_spec=p.proof_spec,
                        depends_on=list(p.depends_on),
                        target_modules=list(p.target_modules),
                        raised_by_role=item.raised_by_role or "CouncilReviewPanel",
                        slice_id=p.slice_id or item.slice_id,
                        spawned_by_item_id=item.item_id,
                        design_revision=self._design_revision,
                        state="blocked",
                        max_revisions=p.max_revisions or self.default_max_revisions,
                        priority=max(2, item.priority + 2),
                        shaping_notes=[f"[Spawned by Council during review of {item.item_id}]: {p.deliberation}"],
                    )
                    self._items[new_item.item_id] = new_item
                    spawned_items.append(new_item)
                    linked_blocker_ids.append(new_item.item_id)
                    for b_id in getattr(p, "blocks_items", None) or []:
                        tgt = self._items.get(b_id)
                        if (
                            tgt is not None
                            and tgt.state not in TERMINAL_STATES
                            and tgt.item_id != new_item.item_id
                            and new_item.item_id not in tgt.depends_on
                            and not self._would_cycle(tgt.item_id, list(tgt.depends_on) + [new_item.item_id])
                        ):
                            tgt.depends_on.append(new_item.item_id)

            for b_id in getattr(decision, "blocked_by_item_ids", None) or []:
                if (
                    b_id in self._items
                    and b_id != item.item_id
                    and self._items[b_id].state not in TERMINAL_STATES
                    and b_id not in linked_blocker_ids
                ):
                    linked_blocker_ids.append(b_id)

            attached_blockers: List[str] = []
            for b_id in linked_blocker_ids:
                if b_id not in item.depends_on:
                    if not self._would_cycle(item.item_id, list(item.depends_on) + [b_id]):
                        item.depends_on.append(b_id)
                        attached_blockers.append(b_id)
                elif self._items[b_id].state != "approved":
                    attached_blockers.append(b_id)

            prune_mods: List[str] = list(getattr(decision, "out_of_scope_modules_to_prune", None) or [])
            for b_id in attached_blockers:
                b_item = self._items.get(b_id)
                if b_item is not None:
                    for bm in b_item.target_modules:
                        if bm not in prune_mods:
                            prune_mods.append(bm)
            if prune_mods:
                self._prune_item_modules_locked(
                    item,
                    prune_mods,
                    attached_blockers or linked_blocker_ids or ["shared_subsystem"],
                )
                # Also prune and link any sibling tasks listed in blocks_items of spawned prerequisite tasks
                for sp in spawned_items:
                    for other in self._items.values():
                        if (
                            other.item_id not in (item.item_id, sp.item_id)
                            and other.state not in TERMINAL_STATES
                            and (sp.item_id in other.depends_on or set(sp.target_modules) & set(other.target_modules))
                        ):
                            if (
                                sp.item_id not in other.depends_on
                                and len(other.target_modules) > 1
                                and not self._would_cycle(other.item_id, list(other.depends_on) + [sp.item_id])
                            ):
                                other.depends_on.append(sp.item_id)
                            self._prune_item_modules_locked(other, list(sp.target_modules), [sp.item_id])

            if not self._deps_satisfied(item):
                # Do not penalize the victim task's revision cap when it is blocked by an out-of-scope/shared prerequisite
                item.cap_base_index = min(len(item.revisions), item.cap_base_index + 1)
                self._discard_yielded(item.item_id)
                item.state = "blocked"
                self._refresh()
                self._persist()
                for sp in spawned_items:
                    self._emit(
                        "backlog_item_created",
                        self._items[sp.item_id],
                        depends_on=sp.depends_on,
                        slice_id=sp.slice_id,
                        spawned_by=item.item_id,
                    )
                self._emit(
                    "backlog_item_changes_requested",
                    item,
                    feedback=decision.feedback,
                    revision=len(item.revisions),
                    blocked_by=attached_blockers,
                )
                return item.model_copy(deep=True)

            self._refresh()
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
                item.failure = ""
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
                item.failure_count = 0
                item.failure = ""
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
                        raised_by_role=item.raised_by_role, slice_id=p.slice_id or item.slice_id,
                        spawned_by_item_id=item.item_id,
                        design_revision=self._design_revision, state="blocked",
                        max_revisions=p.max_revisions or self.default_max_revisions,
                        shaping_notes=list(item.shaping_notes) + [f"[Split from {item.item_id}]: {decision.deliberation}"],
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
        for h in handlers.values():
            w_ctx = getattr(h, "_worker_ctx", None)
            if w_ctx is not None and getattr(w_ctx, "board", None) is None:
                w_ctx.board = board
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
        except StudioPausedError as pe:
            logger.warning("Worker paused on item %s: %s", item.item_id, pe)
            if "circuit breaker" in str(pe).lower():
                self.board._emit("ERROR", item, error=str(pe), reason=str(pe))
            self.board.yield_item(item.item_id, reason="studio_paused")
        except Exception:
            err = traceback.format_exc()
            logger.error("Worker failed for %s:\n%s", item.item_id, err)
            self.board.fail_item(item.item_id, err)
        finally:
            _SCHEDULER_TLS.active = False
            with self._cv:
                self._running -= 1
                self._cv.notify_all()

    def _get_locked_modules(self) -> Optional[Set[str]]:
        for handler in self.handlers.values():
            w_ctx = getattr(handler, "_worker_ctx", None)
            flt = getattr(w_ctx, "file_lock_table", None) if w_ctx is not None else None
            if flt is not None and hasattr(flt, "all_held_paths"):
                try:
                    return set(flt.all_held_paths())
                except Exception:
                    return None
        return None

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
            locked_modules=self._get_locked_modules(),
            code_lock_busy=self.is_code_lock_busy(),
            prefer_parallel_assets=self._get_code_lock() is not None,
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
