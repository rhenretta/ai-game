"""engine/studio/game_memory.py
Per-Game Multi-Bank Long-Term Memory System for Autonomous Studio Agents.

Provides durable cross-task and cross-slice memory persistence partitioned into:
  - Shared Council memory (`memory_bank="council"`)
  - Individual specialist agent memory banks (`CreativeDirector`, `GameplayDesigner`, ...,
    `task_lead`, `swe_engineer`, `qa_test_engineer`, `visual_artist`, `audio_designer`, `script_writer`)

Persistence & Observability:
  - Authoritative per-cartridge file: `<cartridge_dir>/game_memory.json` (copied across slice checkpoints and `active`)
  - Automatic SQLite WAL mirror (`studio_game_memories` table in `TelemetryDB`)
  - Vector semantic retrieval via `gemini-embedding-001` (Rule 2: zero keyword/regex search heuristics)
  - Strict Rationale-First contracts (`deliberation` first on all models and events per Rule 4)
  - Fail-fast error surfacing when semantic vector search is invoked without an embedding client (Rule 5)
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from engine.studio.models import (
    CANONICAL_MEMORY_BANKS,
    GameMemoryEntry,
    GameMemoryLedger,
    MemoryKind,
    MemoryMutationProposal,
    MemoryStatus,
    StudioAgentError,
)

logger = logging.getLogger("GameMemoryStore")

GAME_MEMORY_FILENAME = "game_memory.json"
MEMORY_EMBEDDING_MODEL = "gemini-embedding-001"
MEMORY_EMBEDDING_DIM = 256
MEMORY_DEDUP_COSINE_THRESHOLD = 0.86

MEMORY_SELECTIVITY_RUBRIC = (
    "LONG-TERM MEMORY SELECTIVITY DISCIPLINE (DEFAULT = EMPTY — DO NOT WRITE MEMORIES ROUTINELY):\n"
    "  - Leave `memory_updates` empty (`[]`) / `memory_note` empty (`''`) on ~95% of turns.\n"
    "  - NEVER record task progress logs ('implemented X', 'verified Y'), restatements of the Living Game Bible / task spec / asset catalog, or local details specific to a single file or task.\n"
    "  - ONLY record a memory when it passes the Counterfactual Durability Test: future tasks or vertical slices would repeat a costly failure mode, violate a non-obvious cross-module invariant, or lose an intentionally deferred multi-slice goal if this insight were not persisted.\n"
    "  - Check existing `[mem_...]` entries first: prefer `operation='update'`, `'fulfill'`, `'supersede'`, or `'archive'` on an existing `memory_id` over creating a new entry."
)

_VALID_KINDS = {"strategic_goal", "architectural_decision", "lesson_learned", "domain_note"}
_VALID_STATUSES = {"active", "fulfilled", "superseded", "archived"}

_CANONICAL_ROLE_LOOKUP: Dict[str, str] = {b.lower(): b for b in CANONICAL_MEMORY_BANKS}
_CANONICAL_ROLE_LOOKUP.update(
    {
        "council": "council",
        "shared": "council",
        "council_shared": "council",
        "creative_director": "CreativeDirector",
        "gameplay_designer": "GameplayDesigner",
        "level_and_progression_designer": "LevelAndProgressionDesigner",
        "gameplay_systems_engineer": "GameplaySystemsEngineer",
        "story_world_architect": "StoryWorldArchitect",
        "scriptwriter_agent": "ScriptwriterAgent",
        "character_designer": "CharacterDesigner",
        "visual_art_director": "VisualArtDirector",
        "ux_experience_designer": "UXExperienceDesigner",
        "audio_atmosphere_designer": "AudioAtmosphereDesigner",
    }
)

_LOCKS_GUARD = threading.Lock()
_CARTRIDGE_MEMORY_LOCKS: Dict[str, threading.RLock] = {}


def _lock_for_cartridge(cartridge_dir: Path) -> threading.RLock:
    key = str(cartridge_dir.resolve())
    with _LOCKS_GUARD:
        lk = _CARTRIDGE_MEMORY_LOCKS.get(key)
        if lk is None:
            lk = threading.RLock()
            _CARTRIDGE_MEMORY_LOCKS[key] = lk
        return lk


def normalize_agent_role(raw_role: Optional[str]) -> str:
    """Normalizes composite role strings (e.g. 'council_reviewer:GameplayDesigner') to canonical role IDs."""
    text = str(raw_role or "").strip()
    if not text:
        return "council"
    if ":" in text:
        prefix, suffix = text.split(":", 1)
        if prefix.strip().lower() in (
            "council_reviewer",
            "council_slice_playtest",
            "council_reflection",
            "council_shaping",
            "council",
        ) and suffix.strip():
            text = suffix.strip()
    mapped = _CANONICAL_ROLE_LOOKUP.get(text.lower())
    return mapped if mapped is not None else text


def normalize_memory_bank(raw_bank: Optional[str], caller_role: Optional[str] = None) -> str:
    """Resolves 'self'/'own'/'' to `caller_role`'s canonical bank and 'council'/'shared' to 'council'."""
    token = str(raw_bank or "").strip()
    norm_caller = normalize_agent_role(caller_role)
    if not token or token.lower() in ("self", "own", "mine", "agent"):
        return norm_caller
    if token.lower() in ("council", "shared", "council_shared", "game"):
        return "council"
    return normalize_agent_role(token)


def detect_active_slice_id(cartridge_dir: Path) -> str:
    """Reads the active slice identifier from `game_bible.json` when available."""
    bible_path = cartridge_dir / "game_bible.json"
    if not bible_path.is_file():
        return ""
    try:
        data = json.loads(bible_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return ""
        slices = data.get("slice_roadmap") or data.get("slices")
        if isinstance(slices, list) and slices:
            active_idx = data.get("active_slice_index")
            if isinstance(active_idx, int):
                for s in slices:
                    if isinstance(s, dict) and s.get("slice_index") == active_idx and s.get("slice_id"):
                        return str(s["slice_id"])
            for s in slices:
                if isinstance(s, dict) and str(s.get("status") or "").lower() in ("active", "in_progress") and s.get("slice_id"):
                    return str(s["slice_id"])
            if isinstance(slices[-1], dict) and slices[-1].get("slice_id"):
                return str(slices[-1]["slice_id"])
    except Exception:
        return ""
    return ""


def _read_game_title(cartridge_dir: Path) -> str:
    bible_path = cartridge_dir / "game_bible.json"
    if bible_path.is_file():
        try:
            data = json.loads(bible_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("game_title"):
                return str(data["game_title"])
        except Exception:
            pass
    manifest_path = cartridge_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and (data.get("title") or data.get("game_title")):
                return str(data.get("title") or data.get("game_title"))
        except Exception:
            pass
    return ""


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _entry_embedding_text(entry: GameMemoryEntry) -> str:
    roles_txt = f" | Applicable roles: {', '.join(entry.applicable_roles)}" if entry.applicable_roles else ""
    res_txt = f" | Resolution: {entry.resolution_note}" if entry.resolution_note else ""
    return (
        f"[Memory Bank: {entry.memory_bank} | Kind: {entry.kind} | Status: {entry.status} | Author: {entry.author_role}]\n"
        f"Title: {entry.title}\n"
        f"Content: {entry.content}\n"
        f"Rationale: {entry.deliberation}{roles_txt}{res_txt}"
    )


class GameMemoryStore:
    """Thread-safe manager for per-game multi-bank long-term memories (`game_memory.json` + SQLite WAL)."""

    def __init__(
        self,
        cartridge_dir: Path,
        db: Optional[Any] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.cartridge_dir = Path(cartridge_dir)
        self.memory_path = self.cartridge_dir / GAME_MEMORY_FILENAME
        self._lock = _lock_for_cartridge(self.cartridge_dir)
        self.db = db if db is not None else self._auto_resolve_workspace_db()
        self.event_callback = event_callback
        self.client = client

    def _auto_resolve_workspace_db(self) -> Optional[Any]:
        """Automatically attaches the workspace SQLite WAL when operating on a workspace `cartridges/` directory,
        while keeping isolated unit-test temporary directories decoupled unless `db` is explicitly passed."""
        try:
            resolved = self.cartridge_dir.resolve()
            parts = resolved.parts
            if "cartridges" in parts:
                idx = parts.index("cartridges")
                repo_root = Path(*parts[:idx]) if idx > 0 else resolved.parent
                db_file = repo_root / "data" / "telemetry.db"
                if db_file.parent.is_dir() or (repo_root / "engine" / "telemetry" / "db.py").is_file():
                    from engine.telemetry.db import TelemetryDB

                    return TelemetryDB(str(db_file))
        except Exception:
            logger.debug("Could not auto-resolve workspace TelemetryDB for %s", self.cartridge_dir, exc_info=True)
        return None

    def _emit_event(self, event_type: str, deliberation: str, payload: Dict[str, Any]) -> None:
        ordered_payload: Dict[str, Any] = {"deliberation": deliberation}
        for k, v in payload.items():
            if k != "deliberation":
                ordered_payload[k] = v
        if callable(self.event_callback):
            try:
                self.event_callback(event_type, ordered_payload)
            except Exception:
                logger.debug("Memory event_callback error", exc_info=True)
        try:
            from engine.studio.orchestrator import SwarmEventBus

            SwarmEventBus.publish(
                event_type=event_type,
                agent_name=str(payload.get("author_role") or payload.get("caller_role") or "GameMemoryStore"),
                deliberation=deliberation,
                payload=ordered_payload,
            )
        except Exception:
            pass

    def load_ledger(self) -> GameMemoryLedger:
        """Loads `game_memory.json` from `cartridge_dir`, falling back to `TelemetryDB` if the file is absent."""
        with self._lock:
            cid = self.cartridge_dir.name
            gtitle = _read_game_title(self.cartridge_dir)
            if self.memory_path.is_file():
                raw_text = self.memory_path.read_text(encoding="utf-8")
                ledger = GameMemoryLedger.model_validate_json(raw_text)
                if not ledger.cartridge_id:
                    ledger.cartridge_id = cid
                if gtitle and not ledger.game_title:
                    ledger.game_title = gtitle
                return ledger

            if self.db is not None and hasattr(self.db, "list_game_memories"):
                rows = self.db.list_game_memories(cid)
                if rows:
                    entries = [GameMemoryEntry.model_validate(r) for r in rows]
                    return GameMemoryLedger(
                        deliberation="Hydrated per-game long-term memory ledger from SQLite WAL.",
                        cartridge_id=cid,
                        game_title=gtitle,
                        entries=entries,
                    )

            return GameMemoryLedger(
                deliberation="Initialized empty per-game multi-bank long-term memory ledger.",
                cartridge_id=cid,
                game_title=gtitle,
                entries=[],
            )

    def save_ledger(self, ledger: GameMemoryLedger) -> GameMemoryLedger:
        """Atomically writes `game_memory.json` and mirrors all entries into SQLite WAL."""
        with self._lock:
            self.cartridge_dir.mkdir(parents=True, exist_ok=True)
            ledger.cartridge_id = ledger.cartridge_id or self.cartridge_dir.name
            gtitle = _read_game_title(self.cartridge_dir)
            if gtitle:
                ledger.game_title = gtitle
            ledger.updated_at_utc = datetime.now(timezone.utc).isoformat()

            tmp_path = self.memory_path.with_suffix(".json.tmp")
            tmp_path.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp_path, self.memory_path)

            if self.db is not None and hasattr(self.db, "sync_game_memories"):
                self.db.sync_game_memories(
                    cartridge_id=ledger.cartridge_id,
                    entries=[e.model_dump() for e in ledger.entries],
                    game_title=ledger.game_title,
                )
            return ledger

    def list_entries(
        self,
        *,
        bank_filter: str = "",
        caller_role: str = "",
        kind_filter: str = "",
        status_filter: str = "",
    ) -> List[GameMemoryEntry]:
        """Returns structured-filtered memory entries without semantic ranking."""
        ledger = self.load_ledger()
        resolved_bank = ""
        if bank_filter and bank_filter.lower() != "all":
            resolved_bank = normalize_memory_bank(bank_filter, caller_role=caller_role)
        out: List[GameMemoryEntry] = []
        for entry in ledger.entries:
            if resolved_bank and entry.memory_bank != resolved_bank:
                continue
            if kind_filter and kind_filter != "all" and entry.kind != kind_filter:
                continue
            if status_filter and status_filter != "all" and entry.status != status_filter:
                continue
            out.append(entry)
        return out

    def bank_summary_counts(self, status_filter: str = "") -> Dict[str, int]:
        """Returns entry counts keyed by canonical bank (including 0-count banks when useful)."""
        ledger = self.load_ledger()
        counts: Dict[str, int] = {b: 0 for b in CANONICAL_MEMORY_BANKS}
        for entry in ledger.entries:
            if status_filter and status_filter != "all" and entry.status != status_filter:
                continue
            counts[entry.memory_bank] = counts.get(entry.memory_bank, 0) + 1
        return counts

    def apply_mutations(
        self,
        author_role: str,
        proposals: Sequence[MemoryMutationProposal],
        *,
        slice_id: str = "",
        item_id: str = "",
        client: Optional[Any] = None,
        dedup_similarity_threshold: float = MEMORY_DEDUP_COSINE_THRESHOLD,
    ) -> List[GameMemoryEntry]:
        """Applies one or more structured `MemoryMutationProposal` items atomically.

        On `operation == 'record'`, deduplicates within the target `memory_bank` via:
          1. Exact normalized `title` or `content` match against active entries, and
          2. Vector cosine-similarity (`>= dedup_similarity_threshold`, `gemini-embedding-001`)
             against active entries in the same bank whenever `client` (or `self.client`)
             supports `models.embed_content` (Rule 2: zero keyword/regex heuristics).
        """
        if not proposals:
            return []
        norm_author = normalize_agent_role(author_role)
        eff_client = client if client is not None else self.client
        has_embed = callable(getattr(getattr(eff_client, "models", None), "embed_content", None))
        with self._lock:
            ledger = self.load_ledger()
            eff_slice = slice_id or detect_active_slice_id(self.cartridge_dir)
            now_iso = datetime.now(timezone.utc).isoformat()
            by_id: Dict[str, GameMemoryEntry] = {e.memory_id: e for e in ledger.entries}
            touched: List[GameMemoryEntry] = []
            deduplicated_ids: List[str] = []
            emb_cache: Optional[Dict[str, List[float]]] = None
            emb_cache_dirty = False

            for prop in proposals:
                op = str(prop.operation or "record").strip().lower()
                target_id = str(prop.target_memory_id or "").strip()
                bank = normalize_memory_bank(prop.target_bank, caller_role=norm_author)
                kind_val: MemoryKind = prop.kind if prop.kind in _VALID_KINDS else "lesson_learned"
                norm_applicable = [
                    normalize_agent_role(r) for r in (prop.applicable_roles or []) if str(r or "").strip()
                ]
                title_txt = str(prop.title or "").strip()
                content_txt = str(prop.content or "").strip()
                res_note = str(prop.resolution_note or "").strip()

                if op == "record":
                    if not content_txt and not title_txt:
                        continue
                    if not title_txt:
                        title_txt = content_txt[:72].rstrip(".")
                    if not content_txt:
                        content_txt = title_txt

                    cand_entry = GameMemoryEntry(
                        deliberation=prop.deliberation,
                        memory_bank=bank,
                        author_role=norm_author,
                        kind=kind_val,
                        status="active",
                        title=title_txt,
                        content=content_txt,
                        applicable_roles=norm_applicable,
                        slice_id=eff_slice,
                        item_id=item_id,
                        created_at_utc=now_iso,
                        updated_at_utc=now_iso,
                    )

                    active_same_bank = [
                        cand for cand in ledger.entries if cand.memory_bank == bank and cand.status == "active"
                    ]

                    # 1. Fast exact title/content match within the same active bank
                    existing_dup: Optional[GameMemoryEntry] = None
                    for cand in active_same_bank:
                        if (
                            cand.title.strip().lower() == title_txt.lower()
                            or cand.content.strip().lower() == content_txt.lower()
                        ):
                            existing_dup = cand
                            break

                    # 2. Vector cosine-similarity deduplication within the same active bank
                    if existing_dup is None and active_same_bank and has_embed:
                        if emb_cache is None:
                            emb_cache = self._load_cached_embeddings()
                        cand_txt = _entry_embedding_text(cand_entry)
                        cand_digest = hashlib.sha256(cand_txt.encode("utf-8")).hexdigest()[:24]
                        bank_digests: List[str] = []
                        missing_hashes: List[str] = []
                        missing_texts: List[str] = []
                        if cand_digest not in emb_cache:
                            missing_hashes.append(cand_digest)
                            missing_texts.append(cand_txt)
                        for existing_cand in active_same_bank:
                            etxt = _entry_embedding_text(existing_cand)
                            edigest = hashlib.sha256(etxt.encode("utf-8")).hexdigest()[:24]
                            bank_digests.append(edigest)
                            if edigest not in emb_cache and edigest not in missing_hashes:
                                missing_hashes.append(edigest)
                                missing_texts.append(etxt)
                        if missing_hashes:
                            new_vecs = self._embed_memory_texts(eff_client, missing_texts)
                            for h, vec in zip(missing_hashes, new_vecs):
                                emb_cache[h] = vec
                            emb_cache_dirty = True
                        cand_vec = emb_cache.get(cand_digest) or []
                        best_sim = -1.0
                        best_cand: Optional[GameMemoryEntry] = None
                        for existing_cand, edigest in zip(active_same_bank, bank_digests):
                            sim = _cosine_similarity(cand_vec, emb_cache.get(edigest) or [])
                            if sim > best_sim:
                                best_sim = sim
                                best_cand = existing_cand
                        if best_cand is not None and best_sim >= dedup_similarity_threshold:
                            existing_dup = best_cand

                    if existing_dup is not None:
                        existing_dup.deliberation = prop.deliberation
                        existing_dup.author_role = norm_author
                        existing_dup.kind = kind_val
                        existing_dup.title = title_txt
                        existing_dup.content = content_txt
                        if norm_applicable:
                            existing_dup.applicable_roles = list(
                                dict.fromkeys(existing_dup.applicable_roles + norm_applicable)
                            )
                        if eff_slice:
                            existing_dup.slice_id = eff_slice
                        if item_id:
                            existing_dup.item_id = item_id
                        existing_dup.updated_at_utc = now_iso
                        if existing_dup.memory_id not in deduplicated_ids:
                            deduplicated_ids.append(existing_dup.memory_id)
                        if existing_dup not in touched:
                            touched.append(existing_dup)
                    else:
                        ledger.entries.append(cand_entry)
                        by_id[cand_entry.memory_id] = cand_entry
                        touched.append(cand_entry)

                elif op == "update":
                    existing = by_id.get(target_id)
                    if existing is None:
                        if content_txt or title_txt:
                            # Treat missing target_id with full content as a fresh record rather than losing insight
                            fallback_entry = GameMemoryEntry(
                                deliberation=prop.deliberation,
                                memory_bank=bank,
                                author_role=norm_author,
                                kind=kind_val,
                                status="active",
                                title=title_txt or content_txt[:72],
                                content=content_txt or title_txt,
                                applicable_roles=norm_applicable,
                                slice_id=eff_slice,
                                item_id=item_id,
                                created_at_utc=now_iso,
                                updated_at_utc=now_iso,
                            )
                            ledger.entries.append(fallback_entry)
                            by_id[fallback_entry.memory_id] = fallback_entry
                            touched.append(fallback_entry)
                        continue
                    existing.deliberation = prop.deliberation
                    existing.author_role = norm_author
                    existing.status = "active"
                    if prop.target_bank and prop.target_bank != "self":
                        existing.memory_bank = bank
                    if "kind" in prop.model_fields_set and prop.kind in _VALID_KINDS:
                        existing.kind = kind_val
                    if title_txt:
                        existing.title = title_txt
                    if content_txt:
                        existing.content = content_txt
                    if norm_applicable:
                        existing.applicable_roles = norm_applicable
                    if res_note:
                        existing.resolution_note = res_note
                    if eff_slice:
                        existing.slice_id = eff_slice
                    if item_id:
                        existing.item_id = item_id
                    existing.updated_at_utc = now_iso
                    touched.append(existing)

                elif op in ("fulfill", "archive"):
                    existing = by_id.get(target_id)
                    if existing is None:
                        continue
                    existing.deliberation = prop.deliberation
                    existing.author_role = norm_author
                    existing.status = "fulfilled" if op == "fulfill" else "archived"
                    if res_note or content_txt:
                        existing.resolution_note = res_note or content_txt
                    if eff_slice:
                        existing.slice_id = eff_slice
                    existing.updated_at_utc = now_iso
                    touched.append(existing)

                elif op == "supersede":
                    existing = by_id.get(target_id)
                    replacement: Optional[GameMemoryEntry] = None
                    if content_txt or title_txt:
                        replacement = GameMemoryEntry(
                            deliberation=prop.deliberation,
                            memory_bank=bank if prop.target_bank != "self" or existing is None else existing.memory_bank,
                            author_role=norm_author,
                            kind=kind_val,
                            status="active",
                            title=title_txt or (content_txt[:72] if content_txt else "Updated direction"),
                            content=content_txt or title_txt,
                            applicable_roles=norm_applicable or (list(existing.applicable_roles) if existing else []),
                            slice_id=eff_slice,
                            item_id=item_id,
                            created_at_utc=now_iso,
                            updated_at_utc=now_iso,
                        )
                        ledger.entries.append(replacement)
                        by_id[replacement.memory_id] = replacement
                        touched.append(replacement)
                    if existing is not None:
                        existing.deliberation = prop.deliberation
                        existing.author_role = norm_author
                        existing.status = "superseded"
                        if replacement is not None:
                            existing.superseded_by_id = replacement.memory_id
                        if res_note:
                            existing.resolution_note = res_note
                        existing.updated_at_utc = now_iso
                        touched.append(existing)

            if emb_cache_dirty and emb_cache is not None:
                self._save_cached_embeddings(emb_cache)
            if touched:
                self.save_ledger(ledger)
                self._emit_event(
                    "game_memory_updated",
                    deliberation=touched[-1].deliberation,
                    payload={
                        "cartridge_id": ledger.cartridge_id,
                        "author_role": norm_author,
                        "slice_id": eff_slice,
                        "item_id": item_id,
                        "updated_memory_ids": [e.memory_id for e in touched],
                        "deduplicated_memory_ids": deduplicated_ids,
                        "banks_touched": sorted({e.memory_bank for e in touched}),
                        "total_entries": len(ledger.entries),
                    },
                )
            return touched

    def parse_and_apply_compact_note(
        self,
        author_role: str,
        raw_note: str,
        *,
        deliberation: str,
        slice_id: str = "",
        item_id: str = "",
        client: Optional[Any] = None,
    ) -> Optional[GameMemoryEntry]:
        """Parses compact `WritersRoomTurnProposal.memory_note` syntax:
          - `council: <title> | <content>`
          - `council:strategic_goal: <title> | <content>`
          - `self:lesson_learned: <content>`
          - `<content>` (defaults to `target_bank='self'`, `kind='domain_note'`)
        """
        note = str(raw_note or "").strip()
        if not note:
            return None
        target_bank = "self"
        kind: MemoryKind = "domain_note"

        head_parts = note.split(":", 2)
        if len(head_parts) >= 2:
            p0 = head_parts[0].strip()
            p0_low = p0.lower()
            is_known_bank = p0_low in ("self", "own", "council", "shared") or p0_low in _CANONICAL_ROLE_LOOKUP
            if is_known_bank:
                target_bank = p0
                if len(head_parts) == 3 and head_parts[1].strip().lower() in _VALID_KINDS:
                    kind = head_parts[1].strip().lower()  # type: ignore[assignment]
                    note = head_parts[2].strip()
                else:
                    if p0_low in ("council", "shared"):
                        kind = "strategic_goal"
                    note = ":".join(head_parts[1:]).strip()
            elif p0_low in _VALID_KINDS:
                kind = p0_low  # type: ignore[assignment]
                note = ":".join(head_parts[1:]).strip()

        if "|" in note:
            t_part, c_part = note.split("|", 1)
            title = t_part.strip()
            content = c_part.strip() or title
        elif "::" in note:
            t_part, c_part = note.split("::", 1)
            title = t_part.strip()
            content = c_part.strip() or title
        else:
            title = note[:72].rstrip(".")
            content = note

        if not content:
            return None

        created = self.apply_mutations(
            author_role=author_role,
            proposals=[
                MemoryMutationProposal(
                    deliberation=deliberation or "Specialist recorded durable Writers' Room memory note.",
                    operation="record",
                    target_bank=target_bank,
                    kind=kind,
                    title=title,
                    content=content,
                )
            ],
            slice_id=slice_id,
            item_id=item_id,
            client=client,
        )
        return created[0] if created else None

    def _embed_memory_texts(self, client: Any, texts: Sequence[str]) -> List[List[float]]:
        """Embeds texts with `gemini-embedding-001` (`MEMORY_EMBEDDING_DIM=256`). Fails fast if `client` lacks `embed_content`."""
        if not texts:
            return []
        models_api = getattr(client, "models", None) if client is not None else None
        embed_fn = getattr(models_api, "embed_content", None) if models_api is not None else None
        if not callable(embed_fn):
            raise StudioAgentError(
                "Direct semantic memory query (`query_text`) requires a Gemini client supporting `models.embed_content` (Rule 5)."
            )
        from engine.studio.models import call_gemini_with_transient_retry

        cfg: Any = {"output_dimensionality": MEMORY_EMBEDDING_DIM}
        try:
            from google.genai import types as genai_types

            cfg = genai_types.EmbedContentConfig(output_dimensionality=MEMORY_EMBEDDING_DIM)
        except Exception:
            pass

        resp = call_gemini_with_transient_retry(
            lambda: embed_fn(
                model=MEMORY_EMBEDDING_MODEL,
                contents=list(texts),
                config=cfg,
            ),
            label="game_memory.embed_content",
            candidate_dir=self.cartridge_dir,
        )
        embeddings_attr = getattr(resp, "embeddings", None)
        if not embeddings_attr:
            raise StudioAgentError("`embed_content` returned no embeddings during long-term memory query (Rule 5).")
        vectors: List[List[float]] = []
        for emb in embeddings_attr:
            vals = getattr(emb, "values", None)
            if vals is None and isinstance(emb, dict):
                vals = emb.get("values")
            if not vals:
                raise StudioAgentError("`embed_content` returned an empty vector during long-term memory query (Rule 5).")
            vectors.append([float(x) for x in vals])
        return vectors

    def _load_cached_embeddings(self) -> Dict[str, List[float]]:
        cache_path = self.cartridge_dir / "artifacts" / "memory_embeddings.json"
        if not cache_path.is_file():
            return {}
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    str(k): [float(v) for v in vec]
                    for k, vec in data.items()
                    if isinstance(vec, list) and vec
                }
        except Exception:
            return {}
        return {}

    def _save_cached_embeddings(self, cache: Dict[str, List[float]]) -> None:
        cache_path = self.cartridge_dir / "artifacts" / "memory_embeddings.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        os.replace(tmp, cache_path)

    def query_memories(
        self,
        caller_role: str,
        query_text: str = "",
        *,
        bank_filter: str = "",
        kind_filter: str = "",
        status_filter: str = "active",
        top_k: int = 8,
        client: Any = None,
    ) -> List[Tuple[GameMemoryEntry, float]]:
        """Queries long-term game and agent memories.

        - When `query_text` is empty: returns deterministically filtered entries ordered by bank affinity
          (caller's own bank + 'council' first) and recency.
        - When `query_text` is non-empty: ranks filtered entries by `gemini-embedding-001` cosine similarity
          (cached by SHA-256) with bank-affinity weighting (Rule 2: zero keyword/regex heuristics).
        """
        norm_caller = normalize_agent_role(caller_role)
        candidates = self.list_entries(
            bank_filter=bank_filter,
            caller_role=norm_caller,
            kind_filter=kind_filter,
            status_filter=status_filter,
        )
        q_clean = str(query_text or "").strip()
        if not candidates:
            if q_clean:
                # Still validate client capability if caller explicitly requested semantic search without silent fallback
                models_api = getattr(client, "models", None) if client is not None else None
                if not callable(getattr(models_api, "embed_content", None)):
                    raise StudioAgentError(
                        "Direct semantic memory query (`query_text`) requires a Gemini client supporting `models.embed_content` (Rule 5)."
                    )
            return []

        def _affinity_bonus(entry: GameMemoryEntry) -> float:
            if entry.memory_bank == norm_caller:
                return 0.06
            if entry.memory_bank == "council":
                return 0.04
            if norm_caller in (entry.applicable_roles or []):
                return 0.04
            return 0.0

        if not q_clean:
            ranked = sorted(
                candidates,
                key=lambda e: (_affinity_bonus(e), e.updated_at_utc),
                reverse=True,
            )[: max(1, top_k)]
            return [(e, 1.0) for e in ranked]

        with self._lock:
            emb_cache = self._load_cached_embeddings()
        entry_hashes: List[str] = []
        missing_hashes: List[str] = []
        missing_texts: List[str] = []
        for entry in candidates:
            txt = _entry_embedding_text(entry)
            digest = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:24]
            entry_hashes.append(digest)
            if digest not in emb_cache and digest not in missing_hashes:
                missing_hashes.append(digest)
                missing_texts.append(txt)

        batch_texts = [q_clean] + missing_texts
        batch_vecs = self._embed_memory_texts(client, batch_texts)
        query_vec = batch_vecs[0]
        if missing_hashes:
            with self._lock:
                for h, vec in zip(missing_hashes, batch_vecs[1:]):
                    emb_cache[h] = vec
                self._save_cached_embeddings(emb_cache)

        scored: List[Tuple[GameMemoryEntry, float]] = []
        unfiltered_bank = not bank_filter or bank_filter.lower() == "all"
        for entry, digest in zip(candidates, entry_hashes):
            vec = emb_cache.get(digest) or []
            sim = _cosine_similarity(query_vec, vec)
            score = sim + (_affinity_bonus(entry) if unfiltered_bank else 0.0)
            scored.append((entry, round(float(score), 4)))

        scored.sort(key=lambda pair: (pair[1], pair[0].updated_at_utc), reverse=True)
        top_results = scored[: max(1, top_k)]
        self._emit_event(
            "game_memory_queried",
            deliberation=f"Agent '{norm_caller}' executed semantic memory query '{q_clean[:120]}' across {len(candidates)} candidate memories.",
            payload={
                "cartridge_id": self.cartridge_dir.name,
                "caller_role": norm_caller,
                "query_text": q_clean,
                "bank_filter": bank_filter or "all",
                "result_count": len(top_results),
                "top_memory_ids": [e.memory_id for e, _ in top_results],
            },
        )
        return top_results

    def format_role_memory_briefing(
        self,
        target_role: str,
        *,
        include_fulfilled_for_reflection: bool = False,
        max_entries: int = 12,
    ) -> str:
        """Formats proactive long-term memories relevant to `target_role` ('council' bank + `target_role` bank
        + peer entries explicitly listing `target_role` in `applicable_roles`). Returns '' when empty."""
        norm_role = normalize_agent_role(target_role)
        ledger = self.load_ledger()
        if not ledger.entries:
            return ""

        relevant: List[GameMemoryEntry] = []
        allowed_statuses = {"active", "fulfilled"} if include_fulfilled_for_reflection else {"active"}
        for entry in ledger.entries:
            if entry.status not in allowed_statuses:
                continue
            if (
                entry.memory_bank == "council"
                or entry.memory_bank == norm_role
                or norm_role in (entry.applicable_roles or [])
            ):
                relevant.append(entry)

        if not relevant:
            return ""

        relevant.sort(
            key=lambda e: (
                1 if e.status == "active" else 0,
                2 if e.memory_bank == norm_role else (1 if e.memory_bank == "council" else 0),
                e.updated_at_utc,
            ),
            reverse=True,
        )
        selected = relevant[: max(1, max_entries)]
        lines = [
            f"LONG-TERM GAME & AGENT MEMORY BRIEFING (for `{norm_role}` — persistent across vertical slices; "
            f"prefer updating/fulfilling existing `[mem_...]` IDs over creating duplicates, and leave `memory_updates` empty on routine turns):"
        ]
        for entry in selected:
            prov = f" slice={entry.slice_id}" if entry.slice_id else ""
            item_tag = f" item={entry.item_id}" if entry.item_id else ""
            lines.append(
                f"  - [{entry.memory_id}] (bank={entry.memory_bank}, kind={entry.kind}, status={entry.status}{prov}{item_tag}) "
                f"{entry.title}: {entry.content}"
            )
        return "\n".join(lines) + "\n"


def format_memory_query_results(
    query_text: str,
    bank_filter: str,
    results: Sequence[Tuple[GameMemoryEntry, float]],
) -> str:
    """Formats semantic memory query results so an agent can answer its `memory_query` on the next turn."""
    header = (
        f"LONG-TERM MEMORY QUERY RESULTS for `{query_text}` "
        f"(bank_filter='{bank_filter or 'all'}', matches={len(results)}):"
    )
    if not results:
        return f"{header}\n  (No matching memory entries found in the game memory ledger.)\n"
    lines = [header]
    for entry, score in results:
        prov = f", slice={entry.slice_id}" if entry.slice_id else ""
        res = f" [Resolution: {entry.resolution_note}]" if entry.resolution_note else ""
        lines.append(
            f"  - [{entry.memory_id}] score={score:.3f} (bank={entry.memory_bank}, kind={entry.kind}, "
            f"status={entry.status}, author={entry.author_role}{prov}) {entry.title}: {entry.content}{res}"
        )
    return "\n".join(lines) + "\n"
