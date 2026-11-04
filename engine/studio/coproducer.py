"""engine/studio/coproducer.py
Interactive 1-on-1 Creative Co-Producer & Multimodal Showrunner Agent.

Responsibilities:
- High-creativity ideation & refinement partner (`temperature=0.90` synthesis gear) with inline multimodal
  sketching (16:9 concept mockups, sprite + dual-checkerboard plates, TTS character auditions, DSP SFX/loops).
- Backchannel Council consultations (`SPECIALIST_ROSTER` huddles + divergent fragment/collision rounds) and
  lock-free read-only specialist probes (`swe_engineer`, `visual_artist`, `audio_designer`, `script_writer`,
  `qa_test_engineer`), surfaced as collapsible Consultation Cards while preserving a 1-on-1 voice.
- Primary showrunner window into build progress, assets, code diffs, and playtest videos (`replay.mp4`),
  grounded multimodally via attached `.png` frames and perceptual video/audio evaluators.
- Explicit Action Proposal generation for Two-Mode Council Handoffs (`loose_spark` vs `refined_concept`)
  and Three-Tier Pivots (`pivot_course_correction`, `pivot_structural`, `pivot_fork_cartridge`).

Strict Compliance:
- Rule 1: 100% domain-agnostic prompts, schemas, and logic.
- Rule 2: Zero regex/keyword intent heuristics; all tool & sufficiency routing uses `CoProducerStepDecision`.
- Rule 4: `deliberation` at index 0 on every model, persisted record, and emitted event (`reasoning_rationale`).
- Rule 5: Zero silent fallbacks; missing clients or failed tool syntheses surface authentic errors.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import uuid

from google.genai import types

from engine.studio.models import (
    ConceptArtCritiqueVerdict,
    ConceptArtShotPlan,
    CoProducerActionProposal,
    CoProducerConceptBrief,
    CoProducerConsultationCard,
    CoProducerFinalTurn,
    CoProducerMediaCard,
    CoProducerMediaItem,
    CoProducerStepDecision,
    CoProducerToolCall,
    DEFAULT_CONCEPT_MOCKUP_MODEL,
    DEFAULT_STUDIO_MODEL,
    GameBibleSpec,
    StudioAgentConfigurationError,
    StudioAgentError,
    TOKEN_USAGE_TRACKER,
    bounded_thinking_config,
    call_gemini_with_transient_retry,
    interactive_foreground_call_scope,
    is_real_genai_client,
)

logger = logging.getLogger("CoProducerAgent")

DRAFTS_DIRNAME = "_coproducer_drafts"
THREAD_SNAPSHOT_FILENAME = "co_producer.json"
HANDOFF_BRIEF_FILENAME = "co_producer_brief.json"

_STORE_LOCK = threading.RLock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(raw: str, fallback: str = "sketch") -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(raw or "")).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return (cleaned[:40].strip("_") or fallback)


def _ordered_rationale_dict(deliberation: str, **rest: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "deliberation": str(deliberation or "Co-Producer state persisted.").strip()
        or "Co-Producer state persisted."
    }
    for k, v in rest.items():
        if k != "deliberation":
            out[k] = v
    return out


class CoProducerThreadStore:
    """Dual-persisted store mirroring Co-Producer threads across SQLite WAL and per-cartridge/draft `co_producer.json`."""

    def __init__(self, cartridges_root: Path, db: Optional[Any] = None) -> None:
        self.cartridges_root = Path(cartridges_root)
        self.drafts_root = self.cartridges_root / DRAFTS_DIRNAME
        self.db = db

    def resolve_thread_dir(self, thread_meta: Dict[str, Any]) -> Path:
        scope = str(thread_meta.get("scope_kind") or "cartridge")
        cid = str(thread_meta.get("cartridge_id") or "").strip()
        tid = str(thread_meta.get("thread_id") or "").strip()
        if scope == "cartridge" and cid:
            return self.cartridges_root / cid
        return self.drafts_root / (tid or f"draft_{uuid.uuid4().hex[:8]}")

    def _snapshot_path(self, thread_dir: Path) -> Path:
        return Path(thread_dir) / THREAD_SNAPSHOT_FILENAME

    def _write_disk_snapshot(self, thread_meta: Dict[str, Any], messages: List[Dict[str, Any]]) -> Path:
        t_dir = self.resolve_thread_dir(thread_meta)
        t_dir.mkdir(parents=True, exist_ok=True)
        snap = _ordered_rationale_dict(
            str(thread_meta.get("deliberation") or "Persisted Co-Producer thread snapshot."),
            thread=thread_meta,
            messages=messages,
            updated_at_utc=_utc_now_iso(),
        )
        snap_path = self._snapshot_path(t_dir)
        tmp_path = snap_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_path, snap_path)
        return snap_path

    def _sync_cartridge_brief_file(
        self,
        thread_meta: Dict[str, Any],
        concept_brief: Optional[Dict[str, Any]],
        *,
        override_seed_brief: Optional[str] = None,
        handoff_mode: Optional[str] = None,
    ) -> None:
        """Keeps `<cartridge_dir>/co_producer_brief.json` and `assets/mockups/` synchronized whenever a
        cartridge-bound Co-Producer thread refines its `concept_brief`, preventing stale Turn-N handoff snapshots."""
        if not isinstance(concept_brief, dict) or not concept_brief:
            return
        if str(thread_meta.get("scope_kind") or "") != "cartridge":
            return
        cid = str(thread_meta.get("cartridge_id") or "").strip()
        if not cid:
            return
        c_dir = self.cartridges_root / cid
        if not c_dir.is_dir():
            return
        brief_path = c_dir / HANDOFF_BRIEF_FILENAME
        existing_env: Dict[str, Any] = {}
        if brief_path.is_file():
            try:
                loaded = json.loads(brief_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing_env = loaded
            except Exception:
                existing_env = {}
        elif (c_dir / "game_bible.json").is_file():
            # Do not create a brand-new pre-genesis handoff file on legacy cartridges that started directly from Genesis
            return

        eff_cb = dict(concept_brief)
        remapped_mockups: List[str] = []
        for raw_m in list(eff_cb.get("selected_mockup_paths") or []):
            rel_m = str(raw_m or "").strip().lstrip("/")
            if rel_m.startswith(f"{cid}/"):
                rel_m = rel_m[len(cid) + 1:]
            if not rel_m:
                continue
            src_candidate = c_dir / rel_m
            if src_candidate.is_file():
                dst_mockups = c_dir / "assets" / "mockups"
                dst_mockups.mkdir(parents=True, exist_ok=True)
                dst_file = dst_mockups / src_candidate.name
                if src_candidate.resolve() != dst_file.resolve():
                    shutil.copy2(src_candidate, dst_file)
                mapped_rel = f"assets/mockups/{dst_file.name}"
                if mapped_rel not in remapped_mockups:
                    remapped_mockups.append(mapped_rel)
            elif rel_m not in remapped_mockups:
                remapped_mockups.append(rel_m)
        if remapped_mockups:
            eff_cb["selected_mockup_paths"] = remapped_mockups

        eff_mode = (
            handoff_mode
            or str(existing_env.get("handoff_mode") or "").strip()
            or ("refined_concept" if eff_cb.get("working_title") else "loose_spark")
        )
        pillars_summary = "; ".join(
            str(p).strip()
            for p in (eff_cb.get("user_confirmed_pillars") or eff_cb.get("core_player_verbs") or [])[:6]
            if str(p).strip()
        )
        synthesized_seed = str(override_seed_brief or "").strip()
        if not synthesized_seed:
            pitch_txt = str(eff_cb.get("elevator_pitch") or "").strip()
            if pitch_txt:
                synthesized_seed = (
                    f"{pitch_txt} Core pillars: {pillars_summary}" if pillars_summary else pitch_txt
                )
            else:
                synthesized_seed = str(existing_env.get("steering_or_seed_brief") or "").strip()

        now_iso = _utc_now_iso()
        updated_env = _ordered_rationale_dict(
            str(eff_cb.get("deliberation") or existing_env.get("deliberation") or f"Synced living Co-Producer brief for '{cid}'."),
            handoff_mode=eff_mode,
            graduated_from_thread_id=str(existing_env.get("graduated_from_thread_id") or thread_meta.get("thread_id") or ""),
            cartridge_id=cid,
            steering_or_seed_brief=synthesized_seed,
            concept_brief=eff_cb,
            created_at_utc=str(existing_env.get("created_at_utc") or now_iso),
            updated_at_utc=now_iso,
        )
        tmp_brief = brief_path.with_suffix(".json.tmp")
        tmp_brief.write_text(json.dumps(updated_env, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_brief, brief_path)
        thread_meta["concept_brief"] = eff_cb

    def _sync_from_disk_if_needed(self, thread_dir: Path) -> Optional[Dict[str, Any]]:
        snap_path = self._snapshot_path(thread_dir)
        if not snap_path.is_file():
            return None
        try:
            raw = json.loads(snap_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed reading %s: %s", snap_path, exc)
            return None
        if not isinstance(raw, dict):
            return None
        t_meta = raw.get("thread")
        msgs = raw.get("messages")
        if not isinstance(t_meta, dict) or not str(t_meta.get("thread_id") or "").strip():
            return None
        # Heal stale co_producer_brief.json if co_producer.json has a newer updated_at_utc
        brief_path = Path(thread_dir) / HANDOFF_BRIEF_FILENAME
        if brief_path.is_file() and isinstance(t_meta.get("concept_brief"), dict) and t_meta.get("concept_brief"):
            try:
                b_env = json.loads(brief_path.read_text(encoding="utf-8"))
                b_ts = str((b_env if isinstance(b_env, dict) else {}).get("updated_at_utc") or (b_env if isinstance(b_env, dict) else {}).get("created_at_utc") or "")
                t_ts = str(t_meta.get("updated_at_utc") or "")
                if t_ts and (not b_ts or t_ts > b_ts):
                    self._sync_cartridge_brief_file(t_meta, t_meta["concept_brief"])
            except Exception:
                pass
        if self.db is not None and hasattr(self.db, "get_coproducer_thread"):
            existing = self.db.get_coproducer_thread(str(t_meta["thread_id"]))
            if existing is None:
                self.db.upsert_coproducer_thread(t_meta)
                for m in (msgs if isinstance(msgs, list) else []):
                    if isinstance(m, dict):
                        self.db.insert_coproducer_message(m)
        return {"thread": t_meta, "messages": msgs if isinstance(msgs, list) else []}

    def create_draft_thread(self, title: str = "New Game Concept") -> Dict[str, Any]:
        with _STORE_LOCK:
            now_iso = _utc_now_iso()
            tid = f"draft_{uuid.uuid4().hex[:8]}"
            clean_title = (title or "").strip() or "New Game Concept"
            brief = CoProducerConceptBrief(
                deliberation="Initialized blank pre-genesis concept brief for interactive Co-Producer ideation.",
                working_title=clean_title if clean_title != "New Game Concept" else "",
            ).model_dump()
            thread_meta = _ordered_rationale_dict(
                "Created new pre-genesis Co-Producer draft concept thread.",
                thread_id=tid,
                scope_kind="draft",
                cartridge_id=None,
                title=clean_title,
                graduated_to_cartridge_id="",
                concept_brief=brief,
                message_count=0,
                created_at_utc=now_iso,
                updated_at_utc=now_iso,
            )
            if self.db is not None and hasattr(self.db, "upsert_coproducer_thread"):
                self.db.upsert_coproducer_thread(thread_meta)
            self._write_disk_snapshot(thread_meta, [])
            return thread_meta

    def get_or_create_cartridge_thread(self, cartridge_id: str, game_title: str = "") -> Dict[str, Any]:
        cid = (cartridge_id or "").strip()
        if not cid:
            return self.create_draft_thread(title=game_title or "New Game Concept")
        with _STORE_LOCK:
            c_dir = self.cartridges_root / cid
            if c_dir.is_dir():
                self._sync_from_disk_if_needed(c_dir)
            if self.db is not None and hasattr(self.db, "list_coproducer_threads"):
                existing_list = self.db.list_coproducer_threads(cartridge_id=cid, include_graduated_drafts=True)
                if existing_list:
                    latest = existing_list[0]
                    if game_title and latest.get("title") in ("", "New Game Concept", cid):
                        latest["title"] = game_title
                        self.db.upsert_coproducer_thread(latest)
                    return latest

            # Fallback check on disk snapshot if DB is None
            disk_loaded = self._sync_from_disk_if_needed(c_dir) if c_dir.is_dir() else None
            if disk_loaded and isinstance(disk_loaded.get("thread"), dict):
                return disk_loaded["thread"]

            # Seed initial concept brief from game_bible.json if available
            resolved_title = game_title.strip() if game_title else ""
            bible_brief: Dict[str, Any] = {}
            gb_path = c_dir / "game_bible.json"
            if gb_path.is_file():
                try:
                    gb = json.loads(gb_path.read_text(encoding="utf-8"))
                    if isinstance(gb, dict):
                        resolved_title = resolved_title or str(gb.get("game_title") or "").strip()
                        bible_brief = CoProducerConceptBrief(
                            deliberation="Seeded Co-Producer concept brief from existing cartridge Game Bible.",
                            working_title=str(gb.get("game_title") or ""),
                            core_genre=str(gb.get("core_genre") or ""),
                            elevator_pitch=str(gb.get("elevator_pitch") or ""),
                            core_fantasy=str(gb.get("core_fantasy") or ""),
                            core_player_verbs=[str(p) for p in (gb.get("gameplay_pillars") or []) if str(p).strip()],
                            first_60_seconds_feel=str(gb.get("first_60_seconds_hook") or ""),
                            visual_style_direction=str(gb.get("visual_style_guide") or ""),
                            audio_mood_direction=str(gb.get("audio_direction_guide") or ""),
                            color_palette_hex=[str(c) for c in (gb.get("color_palette_hex") or []) if str(c).strip()],
                        ).model_dump()
                except Exception:
                    pass

            if not bible_brief:
                bible_brief = CoProducerConceptBrief(
                    deliberation=f"Initialized Co-Producer thread for cartridge {cid}.",
                    working_title=resolved_title,
                ).model_dump()

            now_iso = _utc_now_iso()
            tid = f"thread_{_safe_slug(cid, 'cartridge')}"
            thread_meta = _ordered_rationale_dict(
                f"Initialized Co-Producer showrunner thread for cartridge '{cid}'.",
                thread_id=tid,
                scope_kind="cartridge",
                cartridge_id=cid,
                title=resolved_title or cid,
                graduated_to_cartridge_id="",
                concept_brief=bible_brief,
                message_count=0,
                created_at_utc=now_iso,
                updated_at_utc=now_iso,
            )
            if self.db is not None and hasattr(self.db, "upsert_coproducer_thread"):
                self.db.upsert_coproducer_thread(thread_meta)
            if c_dir.is_dir():
                self._write_disk_snapshot(thread_meta, [])
            return thread_meta

    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        tid = (thread_id or "").strip()
        if not tid:
            return None
        with _STORE_LOCK:
            if self.db is not None and hasattr(self.db, "get_coproducer_thread"):
                found = self.db.get_coproducer_thread(tid)
                if found is not None:
                    return found
            # Check draft folder or cartridge folders on disk
            cand_draft_dir = self.drafts_root / tid
            if cand_draft_dir.is_dir():
                loaded = self._sync_from_disk_if_needed(cand_draft_dir)
                if loaded:
                    return loaded["thread"]
            if self.cartridges_root.is_dir():
                for sub in self.cartridges_root.iterdir():
                    if not sub.is_dir() or sub.name.startswith(".") or sub.name == DRAFTS_DIRNAME:
                        continue
                    loaded = self._sync_from_disk_if_needed(sub)
                    if loaded and str(loaded["thread"].get("thread_id") or "") == tid:
                        return loaded["thread"]
            return None

    def get_thread_detail(self, thread_id: str, limit: int = 200) -> Optional[Dict[str, Any]]:
        with _STORE_LOCK:
            meta = self.get_thread(thread_id)
            if meta is None:
                return None
            messages: List[Dict[str, Any]] = []
            if self.db is not None and hasattr(self.db, "list_coproducer_messages"):
                messages = self.db.list_coproducer_messages(meta["thread_id"], limit=limit)
            if not messages:
                t_dir = self.resolve_thread_dir(meta)
                loaded = self._sync_from_disk_if_needed(t_dir)
                if loaded and isinstance(loaded.get("messages"), list):
                    messages = loaded["messages"][-limit:]
            return _ordered_rationale_dict(
                f"Loaded Co-Producer thread '{meta['thread_id']}' with {len(messages)} message(s).",
                thread=meta,
                messages=messages,
            )

    def list_threads(self, active_cartridge_id: Optional[str] = None) -> Dict[str, Any]:
        with _STORE_LOCK:
            disk_threads_by_id: Dict[str, Dict[str, Any]] = {}
            # Opportunistically sync draft folders on disk
            if self.drafts_root.is_dir():
                for d_sub in sorted(self.drafts_root.iterdir()):
                    if d_sub.is_dir():
                        loaded = self._sync_from_disk_if_needed(d_sub)
                        if loaded and isinstance(loaded.get("thread"), dict):
                            t_obj = loaded["thread"]
                            if not str(t_obj.get("graduated_to_cartridge_id") or "").strip():
                                disk_threads_by_id[str(t_obj["thread_id"])] = t_obj

            # Opportunistically sync per-cartridge co_producer.json snapshots across all games on disk
            if self.cartridges_root.is_dir():
                for c_sub in sorted(self.cartridges_root.iterdir()):
                    if not c_sub.is_dir() or c_sub.name.startswith(".") or c_sub.name == DRAFTS_DIRNAME:
                        continue
                    loaded = self._sync_from_disk_if_needed(c_sub)
                    if loaded and isinstance(loaded.get("thread"), dict):
                        t_obj = loaded["thread"]
                        disk_threads_by_id[str(t_obj["thread_id"])] = t_obj

            active_thread: Optional[Dict[str, Any]] = None
            clean_cid = (active_cartridge_id or "").strip()
            if clean_cid and (self.cartridges_root / clean_cid).is_dir():
                active_thread = self.get_or_create_cartridge_thread(clean_cid)
                if active_thread and active_thread.get("thread_id"):
                    disk_threads_by_id[str(active_thread["thread_id"])] = active_thread

            threads: List[Dict[str, Any]] = []
            if self.db is not None and hasattr(self.db, "list_coproducer_threads"):
                threads = self.db.list_coproducer_threads(include_graduated_drafts=False)
            else:
                threads = list(disk_threads_by_id.values())

            draft_threads = [t for t in threads if t.get("scope_kind") == "draft"]
            cartridge_threads = [t for t in threads if t.get("scope_kind") == "cartridge"]
            return _ordered_rationale_dict(
                "Listed Co-Producer draft and cartridge conversation threads.",
                active_cartridge_thread=active_thread,
                draft_threads=draft_threads,
                cartridge_threads=cartridge_threads,
                threads=threads,
            )

    def append_message(
        self,
        thread_meta: Dict[str, Any],
        message: Dict[str, Any],
        *,
        updated_title: Optional[str] = None,
        updated_concept_brief: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with _STORE_LOCK:
            tid = str(thread_meta["thread_id"])
            existing_detail = self.get_thread_detail(tid, limit=500)
            msgs = list((existing_detail or {}).get("messages") or [])
            next_idx = (max((int(m.get("turn_index") or 0) for m in msgs), default=-1) + 1)
            now_iso = _utc_now_iso()
            msg_record = _ordered_rationale_dict(
                str(message.get("deliberation") or "Recorded Co-Producer conversation turn."),
                message_id=str(message.get("message_id") or f"cp_msg_{uuid.uuid4().hex[:10]}"),
                thread_id=tid,
                turn_index=int(message.get("turn_index", next_idx)),
                sender_role=str(message.get("sender_role") or "coproducer"),
                content_markdown=str(message.get("content_markdown") or ""),
                media_cards=list(message.get("media_cards") or []),
                consultation_cards=list(message.get("consultation_cards") or []),
                action_proposals=list(message.get("action_proposals") or []),
                tool_trace=list(message.get("tool_trace") or []),
                prompt_trace=list(message.get("prompt_trace") or []),
                token_usage=dict(message.get("token_usage") or {}),
                created_at_utc=str(message.get("created_at_utc") or now_iso),
            )
            updated_meta = dict(thread_meta)
            if updated_title and updated_title.strip():
                updated_meta["title"] = updated_title.strip()
            if isinstance(updated_concept_brief, dict) and updated_concept_brief:
                updated_meta["concept_brief"] = updated_concept_brief
                self._sync_cartridge_brief_file(updated_meta, updated_concept_brief)
            updated_meta["updated_at_utc"] = now_iso
            msgs.append(msg_record)
            updated_meta["message_count"] = len(msgs)

            if self.db is not None:
                if hasattr(self.db, "upsert_coproducer_thread"):
                    self.db.upsert_coproducer_thread(updated_meta)
                if hasattr(self.db, "insert_coproducer_message"):
                    self.db.insert_coproducer_message(msg_record)
            self._write_disk_snapshot(updated_meta, msgs)
            return updated_meta, msg_record

    def save_concept_brief(
        self,
        thread_meta: Dict[str, Any],
        concept_brief: Dict[str, Any],
    ) -> Dict[str, Any]:
        with _STORE_LOCK:
            tid = str(thread_meta["thread_id"])
            existing_detail = self.get_thread_detail(tid, limit=500)
            msgs = list((existing_detail or {}).get("messages") or [])
            updated_meta = dict(thread_meta)
            updated_meta["concept_brief"] = dict(concept_brief)
            self._sync_cartridge_brief_file(updated_meta, updated_meta["concept_brief"])
            updated_meta["updated_at_utc"] = _utc_now_iso()
            if self.db is not None and hasattr(self.db, "upsert_coproducer_thread"):
                self.db.upsert_coproducer_thread(updated_meta)
            self._write_disk_snapshot(updated_meta, msgs)
            return updated_meta

    def update_action_proposal_status(
        self,
        thread_id: str,
        message_id: str,
        proposal_id: str,
        status: str,
        result_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        with _STORE_LOCK:
            detail = self.get_thread_detail(thread_id, limit=500)
            if detail is None:
                return None
            meta = dict(detail["thread"])
            msgs = list(detail["messages"])
            updated_prop: Optional[Dict[str, Any]] = None
            now_iso = _utc_now_iso()
            for m in msgs:
                if str(m.get("message_id") or "") != message_id:
                    continue
                for p in (m.get("action_proposals") or []):
                    if isinstance(p, dict) and str(p.get("proposal_id") or "") == proposal_id:
                        p["status"] = status
                        p["executed_at_utc"] = now_iso
                        if isinstance(result_meta, dict):
                            p["execution_result"] = result_meta
                        updated_prop = p
                        break
            if self.db is not None and hasattr(self.db, "update_coproducer_message_action_state"):
                db_updated = self.db.update_coproducer_message_action_state(
                    thread_id=thread_id,
                    message_id=message_id,
                    proposal_id=proposal_id,
                    status=status,
                    result_meta=result_meta,
                )
                if updated_prop is None:
                    updated_prop = db_updated
            meta["updated_at_utc"] = now_iso
            self._write_disk_snapshot(meta, msgs)
            return updated_prop

    @staticmethod
    def _rewrite_draft_path(raw_path: str, draft_prefix: str) -> str:
        s = str(raw_path or "").strip()
        if not s:
            return s
        norm = s.lstrip("/")
        if norm.startswith(draft_prefix):
            return norm[len(draft_prefix):].lstrip("/")
        return s

    def graduate_draft_to_cartridge(
        self,
        *,
        draft_thread_id: str,
        new_cartridge_dir: Path,
        handoff_mode: str,
        steering_or_seed_brief: str,
        concept_brief: Optional[Dict[str, Any]] = None,
        carry_over_asset_paths: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Graduates a pre-genesis draft thread and its sketches into `new_cartridge_dir`, writing
        `co_producer_brief.json` and binding the SQLite WAL thread to `new_cartridge_dir.name`."""
        with _STORE_LOCK:
            new_c_dir = Path(new_cartridge_dir)
            new_c_dir.mkdir(parents=True, exist_ok=True)
            cid = new_c_dir.name
            detail = self.get_thread_detail(draft_thread_id, limit=500)
            meta = dict((detail or {}).get("thread") or {})
            msgs = list((detail or {}).get("messages") or [])
            eff_brief = dict(concept_brief or meta.get("concept_brief") or {})

            draft_dir = self.drafts_root / draft_thread_id
            draft_prefix = f"{DRAFTS_DIRNAME}/{draft_thread_id}/"

            # 1. Copy all draft sketches into new_cartridge_dir/assets/sketches (and approved mockups into assets/mockups)
            if draft_dir.is_dir():
                src_sketches = draft_dir / "assets" / "sketches"
                if src_sketches.is_dir():
                    dst_sketches = new_c_dir / "assets" / "sketches"
                    dst_sketches.mkdir(parents=True, exist_ok=True)
                    for item in src_sketches.iterdir():
                        if item.is_file():
                            shutil.copy2(item, dst_sketches / item.name)

            remapped_mockups: List[str] = []
            for raw_m in list(eff_brief.get("selected_mockup_paths") or []):
                rel_m = self._rewrite_draft_path(str(raw_m), draft_prefix)
                src_candidate = new_c_dir / rel_m
                if not src_candidate.is_file() and draft_dir.is_dir():
                    alt = draft_dir / rel_m
                    if alt.is_file():
                        src_candidate = alt
                if src_candidate.is_file():
                    dst_mockups = new_c_dir / "assets" / "mockups"
                    dst_mockups.mkdir(parents=True, exist_ok=True)
                    dst_file = dst_mockups / src_candidate.name
                    if src_candidate.resolve() != dst_file.resolve():
                        shutil.copy2(src_candidate, dst_file)
                    remapped_mockups.append(f"assets/mockups/{dst_file.name}")
                elif rel_m:
                    remapped_mockups.append(rel_m)
            if remapped_mockups:
                eff_brief["selected_mockup_paths"] = remapped_mockups

            remapped_sketches: List[str] = []
            for raw_s in list(eff_brief.get("selected_sketch_asset_paths") or []) + list(carry_over_asset_paths or []):
                rel_s = self._rewrite_draft_path(str(raw_s), draft_prefix)
                if rel_s and rel_s not in remapped_sketches:
                    remapped_sketches.append(rel_s)
            if remapped_sketches:
                eff_brief["selected_sketch_asset_paths"] = remapped_sketches

            # Rewrite draft paths in historical message media_cards so images/audio continue resolving cleanly
            for m in msgs:
                for card in (m.get("media_cards") or []):
                    if not isinstance(card, dict):
                        continue
                    for item in (card.get("items") or []):
                        if isinstance(item, dict) and item.get("artifact_path"):
                            item["artifact_path"] = self._rewrite_draft_path(str(item["artifact_path"]), draft_prefix)

            working_title = str(eff_brief.get("working_title") or meta.get("title") or "").strip() or cid
            brief_envelope = _ordered_rationale_dict(
                str(eff_brief.get("deliberation") or f"Graduated Co-Producer concept into '{cid}' ({handoff_mode})."),
                handoff_mode=handoff_mode,
                graduated_from_thread_id=draft_thread_id,
                cartridge_id=cid,
                steering_or_seed_brief=steering_or_seed_brief,
                concept_brief=eff_brief,
                created_at_utc=_utc_now_iso(),
            )
            (new_c_dir / HANDOFF_BRIEF_FILENAME).write_text(
                json.dumps(brief_envelope, indent=2, default=str),
                encoding="utf-8",
            )

            if meta:
                meta["scope_kind"] = "cartridge"
                meta["cartridge_id"] = cid
                meta["graduated_to_cartridge_id"] = cid
                meta["title"] = working_title
                meta["concept_brief"] = eff_brief
                meta["updated_at_utc"] = _utc_now_iso()
                if self.db is not None and hasattr(self.db, "bind_draft_thread_to_cartridge"):
                    bound = self.db.bind_draft_thread_to_cartridge(
                        draft_thread_id=draft_thread_id,
                        cartridge_id=cid,
                        title=working_title,
                        concept_brief=eff_brief,
                    )
                    if bound is not None:
                        meta = bound
                if self.db is not None and hasattr(self.db, "insert_coproducer_message"):
                    for m in msgs:
                        self.db.insert_coproducer_message(m)
                self._write_disk_snapshot(meta, msgs)
            else:
                meta = self.get_or_create_cartridge_thread(cid, game_title=working_title)
                meta["concept_brief"] = eff_brief
                if self.db is not None and hasattr(self.db, "upsert_coproducer_thread"):
                    self.db.upsert_coproducer_thread(meta)
                self._write_disk_snapshot(meta, [])

            return meta

    def fork_cartridge_thread_and_assets(
        self,
        *,
        source_cartridge_dir: Optional[Path],
        new_cartridge_dir: Path,
        handoff_mode: str,
        steering_or_seed_brief: str,
        concept_brief: Optional[Dict[str, Any]] = None,
        carry_over_asset_paths: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Tier 3 Non-Destructive Cartridge Fork: copies explicitly chosen assets/mockups from
        `source_cartridge_dir` into `new_cartridge_dir`, writes `co_producer_brief.json`, and seeds a fresh thread."""
        with _STORE_LOCK:
            new_c_dir = Path(new_cartridge_dir)
            new_c_dir.mkdir(parents=True, exist_ok=True)
            cid = new_c_dir.name
            eff_brief = dict(concept_brief or {})
            copied_paths: List[str] = []

            if source_cartridge_dir is not None and Path(source_cartridge_dir).is_dir():
                src_root = Path(source_cartridge_dir).resolve()
                requested = list(carry_over_asset_paths or []) + list(eff_brief.get("selected_mockup_paths") or []) + list(
                    eff_brief.get("selected_sketch_asset_paths") or []
                )
                for raw_rel in dict.fromkeys(str(p).strip() for p in requested if str(p).strip()):
                    clean_rel = raw_rel.replace("res://", "").lstrip("/")
                    if clean_rel.startswith(f"{src_root.name}/"):
                        clean_rel = clean_rel[len(src_root.name) + 1 :]
                    cand_src = (src_root / clean_rel).resolve()
                    try:
                        cand_src.relative_to(src_root)
                    except ValueError:
                        continue
                    if cand_src.is_file():
                        dst_file = (new_c_dir / clean_rel).resolve()
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(cand_src, dst_file)
                        copied_paths.append(clean_rel)

            working_title = str(eff_brief.get("working_title") or "").strip() or f"Forked Concept ({cid})"
            brief_envelope = _ordered_rationale_dict(
                str(eff_brief.get("deliberation") or f"Forked new cartridge '{cid}' carrying {len(copied_paths)} asset(s)."),
                handoff_mode=handoff_mode,
                forked_from_cartridge_id=(Path(source_cartridge_dir).name if source_cartridge_dir else ""),
                cartridge_id=cid,
                steering_or_seed_brief=steering_or_seed_brief,
                concept_brief=eff_brief,
                carried_over_assets=copied_paths,
                created_at_utc=_utc_now_iso(),
            )
            (new_c_dir / HANDOFF_BRIEF_FILENAME).write_text(
                json.dumps(brief_envelope, indent=2, default=str),
                encoding="utf-8",
            )
            thread_meta = self.get_or_create_cartridge_thread(cid, game_title=working_title)
            thread_meta["concept_brief"] = eff_brief
            thread_meta["title"] = working_title
            if self.db is not None and hasattr(self.db, "upsert_coproducer_thread"):
                self.db.upsert_coproducer_thread(thread_meta)
            self.append_message(
                thread_meta,
                {
                    "deliberation": f"Recorded non-destructive Tier 3 fork lineage into '{cid}'.",
                    "sender_role": "milestone_notice",
                    "content_markdown": (
                        f"**Forked New Cartridge `{cid}`** ({handoff_mode.replace('_', ' ')})"
                        + (
                            f" — carried over `{len(copied_paths)}` selected asset(s) from `{Path(source_cartridge_dir).name}`."
                            if source_cartridge_dir
                            else "."
                        )
                    ),
                },
                updated_title=working_title,
                updated_concept_brief=eff_brief,
            )
            return self.get_thread(thread_meta["thread_id"]) or thread_meta


class CoProducerAgent:
    """Interactive 1-on-1 Creative Co-Producer, Studio Showrunner, and Specialist Council Chair (`CoProducer`)."""

    def __init__(
        self,
        *,
        client: Any,
        cartridges_root: Path,
        db: Optional[Any] = None,
        model_name: str = DEFAULT_STUDIO_MODEL,
        image_model_name: str = "gemini-3.1-flash-lite-image",
        concept_mockup_model_name: str = DEFAULT_CONCEPT_MOCKUP_MODEL,
        tts_model_name: str = "gemini-3.8-flash-tts",
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        max_tool_hops: int = 4,
        enable_art_direction_pipeline: Optional[bool] = None,
    ) -> None:
        self.client = client
        self.cartridges_root = Path(cartridges_root)
        self.db = db
        self.store = CoProducerThreadStore(self.cartridges_root, db=db)
        self.model_name = (model_name or DEFAULT_STUDIO_MODEL).strip()
        self.image_model_name = (image_model_name or "gemini-3.1-flash-lite-image").strip()
        self.concept_mockup_model_name = (concept_mockup_model_name or DEFAULT_CONCEPT_MOCKUP_MODEL).strip()
        self.tts_model_name = (tts_model_name or "gemini-3.8-flash-tts").strip()
        self.event_callback = event_callback
        self.max_tool_hops = max(1, min(6, int(max_tool_hops)))
        self.enable_art_direction_pipeline = (
            bool(is_real_genai_client(client))
            if enable_art_direction_pipeline is None
            else bool(enable_art_direction_pipeline)
        )

    def _emit(self, event_type: str, deliberation: str, payload: Dict[str, Any]) -> None:
        ordered = _ordered_rationale_dict(
            deliberation,
            reasoning_rationale=deliberation,
            **payload,
        )
        if callable(self.event_callback):
            try:
                self.event_callback(event_type, ordered)
            except Exception:
                logger.debug("CoProducer event_callback error", exc_info=True)
        try:
            from engine.studio.orchestrator import SwarmEventBus

            SwarmEventBus.publish(
                event_type=event_type,
                agent_name="CoProducerAgent",
                deliberation=deliberation,
                payload=ordered,
            )
        except Exception:
            pass

    @staticmethod
    def _load_png_part(path: Path, max_bytes: int = 6 * 1024 * 1024) -> Optional[Any]:
        try:
            if not path.is_file():
                return None
            sz = path.stat().st_size
            if sz <= 0 or sz > max_bytes:
                return None
            ext = path.suffix.lower()
            mime = (
                "image/png"
                if ext == ".png"
                else ("image/jpeg" if ext in (".jpg", ".jpeg") else ("image/webp" if ext == ".webp" else ""))
            )
            if not mime:
                return None
            return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)
        except Exception:
            return None

    @staticmethod
    def _image_dimensions(path: Path) -> Tuple[int, int]:
        try:
            from PIL import Image

            with Image.open(path) as im:
                return int(im.width), int(im.height)
        except Exception:
            return 0, 0

    def _artifact_rel_for_thread(self, file_path: Path, thread_meta: Dict[str, Any]) -> str:
        resolved = file_path.resolve()
        try:
            rel_to_cartridges = resolved.relative_to(self.cartridges_root.resolve())
            parts = rel_to_cartridges.parts
            cid = str(thread_meta.get("cartridge_id") or "").strip()
            if cid and parts and parts[0] == cid:
                return "/".join(parts[1:])
            return "/".join(parts)
        except ValueError:
            return file_path.name

    def _call_structured(
        self,
        *,
        prompt: str,
        schema_cls: Any,
        temperature: float,
        label: str,
        multimodal_parts: Optional[Sequence[Any]] = None,
        candidate_dir: Optional[Path] = None,
        prompt_trace_sink: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Any, Dict[str, int]]:
        if self.client is None:
            raise StudioAgentConfigurationError(
                "CoProducerAgent requires an active google.genai Client (`client` is None; Rule 5: Zero Silent Fallbacks)."
            )
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_cls,
            temperature=temperature,
            thinking_config=bounded_thinking_config("low" if temperature < 0.6 else "medium"),
        )
        contents: Any = list(multimodal_parts) + [prompt] if multimodal_parts else prompt

        tok_before = TOKEN_USAGE_TRACKER.get_summary()
        res = call_gemini_with_transient_retry(
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=cfg,
            ),
            label=label,
            candidate_dir=candidate_dir,
        )
        tok_after = TOKEN_USAGE_TRACKER.get_summary()
        delta_usage = {
            "prompt_tokens": max(0, int(tok_after.get("total_prompt_tokens", 0)) - int(tok_before.get("total_prompt_tokens", 0))),
            "cached_tokens": max(0, int(tok_after.get("total_cached_tokens", 0)) - int(tok_before.get("total_cached_tokens", 0))),
            "output_tokens": max(0, int(tok_after.get("total_output_tokens", 0)) - int(tok_before.get("total_output_tokens", 0))),
            "thought_tokens": max(0, int(tok_after.get("total_thought_tokens", 0)) - int(tok_before.get("total_thought_tokens", 0))),
            "total_tokens": max(0, int(tok_after.get("total_tokens", 0)) - int(tok_before.get("total_tokens", 0))),
        }

        parsed = getattr(res, "parsed", None)
        raw_text = getattr(res, "text", None) or ""
        if parsed is None:
            if not raw_text.strip():
                raise StudioAgentError(f"CoProducerAgent ({label}) received empty response from model '{self.model_name}'.")
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if len(lines) >= 2 and lines[-1].strip().startswith("```"):
                    cleaned = "\n".join(lines[1:-1]).strip()
            parsed_obj = schema_cls.model_validate_json(cleaned)
        elif isinstance(parsed, schema_cls):
            parsed_obj = parsed
        elif isinstance(parsed, dict):
            parsed_obj = schema_cls.model_validate(parsed)
        else:
            parsed_obj = schema_cls.model_validate_json(raw_text)

        if prompt_trace_sink is not None:
            prompt_trace_sink.append(
                _ordered_rationale_dict(
                    parsed_obj.deliberation,
                    step_label=label,
                    model=self.model_name,
                    temperature=temperature,
                    attached_image_parts=sum(1 for p in (multimodal_parts or []) if not isinstance(p, str)),
                    prompt_excerpt=prompt[:4000],
                    response_json=parsed_obj.model_dump(),
                    token_usage=delta_usage,
                )
            )
        return parsed_obj, delta_usage

    def _resolve_active_genre_foundation(self, thread_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        brief = thread_meta.get("concept_brief")
        if isinstance(brief, dict) and isinstance(brief.get("genre_craft_foundation"), dict) and brief.get("genre_craft_foundation"):
            return dict(brief["genre_craft_foundation"])
        scope = str(thread_meta.get("scope_kind") or "cartridge")
        cid = str(thread_meta.get("cartridge_id") or "").strip()
        if scope == "cartridge" and cid:
            c_dir = self.cartridges_root / cid
            gb_path = c_dir / "game_bible.json"
            if gb_path.is_file():
                try:
                    gb = json.loads(gb_path.read_text(encoding="utf-8"))
                    if isinstance(gb, dict) and isinstance(gb.get("genre_craft_foundation"), dict) and gb.get("genre_craft_foundation"):
                        return dict(gb["genre_craft_foundation"])
                except Exception:
                    pass
            art_gcf = c_dir / "artifacts" / "genre_craft_foundation.json"
            if art_gcf.is_file():
                try:
                    loaded = json.loads(art_gcf.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict) and loaded:
                        return loaded
                except Exception:
                    pass
        return None

    def _build_lightweight_context_header(self, thread_meta: Dict[str, Any]) -> str:
        from engine.studio.context_curator import format_genre_craft_foundation_briefing

        scope = str(thread_meta.get("scope_kind") or "cartridge")
        cid = str(thread_meta.get("cartridge_id") or "").strip()
        lines = [
            f"THREAD_ID: {thread_meta.get('thread_id')}",
            f"SCOPE: {scope}" + (f" (cartridge_id={cid})" if cid else " (pre-genesis draft concept sandbox)"),
            f"THREAD_TITLE: {thread_meta.get('title', 'New Game Concept')}",
        ]
        brief = thread_meta.get("concept_brief")
        if isinstance(brief, dict) and any(
            brief.get(k)
            for k in (
                "working_title",
                "core_genre",
                "elevator_pitch",
                "core_player_verbs",
                "user_confirmed_pillars",
                "explicit_non_goals",
                "open_design_questions",
                "genre_craft_foundation",
            )
        ):
            confirmed_pillars = [str(x).strip() for x in (brief.get("user_confirmed_pillars") or []) if str(x).strip()]
            non_goals = [str(x).strip() for x in (brief.get("explicit_non_goals") or []) if str(x).strip()]
            open_qs = [str(x).strip() for x in (brief.get("open_design_questions") or []) if str(x).strip()]
            if confirmed_pillars:
                lines.append(
                    "USER_CONFIRMED_PILLARS_LEDGER (MUST PRESERVE CUMULATIVELY UNLESS USER RETRACTS):\n- "
                    + "\n- ".join(confirmed_pillars)
                )
            if non_goals:
                lines.append(
                    "EXPLICIT_NON_GOALS_AND_VISUAL_EXCLUSIONS (STRICT HARD CONSTRAINTS):\n- "
                    + "\n- ".join(non_goals)
                )
            if open_qs:
                lines.append(
                    "OPEN_DESIGN_QUESTIONS:\n- "
                    + "\n- ".join(open_qs)
                )
            brief_preview = {k: v for k, v in brief.items() if k != "genre_craft_foundation"}
            if brief_preview:
                lines.append(f"EVOLVING_CONCEPT_BRIEF_JSON:\n{json.dumps(brief_preview, indent=2, default=str)[:4500]}")

        active_gcf = self._resolve_active_genre_foundation(thread_meta)
        if active_gcf:
            lines.append(
                f"GENRE_CRAFT_FOUNDATION:\n{format_genre_craft_foundation_briefing(active_gcf)}"
            )
        else:
            lines.append(
                "GENRE_CRAFT_FOUNDATION: (not yet synthesized — call `curate_genre_fun_foundation` when exploring, "
                "pitching, critiquing, or pivoting a game genre so your creative advice and Council consultations are "
                "grounded in what genuinely makes that genre fun!)"
            )

        if scope == "cartridge" and cid:
            c_dir = self.cartridges_root / cid
            gb_path = c_dir / "game_bible.json"
            lines.append(f"HAS_PERSISTED_GAME_BIBLE: {gb_path.is_file()}")
            if gb_path.is_file():
                try:
                    gb = json.loads(gb_path.read_text(encoding="utf-8"))
                    if isinstance(gb, dict):
                        lines.append(
                            f"CARTRIDGE_BIBLE_SNAPSHOT: title={gb.get('game_title')!r}, "
                            f"genre={gb.get('core_genre')!r}, status={gb.get('status')!r}, "
                            f"active_slice_index={gb.get('active_slice_index', 0)}, "
                            f"slices={len(gb.get('slice_roadmap') or [])}"
                        )
                except Exception:
                    pass
            bl_path = c_dir / "backlog.json"
            if bl_path.is_file():
                try:
                    from engine.studio.production_board import ProductionBoard

                    board = ProductionBoard(path=bl_path, read_only=True)
                    counts: Dict[str, int] = {}
                    for it in board.items():
                        counts[it.state] = counts.get(it.state, 0) + 1
                    lines.append(f"PRODUCTION_BOARD_COUNTS: total={len(board.items())}, by_state={counts}")
                except Exception:
                    pass
            paused_file = c_dir / ".studio_paused.json"
            global_paused = self.cartridges_root / ".studio_paused.json"
            is_paused = False
            for pf in (paused_file, global_paused):
                if pf.is_file():
                    try:
                        if json.loads(pf.read_text(encoding="utf-8")).get("paused") is True:
                            is_paused = True
                    except Exception:
                        pass
            lines.append(f"BUILD_PAUSE_FLAG_ON_DISK: {is_paused}")
        return "\n".join(lines)

    @staticmethod
    def _format_conversation_history(messages: Sequence[Dict[str, Any]], max_turns: int = 16) -> str:
        recent = list(messages)[-max_turns:]
        if not recent:
            return "(Start of conversation)"
        blocks: List[str] = []
        for m in recent:
            role = str(m.get("sender_role") or "user").upper()
            txt = str(m.get("content_markdown") or "").strip()
            cards_note = ""
            m_cards = m.get("media_cards") or []
            c_cards = m.get("consultation_cards") or []
            a_props = m.get("action_proposals") or []
            extras: List[str] = []
            if m_cards:
                extras.append(f"media_cards={[c.get('title', c.get('card_kind')) for c in m_cards if isinstance(c, dict)]}")
            if c_cards:
                extras.append(f"consultations={[c.get('synthesis_headline', '')[:80] for c in c_cards if isinstance(c, dict)]}")
            if a_props:
                act_labels = [
                    f"{p.get('action_kind')}:{p.get('status', 'pending')}"
                    for p in a_props
                    if isinstance(p, dict)
                ]
                extras.append(f"actions={act_labels}")
            if extras:
                cards_note = f"\n  [Attached Cards: {'; '.join(extras)}]"
            blocks.append(f"[{role}]: {txt[:1800]}{cards_note}")
        return "\n\n".join(blocks)

    # -------------------------------------------------------------------------
    # Tool Implementations (Safe Read-Only Inspection, Consultations & Sketches)
    # -------------------------------------------------------------------------

    def _tool_inspect_studio_progress(
        self,
        call: CoProducerToolCall,
        thread_meta: Dict[str, Any],
        multimodal_parts: List[Any],
        staged_media_cards: List[CoProducerMediaCard],
    ) -> str:
        cid = str(thread_meta.get("cartridge_id") or "").strip()
        if not cid:
            return (
                "Currently in a pre-genesis Draft Concept thread (no cartridge build has been commissioned yet). "
                f"Session token usage so far: {json.dumps(TOKEN_USAGE_TRACKER.get_summary())}"
            )
        c_dir = self.cartridges_root / cid
        if not c_dir.is_dir():
            return f"Cartridge directory '{cid}' does not exist on disk yet."

        report_lines = [f"=== STUDIO PROGRESS FOR `{cid}` ==="]
        gb_path = c_dir / "game_bible.json"
        active_slice_id = ""
        if gb_path.is_file():
            try:
                gb = json.loads(gb_path.read_text(encoding="utf-8"))
                slices = gb.get("slice_roadmap") or []
                act_idx = int(gb.get("active_slice_index") or 0)
                if 0 <= act_idx < len(slices) and isinstance(slices[act_idx], dict):
                    active_slice_id = str(slices[act_idx].get("slice_id") or "")
                report_lines.append(
                    f"Game Title: {gb.get('game_title')} | Genre: {gb.get('core_genre')} | Bible Status: {gb.get('status')}"
                )
                if slices:
                    slice_summ = [
                        f"[{idx}{'*' if idx == act_idx else ''}] {s.get('slice_id')}: {s.get('title')} ({s.get('status')})"
                        for idx, s in enumerate(slices)
                        if isinstance(s, dict)
                    ]
                    report_lines.append("Slice Roadmap: " + " | ".join(slice_summ))
                utts = gb.get("writers_room_transcript") or []
                if utts:
                    tail_utts = [
                        f"  - {u.get('speaker_role')}: {str(u.get('spoken_dialogue') or '')[:220]}"
                        for u in utts[-4:]
                        if isinstance(u, dict)
                    ]
                    report_lines.append("Recent Writers' Room Activity:\n" + "\n".join(tail_utts))
            except Exception as exc:
                report_lines.append(f"Warning reading game_bible.json: {exc}")

        bl_path = c_dir / "backlog.json"
        metric_items: List[CoProducerMediaItem] = []
        recent_proof_paths: List[Path] = []
        if bl_path.is_file():
            try:
                from engine.studio.production_board import ProductionBoard

                board = ProductionBoard(path=bl_path, read_only=True)
                items = board.items()
                counts: Dict[str, int] = {}
                for it in items:
                    counts[it.state] = counts.get(it.state, 0) + 1
                approved_cnt = counts.get("approved", 0)
                working_cnt = counts.get("in_progress", 0) + counts.get("awaiting_review", 0)
                queued_cnt = counts.get("ready", 0) + counts.get("blocked", 0) + counts.get("changes_requested", 0)
                decision_cnt = counts.get("needs_decision", 0) + counts.get("failed", 0)
                report_lines.append(
                    f"ProductionBoard Total={len(items)} | Approved={approved_cnt} | Active/Review={working_cnt} "
                    f"| Queued={queued_cnt} | NeedsDecision/Failed={decision_cnt} | Breakdown={counts}"
                )
                metric_items.extend(
                    [
                        CoProducerMediaItem(
                            deliberation="Approved task count metric for progress card.",
                            item_id="m_approved",
                            media_kind="metric",
                            label="Approved Tasks",
                            metric_value=f"{approved_cnt} / {len(items)}",
                        ),
                        CoProducerMediaItem(
                            deliberation="In-flight task count metric.",
                            item_id="m_working",
                            media_kind="metric",
                            label="Working / In Review",
                            metric_value=str(working_cnt),
                        ),
                        CoProducerMediaItem(
                            deliberation="Queued task count metric.",
                            item_id="m_queued",
                            media_kind="metric",
                            label="Queued / Blocked",
                            metric_value=str(queued_cnt),
                        ),
                    ]
                )
                item_summaries: List[str] = []
                for it in items:
                    if it.state in ("in_progress", "awaiting_review", "needs_decision", "failed", "changes_requested", "ready"):
                        item_summaries.append(
                            f"  - [{it.item_id}] ({it.state}, kind={it.kind}, slice={it.slice_id}, deps={it.depends_on}): {it.title}"
                            + (f" | reason={it.decision_reason}" if it.decision_reason else "")
                        )
                    if it.revisions:
                        for p_str in it.revisions[-1].proof_images[-2:]:
                            cand_p = Path(p_str) if Path(p_str).is_absolute() else (c_dir / p_str)
                            if cand_p.is_file():
                                recent_proof_paths.append(cand_p)
                if item_summaries:
                    report_lines.append("Non-Terminal Board Items:\n" + "\n".join(item_summaries[:25]))
            except Exception as exc:
                report_lines.append(f"Warning reading backlog.json: {exc}")

        # Check QA audiovisual/filmstrip captures in artifacts/
        art_dir = c_dir / "artifacts"
        qa_videos: List[Path] = []
        if art_dir.is_dir():
            qa_videos = sorted(art_dir.glob("audiovisual_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)[:2]
            filmstrips = sorted(art_dir.glob("filmstrip_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:2]
            for fs in filmstrips:
                if fs not in recent_proof_paths:
                    recent_proof_paths.append(fs)
            if qa_videos:
                report_lines.append(
                    "Recent QA Audiovisual Captures: "
                    + ", ".join(self._artifact_rel_for_thread(v, thread_meta) for v in qa_videos)
                )

        for p_img in recent_proof_paths[-2:]:
            part = self._load_png_part(p_img)
            if part is not None:
                multimodal_parts.append(f"RECENT STUDIO PROOF IMAGE (`{p_img.name}`):")
                multimodal_parts.append(part)

        tok = TOKEN_USAGE_TRACKER.get_summary()
        metric_items.append(
            CoProducerMediaItem(
                deliberation="Session token & estimated cost telemetry.",
                item_id="m_cost",
                media_kind="metric",
                label="Session AI Calls",
                metric_value=f"{tok.get('total_calls', 0)} calls (${tok.get('estimated_cost_usd', 0.0):.3f})",
            )
        )
        if metric_items:
            staged_media_cards.append(
                CoProducerMediaCard(
                    deliberation=f"Curated live ProductionBoard and slice progress summary for '{cid}'.",
                    card_id=f"prog_{uuid.uuid4().hex[:8]}",
                    card_kind="progress_summary",
                    title=f"Build Progress — {cid}",
                    subtitle=f"Active slice: {active_slice_id or 'initial'}",
                    items=metric_items,
                    deep_link_workspace="board",
                    deep_link_target_id=active_slice_id,
                )
            )
        if qa_videos:
            v_file = qa_videos[0]
            rel_v = self._artifact_rel_for_thread(v_file, thread_meta)
            staged_media_cards.append(
                CoProducerMediaCard(
                    deliberation="Attached latest QA audiovisual verification clip for direct showrunner review.",
                    card_id=f"qavid_{uuid.uuid4().hex[:8]}",
                    card_kind="qa_clip",
                    title="Latest QA Audiovisual Capture",
                    subtitle=v_file.name,
                    items=[
                        CoProducerMediaItem(
                            deliberation="QA audiovisual MP4 capture.",
                            item_id=v_file.stem,
                            media_kind="video",
                            label=v_file.name,
                            artifact_path=rel_v,
                            mime_type="video/mp4",
                            width=1280,
                            height=720,
                        )
                    ],
                    deep_link_workspace="board",
                )
            )
        report_lines.append(
            f"Token Telemetry: calls={tok.get('total_calls', 0)}, total_tokens={tok.get('total_tokens', 0)}, "
            f"est_cost_usd=${tok.get('estimated_cost_usd', 0.0):.4f}"
        )
        return "\n".join(report_lines)

    def _tool_inspect_game_bible_and_dna(
        self,
        call: CoProducerToolCall,
        thread_meta: Dict[str, Any],
        multimodal_parts: List[Any],
    ) -> str:
        focus = call.bible_or_dna_focus or "both"
        out_sections: List[str] = []
        cid = str(thread_meta.get("cartridge_id") or "").strip()

        if focus in ("overview", "full_bible", "both") and cid:
            c_dir = self.cartridges_root / cid
            gb_path = c_dir / "game_bible.json"
            if gb_path.is_file():
                try:
                    gb = GameBibleSpec.model_validate_json(gb_path.read_text(encoding="utf-8"))
                    from engine.studio.context_curator import format_genre_craft_foundation_briefing

                    gcf_block = (
                        f"\nGenre Craft & Fun Foundation:\n{format_genre_craft_foundation_briefing(gb.genre_craft_foundation)}"
                        if gb.genre_craft_foundation is not None
                        else ""
                    )
                    out_sections.append(
                        f"=== LIVING GAME BIBLE (`{cid}`) ===\n"
                        f"Title: {gb.game_title}\n"
                        f"Genre: {gb.core_genre}\n"
                        f"Status: {gb.status}\n"
                        f"Elevator Pitch: {gb.elevator_pitch}\n"
                        f"Core Fantasy: {gb.core_fantasy}\n"
                        f"First 60s Hook: {gb.first_60_seconds_hook}\n"
                        f"Gameplay Pillars: {gb.gameplay_pillars}\n"
                        f"Camera & Spatial Paradigm: {gb.camera_and_spatial_paradigm}\n"
                        f"Visual Style Guide: {gb.visual_style_guide[:900]}\n"
                        f"Audio Direction Guide: {gb.audio_direction_guide[:900]}\n"
                        f"Color Palette: {gb.color_palette_hex}\n"
                        f"Visual Mockups: {[m.image_path for m in gb.visual_mockups]}"
                        f"{gcf_block}"
                    )
                    if focus == "full_bible":
                        out_sections.append(
                            f"Slice Roadmap JSON:\n{json.dumps([s.model_dump() for s in gb.slice_roadmap], indent=2, default=str)[:3500]}"
                        )
                    for m in gb.visual_mockups[-2:]:
                        mp = c_dir / m.image_path.lstrip("/")
                        part = self._load_png_part(mp)
                        if part is not None:
                            multimodal_parts.append(f"GAME BIBLE 16:9 CONCEPT MOCKUP (`{m.image_path}`):")
                            multimodal_parts.append(part)
                except Exception as exc:
                    out_sections.append(f"Error parsing game_bible.json for `{cid}`: {exc}")
            else:
                hb_path = c_dir / HANDOFF_BRIEF_FILENAME
                if hb_path.is_file():
                    try:
                        hb = json.loads(hb_path.read_text(encoding="utf-8"))
                        cb = hb.get("concept_brief") if isinstance(hb.get("concept_brief"), dict) else hb
                        out_sections.append(
                            f"=== PRE-BIBLE COMMISSIONED CONCEPT BRIEF (`{cid}/{HANDOFF_BRIEF_FILENAME}`) ===\n"
                            f"Note: `game_bible.json` has not been synthesized by the Writers' Room yet; this cartridge is still in pre-bible refinement.\n"
                            f"Handoff Mode: {hb.get('handoff_mode', 'refined_concept')}\n"
                            f"Working Title: {cb.get('working_title', '')}\n"
                            f"Core Genre: {cb.get('core_genre', '')}\n"
                            f"Elevator Pitch: {cb.get('elevator_pitch', '')}\n"
                            f"Core Fantasy: {cb.get('core_fantasy', '')}\n"
                            f"Player Verbs: {cb.get('core_player_verbs', [])}\n"
                            f"User-Confirmed Pillars: {cb.get('user_confirmed_pillars', [])}\n"
                            f"Visual Style Direction: {cb.get('visual_style_direction', '')}\n"
                            f"Audio Mood Direction: {cb.get('audio_mood_direction', '')}\n"
                            f"Explicit Non-Goals: {cb.get('explicit_non_goals', [])}\n"
                            f"Open Design Questions: {cb.get('open_design_questions', [])}\n"
                            f"Selected Mockups: {cb.get('selected_mockup_paths', [])}"
                        )
                        for raw_mp in list(cb.get("selected_mockup_paths") or [])[-2:]:
                            mp_clean = str(raw_mp or "").lstrip("/")
                            mp = (c_dir / mp_clean) if not mp_clean.startswith(f"{cid}/") else (self.cartridges_root / mp_clean)
                            part = self._load_png_part(mp)
                            if part is not None:
                                multimodal_parts.append(f"PRE-BIBLE CONCEPT MOCKUP (`{raw_mp}`):")
                                multimodal_parts.append(part)
                    except Exception as hb_exc:
                        out_sections.append(f"No `game_bible.json` exists yet for `{cid}` (error reading `{HANDOFF_BRIEF_FILENAME}`: {hb_exc}).")
                else:
                    out_sections.append(f"No `game_bible.json` exists yet for `{cid}`.")

        if focus in ("player_dna", "both"):
            dna_loaded = None
            if self.db is not None and getattr(self.db, "db_path", None):
                try:
                    from engine.steam.dna import SteamDNAPersistence

                    repo = SteamDNAPersistence(Path(self.db.db_path))
                    dna_loaded = repo.get_latest_dna()
                except Exception as exc:
                    out_sections.append(f"Note reading Player DNA: {exc}")
            if dna_loaded is not None:
                top_g = [
                    f"{g.get('name', '?')} ({g.get('playtime_hours', 0)}h)"
                    if isinstance(g, dict)
                    else f"{getattr(g, 'name', '?')} ({getattr(g, 'playtime_hours', 0)}h)"
                    for g in (getattr(dna_loaded, "top_games", None) or [])[:8]
                ]
                out_sections.append(
                    f"=== BACKGROUND PLAYER TASTE DNA (OPTIONAL INSPIRATION ONLY — SUBORDINATE TO EXPLICIT USER DIRECTIVES) ===\n"
                    f"Top Genres: {dna_loaded.top_genres}\n"
                    f"Thematic Biases: {dna_loaded.thematic_biases}\n"
                    f"Mechanical Biases: {dna_loaded.mechanical_biases}\n"
                    f"Pacing Preference: {dna_loaded.pacing_preference} (score={dna_loaded.pacing_score})\n"
                    f"Challenge Tolerance: {dna_loaded.challenge_tolerance}\n"
                    f"Preferred Palette Mood: {dna_loaded.preferred_palette_mood}\n"
                    f"Most-Played Games: {', '.join(top_g)}"
                )
            else:
                out_sections.append("No Steam Player DNA profile is currently stored in TelemetryDB.")

        return "\n\n".join(out_sections) if out_sections else "No Game Bible or Player DNA records found."

    def _tool_inspect_playtest_sessions(
        self,
        call: CoProducerToolCall,
        thread_meta: Dict[str, Any],
        multimodal_parts: List[Any],
        staged_media_cards: List[CoProducerMediaCard],
    ) -> str:
        if self.db is None or not hasattr(self.db, "list_playtest_sessions"):
            return "TelemetryDB is not attached; playtest sessions cannot be queried."

        cid = str(thread_meta.get("cartridge_id") or "").strip()
        target_sid = (call.playtest_session_id or "").strip()
        sessions: List[Dict[str, Any]] = []
        if target_sid:
            one = self.db.get_playtest_session(target_sid)
            if one is not None:
                sessions = [one]
        if not sessions:
            if cid:
                sessions = self.db.list_playtest_sessions(candidate_id=cid, limit=10)
            if not sessions:
                sessions = self.db.list_playtest_sessions(limit=5)

        if not sessions:
            cid_clause = f" for `{cid}`" if cid else ""
            return f"No Blind Playtester sessions have been recorded yet{cid_clause}."

        primary = sessions[0]
        sid = str(primary.get("session_id") or "")
        turns = self.db.list_playtest_turns(sid) if sid else []
        wishes = self.db.list_playtest_wishes(session_id=sid) if (sid and hasattr(self.db, "list_playtest_wishes")) else []
        rep = primary.get("session_report") if isinstance(primary.get("session_report"), dict) else {}

        video_api_url = f"/api/playtest/sessions/{sid}/video"
        video_disk_path = Path(str(primary.get("video_path") or "")) if primary.get("video_path") else None
        has_video = bool(video_disk_path and video_disk_path.is_file()) or primary.get("status") in ("completed", "stopped", "running")

        lines = [
            f"=== BLIND PLAYTEST SESSION `{sid}` ===",
            f"Game: {primary.get('game_title')} | Candidate: {primary.get('candidate_id')} | Slice: {primary.get('slice_id')}",
            f"Status: {primary.get('status')} | Turns: {primary.get('turn_count', len(turns))} | Sim Duration: {primary.get('sim_duration_sec', 0)}s",
            f"Replay Video URL: {video_api_url}",
        ]
        if rep:
            lines.append(
                f"Overall Fun Rating: {rep.get('overall_fun_rating', 'n/a')}/10\n"
                f"Experience Summary: {rep.get('overall_experience_summary', '')}\n"
                f"Standout Moments: {rep.get('standout_moments', [])}\n"
                f"Top Frustrations: {rep.get('biggest_frustrations', [])}"
            )
        if primary.get("bugs_found"):
            lines.append(f"Bugs Reported: {json.dumps(primary.get('bugs_found'), default=str)[:1200]}")
        if wishes:
            lines.append(
                "Player Wishes:\n"
                + "\n".join(
                    f"  - [{w.get('status')}] ({w.get('category')}): {w.get('summary')} — {w.get('rationale', '')[:160]}"
                    for w in wishes[:8]
                )
            )

        # Sample up to 3 evenly spaced turn screenshots for multimodal grounding & gallery
        turn_screens: List[Tuple[Dict[str, Any], Path]] = []
        for t in turns:
            sp_raw = str(t.get("screenshot_path") or "").strip()
            if sp_raw and Path(sp_raw).is_file():
                turn_screens.append((t, Path(sp_raw)))
        sampled_screens: List[Tuple[Dict[str, Any], Path]] = []
        if len(turn_screens) <= 3:
            sampled_screens = turn_screens
        else:
            idxs = [0, len(turn_screens) // 2, len(turn_screens) - 1]
            sampled_screens = [turn_screens[i] for i in idxs]

        screenshot_items: List[CoProducerMediaItem] = []
        for t_dict, s_path in sampled_screens:
            part = self._load_png_part(s_path)
            if part is not None:
                multimodal_parts.append(
                    f"PLAYTEST `{sid}` TURN #{t_dict.get('turn_index')} SCREENSHOT (mood={t_dict.get('mood_state')}):"
                )
                multimodal_parts.append(part)
            w_px, h_px = self._image_dimensions(s_path)
            screenshot_items.append(
                CoProducerMediaItem(
                    deliberation=f"Playtest turn #{t_dict.get('turn_index')} frame capture.",
                    item_id=f"{sid}_t{t_dict.get('turn_index')}",
                    media_kind="image",
                    label=f"Turn {t_dict.get('turn_index')} ({t_dict.get('mood_state') or 'observing'})",
                    caption=str(t_dict.get("append_log_line") or "")[:180],
                    artifact_path=f"/api/playtest/sessions/{sid}/media/{s_path.name}",
                    mime_type="image/png",
                    width=w_px or 1280,
                    height=h_px or 720,
                )
            )

        # Optional perceptual evaluation via VisualEvaluatorAgent / AudioEvaluatorAgent
        if call.evaluate_media_perceptually and self.client is not None:
            c_dir = self.cartridges_root / (cid or str(primary.get("candidate_id") or "active"))
            if sampled_screens and c_dir.is_dir():
                try:
                    from engine.studio.visual_evaluator import VisualEvaluatorAgent

                    ve = VisualEvaluatorAgent(client=self.client, model_name=self.model_name)
                    v_rep, _, _ = ve.evaluate(
                        cartridge_dir=c_dir,
                        asset_paths=[p for _, p in sampled_screens[:2]],
                        inquiry_prompt=call.query_or_question or "Evaluate visual readability, layout clarity, and feedback cues in these playtest frames.",
                        enrich_catalog=False,
                    )
                    lines.append(f"Perceptual Visual Evaluator Findings: {v_rep.model_dump_json()[:1500]}")
                except Exception as v_exc:
                    lines.append(f"VisualEvaluatorAgent note: {v_exc}")
            if video_disk_path and video_disk_path.is_file() and c_dir.is_dir():
                try:
                    from engine.studio.audio_evaluator import AudioEvaluatorAgent

                    ae = AudioEvaluatorAgent(client=self.client, model_name=self.model_name)
                    a_rep = ae.evaluate(
                        cartridge_dir=c_dir,
                        clip_paths=[video_disk_path],
                        inquiry_prompt=call.query_or_question or "Evaluate audiovisual sync, SFX clarity, and sonic mood in this playtest recording.",
                    )
                    lines.append(f"Perceptual Audio/Video Evaluator Findings: {a_rep.model_dump_json()[:1500]}")
                except Exception as a_exc:
                    lines.append(f"AudioEvaluatorAgent note: {a_exc}")

        if sid and has_video:
            vid_items = [
                CoProducerMediaItem(
                    deliberation=f"Playtest broadcast video player for session '{sid}'.",
                    item_id=f"{sid}_video",
                    media_kind="video",
                    label=f"Playtest Replay — {primary.get('game_title') or sid}",
                    caption=(
                        f"{primary.get('turn_count', len(turns))} turns · "
                        + (f"Fun {rep.get('overall_fun_rating')}/10" if rep.get("overall_fun_rating") is not None else str(primary.get("status")))
                    ),
                    artifact_path=video_api_url,
                    mime_type="video/mp4",
                    width=1600,
                    height=720,
                ),
                *screenshot_items,
            ]
            staged_media_cards.append(
                CoProducerMediaCard(
                    deliberation=f"Surfacing playtest broadcast video and key turn frames for session '{sid}'.",
                    card_id=f"ptcard_{sid}",
                    card_kind="playtest_video",
                    title=f"Blind Playtest — {primary.get('game_title') or sid}",
                    subtitle=f"Session {sid} · {primary.get('status')}",
                    items=vid_items,
                    deep_link_workspace="playtest",
                    deep_link_target_id=sid,
                )
            )
        return "\n".join(lines)

    def _tool_browse_cartridge_media_and_code(
        self,
        call: CoProducerToolCall,
        thread_meta: Dict[str, Any],
        multimodal_parts: List[Any],
        staged_media_cards: List[CoProducerMediaCard],
    ) -> str:
        t_dir = self.store.resolve_thread_dir(thread_meta)
        if not t_dir.is_dir():
            return f"Directory `{t_dir.name}` does not exist on disk yet."

        flt = call.media_or_code_filter or "all"
        lines = [f"=== MEDIA, CODE & MEMORY INVENTORY (`{t_dir.name}`, filter={flt}) ==="]

        # 1. Sprites, mockups & sketches
        if flt in ("all", "sprites", "mockups"):
            img_files: List[Path] = []
            for sub_rel in ("assets/mockups", "assets/sprites", "assets/portraits", "assets/sketches"):
                if flt == "mockups" and sub_rel not in ("assets/mockups", "assets/sketches"):
                    continue
                if flt == "sprites" and sub_rel not in ("assets/sprites", "assets/portraits", "assets/sketches"):
                    continue
                sub_p = t_dir / sub_rel
                if sub_p.is_dir():
                    img_files.extend(
                        sorted(
                            [p for p in sub_p.glob("*.png") if p.is_file()],
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )[:8]
                    )
            if img_files:
                rel_imgs = [self._artifact_rel_for_thread(p, thread_meta) for p in img_files[:8]]
                lines.append(f"Visual Assets ({len(img_files)} recent): {rel_imgs}")
                gallery_items: List[CoProducerMediaItem] = []
                for p in img_files[:6]:
                    w_px, h_px = self._image_dimensions(p)
                    rel_p = self._artifact_rel_for_thread(p, thread_meta)
                    gallery_items.append(
                        CoProducerMediaItem(
                            deliberation=f"Visual artifact '{p.name}' selected for showrunner gallery.",
                            item_id=p.stem,
                            media_kind="image",
                            label=p.name,
                            caption=p.parent.name,
                            artifact_path=rel_p,
                            mime_type="image/png",
                            width=w_px or 512,
                            height=h_px or 512,
                        )
                    )
                for p in img_files[:3]:
                    part = self._load_png_part(p)
                    if part is not None:
                        multimodal_parts.append(f"CARTRIDGE VISUAL ARTIFACT (`{p.name}`):")
                        multimodal_parts.append(part)
                staged_media_cards.append(
                    CoProducerMediaCard(
                        deliberation="Curated visual asset & mockup gallery card.",
                        card_id=f"gal_{uuid.uuid4().hex[:8]}",
                        card_kind="image_gallery",
                        title="Cartridge Visual Assets & Mockups",
                        subtitle=f"{len(gallery_items)} artifact(s) from {t_dir.name}",
                        items=gallery_items,
                        deep_link_workspace="assets" if thread_meta.get("scope_kind") == "cartridge" else None,
                    )
                )

        # 2. Audio clips (.wav)
        if flt in ("all", "audio"):
            wav_files: List[Path] = []
            for sub_rel in ("assets/audio", "assets/sketches"):
                sub_p = t_dir / sub_rel
                if sub_p.is_dir():
                    wav_files.extend(
                        sorted(
                            [p for p in sub_p.glob("*.wav") if p.is_file()],
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )[:6]
                    )
            if wav_files:
                rel_wavs = [self._artifact_rel_for_thread(p, thread_meta) for p in wav_files[:6]]
                lines.append(f"Audio Clips ({len(wav_files)} recent): {rel_wavs}")
                staged_media_cards.append(
                    CoProducerMediaCard(
                        deliberation="Curated synthesized audio cues and voice/SFX clips.",
                        card_id=f"aud_{uuid.uuid4().hex[:8]}",
                        card_kind="audio_clips",
                        title="Synthesized Audio & Voice Cues",
                        subtitle=f"{len(wav_files[:6])} WAV clip(s)",
                        items=[
                            CoProducerMediaItem(
                                deliberation=f"Audio clip '{w.name}'.",
                                item_id=w.stem,
                                media_kind="audio",
                                label=w.name,
                                caption=f"{round(w.stat().st_size / 1024, 1)} KB",
                                artifact_path=self._artifact_rel_for_thread(w, thread_meta),
                                mime_type="audio/wav",
                            )
                            for w in wav_files[:6]
                        ],
                        deep_link_workspace="assets" if thread_meta.get("scope_kind") == "cartridge" else None,
                    )
                )
                if call.evaluate_media_perceptually and self.client is not None:
                    try:
                        from engine.studio.audio_evaluator import AudioEvaluatorAgent

                        ae = AudioEvaluatorAgent(client=self.client, model_name=self.model_name)
                        a_rep = ae.evaluate(
                            cartridge_dir=t_dir,
                            clip_paths=wav_files[:2],
                            inquiry_prompt=call.query_or_question or "Evaluate sonic character, clarity, and polish of these audio clips.",
                        )
                        lines.append(f"AudioEvaluatorAgent Findings: {a_rep.model_dump_json()[:1400]}")
                    except Exception as a_exc:
                        lines.append(f"AudioEvaluatorAgent note: {a_exc}")

        # 3. Long-term Game Memories
        if flt in ("all", "memories") and (t_dir / "game_memory.json").is_file():
            try:
                from engine.studio.game_memory import GameMemoryStore

                m_store = GameMemoryStore(t_dir, db=self.db, client=self.client)
                entries = m_store.list_entries(status_filter="active")
                lines.append(
                    f"Active GameMemoryStore Entries ({len(entries)}):\n"
                    + "\n".join(
                        f"  - [{e.memory_id}] ({e.memory_bank}/{e.kind}): {e.title} -> {e.content[:180]}"
                        for e in entries[:15]
                    )
                )
            except Exception as m_exc:
                lines.append(f"GameMemoryStore note: {m_exc}")

        # 4. Recent Code Diffs & GDScript modules
        if flt in ("all", "code_diffs"):
            gd_files = sorted(
                [p.relative_to(t_dir).as_posix() for p in t_dir.rglob("*.gd") if "addons/" not in p.as_posix()]
            )
            if gd_files:
                lines.append(f"GDScript Modules ({len(gd_files)}): {gd_files[:25]}")
            bl_path = t_dir / "backlog.json"
            if bl_path.is_file():
                try:
                    from engine.studio.production_board import ProductionBoard

                    board = ProductionBoard(path=bl_path, read_only=True)
                    diff_items: List[CoProducerMediaItem] = []
                    for it in reversed(board.items()):
                        if it.revisions and it.revisions[-1].code_diffs:
                            for mod_name, diff_txt in list(it.revisions[-1].code_diffs.items())[:2]:
                                if str(diff_txt).strip():
                                    diff_items.append(
                                        CoProducerMediaItem(
                                            deliberation=f"Recent code diff on '{mod_name}' from task '{it.item_id}'.",
                                            item_id=f"{it.item_id}_{_safe_slug(mod_name)}",
                                            media_kind="code_diff",
                                            label=f"{mod_name} ({it.title})",
                                            caption=f"Task {it.item_id} · rev #{it.revisions[-1].attempt}",
                                            artifact_path=mod_name,
                                            diff_unified=str(diff_txt)[:4000],
                                        )
                                    )
                            if len(diff_items) >= 3:
                                break
                    if diff_items:
                        lines.append(f"Surfaced {len(diff_items)} recent code diff(s) from ProductionBoard revisions.")
                        staged_media_cards.append(
                            CoProducerMediaCard(
                                deliberation="Curated recent task code diffs for technical inspection.",
                                card_id=f"diff_{uuid.uuid4().hex[:8]}",
                                card_kind="code_diff",
                                title="Recent Code Changes",
                                subtitle=f"{len(diff_items)} module diff(s)",
                                items=diff_items,
                                deep_link_workspace="assets",
                            )
                        )
                except Exception as d_exc:
                    lines.append(f"Code diff inspection note: {d_exc}")

            if call.query_or_question.strip() and is_real_genai_client(self.client) and gd_files:
                try:
                    from engine.studio.context_curator import ContextCuratorAgent

                    curator = ContextCuratorAgent(
                        cartridge_dir=t_dir,
                        client=self.client,
                        model_name=self.model_name,
                    )
                    pack = curator.curate_context(
                        target_agent_role="CoProducerAgent",
                        task_request=call.query_or_question,
                        item_id="coproducer_code_query",
                        item_kind="discovery",
                    )
                    lines.append(f"Curated Code Context:\n{pack.prompt_block[:3500]}")
                except Exception as c_exc:
                    lines.append(f"ContextCurator note: {c_exc}")

        return "\n".join(lines)

    def _load_dna_summary_text(self) -> str:
        if self.db is not None and getattr(self.db, "db_path", None):
            try:
                from engine.steam.dna import SteamDNAPersistence
                from engine.studio.writers_room import CreativeWritersRoom

                dna_obj = SteamDNAPersistence(Path(self.db.db_path)).get_latest_dna()
                if dna_obj is not None:
                    room = CreativeWritersRoom(
                        client=self.client,
                        model_name=self.model_name,
                        board=None,
                        event_callback=self.event_callback,
                    )
                    return room._format_dna_summary(dna_obj, None)
            except Exception:
                return ""
        return ""

    def _tool_curate_genre_fun_foundation(
        self,
        call: CoProducerToolCall,
        thread_meta: Dict[str, Any],
        consultation_cards: List[CoProducerConsultationCard],
    ) -> str:
        from engine.studio.context_curator import (
            ContextCuratorAgent,
            format_genre_craft_foundation_briefing,
        )

        t_dir = self.store.resolve_thread_dir(thread_meta)
        brief = dict(thread_meta.get("concept_brief") or {})
        existing_gcf = self._resolve_active_genre_foundation(thread_meta)
        raw_q = (call.query_or_question or "").strip()
        genre_hint = (
            raw_q
            or str(brief.get("core_genre") or "").strip()
            or str(brief.get("working_title") or "").strip()
            or str(thread_meta.get("title") or "").strip()
            or "interactive 2D game"
        )
        seed_lines: List[str] = []
        if brief.get("working_title"):
            seed_lines.append(f"Working Title: {brief.get('working_title')}")
        if brief.get("core_genre"):
            seed_lines.append(f"Core Genre: {brief.get('core_genre')}")
        if brief.get("elevator_pitch"):
            seed_lines.append(f"Elevator Pitch: {brief.get('elevator_pitch')}")
        if brief.get("core_fantasy"):
            seed_lines.append(f"Core Fantasy: {brief.get('core_fantasy')}")
        if brief.get("core_player_verbs"):
            seed_lines.append(f"Player Verbs: {brief.get('core_player_verbs')}")
        if brief.get("explicit_non_goals"):
            seed_lines.append(f"Explicit Non-Goals: {brief.get('explicit_non_goals')}")
        if raw_q:
            seed_lines.append(f"Inquiry / Focus: {raw_q}")

        curator = ContextCuratorAgent(
            cartridge_dir=t_dir,
            client=self.client,
            model_name=self.model_name,
            event_callback=self.event_callback,
        )
        foundation = curator.synthesize_genre_craft_foundation(
            genre_hint=genre_hint,
            concept_or_seed="\n".join(seed_lines),
            player_dna_summary=self._load_dna_summary_text(),
            existing_foundation=existing_gcf,
            evolution_directive=raw_q if existing_gcf else "",
        )
        gcf_dict = foundation.model_dump()
        brief.setdefault("deliberation", foundation.deliberation)
        if not str(brief.get("core_genre") or "").strip():
            brief["core_genre"] = foundation.genre_and_subgenre_archetype
        brief["genre_craft_foundation"] = gcf_dict
        thread_meta["concept_brief"] = brief
        self.store.save_concept_brief(thread_meta, brief)

        formatted_md = format_genre_craft_foundation_briefing(foundation)
        consultation_cards.append(
            CoProducerConsultationCard(
                deliberation=foundation.deliberation,
                card_id=f"gcf_{uuid.uuid4().hex[:8]}",
                consultation_kind="genre_foundation",
                question_posed=raw_q or genre_hint,
                participating_roles=["ContextCurator"],
                synthesis_headline=f"Genre Craft & Fun Foundation — {foundation.genre_and_subgenre_archetype}",
                probe_findings_markdown=formatted_md,
            )
        )
        return f"GENRE CRAFT & FUN FOUNDATION CURATED:\n{formatted_md}"

    def _tool_consult_council(
        self,
        call: CoProducerToolCall,
        thread_meta: Dict[str, Any],
        multimodal_parts: List[Any],
        consultation_cards: List[CoProducerConsultationCard],
        *,
        latest_user_message: str = "",
        conversation_history: str = "",
    ) -> str:
        from engine.studio.writers_room import CreativeWritersRoom

        effective_question = (call.query_or_question or "").strip()
        if not effective_question:
            if (latest_user_message or "").strip():
                effective_question = (
                    "Address the Executive Producer's latest message directly while preserving all previously "
                    f"confirmed game pillars and explicit non-goals: {latest_user_message.strip()}"
                )
            else:
                effective_question = "Offer sharp, contrasting creative and systemic ideas."

        if is_real_genai_client(self.client) and self._resolve_active_genre_foundation(thread_meta) is None:
            try:
                auto_call = CoProducerToolCall(
                    deliberation=(
                        "Synthesizing first-principles Genre Craft & Fun Foundation via ContextCurator before convening "
                        "the Council so specialists have zero blind spots on what makes this genre genuinely fun."
                    ),
                    tool_name="curate_genre_fun_foundation",
                    query_or_question=effective_question,
                )
                self._tool_curate_genre_fun_foundation(auto_call, thread_meta, consultation_cards)
            except Exception as gcf_exc:
                logger.warning("Pre-consultation GenreCraftFoundation synthesis warning: %s", gcf_exc)

        room = CreativeWritersRoom(
            client=self.client,
            model_name=self.model_name,
            board=None,
            event_callback=self.event_callback,
        )
        dna_summary = self._load_dna_summary_text()

        ctx_parts = [self._build_lightweight_context_header(thread_meta)]
        if (conversation_history or "").strip():
            ctx_parts.append(f"RECENT CONVERSATION HISTORY:\n{conversation_history.strip()[-3500:]}")
        if (latest_user_message or "").strip():
            ctx_parts.append(
                f"LATEST EXECUTIVE PRODUCER MESSAGE (PRIMARY DIRECTIVE):\n{latest_user_message.strip()}"
            )
        ctx_summary = "\n\n".join(ctx_parts)
        card: CoProducerConsultationCard = room.run_advisory_consultation(
            question=effective_question,
            mode=call.council_mode or "huddle",
            specialist_roles=list(call.specialist_roles or []),
            context_summary=ctx_summary,
            dna_summary=dna_summary,
            multimodal_parts=multimodal_parts or None,
        )
        consultation_cards.append(card)
        return f"COUNCIL CONSULTATION COMPLETED:\n{card.model_dump_json(indent=2)}"

    def _tool_dispatch_specialist_probe(
        self,
        call: CoProducerToolCall,
        thread_meta: Dict[str, Any],
        multimodal_parts: List[Any],
        consultation_cards: List[CoProducerConsultationCard],
    ) -> str:
        from engine.studio.models import WorkItem
        from engine.studio.swe import LeadGameplaySWEAgent
        from engine.studio.tool_registry import normalize_specialist_persona
        from engine.studio.workers import (
            WorkerContext,
            run_audio_designer_specialist,
            run_qa_test_engineer_specialist,
            run_script_writer_specialist,
            run_swe_engineer_specialist,
            run_visual_artist_specialist,
        )

        t_dir = self.store.resolve_thread_dir(thread_meta)
        if not t_dir.is_dir():
            return f"Cannot run specialist probe: `{t_dir.name}` has no cartridge files on disk yet."

        persona = normalize_specialist_persona(call.probe_persona or "swe_engineer")
        brief = (call.query_or_question or "").strip() or "Inspect current cartridge implementation and assets."
        targets = list(call.target_paths or []) or None

        w_ctx = WorkerContext(
            cartridge_dir=t_dir,
            client=self.client,
            model_name=self.model_name,
            swe_agent=LeadGameplaySWEAgent(client=self.client, model_name=self.model_name)
            if persona == "swe_engineer"
            else None,
            board=None,
            event_callback=self.event_callback,
        )
        probe_item = WorkItem(
            deliberation=call.deliberation,
            item_id=f"coproducer_probe_{uuid.uuid4().hex[:6]}",
            kind="discovery",
            title=f"Co-Producer Read-Only Probe ({persona})",
            spec=brief,
            proof_spec=brief,
            target_modules=list(targets or ["main.gd"]),
        )
        if persona == "visual_artist":
            rep = run_visual_artist_specialist(
                w_ctx, probe_item, brief=brief, target_paths=targets, _read_only_probe=True
            )
        elif persona == "audio_designer":
            rep = run_audio_designer_specialist(
                w_ctx, probe_item, brief=brief, target_paths=targets, _force_operation="inspect"
            )
        elif persona == "script_writer":
            rep = run_script_writer_specialist(
                w_ctx, probe_item, brief=brief, target_paths=targets, _force_operation="inspect_only"
            )
        elif persona == "swe_engineer":
            rep = run_swe_engineer_specialist(
                w_ctx, probe_item, brief=brief, target_paths=targets, _inspect_only=True
            )
        else:
            rep = run_qa_test_engineer_specialist(
                w_ctx,
                probe_item,
                brief=f"[CO-PRODUCER READ-ONLY DIAGNOSTIC PROBE]:\n{brief}",
                target_paths=targets,
            )

        rel_proofs: List[str] = []
        for raw_p in (rep.proof_images or [])[-2:]:
            p_path = Path(raw_p) if Path(raw_p).is_absolute() else (t_dir / raw_p)
            if p_path.is_file():
                rel_proofs.append(self._artifact_rel_for_thread(p_path, thread_meta))
                part = self._load_png_part(p_path)
                if part is not None:
                    multimodal_parts.append(f"SPECIALIST PROBE PROOF (`{persona}` — `{p_path.name}`):")
                    multimodal_parts.append(part)

        card = CoProducerConsultationCard(
            deliberation=rep.deliberation,
            card_id=f"probe_{uuid.uuid4().hex[:8]}",
            consultation_kind="specialist_probe",
            question_posed=brief,
            participating_roles=[persona],
            synthesis_headline=f"Read-only `{persona}` probe: {rep.summary[:160]}",
            probe_findings_markdown=rep.summary,
            proof_media_paths=rel_proofs,
        )
        consultation_cards.append(card)
        return f"SPECIALIST PROBE (`{persona}`) COMPLETED:\n{rep.summary}\nProofs: {rel_proofs}"

    def _resolve_prior_sketch_reference_path(
        self,
        thread_meta: Dict[str, Any],
        sketches_dir: Path,
    ) -> Optional[Path]:
        """Resolves the most recent approved 16:9 mockup PNG when targeted reference-based refinement is requested."""
        t_dir = self.store.resolve_thread_dir(thread_meta)
        brief = dict(thread_meta.get("concept_brief") or {})
        for raw_rel in reversed(list(brief.get("selected_mockup_paths") or [])):
            clean_rel = str(raw_rel or "").strip().lstrip("/")
            if not clean_rel:
                continue
            for cand in (t_dir / clean_rel, self.cartridges_root / clean_rel):
                if cand.is_file():
                    return cand
        if sketches_dir.is_dir():
            existing = sorted(sketches_dir.glob("mockup_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            if existing:
                return existing[0]
        return None

    def _compile_concept_art_shot_plan(
        self,
        *,
        title: str,
        raw_scene_request: str,
        effective_style: str,
        neg_constraints: Sequence[str],
        thread_meta: Dict[str, Any],
        latest_user_message: str = "",
        conversation_history: str = "",
        prior_reference_path: Optional[Path] = None,
        candidate_dir: Optional[Path] = None,
        prompt_trace_sink: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[ConceptArtShotPlan, Dict[str, int]]:
        """Compiles raw creative ideation prose and user feedback into a spatially coherent, playable-scale
        `ConceptArtShotPlan` before invoking the image generation model."""
        brief = dict(thread_meta.get("concept_brief") or {})
        pillars = [str(p).strip() for p in (brief.get("user_confirmed_pillars") or []) if str(p).strip()]
        palette = [str(c).strip() for c in (brief.get("color_palette_hex") or []) if str(c).strip()]
        genre = str(brief.get("core_genre") or "").strip()
        pitch = str(brief.get("elevator_pitch") or "").strip()

        mm_parts: List[Any] = []
        if prior_reference_path is not None:
            ref_part = self._load_png_part(prior_reference_path)
            if ref_part is not None:
                mm_parts.append(f"PRIOR CONCEPT SKETCH REFERENCE (`{prior_reference_path.name}`):")
                mm_parts.append(ref_part)

        shot_prompt = (
            "You are the Principal Game Art Director & Cinematography Shot Compiler.\n"
            "Translate the design intent and user feedback below into a single coherent, playable-scale 16:9 in-engine "
            "gameplay viewport `ConceptArtShotPlan`.\n\n"
            "MANDATORY ART-DIRECTION INVARIANTS:\n"
            "1. SINGLE COHERENT CAMERA & VIEWPORT SCALE (RESOLVE MACRO-VS-MICRO PARADOXES):\n"
            "   - Lock ONE consistent camera projection across all playable entities and structures (e.g., pure 90-degree "
            "orthographic top-down 2D, 2D side cross-section, fixed-angle isometric, etc.).\n"
            "   - Whenever a small playable craft, character, or local facility operates inside a vast or colossal world, "
            "keep the camera locked to the LOCAL PLAYABLE VIEWPORT scale where the player unit (~3-7% of frame height) and "
            "multi-module gameplay structures (~20-40% of frame width) are crisp, readable, and proportionally scaled.\n"
            "   - Convey colossal world vastness via edge-to-edge environmental fields and deep background/underlayer parallax "
            "vistas sweeping across or beneath the frame — NEVER zoom out to a whole-world diagram/map with an oversized "
            "continent-scale icon pasted on top, and NEVER shrink a vast world into a tiny centered concentric bullseye or "
            "miniature toy diorama around the player.\n"
            "2. EXPLICIT RENDERING MEDIUM & SURFACE CRAFT:\n"
            "   - Specify the exact rendering medium, shader/material finish, linework, and lighting craft matching the game's "
            "direction (e.g., crisp high-definition 2D digital illustration with clean geometric silhouettes, shader-lit "
            "materials, emissive conduits, and multi-plane parallax; painterly 2D illustration; stylized cel-shaded 3D; or "
            "deliberate pixel art ONLY when pixel art was explicitly requested). Avoid vague 'indie game' phrasing that "
            "causes accidental retro pixel-art drift or simplified mobile-app clip art.\n"
            "3. FUNCTIONAL MULTI-PART GAMEPLAY ARCHITECTURE:\n"
            "   - Describe structures, machinery, terrain, and interactables as believable multi-component game systems with "
            "interconnected sub-modules, clear silhouettes, and active in-world visual state — NEVER a single flat "
            "simplified cross, badge, or logo emblem.\n"
            "4. PURE IN-WORLD VIEWPORT POLICY:\n"
            "   - Unless HUD widgets were explicitly requested for this shot, specify zero title text, zero poster typography, "
            "zero watermarks, zero floating text labels, and zero generic HUD chrome.\n\n"
            f"SHOT WORKING LABEL (INTERNAL ONLY — DO NOT RENDER AS TEXT): {title}\n"
            f"CORE GENRE: {genre or '(evolving)'}\n"
            f"ELEVATOR PITCH: {pitch or '(evolving)'}\n"
            f"CONFIRMED GAMEPLAY PILLARS:\n"
            + ("\n".join(f"- {p}" for p in pillars) if pillars else "- (none locked yet)")
            + f"\n\nSTYLE & CAMERA DIRECTION:\n{effective_style or '(derive crisp high-craft medium fitting the concept)'}\n"
            f"COLOR PALETTE HEX ANCHORS: {', '.join(palette) if palette else '(derive cohesive high-contrast game palette)'}\n"
            f"RECENT CONVERSATION CONTEXT:\n{(conversation_history or '(none)')[-2800:]}\n\n"
            f"LATEST USER MESSAGE:\n{latest_user_message or '(none)'}\n\n"
            f"CO-PRODUCER SCENE REQUEST:\n{raw_scene_request}\n\n"
            f"CUMULATIVE HARD VISUAL EXCLUSIONS:\n"
            + ("\n".join(f"- {n}" for n in neg_constraints) if neg_constraints else "- (apply universal anti-slop exclusions)")
            + "\n\nProvide `deliberation` FIRST."
        )
        return self._call_structured(
            prompt=shot_prompt,
            schema_cls=ConceptArtShotPlan,
            temperature=0.35,
            label="CoProducerAgent.compile_concept_art_shot_plan",
            multimodal_parts=mm_parts or None,
            candidate_dir=candidate_dir,
            prompt_trace_sink=prompt_trace_sink,
        )

    def _critique_concept_art_render(
        self,
        *,
        rendered_path: Path,
        shot_plan: ConceptArtShotPlan,
        latest_user_message: str = "",
        attempt: int = 1,
        candidate_dir: Optional[Path] = None,
        prompt_trace_sink: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[ConceptArtCritiqueVerdict, Dict[str, int]]:
        """Multimodally inspects a rendered 16:9 concept PNG before staging it to catch scale collapse,
        perspective drift, poster text, or violated user constraints."""
        img_part = self._load_png_part(rendered_path)
        if img_part is None:
            raise StudioAgentError(f"Rendered concept art PNG at '{rendered_path}' could not be loaded for multimodal critique.")

        critique_prompt = (
            f"You are the Studio Art-Direction Quality Gate inspecting candidate 16:9 concept render attempt #{attempt}.\n"
            "Examine the attached PNG pixels carefully and honestly against the user's latest feedback and `ConceptArtShotPlan`.\n\n"
            "AUTOMATIC REJECTION CONDITIONS (`passes_art_direction_bar = False` if ANY occur in the image):\n"
            "1. TEXT / POSTER / HUD SLOP: Visible title lettering, poster headers, watermarks, floating callout text labels, "
            "or unrequested HUD bars/minimaps.\n"
            "2. CAMERA PROJECTION DRIFT: Camera angle violates `camera_projection_and_viewport_scale` (e.g., 3D perspective, "
            "isometric tilt, or side-view platformer cross-section when pure top-down 2D was specified, or vice versa).\n"
            "3. SPATIAL SCALE COLLAPSE: Either (a) shrinking a vast environment into a tiny centered concentric bullseye / "
            "miniature toy ring around a single unit, OR (b) zooming out to a whole-world diagram and pasting a giant "
            "continent-sized flat icon/cross on top instead of proportional playable structures.\n"
            "4. RENDERING MEDIUM DRIFT: Unrequested chunky retro pixel art or simplified mobile-app clip-art icon when crisp "
            "high-definition 2D/3D game art was specified.\n"
            "5. HARD EXCLUSION VIOLATION: Any item in `hard_visual_exclusions` appears in the rendered pixels.\n\n"
            f"LATEST USER MESSAGE:\n{latest_user_message or '(none)'}\n\n"
            f"TARGET CONCEPT ART SHOT PLAN JSON:\n{shot_plan.model_dump_json(indent=2)}\n\n"
            "If `passes_art_direction_bar` is False, write a concrete, self-contained `corrective_scene_rewrite` that fixes "
            "every observed defect by describing the exact playable viewport geometry, relative proportions, depth layers, "
            "and surface craft to render instead.\n"
            "Provide `deliberation` FIRST and `passes_art_direction_bar` LAST."
        )
        return self._call_structured(
            prompt=critique_prompt,
            schema_cls=ConceptArtCritiqueVerdict,
            temperature=0.20,
            label=f"CoProducerAgent.critique_concept_art[attempt{attempt}]",
            multimodal_parts=[f"CANDIDATE CONCEPT MOCKUP ATTEMPT #{attempt} (`{rendered_path.name}`):", img_part],
            candidate_dir=candidate_dir,
            prompt_trace_sink=prompt_trace_sink,
        )

    def _tool_sketch_multimodal_concept(
        self,
        call: CoProducerToolCall,
        thread_meta: Dict[str, Any],
        multimodal_parts: List[Any],
        staged_media_cards: List[CoProducerMediaCard],
        *,
        latest_user_message: str = "",
        conversation_history: str = "",
        prompt_trace_sink: Optional[List[Dict[str, Any]]] = None,
        usage_accumulator: Optional[Callable[[Dict[str, int]], None]] = None,
    ) -> str:
        if self.client is None:
            raise StudioAgentConfigurationError(
                "sketch_multimodal_concept requires an active Gemini client (Rule 5: Zero Silent Fallbacks)."
            )
        t_dir = self.store.resolve_thread_dir(thread_meta)
        cid = str(thread_meta.get("cartridge_id") or "").strip()
        cand_dir = (self.cartridges_root / cid) if cid else None
        sketches_dir = t_dir / "assets" / "sketches"
        sketches_dir.mkdir(parents=True, exist_ok=True)

        kind = call.sketch_kind or "concept_mockup_16x9"
        title = (call.sketch_title or "").strip() or kind.replace("_", " ").title()
        prompt_text = (call.query_or_question or "").strip() or title
        secondary = (call.sketch_secondary_direction or "").strip()
        brief = dict(thread_meta.get("concept_brief") or {})
        brief_style = str(brief.get("visual_style_direction") or "").strip()
        effective_style = (
            f"{secondary}\nEstablished Concept Style Direction: {brief_style}"
            if (secondary and brief_style and brief_style.lower() not in secondary.lower())
            else (secondary or brief_style)
        )
        neg_constraints: List[str] = []
        seen_neg: set[str] = set()
        for raw_neg in list(call.sketch_negative_constraints or []) + list(brief.get("explicit_non_goals") or []):
            clean_neg = str(raw_neg or "").strip()
            if clean_neg and clean_neg.lower() not in seen_neg:
                seen_neg.add(clean_neg.lower())
                neg_constraints.append(clean_neg)

        slug = _safe_slug(title, kind)
        stamp = uuid.uuid4().hex[:6]

        if kind == "concept_mockup_16x9":
            from engine.studio.visual import VisualArtSpriteDesignerAgent

            vis = VisualArtSpriteDesignerAgent(
                client=self.client,
                model_name=self.image_model_name,
                concept_mockup_model_name=self.concept_mockup_model_name,
            )
            out_path = sketches_dir / f"mockup_{slug}_{stamp}.png"
            prior_ref_path = (
                self._resolve_prior_sketch_reference_path(thread_meta, sketches_dir)
                if bool(getattr(call, "use_prior_sketch_as_reference", False))
                else None
            )
            ref_list = [prior_ref_path] if prior_ref_path is not None else None

            final_verdict: Optional[ConceptArtCritiqueVerdict] = None
            attempts_used = 1
            caption_summary = prompt_text[:220]

            if self.enable_art_direction_pipeline:
                shot_plan, plan_usage = self._compile_concept_art_shot_plan(
                    title=title,
                    raw_scene_request=prompt_text,
                    effective_style=effective_style,
                    neg_constraints=neg_constraints,
                    thread_meta=thread_meta,
                    latest_user_message=latest_user_message,
                    conversation_history=conversation_history,
                    prior_reference_path=prior_ref_path,
                    candidate_dir=cand_dir,
                    prompt_trace_sink=prompt_trace_sink,
                )
                if callable(usage_accumulator):
                    usage_accumulator(plan_usage)
                caption_summary = shot_plan.playable_midground_composition[:220]

                vis.generate_concept_mockup(
                    {
                        "title": title,
                        "scene_description": prompt_text,
                        "negative_constraints": neg_constraints,
                    },
                    output_path=out_path,
                    style_guide=effective_style,
                    negative_constraints=neg_constraints,
                    reference_images=ref_list,
                    shot_plan=shot_plan,
                )

                verdict_1, crit_usage_1 = self._critique_concept_art_render(
                    rendered_path=out_path,
                    shot_plan=shot_plan,
                    latest_user_message=latest_user_message,
                    attempt=1,
                    candidate_dir=cand_dir,
                    prompt_trace_sink=prompt_trace_sink,
                )
                if callable(usage_accumulator):
                    usage_accumulator(crit_usage_1)
                final_verdict = verdict_1

                if not verdict_1.passes_art_direction_bar and (
                    verdict_1.corrective_scene_rewrite.strip() or verdict_1.flaws_or_constraint_violations
                ):
                    attempts_used = 2
                    self._emit(
                        "coproducer_sketch_regenerating",
                        verdict_1.deliberation,
                        {
                            "thread_id": thread_meta.get("thread_id"),
                            "cartridge_id": cid or None,
                            "title": title,
                            "flaws_detected": list(verdict_1.flaws_or_constraint_violations),
                        },
                    )
                    retry_exclusions: List[str] = []
                    seen_retry_ex: set[str] = set()
                    for ex_item in (
                        list(shot_plan.hard_visual_exclusions)
                        + list(neg_constraints)
                        + list(verdict_1.flaws_or_constraint_violations)
                    ):
                        clean_ex = str(ex_item or "").strip()
                        if clean_ex and clean_ex.lower() not in seen_retry_ex:
                            seen_retry_ex.add(clean_ex.lower())
                            retry_exclusions.append(clean_ex)

                    revised_midground = (
                        verdict_1.corrective_scene_rewrite.strip()
                        or shot_plan.playable_midground_composition
                    )
                    revised_shot_plan = shot_plan.model_copy(
                        update={
                            "playable_midground_composition": revised_midground,
                            "hard_visual_exclusions": retry_exclusions,
                        }
                    )
                    caption_summary = revised_midground[:220]
                    vis.generate_concept_mockup(
                        {
                            "title": title,
                            "scene_description": revised_midground,
                            "negative_constraints": retry_exclusions,
                        },
                        output_path=out_path,
                        style_guide=effective_style,
                        negative_constraints=retry_exclusions,
                        reference_images=None,
                        shot_plan=revised_shot_plan,
                    )
                    verdict_2, crit_usage_2 = self._critique_concept_art_render(
                        rendered_path=out_path,
                        shot_plan=revised_shot_plan,
                        latest_user_message=latest_user_message,
                        attempt=2,
                        candidate_dir=cand_dir,
                        prompt_trace_sink=prompt_trace_sink,
                    )
                    if callable(usage_accumulator):
                        usage_accumulator(crit_usage_2)
                    final_verdict = verdict_2
            else:
                vis.generate_concept_mockup(
                    {
                        "title": title,
                        "scene_description": prompt_text,
                        "negative_constraints": neg_constraints,
                    },
                    output_path=out_path,
                    style_guide=effective_style,
                    negative_constraints=neg_constraints,
                    reference_images=ref_list,
                )

            rel_path = self._artifact_rel_for_thread(out_path, thread_meta)
            w_px, h_px = self._image_dimensions(out_path)
            part = self._load_png_part(out_path)
            if part is not None:
                multimodal_parts.append(f"INLINE 16:9 CONCEPT MOCKUP SKETCH (`{rel_path}`):")
                multimodal_parts.append(part)
            staged_media_cards.append(
                CoProducerMediaCard(
                    deliberation=(
                        final_verdict.deliberation
                        if final_verdict is not None
                        else f"Synthesized 16:9 inline concept mockup '{title}' for creative ideation."
                    ),
                    card_id=f"sketch_{stamp}",
                    card_kind="sketch_showcase",
                    title=f"16:9 Concept Sketch — {title}",
                    subtitle="Inline Co-Producer visual prototype",
                    items=[
                        CoProducerMediaItem(
                            deliberation=call.deliberation,
                            item_id=f"item_{stamp}",
                            media_kind="image",
                            label=title,
                            caption=caption_summary,
                            artifact_path=rel_path,
                            mime_type="image/png",
                            width=w_px or 1280,
                            height=h_px or 720,
                        )
                    ],
                )
            )
            obs_msg = f"Synthesized 16:9 concept mockup at `{rel_path}` ({w_px}x{h_px}, model={self.concept_mockup_model_name}, attempts={attempts_used})."
            if final_verdict is not None:
                obs_msg += (
                    f"\n[Art-Direction Pixel Audit: passed_bar={final_verdict.passes_art_direction_bar} | "
                    f"scale_and_camera={final_verdict.observed_camera_and_scale_assessment} | "
                    f"summary={final_verdict.honest_producer_summary}"
                    + (
                        f" | remaining_imperfections_to_acknowledge_honestly={final_verdict.flaws_or_constraint_violations}"
                        if final_verdict.flaws_or_constraint_violations
                        else ""
                    )
                    + "]"
                )
            return obs_msg

        if kind == "sprite_sketch":
            from engine.studio.visual import VisualArtSpriteDesignerAgent

            vis = VisualArtSpriteDesignerAgent(client=self.client, model_name=self.image_model_name)
            sprite_path = sketches_dir / f"sprite_{slug}_{stamp}.png"
            plate_path = sketches_dir / f"sprite_{slug}_{stamp}_plate.png"
            sprite_prompt = (
                f"{prompt_text}\nSTRICT VISUAL EXCLUSIONS: {'; '.join(neg_constraints)}"
                if neg_constraints
                else prompt_text
            )
            vis.generate_dynamic_sprite(
                {
                    "asset_id": f"sketch_{slug}",
                    "name": title,
                    "asset_type": "sprite",
                    "dimensions": (128, 128),
                    "visual_prompt_or_text": sprite_prompt,
                },
                output_path=sprite_path,
                style_guide=effective_style,
            )
            vis.render_asset_inspection_plate(sprite_path, asset_type="sprite", output_path=plate_path)
            rel_sprite = self._artifact_rel_for_thread(sprite_path, thread_meta)
            rel_plate = self._artifact_rel_for_thread(plate_path, thread_meta)
            pw, ph = self._image_dimensions(plate_path)
            part = self._load_png_part(plate_path)
            if part is not None:
                multimodal_parts.append(f"INLINE SPRITE SKETCH INSPECTION PLATE (`{rel_plate}`):")
                multimodal_parts.append(part)
            staged_media_cards.append(
                CoProducerMediaCard(
                    deliberation=f"Synthesized sprite test '{title}' with dual-checkerboard alpha inspection plate.",
                    card_id=f"sketch_{stamp}",
                    card_kind="sketch_showcase",
                    title=f"Sprite Prototype — {title}",
                    subtitle="Isolated sprite + dual-checkerboard alpha cutout plate",
                    items=[
                        CoProducerMediaItem(
                            deliberation="Dual-tone alpha inspection plate.",
                            item_id=f"plate_{stamp}",
                            media_kind="image",
                            label=f"{title} (Alpha Plate)",
                            caption=prompt_text[:200],
                            artifact_path=rel_plate,
                            mime_type="image/png",
                            width=pw or 384,
                            height=ph or 192,
                        ),
                        CoProducerMediaItem(
                            deliberation="Raw isolated RGBA sprite PNG.",
                            item_id=f"raw_{stamp}",
                            media_kind="image",
                            label=f"{title} (Raw PNG)",
                            artifact_path=rel_sprite,
                            mime_type="image/png",
                            width=128,
                            height=128,
                        ),
                    ],
                )
            )
            return f"Synthesized sprite sketch `{rel_sprite}` and alpha inspection plate `{rel_plate}`."

        if kind == "voice_audition":
            from engine.studio.audio import VoiceActingAudioProducerAgent

            aud = VoiceActingAudioProducerAgent(client=self.client, model_name=self.tts_model_name)
            wav_path = sketches_dir / f"voice_{slug}_{stamp}.wav"
            aud.synthesize_voice_line(
                {
                    "bark_id": f"audition_{stamp}",
                    "speaker_id": title,
                    "line_text": prompt_text,
                    "voice_direction": secondary or "expressive, grounded character delivery",
                },
                output_path=wav_path,
            )
            rel_wav = self._artifact_rel_for_thread(wav_path, thread_meta)
            staged_media_cards.append(
                CoProducerMediaCard(
                    deliberation=f"Synthesized character voice audition '{title}'.",
                    card_id=f"sketch_{stamp}",
                    card_kind="sketch_showcase",
                    title=f"Voice Audition — {title}",
                    subtitle=secondary or "Character TTS prototype",
                    items=[
                        CoProducerMediaItem(
                            deliberation=call.deliberation,
                            item_id=f"voice_{stamp}",
                            media_kind="audio",
                            label=title,
                            caption=f'"{prompt_text}"',
                            artifact_path=rel_wav,
                            mime_type="audio/wav",
                        )
                    ],
                )
            )
            return f"Synthesized character voice audition WAV at `{rel_wav}`."

        # sfx_or_ambience
        from engine.studio.audio import VoiceActingAudioProducerAgent

        aud = VoiceActingAudioProducerAgent(client=self.client, model_name=self.tts_model_name)
        wav_path = sketches_dir / f"{'ambient' if call.loop_audio else 'sfx'}_{slug}_{stamp}.wav"
        aud.generate_sfx(
            prompt=f"{prompt_text} {secondary}".strip(),
            output_path=wav_path,
            kind="ambient" if call.loop_audio else "sfx",
            loop=bool(call.loop_audio),
        )
        rel_wav = self._artifact_rel_for_thread(wav_path, thread_meta)
        staged_media_cards.append(
            CoProducerMediaCard(
                deliberation=f"Synthesized procedural {'ambient loop' if call.loop_audio else 'SFX'} prototype '{title}'.",
                card_id=f"sketch_{stamp}",
                card_kind="sketch_showcase",
                title=f"{'Ambient Bed' if call.loop_audio else 'SFX Sketch'} — {title}",
                subtitle="Parametric DSP audio prototype",
                items=[
                    CoProducerMediaItem(
                        deliberation=call.deliberation,
                        item_id=f"sfx_{stamp}",
                        media_kind="audio",
                        label=title,
                        caption=prompt_text[:200],
                        artifact_path=rel_wav,
                        mime_type="audio/wav",
                    )
                ],
            )
        )
        return f"Synthesized {'ambient loop' if call.loop_audio else 'SFX'} WAV at `{rel_wav}`."

    def _execute_tool_call(
        self,
        call: CoProducerToolCall,
        thread_meta: Dict[str, Any],
        multimodal_parts: List[Any],
        staged_media_cards: List[CoProducerMediaCard],
        consultation_cards: List[CoProducerConsultationCard],
        *,
        latest_user_message: str = "",
        conversation_history: str = "",
        prompt_trace_sink: Optional[List[Dict[str, Any]]] = None,
        usage_accumulator: Optional[Callable[[Dict[str, int]], None]] = None,
    ) -> str:
        t_name = call.tool_name
        if t_name == "inspect_studio_progress":
            return self._tool_inspect_studio_progress(call, thread_meta, multimodal_parts, staged_media_cards)
        if t_name == "inspect_game_bible_and_dna":
            return self._tool_inspect_game_bible_and_dna(call, thread_meta, multimodal_parts)
        if t_name == "inspect_playtest_sessions":
            return self._tool_inspect_playtest_sessions(call, thread_meta, multimodal_parts, staged_media_cards)
        if t_name == "browse_cartridge_media_and_code":
            return self._tool_browse_cartridge_media_and_code(call, thread_meta, multimodal_parts, staged_media_cards)
        if t_name == "curate_genre_fun_foundation":
            return self._tool_curate_genre_fun_foundation(call, thread_meta, consultation_cards)
        if t_name == "consult_council":
            return self._tool_consult_council(
                call,
                thread_meta,
                multimodal_parts,
                consultation_cards,
                latest_user_message=latest_user_message,
                conversation_history=conversation_history,
            )
        if t_name == "dispatch_specialist_probe":
            return self._tool_dispatch_specialist_probe(call, thread_meta, multimodal_parts, consultation_cards)
        if t_name == "sketch_multimodal_concept":
            return self._tool_sketch_multimodal_concept(
                call,
                thread_meta,
                multimodal_parts,
                staged_media_cards,
                latest_user_message=latest_user_message,
                conversation_history=conversation_history,
                prompt_trace_sink=prompt_trace_sink,
                usage_accumulator=usage_accumulator,
            )
        raise StudioAgentError(f"Unsupported CoProducerToolCall.tool_name: {t_name!r}")

    # -------------------------------------------------------------------------
    # Main Interactive Turn Loop
    # -------------------------------------------------------------------------

    def run_turn(
        self,
        *,
        thread_id: str,
        user_message: str,
        selected_sketch_paths: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Executes a complete 1-on-1 Co-Producer turn: persists user message, runs bounded LLM ReAct
        tool/consultation/sketch steps, synthesizes the multimodal response, and persists the turn."""
        if self.client is None:
            raise StudioAgentConfigurationError(
                "CoProducerAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)."
            )
        clean_user_msg = (user_message or "").strip()
        if not clean_user_msg:
            raise StudioAgentError("CoProducerAgent.run_turn requires a non-empty `user_message`.")

        detail = self.store.get_thread_detail(thread_id, limit=200)
        if detail is None:
            raise StudioAgentError(f"Unknown Co-Producer thread_id: {thread_id!r}")
        thread_meta = dict(detail["thread"])
        prior_messages = list(detail["messages"])

        # Reuse trailing unanswered identical user message on retry; otherwise persist immediately
        if (
            prior_messages
            and str(prior_messages[-1].get("sender_role") or "") == "user"
            and str(prior_messages[-1].get("content_markdown") or "").strip() == clean_user_msg
        ):
            user_msg_rec = dict(prior_messages[-1])
            prior_messages = prior_messages[:-1]
        else:
            thread_meta, user_msg_rec = self.store.append_message(
                thread_meta,
                {
                    "deliberation": "User submitted interactive message to Studio Co-Producer.",
                    "sender_role": "user",
                    "content_markdown": clean_user_msg,
                },
            )
        cid = str(thread_meta.get("cartridge_id") or "").strip()
        cand_dir = (self.cartridges_root / cid) if cid else None

        tool_observations: List[str] = []
        tool_trace: List[Dict[str, Any]] = []
        prompt_trace: List[Dict[str, Any]] = []
        multimodal_parts: List[Any] = []
        staged_media_cards: List[CoProducerMediaCard] = []
        consultation_cards: List[CoProducerConsultationCard] = []
        cumulative_usage = {
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "thought_tokens": 0,
            "total_tokens": 0,
        }
        chosen_response_mode = "creative_ideation"

        def _accum_usage(u: Dict[str, int]) -> None:
            for k in cumulative_usage:
                cumulative_usage[k] += int(u.get(k, 0))

        # Attach any explicitly highlighted user reference sketches
        t_dir = self.store.resolve_thread_dir(thread_meta)
        for rel_sk in list(selected_sketch_paths or [])[:3]:
            cand_sk = Path(rel_sk) if Path(rel_sk).is_absolute() else (t_dir / str(rel_sk).lstrip("/"))
            if not cand_sk.is_file():
                cand_sk = self.cartridges_root / str(rel_sk).lstrip("/")
            if cand_sk.is_file():
                part = self._load_png_part(cand_sk)
                if part is not None:
                    multimodal_parts.append(f"USER-SELECTED SKETCH / ASSET (`{rel_sk}`):")
                    multimodal_parts.append(part)

        context_header = self._build_lightweight_context_header(thread_meta)
        history_block = self._format_conversation_history(prior_messages)

        with interactive_foreground_call_scope():
            for hop_idx in range(self.max_tool_hops):
                obs_block = (
                    "\n\n".join(tool_observations)
                    if tool_observations
                    else "(No tools called yet on this turn.)"
                )
                step_prompt = (
                    "You are the Studio Co-Producer (`CoProducer`) — both the Executive Producer's 1-on-1 creative partner AND the Leader & Chair of the Specialist Council.\n"
                    "Decide whether you have enough grounded evidence, specialist input, or sketches to answer the user right now (`action = 'respond'`), "
                    "OR whether calling 1 to 3 tools first will make your response dramatically more grounded, creative, or visually/sonically tangible (`action = 'call_tools'`).\n\n"
                    f"=== THREAD & CARTRIDGE SNAPSHOT ===\n{context_header}\n\n"
                    f"=== PRIOR CONVERSATION ===\n{history_block}\n\n"
                    f"=== LATEST USER MESSAGE ===\n{clean_user_msg}\n\n"
                    f"=== TOOL OBSERVATIONS ALREADY GATHERED THIS TURN (hop {hop_idx}/{self.max_tool_hops}) ===\n{obs_block}\n\n"
                    "AVAILABLE TOOLS:\n"
                    "- `inspect_studio_progress`: Live ProductionBoard task counts/states, slice roadmap, recent QA clips/filmstrips, and token/cost telemetry.\n"
                    "- `inspect_game_bible_and_dna`: Living Game Bible design pillars/mockups and/or local Steam Player DNA taste signals.\n"
                    "- `inspect_playtest_sessions`: Blind Playtester videos (`replay.mp4`), fun ratings, bugs, wishes, turn screenshots, and optional perceptual video/audio evaluation.\n"
                    "- `browse_cartridge_media_and_code`: Cartridge sprites, mockups, `.wav` audio clips, `GameMemoryStore` rules, recent `code_diffs`, and semantic code lookup.\n"
                    "- `curate_genre_fun_foundation`: Have `ContextCuratorAgent` synthesize or evolve the game's first-principles `GenreCraftFoundation` (core fun thesis, 3–5 evolvable pillars, `t=0` bootstrap-to-mastery arc, spatial/UX precision contracts, and genre-killing anti-patterns). Call this whenever exploring, pitching, critiquing, or pivoting a game genre if `GENRE_CRAFT_FOUNDATION` is missing or changing!\n"
                    "- `consult_council`: Convene 1-4 of your Council domain specialists (`GameplayDesigner`, `StoryWorldArchitect`, `VisualArtDirector`, `LevelAndProgressionDesigner`, `GameplaySystemsEngineer`, `UXExperienceDesigner`, `AudioAtmosphereDesigner`, `ScriptwriterAgent`, `CharacterDesigner`) in `council_mode='huddle'`, OR run `council_mode='divergent_pitch_round'` for orthogonal specialist fragments + premise collision under your Council Chair synthesis.\n"
                    "- `dispatch_specialist_probe`: Run a lock-free read-only technical/art/audio/QA probe (`swe_engineer`, `visual_artist`, `audio_designer`, `script_writer`, `qa_test_engineer`).\n"
                    "- `sketch_multimodal_concept`: Synthesize inline creative prototypes (`concept_mockup_16x9`, `sprite_sketch`, `voice_audition`, `sfx_or_ambience`) so the user can see and hear ideas before committing.\n\n"
                    "GUIDELINES:\n"
                    "1. Do NOT repeat a tool call you have already run in `TOOL OBSERVATIONS ALREADY GATHERED THIS TURN`.\n"
                    "2. When the user asks about build status, progress, assets, code, or playtest results/videos, call the matching inspection tool(s) first so you show real cards and never guess.\n"
                    "3. When exploring a new genre, diagnosing why a game isn't fun, or pivoting a game's core loop where `GENRE_CRAFT_FOUNDATION` is not yet synthesized (or needs evolving), call `curate_genre_fun_foundation` so you and the Council share a deep first-principles foundation of what makes that genre genuinely fun.\n"
                    "4. When brainstorming or pressure-testing mechanics/art/story where specialist perspectives or inline visual/audio sketches will delight or clarify, call `consult_council` and/or `sketch_multimodal_concept`.\n"
                    "   - Whenever calling `consult_council`, ALWAYS populate `query_or_question` with the exact design trade-off, system, or user correction raised in `LATEST USER MESSAGE`—never leave `query_or_question` empty.\n"
                    "   - Whenever calling `sketch_multimodal_concept`, ALWAYS populate `query_or_question` with concrete playable viewport composition, single coherent camera projection, and explicit relative entity-to-structure-to-world proportions (keeping the camera locked to the local playable viewport and conveying colossal world scale via edge-to-edge fields + deep background parallax rather than impossible whole-world diagrams); populate `sketch_secondary_direction` with exact rendering medium & camera rules; populate `sketch_negative_constraints` with EVERY excluded visual element, forbidden camera angle, or unrequested UI chrome across the conversation; and set `use_prior_sketch_as_reference = True` ONLY when making targeted refinements to a prior sketch whose composition and scale already worked (leave `False` when resetting a rejected or drifted composition).\n"
                    "5. For quick conversational follow-ups where context is already sufficient, choose `action = 'respond'` immediately.\n"
                    "Provide `deliberation` FIRST and `action` LAST."
                )
                decision, step_usage = self._call_structured(
                    prompt=step_prompt,
                    schema_cls=CoProducerStepDecision,
                    temperature=0.45,
                    label=f"CoProducerAgent.step_decision[hop{hop_idx}]",
                    multimodal_parts=multimodal_parts[-4:] if multimodal_parts else None,
                    candidate_dir=cand_dir,
                    prompt_trace_sink=prompt_trace,
                )
                _accum_usage(step_usage)
                chosen_response_mode = decision.response_mode or chosen_response_mode

                if decision.action == "respond" or not decision.tool_calls:
                    break

                self._emit(
                    "coproducer_step_started",
                    decision.deliberation,
                    {
                        "thread_id": thread_id,
                        "cartridge_id": cid or None,
                        "hop_index": hop_idx,
                        "intent": decision.next_step_intent or f"Executing {len(decision.tool_calls)} studio tool(s)...",
                        "tools": [tc.tool_name for tc in decision.tool_calls[:3]],
                    },
                )

                for tc in decision.tool_calls[:3]:
                    t_start = _utc_now_iso()
                    obs_text = self._execute_tool_call(
                        tc,
                        thread_meta=thread_meta,
                        multimodal_parts=multimodal_parts,
                        staged_media_cards=staged_media_cards,
                        consultation_cards=consultation_cards,
                        latest_user_message=clean_user_msg,
                        conversation_history=history_block,
                        prompt_trace_sink=prompt_trace,
                        usage_accumulator=_accum_usage,
                    )
                    tool_observations.append(f"[Tool `{tc.tool_name}`]:\n{obs_text}")
                    trace_entry = _ordered_rationale_dict(
                        tc.deliberation,
                        tool_name=tc.tool_name,
                        query_or_question=tc.query_or_question,
                        started_at_utc=t_start,
                        completed_at_utc=_utc_now_iso(),
                        observation_excerpt=obs_text[:1500],
                    )
                    tool_trace.append(trace_entry)
                    self._emit(
                        "coproducer_tool_completed",
                        tc.deliberation,
                        {
                            "thread_id": thread_id,
                            "cartridge_id": cid or None,
                            "hop_index": hop_idx,
                            "tool_name": tc.tool_name,
                            "summary": obs_text.splitlines()[0][:200] if obs_text else tc.tool_name,
                        },
                    )
                context_header = self._build_lightweight_context_header(thread_meta)

            # Final Turn Synthesis (High-Creativity 0.90 for ideation, 0.65 for factual studio briefings)
            synth_temp = 0.90 if chosen_response_mode == "creative_ideation" else 0.65
            staged_cards_json = json.dumps([c.model_dump() for c in staged_media_cards], indent=2, default=str)[:4500]
            consult_json = json.dumps([c.model_dump() for c in consultation_cards], indent=2, default=str)[:4500]
            obs_final_block = "\n\n".join(tool_observations) if tool_observations else "(No tool calls executed on this turn.)"

            synth_prompt = (
                "You are the Executive Producer's 1-on-1 Creative Co-Producer (`CoProducer`), Studio Showrunner, AND the active Leader & Chair of the Specialist Council.\n"
                "VOICE, LEADERSHIP & EDITORIAL BREVITY:\n"
                "- You personally lead and chair the Specialist Council in the Writers' Room across concept genesis, task reviews, blind playtests, and slice strategy—own the Council's direction in the first person (e.g. 'I'll have our Gameplay Designer and Systems Engineer...', 'When I convene the Council...') rather than treating the Council as a separate department.\n"
                "- Speak directly in a single, warm, razor-sharp creative partner voice (`spoken_response` in GitHub-flavored Markdown).\n"
                "- Keep `spoken_response` concise, conversational, and high-signal (typically 1–3 focused paragraphs or a tight scannable list—like a human Creative Producer talking across the table). Do NOT dump walls of exhaustive bullet points, raw internal telemetry, or repetitive restatements of attached cards; deep backstage details are already accessible via the UI's Behind-the-Scenes observability drill-down.\n"
                "- NEVER roleplay multiple speakers inline in `spoken_response`; any Council huddles or specialist probes are already attached separately in Behind-the-Scenes observability—synthesize only their sharpest insights and trade-offs in your own voice.\n"
                "- During creative ideation, be imaginative and concrete: ground your pitches, critiques, and trade-offs in `GENRE_CRAFT_FOUNDATION` (why players genuinely love this genre, how `t=0` starts unsolved with manual agency and scarcity, spatial/UX precision, and avoiding genre-killing anti-patterns), offering vivid playable hooks, tactile moment-to-moment player verbs, and contrasting creative forks without jargon salad.\n"
                "- During studio status/playtest/media reviews, ground every statement in the real metrics, attached screenshots, and staged media cards.\n\n"
                "COLLABORATIVE MEMORY, CONSTRAINT FIDELITY & HONEST SKETCH CRITIQUE:\n"
                "1. CUMULATIVE BRIEF INTEGRITY (NO RECENCY OVERWRITING): Treat a targeted user correction or critique on one detail as an incremental delta on top of ALL previously agreed pillars in `PRIOR CONVERSATION` and `USER_CONFIRMED_PILLARS_LEDGER`. NEVER overwrite or narrow `updated_concept_brief` around only the latest message! Carry forward all active confirmed mechanics, scale layers, progression arcs, and exclusions in `user_confirmed_pillars` and `explicit_non_goals`, modifying or removing an entry ONLY if the user explicitly retracts or replaces it. Record unresolved design trade-offs in `open_design_questions`.\n"
                "2. EXECUTIVE PRODUCER PRIMACY OVER BACKGROUND DNA: Steam Player DNA is passive background inspiration only. When the user specifies desired mechanics, systems, or feel, prioritize their explicit goals 100%—never push unrequested simulation chores, physics micromanagement, or unrelated genre detours over the systems the user asked to deepen. When a design tension is open, lay out 2–3 crisp mechanical options and ask a focused follow-up question.\n"
                "3. HONEST MULTIMODAL SKETCH AUDIT (`sketch_fidelity_audit`): Before composing `spoken_response`, carefully inspect BOTH any `[Art-Direction Pixel Audit: ...]` notes in `GATHERED TOOL EVIDENCE` AND the attached sketch/mockup PNG pixels against the user's requested camera angle, relative playable scale, rendering medium, and `EXPLICIT_NON_GOALS_AND_VISUAL_EXCLUSIONS`. If a rendered sketch still drifted or retained any imperfection (e.g., wrong camera perspective, oversized/undersized scale ratio, forbidden background element, or unrequested HUD chrome), call out candidly what landed and what missed in `spoken_response`—NEVER gaslight the user or claim tiny/exact scale when the rendered pixels show otherwise.\n"
                "4. CURATE WHAT THE USER SEES (`presented_card_ids`): Set `presented_card_ids` to ONLY the `card_id`s from `AUTO-STAGED MEDIA CARDS` that you genuinely want displayed front-and-center beneath your reply right now (e.g., newly sketched `sketch_showcase` prototypes, or `playtest_video` / `qa_clip` / `image_gallery` / `code_diff` when showing media directly supports your message). Pass `presented_card_ids = []` when you merely ran inspection tools for background context.\n\n"
                "WHEN TO PROPOSE `action_proposals` (DO NOT BE A PREMATURE CLOSER — DEFAULT TO `action_proposals = []`):\n"
                "- Leave `action_proposals = []` whenever the user is critiquing a sketch, correcting a misunderstanding, expressing reservations, asking questions, or actively exploring `open_design_questions`.\n"
                "1. Pre-Genesis / New Game Commissioning (`commission_game` — ONLY on `SCOPE: draft` threads):\n"
                "   - Propose `commission_game` ONLY when the user explicitly asks to commission/start/build the game, OR explicitly signals they are happy with the converged concept and ready to kick off production.\n"
                "   - NEVER propose `commission_game` when `SCOPE: cartridge` already has a `cartridge_id` and `HAS_PERSISTED_GAME_BIBLE: False` (your `updated_concept_brief` already syncs live to `co_producer_brief.json` automatically; propose `resume_build` ONLY if the user explicitly asks to start the autonomous build).\n"
                "   - `handoff_mode = 'loose_spark'`: User has an open spark and wants the Writers' Room Council to run divergent concept genesis around it.\n"
                "   - `handoff_mode = 'refined_concept'`: User & Co-Producer have converged on a specific game vision (`working_title`, `core_player_verbs`, `user_confirmed_pillars`, `visual_style_direction`, `explicit_non_goals`)—locks that vision into `GameBibleSpec` and skips blind concept genesis.\n"
                "2. Active Cartridge Pivots (Three-Tier Pivot Engine — when `HAS_PERSISTED_GAME_BIBLE: True`):\n"
                "   - `pivot_course_correction` (Tier 1): Additive tuning, feel/art/UI polish, or new mechanic within the current game—queues live `.pending_steering.json` without cancelling active work.\n"
                "   - `pivot_structural` (Tier 2): Major mechanic/camera/loop overhaul—pauses workers, cancels obsolete `cancel_item_ids`, applies `memory_mutations` (`supersede` old rules / `record` new constraints), and triggers Council replan.\n"
                "   - `pivot_fork_cartridge` (Tier 3): Premise/genre reboot—forks a fresh `candidate_genesis_*` cartridge while preserving the current cartridge untouched and copying `carry_over_asset_paths`.\n"
                "3. Direct Studio Operations (ONLY when explicitly requested or clearly warranted by a milestone): `greenlight_bible`, `trigger_next_slice`, `pause_build`, `resume_build`, `launch_game`, `start_blind_playtest`, `record_game_memory`.\n\n"
                f"=== THREAD & CARTRIDGE SNAPSHOT ===\n{context_header}\n\n"
                f"=== PRIOR CONVERSATION ===\n{history_block}\n\n"
                f"=== LATEST USER MESSAGE ===\n{clean_user_msg}\n\n"
                f"=== GATHERED TOOL EVIDENCE ===\n{obs_final_block}\n\n"
                f"=== AUTO-STAGED MEDIA CARDS (select which `card_id`s to show inline via `presented_card_ids`; you may leave `media_cards=[]` or add extra cards using ONLY real artifact paths from GATHERED TOOL EVIDENCE) ===\n{staged_cards_json}\n\n"
                f"=== ATTACHED CONSULTATION CARDS (preserved in Behind-the-Scenes observability) ===\n{consult_json}\n\n"
                "Keep `updated_concept_brief` cumulatively updated across ALL confirmed pillars, player verbs, style direction, sketches, explicit non-goals, and open design questions. "
                "Provide `deliberation` FIRST, `sketch_fidelity_audit` SECOND, and `spoken_response` LAST."
            )

            final_turn, synth_usage = self._call_structured(
                prompt=synth_prompt,
                schema_cls=CoProducerFinalTurn,
                temperature=synth_temp,
                label=f"CoProducerAgent.final_turn[{chosen_response_mode}]",
                multimodal_parts=multimodal_parts[-6:] if multimodal_parts else None,
                candidate_dir=cand_dir,
                prompt_trace_sink=prompt_trace,
            )
            _accum_usage(synth_usage)

        # Merge staged_media_cards (preserving deterministic tool-grounded cards first, plus any non-duplicate model cards)
        merged_media: List[Dict[str, Any]] = []
        seen_paths: set[str] = set()
        presented_set: Optional[set[str]] = (
            {str(cid_val).strip() for cid_val in final_turn.presented_card_ids if str(cid_val).strip()}
            if final_turn.presented_card_ids is not None
            else None
        )
        for sc in staged_media_cards:
            d = sc.model_dump()
            if not d.get("card_id"):
                d["card_id"] = f"mcard_{uuid.uuid4().hex[:8]}"
            if presented_set is not None:
                # Newly synthesized sketch prototypes are always meant to be shown unless another sketch was explicitly chosen
                d["show_by_default"] = bool(
                    d["card_id"] in presented_set or sc.card_kind == "sketch_showcase"
                )
            else:
                d["show_by_default"] = True
            for it in d.get("items") or []:
                if isinstance(it, dict) and it.get("artifact_path"):
                    seen_paths.add(str(it["artifact_path"]))
            merged_media.append(d)

        for mc in final_turn.media_cards or []:
            d = mc.model_dump()
            if not d.get("card_id"):
                d["card_id"] = f"mcard_{uuid.uuid4().hex[:8]}"
            d["show_by_default"] = True
            item_paths = [str(it.get("artifact_path") or "") for it in (d.get("items") or []) if isinstance(it, dict) and it.get("artifact_path")]
            if item_paths and all(p in seen_paths for p in item_paths):
                # If the model explicitly re-emitted an already-staged card to present it, promote its visibility!
                for existing_card in merged_media:
                    ex_paths = [
                        str(it.get("artifact_path") or "")
                        for it in (existing_card.get("items") or [])
                        if isinstance(it, dict) and it.get("artifact_path")
                    ]
                    if ex_paths and any(p in ex_paths for p in item_paths):
                        existing_card["show_by_default"] = True
                continue
            for p in item_paths:
                seen_paths.add(p)
            merged_media.append(d)

        # Normalize action proposals with unique IDs and non-destructive cumulative concept_brief merge
        prev_brief: Dict[str, Any] = (
            dict(thread_meta["concept_brief"]) if isinstance(thread_meta.get("concept_brief"), dict) else {}
        )
        eff_brief_dict: Optional[Dict[str, Any]] = None
        if final_turn.updated_concept_brief is not None:
            eff_brief_dict = final_turn.updated_concept_brief.model_dump()
        elif prev_brief:
            eff_brief_dict = dict(prev_brief)

        if eff_brief_dict is not None:
            prev_title = str(prev_brief.get("working_title") or "").strip()
            new_title = str(eff_brief_dict.get("working_title") or "").strip()
            prev_genre = str(prev_brief.get("core_genre") or "").strip()
            new_genre = str(eff_brief_dict.get("core_genre") or "").strip()
            has_fork_proposal = any(
                getattr(p, "action_kind", "") == "pivot_fork_cartridge"
                for p in (final_turn.action_proposals or [])
            )
            is_premise_reboot = has_fork_proposal or bool(
                prev_title
                and new_title
                and prev_title.lower() != new_title.lower()
                and prev_genre
                and new_genre
                and prev_genre.lower() != new_genre.lower()
            )
            if not is_premise_reboot and prev_brief:
                for list_key in ("user_confirmed_pillars", "explicit_non_goals"):
                    merged_items: List[str] = []
                    seen_norm: set[str] = set()
                    for raw_item in list(prev_brief.get(list_key) or []) + list(eff_brief_dict.get(list_key) or []):
                        clean_item = str(raw_item or "").strip()
                        if clean_item and clean_item.lower() not in seen_norm:
                            seen_norm.add(clean_item.lower())
                            merged_items.append(clean_item)
                    eff_brief_dict[list_key] = merged_items
                for carry_key in (
                    "working_title",
                    "core_genre",
                    "elevator_pitch",
                    "core_fantasy",
                    "first_60_seconds_feel",
                    "visual_style_direction",
                    "audio_mood_direction",
                    "core_player_verbs",
                    "color_palette_hex",
                ):
                    if not eff_brief_dict.get(carry_key) and prev_brief.get(carry_key):
                        eff_brief_dict[carry_key] = (
                            list(prev_brief[carry_key])
                            if isinstance(prev_brief[carry_key], list)
                            else prev_brief[carry_key]
                        )

            existing_mocks = list(eff_brief_dict.get("selected_mockup_paths") or prev_brief.get("selected_mockup_paths") or [])
            sketch_list: List[str] = []
            for sk_p in list(prev_brief.get("selected_sketch_asset_paths") or []) + list(eff_brief_dict.get("selected_sketch_asset_paths") or []):
                clean_sk = str(sk_p or "").strip()
                if clean_sk and clean_sk not in sketch_list:
                    sketch_list.append(clean_sk)

            new_turn_mockups: List[str] = []
            for sc in staged_media_cards:
                if sc.card_kind != "sketch_showcase":
                    continue
                for it in sc.items:
                    ap = str(it.artifact_path or "").strip()
                    if not ap:
                        continue
                    if "mockup_" in ap:
                        if ap not in new_turn_mockups:
                            new_turn_mockups.append(ap)
                    elif ap not in sketch_list:
                        sketch_list.append(ap)

            if new_turn_mockups:
                for old_m in existing_mocks:
                    clean_old = str(old_m or "").strip()
                    if clean_old and clean_old not in new_turn_mockups and clean_old not in sketch_list:
                        sketch_list.append(clean_old)
                mock_list = new_turn_mockups
            else:
                mock_list = existing_mocks

            eff_brief_dict["selected_mockup_paths"] = mock_list
            eff_brief_dict["selected_sketch_asset_paths"] = sketch_list
            prev_gcf = prev_brief.get("genre_craft_foundation")
            if not eff_brief_dict.get("genre_craft_foundation") and isinstance(prev_gcf, dict) and prev_gcf:
                eff_brief_dict["genre_craft_foundation"] = dict(prev_gcf)

        is_pre_bible_cartridge = bool(
            str(thread_meta.get("scope_kind") or "") == "cartridge"
            and cid
            and cand_dir is not None
            and not (cand_dir / "game_bible.json").is_file()
            and not (cand_dir / "backlog.json").is_file()
        )
        normalized_proposals: List[Dict[str, Any]] = []
        for idx, prop in enumerate(final_turn.action_proposals or []):
            p_dict = prop.model_dump()
            if not p_dict.get("proposal_id"):
                p_dict["proposal_id"] = f"act_{uuid.uuid4().hex[:8]}_{idx}"
            p_dict["status"] = "pending"
            if is_pre_bible_cartridge and p_dict.get("action_kind") == "commission_game":
                p_dict["action_kind"] = "resume_build"
                if not p_dict.get("button_label") or "commission" in str(p_dict.get("button_label") or "").lower():
                    p_dict["button_label"] = "Start Autonomous Build"
            if p_dict.get("action_kind") in ("commission_game", "pivot_fork_cartridge", "pivot_structural"):
                if p_dict.get("action_kind") in ("commission_game", "pivot_fork_cartridge") and not p_dict.get("handoff_mode"):
                    has_refined = bool(
                        eff_brief_dict
                        and eff_brief_dict.get("working_title")
                        and eff_brief_dict.get("core_player_verbs")
                    )
                    p_dict["handoff_mode"] = "refined_concept" if has_refined else "loose_spark"
                if not p_dict.get("concept_brief") and eff_brief_dict:
                    p_dict["concept_brief"] = eff_brief_dict
                elif isinstance(p_dict.get("concept_brief"), dict) and eff_brief_dict and eff_brief_dict.get("genre_craft_foundation"):
                    if not p_dict["concept_brief"].get("genre_craft_foundation"):
                        p_dict["concept_brief"]["genre_craft_foundation"] = eff_brief_dict["genre_craft_foundation"]
            normalized_proposals.append(p_dict)

        next_title = (final_turn.updated_thread_title or "").strip()
        if not next_title and eff_brief_dict and str(eff_brief_dict.get("working_title") or "").strip():
            if thread_meta.get("title") in ("", "New Game Concept"):
                next_title = str(eff_brief_dict["working_title"]).strip()

        thread_meta, assistant_msg_rec = self.store.append_message(
            thread_meta,
            {
                "deliberation": final_turn.deliberation,
                "sender_role": "coproducer",
                "content_markdown": final_turn.spoken_response,
                "media_cards": merged_media,
                "consultation_cards": [c.model_dump() for c in consultation_cards],
                "action_proposals": normalized_proposals,
                "tool_trace": tool_trace,
                "prompt_trace": prompt_trace,
                "token_usage": cumulative_usage,
            },
            updated_title=next_title or None,
            updated_concept_brief=eff_brief_dict,
        )

        self._emit(
            "coproducer_turn_completed",
            final_turn.deliberation,
            {
                "thread_id": thread_meta["thread_id"],
                "cartridge_id": thread_meta.get("cartridge_id"),
                "message_id": assistant_msg_rec["message_id"],
                "title": thread_meta.get("title"),
                "media_card_count": len(merged_media),
                "consultation_card_count": len(consultation_cards),
                "action_proposal_count": len(normalized_proposals),
                "token_usage": cumulative_usage,
            },
        )

        return _ordered_rationale_dict(
            final_turn.deliberation,
            thread=thread_meta,
            user_message=user_msg_rec,
            assistant_message=assistant_msg_rec,
        )


def record_milestone_notice(
    *,
    cartridges_root: Path,
    cartridge_id: str,
    deliberation: str,
    content_markdown: str,
    db: Optional[Any] = None,
    media_cards: Optional[List[Dict[str, Any]]] = None,
    action_proposals: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Appends a high-signal studio milestone notice (e.g. AWAITING_GREENLIGHT, completed Blind Playtest
    replay, or escalated build blocker) onto `cartridge_id`'s Co-Producer thread."""
    cid = (cartridge_id or "").strip()
    if not cid:
        return None
    store = CoProducerThreadStore(Path(cartridges_root), db=db)
    thread_meta = store.get_or_create_cartridge_thread(cid)
    updated_meta, msg_rec = store.append_message(
        thread_meta,
        {
            "deliberation": deliberation,
            "sender_role": "milestone_notice",
            "content_markdown": content_markdown,
            "media_cards": list(media_cards or []),
            "action_proposals": list(action_proposals or []),
        },
    )
    try:
        from engine.studio.orchestrator import SwarmEventBus

        SwarmEventBus.publish(
            event_type="coproducer_turn_completed",
            agent_name="CoProducerAgent",
            deliberation=deliberation,
            payload=_ordered_rationale_dict(
                deliberation,
                reasoning_rationale=deliberation,
                thread_id=updated_meta["thread_id"],
                cartridge_id=cid,
                message_id=msg_rec["message_id"],
                sender_role="milestone_notice",
            ),
        )
    except Exception:
        pass
    return msg_rec
