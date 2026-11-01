"""Workers for the Production Board: each turns a council work item into real artifacts plus proof images.

Handlers are plain callables `WorkItem -> WorkerResult` so the BoardScheduler can run them concurrently.
Quality judgements stay with the council; workers only (a) interpret the council's spec into the engine's
structured requirement, (b) run the existing fabrication + automated verification machinery, and (c) hand back
the engine's own inspection plates / contact sheets / renders as proof. Every failure raises with its real
error so the board records it (Rule 5): no placeholder assets, no silent fallbacks.
"""
from __future__ import annotations

import json
import logging
import re
import struct
import threading
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from PIL import Image, ImageDraw

from engine.studio.models import (
    AcceptanceCriterion,
    AssetRequirement,
    AudioAssetPlanBatch,
    FilmstripPlan,
    GameRequirementsSpec,
    QAPostCaptureAudit,
    RequirementTrack,
    ScriptWritingDeliverable,
    SpecialistReport,
    StudioAgentError,
    StudioPausedError,
    TaskLeadDecision,
    WorkItem,
    bounded_thinking_config,
    call_gemini_with_transient_retry,
    compact_bible_json_for_prompt,
    is_real_genai_client,
)
from engine.studio.production_board import WorkerResult, WorkerYielded, is_in_board_scheduler

logger = logging.getLogger("BoardWorkers")

# Serialises merges into the shared asset catalog (each worker fabricates with its own library instance).
_CATALOG_LOCK = threading.Lock()
_ACTIVE_TURN_CTX = threading.local()


def record_turn_substep(
    step_type: str,
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    specialist_agent: Optional[str] = None,
) -> None:
    """Appends a live sub-step ('editing_file', 'running_tests', 'generating_asset', 'rendering_scene',
    'curating_context', 'self_inquiry', etc.) to the active turn on the current worker thread, persists
    `lead_trace_rev{N}.json`, and broadcasts `backlog_item_substep` to the live UI."""
    from datetime import datetime, timezone

    state = getattr(_ACTIVE_TURN_CTX, "state", None)
    if not isinstance(state, dict):
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    spec_role = specialist_agent or state.get("specialist_agent") or "task_lead"
    entry: Dict[str, Any] = {
        "timestamp_utc": now_iso,
        "step_type": step_type,
        "specialist_agent": spec_role,
        "message": message,
    }
    if details:
        entry["details"] = details
    step_log = state.setdefault("step_log", [])
    step_log.append(entry)
    persist_fn = state.get("_persist_fn")
    if callable(persist_fn):
        try:
            persist_fn()
        except Exception:
            pass
    ev_cb = state.get("_event_callback")
    if callable(ev_cb):
        try:
            ev_cb(
                "backlog_item_substep",
                {
                    "deliberation": message,
                    "item_id": state.get("item_id", ""),
                    "revision": state.get("revision", 0),
                    "turn_index": state.get("turn_index", 0),
                    "specialist_agent": spec_role,
                    "step_type": step_type,
                    "message": message,
                    "details": details or {},
                    "step_count": len(step_log),
                    "timestamp_utc": now_iso,
                },
            )
        except Exception:
            pass


def record_turn_epistemic_context(
    *,
    role: str = "specialist",
    curated_context: Optional[str] = None,
    prompt_text: Optional[str] = None,
    self_inquiries: Optional[List[Dict[str, Any]]] = None,
    deliberation: Optional[str] = None,
) -> None:
    """Records the curated context, full prompt, and self-inquiry Q&As for the active turn (either Task Lead or Specialist)
    and persists them to `lead_trace_rev{N}.json` for live UI inspection."""
    state = getattr(_ACTIVE_TURN_CTX, "state", None)
    if not isinstance(state, dict):
        return
    if role == "task_lead":
        if curated_context is not None:
            state["lead_curated_context"] = curated_context
        if prompt_text is not None:
            state["lead_prompt_text"] = prompt_text
        if self_inquiries is not None:
            state["lead_self_inquiries"] = list(self_inquiries)
        if deliberation is not None:
            state["lead_deliberation"] = deliberation
    else:
        if curated_context is not None:
            state["curated_context"] = curated_context
        if prompt_text is not None:
            state["prompt_text"] = prompt_text
        if self_inquiries is not None:
            state["self_inquiries"] = list(self_inquiries)
    persist_fn = state.get("_persist_fn")
    if callable(persist_fn):
        try:
            persist_fn()
        except Exception:
            pass

VISUAL_MODES = ("static_sprite", "sprite_sheet", "articulated_parts", "tile_surface", "parallax_layer", "portrait", "diegetic_ui")
ANIMATION_MODES = ("sprite_sheet", "articulated_parts")
AUDIO_MODES = ("audio_bark", "sfx", "ambient_loop")


class CartridgeFileLockTable:
    """Thread-safe per-file lock manager with canonical lexicographical lock ordering
    (preventing deadlocks), dynamic mid-task lock upgrade support, and a short-lived
    `commit_gate` mutex for atomic file writes and headless Godot verification/capture.
    """

    _CANONICAL_ROOT_FILES = frozenset({
        "main.gd",
        "mechanics.gd",
        "scenes.gd",
        "entities.gd",
        "ui_hud.gd",
        "main.tscn",
        "project.godot",
    })

    def __init__(self) -> None:
        self._table_lock = threading.Lock()
        self._owners: Dict[str, str] = {}
        self.commit_gate = threading.RLock()

    @classmethod
    def normalize_path(cls, raw: str) -> str:
        clean = str(raw or "").strip().replace("\\", "/")
        if clean.startswith("res://"):
            clean = clean[len("res://"):]
        elif clean.startswith("res:/"):
            clean = clean[len("res:/"):]
        clean = clean.lstrip("/")
        if "/" in clean and Path(clean).name in cls._CANONICAL_ROOT_FILES:
            clean = Path(clean).name
        return clean

    def try_acquire_many(self, owner_id: str, paths: Sequence[str]) -> tuple[bool, List[str]]:
        """Atomically acquires all `paths` in canonical sorted order for `owner_id`.
        If any path is currently held by another owner, returns `(False, conflicting_paths)`
        without mutating existing lock ownership.
        """
        canon = sorted({self.normalize_path(p) for p in (paths or []) if self.normalize_path(p)})
        if not canon:
            return True, []
        with self._table_lock:
            conflicts = [
                p for p in canon
                if p in self._owners and self._owners[p] != owner_id
            ]
            if conflicts:
                return False, conflicts
            for p in canon:
                self._owners[p] = owner_id
            return True, []

    def held_paths(self, owner_id: str) -> set[str]:
        with self._table_lock:
            return {p for p, o in self._owners.items() if o == owner_id}

    def all_held_paths(self) -> set[str]:
        with self._table_lock:
            return set(self._owners.keys())

    def release_all(self, owner_id: str) -> None:
        with self._table_lock:
            to_remove = [p for p, o in self._owners.items() if o == owner_id]
            for p in to_remove:
                del self._owners[p]


@dataclass
class WorkerContext:
    """Everything workers need; `library_factory` lets tests inject a fake library."""
    cartridge_dir: Path
    client: Any = None
    visual_agent: Any = None
    voice_agent: Any = None
    swe_agent: Any = None
    model_name: str = "gemini-3.8-flash"
    library_factory: Optional[Callable[..., Any]] = None
    code_verify: Optional[Callable[[Path, Any], None]] = None     # (cartridge_dir, track) -> raises on failure
    filmstrip_fn: Optional[Callable[..., Path]] = None            # injectable for tests; defaults to proofs.capture_filmstrip
    code_lock: Any = None                                         # shared threading.Lock serialising whole-project code work
    file_lock_table: Optional[CartridgeFileLockTable] = None      # optional per-file lock table for parallel modular SWE work
    board: Any = None                                             # optional ProductionBoard for dynamic task spawning (Tasks -> More Tasks)
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


    def __post_init__(self) -> None:
        if self.code_lock is None:
            self.code_lock = threading.Lock()

    def new_library(self) -> Any:
        if self.library_factory is not None:
            return self.library_factory(self)
        from engine.studio.asset_library import CartridgeAssetLibrary

        return CartridgeAssetLibrary(
            cartridge_dir=self.cartridge_dir,
            visual_agent=self.visual_agent,
            voice_agent=self.voice_agent,
            client=self.client,
        )


def _safe_id(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_") or "asset"


def _item_dir(ctx: WorkerContext, item: WorkItem) -> Path:
    d = ctx.cartridge_dir / "artifacts" / "work_items" / _safe_id(item.item_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_live(client: Any) -> bool:
    return client is not None and getattr(type(client), "__module__", "").startswith("google.genai")


def _format_worker_failure_excerpt(failure_text: str, max_chars: int = 1000) -> str:
    """Extracts the authentic exception message from a formatted worker traceback starting at the exception headline,
    avoiding blind tail-slicing that chops off the root error description when messages exceed `max_chars`."""
    raw = str(failure_text or "").strip()
    if not raw:
        return ""
    lines = raw.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith("ACTION REQUIRED FOR TASK LEAD"):
            continue
        if "Error:" in stripped or "Exception:" in stripped:
            joined = "\n".join(lines[idx:]).strip()
            return joined[:max_chars]
    return raw[:max_chars] if len(raw) <= max_chars else raw[-max_chars:]


def plan_asset_requirement(
    ctx: WorkerContext,
    item: WorkItem,
    allowed_modes: tuple,
    attempts: int = 3,
) -> AssetRequirement:
    """Interprets a council spec (plus any revision feedback) as the engine's structured AssetRequirement via an LLM
    contract (rationale first). Validation issues are fed back; exhausting attempts raises (Rule 5)."""
    if not _is_live(ctx.client):
        raise StudioAgentError("Asset planning requires an active google.genai Client (Rule 5)")
    from google.genai import types

    from engine.studio.context_curator import ContextCuratorAgent, is_real_genai_client
    from engine.studio.rendering import build_rendering_guide, load_rendering_strategy

    strategy = load_rendering_strategy(ctx.cartridge_dir)
    rendering_block = build_rendering_guide(strategy) if strategy is not None else ""
    bible_path = ctx.cartridge_dir / "game_bible.json"
    bible_mode = "audio_asset" if item.kind == "audio_asset" else "visual_asset"
    bible_text = (
        compact_bible_json_for_prompt(bible_path.read_text(encoding="utf-8"), mode=bible_mode)
        if bible_path.is_file()
        else "{}"
    )

    prior_req = _item_dir(ctx, item) / "requirement.json"
    prior_block = ""
    if prior_req.is_file():
        prior_block = f"PREVIOUS REQUIREMENT YOU PRODUCED FOR THIS ITEM (revise it per the feedback):\n{prior_req.read_text(encoding='utf-8')}\n"
    feedback_block = ""
    fb = [(r.revision_index + 1, r.feedback) for r in item.revisions if r.feedback]
    if fb:
        feedback_block = "COUNCIL FEEDBACK ON EARLIER REVISIONS (address ALL of it):\n" + "\n".join(f"  rev {n}: {t}" for n, t in fb) + "\n"
    if getattr(item, "failure", ""):
        feedback_block += f"PREVIOUS WORKER EXECUTION ERROR TO FIX:\n{_format_worker_failure_excerpt(item.failure, 1000)}\n"

    curated_block = f"{rendering_block}\n--- GAME BIBLE ---\n{bible_text}\n"
    multimodal_parts: List[Any] = []
    target_role = "audio_designer" if item.kind == "audio_asset" else "visual_artist"
    from engine.studio.game_memory import GameMemoryStore, format_memory_query_results

    mem_store = GameMemoryStore(ctx.cartridge_dir, event_callback=ctx.event_callback, client=ctx.client)
    has_embed_client = callable(getattr(getattr(ctx.client, "models", None), "embed_content", None))
    if is_real_genai_client(ctx.client):
        curator = ContextCuratorAgent(
            cartridge_dir=ctx.cartridge_dir,
            client=ctx.client,
            model_name=ctx.model_name,
            board=ctx.board,
            event_callback=ctx.event_callback,
        )
        pack = curator.curate_context(
            target_agent_role=target_role,
            task_request=f"{item.title}\nSPEC: {item.spec}\nPROOF: {item.proof_spec}\n{feedback_block}",
            item_id=item.item_id,
            item_kind=item.kind,
        )
        curated_block = f"{pack.prompt_block}\n{rendering_block}\n"
        multimodal_parts = list(pack.multimodal_parts)
    elif (ctx.cartridge_dir / "game_memory.json").is_file():
        role_mem = mem_store.format_role_memory_briefing(target_role)
        if role_mem:
            curated_block = f"{role_mem}\n{curated_block}"

    issues: List[str] = []
    mem_query_blocks: List[str] = []
    asked_mem_queries: set[str] = set()
    mem_queries_used = 0
    max_mem_queries = 2 if has_embed_client else 0
    attempt = 1
    while attempt <= attempts:
        retry = "YOUR PREVIOUS ANSWER FAILED VALIDATION; FIX:\n- " + "\n- ".join(issues) + "\n\n" if issues else ""
        mem_q_block = ("\n" + "\n".join(mem_query_blocks)) if mem_query_blocks else ""
        prompt = (
            f"{retry}You are the Visual/Audio Production Lead turning a council requirement into a precise `AssetRequirement`.\n"
            f"COUNCIL REQUIREMENT [{item.item_id}] ({item.kind}): {item.title}\nSPEC: {item.spec}\nPROOF THE COUNCIL WILL JUDGE: {item.proof_spec}\n"
            f"{feedback_block}{prior_block}"
            f"Set `asset_id` to exactly '{item.item_id}'. `fabrication_mode` must be one of {list(allowed_modes)}; `asset_type`, `target_path` "
            f"(under assets/), `dimensions` (the rendering strategy's resolution for this subject), `visual_prompt_or_text` and, for rigs, "
            f"`articulated_parts` must be consistent with it and with the style, camera and palette in the Game Bible.\n"
            f"CRITICAL MODE & PROMPT RULES FOR `visual_prompt_or_text`:\n"
            f"- Avoid `fabrication_mode == 'tile_surface'` for anything that has a silhouette, alpha transparency, endcaps, or distinct shape (such as channels, slabs, platforms, machines, props, or fluid bodies) — `tile_surface` is ONLY for 100% opaque, borderless, full-bleed square surface material swatches. Use `static_sprite` (with `asset_type='prop'` or `'sprite'`), `sprite_sheet`, or `articulated_parts` for any shaped element, structure, or multi-state object so its alpha silhouette is preserved cleanly.\n"
            f"- When `fabrication_mode == 'sprite_sheet'` AND the asset is an animated character/actor whose poses are deformed from a single base figure, `visual_prompt_or_text` is sent to the image generator to synthesize the SINGLE canonical neutral standing/idle figure, and our biomechanical keyframe engine then generates the multi-row, multi-frame sprite sheet (`*_sheet.png`) from `animation_states` (leave `grid_cols=1, grid_rows=1, cell_prompts=[]`).\n"
            f"- MULTI-CELL GRID / STATE SHEET / ATLAS COMPOSITION (`grid_cols`, `grid_rows`, `cell_prompts`): Whenever the Task Lead brief or task spec requests a multi-cell grid, sheet, or atlas of distinct visual states, icons, stages, or variants (e.g., a 2x2 or NxM sheet where each cell depicts a distinct state, graphic, or component rather than biomechanical deformation of one walking figure), set `grid_cols` and `grid_rows` to the exact grid dimensions (e.g. `grid_cols=2, grid_rows=2`) and populate `cell_prompts` with exactly `grid_cols * grid_rows` focused visual descriptions in row-major order (top-left to bottom-right), where EACH string in `cell_prompts` describes the single isolated graphic/state that belongs inside that specific cell!\n"
            f"- When `grid_cols == 1` and `grid_rows == 1` (single isolated asset), `visual_prompt_or_text` MUST describe ONLY ONE single isolated object/prop in its canonical intact state — NEVER a multi-state variation grid.\n"
            f"- When `fabrication_mode == 'audio_bark'`, `visual_prompt_or_text` is sent directly to the neural TTS engine to synthesize ONE single spoken voice line into `target_path` (strictly bounded to [0.1s, 30.0s], max 45 words / 280 characters of direct spoken dialogue). NEVER concatenate multiple barks, speaker labels ('Kaelen barks:', 'Overseer Vhen:'), or parenthetical stage directions into `visual_prompt_or_text`! Write ONLY the exact spoken words of a single representative bark (1–2 sentences, <= 45 words).\n"
            f"Provide `deliberation` FIRST.\n\n{curated_block}{mem_q_block}\n{retry}"
        )
        contents_arg: Any = ([prompt] + multimodal_parts) if multimodal_parts else prompt
        record_turn_epistemic_context(
            role="specialist",
            curated_context=curated_block,
            prompt_text=prompt,
        )
        resp = call_gemini_with_transient_retry(
            lambda: ctx.client.models.generate_content(
                model=ctx.model_name,
                contents=contents_arg,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AssetRequirement,
                    thinking_config=bounded_thinking_config(1536),
                ),
            ),
            label=f"plan_asset_requirement[{item.item_id}]",
        )
        if not resp.text:
            raise StudioAgentError(f"Asset planning for '{item.item_id}' returned an empty response")
        req = AssetRequirement.model_validate_json(resp.text)
        if req.memory_updates:
            mem_store.apply_mutations(
                author_role=target_role,
                proposals=req.memory_updates,
                item_id=item.item_id,
            )
            req.memory_updates = []
        mem_q = (req.memory_query or "").strip()
        mem_q_key = f"{(req.memory_query_bank or '').strip().lower()}::{mem_q.lower()}"
        if (
            mem_q
            and not (req.visual_prompt_or_text or "").strip()
            and has_embed_client
            and mem_queries_used < max_mem_queries
            and mem_q_key not in asked_mem_queries
        ):
            asked_mem_queries.add(mem_q_key)
            mem_queries_used += 1
            hits = mem_store.query_memories(
                caller_role=target_role,
                query_text=mem_q,
                bank_filter=req.memory_query_bank,
                client=ctx.client,
            )
            mem_query_blocks.append(format_memory_query_results(mem_q, req.memory_query_bank, hits))
            issues = []
            continue
        req.memory_query = ""
        req.memory_query_bank = ""
        req.asset_id = item.item_id
        issues = []
        if req.fabrication_mode not in allowed_modes:
            issues.append(f"fabrication_mode '{req.fabrication_mode}' must be one of {list(allowed_modes)}.")
        if not req.target_path.strip():
            issues.append("target_path is required (under assets/).")
        elif not req.target_path.startswith("assets/") or ".." in req.target_path:
            issues.append("target_path must be a relative path under assets/.")
        if req.fabrication_mode == "articulated_parts" and not req.articulated_parts:
            issues.append("articulated_parts mode requires `articulated_parts`.")
        if req.cell_prompts:
            expected_cells = max(1, int(req.grid_cols or 1)) * max(1, int(req.grid_rows or 1))
            if expected_cells <= 1 or len(req.cell_prompts) != expected_cells:
                issues.append(
                    f"When `cell_prompts` is populated ({len(req.cell_prompts)} items), `grid_cols * grid_rows` "
                    f"({req.grid_cols}x{req.grid_rows}={expected_cells}) must be > 1 and equal `len(cell_prompts)`."
                )
        if req.fabrication_mode == "audio_bark":
            words = (req.visual_prompt_or_text or "").strip().split()
            chars = len((req.visual_prompt_or_text or "").strip())
            if len(words) > 50 or chars > 320:
                issues.append(
                    f"For fabrication_mode='audio_bark', visual_prompt_or_text must be a SINGLE concise spoken voice line "
                    f"(<= 50 words and <= 320 characters so TTS duration stays well under the 30.0s limit); got {len(words)} words "
                    f"({chars} chars). Do NOT concatenate multiple barks or speaker labels into one string."
                )
        if not issues:
            # Ensure target_path cannot accidentally collide with another task's asset file,
            # while allowing intentional in-place regeneration when `req.target_path` is explicitly
            # targeted by the WorkItem (`target_modules`, `spec`, or `title`) and no concurrent task owns it.
            safe_item_id = _safe_id(item.item_id)
            norm_req_path = req.target_path.replace("res://", "").lstrip("/")
            explicit_target_blob = "\n".join([
                str(item.title or ""),
                str(item.spec or ""),
                *(str(m or "") for m in (item.target_modules or [])),
            ])
            explicitly_targets_path = bool(
                norm_req_path and (
                    norm_req_path in explicit_target_blob
                    or f"res://{norm_req_path}" in explicit_target_blob
                )
            )
            active_other_ids: Set[str] = set()
            active_states = {
                "ready", "blocked", "in_progress", "waiting_for_lock",
                "awaiting_review", "changes_requested", "needs_decision",
            }
            if ctx.board is not None and hasattr(ctx.board, "items"):
                try:
                    for wi in ctx.board.items():
                        if getattr(wi, "item_id", "") != item.item_id and str(getattr(wi, "state", "")).lower() in active_states:
                            active_other_ids.add(str(wi.item_id))
                except Exception:
                    pass
            elif (ctx.cartridge_dir / "backlog.json").is_file():
                try:
                    b_raw = json.loads((ctx.cartridge_dir / "backlog.json").read_text(encoding="utf-8"))
                    for wi in (b_raw.get("items") or []):
                        if isinstance(wi, dict) and str(wi.get("item_id", "")) != item.item_id and str(wi.get("state", "")).lower() in active_states:
                            active_other_ids.add(str(wi["item_id"]))
                except Exception:
                    pass

            for cat_cand in (ctx.cartridge_dir / "catalog.json", ctx.cartridge_dir / "assets" / "asset_library.json"):
                if cat_cand.is_file():
                    try:
                        cat_data = json.loads(cat_cand.read_text(encoding="utf-8"))
                        entries = cat_data.get("entries", cat_data) if isinstance(cat_data, dict) else {}
                        if isinstance(entries, dict):
                            for other_id, entry_val in entries.items():
                                if other_id == item.item_id or not isinstance(entry_val, dict):
                                    continue
                                other_file = entry_val.get("file") or entry_val.get("file_path") or entry_val.get("target_path")
                                if other_file == req.target_path:
                                    if explicitly_targets_path and str(other_id) not in active_other_ids:
                                        continue
                                    tp = Path(req.target_path)
                                    req.target_path = str(tp.parent / f"{safe_item_id}_{tp.name}")
                                    break
                    except Exception:
                        pass
            if not req.name:
                req.name = Path(req.target_path).name
            prior_req.write_text(req.model_dump_json(indent=2), encoding="utf-8")
            return req
        logger.info("Asset plan attempt %d/%d for %s invalid: %s", attempt, attempts, item.item_id, issues)
        attempt += 1
    raise StudioAgentError(f"Asset planning for '{item.item_id}' still invalid after {attempts} attempts: {issues}")


def _collect_proofs(renders_dir: Path, asset_id: str, asset_file: Optional[Path]) -> List[str]:
    candidates = [
        renders_dir / f"plate_{asset_id}.png",
        renders_dir / f"anim_sheet_{asset_id}.png",
        renders_dir / f"anim_rig_{asset_id}.png",
    ]
    proofs = [str(p) for p in candidates if p.is_file()]
    if asset_file is not None and asset_file.is_file() and asset_file.suffix.lower() in (".png", ".jpg", ".jpeg") and str(asset_file) not in proofs:
        proofs.append(str(asset_file))
    return proofs


def _merge_catalog_entry(ctx: WorkerContext, worker_lib: Any, asset_id: str) -> None:
    """Merges the worker's catalog entry into the shared on-disk catalog without losing concurrent workers' updates,
    superseding any stale catalog entry that previously owned the exact same relative file path."""
    entry = worker_lib.catalog.entries.get(asset_id)
    if entry is None:
        raise StudioAgentError(f"Fabrication of '{asset_id}' produced no catalog entry")
    entry_file = (
        (entry.get("file") or entry.get("file_path") or entry.get("target_path") or "")
        if isinstance(entry, dict)
        else (getattr(entry, "file_path", "") or getattr(entry, "res_path", "") or "")
    )
    norm_entry_file = str(entry_file).replace("res://", "").lstrip("/")
    is_visual_file = norm_entry_file.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    with _CATALOG_LOCK:
        shared = ctx.new_library()
        if is_visual_file and isinstance(shared.catalog.entries, dict):
            stale_ids = []
            for oid, oentry in list(shared.catalog.entries.items()):
                if oid == asset_id:
                    continue
                ofile = (
                    (oentry.get("file") or oentry.get("file_path") or oentry.get("target_path") or "")
                    if isinstance(oentry, dict)
                    else (getattr(oentry, "file_path", "") or getattr(oentry, "res_path", "") or "")
                )
                if str(ofile).replace("res://", "").lstrip("/") == norm_entry_file:
                    stale_ids.append(oid)
            for oid in stale_ids:
                shared.catalog.entries.pop(oid, None)
        shared.catalog.entries[asset_id] = entry
        shared.save_catalog()


def _fabricate_visual(
    ctx: WorkerContext,
    item: WorkItem,
    allowed_modes: tuple,
    force_regenerate: Optional[bool] = None,
) -> WorkerResult:
    req = plan_asset_requirement(ctx, item, allowed_modes)
    record_turn_substep(
        "generating_asset",
        f"Generating visual asset {req.target_path} (mode={req.fabrication_mode}, size={req.dimensions})",
        details={"asset_id": req.asset_id, "target_path": req.target_path, "mode": req.fabrication_mode},
    )
    lib = ctx.new_library()
    eff_force = True if force_regenerate is None else bool(force_regenerate)
    out = lib.fabricate_single_asset(req, force_regenerate=eff_force)
    if out is None or not Path(out).is_file():
        raise StudioAgentError(f"Fabrication of '{item.item_id}' produced no file (mode={req.fabrication_mode}).")
    _merge_catalog_entry(ctx, lib, req.asset_id)
    renders = ctx.cartridge_dir / "artifacts" / "renders"
    proofs = _collect_proofs(renders, req.asset_id, Path(out))
    record_turn_substep(
        "captured_proof",
        f"Rendered visual inspection plate for {req.target_path}",
        details={"proof_images": proofs},
    )
    if not proofs:
        raise StudioAgentError(f"Fabrication of '{item.item_id}' produced no proof images.")
    entry = lib.catalog.entries.get(req.asset_id)
    sheet_meta_note = ""
    if entry is not None and (
        getattr(entry, "fabrication_mode", "") == "sprite_sheet"
        or int(getattr(entry, "hframes", 1) or 1) > 1
        or int(getattr(entry, "vframes", 1) or 1) > 1
    ):
        sheet_meta_note = (
            f" (sheet_res_path='{getattr(entry, 'res_path', '')}', "
            f"hframes={getattr(entry, 'hframes', 1)}, vframes={getattr(entry, 'vframes', 1)}, "
            f"tex_size={getattr(entry, 'texture_pixel_size', ())}, "
            f"display_size={getattr(entry, 'target_display_size', req.dimensions)})"
        )
    notes = (
        f"Fabricated '{req.asset_id}' as {req.fabrication_mode} ({req.asset_type}) at {req.dimensions} -> {req.target_path}{sheet_meta_note}. "
        f"Planning rationale: {req.deliberation[:400]}"
    )
    return WorkerResult(proof_images=proofs, notes=notes)


def visual_asset_handler(ctx: WorkerContext) -> Callable[[WorkItem], WorkerResult]:
    return lambda item: _fabricate_visual(ctx, item, VISUAL_MODES)


def animation_handler(ctx: WorkerContext) -> Callable[[WorkItem], WorkerResult]:
    return lambda item: _fabricate_visual(ctx, item, ANIMATION_MODES)


def plan_audio_requirements(
    ctx: WorkerContext,
    item: WorkItem,
    allowed_modes: tuple = AUDIO_MODES,
    attempts: int = 3,
) -> List[AssetRequirement]:
    """Interprets an audio work item (plus any Task Lead brief and Council revision feedback) into one or more
    structured `AssetRequirement` items via `AudioAssetPlanBatch` (rationale first). When the specification or brief
    requests multiple audio cues / a soundbank, plans a separate `AssetRequirement` for every requested `.wav` file."""
    if not _is_live(ctx.client):
        raise StudioAgentError("Audio asset planning requires an active google.genai Client (Rule 5)")
    from google.genai import types

    from engine.studio.context_curator import ContextCuratorAgent, is_real_genai_client

    bible_path = ctx.cartridge_dir / "game_bible.json"
    bible_text = (
        compact_bible_json_for_prompt(bible_path.read_text(encoding="utf-8"), mode="audio_asset")
        if bible_path.is_file()
        else "{}"
    )

    prior_req = _item_dir(ctx, item) / "requirement.json"
    prior_block = ""
    if prior_req.is_file():
        prior_block = f"PREVIOUS REQUIREMENT YOU PRODUCED FOR THIS ITEM (revise it per the feedback):\n{prior_req.read_text(encoding='utf-8')}\n"
    feedback_block = ""
    fb = [(r.revision_index + 1, r.feedback) for r in item.revisions if r.feedback]
    if fb:
        feedback_block = "COUNCIL FEEDBACK ON EARLIER REVISIONS (address ALL of it):\n" + "\n".join(f"  rev {n}: {t}" for n, t in fb) + "\n"
    if getattr(item, "failure", ""):
        feedback_block += f"PREVIOUS WORKER EXECUTION ERROR TO FIX:\n{_format_worker_failure_excerpt(item.failure, 1000)}\n"

    curated_block = f"--- GAME BIBLE ---\n{bible_text}\n"
    multimodal_parts: List[Any] = []
    from engine.studio.game_memory import GameMemoryStore, format_memory_query_results

    mem_store = GameMemoryStore(ctx.cartridge_dir, event_callback=ctx.event_callback, client=ctx.client)
    has_embed_client = callable(getattr(getattr(ctx.client, "models", None), "embed_content", None))
    if is_real_genai_client(ctx.client):
        curator = ContextCuratorAgent(
            cartridge_dir=ctx.cartridge_dir,
            client=ctx.client,
            model_name=ctx.model_name,
            board=ctx.board,
            event_callback=ctx.event_callback,
        )
        pack = curator.curate_context(
            target_agent_role="audio_designer",
            task_request=f"{item.title}\nSPEC: {item.spec}\nPROOF: {item.proof_spec}\n{feedback_block}",
            item_id=item.item_id,
            item_kind=item.kind,
        )
        curated_block = f"{pack.prompt_block}\n"
        multimodal_parts = list(pack.multimodal_parts)
    elif (ctx.cartridge_dir / "game_memory.json").is_file():
        role_mem = mem_store.format_role_memory_briefing("audio_designer")
        if role_mem:
            curated_block = f"{role_mem}\n{curated_block}"

    issues: List[str] = []
    mem_query_blocks: List[str] = []
    asked_mem_queries: set[str] = set()
    mem_queries_used = 0
    max_mem_queries = 2 if has_embed_client else 0
    attempt = 1
    while attempt <= attempts:
        retry = "YOUR PREVIOUS ANSWER FAILED VALIDATION; FIX:\n- " + "\n- ".join(issues) + "\n\n" if issues else ""
        mem_q_block = ("\n" + "\n".join(mem_query_blocks)) if mem_query_blocks else ""
        prompt = (
            f"{retry}You are the Audio Production Lead turning a council audio requirement (and any Task Lead brief) into an `AudioAssetPlanBatch`.\n"
            f"COUNCIL REQUIREMENT [{item.item_id}] ({item.kind}): {item.title}\nSPEC: {item.spec}\nPROOF THE COUNCIL WILL JUDGE: {item.proof_spec}\n"
            f"{feedback_block}{prior_block}"
            f"CRITICAL MULTI-ASSET SOUNDBANK RULE:\n"
            f"- Carefully read `SPEC`, `PROOF THE COUNCIL WILL JUDGE`, `COUNCIL FEEDBACK ON EARLIER REVISIONS`, and any `TASK LEAD BRIEF FOR AUDIO DESIGNER` / `Requested target path(s)`.\n"
            f"- If the specification or brief requests a SINGLE audio file, populate `assets` with 1 `AssetRequirement`.\n"
            f"- If the specification or brief enumerates MULTIPLE audio cues or a soundbank (e.g., 2 or more named `.wav` files or numbered audio events), you MUST populate `assets` with ONE separate `AssetRequirement` for **EVERY SINGLE requested audio cue** so that each cue is synthesized into its own `.wav` file on disk! Never collapse a multi-cue soundbank into a single `.wav` file.\n"
            f"- For EACH `AssetRequirement` in `assets`:\n"
            f"  * `fabrication_mode` must be one of {list(allowed_modes)} (`'sfx'` for one-shot effects, `'ambient_loop'` for continuous seamless loops, `'audio_bark'` for spoken voice lines).\n"
            f"  * `target_path` must be a unique relative `.wav` path under `assets/` (e.g. `assets/audio/.../*.wav`) matching that cue's requested filename.\n"
            f"  * `asset_id` should be `{item.item_id}` when `len(assets) == 1`, or the unique stem of `target_path` when `len(assets) > 1`.\n"
            f"  * `visual_prompt_or_text` must describe ONLY that specific cue's acoustic texture, frequency band, envelope, or spoken dialogue.\n"
            f"  * When `fabrication_mode == 'audio_bark'`, `visual_prompt_or_text` is sent directly to the neural TTS engine to synthesize ONE single spoken voice line into `target_path` (strictly bounded to [0.1s, 30.0s], max 45 words / 280 characters of direct spoken dialogue). NEVER concatenate multiple barks, speaker labels ('Kaelen barks:', 'Overseer Vhen:'), or parenthetical stage directions into `visual_prompt_or_text`!\n"
            f"Provide `deliberation` FIRST.\n\n{curated_block}{mem_q_block}\n{retry}"
        )
        contents_arg: Any = ([prompt] + multimodal_parts) if multimodal_parts else prompt
        record_turn_epistemic_context(
            role="specialist",
            curated_context=curated_block,
            prompt_text=prompt,
        )
        resp = call_gemini_with_transient_retry(
            lambda: ctx.client.models.generate_content(
                model=ctx.model_name,
                contents=contents_arg,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AudioAssetPlanBatch,
                    thinking_config=bounded_thinking_config(1536),
                ),
            ),
            label=f"plan_audio_requirements[{item.item_id}]",
        )
        if not resp.text:
            raise StudioAgentError(f"Audio asset planning for '{item.item_id}' returned an empty response")
        raw_obj = json.loads(resp.text)
        if isinstance(raw_obj, dict) and (
            isinstance(raw_obj.get("assets"), list)
            or "memory_query" in raw_obj
            or "memory_updates" in raw_obj
        ):
            batch = AudioAssetPlanBatch.model_validate_json(resp.text)
            if batch.memory_updates:
                mem_store.apply_mutations(
                    author_role="audio_designer",
                    proposals=batch.memory_updates,
                    item_id=item.item_id,
                )
            mem_q = (batch.memory_query or "").strip()
            mem_q_key = f"{(batch.memory_query_bank or '').strip().lower()}::{mem_q.lower()}"
            if (
                mem_q
                and not batch.assets
                and has_embed_client
                and mem_queries_used < max_mem_queries
                and mem_q_key not in asked_mem_queries
            ):
                asked_mem_queries.add(mem_q_key)
                mem_queries_used += 1
                hits = mem_store.query_memories(
                    caller_role="audio_designer",
                    query_text=mem_q,
                    bank_filter=batch.memory_query_bank,
                    client=ctx.client,
                )
                mem_query_blocks.append(format_memory_query_results(mem_q, batch.memory_query_bank, hits))
                issues = []
                continue
            reqs = list(batch.assets)
        else:
            reqs = [AssetRequirement.model_validate_json(resp.text)]

        issues = []
        if not reqs:
            issues.append("`assets` must contain at least one `AssetRequirement`.")
        seen_paths: set = set()
        for idx, req in enumerate(reqs):
            if req.fabrication_mode not in allowed_modes:
                issues.append(f"Asset #{idx + 1} fabrication_mode '{req.fabrication_mode}' must be one of {list(allowed_modes)}.")
            tpath = (req.target_path or "").strip().replace("\\", "/")
            if tpath.startswith("res://"):
                tpath = tpath[len("res://"):]
            elif tpath.startswith("res:/"):
                tpath = tpath[len("res:/"):]
            tpath = tpath.lstrip("/")
            if not tpath:
                issues.append(f"Asset #{idx + 1} target_path is required (under assets/).")
            elif not tpath.startswith("assets/") or ".." in tpath:
                issues.append(f"Asset #{idx + 1} target_path '{tpath}' must be a relative path under assets/.")
            else:
                if not tpath.lower().endswith(".wav"):
                    tpath = str(Path(tpath).with_suffix(".wav"))
                req.target_path = tpath
                if tpath in seen_paths:
                    issues.append(
                        f"Duplicate target_path '{tpath}' in `assets`; each audio cue in the batch must have its own distinct `.wav` target_path."
                    )
                seen_paths.add(tpath)
            if req.fabrication_mode == "audio_bark":
                words = (req.visual_prompt_or_text or "").strip().split()
                chars = len((req.visual_prompt_or_text or "").strip())
                if len(words) > 50 or chars > 320:
                    issues.append(
                        f"Asset #{idx + 1} ('{req.target_path}'): for fabrication_mode='audio_bark', visual_prompt_or_text must be a SINGLE concise spoken voice line "
                        f"(<= 50 words and <= 320 characters); got {len(words)} words ({chars} chars)."
                    )
        if not issues:
            safe_item_id = _safe_id(item.item_id)
            used_ids: set = set()
            for idx, req in enumerate(reqs):
                if len(reqs) == 1:
                    req.asset_id = item.item_id
                else:
                    stem_id = _safe_id(Path(req.target_path).stem)
                    cand_id = stem_id if (stem_id and stem_id != "asset") else f"{safe_item_id}_{idx + 1}"
                    if cand_id in used_ids:
                        cand_id = f"{cand_id}_{idx + 1}"
                    req.asset_id = cand_id
                used_ids.add(req.asset_id)
                for cat_cand in (ctx.cartridge_dir / "catalog.json", ctx.cartridge_dir / "assets" / "asset_library.json"):
                    if cat_cand.is_file():
                        try:
                            cat_data = json.loads(cat_cand.read_text(encoding="utf-8"))
                            entries = cat_data.get("entries", cat_data) if isinstance(cat_data, dict) else {}
                            if isinstance(entries, dict):
                                for other_id, entry_val in entries.items():
                                    if other_id in (item.item_id, req.asset_id) or not isinstance(entry_val, dict):
                                        continue
                                    other_file = entry_val.get("file") or entry_val.get("file_path") or entry_val.get("target_path")
                                    if other_file == req.target_path:
                                        tp = Path(req.target_path)
                                        req.target_path = str(tp.parent / f"{safe_item_id}_{tp.name}")
                                        break
                        except Exception:
                            pass
                if not req.name:
                    req.name = Path(req.target_path).name
            if len(reqs) == 1:
                prior_req.write_text(reqs[0].model_dump_json(indent=2), encoding="utf-8")
            else:
                prior_req.write_text(
                    AudioAssetPlanBatch(
                        deliberation=reqs[0].deliberation,
                        assets=reqs,
                    ).model_dump_json(indent=2),
                    encoding="utf-8",
                )
            return reqs
        logger.info("Audio asset plan attempt %d/%d for %s invalid: %s", attempt, attempts, item.item_id, issues)
        attempt += 1
    raise StudioAgentError(f"Audio asset planning for '{item.item_id}' still invalid after {attempts} attempts: {issues}")


def render_waveform_image(
    wav_path: Union[Path, Sequence[Path]],
    out_path: Path,
    size: tuple = (900, 220),
) -> Path:
    """Draws time-domain waveform(s) + FFT frequency-band spectrum + duration/peak/spectral telemetry so the Council
    can visually inspect single audio assets or multi-cue soundbanks."""
    if isinstance(wav_path, (str, Path)):
        wav_paths = [Path(wav_path)]
    else:
        wav_paths = [Path(p) for p in wav_path if p]
    if not wav_paths:
        raise StudioAgentError("render_waveform_image requires at least one WAV file path.")

    w = int(size[0])
    panel_h = int(size[1]) if len(wav_paths) == 1 else max(165, int(size[1]) - 30)
    total_h = panel_h * len(wav_paths)
    img = Image.new("RGB", (w, total_h), (16, 18, 28))
    draw = ImageDraw.Draw(img)

    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore[assignment]

    for idx, wp in enumerate(wav_paths):
        y_top = idx * panel_h
        y_bot = y_top + panel_h
        if idx > 0:
            draw.line([(0, y_top), (w, y_top)], fill=(56, 70, 105), width=2)
        with wave.open(str(wp), "rb") as wf:
            n_frames, rate, width, channels = wf.getnframes(), wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
            raw = wf.readframes(n_frames)
        if width != 2:
            raise StudioAgentError(f"Unsupported WAV sample width {width} for {wp}")
        samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)[::channels] if raw else ()
        duration = n_frames / max(1, rate)
        peak = max((abs(s) for s in samples), default=0) / 32768.0

        wave_w = int(w * 0.66) if np is not None and samples else w
        mid = y_top + (panel_h // 2) + 8
        amp_half = max(16, (panel_h // 2) - 24)
        draw.line([(0, mid), (wave_w - 8, mid)], fill=(60, 64, 90))
        if samples:
            step = max(1, len(samples) // max(1, wave_w - 12))
            for x in range(wave_w - 12):
                chunk = samples[x * step:(x + 1) * step]
                if not chunk:
                    break
                lo, hi = min(chunk) / 32768.0, max(chunk) / 32768.0
                draw.line(
                    [(x + 6, mid - int(hi * amp_half)), (x + 6, mid - int(lo * amp_half))],
                    fill=(120, 220, 255),
                )

        # Compute & draw FFT frequency-band spectrum on the right panel when numpy is available
        spec_summary = ""
        if np is not None and len(samples) >= 64:
            arr = np.asarray(samples, dtype=np.float32) / 32768.0
            arr = arr - float(np.mean(arr))
            window = np.hanning(len(arr))
            fft_mag = np.abs(np.fft.rfft(arr * window))
            freqs = np.fft.rfftfreq(len(arr), d=1.0 / max(1, rate))
            total_mag = float(np.sum(fft_mag)) or 1.0
            centroid_hz = float(np.sum(freqs * fft_mag) / total_mag)
            dom_idx = int(np.argmax(fft_mag[1:]) + 1) if len(fft_mag) > 1 else 0
            dom_hz = float(freqs[dom_idx]) if dom_idx < len(freqs) else 0.0
            spec_summary = f"  |  dom={dom_hz:.0f}Hz  centroid={centroid_hz:.0f}Hz"

            bands = [
                ("<150", 20.0, 150.0, (244, 114, 182)),
                ("150-400", 150.0, 400.0, (251, 146, 60)),
                ("0.4-1k", 400.0, 1000.0, (250, 204, 21)),
                ("1-2.5k", 1000.0, 2500.0, (74, 222, 128)),
                ("2.5-6k", 2500.0, 6000.0, (56, 189, 248)),
                ("6k+", 6000.0, max(6500.0, rate / 2.0), (167, 139, 250)),
            ]
            band_energies: List[float] = []
            for _, f_lo, f_hi, _ in bands:
                mask = (freqs >= f_lo) & (freqs < f_hi)
                val = float(np.sqrt(np.mean(np.square(fft_mag[mask])))) if np.any(mask) else 0.0
                band_energies.append(val)
            max_band = max(band_energies) or 1.0

            sx0 = wave_w + 6
            draw.line([( wave_w, y_top + 6), (wave_w, y_bot - 6)], fill=(45, 55, 80), width=1)
            draw.text((sx0, y_top + 4), "FFT SPECTRUM (Hz bands)", fill=(180, 195, 220))
            bar_area_top = y_top + 24
            bar_area_bot = y_bot - 22
            bar_max_h = max(16, bar_area_bot - bar_area_top)
            avail_w = max(60, w - sx0 - 10)
            col_w = avail_w // len(bands)
            for b_idx, ((b_label, _, _, b_color), b_val) in enumerate(zip(bands, band_energies)):
                norm_h = int(round((b_val / max_band) * bar_max_h)) if max_band > 0 else 0
                bx0 = sx0 + b_idx * col_w + 3
                bx1 = bx0 + max(8, col_w - 8)
                by1 = bar_area_bot
                by0 = max(bar_area_top, by1 - max(2, norm_h))
                draw.rectangle([bx0, by0, bx1, by1], fill=b_color)
                draw.text((bx0 - 1, bar_area_bot + 4), b_label[:6], fill=(165, 175, 200))

        prefix = f"[{idx + 1}/{len(wav_paths)}] " if len(wav_paths) > 1 else ""
        draw.text(
            (8, y_top + 4),
            f"{prefix}{wp.name}  {duration:.2f}s  {rate}Hz  peak={peak:.2f}{spec_summary}",
            fill=(230, 235, 245),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def audio_asset_handler(
    ctx: WorkerContext,
    force_regenerate: Optional[bool] = None,
) -> Callable[[WorkItem], WorkerResult]:
    def handler(item: WorkItem) -> WorkerResult:
        reqs = plan_audio_requirements(ctx, item, AUDIO_MODES)
        if ctx.voice_agent is None:
            raise StudioAgentError(f"No voice/audio agent is configured; cannot build audio item '{item.item_id}'.")
        lib = ctx.new_library()
        eff_force = bool(item.revisions) if force_regenerate is None else bool(force_regenerate)
        batch_wavs: List[Path] = []
        note_parts: List[str] = []
        first_entry = None
        for req in reqs:
            record_turn_substep(
                "generating_asset",
                f"Synthesizing audio asset {req.target_path} (mode={req.fabrication_mode})",
                details={"asset_id": req.asset_id, "target_path": req.target_path, "mode": req.fabrication_mode},
            )
            out = lib.fabricate_single_asset(req, force_regenerate=eff_force)
            if out is None or not Path(out).is_file():
                raise StudioAgentError(f"Audio fabrication of '{req.asset_id}' ({req.target_path}) produced no file.")
            _merge_catalog_entry(ctx, lib, req.asset_id)
            if first_entry is None:
                first_entry = lib.catalog.entries.get(req.asset_id)
            batch_wavs.append(Path(out))
            note_parts.append(
                f"Audio '{req.asset_id}' ({req.fabrication_mode}) -> {req.target_path}. "
                f"Text/prompt: {(req.visual_prompt_or_text or req.description)[:200]}"
            )
        if first_entry is not None and item.item_id not in lib.catalog.entries:
            if hasattr(first_entry, "model_copy"):
                lib.catalog.entries[item.item_id] = first_entry.model_copy(update={"asset_id": item.item_id})
            elif isinstance(first_entry, dict):
                lib.catalog.entries[item.item_id] = {**first_entry, "asset_id": item.item_id}
            else:
                lib.catalog.entries[item.item_id] = first_entry
            _merge_catalog_entry(ctx, lib, item.item_id)

        safe_id = _safe_id(item.item_id)
        item_dir = _item_dir(ctx, item)
        manifest_file = item_dir / f"synthesized_wavs_rev{len(item.revisions)}.json"
        accumulated_wavs: List[Path] = []
        if manifest_file.is_file():
            try:
                for prev_p in json.loads(manifest_file.read_text(encoding="utf-8")):
                    pp = Path(prev_p)
                    if pp.is_file() and pp not in accumulated_wavs:
                        accumulated_wavs.append(pp)
            except Exception:
                pass
        for bw in batch_wavs:
            if bw not in accumulated_wavs:
                accumulated_wavs.append(bw)
        manifest_file.write_text(json.dumps([str(p) for p in accumulated_wavs], indent=2), encoding="utf-8")

        proof = render_waveform_image(
            accumulated_wavs,
            item_dir / f"waveform_{safe_id}_rev{len(item.revisions)}.png",
        )
        record_turn_substep(
            "captured_proof",
            f"Rendered waveform & FFT spectrum inspection proof for {len(accumulated_wavs)} audio cue(s)",
            details={"proof_image": str(proof), "audio_files": [p.name for p in accumulated_wavs]},
        )
        notes = " | ".join(note_parts)
        return WorkerResult(
            proof_images=[str(proof)] + [str(p) for p in accumulated_wavs],
            notes=notes,
        )

    return handler


def level_map_handler(ctx: WorkerContext) -> Callable[[WorkItem], WorkerResult]:
    """Revises/creates the level layout under the council's direction, realizes the world in GDScript (`scenes.gd`),
    and returns both a live Godot viewport filmstrip and the annotated architectural map as proof."""
    from engine.studio.proofs import render_level_map
    lock = ctx.code_lock if ctx.code_lock is not None else threading.Lock()

    def handler(item: WorkItem) -> WorkerResult:
        agent = ctx.swe_agent
        if agent is None or not hasattr(agent, "formulate_level_layout"):
            raise StudioAgentError(f"No SWE agent is configured; cannot build level map item '{item.item_id}'.")
        fb = [(r.revision_index + 1, r.feedback) for r in item.revisions if r.feedback]
        guidance = f"[{item.item_id}] {item.title}: {item.spec}"
        if fb:
            guidance += "\nFeedback to address:\n" + "\n".join(f"  rev {n}: {t}" for n, t in fb)
        layout = agent.formulate_level_layout(ctx.cartridge_dir, force_refresh=True, guidance=guidance)
        safe_id = _safe_id(item.item_id)
        map_proof = render_level_map(layout, _item_dir(ctx, item) / f"level_map_{safe_id}_rev{len(item.revisions)}.png")
        proof_images = [str(map_proof)]
        notes = (
            f"Layout: {len(layout.world_regions)} regions, {len(layout.entities)} entities, {len(layout.triggers)} triggers; "
            f"world {layout.world_bounds} (~{layout.intended_world_scale_in_viewports:.1f} screens). Rationale: {layout.deliberation[:400]}"
        )

        code_modules = [m for m in item.target_modules if m.endswith(CODE_MODULE_SUFFIXES)]
        should_realize_in_godot = hasattr(agent, "implement_requirement_track") and bool(code_modules)
        if should_realize_in_godot:
            from engine.studio.models import AcceptanceCriterion, GameRequirementsSpec, RequirementTrack
            modules = code_modules
            description = (
                f"Realize the authoritative level layout (`level_layout.json`, {len(layout.world_regions)} regions, "
                f"world_bounds={layout.world_bounds}, ~{layout.intended_world_scale_in_viewports:.1f} screens) and "
                f"`level_generation_strategy` in Godot (`{modules}`):\n"
                f"{item.spec}\n\n"
                f"CRITICAL WORLD CONSTRUCTION REQUIREMENTS:\n"
                f"- Execute the Game Bible's `level_generation_strategy` (`generation_approach`, `world_structure`, `runtime_construction_plan`), `rendering_strategy.json`, and load `res://level_layout.json` across all {len(layout.world_regions)} regions (`{[r.region_id for r in layout.world_regions]}`).\n"
                f"- Render world geometry, terrain, fluids, and props using the bespoke rendering techniques from `rendering_strategy.json` (procedural `Polygon2D`/`Line2D` meshes, custom `CanvasItem` shaders / `ShaderMaterial`, `GPUParticles2D`, `PointLight2D`, and synthesized sprites/rigs from `res://assets/sprites/*.png`) matching the 16:9 Concept Art Mockup.\n"
                f"- NEVER replace the designed world with a 3-4 `Rect2` placeholder room.\n\n"
                f"COUNCIL WILL JUDGE: {item.proof_spec}"
            )
            if fb:
                description += "\n\nCOUNCIL FEEDBACK TO ADDRESS (all of it):\n" + "\n".join(f"  rev {n}: {t}" for n, t in fb)
            track = RequirementTrack(
                deliberation=item.deliberation,
                id=item.item_id,
                title=item.title,
                description=description,
                target_modules=modules,
                acceptance_criteria=[
                    AcceptanceCriterion(
                        deliberation="The council's level proof request and world scale realization are the acceptance criterion.",
                        id=f"AC-{item.item_id}",
                        description=item.proof_spec or item.spec,
                        verification_method="headless_playtest",
                    )
                ],
                status="in_progress",
            )
            bible_title = "game"
            bible_path = ctx.cartridge_dir / "game_bible.json"
            if bible_path.is_file():
                try:
                    bible_title = json.loads(bible_path.read_text(encoding="utf-8")).get("game_title") or bible_title
                except ValueError:
                    pass
            req_spec = GameRequirementsSpec(
                deliberation="Board-driven level realization requirement",
                game_title=bible_title,
                requirements=[track],
                asset_manifest=[],
            )
            if is_in_board_scheduler():
                if not lock.acquire(blocking=False):
                    raise WorkerYielded(f"Waiting for code_lock on level_map '{item.item_id}'")
            else:
                lock.acquire()
            try:
                snap = _snapshot_modules(ctx.cartridge_dir, modules + ["project.godot", "main.tscn"])
                try:
                    agent.implement_requirement_track(track=track, staging_dir=ctx.cartridge_dir, requirements_spec=req_spec)
                    if ctx.code_verify is not None:
                        ctx.code_verify(ctx.cartridge_dir, track)
                    plan = plan_filmstrip(ctx, item)
                    script: List[Dict[str, Any]] = []
                    for s in plan.input_segments:
                        seg_d: Dict[str, Any] = {"from": s.from_tick, "to": s.to_tick, "actions": list(s.actions)}
                        if getattr(s, "mouse_click_pos", None) is not None:
                            seg_d["mouse_click_pos"] = list(s.mouse_click_pos)
                        script.append(seg_d)
                    setup_calls = [
                        {
                            "at_tick": c.at_tick,
                            "target_node_or_property": c.target_node_or_property,
                            "method_name": c.method_name,
                            "args_json": c.args_json,
                        }
                        for c in (getattr(plan, "setup_calls", None) or [])
                    ]
                    out_strip_path = _item_dir(ctx, item) / f"level_filmstrip_{safe_id}_rev{len(item.revisions)}.png"
                    if ctx.filmstrip_fn is None:
                        from engine.studio.proofs import capture_filmstrip
                        strip = capture_filmstrip(
                            ctx.cartridge_dir,
                            ticks=list(plan.capture_ticks),
                            out_path=out_strip_path,
                            input_script=script,
                            setup_script=setup_calls or None,
                            verify_counterfactual=False,
                        )
                    else:
                        strip = ctx.filmstrip_fn(
                            ctx.cartridge_dir,
                            ticks=list(plan.capture_ticks),
                            out_path=out_strip_path,
                            input_script=script,
                        )
                    proof_images = [str(map_proof), str(strip)]
                    notes += f" Realized in {modules}; Godot viewport tour captured at ticks {plan.capture_ticks}."
                except Exception:
                    _restore_modules(ctx.cartridge_dir, snap)
                    raise
            finally:
                lock.release()
        return WorkerResult(proof_images=proof_images, notes=notes)

    setattr(handler, "_code_lock", lock)
    return handler


STANDARD_INPUT_ACTIONS = ["ui_up", "ui_down", "ui_left", "ui_right", "ui_accept", "ui_select", "ui_cancel"]
MAX_PROOF_TICKS = 900
MAX_PROOF_FRAMES = 12
CODE_MODULE_SUFFIXES = (".gd", ".tscn", ".godot", ".gdshader")


_GD_ACTION_RE = re.compile(r'Input\.(?:is_action_pressed|is_action_just_pressed|is_action_just_released|get_action_strength)\(\s*["\']([A-Za-z0-9_]+)["\']\s*\)')
_GD_AXIS_RE = re.compile(r'Input\.get_axis\(\s*["\']([A-Za-z0-9_]+)["\']\s*,\s*["\']([A-Za-z0-9_]+)["\']\s*\)')
_GD_KEY_RE = re.compile(r'\b(KEY_[A-Z0-9_]+)\b')


def available_input_actions(cartridge_dir: Path) -> List[str]:
    """Input action names declared in project.godot's [input] section, Godot's standard UI actions,
    plus any custom action names and KEY_* bindings referenced in the cartridge's .gd scripts."""
    actions = list(STANDARD_INPUT_ACTIONS)
    c_dir = Path(cartridge_dir)
    project = c_dir / "project.godot"
    if project.is_file():
        in_input = False
        for line in project.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_input = stripped == "[input]"
                continue
            if in_input and "=" in stripped:
                name = stripped.split("=", 1)[0].strip()
                if name and name not in actions:
                    actions.append(name)
    from engine.sim.godot_verifier import iter_cartridge_gd_files

    for gd_path in iter_cartridge_gd_files(c_dir, include_tests=False, exclude_helper_libs=True):
        try:
            src = gd_path.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in _GD_ACTION_RE.finditer(src):
            act = m.group(1).strip()
            if act and act not in actions:
                actions.append(act)
        for m in _GD_AXIS_RE.finditer(src):
            for grp in (m.group(1), m.group(2)):
                act = grp.strip()
                if act and act not in actions:
                    actions.append(act)
        for m in _GD_KEY_RE.finditer(src):
            kname = m.group(1).strip()
            if kname and kname not in actions:
                actions.append(kname)
    return actions


def _extract_gd_input_and_layer_semantics(src: str) -> Tuple[List[str], List[str]]:
    """Extracts (1) concise input-to-behavior mappings (`Input.*` / `KEY_*` / `MOUSE_BUTTON_*` conditions paired with
    their immediate body statements) and (2) physics collision layer/mask & scene modulation lines from a `.gd` script."""
    raw_lines = src.splitlines()
    input_bindings: List[str] = []
    layer_lines: List[str] = []
    n_lines = len(raw_lines)
    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if any(
            tok in stripped
            for tok in ("collision_layer", "collision_mask", "CanvasModulate", "light_mode", "LIGHT_MODE_")
        ):
            if stripped not in layer_lines and len(layer_lines) < 6:
                layer_lines.append(stripped[:90])
        if (
            "Input." in stripped
            or _GD_KEY_RE.search(stripped)
            or "MOUSE_BUTTON_" in stripped
        ) and not stripped.startswith(("const ", "var ", "#")):
            base_indent = len(line) - len(line.lstrip("\t "))
            body_stmts: List[str] = []
            if stripped.endswith(":"):
                for j in range(idx + 1, min(idx + 5, n_lines)):
                    nxt = raw_lines[j]
                    nxt_s = nxt.strip()
                    if not nxt_s or nxt_s.startswith("#"):
                        continue
                    nxt_indent = len(nxt) - len(nxt.lstrip("\t "))
                    if nxt_indent <= base_indent:
                        break
                    body_stmts.append(nxt_s[:65])
                    if len(body_stmts) >= 2:
                        break
            if body_stmts and all(
                re.match(r"^(?:(?:el)?if\s+\w+:\s*)?_[A-Za-z0-9_]+\s*=\s*(?:true|false|event\.pressed)\b", s)
                for s in body_stmts
            ):
                continue
            cond_short = re.sub(r"event\s+is\s+InputEventKey\s+and\s+event\.pressed\s+and\s+", "", stripped)
            cond_short = re.sub(r"^(?:el)?if\s+", "", cond_short).rstrip(":")
            cond_short = re.sub(r"^_[A-Za-z0-9_]+\s+or\s+", "", cond_short)
            entry = f"{cond_short[:65]} -> {'; '.join(body_stmts)}" if body_stmts else cond_short[:85]
            if entry not in input_bindings and len(input_bindings) < 12:
                input_bindings.append(entry)
    return input_bindings, layer_lines


def _summarize_cartridge_for_qa(cartridge_dir: Path, max_chars: int = 6500) -> str:
    """Extracts a compact GDScript symbol, input-binding semantics, physics layer/mask, world_bounds,
    and asset catalog summary so QA (`plan_filmstrip`), TaskLead, and Council reviewers see exact input
    behaviors, spatial bounds, and all modules without truncation."""
    from engine.sim.godot_verifier import find_broken_script_references, iter_cartridge_gd_files

    c_dir = Path(cartridge_dir).resolve()
    header_lines: List[str] = []
    broken_refs = find_broken_script_references(c_dir, include_tests=False)
    if broken_refs:
        broken_desc = [
            f"{d.file_path.relative_to(c_dir).as_posix()}:{d.line_number} -> {d.error_message}"
            for d in broken_refs
        ]
        header_lines.append(
            "- ⚠️ BROKEN / MISSING SCRIPT OR SCENE REFERENCES ON DISK (load/preload returns null at runtime — MUST FIX!): "
            + " | ".join(broken_desc)
        )
    layout_path = c_dir / "level_layout.json"
    if layout_path.is_file():
        try:
            ldata = json.loads(layout_path.read_text(encoding="utf-8"))
            if isinstance(ldata, dict):
                wb = ldata.get("world_bounds")
                regs = [
                    f"{r.get('region_id')}:{r.get('bounds')}"
                    for r in (ldata.get("world_regions") or [])[:8]
                    if isinstance(r, dict)
                ]
                header_lines.append(f"- level_layout.json: world_bounds={wb} regions={regs}")
        except Exception:
            pass
    input_map_lines: List[str] = []
    layer_map_lines: List[str] = []
    module_lines: List[str] = []
    for gd_path in iter_cartridge_gd_files(c_dir, include_tests=False, exclude_helper_libs=True):
        if gd_path.name.startswith("test_"):
            continue
        try:
            src = gd_path.read_text(encoding="utf-8")
        except Exception:
            continue
        rel_name = gd_path.relative_to(c_dir).as_posix()
        classes = re.findall(r"^\s*class\s+([A-Za-z0-9_]+)", src, flags=re.MULTILINE)
        all_funcs = [
            (fn, args)
            for fn, args in re.findall(r"^\s*(?:static\s+)?func\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)", src, flags=re.MULTILINE)
            if not fn.startswith("simulate_")
        ]
        # Prioritize public functions in the compact per-file listing so internal `_build_*`/`_on_*` callbacks don't crowd out other files
        pub_funcs = [(fn, args) for fn, args in all_funcs if not fn.startswith("_")] or all_funcs[:6]
        vars_top = re.findall(r"^var\s+([A-Za-z0-9_]+)", src, flags=re.MULTILINE)
        bound_assets = sorted(set(re.findall(r"(?:res://)?assets/[A-Za-z0-9_./-]+\.(?:png|wav|ogg|mp3)", src)))
        input_bindings, layer_lines = _extract_gd_input_and_layer_semantics(src)
        if input_bindings:
            input_map_lines.append(f"  * {rel_name}: {input_bindings}")
        if layer_lines:
            layer_map_lines.append(f"  * {rel_name}: {layer_lines}")
        func_sig = ", ".join(f"{fn}({args.strip()})" for fn, args in pub_funcs[:6])
        cls_part = f" classes={classes[:4]}" if classes else ""
        asset_part = f" bound_assets={bound_assets[:4]}" if bound_assets else ""
        mod_line = f"- {rel_name}:{cls_part} vars={vars_top[:6]}{asset_part} funcs=[{func_sig}]"
        module_lines.append(mod_line[:195])
    lines: List[str] = list(header_lines)
    if input_map_lines:
        lines.append("- input_bindings_by_module (exact key/action -> effect in code):\n" + "\n".join(input_map_lines))
    if layer_map_lines:
        lines.append("- physics_and_lighting_layers_by_module:\n" + "\n".join(layer_map_lines))
    cat_path = c_dir / "assets" / "asset_library.json"
    cat_line = ""
    if cat_path.is_file():
        try:
            cat = json.loads(cat_path.read_text(encoding="utf-8"))
            entries = cat.get("entries") if isinstance(cat, dict) else {}
            if isinstance(entries, dict) and entries:
                entry_Descs = [
                    (
                        f"{k} (type={v.get('asset_type', '')}, mode={v.get('fabrication_mode', '')}, "
                        f"tex_size={v.get('texture_pixel_size', [])}, hframes={v.get('hframes', 1)}, vframes={v.get('vframes', 1)} "
                        f"-> {v.get('res_path', '')})"
                    )
                    for k, v in list(entries.items())[:18]
                    if isinstance(v, dict)
                ]
                cat_line = f"- asset_library.json ({len(entries)} assets on disk): {entry_Descs}"[:1400]
        except Exception:
            pass
    lines.extend(module_lines)
    if cat_line:
        lines.append(cat_line)
    summary = "\n".join(lines)
    return summary[:max_chars]


_FORBIDDEN_SETUP_CALL_PREFIXES = ("simulate_", "force_", "complete_", "auto_solve", "instant_win")
_FORBIDDEN_SETUP_CALL_NAMES = frozenset({
    "set_onboarding_step",
    "advance_onboarding_step",
    "advance_step",
    "skip_onboarding",
    "trigger_win",
    "win_game",
})


def validate_filmstrip_plan(plan: FilmstripPlan, actions: List[str], item_kind: str = "") -> List[str]:
    if (getattr(plan, "production_defect_report", "") or "").strip():
        return []
    issues: List[str] = []
    ticks = plan.capture_ticks
    if not ticks or len(ticks) > MAX_PROOF_FRAMES:
        issues.append(f"capture_ticks must contain 1 to {MAX_PROOF_FRAMES} ticks.")
    if any(t < 1 or t > MAX_PROOF_TICKS for t in ticks):
        issues.append(f"every capture tick must be within [1, {MAX_PROOF_TICKS}].")
    for seg in plan.input_segments:
        if seg.from_tick < 1 or seg.to_tick < seg.from_tick or seg.to_tick > MAX_PROOF_TICKS:
            issues.append(f"input segment {seg.from_tick}..{seg.to_tick} is invalid (1 <= from <= to <= {MAX_PROOF_TICKS}).")
        unknown = [a for a in seg.actions if a not in actions]
        if unknown:
            issues.append(f"input segment uses unknown actions {unknown}; available actions: {actions}.")
    raw_setup_calls = list(getattr(plan, "setup_calls", None) or [])
    for call in raw_setup_calls:
        if call.at_tick < 1 or call.at_tick > MAX_PROOF_TICKS:
            issues.append(f"setup_call at_tick={call.at_tick} must be within [1, {MAX_PROOF_TICKS}].")
        m_clean = call.method_name.strip()
        m_lower = m_clean.lower()
        if not m_clean:
            issues.append("setup_call method_name must not be empty.")
        elif m_lower.startswith(_FORBIDDEN_SETUP_CALL_PREFIXES) or m_lower in _FORBIDDEN_SETUP_CALL_NAMES:
            issues.append(
                f"setup_call '{call.method_name}' is a 'simulate_*' / state-forcing shortcut that bypasses real player input/physics or forces onboarding/completion state. "
                f"Drive gameplay mechanics and onboarding step transitions causally via `input_segments` (`actions`, `mouse_world_pos`, `mouse_click_pos`, or `when_snapshot_key` reactive triggers)."
            )
        try:
            parsed_args = json.loads(call.args_json or "[]")
            if not isinstance(parsed_args, list):
                issues.append(f"setup_call args_json for '{call.method_name}' must encode a JSON list.")
        except Exception:
            issues.append(f"setup_call args_json for '{call.method_name}' is not valid JSON.")
    valid_setup_calls = [
        c for c in raw_setup_calls
        if c.method_name.strip()
        and not c.method_name.strip().lower().startswith(_FORBIDDEN_SETUP_CALL_PREFIXES)
        and c.method_name.strip().lower() not in _FORBIDDEN_SETUP_CALL_NAMES
    ]
    has_test_scene_gd = bool((getattr(plan, "test_scene_gd_content", "") or "").strip())
    if item_kind == "integration" and actions and not plan.input_segments and not valid_setup_calls:
        issues.append(
            "FilmstripPlan for 'integration' / slice playtest tasks must include `input_segments` "
            "(`actions`, `mouse_world_pos`, or `mouse_click_pos`) demonstrating real player agency (or populate `production_defect_report` if the production game is broken/deadlocked)."
        )
    elif item_kind == "mechanic_code" and actions and not plan.input_segments and not valid_setup_calls and not has_test_scene_gd:
        issues.append(
            f"FilmstripPlan for '{item_kind}' must include at least one `input_segments` entry (keyboard `actions`, `mouse_world_pos`, or `mouse_click_pos`), "
            f"scenario `setup_calls`, or an isolated `test_scene_gd_content` harness demonstrating causal progression from the t=0 baseline."
        )
    if has_test_scene_gd:
        ts_code = (getattr(plan, "test_scene_gd_content", "") or "").strip()
        if "extends " not in ts_code:
            issues.append("`test_scene_gd_content` must be a valid GDScript 2.0 script starting with `extends Node2D` (or `extends Control`).")
    if item_kind in ("mechanic_code", "ui", "integration", "level_map") and not (getattr(plan, "council_test_scene_summary", "") or "").strip():
        issues.append(
            "`council_test_scene_summary` must be populated with a concise briefing for the Council explaining what test scene is running, "
            "what scenario/inputs are executed across `capture_ticks`, and what QA diagnostic overlays or telemetry readouts are displayed."
        )
    return issues


def _write_qa_test_scene_pair(
    cartridge_dir: Path,
    item_id: str,
    test_gd_code: str,
    raw_scene: str = "",
) -> Tuple[str, str, str]:
    safe_id = _safe_id(item_id)
    rel_scene_base = (raw_scene or "").strip()
    if rel_scene_base.startswith("res://"):
        rel_scene_base = rel_scene_base[len("res://"):]
    elif rel_scene_base.startswith("res:/"):
        rel_scene_base = rel_scene_base[len("res:/"):]
    rel_scene_base = rel_scene_base.lstrip("/")
    if not rel_scene_base or rel_scene_base in ("main.tscn", "main.gd") or not rel_scene_base.startswith("tests/"):
        rel_scene_base = f"tests/test_{safe_id}.tscn"
    if rel_scene_base.endswith(".gd"):
        rel_gd_path = rel_scene_base
        rel_tscn_path = rel_scene_base[:-3] + ".tscn"
    elif rel_scene_base.endswith(".tscn"):
        rel_tscn_path = rel_scene_base
        rel_gd_path = rel_scene_base[:-5] + ".gd"
    else:
        rel_tscn_path = f"{rel_scene_base}.tscn"
        rel_gd_path = f"{rel_scene_base}.gd"
    abs_gd_path = Path(cartridge_dir) / rel_gd_path
    abs_tscn_path = Path(cartridge_dir) / rel_tscn_path
    abs_gd_path.parent.mkdir(parents=True, exist_ok=True)
    abs_gd_path.write_text(test_gd_code.strip() + "\n", encoding="utf-8")
    tscn_content = (
        '[gd_scene load_steps=2 format=3]\n\n'
        f'[ext_resource type="Script" path="res://{rel_gd_path}" id="1_test"]\n\n'
        '[node name="TestScene" type="Node2D"]\n'
        'script = ExtResource("1_test")\n'
    )
    abs_tscn_path.write_text(tscn_content, encoding="utf-8")
    return f"res://{rel_tscn_path}", rel_gd_path, rel_tscn_path


def _serialize_input_segments(plan: FilmstripPlan) -> List[Dict[str, Any]]:
    script: List[Dict[str, Any]] = []
    for s in plan.input_segments:
        seg_d: Dict[str, Any] = {"from": s.from_tick, "to": s.to_tick, "actions": list(s.actions)}
        if getattr(s, "mouse_world_pos", None) is not None:
            seg_d["mouse_world_pos"] = list(s.mouse_world_pos)
        if getattr(s, "mouse_click_pos", None) is not None:
            seg_d["mouse_click_pos"] = list(s.mouse_click_pos)
        if getattr(s, "when_snapshot_key", ""):
            seg_d["when_snapshot_key"] = s.when_snapshot_key
            seg_d["when_operator"] = getattr(s, "when_operator", "") or "=="
            seg_d["when_value"] = getattr(s, "when_value", None)
            seg_d["hold_ticks"] = int(getattr(s, "hold_ticks", 10) or 10)
        script.append(seg_d)
    return script


def plan_filmstrip(
    ctx: WorkerContext,
    item: WorkItem,
    attempts: int = 3,
    prior_capture_feedback: str = "",
    prior_strip_path: Optional[Path] = None,
) -> FilmstripPlan:
    """Asks the model how to demonstrate this item (inputs + scenario setup + frames), validated against the game's real input map and symbols."""
    if not _is_live(ctx.client):
        raise StudioAgentError("Filmstrip planning requires an active google.genai Client (Rule 5)")
    from google.genai import types
    from engine.studio.context_curator import ContextCuratorAgent, is_real_genai_client

    actions = available_input_actions(ctx.cartridge_dir)
    qa_context = _summarize_cartridge_for_qa(ctx.cartridge_dir)
    context_block = f"\nCARTRIDGE SYMBOLS & ASSETS AVAILABLE FOR QA STAGING:\n{qa_context}\n" if qa_context else ""
    multimodal_parts: List[Any] = []
    from engine.studio.game_memory import GameMemoryStore, format_memory_query_results

    mem_store = GameMemoryStore(ctx.cartridge_dir, event_callback=ctx.event_callback, client=ctx.client)
    has_embed_client = callable(getattr(getattr(ctx.client, "models", None), "embed_content", None))
    if is_real_genai_client(ctx.client):
        curator = ContextCuratorAgent(
            cartridge_dir=ctx.cartridge_dir,
            client=ctx.client,
            model_name=ctx.model_name,
            board=ctx.board,
            event_callback=ctx.event_callback,
        )
        pack = curator.curate_context(
            target_agent_role="qa_test_engineer",
            task_request=f"{item.title}\nSPEC: {item.spec}\nPROOF: {item.proof_spec}",
            item_id=item.item_id,
            item_kind=item.kind,
        )
        context_block = f"{context_block}\n{pack.prompt_block}"
        multimodal_parts = list(pack.multimodal_parts)
    elif (ctx.cartridge_dir / "game_memory.json").is_file():
        role_mem = mem_store.format_role_memory_briefing("qa_test_engineer")
        if role_mem:
            context_block = f"{context_block}\n{role_mem}"
    if prior_strip_path is not None and Path(prior_strip_path).is_file():
        try:
            multimodal_parts.append(f"PASS 1 CAPTURED FILMSTRIP THAT FAILED STATE ASSERTIONS ({Path(prior_strip_path).name}):")
            multimodal_parts.append(types.Part.from_bytes(data=Path(prior_strip_path).read_bytes(), mime_type="image/png"))
        except Exception:
            pass
    prior_feedback_block = (
        f"\n⚠️ PASS 1 PLAYTEST EXECUTION FEEDBACK & TELEMETRY (DIAGNOSE ROOT CAUSE: TEST INPUTS vs. PRODUCTION DEFECT):\n"
        f"{prior_capture_feedback}\n"
        f"- Inspect the attached Pass 1 filmstrip, frame-by-frame telemetry, and production script code above to determine WHY Pass 1 failed:\n"
        f"  (A) If your QA play script missed a coordinate, forgot a prerequisite key/selection, or stopped too early, refine `input_segments` / `mouse_world_pos` / `capture_ticks`.\n"
        f"  (B) If the production game code itself is unwired, broken, or deadlocked under valid player inputs (so a real human player could not succeed either), DO NOT compensate by mutating game state inside `test_scene_gd_content`! Instead, populate `production_defect_report` with your concrete root-cause diagnosis and tick/state evidence so `TaskLead` sends it back to `swe_engineer` to fix the game.\n"
        if prior_capture_feedback
        else ""
    )
    issues: List[str] = []
    mem_query_blocks: List[str] = []
    asked_mem_queries: set[str] = set()
    mem_queries_used = 0
    max_mem_queries = 2 if has_embed_client else 0
    attempt = 1
    while attempt <= attempts:
        retry = "\n\nYOUR PREVIOUS PLAN FAILED VALIDATION; FIX:\n- " + "\n- ".join(issues) if issues else ""
        mem_q_block = ("\n" + "\n".join(mem_query_blocks)) if mem_query_blocks else ""
        prompt = (
            f"You are the QA Test Engineer designing and building a Godot 4.3 test scene and scripted playthrough that PROVES a built item to the design council.\n"
            f"ITEM [{item.item_id}] ({item.kind}): {item.title}\nSPEC: {item.spec}\nTHE COUNCIL WANTS TO SEE: {item.proof_spec}\n"
            f"Available input actions: {actions}.{context_block}{mem_q_block}{prior_feedback_block}\n"
            f"1. BUILD YOUR QA TEST SCENE (`target_scene` + `test_scene_gd_content`) OR REPORT A PRODUCTION DEFECT (`production_defect_report`):\n"
            f"   - Set `target_scene = 'res://tests/test_{_safe_id(item.item_id)}.tscn'` and write the complete GDScript 2.0 (`extends Node2D`) test scene in `test_scene_gd_content`.\n"
            f"   - For isolated subsystem/mechanic/UI tasks, `test_scene_gd_content` can instantiate the specific feature script(s) or entities under test.\n"
            f"   - For whole-game `integration` or `level_map` tasks (or when verifying behavior inside the full game), `test_scene_gd_content` can instantiate `preload('res://main.tscn').instantiate()` (or `preload('res://main.gd').new()`) as a child node in `_ready()` so it runs the real game entrypoint while allowing you to attach QA diagnostic overlays and custom telemetry on top.\n"
            f"   - DIAGNOSTIC INTEGRITY & FULL COUNCIL TRANSPARENCY: Your job is to surface truth, never to manufacture a green check. The Council sees your exact `test_scene_gd_content` source code, `setup_calls`, visible HUD text telemetry, and live `res://main.tscn` parity comparison. Never inject mid-run state mutations (e.g. `if sim_tick == ...:` inventory/score grants) inside `test_scene_gd_content` to mask a broken mechanic; if inspecting the code or Pass 1 telemetry reveals a production bug or deadlock in the game itself, report it in `production_defect_report` so `swe_engineer` fixes the production code.\n"
            f"   - Keep production `main.gd` and `ui_hud.gd` 100% pure game code with ZERO debug overlays—all test staging and rich QA diagnostic overlays belong exclusively inside your `test_scene_gd_content`.\n"
            f"2. STRATEGIZE DIAGNOSTIC OVERLAYS, HUMAN-SCALE RESPONSIVENESS & TELEMETRY (`overlays_and_telemetry_strategy`):\n"
            f"   - Reason from first principles: what would a live human player experience when interacting with this feature, and what needs to be measured over time (`capture_ticks`, 60 ticks ~ 1 second) to prove the feature feels responsive, legible, and free of subsystem conflicts?\n"
            f"   - For example, if the feature involves movement, steering, rotation, or physics controls, holding an input over 60–180 ticks (1–3 seconds) must produce a responsive, human-playable rate of change rather than a sluggish crawl or two subsystems fighting/canceling each other out!\n"
            f"   - Build rich test-only visual diagnostic overlays inside `test_scene_gd_content` (e.g., a foreground `CanvasLayer` or `Node2D._draw()` node in `res://tests/test_*.tscn` that draws velocity/heading/force vectors, collider/trigger bounds, camera framing markers, or live delta readouts) AND include `'player_pos'` plus task-specific metrics in `func get_state_snapshot() -> Dictionary` so both the rendered frames and the `[QA] pos=...` bottom telemetry bar prove real progression.\n"
            f"3. WRITE A CLEAR SUMMARY FOR THE COUNCIL (`council_test_scene_summary`):\n"
            f"   - Write `council_test_scene_summary` explaining to the Council: (a) what your test scene instantiates and stages, (b) what scenario and inputs run across `capture_ticks`, and (c) what QA diagnostic overlays or telemetry readouts you added so the Council knows exactly what it is looking at.\n"
            f"4. SCRIPT CAUSAL INPUTS & CAPTURE TICKS (`capture_ticks`, `input_segments`, `state_assertions`):\n"
            f"   - Choose `capture_ticks` (1-{MAX_PROOF_FRAMES} frames, ticks 1-{MAX_PROOF_TICKS}, 60 ticks ~ 1 second), `input_segments` "
            f"(actions or KEY_* held over tick ranges, plus optional `mouse_world_pos` or `mouse_click_pos`), and optional `setup_calls` (ONLY for scenario staging "
            f"such as spawning a hazard or positioning actors—NEVER call `simulate_*`, `force_*`, `complete_*`, or `set_onboarding_step` shortcuts that bypass real player input or physics) "
            f"so the captured frames show:\n"
            f"     * Frame 1 (`capture_ticks[0]`, before player inputs take effect): the UNSOLVED t=0 baseline state (`test_unsolved_baseline_t0`), and\n"
            f"     * Subsequent frames (`capture_ticks[1..]`): the authentic causal progression driven by `input_segments` (`actions`, `mouse_world_pos`, and/or `mouse_click_pos`). Never leave `input_segments` empty on an interactive mechanic, UI control, or integration task!\n"
            f"   - WORLD-SPACE vs. SCREEN-SPACE MOUSE CLICKS (`mouse_world_pos` vs. `mouse_click_pos`):\n"
            f"     * Whenever clicking an in-world location, terrain cell, spire/pedestal, or entity in a scene with a `Camera2D`, ALWAYS set `mouse_world_pos = [wx, wy]` using the exact 2D world coordinates from the level/world scripts! The harness automatically transforms `mouse_world_pos` through the active `Camera2D` canvas transform (`get_canvas_transform() * Vector2(wx, wy)`) so your click lands on the exact world target regardless of camera offset or zoom.\n"
            f"     * Use `mouse_click_pos = [sx, sy]` ONLY when clicking screen-space `CanvasLayer` HUD buttons.\n"
            f"   - STEP-BY-STEP CAUSAL CHAIN PLAYTESTING (NEVER SCRIPT A 1-CLICK SMOKE TRIGGER):\n"
            f"     * Read the target scripts' input handlers (`_unhandled_input`, `_input`) and terrain/placement rules in `CARTRIDGE SYMBOLS & ASSETS AVAILABLE FOR QA STAGING` and the curated code chunks before scripting `input_segments`:\n"
            f"       1. TOOL / MODE SELECTION: Check which tool/building slot starts active at `t=0` and which `KEY_*` or action selects the required intermediate/target structure (e.g., pressing `KEY_2` or `KEY_3` before clicking to place Slot 2 or Slot 3)!\n"
            f"       2. VALID BUILDABLE WORLD COORDINATES: Check the terrain/placement validity rules (`is_placement_valid`, land/spire/pedestal polygons vs. invalid water/void) and choose exact 2D world coordinates `mouse_world_pos = [wx, wy]` that lie on valid buildable terrain along the physical trajectory.\n"
            f"       3. ROTATION / TUNING: Check whether placed structures require rotation (`KEY_Q`, `KEY_E`, `KEY_R`) or impulse tuning so their physical surface normal / trajectory routes items into the goal.\n"
            f"       4. STEP-BY-STEP `state_assertions`: Populate `state_assertions` for EACH link in the causal chain across ticks (e.g., tool slot selected -> structure placed count incremented `> baseline` -> intermediate interaction/ricochet count `> 0` -> final objective/delivery count `> baseline`), rather than only asserting on the final score at the last tick! That way, if any step fails in Pass 1, the telemetry pinpoints the exact link that needs adjustment.\n"
            f"   - REACTIVE TIMING TIP: When a secondary player input depends on dynamic physics or movement reaching a threshold, "
            f"populate `when_snapshot_key`, `when_operator` ('<=', '>=', '==', '!='), `when_value`, and `hold_ticks` on that `InputSegment`.\n"
            f"   - LONG-TERM MEMORY SELECTIVITY: Leave `memory_updates` empty (`[]`) on routine QA runs! Never log single-task test steps or pass/fail status; record ONLY non-obvious harness constraints or verification invariants that future tasks/slices would otherwise trip over.\n"
            f"   - Use no input only if the item kind is a purely static layout preview. Provide `deliberation` FIRST.{retry}"
        )
        contents_arg: Any = ([prompt] + multimodal_parts) if multimodal_parts else prompt
        record_turn_epistemic_context(
            role="qa_test_engineer",
            curated_context=context_block,
            prompt_text=prompt,
        )
        resp = call_gemini_with_transient_retry(
            lambda: ctx.client.models.generate_content(
                model=ctx.model_name,
                contents=contents_arg,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FilmstripPlan,
                    thinking_config=bounded_thinking_config(512),
                ),
            ),
            label=f"plan_filmstrip[{item.item_id}]",
        )
        if not resp.text:
            raise StudioAgentError(f"Filmstrip planning for '{item.item_id}' returned an empty response")
        plan = FilmstripPlan.model_validate_json(resp.text)
        if plan.memory_updates:
            mem_store.apply_mutations(
                author_role="qa_test_engineer",
                proposals=plan.memory_updates,
                item_id=item.item_id,
            )
        mem_q = (plan.memory_query or "").strip()
        mem_q_key = f"{(plan.memory_query_bank or '').strip().lower()}::{mem_q.lower()}"
        if (
            mem_q
            and not (plan.memory_query_answer or "").strip()
            and not plan.production_defect_report.strip()
            and has_embed_client
            and mem_queries_used < max_mem_queries
            and mem_q_key not in asked_mem_queries
        ):
            asked_mem_queries.add(mem_q_key)
            mem_queries_used += 1
            record_turn_substep(
                "memory_query",
                f"QA querying long-term memory: {mem_q}",
                details={"query": mem_q, "bank": plan.memory_query_bank or "all"},
                specialist_agent="qa_test_engineer",
            )
            hits = mem_store.query_memories(
                caller_role="qa_test_engineer",
                query_text=mem_q,
                bank_filter=plan.memory_query_bank,
                client=ctx.client,
            )
            mem_query_blocks.append(format_memory_query_results(mem_q, plan.memory_query_bank, hits))
            issues = []
            continue
        issues = validate_filmstrip_plan(
            plan,
            actions,
            item_kind=item.kind if is_real_genai_client(ctx.client) else "",
        )
        if not issues:
            return plan
        attempt += 1
    raise StudioAgentError(f"Filmstrip plan for '{item.item_id}' still invalid after {attempts} attempts: {issues}")


def _snapshot_modules(
    cartridge_dir: Path,
    modules: List[str],
    scoped_only: bool = False,
) -> Dict[str, Optional[str]]:
    c_dir = Path(cartridge_dir)
    normalized_mods: List[str] = []
    for m in modules:
        s = CartridgeFileLockTable.normalize_path(m)
        if s and s not in normalized_mods:
            normalized_mods.append(s)
    if scoped_only:
        all_names = list(normalized_mods)
    else:
        existing_rel_paths: List[str] = []
        for ext in ("*.gd", "*.tscn", "*.gdshader"):
            for p in sorted(c_dir.rglob(ext)):
                try:
                    rel = p.relative_to(c_dir).as_posix()
                except ValueError:
                    continue
                if rel.startswith((".godot/", "artifacts/")):
                    continue
                existing_rel_paths.append(rel)
        all_names = list(
            dict.fromkeys(
                normalized_mods
                + ["main.gd", "mechanics.gd", "scenes.gd", "entities.gd", "ui_hud.gd", "project.godot", "main.tscn"]
                + existing_rel_paths
            )
        )
    snap: Dict[str, Optional[str]] = {}
    for m in all_names:
        p = c_dir / m
        snap[m] = p.read_text(encoding="utf-8") if p.is_file() else None
    return snap


def _restore_modules(
    cartridge_dir: Path,
    snap: Dict[str, Optional[str]],
    scoped_only: bool = False,
) -> None:
    c_dir = Path(cartridge_dir)
    for m, text in snap.items():
        p = c_dir / m
        if text is None:
            if p.is_file():
                p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
    if not scoped_only:
        for ext in ("*.gd", "*.tscn", "*.gdshader"):
            for extra_p in sorted(c_dir.rglob(ext)):
                try:
                    rel = extra_p.relative_to(c_dir).as_posix()
                except ValueError:
                    continue
                if rel.startswith((".godot/", "artifacts/")):
                    continue
                if rel not in snap:
                    extra_p.unlink()


def _compute_turn_code_diffs(
    cartridge_dir: Path,
    pre_snap: Dict[str, Optional[str]],
    scoped_only: bool = False,
) -> Dict[str, str]:
    """Computes unified diffs for any `.gd`, `.tscn`, `.gdshader`, or `project.godot` file created or modified since `pre_snap`."""
    import difflib

    c_dir = Path(cartridge_dir)
    post_snap = _snapshot_modules(c_dir, list(pre_snap.keys()), scoped_only=scoped_only)
    all_keys = sorted(set(pre_snap.keys()) | set(post_snap.keys()))
    diffs: Dict[str, str] = {}
    for rel_path in all_keys:
        before_txt = pre_snap.get(rel_path) or ""
        after_txt = post_snap.get(rel_path) or ""
        if before_txt == after_txt:
            continue
        diff_lines = list(
            difflib.unified_diff(
                before_txt.splitlines(keepends=True),
                after_txt.splitlines(keepends=True),
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
                n=3,
            )
        )
        if diff_lines:
            diffs[rel_path] = "".join(diff_lines)
    return diffs


def code_item_handler(ctx: WorkerContext) -> Callable[[WorkItem], WorkerResult]:
    """Implements a mechanic/UI/integration item with the SWE agent, gates it with the engine's own verification,
    and proves it with a scripted filmstrip captured from the real game. Whole-project code work is serialised by
    `ctx.code_lock` because the verification gates validate the entire project; failures restore the touched modules."""
    lock = ctx.code_lock if ctx.code_lock is not None else threading.Lock()

    def handler(item: WorkItem) -> WorkerResult:
        agent = ctx.swe_agent
        if agent is None or not hasattr(agent, "implement_requirement_track"):
            raise StudioAgentError(f"No SWE agent is configured; cannot build code item '{item.item_id}'.")
        modules = [m for m in item.target_modules if m.endswith(CODE_MODULE_SUFFIXES)] or ["main.gd"]
        fb = [(r.revision_index + 1, r.feedback) for r in item.revisions if r.feedback]
        description = f"{item.spec}\n\nCOUNCIL WILL JUDGE: {item.proof_spec}"
        if fb:
            description += "\n\nCOUNCIL FEEDBACK TO ADDRESS (all of it):\n" + "\n".join(f"  rev {n}: {t}" for n, t in fb)
        track = RequirementTrack(
            deliberation=item.deliberation,
            id=item.item_id,
            title=item.title,
            description=description,
            target_modules=modules,
            acceptance_criteria=[
                AcceptanceCriterion(
                    deliberation="The council's proof request is the acceptance criterion.",
                    id=f"AC-{item.item_id}", description=item.proof_spec or item.spec, verification_method="headless_playtest",
                )
            ],
            status="in_progress",
        )
        bible_title = "game"
        bible_path = ctx.cartridge_dir / "game_bible.json"
        if bible_path.is_file():
            try:
                bible_title = json.loads(bible_path.read_text(encoding="utf-8")).get("game_title") or bible_title
            except ValueError:
                pass
        req_spec = GameRequirementsSpec(deliberation="Board-driven single-item requirement", game_title=bible_title, requirements=[track], asset_manifest=[])
        if is_in_board_scheduler():
            if not lock.acquire(blocking=False):
                raise WorkerYielded(f"Waiting for code_lock on code item '{item.item_id}'")
        else:
            lock.acquire()
        try:
            snap = _snapshot_modules(ctx.cartridge_dir, modules + ["project.godot", "main.tscn"])
            try:
                agent.implement_requirement_track(track=track, staging_dir=ctx.cartridge_dir, requirements_spec=req_spec)
                if ctx.code_verify is not None:
                    ctx.code_verify(ctx.cartridge_dir, track)
                plan = plan_filmstrip(ctx, item)
                script = _serialize_input_segments(plan)
                setup_calls = [
                    {
                        "at_tick": c.at_tick,
                        "target_node_or_property": c.target_node_or_property,
                        "method_name": c.method_name,
                        "args_json": c.args_json,
                    }
                    for c in (getattr(plan, "setup_calls", None) or [])
                ]
                out_strip_path = _item_dir(ctx, item) / f"filmstrip_{_safe_id(item.item_id)}_rev{len(item.revisions)}.png"
                if ctx.filmstrip_fn is None:
                    from engine.studio.proofs import capture_filmstrip
                    strip = capture_filmstrip(
                        ctx.cartridge_dir,
                        ticks=list(plan.capture_ticks),
                        out_path=out_strip_path,
                        input_script=script,
                        setup_script=setup_calls or None,
                        verify_counterfactual=True,
                        state_assertions=list(getattr(plan, "state_assertions", None) or []),
                    )
                else:
                    strip = ctx.filmstrip_fn(
                        ctx.cartridge_dir,
                        ticks=list(plan.capture_ticks),
                        out_path=out_strip_path,
                        input_script=script,
                    )
            except Exception as swe_run_exc:
                if "Godot headless smoke check failed in res://tests/" not in str(swe_run_exc):
                    _restore_modules(ctx.cartridge_dir, snap)
                raise
        finally:
            lock.release()
        extra_qa = ""
        if setup_calls:
            extra_qa += f" Setup calls: {[c['method_name'] for c in setup_calls]}."
        if getattr(plan, "state_assertions", None):
            extra_qa += f" Assertions: {[(a.at_tick, a.snapshot_key, a.expected_value_or_change) for a in plan.state_assertions]}."
        council_summary = (getattr(plan, "council_test_scene_summary", "") or "").strip()
        overlays_strat = (getattr(plan, "overlays_and_telemetry_strategy", "") or "").strip()
        qa_briefing = ""
        if council_summary:
            qa_briefing += f" QA Test Scene Summary for Council: {council_summary}"
        if overlays_strat:
            qa_briefing += f" QA Overlays & Telemetry: {overlays_strat}"
        notes = (
            f"Implemented '{item.item_id}' in {modules}; engine verification passed. Filmstrip ticks {plan.capture_ticks}.{qa_briefing}{extra_qa}"
        )
        return WorkerResult(proof_images=[str(strip)], notes=notes)

    setattr(handler, "_code_lock", lock)
    return handler


# ---------------------------------------------------------------------------
# Iterative Task Lead Agent & Autonomous Domain Specialists
# ---------------------------------------------------------------------------

ALL_SPECIALIST_AGENTS = ("visual_artist", "audio_designer", "script_writer", "swe_engineer", "qa_test_engineer")


def available_specialists(ctx: WorkerContext, item: Optional[WorkItem] = None) -> List[str]:
    """Returns the autonomous domain specialists configured and available in `ctx` for `item`.

    Standalone asset fabrication tasks (`visual_asset`, `animation`, `audio_asset`) are scoped to their
    asset fabrication specialists so they produce isolated asset files and inspection/waveform plates
    without rewriting shared `.gd` gameplay scenes or hijacking `main.tscn` into a static showcase.
    Gameplay, level, UI, and integration tasks have full access to all specialists (`visual_artist`,
    `audio_designer`, `script_writer`, `swe_engineer`, `qa_test_engineer`).
    """
    has_visual = ctx.library_factory is not None or ctx.visual_agent is not None or _is_live(ctx.client)
    has_audio = ctx.voice_agent is not None
    if item is not None and not item.revisions:
        if item.kind in ("visual_asset", "animation"):
            return ["visual_artist"] if has_visual else []
        if item.kind == "audio_asset":
            return ["audio_designer"] if has_audio else []

    specs: List[str] = []
    if has_visual:
        specs.append("visual_artist")
    if has_audio:
        specs.append("audio_designer")
    if _is_live(ctx.client):
        specs.append("script_writer")
    if ctx.swe_agent is not None and (
        hasattr(ctx.swe_agent, "implement_requirement_track") or hasattr(ctx.swe_agent, "formulate_level_layout")
    ):
        specs.append("swe_engineer")
    if (
        ctx.filmstrip_fn is not None
        or ctx.swe_agent is not None
        or (Path(ctx.cartridge_dir) / "project.godot").is_file()
    ):
        specs.append("qa_test_engineer")
    return specs


def run_visual_artist_specialist(
    ctx: WorkerContext,
    item: WorkItem,
    brief: str,
    target_paths: Optional[List[str]] = None,
    turn_index: int = 0,
) -> SpecialistReport:
    """Autonomous Visual Artist specialist: receives a natural-language brief from the Task Lead,
    has unrestricted access to all visual modes (`VISUAL_MODES`), fabricates the asset into the cartridge
    catalog, and returns a `SpecialistReport` with the generated asset and inspection/contact plates."""
    sub_id = item.item_id
    if target_paths and item.kind not in ("visual_asset", "animation"):
        first_target = str(target_paths[0]).strip()
        if first_target:
            stem = _safe_id(Path(first_target).stem)
            if stem and stem != "asset":
                sub_id = stem
    path_hint = f"\nRequested target path(s): {target_paths}" if target_paths else ""
    sub_item = item.model_copy(
        update={
            "item_id": sub_id,
            "spec": f"{item.spec}\n\nTASK LEAD BRIEF FOR VISUAL ARTIST:\n{brief}{path_hint}",
        }
    )
    res = _fabricate_visual(
        ctx,
        sub_item,
        VISUAL_MODES,
        force_regenerate=True,
    )
    req_file = _item_dir(ctx, sub_item) / "requirement.json"
    touched: List[str] = list(target_paths or [])
    if req_file.is_file():
        try:
            t_path = json.loads(req_file.read_text(encoding="utf-8")).get("target_path")
            if t_path and t_path not in touched:
                touched.insert(0, t_path)
        except Exception:
            pass
    return SpecialistReport(
        deliberation=f"Visual Artist executed brief for '{sub_id}'.",
        turn_index=turn_index,
        specialist_agent="visual_artist",
        brief=brief,
        summary=res.notes,
        touched_files=touched,
        proof_images=list(res.proof_images),
    )


def run_audio_designer_specialist(
    ctx: WorkerContext,
    item: WorkItem,
    brief: str,
    target_paths: Optional[List[str]] = None,
    turn_index: int = 0,
) -> SpecialistReport:
    """Autonomous Audio Designer specialist: receives a brief from the Task Lead, synthesizes the requested
    voice/SFX/ambient asset, renders its waveform proof image, and returns a `SpecialistReport`."""
    sub_id = item.item_id
    if target_paths and item.kind != "audio_asset":
        first_target = str(target_paths[0]).strip()
        if first_target:
            stem = _safe_id(Path(first_target).stem)
            if stem and stem != "asset":
                sub_id = stem
    path_hint = f"\nRequested target path(s): {target_paths}" if target_paths else ""
    sub_item = item.model_copy(
        update={
            "item_id": sub_id,
            "kind": "audio_asset",
            "spec": f"{item.spec}\n\nTASK LEAD BRIEF FOR AUDIO DESIGNER:\n{brief}{path_hint}",
        }
    )
    res = audio_asset_handler(ctx, force_regenerate=bool(item.revisions) or (turn_index > 0))(sub_item)
    touched_wavs: List[str] = []
    for p in res.proof_images:
        if str(p).lower().endswith(".wav"):
            try:
                rel = Path(p).relative_to(ctx.cartridge_dir).as_posix()
            except ValueError:
                rel = str(p)
            if rel not in touched_wavs:
                touched_wavs.append(rel)
    if not touched_wavs:
        touched_wavs = list(target_paths or [f"assets/audio/{sub_id}.wav"])
    return SpecialistReport(
        deliberation=f"Audio Designer executed brief for '{sub_id}'.",
        turn_index=turn_index,
        specialist_agent="audio_designer",
        brief=brief,
        summary=res.notes,
        touched_files=touched_wavs,
        proof_images=list(res.proof_images),
    )


def run_script_writer_specialist(
    ctx: WorkerContext,
    item: WorkItem,
    brief: str,
    target_paths: Optional[List[str]] = None,
    turn_index: int = 0,
) -> SpecialistReport:
    """Autonomous Scriptwriter & Narrative Dramatist specialist: crafts scene scripts (`cutscenes`),
    branching `dialogue_trees`, contextual `dialogue_barks`, and character voices based on the Task Lead's brief,
    persists them into `game_bible.json` / `narrative_bible.json` / `narrative.json`, and renders a visual
    script table proof sheet (`script_sheet_*.png`)."""
    if not _is_live(ctx.client):
        raise StudioAgentError(f"Scriptwriter specialist requires an active google.genai Client for '{item.item_id}' (Rule 5)")
    from google.genai import types
    from engine.studio.context_curator import ContextCuratorAgent, is_real_genai_client

    bible_path = ctx.cartridge_dir / "game_bible.json"
    bible_text = (
        compact_bible_json_for_prompt(bible_path.read_text(encoding="utf-8"))
        if bible_path.is_file()
        else "{}"
    )
    curated_context_block = f"GAME BIBLE CONTEXT:\n{bible_text}\n"
    from engine.studio.game_memory import GameMemoryStore, format_memory_query_results

    mem_store = GameMemoryStore(ctx.cartridge_dir, event_callback=ctx.event_callback, client=ctx.client)
    has_embed_client = callable(getattr(getattr(ctx.client, "models", None), "embed_content", None))
    if is_real_genai_client(ctx.client):
        curator = ContextCuratorAgent(
            cartridge_dir=ctx.cartridge_dir,
            client=ctx.client,
            model_name=ctx.model_name,
            board=ctx.board,
            event_callback=ctx.event_callback,
        )
        pack = curator.curate_context(
            target_agent_role="script_writer",
            task_request=f"{item.title}\nSPEC: {item.spec}\nBRIEF: {brief}",
            item_id=item.item_id,
            item_kind=item.kind,
        )
        curated_context_block = f"{pack.prompt_block}\nGAME BIBLE CONTEXT:\n{bible_text}\n"
    elif (ctx.cartridge_dir / "game_memory.json").is_file():
        role_mem = mem_store.format_role_memory_briefing("script_writer")
        if role_mem:
            curated_context_block = f"{role_mem}\n{curated_context_block}"

    mem_query_blocks: List[str] = []
    asked_mem_queries: set[str] = set()
    max_mem_queries = 2 if has_embed_client else 0
    deliverable: Optional[ScriptWritingDeliverable] = None
    for mem_turn in range(max_mem_queries + 1):
        mem_q_section = ("\n" + "\n".join(mem_query_blocks)) if mem_query_blocks else ""
        prompt = (
            f"You are the Lead Scriptwriter & Narrative Dramatist executing a brief for TASK [{item.item_id}] ({item.kind}): {item.title}\n"
            f"TASK SPEC: {item.spec}\n"
            f"TASK LEAD BRIEF FOR SCRIPTWRITER:\n{brief}\n\n"
            f"{curated_context_block}{mem_q_section}"
            f"Craft the requested scene scripts (`added_or_updated_cutscenes`), branching dialogue (`added_or_updated_dialogue_trees`), "
            f"contextual gameplay barks (`added_or_updated_barks`), or character sheets (`added_or_updated_characters`) suited to THIS game. "
            f"Provide `deliberation` FIRST."
        )
        resp = call_gemini_with_transient_retry(
            lambda: ctx.client.models.generate_content(
                model=ctx.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ScriptWritingDeliverable,
                    thinking_config=bounded_thinking_config(1536),
                ),
            ),
            label=f"run_script_writer_specialist[{item.item_id}:t{turn_index}]",
        )
        if not resp.text:
            raise StudioAgentError(f"Scriptwriter specialist returned an empty response for '{item.item_id}'")
        deliverable = ScriptWritingDeliverable.model_validate_json(resp.text)
        if deliverable.memory_updates:
            mem_store.apply_mutations(
                author_role="script_writer",
                proposals=deliverable.memory_updates,
                item_id=item.item_id,
            )
        mem_q = (deliverable.memory_query or "").strip()
        mem_q_key = f"{(deliverable.memory_query_bank or '').strip().lower()}::{mem_q.lower()}"
        has_deliverable_content = bool(
            deliverable.added_or_updated_characters
            or deliverable.added_or_updated_cutscenes
            or deliverable.added_or_updated_dialogue_trees
            or deliverable.added_or_updated_barks
        )
        if (
            mem_q
            and not has_deliverable_content
            and has_embed_client
            and mem_turn < max_mem_queries
            and mem_q_key not in asked_mem_queries
        ):
            asked_mem_queries.add(mem_q_key)
            hits = mem_store.query_memories(
                caller_role="script_writer",
                query_text=mem_q,
                bank_filter=deliverable.memory_query_bank,
                client=ctx.client,
            )
            mem_query_blocks.append(format_memory_query_results(mem_q, deliverable.memory_query_bank, hits))
            continue
        break
    assert deliverable is not None

    touched: List[str] = []
    with _CATALOG_LOCK:
        bible_path = ctx.cartridge_dir / "game_bible.json"
        nb_data: Dict[str, Any] = {
            "deliberation": deliverable.deliberation,
            "world_title": "Game World",
            "setting_summary": "",
            "characters": [],
            "cutscenes": [],
            "dialogue_trees": [],
            "dialogue_barks": [],
        }
        bible_data: Optional[Dict[str, Any]] = None
        if bible_path.is_file():
            try:
                bible_data = json.loads(bible_path.read_text(encoding="utf-8"))
                if isinstance(bible_data, dict) and isinstance(bible_data.get("narrative_bible"), dict):
                    nb_data.update(bible_data["narrative_bible"])
            except Exception:
                bible_data = None

        chars_by_id = {c.get("character_id"): c for c in nb_data.get("characters", []) if isinstance(c, dict)}
        for c in deliverable.added_or_updated_characters:
            chars_by_id[c.character_id] = c.model_dump()
        nb_data["characters"] = list(chars_by_id.values())

        cs_by_id = {c.get("cutscene_id"): c for c in nb_data.get("cutscenes", []) if isinstance(c, dict)}
        for cs in deliverable.added_or_updated_cutscenes:
            cs_by_id[cs.cutscene_id] = cs.model_dump()
        nb_data["cutscenes"] = list(cs_by_id.values())

        dt_by_id = {t.get("root_node_id"): t for t in nb_data.get("dialogue_trees", []) if isinstance(t, dict)}
        for dt in deliverable.added_or_updated_dialogue_trees:
            dt_by_id[dt.root_node_id] = dt.model_dump()
        nb_data["dialogue_trees"] = list(dt_by_id.values())

        barks_by_id = {b.get("bark_id"): b for b in nb_data.get("dialogue_barks", []) if isinstance(b, dict)}
        for bk in deliverable.added_or_updated_barks:
            barks_by_id[bk.bark_id] = bk.model_dump()
        nb_data["dialogue_barks"] = list(barks_by_id.values())

        for fname in ("narrative_bible.json", "narrative.json"):
            (ctx.cartridge_dir / fname).write_text(json.dumps(nb_data, indent=2), encoding="utf-8")
            touched.append(fname)
        if isinstance(bible_data, dict):
            bible_data["narrative_bible"] = nb_data
            bible_path.write_text(json.dumps(bible_data, indent=2), encoding="utf-8")
            touched.append("game_bible.json")

    safe_id = _safe_id(item.item_id)
    sheet_path = _item_dir(ctx, item) / f"script_sheet_{safe_id}_t{turn_index}.png"
    img = Image.new("RGB", (960, 540), (15, 23, 42))
    draw = ImageDraw.Draw(img)
    draw.rectangle([12, 12, 948, 528], outline=(56, 189, 248), width=2)
    lines = [
        f"SCRIPTWRITER TABLE READ SHEET — TASK [{item.item_id}] (Turn {turn_index})",
        f"Brief: {brief[:110]}",
        f"Notes: {(deliverable.narrative_notes or deliverable.deliberation)[:120]}",
        f"Authored: {len(deliverable.added_or_updated_cutscenes)} cutscenes | "
        f"{len(deliverable.added_or_updated_dialogue_trees)} dialogue trees | "
        f"{len(deliverable.added_or_updated_barks)} barks | "
        f"{len(deliverable.added_or_updated_characters)} characters",
        "-" * 96,
    ]
    for cs in deliverable.added_or_updated_cutscenes[:3]:
        lines.append(f"[Cutscene '{cs.title}' ({cs.cutscene_id}) trigger={cs.trigger}]")
        for beat in cs.beats[:4]:
            lines.append(f"  Beat {beat.beat_number} ({beat.speaker_id or 'stage'}): {beat.dialogue_line or beat.action_description}"[:115])
    for dt in deliverable.added_or_updated_dialogue_trees[:2]:
        lines.append(f"[Dialogue Tree root={dt.root_node_id} ({len(dt.nodes)} nodes)]")
        for nid, node in list(dt.nodes.items())[:3]:
            lines.append(f"  ({nid}) {node.speaker_id}: \"{node.text}\""[:115])
    for bk in deliverable.added_or_updated_barks[:6]:
        lines.append(f"[Bark {bk.bark_id}] ({bk.speaker_id} on {bk.trigger_type}): \"{bk.text}\""[:115])
    y = 26
    for ln in lines[:24]:
        draw.text((28, y), ln, fill=(226, 232, 240))
        y += 20
    img.save(sheet_path)

    summary = (
        f"Scriptwriter authored {len(deliverable.added_or_updated_cutscenes)} cutscene(s), "
        f"{len(deliverable.added_or_updated_dialogue_trees)} dialogue tree(s), "
        f"{len(deliverable.added_or_updated_barks)} bark(s), and "
        f"{len(deliverable.added_or_updated_characters)} character(s). "
        f"{deliverable.narrative_notes or deliverable.deliberation[:200]}"
    )
    return SpecialistReport(
        deliberation=deliverable.deliberation,
        turn_index=turn_index,
        specialist_agent="script_writer",
        brief=brief,
        summary=summary,
        touched_files=touched,
        proof_images=[str(sheet_path)],
    )


def _file_sha256_or_none(path: Path) -> Optional[str]:
    import hashlib

    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_swe_engineer_specialist(
    ctx: WorkerContext,
    item: WorkItem,
    brief: str,
    target_paths: Optional[List[str]] = None,
    turn_index: int = 0,
    _lock_already_held: bool = False,
    _task_baseline_snap_ref: Optional[Dict[str, Optional[str]]] = None,
) -> SpecialistReport:
    """Autonomous SWE Engineer specialist: architects or modifies GDScript modules, world/level layouts,
    scenes, entities, or UI based on the Task Lead's brief and verifies headless engine compilation/boot
    without forcing a bundled filmstrip capture."""
    import contextlib
    from engine.studio.context_curator import is_real_genai_client
    from engine.studio.models import TrackCodeSynthesisPayload

    agent = ctx.swe_agent
    if agent is None:
        raise StudioAgentError(f"No SWE agent is configured; cannot execute swe_engineer for '{item.item_id}'.")

    use_per_file_locks = ctx.file_lock_table is not None
    raw_lock = ctx.code_lock if ctx.code_lock is not None else threading.Lock()
    if use_per_file_locks:
        lock_cm: Any = contextlib.nullcontext()
        commit_gate_cm: Any = ctx.file_lock_table.commit_gate
    else:
        lock_cm = contextlib.nullcontext() if _lock_already_held else raw_lock
        commit_gate_cm = contextlib.nullcontext()

    fb = [(r.revision_index + 1, r.feedback) for r in item.revisions if r.feedback]
    canonical_root_gd = {"main.gd", "mechanics.gd", "scenes.gd", "entities.gd", "ui_hud.gd"}
    excluded_mods = set(getattr(item, "excluded_modules", None) or [])
    explicit_code_modules: List[str] = []
    for raw_mod in (target_paths or item.target_modules):
        clean_mod = str(raw_mod or "").strip().replace("\\", "/")
        if clean_mod.startswith("res://"):
            clean_mod = clean_mod[len("res://"):]
        elif clean_mod.startswith("res:/"):
            clean_mod = clean_mod[len("res:/"):]
        clean_mod = clean_mod.lstrip("/")
        if "/" in clean_mod and Path(clean_mod).name in canonical_root_gd:
            clean_mod = Path(clean_mod).name
        if clean_mod in excluded_mods or clean_mod.startswith("tests/"):
            continue
        if clean_mod.endswith(CODE_MODULE_SUFFIXES) and clean_mod not in explicit_code_modules:
            explicit_code_modules.append(clean_mod)
    if not explicit_code_modules and item.target_modules:
        for own_m in item.target_modules:
            clean_own = str(own_m or "").strip().replace("\\", "/").removeprefix("res://").lstrip("/")
            if (
                clean_own
                and not clean_own.startswith("tests/")
                and clean_own not in excluded_mods
                and clean_own.endswith(CODE_MODULE_SUFFIXES)
                and clean_own not in explicit_code_modules
            ):
                explicit_code_modules.append(clean_own)
    if item.kind == "level_map" and not explicit_code_modules and is_real_genai_client(ctx.client):
        explicit_code_modules = ["scenes.gd"]
    has_existing_layout = (ctx.cartridge_dir / "level_layout.json").is_file()
    explicitly_targets_layout = any("level_layout" in str(p) for p in (target_paths or [])) or (
        "level_layout.json" in (brief or "").lower()
    )
    wants_level_layout = (
        hasattr(agent, "formulate_level_layout")
        and (
            explicitly_targets_layout
            or (
                item.kind == "level_map"
                and (not has_existing_layout or not explicit_code_modules)
            )
            or (not explicit_code_modules and not hasattr(agent, "implement_requirement_track"))
        )
    )
    touched: List[str] = []
    proof_images: List[str] = []
    summaries: List[str] = []

    acquired_directly = False
    try:
        if use_per_file_locks and not _lock_already_held and ctx.file_lock_table is not None:
            initial_locks = list(explicit_code_modules or ["main.gd"])
            if wants_level_layout and "level_layout.json" not in initial_locks:
                initial_locks.append("level_layout.json")
            ok, conflicts = ctx.file_lock_table.try_acquire_many(item.item_id, initial_locks)
            if not ok:
                if ctx.board is not None and hasattr(ctx.board, "expand_item_target_modules"):
                    ctx.board.expand_item_target_modules(item.item_id, initial_locks)
                raise WorkerYielded(f"file_lock_busy:{','.join(conflicts)}")
            acquired_directly = True

        if wants_level_layout:
            from engine.studio.proofs import render_level_map
            guidance = f"[{item.item_id}] {item.title}: {item.spec}\nTask Lead Brief: {brief}"
            if fb:
                guidance += "\nFeedback to address:\n" + "\n".join(f"  rev {n}: {t}" for n, t in fb)
            safe_id = _safe_id(item.item_id)
            with lock_cm:
                with commit_gate_cm:
                    layout = agent.formulate_level_layout(ctx.cartridge_dir, force_refresh=True, guidance=guidance)
                    map_proof = render_level_map(layout, _item_dir(ctx, item) / f"level_map_{safe_id}_rev{len(item.revisions)}.png")
            touched.append("level_layout.json")
            proof_images.append(str(map_proof))
            summaries.append(
                f"Layout: {len(layout.world_regions)} regions, {len(layout.entities)} entities, "
                f"{len(layout.triggers)} triggers; world {layout.world_bounds} (~{layout.intended_world_scale_in_viewports:.1f} screens)."
            )
        elif item.kind == "level_map" and has_existing_layout:
            safe_id = _safe_id(item.item_id)
            existing_map_proof = _item_dir(ctx, item) / f"level_map_{safe_id}_rev{len(item.revisions)}.png"
            if existing_map_proof.is_file():
                proof_images.append(str(existing_map_proof))

        should_run_code = hasattr(agent, "implement_requirement_track") and (
            bool(explicit_code_modules) or not wants_level_layout
        )
        if should_run_code:
            modules = explicit_code_modules or ["main.gd"]
            scope_boundary_note = ""
            if excluded_mods:
                scope_boundary_note = (
                    f"\n\nCOUNCIL SCOPE BOUNDARY (MANDATORY): Modules {sorted(excluded_mods)} have been assigned to a dedicated prerequisite task. "
                    f"Do NOT include or modify {sorted(excluded_mods)} in your output—restrict your edits strictly to {modules}."
                )
            attached_proof_paths: List[str] = []
            item_art_dir = _item_dir(ctx, item)
            cur_rev_idx = len(item.revisions)
            if item_art_dir.is_dir():
                rev_strips = [
                    p
                    for p in sorted(item_art_dir.glob(f"*filmstrip*_rev{cur_rev_idx}.png"))
                    if p.is_file() and "_pass1." not in p.name.lower()
                ] or [
                    p
                    for p in sorted(item_art_dir.glob(f"*filmstrip*_rev{cur_rev_idx}.png"))
                    if p.is_file()
                ]
                if rev_strips:
                    attached_proof_paths.append(str(rev_strips[-1]))
            if not attached_proof_paths and item.revisions:
                for p in (item.revisions[-1].proof_paths or [])[-1:]:
                    if p and Path(p).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and p not in attached_proof_paths:
                        attached_proof_paths.append(p)
            telemetry_note = ""
            if attached_proof_paths:
                try:
                    from engine.studio.proofs import format_filmstrip_telemetry_progression

                    for p_path in attached_proof_paths:
                        t_prog = format_filmstrip_telemetry_progression(Path(p_path))
                        if t_prog:
                            telemetry_note = f"\n\nLATEST QA FILMSTRIP ENGINE TELEMETRY ({Path(p_path).name}):\n  {t_prog}"
                            break
                except Exception:
                    pass
            description = (
                f"{item.spec}\n\n"
                f"TASK LEAD BRIEF FOR SWE ENGINEER:\n{brief}\n\n"
                f"COUNCIL WILL JUDGE: {item.proof_spec}\n\n"
                f"QA HARNESS BOUNDARY (MANDATORY): `qa_test_engineer` exclusively authors `res://tests/test_{_safe_id(item.item_id)}.tscn` + `.gd` and injects external player inputs (`input_segments`), while the engine's Counterfactual Zero-Input Gate (`input_script=[]`) and Entrypoint Parity Gate (`res://main.tscn`) verify that the exact same player inputs satisfy all state assertions natively inside `res://main.tscn` WITHOUT any `tests/test_*.gd` wrapper code! NEVER create or modify any file under `tests/` and NEVER script `sim_tick` timers—implement and wire all authentic signal/input/entity connections directly in your production Target Modules ({modules})."
                f"{scope_boundary_note}"
                f"{telemetry_note}"
            )
            if fb:
                description += "\n\nCOUNCIL FEEDBACK TO ADDRESS (all of it):\n" + "\n".join(f"  rev {n}: {t}" for n, t in fb)
            track = RequirementTrack(
                deliberation=item.deliberation,
                id=item.item_id,
                title=item.title,
                description=description,
                target_modules=modules,
                acceptance_criteria=[
                    AcceptanceCriterion(
                        deliberation="Task Lead brief and Council proof request form the acceptance criterion.",
                        id=f"AC-{item.item_id}",
                        description=brief or item.proof_spec or item.spec,
                        verification_method="headless_playtest",
                    )
                ],
                status="in_progress",
            )
            bible_title = "game"
            bible_path = ctx.cartridge_dir / "game_bible.json"
            if bible_path.is_file():
                try:
                    bible_title = json.loads(bible_path.read_text(encoding="utf-8")).get("game_title") or bible_title
                except ValueError:
                    pass
            req_spec = GameRequirementsSpec(
                deliberation="Task-Lead-driven SWE specialist assignment",
                game_title=bible_title,
                requirements=[track],
                asset_manifest=[],
            )
            stash_path = _item_dir(ctx, item) / f"stashed_swe_payload_rev{len(item.revisions)}.json"
            stashed_bundle: Optional[TrackCodeSynthesisPayload] = None
            if stash_path.is_file():
                try:
                    stash_data = json.loads(stash_path.read_text(encoding="utf-8"))
                    if (
                        isinstance(stash_data, dict)
                        and stash_data.get("turn_index") == turn_index
                        and stash_data.get("brief") == brief
                        and isinstance(stash_data.get("payload"), dict)
                    ):
                        base_hashes = stash_data.get("base_hashes") or {}
                        stale_files = [
                            rel_f
                            for rel_f, expected_hash in base_hashes.items()
                            if _file_sha256_or_none(ctx.cartridge_dir / rel_f) != expected_hash
                        ]
                        if stale_files:
                            logger.info(
                                "Stashed SWE payload for '%s' invalidated because %s changed on disk while yielded; re-synthesizing.",
                                item.item_id,
                                stale_files,
                            )
                            stash_path.unlink(missing_ok=True)
                        else:
                            stashed_bundle = TrackCodeSynthesisPayload.model_validate(stash_data["payload"])
                            logger.info("Loaded valid stashed SWE payload for '%s' (turn %d).", item.item_id, turn_index)
                except Exception as stash_exc:
                    logger.warning("Ignoring unreadable stashed SWE payload at %s: %s", stash_path, stash_exc)

            turn_code_diffs: Dict[str, str] = {}
            with lock_cm:
                if (
                    is_real_genai_client(ctx.client)
                    and hasattr(agent, "formulate_level_layout")
                    and (ctx.cartridge_dir / "game_bible.json").is_file()
                    and not (ctx.cartridge_dir / "level_layout.json").is_file()
                ):
                    try:
                        with commit_gate_cm:
                            agent.formulate_level_layout(ctx.cartridge_dir, force_refresh=False)
                    except Exception as layout_exc:
                        logger.warning("Pre-code formulate_level_layout skipped: %s", layout_exc)
                from engine.studio.models import TOKEN_USAGE_TRACKER
                pre_turn_snap = _snapshot_modules(
                    ctx.cartridge_dir,
                    modules,
                    scoped_only=use_per_file_locks,
                )

                def _pre_write_lock_check(normalized_files: List[str], payload_obj: Any) -> None:
                    live_excluded: set[str] = set(excluded_mods)
                    if ctx.board is not None and hasattr(ctx.board, "get"):
                        try:
                            live_b_item = ctx.board.get(item.item_id)
                            live_excluded.update(getattr(live_b_item, "excluded_modules", None) or [])
                            if live_b_item.state in ("blocked", "cancelled") or (
                                hasattr(ctx.board, "_deps_satisfied") and not ctx.board._deps_satisfied(live_b_item)
                            ):
                                raise WorkerYielded("spawned_prerequisite_tasks")
                        except WorkerYielded:
                            raise
                        except Exception:
                            pass
                    if live_excluded:
                        if hasattr(payload_obj, "modified_files") and isinstance(payload_obj.modified_files, dict):
                            pruned_keys = [
                                k for k in list(payload_obj.modified_files.keys())
                                if CartridgeFileLockTable.normalize_path(k) in live_excluded
                            ]
                            if pruned_keys and len(pruned_keys) < len(payload_obj.modified_files):
                                for pk in pruned_keys:
                                    payload_obj.modified_files.pop(pk, None)
                                normalized_files[:] = [f for f in normalized_files if f not in live_excluded]
                        non_excluded_files = [f for f in normalized_files if f not in live_excluded]
                        if non_excluded_files and len(non_excluded_files) < len(normalized_files):
                            if hasattr(payload_obj, "files") and isinstance(payload_obj.files, list):
                                payload_obj.files = [
                                    fspec for fspec in payload_obj.files
                                    if CartridgeFileLockTable.normalize_path(getattr(fspec, "filename", "")) not in live_excluded
                                ]
                            if hasattr(payload_obj, "patches") and isinstance(payload_obj.patches, list):
                                payload_obj.patches = [
                                    pspec for pspec in payload_obj.patches
                                    if CartridgeFileLockTable.normalize_path(getattr(pspec, "filename", "")) not in live_excluded
                                ]
                            normalized_files[:] = non_excluded_files
                    if ctx.file_lock_table is not None:
                        held = ctx.file_lock_table.held_paths(item.item_id)
                        missing = [f for f in normalized_files if f not in held]
                        if missing:
                            ok_up, conflicts_up = ctx.file_lock_table.try_acquire_many(item.item_id, missing)
                            if ctx.board is not None and hasattr(ctx.board, "expand_item_target_modules"):
                                ctx.board.expand_item_target_modules(item.item_id, normalized_files)
                            if not ok_up:
                                base_hashes = {
                                    rel_f: _file_sha256_or_none(ctx.cartridge_dir / rel_f)
                                    for rel_f in normalized_files
                                }
                                stash_path.parent.mkdir(parents=True, exist_ok=True)
                                stash_path.write_text(
                                    json.dumps(
                                        {
                                            "turn_index": turn_index,
                                            "brief": brief,
                                            "required_files": normalized_files,
                                            "base_hashes": base_hashes,
                                            "payload": payload_obj.model_dump() if hasattr(payload_obj, "model_dump") else {},
                                        },
                                        indent=2,
                                    ),
                                    encoding="utf-8",
                                )
                                record_turn_substep(
                                    "lock_yielded",
                                    f"Stashed synthesized payload and yielding for busy file lock(s): {conflicts_up}",
                                    details={"conflicts": conflicts_up, "required_files": normalized_files},
                                    specialist_agent="swe_engineer",
                                )
                                raise WorkerYielded(f"file_lock_busy:{','.join(conflicts_up)}")
                            extra_snap = _snapshot_modules(ctx.cartridge_dir, missing, scoped_only=True)
                            live_turn_st = getattr(_ACTIVE_TURN_CTX, "state", None)
                            baseline_ref = (
                                live_turn_st.get("_task_baseline_snap_ref")
                                if isinstance(live_turn_st, dict)
                                else _task_baseline_snap_ref
                            )
                            for k_m, v_m in extra_snap.items():
                                if k_m not in pre_turn_snap:
                                    pre_turn_snap[k_m] = v_m
                                if isinstance(baseline_ref, dict) and k_m not in baseline_ref:
                                    baseline_ref[k_m] = v_m
                            for k_m in missing:
                                if k_m not in modules:
                                    modules.append(k_m)
                            record_turn_substep(
                                "lock_upgraded",
                                f"Dynamically acquired additional file lock(s): {missing}",
                                details={"acquired": missing},
                                specialist_agent="swe_engineer",
                            )
                    if stash_path.is_file():
                        stash_path.unlink(missing_ok=True)

                import inspect
                swe_extra_kwargs: Dict[str, Any] = {}
                try:
                    sig = inspect.signature(agent.implement_requirement_track)
                    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    if accepts_var_kw or "extra_image_paths" in sig.parameters:
                        swe_extra_kwargs["extra_image_paths"] = list(attached_proof_paths)
                    if accepts_var_kw or "board" in sig.parameters:
                        swe_extra_kwargs["board"] = ctx.board
                    if use_per_file_locks and ctx.file_lock_table is not None:
                        if accepts_var_kw or "pre_write_hook" in sig.parameters:
                            swe_extra_kwargs["pre_write_hook"] = _pre_write_lock_check
                        if accepts_var_kw or "commit_gate" in sig.parameters:
                            swe_extra_kwargs["commit_gate"] = ctx.file_lock_table.commit_gate
                        if accepts_var_kw or "allowed_files" in sig.parameters:
                            swe_extra_kwargs["allowed_files"] = ctx.file_lock_table.held_paths(item.item_id) | set(modules)
                        if stashed_bundle is not None and (accepts_var_kw or "stashed_bundle" in sig.parameters):
                            swe_extra_kwargs["stashed_bundle"] = stashed_bundle
                except Exception:
                    pass

                written_map = agent.implement_requirement_track(
                    track=track,
                    staging_dir=ctx.cartridge_dir,
                    requirements_spec=req_spec,
                    **swe_extra_kwargs,
                )
                swe_extra_kwargs.pop("stashed_bundle", None)
                diag_req = (getattr(agent, "last_diagnostic_test_scene_request", "") or "").strip()
                if diag_req:
                    logger.info("SWE Engineer requested diagnostic test scene for '%s': %s", item.item_id, diag_req)
                    record_turn_substep(
                        "running_diagnostic_qa",
                        f"Running QA diagnostic test scene requested by SWE Engineer: {diag_req[:120]}",
                        details={"diagnostic_test_scene_request": diag_req},
                        specialist_agent="qa_test_engineer",
                    )
                    qa_diag_report = run_qa_test_engineer_specialist(
                        ctx,
                        item,
                        brief=f"[SWE DIAGNOSTIC TEST SCENE REQUEST — BUILD TEST SCENE WITH OVERLAYS/TELEMETRY TO ISOLATE ISSUE]:\n{diag_req}",
                        turn_index=turn_index,
                        _lock_already_held=True,
                    )
                    for p_img in qa_diag_report.proof_images:
                        if p_img not in proof_images:
                            proof_images.append(p_img)
                        if "extra_image_paths" in swe_extra_kwargs and p_img not in swe_extra_kwargs["extra_image_paths"]:
                            swe_extra_kwargs["extra_image_paths"].append(p_img)
                    for q_tf in qa_diag_report.touched_files:
                        if q_tf not in touched:
                            touched.append(q_tf)
                    summaries.append(f"[SWE Diagnostic QA Scene]: {qa_diag_report.summary}")
                    if not written_map:
                        track = track.model_copy(
                            update={
                                "description": (
                                    f"{track.description}\n\n"
                                    f"SWE DIAGNOSTIC TEST SCENE RESULTS & TELEMETRY (use this to implement the fix now):\n"
                                    f"{qa_diag_report.summary}"
                                )
                            }
                        )
                        written_map = agent.implement_requirement_track(
                            track=track,
                            staging_dir=ctx.cartridge_dir,
                            requirements_spec=req_spec,
                            **swe_extra_kwargs,
                        )
                if ctx.code_verify is not None:
                    record_turn_substep(
                        "running_tests",
                        f"Running headless engine verification & unit checks on {modules}",
                        details={"modules": list(modules)},
                    )
                    try:
                        with commit_gate_cm:
                            ctx.code_verify(ctx.cartridge_dir, track)
                        TOKEN_USAGE_TRACKER.reset_swe_pev_streak()
                    except StudioPausedError:
                        raise
                    except Exception as verify_exc:
                        if is_real_genai_client(ctx.client):
                            logger.warning(
                                "SWE code_verify failed on first pass for '%s' (%s); retrying with prior_error feedback.",
                                item.item_id,
                                verify_exc,
                            )
                            record_turn_substep(
                                "running_tests",
                                f"Verification failed ({str(verify_exc)[:120]}); self-healing and re-verifying {modules}",
                            )
                            written_map = agent.implement_requirement_track(
                                track=track,
                                staging_dir=ctx.cartridge_dir,
                                requirements_spec=req_spec,
                                prior_error=str(verify_exc),
                                **swe_extra_kwargs,
                            )
                            with commit_gate_cm:
                                ctx.code_verify(ctx.cartridge_dir, track)
                            TOKEN_USAGE_TRACKER.reset_swe_pev_streak()
                        else:
                            raise
                turn_code_diffs = _compute_turn_code_diffs(
                    ctx.cartridge_dir,
                    pre_turn_snap,
                    scoped_only=use_per_file_locks,
                )
            actual_modified: List[str] = []
            for m_diff in turn_code_diffs.keys():
                norm_m = CartridgeFileLockTable.normalize_path(str(m_diff))
                if norm_m and norm_m not in actual_modified:
                    actual_modified.append(norm_m)
            if isinstance(written_map, dict):
                for m_written in written_map.keys():
                    norm_w = CartridgeFileLockTable.normalize_path(str(m_written))
                    if norm_w and norm_w not in actual_modified:
                        actual_modified.append(norm_w)
            if not actual_modified and not is_real_genai_client(ctx.client) and written_map is None:
                actual_modified = list(modules)
            for m in actual_modified:
                if m not in touched:
                    touched.append(m)
            untouched_requested = [
                CartridgeFileLockTable.normalize_path(str(m))
                for m in modules
                if CartridgeFileLockTable.normalize_path(str(m))
                and CartridgeFileLockTable.normalize_path(str(m)) not in actual_modified
            ]
            untouched_suffix = (
                f" (UNTOUCHED requested target_modules: {untouched_requested})"
                if untouched_requested
                else ""
            )
            summaries.append(
                f"Implemented '{item.item_id}' — verified modified on disk: {actual_modified}{untouched_suffix}; engine verification passed. "
                f"Available actions: {available_input_actions(ctx.cartridge_dir)}."
            )
    finally:
        if acquired_directly and ctx.file_lock_table is not None:
            ctx.file_lock_table.release_all(item.item_id)

    from datetime import datetime, timezone
    return SpecialistReport(
        deliberation=f"SWE Engineer executed brief for '{item.item_id}'.",
        turn_index=turn_index,
        specialist_agent="swe_engineer",
        brief=brief,
        summary=" ".join(summaries),
        touched_files=touched,
        proof_images=proof_images,
        verification_passed=True,
        code_diffs=turn_code_diffs if "turn_code_diffs" in locals() else {},
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def run_qa_test_engineer_specialist(
    ctx: WorkerContext,
    item: WorkItem,
    brief: str,
    target_paths: Optional[List[str]] = None,
    turn_index: int = 0,
    _lock_already_held: bool = False,
) -> SpecialistReport:
    """Autonomous QA Test Engineer specialist: designs and executes a scripted Godot headless test scenario
    (`FilmstripPlan` with `setup_calls`, `input_segments`, `mouse_click_pos`, `state_assertions`, `capture_ticks`)
    based on the Task Lead's verification brief and returns the captured filmstrip proof and telemetry."""
    import contextlib

    use_per_file_locks = ctx.file_lock_table is not None
    raw_lock = ctx.code_lock if ctx.code_lock is not None else threading.Lock()
    if use_per_file_locks and ctx.file_lock_table is not None:
        lock_cm: Any = contextlib.nullcontext()
        commit_gate_cm: Any = ctx.file_lock_table.commit_gate
    else:
        lock_cm = contextlib.nullcontext() if _lock_already_held else raw_lock
        commit_gate_cm = contextlib.nullcontext()
    qa_item = item.model_copy(
        update={
            "proof_spec": f"{item.proof_spec}\n\nTASK LEAD QA VERIFICATION BRIEF:\n{brief}".strip(),
        }
    )
    with lock_cm:
        record_turn_substep(
            "planning_qa",
            f"Planning headless QA filmstrip scenario & state assertions for '{item.item_id}'",
        )
        from engine.studio.context_curator import is_real_genai_client
        from engine.studio.proofs import capture_filmstrip, format_filmstrip_telemetry_progression

        strip_prefix = "level_filmstrip" if item.kind == "level_map" else "filmstrip"
        safe_id = _safe_id(item.item_id)
        out_strip_path = _item_dir(ctx, item) / f"{strip_prefix}_{safe_id}_rev{len(item.revisions)}.png"
        qa_touched_files: List[str] = []

        def _stage_and_run_plan(cur_plan: FilmstripPlan) -> Tuple[Path, str, List[Dict[str, Any]], List[Dict[str, Any]]]:
            defect_rep = (getattr(cur_plan, "production_defect_report", "") or "").strip()
            if defect_rep:
                record_turn_substep(
                    "error",
                    f"QA diagnosed production defect: {defect_rep[:220]}",
                )
                raise StudioAgentError(f"QA Diagnostic Production Defect Report: {defect_rep}")
            cur_script = _serialize_input_segments(cur_plan)
            cur_setup_calls = [
                {
                    "at_tick": c.at_tick,
                    "target_node_or_property": c.target_node_or_property,
                    "method_name": c.method_name,
                    "args_json": c.args_json,
                }
                for c in (getattr(cur_plan, "setup_calls", None) or [])
            ]
            cur_resolved_scene = ""
            raw_scene = (getattr(cur_plan, "target_scene", "") or "").strip()
            test_gd_code = (getattr(cur_plan, "test_scene_gd_content", "") or "").strip()
            if test_gd_code:
                rel_scene_base = raw_scene
                if rel_scene_base.startswith("res://"):
                    rel_scene_base = rel_scene_base[len("res://"):]
                elif rel_scene_base.startswith("res:/"):
                    rel_scene_base = rel_scene_base[len("res:/"):]
                rel_scene_base = rel_scene_base.lstrip("/")
                if not rel_scene_base or rel_scene_base in ("main.tscn", "main.gd") or not rel_scene_base.startswith("tests/"):
                    rel_scene_base = f"tests/test_{safe_id}.tscn"
                if rel_scene_base.endswith(".gd"):
                    rel_gd_path = rel_scene_base
                    rel_tscn_path = rel_scene_base[:-3] + ".tscn"
                elif rel_scene_base.endswith(".tscn"):
                    rel_tscn_path = rel_scene_base
                    rel_gd_path = rel_scene_base[:-5] + ".gd"
                else:
                    rel_tscn_path = f"{rel_scene_base}.tscn"
                    rel_gd_path = f"{rel_scene_base}.gd"
                abs_gd_path = ctx.cartridge_dir / rel_gd_path
                abs_tscn_path = ctx.cartridge_dir / rel_tscn_path
                self_refs = (
                    f"res://{rel_gd_path}",
                    f"res://{rel_tscn_path}",
                )
                for s_ref in self_refs:
                    if re.search(
                        rf"\b(?:preload|load)\s*\(\s*['\"]" + re.escape(s_ref) + rf"['\"]\s*\)",
                        test_gd_code,
                    ):
                        raise StudioAgentError(
                            f"Self-recursive QA test script rejected in res://{rel_gd_path}: script calls `load`/`preload` on its own "
                            f"path `{s_ref}`, which causes infinite `_ready()` scene-instantiation recursion and freezes Godot. "
                            "Define an inner `class DebugOverlay extends Node2D:` (or draw directly in the root test node's `_draw()`) "
                            f"instead of loading `{s_ref}` inside itself."
                        )
                abs_gd_path.parent.mkdir(parents=True, exist_ok=True)
                abs_gd_path.write_text(test_gd_code + "\n", encoding="utf-8")
                tscn_content = (
                    '[gd_scene load_steps=2 format=3]\n\n'
                    f'[ext_resource type="Script" path="res://{rel_gd_path}" id="1_test"]\n\n'
                    '[node name="TestScene" type="Node2D"]\n'
                    'script = ExtResource("1_test")\n'
                )
                abs_tscn_path.write_text(tscn_content, encoding="utf-8")
                raw_scene = f"res://{rel_tscn_path}"
                for tf in (rel_gd_path, rel_tscn_path):
                    if tf not in qa_touched_files:
                        qa_touched_files.append(tf)
                record_turn_substep(
                    "editing_file",
                    f"Authoring isolated QA test scene res://{rel_tscn_path} + res://{rel_gd_path}",
                    details={"files": [rel_gd_path, rel_tscn_path]},
                )
            if not raw_scene:
                for cand_src in [brief or ""] + list(target_paths or []):
                    m_tscn = re.search(r"(?:res://)?(tests/[A-Za-z0-9_./-]+\.tscn)", str(cand_src))
                    if m_tscn:
                        raw_scene = f"res://{m_tscn.group(1)}"
                        break
            if raw_scene:
                rel_scene = raw_scene
                if rel_scene.startswith("res://"):
                    rel_scene = rel_scene[len("res://"):]
                elif rel_scene.startswith("res:/"):
                    rel_scene = rel_scene[len("res:/"):]
                rel_scene = rel_scene.lstrip("/")
                if (ctx.cartridge_dir / rel_scene).is_file():
                    cur_resolved_scene = f"res://{rel_scene}"
                    if rel_scene.startswith("tests/") and rel_scene.endswith(".tscn"):
                        disk_gd_rel = rel_scene[:-5] + ".gd"
                        disk_gd_abs = ctx.cartridge_dir / disk_gd_rel
                        if disk_gd_abs.is_file():
                            disk_gd_text = disk_gd_abs.read_text(encoding="utf-8", errors="ignore")
                            for s_ref in (f"res://{disk_gd_rel}", f"res://{rel_scene}"):
                                if re.search(
                                    rf"\b(?:preload|load)\s*\(\s*['\"]" + re.escape(s_ref) + rf"['\"]\s*\)",
                                    disk_gd_text,
                                ):
                                    if disk_gd_rel not in qa_touched_files:
                                        qa_touched_files.append(disk_gd_rel)
                                    raise StudioAgentError(
                                        f"Self-recursive QA test script rejected in res://{disk_gd_rel}: script calls `load`/`preload` on its own "
                                        f"path `{s_ref}`, which causes infinite `_ready()` scene-instantiation recursion and freezes Godot. "
                                        "Provide fresh `test_scene_gd_content` using an inner `class DebugOverlay extends Node2D:` "
                                        f"instead of loading `{s_ref}` inside itself."
                                    )
            record_turn_substep(
                "rendering_scene",
                f"Rendering scene {cur_resolved_scene or 'res://main.tscn'} across ticks {list(cur_plan.capture_ticks)}",
                details={"scene": cur_resolved_scene or "res://main.tscn", "ticks": list(cur_plan.capture_ticks)},
            )
            is_diagnostic_probe = (
                item.item_id == "council_diag"
                or str(brief or "").lstrip().startswith(
                    ("[SWE DIAGNOSTIC TEST SCENE REQUEST", "[COUNCIL DIAGNOSTIC TEST SCENE REQUEST")
                )
            )
            with commit_gate_cm:
                if ctx.filmstrip_fn is None:
                    cur_strip = capture_filmstrip(
                        ctx.cartridge_dir,
                        ticks=list(cur_plan.capture_ticks),
                        out_path=out_strip_path,
                        input_script=cur_script,
                        setup_script=cur_setup_calls or None,
                        verify_counterfactual=(item.kind != "level_map" and bool(cur_script)),
                        state_assertions=list(getattr(cur_plan, "state_assertions", None) or []),
                        target_scene=cur_resolved_scene,
                        verify_entrypoint_parity=(item.kind == "integration" and not is_diagnostic_probe),
                    )
                else:
                    cur_strip = ctx.filmstrip_fn(
                        ctx.cartridge_dir,
                        ticks=list(cur_plan.capture_ticks),
                        out_path=out_strip_path,
                        input_script=cur_script,
                    )
            return cur_strip, cur_resolved_scene, cur_script, cur_setup_calls

        plan = plan_filmstrip(ctx, qa_item)
        try:
            strip, resolved_scene, script, setup_calls = _stage_and_run_plan(plan)
        except StudioAgentError as qa_pass1_err:
            err_str = str(qa_pass1_err)
            is_assertion_fail = out_strip_path.is_file() and (
                "QA filmstrip state assertions failed" in err_str
                or "Entrypoint Parity Gate failed" in err_str
            )
            is_test_script_error = (
                "Godot headless smoke check failed in res://tests/" in err_str
                or "Godot simulation failed or timed out in res://tests/" in err_str
                or "Self-recursive QA test script rejected" in err_str
            )
            if is_real_genai_client(ctx.client) and ctx.filmstrip_fn is None and (is_assertion_fail or is_test_script_error):
                t_prog_p1 = format_filmstrip_telemetry_progression(out_strip_path) if out_strip_path.is_file() else ""
                logger.info(
                    "QA Pass 1 failed for '%s' (%s); running Pass 2 self-correction.",
                    item.item_id,
                    qa_pass1_err,
                )
                record_turn_substep(
                    "planning_qa",
                    f"Self-correcting QA test scene/script after Pass 1 failure ({err_str[:140]})",
                )
                p1_copy: Optional[Path] = None
                if out_strip_path.is_file():
                    p1_copy = out_strip_path.with_name(out_strip_path.stem + "_pass1.png")
                    try:
                        p1_copy.write_bytes(out_strip_path.read_bytes())
                    except Exception:
                        p1_copy = out_strip_path
                plan = plan_filmstrip(
                    ctx,
                    qa_item,
                    prior_capture_feedback=f"{qa_pass1_err}\n{t_prog_p1}".strip(),
                    prior_strip_path=p1_copy,
                )
                try:
                    strip, resolved_scene, script, setup_calls = _stage_and_run_plan(plan)
                except StudioAgentError as qa_pass2_err:
                    p2_err = str(qa_pass2_err)
                    if (
                        "Godot headless smoke check failed in res://tests/" in p2_err
                        or "Godot simulation failed or timed out in res://tests/" in p2_err
                        or "Self-recursive QA test script rejected" in p2_err
                    ):
                        for tf in qa_touched_files:
                            if tf.startswith("tests/"):
                                (ctx.cartridge_dir / tf).unlink(missing_ok=True)
                    raise
            else:
                if is_test_script_error:
                    for tf in qa_touched_files:
                        if tf.startswith("tests/"):
                            (ctx.cartridge_dir / tf).unlink(missing_ok=True)
                raise
        record_turn_substep(
            "captured_proof",
            f"Captured QA filmstrip {Path(strip).name}",
            details={"proof_image": str(strip)},
        )
    extra_qa = ""
    if resolved_scene:
        extra_qa += f" Target scene: {resolved_scene}."
    if setup_calls:
        extra_qa += f" Setup calls: {[c['method_name'] for c in setup_calls]}."
    if getattr(plan, "state_assertions", None):
        extra_qa += f" Assertions: {[(a.at_tick, a.snapshot_key, a.expected_value_or_change) for a in plan.state_assertions]}."
    try:
        from engine.studio.proofs import format_filmstrip_telemetry_progression

        t_prog = format_filmstrip_telemetry_progression(Path(strip))
        if t_prog:
            extra_qa += f" {t_prog}."
    except Exception:
        pass

    council_summary = (getattr(plan, "council_test_scene_summary", "") or "").strip()
    overlays_strat = (getattr(plan, "overlays_and_telemetry_strategy", "") or "").strip()

    qa_passed = True
    audit_block = ""
    if is_real_genai_client(ctx.client) and ctx.filmstrip_fn is None and Path(strip).is_file():
        try:
            from google.genai import types

            record_turn_substep(
                "auditing_captured_proof",
                f"Running QA post-capture visual & framing audit on {Path(strip).name}",
            )
            audit_prompt = (
                f"You are the Autonomous QA Test Engineer performing a SKEPTICAL POST-CAPTURE VISUAL AUDIT on the filmstrip you just captured for TASK [{item.item_id}] ({item.kind}): {item.title}\n"
                f"SPEC: {item.spec}\n"
                f"PROOF SPEC: {item.proof_spec}\n"
                f"TASK LEAD BRIEF: {brief}\n"
                f"CAPTURED TICKS: {list(plan.capture_ticks)}\n"
                f"PRE-CAPTURE TEST SCENE SUMMARY: {council_summary}\n"
                f"TELEMETRY SUMMARY: {extra_qa}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Inspect EVERY single frame of the attached `filmstrip_*.png` image from Frame 1 to the final frame.\n"
                f"2. Strategize from first principles what a live player and the Design Council need to see verified for this specific task:\n"
                f"   - Does the rendered viewport clearly frame the active subjects across all frames without losing them off-screen?\n"
                f"   - Are UI/HUD panels, labels, and world/test overlays legible and free of text collisions, panel-border overflows, overlaps, squished icons, cut-off/bleeding sprite-sheet slicing, or unintended production debug clutter?\n"
                f"   - Are world-space indicators/vectors aligned with the actual visual sprites they represent?\n"
                f"   - Across `CAPTURED TICKS` (60 ticks ~ 1 second), do the player inputs produce a responsive, human-playable rate of motion/change rather than a static image or a nearly imperceptible crawl?\n"
                f"3. In `observed_frames_summary`, describe objectively and unsentimentally what is ACTUALLY visible inside the 16:9 viewport of each frame (zero cheerleading or flattery).\n"
                f"4. In `post_capture_test_scene_summary`, write a candid, factual post-capture briefing for the Council explaining what the test scene staged, what inputs/overlays ran, and what the rendered frames show.\n"
                f"5. If any frame fails visual framing, has overlapping/overflowing UI or cut-off sprite slicing, lacks test-scene diagnostic proof, or fails to show responsive progression for THIS task's `SPEC`/`PROOF SPEC`, list each issue in `unaddressed_defects` and set `qa_passed = False`; once THIS task's `SPEC` and `PROOF SPEC` are genuinely proven without visual/causal glitches, set `qa_passed = True` without inventing out-of-scope wishlist demands!\n"
                f"Provide `deliberation` FIRST and `qa_passed` LAST."
            )
            qa_media_res = getattr(getattr(types, "MediaResolution", None), "MEDIA_RESOLUTION_HIGH", None)
            strip_part = (
                types.Part.from_bytes(data=Path(strip).read_bytes(), mime_type="image/png", media_resolution=qa_media_res)
                if qa_media_res is not None
                else types.Part.from_bytes(data=Path(strip).read_bytes(), mime_type="image/png")
            )
            audit_contents: Any = [
                audit_prompt,
                f"CAPTURED FILMSTRIP TO AUDIT ({Path(strip).name}):",
                strip_part,
            ]
            audit_resp = call_gemini_with_transient_retry(
                lambda: ctx.client.models.generate_content(
                    model=ctx.model_name,
                    contents=audit_contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=QAPostCaptureAudit,
                        thinking_config=bounded_thinking_config(512),
                    ),
                ),
                label=f"qa_post_capture_audit[{item.item_id}]",
            )
            if audit_resp and audit_resp.text:
                audit_obj = QAPostCaptureAudit.model_validate_json(audit_resp.text)
                qa_passed = bool(audit_obj.qa_passed)
                if (audit_obj.post_capture_test_scene_summary or "").strip():
                    council_summary = audit_obj.post_capture_test_scene_summary.strip()
                defects_str = "; ".join(audit_obj.unaddressed_defects) if audit_obj.unaddressed_defects else "none"
                if not qa_passed or audit_obj.unaddressed_defects:
                    audit_block = (
                        f" QA Self-Check flagged issues for Task Lead (qa_passed={qa_passed}): "
                        f"observed_frames={audit_obj.observed_frames_summary!r} | unaddressed_defects=[{defects_str}]."
                    )
                else:
                    audit_block = f" Observed frames: {audit_obj.observed_frames_summary}"
        except Exception as audit_exc:
            logger.warning("QAPostCaptureAudit skipped for '%s': %s", item.item_id, audit_exc)

    qa_briefing = ""
    if council_summary:
        qa_briefing += f" QA Test Scene Summary for Council: {council_summary}"
    if overlays_strat:
        qa_briefing += f" QA Overlays & Telemetry Strategy: {overlays_strat}"
    summary = (
        f"Filmstrip ticks {plan.capture_ticks}.{qa_briefing}{extra_qa}{audit_block}"
    )
    return SpecialistReport(
        deliberation=plan.deliberation,
        turn_index=turn_index,
        specialist_agent="qa_test_engineer",
        brief=brief,
        summary=summary,
        touched_files=qa_touched_files,
        proof_images=[str(strip)],
        verification_passed=qa_passed,
    )


def validate_task_lead_decision(
    decision: TaskLeadDecision,
    specialists: List[str],
    collected_proofs: List[str],
    board: Any = None,
    parent_item_id: str = "",
    item_kind: str = "",
    turn_history: Optional[Sequence[SpecialistReport]] = None,
) -> List[str]:
    """Structural validation of a `TaskLeadDecision` turn."""
    issues: List[str] = []
    if not decision.current_strategy.strip():
        issues.append("current_strategy must be a non-empty description of your strategy.")
    if decision.action == "delegate":
        if decision.specialist_agent not in specialists:
            issues.append(
                f"specialist_agent '{decision.specialist_agent}' is not available; choose from {specialists}."
            )
        if not decision.brief.strip():
            issues.append("brief must be non-empty when action == 'delegate'.")
        if decision.specialist_agent == "swe_engineer" and decision.target_paths:
            bad_swe_test_paths = [
                p for p in decision.target_paths
                if str(p or "").strip().replace("\\", "/").removeprefix("res://").lstrip("/").startswith("tests/")
            ]
            if bad_swe_test_paths:
                issues.append(
                    f"swe_engineer cannot target QA test harness path(s) {bad_swe_test_paths} in `target_paths`. "
                    "'swe_engineer' strictly implements production game modules (`main.gd`, `scenes/...`, `mechanics.gd`, "
                    "`entities.gd`, `ui_hud.gd`) so live `res://main.tscn` passes the Entrypoint Parity Gate under real player "
                    "inputs; 'qa_test_engineer' exclusively owns `res://tests/test_<item_id>.tscn` + `.gd` and `input_segments`."
                )
    elif decision.action == "spawn_tasks":
        if not decision.spawned_tasks:
            issues.append("spawned_tasks must contain at least one WorkItemProposal when action == 'spawn_tasks'.")
        elif board is not None and hasattr(board, "validate_proposals"):
            existing_ids = {i.item_id for i in board.items()} if hasattr(board, "items") else set()
            sanitized = []
            for p in decision.spawned_tasks:
                if decision.wait_for_spawned_tasks and p.item_id in existing_ids and p.item_id != parent_item_id:
                    continue
                if decision.wait_for_spawned_tasks and parent_item_id and parent_item_id in p.depends_on:
                    p_copy = p.model_copy(deep=True)
                    p_copy.depends_on = [d for d in p.depends_on if d != parent_item_id]
                    sanitized.append(p_copy)
                else:
                    sanitized.append(p)
            board_issues = board.validate_proposals(sanitized) if sanitized else []
            issues.extend(board_issues)
    elif decision.action == "submit_to_council":
        if not collected_proofs:
            issues.append(
                "Cannot submit_to_council before any proof images have been produced; delegate to a specialist "
                f"({specialists}) to build and capture proof first."
            )
        elif item_kind == "level_map" and "qa_test_engineer" in specialists:
            has_engine_filmstrip = any("filmstrip" in Path(p).name.lower() for p in collected_proofs)
            if not has_engine_filmstrip:
                issues.append(
                    "Cannot submit_to_council for a 'level_map' task with only a 2D wireframe diagram (level_map_*.png). "
                    "You MUST delegate to 'qa_test_engineer' to capture a live in-engine Godot viewport filmstrip "
                    "(level_filmstrip_*.png / filmstrip_*.png) proving the world geometry, surface textures, and Camera2D traversal in Godot."
                )
        if turn_history:
            qa_turns = [t for t in turn_history if getattr(t, "specialist_agent", "") == "qa_test_engineer"]
            if qa_turns and getattr(qa_turns[-1], "verification_passed", True) is False:
                issues.append(
                    "Cannot submit_to_council because the latest 'qa_test_engineer' Post-Capture Visual Audit failed "
                    f"(verification_passed=False: {str(getattr(qa_turns[-1], 'summary', ''))[-360:]}). "
                    "You MUST delegate to 'swe_engineer' (or run a diagnostic test scene) to fix the visual/camera-framing defect and re-run 'qa_test_engineer' first."
                )
        if decision.selected_proof_paths:
            unknown = [
                p for p in decision.selected_proof_paths if p not in collected_proofs and not Path(p).is_file()
            ]
            if unknown:
                issues.append(
                    f"selected_proof_paths contains non-existent proof files {unknown}; available proofs: {collected_proofs}."
                )
    return issues


class _LegacyFakeClientDirectPayload(Exception):
    """Raised internally when a legacy unit-test fake client returns an AssetRequirement or FilmstripPlan
    directly instead of a TaskLeadDecision schema."""


def _build_task_lead_multimodal_parts(
    ctx: WorkerContext,
    collected_proofs: List[str],
    item_kind: str = "level_map",
) -> List[Any]:
    """Builds multimodal image parts (reference 16:9 Concept Art Mockup when visual/spatial + latest collected proof image)
    so the Task Lead Strategist can see the target visual aesthetic when relevant and visually inspect
    captured proofs before submitting to the Council."""
    from engine.studio.context_curator import ContextCuratorAgent
    from google.genai import types

    parts: List[Any] = []
    if ContextCuratorAgent._should_attach_concept_mockup("task_lead", item_kind):
        bible_path = Path(ctx.cartridge_dir) / "game_bible.json"
        if bible_path.is_file():
            try:
                bible_data = json.loads(bible_path.read_text(encoding="utf-8"))
                mockups = bible_data.get("visual_mockups") if isinstance(bible_data, dict) else None
                if isinstance(mockups, list) and mockups:
                    rel = str(mockups[0].get("image_path") or "").strip()
                    if rel:
                        mpath = Path(rel) if Path(rel).is_absolute() else (Path(ctx.cartridge_dir) / rel)
                        if mpath.is_file():
                            mime = "image/jpeg" if mpath.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                            parts.append(f"REFERENCE 16:9 CONCEPT ART MOCKUP ({mpath.name}):")
                            parts.append(types.Part.from_bytes(data=mpath.read_bytes(), mime_type=mime))
            except Exception:
                pass

    valid_proofs = [
        Path(p_str)
        for p_str in collected_proofs
        if Path(p_str).is_file() and Path(p_str).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    ]
    filmstrips = [p for p in valid_proofs if "filmstrip" in p.name.lower()]
    non_filmstrips = [p for p in valid_proofs if "filmstrip" not in p.name.lower()]
    inspection_plates = [
        p
        for p in non_filmstrips
        if p.name.lower().startswith(("plate_", "anim_sheet_", "anim_rig_", "waveform_", "script_sheet_", "level_map_"))
    ]
    deduped_proofs: List[Path] = []
    if inspection_plates:
        deduped_proofs.append(inspection_plates[-1])
    elif non_filmstrips:
        deduped_proofs.append(non_filmstrips[-1])
    if filmstrips:
        deduped_proofs.append(filmstrips[-1])
    if not deduped_proofs and valid_proofs:
        deduped_proofs = valid_proofs[-1:]

    for p in deduped_proofs:
        mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        n_low = p.name.lower()
        qa_note = ""
        if n_low.startswith("plate_"):
            qa_note = (
                " [QA ALPHA INSPECTION PLATE: The side-by-side dark/light square checkerboard grids and cyan bounding-box rectangle "
                "are external QA test-harness backgrounds drawn behind/around the transparent PNG—they are NOT inside the sprite PNG file "
                "and must NEVER be flagged as cross-hatch/grid/box artifacts!]"
            )
        elif n_low.startswith("anim_sheet_") or n_low.startswith("anim_rig_"):
            qa_note = (
                " [QA ANIMATION CONTACT SHEET: The dark square checkerboard grid, cyan cell borders, and horizontal yellow/gold baseline ruler lines "
                "at the bottom of each frame are external QA test-harness alignment guides—they are NOT inside the sprite sheet PNG file and must NEVER "
                "be flagged as yellow line or checkerboard artifacts!]"
            )
        elif "filmstrip" in n_low:
            qa_note = (
                " [QA CONTACT SHEET: The top ('[QA] tick N in:[...]') and bottom ('[QA] pos=...') strips outside each 16:9 frame are external QA "
                "telemetry margins, NOT inside the game viewport.]"
            )
        parts.append(
            f"CAPTURED PROOF IMAGE ({p.name}){qa_note} — inspect this image carefully before choosing submit_to_council:"
        )
        lead_media_res = getattr(getattr(types, "MediaResolution", None), "MEDIA_RESOLUTION_HIGH", None)
        parts.append(
            types.Part.from_bytes(data=p.read_bytes(), mime_type=mime, media_resolution=lead_media_res)
            if lead_media_res is not None
            else types.Part.from_bytes(data=p.read_bytes(), mime_type=mime)
        )
    return parts


def plan_task_lead_turn(
    ctx: WorkerContext,
    item: WorkItem,
    turn_history: List[SpecialistReport],
    collected_proofs: List[str],
    attempts: int = 3,
) -> TaskLeadDecision:
    """Runs one turn of the Task Lead Strategist: inspects the task, cartridge state, prior Council revision
    feedback, and specialist reports + multimodal proof images from earlier turns, then returns a validated `TaskLeadDecision`."""
    if not _is_live(ctx.client):
        raise StudioAgentError("Task Lead planning requires an active google.genai Client (Rule 5)")
    from google.genai import types

    specialists = available_specialists(ctx, item)
    qa_context = _summarize_cartridge_for_qa(ctx.cartridge_dir)
    cat_block = ""
    for cat_candidate in (ctx.cartridge_dir / "catalog.json", ctx.cartridge_dir / "assets" / "asset_library.json"):
        if cat_candidate.is_file():
            try:
                cat_block = f"\nON-DISK ASSET CATALOG ({cat_candidate.name}):\n{cat_candidate.read_text(encoding='utf-8')[:1500]}\n"
                break
            except Exception:
                pass

    slice_context_block = ""
    bible_path = Path(ctx.cartridge_dir) / "game_bible.json"
    if bible_path.is_file():
        try:
            bible_data = json.loads(bible_path.read_text(encoding="utf-8"))
            if isinstance(bible_data, dict):
                dna_summary = str(bible_data.get("target_player_dna_summary") or "").strip()
                core_loop_data = bible_data.get("core_loop") if isinstance(bible_data.get("core_loop"), dict) else {}
                agency_depth = str(core_loop_data.get("player_agency_and_optimization_depth") or "").strip()
                m2m_loop = str(core_loop_data.get("moment_to_moment_loop") or "").strip()
                min2min_loop = str(core_loop_data.get("minute_to_minute_loop") or "").strip()
                sess_loop = str(core_loop_data.get("session_loop") or "").strip()
                goal_data = bible_data.get("goal_hierarchy") if isinstance(bible_data.get("goal_hierarchy"), dict) else {}
                goals_list = goal_data.get("goals") if isinstance(goal_data.get("goals"), list) else []
                goals_summary = "; ".join(
                    f"[{g.get('horizon', '')}] {g.get('description', '')}"
                    for g in goals_list
                    if isinstance(g, dict) and g.get("description")
                )
                dna_and_agency_lines = ""
                if dna_summary:
                    dna_and_agency_lines += f"\nTARGET PLAYER STEAM DNA (YOU ARE BUILDING THIS GAME SPECIFICALLY FOR THIS PLAYER):\n{dna_summary}\n"
                if m2m_loop or min2min_loop or sess_loop or agency_depth:
                    dna_and_agency_lines += (
                        f"\nWHOLE-GAME CORE LOOP & AGENCY DEPTH (DO NOT SHRINK INTO A 1-STEP TUTORIAL):\n"
                        f"  - Moment-to-Moment Loop: {m2m_loop}\n"
                        f"  - Minute-to-Minute Loop: {min2min_loop}\n"
                        f"  - Session Loop: {sess_loop}\n"
                        f"  - Player Agency & Optimization Depth: {agency_depth}\n"
                    )
                if goals_summary:
                    dna_and_agency_lines += f"  - Escalating Goal Hierarchy: {goals_summary}\n"
                roadmap = bible_data.get("slice_roadmap")
                active_idx = int(bible_data.get("active_slice_index") or 0)
                if isinstance(roadmap, list) and roadmap:
                    active_slice = None
                    for s in roadmap:
                        if isinstance(s, dict) and s.get("status") == "active":
                            active_slice = s
                            break
                    if active_slice is None and 0 <= active_idx < len(roadmap):
                        active_slice = roadmap[active_idx]
                    if isinstance(active_slice, dict):
                        slice_context_block = (
                            f"{dna_and_agency_lines}"
                            f"\nACTIVE GAME SLICE [{active_slice.get('slice_id', '')}] (Iteration #{active_slice.get('slice_index', 0)} — {active_slice.get('title', '')}):\n"
                            f"  Objective: {active_slice.get('slice_objective', '')}\n"
                            f"  Player Acclimation & Full-Loop Focus: {active_slice.get('player_acclimation_and_mechanics_focus', '')}\n"
                            f"  Expansion from Prior Slice: {active_slice.get('expansion_from_prior_slice', '')}\n"
                            f"  Required Systems & Scope: {active_slice.get('scope_boundaries', '')}\n"
                        )
                elif dna_and_agency_lines:
                    slice_context_block = dna_and_agency_lines
        except Exception:
            pass

    board_snapshot_block = ""
    if ctx.board is not None and hasattr(ctx.board, "items"):
        try:
            b_lines = [
                f"  - id={b_it.item_id} ({b_it.kind}, state={b_it.state}, slice={b_it.slice_id or 'default'}): {b_it.title}"
                for b_it in ctx.board.items()
            ]
            if b_lines:
                board_snapshot_block = (
                    "\nCURRENT PRODUCTION BOARD ITEMS (check before spawning so you never duplicate existing tasks):\n"
                    + "\n".join(b_lines)
                    + "\n"
                )
        except Exception:
            pass

    prior_revision_block = ""
    if item.revisions:
        rev_lines = [
            f"  rev {r.revision_index + 1}: verdict={r.verdict} | worker_notes={r.worker_notes[:400]!r} | council_feedback={r.feedback!r}"
            for r in item.revisions
        ]
        prior_trace_path = _item_dir(ctx, item) / f"lead_trace_rev{len(item.revisions) - 1}.json"
        prior_trace_text = ""
        if prior_trace_path.is_file():
            try:
                prior_trace_text = f"\nPREVIOUS REVISION STRATEGY & SPECIALIST TRACE:\n{prior_trace_path.read_text(encoding='utf-8')[:2000]}\n"
            except Exception:
                pass
        prior_revision_block = (
            "\nCOUNCIL REVISION FEEDBACK HISTORY (you MUST re-strategize and pivot across specialists as needed to address ALL of it):\n"
            + "\n".join(rev_lines)
            + "\n"
            + prior_trace_text
        )
    if getattr(item, "failure", ""):
        err_excerpt = _format_worker_failure_excerpt(item.failure, 1200)
        if turn_history and any(not str(t.summary).startswith("FAILED") for t in turn_history):
            prior_revision_block += (
                "\nPRIOR ATTEMPT ERROR (occurred BEFORE Turn 0 of this attempt; if a specialist in CURRENT REVISION TURN HISTORY "
                "already updated the code to address this, proceed to the next verification/proof specialist such as qa_test_engineer):\n"
                f"{err_excerpt}\n"
            )
        else:
            prior_revision_block += f"\nPREVIOUS WORKER EXECUTION ERROR TO FIX:\n{err_excerpt}\n"

    verified_modified_prod: List[str] = []
    for t_rep in turn_history:
        if getattr(t_rep, "specialist_agent", "") == "qa_test_engineer":
            continue
        for tf in (getattr(t_rep, "touched_files", None) or []):
            norm_tf = CartridgeFileLockTable.normalize_path(str(tf))
            if norm_tf and not norm_tf.startswith("tests/") and norm_tf not in verified_modified_prod:
                verified_modified_prod.append(norm_tf)
    untouched_target_prod: List[str] = []
    for tm in (getattr(item, "target_modules", None) or []):
        norm_tm = CartridgeFileLockTable.normalize_path(str(tm))
        if (
            norm_tm
            and not norm_tm.startswith("tests/")
            and norm_tm not in verified_modified_prod
            and norm_tm not in untouched_target_prod
        ):
            untouched_target_prod.append(norm_tm)
    verified_diff_block = (
        f"VERIFIED PRODUCTION DISK DIFF SO FAR: modified={verified_modified_prod}, "
        f"untouched_target_modules={untouched_target_prod}\n"
    )
    turns_block = ""
    if turn_history:
        t_lines = [
            f"  turn {t.turn_index}: delegated to '{t.specialist_agent}' | brief={t.brief[:250]!r}\n"
            f"    -> summary: {t.summary}\n"
            f"    -> touched_files: {t.touched_files} | proof_images: {t.proof_images}"
            for t in turn_history
        ]
        turns_block = "\nCURRENT REVISION TURN HISTORY (specialist work completed so far):\n" + "\n".join(t_lines) + "\n"
    else:
        turns_block = "\nCURRENT REVISION TURN HISTORY: (Turn 0 — no specialists invoked yet in this revision)\n"

    spec_descriptions = {
        "visual_artist": (
            "  - 'visual_artist': fabricates any visual asset (static sprite, sprite sheet, articulated parts, tiles, parallax, portrait, UI) "
            "into the cartridge and produces high-resolution asset inspection/contact plates (`plate_*.png`, `anim_sheet_*.png`).\n"
        ),
        "audio_designer": (
            "  - 'audio_designer': synthesizes voice barks, SFX, or ambient loops and renders a waveform proof image (`waveform_*.png`).\n"
        ),
        "script_writer": (
            "  - 'script_writer': Lead Scriptwriter & Narrative Dramatist — crafts scene scripts (`cutscenes`), branching `dialogue_trees`, "
            "contextual `dialogue_barks`, and character sheets into `narrative_bible.json` / `narrative.json` and produces a script table proof sheet (`script_sheet_*.png`).\n"
        ),
        "swe_engineer": (
            "  - 'swe_engineer': architects or modifies PRODUCTION GDScript modules (`.gd` outside `tests/`—NEVER `tests/test_*.gd`), world/level layouts, scenes, entities, or UI so `res://main.tscn` works natively under real player inputs/signals, and verifies headless engine compilation/boot.\n"
        ),
        "qa_test_engineer": (
            "  - 'qa_test_engineer': EXCLUSIVELY owns and builds the Godot QA test scene (`res://tests/test_<item_id>.tscn` + `.gd`), strategizes read-only QA diagnostic overlays or telemetry (`overlays_and_telemetry_strategy`), executes external player inputs (`input_segments`, `mouse_world_pos`, `mouse_click_pos`, `state_assertions`, `capture_ticks`) verified against BOTH `res://tests/test_<item_id>.tscn` and live `res://main.tscn` (Counterfactual + Entrypoint Parity Gates), captures the filmstrip proof image, and authors `council_test_scene_summary`.\n"
        ),
    }
    from engine.studio.context_curator import ContextCuratorAgent, SelfInquiryQAPair, is_real_genai_client

    available_spec_lines = "".join(spec_descriptions[s] for s in specialists if s in spec_descriptions)
    curated_lead_block = ""
    multimodal_parts: List[Any] = []
    prior_rev_proofs: List[str] = []
    if item.revisions:
        for p in (item.revisions[-1].proof_paths or []):
            if p and Path(p).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and p not in prior_rev_proofs:
                prior_rev_proofs.append(p)
    turn_proof_imgs: List[str] = []
    for t_rep in turn_history:
        for p_img in (getattr(t_rep, "proof_images", None) or []):
            if p_img and p_img not in turn_proof_imgs:
                turn_proof_imgs.append(p_img)
    current_turn_proofs = [
        p
        for p in dict.fromkeys(list(collected_proofs) + turn_proof_imgs)
        if p and Path(p).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    ]
    raw_lead_images = current_turn_proofs[-1:] if current_turn_proofs else prior_rev_proofs[-1:]
    lead_image_paths = list(raw_lead_images)
    active_curator: Optional[ContextCuratorAgent] = None
    already_included_cids: List[str] = []
    if is_real_genai_client(ctx.client):
        curator = ContextCuratorAgent(
            cartridge_dir=ctx.cartridge_dir,
            client=ctx.client,
            model_name=ctx.model_name,
            board=ctx.board,
            event_callback=ctx.event_callback,
        )
        active_curator = curator
        pack = curator.curate_context(
            target_agent_role="task_lead",
            task_request=(
                f"{item.title}\nSPEC: {item.spec}\nPROOF: {item.proof_spec}\n"
                f"{prior_revision_block}"
            ),
            item_id=item.item_id,
            item_kind=item.kind,
            extra_image_paths=lead_image_paths,
        )
        already_included_cids = [ch.chunk_id for ch in (pack.retrieved_chunks or [])]
        curated_lead_block = pack.prompt_block
        multimodal_parts = list(pack.multimodal_parts) or _build_task_lead_multimodal_parts(
            ctx, lead_image_paths, item_kind=item.kind
        )
        if collected_proofs and pack.multimodal_parts:
            existing_labels = [p for p in multimodal_parts if isinstance(p, str)]
            extra_proof_parts = _build_task_lead_multimodal_parts(
                ctx, list(collected_proofs)[-1:], item_kind=item.kind
            )
            for idx_p in range(0, len(extra_proof_parts), 2):
                lbl = extra_proof_parts[idx_p]
                if (
                    isinstance(lbl, str)
                    and lbl.startswith("CAPTURED PROOF IMAGE")
                    and idx_p + 1 < len(extra_proof_parts)
                ):
                    m_fname = re.search(r"\(([^)—\]]+)", lbl)
                    fname = m_fname.group(1).strip() if m_fname else ""
                    already_attached = any((fname and fname in ex_lbl) or ex_lbl == lbl for ex_lbl in existing_labels)
                    if not already_attached:
                        multimodal_parts.append(lbl)
                        multimodal_parts.append(extra_proof_parts[idx_p + 1])
                        existing_labels.append(lbl)

    from engine.studio.game_memory import GameMemoryStore, format_memory_query_results

    mem_store = GameMemoryStore(ctx.cartridge_dir, event_callback=ctx.event_callback, client=ctx.client)
    has_embed_client = callable(getattr(getattr(ctx.client, "models", None), "embed_content", None))
    if active_curator is None and (ctx.cartridge_dir / "game_memory.json").is_file():
        lead_mem = mem_store.format_role_memory_briefing("task_lead")
        if lead_mem:
            curated_lead_block = f"{lead_mem}\n{curated_lead_block}" if curated_lead_block else lead_mem

    issues: List[str] = []
    resolved_inquiries: List[SelfInquiryQAPair] = (
        active_curator.load_self_inquiry_memory(item.item_id, "task_lead")
        if active_curator is not None
        else []
    )
    pending_question_pack = None
    asked_questions: set[str] = {p.question.strip().lower() for p in resolved_inquiries if p.question}
    inquiry_budget_used = 0
    max_inquiry_turns = 2 if active_curator is not None else 0
    mem_query_blocks: List[str] = []
    asked_mem_queries: set[str] = set()
    mem_query_budget_used = 0
    max_mem_query_turns = 2 if has_embed_client else 0

    attempt = 1
    while attempt <= attempts:
        retry = "\nYOUR PREVIOUS TaskLeadDecision FAILED VALIDATION; FIX:\n- " + "\n- ".join(issues) + "\n" if issues else ""
        inquiry_block = (
            active_curator.format_self_inquiry_prompt_section(resolved_inquiries, pending_question_pack)
            if active_curator is not None
            else ""
        )
        scope_notes_block = ""
        if getattr(item, "excluded_modules", None):
            scope_notes_block += (
                f"EXCLUDED OUT-OF-SCOPE MODULES (pruned by Council—do NOT target or modify these files): {item.excluded_modules}\n"
            )
        if getattr(item, "shaping_notes", None):
            council_bounds = [n for n in item.shaping_notes if "[Council Scope Boundary]" in str(n)]
            if council_bounds:
                scope_notes_block += "COUNCIL SCOPE BOUNDARY DIRECTIVES:\n  " + "\n  ".join(council_bounds) + "\n"
        prompt = (
            f"You are the Task Lead Strategist for TASK [{item.item_id}] ({item.kind}): {item.title}\n"
            f"SPEC: {item.spec}\n"
            f"PROOF THE COUNCIL WILL JUDGE: {item.proof_spec}\n"
            f"TARGET MODULES HINT: {item.target_modules}\n"
            f"{scope_notes_block}"
            f"AVAILABLE AUTONOMOUS SPECIALISTS: {specialists}\n"
            f"{available_spec_lines}"
            f"INSTRUCTIONS:\n"
            f"0. SKEPTICAL VERIFICATION, EXECUTIVE PRODUCER RESPECT & PRAGMATIC TASK COMPLETION:\n"
            f"   - Treat specialist turn summaries (`swe_engineer`, `visual_artist`, `qa_test_engineer`) as UNVERIFIED CLAIMS—never praise or trust a specialist's prose ('implemented cleanly!') until you verify it directly in the rendered proof pixels, telemetry progression, and `VERIFIED PRODUCTION DISK DIFF SO FAR`.\n"
            f"   - Treat any Executive Producer steering or Council direction in `SPEC` / `COUNCIL DIRECTION` with full respect and cater directly to it without excuses or silent scope downgrading (Rule 5).\n"
            f"   - Keep scope bounded to THIS task: deliver `SPEC` and `PROOF THE COUNCIL WILL JUDGE` cleanly and fix any glaring visual/UI/causal glitches in your modules, then call the task ready (`submit_to_council`) rather than trying to rewrite unrelated systems or boil the ocean in one go.\n"
            f"1. Formulate your bespoke `current_strategy` and `proof_strategy` for creating everything necessary for this task and generating the final proof (and pivoting if the Council requested changes):\n"
            f"   - Creation Strategy (`current_strategy`): Plan every artifact, asset, multi-cell sheet/grid, rig, script, or GDScript module required by `SPEC`, and decide which specialist(s) in {specialists} to delegate to in sequence.\n"
            f"   - Proof Generation Strategy (`proof_strategy`): Plan how the final visual proof artifact(s) (`selected_proof_paths`) will be generated and verified to directly satisfy `PROOF THE COUNCIL WILL JUDGE` (`{item.proof_spec}`).\n"
            f"   - When delegating to `'visual_artist'` for a task that requires a multi-cell grid, sheet, or atlas of distinct visual states, icons, stages, or components (e.g. a 2x2 or NxM state sheet where each cell depicts a distinct graphic/state), explicitly specify the `grid_cols x grid_rows` layout and each cell's visual contents in row-major order in your `brief` so `'visual_artist'` populates `grid_cols`, `grid_rows`, and `cell_prompts`.\n"
            f"   - HOLISTIC SYSTEMS THINKING, SUBSYSTEM HARMONY & HUMAN-SCALE RESPONSIVENESS:\n"
            f"     * Never treat a single node's coordinate (`pos=(x, y)`) as proof that the whole scene works visually! Always reason across the complete interacting system: all active entities/nodes in the scene, how `Camera2D` computes its framing/target, how UI layers compose on screen, and how sibling subsystems interact.\n"
            f"     * Check for subsystem conflicts and human-scale feel: when multiple components act on the same entity (e.g., player input controllers alongside damping/stabilization/constraint scripts), ensure they do not fight or cancel out active player commands, and verify that holding a player input over 1–2 seconds (60–120 ticks) produces a responsive, human-playable rate of motion/turn/action.\n"
            f"     * If a prior task on the board or an earlier turn/revision already modified a script/property and the visual issue still occurs in the rendered viewport, treat the prior hypothesis as FALSIFIED/INCOMPLETE and inspect all other interacting nodes, camera targets, or scripts in the scene!\n"
            f"   - SELF-INQUIRY TOOL (`self_question` -> ContextCuratorAgent research -> `self_question_answer`): If you need to inspect a specific module's code, signals, or design spec that is not in your current context before writing your specialist `brief`, you may output a specific question in `self_question` (leaving `brief` empty). `ContextCuratorAgent` will fetch the relevant chunks on the next turn so YOU can answer your own question in `self_question_answer` and then delegate accurately.\n"
            f"   - LONG-TERM MEMORY (`memory_query` / `memory_updates`): If you want to look up cross-slice strategic goals, prior architectural decisions, or operational lessons learned in the game or agent memory banks before delegating, populate `memory_query` (and optional `memory_query_bank`, leaving `brief` empty). Whenever you uncover a durable lesson learned or cross-task invariant during execution, record it in `memory_updates` (`target_bank='self'` or `'council'`).\n"
            f"   - ON-DEMAND RUNTIME DIAGNOSTIC TEST SCENES ('I NEED MORE DEBUG INFO'): If a prior proof, Council `changes_requested` feedback, or bug report shows a runtime issue whose root cause is NOT already obvious from the existing filmstrip and code, do NOT guess blindly! Delegate to `'qa_test_engineer'` FIRST with a diagnostic brief asking QA to build/update `res://tests/test_<item_id>.tscn` with specific visual debug overlays and `get_state_snapshot()` telemetry keys to isolate the exact cause, then inspect the resulting diagnostic filmstrip on the next turn before briefing `'swe_engineer'`.\n"
            f"2. JUST-IN-TIME ASSET & SUB-TASK DISCOVERY (`action = 'spawn_tasks'` or `'delegate'` to `'visual_artist'`): Assets and animations are NOT built heavily upfront. As you plan or implement this task, reason out what specific assets, surface textures, animations, sounds, or sub-tasks are needed for THIS task in THIS slice.\n"
            f"   - HYBRID PROCEDURAL + GENERATED VISUAL CRAFT (THE BEST OF BOTH WORLDS):\n"
            f"     * NEVER rely on flat untextured `Polygon2D` slabs, blank monochrome backgrounds, or bare 8-point `Line2D` programmer-art zigzags! Instead, combine procedural generation (Simplex noise, cellular automata, Voronoi, spline contours, custom `ShaderMaterial`, `GPUParticles2D`) with fabricated visual assets from `'visual_artist'`:\n"
            f"       - Fabricate seamless `tile_surface` textures in `assets/sprites/*.png` and bind them onto procedural `Polygon2D` terrain/strata via `StudioRigHelper.create_tiled_polygon(points, tex)` (`texture_repeat = CanvasItem.TEXTURE_REPEAT_ENABLED`).\n"
            f"       - Fabricate repeatable strip textures or shaders for `Line2D` conduits/cracks/vines (`texture_mode = Line2D.LINE_TEXTURE_TILE`) and scatter fabricated `static_sprite` environmental decals/landmarks along procedural contours.\n"
            f"     * Characters, creatures, avatars, and interactive world structures MUST be rendered with actual fabricated visual assets (`sprite_sheet`, `articulated_parts`, or `static_sprite` loaded from `res://assets/sprites/*.png`), NEVER left as primitive `draw_circle` / `draw_rect` placeholder stickmen or untextured boxes!\n"
            f"     * When Council feedback critiques crude procedural polygons/lines (`Polygon2D` / `Line2D` / `_draw()`) or missing textures, delegate to `'visual_artist'` FIRST to fabricate the needed `tile_surface` or sprite PNGs in `assets/sprites/*.png` and THEN delegate to `'swe_engineer'` to bind those textures onto the procedural geometry.\n"
            f"   - For `level_map` tasks: A 2D wireframe diagram (`level_map_*.png`) alone is NEVER sufficient! You MUST ensure `scenes.gd` realizes the multi-region `level_layout.json` in Godot with hybrid procedural + textured terrain, environmental props, and an active `Camera2D` tracking the player across `world_bounds` (with HUD isolated on a `CanvasLayer`), and then delegate to `'qa_test_engineer'` to capture a live in-engine Godot viewport filmstrip (`level_filmstrip_*.png` / `filmstrip_*.png`) before `submit_to_council`.\n"
            f"   - If this task implements or tests a character, creature, or interactive entity (`mechanic_code`, `integration`, `level_map`, etc.):\n"
            f"     * First check `ON-DISK ASSET CATALOG` and `CURRENT PRODUCTION BOARD ITEMS`. If a matching sprite, rig, or `tile_surface` texture already exists on disk, instruct `'swe_engineer'` in your `brief` to load and bind that exact PNG/rig (`Image.load_from_file` / `StudioRigHelper`)—preserving its exact `hframes`/`vframes` from `asset_library.json` (remember `articulated_parts` composites have `hframes=1, vframes=1`!).\n"
            f"     * If a needed character rig, structure sprite, or `tile_surface` texture does not exist on disk yet, either delegate to `'visual_artist'` first to fabricate it inside this task, OR choose `action = 'spawn_tasks'` with `wait_for_spawned_tasks = True` so the visual asset is fabricated before `'swe_engineer'` binds it.\n"
            f"3. EMPATHETIC NEW-PLAYER PERSPECTIVE, TARGET PLAYER DNA, WHOLE-GAME LOOP & UNSOLVED t=0 BASELINE (for `mechanic_code`, `ui`, `integration`, `level_map`):\n"
            f"   - DELIVER THE LARGER POINT OF THE GAME (NEVER SHRINK INTO A 1-STEP TUTORIAL): Check `TARGET PLAYER STEAM DNA` and `WHOLE-GAME CORE LOOP & AGENCY DEPTH` below. You are building this game specifically for THIS player. Do NOT build 'the smallest slice possible' that reduces the game to a 4-step linear tutorial or a single pre-placed connection! Onboarding is ONLY the opening 60 seconds; the game itself must deliver the full `moment_to_moment_loop`, `minute_to_minute_loop`, and `session_loop` with genuine player construction, experimentation, optimization, or tactical depth (`core_loop.player_agency_and_optimization_depth`) and escalating goals (`goal_hierarchy`).\n"
            f"   - Think empathetically from the perspective of a brand-new player booting this game for the first time with zero external instructions. Strategize how THIS game communicates: (a) what the player is doing and why (their role, purpose, and stakes), (b) how they interact with the world and how actions causally connect (`input -> entity response -> world consequence`), and (c) what visual/audio feedback occurs when an action succeeds OR when an action is attempted before its prerequisite is met.\n"
            f"   - UNSOLVED t=0 BASELINE (`test_unsolved_baseline_t0`) & QA TEST SCENE OWNERSHIP (`res://tests/test_<item_id>.tscn`) — NO DEBUG OVERLAYS IN PRODUCTION GAME (`res://main.tscn`):\n"
            f"     * `_ready()` at `t=0` in `main.gd` (`res://main.tscn`) is what the live human player boots into! `_ready()` MUST initialize the world in its unbuilt/unsolved starting state as defined by the Game Bible, and production game scenes (`main.gd`, `scenes.gd`, `ui_hud.gd`, etc.) must NEVER contain `func test_*()` staging helpers or debug overlays (`[DEBUG]`, telemetry readouts, wireframe boxes).\n"
            f"     * Conversely, in QA test scenes (`res://tests/test_<item_id>.tscn`), rich diagnostic overlays ARE essential—and `res://tests/test_<item_id>.tscn` + `.gd` is EXCLUSIVELY owned by `'qa_test_engineer'`, NEVER `'swe_engineer'`! NEVER ask `'swe_engineer'` to create or edit `tests/test_*.gd`, and NEVER script `_physics_process` tick timers or direct spawn/place calls inside `tests/test_*.gd` (even if an older feedback note mentioned `tests/test_*.gd`, translate missing gameplay/signal wires into production-module fixes for `'swe_engineer'` so live `res://main.tscn` passes the Entrypoint Parity Gate!). When delegating to `'qa_test_engineer'`, instruct QA to author `res://tests/test_<item_id>.tscn` (`test_scene_gd_content` with read-only camera framing/diagnostic `_draw()` overlays and `get_state_snapshot()` readouts—zero mid-run game-state mutations), drive all actions externally via `input_segments`, and write a clear `council_test_scene_summary` for the Council!\n"
            f"   - REAL INTERACTIVE VERBS & CAMERA NAVIGATION: Every core player verb in `core_loop.player_verbs` and the active slice's `scope_boundaries` (including mouse click/drag placement or connection, interactive component tuning/socketing, hazard clearing, and `Camera2D` panning/scrolling when `world_bounds` exceeds the 1280x720 viewport) MUST be wired to real `_unhandled_input` / `_input` / UI handlers and exercised via `input_segments` (`mouse_world_pos`, `mouse_click_pos`, or actions) in QA filmstrips.\n"
            f"   - DIAGNOSING FAILED QA PLAYTESTS vs. BROKEN MECHANICS (ZERO SHORTCUTS — NEVER PRE-SOLVE THE GAME TO PASS QA):\n"
            f"     * First check `CARTRIDGE SYMBOLS & FILES` below for any `⚠️ BROKEN / MISSING SCRIPT OR SCENE REFERENCES ON DISK`. If a builder/spawner script calls `load('res://...')` on a `.gd` file that does not exist on disk (or wraps it in `ResourceLoader.exists(...) else null`), placement/spawning silently no-ops! Delegate to `'swe_engineer'` to create the missing script or replace the broken reference with real node instantiation.\n"
            f"     * REJECT GOALPOST-SHIFT & HITBOX-INFLATION SHORTCUTS: When a multi-step gameplay loop fails QA state assertions, NEVER work around the failure by moving/pre-spawning a target/receiver into the default un-deflected trajectory in `main.gd._ready()` or by inflating catch/collision radii beyond visual sprite bounds (24–48px)—even if a prior revision note or feedback comment suggested moving the goal! The engine's Counterfactual QA Check runs with `input_script = []` (zero player input) and automatically rejects any build where post-baseline progress assertions pass without player construction.\n"
            f"     * DRIVE THE COMPLETE CAUSAL CHAIN: Instead of taking shortcuts: (a) instruct `'swe_engineer'` to expose step-by-step causal telemetry in `get_state_snapshot()` (selected tool slot, placement validity, placed structure counts, intermediate interaction/ricochet counts, and delivery/objective counts) and ensure placed structures genuinely route/process items with honest sprite-sized hitboxes, and (b) instruct `'qa_test_engineer'` in your `brief` with the exact multi-step causal input recipe (which `KEY_*` selects the required tool slot, which exact valid 2D world coordinates `mouse_world_pos = [wx, wy]` lie on buildable terrain along the path, what rotation/tuning keys are needed, and step-by-step `state_assertions` for every link in the chain).\n"
            f"   - Every input key/action bound in the codebase (`Available actions`) MUST have a discoverable in-world or HUD affordance—never bind hidden/secret keys that the player cannot know exist, and NEVER bind fake tutorial-skip keys (such as `KEY_TAB` or `set_onboarding_step`) that advance onboarding text without the player performing the actual in-world mechanic.\n"
            f"   - Never rig an input handler to bypass physics, force a win state, or teleport world state when a timing window is missed; instead, instruct `'qa_test_engineer'` to use reactive state triggers (`when_snapshot_key`, `when_operator`, `when_value`) to test timing-sensitive mechanics honestly.\n"
            f"   - Prevent duplicate rendering across modules: ensure a single authoritative module owns each world structure or visual effect so `main.gd`, `scenes.gd`, `entities.gd`, and `ui_hud.gd` do not draw overlapping copies of the same geometry or indicator.\n"
            f"4. If `COLLECTED PROOF IMAGES SO FAR` is non-empty, visually inspect the attached proof image(s) (and any `QA Post-Capture Visual Audit` findings in the turn history) before choosing `submit_to_council`:\n"
            f"   - In `filmstrip_*.png` contact sheets, the top (`[QA] tick N in:[...]`) and bottom (`[QA] pos=(...) ...`) bands outside each 16:9 frame border are external QA test-harness annotations—use them to verify coordinates/state, and never mistake them for in-game debug text.\n"
            f"   - In `plate_*.png` and `anim_sheet_*.png` / `anim_rig_*.png`, the dark/light square checkerboard grids, cyan bounding-box borders, and horizontal yellow/gold baseline ruler lines at the bottom of frames are external QA test-harness overlays drawn behind/around the transparent PNG—NEVER mistake those external QA overlays for cross-hatch grids or yellow line artifacts inside the sprite PNG!\n"
            f"   - NEVER SUBMIT PLACEHOLDER STICKMEN, CLIPPED SPRITES, DUPLICATE OVERLAPPING GEOMETRY, OR SHRUNK VIEWPORT BACKGROUNDS: If the proof image or `QA Post-Capture Visual Audit` reports `qa_passed=False`, unaddressed Council defects, clipped/invisible sprites (`hframes > 1` on a single-frame texture), or a shrunk background that leaves dark void around the level, do NOT choose `submit_to_council`! Fix the root cause with `'visual_artist'` / `'swe_engineer'` and re-run `'qa_test_engineer'`.\n"
            f"   - Verify that active gameplay entities and world geometry remain visibly framed inside the 16:9 viewport across ALL filmstrip frames and do NOT disappear off-screen or get hidden behind an opaque full-screen background/border frame drawn at a higher `z_index` or foreground `CanvasLayer`.\n"
            f"   - GROUND-TRUTH DISK DIFF AUDIT (`VERIFIED PRODUCTION DISK DIFF SO FAR`): Before choosing `submit_to_council`, check `untouched_target_modules` below! Never assume or claim in `submission_summary` that an untouched module (such as an input handler, HUD, or `main.gd`) was wired if `swe_engineer` only created an isolated entity script and left the consumer module untouched on disk. If an integration/consumer target module still needs wiring so the live player can actually trigger the feature in `res://main.tscn`, delegate to `'swe_engineer'` first!\n"
            f"5. Choose `action = 'delegate'` to send an open-ended natural-language `brief` to one of {specialists}, `action = 'spawn_tasks'` to create newly discovered tasks on the board, OR `action = 'submit_to_council'` ONLY after all required assets/code are built AND at least one proof image in `COLLECTED PROOF IMAGES SO FAR` visually proves the requirement.\n"
            f"   - CONCISE DELIBERATION & BRIEF (NO RAW CODE IN BRIEF): Keep `deliberation` (<= 120 words) and `brief` (<= 180 words) focused and concise. NEVER write raw GDScript code blocks or paste whole functions inside `brief`—`swe_engineer` already reads the staging files directly! State the target file paths, class/function names, and behavioral/visual acceptance criteria.\n"
            f"   - CRITICAL SCHEMA FIELD ORDER WARNING: In `TaskLeadDecision`, `specialist_agent` and `brief` appear BEFORE `action` in the JSON schema! Decide your turn action inside `deliberation` and `current_strategy` first: whenever you are delegating to a specialist (`action = 'delegate'`), you MUST populate `specialist_agent` (one of {specialists}) and a non-empty `brief` BEFORE emitting `action = 'delegate'`! Never leave `specialist_agent` or `brief` empty when delegating.\n"
            f"   - LONG-TERM MEMORY SELECTIVITY: Leave `memory_updates` empty (`[]`) on ~95% of turns! Never log task progress, delegation steps, or facts already in the Game Bible/codebase; record ONLY when a costly cross-task failure mode, operational invariant, or deferred multi-slice goal would otherwise be lost.\n"
            f"Provide `deliberation` FIRST and `action` LAST.\n"
            f"{slice_context_block}"
            f"{curated_lead_block}"
            f"{inquiry_block}"
            f"{prior_revision_block}"
            f"{board_snapshot_block}"
            f"CARTRIDGE SYMBOLS & FILES:\n{qa_context or '(empty)'}{cat_block}"
            f"{verified_diff_block}"
            + (("\n" + "\n".join(mem_query_blocks)) if mem_query_blocks else "")
            + f"COLLECTED PROOF IMAGES SO FAR: {collected_proofs}\n"
            f"{turns_block}"
            f"{retry}"
        )
        contents_arg: Any = ([prompt] + multimodal_parts) if multimodal_parts else prompt
        record_turn_epistemic_context(
            role="task_lead",
            curated_context=f"{slice_context_block}\n{curated_lead_block}\n{prior_revision_block}".strip(),
            prompt_text=prompt,
            self_inquiries=[p.model_dump() for p in resolved_inquiries],
        )
        record_turn_substep(
            "planning_turn",
            f"Task Lead Strategist planning Turn #{len(turn_history) + 1} for '{item.item_id}'"
            + (f" (self-inquiry pass #{inquiry_budget_used})" if inquiry_budget_used > 0 else ""),
            specialist_agent="task_lead",
        )
        resp = call_gemini_with_transient_retry(
            lambda: ctx.client.models.generate_content(
                model=ctx.model_name,
                contents=contents_arg,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TaskLeadDecision,
                    thinking_config=bounded_thinking_config(768),
                ),
            ),
            label=f"plan_task_lead_turn[{item.item_id}:t{len(turn_history)}]",
        )
        if not resp.text:
            raise StudioAgentError(f"Task Lead planning for '{item.item_id}' returned an empty response")
        if getattr(type(ctx.client), "__module__", "") == "google.genai.fake":
            try:
                raw_dict = json.loads(resp.text)
            except Exception:
                raw_dict = {}
            if isinstance(raw_dict, dict) and "action" not in raw_dict and "current_strategy" not in raw_dict:
                if "fabrication_mode" in raw_dict or "capture_ticks" in raw_dict:
                    raise _LegacyFakeClientDirectPayload(resp.text)
        decision = TaskLeadDecision.model_validate_json(resp.text)
        record_turn_epistemic_context(
            role="task_lead",
            deliberation=decision.deliberation,
        )
        if decision.memory_updates:
            mem_store.apply_mutations(
                author_role="task_lead",
                proposals=decision.memory_updates,
                item_id=item.item_id,
            )

        # 1. If the Task Lead answered a pending self_question on this turn, record Q->A in persistent memory
        if pending_question_pack is not None and active_curator is not None:
            ans_text = (decision.self_question_answer or "").strip() or (decision.deliberation or "").strip()
            if ans_text:
                resolved_inquiries.append(
                    SelfInquiryQAPair(
                        deliberation=f"Task Lead answered its own self-question on '{item.item_id}'.",
                        question=pending_question_pack.question,
                        answer=ans_text,
                        researched_chunk_ids=list(pending_question_pack.selected_chunk_ids),
                    )
                )
                active_curator.save_self_inquiry_memory(item.item_id, "task_lead", resolved_inquiries)
                record_turn_substep(
                    "self_inquiry_answered",
                    f"Task Lead answered internal question: {ans_text[:140]}",
                    details={"question": pending_question_pack.question, "answer": ans_text},
                    specialist_agent="task_lead",
                )
                record_turn_epistemic_context(
                    role="task_lead",
                    self_inquiries=[p.model_dump() for p in resolved_inquiries],
                )
            pending_question_pack = None

        # 2. If the Task Lead asked a self_question instead of delegating/submitting, research it and loop without consuming a validation attempt
        new_q = (decision.self_question or "").strip()
        if (
            new_q
            and not decision.brief.strip()
            and active_curator is not None
            and inquiry_budget_used < max_inquiry_turns
            and new_q.lower() not in asked_questions
        ):
            asked_questions.add(new_q.lower())
            inquiry_budget_used += 1
            logger.info("Task Lead self-inquiry on '%s' (turn %d): %s", item.item_id, inquiry_budget_used, new_q)
            record_turn_substep(
                "self_inquiry",
                f"Task Lead asking internal question: {new_q}",
                details={"question": new_q},
                specialist_agent="task_lead",
            )
            pending_question_pack = active_curator.research_question_context(
                target_agent_role="task_lead",
                question=new_q,
                item_id=item.item_id,
                target_modules=item.target_modules,
                already_included_chunk_ids=already_included_cids,
            )
            for cid in pending_question_pack.selected_chunk_ids:
                if cid not in already_included_cids:
                    already_included_cids.append(cid)
            issues = []
            continue

        # 2b. If the Task Lead issued a direct `memory_query` instead of delegating/submitting, query GameMemoryStore
        new_mem_q = (decision.memory_query or "").strip()
        mem_q_key = f"{(decision.memory_query_bank or '').strip().lower()}::{new_mem_q.lower()}"
        if (
            new_mem_q
            and not decision.brief.strip()
            and has_embed_client
            and mem_query_budget_used < max_mem_query_turns
            and mem_q_key not in asked_mem_queries
        ):
            asked_mem_queries.add(mem_q_key)
            mem_query_budget_used += 1
            logger.info("Task Lead memory query on '%s' (turn %d): %s", item.item_id, mem_query_budget_used, new_mem_q)
            record_turn_substep(
                "memory_query",
                f"Task Lead querying long-term memory: {new_mem_q}",
                details={"query": new_mem_q, "bank": decision.memory_query_bank or "all"},
                specialist_agent="task_lead",
            )
            hits = mem_store.query_memories(
                caller_role="task_lead",
                query_text=new_mem_q,
                bank_filter=decision.memory_query_bank,
                client=ctx.client,
            )
            mem_query_blocks.append(format_memory_query_results(new_mem_q, decision.memory_query_bank, hits))
            issues = []
            continue

        issues = validate_task_lead_decision(
            decision,
            specialists,
            collected_proofs,
            board=ctx.board,
            parent_item_id=item.item_id,
            item_kind=item.kind,
            turn_history=turn_history,
        )
        if not issues:
            return decision
        logger.info("TaskLeadDecision attempt %d/%d for %s invalid: %s", attempt, attempts, item.item_id, issues)
        attempt += 1
    raise StudioAgentError(f"TaskLeadDecision for '{item.item_id}' still invalid after {attempts} attempts: {issues}")


def _dispatch_to_specialist(
    ctx: WorkerContext,
    item: WorkItem,
    decision: TaskLeadDecision,
    turn_index: int,
    _lock_already_held: bool = False,
) -> SpecialistReport:
    spec_name = decision.specialist_agent
    if spec_name == "visual_artist":
        return run_visual_artist_specialist(ctx, item, decision.brief, decision.target_paths, turn_index=turn_index)
    if spec_name == "audio_designer":
        return run_audio_designer_specialist(ctx, item, decision.brief, decision.target_paths, turn_index=turn_index)
    if spec_name == "script_writer":
        return run_script_writer_specialist(ctx, item, decision.brief, decision.target_paths, turn_index=turn_index)
    if spec_name == "swe_engineer":
        return run_swe_engineer_specialist(
            ctx,
            item,
            decision.brief,
            decision.target_paths,
            turn_index=turn_index,
            _lock_already_held=_lock_already_held,
        )
    if spec_name == "qa_test_engineer":
        return run_qa_test_engineer_specialist(
            ctx, item, decision.brief, decision.target_paths, turn_index=turn_index, _lock_already_held=_lock_already_held
        )
    raise StudioAgentError(f"Unknown specialist_agent '{spec_name}' requested for item '{item.item_id}'.")


def _fallback_for_legacy_fake_client(ctx: WorkerContext, item: WorkItem) -> WorkerResult:
    """Used only when a unit test's `google.genai.fake` responder returns AssetRequirement/FilmstripPlan
    directly instead of TaskLeadDecision."""
    if item.kind == "visual_asset":
        return visual_asset_handler(ctx)(item)
    if item.kind == "animation":
        return animation_handler(ctx)(item)
    if item.kind == "audio_asset":
        return audio_asset_handler(ctx)(item)
    if item.kind == "level_map":
        return level_map_handler(ctx)(item)
    return code_item_handler(ctx)(item)


def run_task_lead(
    ctx: WorkerContext,
    item: WorkItem,
    max_turns: int = 6,
) -> WorkerResult:
    """Executes the iterative Task Lead Strategist loop (Observe -> Strategize -> Delegate -> Inspect -> Submit)
    for a ProductionBoard `WorkItem`.

    When a task invokes `swe_engineer` or `qa_test_engineer`, `ctx.code_lock` is held across the entire
    remaining Task Lead session so that `.gd` edits and subsequent Godot headless `qa_test_engineer`
    filmstrip captures execute atomically on `ctx.cartridge_dir` without another worker interleaving
    shared-scene modifications in between.
    """
    if (
        item.kind == "level_map"
        and getattr(type(ctx.client), "__module__", "") == "google.genai.fake"
        and not item.target_modules
    ):
        return level_map_handler(ctx)(item)

    turn_history: List[SpecialistReport] = []
    collected_proofs: List[str] = []
    last_strategy = ""
    last_proof_strategy = ""
    submission_summary = ""
    selected_proofs: List[str] = []
    pending_decision: Optional[TaskLeadDecision] = None
    trace_path = _item_dir(ctx, item) / f"lead_trace_rev{len(item.revisions)}.json"

    if trace_path.is_file():
        try:
            existing_trace = json.loads(trace_path.read_text(encoding="utf-8"))
            if isinstance(existing_trace, dict):
                existing_status = str(existing_trace.get("status", "in_progress"))
                if (
                    existing_status in ("in_progress", "yielded", "waiting_for_spawned_tasks")
                    and not existing_trace.get("submission_summary")
                ):
                    for raw_turn in existing_trace.get("turns", []):
                        turn_history.append(SpecialistReport.model_validate(raw_turn))
                    for p in existing_trace.get("collected_proofs", []):
                        if isinstance(p, str) and Path(p).is_file() and p not in collected_proofs:
                            collected_proofs.append(p)
                    last_strategy = str(existing_trace.get("current_strategy") or "")
                    last_proof_strategy = str(existing_trace.get("proof_strategy") or "")
                    raw_pending = existing_trace.get("pending_decision")
                    if isinstance(raw_pending, dict):
                        pending_decision = TaskLeadDecision.model_validate(raw_pending)
                elif existing_status == "failed" and existing_trace.get("turns"):
                    # Archive prior failed retry trace so auto-retries never overwrite earlier turn chronology
                    fail_idx = 1
                    while (trace_path.parent / f"lead_trace_rev{len(item.revisions)}_failed{fail_idx}.json").is_file():
                        fail_idx += 1
                    archived_trace_path = trace_path.parent / f"lead_trace_rev{len(item.revisions)}_failed{fail_idx}.json"
                    archived_payload = dict(existing_trace)
                    archived_payload["archived_failed_index"] = fail_idx
                    archived_payload["archived_at_utc"] = datetime.now(timezone.utc).isoformat()
                    archived_trace_path.write_text(json.dumps(archived_payload, indent=2), encoding="utf-8")
                    trace_path.unlink(missing_ok=True)
        except Exception:
            logger.debug("Could not hydrate existing lead trace at %s", trace_path, exc_info=True)

    def _persist_trace(
        status: str = "in_progress",
        pending: Optional[TaskLeadDecision] = None,
        include_active_turn: bool = True,
    ) -> None:
        trace_data: Dict[str, Any] = {
            "item_id": item.item_id,
            "revision": len(item.revisions),
            "status": status,
            "current_strategy": last_strategy,
            "proof_strategy": last_proof_strategy,
            "submission_summary": submission_summary,
            "collected_proofs": list(collected_proofs),
            "turns": [t.model_dump() for t in turn_history],
        }
        if pending is not None:
            trace_data["pending_decision"] = pending.model_dump()
        live_turn_state = getattr(_ACTIVE_TURN_CTX, "state", None)
        if include_active_turn and status == "in_progress" and isinstance(live_turn_state, dict):
            trace_data["active_turn"] = {
                k: v for k, v in live_turn_state.items() if not str(k).startswith("_")
            }
        trace_path.write_text(json.dumps(trace_data, indent=2), encoding="utf-8")

    use_per_file_locks = ctx.file_lock_table is not None
    code_lock = ctx.code_lock if ctx.code_lock is not None else threading.Lock()
    code_lock_held = False
    task_baseline_snap: Optional[Dict[str, Optional[str]]] = None
    last_verified_swe_snap: Optional[Dict[str, Optional[str]]] = None
    spawn_turn_bonus = sum(1 for t in turn_history if t.specialist_agent == "board_task_spawner")
    effective_max_turns = max_turns + spawn_turn_bonus
    def _sync_live_board_state_or_yield(pending_dec: Optional[TaskLeadDecision] = None) -> None:
        if ctx.board is None or not hasattr(ctx.board, "get"):
            return
        try:
            live_item = ctx.board.get(item.item_id)
        except Exception:
            return
        item.target_modules = list(live_item.target_modules or [])
        item.excluded_modules = list(getattr(live_item, "excluded_modules", None) or [])
        item.shaping_notes = list(live_item.shaping_notes or [])
        item.depends_on = list(live_item.depends_on or [])
        deps_ok = True
        if hasattr(ctx.board, "_deps_satisfied"):
            try:
                deps_ok = bool(ctx.board._deps_satisfied(live_item))
            except Exception:
                deps_ok = True
        if live_item.state in ("blocked", "cancelled") or not deps_ok:
            _ACTIVE_TURN_CTX.state = None
            _persist_trace(
                status="waiting_for_spawned_tasks" if not deps_ok else "yielded",
                pending=pending_dec,
                include_active_turn=False,
            )
            raise WorkerYielded("spawned_prerequisite_tasks")

    try:
        turn_idx = len(turn_history)
        while turn_idx < effective_max_turns:
            _sync_live_board_state_or_yield(pending_decision)
            if is_in_board_scheduler() and ctx.cartridge_dir is not None:
                pause_file = ctx.cartridge_dir.parent / ".studio_paused.json"
                if pause_file.is_file():
                    try:
                        p_data = json.loads(pause_file.read_text(encoding="utf-8"))
                        if isinstance(p_data, dict) and p_data.get("paused") is True:
                            _ACTIVE_TURN_CTX.state = None
                            _persist_trace(status="yielded", pending=pending_decision, include_active_turn=False)
                            raise WorkerYielded(f"Build paused by operator before Turn #{turn_idx + 1} on '{item.item_id}'")
                    except WorkerYielded:
                        raise
                    except Exception:
                        pass
            _ACTIVE_TURN_CTX.state = {
                "item_id": item.item_id,
                "revision": len(item.revisions),
                "turn_index": turn_idx,
                "phase": "lead_planning",
                "specialist_agent": "task_lead",
                "brief": "",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "lead_deliberation": "",
                "lead_curated_context": "",
                "lead_prompt_text": "",
                "lead_self_inquiries": [],
                "curated_context": "",
                "prompt_text": "",
                "self_inquiries": [],
                "step_log": [],
                "_persist_fn": lambda: _persist_trace(status="in_progress", pending=None, include_active_turn=True),
                "_event_callback": ctx.event_callback,
            }
            _persist_trace(status="in_progress", pending=pending_decision, include_active_turn=True)
            resumed_from_pending = False
            if pending_decision is not None:
                decision = pending_decision
                pending_decision = None
                resumed_from_pending = True
            else:
                if ctx.event_callback is not None:
                    try:
                        ctx.event_callback(
                            "backlog_item_lead_planning",
                            {
                                "deliberation": f"Task Lead Strategist is planning Turn #{turn_idx + 1} for '{item.item_id}' ({item.kind}).",
                                "item_id": item.item_id,
                                "kind": item.kind,
                                "title": item.title,
                                "revision": len(item.revisions),
                                "turn_index": turn_idx,
                            },
                        )
                    except Exception:
                        pass
                try:
                    decision = plan_task_lead_turn(ctx, item, turn_history, collected_proofs)
                except _LegacyFakeClientDirectPayload:
                    _ACTIVE_TURN_CTX.state = None
                    return _fallback_for_legacy_fake_client(ctx, item)

            last_strategy = decision.current_strategy
            if getattr(decision, "proof_strategy", ""):
                last_proof_strategy = decision.proof_strategy
            if isinstance(getattr(_ACTIVE_TURN_CTX, "state", None), dict):
                _ACTIVE_TURN_CTX.state["phase"] = "specialist_executing" if decision.action == "delegate" else decision.action
                _ACTIVE_TURN_CTX.state["specialist_agent"] = decision.specialist_agent or decision.action
                _ACTIVE_TURN_CTX.state["brief"] = decision.brief or ""
                _ACTIVE_TURN_CTX.state["lead_deliberation"] = decision.deliberation or ""
                _persist_trace(status="in_progress", pending=decision if decision.action == "delegate" else None, include_active_turn=True)
            if not resumed_from_pending and ctx.event_callback is not None:
                try:
                    ctx.event_callback(
                        "backlog_item_lead_turn",
                        {
                            "deliberation": decision.deliberation,
                            "item_id": item.item_id,
                            "revision": len(item.revisions),
                            "turn_index": turn_idx,
                            "current_strategy": decision.current_strategy,
                            "action": decision.action,
                            "specialist_agent": decision.specialist_agent,
                            "brief": decision.brief,
                            "spawned_task_ids": [p.item_id for p in decision.spawned_tasks],
                        },
                    )
                except Exception:
                    pass

            if decision.action == "submit_to_council":
                submission_summary = decision.submission_summary or decision.current_strategy
                chosen = [p for p in decision.selected_proof_paths if Path(p).is_file()]
                if chosen and not any(Path(p).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") for p in chosen):
                    img_proofs = [
                        p
                        for p in collected_proofs
                        if Path(p).is_file() and Path(p).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                    ]
                    chosen = img_proofs + [p for p in chosen if p not in img_proofs]
                selected_proofs = chosen if chosen else list(collected_proofs)
                _ACTIVE_TURN_CTX.state = None
                _persist_trace(status="submitted", pending=None, include_active_turn=False)
                break

            if decision.action == "spawn_tasks":
                if ctx.board is None or not hasattr(ctx.board, "spawn_tasks_from_item"):
                    raise StudioAgentError(
                        f"Task '{item.item_id}' chose action='spawn_tasks', but WorkerContext has no active ProductionBoard attached."
                    )
                spawned_items = ctx.board.spawn_tasks_from_item(
                    parent_item_id=item.item_id,
                    proposals=decision.spawned_tasks,
                    wait_for_spawned=decision.wait_for_spawned_tasks,
                )
                spawned_ids = [s.item_id for s in spawned_items]
                live_st = getattr(_ACTIVE_TURN_CTX, "state", None) or {}
                spawn_report = SpecialistReport(
                    turn_index=turn_idx,
                    specialist_agent="board_task_spawner",
                    brief=decision.brief or f"Spawned {len(spawned_ids)} tasks on ProductionBoard: {spawned_ids}",
                    summary=(
                        f"Dynamically spawned {len(spawned_ids)} task(s) on ProductionBoard ({spawned_ids}); "
                        f"wait_for_spawned_tasks={decision.wait_for_spawned_tasks}."
                    ),
                    touched_files=[],
                    proof_images=[],
                    verification_passed=True,
                    step_log=list(live_st.get("step_log") or []),
                    lead_deliberation=str(live_st.get("lead_deliberation") or decision.deliberation or ""),
                    lead_prompt_text=str(live_st.get("lead_prompt_text") or ""),
                    lead_curated_context=str(live_st.get("lead_curated_context") or ""),
                    lead_self_inquiries=list(live_st.get("lead_self_inquiries") or []),
                )
                turn_history.append(spawn_report)
                effective_max_turns += 1
                _ACTIVE_TURN_CTX.state = None
                if decision.wait_for_spawned_tasks:
                    _persist_trace(status="waiting_for_spawned_tasks", pending=None, include_active_turn=False)
                    raise WorkerYielded(
                        f"spawned_prerequisite_tasks:{','.join(spawned_ids)}"
                    )
                _persist_trace(status="in_progress", pending=None, include_active_turn=False)
                turn_idx += 1
                continue

            if decision.specialist_agent in ("swe_engineer", "qa_test_engineer"):
                _sync_live_board_state_or_yield(decision)
                excluded_live = set(getattr(item, "excluded_modules", None) or [])
                if excluded_live and decision.target_paths:
                    decision.target_paths = [
                        tp
                        for tp in decision.target_paths
                        if CartridgeFileLockTable.normalize_path(tp) not in excluded_live
                    ]
                if use_per_file_locks and ctx.file_lock_table is not None:
                    if decision.specialist_agent == "swe_engineer":
                        base_req_list = (
                            list(decision.target_paths)
                            if decision.target_paths
                            else list(item.target_modules or [])
                        )
                        stash_cand = _item_dir(ctx, item) / f"stashed_swe_payload_rev{len(item.revisions)}.json"
                        if stash_cand.is_file():
                            try:
                                st_data = json.loads(stash_cand.read_text(encoding="utf-8"))
                                if isinstance(st_data, dict) and st_data.get("turn_index") == turn_idx:
                                    for req_f in st_data.get("required_files") or []:
                                        if req_f not in base_req_list:
                                            base_req_list.append(req_f)
                            except Exception:
                                pass
                        req_mods = [
                            CartridgeFileLockTable.normalize_path(m)
                            for m in base_req_list
                            if CartridgeFileLockTable.normalize_path(m)
                            and CartridgeFileLockTable.normalize_path(m) not in excluded_live
                        ]
                        req_code_mods = [
                            m for m in req_mods
                            if m.endswith(CODE_MODULE_SUFFIXES) or m == "level_layout.json"
                        ]
                        if not req_code_mods:
                            req_code_mods = ["scenes.gd"] if item.kind == "level_map" else ["main.gd"]
                        ok_lock, conflicts = ctx.file_lock_table.try_acquire_many(item.item_id, req_code_mods)
                        if not ok_lock:
                            if ctx.board is not None and hasattr(ctx.board, "expand_item_target_modules"):
                                ctx.board.expand_item_target_modules(item.item_id, req_code_mods)
                            _ACTIVE_TURN_CTX.state = None
                            _persist_trace(status="yielded", pending=decision, include_active_turn=False)
                            raise WorkerYielded(f"file_lock_busy:{','.join(conflicts)}")
                        if ctx.board is not None and hasattr(ctx.board, "expand_item_target_modules"):
                            ctx.board.expand_item_target_modules(
                                item.item_id,
                                list(ctx.file_lock_table.held_paths(item.item_id)),
                            )
                        code_lock_held = True
                        if task_baseline_snap is None:
                            task_baseline_snap = {}
                        new_locked = [
                            p for p in ctx.file_lock_table.held_paths(item.item_id)
                            if p not in task_baseline_snap
                        ]
                        if new_locked:
                            task_baseline_snap.update(
                                _snapshot_modules(ctx.cartridge_dir, new_locked, scoped_only=True)
                            )
                        _persist_trace(status="in_progress", pending=None, include_active_turn=True)
                elif not code_lock_held:
                    if is_in_board_scheduler():
                        if not code_lock.acquire(blocking=False):
                            _ACTIVE_TURN_CTX.state = None
                            _persist_trace(status="yielded", pending=decision, include_active_turn=False)
                            raise WorkerYielded(
                                f"Waiting for code_lock to run {decision.specialist_agent} on turn {turn_idx + 1} of '{item.item_id}'"
                            )
                    else:
                        code_lock.acquire()
                    code_lock_held = True
                    if task_baseline_snap is None:
                        task_baseline_snap = _snapshot_modules(
                            ctx.cartridge_dir,
                            list(item.target_modules or []) + ["project.godot", "main.tscn"],
                        )
                    _persist_trace(status="in_progress", pending=None, include_active_turn=True)

            if ctx.event_callback is not None:
                try:
                    ctx.event_callback(
                        "backlog_item_specialist_started",
                        {
                            "deliberation": f"Task Lead delegating Turn #{turn_idx + 1} of '{item.item_id}' to {decision.specialist_agent}.",
                            "item_id": item.item_id,
                            "revision": len(item.revisions),
                            "turn_index": turn_idx,
                            "specialist_agent": decision.specialist_agent,
                            "brief": decision.brief,
                        },
                    )
                except Exception:
                    pass

            pre_dispatch_snap: Optional[Dict[str, Optional[str]]] = None
            if decision.specialist_agent == "swe_engineer":
                snap_targets = (
                    list(ctx.file_lock_table.held_paths(item.item_id))
                    if (use_per_file_locks and ctx.file_lock_table is not None)
                    else (list(item.target_modules or []) + list(decision.target_paths or []))
                )
                pre_dispatch_snap = _snapshot_modules(
                    ctx.cartridge_dir,
                    snap_targets,
                    scoped_only=use_per_file_locks,
                )
            if isinstance(getattr(_ACTIVE_TURN_CTX, "state", None), dict):
                _ACTIVE_TURN_CTX.state["_task_baseline_snap_ref"] = task_baseline_snap
            try:
                report = _dispatch_to_specialist(
                    ctx,
                    item,
                    decision,
                    turn_index=turn_idx,
                    _lock_already_held=code_lock_held,
                )
                if decision.specialist_agent == "swe_engineer" and report.verification_passed:
                    if use_per_file_locks and ctx.file_lock_table is not None and pre_dispatch_snap is not None:
                        for touched_m in report.touched_files:
                            if touched_m not in pre_dispatch_snap and task_baseline_snap is not None and touched_m in task_baseline_snap:
                                pre_dispatch_snap[touched_m] = task_baseline_snap[touched_m]
                    last_verified_swe_snap = _snapshot_modules(
                        ctx.cartridge_dir,
                        list(task_baseline_snap.keys())
                        if task_baseline_snap is not None
                        else (list(item.target_modules or []) + list(decision.target_paths or []) + ["project.godot", "main.tscn"]),
                        scoped_only=use_per_file_locks,
                    )
            except WorkerYielded:
                _ACTIVE_TURN_CTX.state = None
                _persist_trace(status="yielded", pending=decision, include_active_turn=False)
                raise
            except StudioPausedError:
                _ACTIVE_TURN_CTX.state = None
                raise
            except Exception as spec_exc:
                qa_proofs: List[str] = []
                qa_telemetry_str = ""
                if decision.specialist_agent == "qa_test_engineer":
                    strip_prefix = "level_filmstrip" if item.kind == "level_map" else "filmstrip"
                    cand_strip = _item_dir(ctx, item) / f"{strip_prefix}_{_safe_id(item.item_id)}_rev{len(item.revisions)}.png"
                    if cand_strip.is_file():
                        qa_proofs.append(str(cand_strip))
                        try:
                            from engine.studio.proofs import format_filmstrip_telemetry_progression
                            qa_telemetry_str = format_filmstrip_telemetry_progression(cand_strip)
                        except Exception:
                            qa_telemetry_str = ""
                is_hard_qa_violation = (
                    "Counterfactual QA check failed" in str(spec_exc)
                    or "Entrypoint Parity Gate failed" in str(spec_exc)
                    or "QA Diagnostic Production Defect Report" in str(spec_exc)
                    or "runtime GDScript error" in str(spec_exc)
                )
                can_recover_turn = (
                    turn_idx + 1 < effective_max_turns
                    and decision.specialist_agent in ("qa_test_engineer", "swe_engineer")
                ) or (
                    decision.specialist_agent == "qa_test_engineer"
                    and not is_hard_qa_violation
                    and bool(qa_proofs or collected_proofs)
                )
                if can_recover_turn:
                    if use_per_file_locks and ctx.file_lock_table is not None and pre_dispatch_snap is not None and task_baseline_snap is not None:
                        for k_locked, v_base in task_baseline_snap.items():
                            if k_locked not in pre_dispatch_snap:
                                pre_dispatch_snap[k_locked] = v_base
                    partial_diffs = (
                        _compute_turn_code_diffs(
                            ctx.cartridge_dir,
                            pre_dispatch_snap,
                            scoped_only=use_per_file_locks,
                        )
                        if pre_dispatch_snap is not None
                        else {}
                    )
                    if (
                        decision.specialist_agent == "swe_engineer"
                        and pre_dispatch_snap is not None
                        and ctx.code_verify is not None
                    ):
                        _restore_modules(
                            ctx.cartridge_dir,
                            pre_dispatch_snap,
                            scoped_only=use_per_file_locks,
                        )
                    logger.warning(
                        "Specialist '%s' failed on Turn #%d of '%s' (%s); reporting failure to Task Lead for self-correction.",
                        decision.specialist_agent,
                        turn_idx + 1,
                        item.item_id,
                        spec_exc,
                    )
                    record_turn_substep(
                        "error",
                        f"Error in {decision.specialist_agent}: {str(spec_exc)[:240]}",
                        specialist_agent=decision.specialist_agent,
                    )
                    report = SpecialistReport(
                        deliberation=(
                            f"Specialist '{decision.specialist_agent}' encountered an error on Turn #{turn_idx + 1}; "
                            f"returning failure report to Task Lead so it can self-correct within the turn budget."
                        ),
                        turn_index=turn_idx,
                        specialist_agent=decision.specialist_agent,
                        brief=decision.brief,
                        summary=(
                            f"ERROR from {decision.specialist_agent}: {spec_exc}"
                            + (f"\n{qa_telemetry_str}" if qa_telemetry_str else "")
                            + "\nACTION REQUIRED FOR TASK LEAD: Diagnose this failure using the error, captured telemetry, and `CARTRIDGE SYMBOLS & FILES` "
                            "and delegate a fix to 'swe_engineer' (if GDScript/scene wiring, missing `load('res://...')` script, or physics is broken—NEVER pre-place target structures in `_ready()`) "
                            "or 'qa_test_engineer' (if the QA play script/mouse_world_pos/input_segments need adjustment)."
                        ),
                        touched_files=list(partial_diffs.keys()),
                        proof_images=qa_proofs,
                        verification_passed=False,
                        code_diffs=partial_diffs,
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    )
                else:
                    _ACTIVE_TURN_CTX.state = None
                    _persist_trace(status="failed", pending=None, include_active_turn=False)
                    raise
            finally:
                if use_per_file_locks and ctx.file_lock_table is not None and decision.specialist_agent == "swe_engineer":
                    ctx.file_lock_table.release_all(item.item_id)
                    code_lock_held = False
            live_st = getattr(_ACTIVE_TURN_CTX, "state", None) or {}
            if isinstance(live_st, dict):
                report.step_log = list(live_st.get("step_log") or report.step_log or [])
                report.curated_context = str(live_st.get("curated_context") or report.curated_context or "")
                report.prompt_text = str(live_st.get("prompt_text") or report.prompt_text or "")
                report.self_inquiries = list(live_st.get("self_inquiries") or report.self_inquiries or [])
                report.lead_deliberation = str(live_st.get("lead_deliberation") or decision.deliberation or "")
                report.lead_prompt_text = str(live_st.get("lead_prompt_text") or "")
                report.lead_curated_context = str(live_st.get("lead_curated_context") or "")
                report.lead_self_inquiries = list(live_st.get("lead_self_inquiries") or [])
                if not report.timestamp_utc:
                    report.timestamp_utc = datetime.now(timezone.utc).isoformat()
            _ACTIVE_TURN_CTX.state = None
            turn_history.append(report)
            if (
                "Counterfactual QA check failed" not in str(report.summary)
                and "Entrypoint Parity Gate failed" not in str(report.summary)
                and "QA Diagnostic Production Defect Report" not in str(report.summary)
            ):
                for p in report.proof_images:
                    if p not in collected_proofs and Path(p).is_file():
                        collected_proofs.append(p)
            _persist_trace(status="in_progress", pending=None, include_active_turn=False)
            if ctx.event_callback is not None:
                try:
                    ctx.event_callback(
                        "backlog_item_specialist_completed",
                        {
                            "deliberation": report.summary,
                            "item_id": item.item_id,
                            "revision": len(item.revisions),
                            "turn_index": turn_idx,
                            "specialist_agent": report.specialist_agent,
                            "summary": report.summary,
                            "proof_images": list(report.proof_images),
                            "verification_passed": report.verification_passed,
                        },
                    )
                except Exception:
                    pass
            turn_idx += 1
        else:
            qa_turns_hist = [t for t in turn_history if getattr(t, "specialist_agent", "") == "qa_test_engineer"]
            if qa_turns_hist and (
                "Counterfactual QA check failed" in str(getattr(qa_turns_hist[-1], "summary", ""))
                or "Entrypoint Parity Gate failed" in str(getattr(qa_turns_hist[-1], "summary", ""))
                or "QA Diagnostic Production Defect Report" in str(getattr(qa_turns_hist[-1], "summary", ""))
            ):
                _persist_trace(status="failed", pending=None)
                qa_raw_summary = str(getattr(qa_turns_hist[-1], "summary", "")).strip()
                if "\nACTION REQUIRED FOR TASK LEAD:" in qa_raw_summary:
                    qa_raw_summary = qa_raw_summary.split("\nACTION REQUIRED FOR TASK LEAD:", 1)[0].strip()
                raise StudioAgentError(
                    f"Task Lead for '{item.item_id}' exhausted {effective_max_turns} turns with failing QA gate: "
                    f"{qa_raw_summary[:1200]}"
                )
            if not collected_proofs:
                _persist_trace(status="failed", pending=None)
                raise StudioAgentError(
                    f"Task Lead for '{item.item_id}' exhausted {max_turns} turns without producing any proof images."
                )
            selected_proofs = list(collected_proofs)
            submission_summary = submission_summary or last_strategy
            _persist_trace(status="submitted", pending=None)
    except WorkerYielded:
        raise
    except Exception:
        restore_snap = last_verified_swe_snap if last_verified_swe_snap is not None else task_baseline_snap
        if restore_snap is not None:
            _restore_modules(ctx.cartridge_dir, restore_snap, scoped_only=use_per_file_locks)
        raise
    finally:
        if use_per_file_locks and ctx.file_lock_table is not None:
            ctx.file_lock_table.release_all(item.item_id)
        elif code_lock_held:
            code_lock.release()

    # If revising an item that previously fabricated asset plates (`plate_*.png`, `anim_sheet_*.png`, `waveform_*.png`),
    # preserve those plates alongside the newly captured proofs if they still exist on disk.
    if item.revisions:
        for prev_rev in item.revisions:
            for prev_p in prev_rev.proof_paths:
                name = Path(prev_p).name
                if (
                    name.startswith(("plate_", "anim_sheet_", "waveform_"))
                    and prev_p not in selected_proofs
                    and Path(prev_p).is_file()
                ):
                    selected_proofs.insert(0, prev_p)

    final_mod_prod: List[str] = []
    for t_rep in turn_history:
        if getattr(t_rep, "specialist_agent", "") == "qa_test_engineer":
            continue
        for tf in (getattr(t_rep, "touched_files", None) or []):
            norm_tf = CartridgeFileLockTable.normalize_path(str(tf))
            if norm_tf and not norm_tf.startswith("tests/") and norm_tf not in final_mod_prod:
                final_mod_prod.append(norm_tf)
    final_untouched_prod: List[str] = []
    for tm in (getattr(item, "target_modules", None) or []):
        norm_tm = CartridgeFileLockTable.normalize_path(str(tm))
        if (
            norm_tm
            and not norm_tm.startswith("tests/")
            and norm_tm not in final_mod_prod
            and norm_tm not in final_untouched_prod
        ):
            final_untouched_prod.append(norm_tm)
    verified_diff_tag = (
        f"[VERIFIED DISK DIFF: modified={final_mod_prod}, untouched_target_modules={final_untouched_prod}]"
    )
    turn_summaries = " | ".join(f"[Turn {t.turn_index} {t.specialist_agent}] {t.summary}" for t in turn_history)
    notes = f"{verified_diff_tag} | Strategy: {last_strategy}"
    if submission_summary and submission_summary != last_strategy:
        notes += f" | Summary: {submission_summary}"
    if turn_summaries:
        notes += f" | Specialist Turns: {turn_summaries}"
    return WorkerResult(proof_images=selected_proofs, notes=notes)


def task_lead_handler(ctx: WorkerContext, max_turns: int = 6) -> Callable[[WorkItem], WorkerResult]:
    """Handler that delegates any `WorkItem` to the iterative Task Lead Strategist loop."""
    if ctx.code_lock is None:
        ctx.code_lock = threading.Lock()
    fn = lambda item: run_task_lead(ctx, item, max_turns=max_turns)
    setattr(fn, "_code_lock", ctx.code_lock)
    setattr(fn, "_worker_ctx", ctx)
    return fn


def default_handlers(ctx: WorkerContext) -> Dict[str, Callable[[WorkItem], WorkerResult]]:
    """Routes all ProductionBoard item kinds through the iterative Task Lead Strategist."""
    lead = task_lead_handler(ctx)
    return {
        "visual_asset": lead,
        "animation": lead,
        "audio_asset": lead,
        "level_map": lead,
        "mechanic_code": lead,
        "ui": lead,
        "integration": lead,
    }
