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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PIL import Image, ImageDraw

from engine.studio.models import (
    AcceptanceCriterion,
    AssetRequirement,
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
            state["lead_curated_context"] = curated_context[:35000]
        if prompt_text is not None:
            state["lead_prompt_text"] = prompt_text[:45000]
        if self_inquiries is not None:
            state["lead_self_inquiries"] = list(self_inquiries)
        if deliberation is not None:
            state["lead_deliberation"] = deliberation
    else:
        if curated_context is not None:
            state["curated_context"] = curated_context[:35000]
        if prompt_text is not None:
            state["prompt_text"] = prompt_text[:45000]
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
        feedback_block += f"PREVIOUS WORKER EXECUTION ERROR TO FIX:\n{item.failure[-600:]}\n"

    curated_block = f"{rendering_block}\n--- GAME BIBLE ---\n{bible_text}\n"
    multimodal_parts: List[Any] = []
    if is_real_genai_client(ctx.client):
        curator = ContextCuratorAgent(
            cartridge_dir=ctx.cartridge_dir,
            client=ctx.client,
            model_name=ctx.model_name,
            board=ctx.board,
            event_callback=ctx.event_callback,
        )
        target_role = "audio_designer" if item.kind == "audio_asset" else "visual_artist"
        pack = curator.curate_context(
            target_agent_role=target_role,
            task_request=f"{item.title}\nSPEC: {item.spec}\nPROOF: {item.proof_spec}\n{feedback_block}",
            item_id=item.item_id,
            item_kind=item.kind,
        )
        curated_block = f"{pack.prompt_block}\n{rendering_block}\n"
        multimodal_parts = list(pack.multimodal_parts)

    issues: List[str] = []
    for attempt in range(1, attempts + 1):
        retry = "YOUR PREVIOUS ANSWER FAILED VALIDATION; FIX:\n- " + "\n- ".join(issues) + "\n\n" if issues else ""
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
            f"Provide `deliberation` FIRST.\n\n{curated_block}\n{retry}"
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
            # Ensure target_path cannot collide with another task's asset file in the shared catalog
            safe_item_id = _safe_id(item.item_id)
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
    """Merges the worker's catalog entry into the shared on-disk catalog without losing concurrent workers' updates."""
    entry = worker_lib.catalog.entries.get(asset_id)
    if entry is None:
        raise StudioAgentError(f"Fabrication of '{asset_id}' produced no catalog entry")
    with _CATALOG_LOCK:
        shared = ctx.new_library()
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
    eff_force = bool(item.revisions) if force_regenerate is None else bool(force_regenerate)
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


def render_waveform_image(wav_path: Path, out_path: Path, size: tuple = (900, 220)) -> Path:
    """Draws a waveform + duration/peak caption so the council can 'see' audio."""
    with wave.open(str(wav_path), "rb") as wf:
        n_frames, rate, width, channels = wf.getnframes(), wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
        raw = wf.readframes(n_frames)
    if width != 2:
        raise StudioAgentError(f"Unsupported WAV sample width {width} for {wav_path}")
    samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)[::channels] if raw else ()
    w, h = size
    img = Image.new("RGB", size, (16, 18, 28))
    draw = ImageDraw.Draw(img)
    mid = h // 2
    draw.line([(0, mid), (w, mid)], fill=(60, 64, 90))
    if samples:
        step = max(1, len(samples) // w)
        for x in range(w):
            chunk = samples[x * step:(x + 1) * step]
            if not chunk:
                break
            lo, hi = min(chunk) / 32768.0, max(chunk) / 32768.0
            draw.line([(x, mid - int(hi * (mid - 8))), (x, mid - int(lo * (mid - 8)))], fill=(120, 220, 255))
    peak = max((abs(s) for s in samples), default=0) / 32768.0
    draw.text((8, 4), f"{wav_path.name}  {n_frames / max(1, rate):.2f}s  {rate}Hz  peak={peak:.2f}", fill=(230, 230, 230))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def audio_asset_handler(ctx: WorkerContext) -> Callable[[WorkItem], WorkerResult]:
    def handler(item: WorkItem) -> WorkerResult:
        req = plan_asset_requirement(ctx, item, AUDIO_MODES)
        if ctx.voice_agent is None:
            raise StudioAgentError(f"No voice/audio agent is configured; cannot build audio item '{item.item_id}'.")
        lib = ctx.new_library()
        out = lib.fabricate_single_asset(req, force_regenerate=bool(item.revisions))
        if out is None or not Path(out).is_file():
            raise StudioAgentError(f"Audio fabrication of '{item.item_id}' produced no file.")
        _merge_catalog_entry(ctx, lib, req.asset_id)
        safe_id = _safe_id(item.item_id)
        proof = render_waveform_image(Path(out), _item_dir(ctx, item) / f"waveform_{safe_id}_rev{len(item.revisions)}.png")
        notes = f"Audio '{req.asset_id}' ({req.fabrication_mode}) -> {req.target_path}. Text/prompt: {(req.visual_prompt_or_text or req.description)[:300]}"
        return WorkerResult(proof_images=[str(proof), str(out)], notes=notes)

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
    for gd_path in sorted(c_dir.glob("*.gd")):
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


def _summarize_cartridge_for_qa(cartridge_dir: Path, max_chars: int = 4000) -> str:
    """Extracts a compact GDScript symbol, property, and asset catalog summary so the QA Director (`plan_filmstrip`)
    can stage interactions, invoke setup methods, and assert state keys accurately."""
    c_dir = Path(cartridge_dir)
    lines: List[str] = []
    for gd_path in sorted(c_dir.glob("*.gd")):
        if gd_path.name.startswith("test_"):
            continue
        try:
            src = gd_path.read_text(encoding="utf-8")
        except Exception:
            continue
        classes = re.findall(r"^\s*class\s+([A-Za-z0-9_]+)", src, flags=re.MULTILINE)
        funcs = [
            (fn, args)
            for fn, args in re.findall(r"^\s*(?:static\s+)?func\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)", src, flags=re.MULTILINE)
            if not fn.startswith("simulate_")
        ]
        vars_top = re.findall(r"^var\s+([A-Za-z0-9_]+)", src, flags=re.MULTILINE)
        bound_assets = sorted(set(re.findall(r"(?:res://)?assets/[A-Za-z0-9_./-]+\.(?:png|wav|ogg|mp3)", src)))
        func_sig = ", ".join(f"{fn}({args.strip()})" for fn, args in funcs[:18])
        lines.append(
            f"- {gd_path.name}: classes={classes[:8]} vars={vars_top[:14]} bound_assets={bound_assets} funcs=[{func_sig}]"
        )
    cat_path = c_dir / "assets" / "asset_library.json"
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
                    for k, v in list(entries.items())[:24]
                    if isinstance(v, dict)
                ]
                lines.append(f"- asset_library.json ({len(entries)} assets on disk): {entry_Descs}")
        except Exception:
            pass
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
    for call in getattr(plan, "setup_calls", []) or []:
        if call.at_tick < 1 or call.at_tick > MAX_PROOF_TICKS:
            issues.append(f"setup_call at_tick={call.at_tick} must be within [1, {MAX_PROOF_TICKS}].")
        m_clean = call.method_name.strip()
        m_lower = m_clean.lower()
        if not m_clean:
            issues.append("setup_call method_name must not be empty.")
        elif m_lower.startswith(_FORBIDDEN_SETUP_CALL_PREFIXES) or m_lower in _FORBIDDEN_SETUP_CALL_NAMES:
            issues.append(
                f"setup_call '{call.method_name}' is a 'simulate_*' / state-forcing shortcut that bypasses real player input/physics or forces onboarding/completion state. "
                f"Drive gameplay mechanics and onboarding step transitions causally via `input_segments` (`actions`, `mouse_click_pos`, or `when_snapshot_key` reactive triggers)."
            )
        try:
            parsed_args = json.loads(call.args_json or "[]")
            if not isinstance(parsed_args, list):
                issues.append(f"setup_call args_json for '{call.method_name}' must encode a JSON list.")
        except Exception:
            issues.append(f"setup_call args_json for '{call.method_name}' is not valid JSON.")
    if item_kind in ("mechanic_code", "integration") and actions and not plan.input_segments and not getattr(plan, "setup_calls", None):
        issues.append(
            f"FilmstripPlan for '{item_kind}' must include at least one `input_segments` entry (keyboard `actions` or `mouse_click_pos`) "
            f"or scenario `setup_calls` demonstrating causal progression from the t=0 baseline."
        )
    return issues


def _serialize_input_segments(plan: FilmstripPlan) -> List[Dict[str, Any]]:
    script: List[Dict[str, Any]] = []
    for s in plan.input_segments:
        seg_d: Dict[str, Any] = {"from": s.from_tick, "to": s.to_tick, "actions": list(s.actions)}
        if getattr(s, "mouse_click_pos", None) is not None:
            seg_d["mouse_click_pos"] = list(s.mouse_click_pos)
        if getattr(s, "when_snapshot_key", ""):
            seg_d["when_snapshot_key"] = s.when_snapshot_key
            seg_d["when_operator"] = getattr(s, "when_operator", "") or "=="
            seg_d["when_value"] = getattr(s, "when_value", None)
            seg_d["hold_ticks"] = int(getattr(s, "hold_ticks", 10) or 10)
        script.append(seg_d)
    return script


def plan_filmstrip(ctx: WorkerContext, item: WorkItem, attempts: int = 3) -> FilmstripPlan:
    """Asks the model how to demonstrate this item (inputs + scenario setup + frames), validated against the game's real input map and symbols."""
    if not _is_live(ctx.client):
        raise StudioAgentError("Filmstrip planning requires an active google.genai Client (Rule 5)")
    from google.genai import types
    from engine.studio.context_curator import ContextCuratorAgent, is_real_genai_client

    actions = available_input_actions(ctx.cartridge_dir)
    qa_context = _summarize_cartridge_for_qa(ctx.cartridge_dir)
    context_block = f"\nCARTRIDGE SYMBOLS & ASSETS AVAILABLE FOR QA STAGING:\n{qa_context}\n" if qa_context else ""
    multimodal_parts: List[Any] = []
    if is_real_genai_client(ctx.client):
        curator = ContextCuratorAgent(
            cartridge_dir=ctx.cartridge_dir,
            client=ctx.client,
            model_name=ctx.model_name,
            board=ctx.board,
            event_callback=ctx.event_callback,
        )
        lead_pack_path = _item_dir(ctx, item) / "context_pack_task_lead.json"
        curation_role = "task_lead" if lead_pack_path.is_file() else "qa_test_engineer"
        pack = curator.curate_context(
            target_agent_role=curation_role,
            task_request=f"{item.title}\nSPEC: {item.spec}\nPROOF: {item.proof_spec}",
            item_id=item.item_id,
            item_kind=item.kind,
        )
        context_block = f"{context_block}\n{pack.prompt_block}"
        multimodal_parts = list(pack.multimodal_parts)
    issues: List[str] = []
    for attempt in range(1, attempts + 1):
        retry = "\n\nYOUR PREVIOUS PLAN FAILED VALIDATION; FIX:\n- " + "\n- ".join(issues) if issues else ""
        prompt = (
            f"You are the QA Director planning a short scripted playthrough that PROVES a built item to the design council.\n"
            f"ITEM [{item.item_id}] ({item.kind}): {item.title}\nSPEC: {item.spec}\nTHE COUNCIL WANTS TO SEE: {item.proof_spec}\n"
            f"Available input actions: {actions}.{context_block}\n"
            f"Choose `capture_ticks` (1-{MAX_PROOF_FRAMES} frames, ticks 1-{MAX_PROOF_TICKS}, 60 ticks ~ 1 second), `input_segments` "
            f"(actions or KEY_* held over tick ranges, plus optional `mouse_click_pos`), and optional `setup_calls` (ONLY for scenario staging "
            f"such as spawning a hazard or positioning actors—NEVER call `simulate_*`, `force_*`, `complete_*`, or `set_onboarding_step` shortcuts that bypass real player input or physics) "
            f"so the captured frames show:\n"
            f"  - Frame 1 (`capture_ticks[0]`, before player inputs take effect): the UNSOLVED t=0 baseline state (`test_unsolved_baseline_t0`), and\n"
            f"  - Subsequent frames (`capture_ticks[1..]`): the authentic causal progression driven by `input_segments` (`actions` and/or `mouse_click_pos`).\n"
            f"REACTIVE TIMING TIP: When a secondary player input depends on dynamic physics or movement reaching a threshold (rather than a fixed tick guess), "
            f"populate `when_snapshot_key`, `when_operator` ('<=', '>=', '==', '!='), `when_value`, and `hold_ticks` on that `InputSegment` so the harness "
            f"fires the input at the exact tick the live simulation state satisfies the condition.\n"
            f"Also populate `state_assertions` for key observable state transitions (which are verified against live engine snapshots). "
            f"If the item spec or QA verification brief explicitly targets an isolated subsystem test scene `.tscn` file (e.g. `res://tests/test_*.tscn`) rather than the integrated main scene, set `target_scene` to that `res://tests/...tscn` path; otherwise leave `target_scene` empty (`\"\"`) to boot `res://main.tscn`. "
            f"Use no input only if the item kind is a purely static layout preview. Provide `deliberation` FIRST.{retry}"
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
        issues = validate_filmstrip_plan(
            plan,
            actions,
            item_kind=item.kind if is_real_genai_client(ctx.client) else "",
        )
        if not issues:
            return plan
    raise StudioAgentError(f"Filmstrip plan for '{item.item_id}' still invalid after {attempts} attempts: {issues}")


def _snapshot_modules(cartridge_dir: Path, modules: List[str]) -> Dict[str, Optional[str]]:
    c_dir = Path(cartridge_dir)
    normalized_mods: List[str] = []
    for m in modules:
        s = str(m or "").strip().replace("\\", "/")
        if s.startswith("res://"):
            s = s[len("res://"):]
        elif s.startswith("res:/"):
            s = s[len("res:/"):]
        s = s.lstrip("/")
        if s:
            normalized_mods.append(s)
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


def _restore_modules(cartridge_dir: Path, snap: Dict[str, Optional[str]]) -> None:
    c_dir = Path(cartridge_dir)
    for m, text in snap.items():
        p = c_dir / m
        if text is None:
            if p.is_file():
                p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
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


def _compute_turn_code_diffs(cartridge_dir: Path, pre_snap: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Computes unified diffs for any `.gd`, `.tscn`, `.gdshader`, or `project.godot` file created or modified since `pre_snap`."""
    import difflib

    c_dir = Path(cartridge_dir)
    post_snap = _snapshot_modules(c_dir, list(pre_snap.keys()))
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
            diffs[rel_path] = "".join(diff_lines)[:12000]
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
        notes = (
            f"Implemented '{item.item_id}' in {modules}; engine verification passed. Filmstrip ticks {plan.capture_ticks}. "
            f"Plan: {plan.deliberation[:300]} Expected frames: {plan.frame_expectations}{extra_qa}"
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
        force_regenerate=bool(item.revisions) or (turn_index > 0),
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
    res = audio_asset_handler(ctx)(sub_item)
    return SpecialistReport(
        deliberation=f"Audio Designer executed brief for '{sub_id}'.",
        turn_index=turn_index,
        specialist_agent="audio_designer",
        brief=brief,
        summary=res.notes,
        touched_files=list(target_paths or [f"assets/audio/{sub_id}.wav"]),
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
    prompt = (
        f"You are the Lead Scriptwriter & Narrative Dramatist executing a brief for TASK [{item.item_id}] ({item.kind}): {item.title}\n"
        f"TASK SPEC: {item.spec}\n"
        f"TASK LEAD BRIEF FOR SCRIPTWRITER:\n{brief}\n\n"
        f"{curated_context_block}"
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


def run_swe_engineer_specialist(
    ctx: WorkerContext,
    item: WorkItem,
    brief: str,
    target_paths: Optional[List[str]] = None,
    turn_index: int = 0,
    _lock_already_held: bool = False,
) -> SpecialistReport:
    """Autonomous SWE Engineer specialist: architects or modifies GDScript modules, world/level layouts,
    scenes, entities, or UI based on the Task Lead's brief and verifies headless engine compilation/boot
    without forcing a bundled filmstrip capture."""
    import contextlib
    from engine.studio.context_curator import is_real_genai_client

    agent = ctx.swe_agent
    if agent is None:
        raise StudioAgentError(f"No SWE agent is configured; cannot execute swe_engineer for '{item.item_id}'.")

    raw_lock = ctx.code_lock if ctx.code_lock is not None else threading.Lock()
    lock_cm = contextlib.nullcontext() if _lock_already_held else raw_lock
    fb = [(r.revision_index + 1, r.feedback) for r in item.revisions if r.feedback]
    canonical_root_gd = {"main.gd", "mechanics.gd", "scenes.gd", "entities.gd", "ui_hud.gd"}
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
        if clean_mod.endswith(CODE_MODULE_SUFFIXES) and clean_mod not in explicit_code_modules:
            explicit_code_modules.append(clean_mod)
    if item.kind == "level_map" and not explicit_code_modules and is_real_genai_client(ctx.client):
        explicit_code_modules = ["scenes.gd"]
    has_existing_layout = (ctx.cartridge_dir / "level_layout.json").is_file()
    explicitly_targets_layout = any("level_layout" in str(p) for p in (target_paths or []))
    wants_level_layout = (
        hasattr(agent, "formulate_level_layout")
        and (
            explicitly_targets_layout
            or (
                item.kind == "level_map"
                and (not has_existing_layout or not target_paths)
            )
            or (not explicit_code_modules and not hasattr(agent, "implement_requirement_track"))
        )
    )
    touched: List[str] = []
    proof_images: List[str] = []
    summaries: List[str] = []

    if wants_level_layout:
        from engine.studio.proofs import render_level_map
        guidance = f"[{item.item_id}] {item.title}: {item.spec}\nTask Lead Brief: {brief}"
        if fb:
            guidance += "\nFeedback to address:\n" + "\n".join(f"  rev {n}: {t}" for n, t in fb)
        safe_id = _safe_id(item.item_id)
        with lock_cm:
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
        description = (
            f"{item.spec}\n\n"
            f"TASK LEAD BRIEF FOR SWE ENGINEER:\n{brief}\n\n"
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
        import inspect
        swe_extra_kwargs: Dict[str, Any] = {}
        try:
            sig = inspect.signature(agent.implement_requirement_track)
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            prior_rev_proofs: List[str] = []
            if item.revisions:
                for p in (item.revisions[-1].proof_paths or []):
                    if p and p not in prior_rev_proofs:
                        prior_rev_proofs.append(p)
            if accepts_var_kw or "extra_image_paths" in sig.parameters:
                swe_extra_kwargs["extra_image_paths"] = prior_rev_proofs
            if accepts_var_kw or "board" in sig.parameters:
                swe_extra_kwargs["board"] = ctx.board
        except Exception:
            pass
        turn_code_diffs: Dict[str, str] = {}
        with lock_cm:
            if (
                is_real_genai_client(ctx.client)
                and hasattr(agent, "formulate_level_layout")
                and (ctx.cartridge_dir / "game_bible.json").is_file()
                and not (ctx.cartridge_dir / "level_layout.json").is_file()
            ):
                try:
                    agent.formulate_level_layout(ctx.cartridge_dir, force_refresh=False)
                except Exception as layout_exc:
                    logger.warning("Pre-code formulate_level_layout skipped: %s", layout_exc)
            from engine.studio.models import TOKEN_USAGE_TRACKER
            pre_turn_snap = _snapshot_modules(ctx.cartridge_dir, modules)
            agent.implement_requirement_track(
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
                        agent.implement_requirement_track(
                            track=track,
                            staging_dir=ctx.cartridge_dir,
                            requirements_spec=req_spec,
                            prior_error=str(verify_exc),
                            **swe_extra_kwargs,
                        )
                        ctx.code_verify(ctx.cartridge_dir, track)
                        TOKEN_USAGE_TRACKER.reset_swe_pev_streak()
                    else:
                        raise
            turn_code_diffs = _compute_turn_code_diffs(ctx.cartridge_dir, pre_turn_snap)
        for m in list(modules) + list(turn_code_diffs.keys()):
            if m not in touched:
                touched.append(m)
        summaries.append(
            f"Implemented '{item.item_id}' in {modules}; engine verification passed. "
            f"Available actions: {available_input_actions(ctx.cartridge_dir)}."
        )

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

    raw_lock = ctx.code_lock if ctx.code_lock is not None else threading.Lock()
    lock_cm = contextlib.nullcontext() if _lock_already_held else raw_lock
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
        plan = plan_filmstrip(ctx, qa_item)
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
        strip_prefix = "level_filmstrip" if item.kind == "level_map" else "filmstrip"
        safe_id = _safe_id(item.item_id)
        out_strip_path = _item_dir(ctx, item) / f"{strip_prefix}_{safe_id}_rev{len(item.revisions)}.png"
        resolved_scene = ""
        raw_scene = (getattr(plan, "target_scene", "") or "").strip()
        if not raw_scene:
            # Check if brief or target_paths explicitly specify a standalone test .tscn that exists on disk
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
                resolved_scene = f"res://{rel_scene}"
        record_turn_substep(
            "rendering_scene",
            f"Rendering scene {resolved_scene or 'res://main.tscn'} across ticks {list(plan.capture_ticks)}",
            details={"scene": resolved_scene or "res://main.tscn", "ticks": list(plan.capture_ticks)},
        )
        if ctx.filmstrip_fn is None:
            from engine.studio.proofs import capture_filmstrip
            strip = capture_filmstrip(
                ctx.cartridge_dir,
                ticks=list(plan.capture_ticks),
                out_path=out_strip_path,
                input_script=script,
                setup_script=setup_calls or None,
                verify_counterfactual=(item.kind != "level_map" and bool(script)),
                state_assertions=list(getattr(plan, "state_assertions", None) or []),
                target_scene=resolved_scene,
            )
        else:
            strip = ctx.filmstrip_fn(
                ctx.cartridge_dir,
                ticks=list(plan.capture_ticks),
                out_path=out_strip_path,
                input_script=script,
            )
        record_turn_substep(
            "captured_proof",
            f"Captured QA filmstrip {Path(strip).name}",
            details={"proof_image": str(strip)},
        )
    extra_qa = ""
    if setup_calls:
        extra_qa += f" Setup calls: {[c['method_name'] for c in setup_calls]}."
    if getattr(plan, "state_assertions", None):
        extra_qa += f" Assertions: {[(a.at_tick, a.snapshot_key, a.expected_value_or_change) for a in plan.state_assertions]}."

    summary = (
        f"Filmstrip ticks {plan.capture_ticks}. Plan: {plan.deliberation[:260]} "
        f"Pre-capture expected frames: {plan.frame_expectations}{extra_qa}"
    )
    return SpecialistReport(
        deliberation=plan.deliberation,
        turn_index=turn_index,
        specialist_agent="qa_test_engineer",
        brief=brief,
        summary=summary,
        touched_files=[],
        proof_images=[str(strip)],
    )


def validate_task_lead_decision(
    decision: TaskLeadDecision,
    specialists: List[str],
    collected_proofs: List[str],
    board: Any = None,
    parent_item_id: str = "",
    item_kind: str = "",
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


def _build_task_lead_multimodal_parts(ctx: WorkerContext, collected_proofs: List[str]) -> List[Any]:
    """Builds multimodal image parts (reference 16:9 Concept Art Mockup + latest collected proof images)
    so the Task Lead Strategist can see the target visual aesthetic from Turn 0 onward and visually inspect
    captured proofs before submitting to the Council."""
    from google.genai import types

    parts: List[Any] = []
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

    valid_proofs = [Path(p_str) for p_str in collected_proofs if Path(p_str).is_file()]
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
        parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime))
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
        if turn_history and any(not str(t.summary).startswith("FAILED") for t in turn_history):
            prior_revision_block += (
                "\nPRIOR ATTEMPT ERROR (occurred BEFORE Turn 0 of this attempt; if a specialist in CURRENT REVISION TURN HISTORY "
                "already updated the code to address this, proceed to the next verification/proof specialist such as qa_test_engineer):\n"
                f"{item.failure[-800:]}\n"
            )
        else:
            prior_revision_block += f"\nPREVIOUS WORKER EXECUTION ERROR TO FIX:\n{item.failure[-800:]}\n"

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
            "  - 'swe_engineer': architects or modifies GDScript modules (.gd), world/level layouts, scenes, entities, or UI and verifies headless engine compilation/boot.\n"
        ),
        "qa_test_engineer": (
            "  - 'qa_test_engineer': designs and executes a scripted Godot headless test scenario (setup_calls, input_segments, mouse_click_pos, state_assertions, capture_ticks) and captures a multi-frame gameplay filmstrip proof image.\n"
        ),
    }
    from engine.studio.context_curator import ContextCuratorAgent, SelfInquiryQAPair, is_real_genai_client

    available_spec_lines = "".join(spec_descriptions[s] for s in specialists if s in spec_descriptions)
    curated_lead_block = ""
    multimodal_parts: List[Any] = []
    prior_rev_proofs: List[str] = []
    if item.revisions:
        for p in (item.revisions[-1].proof_paths or []):
            if p and p not in prior_rev_proofs:
                prior_rev_proofs.append(p)
    lead_image_paths = list(dict.fromkeys(prior_rev_proofs + list(collected_proofs)))
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
        multimodal_parts = list(pack.multimodal_parts) or _build_task_lead_multimodal_parts(ctx, lead_image_paths)
        if collected_proofs and pack.multimodal_parts:
            existing_labels = {p for p in multimodal_parts if isinstance(p, str)}
            extra_proof_parts = _build_task_lead_multimodal_parts(ctx, collected_proofs)
            for idx_p in range(0, len(extra_proof_parts), 2):
                lbl = extra_proof_parts[idx_p]
                if (
                    isinstance(lbl, str)
                    and lbl.startswith("CAPTURED PROOF IMAGE")
                    and lbl not in existing_labels
                    and idx_p + 1 < len(extra_proof_parts)
                ):
                    multimodal_parts.append(lbl)
                    multimodal_parts.append(extra_proof_parts[idx_p + 1])
                    existing_labels.add(lbl)

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

    attempt = 1
    while attempt <= attempts:
        retry = "\nYOUR PREVIOUS TaskLeadDecision FAILED VALIDATION; FIX:\n- " + "\n- ".join(issues) + "\n" if issues else ""
        inquiry_block = (
            active_curator.format_self_inquiry_prompt_section(resolved_inquiries, pending_question_pack)
            if active_curator is not None
            else ""
        )
        prompt = (
            f"You are the Task Lead Strategist for TASK [{item.item_id}] ({item.kind}): {item.title}\n"
            f"SPEC: {item.spec}\n"
            f"PROOF THE COUNCIL WILL JUDGE: {item.proof_spec}\n"
            f"TARGET MODULES HINT: {item.target_modules}\n"
            f"AVAILABLE AUTONOMOUS SPECIALISTS: {specialists}\n"
            f"{available_spec_lines}"
            f"INSTRUCTIONS:\n"
            f"1. Formulate your bespoke `current_strategy` and `proof_strategy` for creating everything necessary for this task and generating the final proof (and pivoting if the Council requested changes):\n"
            f"   - Creation Strategy (`current_strategy`): Plan every artifact, asset, multi-cell sheet/grid, rig, script, or GDScript module required by `SPEC`, and decide which specialist(s) in {specialists} to delegate to in sequence.\n"
            f"   - Proof Generation Strategy (`proof_strategy`): Plan how the final visual proof artifact(s) (`selected_proof_paths`) will be generated and verified to directly satisfy `PROOF THE COUNCIL WILL JUDGE` (`{item.proof_spec}`).\n"
            f"   - When delegating to `'visual_artist'` for a task that requires a multi-cell grid, sheet, or atlas of distinct visual states, icons, stages, or components (e.g. a 2x2 or NxM state sheet where each cell depicts a distinct graphic/state), explicitly specify the `grid_cols x grid_rows` layout and each cell's visual contents in row-major order in your `brief` so `'visual_artist'` populates `grid_cols`, `grid_rows`, and `cell_prompts`.\n"
            f"   - SELF-INQUIRY TOOL (`self_question` -> ContextCuratorAgent research -> `self_question_answer`): If you need to inspect a specific module's code, signals, or design spec that is not in your current context before writing your specialist `brief`, you may output a specific question in `self_question` (leaving `brief` empty). `ContextCuratorAgent` will fetch the relevant chunks on the next turn so YOU can answer your own question in `self_question_answer` and then delegate accurately.\n"
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
            f"   - UNSOLVED t=0 BASELINE (`test_unsolved_baseline_t0`) — NEVER CORRUPT `_ready()` FOR SUBSYSTEM QA PROOFS: `_ready()` at `t=0` is what the live human player boots into! `_ready()` MUST initialize the world in its unbuilt/unsolved starting state as defined by the Game Bible. NEVER pre-connect, pre-build, or pre-solve the player's work inside `_ready()` just so a downstream mechanic is visible at tick 0 in a QA screenshot—instead, instruct `'qa_test_engineer'` to use `input_segments` (mouse clicks/drags and keys) to construct and trigger the mechanic from the clean `t=0` state!\n"
            f"   - REAL INTERACTIVE VERBS & CAMERA NAVIGATION: Every core player verb in `core_loop.player_verbs` and the active slice's `scope_boundaries` (including mouse click/drag placement or connection, interactive component tuning/socketing, hazard clearing, and `Camera2D` panning/scrolling when `world_bounds` exceeds the 1280x720 viewport) MUST be wired to real `_unhandled_input` / `_input` / UI handlers and exercised via `input_segments` (`mouse_click_pos` or actions) in QA filmstrips.\n"
            f"   - Every input key/action bound in the codebase (`Available actions`) MUST have a discoverable in-world or HUD affordance—never bind hidden/secret keys that the player cannot know exist, and NEVER bind fake tutorial-skip keys (such as `KEY_TAB` or `set_onboarding_step`) that advance onboarding text without the player performing the actual in-world mechanic.\n"
            f"   - Never rig an input handler to bypass physics, force a win state, or teleport world state when a timing window is missed; instead, instruct `'qa_test_engineer'` to use reactive state triggers (`when_snapshot_key`, `when_operator`, `when_value`) to test timing-sensitive mechanics honestly.\n"
            f"   - Prevent duplicate rendering across modules: ensure a single authoritative module owns each world structure or visual effect so `main.gd`, `scenes.gd`, `entities.gd`, and `ui_hud.gd` do not draw overlapping copies of the same geometry or indicator.\n"
            f"4. If `COLLECTED PROOF IMAGES SO FAR` is non-empty, visually inspect the attached proof image(s) (and any `QA Post-Capture Visual Audit` findings in the turn history) before choosing `submit_to_council`:\n"
            f"   - In `filmstrip_*.png` contact sheets, the top (`[QA] tick N in:[...]`) and bottom (`[QA] pos=(...) ...`) bands outside each 16:9 frame border are external QA test-harness annotations—use them to verify coordinates/state, and never mistake them for in-game debug text.\n"
            f"   - In `plate_*.png` and `anim_sheet_*.png` / `anim_rig_*.png`, the dark/light square checkerboard grids, cyan bounding-box borders, and horizontal yellow/gold baseline ruler lines at the bottom of frames are external QA test-harness overlays drawn behind/around the transparent PNG—NEVER mistake those external QA overlays for cross-hatch grids or yellow line artifacts inside the sprite PNG!\n"
            f"   - NEVER SUBMIT PLACEHOLDER STICKMEN, CLIPPED SPRITES, DUPLICATE OVERLAPPING GEOMETRY, OR SHRUNK VIEWPORT BACKGROUNDS: If the proof image or `QA Post-Capture Visual Audit` reports `qa_passed=False`, unaddressed Council defects, clipped/invisible sprites (`hframes > 1` on a single-frame texture), or a shrunk background that leaves dark void around the level, do NOT choose `submit_to_council`! Fix the root cause with `'visual_artist'` / `'swe_engineer'` and re-run `'qa_test_engineer'`.\n"
            f"   - Verify that active gameplay entities (`CharacterBody2D` / `Sprite2D` actors and floor/world geometry) are visibly rendered inside the 16:9 viewport and are NOT hidden behind an opaque full-screen background/border frame drawn at a higher `z_index` or foreground `CanvasLayer`.\n"
            f"5. Choose `action = 'delegate'` to send an open-ended natural-language `brief` to one of {specialists}, `action = 'spawn_tasks'` to create newly discovered tasks on the board, OR `action = 'submit_to_council'` ONLY after all required assets/code are built AND at least one proof image in `COLLECTED PROOF IMAGES SO FAR` visually proves the requirement.\n"
            f"   - CRITICAL SCHEMA FIELD ORDER WARNING: In `TaskLeadDecision`, `specialist_agent` and `brief` appear BEFORE `action` in the JSON schema! Decide your turn action inside `deliberation` and `current_strategy` first: whenever you are delegating to a specialist (`action = 'delegate'`), you MUST populate `specialist_agent` (one of {specialists}) and a non-empty `brief` BEFORE emitting `action = 'delegate'`! Never leave `specialist_agent` or `brief` empty when delegating.\n"
            f"Provide `deliberation` FIRST and `action` LAST.\n"
            f"{slice_context_block}"
            f"{curated_lead_block}"
            f"{inquiry_block}"
            f"{prior_revision_block}"
            f"{board_snapshot_block}"
            f"CARTRIDGE SYMBOLS & FILES:\n{qa_context or '(empty)'}{cat_block}"
            f"COLLECTED PROOF IMAGES SO FAR: {collected_proofs}\n"
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
                    thinking_config=bounded_thinking_config(1536),
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

        issues = validate_task_lead_decision(
            decision,
            specialists,
            collected_proofs,
            board=ctx.board,
            parent_item_id=item.item_id,
            item_kind=item.kind,
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
            ctx, item, decision.brief, decision.target_paths, turn_index=turn_index, _lock_already_held=_lock_already_held
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
            if (
                isinstance(existing_trace, dict)
                and existing_trace.get("status", "in_progress")
                in ("in_progress", "yielded", "waiting_for_spawned_tasks")
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

    code_lock = ctx.code_lock if ctx.code_lock is not None else threading.Lock()
    code_lock_held = False
    task_baseline_snap: Optional[Dict[str, Optional[str]]] = None
    last_verified_swe_snap: Optional[Dict[str, Optional[str]]] = None
    spawn_turn_bonus = sum(1 for t in turn_history if t.specialist_agent == "board_task_spawner")
    effective_max_turns = max_turns + spawn_turn_bonus
    try:
        from datetime import datetime, timezone
        turn_idx = len(turn_history)
        while turn_idx < effective_max_turns:
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

            if decision.specialist_agent in ("swe_engineer", "qa_test_engineer") and not code_lock_held:
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
                pre_dispatch_snap = _snapshot_modules(
                    ctx.cartridge_dir,
                    list(item.target_modules or []) + list(decision.target_paths or []),
                )
            try:
                report = _dispatch_to_specialist(
                    ctx,
                    item,
                    decision,
                    turn_index=turn_idx,
                    _lock_already_held=code_lock_held,
                )
                if decision.specialist_agent == "swe_engineer" and report.verification_passed:
                    last_verified_swe_snap = _snapshot_modules(
                        ctx.cartridge_dir,
                        list(task_baseline_snap.keys())
                        if task_baseline_snap is not None
                        else (list(item.target_modules or []) + list(decision.target_paths or []) + ["project.godot", "main.tscn"]),
                    )
            except (WorkerYielded, StudioPausedError):
                _ACTIVE_TURN_CTX.state = None
                raise
            except Exception as spec_exc:
                qa_proofs: List[str] = []
                if decision.specialist_agent == "qa_test_engineer":
                    strip_prefix = "level_filmstrip" if item.kind == "level_map" else "filmstrip"
                    cand_strip = _item_dir(ctx, item) / f"{strip_prefix}_{_safe_id(item.item_id)}_rev{len(item.revisions)}.png"
                    if cand_strip.is_file():
                        qa_proofs.append(str(cand_strip))
                can_recover_turn = (
                    turn_idx + 1 < effective_max_turns
                    and decision.specialist_agent in ("qa_test_engineer", "swe_engineer")
                ) or (
                    decision.specialist_agent == "qa_test_engineer"
                    and bool(qa_proofs or collected_proofs)
                )
                if can_recover_turn:
                    partial_diffs = (
                        _compute_turn_code_diffs(ctx.cartridge_dir, pre_dispatch_snap)
                        if pre_dispatch_snap is not None
                        else {}
                    )
                    if (
                        decision.specialist_agent == "swe_engineer"
                        and pre_dispatch_snap is not None
                        and ctx.code_verify is not None
                    ):
                        _restore_modules(ctx.cartridge_dir, pre_dispatch_snap)
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
                            f"ERROR from {decision.specialist_agent}: {spec_exc}\n"
                            f"ACTION REQUIRED FOR TASK LEAD: Diagnose this failure and delegate a fix to 'swe_engineer' "
                            f"(if GDScript/scene wiring or performance is broken) or 'qa_test_engineer' (if the QA test plan/setup_calls need adjustment)."
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
            _restore_modules(ctx.cartridge_dir, restore_snap)
        raise
    finally:
        if code_lock_held:
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

    turn_summaries = " | ".join(f"[Turn {t.turn_index} {t.specialist_agent}] {t.summary}" for t in turn_history)
    notes = f"Strategy: {last_strategy}"
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
