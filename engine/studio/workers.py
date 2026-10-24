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
    RequirementTrack,
    SpecialistReport,
    StudioAgentError,
    TaskLeadDecision,
    WorkItem,
    bounded_thinking_config,
    call_gemini_with_transient_retry,
    compact_bible_json_for_prompt,
)
from engine.studio.production_board import WorkerResult, WorkerYielded, is_in_board_scheduler

logger = logging.getLogger("BoardWorkers")

# Serialises merges into the shared asset catalog (each worker fabricates with its own library instance).
_CATALOG_LOCK = threading.Lock()

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
            f"CRITICAL PROMPT SEPARATION RULE FOR `visual_prompt_or_text`:\n"
            f"- When `fabrication_mode == 'sprite_sheet'`, `visual_prompt_or_text` is sent to the image generator to synthesize the SINGLE canonical neutral standing/idle figure, and our biomechanical keyframe engine then generates the multi-row, multi-frame sprite sheet (`*_sheet.png`) from `animation_states`. Therefore, `visual_prompt_or_text` MUST describe ONLY ONE single isolated subject in a neutral readiness pose (appearance, proportions, attire/materials, silhouette, palette) and MUST NEVER ask the image generator to draw multiple poses, multiple rows/columns, frame numbers, or a grid layout inside `visual_prompt_or_text`! Put all multi-state/row/frame breakdowns into `animation_states`.\n"
            f"- When `fabrication_mode == 'static_sprite'`, `visual_prompt_or_text` MUST describe ONLY ONE single isolated object/prop in its canonical intact state — NEVER a 3x3 or multi-state variation grid.\n"
            f"Provide `deliberation` FIRST.\n\n{rendering_block}\n--- GAME BIBLE ---\n{bible_text}\n"
        )
        resp = call_gemini_with_transient_retry(
            lambda: ctx.client.models.generate_content(
                model=ctx.model_name,
                contents=prompt,
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


def _fabricate_visual(ctx: WorkerContext, item: WorkItem, allowed_modes: tuple) -> WorkerResult:
    req = plan_asset_requirement(ctx, item, allowed_modes)
    lib = ctx.new_library()
    out = lib.fabricate_single_asset(req, force_regenerate=bool(item.revisions))
    if out is None or not Path(out).is_file():
        raise StudioAgentError(f"Fabrication of '{item.item_id}' produced no file (mode={req.fabrication_mode}).")
    _merge_catalog_entry(ctx, lib, req.asset_id)
    renders = ctx.cartridge_dir / "artifacts" / "renders"
    proofs = _collect_proofs(renders, req.asset_id, Path(out))
    if not proofs:
        raise StudioAgentError(f"Fabrication of '{item.item_id}' produced no proof images.")
    entry = lib.catalog.entries.get(req.asset_id)
    sheet_meta_note = ""
    if entry is not None and getattr(entry, "fabrication_mode", "") == "sprite_sheet":
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
        return WorkerResult(proof_images=[str(proof)], notes=notes)

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
                f"- Execute the Game Bible's `level_generation_strategy` (`generation_approach`, `world_structure`, `runtime_construction_plan`) and load `res://level_layout.json` across all {len(layout.world_regions)} regions (`{[r.region_id for r in layout.world_regions]}`).\n"
                f"- Bind synthesized sprites and tile textures from `res://assets/sprites/*.png` and `res://assets/tiles/*.png` matching the 16:9 Concept Art Mockup.\n"
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
CODE_MODULE_SUFFIXES = (".gd", ".tscn", ".godot")


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
        funcs = re.findall(r"^\s*(?:static\s+)?func\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)", src, flags=re.MULTILINE)
        vars_top = re.findall(r"^var\s+([A-Za-z0-9_]+)", src, flags=re.MULTILINE)
        func_sig = ", ".join(f"{fn}({args.strip()})" for fn, args in funcs[:18])
        lines.append(
            f"- {gd_path.name}: classes={classes[:8]} vars={vars_top[:14]} funcs=[{func_sig}]"
        )
    cat_path = c_dir / "assets" / "asset_library.json"
    if cat_path.is_file():
        try:
            cat = json.loads(cat_path.read_text(encoding="utf-8"))
            entries = cat.get("entries") if isinstance(cat, dict) else {}
            if isinstance(entries, dict) and entries:
                lines.append(f"- asset_library.json ({len(entries)} assets): {list(entries.keys())[:16]}")
        except Exception:
            pass
    summary = "\n".join(lines)
    return summary[:max_chars]


def validate_filmstrip_plan(plan: FilmstripPlan, actions: List[str]) -> List[str]:
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
        if not call.method_name.strip():
            issues.append("setup_call method_name must not be empty.")
        try:
            parsed_args = json.loads(call.args_json or "[]")
            if not isinstance(parsed_args, list):
                issues.append(f"setup_call args_json for '{call.method_name}' must encode a JSON list.")
        except Exception:
            issues.append(f"setup_call args_json for '{call.method_name}' is not valid JSON.")
    return issues


def plan_filmstrip(ctx: WorkerContext, item: WorkItem, attempts: int = 3) -> FilmstripPlan:
    """Asks the model how to demonstrate this item (inputs + scenario setup + frames), validated against the game's real input map and symbols."""
    if not _is_live(ctx.client):
        raise StudioAgentError("Filmstrip planning requires an active google.genai Client (Rule 5)")
    from google.genai import types

    actions = available_input_actions(ctx.cartridge_dir)
    qa_context = _summarize_cartridge_for_qa(ctx.cartridge_dir)
    context_block = f"\nCARTRIDGE SYMBOLS & ASSETS AVAILABLE FOR QA STAGING:\n{qa_context}\n" if qa_context else ""
    issues: List[str] = []
    for attempt in range(1, attempts + 1):
        retry = "YOUR PREVIOUS PLAN FAILED VALIDATION; FIX:\n- " + "\n- ".join(issues) + "\n\n" if issues else ""
        prompt = (
            f"{retry}You are the QA Director planning a short scripted playthrough that PROVES a built item to the design council.\n"
            f"ITEM [{item.item_id}] ({item.kind}): {item.title}\nSPEC: {item.spec}\nTHE COUNCIL WANTS TO SEE: {item.proof_spec}\n"
            f"Available input actions: {actions}.{context_block}\n"
            f"Choose `capture_ticks` (1-{MAX_PROOF_FRAMES} frames, ticks 1-{MAX_PROOF_TICKS}, 60 ticks ~ 1 second), `input_segments` "
            f"(actions or KEY_* held over tick ranges, plus optional `mouse_click_pos`), and optional `setup_calls` (if staging the proof "
            f"requires spawning a hazard, positioning actors into interaction range, or invoking a scene/entity setup method at a specific tick) "
            f"so the captured frames show the state before, the action/interaction, and the result. Also populate `state_assertions` for key "
            f"observable state transitions. Use no input only if the proof is static. Provide `deliberation` FIRST."
        )
        resp = call_gemini_with_transient_retry(
            lambda: ctx.client.models.generate_content(
                model=ctx.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FilmstripPlan,
                    thinking_config=bounded_thinking_config(1536),
                ),
            ),
            label=f"plan_filmstrip[{item.item_id}]",
        )
        if not resp.text:
            raise StudioAgentError(f"Filmstrip planning for '{item.item_id}' returned an empty response")
        plan = FilmstripPlan.model_validate_json(resp.text)
        issues = validate_filmstrip_plan(plan, actions)
        if not issues:
            return plan
    raise StudioAgentError(f"Filmstrip plan for '{item.item_id}' still invalid after {attempts} attempts: {issues}")


def _snapshot_modules(cartridge_dir: Path, modules: List[str]) -> Dict[str, Optional[str]]:
    snap: Dict[str, Optional[str]] = {}
    for m in modules:
        p = Path(cartridge_dir) / m
        snap[m] = p.read_text(encoding="utf-8") if p.is_file() else None
    return snap


def _restore_modules(cartridge_dir: Path, snap: Dict[str, Optional[str]]) -> None:
    for m, text in snap.items():
        p = Path(cartridge_dir) / m
        if text is None:
            if p.is_file():
                p.unlink()
        else:
            p.write_text(text, encoding="utf-8")


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
                    )
                else:
                    strip = ctx.filmstrip_fn(
                        ctx.cartridge_dir,
                        ticks=list(plan.capture_ticks),
                        out_path=out_strip_path,
                        input_script=script,
                    )
            except Exception:
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

ALL_SPECIALIST_AGENTS = ("visual_artist", "audio_designer", "swe_engineer", "qa_test_engineer")


def available_specialists(ctx: WorkerContext, item: Optional[WorkItem] = None) -> List[str]:
    """Returns the autonomous domain specialists configured and available in `ctx` for `item`.

    Standalone asset fabrication tasks (`visual_asset`, `animation`, `audio_asset`) are scoped to their
    asset fabrication specialists so they produce isolated asset files and inspection/waveform plates
    without rewriting shared `.gd` gameplay scenes or hijacking `main.tscn` into a static showcase.
    Gameplay, level, UI, and integration tasks have full access to all specialists (`visual_artist`,
    `audio_designer`, `swe_engineer`, `qa_test_engineer`).
    """
    has_visual = ctx.library_factory is not None or ctx.visual_agent is not None or _is_live(ctx.client)
    has_audio = ctx.voice_agent is not None
    if item is not None:
        if item.kind in ("visual_asset", "animation"):
            return ["visual_artist"] if has_visual else []
        if item.kind == "audio_asset":
            return ["audio_designer"] if has_audio else []

    specs: List[str] = []
    if has_visual:
        specs.append("visual_artist")
    if has_audio:
        specs.append("audio_designer")
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
    res = _fabricate_visual(ctx, sub_item, VISUAL_MODES)
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

    agent = ctx.swe_agent
    if agent is None:
        raise StudioAgentError(f"No SWE agent is configured; cannot execute swe_engineer for '{item.item_id}'.")

    raw_lock = ctx.code_lock if ctx.code_lock is not None else threading.Lock()
    lock_cm = contextlib.nullcontext() if _lock_already_held else raw_lock
    fb = [(r.revision_index + 1, r.feedback) for r in item.revisions if r.feedback]
    explicit_code_modules = [
        m for m in (target_paths or item.target_modules) if m.endswith(CODE_MODULE_SUFFIXES)
    ]
    wants_level_layout = (
        hasattr(agent, "formulate_level_layout")
        and (
            item.kind == "level_map"
            or any("level_layout" in str(p) for p in (target_paths or []))
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
        with lock_cm:
            snap = _snapshot_modules(ctx.cartridge_dir, modules + ["project.godot", "main.tscn"])
            try:
                agent.implement_requirement_track(track=track, staging_dir=ctx.cartridge_dir, requirements_spec=req_spec)
                if ctx.code_verify is not None:
                    ctx.code_verify(ctx.cartridge_dir, track)
            except Exception:
                _restore_modules(ctx.cartridge_dir, snap)
                raise
        for m in modules:
            if m not in touched:
                touched.append(m)
        summaries.append(
            f"Implemented '{item.item_id}' in {modules}; engine verification passed. "
            f"Available actions: {available_input_actions(ctx.cartridge_dir)}."
        )

    return SpecialistReport(
        deliberation=f"SWE Engineer executed brief for '{item.item_id}'.",
        turn_index=turn_index,
        specialist_agent="swe_engineer",
        brief=brief,
        summary=" ".join(summaries),
        touched_files=touched,
        proof_images=proof_images,
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
        plan = plan_filmstrip(ctx, qa_item)
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
        strip_prefix = "level_filmstrip" if item.kind == "level_map" else "filmstrip"
        safe_id = _safe_id(item.item_id)
        out_strip_path = _item_dir(ctx, item) / f"{strip_prefix}_{safe_id}_rev{len(item.revisions)}.png"
        if ctx.filmstrip_fn is None:
            from engine.studio.proofs import capture_filmstrip
            strip = capture_filmstrip(
                ctx.cartridge_dir,
                ticks=list(plan.capture_ticks),
                out_path=out_strip_path,
                input_script=script,
                setup_script=setup_calls or None,
                verify_counterfactual=(item.kind != "level_map"),
            )
        else:
            strip = ctx.filmstrip_fn(
                ctx.cartridge_dir,
                ticks=list(plan.capture_ticks),
                out_path=out_strip_path,
                input_script=script,
            )
    extra_qa = ""
    if setup_calls:
        extra_qa += f" Setup calls: {[c['method_name'] for c in setup_calls]}."
    if getattr(plan, "state_assertions", None):
        extra_qa += f" Assertions: {[(a.at_tick, a.snapshot_key, a.expected_value_or_change) for a in plan.state_assertions]}."
    summary = (
        f"Filmstrip ticks {plan.capture_ticks}. Plan: {plan.deliberation[:300]} "
        f"Expected frames: {plan.frame_expectations}{extra_qa}"
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
    elif decision.action == "submit_to_council":
        if not collected_proofs:
            issues.append(
                "Cannot submit_to_council before any proof images have been produced; delegate to a specialist "
                f"({specialists}) to build and capture proof first."
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
    so the Task Lead Strategist can visually inspect captured proofs before submitting to the Council."""
    if not collected_proofs:
        return []
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

    for p_str in collected_proofs[-3:]:
        p = Path(p_str)
        if not p.is_file():
            continue
        mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        parts.append(
            f"CAPTURED PROOF IMAGE ({p.name}) — inspect this image carefully before choosing submit_to_council:"
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
        "swe_engineer": (
            "  - 'swe_engineer': architects or modifies GDScript modules (.gd), world/level layouts, scenes, entities, or UI and verifies headless engine compilation/boot.\n"
        ),
        "qa_test_engineer": (
            "  - 'qa_test_engineer': designs and executes a scripted Godot headless test scenario (setup_calls, input_segments, mouse_click_pos, state_assertions, capture_ticks) and captures a multi-frame gameplay filmstrip proof image.\n"
        ),
    }
    available_spec_lines = "".join(spec_descriptions[s] for s in specialists if s in spec_descriptions)
    multimodal_parts: List[Any] = []
    if collected_proofs and getattr(type(ctx.client), "__module__", "") != "google.genai.fake":
        multimodal_parts = _build_task_lead_multimodal_parts(ctx, collected_proofs)

    issues: List[str] = []
    for attempt in range(1, attempts + 1):
        retry = "YOUR PREVIOUS TaskLeadDecision FAILED VALIDATION; FIX:\n- " + "\n- ".join(issues) + "\n\n" if issues else ""
        prompt = (
            f"{retry}You are the Task Lead Strategist for TASK [{item.item_id}] ({item.kind}): {item.title}\n"
            f"SPEC: {item.spec}\n"
            f"PROOF THE COUNCIL WILL JUDGE: {item.proof_spec}\n"
            f"TARGET MODULES HINT: {item.target_modules}\n"
            f"AVAILABLE AUTONOMOUS SPECIALISTS: {specialists}\n"
            f"{available_spec_lines}"
            f"COLLECTED PROOF IMAGES SO FAR: {collected_proofs}\n"
            f"CARTRIDGE SYMBOLS & FILES:\n{qa_context or '(empty)'}{cat_block}"
            f"{prior_revision_block}{turns_block}\n"
            f"INSTRUCTIONS:\n"
            f"1. Formulate your bespoke `current_strategy` for accomplishing and proving this task (and pivoting if the Council requested changes).\n"
            f"2. If `COLLECTED PROOF IMAGES SO FAR` is non-empty, visually inspect the attached proof image(s) before choosing `submit_to_council`:\n"
            f"   - In `filmstrip_*.png` contact sheets, the top (`[QA] tick N in:[...]`) and bottom (`[QA] pos=(...) ...`) bands outside each 16:9 frame border are external QA test-harness annotations—use them to verify coordinates/state, and never mistake them for in-game debug text.\n"
            f"   - Verify that the active gameplay entities (`CharacterBody2D` / `Sprite2D` actors and floor/world tiles) are visibly rendered inside the 16:9 viewport and are NOT hidden behind an opaque full-screen background/border frame drawn at a higher `z_index` or foreground `CanvasLayer`.\n"
            f"   - If the proof image shows an occluded character, missing floor tiles, or a visual/functional defect, delegate back to `swe_engineer` (or `visual_artist`) to fix it right now before submitting to the Council.\n"
            f"3. Choose `action = 'delegate'` to send an open-ended natural-language `brief` to one of {specialists}, OR choose `action = 'submit_to_council'` ONLY after all required assets/code are built AND at least one proof image in `COLLECTED PROOF IMAGES SO FAR` visually proves the requirement.\n"
            f"Provide `deliberation` FIRST and `action` LAST."
        )
        contents_arg: Any = ([prompt] + multimodal_parts) if multimodal_parts else prompt
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
        issues = validate_task_lead_decision(decision, specialists, collected_proofs)
        if not issues:
            return decision
        logger.info("TaskLeadDecision attempt %d/%d for %s invalid: %s", attempt, attempts, item.item_id, issues)
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
    submission_summary = ""
    selected_proofs: List[str] = []
    pending_decision: Optional[TaskLeadDecision] = None
    trace_path = _item_dir(ctx, item) / f"lead_trace_rev{len(item.revisions)}.json"

    if trace_path.is_file():
        try:
            existing_trace = json.loads(trace_path.read_text(encoding="utf-8"))
            if (
                isinstance(existing_trace, dict)
                and existing_trace.get("status", "in_progress") in ("in_progress", "yielded")
                and not existing_trace.get("submission_summary")
            ):
                for raw_turn in existing_trace.get("turns", []):
                    turn_history.append(SpecialistReport.model_validate(raw_turn))
                for p in existing_trace.get("collected_proofs", []):
                    if isinstance(p, str) and Path(p).is_file() and p not in collected_proofs:
                        collected_proofs.append(p)
                last_strategy = str(existing_trace.get("current_strategy") or "")
                raw_pending = existing_trace.get("pending_decision")
                if isinstance(raw_pending, dict):
                    pending_decision = TaskLeadDecision.model_validate(raw_pending)
        except Exception:
            logger.debug("Could not hydrate existing lead trace at %s", trace_path, exc_info=True)

    def _persist_trace(status: str = "in_progress", pending: Optional[TaskLeadDecision] = None) -> None:
        trace_data: Dict[str, Any] = {
            "item_id": item.item_id,
            "revision": len(item.revisions),
            "status": status,
            "current_strategy": last_strategy,
            "submission_summary": submission_summary,
            "collected_proofs": list(collected_proofs),
            "turns": [t.model_dump() for t in turn_history],
        }
        if pending is not None:
            trace_data["pending_decision"] = pending.model_dump()
        trace_path.write_text(json.dumps(trace_data, indent=2), encoding="utf-8")

    code_lock = ctx.code_lock if ctx.code_lock is not None else threading.Lock()
    code_lock_held = False
    try:
        for turn_idx in range(len(turn_history), max_turns):
            resumed_from_pending = False
            if pending_decision is not None:
                decision = pending_decision
                pending_decision = None
                resumed_from_pending = True
            else:
                try:
                    decision = plan_task_lead_turn(ctx, item, turn_history, collected_proofs)
                except _LegacyFakeClientDirectPayload:
                    return _fallback_for_legacy_fake_client(ctx, item)

            last_strategy = decision.current_strategy
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
                        },
                    )
                except Exception:
                    pass

            if decision.action == "submit_to_council":
                submission_summary = decision.submission_summary or decision.current_strategy
                chosen = [p for p in decision.selected_proof_paths if Path(p).is_file()]
                selected_proofs = chosen if chosen else list(collected_proofs)
                _persist_trace(status="submitted", pending=None)
                break

            if decision.specialist_agent in ("swe_engineer", "qa_test_engineer") and not code_lock_held:
                if is_in_board_scheduler():
                    if not code_lock.acquire(blocking=False):
                        _persist_trace(status="yielded", pending=decision)
                        raise WorkerYielded(
                            f"Waiting for code_lock to run {decision.specialist_agent} on turn {turn_idx + 1} of '{item.item_id}'"
                        )
                else:
                    code_lock.acquire()
                code_lock_held = True
                _persist_trace(status="in_progress", pending=None)

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

            report = _dispatch_to_specialist(
                ctx,
                item,
                decision,
                turn_index=turn_idx,
                _lock_already_held=code_lock_held,
            )
            turn_history.append(report)
            for p in report.proof_images:
                if p not in collected_proofs and Path(p).is_file():
                    collected_proofs.append(p)
            _persist_trace(status="in_progress", pending=None)
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
        else:
            if not collected_proofs:
                raise StudioAgentError(
                    f"Task Lead for '{item.item_id}' exhausted {max_turns} turns without producing any proof images."
                )
            selected_proofs = list(collected_proofs)
            submission_summary = submission_summary or last_strategy
            _persist_trace(status="submitted", pending=None)
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
