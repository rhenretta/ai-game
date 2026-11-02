"""engine/studio/visual_evaluator.py

General-Purpose Diagnostic Plate Renderer & Multimodal Visual Evaluator (`VisualEvaluatorAgent`),
powering the perceptual inspection capabilities of the unified `visual_artist` Specialist Persona.

Provides:
1. `render_inspection_plate`: Pure observational 1:1 / nearest-neighbor diagnostic plate renderer
   with checkerboard alpha visualization, exact `WxH` pixel labels, optional pixel-coordinate grids
   (`grid_px`), and optional zoomed region callouts (`crop_boxes`). Enforces ZERO prescriptive rules.
2. `VisualEvaluatorAgent`: Native Gemini multimodal VLM inspector evaluating `.png` sprites/atlases
   and diagnostic plates against the delegated `brief`, Game Bible art direction, and optional 16:9
   concept art mockups, persisting `perceptual_description` into `assets/asset_library.json`.

Strictly adheres to Rule 1 (100% Domain-Agnostic), Rule 2 (Zero Keyword/Regex Heuristics),
Rule 4 (Mandatory Rationale-First `deliberation` + `reasoning_rationale` logging),
Rule 5 (Zero Silent Fallbacks), and Rule 6 (Container Runtime Invariance).
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from engine.studio.audio_evaluator import (
    IMAGE_MIME_BY_SUFFIX,
    INLINE_MEDIA_MAX_BYTES,
    extract_interaction_usage_dict,
)
from engine.studio.models import (
    AssetLibraryCatalog,
    DEFAULT_STUDIO_MODEL,
    StudioAgentConfigurationError,
    StudioAgentError,
    VisualEvaluationReport,
    _sanitize_gemini_json_schema,
    call_gemini_with_transient_retry,
    is_real_genai_client,
)

logger = logging.getLogger("VisualEvaluatorAgent")

DEFAULT_VISUAL_EVALUATOR_MODEL = DEFAULT_STUDIO_MODEL

_VISUAL_EVAL_SYSTEM_INSTRUCTION = (
    "You are the perceptual visual inspection engine of the Studio's `visual_artist` Specialist Persona.\n"
    "Examine every attached sprite, atlas, UI texture, diagnostic 1:1 contact-sheet plate, and optional "
    "16:9 concept art mockup carefully.\n"
    "Ground every observation strictly in what is actually visible in the rendered pixels:\n"
    "1. Deliberate first (`deliberation`) on silhouette clarity, alpha/transparency cleanliness (note checkerboard "
    "regions vs opaque backdrop pixels), pixel dimensions, orientation, multi-element or frame layout, and cohesion "
    "with the game's art direction / concept mockup.\n"
    "2. Report concrete pixel-level facts in `observed_visual_properties` and `dimensional_and_framing_notes`.\n"
    "3. Populate `perceptual_descriptions_by_path` mapping each inspected relative asset path (e.g. "
    "`assets/sprites/actor.png`) to a concise, accurate description of what it visually contains so downstream "
    "engineers and QA know exact visual contents, sub-elements, and framing.\n"
    "4. If pixel inspection proves the `.png` asset on disk is already clean/intact and the reported defect actually "
    "lies in another Specialist Persona's domain (e.g. `'swe_engineer'` for GDScript `AtlasTexture` region slicing, "
    "`hframes`/`vframes`, or UI layout), populate `reroute_to_specialist`, `reroute_brief`, and optional `reroute_target_paths`.\n"
    "5. Set `answers_brief_satisfactorily` strictly last."
)


def _draw_checkerboard(
    canvas: Image.Image,
    x0: int,
    y0: int,
    w: int,
    h: int,
    cell: int = 12,
    c1: Tuple[int, int, int, int] = (46, 50, 60, 255),
    c2: Tuple[int, int, int, int] = (68, 74, 88, 255),
) -> None:
    """Draws an alpha-inspection checkerboard inside `(x0, y0, x0+w, y0+h)`."""
    draw = ImageDraw.Draw(canvas)
    for ry in range(0, max(1, h), cell):
        for rx in range(0, max(1, w), cell):
            color = c1 if ((rx // cell) + (ry // cell)) % 2 == 0 else c2
            draw.rectangle(
                [x0 + rx, y0 + ry, min(x0 + w - 1, x0 + rx + cell - 1), min(y0 + h - 1, y0 + ry + cell - 1)],
                fill=color,
            )


def render_inspection_plate(
    cartridge_dir: Union[Path, str],
    asset_paths: Optional[Sequence[Union[Path, str]]] = None,
    *,
    output_path: Optional[Union[Path, str]] = None,
    grid_px: int = 0,
    crop_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    max_assets: int = 24,
) -> Path:
    """Renders a deterministic diagnostic visual contact sheet over selected (or all `assets/**/*.png`)
    files in `cartridge_dir`.

    - Displays each PNG over an alpha checkerboard so transparent vs opaque pixels are unmistakable.
    - Prints the relative path and exact `WxH` pixel dimensions on each card.
    - Optionally overlays coordinate grid lines every `grid_px` pixels (`grid_px > 0`).
    - Optionally appends zoomed sub-region callout cards (`crop_boxes=[(x, y, w, h), ...]`) from the first target asset.
    """
    cdir = Path(cartridge_dir).resolve()
    resolved_files: List[Tuple[str, Path]] = []

    if asset_paths:
        for raw in asset_paths:
            s = str(raw or "").strip()
            if not s:
                continue
            rel = s[len("res://") :] if s.startswith("res://") else s
            cand = Path(rel) if Path(rel).is_absolute() else (cdir / rel.lstrip("/"))
            if cand.is_file() and cand.suffix.lower() in IMAGE_MIME_BY_SUFFIX:
                try:
                    rel_label = cand.resolve().relative_to(cdir).as_posix()
                except ValueError:
                    rel_label = cand.name
                resolved_files.append((rel_label, cand.resolve()))
    else:
        assets_root = cdir / "assets"
        if assets_root.is_dir():
            for png in sorted(assets_root.rglob("*.png")):
                rel_label = png.relative_to(cdir).as_posix()
                if rel_label.startswith("assets/mockups/"):
                    continue
                resolved_files.append((rel_label, png.resolve()))
                if len(resolved_files) >= max_assets:
                    break

    if not resolved_files:
        raise StudioAgentError(
            f"render_inspection_plate found no readable image assets to render in {cdir} (requested={list(asset_paths or [])})."
        )

    cards: List[Tuple[str, str, Image.Image]] = []
    for rel_label, fpath in resolved_files[:max_assets]:
        try:
            with Image.open(fpath) as im:
                rgba = im.convert("RGBA")
        except Exception as exc:
            raise StudioAgentError(f"Failed to decode image '{rel_label}' ({fpath}): {exc}") from exc
        cards.append((rel_label, f"{rgba.width}x{rgba.height}px", rgba))

    if crop_boxes and cards:
        base_label, _, base_img = cards[0]
        for idx, box in enumerate(crop_boxes):
            if len(box) != 4:
                continue
            bx, by, bw, bh = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            if bw <= 0 or bh <= 0:
                continue
            x0 = max(0, min(base_img.width - 1, bx))
            y0 = max(0, min(base_img.height - 1, by))
            x1 = max(x0 + 1, min(base_img.width, bx + bw))
            y1 = max(y0 + 1, min(base_img.height, by + bh))
            cropped = base_img.crop((x0, y0, x1, y1))
            cards.append(
                (
                    f"CROP[{idx}] {Path(base_label).name}",
                    f"Rect2({x0},{y0},{x1 - x0},{y1 - y0})",
                    cropped,
                )
            )

    cols = min(4, max(1, int(math.ceil(math.sqrt(len(cards))))))
    rows = int(math.ceil(len(cards) / cols))
    thumb_box = 280
    header_h = 46
    pad = 18
    card_w = thumb_box + 24
    card_h = thumb_box + header_h + 24
    title_h = 44

    sheet_w = cols * card_w + (cols + 1) * pad
    sheet_h = title_h + rows * card_h + (rows + 1) * pad
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (20, 23, 30, 255))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    grid_note = f" | grid={grid_px}px" if grid_px and grid_px > 0 else ""
    draw.text(
        (pad, 14),
        f"VISUAL DIAGNOSTIC PLATE ({len(cards)} view(s){grid_note})",
        fill=(230, 236, 248, 255),
        font=font,
    )

    for idx, (label, dim_txt, rgba) in enumerate(cards):
        col = idx % cols
        row = idx // cols
        cx = pad + col * (card_w + pad)
        cy = title_h + pad + row * (card_h + pad)

        draw.rectangle([cx, cy, cx + card_w - 1, cy + card_h - 1], fill=(30, 34, 44, 255), outline=(82, 92, 114, 255))
        short_lbl = label if len(label) <= 38 else ("..." + label[-35:])
        draw.text((cx + 8, cy + 6), short_lbl, fill=(235, 240, 250, 255), font=font)
        draw.text((cx + 8, cy + 22), dim_txt, fill=(156, 204, 255, 255), font=font)

        view_x = cx + 12
        view_y = cy + header_h
        _draw_checkerboard(sheet, view_x, view_y, thumb_box, thumb_box)

        w, h = rgba.size
        if w <= 0 or h <= 0:
            continue
        scale = min(thumb_box / float(w), thumb_box / float(h))
        # Preserve crisp nearest-neighbor pixel edges when upscaling small sprites or inspecting grids
        disp_w = max(1, int(round(w * scale)))
        disp_h = max(1, int(round(h * scale)))
        resample_mode = Image.Resampling.NEAREST if scale >= 1.0 else Image.Resampling.LANCZOS
        resized = rgba.resize((disp_w, disp_h), resample=resample_mode)
        ox = view_x + (thumb_box - disp_w) // 2
        oy = view_y + (thumb_box - disp_h) // 2
        sheet.alpha_composite(resized, dest=(ox, oy))

        if grid_px and grid_px > 0:
            gx_step = grid_px * scale
            if gx_step >= 6.0:
                px_x = grid_px
                while px_x < w:
                    lx = int(round(ox + px_x * scale))
                    draw.line([(lx, oy), (lx, oy + disp_h - 1)], fill=(0, 230, 255, 150), width=1)
                    px_x += grid_px
                px_y = grid_px
                while px_y < h:
                    ly = int(round(oy + px_y * scale))
                    draw.line([(ox, ly), (ox + disp_w - 1, ly)], fill=(0, 230, 255, 150), width=1)
                    px_y += grid_px

        draw.rectangle([ox, oy, ox + disp_w - 1, oy + disp_h - 1], outline=(120, 140, 175, 180), width=1)

    if output_path is not None:
        out_p = Path(output_path)
        if not out_p.is_absolute():
            out_p = cdir / str(output_path).replace("res://", "").lstrip("/")
    else:
        out_p = cdir / "proofs" / "visual_inspection_plate.png"

    out_p.parent.mkdir(parents=True, exist_ok=True)
    sheet.convert("RGB").save(out_p, format="PNG")
    return out_p.resolve()


def enrich_catalog_visual_entries(
    cartridge_dir: Path,
    inspected_paths: Sequence[Path],
    report: VisualEvaluationReport,
) -> List[str]:
    """Updates `perceptual_description` on matching entries in `assets/asset_library.json`."""
    catalog_path = cartridge_dir / "assets" / "asset_library.json"
    if not catalog_path.is_file():
        return []
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog = AssetLibraryCatalog.model_validate(raw)
    except Exception as exc:
        logger.warning("VisualEvaluatorAgent could not parse asset_library.json for enrichment: %s", exc)
        return []

    by_rel_desc: Dict[str, str] = {}
    for k, v in (report.perceptual_descriptions_by_path or {}).items():
        norm_k = str(k or "").replace("res://", "").lstrip("/")
        if norm_k and str(v or "").strip():
            by_rel_desc[norm_k] = str(v).strip()

    inspected_rels: List[str] = []
    for p in inspected_paths:
        try:
            inspected_rels.append(p.resolve().relative_to(cartridge_dir.resolve()).as_posix())
        except ValueError:
            continue

    updated_ids: List[str] = []
    fallback_desc = str(report.observed_visual_properties or "").strip()
    for aid, entry in catalog.entries.items():
        fp = str(getattr(entry, "file_path", "") or "").replace("res://", "").lstrip("/")
        if not fp:
            continue
        desc = by_rel_desc.get(fp, "")
        if not desc and len(inspected_rels) == 1 and fp == inspected_rels[0]:
            desc = fallback_desc
        if desc:
            entry.perceptual_description = desc
            updated_ids.append(aid)

    if updated_ids:
        catalog.updated_at_utc = datetime.now(timezone.utc).isoformat()
        catalog_path.write_text(json.dumps(catalog.to_dict(), indent=2, default=str), encoding="utf-8")
    return updated_ids


class VisualEvaluatorAgent:
    """Native VLM perceptual visual evaluator invoked internally by the `visual_artist` Specialist Persona."""

    def __init__(
        self,
        *,
        client: Any,
        model_name: str = DEFAULT_VISUAL_EVALUATOR_MODEL,
        context_curator: Optional[Any] = None,
        trace_logger: Optional[Any] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        inline_max_bytes: int = INLINE_MEDIA_MAX_BYTES,
    ) -> None:
        self.client = client
        self.model_name = (model_name or DEFAULT_VISUAL_EVALUATOR_MODEL).strip()
        self.context_curator = context_curator
        self.trace_logger = trace_logger
        self.event_callback = event_callback
        self.inline_max_bytes = int(inline_max_bytes)
        self.last_usage: Dict[str, int] = {}

    def _find_slice_concept_mockup(self, cartridge_dir: Path, game_bible: Any = None) -> Optional[Path]:
        if game_bible is not None:
            mockups = getattr(game_bible, "visual_mockups", None) or []
            for m in reversed(list(mockups)):
                mp = str(getattr(m, "image_path", "") or (m.get("image_path") if isinstance(m, dict) else "") or "").strip()
                if mp:
                    cand = Path(mp) if Path(mp).is_absolute() else (cartridge_dir / mp.replace("res://", "").lstrip("/"))
                    if cand.is_file() and cand.stat().st_size > 0:
                        return cand.resolve()
        mockups_dir = cartridge_dir / "assets" / "mockups"
        if mockups_dir.is_dir():
            pngs = sorted(mockups_dir.glob("*.png"))
            if pngs:
                return pngs[-1].resolve()
        return None

    def evaluate(
        self,
        *,
        cartridge_dir: Union[Path, str],
        asset_paths: Sequence[Union[Path, str]],
        inquiry_prompt: str,
        item_id: str = "",
        item_kind: str = "",
        grid_px: int = 0,
        crop_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        contact_sheet_output_path: Optional[Union[Path, str]] = None,
        include_concept_mockup: bool = True,
        extra_studio_context: str = "",
        board: Any = None,
        game_bible: Any = None,
        enrich_catalog: bool = True,
        token_tracker: Optional[Any] = None,
    ) -> Tuple[VisualEvaluationReport, Path, Dict[str, int]]:
        """Renders a 1:1 diagnostic inspection plate, performs native VLM inspection, enriches
        `assets/asset_library.json`, and returns `(VisualEvaluationReport, plate_path, usage_dict)`."""
        if self.client is None:
            raise StudioAgentConfigurationError(
                "VisualEvaluatorAgent requires an active Gemini client (`client` is None; Rule 5)."
            )
        if not inquiry_prompt or not inquiry_prompt.strip():
            raise StudioAgentError("VisualEvaluatorAgent.evaluate requires a non-empty `inquiry_prompt`.")

        cdir = Path(cartridge_dir).resolve()
        primary_paths: List[Path] = []
        for raw in asset_paths:
            s = str(raw or "").strip()
            if not s:
                continue
            rel = s[len("res://") :] if s.startswith("res://") else s
            cand = Path(rel) if Path(rel).is_absolute() else (cdir / rel.lstrip("/"))
            if not cand.is_file() or cand.stat().st_size <= 0:
                raise StudioAgentError(f"VisualEvaluatorAgent target image does not exist or is empty: {cand}")
            primary_paths.append(cand.resolve())

        if not primary_paths:
            raise StudioAgentError("VisualEvaluatorAgent.evaluate requires at least one valid image path.")

        plate_path = render_inspection_plate(
            cdir,
            primary_paths,
            output_path=contact_sheet_output_path
            or (f"proofs/{item_id}_visual_inspection_plate.png" if item_id else "proofs/visual_inspection_plate.png"),
            grid_px=grid_px,
            crop_boxes=crop_boxes,
        )

        curated_prompt_block = ""
        curator = self.context_curator
        if curator is None and is_real_genai_client(self.client):
            from engine.studio.context_curator import ContextCuratorAgent

            curator = ContextCuratorAgent(
                cartridge_dir=cdir,
                client=self.client,
                model_name=self.model_name,
                board=board,
                game_bible=game_bible,
                event_callback=self.event_callback,
            )
        if curator is not None:
            pack = curator.curate_context(
                target_agent_role="visual_artist",
                task_request=inquiry_prompt,
                item_id=item_id,
                item_kind=item_kind,
                extra_image_paths=[str(plate_path)],
                max_selected_images=2,
                reuse_revision_cache=False,
            )
            curated_prompt_block = str(getattr(pack, "prompt_block", "") or "").strip()

        sections: List[str] = []
        if curated_prompt_block:
            sections.append(curated_prompt_block)
        if extra_studio_context and extra_studio_context.strip():
            sections.append(f"=== CALLER STUDIO CONTEXT ===\n{extra_studio_context.strip()}")
        rel_names: List[str] = []
        for p in primary_paths:
            try:
                rel_names.append(p.relative_to(cdir).as_posix())
            except ValueError:
                rel_names.append(p.name)
        sections.append(
            f"=== VISUAL INSPECTION INQUIRY ===\n{inquiry_prompt.strip()}\n\n"
            f"Inspected asset paths ({len(rel_names)}): {', '.join(rel_names)}\n"
            f"Diagnostic 1:1 contact-sheet plate: {plate_path.name}"
        )
        prompt_text = "\n\n".join(sections)

        mockup_path = self._find_slice_concept_mockup(cdir, game_bible=game_bible) if include_concept_mockup else None

        interactions_api = getattr(self.client, "interactions", None)
        models_api = getattr(self.client, "models", None)

        usage_dict: Dict[str, int] = {}
        if interactions_api is not None and callable(getattr(interactions_api, "create", None)):
            input_items: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

            def _attach_img(img_p: Path, label: str) -> None:
                mime = IMAGE_MIME_BY_SUFFIX.get(img_p.suffix.lower(), "image/png")
                raw_bytes = img_p.read_bytes()
                if len(raw_bytes) <= self.inline_max_bytes:
                    input_items.append({"type": "text", "text": f"{label}: {img_p.name} ({len(raw_bytes)} bytes)"})
                    input_items.append(
                        {"type": "image", "data": base64.b64encode(raw_bytes).decode("ascii"), "mime_type": mime}
                    )

            _attach_img(plate_path, "DIAGNOSTIC 1:1 INSPECTION PLATE")
            for p in primary_paths[:6]:
                _attach_img(p, "RAW TARGET ASSET PNG")
            if mockup_path is not None and mockup_path != plate_path and mockup_path not in primary_paths:
                _attach_img(mockup_path, "SLICE 16:9 CONCEPT ART MOCKUP REFERENCE")

            sanitized_schema = _sanitize_gemini_json_schema(VisualEvaluationReport.model_json_schema())
            call_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "system_instruction": _VISUAL_EVAL_SYSTEM_INSTRUCTION,
                "input": input_items,
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": sanitized_schema,
                },
                "generation_config": {
                    "temperature": 0.2,
                    "thinking_level": "low",
                },
            }
            interaction = call_gemini_with_transient_retry(
                lambda: interactions_api.create(**call_kwargs),
                label=f"visual_evaluator[{item_id or 'studio'}]",
                candidate_dir=cdir,
            )
            if token_tracker is not None and hasattr(token_tracker, "record_response"):
                token_tracker.record_response(interaction)
            usage_dict = extract_interaction_usage_dict(interaction)
            raw_text = getattr(interaction, "output_text", None)
            if not raw_text or not str(raw_text).strip():
                for out_part in reversed(list(getattr(interaction, "outputs", None) or [])):
                    p_txt = out_part.get("text") if isinstance(out_part, dict) else getattr(out_part, "text", None)
                    if p_txt and str(p_txt).strip():
                        raw_text = str(p_txt)
                        break
            if not raw_text or not str(raw_text).strip():
                raise StudioAgentError(
                    f"VisualEvaluatorAgent received empty `output_text` from Gemini Interactions API (model={self.model_name})."
                )
            report = VisualEvaluationReport.model_validate_json(str(raw_text))
        elif models_api is not None and callable(getattr(models_api, "generate_content", None)):
            from google.genai import types

            contents: List[Any] = [prompt_text]
            for label, img_p in (
                [("DIAGNOSTIC 1:1 INSPECTION PLATE", plate_path)]
                + [("RAW TARGET ASSET PNG", p) for p in primary_paths[:6]]
                + ([("SLICE 16:9 CONCEPT ART MOCKUP REFERENCE", mockup_path)] if mockup_path is not None else [])
            ):
                mime = IMAGE_MIME_BY_SUFFIX.get(img_p.suffix.lower(), "image/png")
                contents.append(f"{label}: {img_p.name}")
                contents.append(types.Part.from_bytes(data=img_p.read_bytes(), mime_type=mime))

            resp = call_gemini_with_transient_retry(
                lambda: models_api.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=_VISUAL_EVAL_SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=VisualEvaluationReport,
                        temperature=0.2,
                    ),
                ),
                label=f"visual_evaluator[{item_id or 'studio'}]",
                candidate_dir=cdir,
            )
            if token_tracker is not None and hasattr(token_tracker, "record_response"):
                token_tracker.record_response(resp)
            parsed = getattr(resp, "parsed", None)
            if isinstance(parsed, VisualEvaluationReport):
                report = parsed
            else:
                raw_txt = str(getattr(resp, "text", "") or "").strip()
                if not raw_txt:
                    raise StudioAgentError("VisualEvaluatorAgent received empty response from generate_content.")
                report = VisualEvaluationReport.model_validate_json(raw_txt)
        else:
            raise StudioAgentConfigurationError(
                f"Client {type(self.client).__name__} exposes neither `interactions.create` nor `models.generate_content`."
            )

        self.last_usage = dict(usage_dict)
        logger.info("reasoning_rationale[visual_artist_evaluator]: %s", report.deliberation)
        telemetry_payload = {
            "deliberation": report.deliberation,
            "reasoning_rationale": report.deliberation,
            "item_id": item_id,
            "inspected_assets": rel_names,
            "plate_path": str(plate_path),
            "answers_brief_satisfactorily": report.answers_brief_satisfactorily,
            "usage": dict(usage_dict),
        }
        if self.trace_logger is not None:
            log_fn = getattr(self.trace_logger, "log_event", None) or getattr(self.trace_logger, "record", None)
            if callable(log_fn):
                try:
                    log_fn("visual_evaluation_completed", telemetry_payload)
                except Exception:
                    pass
        if self.event_callback is not None:
            try:
                self.event_callback("visual_evaluation_completed", telemetry_payload)
            except Exception:
                pass

        if enrich_catalog:
            enrich_catalog_visual_entries(cdir, primary_paths, report)

        return report, plate_path, usage_dict
