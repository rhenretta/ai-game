"""engine/studio/audio_evaluator.py
General-Purpose Multimodal Audio & Audiovisual Evaluator Agent (`AudioEvaluatorAgent`).

Uses Gemini's native audio (`audio/wav`, etc.) and synchronized audiovisual (`video/mp4` with audio track)
understanding via the current `google-genai` Interactions API (`client.interactions.create`) to provide
rich perceptual descriptions, contextual observations, defect detection, and open-ended answers to
strategist-authored inquiries.

Supports two strict execution modes:
1. Studio Mode (`curate_studio_context=True`): Semantically curates relevant Game Bible audio direction,
   GDScript trigger bindings, catalog metadata, and reference media via `ContextCuratorAgent`, and can
   enrich `assets/asset_library.json` with `perceptual_description` and `duration_seconds`.
2. Opaque-Box Mode (`curate_studio_context=False`): Strictly isolated from `ContextCuratorAgent`,
   `game_bible.json`, and cartridge source files—receiving only caller-supplied `opaque_context_text`
   (e.g., `BlindStoreCard` title + premise) and the captured gameplay media slice.

Strictly adheres to Rule 1 (100% Domain-Agnostic), Rule 2 (Zero Keyword/Regex Epistemic Heuristics),
Rule 4 (Mandatory Rationale-First `deliberation` + `reasoning_rationale` logging), Rule 5 (Zero Silent
Fallbacks), and Rule 6 (Container Runtime Invariance).
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from engine.studio.audio import validate_wav_file
from engine.studio.models import (
    AssetLibraryCatalog,
    AudioEvaluationReport,
    DEFAULT_STUDIO_MODEL,
    StudioAgentConfigurationError,
    StudioAgentError,
    _sanitize_gemini_json_schema,
    call_gemini_with_transient_retry,
    is_real_genai_client,
)

logger = logging.getLogger("AudioEvaluatorAgent")

DEFAULT_AUDIO_EVALUATOR_MODEL = DEFAULT_STUDIO_MODEL  # "gemini-3.8-flash"
INLINE_MEDIA_MAX_BYTES = 18 * 1024 * 1024  # 18 MB safe inline cap (< 20 MB total request limit)

SUPPORTED_MEDIA_TYPES: Dict[str, Tuple[str, str]] = {
    ".wav": ("audio", "audio/wav"),
    ".mp3": ("audio", "audio/mp3"),
    ".ogg": ("audio", "audio/ogg"),
    ".flac": ("audio", "audio/flac"),
    ".aac": ("audio", "audio/aac"),
    ".opus": ("audio", "audio/opus"),
    ".m4a": ("audio", "audio/mp4"),
    ".mp4": ("video", "video/mp4"),
    ".webm": ("video", "video/webm"),
    ".mov": ("video", "video/mov"),
    ".png": ("image", "image/png"),
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".webp": ("image", "image/webp"),
}

AUDIO_SUFFIXES = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".opus", ".m4a"}
AUDIO_MIME_BY_SUFFIX: Dict[str, str] = {
    ext: mime for ext, (mtype, mime) in SUPPORTED_MEDIA_TYPES.items() if mtype == "audio"
}
VIDEO_MIME_BY_SUFFIX: Dict[str, str] = {
    ext: mime for ext, (mtype, mime) in SUPPORTED_MEDIA_TYPES.items() if mtype == "video"
}
IMAGE_MIME_BY_SUFFIX: Dict[str, str] = {
    ext: mime for ext, (mtype, mime) in SUPPORTED_MEDIA_TYPES.items() if mtype == "image"
}

_SYSTEM_INSTRUCTION = (
    "You are a perceptual Audio & Synchronized Audiovisual Evaluator.\n"
    "Listen carefully to every attached audio clip and/or watch+listen to every synchronized video+audio "
    "gameplay recording (alongside any attached visual reference images).\n"
    "Ground every observation strictly in what is actually audible and visible in the provided media:\n"
    "- Describe acoustic textures, timbre, pitch contour, envelope/transients, rhythm, layering, prosody, "
    "spatial/mix balance, and key timestamped moments in clear natural language.\n"
    "- When evaluating synchronized video+audio clips, pay close attention to temporal alignment between "
    "visual events/actions and audible cues, missing expected sound feedback, stray or looping sounds, "
    "clipping/clicks, silence, and cohesive atmosphere.\n"
    "- Answer the caller's specific inquiry directly, candidly, and thoroughly without inventing sounds "
    "that are not present in the signal.\n"
    "- Populate `deliberation` FIRST before filling out any other schema fields."
)


def classify_media_path(path: Union[Path, str]) -> Tuple[str, str]:
    """Returns `(interaction_content_type, mime_type)` for a supported media file or raises `StudioAgentError`."""
    p = Path(path)
    ext = p.suffix.lower()
    spec = SUPPORTED_MEDIA_TYPES.get(ext)
    if spec is None:
        raise StudioAgentError(
            f"Unsupported media file extension '{ext}' for AudioEvaluatorAgent: {p} "
            f"(supported: {', '.join(sorted(SUPPORTED_MEDIA_TYPES.keys()))})."
        )
    return spec


def extract_interaction_usage_dict(interaction: Any) -> Dict[str, int]:
    """Extracts normalized token usage counts from a `google.genai.interactions.Interaction` response."""
    usage = getattr(interaction, "usage", None)
    if usage is None:
        usage_meta = getattr(interaction, "usage_metadata", None)
        if usage_meta is not None:
            p_tok = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
            c_tok = int(getattr(usage_meta, "cached_content_token_count", 0) or 0)
            o_tok = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
            t_tok = int(getattr(usage_meta, "thoughts_token_count", 0) or 0)
            tot = int(getattr(usage_meta, "total_token_count", 0) or (p_tok + o_tok + t_tok))
            return {
                "prompt_tokens": p_tok,
                "cached_tokens": c_tok,
                "output_tokens": o_tok,
                "thought_tokens": t_tok,
                "total_tokens": tot,
            }
        return {
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "thought_tokens": 0,
            "total_tokens": 0,
        }

    prompt_tok = int(
        getattr(usage, "total_input_tokens", None)
        or getattr(usage, "input_tokens", 0)
        or 0
    )
    cached_tok = int(
        getattr(usage, "total_cached_tokens", None)
        or getattr(usage, "cached_tokens", 0)
        or 0
    )
    output_tok = int(
        getattr(usage, "total_output_tokens", None)
        or getattr(usage, "output_tokens", 0)
        or 0
    )
    thought_tok = int(
        getattr(usage, "total_thought_tokens", None)
        or getattr(usage, "thought_tokens", 0)
        or 0
    )
    total_tok = int(
        getattr(usage, "total_tokens", 0)
        or (prompt_tok + output_tok + thought_tok)
    )
    return {
        "prompt_tokens": prompt_tok,
        "cached_tokens": cached_tok,
        "output_tokens": output_tok,
        "thought_tokens": thought_tok,
        "total_tokens": total_tok,
    }


def enrich_catalog_audio_entries(
    cartridge_dir: Union[Path, str],
    media_paths: Sequence[Union[Path, str]],
    report: AudioEvaluationReport,
) -> List[str]:
    """Updates `perceptual_description` and `duration_seconds` in `<cartridge_dir>/assets/asset_library.json`
    for any evaluated audio files residing inside `cartridge_dir`. Returns updated `asset_id`s."""
    cdir = Path(cartridge_dir).resolve()
    catalog_path = cdir / "assets" / "asset_library.json"
    if not catalog_path.is_file():
        return []

    target_rel_to_abs: Dict[str, Path] = {}
    for raw_p in media_paths:
        p = Path(raw_p)
        abs_p = (p if p.is_absolute() else (cdir / p)).resolve()
        if not abs_p.is_file() or abs_p.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        try:
            rel_posix = abs_p.relative_to(cdir).as_posix().lstrip("/")
            target_rel_to_abs[rel_posix] = abs_p
        except ValueError:
            continue

    if not target_rel_to_abs:
        return []

    try:
        raw_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog = AssetLibraryCatalog.model_validate(raw_catalog)
    except Exception as exc:
        logger.warning("Could not parse %s during audio catalog enrichment: %s", catalog_path, exc)
        return []

    updated_ids: List[str] = []
    audio_file_count = len(target_rel_to_abs)
    desc_text = report.rich_acoustic_description.strip()
    if audio_file_count > 1:
        desc_text = f"[Multi-cue evaluation] {desc_text}"

    for aid, entry in catalog.entries.items():
        entry_rel = (
            str(entry.file_path or entry.res_path or "")
            .replace("res://", "")
            .lstrip("/")
        )
        matched_abs = target_rel_to_abs.get(entry_rel)
        if matched_abs is None:
            continue
        entry.perceptual_description = desc_text
        if matched_abs.suffix.lower() == ".wav":
            ok_wav, _err_wav, dur = validate_wav_file(matched_abs)
            if ok_wav and dur > 0:
                entry.duration_seconds = round(float(dur), 3)
        updated_ids.append(aid)

    if updated_ids:
        catalog.updated_at_utc = datetime.now(timezone.utc).isoformat()
        tmp_path = catalog_path.with_suffix(".json.tmp")
        tmp_path.write_text(catalog.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(catalog_path)

    return updated_ids


class AudioEvaluatorAgent:
    """Multimodal Audio & Gameplay Soundscape Evaluator powered by Gemini Interactions API."""

    deliberation: str = "Perceptual audio and audiovisual evaluation agent using Gemini Interactions API."

    def __init__(
        self,
        client: Any,
        model_name: str = DEFAULT_AUDIO_EVALUATOR_MODEL,
        inline_max_bytes: int = INLINE_MEDIA_MAX_BYTES,
        context_curator: Optional[Any] = None,
        trace_logger: Optional[Any] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.client = client
        self.model_name = model_name or DEFAULT_AUDIO_EVALUATOR_MODEL
        self.inline_max_bytes = max(1024, int(inline_max_bytes))
        self.context_curator = context_curator
        self.trace_logger = trace_logger
        self.event_callback = event_callback
        self.last_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "thought_tokens": 0,
            "total_tokens": 0,
        }

    def _resolve_and_validate_media(
        self,
        media_paths: Sequence[Union[Path, str]],
        *,
        cartridge_dir: Optional[Path],
        allow_cartridge_relative: bool,
    ) -> List[Path]:
        if not media_paths:
            raise StudioAgentError(
                "AudioEvaluatorAgent.evaluate requires at least one media file path (.wav, .mp4, etc.)."
            )
        resolved: List[Path] = []
        seen: set[Path] = set()
        for raw in media_paths:
            if raw is None or not str(raw).strip():
                raise StudioAgentError("AudioEvaluatorAgent received an empty media path entry.")
            cand = Path(raw)
            if not cand.is_absolute() and allow_cartridge_relative and cartridge_dir is not None:
                cand = cartridge_dir / cand
            if not cand.is_file():
                raise StudioAgentError(
                    f"AudioEvaluatorAgent media file does not exist on disk: {cand}"
                )
            size_bytes = cand.stat().st_size
            if size_bytes <= 0:
                raise StudioAgentError(
                    f"AudioEvaluatorAgent media file is empty (0 bytes): {cand}"
                )
            classify_media_path(cand)
            if cand.suffix.lower() == ".wav":
                ok_wav, wav_err, _dur = validate_wav_file(cand)
                if not ok_wav:
                    raise StudioAgentError(
                        f"AudioEvaluatorAgent received invalid WAV file '{cand.name}': {wav_err}"
                    )
            abs_cand = cand.resolve()
            if abs_cand not in seen:
                seen.add(abs_cand)
                resolved.append(abs_cand)
        return resolved

    def _upload_large_media(self, path: Path, mime_type: str) -> Tuple[str, Optional[str]]:
        """Uploads media exceeding the inline byte threshold via `client.files.upload` and waits until ACTIVE."""
        files_api = getattr(self.client, "files", None)
        upload_fn = getattr(files_api, "upload", None) if files_api is not None else None
        if not callable(upload_fn):
            raise StudioAgentError(
                f"Media file '{path.name}' ({path.stat().st_size} bytes) exceeds inline cap "
                f"({self.inline_max_bytes} bytes), and `client.files.upload` is unavailable."
            )
        uploaded = upload_fn(file=str(path), config={"mime_type": mime_type})
        file_name = getattr(uploaded, "name", None)
        state_obj = getattr(uploaded, "state", None)
        state_str = str(getattr(state_obj, "name", state_obj) or "").upper()
        get_fn = getattr(files_api, "get", None)
        deadline = time.monotonic() + 90.0
        while "PROCESSING" in state_str:
            if not callable(get_fn):
                break
            if time.monotonic() > deadline:
                raise StudioAgentError(
                    f"Timed out waiting for uploaded media '{path.name}' ({file_name}) to reach ACTIVE state."
                )
            time.sleep(1.5)
            uploaded = get_fn(name=file_name)
            state_obj = getattr(uploaded, "state", None)
            state_str = str(getattr(state_obj, "name", state_obj) or "").upper()

        if "FAILED" in state_str:
            raise StudioAgentError(
                f"Gemini Files API processing failed for media '{path.name}' (state={state_str})."
            )
        uri = getattr(uploaded, "uri", None)
        if not uri:
            raise StudioAgentError(
                f"Gemini Files API upload for '{path.name}' did not return a valid file URI."
            )
        return str(uri), (str(file_name) if file_name else None)

    def evaluate(
        self,
        *,
        inquiry_prompt: str,
        media_paths: Sequence[Union[Path, str]],
        curate_studio_context: bool = True,
        cartridge_dir: Optional[Union[Path, str]] = None,
        item_id: str = "",
        item_kind: str = "",
        opaque_context_text: str = "",
        extra_studio_context: str = "",
        model_name: Optional[str] = None,
        board: Any = None,
        game_bible: Any = None,
        enrich_catalog: bool = True,
        token_tracker: Optional[Any] = None,
    ) -> AudioEvaluationReport:
        """Evaluates one or more audio/video/image files against an open-ended `inquiry_prompt`
        and returns a Rationale-First `AudioEvaluationReport` (storing token usage in `self.last_usage`)."""
        report, _usage = self.evaluate_with_usage(
            inquiry_prompt=inquiry_prompt,
            media_paths=media_paths,
            curate_studio_context=curate_studio_context,
            cartridge_dir=cartridge_dir,
            item_id=item_id,
            item_kind=item_kind,
            opaque_context_text=opaque_context_text,
            extra_studio_context=extra_studio_context,
            model_name=model_name,
            board=board,
            game_bible=game_bible,
            enrich_catalog=enrich_catalog,
            token_tracker=token_tracker,
        )
        return report

    def evaluate_with_usage(
        self,
        *,
        inquiry_prompt: str,
        media_paths: Sequence[Union[Path, str]],
        curate_studio_context: bool = True,
        cartridge_dir: Optional[Union[Path, str]] = None,
        item_id: str = "",
        item_kind: str = "",
        opaque_context_text: str = "",
        extra_studio_context: str = "",
        model_name: Optional[str] = None,
        board: Any = None,
        game_bible: Any = None,
        enrich_catalog: bool = True,
        token_tracker: Optional[Any] = None,
    ) -> Tuple[AudioEvaluationReport, Dict[str, int]]:
        """Evaluates one or more audio/video/image files against an open-ended `inquiry_prompt`
        and returns `(AudioEvaluationReport, usage_dict)`."""
        if self.client is None:
            raise StudioAgentConfigurationError(
                "AudioEvaluatorAgent requires an active Gemini client (`client` is None; Rule 5)."
            )
        interactions_api = getattr(self.client, "interactions", None)
        if interactions_api is None or not callable(getattr(interactions_api, "create", None)):
            raise StudioAgentConfigurationError(
                f"Client {type(self.client).__name__} does not expose `interactions.create` required by AudioEvaluatorAgent."
            )
        if not inquiry_prompt or not inquiry_prompt.strip():
            raise StudioAgentError("AudioEvaluatorAgent.evaluate requires a non-empty `inquiry_prompt`.")

        eff_model = (model_name or self.model_name or DEFAULT_AUDIO_EVALUATOR_MODEL).strip()
        studio_cdir = Path(cartridge_dir).resolve() if (curate_studio_context and cartridge_dir is not None) else None

        primary_paths = self._resolve_and_validate_media(
            media_paths,
            cartridge_dir=studio_cdir,
            allow_cartridge_relative=bool(curate_studio_context and studio_cdir is not None),
        )

        curated_prompt_block = ""
        curated_reference_paths: List[Path] = []
        if curate_studio_context and studio_cdir is not None:
            curator = self.context_curator
            if curator is None and is_real_genai_client(self.client):
                from engine.studio.context_curator import ContextCuratorAgent

                curator = ContextCuratorAgent(
                    cartridge_dir=studio_cdir,
                    client=self.client,
                    model_name=eff_model,
                    board=board,
                    game_bible=game_bible,
                    event_callback=self.event_callback,
                )
            if curator is not None:
                pack = curator.curate_context(
                    target_agent_role="audio_evaluator",
                    task_request=inquiry_prompt,
                    item_id=item_id,
                    item_kind=item_kind,
                    extra_image_paths=[str(p) for p in primary_paths],
                    max_selected_images=3,
                    reuse_revision_cache=False,
                )
                curated_prompt_block = str(getattr(pack, "prompt_block", "") or "").strip()
                raw_curated_media: List[str] = list(getattr(pack, "selected_media_paths", None) or [])
                if not raw_curated_media and getattr(pack, "decision", None) is not None:
                    dec = pack.decision
                    raw_curated_media = list(
                        getattr(dec, "selected_media_paths", None)
                        or getattr(dec, "selected_image_paths", None)
                        or []
                    )
                primary_set = set(primary_paths)
                for raw_cm in raw_curated_media:
                    cm_path = Path(raw_cm) if Path(raw_cm).is_absolute() else (studio_cdir / raw_cm)
                    if not cm_path.is_file() or cm_path.stat().st_size <= 0:
                        continue
                    abs_cm = cm_path.resolve()
                    if abs_cm in primary_set or abs_cm in curated_reference_paths:
                        continue
                    if abs_cm.suffix.lower() in SUPPORTED_MEDIA_TYPES:
                        curated_reference_paths.append(abs_cm)

        header_sections: List[str] = []
        if curate_studio_context:
            if curated_prompt_block:
                header_sections.append(curated_prompt_block)
            if extra_studio_context and extra_studio_context.strip():
                header_sections.append(f"=== CALLER STUDIO CONTEXT ===\n{extra_studio_context.strip()}")
        else:
            # Strict Opaque-Box Mode: ONLY `opaque_context_text` may enter the prompt (zero Studio leakage)
            if opaque_context_text and opaque_context_text.strip():
                header_sections.append(
                    f"=== PLAYER-FACING GAME CONTEXT (OPAQUE BOX) ===\n{opaque_context_text.strip()}"
                )

        header_sections.append(
            f"=== EVALUATION INQUIRY ===\n{inquiry_prompt.strip()}\n\n"
            f"Primary media files attached for evaluation ({len(primary_paths)}): "
            + ", ".join(p.name for p in primary_paths)
        )

        input_items: List[Dict[str, Any]] = [
            {"type": "text", "text": "\n\n".join(header_sections)}
        ]

        cumulative_inline_bytes = 0
        uploaded_file_names: List[str] = []

        def _append_media_block(p: Path, role_label: str) -> None:
            nonlocal cumulative_inline_bytes
            c_type, mime_type = classify_media_path(p)
            f_size = p.stat().st_size
            input_items.append(
                {
                    "type": "text",
                    "text": f"{role_label}: {p.name} (type={c_type}, mime_type={mime_type}, bytes={f_size})",
                }
            )
            if cumulative_inline_bytes + f_size <= self.inline_max_bytes:
                b64_payload = base64.b64encode(p.read_bytes()).decode("ascii")
                cumulative_inline_bytes += f_size
                input_items.append(
                    {
                        "type": c_type,
                        "data": b64_payload,
                        "mime_type": mime_type,
                    }
                )
            else:
                file_uri, uploaded_name = self._upload_large_media(p, mime_type)
                if uploaded_name:
                    uploaded_file_names.append(uploaded_name)
                input_items.append(
                    {
                        "type": c_type,
                        "uri": file_uri,
                        "mime_type": mime_type,
                    }
                )

        try:
            for prim_p in primary_paths:
                _append_media_block(prim_p, "PRIMARY EVALUATION MEDIA")
            for ref_p in curated_reference_paths:
                _append_media_block(ref_p, "CURATED STUDIO REFERENCE MEDIA")

            sanitized_schema = _sanitize_gemini_json_schema(AudioEvaluationReport.model_json_schema())
            call_kwargs: Dict[str, Any] = {
                "model": eff_model,
                "system_instruction": _SYSTEM_INSTRUCTION,
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

            call_label = (
                f"audio_evaluator[{item_id or 'studio'}]"
                if curate_studio_context
                else "playtest_audio_evaluator"
            )
            if curate_studio_context:
                interaction = call_gemini_with_transient_retry(
                    lambda: interactions_api.create(**call_kwargs),
                    label=call_label,
                    candidate_dir=studio_cdir,
                )
            else:
                interaction = interactions_api.create(**call_kwargs)
        finally:
            if uploaded_file_names:
                files_api = getattr(self.client, "files", None)
                delete_fn = getattr(files_api, "delete", None) if files_api is not None else None
                if callable(delete_fn):
                    for up_name in uploaded_file_names:
                        try:
                            delete_fn(name=up_name)
                        except Exception:
                            pass

        if token_tracker is not None and hasattr(token_tracker, "record_response"):
            token_tracker.record_response(interaction)

        usage_dict = extract_interaction_usage_dict(interaction)
        self.last_usage = dict(usage_dict)
        raw_text = getattr(interaction, "output_text", None)
        if not raw_text or not str(raw_text).strip():
            outputs_list = getattr(interaction, "outputs", None) or []
            for out_part in reversed(list(outputs_list)):
                p_type = out_part.get("type") if isinstance(out_part, dict) else getattr(out_part, "type", None)
                p_txt = out_part.get("text") if isinstance(out_part, dict) else getattr(out_part, "text", None)
                if p_type in (None, "text") and p_txt and str(p_txt).strip():
                    raw_text = str(p_txt)
                    break
        if not raw_text or not str(raw_text).strip():
            raise StudioAgentError(
                f"AudioEvaluatorAgent received an empty `output_text` from Gemini Interactions API (model={eff_model})."
            )

        try:
            report = AudioEvaluationReport.model_validate_json(str(raw_text))
        except Exception as exc:
            raise StudioAgentError(
                f"AudioEvaluatorAgent output failed AudioEvaluationReport schema validation: {exc}"
            ) from exc

        # Mandatory Observability Parity (Rule 4): log reasoning_rationale
        logger.info("reasoning_rationale[audio_evaluator]: %s", report.deliberation)
        telemetry_payload = {
            "deliberation": report.deliberation,
            "reasoning_rationale": report.deliberation,
            "item_id": item_id,
            "media_files": [p.name for p in primary_paths],
            "curate_studio_context": curate_studio_context,
            "usage": dict(usage_dict),
        }
        if self.trace_logger is not None:
            log_fn = getattr(self.trace_logger, "log_event", None) or getattr(self.trace_logger, "record", None)
            if callable(log_fn):
                try:
                    log_fn("audio_evaluation_completed", telemetry_payload)
                except Exception:
                    pass
        if self.event_callback is not None:
            try:
                self.event_callback("audio_evaluation_completed", telemetry_payload)
            except Exception:
                pass

        if curate_studio_context and enrich_catalog and studio_cdir is not None:
            enrich_catalog_audio_entries(studio_cdir, primary_paths, report)

        return report, usage_dict
