"""engine/supervisor/asset_validator.py
Multimodal Asset Integrity Gate (Gate E) — Feature 25.
Validates voice audio (RIFF headers, duration bounds, non-silence),
visual assets (PNG signature, dimensions, 16-color Style Bible palette conformance),
and narrative bibles (JSON schema, on-disk asset references, dialogue tree reachability/acyclicity).
Strictly adheres to Rule 4 (Rationale-First) and Rule 5 (Zero Silent Degradation).
"""
from __future__ import annotations
import json
import logging
import math
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field

from engine.assets.palette import STYLE_BIBLE_16_PALETTE
from engine.studio.models import BaseEpistemicModel, DEFAULT_STUDIO_MODEL, is_real_genai_client
from engine.supervisor.models import (
    AssetValidationReport,
    GateFailureType,
    QualityGateResult,
)

logger = logging.getLogger("AssetValidator")

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
VALID_AUDIO_SAMPLE_RATES = {8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000}


class AssetSheetMultimodalAuditVerdict(BaseEpistemicModel):
    """Rationale-First (Rule 4) multimodal critic verdict over a cartridge's labeled visual asset contact sheet."""
    offending_assets: List[str] = Field(
        default_factory=list,
        description=(
            "Relative cartridge paths (e.g. 'assets/sprites/example.png', 'assets/ui/icons.png') of assets on the "
            "contact sheet that are visually defective: flat solid-color geometric placeholder shapes clashing with "
            "the game's illustrated/painted style guide, blank/erased multi-cell atlas slots, garbled artifacts, "
            "or severe visual style outliers."
        ),
    )
    offending_scripts: List[str] = Field(
        default_factory=list,
        description="Optional relative '.gd' script paths implicated in texture-slicing or binding mismatches.",
    )
    remediation_summary: str = Field(
        default="",
        description="Concise per-asset explanation of what is broken and how each offending asset must be regenerated.",
    )
    all_assets_coherent_and_intact: bool = Field(
        ...,
        description="True ONLY when every game asset on the contact sheet is intact, populated across expected cells, and stylistically coherent with the Game Bible and concept art.",
    )


def collect_cartridge_visual_asset_files(cartridge_dir: Union[Path, str]) -> List[Path]:
    """Collects production game visual PNGs across `assets/sprites/`, `assets/tiles/`, `assets/ui/`,
    `assets/portraits/`, `assets/backgrounds/`, plus legacy `sprites/`, `tiles/`, and root `*.png`,
    strictly excluding `mockups/`, `concepts/`, `playtest_captures/`, `artifacts/`, and `.godot/`.
    """
    cdir = Path(cartridge_dir).resolve()
    excluded_parts = {"mockups", "concepts", "playtest_captures", "artifacts", ".godot", "tests"}
    found: List[Path] = []
    scan_dirs = [
        cdir / "assets" / "sprites",
        cdir / "assets" / "tiles",
        cdir / "assets" / "ui",
        cdir / "assets" / "portraits",
        cdir / "assets" / "backgrounds",
        cdir / "sprites",
        cdir / "tiles",
    ]
    for sdir in scan_dirs:
        if sdir.is_dir():
            for p in sorted(sdir.rglob("*.png")):
                rel_parts = set(p.relative_to(cdir).parts)
                if rel_parts & excluded_parts:
                    continue
                found.append(p)
    for p in sorted(cdir.glob("*.png")):
        found.append(p)
    return list(dict.fromkeys(found))


def _collect_observational_asset_metadata(
    cartridge_dir: Union[Path, str],
    visual_files: Optional[List[Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Collects pure observational metadata (`size`, `grid`, `in_catalog`, `referenced`, `mode`, `purpose`,
    `perceptual_description`) from `assets/asset_library.json`, production `.gd`/`.tscn`/`.tres`/`.gdshader`
    references, and PNG headers without prescriptive slicing heuristics."""
    cdir = Path(cartridge_dir).resolve()
    vfiles = visual_files if visual_files is not None else collect_cartridge_visual_asset_files(cdir)
    by_rel_path: Dict[str, Dict[str, Any]] = {}
    cat_path = cdir / "assets" / "asset_library.json"
    if cat_path.is_file():
        try:
            raw_cat = json.loads(cat_path.read_text(encoding="utf-8"))
            raw_entries = raw_cat.get("entries", raw_cat) if isinstance(raw_cat, dict) else {}
            if isinstance(raw_entries, dict):
                for aval in raw_entries.values():
                    if isinstance(aval, dict):
                        fp = str(aval.get("file_path") or aval.get("res_path") or "").replace("res://", "").lstrip("/")
                        if fp:
                            by_rel_path[fp] = aval
        except Exception:
            pass

    excluded_src_dirs = {"tests", "artifacts", ".godot", "mockups", "concepts", "playtest_captures"}
    source_chunks: List[str] = []
    if cdir.is_dir():
        for pattern in ("*.gd", "*.tscn", "*.tres", "*.gdshader"):
            for src_file in cdir.rglob(pattern):
                try:
                    rel_parts = set(src_file.relative_to(cdir).parts)
                    if rel_parts & excluded_src_dirs:
                        continue
                    source_chunks.append(src_file.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass
    referenced_blob = "\n".join(source_chunks)

    asset_metadata: Dict[str, Dict[str, Any]] = {}
    for vfile in vfiles:
        try:
            rel_p = vfile.relative_to(cdir).as_posix()
        except Exception:
            rel_p = vfile.name
        c_entry = by_rel_path.get(rel_p, {})
        cols = max(1, int(c_entry.get("hframes") or 1))
        rows = max(1, int(c_entry.get("vframes") or 1))
        w, h = 0, 0
        try:
            with Image.open(vfile) as im:
                w, h = im.size
        except Exception:
            pass
        fname = Path(rel_p).name
        is_referenced = bool(
            referenced_blob and ((rel_p in referenced_blob) or (bool(fname) and fname in referenced_blob))
        )
        asset_metadata[rel_p] = {
            "size": [w, h],
            "grid": [cols, rows],
            "catalog_grid": [cols, rows],
            "in_catalog": bool(c_entry),
            "referenced": is_referenced,
            "mode": str(c_entry.get("fabrication_mode") or c_entry.get("asset_type") or "static_sprite"),
            "purpose": str(c_entry.get("purpose") or ""),
            "perceptual_description": str(c_entry.get("perceptual_description") or ""),
        }
    return asset_metadata


def render_cartridge_asset_contact_sheet(
    cartridge_dir: Union[Path, str],
    output_path: Optional[Union[Path, str]] = None,
    max_tiles: int = 64,
    flagged_assets: Optional[Set[str]] = None,
    active_only: bool = False,
) -> Optional[Path]:
    """Renders a neutral labeled checkerboard contact sheet of a cartridge's production visual assets
    (`assets/sprites/`, `assets/ui/`, `assets/tiles/`, `assets/portraits/`, `assets/backgrounds/`),
    sorting actively referenced in-game assets first and overlaying `(hframes, vframes)` catalog grid lines.
    """
    cdir = Path(cartridge_dir).resolve()
    all_vfiles = collect_cartridge_visual_asset_files(cdir)
    if not all_vfiles:
        return None
    meta_map = _collect_observational_asset_metadata(cdir, all_vfiles)
    any_referenced = any(bool(m.get("referenced")) for m in meta_map.values())

    def _sort_key(vf_path: Path) -> Tuple[int, str]:
        rp = vf_path.relative_to(cdir).as_posix() if vf_path.is_relative_to(cdir) else vf_path.name
        info_m = meta_map.get(rp, {})
        is_ref = bool(info_m.get("referenced"))
        in_cat = bool(info_m.get("in_catalog"))
        is_part = "/parts/" in f"/{rp}"
        if is_part:
            tier = 3
        elif is_ref:
            tier = 0
        elif in_cat:
            tier = 1
        else:
            tier = 2
        return (tier, rp)

    candidate_vfiles = all_vfiles
    if active_only and any_referenced:
        active_subset = [
            vf_p
            for vf_p in all_vfiles
            if bool(
                meta_map.get(
                    vf_p.relative_to(cdir).as_posix() if vf_p.is_relative_to(cdir) else vf_p.name,
                    {},
                ).get("referenced")
            )
        ]
        if active_subset:
            candidate_vfiles = active_subset

    vfiles: List[Path] = sorted(candidate_vfiles, key=_sort_key)[:max_tiles]
    flagged_set: Set[str] = {
        str(p).replace("res://", "").lstrip("/") for p in (flagged_assets or set()) if p
    }

    cols = min(4, max(1, int(math.ceil(math.sqrt(len(vfiles))))))
    rows = int(math.ceil(len(vfiles) / float(cols)))
    card_w, card_h = 260, 216
    pad = 12
    header_h = 36
    sheet_w = cols * card_w + (cols + 1) * pad
    sheet_h = header_h + rows * card_h + (rows + 1) * pad

    sheet = Image.new("RGBA", (sheet_w, sheet_h), (20, 23, 29, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (pad, 10),
        f"CARTRIDGE VISUAL ASSET CONTACT SHEET ({len(vfiles)} assets — checkerboard=alpha, cyan=catalog grid)",
        fill=(235, 240, 248, 255),
    )

    thumb_w, thumb_h = card_w - 20, card_h - 52
    chk = 12
    for idx, vf in enumerate(vfiles):
        r_i = idx // cols
        c_i = idx % cols
        x0 = pad + c_i * (card_w + pad)
        y0 = header_h + pad + r_i * (card_h + pad)
        rel_p = vf.relative_to(cdir).as_posix() if vf.is_relative_to(cdir) else vf.name
        info = meta_map.get(rel_p, {})
        grid = info.get("grid") or [1, 1]
        g_cols, g_rows = max(1, int(grid[0])), max(1, int(grid[1]))
        in_cat = bool(info.get("in_catalog"))
        is_flagged = rel_p in flagged_set

        border_col = (235, 80, 80, 255) if is_flagged else ((95, 165, 215, 255) if in_cat else (82, 94, 114, 255))
        draw.rectangle([x0, y0, x0 + card_w - 1, y0 + card_h - 1], fill=(30, 34, 42, 255), outline=border_col, width=2)

        tx0, ty0 = x0 + 10, y0 + 8
        for cy in range(0, thumb_h, chk):
            for cx in range(0, thumb_w, chk):
                c_tone = (58, 63, 74, 255) if ((cx // chk + cy // chk) % 2 == 0) else (42, 46, 55, 255)
                draw.rectangle(
                    [tx0 + cx, ty0 + cy, min(tx0 + thumb_w - 1, tx0 + cx + chk - 1), min(ty0 + thumb_h - 1, ty0 + cy + chk - 1)],
                    fill=c_tone,
                )

        w_px, h_px = 0, 0
        try:
            with Image.open(vf) as im:
                rgba = im.convert("RGBA")
                w_px, h_px = rgba.size
                scale = min(thumb_w / float(max(1, w_px)), thumb_h / float(max(1, h_px)))
                nw = max(1, int(round(w_px * scale)))
                nh = max(1, int(round(h_px * scale)))
                interp = Image.Resampling.NEAREST if (w_px <= 128 and h_px <= 128) else Image.Resampling.LANCZOS
                resized = rgba.resize((nw, nh), interp)
                px_off = tx0 + (thumb_w - nw) // 2
                py_off = ty0 + (thumb_h - nh) // 2
                sheet.paste(resized, (px_off, py_off), resized)
                draw.rectangle([px_off, py_off, px_off + nw - 1, py_off + nh - 1], outline=(110, 125, 145, 200), width=1)
                if g_cols * g_rows > 1:
                    for gc in range(1, g_cols):
                        gx = px_off + int(round(nw * gc / float(g_cols)))
                        draw.line([(gx, py_off), (gx, py_off + nh - 1)], fill=(0, 225, 255, 210), width=1)
                    for gr in range(1, g_rows):
                        gy = py_off + int(round(nh * gr / float(g_rows)))
                        draw.line([(px_off, gy), (px_off + nw - 1, gy)], fill=(0, 225, 255, 210), width=1)
        except Exception:
            draw.text((tx0 + 8, ty0 + 8), "DECODE ERROR", fill=(255, 90, 90, 255))

        short_rel = rel_p.replace("assets/", "")
        if len(short_rel) > 34:
            short_rel = "..." + short_rel[-31:]
        draw.text((x0 + 8, y0 + thumb_h + 14), short_rel, fill=(230, 236, 245, 255))
        flag_tag = " [FLAGGED]" if is_flagged else (" [UNREF]" if (any_referenced and not bool(info.get("referenced"))) else "")
        meta_lbl = f"{w_px}x{h_px}px | grid={g_cols}x{g_rows}{flag_tag}"
        draw.text(
            (x0 + 8, y0 + thumb_h + 30),
            meta_lbl,
            fill=(255, 120, 120, 255) if is_flagged else (165, 178, 198, 255),
        )

    dest = Path(output_path) if output_path is not None else (cdir / "artifacts" / "proofs" / "asset_contact_sheet.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.convert("RGB").save(dest, format="PNG")
    return dest


class AssetValidator:
    """Evaluates candidate cartridge assets against structural binary & JSON integrity invariants."""
    deliberation: str = "Multimodal Asset Validator verifying binary audio, visual, and narrative integrity."

    def __init__(
        self,
        palette: Optional[List[Tuple[int, int, int]]] = None,
        enforce_palette: bool = False,
        min_audio_duration_sec: float = 0.02,
        max_audio_duration_sec: float = 300.0,
        silence_threshold_rms: float = 1.0,
        gemini_client: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        model_name: str = DEFAULT_STUDIO_MODEL,
    ):
        self.palette = palette or STYLE_BIBLE_16_PALETTE
        self.palette_set = set(self.palette)
        self.enforce_palette = enforce_palette
        self.min_audio_duration = float(min_audio_duration_sec)
        self.max_audio_duration = float(max_audio_duration_sec)
        self.silence_threshold_rms = float(silence_threshold_rms)
        self.gemini_client = gemini_client
        self.event_bus = event_bus
        self.model_name = model_name

    # -------------------------------------------------------------------------
    # Audio Validation (.wav)
    # -------------------------------------------------------------------------
    def validate_voice_audio(self, path: Union[Path, str]) -> AssetValidationReport:
        """Validates RIFF/WAVE header, PCM encoding, sample rate, duration, and non-silence."""
        audio_path = Path(path)
        if not audio_path.is_file():
            return AssetValidationReport(
                deliberation=f"Audio file does not exist at {audio_path}.",
                asset_path=str(audio_path),
                asset_type="audio",
                passed=False,
                errors=[f"File not found: {audio_path}"],
            )

        data = audio_path.read_bytes()
        errors: List[str] = []

        if len(data) < 44:
            return AssetValidationReport(
                deliberation=f"File size ({len(data)} bytes) is below standard 44-byte RIFF header size.",
                asset_path=str(audio_path),
                asset_type="audio",
                passed=False,
                errors=[f"File size {len(data)} < 44 bytes"],
            )

        # 1. RIFF Magic & Format
        riff_chunk_id, riff_size, wave_id = struct.unpack("<4sI4s", data[:12])
        if riff_chunk_id != b"RIFF":
            errors.append(f"Invalid RIFF magic bytes: {riff_chunk_id!r}")
        if wave_id != b"WAVE":
            errors.append(f"Invalid WAVE format marker: {wave_id!r}")

        # 2. Format Chunk
        fmt_chunk_id, fmt_size = struct.unpack("<4sI", data[12:20])
        if fmt_chunk_id != b"fmt ":
            errors.append(f"Invalid fmt chunk marker: {fmt_chunk_id!r}")

        try:
            audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack(
                "<HHIIHH", data[20:36]
            )
        except Exception as exc:
            return AssetValidationReport(
                deliberation=f"Corrupt fmt chunk: {exc}",
                asset_path=str(audio_path),
                asset_type="audio",
                passed=False,
                errors=[f"Failed to unpack fmt chunk: {exc}"],
            )

        if audio_format != 1:
            errors.append(f"Unsupported audio format tag {audio_format}; only standard PCM (1) is permitted")
        if channels not in (1, 2):
            errors.append(f"Unsupported channel count {channels}; only mono (1) or stereo (2) permitted")
        if sample_rate not in VALID_AUDIO_SAMPLE_RATES:
            errors.append(f"Invalid audio sample rate {sample_rate} Hz")

        expected_byte_rate = sample_rate * channels * (bits_per_sample // 8)
        if byte_rate != expected_byte_rate:
            errors.append(f"Mismatched byte rate: got {byte_rate}, expected {expected_byte_rate}")

        # 3. Data Subchunk
        offset = 20 + fmt_size
        data_chunk_found = False
        pcm_bytes = b""
        data_size = 0
        while offset + 8 <= len(data):
            chunk_tag, chunk_len = struct.unpack("<4sI", data[offset:offset+8])
            if chunk_tag == b"data":
                data_chunk_found = True
                data_size = chunk_len
                pcm_bytes = data[offset+8 : offset+8+chunk_len]
                break
            offset += 8 + chunk_len

        if not data_chunk_found:
            errors.append("Missing 'data' subchunk in WAV file")

        duration = 0.0
        if expected_byte_rate > 0 and data_size > 0:
            duration = data_size / float(expected_byte_rate)

        # 4. Duration Bounds [min, max]
        if duration < self.min_audio_duration:
            errors.append(
                f"Audio duration {duration:.2f}s is too short: below acceptable minimum {self.min_audio_duration}s"
            )
        elif duration > self.max_audio_duration:
            errors.append(
                f"Audio duration {duration:.2f}s is too long: exceeds acceptable maximum {self.max_audio_duration}s"
            )

        # 5. Non-Silence Amplitude Check
        max_amplitude = 0
        rms_amplitude = 0.0
        if pcm_bytes and bits_per_sample == 16:
            num_samples = len(pcm_bytes) // 2
            try:
                samples = struct.unpack(f"<{num_samples}h", pcm_bytes[:num_samples * 2])
                if samples:
                    max_amplitude = max(abs(s) for s in samples)
                    sum_sq = sum(float(s) * float(s) for s in samples)
                    rms_amplitude = math.sqrt(sum_sq / float(num_samples))
                    if max_amplitude == 0 or rms_amplitude < self.silence_threshold_rms:
                        errors.append(
                            f"Total silence detected: max amplitude {max_amplitude}, RMS {rms_amplitude:.2f} < {self.silence_threshold_rms}"
                        )
            except Exception as exc:
                errors.append(f"Failed to unpack PCM audio samples: {exc}")

        passed = len(errors) == 0
        delib = (
            f"Voice audio '{audio_path.name}' verified: {duration:.2f}s PCM {sample_rate}Hz {channels}ch, "
            f"peak amplitude={max_amplitude}, RMS={rms_amplitude:.2f}."
            if passed else f"Voice audio '{audio_path.name}' failed validation: {'; '.join(errors)}"
        )
        return AssetValidationReport(
            deliberation=delib,
            asset_path=str(audio_path),
            asset_type="audio",
            passed=passed,
            errors=errors,
            details={
                "duration_sec": duration,
                "duration_seconds": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "bits_per_sample": bits_per_sample,
                "peak_amplitude": max_amplitude,
                "rms_amplitude": rms_amplitude,
            },
        )

    # -------------------------------------------------------------------------
    # Visual Asset Validation (.png)
    # -------------------------------------------------------------------------
    def validate_visual_asset(
        self,
        path: Union[Path, str],
        expected_size: Optional[Tuple[int, int]] = None,
        min_dimension: int = 8,
        max_dimension: int = 8192,
    ) -> AssetValidationReport:
        """Validates 8-byte PNG signature, dimensions, and strict 16-color Style Bible palette conformance."""
        png_path = Path(path)
        if not png_path.is_file():
            return AssetValidationReport(
                deliberation=f"Visual asset file not found at {png_path}.",
                asset_path=str(png_path),
                asset_type="sprite",
                passed=False,
                errors=[f"File not found: {png_path}"],
            )

        raw_bytes = png_path.read_bytes()
        errors: List[str] = []

        # 1. 8-Byte PNG Signature
        if len(raw_bytes) < 8 or raw_bytes[:8] != PNG_SIGNATURE:
            return AssetValidationReport(
                deliberation=f"Invalid PNG signature for {png_path.name}: got {raw_bytes[:8]!r}.",
                asset_path=str(png_path),
                asset_type="sprite",
                passed=False,
                errors=[f"File does not have valid 8-byte PNG signature: {raw_bytes[:8]!r}"],
            )

        # 2. Dimensions & Decode
        try:
            with Image.open(png_path) as img:
                w, h = img.size
                mode = img.mode

                if w < min_dimension or h < min_dimension or w > max_dimension or h > max_dimension:
                    errors.append(f"Image dimensions ({w}x{h}) below minimum {min_dimension}px or outside permissible bounds [{min_dimension}..{max_dimension}]")

                if expected_size is not None and (w, h) != expected_size:
                    errors.append(f"Image dimensions ({w}x{h}) do not match expected size ({expected_size[0]}x{expected_size[1]})")

                # 3. Optional 16-Color Style Bible Palette Conformance
                out_of_palette_count = 0
                if self.enforce_palette:
                    sample_violations: List[Dict[str, Any]] = []

                    if mode == "P":
                        rgba_img = img.convert("RGBA")
                        for y in range(h):
                            for x in range(w):
                                r, g, b, a = rgba_img.getpixel((x, y))
                                if a >= 128 and (r, g, b) not in self.palette_set:
                                    out_of_palette_count += 1
                                    if len(sample_violations) < 5:
                                        sample_violations.append({"pos": (x, y), "rgb": (r, g, b)})
                    elif mode == "RGBA":
                        for y in range(h):
                            for x in range(w):
                                r, g, b, a = img.getpixel((x, y))
                                if a >= 128 and (r, g, b) not in self.palette_set:
                                    out_of_palette_count += 1
                                    if len(sample_violations) < 5:
                                        sample_violations.append({"pos": (x, y), "rgb": (r, g, b)})
                    elif mode == "RGB":
                        for y in range(h):
                            for x in range(w):
                                r, g, b = img.getpixel((x, y))
                                if (r, g, b) not in self.palette_set:
                                    out_of_palette_count += 1
                                    if len(sample_violations) < 5:
                                        sample_violations.append({"pos": (x, y), "rgb": (r, g, b)})
                    else:
                        errors.append(f"Unsupported image color mode '{mode}'; expected 'P', 'RGBA', or 'RGB'")

                    if out_of_palette_count > 0:
                        errors.append(
                            f"Style Bible palette violation: {out_of_palette_count} pixels do not match 16-color palette. "
                            f"Samples: {sample_violations}"
                        )

        except Exception as exc:
            return AssetValidationReport(
                deliberation=f"Image decoding failure for {png_path.name}: {exc}",
                asset_path=str(png_path),
                asset_type="sprite",
                passed=False,
                errors=[f"Corrupt PNG file: {exc}"],
            )

        passed = len(errors) == 0
        delib = (
            f"Visual asset '{png_path.name}' verified: {w}x{h} mode={mode}."
            if passed else f"Visual asset '{png_path.name}' failed validation: {'; '.join(errors)}"
        )
        return AssetValidationReport(
            deliberation=delib,
            asset_path=str(png_path),
            asset_type="sprite",
            passed=passed,
            errors=errors,
            details={"width": w, "height": h, "mode": mode, "out_of_palette_count": out_of_palette_count},
        )

    # -------------------------------------------------------------------------
    # Narrative Bible Validation (narrative.json)
    # -------------------------------------------------------------------------
    def validate_narrative_bible(
        self,
        cartridge_dir: Union[Path, str],
        narrative_path: Optional[Union[Path, str]] = None,
    ) -> AssetValidationReport:
        """Validates narrative.json schema, character portrait paths, voice audio paths, and dialogue trees."""
        cdir = Path(cartridge_dir).resolve()
        npath = Path(narrative_path).resolve() if narrative_path else cdir / "narrative.json"

        if not npath.is_file():
            return AssetValidationReport(
                deliberation=f"narrative.json does not exist at {npath}.",
                asset_path=str(npath),
                asset_type="narrative",
                passed=False,
                errors=[f"narrative.json missing at {npath}"],
            )

        errors: List[str] = []
        try:
            with open(npath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            return AssetValidationReport(
                deliberation=f"Malformed JSON syntax in {npath}: {exc}",
                asset_path=str(npath),
                asset_type="narrative",
                passed=False,
                errors=[f"Invalid JSON: {exc}"],
            )

        # 1. Top-Level Schema Checking
        has_world = "world" in data or "world_title" in data or "deliberation" in data or "title" in data
        has_chars = "characters" in data
        has_dialogues = "dialogues" in data or "dialogue_barks" in data or "dialogue_tree" in data or "dialogue_trees" in data

        if not has_world:
            errors.append("narrative.json missing required 'world' or 'world_title' field")
        if not has_chars:
            errors.append("narrative.json missing required 'characters' field")
        if not has_dialogues:
            errors.append("narrative.json missing required 'dialogues' or 'dialogue_barks' field")

        board_managed = (cdir / "backlog.json").is_file()
        referenced_source_blob = ""
        if board_managed:
            source_chunks: List[str] = []
            for pattern in ("*.gd", "*.tscn"):
                for src_file in cdir.rglob(pattern):
                    if "tests" in src_file.relative_to(cdir).parts:
                        continue
                    try:
                        source_chunks.append(src_file.read_text(encoding="utf-8", errors="ignore"))
                    except Exception:
                        pass
            referenced_source_blob = "\n".join(source_chunks)

        def _is_asset_referenced_in_cartridge(rel_path: str) -> bool:
            if not referenced_source_blob:
                return False
            clean_rel = rel_path.replace("res://", "").lstrip("/")
            fname = Path(clean_rel).name
            return (clean_rel in referenced_source_blob) or (bool(fname) and fname in referenced_source_blob)

        # 2. Check Character Portrait Paths Exist on Disk
        chars_raw = data.get("characters", {})
        if isinstance(chars_raw, dict):
            char_entries: List[Tuple[str, Dict[str, Any]]] = [
                (str(k), v) for k, v in chars_raw.items() if isinstance(v, dict)
            ]
        elif isinstance(chars_raw, list):
            char_entries = [
                (str(c.get("id") or c.get("character_id") or "unknown"), c)
                for c in chars_raw
                if isinstance(c, dict)
            ]
        else:
            char_entries = []

        for key_id, c in char_entries:
            cid = c.get("id") or c.get("character_id") or key_id or "unknown"
            portrait_rel = c.get("portrait") or c.get("portrait_asset") or c.get("portrait_file")
            if portrait_rel:
                clean_portrait_rel = str(portrait_rel).replace("res://", "").lstrip("/")
                portrait_file = cdir / clean_portrait_rel
                if not portrait_file.is_file():
                    pname = Path(clean_portrait_rel).name
                    if (cdir / "assets" / "portraits" / pname).is_file():
                        pass
                    elif (cdir / "assets" / "sprites" / pname).is_file():
                        pass
                    elif (cdir / pname).is_file():
                        pass
                    elif board_managed and not _is_asset_referenced_in_cartridge(clean_portrait_rel):
                        # In Continuous Production Board builds, narrative.json exports full GameBible lore,
                        # while only Council-commissioned or code-referenced assets are fabricated on disk.
                        pass
                    else:
                        errors.append(f"Character '{cid}' references missing portrait file on disk: {portrait_rel}")
            elif not board_managed:
                errors.append(f"Character '{cid}' missing portrait asset reference")

        # 3. Check Voice Audio Paths Exist on Disk
        dialogues_raw = data.get("dialogues") or data.get("dialogue_barks")
        bark_list: List[Dict[str, Any]] = []
        if isinstance(dialogues_raw, list):
            bark_list = [b for b in dialogues_raw if isinstance(b, dict)]
        elif isinstance(dialogues_raw, dict):
            for category_barks in dialogues_raw.values():
                if isinstance(category_barks, list):
                    bark_list.extend(b for b in category_barks if isinstance(b, dict))

        for bark in bark_list:
            voice_rel = bark.get("voice_clip") or bark.get("audio_clip") or bark.get("voice_bark_file") or bark.get("voice_file")
            bark_id = bark.get("bark_id") or bark.get("text", "")[:15]
            if voice_rel:
                clean_voice_rel = str(voice_rel).replace("res://", "").lstrip("/")
                audio_file = cdir / clean_voice_rel
                if not audio_file.is_file():
                    vname = Path(clean_voice_rel).name
                    if (cdir / "assets" / "audio" / vname).is_file():
                        pass
                    elif (cdir / vname).is_file():
                        pass
                    elif board_managed and not _is_asset_referenced_in_cartridge(clean_voice_rel):
                        pass
                    else:
                        errors.append(f"Dialogue cue '{bark_id}' references missing audio file on disk: {voice_rel}")
            elif not board_managed:
                errors.append(f"Dialogue cue '{bark_id}' missing voice clip reference")

        # 4. Dialogue Tree Graph Validation (Cycles & Orphans)
        trees = data.get("dialogue_trees", [])
        if not trees and "dialogue_tree" in data:
            dtree = data["dialogue_tree"]
            if isinstance(dtree, dict) and "nodes" not in dtree:
                trees = [{"root_node_id": "root" if "root" in dtree else next(iter(dtree.keys())), "nodes": dtree}]
            elif isinstance(dtree, dict):
                trees = [dtree]
            elif isinstance(dtree, list):
                trees = dtree
        for tree in trees:
            tree_errors = self._validate_dialogue_graph(tree)
            errors.extend(tree_errors)

        passed = len(errors) == 0
        delib = (
            f"narrative.json verified: {len(char_entries)} characters and {len(bark_list)} dialogue cues confirmed with valid on-disk assets."
            if passed else f"narrative.json failed validation: {'; '.join(errors)}"
        )
        return AssetValidationReport(
            deliberation=delib,
            asset_path=str(npath),
            asset_type="narrative",
            passed=passed,
            errors=errors,
            details={"character_count": len(char_entries), "dialogue_count": len(bark_list)},
        )

    def _validate_dialogue_graph(self, tree: Dict[str, Any]) -> List[str]:
        """Ensures dialogue tree is a valid DAG or directed graph without infinite loops and zero orphan nodes."""
        errors: List[str] = []
        nodes = tree.get("nodes", {})
        if not nodes:
            return ["Dialogue tree contains 0 nodes"]

        root_id = tree.get("root_node_id")
        if not root_id or root_id not in nodes:
            return [f"Root node '{root_id}' does not exist in nodes dictionary"]

        # 1. Reachability from root (BFS)
        visited: Set[str] = {root_id}
        queue: List[str] = [root_id]
        while queue:
            curr = queue.pop(0)
            node_data = nodes.get(curr, {})
            choices = node_data.get("choices", [])
            for ch in choices:
                tgt = ch.get("target_node_id") or ch.get("next_node")
                if tgt and tgt not in nodes:
                    errors.append(f"Node '{curr}' choice targets non-existent node '{tgt}'")
                elif tgt and tgt not in visited:
                    visited.add(tgt)
                    queue.append(tgt)

        # Detect orphans
        orphans = set(nodes.keys()) - visited
        if orphans:
            errors.append(f"Orphan dialogue nodes unreachable from root '{root_id}': {sorted(list(orphans))}")

        # 2. Cycle Analysis (Check for closed infinite loops without escape)
        for n_id, n_data in nodes.items():
            choices = n_data.get("choices", [])
            targets = [c.get("target_node_id") or c.get("next_node") for c in choices if (c.get("target_node_id") or c.get("next_node"))]
            for t in targets:
                if t in nodes and t != n_id:
                    back_targets = [
                        c.get("target_node_id") or c.get("next_node")
                        for c in nodes[t].get("choices", [])
                        if (c.get("target_node_id") or c.get("next_node"))
                    ]
                    if n_id in back_targets and len(choices) == 1 and len(nodes[t].get("choices", [])) == 1:
                        errors.append(f"Closed 2-node infinite loop detected between '{n_id}' and '{t}'")

        return errors

    def _run_multimodal_contact_sheet_audit(
        self,
        cdir: Path,
        contact_sheet_path: Path,
        asset_metadata: Dict[str, Dict[str, Any]],
    ) -> Optional[AssetSheetMultimodalAuditVerdict]:
        """Runs a Rationale-First (Rule 4) multimodal visual audit of `contact_sheet_path` against
        `game_bible.json` `visual_style_guide` and concept mockups using `gemini-3.8-flash`."""
        if not is_real_genai_client(self.gemini_client) or not contact_sheet_path.is_file():
            return None
        try:
            from google.genai import types

            bible_data = json.loads((cdir / "game_bible.json").read_text(encoding="utf-8"))
            style_guide = str(bible_data.get("visual_style_guide") or "")
            camera_paradigm = str(bible_data.get("camera_and_spatial_paradigm") or "")
            game_title = str(bible_data.get("game_title") or cdir.name)

            parts: List[Any] = [
                types.Part.from_bytes(
                    data=contact_sheet_path.read_bytes(),
                    mime_type="image/png",
                    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                )
            ]
            for m_dir in (cdir / "assets" / "mockups", cdir / "artifacts" / "concept_art"):
                if m_dir.is_dir():
                    for mp in sorted(m_dir.glob("*.png"))[:2]:
                        if mp.is_file():
                            parts.append(types.Part.from_bytes(data=mp.read_bytes(), mime_type="image/png"))

            manifest_lines = [
                f"- {rel}: active_in_game={bool(meta.get('referenced'))}, size={meta.get('size')}, grid={meta.get('grid')}, mode={meta.get('mode')}, purpose={str(meta.get('purpose') or '')[:90]}"
                for rel, meta in list(asset_metadata.items())[:64]
            ]
            prompt_txt = (
                f"You are the Gate B Multimodal Visual Asset Integrity Critic auditing '{game_title}'.\n"
                f"Image 1 is the labeled CARTRIDGE VISUAL ASSET CONTACT SHEET (active in-game assets listed first; "
                f"checkerboard = transparent alpha; cyan lines = multi-cell atlas boundaries; red card borders = deterministic warnings). "
                f"Any subsequent images are the Game Bible concept art mockups.\n\n"
                f"=== GAME BIBLE VISUAL STYLE GUIDE ===\n{style_guide}\n"
                f"Camera / Spatial Paradigm: {camera_paradigm}\n\n"
                f"=== REGISTERED VISUAL ASSETS (MATCHING TILE ORDER ON SHEET) ===\n" + "\n".join(manifest_lines) + "\n\n"
                f"Audit EVERY tile on the contact sheet (especially `active_in_game=True` assets):\n"
                f"1. Flag any asset rendered as crude flat solid-color / 16-color indexed geometric placeholder shapes "
                f"(e.g. unshaded triangles, pentagons, rectangles, or circles) that clash with the Game Bible's illustrated/shaded/painted style guide and concept mockups.\n"
                f"2. Flag any multi-cell atlas (`grid=CxR` with `C*R > 1`) that has blank/transparent checkerboard cells where icons or frames should be, "
                f"OR any `1x1` sprite that accidentally packs multiple tiny stacked/squished duplicates surrounded by excessive empty transparent padding.\n"
                f"3. Flag any garbled, anamorphically crushed/stretched, severely clipped, or unreadable asset (and if an `active_in_game=True` file is a corrupted/squished "
                f"duplicate shadowing a cleaner asset on the sheet, flag the corrupted `.png` in `offending_assets` and note the path mismatch in `remediation_summary`).\n"
                f"Deliberate thoroughly first (`deliberation`), list exact relative paths (`assets/.../*.png`) in `offending_assets`, "
                f"and set `all_assets_coherent_and_intact` last."
            )
            parts.append(types.Part.from_text(text=prompt_txt))

            from engine.studio.models import _sanitize_gemini_json_schema
            cfg_kwargs: Dict[str, Any] = {
                "response_mime_type": "application/json",
                "response_schema": _sanitize_gemini_json_schema(
                    AssetSheetMultimodalAuditVerdict.model_json_schema()
                ),
            }
            if hasattr(types, "ThinkingConfig") and hasattr(types, "ThinkingLevel"):
                t_level = getattr(types.ThinkingLevel, "MEDIUM", types.ThinkingLevel.LOW)
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=t_level)
            resp = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=parts,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            verdict = AssetSheetMultimodalAuditVerdict.model_validate_json(resp.text or "{}")
            logger.info("reasoning_rationale[gate_b_asset_sheet]: %s", verdict.deliberation)
            if self.event_bus is not None and hasattr(self.event_bus, "publish_agent_stream"):
                try:
                    self.event_bus.publish_agent_stream(
                        agent_name="AssetValidator",
                        message=f"Multimodal Contact Sheet Audit: coherent={verdict.all_assets_coherent_and_intact}, offending={verdict.offending_assets}",
                        deliberation=verdict.deliberation,
                        log_level="INFO" if verdict.all_assets_coherent_and_intact else "WARNING",
                        category="GATE_CHECK",
                    )
                except Exception:
                    pass
            return verdict
        except Exception as exc:
            logger.warning("Multimodal contact sheet audit skipped/failed on %s: %s", cdir.name, exc)
            return None

    # -------------------------------------------------------------------------
    # Full Cartridge Gate E Evaluation
    # -------------------------------------------------------------------------
    def validate_cartridge_assets(self, cartridge_dir: Union[Path, str]) -> QualityGateResult:
        """Full Gate E evaluation over all audio, visual, and narrative assets in candidate directory."""
        cdir = Path(cartridge_dir).resolve()
        deliberation_parts: List[str] = []
        all_errors: List[str] = []
        diagnostic_data: Dict[str, Any] = {"audio_files": 0, "sprite_files": 0, "has_narrative": False}

        has_any_multimodal_assets = False

        # 1. Validate narrative.json if present
        narrative_file = cdir / "narrative.json"
        if narrative_file.is_file():
            has_any_multimodal_assets = True
            diagnostic_data["has_narrative"] = True
            narrative_report = self.validate_narrative_bible(cdir)
            if not narrative_report.passed:
                all_errors.extend(narrative_report.errors)
            deliberation_parts.append(narrative_report.deliberation)

        # 2. Validate audio files under assets/audio/ or root
        audio_dirs = [cdir / "assets" / "audio", cdir / "audio"]
        audio_files = []
        for adir in audio_dirs:
            if adir.is_dir():
                audio_files.extend(list(adir.glob("*.wav")))
        # Also check root wav files
        audio_files.extend(list(cdir.glob("*.wav")))
        audio_files = list(dict.fromkeys(audio_files))

        if audio_files:
            has_any_multimodal_assets = True
            for audio_file in sorted(audio_files):
                diagnostic_data["audio_files"] += 1
                rep = self.validate_voice_audio(audio_file)
                if not rep.passed:
                    all_errors.extend(rep.errors)
                deliberation_parts.append(rep.deliberation)

        # 3. Validate visual assets across assets/sprites, assets/tiles, assets/ui, assets/portraits, assets/backgrounds, or root
        sprite_files = collect_cartridge_visual_asset_files(cdir)
        offending_assets: List[str] = []
        offending_scripts: List[str] = []
        diagnostic_data["asset_metadata"] = _collect_observational_asset_metadata(cdir, sprite_files)

        if sprite_files:
            has_any_multimodal_assets = True
            for img_file in sorted(sprite_files):
                diagnostic_data["sprite_files"] += 1
                rep = self.validate_visual_asset(img_file)
                if not rep.passed:
                    all_errors.extend(rep.errors)
                    try:
                        rel_bad = img_file.relative_to(cdir).as_posix()
                    except Exception:
                        rel_bad = img_file.name
                    if rel_bad not in offending_assets:
                        offending_assets.append(rel_bad)
                deliberation_parts.append(rep.deliberation)

        # 4. Multimodal Contact-Sheet Style & Completeness Critic (when live Gemini client + Game Bible exist)
        if sprite_files and is_real_genai_client(self.gemini_client) and (cdir / "game_bible.json").is_file():
            contact_sheet_p = render_cartridge_asset_contact_sheet(cdir, flagged_assets=set(offending_assets))
            if contact_sheet_p is not None and contact_sheet_p.is_file():
                diagnostic_data["contact_sheet_path"] = str(contact_sheet_p)
                mm_verdict = self._run_multimodal_contact_sheet_audit(cdir, contact_sheet_p, diagnostic_data["asset_metadata"])
                if mm_verdict is not None:
                    deliberation_parts.append(f"Multimodal Contact Sheet Critic: {mm_verdict.deliberation}")
                    if not mm_verdict.all_assets_coherent_and_intact or mm_verdict.offending_assets:
                        added_mm_flag = False
                        for raw_off in mm_verdict.offending_assets:
                            norm_off = str(raw_off or "").replace("res://", "").lstrip("/")
                            if not norm_off.startswith("assets/") and (cdir / "assets" / norm_off).is_file():
                                norm_off = f"assets/{norm_off}"
                            if (cdir / norm_off).is_file() and norm_off not in offending_assets:
                                offending_assets.append(norm_off)
                                added_mm_flag = True
                        for raw_scr in mm_verdict.offending_scripts:
                            norm_scr = str(raw_scr or "").replace("res://", "").lstrip("/")
                            if (cdir / norm_scr).is_file() and norm_scr not in offending_scripts:
                                offending_scripts.append(norm_scr)
                        if added_mm_flag:
                            try:
                                render_cartridge_asset_contact_sheet(
                                    cdir,
                                    output_path=contact_sheet_p,
                                    flagged_assets=set(offending_assets),
                                )
                            except Exception:
                                pass
                        crit_msg = (
                            mm_verdict.remediation_summary.strip()
                            or f"Multimodal asset sheet critic rejected incoherent/placeholder assets: {mm_verdict.offending_assets}"
                        )
                        all_errors.append(f"Multimodal Contact Sheet Audit: {crit_msg}")

        diagnostic_data["offending_assets"] = offending_assets
        diagnostic_data["offending_scripts"] = offending_scripts
        diagnostic_data["violations"] = list(all_errors)

        # If no multimodal assets found in cartridge, pass cleanly
        if not has_any_multimodal_assets:
            return QualityGateResult(
                deliberation="No multimodal assets (narrative.json, audio, sprites) detected in cartridge; Gate B passed.",
                gate_name="Gate B: Multimodal Asset Verification",
                passed=True,
                failure_type=GateFailureType.NONE,
                details="No multimodal assets present to validate.",
                diagnostic_data=diagnostic_data,
            )

        passed = len(all_errors) == 0
        overall_delib = (
            f"Gate B PASSED: All multimodal assets ({diagnostic_data['audio_files']} audio, "
            f"{diagnostic_data['sprite_files']} sprites/tiles, narrative.json={'present' if diagnostic_data['has_narrative'] else 'none'}) "
            f"meet 100% integrity criteria."
            if passed else
            f"Gate B REJECTED: Multimodal asset integrity violations: {'; '.join(all_errors[:5])}"
        )

        return QualityGateResult(
            deliberation=overall_delib,
            gate_name="Gate B: Multimodal Asset Verification",
            passed=passed,
            failure_type=GateFailureType.NONE if passed else GateFailureType.ASSET_INTEGRITY_VIOLATION,
            details="\n".join(all_errors),
            diagnostic_data=diagnostic_data,
        )
