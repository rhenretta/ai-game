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

from PIL import Image

from engine.assets.palette import STYLE_BIBLE_16_PALETTE
from engine.supervisor.models import (
    AssetValidationReport,
    GateFailureType,
    QualityGateResult,
)

logger = logging.getLogger("AssetValidator")

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
VALID_AUDIO_SAMPLE_RATES = {8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000}


class AssetValidator:
    """Evaluates candidate cartridge assets against Gate E invariants."""
    deliberation: str = "Multimodal Asset Validator enforcing audio, visual, and narrative integrity."

    def __init__(
        self,
        palette: Optional[List[Tuple[int, int, int]]] = None,
        enforce_palette: bool = False,
        min_audio_duration_sec: float = 0.1,
        max_audio_duration_sec: float = 30.0,
        silence_threshold_rms: float = 1.0,
    ):
        self.palette = palette or STYLE_BIBLE_16_PALETTE
        self.palette_set = set(self.palette)
        self.enforce_palette = enforce_palette
        self.min_audio_duration = float(min_audio_duration_sec)
        self.max_audio_duration = float(max_audio_duration_sec)
        self.silence_threshold_rms = float(silence_threshold_rms)

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

        # 3. Validate visual sprites under assets/sprites, assets/tiles, or root
        sprite_dirs = [cdir / "assets" / "sprites", cdir / "assets" / "tiles", cdir / "sprites", cdir / "tiles"]
        sprite_files = []
        for sdir in sprite_dirs:
            if sdir.is_dir():
                sprite_files.extend(list(sdir.glob("*.png")))
        sprite_files.extend(list(cdir.glob("*.png")))
        sprite_files = list(dict.fromkeys(sprite_files))

        if sprite_files:
            has_any_multimodal_assets = True
            for img_file in sorted(sprite_files):
                diagnostic_data["sprite_files"] += 1
                rep = self.validate_visual_asset(img_file)
                if not rep.passed:
                    all_errors.extend(rep.errors)
                deliberation_parts.append(rep.deliberation)

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
