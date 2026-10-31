"""tests/test_m4_challenger_adversarial.py
Adversarial empirical stress-test suite for Milestone M4 Gate Verification (Challenger 1).
Exhaustively challenges:
1. Untraversable geometry & dual-frontier BFS bottleneck extraction (Gate B):
   - Unit-level NavMesh traversability & Dual-Frontier BFS bottleneck extraction.
   - Pipeline-level early-return analysis and diagnostic trace properties.
2. Corrupt multimodal asset integrity across audio, visual, and narrative bibles (Gate E):
   - Audio: truncated headers (<44B), non-PCM codes, duration bounds [0.1s, 30.0s], RMS silence.
   - Visual: corrupt 8-byte PNG signature, out-of-bounds dimensions, 16-color palette violations.
   - Narrative: missing on-disk portraits/voices, orphan dialogue nodes, closed 2-node cycles.
3. Self-healing repair exhaustion, max retries enforcement, and hermetic active staging:
   - Exhaustion to FAILED_MAX_RETRIES without modifying cartridges/active/.
   - Rationale-First schema enforcement across all diagnostic models.
   - Empirical vulnerability reproduction (ASTSecurityError handling and asset dir scoping).
Strictly 0 occurrences of forbidden legacy branding tokens.
"""
from __future__ import annotations
import ast
import copy
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import wave
from typing import Any, Dict, List, Optional, Tuple

import pytest
from PIL import Image

from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.assets.palette import STYLE_BIBLE_16_PALETTE, quantize_image_to_indexed
from engine.compiler.compiler import EngineCompiler
from engine.loader import ASTSecurityError, CartridgeLoader, validate_cartridge_ast
from engine.studio.swe import LeadGameplaySWEAgent
from engine.supervisor.asset_validator import AssetValidator, PNG_SIGNATURE
from engine.supervisor.models import (
    AssetValidationReport,
    BuildEvaluationReport,
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    RemedyType,
    SupervisorGateVerdict,
)
from engine.supervisor.supervisor import BuildSupervisor


# =============================================================================
# Helper Fixtures & Builders
# =============================================================================

@pytest.fixture
def supervisor_suite(tmp_path: Path) -> Tuple[BuildSupervisor, Path, Path]:
    """Sets up an isolated supervisor testing environment with active and candidate dirs."""
    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    cartridges_root.mkdir(parents=True, exist_ok=True)
    active_dir.mkdir(parents=True, exist_ok=True)

    compiler = EngineCompiler(
        cartridges_root=cartridges_root,
        active_cartridge_dir=active_dir,
        specs_dir=tmp_path / "specs",
    )
    swe = LeadGameplaySWEAgent()
    validator = AssetValidator()

    supervisor = BuildSupervisor(
        cartridges_root=cartridges_root,
        active_cartridge_dir=active_dir,
        compiler=compiler,
        swe_agent=swe,
        asset_validator=validator,
        tile_size=32.0,
        max_retries=3,
        early_death_tick_threshold=180,
    )
    return supervisor, cartridges_root, active_dir


def create_minimal_valid_candidate(dest: Path) -> Path:
    """Builds a complete, 100% valid cartridge candidate in dest directory."""
    dest.mkdir(parents=True, exist_ok=True)

    (dest / "manifest.json").write_text(json.dumps({
        "cartridge_id": "cand_challenger_01",
        "title": "Adversarial Candidate",
        "version": "1.0.0",
        "entrypoint": "mechanics.py",
        "author": "Autonomous Studio Swarm",
    }, indent=2), encoding="utf-8")

    (dest / "entities.py").write_text('''"""entities.py"""
from typing import Any, Optional
from engine.api.contracts import BasePlayer, BaseEnemy, BaseHazard

class MockPlayer(BasePlayer):
    deliberation: str = "Mock player entity for testing."
    def __init__(self, x: float = 120.0, y: float = 360.0, telemetry_recorder: Optional[Any] = None):
        super().__init__(x=x, y=y, telemetry_recorder=telemetry_recorder)

class MockEnemy(BaseEnemy):
    deliberation: str = "Mock enemy entity for testing."
    def __init__(self, enemy_id: str = "e1", x: float = 600.0, y: float = 360.0):
        super().__init__(enemy_id=enemy_id, enemy_type="chaser", x=x, y=y, width=32.0, height=32.0, speed=100.0, health=50.0, damage=10.0)

class MockHazard(BaseHazard):
    deliberation: str = "Mock hazard entity for testing."
    def __init__(self, hazard_id: str = "h1", x: float = 400.0, y: float = 360.0):
        super().__init__(hazard_id=hazard_id, hazard_type="spike", x=x, y=y, width=32.0, height=32.0, damage=10.0)
''', encoding="utf-8")

    (dest / "mechanics.py").write_text('''"""mechanics.py"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from engine.api.contracts import BaseCartridge, BaseLevelSpec, BasePlayer, BaseEnemy, BaseHazard
from entities import MockPlayer, MockEnemy, MockHazard

class CandidateCartridge(BaseCartridge):
    deliberation: str = "Candidate cartridge implementing BaseCartridge for testing."
    def __init__(self, manifest: Optional[Dict[str, Any]] = None, cartridge_dir: Optional[Any] = None, narrative: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest, cartridge_dir=cartridge_dir, narrative=narrative)

    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        cdir = Path(self.cartridge_dir) if self.cartridge_dir else Path(".")
        level_file = cdir / "level.json"
        if level_file.exists():
            return BaseLevelSpec.from_dict(json.loads(level_file.read_text(encoding="utf-8")))
        return BaseLevelSpec(
            level_id="test_level",
            name="Testing Grounds",
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            walls=[
                (0.0, 0.0, 1280.0, 32.0),
                (0.0, 688.0, 1280.0, 32.0),
                (0.0, 0.0, 32.0, 720.0),
                (1248.0, 0.0, 32.0, 720.0),
            ],
            enemies=[],
            hazards=[],
        )

    def create_player(self, x: float = 120.0, y: float = 360.0, telemetry_recorder: Optional[Any] = None) -> BasePlayer:
        p = MockPlayer(x, y)
        p.telemetry_recorder = telemetry_recorder
        return p

    def create_enemies(self, spec: Optional[BaseLevelSpec] = None) -> List[BaseEnemy]:
        return []

    def create_hazards(self, spec: Optional[BaseLevelSpec] = None) -> List[BaseHazard]:
        return []
''', encoding="utf-8")

    # Valid level.json
    level_data = {
        "deliberation": "Minimal valid traversable test level.",
        "level_id": "test_level",
        "name": "Testing Grounds",
        "width": 1280.0,
        "height": 720.0,
        "spawn_point": [120.0, 360.0],
        "exit_point": [1160.0, 360.0],
        "walls": [
            [0.0, 0.0, 1280.0, 32.0],
            [0.0, 688.0, 1280.0, 32.0],
            [0.0, 0.0, 32.0, 720.0],
            [1248.0, 0.0, 32.0, 720.0],
        ],
        "enemies": [],
        "hazards": [],
    }
    (dest / "level.json").write_text(json.dumps(level_data, indent=2), encoding="utf-8")

    # Valid Gate E audio asset
    audio_dir = dest / "assets" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / "valid_voice.wav"
    sample_rate = 22050
    duration = 1.0
    total_samples = int(sample_rate * duration)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        raw = bytearray()
        for i in range(total_samples):
            val = int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            raw.extend(struct.pack("<h", val))
        wf.writeframes(raw)

    # Valid Gate E sprite asset
    sprite_dir = dest / "assets" / "sprites"
    sprite_dir.mkdir(parents=True, exist_ok=True)
    img_path = sprite_dir / "valid_sprite.png"
    img = Image.new("RGB", (32, 32), color=STYLE_BIBLE_16_PALETTE[0])
    img.putpixel((16, 16), STYLE_BIBLE_16_PALETTE[4])
    img.save(img_path, format="PNG")

    # Valid Gate E narrative.json
    narrative_data = {
        "deliberation": "Valid narrative bible.",
        "world": {"world_title": "Test Domain", "premise": "Adversarial test harness."},
        "characters": [
            {"id": "hero", "name": "Protagonist", "portrait": "assets/sprites/valid_sprite.png"}
        ],
        "dialogues": [
            {"bark_id": "b1", "text": "Ready.", "voice_clip": "assets/audio/valid_voice.wav"}
        ],
        "dialogue_trees": [
            {
                "root_node_id": "root",
                "nodes": {
                    "root": {
                        "text": "Start node",
                        "choices": [{"text": "Continue", "target_node_id": "end_node"}]
                    },
                    "end_node": {
                        "text": "Terminal node",
                        "choices": []
                    }
                }
            }
        ]
    }
    (dest / "narrative.json").write_text(json.dumps(narrative_data, indent=2), encoding="utf-8")
    return dest


# =============================================================================
# Area 1: Untraversable Geometry & Dual-Frontier Bottleneck Extraction
# =============================================================================

class TestUntraversableGeometryDualFrontier:
    """Adversarially tests level traversability failures and bottleneck extraction at unit and pipeline levels."""

    def test_direct_gate_b_vertical_partition_extracts_bottleneck_chokepoints(self, supervisor_suite: Any, tmp_path: Path):
        """Verifies that direct geometric Gate B verification (sim_result=None) executes Dual-Frontier BFS
        and accurately identifies the solid vertical partition wall separating spawn from exit.
        """
        supervisor, _, _ = supervisor_suite
        candidate_dir = tmp_path / "cand_vertical_direct"
        create_minimal_valid_candidate(candidate_dir)

        # Insert vertical solid wall slicing from y=0 to y=720 at x=640 (column 20)
        level_file = candidate_dir / "level.json"
        data = json.loads(level_file.read_text(encoding="utf-8"))
        data["walls"].append([640.0, 0.0, 32.0, 720.0])
        level_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        cartridge = CartridgeLoader.load_cartridge(candidate_dir)
        level_spec = cartridge.get_level_spec(1)

        gate_b, blocked_nodes = supervisor.verify_gate_b_traversability(cartridge, level_spec, sim_result=None)

        assert gate_b.passed is False
        assert gate_b.failure_type == GateFailureType.CRITICAL_PATH_BLOCKED
        assert "No traversable path from spawn" in gate_b.details
        assert len(blocked_nodes) > 0

        # The partition wall is at column 640 // 32 = 20.
        partition_tx = 640 // 32
        for bx, by in blocked_nodes:
            assert bx == partition_tx, f"Expected blocked node on partition x={partition_tx}, got ({bx}, {by})"

    def test_direct_gate_b_horizontal_partition_extracts_bottleneck_chokepoints(self, supervisor_suite: Any, tmp_path: Path):
        """Verifies that direct geometric Gate B verification identifies horizontal partition obstacles."""
        supervisor, _, _ = supervisor_suite
        candidate_dir = tmp_path / "cand_horizontal_direct"
        create_minimal_valid_candidate(candidate_dir)

        level_file = candidate_dir / "level.json"
        data = json.loads(level_file.read_text(encoding="utf-8"))
        data["spawn_point"] = [640.0, 100.0]
        data["exit_point"] = [640.0, 600.0]
        # Solid horizontal wall from x=0 to x=1280 at y=352 (row 11)
        data["walls"].append([0.0, 352.0, 1280.0, 32.0])
        level_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        cartridge = CartridgeLoader.load_cartridge(candidate_dir)
        level_spec = cartridge.get_level_spec(1)

        gate_b, blocked_nodes = supervisor.verify_gate_b_traversability(cartridge, level_spec, sim_result=None)

        assert gate_b.passed is False
        assert len(blocked_nodes) > 0
        partition_ty = 352 // 32
        for bx, by in blocked_nodes:
            assert by == partition_ty, f"Expected blocked node on partition y={partition_ty}, got ({bx}, {by})"

    def test_direct_gate_b_spawn_inside_solid_wall(self, supervisor_suite: Any, tmp_path: Path):
        """Verifies that direct geometric Gate B verification detects player spawn embedded in solid obstacle."""
        supervisor, _, _ = supervisor_suite
        candidate_dir = tmp_path / "cand_spawn_direct"
        create_minimal_valid_candidate(candidate_dir)

        level_file = candidate_dir / "level.json"
        data = json.loads(level_file.read_text(encoding="utf-8"))
        # Spawn is at (120, 360). Encase in wall at (96, 352, 64, 64) -> tile (3, 11)
        data["walls"].append([96.0, 352.0, 64.0, 64.0])
        level_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        cartridge = CartridgeLoader.load_cartridge(candidate_dir)
        level_spec = cartridge.get_level_spec(1)

        gate_b, blocked_nodes = supervisor.verify_gate_b_traversability(cartridge, level_spec, sim_result=None)

        assert gate_b.passed is False
        assert "Spawn point" in gate_b.details
        assert "overlaps solid obstacle tile" in gate_b.details
        spawn_tx, spawn_ty = int(120 // 32), int(360 // 32)
        assert (spawn_tx, spawn_ty) in blocked_nodes

    def test_direct_gate_b_exit_inside_solid_wall(self, supervisor_suite: Any, tmp_path: Path):
        """Verifies that direct geometric Gate B verification detects exit portal embedded in solid obstacle."""
        supervisor, _, _ = supervisor_suite
        candidate_dir = tmp_path / "cand_exit_direct"
        create_minimal_valid_candidate(candidate_dir)

        level_file = candidate_dir / "level.json"
        data = json.loads(level_file.read_text(encoding="utf-8"))
        # Exit is at (1160, 360). Encase in wall at (1136, 352, 64, 64) -> tile (36, 11)
        data["walls"].append([1136.0, 352.0, 64.0, 64.0])
        level_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        cartridge = CartridgeLoader.load_cartridge(candidate_dir)
        level_spec = cartridge.get_level_spec(1)

        gate_b, blocked_nodes = supervisor.verify_gate_b_traversability(cartridge, level_spec, sim_result=None)

        assert gate_b.passed is False
        assert "Exit portal" in gate_b.details
        assert "overlaps solid obstacle tile" in gate_b.details
        exit_tx, exit_ty = int(1160 // 32), int(360 // 32)
        assert (exit_tx, exit_ty) in blocked_nodes

    def test_pipeline_gate_b_early_return_bypasses_dual_frontier_bottlenecks(self, supervisor_suite: Any, tmp_path: Path):
        """Verifies that Gate B Dual-Frontier BFS bottleneck extraction executes during
        pipeline evaluate_candidate_cartridge even when playtester returns traversable=False without blocked_nodes.
        """
        supervisor, _, _ = supervisor_suite
        candidate_dir = tmp_path / "cand_pipeline_bypass"
        create_minimal_valid_candidate(candidate_dir)

        # Insert vertical solid partition wall at x=640
        level_file = candidate_dir / "level.json"
        data = json.loads(level_file.read_text(encoding="utf-8"))
        data["walls"].append([640.0, 0.0, 32.0, 720.0])
        level_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        gate_results, trace = supervisor.evaluate_candidate_cartridge(candidate_dir)

        # Gate B is rejected
        assert gate_results["gate_b"].passed is False
        assert trace is not None
        assert trace.failure_type == GateFailureType.CRITICAL_PATH_BLOCKED

        # Verification: Dual-Frontier BFS fell through and populated blocked_nodes and failure_coordinates
        assert len(trace.blocked_nodes) > 0, (
            "trace.blocked_nodes should be populated by Dual-Frontier BFS when sim_result.blocked_nodes is empty."
        )
        partition_tx = 640 // 32
        for bx, by in trace.blocked_nodes:
            assert bx == partition_tx, f"Expected blocked node on partition x={partition_tx}, got ({bx}, {by})"
        assert trace.failure_coordinates is not None
        assert trace.failure_coordinates[0] == (partition_tx + 0.5) * 32.0


# =============================================================================
# Area 2: Corrupt Multimodal Asset Integrity (Gate E)
# =============================================================================

class TestGateEMultimodalAssetIntegrity:
    """Adversarially tests audio, visual, and narrative corruption detection in Gate E."""

    # -------------------------------------------------------------------------
    # Audio Stress Testing
    # -------------------------------------------------------------------------
    def test_audio_truncated_headers_under_44_bytes(self, tmp_path: Path):
        validator = AssetValidator()
        for size in [0, 8, 20, 36, 43]:
            corrupt_wav = tmp_path / f"truncated_{size}.wav"
            corrupt_wav.write_bytes(b"\x00" * size)
            rep = validator.validate_voice_audio(corrupt_wav)
            assert rep.passed is False
            assert any("< 44 bytes" in err for err in rep.errors)

    def test_audio_invalid_riff_or_wave_magic(self, tmp_path: Path):
        validator = AssetValidator()

        # Non-RIFF magic
        data_bad_riff = b"FORM" + b"\x00" * 40
        wav_bad_riff = tmp_path / "bad_riff.wav"
        wav_bad_riff.write_bytes(data_bad_riff)
        rep1 = validator.validate_voice_audio(wav_bad_riff)
        assert rep1.passed is False
        assert any("Invalid RIFF magic bytes" in err for err in rep1.errors)

        # Valid RIFF but non-WAVE marker
        data_bad_wave = struct.pack("<4sI4s", b"RIFF", 40, b"AVI ") + b"\x00" * 32
        wav_bad_wave = tmp_path / "bad_wave.wav"
        wav_bad_wave.write_bytes(data_bad_wave)
        rep2 = validator.validate_voice_audio(wav_bad_wave)
        assert rep2.passed is False
        assert any("Invalid WAVE format marker" in err for err in rep2.errors)

    def test_audio_non_pcm_format_codes(self, tmp_path: Path):
        """Tests that non-PCM format codes (e.g. 3=IEEE Float, 6=A-Law, 7=Mu-Law) are rejected."""
        validator = AssetValidator()

        for fmt_code in [2, 3, 6, 7, 0xFFFE]:
            sample_rate = 22050
            channels = 1
            bits_per_sample = 16
            byte_rate = sample_rate * channels * (bits_per_sample // 8)
            block_align = channels * (bits_per_sample // 8)

            fmt_data = struct.pack("<HHIIHH", fmt_code, channels, sample_rate, byte_rate, block_align, bits_per_sample)
            data_subchunk = b"\x00" * 22050
            riff_size = 4 + (8 + len(fmt_data)) + (8 + len(data_subchunk))

            header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", len(fmt_data))
            header += fmt_data
            header += struct.pack("<4sI", b"data", len(data_subchunk))

            wav_file = tmp_path / f"fmt_{fmt_code}.wav"
            wav_file.write_bytes(header + data_subchunk)

            rep = validator.validate_voice_audio(wav_file)
            assert rep.passed is False
            assert any(f"Unsupported audio format tag {fmt_code}" in err for err in rep.errors)

    def test_audio_duration_bounds_out_of_range(self, tmp_path: Path):
        validator = AssetValidator(min_audio_duration_sec=0.1, max_audio_duration_sec=30.0)

        # Duration < 0.1s (e.g. 0.05s = 1102 samples at 22050Hz)
        short_wav = tmp_path / "too_short.wav"
        with wave.open(str(short_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x50\x00" * 1102)

        rep_short = validator.validate_voice_audio(short_wav)
        assert rep_short.passed is False
        assert any("too short" in err for err in rep_short.errors)

        # Duration > 30.0s (e.g. 31.0s = 683550 samples at 22050Hz)
        long_wav = tmp_path / "too_long.wav"
        with wave.open(str(long_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x50\x00" * (22050 * 31))

        rep_long = validator.validate_voice_audio(long_wav)
        assert rep_long.passed is False
        assert any("too long" in err for err in rep_long.errors)

    def test_audio_zero_amplitude_and_rms_silence(self, tmp_path: Path):
        validator = AssetValidator(silence_threshold_rms=1.0)

        # 100% zero amplitude
        silent_wav = tmp_path / "total_silence.wav"
        with wave.open(str(silent_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x00" * 22050)

        rep_silent = validator.validate_voice_audio(silent_wav)
        assert rep_silent.passed is False
        assert any("Total silence detected" in err for err in rep_silent.errors)

    # -------------------------------------------------------------------------
    # Visual Asset Stress Testing
    # -------------------------------------------------------------------------
    def test_png_corrupted_signature_rejected(self, tmp_path: Path):
        validator = AssetValidator()
        bad_png = tmp_path / "bad_magic.png"
        bad_png.write_bytes(b"GIF89a\x00\x00\x00\x00corruptdata")

        rep = validator.validate_visual_asset(bad_png)
        assert rep.passed is False
        assert any("valid 8-byte PNG signature" in err for err in rep.errors)

    def test_png_out_of_bounds_dimensions_rejected(self, tmp_path: Path):
        validator = AssetValidator()

        # Below min dimension (4x4 < 8px)
        tiny_png = tmp_path / "tiny.png"
        img_tiny = Image.new("RGB", (4, 4), color=STYLE_BIBLE_16_PALETTE[0])
        img_tiny.save(tiny_png, format="PNG")

        rep_tiny = validator.validate_visual_asset(tiny_png)
        assert rep_tiny.passed is False
        assert any("below minimum 8px" in err for err in rep_tiny.errors)

        # Above max dimension (8200x16 > 8192px)
        huge_png = tmp_path / "huge.png"
        img_huge = Image.new("RGB", (8200, 16), color=STYLE_BIBLE_16_PALETTE[0])
        img_huge.save(huge_png, format="PNG")

        rep_huge = validator.validate_visual_asset(huge_png)
        assert rep_huge.passed is False
        assert any("outside permissible bounds" in err for err in rep_huge.errors)

    def test_png_non_palette_colors_with_offending_coordinates(self, tmp_path: Path):
        validator = AssetValidator(enforce_palette=True)

        # Create 32x32 image with 1 pixel contaminated by forbidden non-Style Bible color
        bad_color = (255, 20, 147)  # DeepPink - not in STYLE_BIBLE_16_PALETTE
        assert bad_color not in validator.palette_set

        bad_png = tmp_path / "unquantized.png"
        img = Image.new("RGB", (32, 32), color=STYLE_BIBLE_16_PALETTE[0])
        img.putpixel((7, 13), bad_color)
        img.save(bad_png, format="PNG")

        rep = validator.validate_visual_asset(bad_png)
        assert rep.passed is False
        assert any("Style Bible palette violation" in err for err in rep.errors)
        # Verify exact offending color coordinates are returned
        assert "pos': (7, 13)" in rep.errors[0] or "(7, 13)" in str(rep.errors)
        assert str(bad_color) in str(rep.errors)

    # -------------------------------------------------------------------------
    # Narrative Bible Stress Testing
    # -------------------------------------------------------------------------
    def test_narrative_missing_on_disk_portrait_rejected(self, tmp_path: Path):
        validator = AssetValidator()
        c_dir = tmp_path / "cand_missing_portrait"
        create_minimal_valid_candidate(c_dir)

        # Delete the portrait file on disk
        portrait_file = c_dir / "assets" / "sprites" / "valid_sprite.png"
        portrait_file.unlink()

        rep = validator.validate_narrative_bible(c_dir)
        assert rep.passed is False
        assert any("references missing portrait file on disk" in err for err in rep.errors)

    def test_narrative_missing_on_disk_voice_audio_rejected(self, tmp_path: Path):
        validator = AssetValidator()
        c_dir = tmp_path / "cand_missing_voice"
        create_minimal_valid_candidate(c_dir)

        # Delete the voice file on disk
        voice_file = c_dir / "assets" / "audio" / "valid_voice.wav"
        voice_file.unlink()

        rep = validator.validate_narrative_bible(c_dir)
        assert rep.passed is False
        assert any("references missing audio file on disk" in err for err in rep.errors)

    def test_narrative_disconnected_orphan_dialogue_nodes_rejected(self, tmp_path: Path):
        validator = AssetValidator()
        c_dir = tmp_path / "cand_orphan_nodes"
        create_minimal_valid_candidate(c_dir)

        narrative_file = c_dir / "narrative.json"
        data = json.loads(narrative_file.read_text(encoding="utf-8"))
        # Add orphan nodes unreachable from root
        data["dialogue_trees"][0]["nodes"]["orphan_secret_node"] = {
            "text": "Lost dialogue cue that is unreachable",
            "choices": []
        }
        narrative_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        rep = validator.validate_narrative_bible(c_dir)
        assert rep.passed is False
        assert any("Orphan dialogue nodes unreachable from root" in err for err in rep.errors)
        assert "orphan_secret_node" in str(rep.errors)

    def test_narrative_closed_infinite_cycle_rejected(self, tmp_path: Path):
        validator = AssetValidator()
        c_dir = tmp_path / "cand_infinite_cycle"
        create_minimal_valid_candidate(c_dir)

        narrative_file = c_dir / "narrative.json"
        data = json.loads(narrative_file.read_text(encoding="utf-8"))
        # Create a closed 2-node cycle without escape
        data["dialogue_trees"] = [
            {
                "root_node_id": "loop_a",
                "nodes": {
                    "loop_a": {
                        "text": "Endless loop A",
                        "choices": [{"text": "Go to B", "target_node_id": "loop_b"}]
                    },
                    "loop_b": {
                        "text": "Endless loop B",
                        "choices": [{"text": "Go to A", "target_node_id": "loop_a"}]
                    }
                }
            }
        ]
        narrative_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        rep = validator.validate_narrative_bible(c_dir)
        assert rep.passed is False
        assert any("Closed 2-node infinite loop detected" in err for err in rep.errors)


# =============================================================================
# Area 3: Self-Healing Repair & Retries Exhaustion
# =============================================================================

class TestSelfHealingExhaustionAndHermeticStaging:
    """Adversarially tests max retries exhaustion, FAILED_MAX_RETRIES, and active cartridge isolation."""

    def test_unrepairable_defect_exhausts_max_retries_without_modifying_active(
        self, supervisor_suite: Any, tmp_path: Path
    ):
        """Creates an unfixable cartridge candidate (corrupt audio in subfolder that swe.py cannot heal)
        and verifies that evaluate_and_heal:
        1. Exhausts exactly max_retries attempts.
        2. Returns status FAILED_MAX_RETRIES.
        3. Sets promoted_to_active to False.
        4. Leaves cartridges/active/ completely untouched.
        """
        supervisor, cartridges_root, active_dir = supervisor_suite

        # Seed active directory with a known canary file
        active_canary = active_dir / "active_canary.txt"
        canary_content = "ORIGINAL_ACTIVE_VERSION_CANARY_HASH_892348"
        active_canary.write_text(canary_content, encoding="utf-8")
        active_manifest = active_dir / "manifest.json"
        active_manifest.write_text(json.dumps({"version": "0.1.0", "status": "active_stable"}), encoding="utf-8")

        # Create candidate with an unrepairable asset defect (missing referenced portrait)
        candidate_dir = tmp_path / "cand_unrepairable"
        create_minimal_valid_candidate(candidate_dir)
        n_file = candidate_dir / "narrative.json"
        n_data = json.loads(n_file.read_text(encoding="utf-8"))
        n_data["characters"][0]["portrait"] = "assets/sprites/nonexistent_portrait.png"
        n_file.write_text(json.dumps(n_data, indent=2), encoding="utf-8")

        report: BuildEvaluationReport = supervisor.evaluate_and_heal(
            candidate_or_spec=candidate_dir,
            max_retries=3,
            episode_number=1,
        )

        # 1. Status must be FAILED_MAX_RETRIES
        assert report.status == SupervisorGateVerdict.FAILED_MAX_RETRIES
        assert report.promoted_to_active is False
        assert report.attempt_count == 3
        assert report.max_retries == 3
        assert len(report.repair_history) == 3

        # 2. Gate E must be the failing gate across all attempts
        assert report.rejection_trace is not None
        assert report.rejection_trace.failure_gate == "Gate E: Multimodal Asset Integrity"
        assert report.rejection_trace.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION

        # 3. CRITICAL: cartridges/active/ must remain 100% intact and untouched
        assert active_canary.is_file(), "Canary file in cartridges/active/ was deleted!"
        assert active_canary.read_text(encoding="utf-8") == canary_content, "Canary file content was altered!"
        active_data = json.loads(active_manifest.read_text(encoding="utf-8"))
        assert active_data["version"] == "0.1.0", "Active manifest was modified on failed promotion!"

    def test_rejection_trace_rationale_first_contract_under_exhaustion(
        self, supervisor_suite: Any, tmp_path: Path
    ):
        """Verifies that every trace in repair_history complies with Rationale-First schema."""
        supervisor, _, _ = supervisor_suite
        candidate_dir = tmp_path / "cand_trace_contract"
        create_minimal_valid_candidate(candidate_dir)

        # Truncate audio and disable swe_agent to guarantee clean exhaustion without crash
        supervisor.swe_agent = None
        (candidate_dir / "assets" / "audio" / "valid_voice.wav").write_bytes(b"SHORT")

        report = supervisor.evaluate_and_heal(candidate_dir, max_retries=2)
        assert report.status == SupervisorGateVerdict.FAILED_MAX_RETRIES
        assert len(report.repair_history) == 2

        for trace in report.repair_history:
            assert isinstance(trace, RejectionDiagnosticTrace)
            assert hasattr(trace, "deliberation")
            assert len(trace.deliberation.strip()) > 0
            # Verify deliberation is the first field declared in Pydantic schema
            fields = list(trace.__class__.model_fields.keys())
            assert fields[0] == "deliberation", f"Expected 'deliberation' at index 0, found {fields[0]}"

    def test_self_healing_succeeds_when_defect_is_repaired(
        self, supervisor_suite: Any, tmp_path: Path
    ):
        """Verifies that if a defect is repairable (e.g. interior blocking wall), self-healing succeeds
        and promotes cleanly on attempt 2 without reaching max_retries.
        """
        supervisor, _, active_dir = supervisor_suite
        candidate_dir = tmp_path / "cand_repairable_wall"
        create_minimal_valid_candidate(candidate_dir)

        # Add single interior blocking wall that fallback healer removes
        level_file = candidate_dir / "level.json"
        data = json.loads(level_file.read_text(encoding="utf-8"))
        data["walls"].append([640.0, 0.0, 32.0, 720.0])
        level_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        report = supervisor.evaluate_and_heal(candidate_dir, max_retries=3)

        assert report.status == SupervisorGateVerdict.PROMOTED
        assert report.promoted_to_active is True
        assert report.attempt_count == 2
        assert len(report.repair_history) == 1
        assert (active_dir / "manifest.json").is_file()
        assert (active_dir / "level.json").is_file()


# =============================================================================
# Area 4: Empirical Vulnerability Reproduction & Edge Cases
# =============================================================================

class TestEmpiricalVulnerabilitiesAndDiscrepancies:
    """Isolates and proves specific bugs and discrepancies discovered during adversarial review."""

    def test_reproduce_swe_verify_code_module_uncaught_ast_security_error(self):
        """Verifies that LeadGameplaySWEAgent.verify_code_module handles ASTSecurityError cleanly
        with raise_on_ast_error=False (returning False, err_msg) while preserving default exception propagation.
        """
        swe = LeadGameplaySWEAgent()
        prohibited_code = "import subprocess\nsubprocess.run(['ls'])\n"

        # Verify that validate_cartridge_ast actually raises ASTSecurityError
        with pytest.raises(ASTSecurityError):
            validate_cartridge_ast(prohibited_code)

        # Verify default behavior propagates ASTSecurityError (M3 backward compatibility)
        with pytest.raises(ASTSecurityError):
            swe.verify_code_module("test_module.py", prohibited_code)

        # Verify raise_on_ast_error=False catches ASTSecurityError and returns a clean failure tuple
        valid, err_msg = swe.verify_code_module("test_module.py", prohibited_code, raise_on_ast_error=False)
        assert valid is False
        assert "ASTSecurityError" in err_msg

    def test_reproduce_swe_repair_asset_subdirectory_scoping_defect(self, tmp_path: Path):
        """Verifies that LeadGameplaySWEAgent.repair_cartridge_code searches asset subdirectories
        (assets/sprites/ and assets/audio/) via rglob, repairing assets in standard candidate subfolders.
        """
        cand = tmp_path / "cand_subfolder_defect"
        create_minimal_valid_candidate(cand)

        # Contaminate the sprite in assets/sprites/ with non-palette pixel
        sprite_file = cand / "assets" / "sprites" / "valid_sprite.png"
        bad_color = (255, 20, 147)
        img = Image.new("RGB", (32, 32), color=bad_color)
        img.save(sprite_file, format="PNG")

        trace = RejectionDiagnosticTrace(
            deliberation="Palette violation",
            candidate_id="cand_subfolder_defect",
            failure_gate="Gate E: Multimodal Asset Integrity",
            failure_type=GateFailureType.ASSET_INTEGRITY_VIOLATION,
            failure_reason="Palette violation",
            remedy_type=RemedyType.REGENERATE_ASSET,
        )

        swe = LeadGameplaySWEAgent()
        repaired = swe.repair_cartridge_code(cand, trace)

        # Verify that valid_sprite.png was repaired in the subdirectory
        assert "valid_sprite.png" in repaired
        img_check = Image.open(sprite_file)
        assert img_check.convert("RGB").getpixel((0, 0)) in STYLE_BIBLE_16_PALETTE

    def test_evaluate_and_heal_survives_swe_repair_exception(self, supervisor_suite: Any, tmp_path: Path):
        """Verifies that evaluate_and_heal wraps swe_agent.repair_cartridge_code in try-except,
        preventing unexpected repair exceptions from crashing the supervisor retry loop.
        """
        supervisor, cartridges_root, active_dir = supervisor_suite
        candidate_dir = tmp_path / "cand_swe_crash"
        create_minimal_valid_candidate(candidate_dir)

        # Make candidate fail Gate B
        level_file = candidate_dir / "level.json"
        data = json.loads(level_file.read_text(encoding="utf-8"))
        data["walls"].append([640.0, 0.0, 32.0, 720.0])
        level_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Mock swe_agent.repair_cartridge_code to raise an unhandled exception
        class CrashingSWEAgent(LeadGameplaySWEAgent):
            def repair_cartridge_code(self, c_dir: Path, trace: RejectionDiagnosticTrace) -> Dict[str, Path]:
                raise RuntimeError("Simulated unexpected crash during SWE code repair")

        supervisor.swe_agent = CrashingSWEAgent()

        report = supervisor.evaluate_and_heal(candidate_dir, max_retries=2)
        assert report.status == SupervisorGateVerdict.FAILED_MAX_RETRIES
        assert report.promoted_to_active is False
        assert report.attempt_count == 2


# =============================================================================
# Area 5: Exhaustive 5-Gate Failure Modes for Open-Genre Cartridges
# =============================================================================

def create_valid_open_genre_candidate(dest: Path, probe_name: str = "test_probe") -> Path:
    """Helper building a 100% valid open-genre candidate cartridge that satisfies all 5 gates."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(json.dumps({
        "cartridge_id": "cand_open_genre_adv",
        "title": "Open Genre Adversarial Candidate",
        "version": "1.0.0",
        "entrypoint": "mechanics.py",
        "author": "Autonomous Studio Swarm",
    }, indent=2), encoding="utf-8")

    (dest / "mechanics.py").write_text(f'''"""mechanics.py"""
from typing import Any, Dict, Optional
from engine.api.contracts import BaseCartridge, BaseScene
from engine.telemetry.probes import AgentTelemetryProbe

class ValidScene(BaseScene):
    deliberation: str = "Valid open-genre test scene."

    def __init__(self, engine: Any):
        super().__init__(engine)
        self.state_counter = 0
        self.probe = AgentTelemetryProbe(
            recorder=getattr(engine, "telemetry_recorder", None),
            probe_name="{probe_name}",
            category="test",
        )

    def on_enter(self, context: Optional[Dict[str, Any]] = None) -> None:
        self.is_active = True

    def on_exit(self) -> Dict[str, Any]:
        self.is_active = False
        return {{}}

    def handle_event(self, event: Any) -> None:
        self.state_counter += 1

    def update(self, dt: float) -> None:
        self.state_counter += 1
        if self.probe and self.state_counter % 5 == 0:
            self.probe.emit(pos=(100.0, 100.0), data={{"tick": self.state_counter}})

    def render(self, surface: Any) -> None:
        pass


class ValidCartridge(BaseCartridge):
    deliberation: str = "Valid open-genre cartridge implementing scene lifecycle contracts."

    def get_scenes(self, engine: Any) -> Dict[str, BaseScene]:
        return {{"main": ValidScene(engine)}}

    def get_initial_scene(self, engine: Any) -> BaseScene:
        return self.get_scenes(engine)["main"]
''', encoding="utf-8")

    audio_dir = dest / "assets" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / "voice.wav"
    sample_rate = 22050
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        raw = bytearray()
        for i in range(sample_rate // 2):
            val = int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            raw.extend(struct.pack("<h", val))
        wf.writeframes(raw)

    sprite_dir = dest / "assets" / "sprites"
    sprite_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=STYLE_BIBLE_16_PALETTE[0])
    img.save(sprite_dir / "sprite.png", format="PNG")

    (dest / "narrative.json").write_text(json.dumps({
        "world": {"world_title": "Test Realm"},
        "characters": [{"id": "hero", "name": "Protagonist", "portrait": "assets/sprites/sprite.png"}],
        "dialogues": [{"bark_id": "b1", "text": "Ready.", "voice_clip": "assets/audio/voice.wav"}],
        "dialogue_trees": [
            {
                "root_node_id": "n1",
                "nodes": {
                    "n1": {"text": "Hello", "choices": [{"text": "Continue", "target_node_id": "n2"}]},
                    "n2": {"text": "Bye", "choices": []}
                }
            }
        ]
    }, indent=2), encoding="utf-8")

    (dest / "telemetry_manifest.json").write_text(json.dumps({
        "probes": [{"probe_name": probe_name, "category": "test"}]
    }, indent=2), encoding="utf-8")

    return dest


class TestFiveGateOpenGenreFailureModes:
    """Exhaustively stress-tests individual failure modes across all 5 Quality Gates."""

    def test_gate_a_syntax_error_rejected(self, tmp_path: Path):
        """Gate A: Verifies invalid Python syntax is rejected with SYNTAX_AST_ERROR."""
        cand = tmp_path / "cand_syntax_err"
        create_valid_open_genre_candidate(cand)
        (cand / "mechanics.py").write_text("class BrokenSyntax def:\n    pass\n", encoding="utf-8")

        supervisor = BuildSupervisor()
        gates, trace = supervisor.evaluate_candidate_cartridge(cand)

        assert trace is not None
        assert trace.failure_gate == "Gate A: AST & Entrypoint"
        assert trace.failure_type == GateFailureType.SYNTAX_AST_ERROR
        assert gates["gate_a"].passed is False
        assert "syntax error" in trace.failure_reason.lower()

    def test_gate_a_banned_subprocess_import_rejected(self, tmp_path: Path):
        """Gate A: Verifies banned imports (subprocess) trigger ASTSecurityError & SYNTAX_AST_ERROR."""
        cand = tmp_path / "cand_banned_import"
        create_valid_open_genre_candidate(cand)
        content = (cand / "mechanics.py").read_text(encoding="utf-8")
        (cand / "mechanics.py").write_text("import subprocess\n" + content, encoding="utf-8")

        supervisor = BuildSupervisor()
        gates, trace = supervisor.evaluate_candidate_cartridge(cand)

        assert trace is not None
        assert trace.failure_gate == "Gate A: AST & Entrypoint"
        assert trace.failure_type == GateFailureType.SYNTAX_AST_ERROR
        assert gates["gate_a"].passed is False
        assert "AST Security Invariant Violated" in trace.failure_reason or "subprocess" in trace.failure_reason

    def test_gate_a_missing_base_cartridge_subclass_rejected(self, tmp_path: Path):
        """Gate A: Verifies omission of BaseCartridge subclass triggers ENTRYPOINT_MISSING."""
        cand = tmp_path / "cand_missing_cartridge"
        create_valid_open_genre_candidate(cand)
        (cand / "mechanics.py").write_text('''"""mechanics.py"""
class NonCartridge:
    deliberation: str = "Class that does not inherit from BaseCartridge."
    def __init__(self):
        pass
''', encoding="utf-8")

        supervisor = BuildSupervisor()
        gates, trace = supervisor.evaluate_candidate_cartridge(cand)

        assert trace is not None
        assert trace.failure_gate == "Gate A: AST & Entrypoint"
        assert trace.failure_type == GateFailureType.ENTRYPOINT_MISSING
        assert gates["gate_a"].passed is False
        assert "BaseCartridge" in trace.failure_reason

    def test_gate_b_corrupted_wav_under_44_bytes_rejected(self, tmp_path: Path):
        """Gate B: Verifies truncated WAV audio (<44 bytes) triggers ASSET_INTEGRITY_VIOLATION."""
        cand = tmp_path / "cand_wav_under_44"
        create_valid_open_genre_candidate(cand)
        (cand / "assets" / "audio" / "voice.wav").write_bytes(b"RIFF_TRUNCATED")

        supervisor = BuildSupervisor()
        gates, trace = supervisor.evaluate_candidate_cartridge(cand)

        assert trace is not None
        assert trace.failure_gate == "Gate B: Multimodal Asset Verification"
        assert trace.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION
        assert gates["gate_b"].passed is False
        assert any("< 44 bytes" in v or "44-byte" in v for v in trace.asset_violations)

    def test_gate_b_corrupted_wav_non_riff_magic_rejected(self, tmp_path: Path):
        """Gate B: Verifies audio without RIFF magic header triggers ASSET_INTEGRITY_VIOLATION."""
        cand = tmp_path / "cand_wav_non_riff"
        create_valid_open_genre_candidate(cand)
        (cand / "assets" / "audio" / "voice.wav").write_bytes(b"NOT_A_RIFF_AUDIO_FILE_DATA_PADDING_HEADER_HERE_0000")

        supervisor = BuildSupervisor()
        gates, trace = supervisor.evaluate_candidate_cartridge(cand)

        assert trace is not None
        assert trace.failure_gate == "Gate B: Multimodal Asset Verification"
        assert trace.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION
        assert gates["gate_b"].passed is False
        assert any("RIFF" in v for v in trace.asset_violations)

    def test_gate_b_corrupted_png_signature_rejected(self, tmp_path: Path):
        """Gate B: Verifies invalid PNG signature triggers ASSET_INTEGRITY_VIOLATION."""
        cand = tmp_path / "cand_bad_png"
        create_valid_open_genre_candidate(cand)
        (cand / "assets" / "sprites" / "sprite.png").write_bytes(b"CORRUPTED_PNG_SIGNATURE_BYTES")

        supervisor = BuildSupervisor()
        gates, trace = supervisor.evaluate_candidate_cartridge(cand)

        assert trace is not None
        assert trace.failure_gate == "Gate B: Multimodal Asset Verification"
        assert trace.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION
        assert gates["gate_b"].passed is False
        assert any("Corrupt PNG" in v or "signature" in v for v in trace.asset_violations)

    def test_gate_b_cyclic_dialogue_infinite_loop_rejected(self, tmp_path: Path):
        """Gate B: Verifies closed infinite cycles in dialogue tree trigger ASSET_INTEGRITY_VIOLATION."""
        cand = tmp_path / "cand_cyclic_dialogue"
        create_valid_open_genre_candidate(cand)
        (cand / "narrative.json").write_text(json.dumps({
            "world": {"world_title": "Test Realm"},
            "characters": [{"id": "hero", "name": "Protagonist", "portrait": "assets/sprites/sprite.png"}],
            "dialogues": [{"bark_id": "b1", "text": "Ready.", "voice_clip": "assets/audio/voice.wav"}],
            "dialogue_trees": [
                {
                    "root_node_id": "node_alpha",
                    "nodes": {
                        "node_alpha": {"text": "Alpha", "choices": [{"text": "To Beta", "target_node_id": "node_beta"}]},
                        "node_beta": {"text": "Beta", "choices": [{"text": "To Alpha", "target_node_id": "node_alpha"}]}
                    }
                }
            ]
        }, indent=2), encoding="utf-8")

        supervisor = BuildSupervisor()
        gates, trace = supervisor.evaluate_candidate_cartridge(cand)

        assert trace is not None
        assert trace.failure_gate == "Gate B: Multimodal Asset Verification"
        assert trace.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION
        assert gates["gate_b"].passed is False
        assert any("infinite loop" in v.lower() or "cycle" in v.lower() for v in trace.asset_violations)

    def test_gate_c_uncaught_runtime_exception_rejected(self, tmp_path: Path):
        """Gate C: Verifies uncaught exception during scene update triggers UNHANDLED_EXCEPTION."""
        cand = tmp_path / "cand_runtime_crash"
        create_valid_open_genre_candidate(cand)
        content = (cand / "mechanics.py").read_text(encoding="utf-8")
        # Inject exception into update
        crashed_content = content.replace(
            "self.state_counter += 1\n        if self.probe",
            "raise RuntimeError('Adversarial scene crash injected')\n        if self.probe"
        )
        (cand / "mechanics.py").write_text(crashed_content, encoding="utf-8")

        supervisor = BuildSupervisor()
        gates, trace = supervisor.evaluate_candidate_cartridge(cand)

        assert trace is not None
        assert trace.failure_gate == "Gate C: Headless Execution & Progression"
        assert trace.failure_type == GateFailureType.UNHANDLED_EXCEPTION
        assert gates["gate_c"].passed is False
        assert "Adversarial scene crash injected" in trace.failure_reason

    def test_gate_c_frozen_progression_rejected(self, tmp_path: Path):
        """Gate C: Verifies unadvancing state counter triggers PROGRESSION_FAILURE."""
        cand = tmp_path / "cand_frozen_progression"
        create_valid_open_genre_candidate(cand)
        content = (cand / "mechanics.py").read_text(encoding="utf-8")
        # Freeze state counter in both handle_event and update
        frozen_content = content.replace("self.state_counter += 1", "pass")
        (cand / "mechanics.py").write_text(frozen_content, encoding="utf-8")

        supervisor = BuildSupervisor()
        gates, trace = supervisor.evaluate_candidate_cartridge(cand)

        assert trace is not None
        assert trace.failure_gate == "Gate C: Headless Execution & Progression"
        assert trace.failure_type == GateFailureType.PROGRESSION_FAILURE
        assert gates["gate_c"].passed is False
        assert "State counter did not increment" in trace.failure_reason

    def test_gate_d_frame_budget_exceeded_rejected(self, tmp_path: Path):
        """Gate D: Verifies frame tick duration > 16.6ms triggers FRAME_BUDGET_EXCEEDED."""
        cand = tmp_path / "cand_frame_budget"
        create_valid_open_genre_candidate(cand)
        content = (cand / "mechanics.py").read_text(encoding="utf-8")
        # Inject 25ms delay into update loop
        slow_content = "import time\n" + content.replace(
            "self.state_counter += 1\n        if self.probe",
            "self.state_counter += 1\n        time.sleep(0.025)\n        if self.probe"
        )
        (cand / "mechanics.py").write_text(slow_content, encoding="utf-8")

        supervisor = BuildSupervisor()
        gates, trace = supervisor.evaluate_candidate_cartridge(cand)

        assert trace is not None
        assert trace.failure_gate == "Gate D: Frame Budget"
        assert trace.failure_type == GateFailureType.FRAME_BUDGET_EXCEEDED
        assert gates["gate_d"].passed is False
        assert "> 16.60ms" in trace.failure_reason or "exceeds target" in trace.deliberation

    def test_gate_e_missing_custom_telemetry_emission_rejected(self, tmp_path: Path):
        """Gate E: Verifies missing declared probe emission triggers TELEMETRY_EMISSION_FAILURE."""
        cand = tmp_path / "cand_telemetry_missing"
        create_valid_open_genre_candidate(cand)
        # Declare a probe that scene code never emits
        (cand / "telemetry_manifest.json").write_text(json.dumps({
            "probes": [{"probe_name": "never_emitted_critical_probe", "category": "combat"}]
        }, indent=2), encoding="utf-8")

        supervisor = BuildSupervisor()
        gates, trace = supervisor.evaluate_candidate_cartridge(cand)

        assert trace is not None
        assert trace.failure_gate == "Gate E: Custom Telemetry Emission"
        assert trace.failure_type == GateFailureType.TELEMETRY_EMISSION_FAILURE
        assert gates["gate_e"].passed is False
        assert "never_emitted_critical_probe" in trace.failure_reason

    def test_gate_e_successful_custom_telemetry_emission_passes(self, tmp_path: Path):
        """Gate E: Verifies actively emitted declared custom probe passes 100% of all 5 gates."""
        cand = tmp_path / "cand_telemetry_clean"
        create_valid_open_genre_candidate(cand, probe_name="verified_probe")

        supervisor = BuildSupervisor()
        gates, trace = supervisor.evaluate_candidate_cartridge(cand)

        assert trace is None
        assert gates["gate_a"].passed is True
        assert gates["gate_b"].passed is True
        assert gates["gate_c"].passed is True
        assert gates["gate_d"].passed is True
        assert gates["gate_e"].passed is True
        assert gates["gate_e"].failure_type == GateFailureType.NONE


# =============================================================================
# Area 6: Closed-Loop Self-Healing & Hermetic Staging Isolation
# =============================================================================

class TestOpenGenreSelfHealingClosedLoop:
    """Verifies closed-loop self-healing, Rationale-First schemas, and hermetic staging."""

    def test_open_genre_rejection_trace_deliberation_first_across_all_gates(self, tmp_path: Path):
        """Verifies that RejectionDiagnosticTrace strictly places deliberation at field index 0
        across distinct failure modes on all 5 gates.
        """
        supervisor = BuildSupervisor()

        # Gate A defect
        cA = tmp_path / "cA"; create_valid_open_genre_candidate(cA)
        (cA / "mechanics.py").write_text("class Broken def:")
        _, traceA = supervisor.evaluate_candidate_cartridge(cA)
        assert traceA is not None
        assert list(traceA.model_dump().keys())[0] == "deliberation"
        assert list(traceA.__class__.model_fields.keys())[0] == "deliberation"

        # Gate B defect
        cB = tmp_path / "cB"; create_valid_open_genre_candidate(cB)
        (cB / "assets" / "audio" / "voice.wav").write_bytes(b"SHORT")
        _, traceB = supervisor.evaluate_candidate_cartridge(cB)
        assert traceB is not None
        assert list(traceB.model_dump().keys())[0] == "deliberation"

        # Gate C defect
        cC = tmp_path / "cC"; create_valid_open_genre_candidate(cC)
        (cC / "mechanics.py").write_text((cC / "mechanics.py").read_text().replace("self.state_counter += 1", "pass"))
        _, traceC = supervisor.evaluate_candidate_cartridge(cC)
        assert traceC is not None
        assert list(traceC.model_dump().keys())[0] == "deliberation"

        # Gate D defect
        cD = tmp_path / "cD"; create_valid_open_genre_candidate(cD)
        (cD / "mechanics.py").write_text("import time\n" + (cD / "mechanics.py").read_text().replace("self.state_counter += 1\n        if", "self.state_counter += 1\n        time.sleep(0.025)\n        if"))
        _, traceD = supervisor.evaluate_candidate_cartridge(cD)
        assert traceD is not None
        assert list(traceD.model_dump().keys())[0] == "deliberation"

        # Gate E defect
        cE = tmp_path / "cE"; create_valid_open_genre_candidate(cE)
        (cE / "telemetry_manifest.json").write_text(json.dumps({"probes": [{"probe_name": "missing"}]}))
        _, traceE = supervisor.evaluate_candidate_cartridge(cE)
        assert traceE is not None
        assert list(traceE.model_dump().keys())[0] == "deliberation"

    def test_open_genre_closed_loop_swe_repair_and_promotion_success(self, tmp_path: Path):
        """Verifies that an open-genre candidate with a defect is healed by SWE agent on retry,
        passes 100% of gates on re-evaluation, and is promoted cleanly to active/.
        """
        cartridges_root = tmp_path / "cartridges"
        active_dir = cartridges_root / "active"
        cartridges_root.mkdir(parents=True, exist_ok=True)
        active_dir.mkdir(parents=True, exist_ok=True)

        cand_dir = cartridges_root / "cand_heal_success"
        create_valid_open_genre_candidate(cand_dir)

        # Save pristine code and inject syntax error
        good_code = (cand_dir / "mechanics.py").read_text(encoding="utf-8")
        (cand_dir / "mechanics.py").write_text("class BrokenMechanics def:\n    pass\n", encoding="utf-8")

        class MockSWEAgent:
            deliberation: str = "Mock SWE agent repairing open-genre mechanics syntax error."

            def repair_cartridge_code(self, candidate_dir: Path, trace: RejectionDiagnosticTrace):
                assert trace.failure_type == GateFailureType.SYNTAX_AST_ERROR
                (candidate_dir / "mechanics.py").write_text(good_code, encoding="utf-8")
                return {"mechanics.py": candidate_dir / "mechanics.py"}

        supervisor = BuildSupervisor(
            cartridges_root=cartridges_root,
            active_cartridge_dir=active_dir,
            max_retries=3,
            swe_agent=MockSWEAgent(),
        )

        report = supervisor.evaluate_and_heal(cand_dir, max_retries=3)

        assert report.status == SupervisorGateVerdict.PROMOTED
        assert report.promoted_to_active is True
        assert report.attempt_count == 2
        assert len(report.repair_history) == 1
        assert (active_dir / "mechanics.py").is_file()
        assert "ValidCartridge" in (active_dir / "mechanics.py").read_text(encoding="utf-8")

    def test_open_genre_unrepairable_defect_exhausts_max_retries_and_preserves_active(self, tmp_path: Path):
        """Verifies that an unrepairable defect exhausts max_retries, halts with FAILED_MAX_RETRIES,
        and preserves active/ directory completely untouched.
        """
        cartridges_root = tmp_path / "cartridges"
        active_dir = cartridges_root / "active"
        cartridges_root.mkdir(parents=True, exist_ok=True)
        active_dir.mkdir(parents=True, exist_ok=True)

        # Seed active with sentinel file
        sentinel_path = active_dir / "active_canary.txt"
        sentinel_path.write_text("PRISTINE_ACTIVE_STATE", encoding="utf-8")

        cand_dir = cartridges_root / "cand_unrepairable_loop"
        cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / "broken.py").write_text("class BrokenSyntax def:\n    pass\n", encoding="utf-8")

        supervisor = BuildSupervisor(
            cartridges_root=cartridges_root,
            active_cartridge_dir=active_dir,
            max_retries=3,
            swe_agent=None,  # No agent to repair
        )

        report = supervisor.evaluate_and_heal(cand_dir, max_retries=3)

        assert report.status == SupervisorGateVerdict.FAILED_MAX_RETRIES
        assert report.promoted_to_active is False
        assert report.attempt_count == 3
        assert len(report.repair_history) == 3
        assert sentinel_path.is_file()
        assert sentinel_path.read_text(encoding="utf-8") == "PRISTINE_ACTIVE_STATE"
        assert not (active_dir / "broken.py").exists()
