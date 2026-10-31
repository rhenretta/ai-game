"""tests/test_gate_e_asset_validator.py
Authoritative unit and integration test suite for Multimodal Asset Integrity (Gate E, Feature 25).
Validates WAV audio (44-byte RIFF header, PCM 16-bit, durations, non-silence),
PNG visual assets (8-byte signature, dimension bounds, 16-color Style Bible palette conformance),
and narrative bibles (JSON schema, on-disk assets, dialogue tree reachability and cycle detection).
Strictly 0 occurrences of forbidden legacy branding tokens.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import struct
import wave
import pytest
from PIL import Image

from engine.assets.palette import STYLE_BIBLE_16_PALETTE
from engine.supervisor.asset_validator import AssetValidator
from engine.supervisor.models import AssetValidationReport, GateFailureType, QualityGateResult


@pytest.fixture
def validator() -> AssetValidator:
    return AssetValidator(
        palette=STYLE_BIBLE_16_PALETTE,
        min_audio_duration_sec=0.1,
        max_audio_duration_sec=30.0,
        silence_threshold_rms=1.0,
    )


def create_pcm_wave_file(
    path: Path,
    sample_rate: int = 22050,
    channels: int = 1,
    duration: float = 1.0,
    amplitude: int = 16000,
    freq: float = 440.0,
) -> Path:
    total_samples = int(sample_rate * duration)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        raw_data = bytearray()
        for i in range(total_samples):
            val = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate)) if amplitude > 0 else 0
            for _ in range(channels):
                raw_data.extend(struct.pack("<h", val))
        wf.writeframes(raw_data)
    return path


class TestAudioValidation:
    """Verifies Gate E audio format, duration bounds, and non-silence invariants."""
    deliberation: str = "Test suite validating voice audio RIFF headers, PCM encoding, and amplitude."

    def test_valid_voice_audio_passes(self, validator: AssetValidator, tmp_path: Path):
        wav_path = create_pcm_wave_file(tmp_path / "valid.wav", duration=1.5, amplitude=16000)
        report = validator.validate_voice_audio(wav_path)
        assert report.passed is True
        assert len(report.errors) == 0
        assert report.details["duration_sec"] == pytest.approx(1.5, rel=0.05)

    def test_audio_file_not_found(self, validator: AssetValidator, tmp_path: Path):
        report = validator.validate_voice_audio(tmp_path / "non_existent.wav")
        assert report.passed is False
        assert "not found" in report.errors[0].lower()

    def test_audio_file_truncated_header(self, validator: AssetValidator, tmp_path: Path):
        bad_path = tmp_path / "truncated.wav"
        bad_path.write_bytes(b"RIFF1234WAVEfmt ")  # Only 16 bytes (< 44)
        report = validator.validate_voice_audio(bad_path)
        assert report.passed is False
        assert "below standard 44-byte" in report.deliberation

    def test_audio_invalid_riff_magic(self, validator: AssetValidator, tmp_path: Path):
        bad_path = tmp_path / "not_riff.wav"
        data = bytearray(64)
        data[0:4] = b"FORM"  # AIFF magic, not RIFF
        bad_path.write_bytes(data)
        report = validator.validate_voice_audio(bad_path)
        assert report.passed is False
        assert any("RIFF" in err for err in report.errors)

    def test_audio_unsupported_sample_rate(self, validator: AssetValidator, tmp_path: Path):
        bad_path = tmp_path / "bad_rate.wav"
        # 12345 is not in standard set
        create_pcm_wave_file(bad_path, sample_rate=12345, duration=1.0)
        report = validator.validate_voice_audio(bad_path)
        assert report.passed is False
        assert any("sample rate" in err.lower() for err in report.errors)

    def test_audio_duration_too_short(self, validator: AssetValidator, tmp_path: Path):
        short_path = tmp_path / "short.wav"
        # 0.05s is below 0.1s min threshold
        create_pcm_wave_file(short_path, duration=0.05)
        report = validator.validate_voice_audio(short_path)
        assert report.passed is False
        assert any("too short" in err.lower() for err in report.errors)

    def test_audio_duration_too_long(self, validator: AssetValidator, tmp_path: Path):
        long_path = tmp_path / "long.wav"
        # Construct header claiming 35 seconds
        sample_rate = 22050
        duration = 35.0
        total_samples = int(sample_rate * duration)
        with wave.open(str(long_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            # Write minimal frames but claim duration
            wf.writeframes(bytearray(total_samples * 2))
        report = validator.validate_voice_audio(long_path)
        assert report.passed is False
        assert any("too long" in err.lower() for err in report.errors)

    def test_audio_silent_rejection(self, validator: AssetValidator, tmp_path: Path):
        silent_path = tmp_path / "silent.wav"
        create_pcm_wave_file(silent_path, amplitude=0, duration=1.0)
        report = validator.validate_voice_audio(silent_path)
        assert report.passed is False
        assert any("silence" in err.lower() for err in report.errors)


class TestVisualValidation:
    """Verifies Gate E visual PNG format, dimensions, and 16-color Style Bible palette conformance."""
    deliberation: str = "Test suite validating PNG magic headers, dimensions, and Style Bible color palettes."

    def test_valid_indexed_palette_png(self, validator: AssetValidator, tmp_path: Path):
        png_path = tmp_path / "valid_indexed.png"
        img = Image.new("P", (32, 32))
        # Style Bible flat palette
        flat_pal: list[int] = []
        for r, g, b in STYLE_BIBLE_16_PALETTE:
            flat_pal.extend([r, g, b])
        flat_pal.extend([0, 0, 0] * (256 - len(STYLE_BIBLE_16_PALETTE)))
        img.putpalette(flat_pal)
        img.putpixel((10, 10), 12)  # Cyan
        img.save(png_path, format="PNG")

        report = validator.validate_visual_asset(png_path)
        assert report.passed is True
        assert len(report.errors) == 0

    def test_valid_rgba_palette_png(self, validator: AssetValidator, tmp_path: Path):
        png_path = tmp_path / "valid_rgba.png"
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))  # Transparent background
        # Draw with valid palette color
        color = (*STYLE_BIBLE_16_PALETTE[8], 255)  # Neon Red
        for y in range(16, 48):
            for x in range(16, 48):
                img.putpixel((x, y), color)
        img.save(png_path, format="PNG")

        report = validator.validate_visual_asset(png_path)
        assert report.passed is True
        assert len(report.errors) == 0

    def test_unquantized_rgb_palette_violation(self, tmp_path: Path):
        strict_validator = AssetValidator(enforce_palette=True)
        png_path = tmp_path / "bad_palette.png"
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        # Non-palette color: (200, 100, 50)
        img.putpixel((5, 5), (200, 100, 50, 255))
        img.save(png_path, format="PNG")

        report = strict_validator.validate_visual_asset(png_path)
        assert report.passed is False
        assert any("palette violation" in err.lower() for err in report.errors)

    def test_invalid_png_magic_signature(self, validator: AssetValidator, tmp_path: Path):
        bad_path = tmp_path / "fake.png"
        bad_path.write_bytes(b"NOT_A_PNG_SIGNATURE_AT_ALL_1234567890")
        report = validator.validate_visual_asset(bad_path)
        assert report.passed is False
        assert any("signature" in err.lower() for err in report.errors)

    def test_image_dimension_too_small(self, validator: AssetValidator, tmp_path: Path):
        small_path = tmp_path / "tiny.png"
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        img.save(small_path, format="PNG")
        report = validator.validate_visual_asset(small_path)
        assert report.passed is False
        assert any("below minimum" in err.lower() for err in report.errors)


class TestNarrativeBibleValidation:
    """Verifies Gate E narrative.json schema, character/bark file grounding, and dialogue DAG integrity."""
    deliberation: str = "Test suite validating narrative bible schema, on-disk assets, and dialogue trees."

    def test_valid_narrative_bible_passes(self, validator: AssetValidator, tmp_path: Path):
        # Create disk assets
        create_pcm_wave_file(tmp_path / "voice_01.wav", duration=1.0)
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        img.save(tmp_path / "hero_portrait.png", format="PNG")

        narrative_path = tmp_path / "narrative.json"
        data = {
            "deliberation": "Valid narrative specification.",
            "world_title": "Crucible Core",
            "characters": [
                {
                    "character_id": "hero",
                    "name": "Operator",
                    "portrait": "hero_portrait.png",
                }
            ],
            "dialogue_barks": [
                {
                    "bark_id": "bark_start",
                    "speaker": "hero",
                    "text": "Booting systems.",
                    "voice_clip": "voice_01.wav",
                }
            ],
            "dialogue_trees": [
                {
                    "root_node_id": "start",
                    "nodes": {
                        "start": {
                            "text": "Identify yourself.",
                            "choices": [
                                {"text": "I am Operator.", "target_node_id": "end"}
                            ],
                        },
                        "end": {
                            "text": "Access granted.",
                            "choices": [],
                        },
                    },
                }
            ],
        }
        narrative_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        report = validator.validate_narrative_bible(tmp_path, narrative_path)
        assert report.passed is True
        assert len(report.errors) == 0

    def test_missing_character_portrait_rejected(self, validator: AssetValidator, tmp_path: Path):
        narrative_path = tmp_path / "narrative.json"
        data = {
            "deliberation": "Narrative with missing portrait.",
            "world_title": "Sector 01",
            "characters": [
                {
                    "character_id": "ghost",
                    "name": "Ghost",
                    "portrait": "non_existent_portrait.png",
                }
            ],
            "dialogue_barks": [],
        }
        narrative_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        report = validator.validate_narrative_bible(tmp_path, narrative_path)
        assert report.passed is False
        assert any("references missing portrait" in err.lower() for err in report.errors)

    def test_dialogue_orphan_node_rejected(self, validator: AssetValidator, tmp_path: Path):
        narrative_path = tmp_path / "narrative.json"
        data = {
            "deliberation": "Narrative with orphan dialogue node.",
            "world_title": "Sector 01",
            "characters": [],
            "dialogue_barks": [],
            "dialogue_trees": [
                {
                    "root_node_id": "node_a",
                    "nodes": {
                        "node_a": {"text": "Hello", "choices": []},
                        "node_orphan": {"text": "Unreachable", "choices": []},
                    },
                }
            ],
        }
        narrative_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        report = validator.validate_narrative_bible(tmp_path, narrative_path)
        assert report.passed is False
        assert any("orphan" in err.lower() for err in report.errors)

    def test_dialogue_closed_two_node_loop_rejected(self, validator: AssetValidator, tmp_path: Path):
        narrative_path = tmp_path / "narrative.json"
        data = {
            "deliberation": "Narrative with infinite loop.",
            "world_title": "Sector 01",
            "characters": [],
            "dialogue_barks": [],
            "dialogue_trees": [
                {
                    "root_node_id": "loop_a",
                    "nodes": {
                        "loop_a": {
                            "text": "Go to B",
                            "choices": [{"text": "B", "target_node_id": "loop_b"}],
                        },
                        "loop_b": {
                            "text": "Go to A",
                            "choices": [{"text": "A", "target_node_id": "loop_a"}],
                        },
                    },
                }
            ],
        }
        narrative_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        report = validator.validate_narrative_bible(tmp_path, narrative_path)
        assert report.passed is False
        assert any("infinite loop" in err.lower() for err in report.errors)


class TestFullCartridgeGateE:
    """Verifies validate_cartridge_assets aggregate evaluation over candidate cartridge directory."""
    deliberation: str = "Test suite validating aggregate Gate E cartridge asset evaluation."

    def test_cartridge_assets_aggregate_success(self, validator: AssetValidator, tmp_path: Path):
        cart_dir = tmp_path / "candidate_cart"
        cart_dir.mkdir(parents=True, exist_ok=True)

        create_pcm_wave_file(cart_dir / "voice.wav", duration=1.0)
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        img.save(cart_dir / "hero.png", format="PNG")

        narrative_data = {
            "deliberation": "Complete narrative bible.",
            "world_title": "Test World",
            "characters": [
                {"character_id": "hero", "portrait": "hero.png"}
            ],
            "dialogue_barks": [
                {"bark_id": "bark_1", "voice_clip": "voice.wav"}
            ],
        }
        (cart_dir / "narrative.json").write_text(json.dumps(narrative_data), encoding="utf-8")

        res = validator.validate_cartridge_assets(cart_dir)
        assert res.passed is True
        assert res.failure_type == GateFailureType.NONE
        assert list(res.to_dict().keys())[0] == "deliberation"

    def test_high_res_sprite_sheet_and_board_managed_narrative_validation(self, tmp_path: Path):
        """Verifies high-res sprite sheets (up to 8192px) pass Gate B, dict-keyed character IDs are preserved,
        and Continuous Production Board cartridges only fail on uncommissioned narrative assets if referenced in code.
        """
        v = AssetValidator(enforce_palette=False)
        cart_dir = tmp_path / "candidate_board_cart"
        (cart_dir / "assets" / "sprites").mkdir(parents=True, exist_ok=True)
        (cart_dir / "backlog.json").write_text(
            json.dumps({"items": [{"id": "task_1", "state": "approved"}]}),
            encoding="utf-8",
        )
        (cart_dir / "game_bible.json").write_text(
            json.dumps({"slice_roadmap": [{"slice_id": "s0", "status": "completed"}]}),
            encoding="utf-8",
        )
        (cart_dir / "main.gd").write_text("extends Node2D\nfunc _ready() -> void:\n\tpass\n", encoding="utf-8")

        # 4096x512 multi-frame sprite sheet passes default max_dimension=8192
        sheet_path = cart_dir / "assets" / "sprites" / "wide_anim_sheet.png"
        Image.new("RGBA", (4096, 512), (20, 40, 80, 255)).save(sheet_path, format="PNG")
        sheet_rep = v.validate_visual_asset(sheet_path)
        assert sheet_rep.passed is True

        # >8192px is still rejected
        huge_path = tmp_path / "too_wide.png"
        Image.new("RGBA", (8200, 16), (20, 40, 80, 255)).save(huge_path, format="PNG")
        assert v.validate_visual_asset(huge_path).passed is False

        # Unreferenced lore character/bark in narrative.json does not fail a board-managed cartridge
        narrative_data = {
            "deliberation": "Exported GameBible lore.",
            "world_title": "Test World",
            "characters": {
                "npc_mentor": {
                    "name": "Mentor",
                    "portrait": "assets/portraits/npc_mentor_portrait.png",
                }
            },
            "dialogue_barks": [
                {"bark_id": "bark_lore", "voice_clip": "assets/audio/bark_lore.wav"}
            ],
        }
        (cart_dir / "narrative.json").write_text(json.dumps(narrative_data), encoding="utf-8")
        assert v.validate_cartridge_assets(cart_dir).passed is True

        # If cartridge .gd actually references the missing portrait, Gate B fails fast with the real character key
        (cart_dir / "main.gd").write_text(
            'extends Node2D\nconst P = "res://assets/portraits/npc_mentor_portrait.png"\n',
            encoding="utf-8",
        )
        res_missing_ref = v.validate_cartridge_assets(cart_dir)
        assert res_missing_ref.passed is False
        assert "npc_mentor" in res_missing_ref.details
        assert "unknown" not in res_missing_ref.details

    def test_swe_repair_and_gate_f_respect_production_board_cartridges(self, tmp_path: Path):
        """Verifies ASSET_INTEGRITY_VIOLATION never triggers repair_godot_cartridge_holistic and
        Gate F recognizes completed Continuous Production Board cartridges.
        """
        from types import SimpleNamespace
        from engine.studio.swe import LeadGameplaySWEAgent
        from engine.supervisor.models import RejectionDiagnosticTrace, RemedyType
        from engine.supervisor.supervisor import BuildSupervisor

        cart_dir = tmp_path / "candidate_godot_board"
        cart_dir.mkdir(parents=True, exist_ok=True)
        (cart_dir / "project.godot").write_text('[application]\nconfig/name="BoardGame"\n', encoding="utf-8")
        (cart_dir / "backlog.json").write_text(
            json.dumps({"items": [{"id": "item_1", "state": "approved"}]}),
            encoding="utf-8",
        )
        (cart_dir / "game_bible.json").write_text(
            json.dumps(
                {
                    "slice_roadmap": [
                        {"slice_id": "slice_0", "status": "completed"},
                        {"slice_id": "slice_1", "status": "completed"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        (cart_dir / "requirements_spec.json").write_text(
            json.dumps(
                {
                    "deliberation": "Spec",
                    "spec_id": "spec_1",
                    "game_title": "BoardGame",
                    "genre_paradigm": "simulation",
                    "high_concept_summary": "Summary",
                    "requirements": [
                        {
                            "deliberation": "Req",
                            "id": "r1",
                            "title": "Req 1",
                            "description": "Desc",
                            "responsible_agents": ["LeadGameplaySWEAgent"],
                            "target_modules": ["main.gd"],
                            "status": "completed",
                            "acceptance_criteria": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        # 1. ASSET_INTEGRITY_VIOLATION must never invoke repair_godot_cartridge_holistic
        swe = LeadGameplaySWEAgent(client=SimpleNamespace())
        swe._is_live_genai_client = lambda: True
        holistic_calls = []
        swe.repair_godot_cartridge_holistic = lambda *a, **kw: holistic_calls.append(1) or {}

        trace = RejectionDiagnosticTrace(
            deliberation="Asset integrity violation.",
            candidate_id="candidate_godot_board",
            episode_number=1,
            attempt_number=1,
            failure_gate="Gate B: Multimodal Asset Verification",
            failure_type=GateFailureType.ASSET_INTEGRITY_VIOLATION,
            failure_reason="Missing audio clip",
            remedy_type=RemedyType.REGENERATE_ASSET,
            recommended_remedy="Regenerate asset",
        )
        swe.repair_cartridge_code(cart_dir, trace)
        assert holistic_calls == []

        # 2. Gate F passes cleanly when all slices & board items are completed/approved
        sup = BuildSupervisor(
            cartridges_root=tmp_path,
            active_cartridge_dir=tmp_path / "active",
        )
        gate_f, trace_f = sup.verify_gate_f_challenger(cart_dir)
        assert gate_f.passed is True
        assert trace_f is None
        assert gate_f.diagnostic_data["completed_slices"] == 2
        assert gate_f.diagnostic_data["approved_board_items"] == 1

        # 3. If a slice is still planned (incomplete), _completed_production_board_summary returns None
        (cart_dir / "game_bible.json").write_text(
            json.dumps(
                {
                    "slice_roadmap": [
                        {"slice_id": "slice_0", "status": "completed"},
                        {"slice_id": "slice_1", "status": "planned"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert sup._completed_production_board_summary(cart_dir) is None

