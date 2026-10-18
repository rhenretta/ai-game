"""tests/test_challenger_m2_1_empirical.py
Milestone 2 Challenger 1 Empirical Stress Test Suite.

Empirically challenges:
1. ProceduralSpriteRenderer resolution boundaries (8x8, 16x16, 32x32, 64x64, 128x128, 256x256).
2. Rectangular tile defect reproduction: wide aspect ratios (e.g. 128x32, 64x16, 32x8) triggering
   Pillow 'ValueError: y1 must be greater than or equal to y0' in tile style 2.
3. Dimension validation: rejects 0 or negative dimensions with clean ValueError.
4. Zero arena sprite leakage: verify chaser.png, spikes.png, etc. are NOT generated when custom manifest is supplied.
5. Audio bytes safety: MagicMock handling (raises VoiceSynthesisError, never TypeError: memoryview),
   bytearray support, and corrupted RIFF header validation.
6. Rule 1 generalizability: zero-character games return empty lists without spurious files.
7. Parameter verification: geometry_directive dead parameter analysis and CharacterSheet voice_timbre shadowing.
"""
from __future__ import annotations
import io
from pathlib import Path
import struct
from unittest.mock import MagicMock
from PIL import Image
import pytest

from engine.assets.palette import MOOD_PALETTES
from engine.studio.audio import (
    GEMINI_TTS_VOICES,
    GeminiAudioBackend,
    OfflineMockAudioBackend,
    VoiceActingAudioProducerAgent,
    VoiceSynthesisError,
    validate_wav_file,
)
from engine.studio.models import (
    AssetRequirement,
    CharacterSheet,
    DialogueBarkSpec,
    GameRequirementsSpec,
    RequirementTrack,
    StudioAgentError,
)
from engine.studio.visual import (
    PNG_SIGNATURE,
    ProceduralSpriteRenderer,
    VisualArtSpriteDesignerAgent,
)


class TestProceduralSpriteRendererEmpirical:
    """Empirical verification of sprite and tile rendering across resolutions."""

    def test_square_resolutions_rendering_8_to_256(self) -> None:
        """Verifies square resolutions from 8x8 to 256x256 render valid 16-color PNGs."""
        resolutions = [8, 16, 32, 48, 64, 96, 128, 256]
        for sz in resolutions:
            for mood in MOOD_PALETTES:
                renderer = ProceduralSpriteRenderer(mood=mood)
                for atype in ("sprite", "tile", "portrait"):
                    img = renderer.render_dynamic_sprite(
                        dimensions=(sz, sz),
                        prompt=f"Test {atype} at {sz}x{sz}",
                        asset_type=atype,
                        seed=f"seed_{sz}_{atype}",
                    )
                    assert img.size == (sz, sz)
                    buf = io.BytesIO()
                    img.save(buf, "PNG")
                    assert buf.getvalue()[:8] == PNG_SIGNATURE

    def test_dimension_validation_rejects_non_positive_dimensions(self) -> None:
        """Verifies ProceduralSpriteRenderer rejects zero and negative dimensions with ValueError."""
        renderer = ProceduralSpriteRenderer()
        invalid_dims = [
            (0, 32),
            (32, 0),
            (0, 0),
            (-8, 32),
            (32, -8),
            (-16, -16),
        ]
        for w, h in invalid_dims:
            with pytest.raises(ValueError, match="dimensions must be positive integers"):
                renderer.render_dynamic_sprite(dimensions=(w, h))

    def test_empirical_defect_rectangular_tile_crash(self) -> None:
        """VERIFIES REMEDIATION:
        When rendering a tile with wide aspect ratio (w >= 4*h, such as 128x32 or 64x16),
        safe horizontal and vertical insets prevent Pillow 'y1 must be greater than or equal to y0'.
        Valid 16-color PNG is generated with correct dimensions.
        """
        renderer = ProceduralSpriteRenderer()
        for dims in [(128, 32), (64, 16), (32, 8), (96, 24)]:
            img = renderer.render_dynamic_sprite(
                dimensions=dims,
                prompt="Wide platform tile",
                asset_type="tile",
                seed="platform_strip",
            )
            assert img.size == dims
            buf = io.BytesIO()
            img.save(buf, "PNG")
            assert buf.getvalue()[:8] == PNG_SIGNATURE

    def test_generate_cartridge_sprites_crashes_on_wide_tile_manifest(self, tmp_path: Path) -> None:
        """VERIFIES REMEDIATION:
        When an agent's asset_manifest includes a wide rectangular tile (e.g. 128x32 platform),
        generate_cartridge_sprites successfully synthesizes the sprite without crashing.
        """
        designer = VisualArtSpriteDesignerAgent()
        manifest = [
            AssetRequirement(
                deliberation="Wide platform tile",
                asset_id="platform_strip",
                asset_type="tile",
                target_path="assets/sprites/platform_strip.png",
                visual_prompt_or_text="Wide platform tile",
                dimensions=(128, 32),
            )
        ]
        sprites = designer.generate_cartridge_sprites(tmp_path / "sprites", asset_manifest=manifest)
        assert "platform_strip" in sprites
        assert (tmp_path / "sprites" / "platform_strip.png").exists()

    def test_geometry_directive_is_ignored(self) -> None:
        """VERIFIES REMEDIATION:
        geometry_directive argument in render_dynamic_sprite is incorporated into seed hash
        and symmetry/shape selection, producing distinct outputs for radial vs bilateral.
        """
        renderer = ProceduralSpriteRenderer()
        img_radial = renderer.render_dynamic_sprite((32, 32), prompt="shield", geometry_directive="radial")
        img_bilateral = renderer.render_dynamic_sprite((32, 32), prompt="shield", geometry_directive="bilateral")
        b1, b2 = io.BytesIO(), io.BytesIO()
        img_radial.save(b1, "PNG")
        img_bilateral.save(b2, "PNG")
        assert b1.getvalue() != b2.getvalue(), "geometry_directive must alter generated output"

    def test_geometry_directive_shape_and_tile_variety(self) -> None:
        """Verifies geometry directives produce distinct deterministic PNG outputs across directives."""
        renderer = ProceduralSpriteRenderer()
        directives = ["radial", "bilateral", "sharp", "box", "round"]
        outputs = {}
        for d in directives:
            img = renderer.render_dynamic_sprite((32, 32), prompt="cyber core", geometry_directive=d)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            outputs[d] = buf.getvalue()

        # All outputs across directives should be valid PNGs
        for d, b in outputs.items():
            assert b[:8] == PNG_SIGNATURE
        # Radial vs bilateral must differ
        assert outputs["radial"] != outputs["bilateral"]
        assert outputs["sharp"] != outputs["box"]

    def test_wide_tile_seed_sweep_no_crashes(self) -> None:
        """Verifies wide tiles render safely across 50 diverse seeds (exercising all tile styles 0-3)."""
        renderer = ProceduralSpriteRenderer()
        wide_dimensions = [(128, 32), (64, 16), (32, 8), (256, 32), (96, 16)]
        for w, h in wide_dimensions:
            for seed_idx in range(10):
                img = renderer.render_dynamic_sprite(
                    dimensions=(w, h),
                    prompt=f"Platform section {seed_idx}",
                    asset_type="tile",
                    seed=f"seed_wide_{seed_idx}_{w}x{h}",
                )
                assert img.size == (w, h)
                buf = io.BytesIO()
                img.save(buf, "PNG")
                assert buf.getvalue()[:8] == PNG_SIGNATURE



class TestArenaSpriteIsolationEmpirical:
    """Verifies that no arena sprites leak when custom manifest is provided."""

    def test_zero_arena_sprite_leakage_with_custom_manifest(self, tmp_path: Path) -> None:
        """When a custom manifest is passed, only declared assets are written to disk.
        Zero legacy arena sprites (chaser, sniper, boss, spikes, acid, wall, floor) exist.
        """
        designer = VisualArtSpriteDesignerAgent()
        out_dir = tmp_path / "custom_game_sprites"

        manifest = [
            AssetRequirement(
                deliberation="Laser turret entity",
                asset_id="laser_turret",
                asset_type="sprite",
                target_path="assets/sprites/laser_turret.png",
                visual_prompt_or_text="Mounted laser defense turret",
                dimensions=(32, 32),
            ),
            AssetRequirement(
                deliberation="Silicon wafer item",
                asset_id="silicon_wafer",
                asset_type="item",
                target_path="assets/sprites/silicon_wafer.png",
                visual_prompt_or_text="Silicon semiconductor wafer",
                dimensions=(16, 16),
            ),
        ]

        sprites = designer.generate_cartridge_sprites(out_dir, asset_manifest=manifest)

        assert "laser_turret" in sprites
        assert "silicon_wafer" in sprites
        assert (out_dir / "laser_turret.png").exists()
        assert (out_dir / "silicon_wafer.png").exists()

        forbidden_arena_sprites = [
            "chaser.png",
            "sniper.png",
            "boss.png",
            "spikes.png",
            "acid.png",
            "wall.png",
            "floor.png",
            "player_idle.png",
            "player_walk1.png",
            "player_walk2.png",
            "player_dash.png",
        ]
        for forbidden in forbidden_arena_sprites:
            assert not (out_dir / forbidden).exists(), f"Leaked legacy arena sprite: {forbidden}"

    def test_empty_manifest_creates_no_files(self, tmp_path: Path) -> None:
        """When manifest is empty, zero sprite files are generated."""
        designer = VisualArtSpriteDesignerAgent()
        out_dir = tmp_path / "empty_sprites"
        sprites = designer.generate_cartridge_sprites(out_dir, asset_manifest=[])
        assert sprites == {}
        assert len(list(out_dir.glob("*.png"))) == 0


class TestAudioSafetyAndGeneralizabilityEmpirical:
    """Empirical tests for audio bytes validation, mock safety, and zero-character manifests."""

    def test_magicmock_raises_voice_synthesis_error(self, tmp_path: Path) -> None:
        """When Gemini client returns a MagicMock for inline_data.data,
        GeminiAudioBackend must raise VoiceSynthesisError, preventing TypeError: memoryview crash.
        """
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = MagicMock(name="UnconfiguredMockBytes")
        mock_resp.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_client.models.generate_content.return_value = mock_resp

        backend = GeminiAudioBackend(client=mock_client)
        out_wav = tmp_path / "test_mock.wav"

        with pytest.raises(VoiceSynthesisError, match="No audio bytes returned from Gemini TTS"):
            backend.synthesize_speech("Hello test", "Aoede", out_wav)

        assert not out_wav.exists()

    def test_bytearray_input_synthesizes_valid_wav(self, tmp_path: Path) -> None:
        """Verifies that bytearray audio data is accepted and converted to valid bytes."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_part = MagicMock()

        sample_rate = 44100
        num_samples = int(0.5 * sample_rate)
        pcm_bytes = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        mock_part.inline_data.data = bytearray(pcm_bytes)
        mock_resp.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_client.models.generate_content.return_value = mock_resp

        backend = GeminiAudioBackend(client=mock_client)
        out_wav = tmp_path / "bytearray_test.wav"

        result = backend.synthesize_speech("Test speech", "Aoede", out_wav, sample_rate=sample_rate)
        assert out_wav.exists()
        assert result.duration_seconds >= 0.4
        valid, err, dur = validate_wav_file(out_wav)
        assert valid is True
        assert err == ""

    def test_corrupted_riff_bytes_fail_validation(self, tmp_path: Path) -> None:
        """Verifies that corrupt RIFF data raises VoiceSynthesisError."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_part = MagicMock()

        # Truncated RIFF header (< 44 bytes)
        mock_part.inline_data.data = b"RIFF_TRUNCATED"
        mock_resp.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_client.models.generate_content.return_value = mock_resp

        backend = GeminiAudioBackend(client=mock_client)
        out_wav = tmp_path / "corrupt_test.wav"

        with pytest.raises(VoiceSynthesisError, match="Synthesized WAV validation failed"):
            backend.synthesize_speech("Test speech", "Aoede", out_wav)

    def test_zero_character_game_returns_empty_list(self, tmp_path: Path) -> None:
        """Rule 1: games with zero character dialogue return [] cleanly with no disk writes."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        out_dir = tmp_path / "audio_empty"
        results = producer.generate_character_barks(manifest=[], output_dir=out_dir)
        assert results == []
        assert len(list(out_dir.glob("*.wav"))) == 0

    def test_voice_timbre_shadowed_by_charactersheet_default(self) -> None:
        """VERIFIES REMEDIATION:
        CharacterSheet.gemini_voice_name defaults to 'Aoede'.
        In VoiceActingAudioProducerAgent.resolve_voice_name, when gemini_voice_name is 'Aoede'
        and voice_timbre is provided ('Deep resonant bass'), timbre hints are evaluated first.
        Therefore, voice resolves to 'Fenrir' instead of falling back to default 'Aoede'.
        """
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        char_deep = CharacterSheet(
            deliberation="Deep beast character",
            character_id="deep_monster",
            name="DeepMonster",
            role="monster",
            voice_timbre="Deep resonant bass",
        )
        resolved = producer.resolve_voice_name("DeepMonster", character=char_deep)
        assert resolved == "Fenrir", f"Expected timbre hint 'Fenrir', got '{resolved}'"

    def test_all_voice_timbre_categories_with_default_aoede(self) -> None:
        """Verifies all timbre categories resolve properly when gemini_voice_name defaults to 'Aoede'."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        cases = [
            ("Deep resonant bass", "Fenrir"),
            ("Dark and heavy growl", "Fenrir"),
            ("Grave menacing whisper", "Charon"),
            ("Grim sinister tone", "Charon"),
            ("Bright cheerful tone", "Kore"),
            ("Light playful chirp", "Kore"),
            ("Focused sharp command", "Aoede"),
            ("Swift clear speech", "Aoede"),
        ]
        for timbre_text, expected_voice in cases:
            char = CharacterSheet(
                deliberation=f"Character with timbre {timbre_text}",
                character_id="char_test",
                name="CharTest",
                role="npc",
                voice_timbre=timbre_text,
            )
            resolved = producer.resolve_voice_name("CharTest", character=char)
            assert resolved == expected_voice, f"Timbre '{timbre_text}' resolved to '{resolved}', expected '{expected_voice}'"

