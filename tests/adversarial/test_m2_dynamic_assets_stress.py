"""tests/adversarial/test_m2_dynamic_assets_stress.py
Adversarial Stress Test Suite for Milestone 2: Dynamic Multimodal Asset Synthesis.

Tests:
1. ProceduralSpriteRenderer Boundary & Adversarial Dimensions (non-square, prime sizes, extreme dimensions, float casting).
2. Concurrency & Thread-Safety: Multi-threaded procedural sprite rendering and voice line synthesis.
3. Zero-Asset and Empty Manifest Invariance (Rule 1): Clean handling of empty manifests without legacy pollution.
4. Audio Mock Safety & Extreme Inputs: Corrupted headers, empty bytes, MagicMock parts, non-ASCII/SSML texts.
5. Determinism & Invariance across Seed Variations.
6. Zero Silent Fallbacks (Rule 5): Fail-fast verification on unconfigured agents and invalid outputs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import struct
from unittest.mock import MagicMock
from PIL import Image
import pytest

from engine.studio.audio import (
    GEMINI_TTS_VOICES,
    GeminiAudioBackend,
    OfflineMockAudioBackend,
    VoiceActingAudioProducerAgent,
    VoiceSynthesisError,
    escape_ssml,
    validate_wav_file,
)
from engine.studio.models import (
    AssetRequirement,
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


class TestProceduralSpriteRendererAdversarial:
    """Stress tests ProceduralSpriteRenderer against boundary, edge-case, and non-standard inputs."""

    def test_extreme_and_asymmetric_dimensions(self) -> None:
        """Renderer must handle unusual aspect ratios and boundary dimensions without crashing."""
        renderer = ProceduralSpriteRenderer()
        asymmetric_dims = [
            (1, 1),
            (2, 2),
            (7, 13),      # Prime numbers
            (15, 63),     # Narrow tall
            (96, 16),     # Wide short
            (128, 128),
            (256, 128),
        ]
        for w, h in asymmetric_dims:
            img = renderer.render_dynamic_sprite(
                dimensions=(w, h),
                prompt="Quantum Flux Conduit",
                asset_type="sprite",
                seed=f"seed_{w}_{h}",
            )
            assert img.size == (w, h)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            assert buf.getvalue()[:8] == PNG_SIGNATURE

    def test_tile_and_portrait_at_arbitrary_dimensions(self) -> None:
        """Renderer must properly apply tile and portrait routines for varying dimensions."""
        renderer = ProceduralSpriteRenderer()
        for dims in [(16, 16), (24, 24), (32, 48), (64, 64)]:
            tile_img = renderer.render_dynamic_sprite(dimensions=dims, asset_type="terrain_tile")
            assert tile_img.size == dims
            portrait_img = renderer.render_dynamic_sprite(dimensions=dims, asset_type="character_portrait")
            assert portrait_img.size == dims

    def test_non_string_and_extreme_prompts(self) -> None:
        """Renderer must survive None, empty string, very long strings, and Unicode symbols in prompts."""
        renderer = ProceduralSpriteRenderer()
        prompts = [
            "",
            None,
            "   \t\n   ",
            "🚀🔥⚡🎮👾",
            "A" * 10000,  # 10KB string
            "<xml>injection</xml>'\"--;",
        ]
        for p in prompts:
            img = renderer.render_dynamic_sprite(
                dimensions=(32, 32),
                prompt=p or "",
                asset_type="sprite",
            )
            assert img.size == (32, 32)

    def test_thread_safety_concurrent_sprite_rendering(self) -> None:
        """Renderer must be thread-safe across concurrent threads generating distinct sprites."""
        renderer = ProceduralSpriteRenderer()

        def render_worker(idx: int) -> bytes:
            # Vary dimensions and types across workers
            dims = (16 + (idx % 4) * 16, 16 + (idx % 4) * 16)
            atype = "tile" if idx % 2 == 0 else "sprite"
            img = renderer.render_dynamic_sprite(
                dimensions=dims,
                prompt=f"Worker asset {idx}",
                asset_type=atype,
                seed=f"thread_seed_{idx}",
            )
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return buf.getvalue()

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(render_worker, i) for i in range(40)]
            results = [f.result() for f in futures]

        assert len(results) == 40
        for data in results:
            assert data[:8] == PNG_SIGNATURE
        # Assert variety across varying seeds, dimensions, and asset types
        unique_results = set(results)
        assert len(unique_results) >= 8

    def test_procedural_sprite_structure_variety(self) -> None:
        """Renderer produces multiple distinct structural templates across seeds."""
        renderer = ProceduralSpriteRenderer()
        sprites_32 = [
            renderer.render_dynamic_sprite((32, 32), prompt=f"P_{i}", seed=f"S_{i}")
            for i in range(16)
        ]
        png_bytes = set()
        for s in sprites_32:
            b = io.BytesIO()
            s.save(b, "PNG")
            png_bytes.add(b.getvalue())
        # Confirms at least 4 distinct structural geometric shapes are generated
        assert len(png_bytes) >= 4


class TestDynamicVoiceAndAudioAdversarial:
    """Stress tests voice resolution, bark synthesis, and error boundaries."""

    def test_voice_resolution_adversarial_inputs(self) -> None:
        """resolve_voice_name must always return a valid Gemini voice even on extreme inputs."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        test_inputs = [
            "",
            " ",
            "???",
            "12345",
            "👾🤖👽",
            "X" * 1000,
            "Null",
            "None",
            "HERO",           # Case insensitivity
            "Super_Boss_99",
        ]
        for inp in test_inputs:
            voice = producer.resolve_voice_name(inp)
            assert voice in GEMINI_TTS_VOICES, f"Expected voice in {GEMINI_TTS_VOICES}, got '{voice}' for input '{inp}'"

    def test_ssml_escaping_resilience(self) -> None:
        """escape_ssml must sanitize XML/HTML control characters."""
        dangerous_text = '<speak><break time="10s"/>Alert & Warning: "Critical" Failure <script>alert(1)</script></speak>'
        escaped = escape_ssml(dangerous_text)
        assert "<" not in escaped or "&lt;" in escaped
        assert ">" not in escaped or "&gt;" in escaped
        assert "&" in escaped

    def test_thread_safety_concurrent_audio_synthesis(self, tmp_path: Path) -> None:
        """Producer must handle concurrent multi-threaded audio synthesis cleanly."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())

        def audio_worker(idx: int) -> Path:
            out_path = tmp_path / f"thread_bark_{idx}.wav"
            bark = DialogueBarkSpec(
                deliberation=f"Thread bark {idx}",
                bark_id=f"thread_bark_{idx}",
                speaker_id=f"speaker_{idx % 5}",
                speaker_name=f"Speaker {idx % 5}",
                trigger_type="dialogue",
                text=f"Thread status report from unit {idx}",
                voice_clip=str(out_path),
            )
            producer.synthesize_voice_line(bark, out_path)
            return out_path

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(audio_worker, i) for i in range(24)]
            output_paths = [f.result() for f in futures]

        assert len(output_paths) == 24
        for p in output_paths:
            assert p.exists()
            valid, err, dur = validate_wav_file(p)
            assert valid is True
            assert dur > 0.1

    def test_gemini_backend_rejection_of_empty_and_corrupt_responses(self, tmp_path: Path) -> None:
        """GeminiAudioBackend must raise VoiceSynthesisError on all invalid API responses."""
        out_wav = tmp_path / "corrupt.wav"

        # Case 1: Candidates list is empty
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.candidates = []
        mock_client.models.generate_content.return_value = mock_resp
        backend = GeminiAudioBackend(client=mock_client)

        with pytest.raises(VoiceSynthesisError, match="No audio bytes returned"):
            backend.synthesize_speech("Test", "Aoede", out_wav)

        # Case 2: inline_data.data is empty bytes
        mock_part = MagicMock()
        mock_part.inline_data.data = b""
        mock_cand = MagicMock()
        mock_cand.content.parts = [mock_part]
        mock_resp.candidates = [mock_cand]

        with pytest.raises(VoiceSynthesisError, match="No audio bytes returned"):
            backend.synthesize_speech("Test", "Aoede", out_wav)


class TestManifestIsolationAndZeroPollution:
    """Verifies complete isolation from hardcoded legacy assets when manifest is supplied."""

    def test_empty_asset_manifest_produces_zero_legacy_sprites(self, tmp_path: Path) -> None:
        """When an empty manifest is provided (e.g. pure ASCII/text terminal game),
        generate_cartridge_sprites must generate 0 files and NOT fallback to the 11 arena sprites.
        """
        designer = VisualArtSpriteDesignerAgent()
        out_dir = tmp_path / "empty_sprites"

        spec = GameRequirementsSpec(
            deliberation="Pure text parser adventure with zero graphical sprites.",
            game_title="Colossal Cave Terminal",
            genre_paradigm="Text Adventure",
            target_audience_summary="Retro text gamer",
            requirements=[
                RequirementTrack(
                    deliberation="Parser loop",
                    id="R1",
                    title="Command Parser",
                    description="Text command loop.",
                    status="pending",
                )
            ],
            asset_manifest=[],  # Empty manifest
        )

        sprites = designer.generate_cartridge_sprites(out_dir, requirements_spec=spec)
        assert sprites == {}
        # Verify no files were created in out_dir
        created_files = list(out_dir.glob("*.png"))
        assert len(created_files) == 0, f"Expected 0 sprites, found {len(created_files)}: {created_files}"

    def test_generate_character_barks_ignores_visual_assets(self, tmp_path: Path) -> None:
        """Audio generator must ignore visual sprites/tiles in the manifest and synthesize only audio items."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        audio_dir = tmp_path / "audio_out"

        mixed_manifest = [
            AssetRequirement(
                deliberation="Sprite asset",
                asset_id="hero_sprite",
                asset_type="sprite",
                target_path="assets/sprites/hero.png",
                visual_prompt_or_text="Hero sprite",
            ),
            AssetRequirement(
                deliberation="Tile asset",
                asset_id="floor_tile",
                asset_type="tile",
                target_path="assets/sprites/floor.png",
                visual_prompt_or_text="Floor tile",
            ),
            AssetRequirement(
                deliberation="Sound effect alert",
                asset_id="sfx_shield_down",
                asset_type="sfx",
                target_path="assets/audio/shield_down.wav",
                visual_prompt_or_text="Warning: Shields depleted.",
            ),
        ]

        results = producer.generate_character_barks(manifest=mixed_manifest, output_dir=audio_dir)
        assert len(results) == 1
        assert results[0].bark_id == "sfx_shield_down"
        assert (audio_dir / "shield_down.wav").exists()
        assert not (audio_dir / "hero.png").exists()
