"""tests/test_studio_dynamic_assets.py
Milestone 2 Unit & Integration Test Suite — Dynamic Multimodal Asset Synthesis.

Verifies:
1. Dynamic sprite synthesis in VisualArtSpriteDesignerAgent & ProceduralSpriteRenderer
   driven strictly by GameRequirementsSpec / AssetRequirement manifest with ZERO
   hardcoded arena sprite filenames (chaser.png, sniper.png, spikes.png, etc.).
2. Dynamic speaker voice mapping in VoiceActingAudioProducerAgent supporting arbitrary
   character names from narrative barks or asset manifests.
3. Mock safety for Gemini audio synthesis preventing MagicMock memoryview TypeErrors.
4. Rule 1 generalizability (zero-character / zero-audio games supported cleanly).
5. Rule 5 fail-fast error behavior on missing clients or invalid dimensions.
"""
from __future__ import annotations
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


class TestDynamicVisualAssetSynthesis:
    """Verifies dynamic manifest-driven sprite synthesis and procedural rendering."""

    def test_dynamic_sprite_generation_from_game_requirements_spec(self, tmp_path: Path) -> None:
        """Verifies generate_cartridge_sprites synthesizes exact assets declared in GameRequirementsSpec
        and generates ZERO hardcoded arena sprites (chaser, sniper, spikes, acid, wall, floor).
        """
        designer = VisualArtSpriteDesignerAgent()
        sprites_dir = tmp_path / "factory_sprites"

        # Conceived game: Factory Automation Simulator
        spec = GameRequirementsSpec(
            deliberation="Factory automation sim with logistics belts, mining drills, and smelting units.",
            game_title="Foundry Logistics",
            genre_paradigm="Automation Simulation",
            target_audience_summary="Systems management player",
            requirements=[
                RequirementTrack(
                    deliberation="Logistics transport loop.",
                    id="R1",
                    title="Conveyor Infrastructure",
                    description="Grid-based conveyor item transfer.",
                    status="pending",
                )
            ],
            asset_manifest=[
                AssetRequirement(
                    deliberation="Conveyor directional tile",
                    asset_id="conveyor_east",
                    asset_type="tile",
                    target_path="assets/sprites/conveyor_east.png",
                    visual_prompt_or_text="32x32 animated industrial conveyor tile",
                    dimensions=(32, 32),
                ),
                AssetRequirement(
                    deliberation="Heavy ore smelting furnace",
                    asset_id="smelter_furnace",
                    asset_type="sprite",
                    target_path="assets/sprites/smelter_furnace.png",
                    visual_prompt_or_text="64x64 modular smelting unit",
                    dimensions=(64, 64),
                ),
                AssetRequirement(
                    deliberation="Raw iron ore resource chunk",
                    asset_id="iron_chunk",
                    asset_type="sprite",
                    target_path="assets/sprites/iron_chunk.png",
                    visual_prompt_or_text="16x16 metallic iron mineral nugget",
                    dimensions=(16, 16),
                ),
            ],
        )

        sprites = designer.generate_cartridge_sprites(sprites_dir, mood="retro_arcade", requirements_spec=spec)

        # Assert declared assets exist in returned dict and on disk
        assert "conveyor_east" in sprites
        assert "smelter_furnace" in sprites
        assert "iron_chunk" in sprites

        for key in ("conveyor_east", "smelter_furnace", "iron_chunk"):
            p = sprites[key]
            assert p.exists()
            assert p.read_bytes()[:8] == PNG_SIGNATURE
            with Image.open(p) as img:
                assert img.format == "PNG"

        # Assert legacy hardcoded arena sprites DO NOT exist
        assert not (sprites_dir / "chaser.png").exists()
        assert not (sprites_dir / "sniper.png").exists()
        assert not (sprites_dir / "boss.png").exists()
        assert not (sprites_dir / "spikes.png").exists()
        assert not (sprites_dir / "acid.png").exists()
        assert not (sprites_dir / "wall.png").exists()
        assert not (sprites_dir / "floor.png").exists()

    def test_dynamic_sprite_generation_from_asset_manifest_list(self, tmp_path: Path) -> None:
        """Verifies generate_cartridge_sprites works directly from a List[AssetRequirement]."""
        designer = VisualArtSpriteDesignerAgent()
        sprites_dir = tmp_path / "space_sim_sprites"

        # Conceived game: Orbital Physics Strategy
        manifest = [
            AssetRequirement(
                deliberation="Orbital defense satellite beacon",
                asset_id="orbital_beacon",
                asset_type="sprite",
                target_path="assets/sprites/orbital_beacon.png",
                visual_prompt_or_text="48x48 solar satellite beacon",
                dimensions=(48, 48),
            ),
            AssetRequirement(
                deliberation="Dense asteroid hazard",
                asset_id="asteroid_core",
                asset_type="hazard",
                target_path="assets/sprites/asteroid_core.png",
                visual_prompt_or_text="48x48 pitted carbonaceous asteroid",
                dimensions=(48, 48),
            ),
        ]

        sprites = designer.generate_cartridge_sprites(sprites_dir, mood="neon_cyberpunk", asset_manifest=manifest)

        assert "orbital_beacon" in sprites
        assert "asteroid_core" in sprites
        assert (sprites_dir / "orbital_beacon.png").exists()
        assert (sprites_dir / "asteroid_core.png").exists()
        assert not (sprites_dir / "chaser.png").exists()
        assert not (sprites_dir / "spikes.png").exists()

        with Image.open(sprites_dir / "orbital_beacon.png") as img:
            assert img.size == (48, 48)

    def test_dynamic_sprite_resolutions_and_boundary_dimensions(self) -> None:
        """Validates dynamic synthesis across boundary resolutions (8x8 up to 128x128)."""
        renderer = ProceduralSpriteRenderer()
        resolutions = [(8, 8), (16, 16), (32, 32), (48, 48), (64, 64), (128, 128)]

        for w, h in resolutions:
            img = renderer.render_dynamic_sprite(
                dimensions=(w, h),
                prompt=f"Test shape {w}x{h}",
                asset_type="sprite",
            )
            assert img.size == (w, h)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            assert buf.getvalue()[:8] == PNG_SIGNATURE

    def test_procedural_sprite_renderer_determinism_and_variety(self) -> None:
        """Verifies ProceduralSpriteRenderer produces deterministic results for identical seeds
        and distinct output for distinct seeds/prompts.
        """
        renderer = ProceduralSpriteRenderer()

        # Determinism check
        img1 = renderer.render_dynamic_sprite((32, 32), prompt="Quantum Core Reactor", seed="core_1")
        img2 = renderer.render_dynamic_sprite((32, 32), prompt="Quantum Core Reactor", seed="core_1")

        buf1, buf2 = io.BytesIO(), io.BytesIO()
        img1.save(buf1, "PNG")
        img2.save(buf2, "PNG")
        assert buf1.getvalue() == buf2.getvalue(), "Identical seed and prompt must produce identical pixel art"

        # Variety check
        img3 = renderer.render_dynamic_sprite((32, 32), prompt="Organic Bio Hazard Spore", seed="spore_2")
        buf3 = io.BytesIO()
        img3.save(buf3, "PNG")
        assert buf1.getvalue() != buf3.getvalue(), "Different prompts/seeds must produce different sprites"

    def test_procedural_sprite_renderer_rejects_non_positive_dimensions(self) -> None:
        """Verifies ProceduralSpriteRenderer fails fast on invalid dimensions."""
        renderer = ProceduralSpriteRenderer()
        with pytest.raises(ValueError, match="dimensions must be positive"):
            renderer.render_dynamic_sprite(dimensions=(0, 32))
        with pytest.raises(ValueError, match="dimensions must be positive"):
            renderer.render_dynamic_sprite(dimensions=(32, -10))

    def test_render_from_requirement_both_types(self) -> None:
        """Verifies render_from_requirement accepts both AssetRequirement objects and dictionaries."""
        renderer = ProceduralSpriteRenderer()

        # Object form
        req_obj = AssetRequirement(
            deliberation="Crystal shard",
            asset_id="gem_sapphire",
            asset_type="sprite",
            target_path="assets/sprites/gem_sapphire.png",
            visual_prompt_or_text="Sapphire crystal shard",
            dimensions=(24, 24),
        )
        img_obj = renderer.render_from_requirement(req_obj)
        assert img_obj.size == (24, 24)

        # Dict form
        req_dict = {
            "asset_id": "gem_ruby",
            "asset_type": "tile",
            "dimensions": (16, 16),
            "visual_prompt_or_text": "Ruby red faceted tile",
        }
        img_dict = renderer.render_from_requirement(req_dict)
        assert img_dict.size == (16, 16)

    def test_generate_dynamic_sprite_single_helper(self, tmp_path: Path) -> None:
        """Verifies generate_dynamic_sprite helper renders and saves a single asset."""
        designer = VisualArtSpriteDesignerAgent()
        out_file = tmp_path / "custom_beacon.png"

        req = AssetRequirement(
            deliberation="Custom beacon sprite",
            asset_id="custom_beacon",
            asset_type="sprite",
            target_path="assets/sprites/custom_beacon.png",
            visual_prompt_or_text="Floating energy beacon",
            dimensions=(32, 32),
        )
        saved = designer.generate_dynamic_sprite(req, output_path=out_file)
        assert saved.exists()
        assert saved.read_bytes()[:8] == PNG_SIGNATURE

    def test_legacy_mode_when_manifest_is_none(self, tmp_path: Path) -> None:
        """Verifies backward compatibility: if requirements_spec and asset_manifest are None,
        generate_cartridge_sprites generates the standard 11 base sprites so legacy tests don't break.
        """
        designer = VisualArtSpriteDesignerAgent()
        sprites_dir = tmp_path / "legacy_pack"

        sprites = designer.generate_cartridge_sprites(sprites_dir, mood="neon_cyberpunk")
        assert "player_idle" in sprites
        assert "chaser" in sprites
        assert "spikes" in sprites
        assert "wall" in sprites
        assert "floor" in sprites

        for path in sprites.values():
            assert path.exists()
            assert path.read_bytes()[:8] == PNG_SIGNATURE


class TestDynamicAudioAssetSynthesis:
    """Verifies dynamic speaker voice mapping, byte/bytearray validation, and bark generation."""

    def test_dynamic_speaker_names_voice_assignment(self, tmp_path: Path) -> None:
        """Verifies VoiceActingAudioProducerAgent dynamically synthesizes barks for arbitrary speaker names
        without hardcoded archetype constraints (hero, boss, npc, companion, rival).
        """
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())

        custom_speakers = [
            ("station_commander", "Commander Vane", "Hostile craft entering sector perimeter."),
            ("ai_custodian", "CUSTODIAN-9", "Atmospheric seals nominal. Power grid routed."),
            ("deep_core_miner", "Miner Jax", "Found rich mineral seam at sub-level 4."),
            ("alien_hive_mind", "The Collective", "We perceive your presence in our void."),
        ]

        for s_id, s_name, text in custom_speakers:
            bark = DialogueBarkSpec(
                deliberation=f"Dialogue bark for {s_name}",
                bark_id=f"bark_{s_id}",
                speaker_id=s_id,
                speaker_name=s_name,
                trigger_type="dialogue",
                text=text,
                voice_clip=str(tmp_path / f"{s_id}.wav"),
            )
            out_file = tmp_path / f"{s_id}.wav"
            res = producer.synthesize_voice_line(bark, out_file)

            assert res.file_path == str(out_file)
            assert out_file.exists()
            assert 0.1 <= res.duration_seconds <= 30.0
            valid, err, dur = validate_wav_file(out_file)
            assert valid is True
            assert err == ""

    def test_custom_voice_registration_and_resolution(self) -> None:
        """Verifies register_character_voice and resolve_voice_name assign specified Gemini voices."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())

        # Register custom character voice
        producer.register_character_voice("announcer", "Fenrir")
        producer.register_character_voice("operator_claire", "Aoede")

        assert producer.resolve_voice_name("announcer") == "Fenrir"
        assert producer.resolve_voice_name("operator_claire") == "Aoede"

        # Unregistered dynamic name maps deterministically to a valid Gemini voice
        random_speaker_voice = producer.resolve_voice_name("unknown_trader_99")
        assert random_speaker_voice in GEMINI_TTS_VOICES

    def test_generate_character_barks_from_manifest(self, tmp_path: Path) -> None:
        """Verifies generate_character_barks synthesizes all audio assets declared in an AssetRequirement manifest."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        audio_dir = tmp_path / "audio_assets"

        manifest = [
            AssetRequirement(
                deliberation="Reactor warning audio bark",
                asset_id="bark_reactor_alert",
                asset_type="audio_bark",
                target_path="assets/audio/reactor_alert.wav",
                visual_prompt_or_text="Warning: Core temperature exceeding safe operational thresholds.",
            ),
            AssetRequirement(
                deliberation="Logistics docking confirmation audio",
                asset_id="bark_docking_complete",
                asset_type="audio_bark",
                target_path="assets/audio/docking_complete.wav",
                visual_prompt_or_text="Cargo vessel docked successfully.",
            ),
            # Visual asset should be ignored by audio generator
            AssetRequirement(
                deliberation="Visual tile asset",
                asset_id="tile_wall",
                asset_type="tile",
                target_path="assets/sprites/tile_wall.png",
                visual_prompt_or_text="Wall tile",
            ),
        ]

        results = producer.generate_character_barks(manifest=manifest, output_dir=audio_dir)
        assert len(results) == 2

        assert (audio_dir / "reactor_alert.wav").exists()
        assert (audio_dir / "docking_complete.wav").exists()

        for res in results:
            assert Path(res.file_path).exists()
            valid, _, _ = validate_wav_file(Path(res.file_path))
            assert valid is True

    def test_generate_character_barks_empty_manifest_for_zero_character_game(self, tmp_path: Path) -> None:
        """Verifies Rule 1 generalizability: games with zero dialogue barks return an empty list
        without crashing or synthesizing unrequested dummy files.
        """
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        audio_dir = tmp_path / "empty_audio"

        results = producer.generate_character_barks(manifest=[], output_dir=audio_dir)
        assert results == []
        assert len(list(audio_dir.glob("*.wav"))) == 0

    def test_mock_safety_prevents_magicmock_memoryview_crash(self, tmp_path: Path) -> None:
        """CRITICAL FIX VERIFICATION:
        When a unit test or mock client returns a MagicMock for inline_data.data (instead of real bytes),
        GeminiAudioBackend must check `isinstance(data, (bytes, bytearray))` and raise VoiceSynthesisError
        instead of passing MagicMock to Path.write_bytes (which would trigger TypeError: memoryview crash).
        """
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock()

        # Simulate MagicMock object as inline_data.data
        mock_part.inline_data.data = MagicMock(name="UnconfiguredMockBytes")
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        backend = GeminiAudioBackend(client=mock_client)
        out_wav = tmp_path / "mock_test.wav"

        # Must raise VoiceSynthesisError, NEVER TypeError: memoryview
        with pytest.raises(VoiceSynthesisError) as exc_info:
            backend.synthesize_speech(
                text="Test speech line",
                voice_name="Aoede",
                output_path=out_wav,
            )

        assert "No audio bytes returned" in str(exc_info.value)
        assert not out_wav.exists()

    def test_mock_safety_with_valid_bytes_synthesizes_cleanly(self, tmp_path: Path) -> None:
        """Verifies that when mock client returns real PCM or WAV bytes, GeminiAudioBackend
        synthesizes and validates the WAV file successfully.
        """
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock()

        # Build genuine test PCM samples
        sample_rate = 44100
        num_samples = int(0.5 * sample_rate)
        samples = [int(10000.0 * (i % 50) / 50.0) for i in range(num_samples)]
        pcm_bytes = struct.pack(f"<{len(samples)}h", *samples)

        mock_part.inline_data.data = pcm_bytes
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        backend = GeminiAudioBackend(client=mock_client)
        out_wav = tmp_path / "valid_test.wav"

        result = backend.synthesize_speech(
            text="Online and ready.",
            voice_name="Aoede",
            output_path=out_wav,
            sample_rate=sample_rate,
        )

        assert out_wav.exists()
        assert result.duration_seconds > 0.4
        valid, err, _ = validate_wav_file(out_wav)
        assert valid is True

    def test_synthesize_voice_line_from_asset_requirement(self, tmp_path: Path) -> None:
        """Verifies synthesize_voice_line directly accepts AssetRequirement instances."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        out_wav = tmp_path / "alarm.wav"

        req = AssetRequirement(
            deliberation="Warning alert",
            asset_id="reactor_alarm",
            asset_type="audio_bark",
            target_path=str(out_wav),
            visual_prompt_or_text="Critical alert: Radiation leak in sector 7.",
        )

        result = producer.synthesize_voice_line(req, out_wav)
        assert out_wav.exists()
        assert result.bark_id == "reactor_alarm"
        assert 0.1 <= result.duration_seconds <= 30.0

    def test_rule5_fail_fast_when_client_and_backend_none(self) -> None:
        """Rule 5: Zero Silent Fallbacks. Must raise StudioAgentError if neither client nor backend passed."""
        with pytest.raises(StudioAgentError, match="requires an active google.genai Client"):
            VoiceActingAudioProducerAgent(client=None, backend=None)
