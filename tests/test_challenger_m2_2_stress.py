"""tests/test_challenger_m2_2_stress.py
Authoritative Empirical Challenger 2 Stress Test Suite for Milestone 2:
1. Dynamic Voice Resolution: Speaker name hashing distribution, custom character registration,
   dialogue bark synthesis with 0 characters, 1 character, 50 characters, and Rule 5 fail-fast boundaries.
2. Concurrent Sprite Generation: Multi-threaded batch sprite generation from large manifests without
   race conditions, file lock contention, or corrupted PNG headers.
3. Quantization Fidelity: Every generated PNG strictly conforms to 16-color Style Bible palette,
   valid 8-byte PNG signatures, mode 'P' indexed format, non-zero file size, and dimension bounds.
4. Backward Compatibility: Legacy callers invoking generate_cartridge_sprites without a manifest
   reliably receive the complete 11-asset base sprite pack, while manifest callers receive zero
   legacy arena assets.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
from pathlib import Path
import random
import string
import struct
import time
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import MagicMock

from PIL import Image
import pytest

from engine.assets.palette import STYLE_BIBLE_16_PALETTE, MOOD_PALETTES
from engine.studio.audio import (
    GEMINI_TTS_VOICES,
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
    get_seed_integer,
)
from engine.supervisor.asset_validator import AssetValidator


# ============================================================================
# Challenge 1: Dynamic Voice Resolution & Bark Synthesis Scaling
# ============================================================================

class TestDynamicVoiceResolutionAndBarkScaling:
    """Stress-test speaker name hashing, dynamic registration, and 0/1/50 bark scaling."""

    def test_speaker_name_hashing_determinism_and_distribution(self):
        """Verifies hash mapping is strictly deterministic and distributes across all 5 TTS voices."""
        agent = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())

        # Test deterministic consistency
        speakers = [
            "Dr. Elena Vance",
            "Xylok_99",
            "The Ancient Archivist",
            "Unit-734",
            "Sovereign Mind",
            "Starship AI 'Aura'",
            "Commander 🚀",
            "127.0.0.1",
            "",
            "   padded speaker   ",
        ]
        for spk in speakers:
            v1 = agent.resolve_voice_name(spk)
            v2 = agent.resolve_voice_name(spk)
            assert v1 == v2, f"Inconsistent voice resolution for '{spk}': {v1} != {v2}"
            assert v1 in GEMINI_TTS_VOICES, f"Voice '{v1}' not in GEMINI_TTS_VOICES"

        # Test distribution across 150 diverse arbitrary speaker names
        voice_counts: Dict[str, int] = {v: 0 for v in GEMINI_TTS_VOICES}
        for i in range(150):
            rand_name = f"Speaker_{i}_{hashlib.md5(str(i).encode()).hexdigest()[:8]}"
            resolved = agent.resolve_voice_name(rand_name)
            voice_counts[resolved] += 1

        # Every one of the 5 voices must be reached (no dead branches)
        for voice, count in voice_counts.items():
            assert count > 0, f"Voice {voice} was never selected in 150 samples: {voice_counts}"

    def test_custom_character_voice_registration_and_precedence(self):
        """Verifies custom voice registration overrides defaults and follows exact priority hierarchy:
        CharacterSheet.gemini_voice_name > custom registration > legacy archetype > timbre hint > hash mapping.
        """
        agent = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())

        # 1. Register custom voices
        agent.register_character_voice("Grandmaster_Vex", "Fenrir")
        agent.register_character_voice("Solaris_Prime", "Kore")

        # Exact match and case-insensitivity
        assert agent.resolve_voice_name("Grandmaster_Vex") == "Fenrir"
        assert agent.resolve_voice_name("grandmaster_vex") == "Fenrir"
        assert agent.resolve_voice_name("Solaris_Prime") == "Kore"
        assert agent.resolve_voice_name("solaris_prime") == "Kore"

        # 2. Overriding legacy archetype: "hero" default is "Aoede", override to "Charon"
        assert agent.resolve_voice_name("hero") == "Aoede"
        agent.register_character_voice("hero", "Charon")
        assert agent.resolve_voice_name("hero") == "Charon"

        # 3. CharacterSheet explicit gemini_voice_name takes top precedence over registration
        char_explicit = CharacterSheet(
            deliberation="Character with explicit voice specification",
            character_id="grandmaster_vex",
            name="Grandmaster_Vex",
            role="companion",
            gemini_voice_name="Puck",
        )
        assert agent.resolve_voice_name("Grandmaster_Vex", character=char_explicit) == "Puck"

        # 4. Timbre hints when character has no explicit gemini_voice_name
        char_deep = CharacterSheet(
            deliberation="Deep beast character",
            character_id="unknown_beast",
            name="Unknown_Beast",
            role="monster",
            voice_timbre="Deep resonant bass",
            gemini_voice_name="",
        )
        assert agent.resolve_voice_name("Unknown_Beast", character=char_deep) == "Fenrir"

        char_bright = CharacterSheet(
            deliberation="Bright sprite companion",
            character_id="fairy_sprite",
            name="Fairy_Sprite",
            role="companion",
            voice_timbre="Bright light tone",
            gemini_voice_name="",
        )
        assert agent.resolve_voice_name("Fairy_Sprite", character=char_bright) == "Kore"

    def test_dialogue_bark_synthesis_zero_characters(self, tmp_path: Path):
        """Rule 1: Games conceived with 0 characters (e.g. abstract/automation sims) produce empty barks cleanly."""
        agent = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        out_dir = tmp_path / "zero_char_audio"

        # 1. Completely empty call
        results = agent.generate_character_barks(output_dir=out_dir)
        assert results == []
        assert len(list(out_dir.glob("*.wav"))) == 0

        # 2. Empty manifest list
        results = agent.generate_character_barks(output_dir=out_dir, manifest=[])
        assert results == []
        assert len(list(out_dir.glob("*.wav"))) == 0

        # 3. Requirements spec with only visual assets (zero audio assets)
        spec = GameRequirementsSpec(
            deliberation="Abstract puzzle game with geometric blocks and zero spoken dialogue.",
            game_title="Tetra Grid",
            genre_paradigm="Puzzle",
            target_audience_summary="Logic puzzle enthusiasts",
            requirements=[],
            asset_manifest=[
                AssetRequirement(
                    deliberation="Block tile",
                    asset_id="block_blue",
                    asset_type="tile",
                    target_path="assets/sprites/block_blue.png",
                )
            ],
        )
        results = agent.generate_character_barks(output_dir=out_dir, requirements_spec=spec)
        assert results == []
        assert len(list(out_dir.glob("*.wav"))) == 0

    def test_dialogue_bark_synthesis_single_character(self, tmp_path: Path):
        """Verifies single character bark synthesis and authentic Gate E audio validation."""
        agent = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        out_dir = tmp_path / "single_char_audio"
        validator = AssetValidator(enforce_palette=False)

        bark = DialogueBarkSpec(
            deliberation="Introductory pilot check-in line",
            bark_id="pilot_checkin",
            speaker_id="Lt_Ryder",
            speaker_name="Lieutenant Ryder",
            trigger_type="game_start",
            text="Systems operational. All thrusters responding nominal.",
            voice_clip=str(out_dir / "pilot_checkin.wav"),
        )

        results = agent.generate_character_barks(barks=[bark], output_dir=out_dir)
        assert len(results) == 1
        res = results[0]
        assert res.bark_id == "pilot_checkin"
        assert res.duration_seconds >= 0.1
        assert res.duration_seconds <= 30.0
        assert res.file_size_bytes > 44

        p = Path(res.file_path)
        assert p.exists()
        valid, err, dur = validate_wav_file(p)
        assert valid, f"WAV validation failed: {err}"

        # Gate E AssetValidator audit
        report = validator.validate_voice_audio(p)
        assert report.passed, f"Gate E failed on single character audio: {report.errors}"
        assert report.details["peak_amplitude"] > 0
        assert report.details["rms_amplitude"] > 0

    def test_dialogue_bark_synthesis_50_characters_scale(self, tmp_path: Path):
        """Stress-tests synthesizing 50 distinct characters / audio barks in a single batch,
        verifying 50 valid WAV files on disk and Gate E passing on 100% of outputs.
        """
        agent = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        out_dir = tmp_path / "scale_50_chars"
        validator = AssetValidator(enforce_palette=False)

        # Build 50 distinct asset requirements of type 'audio_bark' with varying word counts
        manifest_items: List[AssetRequirement] = []
        for i in range(50):
            spk_id = f"char_{i:02d}_{string.ascii_lowercase[i % 26]}"
            word_list = ["alert", "status", "ready", "command", "shield", "power", "grid", "core"]
            dynamic_text = " ".join([word_list[j % len(word_list)] for j in range(3 + (i % 15))])
            manifest_items.append(
                AssetRequirement(
                    deliberation=f"Audio bark line for operative {spk_id}",
                    asset_id=f"voice_operative_{i:02d}",
                    asset_type="audio_bark",
                    target_path=f"audio/voice_operative_{i:02d}.wav",
                    visual_prompt_or_text=f"Operative {spk_id}: {dynamic_text}",
                )
            )

        t0 = time.perf_counter()
        results = agent.generate_character_barks(output_dir=out_dir, manifest=manifest_items)
        elapsed = time.perf_counter() - t0

        assert len(results) == 50, f"Expected 50 results, got {len(results)}"
        assert len(list(out_dir.glob("*.wav"))) == 50

        # Verify all 50 files pass Gate E validation and have valid non-zero content
        durations: Set[float] = set()
        for res in results:
            p = Path(res.file_path)
            assert p.exists()
            assert p.stat().st_size > 44

            # Check Gate E asset validator
            gate_e_rep = validator.validate_voice_audio(p)
            assert gate_e_rep.passed, f"Gate E failed for {p.name}: {gate_e_rep.errors}"
            assert gate_e_rep.details["peak_amplitude"] > 0

            durations.add(res.duration_seconds)

        # Across varying text lengths, multiple distinct audio durations are produced
        assert len(durations) > 5, f"Expected diverse audio durations, got {len(durations)}"
        assert elapsed < 25.0, f"50-character bark synthesis too slow: {elapsed:.2f}s"

    def test_voice_synthesis_fail_fast_boundaries_rule5(self, tmp_path: Path):
        """Verifies Rule 5: Zero Silent Fallbacks on empty text, missing backend, and duration bounds."""
        # 1. Missing client/backend raises StudioAgentError
        with pytest.raises(StudioAgentError):
            VoiceActingAudioProducerAgent(client=None, backend=None)

        agent = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        out_wav = tmp_path / "bad.wav"

        # 2. Empty text raises ValueError immediately
        with pytest.raises(ValueError, match="TTS text cannot be empty"):
            agent.synthesize_voice_line(bark={"text": ""}, output_path=out_wav)

        with pytest.raises(ValueError, match="TTS text cannot be empty"):
            agent.synthesize_voice_line(bark={"text": "   "}, output_path=out_wav)


# ============================================================================
# Challenge 2: Concurrent Multi-Threaded Sprite Generation
# ============================================================================

class TestConcurrentSpriteGeneration:
    """Stress-test concurrent sprite generation, multi-threaded rendering, and large manifests."""

    def test_concurrent_multi_threaded_cartridge_sprite_synthesis(self, tmp_path: Path):
        """Spawns 12 concurrent worker threads generating dynamic sprite sets simultaneously
        without race conditions, thread contention, or corrupted file headers.
        """
        designer = VisualArtSpriteDesignerAgent()
        thread_count = 12
        assets_per_thread = 8

        def worker_task(tid: int) -> Tuple[int, int, float]:
            worker_dir = tmp_path / f"worker_{tid}"
            manifest = [
                AssetRequirement(
                    deliberation=f"Thread {tid} entity {j}",
                    asset_id=f"t{tid}_entity_{j}",
                    asset_type="sprite" if j % 2 == 0 else "tile",
                    target_path=f"assets/sprites/t{tid}_entity_{j}.png",
                    visual_prompt_or_text=f"Futuristic unit {tid}-{j}",
                    dimensions=(16 + (j % 3) * 16, 16 + (j % 3) * 16),
                )
                for j in range(assets_per_thread)
            ]
            t0 = time.perf_counter()
            sprites = designer.generate_cartridge_sprites(
                output_dir=worker_dir,
                mood="neon_cyberpunk" if tid % 2 == 0 else "retro_arcade",
                asset_manifest=manifest,
            )
            duration = time.perf_counter() - t0
            return tid, len(sprites), duration

        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            for tid in range(thread_count):
                tasks.append(executor.submit(worker_task, tid))

        results = [t.result() for t in concurrent.futures.as_completed(tasks)]
        assert len(results) == thread_count

        for tid, sprite_count, dur in results:
            assert sprite_count >= assets_per_thread
            assert dur < 5.0, f"Worker {tid} generation too slow: {dur:.3f}s"

            # Verify every generated sprite in worker dir has strictly valid PNG signature and mode 'P'
            worker_dir = tmp_path / f"worker_{tid}"
            png_files = list(worker_dir.glob("*.png"))
            assert len(png_files) == assets_per_thread
            for pf in png_files:
                raw = pf.read_bytes()
                assert len(raw) > 8
                assert raw[:8] == PNG_SIGNATURE, f"Corrupted PNG signature in {pf}"
                with Image.open(pf) as img:
                    assert img.format == "PNG"
                    assert img.mode == "P"
                    assert img.size[0] > 0 and img.size[1] > 0

    def test_large_manifest_batch_synthesis_60_assets(self, tmp_path: Path):
        """Stress-tests a single large manifest with 60 diverse visual assets across various
        dimensions (8x8 up to 128x128, non-square 16x32, 64x16), validating all outputs.
        """
        designer = VisualArtSpriteDesignerAgent()
        out_dir = tmp_path / "large_manifest_60"

        dimensions_pool = [
            (8, 8), (16, 16), (24, 24), (32, 32),
            (48, 48), (64, 64), (128, 128),
            (16, 32), (32, 16), (64, 32), (32, 64),
        ]
        asset_types = ["sprite", "tile", "portrait", "entity", "hazard", "item"]

        manifest: List[AssetRequirement] = []
        for i in range(60):
            dims = dimensions_pool[i % len(dimensions_pool)]
            atype = asset_types[i % len(asset_types)]
            manifest.append(
                AssetRequirement(
                    deliberation=f"Asset {i} of type {atype} at {dims}",
                    asset_id=f"asset_req_{i:02d}",
                    asset_type=atype,
                    target_path=f"assets/sprites/asset_{i:02d}.png",
                    visual_prompt_or_text=f"Pixel art {atype} {i} prompt details",
                    dimensions=dims,
                )
            )

        t0 = time.perf_counter()
        sprites = designer.generate_cartridge_sprites(out_dir, mood="grim_dark", asset_manifest=manifest)
        elapsed = time.perf_counter() - t0

        assert len(sprites) >= 60
        png_files = list(out_dir.glob("*.png"))
        assert len(png_files) == 60

        for i, pf in enumerate(sorted(png_files)):
            dims = dimensions_pool[i % len(dimensions_pool)]
            raw = pf.read_bytes()
            assert raw[:8] == PNG_SIGNATURE
            with Image.open(pf) as img:
                assert img.size == dims, f"Dimension mismatch for {pf.name}: expected {dims}, got {img.size}"
                assert img.mode == "P"

        assert elapsed < 5.0, f"60-asset batch generation too slow: {elapsed:.2f}s"


# ============================================================================
# Challenge 3: Quantization Fidelity & Style Bible Palette Conformance
# ============================================================================

class TestQuantizationFidelityAndPaletteConformance:
    """Verifies strict 16-color Style Bible palette quantization across all visual moods."""

    @pytest.mark.parametrize("mood", ["retro_arcade", "neon_cyberpunk", "grim_dark", "solar_punk", "pastel_dream"])
    def test_strict_palette_conformance_across_all_moods(self, tmp_path: Path, mood: str):
        """Audits every pixel generated across all 5 visual moods using Gate E AssetValidator."""
        designer = VisualArtSpriteDesignerAgent()
        out_dir = tmp_path / f"mood_{mood}"
        validator = AssetValidator(enforce_palette=True)

        manifest = [
            AssetRequirement(
                deliberation=f"Hero unit for {mood}",
                asset_id=f"{mood}_hero",
                asset_type="sprite",
                target_path=f"{mood}_hero.png",
                dimensions=(32, 32),
            ),
            AssetRequirement(
                deliberation=f"Environment tile for {mood}",
                asset_id=f"{mood}_tile",
                asset_type="tile",
                target_path=f"{mood}_tile.png",
                dimensions=(32, 32),
            ),
            AssetRequirement(
                deliberation=f"Hazard for {mood}",
                asset_id=f"{mood}_hazard",
                asset_type="hazard",
                target_path=f"{mood}_hazard.png",
                dimensions=(32, 32),
            ),
            AssetRequirement(
                deliberation=f"Portrait for {mood}",
                asset_id=f"{mood}_portrait",
                asset_type="portrait",
                target_path=f"{mood}_portrait.png",
                dimensions=(64, 64),
            ),
        ]

        sprites = designer.generate_cartridge_sprites(out_dir, mood=mood, asset_manifest=manifest)
        assert len(sprites) >= 4

        for pf in out_dir.glob("*.png"):
            # Gate E AssetValidator strict palette audit
            report = validator.validate_visual_asset(pf, min_dimension=8, max_dimension=1024)
            assert report.passed, f"Gate E rejected {pf.name} under mood '{mood}': {report.errors}"

            # Verify every opaque pixel belongs strictly to STYLE_BIBLE_16_PALETTE
            with Image.open(pf) as img:
                assert img.mode == "P"
                rgba = img.convert("RGBA")
                w, h = rgba.size
                for y in range(h):
                    for x in range(w):
                        r, g, b, a = rgba.getpixel((x, y))
                        if a >= 128:
                            assert (r, g, b) in validator.palette_set, (
                                f"Pixel ({x}, {y}) color ({r}, {g}, {b}) outside Style Bible palette in {pf.name}"
                            )

    def test_procedural_renderer_dimension_boundary_validation(self):
        """Verifies non-positive dimensions raise ValueError fail-fast immediately (Rule 5)."""
        renderer = ProceduralSpriteRenderer()

        invalid_dimensions = [
            (0, 0),
            (0, 32),
            (32, 0),
            (-1, 32),
            (32, -1),
            (-16, -16),
        ]
        for dims in invalid_dimensions:
            with pytest.raises(ValueError, match="Sprite dimensions must be positive integers"):
                renderer.render_dynamic_sprite(dimensions=dims)

    def test_procedural_renderer_determinism_and_entropy(self):
        """Verifies identical seed produces bit-identical images, while different seeds produce distinct images."""
        renderer = ProceduralSpriteRenderer()

        # Same seed, prompt, and dimensions -> Identical PNG bytes
        img1 = renderer.render_dynamic_sprite(dimensions=(32, 32), prompt="cybernetic mech", seed="mech_alpha")
        img2 = renderer.render_dynamic_sprite(dimensions=(32, 32), prompt="cybernetic mech", seed="mech_alpha")

        b1 = io.BytesIO()
        b2 = io.BytesIO()
        img1.save(b1, "PNG")
        img2.save(b2, "PNG")
        assert b1.getvalue() == b2.getvalue(), "Deterministic rendering failed: non-identical bytes for same seed"

        # Different seed -> Distinct PNG bytes
        img3 = renderer.render_dynamic_sprite(dimensions=(32, 32), prompt="cybernetic mech", seed="mech_beta")
        b3 = io.BytesIO()
        img3.save(b3, "PNG")
        assert b1.getvalue() != b3.getvalue(), "Entropy failure: identical bytes for different seeds"


# ============================================================================
# Challenge 4: Backward Compatibility & Arena Asset Isolation
# ============================================================================

class TestBackwardCompatibilityAndIsolation:
    """Verifies legacy callers without manifest receive working base sprites,
    while manifest callers receive ZERO legacy arena assets.
    """

    def test_legacy_mode_generates_all_11_base_sprites(self, tmp_path: Path):
        """Calling generate_cartridge_sprites with no manifest produces all 11 legacy base sprites."""
        designer = VisualArtSpriteDesignerAgent()
        legacy_dir = tmp_path / "legacy_sprites"
        validator = AssetValidator(enforce_palette=True)

        sprites = designer.generate_cartridge_sprites(legacy_dir, mood="retro_arcade")

        expected_legacy_keys = [
            "player_idle",
            "player_walk1",
            "player_walk2",
            "player_dash",
            "chaser",
            "sniper",
            "boss",
            "spikes",
            "acid",
            "wall",
            "floor",
        ]

        for key in expected_legacy_keys:
            assert key in sprites, f"Missing legacy sprite key: {key}"
            p = sprites[key]
            assert p.exists(), f"Legacy sprite file does not exist: {p}"
            assert p.stat().st_size > 0

            # Pass Gate E audit
            report = validator.validate_visual_asset(p)
            assert report.passed, f"Legacy sprite {key} failed Gate E: {report.errors}"

        assert len(list(legacy_dir.glob("*.png"))) == 11

    def test_manifest_mode_strictly_isolates_and_excludes_legacy_arena_sprites(self, tmp_path: Path):
        """Calling generate_cartridge_sprites with a manifest strictly generates declared assets
        and produces ZERO unrequested legacy arena assets (chaser.png, sniper.png, etc.).
        """
        designer = VisualArtSpriteDesignerAgent()
        manifest_dir = tmp_path / "manifest_sprites"

        # Game with distinct custom theme (Undersea Exploration)
        manifest = [
            AssetRequirement(
                deliberation="Submersible hull sprite",
                asset_id="submersible",
                asset_type="sprite",
                target_path="assets/sprites/submersible.png",
                visual_prompt_or_text="Exploration submarine",
                dimensions=(48, 48),
            ),
            AssetRequirement(
                deliberation="Bioluminescent coral obstacle",
                asset_id="coral_reef",
                asset_type="tile",
                target_path="assets/sprites/coral_reef.png",
                visual_prompt_or_text="Bioluminescent coral reef",
                dimensions=(32, 32),
            ),
        ]

        sprites = designer.generate_cartridge_sprites(manifest_dir, mood="retro_arcade", asset_manifest=manifest)

        # Custom assets present
        assert "submersible" in sprites
        assert "coral_reef" in sprites
        assert (manifest_dir / "submersible.png").exists()
        assert (manifest_dir / "coral_reef.png").exists()

        # Legacy arena sprites strictly absent
        forbidden_legacy_files = [
            "chaser.png",
            "sniper.png",
            "boss.png",
            "spikes.png",
            "acid.png",
            "wall.png",
            "floor.png",
        ]
        for f in forbidden_legacy_files:
            assert not (manifest_dir / f).exists(), f"Forbidden legacy sprite '{f}' found in manifest build!"
            assert f.replace(".png", "") not in sprites

    def test_legacy_voice_archetype_resolution(self):
        """Verifies legacy character archetypes continue resolving smoothly for backward compatibility."""
        agent = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())

        legacy_expected = {
            "hero": "Aoede",
            "protagonist": "Aoede",
            "boss": "Fenrir",
            "antagonist": "Fenrir",
            "rival": "Charon",
            "npc": "Puck",
            "companion": "Puck",
            "narrator": "Kore",
            "system": "Kore",
        }
        for archetype, expected_voice in legacy_expected.items():
            assert agent.resolve_voice_name(archetype) == expected_voice, (
                f"Legacy archetype '{archetype}' resolved to '{agent.resolve_voice_name(archetype)}', expected '{expected_voice}'"
            )
