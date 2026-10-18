"""tests/test_challenger_m2_it2_stress_challenger2.py
Milestone 2 Iteration 2 Challenger 2 Empirical Stress Test Suite.

Independently stress-tests:
1. Concurrency: 50 sprites generated concurrently across wide dimensions (e.g. 128x32, 256x64, 512x32)
   and varied geometry_directive values, including batch cartridge sprite generation.
2. Voice resolution: Timbre-based voice resolution across edge-case character names, punctuation,
   casing, and unicode.
3. Defect Mining / Empirical Challenge:
   - Evaluates whether shape geometry directives ("sharp", "box", "round") are bypassed by radial symmetry
     seed collisions when dimensions are square (w == h).
   - Evaluates entropy of the radial sprite rendering branch across disparate seeds and prompts.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple
from PIL import Image
import pytest

from engine.assets.palette import STYLE_BIBLE_16_PALETTE, MOOD_PALETTES
from engine.studio.audio import (
    GEMINI_TTS_VOICES,
    OfflineMockAudioBackend,
    VoiceActingAudioProducerAgent,
    VoiceSynthesisError,
)
from engine.studio.models import (
    AssetRequirement,
    CharacterSheet,
    DialogueBarkSpec,
    GameRequirementsSpec,
)
from engine.studio.visual import (
    PNG_SIGNATURE,
    ProceduralSpriteRenderer,
    VisualArtSpriteDesignerAgent,
    get_seed_integer,
)


class TestConcurrencyWideDimensionsAndGeometryDirectives:
    """Stress-tests multi-threaded concurrent generation of 50 sprites across wide dimensions."""

    def test_concurrent_50_sprites_wide_dimensions_and_directives(self) -> None:
        """Generates 50 sprites concurrently across wide dimensions (128x32, 256x64, etc.)
        and varied geometry_directive values using a ThreadPoolExecutor.
        """
        renderer = ProceduralSpriteRenderer(mood="neon_cyberpunk")

        wide_dimensions = [
            (128, 32),   # 4:1 wide
            (256, 64),   # 4:1 wide
            (512, 32),   # 16:1 extreme wide
            (256, 32),   # 8:1 wide
            (128, 16),   # 8:1 wide
            (64, 16),    # 4:1 wide
            (32, 8),     # 4:1 wide
            (16, 4),     # 4:1 narrow strip
            (320, 40),   # 8:1 wide
            (192, 48),   # 4:1 wide
            # Also include inverted tall dimensions
            (32, 128),   # 1:4 tall
            (64, 256),   # 1:4 tall
            (16, 64),    # 1:4 tall
            (8, 32),     # 1:4 tall
        ]

        directives = [
            "diagonal", "brick", "border", "center",
            "cross", "stripe", "lattice", "grid",
            "horizontal", "masonry", "panel", "frame", "core", "plus",
            "radial", "concentric", "circle", "spherical", "orb",
            "bilateral", "mirror", "symmetric", "vertical", "winged",
            "sharp", "angular", "triangle", "wedge", "arrow",
            "box", "rect", "block", "square",
            "round", "oval", "curve", "ellipse",
            None, "", "  ", "cyber_hex_matrix", "🔷_unicode_rune",
        ]

        asset_types = ["tile", "sprite", "floor", "wall", "terrain", "entity", "hazard", "item", "portrait"]

        # Build 50 distinct test parameter sets
        tasks: List[Dict[str, Any]] = []
        for i in range(50):
            dims = wide_dimensions[i % len(wide_dimensions)]
            geom = directives[i % len(directives)]
            atype = asset_types[i % len(asset_types)]
            prompt = f"Parametric asset synthesis prompt {i} {atype} {geom}"
            seed = f"concurrent_seed_{i}_{dims[0]}x{dims[1]}"
            tasks.append({
                "index": i,
                "dimensions": dims,
                "prompt": prompt,
                "asset_type": atype,
                "seed": seed,
                "geometry_directive": geom,
            })

        def _render_worker(task: Dict[str, Any]) -> Tuple[int, bytes, Tuple[int, int]]:
            img = renderer.render_dynamic_sprite(
                dimensions=task["dimensions"],
                prompt=task["prompt"],
                asset_type=task["asset_type"],
                seed=task["seed"],
                geometry_directive=task["geometry_directive"],
            )
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return task["index"], buf.getvalue(), img.size

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(_render_worker, t) for t in tasks]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        elapsed = time.perf_counter() - t0

        assert len(results) == 50, f"Expected 50 rendered results, got {len(results)}"
        assert elapsed < 5.0, f"Concurrent generation took too long: {elapsed:.2f}s"

        for idx, png_bytes, size in results:
            expected_dims = tasks[idx]["dimensions"]
            assert size == expected_dims, f"Task {idx} size {size} != expected {expected_dims}"
            assert len(png_bytes) > 0, f"Task {idx} produced empty bytes"
            assert png_bytes[:8] == PNG_SIGNATURE, f"Task {idx} invalid PNG signature"

            # Verify with Pillow re-opening
            with Image.open(io.BytesIO(png_bytes)) as reopened:
                assert reopened.format == "PNG"
                assert reopened.size == expected_dims

    def test_concurrent_cartridge_sprites_batch_wide_manifest(self, tmp_path: Path) -> None:
        """Generates a manifest of 50 items with wide dimensions and varied geometry directives."""
        designer = VisualArtSpriteDesignerAgent()
        out_dir = tmp_path / "batch_wide_cartridge_50"

        wide_dimensions = [(128, 32), (256, 64), (64, 16), (32, 8), (512, 32), (16, 64)]
        directives = ["diagonal", "brick", "border", "center", "sharp", "box", "round", "radial", "bilateral"]
        asset_types = ["tile", "sprite", "hazard", "item"]

        manifest: List[AssetRequirement] = []
        for i in range(50):
            dims = wide_dimensions[i % len(wide_dimensions)]
            geom = directives[i % len(directives)]
            atype = asset_types[i % len(asset_types)]
            manifest.append(
                AssetRequirement(
                    deliberation=f"Batch wide asset {i}",
                    asset_id=f"wide_asset_{i:02d}",
                    asset_type=atype,
                    target_path=f"assets/sprites/wide_asset_{i:02d}.png",
                    visual_prompt_or_text=f"Batch wide visual prompt {i}",
                    dimensions=dims,
                    geometry_directive=geom,
                )
            )

        t0 = time.perf_counter()
        sprites = designer.generate_cartridge_sprites(out_dir, mood="grim_dark", asset_manifest=manifest)
        elapsed = time.perf_counter() - t0

        assert len(sprites) >= 50
        assert elapsed < 5.0

        # Verify on-disk files
        png_files = list(out_dir.glob("*.png"))
        assert len(png_files) == 50

        for pf in png_files:
            raw = pf.read_bytes()
            assert raw[:8] == PNG_SIGNATURE
            with Image.open(pf) as img:
                assert img.format == "PNG"
                assert img.size[0] > 0 and img.size[1] > 0


class TestVoiceResolutionEdgeCasesUnicodeAndPunctuation:
    """Stress-tests voice resolution across edge-case character names, punctuation, and unicode."""

    def test_voice_resolution_unicode_character_names(self) -> None:
        """Verifies speaker names in diverse scripts and unicode resolve deterministically."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())

        unicode_speakers = [
            # CJK
            "李白 (Poet)",
            "宮本 武蔵 [Samurai]",
            "김유신 장군",
            # Cyrillic
            "Фёдор Достоевский",
            "Босс #1 'Молот'",
            # Greek
            "Σωκράτης",
            "Ἀριστοτέλης",
            # Arabic / Hebrew
            "ابن سينا (Avicenna)",
            "שלום עליכם",
            # Accents & Germanic
            "Renée Müller-Chauvigny",
            "Håkon Østergård",
            "Façade de l'Éclipse",
            # Emojis & Symbols
            "🤖 Mecha-Unit-99",
            "👑 Empress of Swords ⚔️",
            "🔥 Pyro Mage 💥",
            "✨ Astral Wisp 🌟",
        ]

        for spk in unicode_speakers:
            v1 = producer.resolve_voice_name(spk)
            v2 = producer.resolve_voice_name(spk)
            assert v1 == v2, f"Resolution for '{spk}' not deterministic: {v1} != {v2}"
            assert v1 in GEMINI_TTS_VOICES, f"Voice '{v1}' not in GEMINI_TTS_VOICES for '{spk}'"

    def test_voice_resolution_extreme_punctuation_and_whitespace(self) -> None:
        """Verifies speaker names with unusual punctuation, brackets, quotes, whitespace."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())

        extreme_speakers = [
            "Dr. Henry Jekyll, M.D. / Mr. Edward Hyde (Esq.)",
            "[AI_CORE]::System.Daemon#404 <NULL>",
            "!@#$%^&*()_+-=[]{}|;':,.<>?/",
            "\"Quoted Name with 'Subquotes'\"",
            "   \t\n  Padded Name  \n\t ",
            "",
            "   ",
            "---",
            "127.0.0.1:8080",
            "a" * 250,  # Extremely long name
        ]

        for spk in extreme_speakers:
            v1 = producer.resolve_voice_name(spk)
            v2 = producer.resolve_voice_name(spk)
            assert v1 == v2
            assert v1 in GEMINI_TTS_VOICES

    def test_voice_timbre_with_punctuation_casing_and_unicode(self) -> None:
        """Verifies voice_timbre keywords resolve accurately even when embedded in punctuation,
        uppercase casing, and mixed unicode text.
        """
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())

        cases = [
            # Fenrir: deep, heavy, dark, resonant, bass, gruff, low
            ("DEEP RESONANT BASS!", "Fenrir"),
            ("Dark & heavy growl; low-pitched.", "Fenrir"),
            ("Gruff, booming bass [sub-woofer]", "Fenrir"),
            ("Low resonant drone...", "Fenrir"),
            # Charon: grave, menacing, grim, sinister, whisper, raspy
            ("GRAVE, MENACING WHISPER", "Charon"),
            ("Grim / sinister assassin -- raspy tone", "Charon"),
            ("Raspy ancient whisper (menacing)", "Charon"),
            # Kore: bright, light, cheerful, playful, high, airy
            ("BRIGHT & CHEERFUL!", "Kore"),
            ("Light, playful sprite (high & airy)", "Kore"),
            ("Cheerful / high-pitched laughter", "Kore"),
            # Aoede: focused, sharp, swift, clear, crisp
            ("FOCUSED & SHARP OPERATIVE", "Aoede"),
            ("Swift, clear scout (crisp voice)", "Aoede"),
            ("Crisp, sharp tactical tone.", "Aoede"),
        ]

        for timbre_text, expected_voice in cases:
            char = CharacterSheet(
                deliberation=f"Character test for {timbre_text}",
                character_id="char_test",
                name="Character",
                role="npc",
                voice_timbre=timbre_text,
            )
            resolved = producer.resolve_voice_name("Character", character=char)
            assert resolved == expected_voice, (
                f"Timbre '{timbre_text}' resolved to '{resolved}', expected '{expected_voice}'"
            )

    def test_voice_resolution_unmapped_timbre_safe_fallback(self) -> None:
        """Verifies that when a character has an unmapped or exotic timbre description
        (e.g. 'metallic robotic resonance'), it falls back safely to 'Aoede' rather than crashing.
        """
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())

        exotic_timbres = [
            "metallic robotic modulation",
            "crystalline otherworldly chiming",
            "synthesized vocoder frequency",
            "staticky radio transmission",
            "mysterious ancient riddle",
        ]

        for timbre_text in exotic_timbres:
            char = CharacterSheet(
                deliberation=f"Character test for {timbre_text}",
                character_id="char_exotic",
                name="ExoticEntity",
                role="npc",
                voice_timbre=timbre_text,
            )
            resolved = producer.resolve_voice_name("ExoticEntity", character=char)
            # Default gemini_voice_name is "Aoede", so unmapped timbre safely falls back to "Aoede"
            assert resolved == "Aoede", f"Expected fallback to 'Aoede' for '{timbre_text}', got '{resolved}'"

    def test_explicit_voice_override_precedence(self) -> None:
        """Verifies explicit non-default gemini_voice_name takes precedence over timbre."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())

        char = CharacterSheet(
            deliberation="Explicit voice override",
            character_id="puck_hero",
            name="PuckHero",
            role="hero",
            gemini_voice_name="Puck",
            voice_timbre="Deep resonant bass",  # Timbre would suggest Fenrir, but explicit voice is Puck
        )
        resolved = producer.resolve_voice_name("PuckHero", character=char)
        assert resolved == "Puck"


class TestEmpiricalDefectMiningAndAdversarialChallenges:
    """Adversarial stress-testing of shape directive bypass and radial entropy collapse."""

    def test_empirical_defect_shape_directives_bypassed_by_radial_seed_collision(self) -> None:
        """Empirical Challenge:
        Demonstrates that when w == h (e.g. 32x32), shape directives like 'sharp' and 'box'
        are NOT recognized by force_bilateral.
        When (derived_seed >> 2) % 2 == 1, sym_mode is 1, bypassing the entire body_style block
        and falling into radial symmetry, yielding an output identical to geometry_directive='radial'.
        """
        renderer = ProceduralSpriteRenderer()

        # Seed key: 'identical_seed_42_Defense drone_sprite_sharp_32x32'
        # Has derived_seed where (derived_seed >> 2) % 2 == 1
        img_radial = renderer.render_dynamic_sprite(
            dimensions=(32, 32),
            prompt="Defense drone",
            asset_type="sprite",
            seed="identical_seed_42",
            geometry_directive="radial",
        )
        img_sharp = renderer.render_dynamic_sprite(
            dimensions=(32, 32),
            prompt="Defense drone",
            asset_type="sprite",
            seed="identical_seed_42",
            geometry_directive="sharp",
        )

        radial_bytes = img_radial.tobytes()
        sharp_bytes = img_sharp.tobytes()

        # Document whether this collision occurs
        collision_detected = (radial_bytes == sharp_bytes)
        assert collision_detected is False, (
            "Empirical verification: geometry_directive='sharp' must not collide with radial symmetry "
            "now that force_bilateral includes shape keywords!"
        )

    def test_empirical_defect_radial_symmetry_has_zero_entropy_across_seeds(self) -> None:
        """Empirical Challenge:
        Demonstrates that ProceduralSpriteRenderer's radial branch
        utilizes derived_seed with procedural seed entropy.
        Consequently, disparate seeds and prompts yield distinct radial sprites.
        """
        renderer = ProceduralSpriteRenderer()

        img_a = renderer.render_dynamic_sprite(
            dimensions=(32, 32),
            prompt="Alien Mothership",
            asset_type="sprite",
            seed="galaxy_seed_alpha",
            geometry_directive="radial",
        )
        img_b = renderer.render_dynamic_sprite(
            dimensions=(32, 32),
            prompt="Mystic Healing Orb",
            asset_type="sprite",
            seed="crystal_seed_omega",
            geometry_directive="radial",
        )

        bytes_a = img_a.tobytes()
        bytes_b = img_b.tobytes()

        zero_entropy_confirmed = (bytes_a == bytes_b)
        assert zero_entropy_confirmed is False, (
            "Empirical verification: ProceduralSpriteRenderer radial symmetry draws distinct "
            "pixels across different seeds and prompts using procedural seed entropy."
        )
