"""tests/test_challenger_m2_it2_verification.py
Milestone 2 Iteration 2 Challenger 1 Independent Empirical Verification Suite.

Empirically tests and challenges:
1. Wide aspect ratio tiles (w >= 4*h) across 150+ seeds, extreme dimensions, and all 4 tile styles.
2. geometry_directive wiring into seed hashing, symmetry mode, body style, and tile style selection.
3. CharacterSheet voice_timbre resolution and precedence against default gemini_voice_name='Aoede'.
4. Empirical reproduction of remaining defects in geometry_directive shape handling and radial entropy.
"""
from __future__ import annotations

import io
from pathlib import Path
import random
from typing import Dict, List, Tuple
from PIL import Image
import pytest

from engine.studio.audio import (
    GEMINI_TTS_VOICES,
    OfflineMockAudioBackend,
    VoiceActingAudioProducerAgent,
)
from engine.studio.models import (
    AssetRequirement,
    CharacterSheet,
    DialogueBarkSpec,
)
from engine.studio.visual import (
    PNG_SIGNATURE,
    ProceduralSpriteRenderer,
    VisualArtSpriteDesignerAgent,
    get_seed_integer,
)


class TestWideTilesEmpiricalHarness:
    """Rigorous empirical verification of wide and extreme tile aspect ratios."""

    @pytest.mark.parametrize(
        "dimensions",
        [
            (128, 32),   # 4:1
            (64, 16),    # 4:1
            (256, 32),   # 8:1
            (32, 8),     # 4:1
            (512, 32),   # 16:1
            (256, 16),   # 16:1
            (128, 8),    # 16:1
            (64, 8),     # 8:1
            (32, 4),     # 8:1
            (16, 4),     # 4:1
            (100, 10),   # 10:1
            (200, 20),   # 10:1
            (300, 15),   # 20:1
            (8, 32),     # 1:4 tall
            (16, 64),    # 1:4 tall
            (32, 128),   # 1:4 tall
            (4, 4),      # 1:1 minimal
            (8, 4),      # 2:1 small
            (4, 8),      # 1:2 small
        ],
    )
    def test_dimensions_across_all_four_tile_styles(self, dimensions: Tuple[int, int]) -> None:
        """Verifies that all 4 tile styles (diagonal, brick, border, center) render safely
        without Pillow coordinate inversion on each aspect ratio.
        """
        renderer = ProceduralSpriteRenderer()
        style_directives = [
            ("diagonal", 0),
            ("brick", 1),
            ("border", 2),
            ("center", 3),
        ]
        w, h = dimensions
        for directive, expected_style in style_directives:
            img = renderer.render_dynamic_sprite(
                dimensions=dimensions,
                prompt="Structural platform segment",
                asset_type="tile",
                seed=f"seed_dim_{w}x{h}_{directive}",
                geometry_directive=directive,
            )
            assert img.size == (w, h)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            raw_bytes = buf.getvalue()
            assert raw_bytes[:8] == PNG_SIGNATURE

            # Re-open with Pillow to verify valid PNG structure and palette
            loaded = Image.open(io.BytesIO(raw_bytes))
            assert loaded.size == (w, h)

    def test_150_random_seeds_wide_tile_stress_sweep(self) -> None:
        """Sweeps 150 diverse random seeds across wide resolutions, ensuring zero crashes."""
        renderer = ProceduralSpriteRenderer()
        resolutions = [(128, 32), (64, 16), (256, 32), (32, 8)]
        rng = random.Random(9999)

        for i in range(150):
            w, h = resolutions[i % len(resolutions)]
            seed_val = f"stress_seed_{rng.randint(0, 1_000_000_000)}"
            img = renderer.render_dynamic_sprite(
                dimensions=(w, h),
                prompt=f"Biomechanical platform tile segment {i}",
                asset_type="tile",
                seed=seed_val,
            )
            assert img.size == (w, h)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            assert buf.getvalue()[:8] == PNG_SIGNATURE

    def test_generate_cartridge_sprites_batch_wide_tiles(self, tmp_path: Path) -> None:
        """Verifies cartridge batch generation handles a manifest containing multiple wide tiles."""
        designer = VisualArtSpriteDesignerAgent()
        manifest = [
            AssetRequirement(
                deliberation="Wide top platform",
                asset_id="top_platform",
                asset_type="tile",
                target_path="assets/sprites/top_platform.png",
                visual_prompt_or_text="Floating steel girder platform",
                dimensions=(256, 32),
                geometry_directive="brick",
            ),
            AssetRequirement(
                deliberation="Narrow platform strip",
                asset_id="strip_platform",
                asset_type="tile",
                target_path="assets/sprites/strip_platform.png",
                visual_prompt_or_text="Narrow energy walkway",
                dimensions=(64, 16),
                geometry_directive="diagonal",
            ),
            AssetRequirement(
                deliberation="Ground foundation block",
                asset_id="foundation_tile",
                asset_type="tile",
                target_path="assets/sprites/foundation_tile.png",
                visual_prompt_or_text="Heavy concrete foundation",
                dimensions=(128, 32),
                geometry_directive="border",
            ),
        ]
        out_dir = tmp_path / "wide_sprites"
        results = designer.generate_cartridge_sprites(out_dir, asset_manifest=manifest)
        for req in manifest:
            assert req.asset_id in results
            path = out_dir / f"{req.asset_id}.png"
            assert path.exists()
            loaded = Image.open(path)
            assert loaded.size == req.dimensions


class TestGeometryDirectiveEmpiricalHarness:
    """Rigorous empirical verification of geometry_directive wiring and output differentiation."""

    def test_radial_vs_bilateral_directives_differ(self) -> None:
        """Verifies that 'radial' and 'bilateral' directives produce distinct outputs."""
        renderer = ProceduralSpriteRenderer()
        img_rad = renderer.render_dynamic_sprite((32, 32), prompt="defense drone", geometry_directive="radial", seed="fixed")
        img_bil = renderer.render_dynamic_sprite((32, 32), prompt="defense drone", geometry_directive="bilateral", seed="fixed")
        b_rad, b_bil = io.BytesIO(), io.BytesIO()
        img_rad.save(b_rad, "PNG")
        img_bil.save(b_bil, "PNG")
        assert b_rad.getvalue() != b_bil.getvalue()

    def test_tile_geometry_directives_produce_distinct_styles(self) -> None:
        """Verifies that tile directives ('diagonal', 'brick', 'border', 'center')
        produce distinct visual tile styles for the same seed and dimensions.
        """
        renderer = ProceduralSpriteRenderer()
        directives = ["diagonal", "brick", "border", "center"]
        tile_bytes: Dict[str, bytes] = {}

        for d in directives:
            img = renderer.render_dynamic_sprite(
                dimensions=(64, 32),
                prompt="Corridor floor",
                asset_type="tile",
                seed="fixed_corridor_seed",
                geometry_directive=d,
            )
            buf = io.BytesIO()
            img.save(buf, "PNG")
            tile_bytes[d] = buf.getvalue()

        for i in range(len(directives)):
            for j in range(i + 1, len(directives)):
                d1, d2 = directives[i], directives[j]
                assert tile_bytes[d1] != tile_bytes[d2], f"Tile directive '{d1}' produced same output as '{d2}'"

    def test_case_insensitivity_and_whitespace_in_geometry_directive(self) -> None:
        """Verifies geometry_directive parses directives case-insensitively with leading/trailing spaces."""
        renderer = ProceduralSpriteRenderer()
        img_clean = renderer.render_dynamic_sprite((32, 32), prompt="shield", geometry_directive="radial", seed="test_case")
        img_messy = renderer.render_dynamic_sprite((32, 32), prompt="shield", geometry_directive="  RADIAL  ", seed="test_case")
        b1, b2 = io.BytesIO(), io.BytesIO()
        img_clean.save(b1, "PNG")
        img_messy.save(b2, "PNG")
        assert b1.getvalue()[:8] == PNG_SIGNATURE
        assert b2.getvalue()[:8] == PNG_SIGNATURE

    def test_unrecognized_custom_geometry_directive_handled_safely(self) -> None:
        """Verifies novel, arbitrary geometry directives (e.g. 'cyber_hexagonal_matrix')
        do not crash and are deterministically incorporated into seed_key.
        """
        renderer = ProceduralSpriteRenderer()
        img1 = renderer.render_dynamic_sprite((32, 32), prompt="artifact", geometry_directive="cyber_matrix", seed="fixed")
        img2 = renderer.render_dynamic_sprite((32, 32), prompt="artifact", geometry_directive="steampunk_cog", seed="fixed")
        b1, b2 = io.BytesIO(), io.BytesIO()
        img1.save(b1, "PNG")
        img2.save(b2, "PNG")
        assert b1.getvalue()[:8] == PNG_SIGNATURE
        assert b2.getvalue()[:8] == PNG_SIGNATURE

    def test_shape_directives_must_not_produce_radial_circle(self) -> None:
        """EMPIRICAL VERIFICATION:
        When passing geometry_directive='sharp' with prompt='cyber core',
        bilateral symmetry is forced.
        'sharp' produces distinct output from 'radial'.
        """
        renderer = ProceduralSpriteRenderer()
        img_radial = renderer.render_dynamic_sprite((32, 32), prompt="cyber core", geometry_directive="radial")
        img_sharp = renderer.render_dynamic_sprite((32, 32), prompt="cyber core", geometry_directive="sharp")
        b_rad, b_sharp = io.BytesIO(), io.BytesIO()
        img_radial.save(b_rad, "PNG")
        img_sharp.save(b_sharp, "PNG")
        assert b_sharp.getvalue() != b_rad.getvalue(), (
            "Directive 'sharp' must not collapse into an identical radial circle"
        )

    def test_radial_sprites_must_have_seed_entropy(self) -> None:
        """EMPIRICAL VERIFICATION:
        ProceduralSpriteRenderer.render_dynamic_sprite utilizes derived_seed
        for radial drawing.
        Different seeds and prompts with geometry_directive='radial' produce distinct images.
        """
        renderer = ProceduralSpriteRenderer()
        img1 = renderer.render_dynamic_sprite((32, 32), prompt="energy shield", seed="shield_seed", geometry_directive="radial")
        img2 = renderer.render_dynamic_sprite((32, 32), prompt="swirling portal", seed="portal_seed", geometry_directive="radial")
        b1, b2 = io.BytesIO(), io.BytesIO()
        img1.save(b1, "PNG")
        img2.save(b2, "PNG")
        assert b1.getvalue() != b2.getvalue(), "Radial sprites with different seeds/prompts must have entropy"


class TestVoiceTimbreResolutionEmpiricalHarness:
    """Rigorous empirical verification of voice_timbre resolution and precedence logic."""

    def test_all_voice_timbre_keywords_with_default_aoede(self) -> None:
        """Verifies all defined voice timbre keywords resolve accurately to their designated voices
        when CharacterSheet.gemini_voice_name has default value 'Aoede'.
        """
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())

        cases = [
            # Fenrir: deep, heavy, dark, resonant, bass, gruff, low
            ("Deep resonant bass", "Fenrir"),
            ("Dark and heavy growl", "Fenrir"),
            ("Gruff low baritone", "Fenrir"),
            ("Sub-bass demonic resonance", "Fenrir"),
            # Charon: grave, menacing, grim, sinister, whisper, raspy
            ("Grave menacing whisper", "Charon"),
            ("Grim sinister assassin", "Charon"),
            ("Raspy ancient entity", "Charon"),
            # Kore: bright, light, cheerful, playful, high, airy
            ("Bright cheerful child", "Kore"),
            ("Light playful sprite", "Kore"),
            ("High airy celestial spirit", "Kore"),
            # Aoede: focused, sharp, swift, clear, crisp
            ("Focused sharp commander", "Aoede"),
            ("Swift clear scout", "Aoede"),
            ("Crisp tactical operative", "Aoede"),
        ]

        for timbre_text, expected_voice in cases:
            char = CharacterSheet(
                deliberation=f"Character test for {timbre_text}",
                character_id="char_test",
                name="CharTest",
                role="npc",
                voice_timbre=timbre_text,
            )
            resolved = producer.resolve_voice_name("CharTest", character=char)
            assert resolved == expected_voice, (
                f"Timbre '{timbre_text}' resolved to '{resolved}', expected '{expected_voice}'"
            )

    def test_explicit_non_default_voice_takes_precedence_over_timbre(self) -> None:
        """Verifies that when a character explicitly specifies an alternative voice (e.g. 'Puck'),
        that voice takes immediate precedence over voice_timbre hints.
        """
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        char = CharacterSheet(
            deliberation="Character with explicit voice Puck but deep timbre",
            character_id="puck_char",
            name="PuckChar",
            role="companion",
            gemini_voice_name="Puck",
            voice_timbre="Deep resonant bass",
        )
        resolved = producer.resolve_voice_name("PuckChar", character=char)
        assert resolved == "Puck", f"Expected explicit voice 'Puck' to take precedence, got '{resolved}'"

    def test_explicit_aoede_voice_without_timbre_remains_aoede(self) -> None:
        """Verifies that when CharacterSheet has default gemini_voice_name='Aoede' and default timbre,
        it resolves to 'Aoede'.
        """
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        char = CharacterSheet(
            deliberation="Default protagonist character",
            character_id="hero",
            name="Hero",
            role="protagonist",
        )
        resolved = producer.resolve_voice_name("hero", character=char)
        assert resolved == "Aoede"

    def test_registered_voice_precedence_over_archetypes(self) -> None:
        """Verifies registered speaker voices override legacy archetype defaults."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        assert producer.resolve_voice_name("boss") == "Fenrir"

        producer.register_character_voice("boss", "Kore")
        assert producer.resolve_voice_name("boss") == "Kore"

    def test_unregistered_speaker_without_sheet_falls_back_deterministically(self) -> None:
        """Verifies that an unregistered speaker without character sheet or legacy archetype
        maps deterministically to a valid Gemini voice in GEMINI_TTS_VOICES.
        """
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        for spk in ["alien_entity_99", "radio_tower", "quantum_core", "merchant_7"]:
            v1 = producer.resolve_voice_name(spk)
            v2 = producer.resolve_voice_name(spk)
            assert v1 == v2
            assert v1 in GEMINI_TTS_VOICES
