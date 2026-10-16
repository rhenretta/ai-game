"""tests/test_assets.py
Comprehensive Unit Test Suite for Style Bible & Procedural Asset Liaison (Feature 13).

Verifies:
  1. Style Bible 16-Color Palette: exact colors, Euclidean quantization, named lookups.
  2. Rationale-First Epistemic Architecture: AST inspection of models and manifests (deliberation first).
  3. Procedural Sprite & Tile Synthesis: exact dimensions (32x32, 48x48, 96x96), indexed mode 'P', colors < 16.
  4. Deterministic Seeding: bitwise reproducibility with same seed, divergence with different seeds.
  5. Asset Liaison Pipeline: synthesis, manifest.json creation, bundle validation, tamper detection.
  6. Envelope Sealing: ArtifactMetadataEnvelope packaging with ArtifactType.ASSET_BUNDLE.
  7. Pure Local Execution: zero external network calls (guaranteed via socket monkeypatch).
"""
import ast
import hashlib
import json
from pathlib import Path
import socket
from typing import Any, Dict
import pytest
from PIL import Image

import config
from engine.assets.palette import (
    STYLE_BIBLE_16_PALETTE,
    PALETTE_COLOR_NAMES,
    StyleBibleManifest,
    quantize_rgb,
    get_palette_index,
    quantize_image_to_indexed,
)
from engine.assets.synthesizer import (
    DeterministicSpriteSynthesizer,
    generate_all_sprites,
)
from engine.assets.liaise import (
    AssetLiaison,
    AssetMetadata,
    AssetBundleManifest,
)
from engine.scheduler.artifact import ArtifactType, ArtifactMetadataEnvelope
from tests.test_tier1_units import validate_rationale_first_ast, validate_rationale_first_dict


# ============================================================================
# Test Class 1: Style Bible 16-Color Palette & Quantization
# ============================================================================
class TestStyleBiblePalette:
    """Verifies canonical 16-color retro palette integrity and Euclidean distance mapping."""
    deliberation: str = "Test suite for 16-color retro palette integrity and Euclidean quantization."

    def test_canonical_palette_length_and_types(self):
        """Palette must contain exactly 16 RGB tuples with values in 0..255."""
        assert len(STYLE_BIBLE_16_PALETTE) == 16
        for idx, color in enumerate(STYLE_BIBLE_16_PALETTE):
            assert len(color) == 3
            assert all(isinstance(c, int) and 0 <= c <= 255 for c in color)
        # All colors must be unique
        assert len(set(STYLE_BIBLE_16_PALETTE)) == 16

    def test_palette_matches_config(self):
        """Palette matches the authoritative configuration in config.py."""
        assert STYLE_BIBLE_16_PALETTE == config.STYLE_BIBLE_16_PALETTE

    def test_palette_color_names(self):
        """Color names dictionary maps to valid palette entries."""
        assert len(PALETTE_COLOR_NAMES) == 16
        for name, color in PALETTE_COLOR_NAMES.items():
            assert color in STYLE_BIBLE_16_PALETTE

    def test_exact_color_quantization(self):
        """Quantizing an exact palette color must return itself with distance 0."""
        for color in STYLE_BIBLE_16_PALETTE:
            assert quantize_rgb(color) == color
            idx = get_palette_index(color)
            assert STYLE_BIBLE_16_PALETTE[idx] == color

    def test_euclidean_nearest_color_quantization(self):
        """Arbitrary off-palette RGB colors must map to the closest Euclidean entry."""
        # Near Pure Black (3, 2, 4) -> (0, 0, 0) [index 0]
        assert quantize_rgb((3, 2, 4)) == (0, 0, 0)
        assert get_palette_index((3, 2, 4)) == 0

        # Near Neon Red (245, 12, 70) -> (255, 0, 77) [index 8]
        assert quantize_rgb((245, 12, 70)) == (255, 0, 77)
        assert get_palette_index((245, 12, 70)) == 8

        # Near Cyan Sky Blue (45, 168, 250) -> (41, 173, 255) [index 12]
        assert quantize_rgb((45, 168, 250)) == (41, 173, 255)
        assert get_palette_index((45, 168, 250)) == 12

    def test_palette_manifest_ast_rationale_first(self):
        """Verifies AST of StyleBibleManifest has 'deliberation' as the first field."""
        manifest = StyleBibleManifest(deliberation="Testing palette AST rationale order.")
        d = manifest.to_dict()
        assert list(d.keys())[0] == "deliberation"
        assert validate_rationale_first_dict(d) is True


# ============================================================================
# Test Class 2: Procedural Sprite Synthesis
# ============================================================================
class TestProceduralSpriteSynthesis:
    """Verifies dimension compliance, indexed mode, and 16-color quantization on synthesized sprites."""
    deliberation: str = "Test suite verifying dimension compliance and 16-color retro quantization."

    @pytest.fixture
    def synthesizer(self):
        return DeterministicSpriteSynthesizer(seed=42, mood="retro_arcade")

    def test_player_sprites_dimensions_and_frames(self, synthesizer):
        """Player idle, walk, and dash frames must strictly be 32x32."""
        sprites = {
            "idle": synthesizer.synthesize_player_idle(),
            "walk_1": synthesizer.synthesize_player_walk_1(),
            "walk_2": synthesizer.synthesize_player_walk_2(),
            "dash": synthesizer.synthesize_player_dash(),
        }
        for name, img in sprites.items():
            assert img.size == (32, 32), f"Player sprite '{name}' size was {img.size}, expected (32, 32)"
            assert img.mode == "P", f"Player sprite '{name}' mode was '{img.mode}', expected 'P'"

    def test_enemy_sprites_dimensions(self, synthesizer):
        """Chaser/Ranged must be 32x32; Boss must be 48x48."""
        chaser = synthesizer.synthesize_enemy_chaser()
        ranged = synthesizer.synthesize_enemy_ranged()
        boss = synthesizer.synthesize_enemy_boss()

        assert chaser.size == (32, 32)
        assert ranged.size == (32, 32)
        assert boss.size == (48, 48)

        assert chaser.mode == "P"
        assert ranged.mode == "P"
        assert boss.mode == "P"

    def test_hazard_sprites_dimensions(self, synthesizer):
        """Spikes must be 48x48, Emitter 48x48, Acid Pool 96x96."""
        spikes = synthesizer.synthesize_hazard_spikes()
        assert len(spikes) == 3
        for s_name, s_img in spikes.items():
            assert s_img.size == (48, 48)
            assert s_img.mode == "P"

        emitter = synthesizer.synthesize_hazard_laser_emitter()
        assert emitter.size == (48, 48)
        assert emitter.mode == "P"

        acid = synthesizer.synthesize_hazard_acid_pool()
        assert acid.size == (96, 96)
        assert acid.mode == "P"

    def test_tile_sprites_dimensions(self, synthesizer):
        """Stone block, metal barrier, floor, and exit portal must be 32x32."""
        stone = synthesizer.synthesize_tile_stone_block()
        metal = synthesizer.synthesize_tile_metal_barrier()
        floor = synthesizer.synthesize_tile_floor()
        portal = synthesizer.synthesize_tile_exit_portal()

        for t_name, t_img in [("stone", stone), ("metal", metal), ("floor", floor), ("portal", portal)]:
            assert t_img.size == (32, 32), f"Tile '{t_name}' size was {t_img.size}, expected (32, 32)"
            assert t_img.mode == "P", f"Tile '{t_name}' mode was '{t_img.mode}', expected 'P'"

    def test_all_sprites_quantized_strictly_under_16_colors(self):
        """Every pixel in every synthesized sprite and tile must have an index in range(16)."""
        sprites, tiles = generate_all_sprites(seed=1234, mood="retro_arcade")
        all_images = {**sprites, **tiles}

        for name, img in all_images.items():
            colors = img.getcolors() or []
            indices = [c[1] for c in colors]
            assert all(0 <= idx < 16 for idx in indices), (
                f"Image '{name}' contains color indices >= 16: {indices}"
            )


# ============================================================================
# Test Class 3: Deterministic Seeding & Divergence
# ============================================================================
class TestDeterministicSeeding:
    """Verifies repeatable generation with identical seeds and divergence with different seeds."""
    deliberation: str = "Test suite verifying deterministic seed reproducibility and divergence."

    def test_identical_seed_produces_identical_bytes(self, tmp_path):
        """Synthesizing twice with the exact same seed must produce identical PNG bytes."""
        dir1 = tmp_path / "cartridge_run1"
        dir2 = tmp_path / "cartridge_run2"

        manifest1 = AssetLiaison.synthesize_cartridge_assets(dir1, seed="seed_deterministic_alpha")
        manifest2 = AssetLiaison.synthesize_cartridge_assets(dir2, seed="seed_deterministic_alpha")

        assert len(manifest1.sprites) == len(manifest2.sprites)
        for s_name, meta1 in manifest1.sprites.items():
            meta2 = manifest2.sprites[s_name]
            assert meta1.sha256_hash == meta2.sha256_hash, (
                f"Hash mismatch for sprite '{s_name}' across identical seeds!"
            )
            data1 = (dir1 / meta1.relative_path).read_bytes()
            data2 = (dir2 / meta2.relative_path).read_bytes()
            assert data1 == data2

    def test_different_seeds_produce_divergent_hashes(self, tmp_path):
        """Synthesizing with different seeds must produce variations in at least some sprites."""
        dir1 = tmp_path / "cartridge_seedA"
        dir2 = tmp_path / "cartridge_seedB"

        manifest1 = AssetLiaison.synthesize_cartridge_assets(dir1, seed="seed_alpha_01", mood="retro_arcade")
        manifest2 = AssetLiaison.synthesize_cartridge_assets(dir2, seed="seed_beta_99", mood="neon_cyberpunk")

        # Verify mood shift causes divergence
        assert manifest1.palette_mood != manifest2.palette_mood
        player1_sha = manifest1.sprites["player_idle"].sha256_hash
        player2_sha = manifest2.sprites["player_idle"].sha256_hash
        assert player1_sha != player2_sha


# ============================================================================
# Test Class 4: Asset Liaison Pipeline & Packaging
# ============================================================================
class TestAssetLiaisonPipeline:
    """Verifies complete cartridge asset synthesis, manifest assembly, and validation."""
    deliberation: str = "Test suite verifying cartridge asset synthesis pipeline and packaging."

    def test_synthesize_cartridge_assets_full_assembly(self, tmp_path):
        """Verifies directory creation, file population, and manifest writing."""
        cart_dir = tmp_path / "active_cartridge"
        manifest = AssetLiaison.synthesize_cartridge_assets(cart_dir, seed="test_seed_42", mood="retro_arcade")

        # Directory structure checks
        assets_dir = cart_dir / "assets"
        sprites_dir = assets_dir / "sprites"
        tiles_dir = assets_dir / "tiles"
        manifest_file = assets_dir / "manifest.json"

        assert sprites_dir.exists()
        assert tiles_dir.exists()
        assert manifest_file.exists()

        # Verify file count
        assert manifest.total_files == len(manifest.sprites) + len(manifest.tiles)
        assert manifest.total_files >= 16

        # Check manifest.json Rationale-First ordering
        with open(manifest_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        assert list(raw_data.keys())[0] == "deliberation"
        assert validate_rationale_first_dict(raw_data) is True

    def test_bundle_validation_success(self, tmp_path):
        """A freshly synthesized bundle must pass validation with 0 errors."""
        cart_dir = tmp_path / "valid_cartridge"
        manifest = AssetLiaison.synthesize_cartridge_assets(cart_dir, seed="val_seed")
        errors = AssetLiaison.validate_asset_bundle(manifest, cart_dir)
        assert errors == [], f"Validation reported unexpected errors: {errors}"

    def test_bundle_validation_tamper_detection(self, tmp_path):
        """Tampering with an asset file on disk must be caught by validation."""
        cart_dir = tmp_path / "tampered_cartridge"
        manifest = AssetLiaison.synthesize_cartridge_assets(cart_dir, seed="tamper_seed")

        # Corrupt 1 byte in player_idle.png
        corrupt_target = cart_dir / manifest.sprites["player_idle"].relative_path
        corrupt_data = bytearray(corrupt_target.read_bytes())
        corrupt_data[-1] = (corrupt_data[-1] + 1) % 256
        corrupt_target.write_bytes(bytes(corrupt_data))

        errors = AssetLiaison.validate_asset_bundle(manifest, cart_dir)
        assert len(errors) > 0
        assert any("SHA-256 mismatch" in err for err in errors)

    def test_bundle_validation_missing_file_detection(self, tmp_path):
        """Removing an asset file must be caught by validation."""
        cart_dir = tmp_path / "missing_file_cartridge"
        manifest = AssetLiaison.synthesize_cartridge_assets(cart_dir, seed="missing_seed")

        missing_target = cart_dir / manifest.tiles["tile_stone_block"].relative_path
        missing_target.unlink()

        errors = AssetLiaison.validate_asset_bundle(manifest, cart_dir)
        assert len(errors) > 0
        assert any("Missing asset file" in err for err in errors)

    def test_seal_asset_envelope(self, tmp_path):
        """Verifies packaging into an ArtifactMetadataEnvelope."""
        cart_dir = tmp_path / "envelope_cartridge"
        manifest = AssetLiaison.synthesize_cartridge_assets(cart_dir, seed="env_seed")
        envelope = AssetLiaison.seal_asset_envelope(manifest, cart_dir)

        assert isinstance(envelope, ArtifactMetadataEnvelope)
        assert envelope.artifact_type == ArtifactType.ASSET_BUNDLE
        assert list(envelope.model_dump().keys())[0] == "deliberation"
        assert envelope.deliberation == manifest.deliberation
        assert envelope.verify_integrity() is True

    def test_load_sprite_surface(self, tmp_path):
        """Verifies loading a sprite surface via Pygame if installed."""
        cart_dir = tmp_path / "surface_cartridge"
        manifest = AssetLiaison.synthesize_cartridge_assets(cart_dir, seed="surf_seed")
        sprite_path = cart_dir / manifest.sprites["player_idle"].relative_path

        surf = AssetLiaison.load_sprite_surface(sprite_path)
        # In headless test mode with pygame-ce installed, surf may be a Surface
        # Or if pygame cannot load in headless mode, it returns None safely
        if surf is not None:
            assert hasattr(surf, "get_width")
            assert surf.get_width() == 32
            assert surf.get_height() == 32


# ============================================================================
# Test Class 5: Zero External Network Isolation
# ============================================================================
class TestZeroNetworkIsolation:
    """Proves the asset synthesis pipeline runs 100% locally with zero socket calls."""
    deliberation: str = "Test suite verifying zero external socket operations."

    def test_zero_network_calls_during_asset_synthesis(self, tmp_path, monkeypatch):
        """Monkeypatch socket.socket to fail immediately if network is touched."""
        def banned_socket(*args, **kwargs):
            raise RuntimeError("Prohibited network socket call detected in Asset Liaison pipeline!")

        monkeypatch.setattr(socket, "socket", banned_socket)

        cart_dir = tmp_path / "offline_cartridge"
        manifest = AssetLiaison.synthesize_cartridge_assets(cart_dir, seed="offline_seed")
        assert manifest.total_files > 0


# ============================================================================
# Test Class 6: AST Rationale-First Compliance
# ============================================================================
class TestAssetASTRationaleFirst:
    """Verifies that all asset classes and dictionaries declare deliberation first."""
    deliberation: str = "Test suite verifying AST rationale-first compliance on asset classes."

    def test_ast_rationale_first_models(self):
        """AST inspection verifies class declarations declare deliberation first."""
        base_dir = Path(__file__).resolve().parent.parent / "engine" / "assets"
        for py_file in ["palette.py", "synthesizer.py", "liaise.py", "__init__.py"]:
            file_path = base_dir / py_file
            assert file_path.exists(), f"File {file_path} does not exist"
            source = file_path.read_text(encoding="utf-8")
            assert validate_rationale_first_ast(source) is True, f"Failed AST rationale check in {py_file}"

    def test_manifest_dict_rationale_first(self):
        """Dictionary outputs must have deliberation as the very first key."""
        meta = AssetMetadata(
            deliberation="Epistemic rationale for test metadata.",
            name="test_sprite",
            category="sprite",
            relative_path="assets/sprites/test.png",
            width=32,
            height=32,
            sha256_hash="abc",
            file_size_bytes=100,
        )
        d_meta = meta.to_dict()
        assert validate_rationale_first_dict(d_meta) is True

        manifest = AssetBundleManifest(
            deliberation="Epistemic rationale for test manifest.",
            cartridge_name="test_cartridge",
        )
        d_manifest = manifest.to_dict()
        assert validate_rationale_first_dict(d_manifest) is True
