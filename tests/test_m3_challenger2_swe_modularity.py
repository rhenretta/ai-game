"""tests/test_m3_challenger2_swe_modularity.py
Milestone 3 Challenger 2 Empirical Stress Test Suite.

Adversarially stress-tests:
1. Multi-Module Assembly (OpenGenreCartridgeBundle.assemble_cartridge_directory)
2. Cross-Module Import Verification (LeadGameplaySWEAgent.verify_cartridge_cross_module_imports)
3. Sealed Asset Discovery (LeadGameplaySWEAgent._discover_sealed_assets)
4. Generalizability & Zero Arena Assumptions in swe.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import random
import string
from typing import Any, Dict, List
from unittest.mock import MagicMock
import pytest

from engine.api.contracts import BaseCartridge, BaseScene
from engine.studio.models import (
    AcceptanceCriterion,
    GameRequirementsSpec,
    GenesisWorldSpec,
    RequirementTrack,
)
from engine.studio.swe import (
    CodeVerificationError,
    LeadGameplaySWEAgent,
    OpenGenreCartridgeBundle,
)


# ============================================================================
# 1. Multi-Module Assembly Stress Tests
# ============================================================================

class TestMultiModuleAssemblyAdversarial:
    """Stress tests for OpenGenreCartridgeBundle.assemble_cartridge_directory."""

    def test_assemble_cartridge_directory_all_modules(self, tmp_path: Path) -> None:
        """Verifies complete 4-module + manifest + level.json assembly."""
        dest_dir = tmp_path / "cartridge_full"

        bundle = OpenGenreCartridgeBundle(
            deliberation="Complete multi-module cartridge assembly.",
            manifest_data={
                "deliberation": "Manifest deliberation",
                "name": "Orbital Station Sim",
                "version": "1.0.0",
                "core_genre": "Simulation",
                "entrypoint": "mechanics.py",
            },
            entities_code=(
                'class ReactorCore:\n'
                '    deliberation: str = "Core power reactor."\n'
                '    def __init__(self, power: float = 100.0):\n'
                '        self.power = power\n'
            ),
            abilities_code=(
                'class VentCoolant:\n'
                '    deliberation: str = "Coolant vent action."\n'
                '    def execute(self, core):\n'
                '        core.power -= 10.0\n'
            ),
            scenes_code=(
                'from engine.api.contracts import BaseScene\n'
                'from entities import ReactorCore\n'
                'from abilities import VentCoolant\n'
                'class StationScene(BaseScene):\n'
                '    deliberation: str = "Main station control scene."\n'
                '    def __init__(self, engine=None):\n'
                '        super().__init__(engine)\n'
                '        self.core = ReactorCore()\n'
                '        self.action = VentCoolant()\n'
                '        self.state_counter = 0\n'
                '    def update(self, dt: float):\n'
                '        self.state_counter += 1\n'
                '    def render(self, surface):\n'
                '        pass\n'
            ),
            mechanics_code=(
                'from engine.api.contracts import BaseCartridge\n'
                'from scenes import StationScene\n'
                'class StationCartridge(BaseCartridge):\n'
                '    deliberation: str = "Cartridge bootstrap entrypoint."\n'
                '    def get_scenes(self, engine):\n'
                '        return {"main": StationScene(engine)}\n'
                '    def get_initial_scene(self, engine):\n'
                '        return self.get_scenes(engine)["main"]\n'
            ),
            level_data={
                "deliberation": "Level layout for Sector Alpha",
                "width": 1280,
                "height": 720,
                "modules": ["reactor", "hydroponics"],
            },
        )

        results = bundle.assemble_cartridge_directory(dest_dir)

        # Assert all 6 expected files were generated
        expected_files = [
            "manifest.json",
            "entities.py",
            "abilities.py",
            "scenes.py",
            "mechanics.py",
            "level.json",
        ]
        for fname in expected_files:
            assert fname in results, f"Missing {fname} in results mapping"
            fpath = dest_dir / fname
            assert fpath.is_file(), f"File {fpath} was not written to disk"
            assert fpath.stat().st_size > 0, f"File {fpath} is unexpectedly empty"

        # Check JSON validity
        manifest = json.loads((dest_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["name"] == "Orbital Station Sim"
        level = json.loads((dest_dir / "level.json").read_text(encoding="utf-8"))
        assert level["width"] == 1280

    def test_assemble_cartridge_directory_auto_manifest_generation(self, tmp_path: Path) -> None:
        """When manifest_data is empty, assemble_cartridge_directory uses swe_agent._generate_manifest_data."""
        dest_dir = tmp_path / "cartridge_no_manifest"
        swe = LeadGameplaySWEAgent()

        bundle = OpenGenreCartridgeBundle(
            deliberation="Bundle without manifest data.",
            manifest_data={},
            mechanics_code=(
                'from engine.api.contracts import BaseCartridge, BaseScene\n'
                'class SceneA(BaseScene):\n'
                '    deliberation: str = "Scene."\n'
                '    def update(self, dt):\n'
                '        pass\n'
                '    def render(self, surface):\n'
                '        pass\n'
                'class CartridgeA(BaseCartridge):\n'
                '    deliberation: str = "Cartridge."\n'
                '    def get_scenes(self, engine):\n'
                '        return {"main": SceneA(engine)}\n'
                '    def get_initial_scene(self, engine):\n'
                '        return self.get_scenes(engine)["main"]\n'
            ),
        )

        results = bundle.assemble_cartridge_directory(dest_dir, swe_agent=swe)
        assert (dest_dir / "manifest.json").is_file()
        manifest = json.loads((dest_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "deliberation" in manifest
        assert manifest["version"] == "1.0.0"

    def test_assemble_cartridge_directory_nested_nonexistent_path(self, tmp_path: Path) -> None:
        """Verifies deeply nested paths are created with mkdir(parents=True)."""
        deep_dest = tmp_path / "deep" / "nested" / "candidate" / "cartridge"

        bundle = OpenGenreCartridgeBundle(
            deliberation="Deep nested path assembly.",
            mechanics_code=(
                'from engine.api.contracts import BaseCartridge, BaseScene\n'
                'class MinimalScene(BaseScene):\n'
                '    deliberation: str = "Minimal."\n'
                '    def update(self, dt): pass\n'
                '    def render(self, s): pass\n'
                'class MinimalCartridge(BaseCartridge):\n'
                '    deliberation: str = "Minimal."\n'
                '    def get_scenes(self, engine): return {"m": MinimalScene(engine)}\n'
                '    def get_initial_scene(self, engine): return self.get_scenes(engine)["m"]\n'
            ),
        )

        results = bundle.assemble_cartridge_directory(deep_dest)
        assert deep_dest.is_dir()
        assert (deep_dest / "mechanics.py").is_file()

    def test_assemble_cartridge_directory_idempotent_overwrite(self, tmp_path: Path) -> None:
        """Verifies that re-assembling into the same directory cleanly overwrites without file truncation."""
        dest_dir = tmp_path / "overwrite_test"

        bundle_v1 = OpenGenreCartridgeBundle(
            deliberation="V1 bundle.",
            mechanics_code=(
                'from engine.api.contracts import BaseCartridge, BaseScene\n'
                'class S1(BaseScene):\n'
                '    deliberation: str = "S1."\n'
                '    def update(self, dt): pass\n'
                '    def render(self, s): pass\n'
                'class C1(BaseCartridge):\n'
                '    deliberation: str = "C1."\n'
                '    def get_scenes(self, engine): return {"s": S1(engine)}\n'
                '    def get_initial_scene(self, engine): return self.get_scenes(engine)["s"]\n'
            ),
        )
        bundle_v1.assemble_cartridge_directory(dest_dir)
        v1_content = (dest_dir / "mechanics.py").read_text(encoding="utf-8")
        assert "class C1" in v1_content

        bundle_v2 = OpenGenreCartridgeBundle(
            deliberation="V2 bundle with revised mechanics.",
            mechanics_code=(
                'from engine.api.contracts import BaseCartridge, BaseScene\n'
                'class S2(BaseScene):\n'
                '    deliberation: str = "S2."\n'
                '    def update(self, dt): pass\n'
                '    def render(self, s): pass\n'
                'class C2(BaseCartridge):\n'
                '    deliberation: str = "C2 updated."\n'
                '    def get_scenes(self, engine): return {"s": S2(engine)}\n'
                '    def get_initial_scene(self, engine): return self.get_scenes(engine)["s"]\n'
            ),
        )
        bundle_v2.assemble_cartridge_directory(dest_dir)
        v2_content = (dest_dir / "mechanics.py").read_text(encoding="utf-8")
        assert "class C2" in v2_content
        assert "class C1" not in v2_content

    def test_assemble_cartridge_directory_triggers_cross_module_check(self, tmp_path: Path) -> None:
        """When swe_agent is provided and cross-module import is broken, assembly raises CodeVerificationError."""
        dest_dir = tmp_path / "broken_cross_import"
        swe = LeadGameplaySWEAgent()

        bundle = OpenGenreCartridgeBundle(
            deliberation="Broken cross-module import bundle.",
            entities_code=(
                'class ExistingEntity:\n'
                '    deliberation: str = "Valid entity."\n'
            ),
            scenes_code=(
                'from engine.api.contracts import BaseScene\n'
                'from entities import NonExistentEntity\n'
                'class BrokenScene(BaseScene):\n'
                '    deliberation: str = "Scene with broken import."\n'
                '    def update(self, dt): pass\n'
                '    def render(self, s): pass\n'
            ),
            mechanics_code=(
                'from engine.api.contracts import BaseCartridge\n'
                'from scenes import BrokenScene\n'
                'class BrokenCartridge(BaseCartridge):\n'
                '    deliberation: str = "Cartridge."\n'
                '    def get_scenes(self, engine): return {"m": BrokenScene(engine)}\n'
                '    def get_initial_scene(self, engine): return self.get_scenes(engine)["m"]\n'
            ),
        )

        with pytest.raises(CodeVerificationError) as exc_info:
            bundle.assemble_cartridge_directory(dest_dir, swe_agent=swe)

        assert "NonExistentEntity" in str(exc_info.value)
        assert "Cross-module import defect" in str(exc_info.value)


# ============================================================================
# 2. Cross-Module Import Verification Stress Tests
# ============================================================================

class TestCrossModuleImportVerificationStress:
    """Stress tests for LeadGameplaySWEAgent.verify_cartridge_cross_module_imports."""

    def test_cross_module_imports_valid_diamond_dependency(self, tmp_path: Path) -> None:
        """Diamond dependency graph: mechanics -> scenes, abilities; scenes -> entities; abilities -> entities."""
        cart_dir = tmp_path / "diamond_graph"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            'class BaseUnit:\n'
            '    deliberation: str = "Base unit."\n'
            'UNIT_CAP: int = 100\n'
            'active_units: list = []\n',
            encoding="utf-8",
        )
        (cart_dir / "abilities.py").write_text(
            'from entities import BaseUnit, UNIT_CAP\n'
            'class MoveAction:\n'
            '    deliberation: str = "Move unit."\n',
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            'from entities import BaseUnit, active_units\n'
            'from abilities import MoveAction\n'
            'class MainScene:\n'
            '    deliberation: str = "Main scene."\n',
            encoding="utf-8",
        )
        (cart_dir / "mechanics.py").write_text(
            'from scenes import MainScene\n'
            'from abilities import MoveAction\n'
            'class GameCartridge:\n'
            '    deliberation: str = "Cartridge."\n',
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is True
        assert errors == []

    def test_cross_module_imports_all_export_node_types(self, tmp_path: Path) -> None:
        """Verifies export table extraction handles classes, functions, assignments, and typed annotations."""
        cart_dir = tmp_path / "export_types"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            'class HeroClass:\n    deliberation: str = "Hero."\n\n'
            'def spawn_helper():\n    return 42\n\n'
            'CONFIG_VAR = "ACTIVE"\n'
            'TYPED_INT: int = 100\n'
            'TYPED_LIST: list[str] = ["a", "b"]\n',
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            'from entities import HeroClass, spawn_helper, CONFIG_VAR, TYPED_INT, TYPED_LIST\n'
            'class Scene:\n    deliberation: str = "Scene."\n',
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is True
        assert len(errors) == 0

    def test_cross_module_imports_detects_multiple_missing_symbols(self, tmp_path: Path) -> None:
        """A single ImportFrom node with multiple missing symbols surfaces all missing symbols."""
        cart_dir = tmp_path / "multi_missing"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            'class RealEntity:\n    deliberation: str = "Real."\n',
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            'from entities import RealEntity, FakeOne, FakeTwo, FakeThree\n'
            'class Scene:\n    deliberation: str = "Scene."\n',
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 3
        assert any("FakeOne" in e for e in errors)
        assert any("FakeTwo" in e for e in errors)
        assert any("FakeThree" in e for e in errors)

    def test_cross_module_imports_missing_module_file(self, tmp_path: Path) -> None:
        """Importing a core module when its .py file does not exist fails with clear defect notice."""
        cart_dir = tmp_path / "missing_file"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        # mechanics.py imports entities, but entities.py does NOT exist
        (cart_dir / "mechanics.py").write_text(
            'import entities\n'
            'class Cartridge:\n    deliberation: str = "Cartridge."\n',
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 1
        assert "imports 'entities', but 'entities.py' does not exist" in errors[0]

    def test_cross_module_imports_syntax_error_caught_cleanly(self, tmp_path: Path) -> None:
        """A file with broken Python syntax is flagged as a SyntaxError in cross-module verification."""
        cart_dir = tmp_path / "syntax_err"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            'def incomplete_func(\n',
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            'class Scene:\n    deliberation: str = "Scene."\n',
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert any("SyntaxError in entities.py" in e for e in errors)

    def test_cross_module_imports_wildcard_import_allowed(self, tmp_path: Path) -> None:
        """Wildcard import 'from entities import *' does not trigger false positive symbol errors."""
        cart_dir = tmp_path / "wildcard"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            'class WildEntity:\n    deliberation: str = "Wild."\n',
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            'from entities import *\n'
            'class WildScene:\n    deliberation: str = "Wild scene."\n',
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is True
        assert len(errors) == 0

    def test_cross_module_imports_external_contracts_unaffected(self, tmp_path: Path) -> None:
        """External package imports (engine.api.contracts, pygame, math, typing) are not flagged."""
        cart_dir = tmp_path / "external_imports"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "mechanics.py").write_text(
            'import math\n'
            'import random\n'
            'from typing import Any, Dict, List, Optional\n'
            'from engine.api.contracts import BaseCartridge, BaseScene\n'
            'class ExtCartridge(BaseCartridge):\n'
            '    deliberation: str = "Cartridge."\n'
            '    def get_scenes(self, engine): return {}\n'
            '    def get_initial_scene(self, engine): return None\n',
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert len(errors) == 0

    def test_cross_module_imports_empty_directory(self, tmp_path: Path) -> None:
        """Empty cartridge directory passes trivially with zero errors."""
        cart_dir = tmp_path / "empty_dir"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_cross_module_imports_100_random_symbols_stress(self, tmp_path: Path) -> None:
        """Stress tests 100 randomized exported and imported symbols to verify scalability."""
        cart_dir = tmp_path / "stress_100"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        valid_symbols = [f"Entity_{i:03d}_{random.choice(string.ascii_lowercase)}" for i in range(100)]
        fake_symbols = [f"Fake_{i:03d}" for i in range(25)]

        entities_lines = []
        for sym in valid_symbols:
            entities_lines.append(f"class {sym}:\n    deliberation: str = '{sym} entity.'\n")
        (cart_dir / "entities.py").write_text("\n".join(entities_lines), encoding="utf-8")

        # scenes imports all 100 valid symbols + 25 fake symbols
        all_imports = valid_symbols + fake_symbols
        random.shuffle(all_imports)
        scenes_code = (
            f"from entities import {', '.join(all_imports)}\n"
            f"class StressScene:\n    deliberation: str = 'Stress scene.'\n"
        )
        (cart_dir / "scenes.py").write_text(scenes_code, encoding="utf-8")

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 25
        for fake in fake_symbols:
            assert any(fake in e for e in errors), f"Expected defect for {fake} was not detected"


# ============================================================================
# 3. Sealed Asset Discovery Stress Tests
# ============================================================================

class TestSealedAssetDiscoveryAdversarial:
    """Stress tests for LeadGameplaySWEAgent._discover_sealed_assets."""

    def test_discover_sealed_assets_arbitrary_genre_names(self, tmp_path: Path) -> None:
        """Discovers open-genre assets with arbitrary names (automation sim, puzzle, ambient sound)."""
        staging_dir = tmp_path / "staging_open_genre"
        sprites_dir = staging_dir / "assets" / "sprites"
        audio_dir = staging_dir / "assets" / "audio"
        sprites_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        arbitrary_sprites = [
            "conveyor_belt_straight.png",
            "smelter_furnace_tier2.png",
            "circuit_assembler.png",
            "puzzle_node_resonator.png",
            "ui_energy_meter.png",
        ]
        arbitrary_audio = [
            "furnace_hum_loop.wav",
            "conveyor_click.wav",
            "narrator_intro_briefing.wav",
        ]

        for s in arbitrary_sprites:
            (sprites_dir / s).write_bytes(b"\x89PNG\r\n\x1a\nfake")
        for a in arbitrary_audio:
            (audio_dir / a).write_bytes(b"RIFFfake")

        swe = LeadGameplaySWEAgent()
        assets = swe._discover_sealed_assets(staging_dir)

        assert len(assets["sprites"]) == len(arbitrary_sprites)
        assert len(assets["audio"]) == len(arbitrary_audio)

        for s in arbitrary_sprites:
            assert f"assets/sprites/{s}" in assets["sprites"]
        for a in arbitrary_audio:
            assert f"assets/audio/{a}" in assets["audio"]

    def test_discover_sealed_assets_missing_asset_directories(self, tmp_path: Path) -> None:
        """Handles staging directory with completely absent assets/ subfolder."""
        staging_dir = tmp_path / "staging_empty"
        staging_dir.mkdir(parents=True, exist_ok=True)

        swe = LeadGameplaySWEAgent()
        assets = swe._discover_sealed_assets(staging_dir)

        assert assets == {"sprites": [], "portraits": [], "mockups": [], "audio": []}

    def test_discover_sealed_assets_ignores_non_standard_extensions(self, tmp_path: Path) -> None:
        """Ignores non-.png in sprites/ and non-.wav in audio/."""
        staging_dir = tmp_path / "staging_mixed_ext"
        sprites_dir = staging_dir / "assets" / "sprites"
        audio_dir = staging_dir / "assets" / "audio"
        sprites_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Valid assets
        (sprites_dir / "valid_sprite.png").write_bytes(b"fake_png")
        (audio_dir / "valid_clip.wav").write_bytes(b"fake_wav")

        # Extraneous files that MUST be ignored
        (sprites_dir / "junk.jpg").write_text("not a png")
        (sprites_dir / "metadata.json").write_text("{}")
        (sprites_dir / ".DS_Store").write_bytes(b"junk")
        (sprites_dir / "sprite_backup.png.bak").write_text("backup")
        (audio_dir / "sound.mp3").write_text("mp3")
        (audio_dir / "sound.ogg").write_text("ogg")
        (audio_dir / "audio_manifest.txt").write_text("notes")

        swe = LeadGameplaySWEAgent()
        assets = swe._discover_sealed_assets(staging_dir)

        assert assets["sprites"] == ["assets/sprites/valid_sprite.png"]
        assert assets["audio"] == ["assets/audio/valid_clip.wav"]

    def test_discover_sealed_assets_deterministic_sorting(self, tmp_path: Path) -> None:
        """Assets are returned deterministically sorted regardless of filesystem order."""
        staging_dir = tmp_path / "staging_sort"
        sprites_dir = staging_dir / "assets" / "sprites"
        sprites_dir.mkdir(parents=True, exist_ok=True)

        names = [f"sprite_{i:03d}.png" for i in range(50)]
        shuffled = list(names)
        random.shuffle(shuffled)

        for name in shuffled:
            (sprites_dir / name).write_bytes(b"png")

        swe = LeadGameplaySWEAgent()
        assets = swe._discover_sealed_assets(staging_dir)

        expected = [f"assets/sprites/{n}" for n in sorted(names)]
        assert assets["sprites"] == expected

    def test_discover_sealed_assets_high_volume_stress(self, tmp_path: Path) -> None:
        """Discovers 200+ sprites and 200+ audio clips rapidly and accurately."""
        staging_dir = tmp_path / "staging_high_volume"
        sprites_dir = staging_dir / "assets" / "sprites"
        audio_dir = staging_dir / "assets" / "audio"
        sprites_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        for i in range(250):
            (sprites_dir / f"tile_{i:04d}.png").write_bytes(b"png")
            (audio_dir / f"dialogue_{i:04d}.wav").write_bytes(b"wav")

        swe = LeadGameplaySWEAgent()
        assets = swe._discover_sealed_assets(staging_dir)

        assert len(assets["sprites"]) == 250
        assert len(assets["audio"]) == 250
        assert assets["sprites"][0] == "assets/sprites/tile_0000.png"
        assert assets["sprites"][-1] == "assets/sprites/tile_0249.png"


# ============================================================================
# 4. Generalizability & Zero Arena Assumptions in swe.py
# ============================================================================

class TestGeneralizabilityAndZeroArenaAssumptions:
    """Audit of engine/studio/swe.py ensuring zero hardcoded arena assumptions."""

    def test_swe_source_code_has_zero_legacy_arena_tokens(self) -> None:
        """Grep audit of engine/studio/swe.py for banned legacy tokens."""
        swe_path = Path("engine/studio/swe.py")
        assert swe_path.is_file()
        source = swe_path.read_text(encoding="utf-8")

        banned_tokens = [
            "chaser",
            "sniper",
            "spikes",
            "acid",
            "turret",
            "pylon",
            "arena",
        ]

        found_violations = []
        for line_no, line in enumerate(source.splitlines(), start=1):
            line_lower = line.lower()
            for token in banned_tokens:
                if token in line_lower:
                    found_violations.append(f"Line {line_no}: found '{token}' in '{line.strip()}'")

        assert not found_violations, (
            f"Found {len(found_violations)} legacy token violations in engine/studio/swe.py:\n"
            + "\n".join(found_violations[:10])
        )

    def test_open_genre_cartridge_bundle_rationale_first_epistemic_ordering(self) -> None:
        """Verifies OpenGenreCartridgeBundle adheres to Rule 4 (key index 0 == 'deliberation')."""
        bundle = OpenGenreCartridgeBundle(
            deliberation="Epistemic rationale explaining the architecture.",
            mechanics_code='deliberation: str = "Mechanics"\n',
        )

        # 1. model_dump dict key order
        dump_keys = list(bundle.model_dump().keys())
        assert dump_keys[0] == "deliberation", f"Expected 'deliberation' at index 0, got '{dump_keys[0]}'"

        # 2. model_dump_json JSON key order
        json_keys = list(json.loads(bundle.model_dump_json()).keys())
        assert json_keys[0] == "deliberation", f"Expected 'deliberation' at index 0, got '{json_keys[0]}'"

        # 3. schema properties order
        schema = bundle.model_json_schema()
        assert "properties" in schema
        schema_props = list(schema["properties"].keys())
        assert schema_props[0] == "deliberation"

    def test_swe_verify_code_module_open_genre_contracts(self) -> None:
        """Verifies verify_code_module accepts open-genre BaseScene lifecycle without legacy get_level_spec."""
        swe = LeadGameplaySWEAgent()

        valid_mechanics = (
            'from engine.api.contracts import BaseCartridge, BaseScene\n'
            'class SimScene(BaseScene):\n'
            '    deliberation: str = "Sim scene."\n'
            '    def update(self, dt: float):\n'
            '        pass\n'
            '    def render(self, surface):\n'
            '        pass\n'
            'class FactoryCartridge(BaseCartridge):\n'
            '    deliberation: str = "Pure factory automation simulation cartridge."\n'
            '    def get_scenes(self, engine):\n'
            '        return {"sim": SimScene(engine)}\n'
            '    def get_initial_scene(self, engine):\n'
            '        return self.get_scenes(engine)["sim"]\n'
        )

        valid, err = swe.verify_code_module("mechanics.py", valid_mechanics)
        assert valid is True, f"Valid open-genre mechanics was rejected: {err}"

    def test_swe_verify_code_module_rejects_missing_deliberation_first(self) -> None:
        """Verifies verify_code_module rejects classes where deliberation is missing or not first."""
        swe = LeadGameplaySWEAgent()

        # Class where docstring comes before deliberation (violates Rule 4)
        broken_class = (
            'from engine.api.contracts import BaseCartridge\n'
            'class DocstringFirstCartridge(BaseCartridge):\n'
            '    """This docstring violates deliberation-first ordering."""\n'
            '    deliberation: str = "Deliberation after docstring."\n'
            '    def get_scenes(self, engine): return {}\n'
            '    def get_initial_scene(self, engine): return None\n'
        )

        valid, err = swe.verify_code_module("mechanics.py", broken_class)
        assert valid is False
        assert "must declare 'deliberation' as its first attribute" in err

    def test_swe_verify_code_module_rejects_ast_security_violations(self) -> None:
        """Verifies verify_code_module catches banned modules and builtins via AST validator."""
        swe = LeadGameplaySWEAgent()

        # Imports banned 'os'
        os_import_code = (
            'import os\n'
            'from engine.api.contracts import BaseCartridge\n'
            'class BadCartridge(BaseCartridge):\n'
            '    deliberation: str = "Bad."\n'
            '    def get_scenes(self, engine): return {}\n'
            '    def get_initial_scene(self, engine): return None\n'
        )
        valid, err = swe.verify_code_module("mechanics.py", os_import_code, raise_on_ast_error=False)
        assert valid is False
        assert "ASTSecurityError" in err

        # Calls prohibited builtin 'eval'
        eval_call_code = (
            'from engine.api.contracts import BaseCartridge\n'
            'class EvalCartridge(BaseCartridge):\n'
            '    deliberation: str = "Eval."\n'
            '    def get_scenes(self, engine):\n'
            '        eval("1 + 1")\n'
            '        return {}\n'
            '    def get_initial_scene(self, engine): return None\n'
        )
        valid, err = swe.verify_code_module("mechanics.py", eval_call_code, raise_on_ast_error=False)
        assert valid is False
        assert "ASTSecurityError" in err
