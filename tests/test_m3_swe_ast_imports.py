"""tests/test_m3_swe_ast_imports.py
Comprehensive test suite verifying the 4 cross-module AST import remediations
in LeadGameplaySWEAgent.verify_cartridge_cross_module_imports:

1. Defect 1: Missing module in `ast.ImportFrom` for core modules
   ("entities", "scenes", "abilities", "mechanics").
2. Defect 2: ast.AsyncFunctionDef support in module export extraction.
3. Defect 3: ast.Tuple, ast.List, and ast.Starred unpacking in ast.Assign export extraction.
4. Defect 4: Adjacency graph construction and circular import cycle detection across local modules.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from engine.studio.swe import (
    CodeVerificationError,
    LeadGameplaySWEAgent,
    OpenGenreCartridgeBundle,
)


# ============================================================================
# 1. Defect 1: Missing Module in ImportFrom
# ============================================================================

class TestMissingModuleImportFrom:
    """Verifies that `from <module> import <sym>` flags missing .py files for core modules."""

    @pytest.mark.parametrize("mod", ["entities", "scenes", "abilities", "mechanics"])
    def test_import_from_missing_core_module_detected(self, tmp_path: Path, mod: str) -> None:
        """When a core module file does not exist, ImportFrom records a descriptive missing module defect."""
        cart_dir = tmp_path / f"missing_{mod}"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        # Target file imports from missing module
        (cart_dir / "importer.py").write_text(
            f"from {mod} import SomeSymbol\n"
            f"class ImporterClass:\n    deliberation: str = 'Importer'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 1
        expected_msg = f"Module importer.py imports from '{mod}', but '{mod}.py' does not exist."
        assert errors[0] == expected_msg

    def test_import_from_missing_module_raises_code_verification_error(self, tmp_path: Path) -> None:
        """When raise_on_error=True, missing ImportFrom module raises CodeVerificationError."""
        cart_dir = tmp_path / "missing_raise"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "mechanics.py").write_text(
            "from entities import Player\n"
            "class GameCartridge:\n    deliberation: str = 'Cartridge'\n",
            encoding="utf-8",
        )

        with pytest.raises(CodeVerificationError) as exc_info:
            swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert "imports from 'entities', but 'entities.py' does not exist." in str(exc_info.value)

    def test_import_from_existing_module_missing_symbol_distinguished(self, tmp_path: Path) -> None:
        """When the module exists but the requested symbol does not, a symbol defect is produced, not a module missing error."""
        cart_dir = tmp_path / "sym_missing"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "class RealEntity:\n    deliberation: str = 'Real'\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import MissingEntity\n"
            "class MainScene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 1
        assert "symbol 'MissingEntity' is not exported by 'entities.py'" in errors[0]
        assert "does not exist" not in errors[0]


# ============================================================================
# 2. Defect 2: ast.AsyncFunctionDef Support in Exports
# ============================================================================

class TestAsyncFunctionDefExports:
    """Verifies that asynchronous functions (async def) are registered as valid exports."""

    def test_async_function_exported_and_imported_successfully(self, tmp_path: Path) -> None:
        """Async functions in abilities.py are recognized in the export table and imported cleanly."""
        cart_dir = tmp_path / "async_exports"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "abilities.py").write_text(
            "async def perform_dash(target, distance):\n"
            "    pass\n\n"
            "async def cast_spell(caster, spell_id):\n"
            "    pass\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from abilities import perform_dash, cast_spell\n"
            "class BattleScene:\n    deliberation: str = 'Battle'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is True
        assert errors == []

    def test_mixed_sync_and_async_functions_exported(self, tmp_path: Path) -> None:
        """Modules mixing classes, sync functions, and async functions export all symbols cleanly."""
        cart_dir = tmp_path / "mixed_exports"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "abilities.py").write_text(
            "class AbilityRegistry:\n    deliberation: str = 'Registry'\n\n"
            "def calculate_cooldown(base: float) -> float:\n    return base * 0.9\n\n"
            "async def trigger_ultimate(hero) -> None:\n    pass\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from abilities import AbilityRegistry, calculate_cooldown, trigger_ultimate\n"
            "class ArenaScene:\n    deliberation: str = 'Arena'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_missing_async_function_symbol_still_flagged(self, tmp_path: Path) -> None:
        """Importing a non-existent function from an async module still fails verification."""
        cart_dir = tmp_path / "async_missing_sym"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "abilities.py").write_text(
            "async def real_async_ability():\n    pass\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from abilities import real_async_ability, non_existent_async\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert any("non_existent_async" in e for e in errors)


# ============================================================================
# 3. Defect 3: Tuple & List Unpacking in Assignments
# ============================================================================

class TestTupleAndListUnpackedExports:
    """Verifies that unpacked assignment targets are extracted and registered as exports."""

    def test_tuple_unpacked_assignment_exported_and_imported(self, tmp_path: Path) -> None:
        """Constants unpacked via tuple assignment are present in exports."""
        cart_dir = tmp_path / "tuple_unpack"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600\n"
            "MIN_BOUND, MAX_BOUND = 0.0, 100.0\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import SCREEN_WIDTH, SCREEN_HEIGHT, MIN_BOUND, MAX_BOUND\n"
            "class GameScene:\n    deliberation: str = 'Game scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is True
        assert errors == []

    def test_list_unpacked_assignment_exported_and_imported(self, tmp_path: Path) -> None:
        """Constants unpacked via list assignment are present in exports."""
        cart_dir = tmp_path / "list_unpack"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "[GRID_X, GRID_Y] = [32, 32]\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import GRID_X, GRID_Y\n"
            "class GridScene:\n    deliberation: str = 'Grid scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is True
        assert errors == []

    def test_nested_and_starred_unpacking_exported(self, tmp_path: Path) -> None:
        """Nested tuple unpacking and starred targets are extracted recursively."""
        cart_dir = tmp_path / "nested_starred_unpack"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "((TOP_LEFT_X, TOP_LEFT_Y), (BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y)) = ((0, 0), (800, 600))\n"
            "DEFAULT_THEME, *ALT_THEMES = ('dark', 'light', 'neon')\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import TOP_LEFT_X, TOP_LEFT_Y, BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y, DEFAULT_THEME, ALT_THEMES\n"
            "class ComplexScene:\n    deliberation: str = 'Complex scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is True
        assert errors == []

    def test_attribute_and_subscript_assignments_not_falsely_exported(self, tmp_path: Path) -> None:
        """Assigning to attributes or subscripts (e.g. obj.attr = 1, arr[0] = 2) does not export object names."""
        cart_dir = tmp_path / "non_name_targets"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "class Dummy:\n    pass\n"
            "d = Dummy()\n"
            "d.fake_symbol = 42\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import fake_symbol\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert any("fake_symbol" in e for e in errors)


# ============================================================================
# 4. Defect 4: Circular Import Cycle Detection
# ============================================================================

class TestCircularImportCycleDetection:
    """Verifies that circular dependencies between local cartridge modules are statically flagged."""

    def test_two_module_circular_import_cycle_detected(self, tmp_path: Path) -> None:
        """Direct 2-module cycle (entities <-> scenes) is detected and reported."""
        cart_dir = tmp_path / "cycle_2"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "from scenes import MainScene\n"
            "class Player:\n    deliberation: str = 'Player'\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import Player\n"
            "class MainScene:\n    deliberation: str = 'MainScene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 1
        assert "Circular import detected between local modules: entities -> scenes -> entities" in errors[0]

    def test_three_module_circular_import_cycle_detected(self, tmp_path: Path) -> None:
        """3-module cycle (abilities -> scenes -> mechanics -> abilities) is detected and reported."""
        cart_dir = tmp_path / "cycle_3"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "abilities.py").write_text(
            "from scenes import LevelScene\n"
            "class Dash:\n    deliberation: str = 'Dash'\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from mechanics import GameLoop\n"
            "class LevelScene:\n    deliberation: str = 'LevelScene'\n",
            encoding="utf-8",
        )
        (cart_dir / "mechanics.py").write_text(
            "from abilities import Dash\n"
            "class GameLoop:\n    deliberation: str = 'GameLoop'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 1
        assert "Circular import detected between local modules: abilities -> scenes -> mechanics -> abilities" in errors[0]

    def test_self_import_cycle_detected(self, tmp_path: Path) -> None:
        """A module importing itself is detected as a circular dependency."""
        cart_dir = tmp_path / "self_import"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "import entities\n"
            "class SelfRef:\n    deliberation: str = 'Self'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert any("Circular import detected between local modules: entities -> entities" in e for e in errors)

    def test_non_cycle_diamond_dependency_passes(self, tmp_path: Path) -> None:
        """Diamond DAG (mechanics -> scenes, abilities; scenes -> entities; abilities -> entities) passes with zero cycles."""
        cart_dir = tmp_path / "diamond_dag"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "class Entity:\n    deliberation: str = 'Entity'\n",
            encoding="utf-8",
        )
        (cart_dir / "abilities.py").write_text(
            "from entities import Entity\n"
            "class Ability:\n    deliberation: str = 'Ability'\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import Entity\n"
            "from abilities import Ability\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )
        (cart_dir / "mechanics.py").write_text(
            "from scenes import Scene\n"
            "from abilities import Ability\n"
            "class Cartridge:\n    deliberation: str = 'Cartridge'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_circular_import_raises_code_verification_error(self, tmp_path: Path) -> None:
        """When raise_on_error=True, circular dependency raises CodeVerificationError."""
        cart_dir = tmp_path / "cycle_raise"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "from scenes import Scene\n"
            "class Entity:\n    deliberation: str = 'Entity'\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import Entity\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        with pytest.raises(CodeVerificationError) as exc_info:
            swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert "Circular import detected between local modules" in str(exc_info.value)

    def test_external_imports_do_not_create_false_cycles(self, tmp_path: Path) -> None:
        """External package imports (math, pygame, engine.api.contracts) never participate in local cycle detection."""
        cart_dir = tmp_path / "external_dag"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "import math\n"
            "from typing import List\n"
            "class Vector:\n    deliberation: str = 'Vector'\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "import math\n"
            "from typing import Dict\n"
            "from entities import Vector\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_assemble_cartridge_directory_fails_on_circular_import(self, tmp_path: Path) -> None:
        """OpenGenreCartridgeBundle.assemble_cartridge_directory triggers verify_cartridge_cross_module_imports and fails on cycle."""
        cart_dir = tmp_path / "bundle_cycle"
        swe = LeadGameplaySWEAgent()

        bundle = OpenGenreCartridgeBundle(
            deliberation="Testing bundle assembly cycle rejection.",
            mechanics_code=(
                "from engine.api.contracts import BaseCartridge\n"
                "from scenes import BattleScene\n"
                "class Game(BaseCartridge):\n"
                "    deliberation: str = 'Game cartridge'\n"
                "    def get_scenes(self, engine): return {'battle': BattleScene}\n"
                "    def get_initial_scene(self, engine): return 'battle'\n"
            ),
            scenes_code=(
                "from engine.api.contracts import BaseScene\n"
                "from mechanics import Game\n"
                "class BattleScene(BaseScene):\n"
                "    deliberation: str = 'Battle scene'\n"
                "    def update(self, dt): pass\n"
                "    def render(self, screen): pass\n"
            ),
        )

        with pytest.raises(CodeVerificationError) as exc_info:
            bundle.assemble_cartridge_directory(cart_dir, swe_agent=swe)
        assert "Circular import detected between local modules: mechanics -> scenes -> mechanics" in str(exc_info.value)
