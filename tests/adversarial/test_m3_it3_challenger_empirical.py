"""tests/adversarial/test_m3_it3_challenger_empirical.py
Milestone 3 Iteration 3 Challenger 1 Empirical Stress Test Suite.

Adversarially stress-tests the 4 AST import remediations implemented in
engine/studio/swe.py:324-437:
1. Missing core module detection in ImportFrom.
2. ast.AsyncFunctionDef export support.
3. Tuple, list, and starred unpacked assignment export support.
4. Directed cycle detection across local modules (2-node, 3-node, 4-node, self-loops,
   figure-8 double cycles, disconnected subgraphs, diamond and multi-path DAGs).
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List
import pytest

from engine.studio.swe import (
    CodeVerificationError,
    LeadGameplaySWEAgent,
    OpenGenreCartridgeBundle,
)


# ============================================================================
# 1. Defect 1: Missing Core Module in ImportFrom
# ============================================================================

class TestMissingCoreModuleImportFromAdversarial:
    """Stress tests verifying missing core module detection in ImportFrom."""

    @pytest.mark.parametrize("mod", ["entities", "scenes", "abilities", "mechanics"])
    def test_missing_core_module_standard_import_from(self, tmp_path: Path, mod: str) -> None:
        """When any of the 4 core modules is absent, ImportFrom flags the missing file."""
        cart_dir = tmp_path / f"test_missing_{mod}"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "consumer.py").write_text(
            f"from {mod} import TargetSymbol\n"
            f"class ConsumerClass:\n"
            f"    deliberation: str = 'Consumer'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 1
        assert errors[0] == f"Module consumer.py imports from '{mod}', but '{mod}.py' does not exist."

    @pytest.mark.parametrize("mod", ["entities", "scenes", "abilities", "mechanics"])
    def test_missing_core_module_relative_import_from(self, tmp_path: Path, mod: str) -> None:
        """Relative import 'from .<mod> import <sym>' also flags missing core module."""
        cart_dir = tmp_path / f"test_missing_rel_{mod}"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "consumer.py").write_text(
            f"from .{mod} import TargetSymbol\n"
            f"class ConsumerClass:\n"
            f"    deliberation: str = 'Consumer'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert any(f"imports from '{mod}', but '{mod}.py' does not exist" in e for e in errors)

    def test_multiple_missing_core_modules_in_single_file(self, tmp_path: Path) -> None:
        """Single file importing multiple missing core modules reports each missing module."""
        cart_dir = tmp_path / "multi_missing_core"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "mechanics.py").write_text(
            "from entities import Hero\n"
            "from scenes import GameScene\n"
            "from abilities import DashAbility\n"
            "class MechanicsGame:\n"
            "    deliberation: str = 'Game'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 3
        assert any("imports from 'entities', but 'entities.py' does not exist" in e for e in errors)
        assert any("imports from 'scenes', but 'scenes.py' does not exist" in e for e in errors)
        assert any("imports from 'abilities', but 'abilities.py' does not exist" in e for e in errors)

    def test_missing_core_module_with_multiline_import_and_aliases(self, tmp_path: Path) -> None:
        """Importing multiple symbols and aliases from a missing module records exactly 1 defect per statement."""
        cart_dir = tmp_path / "multiline_missing"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "scenes.py").write_text(
            "from entities import (\n"
            "    Hero as H,\n"
            "    Villain as V,\n"
            "    NPC,\n"
            ")\n"
            "class Scene:\n"
            "    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 1
        assert "imports from 'entities', but 'entities.py' does not exist" in errors[0]

    def test_missing_core_module_raises_when_requested(self, tmp_path: Path) -> None:
        """When raise_on_error=True, CodeVerificationError is raised with missing module details."""
        cart_dir = tmp_path / "raise_missing"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "scenes.py").write_text(
            "from abilities import CastSpell\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        with pytest.raises(CodeVerificationError) as exc_info:
            swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert "imports from 'abilities', but 'abilities.py' does not exist" in str(exc_info.value)

    def test_external_standard_and_engine_packages_never_flagged_missing(self, tmp_path: Path) -> None:
        """Importing external modules (math, pygame, engine.api.contracts, typing, json) never flags missing file."""
        cart_dir = tmp_path / "external_packages"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "import math\n"
            "import json\n"
            "from typing import Any, Dict, List, Optional, Tuple, Union\n"
            "from dataclasses import dataclass\n"
            "from pathlib import Path\n"
            "from engine.api.contracts import BaseCartridge, BaseScene\n"
            "class Entity:\n    deliberation: str = 'Entity'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []


# ============================================================================
# 2. Defect 2: ast.AsyncFunctionDef Export Support
# ============================================================================

class TestAsyncFunctionDefExportAdversarial:
    """Stress tests verifying ast.AsyncFunctionDef export and cross-module import."""

    def test_async_functions_complex_signatures_exported(self, tmp_path: Path) -> None:
        """Async functions with default args, varargs, kwargs, and type annotations export cleanly."""
        cart_dir = tmp_path / "async_complex_sig"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "abilities.py").write_text(
            "from typing import Any, Dict\n\n"
            "async def query_state(key: str, default: Any = None, *args, **kwargs) -> Dict[str, Any]:\n"
            "    return {key: default}\n\n"
            "async def dispatch_event(event_name: str, priority: int = 0) -> bool:\n"
            "    return True\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from abilities import query_state, dispatch_event\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_async_generator_functions_exported(self, tmp_path: Path) -> None:
        """Async generator routines (async def with yield) are recognized as exports."""
        cart_dir = tmp_path / "async_generators"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "abilities.py").write_text(
            "async def particle_stream(count: int):\n"
            "    for i in range(count):\n"
            "        yield i\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from abilities import particle_stream\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_high_volume_mixed_exports_sync_async_classes(self, tmp_path: Path) -> None:
        """High volume module mixing 15 classes, 15 sync functions, and 15 async functions."""
        cart_dir = tmp_path / "high_volume_mixed"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        abilities_lines = []
        imported_symbols = []

        for i in range(15):
            cls_name = f"AbilityClass_{i}"
            sync_name = f"sync_helper_{i}"
            async_name = f"async_routine_{i}"

            abilities_lines.append(f"class {cls_name}:\n    deliberation: str = '{cls_name}'\n")
            abilities_lines.append(f"def {sync_name}(): pass\n")
            abilities_lines.append(f"async def {async_name}(): pass\n")

            imported_symbols.extend([cls_name, sync_name, async_name])

        (cart_dir / "abilities.py").write_text("\n".join(abilities_lines), encoding="utf-8")
        (cart_dir / "scenes.py").write_text(
            f"from abilities import {', '.join(imported_symbols)}\n"
            f"class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_async_function_missing_symbol_correctly_rejected(self, tmp_path: Path) -> None:
        """Importing a non-existent async function symbol fails verification and lists available exports."""
        cart_dir = tmp_path / "async_missing_target"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "abilities.py").write_text(
            "async def real_async_step():\n    pass\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from abilities import real_async_step, nonexistent_async_step\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 1
        assert "symbol 'nonexistent_async_step' is not exported by 'abilities.py'" in errors[0]
        assert "real_async_step" in errors[0]

    def test_async_function_imported_with_alias(self, tmp_path: Path) -> None:
        """Importing async function with alias 'from abilities import run_loop as rl' succeeds."""
        cart_dir = tmp_path / "async_alias"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "abilities.py").write_text(
            "async def run_loop():\n    pass\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from abilities import run_loop as rl\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []


# ============================================================================
# 3. Defect 3: Tuple, List, and Starred Unpacking in Assignments
# ============================================================================

class TestUnpackedAssignmentExportsAdversarial:
    """Stress tests verifying recursive extraction of unpacked assignment targets."""

    def test_deep_nested_tuple_and_list_unpacking(self, tmp_path: Path) -> None:
        """Deeply nested tuple and list unpacking: ((A, B), [C, *D]) = ..."""
        cart_dir = tmp_path / "deep_nested_unpack"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "((POS_X, POS_Y), [VEL_X, *EXTRA_VELS]) = ((10, 20), [5, 6, 7, 8])\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import POS_X, POS_Y, VEL_X, EXTRA_VELS\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_ultra_deep_seven_level_nested_unpacking(self, tmp_path: Path) -> None:
        """7-level nested unpacking structure containing tuples, lists, and starred targets."""
        cart_dir = tmp_path / "ultra_deep_unpack"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "[[[[[K1]], [K2]], (K3, (K4, [K5, (K6, *K7)]))]] = [[[[['v1']], ['v2']], ('v3', ('v4', ['v5', ('v6', 'v7', 'v8')]))]]\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import K1, K2, K3, K4, K5, K6, K7\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_starred_target_variations(self, tmp_path: Path) -> None:
        """Various starred target patterns: *HEAD, TAIL; FIRST, *MID, LAST; *SINGLE, = (1,)."""
        cart_dir = tmp_path / "starred_variations"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "*HEAD_ITEMS, TAIL_ITEM = [1, 2, 3]\n"
            "FIRST_NUM, *MID_NUMS, LAST_NUM = (10, 20, 30, 40)\n"
            "*SINGLE_LIST, = (100, 200)\n"
            "[*BRACKET_LIST] = [300, 400]\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import (\n"
            "    HEAD_ITEMS, TAIL_ITEM,\n"
            "    FIRST_NUM, MID_NUMS, LAST_NUM,\n"
            "    SINGLE_LIST, BRACKET_LIST\n"
            ")\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_chained_and_multi_target_assignments(self, tmp_path: Path) -> None:
        """Multiple targets in a single assign: (A, B) = [C, D] = (1, 2); X = Y = Z = 42."""
        cart_dir = tmp_path / "chained_assigns"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "(PAIR_A, PAIR_B) = [PAIR_C, PAIR_D] = (1, 2)\n"
            "CONST_X = CONST_Y = CONST_Z = 42\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import PAIR_A, PAIR_B, PAIR_C, PAIR_D, CONST_X, CONST_Y, CONST_Z\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_non_name_targets_ignored_without_crash_or_false_exports(self, tmp_path: Path) -> None:
        """Assignments targeting obj.attr or dict[key] do not export fake names, but valid names in tuple are exported."""
        cart_dir = tmp_path / "non_name_targets"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "class Container:\n    pass\n"
            "cont = Container()\n"
            "cont.field = 10\n"
            "lookup = {}\n"
            "lookup['key'] = 20\n"
            "(cont.field, VALID_EXPORT) = (30, 40)\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import VALID_EXPORT\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

        # Negative check: attempting to import 'field' or 'key' fails
        (cart_dir / "broken_scene.py").write_text(
            "from entities import field\n"
            "class BrokenScene:\n    deliberation: str = 'Broken'\n",
            encoding="utf-8",
        )
        valid2, errors2 = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid2 is False
        assert any("symbol 'field' is not exported by 'entities.py'" in e for e in errors2)


# ============================================================================
# 4. Defect 4: Directed Cycle Detection Across Local Modules
# ============================================================================

class TestDirectedCycleDetectionAdversarial:
    """Stress tests verifying 3-color DFS cycle detection across diverse graph topologies."""

    def test_two_node_cycle_with_import_from(self, tmp_path: Path) -> None:
        """2-node direct cycle: entities <-> scenes via ImportFrom."""
        cart_dir = tmp_path / "cycle_2_from"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "from scenes import MainScene\n"
            "class Player:\n    deliberation: str = 'Player'\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import Player\n"
            "class MainScene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert any("Circular import detected between local modules: entities -> scenes -> entities" in e for e in errors)

    def test_two_node_cycle_with_plain_import(self, tmp_path: Path) -> None:
        """2-node direct cycle: entities <-> scenes via plain 'import <mod>'."""
        cart_dir = tmp_path / "cycle_2_plain"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "import scenes\n"
            "class Player:\n    deliberation: str = 'Player'\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "import entities\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert any("Circular import detected between local modules: entities -> scenes -> entities" in e for e in errors)

    def test_three_node_cycle(self, tmp_path: Path) -> None:
        """3-node cycle: abilities -> scenes -> mechanics -> abilities."""
        cart_dir = tmp_path / "cycle_3_node"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "abilities.py").write_text(
            "from scenes import LevelScene\n"
            "class Skill:\n    deliberation: str = 'Skill'\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from mechanics import CoreLoop\n"
            "class LevelScene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )
        (cart_dir / "mechanics.py").write_text(
            "from abilities import Skill\n"
            "class CoreLoop:\n    deliberation: str = 'Loop'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert any("Circular import detected between local modules: abilities -> scenes -> mechanics -> abilities" in e for e in errors)

    def test_four_node_cycle(self, tmp_path: Path) -> None:
        """4-node cycle across custom local modules: mod_a -> mod_b -> mod_c -> mod_d -> mod_a."""
        cart_dir = tmp_path / "cycle_4_node"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "mod_a.py").write_text("import mod_b\nclass A: deliberation: str = 'A'\n", encoding="utf-8")
        (cart_dir / "mod_b.py").write_text("import mod_c\nclass B: deliberation: str = 'B'\n", encoding="utf-8")
        (cart_dir / "mod_c.py").write_text("import mod_d\nclass C: deliberation: str = 'C'\n", encoding="utf-8")
        (cart_dir / "mod_d.py").write_text("import mod_a\nclass D: deliberation: str = 'D'\n", encoding="utf-8")

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert any("Circular import detected between local modules: mod_a -> mod_b -> mod_c -> mod_d -> mod_a" in e for e in errors)

    def test_self_import_loop(self, tmp_path: Path) -> None:
        """Self-loop: a module importing itself directly."""
        cart_dir = tmp_path / "self_loop"
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

    def test_figure_eight_double_cycle(self, tmp_path: Path) -> None:
        """Figure-8 topology: mod_a <-> mod_b AND mod_a <-> mod_c."""
        cart_dir = tmp_path / "figure_eight"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "mod_a.py").write_text(
            "import mod_b\n"
            "import mod_c\n"
            "class A: deliberation: str = 'A'\n",
            encoding="utf-8",
        )
        (cart_dir / "mod_b.py").write_text(
            "import mod_a\n"
            "class B: deliberation: str = 'B'\n",
            encoding="utf-8",
        )
        (cart_dir / "mod_c.py").write_text(
            "import mod_a\n"
            "class C: deliberation: str = 'C'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        # Both distinct canonical cycles must be captured
        assert any("mod_a -> mod_b -> mod_a" in e for e in errors)
        assert any("mod_a -> mod_c -> mod_a" in e for e in errors)

    def test_disconnected_subgraphs_with_cycle_in_one(self, tmp_path: Path) -> None:
        """Disconnected components: Component 1 has a cycle (c1_a <-> c1_b), Component 2 is an acyclic chain (c2_x -> c2_y)."""
        cart_dir = tmp_path / "disconnected_comp"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        # Component 1 (cyclic)
        (cart_dir / "c1_a.py").write_text("import c1_b\nclass C1A: deliberation: str = 'A'\n", encoding="utf-8")
        (cart_dir / "c1_b.py").write_text("import c1_a\nclass C1B: deliberation: str = 'B'\n", encoding="utf-8")

        # Component 2 (acyclic)
        (cart_dir / "c2_y.py").write_text("class C2Y: deliberation: str = 'Y'\n", encoding="utf-8")
        (cart_dir / "c2_x.py").write_text("from c2_y import C2Y\nclass C2X: deliberation: str = 'X'\n", encoding="utf-8")

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 1
        assert "Circular import detected between local modules: c1_a -> c1_b -> c1_a" in errors[0]

    def test_complex_multi_path_dag_no_false_positive(self, tmp_path: Path) -> None:
        """Complex 6-node multi-path DAG with multiple forward and cross edges passes with zero cycle errors."""
        cart_dir = tmp_path / "multipath_dag"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        # Topology:
        # layer0: base_utils
        # layer1: entities (imports base_utils), abilities (imports base_utils)
        # layer2: scenes (imports entities, abilities, base_utils)
        # layer3: level (imports scenes, entities, base_utils)
        # layer4: mechanics (imports level, scenes, abilities, entities, base_utils)

        (cart_dir / "base_utils.py").write_text(
            "class Vector:\n    deliberation: str = 'Vec'\n"
            "BASE_GRAVITY = 9.8\n",
            encoding="utf-8",
        )
        (cart_dir / "entities.py").write_text(
            "from base_utils import Vector, BASE_GRAVITY\n"
            "class Entity:\n    deliberation: str = 'Entity'\n",
            encoding="utf-8",
        )
        (cart_dir / "abilities.py").write_text(
            "from base_utils import Vector\n"
            "class Ability:\n    deliberation: str = 'Ability'\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from base_utils import BASE_GRAVITY\n"
            "from entities import Entity\n"
            "from abilities import Ability\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )
        (cart_dir / "level.py").write_text(
            "from base_utils import Vector\n"
            "from entities import Entity\n"
            "from scenes import Scene\n"
            "class Level:\n    deliberation: str = 'Level'\n",
            encoding="utf-8",
        )
        (cart_dir / "mechanics.py").write_text(
            "from base_utils import BASE_GRAVITY\n"
            "from entities import Entity\n"
            "from abilities import Ability\n"
            "from scenes import Scene\n"
            "from level import Level\n"
            "class Cartridge:\n    deliberation: str = 'Cartridge'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_bundle_assembly_aborts_on_cycle(self, tmp_path: Path) -> None:
        """OpenGenreCartridgeBundle.assemble_cartridge_directory raises CodeVerificationError on cyclic bundle."""
        cart_dir = tmp_path / "bundle_cycle_abort"
        swe = LeadGameplaySWEAgent()

        bundle = OpenGenreCartridgeBundle(
            deliberation="Bundle with cycle between mechanics and scenes.",
            mechanics_code=(
                "from engine.api.contracts import BaseCartridge\n"
                "from scenes import SceneA\n"
                "class Cartridge(BaseCartridge):\n"
                "    deliberation: str = 'Cart'\n"
                "    def get_scenes(self, e): return {'a': SceneA}\n"
                "    def get_initial_scene(self, e): return 'a'\n"
            ),
            scenes_code=(
                "from engine.api.contracts import BaseScene\n"
                "from mechanics import Cartridge\n"
                "class SceneA(BaseScene):\n"
                "    deliberation: str = 'Scene'\n"
                "    def update(self, dt): pass\n"
                "    def render(self, s): pass\n"
            ),
        )

        with pytest.raises(CodeVerificationError) as exc_info:
            bundle.assemble_cartridge_directory(cart_dir, swe_agent=swe)
        assert "Circular import detected between local modules: mechanics -> scenes -> mechanics" in str(exc_info.value)


    def test_twenty_node_tournament_dag_performance(self, tmp_path: Path) -> None:
        """Stress-tests a dense 20-node tournament DAG (190 forward edges) with zero false-positive cycles."""
        cart_dir = tmp_path / "tournament_dag_20"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        # Nodes 0..19. Node i imports all j > i.
        num_nodes = 20
        for i in range(num_nodes):
            lines = [f"class Node_{i:02d}:\n    deliberation: str = 'Node {i}'\n"]
            for j in range(i + 1, num_nodes):
                lines.append(f"from node_{j:02d} import Node_{j:02d}")
            (cart_dir / f"node_{i:02d}.py").write_text("\n".join(lines), encoding="utf-8")

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_twenty_node_giant_cycle_detection(self, tmp_path: Path) -> None:
        """Detects a 20-node long directed cycle across local cartridge modules."""
        cart_dir = tmp_path / "giant_cycle_20"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        num_nodes = 20
        for i in range(num_nodes):
            next_node = (i + 1) % num_nodes
            content = (
                f"from ring_{next_node:02d} import Ring_{next_node:02d}\n"
                f"class Ring_{i:02d}:\n    deliberation: str = 'Ring {i}'\n"
            )
            (cart_dir / f"ring_{i:02d}.py").write_text(content, encoding="utf-8")

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=False)
        assert valid is False
        assert len(errors) == 1
        assert "Circular import detected between local modules:" in errors[0]
        # Verify all 20 ring nodes are listed in the cycle chain
        for i in range(num_nodes):
            assert f"ring_{i:02d}" in errors[0]

    def test_empty_and_docstring_only_modules(self, tmp_path: Path) -> None:
        """Handles empty files and files containing only comments or docstrings gracefully."""
        cart_dir = tmp_path / "empty_and_comments"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text("", encoding="utf-8")
        (cart_dir / "scenes.py").write_text("# Just a comment\n# Another comment\n", encoding="utf-8")
        (cart_dir / "abilities.py").write_text('"""Module docstring only."""\n', encoding="utf-8")

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []

    def test_type_annotated_variable_without_initial_value(self, tmp_path: Path) -> None:
        """ast.AnnAssign target without initial value (e.g. VAR: int) is recognized in exports."""
        cart_dir = tmp_path / "annassign_no_val"
        cart_dir.mkdir(parents=True, exist_ok=True)
        swe = LeadGameplaySWEAgent()

        (cart_dir / "entities.py").write_text(
            "UNINITIALIZED_VAR: int\n"
            "INITIALIZED_VAR: str = 'hello'\n",
            encoding="utf-8",
        )
        (cart_dir / "scenes.py").write_text(
            "from entities import UNINITIALIZED_VAR, INITIALIZED_VAR\n"
            "class Scene:\n    deliberation: str = 'Scene'\n",
            encoding="utf-8",
        )

        valid, errors = swe.verify_cartridge_cross_module_imports(cart_dir, raise_on_error=True)
        assert valid is True
        assert errors == []
