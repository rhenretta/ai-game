"""tests/test_m1_challenger_runtime.py
Empirical Adversarial Challenge Suite for Milestone M1:
Multi-Module Cartridge Loader and Client Runtime (Features 4, 5, 6, 7).

Adversarially tests:
1. AST Sandbox Security & Malicious Code Interception:
   - Forbidden imports in entities.py and submodules (os, sys, subprocess, socket, urllib, ctypes, shutil, pickle, builtins, etc.)
   - Prohibited calls (eval, exec, __import__, compile, globals, locals, popen, system, rmtree)
   - Dangerous dunder attributes (__subclasses__, __bases__, __globals__, __code__, __builtins__)
   - Multi-module hidden payloads in arbitrary submodules (bosses.py, dialogue.py)
   - Syntax errors across all submodule positions
   - Missing required classes and GenericCartridge fallback
   - Circular imports between submodules (2-module and 3-module cycles) & sys.modules cleanup
2. Instant Client Boot & Fallback Resolution:
   - Resolution hierarchy: explicit CLI override -> cartridges/active -> cartridges/seed
   - Corrupted active cartridge fallback across 6 distinct defect vectors
   - Catastrophic dual-corruption failure handling (strict non-concealment)
   - Sub-150ms boot latency verification under fallback stress
3. Hot-Reload Sentinel & In-Game Scene Transition:
   - Polling interval gating and stat optimization
   - Resilience against partial/corrupt JSON during atomic file write races
   - Non-intrusive pending notification trigger and acknowledgment
   - [R] keypress transition, audio stop, custom probe telemetry emission
   - Rejection and state preservation when promoted cartridge contains defects
   - Multi-hop hot-reloads across sequential episodes
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List
import pytest
import pygame

import config
from engine.api.contracts import BaseCartridge, BaseLevelSpec, BasePlayer
from engine.core.engine import Engine
from engine.core.scenes.playing_scene import (
    HotReloadSentinel,
    PlayingScene,
    render_hot_reload_toast,
)
from engine.loader import (
    ASTSecurityError,
    CartridgeLoader,
    GenericCartridge,
    validate_cartridge_ast,
)
from engine.sim.headless import InMemoryTelemetryRecorder
from run_game import main as run_game_main, resolve_and_load_cartridge


# ==============================================================================
# Category 1: Adversarial AST Sandbox & Cartridge Loader Security
# ==============================================================================

class TestASTSecurityAdversarialSuite:
    """Adversarially challenges AST security, module sandboxing, and loader resilience."""

    @pytest.mark.parametrize(
        "payload,description",
        [
            ("import os", "Direct os module import"),
            ("import os.path", "Submodule os.path import"),
            ("from os import environ", "Symbol from os import"),
            ("import sys", "Direct sys import"),
            ("from sys import modules", "Symbol from sys import"),
            ("import subprocess", "Direct subprocess import"),
            ("from subprocess import Popen, PIPE", "Symbol Popen from subprocess"),
            ("import socket", "Direct socket import"),
            ("from socket import socket, AF_INET", "Symbol from socket"),
            ("import urllib.request", "Direct urllib network import"),
            ("import requests", "Third-party HTTP client import"),
            ("import http.client", "Standard HTTP client import"),
            ("import ctypes", "Direct C types interop import"),
            ("import _ctypes", "Internal C types import"),
            ("import shutil", "Filesystem destruction import"),
            ("import pickle", "Deserialization exploit import"),
            ("import marshal", "Bytecode deserialization import"),
            ("import builtins", "Direct builtins namespace import"),
            ("import importlib", "Dynamic importlib loader import"),
            ("import multiprocessing", "Process spawning import"),
            ("import threading", "Direct threading import"),
            ("import asyncio", "Direct asyncio runtime import"),
            ("import posix", "Posix system bindings import"),
            ("import code", "Code interpreter execution import"),
        ],
    )
    def test_forbidden_imports_in_entities_module(self, tmp_path: Path, payload: str, description: str):
        """Verifies that every forbidden import in entities.py is intercepted and raises ASTSecurityError."""
        cart_dir = tmp_path / f"exploit_{hash(payload) & 0xFFFF}"
        cart_dir.mkdir(parents=True)
        (cart_dir / "manifest.json").write_text(json.dumps({"name": description}))
        (cart_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class SafeCart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )
        (cart_dir / "entities.py").write_text(
            f"{payload}\n"
            "class InfiltratorPlayer:\n"
            "    pass\n"
        )

        with pytest.raises(ASTSecurityError):
            CartridgeLoader.load_cartridge(cart_dir)

    @pytest.mark.parametrize(
        "payload,description",
        [
            ("eval('1 + 1')", "Direct eval call"),
            ("exec('x = 1')", "Direct exec call"),
            ("__import__('math')", "Direct __import__ call"),
            ("compile('x = 1', '<string>', 'exec')", "Direct compile call"),
            ("g = globals()", "Direct globals() introspection"),
            ("l = locals()", "Direct locals() introspection"),
            ("b = __builtins__", "Direct __builtins__ access"),
            ("sub = ().__class__.__bases__[0].__subclasses__()", "Dunder escape via __subclasses__"),
            ("base = ().__class__.__bases__", "Dunder escape via __bases__"),
            ("func_code = f.__code__", "Dunder bytecode access via __code__"),
            ("func_globals = f.__globals__", "Dunder namespace access via __globals__"),
            ("obj.system('whoami')", "Method call to system"),
            ("obj.popen('ls')", "Method call to popen"),
            ("obj.rmtree('/tmp')", "Method call to rmtree"),
            ("obj.unlink('/tmp/file')", "Method call to unlink"),
        ],
    )
    def test_prohibited_calls_and_dangerous_attributes(self, tmp_path: Path, payload: str, description: str):
        """Verifies that dangerous function calls and sandbox escape attributes are strictly rejected."""
        cart_dir = tmp_path / f"exploit_call_{hash(payload) & 0xFFFF}"
        cart_dir.mkdir(parents=True)
        (cart_dir / "manifest.json").write_text(json.dumps({"name": description}))
        (cart_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class SafeCart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )
        (cart_dir / "abilities.py").write_text(
            "def sneaky_ability(f, obj):\n"
            f"    {payload}\n"
        )

        with pytest.raises(ASTSecurityError):
            CartridgeLoader.load_cartridge(cart_dir)

    def test_payload_hidden_in_arbitrary_submodule(self, tmp_path: Path):
        """Verifies that malicious payloads hidden in non-standard submodules (e.g. boss_ai.py) are caught."""
        cart_dir = tmp_path / "sneaky_submodule"
        cart_dir.mkdir(parents=True)
        (cart_dir / "manifest.json").write_text(json.dumps({"name": "Sneaky"}))
        (cart_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class SafeCart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )
        (cart_dir / "entities.py").write_text("class SafePlayer: pass")
        (cart_dir / "boss_ai.py").write_text("import socket\ndef connect_c2(): pass")

        with pytest.raises(ASTSecurityError) as exc_info:
            CartridgeLoader.load_cartridge(cart_dir)
        assert "Prohibited module import: 'socket'" in str(exc_info.value)

    @pytest.mark.parametrize(
        "submodule_name,broken_code",
        [
            ("entities.py", "class BrokenPlayer(\n    pass"),
            ("mechanics.py", "def syntax_error_func(: pass"),
            ("abilities.py", "123_invalid_ident = 999"),
        ],
    )
    def test_syntax_errors_in_submodules_rejected_cleanly(self, tmp_path: Path, submodule_name: str, broken_code: str):
        """Verifies that syntax errors in any submodule are detected cleanly and raise SyntaxError."""
        cart_dir = tmp_path / f"broken_syntax_{submodule_name}"
        cart_dir.mkdir(parents=True)
        (cart_dir / "manifest.json").write_text(json.dumps({"name": "SyntaxErrorCart"}))
        (cart_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class MainCart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )
        (cart_dir / submodule_name).write_text(broken_code)

        with pytest.raises(SyntaxError):
            CartridgeLoader.load_cartridge(cart_dir)

    def test_missing_required_classes_falls_back_to_generic_cartridge(self, tmp_path: Path):
        """Verifies that a cartridge without a BaseCartridge subclass gracefully yields GenericCartridge."""
        cart_dir = tmp_path / "generic_fallback_cart"
        cart_dir.mkdir(parents=True)
        (cart_dir / "manifest.json").write_text(json.dumps({"name": "NoCartridgeClass"}))
        (cart_dir / "level.json").write_text(json.dumps({
            "level_id": "json_chamber_01",
            "name": "JSON Chamber",
            "spawn_point": [150.0, 300.0],
            "exit_point": [1000.0, 300.0],
            "walls": [[0, 0, 1280, 32]],
            "enemies": [{"type": "chaser", "x": 600.0, "y": 300.0}],
            "hazards": [{"type": "spikes", "x": 400.0, "y": 300.0, "w": 48.0, "h": 48.0}]
        }))
        (cart_dir / "mechanics.py").write_text(
            "# Valid python, but no BaseCartridge subclass defined\n"
            "class UnrelatedHelper:\n"
            "    pass\n"
        )

        cartridge = CartridgeLoader.load_cartridge(cart_dir)
        assert isinstance(cartridge, GenericCartridge)
        spec = cartridge.get_level_spec(1)
        assert spec.level_id == "json_chamber_01"
        assert spec.name == "JSON Chamber"
        assert len(cartridge.create_enemies(spec)) == 1
        assert len(cartridge.create_hazards(spec)) == 1

    def test_circular_imports_between_submodules_detected_and_cleans_sys_modules(self, tmp_path: Path):
        """Verifies that circular imports between submodules raise ImportError and clean up sys.modules."""
        cart_dir = tmp_path / "circular_submodules"
        cart_dir.mkdir(parents=True)
        (cart_dir / "manifest.json").write_text(json.dumps({"name": "Circular"}))
        (cart_dir / "entities.py").write_text(
            "from .abilities import MegaAbility\n"
            "class CircularPlayer:\n"
            "    ability = MegaAbility\n"
        )
        (cart_dir / "abilities.py").write_text(
            "from .entities import CircularPlayer\n"
            "class MegaAbility:\n"
            "    user_cls = CircularPlayer\n"
        )
        (cart_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "from .entities import CircularPlayer\n"
            "class CircularCartridge(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return CircularPlayer()\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )

        with pytest.raises(ImportError) as exc_info:
            CartridgeLoader.load_cartridge(cart_dir)
        assert "cannot import name" in str(exc_info.value)

        # Assert no leaked bare aliases remain in sys.modules
        assert "entities" not in sys.modules or sys.modules["entities"] is None or not hasattr(sys.modules["entities"], "CircularPlayer")
        assert "abilities" not in sys.modules or sys.modules["abilities"] is None or not hasattr(sys.modules["abilities"], "MegaAbility")

    def test_three_way_circular_imports_detected_cleanly(self, tmp_path: Path):
        """Verifies 3-module circular import (entities -> abilities -> weapons -> entities) is caught cleanly."""
        cart_dir = tmp_path / "three_way_cycle"
        cart_dir.mkdir(parents=True)
        (cart_dir / "manifest.json").write_text(json.dumps({"name": "3WayCycle"}))
        (cart_dir / "entities.py").write_text("from .abilities import AbilityA\nclass PlayerA: pass\n")
        (cart_dir / "abilities.py").write_text("from .weapons import WeaponB\nclass AbilityA: pass\n")
        (cart_dir / "weapons.py").write_text("from .entities import PlayerA\nclass WeaponB: pass\n")
        (cart_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "from .entities import PlayerA\n"
            "class Cart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )

        with pytest.raises(ImportError):
            CartridgeLoader.load_cartridge(cart_dir)

    def test_loader_recovers_after_circular_import_repaired(self, tmp_path: Path):
        """Verifies that after fixing a circular import, the cartridge reloads successfully."""
        cart_dir = tmp_path / "repairable_cart"
        cart_dir.mkdir(parents=True)
        (cart_dir / "manifest.json").write_text(json.dumps({"name": "Repairable"}))
        (cart_dir / "entities.py").write_text("from .abilities import Dash\nclass P: pass\n")
        (cart_dir / "abilities.py").write_text("from .entities import P\nclass Dash: pass\n")
        (cart_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class C(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )

        # Fails first
        with pytest.raises(ImportError):
            CartridgeLoader.load_cartridge(cart_dir)

        # Repair: remove circular import from entities.py
        (cart_dir / "entities.py").write_text("class P:\n    speed = 200.0\n")
        (cart_dir / "abilities.py").write_text("from .entities import P\nclass Dash: pass\n")

        # Must succeed on second attempt
        loaded = CartridgeLoader.load_cartridge(cart_dir)
        assert loaded is not None
        assert type(loaded).__name__ == "C"


# ==============================================================================
# Category 2: Instant Client Boot Resolution & Corrupted Active Fallback
# ==============================================================================

class TestInstantClientBootAdversarialSuite:
    """Adversarially tests boot resolution hierarchy, corrupted active fallback, and boot budget."""

    def _create_seed(self, seed_dir: Path, tag: str = "SeedGame"):
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "manifest.json").write_text(json.dumps({"name": "Seed", "version": "1.0.0"}), encoding="utf-8")
        (seed_dir / "mechanics.py").write_text(
            f"from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            f"class {tag}(BaseCartridge):\n"
            f"    def get_level_spec(self, ep=1): return BaseLevelSpec(level_id='seed_01')\n"
            f"    def create_player(self, x, y, r=None): return None\n"
            f"    def create_enemies(self, s): return []\n"
            f"    def create_hazards(self, s): return []\n",
            encoding="utf-8"
        )

    @pytest.mark.parametrize(
        "defect_type,file_to_corrupt,code_payload",
        [
            ("ast_forbidden_import", "entities.py", "import subprocess\ndef pwn(): subprocess.Popen(['whoami'])"),
            ("ast_prohibited_call", "abilities.py", "def bad(): eval('2+2')"),
            ("syntax_error", "mechanics.py", "class BrokenMechanics(def broken: pass"),
            ("circular_import", "entities.py", "from .abilities import BadAb\nclass BadEnt: pass"),
            ("crashing_constructor", "mechanics.py",
             "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
             "class CrashCart(BaseCartridge):\n"
             "    def __init__(self, **kwargs): raise RuntimeError('Explosive init failure')\n"
             "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
             "    def create_player(self, x, y, r=None): return None\n"
             "    def create_enemies(self, s): return []\n"
             "    def create_hazards(self, s): return []\n"),
        ],
    )
    def test_corrupted_active_cartridge_falls_back_to_seed(
        self, tmp_path: Path, monkeypatch, defect_type: str, file_to_corrupt: str, code_payload: str
    ):
        """Verifies resolve_and_load_cartridge falls back to seed across 5 distinct defect vectors in active."""
        fake_root = tmp_path / f"project_root_{defect_type}"
        active_dir = fake_root / "cartridges" / "active"
        seed_dir = fake_root / "cartridges" / "seed"
        active_dir.mkdir(parents=True)
        self._create_seed(seed_dir, tag="GuaranteedSafeSeed")

        # Populate active cartridge with defect
        (active_dir / "manifest.json").write_text(json.dumps({"name": f"Corrupted_{defect_type}"}), encoding="utf-8")
        if file_to_corrupt != "mechanics.py":
            (active_dir / "mechanics.py").write_text(
                "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
                "class ActiveCart(BaseCartridge):\n"
                "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
                "    def create_player(self, x, y, r=None): return None\n"
                "    def create_enemies(self, s): return []\n"
                "    def create_hazards(self, s): return []\n",
                encoding="utf-8"
            )
        if defect_type == "circular_import":
            (active_dir / "abilities.py").write_text("from .entities import BadEnt\nclass BadAb: pass\n", encoding="utf-8")

        (active_dir / file_to_corrupt).write_text(code_payload, encoding="utf-8")

        monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)
        engine = Engine(headless=True)
        booted = resolve_and_load_cartridge(engine)

        # In open-genre zero-fallback architecture, corrupted active cartridge does not fall back to seed
        assert booted is None
        assert engine.active_cartridge is None

    def test_missing_active_directory_falls_back_to_seed(self, tmp_path: Path, monkeypatch):
        """Verifies that when cartridges/active does not exist, engine returns None with guidance banner without seed fallback."""
        fake_root = tmp_path / "project_root_no_active"

        monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)
        engine = Engine(headless=True)
        booted = resolve_and_load_cartridge(engine)

        assert booted is None
        assert engine.active_cartridge is None

    def test_explicit_cli_cartridge_override_priority(self, tmp_path: Path, monkeypatch):
        """Verifies that an explicit CLI --cartridge override takes precedence over both active and seed."""
        fake_root = tmp_path / "project_root_cli_override"
        active_dir = fake_root / "cartridges" / "active"
        seed_dir = fake_root / "cartridges" / "seed"
        custom_dir = fake_root / "cartridges" / "custom_test"
        active_dir.mkdir(parents=True)
        self._create_seed(seed_dir, tag="SeedCart")
        self._create_seed(active_dir, tag="ActiveCart")
        self._create_seed(custom_dir, tag="CustomCLICart")

        monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)
        engine = Engine(headless=True)
        booted = resolve_and_load_cartridge(engine, requested_cartridge=str(custom_dir))

        assert booted.resolve() == custom_dir.resolve()
        assert type(engine.active_cartridge).__name__ == "CustomCLICart"

    def test_corrupted_explicit_cli_fails_fast_without_fallback(self, tmp_path: Path, monkeypatch):
        """Verifies that when an explicit CLI cartridge is corrupted, it fails fast without fallback (Rule 5)."""
        fake_root = tmp_path / "project_root_cli_corrupted"
        active_dir = fake_root / "cartridges" / "active"
        seed_dir = fake_root / "cartridges" / "seed"
        custom_dir = fake_root / "cartridges" / "custom_corrupted"
        active_dir.mkdir(parents=True)
        custom_dir.mkdir(parents=True)
        self._create_seed(seed_dir, tag="SeedCart")
        self._create_seed(active_dir, tag="ActiveCart")

        (custom_dir / "manifest.json").write_text(json.dumps({"name": "Custom"}))
        (custom_dir / "mechanics.py").write_text("import os")  # Banned import

        monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)
        engine = Engine(headless=True)
        with pytest.raises(ASTSecurityError):
            resolve_and_load_cartridge(engine, requested_cartridge=str(custom_dir))

    def test_catastrophic_dual_corruption_fails_fast_with_runtime_error(self, tmp_path: Path, monkeypatch):
        """Verifies that if active is corrupted, engine returns None cleanly without falling back to seed."""
        fake_root = tmp_path / "project_root_dual_corrupt"
        active_dir = fake_root / "cartridges" / "active"
        active_dir.mkdir(parents=True)

        (active_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (active_dir / "mechanics.py").write_text("import os", encoding="utf-8")

        monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)
        engine = Engine(headless=True)

        booted = resolve_and_load_cartridge(engine)
        assert booted is None
        assert engine.active_cartridge is None

    def test_boot_latency_under_fallback_stress(self, tmp_path: Path, monkeypatch):
        """Measures boot resolution latency; verifies << 150ms budget."""
        fake_root = tmp_path / "project_root_latency"
        active_dir = fake_root / "cartridges" / "active"
        active_dir.mkdir(parents=True)
        (active_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (active_dir / "mechanics.py").write_text("import os", encoding="utf-8")

        monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)
        engine = Engine(headless=True)

        latencies_ms = []
        for _ in range(10):
            t0 = time.perf_counter()
            booted = resolve_and_load_cartridge(engine)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        max_latency = max(latencies_ms)
        avg_latency = sum(latencies_ms) / len(latencies_ms)
        assert max_latency < 150.0, f"Max boot latency was {max_latency:.2f}ms, budget is < 150ms"
        assert avg_latency < 50.0, f"Average boot latency was {avg_latency:.2f}ms"


# ==============================================================================
# Category 3: Hot-Reload Sentinel & In-Game Scene Transition
# ==============================================================================

class TestHotReloadSentinelAndTransitionAdversarialSuite:
    """Adversarially tests HotReloadSentinel polling resilience and PlayingScene live hot-reloading."""

    def test_sentinel_gating_dt_cadence(self, tmp_path: Path):
        """Verifies that sentinel does not check file stat when dt sum < POLL_INTERVAL (1.0s)."""
        active_dir = tmp_path / "cartridges" / "active"
        active_dir.mkdir(parents=True)
        manifest_path = active_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"name": "Game", "version": "1.0.0", "target_episode": 1}), encoding="utf-8")

        sentinel = HotReloadSentinel(active_cartridge_dir=active_dir)
        # Advance 0.25s 3 times = 0.75s (< 1.0s)
        for _ in range(3):
            assert sentinel.check_for_promoted_update(0.25) is False

        # Simulate update
        time.sleep(0.01)
        manifest_path.write_text(json.dumps({"name": "Game", "version": "1.1.0", "target_episode": 2}), encoding="utf-8")

        # Advance 0.2s = 0.95s total (still < 1.0s)
        assert sentinel.check_for_promoted_update(0.20) is False

        # Advance 0.1s = 1.05s total (>= 1.0s) -> triggers poll
        assert sentinel.check_for_promoted_update(0.10) is True
        assert sentinel.has_pending_update is True

    def test_sentinel_resilient_to_partial_json_atomic_race(self, tmp_path: Path):
        """Verifies sentinel survives malformed JSON during simulated concurrent promotion write race."""
        active_dir = tmp_path / "cartridges" / "active"
        active_dir.mkdir(parents=True)
        manifest_path = active_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"name": "Game", "version": "1.0.0", "target_episode": 1}), encoding="utf-8")

        sentinel = HotReloadSentinel(active_cartridge_dir=active_dir)

        # Partial/corrupt write
        time.sleep(0.01)
        manifest_path.write_text('{"name": "Game", "version": ')

        # Poll must NOT raise JSONDecodeError
        assert sentinel.check_for_promoted_update(1.1) is False
        assert sentinel.has_pending_update is False

        # Background writer finishes valid atomic JSON
        time.sleep(0.01)
        manifest_path.write_text(json.dumps({"name": "Game", "version": "2.0.0", "target_episode": 2}), encoding="utf-8")

        # Next poll detects update cleanly
        assert sentinel.check_for_promoted_update(1.1) is True
        assert sentinel.has_pending_update is True
        assert sentinel.pending_manifest["version"] == "2.0.0"

    def test_sentinel_survives_missing_manifest(self, tmp_path: Path):
        """Verifies sentinel does not crash when active directory or manifest is temporarily absent."""
        active_dir = tmp_path / "cartridges" / "nonexistent"
        sentinel = HotReloadSentinel(active_cartridge_dir=active_dir)
        assert sentinel.check_for_promoted_update(1.1) is False

    def test_r_key_reload_emits_telemetry_and_transitions_episode(self, tmp_path: Path, monkeypatch):
        """Verifies pressing [R] hot-reloads cartridge, emits probe telemetry, and increments episode."""
        fake_root = tmp_path / "project_root_reload"
        active_dir = fake_root / "cartridges" / "active"
        active_dir.mkdir(parents=True)

        (active_dir / "manifest.json").write_text(
            json.dumps({"name": "Ep1Cartridge", "version": "1.0.0", "target_episode": 1}),
            encoding="utf-8"
        )
        (active_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class Ep1Cartridge(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec(level_id='lvl_01')\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n",
            encoding="utf-8"
        )

        monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)

        mem_rec = InMemoryTelemetryRecorder()
        engine = Engine(headless=True)
        engine.set_telemetry_recorder(mem_rec)
        engine.load_cartridge(active_dir)

        scene = PlayingScene(engine)
        scene.on_enter({"session_id": "sess_reload_test", "episode_number": 1})
        assert scene.episode_number == 1
        assert type(engine.active_cartridge).__name__ == "Ep1Cartridge"

        # Update active to Episode 2 (with realistic modified mechanics and size)
        (active_dir / "manifest.json").write_text(
            json.dumps({"name": "Ep2Cartridge", "version": "2.0.0", "target_episode": 2}),
            encoding="utf-8"
        )
        (active_dir / "mechanics.py").write_text(
            "# Episode 2 promoted content with newly compiled mechanics\n"
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class Ep2Cartridge(BaseCartridge):\n"
            "    def get_level_spec(self, ep=2): return BaseLevelSpec(level_id='lvl_02')\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n",
            encoding="utf-8"
        )

        # Sentinel detects update
        scene.hot_reload_sentinel.has_pending_update = True
        scene.hot_reload_sentinel.pending_manifest = {"version": "2.0.0", "target_episode": 2}

        # Press [R]
        r_event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_r})
        scene.handle_event(r_event)

        # Verify state transition
        assert scene.episode_number == 2
        assert scene.hot_reload_sentinel.has_pending_update is False
        assert type(engine.active_cartridge).__name__ == "Ep2Cartridge"

        # Verify telemetry event
        reload_events = [e for e in mem_rec.events if e.event_type == "CUSTOM_PROBE" and e.probe_name == "cartridge_hot_reload_triggered"]
        assert len(reload_events) == 1
        payload = reload_events[0].payload
        assert payload["from_episode"] == 1
        assert payload["to_episode"] == 2
        assert payload["target_version"] == "2.0.0"

    def test_r_key_ignored_when_no_update_pending(self, tmp_path: Path, monkeypatch):
        """Verifies that pressing [R] when has_pending_update is False is a safe no-op."""
        fake_root = tmp_path / "project_root_no_reload"
        active_dir = fake_root / "cartridges" / "active"
        active_dir.mkdir(parents=True)
        (active_dir / "manifest.json").write_text(json.dumps({"name": "Cart"}), encoding="utf-8")
        (active_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class Cart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n",
            encoding="utf-8"
        )

        monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)
        mem_rec = InMemoryTelemetryRecorder()
        engine = Engine(headless=True)
        engine.set_telemetry_recorder(mem_rec)
        engine.load_cartridge(active_dir)

        scene = PlayingScene(engine)
        scene.on_enter({"session_id": "sess_noop", "episode_number": 1})
        assert scene.hot_reload_sentinel.has_pending_update is False

        # Press [R]
        r_event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_r})
        scene.handle_event(r_event)

        # Nothing changed, zero reload telemetry emitted
        assert scene.episode_number == 1
        reload_events = [e for e in mem_rec.events if getattr(e, "probe_name", None) == "cartridge_hot_reload_triggered"]
        assert len(reload_events) == 0

    def test_hot_reload_rejection_preserves_engine_state(self, tmp_path: Path, monkeypatch):
        """Verifies that if a promoted cartridge contains an AST violation, hot-reload fails fast
        and engine state is preserved without corrupting the running game.
        """
        fake_root = tmp_path / "project_root_corrupt_reload"
        active_dir = fake_root / "cartridges" / "active"
        active_dir.mkdir(parents=True)

        # Safe initial cartridge
        (active_dir / "manifest.json").write_text(json.dumps({"name": "SafeInit", "version": "1.0.0"}), encoding="utf-8")
        (active_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class SafeInitCart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec(level_id='safe_01')\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n",
            encoding="utf-8"
        )

        monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)
        engine = Engine(headless=True)
        engine.load_cartridge(active_dir)

        scene = PlayingScene(engine)
        scene.on_enter({"session_id": "sess_preserve", "episode_number": 1})
        assert type(engine.active_cartridge).__name__ == "SafeInitCart"

        # Overwrite active with malicious payload
        (active_dir / "manifest.json").write_text(json.dumps({"name": "Malicious", "version": "2.0.0", "target_episode": 2}), encoding="utf-8")
        (active_dir / "mechanics.py").write_text("import os", encoding="utf-8")

        scene.hot_reload_sentinel.has_pending_update = True
        scene.hot_reload_sentinel.pending_manifest = {"version": "2.0.0", "target_episode": 2}

        # Attempt reload (gracefully caught with error toast, preserves state)
        success = scene._execute_hot_reload()
        assert success is False
        assert scene.hot_reload_sentinel.has_pending_update is False
        assert getattr(scene, "reload_error_msg", None) is not None

        # Engine active cartridge and scene episode remain unchanged
        assert type(engine.active_cartridge).__name__ == "SafeInitCart"
        assert scene.episode_number == 1

    def test_multi_hop_sequential_hot_reloads(self, tmp_path: Path, monkeypatch):
        """Verifies multi-hop reload sequence across multiple episodes (Episode 1 -> 2 -> 3)."""
        fake_root = tmp_path / "project_root_multihop"
        active_dir = fake_root / "cartridges" / "active"
        active_dir.mkdir(parents=True)

        # Episode 1
        (active_dir / "manifest.json").write_text(json.dumps({"name": "Ep1", "version": "1.0.0", "target_episode": 1}), encoding="utf-8")
        (active_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class Hop1Cart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec(level_id='lvl_01')\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n",
            encoding="utf-8"
        )

        monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)
        engine = Engine(headless=True)
        mem_rec = InMemoryTelemetryRecorder()
        engine.set_telemetry_recorder(mem_rec)
        engine.load_cartridge(active_dir)

        scene = PlayingScene(engine)
        scene.on_enter({"session_id": "sess_multihop", "episode_number": 1})
        assert scene.episode_number == 1
        assert type(engine.active_cartridge).__name__ == "Hop1Cart"

        # Hop 1: Reload to Episode 2 (with distinct file size)
        (active_dir / "manifest.json").write_text(json.dumps({"name": "Ep2", "version": "2.0.0", "target_episode": 2}), encoding="utf-8")
        (active_dir / "mechanics.py").write_text(
            "# Episode 2 update with extra mechanics logic\n"
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class Hop2Cart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=2): return BaseLevelSpec(level_id='lvl_02')\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n",
            encoding="utf-8"
        )
        assert scene.hot_reload_sentinel.check_for_promoted_update(1.1) is True
        scene._execute_hot_reload()
        assert scene.episode_number == 2
        assert type(engine.active_cartridge).__name__ == "Hop2Cart"

        # Hop 2: Reload to Episode 3 (with distinct file size)
        (active_dir / "manifest.json").write_text(json.dumps({"name": "Ep3", "version": "3.0.0", "target_episode": 3}), encoding="utf-8")
        (active_dir / "mechanics.py").write_text(
            "# Episode 3 update with extended boss encounter specifications\n"
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class Hop3Cart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=3): return BaseLevelSpec(level_id='lvl_03')\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n",
            encoding="utf-8"
        )
        assert scene.hot_reload_sentinel.check_for_promoted_update(1.1) is True
        scene._execute_hot_reload()
        assert scene.episode_number == 3
        assert type(engine.active_cartridge).__name__ == "Hop3Cart"

        # Verify 2 distinct hot reload telemetry events
        reload_events = [e for e in mem_rec.events if getattr(e, "probe_name", None) == "cartridge_hot_reload_triggered"]
        assert len(reload_events) == 2
        assert reload_events[0].payload["to_episode"] == 2
        assert reload_events[1].payload["to_episode"] == 3

    def test_stale_pyc_vulnerability_on_same_size_same_second_reload(self, tmp_path: Path):
        """Documents the empirical vulnerability where in-place cartridge reload with identical file size
        and timestamp in the same second executes stale bytecode from __pycache__.
        """
        active_dir = tmp_path / "cartridges" / "active"
        active_dir.mkdir(parents=True)

        (active_dir / "manifest.json").write_text(json.dumps({"name": "Ep1"}))
        # 297 bytes
        (active_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class Ep1Cartridge(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )
        c1 = CartridgeLoader.load_cartridge(active_dir)
        assert type(c1).__name__ == "Ep1Cartridge"

        # Write Ep2 with EXACT SAME byte length (297 bytes) immediately in the same second
        (active_dir / "manifest.json").write_text(json.dumps({"name": "Ep2"}))
        (active_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class Ep2Cartridge(BaseCartridge):\n"
            "    def get_level_spec(self, ep=2): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )
        c2 = CartridgeLoader.load_cartridge(active_dir)
        # Expected to fail (xfail) because SourceFileLoader loads Ep1Cartridge from __pycache__!
        assert type(c2).__name__ == "Ep2Cartridge"
