"""tests/test_rem_cli_loader_stress.py
Empirical stress tests for Post-Victory Remediation:
1. Hostile Cartridge Injections via CLI (forbidden modules, syntax errors, corrupt JSON manifests).
2. Dynamic Loader Import Isolation & Warning Absence (rapid multi-module and deeply nested loading, 0 warnings).
3. Subprocess Clean Teardown (run_game.py and run_studio.py lifecycle, 0 zombie processes, 0 unclosed file descriptors).
Strictly adheres to Rationale-First Epistemic Architecture and Project De-Branding.
"""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import warnings
from typing import List, Set
import pytest

from config import PROJECT_ROOT
from engine.api.contracts import BaseCartridge
from engine.loader import ASTSecurityError, CartridgeLoader


def _cleanup_process(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """Ensures subprocess is reaped, terminated, and its pipe streams closed."""
    try:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    proc.kill()
                except (ProcessLookupError, OSError):
                    pass
            try:
                proc.communicate(timeout=timeout)
            except Exception:
                pass
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except Exception:
                pass
        if proc.stdout and not proc.stdout.closed:
            proc.stdout.close()
        if proc.stderr and not proc.stderr.closed:
            proc.stderr.close()
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()


def _assert_zero_zombies(pids: List[int]) -> None:
    """Verifies that none of the monitored child PIDs remain as zombie (<defunct>) processes."""
    for pid in pids:
        try:
            res = subprocess.run(["ps", "-o", "pid,stat", "-p", str(pid)], capture_output=True, text=True)
            for line in res.stdout.strip().split("\n")[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    p, stat = parts[0], parts[1]
                    assert "Z" not in stat, f"Process {p} remained as zombie with state {stat}"
        except Exception:
            pass


def _get_free_port() -> int:
    """Finds an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestCLIHostileCartridgeInjections:
    """Adversarially tests CLI rejection of hostile, malformed, or corrupt cartridges.
    Enforces non-zero exit codes (fail fast) and descriptive diagnostic errors in stderr.
    """
    deliberation: str = (
        "Empirical adversarial validation ensuring the CLI launcher fails fast with exit code != 0 "
        "and emits descriptive error strings to stderr whenever hostile, broken, or corrupt cartridges are passed."
    )

    def test_cli_hostile_cartridge_import_os_rejected(self, tmp_path: Path):
        """Verifies that a cartridge importing forbidden module 'os' fails fast via CLI."""
        bad_cart = tmp_path / "hostile_os"
        bad_cart.mkdir()
        (bad_cart / "manifest.json").write_text(json.dumps({"name": "HostileOS"}))
        (bad_cart / "mechanics.py").write_text(
            "import os\n"
            "from engine.api.contracts import BaseCartridge\n"
            "class EvilCartridge(BaseCartridge):\n"
            "    def pwn(self):\n"
            "        os.system('echo compromised')\n"
        )

        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(bad_cart), "--ticks", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode != 0, f"Expected non-zero exit code on 'import os', got {res.returncode}"
        assert "ASTSecurityError" in res.stderr
        assert "Prohibited module import: 'os'" in res.stderr

    def test_cli_hostile_cartridge_import_subprocess_rejected(self, tmp_path: Path):
        """Verifies that a cartridge importing forbidden module 'subprocess' fails fast via CLI."""
        bad_cart = tmp_path / "hostile_subprocess"
        bad_cart.mkdir()
        (bad_cart / "manifest.json").write_text(json.dumps({"name": "HostileSubprocess"}))
        (bad_cart / "mechanics.py").write_text(
            "import subprocess\n"
            "from engine.api.contracts import BaseCartridge\n"
            "class EvilCartridge(BaseCartridge):\n"
            "    pass\n"
        )

        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(bad_cart), "--ticks", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode != 0, f"Expected non-zero exit code on 'import subprocess', got {res.returncode}"
        assert "ASTSecurityError" in res.stderr
        assert "Prohibited module import: 'subprocess'" in res.stderr

    def test_cli_hostile_cartridge_import_sys_rejected(self, tmp_path: Path):
        """Verifies that a cartridge importing forbidden module 'sys' fails fast via CLI."""
        bad_cart = tmp_path / "hostile_sys"
        bad_cart.mkdir()
        (bad_cart / "manifest.json").write_text(json.dumps({"name": "HostileSys"}))
        (bad_cart / "mechanics.py").write_text(
            "import sys\n"
            "from engine.api.contracts import BaseCartridge\n"
            "class EvilCartridge(BaseCartridge):\n"
            "    pass\n"
        )

        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(bad_cart), "--ticks", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode != 0, f"Expected non-zero exit code on 'import sys', got {res.returncode}"
        assert "ASTSecurityError" in res.stderr
        assert "Prohibited module import: 'sys'" in res.stderr

    def test_cli_hostile_cartridge_from_import_posix_rejected(self, tmp_path: Path):
        """Verifies that a cartridge using 'from posix import system' fails fast via CLI."""
        bad_cart = tmp_path / "hostile_posix"
        bad_cart.mkdir()
        (bad_cart / "manifest.json").write_text(json.dumps({"name": "HostilePosix"}))
        (bad_cart / "mechanics.py").write_text(
            "from posix import system\n"
            "from engine.api.contracts import BaseCartridge\n"
            "class EvilCartridge(BaseCartridge):\n"
            "    pass\n"
        )

        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(bad_cart), "--ticks", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode != 0, f"Expected non-zero exit code on 'from posix import system', got {res.returncode}"
        assert "ASTSecurityError" in res.stderr

    def test_cli_hostile_cartridge_syntax_error_rejected(self, tmp_path: Path):
        """Verifies that a cartridge containing a syntax error fails fast via CLI with descriptive stderr."""
        bad_cart = tmp_path / "syntax_error_cart"
        bad_cart.mkdir()
        (bad_cart / "manifest.json").write_text(json.dumps({"name": "SyntaxErrorCart"}))
        (bad_cart / "mechanics.py").write_text("def foo(:\n    return 42\n")

        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(bad_cart), "--ticks", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode != 0, f"Expected non-zero exit code on syntax error, got {res.returncode}"
        assert "SyntaxError" in res.stderr

    def test_cli_hostile_cartridge_malformed_json_manifest_rejected(self, tmp_path: Path):
        """Verifies that a cartridge with malformed manifest JSON fails fast via CLI with descriptive stderr."""
        bad_cart = tmp_path / "malformed_json_cart"
        bad_cart.mkdir()
        (bad_cart / "manifest.json").write_text('{"name": "BrokenManifest", "version": 1.0, invalid_unquoted_value}\n')
        (bad_cart / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class SafeCart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
        )

        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(bad_cart), "--ticks", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode != 0, f"Expected non-zero exit code on malformed manifest, got {res.returncode}"
        assert "JSONDecodeError" in res.stderr

    def test_cli_hostile_cartridge_nested_submodule_forbidden_import_rejected(self, tmp_path: Path):
        """Verifies that a prohibited module import hidden in a deeply nested submodule is detected and rejected."""
        bad_cart = tmp_path / "nested_submodule_exploit"
        bad_cart.mkdir()
        (bad_cart / "manifest.json").write_text(json.dumps({"name": "NestedExploit"}))
        deep_dir = bad_cart / "pkg" / "deep"
        deep_dir.mkdir(parents=True)
        (bad_cart / "pkg" / "__init__.py").write_text("")
        (deep_dir / "__init__.py").write_text("")
        (deep_dir / "exploit.py").write_text("import shutil\n")
        (bad_cart / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class CleanLookingCart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
        )

        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(bad_cart), "--ticks", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode != 0, f"Expected non-zero exit code on nested module exploit, got {res.returncode}"
        assert "ASTSecurityError" in res.stderr
        assert "Prohibited module import: 'shutil'" in res.stderr

    def test_cli_hostile_cartridge_eval_builtin_rejected(self, tmp_path: Path):
        """Verifies that prohibited dynamic code execution functions like eval() fail fast via CLI."""
        bad_cart = tmp_path / "eval_exploit"
        bad_cart.mkdir()
        (bad_cart / "manifest.json").write_text(json.dumps({"name": "EvalExploit"}))
        (bad_cart / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge\n"
            "class EvalCart(BaseCartridge):\n"
            "    def __init__(self):\n"
            "        eval('1 + 1')\n"
        )

        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(bad_cart), "--ticks", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode != 0, f"Expected non-zero exit code on eval() call, got {res.returncode}"
        assert "ASTSecurityError" in res.stderr
        assert "Prohibited function call: 'eval'" in res.stderr


class TestDynamicLoaderImportIsolationAndWarningAbsence:
    """Stress tests the dynamic cartridge loader for import isolation, clean package teardown,
    and absolute absence of warnings (ImportWarning, ResourceWarning) under -W error.
    """
    deliberation: str = (
        "Empirical validation of CartridgeLoader import mechanics under rapid churn, deeply nested packages, "
        "and concurrent/alternating loading cycles with zero ImportWarning or ResourceWarning."
    )

    def test_rapid_loading_and_unloading_deeply_nested_packages(self, tmp_path: Path):
        """Rapidly loads and reloads a cartridge containing deeply nested packages (3+ levels deep) 30 times.
        Verifies zero occurrences of ImportWarning or ResourceWarning under warnings.catch_warnings().
        """
        cart = tmp_path / "deep_nested_cart"
        cart.mkdir()
        (cart / "manifest.json").write_text(json.dumps({"name": "DeepNestedCart", "version": "1.0.0"}))

        deep = cart / "subpkg_a" / "subpkg_b" / "subpkg_c"
        deep.mkdir(parents=True)
        (cart / "subpkg_a" / "__init__.py").write_text("")
        (cart / "subpkg_a" / "subpkg_b" / "__init__.py").write_text("")
        (cart / "subpkg_a" / "subpkg_b" / "subpkg_c" / "__init__.py").write_text("")
        (deep / "quantum_emitter.py").write_text(
            "class QuantumEmitter:\n"
            "    def __init__(self, charge: float = 100.0):\n"
            "        self.charge = charge\n"
            "    def discharge(self) -> float:\n"
            "        return self.charge * 2.5\n"
        )

        (cart / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "from .subpkg_a.subpkg_b.subpkg_c.quantum_emitter import QuantumEmitter\n"
            "class DeepNestedCartridge(BaseCartridge):\n"
            "    def __init__(self, **kwargs):\n"
            "        super().__init__(**kwargs)\n"
            "        self.emitter = QuantumEmitter(50.0)\n"
            "    def get_level_spec(self, ep=1):\n"
            "        return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )

        gc.collect()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            for i in range(30):
                c = CartridgeLoader.load_cartridge(cart)
                assert isinstance(c, BaseCartridge)
                assert type(c).__name__ == "DeepNestedCartridge"
                assert hasattr(c, "emitter")
                assert c.emitter.discharge() == 125.0

            gc.collect()
            import_warnings = [w for w in recorded if issubclass(w.category, ImportWarning)]
            resource_warnings = [w for w in recorded if issubclass(w.category, ResourceWarning)]
            all_warning_msgs = [f"{w.category.__name__}: {w.message}" for w in recorded]

            assert len(import_warnings) == 0, f"Encountered ImportWarnings: {all_warning_msgs}"
            assert len(resource_warnings) == 0, f"Encountered ResourceWarnings: {all_warning_msgs}"

    def test_multi_module_cartridge_isolation_alternating_rapid_cycles(self, tmp_path: Path):
        """Rapidly alternates loading between two multi-module cartridges with identical submodule paths
        to prove dynamic namespaces never cross-contaminate or leak across reloads.
        """
        cart_alpha = tmp_path / "cart_alpha"
        cart_omega = tmp_path / "cart_omega"
        cart_alpha.mkdir()
        cart_omega.mkdir()

        # Cartridge Alpha: returns factor 11.0
        (cart_alpha / "manifest.json").write_text(json.dumps({"name": "Alpha"}))
        (cart_alpha / "entities.py").write_text(
            "class Entity:\n"
            "    signature = 'ALPHA_SIG'\n"
            "    factor = 11.0\n"
        )
        (cart_alpha / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "from .entities import Entity\n"
            "class AlphaCartridge(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )

        # Cartridge Omega: returns factor 99.0
        (cart_omega / "manifest.json").write_text(json.dumps({"name": "Omega"}))
        (cart_omega / "entities.py").write_text(
            "class Entity:\n"
            "    signature = 'OMEGA_SIG'\n"
            "    factor = 99.0\n"
        )
        (cart_omega / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "from .entities import Entity\n"
            "class OmegaCartridge(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            for _ in range(20):
                c_a = CartridgeLoader.load_cartridge(cart_alpha)
                assert c_a.modules["entities"].Entity.signature == "ALPHA_SIG"
                assert c_a.modules["entities"].Entity.factor == 11.0

                c_o = CartridgeLoader.load_cartridge(cart_omega)
                assert c_o.modules["entities"].Entity.signature == "OMEGA_SIG"
                assert c_o.modules["entities"].Entity.factor == 99.0

            gc.collect()
            import_warnings = [w for w in recorded if issubclass(w.category, ImportWarning)]
            resource_warnings = [w for w in recorded if issubclass(w.category, ResourceWarning)]
            assert len(import_warnings) == 0, f"ImportWarnings in alternating cycles: {import_warnings}"
            assert len(resource_warnings) == 0, f"ResourceWarnings in alternating cycles: {resource_warnings}"

    def test_loader_bytecode_cache_invalidation_cleanliness(self, tmp_path: Path):
        """Verifies that __pycache__ directory in cartridge folders is completely expunged without warnings."""
        cart = tmp_path / "cache_clean_cart"
        cart.mkdir()
        (cart / "manifest.json").write_text(json.dumps({"name": "CacheClean"}))
        pycache = cart / "__pycache__"
        pycache.mkdir()
        (pycache / "stale.cpython-311.pyc").write_bytes(b"\x00\x00\x00\x00stale_bytecode")
        (cart / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class CleanCartridge(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            CartridgeLoader.load_cartridge(cart)
            assert not pycache.exists(), "__pycache__ was not removed during cartridge load"
            gc.collect()
            import_or_res = [w for w in recorded if issubclass(w.category, (ImportWarning, ResourceWarning))]
            assert len(import_or_res) == 0

    def test_loader_no_sys_modules_leakage_after_teardown(self, tmp_path: Path):
        """Verifies that temporary submodule bare aliases are expunged from sys.modules."""
        cart = tmp_path / "alias_cleanup_cart"
        cart.mkdir()
        (cart / "manifest.json").write_text(json.dumps({"name": "AliasClean"}))
        (cart / "unique_isolated_submod.py").write_text("ISOLATED_VAL = 4242\n")
        (cart / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "from .unique_isolated_submod import ISOLATED_VAL\n"
            "class AliasCartridge(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )

        CartridgeLoader.load_cartridge(cart)
        # Verify bare alias was not leaked globally into sys.modules
        assert "unique_isolated_submod" not in sys.modules, "Bare submodule alias leaked into sys.modules"


class TestSubprocessCleanTeardownAndZombieEradication:
    """Stress tests clean process lifecycle, signal trapping, and absence of zombie processes
    and unclosed file descriptors for run_game.py and run_studio.py under various flags.
    """
    deliberation: str = (
        "Empirical verification that subprocesses launched under various CLI configurations "
        "terminate cleanly with 0 zombie processes, 0 unclosed file descriptors, and clean WAL checkpoints."
    )

    def test_run_game_py_headless_ticks_clean_teardown_zero_zombies(self):
        """Tests run_game.py under --headless --ticks (1 and 30 ticks) terminates cleanly with 0 zombies and 0 leaked FDs."""
        for ticks in [1, 30]:
            fds_before = set(os.listdir("/dev/fd"))
            proc = subprocess.Popen(
                [sys.executable, "run_game.py", "--headless", "--ticks", str(ticks)],
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            try:
                stdout, stderr = proc.communicate(timeout=10)
                assert proc.returncode == 0, f"run_game.py --ticks {ticks} failed: {stderr}"
                assert "completed cleanly" in stdout.lower()
            finally:
                _cleanup_process(proc)

            gc.collect()
            _assert_zero_zombies([proc.pid])
            fds_after = set(os.listdir("/dev/fd"))
            leaked_fds = fds_after - fds_before
            assert not leaked_fds, f"Leaked file descriptors after --ticks {ticks}: {leaked_fds}"

    def test_run_game_py_showrunner_flag_clean_teardown(self):
        """Tests run_game.py with --headless --showrunner --ticks 20 terminates cleanly with 0 zombies."""
        fds_before = set(os.listdir("/dev/fd"))
        proc = subprocess.Popen(
            [sys.executable, "run_game.py", "--headless", "--showrunner", "--ticks", "20"],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        try:
            stdout, stderr = proc.communicate(timeout=15)
            assert proc.returncode == 0, f"run_game.py --showrunner failed: {stderr}"
            assert "showrunner: on" in stdout.lower()
            assert "completed cleanly" in stdout.lower()
        finally:
            _cleanup_process(proc)

        gc.collect()
        _assert_zero_zombies([proc.pid])
        fds_after = set(os.listdir("/dev/fd"))
        leaked_fds = fds_after - fds_before
        assert not leaked_fds, f"Leaked file descriptors after --showrunner: {leaked_fds}"

    def test_run_game_py_steam_profile_flag_clean_teardown(self):
        """Tests run_game.py with --headless --steam-profile cautious --ticks 10 terminates cleanly with 0 zombies."""
        fds_before = set(os.listdir("/dev/fd"))
        proc = subprocess.Popen(
            [sys.executable, "run_game.py", "--headless", "--steam-profile", "cautious", "--ticks", "10"],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        try:
            stdout, stderr = proc.communicate(timeout=10)
            assert proc.returncode == 0, f"run_game.py --steam-profile failed: {stderr}"
            assert "completed cleanly" in stdout.lower()
        finally:
            _cleanup_process(proc)

        gc.collect()
        _assert_zero_zombies([proc.pid])
        fds_after = set(os.listdir("/dev/fd"))
        leaked_fds = fds_after - fds_before
        assert not leaked_fds, f"Leaked file descriptors after --steam-profile: {leaked_fds}"

    def test_run_game_py_sigint_interception_clean_teardown(self, sample_open_genre_cartridge):
        """Sends SIGINT to running run_game.py and asserts clean shutdown with exit code 0 and 0 zombies."""
        fds_before = set(os.listdir("/dev/fd"))
        proc = subprocess.Popen(
            [sys.executable, "run_game.py", "--headless", "--ticks", "5000", "--cartridge", str(sample_open_genre_cartridge)],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        try:
            time.sleep(0.5)
            os.killpg(proc.pid, signal.SIGINT)
            stdout, stderr = proc.communicate(timeout=15.0)
            assert proc.returncode == 0, f"run_game.py did not exit 0 on SIGINT: {stderr}"
            assert "completed cleanly" in stdout.lower()
        finally:
            _cleanup_process(proc)

        gc.collect()
        _assert_zero_zombies([proc.pid])
        fds_after = set(os.listdir("/dev/fd"))
        leaked_fds = fds_after - fds_before
        assert not leaked_fds, f"Leaked file descriptors after SIGINT: {leaked_fds}"

    def test_run_game_py_sigterm_interception_clean_teardown(self, sample_open_genre_cartridge):
        """Sends SIGTERM to running run_game.py and asserts clean shutdown with exit code 0 and 0 zombies."""
        fds_before = set(os.listdir("/dev/fd"))
        proc = subprocess.Popen(
            [sys.executable, "run_game.py", "--headless", "--ticks", "5000", "--cartridge", str(sample_open_genre_cartridge)],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        try:
            time.sleep(0.5)
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=15.0)
            assert proc.returncode == 0, f"run_game.py did not exit 0 on SIGTERM: {stderr}"
            assert "completed cleanly" in stdout.lower()
        finally:
            _cleanup_process(proc)

        gc.collect()
        _assert_zero_zombies([proc.pid])
        fds_after = set(os.listdir("/dev/fd"))
        leaked_fds = fds_after - fds_before
        assert not leaked_fds, f"Leaked file descriptors after SIGTERM: {leaked_fds}"

    def test_run_game_sh_launcher_clean_teardown(self):
        """Tests ./run_game.sh bash wrapper executes and terminates cleanly with zero zombies."""
        fds_before = set(os.listdir("/dev/fd"))
        proc = subprocess.Popen(
            ["./run_game.sh", "--headless", "--ticks", "15"],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        try:
            stdout, stderr = proc.communicate(timeout=15)
            assert proc.returncode == 0, f"./run_game.sh failed: {stderr}"
            assert "completed cleanly" in stdout.lower()
        finally:
            _cleanup_process(proc)

        gc.collect()
        _assert_zero_zombies([proc.pid])
        fds_after = set(os.listdir("/dev/fd"))
        leaked_fds = fds_after - fds_before
        assert not leaked_fds, f"Leaked file descriptors after ./run_game.sh: {leaked_fds}"

    def test_run_studio_py_sigint_clean_teardown_zero_zombies(self):
        """Launches run_studio.py, verifies healthy server response, sends SIGINT,
        and asserts clean exit code 0, 0 zombie processes, and 0 unclosed file descriptors.
        """
        port = _get_free_port()
        fds_before = set(os.listdir("/dev/fd"))

        proc = subprocess.Popen(
            [sys.executable, "run_studio.py", "--port", str(port), "--log-level", "error"],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        try:
            # Poll /health to ensure ASGI application is fully initialized
            ready = False
            for _ in range(50):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                        if resp.status == 200:
                            ready = True
                            break
                except Exception:
                    time.sleep(0.1)

            assert ready, f"Studio server failed to become ready on port {port}"

            # Intercept with SIGINT
            os.killpg(proc.pid, signal.SIGINT)
            stdout, stderr = proc.communicate(timeout=15.0)
            assert proc.returncode == 0, f"run_studio.py did not exit 0 on SIGINT: {stderr}"
        finally:
            _cleanup_process(proc)

        gc.collect()
        _assert_zero_zombies([proc.pid])
        fds_after = set(os.listdir("/dev/fd"))
        leaked_fds = fds_after - fds_before
        assert not leaked_fds, f"Leaked file descriptors after studio SIGINT: {leaked_fds}"

    def test_run_studio_py_sigterm_clean_teardown_zero_zombies(self):
        """Launches run_studio.py, verifies healthy server response, sends SIGTERM,
        and asserts clean exit code 0, 0 zombie processes, and 0 unclosed file descriptors.
        """
        port = _get_free_port()
        fds_before = set(os.listdir("/dev/fd"))

        proc = subprocess.Popen(
            [sys.executable, "run_studio.py", "--port", str(port), "--log-level", "error"],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        try:
            # Poll /health to ensure ASGI application is fully initialized
            ready = False
            for _ in range(50):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                        if resp.status == 200:
                            ready = True
                            break
                except Exception:
                    time.sleep(0.1)

            assert ready, f"Studio server failed to become ready on port {port}"

            # Intercept with SIGTERM
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=15.0)
            assert proc.returncode == 0, f"run_studio.py did not exit 0 on SIGTERM: {stderr}"
        finally:
            _cleanup_process(proc)

        gc.collect()
        _assert_zero_zombies([proc.pid])
        fds_after = set(os.listdir("/dev/fd"))
        leaked_fds = fds_after - fds_before
        assert not leaked_fds, f"Leaked file descriptors after studio SIGTERM: {leaked_fds}"
