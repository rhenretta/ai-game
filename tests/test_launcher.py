"""tests/test_launcher.py
Automated regression tests for Feature 19: Single-Command macOS Launcher (run_game.sh)
and Showrunner Runtime Bridge CLI enhancements.
Strictly adheres to Rationale-First Epistemic Architecture and Project De-Branding.
"""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

from config import PROJECT_ROOT
from engine.core.showrunner_bridge import ShowrunnerRuntimeBridge


class TestMacOSShellLauncher:
    """Verifies run_game.sh syntax, executable permissions, and POSIX signal handling."""
    deliberation: str = "Regression suite verifying run_game.sh POSIX compliance, permissions, and signal trapping."

    @pytest.fixture(autouse=True)
    def setup_paths(self):
        self.script_path = PROJECT_ROOT / "run_game.sh"

    def test_run_game_sh_exists_and_is_executable(self):
        """Asserts run_game.sh exists and possesses executable POSIX permissions."""
        assert self.script_path.exists(), f"Missing launcher script: {self.script_path}"
        assert os.access(self.script_path, os.X_OK), "run_game.sh is not marked executable (+x)"

    def test_run_game_sh_syntax_validation(self):
        """Executes bash -n run_game.sh to verify zero shell syntax errors."""
        res = subprocess.run(["bash", "-n", str(self.script_path)], capture_output=True, text=True)
        assert res.returncode == 0, f"Bash syntax check failed: {res.stderr}"

    def test_run_game_sh_shebang_and_safety_flags(self):
        """Verifies proper shebang and set -euo pipefail safety flags."""
        content = self.script_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        assert lines[0].startswith("#!/"), "run_game.sh must declare a valid shebang"
        assert "set -euo pipefail" in content, "run_game.sh must declare 'set -euo pipefail'"

    def test_run_game_sh_contains_venv_auto_detection(self):
        """Verifies venv auto-detection and activation logic."""
        content = self.script_path.read_text(encoding="utf-8")
        assert ".venv/bin/python3" in content or "VIRTUAL_ENV" in content
        assert "source" in content or "activate" in content

    def test_run_game_sh_contains_cocoa_driver_check(self):
        """Verifies Cocoa display driver environment configuration."""
        content = self.script_path.read_text(encoding="utf-8")
        assert "SDL_VIDEODRIVER" in content
        assert "cocoa" in content
        assert "dummy" in content

    def test_run_game_sh_contains_signal_trapping_and_wal_flush(self):
        """Verifies trap cleanup INT TERM and SQLite WAL checkpointing."""
        content = self.script_path.read_text(encoding="utf-8")
        assert "trap" in content
        assert "INT" in content and "TERM" in content
        assert "wal_checkpoint" in content
        assert "stty sane" in content


class TestPythonEntrypointCLI:
    """Verifies run_game.py CLI options: --cartridge, --headless, --showrunner, --steam-profile, --ticks."""
    deliberation: str = "Regression suite verifying run_game.py CLI argument parsing and discrete headless execution."

    def test_run_game_py_help_flag(self):
        """Asserts python run_game.py --help exits 0 and documents all required flags."""
        res = subprocess.run([sys.executable, "run_game.py", "--help"], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        assert res.returncode == 0, f"--help failed: {res.stderr}"
        stdout = res.stdout
        assert "--cartridge" in stdout
        assert "--headless" in stdout
        assert "--showrunner" in stdout
        assert "--steam-profile" in stdout
        assert "--ticks" in stdout

    def test_run_game_py_headless_60_ticks_clean_exit(self):
        """Verifies running 60 ticks in headless mode completes cleanly with exit code 0."""
        cmd = [sys.executable, "run_game.py", "--headless", "--ticks", "60"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=15)
        assert res.returncode == 0, f"Headless 60 ticks failed: {res.stderr}"
        assert "shutdown completed cleanly" in res.stdout.lower()

    def test_run_game_py_custom_cartridge_flag(self, tmp_path):
        """Verifies booting a specific cartridge path via --cartridge using a hermetic fixture."""
        cart_dir = tmp_path / "hermetic_test_cart"
        cart_dir.mkdir(parents=True)
        (cart_dir / "manifest.json").write_text(
            json.dumps({"name": "HermeticCartridge", "version": "1.0.0"}),
            encoding="utf-8"
        )
        (cart_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge, BaseLevelSpec\n"
            "class HermeticCart(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): return BaseLevelSpec()\n"
            "    def create_player(self, x, y, r=None): return None\n",
            encoding="utf-8"
        )
        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(cart_dir), "--ticks", "30"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=15)
        assert res.returncode == 0, f"Custom cartridge boot failed: {res.stderr}"
        assert cart_dir.name in res.stdout

    def test_run_game_py_no_active_cartridge_reports_guidance(self):
        """Verifies run_game.py prints guidance to run run_studio.py when no active cartridge exists."""
        cmd = [sys.executable, "run_game.py", "--headless"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=15)
        assert res.returncode == 0
        assert "No Active Game Cartridge Found" in res.stdout
        assert "run_studio.py" in res.stdout

    def test_run_game_py_steam_profile_flag(self):
        """Verifies passing --steam-profile tags session correctly."""
        cmd = [sys.executable, "run_game.py", "--headless", "--steam-profile", "rusher", "--ticks", "30"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=15)
        assert res.returncode == 0, f"Steam profile flag failed: {res.stderr}"
        assert "Steam Profile: rusher" in res.stdout

    def test_run_game_py_showrunner_flag(self):
        """Verifies passing --showrunner starts background bridge cleanly."""
        cmd = [sys.executable, "run_game.py", "--headless", "--showrunner", "--ticks", "30"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=15)
        assert res.returncode == 0, f"Showrunner flag failed: {res.stderr}"
        assert "Showrunner: ON" in res.stdout


class TestShowrunnerRuntimeBridge:
    """Verifies ShowrunnerRuntimeBridge implementation and Rationale-First compliance."""
    deliberation: str = "Unit test verifying ShowrunnerRuntimeBridge lifecycle and epistemic rationale ordering."

    def test_bridge_rationale_first_epistemic_architecture(self):
        """Verifies ShowrunnerRuntimeBridge declares deliberation as the very first property."""
        source_path = PROJECT_ROOT / "engine" / "core" / "showrunner_bridge.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ShowrunnerRuntimeBridge":
                first_stmt = node.body[0]
                if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                    # Docstring, check next statement
                    first_stmt = node.body[1]
                assert isinstance(first_stmt, ast.AnnAssign), "Expected annotated assignment for deliberation"
                assert first_stmt.target.id == "deliberation", f"First attribute must be 'deliberation', got '{first_stmt.target.id}'"
                return
        pytest.fail("ShowrunnerRuntimeBridge class not found in engine/core/showrunner_bridge.py")

    def test_bridge_lifecycle_and_status(self):
        """Verifies bridge initialization, start, status query, and shutdown."""
        class MockEngine:
            is_running = True
            def load_cartridge(self, path): pass

        bridge = ShowrunnerRuntimeBridge(engine=MockEngine())
        status = bridge.get_status()
        assert "deliberation" in status
        assert "IDLE" in status["status"]
        bridge.start()
        assert bridge.is_started is True
        bridge.shutdown()
        assert bridge.is_started is False


class TestLauncherDebrandingInvariant:
    """Verifies 0 occurrences of forbidden legacy tokens in launcher files."""
    deliberation: str = "Regression test asserting zero forbidden legacy branding in launcher files."

    def test_zero_forbidden_tokens(self):
        forbidden = "".join(chr(c) for c in (106, 111, 97, 110))
        target_files = [
            PROJECT_ROOT / "run_game.sh",
            PROJECT_ROOT / "run_game.py",
            PROJECT_ROOT / "engine" / "core" / "showrunner_bridge.py",
            PROJECT_ROOT / "tests" / "test_launcher.py",
        ]
        violations = []
        for file_path in target_files:
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for line_no, line in enumerate(content.splitlines(), start=1):
                if forbidden in line.lower():
                    violations.append(f"{file_path.name}:{line_no} -> {line.strip()}")

        assert len(violations) == 0, "Found legacy branding violations:\n" + "\n".join(violations)
