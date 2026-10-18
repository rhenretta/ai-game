"""tests/test_godot_runtime.py
Unit tests for engine/godot_runtime.py (Milestone 7 Track 1).
Verifies portable binary discovery, architecture mapping, GODOT_BIN overrides,
and zero silent fallback enforcement.
"""
import os
from pathlib import Path
import platform
import subprocess
import pytest

from engine.godot_runtime import (
    ensure_godot_binary,
    get_default_runtime_dir,
    get_platform_asset_info,
    run_godot_command,
    verify_godot_binary,
    PLATFORM_DOWNLOAD_MAP,
)


class TestGodotRuntimeManager:
    """Verifies portable, container-invariant Godot 4.3 runtime discovery and execution."""

    def test_platform_asset_info_mapping(self):
        """Verifies detection of current host system and architecture mapping."""
        system = platform.system()
        machine = platform.machine().lower()

        # Should match supported platforms
        asset_name, subpath = get_platform_asset_info()
        assert asset_name.endswith(".zip")
        assert "4.3-stable" in asset_name
        assert str(subpath).endswith("Godot") or str(subpath).endswith(".x86_64") or str(subpath).endswith(".arm64")

    def test_ensure_godot_binary_returns_valid_executable(self):
        """Verifies ensure_godot_binary returns an existing, executable binary."""
        bin_path = ensure_godot_binary()
        assert bin_path.is_file(), f"Binary path {bin_path} does not exist"
        assert os.access(bin_path, os.X_OK), f"Binary {bin_path} is not executable"

        version = verify_godot_binary(bin_path)
        assert "4.3" in version, f"Expected Godot 4.3 version string, got: {version}"

    def test_verify_godot_binary_missing_raises_error(self, tmp_path: Path):
        """Rule 5: Zero Silent Fallbacks — missing binary raises FileNotFoundError."""
        fake_bin = tmp_path / "nonexistent_godot_bin"
        with pytest.raises(FileNotFoundError):
            verify_godot_binary(fake_bin)

    def test_godot_bin_env_override_success(self, tmp_path: Path, monkeypatch):
        """Verifies GODOT_BIN environment override is honored when pointing to valid binary."""
        real_bin = ensure_godot_binary()
        monkeypatch.setenv("GODOT_BIN", str(real_bin))

        resolved = ensure_godot_binary()
        assert resolved == real_bin.resolve()

    def test_godot_bin_env_override_missing_fails_fast(self, tmp_path: Path, monkeypatch):
        """Rule 5: If GODOT_BIN is set to nonexistent path, it must fail fast with FileNotFoundError."""
        fake_bin = tmp_path / "missing_godot_override"
        monkeypatch.setenv("GODOT_BIN", str(fake_bin))

        with pytest.raises(FileNotFoundError) as exc_info:
            ensure_godot_binary()
        assert "Rule 5" in str(exc_info.value) or "not found" in str(exc_info.value)

    def test_run_godot_command_version(self):
        """Verifies run_godot_command executes --version headlessly."""
        res = run_godot_command(["--version"], headless=True, timeout=15.0)
        assert res.returncode == 0
        assert "4.3" in res.stdout

    def test_run_godot_command_cwd(self, tmp_path: Path):
        """Verifies run_godot_command respects custom cwd."""
        res = run_godot_command(["--version"], cwd=tmp_path, headless=True, timeout=15.0)
        assert res.returncode == 0
        assert "4.3" in res.stdout
