"""tests/adversarial/test_tier5_daemon_resilience.py
Tier 5 Adversarial Test Suite 4: Studio Daemon Lock Recovery & Launcher Signal Resilience.
"""
from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest

from engine.studio.daemon import StudioDaemon, StudioDaemonConfig
import config


def _cleanup_process(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.communicate(timeout=2)
        except Exception:
            pass


class TestTier5DaemonResilience:
    """Verifies lockfile orphan recovery, PID crash handling, and launcher signal interception."""

    def test_daemon_stale_lock_recovery_after_sigkill(self, tmp_path: Path):
        """Spawns a child process that writes its PID to studio.lock and is terminated with SIGKILL.
        Asserts that a subsequent StudioDaemon detects the dead PID, purges the stale lockfile,
        and successfully acquires the lock.
        """
        lock_path = tmp_path / "studio.lock"

        # Spawn a short-lived process that exits immediately, giving a dead PID
        child = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
        child.wait()
        dead_pid = child.pid

        # Write dead PID to lockfile (simulating sudden SIGKILL)
        lock_path.write_text(str(dead_pid), encoding="utf-8")

        daemon_cfg = StudioDaemonConfig(lock_file=lock_path, cartridges_root=tmp_path / "cartridges")
        daemon = StudioDaemon(config_obj=daemon_cfg)

        try:
            acquired = daemon.acquire_lock()
            assert acquired is True, "Daemon failed to recover stale lock from dead process"
            assert lock_path.exists(), "Lockfile should exist with new owner PID"
            current_owner = int(lock_path.read_text(encoding="utf-8").strip())
            assert current_owner == os.getpid(), f"Expected owner PID {os.getpid()}, got {current_owner}"
        finally:
            daemon.release_lock()
            assert not lock_path.exists()

    def test_daemon_corrupted_lockfile_recovery(self, tmp_path: Path):
        """Verifies recovery when studio.lock contains empty strings, non-numeric garbage, or negative PIDs."""
        lock_path = tmp_path / "studio.lock"
        daemon_cfg = StudioDaemonConfig(lock_file=lock_path, cartridges_root=tmp_path / "cartridges")

        corrupted_payloads = [
            "",
            "   \n",
            "NOT_A_VALID_PID\n",
            "-99999\n",
            "-1\n",
            "0\n",
            "\x00\x00\x00\x00corrupt",
        ]

        for payload in corrupted_payloads:
            lock_path.write_text(payload, encoding="utf-8", errors="replace")
            daemon = StudioDaemon(config_obj=daemon_cfg)
            try:
                acquired = daemon.acquire_lock()
                assert acquired is True, f"Failed to acquire lock with corrupted payload: {repr(payload)}"
                assert int(lock_path.read_text().strip()) == os.getpid()
            finally:
                daemon.release_lock()

    def test_daemon_active_pid_lock_rejection(self, tmp_path: Path):
        """Verifies that acquire_lock() refuses to steal the lock when owned by an active process."""
        lock_path = tmp_path / "studio.lock"

        # Write current process PID to simulate an active running studio daemon
        lock_path.write_text(str(os.getpid()), encoding="utf-8")

        daemon_cfg = StudioDaemonConfig(lock_file=lock_path, cartridges_root=tmp_path / "cartridges")
        daemon = StudioDaemon(config_obj=daemon_cfg)

        # Attempting to acquire when lock is owned by os.getpid() returns False
        acquired = daemon.acquire_lock()
        assert acquired is False, "Daemon should reject acquiring lock already owned by another instance"

        # Cleanup test fixture
        lock_path.unlink(missing_ok=True)

    def test_run_game_early_sigint_clean_wal_checkpoint(self, tmp_path: Path, sample_open_genre_cartridge):
        """Sends SIGINT to run_game.py during early startup and asserts clean shutdown with exit code 0."""
        cmd = [
            sys.executable,
            "run_game.py",
            "--headless",
            "--ticks", "5000",
            "--cartridge", str(sample_open_genre_cartridge),
        ]
        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(config.PROJECT_ROOT),
        )

        try:
            time.sleep(0.5)
            os.killpg(proc.pid, signal.SIGINT)
            stdout, stderr = proc.communicate(timeout=8)

            assert proc.returncode == 0, f"run_game.py direct SIGINT exit code {proc.returncode}: {stderr}"
            assert "completed cleanly" in stdout.lower() or "shutdown requested" in stdout.lower()
        finally:
            _cleanup_process(proc)

    def test_run_game_early_sigterm_clean_wal_checkpoint(self, tmp_path: Path, sample_open_genre_cartridge):
        """Sends SIGTERM to run_game.py during early startup and asserts clean shutdown with exit code 0."""
        cmd = [
            sys.executable,
            "run_game.py",
            "--headless",
            "--ticks", "5000",
            "--cartridge", str(sample_open_genre_cartridge),
        ]
        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(config.PROJECT_ROOT),
        )

        try:
            time.sleep(0.5)
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=8)

            assert proc.returncode == 0, f"run_game.py direct SIGTERM exit code {proc.returncode}: {stderr}"
            assert "completed cleanly" in stdout.lower() or "signal" in stdout.lower()
        finally:
            _cleanup_process(proc)
