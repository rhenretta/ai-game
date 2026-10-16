"""tests/test_run_studio.py
Comprehensive test suite verifying run_studio.py CLI, process isolation,
lockfile management, disk space checks, and StudioDaemon background loop (Feature 31).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import config
from engine.studio.daemon import StudioDaemon, StudioDaemonConfig
from engine.supervisor.models import BuildEvaluationReport
import run_studio


class TestStudioDaemonCLI:
    """Verifies CLI argument parsing and flags."""

    def test_default_cli_args_invariance(self):
        args = run_studio.parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 8000
        assert args.reload is False
        assert args.log_level == "info"
        assert args.interval == 5.0
        assert args.once is False
        assert args.no_daemon is False
        assert args.steam_profile == "default"
        assert args.cartridges_dir is None
        assert args.db_path is None

    def test_custom_cli_flags(self):
        args = run_studio.parse_args([
            "--port", "8888",
            "--host", "0.0.0.0",
            "--reload",
            "--log-level", "debug",
            "--interval", "2.5",
            "--once",
            "--no-daemon",
            "--steam-profile", "kinetic_rusher",
            "--cartridges-dir", "/tmp/custom_carts",
            "--db-path", "/tmp/telemetry.db",
        ])
        assert args.port == 8888
        assert args.host == "0.0.0.0"
        assert args.reload is True
        assert args.log_level == "debug"
        assert args.interval == 2.5
        assert args.once is True
        assert args.no_daemon is True
        assert args.steam_profile == "kinetic_rusher"
        assert args.cartridges_dir == "/tmp/custom_carts"
        assert args.db_path == "/tmp/telemetry.db"


class TestStudioDaemonLockfileAndDiskSpace:
    """Verifies lockfile safety and disk space guardrails."""

    def test_lockfile_creation_and_release(self, tmp_path: Path):
        lock_file = tmp_path / "studio.lock"
        cfg = StudioDaemonConfig(lock_file=lock_file, cartridges_root=tmp_path)
        daemon1 = StudioDaemon(config_obj=cfg)

        assert daemon1.acquire_lock() is True
        assert lock_file.exists()
        assert int(lock_file.read_text().strip()) == os.getpid()

        # Concurrency guard: second daemon attempt must fail
        daemon2 = StudioDaemon(config_obj=cfg)
        assert daemon2.acquire_lock() is False

        daemon1.release_lock()
        assert not lock_file.exists()

    def test_stale_lockfile_cleanup(self, tmp_path: Path):
        lock_file = tmp_path / "studio.lock"
        # Write dead PID that does not exist
        lock_file.write_text("99999999")
        cfg = StudioDaemonConfig(lock_file=lock_file, cartridges_root=tmp_path)
        daemon = StudioDaemon(config_obj=cfg)

        assert daemon.acquire_lock() is True
        assert int(lock_file.read_text().strip()) == os.getpid()
        daemon.release_lock()

    def test_release_lock_non_owner_does_not_delete_active_lockfile(self, tmp_path: Path):
        """Verifies calling release_lock() from a non-owning process does not delete active lockfile."""
        lock_file = tmp_path / "studio.lock"
        other_pid = os.getppid() if os.getppid() != os.getpid() else os.getpid() + 1000
        lock_file.write_text(str(other_pid))
        cfg = StudioDaemonConfig(lock_file=lock_file, cartridges_root=tmp_path)
        daemon = StudioDaemon(config_obj=cfg)

        assert daemon.acquire_lock() is False
        daemon.release_lock()

        assert lock_file.exists()
        assert lock_file.read_text().strip() == str(other_pid)
        assert daemon.is_running is False

    def test_acquire_lock_permission_error_does_not_steal_lock(self, tmp_path: Path):
        """Verifies acquire_lock() encountering PermissionError treats process as alive and returns False."""
        lock_file = tmp_path / "studio.lock"
        lock_file.write_text("1")
        cfg = StudioDaemonConfig(lock_file=lock_file, cartridges_root=tmp_path)
        daemon = StudioDaemon(config_obj=cfg)

        with patch("os.kill", side_effect=PermissionError):
            assert daemon.acquire_lock() is False
            assert lock_file.exists()
            assert lock_file.read_text().strip() == "1"

    def test_disk_space_check_threshold(self, tmp_path: Path):
        # Generous threshold passes
        cfg_pass = StudioDaemonConfig(min_disk_space_mb=1.0, cartridges_root=tmp_path)
        daemon_pass = StudioDaemon(config_obj=cfg_pass)
        assert daemon_pass.check_disk_space() is True

        # Impossibly large threshold fails
        cfg_fail = StudioDaemonConfig(min_disk_space_mb=999999999.0, cartridges_root=tmp_path)
        daemon_fail = StudioDaemon(config_obj=cfg_fail)
        assert daemon_fail.check_disk_space() is False


class TestColdBootGenesisAndPromotion:
    """Verifies autonomous genesis triggering on missing active cartridge."""

    def test_cold_boot_detected_when_active_empty(self, tmp_path: Path):
        active_dir = tmp_path / "active"
        cfg = StudioDaemonConfig(active_cartridge_dir=active_dir, cartridges_root=tmp_path)
        daemon = StudioDaemon(config_obj=cfg)
        assert daemon.is_cold_boot_required() is True

    def test_cold_boot_false_when_active_has_manifest_and_mechanics(self, tmp_path: Path):
        active_dir = tmp_path / "active"
        active_dir.mkdir(parents=True)
        (active_dir / "manifest.json").write_text('{"title": "Test"}')
        (active_dir / "mechanics.py").write_text('class Mechanics: pass')
        cfg = StudioDaemonConfig(active_cartridge_dir=active_dir, cartridges_root=tmp_path)
        daemon = StudioDaemon(config_obj=cfg)
        assert daemon.is_cold_boot_required() is False

    def test_genesis_cycle_promotes_cartridge(self, tmp_path: Path):
        cart_root = tmp_path / "cartridges"
        active_dir = cart_root / "active"
        seed_dir = cart_root / "seed"
        seed_dir.mkdir(parents=True)
        (seed_dir / "manifest.json").write_text('{"title": "Seed Arena", "episode": 0}')
        (seed_dir / "mechanics.py").write_text('class Mechanics: pass')

        cfg = StudioDaemonConfig(
            cartridges_root=cart_root,
            active_cartridge_dir=active_dir,
            lock_file=tmp_path / "studio.lock",
        )

        from engine.supervisor.models import SupervisorGateVerdict
        mock_supervisor = MagicMock()
        mock_supervisor.evaluate_and_heal.return_value = BuildEvaluationReport(
            deliberation="Verified genesis build passes all gates.",
            status=SupervisorGateVerdict.PROMOTED,
            promoted_to_active=True,
        )

        daemon = StudioDaemon(config_obj=cfg, supervisor=mock_supervisor)
        assert daemon.is_cold_boot_required() is True

        success = daemon.run_genesis_cycle()
        assert success is True
        assert mock_supervisor.evaluate_and_heal.called


class TestDecoupledWALPolling:
    """Verifies SQLite WAL polling without lock contention."""

    def test_wal_polling_with_finished_episode(self, tmp_path: Path):
        db_path = tmp_path / "telemetry_daemon_test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                episode_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                outcome TEXT,
                ended_at REAL
            );
        """)
        conn.execute(
            "INSERT INTO telemetry_episodes (session_id, episode_id, episode_number, status, outcome, ended_at) "
            "VALUES ('s1', 'ep1', 1, 'FINISHED', 'VICTORY', 100.0);"
        )
        conn.commit()
        conn.close()

        cfg = StudioDaemonConfig(
            db_path=db_path,
            cartridges_root=tmp_path / "cartridges",
            active_cartridge_dir=tmp_path / "cartridges" / "active",
            lock_file=tmp_path / "studio.lock",
        )

        from engine.supervisor.models import SupervisorGateVerdict
        mock_supervisor = MagicMock()
        mock_supervisor.evaluate_and_heal.return_value = BuildEvaluationReport(
            deliberation="Verified update passes all gates.",
            status=SupervisorGateVerdict.PROMOTED,
            promoted_to_active=True,
        )

        daemon = StudioDaemon(config_obj=cfg, supervisor=mock_supervisor)
        updated = daemon.poll_and_update()
        assert updated is True
        assert daemon.build_counter == 1


class TestGracefulDaemonLifecycle:
    """Verifies daemon run and stop lifecycle."""

    def test_daemon_single_step(self, tmp_path: Path):
        active_dir = tmp_path / "active"
        active_dir.mkdir(parents=True)
        (active_dir / "manifest.json").write_text('{"title": "Test"}')
        (active_dir / "mechanics.py").write_text('class Mechanics: pass')

        cfg = StudioDaemonConfig(
            cartridges_root=tmp_path,
            active_cartridge_dir=active_dir,
            lock_file=tmp_path / "studio.lock",
            run_once=True,
        )
        daemon = StudioDaemon(config_obj=cfg)
        daemon.step()
        assert daemon.is_cold_boot_required() is False

    def test_daemon_run_in_thread_and_stop(self, tmp_path: Path):
        active_dir = tmp_path / "active"
        active_dir.mkdir(parents=True)
        (active_dir / "manifest.json").write_text('{"title": "Test"}')
        (active_dir / "mechanics.py").write_text('class Mechanics: pass')

        lock_path = tmp_path / "studio.lock"
        cfg = StudioDaemonConfig(
            cartridges_root=tmp_path,
            active_cartridge_dir=active_dir,
            lock_file=lock_path,
            poll_interval=0.1,
        )
        daemon = StudioDaemon(config_obj=cfg)

        t = threading.Thread(target=daemon.run, daemon=True)
        t.start()

        # Wait for daemon to acquire lock
        for _ in range(20):
            if daemon.is_running and lock_path.exists():
                break
            time.sleep(0.05)

        assert daemon.is_running is True
        assert lock_path.exists()

        daemon.stop()
        t.join(timeout=2.0)

        assert daemon.is_running is False
        assert not lock_path.exists()
