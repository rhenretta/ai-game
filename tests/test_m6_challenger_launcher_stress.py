"""tests/test_m6_challenger_launcher_stress.py
Empirical stress tests for Feature 19: Single-Command macOS Launcher (run_game.sh)
and Showrunner Runtime Bridge (engine/core/showrunner_bridge.py).
Strictly adheres to Rationale-First Epistemic Architecture and Project De-Branding.
"""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import random
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock
import pytest

from config import PROJECT_ROOT, DB_PATH
from engine.core.showrunner_bridge import ShowrunnerRuntimeBridge
from engine.loader import ASTSecurityError, CartridgeLoader
from engine.scheduler.scheduler import Priority, PriorityScheduler
from engine.telemetry.recorder import TelemetryRecorder


def _cleanup_process(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """Ensures subprocess is terminated, waited, and its pipe streams closed."""
    try:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                proc.communicate(timeout=timeout)
            except Exception:
                pass
    finally:
        if proc.stdout and not proc.stdout.closed:
            proc.stdout.close()
        if proc.stderr and not proc.stderr.closed:
            proc.stderr.close()


class TestLauncherProcessLifecycleAndSignalStress:
    """Stress suite testing run_game.sh and run_game.py process lifecycle,
    Cocoa/dummy display drivers, POSIX signal interception (SIGINT, SIGTERM),
    and child process termination with zero zombie/orphan processes.
    """
    deliberation: str = "Empirical stress tests verifying shell launcher execution, signal trapping, and zero zombie processes."

    def test_launcher_headless_execution_under_dummy_display(self):
        """Verifies run_game.sh executes cleanly under headless dummy display and audio drivers."""
        env = dict(os.environ)
        env["SDL_VIDEODRIVER"] = "dummy"
        env["SDL_AUDIODRIVER"] = "dummy"

        cmd = ["./run_game.sh", "--headless", "--ticks", "30"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env, timeout=15)
        assert res.returncode == 0, f"Headless execution failed: {res.stderr}"
        assert "completed cleanly" in res.stdout.lower()

    def test_launcher_sigint_signal_trapping_zero_zombies(self, sample_open_genre_cartridge):
        """Sends SIGINT (Ctrl+C) to run_game.sh process group and asserts clean child termination with 0 zombies."""
        proc = subprocess.Popen(
            ["./run_game.sh", "--headless", "--ticks", "5000", "--cartridge", str(sample_open_genre_cartridge)],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        try:
            time.sleep(0.5)

            # Discover child process PIDs
            pgrep_res = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True)
            child_pids = [int(p) for p in pgrep_res.stdout.strip().split() if p.isdigit()]
            assert len(child_pids) > 0, "No child processes spawned by run_game.sh"

            # Dispatch SIGINT to the process group
            os.killpg(proc.pid, signal.SIGINT)
            stdout, stderr = proc.communicate(timeout=8)

            assert proc.returncode == 0, f"run_game.sh did not exit 0 on SIGINT: {stderr}"
            time.sleep(0.3)

            # Assert all child processes are terminated (zero zombies or orphaned processes)
            for cpid in child_pids:
                try:
                    os.kill(cpid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                assert not alive, f"Child process {cpid} remained alive or orphaned after SIGINT"
        finally:
            _cleanup_process(proc)

    def test_launcher_sigterm_signal_trapping_zero_zombies(self, sample_open_genre_cartridge):
        """Sends SIGTERM to run_game.sh process group and asserts clean child termination with 0 zombies."""
        proc = subprocess.Popen(
            ["./run_game.sh", "--headless", "--ticks", "5000", "--cartridge", str(sample_open_genre_cartridge)],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        try:
            time.sleep(0.5)

            # Discover child process PIDs
            pgrep_res = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True)
            child_pids = [int(p) for p in pgrep_res.stdout.strip().split() if p.isdigit()]
            assert len(child_pids) > 0, "No child processes spawned by run_game.sh"

            # Dispatch SIGTERM to the process group
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=8)

            assert proc.returncode == 0, f"run_game.sh did not exit 0 on SIGTERM: {stderr}"
            time.sleep(0.3)

            # Assert all child processes are dead
            for cpid in child_pids:
                try:
                    os.kill(cpid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                assert not alive, f"Child process {cpid} remained alive or orphaned after SIGTERM"
        finally:
            _cleanup_process(proc)

    def test_run_game_py_direct_sigint_clean_exit(self, sample_open_genre_cartridge):
        """Sends SIGINT directly to run_game.py and asserts clean shutdown with exit code 0."""
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
            stdout, stderr = proc.communicate(timeout=8)

            assert proc.returncode == 0, f"run_game.py direct SIGINT exit code {proc.returncode}: {stderr}"
            assert "completed cleanly" in stdout.lower()
        finally:
            _cleanup_process(proc)

    def test_run_game_py_direct_sigterm_clean_exit(self, sample_open_genre_cartridge):
        """Sends SIGTERM directly to run_game.py and asserts clean shutdown with exit code 0."""
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
            stdout, stderr = proc.communicate(timeout=8)

            assert proc.returncode == 0, f"run_game.py direct SIGTERM exit code {proc.returncode}: {stderr}"
            assert "completed cleanly" in stdout.lower()
        finally:
            _cleanup_process(proc)


class TestSQLiteWALCheckpointDurabilityStress:
    """Stress suite verifying continuous SQLite WAL mode integrity,
    concurrent writer/reader durability, and atomic PRAGMA wal_checkpoint(TRUNCATE)
    on signal shutdown without database corruption.
    """
    deliberation: str = "Empirical stress tests verifying SQLite WAL mode durability and TRUNCATE checkpointing."

    def test_wal_checkpoint_truncate_on_clean_exit(self, tmp_path: Path):
        """Verifies that TelemetryRecorder.close() executes PRAGMA wal_checkpoint(TRUNCATE)
        and flushes all queued events to disk with zero data loss.
        """
        db_file = tmp_path / "stress_telemetry.db"
        recorder = TelemetryRecorder(db_path=db_file, batch_size=50, flush_interval_sec=0.05)
        recorder.start_session(player_profile_tag="wal_stress_profile")
        recorder.start_episode(level_id="lvl_wal_stress", episode_index=1)

        # Emit 1,000 events into WAL
        for i in range(1000):
            recorder.emit_event(
                event_type="MOVEMENT_TICK",
                pos=(float(i % 1280), float(i % 720)),
                vel=(10.0, 5.0),
                health=100.0,
                payload={"frame": i, "stress": True},
            )

        recorder.end_episode(outcome="CLEARED")
        recorder.end_session()
        recorder.close()

        # Verify integrity of database
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        res = cursor.fetchone()
        assert res[0] == "ok", f"Integrity check failed: {res}"

        # Verify all 1,000 events are persisted
        cursor.execute("SELECT count(*) FROM telemetry_events;")
        event_count = cursor.fetchone()[0]
        assert event_count == 1000, f"Expected 1000 events in DB, found {event_count}"

        # Verify WAL file is truncated to 0 bytes or removed
        wal_file = db_file.parent / (db_file.name + "-wal")
        if wal_file.exists():
            assert wal_file.stat().st_size == 0, f"WAL file was not truncated: size is {wal_file.stat().st_size}"

        conn.close()

    def test_wal_checkpoint_truncate_on_signal_interception(self, sample_open_genre_cartridge):
        """Verifies that SIGINT signal interception triggers clean PRAGMA wal_checkpoint(TRUNCATE)
        without database corruption on the authoritative telemetry store.
        """
        proc = subprocess.Popen(
            [sys.executable, "run_game.py", "--headless", "--ticks", "5000", "--cartridge", str(sample_open_genre_cartridge)],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        try:
            time.sleep(0.6)
            os.killpg(proc.pid, signal.SIGINT)
            proc.communicate(timeout=8)

            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            row = cursor.fetchone()
            assert row[0] == "ok", f"Integrity check failed: {row}"

            wal_path = DB_PATH.parent / (DB_PATH.name + "-wal")
            if wal_path.exists():
                assert wal_path.stat().st_size == 0, f"WAL file not truncated: size is {wal_path.stat().st_size}"

            conn.close()
        finally:
            _cleanup_process(proc)

    def test_concurrent_reader_during_wal_checkpoint_truncate(self, tmp_path: Path):
        """Stress tests continuous concurrent reader queries while PRAGMA wal_checkpoint(TRUNCATE)
        is executed mid-stream. Verifies zero database locked errors.
        """
        db_file = tmp_path / "concurrent_wal.db"
        recorder = TelemetryRecorder(db_path=db_file, batch_size=25, flush_interval_sec=0.02)
        recorder.start_session(player_profile_tag="concurrent_wal")
        recorder.start_episode(level_id="lvl_concurrent", episode_index=1)

        reader_errors: List[str] = []
        queries_completed = [0]
        stop_readers = threading.Event()

        def reader_worker():
            conn = recorder.db.get_reader_connection()
            cur = conn.cursor()
            while not stop_readers.is_set():
                try:
                    cur.execute("SELECT count(*) FROM telemetry_events;")
                    cur.fetchone()
                    queries_completed[0] += 1
                except Exception as exc:
                    reader_errors.append(str(exc))
                time.sleep(0.001)
            conn.close()

        threads = [threading.Thread(target=reader_worker, daemon=True) for _ in range(3)]
        for t in threads:
            t.start()

        for i in range(500):
            recorder.emit_event(
                event_type="MOVEMENT_TICK",
                pos=(100.0, 100.0),
                vel=(0.0, 0.0),
                health=100.0,
            )
            if i % 100 == 0:
                recorder.flush()
                recorder.db.checkpoint("TRUNCATE")

        stop_readers.set()
        for t in threads:
            t.join(timeout=3.0)

        recorder.close()
        assert len(reader_errors) == 0, f"Reader encountered SQLite errors: {reader_errors}"
        assert queries_completed[0] > 0, "No reader queries executed"


class TestShowrunnerBridgeConcurrencyStress:
    """Concurrency stress testing ShowrunnerRuntimeBridge:
    rapid successive start/shutdown cycles, multi-threaded race conditions,
    exception resilience across pipeline stages, and non-blocking Thread 0 gameplay isolation.
    """
    deliberation: str = "Concurrency stress tests verifying ShowrunnerRuntimeBridge lifecycle, race conditions, and thread safety."

    def test_bridge_rapid_start_shutdown_sequential_cycles(self):
        """Verifies that 25 rapid successive start() and shutdown() cycles cleanly terminate workers without thread leaks."""
        class MockEngine:
            is_running = True
            def load_cartridge(self, path): pass

        bridge = ShowrunnerRuntimeBridge(engine=MockEngine())
        for _ in range(25):
            bridge.start()
            assert bridge.is_started is True
            bridge.shutdown()
            assert bridge.is_started is False

        status = bridge.get_status()
        assert status["status"] == "OFFLINE"
        assert status["is_started"] is False

    def test_bridge_multithreaded_start_shutdown_race(self):
        """Stress-tests multi-threaded race conditions by concurrently starting, shutting down,
        and querying status across 6 threads without deadlocks or unhandled exceptions.
        """
        class MockEngine:
            is_running = True
            def load_cartridge(self, path): pass

        bridge = ShowrunnerRuntimeBridge(engine=MockEngine())
        errors: List[Exception] = []

        def worker_loop():
            for _ in range(20):
                try:
                    if random.random() < 0.5:
                        bridge.start()
                    else:
                        bridge.shutdown()
                    _ = bridge.get_status()
                    time.sleep(0.001)
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=worker_loop) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        bridge.shutdown()
        assert len(errors) == 0, f"Encountered thread race errors: {errors}"
        assert bridge.is_started is False

    def test_bridge_exception_resilience_pipeline_failure(self):
        """Verifies that simulated crashes in TelemetryAnalyst, EpisodicDirector, or BuildSupervisor
        do NOT terminate background workers or corrupt bridge state, and that subsequent valid tasks succeed.
        """
        class MockEngine:
            is_running = True
            def load_cartridge(self, path): pass

        bridge = ShowrunnerRuntimeBridge(engine=MockEngine())
        bridge.analyst.analyze_session = MagicMock(side_effect=RuntimeError("Simulated analyst defect"))
        bridge.start()

        # Dispatch failing task
        bridge.on_episode_finished("sess_fail", "ep_fail", 1, "CLEARED")
        time.sleep(0.3)

        status = bridge.get_status()
        assert "ERROR: Simulated analyst defect" in status["status"]

        # Assert worker thread survived and recovers on subsequent valid task
        mock_spec = MagicMock()
        mock_spec.to_markdown.return_value = "markdown content"
        mock_spec.spec_id = "spec_recovery_002"
        mock_spec.target_episode_number = 2

        bridge.analyst.analyze_session = MagicMock(return_value=MagicMock())
        bridge.director.evaluate_and_update = MagicMock(return_value=mock_spec)
        bridge.director.write_update_spec = MagicMock()
        bridge.supervisor.evaluate_and_heal = MagicMock(return_value=MagicMock(is_promoted=True))

        bridge.on_episode_finished("sess_ok", "ep_ok", 2, "CLEARED")
        time.sleep(0.3)

        status_after = bridge.get_status()
        assert "PROMOTED" in status_after["status"]
        bridge.shutdown()

    def test_p2_showrunner_does_not_starve_p0_thread_zero(self):
        """Verifies that background P2 tasks do NOT monopolize CPU or starve P0 Cocoa main thread tasks."""
        scheduler = PriorityScheduler(max_background_workers=2)
        scheduler.start()

        p2_started = threading.Event()
        p2_finished = threading.Event()

        def heavy_p2_task():
            p2_started.set()
            for _ in range(50):
                scheduler.checkpoint_and_yield(timeout_sec=0.01)
                time.sleep(0.002)
            p2_finished.set()

        scheduler.submit(Priority.P2_SHOWRUNNER, "heavy_p2", heavy_p2_task)
        assert p2_started.wait(timeout=1.0), "P2 task failed to start"

        # Main Thread 0 submits and drains 10 P0 tasks
        p0_executed: List[int] = []
        for i in range(10):
            scheduler.submit(Priority.P0_GAMEPLAY, f"p0_task_{i}", lambda idx=i: p0_executed.append(idx))

        t0 = time.perf_counter()
        executed_count = scheduler.step_main_thread(budget_ms=10.0)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert executed_count == 10, f"Expected 10 P0 tasks drained, got {executed_count}"
        assert len(p0_executed) == 10
        assert t_elapsed_ms < 10.0, f"Thread 0 starved by P2: took {t_elapsed_ms:.2f}ms"

        scheduler.shutdown()

    def test_p2_task_cooperative_yield_on_active_gameplay(self):
        """Verifies that a running P2 task cooperatively yields when gameplay is marked active."""
        scheduler = PriorityScheduler(max_background_workers=2)
        scheduler.start()

        task_started = threading.Event()
        yield_occurred = threading.Event()

        def yielding_p2():
            task_started.set()
            for _ in range(100):
                did_yield = scheduler.checkpoint_and_yield(timeout_sec=0.01)
                if did_yield:
                    yield_occurred.set()
                    break
                time.sleep(0.005)

        scheduler.submit(Priority.P2_SHOWRUNNER, "yielding_p2", yielding_p2)
        assert task_started.wait(timeout=1.0), "P2 task failed to start"

        # Mark gameplay active
        scheduler.set_gameplay_active(True)
        assert yield_occurred.wait(timeout=2.0), "P2 task failed to yield upon active gameplay"

        scheduler.set_gameplay_active(False)
        scheduler.shutdown()


class TestCLIArgumentFuzzingAndRobustness:
    """CLI argument parser fuzzing and boundary stress testing:
    boundary tick values (1, 120, 0), headless showrunner execution,
    and rejection of invalid/corrupt cartridges.
    """
    deliberation: str = "Empirical test suite fuzzing CLI arguments, tick boundaries, and invalid cartridge handling."

    def test_cli_ticks_boundary_minimum_one(self):
        """Verifies executing exactly 1 tick in headless mode completes cleanly."""
        cmd = [sys.executable, "run_game.py", "--headless", "--ticks", "1"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode == 0, f"--ticks 1 failed: {res.stderr}"
        assert "completed cleanly" in res.stdout.lower()

    def test_cli_ticks_boundary_large_120(self):
        """Verifies executing 120 ticks in headless mode completes cleanly without stalls."""
        cmd = [sys.executable, "run_game.py", "--headless", "--ticks", "120"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=15)
        assert res.returncode == 0, f"--ticks 120 failed: {res.stderr}"
        assert "completed cleanly" in res.stdout.lower()

    def test_cli_ticks_boundary_zero(self):
        """Verifies executing 0 ticks in headless mode completes cleanly immediately."""
        cmd = [sys.executable, "run_game.py", "--headless", "--ticks", "0"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode == 0, f"--ticks 0 failed: {res.stderr}"
        assert "completed cleanly" in res.stdout.lower()

    def test_cli_headless_with_showrunner_flag(self):
        """Verifies running headless with --showrunner flag initializes bridge and exits cleanly."""
        cmd = [sys.executable, "run_game.py", "--headless", "--showrunner", "--ticks", "30"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=15)
        assert res.returncode == 0, f"--showrunner execution failed: {res.stderr}"
        assert "showrunner: on" in res.stdout.lower()
        assert "completed cleanly" in res.stdout.lower()

    def test_cli_invalid_cartridge_ast_security_rejection(self, tmp_path: Path):
        """Verifies that an invalid cartridge with forbidden AST imports (os, subprocess) fails fast with exit code != 0."""
        bad_cart = tmp_path / "bad_ast_cart"
        bad_cart.mkdir()
        (bad_cart / "mechanics.py").write_text("import os\nclass BadCartridge:\n    pass\n", encoding="utf-8")

        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(bad_cart), "--ticks", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode != 0, "Expected non-zero exit code on AST security violation"
        assert "ASTSecurityError" in res.stderr or "Prohibited module import" in res.stderr

    def test_cli_invalid_cartridge_syntax_error_rejection(self, tmp_path: Path):
        """Verifies that an invalid cartridge with syntax errors fails fast with exit code != 0."""
        bad_cart = tmp_path / "bad_syntax_cart"
        bad_cart.mkdir()
        (bad_cart / "mechanics.py").write_text("def broken_syntax(:\n", encoding="utf-8")

        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(bad_cart), "--ticks", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode != 0, "Expected non-zero exit code on SyntaxError"
        assert "SyntaxError" in res.stderr

    def test_cli_invalid_cartridge_corrupt_manifest_rejection(self, tmp_path: Path):
        """Verifies that an invalid cartridge with corrupted JSON manifest fails fast with exit code != 0."""
        bad_cart = tmp_path / "bad_manifest_cart"
        bad_cart.mkdir()
        (bad_cart / "manifest.json").write_text("{corrupt_json: invalid\n", encoding="utf-8")

        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(bad_cart), "--ticks", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert res.returncode != 0, "Expected non-zero exit code on corrupt manifest JSON"
        assert "JSONDecodeError" in res.stderr

    def test_cli_nonexistent_cartridge_path_handling(self):
        """Verifies that a nonexistent cartridge directory is not booted, and CartridgeLoader raises FileNotFoundError."""
        nonexistent_path = PROJECT_ROOT / "cartridges" / "nonexistent_cartridge_path_xyz_999"
        assert not nonexistent_path.exists()

        # Verify CartridgeLoader contract raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            CartridgeLoader.load_cartridge(nonexistent_path)

        # Verify run_game.py does not claim to boot the nonexistent cartridge
        cmd = [sys.executable, "run_game.py", "--headless", "--cartridge", str(nonexistent_path), "--ticks", "5"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        assert f"Booted cartridge: {nonexistent_path.name}" not in res.stdout
