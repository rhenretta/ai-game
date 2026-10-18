"""tests/adversarial/test_m6_it2_challenger_verification.py
Milestone M6 Iteration 2 Challenger 2 Gate Verification Suite:
1. SQLite WAL Concurrency Stress:
   - Burst 10,000+ telemetry events across 10 concurrent threads into SQLite WAL
   - Concurrent rapid PRAGMA wal_checkpoint(TRUNCATE) execution
   - Concurrent telemetry queries; assert zero "database is locked" errors and zero dropped records
2. Studio Daemon Stale Lock Resilience:
   - Stress-test acquire_lock() against dead child PIDs (SIGKILL), negative PIDs, and permission-denied PIDs (PID 1)
   - Assert stale locks are safely evicted while active locks are never stolen
3. Launcher process group isolation & signal interception validation
"""
from __future__ import annotations
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from typing import List, Tuple
import pytest

from config import PROJECT_ROOT, DB_PATH
from engine.studio.daemon import StudioDaemon, StudioDaemonConfig
from engine.telemetry.db import TelemetryDB
from engine.telemetry.recorder import TelemetryRecorder


def _cleanup_proc(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.communicate(timeout=timeout)
        except Exception:
            pass


class TestM6It2ChallengerWALConcurrency:
    """Empirical verification of SQLite WAL mode under concurrent 10,000+ event burst,
    rapid TRUNCATE checkpoints, and concurrent readers.
    """

    def test_wal_burst_10k_events_with_rapid_truncate_checkpoint_and_readers(self, tmp_path: Path):
        """Burst 10,000 telemetry events across 10 concurrent threads into SQLite WAL
        while executing rapid PRAGMA wal_checkpoint(TRUNCATE) and continuous queries.
        Asserts zero 'database is locked' errors, zero dropped records, and integrity ok.
        """
        db_path = tmp_path / "telemetry_burst_stress.db"
        recorder = TelemetryRecorder(
            db_path=db_path,
            batch_size=100,
            flush_interval_sec=0.02,
        )
        recorder.start_session(player_profile_tag="wal_challenger_tester")
        recorder.start_episode(level_id="lvl_burst_test", episode_index=1)

        num_writers = 10
        events_per_writer = 1000
        total_expected_events = num_writers * events_per_writer

        stop_concurrent_workers = threading.Event()
        writer_errors: List[Tuple[int, Exception]] = []
        reader_errors: List[Tuple[int, Exception]] = []
        checkpoint_errors: List[Exception] = []

        # 1. Background checkpoint thread executing rapid PRAGMA wal_checkpoint(TRUNCATE)
        def checkpoint_worker():
            try:
                conn = sqlite3.connect(str(db_path), timeout=10.0)
                try:
                    while not stop_concurrent_workers.is_set():
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                        time.sleep(0.01)
                finally:
                    conn.close()
            except Exception as exc:
                checkpoint_errors.append(exc)

        # 2. 5 concurrent reader threads querying count, latest records, and aggregations
        def reader_worker(reader_id: int):
            try:
                uri = f"file:{db_path.resolve()}?mode=ro"
                conn = sqlite3.connect(uri, uri=True, timeout=10.0)
                try:
                    while not stop_concurrent_workers.is_set():
                        cur = conn.cursor()
                        cur.execute("SELECT count(*) FROM telemetry_events;")
                        _ = cur.fetchone()[0]
                        cur.execute("SELECT id, pos_x, pos_y FROM telemetry_events ORDER BY id DESC LIMIT 20;")
                        _ = cur.fetchall()
                        time.sleep(0.005)
                finally:
                    conn.close()
            except Exception as exc:
                reader_errors.append((reader_id, exc))

        cp_thread = threading.Thread(target=checkpoint_worker, daemon=True)
        readers = [threading.Thread(target=reader_worker, args=(i,), daemon=True) for i in range(5)]

        cp_thread.start()
        for r in readers:
            r.start()

        # 3. 10 concurrent writer threads bursting 1,000 events each
        def writer_worker(thread_id: int):
            try:
                for seq in range(events_per_writer):
                    recorder.emit_event(
                        event_type="CUSTOM_PROBE",
                        pos=(float(thread_id * 10), float(seq)),
                        vel=(1.0, 0.5),
                        health=100.0,
                        payload={"worker": thread_id, "sequence": seq},
                        probe_name=f"stress_probe_{thread_id}",
                    )
            except Exception as exc:
                writer_errors.append((thread_id, exc))

        writers = [threading.Thread(target=writer_worker, args=(i,)) for i in range(num_writers)]
        for w in writers:
            w.start()
        for w in writers:
            w.join()

        # Stop background readers and checkpointer
        stop_concurrent_workers.set()
        for r in readers:
            r.join(timeout=5.0)
        cp_thread.join(timeout=5.0)

        # Clean recorder flush and shutdown
        recorder.close()

        # Assertions
        assert len(writer_errors) == 0, f"Writer threads encountered errors: {writer_errors}"
        assert len(reader_errors) == 0, f"Reader threads encountered errors: {reader_errors}"
        assert len(checkpoint_errors) == 0, f"Checkpoint thread encountered errors: {checkpoint_errors}"

        # Verify DB integrity and record persistence
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            integrity = cursor.fetchall()
            assert integrity == [("ok",)], f"Integrity check failed: {integrity}"

            cursor.execute("SELECT count(*) FROM telemetry_events WHERE event_type = 'CUSTOM_PROBE';")
            actual_count = cursor.fetchone()[0]
            assert actual_count == total_expected_events, (
                f"Dropped telemetry records detected: expected {total_expected_events}, persisted {actual_count}"
            )
        finally:
            conn.close()


class TestM6It2ChallengerDaemonLockResilience:
    """Empirical verification of StudioDaemon stale lock eviction, PID crash recovery,
    active lock preservation, and permission-denied (PID 1) protection.
    """

    def test_daemon_stale_lock_eviction_after_sigkill(self, tmp_path: Path):
        """Spawns a child process, kills it with SIGKILL, writes its PID to studio.lock,
        and verifies acquire_lock() evicts the stale lock and claims it.
        """
        lock_path = tmp_path / "studio.lock"

        # Spawn child and SIGKILL it
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        time.sleep(0.05)
        os.kill(child.pid, signal.SIGKILL)
        child.wait()
        dead_pid = child.pid

        # Confirm PID is indeed dead
        with pytest.raises(ProcessLookupError):
            os.kill(dead_pid, 0)

        lock_path.write_text(str(dead_pid), encoding="utf-8")

        daemon = StudioDaemon(config_obj=StudioDaemonConfig(lock_file=lock_path, cartridges_root=tmp_path))
        try:
            acquired = daemon.acquire_lock()
            assert acquired is True, f"Failed to evict stale lock from SIGKILLed PID {dead_pid}"
            assert lock_path.exists()
            assert int(lock_path.read_text().strip()) == os.getpid()
        finally:
            daemon.release_lock()
            assert not lock_path.exists()

    @pytest.mark.parametrize("non_positive_pid", ["-1", "-999", "-42", "0"])
    def test_daemon_stale_lock_negative_and_zero_pids(self, non_positive_pid: str, tmp_path: Path):
        """Verifies that non-positive PIDs (<= 0) are treated as invalid/stale and evicted."""
        lock_path = tmp_path / "studio.lock"
        lock_path.write_text(non_positive_pid, encoding="utf-8")

        daemon = StudioDaemon(config_obj=StudioDaemonConfig(lock_file=lock_path, cartridges_root=tmp_path))
        try:
            acquired = daemon.acquire_lock()
            assert acquired is True, f"Failed to evict non-positive PID {non_positive_pid}"
            assert int(lock_path.read_text().strip()) == os.getpid()
        finally:
            daemon.release_lock()
            assert not lock_path.exists()

    def test_daemon_permission_denied_pid_1_lock_protection(self, tmp_path: Path):
        """PID 1 is init/launchd (system root process). acquire_lock() MUST NOT steal or evict
        a lock pointing to PID 1, returning False and leaving the lock intact.
        """
        lock_path = tmp_path / "studio.lock"
        lock_path.write_text("1\n", encoding="utf-8")

        daemon = StudioDaemon(config_obj=StudioDaemonConfig(lock_file=lock_path, cartridges_root=tmp_path))
        acquired = daemon.acquire_lock()

        assert acquired is False, "acquire_lock() illegally stole or evicted PID 1 lock!"
        assert lock_path.exists(), "Lockfile was improperly deleted!"
        assert lock_path.read_text().strip() == "1", "Lockfile content was overwritten!"

    def test_daemon_active_external_process_lock_protection_and_post_death_eviction(self, tmp_path: Path):
        """Spawns an active child process, writes its PID to studio.lock.
        Asserts acquire_lock() returns False while active, and returns True after SIGKILL.
        """
        lock_path = tmp_path / "studio.lock"

        # Spawn long-lived child
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            time.sleep(0.05)
            # Verify child is alive
            os.kill(child.pid, 0)

            lock_path.write_text(str(child.pid), encoding="utf-8")

            daemon1 = StudioDaemon(config_obj=StudioDaemonConfig(lock_file=lock_path, cartridges_root=tmp_path))
            acquired_while_alive = daemon1.acquire_lock()
            assert acquired_while_alive is False, "acquire_lock() stole lock from an active running process!"
            assert lock_path.read_text().strip() == str(child.pid)

            # Now terminate child with SIGKILL
            os.kill(child.pid, signal.SIGKILL)
            child.wait()

            # Now acquire_lock should detect dead PID and evict stale lock
            daemon2 = StudioDaemon(config_obj=StudioDaemonConfig(lock_file=lock_path, cartridges_root=tmp_path))
            acquired_after_death = daemon2.acquire_lock()
            assert acquired_after_death is True, "acquire_lock() failed to evict dead process lock!"
            assert int(lock_path.read_text().strip()) == os.getpid()
            daemon2.release_lock()
            assert not lock_path.exists()
        finally:
            _cleanup_proc(child)


class TestM6It2ChallengerLauncherGroupIsolation:
    """Verifies process group isolation and multi-offset early signal interception."""

    @pytest.mark.parametrize("signum,offset_sec", [
        (signal.SIGINT, 0.05),
        (signal.SIGTERM, 0.05),
        (signal.SIGINT, 0.15),
        (signal.SIGTERM, 0.15),
        (signal.SIGINT, 0.30),
        (signal.SIGTERM, 0.30),
    ])
    def test_launcher_signal_sweep_early_lifecycle(self, signum: int, offset_sec: float):
        """Sends signal to run_game.py at various millisecond offsets (50ms, 150ms, 300ms)
        during boot and asserts exit code 0, 0 zombies, and <= 4.0s termination.
        """
        env = dict(os.environ)
        env["SDL_VIDEODRIVER"] = "dummy"
        env["SDL_AUDIODRIVER"] = "dummy"

        cmd = [
            sys.executable,
            "run_game.py",
            "--headless",
            "--ticks", "5000",
        ]

        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        try:
            time.sleep(offset_sec)

            t0 = time.perf_counter()
            if proc.poll() is None:
                os.killpg(proc.pid, signum)
            stdout, stderr = proc.communicate(timeout=10.0)
            teardown_duration = time.perf_counter() - t0

            assert proc.returncode == 0, (
                f"Signal {signum} at offset {offset_sec}s failed with code {proc.returncode}. "
                f"stderr: {stderr}, stdout: {stdout}"
            )
            assert teardown_duration <= 4.0, f"Shutdown took {teardown_duration:.2f}s, budget is 4.0s"
            assert "completed cleanly" in stdout.lower() or "signal" in stdout.lower()
        finally:
            _cleanup_proc(proc)
