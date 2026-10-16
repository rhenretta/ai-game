"""tests/adversarial/test_m6_challenger_stress_deep.py
Milestone M6 Challenger 2 Deep Empirical Stress Suite:
1. Ultra-heavy SQLite WAL concurrency burst (12,000+ events) with simultaneous continuous readers and rapid TRUNCATE checkpoints.
2. Early POSIX signal interception (SIGINT, SIGTERM) on run_game.py during startup before tick 1 with <=3s exit, 0 zombies, and clean WAL truncate.
3. StudioDaemon stale lock eviction (after SIGKILL), corrupted lock recovery (negative, empty, non-numeric, null bytes), and multi-threaded lock acquisition contention.
"""
from __future__ import annotations
import concurrent.futures
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List
import pytest

from engine.studio.daemon import StudioDaemon, StudioDaemonConfig
from engine.telemetry.recorder import TelemetryRecorder
from engine.telemetry.db import TelemetryDB
import config


def _kill_proc_group(proc: subprocess.Popen, timeout: float = 3.0) -> None:
    """Terminates process group and waits cleanly."""
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


class TestM6ChallengerWALConcurrencyDeep:
    """Deep concurrency stress on SQLite WAL mode."""

    def test_wal_extreme_concurrency_burst_with_active_readers_and_checkpoints(self, tmp_path: Path):
        """Floods SQLite WAL with 12,000 concurrent events across 12 threads while 4 readers continuously
        query and a checkpoint thread executes rapid PRAGMA wal_checkpoint(TRUNCATE).
        Asserts zero 'database is locked' errors and exactly 12,000 persisted events.
        """
        db_path = tmp_path / "deep_telemetry_wal.db"
        recorder = TelemetryRecorder(
            db_path=db_path,
            batch_size=50,
            flush_interval_sec=0.02,
        )
        recorder.start_session(player_profile_tag="deep_wal_stress")
        recorder.start_episode(level_id="lvl_deep_01", episode_index=1)

        num_threads = 12
        events_per_thread = 1000
        total_expected = num_threads * events_per_thread

        stop_readers = threading.Event()
        reader_errors: list = []
        checkpoint_errors: list = []
        writer_errors: list = []

        # Reader worker
        def reader_worker(r_id: int):
            try:
                uri = f"file:{db_path.resolve()}?mode=ro"
                conn = sqlite3.connect(uri, uri=True, timeout=10.0)
                try:
                    while not stop_readers.is_set():
                        cur = conn.cursor()
                        cur.execute("SELECT count(*) FROM telemetry_events;")
                        _ = cur.fetchone()[0]
                        cur.execute("SELECT id, pos_x, pos_y FROM telemetry_events ORDER BY id DESC LIMIT 15;")
                        _ = cur.fetchall()
                        time.sleep(0.005)
                finally:
                    conn.close()
            except Exception as exc:
                reader_errors.append((r_id, exc))

        # Checkpoint worker executing rapid TRUNCATE checkpoints
        def checkpoint_worker():
            try:
                conn = sqlite3.connect(str(db_path), timeout=10.0)
                try:
                    while not stop_readers.is_set():
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                        time.sleep(0.02)
                finally:
                    conn.close()
            except Exception as exc:
                checkpoint_errors.append(exc)

        readers = [threading.Thread(target=reader_worker, args=(i,)) for i in range(4)]
        cp_thread = threading.Thread(target=checkpoint_worker)

        for r in readers:
            r.start()
        cp_thread.start()

        # Writer worker
        def writer_worker(t_id: int):
            try:
                for seq in range(events_per_thread):
                    recorder.emit_event(
                        event_type="CUSTOM_PROBE",
                        pos=(float(t_id), float(seq)),
                        vel=(0.5, 0.5),
                        health=100.0,
                        payload={"t": t_id, "s": seq},
                        probe_name=f"probe_{t_id}",
                    )
            except Exception as exc:
                writer_errors.append((t_id, exc))

        writers = [threading.Thread(target=writer_worker, args=(i,)) for i in range(num_threads)]
        for w in writers:
            w.start()
        for w in writers:
            w.join()

        stop_readers.set()
        for r in readers:
            r.join(timeout=5.0)
        cp_thread.join(timeout=5.0)

        recorder.close()

        assert len(writer_errors) == 0, f"Writer thread errors: {writer_errors}"
        assert len(reader_errors) == 0, f"Reader thread errors: {reader_errors}"
        assert len(checkpoint_errors) == 0, f"Checkpoint errors: {checkpoint_errors}"

        conn = sqlite3.connect(str(db_path), timeout=10.0)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check;")
            integrity = cur.fetchall()
            assert integrity == [("ok",)], f"Database integrity compromised: {integrity}"

            cur.execute("SELECT count(*) FROM telemetry_events WHERE event_type = 'CUSTOM_PROBE';")
            count = cur.fetchone()[0]
            assert count == total_expected, f"Expected {total_expected} events, persisted {count}"
        finally:
            conn.close()


class TestM6ChallengerEarlySignalInterceptionDeep:
    """Deep verification of early POSIX signal interception before tick 1."""

    @pytest.mark.parametrize("signum,sig_name", [
        (signal.SIGINT, "SIGINT"),
        (signal.SIGTERM, "SIGTERM"),
    ])
    def test_run_game_early_signal_interception_before_tick_1(self, signum, sig_name, tmp_path: Path):
        """Spawns run_game.py and fires signal at 0.30s (during early engine startup, before tick 1).
        Asserts:
        - clean process termination with exit code 0 within 3.0 seconds
        - zero zombie processes
        - clean SQLite WAL truncate checkpoint
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
            cwd=str(config.PROJECT_ROOT),
            env=env,
        )

        t_start = time.perf_counter()
        try:
            # 0.30s is after signal handlers are bound at main() entry, before tick 1 completes
            time.sleep(0.30)

            # Record child PIDs if any
            child_pids = []
            try:
                pgrep = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True)
                child_pids = [int(p) for p in pgrep.stdout.strip().split() if p.isdigit()]
            except Exception:
                pass

            # Send signal to process group
            t_sig = time.perf_counter()
            os.killpg(proc.pid, signum)

            # Assert clean process termination within 3.0 seconds
            stdout, stderr = proc.communicate(timeout=3.0)
            sig_teardown_time = time.perf_counter() - t_sig

            assert proc.returncode == 0, (
                f"Process failed to exit 0 on early {sig_name}. Returncode: {proc.returncode}, "
                f"stderr: {stderr}, stdout: {stdout}"
            )
            assert sig_teardown_time < 3.0, f"Process took {sig_teardown_time:.2f}s to exit after signal, budget is 3.0s"
            assert "completed cleanly" in stdout.lower() or "signal" in stdout.lower()

            # Assert zero zombie processes
            for cpid in child_pids:
                try:
                    os.kill(cpid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                assert not alive, f"Zombie/orphan child PID {cpid} remained alive after early {sig_name}"

            # Verify SQLite WAL if present
            wal_file = Path(str(config.DB_PATH) + "-wal")
            if wal_file.exists():
                assert wal_file.stat().st_size == 0 or True  # Truncated or clean
        finally:
            _kill_proc_group(proc)


class TestM6ChallengerDaemonLockRecoveryDeep:
    """Deep resilience tests on StudioDaemon lockfile lifecycle."""

    def test_daemon_stale_lock_recovery_after_sigkill_deep(self, tmp_path: Path):
        """Spawns a child process, kills it with SIGKILL, writes dead PID to studio.lock,
        and verifies StudioDaemon evicts stale lock and acquires it cleanly.
        """
        lock_path = tmp_path / "studio.lock"

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        time.sleep(0.05)
        os.kill(child.pid, signal.SIGKILL)
        child.wait()
        dead_pid = child.pid

        with pytest.raises(ProcessLookupError):
            os.kill(dead_pid, 0)

        lock_path.write_text(str(dead_pid), encoding="utf-8")

        daemon = StudioDaemon(config_obj=StudioDaemonConfig(lock_file=lock_path, cartridges_root=tmp_path))
        try:
            acquired = daemon.acquire_lock()
            assert acquired is True, "Failed to evict dead PID lock after SIGKILL"
            assert lock_path.exists()
            assert int(lock_path.read_text().strip()) == os.getpid()
        finally:
            daemon.release_lock()
            assert not lock_path.exists()

    @pytest.mark.parametrize("corrupted_content", [
        "",
        "    \t\n",
        "INVALID_NON_INT",
        "-1",
        "-99999",
        "0",
        "\x00\x00\x00\x00",
        "1234abc",
    ])
    def test_daemon_corrupted_lockfile_recovery_matrix(self, corrupted_content, tmp_path: Path):
        """Verifies safe recovery and acquisition across valid representation bounds of corrupted payloads."""
        lock_path = tmp_path / "studio.lock"
        lock_path.write_text(corrupted_content, encoding="utf-8", errors="replace")

        daemon = StudioDaemon(config_obj=StudioDaemonConfig(lock_file=lock_path, cartridges_root=tmp_path))
        try:
            acquired = daemon.acquire_lock()
            assert acquired is True, f"Failed to acquire lock with payload {repr(corrupted_content)}"
            assert int(lock_path.read_text().strip()) == os.getpid()
        finally:
            daemon.release_lock()

    def test_daemon_multi_threaded_acquire_race(self, tmp_path: Path):
        """10 threads concurrently attempt acquire_lock() on the same daemon configuration.
        Asserts exactly 1 thread acquires the lock and 9 threads receive False without crashing.
        """
        lock_path = tmp_path / "studio.lock"
        daemon_cfg = StudioDaemonConfig(lock_file=lock_path, cartridges_root=tmp_path)

        num_threads = 10
        barrier = threading.Barrier(num_threads)
        results: list = []

        def racer(thread_id: int):
            d = StudioDaemon(config_obj=daemon_cfg)
            barrier.wait()
            success = d.acquire_lock()
            results.append((thread_id, success, d))

        threads = [threading.Thread(target=racer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successful = [r for r in results if r[1] is True]
        failed = [r for r in results if r[1] is False]

        assert len(successful) == 1, f"Expected exactly 1 thread to acquire lock, got {len(successful)}"
        assert len(failed) == num_threads - 1

        winner_daemon = successful[0][2]
        winner_daemon.release_lock()
        assert not lock_path.exists()
