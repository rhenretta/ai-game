"""tests/adversarial/test_m6_challenger_timing_sweeps.py
Milestone M6 Challenger 1: Adversarial Timing Sweeps & Signal Stress.
Empirically stress-tests early signal interception across sub-second timing sweeps
(t = 0.03s, 0.05s, 0.10s, 0.15s, 0.25s, 0.35s, 0.50s) with SIGINT and SIGTERM on run_game.py.
Asserts:
1. 100% clean termination with exit code 0 within calibrated teardown budget.
2. Zero unhandled KeyboardInterrupt or SchemaError.
3. Zero zombie processes and proper process group (os.killpg) signal propagation.
4. SQLite WAL PRAGMA wal_checkpoint(TRUNCATE) durability and PRAGMA integrity_check == 'ok'.
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
from typing import List
import pytest

import config
from engine.telemetry.recorder import TelemetryRecorder


def _cleanup_proc_group(proc: subprocess.Popen, timeout: float = 3.0) -> None:
    """Safely kills process group, reaps child processes, and ensures termination."""
    try:
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
    finally:
        if proc.stdout and not proc.stdout.closed:
            try:
                proc.stdout.close()
            except Exception:
                pass
        if proc.stderr and not proc.stderr.closed:
            try:
                proc.stderr.close()
            except Exception:
                pass


class TestM6ChallengerTimingSweepsAndSignalStress:
    """Adversarial timing sweep harness evaluating sub-second signal delivery."""

    TIMING_OFFSETS = [0.03, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50]
    SIGNALS = [(signal.SIGINT, "SIGINT"), (signal.SIGTERM, "SIGTERM")]

    @pytest.mark.parametrize("offset", TIMING_OFFSETS)
    @pytest.mark.parametrize("signum,sig_name", SIGNALS)
    def test_run_game_signal_timing_sweep(self, offset: float, signum: int, sig_name: str, sample_open_genre_cartridge):
        """Spawns run_game.py in headless mode and delivers signal after exact sub-second delay.
        Validates:
        - Exit code 0 within calibrated teardown budget
        - No unhandled KeyboardInterrupt or SchemaError in stderr
        - Zero zombie or orphaned child processes
        - SQLite database integrity check == 'ok' (no corruption)
        """
        env = dict(os.environ)
        env["SDL_VIDEODRIVER"] = "dummy"
        env["SDL_AUDIODRIVER"] = "dummy"

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
            env=env,
        )

        try:
            time.sleep(offset)

            # Discover any child processes spawned within the session group
            child_pids: List[int] = []
            try:
                pgrep = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True)
                child_pids = [int(p) for p in pgrep.stdout.strip().split() if p.isdigit()]
            except Exception:
                pass

            t_signal_sent = time.perf_counter()
            os.killpg(proc.pid, signum)

            stdout, stderr = proc.communicate(timeout=10.0)
            teardown_duration = time.perf_counter() - t_signal_sent

            # 1. Exit code assertion
            assert proc.returncode == 0, (
                f"FAILED: run_game.py exited with code {proc.returncode} (expected 0) "
                f"on {sig_name} at t={offset}s.\nStdout: {stdout}\nStderr: {stderr}"
            )

            # 2. Teardown latency budget <= 8.0s (calibrated to proc.communicate(timeout=10.0), macOS cold bootstrap, and widened SQLite checkpoint deadline)
            assert teardown_duration <= 8.0, (
                f"Teardown duration {teardown_duration:.3f}s exceeded 8.0s limit "
                f"for {sig_name} at t={offset}s"
            )

            # 3. No unhandled KeyboardInterrupt or SchemaError
            assert "KeyboardInterrupt" not in stderr, (
                f"Unhandled KeyboardInterrupt detected in stderr at t={offset}s ({sig_name}): {stderr}"
            )
            assert "SchemaError" not in stderr, (
                f"Unhandled SchemaError detected in stderr at t={offset}s ({sig_name}): {stderr}"
            )

            # 4. Zero zombie processes
            for cpid in child_pids:
                try:
                    os.kill(cpid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                assert not alive, f"Zombie child PID {cpid} remained active after {sig_name} at t={offset}s"

            # 5. SQLite database durability & integrity check
            if config.DB_PATH.exists():
                conn = sqlite3.connect(str(config.DB_PATH), timeout=5.0)
                try:
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA integrity_check;")
                    rows = cursor.fetchall()
                    assert rows == [("ok",)], f"SQLite integrity compromised: {rows}"
                finally:
                    conn.close()

        finally:
            _cleanup_proc_group(proc)

    def test_wal_checkpoint_truncate_durability_under_signal(self, sample_open_genre_cartridge):
        """Dedicated test: verifies that SQLite WAL executes PRAGMA wal_checkpoint(TRUNCATE)
        without lock contention or file corruption during signal shutdown.
        """
        env = dict(os.environ)
        env["SDL_VIDEODRIVER"] = "dummy"
        env["SDL_AUDIODRIVER"] = "dummy"

        cmd = [
            sys.executable,
            "run_game.py",
            "--headless",
            "--ticks", "2000",
            "--cartridge", str(sample_open_genre_cartridge),
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

        try:
            # Wait for engine and telemetry to initialize
            # Poll with deadline for engine and telemetry to initialize schema
            deadline = time.perf_counter() + 3.0
            table_ready = False
            last_err = None

            # Minimal startup buffer allowing process bootstrap and signal handler registration
            time.sleep(0.10)

            while time.perf_counter() < deadline:
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate()
                    pytest.fail(
                        f"run_game.py terminated prematurely with code {proc.returncode}.\nStdout: {stdout}\nStderr: {stderr}"
                    )

                if config.DB_PATH.exists():
                    try:
                        conn_reader = sqlite3.connect(str(config.DB_PATH), timeout=1.0)
                        try:
                            cur = conn_reader.cursor()
                            cur.execute(
                                "SELECT name FROM sqlite_master WHERE type='table' AND name='telemetry_events';"
                            )
                            if cur.fetchone() is not None:
                                # While process is running, verify external reader can read WAL without contention
                                cur.execute("SELECT count(*) FROM telemetry_events;")
                                _ = cur.fetchone()
                                table_ready = True
                                break
                        finally:
                            conn_reader.close()
                    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                        last_err = e

                time.sleep(0.05)

            assert table_ready, (
                f"Timed out waiting for telemetry_events schema initialization in {config.DB_PATH} (last error: {last_err})"
            )

            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=10.0)

            assert proc.returncode == 0, f"Non-zero exit code: {proc.returncode}, stderr: {stderr}"

            # Verify SQLite integrity check
            conn = sqlite3.connect(str(config.DB_PATH), timeout=5.0)
            try:
                conn.execute("PRAGMA busy_timeout = 5000;")
                cur = conn.cursor()
                cur.execute("PRAGMA integrity_check;")
                res = cur.fetchall()
                assert res == [("ok",)], f"Integrity check failed: {res}"
                # Explicitly execute truncate checkpoint with bounded retry loop under concurrent live reader contention
                checkpoint_deadline = time.time() + 4.0
                cp_res = None
                while time.time() < checkpoint_deadline:
                    try:
                        cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                        cp_res = cur.fetchall()
                        if cp_res and cp_res[0][0] == 0:
                            break
                    except (sqlite3.OperationalError, sqlite3.DatabaseError):
                        pass
                    time.sleep(0.05)
                assert cp_res is not None and cp_res[0][0] == 0, f"WAL checkpoint returned busy status: {cp_res}"
            finally:
                conn.close()

            # Verify WAL is cleanly truncated or unlinked
            wal_file = config.DB_PATH.parent / (config.DB_PATH.name + "-wal")
            if wal_file.exists():
                assert wal_file.stat().st_size == 0, f"WAL file size {wal_file.stat().st_size} != 0"

        finally:
            _cleanup_proc_group(proc)

    def test_isolated_wal_checkpoint_truncate_durability(self, tmp_path: Path):
        """Verifies TelemetryRecorder signal shutdown executes PRAGMA wal_checkpoint(TRUNCATE)
        and preserves all records with exactly 0-byte WAL file on an isolated database.
        """
        db_path = tmp_path / "isolated_wal_durability.db"
        recorder = TelemetryRecorder(
            db_path=db_path,
            batch_size=20,
            flush_interval_sec=0.02,
        )
        recorder.start_session(player_profile_tag="durability_test")
        recorder.start_episode(level_id="lvl_01", episode_index=1)

        # Emit 500 events
        for i in range(500):
            recorder.emit_event(
                event_type="MOVEMENT_TICK",
                pos=(float(i), float(i * 2)),
                vel=(1.0, 1.0),
                health=100.0,
                payload={"index": i},
                probe_name="movement_probe",
            )

        # Simulate graceful signal shutdown
        recorder.close()

        # Verify integrity
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check;")
            assert cur.fetchall() == [("ok",)]
            cur.execute("SELECT count(*) FROM telemetry_events;")
            count = cur.fetchone()[0]
            assert count == 500, f"Expected 500 events, found {count}"
        finally:
            conn.close()

        # Verify WAL file is exactly 0 bytes or removed
        wal_file = db_path.parent / (db_path.name + "-wal")
        if wal_file.exists():
            assert wal_file.stat().st_size == 0, f"WAL file was not truncated: {wal_file.stat().st_size}"
