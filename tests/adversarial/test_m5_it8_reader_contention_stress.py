"""tests/adversarial/test_m5_it8_reader_contention_stress.py
Milestone M5 Iteration 8 Challenger: Adversarial Reader Contention Stress.

Simulates background reader threads holding active BEGIN DEFERRED read locks
against data/telemetry.db while triggering SIGINT/SIGTERM teardown on run_game.py.

Asserts:
1. Every trial terminates with returncode 0 within teardown budget.
2. Every trial achieves a physical WAL file size of exactly 0 bytes (or unlinked) immediately upon run_game.py exit.
3. Every trial emits zero [SignalTeardownWarning] messages in stderr.
4. Active reader lock contention occurred during run_game.py execution.
5. SQLite PRAGMA integrity_check passes cleanly ('ok').
6. Post-test WAL file remains exactly 0 bytes or unlinked after all connections close.
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


def _cleanup_proc(proc: subprocess.Popen, timeout: float = 3.0) -> None:
    """Safely kills process group, reaps child processes, and closes pipes."""
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


class TestAdversarialReaderContentionStress:
    """Adversarial stress suite validating SQLite WAL truncate durability under active reader contention."""

    @pytest.mark.parametrize("trial", range(1, 6))
    def test_reader_contention_sigint_teardown(self, trial: int, sample_open_genre_cartridge):
        """Stress tests SIGINT signal teardown while 3 reader threads hold BEGIN DEFERRED locks."""
        self._run_reader_contention_trial(
            trial_id=f"sigint_trial_{trial}",
            signum=signal.SIGINT,
            sig_name="SIGINT",
            cartridge_path=sample_open_genre_cartridge,
        )

    @pytest.mark.parametrize("trial", range(1, 6))
    def test_reader_contention_sigterm_teardown(self, trial: int, sample_open_genre_cartridge):
        """Stress tests SIGTERM signal teardown while 3 reader threads hold BEGIN DEFERRED locks."""
        self._run_reader_contention_trial(
            trial_id=f"sigterm_trial_{trial}",
            signum=signal.SIGTERM,
            sig_name="SIGTERM",
            cartridge_path=sample_open_genre_cartridge,
        )

    def _run_reader_contention_trial(
        self,
        trial_id: str,
        signum: int,
        sig_name: str,
        cartridge_path: Path,
    ) -> None:
        stop_readers = threading.Event()
        reader_queries = [0]
        reader_exceptions: List[str] = []

        def reader_worker(worker_id: int):
            while not stop_readers.is_set():
                if not config.DB_PATH.exists():
                    time.sleep(0.01)
                    continue
                conn = None
                try:
                    conn = sqlite3.connect(str(config.DB_PATH), timeout=2.0)
                    conn.isolation_level = None
                    conn.execute("PRAGMA busy_timeout = 2000;")
                    conn.execute("BEGIN DEFERRED;")
                    cursor = conn.cursor()
                    cursor.execute("SELECT count(*) FROM telemetry_events;")
                    _ = cursor.fetchone()
                    # Hold active shared lock for 25ms to intentionally contend with checkpoint truncate
                    time.sleep(0.025)
                    conn.execute("COMMIT;")
                    reader_queries[0] += 1
                except sqlite3.OperationalError:
                    # Ignore initial table creation race before schema initialization
                    pass
                except Exception as exc:
                    reader_exceptions.append(f"Reader-{worker_id}: {exc}")
                finally:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                time.sleep(0.01)

        # Spawn 3 concurrent background reader threads
        threads = [threading.Thread(target=reader_worker, args=(i,), daemon=True) for i in range(3)]
        for t in threads:
            t.start()

        env = dict(os.environ)
        env["SDL_VIDEODRIVER"] = "dummy"
        env["SDL_AUDIODRIVER"] = "dummy"

        cmd = [
            sys.executable,
            "run_game.py",
            "--headless",
            "--ticks", "5000",
            "--cartridge", str(cartridge_path),
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
            # Let run_game start, initialize schema, and emit events into the WAL
            time.sleep(0.55)

            # Trigger teardown signal while readers are actively querying
            t0 = time.perf_counter()
            os.killpg(proc.pid, signum)
            stdout, stderr = proc.communicate(timeout=10.0)
            teardown_sec = time.perf_counter() - t0

            # 1. Stop background readers first so no new concurrent locks are created
            stop_readers.set()
            for t in threads:
                t.join(timeout=1.0)

            # 2. Check physical WAL file size immediately upon run_game.py process exit
            wal_file = config.DB_PATH.parent / (config.DB_PATH.name + "-wal")
            size_immediate = wal_file.stat().st_size if wal_file.exists() else 0
            assert size_immediate == 0, (
                f"[{trial_id}] run_game.py teardown failed to truncate WAL: size is {size_immediate} bytes. "
                f"Teardown duration: {teardown_sec:.2f}s"
            )

            # 3. Assert exit code 0
            assert proc.returncode == 0, (
                f"[{trial_id}] run_game.py exited with code {proc.returncode} on {sig_name}.\n"
                f"Stdout: {stdout}\nStderr: {stderr}"
            )

            # 4. Assert zero [SignalTeardownWarning] messages
            assert "[SignalTeardownWarning]" not in stderr, (
                f"[{trial_id}] SignalTeardownWarning emitted during {sig_name} teardown ({teardown_sec:.2f}s):\n{stderr}"
            )

            # 5. Assert reader contention was genuinely active during process runtime
            assert reader_queries[0] > 0, f"[{trial_id}] Readers did not complete any queries during test"

            # 6. Verify SQLite integrity check and bounded truncate checkpoint
            conn = sqlite3.connect(str(config.DB_PATH), timeout=5.0)
            try:
                conn.execute("PRAGMA busy_timeout = 5000;")
                cur = conn.cursor()
                cur.execute("PRAGMA integrity_check;")
                rows = cur.fetchall()
                assert rows == [("ok",)], f"[{trial_id}] DB integrity check failed: {rows}"

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

            # 7. Final assertion on disk after all connections closed
            if wal_file.exists():
                wal_size = wal_file.stat().st_size
                assert wal_size == 0, (
                    f"[{trial_id}] WAL file was not 0 bytes after post-test check: size is {wal_size} bytes."
                )

        finally:
            stop_readers.set()
            _cleanup_proc(proc)
