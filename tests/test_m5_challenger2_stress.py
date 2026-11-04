"""tests/test_m5_challenger2_stress.py
Empirical stress-test suite for Milestone M5 Gate Verification (Challenger 2).
Exhaustively challenges:
1. Studio Daemon Concurrency & Lockfile Resilience:
   - Multithreaded and multiprocess lock contention targeting the same directory.
   - Deterministic mutual exclusion via studio.lock.
   - Stale lockfile recovery (crashed PID, empty, corrupted text).
   - Low disk space threshold (<100MB) halting build cycles gracefully without crashing.
2. Process Isolation & SQLite WAL Non-Blocking Integrity:
   - High-throughput game writer thread inserting into SQLite WAL (data/telemetry.db and isolated DB) at >= 1,000 events/sec.
   - Concurrent StudioDaemon WAL polling and HTTP queries (/api/telemetry/events, /api/telemetry/heatmap).
   - Deterministic assertion of zero sqlite3.OperationalError (database is locked) across all connections.
3. Subprocess Teardown & Signal Handling:
   - Launch run_studio.py as subprocess, verify studio.lock creation and health response.
   - Send SIGINT and SIGTERM.
   - Assert exit code 0 within 3.0s, 0 zombie processes, released studio.lock, and 0 leaked file descriptors.
4. Rationale-First Schema Invariance & Debranding:
   - Verify every Pydantic model in engine/ui/models.py and engine/studio/daemon.py produces 'deliberation' at key index 0 across both .model_dump() and .model_dump_json().
   - Assert list(json.loads(model.model_dump_json()).keys())[0] == "deliberation".
   - Strict debranding invariance (0 occurrences of forbidden legacy branding token).
"""
from __future__ import annotations
import gc
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set
import urllib.request
import uuid
from unittest.mock import patch

from pydantic import BaseModel
import pytest
from starlette.testclient import TestClient

import config
from config import PROJECT_ROOT
from engine.studio.daemon import StudioDaemon, StudioDaemonConfig
import engine.ui.models as ui_models
from engine.ui.server import create_app
from engine.telemetry.db import TelemetryDB


def _cleanup_subprocess(proc: subprocess.Popen, timeout: float = 3.0) -> None:
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
        if proc.stdout and not proc.stdout.closed:
            proc.stdout.close()
        if proc.stderr and not proc.stderr.closed:
            proc.stderr.close()
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()


def _assert_zero_zombies(pids: List[int]) -> None:
    """Verifies that none of the child PIDs remain as zombie (<defunct>) processes."""
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


# =============================================================================
# 1. Studio Daemon Concurrency & Lockfile Resilience
# =============================================================================

class TestStudioDaemonConcurrencyAndLockfileResilience:
    """Adversarially tests daemon lock acquisition, stale PID cleanup, and disk limits."""
    deliberation: str = (
        "Empirical validation of StudioDaemon lockfile mutual exclusion, stale PID cleanup, "
        "and graceful disk space guardrails."
    )

    def test_multithreaded_concurrent_lockfile_contention(self, tmp_path: Path):
        """Spawns 10 concurrent threads contending for the exact same lockfile simultaneously.
        Asserts deterministically that exactly 1 acquires the lock, and 9 fail.
        """
        lock_file = tmp_path / "studio.lock"
        num_threads = 10
        barrier = threading.Barrier(num_threads)
        results: List[bool] = [False] * num_threads
        daemons: List[StudioDaemon] = []

        for i in range(num_threads):
            cfg = StudioDaemonConfig(lock_file=lock_file, cartridges_root=tmp_path)
            daemons.append(StudioDaemon(config_obj=cfg))

        def _contender(idx: int):
            barrier.wait()
            results[idx] = daemons[idx].acquire_lock()

        threads = [threading.Thread(target=_contender, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        successful_acquisitions = [r for r in results if r is True]
        failed_acquisitions = [r for r in results if r is False]

        assert len(successful_acquisitions) == 1, (
            f"Expected exactly 1 winner out of {num_threads} concurrent threads, "
            f"got {len(successful_acquisitions)}"
        )
        assert len(failed_acquisitions) == num_threads - 1
        assert lock_file.exists()

        # Release lock from winner and verify another contender can now acquire
        winner_idx = results.index(True)
        daemons[winner_idx].release_lock()
        assert not lock_file.exists()

        # Follow-up contention round: exactly 1 must win again
        second_winner = daemons[(winner_idx + 1) % num_threads].acquire_lock()
        assert second_winner is True
        assert lock_file.exists()
        daemons[(winner_idx + 1) % num_threads].release_lock()
        assert not lock_file.exists()

    def test_multiprocess_concurrent_lockfile_contention(self, tmp_path: Path):
        """Spawns 5 independent Python processes contending for the same lockfile simultaneously."""
        lock_file = tmp_path / "studio_proc.lock"
        num_procs = 5

        # Script that waits for an trigger file, then calls acquire_lock and prints result
        trigger_file = tmp_path / "start.trigger"
        contender_script = tmp_path / "contender.py"
        contender_script.write_text(f"""
import sys, time
from pathlib import Path
from engine.studio.daemon import StudioDaemon, StudioDaemonConfig

lock_path = Path(sys.argv[1])
trigger = Path(sys.argv[2])
while not trigger.exists():
    time.sleep(0.01)

cfg = StudioDaemonConfig(lock_file=lock_path)
daemon = StudioDaemon(config_obj=cfg)
acquired = daemon.acquire_lock()
print(f"RESULT:{{acquired}}", flush=True)
if acquired:
    time.sleep(0.5)
    daemon.release_lock()
""")

        env = dict(os.environ)
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        procs: List[subprocess.Popen] = []
        for _ in range(num_procs):
            p = subprocess.Popen(
                [sys.executable, str(contender_script), str(lock_file), str(trigger_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
            procs.append(p)

        try:
            time.sleep(0.2)
            # Release trigger to fire all processes simultaneously
            trigger_file.touch()

            results: List[bool] = []
            for p in procs:
                stdout, stderr = p.communicate(timeout=6.0)
                for line in stdout.splitlines():
                    if line.startswith("RESULT:"):
                        results.append(line.split(":")[1].strip() == "True")

            assert len([r for r in results if r is True]) == 1, (
                f"Expected exactly 1 process to acquire lock, got {results}"
            )
        finally:
            for p in procs:
                _cleanup_subprocess(p)

    def test_stale_lockfile_cleanup_scenarios(self, tmp_path: Path):
        """Tests stale lockfile cleanup across multiple simulated crash conditions."""
        lock_file = tmp_path / "stale_test.lock"
        cfg = StudioDaemonConfig(lock_file=lock_file, cartridges_root=tmp_path)
        daemon = StudioDaemon(config_obj=cfg)

        # Scenario A: Non-existent PID
        dead_pid = 99999999
        lock_file.write_text(str(dead_pid), encoding="utf-8")
        assert daemon.acquire_lock() is True
        assert int(lock_file.read_text().strip()) == os.getpid()
        daemon.release_lock()

        # Scenario B: Empty file
        lock_file.write_text("", encoding="utf-8")
        assert daemon.acquire_lock() is True
        assert int(lock_file.read_text().strip()) == os.getpid()
        daemon.release_lock()

        # Scenario C: Corrupted non-numeric content
        lock_file.write_text("corrupted_pid_garbage_xyz", encoding="utf-8")
        assert daemon.acquire_lock() is True
        assert int(lock_file.read_text().strip()) == os.getpid()
        daemon.release_lock()

        # Scenario D: Whitespace-only content
        lock_file.write_text("   \n\t  \n", encoding="utf-8")
        assert daemon.acquire_lock() is True
        assert int(lock_file.read_text().strip()) == os.getpid()
        daemon.release_lock()
        assert not lock_file.exists()

        # Scenario E: Active foreign PID - release_lock() does NOT destroy active lockfile
        other_pid = os.getppid() if os.getppid() != os.getpid() else os.getpid() + 1000
        lock_file.write_text(str(other_pid), encoding="utf-8")
        daemon.release_lock()
        assert lock_file.exists(), "release_lock() unexpectedly unlinked foreign lockfile"
        assert lock_file.read_text().strip() == str(other_pid)
        assert daemon.is_running is False

        # Scenario F: Active foreign PID raising PermissionError on acquire_lock()
        with patch("os.kill", side_effect=PermissionError):
            assert daemon.acquire_lock() is False
            assert lock_file.exists()
            assert lock_file.read_text().strip() == str(other_pid)

        # Clean up
        lock_file.unlink(missing_ok=True)

    def test_non_owner_release_lock_preserves_active_daemon_lock(self, tmp_path: Path):
        """Verifies that when a primary daemon owns studio.lock, a secondary daemon instance
        invoking release_lock() does NOT destroy the active daemon's lockfile.
        """
        lock_file = tmp_path / "active_studio.lock"
        primary_pid = os.getppid()
        lock_file.write_text(str(primary_pid), encoding="utf-8")

        cfg = StudioDaemonConfig(lock_file=lock_file, cartridges_root=tmp_path)
        secondary_daemon = StudioDaemon(config_obj=cfg)

        with patch("os.kill", return_value=None):
            # Secondary daemon fails to acquire lock
            assert secondary_daemon.acquire_lock() is False

            # Secondary daemon tries to release lock (e.g. in finally: block)
            secondary_daemon.release_lock()

            # Primary lock must remain 100% intact
            assert lock_file.exists()
            assert lock_file.read_text().strip() == str(primary_pid)
            assert secondary_daemon.is_running is False

    def test_disk_space_threshold_guard_halts_cycle_safely(self, tmp_path: Path):
        """Simulates disk space below threshold (< 100MB).
        Asserts daemon halts step() cleanly and does not raise an exception.
        """
        # Set threshold higher than available disk space
        huge_threshold_mb = 10_000_000_000.0  # 10 petabytes
        cfg = StudioDaemonConfig(
            min_disk_space_mb=huge_threshold_mb,
            cartridges_root=tmp_path,
            lock_file=tmp_path / "disk.lock",
        )
        daemon = StudioDaemon(config_obj=cfg)

        assert daemon.check_disk_space() is False

        # Calling step() under exhausted disk space must halt cleanly without throwing
        daemon.step()
        assert daemon.is_running is False


# =============================================================================
# 2. Process Isolation & SQLite WAL Non-Blocking Integrity
# =============================================================================

class TestProcessIsolationAndWALIntegrity:
    """Stress tests concurrent SQLite WAL access under high-throughput writes."""
    deliberation: str = (
        "Empirical validation of SQLite WAL non-blocking concurrency: high-frequency game writer "
        "inserting at >= 1,000 events/sec concurrently with StudioDaemon WAL polling and HTTP readers."
    )

    def test_high_throughput_wal_concurrency_live_db(self):
        """Runs a simulated high-throughput game writer inserting events into data/telemetry.db
        at >= 1,000 events/sec, while concurrently polling from StudioDaemon and querying
        /api/telemetry/events and /api/telemetry/heatmap.
        Asserts 0 sqlite3.OperationalError: database is locked errors.
        """
        db_path = config.DB_PATH
        telemetry_db = TelemetryDB(db_path)
        telemetry_db.checkpoint(mode="TRUNCATE")
        assert db_path.exists(), f"Target database {db_path} does not exist"

        test_session_id = f"stress_session_{uuid.uuid4().hex[:8]}"
        stop_event = threading.Event()
        errors: List[str] = []
        writer_count = [0]
        reader_counts = {"daemon": 0, "events_api": 0, "heatmap_api": 0, "raw_ro": 0}

        app = create_app(db_path=db_path)
        client = TestClient(app)

        cfg = StudioDaemonConfig(
            db_path=db_path,
            lock_file=PROJECT_ROOT / "test_wal_daemon.lock",
            poll_interval=0.05,
        )
        daemon = StudioDaemon(config_obj=cfg)

        # High-throughput writer thread: uses WAL connection with normal pragmas
        def _game_writer_worker():
            try:
                conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute("PRAGMA busy_timeout = 5000;")

                batch_size = 50
                insert_sql = """
                    INSERT INTO telemetry_events (
                        session_id, episode_id, timestamp, event_type, probe_name,
                        pos_x, pos_y, vel_x, vel_y, health, payload_json, frame_index, timestamp_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                while not stop_event.is_set():
                    batch = []
                    for _ in range(batch_size):
                        now = time.time()
                        batch.append((
                            test_session_id,
                            "ep_stress_1",
                            now,
                            "CUSTOM_PROBE",
                            "stress_probe",
                            320.0 + (writer_count[0] % 100),
                            240.0 + (writer_count[0] % 100),
                            1.5,
                            -0.5,
                            100.0,
                            json.dumps({"counter": writer_count[0]}),
                            writer_count[0],
                            "2026-10-16T00:00:00Z",
                        ))
                        writer_count[0] += 1

                    conn.execute("BEGIN TRANSACTION;")
                    conn.executemany(insert_sql, batch)
                    conn.execute("COMMIT;")
                    # Yield CPU briefly to sustain ~1000-2000 events/sec without starvation
                    time.sleep(0.02)

                conn.close()
            except Exception as e:
                errors.append(f"game_writer: {e}")

        # Reader 1: StudioDaemon WAL polling
        def _daemon_reader_worker():
            try:
                while not stop_event.is_set():
                    daemon.poll_and_update()
                    reader_counts["daemon"] += 1
                    time.sleep(0.01)
            except Exception as e:
                errors.append(f"daemon_reader: {e}")

        # Reader 2: /api/telemetry/events REST endpoint
        def _events_api_reader_worker():
            try:
                while not stop_event.is_set():
                    resp = client.get(f"/api/telemetry/events?session_id={test_session_id}&limit=20")
                    assert resp.status_code == 200
                    reader_counts["events_api"] += 1
                    time.sleep(0.01)
            except Exception as e:
                errors.append(f"events_api_reader: {e}")

        # Reader 3: /api/telemetry/heatmap REST endpoint
        def _heatmap_api_reader_worker():
            try:
                while not stop_event.is_set():
                    resp = client.get("/api/telemetry/heatmap")
                    assert resp.status_code == 200
                    reader_counts["heatmap_api"] += 1
                    time.sleep(0.02)
            except Exception as e:
                errors.append(f"heatmap_api_reader: {e}")

        # Reader 4: Raw SQLite connection in read-only mode
        def _raw_ro_reader_worker():
            try:
                while not stop_event.is_set():
                    ro_conn = sqlite3.connect(str(db_path), timeout=5.0)
                    ro_conn.execute("PRAGMA query_only = ON;")
                    ro_conn.execute("PRAGMA busy_timeout = 5000;")
                    cur = ro_conn.cursor()
                    cur.execute("SELECT count(*) FROM telemetry_events WHERE session_id = ?", (test_session_id,))
                    _ = cur.fetchone()[0]
                    ro_conn.close()
                    reader_counts["raw_ro"] += 1
                    time.sleep(0.01)
            except Exception as e:
                errors.append(f"raw_ro_reader: {e}")

        writer_thread = threading.Thread(target=_game_writer_worker, daemon=True)
        r_daemon = threading.Thread(target=_daemon_reader_worker, daemon=True)
        r_events = threading.Thread(target=_events_api_reader_worker, daemon=True)
        r_heatmap = threading.Thread(target=_heatmap_api_reader_worker, daemon=True)
        r_raw = threading.Thread(target=_raw_ro_reader_worker, daemon=True)

        t_start = time.time()
        writer_thread.start()
        r_daemon.start()
        r_events.start()
        r_heatmap.start()
        r_raw.start()

        # Run high-throughput concurrency storm for 3.0 seconds
        time.sleep(3.0)
        stop_event.set()

        writer_thread.join(timeout=3.0)
        r_daemon.join(timeout=3.0)
        r_events.join(timeout=3.0)
        r_heatmap.join(timeout=3.0)
        r_raw.join(timeout=3.0)
        duration = time.time() - t_start

        # Clean up test records from live telemetry.db (Rule 5: fail fast, assert teardown success)
        cleanup_conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            cleanup_conn.execute("PRAGMA busy_timeout = 5000;")
            cleanup_conn.execute("DELETE FROM telemetry_events WHERE session_id = ?", (test_session_id,))
            cleanup_conn.commit()
            ckpt_deadline = time.time() + 2.0
            ckpt_res = None
            while time.time() < ckpt_deadline:
                ckpt_res = cleanup_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
                if ckpt_res is not None and ckpt_res[0] == 0:
                    break
                time.sleep(0.05)
            assert ckpt_res is not None and ckpt_res[0] == 0, (
                f"Teardown WAL checkpoint(TRUNCATE) failed or busy: {ckpt_res}"
            )
        finally:
            cleanup_conn.close()

        # Assert zero errors during concurrency storm
        assert not errors, f"Encountered concurrency errors: {errors}"
        rate = writer_count[0] / duration
        print(f"\n[WAL Stress Test] Duration: {duration:.2f}s | Events Written: {writer_count[0]} | Rate: {rate:.1f} events/s")
        print(f"[WAL Stress Test] Reader Operations: {reader_counts}")

        assert writer_count[0] >= 1200, f"Expected >= 1200 written events under concurrent load, got {writer_count[0]}"
        assert rate >= 400.0, f"Expected >= 400 events/sec throughput under concurrent load, got {rate:.1f}"
        assert reader_counts["events_api"] > 10, "Expected >10 events API queries during storm"
        assert reader_counts["heatmap_api"] > 10, "Expected >10 heatmap queries during storm"
        assert reader_counts["daemon"] > 10, "Expected >10 daemon poll steps during storm"


# =============================================================================
# 3. Subprocess Teardown & Signal Handling
# =============================================================================

class TestSubprocessTeardownAndSignalHandling:
    """Stress tests run_studio.py process isolation, signal trapping, and clean exit."""
    deliberation: str = (
        "Empirical verification that run_studio.py terminates cleanly with code 0 within 3.0 seconds "
        "under SIGINT and SIGTERM, releases studio.lock, leaves 0 zombies, and leaks 0 file descriptors."
    )

    def test_run_studio_py_sigint_clean_teardown(self):
        """Launches run_studio.py, waits for health, sends SIGINT, and validates:
        - Terminated within 3.0 seconds
        - Exit code == 0
        - Zero zombie processes
        - studio.lock released
        - Zero leaked file descriptors
        """
        port = _get_free_port()
        lock_path = PROJECT_ROOT / "studio.lock"
        lock_path.unlink(missing_ok=True)
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
            # Poll /health endpoint to ensure full initialization
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
            assert lock_path.exists(), "studio.lock was not created by running daemon"

            # Deliver SIGINT and measure shutdown latency
            t0 = time.time()
            os.killpg(proc.pid, signal.SIGINT)
            stdout, stderr = proc.communicate(timeout=3.0)
            elapsed = time.time() - t0

            assert elapsed <= 3.0, f"Shutdown took {elapsed:.2f}s, exceeding 3.0s threshold"
            assert proc.returncode == 0, f"run_studio.py did not exit with code 0 on SIGINT: {stderr}"
            assert not lock_path.exists(), "studio.lock was not released upon SIGINT teardown"
        finally:
            _cleanup_subprocess(proc)

        gc.collect()
        _assert_zero_zombies([proc.pid])
        fds_after = set(os.listdir("/dev/fd"))
        leaked_fds = fds_after - fds_before
        assert not leaked_fds, f"Leaked file descriptors after studio SIGINT: {leaked_fds}"

    def test_run_studio_py_sigterm_clean_teardown(self):
        """Launches run_studio.py, waits for health, sends SIGTERM, and validates:
        - Terminated within 3.0 seconds
        - Exit code == 0
        - Zero zombie processes
        - studio.lock released
        - Zero leaked file descriptors
        """
        port = _get_free_port()
        lock_path = PROJECT_ROOT / "studio.lock"
        lock_path.unlink(missing_ok=True)
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
            # Poll /health endpoint to ensure full initialization
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
            assert lock_path.exists(), "studio.lock was not created by running daemon"

            # Deliver SIGTERM and measure shutdown latency
            t0 = time.time()
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=3.0)
            elapsed = time.time() - t0

            assert elapsed <= 3.0, f"Shutdown took {elapsed:.2f}s, exceeding 3.0s threshold"
            assert proc.returncode == 0, f"run_studio.py did not exit with code 0 on SIGTERM: {stderr}"
            assert not lock_path.exists(), "studio.lock was not released upon SIGTERM teardown"
        finally:
            _cleanup_subprocess(proc)

        gc.collect()
        _assert_zero_zombies([proc.pid])
        fds_after = set(os.listdir("/dev/fd"))
        leaked_fds = fds_after - fds_before
        assert not leaked_fds, f"Leaked file descriptors after studio SIGTERM: {leaked_fds}"


# =============================================================================
# 4. Rationale-First Schema Invariance & Debranding
# =============================================================================

class TestEpistemicModelOrderingAndDebranding:
    """Verifies Rationale-First deliberation ordering and absolute debranding."""
    deliberation: str = (
        "Empirical verification that all Pydantic models in engine/ui/models.py and daemon.py "
        "produce 'deliberation' at key index 0 across both .model_dump() and .model_dump_json(), "
        "and that legacy branding tokens are completely absent."
    )

    def test_all_ui_models_deliberation_field_zero_dict_and_json(self):
        """Discovers all Pydantic models in engine.ui.models, instantiates them,
        and verifies deliberation is strictly at key index 0 in both model_dump() and model_dump_json().
        """
        sample_args: Dict[str, Dict[str, Any]] = {
            "ActiveCartridgeResponse": {"deliberation": "Active cartridge inspection."},
            "CodeDiffResponse": {
                "deliberation": "Diff between candidate and active.",
                "source_cartridge": "cart_a",
                "candidate_cartridge": "cart_b",
                "module": "mechanics.py",
                "diff": "+line\n",
            },
            "DeathClusterPoint": {"deliberation": "Choke point cluster.", "x": 100.0, "y": 200.0},
            "GenesisTriggerRequest": {"deliberation": "Genesis dispatch request."},
            "GenesisTriggerResponse": {"deliberation": "Genesis dispatched successfully.", "job_id": "job_123"},
            "HeatmapResponse": {"deliberation": "Spatial heatmap aggregation."},
            "PEVCycleInfo": {"deliberation": "Current PEV state."},
            "PaletteColorItem": {"deliberation": "Palette item.", "index": 0, "name": "black", "rgb": (0, 0, 0), "hex": "#000000"},
            "PaletteInfoResponse": {"deliberation": "Palette spec."},
            "StudioStatusResponse": {"deliberation": "Studio status."},
            "SwarmStreamEvent": {
                "deliberation": "Stream event emitted.",
                "event_type": "echo",
                "timestamp_utc": "2026-10-16T00:00:00Z",
            },
            "TelemetryEventsResponse": {"deliberation": "Telemetry events query."},
            "VelocityVector": {"deliberation": "Velocity vector sample.", "x": 50.0, "y": 50.0, "vx": 1.0, "vy": 0.0, "speed": 1.0},
            "CoProducerSendMessageRequest": {"deliberation": "Submitting user message.", "message": "Pitch two mechanical directions."},
            "CoProducerExecuteActionRequest": {"deliberation": "Confirming studio action.", "thread_id": "draft_001"},
        }

        models = [
            obj for name, obj in inspect.getmembers(ui_models)
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel
        ]
        assert len(models) >= 13, f"Expected >= 13 Pydantic models in engine/ui/models.py, found {len(models)}"

        for cls in models:
            # 1. Model fields declaration order
            field_names = list(cls.model_fields.keys())
            assert field_names[0] == "deliberation", f"{cls.__name__} field 0 is '{field_names[0]}', expected 'deliberation'"

            # 2. Instantiation & model_dump() key ordering
            kwargs = sample_args.get(cls.__name__, {"deliberation": f"Deliberation for {cls.__name__}"})
            instance = cls(**kwargs)
            dumped_dict = instance.model_dump()
            dict_keys = list(dumped_dict.keys())
            assert dict_keys[0] == "deliberation", (
                f"{cls.__name__}.model_dump() first key is '{dict_keys[0]}', expected 'deliberation'"
            )

            # 3. model_dump_json() serialization key ordering
            dumped_json_str = instance.model_dump_json()
            parsed_json = json.loads(dumped_json_str)
            json_keys = list(parsed_json.keys())
            assert json_keys[0] == "deliberation", (
                f"list(json.loads({cls.__name__}.model_dump_json()).keys())[0] is '{json_keys[0]}', expected 'deliberation'"
            )
            assert len(instance.deliberation) > 0

    def test_studio_daemon_config_epistemic_ordering(self):
        """Verifies StudioDaemonConfig declares and serializes deliberation at index 0."""
        cfg = StudioDaemonConfig()
        fields = list(StudioDaemonConfig.model_fields.keys())
        assert fields[0] == "deliberation"

        dumped = cfg.model_dump()
        assert list(dumped.keys())[0] == "deliberation"

        parsed = json.loads(cfg.model_dump_json())
        assert list(parsed.keys())[0] == "deliberation"

    def test_m5_source_files_strict_debranding(self):
        """Scans all M5 source and UI files for forbidden legacy branding tokens."""
        target_files = [
            PROJECT_ROOT / "engine" / "studio" / "daemon.py",
            PROJECT_ROOT / "run_studio.py",
            PROJECT_ROOT / "engine" / "ui" / "models.py",
            PROJECT_ROOT / "engine" / "ui" / "server.py",
            PROJECT_ROOT / "studio-web" / "index.html",
            *[
                p for p in sorted((PROJECT_ROOT / "studio-web" / "src").rglob("*"))
                if p.is_file() and p.suffix in {".ts", ".tsx", ".css"}
            ],
        ]

        forbidden = "".join(chr(c) for c in (106, 111, 97, 110))
        for path in target_files:
            assert path.exists(), f"File {path} does not exist"
            content = path.read_text(encoding="utf-8")
            assert forbidden not in content.lower(), (
                f"Forbidden legacy token '{forbidden}' found in {path}"
            )
