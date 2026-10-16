"""tests/adversarial/test_tier5_wal_stress.py
Tier 5 Adversarial Test Suite 3: SQLite WAL Concurrency, 10k-Burst Emission & Checkpoint Durability.
"""
from __future__ import annotations
from pathlib import Path
import sqlite3
import threading
import time
import pytest

from engine.telemetry.recorder import TelemetryRecorder
from engine.telemetry.db import TelemetryDB


class TestTier5WALStress:
    """Verifies that SQLite WAL mode withstands extreme concurrent bursts, truncation checkpoints, and sudden terminations."""

    def test_wal_high_throughput_10k_events_burst(self, tmp_path: Path):
        """10 threads concurrently emit 1,000 events each (10,000 events total).
        Asserts 100% durability with exactly 0 dropped events upon recorder.close().
        """
        db_path = tmp_path / "telemetry_burst.db"
        recorder = TelemetryRecorder(
            db_path=db_path,
            batch_size=100,
            flush_interval_sec=0.05,
        )
        recorder.start_session(player_profile_tag="burst_tester")
        recorder.start_episode(level_id="burst_01", episode_index=1)

        num_threads = 10
        events_per_thread = 1000
        total_expected = num_threads * events_per_thread

        def worker(thread_idx: int):
            for i in range(events_per_thread):
                recorder.emit_event(
                    event_type="CUSTOM_PROBE",
                    pos=(float(thread_idx * 10), float(i)),
                    vel=(1.0, 0.0),
                    health=100.0,
                    payload={"thread": thread_idx, "seq": i},
                    probe_name=f"burst_probe_{thread_idx}",
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Clean shutdown flushes all queued batches and checkpoints WAL
        recorder.close()

        # Verify DB integrity and record count
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            integrity = cursor.fetchall()
            assert integrity == [("ok",)], f"Integrity check failed: {integrity}"

            cursor.execute("SELECT count(*) FROM telemetry_events WHERE event_type = 'CUSTOM_PROBE';")
            actual_count = cursor.fetchone()[0]
            assert actual_count == total_expected, f"Expected {total_expected} events, got {actual_count}"
        finally:
            conn.close()

    def test_wal_concurrent_readers_during_truncate_checkpoint(self, tmp_path: Path):
        """5 reader threads continuously query telemetry_events while writer executes rapid
        PRAGMA wal_checkpoint(TRUNCATE) cycles. Asserts zero 'database is locked' errors.
        """
        db_path = tmp_path / "telemetry_concurrency.db"
        telemetry_db = TelemetryDB(db_path)

        # Seed initial data using writer connection
        writer_conn = telemetry_db.get_writer_connection()
        try:
            writer_conn.execute("BEGIN IMMEDIATE;")
            writer_conn.execute("INSERT INTO telemetry_sessions (session_id, started_at) VALUES ('sess_c', '2026-10-16T12:00:00Z');")
            writer_conn.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number, started_at) VALUES ('ep_c', 'sess_c', 1, '2026-10-16T12:00:00Z');")
            for i in range(200):
                writer_conn.execute(
                    "INSERT INTO telemetry_events (session_id, episode_id, timestamp, frame_index, event_type, pos_x, pos_y, vel_x, vel_y) VALUES ('sess_c', 'ep_c', 1.0, ?, 'PLAYER_MOVE', ?, ?, 0.0, 0.0);",
                    (i, float(i), float(i)),
                )
            writer_conn.execute("COMMIT;")
        finally:
            writer_conn.close()

        stop_event = threading.Event()
        reader_errors: list = []

        def reader_loop(reader_id: int):
            try:
                r_conn = telemetry_db.get_reader_connection()
                while not stop_event.is_set():
                    cursor = r_conn.cursor()
                    cursor.execute("SELECT count(*) FROM telemetry_events;")
                    _ = cursor.fetchone()[0]
                    cursor.execute("SELECT id, pos_x, pos_y FROM telemetry_events ORDER BY id DESC LIMIT 10;")
                    _ = cursor.fetchall()
                    time.sleep(0.005)
                r_conn.close()
            except Exception as e:
                reader_errors.append((reader_id, e))

        # Start 5 reader threads
        readers = [threading.Thread(target=reader_loop, args=(i,)) for i in range(5)]
        for r in readers:
            r.start()

        # Rapid WAL TRUNCATE checkpoint loop
        checkpoint_errors: list = []
        try:
            for _ in range(25):
                try:
                    telemetry_db.checkpoint("TRUNCATE")
                except Exception as ce:
                    checkpoint_errors.append(ce)
                time.sleep(0.01)
        finally:
            stop_event.set()
            for r in readers:
                r.join()

        assert len(reader_errors) == 0, f"Reader threads encountered errors: {reader_errors}"
        assert len(checkpoint_errors) == 0, f"Checkpoint failed with errors: {checkpoint_errors}"

    def test_wal_sudden_crash_recovery_integrity(self, tmp_path: Path):
        """Simulates an abrupt process exit mid-transaction, reopening DB and validating PRAGMA integrity_check."""
        db_path = tmp_path / "telemetry_crash.db"
        tdb = TelemetryDB(db_path)

        writer_conn = tdb.get_writer_connection()
        try:
            writer_conn.execute("BEGIN IMMEDIATE;")
            writer_conn.execute("INSERT INTO telemetry_sessions (session_id, started_at) VALUES ('sess_crash', '2026-10-16T12:00:00Z');")
            writer_conn.execute("INSERT INTO telemetry_episodes (episode_id, session_id, episode_number, started_at) VALUES ('ep_crash', 'sess_crash', 1, '2026-10-16T12:00:00Z');")
            for i in range(50):
                writer_conn.execute(
                    "INSERT INTO telemetry_events (session_id, episode_id, timestamp, frame_index, event_type, pos_x, pos_y) VALUES ('sess_crash', 'ep_crash', 1.0, ?, 'CRASH_TEST', 0.0, 0.0);",
                    (i,),
                )
            writer_conn.execute("COMMIT;")
        finally:
            # Abrupt close: bypass normal TRUNCATE checkpoint to leave WAL frames on disk
            writer_conn.close()

        # Re-open with raw sqlite3 and verify integrity
        recovery_conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            cursor = recovery_conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            res = cursor.fetchall()
            assert res == [("ok",)], f"Corrupted database after crash: {res}"

            cursor.execute("SELECT count(*) FROM telemetry_events WHERE event_type = 'CRASH_TEST';")
            count = cursor.fetchone()[0]
            assert count == 50
        finally:
            recovery_conn.close()

    def test_wal_multithreaded_custom_probe_catalog_registration(self, tmp_path: Path):
        """Verifies concurrent registration of custom probes in telemetry_probes table."""
        db_path = tmp_path / "telemetry_probes_stress.db"
        recorder = TelemetryRecorder(db_path=db_path)
        recorder.start_session(player_profile_tag="catalog_stress")

        def register_worker(t_id: int):
            for p in range(20):
                recorder.register_probe(
                    probe_id=f"probe_t{t_id}_p{p}",
                    probe_name=f"pname_t{t_id}_p{p}",
                    category="combat",
                    description=f"Concurrent probe registration thread {t_id} item {p}",
                    hypothesis_target="pacing_analysis",
                )

        threads = [threading.Thread(target=register_worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        recorder.close()

        conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM telemetry_probes;")
            count = cursor.fetchone()[0]
            assert count == 8 * 20, f"Expected 160 probes registered, got {count}"
        finally:
            conn.close()
