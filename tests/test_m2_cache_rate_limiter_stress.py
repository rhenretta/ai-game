"""tests/test_m2_cache_rate_limiter_stress.py
Empirical Challenger Stress Test Suite for Milestone M2:
1. SteamAppCache Multi-threaded Concurrency & Data Integrity
2. SteamAppCache Truncated and Corrupted File Recovery
3. SteamRateLimiter 250-Request Burst Sliding-Window Invariant Verification
4. SteamRateLimiter HTTP 429 Exponential Backoff & Jitter Verification
"""
import concurrent.futures
import json
import os
from pathlib import Path
import random
import tempfile
import threading
import time
import pytest

from engine.steam.cache import SteamAppCache
from engine.steam.models import SteamAppDetails
from engine.steam.store_api import SteamRateLimiter


class TestSteamAppCacheStress:
    """Multi-threaded stress testing and corruption recovery for SteamAppCache."""
    deliberation: str = "Empirical stress testing for SteamAppCache multi-threaded concurrency and recovery."

    def test_concurrent_multithreaded_read_write_stress(self, tmp_path):
        """16 concurrent worker threads performing mixed random reads, writes, and negative cache sets.
        
        Verifies:
          - Zero unhandled exceptions.
          - Zero file corruptions on disk.
          - 100% data integrity under concurrent fsync + atomic replace.
          - Fresh cache instance reloads all entries accurately.
        """
        cache_file = tmp_path / "concurrent_stress_cache.json"
        cache = SteamAppCache(cache_path=cache_file)

        # Seed initial entries
        for aid in range(10):
            cache.set(
                aid,
                SteamAppDetails(deliberation=f"Initial seed {aid}", appid=aid, name=f"SeedApp_{aid}"),
                persist=False
            )
        cache.save()

        num_threads = 16
        ops_per_thread = 100
        errors = []

        def worker(tid: int):
            try:
                for op_idx in range(ops_per_thread):
                    action = random.choice(["get", "contains", "set_positive", "set_negative", "stats"])
                    aid = random.randint(0, 60)

                    if action == "get":
                        cache.get(aid)
                    elif action == "contains":
                        cache.contains(aid)
                    elif action == "set_positive":
                        details = SteamAppDetails(
                            deliberation=f"Thread {tid} op {op_idx}",
                            appid=aid,
                            name=f"App_{aid}_T{tid}",
                            genres=["Indie", "Action"] if aid % 2 == 0 else ["Strategy"],
                            categories=["Single-player"],
                        )
                        cache.set(aid, details, persist=True)
                    elif action == "set_negative":
                        cache.set(aid, None, persist=True)
                    elif action == "stats":
                        stats = cache.stats()
                        assert stats["total_entries"] >= 10
            except Exception as e:
                errors.append((tid, e))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Invariant 1: Zero unhandled exceptions
        assert len(errors) == 0, f"Encountered {len(errors)} thread errors: {errors[:3]}"

        # Invariant 2: Disk file exists and is valid JSON (zero corruption)
        assert cache_file.is_file(), "Cache file was deleted or failed to persist"
        raw_content = cache_file.read_text(encoding="utf-8")
        parsed_data = json.loads(raw_content)
        assert "_meta" in parsed_data
        assert "apps" in parsed_data
        assert parsed_data["_meta"]["total_apps"] == len(parsed_data["apps"])

        # Invariant 3: Independent fresh cache instance reloads disk state with 100% integrity
        fresh_cache = SteamAppCache(cache_path=cache_file)
        assert len(fresh_cache._entries) == len(parsed_data["apps"])

        for aid_str, app_dict in parsed_data["apps"].items():
            aid = int(aid_str)
            assert fresh_cache.contains(aid) is True
            if not app_dict.get("success", True):
                assert fresh_cache.get(aid) is None  # Negative cache
            else:
                loaded = fresh_cache.get(aid)
                assert loaded is not None
                assert loaded.appid == aid

    def test_truncated_json_file_recovery(self, tmp_path):
        """Simulate half-written / truncated JSON on disk.
        
        Verifies:
          - No unhandled exception or crash.
          - Creates .corrupt.<timestamp> backup.
          - Initializes cleanly with 0 entries.
        """
        cache_file = tmp_path / "truncated_cache.json"
        # Truncate halfway through JSON payload
        cache_file.write_text('{"_meta": {"version": "1.0"}, "apps": {"212680": {"deliberation":', encoding="utf-8")

        cache = SteamAppCache(cache_path=cache_file)
        assert len(cache._entries) == 0

        # Verify corrupted backup exists
        backups = list(tmp_path.glob("truncated_cache.json.corrupt.*"))
        assert len(backups) == 1, "Expected .corrupt backup file to be created"

    def test_binary_garbage_recovery(self, tmp_path):
        """Simulate binary garbage / random bytes on disk."""
        cache_file = tmp_path / "garbage_cache.json"
        cache_file.write_bytes(b"\x00\xff\xfe\xfd\x01\x02\x03\x04" * 128)

        cache = SteamAppCache(cache_path=cache_file)
        assert len(cache._entries) == 0

        backups = list(tmp_path.glob("garbage_cache.json.corrupt.*"))
        assert len(backups) == 1

    def test_empty_zero_byte_file_recovery(self, tmp_path):
        """Simulate 0-byte file (e.g. touch / empty file)."""
        cache_file = tmp_path / "empty_cache.json"
        cache_file.write_text("", encoding="utf-8")

        cache = SteamAppCache(cache_path=cache_file)
        assert len(cache._entries) == 0

        # 0-byte file should be treated as empty without creating a corrupt backup
        backups = list(tmp_path.glob("empty_cache.json.corrupt.*"))
        assert len(backups) == 0

    def test_atomic_write_preserves_disk_file_on_write_error(self, tmp_path, monkeypatch):
        """Simulate unexpected disk error during atomic replace.
        
        Verifies:
          - Target file on disk remains completely unmodified.
          - Temporary file is cleaned up.
        """
        cache_file = tmp_path / "protected_cache.json"
        cache = SteamAppCache(cache_path=cache_file)

        initial_sample = SteamAppDetails(
            deliberation="Initial good entry",
            appid=212680,
            name="FTL",
            genres=["Indie"],
        )
        cache.set(212680, initial_sample, persist=True)
        original_bytes = cache_file.read_bytes()

        # Simulate exception during json.dump
        def failing_dump(*args, **kwargs):
            raise OSError("Simulated disk full")

        monkeypatch.setattr(json, "dump", failing_dump)

        new_sample = SteamAppDetails(
            deliberation="New entry",
            appid=361280,
            name="Turmoil",
            genres=["Indie"],
        )
        with pytest.raises(OSError, match="Simulated disk full"):
            cache.set(361280, new_sample, persist=True)

        # File on disk must match original perfectly
        assert cache_file.read_bytes() == original_bytes
        # No orphan .tmp files in directory
        tmp_files = list(tmp_path.glob("steam_app_cache_*.tmp"))
        assert len(tmp_files) == 0


class TestSteamRateLimiterStress:
    """Adversarial stress testing for SteamRateLimiter sliding window and backoff."""
    deliberation: str = "Empirical stress testing for SteamRateLimiter sliding window capacity and backoff/jitter."

    def test_sliding_window_burst_250_requests_invariant(self):
        """Simulate a burst of 250 requests against SteamRateLimiter.
        
        Invariant:
          In ANY 5-minute (300.0s) sliding window [T - 300.0, T], the number of completed
          requests must NEVER exceed max_requests (200).
        """
        simulated_time = 0.0

        def mock_sleep(secs: float):
            nonlocal simulated_time
            simulated_time += secs

        def mock_monotonic() -> float:
            return simulated_time

        limiter = SteamRateLimiter(max_requests=200, window_seconds=300.0, min_interval=0.0)

        # Patch time.monotonic for virtual time simulation
        real_monotonic = time.monotonic
        time.monotonic = mock_monotonic

        completed_timestamps = []
        try:
            for _ in range(250):
                limiter.wait_if_needed(sleep_fn=mock_sleep)
                completed_timestamps.append(simulated_time)
        finally:
            time.monotonic = real_monotonic

        # Verify sliding window invariant across all requests
        violations = []
        for t in completed_timestamps:
            # Count calls in any 300s window ending at t: [t - 300.0, t]
            count = sum(1 for ts in completed_timestamps if t - 300.0 <= ts <= t)
            if count > 200:
                violations.append((t, count))

        # This assertion documents whether the rate limiter adheres to the sliding window
        assert len(violations) == 0, (
            f"Sliding window violated! {len(violations)} timestamps had >200 requests in 300s. "
            f"Worst violation at t={violations[0][0]}: {violations[0][1]} requests in 300s window."
        )

    def test_http_429_backoff_and_jitter_presence(self):
        """Simulate HTTP 429 responses and verify exponential backoff and jitter.
        
        Verifies:
          1. record_429() scales exponentially (e.g. 2, 4, 8, 16...).
          2. record_429(retry_after=N) respects explicit Retry-After header.
          3. Multiplier is bounded by max_backoff (60.0).
          4. Jitter verification: Repeated 429 events should not be 100% deterministic
             across multiple runs to prevent thundering herd.
        """
        limiter = SteamRateLimiter(max_backoff=60.0)

        # 1. Exponential scaling
        b1 = limiter.record_429()
        assert b1 >= 2.0
        b2 = limiter.record_429()
        assert b2 >= b1 * 1.5
        b3 = limiter.record_429()
        assert b3 >= b2 * 1.5

        # 2. Retry-After header
        b_retry = limiter.record_429(retry_after=15.0)
        assert b_retry == 15.0

        # 3. Max backoff cap
        b_cap = limiter.record_429(retry_after=120.0)
        assert b_cap == 60.0

        # 4. Jitter presence check
        # Repeated trials of record_429() from reset state
        trials = []
        for _ in range(10):
            limiter.reset()
            seq = [limiter.record_429() for _ in range(4)]
            trials.append(seq)

        # Check if any variation exists across trials
        has_jitter = any(trials[i] != trials[0] for i in range(1, len(trials)))
        assert has_jitter, (
            "No jitter detected! SteamRateLimiter.record_429() produced identical deterministic "
            f"multipliers {trials[0]} across all 10 runs, creating a thundering herd risk."
        )
