"""tests/test_steam_cache.py
Comprehensive Unit and Integration Test Suite for Feature 10: Persistent Disk Cache.

Verifies:
  1. Initialization with missing vs existing files.
  2. Atomic POSIX disk writes (NamedTemporaryFile + fsync + os.replace).
  3. Corruption recovery (.corrupt backup creation).
  4. Cache hits, misses, negative caching (delisted AppIDs).
  5. Bulk persistence and thread safety.
  6. Rationale-First Epistemic AST compliance.
"""
import json
import os
from pathlib import Path
import threading
import time
import pytest

from engine.steam.cache import SteamAppCache
from engine.steam.models import SteamAppDetails
from tests.test_tier1_units import validate_rationale_first_ast


class TestSteamAppCache:
    """Verifies persistence, atomic replacement, negative caching, and concurrency."""

    def test_cache_initialization_missing_file(self, tmp_path):
        cache_file = tmp_path / "non_existent_cache.json"
        cache = SteamAppCache(cache_path=cache_file)
        assert len(cache._entries) == 0
        assert cache.contains(212680) is False

    def test_cache_initialization_existing_file(self, tmp_path):
        cache_file = tmp_path / "test_cache.json"
        payload = {
            "_meta": {"version": "1.0", "total_apps": 2},
            "apps": {
                "212680": {
                    "deliberation": "FTL details",
                    "appid": 212680,
                    "name": "FTL: Faster Than Light",
                    "genres": ["Indie", "Strategy"],
                    "categories": ["Single-player"],
                    "success": True,
                },
                "361280": {
                    "deliberation": "Turmoil details",
                    "appid": 361280,
                    "name": "Turmoil",
                    "genres": ["Indie", "Simulation"],
                    "categories": ["Single-player"],
                    "success": True,
                },
            },
        }
        cache_file.write_text(json.dumps(payload), encoding="utf-8")

        cache = SteamAppCache(cache_path=cache_file)
        assert cache.contains(212680) is True
        assert cache.contains(361280) is True
        details = cache.get(212680)
        assert details is not None
        assert details.name == "FTL: Faster Than Light"
        assert details.source == "disk_cache"

    def test_atomic_disk_write_valid_json(self, tmp_path):
        cache_file = tmp_path / "atomic_cache.json"
        cache = SteamAppCache(cache_path=cache_file)

        sample = SteamAppDetails(
            deliberation="Sample game",
            appid=1062090,
            name="Timberborn",
            genres=["Simulation", "Strategy"],
            categories=["Single-player"],
            metacritic_score=None,
            developer_tags=["Simulation", "Strategy", "Single-player"],
        )
        cache.set(1062090, sample, persist=True)

        assert cache_file.is_file()
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "_meta" in data
        assert "apps" in data
        assert "1062090" in data["apps"]
        assert data["apps"]["1062090"]["name"] == "Timberborn"

    def test_atomic_disk_write_tempfile_replace(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "replace_cache.json"
        cache = SteamAppCache(cache_path=cache_file)

        replaced_sources = []
        replaced_destinations = []
        orig_replace = os.replace

        def spy_replace(src, dst):
            replaced_sources.append(Path(src))
            replaced_destinations.append(Path(dst))
            return orig_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy_replace)

        sample = SteamAppDetails(
            deliberation="Sample game",
            appid=212680,
            name="FTL",
            genres=["Indie"],
        )
        cache.set(212680, sample, persist=True)

        assert len(replaced_destinations) == 1
        assert replaced_destinations[0] == cache_file.resolve()
        assert replaced_sources[0].parent == cache_file.parent.resolve()
        assert replaced_sources[0].name.startswith("steam_app_cache_")
        assert replaced_sources[0].suffix == ".tmp"

    def test_atomic_write_preserves_file_on_crash(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "crash_test.json"
        cache = SteamAppCache(cache_path=cache_file)

        sample = SteamAppDetails(
            deliberation="Good entry",
            appid=100,
            name="Good App",
            genres=["Action"],
        )
        cache.set(100, sample, persist=True)
        original_content = cache_file.read_text(encoding="utf-8")

        # Simulate failure during json.dump
        def broken_dump(*args, **kwargs):
            raise IOError("Simulated disk write failure")

        monkeypatch.setattr(json, "dump", broken_dump)

        with pytest.raises(IOError):
            cache.set(200, sample, persist=True)

        # Confirm original file on disk remains completely intact and uncorrupted
        assert cache_file.read_text(encoding="utf-8") == original_content

    def test_corrupted_json_recovery_backup(self, tmp_path):
        cache_file = tmp_path / "corrupt_cache.json"
        cache_file.write_text("CORRUPTED { broken JSON [", encoding="utf-8")

        cache = SteamAppCache(cache_path=cache_file)
        # Should initialize empty
        assert len(cache._entries) == 0

        # Backup file with .corrupt suffix should exist in directory
        corrupt_backups = list(tmp_path.glob("corrupt_cache.json.corrupt.*"))
        assert len(corrupt_backups) == 1

    def test_cache_hit_and_miss_lifecycle(self, tmp_path):
        cache_file = tmp_path / "stats_cache.json"
        cache = SteamAppCache(cache_path=cache_file)

        sample = SteamAppDetails(
            deliberation="Hit game",
            appid=500,
            name="Hit App",
            genres=["Action"],
        )
        cache.set(500, sample, persist=False)

        # Cache miss
        assert cache.get(999) is None
        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        # Cache hit
        got = cache.get(500)
        assert got is not None
        assert got.name == "Hit App"
        assert got.source == "disk_cache"
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["hit_ratio"] == 0.5

    def test_negative_cache_storage_and_hit(self, tmp_path):
        cache_file = tmp_path / "negative_cache.json"
        cache = SteamAppCache(cache_path=cache_file)

        # Store negative cache entry (unlisted / delisted game)
        cache.set(999999999, None, persist=True)

        assert cache.contains(999999999) is True
        assert cache.get(999999999) is None

        # Reload from disk and verify negative cache survives serialization
        reloaded = SteamAppCache(cache_path=cache_file)
        assert reloaded.contains(999999999) is True
        assert reloaded.get(999999999) is None

    def test_bulk_set_single_disk_flush(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "bulk_cache.json"
        cache = SteamAppCache(cache_path=cache_file)

        replace_calls = 0
        orig_replace = os.replace

        def count_replace(src, dst):
            nonlocal replace_calls
            replace_calls += 1
            return orig_replace(src, dst)

        monkeypatch.setattr(os, "replace", count_replace)

        batch = {
            1: SteamAppDetails(deliberation="App 1", appid=1, name="App 1"),
            2: SteamAppDetails(deliberation="App 2", appid=2, name="App 2"),
            3: SteamAppDetails(deliberation="App 3", appid=3, name="App 3"),
            4: None,  # Negative entry
        }
        cache.set_many(batch, persist=True)

        # Assert disk flush executed exactly once for all 4 items
        assert replace_calls == 1
        assert cache.contains(1) is True
        assert cache.contains(4) is True

    def test_thread_safe_concurrent_access(self, tmp_path):
        cache_file = tmp_path / "concurrent_cache.json"
        cache = SteamAppCache(cache_path=cache_file)

        def worker(thread_id: int):
            for i in range(20):
                aid = thread_id * 100 + i
                details = SteamAppDetails(
                    deliberation=f"Thread {thread_id} item {i}",
                    appid=aid,
                    name=f"Game {aid}",
                )
                cache.set(aid, details, persist=False)
                cache.get(aid)
                cache.contains(aid)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(cache._entries) == 100

    def test_cache_ast_rationale_first(self):
        root = Path(__file__).parent.parent
        cache_file = root / "engine" / "steam" / "cache.py"
        assert validate_rationale_first_ast(cache_file.read_text(encoding="utf-8")) is True
