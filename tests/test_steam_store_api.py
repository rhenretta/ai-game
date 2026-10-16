"""tests/test_steam_store_api.py
Comprehensive Unit and Integration Test Suite for Feature 9: Steam Store API Bridge.

Verifies:
  1. Metadata extraction (genres, categories, developers, developer tags, metacritic score).
  2. Metacritic score handling (present vs null/omitted).
  3. Sliding-window rate limiter (200 req / 5 min cap, min interval spacing, reset).
  4. HTTP 429 exponential backoff and Retry-After header support.
  5. Negative caching for unlisted/delisted AppIDs.
  6. Offline fallback and Rule 5 (zero silent degradation, error visible in deliberation).
  7. Rationale-First Epistemic AST compliance.
"""
import json
from pathlib import Path
import urllib.error
import pytest

from engine.steam.cache import SteamAppCache
from engine.steam.models import SteamAppDetails
from engine.steam.store_api import SteamRateLimiter, SteamStoreApiClient
from engine.steam.web_api import SteamFixtureLoader, SteamRateLimitError, SteamWebApiError
from tests.test_tier1_units import validate_rationale_first_ast


class TestSteamRateLimiter:
    """Verifies sliding-window rate limiting, minimum interval spacing, and 429 backoff."""

    def test_rate_limiter_min_interval_spacing(self):
        slept_durations = []

        def mock_sleep(secs: float):
            slept_durations.append(secs)

        limiter = SteamRateLimiter(max_requests=10, window_seconds=60.0, min_interval=0.5)

        # First call: no delay needed
        delay1 = limiter.wait_if_needed(sleep_fn=mock_sleep)
        assert delay1 == 0.0

        # Immediate second call: should sleep close to 0.5s
        delay2 = limiter.wait_if_needed(sleep_fn=mock_sleep)
        assert delay2 > 0.4
        assert len(slept_durations) == 1

    def test_rate_limiter_sliding_window_saturation(self):
        slept_durations = []

        def mock_sleep(secs: float):
            slept_durations.append(secs)

        # Capacity 3 in 10-second window
        limiter = SteamRateLimiter(max_requests=3, window_seconds=10.0, min_interval=0.0)

        # Fire 3 requests
        for _ in range(3):
            limiter.wait_if_needed(sleep_fn=mock_sleep)

        # 4th request must sleep for sliding window capacity
        delay = limiter.wait_if_needed(sleep_fn=mock_sleep)
        assert delay > 0.0
        assert len(slept_durations) == 1

    def test_rate_limiter_429_backoff_and_recovery(self):
        limiter = SteamRateLimiter()
        assert limiter._backoff_multiplier == 1.0

        # Record 429 without Retry-After (exponential scaling with jitter)
        b1 = limiter.record_429()
        assert b1 >= 2.0

        b2 = limiter.record_429()
        assert b2 >= b1 * 1.5

        # Record 429 with explicit Retry-After: 15s
        b3 = limiter.record_429(retry_after=15.0)
        assert b3 == 15.0

        # Success gradually resets backoff multiplier
        limiter.record_success()
        assert limiter._backoff_multiplier == 7.5
        limiter.record_success()
        assert limiter._backoff_multiplier == 3.75

        limiter.reset()
        assert limiter._backoff_multiplier == 1.0


class TestSteamStoreApiClient:
    """Verifies Store API queries, metadata parsing, negative caching, and Rule 5 compliance."""

    def test_store_api_metadata_extraction_ftl(self, tmp_path):
        cache = SteamAppCache(cache_path=tmp_path / "cache.json")
        client = SteamStoreApiClient(cache=cache, offline_mode=False)

        mock_payload = {
            "212680": {
                "success": True,
                "data": {
                    "name": "FTL: Faster Than Light",
                    "is_free": False,
                    "genres": [
                        {"id": "23", "description": "Indie"},
                        {"id": "28", "description": "Simulation"},
                        {"id": "2", "description": "Strategy"}
                    ],
                    "categories": [
                        {"id": 2, "description": "Single-player"},
                        {"id": 22, "description": "Steam Achievements"}
                    ],
                    "developers": ["Subset Games"],
                    "publishers": ["Subset Games"],
                    "release_date": {"date": "14 Sep, 2012"},
                    "short_description": "Spaceship simulation roguelike",
                    "metacritic": {"score": 84, "url": "https://www.metacritic.com"},
                }
            }
        }
        client._http_get = lambda url, params=None: (mock_payload, {})

        details = client.get_app_details(212680)
        assert details is not None
        assert details.appid == 212680
        assert "FTL" in details.name
        assert "Strategy" in details.genres
        assert "Single-player" in details.categories
        assert details.metacritic_score == 84
        assert "Single-player" in details.developer_tags
        assert "Subset Games" in details.developers

    def test_store_api_metacritic_score_null_handling(self, tmp_path):
        cache = SteamAppCache(cache_path=tmp_path / "cache.json")
        client = SteamStoreApiClient(cache=cache, offline_mode=False)

        raw_data = {
            "name": "Balatro",
            "is_free": False,
            "genres": [{"description": "Indie"}, {"description": "Strategy"}],
            "categories": [{"description": "Single-player"}],
            "developers": ["LocalThunk"],
            "metacritic": None,
        }
        parsed = client._parse_app_details(2764460, raw_data)
        assert parsed.metacritic_score is None
        assert parsed.name == "Balatro"
        assert parsed.developers == ["LocalThunk"]

    def test_store_api_unlisted_negative_cache(self, tmp_path):
        cache = SteamAppCache(cache_path=tmp_path / "cache.json")
        client = SteamStoreApiClient(cache=cache, offline_mode=False)

        # Mock _http_get returning success=False
        calls = 0

        def mock_http_get(url, params=None):
            nonlocal calls
            calls += 1
            return {"999999999": {"success": False}}, {}

        client._http_get = mock_http_get

        # First call: hits network and stores negative entry
        res1 = client.get_app_details(999999999)
        assert res1 is None
        assert calls == 1
        assert cache.contains(999999999) is True

        # Second call: served instantly from cache, 0 network calls
        res2 = client.get_app_details(999999999)
        assert res2 is None
        assert calls == 1

    def test_http_429_exponential_backoff_retry(self, tmp_path, monkeypatch):
        cache = SteamAppCache(cache_path=tmp_path / "cache.json")
        rate_limiter = SteamRateLimiter(min_interval=0.0)
        client = SteamStoreApiClient(
            cache=cache,
            rate_limiter=rate_limiter,
            max_retries=2,
            base_backoff=0.01,
            offline_mode=False
        )

        attempts = 0

        def mock_http_get(url, params=None):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise SteamRateLimitError("Rate limit 429")
            return {
                "212680": {
                    "success": True,
                    "data": {
                        "name": "FTL",
                        "genres": [{"description": "Indie"}],
                    }
                }
            }, {}

        client._http_get = mock_http_get

        details = client.get_app_details(212680)
        assert details is not None
        assert details.name == "FTL"
        assert attempts == 2

    def test_exhausted_retries_offline_fallback_rule_5(self, tmp_path):
        """Rule 5: When retries fail, fallback must record authentic failure reason in deliberation."""
        cache = SteamAppCache(cache_path=tmp_path / "cache.json")
        rate_limiter = SteamRateLimiter(min_interval=0.0)
        client = SteamStoreApiClient(
            cache=cache,
            rate_limiter=rate_limiter,
            max_retries=1,
            base_backoff=0.01,
            offline_mode=False
        )

        def mock_http_get(url, params=None):
            raise SteamRateLimitError("Rate limit exceeded HTTP 429")

        client._http_get = mock_http_get

        # Default: raise_on_error is False -> falls back with explicit error
        details = client.get_app_details(212680)
        assert details is not None
        assert details.source in ("offline_fixture", "offline_placeholder")
        # Rule 5: Error must be surfaced in deliberation
        assert "HTTP 429" in details.deliberation or "Live Store API failed" in details.deliberation

    def test_raise_on_error_propagation(self, tmp_path):
        cache = SteamAppCache(cache_path=tmp_path / "cache.json")
        rate_limiter = SteamRateLimiter(min_interval=0.0)
        client = SteamStoreApiClient(
            cache=cache,
            rate_limiter=rate_limiter,
            max_retries=1,
            base_backoff=0.01,
            offline_mode=False
        )

        def mock_http_get(url, params=None):
            raise SteamWebApiError("Network socket closed")

        client._http_get = mock_http_get

        with pytest.raises(SteamWebApiError):
            client.get_app_details(212680, raise_on_error=True)

    def test_batch_fetch_multiple_apps(self, tmp_path):
        cache = SteamAppCache(cache_path=tmp_path / "cache.json")
        client = SteamStoreApiClient(cache=cache, offline_mode=True)

        results = client.get_multiple_app_details([212680, 1062090])
        assert 212680 in results
        assert 1062090 in results
        assert results[212680] is not None
        assert results[1062090] is not None

    def test_store_api_ast_rationale_first(self):
        root = Path(__file__).parent.parent
        store_api_file = root / "engine" / "steam" / "store_api.py"
        assert validate_rationale_first_ast(store_api_file.read_text(encoding="utf-8")) is True
