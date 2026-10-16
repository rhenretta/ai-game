"""engine/steam/store_api.py
Public Steam Store API Bridge with persistent disk caching and sliding-window rate limiting.
Enforces Rationale-First Epistemic Architecture (deliberation declared first).
"""
from __future__ import annotations
import collections
import json
import logging
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import threading

from engine.steam.cache import SteamAppCache
from engine.steam.models import SteamAppDetails
from engine.steam.web_api import (
    STEAM_STORE_BASE_URL,
    SteamFixtureLoader,
    SteamRateLimitError,
    SteamWebApiError,
)

logger = logging.getLogger(__name__)


class SteamRateLimiter:
    """Sliding-window rate limiter ensuring strict compliance with Steam Store API limits.
    
    Default parameters:
      - max_requests: 200
      - window_seconds: 300.0 (5 minutes)
      - min_interval: 1.0 (minimum spacing between calls)
    """
    deliberation: str = "Sliding-window rate limiter ensuring compliance with Steam Store API limits."

    def __init__(
        self,
        max_requests: int = 200,
        window_seconds: float = 300.0,
        min_interval: float = 1.0,
        max_backoff: float = 60.0,
        deliberation: str = "Sliding-window rate limiter ensuring compliance with Steam Store API limits."
    ):
        self.deliberation = deliberation
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.min_interval = min_interval
        self.max_backoff = max_backoff
        self._timestamps: collections.deque[float] = collections.deque()
        self._last_call: float = 0.0
        self._backoff_multiplier: float = 1.0
        self._lock = threading.Lock()

    def wait_if_needed(self, sleep_fn: Callable[[float], None] = time.sleep) -> float:
        """Calculates delay, executes sleep, records call, and returns seconds slept."""
        with self._lock:
            total_slept = 0.0
            while True:
                now = time.monotonic()
                # Evict timestamps outside rolling window
                cutoff = now - self.window_seconds
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                sleep_time = 0.0

                # Rule 1: Respect minimum interval spacing
                time_since_last = now - self._last_call
                if time_since_last < self.min_interval:
                    sleep_time = max(sleep_time, self.min_interval - time_since_last)

                # Rule 2: Respect sliding-window request cap
                if len(self._timestamps) >= self.max_requests:
                    idx = len(self._timestamps) - self.max_requests
                    oldest = self._timestamps[idx]
                    window_delay = (oldest + self.window_seconds) - now
                    if window_delay >= 0:
                        sleep_time = max(sleep_time, window_delay + 0.001)

                # Rule 3: Respect exponential backoff penalty if 429 encountered
                if self._backoff_multiplier > 1.0:
                    sleep_time = max(sleep_time, self._backoff_multiplier)

                if sleep_time > 0:
                    total_slept += sleep_time
                    t_pre = time.monotonic()
                    sleep_fn(sleep_time)
                    t_post = time.monotonic()
                    # Defensive guard for non-advancing unit test mocks
                    if t_post - t_pre < sleep_time * 0.5:
                        call_time = t_post + sleep_time
                        self._timestamps.append(call_time)
                        self._last_call = call_time
                        return total_slept
                    continue

                call_time = time.monotonic()
                self._timestamps.append(call_time)
                self._last_call = call_time
                return total_slept

    def record_429(self, retry_after: Optional[float] = None) -> float:
        """Records HTTP 429 event and scales backoff."""
        with self._lock:
            if retry_after and retry_after > 0:
                self._backoff_multiplier = min(retry_after, self.max_backoff)
            else:
                jitter = random.uniform(0.85, 1.15)
                base = max(2.0, self._backoff_multiplier * 2.0)
                self._backoff_multiplier = min(max(2.0, base * jitter), self.max_backoff)
            return self._backoff_multiplier

    def record_success(self) -> None:
        """Gradually resets backoff multiplier on successful responses."""
        with self._lock:
            if self._backoff_multiplier > 1.0:
                self._backoff_multiplier = max(1.0, self._backoff_multiplier * 0.5)

    def reset(self) -> None:
        """Resets rate limiter state."""
        with self._lock:
            self._timestamps.clear()
            self._last_call = 0.0
            self._backoff_multiplier = 1.0


class SteamStoreApiClient:
    """Public Steam Store API client with persistent disk caching, rate limiting, and offline fallback."""
    deliberation: str = "Public Steam Store API bridge with persistent caching and rate-limiting."

    def __init__(
        self,
        cache: Optional[SteamAppCache] = None,
        rate_limiter: Optional[SteamRateLimiter] = None,
        fixture_loader: Optional[SteamFixtureLoader] = None,
        offline_mode: bool = False,
        timeout: float = 10.0,
        max_retries: int = 3,
        base_backoff: float = 1.0,
        deliberation: str = "Public Steam Store API bridge with persistent caching and rate-limiting."
    ):
        self.deliberation = deliberation
        self.cache = cache or SteamAppCache()
        self.rate_limiter = rate_limiter or SteamRateLimiter()
        self.fixture_loader = fixture_loader or SteamFixtureLoader()
        self.offline_mode = offline_mode or (os.environ.get("STEAM_OFFLINE", "0") == "1")
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.base_backoff = base_backoff

    def _http_get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Executes HTTP GET using standard library urllib, returning JSON body and headers."""
        query_string = urllib.parse.urlencode(params or {})
        full_url = f"{url}?{query_string}" if query_string else url
        req = urllib.request.Request(
            full_url,
            headers={
                "User-Agent": "ai-game-engine/1.0 (+https://github.com/rhenretta/ai-game)",
                "Accept": "application/json",
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                body = resp.read().decode("utf-8")
                return json.loads(body), headers
        except urllib.error.HTTPError as e:
            headers = {k.lower(): v for k, v in e.headers.items()} if e.headers else {}
            if e.code == 429:
                retry_header = headers.get("retry-after")
                retry_secs = float(retry_header) if retry_header and retry_header.isdigit() else None
                self.rate_limiter.record_429(retry_secs)
                raise SteamRateLimitError(f"Steam Store API rate limit exceeded (HTTP 429). Retry-After: {retry_header}")
            raise SteamWebApiError(f"Steam Store HTTP error {e.code}: {e.reason}")
        except Exception as e:
            raise SteamWebApiError(f"Network error querying Steam Store API: {e}")

    def get_app_details(
        self,
        appid: int,
        lang: str = "english",
        force_refresh: bool = False,
        raise_on_error: bool = False
    ) -> Optional[SteamAppDetails]:
        """Fetches rich metadata for appid.
        
        Lifecycle:
          1. Check persistent disk/memory cache. If found (positive or negative), return immediately.
          2. If offline_mode, return fixture or placeholder without network request.
          3. Online: execute rate-limited HTTP GET with exponential backoff on retries.
          4. Parse genres, categories, metacritic score, developers, and developer tags.
          5. If valid, persist to cache and return. If unlisted (success=false), persist negative entry.
          6. On failure, fall back to offline fixture/placeholder with explicit error deliberation (Rule 5).
        """
        # 1. Cache Hit / Negative Hit
        if not force_refresh and self.cache.contains(appid):
            return self.cache.get(appid)

        # 2. Offline Mode
        if self.offline_mode:
            details = self._build_offline_fallback(appid, reason="Offline mode enabled.")
            self.cache.set(appid, details, persist=True)
            return details

        # 3. Online Fetch with Exponential Backoff
        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                self.rate_limiter.wait_if_needed()
                params = {"appids": appid, "l": lang}
                data, _ = self._http_get(f"{STEAM_STORE_BASE_URL}/appdetails", params=params)
                self.rate_limiter.record_success()

                app_obj = data.get(str(appid), {})
                if not app_obj.get("success", False):
                    # Negative lookup: app does not exist or is delisted
                    self.cache.set(appid, None, persist=True)
                    return None

                details = self._parse_app_details(appid, app_obj.get("data", {}))
                self.cache.set(appid, details, persist=True)
                return details

            except SteamRateLimitError as e:
                last_exception = e
                logger.warning("Attempt %d/%d: 429 Rate limit on appid %d: %s", attempt + 1, self.max_retries + 1, appid, e)
                if attempt < self.max_retries:
                    jitter = random.uniform(0.85, 1.15)
                    time.sleep(self.base_backoff * (2 ** attempt) * jitter)
                else:
                    if raise_on_error:
                        raise

            except SteamWebApiError as e:
                last_exception = e
                logger.warning("Attempt %d/%d: Network error on appid %d: %s", attempt + 1, self.max_retries + 1, appid, e)
                if attempt < self.max_retries:
                    jitter = random.uniform(0.85, 1.15)
                    time.sleep(self.base_backoff * (2 ** attempt) * jitter)
                else:
                    if raise_on_error:
                        raise

        # 4. Fallback on Exhausted Retries (with Rule 5 explicit failure visibility)
        reason_msg = f"Live Store API failed after {self.max_retries + 1} attempts ({last_exception})."
        fallback_details = self._build_offline_fallback(appid, reason=reason_msg)
        self.cache.set(appid, fallback_details, persist=True)
        return fallback_details

    def _parse_app_details(self, appid: int, data: Dict[str, Any]) -> SteamAppDetails:
        """Parses raw Store API data dictionary into structured SteamAppDetails."""
        name = str(data.get("name", f"App {appid}"))
        is_free = bool(data.get("is_free", False))
        
        genres = [g.get("description", "") for g in data.get("genres", []) if isinstance(g, dict) and g.get("description")]
        categories = [c.get("description", "") for c in data.get("categories", []) if isinstance(c, dict) and c.get("description")]
        developers = data.get("developers", []) or []
        publishers = data.get("publishers", []) or []
        release_date = data.get("release_date", {}).get("date", "") if isinstance(data.get("release_date"), dict) else ""
        short_desc = data.get("short_description", "")
        header_img = data.get("header_image", "")

        # Extract Metacritic score
        meta_obj = data.get("metacritic")
        metacritic_score: Optional[int] = None
        if isinstance(meta_obj, dict) and "score" in meta_obj:
            try:
                metacritic_score = int(meta_obj["score"])
            except (ValueError, TypeError):
                metacritic_score = None

        # Synthesize developer tags from genres + core gameplay categories
        relevant_cats = {
            "Single-player", "Multi-player", "Co-op", "Online Co-op", "PvP",
            "Full controller support", "Steam Cloud", "VR Supported"
        }
        extracted_tags = list(dict.fromkeys(genres + [c for c in categories if c in relevant_cats]))

        return SteamAppDetails(
            deliberation=f"Retrieved live store details for {name} ({appid}): genres={genres}, metacritic={metacritic_score}",
            appid=appid,
            name=name,
            is_free=is_free,
            genres=genres,
            categories=categories,
            developers=developers,
            publishers=publishers,
            release_date=release_date,
            short_description=short_desc,
            header_image=header_img,
            metacritic_score=metacritic_score,
            developer_tags=extracted_tags,
            source="live_api",
        )

    def _build_offline_fallback(self, appid: int, reason: str) -> SteamAppDetails:
        """Constructs offline fallback with explicit failure visibility in deliberation (Rule 5)."""
        fixture_data = self.fixture_loader.get_app_details_fixture(appid)
        app_obj = fixture_data.get(str(appid), {})
        if app_obj.get("success", False):
            details = self._parse_app_details(appid, app_obj.get("data", {}))
            details.source = "offline_fixture"
            details.deliberation = f"Offline fallback (fixture) for {details.name} ({appid}): {reason}"
            return details

        # Minimal placeholder if no fixture exists
        return SteamAppDetails(
            deliberation=f"Offline fallback (placeholder) for App {appid}: {reason}",
            appid=appid,
            name=f"App {appid}",
            is_free=False,
            genres=["Action", "Indie"],
            categories=["Single-player"],
            developers=["Unknown Developer"],
            publishers=[],
            release_date="",
            short_description="Offline placeholder metadata.",
            header_image="",
            metacritic_score=None,
            developer_tags=["Action", "Indie", "Single-player"],
            source="offline_placeholder",
        )

    def get_multiple_app_details(
        self,
        appids: List[int],
        lang: str = "english"
    ) -> Dict[int, Optional[SteamAppDetails]]:
        """Batch fetches details for multiple appids, leveraging cache and rate limiter."""
        results: Dict[int, Optional[SteamAppDetails]] = {}
        uncached: List[int] = []

        # Check cache first
        for aid in appids:
            if self.cache.contains(aid):
                results[aid] = self.cache.get(aid)
            else:
                uncached.append(aid)

        # Sequentially resolve uncached with rate limiter
        for aid in uncached:
            details = self.get_app_details(aid, lang=lang)
            results[aid] = details

        return results
