"""engine/steam/cache.py
Thread-safe persistent disk cache for Steam Store app details metadata.
Enforces Rationale-First Epistemic Architecture (deliberation declared first).
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from engine.steam.models import SteamAppDetails

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("data") / "steam_app_cache.json"


class SteamAppCache:
    """Persistent disk cache for Steam Store appdetails with POSIX atomic replacement.
    
    Invariants:
      1. deliberation declared first on class & models.
      2. Thread-safe concurrency via RLock.
      3. Atomic POSIX disk writes via NamedTemporaryFile + fsync + os.replace in same directory.
      4. Distinguishes cache hit, cache miss, and negative cache hit (delisted apps).
    """
    deliberation: str = "Thread-safe disk-persistent cache for Steam store app details."

    def __init__(
        self,
        cache_path: Optional[Union[str, Path]] = None,
        auto_load: bool = True,
        deliberation: str = "Thread-safe disk-persistent cache for Steam store app details."
    ):
        self.deliberation = deliberation
        self.cache_path = Path(cache_path or DEFAULT_CACHE_PATH).resolve()
        self._lock = threading.RLock()
        # _entries maps appid -> Optional[SteamAppDetails] (None signifies negative cache)
        self._entries: Dict[int, Optional[SteamAppDetails]] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._last_synced_at: Optional[str] = None
        if auto_load:
            self.load()

    def load(self) -> int:
        """Loads cache from disk. Returns count of loaded applications."""
        with self._lock:
            if not self.cache_path.is_file():
                return 0
            try:
                raw_text = self.cache_path.read_text(encoding="utf-8")
                if not raw_text.strip():
                    return 0
                data = json.loads(raw_text)
            except Exception as e:
                logger.warning("Corrupted Steam app cache at %s: %s. Creating backup.", self.cache_path, e)
                try:
                    backup_path = self.cache_path.with_name(f"{self.cache_path.name}.corrupt.{int(datetime.now().timestamp())}")
                    self.cache_path.rename(backup_path)
                except Exception:
                    pass
                return 0

            # Support envelope schema and legacy flat schema
            apps_dict = data.get("apps", data) if isinstance(data, dict) else {}
            loaded_count = 0

            for appid_str, item in apps_dict.items():
                if not isinstance(item, dict):
                    continue
                try:
                    appid = int(item.get("appid", appid_str))
                    if not item.get("success", True):
                        self._entries[appid] = None
                    else:
                        self._entries[appid] = SteamAppDetails.model_validate(item)
                    loaded_count += 1
                except Exception as ex:
                    logger.debug("Skipping unparseable cache item %s: %s", appid_str, ex)

            self._last_synced_at = datetime.now(timezone.utc).isoformat()
            return loaded_count

    def save(self) -> None:
        """Atomically persists cache to disk using tempfile in same dir + fsync + os.replace."""
        with self._lock:
            parent_dir = self.cache_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)

            apps_payload: Dict[str, Any] = {}
            for appid, details in self._entries.items():
                if details is None:
                    apps_payload[str(appid)] = {
                        "deliberation": f"Negative cache entry: AppID {appid} unavailable.",
                        "appid": appid,
                        "success": False,
                        "cached_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                else:
                    d = details.to_dict()
                    d["success"] = True
                    apps_payload[str(appid)] = d

            payload = {
                "_meta": {
                    "version": "1.0",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "total_apps": len(self._entries),
                    "description": "Persistent disk cache for Steam Store API appdetails metadata."
                },
                "apps": apps_payload
            }

            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=parent_dir,
                delete=False,
                prefix="steam_app_cache_",
                suffix=".tmp"
            )
            temp_path = Path(temp_file.name)
            try:
                json.dump(payload, temp_file, indent=2, ensure_ascii=False)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_file.close()
                os.replace(temp_path, self.cache_path)
                self._last_synced_at = payload["_meta"]["updated_at_utc"]
            except Exception as e:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                logger.error("Failed atomic write to Steam cache %s: %s", self.cache_path, e)
                raise

    def contains(self, appid: int) -> bool:
        """Returns True if appid is in cache (including negative lookups)."""
        with self._lock:
            return appid in self._entries

    def get(self, appid: int) -> Optional[SteamAppDetails]:
        """Retrieves details for appid. Returns None if miss OR negative cache hit."""
        with self._lock:
            if appid in self._entries:
                self._hits += 1
                details = self._entries[appid]
                if details:
                    # Mark source as disk_cache for epistemic provenance
                    details.source = "disk_cache"
                return details
            self._misses += 1
            return None

    def set(self, appid: int, details: Optional[SteamAppDetails], persist: bool = True) -> None:
        """Stores app details (or None for negative hit) and optionally persists to disk."""
        with self._lock:
            if details:
                details.cached_at_utc = datetime.now(timezone.utc).isoformat()
            self._entries[appid] = details
            if persist:
                self.save()

    def set_many(self, entries: Dict[int, Optional[SteamAppDetails]], persist: bool = True) -> None:
        """Bulk updates cache entries in memory and persists once."""
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            for appid, details in entries.items():
                if details:
                    details.cached_at_utc = now_iso
                self._entries[appid] = details
            if persist:
                self.save()

    def clear(self, delete_file: bool = False) -> None:
        """Clears memory and optionally unlinks disk cache."""
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0
            if delete_file and self.cache_path.is_file():
                try:
                    self.cache_path.unlink()
                except Exception:
                    pass

    def stats(self) -> Dict[str, Any]:
        """Returns cache telemetry and hit/miss statistics."""
        with self._lock:
            return {
                "deliberation": f"Steam cache metrics: {self._hits} hits, {self._misses} misses, {len(self._entries)} total entries.",
                "total_entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": round(self._hits / (self._hits + self._misses), 3) if (self._hits + self._misses) > 0 else 0.0,
                "last_synced_at": self._last_synced_at,
                "cache_path": str(self.cache_path),
            }
