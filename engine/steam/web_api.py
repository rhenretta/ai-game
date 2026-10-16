"""engine/steam/web_api.py
Steam Web API Client & Deterministic Offline Fixture Bridge.
Integrates with:
  - IPlayerService/GetOwnedGames/v0001
  - ISteamUserStats/GetUserStatsForGame/v0002
  - ISteamUserStats/GetPlayerAchievements/v0001
  - Steam Store API (store.steampowered.com/api/appdetails)
Provides built-in deterministic fixture fallback for offline development & tests.
"""
from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from engine.steam.models import (
    SteamAchievement,
    SteamAppDetails,
    SteamGameRecord,
    SteamLibraryDataset,
    SteamOwnedGame,
    SteamUserStats,
)


STEAM_API_BASE_URL = "https://api.steampowered.com"
STEAM_STORE_BASE_URL = "https://store.steampowered.com/api"


class SteamWebApiError(Exception):
    """Base exception for Steam Web API communication failures."""
    deliberation: str = "Base exception for Steam Web API communication failures."


class SteamAuthenticationError(SteamWebApiError):
    """Raised when Steam Web API key is missing or invalid."""
    deliberation: str = "Raised when Steam Web API key is missing or invalid."


class SteamRateLimitError(SteamWebApiError):
    """Raised when Steam API responds with HTTP 429 Too Many Requests."""
    deliberation: str = "Raised when Steam API responds with HTTP 429 Too Many Requests."


class SteamFixtureLoader:
    """Loads deterministic offline Steam fixtures for testing and offline development."""
    deliberation: str = "Loading deterministic Steam test fixtures."

    def __init__(
        self,
        fixtures_dir: Optional[Union[str, Path]] = None,
        deliberation: str = "Loading deterministic Steam test fixtures."
    ):
        self.deliberation = deliberation
        if fixtures_dir:
            self.fixtures_dir = Path(fixtures_dir).resolve()
        else:
            # Default to tests/fixtures/steam relative to project root
            cwd = Path.cwd()
            self.fixtures_dir = cwd / "tests" / "fixtures" / "steam"

    def get_owned_games_fixture(self) -> Dict[str, Any]:
        """Loads deterministic fixture for IPlayerService/GetOwnedGames."""
        fixture_file = self.fixtures_dir / "web_api" / "get_owned_games.json"
        if fixture_file.is_file():
            return json.loads(fixture_file.read_text(encoding="utf-8"))
        
        # Built-in fallback fixture if file is not on disk
        return {
            "response": {
                "game_count": 3,
                "games": [
                    {
                        "appid": 212680,
                        "name": "FTL: Faster Than Light",
                        "playtime_forever": 5390,
                        "playtime_2weeks": 120,
                        "playtime_mac_forever": 4500,
                        "rtime_last_played": 1668898503,
                        "img_icon_url": "6f982210fe9a377f"
                    },
                    {
                        "appid": 1062090,
                        "name": "Timberborn",
                        "playtime_forever": 2840,
                        "playtime_2weeks": 60,
                        "playtime_mac_forever": 2840,
                        "rtime_last_played": 1698898503,
                        "img_icon_url": "360236e555049f20"
                    },
                    {
                        "appid": 2764460,
                        "name": "Balatro",
                        "playtime_forever": 8920,
                        "playtime_2weeks": 480,
                        "playtime_mac_forever": 8920,
                        "rtime_last_played": 1718898503,
                        "img_icon_url": "e9dee7e306087ce8"
                    }
                ]
            }
        }

    def get_app_details_fixture(self, appid: int) -> Dict[str, Any]:
        """Loads deterministic fixture for store.steampowered.com/api/appdetails."""
        fixture_file = self.fixtures_dir / "web_api" / f"appdetails_{appid}.json"
        if fixture_file.is_file():
            return json.loads(fixture_file.read_text(encoding="utf-8"))

        # Built-in synthetic fixture profiles
        catalogue = {
            212680: {
                "name": "FTL: Faster Than Light",
                "genres": ["Indie", "Simulation", "Strategy", "Roguelike"],
                "categories": ["Single-player", "Steam Achievements"],
                "short_description": "A spaceship simulation roguelike-like with high tactical decision making."
            },
            1062090: {
                "name": "Timberborn",
                "genres": ["Indie", "Simulation", "Strategy", "City Builder"],
                "categories": ["Single-player", "Steam Cloud"],
                "short_description": "Beaver city-building game with water physics and vertical architecture."
            },
            2764460: {
                "name": "Balatro",
                "genres": ["Indie", "Strategy", "Roguelike", "Deckbuilder"],
                "categories": ["Single-player", "Steam Achievements"],
                "short_description": "A poker-themed roguelike deckbuilder with intense synergies."
            }
        }

        entry = catalogue.get(appid, {
            "name": f"Steam App {appid}",
            "genres": ["Action", "Indie"],
            "categories": ["Single-player"],
            "short_description": "Generic steam application."
        })

        return {
            str(appid): {
                "success": True,
                "data": {
                    "type": "game",
                    "name": entry["name"],
                    "steam_appid": appid,
                    "is_free": False,
                    "genres": [{"id": str(i), "description": g} for i, g in enumerate(entry["genres"])],
                    "categories": [{"id": i, "description": c} for i, c in enumerate(entry["categories"])],
                    "release_date": {"coming_soon": False, "date": "1 Jan, 2024"},
                    "developers": ["Indie Studio"],
                    "publishers": ["Indie Publisher"],
                    "short_description": entry["short_description"],
                    "header_image": f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"
                }
            }
        }

    def get_player_achievements_fixture(self, appid: int) -> Dict[str, Any]:
        """Loads deterministic fixture for ISteamUserStats/GetPlayerAchievements."""
        fixture_file = self.fixtures_dir / "web_api" / f"player_achievements_{appid}.json"
        if fixture_file.is_file():
            return json.loads(fixture_file.read_text(encoding="utf-8"))

        return {
            "playerstats": {
                "steamID": "76561197985203987",
                "gameName": f"App {appid}",
                "achievements": [
                    {"apiname": "ACH_FIRST_VICTORY", "achieved": 1, "unlocktime": 1668898500},
                    {"apiname": "ACH_HARD_MODE", "achieved": 1, "unlocktime": 1668900000},
                    {"apiname": "ACH_SECRET_RUN", "achieved": 0, "unlocktime": 0}
                ],
                "success": True
            }
        }


class SteamWebApiClient:
    """Client for Steam Web API and Steam Store API with deterministic offline fallback."""
    deliberation: str = "Managing Steam Web API calls and deterministic fixtures."

    def __init__(
        self,
        api_key: Optional[str] = None,
        steam_id: Optional[str] = None,
        offline_mode: bool = False,
        fixtures_dir: Optional[Union[str, Path]] = None,
        timeout: float = 10.0,
        deliberation: str = "Managing Steam Web API calls and deterministic fixtures."
    ):
        self.deliberation = deliberation
        self.api_key = api_key or os.environ.get("STEAM_WEB_API_KEY") or os.environ.get("STEAM_API_KEY")
        self.steam_id = steam_id or os.environ.get("STEAM_ID") or os.environ.get("STEAM_USER_ID")
        self.offline_mode = offline_mode or (os.environ.get("STEAM_OFFLINE", "0") == "1")
        self.timeout = float(timeout)
        self.fixture_loader = SteamFixtureLoader(fixtures_dir=fixtures_dir)
        self._store_cache: Dict[int, SteamAppDetails] = {}

    def _http_get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Executes HTTP GET request using Python standard library urllib."""
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
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SteamAuthenticationError(f"Steam Web API authorization failed (HTTP {e.code}): {e.reason}")
            elif e.code == 429:
                raise SteamRateLimitError("Steam API rate limit exceeded (HTTP 429)")
            raise SteamWebApiError(f"Steam HTTP error {e.code}: {e.reason}")
        except Exception as e:
            raise SteamWebApiError(f"Network error querying Steam API: {e}")

    def get_owned_games(
        self,
        steam_id: Optional[str] = None,
        include_appinfo: bool = True,
        include_played_free_games: bool = True,
    ) -> List[SteamOwnedGame]:
        """Fetches list of owned games via IPlayerService/GetOwnedGames/v0001.
        
        Falls back deterministically to offline fixture if offline_mode is True or api_key is missing.
        """
        sid = steam_id or self.steam_id

        if self.offline_mode or not self.api_key:
            data = self.fixture_loader.get_owned_games_fixture()
        else:
            if not sid:
                raise SteamWebApiError("SteamID is required to call GetOwnedGames.")
            params = {
                "key": self.api_key,
                "steamid": sid,
                "include_appinfo": 1 if include_appinfo else 0,
                "include_played_free_games": 1 if include_played_free_games else 0,
                "format": "json"
            }
            try:
                data = self._http_get(f"{STEAM_API_BASE_URL}/IPlayerService/GetOwnedGames/v0001/", params=params)
            except SteamWebApiError:
                # If network fails in development, fallback to fixture
                data = self.fixture_loader.get_owned_games_fixture()

        games_raw = data.get("response", {}).get("games", [])
        owned_games: List[SteamOwnedGame] = []

        for g in games_raw:
            appid = int(g.get("appid", 0))
            name = str(g.get("name", f"App {appid}"))
            pt_forever = int(g.get("playtime_forever", 0))
            pt_2weeks = g.get("playtime_2weeks")
            pt_mac = g.get("playtime_mac_forever")
            pt_win = g.get("playtime_windows_forever")
            pt_linux = g.get("playtime_linux_forever")
            last_played = int(g.get("rtime_last_played", 0))
            icon_url = str(g.get("img_icon_url", ""))

            owned_games.append(
                SteamOwnedGame(
                    deliberation=f"Parsed owned game {name} ({appid}) with {pt_forever}m playtime",
                    appid=appid,
                    name=name,
                    playtime_forever=pt_forever,
                    playtime_2weeks=pt_2weeks,
                    playtime_mac_forever=pt_mac,
                    playtime_windows_forever=pt_win,
                    playtime_linux_forever=pt_linux,
                    rtime_last_played=last_played,
                    img_icon_url=icon_url,
                )
            )

        return owned_games

    def get_app_details(self, appid: int, lang: str = "english") -> Optional[SteamAppDetails]:
        """Fetches public store metadata and genres via store.steampowered.com/api/appdetails.
        
        Does NOT require a Steam Web API key. Caches results in memory to avoid rate limits.
        """
        if appid in self._store_cache:
            return self._store_cache[appid]

        if self.offline_mode:
            data = self.fixture_loader.get_app_details_fixture(appid)
        else:
            params = {"appids": appid, "l": lang}
            try:
                data = self._http_get(f"{STEAM_STORE_BASE_URL}/appdetails", params=params)
            except SteamWebApiError:
                data = self.fixture_loader.get_app_details_fixture(appid)

        app_obj = data.get(str(appid), {})
        if not app_obj.get("success", False):
            return None

        app_data = app_obj.get("data", {})
        name = str(app_data.get("name", f"App {appid}"))
        is_free = bool(app_data.get("is_free", False))
        
        genres = [g.get("description", "") for g in app_data.get("genres", []) if g.get("description")]
        categories = [c.get("description", "") for c in app_data.get("categories", []) if c.get("description")]
        developers = app_data.get("developers", [])
        publishers = app_data.get("publishers", [])
        release_date = app_data.get("release_date", {}).get("date", "")
        short_desc = app_data.get("short_description", "")
        header_img = app_data.get("header_image", "")

        details = SteamAppDetails(
            deliberation=f"Retrieved store details for {name} ({appid}): genres={genres}",
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
        )

        self._store_cache[appid] = details
        return details

    def get_player_achievements(
        self,
        steam_id: Optional[str] = None,
        appid: int = 0,
        lang: str = "english"
    ) -> List[SteamAchievement]:
        """Fetches player achievements for a game via ISteamUserStats/GetPlayerAchievements/v0001."""
        sid = steam_id or self.steam_id

        if self.offline_mode or not self.api_key:
            data = self.fixture_loader.get_player_achievements_fixture(appid)
        else:
            if not sid:
                raise SteamWebApiError("SteamID is required to call GetPlayerAchievements.")
            params = {
                "key": self.api_key,
                "steamid": sid,
                "appid": appid,
                "l": lang
            }
            try:
                data = self._http_get(f"{STEAM_API_BASE_URL}/ISteamUserStats/GetPlayerAchievements/v0001/", params=params)
            except SteamWebApiError:
                data = self.fixture_loader.get_player_achievements_fixture(appid)

        stats = data.get("playerstats", {})
        achievements_raw = stats.get("achievements", [])
        achievements: List[SteamAchievement] = []

        for a in achievements_raw:
            apiname = str(a.get("apiname", ""))
            achieved = bool(a.get("achieved", 0) == 1)
            unlock_time = int(a.get("unlocktime", 0))
            name = a.get("name")
            desc = a.get("description")

            achievements.append(
                SteamAchievement(
                    deliberation=f"Achievement {apiname}: achieved={achieved}",
                    apiname=apiname,
                    achieved=achieved,
                    unlock_time=unlock_time,
                    name=name,
                    description=desc,
                )
            )

        return achievements

    def get_user_stats_for_game(
        self,
        steam_id: Optional[str] = None,
        appid: int = 0
    ) -> SteamUserStats:
        """Fetches user stats and achievements via ISteamUserStats/GetUserStatsForGame/v0002."""
        sid = steam_id or self.steam_id
        achievements = self.get_player_achievements(steam_id=sid, appid=appid)

        game_name = f"App {appid}"
        stats_dict: Dict[str, float] = {}

        if not self.offline_mode and self.api_key and sid:
            params = {
                "key": self.api_key,
                "steamid": sid,
                "appid": appid
            }
            try:
                data = self._http_get(f"{STEAM_API_BASE_URL}/ISteamUserStats/GetUserStatsForGame/v0002/", params=params)
                pstats = data.get("playerstats", {})
                game_name = pstats.get("gameName", game_name)
                for s in pstats.get("stats", []):
                    sname = s.get("name")
                    sval = s.get("value", 0.0)
                    if sname:
                        stats_dict[sname] = float(sval)
            except SteamWebApiError:
                pass

        return SteamUserStats(
            deliberation=f"Aggregated user stats for {game_name} ({appid})",
            steam_id=str(sid or "unknown"),
            appid=appid,
            game_name=game_name,
            stats=stats_dict,
            achievements=achievements,
        )

    def to_dataset(
        self,
        games: Optional[List[SteamOwnedGame]] = None,
        player_id: Optional[str] = None,
        enrich_store_details: bool = True
    ) -> SteamLibraryDataset:
        """Converts owned games list into a SteamLibraryDataset enriched with genres."""
        sid = player_id or self.steam_id or "local_player"
        owned = games if games is not None else self.get_owned_games(steam_id=sid)

        records: List[SteamGameRecord] = []
        for g in owned:
            genres: List[str] = []
            if enrich_store_details:
                details = self.get_app_details(g.appid)
                if details:
                    genres = details.genres

            records.append(
                SteamGameRecord(
                    deliberation=f"Converted Web API game {g.name} ({g.appid}) to GameRecord.",
                    appid=g.appid,
                    name=g.name,
                    playtime_forever_minutes=g.playtime_forever,
                    playtime_2weeks_minutes=g.playtime_2weeks,
                    rtime_last_played=g.rtime_last_played,
                    genres=genres,
                )
            )

        return SteamLibraryDataset(
            deliberation=f"Constructed SteamLibraryDataset with {len(records)} games from Web API.",
            player_id=str(sid),
            source="web_api",
            games=records,
        )
