"""engine/steam/models.py
Data models and Rationale-First Pydantic v2 schemas for Steam library ingestion and player DNA profiling.
Every model strictly declares deliberation as its very first property.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SteamUserRecord(BaseModel):
    """Represents a Steam user account discovered in loginusers.vdf."""
    deliberation: str = Field(default="", description="Epistemic rationale describing user account discovery.")
    steam_id_64: str = ""
    account_name: str = ""
    persona_name: str = ""
    is_most_recent: bool = False
    auto_login: bool = False
    timestamp: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "steam_id_64": self.steam_id_64,
            "account_name": self.account_name,
            "persona_name": self.persona_name,
            "is_most_recent": self.is_most_recent,
            "auto_login": self.auto_login,
            "timestamp": self.timestamp,
        }


class SteamLibraryFolder(BaseModel):
    """Represents a Steam library directory configured in libraryfolders.vdf."""
    deliberation: str = Field(default="", description="Epistemic rationale describing library folder discovery.")
    folder_id: int = 0
    path: str = ""
    label: str = ""
    total_size: int = 0
    apps: Dict[int, int] = Field(default_factory=dict)  # appid -> size in bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "folder_id": self.folder_id,
            "path": self.path,
            "label": self.label,
            "total_size": self.total_size,
            "apps": self.apps,
        }


class SteamAppManifest(BaseModel):
    """Represents an installed application manifest (appmanifest_<appid>.acf)."""
    deliberation: str = Field(default="", description="Epistemic rationale describing manifest parsing.")
    appid: int = 0
    name: str = ""
    installdir: str = ""
    size_on_disk: int = 0
    last_updated: int = 0
    bytes_to_download: int = 0
    bytes_downloaded: int = 0
    state_flags: int = 4  # 4 = fully installed
    build_id: int = 0
    last_owner: str = ""
    library_path: str = ""
    manifest_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "appid": self.appid,
            "name": self.name,
            "installdir": self.installdir,
            "size_on_disk": self.size_on_disk,
            "last_updated": self.last_updated,
            "bytes_to_download": self.bytes_to_download,
            "bytes_downloaded": self.bytes_downloaded,
            "state_flags": self.state_flags,
            "build_id": self.build_id,
            "last_owner": self.last_owner,
            "library_path": self.library_path,
            "manifest_path": self.manifest_path,
        }


class SteamAppPlaytime(BaseModel):
    """Represents playtime and last played metadata extracted from localconfig.vdf.
    
    MANDATORY RATIONALE-FIRST: deliberation declared first.
    """
    deliberation: str = Field(default="", description="Epistemic rationale describing playtime extraction.")
    appid: int = 0
    playtime_minutes: int = 0
    playtime_hours: float = 0.0
    playtime_2weeks_minutes: Optional[int] = None
    playtime_disconnected_minutes: Optional[int] = None
    last_played_timestamp: int = 0

    @property
    def playtime_2weeks_hours(self) -> Optional[float]:
        if self.playtime_2weeks_minutes is None:
            return None
        return round(self.playtime_2weeks_minutes / 60.0, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "appid": self.appid,
            "playtime_minutes": self.playtime_minutes,
            "playtime_hours": round(self.playtime_hours, 2),
            "playtime_2weeks_minutes": self.playtime_2weeks_minutes,
            "playtime_2weeks_hours": self.playtime_2weeks_hours,
            "playtime_disconnected_minutes": self.playtime_disconnected_minutes,
            "last_played_timestamp": self.last_played_timestamp,
        }


class SteamLocalLibrary(BaseModel):
    """Aggregated local Steam library state discovered across all paths on macOS."""
    deliberation: str = Field(default="", description="Epistemic rationale describing local library scan aggregation.")
    steam_root: str = ""
    users: List[SteamUserRecord] = Field(default_factory=list)
    active_user: Optional[SteamUserRecord] = None
    library_folders: List[SteamLibraryFolder] = Field(default_factory=list)
    installed_apps: Dict[int, SteamAppManifest] = Field(default_factory=dict)
    playtimes: Dict[int, SteamAppPlaytime] = Field(default_factory=dict)
    total_installed_bytes: int = 0
    total_playtime_hours: float = 0.0

    @property
    def all_app_ids(self) -> set[int]:
        """Union of all installed and played app IDs."""
        return set(self.installed_apps.keys()) | set(self.playtimes.keys())

    @property
    def total_games_discovered(self) -> int:
        return len(self.all_app_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "steam_root": self.steam_root,
            "users": [u.to_dict() for u in self.users],
            "active_user": self.active_user.to_dict() if self.active_user else None,
            "library_folders": [f.to_dict() for f in self.library_folders],
            "installed_apps": {str(k): v.to_dict() for k, v in self.installed_apps.items()},
            "playtimes": {str(k): v.to_dict() for k, v in self.playtimes.items()},
            "total_installed_bytes": self.total_installed_bytes,
            "total_playtime_hours": round(self.total_playtime_hours, 2),
            "total_games_discovered": self.total_games_discovered,
        }


class SteamOwnedGame(BaseModel):
    """Game metadata from Steam Web API IPlayerService/GetOwnedGames."""
    deliberation: str = Field(default="", description="Epistemic rationale describing owned game record.")
    appid: int = 0
    name: str = ""
    playtime_forever: int = 0  # minutes
    playtime_2weeks: Optional[int] = None
    playtime_mac_forever: Optional[int] = None
    playtime_windows_forever: Optional[int] = None
    playtime_linux_forever: Optional[int] = None
    rtime_last_played: int = 0
    img_icon_url: str = ""

    @property
    def playtime_hours(self) -> float:
        return self.playtime_forever / 60.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "appid": self.appid,
            "name": self.name,
            "playtime_forever": self.playtime_forever,
            "playtime_hours": round(self.playtime_hours, 2),
            "playtime_2weeks": self.playtime_2weeks,
            "playtime_mac_forever": self.playtime_mac_forever,
            "playtime_windows_forever": self.playtime_windows_forever,
            "playtime_linux_forever": self.playtime_linux_forever,
            "rtime_last_played": self.rtime_last_played,
            "img_icon_url": self.img_icon_url,
        }


class SteamAchievement(BaseModel):
    """Individual player achievement record from ISteamUserStats/GetPlayerAchievements."""
    deliberation: str = Field(default="", description="Epistemic rationale describing achievement record.")
    apiname: str = ""
    achieved: bool = False
    unlock_time: int = 0
    name: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "apiname": self.apiname,
            "achieved": self.achieved,
            "unlock_time": self.unlock_time,
            "name": self.name,
            "description": self.description,
        }


class SteamUserStats(BaseModel):
    """User gameplay statistics and achievements from ISteamUserStats/GetUserStatsForGame."""
    deliberation: str = Field(default="", description="Epistemic rationale describing user game stats.")
    steam_id: str = ""
    appid: int = 0
    game_name: str = ""
    stats: Dict[str, float] = Field(default_factory=dict)
    achievements: List[SteamAchievement] = Field(default_factory=list)

    @property
    def achievement_completion_rate(self) -> float:
        if not self.achievements:
            return 0.0
        achieved_count = sum(1 for a in self.achievements if a.achieved)
        return achieved_count / len(self.achievements)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "steam_id": self.steam_id,
            "appid": self.appid,
            "game_name": self.game_name,
            "stats": self.stats,
            "achievements": [a.to_dict() for a in self.achievements],
            "achievement_completion_rate": round(self.achievement_completion_rate, 3),
        }


class SteamAppDetails(BaseModel):
    """Rich store metadata and genre taxonomy from Steam Store API (store.steampowered.com/api/appdetails).
    
    MANDATORY RATIONALE-FIRST: deliberation declared first.
    """
    deliberation: str = Field(default="", description="Epistemic rationale describing store app details.")
    appid: int = 0
    name: str = ""
    is_free: bool = False
    genres: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    developers: List[str] = Field(default_factory=list)
    publishers: List[str] = Field(default_factory=list)
    release_date: str = ""
    short_description: str = ""
    header_image: str = ""
    metacritic_score: Optional[int] = None
    developer_tags: List[str] = Field(default_factory=list)
    cached_at_utc: Optional[str] = None
    source: str = "live_api"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "appid": self.appid,
            "name": self.name,
            "is_free": self.is_free,
            "genres": self.genres,
            "categories": self.categories,
            "developers": self.developers,
            "publishers": self.publishers,
            "release_date": self.release_date,
            "short_description": self.short_description,
            "header_image": self.header_image,
            "metacritic_score": self.metacritic_score,
            "developer_tags": self.developer_tags,
            "cached_at_utc": self.cached_at_utc,
            "source": self.source,
        }


class SteamGameRecord(BaseModel):
    """Normalized representation of a single Steam game entry for profiling."""
    deliberation: str = Field(default="", description="Epistemic rationale describing game record for DNA profiling.")
    appid: int = 0
    name: str = ""
    playtime_forever_minutes: int = 0
    playtime_2weeks_minutes: Optional[int] = None
    rtime_last_played: Optional[int] = None
    genres: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    achievements_total: int = 0
    achievements_unlocked: int = 0
    metacritic_score: Optional[int] = None

    @property
    def playtime_hours(self) -> float:
        return round(self.playtime_forever_minutes / 60.0, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "appid": self.appid,
            "name": self.name,
            "playtime_forever_minutes": self.playtime_forever_minutes,
            "playtime_hours": self.playtime_hours,
            "playtime_2weeks_minutes": self.playtime_2weeks_minutes,
            "rtime_last_played": self.rtime_last_played,
            "genres": self.genres,
            "tags": self.tags,
            "categories": self.categories,
            "achievements_total": self.achievements_total,
            "achievements_unlocked": self.achievements_unlocked,
            "metacritic_score": self.metacritic_score,
        }


class SteamLibraryDataset(BaseModel):
    """Collection of ingested Steam games for a player."""
    deliberation: str = Field(default="", description="Epistemic rationale describing the assembled dataset.")
    player_id: str = "local_player"
    source: str = "local_scan"  # "local_scan" | "web_api" | "fixture"
    scanned_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    games: List[SteamGameRecord] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "player_id": self.player_id,
            "source": self.source,
            "scanned_at_utc": self.scanned_at_utc,
            "games": [g.to_dict() for g in self.games],
        }


class SteamPlayerDNA(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation MUST be declared first!
    
    Represents behavioral, kinetic, and aesthetic DNA extracted from Steam history.
    """
    deliberation: str = Field(
        default="",
        description="Epistemic analysis of user library: Rationale-first deliberation analyzing Steam library history, pacing preferences, challenge tolerance, and aesthetic affinities."
    )
    player_id: str = Field(default="local_player", description="SteamID64 or local player profile identifier")
    profile_tag: str = Field(default="genesis_default", description="High-level profile archetype tag (e.g. genesis_rusher, genesis_tactician)")
    total_playtime_hours: float = Field(default=0.0, description="Summed total playtime across all analyzed games")
    total_games_analyzed: int = Field(default=0, description="Count of games contributing to DNA")
    genre_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="L1 normalized probability distribution across canonical genres (sum = 1.0)"
    )
    category_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="L1 normalized distribution across canonical Steam categories (sum = 1.0)"
    )
    playtime_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Normalized playtime distribution across 5 engagement tiers (dabbled_under_2h, casual_2_to_10h, engaged_10_to_50h, dedicated_50_to_100h, mastered_over_100h)"
    )
    pacing_preference: str = Field(
        default="balanced",
        description="'kinetic_high_mobility' | 'tactical_methodical' | 'balanced'"
    )
    pacing_score: float = Field(
        default=0.0,
        description="Continuous kinetic-tactical metric from -1.0 (pure tactical) to +1.0 (pure kinetic)"
    )
    challenge_tolerance: float = Field(
        default=0.5,
        description="Normalized challenge tolerance from 0.0 (casual) to 1.0 (hardcore masochist)"
    )
    thematic_biases: List[str] = Field(
        default_factory=list,
        description="Top 3 thematic keywords (e.g. sci-fi, dark fantasy, cyberpunk)"
    )
    thematic_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Normalized weights across detected thematic motifs"
    )
    mechanical_biases: Dict[str, float] = Field(
        default_factory=dict,
        description="Affinities in [0.0, 1.0] for dodge_roll, ranged_aim, close_combat, speedrun"
    )
    preferred_palette_mood: str = Field(
        default="retro_arcade",
        description="'neon_cyberpunk' | 'grim_dark' | 'retro_arcade' | 'vibrant_fantasy'"
    )
    palette_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Distribution over aesthetic palette moods"
    )
    top_genres: List[str] = Field(
        default_factory=list,
        description="Ordered list of top 3 genres by weight"
    )
    top_categories: List[str] = Field(
        default_factory=list,
        description="Ordered list of top 3 categories by weight"
    )
    top_games: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of top played/installed games with name, appid, playtime_hours, and genres"
    )
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "player_id": self.player_id,
            "profile_tag": self.profile_tag,
            "total_playtime_hours": self.total_playtime_hours,
            "total_games_analyzed": self.total_games_analyzed,
            "genre_weights": self.genre_weights,
            "category_weights": self.category_weights,
            "playtime_distribution": self.playtime_distribution,
            "pacing_preference": self.pacing_preference,
            "pacing_score": self.pacing_score,
            "challenge_tolerance": self.challenge_tolerance,
            "thematic_biases": self.thematic_biases,
            "thematic_weights": self.thematic_weights,
            "mechanical_biases": self.mechanical_biases,
            "preferred_palette_mood": self.preferred_palette_mood,
            "palette_weights": self.palette_weights,
            "top_genres": self.top_genres,
            "top_categories": self.top_categories,
            "top_games": self.top_games,
            "created_at_utc": self.created_at_utc,
        }
