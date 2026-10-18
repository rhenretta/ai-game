"""engine/steam/dna.py
Steam Player DNA extraction profiler, SQLite persistence repository, and MCP tool handler.
Enforces Rationale-First Epistemic Architecture (deliberation declared first in all classes).
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from config import DB_PATH
from engine.steam.models import (
    SteamAppDetails,
    SteamGameRecord,
    SteamLibraryDataset,
    SteamLocalLibrary,
    SteamOwnedGame,
    SteamPlayerDNA,
)

CANONICAL_GENRES: List[str] = [
    "Action",
    "Roguelike",
    "BulletHell",
    "RPG",
    "Strategy",
    "PrecisionPlatformer",
    "Shooter",
    "Simulation",
    "Puzzle",
    "Adventure",
]

CANONICAL_CATEGORIES: List[str] = [
    "Single-player",
    "Multi-player",
    "Co-op",
    "PvP",
    "Full controller support",
    "Steam Cloud",
    "Steam Achievements",
    "Remote Play",
]

CHALLENGE_TAG_LEXICON: set[str] = {
    "soulslike",
    "souls-like",
    "permadeath",
    "difficult",
    "hard",
    "unforgiving",
    "masochist",
    "bullet hell",
    "precision platformer",
    "hardcore",
}

THEMATIC_LEXICON: Dict[str, set[str]] = {
    "sci-fi": {"sci-fi", "science fiction", "space", "futuristic", "alien", "mechs", "spaceship"},
    "dark fantasy": {"dark fantasy", "grimdark", "gothic", "dark", "demons", "horror", "lovecraftian"},
    "cyberpunk": {"cyberpunk", "dystopian", "transhumanism", "synthwave", "neon"},
    "post-apocalyptic": {"post-apocalyptic", "apocalyptic", "wasteland", "survival", "nuclear"},
    "high fantasy": {"fantasy", "magic", "mythology", "medieval fantasy", "dragons"},
    "steampunk": {"steampunk", "victorian", "dieselpunk", "clockwork"},
    "retro arcade": {"retro", "pixel graphics", "arcade", "8-bit", "16-bit", "classic"},
    "military tactical": {"military", "war", "historical", "tactical shooter", "world war"},
}

PLAYTIME_TIER_NAMES: List[str] = [
    "dabbled_under_2h",
    "casual_2_to_10h",
    "engaged_10_to_50h",
    "dedicated_50_to_100h",
    "mastered_over_100h",
]

TAG_TO_GENRE_MAP: Dict[str, str] = {
    "roguelike": "Roguelike",
    "rogue-like": "Roguelike",
    "roguelite": "Roguelike",
    "rogue-lite": "Roguelike",
    "permadeath": "Roguelike",
    "dungeon crawler": "Roguelike",
    "bullet hell": "BulletHell",
    "shoot 'em up": "BulletHell",
    "shmup": "BulletHell",
    "danmaku": "BulletHell",
    "precision platformer": "PrecisionPlatformer",
    "2d platformer": "PrecisionPlatformer",
    "platformer": "PrecisionPlatformer",
    "hardcore": "PrecisionPlatformer",
    "action roguelike": "Action",
    "fast-paced": "Action",
    "hack and slash": "Action",
    "beat 'em up": "Action",
    "action": "Action",
    "rpg": "RPG",
    "action rpg": "RPG",
    "jrpg": "RPG",
    "strategy": "Strategy",
    "turn-based strategy": "Strategy",
    "grand strategy": "Strategy",
    "tactical": "Strategy",
    "turn-based": "Strategy",
    "rts": "Strategy",
    "card game": "Strategy",
    "deckbuilder": "Strategy",
    "shooter": "Shooter",
    "fps": "Shooter",
    "third-person shooter": "Shooter",
    "twin stick shooter": "Shooter",
    "puzzle": "Puzzle",
    "logic": "Puzzle",
    "simulation": "Simulation",
    "colony sim": "Simulation",
    "management": "Simulation",
    "adventure": "Adventure",
    "metroidvania": "Adventure",
    "exploration": "Adventure",
}


def compute_playtime_distribution(games: List[SteamGameRecord], total_hours: float) -> Dict[str, float]:
    """Computes normalized distribution of playtime across 5 engagement tiers."""
    tier_hours = {t: 0.0 for t in PLAYTIME_TIER_NAMES}
    for g in games:
        h = g.playtime_forever_minutes / 60.0
        if h <= 0.0:
            continue
        elif h < 2.0:
            tier_hours["dabbled_under_2h"] += h
        elif h < 10.0:
            tier_hours["casual_2_to_10h"] += h
        elif h < 50.0:
            tier_hours["engaged_10_to_50h"] += h
        elif h < 100.0:
            tier_hours["dedicated_50_to_100h"] += h
        else:
            tier_hours["mastered_over_100h"] += h

    if total_hours <= 0.0:
        return {t: 0.0 for t in PLAYTIME_TIER_NAMES}

    distribution = {t: round(tier_hours[t] / total_hours, 4) for t in PLAYTIME_TIER_NAMES}
    diff = 1.0 - sum(distribution.values())
    if abs(diff) > 1e-5:
        top_tier = max(distribution.items(), key=lambda kv: kv[1])[0]
        distribution[top_tier] = round(distribution[top_tier] + diff, 4)
    return distribution


def compute_category_weights(games: List[SteamGameRecord], game_weights: Dict[int, float]) -> Tuple[Dict[str, float], List[str]]:
    """Calculates L1 normalized category weights and sorted top 3 categories."""
    raw_cat_scores = {c: 0.0 for c in CANONICAL_CATEGORIES}
    for g in games:
        w = game_weights.get(g.appid, 1.0)
        matched = set()
        for cat in (g.categories + g.tags):
            norm = cat.strip().lower()
            for cc in CANONICAL_CATEGORIES:
                if cc.lower() == norm:
                    matched.add(cc)
        if matched:
            per_cat_w = w / len(matched)
            for cc in matched:
                raw_cat_scores[cc] += per_cat_w
        else:
            raw_cat_scores["Single-player"] += w * 0.70
            raw_cat_scores["Full controller support"] += w * 0.30

    tot = sum(raw_cat_scores.values())
    if tot > 0.0:
        weights = {c: round(raw_cat_scores[c] / tot, 4) for c in CANONICAL_CATEGORIES}
        diff = 1.0 - sum(weights.values())
        if abs(diff) > 1e-5:
            top_c = max(weights.items(), key=lambda kv: kv[1])[0]
            weights[top_c] = round(weights[top_c] + diff, 4)
    else:
        weights = {c: round(1.0 / len(CANONICAL_CATEGORIES), 4) for c in CANONICAL_CATEGORIES}

    top_categories = [k for k, _ in sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    return weights, top_categories


def compute_pacing_score(genre_weights: Dict[str, float]) -> Tuple[float, str]:
    """Calculates continuous pacing score bounded in [-1.0, +1.0] and preference label."""
    kinetic_sum = (
        genre_weights.get("Action", 0.0)
        + 1.25 * genre_weights.get("BulletHell", 0.0)
        + 1.0 * genre_weights.get("PrecisionPlatformer", 0.0)
        + 0.60 * genre_weights.get("Shooter", 0.0)
        + 0.30 * genre_weights.get("Roguelike", 0.0)
    )
    tactical_sum = (
        1.25 * genre_weights.get("Strategy", 0.0)
        + 1.0 * genre_weights.get("Puzzle", 0.0)
        + 0.60 * genre_weights.get("Simulation", 0.0)
        + 0.50 * genre_weights.get("RPG", 0.0)
    )
    denom = max(0.05, kinetic_sum + tactical_sum)
    score = round(max(-1.0, min(1.0, (kinetic_sum - tactical_sum) / denom)), 3)

    if score >= 0.20:
        pref = "kinetic_high_mobility"
    elif score <= -0.20:
        pref = "tactical_methodical"
    else:
        pref = "balanced"
    return score, pref


def compute_challenge_tolerance(
    games: List[SteamGameRecord],
    genre_weights: Dict[str, float],
    total_hours: float,
    game_weights: Dict[int, float]
) -> Tuple[float, str]:
    """Calculates challenge tolerance bounded in [0.0, 1.0] with tag difficulty and soulslike affinity."""
    hardcore_genre_affinity = min(
        1.0,
        genre_weights.get("Roguelike", 0.0)
        + genre_weights.get("BulletHell", 0.0)
        + genre_weights.get("PrecisionPlatformer", 0.0)
    )

    tag_matches = 0.0
    total_w = max(1e-5, sum(game_weights.values()))
    for g in games:
        w = game_weights.get(g.appid, 1.0)
        game_challenge_tags = sum(1 for t in g.tags if t.strip().lower() in CHALLENGE_TAG_LEXICON)
        if game_challenge_tags > 0:
            tag_matches += w * min(1.0, float(game_challenge_tags))
    tag_difficulty_affinity = min(1.0, tag_matches / total_w)

    playtime_depth = min(1.0, total_hours / 100.0)

    total_unlocked = sum(g.achievements_unlocked for g in games)
    total_possible = sum(g.achievements_total for g in games)

    if total_possible > 0:
        achievement_rate = min(1.0, total_unlocked / total_possible)
        raw_tol = (
            0.35 * hardcore_genre_affinity
            + 0.20 * tag_difficulty_affinity
            + 0.30 * achievement_rate
            + 0.15 * playtime_depth
        )
        summary = f"{achievement_rate:.1%} achievement rate ({total_unlocked}/{total_possible})"
    else:
        raw_tol = (
            0.45 * hardcore_genre_affinity
            + 0.40 * tag_difficulty_affinity
            + 0.15 * playtime_depth
        )
        summary = f"unindexed achievement profile, {tag_difficulty_affinity:.2f} difficulty tag affinity"

    return round(max(0.0, min(1.0, raw_tol)), 3), summary


def compute_thematic_biases(games: List[SteamGameRecord], game_weights: Dict[int, float]) -> Tuple[List[str], Dict[str, float]]:
    """Identifies top 3 thematic keywords (e.g. sci-fi, dark fantasy, cyberpunk) and normalized weights."""
    theme_scores = {theme: 0.0 for theme in THEMATIC_LEXICON}
    for g in games:
        w = game_weights.get(g.appid, 1.0)
        for item in (g.tags + g.genres):
            norm = item.strip().lower()
            for theme, keywords in THEMATIC_LEXICON.items():
                if norm in keywords or norm == theme:
                    theme_scores[theme] += w

    tot = sum(theme_scores.values())
    if tot > 0.0:
        weights = {t: round(theme_scores[t] / tot, 4) for t in THEMATIC_LEXICON}
        top_themes = [k for k, _ in sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    else:
        weights = {t: round(1.0 / len(THEMATIC_LEXICON), 4) for t in THEMATIC_LEXICON}
        top_themes = ["retro arcade", "sci-fi", "high fantasy"]

    return top_themes, weights


class SteamDNAProfiler:
    """Extracts a structured, rationale-first SteamPlayerDNA from a SteamLibraryDataset."""
    deliberation: str = "Extracting player DNA from gameplay engagement and genre distributions."

    def __init__(
        self,
        recent_playtime_bonus: float = 0.5,
        deliberation: str = "Extracting player DNA from gameplay engagement and genre distributions."
    ):
        self.deliberation = deliberation
        self.recent_bonus = recent_playtime_bonus

    def extract_dna(
        self,
        dataset: SteamLibraryDataset,
        profile_tag: str = "genesis_default"
    ) -> SteamPlayerDNA:
        """Computes multi-dimensional DNA using mathematical formulas and engagement weighting."""
        games = dataset.games
        total_playtime_hours = sum(max(0.0, g.playtime_forever_minutes) for g in games) / 60.0
        n_games = len(games)

        if n_games == 0:
            return self._build_default_dna(dataset.player_id, profile_tag)

        # 1. Compute effective engagement weight per game
        # w_g = ln(1 + hours) * (1 + bonus * min(1.0, recent_hours / 10.0))
        game_weights: Dict[int, float] = {}
        for g in games:
            hours = max(0.0, g.playtime_forever_minutes / 60.0)
            base_w = math.log(1.0 + hours)
            recent_hours = max(0.0, (g.playtime_2weeks_minutes or 0) / 60.0)
            recent_factor = 1.0 + (self.recent_bonus * min(1.0, recent_hours / 10.0))
            game_weights[g.appid] = base_w * recent_factor

        # 2. Accumulate canonical genre scores
        raw_genre_scores: Dict[str, float] = {cg: 0.0 for cg in CANONICAL_GENRES}
        any_genre_metadata = any(bool(g.genres or g.tags) for g in games)
        for g in games:
            w = game_weights[g.appid]
            matched_genres = set()
            for item in (g.genres + g.tags):
                norm = item.strip().lower()
                if norm in TAG_TO_GENRE_MAP:
                    matched_genres.add(TAG_TO_GENRE_MAP[norm])
                for cg in CANONICAL_GENRES:
                    if cg.lower() == norm:
                        matched_genres.add(cg)

            # Distribute game weight across matched canonical genres
            if matched_genres:
                per_genre_w = w / len(matched_genres)
                for cg in matched_genres:
                    raw_genre_scores[cg] += per_genre_w
            elif not any_genre_metadata:
                raw_genre_scores["Action"] += w * 0.5
                raw_genre_scores["Adventure"] += w * 0.5

        if sum(raw_genre_scores.values()) == 0.0:
            for g in games:
                w = game_weights[g.appid]
                raw_genre_scores["Action"] += w * 0.5
                raw_genre_scores["Adventure"] += w * 0.5

        # L1 Normalize genre weights
        total_genre_score = sum(raw_genre_scores.values())
        if total_genre_score > 0.0:
            genre_weights = {
                cg: round(raw_genre_scores[cg] / total_genre_score, 4)
                for cg in CANONICAL_GENRES
            }
        else:
            genre_weights = {cg: round(1.0 / len(CANONICAL_GENRES), 4) for cg in CANONICAL_GENRES}

        # Ensure exact L1 normalization sum = 1.0
        diff = 1.0 - sum(genre_weights.values())
        if abs(diff) > 1e-5:
            top_g = max(genre_weights.items(), key=lambda kv: kv[1])[0]
            genre_weights[top_g] = round(genre_weights[top_g] + diff, 4)

        # 3. Compute continuous Pacing Score & Preference
        pacing_score, pacing_preference = compute_pacing_score(genre_weights)

        # 4. Compute Challenge Tolerance
        challenge_tolerance, achieve_summary = compute_challenge_tolerance(
            games, genre_weights, total_playtime_hours, game_weights
        )

        # 5. Compute Playtime Distribution (5 Tiers)
        playtime_dist = compute_playtime_distribution(games, total_playtime_hours)

        # 6. Compute Category Weights & Top Categories
        category_weights, top_categories = compute_category_weights(games, game_weights)

        # 7. Compute Thematic Biases & Weights
        thematic_biases, thematic_weights = compute_thematic_biases(games, game_weights)

        # 8. Calculate Mechanical Biases [0.0, 1.0]
        mechanical_biases = {
            "dodge_roll": round(min(1.0, 2.5 * (0.45 * genre_weights["BulletHell"] + 0.35 * genre_weights["Roguelike"] + 0.20 * genre_weights["Action"])), 3),
            "ranged_aim": round(min(1.0, 2.0 * (0.65 * genre_weights["Shooter"] + 0.35 * genre_weights["BulletHell"])), 3),
            "close_combat": round(min(1.0, 2.0 * (0.65 * genre_weights["Action"] + 0.35 * genre_weights["RPG"])), 3),
            "speedrun": round(min(1.0, 2.0 * (0.80 * genre_weights["PrecisionPlatformer"] + 0.20 * genre_weights["Action"])), 3),
        }

        # 9. Calculate Palette Mood Weights
        dark_theme_weight = thematic_weights.get("dark fantasy", 0.0) + thematic_weights.get("post-apocalyptic", 0.0)
        neon_score = (
            0.40 * genre_weights["Shooter"]
            + 0.35 * genre_weights["Action"]
            + 0.25 * max(0.0, pacing_score)
        )
        grim_score = (
            0.45 * genre_weights["Roguelike"]
            + 0.35 * max(0.0, dark_theme_weight - 0.25)
            + 0.15 * genre_weights["Strategy"]
        )
        retro_score = (
            0.50 * genre_weights["BulletHell"]
            + 0.50 * genre_weights["PrecisionPlatformer"]
        )
        vibrant_score = (
            0.45 * genre_weights["RPG"]
            + 0.40 * genre_weights["Adventure"]
            + 0.45 * genre_weights["Simulation"]
            + 0.35 * genre_weights["Puzzle"]
            + 0.25 * genre_weights["Strategy"]
        )

        mood_scores = {
            "neon_cyberpunk": max(0.01, neon_score),
            "grim_dark": max(0.01, grim_score),
            "retro_arcade": max(0.01, retro_score),
            "vibrant_fantasy": max(0.01, vibrant_score),
        }
        total_mood = sum(mood_scores.values())
        palette_weights = {k: round(v / total_mood, 4) for k, v in mood_scores.items()}
        preferred_palette_mood = max(palette_weights.items(), key=lambda kv: kv[1])[0]

        # Sorted top genres
        top_genres = [k for k, v in sorted(genre_weights.items(), key=lambda kv: kv[1], reverse=True) if v > 0][:3]

        # Build top games list (prioritizing resolved named titles sorted by playtime)
        sorted_games = sorted(
            games,
            key=lambda g: (not g.name.startswith("App "), g.playtime_forever_minutes),
            reverse=True,
        )
        top_games: List[Dict[str, Any]] = [
            {
                "appid": g.appid,
                "name": g.name,
                "playtime_hours": round(g.playtime_forever_minutes / 60.0, 2),
                "genres": list(g.genres),
                "tags": list(g.tags[:5]),
            }
            for g in sorted_games[:12]
        ]
        named_top_games = [tg for tg in top_games if not str(tg["name"]).startswith("App ")]
        top_games_summary = ""
        if named_top_games:
            top_games_summary = " Top titles: " + ", ".join(
                f"{tg['name']} ({tg['playtime_hours']:.1f}h)" for tg in named_top_games[:6]
            ) + "."

        # 10. Formulate Epistemic Deliberation (Reasoning first!)
        top_genre_desc = ", ".join(f"{g} ({genre_weights[g]:.1%})" for g in top_genres)
        deliberation = (
            f"Analyzed {n_games} Steam titles totaling {total_playtime_hours:.1f} hours of logged playtime.{top_games_summary} "
            f"Primary genre affinities identified: {top_genre_desc}. "
            f"Evaluated continuous pacing index at {pacing_score:+.2f}, establishing a '{pacing_preference}' behavioral profile. "
            f"Challenge tolerance calculated at {challenge_tolerance:.2f} based on {achieve_summary} and hardcore gameplay exposure. "
            f"Dominant mechanical biases emerge for {max(mechanical_biases.items(), key=lambda kv: kv[1])[0]} ({max(mechanical_biases.values()):.2f}). "
            f"Aesthetic palette resolves to '{preferred_palette_mood}' with {palette_weights[preferred_palette_mood]:.1%} affinity. "
            f"Top thematic biases identified: {', '.join(thematic_biases)}."
        )

        return SteamPlayerDNA(
            deliberation=deliberation,
            player_id=dataset.player_id,
            profile_tag=profile_tag,
            total_playtime_hours=round(total_playtime_hours, 2),
            total_games_analyzed=n_games,
            genre_weights=genre_weights,
            category_weights=category_weights,
            playtime_distribution=playtime_dist,
            pacing_preference=pacing_preference,
            pacing_score=pacing_score,
            challenge_tolerance=challenge_tolerance,
            thematic_biases=thematic_biases,
            thematic_weights=thematic_weights,
            mechanical_biases=mechanical_biases,
            preferred_palette_mood=preferred_palette_mood,
            palette_weights=palette_weights,
            top_genres=top_genres,
            top_categories=top_categories,
            top_games=top_games,
        )

    def _build_default_dna(self, player_id: str, profile_tag: str) -> SteamPlayerDNA:
        equal_genre = round(1.0 / len(CANONICAL_GENRES), 4)
        equal_cat = round(1.0 / len(CANONICAL_CATEGORIES), 4)
        equal_theme = round(1.0 / len(THEMATIC_LEXICON), 4)

        deliberation = (
            f"Epistemic analysis of user library: No local Steam manifests or gameplay records detected "
            f"for player '{player_id}' at configured Steam paths. Operating in zero-library fallback mode: "
            f"initializing balanced baseline profile with uniform genre distribution, neutral pacing "
            f"(pacing_score=0.00), moderate challenge tolerance (0.50), balanced categories, and "
            f"retro_arcade aesthetic defaults for Episode 0 Genesis."
        )
        return SteamPlayerDNA(
            deliberation=deliberation,
            player_id=player_id,
            profile_tag=profile_tag,
            total_playtime_hours=0.0,
            total_games_analyzed=0,
            genre_weights={cg: equal_genre for cg in CANONICAL_GENRES},
            category_weights={cc: equal_cat for cc in CANONICAL_CATEGORIES},
            playtime_distribution={t: 0.0 for t in PLAYTIME_TIER_NAMES},
            pacing_preference="balanced",
            pacing_score=0.0,
            challenge_tolerance=0.5,
            thematic_biases=["retro arcade", "sci-fi", "high fantasy"],
            thematic_weights={t: equal_theme for t in THEMATIC_LEXICON},
            mechanical_biases={"dodge_roll": 0.5, "ranged_aim": 0.5, "close_combat": 0.5, "speedrun": 0.5},
            preferred_palette_mood="retro_arcade",
            palette_weights={"neon_cyberpunk": 0.25, "grim_dark": 0.25, "retro_arcade": 0.25, "vibrant_fantasy": 0.25},
            top_genres=["Action", "Adventure", "Strategy"],
            top_categories=["Single-player", "Full controller support", "Steam Cloud"],
            top_games=[],
        )


class SteamDNAMapper:
    """Maps between Steam ingestion models and profiling datasets."""
    deliberation: str = "Mapping Steam scanner and Web API collections into SteamLibraryDataset."

    def __init__(self, deliberation: str = "Mapping Steam scanner and Web API collections into SteamLibraryDataset."):
        self.deliberation = deliberation

    @staticmethod
    def from_local_library(
        local_lib: SteamLocalLibrary,
        details_map: Optional[Dict[int, SteamAppDetails]] = None
    ) -> SteamLibraryDataset:
        """Converts SteamLocalLibrary into a SteamLibraryDataset, enriching with store details if available."""
        player_id = local_lib.active_user.steam_id_64 if local_lib.active_user else "local_player"
        records: List[SteamGameRecord] = []

        all_appids = set(local_lib.installed_apps.keys()) | set(local_lib.playtimes.keys())
        for folder in local_lib.library_folders:
            all_appids.update(folder.apps.keys())

        for appid in sorted(all_appids):
            manifest = local_lib.installed_apps.get(appid)
            name = manifest.name if (manifest and manifest.name) else f"App {appid}"
            pt = local_lib.playtimes.get(appid)
            pt_min = pt.playtime_minutes if pt else 0
            pt_2w = pt.playtime_2weeks_minutes if pt else None
            lp_time = pt.last_played_timestamp if pt else None

            genres: List[str] = []
            categories: List[str] = []
            tags: List[str] = []
            if details_map and appid in details_map and details_map[appid] is not None:
                det = details_map[appid]
                if det.name and (not manifest or not manifest.name):
                    name = det.name
                genres = det.genres
                categories = det.categories
                tags = det.developer_tags

            records.append(
                SteamGameRecord(
                    deliberation=f"Mapped local app {name} ({appid})",
                    appid=appid,
                    name=name,
                    playtime_forever_minutes=pt_min,
                    playtime_2weeks_minutes=pt_2w,
                    rtime_last_played=lp_time,
                    genres=genres,
                    categories=categories,
                    tags=tags,
                )
            )

        return SteamLibraryDataset(
            deliberation=f"Mapped local library with {len(records)} games for player {player_id}.",
            player_id=player_id,
            source="local_scan",
            games=records,
        )

    @staticmethod
    def from_web_api(
        owned_games: List[SteamOwnedGame],
        details_map: Optional[Dict[int, SteamAppDetails]] = None,
        player_id: str = "local_player"
    ) -> SteamLibraryDataset:
        """Converts a list of SteamOwnedGame into a SteamLibraryDataset."""
        records: List[SteamGameRecord] = []

        for g in owned_games:
            genres: List[str] = []
            categories: List[str] = []
            tags: List[str] = []
            if details_map and g.appid in details_map:
                det = details_map[g.appid]
                genres = det.genres
                categories = det.categories
                tags = det.developer_tags

            records.append(
                SteamGameRecord(
                    deliberation=f"Mapped Web API game {g.name} ({g.appid})",
                    appid=g.appid,
                    name=g.name,
                    playtime_forever_minutes=g.playtime_forever,
                    playtime_2weeks_minutes=g.playtime_2weeks,
                    rtime_last_played=g.rtime_last_played,
                    genres=genres,
                    categories=categories,
                    tags=tags,
                )
            )

        return SteamLibraryDataset(
            deliberation=f"Mapped Web API dataset with {len(records)} games for player {player_id}.",
            player_id=player_id,
            source="web_api",
            games=records,
        )


class SteamDNAPersistence:
    """Manages SQLite storage and retrieval for SteamPlayerDNA records in WAL mode."""
    deliberation: str = "Persisting and retrieving SteamPlayerDNA records in SQLite WAL mode."

    SCHEMA_DDL = """
    CREATE TABLE IF NOT EXISTS steam_player_dna (
        dna_id TEXT PRIMARY KEY,
        player_id TEXT NOT NULL,
        session_id TEXT,
        created_at_utc TEXT NOT NULL,
        profile_tag TEXT NOT NULL,
        deliberation TEXT NOT NULL,
        total_playtime_hours REAL NOT NULL,
        total_games_analyzed INTEGER NOT NULL,
        pacing_preference TEXT NOT NULL,
        pacing_score REAL NOT NULL,
        challenge_tolerance REAL NOT NULL,
        preferred_palette_mood TEXT NOT NULL,
        genre_weights_json TEXT NOT NULL,
        category_weights_json TEXT NOT NULL DEFAULT '{}',
        playtime_distribution_json TEXT NOT NULL DEFAULT '{}',
        thematic_biases_json TEXT NOT NULL DEFAULT '[]',
        thematic_weights_json TEXT NOT NULL DEFAULT '{}',
        mechanical_biases_json TEXT NOT NULL,
        palette_weights_json TEXT NOT NULL,
        raw_dna_json TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES telemetry_sessions(session_id)
    );

    CREATE INDEX IF NOT EXISTS idx_dna_player_id ON steam_player_dna(player_id);
    CREATE INDEX IF NOT EXISTS idx_dna_created_at ON steam_player_dna(created_at_utc);
    CREATE INDEX IF NOT EXISTS idx_dna_profile_tag ON steam_player_dna(profile_tag);
    """

    def __init__(
        self,
        db_path: Optional[Union[Path, str]] = None,
        deliberation: str = "Persisting and retrieving SteamPlayerDNA records in SQLite WAL mode."
    ):
        import os
        import sys
        import tempfile
        import config as _cfg
        self.deliberation = deliberation
        resolved = Path(db_path) if db_path is not None else Path(os.environ.get("STUDIO_DB_PATH", str(_cfg.DB_PATH)))
        prod_db = (_cfg.PROJECT_ROOT / "data" / "telemetry.db").resolve()
        if ("PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules) and resolved.resolve() == prod_db:
            env_override = os.environ.get("STUDIO_DB_PATH")
            if env_override and Path(env_override).resolve() != prod_db:
                resolved = Path(env_override)
            else:
                resolved = Path(tempfile.gettempdir()) / f"pytest_steam_dna_{os.getpid()}.db"
        self.db_path = resolved
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        """Applies WAL pragma, foreign keys, and ensures the table schema exists."""
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(self.SCHEMA_DDL)

            # Defensive migration for existing SQLite databases
            for col_name, col_type in (
                ("category_weights_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("playtime_distribution_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("thematic_biases_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("thematic_weights_json", "TEXT NOT NULL DEFAULT '{}'"),
            ):
                try:
                    conn.execute(f"ALTER TABLE steam_player_dna ADD COLUMN {col_name} {col_type};")
                except sqlite3.OperationalError:
                    pass

            conn.commit()
        finally:
            conn.close()

    def save_dna(self, dna: SteamPlayerDNA, session_id: Optional[str] = None) -> str:
        """Stores a SteamPlayerDNA profile and returns its unique dna_id."""
        dna_id = f"dna_{uuid.uuid4().hex[:12]}"
        raw_json = dna.model_dump_json()

        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO steam_player_dna (
                    dna_id, player_id, session_id, created_at_utc, profile_tag,
                    deliberation, total_playtime_hours, total_games_analyzed,
                    pacing_preference, pacing_score, challenge_tolerance,
                    preferred_palette_mood, genre_weights_json, category_weights_json,
                    playtime_distribution_json, thematic_biases_json, thematic_weights_json,
                    mechanical_biases_json, palette_weights_json, raw_dna_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    dna_id,
                    dna.player_id,
                    session_id,
                    dna.created_at_utc,
                    dna.profile_tag,
                    dna.deliberation,
                    dna.total_playtime_hours,
                    dna.total_games_analyzed,
                    dna.pacing_preference,
                    dna.pacing_score,
                    dna.challenge_tolerance,
                    dna.preferred_palette_mood,
                    json.dumps(dna.genre_weights),
                    json.dumps(dna.category_weights),
                    json.dumps(dna.playtime_distribution),
                    json.dumps(dna.thematic_biases),
                    json.dumps(dna.thematic_weights),
                    json.dumps(dna.mechanical_biases),
                    json.dumps(dna.palette_weights),
                    raw_json,
                ),
            )
            conn.commit()
            return dna_id
        finally:
            conn.close()

    def get_latest_dna(
        self,
        player_id: Optional[str] = None,
        profile_tag: Optional[str] = None
    ) -> Optional[SteamPlayerDNA]:
        """Retrieves the most recent SteamPlayerDNA profile matching criteria."""
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            query = "SELECT raw_dna_json FROM steam_player_dna"
            params: List[Any] = []
            clauses = []
            if player_id:
                clauses.append("player_id = ?")
                params.append(player_id)
            if profile_tag:
                clauses.append("profile_tag = ?")
                params.append(profile_tag)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY created_at_utc DESC LIMIT 1;"

            cursor.execute(query, params)
            row = cursor.fetchone()
            if row:
                data = json.loads(row["raw_dna_json"])
                return SteamPlayerDNA.model_validate(data)
            return None
        finally:
            conn.close()

    def list_dna_history(
        self,
        player_id: Optional[str] = None,
        limit: int = 10
    ) -> List[SteamPlayerDNA]:
        """Lists historical SteamPlayerDNA records ordered by creation timestamp."""
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            query = "SELECT raw_dna_json FROM steam_player_dna"
            params: List[Any] = []
            if player_id:
                query += " WHERE player_id = ?"
                params.append(player_id)
            query += " ORDER BY created_at_utc DESC LIMIT ?;"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [SteamPlayerDNA.model_validate(json.loads(r["raw_dna_json"])) for r in rows]
        finally:
            conn.close()


# Compatibility alias
SteamDNARepository = SteamDNAPersistence


def build_rusher_dataset() -> SteamLibraryDataset:
    """Constructs a deterministic synthetic SteamLibraryDataset for a high-mobility kinetic action rusher."""
    return SteamLibraryDataset(
        deliberation="Simulated high-mobility kinetic action rusher with Hades, Dead Cells, Enter the Gungeon, and ULTRAKILL.",
        player_id="steam_user_rusher_001",
        source="fixture",
        games=[
            SteamGameRecord(
                appid=1145360,
                name="Hades",
                playtime_forever_minutes=7200,   # 120 hrs
                playtime_2weeks_minutes=600,     # 10 hrs
                genres=["Action", "Roguelike"],
                tags=["fast-paced", "hack and slash", "bullet hell", "permadeath", "difficult", "hardcore"],
                categories=["Single-player", "Full controller support", "Steam Achievements"],
                achievements_total=49,
                achievements_unlocked=48,
            ),
            SteamGameRecord(
                appid=588650,
                name="Dead Cells",
                playtime_forever_minutes=4800,   # 80 hrs
                genres=["Action", "Roguelike"],
                tags=["action roguelike", "fast-paced", "permadeath", "difficult", "cyberpunk", "soulslike"],
                categories=["Single-player", "Full controller support", "Steam Cloud"],
                achievements_total=100,
                achievements_unlocked=96,
            ),
            SteamGameRecord(
                appid=311690,
                name="Enter the Gungeon",
                playtime_forever_minutes=4800,   # 80 hrs
                genres=["BulletHell", "Roguelike"],
                tags=["bullet hell", "shoot 'em up", "difficult", "retro", "arcade", "hardcore"],
                categories=["Single-player", "Co-op", "Full controller support"],
                achievements_total=55,
                achievements_unlocked=53,
            ),
            SteamGameRecord(
                appid=1229490,
                name="ULTRAKILL",
                playtime_forever_minutes=3600,   # 60 hrs
                genres=["Action", "Shooter", "PrecisionPlatformer"],
                tags=["fast-paced", "cyberpunk", "difficult", "retro", "hardcore", "permadeath"],
                categories=["Single-player"],
                achievements_total=30,
                achievements_unlocked=29,
            ),
        ],
    )


def build_cautious_dataset() -> SteamLibraryDataset:
    """Constructs a deterministic synthetic SteamLibraryDataset for a methodical tactical strategist."""
    return SteamLibraryDataset(
        deliberation="Simulated methodical tactical strategist with Civ VI, Slay the Spire, Into the Breach, Baba Is You, and Timberborn.",
        player_id="steam_user_tactician_002",
        source="fixture",
        games=[
            SteamGameRecord(
                appid=289070,
                name="Sid Meier's Civilization VI",
                playtime_forever_minutes=9000,  # 150 hrs
                playtime_2weeks_minutes=300,
                genres=["Strategy"],
                tags=["turn-based strategy", "grand strategy", "tactical", "historical"],
                categories=["Single-player", "Multi-player", "Steam Cloud"],
                achievements_total=100,
                achievements_unlocked=35,
            ),
            SteamGameRecord(
                appid=646570,
                name="Slay the Spire",
                playtime_forever_minutes=4800,   # 80 hrs
                genres=["Strategy"],
                tags=["card game", "deckbuilder", "turn-based", "dark fantasy"],
                categories=["Single-player", "Steam Achievements"],
                achievements_total=46,
                achievements_unlocked=18,
            ),
            SteamGameRecord(
                appid=590380,
                name="Into the Breach",
                playtime_forever_minutes=3000,   # 50 hrs
                genres=["Strategy", "Puzzle"],
                tags=["turn-based", "tactical", "puzzle", "sci-fi", "mechs"],
                categories=["Single-player", "Steam Cloud"],
                achievements_total=50,
                achievements_unlocked=20,
            ),
            SteamGameRecord(
                appid=736260,
                name="Baba Is You",
                playtime_forever_minutes=1800,   # 30 hrs
                genres=["Puzzle"],
                tags=["puzzle", "logic", "indie"],
                categories=["Single-player"],
                achievements_total=19,
                achievements_unlocked=6,
            ),
            SteamGameRecord(
                appid=1062090,
                name="Timberborn",
                playtime_forever_minutes=4800,   # 80 hrs
                genres=["Simulation", "Strategy"],
                tags=["city builder", "building", "management", "survival", "sandbox"],
                categories=["Single-player", "Steam Cloud"],
                achievements_total=30,
                achievements_unlocked=12,
            ),
        ],
    )


def build_rusher_dna(player_id: str = "steam_user_rusher_001") -> SteamPlayerDNA:
    """Generates a pre-compiled high-kinetic, high-challenge SteamPlayerDNA profile fixture."""
    dataset = build_rusher_dataset()
    dataset.player_id = player_id
    profiler = SteamDNAProfiler()
    dna = profiler.extract_dna(dataset, profile_tag="genesis_rusher")
    return dna


def build_cautious_dna(player_id: str = "steam_user_tactician_002") -> SteamPlayerDNA:
    """Generates a pre-compiled methodical, low-challenge SteamPlayerDNA profile fixture."""
    dataset = build_cautious_dataset()
    dataset.player_id = player_id
    profiler = SteamDNAProfiler()
    dna = profiler.extract_dna(dataset, profile_tag="genesis_tactician")
    return dna


RUSHER_DNA_FIXTURE: SteamPlayerDNA = build_rusher_dna()
CAUTIOUS_DNA_FIXTURE: SteamPlayerDNA = build_cautious_dna()

# Contrasting profile aliases required by Feature 12
RUSHER_PROFILE: SteamPlayerDNA = RUSHER_DNA_FIXTURE
CAUTIOUS_PROFILE: SteamPlayerDNA = CAUTIOUS_DNA_FIXTURE


def handle_get_steam_player_dna(
    repository: SteamDNAPersistence,
    profiler: SteamDNAProfiler,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Handles JSON-RPC 2.0 tool execution for get_steam_player_dna."""
    caller_deliberation = params.get("deliberation", "")
    if not caller_deliberation:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32602,
                "message": "Missing mandatory deliberation parameter. Agent must deliberate before querying Steam Player DNA."
            },
            "id": params.get("id", 1)
        }

    args = params.get("arguments", {})
    player_id = args.get("player_id")
    profile_tag = args.get("profile_tag")

    if profile_tag in ("rusher", "genesis_rusher"):
        dna = build_rusher_dna(player_id or "steam_user_rusher_001")
    elif profile_tag in ("cautious", "genesis_tactician"):
        dna = build_cautious_dna(player_id or "steam_user_tactician_002")
    else:
        dna = repository.get_latest_dna(player_id=player_id, profile_tag=profile_tag)
        if dna is None:
            dna = profiler._build_default_dna(player_id or "local_player", profile_tag or "genesis_default")
            repository.save_dna(dna)

    dna_dict = dna.to_dict()
    return {
        "jsonrpc": "2.0",
        "result": dna_dict,
        "id": params.get("id", 1)
    }


# Convenience alias for tool router
get_steam_player_dna = handle_get_steam_player_dna


def scan_and_extract_local_dna(
    db_path: Union[Path, str] = DB_PATH,
    profile_tag: str = "default",
    max_store_lookups: int = 18,
    persist: bool = True,
) -> Optional[SteamPlayerDNA]:
    """Scans the local macOS Steam installation, enriches top-played and installed games
    via SteamStoreApiClient (with persistent disk caching), computes SteamPlayerDNA,
    and persists to SQLite WAL.
    """
    if profile_tag in ("rusher", "genesis_rusher"):
        dna = build_rusher_dna()
        if persist:
            try:
                SteamDNAPersistence(db_path=db_path).save_dna(dna)
            except Exception:
                pass
        return dna
    if profile_tag in ("cautious", "genesis_tactician"):
        dna = build_cautious_dna()
        if persist:
            try:
                SteamDNAPersistence(db_path=db_path).save_dna(dna)
            except Exception:
                pass
        return dna

    from engine.steam.scanner import SteamScanner
    from engine.steam.store_api import SteamRateLimiter, SteamStoreApiClient

    scanner = SteamScanner()
    if not scanner.is_steam_installed():
        return None

    local_lib = scanner.scan_full_library()
    if not local_lib.installed_apps and not local_lib.playtimes:
        return None

    # Identify top-played AppIDs + all installed AppIDs for Store API enrichment
    NON_GAME_UTILITY_APPIDS = {250820, 228980}  # SteamVR, Steamworks Common Redistributables
    top_playtimes = sorted(
        [p for p in local_lib.playtimes.values() if p.appid not in NON_GAME_UTILITY_APPIDS],
        key=lambda x: x.playtime_minutes,
        reverse=True,
    )[:max_store_lookups]
    target_appids = list(
        {p.appid for p in top_playtimes}
        | {aid for aid in local_lib.installed_apps.keys() if aid not in NON_GAME_UTILITY_APPIDS}
    )

    details_map: Dict[int, SteamAppDetails] = {}
    try:
        store_client = SteamStoreApiClient(rate_limiter=SteamRateLimiter(min_interval=0.1))
        for appid in target_appids:
            try:
                det = store_client.get_app_details(appid)
                if det is not None:
                    details_map[appid] = det
            except Exception:
                continue
    except Exception:
        pass

    dataset = SteamDNAMapper.from_local_library(local_lib, details_map=details_map)
    # Remove known non-game utility AppIDs from dataset.games so they don't appear in top_games
    dataset.games = [g for g in dataset.games if g.appid not in NON_GAME_UTILITY_APPIDS]
    if not dataset.games:
        return None

    profiler = SteamDNAProfiler()
    dna = profiler.extract_dna(dataset, profile_tag=profile_tag or "default")

    if persist:
        try:
            repo = SteamDNAPersistence(db_path=db_path)
            repo.save_dna(dna)
        except Exception:
            pass

    return dna

