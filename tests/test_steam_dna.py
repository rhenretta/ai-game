"""tests/test_steam_dna.py
Comprehensive Unit and Integration Test Suite for Steam Player DNA Profiler & Persistence.
Verifies:
  1. Rationale-First Schema Invariants (deliberation declared first in AST and serialized first).
  2. Profiling algorithms: engagement weighting, L1 genre normalization, continuous pacing, challenge tolerance, mechanical biases, and palette mood.
  3. Behavioral divergence between Rusher and Tactician profiles.
  4. Graceful degradation on empty datasets.
  5. SQLite WAL persistence, indexing, and roundtrip serialization.
  6. Local MCP tool hook (get_steam_player_dna) with mandatory deliberation gating.
"""
from pathlib import Path
import sqlite3
import pytest

from engine.steam.dna import (
    CANONICAL_CATEGORIES,
    CANONICAL_GENRES,
    CAUTIOUS_PROFILE,
    CHALLENGE_TAG_LEXICON,
    PLAYTIME_TIER_NAMES,
    RUSHER_PROFILE,
    THEMATIC_LEXICON,
    SteamDNAMapper,
    SteamDNAPersistence,
    SteamDNAProfiler,
    SteamDNARepository,
    build_cautious_dataset,
    build_cautious_dna,
    build_rusher_dataset,
    build_rusher_dna,
    get_steam_player_dna,
    handle_get_steam_player_dna,
)
from engine.steam.models import (
    SteamAppDetails,
    SteamGameRecord,
    SteamLibraryDataset,
    SteamLocalLibrary,
    SteamOwnedGame,
    SteamPlayerDNA,
)
from tests.test_tier1_units import validate_rationale_first_ast


class TestSteamDNARationaleFirst:
    """Verifies that SteamPlayerDNA adheres strictly to Rationale-First Epistemic Architecture."""

    def test_dna_models_ast_compliance(self):
        root = Path(__file__).parent.parent
        models_file = root / "engine" / "steam" / "models.py"
        dna_file = root / "engine" / "steam" / "dna.py"

        assert validate_rationale_first_ast(models_file.read_text(encoding="utf-8")) is True
        assert validate_rationale_first_ast(dna_file.read_text(encoding="utf-8")) is True

    def test_dna_serialization_deliberation_first(self):
        profiler = SteamDNAProfiler()
        empty_dataset = SteamLibraryDataset(player_id="test_player", games=[])
        dna = profiler.extract_dna(empty_dataset)

        dump = dna.model_dump()
        assert list(dump.keys())[0] == "deliberation"

        d = dna.to_dict()
        assert list(d.keys())[0] == "deliberation"


class TestSteamDNAProfiling:
    """Verifies behavioral profiling calculations and behavioral divergence."""

    @pytest.fixture
    def rusher_dataset(self) -> SteamLibraryDataset:
        return SteamLibraryDataset(
            player_id="76561198000000001",
            source="fixture",
            games=[
                SteamGameRecord(
                    appid=1145360,
                    name="Hades",
                    playtime_forever_minutes=7200,   # 120 hrs
                    playtime_2weeks_minutes=600,     # 10 hrs
                    genres=["Action", "Roguelike"],
                    tags=["fast-paced", "hack and slash", "bullet hell"],
                    achievements_total=49,
                    achievements_unlocked=45,
                ),
                SteamGameRecord(
                    appid=588650,
                    name="Dead Cells",
                    playtime_forever_minutes=4800,   # 80 hrs
                    genres=["Action", "Roguelike"],
                    tags=["metroidvania", "action roguelike"],
                    achievements_total=100,
                    achievements_unlocked=80,
                ),
                SteamGameRecord(
                    appid=504230,
                    name="Celeste",
                    playtime_forever_minutes=2400,   # 40 hrs
                    genres=["Action"],
                    tags=["precision platformer", "2d platformer", "hardcore"],
                    achievements_total=32,
                    achievements_unlocked=28,
                ),
                SteamGameRecord(
                    appid=311690,
                    name="Enter the Gungeon",
                    playtime_forever_minutes=3600,   # 60 hrs
                    genres=["Action", "Roguelike"],
                    tags=["bullet hell", "shoot 'em up", "twin stick shooter"],
                    achievements_total=55,
                    achievements_unlocked=48,
                ),
            ],
        )

    @pytest.fixture
    def tactician_dataset(self) -> SteamLibraryDataset:
        return SteamLibraryDataset(
            player_id="76561198000000002",
            source="fixture",
            games=[
                SteamGameRecord(
                    appid=289070,
                    name="Sid Meier's Civilization VI",
                    playtime_forever_minutes=12000,  # 200 hrs
                    playtime_2weeks_minutes=300,
                    genres=["Strategy"],
                    tags=["turn-based strategy", "grand strategy", "tactical"],
                    achievements_total=100,
                    achievements_unlocked=35,
                ),
                SteamGameRecord(
                    appid=646570,
                    name="Slay the Spire",
                    playtime_forever_minutes=6000,   # 100 hrs
                    genres=["Strategy"],
                    tags=["card game", "deckbuilder", "roguelike", "turn-based"],
                    achievements_total=46,
                    achievements_unlocked=20,
                ),
                SteamGameRecord(
                    appid=590380,
                    name="Into the Breach",
                    playtime_forever_minutes=3000,   # 50 hrs
                    genres=["Strategy"],
                    tags=["turn-based", "tactical", "puzzle"],
                    achievements_total=50,
                    achievements_unlocked=25,
                ),
                SteamGameRecord(
                    appid=736260,
                    name="Baba Is You",
                    playtime_forever_minutes=1800,   # 30 hrs
                    genres=["Puzzle"],
                    tags=["puzzle", "logic"],
                    achievements_total=19,
                    achievements_unlocked=10,
                ),
            ],
        )

    def test_dna_rusher_profile_extraction(self, rusher_dataset):
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(rusher_dataset, profile_tag="genesis_rusher")

        assert dna.player_id == rusher_dataset.player_id
        assert dna.profile_tag == "genesis_rusher"
        assert dna.total_playtime_hours == pytest.approx(300.0, abs=1.0)
        assert dna.total_games_analyzed == 4

        # Genre weights L1 normalized sum to 1.0
        assert sum(dna.genre_weights.values()) == pytest.approx(1.0, abs=1e-3)
        assert all(w >= 0.0 for w in dna.genre_weights.values())

        # Pacing: kinetic
        assert dna.pacing_preference == "kinetic_high_mobility"
        assert dna.pacing_score > 0.20

        # Challenge tolerance: high achievement & hardcore genres
        assert dna.challenge_tolerance > 0.50

        # Mechanical biases: dodge roll should be high
        assert dna.mechanical_biases["dodge_roll"] > 0.50

        # Deliberation: starts with multi-sentence reasoning
        assert len(dna.deliberation) > 50
        assert "Analyzed 4 Steam titles" in dna.deliberation

    def test_dna_tactician_profile_extraction(self, tactician_dataset):
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(tactician_dataset, profile_tag="genesis_tactician")

        assert dna.player_id == tactician_dataset.player_id
        assert dna.profile_tag == "genesis_tactician"
        assert dna.total_playtime_hours == pytest.approx(380.0, abs=1.0)

        # Genre weights L1 normalized
        assert sum(dna.genre_weights.values()) == pytest.approx(1.0, abs=1e-3)

        # Pacing: tactical
        assert dna.pacing_preference == "tactical_methodical"
        assert dna.pacing_score < -0.20

        # Mechanical biases: dodge roll should be relatively lower than tactical
        assert dna.mechanical_biases["dodge_roll"] < 0.50
        assert dna.genre_weights["Strategy"] > 0.30

    def test_empty_dataset_graceful_fallback(self):
        profiler = SteamDNAProfiler()
        empty_dataset = SteamLibraryDataset(player_id="newbie", games=[])
        dna = profiler.extract_dna(empty_dataset)

        assert dna.total_playtime_hours == 0.0
        assert dna.total_games_analyzed == 0
        assert dna.pacing_preference == "balanced"
        assert dna.pacing_score == 0.0
        assert dna.challenge_tolerance == 0.5
        assert sum(dna.genre_weights.values()) == pytest.approx(1.0, abs=1e-3)
        assert len(dna.deliberation) > 0


class TestSteamDNAPersistence:
    """Verifies SQLite WAL persistence, schema integrity, and roundtrip serialization."""

    def test_table_schema_and_wal_mode(self, tmp_path):
        db_file = tmp_path / "telemetry_test.db"
        repo = SteamDNAPersistence(db_path=db_file)

        conn = sqlite3.connect(str(db_file))
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode;")
            journal_mode = cur.fetchone()[0].lower()
            assert journal_mode == "wal"

            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='steam_player_dna';")
            assert cur.fetchone() is not None
        finally:
            conn.close()

    def test_save_and_retrieve_roundtrip(self, tmp_path):
        db_file = tmp_path / "telemetry_test.db"
        repo = SteamDNAPersistence(db_path=db_file)

        profiler = SteamDNAProfiler()
        dataset = SteamLibraryDataset(
            player_id="76561198012345678",
            games=[
                SteamGameRecord(
                    appid=212680,
                    name="FTL: Faster Than Light",
                    playtime_forever_minutes=6000,
                    genres=["Strategy", "Roguelike"],
                    tags=["permadeath", "space"],
                    achievements_total=50,
                    achievements_unlocked=30,
                )
            ]
        )
        dna = profiler.extract_dna(dataset, profile_tag="genesis_ftl")
        dna_id = repo.save_dna(dna)
        assert dna_id.startswith("dna_")

        loaded = repo.get_latest_dna(player_id="76561198012345678")
        assert loaded is not None
        assert loaded.player_id == dna.player_id
        assert loaded.profile_tag == dna.profile_tag
        assert loaded.total_playtime_hours == dna.total_playtime_hours
        assert loaded.pacing_preference == dna.pacing_preference
        assert loaded.pacing_score == dna.pacing_score
        assert loaded.challenge_tolerance == dna.challenge_tolerance
        assert loaded.deliberation == dna.deliberation
        assert loaded.genre_weights == dna.genre_weights

        history = repo.list_dna_history(player_id="76561198012345678")
        assert len(history) == 1
        assert history[0].profile_tag == "genesis_ftl"


class TestSteamDNAMapper:
    """Verifies SteamDNAMapper transformation functions."""

    def test_mapper_from_local_library(self):
        local_lib = SteamLocalLibrary(
            deliberation="Testing mapper",
            total_playtime_hours=50.0,
        )
        dataset = SteamDNAMapper.from_local_library(local_lib)
        assert dataset.source == "local_scan"
        assert dataset.player_id == "local_player"

    def test_mapper_from_web_api(self):
        games = [
            SteamOwnedGame(
                appid=212680,
                name="FTL",
                playtime_forever=3600,
            )
        ]
        details_map = {
            212680: SteamAppDetails(
                appid=212680,
                name="FTL",
                genres=["Roguelike", "Strategy"]
            )
        }
        dataset = SteamDNAMapper.from_web_api(games, details_map=details_map, player_id="player123")
        assert dataset.source == "web_api"
        assert dataset.player_id == "player123"
        assert len(dataset.games) == 1
        assert dataset.games[0].genres == ["Roguelike", "Strategy"]


class TestSteamDNAMCPHook:
    """Verifies Local MCP Server hook integration for get_steam_player_dna."""

    def test_mcp_hook_success_with_deliberation(self, tmp_path):
        db_file = tmp_path / "telemetry_test.db"
        repo = SteamDNAPersistence(db_path=db_file)
        profiler = SteamDNAProfiler()

        params = {
            "deliberation": "Showrunner requires player DNA profile to seed Episode 0 cartridge parameters.",
            "arguments": {
                "player_id": "test_mcp_player",
                "profile_tag": "genesis_default"
            },
            "id": 42
        }

        response = handle_get_steam_player_dna(repo, profiler, params)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 42
        assert "result" in response
        result = response["result"]
        assert list(result.keys())[0] == "deliberation"
        assert result["player_id"] == "test_mcp_player"

    def test_mcp_hook_rejection_missing_deliberation(self, tmp_path):
        db_file = tmp_path / "telemetry_test.db"
        repo = SteamDNAPersistence(db_path=db_file)
        profiler = SteamDNAProfiler()

        params = {
            "deliberation": "",
            "arguments": {"player_id": "rogue_agent"},
            "id": 99
        }

        response = handle_get_steam_player_dna(repo, profiler, params)
        assert response["jsonrpc"] == "2.0"
        assert "error" in response
        assert response["error"]["code"] == -32602
        assert "deliberation" in response["error"]["message"].lower()

    def test_mcp_hook_contrasting_fixture_resolution(self, tmp_path):
        db_file = tmp_path / "telemetry_test.db"
        repo = SteamDNAPersistence(db_path=db_file)
        profiler = SteamDNAProfiler()

        # Query rusher profile via MCP hook
        rusher_params = {
            "deliberation": "Need rusher profile for fast-paced action cartridge.",
            "arguments": {"profile_tag": "rusher"},
            "id": 1,
        }
        res_rusher = handle_get_steam_player_dna(repo, profiler, rusher_params)
        assert res_rusher["result"]["pacing_score"] >= 0.80
        assert res_rusher["result"]["challenge_tolerance"] >= 0.85

        # Query cautious profile via MCP hook
        cautious_params = {
            "deliberation": "Need cautious profile for tactical strategy cartridge.",
            "arguments": {"profile_tag": "cautious"},
            "id": 2,
        }
        res_cautious = handle_get_steam_player_dna(repo, profiler, cautious_params)
        assert res_cautious["result"]["pacing_score"] <= -0.70
        assert res_cautious["result"]["challenge_tolerance"] <= 0.45


class TestSteamDNAMathematicalDerivation:
    """Verifies mathematical derivations for 5-tier playtime, category weights, pacing, and challenge."""

    def test_playtime_hours_and_distribution(self):
        dataset = SteamLibraryDataset(
            player_id="test_playtime_dist",
            games=[
                SteamGameRecord(appid=1, playtime_forever_minutes=60),     # 1h (<2h)
                SteamGameRecord(appid=2, playtime_forever_minutes=300),    # 5h (2-10h)
                SteamGameRecord(appid=3, playtime_forever_minutes=1800),   # 30h (10-50h)
                SteamGameRecord(appid=4, playtime_forever_minutes=4500),   # 75h (50-100h)
                SteamGameRecord(appid=5, playtime_forever_minutes=12000),  # 200h (>100h)
            ]
        )
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(dataset)

        dist = dna.playtime_distribution
        assert set(dist.keys()) == set(PLAYTIME_TIER_NAMES)
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-3)
        assert dist["dabbled_under_2h"] > 0.0
        assert dist["casual_2_to_10h"] > 0.0
        assert dist["engaged_10_to_50h"] > 0.0
        assert dist["dedicated_50_to_100h"] > 0.0
        assert dist["mastered_over_100h"] > 0.0

    def test_category_weights_and_top_categories(self):
        dataset = SteamLibraryDataset(
            player_id="test_categories",
            games=[
                SteamGameRecord(
                    appid=1,
                    playtime_forever_minutes=6000,
                    categories=["Single-player", "Full controller support", "Steam Cloud"]
                ),
                SteamGameRecord(
                    appid=2,
                    playtime_forever_minutes=3000,
                    categories=["Single-player", "Co-op"]
                ),
            ]
        )
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(dataset)

        assert sum(dna.category_weights.values()) == pytest.approx(1.0, abs=1e-3)
        assert len(dna.top_categories) == 3
        assert "Single-player" in dna.top_categories

    def test_pacing_score_continuous_bounds(self):
        profiler = SteamDNAProfiler()

        # Pure kinetic
        kinetic_dataset = SteamLibraryDataset(
            games=[
                SteamGameRecord(appid=1, playtime_forever_minutes=6000, genres=["Action", "BulletHell"])
            ]
        )
        dna_k = profiler.extract_dna(kinetic_dataset)
        assert -1.0 <= dna_k.pacing_score <= 1.0
        assert dna_k.pacing_score >= 0.20
        assert dna_k.pacing_preference == "kinetic_high_mobility"

        # Pure tactical
        tactical_dataset = SteamLibraryDataset(
            games=[
                SteamGameRecord(appid=2, playtime_forever_minutes=6000, genres=["Strategy", "Puzzle"])
            ]
        )
        dna_t = profiler.extract_dna(tactical_dataset)
        assert -1.0 <= dna_t.pacing_score <= 1.0
        assert dna_t.pacing_score <= -0.20
        assert dna_t.pacing_preference == "tactical_methodical"

    def test_challenge_tolerance_difficulty_tags_and_soulslike(self):
        profiler = SteamDNAProfiler()

        # Low difficulty game
        casual_dataset = SteamLibraryDataset(
            games=[
                SteamGameRecord(
                    appid=1,
                    playtime_forever_minutes=1200,
                    genres=["Adventure"],
                    tags=["casual", "relaxing", "exploration"],
                    achievements_total=10,
                    achievements_unlocked=2
                )
            ]
        )
        dna_casual = profiler.extract_dna(casual_dataset)

        # High difficulty game with soulslike and permadeath tags
        hardcore_dataset = SteamLibraryDataset(
            games=[
                SteamGameRecord(
                    appid=2,
                    playtime_forever_minutes=6000,
                    genres=["Roguelike", "Action"],
                    tags=["soulslike", "difficult", "permadeath", "hardcore"],
                    achievements_total=50,
                    achievements_unlocked=45
                )
            ]
        )
        dna_hardcore = profiler.extract_dna(hardcore_dataset)

        assert dna_hardcore.challenge_tolerance > dna_casual.challenge_tolerance
        assert dna_hardcore.challenge_tolerance >= 0.70

    def test_thematic_biases_lexicon_extraction(self):
        dataset = SteamLibraryDataset(
            player_id="thematic_player",
            games=[
                SteamGameRecord(
                    appid=1,
                    playtime_forever_minutes=6000,
                    genres=["Action"],
                    tags=["cyberpunk", "futuristic", "synthwave"]
                ),
                SteamGameRecord(
                    appid=2,
                    playtime_forever_minutes=3000,
                    genres=["RPG"],
                    tags=["dark fantasy", "grimdark", "demons"]
                ),
            ]
        )
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(dataset)

        assert len(dna.thematic_biases) == 3
        assert "cyberpunk" in dna.thematic_biases or "dark fantasy" in dna.thematic_biases
        assert sum(dna.thematic_weights.values()) == pytest.approx(1.0, abs=1e-3)


class TestContrastingProfileFixtures:
    """Verifies Feature 12 contrasting profile fixtures RUSHER_PROFILE and CAUTIOUS_PROFILE."""

    def test_rusher_fixture_characteristics(self):
        assert RUSHER_PROFILE.pacing_score >= 0.80
        assert RUSHER_PROFILE.challenge_tolerance >= 0.85
        assert RUSHER_PROFILE.pacing_preference == "kinetic_high_mobility"
        assert any(g in RUSHER_PROFILE.top_genres for g in ("Action", "Roguelike", "BulletHell"))

    def test_cautious_fixture_characteristics(self):
        assert CAUTIOUS_PROFILE.pacing_score <= -0.70
        assert CAUTIOUS_PROFILE.challenge_tolerance <= 0.45
        assert CAUTIOUS_PROFILE.pacing_preference == "tactical_methodical"
        assert any(g in CAUTIOUS_PROFILE.top_genres for g in ("Strategy", "Puzzle"))

    def test_contrasting_profiles_divergence(self):
        delta_pacing = RUSHER_PROFILE.pacing_score - CAUTIOUS_PROFILE.pacing_score
        assert delta_pacing >= 1.40

        delta_challenge = RUSHER_PROFILE.challenge_tolerance - CAUTIOUS_PROFILE.challenge_tolerance
        assert delta_challenge >= 0.40

        # Divergent aesthetic and mechanical biases
        assert RUSHER_PROFILE.preferred_palette_mood != CAUTIOUS_PROFILE.preferred_palette_mood
        assert RUSHER_PROFILE.mechanical_biases["dodge_roll"] > CAUTIOUS_PROFILE.mechanical_biases["dodge_roll"]
        assert RUSHER_PROFILE.mechanical_biases["speedrun"] > CAUTIOUS_PROFILE.mechanical_biases["speedrun"]

    def test_synthetic_dataset_profiler_roundtrip(self):
        profiler = SteamDNAProfiler()
        rusher_ds = build_rusher_dataset()
        cautious_ds = build_cautious_dataset()

        dna_r = profiler.extract_dna(rusher_ds, profile_tag="test_rusher")
        dna_c = profiler.extract_dna(cautious_ds, profile_tag="test_cautious")

        assert dna_r.pacing_score >= 0.80
        assert dna_r.challenge_tolerance >= 0.85
        assert dna_c.pacing_score <= -0.70
        assert dna_c.challenge_tolerance <= 0.45

    def test_contrasting_profiles_top_genres_strictly_disjoint(self):
        """Feature 12: Assert RUSHER_PROFILE and CAUTIOUS_PROFILE top genres are strictly disjoint.

        Guarantees that a methodical tactical player profile never inherits kinetic
        genres (such as 'Action') from alphabetical tie-breaking across 0.0-weight genres.
        """
        rusher_genres = set(RUSHER_PROFILE.top_genres)
        cautious_genres = set(CAUTIOUS_PROFILE.top_genres)
        assert rusher_genres.isdisjoint(cautious_genres), (
            f"Overlapping top genres found between RUSHER and CAUTIOUS profiles: "
            f"{rusher_genres & cautious_genres}. "
            f"RUSHER={RUSHER_PROFILE.top_genres}, CAUTIOUS={CAUTIOUS_PROFILE.top_genres}"
        )
        assert len(RUSHER_PROFILE.top_genres) > 0
        assert len(CAUTIOUS_PROFILE.top_genres) > 0

        # Dynamic dataset extraction must also be strictly disjoint
        profiler = SteamDNAProfiler()
        dna_r = profiler.extract_dna(build_rusher_dataset(), profile_tag="dyn_rusher")
        dna_c = profiler.extract_dna(build_cautious_dataset(), profile_tag="dyn_cautious")
        assert set(dna_r.top_genres).isdisjoint(set(dna_c.top_genres)), (
            f"Dynamic extraction top genres overlap: {set(dna_r.top_genres) & set(dna_c.top_genres)}"
        )


class TestGracefulDegradation:
    """Verifies Feature 13 graceful degradation and Rule 5 epistemic transparency."""

    def test_zero_achievements_fallback(self):
        profiler = SteamDNAProfiler()
        dataset = SteamLibraryDataset(
            games=[
                SteamGameRecord(appid=1, playtime_forever_minutes=600, achievements_total=0, achievements_unlocked=0)
            ]
        )
        dna = profiler.extract_dna(dataset)
        assert 0.0 <= dna.challenge_tolerance <= 1.0

    def test_empty_genres_and_tags_fallback(self):
        profiler = SteamDNAProfiler()
        dataset = SteamLibraryDataset(
            games=[
                SteamGameRecord(appid=1, playtime_forever_minutes=1200, genres=[], tags=[])
            ]
        )
        dna = profiler.extract_dna(dataset)
        assert sum(dna.genre_weights.values()) == pytest.approx(1.0, abs=1e-3)
        assert dna.total_playtime_hours == 20.0

    def test_degradation_deliberation_rule_5_transparency(self):
        profiler = SteamDNAProfiler()
        empty_dataset = SteamLibraryDataset(player_id="offline_user", games=[])
        dna = profiler.extract_dna(empty_dataset)

        # Rule 5: Explicit notice of fallback in deliberation
        assert "zero-library fallback mode" in dna.deliberation.lower() or "no local steam manifests" in dna.deliberation.lower()
        assert dna.pacing_score == 0.0
        assert dna.challenge_tolerance == 0.50

    def test_negative_playtime_does_not_crash(self):
        """Feature 13: Verifies that datasets with negative playtime do not raise math domain errors
        and produce valid, bounded SteamPlayerDNA profiles.
        """
        profiler = SteamDNAProfiler()

        # Single negative playtime game (triggers math.log(1.0 + hours) with hours <= -1.0)
        neg_dataset = SteamLibraryDataset(
            player_id="corrupted_user_negative_single",
            games=[
                SteamGameRecord(
                    appid=9001,
                    name="CorruptedTitleNegativePlaytime",
                    playtime_forever_minutes=-60,
                    playtime_2weeks_minutes=-120,
                    genres=["Action"],
                    tags=["hack and slash"],
                )
            ],
        )
        dna_single = profiler.extract_dna(neg_dataset)
        assert dna_single is not None
        assert isinstance(dna_single, SteamPlayerDNA)
        assert dna_single.total_playtime_hours >= 0.0
        assert -1.0 <= dna_single.pacing_score <= 1.0
        assert 0.0 <= dna_single.challenge_tolerance <= 1.0
        assert sum(dna_single.genre_weights.values()) == pytest.approx(1.0, abs=1e-3)
        assert sum(dna_single.category_weights.values()) == pytest.approx(1.0, abs=1e-3)
        assert len(dna_single.deliberation) > 0

        # Multi-game adversarial dataset with massive negative values and mixed positives
        mixed_dataset = SteamLibraryDataset(
            player_id="corrupted_user_mixed_negative",
            games=[
                SteamGameRecord(
                    appid=9002,
                    name="ExtremeNegativePlaytime",
                    playtime_forever_minutes=-999999,
                    playtime_2weeks_minutes=-1000,
                    genres=["Strategy"],
                    tags=["turn-based"],
                ),
                SteamGameRecord(
                    appid=9003,
                    name="ValidPositivePlaytime",
                    playtime_forever_minutes=3600,
                    playtime_2weeks_minutes=300,
                    genres=["Puzzle"],
                    tags=["logic"],
                ),
            ],
        )
        dna_mixed = profiler.extract_dna(mixed_dataset)
        assert dna_mixed is not None
        assert dna_mixed.total_playtime_hours >= 0.0
        assert -1.0 <= dna_mixed.pacing_score <= 1.0
        assert 0.0 <= dna_mixed.challenge_tolerance <= 1.0
        assert sum(dna_mixed.genre_weights.values()) == pytest.approx(1.0, abs=1e-3)


class TestSteamDNALLMEpistemicSynthesis:
    """Verifies that SteamDNAProfiler delegates player conclusions to the LLM over raw library evidence."""

    def test_llm_epistemic_dna_synthesis_preserves_open_ended_conclusions(self):
        import json

        captured_prompts = []

        class _StubModels:
            def generate_content(self, *, model, contents, config=None):
                captured_prompts.append(contents)
                payload = {
                    "deliberation": (
                        "Player invests 70% of playtime in deep 100h+ engineering, automation, and physics-mastery "
                        "titles. Primary drive is multi-stage logistics optimization combined with high-ceiling "
                        "vehicular/flight physics and cooperative spatial problem solving."
                    ),
                    "top_genres": [
                        "Automation & Factory Building",
                        "Sandbox Engineering",
                        "Vehicular & Flight Simulation",
                    ],
                    "genre_breakdown": [
                        {"deliberation": "Core factory & logistics hours", "label": "Automation & Factory Building", "weight": 0.45},
                        {"deliberation": "Voxel and contraption construction", "label": "Sandbox Engineering", "weight": 0.30},
                        {"deliberation": "High-skill physics mastery", "label": "Vehicular & Flight Simulation", "weight": 0.15},
                        {"deliberation": "Spatial & co-op puzzles", "label": "Spatial Puzzle", "weight": 0.10},
                    ],
                    "top_categories": ["Single-player", "Co-op", "Steam Workshop"],
                    "category_breakdown": [
                        {"deliberation": "Deep solo systems", "label": "Single-player", "weight": 0.65},
                        {"deliberation": "Shared puzzle & engineering", "label": "Co-op", "weight": 0.20},
                        {"deliberation": "Modding & blueprints", "label": "Steam Workshop", "weight": 0.15},
                    ],
                    "pacing_preference": "tactical_methodical",
                    "pacing_score": -0.45,
                    "challenge_tolerance": 0.88,
                    "thematic_biases": [
                        "industrial automation & logistics",
                        "aerospace & orbital engineering",
                        "colony thermodynamics",
                    ],
                    "thematic_breakdown": [
                        {"deliberation": "Automated production lines", "label": "industrial automation & logistics", "weight": 0.50},
                        {"deliberation": "Spacecraft & flight", "label": "aerospace & orbital engineering", "weight": 0.30},
                        {"deliberation": "Colony survival systems", "label": "colony thermodynamics", "weight": 0.20},
                    ],
                    "mechanical_breakdown": [
                        {"deliberation": "Conveyor and pipeline throughput optimization", "label": "multi_stage_automation_and_logistics", "weight": 0.96},
                        {"deliberation": "Physics-based contraption and base building", "label": "sandbox_engineering_and_construction", "weight": 0.91},
                        {"deliberation": "Precision aerial and vehicular control", "label": "precision_vehicular_and_flight_physics", "weight": 0.84},
                        {"deliberation": "Multi-step spatial puzzle solving", "label": "cooperative_spatial_puzzle_solving", "weight": 0.78},
                    ],
                    "preferred_palette_mood": "industrial_sci_fi",
                    "palette_breakdown": [
                        {"deliberation": "Readable industrial & sci-fi machinery", "label": "industrial_sci_fi", "weight": 0.65},
                        {"deliberation": "Clean high-contrast telemetry", "label": "neon_cyberpunk", "weight": 0.35},
                    ],
                }

                class _Resp:
                    text = json.dumps(payload)

                return _Resp()

        class _StubClient:
            models = _StubModels()

        dataset = SteamLibraryDataset(
            player_id="76561197985203987",
            source="local_scan",
            games=[
                SteamGameRecord(
                    appid=427520,
                    name="Factorio",
                    playtime_forever_minutes=43470,
                    genres=["Indie", "Simulation", "Strategy"],
                    categories=["Single-player", "Co-op"],
                    short_description="Factorio is a game about building and creating automated factories to produce items of increasing complexity.",
                ),
                SteamGameRecord(
                    appid=244850,
                    name="Space Engineers",
                    playtime_forever_minutes=25450,
                    genres=["Action", "Indie", "Simulation", "Strategy"],
                    categories=["Single-player", "Multi-player", "Co-op"],
                    short_description="Space Engineers is a sandbox game about engineering, construction, exploration and survival in space.",
                ),
            ],
        )

        profiler = SteamDNAProfiler(client=_StubClient())
        dna = profiler.extract_dna(dataset, profile_tag="llm_synthesized")

        assert len(captured_prompts) == 1
        assert "Factorio is a game about building and creating automated factories" in captured_prompts[0]
        assert "Space Engineers is a sandbox game about engineering" in captured_prompts[0]
        assert dna.challenge_tolerance == pytest.approx(0.88)
        assert "Automation & Factory Building" in dna.top_genres
        assert dna.mechanical_biases["multi_stage_automation_and_logistics"] == pytest.approx(0.96)
        assert "industrial automation & logistics" in dna.thematic_biases
        assert dna.preferred_palette_mood == "industrial_sci_fi"
        assert sum(dna.genre_weights.values()) == pytest.approx(1.0, abs=1e-3)

    def test_activity_windows_30d_90d_1yr_lifetime(self):
        """Verifies compute_activity_windows buckets games across last_30_days, last_90_days, last_1_year, and lifetime."""
        from engine.steam.dna import compute_activity_windows

        ref_ts = 1_800_000_000
        games = [
            SteamGameRecord(
                appid=1001,
                name="Active 30d Game",
                playtime_forever_minutes=3000,  # 50h
                playtime_2weeks_minutes=600,    # 10h
                rtime_last_played=ref_ts - 5 * 86400,
                genres=["Simulation", "Strategy"],
            ),
            SteamGameRecord(
                appid=1002,
                name="Active 90d Game",
                playtime_forever_minutes=7800,  # 130h
                rtime_last_played=ref_ts - 60 * 86400,
                genres=["Simulation"],
            ),
            SteamGameRecord(
                appid=1003,
                name="Active 1yr Game",
                playtime_forever_minutes=1200,  # 20h
                rtime_last_played=ref_ts - 200 * 86400,
                genres=["Puzzle"],
            ),
            SteamGameRecord(
                appid=1004,
                name="Lifetime Anchor Game",
                playtime_forever_minutes=30000,  # 500h
                rtime_last_played=ref_ts - 900 * 86400,
                genres=["Strategy"],
            ),
        ]

        windows = compute_activity_windows(games, reference_timestamp=ref_ts)
        assert set(windows.keys()) == {"last_30_days", "last_90_days", "last_1_year", "lifetime"}

        assert windows["last_30_days"]["games_count"] == 1
        assert windows["last_30_days"]["recent_2weeks_hours"] == pytest.approx(10.0)
        assert [g["name"] for g in windows["last_30_days"]["games"]] == ["Active 30d Game"]

        assert windows["last_90_days"]["games_count"] == 2
        assert [g["name"] for g in windows["last_90_days"]["games"]] == ["Active 90d Game", "Active 30d Game"]

        assert windows["last_1_year"]["games_count"] == 3
        assert {g["name"] for g in windows["last_1_year"]["games"]} == {
            "Active 30d Game",
            "Active 90d Game",
            "Active 1yr Game",
        }

        assert windows["lifetime"]["games_count"] == 4
        assert windows["lifetime"]["games"][0]["name"] == "Lifetime Anchor Game"




