"""tests/test_steam_stress.py
Empirical Challenger Stress & Adversarial Test Suite for Steam Ingestion Engine.
Covers:
  1. Malformed VDF/ACF Syntax Stress (unbalanced braces, unclosed quotes, escapes, unicode, trailing slashes, inline/attached comments, deep recursion depth > 50).
  2. Missing ACF keys and corrupted data types (SizeOnDisk, StateFlags, BytesToDownload, appid inference).
  3. Multi-Library Edge Cases (missing directories, unmounted volumes, permission-denied directories, duplicate appids).
  4. Steam Web API Offline & Error Resilience (HTTP 429, HTTP 500, timeouts, malformed JSON bodies, in-memory caching).
"""
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import urllib.error
import pytest

from engine.steam.vdf import load_vdf, parse_vdf, tokenize_vdf
from engine.steam.scanner import SteamScanner
from engine.steam.web_api import (
    SteamFixtureLoader,
    SteamWebApiClient,
    SteamWebApiError,
    SteamRateLimitError,
    SteamAuthenticationError,
)
from engine.steam.dna import SteamDNAProfiler
from engine.steam.models import SteamAppManifest, SteamLibraryDataset, SteamGameRecord


# ============================================================================
# 1. Malformed VDF/ACF Syntax Stress
# ============================================================================

class TestVDFSyntaxStress:
    """Stress tests for VDF/ACF tokenizer and recursive descent parser."""

    def test_unbalanced_braces_excess_closing(self):
        """Excess closing braces should be ignored gracefully without unhandled exceptions."""
        vdf_text = '''
        "AppState"
        {
            "appid" "212680"
            "name"  "FTL"
        }
        }
        }
        '''
        data = parse_vdf(vdf_text)
        assert "AppState" in data
        assert data["AppState"]["appid"] == "212680"
        assert data["AppState"]["name"] == "FTL"

    def test_unbalanced_braces_excess_opening(self):
        """Unclosed opening braces should parse available tokens without infinite loops or crashes."""
        vdf_text = '''
        "AppState"
        {
            "SectionA"
            {
                "SectionB"
                {
                    "appid" "212680"
        '''
        data = parse_vdf(vdf_text)
        assert "AppState" in data
        assert "SectionA" in data["AppState"]
        assert "SectionB" in data["AppState"]["SectionA"]
        assert data["AppState"]["SectionA"]["SectionB"]["appid"] == "212680"

    def test_unclosed_quotes_resilience(self):
        """Unclosed quotation marks should not crash tokenizer or parser."""
        vdf_text = '''
        "AppState"
        {
            "appid" "212680
            "name" "FTL: Faster Than Light"
        }
        '''
        data = parse_vdf(vdf_text)
        assert isinstance(data, dict)

    def test_escapes_backslashes_and_quotes(self):
        """Verify unescaping of \\", \\\\, \\n, and \\t inside quoted strings."""
        vdf_text = r'''
        "Config"
        {
            "path"    "C:\\Games\\Steam\\steamapps"
            "quote"   "He said \"Launch Game\" immediately"
            "lines"   "Line1\nLine2\tTabbed"
        }
        '''
        data = parse_vdf(vdf_text)
        config = data.get("Config", {})
        assert config.get("path") == r"C:\Games\Steam\steamapps"
        assert config.get("quote") == 'He said "Launch Game" immediately'
        assert config.get("lines") == "Line1\nLine2\tTabbed"

    def test_trailing_slashes_in_quotes(self):
        """Paths with trailing backslashes inside quotes must not escape closing quote."""
        vdf_text = r'''
        "Paths"
        {
            "library_root" "C:\\Program Files\\Steam\\"
            "next_key"     "valid_value"
        }
        '''
        data = parse_vdf(vdf_text)
        paths = data.get("Paths", {})
        assert paths.get("library_root") == "C:\\Program Files\\Steam\\"
        assert paths.get("next_key") == "valid_value"

    def test_unicode_and_special_characters(self):
        """Unicode characters (CJK, emojis, accented characters) must be preserved."""
        vdf_text = '''
        "InternationalGame"
        {
            "name_ja"     "東方妖々夢 〜 Perfect Cherry Blossom."
            "emoji_tag"   "🚀 Cyber Action 👾"
            "accented"    "Pokémon & Déjà Vu"
        }
        '''
        data = parse_vdf(vdf_text)
        ig = data.get("InternationalGame", {})
        assert ig.get("name_ja") == "東方妖々夢 〜 Perfect Cherry Blossom."
        assert ig.get("emoji_tag") == "🚀 Cyber Action 👾"
        assert ig.get("accented") == "Pokémon & Déjà Vu"

    def test_comments_stress(self):
        """Verify comments on own line, inline with space, attached without space, and inside quotes."""
        vdf_text = '''
        // Top level file comment
        "AppState" // inline comment with leading space
        {
            // Comment inside block
            "key1" "val1"// inline comment without space
            "key2" "https://steamcommunity.com/app/212680" // URL with slashes preserved
            "key3" "contains // inside quote string"
        }
        // Trailing comment
        '''
        data = parse_vdf(vdf_text)
        app = data.get("AppState", {})
        assert app.get("key1") == "val1"
        assert app.get("key2") == "https://steamcommunity.com/app/212680"
        assert app.get("key3") == "contains // inside quote string"

    def test_deeply_nested_structures_depth_60(self):
        """Verify recursion depth > 50 (requirement: > 50) parses without RecursionError."""
        depth = 60
        nested_vdf = ""
        for i in range(depth):
            nested_vdf += f'"level_{i}" {{\n'
        nested_vdf += '"leaf_key" "deep_payload"\n'
        for i in range(depth):
            nested_vdf += "}\n"

        data = parse_vdf(nested_vdf)
        curr = data
        for i in range(depth):
            assert f"level_{i}" in curr
            curr = curr[f"level_{i}"]
        assert curr["leaf_key"] == "deep_payload"

    def test_deeply_nested_structures_depth_100(self):
        """Stress-test deep nesting up to 100 levels."""
        depth = 100
        nested_vdf = "".join(f'"L{i}" {{\n' for i in range(depth))
        nested_vdf += '"target" "found"\n'
        nested_vdf += "".join("}\n" for _ in range(depth))

        data = parse_vdf(nested_vdf)
        curr = data
        for i in range(depth):
            curr = curr[f"L{i}"]
        assert curr["target"] == "found"


# ============================================================================
# 2. Missing ACF Keys & Corrupted Manifest Stress
# ============================================================================

class TestACFManifestStress:
    """Stress tests for ACF manifest extraction with missing or corrupted fields."""

    def test_manifest_missing_all_optional_keys(self, tmp_path):
        """Manifest missing SizeOnDisk, StateFlags, BytesToDownload must default gracefully."""
        acf_content = '''
        "AppState"
        {
            "appid" "9999"
            "name"  "Minimal App"
        }
        '''
        acf_file = tmp_path / "appmanifest_9999.acf"
        acf_file.write_text(acf_content, encoding="utf-8")

        scanner = SteamScanner(steam_root=tmp_path)
        manifest = scanner.parse_app_manifest(acf_file)

        assert manifest.appid == 9999
        assert manifest.name == "Minimal App"
        assert manifest.size_on_disk == 0
        assert manifest.state_flags == 4  # Default fully installed flag
        assert manifest.bytes_to_download == 0
        assert manifest.bytes_downloaded == 0
        assert manifest.last_updated == 0
        assert manifest.installdir == ""

    def test_manifest_corrupted_numeric_types(self, tmp_path):
        """Manifest with string non-numbers for integer fields must fall back to defaults without crash."""
        acf_content = '''
        "AppState"
        {
            "appid"           "5555"
            "name"            "Corrupt Numbers App"
            "SizeOnDisk"      "NaN_NOT_A_NUMBER"
            "StateFlags"      "CORRUPT_FLAG"
            "BytesToDownload" "UNKNOWN_SIZE"
            "LastUpdated"     "YESTERDAY"
        }
        '''
        acf_file = tmp_path / "appmanifest_5555.acf"
        acf_file.write_text(acf_content, encoding="utf-8")

        scanner = SteamScanner(steam_root=tmp_path)
        manifest = scanner.parse_app_manifest(acf_file)

        assert manifest.appid == 5555
        assert manifest.size_on_disk == 0
        assert manifest.state_flags == 4
        assert manifest.bytes_to_download == 0
        assert manifest.last_updated == 0

    def test_manifest_empty_appstate(self, tmp_path):
        """Empty AppState dictionary must return safe default manifest."""
        acf_file = tmp_path / "appmanifest_1234.acf"
        acf_file.write_text('"AppState" {\n}\n', encoding="utf-8")

        scanner = SteamScanner(steam_root=tmp_path)
        manifest = scanner.parse_app_manifest(acf_file)
        assert isinstance(manifest, SteamAppManifest)
        assert manifest.size_on_disk == 0
        assert manifest.state_flags == 4

    def test_manifest_missing_appstate_wrapper(self, tmp_path):
        """Manifest without top-level 'AppState' key should parse keys at root level."""
        acf_content = '''
        "appid" "7777"
        "name"  "Unwrapped App"
        "SizeOnDisk" "1024"
        '''
        acf_file = tmp_path / "appmanifest_7777.acf"
        acf_file.write_text(acf_content, encoding="utf-8")

        scanner = SteamScanner(steam_root=tmp_path)
        manifest = scanner.parse_app_manifest(acf_file)
        assert manifest.appid == 7777
        assert manifest.name == "Unwrapped App"
        assert manifest.size_on_disk == 1024

    def test_manifest_missing_appid_infers_from_filename(self, tmp_path):
        """Verify that when 'appid' key is absent inside the ACF body, the scanner infers it from filename.
        
        EMPIRICAL BUG REPRODUCTION: In scanner.py line 228:
        'appid = int(app_state.get("appid", 0))' causes appid to be 0 instead of falling
        back to filename inference in the except block!
        """
        acf_file = tmp_path / "appmanifest_440.acf"
        acf_file.write_text('''
        "AppState"
        {
            "name" "Team Fortress 2"
            "SizeOnDisk" "15000000000"
        }
        ''', encoding="utf-8")

        scanner = SteamScanner(steam_root=tmp_path)
        manifest = scanner.parse_app_manifest(acf_file)
        assert manifest.appid == 440, (
            f"Expected manifest.appid to be inferred as 440 from filename, got {manifest.appid}"
        )


# ============================================================================
# 3. Multi-Library Edge Cases
# ============================================================================

class TestMultiLibraryStress:
    """Stress tests for multi-library folder discovery, unreadable paths, and duplicate appids."""

    def test_libraryfolders_missing_unmounted_directories_skipped(self, tmp_path):
        """Non-existent or unmounted paths in libraryfolders.vdf must be skipped safely."""
        steam_dir = tmp_path / "Steam"
        steam_dir.mkdir()
        (steam_dir / "steamapps").mkdir()

        missing_dir = tmp_path / "UnmountedVolumes" / "ExternalSSD" / "SteamLibrary"

        lf_vdf = steam_dir / "steamapps" / "libraryfolders.vdf"
        lf_vdf.write_text(f'''
        "libraryfolders"
        {{
            "0"
            {{
                "path" "{steam_dir}"
            }}
            "1"
            {{
                "path" "{missing_dir}"
            }}
        }}
        ''')

        scanner = SteamScanner(steam_root=steam_dir)
        apps = scanner.scan_installed_apps()
        assert isinstance(apps, dict)

    def test_duplicate_appids_keeps_newest_timestamp(self, tmp_path):
        """When an app exists in multiple libraries, the one with newer last_updated wins."""
        lib1 = tmp_path / "LibPrimary"
        lib2 = tmp_path / "LibSecondary"
        (lib1 / "steamapps").mkdir(parents=True)
        (lib2 / "steamapps").mkdir(parents=True)

        # Primary library has older manifest (last_updated=1000)
        (lib1 / "steamapps" / "appmanifest_212680.acf").write_text('''
        "AppState"
        {
            "appid" "212680"
            "name" "FTL Older Build"
            "LastUpdated" "1000"
            "SizeOnDisk" "100000"
        }
        ''')

        # Secondary library has newer manifest (last_updated=2000)
        (lib2 / "steamapps" / "appmanifest_212680.acf").write_text('''
        "AppState"
        {
            "appid" "212680"
            "name" "FTL Newer Build"
            "LastUpdated" "2000"
            "SizeOnDisk" "200000"
        }
        ''')

        scanner = SteamScanner(steam_root=lib1, custom_library_paths=[lib2])
        apps = scanner.scan_installed_apps()
        assert 212680 in apps
        assert apps[212680].name == "FTL Newer Build"
        assert apps[212680].last_updated == 2000
        assert apps[212680].size_on_disk == 200000

    def test_duplicate_appids_older_encountered_second(self, tmp_path):
        """When newer manifest is scanned first and older second, newer is still preserved."""
        lib1 = tmp_path / "LibNewer"
        lib2 = tmp_path / "LibOlder"
        (lib1 / "steamapps").mkdir(parents=True)
        (lib2 / "steamapps").mkdir(parents=True)

        (lib1 / "steamapps" / "appmanifest_212680.acf").write_text('''
        "AppState"
        {
            "appid" "212680"
            "name" "FTL Newer Build"
            "LastUpdated" "5000"
        }
        ''')

        (lib2 / "steamapps" / "appmanifest_212680.acf").write_text('''
        "AppState"
        {
            "appid" "212680"
            "name" "FTL Older Build"
            "LastUpdated" "1000"
        }
        ''')

        scanner = SteamScanner(steam_root=lib1, custom_library_paths=[lib2])
        apps = scanner.scan_installed_apps()
        assert apps[212680].name == "FTL Newer Build"
        assert apps[212680].last_updated == 5000

    def test_libraryfolders_permission_denied_directory_skips_safely(self, tmp_path):
        """A permission-denied directory configured in libraryfolders.vdf must be skipped safely.
        
        EMPIRICAL BUG REPRODUCTION: In scanner.py line 282:
        '(folder_path / "steamapps").is_dir()' raises unhandled PermissionError when
        folder_path has chmod 000!
        """
        steam_dir = tmp_path / "Steam"
        steam_dir.mkdir()
        (steam_dir / "steamapps").mkdir()

        perm_dir = tmp_path / "RestrictedLibrary"
        perm_dir.mkdir()
        (perm_dir / "steamapps").mkdir()

        lf_vdf = steam_dir / "steamapps" / "libraryfolders.vdf"
        lf_vdf.write_text(f'''
        "libraryfolders"
        {{
            "0" {{ "path" "{steam_dir}" }}
            "1" {{ "path" "{perm_dir}" }}
        }}
        ''')

        os.chmod(perm_dir, 0o000)
        try:
            scanner = SteamScanner(steam_root=steam_dir)
            apps = scanner.scan_installed_apps()
            assert isinstance(apps, dict)
        finally:
            os.chmod(perm_dir, 0o777)

    def test_userdata_permission_denied_skips_safely(self, tmp_path):
        """When userdata directory has permission denied, scan_user_playtimes must skip safely.
        
        EMPIRICAL BUG REPRODUCTION: In scanner.py line 312:
        'account_dirs = [d for d in userdata_dir.iterdir() ...]' raises unhandled PermissionError!
        """
        steam_dir = tmp_path / "Steam"
        steam_dir.mkdir()
        userdata = steam_dir / "userdata"
        userdata.mkdir()

        os.chmod(userdata, 0o000)
        try:
            scanner = SteamScanner(steam_root=steam_dir)
            playtimes = scanner.scan_user_playtimes()
            assert isinstance(playtimes, dict)
        finally:
            os.chmod(userdata, 0o777)


# ============================================================================
# 4. Steam Web API Offline & Error Resilience
# ============================================================================

class TestSteamWebApiResilienceStress:
    """Stress tests for Steam Web API client fault resilience."""

    @pytest.fixture
    def active_client(self) -> SteamWebApiClient:
        return SteamWebApiClient(
            api_key="TEST_API_KEY_12345",
            steam_id="76561197985203987",
            offline_mode=False,
            timeout=2.0,
        )

    def test_http_429_rate_limit_fallback_owned_games(self, active_client):
        """HTTP 429 response on GetOwnedGames must fall back to offline fixture without unhandled exception."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                "http://api.steampowered.com", 429, "Too Many Requests", {}, None
            )
            games = active_client.get_owned_games()
            assert len(games) >= 3
            assert any(g.appid == 212680 for g in games)

    def test_http_500_server_error_fallback_owned_games(self, active_client):
        """HTTP 500 response on GetOwnedGames must fall back to offline fixture without crashing."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                "http://api.steampowered.com", 500, "Internal Server Error", {}, None
            )
            games = active_client.get_owned_games()
            assert len(games) >= 3

    def test_network_timeout_fallback_owned_games(self, active_client):
        """Network timeout on GetOwnedGames must fall back to offline fixture."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("Connection timed out after 2000ms")
            games = active_client.get_owned_games()
            assert len(games) >= 3

    def test_malformed_json_fallback_owned_games(self, active_client):
        """Malformed JSON or HTML error response on GetOwnedGames must fall back to fixture."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"<html><head><title>502 Bad Gateway</title></head></html>"
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            games = active_client.get_owned_games()
            assert len(games) >= 3

    def test_store_appdetails_http_429_fallback(self, active_client):
        """Store API rate limit (HTTP 429) must fall back to fixture."""
        active_client._store_cache.clear()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                "http://store.steampowered.com", 429, "Too Many Requests", {}, None
            )
            details = active_client.get_app_details(212680)
            assert details is not None
            assert details.appid == 212680
            assert "Roguelike" in details.genres

    def test_store_appdetails_http_500_fallback(self, active_client):
        """Store API server error (HTTP 500) must fall back to fixture."""
        active_client._store_cache.clear()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                "http://store.steampowered.com", 500, "Internal Server Error", {}, None
            )
            details = active_client.get_app_details(212680)
            assert details is not None
            assert details.appid == 212680

    def test_store_appdetails_timeout_fallback(self, active_client):
        """Store API network timeout must fall back to fixture."""
        active_client._store_cache.clear()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("Store API timed out")
            details = active_client.get_app_details(212680)
            assert details is not None
            assert details.appid == 212680

    def test_store_appdetails_malformed_json_fallback(self, active_client):
        """Store API returning invalid JSON must fall back to fixture."""
        active_client._store_cache.clear()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"{'invalid': json truncated"
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            details = active_client.get_app_details(212680)
            assert details is not None
            assert details.appid == 212680

    def test_store_appdetails_in_memory_cache_hit(self, active_client):
        """Once app details are fetched, subsequent calls hit in-memory cache with zero network calls."""
        active_client._store_cache.clear()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                "http://store.steampowered.com", 429, "Too Many Requests", {}, None
            )
            d1 = active_client.get_app_details(212680)
            assert mock_urlopen.call_count == 1

        # Second call should hit _store_cache without calling urlopen
        with patch("urllib.request.urlopen") as mock_urlopen_2:
            d2 = active_client.get_app_details(212680)
            assert mock_urlopen_2.call_count == 0
            assert d1 == d2

    def test_player_achievements_and_user_stats_network_faults(self, active_client):
        """Achievements and user stats queries recover from timeouts, 429, and malformed JSON."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("Stats timeout")
            achievements = active_client.get_player_achievements(appid=212680)
            assert len(achievements) >= 2

            stats = active_client.get_user_stats_for_game(appid=212680)
            assert stats.appid == 212680
            assert len(stats.achievements) >= 2
            assert stats.achievement_completion_rate > 0.0


# ============================================================================
# 5. Steam DNA Profiler Stress & Edge Cases
# ============================================================================

class TestSteamDNAPerformanceAndEdgeCases:
    """Stress tests for Steam DNA Profiler under boundary conditions and large datasets."""

    def test_dna_empty_and_zero_playtime_datasets(self):
        """Profiler handles 0 games, and games with 0 logged playtime gracefully."""
        profiler = SteamDNAProfiler()

        # 1. Completely empty dataset
        empty_dna = profiler.extract_dna(SteamLibraryDataset(player_id="empty", games=[]))
        assert empty_dna.total_games_analyzed == 0
        assert empty_dna.total_playtime_hours == 0.0
        assert empty_dna.pacing_score == 0.0

        # 2. Dataset with games but 0 playtime and 0 achievements
        zero_dataset = SteamLibraryDataset(
            player_id="zero_time",
            games=[
                SteamGameRecord(appid=1, name="Game 1", playtime_forever_minutes=0),
                SteamGameRecord(appid=2, name="Game 2", playtime_forever_minutes=0),
            ]
        )
        zero_dna = profiler.extract_dna(zero_dataset)
        assert zero_dna.total_games_analyzed == 2
        assert zero_dna.total_playtime_hours == 0.0
        assert sum(zero_dna.genre_weights.values()) == pytest.approx(1.0, abs=1e-3)

    def test_dna_large_dataset_performance(self):
        """Profiler processes a 500-game Steam library under 100 milliseconds."""
        import time

        games = []
        for i in range(500):
            games.append(
                SteamGameRecord(
                    appid=1000 + i,
                    name=f"Steam Game {i}",
                    playtime_forever_minutes=(i * 17) % 6000,
                    genres=["Action", "Roguelike"] if i % 2 == 0 else ["Strategy", "Puzzle"],
                    achievements_total=20,
                    achievements_unlocked=i % 20,
                )
            )

        dataset = SteamLibraryDataset(player_id="heavy_collector", games=games)
        profiler = SteamDNAProfiler()

        start = time.perf_counter()
        dna = profiler.extract_dna(dataset)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert dna.total_games_analyzed == 500
        assert sum(dna.genre_weights.values()) == pytest.approx(1.0, abs=1e-3)
        assert elapsed_ms < 100.0, f"Profiling 500 games took {elapsed_ms:.2f}ms (>100ms threshold)"

    def test_dna_unrecognized_genres_and_tags(self):
        """Games with unrecognized or custom tags fall back to balanced Action/Adventure scores."""
        dataset = SteamLibraryDataset(
            player_id="obscure_gamer",
            games=[
                SteamGameRecord(
                    appid=999999,
                    name="Experimental Title",
                    playtime_forever_minutes=1200,
                    genres=["ExperimentalArt", "UnknownCustomGenre"],
                    tags=["avant-garde", "non-standard"],
                )
            ]
        )
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(dataset)
        assert dna.genre_weights["Action"] > 0.0
        assert dna.genre_weights["Adventure"] > 0.0
        assert sum(dna.genre_weights.values()) == pytest.approx(1.0, abs=1e-3)
