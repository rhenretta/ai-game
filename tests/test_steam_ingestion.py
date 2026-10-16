"""tests/test_steam_ingestion.py
Comprehensive Unit & Integration Test Suite for Steam Library Ingestion Engine.
Verifies:
  1. Pure-Python VDF & ACF KeyValues parsing without external dependencies.
  2. Extraction of all 7 mandatory ACF manifest fields (appid, name, installdir, SizeOnDisk, LastUpdated, bytesToDownload, StateFlags).
  3. Multi-library folder discovery across volumes on macOS.
  4. Local Steam Scanner userdata playtime & last-played extraction.
  5. Rationale-First Epistemic Architecture (deliberation declared first in all classes).
  6. Steam Web API client integration, public store appdetails, and deterministic fixture fallback.
"""
import ast
import json
import os
import tempfile
from pathlib import Path
import pytest

from engine.steam.models import (
    SteamAchievement,
    SteamAppDetails,
    SteamAppManifest,
    SteamAppPlaytime,
    SteamLibraryFolder,
    SteamLocalLibrary,
    SteamOwnedGame,
    SteamUserRecord,
    SteamUserStats,
)
from engine.steam.vdf import load_vdf, parse_vdf, tokenize_vdf
from engine.steam.scanner import SteamScanner
from engine.steam.web_api import (
    SteamFixtureLoader,
    SteamWebApiClient,
    SteamWebApiError,
    SteamAuthenticationError,
)
from tests.test_tier1_units import validate_rationale_first_ast


def validate_rationale_first_ast_file(filepath: Path) -> bool:
    """Uses AST analysis to verify every class in a module declares deliberation first."""
    source_code = filepath.read_text(encoding="utf-8")
    return validate_rationale_first_ast(source_code)


# ============================================================================
# 1. Pure-Python VDF & ACF Parsing Tests
# ============================================================================

class TestPurePythonVDFParser:
    """Verifies that pure Python VDF parser handles all KeyValues 1 syntax variants."""

    def test_basic_key_value_pairs(self):
        vdf_text = '''
        "AppState"
        {
            "appid"        "212680"
            "name"         "FTL: Faster Than Light"
            "StateFlags"   "4"
        }
        '''
        data = parse_vdf(vdf_text)
        assert "AppState" in data
        assert data["AppState"]["appid"] == "212680"
        assert data["AppState"]["name"] == "FTL: Faster Than Light"
        assert data["AppState"]["StateFlags"] == "4"

    def test_comments_and_escapes(self):
        vdf_text = '''
        // This is a header comment
        "Config"
        {
            "description"    "A game with \\"quotes\\" and \\\\backslashes\\\\" // inline comment
            "empty_section"
            {
            }
        }
        '''
        data = parse_vdf(vdf_text)
        assert "Config" in data
        assert data["Config"]["description"] == 'A game with "quotes" and \\backslashes\\'
        assert data["Config"]["empty_section"] == {}

    def test_libraryfolders_structure(self):
        vdf_text = '''
        "libraryfolders"
        {
            "0"
            {
                "path"    "/Users/test/Library/Application Support/Steam"
                "label"   ""
                "totalsize"   "500000000"
                "apps"
                {
                    "212680"    "284228511"
                    "1062090"   "2188484858"
                }
            }
            "1"
            {
                "path"    "/Volumes/ExternalSSD/SteamLibrary"
                "label"   "FastStorage"
                "totalsize"   "2000000000"
                "apps"
                {
                    "2764460"   "616021685"
                }
            }
        }
        '''
        data = parse_vdf(vdf_text)
        lf = data.get("libraryfolders", {})
        assert "0" in lf
        assert "1" in lf
        assert lf["0"]["path"] == "/Users/test/Library/Application Support/Steam"
        assert lf["0"]["apps"]["212680"] == "284228511"
        assert lf["1"]["path"] == "/Volumes/ExternalSSD/SteamLibrary"
        assert lf["1"]["apps"]["2764460"] == "616021685"

    def test_loginusers_structure(self):
        vdf_text = '''
        "users"
        {
            "76561197985203987"
            {
                "AccountName"    "testpilot"
                "PersonaName"    "StarNavigator"
                "RememberPassword"    "1"
                "AutoLogin"    "1"
                "Timestamp"    "1791492678"
            }
        }
        '''
        data = parse_vdf(vdf_text)
        users = data.get("users", {})
        assert "76561197985203987" in users
        assert users["76561197985203987"]["AccountName"] == "testpilot"
        assert users["76561197985203987"]["PersonaName"] == "StarNavigator"
        assert users["76561197985203987"]["AutoLogin"] == "1"


# ============================================================================
# 2. Application Manifest (.acf) Parser Tests
# ============================================================================

class TestAppManifestParser:
    """Verifies that all 7 required fields are extracted correctly from ACF manifests."""

    def test_extract_all_7_mandatory_acf_fields(self, tmp_path):
        acf_content = '''
        "AppState"
        {
            "appid"            "212680"
            "universe"         "1"
            "name"             "FTL: Faster Than Light"
            "StateFlags"       "4"
            "installdir"       "FTL Faster Than Light"
            "LastUpdated"      "1668898503"
            "SizeOnDisk"       "284228511"
            "buildid"          "4710954"
            "LastOwner"        "76561197985203987"
            "BytesToDownload"  "196913392"
            "BytesDownloaded"  "196913392"
        }
        '''
        acf_file = tmp_path / "appmanifest_212680.acf"
        acf_file.write_text(acf_content, encoding="utf-8")

        scanner = SteamScanner(steam_root=tmp_path)
        manifest = scanner.parse_app_manifest(acf_file)

        # Assert all 7 required fields:
        assert manifest.appid == 212680
        assert manifest.name == "FTL: Faster Than Light"
        assert manifest.installdir == "FTL Faster Than Light"
        assert manifest.size_on_disk == 284228511
        assert manifest.last_updated == 1668898503
        assert manifest.bytes_to_download == 196913392
        assert manifest.state_flags == 4
        assert manifest.bytes_downloaded == 196913392
        assert manifest.build_id == 4710954
        assert manifest.last_owner == "76561197985203987"


# ============================================================================
# 3. Multi-Library Folder Discovery & User Playtime Extraction
# ============================================================================

class TestMultiLibraryAndPlaytimeScanner:
    """Verifies multi-library folder scanning and playtime extraction from userdata."""

    @pytest.fixture
    def mock_steam_tree(self, tmp_path):
        steam_dir = tmp_path / "Steam"
        steam_dir.mkdir()
        (steam_dir / "config").mkdir()
        (steam_dir / "steamapps").mkdir()
        (steam_dir / "userdata" / "12345678" / "config").mkdir(parents=True)

        # Secondary library on simulated external volume
        ext_lib = tmp_path / "ExternalVolumes" / "GameDrive" / "SteamLibrary"
        (ext_lib / "steamapps").mkdir(parents=True)

        # 1. config/loginusers.vdf
        login_vdf = steam_dir / "config" / "loginusers.vdf"
        login_vdf.write_text('''
        "users"
        {
            "76561197985203987"
            {
                "AccountName" "captain_mac"
                "PersonaName" "CaptainMac"
                "AutoLogin" "1"
                "Timestamp" "1791492678"
            }
        }
        ''')

        # 2. steamapps/libraryfolders.vdf
        lf_vdf = steam_dir / "steamapps" / "libraryfolders.vdf"
        lf_vdf.write_text(f'''
        "libraryfolders"
        {{
            "0"
            {{
                "path" "{steam_dir}"
                "apps"
                {{
                    "212680" "284228511"
                }}
            }}
            "1"
            {{
                "path" "{ext_lib}"
                "apps"
                {{
                    "1062090" "2188484858"
                }}
            }}
        }}
        ''')

        # 3. Manifest in primary library
        m1 = steam_dir / "steamapps" / "appmanifest_212680.acf"
        m1.write_text('''
        "AppState"
        {
            "appid" "212680"
            "name" "FTL: Faster Than Light"
            "installdir" "FTL"
            "SizeOnDisk" "284228511"
            "LastUpdated" "1668898503"
            "BytesToDownload" "196913392"
            "StateFlags" "4"
        }
        ''')

        # 4. Manifest in external library
        m2 = ext_lib / "steamapps" / "appmanifest_1062090.acf"
        m2.write_text('''
        "AppState"
        {
            "appid" "1062090"
            "name" "Timberborn"
            "installdir" "Timberborn"
            "SizeOnDisk" "2188484858"
            "LastUpdated" "1698898503"
            "BytesToDownload" "1500000000"
            "StateFlags" "4"
        }
        ''')

        # 5. userdata localconfig.vdf
        localconfig = steam_dir / "userdata" / "12345678" / "config" / "localconfig.vdf"
        localconfig.write_text('''
        "UserLocalConfigStore"
        {
            "Software"
            {
                "Valve"
                {
                    "Steam"
                    {
                        "apps"
                        {
                            "212680"
                            {
                                "Playtime" "4800"
                                "LastPlayed" "1668898503"
                            }
                            "1062090"
                            {
                                "Playtime" "1200"
                                "LastPlayed" "1698898503"
                            }
                        }
                    }
                }
            }
        }
        ''')

        return steam_dir, ext_lib

    def test_multi_library_discovery(self, mock_steam_tree):
        steam_dir, ext_lib = mock_steam_tree
        scanner = SteamScanner(steam_root=steam_dir)

        folders = scanner.discover_library_folders()
        paths = [f.path for f in folders]
        assert str(steam_dir.resolve()) in paths
        assert str(ext_lib.resolve()) in paths

    def test_scan_installed_apps_across_libraries(self, mock_steam_tree):
        steam_dir, ext_lib = mock_steam_tree
        scanner = SteamScanner(steam_root=steam_dir)

        apps = scanner.scan_installed_apps()
        assert 212680 in apps
        assert 1062090 in apps
        assert apps[212680].name == "FTL: Faster Than Light"
        assert apps[1062090].name == "Timberborn"

    def test_scan_user_playtimes(self, mock_steam_tree):
        steam_dir, _ = mock_steam_tree
        scanner = SteamScanner(steam_root=steam_dir)

        playtimes = scanner.scan_user_playtimes()
        assert 212680 in playtimes
        assert playtimes[212680].playtime_minutes == 4800
        assert playtimes[212680].playtime_hours == 80.0
        assert playtimes[1062090].playtime_minutes == 1200
        assert playtimes[1062090].playtime_hours == 20.0

    def test_scan_full_library_aggregation(self, mock_steam_tree):
        steam_dir, _ = mock_steam_tree
        scanner = SteamScanner(steam_root=steam_dir)

        library = scanner.scan_full_library()
        assert library.active_user is not None
        assert library.active_user.persona_name == "CaptainMac"
        assert len(library.installed_apps) == 2
        assert library.total_installed_bytes == 284228511 + 2188484858
        assert library.total_playtime_hours == 100.0

    def test_scanner_graceful_when_not_installed(self, tmp_path):
        non_existent = tmp_path / "NonExistentSteam"
        scanner = SteamScanner(steam_root=non_existent)
        assert scanner.is_steam_installed() is False
        assert scanner.get_login_users() == []
        assert scanner.discover_library_folders() == []
        assert scanner.scan_installed_apps() == {}
        assert scanner.scan_user_playtimes() == {}


# ============================================================================
# 4. Rationale-First Epistemic Architecture Checks
# ============================================================================

class TestRationaleFirstEpistemicAST:
    """Verifies that all Steam classes declare deliberation as their very first attribute."""

    def test_all_models_deliberation_first(self):
        root = Path(__file__).parent.parent
        models_file = root / "engine" / "steam" / "models.py"
        scanner_file = root / "engine" / "steam" / "scanner.py"
        web_api_file = root / "engine" / "steam" / "web_api.py"

        assert validate_rationale_first_ast_file(models_file) is True
        assert validate_rationale_first_ast_file(scanner_file) is True
        assert validate_rationale_first_ast_file(web_api_file) is True

    def test_model_dict_deliberation_first_key(self):
        manifest = SteamAppManifest(
            deliberation="Inspecting app manifest",
            appid=212680,
            name="FTL",
            installdir="FTL",
            size_on_disk=1000,
            last_updated=1000,
            bytes_to_download=100,
            state_flags=4
        )
        d = manifest.to_dict()
        assert list(d.keys())[0] == "deliberation"


# ============================================================================
# 5. Steam Web API Client & Deterministic Fixture Bridge
# ============================================================================

class TestSteamWebApiClientWithFixtures:
    """Verifies Steam Web API client integration with deterministic offline fixtures."""

    def test_get_owned_games_fixture(self):
        client = SteamWebApiClient(offline_mode=True)
        games = client.get_owned_games()
        assert len(games) >= 3
        appids = [g.appid for g in games]
        assert 212680 in appids  # FTL
        assert 1062090 in appids # Timberborn
        assert 2764460 in appids # Balatro

        ftl = next(g for g in games if g.appid == 212680)
        assert ftl.name == "FTL: Faster Than Light"
        assert ftl.playtime_forever == 5390
        assert pytest.approx(ftl.playtime_hours, 0.1) == 89.8

    def test_get_app_details_store_fixture(self):
        client = SteamWebApiClient(offline_mode=True)
        details = client.get_app_details(212680)
        assert details is not None
        assert details.appid == 212680
        assert details.name == "FTL: Faster Than Light"
        assert "Strategy" in details.genres
        assert "Roguelike" in details.genres
        assert "Single-player" in details.categories

    def test_get_player_achievements_fixture(self):
        client = SteamWebApiClient(offline_mode=True)
        achievements = client.get_player_achievements(appid=212680)
        assert len(achievements) >= 2
        apiname_list = [a.apiname for a in achievements]
        assert "ACH_WIN_EASY" in apiname_list or "ACH_FIRST_VICTORY" in apiname_list
        unlocked = [a for a in achievements if a.achieved]
        assert len(unlocked) >= 1

    def test_get_user_stats_for_game_fixture(self):
        client = SteamWebApiClient(offline_mode=True)
        stats = client.get_user_stats_for_game(appid=212680)
        assert stats.appid == 212680
        assert len(stats.achievements) >= 2
        assert stats.achievement_completion_rate > 0.0

    def test_missing_api_key_graceful_fixture_fallback(self):
        client = SteamWebApiClient(api_key=None, offline_mode=False)
        games = client.get_owned_games(steam_id="76561197985203987")
        assert len(games) >= 3


# ============================================================================
# 6. Real macOS Host Steam Non-Destructive Probe (Non-Blocking)
# ============================================================================

def test_real_macos_steam_host_probe():
    """If host machine has local Steam installed, verify scanner discovers it non-destructively."""
    scanner = SteamScanner()
    if scanner.is_steam_installed():
        users = scanner.get_login_users()
        assert len(users) >= 1
        active = scanner.get_active_user()
        assert active is not None
        assert active.steam_id_64 != ""

        folders = scanner.discover_library_folders()
        assert len(folders) >= 1

        apps = scanner.scan_installed_apps()
        assert isinstance(apps, dict)
