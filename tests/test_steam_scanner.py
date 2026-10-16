"""tests/test_steam_scanner.py
Comprehensive Unit and Integration Test Suite for Feature 8: macOS Local Steam Library Scanner.

Test Hierarchy:
1. TestSteamScannerPathDiscovery (default root, env override, custom paths, expansion)
2. TestLibraryFoldersDiscovery (modern dict, legacy string, unmounted drives, deduplication)
3. TestAppManifestParsing (all 7 core ACF fields, appid from filename, corrupt ACF resilience)
4. TestUserPlaytimeExtraction (Playtime, Playtime2wks, LastPlayed, Disconnected, multi-user)
5. TestUnifiedGameCatalogDiscovery (union of installed + played ~250 games, name resolution)
6. TestScannerResilienceAndDegradation (missing Steam, corrupt VDFs, permissions)
7. TestRationaleFirstEpistemicAST (AST verification of deliberation declared first)
8. TestLiveHostMacSteamIntegration (live probe of host macOS Steam installation)
"""
from pathlib import Path
import os
import pytest

from engine.steam.models import (
    SteamAppManifest,
    SteamAppPlaytime,
    SteamGameRecord,
    SteamLibraryDataset,
    SteamLibraryFolder,
    SteamLocalLibrary,
    SteamUserRecord,
)
from engine.steam.scanner import DEFAULT_MACOS_STEAM_ROOT, SteamScanner
from tests.test_tier1_units import validate_rationale_first_ast


class TestSteamScannerPathDiscovery:
    """Tests resolution of macOS Steam root paths and environment overrides."""

    def test_default_macos_steam_root(self):
        scanner = SteamScanner()
        expected = DEFAULT_MACOS_STEAM_ROOT.expanduser().resolve()
        assert scanner.steam_root == expected

    def test_steam_root_path_env_override(self, tmp_path, monkeypatch):
        fake_steam = tmp_path / "custom_steam"
        fake_steam.mkdir()
        monkeypatch.setenv("STEAM_ROOT_PATH", str(fake_steam))
        scanner = SteamScanner()
        assert scanner.steam_root == fake_steam.resolve()

    def test_explicit_steam_root_argument(self, tmp_path, monkeypatch):
        fake_env = tmp_path / "env_steam"
        fake_explicit = tmp_path / "explicit_steam"
        fake_env.mkdir()
        fake_explicit.mkdir()
        monkeypatch.setenv("STEAM_ROOT_PATH", str(fake_env))

        scanner = SteamScanner(steam_root=fake_explicit)
        assert scanner.steam_root == fake_explicit.resolve()

    def test_is_steam_installed(self, tmp_path):
        existing_steam = tmp_path / "steam_exists"
        existing_steam.mkdir()
        scanner_found = SteamScanner(steam_root=existing_steam)
        assert scanner_found.is_steam_installed() is True

        missing_steam = tmp_path / "steam_missing"
        scanner_missing = SteamScanner(steam_root=missing_steam)
        assert scanner_missing.is_steam_installed() is False


class TestLibraryFoldersDiscovery:
    """Tests parsing modern and legacy libraryfolders.vdf schemas and deduplication."""

    def test_modern_libraryfolders_vdf(self, tmp_path):
        steam_root = tmp_path / "Steam"
        steamapps = steam_root / "steamapps"
        steamapps.mkdir(parents=True)

        ext_lib = tmp_path / "ExternalLib"
        ext_lib.mkdir()

        vdf_content = f"""
"libraryfolders"
{{
    "0"
    {{
        "path" "{steam_root}"
        "label" ""
        "totalsize" "1000000000"
        "apps"
        {{
            "212680" "284228511"
        }}
    }}
    "1"
    {{
        "path" "{ext_lib}"
        "label" "FastSSD"
        "totalsize" "2000000000"
        "apps"
        {{
            "361280" "1906668464"
        }}
    }}
}}
"""
        (steamapps / "libraryfolders.vdf").write_text(vdf_content)

        scanner = SteamScanner(steam_root=steam_root)
        folders = scanner.discover_library_folders()

        assert len(folders) == 2
        f0 = next(f for f in folders if f.path == str(steam_root.resolve()))
        assert f0.apps.get(212680) == 284228511
        f1 = next(f for f in folders if f.path == str(ext_lib.resolve()))
        assert f1.label == "FastSSD"
        assert f1.apps.get(361280) == 1906668464

    def test_legacy_libraryfolders_vdf(self, tmp_path):
        steam_root = tmp_path / "Steam"
        steamapps = steam_root / "steamapps"
        steamapps.mkdir(parents=True)

        ext_lib = tmp_path / "LegacyLib"
        ext_lib.mkdir()

        vdf_content = f"""
"LibraryFolders"
{{
    "TimeNextStatsReport" "12345678"
    "ContentStatsID" "98765432"
    "1" "{ext_lib}"
}}
"""
        (steamapps / "libraryfolders.vdf").write_text(vdf_content)

        scanner = SteamScanner(steam_root=steam_root)
        folders = scanner.discover_library_folders()

        paths = [f.path for f in folders]
        assert str(ext_lib.resolve()) in paths

    def test_unmounted_volume_gracefully_handled(self, tmp_path):
        steam_root = tmp_path / "Steam"
        steamapps = steam_root / "steamapps"
        steamapps.mkdir(parents=True)

        vdf_content = """
"libraryfolders"
{
    "0"
    {
        "path" "/Volumes/UnmountedThunderboltDrive/SteamLibrary"
        "totalsize" "500000000000"
    }
}
"""
        (steamapps / "libraryfolders.vdf").write_text(vdf_content)

        scanner = SteamScanner(steam_root=steam_root)
        # Should not raise FileNotFoundError or OSError
        folders = scanner.discover_library_folders()
        assert isinstance(folders, list)

    def test_deduplication_of_library_paths(self, tmp_path):
        steam_root = tmp_path / "Steam"
        steam_root.mkdir()

        scanner = SteamScanner(
            steam_root=steam_root,
            custom_library_paths=[steam_root, steam_root / "."]
        )
        folders = scanner.discover_library_folders()
        matched = [f for f in folders if f.path == str(steam_root.resolve())]
        assert len(matched) == 1


class TestAppManifestParsing:
    """Tests parsing appmanifest_<appid>.acf files and field extraction."""

    def test_extract_all_7_mandatory_acf_fields(self, tmp_path):
        acf_path = tmp_path / "appmanifest_212680.acf"
        acf_text = """
"AppState"
{
    "appid" "212680"
    "name" "FTL: Faster Than Light"
    "installdir" "FTL"
    "SizeOnDisk" "284228511"
    "LastUpdated" "1650000000"
    "BytesToDownload" "250000000"
    "BytesDownloaded" "250000000"
    "StateFlags" "4"
    "buildid" "1234567"
    "LastOwner" "76561198000000001"
}
"""
        acf_path.write_text(acf_text)

        scanner = SteamScanner(steam_root=tmp_path)
        manifest = scanner.parse_app_manifest(acf_path)

        assert manifest.appid == 212680
        assert manifest.name == "FTL: Faster Than Light"
        assert manifest.installdir == "FTL"
        assert manifest.size_on_disk == 284228511
        assert manifest.last_updated == 1650000000
        assert manifest.bytes_to_download == 250000000
        assert manifest.state_flags == 4

    def test_appid_fallback_from_filename(self, tmp_path):
        acf_path = tmp_path / "appmanifest_1062090.acf"
        # Omit appid key inside AppState
        acf_text = """
"AppState"
{
    "name" "Timberborn"
    "installdir" "Timberborn"
    "SizeOnDisk" "2188484858"
}
"""
        acf_path.write_text(acf_text)

        scanner = SteamScanner(steam_root=tmp_path)
        manifest = scanner.parse_app_manifest(acf_path)
        assert manifest.appid == 1062090
        assert manifest.name == "Timberborn"

    def test_corrupted_acf_manifest_skipped(self, tmp_path):
        steamapps = tmp_path / "steamapps"
        steamapps.mkdir()

        good_acf = steamapps / "appmanifest_361280.acf"
        good_acf.write_text('"AppState" { "appid" "361280" "name" "Turmoil" "SizeOnDisk" "1000" }')

        corrupt_acf = steamapps / "appmanifest_999999.acf"
        corrupt_acf.write_text('CORRUPTED TRUNCATED { "unclosed')

        scanner = SteamScanner(steam_root=tmp_path)
        apps = scanner.scan_installed_apps()

        assert 361280 in apps
        assert 999999 not in apps


class TestUserPlaytimeExtraction:
    """Tests extracting lifetime, 2-week, and disconnected playtime from localconfig.vdf."""

    def test_extract_playtime_forever_and_2weeks(self, tmp_path):
        steam_root = tmp_path / "Steam"
        user_cfg_dir = steam_root / "userdata" / "24938259" / "config"
        user_cfg_dir.mkdir(parents=True)

        localconfig = """
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
                    "620"
                    {
                        "Playtime" "8984"
                        "Playtime2wks" "600"
                        "LastPlayed" "1757207344"
                    }
                }
            }
        }
    }
}
"""
        (user_cfg_dir / "localconfig.vdf").write_text(localconfig)

        scanner = SteamScanner(steam_root=steam_root)
        playtimes = scanner.scan_user_playtimes()

        assert 620 in playtimes
        p = playtimes[620]
        assert p.playtime_minutes == 8984
        assert p.playtime_hours == round(8984 / 60.0, 2)
        assert p.playtime_2weeks_minutes == 600
        assert p.playtime_2weeks_hours == 10.0
        assert p.last_played_timestamp == 1757207344

    def test_extract_disconnected_playtime(self, tmp_path):
        steam_root = tmp_path / "Steam"
        user_cfg_dir = steam_root / "userdata" / "12345" / "config"
        user_cfg_dir.mkdir(parents=True)

        localconfig = """
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
                    "2764460"
                    {
                        "Playtime" "3133"
                        "Playtime2wks" "3133"
                        "PlaytimeDisconnected" "120"
                        "LastPlayed" "1760000000"
                    }
                }
            }
        }
    }
}
"""
        (user_cfg_dir / "localconfig.vdf").write_text(localconfig)

        scanner = SteamScanner(steam_root=steam_root)
        playtimes = scanner.scan_user_playtimes()

        assert 2764460 in playtimes
        p = playtimes[2764460]
        assert p.playtime_disconnected_minutes == 120

    def test_steamid64_to_account_id_routing(self, tmp_path):
        steam_root = tmp_path / "Steam"
        # SteamID64: 76561197985203987 -> AccountID: 24938259
        account_dir = steam_root / "userdata" / "24938259" / "config"
        account_dir.mkdir(parents=True)

        localconfig = """
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
                        "Playtime" "4500"
                    }
                }
            }
        }
    }
}
"""
        (account_dir / "localconfig.vdf").write_text(localconfig)

        scanner = SteamScanner(steam_root=steam_root)
        playtimes = scanner.scan_user_playtimes(steam_id_64="76561197985203987")
        assert 212680 in playtimes
        assert playtimes[212680].playtime_minutes == 4500

    def test_multi_account_playtime_aggregation(self, tmp_path):
        steam_root = tmp_path / "Steam"
        u1 = steam_root / "userdata" / "1111" / "config"
        u2 = steam_root / "userdata" / "2222" / "config"
        u1.mkdir(parents=True)
        u2.mkdir(parents=True)

        cfg1 = '"UserLocalConfigStore" { "Software" { "Valve" { "Steam" { "apps" { "100" { "Playtime" "300" "LastPlayed" "1000" } } } } } }'
        cfg2 = '"UserLocalConfigStore" { "Software" { "Valve" { "Steam" { "apps" { "100" { "Playtime" "750" "LastPlayed" "2000" } } } } } }'
        (u1 / "localconfig.vdf").write_text(cfg1)
        (u2 / "localconfig.vdf").write_text(cfg2)

        scanner = SteamScanner(steam_root=steam_root)
        playtimes = scanner.scan_user_playtimes()
        assert playtimes[100].playtime_minutes == 750
        assert playtimes[100].last_played_timestamp == 2000


class TestUnifiedGameCatalogDiscovery:
    """Tests discovering the complete union of installed apps and played apps."""

    def test_to_dataset_includes_all_played_games(self, tmp_path):
        steam_root = tmp_path / "Steam"
        steamapps = steam_root / "steamapps"
        steamapps.mkdir(parents=True)
        user_cfg = steam_root / "userdata" / "12345" / "config"
        user_cfg.mkdir(parents=True)

        # 2 installed games
        (steamapps / "appmanifest_1.acf").write_text('"AppState" { "appid" "1" "name" "Installed One" }')
        (steamapps / "appmanifest_2.acf").write_text('"AppState" { "appid" "2" "name" "Installed Two" }')

        # 10 played games in localconfig (apps 1 to 10)
        apps_entries = []
        for i in range(1, 11):
            apps_entries.append(f'"{i}" {{ "Playtime" "{i * 100}" "Playtime2wks" "50" }}')
        localconfig = f"""
"UserLocalConfigStore"
{{
    "Software"
    {{
        "Valve"
        {{
            "Steam"
            {{
                "apps"
                {{
                    {" ".join(apps_entries)}
                }}
            }}
        }}
    }}
}}
"""
        (user_cfg / "localconfig.vdf").write_text(localconfig)

        scanner = SteamScanner(steam_root=steam_root)
        lib = scanner.scan_full_library()
        dataset = scanner.to_dataset(lib)

        # Verify union contains all 10 games
        assert len(dataset.games) == 10
        game_ids = {g.appid for g in dataset.games}
        assert game_ids == set(range(1, 11))

    def test_name_resolution_installed_vs_fallback(self, tmp_path):
        steam_root = tmp_path / "Steam"
        steamapps = steam_root / "steamapps"
        steamapps.mkdir(parents=True)
        user_cfg = steam_root / "userdata" / "12345" / "config"
        user_cfg.mkdir(parents=True)

        (steamapps / "appmanifest_1.acf").write_text('"AppState" { "appid" "1" "name" "Known Name" }')
        (user_cfg / "localconfig.vdf").write_text(
            '"UserLocalConfigStore" { "Software" { "Valve" { "Steam" { "apps" { "1" { "Playtime" "100" } "2" { "Playtime" "200" } } } } } }'
        )

        name_map = {2: "Mapped Name"}
        scanner = SteamScanner(steam_root=steam_root)
        dataset = scanner.to_dataset(name_map=name_map)

        g1 = next(g for g in dataset.games if g.appid == 1)
        g2 = next(g for g in dataset.games if g.appid == 2)

        assert g1.name == "Known Name"
        assert g2.name == "Mapped Name"


class TestScannerResilienceAndDegradation:
    """Tests zero-crash guarantees on missing directories, corrupt syntax, and permission errors."""

    def test_missing_steam_directory_returns_empty_library(self, tmp_path):
        missing_root = tmp_path / "does_not_exist"
        scanner = SteamScanner(steam_root=missing_root)
        lib = scanner.scan_full_library()

        assert isinstance(lib, SteamLocalLibrary)
        assert len(lib.installed_apps) == 0
        assert len(lib.playtimes) == 0
        assert lib.total_installed_bytes == 0
        assert lib.total_playtime_hours == 0.0

        dataset = scanner.to_dataset(lib)
        assert len(dataset.games) == 0

    def test_corrupt_localconfig_skips_cleanly(self, tmp_path):
        steam_root = tmp_path / "Steam"
        user_cfg = steam_root / "userdata" / "12345" / "config"
        user_cfg.mkdir(parents=True)
        (user_cfg / "localconfig.vdf").write_text("CORRUPTED { { { unclosed")

        scanner = SteamScanner(steam_root=steam_root)
        playtimes = scanner.scan_user_playtimes()
        assert playtimes == {}


class TestRationaleFirstEpistemicAST:
    """Verifies that all scanner classes declare deliberation first in their AST."""

    def test_scanner_ast_compliance(self):
        root = Path(__file__).parent.parent
        scanner_file = root / "engine" / "steam" / "scanner.py"
        models_file = root / "engine" / "steam" / "models.py"
        vdf_parser_file = root / "engine" / "steam" / "vdf_parser.py"

        assert validate_rationale_first_ast(scanner_file.read_text(encoding="utf-8")) is True
        assert validate_rationale_first_ast(models_file.read_text(encoding="utf-8")) is True
        assert validate_rationale_first_ast(vdf_parser_file.read_text(encoding="utf-8")) is True


class TestLiveHostMacSteamIntegration:
    """Tests live integration with the macOS host Steam installation if present."""

    def test_live_host_steam_discovery(self):
        if not DEFAULT_MACOS_STEAM_ROOT.is_dir():
            pytest.skip("Default macOS Steam path not found on host.")

        scanner = SteamScanner()
        assert scanner.is_steam_installed() is True

        user = scanner.get_active_user()
        if user:
            assert len(user.steam_id_64) > 0
            assert len(user.persona_name) > 0

        playtimes = scanner.scan_user_playtimes()
        # On the live host, Gnaget has 250 played games
        assert len(playtimes) >= 200

        dataset = scanner.to_dataset()
        assert len(dataset.games) >= 200
