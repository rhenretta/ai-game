"""engine/steam/scanner.py
Local macOS Steam Installation Scanner & Library Ingestion Engine.
Discovers and parses:
  - ~/Library/Application Support/Steam/config/loginusers.vdf
  - ~/Library/Application Support/Steam/steamapps/libraryfolders.vdf
  - Multi-library folder appmanifest_<appid>.acf files across volumes
  - Userdata localconfig.vdf playtime and last-played timestamps
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from engine.steam.models import (
    SteamAppManifest,
    SteamAppPlaytime,
    SteamGameRecord,
    SteamLibraryDataset,
    SteamLibraryFolder,
    SteamLocalLibrary,
    SteamUserRecord,
)
from engine.steam.vdf import load_vdf, parse_vdf


DEFAULT_MACOS_STEAM_ROOT = Path.home() / "Library" / "Application Support" / "Steam"


class SteamScannerError(Exception):
    """Raised when Steam scanner encounters an invalid directory or read error."""
    deliberation: str = "Exception for Steam scanner failures."


class SteamScanner:
    """Discovers, parses, and aggregates local macOS Steam libraries and manifests."""
    deliberation: str = "Scanning local macOS Steam installation and multi-library volumes."

    def __init__(
        self,
        steam_root: Optional[Union[str, Path]] = None,
        custom_library_paths: Optional[List[Union[str, Path]]] = None,
        name_cache: Optional[Dict[int, str]] = None,
        deliberation: str = "Scanning local macOS Steam installation and multi-library volumes."
    ):
        self.deliberation = deliberation
        if steam_root:
            self.steam_root = Path(steam_root).expanduser().resolve()
        elif "STEAM_ROOT_PATH" in os.environ:
            self.steam_root = Path(os.environ["STEAM_ROOT_PATH"]).expanduser().resolve()
        else:
            self.steam_root = DEFAULT_MACOS_STEAM_ROOT.expanduser().resolve()

        self.custom_library_paths = [Path(p).expanduser().resolve() for p in (custom_library_paths or [])]
        self.name_cache = dict(name_cache) if name_cache else {}

    def is_steam_installed(self) -> bool:
        """Checks if the configured Steam root directory exists."""
        return self.steam_root.is_dir()

    def get_login_users(self) -> List[SteamUserRecord]:
        """Parses loginusers.vdf to identify all local Steam accounts.
        
        Extracts SteamID64, AccountName, PersonaName, AutoLogin, and Timestamp.
        Sorted with active/most recent accounts first.
        """
        try:
            loginusers_path = self.steam_root / "config" / "loginusers.vdf"
            if not loginusers_path.is_file():
                return []
            data = load_vdf(loginusers_path)
        except Exception:
            return []

        users_dict = data.get("users", {})
        if not isinstance(users_dict, dict):
            return []

        user_records: List[SteamUserRecord] = []
        for steam_id_64, info in users_dict.items():
            if not isinstance(info, dict):
                continue
            account_name = str(info.get("AccountName", ""))
            persona_name = str(info.get("PersonaName", ""))
            auto_login = str(info.get("AutoLogin", "0")) == "1"
            most_recent = str(info.get("mostrecent", "0")) == "1" or auto_login
            try:
                timestamp = int(info.get("Timestamp", 0))
            except (ValueError, TypeError):
                timestamp = 0

            user_records.append(
                SteamUserRecord(
                    deliberation=f"Discovered user {persona_name} ({steam_id_64})",
                    steam_id_64=str(steam_id_64),
                    account_name=account_name,
                    persona_name=persona_name,
                    is_most_recent=most_recent,
                    auto_login=auto_login,
                    timestamp=timestamp,
                )
            )

        # Sort: auto_login / most_recent first, then highest timestamp
        user_records.sort(key=lambda u: (u.auto_login, u.is_most_recent, u.timestamp), reverse=True)
        return user_records

    def get_active_user(self) -> Optional[SteamUserRecord]:
        """Returns the primary active or most recently logged-in Steam user."""
        users = self.get_login_users()
        return users[0] if users else None

    def discover_library_folders(self) -> List[SteamLibraryFolder]:
        """Discovers all Steam library folders across disks/volumes on macOS.
        
        Parses steamapps/libraryfolders.vdf and merges any custom configured paths.
        """
        folders: List[SteamLibraryFolder] = []
        lf_path = self.steam_root / "steamapps" / "libraryfolders.vdf"
        
        seen_paths = set()

        try:
            if lf_path.is_file():
                data = load_vdf(lf_path)
                lf_dict = None
                for k, v in data.items():
                    if k.lower() == "libraryfolders" and isinstance(v, dict):
                        lf_dict = v
                        break
                if lf_dict is None:
                    lf_dict = data.get("libraryfolders", {})
                if isinstance(lf_dict, dict):
                    for idx_key, folder_info in lf_dict.items():
                        raw_path = None
                        label = ""
                        total_size = 0
                        apps_dict: Dict[int, int] = {}

                        if isinstance(folder_info, dict):
                            raw_path = folder_info.get("path")
                            label = str(folder_info.get("label", ""))
                            try:
                                total_size = int(folder_info.get("totalsize", 0))
                            except (ValueError, TypeError):
                                total_size = 0
                            apps_raw = folder_info.get("apps", {})
                            if isinstance(apps_raw, dict):
                                for aid, asize in apps_raw.items():
                                    try:
                                        apps_dict[int(aid)] = int(asize)
                                    except (ValueError, TypeError):
                                        continue
                        elif isinstance(folder_info, str) and str(idx_key).isdigit():
                            # Legacy Steam libraryfolders format: "1" "/path/to/Steam"
                            raw_path = folder_info

                        if not raw_path:
                            continue

                        try:
                            f_path = Path(raw_path).expanduser().resolve()
                        except (OSError, ValueError):
                            continue

                        if f_path in seen_paths:
                            continue
                        seen_paths.add(f_path)

                        try:
                            folder_id = int(idx_key)
                        except (ValueError, TypeError):
                            folder_id = len(folders)

                        folders.append(
                            SteamLibraryFolder(
                                deliberation=f"Discovered library folder {f_path} with {len(apps_dict)} apps",
                                folder_id=folder_id,
                                path=str(f_path),
                                label=label,
                                total_size=total_size,
                                apps=apps_dict,
                            )
                        )
        except Exception:
            pass

        # If default Steam root was not in libraryfolders.vdf, ensure it is added
        default_resolved = self.steam_root.resolve()
        if default_resolved.is_dir() and default_resolved not in seen_paths:
            seen_paths.add(default_resolved)
            folders.insert(
                0,
                SteamLibraryFolder(
                    deliberation="Default Steam root library folder",
                    folder_id=0,
                    path=str(default_resolved),
                    label="Default",
                    total_size=0,
                    apps={},
                )
            )

        # Merge custom configured library paths
        for cp in self.custom_library_paths:
            resolved_cp = cp.resolve()
            if resolved_cp.is_dir() and resolved_cp not in seen_paths:
                seen_paths.add(resolved_cp)
                folders.append(
                    SteamLibraryFolder(
                        deliberation=f"Custom configured library folder {resolved_cp}",
                        folder_id=len(folders),
                        path=str(resolved_cp),
                        label="Custom",
                        total_size=0,
                        apps={},
                    )
                )

        return folders

    def parse_app_manifest(self, manifest_path: Union[str, Path]) -> SteamAppManifest:
        """Parses a single appmanifest_<appid>.acf file.
        
        Extracts all 7 core fields:
          - appid
          - name
          - installdir
          - SizeOnDisk
          - LastUpdated
          - bytesToDownload / BytesToDownload
          - StateFlags
        """
        path = Path(manifest_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")

        data = load_vdf(path)
        app_state = data.get("AppState")
        if not isinstance(app_state, dict):
            if isinstance(data, dict) and ("appid" in data or "name" in data):
                app_state = data
            else:
                raise ValueError(f"Corrupted or invalid ACF manifest at {path}")

        # Extract appid
        appid_val = app_state.get("appid")
        try:
            appid = int(appid_val) if appid_val is not None and str(appid_val).isdigit() else 0
            if appid == 0:
                raise ValueError("Missing or zero appid")
        except (ValueError, TypeError):
            # Try to infer from filename appmanifest_<appid>.acf
            stem = path.stem
            appid_str = stem.replace("appmanifest_", "")
            appid = int(appid_str) if appid_str.isdigit() else 0

        name = str(app_state.get("name", f"App {appid}"))
        installdir = str(app_state.get("installdir", ""))

        def safe_int(key_variants: List[str], default: int = 0) -> int:
            for k in key_variants:
                if k in app_state:
                    try:
                        return int(app_state[k])
                    except (ValueError, TypeError):
                        pass
            return default

        size_on_disk = safe_int(["SizeOnDisk", "sizeondisk", "Size"])
        last_updated = safe_int(["LastUpdated", "lastupdated"])
        bytes_to_download = safe_int(["BytesToDownload", "bytesToDownload", "bytestodownload"])
        bytes_downloaded = safe_int(["BytesDownloaded", "bytesDownloaded", "bytesdownloaded"])
        state_flags = safe_int(["StateFlags", "stateflags"], default=4)
        build_id = safe_int(["buildid", "TargetBuildID"])
        last_owner = str(app_state.get("LastOwner", ""))

        # Parent steamapps folder & library root
        parent_steamapps = path.parent
        library_root = parent_steamapps.parent if parent_steamapps.name == "steamapps" else parent_steamapps

        return SteamAppManifest(
            deliberation=f"Parsed manifest for appid {appid}: {name}",
            appid=appid,
            name=name,
            installdir=installdir,
            size_on_disk=size_on_disk,
            last_updated=last_updated,
            bytes_to_download=bytes_to_download,
            bytes_downloaded=bytes_downloaded,
            state_flags=state_flags,
            build_id=build_id,
            last_owner=last_owner,
            library_path=str(library_root),
            manifest_path=str(path),
        )

    def scan_installed_apps(self) -> Dict[int, SteamAppManifest]:
        """Scans all discovered Steam library folders for appmanifest_*.acf files."""
        folders = self.discover_library_folders()
        installed_apps: Dict[int, SteamAppManifest] = {}

        for folder in folders:
            try:
                folder_path = Path(folder.path)
                steamapps_dir = folder_path / "steamapps" if (folder_path / "steamapps").is_dir() else folder_path
                if not steamapps_dir.is_dir():
                    continue

                for acf_file in steamapps_dir.glob("appmanifest_*.acf"):
                    try:
                        manifest = self.parse_app_manifest(acf_file)
                        if manifest.appid > 0:
                            # If app exists in multiple libraries, keep latest updated
                            if manifest.appid in installed_apps:
                                if manifest.last_updated > installed_apps[manifest.appid].last_updated:
                                    installed_apps[manifest.appid] = manifest
                            else:
                                installed_apps[manifest.appid] = manifest
                    except Exception:
                        continue
            except (PermissionError, OSError):
                continue

        return installed_apps

    def scan_user_playtimes(self, steam_id_64: Optional[str] = None) -> Dict[int, SteamAppPlaytime]:
        """Extracts playtime (minutes & hours) and last-played timestamps from localconfig.vdf.
        
        Scans ~/Library/Application Support/Steam/userdata/<account_id>/config/localconfig.vdf
        """
        playtimes: Dict[int, SteamAppPlaytime] = {}
        userdata_dir = self.steam_root / "userdata"
        try:
            if not userdata_dir.is_dir():
                return playtimes

            if steam_id_64:
                try:
                    target_account_id = str(int(steam_id_64) & 0xFFFFFFFF)
                    target_dir = userdata_dir / target_account_id
                    if target_dir.is_dir():
                        account_dirs = [target_dir]
                    else:
                        account_dirs = [d for d in userdata_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                except (ValueError, TypeError):
                    account_dirs = [d for d in userdata_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            else:
                account_dirs = [d for d in userdata_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        except (PermissionError, OSError):
            return playtimes

        if not account_dirs:
            return playtimes

        def safe_int_val(val: Any, default: int = 0) -> int:
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        for acc_dir in account_dirs:
            try:
                localconfig_path = acc_dir / "config" / "localconfig.vdf"
                if not localconfig_path.is_file():
                    continue

                data = load_vdf(localconfig_path)
                store = data.get("UserLocalConfigStore", {})
                apps = store.get("Software", {}).get("Valve", {}).get("Steam", {}).get("apps", {})
                if not isinstance(apps, dict):
                    continue

                for appid_str, app_info in apps.items():
                    if not isinstance(app_info, dict) or not appid_str.isdigit():
                        continue
                    appid = int(appid_str)
                    
                    pt_min = safe_int_val(app_info.get("Playtime", 0))
                    
                    pt_2w_raw = app_info.get("Playtime2wks")
                    pt_2w_min = safe_int_val(pt_2w_raw) if pt_2w_raw is not None else None
                    
                    pt_disc_raw = app_info.get("PlaytimeDisconnected")
                    pt_disc_min = safe_int_val(pt_disc_raw) if pt_disc_raw is not None else None
                    
                    lp_time = safe_int_val(app_info.get("LastPlayed", 0))

                    if pt_min > 0 or lp_time > 0 or (pt_2w_min is not None and pt_2w_min > 0):
                        if appid in playtimes:
                            # Aggregate playtime if user has multiple local profiles
                            existing = playtimes[appid]
                            existing.playtime_minutes = max(existing.playtime_minutes, pt_min)
                            existing.playtime_hours = round(existing.playtime_minutes / 60.0, 2)
                            existing.last_played_timestamp = max(existing.last_played_timestamp, lp_time)
                            if pt_2w_min is not None:
                                existing.playtime_2weeks_minutes = max(existing.playtime_2weeks_minutes or 0, pt_2w_min)
                            if pt_disc_min is not None:
                                existing.playtime_disconnected_minutes = max(existing.playtime_disconnected_minutes or 0, pt_disc_min)
                        else:
                            playtimes[appid] = SteamAppPlaytime(
                                deliberation=f"Extracted playtime for app {appid}: {pt_min}m (2wk: {pt_2w_min}m)",
                                appid=appid,
                                playtime_minutes=pt_min,
                                playtime_hours=round(pt_min / 60.0, 2),
                                playtime_2weeks_minutes=pt_2w_min,
                                playtime_disconnected_minutes=pt_disc_min,
                                last_played_timestamp=lp_time,
                            )
            except (PermissionError, OSError, Exception):
                continue

        return playtimes

    def scan_full_library(self, steam_id_64: Optional[str] = None) -> SteamLocalLibrary:
        """Assembles complete local Steam state across all discovered libraries and accounts."""
        if not self.is_steam_installed():
            return SteamLocalLibrary(
                deliberation=f"Steam installation not found at {self.steam_root}. Returning empty library.",
                steam_root=str(self.steam_root),
                users=[],
                active_user=None,
                library_folders=[],
                installed_apps={},
                playtimes={},
                total_installed_bytes=0,
                total_playtime_hours=0.0,
            )

        users = self.get_login_users()
        active_user = users[0] if users else None
        folders = self.discover_library_folders()
        installed_apps = self.scan_installed_apps()
        playtimes = self.scan_user_playtimes(steam_id_64=steam_id_64)

        total_bytes = sum(m.size_on_disk for m in installed_apps.values())
        total_hours = sum(p.playtime_hours for p in playtimes.values())

        return SteamLocalLibrary(
            deliberation=f"Scanned local library: {len(installed_apps)} installed apps, {len(playtimes)} played games, {len(folders)} library folders, {len(users)} users.",
            steam_root=str(self.steam_root),
            users=users,
            active_user=active_user,
            library_folders=folders,
            installed_apps=installed_apps,
            playtimes=playtimes,
            total_installed_bytes=total_bytes,
            total_playtime_hours=round(total_hours, 2),
        )

    def scan_local_library(self, steam_id_64: Optional[str] = None) -> SteamLocalLibrary:
        """Backward-compatible alias for scan_full_library()."""
        return self.scan_full_library(steam_id_64=steam_id_64)

    def to_dataset(
        self,
        local_lib: Optional[SteamLocalLibrary] = None,
        name_map: Optional[Dict[int, str]] = None,
        genre_map: Optional[Dict[int, List[str]]] = None,
    ) -> SteamLibraryDataset:
        """Converts local library scan into a SteamLibraryDataset for the DNA profiler.
        
        Includes the union of ALL games across installed manifests, library folders, and userdata playtimes.
        """
        lib = local_lib or self.scan_full_library()
        player_id = lib.active_user.steam_id_64 if lib.active_user else "local_player"

        # Union of all known AppIDs across installed apps, played apps, and library folders
        all_appids = set(lib.installed_apps.keys()) | set(lib.playtimes.keys())
        for folder in lib.library_folders:
            all_appids.update(folder.apps.keys())

        records: List[SteamGameRecord] = []
        for appid in sorted(all_appids):
            manifest = lib.installed_apps.get(appid)
            pt = lib.playtimes.get(appid)

            # Determine best available name
            if manifest and manifest.name:
                name = manifest.name
            elif name_map and appid in name_map:
                name = name_map[appid]
            elif self.name_cache and appid in self.name_cache:
                name = self.name_cache[appid]
            else:
                name = f"App {appid}"

            pt_forever = pt.playtime_minutes if pt else 0
            pt_2weeks = pt.playtime_2weeks_minutes if pt else None
            lp_timestamp = pt.last_played_timestamp if pt else None
            genres = (genre_map.get(appid) if genre_map else []) or []

            records.append(
                SteamGameRecord(
                    deliberation=f"Ingested Steam game {name} ({appid}): {pt_forever}m forever, {pt_2weeks}m 2wks.",
                    appid=appid,
                    name=name,
                    playtime_forever_minutes=pt_forever,
                    playtime_2weeks_minutes=pt_2weeks,
                    rtime_last_played=lp_timestamp,
                    genres=genres,
                )
            )

        return SteamLibraryDataset(
            deliberation=f"Constructed SteamLibraryDataset with {len(records)} games ({len(lib.installed_apps)} installed, {len(lib.playtimes)} with playtime).",
            player_id=player_id,
            source="local_scan",
            games=records,
        )
