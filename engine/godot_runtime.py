"""engine/godot_runtime.py
Portable, Container-Invariant Godot 4.3 Runtime Manager (Rule 6).
Handles auto-download, caching in .godot_runtime/, GODOT_BIN environment override,
and execution of headless and GUI Godot 4 processes.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import urllib.request
import uuid
import zipfile
from typing import List, Optional, Tuple

logger = logging.getLogger("GodotRuntime")

GODOT_VERSION = "4.3-stable"
GODOT_RELEASE_TAG = "4.3-stable"
RELEASE_BASE_URL = f"https://github.com/godotengine/godot/releases/download/{GODOT_RELEASE_TAG}/"

# Map (system, machine) -> (asset_filename, internal_binary_relative_path)
PLATFORM_DOWNLOAD_MAP = {
    ("Darwin", "x86_64"): ("Godot_v4.3-stable_macos.universal.zip", Path("Godot.app/Contents/MacOS/Godot")),
    ("Darwin", "arm64"): ("Godot_v4.3-stable_macos.universal.zip", Path("Godot.app/Contents/MacOS/Godot")),
    ("Linux", "x86_64"): ("Godot_v4.3-stable_linux.x86_64.zip", Path("Godot_v4.3-stable_linux.x86_64")),
    ("Linux", "amd64"): ("Godot_v4.3-stable_linux.x86_64.zip", Path("Godot_v4.3-stable_linux.x86_64")),
    ("Linux", "aarch64"): ("Godot_v4.3-stable_linux.arm64.zip", Path("Godot_v4.3-stable_linux.arm64")),
    ("Linux", "arm64"): ("Godot_v4.3-stable_linux.arm64.zip", Path("Godot_v4.3-stable_linux.arm64")),
}


def get_default_runtime_dir() -> Path:
    """Returns absolute path to <repo_root>/.godot_runtime/."""
    return Path(__file__).resolve().parent.parent / ".godot_runtime"


def get_platform_asset_info() -> Tuple[str, Path]:
    """Detects current OS and machine architecture and returns (asset_zip_name, binary_subpath)."""
    system = platform.system()
    machine = platform.machine().lower()

    # Normalize common architecture names
    if machine in ("x86_64", "amd64"):
        norm_machine = "x86_64"
    elif machine in ("arm64", "aarch64"):
        norm_machine = "arm64"
    else:
        norm_machine = machine

    key = (system, norm_machine)
    if key in PLATFORM_DOWNLOAD_MAP:
        return PLATFORM_DOWNLOAD_MAP[key]

    raise RuntimeError(
        f"Unsupported OS/architecture for portable Godot runtime: {system} {machine}. "
        f"Supported combinations: {list(PLATFORM_DOWNLOAD_MAP.keys())}"
    )


def _ensure_apple_silicon_native_binary(binary_path: Path) -> None:
    """On Apple Silicon Macs, strips the legacy x86_64 slice from the universal Godot Mach-O binary
    so macOS never launches it under Rosetta 2 (even if spawned from an x86_64 Python/shell process)
    and never triggers 'Support Ending for Intel-based Apps' notifications."""
    if platform.system() != "Darwin" or not binary_path.is_file():
        return
    try:
        is_arm64_capable = platform.machine() == "arm64"
        if not is_arm64_capable:
            hw = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True, timeout=5)
            is_arm64_capable = hw.returncode == 0 and hw.stdout.strip() == "1"
        if not is_arm64_capable:
            return
        archs_res = subprocess.run(["lipo", "-detailed_info", str(binary_path)], capture_output=True, text=True, timeout=10)
        if archs_res.returncode != 0:
            return
        info_txt = archs_res.stdout
        if "Fat header" in info_txt or "x86_64" in info_txt:
            tmp_arm = binary_path.with_name(binary_path.name + ".arm64_only")
            ext_res = subprocess.run(
                ["lipo", "-thin", "arm64", str(binary_path), "-output", str(tmp_arm)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if ext_res.returncode == 0 and tmp_arm.is_file():
                os.chmod(tmp_arm, 0o755)
                tmp_arm.replace(binary_path)
                app_dir = binary_path.parent.parent.parent
                if app_dir.suffix == ".app":
                    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app_dir)], capture_output=True, timeout=30)
                logger.info("Stripped legacy x86_64 slice from %s (pure arm64 Apple Silicon binary)", binary_path)
            else:
                tmp_arm.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("Could not strip x86_64 slice from %s: %s", binary_path, exc)


def verify_godot_binary(binary_path: Path) -> str:
    """Verifies that the binary executes and returns its version string."""
    if not binary_path.is_file():
        raise FileNotFoundError(f"Godot binary does not exist: {binary_path}")
    if not os.access(binary_path, os.X_OK):
        os.chmod(binary_path, 0o755)
    _ensure_apple_silicon_native_binary(binary_path)

    res = subprocess.run([str(binary_path), "--version"], capture_output=True, text=True, timeout=15)
    if res.returncode != 0:
        raise RuntimeError(f"Godot binary verification failed (code {res.returncode}): {res.stderr}")
    version_str = res.stdout.strip()
    if "4.3" not in version_str:
        logger.warning("Godot version mismatch: expected 4.3, got: %s", version_str)
    return version_str


def ensure_godot_binary(cache_dir: Optional[Path] = None) -> Path:
    """Retrieves or auto-downloads the Godot 4.3 binary.

    Priority:
    1. GODOT_BIN environment variable (raises FileNotFoundError if file is invalid).
    2. Local cache directory (.godot_runtime/).
    3. Automated download from official GitHub release into cache directory.
    """
    # 1. Environment Variable Override (Rule 5: Zero Silent Fallbacks)
    env_bin = os.environ.get("GODOT_BIN")
    if env_bin:
        env_path = Path(env_bin).resolve()
        if not env_path.is_file():
            raise FileNotFoundError(
                f"GODOT_BIN environment variable is set to '{env_bin}', but file was not found. "
                f"(Rule 5: Zero Silent Fallbacks)"
            )
        verify_godot_binary(env_path)
        logger.info("Using Godot binary from GODOT_BIN override: %s", env_path)
        return env_path

    # 2. Local Cache Check
    runtime_dir = cache_dir or get_default_runtime_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)

    asset_filename, subpath = get_platform_asset_info()
    target_bin = runtime_dir / subpath

    if target_bin.is_file():
        try:
            verify_godot_binary(target_bin)
            return target_bin
        except Exception as e:
            logger.warning("Existing cached Godot binary failed verification (%s). Re-downloading.", e)
            if target_bin.exists():
                try:
                    target_bin.unlink()
                except Exception:
                    pass

    # Quick optimization: check if explorer downloaded to /tmp/godot4_test
    tmp_candidate = Path("/tmp/godot4_test") / subpath
    if tmp_candidate.is_file():
        try:
            verify_godot_binary(tmp_candidate)
            target_bin.parent.mkdir(parents=True, exist_ok=True)
            if "Godot.app" in str(subpath):
                # Copy app bundle
                src_app = Path("/tmp/godot4_test/Godot.app")
                dst_app = runtime_dir / "Godot.app"
                if not dst_app.exists():
                    shutil.copytree(src_app, dst_app)
            else:
                shutil.copy2(tmp_candidate, target_bin)
            os.chmod(target_bin, 0o755)
            verify_godot_binary(target_bin)
            logger.info("Reused verified local Godot binary from /tmp/godot4_test into %s", target_bin)
            return target_bin
        except Exception as e:
            logger.debug("Could not reuse /tmp/godot4_test: %s", e)

    # 3. Download and Extract
    download_url = RELEASE_BASE_URL + asset_filename
    temp_zip = runtime_dir / f"download_{uuid.uuid4().hex[:8]}.zip"

    logger.info("Downloading portable Godot 4.3 from %s...", download_url)
    t0 = time.time()
    try:
        req = urllib.request.Request(download_url, headers={"User-Agent": "AutonomousStudioGodotRuntime/1.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(temp_zip, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)

        logger.info("Extracting %s into %s...", temp_zip.name, runtime_dir)
        with zipfile.ZipFile(temp_zip, "r") as zf:
            zf.extractall(runtime_dir)

    finally:
        if temp_zip.exists():
            temp_zip.unlink(missing_ok=True)

    dur = time.time() - t0
    logger.info("Godot 4.3 downloaded and extracted in %.2f seconds.", dur)

    # 4. Enforce Executable Permissions and Verify
    if not target_bin.is_file():
        # Search runtime_dir if zip had alternative nesting
        candidates = list(runtime_dir.glob(f"**/{subpath.name}"))
        if candidates and candidates[0].is_file():
            target_bin = candidates[0]
        else:
            raise FileNotFoundError(f"Failed to locate extracted Godot binary at expected path: {target_bin}")

    os.chmod(target_bin, 0o755)
    version = verify_godot_binary(target_bin)
    logger.info("Successfully initialized Godot 4.3 runtime: %s (%s)", target_bin, version)
    return target_bin


def ensure_godot_background_render_binary(cache_dir: Optional[Path] = None) -> Path:
    """Returns a Godot 4.3 binary configured for silent, non-focus-stealing background GPU viewport rendering.
    On macOS, maintains a dedicated `GodotBackground.app` bundle with `LSUIElement=True` in `Info.plist`
    (`NSApplicationActivationPolicyAccessory`) so automated QA filmstrip and scene renders never appear in the Dock,
    never show a splash screen, and never steal keyboard/window focus from the user, while keeping `Godot.app`
    as a normal foreground app for interactive 'Play Game' sessions.
    """
    fg_bin = ensure_godot_binary(cache_dir=cache_dir)
    if platform.system() != "Darwin" or "Godot.app" not in str(fg_bin):
        return fg_bin

    import plistlib

    fg_app = fg_bin.parent.parent.parent  # .../Godot.app
    # Ensure foreground Godot.app has LSUIElement=False so interactive Play Game works normally
    fg_plist = fg_app / "Contents" / "Info.plist"
    if fg_plist.is_file():
        try:
            with fg_plist.open("rb") as f:
                pl_data = plistlib.load(f)
            if pl_data.get("LSUIElement"):
                pl_data.pop("LSUIElement", None)
                with fg_plist.open("wb") as f:
                    plistlib.dump(pl_data, f)
                subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(fg_app)], capture_output=True)
        except Exception:
            pass

    bg_app = fg_app.parent / "GodotBackground.app"
    bg_bin = bg_app / "Contents" / "MacOS" / "Godot"
    bg_plist = bg_app / "Contents" / "Info.plist"

    if not bg_bin.is_file():
        try:
            if bg_app.exists():
                shutil.rmtree(bg_app, ignore_errors=True)
            shutil.copytree(fg_app, bg_app)
        except Exception as copy_exc:
            logger.warning("Could not create GodotBackground.app (%s); falling back to foreground binary", copy_exc)
            return fg_bin

    if bg_bin.is_file():
        _ensure_apple_silicon_native_binary(bg_bin)

    if bg_plist.is_file():
        try:
            with bg_plist.open("rb") as f:
                bg_data = plistlib.load(f)
            if bg_data.get("LSUIElement") is not True or bg_data.get("CFBundleIdentifier") != "org.godotengine.godot.background":
                bg_data["LSUIElement"] = True
                bg_data["CFBundleIdentifier"] = "org.godotengine.godot.background"
                with bg_plist.open("wb") as f:
                    plistlib.dump(bg_data, f)
                subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(bg_app)], capture_output=True)
        except Exception as plist_exc:
            logger.warning("Could not configure Info.plist on GodotBackground.app: %s", plist_exc)

    return bg_bin if bg_bin.is_file() else fg_bin


def run_godot_command(
    args: List[str],
    cwd: Optional[Path] = None,
    timeout: float = 30.0,
    headless: bool = True,
    godot_bin: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Executes a Godot CLI command with automatic binary location."""
    binary = godot_bin or ensure_godot_binary()
    cmd = [str(binary)]
    if headless and "--headless" not in args:
        cmd.append("--headless")
    cmd.extend(args)

    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
