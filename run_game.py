"""run_game.py
Unified launch entrypoint for the Autonomous Game Engine & Showrunner on macOS.
Supports CLI arguments: --cartridge, --headless, --showrunner, --steam-profile, --ticks.
Enforces graceful shutdown and SQLite WAL checkpointing.
"""
from __future__ import annotations
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

# Active Godot 4 child process tracking for signal cleanup
_ACTIVE_GODOT_PROC: Optional[subprocess.Popen] = None

# Tier 1 Early Signal Interception & Unmasking: Active immediately at line 1 of script execution
def _early_signal_handler(signum: int, frame: Any) -> None:
    print(f"\n[Engine] Signal {signum} received during early bootstrap. Shutting down cleanly...", flush=True)
    os._exit(0)

try:
    signal.signal(signal.SIGINT, _early_signal_handler)
    signal.signal(signal.SIGTERM, _early_signal_handler)
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
except (ValueError, AttributeError, OSError):
    pass

# Active subsystem registry for unified signal lifecycle management across all startup & runtime phases
_ACTIVE_SUBSYSTEMS: Dict[str, Any] = {
    "engine": None,
    "telemetry": None,
    "bridge": None,
}
_IN_SIGNAL_HANDLER = False


def handle_signal(signum: int, frame: Any) -> None:
    """Unified signal handler registered at line 1 of process boot.
    Guarantees that SIGINT or SIGTERM received at any microsecond of process startup
    (including during module imports) or active gameplay cleanly flushes SQLite WAL
    and exits immediately via os._exit(0) without C-extension exception clearing or
    Py_FinalizeEx thread-join hangs.
    """
    global _IN_SIGNAL_HANDLER
    if _IN_SIGNAL_HANDLER:
        return
    _IN_SIGNAL_HANDLER = True
    teardown_t0 = time.perf_counter()

    print(f"\n[Engine] Signal {signum} received. Flushing SQLite WAL and shutting down...", flush=True)
    engine = _ACTIVE_SUBSYSTEMS.get("engine")
    bridge = _ACTIVE_SUBSYSTEMS.get("bridge")
    telemetry = _ACTIVE_SUBSYSTEMS.get("telemetry")

    if engine is not None:
        try:
            engine.is_running = False
        except Exception:
            pass
    if bridge is not None:
        try:
            if hasattr(bridge, "scheduler") and hasattr(bridge.scheduler, "shutdown"):
                bridge.scheduler.shutdown(wait=False)
            else:
                bridge.shutdown()
        except Exception:
            pass

    # Note: engine.quit() (which calls pygame.quit()) is strictly omitted inside handle_signal.
    # Calling pygame.quit() inside an asynchronous POSIX signal handler while Pygame C-extensions
    # or the display server are active on the main thread causes deadlocks and latency stalls.
    # os._exit(0) immediately releases all display buffers and process resources directly to the kernel.
    if telemetry is not None:
        try:
            if hasattr(telemetry, "_closed") and not telemetry._closed:
                telemetry._closed = True
                if getattr(telemetry, "current_episode_id", None):
                    try:
                        telemetry.end_episode(outcome="ABORTED")
                    except Exception:
                        pass
                if getattr(telemetry, "session_id", None):
                    try:
                        telemetry.end_session()
                    except Exception:
                        pass
            writer_th = getattr(telemetry, "writer_thread", None)
            if writer_th is not None:
                try:
                    # Drain and bulk commit takes ~25ms; allocate 0.20s safety budget
                    writer_th.stop(timeout=0.20)
                    if writer_th.is_alive():
                        if hasattr(writer_th, "abort"):
                            writer_th.abort()
                        writer_th.join(timeout=0.05)
                except Exception:
                    pass
                finally:
                    # Explicitly ensure writer_thread._conn is closed to release all SQLite locks
                    conn = getattr(writer_th, "_conn", None)
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        writer_th._conn = None
                    telemetry.writer_thread = None
        except Exception:
            pass

    # Resilient SQLite WAL TRUNCATE checkpoint with return value inspection & retry loop
    try:
        db_path = None
        if telemetry is not None and hasattr(telemetry, "db") and hasattr(telemetry.db, "db_path"):
            db_path = str(telemetry.db.db_path)
        elif "config" in sys.modules:
            import config
            db_path = str(config.DB_PATH)
        elif os.path.exists("data/telemetry.db"):
            db_path = "data/telemetry.db"
        if db_path and os.path.exists(db_path):
            import sqlite3
            wal_path = str(db_path) + "-wal"
            start_checkpoint = time.perf_counter()
            elapsed_so_far = start_checkpoint - teardown_t0
            checkpoint_deadline = max(1.0, min(3.5, 4.0 - elapsed_so_far))
            checkpoint_success = False
            retry_count = 0

            while time.perf_counter() - start_checkpoint < checkpoint_deadline:
                conn = None
                try:
                    rem = checkpoint_deadline - (time.perf_counter() - start_checkpoint)
                    busy_ms = max(200, min(1000, int(rem * 1000)))
                    conn = sqlite3.connect(db_path, timeout=rem, isolation_level=None)
                    conn.execute(f"PRAGMA busy_timeout = {busy_ms};")
                    cur = conn.cursor()
                    cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    res = cur.fetchone()
                    # SQLite PRAGMA wal_checkpoint returns (busy, log, checkpointed)
                    # busy == 0 indicates all frames were checkpointed and WAL was truncated
                    if res and res[0] == 0:
                        # Verify physical filesystem state: WAL file either absent or 0 bytes
                        if not os.path.exists(wal_path) or os.stat(wal_path).st_size == 0:
                            checkpoint_success = True
                            break
                except Exception:
                    pass
                finally:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                retry_count += 1
                # Exponential backoff with jitter (20ms to 50ms) allowing concurrent readers to yield shared locks
                sleep_sec = min(0.05, 0.02 + retry_count * 0.003 + (random.random() * 0.008))
                time.sleep(sleep_sec)

            if not checkpoint_success and os.path.exists(wal_path):
                wal_size = os.stat(wal_path).st_size if os.path.exists(wal_path) else 0
                if wal_size > 0:
                    sys.stderr.write(
                        f"[SignalTeardownWarning] SQLite WAL checkpoint(TRUNCATE) deadline exceeded "
                        f"({checkpoint_deadline:.2f}s). Remaining WAL size: {wal_size} bytes.\n"
                    )
    except Exception:
        pass

    global _ACTIVE_GODOT_PROC
    if _ACTIVE_GODOT_PROC is not None:
        try:
            _ACTIVE_GODOT_PROC.terminate()
            _ACTIVE_GODOT_PROC.wait(timeout=0.2)
        except Exception:
            try:
                _ACTIVE_GODOT_PROC.kill()
            except Exception:
                pass

    print("Game and telemetry shutdown completed cleanly.", flush=True)
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)


# Top-of-file signal registration: active before ANY heavy or third-party imports
try:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
except (ValueError, AttributeError):
    # Only the main thread of the main interpreter can register signal handlers
    pass

# Strict Dependency Verification & Clean Guidance (Rule 5 compliance)
try:
    import pygame
except ImportError:
    sys.stderr.write(
        "\n[EngineError] Required dependency 'pygame-ce' is not installed in the active Python environment.\n"
        f"Active interpreter: {sys.executable}\n"
        "Please run via the single-command launcher:\n"
        "    ./run_game.sh\n"
        "Or activate the virtual environment:\n"
        "    source .venv/bin/activate\n\n"
    )
    sys.exit(1)

import argparse
import json
import logging
from pathlib import Path
import time

import config
from engine.core.engine import Engine
from engine.core.scenes.game_over_scene import GameOverScene
from engine.core.scenes.playing_scene import PlayingScene
from engine.core.scenes.title_scene import TitleScene
from engine.core.scenes.victory_scene import EpisodeVictoryScene
from engine.loader import ASTSecurityError, CartridgeLoader

try:
    from engine.loader import CartridgeError
except ImportError:
    class CartridgeError(Exception):
        """Base exception for cartridge loading and validation failures."""
        pass

from engine.steam.dna import SteamDNAPersistence
from engine.steam.models import SteamPlayerDNA
from engine.telemetry.recorder import TelemetryRecorder

logger = logging.getLogger("GameClientBoot")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for the game engine and showrunner runtime."""
    parser = argparse.ArgumentParser(
        prog="run_game.py",
        description="Autonomous Game Engine & Showrunner — Native macOS 60 FPS Event Loop & Continuous Studio",
    )
    parser.add_argument(
        "--cartridge",
        type=str,
        default=None,
        help="Path to the game cartridge directory to boot (default: cartridges/active)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path or identifier of a verified cartridge checkpoint to boot (e.g. candidate_genesis_8993a2_R1)",
    )
    parser.add_argument(
        "--latest-checkpoint",
        action="store_true",
        default=False,
        help="Boot the latest verified cartridge checkpoint from cartridges/checkpoints/",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless simulation mode without opening an OS display window",
    )
    parser.add_argument(
        "--showrunner",
        action="store_true",
        default=False,
        help="Enable background Showrunner analysis, director, compiler, and supervisor loop",
    )
    parser.add_argument(
        "--steam-profile",
        type=str,
        default="default",
        help="Steam profile name, account ID, or archetype tag (e.g. rusher, cautious)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Maximum simulation frames to run before terminating cleanly (useful for tests)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=config.TARGET_FPS,
        help="Target frame rate (default: 60)",
    )
    return parser.parse_args(argv)


def resolve_steam_dna(profile_tag: str) -> Optional[SteamPlayerDNA]:
    """Resolves or synthesizes a deterministic SteamPlayerDNA profile for the player."""
    repo = SteamDNAPersistence(config.DB_PATH)
    try:
        dna = repo.get_latest_dna(player_id=profile_tag, profile_tag=profile_tag)
        if dna is not None:
            return dna
    except Exception:
        pass

    # Construct deterministic profile if unpopulated
    tag = profile_tag.lower()
    if "cautious" in tag or "tactical" in tag:
        dna = SteamPlayerDNA(
            deliberation="Synthesized methodical/tactical Steam DNA profile.",
            player_id=profile_tag,
            profile_tag="methodical_tactical",
            top_genres=["Strategy", "RPG", "Adventure"],
            genre_weights={"Strategy": 0.85, "RPG": 0.75, "Adventure": 0.60},
            total_playtime_hours=240.0,
            challenge_tolerance=0.45,
            pacing_score=-0.65,
            pacing_preference="tactical_methodical",
            preferred_palette_mood="grim_dark",
        )
    else:
        dna = SteamPlayerDNA(
            deliberation="Synthesized aggressive/action Steam DNA profile.",
            player_id=profile_tag,
            profile_tag="high_mobility_rusher",
            top_genres=["Action", "Roguelike", "BulletHell"],
            genre_weights={"Action": 0.90, "Roguelike": 0.85, "BulletHell": 0.70},
            total_playtime_hours=90.0,
            challenge_tolerance=0.75,
            pacing_score=0.85,
            pacing_preference="kinetic_high_mobility",
            preferred_palette_mood="neon_cyberpunk",
        )
    try:
        repo.save_dna(dna)
    except Exception:
        pass
    return dna


NO_ACTIVE_GAME_BANNER = (
    "======================================================================\n"
    "[Engine] No Active Game Cartridge Found in cartridges/active/\n"
    "======================================================================\n"
    "The Autonomous Agentic Game Studio builds games from scratch based on\n"
    "your Steam library DNA. No game has been commissioned or built yet.\n\n"
    "To commission and build your first game, please run:\n"
    "    python run_studio.py\n"
    "======================================================================"
)


def run_godot_cartridge(
    cartridge_dir: Path,
    headless: bool = False,
    ticks: Optional[int] = None,
    telemetry: Optional[Any] = None,
) -> int:
    """Launches Godot 4 binary on target cartridge directory with continuous telemetry streaming."""
    global _ACTIVE_GODOT_PROC
    from engine.godot_runtime import ensure_godot_binary
    godot_bin = ensure_godot_binary()

    cmd = [str(godot_bin), "--path", str(cartridge_dir)]
    if headless:
        cmd.append("--headless")
        from engine.sim.godot_verifier import HARNESS_SCRIPT_PATH
        cmd.extend(["-s", str(HARNESS_SCRIPT_PATH)])
        if ticks:
            cmd.extend(["--max-ticks", str(ticks)])
    elif ticks:
        from engine.sim.godot_verifier import HARNESS_SCRIPT_PATH
        cmd.extend(["-s", str(HARNESS_SCRIPT_PATH), "--max-ticks", str(ticks)])

    print(f"[Godot] Booting Godot 4 cartridge from: {cartridge_dir}", flush=True)
    print(f"[Godot] Command: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(cartridge_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    _ACTIVE_GODOT_PROC = proc

    # Stream stdout telemetry into telemetry recorder
    def _read_stdout():
        if proc.stdout:
            for line in proc.stdout:
                line_str = line.strip()
                print(f"[Godot stdout] {line_str}", flush=True)
                if "[TELEMETRY_EVENT]" in line_str or "[TELEMETRY]" in line_str:
                    tag = "[TELEMETRY_EVENT]" if "[TELEMETRY_EVENT]" in line_str else "[TELEMETRY]"
                    try:
                        raw_json = line_str.split(tag, 1)[1].strip()
                        ev = json.loads(raw_json)
                        pname = ev.get("probe_name") or ev.get("name", "custom_probe")
                        pcat = ev.get("category", "general")
                        pdata = ev.get("payload") or ev.get("data", {})
                        if telemetry is not None and hasattr(telemetry, "emit_custom_probe"):
                            telemetry.emit_custom_probe(
                                probe_name=pname,
                                category=pcat,
                                data=pdata,
                            )
                    except Exception as err:
                        logger.debug("Failed parsing telemetry event: %s", err)

    out_th = threading.Thread(target=_read_stdout, daemon=True)
    out_th.start()

    # Stream stderr
    def _read_stderr():
        if proc.stderr:
            for line in proc.stderr:
                print(f"[Godot stderr] {line.strip()}", file=sys.stderr, flush=True)

    err_th = threading.Thread(target=_read_stderr, daemon=True)
    err_th.start()

    try:
        ret = proc.wait()
    except KeyboardInterrupt:
        print("\n[Godot] KeyboardInterrupt received. Terminating Godot process...", flush=True)
        proc.terminate()
        try:
            ret = proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            ret = proc.wait()
    finally:
        _ACTIVE_GODOT_PROC = None
        if out_th is not None and out_th.is_alive():
            out_th.join(timeout=2.0)
        if err_th is not None and err_th.is_alive():
            err_th.join(timeout=2.0)
        if proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass
        if proc.stderr:
            try:
                proc.stderr.close()
            except Exception:
                pass
        if telemetry is not None:
            if hasattr(telemetry, "flush"):
                try:
                    telemetry.flush()
                except Exception:
                    pass
            if hasattr(telemetry, "close"):
                try:
                    telemetry.close()
                except Exception:
                    pass

    return ret


def resolve_and_load_cartridge(
    engine: Engine,
    requested_cartridge: Optional[str] = None,
    requested_checkpoint: Optional[str] = None,
    use_latest_checkpoint: bool = False,
) -> Optional[Path]:
    """Dynamically resolves and loads the active cartridge or checkpoint into the engine.
    Strictly completes in <150ms without blocking on external processes.

    When requested_cartridge or requested_checkpoint is explicitly passed, enforces
    Strict Configuration Fidelity (Rule 5) and fail-fast behavior.

    When no cartridge is requested and cartridges/active does not exist, automatically
    falls back to the latest verified checkpoint from cartridges/checkpoints/ if one exists.
    If neither exists, prints the guidance banner and returns None.
    """
    checkpoints_dir = config.PROJECT_ROOT / "cartridges" / "checkpoints"

    # 1. Explicit checkpoint request — Strict Configuration Fidelity (Rule 5)
    if requested_checkpoint:
        target = Path(requested_checkpoint)
        if not target.is_absolute():
            candidate = checkpoints_dir / requested_checkpoint
            if candidate.is_dir():
                target = candidate
            else:
                target = target.resolve()
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"Requested checkpoint directory not found: {requested_checkpoint}")
        if (target / "project.godot").is_file():
            logger.info("[Boot] Loaded explicitly requested Godot checkpoint: %s", target.name)
            return target
        engine.load_cartridge(target)
        logger.info("[Boot] Loaded explicitly requested checkpoint: %s", target.name)
        return target

    # 2. Latest checkpoint request
    if use_latest_checkpoint:
        checkpoints = []
        if checkpoints_dir.exists():
            checkpoints = [
                p for p in checkpoints_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".") and ((p / "project.godot").is_file() or (p / "mechanics.py").is_file())
            ]
        # Automatic fallback: scan cartridges/candidate_* with completed tracks if checkpoints/ not populated yet
        if not checkpoints:
            cartridges_root = config.PROJECT_ROOT / "cartridges"
            for cand in cartridges_root.glob("candidate*"):
                if cand.is_dir() and not cand.name.startswith(".") and ((cand / "project.godot").is_file() or (cand / "mechanics.py").is_file()):
                    req_file = cand / "requirements_spec.json"
                    if req_file.is_file():
                        try:
                            req_data = json.loads(req_file.read_text(encoding="utf-8"))
                            completed = [r for r in req_data.get("requirements", []) if r.get("status") == "completed"]
                            if completed:
                                checkpoints.append(cand)
                        except Exception:
                            pass
        if not checkpoints:
            raise FileNotFoundError("No verified checkpoints found in cartridges/checkpoints/ or cartridges/candidate_*/")
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        if (latest / "project.godot").is_file():
            logger.info("[Boot] Loaded latest verified Godot checkpoint: %s", latest.name)
            return latest
        engine.load_cartridge(latest)
        logger.info("[Boot] Loaded latest verified checkpoint: %s", latest.name)
        return latest

    # 3. Explicit CLI cartridge override — Strict Configuration Fidelity (Rule 5): NEVER fall back
    if requested_cartridge:
        target = Path(requested_cartridge).resolve()
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"Requested cartridge directory not found: {target}")

        if (target / "project.godot").is_file():
            logger.info("[Boot] Loaded explicitly requested Godot cartridge: %s", target.name)
            return target

        manifest_path = target / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                json.load(f)

        engine.load_cartridge(target)
        logger.info("[Boot] Loaded explicitly requested cartridge: %s", target.name)
        return target

    # 4. Primary: cartridges/active (only when no explicit CLI cartridge is passed)
    active_dir = config.PROJECT_ROOT / "cartridges" / "active"
    if active_dir.exists() and (
        (active_dir / "project.godot").exists()
        or (active_dir / "manifest.json").exists()
        or (active_dir / "level.json").exists()
        or (active_dir / "mechanics.py").exists()
    ):
        if (active_dir / "project.godot").is_file():
            logger.info("[Boot] Successfully booted active promoted Godot cartridge: %s", active_dir)
            return active_dir
        try:
            engine.load_cartridge(active_dir)
            logger.info("[Boot] Successfully booted active promoted cartridge: %s", active_dir)
            return active_dir
        except Exception as e:
            logger.warning("[Boot] Failed to load active cartridge '%s': %s", active_dir, e)
            print(NO_ACTIVE_GAME_BANNER)
            return None

    # 5. Live In-Progress Progression Fallback:
    # If cartridges/active does not exist yet during a build, check if a verified checkpoint is playable
    candidate_checkpoints = []
    if checkpoints_dir.exists():
        candidate_checkpoints = [
            p for p in checkpoints_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".") and ((p / "project.godot").is_file() or (p / "mechanics.py").is_file())
        ]
    if candidate_checkpoints:
        latest = max(candidate_checkpoints, key=lambda p: p.stat().st_mtime)
        if (latest / "project.godot").is_file():
            logger.info("[Boot] No cartridges/active found; automatically booted latest verified Godot checkpoint: %s", latest.name)
            print(f"[Engine] Playing latest verified checkpoint: {latest.name}")
            return latest
        try:
            engine.load_cartridge(latest)
            logger.info("[Boot] No cartridges/active found; automatically booted latest verified checkpoint: %s", latest.name)
            print(f"[Engine] Playing latest verified checkpoint: {latest.name}")
            return latest
        except Exception as e:
            logger.warning("[Boot] Failed to load latest checkpoint '%s': %s", latest, e)

    # 6. No active cartridge or checkpoint found: clean user guidance without seed fallback
    print(NO_ACTIVE_GAME_BANNER)
    return None


def catalog_cartridge_probes(telemetry_recorder: Optional[TelemetryRecorder], cartridge_dir: Path) -> int:
    """Catalogs custom telemetry probes declared in cartridge's telemetry_manifest.json.

    Ensures foreign key referential integrity during client boot without slow row-by-row
    rescue fallbacks. Executes non-blockingly in sub-millisecond time.
    """
    if telemetry_recorder is None or not hasattr(telemetry_recorder, "register_probe"):
        return 0
    manifest_path = Path(cartridge_dir) / "telemetry_manifest.json"
    if not manifest_path.is_file():
        return 0
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        count = 0
        for p in manifest_data.get("probes", []):
            if isinstance(p, dict) and p.get("probe_name"):
                telemetry_recorder.register_probe(
                    probe_id=str(p.get("probe_id") or f"prb_{p['probe_name']}"),
                    probe_name=str(p["probe_name"]),
                    category=str(p.get("category", "general")),
                    description=str(p.get("description", "")),
                    hypothesis_target=str(p.get("hypothesis_target", "")),
                    schema_json=p.get("payload_schema", {}),
                )
                count += 1
        # Ensure standard built-in engine probes are registered
        telemetry_recorder.register_probe(
            probe_id="probe_dialogue_bark_played",
            probe_name="dialogue_bark_played",
            category="dialogue",
            description="Reactive dialogue bark played in PlayingScene",
            hypothesis_target="H_DIALOGUE_ENGAGEMENT",
            schema_json={},
        )
        telemetry_recorder.register_probe(
            probe_id="probe_cartridge_hot_reload_triggered",
            probe_name="cartridge_hot_reload_triggered",
            category="system",
            description="In-game cartridge hot-reload event",
            hypothesis_target="H_HOT_RELOAD",
            schema_json={},
        )
        return count
    except Exception as exc:
        logger.warning("[Boot] Failed to catalog cartridge probes from %s: %s", manifest_path, exc)
        return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Initializes telemetry recording, game engine subsystems, and starts 60 FPS loop."""
    boot_t0 = time.perf_counter()
    args = parse_args(argv)

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    # Ensure signal handlers are active in the execution context
    try:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    except (ValueError, AttributeError):
        pass

    engine: Optional[Engine] = None
    telemetry: Optional[TelemetryRecorder] = None
    bridge: Optional[Any] = None

    print("=" * 60)
    print("Autonomous Game Engine & Showrunner Harness")
    print("Native macOS 60 FPS Event Loop | Continuous SQLite WAL")
    print(f"Mode: {'HEADLESS' if args.headless else 'COCOA'} | Showrunner: {'ON' if args.showrunner else 'OFF'}")
    print(f"Steam Profile: {args.steam_profile}")
    print("=" * 60)

    try:
        # 1. Resolve Steam DNA Profile
        steam_dna = resolve_steam_dna(args.steam_profile)

        # 2. Initialize Continuous SQLite WAL Telemetry Recorder
        telemetry = TelemetryRecorder(
            db_path=config.DB_PATH,
            batch_size=config.TELEMETRY_BATCH_SIZE,
            flush_interval_sec=config.TELEMETRY_FLUSH_INTERVAL,
            auto_register_probes=True,
        )
        _ACTIVE_SUBSYSTEMS["telemetry"] = telemetry
        telemetry.start_session(
            player_profile_tag=args.steam_profile,
            platform="macOS",
            resolution=f"{config.CANVAS_WIDTH}x{config.CANVAS_HEIGHT}",
            target_fps=args.fps,
        )

        # 3. Initialize Game Engine
        engine = Engine(headless=args.headless, title=config.WINDOW_TITLE)
        _ACTIVE_SUBSYSTEMS["engine"] = engine
        engine.set_telemetry_recorder(telemetry)
        engine.steam_profile_tag = args.steam_profile
        engine.steam_dna = steam_dna

        # 4. Initialize Background Showrunner Runtime Bridge
        if args.showrunner:
            from engine.core.showrunner_bridge import ShowrunnerRuntimeBridge
            bridge = ShowrunnerRuntimeBridge(engine=engine, steam_dna=steam_dna)
            bridge.start()
            engine.showrunner_bridge = bridge
            _ACTIVE_SUBSYSTEMS["bridge"] = bridge
        else:
            engine.showrunner_bridge = None
            _ACTIVE_SUBSYSTEMS["bridge"] = None

        # 5. Resolve & Dynamically Load Game Cartridge (<150ms total boot)
        booted_path: Optional[Path] = None
        try:
            booted_path = resolve_and_load_cartridge(
                engine,
                requested_cartridge=args.cartridge,
                requested_checkpoint=getattr(args, "checkpoint", None),
                use_latest_checkpoint=getattr(args, "latest_checkpoint", False),
            )
        except (ASTSecurityError, FileNotFoundError, json.JSONDecodeError, CartridgeError, Exception) as e:
            logger.critical("[Boot] Fatal error loading cartridge: %s: %s", type(e).__name__, e)
            sys.stderr.write(f"Error: {type(e).__name__}: {e}\n")
            if bridge:
                try:
                    bridge.shutdown()
                except Exception:
                    pass
            if engine:
                try:
                    engine.quit()
                except Exception:
                    pass
            if telemetry:
                try:
                    telemetry.close()
                except Exception:
                    pass
            sys.exit(1)

        if booted_path is None:
            # Clean exit when no game has been built yet
            return 0

        if (booted_path / "project.godot").is_file():
            print(f"[Engine] Detected Godot 4 cartridge in '{booted_path.name}'. Launching Godot 4.3 runtime...")
            if engine:
                try:
                    engine.telemetry_recorder = None
                    engine.quit()
                    _ACTIVE_SUBSYSTEMS["engine"] = None
                except Exception:
                    pass
            catalog_cartridge_probes(telemetry, booted_path)
            return run_godot_cartridge(
                booted_path,
                headless=args.headless,
                ticks=getattr(args, "ticks", None),
                telemetry=telemetry,
            )

        engine.booted_cartridge_path = booted_path
        catalog_cartridge_probes(telemetry, booted_path)

        # 6. Register Scene State Machine
        engine.scene_manager.register_scene("title", TitleScene(engine))
        engine.scene_manager.register_scene("playing", PlayingScene(engine))
        engine.scene_manager.register_scene("game_over", GameOverScene(engine))
        engine.scene_manager.register_scene("victory", EpisodeVictoryScene(engine))

        # Check if cartridge provides open-genre custom scenes
        cartridge = getattr(engine, "active_cartridge", None)
        initial_scene_name = "title"
        if cartridge is not None:
            if hasattr(cartridge, "get_scenes"):
                try:
                    custom_scenes = cartridge.get_scenes(engine) or {}
                    for s_name, s_inst in custom_scenes.items():
                        engine.scene_manager.register_scene(s_name, s_inst)
                except Exception as exc:
                    logger.warning("[Boot] Failed to register custom cartridge scenes: %s", exc)

            if hasattr(cartridge, "get_initial_scene"):
                try:
                    init_scene = cartridge.get_initial_scene(engine)
                    if isinstance(init_scene, str) and init_scene in engine.scene_manager.scenes:
                        initial_scene_name = init_scene
                    elif init_scene is not None and hasattr(init_scene, "update"):
                        engine.scene_manager.register_scene("cartridge_initial", init_scene)
                        initial_scene_name = "cartridge_initial"
                except Exception as exc:
                    logger.warning("[Boot] Failed to resolve cartridge initial scene: %s", exc)
            elif "main" in engine.scene_manager.scenes:
                initial_scene_name = "main"

            if hasattr(cartridge, "setup"):
                try:
                    cartridge.setup(engine)
                except Exception as exc:
                    logger.warning("[Boot] Cartridge setup lifecycle hook raised: %s", exc)

        boot_elapsed_ms = (time.perf_counter() - boot_t0) * 1000.0
        logger.info("[Boot] Client initialization completed in %.2f ms (Budget: <150 ms)", boot_elapsed_ms)
        print(f"[Engine] Booted cartridge '{booted_path.name}' in {boot_elapsed_ms:.1f}ms (60 FPS Ready)")

        engine.scene_manager.switch_to(initial_scene_name)

        # 7. Run 60 FPS Event Loop or Bounded Tick Count
        if args.ticks is not None:
            engine.is_running = True
            for tick in range(args.ticks):
                if not engine.is_running:
                    break
                keep_running = engine.game_loop.step()
                if not keep_running:
                    break
        else:
            engine.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
    finally:
        _ACTIVE_SUBSYSTEMS["engine"] = None
        _ACTIVE_SUBSYSTEMS["telemetry"] = None
        _ACTIVE_SUBSYSTEMS["bridge"] = None
        if bridge:
            try:
                bridge.shutdown()
            except Exception:
                pass
        if engine:
            try:
                engine.quit()
            except Exception:
                pass
        if telemetry:
            try:
                telemetry.close()
            except Exception:
                pass
        print("Game and telemetry shutdown completed cleanly.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
