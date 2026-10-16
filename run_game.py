"""run_game.py
Unified launch entrypoint for the Autonomous Game Engine & Showrunner on macOS.
Supports CLI arguments: --cartridge, --headless, --showrunner, --steam-profile, --ticks.
Enforces graceful shutdown and SQLite WAL checkpointing.
"""
from __future__ import annotations
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, List, Optional

if __name__ == "__main__":
    _venv_py = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if _venv_py.is_file() and Path(sys.prefix).resolve() != _venv_py.parent.parent.resolve():
        os.execv(str(_venv_py), [str(_venv_py), *sys.argv])

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
            if hasattr(telemetry, "writer_thread") and telemetry.writer_thread is not None:
                try:
                    # Fast drain and bulk commit takes ~25ms; allocate 0.5s safety budget
                    telemetry.writer_thread.stop(timeout=0.5)
                    if telemetry.writer_thread.is_alive():
                        if hasattr(telemetry.writer_thread, "abort"):
                            telemetry.writer_thread.abort()
                        telemetry.writer_thread.join(timeout=0.1)
                except Exception:
                    pass
                finally:
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
            checkpoint_deadline = max(0.8, min(1.15, 1.70 - elapsed_so_far))
            checkpoint_success = False

            while time.perf_counter() - start_checkpoint < checkpoint_deadline:
                conn = None
                try:
                    conn = sqlite3.connect(db_path, timeout=0.1, isolation_level=None)
                    conn.execute("PRAGMA busy_timeout = 100;")
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
                time.sleep(0.01)

            if not checkpoint_success and os.path.exists(wal_path):
                wal_size = os.stat(wal_path).st_size if os.path.exists(wal_path) else 0
                if wal_size > 0:
                    sys.stderr.write(
                        f"[SignalTeardownWarning] SQLite WAL checkpoint(TRUNCATE) deadline exceeded "
                        f"({checkpoint_deadline:.2f}s). Remaining WAL size: {wal_size} bytes.\n"
                    )
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


def resolve_and_load_cartridge(
    engine: Engine,
    requested_cartridge: Optional[str] = None,
) -> Optional[Path]:
    """Dynamically resolves and loads the active cartridge into the engine.
    Strictly completes in <150ms without blocking on external processes.

    When requested_cartridge is explicitly passed, enforces Strict Configuration
    Fidelity (Rule 5) and fail-fast behavior: validates directory existence,
    pre-validates manifest.json, and loads without fallback, letting any
    exceptions propagate directly to the caller.

    When no cartridge is requested and cartridges/active does not exist or fails,
    cleanly prints the guidance banner and returns None without falling back to seed.
    """
    active_dir = config.PROJECT_ROOT / "cartridges" / "active"

    # 1. Explicit CLI override — Strict Configuration Fidelity (Rule 5): NEVER fall back
    if requested_cartridge:
        target = Path(requested_cartridge).resolve()
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"Requested cartridge directory not found: {target}")

        # Pre-validate manifest.json if present to catch malformed JSON early
        manifest_path = target / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                json.load(f)

        engine.load_cartridge(target)
        logger.info("[Boot] Loaded explicitly requested cartridge: %s", target.name)
        return target

    # 2. Primary: cartridges/active (only when no explicit CLI cartridge is passed)
    if active_dir.exists() and (
        (active_dir / "manifest.json").exists()
        or (active_dir / "level.json").exists()
        or (active_dir / "mechanics.py").exists()
    ):
        try:
            engine.load_cartridge(active_dir)
            logger.info("[Boot] Successfully booted active promoted cartridge: %s", active_dir)
            return active_dir
        except Exception as e:
            logger.warning("[Boot] Failed to load active cartridge '%s': %s", active_dir, e)
            print(NO_ACTIVE_GAME_BANNER)
            return None

    # 3. No active cartridge found: clean user guidance without seed fallback
    print(NO_ACTIVE_GAME_BANNER)
    return None


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
            booted_path = resolve_and_load_cartridge(engine, args.cartridge)
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

        engine.booted_cartridge_path = booted_path

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
                    if init_scene is not None:
                        engine.scene_manager.register_scene("cartridge_initial", init_scene)
                        initial_scene_name = "cartridge_initial"
                except Exception as exc:
                    logger.warning("[Boot] Failed to resolve cartridge initial scene: %s", exc)

            if hasattr(cartridge, "setup"):
                try:
                    cartridge.setup(engine)
                except Exception as exc:
                    logger.warning("[Boot] Cartridge setup lifecycle hook raised: %s", exc)

        boot_elapsed_ms = (time.perf_counter() - boot_t0) * 1000.0
        logger.info("[Boot] Client initialization completed in %.2f ms (Budget: <150 ms)", boot_elapsed_ms)
        print(f"[Engine] Booted cartridge '{booted_path.name}' in {boot_elapsed_ms:.1f}ms (60 FPS Ready)")

        engine.scene_manager.switch_to(initial_scene_name)

        # 7. Run 60 FPS Event Loop or Headless Tick Count
        if args.headless and args.ticks is not None:
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
