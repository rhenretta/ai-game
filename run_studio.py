#!/usr/bin/env python3
"""run_studio.py
Standalone CLI entrypoint for the Decoupled Real-Time Studio Observability UI ("Mission Control")
and Background Studio Daemon (Milestone M5, Features 28 & 31).
Runs the Starlette ASGI server via Uvicorn and manages the autonomous StudioDaemon loop.
"""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
import signal
import sys
import threading
from typing import List, Optional

if __name__ == "__main__":
    _venv_py = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if _venv_py.is_file() and Path(sys.prefix).resolve() != _venv_py.parent.parent.resolve():
        os.execv(str(_venv_py), [str(_venv_py), *sys.argv])

import uvicorn

import config
from engine.studio.daemon import StudioDaemon, StudioDaemonConfig

logger = logging.getLogger("StudioLauncher")

_active_daemon: Optional[StudioDaemon] = None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for the studio observability runner and background daemon."""
    parser = argparse.ArgumentParser(
        prog="run_studio.py",
        description="Autonomous Game Studio — Real-Time Observability UI & Background Daemon",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind the studio server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reloading upon file modifications (development mode)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level for Uvicorn (default: info)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="SQLite WAL telemetry polling interval in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Execute single daemon evaluation cycle and exit",
    )
    parser.add_argument(
        "--no-daemon",
        action="store_true",
        default=False,
        help="Disable background studio daemon; run ASGI web server only",
    )
    parser.add_argument(
        "--steam-profile",
        type=str,
        default="default",
        help="Steam player DNA profile identifier (default: default)",
    )
    parser.add_argument(
        "--cartridges-dir",
        type=str,
        default=None,
        help="Path to cartridges root directory",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite WAL telemetry database",
    )
    return parser.parse_args(argv)


def setup_signal_handlers(daemon: Optional[StudioDaemon] = None) -> None:
    """Registers graceful termination signals."""
    def _sig_handler(signum, frame):
        logger.info("Termination signal %s received. Shutting down studio...", signum)
        target = daemon or _active_daemon
        if target:
            target.stop()
            target.release_lock()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)


def main(argv: Optional[List[str]] = None) -> int:
    """Parses CLI parameters, sets up signals, and starts the Uvicorn ASGI server and StudioDaemon."""
    global _active_daemon
    args = parse_args(argv)

    daemon_cfg = StudioDaemonConfig(
        poll_interval=args.interval,
        run_once=args.once,
        enable_daemon=not args.no_daemon,
        steam_profile=args.steam_profile,
        cartridges_root=Path(args.cartridges_dir) if args.cartridges_dir else config.PROJECT_ROOT / "cartridges",
        db_path=Path(args.db_path) if args.db_path else config.DB_PATH,
    )

    if not args.no_daemon:
        _active_daemon = StudioDaemon(config_obj=daemon_cfg)
    setup_signal_handlers(_active_daemon)

    # If --once requested without ASGI server: execute single step and exit
    if args.once and args.no_daemon:
        daemon = StudioDaemon(config_obj=daemon_cfg)
        daemon.step()
        return 0

    print("=" * 64)
    print("Autonomous Game Studio — Mission Control Web Observability")
    print(f"Binding address: http://{args.host}:{args.port}")
    print(f"Deliberation bus: ws://{args.host}:{args.port}/ws/swarm")
    if not args.no_daemon:
        print(f"Background daemon: ACTIVE (poll_interval={args.interval}s)")
    print("=" * 64)

    daemon_thread: Optional[threading.Thread] = None
    if _active_daemon and not args.no_daemon:
        daemon_thread = threading.Thread(
            target=_active_daemon.run,
            name="StudioDaemonThread",
            daemon=True,
        )
        daemon_thread.start()

    try:
        uvicorn.run(
            "engine.ui.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )
        return 0
    except SystemExit:
        return 0
    except Exception as exc:
        logger.critical("Studio failed to launch: %s", exc)
        return 1
    finally:
        if _active_daemon:
            _active_daemon.stop()
            _active_daemon.release_lock()
        if daemon_thread and daemon_thread.is_alive():
            daemon_thread.join(timeout=2.0)


if __name__ == "__main__":
    sys.exit(main())
