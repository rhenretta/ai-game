"""engine/studio/daemon.py
Decoupled Background Studio Daemon (Milestone M5, Feature 31).
Coordinates autonomous game evolution as an isolated background service (Process 2).
Polls SQLite WAL telemetry with query_only=ON, manages studio.lock, triggers cold-boot
genesis, drives BuildSupervisor 5-gate pipeline with self-healing, and broadcasts
real-time multi-agent events to Mission Control WebSockets.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, Field, field_validator

import config
from config import PROJECT_ROOT
from engine.compiler.compiler import EngineCompiler
from engine.studio.models import StudioBuildState
from engine.studio.orchestrator import StudioOrchestrator
from engine.supervisor.models import BuildEvaluationReport
from engine.supervisor.supervisor import BuildSupervisor
from engine.ui.server import SwarmEventBus, swarm_bus

logger = logging.getLogger("StudioDaemon")


class StudioDaemonConfig(BaseModel):
    """Configuration parameters for the autonomous background studio daemon."""
    deliberation: str = Field(
        default="Studio daemon configuration establishing polling cadence, lockfile paths, and storage guardrails.",
        min_length=1,
        description="Epistemic rationale on daemon operational parameters"
    )

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("deliberation must not be empty or whitespace")
        return v

    poll_interval: float = Field(default=5.0, description="SQLite WAL telemetry polling interval in seconds")
    run_once: bool = Field(default=False, description="Execute a single evaluation step then exit")
    enable_daemon: bool = Field(default=True, description="Whether background polling is active")
    steam_profile: str = Field(default="default", description="Active Steam player profile identifier")
    db_path: Path = Field(default=config.DB_PATH, description="Path to SQLite WAL telemetry database")
    cartridges_root: Path = Field(default=PROJECT_ROOT / "cartridges", description="Root cartridges directory")
    active_cartridge_dir: Path = Field(default=PROJECT_ROOT / "cartridges" / "active", description="Active cartridge path")
    lock_file: Path = Field(default=PROJECT_ROOT / "studio.lock", description="Process lockfile for concurrency guard")
    min_disk_space_mb: float = Field(default=100.0, description="Minimum free disk space threshold in MB")


_lock_mutex = threading.Lock()
_genesis_build_lock = threading.Lock()


class StudioDaemon:
    """Autonomous background studio daemon polling WAL and running PEV compilation cycles."""
    deliberation: str = "Background studio daemon coordinating autonomous game evolution without blocking game client."

    def __init__(
        self,
        config_obj: Optional[StudioDaemonConfig] = None,
        event_bus: Optional[SwarmEventBus] = None,
        orchestrator: Optional[Any] = None,
        supervisor: Optional[Any] = None,
    ):
        self.config = config_obj or StudioDaemonConfig()
        self.event_bus = event_bus or swarm_bus
        self.is_running = False
        self.build_counter = 0
        self._thread_lock = threading.Lock()
        self._last_cold_boot_attempt_ts: float = 0.0

        # Initialize or inject StudioOrchestrator
        if orchestrator is not None:
            self.orchestrator = orchestrator
        else:
            try:
                self.orchestrator = StudioOrchestrator(event_bus=self.event_bus)
            except Exception as e:
                logger.warning("StudioOrchestrator deferred/offline: %s", e)
                self.orchestrator = None

        # Initialize or inject BuildSupervisor
        if supervisor is not None:
            self.supervisor = supervisor
        else:
            swe = None
            if self.orchestrator and hasattr(self.orchestrator, "agents"):
                swe = self.orchestrator.agents.get("LeadGameplaySWEAgent")
            self.supervisor = BuildSupervisor(
                cartridges_root=self.config.cartridges_root,
                active_cartridge_dir=self.config.active_cartridge_dir,
                swe_agent=swe,
            )

    def acquire_lock(self) -> bool:
        """Acquires studio.lock with atomic mutual exclusion and stale PID detection."""
        with _lock_mutex:
            with self._thread_lock:
                lock_path = self.config.lock_file
                lock_path.parent.mkdir(parents=True, exist_ok=True)

                for attempt in range(2):
                    try:
                        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                        with os.fdopen(fd, "w", encoding="utf-8") as f:
                            f.write(str(os.getpid()))
                        return True
                    except FileExistsError:
                        # Inspect existing lock
                        try:
                            content = lock_path.read_text(encoding="utf-8").strip()
                            if not content:
                                # Short pause in case another process is currently writing the PID
                                time.sleep(0.02)
                                content = lock_path.read_text(encoding="utf-8").strip()

                            if not content:
                                raise ValueError("Empty lockfile content")

                            pid = int(content)
                            if pid <= 0:
                                raise ValueError(f"Invalid non-positive PID in lockfile: {pid}")

                            if pid == os.getpid():
                                # Lock already owned by current process
                                logger.error("Studio daemon lock already owned by current process PID %d", pid)
                                return False

                            os.kill(pid, 0)
                            logger.error("Studio daemon already running with PID %d", pid)
                            return False
                        except ProcessLookupError:
                            logger.warning("Cleaning up stale studio lockfile from dead PID %d.", pid)
                            try:
                                lock_path.unlink(missing_ok=True)
                            except OSError:
                                pass
                            continue
                        except PermissionError:
                            current_pid = locals().get("pid")
                            if current_pid is not None:
                                logger.error("Studio daemon lock owned by active PID %d (permission denied to signal)", current_pid)
                            else:
                                logger.error("Permission denied accessing studio lockfile %s", lock_path)
                            return False
                        except (ValueError, OSError):
                            logger.warning("Cleaning up corrupted studio lockfile.")
                            try:
                                lock_path.unlink(missing_ok=True)
                            except OSError:
                                pass
                            continue
                    except Exception as exc:
                        logger.error("Failed to write lockfile %s: %s", lock_path, exc)
                        return False
                return False

    def release_lock(self) -> None:
        """Safely removes studio.lock only if owned by the current PID."""
        with _lock_mutex:
            with self._thread_lock:
                try:
                    if self.config.lock_file.exists():
                        try:
                            content = self.config.lock_file.read_text(encoding="utf-8").strip()
                            if content and int(content) == os.getpid():
                                self.config.lock_file.unlink(missing_ok=True)
                        except (ValueError, OSError):
                            pass
                except Exception as exc:
                    logger.warning("Error releasing lockfile: %s", exc)
                finally:
                    self.is_running = False

    def check_disk_space(self) -> bool:
        """Verifies at least min_disk_space_mb is available."""
        check_path = self.config.cartridges_root if self.config.cartridges_root.exists() else self.config.cartridges_root.parent
        try:
            usage = shutil.disk_usage(str(check_path))
            free_mb = usage.free / (1024 * 1024)
            if free_mb < self.config.min_disk_space_mb:
                logger.error(
                    "Disk space exhausted: %.1fMB free (requires %.1fMB)",
                    free_mb,
                    self.config.min_disk_space_mb,
                )
                return False
            return True
        except Exception as exc:
            logger.warning("Disk space check failed (%s); allowing step.", exc)
            return True

    def is_cold_boot_required(self) -> bool:
        """Returns True if cartridges/active/ has no valid Godot 4 or Pygame cartridge."""
        if (
            (self.config.cartridges_root / ".no_cold_boot").exists()
            or os.environ.get("STUDIO_NO_COLD_BOOT") == "1"
            or (
                "PYTEST_CURRENT_TEST" in os.environ
                and self.config.cartridges_root.resolve() == (PROJECT_ROOT / "cartridges").resolve()
            )
        ):
            return False
        active = self.config.active_cartridge_dir
        if not active.exists():
            return True
        is_valid_godot = (active / "project.godot").exists() and (
            (active / "main.gd").exists() or (active / "mechanics.gd").exists()
        )
        is_valid_pygame = (active / "manifest.json").exists() and (active / "mechanics.py").exists()
        return not (is_valid_godot or is_valid_pygame)

    @staticmethod
    def _resolve_maybe_coroutine(obj: Any) -> Any:
        import asyncio
        import inspect
        if inspect.iscoroutine(obj):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(obj)
            return asyncio.run(obj)
        return obj

    def run_genesis_cycle(self) -> bool:
        """Executes cold-boot Genesis build cycle."""
        if not _genesis_build_lock.acquire(blocking=False):
            logger.info("Genesis build is already in progress; skipping duplicate invocation.")
            return False

        try:
            self._last_cold_boot_attempt_ts = time.monotonic()
            logger.info("Cold boot detected: Initiating Genesis Build...")
            self.event_bus.publish_threadsafe({
                "deliberation": "Cold boot detected; initiating autonomous Genesis build cycle.",
                "event_type": "genesis_starting",
                "agent_name": "StudioDaemon",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "payload": {"profile_tag": self.config.steam_profile},
            })

            staging_dir = self.config.cartridges_root / f"candidate_genesis_{uuid.uuid4().hex[:6]}"
            staging_dir.mkdir(parents=True, exist_ok=True)
            report = None

            try:
                # Dispatch Genesis build to StudioOrchestrator swarm plan
                if self.orchestrator is not None:
                    if hasattr(self.orchestrator, "generate_swarm_plan") and hasattr(self.orchestrator, "execute_swarm_plan"):
                        dna = None
                        try:
                            from engine.steam.dna import SteamDNAPersistence, scan_and_extract_local_dna
                            repo = SteamDNAPersistence(self.config.db_path)
                            dna = repo.get_latest_dna(profile_tag=self.config.steam_profile if self.config.steam_profile != "default" else None)
                            if dna is None or getattr(dna, "total_games_analyzed", 0) == 0 or not getattr(dna, "top_games", None):
                                dna = scan_and_extract_local_dna(
                                    db_path=self.config.db_path,
                                    profile_tag=self.config.steam_profile,
                                )
                            if dna is not None:
                                self.event_bus.publish_threadsafe({
                                    "deliberation": dna.deliberation,
                                    "event_type": "AGENT_STREAM",
                                    "agent_name": "SteamAndTelemetryAnalystAgent",
                                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                    "payload": {
                                        "player_id": dna.player_id,
                                        "total_games_analyzed": dna.total_games_analyzed,
                                        "total_playtime_hours": dna.total_playtime_hours,
                                        "top_genres": dna.top_genres,
                                        "top_games": getattr(dna, "top_games", [])[:8],
                                    },
                                })
                        except Exception as dna_exc:
                            logger.warning("Steam DNA extraction warning during Genesis: %s", dna_exc)

                        plan = self._resolve_maybe_coroutine(
                            self.orchestrator.generate_swarm_plan(steam_dna=dna, target_episode=0)
                        )
                        report = self._resolve_maybe_coroutine(
                            self.orchestrator.execute_swarm_plan(plan, staging_dir)
                        )
                    elif hasattr(self.orchestrator, "run_genesis_build"):
                        report = self._resolve_maybe_coroutine(
                            self.orchestrator.run_genesis_build(staging_dir=staging_dir)
                        )
                    else:
                        report = self.supervisor.evaluate_and_heal(
                            candidate_or_spec=staging_dir,
                            episode_number=0,
                            max_retries=3,
                        )
                else:
                    report = self.supervisor.evaluate_and_heal(
                        candidate_or_spec=staging_dir,
                        episode_number=0,
                        max_retries=3,
                    )

                if report and getattr(report, "promoted_to_active", False):
                    logger.info("Genesis build successfully promoted to cartridges/active/.")
                    self.event_bus.publish_threadsafe({
                        "deliberation": "Genesis build verified and promoted to active cartridge.",
                        "event_type": "CARTRIDGE_PROMOTED",
                        "agent_name": "BuildSupervisor",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "payload": {"episode": 0, "status": "PROMOTED"},
                    })
                    return True
                else:
                    rej = getattr(report, "rejection_trace", None) if report else "No build report"
                    logger.error("Genesis build failed quality gates: %s", rej)
                    self.event_bus.publish_threadsafe({
                        "deliberation": f"Genesis build rejected by quality gates: {rej}",
                        "event_type": "ERROR",
                        "agent_name": "BuildSupervisor",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "payload": {"episode": 0, "status": "REJECTED"},
                    })
                    return False
            except Exception as exc:
                logger.critical("Error during genesis cycle: %s", exc, exc_info=True)
                self.event_bus.publish_threadsafe({
                    "deliberation": f"Genesis cycle encountered error: {exc}",
                    "event_type": "ERROR",
                    "agent_name": "StudioDaemon",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "payload": {"error": str(exc)},
                })
                raise
            finally:
                if staging_dir.exists() and (not report or not getattr(report, "promoted_to_active", False)):
                    shutil.rmtree(staging_dir, ignore_errors=True)
        finally:
            _genesis_build_lock.release()

    def poll_and_update(self) -> bool:
        """Polls SQLite WAL for new player gameplay events and triggers update cycle."""
        if not self.config.db_path.exists():
            return False

        try:
            conn = sqlite3.connect(str(self.config.db_path), timeout=5.0)
            conn.execute("PRAGMA query_only = ON;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            cur = conn.cursor()

            cur.execute(
                "SELECT session_id, episode_id, episode_number, outcome, ended_at "
                "FROM telemetry_episodes "
                "WHERE status = 'FINISHED' OR outcome IN ('VICTORY', 'DEFEAT') "
                "ORDER BY episode_number DESC LIMIT 1;"
            )
            row = cur.fetchone()
            conn.close()

            if not row:
                return False

            sess_id, ep_id, ep_num, outcome, ended_at = row
            if ep_num > self.build_counter:
                logger.info("New finished episode %d (%s) detected in WAL. Triggering update cycle...", ep_num, outcome)
                self.build_counter = ep_num
                return self.run_episodic_update_cycle(sess_id, ep_num)
        except Exception as exc:
            logger.debug("Telemetry WAL polling check: %s", exc)
        return False

    def run_episodic_update_cycle(self, session_id: str, episode_number: int) -> bool:
        """Runs hypothesis-driven episodic update cycle."""
        logger.info("Starting episodic update cycle for episode %d...", episode_number)
        staging_dir = self.config.cartridges_root / f"candidate_ep{episode_number}_{uuid.uuid4().hex[:6]}"
        staging_dir.mkdir(parents=True, exist_ok=True)
        report = None

        try:
            active_dir = self.config.active_cartridge_dir
            if active_dir.exists():
                for item in active_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, staging_dir / item.name)
                    elif item.is_dir() and item.name == "assets":
                        shutil.copytree(item, staging_dir / "assets", dirs_exist_ok=True)

            # Update manifest episode number
            manifest_file = staging_dir / "manifest.json"
            m_data = {}
            if manifest_file.exists():
                try:
                    m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                except Exception:
                    m_data = {}
            m_data["episode"] = episode_number
            manifest_file.write_text(json.dumps(m_data, indent=2), encoding="utf-8")

            report = self.supervisor.evaluate_and_heal(
                candidate_or_spec=staging_dir,
                episode_number=episode_number,
                max_retries=3,
            )

            if report.promoted_to_active:
                logger.info("Episodic update %d promoted to active.", episode_number)
                self.event_bus.publish_threadsafe({
                    "deliberation": f"Episodic update {episode_number} verified and promoted to active cartridge.",
                    "event_type": "CARTRIDGE_PROMOTED",
                    "agent_name": "BuildSupervisor",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "payload": {"episode": episode_number, "status": "PROMOTED"},
                })
                return True
            return False
        except Exception as exc:
            logger.error("Episodic update failed: %s", exc, exc_info=True)
            return False
        finally:
            if staging_dir.exists() and (not report or not getattr(report, "promoted_to_active", False)):
                shutil.rmtree(staging_dir, ignore_errors=True)

    def step(self) -> None:
        """Executes a single step of the daemon polling loop."""
        if not self.check_disk_space():
            return

        if self.is_cold_boot_required():
            if self.is_running and self._last_cold_boot_attempt_ts > 0.0:
                if (time.monotonic() - self._last_cold_boot_attempt_ts) < 60.0:
                    return
            try:
                self.run_genesis_cycle()
            except Exception as exc:
                logger.error("Cold boot genesis cycle failed: %s", exc)
            return

        self.poll_and_update()

    def run(self) -> None:
        """Main loop executed in background thread or process."""
        if not self.acquire_lock():
            return

        self.is_running = True
        logger.info("StudioDaemon loop started (poll_interval=%.1fs)", self.config.poll_interval)
        try:
            while self.is_running:
                self.step()
                if self.config.run_once:
                    break
                # Sleep in small slices for responsive shutdown
                sleep_duration = self.config.poll_interval
                slept = 0.0
                while self.is_running and slept < sleep_duration:
                    time.sleep(0.1)
                    slept += 0.1
        finally:
            self.release_lock()
            logger.info("StudioDaemon loop stopped and lock released.")

    def stop(self) -> None:
        """Stops the daemon gracefully."""
        self.is_running = False
