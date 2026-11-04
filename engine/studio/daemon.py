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
    creative_prompt: Optional[str] = Field(default=None, description="Optional Executive Producer creative seed/prompt for Genesis")
    resume_current: bool = Field(default=False, description="Resume the latest in-progress or failed Genesis candidate from its persisted Game Bible and Requirements")
    resume_candidate_id: Optional[str] = Field(default=None, description="Optional explicit candidate directory name to resume")
    initial_candidate_dir: Optional[Path] = Field(default=None, description="Optional pre-staged candidate_genesis_* directory created by Co-Producer graduation or fork")
    dataset_name: Optional[str] = Field(default=None, description="Optional dataset/run identifier injected into Gemini API request labels for Google AI Studio Logs")


_lock_mutex = threading.Lock()
_genesis_build_lock = threading.Lock()


def rehydrate_resumable_candidate_from_db(
    db_path: Path,
    cartridges_root: Path,
    preferred_candidate_id: Optional[str] = None,
) -> Optional[Path]:
    """Rehydrates a resumable candidate_genesis_* directory from SQLite if missing or incomplete on disk."""
    from engine.studio.models import (
        AcceptanceCriterion,
        GameBibleSpec,
        GameRequirementsSpec,
        GenesisWorldSpec,
        NarrativeBible,
        RequirementTrack,
    )
    from engine.telemetry.db import TelemetryDB

    cartridges_root.mkdir(parents=True, exist_ok=True)
    bible_dict: Optional[Dict[str, Any]] = None
    req_rows: List[Dict[str, Any]] = []
    cand_name: Optional[str] = None

    if preferred_candidate_id:
        clean_pref = preferred_candidate_id.strip()
        if clean_pref.startswith("build_"):
            clean_pref = clean_pref[len("build_"):]
        if clean_pref and clean_pref != "active" and (cartridges_root / clean_pref).is_dir():
            cand_name = clean_pref
            gb_p = cartridges_root / clean_pref / "game_bible.json"
            if gb_p.is_file():
                try:
                    bible_dict = json.loads(gb_p.read_text(encoding="utf-8"))
                except Exception:
                    bible_dict = None

    if not cand_name and db_path and db_path.exists():
        try:
            db = TelemetryDB(db_path)
            bible_dict = db.get_latest_game_bible()
            if bible_dict:
                raw_cid = str(bible_dict.get("cartridge_id") or "").strip()
                if raw_cid.startswith("build_"):
                    raw_cid = raw_cid[len("build_"):]
                if raw_cid and raw_cid != "active":
                    cand_name = raw_cid
            if cand_name:
                req_rows = db.get_studio_requirements(build_id=f"build_{cand_name}")
            if not req_rows:
                req_rows = db.get_studio_requirements()
                if req_rows and not cand_name:
                    raw_bid = str(req_rows[-1].get("build_id") or "").strip()
                    if raw_bid.startswith("build_"):
                        cand_name = raw_bid[len("build_"):]
                    req_rows = [r for r in req_rows if r.get("build_id") == raw_bid]
        except Exception as exc:
            logger.warning("Failed reading telemetry DB during candidate rehydration: %s", exc)

    # Prefer a newer pre-staged Co-Producer commissioned cartridge (`co_producer_brief.json`) on disk
    # when no explicit `preferred_candidate_id` was provided and its Turn-0 bible has not been written yet.
    if not preferred_candidate_id:
        commissioned_cands = sorted(
            [
                d
                for d in cartridges_root.glob("candidate_genesis_*")
                if d.is_dir() and (d / "co_producer_brief.json").is_file() and not (d / "game_bible.json").is_file()
            ],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if commissioned_cands:
            newest_comm = commissioned_cands[0]
            existing_mtime = (
                (cartridges_root / cand_name).stat().st_mtime
                if cand_name and (cartridges_root / cand_name).is_dir()
                else 0.0
            )
            if newest_comm.stat().st_mtime >= existing_mtime:
                cand_name = newest_comm.name
                bible_dict = None
                req_rows = []

    if not cand_name:
        existing_cands = sorted(
            [
                d
                for d in cartridges_root.glob("candidate_genesis_*")
                if d.is_dir()
            ],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if existing_cands:
            cand_name = existing_cands[0].name

    if not cand_name:
        return None

    staging_dir = cartridges_root / cand_name
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rehydrate game_bible.json, world_spec.json, manifest.json, narrative_bible.json, narrative.json
    world_spec_obj: Optional[GenesisWorldSpec] = None
    if bible_dict:
        gb_path = staging_dir / "game_bible.json"
        if not gb_path.is_file():
            try:
                gb_obj = GameBibleSpec.model_validate(bible_dict)
                gb_path.write_text(gb_obj.model_dump_json(indent=2), encoding="utf-8")
            except Exception:
                gb_path.write_text(json.dumps(bible_dict, indent=2), encoding="utf-8")

        ws_raw = bible_dict.get("world_spec")
        ws_path = staging_dir / "world_spec.json"
        if isinstance(ws_raw, dict) and ws_raw:
            try:
                world_spec_obj = GenesisWorldSpec.model_validate(ws_raw)
                if not ws_path.is_file():
                    ws_path.write_text(world_spec_obj.model_dump_json(indent=2), encoding="utf-8")
                manifest_path = staging_dir / "manifest.json"
                if not manifest_path.is_file():
                    manifest_dict = {
                        "deliberation": world_spec_obj.deliberation,
                        "game_title": world_spec_obj.game_title,
                        "genre": world_spec_obj.core_genre,
                        "engine_type": "godot4",
                        "entrypoint": "res://main.tscn",
                        "version": "0.1.0",
                        "palette_mood": world_spec_obj.palette_mood,
                        "narrative_premise": getattr(world_spec_obj, "narrative_premise", None) or (bible_dict or {}).get("creative_vision_summary", ""),
                    }
                    manifest_path.write_text(
                        json.dumps(manifest_dict, indent=2),
                        encoding="utf-8",
                    )
            except Exception as ws_exc:
                logger.warning("Failed rehydrating world_spec.json: %s", ws_exc)

        nb_raw = bible_dict.get("narrative_bible")
        nb_path = staging_dir / "narrative_bible.json"
        nar_path = staging_dir / "narrative.json"
        if isinstance(nb_raw, dict) and nb_raw and (not nb_path.is_file() or not nar_path.is_file()):
            try:
                nb_obj = NarrativeBible.model_validate(nb_raw)
                if getattr(nb_obj, "dialogue_barks", None):
                    nb_obj.dialogue_barks = list(nb_obj.dialogue_barks)[:3]
                    for idx, b in enumerate(nb_obj.dialogue_barks):
                        wav_name = Path(b.voice_clip).name if getattr(b, "voice_clip", None) else f"{b.bark_id or f'bark_{idx}'}.wav"
                        if not wav_name.endswith(".wav"):
                            wav_name = f"{wav_name}.wav"
                        b.voice_clip = f"assets/audio/{wav_name}"
                if getattr(nb_obj, "characters", None):
                    for idx, c in enumerate(nb_obj.characters):
                        p_name = Path(c.portrait_asset).name if getattr(c, "portrait_asset", None) else f"{c.character_id or f'char_{idx}'}_portrait.png"
                        if not p_name.endswith(".png"):
                            p_name = f"{p_name}.png"
                        c.portrait_asset = f"assets/sprites/{p_name}"
                nb_path.write_text(nb_obj.model_dump_json(indent=2), encoding="utf-8")
                nar_path.write_text(
                    json.dumps(nb_obj.to_cartridge_narrative_dict(), indent=2),
                    encoding="utf-8",
                )
            except Exception as nb_exc:
                logger.warning("Failed rehydrating narrative_bible.json: %s", nb_exc)

        # Ensure 16:9 mockup PNGs exist on disk
        mockups_raw = bible_dict.get("visual_mockups")
        if isinstance(mockups_raw, list) and mockups_raw:
            mockups_dir = staging_dir / "assets" / "mockups"
            mockups_dir.mkdir(parents=True, exist_ok=True)
            for idx, m in enumerate(mockups_raw):
                if not isinstance(m, dict):
                    continue
                rel_p = str(m.get("image_path") or m.get("file_path") or f"assets/mockups/mockup_{idx + 1}.png")
                target_png = staging_dir / rel_p if not Path(rel_p).is_absolute() else mockups_dir / Path(rel_p).name
                if not target_png.is_file():
                    try:
                        from engine.studio.writers_room import CreativeWritersRoom
                        target_png.parent.mkdir(parents=True, exist_ok=True)
                        CreativeWritersRoom._render_procedural_16x9_png(
                            target_png,
                            seed_text=f"{m.get('title', 'Mockup')}:{m.get('scene_description', '')}",
                            palette_hex=["#0b1d26", "#1b4d3e", "#2dd4bf", "#f59e0b"],
                        )
                    except Exception:
                        pass

    if world_spec_obj is None and (staging_dir / "world_spec.json").is_file():
        try:
            world_spec_obj = GenesisWorldSpec.model_validate_json(
                (staging_dir / "world_spec.json").read_text(encoding="utf-8")
            )
        except Exception:
            world_spec_obj = None

    # 2. Rehydrate or reset requirements_spec.json
    req_path = staging_dir / "requirements_spec.json"
    req_spec_obj: Optional[GameRequirementsSpec] = None
    if req_path.is_file():
        try:
            req_spec_obj = GameRequirementsSpec.model_validate_json(req_path.read_text(encoding="utf-8"))
        except Exception:
            req_spec_obj = None

    if req_spec_obj is None and req_rows:
        try:
            g_title = (
                getattr(world_spec_obj, "game_title", None)
                or (bible_dict or {}).get("game_title")
                or "Resumed Game Project"
            )
            g_genre = (
                getattr(world_spec_obj, "core_genre", None)
                or (bible_dict or {}).get("core_genre")
                or "action_adventure"
            )
            req_items: List[RequirementTrack] = []
            for r in req_rows:
                raw_ac = r.get("acceptance_criteria", [])
                ac_objs: List[AcceptanceCriterion] = []
                for idx_ac, ac in enumerate(raw_ac if isinstance(raw_ac, list) else []):
                    if isinstance(ac, dict):
                        ac_objs.append(
                            AcceptanceCriterion(
                                deliberation=str(ac.get("deliberation") or f"Verify {ac.get('id', idx_ac + 1)}"),
                                id=str(ac.get("id") or f"{r.get('requirement_id', 'R1')}-AC{idx_ac + 1}"),
                                description=str(ac.get("description") or "Verify criterion"),
                                verification_method=str(ac.get("verification_method") or "ast_inspection"),
                                passed=bool(ac.get("passed", False)),
                            )
                        )
                st_val = str(r.get("status") or "pending").lower()
                if st_val in ("rejected", "in_progress"):
                    st_val = "pending"
                delib_str = str(r.get("deliberation") or f"Requirement track {r.get('requirement_id')}")
                title_str = str(r.get("title") or "Gameplay Track")
                desc_str = str(r.get("description") or delib_str)
                owner_str = str(r.get("specialist_owner") or "LeadGameplaySWEAgent")
                target_mods = r.get("target_modules")
                if not isinstance(target_mods, list) or not target_mods:
                    target_mods = ["main.gd", "mechanics.gd", "entities.gd", "scenes.gd", "ui_hud.gd"]
                req_items.append(
                    RequirementTrack(
                        deliberation=delib_str,
                        id=str(r.get("requirement_id") or f"R{len(req_items) + 1}"),
                        title=title_str,
                        description=desc_str,
                        responsible_agents=[owner_str],
                        target_modules=target_mods,
                        status=st_val if st_val in ("pending", "in_progress", "completed", "satisfied", "rejected") else "pending",
                        acceptance_criteria=ac_objs,
                    )
                )
            if req_items:
                req_spec_obj = GameRequirementsSpec(
                    deliberation=f"Rehydrated requirements specification for '{g_title}' ({cand_name}) from SQLite telemetry.",
                    spec_id=f"req_{cand_name}",
                    game_title=str(g_title),
                    genre_paradigm=str(g_genre),
                    high_concept_summary=str(
                        getattr(world_spec_obj, "narrative_premise", None)
                        or (bible_dict or {}).get("creative_vision_summary")
                        or f"Playable world of {g_title}"
                    ),
                    requirements=req_items,
                )
        except Exception as req_exc:
            logger.warning("Failed reconstructing GameRequirementsSpec from DB rows: %s", req_exc)

    if req_spec_obj is not None:
        for trk in getattr(req_spec_obj, "requirements", []):
            if getattr(trk, "status", "") in ("rejected", "in_progress"):
                trk.status = "pending"
        req_path.write_text(req_spec_obj.model_dump_json(indent=2), encoding="utf-8")

        # Sync reset statuses back to SQLite
        if db_path and db_path.exists():
            try:
                db = TelemetryDB(db_path)
                build_id = f"build_{cand_name}"
                for trk in req_spec_obj.requirements:
                    db.update_studio_requirement_status(
                        build_id=build_id,
                        requirement_id=trk.id,
                        status=trk.status,
                    )
            except Exception as sync_exc:
                logger.warning("Failed syncing reset requirement status to SQLite: %s", sync_exc)

    backlog_path = staging_dir / "backlog.json"
    if backlog_path.is_file():
        try:
            from engine.studio.production_board import ProductionBoard
            ProductionBoard(path=backlog_path)
        except Exception as pb_exc:
            logger.warning("Failed sanitizing backlog.json during rehydration: %s", pb_exc)

    mem_path = staging_dir / "game_memory.json"
    if not mem_path.is_file() and db_path and db_path.exists():
        try:
            from engine.studio.game_memory import GameMemoryStore

            mem_store = GameMemoryStore(staging_dir, db=TelemetryDB(db_path))
            mem_ledger = mem_store.load_ledger()
            if mem_ledger.entries:
                mem_store.save_ledger(mem_ledger)
        except Exception as mem_exc:
            logger.warning("Failed rehydrating game_memory.json from SQLite: %s", mem_exc)

    return staging_dir


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
            resumed_dir: Optional[Path] = None
            if self.config.resume_current:
                resumed_dir = rehydrate_resumable_candidate_from_db(
                    db_path=self.config.db_path,
                    cartridges_root=self.config.cartridges_root,
                    preferred_candidate_id=self.config.resume_candidate_id,
                )

            from engine.studio.models import set_active_gemini_dataset_context

            if resumed_dir is not None:
                staging_dir = resumed_dir
                staging_mode = "resume"
            elif self.config.initial_candidate_dir is not None:
                staging_dir = Path(self.config.initial_candidate_dir)
                staging_dir.mkdir(parents=True, exist_ok=True)
                staging_mode = "commissioned"
            else:
                staging_dir = self.config.cartridges_root / f"candidate_genesis_{uuid.uuid4().hex[:6]}"
                staging_dir.mkdir(parents=True, exist_ok=True)
                staging_mode = "cold_boot"

            active_bid = f"build_{staging_dir.name}"
            set_active_gemini_dataset_context(
                dataset_name=self.config.dataset_name or staging_dir.name,
                candidate_dir=staging_dir,
            )
            if self.event_bus is not None and hasattr(self.event_bus, "set_active_build_id"):
                try:
                    self.event_bus.set_active_build_id(active_bid)
                except Exception:
                    pass

            if staging_mode == "resume":
                logger.info("Resuming existing Genesis build in %s...", staging_dir)
                self.event_bus.publish_threadsafe({
                    "deliberation": f"Resuming Genesis game development in '{staging_dir.name}' from persisted Game Bible and requirement tracks.",
                    "event_type": "genesis_starting",
                    "agent_name": "StudioDaemon",
                    "build_id": active_bid,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "payload": {
                        "profile_tag": self.config.steam_profile,
                        "resumed_candidate": staging_dir.name,
                        "candidate_id": staging_dir.name,
                        "cartridge_id": staging_dir.name,
                        "resume_current": True,
                    },
                })
            elif staging_mode == "commissioned":
                logger.info("Starting commissioned Genesis build in pre-staged directory %s...", staging_dir)
                self.event_bus.publish_threadsafe({
                    "deliberation": f"Initiating commissioned Genesis build cycle in pre-staged cartridge '{staging_dir.name}'.",
                    "event_type": "genesis_starting",
                    "agent_name": "StudioDaemon",
                    "build_id": active_bid,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "payload": {
                        "profile_tag": self.config.steam_profile,
                        "candidate_id": staging_dir.name,
                        "cartridge_id": staging_dir.name,
                    },
                })
            else:
                logger.info("Cold boot detected: Initiating Genesis Build...")
                self.event_bus.publish_threadsafe({
                    "deliberation": "Cold boot detected; initiating autonomous Genesis build cycle.",
                    "event_type": "genesis_starting",
                    "agent_name": "StudioDaemon",
                    "build_id": active_bid,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "payload": {
                        "profile_tag": self.config.steam_profile,
                        "candidate_id": staging_dir.name,
                        "cartridge_id": staging_dir.name,
                    },
                })
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

                        has_persisted_bible = (staging_dir / "game_bible.json").is_file()
                        effective_prompt = self.config.creative_prompt
                        if not effective_prompt and not has_persisted_bible and (staging_dir / "co_producer_brief.json").is_file():
                            try:
                                cp_env = json.loads((staging_dir / "co_producer_brief.json").read_text(encoding="utf-8"))
                                if isinstance(cp_env, dict):
                                    cb_inner = cp_env.get("concept_brief") if isinstance(cp_env.get("concept_brief"), dict) else {}
                                    effective_prompt = str(
                                        cp_env.get("steering_or_seed_brief")
                                        or cb_inner.get("elevator_pitch")
                                        or ""
                                    ).strip() or None
                            except Exception:
                                effective_prompt = None

                        if effective_prompt:
                            self.orchestrator._creative_prompt = effective_prompt
                            if (
                                self.config.resume_current
                                and has_persisted_bible
                                and self.config.creative_prompt
                                and hasattr(self.orchestrator, "enqueue_user_steering")
                            ):
                                try:
                                    self.orchestrator.enqueue_user_steering(self.config.creative_prompt)
                                except Exception:
                                    pass
                        plan = None
                        if self.config.resume_current and (staging_dir / "world_spec.json").is_file():
                            try:
                                from engine.studio.models import GenesisWorldSpec, StudioSwarmPlan
                                ws_resumed = GenesisWorldSpec.model_validate_json(
                                    (staging_dir / "world_spec.json").read_text(encoding="utf-8")
                                )
                                plan = StudioSwarmPlan(
                                    deliberation=(
                                        f"Resuming specialist agent swarm DAG for '{ws_resumed.game_title}' "
                                        f"({ws_resumed.core_genre}) from persisted staging artifacts in {staging_dir.name}."
                                    ),
                                    game_title=ws_resumed.game_title,
                                    genre_verdict=ws_resumed.core_genre,
                                    target_episode=0,
                                    specialist_agent_roles=[
                                        "Director",
                                        "Writer",
                                        "Voice",
                                        "Visual",
                                        "SWE",
                                        "TelemetryArchitect",
                                        "Supervisor",
                                    ],
                                    dag_edges=[
                                        ["Director", "Writer"],
                                        ["Writer", "Voice"],
                                        ["Writer", "Visual"],
                                        ["Voice", "SWE"],
                                        ["Visual", "SWE"],
                                        ["SWE", "TelemetryArchitect"],
                                        ["TelemetryArchitect", "Supervisor"],
                                    ],
                                    task_manifest=json.dumps({
                                        "Director": "Resume world_spec.json and requirements_spec.json",
                                        "Writer": "Resume narrative_bible.json and narrative.json",
                                        "Voice": "Synthesize or resume character bark WAVs",
                                        "Visual": "Synthesize or resume character portraits and sprites",
                                        "SWE": "Implement remaining requirement tracks in Godot 4",
                                        "TelemetryArchitect": "Instrument telemetry probes",
                                        "Supervisor": "Run 5-gate verification and multi-agent sign-off",
                                    }),
                                )
                            except Exception as resume_plan_exc:
                                logger.warning("Could not build resumed StudioSwarmPlan from world_spec.json: %s", resume_plan_exc)
                                plan = None

                        if plan is None:
                            plan_sig_kwargs: Dict[str, Any] = {"steam_dna": dna, "target_episode": 0}
                            import inspect as _inspect
                            try:
                                if "creative_prompt" in _inspect.signature(self.orchestrator.generate_swarm_plan).parameters:
                                    plan_sig_kwargs["creative_prompt"] = self.config.creative_prompt
                            except Exception:
                                pass
                            plan = self._resolve_maybe_coroutine(
                                self.orchestrator.generate_swarm_plan(**plan_sig_kwargs)
                            )
                        self.orchestrator._resume_current_requested = bool(self.config.resume_current)
                        try:
                            report = self._resolve_maybe_coroutine(
                                self.orchestrator.execute_swarm_plan(plan, staging_dir)
                            )
                        finally:
                            self.orchestrator._resume_current_requested = False
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
                    from engine.studio.models import is_candidate_build_aborted
                    if is_candidate_build_aborted(staging_dir) or "Cancelled by user" in str(rej):
                        logger.info("Genesis build in %s cancelled by new game request.", staging_dir.name)
                        return False
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
                from engine.studio.models import is_candidate_build_aborted
                if is_candidate_build_aborted(staging_dir) or "cancelled" in str(exc).lower() or "preempted" in str(exc).lower():
                    logger.info("Previous Genesis build in %s preempted cleanly: %s", staging_dir.name, exc)
                    return False
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
                if staging_dir.exists() and not any(staging_dir.iterdir()):
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
        from engine.studio.models import set_active_gemini_dataset_context
        set_active_gemini_dataset_context(
            dataset_name=self.config.dataset_name or staging_dir.name,
            candidate_dir=staging_dir,
        )
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
