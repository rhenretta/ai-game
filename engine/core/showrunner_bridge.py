"""engine/core/showrunner_bridge.py
Asynchronous Showrunner Runtime Bridge linking live game telemetry events to
the Telemetry Analyst, Episodic Director, SWE Compiler, and Build Supervisor.
Strictly enforces Rationale-First Epistemic Architecture.
"""
from __future__ import annotations
import logging
from pathlib import Path
import threading
from typing import Any, Dict, Optional, Union

import config
from engine.analyst.analyst import TelemetryAnalyst
from engine.director.episodic import EpisodicDirector
from engine.director.models import CurrentEpisodeConfig, EpisodicUpdateSpec
from engine.loader import CartridgeLoader
from engine.scheduler.scheduler import Priority, PriorityScheduler
from engine.steam.models import SteamPlayerDNA
from engine.supervisor.models import SupervisorGateVerdict
from engine.supervisor.supervisor import BuildSupervisor

logger = logging.getLogger("ShowrunnerBridge")


class ShowrunnerRuntimeBridge:
    """Showrunner Runtime Bridge continuously orchestrating background P2 telemetry analysis,
    spec generation, and cartridge compilation without degrading 60 FPS gameplay.
    """
    deliberation: str = "Showrunner Runtime Bridge continuously orchestrating background P2 telemetry analysis, spec generation, and cartridge compilation without degrading 60 FPS gameplay."

    def __init__(
        self,
        engine: Any,
        scheduler: Optional[PriorityScheduler] = None,
        steam_dna: Optional[SteamPlayerDNA] = None,
        db_path: Union[str, Path] = config.DB_PATH,
        cartridges_root: Union[str, Path] = config.PROJECT_ROOT / "cartridges",
        deliberation: str = "Showrunner Runtime Bridge continuously orchestrating background P2 telemetry analysis, spec generation, and cartridge compilation without degrading 60 FPS gameplay.",
    ):
        self.deliberation = deliberation
        self.engine = engine
        self.scheduler = scheduler or PriorityScheduler(max_background_workers=2)
        self.steam_dna = steam_dna
        self.db_path = Path(db_path)
        self.cartridges_root = Path(cartridges_root)
        self.active_cartridge_dir = self.cartridges_root / "active"

        # Subsystems
        self.analyst = TelemetryAnalyst(db_path=self.db_path)
        self.director = EpisodicDirector(specs_dir=config.SPECS_DIR)
        self.supervisor = BuildSupervisor(
            cartridges_root=self.cartridges_root,
            active_cartridge_dir=self.active_cartridge_dir,
        )

        self._lock = threading.RLock()
        self.status: str = "IDLE (P2 Background)"
        self.last_spec: Optional[EpisodicUpdateSpec] = None
        self.adaptation_count: int = 0
        self.is_started: bool = False

    def start(self) -> None:
        """Starts background worker threads for priority task dispatch."""
        with self._lock:
            if self.is_started:
                return
            self.is_started = True
            self.scheduler.start()
            self.status = "ONLINE (Ready)"
            logger.info("ShowrunnerRuntimeBridge online.")

    def shutdown(self) -> None:
        """Shuts down background scheduler workers gracefully."""
        with self._lock:
            if not self.is_started:
                return
            self.status = "OFFLINE"
            self.scheduler.shutdown(wait=True, timeout=2.0)
            self.is_started = False
            logger.info("ShowrunnerRuntimeBridge shutdown cleanly.")

    def on_episode_finished(
        self,
        session_id: str,
        episode_id: str,
        episode_number: int,
        outcome: str,
    ) -> None:
        """Dispatches an asynchronous P2 task to analyze telemetry and author the next episode."""
        with self._lock:
            self.status = f"QUEUED: Episode {episode_number} ({outcome})"

        self.scheduler.submit(
            Priority.P2_SHOWRUNNER,
            f"showrunner_eval_ep_{episode_number}",
            self._execute_showrunner_pipeline,
            session_id,
            episode_id,
            episode_number,
            outcome,
        )

    def _execute_showrunner_pipeline(
        self,
        session_id: str,
        episode_id: str,
        episode_number: int,
        outcome: str,
    ) -> bool:
        """Runs the complete Analyst -> Director -> Supervisor -> Loader cycle at P2."""
        try:
            with self._lock:
                self.status = f"ANALYZING: Episode {episode_number}"

            # Step 1: Telemetry Analyst extracts BehavioralProfile
            profile = self.analyst.analyze_session(session_id=session_id)
            self.scheduler.checkpoint_and_yield(timeout_sec=0.01)

            # Step 2: Episodic Director emits Rationale-First EpisodicUpdateSpec
            with self._lock:
                self.status = f"DIRECTING: Episode {episode_number + 1}"

            current_cfg = CurrentEpisodeConfig(
                deliberation=f"Baseline parameters prior to Episode {episode_number + 1} adaptation",
                episode_number=episode_number,
                episode_id=episode_id,
            )
            spec = self.director.evaluate_and_update(
                profile=profile,
                current_config=current_cfg,
            )
            self.director.write_update_spec(spec)
            self.last_spec = spec
            self.scheduler.checkpoint_and_yield(timeout_sec=0.01)

            # Step 3: Build Supervisor compiles, validates with synthetic playtester, and heals defects
            with self._lock:
                self.status = f"COMPILING & VALIDATING: Episode {episode_number + 1}"

            report = self.supervisor.evaluate_and_heal(spec)
            self.scheduler.checkpoint_and_yield(timeout_sec=0.01)

            # Step 4: Hot-reload active cartridge into engine if promoted
            is_promoted = (
                getattr(report, "is_promoted", False)
                or getattr(report, "promoted_to_active", False)
                or getattr(report, "status", None) == SupervisorGateVerdict.PROMOTED
                or str(getattr(report, "status", "")).upper().endswith("PROMOTED")
            )
            if is_promoted:
                self.engine.load_cartridge(self.active_cartridge_dir)
                with self._lock:
                    self.adaptation_count += 1
                    self.status = f"PROMOTED: Episode {episode_number + 1} Ready"
                logger.info("Showrunner promoted and hot-reloaded Episode %d", episode_number + 1)
                return True
            else:
                verdict = getattr(report, "status", "REJECTED")
                with self._lock:
                    self.status = f"REJECTED: Episode {episode_number + 1} Failed QA"
                logger.warning("Showrunner rejected Episode %d update: %s", episode_number + 1, verdict)
                return False

        except Exception as e:
            with self._lock:
                self.status = f"ERROR: {e}"
            logger.error("Showrunner pipeline encountered error: %s", e, exc_info=True)
            return False

    def get_status(self) -> Dict[str, Any]:
        """Returns structured status dictionary for HUD rendering and epistemic audit."""
        with self._lock:
            return {
                "deliberation": f"Showrunner status report: {self.status} (adaptations={self.adaptation_count})",
                "status": self.status,
                "is_started": self.is_started,
                "adaptation_count": self.adaptation_count,
                "last_spec_id": self.last_spec.spec_id if self.last_spec else None,
                "target_episode": self.last_spec.target_episode_number if self.last_spec else None,
            }
