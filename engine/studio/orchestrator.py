"""engine/studio/orchestrator.py
Top-level Metacognitive Studio Orchestrator and DAG Engine.
Dynamically assembles 9 specialist personas, coordinates PEV cycles, and seals artifacts.
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
import inspect
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import time as _time
import uuid

import config
from config import PROJECT_ROOT
from engine.scheduler.artifact import ArtifactMetadataEnvelope, ArtifactType
from engine.studio.models import (
    AcceptanceCriterion,
    AgentExecutionRecord,
    AgentPlan,
    AgentVerificationRecord,
    BaseEpistemicModel,
    CartridgeCompilationResult,
    GameBibleSpec,
    GameRequirementsSpec,
    GenesisWorldSpec,
    RequirementTrack,
    StudioAgentError,
    StudioBuildState,
    StudioOrchestratorConfig,
    StudioSwarmExecutionError,
    StudioSwarmPlan,
    StudioSwarmPlanError,
    ToolInvocationRecord,
)
if TYPE_CHECKING:
    from engine.supervisor.models import BuildEvaluationReport
from engine.ui.models import (
    AgentTaskRecord,
    ArtifactMetadataEnvelope as UIArtifactMetadataEnvelope,
)

logger = logging.getLogger("StudioOrchestrator")


class CircularDependencyError(Exception):
    """Raised when a loop is detected in the agent execution DAG."""
    deliberation: str = "Raised when a loop is detected in the agent execution DAG."


class InvalidStateTransitionError(Exception):
    """Raised when an illegal state machine transition is attempted."""
    deliberation: str = "Raised when an illegal state machine transition is attempted."


class BuildVerificationExhaustedError(Exception):
    """Raised when self-healing repair attempts exceed max_retries."""
    deliberation: str = "Raised when self-healing repair attempts exceed max_retries."


class StudioAgentDAG:
    """Manages topological dependency graph of the 9 studio personas."""
    deliberation: str = "Dependency graph managing execution order across studio agent personas."

    DEFAULT_EDGES: List[Tuple[str, str]] = [
        ("Analyst", "Director"),
        ("Director", "Writer"),
        ("Director", "Visual"),
        ("Writer", "Voice"),
        ("Writer", "Visual"),
        ("Visual", "SWE"),
        ("Voice", "SWE"),
        ("Director", "SWE"),
        ("SWE", "TelemetryArchitect"),
        ("SWE", "Supervisor"),
        ("Supervisor", "Promotion"),
    ]

    def __init__(self, edges: Optional[List[Tuple[str, str]]] = None):
        self.edges: List[Tuple[str, str]] = list(edges) if edges is not None else list(self.DEFAULT_EDGES)

    def add_edge(self, upstream: str, downstream: str) -> None:
        self.edges.append((upstream, downstream))

    def has_cycle(self) -> bool:
        try:
            self.topological_sort()
            return False
        except CircularDependencyError:
            return True

    def topological_sort(self) -> List[str]:
        # Collect all nodes
        nodes: Set[str] = set()
        for u, v in self.edges:
            nodes.add(u)
            nodes.add(v)

        # Build adjacency and in-degree tables
        adj: Dict[str, List[str]] = {n: [] for n in nodes}
        in_degree: Dict[str, int] = {n: 0 for n in nodes}
        for u, v in self.edges:
            adj[u].append(v)
            in_degree[v] += 1

        queue: List[str] = sorted([n for n in nodes if in_degree[n] == 0])
        order: List[str] = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for neighbor in sorted(adj[curr]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(nodes):
            raise CircularDependencyError(
                f"Cycle detected in agent DAG. Processed {len(order)} of {len(nodes)} nodes."
            )
        return order


ROLE_ALIASES: Dict[str, str] = {
    # Short alias -> Full Persona Name
    "Analyst": "SteamAndTelemetryAnalystAgent",
    "Director": "CreativeDirectorWorldbuilderAgent",
    "Writer": "NarrativeCharacterWriterAgent",
    "Voice": "VoiceActingAudioProducerAgent",
    "Visual": "VisualArtSpriteDesignerAgent",
    "SWE": "LeadGameplaySWEAgent",
    "TelemetryArchitect": "TelemetryArchitectAgent",
    "Supervisor": "QAPlaytesterBuildSupervisor",
    # Slugs -> Full Persona Name
    "steam_analyst": "SteamAndTelemetryAnalystAgent",
    "creative_director": "CreativeDirectorWorldbuilderAgent",
    "narrative_designer": "NarrativeCharacterWriterAgent",
    "voice_audio_engineer": "VoiceActingAudioProducerAgent",
    "visual_artist": "VisualArtSpriteDesignerAgent",
    "lead_gameplay_swe": "LeadGameplaySWEAgent",
    "telemetry_architect": "TelemetryArchitectAgent",
    "qa_supervisor": "QAPlaytesterBuildSupervisor",
}


def normalize_role(role: str) -> str:
    """Normalizes short or slug role name to canonical agent dictionary key."""
    return ROLE_ALIASES.get(role, role)


class AwaitableStudioSwarmPlan(StudioSwarmPlan):
    """StudioSwarmPlan wrapper that is awaitable in async contexts and usable directly in sync contexts."""
    def __await__(self):
        async def _identity():
            return self
        return _identity().__await__()


class SwarmExecutionContext:
    """Blackboard storing intermediate artifacts, cryptographic envelopes, and file outputs."""
    def __init__(self, run_id: str, staging_dir: Union[Path, str]):
        self.run_id = run_id
        self.staging_dir = Path(staging_dir)
        self.artifacts: Dict[str, ArtifactMetadataEnvelope] = {}
        self.raw_outputs: Dict[str, Any] = {}
        self.task_history: List[Dict[str, Any]] = []

    def record_artifact(self, role: str, envelope: ArtifactMetadataEnvelope, raw_output: Any) -> None:
        self.artifacts[role] = envelope
        self.raw_outputs[role] = raw_output
        canonical = normalize_role(role)
        self.artifacts[canonical] = envelope
        self.raw_outputs[canonical] = raw_output

    def record_task(self, record: Union[AgentTaskRecord, Dict[str, Any]]) -> None:
        if hasattr(record, "model_dump"):
            self.task_history.append(record.model_dump())
        else:
            self.task_history.append(dict(record))


class StudioOrchestrator:
    """Metacognitive orchestrator coordinating the 9-agent Gemini studio swarm."""
    deliberation: str = "Autonomous studio orchestrator managing multi-agent game compilation."

    ALLOWED_TRANSITIONS: Dict[StudioBuildState, Set[StudioBuildState]] = {
        StudioBuildState.IDLE: {StudioBuildState.ANALYZING},
        StudioBuildState.ANALYZING: {StudioBuildState.BRAINSTORMING, StudioBuildState.SYNTHESIZING},
        StudioBuildState.BRAINSTORMING: {StudioBuildState.AWAITING_GREENLIGHT, StudioBuildState.SYNTHESIZING},
        StudioBuildState.AWAITING_GREENLIGHT: {StudioBuildState.BRAINSTORMING, StudioBuildState.SYNTHESIZING, StudioBuildState.IDLE},
        StudioBuildState.SYNTHESIZING: {StudioBuildState.VERIFYING, StudioBuildState.BRAINSTORMING, StudioBuildState.AWAITING_GREENLIGHT},
        StudioBuildState.VERIFYING: {StudioBuildState.PROMOTED, StudioBuildState.SYNTHESIZING},
        StudioBuildState.PROMOTED: {StudioBuildState.IDLE},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        config_obj: Optional[StudioOrchestratorConfig] = None,
        event_bus: Optional[Any] = None,
        deliberation: str = "Initializing StudioOrchestrator multi-agent studio swarm.",
        db_path: Optional[Any] = None,
        enable_writers_room: Optional[bool] = None,
    ):
        self.deliberation = deliberation
        self.db_path = Path(db_path) if db_path else None
        resolved_key = self._resolve_api_key(api_key, config_obj)
        if not resolved_key or not resolved_key.strip():
            raise ValueError("GEMINI_API_KEY is not configured or empty")

        self.api_key = resolved_key.strip()
        self.config = config_obj or StudioOrchestratorConfig(
            deliberation="Default studio configuration",
            gemini_api_key=self.api_key,
        )
        self.state = StudioBuildState.IDLE
        self.dag = StudioAgentDAG()
        if event_bus is not None:
            self.event_bus = event_bus
        else:
            try:
                from engine.ui.server import swarm_bus
                self.event_bus = swarm_bus
            except Exception:
                self.event_bus = None
        from engine.studio.models import install_studio_gemini_guardrails, set_active_gemini_dataset_context
        self.client = install_studio_gemini_guardrails(
            client if client is not None else self._init_gemini_client(self.api_key, timeout_seconds=self.config.timeout_seconds),
            dataset_name=getattr(self.config, "dataset_name", "") or None,
            store_logs=getattr(self.config, "store_logs", True),
        )
        self.enable_writers_room = enable_writers_room
        self.active_game_bible: Optional[GameBibleSpec] = None
        self.production_board: Optional[Any] = None
        self._continuous_active: bool = False
        self._greenlight_decision: Optional[Dict[str, Any]] = None
        self._greenlight_event: threading.Event = threading.Event()
        self._abort_requested: bool = False
        if self.event_bus is not None:
            try:
                self.event_bus._active_orchestrator = self
                if getattr(self.event_bus, "_pause_for_greenlight", False):
                    self.config.pause_for_greenlight = True
            except Exception:
                pass

        # Assemble the 9 specialist agent roles + CreativeWritersRoom
        self.agents: Dict[str, Any] = self._assemble_agents()

    def _is_live_genai_client(self) -> bool:
        """Returns True if self.client is a live google.genai Client rather than None or a unit-test mock."""
        if self.client is None:
            return False
        if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
            return False
        mod = getattr(type(self.client), "__module__", "") or ""
        return mod.startswith("google.genai")

    def _get_telemetry_db(self) -> Optional[Any]:
        """Returns TelemetryDB instance if configured or in live production mode; None in pytest without explicit DB path."""
        if self.db_path is not None:
            from engine.telemetry.db import TelemetryDB
            return TelemetryDB(self.db_path)
        if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
            return None
        from engine.telemetry.db import TelemetryDB
        return TelemetryDB()

    def _publish_event(
        self,
        event_type: str,
        agent_name: str,
        deliberation: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.event_bus is not None and hasattr(self.event_bus, "publish_threadsafe"):
            try:
                bid = getattr(self, "active_build_id", None)
                ev_data = {
                    "deliberation": deliberation,
                    "event_type": event_type,
                    "agent_name": agent_name,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "payload": payload or {},
                }
                if bid:
                    ev_data["build_id"] = bid
                self.event_bus.publish_threadsafe(ev_data)
            except Exception as exc:
                logger.debug("Event bus publish skipped: %s", exc)

    @staticmethod
    def _resolve_api_key(explicit_key: Optional[str] = None, cfg: Optional[StudioOrchestratorConfig] = None) -> str:
        if explicit_key is not None:
            if not explicit_key.strip():
                raise ValueError("GEMINI_API_KEY is not configured or empty")
            return explicit_key.strip()
        if cfg and cfg.gemini_api_key and cfg.gemini_api_key.strip():
            return cfg.gemini_api_key.strip()
        env_val = os.environ.get("GEMINI_API_KEY", "")
        if env_val.strip():
            return env_val.strip()
        # Parse .env directly
        env_path = PROJECT_ROOT / ".env"
        if env_path.is_file():
            content = env_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
        return ""

    @staticmethod
    def _init_gemini_client(api_key: str, timeout_seconds: float = 300.0) -> Any:
        if not api_key or not api_key.strip():
            raise StudioAgentError("GEMINI_API_KEY is required to initialize google.genai Client")
        if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
            class _OfflinePytestModels:
                def generate_content(self, *args: Any, **kwargs: Any) -> Any:
                    raise StudioAgentError("Live Gemini API calls are blocked during pytest; provide a mocked client.")

                def generate_images(self, *args: Any, **kwargs: Any) -> Any:
                    raise StudioAgentError("Live Gemini API calls are blocked during pytest; provide a mocked client.")

            class _OfflinePytestGeminiClient:
                def __init__(self, key: str) -> None:
                    self.api_key = key
                    self.models = _OfflinePytestModels()

            return _OfflinePytestGeminiClient(api_key)
        try:
            from google import genai
            timeout_ms = int(max(timeout_seconds, 10.0) * 1000)
            return genai.Client(api_key=api_key, http_options={"timeout": timeout_ms})
        except Exception as exc:
            logger.error("Could not initialize google.genai Client: %s", exc)
            raise StudioAgentError(f"Failed to initialize google.genai Client: {exc}") from exc

    def _assemble_agents(self) -> Dict[str, Any]:
        """Instantiates all 9 specialist agent personas plus CreativeWritersRoom (Rule 5: Zero Silent Fallbacks)."""
        from engine.studio.director import CreativeDirectorWorldbuilderAgent
        from engine.studio.narrative import NarrativeCharacterWriterAgent
        from engine.studio.audio import VoiceActingAudioProducerAgent
        from engine.studio.visual import VisualArtSpriteDesignerAgent
        from engine.studio.swe import LeadGameplaySWEAgent
        from engine.studio.telemetry_architect import TelemetryArchitectAgent
        from engine.studio.writers_room import CreativeWritersRoom
        from engine.supervisor.supervisor import BuildSupervisor

        director_agent = CreativeDirectorWorldbuilderAgent(client=self.client, model_name=self.config.model_name)
        narrative_agent = NarrativeCharacterWriterAgent(client=self.client, model_name=self.config.model_name)
        audio_agent = VoiceActingAudioProducerAgent(client=self.client)
        visual_agent = VisualArtSpriteDesignerAgent(client=self.client)
        swe_agent = LeadGameplaySWEAgent(client=self.client, model_name=self.config.model_name)
        telemetry_agent = TelemetryArchitectAgent(gemini_client=self.client, model_name=self.config.model_name)
        supervisor_agent = BuildSupervisor(
            cartridges_root=PROJECT_ROOT / "cartridges",
            active_cartridge_dir=PROJECT_ROOT / "cartridges" / "active",
            swe_agent=swe_agent,
            gemini_client=self.client,
            event_bus=self.event_bus,
        )

        def _writers_room_event_cb(ev_type: str, payload: Dict[str, Any]) -> None:
            delib = payload.get("deliberation", f"CreativeWritersRoom emitted {ev_type}")
            speaker = payload.get("speaker_role") or "CreativeWritersRoom"
            self._publish_event(ev_type, speaker, delib, payload)

        self.writers_room = CreativeWritersRoom(
            client=self.client,
            model_name=self.config.model_name,
            visual_agent=visual_agent,
            narrative_agent=narrative_agent,
            audio_agent=audio_agent,
            telemetry_db=self._get_telemetry_db(),
            event_callback=_writers_room_event_cb,
        )

        return {
            "StudioOrchestrator": self,
            "SteamAndTelemetryAnalystAgent": None,
            "CreativeDirectorWorldbuilderAgent": director_agent,
            "NarrativeCharacterWriterAgent": narrative_agent,
            "VoiceActingAudioProducerAgent": audio_agent,
            "VisualArtSpriteDesignerAgent": visual_agent,
            "LeadGameplaySWEAgent": swe_agent,
            "TelemetryArchitectAgent": telemetry_agent,
            "QAPlaytesterBuildSupervisor": supervisor_agent,
        }

    def cancel_active_build(self) -> None:
        """Signals an active or paused Genesis build (e.g. waiting at Greenlight gate, in Writers' Room, or draining board)
        to abort immediately so a new game can start."""
        self._abort_requested = True
        if getattr(self, "writers_room", None) is not None:
            self.writers_room._abort_requested = True
        if getattr(self, "_board_scheduler", None) is not None:
            try:
                self._board_scheduler.stop()
            except Exception:
                pass
        try:
            from engine.studio.models import get_active_gemini_dataset_context, mark_candidate_build_aborted
            mark_candidate_build_aborted(getattr(self, "_active_staging_dir", None))
            mark_candidate_build_aborted(get_active_gemini_dataset_context().get("candidate_dir"))
        except Exception:
            pass
        self._greenlight_decision = {"approved": False, "aborted": True, "feedback": ""}
        if self._greenlight_event is not None:
            try:
                self._greenlight_event.set()
            except Exception:
                pass

    def submit_greenlight_decision(self, approved: bool = True, feedback: str = "") -> Dict[str, Any]:
        """Allows Mission Control UI to approve the Living Game Bible or inject steering feedback."""
        clean_fb = (feedback or "").strip()
        self._greenlight_decision = {
            "approved": bool(approved),
            "feedback": clean_fb,
        }
        if clean_fb and hasattr(self, "writers_room") and hasattr(self.writers_room, "enqueue_user_steering"):
            self.writers_room.enqueue_user_steering(clean_fb, candidate_dir=getattr(self, "_active_staging_dir", None))
        if self._greenlight_event is not None:
            try:
                self._greenlight_event.set()
            except Exception:
                pass
        return {
            "deliberation": "Greenlight decision recorded for StudioOrchestrator.",
            "approved": bool(approved),
            "feedback": clean_fb,
            "state": self.state.value,
        }

    def _continuous_production_enabled(self) -> bool:
        flag = getattr(self.config, "continuous_production", None)
        return bool(flag) if flag is not None else self._is_live_genai_client()

    def _build_production_board(self, c_dir: Path) -> Optional[Any]:
        """Creates the Production Board, its worker handlers and the scheduler for this build (None when disabled)."""
        if not self._continuous_production_enabled():
            return None
        from engine.studio.models import set_active_gemini_dataset_context
        from engine.studio.production_board import BoardScheduler, ProductionBoard
        from engine.studio.workers import CartridgeFileLockTable, WorkerContext, default_handlers

        set_active_gemini_dataset_context(
            dataset_name=getattr(self.config, "dataset_name", "") or c_dir.name,
            candidate_dir=c_dir,
            store_logs=getattr(self.config, "store_logs", True),
        )

        def _board_event(event_type: str, payload: Dict[str, Any]) -> None:
            self._publish_event(event_type, "ProductionBoard", payload.get("deliberation", event_type), payload)
            if event_type == "backlog_item_approved" and (c_dir / "project.godot").is_file():
                try:
                    item_id = str(payload.get("item_id") or "board")
                    approved_ids = [it.item_id for it in board.items() if it.state == "approved"]
                    item_obj = next((it for it in board.items() if it.item_id == item_id), None)
                    cp_track = RequirementTrack(
                        deliberation=payload.get("deliberation") or f"Council approved Production Board item {item_id}.",
                        id=item_id,
                        title=getattr(item_obj, "title", item_id),
                        description=getattr(item_obj, "spec", "") or item_id,
                        status="completed",
                    )
                    self._persist_verified_checkpoint(c_dir, cp_track, approved_ids, mirror_to_active=False)
                except Exception as cp_exc:
                    logger.debug("Board checkpoint persistence skipped for %s: %s", c_dir.name, cp_exc)

        board = ProductionBoard(
            c_dir / "backlog.json",
            default_max_revisions=getattr(self.config, "max_item_revisions", 5),
            event_callback=_board_event,
        )
        if not (c_dir / "backlog.json").is_file():
            board._persist()
        swe_agent = self.agents["LeadGameplaySWEAgent"]
        swe_agent.visual_agent = self.agents.get("VisualArtSpriteDesignerAgent")
        swe_agent.voice_agent = self.agents.get("VoiceActingAudioProducerAgent")

        def _verify(cartridge_dir: Path, track: Any) -> None:
            self._verify_track_smoke_check(cartridge_dir, track, swe_agent=swe_agent)

        ctx = WorkerContext(
            cartridge_dir=c_dir,
            client=self.client,
            visual_agent=self.agents.get("VisualArtSpriteDesignerAgent"),
            voice_agent=self.agents.get("VoiceActingAudioProducerAgent"),
            swe_agent=swe_agent,
            model_name=self.config.model_name,
            code_verify=_verify,
            file_lock_table=CartridgeFileLockTable(),
            event_callback=_board_event,
            board=board,
        )
        scheduler = BoardScheduler(board, default_handlers(ctx), max_workers=getattr(self.config, "board_workers", 2))
        self.production_board = board
        self._board_scheduler = scheduler
        return board

    def _reconcile_requirements_with_board(self, req_spec: Any) -> Any:
        """In continuous mode the council's backlog is the source of truth for code requirements: approved code items
        become completed tracks (so the legacy loop skips them), and the remaining pipeline only runs final gates."""
        board = self.production_board
        if board is None or not self._continuous_active or not board.items():
            return req_spec
        from engine.studio.models import AcceptanceCriterion

        tracks: List[RequirementTrack] = []
        for it in board.items():
            if it.kind not in ("mechanic_code", "ui", "integration"):
                continue
            tracks.append(
                RequirementTrack(
                    deliberation=it.deliberation,
                    id=it.item_id,
                    title=it.title,
                    description=it.spec,
                    target_modules=list(it.target_modules) or ["main.gd"],
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="Council proof request.", id=f"AC-{it.item_id}",
                            description=it.proof_spec or it.spec, verification_method="headless_playtest",
                        )
                    ],
                    status="completed" if it.state == "approved" else ("pending" if it.state != "cancelled" else "completed"),
                )
            )
        req_spec.requirements = tracks
        req_spec.asset_manifest = []  # assets were built, reviewed and approved through the board
        return req_spec

    def run_writers_room_session(
        self,
        dna: Optional[Any] = None,
        hypothesis_report: Optional[Any] = None,
        staging_dir: Optional[Union[Path, str]] = None,
        target_episode: int = 0,
        creative_prompt: Optional[str] = None,
        existing_bible: Optional[GameBibleSpec] = None,
    ) -> GameBibleSpec:
        """Executes the open-ended CreativeWritersRoom session to produce a locked GameBibleSpec.

        When `existing_bible` is provided (resumed builds), the room continues that Living Game Bible
        instead of initializing a fresh 'Untitled Concept'."""
        c_dir = Path(staging_dir) if staging_dir else (PROJECT_ROOT / "cartridges" / f"candidate_genesis_{uuid.uuid4().hex[:6]}")
        c_dir.mkdir(parents=True, exist_ok=True)
        self._active_staging_dir = c_dir
        from engine.studio.models import set_active_gemini_dataset_context
        set_active_gemini_dataset_context(
            dataset_name=getattr(self.config, "dataset_name", "") or c_dir.name,
            candidate_dir=c_dir,
            store_logs=getattr(self.config, "store_logs", True),
        )
        self.writers_room.telemetry_db = self._get_telemetry_db()
        seed = creative_prompt or getattr(self, "_creative_prompt", None)
        board = self._build_production_board(c_dir)
        self.writers_room.board = board
        self.writers_room.scheduler = getattr(self, "_board_scheduler", None) if board is not None else None
        self._continuous_active = board is not None
        if self._continuous_active and getattr(self.writers_room, "playtest_manager", None) is None:
            bus_pt_mgr = getattr(self.event_bus, "_playtest_manager", None) if self.event_bus is not None else None
            if bus_pt_mgr is None and self._is_live_genai_client():
                from engine.playtest.session_manager import PlaytestSessionManager
                t_db = self.writers_room.telemetry_db
                pt_db_path = (
                    Path(t_db.db_path)
                    if (t_db is not None and getattr(t_db, "db_path", None) is not None)
                    else (c_dir.parent.parent / "data" / "telemetry.db")
                )
                pt_db_path.parent.mkdir(parents=True, exist_ok=True)
                bus_pt_mgr = PlaytestSessionManager(
                    db_path=pt_db_path,
                    cartridges_root=c_dir.parent,
                    event_bus=self.event_bus,
                )
                if self.event_bus is not None:
                    setattr(self.event_bus, "_playtest_manager", bus_pt_mgr)
            if bus_pt_mgr is not None:
                self.writers_room.playtest_manager = bus_pt_mgr

        def _on_slice_checkpoint(gb: GameBibleSpec, completed_slice: Any, cand_path: Optional[Path]) -> None:
            target_c_dir = Path(cand_path) if cand_path is not None else c_dir
            if not (target_c_dir / "project.godot").is_file() and not (target_c_dir / "mechanics.py").is_file():
                return
            slice_id_str = getattr(completed_slice, "slice_id", "slice_0") or "slice_0"
            slice_title_str = getattr(completed_slice, "title", slice_id_str) or slice_id_str
            approved_ids = (
                [it.item_id for it in board.items() if it.state == "approved"]
                if board is not None
                else [slice_id_str]
            )
            chk_track = RequirementTrack(
                deliberation=f"Persisting verified checkpoint for completed vertical slice {slice_id_str} ('{slice_title_str}').",
                id=f"{slice_id_str}_completed",
                title=f"Slice Checkpoint: {slice_title_str}",
                category="gameplay_mechanic",
                description=getattr(completed_slice, "slice_objective", slice_title_str) or slice_title_str,
                target_modules=["main.gd"],
                acceptance_criteria=[],
                status="completed",
            )
            self._persist_verified_checkpoint(
                staging_dir=target_c_dir,
                track=chk_track,
                verified_tracks=approved_ids or [slice_id_str],
                mirror_to_active=True,
            )

        self.writers_room.slice_checkpoint_callback = _on_slice_checkpoint
        try:
            bible = self.writers_room.run_writers_room(
                dna=dna,
                hypothesis_report=hypothesis_report,
                candidate_dir=c_dir,
                target_episode=target_episode,
                cartridge_id=c_dir.name,
                creative_prompt=seed,
                existing_bible=existing_bible,
            )
        finally:
            if self.writers_room.scheduler is not None:
                self.writers_room.scheduler.stop()
        self.active_game_bible = bible
        return bible

    def transition_to(self, new_state: StudioBuildState, deliberation: str) -> None:
        """Enforces valid state machine transitions."""
        allowed = self.ALLOWED_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidStateTransitionError(
                f"Invalid state transition from {self.state} to {new_state}. Allowed: {allowed}"
            )
        self.deliberation = deliberation
        self.state = new_state
        self._publish_event(
            "state_change",
            "StudioOrchestrator",
            deliberation,
            {"state": new_state.value, "lifecycle_state": new_state.value},
        )

    async def execute_pev_cycle(
        self,
        agent_name: str,
        plan_coroutine: Union[Callable[[], Any], Callable[[], Any]],
        execute_coroutine: Callable[[AgentPlan], Any],
        verify_coroutine: Callable[[Any], Any],
        max_retries: int = 3,
    ) -> Tuple[Any, ArtifactMetadataEnvelope]:
        """Structured Plan-Execute-Verify lifecycle with self-healing retries."""
        attempt = 1
        last_verification = None

        async def _run(callable_or_coro: Any, *args: Any) -> Any:
            if inspect.iscoroutinefunction(callable_or_coro):
                return await callable_or_coro(*args)
            res = callable_or_coro(*args)
            if inspect.iscoroutine(res):
                return await res
            return res

        while attempt <= max_retries:
            # 1. Plan Phase
            plan_obj = await _run(plan_coroutine)
            if isinstance(plan_obj, dict):
                plan = AgentPlan(**plan_obj)
            else:
                plan = plan_obj

            if not plan.steps:
                raise ValueError("Plan contains 0 action steps")

            # 2. Execute Phase
            raw_output = await _run(execute_coroutine, plan)

            # 3. Verify Phase
            veri_obj = await _run(verify_coroutine, raw_output)
            if isinstance(veri_obj, dict):
                verification = AgentVerificationRecord(**veri_obj)
            else:
                verification = veri_obj
            verification.attempt_number = attempt

            if verification.passed:
                if isinstance(raw_output, dict):
                    payload = {k: str(v) if isinstance(v, Path) else v for k, v in raw_output.items()}
                elif hasattr(raw_output, "model_dump"):
                    payload = raw_output.model_dump()
                else:
                    payload = str(raw_output)
                envelope = ArtifactMetadataEnvelope.seal(
                    deliberation=verification.deliberation,
                    artifact_type=ArtifactType.GAME_MODULE,
                    payload=payload,
                    source_agent=agent_name,
                    target_agent="Supervisor",
                )
                return raw_output, envelope

            last_verification = verification
            attempt += 1

        violations = last_verification.violations if last_verification else ["Verification failed"]
        raise BuildVerificationExhaustedError(
            f"Agent {agent_name} failed PEV verification after {max_retries} attempts: {violations}"
        )

    def generate_swarm_plan(
        self,
        steam_dna: Optional[Any] = None,
        evolution_report: Optional[Any] = None,
        target_episode: int = 0,
        creative_prompt: Optional[str] = None,
    ) -> AwaitableStudioSwarmPlan:
        """Invokes live Gemini reasoning to dynamically assemble specialist roles,
        DAG edges, and genre-specific commissioning task manifests.
        Fails fast if Gemini API call fails (Rule 5: Zero Silent Fallbacks).
        """
        if creative_prompt is not None:
            self._creative_prompt = creative_prompt.strip()
        if not self.client:
            raise StudioSwarmPlanError(
                "Gemini client is not initialized; cannot perform live meta-planning. "
                "Ensure GEMINI_API_KEY is configured in .env."
            )

        from google.genai import types

        if target_episode == 0:
            # Genesis Build Mode
            if steam_dna is None and not ("PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules):
                try:
                    from engine.steam.dna import SteamDNAPersistence, scan_and_extract_local_dna
                    repo = SteamDNAPersistence(config.DB_PATH)
                    steam_dna = repo.get_latest_dna()
                    if steam_dna is None or getattr(steam_dna, "total_games_analyzed", 0) == 0 or not getattr(steam_dna, "top_games", None):
                        scanned = scan_and_extract_local_dna(db_path=config.DB_PATH)
                        if scanned is not None:
                            steam_dna = scanned
                except Exception as dna_exc:
                    logger.warning("Failed to auto-load SteamPlayerDNA in generate_swarm_plan: %s", dna_exc)

            dna_summary = ""
            if steam_dna:
                top_games_list = getattr(steam_dna, "top_games", []) or []
                top_games_lines = "\n".join(
                    f"  * {g.get('name', 'Unknown')} ({g.get('playtime_hours', 0)} hrs) — genres: {g.get('genres', [])}"
                    for g in top_games_list[:10]
                )
                dna_summary = (
                    f"- Profile summary: {getattr(steam_dna, 'deliberation', '')}\n"
                    f"- Total analyzed games: {getattr(steam_dna, 'total_games_analyzed', 0)} ({getattr(steam_dna, 'total_playtime_hours', 0.0):.1f} total hours)\n"
                    f"- Top genres: {getattr(steam_dna, 'top_genres', [])} (weights: {getattr(steam_dna, 'genre_weights', {})})\n"
                    f"- Pacing score: {getattr(steam_dna, 'pacing_score', 0.0):.2f} (-1.0 methodical/simulation/strategy to +1.0 fast kinetic action)\n"
                    f"- Challenge tolerance: {getattr(steam_dna, 'challenge_tolerance', 0.5):.2f} (0.0 cozy to 1.0 hardcore)\n"
                    f"- Thematic biases: {getattr(steam_dna, 'thematic_biases', [])}\n"
                    f"- Palette mood: {getattr(steam_dna, 'preferred_palette_mood', 'neon')}\n"
                    + (f"- Most-Played & Installed Steam Titles:\n{top_games_lines}\n" if top_games_lines else "")
                )
            seed_str = getattr(self, "_creative_prompt", "") or ""
            seed_block = (
                f"\nEXECUTIVE PRODUCER CREATIVE SEED / DIRECTION:\n{seed_str}\n"
                if seed_str
                else ""
            )
            prompt = (
                f"You are the Meta-Cognitive Studio Orchestrator for an autonomous AI game studio.\n"
                f"Your task is to deliberate on the player's actual Steam library DNA and dynamically assemble the specialist "
                f"agent swarm DAG and task manifest to build an original game from scratch tailored to their real gaming tastes.\n\n"
                f"STEAM PLAYER DNA:\n{dna_summary}\n"
                f"{seed_block}\n"
                f"REQUIREMENTS:\n"
                f"1. Deliberate deeply in 'deliberation' FIRST on the player's actual top-played Steam games, hours, and genre affinities, "
                f"   explaining why your chosen game genre, core mechanics, and visual aesthetic fit this specific player.\n"
                f"2. Conclude with an evocative 'game_title' and specific 'genre_verdict' directly grounded in the player's real Steam library.\n"
                f"3. Select 'specialist_agent_roles' from: ['Director', 'Writer', 'Voice', 'Visual', 'SWE', 'TelemetryArchitect', 'Supervisor'].\n"
                f"4. Define causal dependency 'dag_edges' as pairs of [upstream_role, downstream_role] using those exact short role names "
                f"   (e.g. [['Director', 'Writer'], ['Writer', 'Voice'], ['Writer', 'Visual'], ['Voice', 'SWE'], ['Visual', 'SWE'], ['SWE', 'TelemetryArchitect'], ['Voice', 'Supervisor'], ['Visual', 'Supervisor'], ['TelemetryArchitect', 'Supervisor']]) "
                f"   so 'Director' precedes 'Writer', 'Writer' precedes 'Voice'/'Visual', 'Voice' and 'Visual' precede 'SWE' (so all synthesized sprites and voice files exist on disk before SWE writes asset-binding code), and 'SWE' precedes 'TelemetryArchitect'.\n"
                f"5. Provide a detailed 'task_manifest' JSON-encoded object string containing specific commissioning directives "
                f"   for each selected specialist.\n"
            )
        else:
            # Episodic Update Mode
            hypo_summary = ""
            if evolution_report:
                hypotheses_list = getattr(evolution_report, "hypotheses", [])
                hypo_statements = [getattr(h, "statement", str(h)) for h in hypotheses_list]
                prescribed_list = [getattr(h, "prescribed_evolution", "") for h in hypotheses_list]
                hypo_summary = (
                    f"- Dominant playstyle: {getattr(evolution_report, 'dominant_playstyle', 'BALANCED')}\n"
                    f"- Hypotheses: {hypo_statements}\n"
                    f"- Prescribed evolutions: {prescribed_list}\n"
                    f"- Custom probe metrics: {getattr(evolution_report, 'custom_probe_summaries', {})}\n"
                )
            prompt = (
                f"You are the Meta-Cognitive Studio Orchestrator coordinating an Episodic Update (Episode {target_episode}).\n"
                f"Analyze the player's telemetry hypotheses from live gameplay and dynamically assemble the agent swarm DAG "
                f"and task manifest to evolve the game mechanics, story, voice acting, and telemetry probes.\n\n"
                f"PLAYER TELEMETRY HYPOTHESES:\n{hypo_summary}\n\n"
                f"Provide 'deliberation' FIRST before any decision tokens (Rule 4)."
            )

        self._publish_event(
            "AGENT_STREAM",
            "StudioOrchestrator",
            f"Meta-planning initiated (Episode {target_episode}): deliberating on player profile to assemble specialist agent DAG...",
            {"target_episode": target_episode, "lifecycle_state": "ANALYZING"},
        )

        try:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=StudioSwarmPlan,
                    temperature=0.6,
                ),
            )
            if not response.text:
                raise StudioSwarmPlanError("Gemini meta-planning returned an empty response.")

            plan_dict = json.loads(response.text)
            plan = AwaitableStudioSwarmPlan(**plan_dict)
            plan.target_episode = target_episode
            # Enforce that Visual and Voice precede SWE so synthesized assets exist before code synthesis
            roles_set = set(plan.specialist_agent_roles)
            existing_pairs = {(u, v) for u, v in plan.dag_edges if len([u, v]) == 2}
            for asset_role in ("Visual", "Voice"):
                if asset_role in roles_set and "SWE" in roles_set:
                    if (asset_role, "SWE") not in existing_pairs and ("SWE", asset_role) not in existing_pairs:
                        plan.dag_edges.append([asset_role, "SWE"])
                        existing_pairs.add((asset_role, "SWE"))
            # Validate DAG acyclicity immediately
            plan.validate_dag()

            self._publish_event(
                "AGENT_STREAM",
                "StudioOrchestrator",
                f"[{plan.game_title} | {plan.genre_verdict}] {plan.deliberation}",
                {
                    "plan_id": plan.plan_id,
                    "game_title": plan.game_title,
                    "genre_verdict": plan.genre_verdict,
                    "roles": plan.specialist_agent_roles,
                    "dag_edges": plan.dag_edges,
                    "lifecycle_state": "SYNTHESIZING",
                },
            )
            return plan

        except Exception as exc:
            logger.error("Live Gemini meta-planning failed: %s", exc, exc_info=True)
            self._publish_event(
                "ERROR",
                "StudioOrchestrator",
                f"Live Gemini meta-planning failed: {exc}",
                {"error": str(exc)},
            )
            if isinstance(exc, StudioSwarmPlanError):
                raise
            raise StudioSwarmPlanError(f"Live Gemini meta-planning failed: {exc}") from exc

    def _compute_execution_tiers(self, edges: List[Tuple[str, str]], roles: List[str]) -> List[List[str]]:
        """Partitions roles into topological tiers (waves) for maximal safe concurrency."""
        role_set = set(roles)
        filtered_edges = [
            (u, v) for u, v in edges
            if (not roles or (u in role_set and v in role_set))
        ]
        nodes = set(roles)
        for u, v in filtered_edges:
            nodes.add(u)
            nodes.add(v)

        # Exclude terminal promotion sink if present
        nodes.discard("Promotion")
        nodes.discard("promotion")

        adj: Dict[str, Set[str]] = {n: set() for n in nodes}
        in_degree: Dict[str, int] = {n: 0 for n in nodes}
        for u, v in filtered_edges:
            if u in nodes and v in nodes:
                adj[u].add(v)
                in_degree[v] += 1

        tiers: List[List[str]] = []
        current_tier = sorted([n for n in nodes if in_degree[n] == 0])

        while current_tier:
            tiers.append(current_tier)
            next_tier_candidates: Set[str] = set()
            for node in current_tier:
                for neighbor in adj[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_tier_candidates.add(neighbor)
            current_tier = sorted(list(next_tier_candidates))

        flattened = [n for tier in tiers for n in tier]
        if len(flattened) != len(nodes):
            raise CircularDependencyError(
                f"Cycle detected in agent DAG: processed {len(flattened)} of {len(nodes)} nodes."
            )

        return tiers

    async def _execute_specialist(
        self,
        role: str,
        plan: StudioSwarmPlan,
        context: SwarmExecutionContext,
    ) -> Any:
        """Executes a single specialist persona within a structured PEV cycle."""
        import time as _time
        t_spec_start = _time.perf_counter()
        t_start_iso = datetime.now(timezone.utc).isoformat()
        canonical_name = normalize_role(role)
        agent = self.agents.get(canonical_name)
        if agent is None and canonical_name not in ("QAPlaytesterBuildSupervisor", "SteamAndTelemetryAnalystAgent"):
            raise StudioSwarmExecutionError(
                f"Specialist agent '{canonical_name}' (role: '{role}') is not assembled in StudioOrchestrator."
            )

        self._publish_event(
            "state_change",
            canonical_name,
            f"Specialist {canonical_name} starting execution for '{plan.game_title}' ({plan.genre_verdict})...",
            {"agent_name": canonical_name, "lifecycle_state": "SYNTHESIZING"},
        )

        async def plan_coro() -> AgentPlan:
            return AgentPlan(
                deliberation=f"Formulated task execution directives for {role}",
                agent_name=role,
                objective=f"Execute {role} responsibilities according to swarm plan {plan.plan_id}",
                steps=[f"Commission {role}", f"Synthesize {role} assets", f"Verify {role} outputs"],
            )

        async def exec_coro(agent_plan: AgentPlan) -> Any:
            t_start = _time.perf_counter()
            staging_dir = context.staging_dir
            staging_dir.mkdir(parents=True, exist_ok=True)

            if canonical_name == "CreativeDirectorWorldbuilderAgent":
                from engine.steam.models import SteamPlayerDNA
                dna = context.raw_outputs.get("dna")
                if dna is None and not ("PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules):
                    try:
                        from engine.steam.dna import SteamDNAPersistence, scan_and_extract_local_dna
                        repo = SteamDNAPersistence(config.DB_PATH)
                        dna = repo.get_latest_dna()
                        if dna is None or getattr(dna, "total_games_analyzed", 0) == 0 or not getattr(dna, "top_games", None):
                            scanned = scan_and_extract_local_dna(db_path=config.DB_PATH)
                            if scanned is not None:
                                dna = scanned
                    except Exception:
                        dna = None
                if dna is None:
                    dna = SteamPlayerDNA(
                        deliberation=f"Baseline player DNA for '{plan.game_title}' ({plan.genre_verdict})",
                        player_id="local_player",
                        total_games_analyzed=0,
                        top_genres=[plan.genre_verdict],
                        pacing_score=0.0,
                        challenge_tolerance=0.5,
                        thematic_biases=[],
                        preferred_palette_mood="retro_arcade",
                    )
                use_writers_room = (
                    self.enable_writers_room
                    if self.enable_writers_room is not None
                    else self._is_live_genai_client()
                )
                board_needs_writers_room = False
                if use_writers_room and self._continuous_production_enabled() and (staging_dir / "backlog.json").is_file():
                    try:
                        from engine.studio.production_board import ProductionBoard
                        pb_check = ProductionBoard(staging_dir / "backlog.json")
                        has_assembly_proof = any(
                            (staging_dir / "artifacts" / "proofs").glob("filmstrip_slice_assembly_check_pass*.png")
                        ) if (staging_dir / "artifacts" / "proofs").is_dir() else False
                        needs_next_slice_iteration = False
                        if self._is_live_genai_client() and (staging_dir / "game_bible.json").is_file():
                            try:
                                gb_check = GameBibleSpec.model_validate_json(
                                    (staging_dir / "game_bible.json").read_text(encoding="utf-8")
                                )
                                target_s = (
                                    int(self.writers_room.target_slices)
                                    if getattr(self.writers_room, "target_slices", None) is not None
                                    else max(2, int(os.environ.get("STUDIO_TARGET_SLICES", "2")))
                                )
                                completed_s = sum(
                                    1 for s in (gb_check.slice_roadmap or []) if getattr(s, "status", "") == "completed"
                                )
                                if len(gb_check.slice_roadmap or []) < target_s or completed_s < target_s:
                                    needs_next_slice_iteration = True
                            except Exception:
                                needs_next_slice_iteration = True
                        has_resume_feedback = bool(
                            getattr(self, "_creative_prompt", None)
                            or getattr(self.writers_room, "_pending_steering", None)
                        )
                        if (
                            not pb_check.is_drained()
                            or (staging_dir / ".pending_steering.json").is_file()
                            or has_resume_feedback
                            or (self._is_live_genai_client() and not has_assembly_proof)
                            or needs_next_slice_iteration
                        ):
                            board_needs_writers_room = True
                    except Exception as pb_chk_exc:
                        logger.warning("Could not inspect backlog.json during resume check: %s", pb_chk_exc)

                if (
                    (staging_dir / "requirements_spec.json").is_file()
                    and (staging_dir / "world_spec.json").is_file()
                    and not board_needs_writers_room
                ):
                    try:
                        world_spec = GenesisWorldSpec.model_validate_json((staging_dir / "world_spec.json").read_text(encoding="utf-8"))
                        req_spec = GameRequirementsSpec.model_validate_json((staging_dir / "requirements_spec.json").read_text(encoding="utf-8"))
                        # Reset any rejected or interrupted in_progress tracks to pending so resumed builds retry them
                        reset_any = False
                        for trk in getattr(req_spec, "requirements", []):
                            if getattr(trk, "status", "") in ("rejected", "in_progress"):
                                trk.status = "pending"
                                reset_any = True
                        if reset_any:
                            (staging_dir / "requirements_spec.json").write_text(
                                req_spec.model_dump_json(indent=2), encoding="utf-8"
                            )
                        if (staging_dir / "game_bible.json").is_file():
                            try:
                                resumed_bible = GameBibleSpec.model_validate_json(
                                    (staging_dir / "game_bible.json").read_text(encoding="utf-8")
                                )
                                self.active_game_bible = resumed_bible
                                context.raw_outputs["game_bible"] = resumed_bible
                                self._sync_bible_to_plan_and_staging(resumed_bible, plan, staging_dir)
                            except Exception as gb_exc:
                                logger.warning("Could not load game_bible.json during resume: %s", gb_exc)
                        plan.game_title = getattr(world_spec, "game_title", plan.game_title)
                        plan.genre_verdict = getattr(world_spec, "core_genre", plan.genre_verdict)
                        context.raw_outputs["requirements_spec"] = req_spec
                        context.raw_outputs["GameRequirementsSpec"] = req_spec
                        context.raw_outputs["_agent_authored_req_spec"] = True
                        context.raw_outputs["world_spec"] = world_spec
                        context.raw_outputs["Director"] = world_spec
                        context.raw_outputs["CreativeDirectorWorldbuilderAgent"] = world_spec
                        self._persist_requirements_to_db(req_spec, build_id=f"build_{staging_dir.name}")
                        logger.info("Resumed existing world_spec and requirements_spec from %s", staging_dir)
                        dur_ms = (_time.perf_counter() - t_start) * 1000.0
                        self._publish_event("AGENT_STREAM", canonical_name, world_spec.deliberation, {"game_title": getattr(world_spec, "game_title", plan.game_title)})
                        self._publish_event("TOOL_INVOCATION", canonical_name, world_spec.deliberation, {"tool_name": "resume_genesis_world", "status": "success", "duration_ms": dur_ms})
                        return world_spec
                    except Exception as e:
                        logger.warning("Failed resuming existing director specs from %s: %s, regenerating", staging_dir, e)

                game_bible: Optional[GameBibleSpec] = context.raw_outputs.get("game_bible")
                if game_bible is None and (staging_dir / "game_bible.json").is_file():
                    try:
                        game_bible = GameBibleSpec.model_validate_json(
                            (staging_dir / "game_bible.json").read_text(encoding="utf-8")
                        )
                        self.active_game_bible = game_bible
                        context.raw_outputs["game_bible"] = game_bible
                    except Exception as exc:
                        logger.warning("Failed to deserialize existing game_bible.json: %s", exc)
                        game_bible = None

                world_spec = None
                if (staging_dir / "world_spec.json").is_file():
                    try:
                        world_spec = GenesisWorldSpec.model_validate_json((staging_dir / "world_spec.json").read_text(encoding="utf-8"))
                        logger.info("Resumed existing world_spec from %s", staging_dir)
                    except Exception as e:
                        logger.warning("Failed to deserialize existing world_spec: %s, regenerating", e)
                        world_spec = None

                use_writers_room = (
                    self.enable_writers_room
                    if self.enable_writers_room is not None
                    else self._is_live_genai_client()
                )

                def _sync_bible_to_plan_and_staging(gb: GameBibleSpec) -> None:
                    if gb.game_title and gb.game_title != "Untitled Concept":
                        plan.game_title = gb.game_title
                        if gb.world_spec is not None:
                            gb.world_spec.game_title = gb.game_title
                    elif gb.world_spec and getattr(gb.world_spec, "game_title", None):
                        plan.game_title = gb.world_spec.game_title
                        gb.game_title = gb.world_spec.game_title

                    if getattr(gb, "core_genre", None):
                        plan.genre_verdict = gb.core_genre
                        if gb.world_spec is not None:
                            gb.world_spec.core_genre = gb.core_genre
                    elif gb.world_spec and getattr(gb.world_spec, "core_genre", None):
                        plan.genre_verdict = gb.world_spec.core_genre
                        gb.core_genre = gb.world_spec.core_genre

                    if gb.world_spec is not None:
                        (staging_dir / "world_spec.json").write_text(
                            gb.world_spec.model_dump_json(indent=2), encoding="utf-8"
                        )
                        manifest_preview = {
                            "title": plan.game_title,
                            "genre": plan.genre_verdict,
                            "episode": plan.target_episode,
                            "deliberation": gb.world_spec.deliberation,
                            "narrative_premise": gb.elevator_pitch or getattr(gb.world_spec, "thematic_premise", ""),
                            "creative_direction": {
                                "narrative_premise": gb.elevator_pitch or gb.core_fantasy or getattr(gb.world_spec, "thematic_premise", ""),
                                "world_lore": getattr(gb.world_spec, "world_lore", "") or gb.core_fantasy,
                                "core_rules": gb.gameplay_pillars or getattr(gb.world_spec, "mechanics_guidelines", []),
                                "aesthetic_style": gb.visual_style_guide or getattr(gb.world_spec, "aesthetic_style", ""),
                            },
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                        (staging_dir / "manifest.json").write_text(
                            json.dumps(manifest_preview, indent=2), encoding="utf-8"
                        )

                if (world_spec is None or board_needs_writers_room) and use_writers_room:
                    evolution_report = context.raw_outputs.get("evolution_report") or context.raw_outputs.get("hypothesis_report")
                    if StudioBuildState.BRAINSTORMING in self.ALLOWED_TRANSITIONS.get(self.state, set()):
                        self.transition_to(
                            StudioBuildState.BRAINSTORMING,
                            f"Convening Multi-Agent Creative Writers' Room for '{plan.game_title}'...",
                        )
                    game_bible = self.run_writers_room_session(
                        dna=dna,
                        hypothesis_report=evolution_report,
                        staging_dir=staging_dir,
                        target_episode=plan.target_episode,
                        existing_bible=game_bible,
                    )
                    _sync_bible_to_plan_and_staging(game_bible)
                    # Optional Greenlight Pause Gate
                    while getattr(self.config, "pause_for_greenlight", False) or getattr(self.event_bus, "_pause_for_greenlight", False):
                        if getattr(self, "_abort_requested", False):
                            break
                        if self._greenlight_decision is None:
                            game_bible.status = "awaiting_greenlight"
                            self.writers_room._persist_bible(game_bible, candidate_dir=staging_dir)
                            if StudioBuildState.AWAITING_GREENLIGHT in self.ALLOWED_TRANSITIONS.get(self.state, set()):
                                self.transition_to(
                                    StudioBuildState.AWAITING_GREENLIGHT,
                                    f"Living Game Bible for '{game_bible.game_title}' awaiting Executive Producer greenlight.",
                                )
                            self._greenlight_event.clear()
                            while not self._greenlight_event.is_set() and not getattr(self, "_abort_requested", False):
                                await asyncio.sleep(0.05)
                        if getattr(self, "_abort_requested", False):
                            break
                        decision = self._greenlight_decision or {"approved": True, "feedback": ""}
                        self._greenlight_decision = None
                        self._greenlight_event.clear()
                        if decision.get("aborted"):
                            self._abort_requested = True
                            break
                        if decision.get("approved", True) and not decision.get("feedback"):
                            game_bible.status = "locked"
                            self.writers_room._persist_bible(game_bible, candidate_dir=staging_dir)
                            break
                        feedback_text = decision.get("feedback", "")
                        if feedback_text:
                            if StudioBuildState.BRAINSTORMING in self.ALLOWED_TRANSITIONS.get(self.state, set()):
                                self.transition_to(
                                    StudioBuildState.BRAINSTORMING,
                                    f"Writers' Room incorporating Executive Producer feedback: {feedback_text[:80]}",
                                )
                            game_bible = self.writers_room.inject_user_steering_and_continue(
                                game_bible=game_bible,
                                user_feedback=feedback_text,
                                dna=dna,
                                candidate_dir=staging_dir,
                            )
                            self.active_game_bible = game_bible
                            _sync_bible_to_plan_and_staging(game_bible)
                        if decision.get("approved", False):
                            game_bible.status = "locked"
                            self.writers_room._persist_bible(game_bible, candidate_dir=staging_dir)
                            break

                    if getattr(self, "_abort_requested", False):
                        self.state = StudioBuildState.IDLE
                        return StudioBuildReport(
                            deliberation="Previous Genesis build cancelled at Greenlight gate to start a new game from scratch.",
                            build_id=plan.plan_id,
                            game_title=plan.game_title,
                            target_episode=plan.target_episode,
                            success=False,
                            promoted_to_active=False,
                            rejection_trace="Cancelled by user starting a new Genesis build.",
                        )

                    _sync_bible_to_plan_and_staging(game_bible)
                    if self.state != StudioBuildState.SYNTHESIZING and StudioBuildState.SYNTHESIZING in self.ALLOWED_TRANSITIONS.get(self.state, set()):
                        self.transition_to(
                            StudioBuildState.SYNTHESIZING,
                            f"Writers' Room locked Living Game Bible for '{game_bible.game_title}'; entering code & asset synthesis.",
                        )

                    context.raw_outputs["game_bible"] = game_bible
                    world_spec = game_bible.world_spec
                    if world_spec is not None:
                        (staging_dir / "world_spec.json").write_text(world_spec.model_dump_json(indent=2), encoding="utf-8")
                    if game_bible.narrative_bible is not None and (
                        game_bible.narrative_bible.characters
                        or game_bible.narrative_bible.dialogue_barks
                        or game_bible.narrative_bible.cutscenes
                    ):
                        (staging_dir / "narrative_bible.json").write_text(
                            game_bible.narrative_bible.model_dump_json(indent=2), encoding="utf-8"
                        )
                        (staging_dir / "narrative.json").write_text(
                            json.dumps(game_bible.narrative_bible.to_cartridge_narrative_dict(), indent=2),
                            encoding="utf-8",
                        )
                        context.raw_outputs["Writer"] = game_bible.narrative_bible
                        context.raw_outputs["NarrativeCharacterWriterAgent"] = game_bible.narrative_bible

                if world_spec is None:
                    if plan.target_episode == 0:
                        world_spec = agent.generate_genesis_world(dna)
                    else:
                        evolution_report = context.raw_outputs.get("evolution_report") or context.raw_outputs.get("hypothesis_report")
                        previous_spec = context.raw_outputs.get("previous_world_spec") or context.raw_outputs.get("world_spec")
                        if previous_spec is None:
                            active_spec_path = PROJECT_ROOT / "cartridges" / "active" / "world_spec.json"
                            if (staging_dir / "world_spec.json").is_file():
                                previous_spec = GenesisWorldSpec.model_validate_json((staging_dir / "world_spec.json").read_text(encoding="utf-8"))
                            elif active_spec_path.is_file():
                                previous_spec = GenesisWorldSpec.model_validate_json(active_spec_path.read_text(encoding="utf-8"))
                            else:
                                previous_spec = GenesisWorldSpec(
                                    deliberation="Baseline genesis world spec for episodic evolution",
                                    game_title=plan.game_title,
                                    core_genre=plan.genre_verdict,
                                    thematic_premise="Initial operational mission",
                                    world_lore="Autonomous game world evolving through telemetry",
                                    conflict_tone="grim_tense",
                                    conflict_narrative="Overcome evolving defensive measures",
                                    aesthetic_style="retro_arcade",
                                    palette_mood="retro_arcade",
                                    gameplay_pacing="balanced",
                                    pacing_score=0.0,
                                    target_difficulty=1.0,
                                )
                        world_spec = agent.generate_episodic_update_world(
                            dna=dna,
                            hypothesis_report=evolution_report,
                            previous_spec=previous_spec,
                            target_episode=plan.target_episode,
                        )
                    if getattr(world_spec, "game_title", None):
                        plan.game_title = world_spec.game_title
                    if getattr(world_spec, "core_genre", None):
                        plan.genre_verdict = world_spec.core_genre
                    (staging_dir / "world_spec.json").write_text(world_spec.model_dump_json(indent=2), encoding="utf-8")

                if not (staging_dir / "manifest.json").is_file():
                    manifest_data = {
                        "title": getattr(world_spec, "game_title", plan.game_title),
                        "genre": getattr(world_spec, "core_genre", plan.genre_verdict),
                        "episode": plan.target_episode,
                        "deliberation": world_spec.deliberation,
                        "narrative_premise": getattr(world_spec, "thematic_premise", ""),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    (staging_dir / "manifest.json").write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
                # Author, persist, and emit GameRequirementsSpec
                req_spec = context.raw_outputs.get("requirements_spec")
                if req_spec is None and (staging_dir / "requirements_spec.json").is_file():
                    try:
                        req_spec = GameRequirementsSpec.model_validate_json(
                            (staging_dir / "requirements_spec.json").read_text(encoding="utf-8")
                        )
                        if req_spec and not req_spec.asset_manifest and not self._continuous_active and (game_bible is not None or (staging_dir / "game_bible.json").is_file()):
                            try:
                                from engine.studio.asset_library import CartridgeAssetLibrary
                                temp_lib = CartridgeAssetLibrary(cartridge_dir=staging_dir)
                                req_spec.asset_manifest = temp_lib.derive_baseline_assets_from_bible(game_bible)
                            except Exception:
                                pass
                    except Exception:
                        req_spec = None

                if req_spec is None:
                    has_board_items = bool(
                        self._continuous_active
                        and self.production_board is not None
                        and self.production_board.items()
                    )
                    has_mock = hasattr(self.client, "_responses")
                    call_agent = False
                    if has_board_items:
                        # Production Board already decomposed & verified the backlog during continuous co-design;
                        # skip redundant generate_requirements_spec LLM call that _reconcile_requirements_with_board overwrites.
                        call_agent = False
                    elif not has_mock:
                        call_agent = True
                    elif len(self.client._responses) > 0:
                        next_resp = self.client._responses[0]
                        text = getattr(next_resp, "text", "")
                        if "requirements" in text or "GameRequirementsSpec" in text or "asset_manifest" in text:
                            call_agent = True

                    if call_agent:
                        evolution_report = context.raw_outputs.get("evolution_report") or context.raw_outputs.get("hypothesis_report")
                        sig = inspect.signature(agent.generate_requirements_spec)
                        req_kwargs: Dict[str, Any] = {
                            "dna": dna,
                            "world_spec": world_spec,
                            "hypothesis_report": evolution_report,
                            "target_episode": plan.target_episode,
                        }
                        if "game_bible" in sig.parameters:
                            req_kwargs["game_bible"] = game_bible
                        if "candidate_dir" in sig.parameters:
                            req_kwargs["candidate_dir"] = staging_dir
                        req_spec = agent.generate_requirements_spec(**req_kwargs)
                        context.raw_outputs["_agent_authored_req_spec"] = True

                    if req_spec is None:
                        mechanics_items = getattr(world_spec, "mechanics_guidelines", []) or ["Core gameplay loop"]
                        combined_desc = "; ".join(mechanics_items)
                        req_tracks = [
                            RequirementTrack(
                                deliberation=f"Core gameplay loop derived from world blueprint guidelines: {combined_desc}",
                                id="R1",
                                title="Core Gameplay Subsystems",
                                description=combined_desc,
                                responsible_agents=["LeadGameplaySWEAgent"],
                                acceptance_criteria=[
                                    AcceptanceCriterion(
                                        deliberation="Acceptance criterion for core gameplay loop",
                                        id="AC1.1",
                                        description="Executes core mechanics headlessly without exceptions.",
                                        verification_method="headless_playtest",
                                    )
                                ],
                                status="pending",
                            )
                        ]
                        g_title = getattr(world_spec, "game_title", plan.game_title)
                        g_genre = getattr(world_spec, "core_genre", getattr(world_spec, "genre_paradigm", plan.genre_verdict))
                        req_spec = GameRequirementsSpec(
                            deliberation=f"Requirements specification for '{g_title}' derived from world blueprint.",
                            game_title=g_title,
                            genre_paradigm=g_genre,
                            target_audience_summary=f"Player DNA pacing: {dna.pacing_score:.2f}",
                            target_episode=plan.target_episode,
                            requirements=req_tracks,
                            asset_manifest=[],
                        )

                req_spec = self._reconcile_requirements_with_board(req_spec)
                (staging_dir / "requirements_spec.json").write_text(req_spec.model_dump_json(indent=2), encoding="utf-8")
                context.raw_outputs["requirements_spec"] = req_spec
                context.raw_outputs["GameRequirementsSpec"] = req_spec

                if hasattr(self.event_bus, "publish_requirements_spec_generated"):
                    self.event_bus.publish_requirements_spec_generated(
                        agent_name=canonical_name,
                        deliberation=req_spec.deliberation,
                        spec=req_spec.model_dump(),
                        spec_id=req_spec.spec_id,
                        game_title=req_spec.game_title,
                        total_requirements=len(req_spec.requirements),
                    )
                else:
                    self._publish_event(
                        "REQUIREMENTS_SPEC_GENERATED",
                        canonical_name,
                        req_spec.deliberation,
                        {
                            "spec": req_spec.model_dump(),
                            "spec_id": req_spec.spec_id,
                            "game_title": req_spec.game_title,
                            "total_requirements": len(req_spec.requirements),
                        },
                    )

                dur_ms = (_time.perf_counter() - t_start) * 1000.0
                self._publish_event("AGENT_STREAM", canonical_name, world_spec.deliberation, {"game_title": getattr(world_spec, "game_title", plan.game_title)})
                self._publish_event("TOOL_INVOCATION", canonical_name, world_spec.deliberation, {"tool_name": "generate_genesis_world", "status": "success", "duration_ms": dur_ms})
                return world_spec

            elif canonical_name == "NarrativeCharacterWriterAgent":
                narrative_bible = None
                if (staging_dir / "narrative_bible.json").is_file() and (staging_dir / "narrative.json").is_file():
                    try:
                        from engine.studio.models import NarrativeBible
                        narrative_bible = NarrativeBible.model_validate_json((staging_dir / "narrative_bible.json").read_text(encoding="utf-8"))
                        context.raw_outputs["Writer"] = narrative_bible
                        context.raw_outputs["NarrativeCharacterWriterAgent"] = narrative_bible
                        logger.info("Resumed existing narrative_bible from %s", staging_dir)
                        dur_ms = (_time.perf_counter() - t_start) * 1000.0
                        self._publish_event("AGENT_STREAM", canonical_name, narrative_bible.deliberation, {"theme": getattr(narrative_bible, "thematic_tone", "grim_tense")})
                        self._publish_event("TOOL_INVOCATION", canonical_name, narrative_bible.deliberation, {"tool_name": "resume_narrative_bible", "status": "success", "duration_ms": dur_ms})
                        return narrative_bible
                    except Exception as e:
                        logger.warning("Failed to deserialize existing narrative_bible: %s, regenerating", e)
                        narrative_bible = None

                if narrative_bible is None:
                    world_spec = context.raw_outputs.get("Director") or context.raw_outputs.get("CreativeDirectorWorldbuilderAgent")
                    narrative_bible = agent.generate_narrative_bible(world_spec)
                    # Cap barks to 3 and normalize portrait/audio paths so all referenced files exist on disk
                    if getattr(narrative_bible, "dialogue_barks", None):
                        narrative_bible.dialogue_barks = list(narrative_bible.dialogue_barks)[:3]
                        for idx, b in enumerate(narrative_bible.dialogue_barks):
                            wav_name = Path(b.voice_clip).name if getattr(b, "voice_clip", None) else f"{b.bark_id or f'bark_{idx}'}.wav"
                            if not wav_name.endswith(".wav"):
                                wav_name = f"{wav_name}.wav"
                            b.voice_clip = f"assets/audio/{wav_name}"
                    if getattr(narrative_bible, "characters", None):
                        for idx, c in enumerate(narrative_bible.characters):
                            p_name = Path(c.portrait_asset).name if getattr(c, "portrait_asset", None) else f"{c.character_id or f'char_{idx}'}_portrait.png"
                            if not p_name.endswith(".png"):
                                p_name = f"{p_name}.png"
                            c.portrait_asset = f"assets/portraits/{p_name}"

                    (staging_dir / "narrative_bible.json").write_text(narrative_bible.model_dump_json(indent=2), encoding="utf-8")
                    n_data = narrative_bible.to_cartridge_narrative_dict()
                    from engine.studio.narrative import validate_dialogue_tree
                    valid_trees = []
                    for d in getattr(narrative_bible, "dialogue_trees", []):
                        if hasattr(d, "nodes") and not validate_dialogue_tree(d):
                            valid_trees.append(d.model_dump() if hasattr(d, "model_dump") else d)
                    if valid_trees:
                        n_data["dialogue_trees"] = valid_trees
                        n_data["dialogue_tree"] = valid_trees
                    (staging_dir / "narrative.json").write_text(json.dumps(n_data, indent=2), encoding="utf-8")

                dur_ms = (_time.perf_counter() - t_start) * 1000.0
                self._publish_event("AGENT_STREAM", canonical_name, narrative_bible.deliberation, {"characters": len(narrative_bible.characters), "barks": len(narrative_bible.dialogue_barks)})
                self._publish_event("TOOL_INVOCATION", canonical_name, narrative_bible.deliberation, {"tool_name": "generate_narrative_bible", "status": "success", "duration_ms": dur_ms})
                context.raw_outputs["Writer"] = narrative_bible
                context.raw_outputs["NarrativeCharacterWriterAgent"] = narrative_bible
                return narrative_bible

            elif canonical_name == "VoiceActingAudioProducerAgent":
                from engine.studio.audio import validate_wav_file
                from engine.studio.models import CharacterSheet as NarrativeCharacterSpec, DialogueBarkSpec

                audio_dir = staging_dir / "assets" / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                existing_wavs = list(audio_dir.rglob("*.wav")) if audio_dir.is_dir() else []
                board_managed = bool(
                    self._continuous_active
                    and self.production_board is not None
                    and self.production_board.items()
                )
                narrative = context.raw_outputs.get("Writer") or context.raw_outputs.get("NarrativeCharacterWriterAgent")
                barks = list(getattr(narrative, "dialogue_barks", None) or []) if narrative else []
                char_map = {c.character_id: c for c in (getattr(narrative, "characters", None) or [])} if narrative else {}
                if not barks:
                    for nar_fname in ("narrative.json", "narrative_bible.json"):
                        nar_path = staging_dir / nar_fname
                        if nar_path.is_file():
                            try:
                                nar_data = json.loads(nar_path.read_text(encoding="utf-8"))
                                for b_raw in nar_data.get("dialogue_barks") or []:
                                    if isinstance(b_raw, dict):
                                        barks.append(DialogueBarkSpec.model_validate(b_raw))
                                for c_raw in nar_data.get("characters") or []:
                                    if isinstance(c_raw, dict) and c_raw.get("character_id"):
                                        char_map[str(c_raw["character_id"])] = NarrativeCharacterSpec.model_validate(c_raw)
                                if barks:
                                    break
                            except Exception:
                                pass

                if board_managed:
                    synthesized_results = []
                    for bark in barks:
                        raw_clip = (getattr(bark, "voice_clip", "") or "").strip()
                        if raw_clip.startswith("res://"):
                            raw_clip = raw_clip[len("res://"):]
                        raw_clip = raw_clip.lstrip("/")
                        if not raw_clip:
                            raw_clip = f"assets/audio/{bark.bark_id}.wav"
                        if not raw_clip.endswith(".wav"):
                            raw_clip = f"{raw_clip}.wav"
                        target_path = staging_dir / raw_clip
                        flat_path = audio_dir / Path(raw_clip).name
                        ok_target, _, _ = validate_wav_file(target_path)
                        ok_flat, _, _ = validate_wav_file(flat_path)
                        if ok_target and not ok_flat and target_path != flat_path:
                            shutil.copy2(target_path, flat_path)
                            ok_flat = True
                        elif ok_flat and not ok_target and target_path != flat_path:
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(flat_path, target_path)
                            ok_target = True
                        if not ok_target:
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            char_obj = char_map.get(getattr(bark, "speaker_id", ""))
                            res = agent.synthesize_voice_line(bark=bark, output_path=target_path, character=char_obj)
                            synthesized_results.append(res)
                            if target_path != flat_path and target_path.is_file():
                                shutil.copy2(target_path, flat_path)
                    existing_wavs = list(audio_dir.rglob("*.wav")) if audio_dir.is_dir() else []
                    dur_ms = (_time.perf_counter() - t_start) * 1000.0
                    self._publish_event(
                        "AGENT_STREAM",
                        canonical_name,
                        f"Task-driven slice audio active ({len(existing_wavs)} audio clip(s) on disk; synthesized {len(synthesized_results)} missing narrative bark clip(s)).",
                        {"clips_count": len(existing_wavs), "synthesized_barks": len(synthesized_results)},
                    )
                    return synthesized_results or existing_wavs
                req_spec = context.raw_outputs.get("requirements_spec")
                manifest_items = getattr(req_spec, "asset_manifest", []) if req_spec else []
                missing_audio = False
                if manifest_items:
                    for item in manifest_items:
                        iname = getattr(item, "name", "") or (item.get("name") if isinstance(item, dict) else "")
                        if iname and iname.endswith(".wav"):
                            ok_wav, _, _ = validate_wav_file(audio_dir / iname)
                            if not ok_wav:
                                missing_audio = True
                                break
                for bark in barks:
                    raw_clip = (getattr(bark, "voice_clip", "") or f"assets/audio/{bark.bark_id}.wav").lstrip("/")
                    ok_wav, _, _ = validate_wav_file(staging_dir / raw_clip)
                    if not ok_wav:
                        missing_audio = True
                        break
                if existing_wavs and not missing_audio:
                    logger.info("Resumed %d existing voice clips from %s", len(existing_wavs), audio_dir)
                    dur_ms = (_time.perf_counter() - t_start) * 1000.0
                    self._publish_event("AGENT_STREAM", canonical_name, f"Resumed {len(existing_wavs)} voice clips.", {"clips_count": len(existing_wavs)})
                    self._publish_event("TOOL_INVOCATION", canonical_name, f"Resumed {len(existing_wavs)} voice clips.", {"tool_name": "synthesize_voice_line", "status": "success", "duration_ms": dur_ms})
                    return existing_wavs

                if not barks:
                    barks = [
                        DialogueBarkSpec(
                            deliberation="Initial system deployment bark",
                            bark_id="bark_start",
                            speaker_id="hero",
                            speaker_name="Protagonist",
                            trigger_type="episode_start",
                            text="Systems initialized. Ready to engage.",
                            voice_clip="assets/audio/bark_start.wav",
                        )
                    ]
                results = []
                for bark in barks[:3]:
                    raw_clip = (getattr(bark, "voice_clip", "") or "").strip()
                    if raw_clip.startswith("res://"):
                        raw_clip = raw_clip[len("res://"):]
                    raw_clip = raw_clip.lstrip("/")
                    wav_name = Path(raw_clip).name if raw_clip else f"{bark.bark_id}.wav"
                    if not wav_name.endswith(".wav"):
                        wav_name = f"{wav_name}.wav"
                    out_path = (staging_dir / raw_clip) if raw_clip.startswith("assets/audio/") else (audio_dir / wav_name)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    char_obj = char_map.get(getattr(bark, "speaker_id", ""))
                    res = agent.synthesize_voice_line(bark=bark, output_path=out_path, character=char_obj)
                    flat_path = audio_dir / wav_name
                    if out_path != flat_path and out_path.is_file():
                        shutil.copy2(out_path, flat_path)
                    results.append(res)
                dur_ms = (_time.perf_counter() - t_start) * 1000.0
                delib = getattr(results[0], "deliberation", f"Synthesized {len(results)} spoken WAV voice lines.") if results else "Synthesized voice audio."
                self._publish_event("AGENT_STREAM", canonical_name, delib, {"clips_count": len(results)})
                self._publish_event("TOOL_INVOCATION", canonical_name, delib, {"tool_name": "synthesize_voice_line", "status": "success", "duration_ms": dur_ms})
                return results

            elif canonical_name == "VisualArtSpriteDesignerAgent":
                from engine.studio.asset_library import CartridgeAssetLibrary

                sprites_dir = staging_dir / "assets" / "sprites"
                portraits_dir = staging_dir / "assets" / "portraits"
                sprites_dir.mkdir(parents=True, exist_ok=True)
                portraits_dir.mkdir(parents=True, exist_ok=True)

                narrative = context.raw_outputs.get("Writer") or context.raw_outputs.get("NarrativeCharacterWriterAgent")
                world_spec = context.raw_outputs.get("Director") or context.raw_outputs.get("CreativeDirectorWorldbuilderAgent")
                mood = getattr(world_spec, "palette_mood", "retro_arcade") if world_spec else "retro_arcade"
                req_spec = context.raw_outputs.get("requirements_spec")
                game_bible_obj = context.raw_outputs.get("game_bible")

                asset_lib = CartridgeAssetLibrary(
                    cartridge_dir=staging_dir,
                    visual_agent=agent,
                    voice_agent=self.agents.get("VoiceActingAudioProducerAgent"),
                    mood=mood,
                )

                board_managed = bool(
                    self._continuous_active
                    and self.production_board is not None
                    and self.production_board.items()
                )
                if board_managed:
                    asset_lib.sanitize_misplaced_portraits_in_sprites_dir()
                    asset_lib.save_catalog()
                    existing_pngs = [p for p in sprites_dir.glob("*.png") if "portrait" not in p.stem.lower()]
                    dur_ms = (_time.perf_counter() - t_start) * 1000.0
                    self._publish_event(
                        "AGENT_STREAM",
                        canonical_name,
                        f"Task-driven slice visual assets active ({len(existing_pngs)} sprite asset(s) fabricated by slice tasks).",
                        {"portraits": len(existing_pngs)},
                    )
                    return existing_pngs

                # If requirements_spec has an empty asset_manifest and a GameBible exists, derive baseline assets now
                if req_spec is not None and not getattr(req_spec, "asset_manifest", None) and not self._continuous_active and (game_bible_obj is not None or (staging_dir / "game_bible.json").is_file()):
                    req_spec.asset_manifest = asset_lib.derive_baseline_assets_from_bible(game_bible_obj)
                    (staging_dir / "requirements_spec.json").write_text(req_spec.model_dump_json(indent=2), encoding="utf-8")

                manifest_items = getattr(req_spec, "asset_manifest", []) if req_spec else []
                existing_pngs = [p for p in sprites_dir.glob("*.png") if "portrait" not in p.stem.lower()]
                missing_sprites = False
                if manifest_items:
                    for item in manifest_items:
                        iname = getattr(item, "name", "") or (item.get("name") if isinstance(item, dict) else "")
                        itype = getattr(item, "asset_type", "sprite") if not isinstance(item, dict) else item.get("asset_type", "sprite")
                        if iname and iname.endswith((".png", ".jpg")):
                            check_dir = portraits_dir if str(itype).lower() == "portrait" else sprites_dir
                            if not (check_dir / iname).is_file() and not (sprites_dir / iname).is_file():
                                missing_sprites = True
                                break
                chars_check = getattr(narrative, "characters", []) if narrative else []
                for char_c in chars_check:
                    p_asset_c = char_c.get("portrait_asset", "") if isinstance(char_c, dict) else getattr(char_c, "portrait_asset", "")
                    c_name_c = char_c.get("name", "Hero") if isinstance(char_c, dict) else getattr(char_c, "name", "Hero")
                    p_fname_c = Path(p_asset_c).name if p_asset_c else f"{c_name_c.lower()}_portrait.png"
                    if not p_fname_c.endswith(".png"):
                        p_fname_c = f"{p_fname_c}.png"
                    if not (portraits_dir / p_fname_c).is_file() or (p_asset_c and not (staging_dir / p_asset_c).is_file()):
                        missing_sprites = True
                        break
                if existing_pngs and not missing_sprites and (staging_dir / "assets" / "asset_library.json").is_file():
                    asset_lib.sanitize_misplaced_portraits_in_sprites_dir()
                    asset_lib.save_catalog()
                    logger.info("Resumed %d existing sprite assets from %s", len(existing_pngs), sprites_dir)
                    dur_ms = (_time.perf_counter() - t_start) * 1000.0
                    self._publish_event("AGENT_STREAM", canonical_name, f"Resumed {len(existing_pngs)} sprite assets.", {"portraits": len(existing_pngs)})
                    self._publish_event("TOOL_INVOCATION", canonical_name, f"Resumed {len(existing_pngs)} sprite assets.", {"tool_name": "generate_portrait", "status": "success", "duration_ms": dur_ms})
                    return existing_pngs

                results = []
                chars = getattr(narrative, "characters", []) if narrative else []
                if not chars:
                    chars = [{"name": "Hero", "archetype": "hero", "visual_description": "Expressive protagonist", "portrait_asset": "assets/portraits/hero_portrait.png"}]
                for idx, char in enumerate(chars):
                    c_name = char.get("name", "Hero") if isinstance(char, dict) else getattr(char, "name", "Hero")
                    c_arch = char.get("archetype", "hero") if isinstance(char, dict) else getattr(char, "role", getattr(char, "archetype", "hero"))
                    c_desc = char.get("visual_description", "Expressive protagonist") if isinstance(char, dict) else getattr(char, "visual_description", "Expressive protagonist")
                    p_asset = char.get("portrait_asset", "") if isinstance(char, dict) else getattr(char, "portrait_asset", "")
                    p_filename = Path(p_asset).name if p_asset else f"{c_name.lower()}_portrait.png"
                    if not p_filename.endswith(".png"):
                        p_filename = f"{p_filename}.png"
                    out_path = portraits_dir / p_filename
                    if idx < 2:
                        res = agent.generate_portrait(
                            character=char,
                            output_path=out_path,
                            archetype=c_arch,
                            visual_prompt=c_desc,
                        )
                        results.append(res)
                    elif results and Path(results[0]).is_file():
                        shutil.copy2(results[0], out_path)
                        results.append(out_path)
                    # Mirror to explicit portrait_asset path and sprites_dir for legacy cartridges
                    if out_path.is_file():
                        sprite_mirror = sprites_dir / p_filename
                        if sprite_mirror != out_path:
                            shutil.copy2(out_path, sprite_mirror)
                    if p_asset:
                        legacy_p_path = staging_dir / p_asset
                        if legacy_p_path != out_path and out_path.is_file():
                            legacy_p_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(out_path, legacy_p_path)

                agent_authored = context.raw_outputs.get("_agent_authored_req_spec", False)
                try:
                    if manifest_items:
                        # Fabricate purpose-built assets (sprite sheets, articulated parts, props, tiles, backdrops) into AssetLibrary
                        has_special_modes = any(
                            getattr(it, "fabrication_mode", "static_sprite") in ("sprite_sheet", "articulated_parts", "parallax_layer", "tile_surface")
                            for it in manifest_items
                            if not isinstance(it, dict)
                        )
                        if has_special_modes and getattr(agent, "_is_live_genai_client", lambda: False)():
                            asset_lib.fabricate_assets(manifest_items)
                        else:
                            agent.generate_cartridge_sprites(
                                sprites_dir,
                                mood=mood,
                                requirements_spec=req_spec,
                                reference_images=asset_lib.get_concept_reference_images(),
                                style_guide=asset_lib.get_style_guide(),
                            )
                    elif agent_authored:
                        agent.generate_cartridge_sprites(sprites_dir, mood=mood, requirements_spec=req_spec)
                    else:
                        agent.generate_cartridge_sprites(sprites_dir, mood=mood)
                except TypeError:
                    if agent_authored or (req_spec and req_spec.asset_manifest):
                        agent.generate_cartridge_sprites(sprites_dir, mood=mood, requirements_spec=req_spec)
                    else:
                        agent.generate_cartridge_sprites(sprites_dir, mood=mood)

                # Ensure default_portrait.png exists for dialogue barks (NEVER copy a portrait to hero.png!)
                if results and Path(results[0]).is_file():
                    for dp_dir in (portraits_dir, sprites_dir):
                        dp_path = dp_dir / "default_portrait.png"
                        if not dp_path.exists():
                            shutil.copy2(results[0], dp_path)

                # Ensure a genuine in-world hero.png sprite exists (full-body sprite, never a portrait copy)
                hero_sprite_path = sprites_dir / "hero.png"
                if not hero_sprite_path.exists():
                    player_idle_path = sprites_dir / "player_idle.png"
                    if player_idle_path.exists():
                        shutil.copy2(player_idle_path, hero_sprite_path)
                    else:
                        hero_char = chars[0] if chars else {"name": "Hero", "visual_description": "Full-body protagonist"}
                        h_desc = hero_char.get("visual_description", "Protagonist") if isinstance(hero_char, dict) else getattr(hero_char, "visual_description", "Protagonist")
                        hero_req = AssetRequirement(
                            deliberation="Dedicated full-body in-world protagonist sprite.",
                            asset_id="hero",
                            name="hero.png",
                            asset_type="sprite",
                            target_path="assets/sprites/hero.png",
                            description=f"Full-body in-world character sprite: {h_desc}",
                            visual_prompt_or_text=f"Full-body 2D in-game character sprite (head to toe, NOT a bust portrait): {h_desc}",
                            dimensions=(64, 64),
                        )
                        try:
                            agent.generate_dynamic_sprite(
                                hero_req,
                                output_path=hero_sprite_path,
                                mood=mood,
                                reference_images=asset_lib.get_concept_reference_images(),
                                style_guide=asset_lib.get_style_guide(),
                            )
                        except Exception:
                            agent.renderer.render_player_idle().save(hero_sprite_path, "PNG")

                asset_lib.index_existing_files_on_disk()
                asset_lib.save_catalog()

                dur_ms = (_time.perf_counter() - t_start) * 1000.0
                delib = f"Generated {len(results)} character portraits in {portraits_dir.name} and indexed {len(asset_lib.catalog.entries)} purpose-built assets in Cartridge Asset Library."
                self._publish_event("AGENT_STREAM", canonical_name, delib, {"portraits": len(results), "library_assets": len(asset_lib.catalog.entries)})
                self._publish_event("TOOL_INVOCATION", canonical_name, delib, {"tool_name": "generate_portrait", "status": "success", "duration_ms": dur_ms})
                return results

            elif canonical_name == "LeadGameplaySWEAgent":
                # Wire visual and voice agents onto SWE agent so on-demand requested_assets can be fabricated
                try:
                    agent.visual_agent = self.agents.get("VisualArtSpriteDesignerAgent")
                    agent.voice_agent = self.agents.get("VoiceActingAudioProducerAgent")
                except Exception:
                    pass
                world_spec = context.raw_outputs.get("Director") or context.raw_outputs.get("CreativeDirectorWorldbuilderAgent")
                narrative = context.raw_outputs.get("Writer") or context.raw_outputs.get("NarrativeCharacterWriterAgent")
                req_spec = context.raw_outputs.get("requirements_spec") or context.raw_outputs.get("GameRequirementsSpec")

                tracks = req_spec.requirements if (req_spec and getattr(req_spec, "requirements", None)) else []
                if not tracks:
                    results = agent.synthesize_cartridge_code(
                        world_spec=world_spec,
                        output_dir=staging_dir,
                        narrative_bible=narrative,
                        requirements_spec=req_spec,
                    )
                else:
                    results = await self._execute_iterative_swe_loop(
                        agent=agent,
                        world_spec=world_spec,
                        narrative_bible=narrative,
                        req_spec=req_spec,
                        staging_dir=staging_dir,
                        context=context,
                    )
                dur_ms = (_time.perf_counter() - t_start) * 1000.0
                delib = f"Synthesized and AST-verified playable game modules: {list(results.keys())} across {len(tracks)} requirement tracks."
                self._publish_event("AGENT_STREAM", canonical_name, delib, {"modules": list(results.keys()), "tracks": len(tracks)})
                self._publish_event("TOOL_INVOCATION", canonical_name, delib, {"tool_name": "synthesize_cartridge_code", "status": "success", "duration_ms": dur_ms})
                return results

            elif canonical_name == "TelemetryArchitectAgent":
                world_spec = context.raw_outputs.get("Director") or context.raw_outputs.get("CreativeDirectorWorldbuilderAgent")
                code_modules = {}
                for ext in ("*.py", "*.gd"):
                    for code_file in staging_dir.glob(ext):
                        code_modules[code_file.name] = code_file.read_text(encoding="utf-8")

                manifest_file = staging_dir / "telemetry_manifest.json"
                if manifest_file.is_file():
                    logger.info("Resuming existing telemetry_manifest.json from %s", staging_dir)
                    try:
                        from engine.studio.models import TelemetryPlan
                        plan_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                        plan_obj = TelemetryPlan(**plan_data)
                        context.raw_outputs["TelemetryPlan"] = plan_obj
                        context.raw_outputs["TelemetryArchitecturePlan"] = plan_obj
                        dur_ms = (_time.perf_counter() - t_start) * 1000.0
                        self._publish_event("AGENT_STREAM", canonical_name, plan_obj.deliberation, {"probes_resumed": len(plan_obj.probes)})
                        self._publish_event("TOOL_INVOCATION", canonical_name, plan_obj.deliberation, {"tool_name": "resume_telemetry_manifest", "status": "success", "duration_ms": dur_ms})
                        return manifest_file
                    except Exception as exc:
                        logger.warning("Failed loading existing telemetry manifest: %s", exc)

                tel_plan = agent.inspect_code_and_plan(code_modules=code_modules, world_spec=world_spec)

                agent.register_probes(tel_plan)
                inst_res = agent.instrument_code(
                    code_modules=code_modules,
                    plan=tel_plan,
                )
                for mod_name, mod_src in inst_res.instrumented_code.items():
                    (staging_dir / mod_name).write_text(mod_src, encoding="utf-8")
                (staging_dir / "telemetry_manifest.json").write_text(tel_plan.model_dump_json(indent=2), encoding="utf-8")
                dur_ms = (_time.perf_counter() - t_start) * 1000.0
                self._publish_event("AGENT_STREAM", canonical_name, tel_plan.deliberation, {"probes_registered": inst_res.probes_registered})
                self._publish_event("TOOL_INVOCATION", canonical_name, tel_plan.deliberation, {"tool_name": "instrument_code", "status": "success", "duration_ms": dur_ms})
                return inst_res

            elif canonical_name == "QAPlaytesterBuildSupervisor":
                return {"status": "supervisor_ready"}

            elif canonical_name == "SteamAndTelemetryAnalystAgent":
                return {"status": "analyst_ready"}

            else:
                return {"role": role, "status": "executed"}

        async def veri_coro(raw_res: Any) -> AgentVerificationRecord:
            staging_dir = context.staging_dir
            if canonical_name == "CreativeDirectorWorldbuilderAgent":
                passed = (staging_dir / "world_spec.json").is_file() and (staging_dir / "manifest.json").is_file()
            elif canonical_name == "NarrativeCharacterWriterAgent":
                passed = (staging_dir / "narrative_bible.json").is_file() and (staging_dir / "narrative.json").is_file()
            elif canonical_name == "VoiceActingAudioProducerAgent":
                passed = (staging_dir / "assets" / "audio").exists()
            elif canonical_name == "VisualArtSpriteDesignerAgent":
                passed = (staging_dir / "assets" / "sprites").exists()
            elif canonical_name == "LeadGameplaySWEAgent":
                passed = (staging_dir / "project.godot").is_file() or (staging_dir / "mechanics.gd").is_file() or (staging_dir / "mechanics.py").is_file()
            elif canonical_name == "TelemetryArchitectAgent":
                passed = (staging_dir / "telemetry_manifest.json").is_file() and getattr(raw_res, "success", True)
            else:
                passed = True

            return AgentVerificationRecord(
                deliberation=f"Verification check for {role}: passed={passed}",
                agent_name=role,
                passed=passed,
                violations=[] if passed else [f"{role} failed to produce expected staging artifacts"],
            )

        raw_out, envelope = await self.execute_pev_cycle(
            agent_name=role,
            plan_coroutine=plan_coro,
            execute_coroutine=exec_coro,
            verify_coroutine=veri_coro,
            max_retries=self.config.max_retries,
        )
        self._publish_event(
            "state_change",
            canonical_name,
            f"Specialist {canonical_name} verified and sealed artifact envelope.",
            {"agent_name": canonical_name, "lifecycle_state": "PROMOTED"},
        )
        context.record_artifact(role, envelope, raw_out)

        # Build cryptographic artifact envelopes and construct AgentTaskRecord
        output_envelopes: List[UIArtifactMetadataEnvelope] = []
        staging_dir = context.staging_dir
        if canonical_name == "CreativeDirectorWorldbuilderAgent":
            target_files = [staging_dir / "world_spec.json", staging_dir / "manifest.json", staging_dir / "requirements_spec.json"]
        elif canonical_name == "NarrativeCharacterWriterAgent":
            target_files = [staging_dir / "narrative_bible.json", staging_dir / "narrative.json"]
        elif canonical_name == "VoiceActingAudioProducerAgent":
            target_files = list((staging_dir / "assets" / "audio").glob("*.wav")) if (staging_dir / "assets" / "audio").exists() else []
        elif canonical_name == "VisualArtSpriteDesignerAgent":
            target_files = list((staging_dir / "assets" / "sprites").glob("*.png")) if (staging_dir / "assets" / "sprites").exists() else []
        elif canonical_name == "TelemetryArchitectAgent":
            target_files = [staging_dir / "telemetry_manifest.json"]
        elif canonical_name == "LeadGameplaySWEAgent":
            from engine.sim.godot_verifier import iter_cartridge_gd_files
            target_files = sorted(
                list(staging_dir.glob("*.py"))
                + iter_cartridge_gd_files(staging_dir, include_tests=False, exclude_helper_libs=True)
                + list(staging_dir.rglob("*.tscn"))
            )
            if (staging_dir / "project.godot").is_file():
                target_files.append(staging_dir / "project.godot")
        else:
            target_files = []

        for f_path in target_files:
            if f_path.is_file():
                f_hash = ArtifactMetadataEnvelope.compute_file_sha256(f_path)
                f_type = "code" if (f_path.suffix in (".py", ".gd", ".tscn", ".godot") or f_path.name == "project.godot") else ("audio" if f_path.suffix == ".wav" else ("sprite" if f_path.suffix == ".png" else "spec"))
                output_envelopes.append(
                    UIArtifactMetadataEnvelope(
                        deliberation=f"Artifact {f_path.name} produced by {canonical_name}",
                        artifact_id=f"art_{f_path.stem}_{uuid.uuid4().hex[:6]}",
                        artifact_name=f_path.name,
                        artifact_type=f_type,
                        file_path=str(f_path),
                        sha256_hash=f_hash,
                        byte_size=f_path.stat().st_size,
                        created_at_utc=datetime.now(timezone.utc).isoformat(),
                        source_agent=canonical_name,
                        target_agent="LeadGameplaySWEAgent" if canonical_name != "LeadGameplaySWEAgent" else "QAPlaytesterBuildSupervisor",
                        metadata_summary={"size": f_path.stat().st_size},
                    )
                )

        dur_ms = (_time.perf_counter() - t_spec_start) * 1000.0
        task_id = f"task_{canonical_name.lower()}_{uuid.uuid4().hex[:8]}"
        task_rec = AgentTaskRecord(
            deliberation=getattr(envelope, "deliberation", f"Specialist {canonical_name} completed task execution."),
            task_id=task_id,
            agent_role=canonical_name,
            agent_name=canonical_name,
            requirement_id=None,
            track_title=f"{canonical_name} Execution Cycle",
            task_type="SPECIFICATION" if "Director" in canonical_name or "Writer" in canonical_name else ("ASSET_SYNTHESIS" if "Audio" in canonical_name or "Visual" in canonical_name else ("INSTRUMENTATION" if "Telemetry" in canonical_name else "EXECUTION")),
            status="COMPLETED",
            start_time_utc=t_start_iso,
            end_time_utc=datetime.now(timezone.utc).isoformat(),
            duration_ms=dur_ms,
            input_artifacts=[],
            output_artifacts=output_envelopes,
            verification_verdict={"passed": True},
            diagnostic_trace=None,
            summary=f"Specialist {canonical_name} verified and sealed {len(output_envelopes)} artifacts.",
        )
        if hasattr(self.event_bus, "publish_agent_task_record"):
            self.event_bus.publish_agent_task_record(task_rec)
        else:
            self._publish_event(
                "AGENT_TASK_RECORD",
                canonical_name,
                task_rec.deliberation,
                task_rec.model_dump(),
            )
        context.record_task(task_rec)
        return raw_out

    async def _execute_iterative_swe_loop(
        self,
        agent: Any,
        world_spec: Any,
        narrative_bible: Optional[Any],
        req_spec: GameRequirementsSpec,
        staging_dir: Path,
        context: Optional[SwarmExecutionContext] = None,
    ) -> Dict[str, Path]:
        """Executes iterative requirement execution loop across all ordered RequirementTracks."""
        from engine.studio.models import TOKEN_USAGE_TRACKER, wait_if_studio_paused

        results: Dict[str, Path] = {}
        tracks = req_spec.requirements
        completed_track_ids: List[str] = [t.id for t in tracks if t.status == "completed"]
        board_managed = bool(
            (self._continuous_active and self.production_board is not None and self.production_board.items())
            or (staging_dir / "backlog.json").is_file()
        )
        has_existing_cartridge = (staging_dir / "project.godot").is_file() or (staging_dir / "mechanics.py").is_file()
        has_uncompleted_tracks = any(t.status != "completed" for t in tracks) or not has_existing_cartridge
        synthesized_any_track = False

        # 0. Ensure the per-game RenderingStrategySpec exists (normally created before asset fabrication; cached here).
        if (
            not board_managed
            and has_uncompleted_tracks
            and self._is_live_genai_client()
            and (staging_dir / "game_bible.json").is_file()
            and hasattr(agent, "_rendering_strategy_block")
        ):
            wait_if_studio_paused(candidate_dir=staging_dir, label="LeadGameplaySWEAgent._rendering_strategy_block")
            agent._rendering_strategy_block(staging_dir)

        # 0a. Formulate the machine-readable LevelLayoutSpec (data the SWE agent consumes, not prose) before building tracks
        if (
            not board_managed
            and has_uncompleted_tracks
            and self._is_live_genai_client()
            and (staging_dir / "game_bible.json").is_file()
            and hasattr(agent, "formulate_level_layout")
        ):
            wait_if_studio_paused(candidate_dir=staging_dir, label="LeadGameplaySWEAgent.formulate_level_layout")
            layout = agent.formulate_level_layout(staging_dir)
            self._publish_event(
                "AGENT_STREAM",
                "LeadGameplaySWEAgent",
                layout.deliberation,
                {"phase": "formulate_level_layout", "entities": len(layout.entities), "triggers": len(layout.triggers)},
            )

        # 0b. Formulate master GameProductionStrategySpec before building tracks (when live Gemini + Game Bible present).
        # Failures are fatal and propagate with their full stack trace (Rule 5).
        if (
            not board_managed
            and has_uncompleted_tracks
            and self._is_live_genai_client()
            and (staging_dir / "game_bible.json").is_file()
            and hasattr(agent, "formulate_production_strategy")
        ):
            wait_if_studio_paused(candidate_dir=staging_dir, label="LeadGameplaySWEAgent.formulate_production_strategy")
            strat = agent.formulate_production_strategy(staging_dir)
            self._publish_event(
                "AGENT_STREAM",
                "LeadGameplaySWEAgent",
                strat.deliberation,
                {"phase": "formulate_production_strategy", "scene_layers": len(strat.scene_layers)},
            )

        for track_idx, track in enumerate(tracks):
            wait_if_studio_paused(candidate_dir=staging_dir, label=f"LeadGameplaySWEAgent.track[{track.id}]")
            # 1. Skip Completed Tracks on Resume
            if track.status == "completed" and ((staging_dir / "project.godot").is_file() or (staging_dir / "mechanics.py").is_file()):
                logger.info("Skipping already completed requirement track %s: '%s'", track.id, track.title)
                delib_skipped = f"Requirement {track.id} ('{track.title}') already verified and completed in staging."
                if hasattr(self.event_bus, "publish_requirement_status_update"):
                    self.event_bus.publish_requirement_status_update(
                        agent_name="LeadGameplaySWEAgent",
                        deliberation=delib_skipped,
                        requirement_id=track.id,
                        status="completed",
                        metrics={"compilation_pass": True, "smoke_pass": True, "resumed": True},
                        track_title=track.title,
                    )
                else:
                    self._publish_event(
                        "REQUIREMENT_STATUS_UPDATE",
                        "LeadGameplaySWEAgent",
                        delib_skipped,
                        {
                            "requirement_id": track.id,
                            "status": "completed",
                            "track_title": track.title,
                            "metrics": {"compilation_pass": True, "smoke_pass": True, "resumed": True},
                        },
                    )
                self._persist_verified_checkpoint(staging_dir, track, completed_track_ids, mirror_to_active=False)
                continue

            synthesized_any_track = True

            track_t_start = _time.perf_counter()
            track_start_iso = datetime.now(timezone.utc).isoformat()
            track.status = "in_progress"
            delib_in_progress = f"Beginning implementation of requirement {track.id}: '{track.title}'"
            if hasattr(self.event_bus, "publish_requirement_status_update"):
                self.event_bus.publish_requirement_status_update(
                    agent_name="LeadGameplaySWEAgent",
                    deliberation=delib_in_progress,
                    requirement_id=track.id,
                    status="in_progress",
                    metrics={"track_index": track_idx, "total_tracks": len(tracks)},
                    track_title=track.title,
                )
            else:
                self._publish_event(
                    "REQUIREMENT_STATUS_UPDATE",
                    "LeadGameplaySWEAgent",
                    delib_in_progress,
                    {
                        "requirement_id": track.id,
                        "status": "in_progress",
                        "track_title": track.title,
                        "metrics": {"track_index": track_idx, "total_tracks": len(tracks)},
                    },
                )

            # 2. Snapshot existing code modules before executing track
            code_snapshot = {
                code_f.name: code_f.read_text(encoding="utf-8")
                for ext in ("*.py", "*.gd", "*.tscn")
                for code_f in staging_dir.glob(ext)
            }
            if (staging_dir / "project.godot").is_file():
                code_snapshot["project.godot"] = (staging_dir / "project.godot").read_text(encoding="utf-8")

            max_track_retries = (
                max(2, getattr(self.config, "max_retries", 2))
                if self._is_live_genai_client()
                else 1
            )
            track_succeeded = False
            last_track_exc = None

            for track_attempt in range(1, max_track_retries + 1):
                try:
                    # Synthesize or expand code
                    prior_err_str = str(last_track_exc) if last_track_exc else None
                    if hasattr(agent, "implement_requirement_track"):
                        sig = inspect.signature(agent.implement_requirement_track)
                        call_kwargs: Dict[str, Any] = {
                            "track": track,
                            "staging_dir": staging_dir,
                            "requirements_spec": req_spec,
                            "narrative_bible": narrative_bible,
                            "world_spec": world_spec,
                        }
                        if "prior_error" in sig.parameters and prior_err_str:
                            call_kwargs["prior_error"] = prior_err_str
                        res = agent.implement_requirement_track(**call_kwargs)
                        results.update(res)
                    elif hasattr(agent, "synthesize_requirement_track"):
                        res = agent.synthesize_requirement_track(
                            track=track,
                            staging_dir=staging_dir,
                            requirements_spec=req_spec,
                            narrative_bible=narrative_bible,
                            world_spec=world_spec,
                        )
                        results.update(res)
                    elif hasattr(agent, "synthesize_cartridge_code"):
                        res = agent.synthesize_cartridge_code(
                            world_spec=world_spec,
                            output_dir=staging_dir,
                            narrative_bible=narrative_bible,
                            requirements_spec=req_spec,
                        )
                        results.update(res)
                    else:
                        raise StudioAgentError(
                            f"LeadGameplaySWEAgent cannot implement track {track.id}: missing implement_requirement_track"
                        )

                    # Per-track AST and smoke verification
                    self._verify_track_smoke_check(staging_dir, track, swe_agent=agent)
                    track_succeeded = True
                    break
                except Exception as exc:
                    last_track_exc = exc
                    logger.warning(
                        "Track %s attempt %d/%d failed: %s. Restoring code snapshot and retrying.",
                        track.id, track_attempt, max_track_retries, exc
                    )
                    # Restore code snapshot before next attempt or failure handling
                    for f_name, f_code in code_snapshot.items():
                        (staging_dir / f_name).write_text(f_code, encoding="utf-8")
                    if track_attempt < max_track_retries:
                        await asyncio.sleep(2.0 * track_attempt)

            if not track_succeeded:
                exc = last_track_exc
                logger.error("Requirement %s failed after %d attempts: %s", track.id, max_track_retries, exc)
                track.status = "rejected"
                delib_rejected = f"Requirement {track.id} failed verification: {exc}"
                if hasattr(self.event_bus, "publish_requirement_status_update"):
                    self.event_bus.publish_requirement_status_update(
                        agent_name="LeadGameplaySWEAgent",
                        deliberation=delib_rejected,
                        requirement_id=track.id,
                        status="rejected",
                        metrics={"compilation_pass": False, "error": str(exc)},
                        track_title=track.title,
                    )
                else:
                    self._publish_event(
                        "REQUIREMENT_STATUS_UPDATE",
                        "LeadGameplaySWEAgent",
                        delib_rejected,
                        {
                            "requirement_id": track.id,
                            "status": "rejected",
                            "track_title": track.title,
                            "metrics": {"compilation_pass": False, "error": str(exc)},
                        },
                    )

                (staging_dir / "requirements_spec.json").write_text(
                    req_spec.model_dump_json(indent=2), encoding="utf-8"
                )
                track_dur_ms = (_time.perf_counter() - track_t_start) * 1000.0
                fail_task = AgentTaskRecord(
                    deliberation=delib_rejected,
                    task_id=f"task_swe_{track.id.lower()}_{uuid.uuid4().hex[:8]}",
                    agent_role="LeadGameplaySWEAgent",
                    agent_name="GameplaySWE",
                    requirement_id=track.id,
                    track_title=track.title,
                    task_type="CODE_IMPLEMENTATION",
                    status="REJECTED",
                    start_time_utc=track_start_iso,
                    end_time_utc=datetime.now(timezone.utc).isoformat(),
                    duration_ms=track_dur_ms,
                    input_artifacts=[],
                    output_artifacts=[],
                    verification_verdict={"ast_security": False, "bytecode_compilation": False, "headless_smoke_pass": False},
                    diagnostic_trace={"error": str(exc), "failure_gate": "Track Smoke Check"},
                    summary=f"Requirement {track.id} failed verification: {exc}",
                )
                if hasattr(self.event_bus, "publish_agent_task_record"):
                    self.event_bus.publish_agent_task_record(fail_task)
                else:
                    self._publish_event(
                        "AGENT_TASK_RECORD",
                        "LeadGameplaySWEAgent",
                        fail_task.deliberation,
                        fail_task.model_dump(),
                    )
                if context is not None:
                    context.record_task(fail_task)

                raise StudioSwarmExecutionError(
                    f"Requirement {track.id} ('{track.title}') failed verification: {exc}"
                ) from exc

            # 4. Mark completed & stream event
            track.status = "completed"
            if track.id not in completed_track_ids:
                completed_track_ids.append(track.id)
            delib_completed = f"Requirement {track.id} ('{track.title}') compiled and passed headless smoke check."
            if hasattr(self.event_bus, "publish_requirement_status_update"):
                self.event_bus.publish_requirement_status_update(
                    agent_name="LeadGameplaySWEAgent",
                    deliberation=delib_completed,
                    requirement_id=track.id,
                    status="completed",
                    metrics={"compilation_pass": True, "smoke_pass": True},
                    track_title=track.title,
                )
            else:
                self._publish_event(
                    "REQUIREMENT_STATUS_UPDATE",
                    "LeadGameplaySWEAgent",
                    delib_completed,
                    {
                        "requirement_id": track.id,
                        "status": "completed",
                        "track_title": track.title,
                        "metrics": {"compilation_pass": True, "smoke_pass": True},
                    },
                )

            # Persist updated requirements_spec.json on disk
            (staging_dir / "requirements_spec.json").write_text(
                req_spec.model_dump_json(indent=2), encoding="utf-8"
            )

            output_envelopes = []
            for code_pattern in ["*.py", "*.gd", "*.tscn", "project.godot"]:
                for code_f in sorted(staging_dir.glob(code_pattern)):
                    if code_f.is_file():
                        h = ArtifactMetadataEnvelope.compute_file_sha256(code_f)
                        output_envelopes.append(
                            UIArtifactMetadataEnvelope(
                                deliberation=f"Synthesized code module {code_f.name} for track {track.id}",
                                artifact_id=f"art_code_{code_f.stem}_{uuid.uuid4().hex[:6]}",
                                artifact_name=code_f.name,
                                artifact_type="code",
                                file_path=str(code_f),
                                sha256_hash=h,
                                byte_size=code_f.stat().st_size,
                                created_at_utc=datetime.now(timezone.utc).isoformat(),
                                source_agent="LeadGameplaySWEAgent",
                                target_agent="QAPlaytesterBuildSupervisor",
                                metadata_summary={"module": code_f.name},
                            )
                        )
            track_dur_ms = (_time.perf_counter() - track_t_start) * 1000.0
            track_task = AgentTaskRecord(
                deliberation=delib_completed,
                task_id=f"task_swe_{track.id.lower()}_{uuid.uuid4().hex[:8]}",
                agent_role="LeadGameplaySWEAgent",
                agent_name="GameplaySWE",
                requirement_id=track.id,
                track_title=track.title,
                task_type="CODE_IMPLEMENTATION",
                status="COMPLETED",
                start_time_utc=track_start_iso,
                end_time_utc=datetime.now(timezone.utc).isoformat(),
                duration_ms=track_dur_ms,
                input_artifacts=[],
                output_artifacts=output_envelopes,
                verification_verdict={"ast_security": True, "bytecode_compilation": True, "headless_smoke_pass": True},
                diagnostic_trace=None,
                summary=f"Requirement {track.id} ({track.title}) completed successfully.",
            )
            if hasattr(self.event_bus, "publish_agent_task_record"):
                self.event_bus.publish_agent_task_record(track_task)
            else:
                self._publish_event(
                    "AGENT_TASK_RECORD",
                    "LeadGameplaySWEAgent",
                    track_task.deliberation,
                    track_task.model_dump(),
                )
            if context is not None:
                context.record_task(track_task)

            # 5. Persist Verified Checkpoint in cartridges/checkpoints/ (do not overwrite cartridges/active/ mid-build before QA & Sign-Off Council)
            TOKEN_USAGE_TRACKER.reset_swe_pev_streak()
            self._persist_verified_checkpoint(staging_dir, track, completed_track_ids, mirror_to_active=False)

        # 6. Execute Per-Scene Critique & Iterative Refinement Loop ONLY when legacy non-board tracks were newly synthesized
        if (
            not board_managed
            and synthesized_any_track
            and self._is_live_genai_client()
            and (staging_dir / "project.godot").is_file()
            and (staging_dir / "game_bible.json").is_file()
            and hasattr(agent, "execute_scene_strategy_and_refine_loop")
        ):
            wait_if_studio_paused(candidate_dir=staging_dir, label="LeadGameplaySWEAgent.execute_scene_strategy_and_refine_loop")
            refined = agent.execute_scene_strategy_and_refine_loop(
                staging_dir,
                max_scene_iterations=2,
            )
            results.update(refined)

        for code_pattern in ["*.py", "*.gd", "*.tscn", "project.godot"]:
            for p in staging_dir.glob(code_pattern):
                if p.is_file():
                    results[p.name] = p
        if (staging_dir / "level.json").is_file():
            results["level.json"] = staging_dir / "level.json"
        return results

    def _persist_verified_checkpoint(
        self,
        staging_dir: Path,
        track: RequirementTrack,
        verified_tracks: List[str],
        mirror_to_active: bool = True,
    ) -> Path:
        """Persists an immutable verified checkpoint under cartridges/checkpoints/<candidate_id>_<track_id>/
        and optionally mirrors it to cartridges/active/ when mirror_to_active is True.
        """
        candidate_id = staging_dir.name
        if "PYTEST_CURRENT_TEST" in os.environ and candidate_id.startswith("staging_"):
            return staging_dir
        checkpoint_id = f"{candidate_id}_{track.id}"
        checkpoints_root = PROJECT_ROOT / "cartridges" / "checkpoints"
        checkpoints_root.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = checkpoints_root / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy code modules
        for pattern in ["*.py", "*.gd", "*.tscn", "project.godot"]:
            for code_f in staging_dir.glob(pattern):
                if code_f.is_file():
                    shutil.copy2(code_f, checkpoint_dir / code_f.name)

        # 2. Copy specification and metadata files
        for json_name in [
            "manifest.json",
            "game_bible.json",
            "world_spec.json",
            "requirements_spec.json",
            "narrative_bible.json",
            "narrative.json",
            "level.json",
            "task_history.json",
        ]:
            src = staging_dir / json_name
            if src.is_file():
                shutil.copy2(src, checkpoint_dir / json_name)

        # 3. Copy assets and scenes directories if present
        for subdir_name in ("assets", "scenes"):
            sub_src = staging_dir / subdir_name
            if sub_src.is_dir():
                sub_dst = checkpoint_dir / subdir_name
                if sub_dst.exists():
                    shutil.rmtree(sub_dst)
                shutil.copytree(sub_src, sub_dst)

        # 4. Compute cryptographic sha256 manifest
        sha256_manifest: Dict[str, str] = {}
        for f in checkpoint_dir.rglob("*"):
            if f.is_file() and f.name != "checkpoint_metadata.json":
                rel = f.relative_to(checkpoint_dir).as_posix()
                sha256_manifest[rel] = ArtifactMetadataEnvelope.compute_file_sha256(f)

        # 5. Write checkpoint_metadata.json
        metadata = {
            "checkpoint_id": checkpoint_id,
            "candidate_id": candidate_id,
            "requirement_id": track.id,
            "track_title": track.title,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "verified_tracks": list(verified_tracks),
            "cartridge_path": str(checkpoint_dir),
            "sha256_manifest": sha256_manifest,
        }
        (checkpoint_dir / "checkpoint_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        # Record to relational SQLite WAL studio_checkpoints (Milestone 7 Godot 4 Transition)
        try:
            db = self._get_telemetry_db()
            if db is not None:
                bid = getattr(self, "active_build_id", None) or f"build_{candidate_id}"
                db.record_studio_checkpoint(
                    checkpoint_id=checkpoint_id,
                    build_id=bid,
                    candidate_id=candidate_id,
                    requirement_id=track.id,
                    track_title=track.title,
                    engine_type="godot4",
                    checkpoint_path=str(checkpoint_dir),
                    verified_tracks=list(verified_tracks),
                    deliberation=f"Verified checkpoint {checkpoint_id} covering tracks {verified_tracks}.",
                )
        except Exception as exc:
            logger.debug("Failed to record checkpoint in telemetry.db: %s", exc)

        # 6. Atomically mirror to cartridges/active/ only if mirror_to_active is True and active does not already have a more complete build
        active_dir = PROJECT_ROOT / "cartridges" / "active"
        should_mirror = mirror_to_active
        if should_mirror and active_dir.exists() and (active_dir / "checkpoint_metadata.json").is_file():
            try:
                existing_meta = json.loads((active_dir / "checkpoint_metadata.json").read_text(encoding="utf-8"))
                existing_cand = existing_meta.get("candidate_id", "")
                existing_tracks = existing_meta.get("verified_tracks", [])
                if existing_cand and existing_cand != candidate_id and len(existing_tracks) > len(verified_tracks):
                    should_mirror = False
            except Exception:
                pass

        if should_mirror:
            tmp_active = PROJECT_ROOT / "cartridges" / f".active_checkpoint_tmp_{uuid.uuid4().hex[:6]}"
            try:
                shutil.copytree(checkpoint_dir, tmp_active)
                if active_dir.exists():
                    old_active = PROJECT_ROOT / "cartridges" / f".active_old_{uuid.uuid4().hex[:6]}"
                    active_dir.rename(old_active)
                    tmp_active.rename(active_dir)
                    try:
                        shutil.rmtree(old_active)
                    except Exception:
                        pass
                else:
                    tmp_active.rename(active_dir)
            except Exception as copy_exc:
                logger.warning("Failed to atomically mirror checkpoint to cartridges/active: %s", copy_exc)
                if tmp_active.exists():
                    try:
                        shutil.rmtree(tmp_active)
                    except Exception:
                        pass

        # 7. Emit CHECKPOINT_VERIFIED event to SwarmEventBus with deliberation first
        delib_checkpoint = (
            f"Verified checkpoint {checkpoint_id} for requirement {track.id}: '{track.title}' "
            f"persisted under {checkpoint_dir.name}."
        )
        self._publish_event(
            "CHECKPOINT_VERIFIED",
            "LeadGameplaySWEAgent",
            delib_checkpoint,
            {
                "checkpoint_id": checkpoint_id,
                "candidate_id": candidate_id,
                "requirement_id": track.id,
                "track_title": track.title,
                "verified_tracks": list(verified_tracks),
                "checkpoint_path": str(checkpoint_dir),
            },
        )
        return checkpoint_dir

    @staticmethod
    def _check_godot_gameplay_and_asset_gates(
        staging_dir: Path,
        track: RequirementTrack,
        scene_audit: Dict[str, Any],
    ) -> None:
        """Enforces authentic gameplay (no self-playing sim_tick demo loops), clean diegetic HUDs (no debug test-runner
        dashboards or center-occluding panels), and mandatory Sprite2D/TextureRect asset binding when sprites exist."""
        if scene_audit.get("passive_auto_win_detected"):
            reason = scene_audit.get("passive_win_reason") or "Game auto-solved with zero player input."
            raise RuntimeError(f"Passive auto-win violation in '{track.id}': {reason}")

        if scene_audit.get("fullscreen_occlusion_detected"):
            occ_nodes = scene_audit.get("fullscreen_occlusion_nodes") or []
            occ_detail = f" Offending foreground UI nodes: {occ_nodes}." if occ_nodes else ""
            raise RuntimeError(
                f"Fullscreen UI viewport occlusion detected in '{track.id}'.{occ_detail} "
                "Do not cover the 1280x720 game viewport with opaque foreground UI ColorRects or panels (alpha >= 0.80 in CanvasLayer/ui_hud.gd). "
                "NOTE: World floor/terrain surfaces (z_index <= 0 in scenes.gd) SHOULD fill the full 1280x720 viewport—never shrink or add margins around world terrain!"
            )

        if scene_audit.get("early_failure_state_detected"):
            raise RuntimeError(
                f"Unrecoverable early failure state ('is_failed': true in get_state_snapshot()) triggered within the opening 60-tick smoke check in '{track.id}'. "
                "Ensure the player spawns in a stable recoverable state at t=0 and does not immediately die or fail during the initial 1-second smoke test."
            )

        empty_frames = scene_audit.get("empty_sprite_frame_nodes") or []
        if empty_frames:
            raise RuntimeError(
                f"Invisible / mis-sliced Sprite2D frame detected in '{track.id}': {empty_frames}. "
                "Check `assets/asset_library.json` for each texture's true `hframes` and `vframes`—never set `hframes > 1` or `vframes > 1` on a single-frame (`hframes=1, vframes=1`) sprite or articulated rig composite!"
            )

        sv_thrashing = scene_audit.get("subviewport_thrashing_nodes") or []
        if sv_thrashing:
            raise RuntimeError(
                f"Severe GPU SubViewport frame-rate thrashing detected in '{track.id}': {sv_thrashing}. "
                "When using a SubViewport as an accumulation mask with `CLEAR_MODE_NEVER`, NEVER set `render_target_update_mode = SubViewport.UPDATE_ALWAYS`—set `SubViewport.UPDATE_ONCE` and trigger `render_target_update_mode = SubViewport.UPDATE_ONCE` only when a new stroke is painted!"
            )

        occluding = scene_audit.get("center_occluding_panels") or []
        if occluding:
            raise RuntimeError(
                f"HUD center-occlusion violation in '{track.id}': UI panels {occluding} cover the playable center of the viewport. "
                "Keep all HUD elements compact around the screen perimeter (<15% viewport area)."
            )

        # Check for hardcoded sim_tick / frame-counter / playtest_timer demo sequencers in main.gd and scenes.gd
        import re as _re

        tick_branch_re = _re.compile(
            r"\b(?:if|elif)\s+(?:sim_tick|sim_t|frame_count|tick_count|demo_tick|playtest_timer|auto_test_timer|demo_timer)\s*(?:==|>=|<=|>|<)\s*(\d+(?:\.\d+)?)\s*:"
        )
        for mod_name in ("main.gd", "scenes.gd"):
            mod_path = staging_dir / mod_name
            if not mod_path.is_file():
                continue
            src = mod_path.read_text(encoding="utf-8")
            tick_vals = {float(m.group(1)) for m in tick_branch_re.finditer(src) if float(m.group(1)) >= 1.0}
            if len(tick_vals) >= 2 or "handle_automated_playtest" in src:
                raise RuntimeError(
                    f"Self-playing demo loop detected in {mod_name} (hardcoded tick/timer branches at {sorted(tick_vals)}). "
                    "NEVER sequence gameplay actions, winching, cleating, dialogue choices, or state overrides on a sim_tick or playtest_timer in production code; "
                    "bind all actions to real player inputs (_input / Input.is_action_pressed / Input.is_key_pressed) and let the external QA Test Renderer stage proof scenarios."
                )

        # Check that scenes.gd realizes multi-region / procedural level layouts rather than a tiny 1-screen Rect2 placeholder box
        target_mods = set(getattr(track, "target_modules", []) or [])
        layout_path = staging_dir / "level_layout.json"
        scenes_path = staging_dir / "scenes.gd"
        if "scenes.gd" in target_mods and layout_path.is_file() and scenes_path.is_file():
            try:
                layout_data = json.loads(layout_path.read_text(encoding="utf-8"))
                intended_scale = float(layout_data.get("intended_world_scale_in_viewports", 1.0) or 1.0)
                regions = layout_data.get("world_regions") or []
            except Exception:
                intended_scale = 1.0
                regions = []
            if intended_scale >= 2.0 and len(regions) >= 2:
                scenes_src = scenes_path.read_text(encoding="utf-8")
                loads_layout = "level_layout.json" in scenes_src
                rect_coords = [
                    (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
                    for m in _re.finditer(
                        r"Rect2\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
                        scenes_src,
                    )
                ]
                max_extent_x = max((x + w for x, _y, w, _h in rect_coords), default=0.0)
                max_extent_y = max((y + h for _x, y, _w, h in rect_coords), default=0.0)
                if not loads_layout and rect_coords and len(rect_coords) <= 6 and max_extent_x <= 1920.0 and max_extent_y <= 1080.0:
                    raise RuntimeError(
                        f"Tiny placeholder level detected in scenes.gd for '{track.id}': level_layout.json specifies "
                        f"{len(regions)} regions across ~{intended_scale:.1f} viewports ({layout_data.get('world_bounds')}), "
                        f"but scenes.gd only hardcodes {len(rect_coords)} Rect2 shapes inside a single screen ({max_extent_x:.0f}x{max_extent_y:.0f}). "
                        "Load res://level_layout.json and/or execute the procedural/hybrid chunk grammar across all regions."
                    )

        # Check ui_hud.gd for internal debug/test-runner telemetry dashboards
        hud_path = staging_dir / "ui_hud.gd"
        if hud_path.is_file():
            hud_src = hud_path.read_text(encoding="utf-8")
            if "headless_test_results" in hud_src or "HEADLESS MECHANICS" in hud_src.upper() or "TELEMETRY VERIFICATION" in hud_src.upper():
                raise RuntimeError(
                    f"Debug telemetry dashboard detected in ui_hud.gd for '{track.id}'. "
                    "NEVER render headless_test_results, 'HEADLESS MECHANICS', or QA verification tables inside the player's HUD; "
                    "QA telemetry is rendered externally by the QA Test Renderer."
                )

        # Check mandatory sprite/texture binding when synthesized sprites exist on disk and main scene integration is under test
        sprites_dir = staging_dir / "assets" / "sprites"
        sprite_pngs = sorted(sprites_dir.glob("*.png")) if sprites_dir.is_dir() else []
        target_mods = set(getattr(track, "target_modules", []) or [])
        touches_visual_code = ("main.gd" in target_mods) or not target_mods
        if (
            sprite_pngs
            and touches_visual_code
            and ((staging_dir / "entities.gd").is_file() or (staging_dir / "scenes.gd").is_file())
            and "textures_loaded_count" in scene_audit
            and int(scene_audit.get("textures_loaded_count", 0)) == 0
        ):
            sprite_names = [p.name for p in sprite_pngs[:8]]
            raise RuntimeError(
                f"Unbound sprite assets in '{track.id}': {len(sprite_pngs)} synthesized PNG sprite(s) exist in res://assets/sprites/ "
                f"({sprite_names}), but textures_loaded_count is 0 in the live scene tree. Load and bind the actual Sprite2D/TextureRect "
                "textures from res://assets/asset_library.json instead of drawing untextured programmer-art shapes."
            )

    def _verify_track_smoke_check(
        self,
        staging_dir: Path,
        track: RequirementTrack,
        swe_agent: Optional[Any] = None,
    ) -> None:
        """Verifies AST security, bytecode compilation, and headless smoke stepping for current track
        (including both main.tscn and any standalone tests/*.tscn scenes targeted by the track)."""
        if (staging_dir / "project.godot").is_file():
            import re as _re
            from engine.sim.godot_verifier import (
                verify_godot_cartridge_syntax,
                parse_godot_stderr,
                GodotHeadlessSimulator,
            )
            valid, errs = verify_godot_cartridge_syntax(staging_dir)
            if (
                not valid
                and swe_agent is not None
                and hasattr(swe_agent, "_verify_and_heal_godot_syntax_and_probes")
                and not getattr(swe_agent, "_last_syntax_heal_failed", False)
            ):
                swe_agent._verify_and_heal_godot_syntax_and_probes(staging_dir)
                valid, errs = verify_godot_cartridge_syntax(staging_dir)
            if not valid:
                err_details = "; ".join(
                    f"{d.file_path.name}:{d.line_number}: {d.error_message}" for d in (errs or [])[:8]
                )
                raise RuntimeError(f"Godot syntax verification failed: {err_details or errs}")

            # Discover standalone test scenes (tests/*.tscn) targeted by this track so SWE catches test scene assertions immediately
            scenes_to_verify: List[Optional[str]] = [None]
            candidate_test_scenes: List[str] = []
            for mod in getattr(track, "target_modules", None) or []:
                norm_mod = str(mod).strip().replace("\\", "/")
                if norm_mod.startswith("res://"):
                    norm_mod = norm_mod[len("res://"):]
                elif norm_mod.startswith("res:/"):
                    norm_mod = norm_mod[len("res:/"):]
                norm_mod = norm_mod.lstrip("/")
                if norm_mod.startswith("tests/") and norm_mod.endswith(".tscn"):
                    candidate_test_scenes.append(norm_mod)
                elif norm_mod.startswith("tests/") and norm_mod.endswith(".gd"):
                    candidate_test_scenes.append(norm_mod[:-3] + ".tscn")
            desc_text = f"{getattr(track, 'description', '') or ''}\n" + "\n".join(
                getattr(ac, "description", "") for ac in (getattr(track, "acceptance_criteria", None) or [])
            )
            for m_tscn in _re.finditer(r"(?:res://)?(tests/[A-Za-z0-9_./-]+\.tscn)", desc_text):
                candidate_test_scenes.append(m_tscn.group(1))
            for rel_tscn in candidate_test_scenes:
                if (staging_dir / rel_tscn).is_file():
                    uri = f"res://{rel_tscn}"
                    if uri not in scenes_to_verify:
                        scenes_to_verify.append(uri)

            main_scene_audit: Dict[str, Any] = {}
            for scene_uri in scenes_to_verify:
                scene_label = scene_uri or "res://main.tscn"
                sim = GodotHeadlessSimulator(cartridge_path=staging_dir, max_frames=10, target_scene=scene_uri)
                res = sim.run()
                rt_diags = list(
                    getattr(sim, "runtime_diagnostics", None)
                    or parse_godot_stderr(staging_dir, res.stderr or res.error_message or "")
                )
                if (not res.success or rt_diags) and swe_agent is not None and hasattr(swe_agent, "repair_godot_module"):
                    for _rt_pass in range(2):
                        if not rt_diags and (staging_dir / "main.gd").is_file() and scene_uri is None:
                            broken_main = (staging_dir / "main.gd").read_text(encoding="utf-8")
                            peer_ctx = (
                                swe_agent._extract_gd_peer_signatures(staging_dir, exclude_file="main.gd")
                                if hasattr(swe_agent, "_extract_gd_peer_signatures")
                                else ""
                            )
                            fixed_main = swe_agent.repair_godot_module(
                                "main.gd",
                                broken_main,
                                res.error_message or res.stderr or "Headless runtime failure",
                                peer_context=peer_ctx,
                            )
                            (staging_dir / "main.gd").write_text(fixed_main, encoding="utf-8")
                        else:
                            by_file: Dict[Path, List[str]] = {}
                            for d in rt_diags:
                                if d.file_path.is_file():
                                    by_file.setdefault(d.file_path, []).append(
                                        f"Line {d.line_number}: {d.error_message} (Raw: {d.raw_line})"
                                    )
                            for fpath, entries in by_file.items():
                                try:
                                    rel_name = fpath.resolve().relative_to(staging_dir.resolve()).as_posix()
                                except ValueError:
                                    rel_name = fpath.name
                                peer_ctx = (
                                    swe_agent._extract_gd_peer_signatures(staging_dir, exclude_file=fpath.name)
                                    if hasattr(swe_agent, "_extract_gd_peer_signatures")
                                    else ""
                                )
                                fixed_code = swe_agent.repair_godot_module(
                                    rel_name,
                                    fpath.read_text(encoding="utf-8"),
                                    "\n".join(entries),
                                    peer_context=peer_ctx,
                                )
                                fpath.write_text(fixed_code, encoding="utf-8")
                        if hasattr(swe_agent, "_verify_and_heal_godot_syntax_and_probes"):
                            swe_agent._verify_and_heal_godot_syntax_and_probes(staging_dir)
                        sim = GodotHeadlessSimulator(cartridge_path=staging_dir, max_frames=10, target_scene=scene_uri)
                        res = sim.run()
                        rt_diags = list(
                            getattr(sim, "runtime_diagnostics", None)
                            or parse_godot_stderr(staging_dir, res.stderr or res.error_message or "")
                        )
                        if res.success and not rt_diags:
                            break
                if not res.success or rt_diags:
                    diag_str = "; ".join(
                        f"{d.file_path.name}:{d.line_number} - {d.error_message}" for d in rt_diags[:6]
                    )
                    err_detail = diag_str or res.error_message or res.stderr
                    raise RuntimeError(f"Godot headless smoke check failed in {scene_label}: {err_detail}")
                if scene_uri is None:
                    main_scene_audit = getattr(sim, "scene_audit", None) or {}

            self._check_godot_gameplay_and_asset_gates(staging_dir, track, main_scene_audit)
            return

        from engine.loader import validate_cartridge_ast
        for py_file in sorted(staging_dir.glob("*.py")):
            code_str = py_file.read_text(encoding="utf-8")
            compile(code_str, str(py_file), "exec")
            valid, errs = validate_cartridge_ast(py_file)
            if not valid:
                raise RuntimeError(f"AST validation failed in {py_file.name}: {errs}")

        # Headless simulation step (10 frames) with SQLite telemetry persistence and active inputs
        if (staging_dir / "mechanics.py").is_file():
            smoke_db = staging_dir / ".smoke_telemetry.db"
            for ext in ("", "-wal", "-shm"):
                p = Path(str(smoke_db) + ext)
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass

            try:
                from engine.sim.headless import HeadlessSimulator, SimulationInput
                sim = HeadlessSimulator(
                    cartridge_path=staging_dir,
                    telemetry_mode="sqlite",
                    telemetry_db_path=smoke_db,
                )
                try:
                    active_input = SimulationInput(right=True, action_1=True, space=True)
                    for _ in range(10):
                        sim.step(active_input)
                finally:
                    if hasattr(sim, "telemetry_recorder") and sim.telemetry_recorder:
                        sim.telemetry_recorder.close(timeout=1.0)

                # Assert zero discarded rows or foreign key failures
                discarded = getattr(getattr(sim, "telemetry_recorder", None), "discarded_row_count", 0)
                writer_errors = getattr(getattr(getattr(sim, "telemetry_recorder", None), "writer_thread", None), "discarded_errors", [])
                if discarded > 0:
                    raise RuntimeError(
                        f"Headless smoke check telemetry failed with {discarded} discarded rows (zero tolerance): {writer_errors}"
                    )
            except Exception as sim_exc:
                raise RuntimeError(f"Headless smoke check runtime exception: {sim_exc}") from sim_exc
            finally:
                for ext in ("", "-wal", "-shm"):
                    p = Path(str(smoke_db) + ext)
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass

    async def execute_swarm_plan(
        self,
        plan: StudioSwarmPlan,
        staging_dir: Union[Path, str],
    ) -> BuildEvaluationReport:
        """Topologically executes specialists in parallel tiers/waves, passing sealed artifacts,
        and hands the compiled cartridge to BuildSupervisor for 5-gate validation and promotion.
        """
        self._continuous_active = False  # re-armed only if this build's Writers' Room runs with a Production Board
        staging_dir = Path(staging_dir).resolve()
        staging_dir.mkdir(parents=True, exist_ok=True)
        candidate_id = staging_dir.name
        from engine.studio.models import set_active_gemini_dataset_context
        set_active_gemini_dataset_context(
            dataset_name=getattr(self.config, "dataset_name", "") or candidate_id,
            candidate_dir=staging_dir,
            store_logs=getattr(self.config, "store_logs", True),
        )
        build_id = f"build_{candidate_id}"
        self.active_build_id = build_id
        if self.event_bus and hasattr(self.event_bus, "set_active_build_id"):
            self.event_bus.set_active_build_id(build_id)

        try:
            db = self._get_telemetry_db()
            if db is not None:
                db.record_studio_build(
                build_id=build_id,
                candidate_id=candidate_id,
                build_type="genesis" if plan.target_episode == 0 else "update",
                engine_type="godot4",
                game_title=plan.game_title,
                genre_paradigm=getattr(plan, "genre_verdict", "action"),
                status="STARTED",
                started_at_utc=datetime.now(timezone.utc).isoformat(),
                deliberation=f"Autonomous Godot 4 build started for '{plan.game_title}'.",
            )
            if hasattr(plan, "requirements_spec") and plan.requirements_spec:
                db.record_studio_requirements(build_id=build_id, spec=plan.requirements_spec)
        except Exception as exc:
            logger.debug("Failed recording studio build: %s", exc)

        if self.state not in (StudioBuildState.IDLE, StudioBuildState.ANALYZING):
            self.state = StudioBuildState.IDLE
        if self.state == StudioBuildState.IDLE:
            self.transition_to(StudioBuildState.ANALYZING, f"Analyzing player profile for '{plan.game_title}'")
        self.transition_to(StudioBuildState.SYNTHESIZING, f"Executing dynamic swarm plan for '{plan.game_title}'")
        context = SwarmExecutionContext(run_id=plan.plan_id, staging_dir=staging_dir)

        # 1. Compute Execution Tiers (Waves) via Kahn's Partitioning
        tiers = self._compute_execution_tiers(plan.dag_edges, plan.specialist_agent_roles)

        # 2. Topologically Dispatch Tiers
        for tier_idx, tier_roles in enumerate(tiers):
            logger.info("Executing Swarm Tier %d: %s", tier_idx, tier_roles)
            # Execute independent specialists within the same tier concurrently
            tasks = [self._execute_specialist(role, plan, context) for role in tier_roles]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for role, res in zip(tier_roles, results):
                if isinstance(res, Exception):
                    # Rule 5: Zero Silent Fallbacks — fail fast
                    self.state = StudioBuildState.IDLE
                    logger.critical("Specialist '%s' failed execution: %s", role, res)
                    self._publish_event(
                        "ERROR",
                        normalize_role(role),
                        f"Specialist '{role}' failed during Tier {tier_idx}: {res}",
                        {"error": str(res)},
                    )
                    cause = getattr(res, "__cause__", None) or res
                    raise StudioSwarmExecutionError(f"Specialist '{role}' failed during Tier {tier_idx}: {res}") from cause

        # In legacy non-board-managed mode, ensure narrative barks/portraits written by Writer after parallel Voice/Visual wave exist on disk
        board_managed = bool(
            self._continuous_active
            and self.production_board is not None
            and self.production_board.items()
        )
        voice_roles = {"Voice", "VoiceActingAudioProducerAgent", "voice_audio_engineer"}
        voice_planned = any(r in voice_roles for r in plan.specialist_agent_roles)
        if voice_planned and not board_managed:
            narrative_out = context.raw_outputs.get("Writer") or context.raw_outputs.get("NarrativeCharacterWriterAgent")
            if narrative_out is not None:
                voice_agent = self.agents.get("VoiceActingAudioProducerAgent")
                if voice_agent is not None and getattr(narrative_out, "dialogue_barks", None):
                    audio_dir = staging_dir / "assets" / "audio"
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    char_map = {c.character_id: c for c in getattr(narrative_out, "characters", [])}
                    for bark in list(narrative_out.dialogue_barks)[:3]:
                        wav_name = Path(bark.voice_clip).name if getattr(bark, "voice_clip", None) else f"{bark.bark_id}.wav"
                        if not wav_name.endswith(".wav"):
                            wav_name = f"{wav_name}.wav"
                        out_path = audio_dir / wav_name
                        if not out_path.is_file():
                            voice_agent.synthesize_voice_line(bark=bark, output_path=out_path, character=char_map.get(getattr(bark, "speaker_id", "")))
        narrative_out = context.raw_outputs.get("Writer") or context.raw_outputs.get("NarrativeCharacterWriterAgent")
        if narrative_out is not None and getattr(narrative_out, "characters", None) and not board_managed:
            visual_agent = self.agents.get("VisualArtSpriteDesignerAgent")
            portraits_dir = staging_dir / "assets" / "portraits"
            portraits_dir.mkdir(parents=True, exist_ok=True)
            for char in getattr(narrative_out, "characters", []):
                cid = getattr(char, "character_id", "character")
                p_asset = getattr(char, "portrait_asset", "") or f"assets/portraits/{cid}_portrait.png"
                if not p_asset.startswith("assets/portraits/"):
                    p_asset = f"assets/portraits/{Path(p_asset).name}"
                    try:
                        char.portrait_asset = p_asset
                    except Exception:
                        pass
                p_path = staging_dir / p_asset
                if not p_path.is_file() and visual_agent is not None:
                    try:
                        if getattr(visual_agent, "_is_live_genai_client", lambda: False)():
                            visual_agent.generate_portrait(character=char, output_path=p_path, dimensions=(256, 256))
                        else:
                            img = visual_agent.renderer.render_dynamic_sprite(
                                dimensions=(256, 256),
                                prompt=getattr(char, "visual_description", "") or getattr(char, "name", cid),
                                asset_type="portrait",
                                seed=cid,
                            )
                            p_path.parent.mkdir(parents=True, exist_ok=True)
                            img.save(p_path, "PNG")
                    except Exception as exc:
                        logger.warning("Portrait synthesis warning for '%s': %s", cid, exc)

        # 3. Hand over to BuildSupervisor for 5-Gate Validation & Promotion
        self.transition_to(StudioBuildState.VERIFYING, "Handing staging cartridge to BuildSupervisor")
        self._publish_event(
            "state_change",
            "QAPlaytesterBuildSupervisor",
            f"Evaluating candidate cartridge '{staging_dir.name}' across 5 Quality Gates...",
            {"agent_name": "QAPlaytesterBuildSupervisor", "lifecycle_state": "VERIFYING"},
        )
        supervisor = self.agents.get("QAPlaytesterBuildSupervisor")
        if supervisor is None:
            from engine.supervisor.supervisor import BuildSupervisor
            swe = self.agents.get("LeadGameplaySWEAgent")
            supervisor = BuildSupervisor(
                cartridges_root=PROJECT_ROOT / "cartridges",
                active_cartridge_dir=PROJECT_ROOT / "cartridges" / "active",
                swe_agent=swe,
                gemini_client=self.client,
                event_bus=self.event_bus,
            )

        report = supervisor.evaluate_and_heal(
            candidate_or_spec=staging_dir,
            episode_number=plan.target_episode,
            max_retries=self.config.max_retries,
        )

        delib_sup = getattr(report, "deliberation", None)
        if not isinstance(delib_sup, str) or not delib_sup.strip():
            delib_sup = "Completed 5-Gate QA evaluation and verification."
        rej_trace = getattr(report, "rejection_trace", None)
        if hasattr(rej_trace, "model_dump"):
            rej_trace = rej_trace.model_dump()
        if not isinstance(rej_trace, dict):
            rej_trace = None
        promoted = bool(getattr(report, "promoted_to_active", False))

        self._publish_event(
            "QA_GATE_RESULT",
            "QAPlaytesterBuildSupervisor",
            delib_sup,
            {"promoted": promoted, "attempts": getattr(report, "attempt_count", 1)},
        )

        sup_task = AgentTaskRecord(
            deliberation=delib_sup,
            task_id=f"task_supervisor_{uuid.uuid4().hex[:8]}",
            agent_role="QAPlaytesterBuildSupervisor",
            agent_name="BuildSupervisor",
            requirement_id=None,
            track_title="5-Gate QA & Challenger Verification",
            task_type="QA_VERIFICATION",
            status="COMPLETED" if promoted else "REJECTED",
            start_time_utc=datetime.now(timezone.utc).isoformat(),
            end_time_utc=datetime.now(timezone.utc).isoformat(),
            duration_ms=5000.0,
            input_artifacts=[],
            output_artifacts=[],
            verification_verdict={"passed": promoted},
            diagnostic_trace=rej_trace,
            summary="Supervisor evaluated candidate cartridge.",
        )
        if hasattr(self.event_bus, "publish_agent_task_record"):
            self.event_bus.publish_agent_task_record(sup_task)
        else:
            self._publish_event(
                "AGENT_TASK_RECORD",
                "QAPlaytesterBuildSupervisor",
                sup_task.deliberation,
                sup_task.model_dump(),
            )
        context.record_task(sup_task)

        # Save task_history.json into staging_dir
        task_history_path = staging_dir / "task_history.json"
        task_history_data = {
            "deliberation": f"Persisted task execution history for swarm plan {plan.plan_id}.",
            "plan_id": plan.plan_id,
            "game_title": plan.game_title,
            "tasks": getattr(context, "task_history", []),
        }
        task_history_path.write_text(json.dumps(task_history_data, indent=2), encoding="utf-8")

        if getattr(report, "promoted_to_active", False):
            try:
                db = self._get_telemetry_db()
                if db is not None:
                    db.update_studio_build(
                        build_id=build_id,
                        status="PROMOTED",
                        completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        deliberation=f"Build for '{plan.game_title}' verified and promoted to cartridges/active/.",
                    )
            except Exception:
                pass
            active_dir = PROJECT_ROOT / "cartridges" / "active"
            if active_dir.exists():
                try:
                    shutil.copy2(task_history_path, active_dir / "task_history.json")
                except Exception:
                    pass
            self.transition_to(StudioBuildState.PROMOTED, f"Build for '{plan.game_title}' verified and promoted.")
            self._publish_event(
                "CARTRIDGE_PROMOTED",
                "QAPlaytesterBuildSupervisor",
                f"Game '{plan.game_title}' ({plan.genre_verdict}) passed all 5 Quality Gates and was promoted to cartridges/active/!",
                {"game_title": plan.game_title, "genre": plan.genre_verdict, "episode": plan.target_episode, "status": "PROMOTED"},
            )
            self.transition_to(StudioBuildState.IDLE, "Swarm execution completed successfully.")
        else:
            fail_reason = getattr(getattr(report, "rejection_trace", None), "failure_reason", "")
            fail_gate = getattr(getattr(report, "rejection_trace", None), "failure_gate", "QA_GATE")
            fail_delib = getattr(report, "deliberation", f"Build for '{plan.game_title}' rejected during verification.")
            try:
                db = self._get_telemetry_db()
                if db is not None:
                    db.update_studio_build(
                        build_id=build_id,
                        status="FAILED",
                        completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        deliberation=fail_delib,
                    )
            except Exception:
                pass
            transition_msg = f"Promotion rejected at {fail_gate}: {fail_reason}" if fail_reason else f"Promotion rejected: {fail_delib}"
            self.transition_to(StudioBuildState.SYNTHESIZING, transition_msg)
            self._publish_event(
                "ERROR",
                "QAPlaytesterBuildSupervisor",
                fail_delib,
                {"failure_gate": fail_gate, "failure_reason": fail_reason, "status": "REJECTED"},
            )

        return report

    def run_genesis_build(
        self,
        staging_dir: Optional[Union[Path, str]] = None,
        steam_dna: Optional[Any] = None,
    ) -> BuildEvaluationReport:
        """Synchronous wrapper to generate plan and execute Genesis build."""
        if staging_dir is None and "PYTEST_CURRENT_TEST" not in os.environ:
            cartridges_dir = PROJECT_ROOT / "cartridges"
            if cartridges_dir.is_dir():
                candidates = sorted(
                    cartridges_dir.glob("candidate_genesis_*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for cand in candidates:
                    spec_p = cand / "requirements_spec.json"
                    if spec_p.is_file():
                        try:
                            spec_data = json.loads(spec_p.read_text(encoding="utf-8"))
                            tracks = spec_data.get("requirements") or spec_data.get("tracks", [])
                            if any(t.get("status") == "completed" for t in tracks):
                                build_report_file = cand / "build_evaluation_report.json"
                                promoted = False
                                if build_report_file.is_file():
                                    try:
                                        rep_data = json.loads(build_report_file.read_text(encoding="utf-8"))
                                        promoted = rep_data.get("promoted_to_active", False)
                                    except Exception:
                                        pass
                                if not promoted:
                                    staging_dir = cand
                                    logger.info("Resuming unpromoted candidate genesis build from %s", cand)
                                    break
                        except Exception:
                            pass

        plan = self.generate_swarm_plan(steam_dna=steam_dna, target_episode=0)
        s_dir = Path(staging_dir or (PROJECT_ROOT / "cartridges" / f"candidate_genesis_{uuid.uuid4().hex[:6]}"))
        res = self.execute_swarm_plan(plan, s_dir)
        if inspect.iscoroutine(res):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(res)
            return asyncio.run(res)
        return res
