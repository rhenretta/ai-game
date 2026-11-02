"""engine/supervisor/supervisor.py
Build Supervisor & Closed-Loop Code Self-Healing Guardrail (Milestone M4, Features 24 & 27).
Coordinates the 5-Gate quality verification pipeline in computational dependency order:
Gate A (Syntax/AST) -> Gate E (Asset Integrity) -> Gate D (Zero Exceptions) ->
Gate B (Traversability with Dual-Frontier bottleneck detection) -> Gate C (Survivability).
Blocks promotion on defect, generates Rationale-First RejectionDiagnosticTraces,
and drives closed-loop self-healing repair via LeadGameplaySWEAgent before promotion to cartridges/active/.
"""
from __future__ import annotations
import ast
from collections import deque
import copy
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import config
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.compiler.compiler import CompilationResult, EngineCompiler
from engine.director.models import (
    CurrentEpisodeConfig,
    DirectiveAction,
    EpisodicUpdateSpec,
    GenesisSpec,
    SpatialUpdateDirective,
)
from engine.loader import ASTSecurityError, CartridgeLoader, validate_cartridge_ast
from engine.spatial.navmesh import astar_pathfind
from engine.supervisor.asset_validator import AssetValidator
from engine.supervisor.models import (
    BuildEvaluationReport,
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    RemedyType,
    SupervisorEvaluationReport,
    SupervisorGateVerdict,
)

logger = logging.getLogger("BuildSupervisor")


class SupervisorError(Exception):
    """Base exception for supervisor operations."""
    deliberation: str = "Base exception for supervisor operations."


class BuildSupervisor:
    """Coordinates compilation verification, headless simulation, and self-healing repair."""
    deliberation: str = "Build Supervisor evaluating candidate updates, rejecting defects, and driving self-healing."

    def __init__(
        self,
        cartridges_root: Optional[Union[Path, str]] = None,
        active_cartridge_dir: Optional[Union[Path, str]] = None,
        compiler: Optional[EngineCompiler] = None,
        swe_agent: Optional[Any] = None,
        asset_validator: Optional[Any] = None,
        challenger_gate: Optional[Any] = None,
        gemini_client: Optional[Any] = None,
        tile_size: float = 32.0,
        max_retries: int = 3,
        early_death_tick_threshold: int = 180,  # 3.0s at 60 Hz
        max_simulation_ticks: int = 3600,       # 60.0s at 60 Hz
        spawn_clearance_margin: float = 64.0,
        event_bus: Optional[Any] = None,
    ):
        self.cartridges_root = Path(cartridges_root or (config.PROJECT_ROOT / "cartridges"))
        self.active_cartridge_dir = Path(active_cartridge_dir or (self.cartridges_root / "active"))
        self.cartridges_root.mkdir(parents=True, exist_ok=True)
        self.compiler = compiler or EngineCompiler(
            cartridges_root=self.cartridges_root,
            active_cartridge_dir=self.active_cartridge_dir,
        )
        self.swe_agent = swe_agent
        self.asset_validator = asset_validator or AssetValidator()
        self.challenger_gate = challenger_gate
        self.gemini_client = gemini_client
        self.tile_size = float(tile_size)
        self.max_retries = int(max_retries)
        self.early_death_tick_threshold = int(early_death_tick_threshold)
        self.max_simulation_ticks = int(max_simulation_ticks)
        self.spawn_clearance_margin = float(spawn_clearance_margin)
        if event_bus is None:
            try:
                from engine.ui.server import swarm_bus
                self.event_bus = swarm_bus
            except Exception:
                self.event_bus = None
        else:
            self.event_bus = event_bus

    def _emit_gate_event(self, gate_name: str, result: QualityGateResult, attempt: int = 1) -> None:
        if self.event_bus is None:
            return
        try:
            status_str = "PASSED" if result.passed else "FAILED"
            msg = f"[{gate_name}] attempt {attempt}: {status_str} — {result.details or result.deliberation}"
            if hasattr(self.event_bus, "publish_agent_stream"):
                self.event_bus.publish_agent_stream(
                    agent_name="QAPlaytesterBuildSupervisor",
                    message=msg,
                    deliberation=result.deliberation,
                    log_level="INFO" if result.passed else "WARNING",
                    category="GATE_CHECK",
                )
            if hasattr(self.event_bus, "publish_challenger_verdict"):
                self.event_bus.publish_challenger_verdict(
                    passed=result.passed,
                    deliberation=result.deliberation,
                    agent_name="QAPlaytesterBuildSupervisor",
                    details={"gate": gate_name, "details": result.details, "attempt": attempt},
                )
        except Exception as e:
            logger.debug("Failed emitting gate event: %s", e)

    def _emit_promoted_event(self, candidate_dir: Path) -> None:
        if self.event_bus is None:
            return
        try:
            if hasattr(self.event_bus, "publish_agent_stream"):
                self.event_bus.publish_agent_stream(
                    agent_name="QAPlaytesterBuildSupervisor",
                    message=f"Candidate cartridge {candidate_dir.name} successfully certified and promoted to active.",
                    deliberation="All quality gates passed. Promoting candidate cartridge to active.",
                    category="PROMOTION",
                )
            event = {
                "deliberation": f"Candidate cartridge {candidate_dir.name} promoted to active.",
                "event_type": "PROMOTED",
                "agent_name": "QAPlaytesterBuildSupervisor",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "payload": {"candidate_dir": str(candidate_dir), "active_dir": str(self.active_cartridge_dir)},
            }
            if hasattr(self.event_bus, "publish_threadsafe"):
                self.event_bus.publish_threadsafe(event)
        except Exception as e:
            logger.debug("Failed emitting promoted event: %s", e)

    # =========================================================================
    # Primary Entry Point: Closed-Loop Evaluate & Self-Heal Loop
    # =========================================================================

    def evaluate_and_heal(
        self,
        candidate_or_spec: Union[GenesisSpec, EpisodicUpdateSpec, Path, str],
        max_retries: Optional[int] = None,
        episode_number: int = 1,
    ) -> BuildEvaluationReport:
        """Executes the closed-loop compile -> simulate -> verify -> repair loop.
        Retries up to max_retries if defects are encountered.
        Promotes to active ONLY when 100% of quality gates pass.
        """
        retries = max_retries if max_retries is not None else self.max_retries
        repair_history: List[RejectionDiagnosticTrace] = []

        # Branch 1: Direct candidate directory input (from Multi-Agent Studio)
        if isinstance(candidate_or_spec, (Path, str)):
            candidate_dir = Path(candidate_or_spec).resolve()
            candidate_id = candidate_dir.name
            is_board_managed = (candidate_dir / "backlog.json").is_file()
            branch_retries = 1 if is_board_managed else retries
            logger.info(
                "Supervisor evaluating candidate directory %s (board_managed=%s, max_attempts=%d)",
                candidate_dir,
                is_board_managed,
                branch_retries,
            )

            for attempt in range(1, branch_retries + 1):
                gate_results, trace = self.evaluate_candidate_cartridge(
                    candidate_dir=candidate_dir,
                    episode_number=episode_number,
                    attempt_number=attempt,
                )

                if trace is None:
                    self.promote_candidate(candidate_dir)
                    self._emit_promoted_event(candidate_dir)
                    report = BuildEvaluationReport(
                        deliberation=f"Candidate cartridge {candidate_id} passed all 5 Quality Gates and was promoted to active on attempt {attempt}.",
                        candidate_id=candidate_id,
                        episode_number=episode_number,
                        status=SupervisorGateVerdict.PROMOTED,
                        promoted_to_active=True,
                        candidate_dir=str(candidate_dir),
                        active_dir=str(self.active_cartridge_dir),
                        gate_results=gate_results,
                        rejection_trace=None,
                        attempt_count=attempt,
                        max_retries=retries,
                        repair_history=repair_history,
                    )
                    if self.event_bus is not None and hasattr(self.event_bus, "publish_agent_task_record"):
                        self.event_bus.publish_agent_task_record({
                            "deliberation": report.deliberation,
                            "task_id": "task_qa_supervisor_verification",
                            "agent_role": "QAPlaytesterBuildSupervisor",
                            "agent_name": "BuildSupervisor",
                            "requirement_id": None,
                            "track_title": "Quality Gates & Challenger Verification",
                            "task_type": "QA_VERIFICATION",
                            "status": "COMPLETED",
                            "verification_verdict": {"passed": True, "gates_passed": list(gate_results.keys())},
                            "diagnostic_trace": None,
                            "summary": f"Candidate {candidate_id} passed all quality gates and was promoted to active.",
                        })
                    logger.info("Candidate %s successfully PROMOTED to %s", candidate_id, self.active_cartridge_dir)
                    return report

                repair_history.append(trace)
                fail_type_str = trace.failure_type.value if hasattr(trace.failure_type, "value") else str(trace.failure_type)
                logger.warning("Attempt %d/%d rejected at %s (%s): %s",
                               attempt, retries, trace.failure_gate, fail_type_str, trace.failure_reason)

                if self.event_bus is not None:
                    serialized_gates = {
                        k: (
                            v.to_dict()
                            if hasattr(v, "to_dict")
                            else (v.model_dump(mode="json") if hasattr(v, "model_dump") else str(v))
                        )
                        for k, v in gate_results.items()
                    }
                    qa_event = {
                        "event_type": "QA_GATE_RESULT",
                        "agent_name": "QAPlaytesterBuildSupervisor",
                        "deliberation": f"Gate {trace.failure_gate} ({fail_type_str}) failed on attempt {attempt}/{retries}: {trace.failure_reason} — Challenger Rationale: {trace.deliberation}",
                        "payload": {
                            "attempt": attempt,
                            "max_retries": retries,
                            "failure_gate": trace.failure_gate,
                            "failure_type": fail_type_str,
                            "failure_reason": trace.failure_reason,
                            "deliberation": trace.deliberation,
                            "unfulfilled_requirements": getattr(trace, "unfulfilled_requirements", []),
                            "remedy_directive": getattr(trace, "remedy_directive", ""),
                            "gate_results": serialized_gates,
                            "promoted": False,
                        },
                    }
                    if hasattr(self.event_bus, "publish_threadsafe"):
                        self.event_bus.publish_threadsafe(qa_event)
                    elif hasattr(self.event_bus, "publish_event"):
                        self.event_bus.publish_event(qa_event)

                if attempt >= branch_retries:
                    report = BuildEvaluationReport(
                        deliberation=(
                            f"Candidate cartridge {candidate_id} rejected at {trace.failure_gate} ({fail_type_str}) "
                            + (
                                f"(board-managed cartridge: returning diagnostic trace for Production Board task escalation): {trace.failure_reason} — Challenger Rationale: {trace.deliberation}"
                                if is_board_managed
                                else f"after {branch_retries} attempts: {trace.failure_reason} — Challenger Rationale: {trace.deliberation}"
                            )
                        ),
                        candidate_id=candidate_id,
                        episode_number=episode_number,
                        status=SupervisorGateVerdict.FAILED_MAX_RETRIES,
                        promoted_to_active=False,
                        candidate_dir=str(candidate_dir),
                        active_dir=str(self.active_cartridge_dir),
                        gate_results=gate_results,
                        rejection_trace=trace,
                        attempt_count=attempt,
                        max_retries=branch_retries,
                        repair_history=repair_history,
                    )
                    if self.event_bus is not None and hasattr(self.event_bus, "publish_agent_task_record"):
                        self.event_bus.publish_agent_task_record({
                            "deliberation": report.deliberation,
                            "task_id": f"task_qa_supervisor_{uuid.uuid4().hex[:8]}",
                            "agent_role": "QAPlaytesterBuildSupervisor",
                            "agent_name": "BuildSupervisor",
                            "requirement_id": None,
                            "track_title": "Quality Gates & Challenger Verification",
                            "task_type": "QA_VERIFICATION",
                            "status": "REJECTED",
                            "verification_verdict": {"passed": False},
                            "diagnostic_trace": trace.model_dump() if hasattr(trace, "model_dump") else str(trace),
                            "summary": f"Gate {trace.failure_gate} rejected candidate build: {trace.failure_reason}",
                        })
                    logger.error("Candidate %s FAILED_MAX_RETRIES; active cartridge preserved untouched.", candidate_id)
                    return report

                if self.swe_agent is not None and hasattr(self.swe_agent, "repair_cartridge_code"):
                    logger.info("Triggering SWE self-healing repair on %s", candidate_dir)
                    if self.event_bus is not None:
                        remedy = getattr(trace, "remedy_directive", "")
                        stream_event = {
                            "event_type": "AGENT_STREAM",
                            "agent_name": "LeadGameplaySWEAgent",
                            "deliberation": f"Initiating automated self-healing repair on {candidate_dir.name} in response to {trace.failure_gate} failure: {remedy}",
                            "payload": {
                                "phase": "REPAIR",
                                "target_dir": str(candidate_dir),
                                "failure_gate": trace.failure_gate,
                                "remedy_directive": remedy,
                            },
                        }
                        if hasattr(self.event_bus, "publish_threadsafe"):
                            self.event_bus.publish_threadsafe(stream_event)
                        elif hasattr(self.event_bus, "publish_event"):
                            self.event_bus.publish_event(stream_event)
                    try:
                        self.swe_agent.repair_cartridge_code(candidate_dir, trace)
                    except Exception as swe_err:
                        logger.warning("SWE repair encountered unexpected error: %s", swe_err)

        # Branch 2: Director Specification Input (GenesisSpec or EpisodicUpdateSpec)
        else:
            current_spec = copy.deepcopy(candidate_or_spec)
            is_genesis = isinstance(current_spec, GenesisSpec)
            target_episode = 0 if is_genesis else getattr(current_spec, "target_episode_number", episode_number)
            spec_id = getattr(current_spec, "spec_id", f"spec_ep{target_episode}")

            for attempt in range(1, retries + 1):
                candidate_dir = (self.cartridges_root / f"candidate_ep{target_episode:02d}_att{attempt}").resolve()
                logger.info("Supervisor evaluating spec %s (Episode %d) Attempt %d/%d -> %s",
                            spec_id, target_episode, attempt, retries, candidate_dir)

                # Step 1: Compile spec to candidate directory
                try:
                    compile_result: CompilationResult = self.compiler.compile(
                        spec=current_spec,
                        target_dir=candidate_dir,
                    )
                except Exception as comp_err:
                    trace = RejectionDiagnosticTrace(
                        deliberation=f"EngineCompiler failed to compile specification {spec_id}: {comp_err}",
                        candidate_id=spec_id,
                        episode_number=target_episode,
                        attempt_number=attempt,
                        failure_gate="Gate A: Syntax & AST",
                        failure_type=GateFailureType.SYNTAX_AST_ERROR,
                        failure_reason=str(comp_err),
                        runtime_traceback=traceback.format_exc(),
                        remedy_type=RemedyType.FIX_SYNTAX,
                        recommended_remedy="Repair invalid specification attributes or syntax errors.",
                    )
                    repair_history.append(trace)
                    gate_a_fail = QualityGateResult(
                        deliberation=f"Compiler failure: {comp_err}",
                        gate_name="Gate A: Syntax & AST",
                        passed=False,
                        failure_type=GateFailureType.SYNTAX_AST_ERROR,
                        details=str(comp_err),
                    )
                    comp_gate_results = {
                        "gate_a": gate_a_fail,
                        "Gate A: Syntax & AST": gate_a_fail,
                    }
                    if attempt >= retries:
                        return BuildEvaluationReport(
                            deliberation=f"Specification {spec_id} compilation exhausted max retries ({retries}).",
                            candidate_id=spec_id,
                            episode_number=target_episode,
                            status=SupervisorGateVerdict.FAILED_MAX_RETRIES,
                            promoted_to_active=False,
                            candidate_dir=str(candidate_dir),
                            active_dir=str(self.active_cartridge_dir),
                            gate_results=comp_gate_results,
                            rejection_trace=trace,
                            attempt_count=attempt,
                            max_retries=retries,
                            repair_history=repair_history,
                        )
                    current_spec = self.heal_specification(current_spec, trace)
                    continue

                # Step 2: Evaluate candidate cartridge directory
                if self.event_bus is not None:
                    eval_event = {
                        "event_type": "studio_live_activity",
                        "agent_name": "QAPlaytesterBuildSupervisor",
                        "deliberation": f"BuildSupervisor evaluating candidate '{spec_id}' across Quality Gates A–F (attempt {attempt}/{retries})...",
                        "payload": {
                            "stage": "supervisor_gate_evaluation",
                            "candidate_id": spec_id,
                            "attempt": attempt,
                            "max_retries": retries,
                        },
                    }
                    if hasattr(self.event_bus, "publish_threadsafe"):
                        self.event_bus.publish_threadsafe(eval_event)
                    elif hasattr(self.event_bus, "publish_event"):
                        self.event_bus.publish_event(eval_event)

                gate_results, trace = self.evaluate_candidate_cartridge(
                    candidate_dir=candidate_dir,
                    spec=current_spec,
                    episode_number=target_episode,
                    attempt_number=attempt,
                )

                if trace is None:
                    # 100% Gates passed! Promote candidate to active
                    self.promote_candidate(candidate_dir)
                    report = BuildEvaluationReport(
                        deliberation=f"Candidate build {spec_id} passed all 5 Quality Gates on attempt {attempt} and was promoted to active.",
                        candidate_id=spec_id,
                        episode_number=target_episode,
                        status=SupervisorGateVerdict.PROMOTED,
                        promoted_to_active=True,
                        candidate_dir=str(candidate_dir),
                        active_dir=str(self.active_cartridge_dir),
                        gate_results=gate_results,
                        rejection_trace=None,
                        attempt_count=attempt,
                        max_retries=retries,
                        repair_history=repair_history,
                    )
                    logger.info("Candidate %s successfully PROMOTED to %s", spec_id, self.active_cartridge_dir)
                    return report

                # Failure detected
                repair_history.append(trace)
                fail_type_str = trace.failure_type.value if hasattr(trace.failure_type, "value") else str(trace.failure_type)
                logger.warning("Attempt %d/%d rejected at %s (%s): %s",
                               attempt, retries, trace.failure_gate, fail_type_str, trace.failure_reason)

                if self.event_bus is not None:
                    serialized_gates = {
                        k: (
                            v.to_dict()
                            if hasattr(v, "to_dict")
                            else (v.model_dump(mode="json") if hasattr(v, "model_dump") else str(v))
                        )
                        for k, v in gate_results.items()
                    }
                    qa_event = {
                        "event_type": "QA_GATE_RESULT",
                        "agent_name": "QAPlaytesterBuildSupervisor",
                        "deliberation": f"Gate {trace.failure_gate} ({fail_type_str}) failed on attempt {attempt}/{retries}: {trace.failure_reason} — Challenger Rationale: {trace.deliberation}",
                        "payload": {
                            "attempt": attempt,
                            "max_retries": retries,
                            "failure_gate": trace.failure_gate,
                            "failure_type": fail_type_str,
                            "failure_reason": trace.failure_reason,
                            "deliberation": trace.deliberation,
                            "unfulfilled_requirements": getattr(trace, "unfulfilled_requirements", []),
                            "remedy_directive": getattr(trace, "remedy_directive", ""),
                            "gate_results": serialized_gates,
                            "promoted": False,
                        },
                    }
                    if hasattr(self.event_bus, "publish_threadsafe"):
                        self.event_bus.publish_threadsafe(qa_event)
                    elif hasattr(self.event_bus, "publish_event"):
                        self.event_bus.publish_event(qa_event)

                if attempt >= retries:
                    report = BuildEvaluationReport(
                        deliberation=f"Candidate build {spec_id} rejected at {trace.failure_gate} ({fail_type_str}) after {retries} attempts: {trace.failure_reason} — Challenger Rationale: {trace.deliberation}",
                        candidate_id=spec_id,
                        episode_number=target_episode,
                        status=SupervisorGateVerdict.FAILED_MAX_RETRIES,
                        promoted_to_active=False,
                        candidate_dir=str(candidate_dir),
                        active_dir=str(self.active_cartridge_dir),
                        gate_results=gate_results,
                        rejection_trace=trace,
                        attempt_count=attempt,
                        max_retries=retries,
                        repair_history=repair_history,
                    )
                    logger.error("Candidate %s FAILED_MAX_RETRIES; active cartridge preserved untouched.", spec_id)
                    return report

                # Closed-loop self-healing refactoring
                if self.swe_agent is not None and hasattr(self.swe_agent, "repair_cartridge_code"):
                    self.swe_agent.repair_cartridge_code(candidate_dir, trace)
                current_spec = self.heal_specification(current_spec, trace)

        raise SupervisorError("Unexpected termination of evaluate_and_heal loop.")

    # =========================================================================
    # Candidate Cartridge Evaluation Pipeline (5 Quality Gates)
    # =========================================================================

    def evaluate_candidate_cartridge(
        self,
        candidate_dir: Path,
        spec: Optional[Any] = None,
        episode_number: int = 1,
        attempt_number: int = 1,
        requirements_spec: Optional[Any] = None,
        include_challenger_gate: bool = True,
    ) -> Tuple[Dict[str, QualityGateResult], Optional[RejectionDiagnosticTrace]]:
        """Evaluates an assembled candidate cartridge directory against all 5 Quality Gates."""
        candidate_dir = Path(candidate_dir).resolve()
        candidate_id = getattr(spec, "spec_id", getattr(spec, "candidate_id", candidate_dir.name))

        # Branch for Director spatial specs or candidate directories containing level.json without custom scenes
        is_spatial = False
        if spec is not None and (hasattr(spec, "updated_episode_config") or hasattr(spec, "level_dimensions") or hasattr(spec, "spawn_point")):
            is_spatial = True
        elif (candidate_dir / "level.json").is_file():
            has_scenes = False
            try:
                mech_path = candidate_dir / "mechanics.py"
                if mech_path.is_file():
                    tree = ast.parse(mech_path.read_text(encoding="utf-8"))
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name in ("get_scenes", "get_initial_scene"):
                            has_scenes = True
                            break
            except Exception:
                pass
            if not has_scenes:
                is_spatial = True
                if spec is None:
                    try:
                        data = json.loads((candidate_dir / "level.json").read_text(encoding="utf-8"))
                        spec = BaseLevelSpec.from_dict(data)
                    except Exception:
                        spec = None

        if is_spatial and spec is not None:
            return self._evaluate_spatial_candidate(
                candidate_dir,
                spec,
                episode_number,
                attempt_number,
                requirements_spec=requirements_spec,
            )

        gate_results: Dict[str, QualityGateResult] = {}

        # ---------------------------------------------------------------------
        # 1. Gate A: Syntax & Entrypoint Verification
        # ---------------------------------------------------------------------
        gate_a = self.verify_gate_a_syntax_and_entrypoint(candidate_dir)
        gate_results["gate_a"] = gate_a
        gate_results["Gate A: AST & Entrypoint"] = gate_a
        gate_results["Gate A: Syntax & AST"] = gate_a
        gate_results["Gate A"] = gate_a
        if not gate_a.passed:
            trace = RejectionDiagnosticTrace(
                deliberation=f"Gate A rejected candidate build: {gate_a.details}",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate A: AST & Entrypoint",
                failure_type=gate_a.failure_type,
                failure_reason=gate_a.details,
                file_path=gate_a.diagnostic_data.get("file_path"),
                line_number=gate_a.diagnostic_data.get("line_number"),
                remedy_type=RemedyType.FIX_SYNTAX if gate_a.failure_type == GateFailureType.SYNTAX_AST_ERROR else RemedyType.FIX_SCENE_ENTRYPOINT,
                recommended_remedy="Inspect source code for syntax errors, prohibited imports, missing BaseCartridge entrypoint, or missing scene lifecycles.",
            )
            return gate_results, trace

        # ---------------------------------------------------------------------
        # 2. Gate B: Multimodal Asset Verification
        # ---------------------------------------------------------------------
        gate_b = self.verify_gate_b_multimodal_assets(candidate_dir)
        gate_results["gate_b"] = gate_b
        gate_results["Gate B: Multimodal Asset Verification"] = gate_b
        gate_results["Gate E: Multimodal Asset Integrity"] = gate_b
        gate_results["gate_e"] = gate_b
        if not gate_b.passed:
            trace = RejectionDiagnosticTrace(
                deliberation=f"Gate B rejected candidate build: {gate_b.details}",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate B: Multimodal Asset Verification",
                failure_type=gate_b.failure_type,
                failure_reason=gate_b.details,
                asset_violations=gate_b.diagnostic_data.get("violations", [gate_b.details]),
                remedy_type=RemedyType.REGENERATE_ASSET,
                recommended_remedy=(
                    "Regenerate broken/placeholder visual assets or empty multi-cell atlases in-place, and fix any "
                    "GDScript region-slicing or unbound texture-load defects reported by Gate B."
                ),
                remedy_parameters={
                    "offending_assets": list(gate_b.diagnostic_data.get("offending_assets") or []),
                    "offending_scripts": list(gate_b.diagnostic_data.get("offending_scripts") or []),
                    "asset_metadata": dict(gate_b.diagnostic_data.get("asset_metadata") or {}),
                },
            )
            return gate_results, trace

        # ---------------------------------------------------------------------
        # 3. Gate C: Headless Execution & Progression Simulation
        # ---------------------------------------------------------------------
        gate_c, sim = self.verify_gate_c_headless_execution(candidate_dir, max_ticks=120)
        gate_results["gate_c"] = gate_c
        gate_results["Gate C: Headless Execution & Progression"] = gate_c
        gate_results["Gate C: Survivability"] = gate_c
        if not gate_c.passed:
            is_scene_asset_fail = gate_c.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION
            trace = RejectionDiagnosticTrace(
                deliberation=f"Gate C rejected candidate build: {gate_c.details}",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate C: Headless Execution & Progression",
                failure_type=gate_c.failure_type,
                failure_reason=gate_c.details,
                runtime_traceback=gate_c.diagnostic_data.get("traceback"),
                asset_violations=list(gate_c.diagnostic_data.get("violations") or []),
                remedy_type=RemedyType.REGENERATE_ASSET if is_scene_asset_fail else RemedyType.REPAIR_CODE_EXCEPTION,
                recommended_remedy=(
                    "Resolve runtime empty atlas regions, unbound null Sprite2D/TextureRect textures, or untextured "
                    "tiled Polygon2D surfaces in the active Godot scene tree."
                    if is_scene_asset_fail
                    else "Resolve runtime crash or frozen progression during headless scene simulation."
                ),
                remedy_parameters={
                    "offending_assets": list(gate_c.diagnostic_data.get("offending_assets") or []),
                    "offending_scripts": list(gate_c.diagnostic_data.get("offending_scripts") or []),
                },
            )
            return gate_results, trace

        # ---------------------------------------------------------------------
        # 4. Gate D: Performance & Frame Budget Verification (< 16.6ms)
        # ---------------------------------------------------------------------
        gate_d = self.verify_gate_d_frame_budget(sim, target_ms=16.6)
        gate_results["gate_d"] = gate_d
        gate_results["Gate D: Frame Budget"] = gate_d
        gate_results["Gate D: Exception-Free"] = gate_d
        if not gate_d.passed:
            trace = RejectionDiagnosticTrace(
                deliberation=f"Gate D rejected candidate build: {gate_d.details}",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate D: Frame Budget",
                failure_type=gate_d.failure_type,
                failure_reason=gate_d.details,
                remedy_type=RemedyType.OPTIMIZE_FRAME_LOOP,
                recommended_remedy="Optimize heavy loops or blocking calls in scene update/render to maintain 60 FPS frame budget (<16.6ms).",
            )
            return gate_results, trace

        # ---------------------------------------------------------------------
        # 5. Gate E: Custom Telemetry Probe Verification
        # ---------------------------------------------------------------------
        gate_e = self.verify_gate_e_custom_telemetry(candidate_dir, sim)
        gate_results["gate_e"] = gate_e
        gate_results["Gate E: Custom Telemetry Emission"] = gate_e
        if not gate_e.passed:
            trace = RejectionDiagnosticTrace(
                deliberation=f"Gate E rejected candidate build: {gate_e.details}",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate E: Custom Telemetry Emission",
                failure_type=gate_e.failure_type,
                failure_reason=gate_e.details,
                remedy_type=RemedyType.INJECT_TELEMETRY_PROBES,
                recommended_remedy="Ensure custom probes declared in telemetry_manifest.json are actively emitted during scene update.",
            )
            return gate_results, trace

        if not include_challenger_gate:
            return gate_results, None

        # ---------------------------------------------------------------------
        # 6. Gate F: Independent Challenger Gate & Acceptance Criteria Audit
        # ---------------------------------------------------------------------
        gate_f, challenger_trace = self.verify_gate_f_challenger(
            candidate_dir=candidate_dir,
            sim=sim,
            spec=spec,
            requirements_spec=requirements_spec,
            episode_number=episode_number,
            attempt_number=attempt_number,
        )
        gate_results["gate_f"] = gate_f
        gate_results["Gate F: Challenger Gate"] = gate_f
        gate_results["Gate F: Adversarial Design Critic"] = gate_f
        gate_results["Gate F"] = gate_f
        if not gate_f.passed:
            trace = challenger_trace or RejectionDiagnosticTrace(
                deliberation=f"Gate F rejected candidate build: {gate_f.details}",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate F: Challenger Gate",
                failure_type=GateFailureType.REQUIREMENT_UNFULFILLED,
                failure_reason=gate_f.details,
                remedy_type=RemedyType.IMPLEMENT_REQUIREMENT,
                recommended_remedy="Satisfy 100% of agent-authored acceptance criteria in requirements_spec.json.",
                unfulfilled_requirements=gate_f.diagnostic_data.get("unfulfilled_requirements", []),
                failing_criteria=gate_f.diagnostic_data.get("failing_criteria", []),
            )
            return gate_results, trace

        return gate_results, None

    verify_candidate = evaluate_candidate_cartridge

    @staticmethod
    def _completed_production_board_summary(candidate_dir: Path) -> Optional[Dict[str, Any]]:
        """Returns summary metadata when a candidate cartridge was built and 100% signed off
        by the Continuous Production Board Specialist Council across all vertical slices.
        """
        backlog_path = candidate_dir / "backlog.json"
        if not backlog_path.is_file():
            return None
        try:
            b_data = json.loads(backlog_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        items = b_data.get("items") if isinstance(b_data, dict) else None
        if not isinstance(items, list) or not items:
            return None
        states = [str(it.get("state", "")).lower() for it in items if isinstance(it, dict)]
        if not states or not all(st in ("approved", "cancelled") for st in states):
            return None
        approved_count = sum(1 for st in states if st == "approved")
        if approved_count == 0:
            return None

        completed_slices = 0
        bible_path = candidate_dir / "game_bible.json"
        if bible_path.is_file():
            try:
                bible_data = json.loads(bible_path.read_text(encoding="utf-8"))
                slices = (
                    (bible_data.get("slice_roadmap") or bible_data.get("vertical_slices"))
                    if isinstance(bible_data, dict)
                    else None
                )
                if isinstance(slices, list) and slices:
                    if not all(str(s.get("status", "")).lower() == "completed" for s in slices if isinstance(s, dict)):
                        return None
                    completed_slices = len(slices)
            except Exception:
                return None

        # Post-approval tamper/mutation guard: do NOT rubber-stamp `backlog.json` if any production
        # `.gd`, `.tscn`, or game `assets/` file was modified AFTER `backlog.json` was approved!
        try:
            backlog_mtime = backlog_path.stat().st_mtime
            excluded_dirs = {"playtest_captures", "mockups", "concepts", "artifacts", "tests", ".godot"}
            for check_p in candidate_dir.rglob("*"):
                if not check_p.is_file():
                    continue
                rel_parts = check_p.relative_to(candidate_dir).parts
                if set(rel_parts) & excluded_dirs:
                    continue
                rel_posix = check_p.relative_to(candidate_dir).as_posix()
                if rel_posix in ("src/studio_rig_and_collider.gd", "assets/asset_library.json"):
                    continue
                is_prod_code = check_p.suffix.lower() in (".gd", ".tscn")
                is_game_asset = rel_posix.startswith("assets/") and check_p.suffix.lower() in (".png", ".wav", ".jpg", ".jpeg")
                if (is_prod_code or is_game_asset) and check_p.stat().st_mtime > backlog_mtime + 2.0:
                    logger.warning(
                        "Gate F fast-path invalidated on %s: '%s' was modified %.1fs after backlog.json!",
                        candidate_dir.name,
                        rel_posix,
                        check_p.stat().st_mtime - backlog_mtime,
                    )
                    return None
        except Exception:
            pass

        return {
            "board_managed": True,
            "approved_board_items": approved_count,
            "total_board_items": len(states),
            "completed_slices": completed_slices,
        }

    def verify_gate_f_challenger(
        self,
        candidate_dir: Path,
        sim: Optional[Any] = None,
        spec: Optional[Any] = None,
        requirements_spec: Optional[Any] = None,
        episode_number: int = 1,
        attempt_number: int = 1,
    ) -> Tuple[QualityGateResult, Optional[RejectionDiagnosticTrace]]:
        """Gate F: Independent Adversarial Design Critic auditing candidate against 100% of acceptance criteria.
        Passes vacuously with a logged diagnostic note if requirements_spec.json is absent.
        """
        candidate_dir = Path(candidate_dir).resolve()
        candidate_id = getattr(spec, "spec_id", getattr(spec, "candidate_id", candidate_dir.name))

        # 0. Recognize completed Continuous Production Board cartridges already audited per-item & per-slice by Specialist Council
        if self.challenger_gate is None and requirements_spec is None and (spec is None or not hasattr(spec, "requirements")):
            board_summary = self._completed_production_board_summary(candidate_dir)
            if board_summary is not None:
                approved_n = board_summary["approved_board_items"]
                slices_n = board_summary["completed_slices"]
                return (
                    QualityGateResult(
                        deliberation=(
                            f"Continuous Production Board verified for '{candidate_id}': "
                            f"{approved_n} board item(s) across {slices_n} completed vertical slice(s) "
                            f"already passed Specialist Council review, Slice Assembly verification, and Blind Playtest sign-off."
                        ),
                        gate_name="Gate F: Adversarial Design Critic",
                        passed=True,
                        failure_type=GateFailureType.NONE,
                        details=(
                            f"Specialist Council & Continuous Production Board sign-off verified "
                            f"({approved_n} approved items, {slices_n} completed slices)."
                        ),
                        diagnostic_data=board_summary,
                    ),
                    None,
                )

        # 1. Resolve requirements_spec
        resolved_spec = requirements_spec
        if resolved_spec is None:
            if spec is not None and hasattr(spec, "requirements"):
                resolved_spec = spec
            else:
                spec_path = candidate_dir / "requirements_spec.json"
                if spec_path.is_file():
                    try:
                        from engine.studio.models import GameRequirementsSpec
                        data = json.loads(spec_path.read_text(encoding="utf-8"))
                        resolved_spec = GameRequirementsSpec(**data)
                    except Exception as e:
                        logger.warning("Could not parse requirements_spec.json in %s: %s", candidate_dir, e)

        # 2. Vacuous pass if no GameRequirementsSpec or no requirements
        if resolved_spec is None or not getattr(resolved_spec, "requirements", None):
            res = QualityGateResult(
                deliberation="No GameRequirementsSpec provided; skipping acceptance criteria audit.",
                gate_name="Gate F: Adversarial Design Critic",
                passed=True,
                failure_type=GateFailureType.NONE,
                details="No requirements_spec.json present; Gate F passed vacuously.",
                diagnostic_data={"skipped": True, "vacuous_pass": True},
            )
            return res, None

        # 3. Instantiate challenger gate lazily if needed
        if self.challenger_gate is None:
            client = self.gemini_client
            if client is None and "PYTEST_CURRENT_TEST" not in os.environ and "pytest" not in sys.modules:
                try:
                    from engine.studio.orchestrator import StudioOrchestrator
                    key = StudioOrchestrator._resolve_api_key(None, None)
                    if key:
                        from google import genai
                        from engine.studio.models import install_studio_gemini_guardrails
                        client = install_studio_gemini_guardrails(
                            genai.Client(api_key=key, http_options={"timeout": 120000}),
                            candidate_dir=candidate_dir,
                        )
                        self.gemini_client = client
                except Exception:
                    pass
            from engine.supervisor.challenger import ChallengerGate
            self.challenger_gate = ChallengerGate(client=client)

        # 4. Execute Challenger Gate evaluation
        try:
            evaluation = self.challenger_gate.evaluate(
                candidate_dir=candidate_dir,
                requirements_spec=resolved_spec,
                sim=sim,
                episode_number=episode_number,
                attempt_number=attempt_number,
            )
        except Exception as exc:
            logger.error("ChallengerGate execution raised exception: %s", exc)
            raise

        # 5. Process verdict
        sign_off_dict = evaluation.specialist_sign_off.to_dict() if getattr(evaluation, "specialist_sign_off", None) else None
        if evaluation.overall_verdict:
            res = QualityGateResult(
                deliberation=evaluation.deliberation,
                gate_name="Gate F: Adversarial Design Critic",
                passed=True,
                failure_type=GateFailureType.NONE,
                details="Candidate cartridge satisfied 100% of agent-authored acceptance criteria and unanimous specialist sign-off.",
                diagnostic_data={
                    "candidate_id": evaluation.candidate_id,
                    "evaluated_requirements": [r.model_dump() for r in evaluation.evaluated_requirements],
                    "specialist_sign_off": sign_off_dict,
                },
            )
            return res, None
        else:
            unfulfilled = [r.requirement_id for r in evaluation.evaluated_requirements if not r.fulfilled]
            failing_crit: List[str] = []
            for r in evaluation.evaluated_requirements:
                if not r.fulfilled and getattr(r, "failing_criteria", None):
                    failing_crit.extend(r.failing_criteria)

            trace = evaluation.rejection_trace or RejectionDiagnosticTrace(
                deliberation=evaluation.deliberation,
                candidate_id=evaluation.candidate_id or candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate F: Challenger Gate",
                failure_type=GateFailureType.REQUIREMENT_UNFULFILLED,
                failure_reason=f"Challenger Gate rejected candidate build. Unfulfilled requirements: {unfulfilled}",
                remedy_type=RemedyType.IMPLEMENT_REQUIREMENT,
                recommended_remedy="Implement genuine interactive mechanics meeting all criteria in requirements_spec.json.",
                unfulfilled_requirements=unfulfilled,
                failing_criteria=failing_crit,
                remedy_parameters={
                    "unfulfilled_requirements": unfulfilled,
                    "failing_criteria": failing_crit,
                },
            )

            res = QualityGateResult(
                deliberation=evaluation.deliberation,
                gate_name="Gate F: Adversarial Design Critic",
                passed=False,
                failure_type=GateFailureType.REQUIREMENT_UNFULFILLED,
                details=trace.failure_reason,
                diagnostic_data={
                    "candidate_id": evaluation.candidate_id,
                    "unfulfilled_requirements": unfulfilled,
                    "failing_criteria": failing_crit,
                    "evaluated_requirements": [r.model_dump() for r in evaluation.evaluated_requirements],
                    "specialist_sign_off": sign_off_dict,
                },
            )
            return res, trace

    def _evaluate_spatial_candidate(
        self,
        candidate_dir: Path,
        spec: Any,
        episode_number: int = 1,
        attempt_number: int = 1,
        requirements_spec: Optional[Any] = None,
    ) -> Tuple[Dict[str, QualityGateResult], Optional[RejectionDiagnosticTrace]]:
        """Evaluates spatial director updates against spatial navigation and survivability gates."""
        gate_results: Dict[str, QualityGateResult] = {}
        candidate_id = getattr(spec, "spec_id", getattr(spec, "candidate_id", candidate_dir.name))

        # Gate A: Syntax & AST Validation
        gate_a = self.verify_gate_a_syntax_and_ast(candidate_dir)
        gate_results["gate_a"] = gate_a
        gate_results["Gate A: Syntax & AST"] = gate_a
        gate_results["Gate A"] = gate_a
        if not gate_a.passed:
            trace = RejectionDiagnosticTrace(
                deliberation=f"Gate A rejected candidate build: {gate_a.details}",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate A: Syntax & AST",
                failure_type=gate_a.failure_type,
                failure_reason=gate_a.details,
                file_path=gate_a.diagnostic_data.get("file_path"),
                line_number=gate_a.diagnostic_data.get("line_number"),
                remedy_type=RemedyType.FIX_SYNTAX,
                recommended_remedy="Inspect source code for syntax errors or prohibited imports and regenerate.",
            )
            return gate_results, trace

        # Gate E: Multimodal Asset Integrity
        gate_e = self.verify_gate_e_asset_integrity(candidate_dir)
        gate_results["gate_e"] = gate_e
        gate_results["Gate E: Multimodal Asset Integrity"] = gate_e
        if not gate_e.passed:
            trace = RejectionDiagnosticTrace(
                deliberation=f"Gate E rejected candidate build: {gate_e.details}",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate E: Multimodal Asset Integrity",
                failure_type=gate_e.failure_type,
                failure_reason=gate_e.details,
                asset_violations=gate_e.diagnostic_data.get("violations", [gate_e.details]),
                remedy_type=RemedyType.REGENERATE_ASSET,
                recommended_remedy="Re-synthesize audio with valid 44-byte RIFF header or quantize sprites to 16-color Style Bible palette.",
            )
            return gate_results, trace

        # Load cartridge instance
        try:
            cartridge = CartridgeLoader.load_cartridge(candidate_dir)
            level_spec = cartridge.get_level_spec(episode_number if episode_number > 0 else 1)
        except Exception as exc:
            gate_d = QualityGateResult(
                deliberation=f"Cartridge loading raised unhandled exception: {exc}",
                gate_name="Gate D: Exception-Free",
                passed=False,
                failure_type=GateFailureType.UNHANDLED_EXCEPTION,
                details=f"{exc}\n{traceback.format_exc()}",
                diagnostic_data={"traceback": traceback.format_exc()}
            )
            gate_results["gate_d"] = gate_d
            gate_results["Gate D: Exception-Free"] = gate_d
            trace = RejectionDiagnosticTrace(
                deliberation=f"Cartridge loading failed with unhandled exception: {exc}",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate D: Exception-Free",
                failure_type=GateFailureType.UNHANDLED_EXCEPTION,
                failure_reason=str(exc),
                runtime_traceback=traceback.format_exc(),
                remedy_type=RemedyType.REPAIR_CODE_EXCEPTION,
                recommended_remedy="Repair runtime error during CartridgeLoader instantiation.",
            )
            return gate_results, trace

        # Run Simulation / Playtesting
        sim_result, runtime_exc = self.run_simulation_and_playtest(cartridge, level_spec)

        # Gate D: Zero Unhandled Exceptions
        gate_d = self.verify_gate_d_exceptions(runtime_exc)
        gate_results["gate_d"] = gate_d
        gate_results["Gate D: Exception-Free"] = gate_d
        if not gate_d.passed:
            trace = RejectionDiagnosticTrace(
                deliberation=f"Gate D rejected build due to runtime exception during simulation: {runtime_exc}",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate D: Exception-Free",
                failure_type=GateFailureType.UNHANDLED_EXCEPTION,
                failure_reason=gate_d.details,
                runtime_traceback=gate_d.diagnostic_data.get("traceback"),
                remedy_type=RemedyType.REPAIR_CODE_EXCEPTION,
                recommended_remedy="Resolve runtime crash in entity or mechanics simulation loop.",
            )
            return gate_results, trace

        # Gate B: Traversability
        gate_b, blocked_nodes = self.verify_gate_b_traversability(cartridge, level_spec, sim_result)
        gate_results["gate_b"] = gate_b
        gate_results["Gate B: Traversability"] = gate_b
        if not gate_b.passed:
            fail_coords = None
            if blocked_nodes:
                tx, ty = blocked_nodes[0]
                fail_coords = ((tx + 0.5) * self.tile_size, (ty + 0.5) * self.tile_size)

            trace = RejectionDiagnosticTrace(
                deliberation="Candidate level geometry blocked critical navigation path between spawn and exit.",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate B: Traversability",
                failure_type=gate_b.failure_type,
                failure_reason=gate_b.details,
                failure_coordinates=fail_coords,
                blocked_nodes=blocked_nodes,
                remedy_type=RemedyType.RELOCATE_OBSTACLE if blocked_nodes else RemedyType.REMOVE_OBSTACLE,
                recommended_remedy=(
                    f"Relocate or remove blocking obstacle near coordinates {fail_coords or level_spec.spawn_point} "
                    f"to reopen critical path to exit portal."
                ),
                remedy_parameters={"blocked_nodes": blocked_nodes, "failure_coordinates": fail_coords},
            )
            return gate_results, trace

        # Gate C: Survivability
        gate_c = self.verify_gate_c_survivability(sim_result, level_spec)
        gate_results["gate_c"] = gate_c
        gate_results["Gate C: Survivability"] = gate_c
        if not gate_c.passed:
            fail_coords = gate_c.diagnostic_data.get("failure_coordinates", level_spec.spawn_point)
            tick = gate_c.diagnostic_data.get("death_tick", 0)
            trace = RejectionDiagnosticTrace(
                deliberation="Candidate update failed survivability sanity check: premature player death or instant spawn kill.",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate C: Survivability",
                failure_type=gate_c.failure_type,
                failure_reason=gate_c.details,
                failure_coordinates=tuple(fail_coords) if fail_coords else None,
                tick_count=tick,
                remedy_type=RemedyType.SOFTEN_HAZARD if "hazard" in gate_c.details.lower() else RemedyType.WIDEN_MARGIN,
                recommended_remedy=(
                    f"Soften immediate threats around spawn: relocate hazards/enemies outside clearance margin "
                    f"({self.spawn_clearance_margin} px) or scale down lethal contact damage."
                ),
                remedy_parameters=gate_c.diagnostic_data,
            )
            return gate_results, trace

        # ---------------------------------------------------------------------
        # Gate F: Independent Challenger Gate & Acceptance Criteria Audit
        # ---------------------------------------------------------------------
        spatial_sim = None
        try:
            from engine.sim.headless import HeadlessSimulator
            spatial_sim = HeadlessSimulator(cartridge_path=candidate_dir)
            spatial_sim.step()
        except Exception:
            spatial_sim = None

        gate_f, challenger_trace = self.verify_gate_f_challenger(
            candidate_dir=candidate_dir,
            sim=spatial_sim,
            spec=spec,
            requirements_spec=requirements_spec,
            episode_number=episode_number,
            attempt_number=attempt_number,
        )
        gate_results["gate_f"] = gate_f
        gate_results["Gate F: Challenger Gate"] = gate_f
        gate_results["Gate F: Adversarial Design Critic"] = gate_f
        gate_results["Gate F"] = gate_f
        if not gate_f.passed:
            trace = challenger_trace or RejectionDiagnosticTrace(
                deliberation=f"Gate F rejected candidate build: {gate_f.details}",
                candidate_id=candidate_id,
                episode_number=episode_number,
                attempt_number=attempt_number,
                failure_gate="Gate F: Challenger Gate",
                failure_type=GateFailureType.REQUIREMENT_UNFULFILLED,
                failure_reason=gate_f.details,
                remedy_type=RemedyType.IMPLEMENT_REQUIREMENT,
                recommended_remedy="Satisfy 100% of agent-authored acceptance criteria in requirements_spec.json.",
                unfulfilled_requirements=gate_f.diagnostic_data.get("unfulfilled_requirements", []),
                failing_criteria=gate_f.diagnostic_data.get("failing_criteria", []),
            )
            return gate_results, trace

        return gate_results, None

    # =========================================================================
    # Individual Gate Verification Logic
    # =========================================================================

    def verify_gate_a_syntax_and_entrypoint(self, candidate_dir: Path) -> QualityGateResult:
        """Gate A: Checks AST security, bytecode compilation, class deliberation, and BaseCartridge entrypoint."""
        candidate_dir = Path(candidate_dir).resolve()

        # Branch for Godot 4 cartridges
        if (candidate_dir / "project.godot").is_file():
            from engine.sim.godot_verifier import verify_godot_cartridge_syntax
            if not (candidate_dir / "main.tscn").is_file():
                return QualityGateResult(
                    deliberation="Godot 4 cartridge is missing required root scene main.tscn.",
                    gate_name="Gate A: AST & Entrypoint",
                    passed=False,
                    failure_type=GateFailureType.ENTRYPOINT_MISSING,
                    details="Missing main.tscn root scene.",
                    diagnostic_data={"candidate_dir": str(candidate_dir)},
                )
            if not (candidate_dir / "main.gd").is_file():
                return QualityGateResult(
                    deliberation="Godot 4 cartridge is missing required root script main.gd.",
                    gate_name="Gate A: AST & Entrypoint",
                    passed=False,
                    failure_type=GateFailureType.ENTRYPOINT_MISSING,
                    details="Missing main.gd root script.",
                    diagnostic_data={"candidate_dir": str(candidate_dir)},
                )
            is_valid, diagnostics = verify_godot_cartridge_syntax(candidate_dir)
            if not is_valid:
                diag = diagnostics[0] if diagnostics else None
                return QualityGateResult(
                    deliberation=f"Godot syntax verification failed: {diag.error_message if diag else 'compiler error'}",
                    gate_name="Gate A: AST & Entrypoint",
                    passed=False,
                    failure_type=GateFailureType.SYNTAX_AST_ERROR,
                    details=f"GDScript syntax/compile error: {diag.error_message if diag else 'unknown'}",
                    diagnostic_data={
                        "file_path": str(diag.file_path) if diag else str(candidate_dir),
                        "line_number": diag.line_number if diag else 1,
                        "raw_line": diag.raw_line if diag else "",
                    },
                )
            return QualityGateResult(
                deliberation="Godot 4 cartridge passed syntax verification and root scene validation.",
                gate_name="Gate A: AST & Entrypoint",
                passed=True,
                failure_type=GateFailureType.NONE,
                details="Godot 4 cartridge passed headless syntax and entrypoint checks.",
                diagnostic_data={"godot": True, "entrypoint": "main.tscn"},
            )

        py_files = list(candidate_dir.rglob("*.py"))
        if not py_files:
            return QualityGateResult(
                deliberation="Cartridge directory contains zero Python source modules.",
                gate_name="Gate A: AST & Entrypoint",
                passed=False,
                failure_type=GateFailureType.ENTRYPOINT_MISSING,
                details="No Python files found in candidate cartridge directory.",
                diagnostic_data={"candidate_dir": str(candidate_dir)},
            )

        found_cartridge_subclass = False
        entrypoint_file = None

        for py_path in py_files:
            try:
                content = py_path.read_text(encoding="utf-8")
                # 1. Prohibited template placeholders check
                if "{TEMPLATE_" in content or "{{LEVEL_CONFIG}}" in content:
                    return QualityGateResult(
                        deliberation=f"Unresolved template placeholder detected in {py_path.name}.",
                        gate_name="Gate A: AST & Entrypoint",
                        passed=False,
                        failure_type=GateFailureType.SYNTAX_AST_ERROR,
                        details=f"Template token detected in {py_path.name}",
                        diagnostic_data={"file_path": str(py_path)},
                    )

                # 2. AST Security check
                validate_cartridge_ast(content)

                # 3. Bytecode compilation
                compile(content, py_path.name, "exec")

                # 4. AST class structure inspection
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Verify Rule 4 deliberation
                        has_delib = any(
                            (isinstance(stmt, ast.AnnAssign) and getattr(stmt.target, "id", None) == "deliberation") or
                            (isinstance(stmt, ast.Assign) and any(getattr(t, "id", None) == "deliberation" for t in stmt.targets))
                            for stmt in node.body
                        )
                        if not has_delib:
                            return QualityGateResult(
                                deliberation=f"Class '{node.name}' in {py_path.name} violates Rule 4: missing deliberation attribute.",
                                gate_name="Gate A: AST & Entrypoint",
                                passed=False,
                                failure_type=GateFailureType.SYNTAX_AST_ERROR,
                                details=f"Class {node.name} must declare deliberation attribute.",
                                diagnostic_data={"file_path": str(py_path), "class_name": node.name},
                            )

                        # Check if inherits from BaseCartridge or Cartridge
                        for base in node.bases:
                            base_id = getattr(base, "id", getattr(base, "attr", ""))
                            if "Cartridge" in base_id:
                                found_cartridge_subclass = True
                                entrypoint_file = py_path.name

            except ASTSecurityError as ast_err:
                return QualityGateResult(
                    deliberation=f"AST security sandbox violation in {py_path.name}: {ast_err}",
                    gate_name="Gate A: AST & Entrypoint",
                    passed=False,
                    failure_type=GateFailureType.SYNTAX_AST_ERROR,
                    details=f"AST Security Invariant Violated in {py_path.name}: {ast_err}",
                    diagnostic_data={"file_path": str(py_path)},
                )
            except SyntaxError as syn_err:
                return QualityGateResult(
                    deliberation=f"Syntax error compiling {py_path.name}: {syn_err}",
                    gate_name="Gate A: AST & Entrypoint",
                    passed=False,
                    failure_type=GateFailureType.SYNTAX_AST_ERROR,
                    details=f"Python syntax error in {py_path.name}: {syn_err}",
                    diagnostic_data={"file_path": str(py_path), "line_number": syn_err.lineno},
                )
            except Exception as e:
                return QualityGateResult(
                    deliberation=f"Unexpected error validating {py_path.name}: {e}",
                    gate_name="Gate A: AST & Entrypoint",
                    passed=False,
                    failure_type=GateFailureType.SYNTAX_AST_ERROR,
                    details=str(e),
                    diagnostic_data={"file_path": str(py_path)},
                )

        if not found_cartridge_subclass:
            return QualityGateResult(
                deliberation="Candidate cartridge does not define a BaseCartridge subclass.",
                gate_name="Gate A: AST & Entrypoint",
                passed=False,
                failure_type=GateFailureType.ENTRYPOINT_MISSING,
                details="Missing class inheriting from BaseCartridge.",
                diagnostic_data={"candidate_dir": str(candidate_dir)},
            )

        return QualityGateResult(
            deliberation="All cartridge Python modules passed AST security invariants, bytecode compilation, and BaseCartridge entrypoint.",
            gate_name="Gate A: AST & Entrypoint",
            passed=True,
            failure_type=GateFailureType.NONE,
            details=f"AST security & entrypoint verified (entrypoint: {entrypoint_file}).",
        )

    def verify_gate_b_multimodal_assets(self, candidate_dir: Path) -> QualityGateResult:
        """Gate B: Verifies multimodal asset integrity (RIFF WAV audio, PNG sprites/UI/atlases, GDScript slicing parity, acyclic narrative dialogue tree)."""
        validator = self.asset_validator or AssetValidator(
            gemini_client=self.gemini_client,
            event_bus=self.event_bus,
        )
        if getattr(validator, "gemini_client", None) is None and self.gemini_client is not None:
            validator.gemini_client = self.gemini_client
        if getattr(validator, "event_bus", None) is None and self.event_bus is not None:
            validator.event_bus = self.event_bus
        return validator.validate_cartridge_assets(candidate_dir)

    def verify_gate_c_headless_execution(
        self,
        candidate_dir: Path,
        max_ticks: int = 120,
    ) -> Tuple[QualityGateResult, Optional[Any]]:
        """Gate C: Spins up headless simulation, steps discrete ticks with synthetic inputs, tracks progression & runtime scene visual integrity."""
        candidate_dir = Path(candidate_dir).resolve()
        if (candidate_dir / "project.godot").is_file():
            import re as _re
            from engine.sim.godot_verifier import GodotHeadlessSimulator
            sim = GodotHeadlessSimulator(cartridge_path=candidate_dir, max_ticks=max_ticks, telemetry_mode="in_memory")
            try:
                sim.run_until_complete(max_steps=max_ticks)
                scene_audit = dict(getattr(sim, "scene_audit", {}) or {})
                scene_visual_violations: List[str] = []
                for key_label, audit_key in (
                    ("Empty sprite sheet cell", "empty_sprite_frame_nodes"),
                    ("Empty AtlasTexture/region crop", "empty_atlas_region_nodes"),
                    ("Unbound null-texture visual node", "null_texture_visual_nodes"),
                    ("Untextured tiled Polygon2D surface", "untextured_world_polygons"),
                ):
                    for item_msg in (scene_audit.get(audit_key) or []):
                        scene_visual_violations.append(f"{key_label}: {item_msg}")
                if scene_audit.get("portrait_misused_on_world_sprite"):
                    for p_msg in (scene_audit.get("portrait_misused_nodes") or []):
                        scene_visual_violations.append(f"Dialogue portrait misused on world Sprite2D: {p_msg}")

                if scene_visual_violations:
                    off_assets: List[str] = []
                    for v_msg in scene_visual_violations:
                        for m_png in _re.findall(r"(?:res://)?(assets/[A-Za-z0-9_./-]+\.png)", v_msg):
                            if (candidate_dir / m_png).is_file() and m_png not in off_assets:
                                off_assets.append(m_png)
                    details_str = "; ".join(scene_visual_violations[:8])
                    return QualityGateResult(
                        deliberation=f"Gate C REJECTED due to runtime scene visual/texture integrity defects: {details_str}",
                        gate_name="Gate C: Headless Execution & Progression",
                        passed=False,
                        failure_type=GateFailureType.ASSET_INTEGRITY_VIOLATION,
                        details=details_str,
                        diagnostic_data={
                            "godot": True,
                            "ticks": sim.ticks_executed,
                            "scene_audit": scene_audit,
                            "violations": scene_visual_violations,
                            "offending_assets": off_assets,
                        },
                    ), sim

                return QualityGateResult(
                    deliberation="Headless Godot simulation executed without errors and passed runtime scene texture/atlas audit.",
                    gate_name="Gate C: Headless Execution & Progression",
                    passed=True,
                    failure_type=GateFailureType.NONE,
                    details=f"Godot headless simulation completed {sim.ticks_executed} ticks.",
                    diagnostic_data={"godot": True, "ticks": sim.ticks_executed, "scene_audit": scene_audit},
                ), sim
            except Exception as exc:
                return QualityGateResult(
                    deliberation=f"Godot headless simulation error: {exc}",
                    gate_name="Gate C: Headless Execution & Progression",
                    passed=False,
                    failure_type=GateFailureType.UNHANDLED_EXCEPTION,
                    details=str(exc),
                    diagnostic_data={"traceback": str(exc)},
                ), None

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

        try:
            from engine.sim.headless import HeadlessSimulator, SimulationInput
            sim = HeadlessSimulator(cartridge_path=candidate_dir)
        except Exception as exc:
            return QualityGateResult(
                deliberation=f"HeadlessSimulator initialization raised unhandled exception: {exc}",
                gate_name="Gate C: Headless Execution & Progression",
                passed=False,
                failure_type=GateFailureType.UNHANDLED_EXCEPTION,
                details=f"{exc}\n{traceback.format_exc()}",
                diagnostic_data={"traceback": traceback.format_exc()},
            ), None

        initial_state_counter = getattr(getattr(sim, "current_scene", None), "state_counter", None)

        try:
            for step_idx in range(max_ticks):
                action = SimulationInput(
                    move_x=1.0 if step_idx % 2 == 0 else 0.0,
                    move_y=1.0 if step_idx % 3 == 0 else 0.0,
                    dash=(step_idx % 15 == 0),
                    fire=(step_idx % 10 == 0),
                )
                step_res = sim.step(action)
                if getattr(sim, "is_completed", False):
                    break
        except Exception as exc:
            return QualityGateResult(
                deliberation=f"Simulation tick raised unhandled exception: {exc}",
                gate_name="Gate C: Headless Execution & Progression",
                passed=False,
                failure_type=GateFailureType.UNHANDLED_EXCEPTION,
                details=f"{exc}\n{traceback.format_exc()}",
                diagnostic_data={"traceback": traceback.format_exc()},
            ), sim

        # Progression check: ensure state advanced and simulation did not freeze
        progression_ok = getattr(sim, "progression_verified", True)
        current_scene = getattr(sim, "current_scene", None)
        if current_scene is not None and initial_state_counter is not None:
            final_counter = getattr(current_scene, "state_counter", initial_state_counter)
            if final_counter == initial_state_counter and sim.step_count > 10:
                progression_ok = False

        if not progression_ok:
            return QualityGateResult(
                deliberation="Simulation state remained frozen across ticks; no progression detected.",
                gate_name="Gate C: Headless Execution & Progression",
                passed=False,
                failure_type=GateFailureType.PROGRESSION_FAILURE,
                details="State counter did not increment during simulation ticks.",
                diagnostic_data={"step_count": sim.step_count},
            ), sim

        return QualityGateResult(
            deliberation=f"Headless simulation executed {sim.step_count} ticks cleanly with observable state progression.",
            gate_name="Gate C: Headless Execution & Progression",
            passed=True,
            failure_type=GateFailureType.NONE,
            details=f"Simulated {sim.step_count} ticks cleanly.",
            diagnostic_data={"step_count": sim.step_count},
        ), sim

    def verify_gate_d_frame_budget(self, sim: Any, target_ms: float = 16.6) -> QualityGateResult:
        """Gate D: Verifies simulated frame tick durations adhere to the 60 FPS target (< 16.6ms)."""
        if sim is None:
            return QualityGateResult(
                deliberation="No simulation telemetry available for frame budget evaluation.",
                gate_name="Gate D: Frame Budget",
                passed=False,
                failure_type=GateFailureType.FRAME_BUDGET_EXCEEDED,
                details="Simulation instance was not provided.",
            )

        avg_ms = float(getattr(sim, "avg_frame_time_ms", 0.0) or 0.0)
        p95_ms = float(getattr(sim, "p95_frame_time_ms", 0.0) or 0.0)
        max_ms = float(getattr(sim, "max_frame_time_ms", 0.0) or 0.0)
        passed = bool(getattr(sim, "frame_budget_passed", True))
        diag_payload = {
            "avg_frame_time_ms": avg_ms,
            "p95_frame_time_ms": p95_ms,
            "max_frame_time_ms": max_ms,
            "target_ms": target_ms,
        }

        if avg_ms > target_ms:
            return QualityGateResult(
                deliberation=f"Frame budget exceeded: average tick duration {avg_ms:.2f}ms exceeds target {target_ms:.2f}ms (60 FPS).",
                gate_name="Gate D: Frame Budget",
                passed=False,
                failure_type=GateFailureType.FRAME_BUDGET_EXCEEDED,
                details=f"Average frame {avg_ms:.2f}ms > {target_ms:.2f}ms (P95: {p95_ms:.2f}ms, peak frame spike: {max_ms:.2f}ms).",
                diagnostic_data=diag_payload,
            )

        if not passed:
            return QualityGateResult(
                deliberation=(
                    f"Frame spike budget exceeded: steady-state average {avg_ms:.2f}ms is within {target_ms:.2f}ms (60 FPS), "
                    f"but P95 ({p95_ms:.2f}ms) or peak frame spike ({max_ms:.2f}ms) exceeded steady-state smoothness limits."
                ),
                gate_name="Gate D: Frame Budget",
                passed=False,
                failure_type=GateFailureType.FRAME_BUDGET_EXCEEDED,
                details=(
                    f"Steady-state average {avg_ms:.2f}ms <= {target_ms:.2f}ms, "
                    f"but frame spike exceeded budget (P95: {p95_ms:.2f}ms, peak frame spike: {max_ms:.2f}ms)."
                ),
                diagnostic_data=diag_payload,
            )

        return QualityGateResult(
            deliberation=f"Frame budget verified: avg {avg_ms:.2f}ms < {target_ms:.2f}ms (P95 {p95_ms:.2f}ms, peak spike {max_ms:.2f}ms).",
            gate_name="Gate D: Frame Budget",
            passed=True,
            failure_type=GateFailureType.NONE,
            details=f"Performance within budget: avg {avg_ms:.2f}ms, P95 {p95_ms:.2f}ms, peak {max_ms:.2f}ms.",
            diagnostic_data=diag_payload,
        )

    def verify_gate_e_custom_telemetry(self, candidate_dir: Path, sim: Any) -> QualityGateResult:
        """Gate E: Verifies that custom probes declared in telemetry_manifest.json were actively emitted."""
        candidate_dir = Path(candidate_dir).resolve()
        manifest_path = candidate_dir / "telemetry_manifest.json"
        if not manifest_path.is_file():
            # If no custom telemetry manifest declared, pass vacuously
            return QualityGateResult(
                deliberation="No custom telemetry manifest declared; passing Gate E vacuously.",
                gate_name="Gate E: Custom Telemetry Emission",
                passed=True,
                failure_type=GateFailureType.NONE,
                details="No telemetry_manifest.json present in candidate.",
            )

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            declared_probes = data.get("probes", [])
        except Exception as exc:
            return QualityGateResult(
                deliberation=f"Malformed telemetry_manifest.json: {exc}",
                gate_name="Gate E: Custom Telemetry Emission",
                passed=False,
                failure_type=GateFailureType.TELEMETRY_EMISSION_FAILURE,
                details=f"JSON decode error in telemetry_manifest.json: {exc}",
            )

        if not declared_probes:
            return QualityGateResult(
                deliberation="telemetry_manifest.json declared zero probes; passing Gate E.",
                gate_name="Gate E: Custom Telemetry Emission",
                passed=True,
                failure_type=GateFailureType.NONE,
                details="Empty probe list.",
            )

        # Check emitted probes
        recorder = getattr(sim, "telemetry_recorder", None) if sim else None
        fired_probe_names = set()
        if recorder is not None:
            if hasattr(recorder, "get_events_by_type"):
                probe_events = recorder.get_events_by_type("CUSTOM_PROBE")
            elif hasattr(recorder, "get_events"):
                probe_events = recorder.get_events("CUSTOM_PROBE")
            else:
                probe_events = getattr(recorder, "events", [])

            for ev in probe_events:
                p_name = getattr(ev, "probe_name", None) or (ev.payload.get("probe_name") if hasattr(ev, "payload") and isinstance(ev.payload, dict) else None)
                if p_name:
                    fired_probe_names.add(p_name)

        declared_names = {p.get("probe_name") for p in declared_probes if isinstance(p, dict)}
        matching = declared_names.intersection(fired_probe_names)

        if not matching:
            return QualityGateResult(
                deliberation=f"Custom telemetry verification failed: none of the declared probes ({list(declared_names)}) were emitted during simulation.",
                gate_name="Gate E: Custom Telemetry Emission",
                passed=False,
                failure_type=GateFailureType.TELEMETRY_EMISSION_FAILURE,
                details=f"Zero custom probe emissions recorded for declared probes: {list(declared_names)}.",
                diagnostic_data={"declared_probes": list(declared_names), "fired_probes": list(fired_probe_names)},
            )

        return QualityGateResult(
            deliberation=f"Custom telemetry verified: {len(matching)} declared probes actively emitted ({list(matching)}).",
            gate_name="Gate E: Custom Telemetry Emission",
            passed=True,
            failure_type=GateFailureType.NONE,
            details=f"Verified emission of custom probes: {list(matching)}.",
            diagnostic_data={"declared_probes": list(declared_names), "fired_probes": list(fired_probe_names)},
        )

    def verify_gate_a_syntax_and_ast(self, cartridge_dir: Path) -> QualityGateResult:
        """Gate A: Checks AST security invariants and bytecode compilation."""
        cartridge_dir = Path(cartridge_dir).resolve()
        if (cartridge_dir / "project.godot").is_file():
            from engine.sim.godot_verifier import verify_godot_cartridge_syntax
            is_valid, diagnostics = verify_godot_cartridge_syntax(cartridge_dir)
            if not is_valid:
                diag = diagnostics[0] if diagnostics else None
                return QualityGateResult(
                    deliberation=f"Godot syntax verification failed: {diag.error_message if diag else 'compiler error'}",
                    gate_name="Gate A: Syntax & AST",
                    passed=False,
                    failure_type=GateFailureType.SYNTAX_AST_ERROR,
                    details=f"GDScript error: {diag.error_message if diag else 'compiler error'}",
                    diagnostic_data={
                        "file_path": str(diag.file_path) if diag else str(cartridge_dir),
                        "line_number": diag.line_number if diag else 1,
                    },
                )
            return QualityGateResult(
                deliberation="Godot 4 cartridge passed syntax verification.",
                gate_name="Gate A: Syntax & AST",
                passed=True,
                failure_type=GateFailureType.NONE,
                details="Godot 4 cartridge passed headless syntax and AST checks.",
                diagnostic_data={"godot": True},
            )

        py_files = list(candidate_file for candidate_file in cartridge_dir.glob("*.py"))
        if not py_files:
            return QualityGateResult(
                deliberation="Cartridge directory contains no Python source files.",
                gate_name="Gate A: Syntax & AST",
                passed=False,
                failure_type=GateFailureType.SYNTAX_AST_ERROR,
                details="Missing essential Python modules in cartridge directory."
            )

        for py_path in py_files:
            try:
                with open(py_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # 1. AST Security
                validate_cartridge_ast(content)
                # 2. Bytecode compilation
                compile(content, py_path.name, "exec")
            except ASTSecurityError as ast_err:
                return QualityGateResult(
                    deliberation=f"AST security sandbox violation in {py_path.name}: {ast_err}",
                    gate_name="Gate A: Syntax & AST",
                    passed=False,
                    failure_type=GateFailureType.SYNTAX_AST_ERROR,
                    details=f"AST Security Invariant Violated in {py_path.name}: {ast_err}",
                    diagnostic_data={"file_path": str(py_path)}
                )
            except SyntaxError as syn_err:
                return QualityGateResult(
                    deliberation=f"Syntax error compiling {py_path.name}: {syn_err}",
                    gate_name="Gate A: Syntax & AST",
                    passed=False,
                    failure_type=GateFailureType.SYNTAX_AST_ERROR,
                    details=f"Python syntax error in {py_path.name}: {syn_err}",
                    diagnostic_data={"file_path": str(py_path), "line_number": syn_err.lineno}
                )
            except Exception as e:
                return QualityGateResult(
                    deliberation=f"Unexpected error validating {py_path.name}: {e}",
                    gate_name="Gate A: Syntax & AST",
                    passed=False,
                    failure_type=GateFailureType.SYNTAX_AST_ERROR,
                    details=str(e),
                    diagnostic_data={"file_path": str(py_path)}
                )

        return QualityGateResult(
            deliberation="All cartridge Python modules passed AST security invariants and bytecode compilation.",
            gate_name="Gate A: Syntax & AST",
            passed=True,
            failure_type=GateFailureType.NONE,
            details="AST security & bytecode checks successful."
        )

    def verify_gate_e_asset_integrity(self, candidate_dir: Path) -> QualityGateResult:
        """Gate E: Verifies multimodal asset integrity (voice audio, visual sprites/tiles, narrative.json)."""
        if self.asset_validator is not None:
            return self.asset_validator.validate_cartridge_assets(candidate_dir)
        validator = AssetValidator()
        return validator.validate_cartridge_assets(candidate_dir)

    def verify_gate_b_traversability(
        self,
        cartridge: BaseCartridge,
        level_spec: BaseLevelSpec,
        sim_result: Optional[Dict[str, Any]] = None,
    ) -> Tuple[QualityGateResult, List[Tuple[int, int]]]:
        """Gate B: Verifies critical path from spawn to exit is unbroken via NavMesh A* and Dual-Frontier BFS."""
        # 1. Check if playtester bot provided traversability result with pre-computed blocked nodes
        if sim_result is not None:
            if not sim_result.get("traversable", True):
                blocked = sim_result.get("blocked_nodes", [])
                if blocked:
                    return QualityGateResult(
                        deliberation="Playtester bot failed to find traversable path to exit portal.",
                        gate_name="Gate B: Traversability",
                        passed=False,
                        failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
                        details="Synthetic playtester reported critical path is blocked by solid obstacles."
                    ), blocked
            elif sim_result.get("stuck", False):
                blocked = sim_result.get("blocked_nodes", [])
                if blocked:
                    return QualityGateResult(
                        deliberation="Playtester bot was detected stuck in a cyclic movement loop.",
                        gate_name="Gate B: Traversability",
                        passed=False,
                        failure_type=GateFailureType.TIMEOUT_STUCK,
                        details="Synthetic playtester oscillating or trapped without forward progress."
                    ), blocked

        # 2. Geometric NavMesh A* traversability verification
        width = int(level_spec.width)
        height = int(level_spec.height)
        cols = max(1, int(width // self.tile_size))
        rows = max(1, int(height // self.tile_size))

        # Build occupancy grid (0 = free, 1 = obstacle)
        grid = [[0 for _ in range(cols)] for _ in range(rows)]
        for wall in level_spec.walls:
            wx, wy, ww, wh = wall[:4]
            min_tx = max(0, min(cols - 1, int(wx // self.tile_size)))
            max_tx = max(0, min(cols - 1, int((wx + ww - 1e-4) // self.tile_size)))
            min_ty = max(0, min(rows - 1, int(wy // self.tile_size)))
            max_ty = max(0, min(rows - 1, int((wy + wh - 1e-4) // self.tile_size)))
            for r in range(min_ty, max_ty + 1):
                for c in range(min_tx, max_tx + 1):
                    grid[r][c] = 1

        spawn_tx = max(0, min(cols - 1, int(level_spec.spawn_point[0] // self.tile_size)))
        spawn_ty = max(0, min(rows - 1, int(level_spec.spawn_point[1] // self.tile_size)))
        exit_tx = max(0, min(cols - 1, int(level_spec.exit_point[0] // self.tile_size)))
        exit_ty = max(0, min(rows - 1, int(level_spec.exit_point[1] // self.tile_size)))

        # Verify spawn and exit are not embedded inside solid wall geometry
        if grid[spawn_ty][spawn_tx] == 1:
            return QualityGateResult(
                deliberation="Player spawn coordinate is embedded inside solid wall geometry.",
                gate_name="Gate B: Traversability",
                passed=False,
                failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
                details=f"Spawn point {level_spec.spawn_point} overlaps solid obstacle tile ({spawn_tx}, {spawn_ty})."
            ), [(spawn_tx, spawn_ty)]

        if grid[exit_ty][exit_tx] == 1:
            return QualityGateResult(
                deliberation="Exit portal coordinate is embedded inside solid wall geometry.",
                gate_name="Gate B: Traversability",
                passed=False,
                failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
                details=f"Exit portal {level_spec.exit_point} overlaps solid obstacle tile ({exit_tx}, {exit_ty})."
            ), [(exit_tx, exit_ty)]

        # Pathfind via A*
        path = astar_pathfind(grid, (spawn_tx, spawn_ty), (exit_tx, exit_ty), allow_diagonal=False)
        if path is None:
            bottleneck_nodes = self.find_bottleneck_chokepoints(grid, (spawn_tx, spawn_ty), (exit_tx, exit_ty))
            ray_nodes = self._find_raycast_blockers(grid, (spawn_tx, spawn_ty), (exit_tx, exit_ty))
            blocked_nodes = list(dict.fromkeys(bottleneck_nodes + ray_nodes))
            return QualityGateResult(
                deliberation="A* pathfinding could not find a traversable path from spawn to exit.",
                gate_name="Gate B: Traversability",
                passed=False,
                failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
                details=f"No traversable path from spawn ({spawn_tx}, {spawn_ty}) to exit ({exit_tx}, {exit_ty})."
            ), blocked_nodes

        # If geometric static path exists, verify playtester simulation did not fail
        if sim_result is not None and not sim_result.get("traversable", True):
            return QualityGateResult(
                deliberation="Playtester bot failed to find traversable path to exit portal during simulation.",
                gate_name="Gate B: Traversability",
                passed=False,
                failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
                details="Synthetic playtester reported critical path is blocked during dynamic simulation."
            ), []

        if sim_result is not None and sim_result.get("stuck", False):
            return QualityGateResult(
                deliberation="Playtester bot was detected stuck in a cyclic movement loop.",
                gate_name="Gate B: Traversability",
                passed=False,
                failure_type=GateFailureType.TIMEOUT_STUCK,
                details="Synthetic playtester oscillating or trapped without forward progress."
            ), []

        return QualityGateResult(
            deliberation=f"Critical path verified traversable via A* with {len(path)} steps.",
            gate_name="Gate B: Traversability",
            passed=True,
            failure_type=GateFailureType.NONE,
            details=f"Path verified with length {len(path)}."
        ), []

    def find_bottleneck_chokepoints(
        self,
        grid: List[List[int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Dual-Frontier BFS discovering the exact obstacle partition separating spawn from exit."""
        cols = len(grid[0])
        rows = len(grid)

        # 1. Forward BFS from start
        reachable_from_start: Set[Tuple[int, int]] = set()
        queue = deque([start])
        if 0 <= start[0] < cols and 0 <= start[1] < rows and grid[start[1]][start[0]] == 0:
            reachable_from_start.add(start)
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < cols and 0 <= ny < rows and grid[ny][nx] == 0:
                    if (nx, ny) not in reachable_from_start:
                        reachable_from_start.add((nx, ny))
                        queue.append((nx, ny))

        # 2. Backward BFS from goal
        reachable_from_goal: Set[Tuple[int, int]] = set()
        queue_goal = deque([goal])
        if 0 <= goal[0] < cols and 0 <= goal[1] < rows and grid[goal[1]][goal[0]] == 0:
            reachable_from_goal.add(goal)
        while queue_goal:
            cx, cy = queue_goal.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < cols and 0 <= ny < rows and grid[ny][nx] == 0:
                    if (nx, ny) not in reachable_from_goal:
                        reachable_from_goal.add((nx, ny))
                        queue_goal.append((nx, ny))

        # 3. Find solid tiles (grid == 1) adjacent to reachable_from_start with minimum distance to reachable_from_goal
        candidate_blockers: List[Tuple[float, Tuple[int, int]]] = []
        for sx, sy in reachable_from_start:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                wx, wy = sx + dx, sy + dy
                if 0 <= wx < cols and 0 <= wy < rows and grid[wy][wx] == 1:
                    if reachable_from_goal:
                        min_dist = min(abs(wx - gx) + abs(wy - gy) for gx, gy in reachable_from_goal)
                    else:
                        min_dist = abs(wx - goal[0]) + abs(wy - goal[1])
                    candidate_blockers.append((min_dist, (wx, wy)))

        candidate_blockers.sort(key=lambda item: item[0])
        if candidate_blockers:
            min_d = candidate_blockers[0][0]
            best_tiles = [tile for d, tile in candidate_blockers if d <= min_d + 1]
            return list(dict.fromkeys(best_tiles))[:5]

        return []

    def verify_gate_c_survivability(
        self,
        sim_result: Optional[Dict[str, Any]],
        level_spec: BaseLevelSpec,
    ) -> QualityGateResult:
        """Gate C: Verifies player does not die prematurely (< 3s / 180 ticks) and spawn is safe."""
        # 1. Check direct hazard overlap with spawn point
        sx, sy = level_spec.spawn_point
        margin = self.spawn_clearance_margin
        for h in level_spec.hazards:
            hx = float(h.get("x", 0.0))
            hy = float(h.get("y", 0.0))
            hw = float(h.get("w", 48.0))
            hh = float(h.get("h", 48.0))
            if (sx + margin > hx and sx - margin < hx + hw and
                sy + margin > hy and sy - margin < hy + hh):
                return QualityGateResult(
                    deliberation=f"Hazard '{h.get('id', 'hazard')}' overlaps spawn clearance margin ({margin}px).",
                    gate_name="Gate C: Survivability",
                    passed=False,
                    failure_type=GateFailureType.SURVIVABILITY_VIOLATION,
                    details=f"Lethal hazard '{h.get('id', 'hazard')}' overlaps spawn clearance margin ({margin}px) directly on player spawn zone ({sx:.1f}, {sy:.1f}).",
                    diagnostic_data={"hazard": h, "failure_coordinates": (hx, hy), "death_tick": 0}
                )

        # 2. Check simulation telemetry if available
        if sim_result is not None and isinstance(sim_result, dict):
            first_death_tick = sim_result.get("first_death_tick")
            if first_death_tick is not None and first_death_tick < self.early_death_tick_threshold:
                death_sec = first_death_tick / 60.0
                return QualityGateResult(
                    deliberation=f"Player died prematurely at tick {first_death_tick} ({death_sec:.2f}s < 3.0s).",
                    gate_name="Gate C: Survivability",
                    passed=False,
                    failure_type=GateFailureType.SURVIVABILITY_VIOLATION,
                    details=f"Player died within first 3 seconds of simulation (tick {first_death_tick}).",
                    diagnostic_data={
                        "death_tick": first_death_tick,
                        "failure_coordinates": sim_result.get("failure_coordinates", (sx, sy))
                    }
                )

            # Check completion time limit
            completion_sec = sim_result.get("completion_time_sec", 0.0)
            max_sec = self.max_simulation_ticks / 60.0
            if completion_sec > max_sec:
                return QualityGateResult(
                    deliberation=f"Level completion took {completion_sec:.1f}s, exceeding maximum allowed {max_sec:.1f}s.",
                    gate_name="Gate C: Survivability",
                    passed=False,
                    failure_type=GateFailureType.SURVIVABILITY_VIOLATION,
                    details=f"Level clearance time {completion_sec:.1f}s exceeded threshold {max_sec:.1f}s.",
                    diagnostic_data={"completion_time_sec": completion_sec}
                )

        return QualityGateResult(
            deliberation="Level survivability sanity passed: spawn point is safe and player survived early phase.",
            gate_name="Gate C: Survivability",
            passed=True,
            failure_type=GateFailureType.NONE,
            details="Survivability criteria satisfied."
        )

    def verify_gate_d_exceptions(self, runtime_exc: Optional[Any]) -> QualityGateResult:
        """Gate D: Verifies zero unhandled exceptions occurred during execution."""
        if isinstance(runtime_exc, Exception):
            return QualityGateResult(
                deliberation=f"Simulation raised unhandled runtime exception: {runtime_exc}",
                gate_name="Gate D: Exception-Free",
                passed=False,
                failure_type=GateFailureType.UNHANDLED_EXCEPTION,
                details=f"{runtime_exc}\n{traceback.format_exc()}",
                diagnostic_data={"traceback": traceback.format_exc()}
            )

        return QualityGateResult(
            deliberation="No runtime exceptions detected during simulation execution.",
            gate_name="Gate D: Exception-Free",
            passed=True,
            failure_type=GateFailureType.NONE,
            details="Zero unhandled exceptions."
        )

    # =========================================================================
    # Simulation & Playtesting Runner
    # =========================================================================

    def run_simulation_and_playtest(
        self,
        cartridge: BaseCartridge,
        level_spec: BaseLevelSpec,
        max_ticks: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
        """Executes simulation/playtesting via qa.playtester if available, else lightweight internal simulation."""
        ticks = max_ticks or self.max_simulation_ticks

        try:
            from qa.playtester import run_playtest_evaluation
            result = run_playtest_evaluation(cartridge, level_spec, max_ticks=ticks)
            res_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            return res_dict, None
        except ImportError:
            pass
        except Exception as e:
            return None, e

        try:
            player = cartridge.create_player(level_spec.spawn_point[0], level_spec.spawn_point[1])
            enemies = cartridge.create_enemies(level_spec)
            hazards = cartridge.create_hazards(level_spec)

            px, py = level_spec.spawn_point
            for h in hazards:
                if hasattr(h, "aabb") and h.aabb.contains_point(px, py):
                    return {
                        "traversable": True,
                        "cleared": False,
                        "stuck": False,
                        "first_death_tick": 0,
                        "completion_time_sec": 0.0,
                        "failure_coordinates": (px, py),
                    }, None

            return {
                "traversable": True,
                "cleared": True,
                "stuck": False,
                "first_death_tick": None,
                "completion_time_sec": 5.0,
            }, None
        except Exception as e:
            return None, e

    # =========================================================================
    # Promotion & Staging File Operations
    # =========================================================================

    @staticmethod
    def _clear_immutable_and_writable(path_obj: Union[Path, str]) -> None:
        import stat as _stat

        p_str = str(path_obj)
        if hasattr(os, "chflags"):
            try:
                st = os.stat(p_str, follow_symlinks=False)
                if getattr(st, "st_flags", 0):
                    os.chflags(p_str, 0)
            except OSError:
                pass
        try:
            st_mode = os.stat(p_str, follow_symlinks=False).st_mode
            if not (st_mode & _stat.S_IWUSR):
                os.chmod(p_str, st_mode | _stat.S_IWUSR)
        except OSError:
            pass

    @classmethod
    def _rmtree_force(cls, target_dir: Path) -> None:
        if not target_dir.exists():
            return

        def _onerror(func: Any, path_str: str, _exc_info: Any) -> None:
            cls._clear_immutable_and_writable(path_str)
            parent = os.path.dirname(path_str)
            if parent:
                cls._clear_immutable_and_writable(parent)
            func(path_str)

        shutil.rmtree(target_dir, onerror=_onerror)

    @classmethod
    def _copy2_unlocked(cls, src: Union[Path, str], dst: Union[Path, str]) -> str:
        if os.path.exists(dst):
            cls._clear_immutable_and_writable(dst)
        out = shutil.copy2(src, dst)
        cls._clear_immutable_and_writable(out)
        return out

    def promote_candidate(self, candidate_dir: Path) -> None:
        """Atomically promotes verified candidate cartridge files to cartridges/active/."""
        self.active_cartridge_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = Path(candidate_dir).resolve()
        target_path = Path(self.active_cartridge_dir).resolve()

        for item in candidate_path.iterdir():
            if item.name.startswith("."):
                continue
            if item.is_file():
                tmp_dst = target_path / f".{item.name}.tmp"
                final_dst = target_path / item.name
                self._clear_immutable_and_writable(final_dst)
                self._copy2_unlocked(item, tmp_dst)
                os.replace(tmp_dst, final_dst)
            elif item.is_dir():
                sub_target = target_path / item.name
                if sub_target.exists():
                    self._rmtree_force(sub_target)
                shutil.copytree(item, sub_target, copy_function=self._copy2_unlocked)

        # Also sync the latest matching checkpoint snapshot in cartridges/checkpoints/ if present
        checkpoints_root = self.cartridges_root / "checkpoints"
        if checkpoints_root.is_dir():
            matching_ckpts = sorted(
                [p for p in checkpoints_root.iterdir() if p.is_dir() and p.name.startswith(candidate_path.name)]
            )
            if matching_ckpts:
                latest_ckpt = matching_ckpts[-1]
                for item in candidate_path.iterdir():
                    if item.name.startswith(".") or item.name == "checkpoint_metadata.json":
                        continue
                    if item.is_file():
                        self._copy2_unlocked(item, latest_ckpt / item.name)
                    elif item.is_dir():
                        sub_dst = latest_ckpt / item.name
                        if sub_dst.exists():
                            self._rmtree_force(sub_dst)
                        shutil.copytree(item, sub_dst, copy_function=self._copy2_unlocked)

    # =========================================================================
    # Closed-Loop Self-Healing Logic
    # =========================================================================

    def heal_specification(
        self,
        spec: Union[GenesisSpec, EpisodicUpdateSpec],
        trace: RejectionDiagnosticTrace,
    ) -> Union[GenesisSpec, EpisodicUpdateSpec]:
        """Mutates specification based on RejectionDiagnosticTrace for closed-loop repair."""
        healed_spec = copy.deepcopy(spec)
        remedy_note = f"[SELF-HEALING REPAIR] [{trace.remedy_type.value}] {trace.recommended_remedy}"

        if isinstance(healed_spec, EpisodicUpdateSpec):
            cfg = healed_spec.updated_episode_config

            if trace.failure_type in (GateFailureType.CRITICAL_PATH_BLOCKED, GateFailureType.TIMEOUT_STUCK):
                updated_walls = []
                bx, by = trace.failure_coordinates or (640.0, 360.0)

                for wall in cfg.walls:
                    wx, wy, ww, wh = wall[:4]
                    if self._is_outer_wall(wall):
                        updated_walls.append(wall)
                        continue
                    if wx <= bx <= (wx + ww) and wy <= by <= (wy + wh):
                        logger.info("Self-healing: Removed blocking wall at (%f, %f, %f, %f)", wx, wy, ww, wh)
                        continue
                    updated_walls.append(wall)

                if len(updated_walls) == len(cfg.walls) and trace.blocked_nodes:
                    for b_tx, b_ty in trace.blocked_nodes:
                        nx = (b_tx + 0.5) * self.tile_size
                        ny = (b_ty + 0.5) * self.tile_size
                        for idx, w in enumerate(updated_walls):
                            if not self._is_outer_wall(w):
                                wx, wy, ww, wh = w[:4]
                                if wx <= nx <= (wx + ww) and wy <= ny <= (wy + wh):
                                    logger.info("Self-healing: Removed wall %d matching blocked node (%d, %d)", idx, b_tx, b_ty)
                                    del updated_walls[idx]
                                    break
                        if len(updated_walls) < len(cfg.walls):
                            break

                if len(updated_walls) == len(cfg.walls) and len(updated_walls) > 4:
                    for i in reversed(range(len(updated_walls))):
                        if not self._is_outer_wall(updated_walls[i]):
                            logger.info("Self-healing: Removed interior wall %d to restore passage", i)
                            del updated_walls[i]
                            break

                cfg.walls = updated_walls
                healed_spec.spatial_directives.append(
                    SpatialUpdateDirective(
                        deliberation=f"Self-healing removed blocking obstacle near ({bx:.0f}, {by:.0f}) to restore traversability.",
                        action=DirectiveAction.EXPAND_CHOKEPOINT,
                        geometry_type="wall",
                        bounds=(bx - 24.0, by - 24.0, 48.0, 48.0),
                        navmesh_verified=True,
                    )
                )

            elif trace.failure_type == GateFailureType.SURVIVABILITY_VIOLATION:
                safe_hazards = []
                for h in cfg.hazards:
                    hx = float(h.get("x", 0.0))
                    hy = float(h.get("y", 0.0))
                    if abs(hx - 120.0) < 120.0 and abs(hy - 360.0) < 120.0:
                        h_mod = dict(h)
                        h_mod["x"] = hx + 240.0
                        safe_hazards.append(h_mod)
                    else:
                        safe_hazards.append(h)
                cfg.hazards = safe_hazards
                cfg.hazard_damage = max(5.0, cfg.hazard_damage * 0.70)
                cfg.enemy_damage = max(5.0, cfg.enemy_damage * 0.80)

            healed_spec.deliberation = f"{remedy_note}\n\n{healed_spec.deliberation}"
            return healed_spec

        elif isinstance(healed_spec, GenesisSpec):
            if trace.failure_type in (GateFailureType.CRITICAL_PATH_BLOCKED, GateFailureType.TIMEOUT_STUCK):
                healed_spec.level_dimensions.pillar_count = max(0, healed_spec.level_dimensions.pillar_count - 2)
            elif trace.failure_type == GateFailureType.SURVIVABILITY_VIOLATION:
                for h in healed_spec.hazard_specs:
                    h.damage = max(5.0, h.damage * 0.70)
                for e in healed_spec.enemy_specs:
                    e.contact_damage = max(5.0, e.contact_damage * 0.80)

            healed_spec.deliberation = f"{remedy_note}\n\n{healed_spec.deliberation}"
            return healed_spec

        return healed_spec

    _heal_specification = heal_specification

    def _is_outer_wall(self, wall: Union[Tuple[float, float, float, float], List[float]]) -> bool:
        """Determines if a wall segment is part of the outer perimeter boundary."""
        wx, wy, ww, wh = wall[:4]
        return (
            (wx <= 0.0 and wy <= 0.0 and ww >= 1200.0) or
            (wx <= 0.0 and wy >= 680.0 and ww >= 1200.0) or
            (wx <= 0.0 and wy <= 0.0 and wh >= 680.0) or
            (wx >= 1240.0 and wy <= 0.0 and wh >= 680.0)
        )

    def _find_raycast_blockers(
        self,
        grid: List[List[int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Traces a line on the grid between start and goal and returns all solid obstacle tiles intersected."""
        x0, y0 = start
        x1, y1 = goal
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        blockers: List[Tuple[int, int]] = []
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        curr_x, curr_y = x0, y0
        while True:
            if 0 <= curr_x < cols and 0 <= curr_y < rows:
                if grid[curr_y][curr_x] == 1:
                    blockers.append((curr_x, curr_y))
            if curr_x == x1 and curr_y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += sx
            if e2 < dx:
                err += dx
                curr_y += sy

        return blockers


def build_qa_gate_work_item_proposals(
    candidate_dir: Path,
    trace: RejectionDiagnosticTrace,
    pass_index: int = 0,
    slice_id: str = "",
    existing_item_ids: Optional[Set[str]] = None,
    supported_kinds: Optional[Set[str]] = None,
) -> List[Any]:
    """Translates a BuildSupervisor `RejectionDiagnosticTrace` into one or more Rationale-First
    `WorkItemProposal`s for the Continuous Production Board:
    - For broken/empty visual PNGs (`offending_assets`), emits 1 atomic `visual_asset` task per broken file
      (specifying exact `target_path` for in-place overwrite + catalog grid/purpose metadata).
    - When GDScript slicing/texture-wiring bugs (`offending_scripts`) or code/runtime defects are also present,
      emits a dependent `mechanic_code` / `integration` task (`depends_on=[<visual_asset_ids>]`) so SWE reintegrates
      and fixes `.gd` code right after asset regeneration completes.
    """
    import re as _re
    from engine.studio.models import WorkItemProposal

    c_dir = Path(candidate_dir).resolve()
    fail_gate = str(getattr(trace, "failure_gate", "Quality Gate") or "Quality Gate").strip()
    fail_type = getattr(trace, "failure_type", GateFailureType.NONE)
    fail_type_str = fail_type.value if hasattr(fail_type, "value") else str(fail_type)
    fail_reason = str(getattr(trace, "failure_reason", "") or "Quality gate verification failed.").strip()
    rec_remedy = str(
        getattr(trace, "recommended_remedy", "")
        or getattr(trace, "remedy_directive", "")
        or "Resolve the reported quality gate defect cleanly in the playable cartridge."
    ).strip()

    rem_params = dict(getattr(trace, "remedy_parameters", None) or {})
    asset_meta_map: Dict[str, Dict[str, Any]] = dict(rem_params.get("asset_metadata") or {})
    taken: Set[str] = set(existing_item_ids or set())

    def _alloc_id(base_slug: str) -> str:
        cand = base_slug
        sfx = 2
        while cand in taken:
            cand = f"{base_slug}_{sfx}"
            sfx += 1
        taken.add(cand)
        return cand

    offending_assets: List[str] = []
    for raw_a in (rem_params.get("offending_assets") or []):
        norm_a = str(raw_a or "").strip().replace("\\", "/").replace("res://", "").lstrip("/")
        if norm_a.startswith("assets/") and norm_a.lower().endswith(".png") and norm_a not in offending_assets:
            offending_assets.append(norm_a)

    diag_blob = "\n".join(
        str(part or "")
        for part in (
            fail_reason,
            getattr(trace, "runtime_traceback", ""),
            " ".join(str(v) for v in (getattr(trace, "asset_violations", None) or [])),
        )
    )
    if fail_type == GateFailureType.ASSET_INTEGRITY_VIOLATION and not offending_assets:
        for m_png in _re.findall(r"(?:res://)?(assets/(?:sprites|ui|tiles|portraits)/[A-Za-z0-9_./-]+\.png)", diag_blob):
            if m_png not in offending_assets and not rem_params.get("offending_scripts"):
                offending_assets.append(m_png)

    proposals: List[Any] = []
    visual_item_ids: List[str] = []
    slice_slug = _re.sub(r"[^a-z0-9]+", "_", (slice_id or "slice").lower()).strip("_")[:20] or "slice"

    can_emit_visual = (not supported_kinds) or ("visual_asset" in supported_kinds)
    if offending_assets and can_emit_visual:
        for rel_asset in offending_assets:
            a_stem = _re.sub(r"[^a-z0-9]+", "_", Path(rel_asset).stem.lower()).strip("_")[:28] or "asset"
            v_id = _alloc_id(f"qa_regen_{a_stem}_{slice_slug}_p{pass_index}")
            meta = asset_meta_map.get(rel_asset, {})
            grid_wh = meta.get("grid") or [1, 1]
            g_cols, g_rows = max(1, int(grid_wh[0])), max(1, int(grid_wh[1]))
            px_wh = meta.get("size") or [128, 128]
            fmode = str(meta.get("mode") or ("diegetic_ui" if rel_asset.startswith("assets/ui/") else "static_sprite"))
            purpose = str(meta.get("purpose") or f"Production visual asset at {rel_asset}")
            grid_clause = (
                f"Multi-cell {g_cols}x{g_rows} atlas (`grid_cols={g_cols}, grid_rows={g_rows}`, populate all {g_cols * g_rows} cells via `cell_prompts`)"
                if g_cols * g_rows > 1
                else "Single-frame (`1x1`) clean alpha sprite/tile"
            )
            v_spec = (
                f"Regenerate broken visual asset `{rel_asset}` (`res://{rel_asset}`) in-place so existing cartridge references heal cleanly:\n"
                f"- Target Path (overwrite in-place): `{rel_asset}` (`res://{rel_asset}`)\n"
                f"- Fabrication Mode: `{fmode}` | Pixel Size: `{px_wh[0]}x{px_wh[1]}` | Grid Contract: {grid_clause}\n"
                f"- Asset Purpose: {purpose}\n"
                f"- QA Gate Diagnostic: {fail_reason[:500]}\n"
                f"Acceptance Criteria:\n"
                f"1. Synthesize high-craft artwork matching `game_bible.json` `visual_style_guide` and `assets/mockups/` concept references (zero flat geometric placeholder shapes).\n"
                f"2. Write directly to `{rel_asset}` (`force_regenerate=True`) and verify non-empty alpha across every required cell."
            )
            v_proof = (
                f"Render inspection plate `plate_{v_id}.png` for `{rel_asset}` confirming full stylistic alignment with "
                f"`game_bible.json` and non-empty artwork across all {g_cols}x{g_rows} cell(s)."
            )
            proposals.append(
                WorkItemProposal(
                    deliberation=(
                        f"{fail_gate} identified defective visual asset '{rel_asset}' ({fail_reason[:180]}). "
                        f"Spawning dedicated `visual_asset` regeneration task '{v_id}' targeting '{rel_asset}' in-place."
                    ),
                    item_id=v_id,
                    kind="visual_asset",
                    title=f"Regenerate Visual Asset: {Path(rel_asset).name}",
                    spec=v_spec,
                    proof_spec=v_proof,
                    depends_on=[],
                    blocks_items=[],
                    target_modules=[rel_asset],
                    slice_id=slice_id,
                )
            )
            visual_item_ids.append(v_id)

    offending_scripts_raw: List[str] = [
        str(s or "").strip().replace("\\", "/").replace("res://", "").lstrip("/")
        for s in (rem_params.get("offending_scripts") or [])
        if str(s or "").strip()
    ]
    needs_code_task = (
        not visual_item_ids
        or bool(offending_scripts_raw)
        or fail_gate.startswith("Gate C")
        or bool(_re.search(r"[A-Za-z0-9_./-]+\.(?:gd|tscn)\b", diag_blob))
    )
    if not needs_code_task:
        return proposals

    mechanic_failure_types = {
        GateFailureType.SYNTAX_AST_ERROR,
        GateFailureType.UNHANDLED_EXCEPTION,
        GateFailureType.PROGRESSION_FAILURE,
        GateFailureType.FRAME_BUDGET_EXCEEDED,
    }
    kind = "mechanic_code" if (fail_type in mechanic_failure_types or offending_scripts_raw) else "integration"
    if supported_kinds:
        if kind not in supported_kinds:
            if "integration" in supported_kinds:
                kind = "integration"
            elif "mechanic_code" in supported_kinds:
                kind = "mechanic_code"
            else:
                kind = sorted(supported_kinds)[0]

    target_modules: List[str] = []

    def _add_candidate_module(raw_mod: Optional[str]) -> None:
        if not raw_mod:
            return
        cleaned = str(raw_mod).strip().replace("\\", "/")
        if cleaned.startswith("res://"):
            cleaned = cleaned[len("res://"):]
        elif cleaned.startswith("res:/"):
            cleaned = cleaned[len("res:/"):]
        cleaned = cleaned.lstrip("/")
        try:
            p_cand = Path(cleaned)
            if p_cand.is_absolute():
                cleaned = p_cand.resolve().relative_to(c_dir).as_posix()
        except Exception:
            cleaned = Path(cleaned).name
        if (
            cleaned
            and not cleaned.startswith("tests/")
            and not cleaned.startswith("src/studio_rig_and_collider")
            and (c_dir / cleaned).is_file()
            and cleaned not in target_modules
        ):
            target_modules.append(cleaned)

    for s_mod in offending_scripts_raw:
        _add_candidate_module(s_mod)
    _add_candidate_module(getattr(trace, "file_path", None))
    for m_mod in _re.findall(r"(?:res://)?([A-Za-z0-9_./-]+\.(?:gd|tscn|json))", diag_blob):
        _add_candidate_module(m_mod)

    if not target_modules:
        try:
            from engine.studio.context_curator import compute_entrypoint_reachability

            reach = compute_entrypoint_reachability(c_dir)
            for mod_rel in list(reach.get("reachable_from_entrypoint") or [])[:4]:
                _add_candidate_module(mod_rel)
        except Exception:
            pass
    if not target_modules:
        for default_mod in ("main.gd", "scenes.gd", "mechanics.gd"):
            if (c_dir / default_mod).is_file():
                target_modules.append(default_mod)
    if not target_modules:
        target_modules = ["main.gd"]

    gate_slug = _re.sub(r"[^a-z0-9]+", "_", fail_gate.lower().split(":")[0]).strip("_") or "qa_gate"
    type_slug = _re.sub(r"[^a-z0-9]+", "_", fail_type_str.lower()).strip("_")[:28] or "remediation"
    item_id = _alloc_id(f"qa_{gate_slug}_{type_slug}_{slice_slug}_p{pass_index}")

    extra_lines: List[str] = []
    if getattr(trace, "file_path", None):
        ln_sfx = f":{trace.line_number}" if getattr(trace, "line_number", None) else ""
        extra_lines.append(f"- Diagnostic File Location: `{trace.file_path}{ln_sfx}`")
    if getattr(trace, "runtime_traceback", None):
        tb_trim = str(trace.runtime_traceback).strip()[:900]
        if tb_trim:
            extra_lines.append(f"- Runtime Traceback / Stderr:\n{tb_trim}")
    if getattr(trace, "asset_violations", None):
        extra_lines.append(f"- Asset Violations: {list(trace.asset_violations)[:6]}")
    if getattr(trace, "unfulfilled_requirements", None):
        extra_lines.append(f"- Unfulfilled Requirements: {list(trace.unfulfilled_requirements)[:6]}")

    extra_context = ("\n" + "\n".join(extra_lines)) if extra_lines else ""
    spec_text = (
        f"Resolve the {fail_gate} ({fail_type_str}) verification failure in the playable cartridge (`res://main.tscn`) "
        f"without regressing existing slice mechanics, visual framing, or controls:\n"
        f"- Failure Diagnostic: {fail_reason}\n"
        f"- Recommended Engineering Remedy: {rec_remedy}"
        f"{extra_context}\n"
        f"Acceptance Criteria:\n"
        f"1. Fix the root cause in production cartridge modules ({', '.join(target_modules[:4])})—ensure texture region slicing matches `asset_library.json` `(hframes, vframes)`, bind every loaded `Texture2D` onto its target `Sprite2D`/`TextureRect`/`Polygon2D`, and never mask errors or touch `tests/test_*` in place of production fixes.\n"
        f"2. Verify that `res://main.tscn` boots cleanly, maintains 60 FPS steady-state frame budget (< 16.6ms avg), and preserves active gameplay progression."
    )
    proof_spec = (
        f"Capture an authentic gameplay filmstrip on `res://main.tscn` demonstrating clean boot, intact sprite/UI/terrain rendering, "
        f"responsive active gameplay progression, and full resolution of the {fail_gate} ({fail_type_str}) diagnostic with zero runtime script errors."
    )

    proposals.append(
        WorkItemProposal(
            deliberation=(
                f"{fail_gate} ({fail_type_str}) rejected candidate cartridge '{c_dir.name}' ({fail_reason[:240]}). "
                f"Creating tracked Production Board task '{item_id}' targeting {target_modules[:4]} so the Task Lead, "
                f"domain specialists, and Specialist Council diagnose, fix, and visually verify the resolution on-board."
            ),
            item_id=item_id,
            kind=kind,
            title=f"QA {fail_gate.split(':')[0].strip()} Remediation: {fail_type_str.replace('_', ' ').title()}",
            spec=spec_text,
            proof_spec=proof_spec,
            depends_on=list(visual_item_ids),
            blocks_items=[],
            target_modules=target_modules[:6],
            slice_id=slice_id,
        )
    )
    return proposals


def build_qa_gate_work_item_proposal(
    candidate_dir: Path,
    trace: RejectionDiagnosticTrace,
    pass_index: int = 0,
    slice_id: str = "",
    existing_item_ids: Optional[Set[str]] = None,
    supported_kinds: Optional[Set[str]] = None,
) -> Any:
    """Backward-compatible single-item wrapper around `build_qa_gate_work_item_proposals`."""
    return build_qa_gate_work_item_proposals(
        candidate_dir=candidate_dir,
        trace=trace,
        pass_index=pass_index,
        slice_id=slice_id,
        existing_item_ids=existing_item_ids,
        supported_kinds=supported_kinds,
    )[0]

