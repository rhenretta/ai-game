"""engine/supervisor/models.py
Rationale-First Data Models & Enums for Five-Gate Build Supervisor (Milestone M4, Features 24 & 27).
In strict compliance with Architectural Integrity Rule 4 and AGENTS.md,
all Pydantic models declare 'deliberation' as their very first property at field index 0.
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator


class SupervisorGateVerdict(str, Enum):
    """Overall outcome of the Build Supervisor evaluation."""
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    FAILED_MAX_RETRIES = "FAILED_MAX_RETRIES"


class GateFailureType(str, Enum):
    """Categorization of quality gate failures across the 5 gates."""
    NONE = "NONE"
    SYNTAX_AST_ERROR = "SYNTAX_AST_ERROR"              # Gate A
    ENTRYPOINT_MISSING = "ENTRYPOINT_MISSING"          # Gate A
    INVALID_SCENE_LIFECYCLE = "INVALID_SCENE_LIFECYCLE"# Gate A
    ASSET_INTEGRITY_VIOLATION = "ASSET_INTEGRITY_VIOLATION"  # Gate B
    UNHANDLED_EXCEPTION = "UNHANDLED_EXCEPTION"        # Gate C
    PROGRESSION_FAILURE = "PROGRESSION_FAILURE"        # Gate C
    FRAME_BUDGET_EXCEEDED = "FRAME_BUDGET_EXCEEDED"    # Gate D
    TELEMETRY_EMISSION_FAILURE = "TELEMETRY_EMISSION_FAILURE"  # Gate E
    TELEMETRY_EMISSION_MISSING = "TELEMETRY_EMISSION_FAILURE"  # Gate E alias
    # Backward-compatible legacy aliases
    CRITICAL_PATH_BLOCKED = "CRITICAL_PATH_BLOCKED"
    TIMEOUT_STUCK = "TIMEOUT_STUCK"
    SURVIVABILITY_VIOLATION = "SURVIVABILITY_VIOLATION"
    WALL_CLIPPING = "WALL_CLIPPING"
    SPAWN_HAZARD_OVERLAP = "SURVIVABILITY_VIOLATION"
    PLAYER_DEATH = "SURVIVABILITY_VIOLATION"


class RemedyType(str, Enum):
    """Actionable category of recommended remedy for self-healing repair."""
    FIX_SYNTAX = "fix_syntax"
    FIX_SCENE_ENTRYPOINT = "fix_scene_entrypoint"
    REGENERATE_ASSET = "regenerate_asset"
    FIX_NARRATIVE_REFERENCE = "fix_narrative_reference"
    FIX_NARRATIVE_SCHEMA = "fix_narrative_schema"
    REPAIR_CODE_EXCEPTION = "repair_code_exception"
    OPTIMIZE_FRAME_LOOP = "optimize_frame_loop"
    INJECT_TELEMETRY_PROBES = "inject_telemetry_probes"
    # Backward-compatible legacy aliases:
    RELOCATE_OBSTACLE = "relocate_obstacle"
    REMOVE_OBSTACLE = "remove_obstacle"
    WIDEN_MARGIN = "widen_margin"
    SOFTEN_HAZARD = "soften_hazard"
    REDUCE_ENEMY_DENSITY = "reduce_enemy_density"


class QualityGateResult(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first at field index 0.
    Encapsulates pass/fail verification for a single quality gate.
    """
    deliberation: str = Field(
        default="Epistemic rationale explaining verification results and decision criteria.",
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining verification results and decision criteria."
    )
    gate_name: str = Field(..., description="Name of the evaluated gate ('Gate A: AST Security', etc.).")
    passed: bool = Field(..., description="True if candidate met 100% of gate criteria.")
    failure_type: GateFailureType = Field(default=GateFailureType.NONE, description="Failure category if failed.")
    details: str = Field(default="", description="Detailed human-readable diagnostic log or traceback.")
    diagnostic_data: Dict[str, Any] = Field(default_factory=dict, description="Structured failure metrics.")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("deliberation must be a non-empty string.")
        return v

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "gate_name": self.gate_name,
            "passed": self.passed,
            "failure_type": self.failure_type.value,
            "details": self.details,
            "diagnostic_data": self.diagnostic_data,
        }


class RejectionDiagnosticTrace(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first at field index 0.
    Structured failure report generated upon gate rejection, informing closed-loop self-healing repair.
    """
    deliberation: str = Field(
        default="Epistemic rationale analyzing why failure occurred and why remedy was selected.",
        description="MANDATORY FIRST PROPERTY: Epistemic rationale analyzing why failure occurred and why remedy was selected."
    )
    trace_id: str = Field(
        default_factory=lambda: f"trace_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this diagnostic trace."
    )
    candidate_id: str = Field(default="", description="Identifier of the failing candidate cartridge or spec.")
    episode_number: int = Field(default=0, description="Target episode number.")
    attempt_number: int = Field(default=1, description="Evaluation attempt number (1..max_retries).")
    failure_gate: str = Field(..., description="Quality gate that rejected the build ('Gate A', 'Gate E', 'Gate D', 'Gate B', 'Gate C').")
    failure_type: GateFailureType = Field(..., description="Categorized failure classification.")
    failure_reason: str = Field(..., description="Clear explanation of the failure.")
    file_path: Optional[str] = Field(default=None, description="Path to file where defect was detected.")
    line_number: Optional[int] = Field(default=None, description="Line number where syntax/exception occurred.")
    runtime_traceback: Optional[str] = Field(default=None, description="Full Python traceback if unhandled exception.")
    failure_coordinates: Optional[Tuple[float, float]] = Field(
        default=None,
        description="World coordinates (x, y) where failure was detected (e.g. blockage or death)."
    )
    tick_count: int = Field(default=0, description="Simulation tick where defect occurred.")
    blocked_nodes: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="Grid tile coordinates (tx, ty) blocking critical navigation path."
    )
    asset_violations: List[str] = Field(
        default_factory=list,
        description="List of asset validation failures if Gate E rejected candidate."
    )
    remedy_type: RemedyType = Field(
        default=RemedyType.RELOCATE_OBSTACLE,
        description="Categorical remedy strategy."
    )
    recommended_remedy: str = Field(
        default="",
        description="Actionable instruction consumed by self-healing repair engine."
    )
    remedy_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specific parameters for repair (e.g. coordinates, target wall index)."
    )
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp."
    )

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("deliberation must be a non-empty string.")
        return v

    @model_validator(mode="before")
    @classmethod
    def handle_spec_id_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "spec_id" in data and not data.get("candidate_id"):
                data["candidate_id"] = data["spec_id"]
            elif "candidate_id" in data and not data.get("spec_id"):
                data["spec_id"] = data["candidate_id"]
        return data

    @property
    def spec_id(self) -> str:
        """Backward-compatible alias for candidate_id."""
        return self.candidate_id

    @spec_id.setter
    def spec_id(self, val: str) -> None:
        self.candidate_id = val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "trace_id": self.trace_id,
            "candidate_id": self.candidate_id,
            "spec_id": self.candidate_id,
            "episode_number": self.episode_number,
            "attempt_number": self.attempt_number,
            "failure_gate": self.failure_gate,
            "failure_type": self.failure_type.value,
            "failure_reason": self.failure_reason,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "runtime_traceback": self.runtime_traceback,
            "failure_coordinates": list(self.failure_coordinates) if self.failure_coordinates else None,
            "tick_count": self.tick_count,
            "blocked_nodes": [list(n) for n in self.blocked_nodes],
            "asset_violations": list(self.asset_violations),
            "remedy_type": self.remedy_type.value,
            "recommended_remedy": self.recommended_remedy,
            "remedy_parameters": self.remedy_parameters,
            "created_at_utc": self.created_at_utc,
        }


class BuildEvaluationReport(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first at field index 0.
    Complete audit trail, quality gate scores, and promotion verdict emitted by BuildSupervisor.
    """
    deliberation: str = Field(
        default="Epistemic rationale summarizing overall evaluation outcome and promotion decision.",
        description="MANDATORY FIRST PROPERTY: Epistemic rationale summarizing overall evaluation outcome and promotion decision."
    )
    candidate_id: str = Field(default="", description="Candidate cartridge or specification identifier.")
    episode_number: int = Field(default=0, description="Target episode number.")
    status: SupervisorGateVerdict = Field(..., description="Overall verdict (PROMOTED / REJECTED / FAILED_MAX_RETRIES).")
    promoted_to_active: bool = Field(default=False, description="True if files were copied to cartridges/active/.")
    candidate_dir: str = Field(default="", description="Path to evaluated candidate directory.")
    active_dir: str = Field(default="", description="Path to active cartridge directory.")
    gate_results: Dict[str, QualityGateResult] = Field(
        default_factory=dict,
        description="Evaluation results for each of the 5 quality gates (gate_a, gate_e, gate_d, gate_b, gate_c)."
    )
    rejection_trace: Optional[RejectionDiagnosticTrace] = Field(
        default=None,
        description="Diagnostic trace from the latest failed attempt, if any."
    )
    attempt_count: int = Field(default=1, description="Number of evaluation/repair attempts executed.")
    max_retries: int = Field(default=3, description="Maximum allowed repair attempts.")
    repair_history: List[RejectionDiagnosticTrace] = Field(
        default_factory=list,
        description="History of diagnostic traces across previous repair attempts."
    )
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp."
    )

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("deliberation must be a non-empty string.")
        return v

    @model_validator(mode="before")
    @classmethod
    def handle_spec_id_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "spec_id" in data and not data.get("candidate_id"):
                data["candidate_id"] = data["spec_id"]
            elif "candidate_id" in data and not data.get("spec_id"):
                data["spec_id"] = data["candidate_id"]
        return data

    @property
    def spec_id(self) -> str:
        """Backward-compatible alias for candidate_id."""
        return self.candidate_id

    @spec_id.setter
    def spec_id(self, val: str) -> None:
        self.candidate_id = val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "candidate_id": self.candidate_id,
            "spec_id": self.candidate_id,
            "episode_number": self.episode_number,
            "status": self.status.value,
            "promoted_to_active": self.promoted_to_active,
            "candidate_dir": self.candidate_dir,
            "active_dir": self.active_dir,
            "gate_results": {k: v.to_dict() for k, v in self.gate_results.items()},
            "rejection_trace": self.rejection_trace.to_dict() if self.rejection_trace else None,
            "attempt_count": self.attempt_count,
            "max_retries": self.max_retries,
            "repair_history": [t.to_dict() for t in self.repair_history],
            "created_at_utc": self.created_at_utc,
        }


# Backward-compatible alias for existing test suites
SupervisorEvaluationReport = BuildEvaluationReport


class AssetValidationReport(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first at field index 0.
    Encapsulates pass/fail verification for an individual multimodal asset.
    """
    deliberation: str = Field(
        default="Epistemic rationale explaining asset verification outcome and compliance.",
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining asset verification outcome and compliance."
    )
    asset_path: str = Field(..., description="Path to the evaluated asset file.")
    asset_type: str = Field(..., description="'audio' | 'sprite' | 'tile' | 'narrative'")
    passed: bool = Field(..., description="True if asset strictly meets format and content invariants.")
    errors: List[str] = Field(default_factory=list, description="List of detected validation errors.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic metadata (e.g. duration, dimensions, palette).")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("deliberation must be a non-empty string.")
        return v

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "asset_path": self.asset_path,
            "asset_type": self.asset_type,
            "passed": self.passed,
            "errors": list(self.errors),
            "details": self.details,
        }
