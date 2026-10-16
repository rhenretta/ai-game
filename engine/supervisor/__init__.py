"""engine/supervisor package
Build Supervisor & Closed-Loop Code Self-Healing Guardrail (Milestone 5, Feature 16).
"""
from engine.supervisor.models import (
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    RemedyType,
    SupervisorEvaluationReport,
    SupervisorGateVerdict,
)
from engine.supervisor.supervisor import (
    BuildSupervisor,
    SupervisorError,
)

__all__ = [
    "BuildSupervisor",
    "SupervisorError",
    "SupervisorGateVerdict",
    "GateFailureType",
    "RemedyType",
    "QualityGateResult",
    "RejectionDiagnosticTrace",
    "SupervisorEvaluationReport",
]
