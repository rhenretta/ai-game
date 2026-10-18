"""engine/supervisor package
Build Supervisor & Closed-Loop Code Self-Healing Guardrail (Milestone 5, Feature 16).
"""
from engine.supervisor.models import (
    ChallengerGateEvaluation,
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    RemedyType,
    RequirementEvaluationRecord,
    SupervisorEvaluationReport,
    SupervisorGateVerdict,
)
from engine.supervisor.supervisor import (
    BuildSupervisor,
    SupervisorError,
)
from engine.supervisor.challenger import (
    AdversarialDesignCritic,
    CandidateASTInspector,
    CandidateTelemetryAuditor,
    CandidateVisualPlaytester,
    ChallengerGate,
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
    "RequirementEvaluationRecord",
    "ChallengerGateEvaluation",
    "ChallengerGate",
    "AdversarialDesignCritic",
    "CandidateASTInspector",
    "CandidateTelemetryAuditor",
    "CandidateVisualPlaytester",
]
