"""tests/test_priority_scheduler.py
Comprehensive Unit and Integration Test Suite for Priority Scheduler & Artifact Envelopes.
Verifies:
  1. Priority tier ordering (P0 < P1 < P2) and compatibility aliases.
  2. FrameBudgetMonitor safety threshold (>14ms at 60 FPS).
  3. Cooperative yielding contract during active gameplay and preemption.
  4. Preemption requests and system free state transitions.
  5. Asynchronous task execution with ScheduledTaskFuture.
  6. Rationale-First ArtifactMetadataEnvelope schema and key order invariants.
  7. Cryptographic SHA-256 integrity validation and tampering detection.
  8. Disk serialization, external payload separation, and rehydration.
  9. AST compliance for deliberation-first property declarations.
"""
from pathlib import Path
import time
import pytest

from engine.scheduler.artifact import (
    ArtifactIntegrityError,
    ArtifactMetadataEnvelope,
    ArtifactNotFoundError,
    ArtifactType,
)
from engine.scheduler.scheduler import (
    FrameBudgetMonitor,
    Priority,
    PriorityScheduler,
    ScheduledTask,
    ScheduledTaskFuture,
    TaskPriority,
    TaskState,
)
from tests.test_tier1_units import validate_rationale_first_ast


class TestPrioritySchedulerProperties:
    """Verifies priority enum hierarchy, aliases, and queue ordering."""

    def test_priority_ordering_and_aliases(self):
        assert Priority.P0_GAMEPLAY < Priority.P1_TELEMETRY < Priority.P2_SHOWRUNNER
        assert TaskPriority.PRIORITY_0_GAMEPLAY == 0
        assert TaskPriority.PRIORITY_1_TELEMETRY == 1
        assert TaskPriority.PRIORITY_2_SHOWRUNNER == 2

    def test_frame_budget_monitor_threshold(self):
        monitor = FrameBudgetMonitor(target_fps=60, budget_threshold_ms=14.0)
        monitor.start_frame()
        time.sleep(0.016)  # Sleep 16ms (> 14ms threshold)
        assert monitor.is_budget_exceeded() is True
        telemetry = monitor.end_frame()
        assert telemetry.budget_exceeded is True
        assert monitor.violation_count == 1
        assert telemetry.elapsed_ms >= 14.0

    def test_frame_budget_monitor_normal(self):
        monitor = FrameBudgetMonitor(target_fps=60, budget_threshold_ms=14.0)
        monitor.start_frame()
        # Fast frame (< 1ms)
        assert monitor.is_budget_exceeded() is False
        telemetry = monitor.end_frame()
        assert telemetry.budget_exceeded is False
        assert monitor.violation_count == 0

    def test_cooperative_yielding_under_gameplay(self):
        scheduler = PriorityScheduler(max_background_workers=1)
        scheduler.set_gameplay_active(True)
        assert scheduler.is_system_free() is False
        assert scheduler.should_yield() is True

        # Calling checkpoint_and_yield must yield (return True)
        yielded = scheduler.checkpoint_and_yield(timeout_sec=0.01)
        assert yielded is True

        # When gameplay is inactive, checkpoint_and_yield returns False immediately
        scheduler.set_gameplay_active(False)
        assert scheduler.is_system_free() is True
        assert scheduler.should_yield() is False
        yielded_free = scheduler.checkpoint_and_yield(timeout_sec=0.01)
        assert yielded_free is False

    def test_preemption_request_pauses_system(self):
        scheduler = PriorityScheduler(max_background_workers=1)
        assert scheduler.is_system_free() is True

        scheduler.request_preemption(reason="test_preemption")
        assert scheduler.is_system_free() is False
        assert scheduler.should_yield() is True

        scheduler.resume_background_tasks()
        assert scheduler.is_system_free() is True
        assert scheduler.should_yield() is False

    def test_p0_main_thread_stepping(self):
        scheduler = PriorityScheduler(max_background_workers=1)
        results = []

        f1 = scheduler.submit(Priority.P0_GAMEPLAY, "p0_first", lambda: results.append(1) or 1)
        f2 = scheduler.submit(Priority.P0_GAMEPLAY, "p0_second", lambda: results.append(2) or 2)

        executed = scheduler.step_main_thread(budget_ms=10.0)
        assert executed == 2
        assert results == [1, 2]
        assert f1.done() is True
        assert f1.result() == 1
        assert f2.result() == 2

    def test_background_worker_execution(self):
        scheduler = PriorityScheduler(max_background_workers=1)
        scheduler.start()
        try:
            future = scheduler.submit(
                Priority.P1_TELEMETRY,
                "p1_flush",
                lambda x, y: x * y,
                6,
                7
            )
            result = future.result(timeout=2.0)
            assert result == 42
            assert future.done() is True
        finally:
            scheduler.shutdown(wait=True, timeout=1.0)

    def test_scheduler_telemetry(self):
        scheduler = PriorityScheduler(max_background_workers=1)
        scheduler.submit(Priority.P2_SHOWRUNNER, "p2_compile", lambda: None)
        telemetry = scheduler.get_telemetry()

        assert "pending_tasks" in telemetry
        assert telemetry["pending_tasks"]["P2"] == 1
        assert telemetry["pending_tasks"]["total"] == 1
        assert "is_system_free" in telemetry
        assert "deliberation" in telemetry


class TestArtifactMetadataEnvelope:
    """Verifies typed artifact envelopes and cryptographic SHA-256 validation."""

    def test_envelope_deliberation_first(self):
        envelope = ArtifactMetadataEnvelope.seal(
            deliberation="Analyzed player telemetry indicating high mobility; proposing wider arena.",
            artifact_type=ArtifactType.UPDATE_SPEC,
            payload={"speed_multiplier": 1.15},
            source_agent="CognitiveDirector",
            target_agent="SWECompiler",
        )
        d = envelope.to_dict()
        assert list(d.keys())[0] == "deliberation"
        assert envelope.verify_integrity() is True

    def test_artifact_cryptographic_tampering_detection(self):
        envelope = ArtifactMetadataEnvelope.seal(
            deliberation="Initial valid envelope.",
            artifact_type=ArtifactType.BEHAVIORAL_PROFILE,
            payload={"avg_speed": 220.0},
        )
        assert envelope.verify_integrity() is True

        # Tamper with payload
        envelope.payload["avg_speed"] = 999.0
        assert envelope.verify_integrity() is False
        with pytest.raises(ArtifactIntegrityError):
            envelope.verify_integrity(raise_on_error=True)

    def test_artifact_disk_persistence_and_rehydration(self, tmp_path):
        envelope = ArtifactMetadataEnvelope.seal(
            deliberation="Generated level module geometry.",
            artifact_type=ArtifactType.GAME_MODULE,
            payload={"walls": [[0, 0, 100, 20]], "spawns": [10, 10]},
            source_agent="SWECompiler",
        )
        saved_file = envelope.save_to_disk(tmp_path, separate_payload_file=True)
        assert saved_file.exists()

        loaded = ArtifactMetadataEnvelope.load_from_disk(saved_file, verify=True)
        assert loaded.deliberation == "Generated level module geometry."
        assert loaded.sha256_hash == envelope.sha256_hash
        assert loaded.verify_integrity() is True

    def test_envelope_ast_rationale_first(self):
        root = Path(__file__).parent.parent
        artifact_file = root / "engine" / "scheduler" / "artifact.py"
        scheduler_file = root / "engine" / "scheduler" / "scheduler.py"

        assert validate_rationale_first_ast(artifact_file.read_text(encoding="utf-8")) is True
        assert validate_rationale_first_ast(scheduler_file.read_text(encoding="utf-8")) is True
