"""tests/test_m2_scheduler_stress.py
Empirical Stress Test Suite for Milestone 2:
  - 3-Tier OS Priority Scheduler Concurrency, Preemption Latency (<1ms), and Starvation Safety
  - Cocoa Frame Budget Spike Throttling (>14.0ms), Rapid Oscillations, and Deadlock Safety
  - Artifact Envelope Cryptographic Validation, SHA-256 Tampering, Atomic Persistence, and Non-ASCII/Binary Serialization
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Dict, List
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
    ScheduledTaskFuture,
    TaskPriority,
    TaskState,
)


class TestPrioritySchedulerConcurrencyAndPreemptionStress:
    """Stress tests for 3-tier priority queue, preemption latency, and starvation safety."""
    deliberation: str = "Empirical stress testing of PriorityScheduler concurrency, preemption latency, and starvation safety."

    def test_high_concurrency_worker_flood(self):
        """Stress: Flood scheduler with concurrent P2, P1, and P0 submissions from multiple threads."""
        scheduler = PriorityScheduler(max_background_workers=4)
        scheduler.start()

        errors: List[Exception] = []
        p0_completed: List[int] = []
        p1_completed: List[int] = []
        p2_completed: List[int] = []

        def p2_producer(worker_id: int):
            for i in range(40):
                try:
                    f = scheduler.submit(
                        Priority.P2_SHOWRUNNER,
                        f"p2_{worker_id}_{i}",
                        lambda w=worker_id, idx=i: (w, idx, scheduler.checkpoint_and_yield(timeout_sec=0.005)),
                    )
                    time.sleep(random.uniform(0.0001, 0.001))
                except Exception as e:
                    errors.append(e)

        def p1_producer():
            for i in range(40):
                try:
                    f = scheduler.submit(
                        Priority.P1_TELEMETRY,
                        f"p1_{i}",
                        lambda idx=i: idx * 10,
                    )
                    time.sleep(random.uniform(0.0001, 0.001))
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=p2_producer, args=(w,)) for w in range(4)]
        threads.append(threading.Thread(target=p1_producer))

        for t in threads:
            t.start()

        # Cocoa Thread 0 stepping loop
        for frame in range(80):
            f_p0 = scheduler.submit(Priority.P0_GAMEPLAY, f"p0_frame_{frame}", lambda f=frame: f)
            scheduler.step_main_thread(budget_ms=2.0)
            if f_p0.done():
                p0_completed.append(f_p0.result())
            time.sleep(0.001)

        for t in threads:
            t.join()

        # Step remaining main thread tasks
        scheduler.step_main_thread(budget_ms=10.0)

        # Allow background workers to drain remaining tasks
        drain_deadline = time.perf_counter() + 5.0
        while time.perf_counter() < drain_deadline:
            telemetry = scheduler.get_telemetry()
            if telemetry["pending_tasks"]["total"] == 0 and telemetry["active_tasks_count"] == 0:
                break
            time.sleep(0.01)

        scheduler.shutdown(wait=True, timeout=2.0)

        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
        assert len(p0_completed) == 80, f"Expected 80 P0 tasks completed, got {len(p0_completed)}"
        telemetry = scheduler.get_telemetry()
        assert telemetry["pending_tasks"]["total"] == 0

    def test_preemption_latency_under_one_millisecond(self):
        """Stress: Verify request_preemption() forces checkpoint_and_yield() to yield in < 1ms."""
        scheduler = PriorityScheduler(max_background_workers=1)
        scheduler.start()

        timestamps: Dict[str, float] = {}
        active = True

        def worker_loop():
            while active:
                if scheduler.should_yield():
                    if "yield_start" not in timestamps:
                        timestamps["yield_start"] = time.perf_counter()
                scheduler.checkpoint_and_yield(timeout_sec=0.1)

        future = scheduler.submit(Priority.P2_SHOWRUNNER, "continuous_worker", worker_loop)
        time.sleep(0.005)  # Let worker start

        timestamps["preempt_called"] = time.perf_counter()
        scheduler.request_preemption(reason="stress_latency_test")

        # Wait for worker to yield
        yield_deadline = time.perf_counter() + 0.1
        while "yield_start" not in timestamps and time.perf_counter() < yield_deadline:
            time.sleep(0.0001)

        active = False
        scheduler.resume_background_tasks()
        future.result(timeout=2.0)
        scheduler.shutdown(wait=True)

        assert "yield_start" in timestamps, "Worker failed to detect preemption"
        latency_ms = (timestamps["yield_start"] - timestamps["preempt_called"]) * 1000.0
        assert latency_ms < 1.0, f"Preemption latency {latency_ms:.3f}ms exceeded 1.0ms limit!"

    def test_p0_starvation_safety_under_massive_queue(self):
        """Stress: Verify P0 tasks execute immediately with zero delay regardless of P2 queue depth."""
        scheduler = PriorityScheduler(max_background_workers=2)

        # Flood heap with 5,000 P2 tasks
        for i in range(5000):
            scheduler.submit(Priority.P2_SHOWRUNNER, f"p2_bulk_{i}", lambda: None)

        telemetry = scheduler.get_telemetry()
        assert telemetry["pending_tasks"]["P2"] == 5000

        # Now submit 1 critical P0 gameplay task
        t_before = time.perf_counter()
        f_p0 = scheduler.submit(Priority.P0_GAMEPLAY, "p0_vital_physics", lambda: 999)
        t_submitted = time.perf_counter()

        executed = scheduler.step_main_thread(budget_ms=5.0)
        t_after = time.perf_counter()

        submit_duration_ms = (t_submitted - t_before) * 1000.0
        step_duration_ms = (t_after - t_submitted) * 1000.0

        assert executed >= 1, "P0 task was not executed by step_main_thread"
        assert f_p0.done() is True
        assert f_p0.result() == 999
        # Submit and step must both complete in < 2ms without scanning 5000 P2 tasks
        assert submit_duration_ms < 2.0, f"P0 submit took {submit_duration_ms:.3f}ms under 5k queue"
        assert step_duration_ms < 2.0, f"P0 step took {step_duration_ms:.3f}ms under 5k queue"

    def test_active_gameplay_suspends_p2_not_p1(self):
        """Stress: Under active gameplay, P1 telemetry executes but P2 showrunner is halted."""
        scheduler = PriorityScheduler(max_background_workers=2)
        scheduler.start()

        scheduler.set_gameplay_active(True)
        assert scheduler.is_gameplay_active() is True
        assert scheduler.is_system_free() is False

        p1_done = []
        p2_done = []

        for i in range(5):
            scheduler.submit(Priority.P1_TELEMETRY, f"p1_{i}", lambda i=i: p1_done.append(i))
            scheduler.submit(Priority.P2_SHOWRUNNER, f"p2_{i}", lambda i=i: p2_done.append(i))

        # Give workers time to process
        time.sleep(0.05)

        assert len(p1_done) == 5, f"Expected all 5 P1 tasks to complete, got {len(p1_done)}"
        assert len(p2_done) == 0, f"P2 tasks should NOT execute during active gameplay, got {len(p2_done)}"

        # Resume gameplay
        scheduler.set_gameplay_active(False)
        time.sleep(0.05)

        assert len(p2_done) == 5, f"Expected all 5 P2 tasks to complete after gameplay inactive, got {len(p2_done)}"
        scheduler.shutdown(wait=True)


class TestFrameBudgetSpikeThrottling:
    """Stress tests for Cocoa Frame Budget Monitor (>14ms threshold) and oscillation stability."""
    deliberation: str = "Empirical stress testing of FrameBudgetMonitor spike throttling and oscillation stability."

    def test_rapid_oscillations_across_14ms_stability_and_no_deadlock(self):
        """Stress: Verify stability and zero deadlocks under rapid frame time oscillations across 14ms."""
        scheduler = PriorityScheduler(max_background_workers=2, frame_budget_ms=14.0)
        scheduler.start()

        spike_sequence = [0.015, 0.005, 0.016, 0.008, 0.020, 0.006, 0.015, 0.004] * 4  # 32 frames

        completed_tasks: List[int] = []

        def background_task(task_id: int):
            for _ in range(30):
                scheduler.checkpoint_and_yield(timeout_sec=0.005)
                time.sleep(0.0002)
            completed_tasks.append(task_id)

        futures = [scheduler.submit(Priority.P2_SHOWRUNNER, f"bg_{i}", background_task, i) for i in range(4)]

        for dur in spike_sequence:
            scheduler.frame_monitor.start_frame()
            time.sleep(dur)
            scheduler.frame_monitor.end_frame()

        for f in futures:
            f.result(timeout=5.0)

        scheduler.shutdown(wait=True)

        assert len(completed_tasks) == 4
        # 16 of the 32 frames exceeded 14ms (0.015, 0.016, 0.020, 0.015 repeated 4 times)
        assert scheduler.frame_monitor.violation_count == 16

    def test_frame_spike_halts_background_workers(self):
        """Stress: Simulate frame time spikes (>14.0ms, e.g. 15ms, 16.6ms, 20ms) on Thread 0.
        Verify that FrameBudgetMonitor triggers preemption and background workers halt until frame budget normalizes (< 14.0ms).
        """
        scheduler = PriorityScheduler(max_background_workers=1, frame_budget_ms=14.0)
        scheduler.start()

        iterations_during_spike = 0
        running = True

        def heavy_compute():
            nonlocal iterations_during_spike, running
            while running:
                iterations_during_spike += 1
                scheduler.checkpoint_and_yield(timeout_sec=0.05)
                time.sleep(0.0001)

        future = scheduler.submit(Priority.P2_SHOWRUNNER, "heavy_compute", heavy_compute)
        time.sleep(0.005)  # Let worker start

        # Reset iteration count right before starting spiked frame
        iterations_during_spike = 0

        # Simulate a 20ms frame spike on Thread 0
        scheduler.frame_monitor.start_frame()
        time.sleep(0.020)  # 20ms spike (> 14ms)
        telemetry = scheduler.frame_monitor.end_frame()

        running = False
        scheduler.shutdown(wait=True)

        # EMPIRICAL ASSERTION:
        # If FrameBudgetMonitor triggers preemption and halts background workers,
        # the worker should yield and NOT spin hundreds of iterations during the 20ms spike.
        # If the worker executed > 50 iterations, it completely failed to halt!
        assert telemetry.budget_exceeded is True
        assert iterations_during_spike <= 20, (
            f"Background worker failed to halt during 20ms frame spike! "
            f"Ran {iterations_during_spike} iterations while Thread 0 frame exceeded 14ms budget."
        )


class TestArtifactEnvelopeCryptographicStress:
    """Stress tests for ArtifactMetadataEnvelope cryptographic integrity, serialization, and tampering."""
    deliberation: str = "Empirical stress testing of ArtifactMetadataEnvelope cryptographic integrity and serialization."

    def test_tamper_detection_in_memory(self):
        """Stress: Mutate 1 byte in memory for dict, list, string, and verify verify_integrity() fails."""
        # Dict payload
        env_dict = ArtifactMetadataEnvelope.seal(
            deliberation="Dict test",
            artifact_type=ArtifactType.BEHAVIORAL_PROFILE,
            payload={"velocity_x": 100.0, "dash_count": 5},
        )
        assert env_dict.verify_integrity() is True
        env_dict.payload["velocity_x"] = 100.1
        assert env_dict.verify_integrity() is False
        with pytest.raises(ArtifactIntegrityError):
            env_dict.verify_integrity(raise_on_error=True)

        # List payload
        env_list = ArtifactMetadataEnvelope.seal(
            deliberation="List test",
            artifact_type=ArtifactType.GENESIS_SPEC,
            payload=[1, 2, 3],
        )
        assert env_list.verify_integrity() is True
        env_list.payload[1] = 99
        assert env_list.verify_integrity() is False

        # String payload
        env_str = ArtifactMetadataEnvelope.seal(
            deliberation="String test",
            artifact_type=ArtifactType.UPDATE_SPEC,
            payload="original text",
        )
        assert env_str.verify_integrity() is True
        env_str.payload = "tampered text"
        assert env_str.verify_integrity() is False

    def test_tamper_detection_on_disk(self, tmp_path):
        """Stress: Mutate 1 byte in disk payload and verify verify_integrity() detects tampering."""
        env = ArtifactMetadataEnvelope.seal(
            deliberation="Disk tamper test",
            artifact_type=ArtifactType.GAME_MODULE,
            payload={"entities": ["Player", "Enemy"], "health": 100},
        )
        saved_envelope = env.save_to_disk(tmp_path, separate_payload_file=True)
        assert Path(env.payload_path).exists()
        assert env.verify_integrity() is True

        # Mutate 1 byte on disk
        payload_file = Path(env.payload_path)
        content = payload_file.read_bytes()
        tampered_content = content[:-2] + b"X" + content[-1:]
        payload_file.write_bytes(tampered_content)

        # Verify envelope on disk and in memory
        assert env.verify_integrity() is False
        with pytest.raises(ArtifactIntegrityError):
            env.verify_integrity(raise_on_error=True)

        with pytest.raises(ArtifactIntegrityError):
            ArtifactMetadataEnvelope.load_from_disk(saved_envelope, verify=True)

    def test_missing_disk_payload_verify_integrity_returns_false(self, tmp_path):
        """Stress: Verify verify_integrity(raise_on_error=False) returns False when payload file is missing."""
        env = ArtifactMetadataEnvelope.seal(
            deliberation="Missing file test",
            artifact_type=ArtifactType.BEHAVIORAL_PROFILE,
            payload={"key": "val"},
        )
        saved_envelope = env.save_to_disk(tmp_path, separate_payload_file=True)
        # Delete payload file
        Path(env.payload_path).unlink()

        # Calling verify_integrity(raise_on_error=False) should return False, not crash with unhandled exception
        result = env.verify_integrity(raise_on_error=False)
        assert result is False

    def test_separate_payload_serialization_hash_invariance(self, tmp_path):
        """Stress: Verify that saving with separate_payload_file does NOT mutate the sealed sha256_hash."""
        original_payload = {"speed": 1.5, "cooldown": 0.25, "tags": ["kinetic", "rusher"]}
        env = ArtifactMetadataEnvelope.seal(
            deliberation="Hash invariance test",
            artifact_type=ArtifactType.BEHAVIORAL_PROFILE,
            payload=original_payload,
        )
        sealed_hash = env.sha256_hash

        # Save to disk with separate payload file
        saved_path = env.save_to_disk(tmp_path, separate_payload_file=True)

        # EMPIRICAL ASSERTION:
        # Sealing an artifact produces an authoritative cryptographic contract.
        # Saving the artifact to disk must NOT alter its recorded sha256_hash!
        assert env.sha256_hash == sealed_hash, (
            f"Envelope sha256_hash was mutated upon saving to disk! "
            f"Sealed: {sealed_hash}, After save: {env.sha256_hash}"
        )

    def test_binary_payload_serialization(self, tmp_path):
        """Stress: Verify binary (bytes) payload handling in separate payload serialization."""
        binary_data = b"\x00\x01\x02\xff\xfe\xfd\xaa\x55"
        env = ArtifactMetadataEnvelope.seal(
            deliberation="Binary asset payload test",
            artifact_type=ArtifactType.ASSET_BUNDLE,
            payload=binary_data,
        )
        original_hash = env.sha256_hash
        assert env.verify_integrity() is True

        saved_path = env.save_to_disk(tmp_path, separate_payload_file=True)

        payload_path = Path(env.payload_path)
        disk_bytes = payload_path.read_bytes()

        # EMPIRICAL ASSERTION:
        # Binary payloads must be written as authentic binary bytes, NOT as str(b"...")
        assert disk_bytes == binary_data, (
            f"Binary payload was corrupted on disk! "
            f"Expected {len(binary_data)} bytes ({binary_data}), but got {len(disk_bytes)} bytes ({disk_bytes})"
        )
        assert env.sha256_hash == original_hash
        assert env.verify_integrity() is True

    def test_non_ascii_unicode_payload_handling(self, tmp_path):
        """Stress: Verify multi-lingual UTF-8 and emoji payloads preserve integrity on disk."""
        payload = {
            "title": "英雄の試練",
            "emoji_banner": "⚔️🛡️🔥🏆",
            "locales": {"zh": "冒险模式", "ru": "Приключение"},
        }
        env = ArtifactMetadataEnvelope.seal(
            deliberation="Non-ASCII Unicode payload test",
            artifact_type=ArtifactType.GENESIS_SPEC,
            payload=payload,
        )
        original_hash = env.sha256_hash
        saved_path = env.save_to_disk(tmp_path, separate_payload_file=True)

        # Loaded envelope must verify integrity
        loaded = ArtifactMetadataEnvelope.load_from_disk(saved_path, verify=True)
        assert loaded.sha256_hash == original_hash
        assert loaded.verify_integrity() is True

    def test_rationale_first_field_ordering_in_json(self):
        """Stress: Verify 'deliberation' key is strictly first in JSON output."""
        env = ArtifactMetadataEnvelope.seal(
            deliberation="MANDATORY FIRST PROPERTY: Deliberation rationale.",
            artifact_type=ArtifactType.UPDATE_SPEC,
            payload={"rate": 1.2},
            source_agent="Director",
            target_agent="Compiler",
        )
        json_str = env.to_json()
        raw_dict = json.loads(json_str)

        assert list(raw_dict.keys())[0] == "deliberation"

        # Also verify raw string formatting has deliberation immediately after opening brace
        lines = [line.strip() for line in json_str.splitlines() if line.strip()]
        assert lines[0] == "{"
        assert lines[1].startswith('"deliberation":')
