"""engine/scheduler/scheduler.py
3-Tier Resource-Aware OS Priority Scheduler for macOS 60 FPS Engine.
Enforces Rationale-First Epistemic Architecture (deliberation declared first in all classes).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import heapq
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger("PriorityScheduler")


class Priority(IntEnum):
    """Authoritative Three-Tier OS Priority Hierarchy.
    
    Lower integer value = Higher execution priority in min-heap queues.
    """
    deliberation: str
    P0_GAMEPLAY = 0    # Synchronous Cocoa main thread event pump, 60 FPS rendering, input, physics
    P1_TELEMETRY = 1   # High-priority background SQLite WAL event queue flushing & HUD monitor
    P2_SHOWRUNNER = 2  # Background SWE game compilation, asset synthesis, synthetic playtesting

    # Compatibility aliases for tests/test_tier1_units.py
    PRIORITY_0_GAMEPLAY = 0
    PRIORITY_1_TELEMETRY = 1
    PRIORITY_2_SHOWRUNNER = 2


TaskPriority = Priority


class TaskState(str, Enum):
    """Lifecycle states of a scheduled task."""
    deliberation: str
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"          # Yielded due to active gameplay or frame budget
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class ScheduledTaskFuture:
    """Thread-safe future for retrieving scheduled task execution results."""
    deliberation: str = "Thread-safe future for retrieving task results."

    def __init__(self, task_id: str, deliberation: str = "Thread-safe future for retrieving task results."):
        self.deliberation = deliberation
        self.task_id = task_id
        self._event = threading.Event()
        self._result: Any = None
        self._exception: Optional[BaseException] = None
        self._cancelled: bool = False

    def set_result(self, result: Any) -> None:
        self._result = result
        self._event.set()

    def set_exception(self, exc: BaseException) -> None:
        self._exception = exc
        self._event.set()

    def cancel(self) -> bool:
        if self._event.is_set():
            return False
        self._cancelled = True
        self._event.set()
        return True

    def done(self) -> bool:
        return self._event.is_set()

    def cancelled(self) -> bool:
        return self._cancelled

    def result(self, timeout: Optional[float] = None) -> Any:
        if not self._event.wait(timeout=timeout):
            raise TimeoutError(f"Task {self.task_id} timed out waiting for result")
        if self._cancelled:
            raise RuntimeError(f"Task {self.task_id} was cancelled")
        if self._exception is not None:
            raise self._exception
        return self._result


@dataclass(order=True)
class ScheduledTask:
    """Internal task envelope managed by PriorityScheduler min-heap."""
    deliberation: str = field(default="", compare=False)
    priority: int = 0
    sequence_id: int = 0
    task_id: str = field(default="", compare=False)
    name: str = field(default="", compare=False)
    fn: Callable[..., Any] = field(default=lambda: None, compare=False)
    args: Tuple[Any, ...] = field(default_factory=tuple, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    can_preempt: bool = field(default=True, compare=False)
    future: Optional[ScheduledTaskFuture] = field(default=None, compare=False)
    created_at_utc: float = field(default_factory=time.time, compare=False)
    state: TaskState = field(default=TaskState.PENDING, compare=False)
    yield_count: int = field(default=0, compare=False)
    checkpoint_state: Optional[Dict[str, Any]] = field(default=None, compare=False)


@dataclass
class FrameTelemetry:
    """Frame measurement metrics captured by FrameBudgetMonitor."""
    deliberation: str = field(default="Frame execution timing telemetry.")
    frame_index: int = 0
    elapsed_ms: float = 0.0
    fps: float = 60.0
    budget_exceeded: bool = False
    budget_threshold_ms: float = 14.0


class FrameBudgetMonitor:
    """Monitors Thread 0 frame-time budget to protect 60 FPS Cocoa main thread."""
    deliberation: str = "Monitors Thread 0 frame-time budget to protect 60 FPS Cocoa main thread."

    def __init__(
        self,
        target_fps: int = 60,
        budget_threshold_ms: float = 14.0,
        deliberation: str = "Monitors Thread 0 frame-time budget to protect 60 FPS Cocoa main thread.",
        scheduler: Optional[Any] = None,
    ):
        self.deliberation = deliberation
        self.target_fps = target_fps
        self.frame_interval_ms = 1000.0 / target_fps  # 16.666 ms
        self.budget_threshold_ms = budget_threshold_ms
        self.frame_count: int = 0
        self._frame_start_time: float = 0.0
        self._last_frame_telemetry: Optional[FrameTelemetry] = None
        self._violation_count: int = 0
        self.scheduler: Optional[Any] = scheduler

    def start_frame(self) -> None:
        """Called at the beginning of each Cocoa main loop frame."""
        self.frame_count += 1
        self._frame_start_time = time.perf_counter()

    def get_elapsed_ms(self) -> float:
        """Returns elapsed time in milliseconds since current frame started."""
        if self._frame_start_time == 0.0:
            return 0.0
        return (time.perf_counter() - self._frame_start_time) * 1000.0

    def is_budget_exceeded(self) -> bool:
        """Returns True if current frame duration exceeds safety threshold (14ms)."""
        exceeded = self.get_elapsed_ms() > self.budget_threshold_ms
        if exceeded and self.scheduler is not None:
            self.scheduler.request_preemption(reason="frame_budget_exceeded")
        return exceeded

    def end_frame(self) -> FrameTelemetry:
        """Called at the end of each frame after render_and_present()."""
        elapsed = self.get_elapsed_ms()
        exceeded = elapsed > self.budget_threshold_ms
        if exceeded:
            self._violation_count += 1

        fps = 1000.0 / elapsed if elapsed > 0.0 else float(self.target_fps)
        telemetry = FrameTelemetry(
            deliberation=f"Frame {self.frame_count} completed in {elapsed:.2f}ms (exceeded={exceeded})",
            frame_index=self.frame_count,
            elapsed_ms=elapsed,
            fps=fps,
            budget_exceeded=exceeded,
            budget_threshold_ms=self.budget_threshold_ms,
        )
        self._last_frame_telemetry = telemetry
        self._frame_start_time = 0.0

        if self.scheduler is not None:
            self.scheduler._on_frame_end(telemetry)

        return telemetry

    @property
    def violation_count(self) -> int:
        return self._violation_count


class PriorityScheduler:
    """Asynchronous Resource-Aware OS Priority Scheduler with cooperative yielding.
    
    Coordinates:
      - P0 (Gameplay): Synchronous Cocoa main loop, 60 FPS rendering, input.
      - P1 (Telemetry): SQLite WAL queue drain, HUD monitoring.
      - P2 (Showrunner): Heavy background analysis, compilation, synthetic playtesting.
    """
    deliberation: str = "Asynchronous Resource-Aware OS Priority Scheduler with cooperative yielding."

    def __init__(
        self,
        max_background_workers: int = 2,
        frame_budget_ms: float = 14.0,
        deliberation: str = "Asynchronous Resource-Aware OS Priority Scheduler with cooperative yielding.",
        consecutive_frames_to_resume: int = 1,
    ):
        self.deliberation = deliberation
        self.max_background_workers = max_background_workers
        self.frame_monitor = FrameBudgetMonitor(
            target_fps=60,
            budget_threshold_ms=frame_budget_ms,
            scheduler=self,
        )
        self._consecutive_normal_frames: int = 0
        self._consecutive_normal_required: int = consecutive_frames_to_resume

        # Thread synchronization primitives
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._preemption_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()  # Initially running/resumed
        self._gameplay_active = threading.Event()
        self._stop_event = threading.Event()

        # Queues & Tasks
        self._task_heap: List[ScheduledTask] = []
        self._active_tasks: Dict[str, ScheduledTask] = {}
        self._sequence_counter: int = 0
        self._total_yield_count: int = 0
        self._preemption_request_count: int = 0

        # Background worker threads for P1 and P2
        self._workers: List[threading.Thread] = []
        self._is_started: bool = False
        self._workers_ready_count: int = 0

    def start(self) -> None:
        """Spawns background worker threads for P1 and P2 execution."""
        with self._lock:
            if self._is_started:
                return
            self._is_started = True
            self._stop_event.clear()
            self._workers_ready_count = 0
            for i in range(self.max_background_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"OS-Scheduler-Worker-{i}",
                    daemon=True,
                )
                self._workers.append(worker)
                worker.start()

            # Wait for workers to initialize and enter listening state
            deadline = time.perf_counter() + 1.0
            while self._workers_ready_count < self.max_background_workers and time.perf_counter() < deadline:
                self._condition.wait(timeout=0.01)

    def shutdown(self, wait: bool = True, timeout: float = 2.0) -> None:
        """Signals stop event and joins background worker threads."""
        with self._lock:
            self._stop_event.set()
            self._resume_event.set()
            self._condition.notify_all()

        if wait:
            for worker in self._workers:
                worker.join(timeout=timeout)
        self._workers.clear()
        self._is_started = False

    def set_gameplay_active(self, active: bool) -> None:
        """Sets active combat state. When active, background P2 tasks are preempted."""
        with self._lock:
            if active:
                self._gameplay_active.set()
                self.request_preemption(reason="gameplay_active")
            else:
                self._gameplay_active.clear()
                self.resume_background_tasks()

    def is_gameplay_active(self) -> bool:
        return self._gameplay_active.is_set()

    def request_preemption(self, reason: str = "general") -> None:
        """Raises preemption event to force running P2 tasks to checkpoint and yield."""
        with self._lock:
            self._preemption_request_count += 1
            self._preemption_event.set()
            self._resume_event.clear()
            logger.debug("Scheduler preemption requested: reason=%s (count=%d)", reason, self._preemption_request_count)

    def resume_background_tasks(self) -> None:
        """Clears preemption and resumes paused background tasks if system is free."""
        with self._lock:
            if not self._gameplay_active.is_set() and not self.frame_monitor.is_budget_exceeded():
                self._preemption_event.clear()
                self._resume_event.set()
                self._condition.notify_all()
                logger.debug("Scheduler resumed background tasks")

    def _on_frame_end(self, telemetry: FrameTelemetry) -> None:
        """Called by FrameBudgetMonitor.end_frame() to manage throttling state."""
        with self._lock:
            if telemetry.budget_exceeded:
                self._consecutive_normal_frames = 0
                self.request_preemption(reason="frame_budget_exceeded")
            else:
                self._consecutive_normal_frames += 1
                if self._consecutive_normal_frames >= self._consecutive_normal_required:
                    self.resume_background_tasks()

    def is_system_free(self) -> bool:
        """Returns True iff system has idle headroom for background P2 showrunner compute.
        
        System is free when:
          1. Active interactive combat is NOT running.
          2. Preemption has NOT been explicitly requested.
          3. Frame budget is NOT exceeded.
        """
        with self._lock:
            if self._gameplay_active.is_set():
                return False
            if self._preemption_event.is_set():
                return False
            if self.frame_monitor.is_budget_exceeded():
                return False
            return True

    def should_yield(self) -> bool:
        """Checks if a cooperative worker should yield right now."""
        return (
            self._preemption_event.is_set()
            or self._gameplay_active.is_set()
            or self.frame_monitor.is_budget_exceeded()
        )

    def checkpoint_and_yield(
        self,
        timeout_sec: float = 0.05,
        checkpoint_data: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> bool:
        """Cooperative checkpoint called by long-running P2 loops.
        
        If preemption is requested, gameplay is active, or frame budget is exceeded:
          - Stores checkpoint state (if provided)
          - Yields CPU/GIL execution slices via _resume_event.wait(timeout_sec)
          - Returns True (indicating a yield occurred)
        If frame is actively rendering on Cocoa Thread 0:
          - Cooperatively yields a 1ms CPU slice to preserve 60 FPS
          - Returns True
        If system is free and idle:
          - Returns False immediately (< 1 microsecond overhead)
        """
        # Proactively trigger preemption if budget threshold was just crossed mid-frame
        if self.frame_monitor.is_budget_exceeded():
            self.request_preemption(reason="frame_budget_exceeded")

        if self.should_yield():
            with self._lock:
                self._total_yield_count += 1
                if task_id and task_id in self._active_tasks:
                    task = self._active_tasks[task_id]
                    task.yield_count += 1
                    task.state = TaskState.PAUSED
                    if checkpoint_data:
                        task.checkpoint_state = checkpoint_data

            # Sleep/wait to yield GIL and CPU cores back to Thread 0 and P1
            # When preemption is active, _resume_event is cleared, so wait() blocks
            if not self._resume_event.is_set():
                is_cocoa_or_combat_active = (
                    self.frame_monitor.get_elapsed_ms() > 0.0
                    or self.frame_monitor.is_budget_exceeded()
                    or self._gameplay_active.is_set()
                )
                wait_timeout = timeout_sec if is_cocoa_or_combat_active else min(timeout_sec, 0.0005)
                self._resume_event.wait(timeout=wait_timeout)
            else:
                time.sleep(min(timeout_sec, 0.005))

            with self._lock:
                if task_id and task_id in self._active_tasks:
                    self._active_tasks[task_id].state = TaskState.RUNNING

            return True

        # When a frame is actively executing on Thread 0, yield a 2ms slice
        # to ensure background workers never starve Cocoa Thread 0 of GIL/CPU
        if self.frame_monitor.get_elapsed_ms() > 0.0:
            time.sleep(0.002)
            return True

        return False

    def submit(
        self,
        priority: Priority,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        can_preempt: bool = True,
        **kwargs: Any,
    ) -> ScheduledTaskFuture:
        """Enqueues a task into the priority min-heap."""
        task_id = f"task_{int(priority)}_{uuid.uuid4().hex[:8]}"
        future = ScheduledTaskFuture(task_id)

        with self._lock:
            self._sequence_counter += 1
            task = ScheduledTask(
                deliberation=f"Scheduled task '{name}' at priority {priority.name}",
                priority=int(priority),
                sequence_id=self._sequence_counter,
                task_id=task_id,
                name=name,
                fn=fn,
                args=args,
                kwargs=kwargs,
                can_preempt=can_preempt,
                future=future,
            )
            heapq.heappush(self._task_heap, task)
            self._condition.notify()

        return future

    def step_main_thread(self, budget_ms: float = 2.0) -> int:
        """Drains pending P0 tasks directly on Cocoa Thread 0 within small time slice."""
        start = time.perf_counter()
        executed = 0

        while True:
            elapsed = (time.perf_counter() - start) * 1000.0
            if elapsed >= budget_ms:
                break

            task = None
            with self._lock:
                if self._task_heap and self._task_heap[0].priority == Priority.P0_GAMEPLAY:
                    task = heapq.heappop(self._task_heap)
                    self._active_tasks[task.task_id] = task

            if task is None:
                break

            task.state = TaskState.RUNNING
            try:
                result = task.fn(*task.args, **task.kwargs)
                task.state = TaskState.COMPLETED
                task.future.set_result(result)
            except BaseException as e:
                task.state = TaskState.FAILED
                task.future.set_exception(e)
            finally:
                with self._lock:
                    self._active_tasks.pop(task.task_id, None)
            executed += 1

        return executed

    def _worker_loop(self) -> None:
        """Background worker thread popping P1 and P2 tasks."""
        with self._lock:
            self._workers_ready_count += 1
            self._condition.notify_all()

        while not self._stop_event.is_set():
            task: Optional[ScheduledTask] = None

            with self._lock:
                while not self._stop_event.is_set() and (not self._task_heap or self._task_heap[0].priority == Priority.P0_GAMEPLAY):
                    # P0 tasks are reserved for Cocoa main thread; workers wait for P1/P2 tasks
                    self._condition.wait(timeout=0.1)

                if self._stop_event.is_set():
                    break

                if self._task_heap and self._task_heap[0].priority != Priority.P0_GAMEPLAY:
                    # If highest priority task is P2, verify system is free or wait
                    if self._task_heap[0].priority == Priority.P2_SHOWRUNNER and not self.is_system_free():
                        self._condition.wait(timeout=0.05)
                        continue

                    task = heapq.heappop(self._task_heap)
                    self._active_tasks[task.task_id] = task

            if task is None:
                continue

            # Execute task
            task.state = TaskState.RUNNING
            try:
                result = task.fn(*task.args, **task.kwargs)
                task.state = TaskState.COMPLETED
                task.future.set_result(result)
            except BaseException as e:
                task.state = TaskState.FAILED
                task.future.set_exception(e)
            finally:
                with self._lock:
                    self._active_tasks.pop(task.task_id, None)

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns runtime scheduler telemetry for epistemic observability and HUD."""
        with self._lock:
            pending_p0 = sum(1 for t in self._task_heap if t.priority == Priority.P0_GAMEPLAY)
            pending_p1 = sum(1 for t in self._task_heap if t.priority == Priority.P1_TELEMETRY)
            pending_p2 = sum(1 for t in self._task_heap if t.priority == Priority.P2_SHOWRUNNER)
            return {
                "deliberation": f"Telemetry snapshot: {len(self._task_heap)} pending tasks, active_tasks={len(self._active_tasks)}, yields={self._total_yield_count}",
                "is_gameplay_active": self._gameplay_active.is_set(),
                "is_preemption_requested": self._preemption_event.is_set(),
                "is_system_free": self.is_system_free(),
                "pending_tasks": {
                    "P0": pending_p0,
                    "P1": pending_p1,
                    "P2": pending_p2,
                    "total": len(self._task_heap),
                },
                "active_tasks_count": len(self._active_tasks),
                "total_yield_count": self._total_yield_count,
                "preemption_request_count": self._preemption_request_count,
                "frame_violations": self.frame_monitor.violation_count,
                "consecutive_normal_frames": self._consecutive_normal_frames,
            }
