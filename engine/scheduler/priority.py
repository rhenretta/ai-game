"""engine/scheduler/priority.py
Priority scheduling primitives, budget monitors, and task futures.
"""
from __future__ import annotations
from engine.scheduler.scheduler import (
    FrameBudgetMonitor,
    FrameTelemetry,
    Priority,
    PriorityScheduler,
    ScheduledTask,
    ScheduledTaskFuture,
    TaskPriority,
    TaskState,
)

# Ensure backward-compatible properties and methods
if not hasattr(ScheduledTaskFuture, "is_completed"):
    ScheduledTaskFuture.is_completed = property(lambda self: self.done())

if not hasattr(FrameBudgetMonitor, "begin_frame"):
    FrameBudgetMonitor.begin_frame = lambda self: self.start_frame()

__all__ = [
    "FrameBudgetMonitor",
    "FrameTelemetry",
    "Priority",
    "PriorityScheduler",
    "ScheduledTask",
    "ScheduledTaskFuture",
    "TaskPriority",
    "TaskState",
]
