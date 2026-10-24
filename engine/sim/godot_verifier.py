"""engine/sim/godot_verifier.py
Godot 4 Headless Verification, Stderr Compiler Diagnostics, and Simulator Host.
Bridges Godot 4 execution to Python Host, TelemetryRecorder, and SQLite WAL (Rule 6).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from engine.godot_runtime import ensure_godot_binary, run_godot_command
from engine.sim.headless import (
    InMemoryTelemetryRecorder,
    SimulationInput,
    SimulationState,
)
from engine.telemetry.recorder import NullTelemetryRecorder, TelemetryRecorder

logger = logging.getLogger("GodotVerifier")

HARNESS_SCRIPT_PATH = Path(__file__).resolve().parent / "godot_harness.gd"


class GDScriptDiagnostic(NamedTuple):
    file_path: Path
    line_number: int
    error_message: str
    raw_line: str


class GodotSimResult(NamedTuple):
    success: bool
    error_message: Optional[str]
    stderr: str
    state: SimulationState
    avg_frame_time_ms: float = 0.0
    frame_budget_passed: bool = True
    ticks_executed: int = 0


GDSCRIPT_ERROR_PATTERNS = [
    # Pattern 1: res://path.gd:123 - SCRIPT ERROR: Parse Error: ...
    re.compile(r'(?:at:\s*)?(res://[^\s:]+):(\d+)\s*-\s*SCRIPT ERROR:\s*(?:Parse Error:\s*)?(.*)'),
    # Pattern 2: at: GDScript::reload (res://path.gd:123)
    re.compile(r'SCRIPT ERROR:\s*(?:Parse Error:\s*)?(.*?)\s+at:\s*GDScript::\w+\s*\((res://[^:]+):(\d+)\)'),
    # Pattern 3: ERROR: ... [Resource file res://path:123]
    re.compile(r'ERROR:\s*.*?\[Resource file (res://[^:]+):(\d+)\]:\s*(.*)'),
    # Pattern 4: Script error at "res://path.gd":123 - description
    re.compile(r'Script error at\s*"(res://[^"]+)":(\d+)\s*-\s*(.*)'),
    # Pattern 5: at: load (modules/gdscript/gdscript.cpp:2936)
    re.compile(r'ERROR:\s*Failed to load script "(res://[^"]+)" with error "(.*?)"'),
]


MULTILINE_ERROR_PATTERNS = [
    re.compile(r'SCRIPT ERROR:\s*(?:Parse Error:\s*)?([^\n]+)\n\s*at:\s*[^\n]*?\((res://[^:]+):(\d+)\)'),
    re.compile(r'ERROR:\s*([^\n]+)\n\s*at:\s*[^\n]*?\((res://[^:]+):(\d+)\)'),
]


ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')


def parse_godot_stderr(cartridge_dir: Union[Path, str], stderr_text: str) -> List[GDScriptDiagnostic]:
    """Parses Godot 4 stderr stream and extracts structured GDScript diagnostics."""
    diagnostics: List[GDScriptDiagnostic] = []
    c_dir = Path(cartridge_dir).resolve()
    stderr_text = ANSI_ESCAPE_RE.sub("", stderr_text)

    matched_spans: List[Tuple[int, int]] = []
    for pat in MULTILINE_ERROR_PATTERNS:
        for m in pat.finditer(stderr_text):
            err_msg = m.group(1).strip()
            res_path = m.group(2).strip()
            line_no = int(m.group(3))
            rel_path = res_path.replace("res://", "")
            resolved_file = c_dir / rel_path
            diagnostics.append(
                GDScriptDiagnostic(
                    file_path=resolved_file,
                    line_number=line_no,
                    error_message=err_msg,
                    raw_line=m.group(0),
                )
            )
            matched_spans.append(m.span())

    # Check single-line errors (including native-type script attachment errors)
    for line in stderr_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if "Script inherits from native type" in line and "can't be assigned to an object of type" in line:
            diagnostics.append(
                GDScriptDiagnostic(
                    file_path=c_dir / "main.gd",
                    line_number=1,
                    error_message=line,
                    raw_line=line,
                )
            )
            continue

        if not matched_spans:
            matched = False
            for pat in GDSCRIPT_ERROR_PATTERNS:
                m = pat.search(line)
                if m:
                    groups = m.groups()
                    if len(groups) == 3:
                        if groups[0].startswith("res://"):
                            res_path = groups[0]
                            line_no = int(groups[1])
                            err_msg = groups[2].strip()
                        elif groups[1].startswith("res://"):
                            err_msg = groups[0].strip()
                            res_path = groups[1]
                            line_no = int(groups[2])
                        else:
                            continue
                    elif len(groups) == 2:
                        res_path = groups[0]
                        line_no = 1
                        err_msg = groups[1].strip()
                    else:
                        continue

                    rel_path = res_path.replace("res://", "")
                    resolved_file = c_dir / rel_path
                    diagnostics.append(
                        GDScriptDiagnostic(
                            file_path=resolved_file,
                            line_number=line_no,
                            error_message=err_msg,
                            raw_line=line,
                        )
                    )
                    matched = True
                    break

            if not matched and ("SCRIPT ERROR:" in line or "Parse Error:" in line):
                diagnostics.append(
                    GDScriptDiagnostic(
                        file_path=c_dir / "main.gd",
                        line_number=1,
                        error_message=line,
                        raw_line=line,
                    )
                )

    # Filter out cascading preload errors ("Could not preload resource script 'res://X.gd'")
    # when X.gd itself already has a direct syntax/parse diagnostic
    direct_broken_files = {
        d.file_path.name
        for d in diagnostics
        if "Could not preload resource script" not in d.error_message
        and "Failed to load script" not in d.error_message
    }
    if direct_broken_files:
        filtered: List[GDScriptDiagnostic] = []
        for d in diagnostics:
            if (
                "Could not preload resource script" in d.error_message
                or "Failed to load script" in d.error_message
            ) and any(bf in d.error_message for bf in direct_broken_files):
                continue
            filtered.append(d)
        if filtered:
            diagnostics = filtered

    return diagnostics


def verify_godot_cartridge_syntax(
    cartridge_dir: Union[Path, str],
    timeout: float = 30.0,
) -> Tuple[bool, List[GDScriptDiagnostic]]:
    """Validates Godot 4 cartridge syntax and project integrity headlessly across all .gd files
    plus a 2-tick runtime initialization check.
    Returns (is_valid, diagnostics).
    """
    c_dir = Path(cartridge_dir).resolve()
    if not (c_dir / "project.godot").is_file():
        return False, [
            GDScriptDiagnostic(
                file_path=c_dir / "project.godot",
                line_number=1,
                error_message="Missing project.godot engine configuration file.",
                raw_line="Missing project.godot",
            )
        ]

    diagnostics: List[GDScriptDiagnostic] = []

    # 1. Run check-only on every .gd script in the cartridge so dynamically loaded scripts are also compiled
    gd_files = sorted(c_dir.glob("*.gd"))
    for gd_file in gd_files:
        res_script = run_godot_command(
            ["--path", str(c_dir), "--check-only", "--script", f"res://{gd_file.name}"],
            cwd=c_dir,
            timeout=timeout,
            headless=True,
        )
        script_diags = parse_godot_stderr(c_dir, res_script.stderr)
        for d in script_diags:
            if d not in diagnostics:
                diagnostics.append(d)

    # 2. Run project boot check (or 2-tick harness smoke check if main.gd exists and no compile errors yet)
    if not diagnostics and (c_dir / "main.gd").is_file() and HARNESS_SCRIPT_PATH.is_file():
        res = run_godot_command(
            ["--path", str(c_dir), "-s", str(HARNESS_SCRIPT_PATH), "--max-ticks", "2"],
            cwd=c_dir,
            timeout=timeout,
            headless=True,
        )
    else:
        res = run_godot_command(
            ["--path", str(c_dir), "--quit"],
            cwd=c_dir,
            timeout=timeout,
            headless=True,
        )

    boot_diags = parse_godot_stderr(c_dir, res.stderr)
    for d in boot_diags:
        if d not in diagnostics:
            diagnostics.append(d)

    is_valid = (res.returncode == 0) and (len(diagnostics) == 0)
    return is_valid, diagnostics


class GodotHeadlessSimulator:
    """Headless simulation runner for Godot 4 cartridges.
    Matches interface of HeadlessSimulator for seamless polymorphism.
    """

    def __init__(
        self,
        cartridge_path: Union[Path, str],
        telemetry_mode: str = "sqlite",
        telemetry_db_path: Optional[Path] = None,
        max_ticks: int = 60,
        ticks: Optional[int] = None,
        max_frames: Optional[int] = None,
        session_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        capture_viewport: bool = False,
        capture_ticks: Optional[List[int]] = None,
        input_script: Optional[List[Dict[str, Any]]] = None,
        setup_script: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        self.cartridge_path = Path(cartridge_path).resolve()
        self.telemetry_mode = telemetry_mode
        self.max_ticks = max_frames or ticks or max_ticks
        self.capture_viewport = capture_viewport
        self.capture_ticks = sorted(set(int(t) for t in (capture_ticks or [])))
        self.input_script = input_script
        self.setup_script = setup_script
        self.viewport_captured = False
        self.step_count = 0
        self.elapsed_time = 0.0

        self.avg_frame_time_ms = 0.0
        self.max_frame_time_ms = 0.0
        self.frame_budget_passed = True
        self.ticks_executed = 0
        self.pos_delta = 0.0
        self.progression_verified = True
        self.is_completed = False
        self.outcome = "RUNNING"

        self.player_pos: Tuple[float, float] = (0.0, 0.0)
        self.player_vel: Tuple[float, float] = (0.0, 0.0)
        self.player_health: float = 100.0
        self.scene_audit: Dict[str, Any] = {}
        self.final_snapshot: Dict[str, Any] = {}
        self.tick_snapshots: Dict[int, Dict[str, Any]] = {}
        self.qa_setup_calls: List[Dict[str, Any]] = []

        # Initialize telemetry recorder
        if telemetry_mode in ("sqlite", "wal"):
            db_path = telemetry_db_path or (self.cartridge_path.parent.parent / "data" / "telemetry.db")
            self.telemetry_recorder = TelemetryRecorder(
                db_path=db_path,
                auto_register_probes=True,
            )
            self.telemetry_recorder.start_session()
            if session_id:
                self.telemetry_recorder.session_id = session_id
            self.telemetry_recorder.start_episode()
            if episode_id:
                self.telemetry_recorder.current_episode_id = episode_id
        elif telemetry_mode in ("in_memory", "memory"):
            self.telemetry_recorder = InMemoryTelemetryRecorder()
        else:
            self.telemetry_recorder = NullTelemetryRecorder()

    def get_events_by_type(self, event_type: Any) -> List[Any]:
        if self.telemetry_recorder and hasattr(self.telemetry_recorder, "get_events_by_type"):
            return self.telemetry_recorder.get_events_by_type(event_type)
        return []

    def get_events(self, event_type: Any) -> List[Any]:
        return self.get_events_by_type(event_type)

    def run_until_complete(
        self,
        max_steps: Optional[int] = None,
        capture_viewport: Optional[bool] = None,
    ) -> SimulationState:
        """Executes Godot playtest harness and streams metrics (and real viewport PNGs when capture_viewport=True)."""
        import platform
        import shutil

        limit = max_steps or self.max_ticks
        if self.capture_ticks:
            limit = max(limit, max(self.capture_ticks))
        use_viewport = self.capture_viewport if capture_viewport is None else bool(capture_viewport)
        extra_args: List[str] = []
        if self.capture_ticks:
            extra_args += ["--capture-ticks", ",".join(str(t) for t in self.capture_ticks)]
        if self.input_script is not None:
            script_path = self.cartridge_path / "assets" / "playtest_captures" / "_input_script.json"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(json.dumps(self.input_script), encoding="utf-8")
            extra_args += ["--input-script", str(script_path)]
        if self.setup_script:
            setup_path = self.cartridge_path / "assets" / "playtest_captures" / "_setup_script.json"
            setup_path.parent.mkdir(parents=True, exist_ok=True)
            setup_path.write_text(json.dumps(self.setup_script), encoding="utf-8")
            extra_args += ["--setup-script", str(setup_path)]
        godot_bin = ensure_godot_binary()

        if use_viewport:
            cap_dir = self.cartridge_path / "assets" / "playtest_captures"
            for stale_name in ("playtest_t0.png", "playtest_tend.png", "playtest_collision_overlay.png"):
                stale_p = cap_dir / stale_name
                if stale_p.is_file():
                    try:
                        stale_p.unlink()
                    except Exception:
                        pass

        sys_name = platform.system()
        has_display = (sys_name == "Darwin") or bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))
        xvfb_bin = shutil.which("xvfb-run") if (sys_name == "Linux" and not has_display) else None

        if use_viewport and (has_display or xvfb_bin):
            cmd: List[str] = []
            if xvfb_bin:
                cmd.extend([xvfb_bin, "-a"])
            cmd.extend([
                str(godot_bin),
                "--audio-driver", "Dummy",
                "--position", "-4000,-4000",
                "--path", str(self.cartridge_path),
                "-s", str(HARNESS_SCRIPT_PATH),
                "--max-ticks", str(limit),
            ] + extra_args)
        else:
            cmd = [
                str(godot_bin),
                "--headless",
                "--path", str(self.cartridge_path),
                "-s", str(HARNESS_SCRIPT_PATH),
                "--max-ticks", str(limit),
            ] + extra_args

        logger.info("Executing Godot playtest harness: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.cartridge_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        def _read_stderr():
            if proc.stderr:
                for line in proc.stderr:
                    stderr_lines.append(line)

        err_thread = threading.Thread(target=_read_stderr, daemon=True)
        err_thread.start()

        if proc.stdout:
            for line in proc.stdout:
                stdout_lines.append(line)
                stripped = line.strip()

                # Telemetry intercept
                if stripped.startswith("[TELEMETRY_EVENT] ") or stripped.startswith("[TELEMETRY] "):
                    prefix_len = len("[TELEMETRY_EVENT] ") if stripped.startswith("[TELEMETRY_EVENT] ") else len("[TELEMETRY] ")
                    raw_json = stripped[prefix_len:]
                    try:
                        ev = json.loads(raw_json)
                        pname = ev.get("probe_name") or ev.get("name", "custom_probe")
                        pcat = ev.get("category", "general")
                        pdata = ev.get("payload") or ev.get("data", {})
                        if self.telemetry_recorder:
                            self.telemetry_recorder.emit_custom_probe(
                                probe_name=pname,
                                category=pcat,
                                data=pdata,
                            )
                    except Exception as err:
                        logger.debug("Failed parsing telemetry event: %s", err)

                elif stripped.startswith("[STATE_SNAPSHOT] "):
                    raw_json = stripped[len("[STATE_SNAPSHOT] "):]
                    try:
                        snap = json.loads(raw_json)
                        self.final_snapshot = snap
                        pos = snap.get("player_pos", [0.0, 0.0])
                        vel = snap.get("player_vel", [0.0, 0.0])
                        self.player_pos = (float(pos[0]), float(pos[1]))
                        self.player_vel = (float(vel[0]), float(vel[1]))
                        self.player_health = float(snap.get("player_health", 100.0))
                    except Exception as err:
                        logger.debug("Failed parsing state snapshot: %s", err)

                elif stripped.startswith("[QA_TICK_SNAPSHOT] "):
                    raw_json = stripped[len("[QA_TICK_SNAPSHOT] "):]
                    try:
                        tsnap = json.loads(raw_json)
                        t_num = int(tsnap.get("tick", 0))
                        if t_num > 0:
                            self.tick_snapshots[t_num] = tsnap
                    except Exception as err:
                        logger.debug("Failed parsing QA tick snapshot: %s", err)

                elif stripped.startswith("[QA_SETUP_CALL] "):
                    raw_json = stripped[len("[QA_SETUP_CALL] "):]
                    try:
                        self.qa_setup_calls.append(json.loads(raw_json))
                    except Exception as err:
                        logger.debug("Failed parsing QA setup call: %s", err)

                elif stripped.startswith("[SCENE_AUDIT] "):
                    raw_json = stripped[len("[SCENE_AUDIT] "):]
                    try:
                        self.scene_audit = json.loads(raw_json)
                    except Exception as err:
                        logger.debug("Failed parsing scene audit: %s", err)

                elif stripped.startswith("[PERF_METRICS] "):
                    raw_json = stripped[len("[PERF_METRICS] "):]
                    try:
                        metrics = json.loads(raw_json)
                        self.avg_frame_time_ms = float(metrics.get("avg_frame_time_ms", 0.0))
                        self.max_frame_time_ms = float(metrics.get("max_frame_time_ms", 0.0))
                        self.frame_budget_passed = bool(metrics.get("frame_budget_passed", True))
                        self.ticks_executed = int(metrics.get("ticks_executed", 0))
                    except Exception as err:
                        logger.debug("Failed parsing perf metrics: %s", err)

                elif stripped.startswith("[PROGRESSION] "):
                    raw_json = stripped[len("[PROGRESSION] "):]
                    try:
                        prog = json.loads(raw_json)
                        self.pos_delta = float(prog.get("pos_delta", 0.0))
                        self.progression_verified = bool(prog.get("progression_verified", True))
                    except Exception as err:
                        logger.debug("Failed parsing progression: %s", err)

        try:
            returncode = proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = proc.wait()

        err_thread.join(timeout=1.0)
        stderr_full = "".join(stderr_lines)
        self.stderr = stderr_full
        self.runtime_diagnostics = parse_godot_stderr(self.cartridge_path, stderr_full)

        self.step_count = self.ticks_executed or limit
        self.elapsed_time = self.step_count * (1.0 / 60.0)
        self.is_completed = True
        self.outcome = "COMPLETED" if returncode == 0 else "CRASHED"
        self.viewport_captured = bool(
            use_viewport
            and (self.cartridge_path / "assets" / "playtest_captures" / "playtest_t0.png").is_file()
        )

        if returncode != 0:
            raise RuntimeError(f"Godot headless simulation failed with code {returncode}: {stderr_full}")

        return self.get_state()

    def step(self, sim_input: Optional[SimulationInput] = None) -> SimulationState:
        """Step simulation discrete frames."""
        self.step_count += 1
        self.elapsed_time += (1.0 / 60.0)
        return self.get_state()

    def get_state(self) -> SimulationState:
        """Constructs SimulationState reflecting Godot playtest data."""
        return SimulationState(
            deliberation="Godot headless simulation state snapshot.",
            step_count=self.step_count,
            elapsed_time=self.elapsed_time,
            player_pos=self.player_pos,
            player_vel=self.player_vel,
            player_health=self.player_health,
            player_stamina=100.0,
            is_dashing=False,
            enemies=[],
            active_projectiles=0,
            hazards_count=0,
            portal_unlocked=False,
            portal_pos=(0.0, 0.0),
            is_completed=self.is_completed,
            outcome=self.outcome,
        )

    def run(self, max_steps: Optional[int] = None) -> GodotSimResult:
        """Executes playtest run and returns a structured GodotSimResult."""
        try:
            state = self.run_until_complete(max_steps=max_steps)
            return GodotSimResult(
                success=True,
                error_message=None,
                stderr=getattr(self, "stderr", ""),
                state=state,
                avg_frame_time_ms=self.avg_frame_time_ms,
                frame_budget_passed=self.frame_budget_passed,
                ticks_executed=self.ticks_executed,
            )
        except Exception as exc:
            return GodotSimResult(
                success=False,
                error_message=str(exc),
                stderr=str(exc),
                state=self.get_state(),
                avg_frame_time_ms=self.avg_frame_time_ms,
                frame_budget_passed=False,
                ticks_executed=self.ticks_executed,
            )

    def close(self) -> None:
        if self.telemetry_recorder and hasattr(self.telemetry_recorder, "close"):
            try:
                self.telemetry_recorder.close(timeout=1.0)
            except TypeError:
                self.telemetry_recorder.close()
