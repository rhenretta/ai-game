"""tests/test_challenger_m7_1_headless_signal.py
Empirical Challenger Test Suite for Milestone 7:
1. Headless simulation and frame budget (<16.6ms) for active cartridge and latest checkpoint.
2. Custom telemetry emission on stdout and schema conformance with telemetry_manifest.json.
3. Process lifecycle and signal safety (SIGINT and SIGTERM) with clean child Godot reaping.
4. Stress-test SQLite WAL telemetry bridging in run_game.py.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _get_child_pids(parent_pid: int) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-P", str(parent_pid)], text=True)
        return [int(line.strip()) for line in out.strip().splitlines() if line.strip()]
    except subprocess.CalledProcessError:
        return []


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_headless_active_frame_budget():
    """Verify run_game.py --headless --ticks 120 on cartridges/active achieves <16.6ms avg frame time."""
    cmd = [str(VENV_PYTHON), "run_game.py", "--headless", "--ticks", "120"]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=15)
    
    assert res.returncode == 0, f"Headless run failed with code {res.returncode}. Stderr: {res.stderr}"
    assert "Launching Godot 4.3 runtime" in res.stdout
    assert "Game and telemetry shutdown completed cleanly." in res.stdout
    
    # Parse perf metrics
    perf_metrics = None
    for line in res.stdout.splitlines():
        if "[PERF_METRICS]" in line:
            raw = line.split("[PERF_METRICS]", 1)[1].strip()
            perf_metrics = json.loads(raw)
            break
            
    assert perf_metrics is not None, "Did not find [PERF_METRICS] in stdout"
    assert perf_metrics.get("frame_budget_passed") is True, f"Frame budget failed: {perf_metrics}"
    avg_ms = perf_metrics.get("avg_frame_time_ms", 999.0)
    assert avg_ms < 16.6, f"Average frame time {avg_ms}ms exceeds 16.6ms budget"
    max_ms = perf_metrics.get("max_frame_time_ms", 999.0)
    assert max_ms < 33.3, f"Max frame time {max_ms}ms indicates stall spike"
    assert perf_metrics.get("ticks_executed") == 120


def test_headless_latest_checkpoint_frame_budget():
    """Verify run_game.py --latest-checkpoint --headless --ticks 120 achieves <16.6ms avg frame time."""
    cmd = [str(VENV_PYTHON), "run_game.py", "--latest-checkpoint", "--headless", "--ticks", "120"]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=15)
    
    assert res.returncode == 0, f"Latest checkpoint run failed with code {res.returncode}. Stderr: {res.stderr}"
    assert "Launching Godot 4.3 runtime" in res.stdout
    assert "Game and telemetry shutdown completed cleanly." in res.stdout
    
    perf_metrics = None
    for line in res.stdout.splitlines():
        if "[PERF_METRICS]" in line:
            raw = line.split("[PERF_METRICS]", 1)[1].strip()
            perf_metrics = json.loads(raw)
            break
            
    assert perf_metrics is not None, "Did not find [PERF_METRICS] in stdout"
    assert perf_metrics.get("frame_budget_passed") is True, f"Frame budget failed: {perf_metrics}"
    avg_ms = perf_metrics.get("avg_frame_time_ms", 999.0)
    assert avg_ms < 16.6, f"Average frame time {avg_ms}ms exceeds 16.6ms budget"


def test_custom_telemetry_emission_stdout():
    """Verify stdout contains valid [TELEMETRY_EVENT] JSON conforming to telemetry_manifest.json."""
    manifest_path = PROJECT_ROOT / "cartridges" / "active" / "telemetry_manifest.json"
    assert manifest_path.is_file(), "telemetry_manifest.json missing"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_probes = {p["probe_name"]: p for p in manifest.get("probes", [])}
    
    cmd = [str(VENV_PYTHON), "run_game.py", "--headless", "--ticks", "120"]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=15)
    
    emitted_events = []
    for line in res.stdout.splitlines():
        if "[TELEMETRY_EVENT]" in line:
            raw = line.split("[TELEMETRY_EVENT]", 1)[1].strip()
            ev = json.loads(raw)
            emitted_events.append(ev)
            
    assert len(emitted_events) > 0, "No [TELEMETRY_EVENT] lines emitted on stdout"
    
    for ev in emitted_events:
        probe_name = ev.get("probe_name")
        assert probe_name in declared_probes, f"Emitted probe '{probe_name}' not in declared manifest: {list(declared_probes.keys())}"
        declared = declared_probes[probe_name]
        assert ev.get("category") == declared.get("category"), f"Category mismatch for {probe_name}: {ev.get('category')} != {declared.get('category')}"


def test_signal_safety_clean_child_reaping_sigint():
    """Launch headless game in background, send SIGINT, verify clean termination and zero orphaned Godot processes."""
    cmd = [str(VENV_PYTHON), "run_game.py", "--latest-checkpoint", "--headless", "--ticks", "100000"]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    godot_pid = None
    t0 = time.time()
    while time.time() - t0 < 8.0:
        children = _get_child_pids(proc.pid)
        if children:
            godot_pid = children[0]
            break
        time.sleep(0.05)
        
    assert godot_pid is not None, "Failed to spawn or discover Godot child process within 8s"
    assert _is_pid_alive(godot_pid), f"Godot child PID {godot_pid} not alive before signal"
    
    # Send SIGINT
    os.kill(proc.pid, signal.SIGINT)
    stdout, stderr = proc.communicate(timeout=6.0)
    
    assert proc.returncode == 0, f"Python process exited with non-zero code {proc.returncode}"
    assert "Signal 2 received" in stdout
    assert "Game and telemetry shutdown completed cleanly." in stdout
    
    time.sleep(0.3)
    assert not _is_pid_alive(godot_pid), f"CRITICAL LEAK: Godot child process {godot_pid} orphaned after SIGINT!"


def test_signal_safety_clean_child_reaping_sigterm():
    """Launch headless game in background, send SIGTERM, verify clean termination and zero orphaned Godot processes."""
    cmd = [str(VENV_PYTHON), "run_game.py", "--latest-checkpoint", "--headless", "--ticks", "100000"]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    godot_pid = None
    t0 = time.time()
    while time.time() - t0 < 8.0:
        children = _get_child_pids(proc.pid)
        if children:
            godot_pid = children[0]
            break
        time.sleep(0.05)
        
    assert godot_pid is not None, "Failed to spawn or discover Godot child process within 8s"
    assert _is_pid_alive(godot_pid), f"Godot child PID {godot_pid} not alive before signal"
    
    # Send SIGTERM
    os.kill(proc.pid, signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=6.0)
    
    assert proc.returncode == 0, f"Python process exited with non-zero code {proc.returncode}"
    assert "Signal 15 received" in stdout
    assert "Game and telemetry shutdown completed cleanly." in stdout
    
    time.sleep(0.3)
    assert not _is_pid_alive(godot_pid), f"CRITICAL LEAK: Godot child process {godot_pid} orphaned after SIGTERM!"


def test_telemetry_persisted_to_sqlite_wal():
    """Adversarial stress test: verify custom telemetry events from Godot are persisted to SQLite WAL custom_telemetry_events table."""
    import sqlite3
    db_path = PROJECT_ROOT / "data" / "telemetry.db"
    
    conn = sqlite3.connect(str(db_path))
    c0 = conn.cursor().execute("SELECT count(*) FROM custom_telemetry_events").fetchone()[0]
    conn.close()
    
    cmd = [str(VENV_PYTHON), "run_game.py", "--headless", "--ticks", "120"]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=15)
    assert res.returncode == 0
    
    conn = sqlite3.connect(str(db_path))
    c1 = conn.cursor().execute("SELECT count(*) FROM custom_telemetry_events").fetchone()[0]
    conn.close()
    
    # This exposes the bug: engine.quit() terminated telemetry before run_godot_cartridge,
    # causing 0 custom telemetry events to be saved.
    assert c1 > c0, f"Custom telemetry events were dropped! Before={c0}, After={c1}"
