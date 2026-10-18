"""tests/test_challenger_m7_1_headless_signal.py
Empirical Challenger Test Suite for Milestone 7:
1. Headless simulation and frame budget (<16.6ms) for active cartridge and latest checkpoint.
2. Custom telemetry emission on stdout and schema conformance with telemetry_manifest.json.
3. Process lifecycle and signal safety (SIGINT and SIGTERM) with clean child Godot reaping.
4. Stress-test SQLite WAL telemetry bridging in run_game.py using an isolated temporary SQLite DB.
"""
import json
import os
from pathlib import Path
import signal
import sqlite3
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


def _create_hermetic_godot_cartridge(cartridge_dir: Path) -> Path:
    """Creates a minimal, self-contained Godot 4.3 cartridge with custom telemetry emission in tmp_path."""
    cartridge_dir.mkdir(parents=True, exist_ok=True)
    (cartridge_dir / "project.godot").write_text(
        '; Engine configuration file.\n'
        'config_version=5\n\n'
        '[application]\n'
        'config/name="HermeticM7Test"\n'
        'run/main_scene="res://main.tscn"\n'
        'config/features=PackedStringArray("4.3")\n',
        encoding="utf-8",
    )
    (cartridge_dir / "main.tscn").write_text(
        '[gd_scene load_steps=2 format=3]\n'
        '[ext_resource type="Script" path="res://main.gd" id="1"]\n'
        '[node name="Main" type="Node2D"]\n'
        'script = ExtResource("1")\n',
        encoding="utf-8",
    )
    (cartridge_dir / "main.gd").write_text(
        'extends Node2D\n'
        'var tick_count: int = 0\n\n'
        'func _physics_process(_delta: float) -> void:\n'
        '\ttick_count += 1\n'
        '\tif tick_count % 10 == 0:\n'
        '\t\tvar ev = {\n'
        '\t\t\t"probe_name": "leyline_pressure_sample",\n'
        '\t\t\t"category": "economy",\n'
        '\t\t\t"payload": {"tick": tick_count, "pressure": 42.5}\n'
        '\t\t}\n'
        '\t\tprint("[TELEMETRY_EVENT] " + JSON.stringify(ev))\n\n'
        'func get_state_snapshot() -> Dictionary:\n'
        '\treturn {"tick": tick_count}\n',
        encoding="utf-8",
    )
    manifest = {
        "deliberation": "Hermetic telemetry manifest for M7 signal and WAL persistence verification.",
        "probes": [
            {
                "deliberation": "Tracks leyline pressure samples every 10 ticks.",
                "probe_id": "probe_leyline_pressure_sample",
                "probe_name": "leyline_pressure_sample",
                "category": "economy",
                "description": "Emits periodic pressure telemetry.",
                "trigger_condition": "Every 10 physics ticks",
                "expected_frequency": "6 Hz",
                "target_hypothesis": "H1",
            }
        ],
    }
    (cartridge_dir / "telemetry_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (cartridge_dir / "manifest.json").write_text(
        json.dumps(
            {
                "deliberation": "Hermetic Godot 4 cartridge manifest.",
                "name": "HermeticM7Test",
                "version": "0.1.0",
                "entrypoint": "res://main.tscn",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cartridge_dir


def _hermetic_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    env["STUDIO_DATA_DIR"] = str(data_dir)
    env["STUDIO_DB_PATH"] = str(data_dir / "telemetry.db")
    return env


def test_headless_active_frame_budget(tmp_path: Path):
    """Verify run_game.py --cartridge <dir> --headless --ticks 120 achieves <16.6ms avg frame time."""
    cart_dir = _create_hermetic_godot_cartridge(tmp_path / "active_cartridge")
    env = _hermetic_env(tmp_path)
    cmd = [str(VENV_PYTHON), "run_game.py", "--cartridge", str(cart_dir), "--headless", "--ticks", "120"]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=15)

    assert res.returncode == 0, f"Headless run failed with code {res.returncode}. Stderr: {res.stderr}"
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
    max_ms = perf_metrics.get("max_frame_time_ms", 999.0)
    assert max_ms < 33.3, f"Max frame time {max_ms}ms indicates stall spike"
    assert perf_metrics.get("ticks_executed") == 120


def test_headless_latest_checkpoint_frame_budget(tmp_path: Path):
    """Verify run_game.py --latest-checkpoint --headless --ticks 120 achieves <16.6ms avg frame time."""
    env = _hermetic_env(tmp_path)
    cmd = [str(VENV_PYTHON), "run_game.py", "--latest-checkpoint", "--headless", "--ticks", "120"]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=15)

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


def test_custom_telemetry_emission_stdout(tmp_path: Path):
    """Verify stdout contains valid [TELEMETRY_EVENT] JSON conforming to telemetry_manifest.json."""
    cart_dir = _create_hermetic_godot_cartridge(tmp_path / "telemetry_cartridge")
    env = _hermetic_env(tmp_path)
    manifest_path = cart_dir / "telemetry_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_probes = {p["probe_name"]: p for p in manifest.get("probes", [])}

    cmd = [str(VENV_PYTHON), "run_game.py", "--cartridge", str(cart_dir), "--headless", "--ticks", "120"]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=15)

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


def test_signal_safety_clean_child_reaping_sigint(tmp_path: Path):
    """Launch headless game in background, send SIGINT, verify clean termination and zero orphaned Godot processes."""
    cart_dir = _create_hermetic_godot_cartridge(tmp_path / "sigint_cartridge")
    env = _hermetic_env(tmp_path)
    cmd = [str(VENV_PYTHON), "run_game.py", "--cartridge", str(cart_dir), "--headless", "--ticks", "100000"]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

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

    os.kill(proc.pid, signal.SIGINT)
    stdout, _stderr = proc.communicate(timeout=6.0)

    assert proc.returncode == 0, f"Python process exited with non-zero code {proc.returncode}"
    assert "Signal 2 received" in stdout
    assert "Game and telemetry shutdown completed cleanly." in stdout

    time.sleep(0.3)
    assert not _is_pid_alive(godot_pid), f"CRITICAL LEAK: Godot child process {godot_pid} orphaned after SIGINT!"


def test_signal_safety_clean_child_reaping_sigterm(tmp_path: Path):
    """Launch headless game in background, send SIGTERM, verify clean termination and zero orphaned Godot processes."""
    cart_dir = _create_hermetic_godot_cartridge(tmp_path / "sigterm_cartridge")
    env = _hermetic_env(tmp_path)
    cmd = [str(VENV_PYTHON), "run_game.py", "--cartridge", str(cart_dir), "--headless", "--ticks", "100000"]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

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

    os.kill(proc.pid, signal.SIGTERM)
    stdout, _stderr = proc.communicate(timeout=6.0)

    assert proc.returncode == 0, f"Python process exited with non-zero code {proc.returncode}"
    assert "Signal 15 received" in stdout
    assert "Game and telemetry shutdown completed cleanly." in stdout

    time.sleep(0.3)
    assert not _is_pid_alive(godot_pid), f"CRITICAL LEAK: Godot child process {godot_pid} orphaned after SIGTERM!"


def test_telemetry_persisted_to_sqlite_wal(tmp_path: Path):
    """Adversarial stress test: verify custom telemetry events from Godot are persisted to an isolated SQLite WAL custom_telemetry_events table."""
    cart_dir = _create_hermetic_godot_cartridge(tmp_path / "wal_persist_cartridge")
    env = _hermetic_env(tmp_path)
    db_path = Path(env["STUDIO_DB_PATH"])

    cmd = [str(VENV_PYTHON), "run_game.py", "--cartridge", str(cart_dir), "--headless", "--ticks", "120"]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=15)
    assert res.returncode == 0

    conn = sqlite3.connect(str(db_path))
    c1 = conn.cursor().execute("SELECT count(*) FROM custom_telemetry_events").fetchone()[0]
    conn.close()

    assert c1 > 0, f"Custom telemetry events were dropped! Count={c1}"
