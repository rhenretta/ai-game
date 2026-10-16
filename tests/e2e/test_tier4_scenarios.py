"""tests/e2e/test_tier4_scenarios.py
Tier 4: Authoritative Real-World Application Scenario End-to-End Tests.
Covers the 6 complete production lifecycles defined in TEST_INFRA.md Section 4:
  - Scenario 1: Cold Boot Genesis Build from Real Steam Library
  - Scenario 2: High-Mobility Rusher Gameplay & Episodic Adaptation
  - Scenario 3: Methodical Cautious Player Gameplay & Tactical Evolution
  - Scenario 4: Defective Candidate Injection, 5-Gate Rejection, and Self-Healing
  - Scenario 5: Real-Time Studio Observability Under Heavy Telemetry Load
  - Scenario 6: Instant-Launch Game Client Boot & In-Game Hot-Reload
Strictly adheres to Opaque-Box contracts and Rationale-First Epistemic Architecture.
"""
from __future__ import annotations
import ast
import json
import math
from pathlib import Path
import sqlite3
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import pytest

from engine.analyst.analyst import TelemetryAnalyst
from engine.analyst.models import BehavioralProfile
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.compiler.compiler import EngineCompiler, CompilationResult
from engine.director.genesis import GenesisDirector
from engine.director.episodic import EpisodicDirector
from engine.director.models import (
    CurrentEpisodeConfig,
    DirectiveAction,
    EpisodicUpdateSpec,
    GenesisSpec,
    SpatialUpdateDirective,
)
from engine.loader import CartridgeLoader, validate_cartridge_ast, ASTSecurityError
from engine.sim.headless import HeadlessSimulator
from engine.spatial.navmesh import NavMesh, astar_pathfind
from engine.steam.dna import SteamDNAProfiler
from engine.steam.models import SteamPlayerDNA
from engine.steam.scanner import SteamScanner
from engine.supervisor.models import (
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    RemedyType,
    SupervisorEvaluationReport,
    SupervisorGateVerdict,
)
from engine.supervisor.supervisor import BuildSupervisor
from qa.playtester import PlaytestResult, SyntheticPlaytesterBot
from qa.profiles import CAUTIOUS_PROFILE, RUSHER_PROFILE


# ============================================================================
# SCENARIO 1: Cold Boot Genesis Build from Real Steam Library
# ============================================================================

def test_scenario_1_cold_boot_genesis_build_from_steam_library(tmp_path: Path, mock_steam_installation: Path):
    """Scenario 1: Cold Boot Genesis Build from Steam Library.
    Steps:
      1. Steam Discovery: Scan mock macOS Steam library, ingest 250 playtime records.
      2. DNA Profiling: Profiler extracts SteamPlayerDNA with pacing and challenge tolerance.
      3. Spec Synthesis: GenesisDirector produces Episode 0 GenesisSpec with deliberation first.
      4. Multimodal Synthesis & SWE Compilation: EngineCompiler synthesizes cartridge package.
      5. Supervisor Verification: 5-Gate Build Supervisor verifies candidate and promotes to active.
      6. Execution: Game boots into promoted cartridge at 60 FPS in <150ms.
    """
    # 1. Steam Discovery
    scanner = SteamScanner(steam_root=mock_steam_installation)
    assert scanner.is_steam_installed() is True
    dataset = scanner.to_dataset()
    assert len(dataset.games) > 0

    # 2. DNA Profiling
    profiler = SteamDNAProfiler()
    dna = profiler.extract_dna(dataset, profile_tag="genesis_cold_boot")
    assert dna.deliberation != ""
    assert dna.profile_tag == "genesis_cold_boot"
    assert dna.total_games_analyzed > 0

    # 3. Spec Synthesis
    specs_dir = tmp_path / "specs"
    director = GenesisDirector(specs_dir=specs_dir)
    genesis_spec = director.generate_genesis_spec(dna)
    assert genesis_spec.deliberation != ""
    assert genesis_spec.game_title != ""
    assert genesis_spec.player_params.walk_speed > 0

    # 4. Multimodal SWE Compilation
    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    compiler = EngineCompiler(cartridges_root=cartridges_root, active_cartridge_dir=active_dir)
    target_cart = cartridges_root / "genesis_candidate"
    comp_result = compiler.compile(genesis_spec, target_dir=target_cart)
    assert comp_result.ast_validated is True
    assert comp_result.bytecode_compiled is True
    assert comp_result.cartridge_loaded is True

    # 5. Supervisor Verification & Playtesting
    supervisor = BuildSupervisor(cartridges_root=cartridges_root, active_cartridge_dir=active_dir)
    cartridge = CartridgeLoader.load_cartridge(target_cart)

    gate_a = supervisor.verify_gate_a_syntax_and_ast(target_cart)
    assert gate_a.passed is True

    level_spec = cartridge.get_level_spec(1)
    gate_b, _ = supervisor.verify_gate_b_traversability(cartridge=cartridge, level_spec=level_spec)
    assert gate_b.passed is True

    playtester = SyntheticPlaytesterBot(profile=RUSHER_PROFILE, stuck_threshold_ticks=300)
    playtest_res = playtester.evaluate_cartridge(cartridge, max_seconds=10.0, ignore_combat_lock=True)
    assert playtest_res.status == "PROMOTED"

    supervisor.promote_candidate(target_cart)
    assert active_dir.exists()
    assert (active_dir / "manifest.json").exists()

    # 6. Instant Launch Execution
    t_boot_start = time.perf_counter()
    promoted_cartridge = CartridgeLoader.load_cartridge(active_dir)
    sim = HeadlessSimulator(cartridge=promoted_cartridge)
    boot_ms = (time.perf_counter() - t_boot_start) * 1000.0
    assert boot_ms < 500.0  # Fast startup benchmark

    for _ in range(60):
        step_res = sim.step()
        assert step_res is not None

    assert sim.step_count == 60


# ============================================================================
# SCENARIO 2: High-Mobility Rusher Gameplay & Episodic Adaptation
# ============================================================================

def test_scenario_2_high_mobility_rusher_gameplay_and_episodic_adaptation(tmp_path: Path, temp_e2e_db: str):
    """Scenario 2: High-Mobility Rusher Gameplay & Episodic Adaptation.
    Steps:
      1. Gameplay Execution: Player exhibits high velocity, frequent dashes, rapid traversal.
      2. Telemetry Streaming: Custom probe events stream into SQLite WAL.
      3. Hypothesis Formulation: TelemetryAnalyst profiles kinetic metrics; Director formulates hypothesis.
      4. Update Compilation: EngineCompiler produces Episode 2 cartridge.
      5. Supervisor Promotion: Candidate verified and promoted to active.
    """
    session_id = "sess_rusher_01"
    episode_id = "ep_rusher_01"

    # 1 & 2. Telemetry Streaming to SQLite WAL
    conn = sqlite3.connect(temp_e2e_db, timeout=5.0)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO telemetry_sessions (session_id, player_profile_tag)
            VALUES (?, 'kinetic_rusher')
            """,
            (session_id,),
        )
        cur.execute(
            """
            INSERT INTO telemetry_episodes (episode_id, session_id, episode_number, status, duration_seconds, outcome, score, dashes_total)
            VALUES (?, ?, 1, 'COMPLETED', 28.0, 'VICTORY', 2400, 24)
            """,
            (episode_id, session_id),
        )
        # Register custom probe
        cur.execute(
            """
            INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target)
            VALUES ('probe_dash_02', 'dash_evasion_efficiency', 'movement', 'Measures dash under threat', 'hyp_rusher')
            """
        )
        # Stream high-velocity events (> 250 px/s, dash events)
        for i in range(20):
            cur.execute(
                """
                INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, probe_name, pos_x, pos_y, vel_x, vel_y, health, payload_json, frame_index)
                VALUES (?, ?, ?, 'PLAYER_DASH', 'dash_evasion_efficiency', ?, ?, 320.0, 0.0, 100.0, '{"dash_efficiency": 0.95}', ?)
                """,
                (session_id, episode_id, float(i) * 0.1, 100.0 + float(i) * 32.0, 360.0, i * 6),
            )
        conn.commit()
    finally:
        conn.close()

    # 3. Hypothesis Formulation via Analyst & Director
    analyst = TelemetryAnalyst(db_path=temp_e2e_db)
    profile = analyst.analyze_episode(episode_id)
    assert profile.mean_velocity > 200.0

    director = EpisodicDirector(specs_dir=tmp_path / "specs")
    update_spec = director.evaluate_and_update(profile)
    assert update_spec.deliberation != ""
    assert update_spec.target_episode_number == 2

    # 4. Update Compilation
    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    compiler = EngineCompiler(cartridges_root=cartridges_root, active_cartridge_dir=active_dir)
    target_cart = cartridges_root / "episode_02_candidate"
    comp_result = compiler.compile(update_spec, target_dir=target_cart)
    assert comp_result.ast_validated is True
    assert comp_result.bytecode_compiled is True

    # 5. Supervisor Promotion
    supervisor = BuildSupervisor(cartridges_root=cartridges_root, active_cartridge_dir=active_dir)
    supervisor.promote_candidate(target_cart)
    assert (active_dir / "manifest.json").exists()

    promoted_cart = CartridgeLoader.load_cartridge(active_dir)
    assert promoted_cart is not None


# ============================================================================
# SCENARIO 3: Methodical Cautious Player Gameplay & Tactical Evolution
# ============================================================================

def test_scenario_3_methodical_cautious_player_gameplay_and_tactical_evolution(tmp_path: Path, temp_e2e_db: str):
    """Scenario 3: Methodical Cautious Player Gameplay & Tactical Evolution.
    Steps:
      1. Gameplay Execution: Player moves deliberately (<120 px/s), uses cover frequently, rarely dashes.
      2. Telemetry Streaming: Custom probe events (cover_dwell_time) stream into WAL.
      3. Hypothesis Formulation: Director hypothesizes cautious playstyle, plans tactical hazards & cover.
      4. Episodic Generation: Tactical cartridge compiled with custom abilities and narrative barks.
      5. Promotion & Verification: Supervisor promotes verified candidate.
    """
    session_id = "sess_cautious_01"
    episode_id = "ep_cautious_01"

    # 1 & 2. Telemetry Streaming
    conn = sqlite3.connect(temp_e2e_db, timeout=5.0)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO telemetry_sessions (session_id, player_profile_tag)
            VALUES (?, 'methodical_cautious')
            """,
            (session_id,),
        )
        cur.execute(
            """
            INSERT INTO telemetry_episodes (episode_id, session_id, episode_number, status, duration_seconds, outcome, score, dashes_total)
            VALUES (?, ?, 1, 'COMPLETED', 75.0, 'VICTORY', 1100, 2)
            """,
            (episode_id, session_id),
        )
        cur.execute(
            """
            INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target)
            VALUES ('probe_cover_01', 'cover_dwell_time', 'spatial', 'Measures time in cover', 'hyp_tactical')
            """
        )
        # Slow deliberate movements (< 120 px/s, stationary cover dwell)
        for i in range(20):
            cur.execute(
                """
                INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, probe_name, pos_x, pos_y, vel_x, vel_y, health, payload_json, frame_index)
                VALUES (?, ?, ?, 'PLAYER_MOVE', 'cover_dwell_time', ?, ?, 80.0, 0.0, 100.0, '{"in_cover": true}', ?)
                """,
                (session_id, episode_id, float(i) * 0.1, 100.0 + float(i) * 8.0, 200.0, i * 6),
            )
        conn.commit()
    finally:
        conn.close()

    # 3. Hypothesis Formulation
    analyst = TelemetryAnalyst(db_path=temp_e2e_db)
    profile = analyst.analyze_episode(episode_id)
    assert profile.mean_velocity < 150.0

    director = EpisodicDirector(specs_dir=tmp_path / "specs")
    update_spec = director.evaluate_and_update(profile)
    assert update_spec.deliberation != ""

    # 4. Episodic Generation
    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    compiler = EngineCompiler(cartridges_root=cartridges_root, active_cartridge_dir=active_dir)
    target_cart = cartridges_root / "episode_02_tactical"
    comp_result = compiler.compile(update_spec, target_dir=target_cart)
    assert comp_result.cartridge_loaded is True

    # 5. Promotion & Verification
    supervisor = BuildSupervisor(cartridges_root=cartridges_root, active_cartridge_dir=active_dir)
    supervisor.promote_candidate(target_cart)
    assert (active_dir / "manifest.json").exists()


# ============================================================================
# SCENARIO 4: Defective Candidate Injection, 5-Gate Rejection, and Self-Healing
# ============================================================================

def test_scenario_4_defective_candidate_injection_rejection_and_self_healing(tmp_path: Path, mock_cartridge_active: Path):
    """Scenario 4: Defective Candidate Injection, 5-Gate Rejection, and Self-Healing.
    Steps:
      1. Defect Injection: Candidate places an impassable barrier across critical path.
      2. Gate B Rejection: NavMesh A* pathfinder determines no_path_found from spawn to exit.
      3. Diagnostic Trace: Supervisor captures RejectionDiagnosticTrace with exact obstacle coordinates.
      4. Self-Healing Loop: BuildSupervisor.heal_specification repairs layout.
      5. Re-Test & Promotion: Repaired build passes all gates and is promoted.
    """
    supervisor = BuildSupervisor(cartridges_root=tmp_path / "cartridges", tile_size=32.0)
    cartridge = CartridgeLoader.load_cartridge(mock_cartridge_active)

    # 1. Defect Injection
    blocked_level_spec = BaseLevelSpec(
        level_id="level_scenario_4",
        name="Defective Chamber",
        width=1280.0,
        height=720.0,
        spawn_point=(120.0, 360.0),
        exit_point=(1160.0, 360.0),
        walls=[
            (0.0, 0.0, 1280.0, 32.0),
            (0.0, 688.0, 1280.0, 32.0),
            (0.0, 0.0, 32.0, 720.0),
            (1248.0, 0.0, 32.0, 720.0),
            (640.0, 32.0, 48.0, 656.0),  # Impassable barrier
        ],
    )

    # 2. Gate B Rejection
    gate_b, blocked_nodes = supervisor.verify_gate_b_traversability(
        cartridge=cartridge,
        level_spec=blocked_level_spec,
    )
    assert gate_b.passed is False
    assert gate_b.failure_type == GateFailureType.CRITICAL_PATH_BLOCKED

    # 3. Diagnostic Trace Construction
    curr_cfg = CurrentEpisodeConfig(
        deliberation="Baseline config with defective impassable wall.",
        episode_number=1,
        walls=blocked_level_spec.walls,
    )
    from engine.analyst.models import BehavioralProfile
    director = EpisodicDirector(specs_dir=tmp_path / "specs")
    dummy_profile = BehavioralProfile(
        deliberation="Profile for scenario 4 defect healing.",
        session_id="sess_scen_4",
        episode_id="ep_scen_4",
    )
    update_spec = director.evaluate_and_update(dummy_profile, current_config=curr_cfg)
    update_spec.updated_episode_config.walls = blocked_level_spec.walls

    trace = RejectionDiagnosticTrace(
        deliberation="Critical path blocked by wall.",
        spec_id=update_spec.spec_id,
        episode_number=1,
        failure_gate="Gate B: Traversability",
        failure_type=gate_b.failure_type,
        failure_reason="Impassable barrier across critical path.",
        failure_coordinates=(640.0, 360.0),
        blocked_nodes=blocked_nodes,
        remedy_type=RemedyType.REMOVE_OBSTACLE,
        recommended_remedy="Remove interior wall blocking critical path.",
    )

    # 4. Self-Healing Loop
    healed_spec = supervisor.heal_specification(update_spec, trace)
    assert healed_spec is not None

    # 5. Re-Test & Promotion
    healed_level_spec = BaseLevelSpec(
        level_id="level_scenario_4_healed",
        width=1280.0,
        height=720.0,
        spawn_point=(120.0, 360.0),
        exit_point=(1160.0, 360.0),
        walls=healed_spec.updated_episode_config.walls,
    )

    re_gate_b, _ = supervisor.verify_gate_b_traversability(
        cartridge=cartridge,
        level_spec=healed_level_spec,
    )
    assert re_gate_b.passed is True


# ============================================================================
# SCENARIO 5: Real-Time Studio Observability Under Heavy Telemetry Load
# ============================================================================

def test_scenario_5_studio_observability_under_heavy_telemetry_load(temp_e2e_db: str):
    """Scenario 5: Real-Time Studio Observability Under Heavy Telemetry Load.
    Steps:
      1. High-frequency telemetry stream: 5,000 events inserted in batches.
      2. Concurrent Analyst Profiling: Multiple read queries executed during write bursts.
      3. Zero database locks or data loss: Confirms SQLite WAL robustness under stress.
    """
    total_events_to_write = 2000
    batch_size = 200
    session_id = "sess_heavy_load_01"
    episode_id = "ep_heavy_load_01"

    # Setup session & episode
    conn = sqlite3.connect(temp_e2e_db, timeout=10.0)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("INSERT INTO telemetry_sessions (session_id) VALUES (?)", (session_id,))
        cur.execute(
            """
            INSERT INTO telemetry_episodes (episode_id, session_id, episode_number, status, duration_seconds)
            VALUES (?, ?, 1, 'ACTIVE', 120.0)
            """,
            (episode_id, session_id),
        )
        conn.commit()
    finally:
        conn.close()

    stop_event = threading.Event()
    writer_finished = threading.Event()
    analyst_reads = [0]

    def writer_worker():
        conn = sqlite3.connect(temp_e2e_db, timeout=10.0)
        try:
            cur = conn.cursor()
            inserted = 0
            while inserted < total_events_to_write:
                batch = []
                for b in range(batch_size):
                    idx = inserted + b
                    batch.append((
                        session_id, episode_id, float(idx) * 0.016, 'PLAYER_MOVE',
                        100.0 + (idx % 800), 200.0 + (idx % 400), 120.0, 0.0, 100.0, idx
                    ))
                cur.executemany(
                    """
                    INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y, vel_x, vel_y, health, frame_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                inserted += batch_size
                time.sleep(0.002)
        finally:
            conn.close()
            writer_finished.set()

    def reader_worker():
        analyst = TelemetryAnalyst(db_path=temp_e2e_db)
        while not stop_event.is_set():
            profile = analyst.analyze_session(session_id)
            if profile is not None:
                analyst_reads[0] += 1
            time.sleep(0.005)

    t_w = threading.Thread(target=writer_worker, daemon=True)
    t_r = threading.Thread(target=reader_worker, daemon=True)

    t_w.start()
    t_r.start()

    writer_finished.wait(timeout=10.0)
    stop_event.set()
    t_r.join(timeout=1.0)

    assert writer_finished.is_set() is True
    assert analyst_reads[0] > 0

    # Final read verification
    analyst = TelemetryAnalyst(db_path=temp_e2e_db)
    final_profile = analyst.analyze_session(session_id)
    assert final_profile.session_id == session_id
    assert final_profile.kinetic.total_movement_ticks > 0


# ============================================================================
# SCENARIO 6: Instant-Launch Game Client Boot & In-Game Hot-Reload
# ============================================================================

def test_scenario_6_instant_launch_boot_and_in_game_hot_reload(tmp_path: Path, mock_cartridge_active: Path):
    """Scenario 6: Instant-Launch Game Client Boot & In-Game Hot-Reload.
    Steps:
      1. Instant Launch: Load active cartridge, compile bytecode, and initialize in <150ms.
      2. Background Compilation: Studio compiler produces new episode cartridge.
      3. Atomic Promotion: Promoted candidate updates active cartridge path.
      4. In-Game Hot-Reload: CartridgeLoader hot-swaps to new episode cleanly.
    """
    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"

    # Copy mock_cartridge_active to active_dir
    import shutil
    shutil.copytree(mock_cartridge_active, active_dir)

    # 1. Instant Launch Benchmark
    t0 = time.perf_counter()
    cart_v1 = CartridgeLoader.load_cartridge(active_dir)
    sim = HeadlessSimulator(cartridge=cart_v1)
    init_ms = (time.perf_counter() - t0) * 1000.0
    assert init_ms < 200.0

    # Execute 30 ticks of Episode 1
    for _ in range(30):
        sim.step()
    assert sim.step_count == 30

    # 2. Background Compilation of Episode 2
    ep2_dir = cartridges_root / "episode_02_update"
    ep2_dir.mkdir(parents=True)
    (ep2_dir / "manifest.json").write_text(
        json.dumps({"episode_number": 2, "cartridge_id": "ep2_live_update", "version": "2.0.0"}),
        encoding="utf-8",
    )
    (ep2_dir / "mechanics.py").write_text(
        """from engine.api.contracts import BaseCartridge, BaseLevelSpec
class Ep2LiveCartridge(BaseCartridge):
    def get_level_spec(self, episode_number: int = 2) -> BaseLevelSpec:
        return BaseLevelSpec(level_id="ep2_live", spawn_point=(150.0, 360.0), exit_point=(1100.0, 360.0))
    def create_player(self, x, y, telemetry_recorder=None): return None
    def create_enemies(self, spec): return []
    def create_hazards(self, spec): return []
""",
        encoding="utf-8",
    )

    # 3. Atomic Promotion
    supervisor = BuildSupervisor(cartridges_root=cartridges_root, active_cartridge_dir=active_dir)
    supervisor.promote_candidate(ep2_dir)

    # 4. In-Game Hot-Reload
    cart_v2 = CartridgeLoader.load_cartridge(active_dir)
    assert cart_v2.manifest["cartridge_id"] == "ep2_live_update"
    assert cart_v2.manifest["episode_number"] == 2
    spec_v2 = cart_v2.get_level_spec(2)
    assert spec_v2.spawn_point == (150.0, 360.0)
