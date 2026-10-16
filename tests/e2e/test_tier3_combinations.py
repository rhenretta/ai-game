"""tests/e2e/test_tier3_combinations.py
Tier 3: Authoritative Cross-Feature Combinations & Pairwise Milestone Contract Tests.
Verifies cross-cutting integration contracts between Milestones M1, M2, M3, M4, M5, and M6.
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
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.assets.palette import STYLE_BIBLE_16_PALETTE, MOOD_PALETTES
from engine.assets.synthesizer import DeterministicSpriteSynthesizer
from engine.compiler.compiler import EngineCompiler, CompilationResult
from engine.core.engine import Engine
from engine.director.genesis import GenesisDirector, PALETTE_THEMES
from engine.director.episodic import EpisodicDirector
from engine.director.models import (
    CurrentEpisodeConfig,
    DirectiveAction,
    EpisodicUpdateSpec,
    GenesisSpec,
    SpatialUpdateDirective,
)
from engine.loader import CartridgeLoader, validate_cartridge_ast, ASTSecurityError
from engine.scheduler.artifact import ArtifactMetadataEnvelope, ArtifactType
from engine.scheduler.scheduler import Priority, PriorityScheduler, TaskState, FrameBudgetMonitor
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


# ============================================================================
# M1 x M2: Engine Telemetry Sessions x Steam Player Profile Tagging
# ============================================================================

def test_m1_m2_steam_dna_profile_tags_engine_session(temp_e2e_db: str, mock_steam_installation: Path):
    """M1 x M2: Verifies Steam scanner DNA extraction flows into SQLite engine session tagging."""
    scanner = SteamScanner(steam_root=mock_steam_installation)
    dataset = scanner.to_dataset()
    assert len(dataset.games) > 0

    profiler = SteamDNAProfiler()
    dna = profiler.extract_dna(dataset, profile_tag="steam_kinetic_user")
    assert dna.profile_tag == "steam_kinetic_user"
    assert dna.deliberation != ""

    # Attach DNA profile tag to engine session record in WAL database
    conn = sqlite3.connect(temp_e2e_db, timeout=5.0)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO telemetry_sessions (session_id, player_profile_tag, platform, resolution, target_fps)
            VALUES (?, ?, 'macOS', '1280x720', 60)
            """,
            ("sess_m1_m2_01", dna.profile_tag),
        )
        conn.commit()

        cur.execute("SELECT session_id, player_profile_tag FROM telemetry_sessions WHERE session_id = ?", ("sess_m1_m2_01",))
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "sess_m1_m2_01"
        assert row[1] == dna.profile_tag
    finally:
        conn.close()


# ============================================================================
# M2 x M3: Steam Player DNA x Cognitive Genesis Director
# ============================================================================

def test_m2_m3_divergent_steam_dna_yields_divergent_genesis_specs(tmp_path: Path):
    """M2 x M3: Contrasting Player DNA (Rusher vs Cautious) produces divergent game designs."""
    rusher_dna = SteamPlayerDNA(
        deliberation="High-mobility action DNA preferring kinetic speed and intense combat.",
        player_id="player_rusher",
        profile_tag="kinetic_rusher",
        total_playtime_hours=450.0,
        total_games_analyzed=120,
        genre_weights={"Action": 0.85, "Shooter": 0.70},
        pacing_preference="kinetic_high_mobility",
        pacing_score=0.82,
        challenge_tolerance=0.88,
        mechanical_biases={"dash": 0.90, "aggression": 0.85},
        preferred_palette_mood="neon_cyberpunk",
        top_genres=["Action", "Shooter"],
    )

    cautious_dna = SteamPlayerDNA(
        deliberation="Methodical tactical DNA preferring calculated positioning and safety.",
        player_id="player_cautious",
        profile_tag="methodical_cautious",
        total_playtime_hours=310.0,
        total_games_analyzed=80,
        genre_weights={"Tactical": 0.80, "Puzzle": 0.75},
        pacing_preference="tactical_methodical",
        pacing_score=-0.70,
        challenge_tolerance=0.35,
        mechanical_biases={"dash": 0.15, "cover": 0.85},
        preferred_palette_mood="grim_dark",
        top_genres=["Tactical", "Puzzle"],
    )

    director = GenesisDirector(specs_dir=tmp_path / "specs")
    spec_rusher = director.generate_genesis_spec(rusher_dna)
    spec_cautious = director.generate_genesis_spec(cautious_dna)

    # 1. Deliberation first
    assert spec_rusher.deliberation != ""
    assert spec_cautious.deliberation != ""

    # 2. Player movement speed divergence (Rusher walk_speed > Cautious walk_speed)
    assert spec_rusher.player_params.walk_speed > spec_cautious.player_params.walk_speed

    # 3. Dash cooldown divergence (Rusher shorter cooldown than Cautious)
    assert spec_rusher.player_params.dash_cooldown < spec_cautious.player_params.dash_cooldown

    # 4. Difficulty baseline divergence
    assert spec_rusher.target_difficulty > spec_cautious.target_difficulty

    # 5. Aesthetic Palette Mood divergence
    assert spec_rusher.palette_mood == "neon_cyberpunk"
    assert spec_cautious.palette_mood == "grim_dark"


# ============================================================================
# M2 x M4: Steam Player DNA x SWE Engine Compiler
# ============================================================================

def test_m2_m4_steam_dna_compiles_to_runnable_cartridge(tmp_path: Path):
    """M2 x M4: Steam DNA transforms via GenesisDirector and EngineCompiler into loadable cartridge."""
    dna = SteamPlayerDNA(
        deliberation="Balanced arcade player profile.",
        player_id="player_arcade",
        profile_tag="arcade_veteran",
        total_playtime_hours=150.0,
        total_games_analyzed=45,
        genre_weights={"Arcade": 0.80},
        pacing_score=0.40,
        challenge_tolerance=0.60,
        preferred_palette_mood="retro_arcade",
        top_genres=["Arcade"],
    )

    director = GenesisDirector(specs_dir=tmp_path / "specs")
    spec = director.generate_genesis_spec(dna)

    cartridges_dir = tmp_path / "cartridges"
    compiler = EngineCompiler(cartridges_root=cartridges_dir)
    target_cart = cartridges_dir / "episode_0"
    result = compiler.compile(spec, target_dir=target_cart)

    # Verification of compilation result
    assert result.deliberation != ""
    assert result.ast_validated is True
    assert result.bytecode_compiled is True
    assert result.cartridge_loaded is True

    # Verification of on-disk artifacts
    assert (target_cart / "manifest.json").exists()
    assert (target_cart / "mechanics.py").exists()
    assert (target_cart / "entities.py").exists()
    assert (target_cart / "level.json").exists()

    # Dynamic loading verification
    cartridge = CartridgeLoader.load_cartridge(target_cart)
    assert isinstance(cartridge, BaseCartridge)
    level_spec = cartridge.get_level_spec(1)
    assert level_spec is not None


# ============================================================================
# M2 x M5: Steam Player DNA x Studio API Observability Contracts
# ============================================================================

def test_m2_m5_steam_dna_metrics_accessible_via_studio_contracts(mock_steam_installation: Path):
    """M2 x M5: Steam DNA models map correctly to Studio REST/WS data contracts."""
    scanner = SteamScanner(steam_root=mock_steam_installation)
    dataset = scanner.to_dataset()
    profiler = SteamDNAProfiler()
    dna = profiler.extract_dna(dataset, profile_tag="studio_observed_user")

    dna_dict = dna.to_dict()
    assert "deliberation" in dna_dict
    assert "pacing_score" in dna_dict
    assert "challenge_tolerance" in dna_dict
    assert "genre_weights" in dna_dict
    assert isinstance(dna_dict["genre_weights"], dict)

    # Ensure serialization roundtrips cleanly into valid JSON
    serialized = json.dumps(dna_dict)
    deserialized = json.loads(serialized)
    assert deserialized["profile_tag"] == "studio_observed_user"


# ============================================================================
# M1 x M3: Custom Telemetry Probes x Telemetry Analyst Behavioral Ingestion
# ============================================================================

def test_m1_m3_custom_probe_wal_emission_and_analyst_behavioral_profile(temp_e2e_db: str):
    """M1 x M3: Custom probe events in SQLite WAL are ingested and profiled by TelemetryAnalyst."""
    conn = sqlite3.connect(temp_e2e_db, timeout=5.0)
    try:
        cur = conn.cursor()
        # Register custom probe
        cur.execute(
            """
            INSERT INTO telemetry_probes (probe_id, probe_name, category, description, hypothesis_target)
            VALUES ('probe_dash_01', 'dash_evasion_efficiency', 'movement', 'Measures dash under threat', 'hyp_kinetic')
            """
        )
        # Register session and episode
        cur.execute(
            """
            INSERT INTO telemetry_sessions (session_id, player_profile_tag)
            VALUES ('sess_m1_m3', 'rusher_tester')
            """
        )
        cur.execute(
            """
            INSERT INTO telemetry_episodes (episode_id, session_id, episode_number, status, duration_seconds, outcome, score, dashes_total)
            VALUES ('ep_m1_m3_01', 'sess_m1_m3', 1, 'COMPLETED', 45.0, 'VICTORY', 1500, 18)
            """
        )
        # Stream 10 telemetry events with movement and custom probe
        for i in range(10):
            cur.execute(
                """
                INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, probe_name, pos_x, pos_y, vel_x, vel_y, health, payload_json, frame_index)
                VALUES ('sess_m1_m3', 'ep_m1_m3_01', ?, 'PLAYER_MOVE', 'dash_evasion_efficiency', ?, ?, 240.0, 0.0, 100.0, '{"dash_active": true}', ?)
                """,
                (float(i) * 0.1, 100.0 + float(i) * 24.0, 360.0, i * 6),
            )
        conn.commit()
    finally:
        conn.close()

    # Analyst queries WAL database directly
    analyst = TelemetryAnalyst(db_path=temp_e2e_db)
    profile = analyst.analyze_episode("ep_m1_m3_01")

    assert profile.deliberation != ""
    assert profile.session_id == "sess_m1_m3"
    assert profile.episode_id == "ep_m1_m3_01"
    assert profile.kinetic.total_movement_ticks > 0
    assert profile.mean_velocity > 0.0


# ============================================================================
# M1 x M3: Spatial Collision Geometry x NavMesh A* Pathfinding
# ============================================================================

def test_m1_m3_spatial_navmesh_astar_with_engine_wall_geometry():
    """M1 x M3: Spatial wall geometry creates NavMesh grid obstacles traversable via A*."""
    grid = [[0 for _ in range(20)] for _ in range(15)]
    # Build a wall with a door gap in column 10: rows 0-6 wall, row 7 door gap, rows 8-14 wall
    for y in range(15):
        if y != 7:
            grid[y][10] = 1  # Solid wall

    navmesh = NavMesh(grid=grid, tile_size=32.0)
    assert navmesh.width == 20
    assert navmesh.height == 15

    start_tile = (2, 7)
    goal_tile = (16, 7)

    path = navmesh.find_path(start_tile, goal_tile)
    assert path is not None
    assert len(path) > 0
    # Confirm path traversed through the door gap tile (10, 7)
    assert (10, 7) in path


# ============================================================================
# M3 x M4: Cognitive SWE Synthesis x Build Supervisor Gate Rejection
# ============================================================================

def test_m3_m4_supervisor_rejects_ast_violation_with_diagnostic_trace(tmp_path: Path):
    """M3 x M4: Candidate cartridge containing prohibited imports is rejected with diagnostic trace."""
    bad_cart = tmp_path / "bad_cartridge"
    bad_cart.mkdir()
    (bad_cart / "manifest.json").write_text(
        json.dumps({"entry_point": "mechanics.py", "version": "1.0.0"}), encoding="utf-8"
    )
    # Prohibited import os
    (bad_cart / "mechanics.py").write_text(
        "import os\nfrom engine.api.contracts import BaseCartridge\nclass BadCart(BaseCartridge):\n    pass\n",
        encoding="utf-8",
    )

    supervisor = BuildSupervisor(cartridges_root=tmp_path / "cartridges")
    gate_a_result = supervisor.verify_gate_a_syntax_and_ast(bad_cart)

    assert gate_a_result.passed is False
    assert gate_a_result.failure_type == GateFailureType.SYNTAX_AST_ERROR
    assert "os" in gate_a_result.details


# ============================================================================
# M3 x M4: Build Supervisor Closed-Loop Self-Healing Spatial Unreachable
# ============================================================================

def test_m3_m4_supervisor_closed_loop_self_healing_spatial_unreachable(tmp_path: Path, mock_cartridge_active: Path):
    """M3 x M4: Defective update spec with impassable wall is repaired via self-healing loop."""
    supervisor = BuildSupervisor(cartridges_root=tmp_path / "cartridges", tile_size=32.0)
    cartridge = CartridgeLoader.load_cartridge(mock_cartridge_active)

    # Create blocked spec where an interior wall seals off the exit
    blocked_level_spec = BaseLevelSpec(
        level_id="level_01",
        name="Chamber Blocked",
        width=1280.0,
        height=720.0,
        spawn_point=(120.0, 360.0),
        exit_point=(1160.0, 360.0),
        walls=[
            (0.0, 0.0, 1280.0, 32.0),
            (0.0, 688.0, 1280.0, 32.0),
            (0.0, 0.0, 32.0, 720.0),
            (1248.0, 0.0, 32.0, 720.0),
            (600.0, 32.0, 64.0, 656.0),  # Blocking wall across whole chamber
        ],
    )

    # Verify Gate B detects failure
    gate_b, blocked_nodes = supervisor.verify_gate_b_traversability(
        cartridge=cartridge,
        level_spec=blocked_level_spec,
    )
    assert gate_b.passed is False
    assert gate_b.failure_type == GateFailureType.CRITICAL_PATH_BLOCKED

    # Construct EpisodicUpdateSpec with blocked config and heal it
    curr_cfg = CurrentEpisodeConfig(
        deliberation="Episode 1 baseline with impassable wall.",
        episode_number=1,
        walls=blocked_level_spec.walls,
    )
    from engine.analyst.models import BehavioralProfile
    director = EpisodicDirector(specs_dir=tmp_path / "specs")
    dummy_profile = BehavioralProfile(
        deliberation="Profile for healing test",
        session_id="sess_heal",
        episode_id="ep_heal",
    )
    update_spec = director.evaluate_and_update(dummy_profile, current_config=curr_cfg)
    update_spec.updated_episode_config.walls = blocked_level_spec.walls
    trace = RejectionDiagnosticTrace(
        deliberation="Critical path blocked by wall.",
        spec_id="spec_test_heal",
        episode_number=1,
        failure_gate="Gate B: Traversability",
        failure_type=gate_b.failure_type,
        failure_reason="Interior wall blocking path from spawn to exit.",
        failure_coordinates=(600.0, 360.0),
        blocked_nodes=blocked_nodes,
        remedy_type=RemedyType.REMOVE_OBSTACLE,
        recommended_remedy="Remove interior wall blocking critical path.",
    )

    healed_spec = supervisor.heal_specification(update_spec, trace)
    assert healed_spec is not None

    healed_level_spec = BaseLevelSpec(
        level_id="level_01",
        width=1280.0,
        height=720.0,
        spawn_point=(120.0, 360.0),
        exit_point=(1160.0, 360.0),
        walls=healed_spec.updated_episode_config.walls,
    )

    # Re-verify Gate B on healed geometry
    re_gate_b, _ = supervisor.verify_gate_b_traversability(
        cartridge=cartridge,
        level_spec=healed_level_spec,
    )
    assert re_gate_b.passed is True


# ============================================================================
# M1 x M4: Cartridge Loader Dynamic Episode Hot-Swap
# ============================================================================

def test_m1_m4_cartridge_loader_dynamic_episode_hotswap(tmp_path: Path):
    """M1 x M4: CartridgeLoader hot-swaps between distinct episodes cleanly without namespace collision."""
    cart_root = tmp_path / "cartridges"
    ep1_dir = cart_root / "ep1"
    ep2_dir = cart_root / "ep2"
    ep1_dir.mkdir(parents=True)
    ep2_dir.mkdir(parents=True)

    # Write Episode 1 cartridge
    (ep1_dir / "manifest.json").write_text(
        json.dumps({"episode_number": 1, "cartridge_id": "ep1", "version": "1.0.0"}), encoding="utf-8"
    )
    (ep1_dir / "mechanics.py").write_text(
        """from engine.api.contracts import BaseCartridge, BaseLevelSpec
class Ep1Cartridge(BaseCartridge):
    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(level_id="ep1", spawn_point=(100.0, 100.0), exit_point=(500.0, 500.0))
    def create_player(self, x, y, telemetry_recorder=None): return None
    def create_enemies(self, spec): return []
    def create_hazards(self, spec): return []
""",
        encoding="utf-8",
    )

    # Write Episode 2 cartridge
    (ep2_dir / "manifest.json").write_text(
        json.dumps({"episode_number": 2, "cartridge_id": "ep2", "version": "2.0.0"}), encoding="utf-8"
    )
    (ep2_dir / "mechanics.py").write_text(
        """from engine.api.contracts import BaseCartridge, BaseLevelSpec
class Ep2Cartridge(BaseCartridge):
    def get_level_spec(self, episode_number: int = 2) -> BaseLevelSpec:
        return BaseLevelSpec(level_id="ep2", spawn_point=(200.0, 200.0), exit_point=(800.0, 800.0))
    def create_player(self, x, y, telemetry_recorder=None): return None
    def create_enemies(self, spec): return []
    def create_hazards(self, spec): return []
""",
        encoding="utf-8",
    )

    cart1 = CartridgeLoader.load_cartridge(ep1_dir)
    assert cart1.manifest["episode_number"] == 1
    assert cart1.get_level_spec(1).spawn_point == (100.0, 100.0)

    cart2 = CartridgeLoader.load_cartridge(ep2_dir)
    assert cart2.manifest["episode_number"] == 2
    assert cart2.get_level_spec(2).spawn_point == (200.0, 200.0)


# ============================================================================
# M1 x M4: Multimodal Dialogue & Voice Audio Lookup
# ============================================================================

def test_m1_m4_multimodal_audio_and_dialogue_bark_resolution(mock_cartridge_active: Path):
    """M1 x M4: Cartridge contracts resolve narrative dialogue barks and voice audio assets."""
    cartridge = CartridgeLoader.load_cartridge(mock_cartridge_active)

    # 1. Dialogue barks lookup
    on_damage_barks = cartridge.get_dialogue_barks("on_damage")
    assert len(on_damage_barks) > 0
    bark = on_damage_barks[0]
    assert "character_id" in bark
    assert "text" in bark

    # 2. Audio asset resolution
    audio_rel = bark.get("audio_clip", "hero_start.wav")
    audio_full = mock_cartridge_active / "assets" / "audio" / audio_rel
    assert audio_full.exists()
    assert audio_full.read_bytes()[:4] == b"RIFF"


# ============================================================================
# M1 x M4 x M5: Atomic Promotion & Hot-Reload Pipeline
# ============================================================================

def test_m1_m4_m5_atomic_promotion_and_hot_reload_pipeline(tmp_path: Path):
    """M1 x M4 x M5: Candidate cartridge is promoted to active; loader reads new active version."""
    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    candidate_dir = cartridges_root / "candidate_01"
    candidate_dir.mkdir(parents=True)

    (candidate_dir / "manifest.json").write_text(
        json.dumps({"episode_number": 3, "cartridge_id": "ep3_promoted", "version": "3.0.0"}), encoding="utf-8"
    )
    (candidate_dir / "mechanics.py").write_text(
        """from engine.api.contracts import BaseCartridge, BaseLevelSpec
class PromotedCartridge(BaseCartridge):
    def get_level_spec(self, episode_number: int = 3) -> BaseLevelSpec:
        return BaseLevelSpec(level_id="ep3", spawn_point=(300.0, 300.0), exit_point=(900.0, 900.0))
    def create_player(self, x, y, telemetry_recorder=None): return None
    def create_enemies(self, spec): return []
    def create_hazards(self, spec): return []
""",
        encoding="utf-8",
    )

    supervisor = BuildSupervisor(cartridges_root=cartridges_root, active_cartridge_dir=active_dir)
    supervisor.promote_candidate(candidate_dir)

    assert active_dir.exists()
    assert (active_dir / "manifest.json").exists()

    reloaded_cart = CartridgeLoader.load_cartridge(active_dir)
    assert reloaded_cart.manifest["cartridge_id"] == "ep3_promoted"
    assert reloaded_cart.manifest["episode_number"] == 3


# ============================================================================
# M3 x M5: Cognitive Director x 3-Tier OS Priority Scheduler
# ============================================================================

def test_m3_m5_priority_scheduler_deliberation_and_priority_inversion_protection():
    """M3 x M5: Scheduler strictly enforces min-heap execution order (P0 > P1 > P2)."""
    scheduler = PriorityScheduler(max_background_workers=0)
    executed_order: List[str] = []

    def task_showrunner():
        executed_order.append("P2_SHOWRUNNER")

    def task_telemetry():
        executed_order.append("P1_TELEMETRY")

    def task_gameplay():
        executed_order.append("P0_GAMEPLAY")

    # Enqueue in reverse order: P2 first, then P1, then P0
    f_p2 = scheduler.submit(Priority.P2_SHOWRUNNER, "sw_synthesis", task_showrunner)
    f_p1 = scheduler.submit(Priority.P1_TELEMETRY, "wal_drain", task_telemetry)
    f_p0 = scheduler.submit(Priority.P0_GAMEPLAY, "cocoa_render", task_gameplay)

    # Step main thread to drain P0 tasks
    scheduler.step_main_thread(budget_ms=10.0)
    assert f_p0.done() is True
    assert executed_order == ["P0_GAMEPLAY"]

    # Pop remaining tasks directly from min-heap
    with scheduler._lock:
        import heapq
        while scheduler._task_heap:
            task = heapq.heappop(scheduler._task_heap)
            task.fn()

    assert executed_order == ["P0_GAMEPLAY", "P1_TELEMETRY", "P2_SHOWRUNNER"]


# ============================================================================
# M4 x M5: Cryptographic Artifact Envelope Tamper Detection
# ============================================================================

def test_m4_m5_cryptographic_artifact_envelope_tamper_detection(tmp_path: Path):
    """M4 x M5: Artifact metadata envelope detects file content tampering via SHA-256 mismatch."""
    cart_dir = tmp_path / "envelope_cart"
    cart_dir.mkdir()
    code_file = cart_dir / "mechanics.py"
    code_file.write_text("class OriginalCart:\n    pass\n", encoding="utf-8")

    envelope = ArtifactMetadataEnvelope.seal(
        deliberation="Sealing original cartridge mechanics file.",
        artifact_type=ArtifactType.GAME_MODULE,
        payload_path=code_file,
    )

    # Verification passes on unmodified file
    assert envelope.verify_integrity() is True

    # Tampering: modify 1 byte in mechanics.py
    code_file.write_text("class TamperedCart:\n    pass\n", encoding="utf-8")

    # Verification fails on modified file
    assert envelope.verify_integrity() is False


# ============================================================================
# M1 x M5: Concurrent WAL Telemetry Streaming & Studio Reading
# ============================================================================

def test_m1_m5_concurrent_wal_telemetry_streaming_and_analyst_queries(temp_e2e_db: str):
    """M1 x M5: Heavy background WAL telemetry writes do not block concurrent Analyst reads."""
    stop_event = threading.Event()
    write_count = [0]
    read_count = [0]

    def writer_thread():
        conn = sqlite3.connect(temp_e2e_db, timeout=10.0)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode = WAL;")
            cur.execute("INSERT OR IGNORE INTO telemetry_sessions (session_id) VALUES ('sess_concur')")
            cur.execute("INSERT OR IGNORE INTO telemetry_episodes (episode_id, session_id) VALUES ('ep_concur', 'sess_concur')")
            conn.commit()

            i = 0
            while not stop_event.is_set() and i < 100:
                cur.execute(
                    """
                    INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y, vel_x, vel_y)
                    VALUES ('sess_concur', 'ep_concur', ?, 'PLAYER_MOVE', ?, ?, 100.0, 0.0)
                    """,
                    (float(i) * 0.016, 100.0 + float(i), 200.0),
                )
                conn.commit()
                write_count[0] += 1
                i += 1
                time.sleep(0.001)
        finally:
            conn.close()

    def reader_thread():
        analyst = TelemetryAnalyst(db_path=temp_e2e_db)
        while not stop_event.is_set() and read_count[0] < 20:
            profile = analyst.analyze_session("sess_concur")
            assert profile is not None
            read_count[0] += 1
            time.sleep(0.005)

    t_writer = threading.Thread(target=writer_thread, daemon=True)
    t_reader = threading.Thread(target=reader_thread, daemon=True)

    t_writer.start()
    t_reader.start()

    t_writer.join(timeout=3.0)
    stop_event.set()
    t_reader.join(timeout=1.0)

    assert write_count[0] > 0
    assert read_count[0] > 0


# ============================================================================
# M2 x M4 x M6: End-to-End Steam DNA to Headless Simulation Loop
# ============================================================================

def test_m2_m4_m6_end_to_end_dna_to_headless_simulation(tmp_path: Path, mock_steam_installation: Path):
    """M2 x M4 x M6: Steam DNA -> Genesis Director -> Compiler -> Headless Sim runs 60 ticks cleanly."""
    scanner = SteamScanner(steam_root=mock_steam_installation)
    dataset = scanner.to_dataset()
    dna = SteamDNAProfiler().extract_dna(dataset, profile_tag="sim_user")

    director = GenesisDirector(specs_dir=tmp_path / "specs")
    spec = director.generate_genesis_spec(dna)

    cartridges_dir = tmp_path / "cartridges"
    compiler = EngineCompiler(cartridges_root=cartridges_dir)
    target_cart = cartridges_dir / "sim_cartridge"
    result = compiler.compile(spec, target_dir=target_cart)
    assert result.cartridge_loaded is True

    cartridge = CartridgeLoader.load_cartridge(target_cart)
    sim = HeadlessSimulator(cartridge=cartridge)
    for _ in range(60):
        step_res = sim.step()
        assert step_res is not None

    assert sim.step_count == 60


# ============================================================================
# M1 x M6: Frame Timing Budget Monitor & Scheduler Preemption Handshake
# ============================================================================

def test_m1_m6_frame_timing_budget_monitor_and_scheduler_preemption():
    """M1 x M6: Frame budget monitor detects frames exceeding 14.0ms and triggers preemption."""
    scheduler = PriorityScheduler(max_background_workers=0, frame_budget_ms=14.0)
    monitor = scheduler.frame_monitor

    # Simulate frame within budget (5.0ms)
    monitor.start_frame()
    time.sleep(0.005)
    t_normal = monitor.end_frame()
    assert t_normal.budget_exceeded is False
    assert scheduler.is_system_free() is True

    # Simulate frame exceeding budget (16.0ms)
    monitor.start_frame()
    time.sleep(0.016)
    t_exceeded = monitor.end_frame()
    assert t_exceeded.budget_exceeded is True
    assert scheduler.should_yield() is True
