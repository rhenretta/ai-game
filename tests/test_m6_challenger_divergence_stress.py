"""tests/test_m6_challenger_divergence_stress.py
Empirical Adversarial Stress Test Suite for Milestone 6:
Dual Steam Profile Divergence Simulation (Feature 17) & Autonomous E2E Acceptance Harness (Feature 18).

Adversarial Stress Dimensions:
  1. Pathological Steam Library Profiles: pure Bullet Hell vs pure Grand Strategy, 0 playtime, missing metadata,
     extreme outlier numbers to verify zero division or NaN and bound preservation.
  2. Velocity Delta Scaling: verify velocity delta strictly scales with pacing preference (>= 20%).
  3. AST Collision Invariance & Weapon Class Divergence: verify compiled cartridges have 0 AST collisions
     and distinct weapon classes.
  4. Concurrent Simulator Execution: verify simultaneous execution of both compiled cartridges in separate
     HeadlessSimulator instances operates without state contamination or thread race conditions.
  5. Acceptance Harness Rapid Stress: rapid back-to-back programmatic runs of verify_engine.py verifying
     no SQLite WAL database locking, uninitialized directory auto-creation, and CLI variations.
  6. Rationale-First Epistemic Architecture and project debranding compliance.
"""
from __future__ import annotations
import ast
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple
import pytest

import config
from engine.api.contracts import BaseCartridge
from engine.combat.weapon import BlasterWeapon, SpreadWeapon
from engine.compiler.compiler import CompilationResult, EngineCompiler
from engine.director.genesis import GenesisDirector
from engine.director.models import GenesisSpec
from engine.loader import CartridgeLoader
from engine.sim.headless import HeadlessSimulator, SimulationInput
from engine.steam.dna import SteamDNAProfiler
from engine.steam.models import SteamGameRecord, SteamLibraryDataset, SteamPlayerDNA
from verify_engine import EngineAcceptanceHarness, EngineVerificationReport, StageResult
from tests.test_tier1_units import validate_rationale_first_ast


# ============================================================================
# Pathological Fixtures
# ============================================================================

@pytest.fixture
def pure_bullet_hell_dataset() -> SteamLibraryDataset:
    """Extreme 100% Bullet Hell & Action Roguelike Steam library."""
    return SteamLibraryDataset(
        deliberation="Extreme pure bullet hell library with 10,000 total hours and 100% achievement completion.",
        player_id="steam_extreme_bullet_hell",
        source="stress_fixture",
        games=[
            SteamGameRecord(
                appid=311690,
                name="Enter the Gungeon",
                playtime_forever_minutes=300000,  # 5,000 hrs
                playtime_2weeks_minutes=6000,     # 100 hrs
                genres=["BulletHell", "Action"],
                tags=["bullet hell", "shoot 'em up", "danmaku", "fast-paced"],
                achievements_total=100,
                achievements_unlocked=100,
            ),
            SteamGameRecord(
                appid=1145360,
                name="Hades",
                playtime_forever_minutes=180000,  # 3,000 hrs
                playtime_2weeks_minutes=3000,     # 50 hrs
                genres=["Action", "Roguelike"],
                tags=["bullet hell", "hack and slash", "fast-paced"],
                achievements_total=50,
                achievements_unlocked=50,
            ),
            SteamGameRecord(
                appid=588650,
                name="Dead Cells",
                playtime_forever_minutes=120000,  # 2,000 hrs
                playtime_2weeks_minutes=1200,     # 20 hrs
                genres=["Action", "Roguelike"],
                tags=["action roguelike", "fast-paced"],
                achievements_total=32,
                achievements_unlocked=32,
            ),
        ],
    )


@pytest.fixture
def pure_grand_strategy_dataset() -> SteamLibraryDataset:
    """Extreme 100% Turn-Based Grand Strategy & Tactical Simulation Steam library."""
    return SteamLibraryDataset(
        deliberation="Extreme pure grand strategy library with 10,000 total hours and tactical simulation.",
        player_id="steam_extreme_grand_strategy",
        source="stress_fixture",
        games=[
            SteamGameRecord(
                appid=289070,
                name="Sid Meier's Civilization VI",
                playtime_forever_minutes=360000,  # 6,000 hrs
                playtime_2weeks_minutes=3600,     # 60 hrs
                genres=["Strategy"],
                tags=["turn-based strategy", "grand strategy", "tactical"],
                achievements_total=150,
                achievements_unlocked=10,
            ),
            SteamGameRecord(
                appid=200510,
                name="XCOM: Enemy Unknown",
                playtime_forever_minutes=120000,  # 2,000 hrs
                playtime_2weeks_minutes=1200,     # 20 hrs
                genres=["Strategy", "Shooter"],
                tags=["turn-based tactics", "tactical"],
                achievements_total=85,
                achievements_unlocked=15,
            ),
            SteamGameRecord(
                appid=590380,
                name="Into the Breach",
                playtime_forever_minutes=120000,  # 2,000 hrs
                playtime_2weeks_minutes=1200,     # 20 hrs
                genres=["Strategy", "Puzzle"],
                tags=["turn-based", "tactical", "puzzle"],
                achievements_total=50,
                achievements_unlocked=5,
            ),
        ],
    )


# ============================================================================
# 1. Pathological Steam Library Profiles & Bound Preservation
# ============================================================================

class TestPathologicalSteamProfileStress:
    """Adversarial stress-testing of SteamDNAProfiler and GenesisDirector under pathological inputs."""
    deliberation: str = "Stress testing Steam DNA profiling and Genesis generation under pathological input distributions."

    def test_pure_bullet_hell_profile_extremes(
        self,
        pure_bullet_hell_dataset: SteamLibraryDataset,
    ) -> None:
        """Verifies mathematical outputs of pure Bullet Hell profile reach expected extreme bounds without NaN."""
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(pure_bullet_hell_dataset, profile_tag="extreme_bullet_hell")

        assert dna.pacing_preference == "kinetic_high_mobility"
        assert dna.pacing_score > 0.80
        assert dna.challenge_tolerance > 0.75
        assert dna.mechanical_biases["dodge_roll"] > 0.50
        assert math.isnan(dna.pacing_score) is False
        assert math.isnan(dna.challenge_tolerance) is False
        assert abs(sum(dna.genre_weights.values()) - 1.0) < 1e-3

    def test_pure_grand_strategy_profile_extremes(
        self,
        pure_grand_strategy_dataset: SteamLibraryDataset,
    ) -> None:
        """Verifies mathematical outputs of pure Grand Strategy profile reach expected negative bounds without NaN."""
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(pure_grand_strategy_dataset, profile_tag="extreme_grand_strategy")

        assert dna.pacing_preference == "tactical_methodical"
        assert dna.pacing_score < -0.70
        assert 0.0 <= dna.challenge_tolerance <= 1.0
        assert dna.mechanical_biases["dodge_roll"] < 0.20
        assert math.isnan(dna.pacing_score) is False
        assert math.isnan(dna.challenge_tolerance) is False
        assert abs(sum(dna.genre_weights.values()) - 1.0) < 1e-3

    def test_zero_playtime_and_empty_library_resilience(self) -> None:
        """Verifies zero division guards when playtime is 0 or library contains zero games."""
        profiler = SteamDNAProfiler()

        # Case A: Games exist but 0 playtime and 0 achievements
        zero_playtime_dataset = SteamLibraryDataset(
            deliberation="Zero playtime library with 2 unplayed games.",
            player_id="steam_zero_playtime",
            source="stress_fixture",
            games=[
                SteamGameRecord(
                    appid=1001,
                    name="Unplayed Game 1",
                    playtime_forever_minutes=0,
                    playtime_2weeks_minutes=0,
                    genres=["Action"],
                    tags=["fast-paced"],
                    achievements_total=0,
                    achievements_unlocked=0,
                ),
                SteamGameRecord(
                    appid=1002,
                    name="Unplayed Game 2",
                    playtime_forever_minutes=0,
                    playtime_2weeks_minutes=0,
                    genres=["Strategy"],
                    tags=["turn-based"],
                    achievements_total=0,
                    achievements_unlocked=0,
                ),
            ],
        )
        dna_zero = profiler.extract_dna(zero_playtime_dataset, profile_tag="zero_playtime")
        assert math.isnan(dna_zero.pacing_score) is False
        assert math.isnan(dna_zero.challenge_tolerance) is False
        assert -1.0 <= dna_zero.pacing_score <= 1.0
        assert 0.0 <= dna_zero.challenge_tolerance <= 1.0
        assert abs(sum(dna_zero.genre_weights.values()) - 1.0) < 1e-3

        # Case B: Completely empty library (0 games)
        empty_dataset = SteamLibraryDataset(
            deliberation="Completely empty Steam library.",
            player_id="steam_empty_lib",
            source="stress_fixture",
            games=[],
        )
        dna_empty = profiler.extract_dna(empty_dataset, profile_tag="empty_library")
        assert dna_empty.total_games_analyzed == 0
        assert dna_empty.total_playtime_hours == 0.0
        assert dna_empty.pacing_score == 0.0
        assert dna_empty.challenge_tolerance == 0.5
        assert math.isnan(dna_empty.pacing_score) is False
        assert abs(sum(dna_empty.genre_weights.values()) - 1.0) < 1e-3

    def test_missing_and_corrupt_metadata_handling(self) -> None:
        """Verifies resilience when genres and tags are completely empty and 2-week playtime is None."""
        profiler = SteamDNAProfiler()
        corrupt_dataset = SteamLibraryDataset(
            deliberation="Dataset with missing genres, tags, and null recent playtime.",
            player_id="steam_corrupt_meta",
            source="stress_fixture",
            games=[
                SteamGameRecord(
                    appid=2001,
                    name="Blank App 1",
                    playtime_forever_minutes=1200,
                    playtime_2weeks_minutes=None,
                    genres=[],
                    tags=[],
                    achievements_total=0,
                    achievements_unlocked=0,
                ),
                SteamGameRecord(
                    appid=2002,
                    name="Blank App 2",
                    playtime_forever_minutes=600,
                    playtime_2weeks_minutes=None,
                    genres=[],
                    tags=[],
                    achievements_total=0,
                    achievements_unlocked=0,
                ),
            ],
        )
        dna = profiler.extract_dna(corrupt_dataset, profile_tag="corrupt_metadata")
        assert math.isnan(dna.pacing_score) is False
        assert math.isnan(dna.challenge_tolerance) is False
        assert abs(sum(dna.genre_weights.values()) - 1.0) < 1e-3
        assert len(dna.top_genres) > 0

    def test_extreme_outlier_numbers(self) -> None:
        """Verifies numeric stability when numbers are absurdly massive."""
        profiler = SteamDNAProfiler()
        outlier_dataset = SteamLibraryDataset(
            deliberation="Dataset with massive outlier playtime numbers.",
            player_id="steam_outlier",
            source="stress_fixture",
            games=[
                SteamGameRecord(
                    appid=3001,
                    name="Immense Playtime Game",
                    playtime_forever_minutes=100_000_000,
                    playtime_2weeks_minutes=500_000,
                    genres=["Action", "BulletHell"],
                    tags=["bullet hell", "action"],
                    achievements_total=999_999,
                    achievements_unlocked=999_999,
                )
            ],
        )
        dna = profiler.extract_dna(outlier_dataset, profile_tag="outlier")
        assert math.isfinite(dna.pacing_score)
        assert math.isfinite(dna.challenge_tolerance)
        assert -1.0 <= dna.pacing_score <= 1.0
        assert 0.0 <= dna.challenge_tolerance <= 1.0
        assert all(math.isfinite(w) for w in dna.genre_weights.values())

    def test_genesis_director_bound_preservation_under_extremes(
        self,
        pure_bullet_hell_dataset: SteamLibraryDataset,
        pure_grand_strategy_dataset: SteamLibraryDataset,
    ) -> None:
        """Verifies that GenesisDirector strictly bounds all synthesized player and level parameters."""
        profiler = SteamDNAProfiler()
        director = GenesisDirector()

        dna_kinetic = profiler.extract_dna(pure_bullet_hell_dataset, profile_tag="kinetic")
        dna_tactical = profiler.extract_dna(pure_grand_strategy_dataset, profile_tag="tactical")

        spec_kinetic = director.generate_genesis_spec(dna_kinetic)
        spec_tactical = director.generate_genesis_spec(dna_tactical)

        for spec in (spec_kinetic, spec_tactical):
            p = spec.player_params
            assert 200.0 <= p.walk_speed <= 360.0
            assert 450.0 <= p.dash_speed <= 850.0
            assert 0.35 <= p.dash_cooldown <= 1.00
            assert 0.70 <= spec.target_difficulty <= 1.30
            assert math.isfinite(p.walk_speed)
            assert math.isfinite(p.dash_speed)
            assert math.isfinite(p.dash_cooldown)


# ============================================================================
# 2. Velocity Delta Scaling with Pacing Preference
# ============================================================================

class TestPacingVelocityDeltaScaling:
    """Verifies that player velocities scale strictly with pacing preference (>= 20%)."""
    deliberation: str = "Verifying velocity delta scaling between extreme kinetic and tactical profiles (>= 20%)."

    def test_velocity_delta_strictly_scales_with_pacing_preference(
        self,
        pure_bullet_hell_dataset: SteamLibraryDataset,
        pure_grand_strategy_dataset: SteamLibraryDataset,
    ) -> None:
        """Asserts walk speed delta and dash speed delta strictly exceed 20% between kinetic and tactical."""
        profiler = SteamDNAProfiler()
        director = GenesisDirector()

        dna_kinetic = profiler.extract_dna(pure_bullet_hell_dataset, profile_tag="kinetic")
        dna_tactical = profiler.extract_dna(pure_grand_strategy_dataset, profile_tag="tactical")

        spec_kinetic = director.generate_genesis_spec(dna_kinetic)
        spec_tactical = director.generate_genesis_spec(dna_tactical)

        walk_k = spec_kinetic.player_params.walk_speed
        walk_t = spec_tactical.player_params.walk_speed
        dash_k = spec_kinetic.player_params.dash_speed
        dash_t = spec_tactical.player_params.dash_speed
        cd_k = spec_kinetic.player_params.dash_cooldown
        cd_t = spec_tactical.player_params.dash_cooldown

        # Walk Speed Delta >= 20%
        walk_delta = (walk_k - walk_t) / walk_t
        assert walk_delta >= 0.20, f"Walk speed delta was {walk_delta:.2%}, expected >= 20%"
        assert walk_k > walk_t

        # Dash Speed Delta >= 20%
        dash_delta = (dash_k - dash_t) / dash_t
        assert dash_delta >= 0.20, f"Dash speed delta was {dash_delta:.2%}, expected >= 20%"
        assert dash_k > dash_t

        # Dash Cooldown Scaling: tactical cooldown must be substantially longer
        assert cd_t > cd_k * 1.30, f"Expected tactical cooldown {cd_t:.2f}s > kinetic {cd_k:.2f}s * 1.30"

        # Jump force scaling
        assert spec_kinetic.player_params.jump_force > spec_tactical.player_params.jump_force


# ============================================================================
# 3. Compiled Cartridge AST Collision & Weapon Divergence
# ============================================================================

class TestCompiledCartridgeASTCollisionAndWeaponDivergence:
    """Verifies compiled cartridges have zero AST collisions and distinct weapon classes."""
    deliberation: str = "Verifying compiled cartridge AST collision invariance and polymorphic weapon divergence."

    def test_compiled_cartridges_zero_ast_collisions_and_distinct_weapons(
        self,
        pure_bullet_hell_dataset: SteamLibraryDataset,
        pure_grand_strategy_dataset: SteamLibraryDataset,
        tmp_path: Path,
    ) -> None:
        """Asserts 0 AST syntax collisions and distinct runtime weapon classes across compiled cartridges."""
        profiler = SteamDNAProfiler()
        director = GenesisDirector()
        compiler = EngineCompiler(cartridges_root=tmp_path / "cartridges")

        dna_k = profiler.extract_dna(pure_bullet_hell_dataset, profile_tag="kinetic")
        dna_t = profiler.extract_dna(pure_grand_strategy_dataset, profile_tag="tactical")

        spec_k = director.generate_genesis_spec(dna_k)
        spec_t = director.generate_genesis_spec(dna_t)

        dir_k = tmp_path / "cartridges" / "extreme_kinetic"
        dir_t = tmp_path / "cartridges" / "extreme_tactical"

        res_k = compiler.compile(spec_k, target_dir=dir_k)
        res_t = compiler.compile(spec_t, target_dir=dir_t)

        assert res_k.ast_validated is True
        assert res_t.ast_validated is True
        assert res_k.bytecode_compiled is True
        assert res_t.bytecode_compiled is True

        # AST Collision Check
        for py_name in ("entities.py", "mechanics.py"):
            code_k = (dir_k / py_name).read_text(encoding="utf-8")
            code_t = (dir_t / py_name).read_text(encoding="utf-8")

            ast_k = ast.dump(ast.parse(code_k))
            ast_t = ast.dump(ast.parse(code_t))
            assert ast_k != ast_t, f"AST syntax tree collision detected in {py_name}!"

            hash_k = hashlib.sha256(ast_k.encode("utf-8")).hexdigest()
            hash_t = hashlib.sha256(ast_t.encode("utf-8")).hexdigest()
            assert hash_k != hash_t, f"AST SHA-256 collision in {py_name}!"

        # Dynamic Cartridge Loading & Weapon Class Divergence
        cart_k = CartridgeLoader.load_cartridge(dir_k)
        cart_t = CartridgeLoader.load_cartridge(dir_t)

        player_k = cart_k.create_player(100.0, 100.0)
        player_t = cart_t.create_player(100.0, 100.0)

        # Distinct weapon classes
        assert isinstance(player_k.weapon, SpreadWeapon), f"Expected SpreadWeapon, got {type(player_k.weapon)}"
        assert isinstance(player_t.weapon, BlasterWeapon), f"Expected BlasterWeapon, got {type(player_t.weapon)}"
        assert type(player_k.weapon) is not type(player_t.weapon)

        # Distinct weapon attributes
        assert player_k.weapon.damage != player_t.weapon.damage
        assert player_k.weapon.projectile_speed != player_t.weapon.projectile_speed


# ============================================================================
# 4. Concurrent Simulator Execution Isolation
# ============================================================================

class TestConcurrentSimulatorExecutionIsolation:
    """Verifies concurrent execution of both compiled cartridges in separate HeadlessSimulator instances."""
    deliberation: str = "Verifying concurrent execution isolation and zero state leakage across HeadlessSimulator instances."

    def test_concurrent_headless_simulation_zero_state_leakage(
        self,
        pure_bullet_hell_dataset: SteamLibraryDataset,
        pure_grand_strategy_dataset: SteamLibraryDataset,
        tmp_path: Path,
    ) -> None:
        """Asserts concurrent execution of two simulators does not cross-contaminate positions, stats, or telemetry."""
        profiler = SteamDNAProfiler()
        director = GenesisDirector()
        compiler = EngineCompiler(cartridges_root=tmp_path / "cartridges")

        dna_k = profiler.extract_dna(pure_bullet_hell_dataset, profile_tag="kinetic")
        dna_t = profiler.extract_dna(pure_grand_strategy_dataset, profile_tag="tactical")

        spec_k = director.generate_genesis_spec(dna_k)
        spec_t = director.generate_genesis_spec(dna_t)

        dir_k = tmp_path / "cartridges" / "sim_kinetic"
        dir_t = tmp_path / "cartridges" / "sim_tactical"

        compiler.compile(spec_k, target_dir=dir_k)
        compiler.compile(spec_t, target_dir=dir_t)

        cart_k = CartridgeLoader.load_cartridge(dir_k)
        cart_t = CartridgeLoader.load_cartridge(dir_t)

        sim_k = HeadlessSimulator(
            cartridge=cart_k,
            episode_number=1,
            telemetry_mode="in_memory",
            ignore_combat_lock=True,
            max_steps=120,
        )
        sim_t = HeadlessSimulator(
            cartridge=cart_t,
            episode_number=1,
            telemetry_mode="in_memory",
            ignore_combat_lock=True,
            max_steps=120,
        )

        def run_sim_worker(sim: HeadlessSimulator, move_vec: Tuple[float, float], dash: bool) -> None:
            for _ in range(60):
                sim.step(SimulationInput(move_x=move_vec[0], move_y=move_vec[1], dash=dash))

        # Run both simulations concurrently via ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_k = executor.submit(run_sim_worker, sim_k, (1.0, 0.0), True)
            future_t = executor.submit(run_sim_worker, sim_t, (0.0, 1.0), False)
            future_k.result(timeout=10.0)
            future_t.result(timeout=10.0)

        # 1. State Isolation: Positions must diverge along assigned movement axes
        assert sim_k.step_count == 60
        assert sim_t.step_count == 60
        assert sim_k.player.pos.x != sim_t.player.pos.x
        assert sim_k.player.pos.y != sim_t.player.pos.y

        # 2. Attribute Invariance: Speeds must remain independent
        assert sim_k.player.walk_speed > sim_t.player.walk_speed
        assert (sim_k.player.walk_speed - sim_t.player.walk_speed) / sim_t.player.walk_speed >= 0.20

        # 3. Action Isolation: Dashes counted on kinetic, 0 on tactical
        assert sim_k.stats["dashes"] > 0
        assert sim_t.stats["dashes"] == 0

        # 4. Telemetry Buffer & Recorder Isolation
        rec_k = sim_k.telemetry_recorder
        rec_t = sim_t.telemetry_recorder
        assert rec_k is not rec_t
        assert rec_k.events is not rec_t.events
        assert len(rec_k.events) > 0
        assert len(rec_t.events) > 0

        # Verify event isolation: mutating or clearing rec_k does not affect rec_t
        len_t_before = len(rec_t.events)
        rec_k.events.clear()
        assert len(rec_k.events) == 0
        assert len(rec_t.events) == len_t_before


# ============================================================================
# 5. Acceptance Harness Rapid Execution Stress (No SQLite WAL Locking)
# ============================================================================

class TestAcceptanceHarnessRapidExecutionStress:
    """Verifies verify_engine.py handles back-to-back runs without SQLite WAL database locking."""
    deliberation: str = "Stress testing verify_engine.py across rapid consecutive runs and directory edge cases."

    def test_rapid_back_to_back_runs_no_sqlite_wal_locking(self, tmp_path: Path) -> None:
        """Executes 3 rapid consecutive programmatic runs of verify_engine on the same database path."""
        db_path = tmp_path / "rapid_telemetry.db"
        cartridges_dir = tmp_path / "rapid_cartridges"
        specs_dir = tmp_path / "rapid_specs"

        for cycle in range(1, 4):
            harness = EngineAcceptanceHarness(
                project_root=config.PROJECT_ROOT,
                db_path=db_path,
                cartridges_dir=cartridges_dir,
                specs_dir=specs_dir,
                verbose=False,
            )
            report = harness.run_all()
            assert report.passed is True, f"Cycle {cycle} failed: {[s.error_message for s in report.stages if not s.passed]}"
            assert len(report.stages) == 6
            assert all(s.passed for s in report.stages)

            # Assert database remains valid WAL without corruption or deadlocks
            conn = sqlite3.connect(str(db_path), timeout=5.0)
            try:
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode;")
                row = cur.fetchone()
                assert row is not None and row[0].lower() == "wal"
                cur.execute("SELECT COUNT(*) FROM telemetry_events;")
                count = cur.fetchone()[0]
                assert count > 0
            finally:
                conn.close()

    def test_uninitialized_and_nested_directory_auto_creation(self, tmp_path: Path) -> None:
        """Verifies that missing or deeply nested directories are auto-created without error."""
        deep_root = tmp_path / "deep" / "nested" / "sandbox"
        deep_db = deep_root / "data" / "telemetry.db"
        deep_cartridges = deep_root / "cartridges"
        deep_specs = deep_root / "specs"

        assert deep_root.exists() is False

        harness = EngineAcceptanceHarness(
            project_root=config.PROJECT_ROOT,
            db_path=deep_db,
            cartridges_dir=deep_cartridges,
            specs_dir=deep_specs,
            verbose=False,
        )
        report = harness.run_all()

        assert report.passed is True
        assert deep_db.is_file()
        assert (deep_cartridges / "genesis").is_dir()
        assert (deep_cartridges / "episode_01").is_dir()

    def test_cli_execution_with_arguments(self, tmp_path: Path) -> None:
        """Verifies CLI subprocess execution of verify_engine.py with argument variations."""
        cli_db = tmp_path / "cli_telemetry.db"
        cli_cartridges = tmp_path / "cli_cartridges"
        cli_specs = tmp_path / "cli_specs"

        cmd = [
            sys.executable,
            str(config.PROJECT_ROOT / "verify_engine.py"),
            "--db-path", str(cli_db),
            "--cartridges-dir", str(cli_cartridges),
            "--specs-dir", str(cli_specs),
            "--quiet",
        ]
        res = subprocess.run(cmd, cwd=str(config.PROJECT_ROOT), capture_output=True, text=True)
        assert res.returncode == 0, f"CLI verify_engine failed with exit code {res.returncode}: {res.stderr}"
        assert cli_db.is_file()
        assert (cli_cartridges / "genesis").is_dir()


# ============================================================================
# 6. Rationale-First Epistemic Architecture & Project Debranding
# ============================================================================

class TestChallengerCompliance:
    """Verifies that this stress test file adheres strictly to architectural invariants."""
    deliberation: str = "Verifying Rationale-First AST compliance and zero occurrences of forbidden legacy branding."

    def test_test_suite_ast_deliberation_first(self) -> None:
        """Assures all test classes in this suite declare deliberation strictly first."""
        this_file = Path(__file__).resolve()
        source_code = this_file.read_text(encoding="utf-8")
        assert validate_rationale_first_ast(source_code) is True

    def test_zero_forbidden_branding_in_test_source(self) -> None:
        """Assures exactly 0 occurrences of forbidden legacy branding exist in this test file."""
        forbidden = "".join(chr(c) for c in (106, 111, 97, 110))
        this_file = Path(__file__).resolve()
        content = this_file.read_text(encoding="utf-8").lower()
        assert forbidden not in content
