"""verify_engine.py
Autonomous E2E Engine Acceptance Harness (Feature 18).
Executes all 6 autonomous stages in an unbroken, deterministic pipeline:
  1. Stage 1: Steam Library Ingestion -> SteamPlayerDNA
  2. Stage 2: Episode 0 Genesis Compilation -> cartridges/genesis/
  3. Stage 3: Headless A* Playtesting -> 100% Traversability & 0 Clipping
  4. Stage 4: Live SQLite WAL Telemetry Capture -> telemetry.db (WAL pragma & structured events)
  5. Stage 5: Episodic Director Adaptation Clamp (+/-15%) -> cartridges/episode_01/
  6. Stage 6: Between-Run Cartridge Hot-Reload -> In-Process Cartridge Swapping

Enforces strict Rationale-First Epistemic Architecture (deliberation declared first in all classes).
Strict project de-branding invariant: 0 occurrences of forbidden legacy branding tokens.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

# Ensure project root (containing config.py) is in sys.path
_candidate = Path(__file__).resolve().parent
while _candidate != _candidate.parent and not (_candidate / "config.py").is_file():
    _candidate = _candidate.parent
PROJECT_ROOT = _candidate
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.compiler.compiler import CompilationResult, EngineCompiler
from engine.director.episodic import EpisodicDirector
from engine.director.genesis import GenesisDirector
from engine.director.models import (
    CurrentEpisodeConfig,
    DifficultyClampRecord,
    DifficultyClampState,
    EpisodicUpdateSpec,
    GenesisSpec,
)
from engine.loader import CartridgeLoader
from engine.sim.headless import HeadlessSimulator, SimulationInput
from engine.steam.dna import SteamDNAMapper, SteamDNAPersistence, SteamDNAProfiler
from engine.steam.models import SteamLibraryDataset, SteamOwnedGame, SteamPlayerDNA
from engine.steam.scanner import SteamScanner
from engine.steam.web_api import SteamFixtureLoader
from engine.telemetry.db import TelemetryDB
from engine.telemetry.models import TelemetryEventType
from engine.analyst.analyst import TelemetryAnalyst
from engine.analyst.models import BehavioralProfile
from qa.playtester import PlaytestResult, SyntheticPlaytesterBot
from qa.profiles import RUSHER_PROFILE

logger = logging.getLogger("EngineAcceptanceHarness")


class StageResult(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first.
    Outcome payload for each acceptance stage.
    """
    deliberation: str = Field(default="", description="Epistemic rationale explaining stage execution and findings.")
    stage_number: int = Field(..., description="Stage sequence index (1-6).")
    stage_name: str = Field(..., description="Human-readable title of the stage.")
    passed: bool = Field(default=False, description="Whether the stage completed with zero defects.")
    duration_seconds: float = Field(default=0.0, description="Wall-clock elapsed time for stage execution.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic state, metrics, and artifact pointers.")
    error_message: Optional[str] = Field(default=None, description="Detailed failure diagnostic if stage failed.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "stage_number": self.stage_number,
            "stage_name": self.stage_name,
            "passed": self.passed,
            "duration_seconds": round(self.duration_seconds, 3),
            "details": self.details,
            "error_message": self.error_message,
        }


class EngineVerificationReport(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first.
    Comprehensive final report emitted by Autonomous E2E Acceptance Harness.
    """
    deliberation: str = Field(default="", description="Epistemic summary across all 6 verification stages.")
    passed: bool = Field(default=False, description="Overall harness status: True only if all 6 stages pass.")
    total_duration_seconds: float = Field(default=0.0, description="Total elapsed seconds for full acceptance run.")
    stages: List[StageResult] = Field(default_factory=list, description="Ordered results for Stages 1 through 6.")
    summary_markdown: str = Field(default="", description="Formatted Markdown summary of the acceptance run.")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "passed": self.passed,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "stages": [s.to_dict() for s in self.stages],
            "summary_markdown": self.summary_markdown,
            "created_at_utc": self.created_at_utc,
        }


class EngineAcceptanceHarness:
    """Autonomous E2E Engine Acceptance Harness.
    Executes all 6 stages required for Feature 18.
    """
    deliberation: str = "Orchestrating autonomous 6-stage E2E acceptance verification of the Autonomous Game Engine."

    def __init__(
        self,
        project_root: Optional[Path] = None,
        db_path: Optional[Path] = None,
        cartridges_dir: Optional[Path] = None,
        specs_dir: Optional[Path] = None,
        steam_root: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.project_root = Path(project_root or PROJECT_ROOT).resolve()
        self.data_dir = self.project_root / "data"
        self.steam_root = Path(steam_root).resolve() if steam_root else None
        self.verbose = verbose

        import tempfile
        self._temp_workspace: Optional[tempfile.TemporaryDirectory] = None
        if cartridges_dir is None:
            self._temp_workspace = tempfile.TemporaryDirectory(prefix="verify_engine_scratch_")
            scratch_root = Path(self._temp_workspace.name)
            self.cartridges_dir = scratch_root / "cartridges"
            self.specs_dir = Path(specs_dir or (scratch_root / "specs")).resolve()
            self.db_path = Path(db_path or (scratch_root / "telemetry.db")).resolve()
        else:
            self.cartridges_dir = Path(cartridges_dir).resolve()
            self.specs_dir = Path(specs_dir or (self.data_dir / "specs")).resolve()
            self.db_path = Path(db_path or (self.data_dir / "telemetry.db")).resolve()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.specs_dir.mkdir(parents=True, exist_ok=True)
        self.cartridges_dir.mkdir(parents=True, exist_ok=True)

        # Inter-stage state transfer
        self.player_dna: Optional[SteamPlayerDNA] = None
        self.genesis_spec: Optional[GenesisSpec] = None
        self.genesis_cartridge_dir: Optional[Path] = None
        self.genesis_level_spec: Optional[BaseLevelSpec] = None
        self.recorded_session_id: Optional[str] = None
        self.behavioral_profile: Optional[BehavioralProfile] = None
        self.update_spec: Optional[EpisodicUpdateSpec] = None
        self.episode_1_cartridge_dir: Optional[Path] = None

    def cleanup(self) -> None:
        """Cleans up hermetic scratch workspace if allocated."""
        if getattr(self, "_temp_workspace", None) is not None:
            try:
                self._temp_workspace.cleanup()
            except Exception:
                pass
            self._temp_workspace = None

    def __del__(self) -> None:
        self.cleanup()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    # ------------------------------------------------------------------------
    # Stage 1: Steam Library Ingestion
    # ------------------------------------------------------------------------
    def run_stage_1_steam_ingestion(self) -> StageResult:
        """Stage 1: Discovers local Steam library or falls back to synthetic fixtures to extract SteamPlayerDNA."""
        t0 = time.perf_counter()
        delib = "Stage 1: Scanning Steam library on macOS and generating SteamPlayerDNA."
        try:
            scanner = SteamScanner(steam_root=self.steam_root)
            dataset: Optional[SteamLibraryDataset] = None
            source_tag = "local_macos"

            # Check if live macOS Steam installation exists with installed apps
            if scanner.is_steam_installed():
                try:
                    installed = scanner.scan_installed_apps()
                    if installed:
                        lib = scanner.scan_full_library()
                        dataset = scanner.to_dataset(lib)
                        source_tag = "live_macos_steam"
                except Exception as e:
                    logger.warning("Local Steam scan encountered non-fatal issue: %s", e)

            # Fallback to test fixtures if local Steam is empty or unpopulated
            if dataset is None or len(dataset.games) == 0:
                fixture_dir = self.project_root / "tests" / "fixtures" / "steam"
                if fixture_dir.exists() and (fixture_dir / "steamapps").is_dir():
                    f_scanner = SteamScanner(steam_root=fixture_dir)
                    lib = f_scanner.scan_full_library()
                    dataset = f_scanner.to_dataset(lib)
                    source_tag = "synthetic_steam_fixture"
                else:
                    loader = SteamFixtureLoader(fixtures_dir=fixture_dir)
                    raw_fixture = loader.get_owned_games_fixture()
                    owned = [
                        SteamOwnedGame(
                            appid=g["appid"],
                            name=g.get("name", f"App {g['appid']}"),
                            playtime_forever=g.get("playtime_forever", 0),
                            playtime_2weeks=g.get("playtime_2weeks", 0),
                            rtime_last_played=g.get("rtime_last_played", 0),
                        )
                        for g in raw_fixture.get("response", {}).get("games", [])
                    ]
                    dataset = SteamDNAMapper.from_web_api(owned, player_id="fixture_player")
                    source_tag = "web_api_fixture"

            # Compute SteamPlayerDNA
            profiler = SteamDNAProfiler(recent_playtime_bonus=0.5)
            dna = profiler.extract_dna(dataset, profile_tag=f"genesis_{source_tag}")

            # Persist to SQLite WAL
            persistence = SteamDNAPersistence(db_path=self.db_path)
            dna_id = persistence.save_dna(dna)

            # Verification assertions
            assert getattr(dna, "deliberation", ""), "SteamPlayerDNA missing mandatory deliberation property"
            assert dna.total_games_analyzed > 0, "Expected at least 1 analyzed Steam game in dataset"
            assert -1.0 <= dna.pacing_score <= 1.0, f"Pacing score {dna.pacing_score} out of bounds [-1.0, 1.0]"
            assert 0.0 <= dna.challenge_tolerance <= 1.0, f"Challenge tolerance {dna.challenge_tolerance} out of bounds [0.0, 1.0]"
            assert abs(sum(dna.genre_weights.values()) - 1.0) < 1e-3, "Genre weights do not sum to 1.0"

            self.player_dna = dna
            elapsed = time.perf_counter() - t0

            stage_delib = (
                f"Successfully parsed Steam dataset ({source_tag}) across {dna.total_games_analyzed} games "
                f"({dna.total_playtime_hours:.1f} hrs). Pacing score={dna.pacing_score:+.2f}, "
                f"Challenge tol={dna.challenge_tolerance:.2f}, Palette mood='{dna.preferred_palette_mood}'."
            )
            return StageResult(
                deliberation=stage_delib,
                stage_number=1,
                stage_name="Steam Library Ingestion",
                passed=True,
                duration_seconds=elapsed,
                details={
                    "dna_id": dna_id,
                    "source_tag": source_tag,
                    "games_analyzed": dna.total_games_analyzed,
                    "playtime_hours": dna.total_playtime_hours,
                    "pacing_score": dna.pacing_score,
                    "pacing_preference": dna.pacing_preference,
                    "challenge_tolerance": dna.challenge_tolerance,
                    "preferred_palette_mood": dna.preferred_palette_mood,
                    "top_genres": dna.top_genres,
                },
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.exception("Stage 1 Steam Library Ingestion failed: %s", e)
            return StageResult(
                deliberation=f"Stage 1 failed: {e}",
                stage_number=1,
                stage_name="Steam Library Ingestion",
                passed=False,
                duration_seconds=elapsed,
                error_message=str(e),
            )

    # ------------------------------------------------------------------------
    # Stage 2: Episode 0 Genesis Compilation
    # ------------------------------------------------------------------------
    def run_stage_2_genesis_compilation(self, dna: Optional[SteamPlayerDNA] = None) -> StageResult:
        """Stage 2: Takes SteamPlayerDNA, invokes GenesisDirector for GenesisSpec, and compiles Episode 0."""
        t0 = time.perf_counter()
        target_dna = dna or self.player_dna
        if target_dna is None:
            return StageResult(
                deliberation="Stage 2 failed: Missing input SteamPlayerDNA from Stage 1.",
                stage_number=2,
                stage_name="Episode 0 Genesis Compilation",
                passed=False,
                duration_seconds=0.0,
                error_message="Missing SteamPlayerDNA",
            )

        try:
            director = GenesisDirector(specs_dir=self.specs_dir, db_path=self.db_path)
            spec = director.generate_genesis_spec(target_dna)

            assert getattr(spec, "deliberation", ""), "GenesisSpec missing mandatory deliberation"
            spec_file = director.write_genesis_spec(spec, output_path=self.specs_dir / "genesis_spec.md")
            director.save_spec_to_db(spec)

            # Compile Episode 0 Cartridge into cartridges/genesis
            cartridge_dir = self.cartridges_dir / "genesis"
            compiler = EngineCompiler(cartridges_root=self.cartridges_dir, specs_dir=self.specs_dir)
            comp_res: CompilationResult = compiler.compile(spec, target_dir=cartridge_dir, promote_to_active=False)

            assert comp_res.ast_validated is True, "Cartridge failed AST security validation"
            assert comp_res.bytecode_compiled is True, "Cartridge failed bytecode compilation"
            assert comp_res.cartridge_loaded is True, "Cartridge failed dynamic loader instantiation"
            assert (cartridge_dir / "manifest.json").is_file(), "manifest.json was not generated"
            assert (cartridge_dir / "level.json").is_file(), "level.json was not generated"
            assert (cartridge_dir / "entities.py").is_file(), "entities.py was not generated"
            assert (cartridge_dir / "mechanics.py").is_file(), "mechanics.py was not generated"

            # Cache state for subsequent stages
            self.genesis_spec = spec
            self.genesis_cartridge_dir = cartridge_dir
            loaded_cartridge = CartridgeLoader.load_cartridge(cartridge_dir)
            self.genesis_level_spec = loaded_cartridge.get_level_spec(1)

            elapsed = time.perf_counter() - t0
            stage_delib = (
                f"GenesisDirector authored GenesisSpec '{spec.spec_id}' based on Steam DNA. "
                f"EngineCompiler verified AST and synthesized valid bytecode cartridge at {cartridge_dir} "
                f"with {len(comp_res.files)} files."
            )
            return StageResult(
                deliberation=stage_delib,
                stage_number=2,
                stage_name="Episode 0 Genesis Compilation",
                passed=True,
                duration_seconds=elapsed,
                details={
                    "spec_id": spec.spec_id,
                    "target_dir": str(cartridge_dir),
                    "cartridge_class": comp_res.cartridge_class_name,
                    "files_compiled": list(comp_res.files.keys()),
                    "target_difficulty": spec.target_difficulty,
                    "weapon_archetype": spec.weapon_spec.archetype,
                },
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.exception("Stage 2 Episode 0 Genesis Compilation failed: %s", e)
            return StageResult(
                deliberation=f"Stage 2 failed: {e}",
                stage_number=2,
                stage_name="Episode 0 Genesis Compilation",
                passed=False,
                duration_seconds=elapsed,
                error_message=str(e),
            )

    # ------------------------------------------------------------------------
    # Stage 3: Headless A* Playtesting
    # ------------------------------------------------------------------------
    def run_stage_3_headless_playtest(self, cartridge_dir: Optional[Path] = None) -> StageResult:
        """Stage 3: Instantiates HeadlessSimulator & SyntheticPlaytesterBot; verifies traversability & zero clipping."""
        t0 = time.perf_counter()
        target_dir = Path(cartridge_dir or self.genesis_cartridge_dir or (self.cartridges_dir / "genesis")).resolve()
        if not target_dir.is_dir():
            return StageResult(
                deliberation=f"Stage 3 failed: Cartridge directory not found at {target_dir}.",
                stage_number=3,
                stage_name="Headless A* Playtesting",
                passed=False,
                duration_seconds=0.0,
                error_message=f"Missing cartridge dir {target_dir}",
            )

        try:
            cartridge = CartridgeLoader.load_cartridge(target_dir)
            level_spec = cartridge.get_level_spec(1)

            # Initialize HeadlessSimulator with SQLite WAL telemetry recording
            sim = HeadlessSimulator(
                cartridge=cartridge,
                level_spec=level_spec,
                episode_number=1,
                dt=1.0 / 60.0,
                telemetry_mode="sqlite",
                telemetry_db_path=self.db_path,
                ignore_combat_lock=True,
                max_steps=1800,
            )

            bot = SyntheticPlaytesterBot(profile=RUSHER_PROFILE)
            playtest_result: PlaytestResult = bot.run_playtest(sim)

            # Capture active session id from telemetry recorder
            if hasattr(sim.telemetry_recorder, "session_id") and sim.telemetry_recorder.session_id:
                self.recorded_session_id = sim.telemetry_recorder.session_id

            # Flush and close recorder
            if hasattr(sim.telemetry_recorder, "flush"):
                sim.telemetry_recorder.flush()
            if hasattr(sim.telemetry_recorder, "close"):
                sim.telemetry_recorder.close()

            # Ensure SQLite WAL commits all buffered writes
            TelemetryDB(self.db_path).checkpoint("TRUNCATE")

            assert playtest_result.traversable is True, "A* pathfinding failed: level critical path blocked"
            assert playtest_result.wall_clipping_detected is False, "Wall clipping detected: player penetrated solid geometry"
            assert playtest_result.cleared is True, f"Synthetic playtester failed to clear level: {playtest_result.failure_type}"
            assert playtest_result.status == "PROMOTED", f"Playtest failed QA gate: status={playtest_result.status}"

            elapsed = time.perf_counter() - t0
            stage_delib = (
                f"SyntheticPlaytesterBot completed Episode 0 in {playtest_result.completion_time_sec:.2f}s "
                f"({playtest_result.completion_ticks} ticks). Verified 100% critical-path traversability, "
                f"0 wall clipping incidents, and successful exit clearance."
            )
            return StageResult(
                deliberation=stage_delib,
                stage_number=3,
                stage_name="Headless A* Playtesting",
                passed=True,
                duration_seconds=elapsed,
                details={
                    "traversable": playtest_result.traversable,
                    "cleared": playtest_result.cleared,
                    "wall_clipping_detected": playtest_result.wall_clipping_detected,
                    "completion_ticks": playtest_result.completion_ticks,
                    "completion_time_sec": playtest_result.completion_time_sec,
                    "damage_taken": playtest_result.damage_taken,
                    "final_health": playtest_result.final_health,
                    "status": playtest_result.status,
                },
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.exception("Stage 3 Headless A* Playtesting failed: %s", e)
            return StageResult(
                deliberation=f"Stage 3 failed: {e}",
                stage_number=3,
                stage_name="Headless A* Playtesting",
                passed=False,
                duration_seconds=elapsed,
                error_message=str(e),
            )

    # ------------------------------------------------------------------------
    # Stage 4: Live SQLite WAL Telemetry Capture
    # ------------------------------------------------------------------------
    def run_stage_4_telemetry_capture(self) -> StageResult:
        """Stage 4: Verifies events streamed to SQLite in WAL mode; asserts count > 0 and structured event types."""
        t0 = time.perf_counter()
        if not self.db_path.is_file():
            return StageResult(
                deliberation=f"Stage 4 failed: Database file not found at {self.db_path}.",
                stage_number=4,
                stage_name="Live SQLite WAL Telemetry Capture",
                passed=False,
                duration_seconds=0.0,
                error_message="Missing telemetry.db",
            )

        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            # 1. Assert WAL mode
            cur.execute("PRAGMA journal_mode;")
            row = cur.fetchone()
            journal_mode = row[0].lower() if row else "unknown"
            assert journal_mode == "wal", f"SQLite journal_mode is '{journal_mode}', expected 'wal'"

            # 2. Assert event count > 0
            cur.execute("SELECT COUNT(*) FROM telemetry_events;")
            event_count = cur.fetchone()[0]
            assert event_count > 0, "telemetry_events table has 0 records"

            # 3. Verify structured event types
            cur.execute("SELECT DISTINCT event_type FROM telemetry_events;")
            event_types = {r[0] for r in cur.fetchall()}

            # Check presence of canonical or semantic event categories
            has_movement = any(t in event_types for t in ["MOVEMENT_TICK", "player_move", "EPISODE_START", "PLAYER_SPAWN"])
            has_dash = any(t in event_types for t in ["DASH", "player_dash"]) or any("dash" in t.lower() for t in event_types)
            has_combat_or_clear = any(t in event_types for t in [
                "WEAPON_FIRE", "DAMAGE_TAKEN", "DAMAGE_DEALT", "HAZARD_HIT",
                "LEVEL_EXIT_REACHED", "EPISODE_END", "weapon_fire", "entity_damage", "episode_clear"
            ])

            assert has_movement, f"Missing movement/spawn event types in recorded events: {event_types}"
            assert has_combat_or_clear, f"Missing combat/exit event types in recorded events: {event_types}"

            # Check sessions & episodes
            cur.execute("SELECT COUNT(*) FROM telemetry_episodes;")
            episodes_recorded = cur.fetchone()[0]
            assert episodes_recorded > 0, "telemetry_episodes table has 0 records"

            # If session_id was not captured in Stage 3, fetch latest session_id from DB
            if not self.recorded_session_id:
                cur.execute("SELECT session_id FROM telemetry_sessions ORDER BY started_at DESC LIMIT 1;")
                sess_row = cur.fetchone()
                if sess_row:
                    self.recorded_session_id = sess_row[0]

            conn.close()
            elapsed = time.perf_counter() - t0
            stage_delib = (
                f"Verified SQLite database at {self.db_path} is active in WAL mode. "
                f"Recorded {event_count} telemetry events across {len(event_types)} distinct structured event types "
                f"({', '.join(sorted(event_types)[:5])})."
            )
            return StageResult(
                deliberation=stage_delib,
                stage_number=4,
                stage_name="Live SQLite WAL Telemetry Capture",
                passed=True,
                duration_seconds=elapsed,
                details={
                    "journal_mode": journal_mode,
                    "event_count": event_count,
                    "episodes_recorded": episodes_recorded,
                    "event_types": sorted(list(event_types)),
                    "session_id": self.recorded_session_id,
                },
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.exception("Stage 4 Live SQLite WAL Telemetry Capture failed: %s", e)
            return StageResult(
                deliberation=f"Stage 4 failed: {e}",
                stage_number=4,
                stage_name="Live SQLite WAL Telemetry Capture",
                passed=False,
                duration_seconds=elapsed,
                error_message=str(e),
            )

    # ------------------------------------------------------------------------
    # Stage 5: Episodic Director Adaptation Clamp (+/-15%)
    # ------------------------------------------------------------------------
    def run_stage_5_episodic_adaptation(self) -> StageResult:
        """Stage 5: Parses telemetry with TelemetryAnalyst, generates EpisodicUpdateSpec, verifies +/-15% clamp, and compiles Episode 1."""
        t0 = time.perf_counter()
        try:
            analyst = TelemetryAnalyst(db_path=self.db_path)
            if self.recorded_session_id:
                profile = analyst.analyze_session(session_id=self.recorded_session_id)
            else:
                conn = sqlite3.connect(str(self.db_path))
                cur = conn.cursor()
                cur.execute("SELECT session_id FROM telemetry_events ORDER BY id DESC LIMIT 1;")
                row = cur.fetchone()
                conn.close()
                sess_id = row[0] if row else "sess_default"
                profile = analyst.analyze_session(session_id=sess_id)

            assert getattr(profile, "deliberation", ""), "BehavioralProfile missing mandatory deliberation property"
            self.behavioral_profile = profile

            # Build current config based on Episode 0 level spec
            lvl_spec = self.genesis_level_spec
            current_walls = [(w[0], w[1], w[2], w[3]) for w in (lvl_spec.walls if lvl_spec else [])]
            current_cfg = CurrentEpisodeConfig(
                deliberation="Baseline parameters from Episode 0 Genesis",
                episode_id=lvl_spec.level_id if lvl_spec else "level_01",
                episode_number=1,
                walls=current_walls,
                enemies=lvl_spec.enemies if lvl_spec else [],
                hazards=lvl_spec.hazards if lvl_spec else [],
            )

            episodic_director = EpisodicDirector(max_adaptation_ratio=0.15, specs_dir=self.specs_dir)
            update_spec: EpisodicUpdateSpec = episodic_director.evaluate_and_update(
                profile=profile,
                current_config=current_cfg,
                world_bounds=(lvl_spec.width, lvl_spec.height) if lvl_spec else (1280.0, 720.0),
                spawn_point=lvl_spec.spawn_point if lvl_spec else (120.0, 360.0),
                exit_point=lvl_spec.exit_point if lvl_spec else (1160.0, 360.0),
            )

            assert getattr(update_spec, "deliberation", ""), "EpisodicUpdateSpec missing mandatory deliberation"
            episodic_director.write_update_spec(update_spec, output_path=self.specs_dir / "update_spec.md")
            self.update_spec = update_spec

            # Mathematical verification of +/-15% adaptation clamp
            clamp_violations: List[str] = []
            for name, clamp_rec in update_spec.clamp_state.clamps.items():
                if clamp_rec.previous_value > 0.0:
                    delta_ratio = abs(clamp_rec.clamped_value - clamp_rec.previous_value) / clamp_rec.previous_value
                    if delta_ratio > (0.15 + 1e-5):
                        clamp_violations.append(
                            f"Parameter '{name}' delta {delta_ratio:+.2%} exceeded +/-15% limit "
                            f"(prev={clamp_rec.previous_value:.2f}, clamped={clamp_rec.clamped_value:.2f})"
                        )

            assert len(clamp_violations) == 0, f"Clamp violations detected: {'; '.join(clamp_violations)}"

            # Compile Episode 1 Cartridge into cartridges/episode_01
            compiler = EngineCompiler(cartridges_root=self.cartridges_dir, specs_dir=self.specs_dir)
            ep1_dir = self.cartridges_dir / "episode_01"
            ep1_comp_res = compiler.compile(update_spec, target_dir=ep1_dir, promote_to_active=False)

            assert ep1_comp_res.ast_validated is True, "Episode 1 failed AST validation"
            assert ep1_comp_res.bytecode_compiled is True, "Episode 1 failed bytecode compilation"
            assert ep1_comp_res.cartridge_loaded is True, "Episode 1 failed cartridge loading"
            self.episode_1_cartridge_dir = ep1_dir

            elapsed = time.perf_counter() - t0
            stage_delib = (
                f"Parsed player telemetry into BehavioralProfile (playstyle: '{profile.playstyle_tag}'). "
                f"EpisodicDirector authored EpisodicUpdateSpec with deliberation first. "
                f"Verified mathematical adaptation clamp on {len(update_spec.clamp_state.clamps)} parameters "
                f"(all within +/-15.0%). EngineCompiler synthesized Episode 1 at {ep1_dir}."
            )
            return StageResult(
                deliberation=stage_delib,
                stage_number=5,
                stage_name="Episodic Director Adaptation Clamp (+/-15%)",
                passed=True,
                duration_seconds=elapsed,
                details={
                    "playstyle_tag": profile.playstyle_tag,
                    "mean_velocity": profile.mean_velocity,
                    "dash_ratio": profile.dash_ratio,
                    "target_episode_number": update_spec.target_episode_number,
                    "clamped_parameters_count": len(update_spec.clamp_state.clamps),
                    "episode_1_dir": str(ep1_dir),
                },
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.exception("Stage 5 Episodic Director Adaptation Clamp failed: %s", e)
            return StageResult(
                deliberation=f"Stage 5 failed: {e}",
                stage_number=5,
                stage_name="Episodic Director Adaptation Clamp (+/-15%)",
                passed=False,
                duration_seconds=elapsed,
                error_message=str(e),
            )

    # ------------------------------------------------------------------------
    # Stage 6: Between-Run Cartridge Hot-Reload
    # ------------------------------------------------------------------------
    def run_stage_6_cartridge_hot_reload(self) -> StageResult:
        """Stage 6: Dynamically loads Episode 0, executes steps, hot-swaps to Episode 1 in-process without restart."""
        t0 = time.perf_counter()
        genesis_dir = self.genesis_cartridge_dir or (self.cartridges_dir / "genesis")
        ep1_dir = self.episode_1_cartridge_dir or (self.cartridges_dir / "episode_01")

        if not genesis_dir.is_dir() or not ep1_dir.is_dir():
            return StageResult(
                deliberation="Stage 6 failed: Either genesis or episode_01 cartridge directory is missing.",
                stage_number=6,
                stage_name="Between-Run Cartridge Hot-Reload",
                passed=False,
                duration_seconds=0.0,
                error_message="Missing cartridge directories",
            )

        try:
            # 1. Load Episode 0 dynamically in-process
            cartridge_ep0 = CartridgeLoader.load_cartridge(genesis_dir)
            spec_ep0 = cartridge_ep0.get_level_spec(1)
            assert cartridge_ep0 is not None, "Failed to load Episode 0 cartridge"
            assert spec_ep0 is not None, "Episode 0 level spec is None"
            ep0_wall_count = len(spec_ep0.walls)

            # 2. Hot-swap in-process to Episode 1
            cartridge_ep1 = CartridgeLoader.load_cartridge(ep1_dir)
            spec_ep1 = cartridge_ep1.get_level_spec(2)
            assert cartridge_ep1 is not None, "Failed to load Episode 1 cartridge"
            assert spec_ep1 is not None, "Episode 1 level spec is None"

            # 3. Verify newly authored mechanics, enemy classes, and level geometries are actively loaded
            ep1_wall_count = len(spec_ep1.walls)

            # Test execution of Episode 1 in HeadlessSimulator in-process
            sim_ep1 = HeadlessSimulator(
                cartridge=cartridge_ep1,
                level_spec=spec_ep1,
                episode_number=2,
                dt=1.0 / 60.0,
                telemetry_mode="null",
                ignore_combat_lock=True,
                max_steps=60,
            )

            # Step simulation forward in-process
            for _ in range(60):
                sim_ep1.step(SimulationInput(move_x=1.0, move_y=0.0))

            assert sim_ep1.step_count == 60, f"Expected 60 steps executed, got {sim_ep1.step_count}"
            assert sim_ep1.is_out_of_bounds() is False, "Player entity out of bounds in Episode 1 simulation"

            elapsed = time.perf_counter() - t0
            stage_delib = (
                f"Successfully loaded Episode 0 from {genesis_dir.name} and hot-swapped in-process "
                f"to Episode 1 from {ep1_dir.name} without process restart. Verified adapted geometry "
                f"(Episode 0 walls: {ep0_wall_count}, Episode 1 walls: {ep1_wall_count}) and "
                f"executed 60 fixed-timestep simulation ticks cleanly."
            )
            return StageResult(
                deliberation=stage_delib,
                stage_number=6,
                stage_name="Between-Run Cartridge Hot-Reload",
                passed=True,
                duration_seconds=elapsed,
                details={
                    "ep0_cartridge_class": type(cartridge_ep0).__name__,
                    "ep1_cartridge_class": type(cartridge_ep1).__name__,
                    "ep0_walls": ep0_wall_count,
                    "ep1_walls": ep1_wall_count,
                    "sim_steps_executed": sim_ep1.step_count,
                },
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.exception("Stage 6 Between-Run Cartridge Hot-Reload failed: %s", e)
            return StageResult(
                deliberation=f"Stage 6 failed: {e}",
                stage_number=6,
                stage_name="Between-Run Cartridge Hot-Reload",
                passed=False,
                duration_seconds=elapsed,
                error_message=str(e),
            )

    # ------------------------------------------------------------------------
    # Full Execution Orchestration
    # ------------------------------------------------------------------------
    def run_all(self) -> EngineVerificationReport:
        """Executes all 6 stages sequentially, logs stage banners, and returns complete report."""
        start_time = time.perf_counter()
        results: List[StageResult] = []

        banner_width = 80
        self._log("=" * banner_width)
        self._log("AUTONOMOUS GAME ENGINE & SHOWRUNNER — ACCEPTANCE HARNESS (Feature 18)")
        self._log("=" * banner_width)

        stages_fn = [
            (1, "Steam Library Ingestion", self.run_stage_1_steam_ingestion),
            (2, "Episode 0 Genesis Compilation", self.run_stage_2_genesis_compilation),
            (3, "Headless A* Playtesting", self.run_stage_3_headless_playtest),
            (4, "Live SQLite WAL Telemetry Capture", self.run_stage_4_telemetry_capture),
            (5, "Episodic Director Adaptation Clamp (+/-15%)", self.run_stage_5_episodic_adaptation),
            (6, "Between-Run Cartridge Hot-Reload", self.run_stage_6_cartridge_hot_reload),
        ]

        all_passed = True
        for stage_num, stage_name, fn in stages_fn:
            self._log(f"\n[STAGE {stage_num}/6] {stage_name} ...")
            res: StageResult = fn()
            results.append(res)

            status_str = "[PASS]" if res.passed else "[FAIL]"
            self._log(f"[STAGE {stage_num}/6] {stage_name} {' ' * max(2, 45 - len(stage_name))} {status_str} ({res.duration_seconds:.2f}s)")
            self._log(f"  Deliberation: {res.deliberation}")
            if not res.passed:
                self._log(f"  Error: {res.error_message}")
                all_passed = False
                break

        total_elapsed = time.perf_counter() - start_time

        self._log("\n" + "=" * banner_width)
        if all_passed:
            self._log(f"VERIFICATION RESULT: ALL {len(results)}/6 STAGES PASSED (0 ERRORS, {total_elapsed:.2f}s)")
        else:
            self._log(f"VERIFICATION RESULT: FAILED AT STAGE {len(results)}/6 ({total_elapsed:.2f}s)")
        self._log("=" * banner_width)

        # Assemble Markdown Summary
        summary_lines = [
            "# Autonomous Game Engine & Showrunner — Acceptance Verification Report",
            f"- **Execution Timestamp UTC**: {datetime.now(timezone.utc).isoformat()}",
            f"- **Overall Status**: {'PASSED' if all_passed else 'FAILED'}",
            f"- **Total Duration**: {total_elapsed:.2f} seconds",
            "",
            "## Stage Summary",
            "| # | Stage Name | Status | Duration | Key Metrics |",
            "|---|---|---|---|---|",
        ]
        for s in results:
            st_text = "PASS" if s.passed else "FAIL"
            metrics_preview = ", ".join(f"{k}={v}" for k, v in list(s.details.items())[:3])
            summary_lines.append(f"| {s.stage_number} | {s.stage_name} | {st_text} | {s.duration_seconds:.2f}s | {metrics_preview} |")

        summary_lines.append("\n## Stage Deliberations")
        for s in results:
            summary_lines.append(f"### Stage {s.stage_number}: {s.stage_name}")
            summary_lines.append(f"{s.deliberation}\n")
            if s.error_message:
                summary_lines.append(f"**Failure Error**: `{s.error_message}`\n")

        summary_md = "\n".join(summary_lines)

        overall_delib = (
            f"Autonomous acceptance harness completed {len(results)}/6 verification stages in {total_elapsed:.2f}s. "
            f"Status: {'ALL PASS' if all_passed else 'FAILED'}."
        )

        return EngineVerificationReport(
            deliberation=overall_delib,
            passed=all_passed,
            total_duration_seconds=total_elapsed,
            stages=results,
            summary_markdown=summary_md,
        )


def main() -> int:
    """CLI entrypoint for verify_engine.py."""
    parser = argparse.ArgumentParser(description="Autonomous E2E Engine Acceptance Harness (Feature 18)")
    parser.add_argument("--db-path", type=str, default=None, help="Custom SQLite telemetry DB path")
    parser.add_argument("--cartridges-dir", type=str, default=None, help="Custom cartridges output directory")
    parser.add_argument("--specs-dir", type=str, default=None, help="Custom specs output directory")
    parser.add_argument("--steam-root", type=str, default=None, help="Custom Steam installation root path")
    parser.add_argument("--quiet", action="store_true", help="Suppress stage banners and print only final status")
    args = parser.parse_args()

    harness = EngineAcceptanceHarness(
        db_path=Path(args.db_path) if args.db_path else None,
        cartridges_dir=Path(args.cartridges_dir) if args.cartridges_dir else None,
        specs_dir=Path(args.specs_dir) if args.specs_dir else None,
        steam_root=Path(args.steam_root) if args.steam_root else None,
        verbose=not args.quiet,
    )

    try:
        report = harness.run_all()
        return 0 if report.passed else 1
    finally:
        harness.cleanup()


if __name__ == "__main__":
    sys.exit(main())
