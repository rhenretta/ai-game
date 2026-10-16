"""tests/test_m5_supervisor_stress.py
Empirical Challenger Stress & Adversarial Test Suite for Build Supervisor & Closed-Loop Code Self-Healing Guardrail (Feature 16).

Verifies under hostile injections and extreme conditions:
  1. Hostile & Defective Candidate Injections:
     - Wall completely sealing exit portal rejected at Gate B without touching active cartridge directory.
     - Spawn coordinate embedded inside solid obstacle rejected at Gate B.
     - Lethal hazard placed directly on player spawn clearance rejected at Gate C.
     - Premature player death (< 3.0s / 180 ticks) and excessive clearance time rejected at Gate C.
     - Hostile AST imports (os, subprocess, socket, shutil, sys) rejected at Gate A.
     - Hostile AST calls (eval, exec, __import__) rejected at Gate A.
     - Malformed Python syntax rejected at Gate A.
  2. Closed-Loop Self-Healing Verification:
     - Injected blocked-path candidate successfully healed via heal_specification on attempt 2 and promoted.
     - Injected lethal spawn hazard candidate healed (hazard pushed away) and promoted on attempt 2.
     - Blocked nodes tile coordinate matching correctly triggers obstacle removal.
     - GenesisSpec parameter and pillar self-healing.
  3. Max Retries Exhaustion:
     - Unhealable candidate (exit inside outer perimeter wall) exhausts max_retries=3 with FAILED_MAX_RETRIES.
     - Unhealable compiler crash exhausts max_retries without active promotion.
     - Custom max_retries limits (1, 2, 4) strictly respected.
  4. Staging Directory Isolation & Concurrency Safety:
     - Candidate staging folders (candidate_epXX_attY) remain completely isolated and preserved on disk.
     - Active cartridge directory receives zero pollution on failure.
     - Existing healthy active cartridge files are preserved untouched on candidate rejection.
     - Atomic promotion leaves zero dangling temporary files.
     - Multi-episode staging isolation and concurrent supervisor independence.
  5. Epistemic Rationale-First Architecture & Schema Invariants:
     - All supervisor models, traces, and reports declare deliberation strictly first.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import pytest

from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.compiler.compiler import CompilerError, EngineCompiler
from engine.director.episodic import CurrentEpisodeConfig, EpisodicDirector
from engine.director.genesis import GenesisDirector
from engine.director.models import (
    CurrentEpisodeConfig,
    DirectiveAction,
    EpisodicUpdateSpec,
    GenesisSpec,
    SpatialUpdateDirective,
)
from engine.loader import CartridgeLoader, GenericCartridge
from engine.supervisor.models import (
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    RemedyType,
    SupervisorEvaluationReport,
    SupervisorGateVerdict,
)
from engine.supervisor.supervisor import BuildSupervisor, SupervisorError
from tests.test_director_episodic import MockBehavioralProfile
from tests.test_director_genesis import rusher_dna
from tests.test_tier1_units import validate_rationale_first_ast, validate_rationale_first_dict


@pytest.fixture
def supervisor(tmp_path: Path) -> BuildSupervisor:
    """Creates a hermetic BuildSupervisor instance pointing to temporary test directories."""
    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    compiler = EngineCompiler(
        cartridges_root=cartridges_root,
        active_cartridge_dir=active_dir,
        specs_dir=tmp_path / "specs",
    )
    return BuildSupervisor(
        cartridges_root=cartridges_root,
        active_cartridge_dir=active_dir,
        compiler=compiler,
        tile_size=32.0,
        max_retries=3,
        early_death_tick_threshold=180,
        max_simulation_ticks=3600,
        spawn_clearance_margin=64.0,
    )


# ============================================================================
# 1. Hostile & Defective Candidate Injections
# ============================================================================

class TestHostileDefectiveCandidateInjections:
    """Stress tests verifying strict rejection of hostile, defective, or unsafe candidates."""
    deliberation: str = "Test suite evaluating hostile candidate injections across Gates A, B, C, D."

    def test_hostile_sealed_exit_portal_gate_b_rejection(self, supervisor: BuildSupervisor):
        """Injects a solid barrier completely sealing the exit portal.
        Verifies Gate B rejects candidate and cartridges/active/ remains empty.
        """
        director = EpisodicDirector()
        spec = director.evaluate_and_update(MockBehavioralProfile())

        # Inject an impassable wall barrier encasing the exit portal (1160, 360)
        spec.updated_episode_config.walls.append((1100.0, 300.0, 100.0, 100.0))

        candidate_dir = (supervisor.cartridges_root / "candidate_hostile_sealed_exit").resolve()
        supervisor.compiler.compile(spec, target_dir=candidate_dir, promote_to_active=False)

        gate_results, rejection_trace = supervisor.evaluate_candidate_cartridge(
            candidate_dir=candidate_dir,
            spec=spec,
            episode_number=1,
        )

        # Gate B must fail
        assert gate_results["gate_b"].passed is False
        assert gate_results["gate_b"].failure_type == GateFailureType.CRITICAL_PATH_BLOCKED
        assert (
            "blocked by solid obstacles" in gate_results["gate_b"].details or
            "overlaps solid obstacle" in gate_results["gate_b"].details
        )

        # Diagnostic trace must capture exit blockage
        assert rejection_trace is not None
        assert rejection_trace.failure_gate == "Gate B: Traversability"
        assert rejection_trace.failure_type == GateFailureType.CRITICAL_PATH_BLOCKED

        # Active directory must remain completely empty (zero files copied)
        assert not supervisor.active_cartridge_dir.exists() or len(list(supervisor.active_cartridge_dir.iterdir())) == 0

    def test_hostile_spawn_inside_solid_wall_gate_b_rejection(self, supervisor: BuildSupervisor):
        """Injects a solid obstacle directly overlapping the player spawn coordinate.
        Verifies Gate B immediately detects the obstacle and blocks promotion.
        """
        director = EpisodicDirector()
        spec = director.evaluate_and_update(MockBehavioralProfile())

        # Spawn is at (120, 360). Place solid wall directly on spawn point
        spec.updated_episode_config.walls.append((100.0, 340.0, 64.0, 64.0))

        candidate_dir = (supervisor.cartridges_root / "candidate_spawn_overlap").resolve()
        supervisor.compiler.compile(spec, target_dir=candidate_dir, promote_to_active=False)

        gate_results, rejection_trace = supervisor.evaluate_candidate_cartridge(
            candidate_dir=candidate_dir,
            spec=spec,
            episode_number=1,
        )

        # Gate B must fail
        assert gate_results["gate_b"].passed is False
        assert gate_results["gate_b"].failure_type == GateFailureType.CRITICAL_PATH_BLOCKED

        # Direct geometric NavMesh check without simulation
        cartridge = CartridgeLoader.load_cartridge(candidate_dir)
        level_spec = cartridge.get_level_spec(1)
        res_geom, blocked_nodes = supervisor.verify_gate_b_traversability(cartridge, level_spec, sim_result=None)
        assert res_geom.passed is False
        assert res_geom.failure_type == GateFailureType.CRITICAL_PATH_BLOCKED
        assert "overlaps solid obstacle" in res_geom.details

        assert not supervisor.active_cartridge_dir.exists() or len(list(supervisor.active_cartridge_dir.iterdir())) == 0

    def test_hostile_lethal_hazard_on_spawn_point_gate_c_rejection(self, supervisor: BuildSupervisor):
        """Injects a lethal hazard directly within the 64px spawn clearance margin.
        Verifies Gate C fails with SURVIVABILITY_VIOLATION and active directory is untouched.
        """
        director = EpisodicDirector()
        spec = director.evaluate_and_update(MockBehavioralProfile())

        # Add hazard overlapping player spawn (120, 360)
        spec.updated_episode_config.hazards.append({
            "id": "lethal_spawn_pit",
            "type": "lava",
            "x": 120.0,
            "y": 360.0,
            "w": 48.0,
            "h": 48.0,
        })

        candidate_dir = (supervisor.cartridges_root / "candidate_spawn_hazard").resolve()
        supervisor.compiler.compile(spec, target_dir=candidate_dir, promote_to_active=False)

        gate_results, rejection_trace = supervisor.evaluate_candidate_cartridge(
            candidate_dir=candidate_dir,
            spec=spec,
            episode_number=1,
        )

        assert gate_results["gate_c"].passed is False
        assert gate_results["gate_c"].failure_type == GateFailureType.SURVIVABILITY_VIOLATION
        assert "overlaps spawn clearance margin" in gate_results["gate_c"].details

        assert rejection_trace is not None
        assert rejection_trace.failure_gate == "Gate C: Survivability"
        assert rejection_trace.remedy_type in (RemedyType.SOFTEN_HAZARD, RemedyType.WIDEN_MARGIN)
        assert not supervisor.active_cartridge_dir.exists() or len(list(supervisor.active_cartridge_dir.iterdir())) == 0

    def test_hostile_early_death_survivability_violation(self, supervisor: BuildSupervisor):
        """Verifies Gate C detects premature player death occurring under 3.0s (180 ticks)."""
        level_spec = BaseLevelSpec(
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            hazards=[],
            deliberation="Early death test spec",
        )
        sim_result = {
            "traversable": True,
            "cleared": False,
            "first_death_tick": 45,  # 0.75s into simulation
            "completion_time_sec": 0.75,
            "failure_coordinates": (140.0, 360.0),
        }

        gate_c = supervisor.verify_gate_c_survivability(sim_result, level_spec)
        assert gate_c.passed is False
        assert gate_c.failure_type == GateFailureType.SURVIVABILITY_VIOLATION
        assert "Player died prematurely at tick 45" in gate_c.deliberation
        assert gate_c.diagnostic_data["death_tick"] == 45

    def test_hostile_excessive_clearance_time_gate_c_rejection(self, supervisor: BuildSupervisor):
        """Verifies Gate C detects level completion exceeding maximum allowed simulation time."""
        level_spec = BaseLevelSpec(
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            hazards=[],
            deliberation="Timeout test spec",
        )
        sim_result = {
            "traversable": True,
            "cleared": True,
            "first_death_tick": None,
            "completion_time_sec": 75.0,  # Exceeds max 60.0s (3600 ticks)
        }

        gate_c = supervisor.verify_gate_c_survivability(sim_result, level_spec)
        assert gate_c.passed is False
        assert gate_c.failure_type == GateFailureType.SURVIVABILITY_VIOLATION
        assert "exceeding maximum allowed" in gate_c.deliberation

    def test_hostile_prohibited_ast_imports_gate_a_rejection(self, supervisor: BuildSupervisor, tmp_path: Path):
        """Injects forbidden system and network modules into cartridge candidate.
        Verifies Gate A rejects them all with SYNTAX_AST_ERROR and prevents promotion.
        """
        forbidden_imports = [
            ("import os\nos.system('echo 1')", "os"),
            ("import subprocess\nsubprocess.Popen(['ls'])", "subprocess"),
            ("import sys\nsys.exit(1)", "sys"),
            ("import socket\ns = socket.socket()", "socket"),
            ("import shutil\nshutil.rmtree('/')", "shutil"),
        ]

        for code_snippet, module_name in forbidden_imports:
            cart_dir = tmp_path / f"malicious_{module_name}"
            cart_dir.mkdir(parents=True, exist_ok=True)
            (cart_dir / "mechanics.py").write_text(code_snippet, encoding="utf-8")

            res = supervisor.verify_gate_a_syntax_and_ast(cart_dir)
            assert res.passed is False, f"Expected Gate A to reject banned module '{module_name}'"
            assert res.failure_type == GateFailureType.SYNTAX_AST_ERROR
            assert "AST Security Invariant Violated" in res.details

    def test_hostile_prohibited_ast_calls_gate_a_rejection(self, supervisor: BuildSupervisor, tmp_path: Path):
        """Injects prohibited execution primitives (eval, exec, __import__) into cartridge candidate."""
        prohibited_calls = [
            "result = eval('1 + 1')",
            "exec('x = 42')",
            "mod = __import__('math')",
        ]

        for call_code in prohibited_calls:
            cart_dir = tmp_path / f"prohibited_call_{abs(hash(call_code))}"
            cart_dir.mkdir(parents=True, exist_ok=True)
            (cart_dir / "mechanics.py").write_text(
                f"def malicious_hook():\n    {call_code}\n",
                encoding="utf-8"
            )

            res = supervisor.verify_gate_a_syntax_and_ast(cart_dir)
            assert res.passed is False
            assert res.failure_type == GateFailureType.SYNTAX_AST_ERROR
            assert "AST Security Invariant Violated" in res.details

    def test_hostile_raw_syntax_error_gate_a_rejection(self, supervisor: BuildSupervisor, tmp_path: Path):
        """Injects unparseable Python syntax into candidate cartridge.
        Verifies Gate A catches SyntaxError and returns SYNTAX_AST_ERROR.
        """
        cart_dir = tmp_path / "syntax_error_cart"
        cart_dir.mkdir(parents=True, exist_ok=True)
        (cart_dir / "mechanics.py").write_text("class BrokenCandidate(:\n    def broken(\n", encoding="utf-8")

        res = supervisor.verify_gate_a_syntax_and_ast(cart_dir)
        assert res.passed is False
        assert res.failure_type == GateFailureType.SYNTAX_AST_ERROR
        assert "Python syntax error" in res.details

    def test_hostile_missing_python_modules_gate_a_rejection(self, supervisor: BuildSupervisor, tmp_path: Path):
        """Candidate directory lacking Python source files must fail Gate A immediately."""
        cart_dir = tmp_path / "empty_cart"
        cart_dir.mkdir(parents=True, exist_ok=True)

        res = supervisor.verify_gate_a_syntax_and_ast(cart_dir)
        assert res.passed is False
        assert res.failure_type == GateFailureType.SYNTAX_AST_ERROR
        assert "Missing essential Python modules" in res.details


# ============================================================================
# 2. Closed-Loop Self-Healing Verification
# ============================================================================

class TestClosedLoopSelfHealingStress:
    """Stress tests verifying closed-loop repair and promotion across multiple defect types."""
    deliberation: str = "Test suite verifying closed-loop self-healing repair workflows."

    def test_closed_loop_heals_blocked_corridor_on_attempt_2(self, supervisor: BuildSupervisor):
        """Injects an impassable interior wall barrier at x=640.
        Verifies attempt 1 fails Gate B, self-healing removes the wall, and attempt 2 is promoted cleanly.
        """
        director = EpisodicDirector()
        spec = director.evaluate_and_update(MockBehavioralProfile())

        # Inject blocking interior wall
        barrier = (640.0, 32.0, 48.0, 656.0)
        spec.updated_episode_config.walls.append(barrier)

        assert not supervisor.active_cartridge_dir.exists() or len(list(supervisor.active_cartridge_dir.iterdir())) == 0

        report = supervisor.evaluate_and_heal(spec, max_retries=3)

        # Verification of closed-loop repair outcome
        assert report.status == SupervisorGateVerdict.PROMOTED
        assert report.promoted_to_active is True
        assert report.attempt_count == 2
        assert len(report.repair_history) == 1
        assert report.repair_history[0].failure_type == GateFailureType.CRITICAL_PATH_BLOCKED
        assert report.rejection_trace is None

        # Verify active directory has promoted files
        assert (supervisor.active_cartridge_dir / "manifest.json").is_file()
        assert (supervisor.active_cartridge_dir / "level.json").is_file()
        assert (supervisor.active_cartridge_dir / "mechanics.py").is_file()
        assert (supervisor.active_cartridge_dir / "entities.py").is_file()

        # Verify that the active cartridge loads and level is traversable
        cartridge = CartridgeLoader.load_cartridge(supervisor.active_cartridge_dir)
        level_spec = cartridge.get_level_spec(1)
        res_b, blocked = supervisor.verify_gate_b_traversability(cartridge, level_spec)
        assert res_b.passed is True
        assert len(blocked) == 0

    def test_closed_loop_heals_lethal_spawn_hazard_on_attempt_2(self, supervisor: BuildSupervisor):
        """Injects a lethal hazard directly on player spawn.
        Verifies attempt 1 fails Gate C, self-healing relocates hazard away from spawn,
        and attempt 2 is promoted cleanly.
        """
        director = EpisodicDirector()
        spec = director.evaluate_and_update(MockBehavioralProfile())

        # Place hazard directly over spawn (120, 360)
        spawn_hazard = {
            "id": "lethal_spawn_spikes",
            "type": "spikes",
            "x": 120.0,
            "y": 360.0,
            "w": 48.0,
            "h": 48.0,
        }
        spec.updated_episode_config.hazards.append(spawn_hazard)

        report = supervisor.evaluate_and_heal(spec, max_retries=3)

        assert report.status == SupervisorGateVerdict.PROMOTED
        assert report.promoted_to_active is True
        assert report.attempt_count == 2
        assert len(report.repair_history) == 1
        assert report.repair_history[0].failure_type == GateFailureType.SURVIVABILITY_VIOLATION

        # Verify that promoted active cartridge has hazard relocated away from spawn
        cartridge = CartridgeLoader.load_cartridge(supervisor.active_cartridge_dir)
        level_spec = cartridge.get_level_spec(1)
        res_c = supervisor.verify_gate_c_survivability(None, level_spec)
        assert res_c.passed is True

        # Check relocated hazard coordinates
        matching_hazards = [h for h in level_spec.hazards if h.get("id") == "lethal_spawn_spikes"]
        assert len(matching_hazards) == 1
        assert matching_hazards[0]["x"] >= 340.0  # Pushed east by +240px

    def test_closed_loop_heals_interior_wall_via_blocked_nodes(self, supervisor: BuildSupervisor):
        """Verifies self-healing identifies and removes an interior wall via blocked_nodes coordinates."""
        director = EpisodicDirector()
        spec = director.evaluate_and_update(MockBehavioralProfile())

        # Inject interior wall at (500, 300, 48, 120)
        spec.updated_episode_config.walls.append((500.0, 300.0, 48.0, 120.0))

        # Create diagnostic trace referencing tile nodes inside that wall
        trace = RejectionDiagnosticTrace(
            deliberation="Navigation path blocked by obstacle node at (16, 10).",
            spec_id=spec.spec_id,
            episode_number=1,
            failure_gate="Gate B: Traversability",
            failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
            failure_reason="Critical path obstructed.",
            failure_coordinates=(512.0, 320.0),
            blocked_nodes=[(16, 10)],  # (16*32=512, 10*32=320)
            remedy_type=RemedyType.REMOVE_OBSTACLE,
            recommended_remedy="Remove interior wall at obstacle node.",
        )

        initial_wall_count = len(spec.updated_episode_config.walls)
        healed_spec = supervisor.heal_specification(spec, trace)

        assert isinstance(healed_spec, EpisodicUpdateSpec)
        # Verify the interior wall was removed
        assert len(healed_spec.updated_episode_config.walls) == initial_wall_count - 1
        assert "[SELF-HEALING REPAIR]" in healed_spec.deliberation

    def test_closed_loop_genesis_spec_pillar_and_damage_softening(self, supervisor: BuildSupervisor, rusher_dna):
        """Verifies self-healing on GenesisSpec reduces pillar count and scales down damage."""
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        spec.level_dimensions.pillar_count = 6
        initial_hazard_damage = spec.hazard_specs[0].damage

        # Simulate survivability violation trace
        trace = RejectionDiagnosticTrace(
            deliberation="Player died prematurely from high hazard damage.",
            spec_id=spec.spec_id,
            episode_number=0,
            failure_gate="Gate C: Survivability",
            failure_type=GateFailureType.SURVIVABILITY_VIOLATION,
            failure_reason="Lethal hazard contact in early phase.",
            remedy_type=RemedyType.SOFTEN_HAZARD,
            recommended_remedy="Soften hazard damage.",
        )

        healed = supervisor.heal_specification(spec, trace)
        assert isinstance(healed, GenesisSpec)
        assert healed.hazard_specs[0].damage < initial_hazard_damage
        assert "[SELF-HEALING REPAIR]" in healed.deliberation


# ============================================================================
# 3. Max Retries Exhaustion Stress
# ============================================================================

class TestMaxRetriesExhaustionStress:
    """Stress tests verifying strict termination and rejection upon exhausting retry limits."""
    deliberation: str = "Test suite verifying max retries exhaustion and promotion blocking."

    def test_unhealable_exit_inside_perimeter_wall_exhausts_max_retries(self, supervisor: BuildSupervisor, rusher_dna):
        """Constructs an unhealable candidate where exit portal is embedded inside an outer perimeter wall.
        Because perimeter walls cannot be removed, the supervisor must exhaust all 3 retries and fail
        with FAILED_MAX_RETRIES without promoting files to cartridges/active/.
        """
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)

        # Set exit point inside the top outer perimeter wall (0, 0, 1280, 32)
        spec.level_dimensions.exit_point = (16.0, 16.0)

        report = supervisor.evaluate_and_heal(spec, max_retries=3)

        assert report.status == SupervisorGateVerdict.FAILED_MAX_RETRIES
        assert report.promoted_to_active is False
        assert report.attempt_count == 3
        assert len(report.repair_history) == 3
        assert report.rejection_trace is not None
        assert report.rejection_trace.failure_gate == "Gate B: Traversability"

        # cartridges/active/ must remain strictly untouched
        assert not supervisor.active_cartridge_dir.exists() or len(list(supervisor.active_cartridge_dir.iterdir())) == 0

    def test_unhealable_compiler_failure_exhausts_max_retries(self, supervisor: BuildSupervisor, monkeypatch):
        """Forces the compiler to raise an unrecoverable exception on every attempt.
        Verifies the supervisor logs compiler failure traces and permanently blocks promotion at max_retries.
        """
        director = EpisodicDirector()
        spec = director.evaluate_and_update(MockBehavioralProfile())

        def crashing_compile(*args, **kwargs):
            raise CompilerError("Synthetic unrecoverable compilation failure")

        monkeypatch.setattr(supervisor.compiler, "compile", crashing_compile)

        report = supervisor.evaluate_and_heal(spec, max_retries=3)

        assert report.status == SupervisorGateVerdict.FAILED_MAX_RETRIES
        assert report.promoted_to_active is False
        assert report.attempt_count == 3
        assert len(report.repair_history) == 3
        assert report.gate_results["gate_a"].passed is False
        assert not supervisor.active_cartridge_dir.exists() or len(list(supervisor.active_cartridge_dir.iterdir())) == 0

    @pytest.mark.parametrize("retry_limit", [1, 2, 4])
    def test_custom_max_retries_parameter_strictly_respected(self, supervisor: BuildSupervisor, rusher_dna, retry_limit: int):
        """Verifies that custom max_retries limits (1, 2, 4) are strictly honored."""
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)

        # Embedded exit in perimeter wall ensures failure on every attempt
        spec.level_dimensions.exit_point = (16.0, 16.0)

        report = supervisor.evaluate_and_heal(spec, max_retries=retry_limit)

        assert report.status == SupervisorGateVerdict.FAILED_MAX_RETRIES
        assert report.promoted_to_active is False
        assert report.attempt_count == retry_limit
        assert len(report.repair_history) == retry_limit
        assert report.max_retries == retry_limit


# ============================================================================
# 4. Staging Directory Isolation & Concurrency Safety
# ============================================================================

class TestStagingDirectoryIsolationAndConcurrencyStress:
    """Stress tests verifying file isolation, non-pollution, and atomic promotion."""
    deliberation: str = "Test suite verifying candidate staging isolation and concurrency safety."

    def test_staging_directories_strictly_isolated_across_attempts(self, supervisor: BuildSupervisor):
        """Verifies that candidate directories for attempt 1 and attempt 2 exist as separate folders,
        and attempt 1 retains its original failing state on disk without being mutated by attempt 2.
        """
        director = EpisodicDirector()
        spec = director.evaluate_and_update(MockBehavioralProfile())

        # Inject blocking wall
        barrier = (640.0, 32.0, 48.0, 656.0)
        spec.updated_episode_config.walls.append(barrier)

        report = supervisor.evaluate_and_heal(spec, max_retries=3)
        assert report.status == SupervisorGateVerdict.PROMOTED
        assert report.attempt_count == 2

        ep = spec.target_episode_number
        att1_dir = supervisor.cartridges_root / f"candidate_ep{ep:02d}_att1"
        att2_dir = supervisor.cartridges_root / f"candidate_ep{ep:02d}_att2"

        # Both staging directories must exist independently
        assert att1_dir.is_dir()
        assert att2_dir.is_dir()
        assert att1_dir.resolve() != att2_dir.resolve()

        # att1 level.json must still contain the blocking barrier wall
        with open(att1_dir / "level.json", "r", encoding="utf-8") as f:
            att1_level = json.load(f)
        att1_walls = [tuple(w) for w in att1_level.get("walls", [])]
        assert barrier in att1_walls

        # att2 level.json must have the barrier wall removed
        with open(att2_dir / "level.json", "r", encoding="utf-8") as f:
            att2_level = json.load(f)
        att2_walls = [tuple(w) for w in att2_level.get("walls", [])]
        assert barrier not in att2_walls

    def test_failed_candidate_zero_leakage_to_active_directory(self, supervisor: BuildSupervisor, rusher_dna):
        """Verifies that an unhealable failing candidate leaves zero files in cartridges/active/."""
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        spec.level_dimensions.exit_point = (16.0, 16.0)

        report = supervisor.evaluate_and_heal(spec, max_retries=2)
        assert report.status == SupervisorGateVerdict.FAILED_MAX_RETRIES

        # Verify staging exists
        assert (supervisor.cartridges_root / "candidate_ep00_att1").exists()
        assert (supervisor.cartridges_root / "candidate_ep00_att2").exists()

        # Active dir must have zero promoted files
        if supervisor.active_cartridge_dir.exists():
            active_files = list(supervisor.active_cartridge_dir.iterdir())
            assert len(active_files) == 0

    def test_active_directory_preserves_existing_files_on_candidate_rejection(self, supervisor: BuildSupervisor, rusher_dna):
        """Pre-populates cartridges/active/ with a healthy cartridge.
        Verifies that when a subsequent candidate is rejected, the existing active cartridge
        remains completely unaltered and uncorrupted.
        """
        supervisor.active_cartridge_dir.mkdir(parents=True, exist_ok=True)
        sentinel_manifest = {"name": "HealthyActiveCartridge", "version": "1.0.0"}
        sentinel_code = "class ActiveCartridge: pass\n"

        (supervisor.active_cartridge_dir / "manifest.json").write_text(json.dumps(sentinel_manifest), encoding="utf-8")
        (supervisor.active_cartridge_dir / "mechanics.py").write_text(sentinel_code, encoding="utf-8")

        # Now evaluate an unhealable failing candidate
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        spec.level_dimensions.exit_point = (16.0, 16.0)

        report = supervisor.evaluate_and_heal(spec, max_retries=2)
        assert report.status == SupervisorGateVerdict.FAILED_MAX_RETRIES
        assert report.promoted_to_active is False

        # Verify baseline active files are bitwise intact
        active_manifest = json.loads((supervisor.active_cartridge_dir / "manifest.json").read_text(encoding="utf-8"))
        assert active_manifest["name"] == "HealthyActiveCartridge"
        assert active_manifest["version"] == "1.0.0"
        assert (supervisor.active_cartridge_dir / "mechanics.py").read_text(encoding="utf-8") == sentinel_code

    def test_atomic_promotion_cleans_temporary_files(self, supervisor: BuildSupervisor, tmp_path: Path):
        """Verifies that atomic promotion via promote_candidate leaves zero dangling .tmp files."""
        candidate_dir = tmp_path / "candidate_atomic"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        essential_files = ["manifest.json", "level.json", "entities.py", "mechanics.py", "envelope.json"]
        for fname in essential_files:
            (candidate_dir / fname).write_text(f'{{"file": "{fname}"}}', encoding="utf-8")

        supervisor.promote_candidate(candidate_dir)

        # All essential files must be present
        for fname in essential_files:
            assert (supervisor.active_cartridge_dir / fname).is_file()

        # Zero temporary files matching .*tmp
        dangling_tmp = [f.name for f in supervisor.active_cartridge_dir.iterdir() if "tmp" in f.name]
        assert len(dangling_tmp) == 0

    def test_multi_episode_candidate_staging_isolation(self, supervisor: BuildSupervisor):
        """Verifies that Episode 1 and Episode 2 candidate staging directories use disjoint paths."""
        director = EpisodicDirector()
        spec1 = director.evaluate_and_update(MockBehavioralProfile())
        spec1.target_episode_number = 1

        spec2 = director.evaluate_and_update(MockBehavioralProfile())
        spec2.target_episode_number = 2

        report1 = supervisor.evaluate_and_heal(spec1, max_retries=1)
        report2 = supervisor.evaluate_and_heal(spec2, max_retries=1)

        assert report1.status == SupervisorGateVerdict.PROMOTED
        assert report2.status == SupervisorGateVerdict.PROMOTED

        ep1_dir = supervisor.cartridges_root / "candidate_ep01_att1"
        ep2_dir = supervisor.cartridges_root / "candidate_ep02_att1"

        assert ep1_dir.is_dir()
        assert ep2_dir.is_dir()
        assert ep1_dir.resolve() != ep2_dir.resolve()

    def test_independent_supervisor_instances_concurrency(self, tmp_path: Path):
        """Verifies two separate BuildSupervisor instances with separate root paths operate independently."""
        root_a = tmp_path / "supervisor_a"
        root_b = tmp_path / "supervisor_b"

        sup_a = BuildSupervisor(cartridges_root=root_a / "cartridges", active_cartridge_dir=root_a / "active")
        sup_b = BuildSupervisor(cartridges_root=root_b / "cartridges", active_cartridge_dir=root_b / "active")

        director = EpisodicDirector()
        spec = director.evaluate_and_update(MockBehavioralProfile())

        rep_a = sup_a.evaluate_and_heal(spec, max_retries=1)
        rep_b = sup_b.evaluate_and_heal(spec, max_retries=1)

        assert rep_a.status == SupervisorGateVerdict.PROMOTED
        assert rep_b.status == SupervisorGateVerdict.PROMOTED
        assert (sup_a.active_cartridge_dir / "manifest.json").is_file()
        assert (sup_b.active_cartridge_dir / "manifest.json").is_file()
        assert sup_a.active_cartridge_dir.resolve() != sup_b.active_cartridge_dir.resolve()


# ============================================================================
# 5. Epistemic Rationale-First Architecture & Schema Invariants
# ============================================================================

class TestSupervisorEpistemicRationaleFirstStress:
    """Verifies that all models, traces, and reports strictly declare deliberation first."""
    deliberation: str = "Test suite verifying Rationale-First epistemic ordering on all supervisor outputs."

    def test_quality_gate_result_deliberation_first(self):
        """QualityGateResult dictionary serialization must declare deliberation as key index 0."""
        gate = QualityGateResult(
            deliberation="Gate B verified unbroken A* path.",
            gate_name="Gate B",
            passed=True,
        )
        d = gate.to_dict()
        assert validate_rationale_first_dict(d) is True
        assert list(d.keys())[0] == "deliberation"

    def test_rejection_trace_deliberation_first(self):
        """RejectionDiagnosticTrace dictionary serialization must declare deliberation as key index 0."""
        trace = RejectionDiagnosticTrace(
            deliberation="Obstacle blocks path between spawn and portal.",
            spec_id="test_spec",
            episode_number=1,
            failure_gate="Gate B: Traversability",
            failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
            failure_reason="Corridor obstruction.",
        )
        d = trace.to_dict()
        assert validate_rationale_first_dict(d) is True
        assert list(d.keys())[0] == "deliberation"

    def test_evaluation_report_deliberation_first(self):
        """SupervisorEvaluationReport dictionary serialization must declare deliberation as key index 0."""
        report = SupervisorEvaluationReport(
            deliberation="Candidate build passed all gates and was promoted.",
            spec_id="test_spec",
            episode_number=1,
            status=SupervisorGateVerdict.PROMOTED,
            promoted_to_active=True,
        )
        d = report.to_dict()
        assert validate_rationale_first_dict(d) is True
        assert list(d.keys())[0] == "deliberation"

    def test_healed_spec_prepends_repair_rationale(self, supervisor: BuildSupervisor):
        """heal_specification must prepend the repair rationale to specification deliberation."""
        director = EpisodicDirector()
        spec = director.evaluate_and_update(MockBehavioralProfile())
        original_delib = spec.deliberation

        trace = RejectionDiagnosticTrace(
            deliberation="Blocked corridor at (640, 360).",
            spec_id=spec.spec_id,
            episode_number=1,
            failure_gate="Gate B: Traversability",
            failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
            failure_reason="Obstacle block.",
            remedy_type=RemedyType.REMOVE_OBSTACLE,
            recommended_remedy="Remove blocking wall at x=640.",
        )

        healed = supervisor.heal_specification(spec, trace)
        assert healed.deliberation.startswith("[SELF-HEALING REPAIR]")
        assert original_delib in healed.deliberation

    def test_supervisor_models_ast_rationale_first(self):
        """Inspects engine/supervisor/models.py AST to ensure all classes declare deliberation first."""
        models_path = Path("engine/supervisor/models.py")
        code = models_path.read_text(encoding="utf-8")
        assert validate_rationale_first_ast(code) is True

    def test_supervisor_supervisor_ast_rationale_first(self):
        """Inspects engine/supervisor/supervisor.py AST to ensure all classes declare deliberation first."""
        supervisor_path = Path("engine/supervisor/supervisor.py")
        code = supervisor_path.read_text(encoding="utf-8")
        assert validate_rationale_first_ast(code) is True
