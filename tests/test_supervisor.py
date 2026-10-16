"""tests/test_supervisor.py
Comprehensive unit and integration test suite for Build Supervisor & Closed-Loop Self-Healing (Feature 16).
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import pytest

from engine.compiler.compiler import EngineCompiler
from engine.director.episodic import CurrentEpisodeConfig, EpisodicDirector
from engine.director.genesis import GenesisDirector
from engine.director.models import (
    CurrentEpisodeConfig,
    EpisodicUpdateSpec,
    GenesisSpec,
)
from engine.loader import CartridgeLoader
from engine.supervisor.models import (
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    RemedyType,
    SupervisorEvaluationReport,
    SupervisorGateVerdict,
)
from engine.supervisor.supervisor import BuildSupervisor
from tests.test_director_episodic import MockBehavioralProfile
from tests.test_director_genesis import rusher_dna
from tests.test_tier1_units import validate_rationale_first_ast, validate_rationale_first_dict


@pytest.fixture
def supervisor(tmp_path: Path) -> BuildSupervisor:
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
    )


class TestSupervisorRationaleFirst:
    """Verifies that all supervisor models adhere strictly to Rationale-First architecture."""
    deliberation: str = "Test suite verifying Rationale-First architecture on supervisor models."

    def test_supervisor_models_ast_deliberation_first(self):
        models_path = Path("engine/supervisor/models.py")
        if models_path.exists():
            with open(models_path, "r", encoding="utf-8") as f:
                code = f.read()
            assert validate_rationale_first_ast(code) is True

    def test_gate_result_dict_starts_with_deliberation(self):
        result = QualityGateResult(
            deliberation="Syntax checks verified successfully without security violations.",
            gate_name="Gate A",
            passed=True,
        )
        data = result.to_dict()
        assert list(data.keys())[0] == "deliberation"

    def test_rejection_trace_dict_starts_with_deliberation(self):
        trace = RejectionDiagnosticTrace(
            deliberation="Critical path blocked by obstacle wall at x=640.",
            spec_id="spec_test",
            episode_number=1,
            failure_gate="Gate B: Traversability",
            failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
            failure_reason="Path from spawn to exit is blocked.",
        )
        data = trace.to_dict()
        assert list(data.keys())[0] == "deliberation"

    def test_evaluation_report_dict_starts_with_deliberation(self):
        report = SupervisorEvaluationReport(
            deliberation="Candidate build passed all gates and was promoted.",
            spec_id="spec_test",
            episode_number=1,
            status=SupervisorGateVerdict.PROMOTED,
            promoted_to_active=True,
        )
        data = report.to_dict()
        assert list(data.keys())[0] == "deliberation"


class TestQualityGates:
    """Verifies individual behavior of Gates A, B, C, D."""
    deliberation: str = "Test suite verifying 4 Quality Gates."

    def test_gate_a_syntax_and_ast_valid(self, supervisor: BuildSupervisor, tmp_path: Path):
        cart_dir = tmp_path / "valid_cart"
        cart_dir.mkdir()
        (cart_dir / "mechanics.py").write_text(
            "from engine.api.contracts import BaseCartridge\n"
            "class ValidCartridge(BaseCartridge):\n"
            "    def get_level_spec(self, ep=1): pass\n"
            "    def create_player(self, x, y, r=None): pass\n"
            "    def create_enemies(self, s): return []\n"
            "    def create_hazards(self, s): return []\n"
        )
        res = supervisor.verify_gate_a_syntax_and_ast(cart_dir)
        assert res.passed is True

    def test_gate_a_ast_security_rejection(self, supervisor: BuildSupervisor, tmp_path: Path):
        cart_dir = tmp_path / "malicious_cart"
        cart_dir.mkdir()
        (cart_dir / "mechanics.py").write_text("import os\nos.system('echo hacked')\n")
        res = supervisor.verify_gate_a_syntax_and_ast(cart_dir)
        assert res.passed is False
        assert res.failure_type == GateFailureType.SYNTAX_AST_ERROR
        assert "AST Security Invariant Violated" in res.details

    def test_gate_b_traversability_open_pass(self, supervisor: BuildSupervisor):
        from engine.api.contracts import BaseLevelSpec
        level_spec = BaseLevelSpec(
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            walls=[
                (0.0, 0.0, 1280.0, 32.0),
                (0.0, 688.0, 1280.0, 32.0),
                (0.0, 0.0, 32.0, 720.0),
                (1248.0, 0.0, 32.0, 720.0),
            ],
            deliberation="Open chamber test",
        )
        from engine.loader import GenericCartridge
        cart = GenericCartridge(Path("."))
        res, blocked = supervisor.verify_gate_b_traversability(cart, level_spec)
        assert res.passed is True
        assert len(blocked) == 0

    def test_gate_b_traversability_blocked_fail(self, supervisor: BuildSupervisor):
        # Full vertical wall dividing the room completely
        from engine.api.contracts import BaseLevelSpec
        level_spec = BaseLevelSpec(
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            walls=[
                (0.0, 0.0, 1280.0, 32.0),
                (0.0, 688.0, 1280.0, 32.0),
                (0.0, 0.0, 32.0, 720.0),
                (1248.0, 0.0, 32.0, 720.0),
                (640.0, 0.0, 48.0, 720.0),  # Impassable barrier
            ],
            deliberation="Blocked chamber test",
        )
        from engine.loader import GenericCartridge
        cart = GenericCartridge(Path("."))
        res, blocked = supervisor.verify_gate_b_traversability(cart, level_spec)
        assert res.passed is False
        assert res.failure_type == GateFailureType.CRITICAL_PATH_BLOCKED
        assert len(blocked) > 0

    def test_gate_c_survivability_spawn_hazard_overlap(self, supervisor: BuildSupervisor):
        from engine.api.contracts import BaseLevelSpec
        level_spec = BaseLevelSpec(
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            hazards=[
                {"id": "spikes_1", "type": "spikes", "x": 110.0, "y": 350.0, "w": 48.0, "h": 48.0}
            ],
            deliberation="Hazard overlap test",
        )
        res = supervisor.verify_gate_c_survivability(None, level_spec)
        assert res.passed is False
        assert res.failure_type == GateFailureType.SURVIVABILITY_VIOLATION
        assert "overlaps spawn clearance margin" in res.details

    def test_gate_d_zero_unhandled_exceptions(self, supervisor: BuildSupervisor):
        # Positive
        res_pass = supervisor.verify_gate_d_exceptions(None)
        assert res_pass.passed is True

        # Negative
        exc = RuntimeError("Simulated physics crash in collision loop")
        res_fail = supervisor.verify_gate_d_exceptions(exc)
        assert res_fail.passed is False
        assert res_fail.failure_type == GateFailureType.UNHANDLED_EXCEPTION


class TestClosedLoopSelfHealing:
    """Verifies that defects are rejected, diagnosed, repaired, and successfully promoted."""
    deliberation: str = "Test suite verifying closed-loop self-healing repair."

    def test_closed_loop_repair_blocked_corridor(self, supervisor: BuildSupervisor):
        director = EpisodicDirector()
        profile = MockBehavioralProfile()
        spec = director.evaluate_and_update(profile)

        # Inject impassable wall barrier at x=640
        spec.updated_episode_config.walls.append((640.0, 32.0, 48.0, 656.0))

        # Active cartridge dir must be empty before run
        active_dir = supervisor.active_cartridge_dir
        assert not (active_dir / "mechanics.py").exists()

        # Run evaluate_and_heal: Attempt 1 must fail Gate B, self-heal, Attempt 2 must pass
        report = supervisor.evaluate_and_heal(spec, max_retries=3)

        assert report.status == SupervisorGateVerdict.PROMOTED
        assert report.promoted_to_active is True
        assert report.attempt_count == 2
        assert len(report.repair_history) == 1
        assert report.repair_history[0].failure_type == GateFailureType.CRITICAL_PATH_BLOCKED

        # Active directory must now contain promoted files
        assert (active_dir / "mechanics.py").is_file()
        assert (active_dir / "level.json").is_file()

    def test_max_retries_exhaustion_blocks_promotion(self, supervisor: BuildSupervisor, monkeypatch):
        director = EpisodicDirector()
        spec = director.evaluate_and_update(MockBehavioralProfile())

        # Force gate verification to always fail Gate B
        from engine.supervisor.models import QualityGateResult, GateFailureType
        def mock_verify_gate_b(*args, **kwargs):
            return QualityGateResult(
                deliberation="Unfixable maze geometry",
                gate_name="Gate B",
                passed=False,
                failure_type=GateFailureType.CRITICAL_PATH_BLOCKED,
                details="Permanent unfixable blockage."
            ), [(20, 11)]

        monkeypatch.setattr(supervisor, "verify_gate_b_traversability", mock_verify_gate_b)

        report = supervisor.evaluate_and_heal(spec, max_retries=2)

        assert report.status == SupervisorGateVerdict.FAILED_MAX_RETRIES
        assert report.promoted_to_active is False
        assert report.attempt_count == 2
        assert len(report.repair_history) == 2

        # Active dir must NOT be created or written to
        assert not (supervisor.active_cartridge_dir / "mechanics.py").exists()


class TestEndToEndDirectorSupervisorIntegration:
    """Verifies seamless integration from Genesis/Episodic Directors through Build Supervisor."""
    deliberation: str = "Test suite verifying Director-to-Supervisor end-to-end integration."

    def test_genesis_spec_clean_promotion(self, supervisor: BuildSupervisor, rusher_dna):
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        report = supervisor.evaluate_and_heal(spec)

        assert report.status == SupervisorGateVerdict.PROMOTED
        assert report.promoted_to_active is True
        assert report.attempt_count == 1
        assert (supervisor.active_cartridge_dir / "manifest.json").is_file()

        # Verify loaded active cartridge
        cartridge = CartridgeLoader.load_cartridge(supervisor.active_cartridge_dir)
        level_spec = cartridge.get_level_spec(1)
        assert len(level_spec.walls) > 0
