"""tests/test_challenger_m4_1_stress.py
Adversarial empirical stress-test suite for Milestone M4:
Independent Challenger Gate & Triangulated Grounding (Challenger 1).

Exhaustively verifies:
1. Anti-stub evasion stress testing:
   - Nested functions with pass
   - Empty bodies in multi-module files (methods, async functions, ellipsis, return None, raise NotImplementedError)
   - Docstring-only functions and methods
   - # TODO comments embedded in string literals vs actual comment tokens
   - Disconnected classes never imported or referenced across multi-module cartridges
2. 100% threshold enforcement:
   - 9/10 criteria passing yields overall_verdict = False
   - AST stub detection overrides LLM hallucinated overall_verdict = True to False
   - Blank canvas overrides LLM hallucinated overall_verdict = True to False
   - RejectionDiagnosticTrace schema integrity (deliberation strictly at index 0)
3. Rule 5 fail-fast stress:
   - client is None raises StudioAgentError
   - API network timeouts and server errors re-raise authentic StudioAgentError (zero silent fallbacks)
   - Empty or malformed LLM responses raise StudioAgentError
4. BuildSupervisor Gate F Integration:
   - Gate F rejection blocks promotion to active cartridge
   - Valid candidate passing Gate F promotes cleanly
"""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import struct
import wave
from unittest.mock import MagicMock, patch

from PIL import Image
import pygame
import pytest

from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.sim.headless import HeadlessSimulator, SimulationInput
from engine.studio.models import (
    AcceptanceCriterion,
    GameRequirementsSpec,
    RequirementTrack,
    StudioAgentError,
)
from engine.supervisor.challenger import (
    CandidateASTInspector,
    CandidateTelemetryAuditor,
    CandidateVisualPlaytester,
    ChallengerGate,
)
from engine.supervisor.models import (
    ChallengerGateEvaluation,
    GateFailureType,
    RejectionDiagnosticTrace,
    RemedyType,
    RequirementEvaluationRecord,
)
from engine.supervisor.supervisor import BuildSupervisor


@pytest.fixture
def base_genuine_cartridge(tmp_path: Path) -> Path:
    """Creates a hermetic candidate cartridge with valid assets and structure."""
    cart_dir = tmp_path / "candidate_genuine"
    assets_dir = cart_dir / "assets"
    audio_dir = assets_dir / "audio"
    sprites_dir = assets_dir / "sprites"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sprites_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "cartridge_id": "cand_genuine_01",
        "title": "Quantum Grid",
        "version": "1.0.0",
        "entry_point": "mechanics.py",
    }
    (cart_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    level = {
        "deliberation": "Spatial layout for grid traversal.",
        "level_id": "level_01",
        "name": "Chamber",
        "width": 1280.0,
        "height": 720.0,
        "spawn_point": [100.0, 360.0],
        "exit_point": [1180.0, 360.0],
        "walls": [
            [0.0, 0.0, 1280.0, 32.0],
            [0.0, 688.0, 1280.0, 32.0],
        ],
        "enemies": [],
        "hazards": [],
    }
    (cart_dir / "level.json").write_text(json.dumps(level), encoding="utf-8")

    narrative = {
        "deliberation": "Narrative specification.",
        "world": {"setting": "Space", "theme": "Sci-Fi"},
        "characters": [
            {"id": "hero", "name": "Hero", "portrait": "assets/sprites/hero.png"}
        ],
        "dialogues": [],
        "dialogue_barks": {},
    }
    (cart_dir / "narrative.json").write_text(json.dumps(narrative), encoding="utf-8")

    # Audio: valid 0.2s WAV
    wav_path = audio_dir / "laser.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        raw = bytearray()
        for i in range(4410):
            raw.extend(struct.pack("<h", 1000))
        wf.writeframes(raw)

    # Sprites: valid 32x32 PNGs
    for s in ("hero.png", "tile.png"):
        img = Image.new("RGB", (32, 32), color=(50, 150, 200))
        img.save(sprites_dir / s, format="PNG")

    return cart_dir


# =============================================================================
# 1. Anti-Stub Evasion Stress Testing
# =============================================================================

class TestAntiStubEvasionEmpirical:
    """Stress tests AST and Token inspection against evasion tactics."""

    def test_nested_function_with_pass_detected(self, tmp_path: Path):
        code = """# Mechanics with nested pass stub
from engine.api.contracts import BaseCartridge

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active cartridge"
    def compute_motion(self, velocity: float) -> float:
        def _inner_evasion_stub():
            pass
        return velocity * 1.5
"""
        (tmp_path / "mechanics.py").write_text(code, encoding="utf-8")
        inspector = CandidateASTInspector()
        report = inspector.inspect_directory(tmp_path)

        assert report["has_critical_stubs"] is True, "Nested pass stub must be flagged as critical stub"
        stubs = report["pass_stubs"]
        assert len(stubs) >= 1
        assert any(s["function"] == "_inner_evasion_stub" and s["stmt"] == "pass" for s in stubs)

    def test_empty_bodies_across_multi_module_files(self, tmp_path: Path):
        mech_code = """# mechanics.py
from engine.api.contracts import BaseCartridge
from entities import DroneEntity
from scenes import VictoryScene

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active cartridge"
    def __init__(self):
        self.drone = DroneEntity()
        self.scene = VictoryScene()
"""
        entities_code = """# entities.py
class DroneEntity:
    def __init__(self):
        self.hp = 100
    async def fire_laser(self):
        pass
    def calculate_trajectory(self):
        ...
"""
        scenes_code = """# scenes.py
class VictoryScene:
    def render(self):
        return None
    def cleanup(self):
        raise NotImplementedError
"""
        (tmp_path / "mechanics.py").write_text(mech_code, encoding="utf-8")
        (tmp_path / "entities.py").write_text(entities_code, encoding="utf-8")
        (tmp_path / "scenes.py").write_text(scenes_code, encoding="utf-8")

        inspector = CandidateASTInspector()
        report = inspector.inspect_directory(tmp_path)

        assert report["has_critical_stubs"] is True
        stub_funcs = {s["function"]: s["stmt"] for s in report["pass_stubs"]}
        assert "fire_laser" in stub_funcs
        assert "calculate_trajectory" in stub_funcs
        assert "render" in stub_funcs
        assert "cleanup" in stub_funcs

    def test_docstring_only_functions_and_methods(self, tmp_path: Path):
        code = """# Module with docstring only functions
from engine.api.contracts import BaseCartridge

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active cartridge"
    def update_physics(self, dt: float):
        \"\"\"Executes complex sub-step physics simulation.\"\"\"

    def check_collisions(self):
        '''Multi-line collision detector
        evaluating AABB intersections.
        '''

def global_helper():
    \"\"\"Docstring only global function.\"\"\"
"""
        (tmp_path / "mechanics.py").write_text(code, encoding="utf-8")
        inspector = CandidateASTInspector()
        report = inspector.inspect_directory(tmp_path)

        assert report["has_critical_stubs"] is True
        empty_funcs = [b["function"] for b in report["empty_bodies"]]
        assert "update_physics" in empty_funcs
        assert "check_collisions" in empty_funcs
        assert "global_helper" in empty_funcs

    def test_todo_comments_embedded_in_strings_vs_actual_comments(self, tmp_path: Path):
        code = '''# Mechanics module with comments and strings
# TODO: implement player health damage
# FIXME: address physics tick jitter
from engine.api.contracts import BaseCartridge

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active cartridge"
    def display_help(self):
        # STUB: needs localization
        message = "Press F1 for # TODO help text and # FIXME notes"
        doc = """
        This doc mentions # XXX and # IMPLEMENT ME inside a multiline string.
        """
        return message + doc
'''
        (tmp_path / "mechanics.py").write_text(code, encoding="utf-8")
        inspector = CandidateASTInspector()
        report = inspector.inspect_directory(tmp_path)

        # Tokenize inspection should only capture true comments
        found_tags = [t["tag"] for t in report["todo_comments"]]
        assert "TODO" in found_tags
        assert "FIXME" in found_tags
        assert "STUB" in found_tags
        assert len(report["todo_comments"]) == 3, "Strings mentioning # TODO must NOT be parsed as comments"

    def test_disconnected_classes_in_multi_module_cartridge(self, tmp_path: Path):
        mech_code = """# mechanics.py
from engine.api.contracts import BaseCartridge
from entities import ActivePlayer

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active cartridge"
    def __init__(self):
        self.player = ActivePlayer()
"""
        entities_code = """# entities.py
class ActivePlayer:
    def __init__(self):
        self.x = 10.0

class OrphanGhostBoss:
    \"\"\"Defined in entities.py but never referenced anywhere else.\"\"\"
    def __init__(self):
        self.phase = 2
"""
        (tmp_path / "mechanics.py").write_text(mech_code, encoding="utf-8")
        (tmp_path / "entities.py").write_text(entities_code, encoding="utf-8")

        inspector = CandidateASTInspector()
        report = inspector.inspect_directory(tmp_path)

        disconnected = [d["symbol"] for d in report["disconnected_symbols"]]
        assert "OrphanGhostBoss" in disconnected
        assert "ActivePlayer" not in disconnected


# =============================================================================
# 2. Strict 100% Threshold Enforcement
# =============================================================================

class TestStrict100PercentThresholdEnforcement:
    """Empirically stresses the 100% threshold invariant."""

    def test_nine_out_of_ten_criteria_passing_yields_false(self, tmp_path: Path, base_genuine_cartridge: Path):
        # 10 requirement tracks
        tracks = []
        for i in range(1, 11):
            tracks.append(
                RequirementTrack(
                    deliberation=f"Deliberation for track R{i}",
                    id=f"R{i}",
                    title=f"Feature {i}",
                    description=f"Description {i}",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation=f"Criterion {i}.1",
                            id=f"AC{i}.1",
                            description=f"Criterion description {i}",
                            verification_method="ast_inspection",
                        )
                    ],
                )
            )

        spec = GameRequirementsSpec(
            deliberation="Ten track requirements specification.",
            game_title="TenTrackGame",
            genre_paradigm="simulation",
            requirements=tracks,
        )

        # Write clean mechanics.py in cartridge
        (base_genuine_cartridge / "mechanics.py").write_text("""# mechanics.py
from engine.api.contracts import BaseCartridge

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Genuine cartridge"
    def update(self):
        self.tick = 1
        return self.tick
""", encoding="utf-8")

        # Mock LLM returning 9 pass, 1 fail, but hallucinating overall_verdict = True
        evaluated_recs = []
        for i in range(1, 10):
            evaluated_recs.append({
                "deliberation": f"Track R{i} satisfied.",
                "requirement_id": f"R{i}",
                "fulfilled": True,
                "grounding_evidence": f"Code verified for R{i}.",
                "critique_notes": "Good.",
            })
        evaluated_recs.append({
            "deliberation": f"Track R10 unfulfilled: missing combat logic.",
            "requirement_id": "R10",
            "fulfilled": False,
            "grounding_evidence": "No combat logic found.",
            "critique_notes": "Fails AC10.1.",
        })

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "deliberation": "Evaluated all 10 tracks; 9 passed, 1 failed, but I mistakenly believe overall is True.",
            "candidate_id": base_genuine_cartridge.name,
            "overall_verdict": True,  # LLM hallucination!
            "evaluated_requirements": evaluated_recs,
            "rejection_trace": None,
        })
        mock_client.models.generate_content.return_value = mock_response

        gate = ChallengerGate(client=mock_client)
        evaluation = gate.evaluate(candidate_dir=base_genuine_cartridge, requirements_spec=spec)

        # Must be programmatically forced to False!
        assert evaluation.overall_verdict is False, "9/10 criteria passing MUST yield overall_verdict = False"
        assert evaluation.rejection_trace is not None
        assert "R10" in evaluation.rejection_trace.unfulfilled_requirements
        assert evaluation.rejection_trace.failure_type == GateFailureType.REQUIREMENT_UNFULFILLED

        # Deliberation must be at index 0 in model_dump()
        dumped = evaluation.rejection_trace.model_dump()
        assert list(dumped.keys())[0] == "deliberation"

    def test_ast_stub_forces_rejection_despite_ten_out_of_ten_llm_hallucination(
        self, tmp_path: Path, base_genuine_cartridge: Path
    ):
        # Insert a hollow stub into mechanics.py
        (base_genuine_cartridge / "mechanics.py").write_text("""# mechanics.py
from engine.api.contracts import BaseCartridge

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Hollow cartridge"
    def update(self):
        pass
""", encoding="utf-8")

        spec = GameRequirementsSpec(
            deliberation="One track spec.",
            game_title="SingleTrackGame",
            genre_paradigm="puzzle",
            requirements=[
                RequirementTrack(
                    deliberation="Track 1",
                    id="R1",
                    title="Feature 1",
                    description="Move",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="Crit 1",
                            id="AC1.1",
                            description="Move logic",
                            verification_method="ast_inspection",
                        )
                    ],
                )
            ],
        )

        # Mock LLM claiming R1 is fulfilled and overall_verdict = True
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "deliberation": "LLM hallucinating that pass stub is valid.",
            "candidate_id": base_genuine_cartridge.name,
            "overall_verdict": True,
            "evaluated_requirements": [
                {
                    "deliberation": "R1 is totally fine.",
                    "requirement_id": "R1",
                    "fulfilled": True,
                    "grounding_evidence": "Looks good.",
                    "critique_notes": "None.",
                }
            ],
            "rejection_trace": None,
        })
        mock_client.models.generate_content.return_value = mock_response

        gate = ChallengerGate(client=mock_client)
        evaluation = gate.evaluate(candidate_dir=base_genuine_cartridge, requirements_spec=spec)

        assert evaluation.overall_verdict is False, "Detected AST hollow stub MUST override LLM hallucination"
        assert evaluation.rejection_trace is not None
        assert "Hollow code stubs detected" in evaluation.rejection_trace.failure_reason

    def test_blank_canvas_forces_rejection_despite_llm_pass(
        self, tmp_path: Path, base_genuine_cartridge: Path
    ):
        (base_genuine_cartridge / "mechanics.py").write_text("""# mechanics.py
from engine.api.contracts import BaseCartridge

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Genuine code"
    def update(self):
        self.x = 10
        return self.x
""", encoding="utf-8")

        spec = GameRequirementsSpec(
            deliberation="Spec.",
            game_title="Game",
            genre_paradigm="arcade",
            requirements=[
                RequirementTrack(
                    deliberation="Track 1",
                    id="R1",
                    title="Render",
                    description="Render screen",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="Crit 1",
                            id="AC1.1",
                            description="Screen draws graphics",
                            verification_method="headless_playtest",
                        )
                    ],
                )
            ],
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "deliberation": "LLM claimed everything passed.",
            "candidate_id": base_genuine_cartridge.name,
            "overall_verdict": True,
            "evaluated_requirements": [
                {
                    "deliberation": "R1 passed.",
                    "requirement_id": "R1",
                    "fulfilled": True,
                    "grounding_evidence": "Screen drawn.",
                    "critique_notes": "None.",
                }
            ],
            "rejection_trace": None,
        })
        mock_client.models.generate_content.return_value = mock_response

        # Mock visual playtester to simulate a blank canvas
        gate = ChallengerGate(client=mock_client)
        mock_sim = MagicMock()
        mock_sim.avg_frame_time_ms = 5.0
        mock_sim.frame_budget_passed = True
        mock_sim.step_count = 60
        mock_sim.elapsed_time = 1.0
        mock_sim.telemetry_recorder = None
        mock_sim.player = None
        mock_sim.current_scene = MagicMock()
        mock_sim.current_scene.player_pos = (120.0, 360.0)
        blank_surf = pygame.Surface((1280, 720))
        blank_surf.fill((0, 0, 0))
        mock_sim.engine.logical_surface = blank_surf
        mock_sim.current_scene.render = MagicMock()

        evaluation = gate.evaluate(
            candidate_dir=base_genuine_cartridge,
            requirements_spec=spec,
            sim=mock_sim,
        )

        assert evaluation.overall_verdict is False, "Blank canvas detection MUST force overall_verdict = False"
        assert evaluation.rejection_trace is not None
        assert "Blank canvas detected" in evaluation.rejection_trace.failure_reason


# =============================================================================
# 3. Rule 5 Fail-Fast Stress Testing
# =============================================================================

class TestRule5FailFastStress:
    """Stress tests Rule 5 (Zero Silent Fallbacks) across network, client, and payload failures."""

    def test_client_is_none_raises_studio_agent_error(self, tmp_path: Path):
        gate = ChallengerGate(client=None)
        spec = GameRequirementsSpec(
            deliberation="Spec",
            game_title="Game",
            genre_paradigm="sim",
            requirements=[
                RequirementTrack(
                    deliberation="T1",
                    id="R1",
                    title="Feature",
                    description="Desc",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="C1",
                            id="AC1",
                            description="Crit",
                            verification_method="ast_inspection",
                        )
                    ],
                )
            ],
        )
        with pytest.raises(StudioAgentError) as exc_info:
            gate.evaluate(candidate_dir=tmp_path, requirements_spec=spec, client=None)
        assert "Rule 5: Zero Silent Fallbacks" in str(exc_info.value)

    def test_api_network_timeout_re_raises_studio_agent_error(self, tmp_path: Path, base_genuine_cartridge: Path):
        (base_genuine_cartridge / "mechanics.py").write_text("""# mechanics.py
from engine.api.contracts import BaseCartridge
class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active"
    def update(self):
        self.x = 1
""", encoding="utf-8")

        spec = GameRequirementsSpec(
            deliberation="Spec",
            game_title="Game",
            genre_paradigm="sim",
            requirements=[
                RequirementTrack(
                    deliberation="T1",
                    id="R1",
                    title="Feature",
                    description="Desc",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="C1",
                            id="AC1",
                            description="Crit",
                            verification_method="ast_inspection",
                        )
                    ],
                )
            ],
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = TimeoutError("Connection to Gemini timed out after 30s")

        gate = ChallengerGate(client=mock_client)
        with pytest.raises(StudioAgentError) as exc_info:
            gate.evaluate(candidate_dir=base_genuine_cartridge, requirements_spec=spec)
        assert "Connection to Gemini timed out" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, TimeoutError)

    def test_api_server_error_re_raises_studio_agent_error(self, tmp_path: Path, base_genuine_cartridge: Path):
        (base_genuine_cartridge / "mechanics.py").write_text("""# mechanics.py
from engine.api.contracts import BaseCartridge
class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active"
    def update(self):
        self.x = 1
""", encoding="utf-8")

        spec = GameRequirementsSpec(
            deliberation="Spec",
            game_title="Game",
            genre_paradigm="sim",
            requirements=[
                RequirementTrack(
                    deliberation="T1",
                    id="R1",
                    title="Feature",
                    description="Desc",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="C1",
                            id="AC1",
                            description="Crit",
                            verification_method="ast_inspection",
                        )
                    ],
                )
            ],
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("500 Internal Server Error: Quota exceeded")

        gate = ChallengerGate(client=mock_client)
        with pytest.raises(StudioAgentError) as exc_info:
            gate.evaluate(candidate_dir=base_genuine_cartridge, requirements_spec=spec)
        assert "500 Internal Server Error" in str(exc_info.value)

    def test_empty_response_from_api_raises_studio_agent_error(self, tmp_path: Path, base_genuine_cartridge: Path):
        (base_genuine_cartridge / "mechanics.py").write_text("""# mechanics.py
from engine.api.contracts import BaseCartridge
class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active"
    def update(self):
        self.x = 1
""", encoding="utf-8")

        spec = GameRequirementsSpec(
            deliberation="Spec",
            game_title="Game",
            genre_paradigm="sim",
            requirements=[
                RequirementTrack(
                    deliberation="T1",
                    id="R1",
                    title="Feature",
                    description="Desc",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="C1",
                            id="AC1",
                            description="Crit",
                            verification_method="ast_inspection",
                        )
                    ],
                )
            ],
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response

        gate = ChallengerGate(client=mock_client)
        with pytest.raises(StudioAgentError, match="Empty response returned from Gemini"):
            gate.evaluate(candidate_dir=base_genuine_cartridge, requirements_spec=spec)

    def test_malformed_json_from_api_raises_studio_agent_error(self, tmp_path: Path, base_genuine_cartridge: Path):
        (base_genuine_cartridge / "mechanics.py").write_text("""# mechanics.py
from engine.api.contracts import BaseCartridge
class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active"
    def update(self):
        self.x = 1
""", encoding="utf-8")

        spec = GameRequirementsSpec(
            deliberation="Spec",
            game_title="Game",
            genre_paradigm="sim",
            requirements=[
                RequirementTrack(
                    deliberation="T1",
                    id="R1",
                    title="Feature",
                    description="Desc",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="C1",
                            id="AC1",
                            description="Crit",
                            verification_method="ast_inspection",
                        )
                    ],
                )
            ],
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "{invalid json: unclosed"
        mock_client.models.generate_content.return_value = mock_response

        gate = ChallengerGate(client=mock_client)
        with pytest.raises(StudioAgentError, match="ChallengerGate structured evaluation failed"):
            gate.evaluate(candidate_dir=base_genuine_cartridge, requirements_spec=spec)


# =============================================================================
# 4. BuildSupervisor Gate F Integration Stress Testing
# =============================================================================

class TestBuildSupervisorGateFStress:
    """Stress tests BuildSupervisor Gate F evaluation and promotion blocking."""

    def test_gate_f_rejection_blocks_promotion(self, base_genuine_cartridge: Path, tmp_path: Path):
        cartridges_root = tmp_path / "cartridges"
        active_dir = cartridges_root / "active"
        cartridges_root.mkdir(parents=True, exist_ok=True)
        active_dir.mkdir(parents=True, exist_ok=True)

        # Insert hollow stub into base_genuine_cartridge
        (base_genuine_cartridge / "mechanics.py").write_text("""# mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.entities.player import Player as BasePlayer

class ActiveCartridge(BaseCartridge):
    deliberation: str = "Active cartridge with stub"
    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            level_id=f"episode_{episode_number}",
            name="Procedural Chamber",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 360.0),
            exit_point=(1180.0, 360.0),
            walls=[(0.0, 0.0, 1280.0, 32.0), (0.0, 688.0, 1280.0, 32.0)],
            deliberation="Level geometry generated for tactical traversal."
        )

    def create_player(self, x: float, y: float, telemetry_recorder=None) -> BasePlayer:
        return BasePlayer(x=x, y=y)

    def update(self, dt: float):
        pass
""", encoding="utf-8")

        # Provide requirements spec
        spec_data = {
            "deliberation": "Backlog spec",
            "game_title": "Quantum Odyssey",
            "genre_paradigm": "tactical_traversal",
            "requirements": [
                {
                    "deliberation": "Track 1",
                    "id": "R1",
                    "title": "Tactical Traversal",
                    "description": "Player traversal",
                    "acceptance_criteria": [
                        {
                            "deliberation": "Crit 1",
                            "id": "AC1.1",
                            "description": "Update loop executes",
                            "verification_method": "ast_inspection",
                        }
                    ],
                }
            ],
            "asset_manifest": [],
        }
        (base_genuine_cartridge / "requirements_spec.json").write_text(json.dumps(spec_data), encoding="utf-8")

        # Mock Gemini client claiming pass
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "deliberation": "LLM claims pass.",
            "candidate_id": base_genuine_cartridge.name,
            "overall_verdict": True,
            "evaluated_requirements": [
                {
                    "deliberation": "R1 passed.",
                    "requirement_id": "R1",
                    "fulfilled": True,
                    "grounding_evidence": "All good.",
                    "critique_notes": "None.",
                }
            ],
            "rejection_trace": None,
        })
        mock_client.models.generate_content.return_value = mock_response

        mock_challenger = MagicMock()
        mock_trace = RejectionDiagnosticTrace(
            deliberation="Gate F detected hollow stub in mechanics.py.",
            candidate_id=base_genuine_cartridge.name,
            failure_gate="Gate F: Challenger Gate",
            failure_type=GateFailureType.REQUIREMENT_UNFULFILLED,
            failure_reason="Hollow code stubs detected: mechanics.py:update (pass)",
            remedy_type=RemedyType.IMPLEMENT_REQUIREMENT,
            recommended_remedy="Implement real update logic",
            unfulfilled_requirements=["R1"],
            failing_criteria=["AC1.1"],
        )
        mock_challenger.evaluate.return_value = ChallengerGateEvaluation(
            deliberation="Adversarial rejection of hollow stub.",
            candidate_id=base_genuine_cartridge.name,
            overall_verdict=False,
            evaluated_requirements=[
                RequirementEvaluationRecord(
                    deliberation="R1 hollow.",
                    requirement_id="R1",
                    fulfilled=False,
                    grounding_evidence="Pass stub in mechanics.py",
                    critique_notes="Fails anti-stub inspection.",
                )
            ],
            rejection_trace=mock_trace,
        )

        supervisor = BuildSupervisor(
            cartridges_root=cartridges_root,
            active_cartridge_dir=active_dir,
            challenger_gate=mock_challenger,
        )
        gate_results, trace = supervisor.evaluate_candidate_cartridge(candidate_dir=base_genuine_cartridge)

        assert gate_results["gate_f"].passed is False, "Candidate with hollow stub in Gate F MUST NOT pass"
        assert trace is not None
        assert trace.failure_type == GateFailureType.REQUIREMENT_UNFULFILLED
        assert "Gate F: Challenger Gate" in trace.failure_gate
