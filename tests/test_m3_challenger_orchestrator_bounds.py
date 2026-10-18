"""tests/test_m3_challenger_orchestrator_bounds.py
Empirical Adversarial Stress Harness for Milestone M3 Gate Verification.
Challenger 1: Orchestrator Resilience, AST Security & Multimodal Bounds.
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
import asyncio
import ast
import io
import json
from pathlib import Path
import random
import struct
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock
import pytest
from PIL import Image

from engine.assets.palette import STYLE_BIBLE_16_PALETTE
from engine.loader import ASTSecurityError, validate_cartridge_ast
from engine.scheduler.artifact import ArtifactMetadataEnvelope
from engine.studio.audio import (
    AudioSynthesisBackend,
    AudioSynthesisResult,
    VoiceActingAudioProducerAgent,
    VoiceSynthesisError,
    create_wav_header,
    escape_ssml,
    validate_wav_file,
)
from engine.studio.models import (
    AcceptanceCriterion,
    AgentPlan,
    AgentVerificationRecord,
    CharacterSheet,
    DialogueBarkSpec,
    GameRequirementsSpec,
    GenesisWorldSpec,
    NarrativeBible,
    RequirementTrack,
    StudioBuildState,
    StudioSwarmPlan,
    TelemetryArchitecturePlan,
    UpdateWorldSpec,
)
from engine.studio.orchestrator import (
    BuildVerificationExhaustedError,
    CircularDependencyError,
    InvalidStateTransitionError,
    StudioAgentDAG,
    StudioOrchestrator,
    StudioSwarmExecutionError,
)
from engine.studio.swe import LeadGameplaySWEAgent
from engine.studio.visual import VisualArtSpriteDesignerAgent
from engine.ui.models import SwarmEventType
from engine.ui.server import SwarmEventBus
from tests.conftest import MockGeminiClient


# ===========================================================================
# 1. Orchestrator DAG & State Machine Adversarial Harness
# ===========================================================================

class TestOrchestratorDAGAdversarial:
    """Stress tests DAG topological sorting, cycle detection, and edge cases."""
    deliberation: str = "Empirical verification of StudioAgentDAG cycle rejection and ordering."

    def test_dag_self_loop_rejection(self) -> None:
        """Adversarial self-referential loop A -> A raises CircularDependencyError."""
        dag = StudioAgentDAG(edges=[("NodeA", "NodeA")])
        assert dag.has_cycle() is True
        with pytest.raises(CircularDependencyError) as exc_info:
            dag.topological_sort()
        assert "Cycle detected" in str(exc_info.value)

    def test_dag_two_node_cycle_rejection(self) -> None:
        """Adversarial two-node mutual dependency A -> B -> A raises CircularDependencyError."""
        dag = StudioAgentDAG(edges=[("A", "B"), ("B", "A")])
        assert dag.has_cycle() is True
        with pytest.raises(CircularDependencyError) as exc_info:
            dag.topological_sort()
        assert "Cycle detected" in str(exc_info.value)

    def test_dag_multi_node_cycle_rejection(self) -> None:
        """Adversarial 5-node ring A -> B -> C -> D -> E -> A raises CircularDependencyError."""
        dag = StudioAgentDAG(edges=[
            ("Node1", "Node2"),
            ("Node2", "Node3"),
            ("Node3", "Node4"),
            ("Node4", "Node5"),
            ("Node5", "Node1"),
        ])
        assert dag.has_cycle() is True
        with pytest.raises(CircularDependencyError) as exc_info:
            dag.topological_sort()
        assert "Cycle detected" in str(exc_info.value)

    def test_dag_cycle_embedded_in_standard_pipeline(self) -> None:
        """Standard studio pipeline with back-edge Supervisor -> Director raises CircularDependencyError."""
        dag = StudioAgentDAG()
        assert dag.has_cycle() is False
        dag.add_edge("Supervisor", "Director")
        assert dag.has_cycle() is True
        with pytest.raises(CircularDependencyError) as exc_info:
            dag.topological_sort()
        assert "Cycle detected" in str(exc_info.value)

    def test_dag_disconnected_component_cycle(self) -> None:
        """Cycle in one component while another is valid must still raise CircularDependencyError."""
        dag = StudioAgentDAG(edges=[
            ("ValidRoot", "ValidChild"),
            ("ValidChild", "ValidLeaf"),
            ("CyclicX", "CyclicY"),
            ("CyclicY", "CyclicX"),
        ])
        assert dag.has_cycle() is True
        with pytest.raises(CircularDependencyError):
            dag.topological_sort()

    def test_dag_empty_edges_succeeds(self) -> None:
        """Empty graph has zero cycles and returns empty topological sort."""
        dag = StudioAgentDAG(edges=[])
        assert dag.has_cycle() is False
        assert dag.topological_sort() == []


class TestOrchestratorStateMachineAdversarial:
    """Stress tests state machine lifecycle and exhaustive invalid transitions."""
    deliberation: str = "Empirical verification of 5-state lifecycle and illegal transition barriers."

    def test_exhaustive_invalid_state_transitions(self) -> None:
        """Exhaustively verifies all invalid transitions across all StudioBuildState values raise InvalidStateTransitionError."""
        allowed: Dict[StudioBuildState, Set[StudioBuildState]] = {
            StudioBuildState.IDLE: {StudioBuildState.ANALYZING},
            StudioBuildState.ANALYZING: {StudioBuildState.BRAINSTORMING, StudioBuildState.SYNTHESIZING},
            StudioBuildState.BRAINSTORMING: {StudioBuildState.AWAITING_GREENLIGHT, StudioBuildState.SYNTHESIZING},
            StudioBuildState.AWAITING_GREENLIGHT: {StudioBuildState.BRAINSTORMING, StudioBuildState.SYNTHESIZING, StudioBuildState.IDLE},
            StudioBuildState.SYNTHESIZING: {StudioBuildState.VERIFYING, StudioBuildState.BRAINSTORMING, StudioBuildState.AWAITING_GREENLIGHT},
            StudioBuildState.VERIFYING: {StudioBuildState.PROMOTED, StudioBuildState.SYNTHESIZING},
            StudioBuildState.PROMOTED: {StudioBuildState.IDLE},
        }
        all_states = list(StudioBuildState)

        for current_state in all_states:
            for target_state in all_states:
                orch = StudioOrchestrator()
                # Fast forward to current_state using valid sequence
                if current_state != StudioBuildState.IDLE:
                    if current_state == StudioBuildState.ANALYZING:
                        orch.transition_to(StudioBuildState.ANALYZING, "Setup")
                    elif current_state == StudioBuildState.BRAINSTORMING:
                        orch.transition_to(StudioBuildState.ANALYZING, "Setup")
                        orch.transition_to(StudioBuildState.BRAINSTORMING, "Setup")
                    elif current_state == StudioBuildState.AWAITING_GREENLIGHT:
                        orch.transition_to(StudioBuildState.ANALYZING, "Setup")
                        orch.transition_to(StudioBuildState.BRAINSTORMING, "Setup")
                        orch.transition_to(StudioBuildState.AWAITING_GREENLIGHT, "Setup")
                    elif current_state == StudioBuildState.SYNTHESIZING:
                        orch.transition_to(StudioBuildState.ANALYZING, "Setup")
                        orch.transition_to(StudioBuildState.SYNTHESIZING, "Setup")
                    elif current_state == StudioBuildState.VERIFYING:
                        orch.transition_to(StudioBuildState.ANALYZING, "Setup")
                        orch.transition_to(StudioBuildState.SYNTHESIZING, "Setup")
                        orch.transition_to(StudioBuildState.VERIFYING, "Setup")
                    elif current_state == StudioBuildState.PROMOTED:
                        orch.transition_to(StudioBuildState.ANALYZING, "Setup")
                        orch.transition_to(StudioBuildState.SYNTHESIZING, "Setup")
                        orch.transition_to(StudioBuildState.VERIFYING, "Setup")
                        orch.transition_to(StudioBuildState.PROMOTED, "Setup")

                assert orch.state == current_state

                if target_state in allowed[current_state]:
                    orch.transition_to(target_state, f"Valid: {current_state} -> {target_state}")
                    assert orch.state == target_state
                else:
                    initial_delib = orch.deliberation
                    with pytest.raises(InvalidStateTransitionError) as exc_info:
                        orch.transition_to(target_state, f"Illegal jump to {target_state}")
                    assert f"Invalid state transition from {current_state} to {target_state}" in str(exc_info.value)
                    # State and deliberation remain uncorrupted
                    assert orch.state == current_state
                    assert orch.deliberation == initial_delib

    def test_direct_idle_to_promoted_rejection(self) -> None:
        """Direct jump from IDLE to PROMOTED must be strictly rejected."""
        orch = StudioOrchestrator()
        assert orch.state == StudioBuildState.IDLE
        with pytest.raises(InvalidStateTransitionError):
            orch.transition_to(StudioBuildState.PROMOTED, "Direct bypass")
        assert orch.state == StudioBuildState.IDLE

    def test_self_healing_reentry_permits_verifying_to_synthesizing_only(self) -> None:
        """VERIFYING permits return to SYNTHESIZING for repair, but forbids jumping to ANALYZING or IDLE."""
        orch = StudioOrchestrator()
        orch.transition_to(StudioBuildState.ANALYZING, "Step 1")
        orch.transition_to(StudioBuildState.SYNTHESIZING, "Step 2")
        orch.transition_to(StudioBuildState.VERIFYING, "Step 3")

        # Illegal backward jumps
        with pytest.raises(InvalidStateTransitionError):
            orch.transition_to(StudioBuildState.ANALYZING, "Cannot jump backwards to ANALYZING")
        with pytest.raises(InvalidStateTransitionError):
            orch.transition_to(StudioBuildState.IDLE, "Cannot jump directly to IDLE from VERIFYING")

        # Legal self-healing loop
        orch.transition_to(StudioBuildState.SYNTHESIZING, "Self-healing repair iteration")
        assert orch.state == StudioBuildState.SYNTHESIZING


class TestPEVRetryExhaustionAdversarial:
    """Stress tests Plan-Execute-Verify retry counter, fault isolation, and exhaustion."""
    deliberation: str = "Empirical verification of PEV retry exhaustion and recovery behavior."

    @pytest.mark.anyio
    async def test_pev_retry_exhaustion_exact_attempt_counting(self) -> None:
        """PEV loop runs exactly max_retries before raising BuildVerificationExhaustedError."""
        orch = StudioOrchestrator()
        attempts: List[int] = []

        async def plan():
            return AgentPlan(deliberation="Plan", agent_name="AgentX", objective="Obj", steps=["Do X"])

        async def execute(p):
            return {"payload": "broken"}

        async def verify(res):
            attempts.append(len(attempts) + 1)
            return AgentVerificationRecord(
                deliberation="Defect found",
                agent_name="AgentX",
                passed=False,
                violations=[f"Violation at attempt {len(attempts)}"],
            )

        with pytest.raises(BuildVerificationExhaustedError) as exc_info:
            await orch.execute_pev_cycle("AgentX", plan, execute, verify, max_retries=4)

        assert len(attempts) == 4
        assert "Agent AgentX failed PEV verification after 4 attempts" in str(exc_info.value)
        assert "Violation at attempt 4" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_pev_eventual_success_within_retry_limit(self) -> None:
        """PEV recovers on attempt 3 if verification succeeds before max_retries."""
        orch = StudioOrchestrator()
        attempts: List[int] = []

        async def plan():
            return AgentPlan(deliberation="Plan", agent_name="AgentY", objective="Obj", steps=["Do Y"])

        async def execute(p):
            return {"attempt": len(attempts) + 1}

        async def verify(res):
            attempts.append(len(attempts) + 1)
            is_valid = len(attempts) >= 3
            return AgentVerificationRecord(
                deliberation="Verification check",
                agent_name="AgentY",
                passed=is_valid,
                violations=[] if is_valid else ["Syntax error attempt"],
            )

        output, envelope = await orch.execute_pev_cycle("AgentY", plan, execute, verify, max_retries=5)
        assert len(attempts) == 3
        assert output["attempt"] == 3
        assert isinstance(envelope, ArtifactMetadataEnvelope)
        assert len(envelope.sha256_hash) == 64
        assert envelope.source_agent == "AgentY"

    @pytest.mark.anyio
    async def test_pev_empty_plan_steps_immediate_rejection(self) -> None:
        """Empty plan steps raises ValueError prior to execution dispatch (both via Pydantic validator and orchestrator check)."""
        orch = StudioOrchestrator()
        exec_called = False

        class CustomPlan:
            deliberation = "Custom"
            steps: List[str] = []

        async def plan():
            return CustomPlan()

        async def execute(p):
            nonlocal exec_called
            exec_called = True
            return {}

        async def verify(r):
            return AgentVerificationRecord(deliberation="V", agent_name="AgentZ", passed=True)

        with pytest.raises(ValueError) as exc_info:
            await orch.execute_pev_cycle("AgentZ", plan, execute, verify)

        assert "Plan contains 0 action steps" in str(exc_info.value)
        assert exec_called is False

        # Also test Pydantic validator rejection on empty steps
        with pytest.raises(ValueError) as exc_info2:
            AgentPlan(deliberation="Empty", agent_name="AgentZ", objective="None", steps=[])
        assert "Plan must contain at least one step" in str(exc_info2.value)


# ===========================================================================
# 2. SWE Agent AST Security & Anti-Template Check
# ===========================================================================

class TestSWEAgentASTSecurityAdversarial:
    """Stress tests AST security sandbox and anti-template rejection."""
    deliberation: str = "Empirical verification of AST security barriers and zero-template enforcement."

    def test_anti_template_markers_exhaustive_rejection(self) -> None:
        """Synthesized code containing {TEMPLATE_ or {{LEVEL_CONFIG}} is rejected immediately."""
        swe = LeadGameplaySWEAgent()

        template_payloads = [
            'x = "{TEMPLATE_LEVEL_CONFIG}"',
            'player_speed = "{TEMPLATE_SPEED}"',
            'class Foo:\n    deliberation = "Ok"\n    cfg = "{{LEVEL_CONFIG}}"',
            '# Comment with {TEMPLATE_MARKER}\nx = 1',
            '"""Docstring with {{LEVEL_CONFIG}}"""\nx = 1',
        ]

        for code in template_payloads:
            valid, err = swe.verify_code_module("test_module.py", code)
            assert valid is False, f"Payload was not rejected: {code}"
            assert "Detected static string template placeholders" in err

    def test_forbidden_imports_blocked_by_ast(self) -> None:
        """Forbidden system, process, and introspection modules are blocked by validate_cartridge_ast."""
        forbidden_modules = [
            "os",
            "sys",
            "subprocess",
            "inspect",
            "importlib",
            "ctypes",
            "shutil",
            "builtins",
            "socket",
            "pickle",
            "traceback",
            "asyncio",
            "threading",
            "multiprocessing",
        ]

        for mod in forbidden_modules:
            # Direct import
            code1 = f"import {mod}"
            with pytest.raises(ASTSecurityError) as exc_info1:
                validate_cartridge_ast(code1)
            assert f"Prohibited module import: '{mod}'" in str(exc_info1.value)

            # From import
            code2 = f"from {mod} import something"
            with pytest.raises(ASTSecurityError) as exc_info2:
                validate_cartridge_ast(code2)
            assert f"Prohibited module import: '{mod}'" in str(exc_info2.value) or "Prohibited" in str(exc_info2.value)

    def test_introspection_and_dunder_sandbox_escapes_blocked(self) -> None:
        """Introspection attributes and dunder sandbox escape primitives are blocked."""
        dangerous_snippets = [
            ("dunder subclasses", "x = ().__class__.__bases__[0].__subclasses__()"),
            ("dunder globals", "f.__globals__['os']"),
            ("dunder code", "f.__code__"),
            ("dunder mro", "int.__mro__"),
            ("eval call", "eval('1 + 1')"),
            ("exec call", "exec('x = 2')"),
            ("__import__ call", "__import__('os')"),
            ("compile call", "compile('x = 1', '', 'exec')"),
        ]

        for name, snippet in dangerous_snippets:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(snippet)
            assert "AST Security violation" in str(exc_info.value)

    def test_dynamic_reflection_getattr_evasion_blocked(self) -> None:
        """Dynamic getattr evasion attempts are blocked."""
        getattr_snippets = [
            "getattr(x, '__subclasses__')",
            "getattr(x, 'os')",
            "getattr(x, 'eval')",
            "getattr(x, 'sys')",
            "getattr(x, dynamic_variable)",  # Non-string literal must be rejected
        ]

        for snippet in getattr_snippets:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(snippet)
            assert "Prohibited dynamic attribute access" in str(exc_info.value)

    def test_verify_code_module_blocks_forbidden_import_via_ast_error(self) -> None:
        """SWE verify_code_module propagates ASTSecurityError when code has forbidden import."""
        swe = LeadGameplaySWEAgent()
        with pytest.raises(ASTSecurityError) as exc_info:
            swe.verify_code_module("rogue.py", "import os\nclass Rogue:\n    deliberation: str = 'bad'")
        assert "Prohibited module import: 'os'" in str(exc_info.value)


# ===========================================================================
# 3. Audio Producer & Visual Designer Bounds Adversarial Harness
# ===========================================================================

class MockDurationAudioBackend(AudioSynthesisBackend):
    """Custom mock audio backend returning arbitrary specified duration in seconds."""

    def __init__(self, duration: float):
        self.duration = duration

    def synthesize_speech(
        self,
        text: str,
        voice_name: str,
        output_path: Path,
        sample_rate: int = 44100,
        bark_id: str = "bark",
    ) -> AudioSynthesisResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Create minimal PCM data corresponding to self.duration
        num_samples = max(1, int(self.duration * sample_rate))
        pcm = b"\x00\x00" * num_samples
        header = create_wav_header(len(pcm), sample_rate=sample_rate, channels=1, bits_per_sample=16)
        full_wav = header + pcm
        output_path.write_bytes(full_wav)
        return AudioSynthesisResult(
            deliberation=f"Synthesized duration {self.duration}s",
            bark_id=bark_id,
            file_path=str(output_path),
            sample_rate=sample_rate,
            duration_seconds=self.duration,
            file_size_bytes=len(full_wav),
            sha256_hash="deadbeef",
        )


class TestAudioProducerBoundsAdversarial:
    """Stress tests audio duration bounds, corrupted WAV headers, and SSML sanitization."""
    deliberation: str = "Empirical verification of audio bounds [0.1s, 30.0s] and header validation."

    def test_ultra_short_audio_rejection(self) -> None:
        """Audio duration < 0.1s (e.g. 0.05s) raises VoiceSynthesisError."""
        producer = VoiceActingAudioProducerAgent(backend=MockDurationAudioBackend(0.05))
        bark = DialogueBarkSpec(
            deliberation="Short bark",
            bark_id="short_1",
            speaker_id="hero",
            speaker_name="Hero",
            trigger_type="test",
            text="Halt!",
            voice_clip="assets/audio/short.wav",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "short.wav"
            with pytest.raises(VoiceSynthesisError) as exc_info:
                producer.synthesize_voice_line(bark, out_file)
            assert "out of acceptable bounds [0.1s, 30.0s]" in str(exc_info.value)

    def test_over_length_audio_rejection(self) -> None:
        """Audio duration > 30.0s (e.g. 35.0s) raises VoiceSynthesisError."""
        producer = VoiceActingAudioProducerAgent(backend=MockDurationAudioBackend(35.0))
        bark = DialogueBarkSpec(
            deliberation="Long bark",
            bark_id="long_1",
            speaker_id="boss",
            speaker_name="Overlord",
            trigger_type="test",
            text="Long monologue beyond acceptable duration limit...",
            voice_clip="assets/audio/long.wav",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "long.wav"
            with pytest.raises(VoiceSynthesisError) as exc_info:
                producer.synthesize_voice_line(bark, out_file)
            assert "out of acceptable bounds [0.1s, 30.0s]" in str(exc_info.value)

    def test_exact_duration_bounds_acceptance(self) -> None:
        """Audio duration exactly at boundaries (0.10s and 30.00s) passes validation."""
        for bound in [0.10, 30.00]:
            producer = VoiceActingAudioProducerAgent(backend=MockDurationAudioBackend(bound))
            bark = DialogueBarkSpec(
                deliberation=f"Bound {bound} bark",
                bark_id=f"bound_{int(bound)}",
                speaker_id="hero",
                speaker_name="Hero",
                trigger_type="test",
                text="Testing boundary.",
                voice_clip=f"assets/audio/bound_{int(bound)}.wav",
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                out_file = Path(tmpdir) / f"bound_{bound}.wav"
                result = producer.synthesize_voice_line(bark, out_file)
                assert result.duration_seconds == pytest.approx(bound, rel=1e-3)

    def test_corrupted_wav_headers_detection(self) -> None:
        """validate_wav_file identifies truncated files and corrupted RIFF/WAVE/fmt headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tdir = Path(tmpdir)

            # 1. Truncated (< 44 bytes)
            p_trunc = tdir / "truncated.wav"
            p_trunc.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
            valid, err, _ = validate_wav_file(p_trunc)
            assert valid is False
            assert "smaller than 44-byte RIFF header" in err

            # 2. Corrupted RIFF magic
            p_badriff = tdir / "badriff.wav"
            hdr_badriff = bytearray(create_wav_header(100))
            hdr_badriff[:4] = b"CORR"
            p_badriff.write_bytes(hdr_badriff)
            valid, err, _ = validate_wav_file(p_badriff)
            assert valid is False
            assert "Invalid RIFF magic bytes" in err

            # 3. Corrupted WAVE format marker
            p_badwave = tdir / "badwave.wav"
            hdr_badwave = bytearray(create_wav_header(100))
            hdr_badwave[8:12] = b"FLAC"
            p_badwave.write_bytes(hdr_badwave)
            valid, err, _ = validate_wav_file(p_badwave)
            assert valid is False
            assert "Invalid WAVE format marker" in err

            # 4. Corrupted fmt subchunk marker
            p_badfmt = tdir / "badfmt.wav"
            hdr_badfmt = bytearray(create_wav_header(100))
            hdr_badfmt[12:16] = b"data"
            p_badfmt.write_bytes(hdr_badfmt)
            valid, err, _ = validate_wav_file(p_badfmt)
            assert valid is False
            assert "Invalid fmt subchunk marker" in err

            # 5. Non-existent file
            valid, err, _ = validate_wav_file(tdir / "does_not_exist.wav")
            assert valid is False
            assert "does not exist" in err

    def test_malicious_ssml_special_characters_escaping(self) -> None:
        """Sanitizes malicious script tags, XML delimiters, and ampersands."""
        adversarial_text = '<script>alert("PWNED & EXPLOITED")</script>\'test\'<tag attr="val">'
        escaped = escape_ssml(adversarial_text)
        assert "<" not in escaped
        assert ">" not in escaped
        assert '"' not in escaped
        assert "'" not in escaped
        assert "&lt;script&gt;" in escaped
        assert "&amp;" in escaped
        assert "&quot;" in escaped
        assert "&apos;" in escaped


class TestVisualDesignerPaletteQuantizationAdversarial:
    """Stress tests 16-color Euclidean quantization across extreme points and random RGB space."""
    deliberation: str = "Empirical verification of 100% adherence to STYLE_BIBLE_16_PALETTE."

    def test_extreme_rgb_colors_quantization(self) -> None:
        """Quantizes pure primary/secondary and corner RGB colors strictly into STYLE_BIBLE_16_PALETTE."""
        designer = VisualArtSpriteDesignerAgent()
        palette_set = set(STYLE_BIBLE_16_PALETTE)

        extreme_colors = [
            (0, 0, 0),        # Pure Black -> (0, 0, 0)
            (255, 255, 255),  # Pure White -> (255, 241, 232)
            (255, 0, 0),      # Pure Red -> (255, 0, 77)
            (0, 255, 0),      # Pure Green -> (0, 228, 54)
            (0, 0, 255),      # Pure Blue -> (41, 173, 255)
            (255, 255, 0),    # Pure Yellow -> (255, 236, 39)
            (0, 255, 255),    # Pure Cyan -> (41, 173, 255)
            (255, 0, 255),    # Pure Magenta -> (255, 119, 168)
            (128, 128, 128),  # Mid Gray
            (1, 1, 1),        # Near Black
            (254, 254, 254),  # Near White
        ]

        for rgb in extreme_colors:
            quantized = designer.quantize_color(rgb)
            assert quantized in palette_set, f"Color {rgb} quantized to {quantized} which is outside palette!"

    def test_fuzzy_rgb_space_palette_enclosure(self) -> None:
        """10,000 pseudo-random RGB points all quantize strictly into STYLE_BIBLE_16_PALETTE."""
        designer = VisualArtSpriteDesignerAgent()
        palette_set = set(STYLE_BIBLE_16_PALETTE)

        rng = random.Random(1337)
        for _ in range(10000):
            r = rng.randint(0, 255)
            g = rng.randint(0, 255)
            b = rng.randint(0, 255)
            quantized = designer.quantize_color((r, g, b))
            assert quantized in palette_set, f"Random RGB ({r}, {g}, {b}) mapped to non-palette color {quantized}"

    def test_quantize_image_to_indexed_transparency_and_palette(self) -> None:
        """Pillow image quantization maps RGBA to mode P with alpha transparency key 0."""
        from engine.assets.palette import quantize_image_to_indexed
        # Create 16x16 test RGBA image with transparent corner
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        # Set top-left pixel to transparent
        img.putpixel((0, 0), (0, 0, 0, 0))

        quantized_p = quantize_image_to_indexed(img, STYLE_BIBLE_16_PALETTE, transparent_index=0)
        assert quantized_p.mode == "P"
        assert quantized_p.info.get("transparency") == 0
        assert quantized_p.getpixel((0, 0)) == 0  # Transparent index


# ===========================================================================
# 7. Orchestrator Swarm Plan Execution, DAG Topologies & Staging Artifacts
# ===========================================================================

class TestExecuteSwarmPlanTopologiesAndArtifactsStress:
    """Empirical Stress Harness for StudioOrchestrator.execute_swarm_plan.
    Validates:
    1. Happy-path execution with rigorous audit of all staging artifacts:
       - world_spec.json
       - manifest.json
       - narrative_bible.json
       - narrative.json
       - mechanics.py (with injected probes and clean AST)
       - telemetry_manifest.json
       - assets/sprites/*.png (portrait and procedural sprites)
    2. Varying DAG Topologies:
       - 2-wave parallel topology (e.g. Director -> [SWE, TelemetryArchitect], Writer -> Visual)
       - 3-wave topology (Director -> [SWE, Writer] -> [TelemetryArchitect, Visual])
       - Disconnected components topology (3 disjoint subgraphs including isolated Analyst)
       - Degenerate single-node plan (Director only)
    3. Multimodal Voice Audio Synthesis (.wav) with RIFF/WAVE header and duration validation.
    4. Episodic Update Mode (target_episode=1) with UpdateWorldSpec.
    5. Causal Fail-Fast and Zero Silent Fallback (Rule 5) on cycle and agent exceptions.
    6. Supervisor Rejection Handling (transitions to SYNTHESIZING for self-healing repair).
    """
    deliberation: str = "Empirical stress-testing of execute_swarm_plan across topologies and artifact gates."

    @staticmethod
    def _get_world_spec_dict(target_episode: int = 0) -> Dict[str, Any]:
        return {
            "deliberation": "Formulating Genesis world specification for cybernetic survival roguelike.",
            "spec_id": f"world_spec_genesis_{target_episode:03d}",
            "target_episode": target_episode,
            "game_title": "Neon Grid Survival",
            "core_genre": "Action Roguelike",
            "thematic_premise": "A rogue operative purges corrupted subroutines across an infected mainframe grid.",
            "world_lore": "In 2099, the Grid core has succumbed to sentient malware. Operatives must neutralize nodes.",
            "conflict_tone": "grim_tense",
            "conflict_narrative": "Kael must infiltrate the security sector and disable the Sentinel Core.",
            "aesthetic_style": "16-color retro neon pixel art",
            "palette_mood": "neon_cyberpunk",
            "palette_indices": [0, 1, 8, 12],
            "gameplay_pacing": "kinetic_high_mobility",
            "pacing_score": 0.5,
            "target_difficulty": 1.0,
            "camera_style": "arena_framing",
            "mechanics_guidelines": [
                "Implement 8-directional movement",
                "Support directional projectile shooting",
                "Integrate telemetry event probes for survival duration and damage",
            ],
            "arena_layout_directive": "open_arena",
        }

    @staticmethod
    def _get_update_world_spec_dict(target_episode: int = 1) -> Dict[str, Any]:
        return {
            "deliberation": f"Episodic update {target_episode} adapting world lore and mechanics based on telemetry.",
            "spec_id": f"update_spec_ep{target_episode}_001",
            "target_episode": target_episode,
            "previous_episode": target_episode - 1,
            "narrative_progression": "The mainframe counter-measures evolved after the breach.",
            "escalation_factor": 0.10,
            "new_environmental_threats": ["Surge Nodes"],
            "pacing_adjustment": 0.05,
            "mechanics_evolution_directives": ["Introduce dash cooldown penalty"],
            "hypotheses_addressed": ["Players rely on dash evasion"],
        }

    @staticmethod
    def _get_narrative_bible_dict() -> Dict[str, Any]:
        return {
            "deliberation": "Formulating narrative lore, character sheets, cutscene script, and dialogue trees.",
            "bible_id": "narrative_genesis_001",
            "world_title": "Neon Grid Survival",
            "setting_summary": "Infected digital mainframe where rogue security processes attack all intrusions.",
            "characters": [
                {
                    "deliberation": "Protagonist character archetype with decisive, focused demeanor.",
                    "character_id": "hero",
                    "name": "Kael",
                    "role": "protagonist",
                    "personality_traits": ["calm", "decisive", "tenacious"],
                    "backstory": "Former mainframe engineer turned rogue infiltration operative.",
                    "voice_timbre": "focused",
                    "gemini_voice_name": "Aoede",
                    "visual_description": "Cybernetic operative in dark trenchcoat with glowing cyan visor.",
                    "portrait_asset": "assets/sprites/kael_portrait.png",
                }
            ],
            "cutscenes": [],
            "dialogue_trees": [
                {
                    "deliberation": "Terminal dialogue tree at episode entry.",
                    "root_node_id": "node_entry",
                    "nodes": {
                        "node_entry": {
                            "deliberation": "Entry briefing node.",
                            "node_id": "node_entry",
                            "speaker_id": "hero",
                            "text": "Mainframe connection established. Infiltration underway.",
                            "choices": [],
                        }
                    },
                }
            ],
            "dialogue_barks": [
                {
                    "deliberation": "Low health alert bark.",
                    "bark_id": "bark_critical_health",
                    "speaker_name": "Kael",
                    "speaker_id": "hero",
                    "trigger_type": "low_health",
                    "text": "System integrity compromised. Need extraction.",
                    "voice_clip": "assets/audio/bark_critical_health.wav",
                }
            ],
        }

    @staticmethod
    def _get_cartridge_bundle_dict(episode: int = 0) -> Dict[str, Any]:
        mechanics_source = (
            '"""mechanics.py - Synthesized Autonomous Game Cartridge."""\n'
            'from typing import Any, Dict\n'
            'from engine.api.contracts import BaseCartridge, BaseScene\n\n\n'
            'class GridSurvivalScene(BaseScene):\n'
            '    deliberation: str = "Primary gameplay scene implementing open-genre Pygame lifecycle."\n\n'
            '    def on_enter(self, **kwargs: Any) -> None:\n'
            '        self.active = True\n\n'
            '    def on_exit(self, **kwargs: Any) -> None:\n'
            '        self.active = False\n\n'
            '    def handle_event(self, event: Any) -> bool:\n'
            '        return False\n\n'
            '    def update(self, dt: float) -> None:\n'
            '        pass\n\n'
            '    def render(self, surface: Any) -> None:\n'
            '        pass\n\n\n'
            'class GridSurvivalCartridge(BaseCartridge):\n'
            '    deliberation: str = "Cartridge entrypoint subclassing BaseCartridge with open-genre scenes."\n\n'
            '    def get_scenes(self, engine: Any) -> Dict[str, Any]:\n'
            '        return {"main": GridSurvivalScene()}\n\n'
            '    def get_initial_scene(self, engine: Any) -> Any:\n'
            '        return GridSurvivalScene()\n'
        )
        return {
            "deliberation": "Formulating complete open-genre cartridge code bundle implementing BaseCartridge.",
            "manifest_data": {
                "title": "Neon Grid Survival",
                "genre": "Action Roguelike",
                "version": f"1.{episode}.0",
                "entrypoint": "mechanics:GridSurvivalCartridge",
            },
            "mechanics_code": mechanics_source,
            "scenes_code": None,
            "entities_code": None,
            "abilities_code": None,
            "level_data": None,
        }

    @staticmethod
    def _get_telemetry_plan_dict() -> Dict[str, Any]:
        return {
            "deliberation": "Formulating hypothesis-driven custom telemetry probes for player survival.",
            "plan_id": "tel_plan_genesis_001",
            "cartridge_id": "active",
            "probes": [
                {
                    "deliberation": "Probe tracking player survival tick duration in active combat.",
                    "probe_id": "probe_survival_duration",
                    "probe_name": "survival_tick_counter",
                    "category": "combat",
                    "description": "Records continuous player survival tick count per sector run.",
                    "hypothesis_target": "Players adopt evasion over aggressive engagements under high difficulty.",
                    "payload_schema": {
                        "type": "object",
                        "properties": {
                            "survival_ticks": {"type": "integer"},
                        },
                    },
                    "target_module": "mechanics.py",
                    "target_class": "GridSurvivalScene",
                    "target_method": "update",
                    "injection_location": "method_exit",
                    "max_event_rate_hz": 60.0,
                }
            ],
            "hypotheses_addressed": [
                "Players adopt evasion over aggressive engagements under high difficulty."
            ],
            "estimated_total_throughput_hz": 60.0,
        }

    @staticmethod
    def _get_test_png_bytes() -> bytes:
        buf = io.BytesIO()
        img = Image.new("RGBA", (16, 16), color=(0, 255, 200, 255))
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _get_test_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 44100) -> bytes:
        pcm = b"\x00\x00" * int(sample_rate * duration_sec)
        hdr = create_wav_header(len(pcm), sample_rate=sample_rate, channels=1, bits_per_sample=16)
        return hdr + pcm

    @staticmethod
    def _mock_supervisor(orch: StudioOrchestrator, promoted: bool = True) -> MagicMock:
        mock_sup = MagicMock()
        mock_report = MagicMock()
        mock_report.promoted_to_active = promoted
        mock_sup.evaluate_and_heal.return_value = mock_report
        orch.agents["QAPlaytesterBuildSupervisor"] = mock_sup
        return mock_sup

    @pytest.mark.anyio
    async def test_execute_swarm_plan_happy_path_artifact_completeness(
        self,
        tmp_path: Path,
        mock_gemini_client: MockGeminiClient,
    ) -> None:
        """Adversarially audits all 7 staging artifacts for happy-path execute_swarm_plan."""
        mock_gemini_client.set_json_response(self._get_world_spec_dict(0))
        mock_gemini_client.set_json_response(self._get_narrative_bible_dict())
        mock_gemini_client.set_json_response(self._get_cartridge_bundle_dict(0))
        mock_gemini_client.set_json_response(self._get_telemetry_plan_dict())
        mock_gemini_client.set_image_response(self._get_test_png_bytes())

        orch = StudioOrchestrator(client=mock_gemini_client)
        mock_sup = self._mock_supervisor(orch, promoted=True)

        plan = StudioSwarmPlan(
            deliberation="Adversarial audit of 5-role pipeline staging artifacts.",
            plan_id="plan_audit_artifacts_001",
            game_title="Neon Grid Survival",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director", "Writer", "SWE", "TelemetryArchitect", "Visual"],
            dag_edges=[
                ("Director", "Writer"),
                ("Writer", "SWE"),
                ("SWE", "TelemetryArchitect"),
                ("TelemetryArchitect", "Visual"),
            ],
            task_manifest={},
            target_episode=0,
        )

        staging_dir = tmp_path / "staging_artifact_audit"
        report = await orch.execute_swarm_plan(plan, staging_dir)

        assert report.promoted_to_active is True
        assert orch.state == StudioBuildState.IDLE

        # 1. world_spec.json
        world_spec_path = staging_dir / "world_spec.json"
        assert world_spec_path.is_file() and world_spec_path.stat().st_size > 0
        w_data = json.loads(world_spec_path.read_text(encoding="utf-8"))
        GenesisWorldSpec.model_validate(w_data)
        assert list(w_data.keys())[0] == "deliberation", "deliberation must be key 0"
        assert w_data["game_title"] == "Neon Grid Survival"
        assert w_data["core_genre"] == "Action Roguelike"

        # 2. manifest.json
        manifest_path = staging_dir / "manifest.json"
        assert manifest_path.is_file() and manifest_path.stat().st_size > 0
        m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("title", "genre", "version"):
            assert field in m_data, f"Field '{field}' missing from manifest.json"

        # 3. narrative_bible.json
        narrative_bible_path = staging_dir / "narrative_bible.json"
        assert narrative_bible_path.is_file() and narrative_bible_path.stat().st_size > 0
        nb_data = json.loads(narrative_bible_path.read_text(encoding="utf-8"))
        NarrativeBible.model_validate(nb_data)
        assert len(nb_data["characters"]) >= 1
        assert len(nb_data["dialogue_trees"]) >= 1
        assert len(nb_data["dialogue_barks"]) >= 1

        # 4. narrative.json
        narrative_json_path = staging_dir / "narrative.json"
        assert narrative_json_path.is_file() and narrative_json_path.stat().st_size > 0
        n_data = json.loads(narrative_json_path.read_text(encoding="utf-8"))
        assert "dialogue_trees" in n_data and "dialogue_tree" in n_data
        assert n_data["dialogue_trees"] == n_data["dialogue_tree"]

        # 5. mechanics.py
        mechanics_path = staging_dir / "mechanics.py"
        assert mechanics_path.is_file() and mechanics_path.stat().st_size > 0
        m_src = mechanics_path.read_text(encoding="utf-8")
        parsed_ast = ast.parse(m_src)
        assert isinstance(parsed_ast, ast.Module)
        assert "BaseCartridge" in m_src
        assert "GridSurvivalCartridge" in m_src
        assert "survival_tick_counter" in m_src or "AgentTelemetryProbe" in m_src

        # 6. telemetry_manifest.json
        telemetry_manifest_path = staging_dir / "telemetry_manifest.json"
        assert telemetry_manifest_path.is_file() and telemetry_manifest_path.stat().st_size > 0
        t_data = json.loads(telemetry_manifest_path.read_text(encoding="utf-8"))
        TelemetryArchitecturePlan.model_validate(t_data)
        assert len(t_data["probes"]) >= 1
        assert t_data["probes"][0]["probe_name"] == "survival_tick_counter"

        # 7. Sprite assets
        portrait_path = staging_dir / "assets" / "sprites" / "kael_portrait.png"
        assert portrait_path.is_file() and portrait_path.stat().st_size > 0
        p_bytes = portrait_path.read_bytes()
        assert p_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Portrait must be valid PNG"
        with Image.open(portrait_path) as im:
            assert im.format == "PNG"

        # Procedural cartridge sprites
        for sprite_name in ("player_idle.png", "chaser.png", "boss.png", "spikes.png"):
            s_path = staging_dir / "assets" / "sprites" / sprite_name
            assert s_path.is_file() and s_path.stat().st_size > 0

        mock_sup.evaluate_and_heal.assert_called_once()

    @pytest.mark.anyio
    async def test_execute_swarm_plan_two_wave_topology(
        self,
        tmp_path: Path,
        mock_gemini_client: MockGeminiClient,
    ) -> None:
        """Executes a 2-wave parallel DAG: Wave 0 [Director, Writer], Wave 1 [SWE, TelemetryArchitect, Visual]."""
        # Tier 0 execution: Director, Writer
        mock_gemini_client.set_json_response(self._get_world_spec_dict(0))
        mock_gemini_client.set_json_response(self._get_narrative_bible_dict())
        # Tier 1 execution: SWE, TelemetryArchitect, Visual
        mock_gemini_client.set_json_response(self._get_cartridge_bundle_dict(0))
        mock_gemini_client.set_json_response(self._get_telemetry_plan_dict())
        mock_gemini_client.set_image_response(self._get_test_png_bytes())

        orch = StudioOrchestrator(client=mock_gemini_client)
        self._mock_supervisor(orch, promoted=True)

        plan = StudioSwarmPlan(
            deliberation="2-wave DAG topology verification.",
            plan_id="plan_two_wave_001",
            game_title="Neon Grid Survival",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director", "Writer", "SWE", "TelemetryArchitect", "Visual"],
            dag_edges=[
                ("Director", "SWE"),
                ("Director", "TelemetryArchitect"),
                ("Writer", "Visual"),
            ],
            task_manifest={},
            target_episode=0,
        )

        tiers = orch._compute_execution_tiers(plan.dag_edges, plan.specialist_agent_roles)
        assert len(tiers) == 2, f"Expected exactly 2 waves, got {len(tiers)}: {tiers}"
        assert tiers[0] == ["Director", "Writer"]
        assert tiers[1] == ["SWE", "TelemetryArchitect", "Visual"]

        staging_dir = tmp_path / "staging_two_wave"
        report = await orch.execute_swarm_plan(plan, staging_dir)

        assert report.promoted_to_active is True
        assert (staging_dir / "world_spec.json").is_file()
        assert (staging_dir / "manifest.json").is_file()
        assert (staging_dir / "narrative_bible.json").is_file()
        assert (staging_dir / "mechanics.py").is_file()
        assert (staging_dir / "telemetry_manifest.json").is_file()
        assert (staging_dir / "assets" / "sprites" / "kael_portrait.png").is_file()

    @pytest.mark.anyio
    async def test_execute_swarm_plan_three_wave_topology(
        self,
        tmp_path: Path,
        mock_gemini_client: MockGeminiClient,
    ) -> None:
        """Executes a 3-wave DAG: Wave 0 [Director] -> Wave 1 [SWE, Writer] -> Wave 2 [TelemetryArchitect, Visual]."""
        # Tier 0: Director
        mock_gemini_client.set_json_response(self._get_world_spec_dict(0))
        # Tier 1: SWE, Writer
        mock_gemini_client.set_json_response(self._get_cartridge_bundle_dict(0))
        mock_gemini_client.set_json_response(self._get_narrative_bible_dict())
        # Tier 2: TelemetryArchitect, Visual
        mock_gemini_client.set_json_response(self._get_telemetry_plan_dict())
        mock_gemini_client.set_image_response(self._get_test_png_bytes())

        orch = StudioOrchestrator(client=mock_gemini_client)
        self._mock_supervisor(orch, promoted=True)

        plan = StudioSwarmPlan(
            deliberation="3-wave DAG topology verification.",
            plan_id="plan_three_wave_001",
            game_title="Neon Grid Survival",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director", "Writer", "SWE", "TelemetryArchitect", "Visual"],
            dag_edges=[
                ("Director", "Writer"),
                ("Director", "SWE"),
                ("SWE", "TelemetryArchitect"),
                ("Writer", "Visual"),
            ],
            task_manifest={},
            target_episode=0,
        )

        tiers = orch._compute_execution_tiers(plan.dag_edges, plan.specialist_agent_roles)
        assert len(tiers) == 3, f"Expected exactly 3 waves, got {len(tiers)}: {tiers}"
        assert tiers[0] == ["Director"]
        assert tiers[1] == ["SWE", "Writer"]
        assert tiers[2] == ["TelemetryArchitect", "Visual"]

        staging_dir = tmp_path / "staging_three_wave"
        report = await orch.execute_swarm_plan(plan, staging_dir)

        assert report.promoted_to_active is True
        assert (staging_dir / "world_spec.json").is_file()
        assert (staging_dir / "narrative_bible.json").is_file()
        assert (staging_dir / "mechanics.py").is_file()
        assert (staging_dir / "telemetry_manifest.json").is_file()
        assert (staging_dir / "assets" / "sprites" / "kael_portrait.png").is_file()

    @pytest.mark.anyio
    async def test_execute_swarm_plan_disconnected_components_topology(
        self,
        tmp_path: Path,
        mock_gemini_client: MockGeminiClient,
    ) -> None:
        """Executes a DAG with disconnected components:
        Component 1: Director -> SWE -> TelemetryArchitect
        Component 2: Writer -> Visual
        Component 3: Analyst (isolated)
        """
        # Tier 0: Analyst (no call), Director, Writer
        mock_gemini_client.set_json_response(self._get_world_spec_dict(0))
        mock_gemini_client.set_json_response(self._get_narrative_bible_dict())
        # Tier 1: SWE, Visual
        mock_gemini_client.set_json_response(self._get_cartridge_bundle_dict(0))
        mock_gemini_client.set_image_response(self._get_test_png_bytes())
        # Tier 2: TelemetryArchitect
        mock_gemini_client.set_json_response(self._get_telemetry_plan_dict())

        orch = StudioOrchestrator(client=mock_gemini_client)
        self._mock_supervisor(orch, promoted=True)

        plan = StudioSwarmPlan(
            deliberation="Disconnected components DAG topology verification.",
            plan_id="plan_disconnected_001",
            game_title="Neon Grid Survival",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Analyst", "Director", "SWE", "TelemetryArchitect", "Visual", "Writer"],
            dag_edges=[
                ("Director", "SWE"),
                ("SWE", "TelemetryArchitect"),
                ("Writer", "Visual"),
            ],
            task_manifest={},
            target_episode=0,
        )

        tiers = orch._compute_execution_tiers(plan.dag_edges, plan.specialist_agent_roles)
        assert len(tiers) == 3
        assert tiers[0] == ["Analyst", "Director", "Writer"]
        assert tiers[1] == ["SWE", "Visual"]
        assert tiers[2] == ["TelemetryArchitect"]

        staging_dir = tmp_path / "staging_disconnected"
        report = await orch.execute_swarm_plan(plan, staging_dir)

        assert report.promoted_to_active is True
        assert (staging_dir / "world_spec.json").is_file()
        assert (staging_dir / "narrative_bible.json").is_file()
        assert (staging_dir / "mechanics.py").is_file()
        assert (staging_dir / "telemetry_manifest.json").is_file()
        assert (staging_dir / "assets" / "sprites" / "kael_portrait.png").is_file()

    @pytest.mark.anyio
    async def test_execute_swarm_plan_single_node_degenerate_dag(
        self,
        tmp_path: Path,
        mock_gemini_client: MockGeminiClient,
    ) -> None:
        """Executes a degenerate 1-node DAG with 0 edges."""
        mock_gemini_client.set_json_response(self._get_world_spec_dict(0))

        orch = StudioOrchestrator(client=mock_gemini_client)
        self._mock_supervisor(orch, promoted=True)

        plan = StudioSwarmPlan(
            deliberation="Single node degenerate DAG verification.",
            plan_id="plan_single_node_001",
            game_title="Neon Grid Survival",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director"],
            dag_edges=[],
            task_manifest={},
            target_episode=0,
        )

        tiers = orch._compute_execution_tiers(plan.dag_edges, plan.specialist_agent_roles)
        assert len(tiers) == 1
        assert tiers[0] == ["Director"]

        staging_dir = tmp_path / "staging_single_node"
        report = await orch.execute_swarm_plan(plan, staging_dir)

        assert report.promoted_to_active is True
        assert (staging_dir / "world_spec.json").is_file()
        assert (staging_dir / "manifest.json").is_file()

    @pytest.mark.anyio
    async def test_execute_swarm_plan_with_voice_audio_synthesis(
        self,
        tmp_path: Path,
        mock_gemini_client: MockGeminiClient,
    ) -> None:
        """Executes a 6-role plan including VoiceActingAudioProducerAgent synthesizing .wav voice clips."""
        # Tier 0: Director
        mock_gemini_client.set_json_response(self._get_world_spec_dict(0))
        # Tier 1: SWE, Writer
        mock_gemini_client.set_json_response(self._get_cartridge_bundle_dict(0))
        mock_gemini_client.set_json_response(self._get_narrative_bible_dict())
        # Tier 2: TelemetryArchitect, Visual, Voice
        mock_gemini_client.set_json_response(self._get_telemetry_plan_dict())
        mock_gemini_client.set_image_response(self._get_test_png_bytes())
        mock_gemini_client.set_audio_response(self._get_test_wav_bytes(duration_sec=1.5))

        orch = StudioOrchestrator(client=mock_gemini_client)
        self._mock_supervisor(orch, promoted=True)

        plan = StudioSwarmPlan(
            deliberation="6-role swarm with voice acting audio synthesis.",
            plan_id="plan_voice_audio_001",
            game_title="Neon Grid Survival",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director", "Writer", "Voice", "Visual", "SWE", "TelemetryArchitect"],
            dag_edges=[
                ("Director", "Writer"),
                ("Director", "SWE"),
                ("Writer", "Voice"),
                ("Writer", "Visual"),
                ("SWE", "TelemetryArchitect"),
            ],
            task_manifest={},
            target_episode=0,
        )

        tiers = orch._compute_execution_tiers(plan.dag_edges, plan.specialist_agent_roles)
        assert len(tiers) == 3
        assert tiers[0] == ["Director"]
        assert tiers[1] == ["SWE", "Writer"]
        assert tiers[2] == ["TelemetryArchitect", "Visual", "Voice"]

        staging_dir = tmp_path / "staging_voice_audio"
        report = await orch.execute_swarm_plan(plan, staging_dir)

        assert report.promoted_to_active is True
        audio_path = staging_dir / "assets" / "audio" / "bark_critical_health.wav"
        assert audio_path.is_file(), "bark_critical_health.wav must be synthesized"
        assert audio_path.stat().st_size > 44, "Audio file must exceed WAV header size"
        assert audio_path.read_bytes()[:4] == b"RIFF", "Audio file must start with RIFF header"

        valid, err, duration = validate_wav_file(audio_path)
        assert valid is True, f"Synthesized WAV validation failed: {err}"
        assert 0.1 <= duration <= 30.0

    @pytest.mark.anyio
    async def test_execute_swarm_plan_episodic_update_director_standalone(
        self,
        tmp_path: Path,
        mock_gemini_client: MockGeminiClient,
    ) -> None:
        """Executes an episodic update (target_episode=1) with Director producing UpdateWorldSpec."""
        mock_gemini_client.set_json_response(self._get_update_world_spec_dict(1))

        orch = StudioOrchestrator(client=mock_gemini_client)
        self._mock_supervisor(orch, promoted=True)

        plan = StudioSwarmPlan(
            deliberation="Episodic update 1 worldbuilding swarm plan.",
            plan_id="plan_episodic_ep1_001",
            game_title="Neon Grid Survival",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director"],
            dag_edges=[],
            task_manifest={},
            target_episode=1,
        )

        staging_dir = tmp_path / "staging_episodic_ep1"
        report = await orch.execute_swarm_plan(plan, staging_dir)

        assert report.promoted_to_active is True
        manifest_data = json.loads((staging_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest_data["episode"] == 1

        world_spec_data = json.loads((staging_dir / "world_spec.json").read_text(encoding="utf-8"))
        assert world_spec_data["target_episode"] == 1

    @pytest.mark.anyio
    async def test_execute_swarm_plan_episodic_with_swe_surfaces_contract_gap(
        self,
        tmp_path: Path,
        mock_gemini_client: MockGeminiClient,
    ) -> None:
        """Verifies that when LeadGameplaySWEAgent runs with UpdateWorldSpec during target_episode > 0
        without a queued track synthesis response, it surfaces a StudioSwarmExecutionError cleanly.
        """
        mock_gemini_client.set_json_response(self._get_update_world_spec_dict(1))

        orch = StudioOrchestrator(client=mock_gemini_client)
        self._mock_supervisor(orch, promoted=True)

        plan = StudioSwarmPlan(
            deliberation="Episodic update 1 with SWE.",
            plan_id="plan_episodic_swe_001",
            game_title="Neon Grid Survival",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director", "SWE"],
            dag_edges=[("Director", "SWE")],
            task_manifest={},
            target_episode=1,
        )

        staging_dir = tmp_path / "staging_episodic_swe"
        with pytest.raises(StudioSwarmExecutionError) as exc_info:
            await orch.execute_swarm_plan(plan, staging_dir)

        assert "Specialist 'SWE' failed" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_swarm_plan_cyclic_dag_fails_fast_circular_dependency(
        self,
        tmp_path: Path,
        mock_gemini_client: MockGeminiClient,
    ) -> None:
        """Adversarial cyclic swarm plan fails fast with CircularDependencyError before executing specialists."""
        orch = StudioOrchestrator(client=mock_gemini_client)

        plan = StudioSwarmPlan(
            deliberation="Adversarial cyclic swarm plan.",
            plan_id="plan_cyclic_001",
            game_title="Neon Grid Survival",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director", "SWE"],
            dag_edges=[
                ("Director", "SWE"),
                ("SWE", "Director"),
            ],
            task_manifest={},
            target_episode=0,
        )

        staging_dir = tmp_path / "staging_cyclic"
        with pytest.raises(CircularDependencyError) as exc_info:
            await orch.execute_swarm_plan(plan, staging_dir)

        assert "Cycle detected" in str(exc_info.value)
        assert not (staging_dir / "world_spec.json").exists()

    @pytest.mark.anyio
    async def test_execute_swarm_plan_specialist_failure_fails_fast_zero_silent_fallbacks(
        self,
        tmp_path: Path,
        mock_gemini_client: MockGeminiClient,
    ) -> None:
        """Specialist failure during swarm execution fails fast with StudioSwarmExecutionError (Rule 5)."""
        mock_gemini_client.set_error(RuntimeError("Gemini live connection timeout"))

        orch = StudioOrchestrator(client=mock_gemini_client)

        plan = StudioSwarmPlan(
            deliberation="Adversarial plan triggering specialist failure.",
            plan_id="plan_failing_specialist_001",
            game_title="Neon Grid Survival",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director", "SWE"],
            dag_edges=[("Director", "SWE")],
            task_manifest={},
            target_episode=0,
        )

        staging_dir = tmp_path / "staging_failing_specialist"
        with pytest.raises(StudioSwarmExecutionError) as exc_info:
            await orch.execute_swarm_plan(plan, staging_dir)

        assert "Specialist 'Director' failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    @pytest.mark.anyio
    async def test_execute_swarm_plan_supervisor_rejection_reenters_synthesizing(
        self,
        tmp_path: Path,
        mock_gemini_client: MockGeminiClient,
    ) -> None:
        """When BuildSupervisor rejects promotion, orchestrator state transitions to SYNTHESIZING."""
        mock_gemini_client.set_json_response(self._get_world_spec_dict(0))

        orch = StudioOrchestrator(client=mock_gemini_client)
        self._mock_supervisor(orch, promoted=False)

        plan = StudioSwarmPlan(
            deliberation="Plan testing supervisor rejection state transition.",
            plan_id="plan_sup_rejection_001",
            game_title="Neon Grid Survival",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director"],
            dag_edges=[],
            task_manifest={},
            target_episode=0,
        )

        staging_dir = tmp_path / "staging_sup_rejection"
        report = await orch.execute_swarm_plan(plan, staging_dir)

        assert report.promoted_to_active is False
        assert orch.state == StudioBuildState.SYNTHESIZING, "Must re-enter SYNTHESIZING on rejected build"


# ===========================================================================
# 8. Milestone 3 Empirical Challenger 1 Stress Harness
# ===========================================================================

class TestMilestone3Challenger1EmpiricalStress:
    """Empirical verification of Milestone 3 requirements:
    1. DAG topological ordering via Kahn's algorithm guarantees Visual & Voice strictly before SWE.
    2. Ad-hoc voice gating omits voice synthesis when Voice role is omitted from plan.
    3. Iterative SWE loop status transitions (pending -> in_progress -> completed/rejected).
    4. Epistemic deliberation ordering (field index 0) across all emitted swarm events.
    5. Immediate fail-fast halting upon track failure (AST, syntax, headless simulation).
    """
    deliberation: str = "Empirical stress testing for Milestone 3 DAG ordering, voice gating, and iterative SWE loop."

    def test_dag_kahn_topological_sort_places_visual_and_voice_strictly_before_swe(self) -> None:
        """Kahn's topological sort on DEFAULT_EDGES invariant: Visual and Voice must strictly precede SWE."""
        dag = StudioAgentDAG()
        order = dag.topological_sort()

        # Invariant 1: Visual and Voice strictly precede SWE in linear topological sort
        assert "Visual" in order
        assert "Voice" in order
        assert "SWE" in order

        visual_idx = order.index("Visual")
        voice_idx = order.index("Voice")
        swe_idx = order.index("SWE")
        director_idx = order.index("Director")
        writer_idx = order.index("Writer")

        assert visual_idx < swe_idx, f"Visual (index {visual_idx}) must precede SWE (index {swe_idx})"
        assert voice_idx < swe_idx, f"Voice (index {voice_idx}) must precede SWE (index {swe_idx})"
        assert director_idx < visual_idx, f"Director (index {director_idx}) must precede Visual (index {visual_idx})"
        assert writer_idx < voice_idx, f"Writer (index {writer_idx}) must precede Voice (index {voice_idx})"
        assert writer_idx < visual_idx, f"Writer (index {writer_idx}) must precede Visual (index {visual_idx})"

        # Permutation stress test: 100 randomized edge orderings must all produce Visual & Voice before SWE
        rng = random.Random(42)
        edges_copy = list(StudioAgentDAG.DEFAULT_EDGES)
        for trial in range(100):
            rng.shuffle(edges_copy)
            trial_dag = StudioAgentDAG(edges=edges_copy)
            trial_order = trial_dag.topological_sort()
            t_vis = trial_order.index("Visual")
            t_vox = trial_order.index("Voice")
            t_swe = trial_order.index("SWE")
            assert t_vis < t_swe, f"Trial {trial}: Visual must precede SWE"
            assert t_vox < t_swe, f"Trial {trial}: Voice must precede SWE"

        # Tier wave partitioning test: Visual and Voice must be partitioned into an earlier wave tier than SWE
        orch = StudioOrchestrator()
        all_roles = [
            "Analyst", "Director", "Writer", "Visual", "Voice", "SWE", "TelemetryArchitect", "Supervisor"
        ]
        tiers = orch._compute_execution_tiers(StudioAgentDAG.DEFAULT_EDGES, all_roles)

        role_tier_map = {}
        for tier_idx, wave in enumerate(tiers):
            for role in wave:
                role_tier_map[role] = tier_idx

        assert role_tier_map["Visual"] < role_tier_map["SWE"], "Visual tier must be strictly earlier than SWE tier"
        assert role_tier_map["Voice"] < role_tier_map["SWE"], "Voice tier must be strictly earlier than SWE tier"
        assert role_tier_map["Director"] < role_tier_map["Visual"], "Director must precede Visual tier"
        assert role_tier_map["Writer"] < role_tier_map["Voice"], "Writer must precede Voice tier"

        # Subsets of roles: verify invariant holds even when optional roles are omitted
        subset_a = ["Director", "Visual", "Voice", "SWE"]
        tiers_a = orch._compute_execution_tiers(StudioAgentDAG.DEFAULT_EDGES, subset_a)
        tier_map_a = {r: idx for idx, wave in enumerate(tiers_a) for r in wave}
        assert tier_map_a["Visual"] < tier_map_a["SWE"]
        assert tier_map_a["Voice"] < tier_map_a["SWE"]

        subset_b = ["Visual", "SWE"]
        tiers_b = orch._compute_execution_tiers(StudioAgentDAG.DEFAULT_EDGES, subset_b)
        tier_map_b = {r: idx for idx, wave in enumerate(tiers_b) for r in wave}
        assert tier_map_b["Visual"] < tier_map_b["SWE"]

    @pytest.mark.anyio
    async def test_ad_hoc_voice_gating_when_voice_omitted_vs_included(
        self,
        tmp_path: Path,
    ) -> None:
        """When plan.specialist_agent_roles excludes Voice, zero voice synthesis calls occur.
        When Voice is included, voice synthesis executes properly."""
        mock_client = MockGeminiClient()
        director_dict = TestExecuteSwarmPlanTopologiesAndArtifactsStress._get_world_spec_dict(0)
        writer_dict = TestExecuteSwarmPlanTopologiesAndArtifactsStress._get_narrative_bible_dict()
        swe_bundle_dict = TestExecuteSwarmPlanTopologiesAndArtifactsStress._get_cartridge_bundle_dict(0)

        # 1. PLAN WITHOUT VOICE
        mock_client.set_json_response(director_dict)
        mock_client.set_json_response(writer_dict)
        mock_client.set_json_response(swe_bundle_dict)

        orch = StudioOrchestrator(client=mock_client)

        # Mock supervisor so promotion succeeds
        mock_sup = MagicMock()
        mock_report = MagicMock()
        mock_report.promoted_to_active = True
        mock_report.deliberation = "Passed all QA gates"
        mock_report.attempt_count = 1
        mock_sup.evaluate_and_heal.return_value = mock_report
        orch.agents["QAPlaytesterBuildSupervisor"] = mock_sup

        # Spy on voice agent
        voice_agent = orch.agents["VoiceActingAudioProducerAgent"]
        voice_spy = MagicMock(wraps=voice_agent.synthesize_voice_line)
        voice_agent.synthesize_voice_line = voice_spy

        plan_no_voice = StudioSwarmPlan(
            deliberation="Swarm plan explicitly excluding Voice specialist.",
            plan_id="plan_no_voice_001",
            game_title="Voice Gating Test",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director", "Writer", "SWE"],
            dag_edges=[("Director", "Writer"), ("Writer", "SWE")],
            task_manifest={},
            target_episode=0,
        )

        staging_dir_no_voice = tmp_path / "staging_no_voice"
        report = await orch.execute_swarm_plan(plan_no_voice, staging_dir_no_voice)

        assert report.promoted_to_active is True
        # VERIFY ZERO VOICE CALLS
        assert voice_spy.call_count == 0, "No voice synthesis calls should occur when Voice is omitted"
        audio_dir = staging_dir_no_voice / "assets" / "audio"
        if audio_dir.exists():
            wav_files = list(audio_dir.glob("*.wav"))
            assert len(wav_files) == 0, "No WAV files should be generated when Voice is omitted"

        # 2. PLAN WITH VOICE (Using role slug "Voice")
        mock_client_with_voice = MockGeminiClient()
        mock_client_with_voice.set_json_response(director_dict)
        mock_client_with_voice.set_json_response(writer_dict)
        mock_client_with_voice.set_audio_response(
            TestExecuteSwarmPlanTopologiesAndArtifactsStress._get_test_wav_bytes(duration_sec=1.5)
        )
        mock_client_with_voice.set_json_response(swe_bundle_dict)

        orch_voice = StudioOrchestrator(client=mock_client_with_voice)
        orch_voice.agents["QAPlaytesterBuildSupervisor"] = mock_sup

        voice_agent_2 = orch_voice.agents["VoiceActingAudioProducerAgent"]
        voice_spy_2 = MagicMock(wraps=voice_agent_2.synthesize_voice_line)
        voice_agent_2.synthesize_voice_line = voice_spy_2

        plan_with_voice = StudioSwarmPlan(
            deliberation="Swarm plan including Voice specialist via canonical alias.",
            plan_id="plan_with_voice_001",
            game_title="Voice Gating Test",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director", "Writer", "Voice", "SWE"],
            dag_edges=[
                ("Director", "Writer"),
                ("Writer", "Voice"),
                ("Voice", "SWE"),
                ("Director", "SWE"),
            ],
            task_manifest={},
            target_episode=0,
        )

        staging_dir_with_voice = tmp_path / "staging_with_voice"
        report_voice = await orch_voice.execute_swarm_plan(plan_with_voice, staging_dir_with_voice)

        assert report_voice.promoted_to_active is True
        assert voice_spy_2.call_count > 0, "Voice synthesis must be executed when Voice role is included"
        audio_dir_voice = staging_dir_with_voice / "assets" / "audio"
        assert audio_dir_voice.exists()
        assert len(list(audio_dir_voice.glob("*.wav"))) > 0

    @pytest.mark.anyio
    async def test_iterative_swe_loop_multi_track_status_sequence_and_deliberation_index_zero(
        self,
        tmp_path: Path,
    ) -> None:
        """Stress tests iterative SWE loop across 3 tracks:
        - Sequence: R1(in_progress) -> R1(completed) -> R2(in_progress) -> R2(completed) -> R3(in_progress) -> R3(completed).
        - Rule 4 Invariance: 'deliberation' is strictly key index 0 in event and payload.
        - Persisted requirements_spec.json contains all tracks completed.
        """
        import json as _json

        client = MockGeminiClient()
        cartridge_bundle_data = {
            "deliberation": "Initial bundle for multi-track test.",
            "manifest_data": {
                "title": "Multi-Track Cartridge",
                "genre": "Action Roguelike",
                "version": "1.0.0",
                "entrypoint": "mechanics.py",
            },
            "mechanics_code": (
                '"""mechanics.py"""\n'
                'from engine.api.contracts import BaseCartridge, BaseScene\n'
                'class MultiScene(BaseScene):\n'
                '    deliberation: str = "Multi-track scene."\n'
                '    def __init__(self, engine=None):\n'
                '        super().__init__(engine)\n'
                '        self.counter = 0\n'
                '    def update(self, dt):\n'
                '        self.counter += 1\n'
                '    def render(self, surface):\n'
                '        pass\n'
                'class MultiCartridge(BaseCartridge):\n'
                '    deliberation: str = "Multi-track cartridge."\n'
                '    def get_scenes(self, engine):\n'
                '        return {"main": MultiScene(engine)}\n'
                '    def get_initial_scene(self, engine):\n'
                '        return self.get_scenes(engine)["main"]\n'
            ),
            "scenes_code": None,
            "entities_code": None,
            "abilities_code": None,
            "level_data": None,
        }
        client.set_json_response(cartridge_bundle_data)
        client.set_json_response(cartridge_bundle_data)
        client.set_json_response(cartridge_bundle_data)

        event_log = tmp_path / "multi_track_events.jsonl"
        bus = SwarmEventBus(log_path=event_log)
        orch = StudioOrchestrator(client=client, event_bus=bus)

        staging_dir = tmp_path / "staging_multi_track"
        staging_dir.mkdir(parents=True, exist_ok=True)

        req_spec = GameRequirementsSpec(
            deliberation="Epistemic decomposition into 3 distinct requirement tracks.",
            game_title="Multi-Track Cartridge",
            genre_paradigm="Action Roguelike",
            target_audience_summary="Fast kinetic testing",
            target_episode=0,
            requirements=[
                RequirementTrack(
                    deliberation="Track 1: Core simulation",
                    id="R1",
                    title="Core Engine",
                    description="Base mechanics",
                    responsible_agents=["LeadGameplaySWEAgent"],
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="AC1: Sim step",
                            id="AC1.1",
                            description="Headless steps",
                            verification_method="headless_playtest",
                        )
                    ],
                    status="pending",
                ),
                RequirementTrack(
                    deliberation="Track 2: Combat subsystem",
                    id="R2",
                    title="Combat System",
                    description="Projectiles",
                    responsible_agents=["LeadGameplaySWEAgent"],
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="AC2: AST check",
                            id="AC2.1",
                            description="No banned modules",
                            verification_method="ast_inspection",
                        )
                    ],
                    status="pending",
                ),
                RequirementTrack(
                    deliberation="Track 3: UI and scoring",
                    id="R3",
                    title="Scoring Subsystem",
                    description="Score tracker",
                    responsible_agents=["LeadGameplaySWEAgent"],
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="AC3: Metric validation",
                            id="AC3.1",
                            description="Score tracking",
                            verification_method="runtime_telemetry",
                        )
                    ],
                    status="pending",
                ),
            ],
            asset_manifest=[],
        )

        (staging_dir / "requirements_spec.json").write_text(
            req_spec.model_dump_json(indent=2), encoding="utf-8"
        )

        swe = orch.agents["LeadGameplaySWEAgent"]
        world_spec = GenesisWorldSpec(
            deliberation="World spec for multi-track test",
            game_title="Multi-Track Cartridge",
            core_genre="Action Roguelike",
            thematic_premise="Grid testing",
            world_lore="Lore",
            conflict_tone="tense",
            conflict_narrative="combat",
            aesthetic_style="retro_arcade",
            palette_mood="retro_arcade",
            gameplay_pacing="fast",
            pacing_score=0.5,
            target_difficulty=1.0,
        )

        results = await orch._execute_iterative_swe_loop(
            agent=swe,
            world_spec=world_spec,
            narrative_bible=None,
            req_spec=req_spec,
            staging_dir=staging_dir,
        )

        assert len(results) >= 1
        assert all(t.status == "completed" for t in req_spec.requirements)

        # Allow async event tasks to flush to disk
        await asyncio.sleep(0.1)

        # Verify disk persistence
        on_disk = GameRequirementsSpec.model_validate_json(
            (staging_dir / "requirements_spec.json").read_text(encoding="utf-8")
        )
        assert all(t.status == "completed" for t in on_disk.requirements)

        # Inspect all emitted events in JSONL log or bus
        events = bus.get_recent_events()
        assert len(events) >= 6

        status_events = []
        for ev in events:
            # Rule 4: Deliberation MUST be field index 0 in raw event JSON
            first_key = list(ev.keys())[0]
            assert first_key == "deliberation", f"Event must place 'deliberation' at index 0, got '{first_key}'"
            assert isinstance(ev["deliberation"], str) and len(ev["deliberation"]) > 0

            # Rule 4: Deliberation MUST be field index 0 in payload
            if "payload" in ev and isinstance(ev["payload"], dict) and "deliberation" in ev["payload"]:
                payload_first_key = list(ev["payload"].keys())[0]
                assert payload_first_key == "deliberation", f"Payload must place 'deliberation' at index 0, got '{payload_first_key}'"

            if ev.get("event_type") == SwarmEventType.REQUIREMENT_STATUS_UPDATE.value:
                status_events.append((ev["payload"]["requirement_id"], ev["payload"]["status"]))

        # Verify exact sequence across all 3 tracks:
        expected_sequence = [
            ("R1", "in_progress"),
            ("R1", "completed"),
            ("R2", "in_progress"),
            ("R2", "completed"),
            ("R3", "in_progress"),
            ("R3", "completed"),
        ]
        assert status_events == expected_sequence, f"Status events sequence mismatch: {status_events}"

    @pytest.mark.anyio
    async def test_iterative_swe_loop_halts_immediately_on_track_failure(
        self,
        tmp_path: Path,
    ) -> None:
        """When an intermediate track fails, the loop halts immediately:
        - Failing track marked 'rejected'
        - Subsequent tracks remain 'pending' (never executed)
        - Event log ends with 'rejected' for failing track
        - StudioSwarmExecutionError raised
        """
        import json as _json

        client = MagicMock(spec=[])
        event_log = tmp_path / "fail_fast_events.jsonl"
        bus = SwarmEventBus(log_path=event_log)
        orch = StudioOrchestrator(client=client, event_bus=bus)

        staging_dir = tmp_path / "staging_fail_fast"
        staging_dir.mkdir(parents=True, exist_ok=True)

        req_spec = GameRequirementsSpec(
            deliberation="4-track specification to test immediate loop halting upon track failure.",
            game_title="Fail Fast Test",
            genre_paradigm="Action Roguelike",
            target_audience_summary="Pacing index 0.5",
            target_episode=0,
            requirements=[
                RequirementTrack(
                    deliberation="Track 1: Initial mechanics",
                    id="R1",
                    title="Track 1 Valid",
                    description="Initial mechanics",
                    responsible_agents=["LeadGameplaySWEAgent"],
                    status="pending",
                ),
                RequirementTrack(
                    deliberation="Track 2: Broken requirement",
                    id="R2",
                    title="Track 2 Broken",
                    description="Synthesizes broken python file",
                    responsible_agents=["LeadGameplaySWEAgent"],
                    status="pending",
                ),
                RequirementTrack(
                    deliberation="Track 3: Should never execute",
                    id="R3",
                    title="Track 3 Unreachable",
                    description="Unreachable track",
                    responsible_agents=["LeadGameplaySWEAgent"],
                    status="pending",
                ),
                RequirementTrack(
                    deliberation="Track 4: Should never execute",
                    id="R4",
                    title="Track 4 Unreachable",
                    description="Unreachable track",
                    responsible_agents=["LeadGameplaySWEAgent"],
                    status="pending",
                ),
            ],
            asset_manifest=[],
        )

        (staging_dir / "requirements_spec.json").write_text(
            req_spec.model_dump_json(indent=2), encoding="utf-8"
        )

        # Write valid initial mechanics
        (staging_dir / "mechanics.py").write_text(
            '"""mechanics.py"""\n'
            'from engine.api.contracts import BaseCartridge, BaseScene\n'
            'class S(BaseScene):\n'
            '    deliberation: str = "Scene."\n'
            '    def __init__(self, engine=None):\n'
            '        super().__init__(engine)\n'
            '        self.counter = 0\n'
            '    def update(self, dt):\n'
            '        self.counter += 1\n'
            '    def render(self, s):\n'
            '        pass\n'
            'class C(BaseCartridge):\n'
            '    deliberation: str = "Cart."\n'
            '    def get_scenes(self, e):\n'
            '        return {"main": S(e)}\n'
            '    def get_initial_scene(self, e):\n'
            '        return self.get_scenes(e)["main"]\n',
            encoding="utf-8",
        )

        swe = orch.agents["LeadGameplaySWEAgent"]
        swe.client = MagicMock(spec=[])

        # Mock implement_requirement_track so track 2 injects a broken python file
        def mock_implement(track, staging_dir, **kwargs):
            if track.id == "R2":
                (staging_dir / "broken_module.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
            return {"mechanics.py": staging_dir / "mechanics.py"}

        swe.implement_requirement_track = MagicMock(side_effect=mock_implement)

        world_spec = GenesisWorldSpec(
            deliberation="World spec",
            game_title="Fail Fast Test",
            core_genre="Action Roguelike",
            thematic_premise="Premise",
            world_lore="Lore",
            conflict_tone="tense",
            conflict_narrative="combat",
            aesthetic_style="retro_arcade",
            palette_mood="retro_arcade",
            gameplay_pacing="fast",
            pacing_score=0.5,
            target_difficulty=1.0,
        )

        with pytest.raises(StudioSwarmExecutionError) as exc_info:
            await orch._execute_iterative_swe_loop(
                agent=swe,
                world_spec=world_spec,
                narrative_bible=None,
                req_spec=req_spec,
                staging_dir=staging_dir,
            )

        assert "Requirement R2 ('Track 2 Broken') failed verification" in str(exc_info.value)

        # Track status assertions
        assert req_spec.requirements[0].status == "completed", "Track R1 should be completed"
        assert req_spec.requirements[1].status == "rejected", "Track R2 should be rejected"
        assert req_spec.requirements[2].status == "pending", "Track R3 must remain pending"
        assert req_spec.requirements[3].status == "pending", "Track R4 must remain pending"

        # On-disk requirements_spec.json assertion
        disk_spec = GameRequirementsSpec.model_validate_json(
            (staging_dir / "requirements_spec.json").read_text(encoding="utf-8")
        )
        assert disk_spec.requirements[0].status == "completed"
        assert disk_spec.requirements[1].status == "rejected"
        assert disk_spec.requirements[2].status == "pending"
        assert disk_spec.requirements[3].status == "pending"

        # Event log assertion: R3 and R4 must NOT appear
        await asyncio.sleep(0.1)
        events = bus.get_recent_events()
        status_pairs = []
        for ev in events:
            if ev.get("event_type") == SwarmEventType.REQUIREMENT_STATUS_UPDATE.value:
                status_pairs.append((ev["payload"]["requirement_id"], ev["payload"]["status"]))

        assert status_pairs == [
            ("R1", "in_progress"),
            ("R1", "completed"),
            ("R2", "in_progress"),
            ("R2", "rejected"),
        ]

    @pytest.mark.anyio
    async def test_iterative_swe_loop_halts_on_ast_security_violation(
        self,
        tmp_path: Path,
    ) -> None:
        """AST security violations (e.g. prohibited subprocess/os import) halt the loop immediately."""
        client = MockGeminiClient()
        orch = StudioOrchestrator(client=client)

        staging_dir = tmp_path / "staging_ast_violation"
        staging_dir.mkdir(parents=True, exist_ok=True)

        req_spec = GameRequirementsSpec(
            deliberation="Spec for AST security test",
            game_title="AST Violation Test",
            genre_paradigm="Action Roguelike",
            target_audience_summary="Pacing 0.5",
            target_episode=0,
            requirements=[
                RequirementTrack(
                    deliberation="Track 1: Injects forbidden import",
                    id="R1",
                    title="Malicious Import Track",
                    description="Imports forbidden module",
                    responsible_agents=["LeadGameplaySWEAgent"],
                    status="pending",
                )
            ],
            asset_manifest=[],
        )

        # Inject forbidden import
        (staging_dir / "entities.py").write_text(
            'deliberation = "Entities."\nimport subprocess\n',
            encoding="utf-8",
        )

        swe = orch.agents["LeadGameplaySWEAgent"]
        swe.synthesize_cartridge_code = MagicMock(return_value={"entities.py": staging_dir / "entities.py"})
        swe.implement_requirement_track = MagicMock(return_value={"entities.py": staging_dir / "entities.py"})

        world_spec = GenesisWorldSpec(
            deliberation="World spec",
            game_title="AST Violation Test",
            core_genre="Action Roguelike",
            thematic_premise="Premise",
            world_lore="Lore",
            conflict_tone="tense",
            conflict_narrative="combat",
            aesthetic_style="retro_arcade",
            palette_mood="retro_arcade",
            gameplay_pacing="fast",
            pacing_score=0.5,
            target_difficulty=1.0,
        )

        with pytest.raises(StudioSwarmExecutionError) as exc_info:
            await orch._execute_iterative_swe_loop(
                agent=swe,
                world_spec=world_spec,
                narrative_bible=None,
                req_spec=req_spec,
                staging_dir=staging_dir,
            )

        assert "AST" in str(exc_info.value)
        assert "subprocess" in str(exc_info.value)
        assert req_spec.requirements[0].status == "rejected"

    @pytest.mark.anyio
    async def test_iterative_swe_loop_halts_on_headless_runtime_exception(
        self,
        tmp_path: Path,
    ) -> None:
        """Headless simulation crash during the 10-frame smoke check halts the loop immediately."""
        client = MockGeminiClient()
        orch = StudioOrchestrator(client=client)

        staging_dir = tmp_path / "staging_headless_crash"
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Write mechanics code that passes AST but crashes during headless step
        (staging_dir / "mechanics.py").write_text(
            '"""mechanics.py"""\n'
            'from engine.api.contracts import BaseCartridge, BaseScene\n'
            'class CrashScene(BaseScene):\n'
            '    deliberation: str = "Crashing scene."\n'
            '    def __init__(self, engine=None):\n'
            '        super().__init__(engine)\n'
            '    def update(self, dt):\n'
            '        # Deliberate runtime exception during step\n'
            '        x = 1 / 0\n'
            '    def render(self, surface):\n'
            '        pass\n'
            'class CrashCartridge(BaseCartridge):\n'
            '    deliberation: str = "Crash cartridge."\n'
            '    def get_scenes(self, engine):\n'
            '        return {"main": CrashScene(engine)}\n'
            '    def get_initial_scene(self, engine):\n'
            '        return self.get_scenes(engine)["main"]\n',
            encoding="utf-8",
        )

        req_spec = GameRequirementsSpec(
            deliberation="Spec for headless crash testing",
            game_title="Crash Test",
            genre_paradigm="Action Roguelike",
            target_audience_summary="Pacing 0.5",
            target_episode=0,
            requirements=[
                RequirementTrack(
                    deliberation="Track 1: Runtime crash in update()",
                    id="R1",
                    title="Crashing Track",
                    description="Crashes on update",
                    responsible_agents=["LeadGameplaySWEAgent"],
                    status="pending",
                )
            ],
            asset_manifest=[],
        )

        swe = orch.agents["LeadGameplaySWEAgent"]
        swe.implement_requirement_track = MagicMock(return_value={"mechanics.py": staging_dir / "mechanics.py"})
        world_spec = GenesisWorldSpec(
            deliberation="World spec",
            game_title="Crash Test",
            core_genre="Action Roguelike",
            thematic_premise="Premise",
            world_lore="Lore",
            conflict_tone="tense",
            conflict_narrative="combat",
            aesthetic_style="retro_arcade",
            palette_mood="retro_arcade",
            gameplay_pacing="fast",
            pacing_score=0.5,
            target_difficulty=1.0,
        )

        with pytest.raises(StudioSwarmExecutionError) as exc_info:
            await orch._execute_iterative_swe_loop(
                agent=swe,
                world_spec=world_spec,
                narrative_bible=None,
                req_spec=req_spec,
                staging_dir=staging_dir,
            )

        assert "Headless smoke check runtime exception" in str(exc_info.value)
        assert req_spec.requirements[0].status == "rejected"


