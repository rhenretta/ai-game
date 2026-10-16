"""tests/test_m3_challenger_rule5_stress.py
Empirical Adversarial Stress Test Suite for Milestone M3 Rule 5 Zero-Fallback Behavior.
Authored by Challenger challenger_m3_1.

Adversarially verifies:
1. Simulated network / Gemini API exceptions (APIError, Timeout, RateLimitError, ConnectionResetError, client=None)
   across all 8 studio specialist agents (director, narrative, audio, visual, swe, telemetry_architect, daemon, evolution).
2. Authentic exception propagation and strict stack trace preservation (__cause__ chaining) with zero silent degradation,
   zero procedural fallbacks, and zero partial artifacts written to disk.
3. DAG execution stress with cyclic dependencies (2-node, self-loop, multi-node) and missing dependencies, verifying
   StudioSwarmPlanError, StudioSwarmExecutionError, and CircularDependencyError are strictly raised.
"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock
import pytest
from google.genai.errors import APIError, ClientError, ServerError

from engine.analyst.evolution import HypothesisDrivenEvolutionEngine
from engine.steam.models import SteamPlayerDNA
from engine.studio.audio import VoiceActingAudioProducerAgent, VoiceSynthesisError
from engine.studio.daemon import StudioDaemon, StudioDaemonConfig
from engine.studio.director import CreativeDirectorWorldbuilderAgent
from engine.studio.models import (
    DialogueBarkSpec,
    GenesisWorldSpec,
    StudioAgentError,
    StudioSwarmExecutionError,
    StudioSwarmPlan,
    StudioSwarmPlanError,
    UpdateWorldSpec,
)
from engine.studio.narrative import NarrativeBible, NarrativeCharacterWriterAgent
from engine.studio.orchestrator import (
    CircularDependencyError,
    StudioOrchestrator,
    StudioOrchestratorConfig,
)
from engine.studio.swe import CodeVerificationError, LeadGameplaySWEAgent
from engine.studio.telemetry_architect import TelemetryArchitectAgent
from engine.studio.visual import VisualArtSpriteDesignerAgent


def make_api_error(code: int, message: str) -> APIError:
    """Helper creating authentic google.genai.errors.APIError."""
    return APIError(code, {"error": {"message": message, "code": code, "status": "RESOURCE_EXHAUSTED" if code == 429 else "INTERNAL"}})


# =============================================================================
# 1. Specialist Agents Empirical Network & API Exception Stress Tests
# =============================================================================

class TestRule5SpecialistsNetworkAndAPIErrors:
    """Adversarial stress-testing of specialist agents under network/API errors."""
    deliberation: str = "Empirically challenge specialist agents with simulated network failures, timeouts, and rate limits."

    @pytest.fixture
    def sample_dna(self) -> SteamPlayerDNA:
        return SteamPlayerDNA(
            deliberation="Adversarial stress test DNA",
            account_id="stress_tester_01",
            total_games_analyzed=15,
            top_genres=["action", "tactical"],
            pacing_score=0.75,
            challenge_tolerance=0.9,
            thematic_biases=["cyberpunk", "grimdark"],
            preferred_palette_mood="dark_industrial",
        )

    @pytest.fixture
    def sample_world_spec(self) -> GenesisWorldSpec:
        return GenesisWorldSpec(
            deliberation="Adversarial world spec",
            game_title="Chronoshift",
            spec_id="spec_chrono_99",
            target_episode=0,
            core_genre="action_roguelike",
            thematic_premise="Temporal destabilization in an abandoned orbital station.",
            world_lore="Time paradoxes spawn anomalous hostiles.",
            conflict_tone="grim_futuristic",
            conflict_narrative="The protagonist must reset the central temporal regulator.",
            aesthetic_style="Neon pixel art cyberpunk",
            palette_mood="neon",
            gameplay_pacing="fast_kinetic",
            pacing_score=0.75,
            target_difficulty=0.85,
        )

    # -------------------------------------------------------------------------
    # CreativeDirectorWorldbuilderAgent
    # -------------------------------------------------------------------------

    def test_director_rate_limit_429_propagates_with_cause(self, sample_dna: SteamPlayerDNA, mock_gemini_client) -> None:
        """Rule 5: Director under HTTP 429 rate limit raises StudioAgentError with authentic cause preserved."""
        err = make_api_error(429, "Rate limit exceeded (quota depleted)")
        mock_gemini_client.set_error(err)
        director = CreativeDirectorWorldbuilderAgent(client=mock_gemini_client)

        with pytest.raises(StudioAgentError) as exc_info:
            director.generate_genesis_world(sample_dna)

        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.__cause__ is err

    def test_director_timeout_propagates_with_cause(self, sample_dna: SteamPlayerDNA, mock_gemini_client) -> None:
        """Rule 5: Director under socket timeout raises StudioAgentError preserving TimeoutError cause."""
        err = TimeoutError("HTTP request to Gemini endpoint timed out after 30s")
        mock_gemini_client.set_error(err)
        director = CreativeDirectorWorldbuilderAgent(client=mock_gemini_client)

        with pytest.raises(StudioAgentError) as exc_info:
            director.generate_genesis_world(sample_dna)

        assert "timed out after 30s" in str(exc_info.value)
        assert exc_info.value.__cause__ is err

    def test_director_empty_response_fails_fast(self, sample_dna: SteamPlayerDNA, mock_gemini_client) -> None:
        """Rule 5: Empty response from Gemini raises StudioAgentError without generating fallback."""
        mock_gemini_client.set_json_response("")
        director = CreativeDirectorWorldbuilderAgent(client=mock_gemini_client)

        with pytest.raises(StudioAgentError) as exc_info:
            director.generate_genesis_world(sample_dna)

        assert "Empty response" in str(exc_info.value)

    def test_director_episodic_update_fails_fast_when_client_missing(
        self,
        sample_dna: SteamPlayerDNA,
        sample_world_spec: GenesisWorldSpec,
    ) -> None:
        """Rule 5: generate_episodic_update_world raises StudioAgentError when client is None."""
        director = CreativeDirectorWorldbuilderAgent(client=None)
        with pytest.raises(StudioAgentError) as exc_info:
            director.generate_episodic_update_world(
                dna=sample_dna,
                hypothesis_report=None,
                previous_spec=sample_world_spec,
                target_episode=1,
            )
        assert "requires an active google.genai Client" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # NarrativeCharacterWriterAgent
    # -------------------------------------------------------------------------

    def test_narrative_server_error_503_propagates_with_cause(
        self,
        sample_world_spec: GenesisWorldSpec,
        mock_gemini_client,
    ) -> None:
        """Rule 5: NarrativeWriter under HTTP 503 raises StudioAgentError with ServerError cause."""
        err = make_api_error(503, "Gemini backend unavailable")
        mock_gemini_client.set_error(err)
        writer = NarrativeCharacterWriterAgent(client=mock_gemini_client)

        with pytest.raises(StudioAgentError) as exc_info:
            writer.generate_narrative_bible(sample_world_spec)

        assert "backend unavailable" in str(exc_info.value)
        assert exc_info.value.__cause__ is err

    def test_narrative_connection_reset_propagates_with_cause(
        self,
        sample_world_spec: GenesisWorldSpec,
        mock_gemini_client,
    ) -> None:
        """Rule 5: NarrativeWriter under connection reset raises StudioAgentError with ConnectionResetError cause."""
        err = ConnectionResetError("TCP connection reset by peer during streaming")
        mock_gemini_client.set_error(err)
        writer = NarrativeCharacterWriterAgent(client=mock_gemini_client)

        with pytest.raises(StudioAgentError) as exc_info:
            writer.generate_narrative_bible(sample_world_spec)

        assert "TCP connection reset" in str(exc_info.value)
        assert exc_info.value.__cause__ is err

    def test_narrative_empty_response_fails_fast(
        self,
        sample_world_spec: GenesisWorldSpec,
        mock_gemini_client,
    ) -> None:
        """Rule 5: Empty narrative response raises StudioAgentError without static bible fallback."""
        mock_gemini_client.set_json_response("")
        writer = NarrativeCharacterWriterAgent(client=mock_gemini_client)

        with pytest.raises(StudioAgentError) as exc_info:
            writer.generate_narrative_bible(sample_world_spec)

        assert "Empty response" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # VoiceActingAudioProducerAgent
    # -------------------------------------------------------------------------

    def test_audio_producer_rate_limit_429_propagates_with_cause(
        self,
        tmp_path: Path,
        mock_gemini_client,
    ) -> None:
        """Rule 5: Audio producer under HTTP 429 raises VoiceSynthesisError with APIError cause."""
        err = make_api_error(429, "TTS resource quota exceeded")
        mock_gemini_client.set_error(err)
        audio_agent = VoiceActingAudioProducerAgent(client=mock_gemini_client)

        bark = DialogueBarkSpec(
            deliberation="Test bark",
            bark_id="bark_stress_429",
            speaker_id="hero",
            speaker_name="Protagonist",
            trigger_type="combat",
            text="Reloading weapons matrix.",
            voice_clip="assets/audio/bark_stress_429.wav",
            duration=1.5,
        )
        out_wav = tmp_path / "bark_stress_429.wav"

        with pytest.raises(VoiceSynthesisError) as exc_info:
            audio_agent.synthesize_voice_line(bark, out_wav)

        assert "TTS resource quota exceeded" in str(exc_info.value)
        assert exc_info.value.__cause__ is err
        assert not out_wav.exists()

    def test_audio_producer_empty_audio_bytes_fails_fast(
        self,
        tmp_path: Path,
        mock_gemini_client,
    ) -> None:
        """Rule 5: Audio producer receiving empty audio bytes raises VoiceSynthesisError."""
        mock_gemini_client.set_audio_response(b"")
        audio_agent = VoiceActingAudioProducerAgent(client=mock_gemini_client)

        bark = DialogueBarkSpec(
            deliberation="Test empty bark",
            bark_id="bark_empty",
            speaker_id="hero",
            speaker_name="Protagonist",
            trigger_type="combat",
            text="Affirmative.",
            voice_clip="assets/audio/bark_empty.wav",
            duration=1.0,
        )
        out_wav = tmp_path / "bark_empty.wav"

        with pytest.raises(VoiceSynthesisError) as exc_info:
            audio_agent.synthesize_voice_line(bark, out_wav)

        assert "No audio bytes returned" in str(exc_info.value)
        assert not out_wav.exists()

    # -------------------------------------------------------------------------
    # VisualArtSpriteDesignerAgent
    # -------------------------------------------------------------------------

    def test_visual_rate_limit_429_propagates_with_cause(
        self,
        tmp_path: Path,
        mock_gemini_client,
    ) -> None:
        """Rule 5: Visual designer under HTTP 429 raises StudioAgentError with APIError cause."""
        err = make_api_error(429, "Imagen generation quota exceeded")
        mock_gemini_client.set_error(err)
        visual = VisualArtSpriteDesignerAgent(client=mock_gemini_client)
        out_png = tmp_path / "portrait_stress.png"

        with pytest.raises(StudioAgentError) as exc_info:
            visual.generate_portrait(
                character="Boss",
                output_path=out_png,
                visual_prompt="Armored mechanical colossus",
            )

        assert "Imagen generation quota exceeded" in str(exc_info.value)
        assert exc_info.value.__cause__ is err
        assert not out_png.exists()

    def test_visual_timeout_propagates_with_cause(
        self,
        tmp_path: Path,
        mock_gemini_client,
    ) -> None:
        """Rule 5: Visual designer under timeout raises StudioAgentError with TimeoutError cause."""
        err = TimeoutError("Imagen generation network timeout")
        mock_gemini_client.set_error(err)
        visual = VisualArtSpriteDesignerAgent(client=mock_gemini_client)
        out_png = tmp_path / "portrait_timeout.png"

        with pytest.raises(StudioAgentError) as exc_info:
            visual.generate_portrait(
                character="Companion",
                output_path=out_png,
                visual_prompt="Recon drone",
            )

        assert "network timeout" in str(exc_info.value)
        assert exc_info.value.__cause__ is err
        assert not out_png.exists()

    # -------------------------------------------------------------------------
    # LeadGameplaySWEAgent
    # -------------------------------------------------------------------------

    def test_swe_agent_rate_limit_429_propagates_with_cause(
        self,
        sample_world_spec: GenesisWorldSpec,
        mock_gemini_client,
    ) -> None:
        """Rule 5: SWE agent under HTTP 429 raises StudioAgentError with APIError cause."""
        err = make_api_error(429, "Code generation quota exhausted")
        mock_gemini_client.set_error(err)
        swe = LeadGameplaySWEAgent(client=mock_gemini_client)

        with pytest.raises(StudioAgentError) as exc_info:
            swe.synthesize_cartridge_code(sample_world_spec)

        assert "quota exhausted" in str(exc_info.value)
        assert exc_info.value.__cause__ is err

    def test_swe_agent_empty_response_fails_fast(
        self,
        sample_world_spec: GenesisWorldSpec,
        mock_gemini_client,
    ) -> None:
        """Rule 5: SWE agent receiving empty response raises CodeVerificationError."""
        mock_gemini_client.set_json_response("")
        swe = LeadGameplaySWEAgent(client=mock_gemini_client)

        with pytest.raises(CodeVerificationError) as exc_info:
            swe.synthesize_cartridge_code(sample_world_spec)

        assert "Empty response" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # TelemetryArchitectAgent
    # -------------------------------------------------------------------------

    def test_telemetry_architect_rate_limit_429_propagates_with_cause(
        self,
        tmp_path: Path,
        mock_gemini_client,
    ) -> None:
        """Rule 5: Telemetry architect under HTTP 429 raises StudioAgentError with APIError cause."""
        db_path = tmp_path / "telemetry_stress.db"
        err = make_api_error(429, "Telemetry probe design quota exceeded")
        mock_gemini_client.set_error(err)
        architect = TelemetryArchitectAgent(db_path=db_path, gemini_client=mock_gemini_client)

        with pytest.raises(StudioAgentError) as exc_info:
            architect.inspect_code_and_plan({"mechanics.py": "class Game: pass"})

        assert "quota exceeded" in str(exc_info.value)
        assert exc_info.value.__cause__ is err

    def test_telemetry_architect_timeout_propagates_with_cause(
        self,
        tmp_path: Path,
        mock_gemini_client,
    ) -> None:
        """Rule 5: Telemetry architect under timeout raises StudioAgentError with TimeoutError cause."""
        db_path = tmp_path / "telemetry_stress.db"
        err = TimeoutError("Telemetry planning network timeout")
        mock_gemini_client.set_error(err)
        architect = TelemetryArchitectAgent(db_path=db_path, gemini_client=mock_gemini_client)

        with pytest.raises(StudioAgentError) as exc_info:
            architect.inspect_code_and_plan({"mechanics.py": "class Game: pass"})

        assert "network timeout" in str(exc_info.value)
        assert exc_info.value.__cause__ is err

    # -------------------------------------------------------------------------
    # StudioDaemon Genesis Cycle
    # -------------------------------------------------------------------------

    def test_daemon_fails_fast_on_api_error_without_fallback(
        self,
        tmp_path: Path,
        mock_gemini_client,
    ) -> None:
        """Rule 5: StudioDaemon.run_genesis_cycle re-raises APIError without falling back to seed."""
        err = make_api_error(500, "Gemini meta-planning cluster crash")
        mock_gemini_client.set_error(err)
        orch = StudioOrchestrator(client=mock_gemini_client)

        cartridges_root = tmp_path / "cartridges"
        config = StudioDaemonConfig(
            cartridges_root=cartridges_root,
            active_cartridge_dir=cartridges_root / "active",
            lock_file=tmp_path / "studio.lock",
            db_path=tmp_path / "telemetry.db",
        )
        daemon = StudioDaemon(config_obj=config, orchestrator=orch)

        with pytest.raises(StudioSwarmPlanError) as exc_info:
            daemon.run_genesis_cycle()

        assert "cluster crash" in str(exc_info.value)
        # Ensure active cartridge was NOT created
        assert not (cartridges_root / "active").exists()

    # -------------------------------------------------------------------------
    # HypothesisDrivenEvolutionEngine
    # -------------------------------------------------------------------------

    def test_evolution_engine_fails_fast_on_database_corruption(self, tmp_path: Path) -> None:
        """Rule 5: Evolution engine raises authentic database exception without falling back to fake profile."""
        db_path = tmp_path / "corrupted_telemetry.db"
        # 1. Verification of fail-fast initialization on corrupt SQLite file
        db_path.write_bytes(b"CORRUPTED_SQLITE_GARBAGE_HEADER")
        with pytest.raises(Exception) as exc_init:
            HypothesisDrivenEvolutionEngine(db_path=db_path)
        assert "not a database" in str(exc_init.value) or "DatabaseError" in type(exc_init.value).__name__

        # 2. Verification of fail-fast during evaluate_episode on runtime WAL query failure
        valid_db = tmp_path / "valid_telemetry.db"
        engine = HypothesisDrivenEvolutionEngine(db_path=valid_db)
        engine.analyst.analyze_episode = MagicMock(side_effect=IOError("Corrupted SQLite WAL trace during analysis"))

        with pytest.raises(IOError) as exc_eval:
            engine.evaluate_episode("ep_corrupted_test")

        assert "Corrupted SQLite WAL trace" in str(exc_eval.value)


# =============================================================================
# 2. DAG Execution Stress: Cycles & Missing Dependencies
# =============================================================================

class TestRule5DAGExecutionCyclesAndDependencies:
    """Stress tests DAG generation and execution with cycles, self-loops, and missing dependencies."""
    deliberation: str = "Empirically challenge DAG execution with circular dependencies and un-assembled roles."

    def test_generate_swarm_plan_raises_swarm_plan_error_on_dag_cycle(self, mock_gemini_client) -> None:
        """Rule 5: Gemini returning cyclic DAG edges raises StudioSwarmPlanError caused by CircularDependencyError."""
        cyclic_plan_payload = {
            "deliberation": "Deliberation for a cyclic plan that should be rejected immediately.",
            "plan_id": "plan_cyclic_01",
            "game_title": "Paradox Realm",
            "genre_verdict": "puzzle",
            "specialist_agent_roles": ["Director", "Writer"],
            "dag_edges": [("Director", "Writer"), ("Writer", "Director")],
            "task_manifest": {},
            "target_episode": 0,
        }
        mock_gemini_client.set_json_response(cyclic_plan_payload)
        orch = StudioOrchestrator(client=mock_gemini_client)

        with pytest.raises(StudioSwarmPlanError) as exc_info:
            orch.generate_swarm_plan(target_episode=0)

        assert "Cycle detected" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, CircularDependencyError)

    def test_generate_swarm_plan_raises_swarm_plan_error_on_self_loop(self, mock_gemini_client) -> None:
        """Rule 5: Gemini returning self-referential edge raises StudioSwarmPlanError caused by CircularDependencyError."""
        self_loop_payload = {
            "deliberation": "Deliberation for self-referential role edge.",
            "plan_id": "plan_self_loop",
            "game_title": "Infinite Loop",
            "genre_verdict": "action",
            "specialist_agent_roles": ["Director"],
            "dag_edges": [("Director", "Director")],
            "task_manifest": {},
            "target_episode": 0,
        }
        mock_gemini_client.set_json_response(self_loop_payload)
        orch = StudioOrchestrator(client=mock_gemini_client)

        with pytest.raises(StudioSwarmPlanError) as exc_info:
            orch.generate_swarm_plan(target_episode=0)

        assert "Cycle detected" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, CircularDependencyError)

    def test_generate_swarm_plan_raises_swarm_plan_error_on_4_node_cycle(self, mock_gemini_client) -> None:
        """Rule 5: Gemini returning 4-node ring raises StudioSwarmPlanError caused by CircularDependencyError."""
        ring_payload = {
            "deliberation": "Deliberation for 4-node circular dependency chain.",
            "plan_id": "plan_4node_ring",
            "game_title": "Ouroboros",
            "genre_verdict": "tactical",
            "specialist_agent_roles": ["Director", "Writer", "Visual", "SWE"],
            "dag_edges": [
                ("Director", "Writer"),
                ("Writer", "Visual"),
                ("Visual", "SWE"),
                ("SWE", "Director"),
            ],
            "task_manifest": {},
            "target_episode": 0,
        }
        mock_gemini_client.set_json_response(ring_payload)
        orch = StudioOrchestrator(client=mock_gemini_client)

        with pytest.raises(StudioSwarmPlanError) as exc_info:
            orch.generate_swarm_plan(target_episode=0)

        assert "Cycle detected" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, CircularDependencyError)

    def test_execute_swarm_plan_direct_cycle_raises_circular_dependency_error(
        self,
        tmp_path: Path,
        mock_gemini_client,
    ) -> None:
        """Rule 5: Direct execution of plan with cyclic edges raises CircularDependencyError in _compute_execution_tiers."""
        orch = StudioOrchestrator(client=mock_gemini_client)
        cyclic_plan = StudioSwarmPlan(
            deliberation="Manually constructed cyclic plan for execution stress",
            plan_id="plan_exec_cycle",
            game_title="Direct Cycle",
            genre_verdict="action",
            specialist_agent_roles=["Director", "SWE"],
            dag_edges=[("Director", "SWE"), ("SWE", "Director")],
            task_manifest={},
            target_episode=0,
        )
        staging_dir = tmp_path / "staging_cycle"

        # _compute_execution_tiers called during execute_swarm_plan must raise CircularDependencyError
        with pytest.raises(CircularDependencyError) as exc_info:
            asyncio.run(orch.execute_swarm_plan(cyclic_plan, staging_dir))

        assert "Cycle detected in agent DAG" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_swarm_plan_with_unassembled_role_raises_swarm_execution_error(
        self,
        tmp_path: Path,
        mock_gemini_client,
    ) -> None:
        """Rule 5: Swarm plan referencing an un-assembled / non-existent specialist role raises StudioSwarmExecutionError."""
        orch = StudioOrchestrator(client=mock_gemini_client)
        missing_role_plan = StudioSwarmPlan(
            deliberation="Plan referencing unknown role that cannot be assembled",
            plan_id="plan_missing_role",
            game_title="Alien Role Game",
            genre_verdict="action",
            specialist_agent_roles=["NonExistentSpecialist"],
            dag_edges=[],
            task_manifest={},
            target_episode=0,
        )
        staging_dir = tmp_path / "staging_missing_role"

        with pytest.raises(StudioSwarmExecutionError) as exc_info:
            await orch.execute_swarm_plan(missing_role_plan, staging_dir)

        assert "is not assembled in StudioOrchestrator" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None or "failed during Tier" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_swarm_plan_specialist_api_error_fails_fast_with_cause(
        self,
        tmp_path: Path,
        mock_gemini_client,
    ) -> None:
        """Rule 5: Execution of swarm plan when specialist encounters APIError raises StudioSwarmExecutionError with cause."""
        orch = StudioOrchestrator(client=mock_gemini_client)
        plan = StudioSwarmPlan(
            deliberation="Plan where Director fails immediately due to API outage",
            plan_id="plan_api_fail",
            game_title="Failing Build Game",
            genre_verdict="rpg",
            specialist_agent_roles=["Director"],
            dag_edges=[],
            task_manifest={},
            target_episode=0,
        )
        staging_dir = tmp_path / "staging_api_fail"

        err = make_api_error(503, "Director model service unavailable")
        mock_gemini_client.set_error(err)

        with pytest.raises(StudioSwarmExecutionError) as exc_info:
            await orch.execute_swarm_plan(plan, staging_dir)

        assert "failed during Tier 0" in str(exc_info.value)
        # Check that cause is preserved
        assert exc_info.value.__cause__ is not None
        assert "Director" in str(exc_info.value)
