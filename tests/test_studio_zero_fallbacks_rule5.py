"""tests/test_studio_zero_fallbacks_rule5.py
Authoritative Test Suite Verifying Rule 5: Absolute Ban on Silent Fallbacks & Error Concealment.

Ensures that every studio specialist agent:
1. Fails fast with authentic exceptions when credentials, client, or endpoints fail.
2. Preserves the full underlying exception chain without masking or concealment.
3. NEVER falls back to deterministic approximations, static bibles, procedural placeholders, or seed cartridges.
"""
from __future__ import annotations
import inspect
from pathlib import Path
from typing import Any, Dict
import pytest

from engine.analyst.evolution import HypothesisDrivenEvolutionEngine
from engine.analyst.models import (
    BehavioralProfile,
    DashMetrics,
    KineticMetrics,
    RetryMetrics,
    SpatialOccupancyMap,
)
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
)
from engine.studio.narrative import NarrativeCharacterWriterAgent
from engine.studio.orchestrator import (
    CircularDependencyError,
    StudioOrchestrator,
)
from engine.studio.swe import CodeVerificationError, LeadGameplaySWEAgent
from engine.studio.telemetry_architect import TelemetryArchitectAgent
from engine.studio.visual import VisualArtSpriteDesignerAgent


class TestStudioZeroFallbacksRule5:
    """Verifies that all studio specialists fail fast without silent fallbacks."""
    deliberation: str = "Authoritative verification suite confirming zero silent fallbacks across all studio components."

    # -------------------------------------------------------------------------
    # 1. CreativeDirectorWorldbuilderAgent
    # -------------------------------------------------------------------------

    def test_creative_director_fails_fast_when_client_missing(self) -> None:
        """Rule 5: CreativeDirector must raise StudioAgentError when client is None."""
        director = CreativeDirectorWorldbuilderAgent(client=None)
        dna = SteamPlayerDNA(
            deliberation="Test DNA",
            account_id="test_player",
            total_games_analyzed=10,
            top_genres=["action"],
            pacing_score=0.5,
            challenge_tolerance=0.8,
            thematic_biases=["cyberpunk"],
            preferred_palette_mood="neon",
        )
        with pytest.raises(StudioAgentError) as exc_info:
            director.generate_genesis_world(dna)
        assert "requires an active google.genai Client" in str(exc_info.value)

    def test_creative_director_fails_fast_on_gemini_api_error(self, mock_gemini_client) -> None:
        """Rule 5: CreativeDirector must propagate exception without deterministic world fallback."""
        mock_gemini_client.set_error(ConnectionResetError("Socket dropped during world generation"))
        director = CreativeDirectorWorldbuilderAgent(client=mock_gemini_client)
        dna = SteamPlayerDNA(
            deliberation="Test DNA",
            account_id="test_player",
            total_games_analyzed=10,
            top_genres=["action"],
            pacing_score=0.5,
            challenge_tolerance=0.8,
            thematic_biases=["cyberpunk"],
            preferred_palette_mood="neon",
        )
        with pytest.raises(StudioAgentError) as exc_info:
            director.generate_genesis_world(dna)
        assert "Socket dropped during world generation" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ConnectionResetError)

    # -------------------------------------------------------------------------
    # 2. NarrativeCharacterWriterAgent
    # -------------------------------------------------------------------------

    def test_narrative_writer_fails_fast_when_client_missing(self) -> None:
        """Rule 5: NarrativeWriter must raise StudioAgentError when client is None."""
        writer = NarrativeCharacterWriterAgent(client=None)
        world_spec = GenesisWorldSpec(
            deliberation="World spec deliberation",
            game_title="Void Rogue",
            spec_id="spec_void",
            target_episode=0,
            core_genre="action_roguelike",
            thematic_premise="Deep space salvager fighting through adrift derelicts.",
            world_lore="The outer perimeter is abandoned.",
            conflict_tone="tense",
            conflict_narrative="Rogue drones defend the core mainframe.",
            aesthetic_style="Pixel art retro sci-fi",
            palette_mood="dark_industrial",
            gameplay_pacing="kinetic",
            pacing_score=0.5,
            target_difficulty=0.8,
        )
        with pytest.raises(StudioAgentError) as exc_info:
            writer.generate_narrative_bible(world_spec)
        assert "requires an active google.genai Client" in str(exc_info.value)

    def test_narrative_writer_fails_fast_on_gemini_api_error(self, mock_gemini_client) -> None:
        """Rule 5: NarrativeWriter must raise StudioAgentError without static 173-line bible fallback."""
        mock_gemini_client.set_error(RuntimeError("Gemini quota exceeded (HTTP 429)"))
        writer = NarrativeCharacterWriterAgent(client=mock_gemini_client)
        world_spec = GenesisWorldSpec(
            deliberation="World spec deliberation",
            game_title="Void Rogue",
            spec_id="spec_void",
            target_episode=0,
            core_genre="action_roguelike",
            thematic_premise="Deep space salvager fighting through adrift derelicts.",
            world_lore="The outer perimeter is abandoned.",
            conflict_tone="tense",
            conflict_narrative="Rogue drones defend the core mainframe.",
            aesthetic_style="Pixel art retro sci-fi",
            palette_mood="dark_industrial",
            gameplay_pacing="kinetic",
            pacing_score=0.5,
            target_difficulty=0.8,
        )
        with pytest.raises(StudioAgentError) as exc_info:
            writer.generate_narrative_bible(world_spec)
        assert "quota exceeded" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    # -------------------------------------------------------------------------
    # 3. VoiceActingAudioProducerAgent
    # -------------------------------------------------------------------------

    def test_audio_producer_fails_fast_when_client_missing_and_no_backend(self) -> None:
        """Rule 5: VoiceActingAudioProducerAgent must raise StudioAgentError if client is None and no backend passed."""
        with pytest.raises(StudioAgentError) as exc_info:
            VoiceActingAudioProducerAgent(client=None, backend=None)
        assert "requires an active google.genai Client or explicit backend" in str(exc_info.value)

    def test_audio_producer_fails_fast_on_tts_synthesis_error(self, tmp_path: Path, mock_gemini_client) -> None:
        """Rule 5: VoiceActingAudioProducerAgent must fail fast when TTS API fails."""
        mock_gemini_client.set_error(TimeoutError("Gemini Live TTS WebSocket connection timed out"))
        audio_agent = VoiceActingAudioProducerAgent(client=mock_gemini_client)
        bark = DialogueBarkSpec(
            deliberation="Bark deliberation",
            bark_id="bark_alert",
            speaker_id="sentinel",
            speaker_name="Sentinel",
            trigger_type="boss_encounter",
            text="Intruder detected in sector 4.",
            voice_clip="assets/audio/bark_alert.wav",
            duration=2.0,
        )
        out_wav = tmp_path / "bark_alert.wav"
        with pytest.raises(VoiceSynthesisError) as exc_info:
            audio_agent.synthesize_voice_line(bark, out_wav)
        assert "WebSocket connection timed out" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # 4. VisualArtSpriteDesignerAgent
    # -------------------------------------------------------------------------

    def test_visual_designer_fails_fast_when_client_missing(self, tmp_path: Path) -> None:
        """Rule 5: VisualArtSpriteDesignerAgent must raise StudioAgentError when client is None."""
        visual = VisualArtSpriteDesignerAgent(client=None)
        out_png = tmp_path / "hero.png"
        with pytest.raises(StudioAgentError) as exc_info:
            visual.generate_portrait(
                character="Hero",
                output_path=out_png,
                visual_prompt="Cybernetic operative",
            )
        assert "requires an active google.genai Client" in str(exc_info.value)

    def test_visual_designer_fails_fast_on_image_api_error(self, tmp_path: Path, mock_gemini_client) -> None:
        """Rule 5: VisualArtSpriteDesignerAgent must fail fast without procedural portrait fallback."""
        mock_gemini_client.set_error(ValueError("Invalid response schema from Imagen"))
        visual = VisualArtSpriteDesignerAgent(client=mock_gemini_client)
        out_png = tmp_path / "hero.png"
        with pytest.raises(StudioAgentError) as exc_info:
            visual.generate_portrait(
                character="Hero",
                output_path=out_png,
                visual_prompt="Cybernetic operative",
            )
        assert "Invalid response schema from Imagen" in str(exc_info.value)
        assert not out_png.exists()

    # -------------------------------------------------------------------------
    # 5. LeadGameplaySWEAgent
    # -------------------------------------------------------------------------

    def test_swe_agent_repair_fails_fast_when_client_missing(self) -> None:
        """Rule 5: LeadGameplaySWEAgent.repair_code_module must raise CodeVerificationError when client is None."""
        swe = LeadGameplaySWEAgent(client=None)
        broken = "def update():\n    pass"
        with pytest.raises(CodeVerificationError) as exc_info:
            swe.repair_code_module("mechanics.py", broken, "Missing deliberation")
        assert "requires an active google.genai Client for code repair" in str(exc_info.value)

    def test_swe_agent_deterministic_remedy_is_strictly_prohibited(self) -> None:
        """Rule 5: _deterministic_code_remedy must raise CodeVerificationError immediately."""
        swe = LeadGameplaySWEAgent(client=None)
        with pytest.raises(CodeVerificationError) as exc_info:
            swe._deterministic_code_remedy("mechanics.py", "import os\n")
        assert "strictly prohibited under Rule 5" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # 6. TelemetryArchitectAgent
    # -------------------------------------------------------------------------

    def test_telemetry_architect_fails_fast_when_client_missing(self, tmp_path: Path) -> None:
        """Rule 5: TelemetryArchitectAgent must raise StudioAgentError when gemini_client is None."""
        db_path = tmp_path / "telemetry_test.db"
        architect = TelemetryArchitectAgent(db_path=db_path, gemini_client=None)
        with pytest.raises(StudioAgentError) as exc_info:
            architect.inspect_code_and_plan({"entities.py": "class Player: pass"})
        assert "requires an active google.genai Client" in str(exc_info.value)

    def test_telemetry_architect_fails_fast_on_gemini_api_error(self, tmp_path: Path, mock_gemini_client) -> None:
        """Rule 5: TelemetryArchitectAgent must fail fast without AST keyword heuristics."""
        db_path = tmp_path / "telemetry_test.db"
        mock_gemini_client.set_error(RuntimeError("API quota exhausted for telemetry design"))
        architect = TelemetryArchitectAgent(db_path=db_path, gemini_client=mock_gemini_client)
        with pytest.raises(StudioAgentError) as exc_info:
            architect.inspect_code_and_plan({"entities.py": "class Player:\n    def dash(self): pass"})
        assert "API quota exhausted" in str(exc_info.value)

    def test_telemetry_architect_ast_heuristic_plan_is_strictly_prohibited(self, tmp_path: Path) -> None:
        """Rule 2 & 5: _generate_ast_driven_plan must raise StudioAgentError immediately."""
        db_path = tmp_path / "telemetry_test.db"
        architect = TelemetryArchitectAgent(db_path=db_path, gemini_client=None)
        with pytest.raises(StudioAgentError) as exc_info:
            architect._generate_ast_driven_plan([])
        assert "prohibited under Rule 2 & Rule 5" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # 7. StudioOrchestrator Meta-Planning & DAG Execution
    # -------------------------------------------------------------------------

    def test_orchestrator_generate_swarm_plan_fails_fast_on_error(self, mock_gemini_client) -> None:
        """Rule 5: StudioOrchestrator.generate_swarm_plan raises StudioSwarmPlanError on Gemini failure."""
        mock_gemini_client.set_error(ConnectionError("Meta-planning endpoint unreachable"))
        orch = StudioOrchestrator(client=mock_gemini_client)
        with pytest.raises(StudioSwarmPlanError) as exc_info:
            orch.generate_swarm_plan(target_episode=0)
        assert "Meta-planning endpoint unreachable" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_orchestrator_execute_swarm_plan_fails_fast_on_specialist_error(
        self,
        tmp_path: Path,
        mock_gemini_client,
    ) -> None:
        """Rule 5: StudioOrchestrator.execute_swarm_plan raises StudioSwarmExecutionError when a specialist fails."""
        orch = StudioOrchestrator(client=mock_gemini_client)
        plan = StudioSwarmPlan(
            deliberation="Test swarm plan deliberation",
            plan_id="plan_fail_test",
            game_title="Failing Game",
            genre_verdict="action",
            specialist_agent_roles=["Director", "SWE"],
            dag_edges=[("Director", "SWE")],
            task_manifest={},
            target_episode=0,
        )
        staging_dir = tmp_path / "staging_candidate"
        # Force specialist to raise an authentic error
        mock_gemini_client.set_error(RuntimeError("Specialist generative crash"))
        with pytest.raises(StudioSwarmExecutionError) as exc_info:
            await orch.execute_swarm_plan(plan, staging_dir)
        assert "failed during Tier" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # 8. StudioDaemon Cold Boot Genesis
    # -------------------------------------------------------------------------

    def test_daemon_genesis_cycle_fails_fast_without_silent_swallowing(self, tmp_path: Path, mock_gemini_client) -> None:
        """Rule 5: StudioDaemon.run_genesis_cycle raises exception instead of silently returning False."""
        mock_gemini_client.set_error(RuntimeError("Critical swarm pipeline failure"))
        orch = StudioOrchestrator(client=mock_gemini_client)
        config = StudioDaemonConfig(
            cartridges_root=tmp_path / "cartridges",
            active_cartridge_dir=tmp_path / "cartridges" / "active",
            lock_file=tmp_path / "studio.lock",
            db_path=tmp_path / "telemetry.db",
        )
        daemon = StudioDaemon(config_obj=config, orchestrator=orch)
        with pytest.raises(Exception):
            daemon.run_genesis_cycle()

    # -------------------------------------------------------------------------
    # 9. HypothesisDrivenEvolutionEngine Telemetry Evaluation
    # -------------------------------------------------------------------------

    def test_evolution_engine_fails_fast_on_telemetry_analysis_failure(self, tmp_path: Path) -> None:
        """Rule 5: HypothesisDrivenEvolutionEngine.evaluate_episode raises authentic error instead of fake profile fallback."""
        from unittest.mock import MagicMock
        db_path = tmp_path / "telemetry.db"
        engine = HypothesisDrivenEvolutionEngine(db_path=db_path)
        engine.analyst.analyze_episode = MagicMock(side_effect=IOError("Corrupted SQLite WAL trace"))
        with pytest.raises(IOError) as exc_info:
            engine.evaluate_episode(episode_id="ep_corrupted")
        assert "Corrupted SQLite WAL trace" in str(exc_info.value)
