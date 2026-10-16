"""tests/test_studio_orchestrator.py
Comprehensive test suite verifying StudioOrchestrator, StudioAgentDAG, and PEV lifecycle.
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from engine.scheduler.artifact import ArtifactMetadataEnvelope
from engine.studio.models import (
    AgentPlan,
    AgentVerificationRecord,
    StudioBuildState,
    StudioOrchestratorConfig,
    StudioSwarmPlan,
)
from engine.studio.orchestrator import (
    BuildVerificationExhaustedError,
    CircularDependencyError,
    InvalidStateTransitionError,
    StudioAgentDAG,
    StudioOrchestrator,
)
from tests.conftest import MockGeminiClient


class TestStudioOrchestrator:
    """Verifies StudioOrchestrator persona assembly, DAG ordering, and PEV lifecycle."""
    deliberation: str = "Test suite validating studio orchestrator coordination and fault isolation."

    def test_orchestrator_initializes_nine_agent_roles(self) -> None:
        """F14.1: Orchestrator instantiates all 9 specialist agent roles."""
        orch = StudioOrchestrator()
        expected = [
            "StudioOrchestrator",
            "SteamAndTelemetryAnalystAgent",
            "CreativeDirectorWorldbuilderAgent",
            "NarrativeCharacterWriterAgent",
            "VoiceActingAudioProducerAgent",
            "VisualArtSpriteDesignerAgent",
            "LeadGameplaySWEAgent",
            "TelemetryArchitectAgent",
            "QAPlaytesterBuildSupervisor",
        ]
        assert len(orch.agents) == 9
        for name in expected:
            assert name in orch.agents

    def test_orchestrator_empty_api_key_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """F14.B1: Empty or whitespace-only API key raises ValueError."""
        monkeypatch.setenv("GEMINI_API_KEY", "")
        # Even when .env is bypassed, empty key must raise ValueError
        with pytest.raises(ValueError):
            StudioOrchestrator(api_key="")

        with pytest.raises(ValueError):
            StudioOrchestrator(api_key="   ")

    def test_agent_dag_topological_sort_and_dependencies(self) -> None:
        """F14.3: DAG topological sort produces correct causal execution order."""
        dag = StudioAgentDAG()
        order = dag.topological_sort()
        assert len(order) >= 8

        # Assert dependencies precede dependents
        assert order.index("Analyst") < order.index("Director")
        assert order.index("Director") < order.index("Writer")
        assert order.index("Writer") < order.index("Voice")
        assert order.index("Writer") < order.index("Visual")
        assert order.index("Director") < order.index("SWE")
        assert order.index("SWE") < order.index("TelemetryArchitect")
        assert order.index("SWE") < order.index("Supervisor")
        assert order.index("Supervisor") < order.index("Promotion")

    def test_agent_dag_cycle_detection(self) -> None:
        """F14.B3: Detects circular dependencies and raises CircularDependencyError."""
        dag = StudioAgentDAG()
        assert dag.has_cycle() is False

        # Add cycle
        dag.add_edge("Promotion", "Analyst")
        assert dag.has_cycle() is True
        with pytest.raises(CircularDependencyError):
            dag.topological_sort()

    def test_orchestrator_state_machine_valid_lifecycle(self) -> None:
        """F14.5: Verifies valid forward state machine lifecycle."""
        orch = StudioOrchestrator()
        assert orch.state == StudioBuildState.IDLE

        orch.transition_to(StudioBuildState.ANALYZING, "Ingesting Steam DNA")
        assert orch.state == StudioBuildState.ANALYZING

        orch.transition_to(StudioBuildState.SYNTHESIZING, "Synthesizing world and code")
        assert orch.state == StudioBuildState.SYNTHESIZING

        orch.transition_to(StudioBuildState.VERIFYING, "Running 5-gate supervisor")
        assert orch.state == StudioBuildState.VERIFYING

        orch.transition_to(StudioBuildState.PROMOTED, "All quality gates passed")
        assert orch.state == StudioBuildState.PROMOTED

        orch.transition_to(StudioBuildState.IDLE, "Build cycle complete")
        assert orch.state == StudioBuildState.IDLE

    def test_orchestrator_state_machine_self_healing_transition(self) -> None:
        """State machine permits VERIFYING -> SYNTHESIZING for self-healing repair."""
        orch = StudioOrchestrator()
        orch.transition_to(StudioBuildState.ANALYZING, "DNA analysis")
        orch.transition_to(StudioBuildState.SYNTHESIZING, "Synthesis")
        orch.transition_to(StudioBuildState.VERIFYING, "Verification")
        orch.transition_to(StudioBuildState.SYNTHESIZING, "Repairing AST defect")
        assert orch.state == StudioBuildState.SYNTHESIZING

    def test_orchestrator_state_machine_invalid_transitions(self) -> None:
        """F14.B5: Illegal transitions (e.g. IDLE -> PROMOTED) raise InvalidStateTransitionError."""
        orch = StudioOrchestrator()
        with pytest.raises(InvalidStateTransitionError):
            orch.transition_to(StudioBuildState.PROMOTED, "Illegal direct jump to PROMOTED")

        with pytest.raises(InvalidStateTransitionError):
            orch.transition_to(StudioBuildState.VERIFYING, "Illegal jump to VERIFYING")

    @pytest.mark.anyio
    async def test_pev_cycle_success_and_cryptographic_envelope(self) -> None:
        """F14.4 & F16: PEV cycle executes Plan, Execute, Verify and seals envelope."""
        orch = StudioOrchestrator()

        async def plan():
            return AgentPlan(
                deliberation="Planning narrative generation",
                agent_name="Writer",
                objective="Write characters",
                steps=["Design protagonist", "Write barks"],
            )

        async def execute(p):
            return {"characters": ["hero", "boss"], "status": "OK"}

        async def verify(res):
            return AgentVerificationRecord(
                deliberation="Output meets all criteria",
                agent_name="Writer",
                status="VERIFIED",
                passed=True,
            )

        output, envelope = await orch.execute_pev_cycle("Writer", plan, execute, verify)
        assert output["status"] == "OK"
        assert isinstance(envelope, ArtifactMetadataEnvelope)
        assert len(envelope.sha256_hash) == 64
        assert envelope.source_agent == "Writer"

    @pytest.mark.anyio
    async def test_pev_cycle_empty_plan_steps_rejection(self) -> None:
        """F16.B1: Plan with steps=[] raises ValueError."""
        orch = StudioOrchestrator()

        async def plan():
            return {"deliberation": "Empty plan", "agent_name": "test", "objective": "fail", "steps": []}

        async def execute(p):
            return {}

        async def verify(r):
            return AgentVerificationRecord(deliberation="V", agent_name="test", passed=True)

        with pytest.raises(ValueError):
            await orch.execute_pev_cycle("test", plan, execute, verify)

    @pytest.mark.anyio
    async def test_pev_cycle_retry_exhaustion(self) -> None:
        """F16.B2: Exceeding max_retries on verification raises BuildVerificationExhaustedError."""
        orch = StudioOrchestrator()

        async def plan():
            return AgentPlan(
                deliberation="Plan",
                agent_name="FailingAgent",
                objective="Fail",
                steps=["Step 1"],
            )

        async def execute(p):
            return {"broken": True}

        async def verify(res):
            return AgentVerificationRecord(
                deliberation="Defects detected",
                agent_name="FailingAgent",
                status="VERIFICATION_FAILED",
                passed=False,
                violations=["SyntaxError: line 10", "Unreachable code"],
            )

        with pytest.raises(BuildVerificationExhaustedError):
            await orch.execute_pev_cycle("FailingAgent", plan, execute, verify, max_retries=2)

    @pytest.mark.anyio
    async def test_orchestrator_execute_swarm_plan_happy_path(
        self,
        tmp_path: Path,
        mock_gemini_client: MockGeminiClient,
    ) -> None:
        """Integration test verifying end-to-end happy path execution of execute_swarm_plan.
        Exercises 5 specialist roles across topological tiers:
        Director -> Writer -> SWE -> TelemetryArchitect -> Visual.
        Verifies all staging artifacts are generated without TypeError or AttributeError.
        """
        import io
        from PIL import Image

        # 1. Setup Director GenesisWorldSpec response
        world_spec_data = {
            "deliberation": "Formulating Genesis world specification for cybernetic survival roguelike.",
            "spec_id": "world_spec_genesis_001",
            "target_episode": 0,
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
        mock_gemini_client.set_json_response(world_spec_data)

        # 2. Setup Writer NarrativeBible response
        narrative_bible_data = {
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
        mock_gemini_client.set_json_response(narrative_bible_data)

        # 3. Setup SWE OpenGenreCartridgeBundle response
        mechanics_source = (
            '"""mechanics.py - Synthesized Autonomous Game Cartridge."""\n'
            'from typing import Any, Dict\n'
            'from engine.api.contracts import BaseCartridge, BaseScene\n\n\n'
            'class GridSurvivalScene(BaseScene):\n'
            '    deliberation: str = "Primary gameplay scene implementing open-genre Pygame lifecycle."\n\n'
            '    def on_enter(self, **kwargs: Any) -> None:\n'
            '        self.active = True\n\n'
            '    def on_exit(self) -> None:\n'
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
        cartridge_bundle_data = {
            "deliberation": "Formulating complete open-genre cartridge code bundle implementing BaseCartridge.",
            "manifest_data": {
                "title": "Neon Grid Survival",
                "genre": "Action Roguelike",
                "version": "1.0.0",
                "entrypoint": "mechanics:GridSurvivalCartridge",
            },
            "mechanics_code": mechanics_source,
            "scenes_code": None,
            "entities_code": None,
            "abilities_code": None,
            "level_data": None,
        }
        mock_gemini_client.set_json_response(cartridge_bundle_data)

        # 4. Setup TelemetryArchitect TelemetryArchitecturePlan response
        telemetry_plan_data = {
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
        mock_gemini_client.set_json_response(telemetry_plan_data)

        # 5. Setup Visual image response
        buf = io.BytesIO()
        img = Image.new("RGBA", (16, 16), color=(0, 255, 200, 255))
        img.save(buf, format="PNG")
        mock_gemini_client.set_image_response(buf.getvalue())

        # Assemble Orchestrator
        orch = StudioOrchestrator(client=mock_gemini_client)

        # Mock supervisor evaluation boundary to isolate swarm tier synthesis from M4 QA
        mock_sup = MagicMock()
        mock_report = MagicMock()
        mock_report.promoted_to_active = True
        mock_sup.evaluate_and_heal.return_value = mock_report
        orch.agents["QAPlaytesterBuildSupervisor"] = mock_sup

        # 6. Define multi-role StudioSwarmPlan: Director -> Writer -> SWE -> TelemetryArchitect -> Visual
        plan = StudioSwarmPlan(
            deliberation="5-role specialist swarm plan for Genesis creation.",
            plan_id="plan_happy_path_001",
            game_title="Neon Grid Survival",
            genre_verdict="Action Roguelike",
            specialist_agent_roles=["Director", "Writer", "SWE", "TelemetryArchitect", "Visual"],
            dag_edges=[
                ("Director", "Writer"),
                ("Writer", "SWE"),
                ("SWE", "TelemetryArchitect"),
                ("TelemetryArchitect", "Visual"),
            ],
            task_manifest={
                "Director": "Generate Genesis world specification",
                "Writer": "Generate character bible and dialogue barks",
                "SWE": "Synthesize BaseCartridge mechanics code",
                "TelemetryArchitect": "Inject custom telemetry probes into mechanics",
                "Visual": "Synthesize character portrait sprites",
            },
            target_episode=0,
        )

        staging_dir = tmp_path / "staging_happy_path"

        # 7. Execute Swarm Plan to Completion
        report = await orch.execute_swarm_plan(plan, staging_dir)

        # Assertions
        assert report is not None
        assert report.promoted_to_active is True
        assert orch.state == StudioBuildState.IDLE

        # Assert staging file generation
        world_spec_path = staging_dir / "world_spec.json"
        manifest_path = staging_dir / "manifest.json"
        narrative_bible_path = staging_dir / "narrative_bible.json"
        narrative_json_path = staging_dir / "narrative.json"
        mechanics_path = staging_dir / "mechanics.py"
        telemetry_manifest_path = staging_dir / "telemetry_manifest.json"
        portrait_path = staging_dir / "assets" / "sprites" / "kael_portrait.png"

        assert world_spec_path.is_file(), "world_spec.json must be generated"
        assert manifest_path.is_file(), "manifest.json must be generated"
        assert narrative_bible_path.is_file(), "narrative_bible.json must be generated"
        assert narrative_json_path.is_file(), "narrative.json must be generated"
        assert mechanics_path.is_file(), "mechanics.py cartridge code must be generated"
        assert telemetry_manifest_path.is_file(), "telemetry_manifest.json must be generated"
        assert portrait_path.is_file(), "kael_portrait.png must be generated"

        # Assert artifact contents
        world_data = json.loads(world_spec_path.read_text(encoding="utf-8"))
        assert world_data["game_title"] == "Neon Grid Survival"
        assert world_data["core_genre"] == "Action Roguelike"

        bible_data = json.loads(narrative_bible_path.read_text(encoding="utf-8"))
        assert len(bible_data["characters"]) >= 1
        assert bible_data["characters"][0]["name"] == "Kael"

        mechanics_content = mechanics_path.read_text(encoding="utf-8")
        assert "GridSurvivalCartridge" in mechanics_content
        assert "BaseCartridge" in mechanics_content
        # Probe injection check
        assert "survival_tick_counter" in mechanics_content or "AgentTelemetryProbe" in mechanics_content

        tel_data = json.loads(telemetry_manifest_path.read_text(encoding="utf-8"))
        assert len(tel_data["probes"]) >= 1
        assert tel_data["probes"][0]["probe_name"] == "survival_tick_counter"

        assert portrait_path.stat().st_size > 0, "Portrait PNG must have non-zero file size"

        mock_sup.evaluate_and_heal.assert_called_once_with(
            candidate_or_spec=staging_dir,
            episode_number=0,
            max_retries=orch.config.max_retries,
        )

