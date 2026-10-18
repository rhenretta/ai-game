"""tests/adversarial/test_m3_it2_challenger_empirical.py
Milestone 3 Iteration 2 Challenger 1 Empirical Stress Test Suite.

Empirically verifies:
1. Empty Asset Manifest Sprite Isolation:
   - Evaluates empty manifest representations: `asset_manifest=[]`, `requirements_spec=spec(asset_manifest=[])`,
     dict `{"asset_manifest": []}`, and object with `.assets = []`.
   - Stress-tests across varied output directory structures: uncreated dirs, deeply nested paths,
     and pre-populated dirs.
   - Verifies across all palette moods that exactly 0 sprite files are generated and zero legacy
     arena sprites pollute disk.
   - Verifies manifests containing only audio assets produce 0 sprite files.
   - Verifies legacy mode activates strictly when manifests are None (negative control).
2. Multi-Track Iterative SWE Execution:
   - Verifies 3-track synthesis lifecycle (R1, R2, R3) progressing cleanly to completed.
   - Verifies 4-track execution fails fast when responses are exhausted at R3, marking R3 rejected
     and leaving R4 strictly pending (no self-certification, immediate halt).
   - Verifies track smoke failure at R2 halts execution immediately leaving R3 pending.
   - Verifies Director requirements authoring errors propagate authentic exceptions without silent swallow.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

from engine.assets.palette import MOOD_PALETTES
from engine.studio.models import (
    AcceptanceCriterion,
    AssetRequirement,
    GameRequirementsSpec,
    GenesisWorldSpec,
    RequirementTrack,
    StudioAgentError,
    StudioSwarmExecutionError,
    StudioSwarmPlan,
)
from engine.studio.orchestrator import StudioOrchestrator
from engine.studio.swe import LeadGameplaySWEAgent
from engine.studio.visual import (
    PNG_SIGNATURE,
    ProceduralSpriteRenderer,
    VisualArtSpriteDesignerAgent,
)
from engine.ui.server import SwarmEventBus
from tests.conftest import MockGeminiClient

FORBIDDEN_LEGACY_SPRITES = [
    "player_idle.png",
    "player_walk1.png",
    "player_walk2.png",
    "player_dash.png",
    "chaser.png",
    "sniper.png",
    "boss.png",
    "spikes.png",
    "acid.png",
    "wall.png",
    "floor.png",
]


class TestEmptyAssetManifestSpriteIsolationEmpirical:
    """Stress tests isolation guarantees when asset manifests are empty or non-visual."""

    def test_empty_manifest_representations_across_directory_structures(self, tmp_path: Path) -> None:
        """Verifies all representations of empty manifest produce zero files across various dir structures."""
        designer = VisualArtSpriteDesignerAgent()

        # Case 1: Direct list asset_manifest=[] in deeply nested uncreated dir
        deep_dir = tmp_path / "deep" / "nested" / "sprites"
        sprites_1 = designer.generate_cartridge_sprites(deep_dir, asset_manifest=[])
        assert sprites_1 == {}, "Expected empty dict for asset_manifest=[]"
        assert list(deep_dir.rglob("*.png")) == [], "No .png files should exist in deep_dir"
        for name in FORBIDDEN_LEGACY_SPRITES:
            assert not (deep_dir / name).exists(), f"Legacy sprite {name} leaked into deep_dir"

        # Case 2: GameRequirementsSpec with asset_manifest=[] in dir with existing file
        existing_dir = tmp_path / "pre_existing_dir"
        existing_dir.mkdir(parents=True, exist_ok=True)
        (existing_dir / "keep_me.txt").write_text("critical data", encoding="utf-8")

        spec_empty = GameRequirementsSpec(
            deliberation="Text adventure or pure logic puzzle requiring zero visual sprites.",
            game_title="Terminal Logic Grid",
            genre_paradigm="Logic Simulator",
            target_audience_summary="Analytical puzzle solver",
            requirements=[
                RequirementTrack(
                    deliberation="Core puzzle loop",
                    id="R1",
                    title="Logic Board",
                    description="Board state simulation.",
                    status="pending",
                )
            ],
            asset_manifest=[],
        )
        sprites_2 = designer.generate_cartridge_sprites(existing_dir, requirements_spec=spec_empty)
        assert sprites_2 == {}, "Expected empty dict for GameRequirementsSpec with asset_manifest=[]"
        assert (existing_dir / "keep_me.txt").read_text(encoding="utf-8") == "critical data"
        assert list(existing_dir.rglob("*.png")) == [], "No .png files should be written"
        for name in FORBIDDEN_LEGACY_SPRITES:
            assert not (existing_dir / name).exists(), f"Legacy sprite {name} leaked into pre-existing dir"

        # Case 3: Dict with asset_manifest: []
        dict_dir = tmp_path / "dict_manifest_dir"
        dict_spec = {"asset_manifest": []}
        sprites_3 = designer.generate_cartridge_sprites(dict_dir, requirements_spec=dict_spec)
        assert sprites_3 == {}, "Expected empty dict for dict with asset_manifest: []"
        assert list(dict_dir.rglob("*.png")) == []

        # Case 4: Custom object with .assets = []
        obj_dir = tmp_path / "obj_manifest_dir"
        class CustomManifestHolder:
            assets = []
        sprites_4 = designer.generate_cartridge_sprites(obj_dir, asset_manifest=CustomManifestHolder())
        assert sprites_4 == {}, "Expected empty dict for object with .assets = []"
        assert list(obj_dir.rglob("*.png")) == []

    def test_empty_manifest_palette_mood_sweep(self, tmp_path: Path) -> None:
        """Verifies that across all supported palette moods, empty manifest produces 0 sprites."""
        for mood in MOOD_PALETTES:
            designer = VisualArtSpriteDesignerAgent()
            out_dir = tmp_path / f"mood_{mood}_sprites"
            sprites = designer.generate_cartridge_sprites(out_dir, mood=mood, asset_manifest=[])
            assert sprites == {}, f"Mood {mood} must yield empty sprites dict"
            assert list(out_dir.rglob("*.png")) == [], f"Mood {mood} leaked sprites to disk"
            for name in FORBIDDEN_LEGACY_SPRITES:
                assert not (out_dir / name).exists(), f"Mood {mood} generated legacy sprite {name}"

    def test_manifest_with_only_audio_assets_produces_zero_sprites(self, tmp_path: Path) -> None:
        """Verifies that a manifest containing only non-visual (audio/sfx) assets yields zero sprites."""
        designer = VisualArtSpriteDesignerAgent()
        out_dir = tmp_path / "audio_only_sprites"
        audio_manifest = [
            AssetRequirement(
                deliberation="Main ambient synth track",
                asset_id="bgm_ambient",
                asset_type="music",
                target_path="assets/audio/ambient.wav",
                visual_prompt_or_text="Deep space drone and minimal sub pulses",
            ),
            AssetRequirement(
                deliberation="Button click sound effect",
                asset_id="sfx_click",
                asset_type="sfx",
                target_path="assets/audio/click.wav",
                visual_prompt_or_text="Crisp mechanical keystroke click",
            ),
        ]
        sprites = designer.generate_cartridge_sprites(out_dir, asset_manifest=audio_manifest)
        assert sprites == {}, "Visual designer must ignore non-visual assets and return empty dict"
        assert list(out_dir.rglob("*.png")) == [], "No sprites should be written for audio manifest"
        for name in FORBIDDEN_LEGACY_SPRITES:
            assert not (out_dir / name).exists(), f"Legacy sprite {name} generated for audio manifest"

    def test_legacy_fallback_when_manifest_unspecified(self, tmp_path: Path) -> None:
        """Negative control: when no manifest is provided (None), legacy sprites ARE generated.
        This proves the test harness assertion is active and discriminative.
        """
        designer = VisualArtSpriteDesignerAgent()
        out_dir = tmp_path / "legacy_sprites"
        sprites = designer.generate_cartridge_sprites(out_dir, asset_manifest=None, requirements_spec=None)
        assert len(sprites) >= 11, "Legacy mode must generate base procedural sprites"
        for name in ["player_idle.png", "chaser.png", "boss.png"]:
            assert (out_dir / name).exists(), f"Legacy mode should have generated {name}"


class TestMultiTrackIterativeSWEExecutionEmpirical:
    """Stress tests iterative SWE multi-track execution lifecycle, fail-fast halting, and error propagation."""

    @staticmethod
    def _create_multi_track_spec(num_tracks: int = 3) -> GameRequirementsSpec:
        tracks = []
        for i in range(1, num_tracks + 1):
            tracks.append(
                RequirementTrack(
                    deliberation=f"Requirement Track {i} deliberation",
                    id=f"R{i}",
                    title=f"Subsystem Track {i}",
                    description=f"Description of feature track {i}",
                    responsible_agents=["LeadGameplaySWEAgent"],
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation=f"AC{i}.1 criterion deliberation",
                            id=f"AC{i}.1",
                            description=f"Criterion for track {i}",
                            verification_method="ast_inspection",
                        )
                    ],
                    status="pending",
                )
            )
        return GameRequirementsSpec(
            deliberation=f"Specification with {num_tracks} discrete tracks",
            game_title="Modular Swarm Game",
            genre_paradigm="Action Strategy",
            target_audience_summary="Strategy enthusiast",
            requirements=tracks,
            asset_manifest=[],
        )

    @staticmethod
    def _create_world_spec() -> GenesisWorldSpec:
        return GenesisWorldSpec(
            deliberation="World spec for multi-track empirical verification",
            game_title="Modular Swarm Game",
            core_genre="Action Strategy",
            thematic_premise="Autonomous modular simulation",
            world_lore="Swarm simulation matrix",
            conflict_tone="tense",
            conflict_narrative="Overcome distributed obstacles",
            aesthetic_style="minimal_vector",
            palette_mood="cyber_neon",
            gameplay_pacing="fast",
            pacing_score=0.6,
            target_difficulty=1.0,
        )

    @staticmethod
    def _make_bundle_data(track_id: str) -> Dict[str, Any]:
        return {
            "deliberation": f"Code synthesis bundle for {track_id}",
            "manifest_data": {
                "title": "Modular Swarm Game",
                "genre": "Action Strategy",
                "version": "1.0.0",
                "entrypoint": "mechanics.py",
            },
            "mechanics_code": (
                '"""mechanics.py"""\n'
                'from engine.api.contracts import BaseCartridge, BaseScene\n'
                f'# Implementation for {track_id}\n'
                'class MainScene(BaseScene):\n'
                f'    deliberation: str = "Scene for {track_id}."\n'
                '    def __init__(self, engine=None):\n'
                '        super().__init__(engine)\n'
                f'        self.track_tag = "{track_id}"\n'
                '    def update(self, dt):\n'
                '        pass\n'
                '    def render(self, surface):\n'
                '        pass\n'
                'class SwarmCartridge(BaseCartridge):\n'
                f'    deliberation: str = "Cartridge for {track_id}."\n'
                '    def get_scenes(self, engine):\n'
                '        return {"main": MainScene(engine)}\n'
                '    def get_initial_scene(self, engine):\n'
                '        return self.get_scenes(engine)["main"]\n'
            ),
            "scenes_code": None,
            "entities_code": None,
            "abilities_code": None,
            "level_data": None,
        }

    @pytest.mark.anyio
    async def test_three_track_synthesis_lifecycle_progression(self, tmp_path: Path) -> None:
        """Verifies a 3-track backlog progresses all tracks from pending -> in_progress -> completed."""
        client = MockGeminiClient()
        # Queue 3 distinct bundle responses for R1, R2, R3
        client.set_json_response(self._make_bundle_data("R1"))
        client.set_json_response(self._make_bundle_data("R2"))
        client.set_json_response(self._make_bundle_data("R3"))

        orch = StudioOrchestrator(client=client)
        staging_dir = tmp_path / "staging_three_tracks"
        staging_dir.mkdir(parents=True, exist_ok=True)

        req_spec = self._create_multi_track_spec(num_tracks=3)
        (staging_dir / "requirements_spec.json").write_text(
            req_spec.model_dump_json(indent=2), encoding="utf-8"
        )

        swe = orch.agents["LeadGameplaySWEAgent"]
        world_spec = self._create_world_spec()

        results = await orch._execute_iterative_swe_loop(
            agent=swe,
            world_spec=world_spec,
            narrative_bible=None,
            req_spec=req_spec,
            staging_dir=staging_dir,
        )

        assert "mechanics.py" in results
        # All 3 tracks in memory must be completed
        for track in req_spec.requirements:
            assert track.status == "completed", f"Track {track.id} should be completed, got {track.status}"

        # Disk persistence must also show all 3 tracks completed
        saved_spec = GameRequirementsSpec.model_validate_json(
            (staging_dir / "requirements_spec.json").read_text(encoding="utf-8")
        )
        assert len(saved_spec.requirements) == 3
        for track in saved_spec.requirements:
            assert track.status == "completed", f"On disk, track {track.id} should be completed, got {track.status}"

    @pytest.mark.anyio
    async def test_four_track_fails_fast_when_mock_responses_exhausted_leaving_later_tracks_pending(
        self, tmp_path: Path
    ) -> None:
        """When 4 tracks are declared and responses run out at R3:
        R1 and R2 complete, R3 transitions to rejected, loop halts immediately, R4 remains pending.
        """
        client = MockGeminiClient()
        # Only queue 2 responses for 4 tracks
        client.set_json_response(self._make_bundle_data("R1"))
        client.set_json_response(self._make_bundle_data("R2"))

        orch = StudioOrchestrator(client=client)
        staging_dir = tmp_path / "staging_exhausted_r3"
        staging_dir.mkdir(parents=True, exist_ok=True)

        req_spec = self._create_multi_track_spec(num_tracks=4)
        (staging_dir / "requirements_spec.json").write_text(
            req_spec.model_dump_json(indent=2), encoding="utf-8"
        )

        swe = orch.agents["LeadGameplaySWEAgent"]
        world_spec = self._create_world_spec()

        with pytest.raises(StudioSwarmExecutionError) as exc_info:
            await orch._execute_iterative_swe_loop(
                agent=swe,
                world_spec=world_spec,
                narrative_bible=None,
                req_spec=req_spec,
                staging_dir=staging_dir,
            )

        # Must indicate failure of R3
        assert "Requirement R3 ('Subsystem Track 3') failed verification" in str(exc_info.value)

        # Exact status assertions
        assert req_spec.requirements[0].status == "completed", "R1 must be completed"
        assert req_spec.requirements[1].status == "completed", "R2 must be completed"
        assert req_spec.requirements[2].status == "rejected", "R3 must be rejected"
        assert req_spec.requirements[3].status == "pending", "R4 must remain strictly pending"

        # Verify disk persistence reflects exact state
        saved_spec = GameRequirementsSpec.model_validate_json(
            (staging_dir / "requirements_spec.json").read_text(encoding="utf-8")
        )
        assert saved_spec.requirements[0].status == "completed"
        assert saved_spec.requirements[1].status == "completed"
        assert saved_spec.requirements[2].status == "rejected"
        assert saved_spec.requirements[3].status == "pending"

    @pytest.mark.anyio
    async def test_track_smoke_failure_halts_immediately(self, tmp_path: Path) -> None:
        """When R2 synthesis produces broken Python syntax, smoke verification fails fast:
        R1 is completed, R2 is rejected, R3 remains pending.
        """
        client = MockGeminiClient()
        client.set_json_response(self._make_bundle_data("R1"))
        # R2 bundle with invalid syntax
        broken_bundle = self._make_bundle_data("R2")
        broken_bundle["mechanics_code"] = "def invalid_syntax_here(:\n    pass\n"
        client.set_json_response(broken_bundle)
        # R3 bundle should never be consumed
        client.set_json_response(self._make_bundle_data("R3"))

        orch = StudioOrchestrator(client=client)
        staging_dir = tmp_path / "staging_syntax_failure"
        staging_dir.mkdir(parents=True, exist_ok=True)

        req_spec = self._create_multi_track_spec(num_tracks=3)
        (staging_dir / "requirements_spec.json").write_text(
            req_spec.model_dump_json(indent=2), encoding="utf-8"
        )

        swe = orch.agents["LeadGameplaySWEAgent"]
        world_spec = self._create_world_spec()

        with pytest.raises(StudioSwarmExecutionError) as exc_info:
            await orch._execute_iterative_swe_loop(
                agent=swe,
                world_spec=world_spec,
                narrative_bible=None,
                req_spec=req_spec,
                staging_dir=staging_dir,
            )

        assert "Requirement R2" in str(exc_info.value)
        assert req_spec.requirements[0].status == "completed"
        assert req_spec.requirements[1].status == "rejected"
        assert req_spec.requirements[2].status == "pending"

        saved_spec = GameRequirementsSpec.model_validate_json(
            (staging_dir / "requirements_spec.json").read_text(encoding="utf-8")
        )
        assert saved_spec.requirements[0].status == "completed"
        assert saved_spec.requirements[1].status == "rejected"
        assert saved_spec.requirements[2].status == "pending"

    @pytest.mark.anyio
    async def test_director_requirements_authoring_error_propagates_without_silent_degradation(
        self, tmp_path: Path
    ) -> None:
        """Verifies that an error in Director requirements authoring propagates authentic errors (Rule 5)."""
        client = MockGeminiClient()
        world_spec_data = {
            "deliberation": "Genesis world spec",
            "game_title": "Test Title",
            "core_genre": "Action",
            "thematic_premise": "Premise",
            "world_lore": "Lore",
            "conflict_tone": "dark",
            "conflict_narrative": "Conflict",
            "aesthetic_style": "vector",
            "palette_mood": "cyber_neon",
            "gameplay_pacing": "fast",
            "pacing_score": 0.5,
            "target_difficulty": 1.0,
            "mechanics_guidelines": ["Core combat"],
        }
        client.set_json_response(world_spec_data)
        # Next response will be inspected by orchestrator for requirements authoring
        # We put a text response containing 'requirements' so call_agent=True
        req_resp = MagicMock()
        req_resp.text = '{"requirements": []}'
        client._responses.append(req_resp)

        plan = StudioSwarmPlan(
            deliberation="Swarm plan for Director requirements error testing",
            plan_id="plan_director_error_test",
            game_title="Test Title",
            genre_verdict="Action",
            specialist_agent_roles=["Director", "SWE"],
            dag_edges=[("Director", "SWE")],
            task_manifest={
                "Director": "Generate Genesis world specification and requirements spec",
                "SWE": "Synthesize BaseCartridge mechanics code",
            },
            target_episode=0,
        )

        orch = StudioOrchestrator(client=client)
        director = orch.agents["CreativeDirectorWorldbuilderAgent"]

        # Make Director raise an authentic API exception during generate_requirements_spec
        with patch.object(
            director,
            "generate_requirements_spec",
            side_effect=StudioAgentError("Authentic Gemini API rate limit failure"),
        ):
            with pytest.raises(Exception) as exc_info:
                await orch.execute_swarm_plan(plan, tmp_path / "genesis_test")

            # Must NOT silently swallow error or fall back
            assert "Authentic Gemini API rate limit failure" in str(exc_info.value)
