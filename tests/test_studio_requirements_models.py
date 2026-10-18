"""tests/test_studio_requirements_models.py
Unit tests verifying GameRequirementsSpec, RequirementTrack, AcceptanceCriterion,
AssetRequirement, AssetManifestSpec, and generalizable studio models for Milestone 1.
Enforces Rule 1 (Pure Generalizability), Rule 4 (Rationale-First Epistemic Architecture),
and Rule 5 (Zero Silent Fallbacks).
"""
from __future__ import annotations
import ast
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock
import pytest
from pydantic import BaseModel, ValidationError

import engine.studio.models as sm
from engine.studio.models import (
    AcceptanceCriterion,
    AssetManifestSpec,
    AssetRequirement,
    BaseEpistemicModel,
    GameRequirementsSpec,
    GenesisWorldSpec,
    NarrativeBible,
    RequirementTrack,
    StudioAgentError,
)
from engine.studio.director import GameConceptDirectorAgent, CreativeDirectorWorldbuilderAgent
from engine.studio.narrative import NarrativeCharacterWriterAgent
from engine.steam.models import SteamPlayerDNA
import engine.ui.models as uim
from engine.ui.models import (
    SwarmEventType,
    RequirementsSpecPayload,
    RequirementStatusPayload,
    ChallengerGateVerdictPayload,
)


class TestStudioRequirementsModels:
    """Verifies that all requirements models satisfy Rule 4 (deliberation first) and Rule 1."""
    deliberation: str = "Test suite validating requirements and asset models for Milestone 1."

    def test_requirements_models_declare_deliberation_at_field_index_zero(self) -> None:
        """AcceptanceCriterion, AssetRequirement, AssetManifestSpec, RequirementTrack, GameRequirementsSpec
        must strictly declare 'deliberation' as field index 0.
        """
        target_models = [
            AcceptanceCriterion,
            AssetRequirement,
            AssetManifestSpec,
            RequirementTrack,
            GameRequirementsSpec,
        ]
        for model_cls in target_models:
            fields = list(model_cls.model_fields.keys())
            assert fields[0] == "deliberation", (
                f"Model {model_cls.__name__} violates Rule 4: first field is '{fields[0]}', expected 'deliberation'."
            )

    def test_ast_inspection_confirms_deliberation_as_first_statement(self) -> None:
        """AST inspection verifies that the class body of each requirements model places deliberation first."""
        models_path = Path(sm.__file__)
        tree = ast.parse(models_path.read_text(encoding="utf-8"))

        target_class_names = {
            "AcceptanceCriterion",
            "AssetRequirement",
            "AssetManifestSpec",
            "RequirementTrack",
            "GameRequirementsSpec",
        }

        class_defs = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in target_class_names]
        assert len(class_defs) == len(target_class_names)

        for c_node in class_defs:
            first_stmt = c_node.body[0]
            if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                # Skip docstring if present
                first_stmt = c_node.body[1]

            attr_name = None
            if isinstance(first_stmt, ast.AnnAssign) and isinstance(first_stmt.target, ast.Name):
                attr_name = first_stmt.target.id
            elif isinstance(first_stmt, ast.Assign):
                for target in first_stmt.targets:
                    if isinstance(target, ast.Name):
                        attr_name = target.id
                        break

            assert attr_name == "deliberation", (
                f"Class {c_node.name} declares '{attr_name}' first instead of 'deliberation' (Rule 4 violation)"
            )

    def test_acceptance_criterion_instantiation_and_serialization(self) -> None:
        """AcceptanceCriterion instantiates cleanly with both id/criterion_id and verification_method."""
        crit = AcceptanceCriterion(
            deliberation="Verify player movement response within 16ms frame budget.",
            criterion_id="AC1.1",
            description="Player velocity vector responds to WASD key presses.",
            verification_method="headless_playtest",
        )
        assert crit.deliberation.startswith("Verify player movement")
        assert crit.criterion_id == "AC1.1"
        assert crit.id == "AC1.1"
        assert crit.verification_method == "headless_playtest"
        assert crit.verification_type == "headless_playtest"
        assert crit.passed is False

        # Alias support via dict
        crit_alias = AcceptanceCriterion(
            deliberation="Criterion using id and verification_type aliases.",
            id="AC1.2",
            description="Collision check with level boundaries.",
            verification_type="ast_inspection",
        )
        assert crit_alias.criterion_id == "AC1.2"
        assert crit_alias.verification_method == "ast_inspection"

        # Check serialization key ordering
        dumped = crit.model_dump()
        assert list(dumped.keys())[0] == "deliberation"

        json_str = crit.model_dump_json()
        data = json.loads(json_str)
        assert list(data.keys())[0] == "deliberation"

    def test_acceptance_criterion_rejects_missing_or_empty_deliberation(self) -> None:
        """AcceptanceCriterion raises ValidationError if deliberation is missing or empty."""
        with pytest.raises(ValidationError):
            AcceptanceCriterion(
                description="Missing deliberation criterion",
            )
        with pytest.raises(ValidationError):
            AcceptanceCriterion(
                deliberation="",
                description="Empty deliberation criterion",
            )
        with pytest.raises(ValidationError):
            AcceptanceCriterion(
                deliberation="   \t  ",
                description="Whitespace deliberation criterion",
            )

    def test_asset_requirement_instantiation_and_serialization(self) -> None:
        """AssetRequirement captures sprite/audio asset metadata with deliberation strictly first."""
        asset = AssetRequirement(
            deliberation="Dynamic sprite required for fuel canister collectable.",
            asset_id="sprite_fuel_canister",
            asset_type="sprite",
            target_path="assets/sprites/fuel_canister.png",
            visual_prompt_or_text="Pixel art glowing blue cryogenic fuel canister, 32x32, top-down perspective",
            dimensions=(32, 32),
            status="pending",
        )
        assert asset.asset_id == "sprite_fuel_canister"
        assert asset.dimensions == (32, 32)
        assert asset.status == "pending"

        dumped = asset.model_dump()
        assert list(dumped.keys())[0] == "deliberation"
        json_data = json.loads(asset.model_dump_json())
        assert list(json_data.keys())[0] == "deliberation"

    def test_asset_manifest_spec_aggregation(self) -> None:
        """AssetManifestSpec aggregates multiple dynamic AssetRequirement entries."""
        manifest = AssetManifestSpec(
            deliberation="Dynamic asset manifest for tactical space salvage game.",
            game_title="Void Salvage",
            assets=[
                AssetRequirement(
                    deliberation="Player salvage vessel sprite.",
                    asset_id="sprite_vessel",
                    asset_type="sprite",
                    target_path="assets/sprites/vessel.png",
                    visual_prompt_or_text="Top-down industrial salvage tugboat with grapple arm",
                ),
                AssetRequirement(
                    deliberation="Audio cue on docking clamp lock.",
                    asset_id="sfx_clamp_lock",
                    asset_type="sfx",
                    target_path="assets/audio/clamp_lock.wav",
                    visual_prompt_or_text="Heavy metallic pneumatic latch locking into place",
                ),
            ],
        )
        assert len(manifest.assets) == 2
        d = manifest.to_dict()
        assert list(d.keys())[0] == "deliberation"
        assert d["game_title"] == "Void Salvage"

    def test_requirement_track_decomposition_and_normalization(self) -> None:
        """RequirementTrack holds ordered acceptance criteria and normalizes fields."""
        track = RequirementTrack(
            deliberation="Track R1: Core orbital physics simulation and thrust controls.",
            track_id="R1",
            title="Inertial Flight Dynamics",
            description="Implement Newtonian momentum, thruster firing, and velocity damping.",
            responsible_agents=["LeadGameplaySWEAgent"],
            target_modules=["mechanics.py", "entities.py"],
            acceptance_criteria=[
                AcceptanceCriterion(
                    deliberation="Verify momentum conservation when thrusters disengaged.",
                    criterion_id="AC1.1",
                    description="Vessel maintains constant velocity in vacuum when keys released.",
                    verification_method="headless_playtest",
                ),
                AcceptanceCriterion(
                    deliberation="Verify fuel expenditure during burn.",
                    criterion_id="AC1.2",
                    description="Fuel counter decreases proportionally to thruster active frames.",
                    verification_method="runtime_telemetry",
                ),
            ],
            status="in_progress",
        )
        assert track.id == "R1"
        assert len(track.acceptance_criteria) == 2
        assert track.status == "in_progress"

        dumped = track.model_dump()
        assert list(dumped.keys())[0] == "deliberation"

        # Alias support: single string responsible_agent normalized to list
        track_alias = RequirementTrack(
            deliberation="Track using responsible_agent alias.",
            id="R2",
            title="Docking Mechanics",
            description="Docking sequence with derelict hulls.",
            responsible_agent="LeadGameplaySWEAgent",
        )
        assert track_alias.track_id == "R2"
        assert track_alias.responsible_agents == ["LeadGameplaySWEAgent"]

    def test_game_requirements_spec_full_round_trip(self) -> None:
        """GameRequirementsSpec supports full round-trip serialization and adheres to Rule 4."""
        spec = GameRequirementsSpec(
            deliberation="Conceiving orbital automation simulator grounded in player preference for Factorio and Kerbal.",
            game_title="Orbital Conveyor",
            genre_paradigm="factory_automation_sim",
            target_audience_summary="Player enjoys methodical logistics and supply-chain routing.",
            high_concept_summary="Build automated conveyor pipelines connecting asteroid mining rigs to orbital refinery stations.",
            target_episode=0,
            requirements=[
                RequirementTrack(
                    deliberation="R1: Asteroid mining rig entity and drill cycle.",
                    track_id="R1",
                    title="Resource Extraction",
                    description="Mining drill extracts ore deposits per interval.",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="Drill increments extracted ore count.",
                            criterion_id="AC1.1",
                            description="Ore counter advances every 60 simulation ticks.",
                            verification_method="runtime_telemetry",
                        )
                    ],
                ),
                RequirementTrack(
                    deliberation="R2: Conduit logistics transport network.",
                    track_id="R2",
                    title="Conduit Pipeline Routing",
                    description="Connect drills to refinery using directional conduit nodes.",
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="Verify ore transfer along connected conduits.",
                            criterion_id="AC2.1",
                            description="Ore moves along path to refinery destination.",
                            verification_method="headless_playtest",
                        )
                    ],
                ),
            ],
            asset_manifest=[
                AssetRequirement(
                    deliberation="Ore chunk sprite for conveyor rendering.",
                    asset_id="sprite_iron_ore",
                    asset_type="sprite",
                    target_path="assets/sprites/iron_ore.png",
                    visual_prompt_or_text="Rough metallic iron ore nugget with reflective facets, 16x16",
                    dimensions=(16, 16),
                )
            ],
        )

        assert spec.game_title == "Orbital Conveyor"
        assert spec.genre_verdict == "factory_automation_sim"
        assert len(spec.requirements) == 2
        assert len(spec.asset_manifest) == 1

        # Dict dump
        d = spec.model_dump()
        assert list(d.keys())[0] == "deliberation"

        # JSON dump
        json_str = spec.model_dump_json(indent=2)
        parsed = json.loads(json_str)
        assert list(parsed.keys())[0] == "deliberation"

        # Re-parse into model
        reconstructed = GameRequirementsSpec(**parsed)
        assert reconstructed.game_title == "Orbital Conveyor"
        assert len(reconstructed.requirements) == 2
        assert reconstructed.requirements[0].track_id == "R1"
        assert reconstructed.requirements[0].acceptance_criteria[0].criterion_id == "AC1.1"

    def test_genesis_world_spec_defaults_are_generalizable(self) -> None:
        """GenesisWorldSpec removes hardcoded arena framing and layout defaults (Rule 1)."""
        spec = GenesisWorldSpec(
            deliberation="Conceiving an abstract puzzle mechanics game.",
            game_title="Geometric Cascade",
            core_genre="Physics Puzzler",
            thematic_premise="Manipulate light prisms and refractive crystals.",
            world_lore="Prismatic harmonics holding the dimensional planes together.",
            conflict_tone="cozy_exploratory",
            conflict_narrative="Align fractured prism nodes to restore harmony.",
            aesthetic_style="Minimalist vector neon crystal geometry.",
            palette_mood="neon_cyberpunk",
            gameplay_pacing="tactical_methodical",
            pacing_score=-0.75,
            target_difficulty=0.8,
        )
        # Default camera and layout should be agent-directed, not hardcoded open_arena
        assert spec.camera_style == "agent_directed"
        assert spec.arena_layout_directive == "agent_directed"

        # Explicit non-arena styles should also be fully supported
        custom_spec = GenesisWorldSpec(
            deliberation="Modular grid spatial simulation.",
            game_title="Modular Grid",
            core_genre="Automation Sim",
            thematic_premise="Assembly lines",
            world_lore="Factory floor",
            conflict_tone="heroic_hopeful",
            conflict_narrative="Solve resource deficits",
            aesthetic_style="Isometric industrial sprites",
            palette_mood="retro_arcade",
            gameplay_pacing="balanced",
            pacing_score=0.0,
            target_difficulty=1.0,
            camera_style="isometric_fixed",
            arena_layout_directive="procedural_grid",
        )
        assert custom_spec.camera_style == "isometric_fixed"
        assert custom_spec.arena_layout_directive == "procedural_grid"

    def test_narrative_bible_allows_zero_characters(self) -> None:
        """NarrativeBible allows empty characters list for abstract/systems/puzzle games (Rule 1)."""
        bible = NarrativeBible(
            deliberation="Deliberation for character-less automation game without narrative characters.",
            world_title="Automation Grid",
            setting_summary="Autonomous orbital mining station with no living crew.",
            characters=[],
        )
        assert len(bible.characters) == 0
        narrative_dict = bible.to_cartridge_narrative_dict()
        assert narrative_dict["characters"] == {}
        assert narrative_dict["dialogue_barks"] == []

    def test_game_concept_director_agent_exports_and_aliases(self) -> None:
        """Both GameConceptDirectorAgent and CreativeDirectorWorldbuilderAgent are exported."""
        assert GameConceptDirectorAgent is not None
        assert CreativeDirectorWorldbuilderAgent is not None
        # They should be interchangeable
        agent1 = GameConceptDirectorAgent(client=None)
        agent2 = CreativeDirectorWorldbuilderAgent(client=None)
        assert isinstance(agent1, GameConceptDirectorAgent)
        assert isinstance(agent2, CreativeDirectorWorldbuilderAgent)
        assert isinstance(agent1, CreativeDirectorWorldbuilderAgent)

    def test_game_concept_director_generate_requirements_spec_fails_without_client(self) -> None:
        """GameConceptDirectorAgent.generate_requirements_spec raises StudioAgentError when client is None (Rule 5)."""
        director = GameConceptDirectorAgent(client=None)
        dna = SteamPlayerDNA(
            deliberation="Player DNA summary",
            player_id="test_player",
            total_playtime_hours=150.0,
            top_genres=["Strategy", "Simulation"],
            pacing_score=-0.5,
            challenge_tolerance=0.7,
        )
        with pytest.raises(StudioAgentError) as exc_info:
            director.generate_requirements_spec(dna=dna)
        assert "requires an active google.genai Client" in str(exc_info.value)

    def test_game_concept_director_generate_requirements_spec_happy_path(self) -> None:
        """GameConceptDirectorAgent.generate_requirements_spec authors valid GameRequirementsSpec with mock Gemini."""
        mock_client = MagicMock()
        mock_response = MagicMock()

        spec_data = {
            "deliberation": "Player DNA demonstrates high affinity for tactical turn-based strategy; authoring 3 requirement tracks.",
            "spec_id": "req_tactical_001",
            "game_title": "Tactical Sector 9",
            "genre_paradigm": "turn_based_tactics",
            "target_audience_summary": "Players who love methodical grid combat and unit positioning.",
            "high_concept_summary": "Squad-level turn-based tactics across hex grid sectors.",
            "target_episode": 0,
            "requirements": [
                {
                    "deliberation": "R1: Hexagonal movement grid and action point economy.",
                    "track_id": "R1",
                    "title": "Grid Navigation & AP Economy",
                    "description": "Units move on hex grid consuming action points.",
                    "responsible_agents": ["LeadGameplaySWEAgent"],
                    "target_modules": ["mechanics.py", "scenes.py"],
                    "acceptance_criteria": [
                        {
                            "deliberation": "Action points decrement per hex traversed.",
                            "criterion_id": "AC1.1",
                            "description": "AP is reduced by 1 for each adjacent hex step.",
                            "verification_method": "headless_playtest",
                            "passed": False,
                        }
                    ],
                    "status": "pending",
                }
            ],
            "asset_manifest": [
                {
                    "deliberation": "Tactical squad member sprite.",
                    "asset_id": "sprite_unit_recon",
                    "asset_type": "sprite",
                    "target_path": "assets/sprites/unit_recon.png",
                    "visual_prompt_or_text": "Top-down tactical operator with blue chevron marking, 32x32",
                    "dimensions": [32, 32],
                    "status": "pending",
                }
            ],
        }

        mock_response.text = json.dumps(spec_data)
        mock_client.models.generate_content.return_value = mock_response

        director = GameConceptDirectorAgent(client=mock_client)
        dna = SteamPlayerDNA(
            deliberation="Player DNA summary",
            player_id="test_player",
            total_playtime_hours=200.0,
            top_genres=["Strategy"],
            pacing_score=-0.8,
            challenge_tolerance=0.8,
        )

        result = director.generate_requirements_spec(dna=dna)
        assert isinstance(result, GameRequirementsSpec)
        assert result.deliberation.startswith("Player DNA demonstrates")
        assert result.game_title == "Tactical Sector 9"
        assert len(result.requirements) == 1
        assert result.requirements[0].track_id == "R1"
        assert len(result.asset_manifest) == 1
        assert result.asset_manifest[0].asset_id == "sprite_unit_recon"

    def test_narrative_character_writer_prompt_contains_no_hardcoded_boss_archetypes(self) -> None:
        """NarrativeCharacterWriterAgent prompt should not mandate 'hero, rival/companion, boss' or combat situations."""
        writer_path = Path(inspect.getfile(NarrativeCharacterWriterAgent))
        code_text = writer_path.read_text(encoding="utf-8")

        # The old offending string was: "Create character sheets (hero, rival/companion, boss), reactive dialogue barks for combat/low-health/boss encounters"
        assert "Create character sheets (hero, rival/companion, boss)" not in code_text, (
            "narrative.py still contains hardcoded hero/rival/boss character prescriptions (Rule 1 violation)"
        )
        assert "for combat/low-health/boss encounters" not in code_text, (
            "narrative.py still contains hardcoded combat/boss dialogue bark prescriptions (Rule 1 violation)"
        )

    def test_ui_swarm_event_types_include_milestone_1_events(self) -> None:
        """SwarmEventType contains REQUIREMENTS_SPEC_GENERATED, REQUIREMENT_STATUS_UPDATE, and CHALLENGER_GATE_VERDICT."""
        assert "REQUIREMENTS_SPEC_GENERATED" in SwarmEventType.__members__
        assert "REQUIREMENT_STATUS_UPDATE" in SwarmEventType.__members__
        assert "CHALLENGER_GATE_VERDICT" in SwarmEventType.__members__
        assert SwarmEventType.REQUIREMENTS_SPEC_GENERATED.value == "REQUIREMENTS_SPEC_GENERATED"
        assert SwarmEventType.REQUIREMENT_STATUS_UPDATE.value == "REQUIREMENT_STATUS_UPDATE"
        assert SwarmEventType.CHALLENGER_GATE_VERDICT.value == "CHALLENGER_GATE_VERDICT"

    def test_ui_payload_schemas_rationale_first(self) -> None:
        """RequirementsSpecPayload, RequirementStatusPayload, ChallengerGateVerdictPayload have deliberation first."""
        payload_classes = [
            RequirementsSpecPayload,
            RequirementStatusPayload,
            ChallengerGateVerdictPayload,
        ]
        for cls in payload_classes:
            fields = list(cls.model_fields.keys())
            assert fields[0] == "deliberation", f"{cls.__name__} field 0 must be 'deliberation' (Rule 4)"

            # Verify missing deliberation raises ValidationError
            with pytest.raises(ValidationError):
                cls(deliberation="")

            with pytest.raises(ValidationError):
                cls(deliberation="   \t  ")

        # Verify dump orders
        req_payload = RequirementsSpecPayload(
            deliberation="Requirements specification generated by GameConceptDirector.",
            spec={"spec_id": "req_001", "game_title": "Test Game"},
            spec_id="req_001",
            game_title="Test Game",
            total_requirements=3,
        )
        assert list(req_payload.model_dump().keys())[0] == "deliberation"
        assert list(json.loads(req_payload.model_dump_json()).keys())[0] == "deliberation"

        stat_payload = RequirementStatusPayload(
            deliberation="Track R1 completed headless compilation smoke check.",
            requirement_id="R1",
            status="completed",
            metrics={"compilation_pass": True},
        )
        assert list(stat_payload.model_dump().keys())[0] == "deliberation"

        gate_payload = ChallengerGateVerdictPayload(
            deliberation="Challenger gate audited all 3 acceptance criteria with zero defects.",
            candidate_id="cand_001",
            passed=True,
            details={"audited_tracks": 3},
        )
        assert list(gate_payload.model_dump().keys())[0] == "deliberation"
