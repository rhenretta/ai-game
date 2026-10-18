"""tests/adversarial/test_m1_challenger_stress.py
Milestone M1 Empirical Challenger Adversarial Stress Suite.
Adversarial stress-testing of Milestone 1 models and requirements authoring logic:
- Epistemic Models (GameRequirementsSpec, RequirementTrack, AcceptanceCriterion, AssetRequirement, AssetManifestSpec)
- Pure Generalizability (Rule 1: zero genre/character prescriptions, abstract/character-less games)
- Rationale-First Architecture (Rule 4: AST and serialization deliberation at index 0)
- Fail-Fast Enforcement (Rule 5: Zero Silent Fallbacks on GameConceptDirectorAgent and NarrativeCharacterWriterAgent)
- Observability UI event models (RequirementsSpecPayload, RequirementStatusPayload, ChallengerGateVerdictPayload)
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

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


# ============================================================================
# 1. TestModelBoundaryAndExtremeInputs
# ============================================================================

class TestModelBoundaryAndExtremeInputs:
    """Stress tests epistemic models with boundary conditions, extreme strings, and scaling limits."""
    deliberation: str = "Empirical challenge: testing model integrity on boundary and adversarial inputs."

    @pytest.mark.parametrize("model_cls, kwargs", [
        (AcceptanceCriterion, {"description": "Test"}),
        (AssetRequirement, {"name": "test.png", "asset_type": "sprite"}),
        (AssetManifestSpec, {"game_title": "Test"}),
        (RequirementTrack, {"title": "Test", "description": "Test"}),
        (GameRequirementsSpec, {"game_title": "Test"}),
    ])
    def test_deliberation_rejects_missing_empty_and_whitespace(self, model_cls, kwargs):
        """Every epistemic model MUST reject missing, empty, or whitespace-only deliberation."""
        # 1. Missing deliberation
        with pytest.raises(ValidationError):
            model_cls(**kwargs)

        # 2. Empty string
        with pytest.raises(ValidationError):
            model_cls(deliberation="", **kwargs)

        # 3. Whitespace variations (spaces, tabs, newlines, carriage returns)
        for ws in ["   ", "\t\t", "\n\n", " \t \r\n ", "\u00a0"]:
            with pytest.raises(ValidationError):
                model_cls(deliberation=ws, **kwargs)

    def test_deliberation_accepts_valid_unicode_and_markdown(self):
        """Models accept rich multi-line Markdown, Unicode, Emojis, and RTL text without corruption."""
        adversarial_deliberations = [
            "🎮 # Design Philosophy\n\n- System 1: Inertial momentum\n- System 2: Gravitational well\n*Note*: 🚀 100% compliant.",
            "مراجعة تصميم اللعبة والتحقق من المنطق الاستراتيجي بدون شخصيات.",
            "ゲームデザインの検証：抽象的なパズルとグリッド自動化シミュレーション。",
            "Deliberation with symbols: <>&\"'\\//;;--/**/ && || == !== %s %d %x",
        ]
        for text in adversarial_deliberations:
            crit = AcceptanceCriterion(
                deliberation=text,
                description="Check Unicode handling.",
            )
            assert crit.deliberation == text
            dumped = crit.model_dump()
            assert dumped["deliberation"] == text
            loaded = AcceptanceCriterion.model_validate_json(crit.model_dump_json())
            assert loaded.deliberation == text

    def test_huge_deliberation_and_description_strings(self):
        """Models handle massive text blocks (50KB+) without crash or truncation."""
        huge_text = "Deliberation reasoning token. " * 2000  # ~60KB
        spec = GameRequirementsSpec(
            deliberation=huge_text,
            game_title="Massive Text Game",
            high_concept_summary="Summary " * 1000,
        )
        assert len(spec.deliberation) > 50000
        json_str = spec.model_dump_json()
        assert len(json_str) > 50000
        restored = GameRequirementsSpec.model_validate_json(json_str)
        assert restored.deliberation == huge_text

    def test_empty_collections_allowed_on_spec(self):
        """GameRequirementsSpec allows empty requirements and asset_manifest lists (e.g. preliminary specs)."""
        spec = GameRequirementsSpec(
            deliberation="Initial preliminary requirements spec before decomposition phase.",
            game_title="Void Explorer",
            genre_paradigm="space_sim",
            requirements=[],
            asset_manifest=[],
        )
        assert len(spec.requirements) == 0
        assert len(spec.asset_manifest) == 0
        dumped = spec.model_dump()
        assert dumped["requirements"] == []
        assert dumped["asset_manifest"] == []

    def test_scaling_stress_100_tracks_with_500_criteria(self):
        """Model efficiently scales to 100 RequirementTrack items each with 5 AcceptanceCriteria (500 criteria total)."""
        tracks = []
        for i in range(1, 101):
            criteria = [
                AcceptanceCriterion(
                    deliberation=f"Criterion deliberation for R{i}.{j}",
                    criterion_id=f"AC{i}.{j}",
                    description=f"Automated test condition for track {i} criterion {j}",
                    verification_method="runtime_telemetry" if j % 2 == 0 else "headless_playtest",
                )
                for j in range(1, 6)
            ]
            tracks.append(
                RequirementTrack(
                    deliberation=f"Track R{i} deliberation on subsystem {i}",
                    track_id=f"R{i}",
                    title=f"Subsystem {i}",
                    description=f"Detailed spec for subsystem {i}",
                    acceptance_criteria=criteria,
                    status="pending",
                )
            )

        assets = [
            AssetRequirement(
                deliberation=f"Asset {k} required for game",
                asset_id=f"sprite_entity_{k}",
                asset_type="sprite",
                dimensions=(16, 16),
            )
            for k in range(1, 101)
        ]

        spec = GameRequirementsSpec(
            deliberation="Massive complex game spec with 100 feature tracks and 100 assets.",
            game_title="Mega Sim 100",
            genre_paradigm="grand_strategy",
            requirements=tracks,
            asset_manifest=assets,
        )

        assert len(spec.requirements) == 100
        assert len(spec.asset_manifest) == 100

        # Measure serialization and deserialization fidelity
        json_output = spec.model_dump_json()
        parsed_dict = json.loads(json_output)
        assert list(parsed_dict.keys())[0] == "deliberation"

        reconstructed = GameRequirementsSpec.model_validate_json(json_output)
        assert len(reconstructed.requirements) == 100
        assert len(reconstructed.requirements[99].acceptance_criteria) == 5
        assert reconstructed.requirements[99].acceptance_criteria[4].criterion_id == "AC100.5"


# ============================================================================
# 2. TestModelNormalizationAndCoercion
# ============================================================================

class TestModelNormalizationAndCoercion:
    """Tests bidirectional field normalization, alias resolution, and Gemini dictionary coercion."""
    deliberation: str = "Empirical challenge: verifying normalization contracts and field aliases."

    def test_acceptance_criterion_normalization(self):
        """Tests that id <-> criterion_id and verification_method <-> verification_type synchronize properly."""
        # Case A: Only criterion_id and verification_type provided
        c1 = AcceptanceCriterion(
            deliberation="Criterion A",
            criterion_id="AC_ALPHA",
            description="Alpha condition",
            verification_type="ast_inspection",
        )
        assert c1.id == "AC_ALPHA"
        assert c1.criterion_id == "AC_ALPHA"
        assert c1.verification_method == "ast_inspection"
        assert c1.verification_type == "ast_inspection"

        # Case B: Only id and verification_method provided
        c2 = AcceptanceCriterion(
            deliberation="Criterion B",
            id="AC_BETA",
            description="Beta condition",
            verification_method="multi_modal",
        )
        assert c2.id == "AC_BETA"
        assert c2.criterion_id == "AC_BETA"
        assert c2.verification_method == "multi_modal"
        assert c2.verification_type == "multi_modal"

        # Case C: Neither id provided -> auto-generates ID starting with AC_
        c3 = AcceptanceCriterion(
            deliberation="Criterion C",
            description="Auto id condition",
        )
        assert c3.id.startswith("AC_")
        assert c3.criterion_id == c3.id
        assert c3.verification_method == "headless_playtest"

    def test_asset_requirement_normalization(self):
        """Tests name, asset_id, target_path, and visual_prompt_or_text auto-generation and synchronization."""
        # Case 1: Sprite with only asset_id given
        a1 = AssetRequirement(
            deliberation="Asset 1",
            asset_id="ore_crystal",
            asset_type="sprite",
        )
        assert a1.name == "ore_crystal.png"
        assert a1.asset_id == "ore_crystal"
        assert a1.target_path == "assets/sprites/ore_crystal.png"
        assert a1.visual_prompt_or_text == "Asset ore_crystal"
        assert a1.description == "Asset ore_crystal"

        # Case 2: Audio bark with only target_path given
        a2 = AssetRequirement(
            deliberation="Asset 2",
            asset_type="audio_bark",
            target_path="assets/audio/ambient_hum.wav",
        )
        assert a2.name == "ambient_hum.wav"
        assert a2.asset_id == "ambient_hum"

        # Case 3: Sfx with name given without extension
        a3 = AssetRequirement(
            deliberation="Asset 3",
            asset_type="sfx",
            name="laser_blast",
        )
        assert a3.name == "laser_blast"
        assert a3.asset_id == "laser_blast"
        assert a3.target_path == "assets/audio/laser_blast"

    def test_requirement_track_responsible_agent_normalization(self):
        """Tests singular responsible_agent string coerced to responsible_agents list."""
        t1 = RequirementTrack(
            deliberation="Track 1",
            title="Single Agent Track",
            description="Desc",
            responsible_agent="LeadGameplaySWEAgent",
        )
        assert t1.responsible_agents == ["LeadGameplaySWEAgent"]

        t2 = RequirementTrack(
            deliberation="Track 2",
            title="Multi Agent Track",
            description="Desc",
            responsible_agents=["LeadGameplaySWEAgent", "VisualArtSpriteDesignerAgent"],
        )
        assert len(t2.responsible_agents) == 2

    def test_game_requirements_spec_genre_verdict_synchronization(self):
        """Tests genre_verdict <-> genre_paradigm synchronization in GameRequirementsSpec."""
        spec1 = GameRequirementsSpec(
            deliberation="Spec 1",
            game_title="Game 1",
            genre_verdict="puzzle_automation",
        )
        assert spec1.genre_paradigm == "puzzle_automation"
        assert spec1.genre_verdict == "puzzle_automation"

        spec2 = GameRequirementsSpec(
            deliberation="Spec 2",
            game_title="Game 2",
            genre_paradigm="turn_based_tactics",
        )
        assert spec2.genre_paradigm == "turn_based_tactics"
        assert spec2.genre_verdict == "turn_based_tactics"


# ============================================================================
# 3. TestRule1GeneralizabilityAndDebranding
# ============================================================================

class TestRule1GeneralizabilityAndDebranding:
    """Stress tests Rule 1: Zero domain/genre prescriptions and pure generalizability."""
    deliberation: str = "Empirical challenge: proving domain-agnostic behavior for abstract and character-less games."

    def test_abstract_characterless_narrative_bible_and_export(self, tmp_path: Path):
        """NarrativeBible validates with 0 characters and exports valid cartridge JSON without error."""
        bible = NarrativeBible(
            deliberation="Character-less factory automation game; no characters, dialogue, or cutscenes required.",
            world_title="Autonomous Fabricator",
            setting_summary="Deep-space fully autonomous assembly mega-structure.",
            characters=[],
            dialogue_trees=[],
            dialogue_barks=[],
            cutscenes=[],
        )
        assert len(bible.characters) == 0

        # Export test
        writer = NarrativeCharacterWriterAgent(client=None)
        export_path = tmp_path / "cartridge" / "narrative.json"
        written_path = writer.export_narrative_json(bible, export_path)
        assert written_path.exists()

        content = json.loads(written_path.read_text(encoding="utf-8"))
        assert list(content.keys())[0] == "deliberation"
        assert content["world_title"] == "Autonomous Fabricator"
        assert content["characters"] == {}
        assert content["dialogue_barks"] == []

    def test_genesis_world_spec_agent_directed_defaults(self):
        """GenesisWorldSpec uses 'agent_directed' defaults rather than legacy arena framing or open arena."""
        spec = GenesisWorldSpec(
            deliberation="Abstract rhythm action game.",
            game_title="Rhythm Matrix",
            core_genre="Rhythm Arcade",
            thematic_premise="Beat synchronization",
            world_lore="Pure light frequency continuum",
            conflict_tone="heroic_hopeful",
            conflict_narrative="Harmonize divergent wave tracks",
            aesthetic_style="Minimalist neon audio visualizer",
            palette_mood="neon_cyberpunk",
            gameplay_pacing="kinetic_high_mobility",
            pacing_score=0.9,
            target_difficulty=1.2,
        )
        assert spec.camera_style == "agent_directed"
        assert spec.arena_layout_directive == "agent_directed"
        assert spec.mechanics_guidelines == []

    def test_no_forbidden_branding_or_hardcoded_prescriptions(self):
        """Scans codebase files for legacy arena names, character prescriptions, or forbidden branding tokens."""
        models_code = Path(sm.__file__).read_text(encoding="utf-8")
        director_code = Path(inspect.getfile(GameConceptDirectorAgent)).read_text(encoding="utf-8")
        narrative_code = Path(inspect.getfile(NarrativeCharacterWriterAgent)).read_text(encoding="utf-8")

        # Forbidden character prescriptions in narrative writer
        assert "hero, rival/companion, boss" not in narrative_code
        assert "for combat/low-health/boss encounters" not in narrative_code

        # Forbidden hardcoded arena sprite names in models or director prompt
        for forbidden in ["chaser.png", "sniper.png", "spikes.png", "acid.png"]:
            assert forbidden not in director_code, f"Found hardcoded sprite {forbidden} in director.py"

        # Verify debranding header in all 3 modules
        assert "Zero occurrences of legacy branding tokens" in models_code
        assert "Zero occurrences of legacy branding tokens" in director_code
        assert "Zero occurrences of legacy branding tokens" in narrative_code


# ============================================================================
# 4. TestRule4RationaleFirstEpistemicArchitecture
# ============================================================================

class TestRule4RationaleFirstEpistemicArchitecture:
    """Empirically verifies AST and runtime serialization ordering for deliberation strictly first."""
    deliberation: str = "Empirical challenge: testing deliberation-first invariant across all M1 classes."

    def test_ast_deliberation_strictly_first_statement(self):
        """AST inspection confirms deliberation is statement 0 across all 8 M1 classes."""
        targets = [
            (sm, ["AcceptanceCriterion", "AssetRequirement", "AssetManifestSpec", "RequirementTrack", "GameRequirementsSpec"]),
            (uim, ["RequirementsSpecPayload", "RequirementStatusPayload", "ChallengerGateVerdictPayload"]),
        ]
        for mod, class_names in targets:
            tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
            for c_name in class_names:
                c_node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == c_name), None)
                assert c_node is not None, f"Class {c_name} not found in {mod.__name__}"

                # First statement after optional docstring
                stmt = c_node.body[0]
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    stmt = c_node.body[1]

                attr_name = None
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    attr_name = stmt.target.id
                elif isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                    attr_name = stmt.targets[0].id

                assert attr_name == "deliberation", (
                    f"Class {c_name} violates Rule 4: first statement is '{attr_name}', not 'deliberation'."
                )

    def test_runtime_dict_and_json_order_regardless_of_kwarg_order(self):
        """Even if 'deliberation' is passed last in kwargs, model_dump() and model_dump_json() place it at index 0."""
        # Pass deliberation last
        crit = AcceptanceCriterion(
            passed=False,
            verification_method="headless_playtest",
            description="Verify order",
            id="AC1",
            deliberation="Passed as last keyword argument to constructor.",
        )
        # Check model_dump
        dumped = crit.model_dump()
        assert list(dumped.keys())[0] == "deliberation"

        # Check model_dump_json
        parsed = json.loads(crit.model_dump_json())
        assert list(parsed.keys())[0] == "deliberation"

        # Repeat for GameRequirementsSpec
        spec = GameRequirementsSpec(
            game_title="Order Game",
            genre_paradigm="tactics",
            deliberation="Spec deliberation passed last.",
        )
        assert list(spec.model_dump().keys())[0] == "deliberation"
        assert list(json.loads(spec.model_dump_json()).keys())[0] == "deliberation"


# ============================================================================
# 5. TestRule5FailFastDirectorBehavior
# ============================================================================

class TestRule5FailFastDirectorBehavior:
    """Stress tests Rule 5: Zero Silent Fallbacks and fail-fast exception behavior."""
    deliberation: str = "Empirical challenge: testing fail-fast guarantees on specialist agents."

    def test_director_fails_fast_when_client_is_none(self):
        """All director methods raise StudioAgentError immediately when client is None."""
        director = GameConceptDirectorAgent(client=None)

        # 1. generate_requirements_spec
        with pytest.raises(StudioAgentError) as exc1:
            director.generate_requirements_spec()
        assert "requires an active google.genai Client" in str(exc1.value)

        # 2. generate_genesis_world
        dna = SteamPlayerDNA(
            deliberation="DNA",
            player_id="test",
            top_genres=["Puzzle"],
        )
        with pytest.raises(StudioAgentError) as exc2:
            director.generate_genesis_world(dna=dna)
        assert "requires an active google.genai Client" in str(exc2.value)

        # 3. generate_episodic_update_world
        with pytest.raises(StudioAgentError) as exc3:
            director.generate_episodic_update_world(dna=dna, hypothesis_report=None, previous_spec=None, target_episode=1)
        assert "requires an active google.genai Client" in str(exc3.value)

    def test_director_propagates_gemini_exceptions_with_cause(self):
        """When Gemini client throws Network, Auth, or Quota errors, director wraps and preserves the cause."""
        mock_client = MagicMock()
        network_error = ConnectionResetError("Gemini socket reset")
        mock_client.models.generate_content.side_effect = network_error

        director = GameConceptDirectorAgent(client=mock_client)
        with pytest.raises(StudioAgentError) as exc_info:
            director.generate_requirements_spec()

        assert "Gemini requirements authoring failed" in str(exc_info.value) or "GameConceptDirectorAgent failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is network_error

    @pytest.mark.parametrize("bad_response_text", [
        "",
        None,
        "   \n  ",
        "Not valid JSON { missing bracket",
        json.dumps({"game_title": "No deliberation"}),
    ])
    def test_director_fails_on_empty_malformed_or_invalid_gemini_payloads(self, bad_response_text):
        """Director fails fast with StudioAgentError on empty, unparseable, or schema-violating Gemini output."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = bad_response_text
        mock_client.models.generate_content.return_value = mock_response

        director = GameConceptDirectorAgent(client=mock_client)
        with pytest.raises(StudioAgentError):
            director.generate_requirements_spec()


# ============================================================================
# 6. TestUIEventBusModelIntegrity
# ============================================================================

class TestUIEventBusModelIntegrity:
    """Stress tests UI event models and serialization."""
    deliberation: str = "Empirical challenge: testing UI event bus payloads and event types."

    def test_event_types_and_payload_round_trips(self):
        """Tests event types and round-trip serialization of RequirementsSpecPayload, RequirementStatusPayload, ChallengerGateVerdictPayload."""
        assert SwarmEventType.REQUIREMENTS_SPEC_GENERATED == "REQUIREMENTS_SPEC_GENERATED"
        assert SwarmEventType.REQUIREMENT_STATUS_UPDATE == "REQUIREMENT_STATUS_UPDATE"
        assert SwarmEventType.CHALLENGER_GATE_VERDICT == "CHALLENGER_GATE_VERDICT"

        # RequirementsSpecPayload
        p1 = RequirementsSpecPayload(
            deliberation="Event payload deliberation.",
            spec={"game_title": "Test Title", "deliberation": "Inner deliberation"},
            total_requirements=5,
        )
        assert list(p1.model_dump().keys())[0] == "deliberation"
        r1 = RequirementsSpecPayload.model_validate_json(p1.model_dump_json())
        assert r1.total_requirements == 5

        # RequirementStatusPayload
        p2 = RequirementStatusPayload(
            deliberation="Status update deliberation.",
            requirement_id="R1",
            status="in_progress",
            metrics={"headless_time_ms": 12.5},
        )
        assert list(p2.model_dump().keys())[0] == "deliberation"
        r2 = RequirementStatusPayload.model_validate_json(p2.model_dump_json())
        assert r2.requirement_id == "R1"
        assert r2.metrics["headless_time_ms"] == 12.5

        # ChallengerGateVerdictPayload
        p3 = ChallengerGateVerdictPayload(
            deliberation="Challenger gate verdict deliberation.",
            candidate_id="cand_123",
            passed=False,
            rejection_trace={"unfulfilled": ["R2"]},
        )
        assert list(p3.model_dump().keys())[0] == "deliberation"
        r3 = ChallengerGateVerdictPayload.model_validate_json(p3.model_dump_json())
        assert r3.candidate_id == "cand_123"
        assert r3.passed is False
        assert r3.rejection_trace["unfulfilled"] == ["R2"]


# ============================================================================
# 7. TestPromptConstructionAndSchemaSanitization
# ============================================================================

class TestPromptConstructionAndSchemaSanitization:
    """Stress tests director prompt generation across degenerate contexts and Gemini JSON schema sanitization."""
    deliberation: str = "Empirical challenge: testing prompt formatting robustness and Gemini API schema compliance."

    def test_director_prompt_handles_degenerate_dna_and_none_contexts(self):
        """Director handles completely None or degenerate DNA/world/hypothesis contexts without exceptions."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "deliberation": "Conceived minimal valid spec.",
            "game_title": "Minimal Game",
            "genre_paradigm": "abstract",
            "requirements": [],
            "asset_manifest": [],
        })
        mock_client.models.generate_content.return_value = mock_response
        director = GameConceptDirectorAgent(client=mock_client)

        # 1. All arguments None
        spec = director.generate_requirements_spec(dna=None, world_spec=None, hypothesis_report=None)
        assert spec.game_title == "Minimal Game"
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert "None provided." in call_kwargs["contents"]
        assert "ARCHITECTURAL & AGENTIC DIRECTIVES" in call_kwargs["contents"]

        # 2. Degenerate DNA (empty lists, 0 playtime)
        deg_dna = SteamPlayerDNA(
            deliberation="Degenerate DNA",
            player_id="deg_user",
            total_playtime_hours=0.0,
            total_games_analyzed=0,
            top_genres=[],
            top_games=[],
            thematic_biases=[],
        )
        spec2 = director.generate_requirements_spec(dna=deg_dna)
        assert spec2.game_title == "Minimal Game"

    def test_gemini_json_schema_strips_additional_properties(self):
        """Gemini Developer API requires schemas without unsupported additionalProperties keywords."""
        schema = GameRequirementsSpec.model_json_schema()

        def assert_no_additional_properties(node):
            if isinstance(node, dict):
                assert "additionalProperties" not in node, f"Found additionalProperties in schema node: {node}"
                assert "prefixItems" not in node, f"Found prefixItems in schema node: {node}"
                for v in node.values():
                    assert_no_additional_properties(v)
            elif isinstance(node, list):
                for item in node:
                    assert_no_additional_properties(item)

        assert_no_additional_properties(schema)

    def test_stringified_json_dict_coercion(self):
        """BaseEpistemicModel coerces JSON-encoded strings in Dict fields returned by Gemini."""
        from engine.studio.models import StudioSwarmPlan

        raw_data = {
            "deliberation": "Deliberation for swarm plan.",
            "game_title": "Test Title",
            "genre_verdict": "action_roguelike",
            "specialist_agent_roles": ["LeadGameplaySWEAgent"],
            "task_manifest": json.dumps({"SWE": {"directive": "build mechanics"}}),
        }
        plan = StudioSwarmPlan(**raw_data)
        assert isinstance(plan.task_manifest, dict)
        assert plan.task_manifest["SWE"]["directive"] == "build mechanics"
