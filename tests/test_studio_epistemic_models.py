"""tests/test_studio_epistemic_models.py
Comprehensive test suite verifying Rationale-First Epistemic Data Models for Studio Swarm.
Enforces Rule 4: deliberation is declared strictly first on 100% of Pydantic models,
serialized dictionaries, and JSON strings.
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
import ast
import inspect
import json
from pathlib import Path
import pytest
from pydantic import BaseModel, ValidationError

import engine.studio.models as m
from engine.studio.models import (
    AgentExecutionRecord,
    AgentPlan,
    AgentVerificationRecord,
    AudioSynthesisResult,
    BaseEpistemicModel,
    BoundedAdaptationSpec,
    CartridgeCompilationResult,
    CharacterSheet,
    CutsceneBeat,
    CutsceneScript,
    DialogueBarkSpec,
    DialogueChoice,
    DialogueNode,
    DialogueTreeSpec,
    EvolutionHypothesisReport,
    GenesisWorldSpec,
    HypothesisEvidence,
    InstrumentationResult,
    NarrativeBible,
    PlayerPreferenceHypothesis,
    StudioBuildState,
    StudioOrchestratorConfig,
    TelemetryArchitecturePlan,
    TelemetryProbeDesignSpec,
    ToolInvocationRecord,
    UpdateWorldSpec,
)


class TestStudioEpistemicModels:
    """Verifies that all studio models satisfy the Rationale-First Epistemic Architecture."""
    deliberation: str = "Test suite validating Rationale-First property ordering across studio schemas."

    def test_all_models_have_deliberation_at_field_index_zero(self) -> None:
        """Every model in engine.studio.models must declare deliberation as field 0."""
        models = [
            obj for name, obj in inspect.getmembers(m)
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel and obj is not BaseEpistemicModel
        ]
        assert len(models) >= 15, f"Expected at least 15 models, found {len(models)}"
        for model in models:
            fields = list(model.model_fields.keys())
            assert fields[0] == "deliberation", (
                f"Model {model.__name__} violates Rule 4: first field is '{fields[0]}', expected 'deliberation'."
            )

    def test_ast_inspection_verifies_rationale_first_class_attributes(self) -> None:
        """AST inspection confirms every class definition in engine/studio/models.py places deliberation first."""
        models_path = Path(m.__file__)
        tree = ast.parse(models_path.read_text(encoding="utf-8"))

        class_defs = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        assert len(class_defs) >= 15

        for c_node in class_defs:
            # Skip enum classes
            base_names = [b.id for b in c_node.bases if isinstance(b, ast.Name)]
            if "Enum" in base_names or "str" in base_names and "Enum" in base_names:
                continue

            # First attribute must be deliberation
            first_stmt = c_node.body[0]
            if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                # Skip docstring if present and check next statement
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
                f"Class {c_node.name} in models.py declares '{attr_name}' first instead of 'deliberation'"
            )

    def test_json_dump_deliberation_strictly_first_key(self) -> None:
        """Serialized JSON must emit 'deliberation' as the first key."""
        spec = GenesisWorldSpec(
            deliberation="Initial world conception deliberation.",
            game_title="Orbital Ruin",
            core_genre="Action Roguelike",
            thematic_premise="Derelict space station",
            world_lore="Ancient defense systems running rampant.",
            conflict_tone="grim_tense",
            conflict_narrative="Rogue AI lockdown.",
            aesthetic_style="Pixel art",
            palette_mood="neon_cyberpunk",
            gameplay_pacing="kinetic_high_mobility",
            pacing_score=0.8,
            target_difficulty=1.1,
        )
        json_str = spec.model_dump_json()
        data = json.loads(json_str)
        assert list(data.keys())[0] == "deliberation"

    def test_dict_dump_deliberation_strictly_first_key(self) -> None:
        """Serialized dict must emit 'deliberation' as the first key."""
        plan = AgentPlan(
            deliberation="Planning execution steps.",
            agent_name="GameplaySWE",
            objective="Compile mechanics.py",
            steps=["Step 1", "Step 2"],
        )
        d = plan.model_dump()
        assert list(d.keys())[0] == "deliberation"

    def test_validation_rejects_missing_deliberation(self) -> None:
        """Pydantic raises ValidationError if deliberation is missing."""
        with pytest.raises(ValidationError):
            GenesisWorldSpec(
                game_title="No Deliberation Game",
                core_genre="Action",
                thematic_premise="Premise",
                world_lore="Lore",
                conflict_tone="grim_tense",
                aesthetic_style="Style",
                gameplay_pacing="balanced",
                pacing_score=0.0,
                target_difficulty=1.0,
            )

    def test_validation_rejects_empty_deliberation(self) -> None:
        """Pydantic raises ValidationError if deliberation is an empty string."""
        with pytest.raises(ValidationError):
            AgentPlan(
                deliberation="",
                agent_name="Tester",
                objective="Test",
                steps=["Step 1"],
            )

    def test_validation_rejects_whitespace_only_deliberation(self) -> None:
        """Pydantic raises ValidationError if deliberation is only whitespace."""
        with pytest.raises(ValidationError):
            ToolInvocationRecord(
                deliberation="   \t\n  ",
                tool_name="synthesize",
            )

    def test_massive_deliberation_text_supported(self) -> None:
        """Massive 50,000 character deliberation is safely validated and preserved."""
        large_delib = "Deliberating extensively on systemic trade-offs... " * 1000
        assert len(large_delib) > 40000
        rec = AgentVerificationRecord(
            deliberation=large_delib,
            agent_name="Supervisor",
            status="VERIFIED",
            passed=True,
        )
        assert rec.deliberation == large_delib
        assert rec.model_dump()["deliberation"] == large_delib

    def test_agent_plan_rejects_empty_steps(self) -> None:
        """AgentPlan must have at least one step in steps list."""
        with pytest.raises(ValidationError):
            AgentPlan(
                deliberation="Plan with zero steps",
                agent_name="SWE",
                objective="Fail",
                steps=[],
            )

    def test_narrative_bible_rejects_empty_characters(self) -> None:
        """NarrativeBible requires at least one character in characters list."""
        with pytest.raises(ValidationError):
            NarrativeBible(
                deliberation="Bible without characters",
                world_title="Empty World",
                setting_summary="Deserted void",
                characters=[],
            )

    def test_nested_models_preserve_rationale_first(self) -> None:
        """Nested sub-models (e.g. CutsceneBeat, DialogueChoice, HypothesisEvidence) also declare deliberation first."""
        beat = CutsceneBeat(
            deliberation="Beat deliberation",
            beat_number=1,
            action_description="Door slides open",
        )
        assert list(beat.model_dump().keys())[0] == "deliberation"

        choice = DialogueChoice(
            deliberation="Choice deliberation",
            text="Take alternate path",
            target_node_id="node_alt",
        )
        assert list(choice.model_dump().keys())[0] == "deliberation"

        evidence = HypothesisEvidence(
            deliberation="Empirical observation of velocity",
            metric_name="mean_speed",
            observed_value=210.5,
        )
        assert list(evidence.model_dump().keys())[0] == "deliberation"
