"""engine/studio/models.py
Rationale-First Epistemic Data Models for Autonomous Studio Harness.
Enforces Rule 4: deliberation is declared strictly first on every single model.
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator


def _sanitize_gemini_json_schema(node: Any) -> Any:
    """Recursively strips 'additionalProperties' and 'prefixItems' for Gemini Developer API compatibility,
    and converts open-ended object dicts (with no explicit properties) to JSON-encoded string fields.
    """
    if isinstance(node, dict):
        node.pop("additionalProperties", None)
        if "prefixItems" in node:
            prefix = node.pop("prefixItems")
            if "items" not in node and isinstance(prefix, list) and prefix:
                node["items"] = _sanitize_gemini_json_schema(prefix[0])
        for k, v in list(node.items()):
            node[k] = _sanitize_gemini_json_schema(v)
        if node.get("type") == "object" and not node.get("properties"):
            desc = node.get("description", "")
            node["type"] = "string"
            node["description"] = f"{desc} (Provide as a valid JSON-encoded object string)".strip()
    elif isinstance(node, list):
        return [_sanitize_gemini_json_schema(item) for item in node]
    return node


class StudioBuildState(str, Enum):
    """Authoritative State Machine States for Studio Orchestrator."""
    IDLE = "IDLE"
    ANALYZING = "ANALYZING"
    SYNTHESIZING = "SYNTHESIZING"
    VERIFYING = "VERIFYING"
    PROMOTED = "PROMOTED"


class StudioAgentError(Exception):
    """Base exception for autonomous studio agent failures (Rule 5: Zero Silent Fallbacks)."""
    deliberation: str = "Authentic epistemic failure surfaced to observability traces."


class StudioAgentConfigurationError(StudioAgentError):
    """Raised when an agent is invoked without required credentials or configuration."""
    deliberation: str = "Raised when agent configuration or required client is missing."


class StudioSwarmPlanError(StudioAgentError):
    """Raised when live Gemini meta-planning fails or yields an invalid DAG/schema."""
    deliberation: str = "Raised when live Gemini meta-planning fails or returns invalid schema."


class StudioSwarmExecutionError(StudioAgentError):
    """Raised when specialist agent execution fails during swarm DAG execution."""
    deliberation: str = "Raised when specialist execution fails during swarm DAG execution."


class BaseEpistemicModel(BaseModel):
    """Base model enforcing Pre-Decision Token Generation (Rule 4).
    All models must declare deliberation as field index 0.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic justification and reasoning.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_json_string_dicts(cls, data: Any) -> Any:
        """Automatically parses JSON-encoded object strings returned by Gemini for Dict-typed fields."""
        if not isinstance(data, dict):
            return data
        dict_field_names = {
            "task_manifest",
            "nodes",
            "payload_schema",
            "custom_probe_summaries",
            "manifest_data",
            "level_data",
            "arguments",
            "manifest",
            "file_hashes",
            "instrumented_code",
            "clamp_state",
            "adapted_parameters",
            "updated_episode_config",
        }
        out = dict(data)
        for fname in dict_field_names:
            if fname in out and isinstance(out[fname], str):
                raw_str = out[fname].strip()
                if not raw_str:
                    out[fname] = {}
                    continue
                try:
                    parsed = json.loads(raw_str)
                    if isinstance(parsed, dict):
                        out[fname] = parsed
                    elif isinstance(parsed, list) and fname == "nodes":
                        out[fname] = {
                            (item.get("node_id") or f"node_{idx}"): item
                            for idx, item in enumerate(parsed)
                            if isinstance(item, dict)
                        }
                    else:
                        out[fname] = {"value": parsed}
                except Exception:
                    out[fname] = {"directive": raw_str}
        return out

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
        return _sanitize_gemini_json_schema(schema)

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("deliberation cannot be empty or whitespace-only")
        return v

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        data = super().model_dump(*args, **kwargs)
        if "deliberation" in data:
            ordered: Dict[str, Any] = {"deliberation": data["deliberation"]}
            for k, v in data.items():
                if k != "deliberation":
                    ordered[k] = v
            return ordered
        return data

    def model_dump_json(
        self,
        indent: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Serializes model to JSON string, guaranteeing 'deliberation' at key index 0 (Rule 4).
        Decouples JSON formatting kwargs (indent) from Pydantic model_dump kwargs.
        """
        if "indent" in kwargs:
            indent = kwargs.pop("indent")
        mode = kwargs.pop("mode", "json")
        dumped_dict = self.model_dump(*args, mode=mode, **kwargs)
        return json.dumps(dumped_dict, indent=indent)



class StudioOrchestratorConfig(BaseEpistemicModel):
    deliberation: str = Field(default="Studio orchestrator configuration", description="Epistemic rationale")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    model_name: str = Field(default="gemini-3.8-flash", description="Default model for text/code reasoning")
    tts_model_name: str = Field(default="gemini-3.8-flash-tts", description="Model for voice synthesis")
    image_model_name: str = Field(default="gemini-3.1-flash-lite-image", description="Model for image generation")
    max_retries: int = Field(default=3, description="Maximum self-healing repair iterations")
    timeout_seconds: float = Field(default=300.0, description="Per-agent inference timeout")


class AgentPlan(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale justifying plan steps and expected outputs.")
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    agent_name: str = Field(..., description="Name of the planning agent")
    objective: str = Field(..., description="High-level goal of this execution cycle")
    steps: List[str] = Field(..., description="Ordered non-empty action steps")
    expected_artifacts: List[str] = Field(default_factory=list, description="Expected artifact filenames")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("steps")
    @classmethod
    def validate_steps_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Plan must contain at least one step")
        return v


class ToolInvocationRecord(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale justifying this tool invocation.")
    invocation_id: str = Field(default_factory=lambda: f"tool_{uuid.uuid4().hex[:8]}")
    tool_name: str = Field(..., description="Invoked tool identifier")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Input arguments")
    duration_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    status: str = Field(default="success", description="'success' | 'failed'")
    error_message: Optional[str] = Field(default=None, description="Error details if failed")
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentExecutionRecord(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale evaluating overall agent execution results.")
    execution_id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:8]}")
    agent_name: str = Field(..., description="Producing agent identifier")
    plan_id: str = Field(..., description="Associated plan identifier")
    tool_invocations: List[ToolInvocationRecord] = Field(default_factory=list)
    raw_output: Any = Field(default=None, description="Raw agent output payload")
    completed_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentVerificationRecord(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic evaluation of verification criteria and detected violations.")
    verification_id: str = Field(default_factory=lambda: f"veri_{uuid.uuid4().hex[:8]}")
    agent_name: str = Field(..., description="Verified agent identifier")
    status: str = Field(default="VERIFIED", description="'VERIFIED' | 'VERIFICATION_FAILED'")
    passed: bool = Field(..., description="True if output met 100% of schema & quality criteria")
    violations: List[str] = Field(default_factory=list, description="List of detected defects")
    attempt_number: int = Field(default=1, description="Current retry attempt number")
    verified_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GenesisWorldSpec(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale grounding world lore, conflict, and aesthetic style.")
    spec_id: str = Field(default_factory=lambda: f"world_spec_{uuid.uuid4().hex[:8]}")
    target_episode: int = Field(default=0)
    game_title: str = Field(..., description="Evocative title for the game.")
    core_genre: str = Field(..., description="Primary genre blend, e.g. 'Action Roguelike'.")
    thematic_premise: str = Field(..., description="High-concept setting and thematic hook.")
    world_lore: str = Field(..., description="Detailed lore describing world history, factions, and central conflict.")
    conflict_tone: str = Field(..., description="'grim_tense' | 'unforgiving_apocalyptic' | 'heroic_hopeful' | 'cozy_exploratory'.")
    conflict_narrative: str = Field(..., description="Core protagonist conflict and antagonist motivation")
    aesthetic_style: str = Field(..., description="Visual art direction description for sprite/tile generation.")
    palette_mood: str = Field(..., description="Canonical mood: 'neon_cyberpunk' | 'grim_dark' | 'retro_arcade' | 'vibrant_fantasy'.")
    palette_indices: List[int] = Field(default_factory=lambda: [0, 1, 8, 12], description="Color indices from STYLE_BIBLE_16_PALETTE.")
    gameplay_pacing: str = Field(..., description="'kinetic_high_mobility' | 'tactical_methodical' | 'balanced'.")
    pacing_score: float = Field(..., description="Continuous kinetic-tactical metric [-1.0, 1.0].")
    target_difficulty: float = Field(..., description="Normalized baseline difficulty in [0.5, 1.5].")
    camera_style: str = Field(default="agent_directed", description="Presentation or camera perspective determined by game concept (e.g. 'follow_player', 'static_view', 'scrolling', 'isometric', 'arena_framing', 'agent_directed').")
    mechanics_guidelines: List[str] = Field(default_factory=list, description="Directives for Gameplay SWE agent.")
    arena_layout_directive: str = Field(default="agent_directed", description="Spatial layout or environment structure determined by game concept (e.g. 'open_field', 'procedural_grid', 'modular_rooms', 'boundless', 'open_arena', 'agent_directed').")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class UpdateWorldSpec(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale detailing how player telemetry observations drive evolution.")
    spec_id: str = Field(default_factory=lambda: f"update_spec_{uuid.uuid4().hex[:8]}")
    target_episode: int = Field(..., description="Target episode number (>= 1).")
    previous_episode: int = Field(..., description="Preceding episode number.")
    narrative_progression: str = Field(..., description="Story development continuing the previous episode's climax.")
    escalation_factor: float = Field(..., description="Difficulty escalation factor bounded within +/-15%.")
    new_environmental_threats: List[str] = Field(default_factory=list)
    pacing_adjustment: float = Field(default=0.0, description="Pacing delta bounded within [-0.15, +0.15].")
    mechanics_evolution_directives: List[str] = Field(default_factory=list, description="Directives for Gameplay SWE agent.")
    hypotheses_addressed: List[str] = Field(default_factory=list, description="Player preferences addressed.")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CartridgeCompilationResult(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale verifying compilation and promotion readiness.")
    candidate_id: str = Field(..., description="Candidate build identifier")
    episode_number: int = Field(default=0, description="Target episode")
    promoted_path: str = Field(..., description="Path to promoted cartridge (cartridges/active)")
    manifest: Dict[str, Any] = Field(default_factory=dict, description="Promoted manifest.json")
    file_hashes: Dict[str, str] = Field(default_factory=dict, description="SHA256 of all files")
    envelope_hash: str = Field(..., description="SHA256 of sealed envelope.json")
    promoted_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CharacterSheet(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale defining character archetype and voice style.")
    character_id: str = Field(..., description="Slug identifier: 'hero', 'rival', 'boss', 'npc'.")
    name: str = Field(..., description="Display name.")
    role: str = Field(..., description="'protagonist' | 'rival' | 'boss' | 'npc' | 'companion'.")
    personality_traits: List[str] = Field(default_factory=list)
    backstory: str = Field(default="")
    voice_timbre: str = Field(default="focused")
    gemini_voice_name: str = Field(default="Aoede", description="'Aoede' | 'Puck' | 'Charon' | 'Fenrir' | 'Kore'.")
    visual_description: str = Field(default="Pixel sprite entity")
    portrait_asset: str = Field(default="assets/sprites/hero.png")


class CutsceneBeat(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale framing this cinematic beat.")
    beat_number: int = Field(...)
    action_description: str = Field(...)
    speaker_id: Optional[str] = Field(default=None)
    dialogue_line: Optional[str] = Field(default=None)


class CutsceneScript(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale for cutscene structure and emotional arc.")
    cutscene_id: str = Field(...)
    title: str = Field(...)
    trigger: str = Field(default="episode_start")
    beats: List[CutsceneBeat] = Field(default_factory=list)


class DialogueChoice(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale for dialogue branch option.")
    text: str = Field(...)
    target_node_id: str = Field(...)
    condition: Optional[str] = Field(default=None)


class DialogueNode(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale for dialogue node text and branches.")
    node_id: str = Field(...)
    speaker_id: str = Field(...)
    text: str = Field(...)
    choices: List[DialogueChoice] = Field(default_factory=list)


class DialogueTreeSpec(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale validating dialogue graph acyclic reachability.")
    root_node_id: str = Field(default="root")
    nodes: Dict[str, DialogueNode] = Field(default_factory=dict)

    @field_validator("nodes", mode="before")
    @classmethod
    def _normalize_dialogue_nodes(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        normalized: Dict[str, Any] = {}
        for k, node_val in v.items():
            if isinstance(node_val, dict):
                nd = dict(node_val)
                if "deliberation" not in nd:
                    nd["deliberation"] = f"Dialogue node {k} deliberation."
                if "node_id" not in nd:
                    nd["node_id"] = str(k)
                if "speaker_id" not in nd:
                    nd["speaker_id"] = "hero"
                if "text" not in nd:
                    nd["text"] = "..."
                if isinstance(nd.get("choices"), list):
                    norm_choices = []
                    for ch in nd["choices"]:
                        if isinstance(ch, dict):
                            cd = dict(ch)
                            if "deliberation" not in cd:
                                cd["deliberation"] = "Dialogue branch option."
                            if "text" in cd and "target_node_id" in cd:
                                norm_choices.append(cd)
                        else:
                            norm_choices.append(ch)
                    nd["choices"] = norm_choices
                normalized[str(k)] = nd
            elif isinstance(node_val, str):
                normalized[str(k)] = {
                    "deliberation": f"Dialogue node {k} deliberation.",
                    "node_id": str(k),
                    "speaker_id": "hero",
                    "text": node_val,
                    "choices": [],
                }
            else:
                normalized[str(k)] = node_val
        return normalized


class DialogueBarkSpec(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale justifying voice bark line and trigger conditions.")
    bark_id: str = Field(...)
    speaker_id: str = Field(...)
    speaker_name: str = Field(...)
    trigger_type: str = Field(
        ...,
        description="'episode_start' | 'low_health' | 'boss_encounter' | 'kill_streak' | 'exit_unlocked' | 'ambient'."
    )
    text: str = Field(...)
    voice_clip: str = Field(..., description="Relative path, e.g. 'assets/audio/hero_bark_start.wav'.")
    portrait: str = Field(default="assets/sprites/hero.png")
    priority: int = Field(default=1)
    duration: float = Field(default=4.5)


class NarrativeBible(BaseEpistemicModel):
    deliberation: str = Field(..., description="Rationale-first deliberation on narrative cohesion and character arcs.")
    bible_id: str = Field(default_factory=lambda: f"bible_{uuid.uuid4().hex[:8]}")
    world_title: str = Field(...)
    setting_summary: str = Field(...)
    characters: List[CharacterSheet] = Field(default_factory=list, description="Character sheets (can be empty for abstract/character-less games).")
    cutscenes: List[CutsceneScript] = Field(default_factory=list)
    dialogue_trees: List[DialogueTreeSpec] = Field(default_factory=list)
    dialogue_barks: List[DialogueBarkSpec] = Field(default_factory=list)
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("characters")
    @classmethod
    def validate_characters_present(cls, v: List[CharacterSheet]) -> List[CharacterSheet]:
        # Rule 1: Pure Generalizability — abstract, puzzle, simulation, or character-less games
        # with 0 characters (len(characters) == 0) are valid.
        return v if v is not None else []

    def to_cartridge_narrative_dict(self) -> Dict[str, Any]:
        """Converts Bible to narrative.json format consumed by PlayingScene."""
        chars_dict = {}
        for c in self.characters:
            chars_dict[c.character_id] = {
                "name": c.name,
                "role": c.role,
                "portrait": c.portrait_asset or "assets/sprites/default_portrait.png",
                "voice_timbre": c.voice_timbre,
                "gemini_voice": c.gemini_voice_name,
            }
        barks_list = []
        for b in self.dialogue_barks:
            barks_list.append({
                "deliberation": b.deliberation,
                "bark_id": b.bark_id,
                "speaker_id": b.speaker_id,
                "speaker_name": b.speaker_name,
                "trigger_type": b.trigger_type,
                "text": b.text,
                "voice_clip": b.voice_clip,
                "portrait": b.portrait or "assets/sprites/default_portrait.png",
                "priority": b.priority,
                "duration": b.duration,
            })
        return {
            "deliberation": self.deliberation,
            "world_title": self.world_title,
            "setting_summary": self.setting_summary,
            "characters": chars_dict,
            "dialogue_barks": barks_list,
        }


class AudioSynthesisResult(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale verifying synthesized voice quality and duration.")
    bark_id: str = Field(...)
    file_path: str = Field(...)
    sample_rate: int = Field(default=44100)
    duration_seconds: float = Field(...)
    file_size_bytes: int = Field(...)
    sha256_hash: str = Field(...)


class TelemetryProbeDesignSpec(BaseEpistemicModel):
    deliberation: str = Field(
        ...,
        description="Epistemic rationale explaining why this probe is necessary and what hypothesis it tests."
    )
    probe_id: str = Field(
        default_factory=lambda: f"probe_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for database registration."
    )
    probe_name: str = Field(
        ...,
        description="Unique alphanumeric snake_case identifier."
    )
    category: str = Field(
        ...,
        description="Canonical probe category ('combat', 'movement', 'dialogue', 'ability', 'risk_reward', 'exploration')."
    )
    description: str = Field(
        ...,
        description="Human and agent-readable description of what this probe records."
    )
    hypothesis_target: str = Field(
        ...,
        description="Falsifiable epistemic hypothesis this probe is designed to validate or refute."
    )
    payload_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema (Draft-07 compliant) defining the expected keys in data payload."
    )
    target_module: str = Field(
        default="mechanics.gd",
        description="Filename of the module being instrumented."
    )
    target_class: str = Field(
        ...,
        description="Target class name where probe should be declared."
    )
    target_method: str = Field(
        ...,
        description="Target method name where probe.emit() should be invoked."
    )
    injection_location: str = Field(
        default="method_exit",
        description="'method_entry' | 'method_exit' | 'damage_event' | 'state_transition'"
    )
    max_event_rate_hz: float = Field(
        default=60.0,
        description="Rate-limiting ceiling in Hz (maximum allowed 120.0 Hz)."
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class TelemetryArchitecturePlan(BaseEpistemicModel):
    deliberation: str = Field(
        ...,
        description="Epistemic deliberation analyzing cartridge architecture, player DNA, and telemetry strategy."
    )
    plan_id: str = Field(
        default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}",
        description="Unique plan identifier."
    )
    cartridge_id: str = Field(default="active", description="Target cartridge identifier.")
    probes: List[TelemetryProbeDesignSpec] = Field(default_factory=list)
    hypotheses_addressed: List[str] = Field(default_factory=list)
    estimated_total_throughput_hz: float = Field(default=0.0)
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


TelemetryPlan = TelemetryArchitecturePlan


class InstrumentationResult(BaseEpistemicModel):
    deliberation: str = Field(
        ...,
        description="Epistemic evaluation of instrumentation execution and AST security compliance."
    )
    success: bool = Field(..., description="True if 100% of probes were successfully injected and AST-verified.")
    instrumented_code: Dict[str, str] = Field(default_factory=dict, description="module_name -> instrumented_source")
    probes_registered: int = Field(default=0)
    ast_security_passed: bool = Field(default=False)
    validation_errors: List[str] = Field(default_factory=list)


class HypothesisEvidence(BaseEpistemicModel):
    deliberation: str = Field(..., description="Epistemic rationale explaining the observed empirical signal.")
    metric_name: str = Field(..., description="Name of the quantified metric or custom probe.")
    observed_value: float = Field(..., description="Observed empirical value from SQLite WAL.")
    baseline_value: Optional[float] = Field(default=None, description="Expected baseline or historic value.")
    sample_size: int = Field(default=1, description="Number of events or instances evaluated.")
    statistical_significance: float = Field(default=1.0, description="Confidence metric in [0.0, 1.0].")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class PlayerPreferenceHypothesis(BaseEpistemicModel):
    deliberation: str = Field(
        ...,
        description="Epistemic deliberation grounding this hypothesis in empirical telemetry observations."
    )
    hypothesis_id: str = Field(
        default_factory=lambda: f"hyp_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this hypothesis."
    )
    domain: str = Field(
        ...,
        description="'combat_style' | 'weapon_preference' | 'hazard_fatigue' | 'pacing_alignment' | 'risk_affinity'"
    )
    statement: str = Field(
        ...,
        description="Explicit, falsifiable hypothesis statement describing player preference or friction."
    )
    status: str = Field(
        default="CONFIRMED",
        description="'CONFIRMED' | 'REFUTED' | 'INCONCLUSIVE' | 'PENDING_EVALUATION'"
    )
    confidence_score: float = Field(
        ...,
        description="Empirical statistical confidence score in [0.0, 1.0]."
    )
    evidence: List[HypothesisEvidence] = Field(
        default_factory=list,
        description="Chain of empirical evidence supporting or refuting this hypothesis."
    )
    falsification_condition: str = Field(
        ...,
        description="Explicit observable telemetry condition that would falsify/refute this hypothesis."
    )
    prescribed_evolution: str = Field(
        ...,
        description="Actionable mechanical, narrative, or spatial evolution prescribed to address this hypothesis."
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class EvolutionHypothesisReport(BaseEpistemicModel):
    deliberation: str = Field(
        ...,
        description="Epistemic synthesis across all telemetry streams, custom probe payloads, and Steam DNA alignment."
    )
    report_id: str = Field(
        default_factory=lambda: f"report_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this report."
    )
    session_id: str = Field(..., description="Target session identifier.")
    episode_id: Optional[str] = Field(default=None, description="Analyzed episode identifier.")
    dominant_playstyle: str = Field(..., description="Classified playstyle archetype.")
    player_dna_alignment_score: float = Field(
        default=0.0,
        description="Cosine correlation [-1.0, 1.0] between live gameplay metrics and historic SteamPlayerDNA."
    )
    hypotheses: List[PlayerPreferenceHypothesis] = Field(default_factory=list)
    custom_probe_summaries: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    bottlenecks_detected: List[str] = Field(default_factory=list)
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class BoundedAdaptationSpec(BaseEpistemicModel):
    deliberation: str = Field(
        ...,
        description="Epistemic rationale justifying difficulty parameter adaptations within strict bounds."
    )
    spec_id: str = Field(
        default_factory=lambda: f"spec_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this adaptation spec."
    )
    current_episode_number: int = Field(...)
    target_episode_number: int = Field(...)
    clamp_state: Dict[str, Any] = Field(default_factory=dict)
    adapted_parameters: Dict[str, float] = Field(default_factory=dict)
    mechanical_directives: List[Dict[str, Any]] = Field(default_factory=list)
    spatial_directives: List[Dict[str, Any]] = Field(default_factory=list)
    updated_episode_config: Dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class StudioSwarmPlan(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Dynamic multi-agent execution DAG and genre-specific commissioning task manifest
    authored by live Gemini meta-planning based on player Steam DNA or telemetry hypotheses.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic deliberation analyzing Steam player DNA "
                    "or telemetry evolution hypotheses to dynamically assemble agent roles, "
                    "DAG causal dependencies, and genre-specific task directives.",
    )
    plan_id: str = Field(
        default_factory=lambda: f"swarm_plan_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for the dynamic swarm plan.",
    )
    game_title: str = Field(
        ...,
        description="Evocative title for the game conceived by meta-planning.",
    )
    genre_verdict: str = Field(
        ...,
        description="Specific genre verdict (e.g. 'tactical_deckbuilder', 'bullet_heaven', "
                    "'turn_based_tactics', 'action_roguelike', 'physics_puzzler', 'narrative_adventure').",
    )
    specialist_agent_roles: List[str] = Field(
        ...,
        description="Dynamically selected specialist agent roles required to build this specific game.",
    )
    dag_edges: List[List[str]] = Field(
        default_factory=list,
        description="Directed acyclic graph edges defining causal dependencies between specialist roles [upstream, downstream].",
    )
    task_manifest: Dict[str, Any] = Field(
        default_factory=dict,
        description="Genre-specific commissioning prompts and execution parameters for each specialist role.",
    )
    target_episode: int = Field(
        default=0,
        description="Target build episode (0 for Genesis build, >= 1 for Episodic Updates).",
    )
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC creation timestamp.",
    )

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("deliberation cannot be empty or whitespace-only")
        return v

    @field_validator("specialist_agent_roles")
    @classmethod
    def validate_roles_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("specialist_agent_roles must contain at least one specialist role")
        return v

    @property
    def edges_as_tuples(self) -> List[Tuple[str, str]]:
        return [(edge[0], edge[1]) for edge in self.dag_edges if len(edge) >= 2]

    def validate_dag(self) -> List[str]:
        """Validates that dag_edges form a valid acyclic graph referencing selected roles.
        Returns the sorted topological order if valid, or raises CircularDependencyError.
        """
        from engine.studio.orchestrator import CircularDependencyError, StudioAgentDAG
        dag = StudioAgentDAG(edges=[(edge[0], edge[1]) for edge in self.dag_edges if len(edge) >= 2])
        if dag.has_cycle():
            raise CircularDependencyError(f"Cycle detected in StudioSwarmPlan DAG: {self.dag_edges}")
        return dag.topological_sort()

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class AcceptanceCriterion(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Explicit, verifiable criterion for satisfying a requirement track.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Epistemic justification for this acceptance criterion.")
    id: str = Field(default="", description="Unique criterion identifier, e.g. 'AC-R1.1' or 'AC1.1'")
    criterion_id: str = Field(default="", description="Unique criterion identifier alias")
    description: str = Field(..., description="Explicit, verifiable description of expected behavior.")
    verification_method: str = Field(
        default="headless_playtest",
        description="'ast_inspection' | 'runtime_telemetry' | 'headless_playtest' | 'multi_modal' | 'design_critic' | 'asset_presence'"
    )
    verification_type: Optional[str] = Field(default=None, description="Verification method alias")
    passed: bool = Field(default=False, description="Verification status.")

    @model_validator(mode="before")
    @classmethod
    def _normalize_criterion_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            cid = out.get("id") or out.get("criterion_id") or f"AC_{uuid.uuid4().hex[:6]}"
            out["id"] = cid
            out["criterion_id"] = cid
            vm = out.get("verification_method") or out.get("verification_type") or "headless_playtest"
            out["verification_method"] = vm
            out["verification_type"] = vm
            return out
        return data


class AssetRequirement(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Dynamic multimodal visual or audio asset required by the conceived game.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Rationale explaining why this asset is required.")
    name: str = Field(default="", description="Asset filename, e.g. 'hero_ship.png'")
    asset_id: str = Field(default="", description="Identifier slug, e.g. 'sprite_resource_ore' or 'bark_start'")
    asset_type: str = Field(default="sprite", description="'sprite' | 'tile' | 'portrait' | 'audio_bark' | 'sfx'")
    target_path: str = Field(default="", description="Relative destination, e.g. 'assets/sprites/ore.png'")
    description: str = Field(default="", description="Asset description or visual prompt")
    visual_prompt_or_text: str = Field(default="", description="Prompt or spoken text used for generation.")
    dimensions: Optional[Tuple[int, int]] = Field(default=(32, 32), description="Image dimensions (width, height) if visual asset.")
    geometry_directive: Optional[str] = Field(default=None, description="Geometry directive for procedural rendering")
    status: str = Field(default="pending", description="'pending' | 'generated' | 'failed'")

    @model_validator(mode="before")
    @classmethod
    def _normalize_asset_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            atype = out.get("asset_type") or "sprite"
            out["asset_type"] = atype
            # Sync name, asset_id, target_path
            name = out.get("name")
            aid = out.get("asset_id")
            tpath = out.get("target_path")
            if not name and tpath:
                name = Path(tpath).name
            if not name and aid:
                ext = ".png" if atype in ("sprite", "tile", "portrait") else ".wav"
                name = f"{aid}{ext}" if not aid.endswith((".png", ".wav")) else aid
            if not aid and name:
                aid = name.replace(".png", "").replace(".wav", "")
            if not name and not aid:
                aid = f"asset_{uuid.uuid4().hex[:6]}"
                ext = ".png" if atype in ("sprite", "tile", "portrait") else ".wav"
                name = f"{aid}{ext}"
            if not tpath:
                subfolder = "sprites" if atype in ("sprite", "tile", "portrait") else "audio"
                tpath = f"assets/{subfolder}/{name}"
            out["name"] = name
            out["asset_id"] = aid
            out["target_path"] = tpath

            # Sync description and visual_prompt_or_text
            desc = out.get("description")
            vprompt = out.get("visual_prompt_or_text")
            final_text = desc or vprompt or f"Asset {aid}"
            out["description"] = final_text
            out["visual_prompt_or_text"] = final_text
            return out
        return data


class AssetManifestSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Specification of all dynamic multimodal assets required by the cartridge.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Epistemic rationale justifying required asset catalog.")
    manifest_id: str = Field(default_factory=lambda: f"manifest_{uuid.uuid4().hex[:8]}", description="Unique manifest ID")
    game_title: str = Field(default="", description="Associated game title")
    assets: List[AssetRequirement] = Field(default_factory=list, description="List of required assets")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class RequirementTrack(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Discrete, ordered implementation track decomposing game systems into verifiable steps.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Epistemic rationale detailing feature architecture and dependencies.")
    id: str = Field(default="", description="Track identifier: 'R1', 'R2', ...")
    track_id: str = Field(default="", description="Track identifier alias: 'R1', 'R2', ...")
    title: str = Field(..., description="Concise feature track title.")
    description: str = Field(..., description="Detailed architectural and behavioral requirements.")
    responsible_agents: List[str] = Field(
        default_factory=lambda: ["LeadGameplaySWEAgent"],
        description="Specialist agents responsible for implementing this requirement."
    )
    target_modules: List[str] = Field(default_factory=lambda: ["mechanics.gd"], description="Target code/asset modules.")
    acceptance_criteria: List[AcceptanceCriterion] = Field(default_factory=list, description="Explicit verifiable criteria.")
    status: str = Field(default="pending", description="'pending' | 'in_progress' | 'completed' | 'satisfied' | 'rejected'")

    @model_validator(mode="before")
    @classmethod
    def _normalize_track_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            tid = out.get("id") or out.get("track_id") or f"R{uuid.uuid4().hex[:4]}"
            out["id"] = tid
            out["track_id"] = tid
            if "responsible_agent" in out and "responsible_agents" not in out:
                agent = out["responsible_agent"]
                out["responsible_agents"] = [agent] if isinstance(agent, str) else agent
            if "target_modules" in out and isinstance(out["target_modules"], list):
                sanitized = []
                for m in out["target_modules"]:
                    if isinstance(m, str):
                        s = m.replace(".py", ".gd")
                        sanitized.append(s)
                    else:
                        sanitized.append(m)
                out["target_modules"] = sanitized
            return out
        return data


class GameRequirementsSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Agent-authored comprehensive game requirements and verifiable backlog derived from player DNA.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST: Deep deliberation on player Steam DNA / telemetry hypotheses translating into core game requirements."
    )
    spec_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}", description="Requirements specification ID")
    game_title: str = Field(..., description="Game title conceived by agents")
    genre_paradigm: str = Field(default="action_roguelike", description="Genre classification or blend")
    target_audience_summary: str = Field(default="", description="Summary of target player DNA alignment")
    high_concept_summary: Optional[str] = Field(default=None, description="High-concept summary of game mechanics and loops")
    target_episode: int = Field(default=0, description="Target episode number (0 = Genesis)")
    requirements: List[RequirementTrack] = Field(default_factory=list, description="Ordered implementation tracks R1, R2, ...")
    asset_manifest: List[AssetRequirement] = Field(default_factory=list, description="Dynamic multimodal asset requirements")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="before")
    @classmethod
    def _normalize_spec_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            if "genre_verdict" in out and "genre_paradigm" not in out:
                out["genre_paradigm"] = out["genre_verdict"]
            elif "genre_paradigm" in out and "genre_verdict" not in out:
                out["genre_verdict"] = out["genre_paradigm"]
            return out
        return data

    @property
    def genre_verdict(self) -> str:
        return self.genre_paradigm

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ModuleFileSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Represents an individual synthesized file in the cartridge directory.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining the file's architectural role, node hierarchy, and API contracts."
    )
    filename: str = Field(
        ...,
        description="Relative file path within cartridge (e.g. 'main.gd', 'mechanics.gd', 'entities.gd', 'scenes.gd', 'ui_hud.gd', 'project.godot', 'main.tscn')."
    )
    content: str = Field(
        ...,
        description="Complete, syntactically valid source code or configuration for this module."
    )


class TrackCodeSynthesisPayload(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Payload returned by LeadGameplaySWEAgent for full synthesis or requirement track implementation.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining the architectural decisions, Godot 4 node composition, and track requirement fulfillment."
    )
    files: List[ModuleFileSpec] = Field(
        default_factory=list,
        description="Collection of cartridge files to create or update."
    )
    manifest_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="manifest.json content with metadata, genre, version, entrypoint."
    )

    def assemble_cartridge_directory(
        self,
        target_dir: Path,
        swe_agent: Optional[Any] = None,
    ) -> Dict[str, Path]:
        """Assembles, verifies, and writes modular cartridge files to target_dir.
        Returns dictionary mapping filename -> Path.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        results: Dict[str, Path] = {}

        # 1. Ensure minimal main.tscn exists
        main_tscn = target_dir / "main.tscn"
        if not main_tscn.is_file():
            main_tscn.write_text(
                '[gd_scene load_steps=2 format=3]\n\n'
                '[ext_resource type="Script" path="res://main.gd" id="1_main"]\n\n'
                '[node name="Main" type="Node2D"]\n'
                'script = ExtResource("1_main")\n',
                encoding="utf-8"
            )
            results["main.tscn"] = main_tscn

        # 2. Ensure project.godot exists
        project_godot = target_dir / "project.godot"
        if not project_godot.is_file():
            project_godot.write_text(
                'config_version=5\n\n'
                '[application]\n'
                'config/name="Game Cartridge"\n'
                'run/main_scene="res://main.tscn"\n'
                'config/features=PackedStringArray("4.3", "Forward Plus")\n\n'
                '[display]\n'
                'window/size/viewport_width=1280\n'
                'window/size/viewport_height=720\n'
                'window/stretch/mode="canvas_items"\n'
                'window/stretch/aspect="keep"\n\n'
                '[rendering]\n'
                'textures/canvas_textures/default_texture_filter=0\n',
                encoding="utf-8"
            )
            results["project.godot"] = project_godot

        # 3. Write synthesized module files
        for fspec in self.files:
            file_path = target_dir / fspec.filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(fspec.content, encoding="utf-8")
            results[fspec.filename] = file_path

        # 4. Write manifest.json
        if self.manifest_data:
            manifest_path = target_dir / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest_data, indent=2), encoding="utf-8")
            results["manifest.json"] = manifest_path

        return results


class CodeRepairSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Targeted self-healing repair payload for a broken GDScript or project file.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale diagnosing the error and detailing the self-healing modifications."
    )
    filename: str = Field(
        ...,
        description="Name of the file being repaired (e.g. 'main.gd', 'mechanics.gd')."
    )
    repaired_content: str = Field(
        ...,
        description="Complete, syntactically valid repaired source code."
    )


def __getattr__(name: str) -> Any:
    if name == "OpenGenreCartridgeBundle":
        from engine.studio.swe import OpenGenreCartridgeBundle
        return OpenGenreCartridgeBundle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

