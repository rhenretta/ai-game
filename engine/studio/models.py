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
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
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
    BRAINSTORMING = "BRAINSTORMING"
    AWAITING_GREENLIGHT = "AWAITING_GREENLIGHT"
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
    pause_for_greenlight: bool = Field(
        default=False,
        description="When True, pause at AWAITING_GREENLIGHT after the Writers' Room locks the Game Bible.",
    )


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
                "personality_traits": list(c.personality_traits),
                "backstory": c.backstory,
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
        cutscenes_list = [cs.model_dump() for cs in self.cutscenes]
        trees_list = [dt.model_dump() for dt in self.dialogue_trees]
        return {
            "deliberation": self.deliberation,
            "world_title": self.world_title,
            "setting_summary": self.setting_summary,
            "characters": chars_dict,
            "cutscenes": cutscenes_list,
            "dialogue_trees": trees_list,
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


class AnimationStateSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Defines an agent-authored animation state or pose sequence for a sprite sheet / animated asset.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Rationale for why this animation state is needed for gameplay/visual feedback.")
    state_name: str = Field(..., description="Animation state identifier (e.g. 'idle', 'move', 'jump', 'interact', 'active', 'damaged').")
    motion_profile: Literal["idle", "locomotion", "aerial", "action"] = Field(
        default="idle",
        description="Explicit kinematic articulation profile: 'idle' | 'locomotion' | 'aerial' | 'action'."
    )
    frame_count: int = Field(default=4, ge=1, le=12, description="Number of frames in this animation state.")
    fps: float = Field(default=8.0, description="Playback rate in frames per second.")
    loop: bool = Field(default=True, description="Whether this animation loops continuously.")
    pose_progression_prompt: str = Field(
        default="",
        description="Detailed visual description of how the silhouette, limbs, or mechanism progress across frames."
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_animation_state(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            if not out.get("pose_progression_prompt") and out.get("motion_description"):
                out["pose_progression_prompt"] = out["motion_description"]
            return out
        return data


class ArticulatedPartSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Defines an independently movable/rotatable component of a multi-part 2D rig.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Rationale for separating this part for procedural articulation in Godot.")
    part_id: str = Field(..., description="Unique part identifier within the rig (e.g. 'body', 'arm_tool', 'wing_left', 'rotor_blade').")
    visual_prompt: str = Field(default="", description="Visual prompt for this isolated component grounded in the concept art.")
    parent_part_id: Optional[str] = Field(default=None, description="Parent part_id in the rig hierarchy, or None for the root part.")
    pivot_normalized: Tuple[float, float] = Field(
        default=(0.5, 0.5),
        description="Normalized rotation pivot (u, v) in [0.0, 1.0] within this part's texture."
    )
    attach_offset_px: Tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="(x, y) pixel offset where this part attaches to its parent part."
    )
    z_index_offset: int = Field(default=0, description="Relative z-order offset compared to the root part.")
    dimensions: Tuple[int, int] = Field(default=(48, 48), description="Target world-space pixel dimensions (width, height) for this part.")

    @model_validator(mode="before")
    @classmethod
    def _normalize_articulated_part(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            if not out.get("visual_prompt") and out.get("part_description"):
                out["visual_prompt"] = out["part_description"]
            return out
        return data


class VisualSceneLayerSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Decomposes a 16:9 Writers' Room concept mockup into constructible depth layers for Godot 4 scenes.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Artistic and spatial rationale for this depth layer.")
    layer_id: str = Field(..., description="Identifier such as 'sky_backdrop', 'parallax_midground', 'primary_terrain', 'interactive_props', 'lighting_fx'.")
    depth_order: int = Field(default=0, description="Depth ordering from 0 (furthest background) to foreground.")
    scroll_parallax_factor: Tuple[float, float] = Field(default=(1.0, 1.0), description="Relative camera scroll parallax multiplier (x, y).")
    visual_description: str = Field(..., description="Detailed visual composition, geometry, textures, and colors of this layer.")
    associated_asset_ids: List[str] = Field(default_factory=list, description="Asset IDs from the Asset Library used on this layer.")


class AssetRequirement(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Dynamic multimodal visual or audio asset required by the conceived game and fabricated into the Cartridge Asset Library.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Rationale explaining why this purpose-built asset is required.")
    name: str = Field(default="", description="Asset filename, e.g. 'hero_sheet.png' or 'vesper_portrait.png'")
    asset_id: str = Field(default="", description="Identifier slug, e.g. 'hero', 'conduit_pylon', 'vesper_portrait', 'bark_intro'")
    asset_type: str = Field(
        default="sprite",
        description="'sprite' | 'sprite_sheet' | 'articulated_rig' | 'prop' | 'tile' | 'backdrop' | 'portrait' | 'diegetic_ui' | 'audio_bark' | 'sfx' | 'ambient_loop'"
    )
    fabrication_mode: str = Field(
        default="static_sprite",
        description="'static_sprite' | 'sprite_sheet' | 'articulated_parts' | 'tile_surface' | 'parallax_layer' | 'portrait' | 'diegetic_ui' | 'audio_bark' | 'sfx' | 'ambient_loop'"
    )
    purpose: str = Field(default="", description="Exact functional purpose in gameplay, scene staging, or UI.")
    concept_reference_id: Optional[str] = Field(
        default=None,
        description="Optional character_id or mockup_id from GameBibleSpec to condition visual generation."
    )
    target_path: str = Field(default="", description="Relative destination, e.g. 'assets/sprites/hero.png' or 'assets/portraits/vesper_portrait.png'")
    description: str = Field(default="", description="Asset description or visual prompt")
    visual_prompt_or_text: str = Field(default="", description="Prompt or spoken text used for generation.")
    dimensions: Optional[Tuple[int, int]] = Field(
        default=(64, 64),
        description="Target world-space display dimensions (width, height) in pixels if visual asset."
    )
    animation_states: List[AnimationStateSpec] = Field(
        default_factory=list,
        description="Optional multi-pose animation states when fabricating an animated sprite sheet."
    )
    articulated_parts: List[ArticulatedPartSpec] = Field(
        default_factory=list,
        description="Optional multi-part rig breakdown when fabricating articulated components."
    )
    geometry_directive: Optional[str] = Field(default=None, description="Geometry directive for procedural rendering")
    status: str = Field(default="pending", description="'pending' | 'generated' | 'failed'")

    @model_validator(mode="before")
    @classmethod
    def _normalize_asset_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            atype = str(out.get("asset_type") or out.get("category") or "sprite").lower().strip()
            fmode = str(out.get("fabrication_mode") or "").lower().strip()

            # Infer fabrication_mode and asset_type consistently
            if not fmode:
                if out.get("animation_states") or atype in ("sprite_sheet", "animated_sprite", "animation"):
                    fmode = "sprite_sheet"
                elif out.get("articulated_parts") or atype in ("articulated_rig", "articulated_parts", "rig"):
                    fmode = "articulated_parts"
                elif atype == "portrait":
                    fmode = "portrait"
                elif atype in ("tile", "tile_surface", "terrain"):
                    fmode = "tile_surface"
                elif atype in ("backdrop", "parallax_layer", "background"):
                    fmode = "parallax_layer"
                elif atype in ("diegetic_ui", "ui"):
                    fmode = "diegetic_ui"
                elif atype in ("audio_bark", "voice_bark", "voice", "bark"):
                    fmode = "audio_bark"
                    atype = "audio_bark"
                elif atype in ("sfx", "sound_effect", "audio"):
                    fmode = "sfx"
                    atype = "sfx"
                elif atype in ("ambient_loop", "ambience", "music"):
                    fmode = "ambient_loop"
                    atype = "ambient_loop"
                else:
                    fmode = "static_sprite"

            if fmode == "portrait":
                atype = "portrait"
            elif fmode == "sprite_sheet" and atype == "sprite":
                atype = "sprite_sheet"
            elif fmode == "articulated_parts" and atype == "sprite":
                atype = "articulated_rig"

            out["asset_type"] = atype
            out["fabrication_mode"] = fmode

            is_audio = atype in ("audio_bark", "sfx", "ambient_loop") or fmode in ("audio_bark", "sfx", "ambient_loop")
            is_portrait = atype == "portrait" or fmode == "portrait"

            name = out.get("name")
            aid = out.get("asset_id")
            tpath = out.get("target_path")

            if not name and tpath:
                name = Path(str(tpath)).name
            if not name and aid:
                ext = ".wav" if is_audio else ".png"
                name = f"{aid}{ext}" if not str(aid).endswith((".png", ".wav")) else str(aid)
            if not aid and name:
                aid = str(name).replace(".png", "").replace(".wav", "")
            if not name and not aid:
                aid = f"asset_{uuid.uuid4().hex[:6]}"
                ext = ".wav" if is_audio else ".png"
                name = f"{aid}{ext}"

            # Enforce strict directory separation: portraits NEVER go into assets/sprites/
            if is_portrait:
                fname = Path(str(name)).name
                if not fname.endswith(".png"):
                    fname = f"{fname}.png"
                if not tpath or str(tpath).startswith("assets/sprites/"):
                    tpath = f"assets/portraits/{fname}"
            elif is_audio:
                if not tpath:
                    tpath = f"assets/audio/{name}"
            else:
                if not tpath or str(tpath).startswith("assets/portraits/"):
                    tpath = f"assets/sprites/{Path(str(name)).name}"

            out["name"] = name
            out["asset_id"] = aid
            out["target_path"] = tpath

            if "dimensions" not in out or not out["dimensions"]:
                if is_portrait:
                    out["dimensions"] = (256, 256)
                elif fmode == "parallax_layer":
                    out["dimensions"] = (640, 360)
                elif fmode == "tile_surface":
                    out["dimensions"] = (64, 64)
                else:
                    out["dimensions"] = (64, 64)

            desc = out.get("description")
            vprompt = out.get("visual_prompt_or_text")
            final_text = desc or vprompt or out.get("purpose") or f"Asset {aid}"
            out["description"] = final_text
            out["visual_prompt_or_text"] = vprompt or final_text
            return out
        return data


class AssetLibraryEntry(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Indexed record in the Cartridge Asset Library (assets/asset_library.json) describing a fabricated asset,
    its animation frames/states, or its articulated rig parts for GDScript consumption.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Epistemic summary of how this asset was fabricated and grounded.")
    asset_id: str = Field(..., description="Unique asset slug in the library.")
    asset_type: str = Field(default="sprite")
    fabrication_mode: str = Field(default="static_sprite")
    purpose: str = Field(default="")
    res_path: str = Field(..., description="Godot resource path, e.g. 'res://assets/sprites/hero.png'.")
    file_path: str = Field(..., description="Relative file path in cartridge, e.g. 'assets/sprites/hero.png'.")
    target_display_size: Tuple[int, int] = Field(default=(64, 64), description="Intended in-world display size (width, height) in pixels.")
    texture_pixel_size: Tuple[int, int] = Field(default=(256, 256), description="Actual PNG dimensions (width, height) on disk.")
    hframes: int = Field(default=1, ge=1, description="Number of horizontal frames if sprite sheet.")
    vframes: int = Field(default=1, ge=1, description="Number of vertical rows if sprite sheet.")
    animation_states: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Map of state_name -> {row, start_frame, frame_count, fps, loop, frame_res_paths}."
    )
    articulated_parts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of articulated rig parts with {part_id, res_path, parent_part_id, pivot_normalized, attach_offset_px, z_index_offset, dimensions}."
    )
    pivot_normalized: Tuple[float, float] = Field(default=(0.5, 0.5))
    concept_reference_used: Optional[str] = Field(default=None)
    sha256_hash: str = Field(default="")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AssetLibraryCatalog(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Persistent catalog written to assets/asset_library.json indexing all purpose-built cartridge assets.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Holistic rationale for the cartridge's multimodal asset library.")
    catalog_id: str = Field(default_factory=lambda: f"lib_{uuid.uuid4().hex[:8]}")
    game_title: str = Field(default="")
    camera_and_spatial_paradigm: str = Field(default="")
    color_palette_hex: List[str] = Field(default_factory=list)
    entries: Dict[str, AssetLibraryEntry] = Field(default_factory=dict)
    updated_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


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
    requested_assets: List[AssetRequirement] = Field(
        default_factory=list,
        description="On-demand purpose-built assets (sprite sheets, articulated parts, props, tiles, parallax layers, UI, SFX) requested by the SWE agent for the Cartridge Asset Library."
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

        # 0. Fabricate any on-demand requested_assets into the CartridgeAssetLibrary first
        try:
            from engine.studio.asset_library import CartridgeAssetLibrary
            visual_agent = getattr(swe_agent, "visual_agent", None) if swe_agent is not None else None
            voice_agent = getattr(swe_agent, "voice_agent", None) if swe_agent is not None else None
            lib = CartridgeAssetLibrary(
                cartridge_dir=target_dir,
                visual_agent=visual_agent,
                voice_agent=voice_agent,
            )
            if self.requested_assets:
                fabricated = lib.fabricate_assets(self.requested_assets)
                results.update(fabricated)
            lib.save_catalog()
        except Exception:
            pass

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
                'window/size/window_width_override=2560\n'
                'window/size/window_height_override=1440\n'
                'window/size/resizable=true\n'
                'window/dpi/allow_hidpi=true\n'
                'window/stretch/mode="canvas_items"\n'
                'window/stretch/aspect="keep"\n\n'
                '[rendering]\n'
                'textures/canvas_textures/default_texture_filter=2\n',
                encoding="utf-8"
            )
            results["project.godot"] = project_godot

        # 3. Write synthesized module files
        for fspec in self.files:
            file_path = target_dir / fspec.filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(fspec.content, encoding="utf-8")
            results[fspec.filename] = file_path

        # 3b. Enforce HiDPI Retina window sizing & resizable=true even if LLM overwrote project.godot
        if project_godot.is_file():
            enforce_hidpi_project_godot(project_godot)
            results["project.godot"] = project_godot

        # 4. Write manifest.json
        if self.manifest_data:
            manifest_path = target_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest_data_clean := self.manifest_data, indent=2), encoding="utf-8")
            results["manifest.json"] = manifest_path

        # 5. Run Godot 4 compiler syntax verification & self-healing repair loop when live SWE agent is active
        if (
            swe_agent is not None
            and hasattr(swe_agent, "_verify_and_heal_godot_syntax_and_probes")
            and hasattr(swe_agent, "_is_live_genai_client")
            and swe_agent._is_live_genai_client()
            and any(f.filename.endswith(".gd") for f in self.files)
        ):
            healed = swe_agent._verify_and_heal_godot_syntax_and_probes(target_dir)
            results.update(healed)

        return results


def enforce_hidpi_project_godot(project_godot_path: Path) -> None:
    """Ensures project.godot configures a crisp, resizable 2560x1440 HiDPI window on Retina displays."""
    if not project_godot_path.is_file():
        return
    raw = project_godot_path.read_text(encoding="utf-8")
    raw = raw.replace("window/size/resizable=false", "window/size/resizable=true")
    raw = raw.replace("textures/canvas_textures/default_texture_filter=0", "textures/canvas_textures/default_texture_filter=2")

    display_defaults = {
        "window/size/viewport_width": "1280",
        "window/size/viewport_height": "720",
        "window/size/window_width_override": "2560",
        "window/size/window_height_override": "1440",
        "window/size/resizable": "true",
        "window/dpi/allow_hidpi": "true",
        "window/stretch/mode": '"canvas_items"',
        "window/stretch/aspect": '"keep"',
    }
    if "[display]" not in raw:
        raw = raw.rstrip() + "\n\n[display]\n" + "\n".join(f"{k}={v}" for k, v in display_defaults.items()) + "\n"
    else:
        missing_lines = [f"{k}={v}" for k, v in display_defaults.items() if f"{k}=" not in raw]
        if missing_lines:
            raw = raw.replace("[display]", "[display]\n" + "\n".join(missing_lines), 1)

    if "[rendering]" not in raw:
        raw = raw.rstrip() + "\n\n[rendering]\ntextures/canvas_textures/default_texture_filter=2\n"
    elif "textures/canvas_textures/default_texture_filter=" not in raw:
        raw = raw.replace("[rendering]", "[rendering]\ntextures/canvas_textures/default_texture_filter=2", 1)

    project_godot_path.write_text(raw, encoding="utf-8")



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


class CodeSliceRequest(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Scoped code inspection request issued by the SWE agent during the PEV Plan/Research phase.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining why this specific line range or symbol is needed to diagnose or fix the defect."
    )
    filename: str = Field(
        ...,
        description="Target file in cartridge (e.g. 'scenes.gd', 'entities.gd', 'main.gd', 'ui_hud.gd', 'mechanics.gd')."
    )
    start_line: int = Field(
        default=1,
        ge=1,
        description="1-indexed starting line number to inspect."
    )
    end_line: int = Field(
        default=80,
        ge=1,
        description="1-indexed inclusive ending line number to inspect (keep slices focused, e.g. 40-120 lines)."
    )
    symbol_query: Optional[str] = Field(
        default=None,
        description="Optional class, function, or identifier name to locate and extract with surrounding context."
    )


class DebugResearchPlan(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    PEV Phase 1 (Plan & Research): Formulates root-cause hypothesis and requests only the relevant code slices needed for context.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic reasoning analyzing the defect/diagnostic against the symbol outline and formulating a focused research plan."
    )
    root_cause_hypothesis: str = Field(
        ...,
        description="Specific technical hypothesis for what is causing the compiler, runtime, visual, or gameplay defect."
    )
    slice_requests: List[CodeSliceRequest] = Field(
        ...,
        description="Focused list of code slices or symbols to inspect (do NOT request entire large files; request only the relevant functions/classes)."
    )


class SurgicalCodePatch(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A targeted line-range replacement within a cartridge file.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining why replacing lines [start_line..end_line] in filename fixes the root cause while preserving surrounding indentation and logic."
    )
    filename: str = Field(
        ...,
        description="Target file in cartridge (e.g. 'scenes.gd', 'main.gd')."
    )
    start_line: int = Field(
        ...,
        ge=1,
        description="1-indexed first line of the contiguous block in `filename` to replace."
    )
    end_line: int = Field(
        ...,
        ge=1,
        description="1-indexed last line (inclusive) of the contiguous block in `filename` to replace."
    )
    replacement_code: str = Field(
        ...,
        description="Exact replacement GDScript lines (without line number prefixes, using tab '\\t' indentation matching the surrounding block) to substitute for lines [start_line..end_line]."
    )


class SurgicalPatchPayload(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    PEV Phase 2 (Execute): Produces surgical line-range patches from the researched code slices, or requests additional slices if needed.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale synthesizing findings from the inspected code slices and justifying the surgical patches."
    )
    additional_slice_requests: List[CodeSliceRequest] = Field(
        default_factory=list,
        description="If and only if the inspected slices revealed another caller/callee that must be read before patching, request those additional slices here."
    )
    patches: List[SurgicalCodePatch] = Field(
        default_factory=list,
        description="Surgical line-range replacements to apply to the cartridge files."
    )
    verification_expectation: str = Field(
        default="",
        description="What the Verify phase (compiler + runtime harness check) should confirm after applying these patches."
    )


class SceneLayerStrategy(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Defines how a specific depth layer of the 2D environment is constructed in Godot 4.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining how this layer contributes to a cohesive, organic environment matching the Game Bible and concept art."
    )
    layer_name: str = Field(
        ...,
        description="Descriptive name of the scene layer (e.g. 'Atmospheric Sky & Floating Horizon', 'Sculpted Floating Terraces & Cliff Undersides', 'Continuous Hydro-Mana River & Geyser Plume', 'Interactive Flora & Mechanical Landmarks')."
    )
    z_index: int = Field(
        default=0,
        description="Godot 4 z_index ordering for this layer."
    )
    construction_technique: str = Field(
        ...,
        description="Concrete Godot 4 construction technique (e.g. multi-vertex Polygon2D silhouettes with tapered rocky undersides, lush overhanging moss rim polygons, continuous curved Line2D/Polygon2D fluid streams with caustic highlights, animated Sprite2D flora/mechanisms anchored to terrain surfaces—never raw axis-aligned draw_rect blocks or stick-figure lines)."
    )
    required_asset_ids: List[str] = Field(
        default_factory=list,
        description="IDs of sprite sheets, tiles, or prop textures required by this layer."
    )


class AudioCueStrategy(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Defines a required audio asset or procedural acoustic cue and its gameplay trigger.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale grounding this audio cue in the Game Bible's audio direction."
    )
    cue_id: str = Field(..., description="Unique identifier for the voice bark or sound effect cue.")
    trigger_condition: str = Field(..., description="Exact gameplay event or state transition that triggers this sound.")
    synthesis_or_bark_spec: str = Field(..., description="How this sound is loaded from disk or synthesized at 24,000 Hz.")


class GameProductionStrategySpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Master multi-disciplinary strategy formulated by the agents before and during scene construction,
    mapping the Game Bible and 16:9 Concept Mockups into concrete assets, environment topology, audio, and code architecture.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Holistic strategic reasoning decomposing the Game Bible and concept mockups into real multi-layered environments, required assets, audio, and code."
    )
    environment_topology_strategy: str = Field(
        ...,
        description="Detailed strategy for constructing organic, multi-layered level geometry (sculpted island silhouettes, cliff faces, lush top rims, tapered floating crags, continuous fluid waterways, and surface-anchored landmarks) rather than flat rectangular blocks or isolated sprites."
    )
    camera_and_perspective_coherence_strategy: str = Field(
        default="",
        description="Strategy ensuring 100% coherence between the gameplay movement plane (e.g. 3/4 top-down 2D plateau where the player moves in X and Y across the shelf Rect2 vs. 2D side-scroller) and terrain rendering (filling the entire walkable Rect2 with top-down ground surface terrain_surface.png, hanging vertical cliff walls structure_wall_tile.png ONLY below the southern bottom edge s.end.y, and distributing flora/landmarks across the 2D plateau floor)."
    )
    scene_layers: List[SceneLayerStrategy] = Field(
        default_factory=list,
        description="Ordered depth layers composing the playable scene."
    )
    required_visual_assets: List[AssetRequirement] = Field(
        default_factory=list,
        description="Purpose-built visual assets (sprite sheets, flora, props, terrain textures, ability icons) required to build the environment without programmer-art primitives."
    )
    audio_cues: List[AudioCueStrategy] = Field(
        default_factory=list,
        description="Voice barks and acoustic SFX cues required by the Game Bible."
    )
    psychoacoustic_audio_strategy: str = Field(
        default="",
        description="Strategy ensuring pleasant, event-driven audio synthesis with zero continuous looping high-pitched sine/square wave drones (no LOOP_FORWARD on >220Hz tones), zero piercing >800Hz squeals, and smooth exponential decay envelopes."
    )
    code_architecture_strategy: str = Field(
        ...,
        description="Cross-module code strategy across mechanics.gd, scenes.gd, entities.gd, ui_hud.gd, and main.gd, including surface coordinate anchoring so every entity/landmark sits naturally on the terrain."
    )
    diegetic_ui_strategy: str = Field(
        ...,
        description="Strategy for realizing the Game Bible's diegetic UI philosophy and clean fantasy HUD (vitals, objective progress, ability icons, and bottom dialogue bar without raw engineering debug text)."
    )


class SceneDeficiencyItem(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A specific visual, environmental, spatial, asset, or UX deficiency identified when analyzing a rendered scene.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic reasoning explaining why this visual, perspective, audio, or structural element in the rendered scene falls short of the Game Bible and Concept Mockup."
    )
    target_file: str = Field(
        ...,
        description="The cartridge `.gd` module responsible for this deficiency (e.g. 'scenes.gd', 'ui_hud.gd', 'entities.gd', 'main.gd')."
    )
    target_symbol_or_lines: str = Field(
        ...,
        description="The specific class, function name (e.g. 'SkyBackdrop._draw', 'TerraceRenderer._draw', 'FernsCluster._init', 'acoustics_hum_setup'), or line range where the deficiency originates."
    )
    deficiency_description: str = Field(
        ...,
        description="Concrete description of the deficiency visible in the rendered Godot viewport or audio code (e.g. 'Character moves in 2D top-down space across y=260..700 but TerraceRenderer draws y=260..700 as a vertical side-view cliff wall so she appears to walk on the side of a cliff', 'main.gd plays a constant high-pitched 523Hz looping sine wave drone')."
    )
    remediation_strategy: str = Field(
        ...,
        description="Concrete surgical code strategy to replace the deficient rendering/audio logic with perspective-coherent 3/4 plateau rendering or pleasant event-driven audio matching the Game Bible."
    )


class SceneCritiqueAndStrategySpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Per-scene multimodal analysis and re-strategizing contract comparing the live Godot 4 viewport render
    against the Game Bible, Concept Mockups, and Production Strategy Blueprint.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Comprehensive side-by-side analysis of the rendered Godot 4 viewport and audio synthesis against the Game Bible and 16:9 Concept Mockups."
    )
    environment_fidelity_critique: str = Field(
        ...,
        description="Detailed evaluation of the rendered environment topology, island silhouettes, terrain layering, fluid/canal rendering, background vistas, and prop placement."
    )
    perspective_and_camera_coherence_critique: str = Field(
        default="",
        description="Detailed evaluation of whether the camera perspective and character movement plane match the landscape orientation (e.g., verifying that a top-down/3/4 character walks on a top-down lush plateau surface rather than on the side of a vertical cliff wall)."
    )
    audio_and_psychoacoustic_critique: str = Field(
        default="",
        description="Detailed evaluation of the procedural audio synthesis and playback in main.gd (verifying there are no constant high-pitched looping sine-wave drones, piercing >800Hz tones, or harsh repetitive buzzes)."
    )
    ui_and_experience_critique: str = Field(
        ...,
        description="Detailed evaluation of the HUD against the Game Bible's diegetic UI philosophy and the concept art HUD."
    )
    deficiencies: List[SceneDeficiencyItem] = Field(
        default_factory=list,
        description="Ordered list of concrete deficiencies that must be surgically repaired in the `.gd` modules."
    )
    new_assets_needed: List[AssetRequirement] = Field(
        default_factory=list,
        description="Any missing visual or audio assets that must be fabricated so the scene does not rely on geometric programmer art."
    )
    scene_matches_game_bible: bool = Field(
        ...,
        description="Decision verdict (strictly last per Rule 4): True ONLY if the rendered scene genuinely matches the Game Bible and Concept Mockups with coherent camera/landscape perspective (character walking on top of a plateau, never on a cliff side), pleasant non-droney audio, zero blocky slab islands, zero stick-figure/triangle programmer art, and a clean fantasy HUD."
    )


class WritersRoomListenCheck(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Determines whether a specialist agent listening to the Writers' Room has a high-value
    critique, idea, or artifact to inject at the current moment in the conversation.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic reasoning evaluating the recent conversation from this specialist's perspective."
    )
    should_interject: bool = Field(
        ...,
        description="True if this specialist has an urgent critique, synergy, or creative artifact to contribute right now."
    )
    urgency_score: float = Field(
        default=0.0,
        description="Urgency/relevance score in [0.0, 1.0] for speaking next."
    )
    intended_contribution_summary: str = Field(
        default="",
        description="Concise 1-sentence summary of what the agent wants to say or propose."
    )


class ConceptEvolutionEntry(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Records a key milestone in how the game idea evolved during the Writers' Room.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining why the concept evolved or pivoted at this milestone."
    )
    entry_id: str = Field(default_factory=lambda: f"evo_{uuid.uuid4().hex[:8]}")
    turn_index: int = Field(default=0)
    author_role: str = Field(..., description="Specialist role that triggered this evolution.")
    previous_premise: str = Field(default="", description="What the concept/feature was before this shift.")
    evolved_premise: str = Field(..., description="How the concept/feature is now defined.")
    pivot_reason: str = Field(..., description="Why the previous idea was rejected or deepened.")
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExperienceStoryboardBeat(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Defines a beat in the player's emotional & sensory experience arc.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale grounding this emotional/experiential beat."
    )
    beat_index: int = Field(...)
    phase_title: str = Field(
        ...,
        description="E.g., 'First 30 Seconds: Sensory Cold Open', 'Diegetic Discovery', 'Mid-Episode Escalation', 'Climactic Resolution'."
    )
    player_emotional_state: str = Field(
        ...,
        description="What the player should feel at this moment (curiosity, dread, mastery, awe)."
    )
    sensory_and_camera_staging: str = Field(
        ...,
        description="What the player sees and hears on screen—lighting, camera framing, audio atmosphere, environmental storytelling."
    )
    diegetic_teaching_moment: str = Field(
        ...,
        description="How the player intuitively discovers or masters an interaction through the world itself without text-manual HUD overlays."
    )
    narrative_integration: str = Field(
        ...,
        description="How story, characters, or world lore manifest directly in this gameplay moment."
    )


class VisualMockupArtifact(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Represents a synthesized 16:9 concept art / gameplay UI mockup board in the Game Bible.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Artistic rationale for composition, palette, lighting, and diegetic UI framing."
    )
    mockup_id: str = Field(default_factory=lambda: f"mockup_{uuid.uuid4().hex[:8]}")
    title: str = Field(..., description="Title of the visual mockup board (e.g., 'Core Gameplay Loop & Diegetic UI', 'Opening Scene Staging').")
    scene_description: str = Field(..., description="Detailed visual description and prompt used to synthesize the mockup.")
    image_path: str = Field(..., description="Relative cartridge path, e.g. 'assets/mockups/gameplay_mockup_1.png'.")
    aspect_ratio: str = Field(default="16:9")
    camera_and_spatial_paradigm: str = Field(
        default="",
        description="Explicit camera angle & spatial movement architecture depicted in this mockup (e.g., '2D side-scrolling platformer with vertical elevation and gravity', '3/4 top-down exploration', 'isometric tactical arena')."
    )
    color_palette_hex: List[str] = Field(
        default_factory=list,
        description="6-10 hex color codes (#RRGGBB) defining the sky/backdrop, primary terrain, secondary architecture, accent glow, and UI palette."
    )
    scene_layers: List[VisualSceneLayerSpec] = Field(
        default_factory=list,
        description="Ordered breakdown of visual depth layers depicted in this mockup."
    )
    ui_callout_notes: List[str] = Field(
        default_factory=list,
        description="Specific visual callouts for SWE & Challenger (e.g., 'Health is shown via chassis sparks/glow rather than a text label')."
    )
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WritersRoomUtterance(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A single conversational turn in the open-ended Writers' Room transcript.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Internal epistemic reasoning before speaking to the room."
    )
    utterance_id: str = Field(default_factory=lambda: f"utt_{uuid.uuid4().hex[:8]}")
    turn_index: int = Field(...)
    speaker_role: str = Field(
        ...,
        description="'CreativeDirector' | 'GameplayDesigner' | 'StoryWorldArchitect' | 'CharacterDesigner' | 'UXExperienceDesigner' | 'VisualArtDirector' | 'AudioAtmosphereDesigner' | 'UserSteering'"
    )
    speaker_display_name: str = Field(..., description="Human-readable specialist title.")
    spoken_dialogue: str = Field(
        ...,
        description="Natural, opinionated conversational contribution addressing the room, critiquing weak ideas, or pitching improvements."
    )
    addressed_to: List[str] = Field(
        default_factory=list,
        description="Other specialists specifically called upon or challenged in this turn."
    )
    bible_updates_summary: List[str] = Field(
        default_factory=list,
        description="Summary of concrete additions/revisions made to the Living Game Bible during this turn."
    )
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WritersRoomTurnProposal(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Structured output produced when the selected specialist speaks and updates the Living Game Bible.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic deliberation on how to push the game experience forward and what to update in the Game Bible."
    )
    spoken_dialogue: str = Field(
        ...,
        description="Conversational statement to the Writers' Room."
    )
    addressed_to: List[str] = Field(default_factory=list)
    updated_game_title: Optional[str] = Field(default=None)
    updated_elevator_pitch: Optional[str] = Field(default=None)
    updated_core_fantasy: Optional[str] = Field(default=None)
    updated_first_60_seconds_hook: Optional[str] = Field(default=None)
    updated_diegetic_ui_philosophy: Optional[str] = Field(default=None)
    updated_camera_and_spatial_paradigm: Optional[str] = Field(default=None)
    updated_color_palette_hex: List[str] = Field(default_factory=list)
    added_or_updated_scene_layers: List[VisualSceneLayerSpec] = Field(default_factory=list)
    concept_evolution_milestone: Optional[ConceptEvolutionEntry] = Field(default=None)
    added_or_updated_storyboard_beats: List[ExperienceStoryboardBeat] = Field(default_factory=list)
    added_or_updated_characters: List[CharacterSheet] = Field(default_factory=list)
    added_or_updated_cutscenes: List[CutsceneScript] = Field(default_factory=list)
    added_or_updated_dialogue_trees: List[DialogueTreeSpec] = Field(default_factory=list)
    added_or_updated_barks: List[DialogueBarkSpec] = Field(default_factory=list)
    mockup_to_generate: Optional[VisualMockupArtifact] = Field(default=None)
    gameplay_pillars_additions: List[str] = Field(default_factory=list)
    visual_style_guide_update: Optional[str] = Field(default=None)
    audio_direction_update: Optional[str] = Field(default=None)


class DirectorConvergenceCheck(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Evaluated by the CreativeDirector to decide whether the Writers' Room should keep brainstorming
    or if the Living Game Bible is cohesive, rich, and ready to lock.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Director's critical evaluation of whether the experience, story, UX, art mockups, and characters feel like a cohesive, memorable game."
    )
    is_ready_to_lock_bible: bool = Field(
        ...,
        description="True ONLY when the Game Bible has a compelling hook, rich experience storyboard, diegetic UX plan, characters/world lore, and visual mockups."
    )
    experience_cohesion_score: float = Field(
        ...,
        description="Cohesion and polish readiness score in [0.0, 1.0]."
    )
    missing_or_weak_elements: List[str] = Field(
        default_factory=list,
        description="Specific gaps the room still needs to resolve if not ready."
    )
    director_steering_prompt: str = Field(
        default="",
        description="If not ready, the Director's next challenge or question to the room."
    )


class GameBibleSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    The Living Game Bible collaboratively authored by the Writers' Room and persisted to game_bible.json.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Holistic epistemic synthesis of the game's creative vision, player experience, and aesthetic identity."
    )
    bible_id: str = Field(default_factory=lambda: f"game_bible_{uuid.uuid4().hex[:8]}")
    cartridge_id: str = Field(default="active")
    target_episode: int = Field(default=0)
    status: str = Field(
        default="brainstorming",
        description="'brainstorming' | 'awaiting_greenlight' | 'locked'"
    )
    game_title: str = Field(default="Untitled Concept")
    core_genre: str = Field(default="unclassified")
    elevator_pitch: str = Field(default="")
    core_fantasy: str = Field(
        default="",
        description="Who the player is, what they feel, and why this world matters."
    )
    first_60_seconds_hook: str = Field(
        default="",
        description="Exact sensory and interactive experience of the first 60 seconds of play."
    )
    diegetic_ui_philosophy: str = Field(
        default="",
        description="How state, feedback, and onboarding are communicated naturally through the world and clean visual indicators rather than static instruction text walls."
    )
    camera_and_spatial_paradigm: str = Field(
        default="",
        description="Locked 2D camera perspective and spatial movement model agreed upon by the Writers' Room."
    )
    color_palette_hex: List[str] = Field(
        default_factory=list,
        description="Hex color palette (#RRGGBB) governing sky, terrain, architecture, energy accents, and diegetic UI."
    )
    scene_layers: List[VisualSceneLayerSpec] = Field(
        default_factory=list,
        description="Ordered visual depth layers decomposing the 16:9 mockup into constructible Godot 4 scene geometry."
    )
    gameplay_pillars: List[str] = Field(default_factory=list)
    visual_style_guide: str = Field(default="")
    audio_direction_guide: str = Field(default="")
    concept_evolution: List[ConceptEvolutionEntry] = Field(default_factory=list)
    writers_room_transcript: List[WritersRoomUtterance] = Field(default_factory=list)
    experience_storyboard: List[ExperienceStoryboardBeat] = Field(default_factory=list)
    visual_mockups: List[VisualMockupArtifact] = Field(default_factory=list)
    world_spec: Optional[GenesisWorldSpec] = Field(default=None)
    narrative_bible: Optional[NarrativeBible] = Field(default=None)
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def __getattr__(name: str) -> Any:
    if name == "OpenGenreCartridgeBundle":
        from engine.studio.swe import OpenGenreCartridgeBundle
        return OpenGenreCartridgeBundle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

