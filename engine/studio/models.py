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
from typing import Any, ClassVar, Dict, List, Literal, Optional, Sequence, Set, Tuple, Type, Union
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_STUDIO_MODEL = "gemini-3.8-flash"
DEFAULT_CONCEPT_MOCKUP_MODEL = "gemini-nano-banana-2.1"


def is_real_genai_client(client: Any) -> bool:
    """Returns True when `client` is a live `google.genai` Client rather than None or a unit-test fake."""
    mod = getattr(type(client), "__module__", "") if client is not None else ""
    return mod.startswith("google.genai") and mod != "google.genai.fake"



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


class SceneVerificationError(StudioAgentError):
    """Raised when a scene/visual verifier cannot produce an authentic verdict (Rule 5: no fabricated passes)."""
    deliberation: str = "Scene verification could not be completed; no verdict may be fabricated."


class SceneFidelityRejection(StudioAgentError):
    """Raised when scene/beat fidelity could not be achieved within an explicitly configured iteration ceiling."""
    deliberation: str = "Scene fidelity was not achieved within the configured iteration ceiling."

    def __init__(self, message: str, critique: Any = None, beat_report: Any = None) -> None:
        super().__init__(message)
        self.critique = critique
        self.beat_report = beat_report


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
        """Automatically parses JSON-encoded object strings returned by Gemini for Dict-typed fields
        (including schemas sanitized via `_sanitize_gemini_json_schema`, which rewrites open `Dict`
        objects without fixed properties into JSON-encoded strings)."""
        if not isinstance(data, dict):
            return data
        from typing import Union, get_args, get_origin

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
            "perceptual_descriptions_by_path",
            "code_diffs",
            "animation_states",
            "entries",
        }
        for f_key, f_info in getattr(cls, "model_fields", {}).items():
            ann = getattr(f_info, "annotation", None)
            origin = get_origin(ann)
            if origin is dict:
                dict_field_names.add(f_key)
            elif origin is Union:
                if any(get_origin(arg) is dict or arg is dict for arg in get_args(ann)):
                    dict_field_names.add(f_key)

        out = dict(data)
        for fname in dict_field_names:
            if fname in out and isinstance(out[fname], str):
                raw_str = out[fname].strip()
                if raw_str.startswith("```"):
                    lines = raw_str.splitlines()
                    if len(lines) >= 2 and lines[-1].strip().startswith("```"):
                        raw_str = "\n".join(lines[1:-1]).strip()
                if not raw_str:
                    out[fname] = {}
                    continue
                try:
                    parsed = json.loads(raw_str)
                    if isinstance(parsed, dict):
                        if fname == "perceptual_descriptions_by_path":
                            out[fname] = {
                                str(k): (json.dumps(v) if isinstance(v, (dict, list)) else str(v))
                                for k, v in parsed.items()
                                if str(k).strip()
                            }
                        else:
                            out[fname] = parsed
                    elif isinstance(parsed, list) and fname == "nodes":
                        out[fname] = {
                            (item.get("node_id") or f"node_{idx}"): item
                            for idx, item in enumerate(parsed)
                            if isinstance(item, dict)
                        }
                    elif isinstance(parsed, list) and fname == "perceptual_descriptions_by_path":
                        mapped: Dict[str, str] = {}
                        for idx, item in enumerate(parsed):
                            if isinstance(item, dict):
                                k = str(
                                    item.get("path")
                                    or item.get("asset_path")
                                    or item.get("file")
                                    or f"asset_{idx}"
                                ).strip()
                                v = str(
                                    item.get("description")
                                    or item.get("perceptual_description")
                                    or item.get("summary")
                                    or ""
                                ).strip()
                                if k:
                                    mapped[k] = v
                        out[fname] = mapped
                    elif parsed is None:
                        out[fname] = {}
                    else:
                        out[fname] = {"value": str(parsed) if fname == "perceptual_descriptions_by_path" else parsed}
                except Exception:
                    out[fname] = {"description": raw_str} if fname == "perceptual_descriptions_by_path" else {"directive": raw_str}
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
    concept_mockup_model_name: str = Field(
        default=DEFAULT_CONCEPT_MOCKUP_MODEL,
        description="High-craft model for 16:9 concept art & gameplay mockups",
    )
    max_retries: int = Field(default=3, description="Maximum self-healing repair iterations")
    timeout_seconds: float = Field(default=300.0, description="Per-agent inference timeout")
    pause_for_greenlight: bool = Field(
        default=False,
        description="When True, pause at AWAITING_GREENLIGHT after the Writers' Room locks the Game Bible.",
    )
    continuous_production: Optional[bool] = Field(
        default=None,
        description="Run the council and the implementation side as one continuous loop (backlog, workers, proof review). None = enabled whenever a live Gemini client is used.",
    )
    board_workers: int = Field(default=2, description="Maximum concurrent Production Board workers.")
    max_item_revisions: int = Field(default=5, description="Default revision cap per backlog item before the council must decide.")
    dataset_name: str = Field(
        default="",
        description="Optional dataset/run name injected into Gemini API request labels (`dataset`, `run_id`) for filtering and creating datasets in Google AI Studio Logs.",
    )
    store_logs: bool = Field(
        default=True,
        description="When True, sets `store=True` on Gemini API `generate_content` requests so logs are stored in Google AI Studio for dataset curation.",
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
    deliberation: str = Field(..., description="Epistemic rationale defining character or entity identity and role in the game.")
    character_id: str = Field(..., description="Unique snake_case identifier for this actor or entity.")
    name: str = Field(..., description="Display name.")
    role: str = Field(..., description="Role in the game world or narrative (devised for this game).")
    personality_traits: List[str] = Field(default_factory=list)
    backstory: str = Field(default="")
    voice_timbre: str = Field(default="focused")
    gemini_voice_name: str = Field(default="Aoede", description="'Aoede' | 'Puck' | 'Charon' | 'Fenrir' | 'Kore'.")
    visual_description: str = Field(default="In-game entity")
    portrait_asset: str = Field(default="")


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
        description="Gameplay event or condition that triggers this bark (devised for this game's mechanics)."
    )
    text: str = Field(...)
    voice_clip: str = Field(..., description="Relative path under assets/audio/, e.g. 'assets/audio/bark_01.wav'.")
    portrait: str = Field(default="")
    priority: int = Field(default=1)
    duration: float = Field(default=4.5)


NarrativeCharacterSpec = CharacterSheet


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


class AudioEvaluationReport(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Open-ended perceptual evaluation of audio cues (.wav, etc.) and/or synchronized gameplay clips (.mp4).
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Step-by-step acoustic and/or audiovisual deliberation analyzing timbre, timing, sync with visual events, mix balance, and context before concluding.",
    )
    rich_acoustic_description: str = Field(
        ...,
        description="Detailed natural-language description of what the audio or gameplay clip actually sounds like to a human listener (textures, layers, prosody, mood, and timestamped moments).",
    )
    contextual_observations: List[str] = Field(
        default_factory=list,
        description="Concrete observations connecting what is heard to the caller's context, visual gameplay events, or creative direction.",
    )
    notable_concerns: List[str] = Field(
        default_factory=list,
        description="Audible defects, mismatches, sync lag, harsh synthesis artifacts, loop clicks, silence, or mix clashes observed (empty list if none).",
    )
    answer_to_inquiry: str = Field(
        ...,
        description="Direct, comprehensive answer to the caller's evaluation prompt, including any scores, comparisons, or actionable suggestions requested.",
    )


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


class KeyframeTransformSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Defines an explicit keyframe pose transform at normalized time t in [0.0, 1.0].
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Biomechanical rationale for this keyframe pose at normalized time t.")
    time_normalized: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized timestamp in [0.0, 1.0] within the animation cycle.")
    position_offset_px: Tuple[float, float] = Field(default=(0.0, 0.0), description="(dx, dy) pixel translation relative to rest attachment.")
    rotation_degrees: float = Field(default=0.0, description="Rotation angle in degrees around the part or rig pivot.")
    scale_xy: Tuple[float, float] = Field(default=(1.0, 1.0), description="(sx, sy) squash/stretch scale factor at this keyframe.")


class PartKeyframeTrackSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Defines an ordered sequence of KeyframeTransformSpecs for a specific body/mechanism part.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Why this part's keyframe curve communicates the state's motion.")
    part_id: str = Field(..., description="Target component or part_id within the articulated rig or entity.")
    keyframes: List[KeyframeTransformSpec] = Field(default_factory=list, description="Ordered keyframes from t=0.0 to t=1.0.")


class AnimationStateSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Defines an agent-authored animation state or pose sequence for a sprite sheet / animated asset.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Rationale for why this animation state is needed for gameplay/visual feedback.")
    state_name: str = Field(..., description="Animation state identifier tailored to this entity.")
    motion_profile: str = Field(
        default="idle",
        description="Kinematic articulation or motion profile tailored to this entity and state (devised for the subject rather than restricted to a fixed list)."
    )
    frame_count: int = Field(default=4, ge=1, le=12, description="Number of frames in this animation state.")
    fps: float = Field(default=8.0, description="Playback rate in frames per second.")
    loop: bool = Field(default=True, description="Whether this animation loops continuously.")
    pose_progression_prompt: str = Field(
        default="",
        description="Detailed visual description of how the silhouette, limbs, or mechanism progress across frames."
    )
    keyframe_tracks: List[PartKeyframeTrackSpec] = Field(
        default_factory=list,
        description="Optional explicit keyframe transform tracks per part/region across t in [0.0, 1.0]."
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


MemoryKind = Literal[
    "strategic_goal",
    "architectural_decision",
    "lesson_learned",
    "domain_note",
]

MemoryStatus = Literal[
    "active",
    "fulfilled",
    "superseded",
    "archived",
]

MemoryMutationOperation = Literal[
    "record",
    "update",
    "fulfill",
    "supersede",
    "archive",
]

CANONICAL_MEMORY_BANKS: Tuple[str, ...] = (
    "council",
    "CoProducer",
    "CreativeDirector",
    "GameplayDesigner",
    "LevelAndProgressionDesigner",
    "GameplaySystemsEngineer",
    "StoryWorldArchitect",
    "ScriptwriterAgent",
    "CharacterDesigner",
    "VisualArtDirector",
    "UXExperienceDesigner",
    "AudioAtmosphereDesigner",
    "task_lead",
    "swe_engineer",
    "qa_test_engineer",
    "visual_artist",
    "audio_designer",
    "script_writer",
)


class GameMemoryEntry(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A durable per-game memory record persisted across tasks and vertical slices inside either
    the shared 'council' bank or an individual specialist agent's memory bank.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining why this durable memory entry matters across tasks or vertical slices.",
    )
    memory_id: str = Field(
        default_factory=lambda: f"mem_{uuid.uuid4().hex[:10]}",
        description="Stable unique identifier for this memory entry within the game.",
    )
    memory_bank: str = Field(
        default="council",
        description="Owning memory bank ('council' for shared Council memory, or a canonical specialist role bank).",
    )
    author_role: str = Field(
        default="council",
        description="Agent role or 'executive_producer' that recorded or last updated this memory.",
    )
    kind: MemoryKind = Field(
        default="lesson_learned",
        description="Category: 'strategic_goal' (multi-slice or deferred direction), 'architectural_decision' (design/technical trade-off), 'lesson_learned' (operational/verification constraint), or 'domain_note' (craft knowledge).",
    )
    status: MemoryStatus = Field(
        default="active",
        description="Lifecycle state: 'active', 'fulfilled', 'superseded', or 'archived'.",
    )
    title: str = Field(
        ...,
        description="Concise headline summarizing the durable goal, decision, or lesson.",
    )
    content: str = Field(
        ...,
        description="Detailed, actionable guidance, rationale, or constraint for future tasks and slices.",
    )
    applicable_roles: List[str] = Field(
        default_factory=list,
        description="Optional target roles for whom ContextCuratorAgent should proactively surface this memory (empty means general to bank/game).",
    )
    slice_id: str = Field(
        default="",
        description="Slice identifier active when this memory was created or updated.",
    )
    item_id: str = Field(
        default="",
        description="Optional Production Board work item identifier linked to this memory.",
    )
    resolution_note: str = Field(
        default="",
        description="Optional explanation recorded when a memory entry is marked fulfilled, superseded, or archived.",
    )
    superseded_by_id: str = Field(
        default="",
        description="Optional memory_id that superseded this entry.",
    )
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    updated_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class MemoryMutationProposal(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Structured mutation emitted by a Council or Production Board agent to record, refine,
    fulfill, supersede, or archive a durable per-game memory entry.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Counterfactual durability rationale explaining why future tasks or vertical slices would fail or lose deferred direction without this memory, and why it is NOT already captured in the Living Game Bible, codebase, or current task spec.",
    )
    operation: MemoryMutationOperation = Field(
        default="record",
        description="'record' to add a durable memory, 'update' to refine an existing entry, 'fulfill' when a strategic goal is now delivered, 'supersede' when replaced, or 'archive' when obsolete.",
    )
    target_memory_id: str = Field(
        default="",
        description="Existing `memory_id` when operation is 'update', 'fulfill', 'supersede', or 'archive' (leave '' when operation='record').",
    )
    target_bank: str = Field(
        default="self",
        description="'self' to store in your own agent memory bank, 'council' to share cross-cutting goals/lessons with the Council, or a canonical role bank.",
    )
    kind: MemoryKind = Field(
        default="lesson_learned",
        description="'strategic_goal', 'architectural_decision', 'lesson_learned', or 'domain_note'.",
    )
    title: str = Field(
        default="",
        description="Concise headline (required when operation='record').",
    )
    content: str = Field(
        default="",
        description="Actionable long-term guidance or constraint (required when operation in ('record', 'update')).",
    )
    applicable_roles: List[str] = Field(
        default_factory=list,
        description="Optional peer roles that should proactively receive this memory during context curation.",
    )
    resolution_note: str = Field(
        default="",
        description="Optional note explaining how a strategic goal was fulfilled, superseded, or archived.",
    )


class GameMemoryLedger(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Per-cartridge long-term memory container persisted in `game_memory.json` and mirrored to SQLite WAL.
    """
    deliberation: str = Field(
        default="Authoritative per-game multi-bank long-term memory ledger.",
        description="MANDATORY FIRST PROPERTY: Epistemic summary of the game memory ledger.",
    )
    cartridge_id: str = Field(
        default="",
        description="Cartridge directory identifier owning this memory ledger.",
    )
    game_title: str = Field(
        default="",
        description="Current title of the game.",
    )
    updated_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    entries: List[GameMemoryEntry] = Field(
        default_factory=list,
        description="All durable memory entries across 'council' and individual agent memory banks.",
    )


class AssetRequirement(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Dynamic multimodal visual or audio asset required by the conceived game and fabricated into the Cartridge Asset Library.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST: Rationale explaining why this purpose-built asset is required.")
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, summarize what you learned from the retrieved memory entries before finalizing this asset.",
    )
    memory_query: str = Field(
        default="",
        description="Optional semantic search query against long-term game/agent memory banks before finalizing this asset (leave '' when current context suffices).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter for `memory_query` ('', 'all', 'self', 'council', or a canonical role name).",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="DEFAULT TO [] — LEAVE EMPTY ON ROUTINE TURNS. Never log task progress, summarize fabricated assets, or restate the Game Bible/asset catalog. Populate ONLY for non-obvious reusable visual craft invariants or asset-pipeline lessons needed across future tasks/slices (or to update/fulfill an existing memory_id).",
    )
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
    grid_cols: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of horizontal grid columns (hframes) when fabricating a multi-cell sheet/atlas of distinct states, icons, stages, or variants (1 for a single asset).",
    )
    grid_rows: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of vertical grid rows (vframes) when fabricating a multi-cell sheet/atlas of distinct states, icons, stages, or variants (1 for a single asset).",
    )
    cell_prompts: List[str] = Field(
        default_factory=list,
        description="When grid_cols * grid_rows > 1 and the task requires distinct visual states, icons, stages, or variants in each grid cell (rather than a cyclical character deformation), provide the specific visual prompt for each cell in row-major order (top-left to bottom-right, length == grid_cols * grid_rows).",
    )
    cell_target_sizes: List[Tuple[int, int]] = Field(
        default_factory=list,
        description=(
            "Optional per-cell foreground bounding-box footprints `(max_width_px, max_height_px)` inside each uniform "
            "`(cell_width, cell_height)` grid cell (row-major order, `length == grid_cols * grid_rows`, or empty `[]` "
            "when every cell should fill ~88% of its grid cell normally). Populate this whenever a multi-cell grid/sheet "
            "(or padded sprite) requires distinct subject pixel sizes or size tiers centered inside uniform grid cells."
        ),
    )
    chroma_key_hex: Literal["#FF00FF", "#00FF00", "#0000FF", "#FFFF00"] = Field(
        default="#FF00FF",
        description=(
            "Solid studio chroma-key backdrop color used during sprite cutout synthesis ('#FF00FF' magenta, "
            "'#00FF00' green, '#0000FF' blue, or '#FFFF00' yellow). Choose a chroma key with maximum RGB separation "
            "from ALL foreground colors, glows, and accents in the asset: do NOT pick '#FF00FF' when the asset "
            "contains magenta/pink/purple/crimson; do NOT pick '#00FF00' when it contains green/lime; do NOT pick "
            "'#0000FF' when it contains saturated blue/indigo; do NOT pick '#FFFF00' when it contains yellow/amber/gold."
        ),
    )
    camera_projection: str = Field(
        default="",
        description=(
            "Locked spatial camera projection inherited from GameBibleSpec.camera_and_spatial_paradigm "
            "(e.g. 'strict_top_down_90deg', 'side_orthographic', 'three_quarter_top_down', 'isometric_2_to_1')—"
            "every asset in the cartridge MUST share one consistent projection and never mix incompatible perspectives."
        ),
    )
    world_footprint_px: Optional[Tuple[int, int]] = Field(
        default=None,
        description=(
            "Intended rendered footprint `(width_px, height_px)` in world coordinates so the native texture/cell "
            "resolution is synthesized at `>= 128x128` (and `>= 1.5x` `world_footprint_px`), downsampled cleanly in Godot "
            "(`scale <= 1.0`) without upscaling blur or overflowing adjacent grid/socket spacing."
        ),
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

            if out.get("cell_target_sizes") is None:
                out["cell_target_sizes"] = []
            raw_ck = str(out.get("chroma_key_hex") or "#FF00FF").strip().upper()
            if not raw_ck.startswith("#") and len(raw_ck) == 6:
                raw_ck = f"#{raw_ck}"
            out["chroma_key_hex"] = raw_ck if raw_ck in ("#FF00FF", "#00FF00", "#0000FF", "#FFFF00") else "#FF00FF"

            desc = out.get("description")
            vprompt = out.get("visual_prompt_or_text")
            if str(out.get("memory_query") or "").strip() and not vprompt:
                out["description"] = desc or ""
                out["visual_prompt_or_text"] = ""
            else:
                final_text = desc or vprompt or out.get("purpose") or f"Asset {aid}"
                out["description"] = final_text
                out["visual_prompt_or_text"] = vprompt or final_text
            return out
        return data


class AudioAssetPlanBatch(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Plans one or more audio assets (a single audio cue or a multi-cue soundbank) required by an audio task or brief.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST: Rationale enumerating every audio file/cue requested by the task specification or brief and its acoustic character, frequency register, and fabrication mode.",
    )
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, summarize what you learned from the retrieved memory entries before finalizing `assets`.",
    )
    memory_query: str = Field(
        default="",
        description="Optional semantic search query against long-term game/agent memory banks before finalizing `assets` (leave '' when current context suffices).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter for `memory_query` ('', 'all', 'self', 'council', or a canonical role name).",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="DEFAULT TO [] — LEAVE EMPTY ON ROUTINE TURNS. Never log routine cue synthesis or restate the Game Bible/asset catalog. Populate ONLY for non-obvious acoustic invariants or audio-pipeline lessons needed across future tasks/slices (or to update/fulfill an existing memory_id).",
    )
    assets: List[AssetRequirement] = Field(
        default_factory=list,
        description="One AssetRequirement per requested audio cue (.wav file). When the task or brief requests multiple audio cues or a soundbank, include a separate AssetRequirement for EVERY requested cue with its own target_path (under assets/audio/...) and visual_prompt_or_text.",
    )


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
    perceptual_description: str = Field(
        default="",
        description="Optional natural-language perceptual description of an asset's actual rendered pixels, silhouette/framing, or acoustic sound/texture/timing from VisualEvaluatorAgent or AudioEvaluatorAgent.",
    )
    duration_seconds: float = Field(
        default=0.0,
        description="Duration in seconds for audio or video assets (0.0 for static/sprite assets).",
    )
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
                        s = m.strip().replace("\\", "/")
                        if s.startswith("res://"):
                            s = s[len("res://"):]
                        elif s.startswith("res:/"):
                            s = s[len("res:/"):]
                        s = s.lstrip("/").replace(".py", ".gd")
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
        description="MANDATORY FIRST PROPERTY: Concise 1-sentence rationale (<= 20 words) stating this file's architectural role and contract."
    )
    filename: str = Field(
        ...,
        description="Relative file path within cartridge without 'res://' prefix (e.g. 'main.gd', 'mechanics.gd', 'entities.gd', 'scenes.gd', 'ui_hud.gd', 'project.godot', 'main.tscn')."
    )
    content: str = Field(
        ...,
        description="Complete, syntactically valid source code or configuration for this module."
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_filename(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("filename"), str):
            out = dict(data)
            fn = out["filename"].strip().replace("\\", "/")
            if fn.startswith("res://"):
                fn = fn[len("res://"):]
            elif fn.startswith("res:/"):
                fn = fn[len("res:/"):]
            fn = fn.lstrip("/")
            if "/" in fn and Path(fn).name in {
                "main.gd",
                "mechanics.gd",
                "scenes.gd",
                "entities.gd",
                "ui_hud.gd",
                "main.tscn",
                "project.godot",
            }:
                fn = Path(fn).name
            out["filename"] = fn
            return out
        return data


class SurgicalCodePatch(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A targeted line-range replacement within an existing cartridge file.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Concise 1-sentence rationale (<= 20 words) explaining why replacing lines [start_line..end_line] in filename satisfies the requirement."
    )
    filename: str = Field(
        ...,
        description="Target relative file path in cartridge without 'res://' prefix (e.g. 'scenes/camera/train_camera.gd', 'level_map.gd', 'main.gd')."
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
    target_snippet: str = Field(
        default="",
        description="Optional exact existing lines at [start_line..end_line] being replaced (without line-number prefixes) to anchor the patch if line numbers shifted."
    )
    replacement_code: str = Field(
        ...,
        description="Exact replacement GDScript lines (without line number prefixes, using tab '\\t' indentation matching the surrounding block) to substitute for lines [start_line..end_line]."
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_patch_filename(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("filename"), str):
            out = dict(data)
            fn = out["filename"].strip().replace("\\", "/")
            if fn.startswith("res://"):
                fn = fn[len("res://"):]
            elif fn.startswith("res:/"):
                fn = fn[len("res:/"):]
            fn = fn.lstrip("/")
            if "/" in fn and Path(fn).name in {
                "main.gd",
                "mechanics.gd",
                "scenes.gd",
                "entities.gd",
                "ui_hud.gd",
                "main.tscn",
                "project.godot",
            }:
                fn = Path(fn).name
            out["filename"] = fn
            return out
        return data


def _strip_numbered_line_prefix(raw_line: str) -> str:
    """Strips an accidental ` 123: ` line-number prefix if an LLM echoed numbered slice output."""
    if len(raw_line) >= 6 and raw_line[:4].strip().isdigit() and raw_line[4:6] == ": ":
        return raw_line[6:]
    return raw_line


def _locate_snippet_line_range(
    file_lines: List[str],
    snippet_text: str,
    hint_start_line: int,
) -> Optional[Tuple[int, int]]:
    """Attempts to locate `snippet_text` in `file_lines` (1-indexed inclusive `(start, end)`),
    preferring the match closest to `hint_start_line`."""
    raw_target = [_strip_numbered_line_prefix(ln) for ln in snippet_text.splitlines()]
    while raw_target and not raw_target[0].strip():
        raw_target.pop(0)
    while raw_target and not raw_target[-1].strip():
        raw_target.pop()
    if not raw_target:
        return None
    n_t = len(raw_target)
    total = len(file_lines)
    if n_t > total:
        return None

    # First try exact line match (after stripping trailing whitespace)
    stripped_target = [ln.rstrip() for ln in raw_target]
    stripped_file = [ln.rstrip() for ln in file_lines]
    candidates: List[Tuple[int, int]] = []
    for i in range(total - n_t + 1):
        if stripped_file[i : i + n_t] == stripped_target:
            candidates.append((i + 1, i + n_t))

    # Fallback: match ignoring leading/trailing whitespace if multi-line or non-trivial single line
    if not candidates and (n_t >= 2 or len(raw_target[0].strip()) >= 12):
        norm_target = [ln.strip() for ln in raw_target]
        norm_file = [ln.strip() for ln in file_lines]
        for i in range(total - n_t + 1):
            if norm_file[i : i + n_t] == norm_target:
                candidates.append((i + 1, i + n_t))

    if not candidates:
        return None
    candidates.sort(key=lambda rng: abs(rng[0] - max(1, hint_start_line)))
    return candidates[0]


def apply_surgical_patches_to_directory(
    target_dir: Path,
    patches: Sequence[SurgicalCodePatch],
    allowed_files: Optional[Set[str]] = None,
    path_resolver: Optional[Any] = None,
) -> Dict[str, Path]:
    """Applies `SurgicalCodePatch` entries to files in `target_dir` in descending line order,
    using `target_snippet` anchoring when provided."""
    updated: Dict[str, Path] = {}
    if not patches:
        return updated

    canonical_root_only = {"main.gd", "main.tscn", "project.godot"}
    patches_by_file: Dict[str, List[SurgicalCodePatch]] = {}
    resolved_paths: Dict[str, Path] = {}

    for p in patches:
        if callable(path_resolver):
            rel_name, fpath = path_resolver(target_dir, p.filename)
        else:
            rel_name = p.filename.strip().replace("\\", "/")
            if rel_name.startswith("res://"):
                rel_name = rel_name[len("res://"):]
            elif rel_name.startswith("res:/"):
                rel_name = rel_name[len("res:/"):]
            rel_name = rel_name.lstrip("/")
            if "/" in rel_name and Path(rel_name).name in canonical_root_only:
                rel_name = Path(rel_name).name
            fpath = target_dir / rel_name
            if not fpath.is_file():
                bname = Path(rel_name).name
                matches = [
                    cand for cand in target_dir.rglob(bname)
                    if cand.is_file() and not any(part in (".godot", "artifacts") for part in cand.relative_to(target_dir).parts)
                ]
                if len(matches) == 1:
                    fpath = matches[0]
                    rel_name = fpath.relative_to(target_dir).as_posix()

        if allowed_files is not None and rel_name not in allowed_files:
            continue
        if rel_name.endswith((".gd", ".tscn", ".gdshader", ".godot")) and fpath.is_file():
            patches_by_file.setdefault(rel_name, []).append(p)
            resolved_paths[rel_name] = fpath

    for fname, file_patches in patches_by_file.items():
        fpath = resolved_paths[fname]
        if not fpath.is_file():
            continue
        lines = fpath.read_text(encoding="utf-8").splitlines()
        total = len(lines)

        resolved_ranges: List[Tuple[int, int, SurgicalCodePatch]] = []
        for patch in file_patches:
            s = max(1, min(patch.start_line, max(1, total)))
            e = max(s, min(patch.end_line, max(1, total)))
            if getattr(patch, "target_snippet", ""):
                anchored = _locate_snippet_line_range(lines, patch.target_snippet, s)
                if anchored is not None:
                    s, e = anchored
            resolved_ranges.append((s, e, patch))

        # Sort descending by start_line so earlier line indices do not shift
        resolved_ranges.sort(key=lambda item: (item[0], item[1]), reverse=True)
        last_start = total + 1
        applied_any = False

        for s, e, patch in resolved_ranges:
            if e >= last_start:
                continue
            cleaned_replacement = [
                _strip_numbered_line_prefix(raw_line)
                for raw_line in patch.replacement_code.splitlines()
            ]
            lines[s - 1 : e] = cleaned_replacement
            last_start = s
            applied_any = True

        if applied_any:
            fpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
            updated[fname] = fpath

    return updated


_PROTECTED_CARTRIDGE_DELETE_FILES: Set[str] = {
    "project.godot",
    "main.tscn",
    "main.gd",
    "game_bible.json",
    "backlog.json",
    "production_board.json",
    "game_memory.json",
    "manifest.json",
    "narrative.json",
    "level_layout.json",
    "assets/asset_library.json",
    "asset_library.json",
}
_PROTECTED_CARTRIDGE_DELETE_DIRS: Set[str] = {
    ".godot",
    "artifacts",
    "tests",
    "mockups",
    "concepts",
    "playtest_captures",
}
_MEDIA_DELETE_SUFFIXES: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".svg",
    ".wav",
    ".ogg",
    ".mp3",
    ".flac",
    ".import",
)


def apply_sandboxed_cartridge_deletions(
    cartridge_dir: Union[Path, str],
    delete_paths: Sequence[str],
    *,
    caller_role: str = "",
) -> List[str]:
    """Deletes obsolete/orphaned files strictly inside `cartridge_dir`, blocking path traversal,
    protected root/control files, and out-of-domain deletions, and prunes deleted assets from
    `assets/asset_library.json`. Returns normalized relative paths deleted."""
    if not delete_paths:
        return []
    cdir = Path(cartridge_dir).resolve()
    if not cdir.is_dir():
        return []

    deleted_rel: List[str] = []
    for raw in delete_paths:
        rel = str(raw or "").strip().replace("\\", "/")
        if rel.startswith("res://"):
            rel = rel[len("res://") :]
        elif rel.startswith("res:/"):
            rel = rel[len("res:/") :]
        rel = rel.lstrip("/")
        if not rel or ".." in Path(rel).parts:
            raise StudioAgentError(
                f"Unsafe or invalid delete_paths entry '{raw}' rejected by cartridge sandbox."
            )
        if rel in _PROTECTED_CARTRIDGE_DELETE_FILES:
            raise StudioAgentError(
                f"Protected cartridge file '{rel}' cannot be deleted via delete_paths."
            )
        rel_parts = set(Path(rel).parts)
        if rel_parts & _PROTECTED_CARTRIDGE_DELETE_DIRS:
            raise StudioAgentError(
                f"Protected cartridge directory path '{rel}' cannot be deleted via delete_paths."
            )
        if caller_role in ("visual_artist", "audio_designer"):
            if not rel.startswith("assets/") or not rel.lower().endswith(_MEDIA_DELETE_SUFFIXES):
                raise StudioAgentError(
                    f"Specialist '{caller_role}' may only delete obsolete media files under 'assets/*' (got '{rel}')."
                )
        cand = (cdir / rel).resolve()
        if not cand.is_relative_to(cdir) or cand == cdir:
            raise StudioAgentError(
                f"Path '{raw}' resolves outside cartridge directory '{cdir}' and was blocked."
            )
        if cand.is_dir():
            raise StudioAgentError(
                f"Directory deletion is not permitted via delete_paths (got '{rel}'); specify individual file paths."
            )
        if cand.is_file():
            cand.unlink()
            sidecar_import = Path(str(cand) + ".import")
            if sidecar_import.is_file() and sidecar_import.resolve().is_relative_to(cdir):
                sidecar_import.unlink(missing_ok=True)
            if rel not in deleted_rel:
                deleted_rel.append(rel)

    if deleted_rel:
        cat_path = cdir / "assets" / "asset_library.json"
        if cat_path.is_file():
            try:
                raw_cat = json.loads(cat_path.read_text(encoding="utf-8"))
                entries = raw_cat.get("entries") if isinstance(raw_cat, dict) else None
                if isinstance(entries, dict):
                    deleted_set = set(deleted_rel)
                    keys_to_prune = [
                        k
                        for k, ent in entries.items()
                        if isinstance(ent, dict)
                        and str(ent.get("file_path") or ent.get("res_path") or "")
                        .replace("res://", "")
                        .lstrip("/")
                        in deleted_set
                    ]
                    if keys_to_prune:
                        for pk in keys_to_prune:
                            entries.pop(pk, None)
                        cat_path.write_text(json.dumps(raw_cat, indent=2), encoding="utf-8")
            except Exception as cat_exc:
                logger.warning("Failed pruning deleted assets %s from asset_library.json: %s", deleted_rel, cat_exc)

    return deleted_rel


class TrackCodeSynthesisPayload(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Payload returned by LeadGameplaySWEAgent for full synthesis or requirement track implementation.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining the architectural decisions, Godot 4 node composition, and track requirement fulfillment."
    )
    self_question_answer: str = Field(
        default="",
        description="If you asked a `self_question` on the previous turn, write your concise, factual answer here FIRST based on the retrieved context chunks (citing exact method/signal/variable names) before writing `patches` or `files`.",
    )
    self_question: str = Field(
        default="",
        description="Optional self-inquiry: If you lack critical context (e.g. an imported script's exact method/signal signature, node structure, shader uniforms, or design spec) needed to write code without guessing, output your specific question here and leave `patches` and `files` empty (`[]`). ContextCuratorAgent will fetch the needed context chunks so you can answer your own question on the next turn.",
    )
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, write your concise synthesis of the retrieved long-term memory entries here before writing `patches` or `files`.",
    )
    memory_query: str = Field(
        default="",
        description="Optional direct query against long-term game/agent memory banks: if you want to look up prior architectural decisions, multi-slice goals, or technical lessons learned before coding, enter a semantic search query here and leave `patches` and `files` empty (`[]`).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter when issuing `memory_query` ('', 'all', 'self', 'council', or a canonical specialist role name).",
    )
    tool_search_answer: str = Field(
        default="",
        description="If you issued `tool_search_query` on the previous turn, summarize the relevant tool capabilities/modes here before proceeding.",
    )
    tool_search_query: str = Field(
        default="",
        description="Optional progressive tool discovery query: if you want detailed schemas, sub-modes, or composition patterns for Studio tools/specialists before acting, enter a semantic query here and leave `patches` and `files` empty (`[]`).",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="DEFAULT TO [] — LEAVE EMPTY ON ROUTINE TURNS. Never log what code you just wrote, restate the Game Bible/spec, or record local single-file fixes. Populate ONLY when establishing a cross-module architectural invariant or hard-won runtime lesson future tasks/slices would break without knowing (or to update/fulfill an existing memory_id).",
    )
    diagnostic_test_scene_request: str = Field(
        default="",
        description=(
            "Optional runtime diagnostic request for `qa_test_engineer`: If you run into trouble fixing an issue or are uncertain what is "
            "causing an observed behavior at runtime, describe the diagnostic test scene, visual debug overlays, "
            "and `get_state_snapshot()` variables you need QA to capture. If you need to inspect that diagnostic output BEFORE writing code changes, "
            "populate `diagnostic_test_scene_request` and leave `patches` and `files` empty (`[]`)."
        ),
    )
    inspection_report: str = Field(
        default="",
        description=(
            "When `operation_mode == 'inspect_only'` (a read-only architecture, scene, shader, or asset-wiring audit requested by TaskLead or Council), "
            "write your structured technical findings citing exact file paths, node paths, method signatures, and asset dimensions here."
        ),
    )
    recommended_followups: List[str] = Field(
        default_factory=list,
        description="Optional concise follow-up recommendations or out-of-scope observations surfaced back to TaskLead.",
    )
    requested_assets: List[AssetRequirement] = Field(
        default_factory=list,
        description="On-demand purpose-built assets (sprite sheets, articulated parts, props, tiles, parallax layers, UI, SFX) requested by the SWE agent for the Cartridge Asset Library."
    )
    patches: List[SurgicalCodePatch] = Field(
        default_factory=list,
        description=(
            "PREFERRED FOR EXISTING FILES ON DISK: Surgical line-range replacements (`filename`, `start_line`, `end_line`, "
            "`target_snippet`, `replacement_code`) to modify existing `.gd` / `.gdshader` / `.tscn` files without regenerating the entire file."
        ),
    )
    files: List[ModuleFileSpec] = Field(
        default_factory=list,
        description=(
            "For brand-new files that do not yet exist on disk (or complete replacements of small <100-line files): "
            "collection of cartridge files to create."
        ),
    )
    manifest_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="manifest.json content with metadata, genre, version, entrypoint."
    )
    delete_paths: List[str] = Field(
        default_factory=list,
        description=(
            "Optional cartridge-relative paths of obsolete/orphaned files (e.g. unused `assets/...` PNG/WAV files "
            "or superseded helper scripts—never `project.godot`, `main.tscn`, `main.gd`, or `tests/*`) to delete cleanly from disk."
        ),
    )
    reroute_to_specialist: str = Field(
        default="",
        description=(
            "Optional peer handoff: if inspecting the codebase proves the root cause belongs to another Specialist Persona "
            "('visual_artist', 'audio_designer', 'script_writer', or 'qa_test_engineer'—for example, a corrupted/blank `.png` "
            "file on disk rather than a GDScript bug), set that specialist's name here and set `operation_mode='reroute'`."
        ),
    )
    reroute_brief: str = Field(
        default="",
        description="When `reroute_to_specialist` is non-empty, concrete handoff brief explaining what you verified in code and what the target specialist needs to inspect, create, or fix.",
    )
    reroute_target_paths: List[str] = Field(
        default_factory=list,
        description="Optional cartridge-relative file paths for `reroute_to_specialist`.",
    )
    operation_mode: Literal["edit", "inspect_only", "reroute"] = Field(
        default="edit",
        description="Decision (strictly last): 'edit' when creating/modifying/deleting files or requesting assets; 'inspect_only' for a lock-free read-only technical audit with `inspection_report`; 'reroute' to hand off to `reroute_to_specialist` without mutating files.",
    )

    def assemble_cartridge_directory(
        self,
        target_dir: Path,
        swe_agent: Optional[Any] = None,
        pre_write_hook: Optional[Any] = None,
        commit_gate: Optional[Any] = None,
        allowed_files: Optional[Set[str]] = None,
    ) -> Dict[str, Path]:
        """Assembles, verifies, and writes modular cartridge files and surgical patches to target_dir.
        Returns dictionary mapping filename -> Path.
        """
        import contextlib

        target_dir.mkdir(parents=True, exist_ok=True)
        results: Dict[str, Path] = {}

        # 0. Fabricate any on-demand requested_assets into the CartridgeAssetLibrary first
        try:
            from engine.studio.asset_library import CartridgeAssetLibrary
            visual_agent = getattr(swe_agent, "visual_agent", None) if swe_agent is not None else None
            voice_agent = getattr(swe_agent, "voice_agent", None) if swe_agent is not None else None
            swe_client = getattr(swe_agent, "client", None) if swe_agent is not None else None
            lib = CartridgeAssetLibrary(
                cartridge_dir=target_dir,
                visual_agent=visual_agent,
                voice_agent=voice_agent,
                client=swe_client,
            )
            if self.requested_assets:
                try:
                    from engine.studio.workers import record_turn_substep
                    record_turn_substep(
                        "generating_asset",
                        f"Generating on-demand asset(s): {', '.join(getattr(a, 'asset_id', 'asset') for a in self.requested_assets)}",
                        details={"assets": [getattr(a, "asset_id", "") for a in self.requested_assets]},
                    )
                except Exception:
                    pass
                fabricated = lib.fabricate_assets(self.requested_assets)
                results.update(fabricated)
            lib.save_catalog()
        except Exception as exc:
            # Rule 5: required-asset fabrication failures must never be silently swallowed.
            raise StudioAgentError(
                f"On-demand asset fabrication failed for cartridge '{target_dir}': {exc!r}"
            ) from exc

        canonical_root_only = {"main.gd", "main.tscn", "project.godot"}

        def _norm_rel(raw_path: str) -> str:
            rel_fname = str(raw_path or "").strip().replace("\\", "/")
            if rel_fname.startswith("res://"):
                rel_fname = rel_fname[len("res://"):]
            elif rel_fname.startswith("res:/"):
                rel_fname = rel_fname[len("res:/"):]
            rel_fname = rel_fname.lstrip("/")
            if "/" in rel_fname and Path(rel_fname).name in canonical_root_only:
                rel_fname = Path(rel_fname).name
            return rel_fname

        normalized_files: List[Tuple[str, str]] = []
        for fspec in self.files:
            rel_fname = _norm_rel(fspec.filename)
            if rel_fname:
                cleaned_lines = [_strip_numbered_line_prefix(ln) for ln in fspec.content.splitlines()]
                cleaned_content = "\n".join(cleaned_lines) + ("\n" if fspec.content.endswith("\n") else "")
                normalized_files.append((rel_fname, cleaned_content))

        normalized_patches: List[SurgicalCodePatch] = []
        for pspec in self.patches:
            rel_fname = _norm_rel(pspec.filename)
            if rel_fname:
                normalized_patches.append(pspec.model_copy(update={"filename": rel_fname}))

        all_touched_rel: List[str] = list(
            dict.fromkeys(
                [rel_fname for rel_fname, _ in normalized_files]
                + [p.filename for p in normalized_patches]
            )
        )

        if callable(pre_write_hook):
            pre_write_hook(all_touched_rel, self)
            allowed_after_hook = set(all_touched_rel)
            normalized_files = [(f, c) for f, c in normalized_files if f in allowed_after_hook]
            normalized_patches = [p for p in normalized_patches if p.filename in allowed_after_hook]

        gate_cm = commit_gate if commit_gate is not None else contextlib.nullcontext()
        with gate_cm:
            if self.delete_paths:
                deleted_rel_list = apply_sandboxed_cartridge_deletions(
                    target_dir, self.delete_paths, caller_role="swe_engineer"
                )
                for d_rel in deleted_rel_list:
                    results[d_rel] = target_dir / d_rel
                if deleted_rel_list:
                    try:
                        from engine.studio.workers import record_turn_substep
                        record_turn_substep(
                            "deleting_obsolete_files",
                            f"Deleted obsolete cartridge file(s): {deleted_rel_list}",
                            details={"deleted_paths": deleted_rel_list},
                        )
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

            # 3a. Write synthesized new/full module files
            for rel_fname, content in normalized_files:
                file_path = target_dir / rel_fname
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
                results[rel_fname] = file_path
                try:
                    from engine.studio.workers import record_turn_substep
                    record_turn_substep(
                        "editing_file",
                        f"Editing file {rel_fname} ({len(content.splitlines())} lines)",
                        details={"file": rel_fname, "lines": len(content.splitlines())},
                    )
                except Exception:
                    pass

            # 3b. Apply surgical patches to existing files
            if normalized_patches:
                resolver = getattr(swe_agent, "_resolve_cartridge_gd_path", None) if swe_agent is not None else None
                patched_map = apply_surgical_patches_to_directory(
                    target_dir=target_dir,
                    patches=normalized_patches,
                    allowed_files=allowed_files,
                    path_resolver=resolver,
                )
                for rel_fname, fpath in patched_map.items():
                    results[rel_fname] = fpath
                    patch_cnt = sum(1 for p in normalized_patches if p.filename == rel_fname)
                    try:
                        from engine.studio.workers import record_turn_substep
                        total_lines = len(fpath.read_text(encoding="utf-8").splitlines()) if fpath.is_file() else 0
                        record_turn_substep(
                            "editing_file",
                            f"Patching file {rel_fname} ({patch_cnt} surgical hunk(s), {total_lines} lines total)",
                            details={"file": rel_fname, "patches": patch_cnt, "lines": total_lines},
                        )
                    except Exception:
                        pass

            # 3c. Enforce HiDPI Retina window sizing & resizable=true even if LLM overwrote project.godot
            if project_godot.is_file():
                enforce_hidpi_project_godot(project_godot)
                results["project.godot"] = project_godot

            # 4. Write manifest.json
            if self.manifest_data:
                manifest_path = target_dir / "manifest.json"
                manifest_path.write_text(json.dumps(self.manifest_data, indent=2), encoding="utf-8")
                results["manifest.json"] = manifest_path

            # 5. Run Godot 4 compiler syntax verification & self-healing repair loop when live SWE agent is active
            has_gd_changes = any(
                rel_fname.endswith(".gd")
                for rel_fname in list(results.keys()) + all_touched_rel
            )
            if (
                swe_agent is not None
                and hasattr(swe_agent, "_verify_and_heal_godot_syntax_and_probes")
                and hasattr(swe_agent, "_is_live_genai_client")
                and swe_agent._is_live_genai_client()
                and has_gd_changes
            ):
                try:
                    from engine.studio.workers import record_turn_substep
                    record_turn_substep(
                        "running_tests",
                        "Running headless Godot compiler syntax & probe verification",
                    )
                except Exception:
                    pass
                effective_allowed = set(allowed_files) if allowed_files is not None else None
                if effective_allowed is not None:
                    for rel_fname in all_touched_rel:
                        effective_allowed.add(rel_fname)
                try:
                    healed = swe_agent._verify_and_heal_godot_syntax_and_probes(
                        target_dir, allowed_files=effective_allowed
                    )
                except TypeError:
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
    Defines how a specific depth layer of the environment is constructed in Godot 4.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining how this layer contributes to a cohesive environment matching the Game Bible and concept art."
    )
    layer_name: str = Field(
        ...,
        description="Descriptive name of the scene layer."
    )
    z_index: int = Field(
        default=0,
        description="Godot 4 z_index ordering for this layer."
    )
    construction_technique: str = Field(
        ...,
        description="Concrete Godot 4 construction technique suited to this layer (e.g. procedural Polygon2D/Line2D geometry, custom CanvasItem shaders, GPUParticles2D, lighting, or sprite/rig nodes)."
    )
    required_asset_ids: List[str] = Field(
        default_factory=list,
        description="Optional IDs of discrete sprite sheets, rigs, or prop textures used by this layer (empty if rendered procedurally)."
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
    mapping the Game Bible and 16:9 Concept Mockups into concrete environment topology, rendering techniques, assets, audio, and code architecture.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Holistic strategic reasoning decomposing the Game Bible and concept mockups into cohesive environments, procedural rendering/shaders, required assets, audio, and code."
    )
    environment_topology_strategy: str = Field(
        ...,
        description="Detailed strategy for constructing the game's world and level geometry (using procedural Polygon2D/Line2D meshes, shaders, particles, lighting, or bespoke assets suited to this specific game)."
    )
    camera_and_perspective_coherence_strategy: str = Field(
        default="",
        description="Strategy ensuring coherence between the camera perspective, gameplay movement plane, and world rendering."
    )
    scene_layers: List[SceneLayerStrategy] = Field(
        default_factory=list,
        description="Ordered depth layers composing the playable scene."
    )
    required_visual_assets: List[AssetRequirement] = Field(
        default_factory=list,
        description="Discrete visual assets (sprite sheets, articulated rigs, props) that genuinely benefit from image fabrication rather than procedural/shader rendering."
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
        description="Cross-module code strategy across mechanics.gd, scenes.gd, entities.gd, ui_hud.gd, and main.gd."
    )
    diegetic_ui_strategy: str = Field(
        ...,
        description="Strategy for realizing the Game Bible's UI, HUD, and player feedback design (choosing whatever combination of HUD counters/widgets, text labels, overlays, or in-world visual cues best serves this game)."
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
        description="The specific class, function name, or line range where the deficiency originates."
    )
    deficiency_description: str = Field(
        ...,
        description="Concrete description of the deficiency visible in the rendered Godot viewport or audio code."
    )
    remediation_strategy: str = Field(
        ...,
        description="Concrete surgical code strategy to repair the rendering, spatial, UI, or audio deficiency to match the Game Bible."
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
        description="Detailed evaluation of the rendered environment topology, visual layering, procedural shaders/effects, lighting, and prop/actor placement."
    )
    perspective_and_camera_coherence_critique: str = Field(
        default="",
        description="Detailed evaluation of whether the camera perspective and movement plane match the world rendering."
    )
    audio_and_psychoacoustic_critique: str = Field(
        default="",
        description="Detailed evaluation of the procedural audio synthesis and playback in main.gd (verifying there are no constant high-pitched looping sine-wave drones, piercing >800Hz tones, or harsh repetitive buzzes)."
    )
    ui_and_experience_critique: str = Field(
        ...,
        description="Detailed evaluation of the HUD and player feedback against the Game Bible's UI philosophy and the concept art."
    )
    deficiencies: List[SceneDeficiencyItem] = Field(
        default_factory=list,
        description="Ordered list of concrete deficiencies that must be surgically repaired in the `.gd` modules."
    )
    new_assets_needed: List[AssetRequirement] = Field(
        default_factory=list,
        description="Any missing discrete visual or audio assets that must be fabricated."
    )
    scene_matches_game_bible: bool = Field(
        ...,
        description="Decision verdict (strictly last per Rule 4): True ONLY if the rendered scene genuinely matches the Game Bible and Concept Mockups with coherent camera/spatial perspective, pleasant audio, cohesive visual execution, and a clear, readable UI/HUD."
    )


class WritersRoomListenCheck(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Determines whether a specialist agent listening to the Writers' Room has a high-value
    critique, idea, or artifact to inject at the current moment in the conversation, or whether
    a Core Council Lead wants to call in one of their specialized Sub-Agents to chime in.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic reasoning evaluating the recent conversation from this specialist's perspective."
    )
    should_interject: bool = Field(
        ...,
        description="True if this specialist (or one of their sub-agents) has an urgent critique, synergy, or creative artifact to contribute right now."
    )
    urgency_score: float = Field(
        default=0.0,
        description="Urgency/relevance score in [0.0, 1.0] for speaking next."
    )
    intended_contribution_summary: str = Field(
        default="",
        description="Concise 1-sentence summary of what the agent (or consulted sub-agent) wants to say or propose."
    )
    consult_subagent_role: Optional[str] = Field(
        default=None,
        description="Optional sub-agent role under your domain to call in to speak if their specialized craft will best advance this game right now (or None to speak yourself)."
    )
    subagent_brief: str = Field(
        default="",
        description="If calling in a sub-agent via consult_subagent_role, the creative context or question you want them to address."
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
        description="Concise title for this phase of the player experience."
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
        description="How the player learns or discovers the interaction or objective at this moment (via UI cues, HUD indicators, environmental feedback, or level design)."
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
        description="MANDATORY FIRST PROPERTY: Artistic rationale for composition, palette, lighting, and UI/HUD framing."
    )
    mockup_id: str = Field(default_factory=lambda: f"mockup_{uuid.uuid4().hex[:8]}")
    title: str = Field(..., description="Title of the visual mockup board.")
    scene_description: str = Field(..., description="Detailed visual description and prompt used to synthesize the mockup.")
    image_path: str = Field(..., description="Relative cartridge path, e.g. 'assets/mockups/gameplay_mockup_1.png'.")
    aspect_ratio: str = Field(default="16:9")
    camera_and_spatial_paradigm: str = Field(
        default="",
        description="Explicit camera angle & spatial movement architecture depicted in this mockup."
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
        description="Specific UI, HUD, and visual feedback callouts designed for this game (whatever HUD widgets, counters, labels, gauges, or in-world visual cues best serve playability)."
    )
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MechanicEngineeringSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Engineering strategy authored by GameplaySystemsEngineer in the Writers' Room translating
    a creative gameplay pillar or scene element into concrete Godot 4 nodes, state machines,
    procedural rendering or discrete assets, and falsifiable headless test invariants.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Engineering rationale translating the creative gameplay mechanic or spatial concept into a concrete, deterministic Godot 4 implementation."
    )
    mechanic_id: str = Field(
        ...,
        description="Unique snake_case identifier for this mechanic or system slice."
    )
    title: str = Field(
        ...,
        description="Concise human-readable title of the gameplay system or spatial mechanic."
    )
    target_gd_modules: List[str] = Field(
        default_factory=lambda: ["mechanics.gd", "scenes.gd", "entities.gd", "main.gd"],
        description="Target Godot 4 GDScript files ('main.gd', 'mechanics.gd', 'scenes.gd', 'entities.gd', 'ui_hud.gd') that implement this system."
    )
    godot_node_and_collider_architecture: str = Field(
        ...,
        description="Concrete Godot 4 node hierarchy, physics body motion_mode, procedural/shader rendering or sprite nodes, and 1:1 visual-to-physical collider topology."
    )
    state_machine_and_input_contract: str = Field(
        ...,
        description="Deterministic input triggers, state variables, unsolved t=0 baseline state, and causal transition rules."
    )
    required_sprite_and_tile_assets: List[str] = Field(
        default_factory=list,
        description="Required visual asset specs in format '<asset_id>:<prop|sprite_sheet|articulated_rig|tile>:<visual description>' where EACH distinct character, creature, machine, or interactive structure rig is its own separate entry (1 entity/machine rig per PNG asset—never bundle multiple machines or characters into one asset_id).",
    )
    falsifiable_test_invariants: List[str] = Field(
        default_factory=list,
        description="Concrete falsifiable assertions for headless Godot testing covering unsolved_baseline at t=0, spatial_boundary / negative_counterfactual, and positive_causal state progression."
    )


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
        description="'CoProducer' | 'GameplayDesigner' | 'LevelAndProgressionDesigner' | 'GameplaySystemsEngineer' | 'StoryWorldArchitect' | 'ScriptwriterAgent' | 'CharacterDesigner' | 'VisualArtDirector' | 'UXExperienceDesigner' | 'AudioAtmosphereDesigner' | 'CreativeDirector' | 'UserSteering'"
    )
    speaker_display_name: str = Field(..., description="Human-readable specialist title.")
    spoken_dialogue: str = Field(
        ...,
        description=(
            "Direct, candid conversational contribution to the room—NEVER open with agreement filler or peer flattery "
            "('I completely agree', 'I echo', 'Spot on', 'wonderful'). Lead immediately with concrete critique, unaddressed "
            "risks, trade-offs, or net-new domain proposals while treating any Executive Producer message with full respect."
        ),
    )
    addressed_to: List[str] = Field(
        default_factory=list,
        description="Other specialists specifically called upon or challenged in this turn."
    )
    bible_updates_summary: List[str] = Field(
        default_factory=list,
        description="Summary of concrete additions/revisions made to the Living Game Bible during this turn."
    )
    proof_image_paths: List[str] = Field(
        default_factory=list,
        description="Optional paths to proof or slice-playtest filmstrips/renders attached to this utterance for inline Writers' Room UI display.",
    )
    artifact_emitted: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured artifact metadata (e.g. type='filmstrip' or 'mockup', title, file_path, notes) for inline UI card rendering.",
    )
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GameSliceSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A playable vertical slice of the game strategized by the Council.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Strategic rationale for how this slice finds and proves genuine fun—delivering a cohesive, skill/strategy-driven playable loop with 100% honest mechanics, real pushback/scarcity, and zero decorative/self-playing stubs.",
    )
    slice_id: str = Field(..., description="Unique snake_case identifier for this slice (e.g. 'slice_0_find_the_fun').")
    slice_index: int = Field(
        default=0,
        description="0-based iteration order (0 = foundational 'Find the Fun' playable vertical slice delivering 2-3 deeply interactive, 100%-real verbs/entities with real scarcity, meaningful decisions, clear goal framing, cohesive art/audio, and zero dead/fake props; 1+ = focused progression slices layering deeper mechanics, tiers, and content onto a proven core loop).",
    )
    title: str = Field(..., description="Concise title for this playable slice.")
    slice_objective: str = Field(
        ...,
        description="What this slice achieves as a tight, cohesive, replayable game experience where player skill/strategy decides the outcome, careless or idle play fails, and every visible tool/entity/HUD element is 100% functional (NEVER a self-completing loop or a bloated grab-bag of half-wired stubs).",
    )
    player_acclimation_and_mechanics_focus: str = Field(
        default="",
        description="How the opening 60 seconds make the player's goal, controls, and causal rules clear on screen while immediately demanding real player judgment, positioning, timing, or resource trade-offs.",
    )
    expansion_from_prior_slice: str = Field(
        default="",
        description="How this slice realizes the core loop (Slice 0) or expands mechanics, challenge, and progression depth onto the verified prior slice (Slice 1+).",
    )
    scope_boundaries: List[str] = Field(
        default_factory=list,
        description="Bounded, cohesive set of playable systems, verbs, world rules, and goals included in THIS slice (explicitly deferring secondary tiers/tools to Slice 1+ rather than shipping disconnected or cosmetic-only placeholders).",
    )
    status: Literal["planned", "active", "completed"] = Field(
        default="planned",
        description="Decision/lifecycle status (strictly last per Rule 4): 'planned' | 'active' | 'completed'.",
    )


class GenrePillarSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A foundational, evolvable genre-craft pillar synthesized by ContextCuratorAgent and evolved by the Council,
    explaining why a core facet of the target genre is intrinsically fun and how simulation mechanics must honor it.
    """
    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: First-principles reasoning examining what intrinsic human cognitive, spatial, "
            "kinetic, or strategic satisfaction this pillar delivers and what breaks it."
        ),
    )
    pillar_id: str = Field(
        ...,
        description="Stable snake_case identifier for this pillar so the Council can evolve it across slices.",
    )
    title: str = Field(
        ...,
        description="Crisp, evocative pillar title.",
    )
    why_it_creates_fun: str = Field(
        ...,
        description="Why human players find this pillar genuinely rewarding, engaging, and replayable.",
    )
    systemic_mechanics_expression: str = Field(
        ...,
        description=(
            "How world rules, active player verbs, scarcity/constraints, spatial precision, and causal feedback "
            "must interlock in real simulation rather than cosmetic surface dressing."
        ),
    )
    anti_pattern_to_avoid: str = Field(
        ...,
        description=(
            "Concrete implementation or design failure mode that accidentally automates away, fakes, obscures, "
            "or trivializes this pillar."
        ),
    )


class GenreCraftFoundation(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Living genre-craft and player-fun foundation generated by ContextCuratorAgent to ground the Co-Producer,
    seed the start of the Living Game Bible, and evolve alongside Council deliberations and slices.
    """
    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: First-principles deconstruction of the target genre/subgenre's psychology of fun, "
            "tactile agency, progression arc, spatial legibility, and classic design blind spots."
        ),
    )
    genre_and_subgenre_archetype: str = Field(
        ...,
        description="Clear characterization of the genre, subgenre, or hybrid design archetype.",
    )
    core_fun_thesis: str = Field(
        ...,
        description=(
            "First-principles thesis explaining what makes this genre genuinely fun for a human player: "
            "the signature cycle of curiosity, friction/scarcity, active decision-making, mastery/optimization, and payoff."
        ),
    )
    bootstrap_to_mastery_arc: str = Field(
        ...,
        description=(
            "How `t=0` begins unsolved with scarce, tactile manual player agency and active pushback, and how mid/late "
            "capabilities are earned by the player without ever turning the game into a self-completing spectator loop."
        ),
    )
    essential_spatial_and_ux_contracts: List[str] = Field(
        default_factory=list,
        description=(
            "Non-negotiable input-to-world precision, visual silhouette differentiation, causal readability, and HUD/onboarding "
            "clarity contracts required for this genre to feel intuitive and responsive."
        ),
    )
    genre_killing_anti_patterns: List[str] = Field(
        default_factory=list,
        description=(
            "Classic design and engineering blind spots that ruin games in this genre (such as free starting handouts, "
            "imprecise input mapping, zero strategic trade-offs, illegible visual noise, or self-playing loops)."
        ),
    )
    foundational_pillars: List[GenrePillarSpec] = Field(
        default_factory=list,
        description="3 to 5 concrete, evolvable genre-craft pillars that seed and evolve inside the Living Game Bible.",
    )


class WritersRoomTurnProposal(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Structured output produced when the selected specialist speaks and updates the Living Game Bible.
    """
    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: Skeptical first-principles deliberation examining blind spots, unverified assumptions, "
            "or scope bloat in peer proposals, while treating any Executive Producer steering with full respect and catering "
            "directly to it within realistic active-slice scope."
        ),
    )
    spoken_dialogue: str = Field(
        ...,
        description=(
            "Direct, unvarnished statement to the Writers' Room—NEVER start with 'I completely agree with...', 'I echo...', "
            "peer roll-call praise, or cheerleading adjectives. Jump straight into concrete domain pushback, risk diagnosis, "
            "or net-new design substance (while treating Executive Producer direction with complete respect and catering to it)."
        ),
    )
    addressed_to: List[str] = Field(default_factory=list)
    updated_game_title: Optional[str] = Field(default=None)
    updated_elevator_pitch: Optional[str] = Field(default=None)
    updated_core_fantasy: Optional[str] = Field(default=None)
    updated_first_60_seconds_hook: Optional[str] = Field(default=None)
    updated_diegetic_ui_philosophy: Optional[str] = Field(default=None)
    updated_camera_and_spatial_paradigm: Optional[str] = Field(default=None)
    concept_evolution_milestone: Optional[ConceptEvolutionEntry] = Field(default=None)
    added_or_updated_storyboard_beats: List[ExperienceStoryboardBeat] = Field(default_factory=list)
    added_or_updated_characters: List[CharacterSheet] = Field(default_factory=list)
    added_or_updated_cutscenes: List[CutsceneScript] = Field(default_factory=list)
    added_or_updated_dialogue_trees: List[DialogueTreeSpec] = Field(default_factory=list)
    added_or_updated_barks: List[DialogueBarkSpec] = Field(default_factory=list)
    mockup_to_generate: Optional[VisualMockupArtifact] = Field(default=None)
    updated_genre_craft_foundation: Optional[GenreCraftFoundation] = Field(
        default=None,
        description=(
            "Optional evolved GenreCraftFoundation refining the game's genre-craft pillars, core fun thesis, "
            "bootstrap-to-mastery arc, UX precision contracts, or anti-patterns as the concept and slices mature."
        ),
    )
    gameplay_pillars_additions: List[str] = Field(default_factory=list)
    added_or_updated_engineering_specs: List[MechanicEngineeringSpec] = Field(default_factory=list)
    visual_style_guide_update: Optional[str] = Field(default=None)
    updated_color_palette_hex: List[str] = Field(default_factory=list)
    added_or_updated_scene_layers: List[VisualSceneLayerSpec] = Field(default_factory=list)
    audio_direction_update: Optional[str] = Field(default=None)
    updated_level_generation_strategy: Optional[LevelGenerationStrategy] = Field(default=None)
    updated_core_loop: Optional[CoreLoopSpec] = Field(default=None)
    updated_engagement_plan: Optional[EngagementPlan] = Field(default=None)
    updated_onboarding_plan: Optional[OnboardingPlan] = Field(default=None)
    updated_goal_hierarchy: Optional[GoalHierarchy] = Field(default=None)
    updated_canonical_telemetry_keys: List[str] = Field(
        default_factory=list,
        description=(
            "Concise canonical snake_case telemetry keys exposed by `get_state_snapshot()` across the active slice "
            "(e.g. primary resource counters, active entity/placement counts, and current objective progress) so "
            "`swe_engineer` and `qa_test_engineer` share one authoritative state vocabulary instead of inventing duplicate aliases."
        ),
    )
    added_or_updated_slices: List[GameSliceSpec] = Field(
        default_factory=list,
        description=(
            "Playable game slices strategized for this game. Slice 0 MUST 'Find the Fun' first with one tight, cohesive, "
            "replayable core loop (2-3 deeply interactive, 100%-real verbs/entities, real pushback/scarcity, clear goal framing, "
            "and zero decorative/self-playing placeholders), while Slice 1+ layers deeper tiers, mechanics, and progression onto a proven foundation."
        ),
    )
    backlog_additions: List[WorkItemProposal] = Field(default_factory=list, description="High-level requirements for the active slice (tasks can dynamically spawn their own asset/animation/sub-mechanic tasks as needed).")
    backlog_refinements: List[WorkItemRefinement] = Field(
        default_factory=list,
        description="Pre-execution Council input on existing unstarted tasks on the Production Board: enrich a task's spec with domain direction/contracts (`action='enrich'`), wire dependencies (`add_depends_on`), or split an oversized task into smaller cohesive tasks (`action='split'`).",
    )
    memory_query: str = Field(
        default="",
        description="Optional semantic search query against long-term game and agent memory banks before finalizing your turn (leave '' when current context suffices).",
    )
    memory_note: str = Field(
        default="",
        description="DEFAULT TO '' — LEAVE EMPTY ON ROUTINE TURNS. Never duplicate Game Bible updates, task specs, or ordinary dialogue points here. Populate ONLY when deferring a multi-slice strategic goal beyond the active slice or recording a non-obvious cross-slice constraint (format '<bank>:<kind>: <title> | <content>').",
    )
    route_to_council_role: str = Field(
        default="",
        description=(
            "Optional dynamic Council peer handoff: if an issue, steering request, or design decision belongs in another "
            "Council specialist's domain (or needs immediate follow-up from a colleague), set `<RoleName>` or "
            "`<RoleName>: <concise handoff reason>` (e.g. 'VisualArtDirector: inspect active sprite rig', "
            "'GameplaySystemsEngineer', 'AudioAtmosphereDesigner', 'UXExperienceDesigner', 'GameplayDesigner', "
            "'LevelAndProgressionDesigner', 'StoryWorldArchitect', 'ScriptwriterAgent', 'CharacterDesigner') "
            "to route the floor directly to them next."
        ),
    )
    routing_handoff_reason: str = Field(
        default="",
        description="When `route_to_council_role` is non-empty, concise explanation of why you are routing to that colleague and what they should address.",
    )

    _ROLE_SCHEMA_CACHE: ClassVar[Dict[str, Type["BaseEpistemicModel"]]] = {}
    _SHARED_TURN_FIELDS: ClassVar[Tuple[str, ...]] = (
        "deliberation",
        "spoken_dialogue",
        "addressed_to",
        "updated_game_title",
        "updated_elevator_pitch",
        "updated_core_fantasy",
        "concept_evolution_milestone",
        "added_or_updated_slices",
        "backlog_additions",
        "backlog_refinements",
        "memory_query",
        "memory_note",
        "route_to_council_role",
    )
    _ROLE_TURN_FIELDS: ClassVar[Dict[str, Tuple[str, ...]]] = {
        "CoProducer": (
            "updated_first_60_seconds_hook",
            "updated_diegetic_ui_philosophy",
            "updated_camera_and_spatial_paradigm",
            "updated_genre_craft_foundation",
            "gameplay_pillars_additions",
            "visual_style_guide_update",
            "audio_direction_update",
            "added_or_updated_storyboard_beats",
            "mockup_to_generate",
        ),
        "CreativeDirector": (
            "updated_first_60_seconds_hook",
            "updated_diegetic_ui_philosophy",
            "updated_camera_and_spatial_paradigm",
            "updated_genre_craft_foundation",
            "gameplay_pillars_additions",
            "visual_style_guide_update",
            "audio_direction_update",
            "added_or_updated_storyboard_beats",
            "mockup_to_generate",
        ),
        "StoryWorldArchitect": (
            "added_or_updated_cutscenes",
            "added_or_updated_characters",
            "added_or_updated_storyboard_beats",
        ),
        "ScriptwriterAgent": (
            "added_or_updated_cutscenes",
            "added_or_updated_dialogue_trees",
            "added_or_updated_barks",
            "added_or_updated_characters",
        ),
        "CharacterDesigner": (
            "added_or_updated_characters",
            "added_or_updated_dialogue_trees",
            "added_or_updated_barks",
        ),
        "GameplayDesigner": (
            "updated_genre_craft_foundation",
            "gameplay_pillars_additions",
            "updated_core_loop",
            "updated_engagement_plan",
            "updated_first_60_seconds_hook",
        ),
        "LevelAndProgressionDesigner": (
            "updated_genre_craft_foundation",
            "updated_level_generation_strategy",
            "updated_goal_hierarchy",
        ),
        "GameplaySystemsEngineer": (
            "updated_genre_craft_foundation",
            "added_or_updated_engineering_specs",
            "updated_camera_and_spatial_paradigm",
            "gameplay_pillars_additions",
        ),
        "VisualArtDirector": (
            "visual_style_guide_update",
            "updated_camera_and_spatial_paradigm",
            "updated_color_palette_hex",
            "added_or_updated_scene_layers",
            "mockup_to_generate",
        ),
        "UXExperienceDesigner": (
            "updated_genre_craft_foundation",
            "updated_first_60_seconds_hook",
            "updated_diegetic_ui_philosophy",
            "updated_onboarding_plan",
            "added_or_updated_storyboard_beats",
        ),
        "AudioAtmosphereDesigner": (
            "audio_direction_update",
            "added_or_updated_barks",
        ),
    }

    @classmethod
    def for_role(cls, role: Optional[str] = None) -> Type["BaseEpistemicModel"]:
        """Returns a role-scoped Pydantic schema class (named `WritersRoomTurnProposal`) containing the shared
        Writers' Room turn fields plus the domain fields owned by `role`, keeping `$defs` and total properties
        well within Gemini's structured-output schema state limits."""
        from pydantic import create_model

        key = (role or "CoProducer").strip() or "CoProducer"
        cached = cls._ROLE_SCHEMA_CACHE.get(key)
        if cached is not None:
            return cached
        role_specific = cls._ROLE_TURN_FIELDS.get(key, cls._ROLE_TURN_FIELDS["CoProducer"])
        keep_fields = list(dict.fromkeys(cls._SHARED_TURN_FIELDS + role_specific))
        field_defs: Dict[str, Any] = {}
        for fname in keep_fields:
            if fname in cls.model_fields:
                finfo = cls.model_fields[fname]
                field_defs[fname] = (finfo.annotation, finfo)
        scoped_cls = create_model("WritersRoomTurnProposal", __base__=BaseEpistemicModel, **field_defs)
        scoped_cls.model_rebuild(_types_namespace=globals())
        cls._ROLE_SCHEMA_CACHE[key] = scoped_cls
        return scoped_cls


class DirectorConvergenceCheck(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Evaluated by the CoProducer (Council Chair) to decide whether the Writers' Room should keep brainstorming
    or if the Living Game Bible is cohesive, rich, and ready to lock.
    """
    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: Director's skeptical yet pragmatic evaluation of whether the experience, UX, art mockups, "
            "engineering blueprint, and Executive Producer requests cohere into a genuinely fun, buildable game without scope bloat."
        ),
    )
    missing_or_weak_elements: List[str] = Field(
        default_factory=list,
        description="Specific gaps the room still needs to resolve for THIS active slice if not ready.",
    )
    director_steering_prompt: str = Field(
        default="",
        description="If not ready, the Director's candid next challenge or question to the room (zero peer flattery).",
    )
    experience_cohesion_score: float = Field(
        ...,
        description="Cohesion and polish readiness score in [0.0, 1.0].",
    )
    is_ready_to_lock_bible: bool = Field(
        ...,
        description=(
            "True when the Game Bible has been pressure-tested across domains, faithfully honors any Executive Producer direction, "
            "and defines a cohesive, buildable slice scope without trying to cram every future idea into one pass (strictly last per Rule 4)."
        ),
    )


class CreativeProvocation(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    One divergent creative provocation assigned to a single specialist.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Why this provocation opens a distinct creative direction.")
    specialist_role: str = Field(..., description="Exact specialist role this provocation is for.")
    provocation: str = Field(..., description="An open-ended creative prompt, constraint or question, unrelated to any existing game or genre label.")


class CreativeProvocationSet(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4)."""
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: How these provocations span orthogonal creative axes and differ from prior concepts.")
    provocations: List[CreativeProvocation] = Field(default_factory=list)


class ConceptFragment(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A single independent creative fragment from one specialist's domain; NOT a complete game idea.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Why this fragment is surprising and worth building a game around.")
    fragment_id: str = Field(..., description="Unique identifier within the room.")
    contributor_role: str = Field(default="", description="Specialist role that authored the fragment.")
    fragment_kind: str = Field(..., description="Free-form kind such as an interaction verb/system, a world tension, an emotional experience, a constraint, or a sensory signature.")
    fragment: str = Field(..., description="The fragment itself, concrete and vivid.")


class ConceptSeedProposal(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4)."""
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: How the assigned provocation was interpreted from this specialist's domain.")
    fragments: List[ConceptFragment] = Field(default_factory=list)


class ConceptCandidate(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A candidate premise formed by colliding fragments from different specialists.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: How the colliding fragments produce something none of them implied alone.")
    candidate_id: str = Field(..., description="Unique candidate identifier.")
    working_title: str = Field(..., description="Evocative provisional title.")
    premise: str = Field(..., description="Provisional premise (a starting point that the room will evolve, not a finished design).")
    source_fragment_ids: List[str] = Field(default_factory=list, description="Fragments combined; must come from at least two different contributors.")
    core_tension: str = Field(default="", description="The central tension or question that makes the premise generative.")


class ConceptCollisionProposal(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4)."""
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Strategy for colliding fragments into distinct candidates.")
    candidates: List[ConceptCandidate] = Field(default_factory=list)


class NoveltyJudgement(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Judges whether a candidate concept is genuinely distinct from previously produced concepts.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Element-by-element comparison of the candidate against each prior concept (core interaction, setting, tone, structure).")
    candidate_id: str = Field(..., description="Candidate being judged.")
    overlapping_elements: List[str] = Field(default_factory=list, description="Core elements shared with prior concepts.")
    is_distinct_from_prior_concepts: bool = Field(..., description="Decision (strictly last): True only if the candidate's core interaction AND setting/tone are meaningfully different from every prior concept.")


class ConceptSelection(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4)."""
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Director's comparison of the candidates by potential for evolution, freshness and buildability.")
    opening_dialogue: str = Field(..., description="What the Creative Director says to open the room with the chosen provisional premise, inviting specialists to challenge and reshape it.")
    selected_candidate_id: str = Field(..., description="Decision (strictly last): the chosen candidate_id.")


class LevelArchetypeSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    One recurring kind of level / region / area the game is built from.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Why this archetype exists and what it adds to the experience.")
    archetype_id: str = Field(..., description="Unique snake_case identifier.")
    purpose: str = Field(..., description="Its role in the player's journey (teach, test, reward, surprise, rest, climax...).")
    visual_identity: str = Field(..., description="What it looks, sounds and feels like, and how it differs from other archetypes.")
    signature_challenge: str = Field(..., description="The characteristic challenge or activity found here.")
    building_blocks: List[str] = Field(default_factory=list, description="Reusable elements it is assembled from (terrain features, hazards, mechanisms, set pieces).")
    introduced_at_progression_order: int = Field(default=0, description="0 = encountered first; larger values appear later.")


class LevelGenerationStrategy(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    How the game's levels/world are produced and what they look like, invented specifically for this game.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Why this bespoke world-building strategy, structure, and scale suit this specific game's loop, scope and replayability.")
    generation_approach: str = Field(..., description="Bespoke strategy for how this game's world/levels are constructed and populated (devised specifically for this game rather than chosen from a fixed menu).")
    world_structure: str = Field(..., description="How the world is organised and scaled for this specific game, how its spaces connect or evolve, and why that scale serves the core loop.")
    runtime_construction_plan: str = Field(
        default="",
        description="Concrete technical plan for how scenes.gd and mechanics.gd construct, generate, or stream the world at runtime.",
    )
    level_archetypes: List[LevelArchetypeSpec] = Field(default_factory=list, description="The kinds of spaces, stages, or biomes the game is built from.")
    generation_rules: List[str] = Field(default_factory=list, description="Concrete constraints, grammar, or invariants: guaranteed elements, placement/connectivity rules, solvability and fairness guarantees.")
    difficulty_and_pacing_curve: str = Field(..., description="How challenge and intensity ramp and relax across the experience.")
    variety_and_replayability: str = Field(default="", description="What keeps the world from feeling repetitive across a session.")


class CoreLoopSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    The gameplay loop at three time scales, anchored by an explicit Fun & Tension Pact.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Why this loop is genuinely fun, what skill/strategy tension drives it, and how its scales reinforce one another without ever playing itself.")
    player_verbs: List[str] = Field(default_factory=list, description="The concrete actions the player actively performs.")
    core_decision_or_skill_test: str = Field(
        default="",
        description="The core perceptual, spatial, timing, or strategic trade-off the player must actively judge—explaining why thoughtful/skilled play succeeds while random, careless, or unthinking clicking fails.",
    )
    pushback_and_scarcity_model: str = Field(
        default="",
        description="What active world resistance, constraint, hazard, or resource scarcity pushes back on the player so doing nothing (hands-off idle state) or repeating one mindless action stalls or loses.",
    )
    earned_vs_given_progression: str = Field(
        default="",
        description="Conservation of Challenge rule: why the t=0 starting state seeds ONLY active manual/tactical play (zero passive self-collection or lavish free budgets at spawn), and how any efficiency, multiplier, or automation must be earned by the player.",
    )
    player_agency_and_optimization_depth: str = Field(
        default="",
        description="How the player exercises open-ended agency, creative construction/expression, and iterative optimization tailored to the target player's Steam DNA (e.g. why there is no single scripted 'correct' solution, how the player builds/experiments/optimizes their own approach, and what live efficiency/performance feedback rewards mastery).",
    )
    moment_to_moment_loop: str = Field(..., description="Seconds-scale loop: act -> world response -> feedback -> next decision.")
    minute_to_minute_loop: str = Field(..., description="Minutes-scale loop: objectives, encounters, resource flow.")
    session_loop: str = Field(..., description="Whole-session loop: how a play session starts, builds and ends, and why the player returns.")
    feedback_and_juice: str = Field(default="", description="How the game communicates success, failure and progress through visuals, audio and motion.")
    reward_cadence: str = Field(default="", description="When and how the player is rewarded.")
    failure_and_recovery: str = Field(default="", description="What careless/poor play costs and how the player recovers cleanly via skill/strategy (never via free passive bailouts).")


class EngagementPlan(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    How the player is kept entertained over time.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: What will make a player want to keep playing, and the evidence for it in the design.")
    entertainment_pillars: List[str] = Field(default_factory=list, description="The distinct sources of enjoyment (e.g. expression, mastery, discovery, tension and release).")
    variety_injectors: List[str] = Field(default_factory=list, description="Mechanisms that change the experience (new abilities, twists, set pieces, events).")
    pacing_and_intensity_curve: str = Field(..., description="Rise and fall of intensity across a session.")
    surprise_and_discovery: str = Field(default="", description="What the player discovers and how surprises are staged.")
    mastery_and_expression: str = Field(default="", description="How skill growth and personal style are supported and rewarded.")
    emotional_arc: str = Field(default="", description="The feelings the experience moves through.")


class OnboardingStep(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4)."""
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Why this is the right next thing to teach or reveal.")
    step_index: int = Field(..., description="Order within the introduction (0 first).")
    concept_introduced: str = Field(..., description="The single mechanic, rule or world fact introduced here.")
    diegetic_method: str = Field(..., description="How this concept is taught or communicated to the player (through level design, UI/HUD cues, prompts, or world feedback suited to this game).")
    player_proof_of_learning: str = Field(default="", description="How the game can tell the player understood.")
    world_reveal: str = Field(default="", description="What the player learns about the world at this step.")


class OnboardingPlan(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    How the player is introduced to the world and the mechanics.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Empathetic evaluation from the perspective of a brand-new player who has zero prior context: what might confuse them at first glance, and how this introduction builds clear purpose ('why am I doing this?'), causal understanding ('how do my actions affect the world?'), and confidence.",
    )
    world_introduction: str = Field(..., description="How the player first meets the world, their role and the stakes.")
    purpose_and_stakes_framing: str = Field(
        default="",
        description="Strategy for how THIS game makes the player's goal, motivation, and stakes immediately understandable on screen without relying on external knowledge.",
    )
    causal_chain_and_feedback_strategy: str = Field(
        default="",
        description="Strategy for how the player visually and physically perceives the causal link between their controls, the world's response, and their objective—including how the game responds when a naive player tries an action whose precondition is not yet met.",
    )
    steps: List[OnboardingStep] = Field(default_factory=list, description="Ordered teaching steps, each introducing one concept.")
    confusion_recovery: str = Field(default="", description="How a stuck or lost player is gently helped.")


class GoalSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4)."""
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Why the player will care about this goal.")
    goal_id: str = Field(..., description="Unique identifier.")
    horizon: str = Field(..., description="Time scale or cadence of the goal (devised for this game's loop, e.g. immediate, short_term, per_round, session, mastery, long_term).")
    description: str = Field(..., description="What the player is trying to achieve.")
    motivation: str = Field(default="", description="Why it matters (emotional / mechanical stakes).")
    progress_feedback: str = Field(default="", description="How the player sees progress toward it.")
    reward: str = Field(default="", description="What achieving it gives the player.")


class GoalHierarchy(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    The player's short-term, mid-term and long-term goals and how they connect.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: How the goals nest so that short-term play always serves the long-term arc.")
    goals: List[GoalSpec] = Field(default_factory=list, description="Goals across all horizons (immediate through endgame).")
    progression_structure: str = Field(default="", description="How the player advances (unlocks, abilities, story, mastery).")
    end_state: str = Field(default="", description="What completion, or an endless-play equilibrium, looks like.")


def validate_game_design(bible: "GameBibleSpec", *, enforce_fun_pact: bool = False) -> List[str]:
    """Structural completeness of the game-design sections. Returns issues (empty = complete)."""
    issues: List[str] = []
    lg, loop, eng, onb, gh = (
        bible.level_generation_strategy, bible.core_loop, bible.engagement_plan,
        bible.onboarding_plan, bible.goal_hierarchy,
    )
    if lg is None:
        issues.append("level_generation_strategy is missing.")
    else:
        if not lg.level_archetypes:
            issues.append("level_generation_strategy needs at least one level archetype.")
        if not lg.generation_rules:
            issues.append("level_generation_strategy needs generation_rules.")
    if loop is None:
        issues.append("core_loop is missing.")
    else:
        if not loop.player_verbs:
            issues.append("core_loop needs player_verbs.")
        if enforce_fun_pact:
            if not (loop.core_decision_or_skill_test or "").strip():
                issues.append("core_loop needs core_decision_or_skill_test (Fun & Tension Pact: why player judgment/skill beats random clicking).")
            if not (loop.pushback_and_scarcity_model or "").strip():
                issues.append("core_loop needs pushback_and_scarcity_model (Fun & Tension Pact: what world resistance/scarcity makes idle or careless play fail).")
            if not (loop.earned_vs_given_progression or "").strip():
                issues.append("core_loop needs earned_vs_given_progression (Fun & Tension Pact: why t=0 seeds active play only and power/automation must be earned).")
    if eng is None:
        issues.append("engagement_plan is missing.")
    elif not eng.entertainment_pillars:
        issues.append("engagement_plan needs entertainment_pillars.")
    if onb is None:
        issues.append("onboarding_plan is missing.")
    elif not onb.steps:
        issues.append("onboarding_plan needs at least one step.")
    if gh is None:
        issues.append("goal_hierarchy is missing.")
    elif not gh.goals:
        issues.append("goal_hierarchy needs at least one goal.")
    return issues


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
    target_player_dna_summary: str = Field(
        default="",
        description="Persisted Steam Player DNA profile and player-centric design alignment for the specific human player this game is being built for.",
    )
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
        description="UI, HUD, and player feedback philosophy strategized for this game (how state, controls, objectives, and feedback are presented via HUD elements, labels, counters, overlays, or in-world cues)."
    )
    camera_and_spatial_paradigm: str = Field(
        default="",
        description="Locked 2D camera perspective and spatial movement model agreed upon by the Writers' Room."
    )
    canonical_telemetry_keys: List[str] = Field(
        default_factory=list,
        description=(
            "Authoritative snake_case telemetry keys returned by `get_state_snapshot()` across the cartridge so "
            "SWE and QA reference one shared vocabulary for preconditions, event-driven captures, and state assertions."
        ),
    )
    color_palette_hex: List[str] = Field(
        default_factory=list,
        description="Hex color palette (#RRGGBB) governing sky, terrain, architecture, energy accents, and UI/HUD."
    )
    scene_layers: List[VisualSceneLayerSpec] = Field(
        default_factory=list,
        description="Ordered visual depth layers decomposing the 16:9 mockup into constructible Godot 4 scene geometry."
    )
    genre_craft_foundation: Optional[GenreCraftFoundation] = Field(
        default=None,
        description=(
            "Foundational genre-craft and player-fun blueprint curated by ContextCuratorAgent at Game Bible genesis "
            "and evolved across Council turns and slice transitions."
        ),
    )
    gameplay_pillars: List[str] = Field(default_factory=list)
    technical_architecture_blueprint: List[MechanicEngineeringSpec] = Field(
        default_factory=list,
        description="Concrete Godot 4 engineering strategies authored by GameplaySystemsEngineer mapping gameplay pillars and visual mockups to nodes, colliders, state machines, required assets, and test invariants."
    )
    visual_style_guide: str = Field(default="")
    audio_direction_guide: str = Field(default="")
    concept_evolution: List[ConceptEvolutionEntry] = Field(default_factory=list)
    level_generation_strategy: Optional[LevelGenerationStrategy] = Field(default=None, description="How levels are generated and what they look like.")
    core_loop: Optional[CoreLoopSpec] = Field(default=None, description="Moment-to-moment, minute-to-minute and session loops.")
    engagement_plan: Optional[EngagementPlan] = Field(default=None, description="How the player stays entertained.")
    onboarding_plan: Optional[OnboardingPlan] = Field(default=None, description="How the player is introduced to the world and mechanics.")
    goal_hierarchy: Optional[GoalHierarchy] = Field(default=None, description="Short-, mid- and long-term goals.")
    slice_roadmap: List[GameSliceSpec] = Field(
        default_factory=list,
        description="Playable vertical slices strategized specifically for this game. Slice 0 must deliver the complete playable game loop (moment-to-moment, minute-to-minute, and session loops, player toolkit/agency, and escalating goals), never a tutorial-only stub.",
    )
    active_slice_index: int = Field(
        default=0,
        description="Index in slice_roadmap currently in active production.",
    )
    concept_fragments: List[ConceptFragment] = Field(default_factory=list, description="Independent divergent fragments contributed before any concept was chosen.")
    concept_candidates: List[ConceptCandidate] = Field(default_factory=list, description="Candidate premises formed by colliding fragments (the selected one seeds the room).")
    writers_room_transcript: List[WritersRoomUtterance] = Field(default_factory=list)
    experience_storyboard: List[ExperienceStoryboardBeat] = Field(default_factory=list)
    visual_mockups: List[VisualMockupArtifact] = Field(default_factory=list)
    world_spec: Optional[GenesisWorldSpec] = Field(default=None)
    narrative_bible: Optional[NarrativeBible] = Field(default=None)
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def get_active_slice(self) -> Optional[GameSliceSpec]:
        """Returns the currently active GameSliceSpec from slice_roadmap, if any."""
        if not self.slice_roadmap:
            return None
        for s in self.slice_roadmap:
            if s.slice_index == self.active_slice_index:
                return s
        for s in self.slice_roadmap:
            if s.status == "active":
                return s
        idx = max(0, min(self.active_slice_index, len(self.slice_roadmap) - 1))
        return self.slice_roadmap[idx]

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class AssetVisualValidationVerdict(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Tier 1 per-asset visual inspection verdict evaluating an isolated rendered asset plate
    (dual-checkerboard alpha cutout plate for sprites/portraits, or 3x3 tiled grid plate for tiles).
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Detailed visual inspection of the rendered asset plate (alpha cutout cleanliness, subject completeness, 3x3 tile edge continuity, and art direction match)."
    )
    asset_id: str = Field(..., description="Identifier of the inspected asset.")
    alpha_cleanliness_passed: bool = Field(
        default=True,
        description="True if sprites/portraits have clean alpha cutouts with zero leftover background boxes, outer picture frames, or magenta halos."
    )
    interior_hole_cleanliness_passed: bool = Field(
        default=True,
        description="True if enclosed interior pockets (between legs, under arms, inside arches/spokes) have clean transparency rather than trapped background patches."
    )
    subject_framing_and_completeness_passed: bool = Field(
        default=True,
        description="True if the subject is complete (e.g., full-body actor rather than cropped bust for sprites, zero baked-in ground pedestals or surfing water wakes under feet)."
    )
    tile_seam_continuity_passed: bool = Field(
        default=True,
        description="True if 3x3 tiled surface plates repeat seamlessly without harsh grid borders, Rorschach/kaleidoscope mirror symmetry, white/magenta corner triangles, or baked-in perspective horizons."
    )
    mirror_symmetry_score: float = Field(
        default=0.0,
        description="Maximum bilateral horizontal/vertical mirror correlation in [0.0, 1.0]. Tiles above 0.65 indicate artificial Rorschach mirror-blending defects."
    )
    style_coherence_score: int = Field(
        default=8,
        ge=1,
        le=10,
        description="Coherence score (1..10) against the Game Bible style guide and reference mockup."
    )
    visual_defects: List[str] = Field(
        default_factory=list,
        description="Specific visual defects observed on the rendered inspection plate."
    )
    defective_cell_indices: List[int] = Field(
        default_factory=list,
        description=(
            "For multi-cell grid sheets (grid_cols * grid_rows > 1), 0-based row-major cell indices "
            "(idx = row * grid_cols + col) that exhibit visual, framing, or alpha defects so clean cells "
            "can be preserved during surgical regeneration. Leave empty for 1x1 assets or global sheet defects."
        ),
    )
    prompt_refinement_directive: str = Field(
        default="",
        description="Targeted prompt or parameter refinement directive to fix the asset on regeneration if rejected."
    )
    accepted: bool = Field(
        ...,
        description="Decision token (strictly last per Rule 4): True ONLY if the rendered asset passes all visual quality and alpha/seam checks."
    )


class AnimationValidationVerdict(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Tier 2 per-animation contact-sheet and in-engine multi-frame strip validation verdict.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Frame-by-frame visual analysis of the rendered animation strip / contact sheet across timestamps t0..tk."
    )
    target_id: str = Field(..., description="Asset ID or atomic unit ID of the inspected animation.")
    motion_progression_verified: bool = Field(
        default=True,
        description="True if frames exhibit clear, purposeful keyframe pose or articulated-part progression across t0..tk rather than static duplicate or micro-shifted frames."
    )
    optical_flow_magnitude_mean: float = Field(
        default=0.0,
        description="Mean dense optical flow magnitude across consecutive frames computed via OpenCV Farneback flow."
    )
    silhouette_and_pivot_integrity_passed: bool = Field(
        default=True,
        description="True if limbs/parts remain coherently attached without wild clipping, detached floating pieces, or broken proportions."
    )
    baseline_grounding_passed: bool = Field(
        default=True,
        description="True if grounded states (idle/move) maintain consistent foot/base contact along the baseline guide without erratic vertical jitter."
    )
    animation_defects: List[str] = Field(
        default_factory=list,
        description="Specific frame-level or motion defects observed in the rendered animation strip."
    )
    remediation_directive: str = Field(
        default="",
        description="Specific instructions to repair the sprite sheet articulation or in-engine GDScript animation logic."
    )
    accepted: bool = Field(
        ...,
        description="Decision token (strictly last per Rule 4): True ONLY if the rendered animation strip demonstrates clean, readable, non-static motion."
    )



class SceneUnitVisualVerdict(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Tier 3 & Tier 4 visual & spatial-collider alignment verdict evaluating rendered scene components
    (clean viewport + physics collision overlay) and multi-state level progression snapshots.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Side-by-side analysis of the clean Godot render, the live physics collision overlay render, and the concept art."
    )
    unit_id: str = Field(..., description="Atomic unit ID or stage ID being visually verified.")
    visual_to_collider_alignment_passed: bool = Field(
        default=True,
        description="True ONLY if every visual cliff edge, sky backdrop boundary, water/chasm gap, and solid structure in the clean render is accurately bounded by physical StaticBody2D/CollisionPolygon2D shapes in the collision overlay."
    )
    spatial_depth_and_proportion_passed: bool = Field(
        default=True,
        description="True if sprites, tiles, and landmarks are properly scaled and layered without giant unscaled textures or raw programmer-art rectangles."
    )
    state_progression_visibility_passed: bool = Field(
        default=True,
        description="True if active/completed state renders show clear, gratifying visual feedback and world transformation compared to the unsolved t0 render."
    )
    blocking_visual_or_spatial_defects: List[str] = Field(
        default_factory=list,
        description="Concrete visual, perspective, or visual-to-collider misalignment defects."
    )
    remediation_instructions: List[str] = Field(
        default_factory=list,
        description="Surgical GDScript remediation steps to fix the identified visual or collider alignment defects."
    )
    accepted: bool = Field(
        ...,
        description="Decision token (strictly last per Rule 4): True ONLY if visual composition, collider alignment, and state progression all pass."
    )


class AtomicInvariantSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A single falsifiable behavioral or spatial invariant tested by a headless Godot atomic unit test.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining why this invariant is necessary to prove the atomic unit works and cannot be gamed."
    )
    invariant_id: str = Field(..., description="Unique invariant identifier, e.g. 'INV_R1_U1_BASE' or 'INV_R1_U1_BOUNDARY'.")
    invariant_type: Literal[
        "unsolved_baseline",
        "negative_counterfactual",
        "spatial_boundary",
        "positive_causal",
    ] = Field(
        ...,
        description="Category of invariant: 'unsolved_baseline' | 'negative_counterfactual' | 'spatial_boundary' | 'positive_causal'."
    )
    setup_description: str = Field(..., description="Exact node/state initialization for this test scenario.")
    action_sequence: str = Field(..., description="Exact stimulus, method calls, or physics frames applied during the test.")
    expected_assertion: str = Field(..., description="Exact falsifiable assertion condition that must hold.")


class AtomicWorkUnit(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A granular, self-contained implementation and verification unit decomposed from a RequirementTrack.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Architectural rationale for isolating this atomic piece and how it will be verified behaviorally and visually."
    )
    unit_id: str = Field(..., description="Unique atomic unit identifier, e.g. 'R1_U1_terrace_colliders'.")
    parent_track_id: str = Field(default="R1", description="Parent RequirementTrack ID.")
    title: str = Field(..., description="Concise title of this atomic work unit.")
    single_responsibility_scope: str = Field(
        ...,
        description="Exact single responsibility of this unit (one cohesive class, mechanic solver, collider topology, or animated entity)."
    )
    verification_modality: Literal[
        "behavioral_and_scene_overlay",
        "behavioral_and_animation_strip",
        "behavioral_and_level_progression",
        "behavioral_only",
    ] = Field(
        default="behavioral_and_scene_overlay",
        description="Which visual render verification tier accompanies the behavioral Godot unit test."
    )
    target_files: List[str] = Field(
        default_factory=lambda: ["mechanics.gd"],
        description="Cartridge `.gd` files created or updated by this atomic unit."
    )
    target_classes_and_methods: List[str] = Field(
        default_factory=list,
        description="Specific classes and methods targeted by this unit."
    )
    invariants: List[AtomicInvariantSpec] = Field(
        default_factory=list,
        description="Mandatory falsifiable invariants (must include baseline/negative/boundary + positive causal)."
    )


class TrackAtomicDecompositionPlan(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Decomposes a high-level RequirementTrack (or cartridge repair scope) into ordered AtomicWorkUnits.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Engineering deliberation decomposing the track into small, independently verifiable atomic pieces."
    )
    track_id: str = Field(..., description="Identifier of the parent RequirementTrack.")
    atomic_units: List[AtomicWorkUnit] = Field(
        default_factory=list,
        description="Ordered list of atomic work units to build, test, render, and verify one by one."
    )


class AtomicTestSynthesisPayload(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Structured output containing an executable Godot 4 SceneTree unit test script (`tests/test_<unit_id>.gd`)
    that tests all invariants of an AtomicWorkUnit.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Step-by-step design of the headless Godot 4 SceneTree test script verifying unsolved baseline, negative/counterfactual/boundary, and positive causal assertions."
    )
    unit_id: str = Field(..., description="Atomic unit ID tested by this script.")
    test_script_filename: str = Field(
        ...,
        description="Relative path in the cartridge, e.g. 'tests/test_R1_U1.gd'."
    )
    test_gdscript_code: str = Field(
        ...,
        description="Complete valid Godot 4 GDScript code (`extends SceneTree`) that instantiates the target classes/scene, executes each invariant check, prints `[ATOMIC_ASSERT_PASS]` / `[ATOMIC_ASSERT_FAIL]` lines, and calls `quit(0)` or `quit(1)`."
    )


class SubjectRenderTreatment(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    How ONE subject (environment element, character, mechanism, UI, or effect) is rendered.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Why this technique (procedural shader/geometry/particles vs. discrete sprite/rig) suits this subject given the Game Bible's camera, tone and mechanics.")
    subject_id: str = Field(..., description="Identifier of the environment element, character, mechanism, UI, or effect.")
    technique: str = Field(..., description="Bespoke rendering technique devised for this subject (e.g. procedural Polygon2D/Line2D + CanvasItem shader, GPUParticles2D, custom _draw(), articulated rig, or sprite sheet — prefer procedural/shader rendering over repeating tiles for dynamic, fluid, or geometric elements).")
    target_resolution: Tuple[int, int] = Field(..., description="On-screen or source-art pixel resolution for this subject.")
    notes: str = Field(default="", description="Specific implementation guidance (shader uniforms, procedural mesh/curve construction, layer depth, parts breakdown, motion cues, lighting interaction).")


class RenderingStrategySpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Per-game rendering approach decided by the model from the Game Bible. Drives asset decomposition and
    fabrication, SWE rendering code, and critic criteria.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: How the Bible's experience, camera, tone and mechanics imply a modern rendering approach.")
    rendering_paradigm: str = Field(..., description="Overall rendering paradigm devised specifically for this game.")
    renderer: Literal["compatibility", "mobile", "forward_plus"] = Field(..., description="Godot renderer. 'compatibility' is the most portable but has NO WorldEnvironment glow.")
    uses_environment_glow: bool = Field(default=False, description="Whether WorldEnvironment glow/bloom is used. Must be False when renderer is 'compatibility'.")
    pixelated_style_intended: bool = Field(default=False, description="True ONLY if the Game Bible explicitly calls for a pixel-art / low-resolution aesthetic.")
    treatments: List[SubjectRenderTreatment] = Field(default_factory=list, description="Treatment for every significant subject.")
    lighting_and_atmosphere: str = Field(default="", description="Lights, fog, colour grading and atmospheric depth plan.")
    effects_stack: List[str] = Field(default_factory=list, description="Concrete Godot features to use (particles, shaders, light nodes, trails, post-processing).")
    asset_resolution_policy: str = Field(default="", description="How source-art resolution scales with on-screen size and camera zoom.")


WorkItemKind = Literal[
    "visual_asset", "audio_asset", "level_map", "animation", "mechanic_code", "integration", "ui", "narrative", "discovery", "other",
]
WorkItemState = Literal[
    "blocked", "ready", "in_progress", "waiting_for_lock", "awaiting_review", "changes_requested",
    "needs_decision", "approved", "failed", "cancelled",
]


class QASetupCall(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A scenario-setup action executed by the QA test harness at a specific tick (outside production _physics_process)
    to position actors, spawn a hazard, or trigger an interaction precondition for proof capture.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Concise 1-sentence rationale (<= 15 words) for this setup call.")
    at_tick: int = Field(default=1, description="Tick at which to invoke this setup hook (1 <= at_tick <= 900).")
    target_node_or_property: str = Field(
        default="",
        description="Node name/path or root property on main scene (empty string targets main scene root directly).",
    )
    method_name: str = Field(..., description="Method to call on the target node (or 'set_property' to assign a property).")
    args_json: str = Field(default="[]", description="JSON-encoded list of arguments to pass to method_name.")


class QAStateAssertion(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    An observable state assertion evaluated at a specific tick or key capture event during the QA filmstrip run.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Concise 1-sentence rationale (<= 15 words) for this assertion.")
    at_tick: int = Field(default=0, description="Tick at which to check the assertion (or 0 when binding to `at_event_label`).")
    at_event_label: str = Field(
        default="",
        description="Optional `event_label` from `capture_events` whose resolved trigger tick should evaluate this assertion (preferred over guessing a hardcoded `at_tick` when waiting on dynamic travel/physics).",
    )
    snapshot_key: str = Field(..., description="Key in get_state_snapshot() or entity property to inspect.")
    expected_value_or_change: str = Field(..., description="Expected value or transition description (e.g. '== 0', '>= 1', '> 0', '== true').")


class QACaptureEvent(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A key gameplay milestone planned by QA to capture a proof screenshot dynamically when an in-game condition
    or state transition occurs (or at an explicit baseline tick) across multi-second simulation runs.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Concise 1-sentence rationale (<= 18 words) explaining why this gameplay milestone proves a link in the causal loop.",
    )
    event_label: str = Field(
        ...,
        description="Concise snake_case milestone label stamped onto the captured frame header (e.g. 't0_unsolved_baseline', 'player_action_applied', 'world_interaction', 'objective_advanced').",
    )
    when_snapshot_key: str = Field(
        default="",
        description="Snapshot key from `get_state_snapshot()` to watch when `trigger_mode` is 'on_condition' or 'on_change' (leave '' when `trigger_mode == 'at_tick'`).",
    )
    when_operator: str = Field(
        default="",
        description="Comparison operator when `trigger_mode == 'on_condition'` ('==', '!=', '>=', '<=', '>', '<'); leave '' for 'at_tick' or 'on_change'.",
    )
    when_value: str = Field(
        default="",
        description="Target value string, number, or boolean when `trigger_mode == 'on_condition'`.",
    )
    at_tick: int = Field(
        default=2,
        ge=1,
        le=3600,
        description="Exact tick to capture when `trigger_mode == 'at_tick'` (e.g. tick 2 for t=0 baseline), OR earliest tick (`min_tick`) to start watching `when_snapshot_key`.",
    )
    settle_ticks: int = Field(
        default=0,
        ge=0,
        le=60,
        description="Optional post-trigger settle delay (0..60 ticks) after the condition/change fires before snapping the screenshot so animations, VFX, or HUD updates are visibly rendered.",
    )
    trigger_mode: Literal["at_tick", "on_condition", "on_change"] = Field(
        default="at_tick",
        description="Decision (strictly last per Rule 4): 'at_tick' for fixed baseline/checkpoint ticks; 'on_condition' when `when_snapshot_key` satisfies `when_operator` `when_value`; 'on_change' when `when_snapshot_key` changes from its baseline value at `at_tick`.",
    )


class InputSegment(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Actions held down for a range of physics ticks (or triggered reactively when a state condition is met) during a scripted proof run.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Concise 1-sentence rationale (<= 15 words) for this input segment.")
    from_tick: int = Field(..., description="First tick (1-based) the actions are held (or earliest tick to start watching `when_snapshot_key`).")
    to_tick: int = Field(..., description="Last tick (inclusive) the actions are held (or latest tick to watch `when_snapshot_key`).")
    actions: List[str] = Field(default_factory=list, description="Input action names or KEY_* names, chosen only from the game's available actions.")
    mouse_click_pos: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Optional (x, y) screen/viewport coordinate to click at from_tick and release at to_tick (use for HUD/CanvasLayer buttons).",
    )
    mouse_world_pos: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Optional 2D world-space coordinate (wx, wy) to move/click the mouse at during this segment; automatically transformed through the active Camera2D canvas transform at runtime. Preferred over `mouse_click_pos` whenever clicking in-world entities, terrain cells, or placement targets.",
    )
    when_snapshot_key: str = Field(
        default="",
        description="Optional state snapshot key (e.g. from get_state_snapshot()) to watch between from_tick and to_tick; when the condition is met, the input triggers reactively instead of requiring a pre-guessed tick.",
    )
    when_operator: str = Field(
        default="",
        description="Comparison operator when `when_snapshot_key` is set ('==', '!=', '>=', '<=', '>', '<').",
    )
    when_value: str = Field(
        default="",
        description="Target value string or number to compare `when_snapshot_key` against.",
    )
    hold_ticks: int = Field(
        default=5,
        description="When `when_snapshot_key` is set, how many ticks to hold `actions` once the condition triggers.",
    )


class FilmstripPlan(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    How the QA Engineer builds a test scene (`test_scene_gd_content`), renders diagnostic overlays/telemetry
    (`overlays_and_telemetry_strategy`), executes scripted inputs (`input_segments`, `setup_calls`, `state_assertions`),
    summarizes the test scene for the Council (`council_test_scene_summary`), or reports an authentic defect in
    production code (`production_defect_report`) when the game itself is broken or deadlocked.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Why this test scene architecture, diagnostic overlay/telemetry strategy, scripted inputs, and capture ticks best prove the item meets its spec (or why the production code itself is broken/deadlocked).")
    self_question_answer: str = Field(
        default="",
        description="If you asked `self_question` on the previous turn, summarize the exact node paths, method signatures, or asset properties retrieved before finalizing your QA plan.",
    )
    self_question: str = Field(
        default="",
        description="Optional semantic inquiry against the codebase, Game Bible, or asset catalog if you need exact node names, method signatures, or signals before writing `test_scene_gd_content` (leave '' when current context suffices).",
    )
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, summarize what you learned from the retrieved long-term memory entries before finalizing your QA plan.",
    )
    memory_query: str = Field(
        default="",
        description="Optional semantic search query against long-term game/agent memory banks if you want to check prior QA/verification lessons or architectural invariants before finalizing your test scene plan (leave '' when current context suffices).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter when issuing `memory_query` ('', 'all', 'self', 'council', or a canonical role name).",
    )
    tool_search_answer: str = Field(
        default="",
        description="If you issued `tool_search_query` on the previous turn, summarize the relevant QA harness capabilities here.",
    )
    tool_search_query: str = Field(
        default="",
        description="Optional progressive tool discovery query for QA harness capabilities, setup_calls, or telemetry features (leave '' when current context suffices).",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="DEFAULT TO [] — LEAVE EMPTY ON ROUTINE TURNS. Never log standard test runs or single-task test steps. Populate ONLY when discovering a non-obvious harness limitation or verification invariant that future QA/Task Lead turns would otherwise violate (or to update/fulfill an existing memory_id).",
    )
    production_defect_report: str = Field(
        default="",
        description=(
            "Leave empty ('') when executing a valid QA proof capture. Populate ONLY when inspecting the cartridge code "
            "or Pass 1 runtime telemetry proves that the production game code itself is broken, unwired, or deadlocked "
            "under honest player inputs—describing the exact root cause and tick/state evidence so TaskLead delegates "
            "a production fix to `swe_engineer` instead of faking state mutations inside `test_scene_gd_content`."
        ),
    )
    overlays_and_telemetry_strategy: str = Field(
        default="",
        description=(
            "QA Engineer's strategy for what visual diagnostic overlays (e.g. a test-only foreground CanvasLayer/Node2D._draw() "
            "overlay in `test_scene_gd_content` drawing velocity vectors, collider bounds, target zones, camera framing markers, "
            "or state callouts) and/or `get_state_snapshot()` telemetry keys are included in the test scene to prove the requirement."
        ),
    )
    council_test_scene_summary: str = Field(
        default="",
        description=(
            "Concise briefing written for the Council explaining: (1) what the test scene (`target_scene` / `test_scene_gd_content`) "
            "instantiates and stages, (2) what scenario and inputs run across `capture_ticks`, and (3) what QA diagnostic overlays "
            "or telemetry readouts are displayed so the Council knows what it is looking at."
        ),
    )
    target_scene: str = Field(
        default="",
        description="Godot test scene path (e.g. 'res://tests/test_<item_id>.tscn') built by QA to prove this requirement.",
    )
    test_scene_gd_content: str = Field(
        default="",
        description=(
            "Complete GDScript 2.0 (`extends Node2D`) source code for the QA test scene at `target_scene` "
            "(e.g. `res://tests/test_<item_id>.gd` paired with `res://tests/test_<item_id>.tscn`). "
            "Leave empty ('') when reusing an already-valid on-disk `res://tests/test_<item_id>.gd` unchanged or when "
            "applying targeted edits via `test_scene_patches`."
        ),
    )
    test_scene_patches: List[SurgicalCodePatch] = Field(
        default_factory=list,
        description=(
            "Optional surgical line-range patches applied to an existing `res://tests/test_<item_id>.gd` script "
            "(leave `test_scene_gd_content=''` when patching only specific lines instead of rewriting the whole file)."
        ),
    )
    max_sim_ticks: int = Field(
        default=900,
        ge=60,
        le=3600,
        description=(
            "Maximum simulation horizon in 60Hz ticks (60..3600, i.e. 1s..60s of game time; runs uncapped with VSync disabled "
            "and exits early as soon as all `capture_events`, `input_segments`, and `state_assertions` complete)."
        ),
    )
    counterfactual_precondition_check: str = Field(
        default="",
        description=(
            "Explicit causal-necessity hypothesis verified by this run: which precondition/resource/action gates progression "
            "(remaining 0/unchanged at t=0 and under zero player input) and only advances after authentic player input."
        ),
    )
    capture_events: List[QACaptureEvent] = Field(
        default_factory=list,
        description=(
            "Preferred event-driven keyframe plan (1 to 12 `QACaptureEvent` milestones, e.g. `at_tick=2` unsolved baseline followed by "
            "`on_condition` / `on_change` milestones as player actions and world interactions fire across up to `max_sim_ticks`)."
        ),
    )
    capture_ticks: List[int] = Field(
        default_factory=list,
        description="Optional explicit tick numbers (1..3600, up to 12 frames) when fixed-tick sampling is used alongside or instead of `capture_events`.",
    )
    input_segments: List[InputSegment] = Field(default_factory=list, description="Scripted inputs; empty means no input is applied.")
    setup_calls: List[QASetupCall] = Field(
        default_factory=list,
        description="Optional QA scenario setup calls invoked by the test harness on the test scene.",
    )
    state_assertions: List[QAStateAssertion] = Field(
        default_factory=list,
        description="Observable state assertions verified by the QA harness across the captured ticks.",
    )
    frame_expectations: List[str] = Field(default_factory=list, description="Optional internal notes on what each captured tick tests.")
    record_audiovisual_clip: bool = Field(
        default=False,
        description="Set True when the QA brief calls for capturing a synchronized gameplay .mp4 video+audio clip alongside the frame contact sheet.",
    )
    reroute_to_specialist: str = Field(
        default="",
        description=(
            "Optional peer handoff: if inspecting the cartridge before QA staging proves this brief was sent to QA prematurely "
            "or belongs to another Specialist Persona ('swe_engineer', 'visual_artist', 'audio_designer', 'script_writer'), "
            "set that specialist's name here and set `verification_mode='reroute'`."
        ),
    )
    reroute_brief: str = Field(
        default="",
        description="When `reroute_to_specialist` is non-empty, concrete handoff brief explaining the root cause and what the target specialist should implement or fix.",
    )
    reroute_target_paths: List[str] = Field(
        default_factory=list,
        description="Optional cartridge-relative target paths for `reroute_to_specialist`.",
    )
    verification_mode: Literal["causal_gameplay", "exploratory_diagnostic", "reroute"] = Field(
        default="causal_gameplay",
        description="Decision (strictly last): 'causal_gameplay' when proving interactive gameplay behavior via player inputs/assertions; 'exploratory_diagnostic' when performing a diagnostic scene/telemetry/visual probe without mandatory input choreography; 'reroute' to hand off directly to `reroute_to_specialist`.",
    )


class WorkItemProposal(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A backlog requirement raised by the council or dynamically discovered and spawned by an executing task.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Why this is needed, what it unblocks, and why its dependencies are what they are.")
    item_id: str = Field(..., description="Unique snake_case identifier.")
    kind: WorkItemKind = Field(..., description="Kind of work; selects the worker that builds it.")
    title: str = Field(..., description="Concise title.")
    spec: str = Field(
        ...,
        description=(
            "What must be built or resolved: define the target player experience, behavioral requirements, system interactions, and acceptance criteria "
            "(leaving low-level code diagnosis and implementation strategy to the Task Lead and SWE Engineer rather than prescribing narrow single-variable workarounds). "
            "For 'visual_asset' items, each item maps 1-to-1 to a single PNG file and a single AssetRequirement entry—propose a separate 'visual_asset' "
            "WorkItemProposal for each distinct character, creature, or machine rig rather than bundling multiple rigs into one item."
        ),
    )
    proof_spec: str = Field(..., description="The evidence (image or images) the council wants to see to judge the result, e.g. which views, states or frames.")
    depends_on: List[str] = Field(default_factory=list, description="item_ids that must be approved before this can start.")
    blocks_items: List[str] = Field(
        default_factory=list,
        description="Existing item_ids on the board (or in this batch) that require this item first and must be blocked until this item is approved (e.g. a visual_asset rig blocking a mechanic_code controller).",
    )
    target_modules: List[str] = Field(default_factory=list, description="Code modules or files this will modify (used for exclusive-access locking).")
    slice_id: str = Field(default="", description="Optional slice_id of the GameSliceSpec this task belongs to (defaults to the active slice).")
    max_revisions: Optional[int] = Field(default=None, description="Optional per-item revision cap override.")

    @field_validator("target_modules", mode="before")
    @classmethod
    def _normalize_target_modules(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        cleaned: List[str] = []
        for item in v:
            s = str(item or "").strip().replace("\\", "/")
            if s.startswith("res://"):
                s = s[len("res://"):]
            elif s.startswith("res:/"):
                s = s[len("res:/"):]
            s = s.lstrip("/")
            if s and s not in cleaned:
                cleaned.append(s)
        return cleaned


class WorkItemRefinement(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A Council specialist's pre-execution direction, cross-domain enrichment, dependency wiring,
    or task decomposition ('split') applied to an existing unstarted WorkItem on the Production Board.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Why this existing task needs domain direction, cross-module contract alignment, prerequisite dependencies, or decomposition into smaller cohesive tasks.",
    )
    item_id: str = Field(..., description="Existing item_id on the Production Board to enrich or split.")
    spec_addition_or_rewrite: str = Field(
        default="",
        description="When action is 'enrich', concrete domain direction, data-contract requirements, UX affordances, or audio/visual integration notes to append to the task's spec.",
    )
    proof_spec_update: str = Field(
        default="",
        description="Optional refinement or addition to the item's proof_spec so the Council sees evidence of the enriched requirements.",
    )
    add_depends_on: List[str] = Field(
        default_factory=list,
        description="Prerequisite item_ids (existing on the board or newly proposed in this turn) that this task must wait for before starting.",
    )
    add_target_modules: List[str] = Field(
        default_factory=list,
        description="Additional GDScript or data modules this task will read/write.",
    )
    split_items: List[WorkItemProposal] = Field(
        default_factory=list,
        description="When action is 'split', the smaller, focused WorkItemProposals that replace this oversized task (downstream dependents on the board are automatically re-pointed).",
    )
    action: Literal["enrich", "split", "cancel", "no_input"] = Field(
        default="enrich",
        description="Shaping action: 'enrich' to add specialist direction/dependencies to the existing task, 'split' to decompose it into smaller tasks, 'cancel' to retire an obsolete unstarted task superseded by a pivot, or 'no_input' if your domain has no additional requirements for this task (strictly last per Rule 4).",
    )

    @field_validator("add_target_modules", mode="before")
    @classmethod
    def _normalize_add_target_modules(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        cleaned: List[str] = []
        for item in v:
            s = str(item or "").strip().replace("\\", "/")
            if s.startswith("res://"):
                s = s[len("res://"):]
            elif s.startswith("res:/"):
                s = s[len("res:/"):]
            s = s.lstrip("/")
            if s and s not in cleaned:
                cleaned.append(s)
        return cleaned


class TaskShapingBatch(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A Council specialist's review of the unstarted backlog on the Production Board prior to execution:
    giving domain direction, wiring dependencies, splitting monolithic tasks, and raising missing companion tasks.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Cross-functional evaluation of the current Production Board tasks from your specialist domain—identifying any monolithic tasks that should be split, tasks missing domain direction/contracts, or missing prerequisite tasks.",
    )
    backlog_additions: List[WorkItemProposal] = Field(
        default_factory=list,
        description="New companion or prerequisite tasks (e.g. visual_asset rigs, audio_asset cues, UI/HUD affordances, or sub-mechanics) that existing tasks need.",
    )
    backlog_refinements: List[WorkItemRefinement] = Field(
        default_factory=list,
        description="Refinements ('enrich') or decompositions ('split') to apply to existing unstarted tasks on the Production Board.",
    )
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, summarize what you learned from the retrieved memory entries.",
    )
    memory_query: str = Field(
        default="",
        description="Optional semantic search query against long-term game/agent memory banks before finalizing task shaping (leave '' when current context suffices).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter for `memory_query` ('', 'all', 'self', 'council', or a canonical role name).",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="DEFAULT TO [] — LEAVE EMPTY ON ROUTINE TURNS. Never duplicate task specs or Bible content here. Populate ONLY when deferring an out-of-slice ambition or recording a cross-slice invariant (or updating an existing memory_id).",
    )


class ReviewDecision(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    The council's review of an item that has returned with proof.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Close examination of the proof images against the spec and the game's design, including cross-task root-cause attribution.")
    validation_criteria_for_my_role: str = Field(
        default="",
        description=(
            "Before judging the proof, define from your specialist perspective what must be true for this change to genuinely work and feel right "
            "for a human player in real gameplay (e.g., human-scale responsiveness and magnitude of motion/action over time under real player inputs, "
            "absence of conflicting subsystems fighting player input, visual/UI harmony, and whether the QA test scene + debug overlays actually prove that behavior rather than a static or trivial state)."
        ),
    )
    item_id: str = Field(..., description="Item under review.")
    feedback: str = Field(default="", description="Concrete, actionable changes required for THIS item's own scope (empty when approving).")
    blocked_by_item_ids: List[str] = Field(
        default_factory=list,
        description="Optional IDs of existing unapproved items on the Production Board that own an out-of-scope or shared defect visible in this item's proof. This item will be blocked on those items rather than re-implementing their modules.",
    )
    spawned_prerequisite_tasks: List[WorkItemProposal] = Field(
        default_factory=list,
        description="Optional new high-priority prerequisite bugfix task(s) to spawn when a defect visible in this item's proof belongs to a shared/already-approved module outside this item's core responsibility.",
    )
    out_of_scope_modules_to_prune: List[str] = Field(
        default_factory=list,
        description="Optional module paths (e.g. shared camera, world scene, or hub files) that belong to a linked blocker or spawned prerequisite task and must be pruned from this item's target_modules so workers do not collide.",
    )
    verdict: Literal["approve", "request_changes"] = Field(..., description="Decision (strictly last).")

    @field_validator("out_of_scope_modules_to_prune", mode="before")
    @classmethod
    def _normalize_prune_modules(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        cleaned: List[str] = []
        for item in v:
            s = str(item or "").strip().replace("\\", "/")
            if s.startswith("res://"):
                s = s[len("res://"):]
            elif s.startswith("res:/"):
                s = s[len("res:/"):]
            s = s.lstrip("/")
            if s and s not in cleaned:
                cleaned.append(s)
        return cleaned


class CapDecision(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    The council's explicit decision for an item that hit its revision cap or stalled.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Review of the full revision history and why this resolution is right.")
    item_id: str = Field(..., description="Item needing a decision.")
    replacement_spec: str = Field(default="", description="New spec when action is 'rewrite_spec'.")
    split_items: List[WorkItemProposal] = Field(default_factory=list, description="Smaller replacement items when action is 'split'.")
    action: Literal["accept_as_is", "rewrite_spec", "split", "cancel"] = Field(..., description="Resolution (strictly last).")


class ReviewBatch(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    One reviewer's decisions for every item currently waiting on them.
    """
    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: Skeptical independent assessment of the rendered proofs and telemetry against each item's "
            "spec, Executive Producer direction, and active-slice scope—distrusting unverified worker claims without inventing out-of-scope feature creep."
        ),
    )
    spoken_dialogue: str = Field(
        default="",
        description=(
            "Direct, zero-flattery critique in the Council Review Roundtable—NEVER open with 'I completely agree', 'I echo', "
            "'I couldn't agree more', or peer praise, and NEVER repeat points an earlier reviewer already made. Lead straight "
            "with what YOUR discipline uniquely caught in the proof frames/telemetry (or where you disagree with a peer), "
            "while catering respectfully to any Executive Producer direction and calling in-scope items done when satisfied."
        ),
    )
    addressed_to: List[str] = Field(
        default_factory=list,
        description="Fellow Council specialist role(s) you are addressing in this review roundtable.",
    )
    review_decisions: List[ReviewDecision] = Field(default_factory=list, description="One decision per item awaiting review.")
    cap_decisions: List[CapDecision] = Field(default_factory=list, description="One decision per item that hit its revision cap or stalled.")
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, summarize what you learned from the retrieved memory entries.",
    )
    memory_query: str = Field(
        default="",
        description="Optional semantic search query against long-term game/agent memory banks before finalizing your review decisions (leave '' when current context suffices).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter for `memory_query` ('', 'all', 'self', 'council', or a canonical role name).",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="DEFAULT TO [] — LEAVE EMPTY ON ROUTINE REVIEWS. Item-specific revision feedback belongs in `verdicts`, NOT long-term memory. Populate ONLY for systemic cross-task/cross-slice lessons, deferred future-slice goals, or fulfilling an existing memory_id.",
    )


class SlicePlaytestRoundtableTurn(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    One Council specialist's conversational turn evaluating the automated slice playthrough filmstrip and telemetry.
    """
    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: Skeptical yet scope-disciplined reasoning examining what the playthrough filmstrip and "
            "telemetry actually prove, separating must-fix current-slice problem areas (and Executive Producer priorities) "
            "from broader changes that belong in future slices."
        ),
    )
    validation_criteria_for_my_role: str = Field(
        default="",
        description="What you determined must be true from your discipline's perspective (e.g., human-scale control responsiveness over time, subsystem harmony, UX clarity, visual/HUD layout cohesion, and genuine fun/agency through the slice goal).",
    )
    spoken_dialogue: str = Field(
        ...,
        description=(
            "Candid, zero-flattery critique to the Writers' Room—NEVER open with 'I completely agree', 'I echo', or 'feels wonderful'. "
            "State plainly what works, what is rough, which problem areas must be fixed inside this slice's scope vs. deferred, "
            "and treat any Executive Producer direction with full respect."
        ),
    )
    addressed_to: List[str] = Field(
        default_factory=list,
        description="Fellow Council specialist role(s) you are addressing.",
    )
    observed_issues_or_gaps: List[str] = Field(
        default_factory=list,
        description="Concrete playability, responsiveness, visual/UI, or assembly issues observed in the slice playthrough (empty if none).",
    )
    is_playable_and_fun_from_my_perspective: bool = Field(
        ...,
        description=(
            "True once the active slice's intended scope works end-to-end at human scale and major current-slice problem areas "
            "are cleaned up (knowing we cannot change everything in one go); False if core slice scope or glaring visual/UX/gameplay "
            "defects still need a surgical fix pass (strictly last)."
        ),
    )


class SliceAssemblyDecision(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Council evaluation when all current Production Board tasks for the active slice have been approved,
    determining whether the built modules and assets are cohesively assembled in the playable entrypoint
    and proven playable and fun through the slice playthrough, or whether integration/fix task(s)
    should be added to the Production Board before closing the slice.
    """
    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: Skeptical yet pragmatic evaluation of the slice playthrough filmstrip, live scene tree, "
            "and entrypoint reachability—verifying whether active-slice scope and Executive Producer directives are met and "
            "major problem areas are cleaned up without trying to rewrite everything in one go."
        ),
    )
    playability_and_fun_evaluation: str = Field(
        default="",
        description=(
            "Evaluation of the automated slice playthrough filmstrip and telemetry: did the playthrough script achieve the slice goal using real player controls, "
            "do controls/movement/actions respond at a playable human scale without fighting each other, and is the assembled experience cohesive, legible, and fun?"
        ),
    )
    spoken_dialogue: str = Field(
        default="",
        description=(
            "Direct, candid synthesis to the Writers' Room—no cheerleading or 'I completely agree' filler—stating plainly "
            "either which bounded current-slice problem areas are being fixed now OR why the slice's scope is covered and ready to call done."
        ),
    )
    unmounted_or_disconnected_systems: List[str] = Field(
        default_factory=list,
        description="Specific built modules, UI instruments, or gameplay loops that are required for this slice's core loop but are not yet mounted, wired, or playable in the entrypoint.",
    )
    backlog_additions: List[WorkItemProposal] = Field(
        default_factory=list,
        description=(
            "Tight, surgical integration or gameplay/UI/visual fix WorkItemProposal tasks (typically 1-3 items) to wire unmounted "
            "slice modules or clean up high-impact current-slice problem areas—never an unbounded rewrite of everything at once "
            "(empty when is_slice_assembled=True)."
        ),
    )
    is_slice_assembled: bool = Field(
        ...,
        description=(
            "True when the active slice's scope is mounted and working end-to-end in the main entrypoint and major current-slice "
            "problem areas are resolved (calling the slice done appropriately rather than chasing endless perfection); False when "
            "targeted current-slice fixes are still required (strictly last)."
        ),
    )


class BlindPlaytestCouncilTurn(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    One Council specialist's conversational turn deliberating on the Blind AI Playtester's end-of-slice
    Experience Report, auto-delivered player wishes, discovered controls rulebook, and session screenshots.
    """
    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: Skeptical, evidence-first analysis of the Blind Playtester's screenshots, action tape, "
            "report, and wishes—weighing which defects must be cleaned up within this slice's scope (plus any Executive Producer "
            "priorities) versus what should be deferred so we do not try to change everything in one go."
        ),
    )
    domain_reaction_to_blind_player: str = Field(
        default="",
        description=(
            "Your discipline-specific diagnosis distinguishing (A) UX / Affordance / Spatial Bug Failures (clicks landing offset "
            "from cursor, illegible rules/goals, overlapping UI, blurry/colliding sprites, or working setups failing to credit) vs. "
            "(B) Good Strategic Failure (player spent carelessly or ignored positioning and lost—which MUST be preserved as real game "
            "challenge, NEVER 'fixed' by inflating t=0 starting handouts or adding passive auto-collection)."
        ),
    )
    spoken_dialogue: str = Field(
        ...,
        description=(
            "Candid, zero-flattery contribution to the Writers' Room—NEVER open with 'I completely agree', 'I echo', or peer "
            "back-patting. State directly what the screenshots/tape reveal, what few high-impact issues (if any) belong in a "
            "current-slice cleanup vs. next slice, and treat any Executive Producer input with full respect."
        ),
    )
    addressed_to: List[str] = Field(
        default_factory=list,
        description="Fellow Council specialist role(s) you are addressing.",
    )
    priority_fixes_or_refinements: List[str] = Field(
        default_factory=list,
        description=(
            "Bounded, high-impact current-slice fixes (broken slice mechanics, stalled objectives, glaring visual/UI craft "
            "defects, silent audio, or Executive Producer priorities) needed before closing this slice—leave [] when the "
            "slice's scope is covered and remaining items belong in future slices."
        ),
    )
    recommends_slice_approval: bool = Field(
        ...,
        description=(
            "True when the active slice's scope works end-to-end and major problem areas have been cleaned up so it is appropriate "
            "to call this slice done and carry broader improvements into the next slice; False when targeted current-slice fixes "
            "are still needed first (strictly last)."
        ),
    )


class BlindPlaytestCouncilVerdict(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Council Creative Director synthesis of the Blind Playtester's Experience Report, wishes, and specialist
    roundtable—deciding whether to iterate on the current slice (`iterate_current_slice`) or approve it (`approve_slice`).
    """
    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: Critical yet pragmatic synthesis auditing the Blind Playtester's screenshots, action tape, "
            "report, wishes, and Executive Producer priorities—deciding whether a bounded current-slice cleanup pass is needed "
            "or whether the slice's scope is covered and should be called done now."
        ),
    )
    player_experience_synthesis: str = Field(
        ...,
        description=(
            "Candid synthesis of what actually worked, what was rough or broken, and whether player struggles stemmed from "
            "(A) UX / spatial / wiring defects (fix clarity, camera coordinate math, or bugs) vs. (B) Good Strategic Failure "
            "(preserve challenge and scarcity—never inflate t=0 free resources or automate away the core verb)."
        ),
    )
    spoken_dialogue: str = Field(
        ...,
        description=(
            "Direct, candid Creative Director synthesis—zero cheerleading or 'I completely agree' filler—announcing either "
            "a focused current-slice fix pass or calling the slice done while naming what carries into the next slice."
        ),
    )
    next_slice_carryover_notes: List[str] = Field(
        default_factory=list,
        description="Non-blocking player wishes, feature expansions, and honest rough-edge notes carried into the next slice so we do not try to change everything in one go.",
    )
    backlog_additions: List[WorkItemProposal] = Field(
        default_factory=list,
        description=(
            "Surgical, bounded current-slice WorkItemProposal tasks (typically 1-3 high-impact fixes) for broken slice scope, "
            "onboarding/UX blockers, glaring visual/UI craft bugs, missing audio, or Executive Producer requests (empty when verdict='approve_slice')."
        ),
    )
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, summarize what you learned from the retrieved memory entries.",
    )
    memory_query: str = Field(
        default="",
        description="Optional semantic search query against long-term game/agent memory banks before finalizing the blind playtest verdict (leave '' when current context suffices).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter for `memory_query` ('', 'all', 'self', 'council', or a canonical role name).",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="DEFAULT TO [] UNLESS HIGH-SIGNAL. Do NOT repeat active-slice fix requests already in `new_board_tasks`. Populate ONLY to persist deferred future-slice strategic goals, systemic playability lessons, or `fulfill`/`supersede`/`archive` updates on existing memory_ids.",
    )
    verdict: Literal["approve_slice", "iterate_current_slice"] = Field(
        ...,
        description=(
            "Set 'iterate_current_slice' when focused current-slice fixes or problem-area cleanups are needed in backlog_additions; "
            "set 'approve_slice' once the active slice's scope is covered and major rough spots are addressed so the slice is called done appropriately (strictly last)."
        ),
    )


class SliceStateReflectionTurn(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    One Council specialist's first-principles reflection on the current empirical state of the game
    at the end of a slice, diagnosing what is missing in cohesion, intuitiveness ('knowing how to play'),
    depth/fun for the target player, sense of accomplishment, and short-term / long-term goals before
    strategizing the next slice.
    """
    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: Skeptical first-principles reflection examining the rendered filmstrip, scene tree, "
            "and codebase through the target player's eyes while honoring Executive Producer direction and scoping realistically for one slice."
        ),
    )
    honest_current_state_critique: str = Field(
        ...,
        description=(
            "Brutally honest assessment of what a human player actually sees and experiences booting res://main.tscn right now "
            "(calling out any disconnected widget/slider dashboards, center-occluding panels, sprite/UI craft glitches, leaked debug harnesses, static/unreactive world visuals, or un-cohesive systems)."
        ),
    )
    how_to_play_and_intuitiveness_gaps: str = Field(
        ...,
        description=(
            "Diagnosis of whether a brand-new player sitting down right now with zero outside explanation knows how to play: "
            "what their controls do, what is interactive, and how their actions causally affect the game world."
        ),
    )
    depth_and_fun_gaps: str = Field(
        ...,
        description=(
            "Diagnosis of whether the current interaction loop is genuinely entertaining, interesting, and mechanically deep for this specific player's DNA—identifying "
            "where meaningful decisions, tension, resource/timing trade-offs, or dynamic world pushback are missing."
        ),
    )
    accomplishment_and_goals_gaps: str = Field(
        ...,
        description=(
            "Diagnosis of what short-term goals (immediate 15-60s actionable targets), long-term goals (overarching session progression milestones), "
            "and visceral sense of accomplishment/reward are missing or unfelt in the current build."
        ),
    )
    strategy_for_next_slice: str = Field(
        ...,
        description=(
            "Focused, achievable strategy from your specialist discipline for the highest-leverage improvements Slice #N+1 should deliver "
            "(respecting Executive Producer direction without trying to rebuild everything in one go)."
        ),
    )
    spoken_dialogue: str = Field(
        ...,
        description=(
            "Direct, candid contribution to the Slice Reflection Roundtable—NEVER open with 'I completely agree' or peer flattery; "
            "call out concrete flaws, push back on over-scoped wishlists, and cater respectfully to Executive Producer directives."
        ),
    )
    addressed_to: List[str] = Field(
        default_factory=list,
        description="Fellow Council specialist role(s) you are addressing or challenging.",
    )
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, summarize what you learned from the retrieved memory entries.",
    )
    memory_query: str = Field(
        default="",
        description="Optional semantic search query ONLY when searching other specialist banks not already listed in your `LONG-TERM GAME & AGENT MEMORY BRIEFING` above (setting `memory_query` triggers a 2-pass lookup turn; leave '' when finalizing your slice reflection).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter for `memory_query` ('', 'all', 'self', 'council', or a canonical role name).",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="High-signal cross-slice memory curation ONLY (never write slice recap summaries or duplicate Bible fields): mark completed/obsolete `[mem_...]` entries as `'fulfill'`/`'supersede'`/`'archive'`, and record ONLY genuinely deferred multi-slice goals or hard-won systemic lessons.",
    )
    current_slice_quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Honest quality and player-experience score of the current slice in [0.0, 1.0] (strictly last per Rule 4).",
    )


class NextSliceStrategyPlan(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Master strategy synthesized by the CoProducer (Council Chair) after the Council's Slice Reflection Roundtable,
    defining how the next slice transforms the game's cohesion, onboarding/intuitiveness, depth/fun,
    short-term & long-term goals, and sense of accomplishment for the target player.
    """
    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: Skeptical, realistic synthesis of the Council's empirical reflection on the current slice, "
            "catering directly and respectfully to any Executive Producer slice directive while scoping the next slice to a coherent, "
            "achievable increment (fixing key rough spots + delivering the next slice's core loop expansion without trying to rewrite everything at once)."
        ),
    )
    spoken_dialogue: str = Field(
        ...,
        description=(
            "Direct, candid first-person address to the Writers' Room laying out what the previous slice actually fell short on, "
            "how we are catering to any Executive Producer direction, and the bounded, achievable plan for THIS next slice "
            "(never open with peer flattery or 'I completely agree with everyone')."
        ),
    )
    current_slice_honest_diagnosis: str = Field(
        ...,
        description="Consolidated diagnosis of the current slice's shortcomings in cohesion, knowing how to play, depth/fun, accomplishment, and short/long-term goals.",
    )
    cohesion_and_world_integration_strategy: str = Field(
        ...,
        description=(
            "Strategy for unifying the viewport into a living, reactive game world in the next slice "
            "(eliminating any center-occluding UI boxes or debug harnesses and visually connecting player actions to physical world reactions)."
        ),
    )
    how_to_play_and_onboarding_strategy: str = Field(
        ...,
        description=(
            "Strategy for ensuring a player booting the game immediately and intuitively knows how to play, what controls to use, "
            "why they are acting, and how the world responds."
        ),
    )
    depth_and_interesting_decisions_strategy: str = Field(
        ...,
        description=(
            "Strategy for introducing genuine mechanical depth, dynamic world pushback/hazards, and meaningful trade-offs/decisions "
            "tailored to the target player's SteamPlayerDNA."
        ),
    )
    short_term_and_long_term_goals_strategy: str = Field(
        ...,
        description=(
            "Concrete short-term goals (immediate 15-60s objectives) and long-term goals (multi-stage session progression arc) "
            "that will be visibly tracked and playable on screen in the next slice."
        ),
    )
    sense_of_accomplishment_and_reward_strategy: str = Field(
        ...,
        description=(
            "Strategy for how completing short-term and long-term goals delivers an unmistakable sense of accomplishment "
            "(visceral visual/audio/world transformation, milestone ceremonies, and meaningful progression rewards/unlocks)."
        ),
    )
    next_slice_spec: GameSliceSpec = Field(
        ...,
        description="The newly strategized GameSliceSpec for the next slice (with status='active').",
    )
    updated_genre_craft_foundation: Optional[GenreCraftFoundation] = Field(
        default=None,
        description=(
            "Optional evolved GenreCraftFoundation refining foundational pillars, core fun thesis, "
            "bootstrap-to-mastery progression, UX precision contracts, or anti-patterns for the next slice."
        ),
    )
    updated_core_loop: Optional[CoreLoopSpec] = Field(
        default=None,
        description="Optional evolved CoreLoopSpec deepening moment-to-moment, minute-to-minute, and session loops, agency, and reward cadence.",
    )
    updated_onboarding_plan: Optional[OnboardingPlan] = Field(
        default=None,
        description="Optional evolved OnboardingPlan ensuring intuitive in-game 'how to play' guidance and causal feedback.",
    )
    updated_goal_hierarchy: Optional[GoalHierarchy] = Field(
        default=None,
        description="Optional evolved GoalHierarchy defining explicit immediate, short-term, and long-term goals, progress feedback, and rewards.",
    )
    updated_engagement_plan: Optional[EngagementPlan] = Field(
        default=None,
        description="Optional evolved EngagementPlan deepening variety injectors, pacing curve, and mastery.",
    )
    updated_diegetic_ui_philosophy: Optional[str] = Field(
        default=None,
        description="Optional updated UI/HUD philosophy ensuring non-occluding perimeter instruments and clear goal/control legibility.",
    )
    backlog_additions: List[WorkItemProposal] = Field(
        default_factory=list,
        description="Initial high-level WorkItemProposal tasks on the Production Board to execute the next-slice strategy.",
    )
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, summarize what you learned from the retrieved memory entries.",
    )
    memory_query: str = Field(
        default="",
        description="Optional semantic search query ONLY when searching other specialist banks not already listed in your `LONG-TERM GAME & AGENT MEMORY BRIEFING` above (setting `memory_query` triggers a 2-pass lookup turn; leave '' when finalizing `NextSliceStrategyPlan`).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter for `memory_query` ('', 'all', 'self', 'council', or a canonical role name).",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="High-signal Council memory curation ONLY (never duplicate `next_slice_spec` or Bible updates): mark delivered/replaced `[mem_...]` entries as `'fulfill'`/`'supersede'`/`'archive'`, and record ONLY deferred multi-slice strategic goals, cross-slice architectural decisions, or studio-wide lessons in `'council'`.",
    )


class CouncilDiagnosticProbe(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Evaluated by the Council when an Executive Producer steering / bug report or slice review question arrives,
    determining whether the root cause is already clear from static context or whether ANY of the 5 Specialist Personas
    (`visual_artist`, `audio_designer`, `script_writer`, `swe_engineer`, `qa_test_engineer`) should first run a
    read-only/diagnostic inspection probe to gather ground-truth evidence.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Technical/artistic analysis of the reported issue against the cartridge state and whether a specialist diagnostic probe is needed.",
    )
    specialist_agent: Literal[
        "visual_artist",
        "audio_designer",
        "script_writer",
        "swe_engineer",
        "qa_test_engineer",
    ] = Field(
        default="qa_test_engineer",
        description="Which Specialist Persona to invoke in read-only/diagnostic inspection mode when `needs_runtime_debug_scene=True`.",
    )
    diagnostic_brief: str = Field(
        default="",
        description="Concrete diagnostic/inspection brief for `specialist_agent` specifying what asset, audio clip, script, code module, or runtime scene to inspect.",
    )
    diagnostic_qa_brief: str = Field(
        default="",
        description="Optional alias for `diagnostic_brief` when targeting `qa_test_engineer`.",
    )
    target_paths: List[str] = Field(
        default_factory=list,
        description="Optional cartridge-relative asset, audio, narrative, or `.gd` paths to inspect.",
    )
    needs_runtime_debug_scene: bool = Field(
        ...,
        description="True when a specialist diagnostic/inspection probe should run before finalizing Council task direction; False when static context is already sufficient (strictly last).",
    )

    @property
    def effective_brief(self) -> str:
        return (self.diagnostic_brief or "").strip() or (self.diagnostic_qa_brief or "").strip()


class CouncilSteeringRoutingPlan(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    LLM-driven intake routing plan when Executive Producer steering or playtest feedback arrives,
    selecting the right Council domain lead(s) and optional Specialist Persona diagnostic probe
    rather than hardcoding a fixed role order.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Multi-domain analysis of the incoming steering/feedback against the cartridge codebase, visual asset inventory/contact sheet, audio, UX, and design.",
    )
    lead_council_role: str = Field(
        ...,
        description=(
            "Primary Council specialist best suited to own this steering/report "
            "('CoProducer', 'GameplayDesigner', 'LevelAndProgressionDesigner', 'VisualArtDirector', "
            "'UXExperienceDesigner', 'AudioAtmosphereDesigner', 'StoryWorldArchitect', 'ScriptwriterAgent', "
            "'CharacterDesigner', or 'GameplaySystemsEngineer')."
        ),
    )
    supporting_council_roles: List[str] = Field(
        default_factory=list,
        description="0 to 3 additional Council specialists whose domains directly intersect this request (in speaking order).",
    )
    specialist_probe_agent: str = Field(
        default="",
        description="When `needs_specialist_probe=True`, which of the 5 Specialist Personas ('visual_artist', 'audio_designer', 'script_writer', 'swe_engineer', 'qa_test_engineer') should run a read-only diagnostic probe first.",
    )
    diagnostic_brief: str = Field(
        default="",
        description="When `needs_specialist_probe=True`, concrete inspection brief for `specialist_probe_agent`.",
    )
    target_paths: List[str] = Field(
        default_factory=list,
        description="Optional cartridge-relative asset, audio, narrative, or `.gd` paths to inspect during the probe.",
    )
    needs_specialist_probe: bool = Field(
        default=False,
        description="True when a Specialist Persona diagnostic probe should execute before Council turns; False when current codebase + visual contact-sheet context already suffices (strictly last).",
    )


class RevisionRecord(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    One build attempt of a work item: what the worker produced and how the council responded.
    """
    deliberation: str = Field(default="Worker attempt recorded.", description="Why this revision exists.")
    revision_index: int = Field(default=0)
    worker_notes: str = Field(default="")
    proof_paths: List[str] = Field(default_factory=list)
    proof_hashes: List[str] = Field(default_factory=list)
    feedback: str = Field(default="", description="Council feedback on this revision.")
    verdict: str = Field(default="", description="'approve' | 'request_changes' | ''")
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WorkItem(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A backlog item on the Production Board with its full life-cycle state.
    """
    deliberation: str = Field(..., description="Why this item exists.")
    item_id: str
    kind: WorkItemKind
    title: str
    spec: str
    proof_spec: str = Field(default="")
    depends_on: List[str] = Field(default_factory=list)
    target_modules: List[str] = Field(default_factory=list)
    raised_by_role: str = Field(default="")
    slice_id: str = Field(default="", description="Identifier of the GameSliceSpec this item belongs to.")
    spawned_by_item_id: str = Field(default="", description="If dynamically discovered and spawned by an executing task, the parent item_id.")
    design_revision: int = Field(default=0, description="Number of Game Bible updates when this item was raised.")
    state: WorkItemState = Field(default="blocked")
    max_revisions: int = Field(default=5)
    spec_resets: int = Field(default=0)
    cap_base_index: int = Field(default=0, description="Revisions before this index do not count against the cap (set when the spec is rewritten).")
    revisions: List[RevisionRecord] = Field(default_factory=list)
    accepted_compromise: str = Field(default="", description="Set when the council accepted the item as-is after the cap/stall.")
    failure: str = Field(default="", description="Full error text of the last worker failure.")
    failure_count: int = Field(default=0, description="Number of worker failures so far.")
    attempt_count: int = Field(
        default=0,
        description="Cumulative 1-based execution attempt count across all revisions, active runs, and auto-retried worker failures.",
    )
    priority: int = Field(default=0)
    decision_reason: str = Field(default="", description="Why the item is in needs_decision.")
    shaping_notes: List[str] = Field(
        default_factory=list,
        description="Pre-execution Council specialist direction and shaping notes contributed to this task.",
    )
    excluded_modules: List[str] = Field(
        default_factory=list,
        description="Module paths explicitly pruned from this item's scope by a [Council Scope Boundary] decision so workers do not overwrite shared/external modules.",
    )
    waiting_on_locks: List[str] = Field(
        default_factory=list,
        description="File paths or lock names this item is currently waiting on after yielding mid-run.",
    )

    @field_validator("target_modules", "excluded_modules", "waiting_on_locks", mode="before")
    @classmethod
    def _normalize_item_target_modules(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        cleaned: List[str] = []
        for item in v:
            s = str(item or "").strip().replace("\\", "/")
            if s.startswith("res://"):
                s = s[len("res://"):]
            elif s.startswith("res:/"):
                s = s[len("res:/"):]
            s = s.lstrip("/")
            if s and s not in cleaned:
                cleaned.append(s)
        return cleaned


class SpecialistReport(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Result returned by an autonomous domain specialist to the Task Lead Agent on a single delegation turn.
    """
    deliberation: str = Field(default="Specialist execution completed.", description="MANDATORY FIRST PROPERTY: Specialist's reasoning or execution notes.")
    turn_index: int = Field(default=0, description="Turn index within the current board revision.")
    specialist_agent: str = Field(..., description="Specialist identifier ('visual_artist', 'audio_designer', 'script_writer', 'swe_engineer', 'qa_test_engineer').")
    brief: str = Field(..., description="The natural-language brief delegated by the Task Lead.")
    summary: str = Field(default="", description="Concrete summary of what the specialist created, modified, or verified.")
    touched_files: List[str] = Field(default_factory=list, description="Paths of assets, data files, or GDScript modules created or modified.")
    proof_images: List[str] = Field(default_factory=list, description="Proof image paths generated during this specialist step.")
    diagnostic_artifact_paths: List[str] = Field(
        default_factory=list,
        description="Optional diagnostic plates, contact sheets, or telemetry artifacts produced during inspection.",
    )
    recommended_followups: List[str] = Field(
        default_factory=list,
        description="Optional concise follow-up recommendations or out-of-scope observations surfaced to the Task Lead.",
    )
    rerouted_from: List[str] = Field(
        default_factory=list,
        description="Ordered chain of Specialist Personas that dynamically re-routed this turn before execution completed.",
    )
    reroute_to_specialist: str = Field(
        default="",
        description="Target Specialist Persona when this specialist determined another persona should handle the work.",
    )
    reroute_brief: str = Field(
        default="",
        description="Concrete handoff brief formulated by this specialist for `reroute_to_specialist`.",
    )
    reroute_target_paths: List[str] = Field(
        default_factory=list,
        description="Target file paths recommended for `reroute_to_specialist`.",
    )
    verification_passed: bool = Field(default=True, description="Whether engine/QA verification passed on this specialist turn.")
    code_diffs: Dict[str, str] = Field(default_factory=dict, description="Per-file unified diffs of code/scene modules modified during this turn.")
    step_log: List[Dict[str, Any]] = Field(default_factory=list, description="Chronological log of sub-steps executed during this turn.")
    curated_context: str = Field(default="", description="Curated context block assembled for this specialist turn.")
    prompt_text: str = Field(default="", description="Full prompt text sent to the LLM during this turn.")
    self_inquiries: List[Dict[str, Any]] = Field(default_factory=list, description="Internal self-inquiry questions asked and answered during this turn.")
    lead_deliberation: str = Field(default="", description="Task Lead Strategist deliberation when planning this turn.")
    lead_prompt_text: str = Field(default="", description="Task Lead Strategist prompt text when planning this turn.")
    lead_curated_context: str = Field(default="", description="Task Lead Strategist curated context when planning this turn.")
    lead_self_inquiries: List[Dict[str, Any]] = Field(default_factory=list, description="Task Lead Strategist self-inquiry Q&As during this turn.")
    timestamp_utc: str = Field(default="", description="ISO-8601 UTC timestamp when this turn completed.")


class ScriptWritingDeliverable(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Dramatic scenes, cutscenes, interactive dialogue trees, contextual barks, and character voices authored by the Scriptwriter specialist for a game slice or narrative task.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Dramatic rationale evaluating how these scenes, dialogue lines, or character voices serve this specific game's tone, pacing, and gameplay.",
    )
    self_question_answer: str = Field(
        default="",
        description="If you asked `self_question` on the previous turn, summarize the retrieved codebase/Bible facts here before authoring or inspecting.",
    )
    self_question: str = Field(
        default="",
        description="Optional semantic inquiry against the codebase, Game Bible, or asset catalog when you need exact event names, character states, or script hooks before authoring (leave '' when current context suffices).",
    )
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, summarize what you learned from the retrieved memory entries before writing scenes or dialogue.",
    )
    memory_query: str = Field(
        default="",
        description="Optional semantic search query against long-term game/agent memory banks before finalizing narrative deliverables (leave '' when current context suffices).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter for `memory_query` ('', 'all', 'self', 'council', or a canonical role name).",
    )
    tool_search_answer: str = Field(
        default="",
        description="If you issued `tool_search_query` on the previous turn, summarize the relevant narrative tool capabilities here.",
    )
    tool_search_query: str = Field(
        default="",
        description="Optional progressive tool discovery query for narrative authoring/inspection schemas (leave '' when current context suffices).",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="DEFAULT TO [] — LEAVE EMPTY ON ROUTINE TURNS. Never duplicate `NarrativeBible` sheets, barks, or cutscenes here. Populate ONLY for non-obvious cross-slice narrative continuity invariants (or updating an existing memory_id).",
    )
    added_or_updated_characters: List[CharacterSheet] = Field(
        default_factory=list,
        description="Characters added or enriched with vocal cadence, motivation, and role.",
    )
    added_or_updated_cutscenes: List[CutsceneScript] = Field(
        default_factory=list,
        description="Dramatic scene scripts or cinematic/in-engine sequences with line-by-line dialogue, camera staging, and emotional delivery.",
    )
    added_or_updated_dialogue_trees: List[DialogueTreeSpec] = Field(
        default_factory=list,
        description="Interactive branching dialogue trees with player choices, tone tags, and consequence flags.",
    )
    added_or_updated_barks: List[DialogueBarkSpec] = Field(
        default_factory=list,
        description="Short contextual in-gameplay lines or barks triggered by specific game events.",
    )
    narrative_notes: str = Field(
        default="",
        description="Summary of the dramatic writing crafted in this pass and how GDScript or audio systems can hook into it.",
    )
    inspection_report: str = Field(
        default="",
        description="When `operation_mode == 'inspect_only'`, structured continuity, tone, character-voice, and dialogue-hook audit findings.",
    )
    recommended_followups: List[str] = Field(
        default_factory=list,
        description="Optional concise follow-up recommendations surfaced back to the Task Lead.",
    )
    reroute_to_specialist: str = Field(
        default="",
        description=(
            "Optional peer handoff: if this brief actually belongs to another Specialist Persona "
            "('swe_engineer', 'visual_artist', 'audio_designer', 'qa_test_engineer'), set that specialist's name here and set `operation_mode='reroute'`."
        ),
    )
    reroute_brief: str = Field(
        default="",
        description="When `reroute_to_specialist` is non-empty, concrete handoff brief explaining what the target specialist should do.",
    )
    reroute_target_paths: List[str] = Field(
        default_factory=list,
        description="Optional cartridge-relative paths for `reroute_to_specialist`.",
    )
    operation_mode: Literal["author", "inspect_only", "reroute"] = Field(
        default="author",
        description="Decision (strictly last): 'author' when creating/updating narrative bible & cartridge dialogue data; 'inspect_only' for a read-only narrative continuity/tone audit; 'reroute' to hand off to `reroute_to_specialist`.",
    )


class QAPostCaptureAudit(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Post-capture multimodal visual & behavioral audit performed by `qa_test_engineer` on the actual captured
    filmstrip frames before reporting back to the Task Lead.
    """
    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: Objective frame-by-frame visual and behavioral inspection of the captured filmstrip image "
            "(Frame 1 through Frame N) against the 16:9 Concept Art Mockup, task spec, proof spec, and any prior Council rejection feedback. "
            "Verify that the intended subjects, interactions, and scene elements remain visibly observable inside the 16:9 viewport across all captured frames."
        ),
    )
    observed_frames_summary: str = Field(
        ...,
        description=(
            "Honest frame-by-frame description of what is ACTUALLY visible in the rendered pixels of each captured frame "
            "(never pre-capture expectations or text telemetry assumptions)."
        ),
    )
    post_capture_test_scene_summary: str = Field(
        default="",
        description=(
            "Updated post-capture briefing for the Council explaining what the test scene staged, what inputs/overlays ran, "
            "and what the rendered frames actually show."
        ),
    )
    unaddressed_defects: List[str] = Field(
        default_factory=list,
        description=(
            "List of any visual, framing, or behavioral defects visible in the captured frames (e.g., subjects disappearing from the viewport, "
            "missing/clipped sprites, or unfulfilled acceptance criteria)."
        ),
    )
    recommended_followups: List[str] = Field(
        default_factory=list,
        description="Optional concise follow-up suggestions or non-blocking polish observations surfaced back to the Task Lead.",
    )
    qa_passed: bool = Field(
        ...,
        description=(
            "True ONLY if every captured frame visually and behaviorally satisfies the proof_spec and keeps the relevant subjects "
            "visibly observable inside the viewport across the scenario."
        ),
    )


class VisualArtistPlan(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Internal tool-selection plan for the unified `visual_artist` persona (creator + perceptual visual evaluator).
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Rationale evaluating the delegated `brief`, existing assets/mockups, and whether this step requires asset fabrication, diagnostic contact-sheet rendering, native VLM perceptual inspection, or both.",
    )
    self_question_answer: str = Field(
        default="",
        description="If you asked `self_question` on the previous turn, summarize the retrieved codebase/Bible/catalog facts before acting.",
    )
    self_question: str = Field(
        default="",
        description="Optional semantic inquiry against the codebase, Game Bible, or asset catalog when you need exact pixel dimensions, sprite usage, or art direction before acting (leave '' when current context suffices).",
    )
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, summarize the retrieved institutional lessons before acting.",
    )
    memory_query: str = Field(
        default="",
        description="Optional semantic search query against long-term memory banks before acting (leave '' when current context suffices).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter when issuing `memory_query`.",
    )
    tool_search_answer: str = Field(
        default="",
        description="If you issued `tool_search_query` on the previous turn, summarize the relevant visual fabrication/inspection capabilities here.",
    )
    tool_search_query: str = Field(
        default="",
        description="Optional progressive tool discovery query for visual fabrication modes, contact-sheet overlays, or VLM inspection options (leave '' when current context suffices).",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="DEFAULT TO [] — LEAVE EMPTY ON ROUTINE TURNS. Populate ONLY for non-obvious cross-slice visual/art-direction invariants.",
    )
    target_asset_paths: List[str] = Field(
        default_factory=list,
        description="Relative cartridge paths (e.g. 'assets/sprites/hero.png') to fabricate, render onto a contact sheet, or inspect.",
    )
    fabrication_notes: str = Field(
        default="",
        description="Optional synthesis/refinement notes when `operation` includes fabrication.",
    )
    contact_sheet_output_path: str = Field(
        default="",
        description="Optional relative output path for a diagnostic 1:1 contact sheet plate (e.g. 'proofs/visual_audit_plate.png').",
    )
    grid_px: int = Field(
        default=0,
        ge=0,
        description="Optional pixel-grid ruler spacing (e.g. 16, 32, 64) to draw on a diagnostic contact sheet when inspecting atlas/tile alignment; 0 for clean un-gridded 1:1 display.",
    )
    crop_boxes: List[Tuple[int, int, int, int]] = Field(
        default_factory=list,
        description="Optional list of (x, y, width, height) pixel sub-regions on the first target asset to render as zoomed 1:1 callouts on the diagnostic plate.",
    )
    inspection_inquiry: str = Field(
        default="",
        description="Specific perceptual question(s) for VLM visual inspection when `operation` includes inspection (e.g. silhouette clarity, transparency, framing, atlas sub-elements, mockup fidelity).",
    )
    include_concept_mockup: bool = Field(
        default=True,
        description="Whether to attach the active slice's 16:9 concept art mockup during VLM visual inspection for art-direction grounding.",
    )
    delete_paths: List[str] = Field(
        default_factory=list,
        description="Optional cartridge-relative `assets/...` media paths of obsolete/orphaned files to delete cleanly from disk and prune from `assets/asset_library.json`.",
    )
    reroute_to_specialist: str = Field(
        default="",
        description=(
            "Optional peer handoff: if this brief/issue belongs to another Specialist Persona (e.g. `'swe_engineer'` when "
            "the `.png` sprite/atlas on disk is clean and the bug is GDScript `AtlasTexture`/`region_rect`/`hframes` slicing or UI layout, "
            "`'audio_designer'` for audio, `'script_writer'` for narrative, or `'qa_test_engineer'` for runtime verification), "
            "set that specialist's name here and set `operation='reroute'`."
        ),
    )
    reroute_brief: str = Field(
        default="",
        description="When `reroute_to_specialist` is non-empty, concrete handoff brief explaining why you are re-routing and what the target specialist should inspect or modify.",
    )
    reroute_target_paths: List[str] = Field(
        default_factory=list,
        description="Optional cartridge-relative paths (e.g. `.gd`, `.tscn`, `.wav`) for `reroute_to_specialist`.",
    )
    operation: Literal["fabricate", "inspect", "fabricate_and_inspect", "render_contact_sheet", "cleanup", "reroute"] = Field(
        ...,
        description="Decision (strictly last): 'fabricate' to generate/update visual asset(s); 'inspect' to perceptually evaluate existing visual asset(s) via VLM + 1:1 diagnostic plate; 'fabricate_and_inspect' to generate asset(s) and immediately perceptually audit them; 'render_contact_sheet' for deterministic 1:1 plate rendering; 'cleanup' to delete obsolete `delete_paths` without fabricating; 'reroute' to hand off to `reroute_to_specialist`.",
    )


class VisualEvaluationReport(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Structured multimodal VLM perceptual inspection output from VisualEvaluatorAgent.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Detailed pixel-grounded visual deliberation examining silhouette, transparency/alpha cleanliness, proportions, sub-element layout, and cohesion with the Game Bible / 16:9 concept mockup.",
    )
    observed_visual_properties: str = Field(
        ...,
        description="Concrete description of what is actually visible in the rendered PNG pixels (orientation, sub-elements, palette, alpha boundaries).",
    )
    dimensional_and_framing_notes: str = Field(
        default="",
        description="Observations on pixel dimensions, padding, frame/cell layout, or region coordinates.",
    )
    style_and_mockup_alignment: str = Field(
        default="",
        description="Evaluation of how well the inspected visuals align with the game's art direction and 16:9 concept mockup.",
    )
    notable_concerns: List[str] = Field(
        default_factory=list,
        description="Specific pixel, alpha, framing, or style discrepancies observed (empty if clean).",
    )
    answer_to_inquiry: str = Field(
        default="",
        description="Direct answer to the `inspection_inquiry` grounded in the rendered pixels.",
    )
    perceptual_descriptions_by_path: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of relative asset path -> concise perceptual description persisted to `asset_library.json` so SWE/QA/Council know what each sprite/atlas visually contains.",
    )
    recommended_followups: List[str] = Field(
        default_factory=list,
        description="Concrete follow-up actions if visual defects, framing mismatches, or missing variants were observed.",
    )
    reroute_to_specialist: str = Field(
        default="",
        description=(
            "Optional post-inspection peer handoff: if pixel inspection proves the `.png` asset on disk is intact and the reported defect "
            "actually lies in another Specialist Persona's domain (e.g. `'swe_engineer'` for GDScript `AtlasTexture` region slicing, `hframes`/`vframes`, "
            "node scaling, or unbound texture loading), set that specialist's name here."
        ),
    )
    reroute_brief: str = Field(
        default="",
        description="When `reroute_to_specialist` is non-empty, concrete handoff brief citing the exact inspected PNG dimensions/cell layout and what the target specialist should fix.",
    )
    reroute_target_paths: List[str] = Field(
        default_factory=list,
        description="Optional cartridge-relative paths (e.g. `.gd` scripts) for `reroute_to_specialist`.",
    )
    answers_brief_satisfactorily: bool = Field(
        ...,
        description="Decision (strictly last): True if the inspected visual asset(s) satisfy the brief's visual and structural intent; False if defects or mismatches require rework.",
    )


class AudioDesignerPlan(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Internal tool-selection plan for the unified `audio_designer` persona (creator + perceptual audio/audiovisual evaluator).
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Rationale evaluating the delegated `brief` and target/media paths to decide whether to synthesize audio, perceptually listen/inspect existing `.wav`/`.mp4` media, or synthesize and immediately listen/inspect.",
    )
    self_question_answer: str = Field(
        default="",
        description="If you asked `self_question` on the previous turn, summarize the retrieved codebase/Bible/catalog facts before acting.",
    )
    self_question: str = Field(
        default="",
        description="Optional semantic inquiry against the codebase, Game Bible, or asset catalog when you need exact cue hooks, bus setup, or character voice specs before acting (leave '' when current context suffices).",
    )
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, summarize the retrieved acoustic/institutional lessons before acting.",
    )
    memory_query: str = Field(
        default="",
        description="Optional semantic search query against long-term memory banks before acting (leave '' when current context suffices).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter when issuing `memory_query`.",
    )
    tool_search_answer: str = Field(
        default="",
        description="If you issued `tool_search_query` on the previous turn, summarize the relevant audio synthesis/listening capabilities here.",
    )
    tool_search_query: str = Field(
        default="",
        description="Optional progressive tool discovery query for audio synthesis modes or multimodal listening options (leave '' when current context suffices).",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="DEFAULT TO [] — LEAVE EMPTY ON ROUTINE TURNS. Populate ONLY for non-obvious cross-slice acoustic/mixing invariants.",
    )
    synthesis_notes: str = Field(
        default="",
        description="Optional synthesis/mixing guidance when `operation` includes synthesis.",
    )
    media_paths_to_inspect: List[str] = Field(
        default_factory=list,
        description="Relative `.wav` or `.mp4` paths to listen to via AudioEvaluatorAgent when `operation` is 'inspect' or 'synthesize_and_inspect'.",
    )
    inspection_inquiry: str = Field(
        default="",
        description="Specific acoustic/audiovisual question(s) to answer when listening to the media.",
    )
    delete_paths: List[str] = Field(
        default_factory=list,
        description="Optional cartridge-relative `assets/...` media paths of obsolete/orphaned audio files to delete cleanly from disk and prune from `assets/asset_library.json`.",
    )
    reroute_to_specialist: str = Field(
        default="",
        description=(
            "Optional peer handoff: if this brief/issue belongs to another Specialist Persona (e.g. `'swe_engineer'` for "
            "`AudioStreamPlayer` bus/signal trigger wiring in GDScript, `'script_writer'` for dialogue scripts, `'visual_artist'` for visuals, "
            "or `'qa_test_engineer'` for runtime capture), set that specialist's name here and set `operation='reroute'`."
        ),
    )
    reroute_brief: str = Field(
        default="",
        description="When `reroute_to_specialist` is non-empty, concrete handoff brief explaining what the target specialist should inspect or modify.",
    )
    reroute_target_paths: List[str] = Field(
        default_factory=list,
        description="Optional cartridge-relative paths for `reroute_to_specialist`.",
    )
    operation: Literal["synthesize", "inspect", "synthesize_and_inspect", "cleanup", "reroute"] = Field(
        ...,
        description="Decision (strictly last): 'synthesize' to fabricate audio asset(s); 'inspect' to perceptually listen to existing `.wav`/`.mp4` media; 'synthesize_and_inspect' to fabricate audio and immediately verify it by listening; 'cleanup' to delete obsolete `delete_paths` without synthesizing; 'reroute' to hand off to `reroute_to_specialist`.",
    )


class TaskLeadDecision(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    One turn of the Task Lead Agent's iterative Observe -> Strategize -> Delegate -> Inspect loop.
    """
    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Concise (<= 120 words) evaluation of the task requirements, current cartridge state, prior Council feedback (if any), and the results returned by specialists on previous turns.",
    )
    self_question_answer: str = Field(
        default="",
        description="If you asked a `self_question` on the previous turn, write your concise, factual answer here FIRST based on the retrieved context chunks.",
    )
    self_question: str = Field(
        default="",
        description="Optional self-inquiry: If you need more context from the codebase, Game Bible, or backlog before deciding how to delegate or submit, output a specific question here. ContextCuratorAgent will fetch the needed chunks so you can answer your own question on the next turn.",
    )
    memory_query_answer: str = Field(
        default="",
        description="If you issued `memory_query` on the previous turn, write your concise synthesis of the retrieved long-term memory entries here.",
    )
    memory_query: str = Field(
        default="",
        description="Optional direct query against long-term game/agent memory banks: if you want to look up cross-slice strategic goals, prior architectural decisions, or operational lessons learned before delegating or submitting, output a semantic search query here (leaving `brief` empty).",
    )
    memory_query_bank: str = Field(
        default="",
        description="Optional bank filter when issuing `memory_query` ('', 'all', 'self', 'council', or a canonical specialist role name).",
    )
    tool_search_answer: str = Field(
        default="",
        description="If you issued `tool_search_query` on the previous turn, write your concise synthesis of the retrieved Studio tool capabilities/modes here before delegating, spawning, splitting, or submitting.",
    )
    tool_search_query: str = Field(
        default="",
        description="Optional progressive tool discovery query: if you want detailed schemas, internal sub-tools, parameters, or composition strategies for any Specialist Persona or board action before acting, output a semantic search query here.",
    )
    memory_updates: List[MemoryMutationProposal] = Field(
        default_factory=list,
        description="DEFAULT TO [] — LEAVE EMPTY ON ~95% OF TURNS. Never log task progress, delegation status, or facts already in the Game Bible/codebase. Populate ONLY when a costly cross-task failure mode, operational constraint, or deferred multi-slice goal would otherwise be lost (or to update/fulfill an existing memory_id).",
    )
    current_strategy: str = Field(
        ...,
        description="Concise (<= 60 words) strategy for this turn (state explicitly whether this turn will 'delegate' to a specific specialist_agent, 'spawn_tasks', 'split_task', or 'submit_to_council').",
    )
    proof_strategy: str = Field(
        default="",
        description="Concise (<= 40 words) strategy for how the final proof artifact(s) or discovery findings will be generated and verified.",
    )
    specialist_agent: str = Field(
        default="",
        description="REQUIRED non-empty specialist identifier whenever this turn delegates work ('visual_artist', 'audio_designer', 'script_writer', 'swe_engineer', or 'qa_test_engineer'). Because this field appears BEFORE `action` in the JSON schema, you MUST populate `specialist_agent` here whenever you will set `action` to 'delegate'!",
    )
    brief: str = Field(
        default="",
        description="REQUIRED non-empty natural-language instructions (<= 180 words) for `specialist_agent` whenever this turn delegates work. Name the target files, symbols, and behavioral/visual/acoustic creation or inspection goals concisely—NEVER write raw GDScript code blocks inside `brief`! Because `brief` appears BEFORE `action` in the JSON schema, you MUST write the specialist brief here whenever you will set `action` to 'delegate'!",
    )
    target_paths: List[str] = Field(
        default_factory=list,
        description="Optional file paths or GDScript modules the specialist should create, modify, or inspect (e.g. .gd files, .png sprites, .wav cues, or .mp4 gameplay clips).",
    )
    submission_summary: str = Field(
        default="",
        description="When action == 'submit_to_council', comprehensive summary explaining the strategy executed, specialist contributions, and how acceptance criteria or discovery questions were satisfied.",
    )
    discovery_findings: str = Field(
        default="",
        description="When completing a `kind == 'discovery'` task (or surfacing structured investigation findings alongside a build submission), write the concrete, evidence-backed findings citing exact file paths, dimensions, node properties, or visual/audio observations.",
    )
    selected_proof_paths: List[str] = Field(
        default_factory=list,
        description="When action == 'submit_to_council', paths of the proof or diagnostic artifacts (.png, .wav, and/or .mp4) generated during this session.",
    )
    spawned_tasks: List[WorkItemProposal] = Field(
        default_factory=list,
        description="When action == 'spawn_tasks', new tasks dynamically reasoned out while working on this task—including `kind='discovery'` investigation subtasks or prerequisite/follow-up build subtasks.",
    )
    wait_for_spawned_tasks: bool = Field(
        default=True,
        description="When action == 'spawn_tasks', True if the current task depends on the newly spawned tasks completing first (yields locks and resumes automatically with child findings/artifacts injected); False if they are independent follow-up/parallel tasks.",
    )
    split_tasks: List[WorkItemProposal] = Field(
        default_factory=list,
        description="When action == 'split_task', 2 or more cohesive child WorkItemProposals that decompose and replace the current oversized or multi-concern task (may declare intra-batch `depends_on` ordering; downstream board dependents automatically rewire to wait on these children).",
    )
    preserve_verified_changes_on_split: bool = Field(
        default=False,
        description="When action == 'split_task', False (default) cleanly reverts uncommitted staging edits from this partial session before splitting; True preserves already-verified disk changes made in earlier turns of this session.",
    )
    design_concern_report: str = Field(
        default="",
        description=(
            "When action == 'escalate_design_concern', concrete co-designer report explaining why this task's spec as written "
            "would force fake/Potemkin props, self-completing t=0 automation, zero-decision gameplay, or irreconcilable rules "
            "with existing systems—asking the Council to revise or split the spec cleanly rather than shipping a dishonest stub."
        ),
    )
    action: Literal["delegate", "spawn_tasks", "split_task", "escalate_design_concern", "submit_to_council"] = Field(
        ...,
        description="Decision (strictly last): 'delegate' to invoke a Specialist Persona (for creation or inspection); 'spawn_tasks' to add discovery or build subtasks to the Production Board; 'split_task' to decompose this task into 2..N smaller cohesive tasks; 'escalate_design_concern' to kick a fun/agency-breaking or contradictory spec back to the Council; 'submit_to_council' when work/findings are complete.",
    )



class LevelEntitySpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A single concrete entity (landmark, mechanism, spawn point, hazard, goal, region) placed in the level.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Concise 1-sentence rationale (<= 18 words) for this entity's placement and role.")
    entity_id: str = Field(..., description="Unique snake_case identifier referenced by triggers and beats.")
    entity_kind: str = Field(..., description="Free-form role such as 'spawn', 'mechanism', 'landmark', 'region', 'goal', 'hazard', 'collectible'.")
    position: Tuple[float, float] = Field(..., description="World-space (x, y) centre in the game's coordinate system.")
    size: Tuple[float, float] = Field(default=(0.0, 0.0), description="World-space (width, height) extent; (0, 0) for point entities.")
    visual_description: str = Field(default="", description="What this entity must look like when rendered.")
    asset_id: str = Field(default="", description="Associated asset id from the asset library, if any.")
    initial_state: str = Field(default="dormant", description="State at t=0 (e.g. 'dormant', 'locked', 'active').")


class LevelTriggerSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A causal relation: when `condition` holds on `source_entity_id`, `target_entity_id` transitions to `resulting_state`.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Concise 1-sentence rationale (<= 18 words) for this causal link.")
    trigger_id: str = Field(..., description="Unique trigger identifier.")
    source_entity_id: str = Field(..., description="Entity whose interaction/state change fires the trigger.")
    condition: str = Field(..., description="Precise, testable condition (e.g. 'player overlaps region while speed >= threshold').")
    target_entity_id: str = Field(..., description="Entity whose state changes in response.")
    resulting_state: str = Field(..., description="State the target must enter, observable by the game.")
    delay_seconds: float = Field(default=0.0, description="Delay between firing and the observable transition.")


class LevelRegionSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    A named area of the world. The concept-art composition is ONE region (the opening), not the whole world.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Concise 1-sentence rationale (<= 20 words) for this region's bounds and progression role.")
    region_id: str = Field(..., description="Unique snake_case identifier.")
    bounds: Tuple[float, float, float, float] = Field(..., description="(x, y, width, height) world rectangle of this region.")
    progression_order: int = Field(default=0, description="0 = opening region; increasing values are reached later in the experience.")
    purpose: str = Field(default="", description="What the player does / experiences here.")
    archetype_id: str = Field(default="", description="Matching LevelArchetypeSpec.archetype_id from level_generation_strategy, if applicable.")
    procedural_chunk_grammar: str = Field(
        default="",
        description="When generation_approach is procedural or hybrid, rules/parameters used by scenes.gd to populate this region with dense terrain, platforms, hazards, and collectibles at runtime.",
    )
    target_chunk_or_feature_count: int = Field(
        default=1,
        description="Number of procedural chunks or repeated structural modules to generate inside this region at runtime.",
    )


class LevelLayoutSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Machine-readable level layout consumed as DATA by the SWE agent (instead of improvising layout from prose).
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Spatial reasoning translating the Game Bible (scope, progression, storyboard, concept art) into a concrete world layout.")
    viewport_size: Tuple[float, float] = Field(default=(1280.0, 720.0), description="(width, height) of one camera viewport screen in world units.")
    intended_world_scale_in_viewports: float = Field(default=0.0, description="Total playable area expressed as a multiple of one viewport's area, justified by the Game Bible's level_generation_strategy and progression (a 16:9 concept mockup is a single camera frame inside the world, NOT the bounding box of the entire world unless a fixed single-screen stage is explicitly specified).")
    world_bounds: Tuple[float, float, float, float] = Field(..., description="(x, y, width, height) bounding rectangle of the ENTIRE playable world; must contain every region.")
    world_regions: List[LevelRegionSpec] = Field(default_factory=list, description="Ordered regions covering the Game Bible's progression across its level archetypes (when revising an existing layout, you may emit only added/modified regions—unchanged regions are preserved automatically by region_id).")
    camera_framing: str = Field(..., description="Camera behaviour (e.g. Camera2D following the player across world_bounds with HUD isolated on a CanvasLayer, or fixed stage camera when single-screen).")
    entities: List[LevelEntitySpec] = Field(default_factory=list, description="Entities with exact positions and sizes (when revising an existing layout, you may emit only added/modified entities—unchanged entities are preserved automatically by entity_id).")
    triggers: List[LevelTriggerSpec] = Field(default_factory=list, description="Causal relations between entities (when revising an existing layout, you may emit only added/modified triggers—unchanged triggers are preserved automatically by trigger_id).")
    remove_entity_ids: List[str] = Field(default_factory=list, description="Optional entity_ids to remove when revising an existing layout.")
    remove_trigger_ids: List[str] = Field(default_factory=list, description="Optional trigger_ids to remove when revising an existing layout.")


def merge_level_layout(base: "LevelLayoutSpec", patch: "LevelLayoutSpec") -> "LevelLayoutSpec":
    """Incrementally merges `patch` onto `base` by `region_id`, `entity_id`, and `trigger_id` so revisions
    do not need to re-emit unchanged regions, entities, or triggers."""
    bx0, by0, bw0, bh0 = base.world_bounds
    bx1, by1, bw1, bh1 = patch.world_bounds
    if bw1 > 0 and bh1 > 0 and bw0 > 0 and bh0 > 0:
        min_x = min(bx0, bx1)
        min_y = min(by0, by1)
        max_x = max(bx0 + bw0, bx1 + bw1)
        max_y = max(by0 + bh0, by1 + bh1)
        merged_bounds = (min_x, min_y, max_x - min_x, max_y - min_y)
    elif bw1 > 0 and bh1 > 0:
        merged_bounds = patch.world_bounds
    else:
        merged_bounds = base.world_bounds

    regions_by_id: Dict[str, LevelRegionSpec] = {r.region_id: r for r in base.world_regions}
    for r in patch.world_regions:
        regions_by_id[r.region_id] = r
    merged_regions = sorted(regions_by_id.values(), key=lambda r: r.progression_order)

    remove_e = set(patch.remove_entity_ids or [])
    entities_by_id: Dict[str, LevelEntitySpec] = {
        e.entity_id: e for e in base.entities if e.entity_id not in remove_e
    }
    for e in patch.entities:
        entities_by_id[e.entity_id] = e
    merged_entities = list(entities_by_id.values())

    remove_t = set(patch.remove_trigger_ids or [])
    triggers_by_id: Dict[str, LevelTriggerSpec] = {
        t.trigger_id: t for t in base.triggers if t.trigger_id not in remove_t and t.source_entity_id not in remove_e and t.target_entity_id not in remove_e
    }
    for t in patch.triggers:
        triggers_by_id[t.trigger_id] = t
    merged_triggers = list(triggers_by_id.values())

    return LevelLayoutSpec(
        deliberation=patch.deliberation or base.deliberation,
        viewport_size=patch.viewport_size if (patch.viewport_size[0] > 0 and patch.viewport_size[1] > 0) else base.viewport_size,
        intended_world_scale_in_viewports=max(base.intended_world_scale_in_viewports, patch.intended_world_scale_in_viewports),
        world_bounds=merged_bounds,
        world_regions=merged_regions,
        camera_framing=patch.camera_framing or base.camera_framing,
        entities=merged_entities,
        triggers=merged_triggers,
    )


def validate_level_layout(
    layout: "LevelLayoutSpec",
    strategy: Optional["LevelGenerationStrategy"] = None,
) -> List[str]:
    """Structural validation of a LevelLayoutSpec. Returns human-readable issues (empty list = valid).
    Purely structural/arithmetic consistency checks against the model's OWN declared intent; no content heuristics.
    """
    issues: List[str] = []
    bx, by, bw, bh = layout.world_bounds
    vw, vh = layout.viewport_size
    if bw <= 0 or bh <= 0 or vw <= 0 or vh <= 0:
        issues.append("world_bounds and viewport_size must have positive width/height.")
        return issues

    def inside(rect_or_pt, outer, label):
        ox, oy, ow, oh = outer
        if len(rect_or_pt) == 2:
            x, y = rect_or_pt; w = h = 0.0
        else:
            x, y, w, h = rect_or_pt
        if x - w / 2 < ox or y - h / 2 < oy or x + w / 2 > ox + ow or y + h / 2 > oy + oh:
            issues.append(f"{label} extends outside {outer}.")

    actual_scale = (bw * bh) / (vw * vh)
    declared = layout.intended_world_scale_in_viewports
    if declared <= 0:
        issues.append("intended_world_scale_in_viewports must be declared (> 0) and justified by the Game Bible's scope.")
    elif actual_scale < declared * 0.8:
        issues.append(
            f"world_bounds cover only {actual_scale:.1f} viewports but intended_world_scale_in_viewports={declared:.1f}."
        )
    if not layout.world_regions:
        issues.append("world_regions is empty; the layout must cover at least one world region.")
    ids = {e.entity_id for e in layout.entities}
    if len(ids) != len(layout.entities):
        issues.append("Duplicate entity_id values.")
    for r in layout.world_regions:
        rx, ry, rw, rh = r.bounds
        if rx < bx or ry < by or rx + rw > bx + bw or ry + rh > by + bh:
            issues.append(f"Region '{r.region_id}' extends outside world_bounds.")
    for e in layout.entities:
        inside((e.position[0], e.position[1], e.size[0], e.size[1]), layout.world_bounds, f"Entity '{e.entity_id}'")
    for t in layout.triggers:
        for ref in (t.source_entity_id, t.target_entity_id):
            if ref not in ids:
                issues.append(f"Trigger '{t.trigger_id}' references unknown entity '{ref}'.")
        if t.source_entity_id == t.target_entity_id:
            issues.append(f"Trigger '{t.trigger_id}' targets its own source entity.")
    return issues


class EntityStateExpectation(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    One expected (entity, state) pair. A list of these replaces a free-form dict, which structured-output schemas handle poorly.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Concise 1-sentence rationale (<= 15 words) for this expected state.")
    entity_id: str = Field(..., description="entity_id from the level layout.")
    expected_state: str = Field(..., description="State the entity must be in.")


class BeatStepSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    One timed step of an authored beat: scripted input then an expected observable state.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Concise 1-sentence rationale (<= 18 words) for why this input produces this expected state.")
    step_id: str = Field(..., description="Unique step identifier.")
    scripted_input: str = Field(..., description="Exact input to apply (actions held, frames/duration), phrased so a headless test can execute it.")
    duration_frames: int = Field(default=30, description="Physics frames to simulate for this step.")
    expected_entity_states: List["EntityStateExpectation"] = Field(default_factory=list, description="Entity states (by entity_id) that must hold after this step.")


class BeatTestSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Executable specification of one authored experience beat, derived from the storyboard BEFORE implementation.
    """
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: Why this beat is the correct falsifiable translation of the storyboard beat.")
    beat_id: str = Field(..., description="Unique beat identifier, e.g. 'beat_1'.")
    storyboard_beat_index: int = Field(default=0, description="Index of the storyboard beat this verifies.")
    steps: List[BeatStepSpec] = Field(default_factory=list, description="Ordered scripted steps with expected states.")


class BeatPlanSpec(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4)."""
    deliberation: str = Field(..., description="MANDATORY FIRST PROPERTY: How the storyboard maps onto executable beat tests.")
    beats: List[BeatTestSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Context Hygiene & Transient Gemini Resilience Helpers
# ---------------------------------------------------------------------------

_BIBLE_HISTORICAL_KEYS = frozenset({
    "writers_room_transcript",
    "concept_fragments",
    "concept_candidates",
    "concept_evolution",
    "final_signoff_report",
    "production_board",
    "created_at_utc",
    "updated_at_utc",
})

_BIBLE_VISUAL_ASSET_KEYS = frozenset({
    "game_title",
    "core_genre",
    "elevator_pitch",
    "core_fantasy",
    "camera_and_spatial_paradigm",
    "color_palette_hex",
    "visual_style_guide",
    "diegetic_ui_philosophy",
    "scene_layers",
    "visual_mockups",
    "world_spec",
    "slice_roadmap",
    "active_slice_index",
})

_BIBLE_AUDIO_ASSET_KEYS = frozenset({
    "game_title",
    "core_genre",
    "elevator_pitch",
    "core_fantasy",
    "audio_direction_guide",
    "first_60_seconds_hook",
    "gameplay_pillars",
    "experience_storyboard",
    "world_spec",
    "slice_roadmap",
    "active_slice_index",
})

_BIBLE_RENDERING_KEYS = frozenset({
    "game_title",
    "core_genre",
    "elevator_pitch",
    "core_fantasy",
    "camera_and_spatial_paradigm",
    "color_palette_hex",
    "visual_style_guide",
    "scene_layers",
    "visual_mockups",
    "gameplay_pillars",
    "first_60_seconds_hook",
    "diegetic_ui_philosophy",
    "level_generation_strategy",
    "experience_storyboard",
    "world_spec",
    "slice_roadmap",
    "active_slice_index",
})


def _strip_nested_deliberations(obj: Any) -> Any:
    """Recursively strips historical per-object `deliberation` fields and empty placeholders from a serialized Game Bible dict."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k == "deliberation":
                continue
            cleaned_v = _strip_nested_deliberations(v)
            if cleaned_v not in (None, "", [], {}):
                out[k] = cleaned_v
        return out
    if isinstance(obj, list):
        items = [_strip_nested_deliberations(x) for x in obj]
        return [x for x in items if x not in (None, "", [], {})]
    return obj


def compact_bible_json_for_prompt(raw_bible_text: str, *, mode: str = "full") -> str:
    """Compacts a raw `game_bible.json` string for LLM prompts by stripping historical brainstorming bloat
    (`writers_room_transcript`, `concept_fragments`, `concept_candidates`, `concept_evolution`, nested `deliberation` essays)
    while preserving 100% of the authoritative design specification needed for `mode` ('full' | 'rendering' | 'visual_asset' | 'audio_asset').
    Returns `raw_bible_text` unchanged if it is not a JSON object.
    """
    if not raw_bible_text or not raw_bible_text.strip():
        return raw_bible_text
    try:
        parsed = json.loads(raw_bible_text)
    except Exception:
        return raw_bible_text
    if not isinstance(parsed, dict) or not parsed:
        return raw_bible_text

    cleaned: Dict[str, Any] = {}
    allowed_keys: Optional[frozenset] = None
    if mode == "visual_asset":
        allowed_keys = _BIBLE_VISUAL_ASSET_KEYS
    elif mode == "audio_asset":
        allowed_keys = _BIBLE_AUDIO_ASSET_KEYS
    elif mode == "rendering":
        allowed_keys = _BIBLE_RENDERING_KEYS

    for k, v in parsed.items():
        if k in _BIBLE_HISTORICAL_KEYS or k == "deliberation":
            continue
        if allowed_keys is not None and k not in allowed_keys:
            if k == "narrative_bible" and isinstance(v, dict):
                chars = v.get("characters") or []
                if chars and mode in ("visual_asset", "rendering"):
                    cleaned["characters"] = [
                        {
                            ck: cv
                            for ck, cv in c.items()
                            if ck in ("character_id", "name", "role", "visual_description", "portrait_asset") and cv
                        }
                        for c in chars
                        if isinstance(c, dict)
                    ]
                elif chars and mode == "audio_asset":
                    cleaned["characters"] = [
                        {
                            ck: cv
                            for ck, cv in c.items()
                            if ck in ("character_id", "name", "role", "voice_profile") and cv
                        }
                        for c in chars
                        if isinstance(c, dict)
                    ]
                    barks = v.get("dialogue_barks") or []
                    if barks:
                        cleaned["dialogue_barks"] = _strip_nested_deliberations(barks[:16])
            continue
        stripped_v = _strip_nested_deliberations(v)
        if stripped_v not in (None, "", [], {}):
            cleaned[k] = stripped_v

    if not cleaned:
        return raw_bible_text
    return json.dumps(cleaned, ensure_ascii=False)


def bounded_thinking_config(budget: int = 2048) -> Any:
    """Returns a `google.genai.types.ThinkingConfig(thinking_budget=budget)` when supported by the SDK,
    preventing unbounded internal reasoning stalls on structured schemas that already generate `deliberation` first.
    """
    try:
        from google.genai import types
        thinking_cls = getattr(types, "ThinkingConfig", None)
        if thinking_cls is not None:
            return thinking_cls(thinking_budget=int(budget))
    except Exception:
        pass
    return None


def is_transient_gemini_error(exc: BaseException) -> bool:
    """Returns True if `exc` is a transient upstream deadline, rate-limit, overload, DNS, socket,
    or bare generic 'Request contains an invalid argument.' backend glitch."""
    msg = f"{type(exc).__name__}: {exc}".upper()
    return any(
        token in msg
        for token in (
            "504",
            "DEADLINE_EXCEEDED",
            "DEADLINE EXPIRED",
            "503",
            "UNAVAILABLE",
            "502",
            "BAD GATEWAY",
            "500",
            "INTERNAL",
            "429",
            "RESOURCE_EXHAUSTED",
            "REQUEST CONTAINS AN INVALID ARGUMENT",
            "READTIMEOUT",
            "CONNECTTIMEOUT",
            "TIMED OUT",
            "NODENAME NOR SERVNAME",
            "TEMPORARY FAILURE IN NAME RESOLUTION",
            "NAME OR SERVICE NOT KNOWN",
            "GAIERROR",
            "[ERRNO 8]",
            "CONNECTERROR",
            "REMOTEPROTOCOLERROR",
            "SERVERDISCONNECTED",
            "CONNECTION RESET",
            "CONNECTIONRESETERROR",
            "BROKEN PIPE",
            "UNEXPECTED_EOF_WHILE_READING",
        )
    )


class StudioPausedError(StudioAgentError):
    """Raised when an outbound LLM call is blocked or aborted because the Studio is paused."""
    deliberation: str = "Studio execution paused; outbound LLM request aborted."


def is_studio_paused(candidate_dir: Optional[Path] = None) -> bool:
    """Returns True when the Studio is paused via `.studio_paused.json` or `swarm_bus._build_paused`.
    During pytest runs, only checks an explicit `candidate_dir` pause sentinel or `STUDIO_ENFORCE_GLOBAL_PAUSE=1`
    so unrelated unit tests do not block on a developer's local `cartridges/.studio_paused.json`.
    """
    import json as _json
    import os as _os
    import sys as _sys

    in_pytest = bool("PYTEST_CURRENT_TEST" in _os.environ or "pytest" in _sys.modules)
    enforce_global = not in_pytest or _os.environ.get("STUDIO_ENFORCE_GLOBAL_PAUSE") == "1"

    if candidate_dir is not None:
        c_path = Path(candidate_dir)
        for p_file in (c_path / ".studio_paused.json", c_path.parent / ".studio_paused.json"):
            if p_file.is_file():
                try:
                    data = _json.loads(p_file.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and data.get("paused") is True:
                        return True
                except Exception:
                    pass

    if enforce_global:
        try:
            import config as _cfg
            global_pause = _cfg.PROJECT_ROOT / "cartridges" / ".studio_paused.json"
            if global_pause.is_file():
                data = _json.loads(global_pause.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("paused") is True:
                    return True
        except Exception:
            pass
        try:
            from engine.ui.server import swarm_bus as _sb
            if getattr(_sb, "_build_paused", False):
                return True
        except Exception:
            pass

    return False


def _check_gateway_abort(candidate_dir: Optional[Path] = None, label: str = "generate_content") -> None:
    """Raises `StudioPausedError` immediately if the active orchestrator or `candidate_dir` has been aborted/preempted."""
    if getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "interactive_foreground_turn", False):
        return
    if is_candidate_build_aborted(candidate_dir):
        raise StudioPausedError(
            f"Aborted outbound Gemini API call '{label}' because build for '{Path(candidate_dir).name if candidate_dir else 'candidate'}' was cancelled by a new game request."
        )
    try:
        from engine.ui.server import swarm_bus as _sb
        orch = getattr(_sb, "_active_orchestrator", None)
        if orch is not None and getattr(orch, "_abort_requested", False):
            raise StudioPausedError(
                f"Aborted outbound Gemini API call '{label}' because active build was cancelled."
            )
    except StudioPausedError:
        raise
    except Exception:
        pass


def is_interactive_foreground_turn() -> bool:
    """Returns True when the current thread is executing an interactive user-initiated foreground turn."""
    return bool(getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "interactive_foreground_turn", False))


def interactive_foreground_call_scope():
    """Marks the current thread as an interactive user-initiated foreground turn (e.g. Co-Producer chat)
    so the user can converse, inspect, consult specialists, and sketch concepts even while background
    autonomous cartridge builds are paused or stopped."""
    import contextlib as _contextlib

    @_contextlib.contextmanager
    def _scope():
        prev = getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "interactive_foreground_turn", False)
        _THREAD_LOCAL_GEMINI_CALL_CTX.interactive_foreground_turn = True
        try:
            yield
        finally:
            _THREAD_LOCAL_GEMINI_CALL_CTX.interactive_foreground_turn = prev

    return _scope()


def wait_if_studio_paused(
    candidate_dir: Optional[Path] = None,
    *,
    label: str = "generate_content",
    fail_fast: Optional[bool] = None,
) -> float:
    """Gateway pause gate enforced before outbound Gemini calls and inside long-running SWE/PEV loops.
    - If `fail_fast=True` or `STUDIO_FAIL_FAST_ON_PAUSE=1`, immediately raises `StudioPausedError`.
    - Otherwise blocks in 0.2s slices while `is_studio_paused(candidate_dir)` is True, raising
      `StudioPausedError` immediately if the active orchestrator requests an abort.
    """
    import os as _os
    import time as _time

    if getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "interactive_foreground_turn", False):
        return 0.0

    should_fail_fast = (
        fail_fast
        if fail_fast is not None
        else (_os.environ.get("STUDIO_FAIL_FAST_ON_PAUSE") == "1")
    )
    _check_gateway_abort(candidate_dir, label=label)
    t0 = _time.monotonic()
    while is_studio_paused(candidate_dir):
        if should_fail_fast:
            raise StudioPausedError(
                f"Blocked outbound Gemini API call '{label}' because Studio build is PAUSED (.studio_paused.json)."
            )
        _check_gateway_abort(candidate_dir, label=label)
        _time.sleep(0.2)
    _check_gateway_abort(candidate_dir, label=label)
    return max(0.0, _time.monotonic() - t0)


class TokenUsageTracker:
    """Thread-safe tracker for Gemini API `usage_metadata` including implicit prefix cache hits
    and a hard runaway-loop circuit breaker."""
    deliberation: str = "Tracks cumulative prompt, cached-prefix, output, and thinking token metrics across studio LLM calls."

    def __init__(self) -> None:
        import threading as _threading
        self._lock = _threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.total_calls: int = 0
            self.calls_with_cache_hit: int = 0
            self.total_prompt_tokens: int = 0
            self.total_cached_tokens: int = 0
            self.total_output_tokens: int = 0
            self.total_thought_tokens: int = 0
            self.total_tokens: int = 0
            self.image_generations: int = 0
            self.image_output_tokens_deducted: int = 0
            self.explicit_caches_created: int = 0
            self.cache_storage_token_hours: float = 0.0
            self.tts_generations: int = 0
            self.tts_audio_seconds: float = 0.0
            self.embedding_calls: int = 0
            self.embedding_estimated_tokens: int = 0
            self.playtest_calls: int = 0
            self.playtest_prompt_tokens: int = 0
            self.playtest_cached_tokens: int = 0
            self.playtest_output_tokens: int = 0
            self.playtest_thought_tokens: int = 0
            self.playtest_tokens: int = 0
            self.by_label: Dict[str, Dict[str, Any]] = {}
            self.consecutive_swe_pev_calls: int = 0
            self._recorded_response_ids: List[int] = []

    def reset_swe_pev_streak(self) -> None:
        with self._lock:
            self.consecutive_swe_pev_calls = 0

    def check_circuit_breaker(self, label: str = "generate_content", candidate_dir: Optional[Path] = None) -> None:
        """Trips a hard circuit breaker if a single SWE PEV / scene critique loop fires an unbounded
        streak of consecutive calls without advancing a board task or checkpoint."""
        import json as _json
        import os as _os
        from datetime import datetime as _dt, timezone as _tz

        max_streak = int(_os.environ.get("STUDIO_MAX_CONSECUTIVE_SWE_CALLS", "36"))
        low = (label or "").lower()
        is_swe_loop_call = any(
            tok in low
            for tok in ("pev_phase", "scene_critique", "analyze_scene", "debug_and_patch")
        )
        with self._lock:
            if is_swe_loop_call:
                self.consecutive_swe_pev_calls += 1
                streak = self.consecutive_swe_pev_calls
            else:
                self.consecutive_swe_pev_calls = 0
                streak = 0

        if max_streak > 0 and streak > max_streak:
            # Trip circuit breaker: auto-pause studio on disk and in memory so no further calls can fire
            err_msg = (
                f"Runaway LLM call circuit breaker tripped: '{label}' exceeded {max_streak} consecutive SWE/PEV calls without task progression."
            )
            now_iso = _dt.now(_tz.utc).isoformat()
            try:
                import config as _cfg
                in_pytest = "PYTEST_CURRENT_TEST" in _os.environ
                target_pause = (
                    (Path(candidate_dir).parent / ".studio_paused.json")
                    if candidate_dir is not None
                    else (None if in_pytest else (_cfg.PROJECT_ROOT / "cartridges" / ".studio_paused.json"))
                )
                if target_pause is not None:
                    target_pause.parent.mkdir(parents=True, exist_ok=True)
                    target_pause.write_text(
                        _json.dumps({
                            "paused": True,
                            "reason": err_msg,
                            "timestamp_utc": now_iso,
                        }),
                        encoding="utf-8",
                    )
                if not in_pytest:
                    from engine.ui.server import swarm_bus as _sb
                    if _sb is not None and hasattr(_sb, "publish_threadsafe"):
                        _sb.publish_threadsafe({
                            "deliberation": err_msg,
                            "event_type": "ERROR",
                            "agent_name": (label or "StudioCircuitBreaker").split(".", 1)[0],
                            "timestamp_utc": now_iso,
                            "payload": {
                                "error": err_msg,
                                "reason": err_msg,
                                "label": label,
                                "streak": streak,
                            },
                        })
            except Exception:
                pass
            raise StudioPausedError(err_msg)

    @staticmethod
    def _compute_cost_usd(
        *,
        uncached_prompt_tokens: int,
        cached_tokens: int,
        output_and_thought_tokens: int,
        image_generations: int = 0,
        embedding_tokens: int = 0,
        tts_audio_seconds: float = 0.0,
        cache_storage_token_hours: float = 0.0,
        standard_2027: bool = False,
    ) -> float:
        import os as _os

        if standard_2027:
            in_rate = 1.50
            cached_rate = 0.15
            out_rate = 7.50
        else:
            in_rate = float(_os.environ.get("STUDIO_PRICE_INPUT_PER_1M", "0.75"))
            cached_rate = float(_os.environ.get("STUDIO_PRICE_CACHED_PER_1M", "0.075"))
            out_rate = float(_os.environ.get("STUDIO_PRICE_OUTPUT_PER_1M", "3.75"))
        img_rate = float(_os.environ.get("STUDIO_PRICE_IMAGE_EACH", "0.039"))
        embed_rate = float(_os.environ.get("STUDIO_PRICE_EMBED_PER_1M", "0.15"))
        tts_per_min = float(_os.environ.get("STUDIO_PRICE_TTS_PER_MIN", "0.015"))
        storage_rate = float(_os.environ.get("STUDIO_PRICE_CACHE_STORAGE_PER_1M_HR", "1.00"))

        usd = (
            (max(0, uncached_prompt_tokens) / 1_000_000.0) * in_rate
            + (max(0, cached_tokens) / 1_000_000.0) * cached_rate
            + (max(0, output_and_thought_tokens) / 1_000_000.0) * out_rate
            + max(0, image_generations) * img_rate
            + (max(0, embedding_tokens) / 1_000_000.0) * embed_rate
            + (max(0.0, tts_audio_seconds) / 60.0) * tts_per_min
            + (max(0.0, cache_storage_token_hours) / 1_000_000.0) * storage_rate
        )
        return round(usd, 6)

    def record_explicit_cache_lifecycle(
        self,
        cached_tokens: int,
        lifetime_seconds: float,
    ) -> float:
        """Records an explicit `CachedContent` storage lifecycle (tokens * seconds / 3600) on scope teardown."""
        tok_hrs = (max(0, int(cached_tokens)) * max(1.0, float(lifetime_seconds))) / 3600.0
        with self._lock:
            self.explicit_caches_created += 1
            self.cache_storage_token_hours = round(self.cache_storage_token_hours + tok_hrs, 6)
        return tok_hrs

    def record_response(
        self,
        response: Any,
        label: str = "generate_content",
        *,
        input_chars_hint: int = 0,
        method_name: str = "generate_content",
    ) -> Optional[Dict[str, Any]]:
        if response is None or getattr(response, "_studio_usage_recorded", False):
            return None
        marked_on_obj = False
        try:
            setattr(response, "_studio_usage_recorded", True)
            marked_on_obj = True
        except Exception:
            pass
        if not marked_on_obj:
            resp_id = id(response)
            with self._lock:
                if resp_id in self._recorded_response_ids:
                    return None
                self._recorded_response_ids.append(resp_id)
                if len(self._recorded_response_ids) > 256:
                    del self._recorded_response_ids[:-256]
        usage = getattr(response, "usage_metadata", None)
        interaction_usage = getattr(response, "usage", None) if usage is None else None
        if usage is not None:
            prompt_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
            cached_tok = int(getattr(usage, "cached_content_token_count", 0) or 0)
            output_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
            thought_tok = int(getattr(usage, "thoughts_token_count", 0) or 0)
            total_tok = int(getattr(usage, "total_token_count", 0) or (prompt_tok + output_tok + thought_tok))
        elif interaction_usage is not None:
            prompt_tok = int(
                getattr(interaction_usage, "total_input_tokens", None)
                or getattr(interaction_usage, "input_tokens", 0)
                or 0
            )
            cached_tok = int(
                getattr(interaction_usage, "total_cached_tokens", None)
                or getattr(interaction_usage, "cached_tokens", 0)
                or 0
            )
            output_tok = int(
                getattr(interaction_usage, "total_output_tokens", None)
                or getattr(interaction_usage, "output_tokens", 0)
                or 0
            )
            thought_tok = int(
                getattr(interaction_usage, "total_thought_tokens", None)
                or getattr(interaction_usage, "thought_tokens", 0)
                or 0
            )
            total_tok = int(
                getattr(interaction_usage, "total_tokens", 0)
                or (prompt_tok + output_tok + thought_tok)
            )
        else:
            prompt_tok = 0
            cached_tok = 0
            output_tok = 0
            thought_tok = 0
            total_tok = 0

        # Detect non-text media synthesis & embedding modalities even when usage_metadata is omitted
        gen_imgs = getattr(response, "generated_images", None)
        image_count = len(gen_imgs) if isinstance(gen_imgs, (list, tuple)) else 0
        audio_bytes_len = 0
        candidates_list = getattr(response, "candidates", None)
        if isinstance(candidates_list, (list, tuple)):
            for cand in candidates_list:
                c_obj = getattr(cand, "content", None)
                c_parts = getattr(c_obj, "parts", None) if c_obj is not None else None
                if isinstance(c_parts, (list, tuple)):
                    for pt in c_parts:
                        inl = getattr(pt, "inline_data", None)
                        if inl is not None:
                            mime = str(getattr(inl, "mime_type", "") or "").lower()
                            raw_b = getattr(inl, "data", None)
                            b_len = len(raw_b) if isinstance(raw_b, (bytes, bytearray)) else 0
                            if mime.startswith("image/") and b_len > 0:
                                image_count += 1
                            elif (mime.startswith("audio/") or "tts" in (label or "").lower()) and b_len > 0:
                                audio_bytes_len += b_len

        is_embed = (
            method_name == "embed_content"
            or "embed_content" in (label or "").lower()
            or getattr(response, "embeddings", None) is not None
        )
        embed_est_tok = 0
        if is_embed and total_tok == 0:
            if input_chars_hint > 0:
                embed_est_tok = max(1, int(input_chars_hint) // 4)
            else:
                embs = getattr(response, "embeddings", None)
                embed_est_tok = max(1, (len(embs) if isinstance(embs, (list, tuple)) else 1) * 64)

        tts_sec = round(audio_bytes_len / 48000.0, 3) if audio_bytes_len > 0 else 0.0
        uncached_prompt = max(0, prompt_tok - cached_tok)
        # Image synthesis models include ~1,290 image representation tokens in `candidates_token_count`
        # which are already priced per image via `image_generations * img_rate` ($0.039/image).
        img_out_deduct = min(output_tok, 1290 * image_count) if image_count > 0 else 0
        billable_call_out = max(0, output_tok - img_out_deduct) + thought_tok
        call_cost_usd = self._compute_cost_usd(
            uncached_prompt_tokens=uncached_prompt,
            cached_tokens=cached_tok,
            output_and_thought_tokens=billable_call_out,
            image_generations=image_count,
            embedding_tokens=embed_est_tok,
            tts_audio_seconds=tts_sec,
            standard_2027=False,
        )

        family = (label or "generate_content").split("[", 1)[0].strip() or "generate_content"
        with self._lock:
            self.total_calls += 1
            if cached_tok > 0:
                self.calls_with_cache_hit += 1
            self.total_prompt_tokens += prompt_tok
            self.total_cached_tokens += cached_tok
            self.total_output_tokens += output_tok
            self.total_thought_tokens += thought_tok
            self.total_tokens += total_tok
            self.image_generations += image_count
            self.image_output_tokens_deducted += img_out_deduct
            if tts_sec > 0.0:
                self.tts_generations += 1
                self.tts_audio_seconds = round(self.tts_audio_seconds + tts_sec, 3)
            if is_embed:
                self.embedding_calls += 1
                self.embedding_estimated_tokens += embed_est_tok
            bucket = self.by_label.setdefault(
                family,
                {
                    "calls": 0,
                    "cache_hits": 0,
                    "prompt_tokens": 0,
                    "cached_tokens": 0,
                    "output_tokens": 0,
                    "thought_tokens": 0,
                    "total_tokens": 0,
                    "image_generations": 0,
                    "tts_generations": 0,
                    "embedding_estimated_tokens": 0,
                },
            )
            bucket["calls"] += 1
            if cached_tok > 0:
                bucket["cache_hits"] += 1
            bucket["prompt_tokens"] += prompt_tok
            bucket["cached_tokens"] += cached_tok
            bucket["output_tokens"] += output_tok
            bucket["thought_tokens"] += thought_tok
            bucket["total_tokens"] += total_tok
            bucket["image_generations"] += image_count
            if tts_sec > 0.0:
                bucket["tts_generations"] += 1
            bucket["embedding_estimated_tokens"] += embed_est_tok
            session_uncached = max(0, self.total_prompt_tokens - self.total_cached_tokens)
            session_billable_out = (
                max(0, self.total_output_tokens - self.image_output_tokens_deducted)
                + self.total_thought_tokens
            )
            session_cost_usd = self._compute_cost_usd(
                uncached_prompt_tokens=session_uncached,
                cached_tokens=self.total_cached_tokens,
                output_and_thought_tokens=session_billable_out,
                image_generations=self.image_generations,
                embedding_tokens=self.embedding_estimated_tokens,
                tts_audio_seconds=self.tts_audio_seconds,
                cache_storage_token_hours=self.cache_storage_token_hours,
                standard_2027=False,
            )
            session_cache_share_pct = (
                round(100.0 * self.total_cached_tokens / self.total_prompt_tokens, 2)
                if self.total_prompt_tokens > 0
                else 0.0
            )
            session_total_tokens = self.total_tokens

        return {
            "prompt_tokens": prompt_tok,
            "cached_tokens": cached_tok,
            "uncached_prompt_tokens": uncached_prompt,
            "output_tokens": output_tok,
            "thought_tokens": thought_tok,
            "total_tokens": total_tok,
            "image_count": image_count,
            "tts_audio_seconds": tts_sec,
            "embedding_estimated_tokens": embed_est_tok,
            "estimated_call_cost_usd": call_cost_usd,
            "session_estimated_cost_usd": session_cost_usd,
            "session_cache_hit_pct": session_cache_share_pct,
            "session_total_tokens": session_total_tokens,
        }

    def record_playtest_response(
        self,
        response: Any,
        label: str = "BlindPlaytester",
    ) -> Optional[Dict[str, Any]]:
        """Records decoupled Playtester LLM usage into dedicated `playtest_*` counters and USD totals
        without incrementing Studio build pipeline `total_calls` or tripping Studio circuit breakers."""
        if response is None or getattr(response, "_studio_usage_recorded", False):
            return None
        marked_on_obj = False
        try:
            setattr(response, "_studio_usage_recorded", True)
            marked_on_obj = True
        except Exception:
            pass
        if not marked_on_obj:
            resp_id = id(response)
            with self._lock:
                if resp_id in self._recorded_response_ids:
                    return None
                self._recorded_response_ids.append(resp_id)
                if len(self._recorded_response_ids) > 256:
                    del self._recorded_response_ids[:-256]
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
            cached_tok = int(getattr(usage, "cached_content_token_count", 0) or 0)
            output_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
            thought_tok = int(getattr(usage, "thoughts_token_count", 0) or 0)
            total_tok = int(getattr(usage, "total_token_count", 0) or (prompt_tok + output_tok + thought_tok))
        else:
            prompt_tok = 0
            cached_tok = 0
            output_tok = 0
            thought_tok = 0
            total_tok = 0
        uncached_prompt = max(0, prompt_tok - cached_tok)
        call_cost_usd = self._compute_cost_usd(
            uncached_prompt_tokens=uncached_prompt,
            cached_tokens=cached_tok,
            output_and_thought_tokens=output_tok + thought_tok,
            standard_2027=False,
        )
        with self._lock:
            self.playtest_calls += 1
            self.playtest_prompt_tokens += prompt_tok
            self.playtest_cached_tokens += cached_tok
            self.playtest_output_tokens += output_tok
            self.playtest_thought_tokens += thought_tok
            self.playtest_tokens += total_tok
            combined_uncached = max(0, self.total_prompt_tokens - self.total_cached_tokens) + max(
                0, self.playtest_prompt_tokens - self.playtest_cached_tokens
            )
            combined_cached = self.total_cached_tokens + self.playtest_cached_tokens
            combined_out = (
                max(0, self.total_output_tokens - self.image_output_tokens_deducted)
                + self.total_thought_tokens
                + self.playtest_output_tokens
                + self.playtest_thought_tokens
            )
            session_cost_usd = self._compute_cost_usd(
                uncached_prompt_tokens=combined_uncached,
                cached_tokens=combined_cached,
                output_and_thought_tokens=combined_out,
                image_generations=self.image_generations,
                embedding_tokens=self.embedding_estimated_tokens,
                tts_audio_seconds=self.tts_audio_seconds,
                cache_storage_token_hours=self.cache_storage_token_hours,
                standard_2027=False,
            )
            combined_prompt = self.total_prompt_tokens + self.playtest_prompt_tokens
            session_cache_share_pct = (
                round(100.0 * combined_cached / combined_prompt, 2) if combined_prompt > 0 else 0.0
            )
            session_total_tokens = self.total_tokens + self.playtest_tokens

        return {
            "prompt_tokens": prompt_tok,
            "cached_tokens": cached_tok,
            "uncached_prompt_tokens": uncached_prompt,
            "output_tokens": output_tok,
            "thought_tokens": thought_tok,
            "total_tokens": total_tok,
            "image_count": 0,
            "tts_audio_seconds": 0.0,
            "embedding_estimated_tokens": 0,
            "estimated_call_cost_usd": call_cost_usd,
            "session_estimated_cost_usd": session_cost_usd,
            "session_cache_hit_pct": session_cache_share_pct,
            "session_total_tokens": session_total_tokens,
        }

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            hit_rate = (
                round(100.0 * self.calls_with_cache_hit / self.total_calls, 2)
                if self.total_calls > 0
                else 0.0
            )
            cached_share = (
                round(100.0 * self.total_cached_tokens / self.total_prompt_tokens, 2)
                if self.total_prompt_tokens > 0
                else 0.0
            )
            uncached_prompt = max(0, self.total_prompt_tokens - self.total_cached_tokens)
            playtest_uncached = max(0, self.playtest_prompt_tokens - self.playtest_cached_tokens)
            playtest_usd = self._compute_cost_usd(
                uncached_prompt_tokens=playtest_uncached,
                cached_tokens=self.playtest_cached_tokens,
                output_and_thought_tokens=self.playtest_output_tokens + self.playtest_thought_tokens,
                standard_2027=False,
            )
            billable_out = (
                max(0, self.total_output_tokens - self.image_output_tokens_deducted)
                + self.total_thought_tokens
                + self.playtest_output_tokens
                + self.playtest_thought_tokens
            )
            est_usd = self._compute_cost_usd(
                uncached_prompt_tokens=uncached_prompt + playtest_uncached,
                cached_tokens=self.total_cached_tokens + self.playtest_cached_tokens,
                output_and_thought_tokens=billable_out,
                image_generations=self.image_generations,
                embedding_tokens=self.embedding_estimated_tokens,
                tts_audio_seconds=self.tts_audio_seconds,
                cache_storage_token_hours=self.cache_storage_token_hours,
                standard_2027=False,
            )
            est_usd_2027 = self._compute_cost_usd(
                uncached_prompt_tokens=uncached_prompt + playtest_uncached,
                cached_tokens=self.total_cached_tokens + self.playtest_cached_tokens,
                output_and_thought_tokens=billable_out,
                image_generations=self.image_generations,
                embedding_tokens=self.embedding_estimated_tokens,
                tts_audio_seconds=self.tts_audio_seconds,
                cache_storage_token_hours=self.cache_storage_token_hours,
                standard_2027=True,
            )
            return {
                "total_calls": self.total_calls,
                "calls_with_cache_hit": self.calls_with_cache_hit,
                "cache_hit_call_rate_pct": hit_rate,
                "explicit_caches_created": self.explicit_caches_created,
                "cache_storage_token_hours": round(self.cache_storage_token_hours, 4),
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_cached_tokens": self.total_cached_tokens,
                "uncached_prompt_tokens": uncached_prompt,
                "cached_token_share_pct": cached_share,
                "total_output_tokens": self.total_output_tokens,
                "total_thought_tokens": self.total_thought_tokens,
                "total_tokens": self.total_tokens,
                "image_generations": self.image_generations,
                "tts_generations": self.tts_generations,
                "tts_audio_seconds": self.tts_audio_seconds,
                "embedding_calls": self.embedding_calls,
                "embedding_estimated_tokens": self.embedding_estimated_tokens,
                "playtest_calls": self.playtest_calls,
                "playtest_tokens": self.playtest_tokens,
                "playtest_estimated_cost_usd": playtest_usd,
                "estimated_cost_usd": est_usd,
                "estimated_cost_usd_2027_std": est_usd_2027,
                "by_label": {k: dict(v) for k, v in self.by_label.items()},
            }


TOKEN_USAGE_TRACKER = TokenUsageTracker()


def _emit_gateway_llm_activity(label: str, stats: Optional[Dict[str, Any]] = None) -> None:
    """Broadcasts live LLM call activity to Mission Control UI when running outside pytest so
    background LLM calls can never execute invisibly while the UI appears idle."""
    import os as _os
    import sys as _sys
    from datetime import datetime as _dt, timezone as _tz

    if "PYTEST_CURRENT_TEST" in _os.environ or "pytest" in _sys.modules:
        return
    try:
        from engine.ui.server import swarm_bus as _sb
        if _sb is not None and hasattr(_sb, "publish_threadsafe"):
            tok_info = f" ({stats.get('total_tokens', 0)} tokens)" if stats else ""
            _sb.publish_threadsafe({
                "deliberation": f"Live Gemini API call completed: {label}{tok_info}",
                "event_type": "studio_live_activity",
                "agent_name": (label or "StudioAgent").split(".", 1)[0],
                "timestamp_utc": _dt.now(_tz.utc).isoformat(),
                "payload": {
                    "phase": "llm_gateway_call",
                    "label": label,
                    "token_stats": stats or {},
                    "total_calls": TOKEN_USAGE_TRACKER.total_calls,
                },
            })
    except Exception:
        pass


import threading as _gateway_threading

_ACTIVE_DATASET_LOCK = _gateway_threading.Lock()
_ACTIVE_DATASET_CONTEXT: Dict[str, Any] = {
    "dataset_name": "",
    "candidate_dir": None,
    "store_logs": True,
}
_ABORTED_CANDIDATE_NAMES: Set[str] = set()
_THREAD_LOCAL_GEMINI_CALL_CTX = _gateway_threading.local()


def mark_candidate_build_aborted(candidate_dir: Optional[Union[Path, str]] = None) -> None:
    """Marks a candidate directory (or the currently active dataset candidate_dir) as aborted/preempted
    so any in-flight or paused thread working on that candidate exits immediately via `StudioPausedError`."""
    with _ACTIVE_DATASET_LOCK:
        target = candidate_dir if candidate_dir is not None else _ACTIVE_DATASET_CONTEXT.get("candidate_dir")
        if target:
            name = Path(target).name.strip()
            if name and name != "active":
                _ABORTED_CANDIDATE_NAMES.add(name)


def clear_candidate_build_aborted(candidate_dir: Optional[Union[Path, str]] = None) -> None:
    """Clears the aborted state for a specific candidate directory when explicitly resumed."""
    with _ACTIVE_DATASET_LOCK:
        if candidate_dir is None:
            _ABORTED_CANDIDATE_NAMES.clear()
        else:
            name = Path(candidate_dir).name.strip()
            _ABORTED_CANDIDATE_NAMES.discard(name)


def is_candidate_build_aborted(candidate_dir: Optional[Union[Path, str]] = None) -> bool:
    """Returns True if `candidate_dir` (or the thread-local candidate_dir) has been marked aborted."""
    eff_c_dir = (
        Path(candidate_dir)
        if candidate_dir
        else getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "candidate_dir", None)
    )
    if not eff_c_dir:
        return False
    name = Path(eff_c_dir).name.strip()
    if not name:
        return False
    with _ACTIVE_DATASET_LOCK:
        return name in _ABORTED_CANDIDATE_NAMES


def sanitize_gemini_label_value(raw: Any, max_len: int = 63) -> str:
    """Normalizes any string into a valid Google Cloud / Gemini API label value:
    lowercase ASCII letters, digits, underscores, and dashes, up to 63 characters."""
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    out_chars: List[str] = []
    prev_sep = False
    for ch in text:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "-":
            out_chars.append(ch)
            prev_sep = False
        else:
            if not prev_sep and out_chars:
                out_chars.append("_")
                prev_sep = True
    cleaned = "".join(out_chars).strip("_-")
    return cleaned[:max_len].rstrip("_-")


def set_active_gemini_dataset_context(
    *,
    dataset_name: Optional[str] = None,
    candidate_dir: Optional[Union[Path, str]] = None,
    store_logs: Optional[bool] = None,
) -> Dict[str, Any]:
    """Sets the active Studio run/dataset context used to tag outbound Gemini API calls
    with `labels` (`dataset`, `run_id`, `cartridge`, `agent`, `operation`) and `store=True`."""
    with _ACTIVE_DATASET_LOCK:
        if dataset_name is not None:
            _ACTIVE_DATASET_CONTEXT["dataset_name"] = str(dataset_name).strip()
        if candidate_dir is not None:
            c_path = Path(candidate_dir) if candidate_dir else None
            _ACTIVE_DATASET_CONTEXT["candidate_dir"] = c_path
            if c_path is not None and c_path.name.strip():
                _ABORTED_CANDIDATE_NAMES.discard(c_path.name.strip())
        if store_logs is not None:
            _ACTIVE_DATASET_CONTEXT["store_logs"] = bool(store_logs)
        return dict(_ACTIVE_DATASET_CONTEXT)


def get_active_gemini_dataset_context() -> Dict[str, Any]:
    """Returns a snapshot of the active Studio dataset/run logging context."""
    with _ACTIVE_DATASET_LOCK:
        return dict(_ACTIVE_DATASET_CONTEXT)


def build_gemini_request_labels(
    *,
    label: str = "generate_content",
    candidate_dir: Optional[Union[Path, str]] = None,
    dataset_name: Optional[str] = None,
) -> Dict[str, str]:
    """Constructs sanitized Gemini API request labels (`dataset`, `run_id`, `cartridge`, `agent`, `operation`)
    so every call in a Studio run can be filtered and saved as a Dataset in Google AI Studio."""
    import os as _os

    ctx = get_active_gemini_dataset_context()
    eff_c_dir = (
        Path(candidate_dir)
        if candidate_dir
        else (getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "candidate_dir", None) or ctx.get("candidate_dir"))
    )
    cartridge_name = eff_c_dir.name if isinstance(eff_c_dir, Path) else ""

    raw_dataset = (
        (dataset_name or "").strip()
        or _os.environ.get("GEMINI_DATASET_NAME", "").strip()
        or _os.environ.get("STUDIO_DATASET_NAME", "").strip()
        or str(ctx.get("dataset_name") or "").strip()
        or cartridge_name
        or "studio_run"
    )
    eff_label = (
        label
        if label and label != "genai.models.generate_content"
        else (getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "label", None) or label or "generate_content")
    )
    agent_raw = (eff_label or "studio").split(".", 1)[0].split("[", 1)[0]

    labels_out: Dict[str, str] = {}
    ds_val = sanitize_gemini_label_value(raw_dataset)
    if ds_val:
        labels_out["dataset"] = ds_val
        labels_out["run_id"] = ds_val
    cart_val = sanitize_gemini_label_value(cartridge_name)
    if cart_val:
        labels_out["cartridge"] = cart_val
    agent_val = sanitize_gemini_label_value(agent_raw)
    if agent_val:
        labels_out["agent"] = agent_val
    op_val = sanitize_gemini_label_value(eff_label)
    if op_val:
        labels_out["operation"] = op_val
    return labels_out


def inject_gemini_dataset_config(
    kwargs: Dict[str, Any],
    *,
    label: str = "generate_content",
    candidate_dir: Optional[Union[Path, str]] = None,
    dataset_name: Optional[str] = None,
    store_logs: Optional[bool] = None,
) -> Dict[str, Any]:
    """Non-destructively injects `labels` and `http_options.extra_body['store'] = True` into
    `generate_content` kwargs so requests are tagged and stored in Google AI Studio Logs."""
    import os as _os

    out_kwargs = dict(kwargs)
    skip_dataset_meta = bool(getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "skip_dataset_store", False))
    auto_labels = (
        {}
        if skip_dataset_meta
        else build_gemini_request_labels(
            label=label,
            candidate_dir=candidate_dir,
            dataset_name=dataset_name,
        )
    )

    env_store = _os.environ.get("GEMINI_STORE_LOGS", "").strip().lower()
    if skip_dataset_meta:
        should_store = False
    elif store_logs is not None:
        should_store = bool(store_logs)
    elif env_store in ("0", "false", "no", "off"):
        should_store = False
    elif env_store in ("1", "true", "yes", "on"):
        should_store = True
    else:
        should_store = bool(get_active_gemini_dataset_context().get("store_logs", True))

    model_str = str(out_kwargs.get("model") or "").lower()
    cfg = out_kwargs.get("config")

    # Determine if this is an image or audio/TTS synthesis request where AI Studio request storage is unsupported
    modalities: List[str] = []
    if isinstance(cfg, dict):
        modalities = [str(m).upper() for m in (cfg.get("response_modalities") or cfg.get("responseModalities") or [])]
    elif cfg is not None and hasattr(cfg, "response_modalities") and getattr(cfg, "response_modalities"):
        modalities = [str(m).upper() for m in getattr(cfg, "response_modalities")]
    is_media_synthesis = (
        "image" in model_str
        or "nano-banana" in model_str
        or "tts" in model_str
        or "IMAGE" in modalities
        or "AUDIO" in modalities
    )
    if is_media_synthesis:
        should_store = False

    try:
        from google.genai import types as _genai_types
    except Exception:
        _genai_types = None

    if cfg is None:
        if _genai_types is not None:
            http_opts = _genai_types.HttpOptions(extra_body={"store": True}) if should_store else None
            default_thinking = bounded_thinking_config(1024) if not is_media_synthesis else None
            out_kwargs["config"] = _genai_types.GenerateContentConfig(
                labels=auto_labels,
                http_options=http_opts,
                thinking_config=default_thinking,
            )
        else:
            new_cfg: Dict[str, Any] = {"labels": auto_labels}
            if should_store:
                new_cfg["http_options"] = {"extra_body": {"store": True}}
            out_kwargs["config"] = new_cfg
        return out_kwargs

    if isinstance(cfg, dict):
        cfg_copy = dict(cfg)
        existing_labels = dict(cfg_copy.get("labels") or {})
        cfg_copy["labels"] = {**auto_labels, **existing_labels}
        if not is_media_synthesis and cfg_copy.get("thinking_config") is None:
            default_thinking = bounded_thinking_config(1024)
            if default_thinking is not None:
                cfg_copy["thinking_config"] = default_thinking
        if should_store:
            existing_http = cfg_copy.get("http_options")
            if isinstance(existing_http, dict):
                http_copy = dict(existing_http)
                eb = dict(http_copy.get("extra_body") or {})
                eb.setdefault("store", True)
                http_copy["extra_body"] = eb
                cfg_copy["http_options"] = http_copy
            elif existing_http is None:
                cfg_copy["http_options"] = {"extra_body": {"store": True}}
        out_kwargs["config"] = cfg_copy
        return out_kwargs

    if hasattr(cfg, "model_copy"):
        existing_labels = dict(getattr(cfg, "labels", None) or {})
        merged_labels = {**auto_labels, **existing_labels}
        updates: Dict[str, Any] = {"labels": merged_labels}
        if not is_media_synthesis and getattr(cfg, "thinking_config", None) is None:
            default_thinking = bounded_thinking_config(1024)
            if default_thinking is not None:
                updates["thinking_config"] = default_thinking
        if should_store and _genai_types is not None:
            existing_http = getattr(cfg, "http_options", None)
            if existing_http is None:
                updates["http_options"] = _genai_types.HttpOptions(extra_body={"store": True})
            elif isinstance(existing_http, dict):
                http_copy = dict(existing_http)
                eb = dict(http_copy.get("extra_body") or {})
                eb.setdefault("store", True)
                http_copy["extra_body"] = eb
                updates["http_options"] = _genai_types.HttpOptions(**http_copy)
            elif hasattr(existing_http, "model_copy"):
                eb = dict(getattr(existing_http, "extra_body", None) or {})
                eb.setdefault("store", True)
                updates["http_options"] = existing_http.model_copy(update={"extra_body": eb})
        out_kwargs["config"] = cfg.model_copy(update=updates)
        return out_kwargs

    return out_kwargs


def inject_gemini_interactions_config(
    kwargs: Dict[str, Any],
    *,
    label: str = "interactions.create",
    candidate_dir: Optional[Union[Path, str]] = None,
    dataset_name: Optional[str] = None,
    store_logs: Optional[bool] = None,
) -> Dict[str, Any]:
    """Non-destructively injects top-level `labels` and `store=True` into `client.interactions.create` kwargs
    so Interactions API requests receive identical AI Studio Dataset tagging as `generate_content`."""
    import os as _os

    out_kwargs = dict(kwargs)
    if "request" in out_kwargs and out_kwargs["request"] is not None:
        return out_kwargs

    skip_dataset_meta = bool(getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "skip_dataset_store", False))
    if skip_dataset_meta:
        return out_kwargs

    auto_labels = build_gemini_request_labels(
        label=label,
        candidate_dir=candidate_dir,
        dataset_name=dataset_name,
    )
    if auto_labels:
        existing_labels = dict(out_kwargs.get("labels") or {})
        out_kwargs["labels"] = {**auto_labels, **existing_labels}

    env_store = _os.environ.get("GEMINI_STORE_LOGS", "").strip().lower()
    if store_logs is not None:
        should_store = bool(store_logs)
    elif env_store in ("0", "false", "no", "off"):
        should_store = False
    elif env_store in ("1", "true", "yes", "on"):
        should_store = True
    else:
        should_store = bool(get_active_gemini_dataset_context().get("store_logs", True))

    model_str = str(out_kwargs.get("model") or "").lower()
    modalities = [str(m).upper() for m in (out_kwargs.get("response_modalities") or [])]
    is_media_synthesis = (
        "image" in model_str
        or "nano-banana" in model_str
        or "tts" in model_str
        or "IMAGE" in modalities
        or "AUDIO" in modalities
        or "VIDEO" in modalities
    )
    if should_store and not is_media_synthesis and "store" not in out_kwargs:
        out_kwargs["store"] = True
    return out_kwargs


def install_studio_gemini_guardrails(
    client: Any,
    candidate_dir: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    store_logs: Optional[bool] = None,
) -> Any:
    """Installs transport-level pause enforcement, circuit-breaker checks, AI Studio dataset label injection,
    and token tracking directly onto `client.models.generate_content`, `embed_content`, `generate_images`,
    and `client.interactions.create` so even direct SDK calls cannot bypass studio pause, dataset tagging,
    or token accounting."""
    if candidate_dir is not None or dataset_name is not None or store_logs is not None:
        set_active_gemini_dataset_context(
            dataset_name=dataset_name,
            candidate_dir=candidate_dir,
            store_logs=store_logs,
        )
    if client is None or not is_real_genai_client(client):
        return client
    models_obj = getattr(client, "models", None)
    if models_obj is not None and not getattr(models_obj, "_studio_guardrails_installed", False):
        for method_name in ("generate_content", "embed_content", "generate_images"):
            orig_fn = getattr(models_obj, method_name, None)
            if not callable(orig_fn):
                continue

            def _make_guarded(fn: Any, m_name: str) -> Any:
                def _guarded(*args: Any, **kwargs: Any) -> Any:
                    import random as _random
                    import time as _time

                    ctx_label = getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "label", None)
                    ctx_c_dir = getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "candidate_dir", None) or candidate_dir
                    in_retry_wrapper = bool(getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "in_retry_wrapper", False))
                    lbl = ctx_label or f"genai.models.{m_name}"
                    wait_if_studio_paused(candidate_dir=ctx_c_dir, label=lbl)
                    TOKEN_USAGE_TRACKER.check_circuit_breaker(label=lbl, candidate_dir=ctx_c_dir)
                    call_kwargs = (
                        inject_gemini_dataset_config(
                            kwargs,
                            label=lbl,
                            candidate_dir=ctx_c_dir,
                            dataset_name=dataset_name,
                            store_logs=store_logs,
                        )
                        if m_name == "generate_content"
                        else kwargs
                    )
                    max_direct_retries = 0 if in_retry_wrapper else 3
                    for attempt in range(max_direct_retries + 1):
                        try:
                            if attempt > 0:
                                call_kwargs = (
                                    inject_gemini_dataset_config(
                                        kwargs,
                                        label=lbl,
                                        candidate_dir=ctx_c_dir,
                                        dataset_name=dataset_name,
                                        store_logs=store_logs,
                                    )
                                    if m_name == "generate_content"
                                    else kwargs
                                )
                            res = fn(*args, **call_kwargs)
                            _check_gateway_abort(ctx_c_dir, label=lbl)
                            break
                        except StudioPausedError:
                            raise
                        except Exception as exc:
                            if attempt < max_direct_retries and is_transient_gemini_error(exc):
                                if "400" in str(exc) or "INVALID_ARGUMENT" in str(exc).upper():
                                    _THREAD_LOCAL_GEMINI_CALL_CTX.skip_dataset_store = True
                                delay = 1.5 * (2 ** attempt) + _random.uniform(0.0, 0.35)
                                _retry_sleep(delay)
                                wait_if_studio_paused(candidate_dir=ctx_c_dir, label=lbl)
                                continue
                            raise
                    embed_chars_hint = 0
                    if m_name == "embed_content":
                        raw_embed_c = kwargs.get("contents", args[0] if args else "")
                        if isinstance(raw_embed_c, (list, tuple)):
                            embed_chars_hint = sum(len(str(x)) for x in raw_embed_c)
                        elif raw_embed_c is not None:
                            embed_chars_hint = len(str(raw_embed_c))
                    stats = TOKEN_USAGE_TRACKER.record_response(
                        res,
                        label=lbl,
                        input_chars_hint=embed_chars_hint,
                        method_name=m_name,
                    )
                    if stats is not None:
                        _emit_gateway_llm_activity(lbl, stats)
                    return res
                return _guarded

            try:
                setattr(models_obj, method_name, _make_guarded(orig_fn, method_name))
            except Exception:
                pass

        try:
            setattr(models_obj, "_studio_guardrails_installed", True)
        except Exception:
            pass

    interactions_obj = getattr(client, "interactions", None)
    if interactions_obj is not None and not getattr(interactions_obj, "_studio_guardrails_installed", False):
        orig_create = getattr(interactions_obj, "create", None)
        if callable(orig_create):
            def _make_guarded_interactions(fn: Any) -> Any:
                def _guarded_interactions(*args: Any, **kwargs: Any) -> Any:
                    import random as _random
                    import time as _time

                    ctx_label = getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "label", None)
                    ctx_c_dir = getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "candidate_dir", None) or candidate_dir
                    in_retry_wrapper = bool(getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "in_retry_wrapper", False))
                    lbl = ctx_label or "genai.interactions.create"
                    wait_if_studio_paused(candidate_dir=ctx_c_dir, label=lbl)
                    TOKEN_USAGE_TRACKER.check_circuit_breaker(label=lbl, candidate_dir=ctx_c_dir)
                    call_kwargs = inject_gemini_interactions_config(
                        kwargs,
                        label=lbl,
                        candidate_dir=ctx_c_dir,
                        dataset_name=dataset_name,
                        store_logs=store_logs,
                    )
                    max_direct_retries = 0 if in_retry_wrapper else 3
                    for attempt in range(max_direct_retries + 1):
                        try:
                            if attempt > 0:
                                call_kwargs = inject_gemini_interactions_config(
                                    kwargs,
                                    label=lbl,
                                    candidate_dir=ctx_c_dir,
                                    dataset_name=dataset_name,
                                    store_logs=store_logs,
                                )
                            res = fn(*args, **call_kwargs)
                            _check_gateway_abort(ctx_c_dir, label=lbl)
                            break
                        except StudioPausedError:
                            raise
                        except Exception as exc:
                            if attempt < max_direct_retries and is_transient_gemini_error(exc):
                                if "400" in str(exc) or "INVALID_ARGUMENT" in str(exc).upper():
                                    _THREAD_LOCAL_GEMINI_CALL_CTX.skip_dataset_store = True
                                delay = 1.5 * (2 ** attempt) + _random.uniform(0.0, 0.35)
                                _retry_sleep(delay)
                                wait_if_studio_paused(candidate_dir=ctx_c_dir, label=lbl)
                                continue
                            raise
                    stats = TOKEN_USAGE_TRACKER.record_response(
                        res,
                        label=lbl,
                        method_name="interactions.create",
                    )
                    if stats is not None:
                        _emit_gateway_llm_activity(lbl, stats)
                    return res
                return _guarded_interactions

            try:
                setattr(interactions_obj, "create", _make_guarded_interactions(orig_create))
                setattr(interactions_obj, "_studio_guardrails_installed", True)
            except Exception:
                pass
    return client


def _retry_sleep(seconds: float) -> None:
    import time as _time
    _time.sleep(seconds)


def _is_client_explicit_cache_eligible(client: Any) -> bool:
    import os as _os

    if _os.environ.get("STUDIO_ENABLE_EXPLICIT_CACHE", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    if client is None:
        return False
    if not (is_real_genai_client(client) or bool(getattr(client, "_allow_explicit_prefix_cache", False))):
        return False
    caches_obj = getattr(client, "caches", None)
    return callable(getattr(caches_obj, "create", None))


def _estimate_prefix_token_floor(parts: Sequence[Any], system_instruction: Optional[str] = None) -> int:
    text_chars = len(system_instruction) if isinstance(system_instruction, str) else 0
    media_tokens = 0
    for p in parts:
        if isinstance(p, str):
            text_chars += len(p)
            continue
        p_txt = getattr(p, "text", None)
        if isinstance(p_txt, str) and p_txt:
            text_chars += len(p_txt)
        inl = getattr(p, "inline_data", None)
        if inl is not None:
            mime = str(getattr(inl, "mime_type", "") or "").lower()
            raw_b = getattr(inl, "data", None)
            if isinstance(raw_b, (bytes, bytearray)) and len(raw_b) > 0:
                if mime.startswith("image/"):
                    m_res = str(getattr(p, "media_resolution", "") or "").upper()
                    if "HIGH" in m_res:
                        media_tokens += 1120
                    elif "LOW" in m_res:
                        media_tokens += 280
                    else:
                        media_tokens += 560
                else:
                    media_tokens += 128
    return int(text_chars / 4.5) + media_tokens


def _hash_explicit_prefix(
    model: str,
    system_instruction: Optional[str],
    parts: Sequence[Any],
) -> str:
    import hashlib as _hashlib

    h = _hashlib.sha256()
    h.update(f"M:{model}\0SI:{system_instruction or ''}\0".encode("utf-8", errors="replace"))
    for p in parts:
        if isinstance(p, str):
            b = p.encode("utf-8", errors="replace")
            h.update(b"S:")
            h.update(len(b).to_bytes(8, "big"))
            h.update(b)
            continue
        p_txt = getattr(p, "text", None)
        if isinstance(p_txt, str) and p_txt:
            b = p_txt.encode("utf-8", errors="replace")
            h.update(b"PT:")
            h.update(len(b).to_bytes(8, "big"))
            h.update(b)
        inl = getattr(p, "inline_data", None)
        if inl is not None:
            mime = str(getattr(inl, "mime_type", "") or "")
            raw_b = getattr(inl, "data", b"") or b""
            h.update(f"PI:{mime}:".encode("utf-8", errors="replace"))
            h.update(len(raw_b).to_bytes(8, "big"))
            h.update(bytes(raw_b))
            continue
        if not (isinstance(p_txt, str) and p_txt):
            rb = repr(p).encode("utf-8", errors="replace")
            h.update(b"R:")
            h.update(len(rb).to_bytes(8, "big"))
            h.update(rb)
    return h.hexdigest()


class ExplicitPrefixCacheScope:
    """Scoped, thread-safe manager for deterministic Gemini `client.caches.create` prefix caching.
    Automatically deletes all created `CachedContent` resources upon scope exit so storage charges
    last only for the exact seconds of the multi-turn loop or Council review round."""
    deliberation: str = "Scoped, thread-safe manager for deterministic Gemini explicit prefix caching and automatic teardown."

    def __init__(self, client: Any, *, ttl_seconds: int = 360) -> None:
        import threading as _threading

        self.client = client
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.closed = False
        self._lock = _threading.Lock()
        self._seen_counts: Dict[str, int] = {}
        self._created_caches: Dict[str, Dict[str, Any]] = {}
        self._unsuitable_keys: set[str] = set()
        self._inflight_creates: Dict[str, Any] = {}

    def resolve(
        self,
        *,
        model: str,
        prefix_parts: Sequence[Any],
        tail_parts: Sequence[Any],
        unsplit_contents: Any = None,
        system_instruction: Optional[str] = None,
        min_calls_before_create: int = 2,
        display_name: str = "studio_prefix",
    ) -> Tuple[Any, Optional[str]]:
        import logging as _logging
        import os as _os
        import time as _time

        fallback = (
            unsplit_contents
            if unsplit_contents is not None
            else (list(prefix_parts) + list(tail_parts))
        )
        if not prefix_parts or not _is_client_explicit_cache_eligible(self.client):
            return fallback, None

        min_tok_floor = int(_os.environ.get("STUDIO_EXPLICIT_CACHE_MIN_TOKENS", "4500"))
        est_tokens = _estimate_prefix_token_floor(prefix_parts, system_instruction=system_instruction)
        if est_tokens < min_tok_floor:
            return fallback, None

        eff_tail_list = [
            p for p in tail_parts if not (isinstance(p, str) and not p.strip())
        ]
        if not eff_tail_list:
            eff_tail_list = ["Proceed with the task above and return the structured JSON response."]

        key = _hash_explicit_prefix(model, system_instruction, prefix_parts)
        expired_entry: Optional[Dict[str, Any]] = None
        wait_event: Optional[Any] = None
        creator_event: Optional[Any] = None
        now_check = _time.monotonic()
        ttl_safety_margin = min(45.0, max(5.0, float(self.ttl_seconds) * 0.15))
        max_reusable_age = max(5.0, float(self.ttl_seconds) - ttl_safety_margin)
        import threading as _threading

        with self._lock:
            if self.closed or key in self._unsuitable_keys:
                return fallback, None
            seen = self._seen_counts.get(key, 0) + 1
            self._seen_counts[key] = seen
            existing = self._created_caches.get(key)
            if existing is not None:
                age_s = max(0.0, now_check - float(existing.get("created_at") or now_check))
                if age_s < max_reusable_age:
                    return eff_tail_list, str(existing["name"])
                expired_entry = self._created_caches.pop(key, None)
            if seen < max(1, int(min_calls_before_create)):
                return fallback, None
            inflight_ev = self._inflight_creates.get(key)
            if inflight_ev is not None:
                wait_event = inflight_ev
            else:
                creator_event = _threading.Event()
                self._inflight_creates[key] = creator_event

        if expired_entry is not None:
            exp_tokens = int(expired_entry.get("tokens") or 0)
            exp_age = min(
                float(self.ttl_seconds),
                max(1.0, now_check - float(expired_entry.get("created_at") or now_check)),
            )
            TOKEN_USAGE_TRACKER.record_explicit_cache_lifecycle(exp_tokens, exp_age)
            exp_name = str(expired_entry.get("name") or "")
            delete_fn_exp = getattr(getattr(self.client, "caches", None), "delete", None)
            if exp_name and callable(delete_fn_exp):
                try:
                    delete_fn_exp(name=exp_name)
                except Exception:
                    pass

        if wait_event is not None:
            wait_event.wait(timeout=30.0)
            with self._lock:
                if self.closed or key in self._unsuitable_keys:
                    return fallback, None
                existing_after = self._created_caches.get(key)
                if existing_after is not None:
                    return eff_tail_list, str(existing_after["name"])
            return fallback, None

        _log = _logging.getLogger("GeminiExplicitCache")
        try:
            wait_if_studio_paused(label=f"caches.create[{display_name}]")
            try:
                from google.genai import types as _genai_types

                safe_disp = sanitize_gemini_label_value(display_name) or "studio_prefix"
                cache_cfg = _genai_types.CreateCachedContentConfig(
                    contents=list(prefix_parts),
                    system_instruction=system_instruction,
                    ttl=f"{self.ttl_seconds}s",
                    display_name=safe_disp[:60],
                )
            except Exception:
                cache_cfg = {
                    "contents": list(prefix_parts),
                    "ttl": f"{self.ttl_seconds}s",
                    "display_name": (display_name or "studio_prefix")[:60],
                }
                if system_instruction:
                    cache_cfg["system_instruction"] = system_instruction

            created_obj = None
            for c_attempt in range(2):
                try:
                    created_obj = self.client.caches.create(model=model, config=cache_cfg)
                    break
                except StudioPausedError:
                    raise
                except Exception as exc:
                    err_low = str(exc).lower()
                    if (
                        "too small" in err_low
                        or "min_total_token_count" in err_low
                        or ("minimum" in err_low and "token" in err_low)
                        or "4096" in err_low
                    ):
                        with self._lock:
                            self._unsuitable_keys.add(key)
                        _log.debug("Prefix below model explicit cache minimum for %s: %s", display_name, exc)
                        return fallback, None
                    if c_attempt == 0 and is_transient_gemini_error(exc):
                        _retry_sleep(1.0)
                        continue
                    raise

            cache_name = str(getattr(created_obj, "name", "") or "").strip()
            if not cache_name:
                return fallback, None
            c_usage = getattr(created_obj, "usage_metadata", None)
            tok_cnt = int(getattr(c_usage, "total_token_count", 0) or est_tokens)
            now_mono = _time.monotonic()
            dup_to_delete: Optional[str] = None
            with self._lock:
                if self.closed:
                    dup_to_delete = cache_name
                elif key in self._created_caches:
                    dup_to_delete = cache_name
                    cache_name = str(self._created_caches[key]["name"])
                else:
                    self._created_caches[key] = {
                        "name": cache_name,
                        "created_at": now_mono,
                        "tokens": tok_cnt,
                    }
            if dup_to_delete:
                try:
                    self.client.caches.delete(name=dup_to_delete)
                except Exception:
                    pass
                if self.closed:
                    return fallback, None
            return eff_tail_list, cache_name
        finally:
            if creator_event is not None:
                with self._lock:
                    if self._inflight_creates.get(key) is creator_event:
                        self._inflight_creates.pop(key, None)
                creator_event.set()

    def close(self) -> None:
        import logging as _logging
        import time as _time

        with self._lock:
            if self.closed:
                return
            self.closed = True
            to_cleanup = list(self._created_caches.values())
            self._created_caches.clear()

        if not to_cleanup:
            return
        _log = _logging.getLogger("GeminiExplicitCache")
        delete_fn = getattr(getattr(self.client, "caches", None), "delete", None)
        now_mono = _time.monotonic()
        for entry in to_cleanup:
            c_name = str(entry.get("name") or "")
            tok_cnt = int(entry.get("tokens") or 0)
            lifetime_s = min(
                float(self.ttl_seconds),
                max(1.0, now_mono - float(entry.get("created_at") or now_mono)),
            )
            TOKEN_USAGE_TRACKER.record_explicit_cache_lifecycle(tok_cnt, lifetime_s)
            if c_name and callable(delete_fn):
                try:
                    delete_fn(name=c_name)
                except Exception as exc:
                    _log.debug("Explicit cache delete cleanup note for %s: %s", c_name, exc)


def get_active_explicit_prefix_cache_scope(client: Optional[Any] = None) -> Optional[ExplicitPrefixCacheScope]:
    stack = getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "explicit_cache_scopes", None)
    if not stack:
        return None
    for scope in reversed(stack):
        if scope.closed:
            continue
        if client is None or scope.client is client:
            return scope
    return None


def bind_explicit_prefix_cache_scope(scope: Optional[ExplicitPrefixCacheScope]):
    """Temporarily binds an existing `ExplicitPrefixCacheScope` into a worker thread's TLS stack without closing it."""
    import contextlib as _contextlib

    @_contextlib.contextmanager
    def _bind():
        if scope is None or scope.closed:
            yield scope
            return
        stack = getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "explicit_cache_scopes", None)
        if stack is None:
            stack = []
            _THREAD_LOCAL_GEMINI_CALL_CTX.explicit_cache_scopes = stack
        stack.append(scope)
        try:
            yield scope
        finally:
            if stack and stack[-1] is scope:
                stack.pop()
            elif scope in stack:
                stack.remove(scope)

    return _bind()


def explicit_prefix_cache_scope(
    client: Any,
    *,
    ttl_seconds: int = 360,
    reuse_outer: bool = True,
):
    import contextlib as _contextlib

    @_contextlib.contextmanager
    def _mgr():
        if reuse_outer:
            outer = get_active_explicit_prefix_cache_scope(client)
            if outer is not None:
                yield outer
                return
        scope = ExplicitPrefixCacheScope(client, ttl_seconds=ttl_seconds)
        stack = getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "explicit_cache_scopes", None)
        if stack is None:
            stack = []
            _THREAD_LOCAL_GEMINI_CALL_CTX.explicit_cache_scopes = stack
        stack.append(scope)
        try:
            yield scope
        finally:
            try:
                if stack and stack[-1] is scope:
                    stack.pop()
                elif scope in stack:
                    stack.remove(scope)
            finally:
                scope.close()

    return _mgr()


def resolve_explicit_cached_request(
    client: Any,
    *,
    model: str,
    prefix_parts: Sequence[Any],
    tail_parts: Sequence[Any],
    unsplit_contents: Any = None,
    system_instruction: Optional[str] = None,
    min_calls_before_create: int = 2,
    display_name: str = "studio_prefix",
) -> Tuple[Any, Optional[str]]:
    """Resolves `(effective_contents, cached_content_name)` against the active `ExplicitPrefixCacheScope`
    for `client`, falling back to `(unsplit_contents, None)` when outside a scope or below token threshold."""
    fallback = (
        unsplit_contents
        if unsplit_contents is not None
        else (list(prefix_parts) + list(tail_parts))
    )
    scope = get_active_explicit_prefix_cache_scope(client)
    if scope is None:
        return fallback, None
    return scope.resolve(
        model=model,
        prefix_parts=prefix_parts,
        tail_parts=tail_parts,
        unsplit_contents=fallback,
        system_instruction=system_instruction,
        min_calls_before_create=min_calls_before_create,
        display_name=display_name,
    )


def call_gemini_with_transient_retry(
    call_fn: Any,
    *,
    max_transient_retries: int = 4,
    base_delay_s: float = 1.5,
    label: str = "generate_content",
    candidate_dir: Optional[Path] = None,
) -> Any:
    """Executes `call_fn()` with:
    1. Mandatory gateway pause enforcement (`wait_if_studio_paused`) BEFORE every attempt.
    2. Hard runaway-loop circuit breaker (`TOKEN_USAGE_TRACKER.check_circuit_breaker`).
    3. Thread-local call context propagation (`label`, `candidate_dir`) so `install_studio_gemini_guardrails`
       automatically tags the outbound `generate_content` request with AI Studio Dataset `labels` and `store=True`.
    4. Jittered exponential backoff strictly for transient 504/503/429 and DNS/socket errors.
    5. Automatic `TOKEN_USAGE_TRACKER` accounting and live Mission Control UI telemetry emission.
    """
    import logging as _logging
    import random as _random
    import time as _time

    _log = _logging.getLogger("GeminiRetry")
    for attempt in range(max_transient_retries + 1):
        wait_if_studio_paused(candidate_dir=candidate_dir, label=label)
        TOKEN_USAGE_TRACKER.check_circuit_breaker(label=label, candidate_dir=candidate_dir)
        prev_label = getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "label", None)
        prev_c_dir = getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "candidate_dir", None)
        prev_in_retry = getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "in_retry_wrapper", False)
        prev_skip_store = getattr(_THREAD_LOCAL_GEMINI_CALL_CTX, "skip_dataset_store", False)
        _THREAD_LOCAL_GEMINI_CALL_CTX.label = label
        _THREAD_LOCAL_GEMINI_CALL_CTX.in_retry_wrapper = True
        if candidate_dir is not None:
            _THREAD_LOCAL_GEMINI_CALL_CTX.candidate_dir = candidate_dir
        try:
            res = call_fn()
            _check_gateway_abort(candidate_dir, label=label)
            stats = TOKEN_USAGE_TRACKER.record_response(res, label=label)
            if stats is not None:
                _emit_gateway_llm_activity(label, stats)
            if stats is not None and stats["cached_tokens"] > 0:
                _log.debug(
                    "%s token usage: prompt=%d cached=%d (%.1f%%) output=%d thoughts=%d",
                    label,
                    stats["prompt_tokens"],
                    stats["cached_tokens"],
                    (100.0 * stats["cached_tokens"] / max(1, stats["prompt_tokens"])),
                    stats["output_tokens"],
                    stats["thought_tokens"],
                )
            return res
        except StudioPausedError:
            raise
        except Exception as exc:
            if attempt < max_transient_retries and is_transient_gemini_error(exc):
                if "400" in str(exc) or "INVALID_ARGUMENT" in str(exc).upper():
                    _THREAD_LOCAL_GEMINI_CALL_CTX.skip_dataset_store = True
                    prev_skip_store = True
                delay = base_delay_s * (2 ** attempt) + _random.uniform(0.0, 0.35)
                _log.warning(
                    "%s transient error (attempt %d/%d), backing off %.2fs: %s",
                    label,
                    attempt + 1,
                    max_transient_retries + 1,
                    delay,
                    exc,
                )
                _retry_sleep(delay)
                continue
            raise
        finally:
            _THREAD_LOCAL_GEMINI_CALL_CTX.label = prev_label
            _THREAD_LOCAL_GEMINI_CALL_CTX.candidate_dir = prev_c_dir
            _THREAD_LOCAL_GEMINI_CALL_CTX.in_retry_wrapper = prev_in_retry
            _THREAD_LOCAL_GEMINI_CALL_CTX.skip_dataset_store = prev_skip_store


# =============================================================================
# Interactive Studio Co-Producer & Multimodal Showrunner Models (Rule 4)
# =============================================================================

CoProducerToolName = Literal[
    "inspect_studio_progress",
    "inspect_game_bible_and_dna",
    "inspect_playtest_sessions",
    "browse_cartridge_media_and_code",
    "curate_genre_fun_foundation",
    "consult_council",
    "dispatch_specialist_probe",
    "sketch_multimodal_concept",
]

CoProducerActionKind = Literal[
    "commission_game",
    "pivot_course_correction",
    "pivot_structural",
    "pivot_fork_cartridge",
    "greenlight_bible",
    "trigger_next_slice",
    "pause_build",
    "resume_build",
    "launch_game",
    "start_blind_playtest",
    "record_game_memory",
]


class CoProducerConceptBrief(BaseEpistemicModel):
    """Structured living synthesis of a co-designed game concept tracked across Co-Producer turns."""

    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Synthesis reasoning explaining how the user's preferences and consultations shaped this brief.",
    )
    working_title: str = Field(default="", description="Working game title.")
    core_genre: str = Field(default="", description="Primary genre and structural paradigm.")
    elevator_pitch: str = Field(default="", description="Crisp 1-3 sentence hook summarizing the playable premise.")
    core_fantasy: str = Field(default="", description="Emotional and systemic fantasy experienced by the player.")
    core_player_verbs: List[str] = Field(
        default_factory=list,
        description="Concrete moment-to-moment player actions and verbs.",
    )
    first_60_seconds_feel: str = Field(
        default="",
        description="What the player sees, hears, and manipulates in the opening 60 seconds.",
    )
    visual_style_direction: str = Field(
        default="",
        description="Art direction, camera framing, silhouette readability, and lighting mood.",
    )
    audio_mood_direction: str = Field(
        default="",
        description="Soundscape, tactile SFX character, ambient beds, and voice tone.",
    )
    color_palette_hex: List[str] = Field(
        default_factory=list,
        description="Optional hex palette swatches (e.g. '#1E293B').",
    )
    user_confirmed_pillars: List[str] = Field(
        default_factory=list,
        description=(
            "Cumulative ledger of core pillars, systemic depth requirements, spatial/camera rules, and progression arcs "
            "explicitly requested or confirmed by the user across all prior turns. Must be preserved across turns unless "
            "explicitly retracted or replaced by the user."
        ),
    )
    explicit_non_goals: List[str] = Field(
        default_factory=list,
        description="Cumulative explicit anti-goals, rejected visual tropes, or unwanted mechanics the studio and concept art must strictly avoid.",
    )
    open_design_questions: List[str] = Field(
        default_factory=list,
        description="Unresolved design or feel questions still being collaboratively shaped with the user before commissioning.",
    )
    selected_mockup_paths: List[str] = Field(
        default_factory=list,
        description="Approved 16:9 concept mockup artifact paths.",
    )
    selected_sketch_asset_paths: List[str] = Field(
        default_factory=list,
        description="Approved inline sprite, voice audition, or SFX sketch paths.",
    )
    genre_craft_foundation: Optional[GenreCraftFoundation] = Field(
        default=None,
        description=(
            "Curated GenreCraftFoundation articulating what makes this genre genuinely fun, its foundational pillars, "
            "bootstrap-to-mastery arc, UX contracts, and genre-killing anti-patterns."
        ),
    )


class CoProducerToolCall(BaseEpistemicModel):
    """Typed tool call requested by CoProducerStepDecision without untyped dict ambiguity."""

    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Justification for why this specific tool call is needed before responding.",
    )
    tool_name: CoProducerToolName = Field(
        ...,
        description="Exact Co-Producer tool to execute.",
    )
    query_or_question: str = Field(
        default="",
        description="Primary question, inspection brief, semantic query, voice line text, or visual/audio synthesis prompt.",
    )
    bible_or_dna_focus: Literal["overview", "full_bible", "player_dna", "both"] = Field(
        default="both",
        description="Scope when tool_name == 'inspect_game_bible_and_dna'.",
    )
    playtest_session_id: str = Field(
        default="",
        description="Optional playtest session_id when tool_name == 'inspect_playtest_sessions' (empty = latest).",
    )
    evaluate_media_perceptually: bool = Field(
        default=False,
        description="When True during inspect_playtest_sessions or browse_cartridge_media_and_code, run perceptual video/audio evaluators on key clips.",
    )
    media_or_code_filter: Literal[
        "all",
        "sprites",
        "mockups",
        "audio",
        "qa_proofs",
        "code_diffs",
        "memories",
    ] = Field(
        default="all",
        description="Filter when tool_name == 'browse_cartridge_media_and_code'.",
    )
    council_mode: Literal["huddle", "divergent_pitch_round"] = Field(
        default="huddle",
        description="For consult_council: 'huddle' convenes 1-4 specialists; 'divergent_pitch_round' runs orthogonal fragments + premise collision.",
    )
    specialist_roles: List[str] = Field(
        default_factory=list,
        description="For consult_council: 1 to 4 specialist role names from SPECIALIST_ROSTER.",
    )
    probe_persona: Literal[
        "swe_engineer",
        "visual_artist",
        "audio_designer",
        "script_writer",
        "qa_test_engineer",
    ] = Field(
        default="swe_engineer",
        description="For dispatch_specialist_probe: lock-free read-only specialist persona.",
    )
    target_paths: List[str] = Field(
        default_factory=list,
        description="Optional relative file, module, or asset paths to inspect.",
    )
    sketch_kind: Literal[
        "concept_mockup_16x9",
        "sprite_sketch",
        "voice_audition",
        "sfx_or_ambience",
    ] = Field(
        default="concept_mockup_16x9",
        description="For sketch_multimodal_concept: modality to synthesize inline.",
    )
    sketch_title: str = Field(
        default="",
        description="Short descriptive title, character name, or asset label for sketch_multimodal_concept.",
    )
    sketch_secondary_direction: str = Field(
        default="",
        description="Optional style guide, camera/perspective lock, voice acting direction, or secondary constraint for sketch_multimodal_concept.",
    )
    sketch_negative_constraints: List[str] = Field(
        default_factory=list,
        description="Explicit visual elements, camera angles, background tropes, or HUD chrome that MUST NOT appear in sketch_multimodal_concept.",
    )
    use_prior_sketch_as_reference: bool = Field(
        default=False,
        description=(
            "For sketch_multimodal_concept ('concept_mockup_16x9'): Set True ONLY when making targeted refinements "
            "to a strong prior sketch whose camera angle, relative scale, and composition already worked. "
            "Leave False when creating a new shot or resetting a drifted/rejected composition."
        ),
    )
    loop_audio: bool = Field(
        default=False,
        description="For sfx_or_ambience: True for seamless ambient loop, False for one-shot SFX.",
    )


class ConceptArtShotPlan(BaseEpistemicModel):
    """Rationale-First Art-Direction Shot Compiler contract translating raw design prose into a coherent,
    playable-scale in-world game viewport specification before image synthesis (Rule 4)."""

    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: Art-direction reasoning resolving any macro-vs-micro camera/scale paradoxes, "
            "reconciling gameplay readability with environmental vastness, and translating confirmed gameplay pillars "
            "into concrete multi-layer visual staging."
        ),
    )
    rendering_medium_and_surface_craft: str = Field(
        ...,
        description=(
            "Exact visual medium, shader/material finish, linework, and surface craft matching the game's art direction "
            "(e.g., crisp high-definition 2D digital illustration with clean geometric silhouettes and emissive shaders, "
            "hand-painted gouache, stylized cel-shaded 3D, or crisp pixel art ONLY when pixel art was explicitly requested). "
            "Never default to generic mobile-icon clip art or unrequested pixel art."
        ),
    )
    camera_projection_and_viewport_scale: str = Field(
        ...,
        description=(
            "Single coherent camera lens/projection and world-space span visible in the 16:9 viewport. "
            "When staging a small playable craft, character, or local facility inside a colossal environment, "
            "keep the camera locked to the LOCAL PLAYABLE VIEWPORT scale while expressing world vastness via "
            "edge-to-edge environmental fields and deep background/underlayer parallax—NEVER zoom out to a whole-world "
            "diagram with oversized icons, and NEVER shrink a vast world into a tiny centered bullseye/diorama."
        ),
    )
    playable_midground_composition: str = Field(
        ...,
        description=(
            "Primary playable entities, multi-component functional gameplay structures (interconnected modules, conduits, "
            "machinery, terrain topology—never a single flat simplified emblem/icon), and explicit relative proportions "
            "(e.g., player unit ~3-7% of frame height vs. multi-part structure ~20-40% of frame width)."
        ),
    )
    foreground_and_parallax_depth_layers: str = Field(
        ...,
        description=(
            "Multi-plane depth staging across subtle foreground framing cues and deep background/underlayer parallax vistas "
            "that convey environmental scale without violating the locked playable camera projection."
        ),
    )
    lighting_and_color_script: str = Field(
        ...,
        description=(
            "Key light direction, contrast hierarchy separating playable foreground/midground silhouettes from deep "
            "background layers, and concrete hex/material color anchors."
        ),
    )
    diegetic_feedback_and_ui_policy: str = Field(
        default="Pure in-world gameplay viewport capture with readable world-space state cues; zero title text, zero floating labels, and zero HUD chrome unless explicitly requested.",
        description="In-world visual state readability and strict UI/HUD policy for this frame.",
    )
    hard_visual_exclusions: List[str] = Field(
        default_factory=list,
        description=(
            "Cumulative list of user non-goals plus mandatory anti-slop exclusions (zero rendered title/poster text, "
            "zero watermarks, zero unrequested HUD bars/minimaps, zero simplified single-icon base emblems, zero "
            "centered miniature bullseye/toy-diorama scale collapse)."
        ),
    )


class ConceptArtCritiqueVerdict(BaseEpistemicModel):
    """Rationale-First multimodal art-director critique evaluating a rendered concept PNG before presentation (Rule 4)."""

    deliberation: str = Field(
        ...,
        description=(
            "MANDATORY FIRST PROPERTY: Pixel-level multimodal examination of the rendered PNG against the user's latest "
            "feedback, confirmed pillars, locked camera projection, relative entity-to-structure-to-world scale, "
            "rendering medium, and hard_visual_exclusions."
        ),
    )
    observed_camera_and_scale_assessment: str = Field(
        ...,
        description=(
            "Objective description of the camera angle and relative unit/structure/environment scale actually visible "
            "in the rendered pixels (explicitly checking for perspective drift, miniature bullseye diorama collapse, "
            "or oversized toy icon pasted onto a macro diagram)."
        ),
    )
    strengths_observed: List[str] = Field(
        default_factory=list,
        description="Concrete visual qualities that succeeded in the rendered frame.",
    )
    flaws_or_constraint_violations: List[str] = Field(
        default_factory=list,
        description="Specific visual defects, scale/camera drift, rendering-medium mismatch, text artifacts, or violated exclusions (empty if none).",
    )
    honest_producer_summary: str = Field(
        ...,
        description=(
            "Candid 1-2 sentence assessment for the Co-Producer's tool observation accurately describing what succeeded "
            "and any remaining imperfections so the Co-Producer never gaslights the user."
        ),
    )
    corrective_scene_rewrite: str = Field(
        default="",
        description=(
            "When passes_art_direction_bar is False, a complete, self-contained revised playable scene & camera staging "
            "directive that eliminates every observed defect on retry; empty when passes_art_direction_bar is True."
        ),
    )
    passes_art_direction_bar: bool = Field(
        ...,
        description=(
            "FINAL DECISION TOKEN: True ONLY when camera projection, relative scale, rendering medium, functional "
            "structure complexity, and all hard_visual_exclusions are faithfully satisfied."
        ),
    )


class CoProducerStepDecision(BaseEpistemicModel):
    """Rationale-First step controller schema deciding whether to gather evidence/sketches or synthesize response."""

    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Reasoning about the user's goal, sufficiency of gathered context, and which tools or consultations add real value.",
    )
    next_step_intent: str = Field(
        default="",
        description="Concise human-readable status message streamed to the UI while executing this step.",
    )
    response_mode: Literal["creative_ideation", "studio_briefing"] = Field(
        default="creative_ideation",
        description="'creative_ideation' for brainstorming, pitching, or design trade-offs; 'studio_briefing' for factual progress/media inspection.",
    )
    tool_calls: List[CoProducerToolCall] = Field(
        default_factory=list,
        description="1 to 3 tool calls to execute when action == 'call_tools'; empty when action == 'respond'.",
    )
    action: Literal["call_tools", "respond"] = Field(
        ...,
        description="FINAL DECISION TOKEN: 'call_tools' to inspect/consult/sketch first, or 'respond' to deliver the final turn.",
    )


class CoProducerAdvisorySpecialistTurn(BaseEpistemicModel):
    """Structured advisory response produced by a Council specialist during a read-only Co-Producer huddle."""

    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Specialist's domain-specific trade-off analysis.",
    )
    key_recommendation: str = Field(
        ...,
        description="One-sentence actionable recommendation or creative angle from this specialist's lens.",
    )
    creative_angle_or_risk: str = Field(
        default="",
        description="Specific opportunity, unexpected twist, or technical/UX risk to watch.",
    )
    spoken_quote: str = Field(
        ...,
        description="Natural first-person studio remark (2-4 sentences) capturing this specialist's perspective.",
    )


class CoProducerCouncilHuddleTurn(BaseEpistemicModel):
    """Persisted specialist perspective rendered inside an expandable Co-Producer Consultation Card."""

    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Specialist's internal deliberation.",
    )
    speaker_role: str = Field(..., description="Specialist role title from SPECIALIST_ROSTER.")
    key_recommendation: str = Field(..., description="Headline recommendation.")
    creative_angle_or_risk: str = Field(default="", description="Notable twist, synergy, or risk.")
    spoken_quote: str = Field(..., description="Specialist's verbatim studio quote.")


class CoProducerDivergentPitchOption(BaseEpistemicModel):
    """One collided game pitch option surfaced from an advisory divergent pitch round."""

    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Why these orthogonal fragments collide into a compelling playable hook.",
    )
    pitch_id: str = Field(..., description="Stable pitch option identifier.")
    working_title: str = Field(..., description="Distinctive working title.")
    core_genre: str = Field(..., description="Genre and camera/control framing.")
    hook_summary: str = Field(..., description="2-3 sentence high-concept hook.")
    core_loop_sketch: str = Field(..., description="Moment-to-moment player loop.")
    visual_and_audio_vibe: str = Field(default="", description="Signature visual and sonic aesthetic.")


class CoProducerConsultationCard(BaseEpistemicModel):
    """Expandable backchannel consultation or read-only specialist probe record attached to a Co-Producer turn."""

    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Synthesis reasoning across the consulted specialists or probe output.",
    )
    card_id: str = Field(..., description="Unique card identifier.")
    consultation_kind: Literal[
        "council_huddle",
        "divergent_pitch_round",
        "specialist_probe",
        "genre_foundation",
    ] = Field(
        ...,
        description="Type of backchannel studio consultation performed.",
    )
    question_posed: str = Field(..., description="Question or brief sent by the Co-Producer.")
    participating_roles: List[str] = Field(
        default_factory=list,
        description="Council roles or specialist persona consulted.",
    )
    synthesis_headline: str = Field(
        ...,
        description="One-line executive takeaway shown on the collapsed disclosure header.",
    )
    specialist_turns: List[CoProducerCouncilHuddleTurn] = Field(
        default_factory=list,
        description="Individual Council specialist huddle responses.",
    )
    divergent_pitches: List[CoProducerDivergentPitchOption] = Field(
        default_factory=list,
        description="Collided pitch candidates when consultation_kind == 'divergent_pitch_round'.",
    )
    probe_findings_markdown: str = Field(
        default="",
        description="Detailed findings when consultation_kind == 'specialist_probe'.",
    )
    proof_media_paths: List[str] = Field(
        default_factory=list,
        description="Any artifact paths inspected or captured during the probe.",
    )


class CoProducerMediaItem(BaseEpistemicModel):
    """Single media, metric, or code diff entry inside a CoProducerMediaCard."""

    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Why this specific media item or metric answers the user's question.",
    )
    item_id: str = Field(default="", description="Item identifier.")
    media_kind: Literal["video", "image", "audio", "metric", "code_diff"] = Field(
        ...,
        description="Render modality for this item.",
    )
    label: str = Field(..., description="Display title for this media item or metric.")
    caption: str = Field(default="", description="Contextual note, timestamp, or perceptual observation.")
    artifact_path: str = Field(
        default="",
        description="Cartridge-relative or _coproducer_drafts-relative path served via /api/artifacts/content.",
    )
    mime_type: str = Field(default="", description="MIME type hint (e.g. 'video/mp4', 'image/png', 'audio/wav').")
    width: int = Field(default=0, description="Intrinsic pixel width when known (prevents layout shift).")
    height: int = Field(default=0, description="Intrinsic pixel height when known (prevents layout shift).")
    metric_value: str = Field(default="", description="Formatted value when media_kind == 'metric'.")
    diff_unified: str = Field(default="", description="Unified diff snippet when media_kind == 'code_diff'.")


class CoProducerMediaCard(BaseEpistemicModel):
    """Typed rich inline media, progress, or diff presentation card rendered inside CoProducerView."""

    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Editorial rationale for curating and presenting this card inline.",
    )
    card_id: str = Field(default="", description="Unique card identifier.")
    card_kind: Literal[
        "playtest_video",
        "qa_clip",
        "image_gallery",
        "sketch_showcase",
        "audio_clips",
        "progress_summary",
        "code_diff",
    ] = Field(..., description="Presentation layout kind.")
    title: str = Field(..., description="Card headline.")
    subtitle: str = Field(default="", description="Optional secondary context (slice, session, task counts).")
    items: List[CoProducerMediaItem] = Field(default_factory=list, description="Ordered media/metric items.")
    deep_link_workspace: Optional[
        Literal["board", "writers-room", "bible", "memory", "assets", "playtest", "dna"]
    ] = Field(
        default=None,
        description="Optional Mission Control inspector tab to jump to for full telemetry.",
    )
    deep_link_target_id: str = Field(
        default="",
        description="Optional session_id, item_id, or slice_id hint for the deep link.",
    )
    show_by_default: bool = Field(
        default=True,
        description=(
            "True when the Co-Producer intentionally curated this card to appear in the primary reply flow; "
            "False when gathered as background inspection evidence and tucked into the Behind-the-Scenes drawer."
        ),
    )


class CoProducerMemoryMutationSpec(BaseEpistemicModel):
    """Explicit durable GameMemoryStore mutation proposed inside a Co-Producer Action Card."""

    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Why this memory should be recorded or superseded.",
    )
    operation: Literal["record", "supersede"] = Field(default="record")
    target_memory_id: str = Field(
        default="",
        description="Existing memory_id to supersede when operation == 'supersede'.",
    )
    memory_bank: Literal["council", "engineering", "visual", "audio", "narrative", "qa"] = Field(
        default="council"
    )
    kind: Literal[
        "strategic_goal",
        "architectural_decision",
        "lesson_learned",
        "domain_note",
    ] = Field(default="architectural_decision")
    title: str = Field(default="", description="Short rule or constraint title.")
    content: str = Field(default="", description="Concrete durable constraint or architectural note.")


class CoProducerActionProposal(BaseEpistemicModel):
    """Explicit 1-click user-confirmable studio mutation card separating conversation from execution."""

    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Justification for proposing this studio action and choosing its handoff/pivot tier.",
    )
    proposal_id: str = Field(default="", description="Unique proposal identifier.")
    action_kind: CoProducerActionKind = Field(..., description="Studio action to execute upon user confirmation.")
    handoff_mode: Optional[Literal["loose_spark", "refined_concept"]] = Field(
        default=None,
        description="For commission_game or pivot_fork_cartridge: 'loose_spark' enters Council genesis brainstorm; 'refined_concept' locks co-designed brief & skips blind concept genesis.",
    )
    title: str = Field(..., description="Clear action button/card headline.")
    summary: str = Field(..., description="Plain-language explanation of what will happen when confirmed.")
    steering_or_seed_brief: str = Field(
        default="",
        description="Complete directive or seed prompt sent to StudioDaemon / CreativeWritersRoom.",
    )
    pause_for_greenlight: bool = Field(
        default=False,
        description="When True on commission_game or pivot_fork_cartridge, pause after Game Bible sign-off before building.",
    )
    concept_brief: Optional[CoProducerConceptBrief] = Field(
        default=None,
        description="Structured concept brief snapshot for commission_game or pivot_fork_cartridge.",
    )
    cancel_item_ids: List[str] = Field(
        default_factory=list,
        description="For pivot_structural: obsolete ProductionBoard item_ids to cancel and unwire.",
    )
    memory_mutations: List[CoProducerMemoryMutationSpec] = Field(
        default_factory=list,
        description="Durable GameMemoryStore records or supersessions to apply on commit.",
    )
    carry_over_asset_paths: List[str] = Field(
        default_factory=list,
        description="For pivot_fork_cartridge or commission_game: relative asset/sketch paths to copy into the new cartridge.",
    )
    status: Literal["pending", "executed", "dismissed"] = Field(
        default="pending",
        description="Lifecycle state of this action proposal.",
    )


class CoProducerFinalTurn(BaseEpistemicModel):
    """Final synthesized 1-on-1 Co-Producer turn returned to the user."""

    deliberation: str = Field(
        ...,
        description="MANDATORY FIRST PROPERTY: Full showrunner and creative partner deliberation weighing user intent, cumulative prior commitments, specialist feedback, visual/audio evidence, and contrasting creative options.",
    )
    sketch_fidelity_audit: str = Field(
        default="",
        description=(
            "When an inline visual sketch was synthesized on this turn, audit the actual rendered PNG pixels against "
            "the user's explicit camera, spatial scale, aesthetic, and negative constraints (`explicit_non_goals`) BEFORE "
            "writing `spoken_response`. Explicitly note any discrepancy so `spoken_response` stays 100% honest."
        ),
    )
    updated_thread_title: str = Field(
        default="",
        description="Optional updated 2-6 word working title for this conversation thread.",
    )
    updated_concept_brief: Optional[CoProducerConceptBrief] = Field(
        default=None,
        description="Updated living game concept brief if ideation, style, verbs, or non-goals evolved.",
    )
    presented_card_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of `card_id`s from AUTO-STAGED MEDIA CARDS that you genuinely want displayed inline "
            "with your reply right now. Pass `[]` if you inspected progress/assets/diffs purely as background "
            "context and want a clean text-only reply; list specific `card_id`s when showing the user a sketch, "
            "playtest video, QA clip, or requested asset."
        ),
    )
    media_cards: List[CoProducerMediaCard] = Field(
        default_factory=list,
        description="Inline rich media, playtest video, sketch showcase, audio, progress, or diff cards.",
    )
    action_proposals: List[CoProducerActionProposal] = Field(
        default_factory=list,
        description="Explicit user-confirmable studio action cards when a concrete handoff, pivot, or studio command is ready.",
    )
    spoken_response: str = Field(
        ...,
        description="Direct 1-on-1 Co-Producer reply in GitHub-flavored Markdown. Speaks in a single creative, insightful partner voice.",
    )


def __getattr__(name: str) -> Any:
    if name == "OpenGenreCartridgeBundle":
        from engine.studio.swe import OpenGenreCartridgeBundle
        return OpenGenreCartridgeBundle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


