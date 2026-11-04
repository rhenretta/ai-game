"""engine/ui/models.py
Epistemic data contracts and response models for Studio Observability UI.
Strictly declares deliberation first across all models (field index 0).
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
import uuid
from pydantic import BaseModel, Field, field_validator


def _validate_deliberation(v: str) -> str:
    if not v or not v.strip():
        raise ValueError("deliberation must not be empty or whitespace")
    return v


class SwarmEventType(str, Enum):
    """Canonical event types emitted over the WebSocket streaming hub."""
    AGENT_STREAM = "AGENT_STREAM"
    TOOL_INVOCATION = "TOOL_INVOCATION"
    ARTIFACT_EMITTED = "ARTIFACT_EMITTED"
    QA_GATE_RESULT = "QA_GATE_RESULT"
    CARTRIDGE_PROMOTED = "CARTRIDGE_PROMOTED"
    STATE_CHANGE = "state_change"
    CONNECTED = "connected"
    ECHO = "echo"
    GENESIS_TRIGGERED = "genesis_triggered"
    REQUIREMENTS_SPEC_GENERATED = "REQUIREMENTS_SPEC_GENERATED"
    REQUIREMENT_STATUS_UPDATE = "REQUIREMENT_STATUS_UPDATE"
    CHALLENGER_GATE_VERDICT = "CHALLENGER_GATE_VERDICT"
    AGENT_TASK_RECORD = "AGENT_TASK_RECORD"
    CHECKPOINT_VERIFIED = "CHECKPOINT_VERIFIED"
    WRITERS_ROOM_UTTERANCE = "writers_room_utterance"
    CONCEPT_EVOLUTION_LOGGED = "concept_evolution_logged"
    BIBLE_MOCKUP_GENERATED = "bible_mockup_generated"
    GAME_BIBLE_UPDATED = "game_bible_updated"
    COPRODUCER_STEP_STARTED = "coproducer_step_started"
    COPRODUCER_TOOL_COMPLETED = "coproducer_tool_completed"
    COPRODUCER_TURN_COMPLETED = "coproducer_turn_completed"
    COPRODUCER_THREAD_UPDATED = "coproducer_thread_updated"


class PEVCycleInfo(BaseModel):
    """Current Plan-Execute-Verify lifecycle state of active swarm agents."""
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale on current PEV cycle phase")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    current_agent: Optional[str] = Field(default=None, description="Agent currently executing")
    phase: str = Field(default="IDLE", description="Cycle phase: IDLE, PLAN, EXECUTE, VERIFY")
    attempt_number: int = Field(default=0, description="Current retry attempt number")
    max_retries: int = Field(default=3, description="Maximum allowed PEV retries")
    last_plan: Optional[str] = Field(default=None, description="Summary of current plan")
    last_verification: Optional[Dict[str, Any]] = Field(default=None, description="Verification outcome")


class StudioStatusResponse(BaseModel):
    """Overall status of the autonomous game engine studio daemon."""
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale on current studio and swarm state")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    status: str = Field(default="ONLINE", description="Studio lifecycle state (e.g. ONLINE, IDLE, PROCESSING)")
    active_cartridge: Optional[str] = Field(default="active", description="Name of currently active cartridge")
    cartridges: List[str] = Field(default_factory=list, description="Available cartridge packages")
    agent_state: Dict[str, str] = Field(default_factory=dict, description="Active status per swarm agent")
    database_connected: bool = Field(default=True, description="SQLite WAL connection status")
    db_event_count: int = Field(default=0, description="Total recorded events in SQLite WAL telemetry store")
    supervisor_state: str = Field(default="IDLE", description="Current Build Supervisor state (IDLE, EVALUATING, PROMOTED)")
    current_episode: int = Field(default=0, description="Current active or target episode number")
    pev_cycle: Optional[PEVCycleInfo] = Field(default=None, description="Active or last PEV cycle state")
    uptime_seconds: float = Field(default=0.0, description="Uptime in seconds since studio boot")
    token_usage: Dict[str, Any] = Field(default_factory=dict, description="Cumulative Gemini API token usage and prefix cache hit telemetry")


class TelemetryEventsResponse(BaseModel):
    """Paginated collection of telemetry events queried from SQLite WAL."""
    deliberation: str = Field(..., min_length=1, description="Epistemic analysis of queried telemetry events")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    total_count: int = Field(default=0, description="Total matching events in database")
    limit: int = Field(default=50, description="Query limit applied")
    offset: int = Field(default=0, description="Query offset applied")
    events: List[Dict[str, Any]] = Field(default_factory=list, description="List of telemetry event records")


class ActiveCartridgeResponse(BaseModel):
    """Metadata, manifest, and asset inventory for the currently promoted active cartridge."""
    deliberation: str = Field(..., min_length=1, description="Epistemic analysis of currently active cartridge")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    name: str = Field(default="active", description="Cartridge identifier")
    path: str = Field(default="", description="Filesystem path to cartridge")
    exists: bool = Field(default=True, description="Whether active cartridge directory exists")
    manifest: Dict[str, Any] = Field(default_factory=dict, description="Parsed manifest.json contents")
    files: List[str] = Field(default_factory=list, description="List of files present in cartridge root")
    assets: Dict[str, List[str]] = Field(default_factory=dict, description="Discovered audio and sprite assets")
    episode: Optional[int] = Field(default=None, description="Episode number from manifest")
    has_audio: bool = Field(default=False, description="Whether playable audio files exist")
    has_sprites: bool = Field(default=False, description="Whether sprite/portrait images exist")


class DeathClusterPoint(BaseModel):
    """Spatial concentration of player death incidents."""
    deliberation: str = Field(default="Spatial concentration of player death incidents.", min_length=1, description="Epistemic analysis of clustered elimination coordinates")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    x: float = Field(..., description="Arena X coordinate (px)")
    y: float = Field(..., description="Arena Y coordinate (px)")
    death_count: int = Field(default=1, description="Number of deaths clustered at location")
    radius: float = Field(default=32.0, description="Dispersion radius (px)")
    primary_cause: str = Field(default="hazard", description="Dominant cause of death")


class VelocityVector(BaseModel):
    """Sampled player movement vector across arena grid."""
    deliberation: str = Field(default="Sampled player movement vector across arena grid.", min_length=1, description="Epistemic analysis of movement vector")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    x: float = Field(..., description="Grid cell center X coordinate")
    y: float = Field(..., description="Grid cell center Y coordinate")
    vx: float = Field(..., description="Horizontal velocity component (px/s)")
    vy: float = Field(..., description="Vertical velocity component (px/s)")
    speed: float = Field(..., description="Scalar movement speed (px/s)")
    direction_deg: float = Field(default=0.0, description="Movement angle in degrees [-180, 180]")


class HeatmapResponse(BaseModel):
    """2D spatial coordinate frequency matrix for player telemetry heatmap."""
    deliberation: str = Field(..., min_length=1, description="Epistemic analysis of 2D spatial occupancy grid")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    grid_width: int = Field(default=16, description="Horizontal grid dimension")
    grid_height: int = Field(default=16, description="Vertical grid dimension")
    total_samples: int = Field(default=0, description="Total telemetry location points aggregated")
    heatmap: List[List[float]] = Field(default_factory=list, description="2D frequency matrix [y][x]")
    death_clusters: List[DeathClusterPoint] = Field(default_factory=list, description="Aggregated player elimination clusters")
    velocity_vectors: List[VelocityVector] = Field(default_factory=list, description="Sampled movement vector field")
    world_bounds: Tuple[float, float, float, float] = Field(default=(0.0, 0.0, 1280.0, 720.0), description="Arena bounds")


class PaletteColorItem(BaseModel):
    """Single color chip within the 16-color Style Bible palette."""
    deliberation: str = Field(default="Style Bible 16-color palette item.", min_length=1, description="Epistemic description of color role")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    index: int = Field(..., description="Style Bible palette index [0..15]")
    name: str = Field(..., description="Canonical color name")
    rgb: Tuple[int, int, int] = Field(..., description="RGB components")
    hex: str = Field(..., description="Hexadecimal color string #RRGGBB")


class PaletteInfoResponse(BaseModel):
    """16-color Style Bible palette specification with semantic mood mappings."""
    deliberation: str = Field(..., min_length=1, description="Epistemic summary of Style Bible 16-color palette")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    palette_name: str = Field(default="Style Bible 16", description="Palette identifier")
    colors: List[PaletteColorItem] = Field(default_factory=list, description="16-color palette entries")
    mood_roles: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Mood to color role index mappings")


class CodeDiffResponse(BaseModel):
    """Unified code diff comparing candidate cartridge against active cartridge."""
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale on code changes between cartridge revisions")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    source_cartridge: str = Field(..., description="Base cartridge name (e.g. active)")
    candidate_cartridge: str = Field(..., description="Candidate cartridge name")
    module: str = Field(..., description="Module filename (e.g. mechanics.py)")
    diff: str = Field(..., description="Unified diff text")
    has_changes: bool = Field(default=False, description="True if differences exist")
    additions: int = Field(default=0, description="Count of added lines")
    deletions: int = Field(default=0, description="Count of deleted lines")


class GenesisTriggerRequest(BaseModel):
    """Payload to trigger an autonomous Genesis build cycle."""
    deliberation: str = Field(
        default="Requesting autonomous Genesis build from Steam library telemetry.",
        min_length=1,
        description="Operator rationale"
    )

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    profile_tag: str = Field(default="default", description="Steam player DNA profile tag")
    player_id: str = Field(default="default", description="Steam account or player identifier")


class GenesisTriggerResponse(BaseModel):
    """Response returned upon triggering a Genesis build."""
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale for dispatching Genesis build")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    job_id: str = Field(..., description="Unique job execution identifier")
    status: str = Field(default="triggered", description="Status of build dispatch")
    target_episode: int = Field(default=0, description="Target episode number")
    message: str = Field(default="", description="Human-readable status summary")


class SwarmStreamEvent(BaseModel):
    """Real-time event broadcast to WebSocket subscribers."""
    deliberation: str = Field(..., min_length=1, description="Agent deliberation thought or reasoning step")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    event_type: str = Field(..., description="Type of event (SwarmEventType value or string)")
    agent_name: str = Field(default="SwarmBus", description="Agent publishing the event")
    timestamp_utc: str = Field(..., description="ISO 8601 UTC timestamp")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event-specific metadata")


class TelemetryDashboardResponse(BaseModel):
    """Data-driven dynamic telemetry dashboard and hypotheses for active game."""
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale on active game telemetry and hypotheses")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    has_active_game: bool = Field(default=False, description="Whether active cartridge exists")
    game_title: str = Field(default="", description="Title of active game")
    genre: str = Field(default="", description="Genre of active game")
    manifest: Optional[Dict[str, Any]] = Field(default=None, description="Active telemetry manifest contents")
    probes: List[Dict[str, Any]] = Field(default_factory=list, description="Registered dynamic probes catalog")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Aggregated metric counters and stats")
    visualizations: List[Dict[str, Any]] = Field(default_factory=list, description="Visualizations declared in telemetry manifest")
    hypotheses: List[Union[Dict[str, Any], str]] = Field(default_factory=list, description="Player preference hypotheses")
    recent_custom_events: List[Dict[str, Any]] = Field(default_factory=list, description="Recent custom telemetry probe events")


class ArtifactInventoryResponse(BaseModel):
    """Complete inventory of multimodal and code artifacts for the active cartridge."""
    deliberation: str = Field(..., min_length=1, description="Epistemic inventory of generated cartridge artifacts")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    exists: bool = Field(default=False, description="Whether active cartridge exists")
    manifest: Dict[str, Any] = Field(default_factory=dict, description="Parsed manifest.json")
    world_spec: Optional[Dict[str, Any]] = Field(default=None, description="Parsed world_spec.json if present")
    narrative_bible: Optional[Dict[str, Any]] = Field(default=None, description="Parsed narrative_bible.json if present")
    audio_clips: List[Dict[str, Any]] = Field(default_factory=list, description="Discovered audio clips with metadata")
    sprites: List[Dict[str, Any]] = Field(default_factory=list, description="Discovered sprite and portrait assets")
    code_modules: List[str] = Field(default_factory=list, description="Discovered Python code module filenames")
    telemetry_manifest: Optional[Dict[str, Any]] = Field(default=None, description="Parsed telemetry_manifest.json if present")


class SteamDNAResponse(BaseModel):
    """Player Steam library profile insights and architectural gaming preferences."""
    deliberation: str = Field(..., min_length=1, description="Epistemic analysis of player Steam library")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    player_id: str = Field(default="local_player", description="Steam player identifier")
    profile_tag: str = Field(default="default", description="Player profile classification tag")
    total_playtime_hours: float = Field(default=0.0, description="Total aggregated playtime hours")
    total_games_analyzed: int = Field(default=0, description="Number of games analyzed")
    top_genres: List[str] = Field(default_factory=list, description="Top genres by playtime weight")
    genre_weights: Dict[str, float] = Field(default_factory=dict, description="Relative genre weight distribution")
    top_categories: List[str] = Field(default_factory=list, description="Top Steam categories by playtime weight")
    category_weights: Dict[str, float] = Field(default_factory=dict, description="Relative Steam category weight distribution")
    playtime_distribution: Dict[str, float] = Field(default_factory=dict, description="5-tier engagement and mastery playtime distribution")
    top_games: List[Dict[str, Any]] = Field(default_factory=list, description="Top games with playtime, appid, genres, tags, and engagement weights")
    activity_windows: Dict[str, Any] = Field(default_factory=dict, description="Gameplay activity breakdown across last_30_days, last_90_days, last_1_year, and lifetime horizons")
    pacing_preference: str = Field(default="balanced", description="Inferred gameplay pacing preference")
    pacing_score: float = Field(default=0.0, description="Pacing continuum score (-1.0 tactical to +1.0 kinetic)")
    challenge_tolerance: float = Field(default=0.5, description="Inferred challenge tolerance score (0.0 to 1.0)")
    thematic_biases: List[str] = Field(default_factory=list, description="Top thematic biases inferred from library tags")
    thematic_weights: Dict[str, float] = Field(default_factory=dict, description="Normalized thematic lexicon weights")
    mechanical_biases: Dict[str, float] = Field(default_factory=dict, description="Normalized mechanical bias scores (dodge_roll, ranged_aim, close_combat, speedrun)")
    preferred_palette_mood: str = Field(default="vibrant_fantasy", description="Inferred Style Bible 16-color palette mood")
    palette_weights: Dict[str, float] = Field(default_factory=dict, description="Relative affinity weights across Style Bible palette moods")
    created_at_utc: str = Field(default="", description="ISO 8601 UTC timestamp when DNA profile was synthesized")
    ingestion_provenance: Dict[str, Any] = Field(default_factory=dict, description="Provenance of local Steam VDF scan, Store API cache enrichment, and LLM epistemic synthesis")
    live_writers_room_dna_directive: str = Field(default="", description="Live formatted Target Player DNA directive synthesized from the current SteamPlayerDNA profile")


TelemetryDashboardResponse.model_rebuild()
ArtifactInventoryResponse.model_rebuild()
SteamDNAResponse.model_rebuild()


class RequirementsSpecPayload(BaseModel):
    """Payload for REQUIREMENTS_SPEC_GENERATED swarm events."""
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale on generated requirements specification")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    spec: Dict[str, Any] = Field(default_factory=dict, description="Full serialized GameRequirementsSpec dictionary")
    spec_id: Optional[str] = Field(default=None, description="Requirements specification ID")
    game_title: Optional[str] = Field(default=None, description="Title of the conceived game")
    total_requirements: int = Field(default=0, description="Total count of requirement tracks")


class RequirementStatusPayload(BaseModel):
    """Payload for REQUIREMENT_STATUS_UPDATE swarm events."""
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale on requirement track execution state")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    requirement_id: str = Field(default="", description="Requirement identifier, e.g. R1")
    status: str = Field(default="pending", description="Requirement status (pending, in_progress, completed, rejected)")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Execution and verification metrics")
    track_title: Optional[str] = Field(default=None, description="Title of the requirement track")


class ChallengerGateVerdictPayload(BaseModel):
    """Payload for CHALLENGER_GATE_VERDICT swarm events."""
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale on challenger gate design critic verdict")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    candidate_id: str = Field(default="", description="Candidate build identifier")
    passed: bool = Field(default=False, description="True if candidate satisfied 100% of requirements")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed critique and verification records")
    rejection_trace: Optional[Dict[str, Any]] = Field(default=None, description="Rejection diagnostic trace if rejected")


RequirementsSpecPayload.model_rebuild()
RequirementStatusPayload.model_rebuild()
ChallengerGateVerdictPayload.model_rebuild()


class ArtifactMetadataEnvelope(BaseModel):
    """Structured cryptographic envelope and handoff documentation for artifacts.
    Enforces Rule 4: deliberation declared strictly first (field index 0).
    """
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale explaining why this artifact was produced")

    class _DualSealMethod:
        deliberation: str = "Helper descriptor for dual seal invocation."

        def __get__(self, instance, owner):
            if instance is None:
                def class_seal(*args, **kwargs):
                    return owner._class_seal(*args, **kwargs)
                return class_seal
            else:
                def instance_seal(*args, **kwargs):
                    return instance._instance_seal(*args, **kwargs)
                return instance_seal

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique artifact identifier UUID")
    artifact_name: str = Field(default="artifact", description="Canonical filename or artifact label, e.g. 'mechanics.py', 'hero.png'")
    artifact_type: str = Field(default="code", description="Categorical artifact type: 'code', 'sprite', 'audio', 'json', 'spec', 'report'")
    file_path: Optional[str] = Field(default=None, description="Relative or absolute filesystem path to artifact")
    payload_path: Optional[str] = Field(default=None, description="Path to external disk file holding raw payload")
    sha256_hash: str = Field(default="", description="Cryptographic SHA256 hex digest (64 chars) of payload or file")
    output_sha256: Optional[str] = Field(default=None, description="Provenance tracking: SHA256 of the output artifact")
    input_sha256: Optional[str] = Field(default=None, description="Provenance tracking: SHA256 of the predecessor input")
    byte_size: int = Field(default=0, ge=0, description="Size of artifact in bytes")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO 8601 UTC timestamp")
    sealed_at_utc: Optional[str] = Field(default=None, description="ISO 8601 UTC sealing timestamp")
    source_agent: str = Field(default="", description="Producing specialist agent role")
    target_agent: Optional[str] = Field(default=None, description="Intended consumer agent role")
    metadata_summary: Dict[str, Any] = Field(default_factory=dict, description="Structured metadata summary (e.g. lines, ast_nodes, duration_s, sample_rate)")

    @classmethod
    def compute_sha256_bytes(cls, data: bytes) -> str:
        import hashlib
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def compute_file_sha256(cls, file_path: Union[str, Path]) -> str:
        import hashlib
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _instance_seal(self, file_path: Optional[Union[str, Path]] = None, artifact_name: Optional[str] = None) -> "ArtifactMetadataEnvelope":
        target = file_path or self.file_path or self.payload_path
        if target:
            p = Path(target)
            self.file_path = str(p)
            self.payload_path = str(p)
            if artifact_name:
                self.artifact_name = artifact_name
            elif not self.artifact_name or self.artifact_name == "artifact":
                self.artifact_name = p.name
            if p.is_file():
                self.byte_size = p.stat().st_size
                h = self.compute_file_sha256(p)
                self.sha256_hash = h
                self.output_sha256 = h
        now_iso = datetime.now(timezone.utc).isoformat()
        self.created_at_utc = now_iso
        self.sealed_at_utc = now_iso
        return self

    @classmethod
    def _class_seal(
        cls,
        deliberation: str,
        artifact_name: Optional[str] = None,
        artifact_type: str = "code",
        file_path: Optional[Union[str, Path]] = None,
        payload_path: Optional[Union[str, Path]] = None,
        payload_bytes: Optional[bytes] = None,
        source_agent: str = "",
        target_agent: Optional[str] = None,
        metadata_summary: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ArtifactMetadataEnvelope:
        import hashlib
        target_file = file_path or payload_path or kwargs.get("target_file")
        calculated_hash = ""
        byte_size = 0
        resolved_path = None
        if target_file:
            p = Path(target_file)
            resolved_path = str(p)
            if p.is_file():
                byte_size = p.stat().st_size
                calculated_hash = cls.compute_file_sha256(p)
        elif payload_bytes is not None:
            byte_size = len(payload_bytes)
            calculated_hash = cls.compute_sha256_bytes(payload_bytes)
        else:
            calculated_hash = hashlib.sha256(b"").hexdigest()

        now_iso = datetime.now(timezone.utc).isoformat()
        return cls(
            deliberation=deliberation,
            artifact_name=artifact_name or (Path(resolved_path).name if resolved_path else "artifact"),
            artifact_type=str(artifact_type),
            file_path=resolved_path,
            payload_path=resolved_path,
            sha256_hash=calculated_hash,
            output_sha256=calculated_hash,
            byte_size=byte_size,
            created_at_utc=now_iso,
            sealed_at_utc=now_iso,
            source_agent=source_agent,
            target_agent=target_agent,
            metadata_summary=metadata_summary or {},
        )

    seal: ClassVar[Any] = _DualSealMethod()

    def verify_integrity(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        target = file_path or self.file_path or self.payload_path
        h = self.output_sha256 or self.sha256_hash
        if not h or len(h) != 64:
            return False
        if target:
            p = Path(target)
            if not p.is_file():
                return False
            return self.compute_file_sha256(p) == h
        return True


class AgentTaskRecord(BaseModel):
    """Structured record of an agent or subagent task, reasoning, and handoff documentation.
    Strictly declares deliberation as field index 0 (Rule 4).
    """
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale and reasoning executed during this task")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}", description="Unique task execution identifier (e.g. 'task_swe_r1_1a2b3c4d')")
    agent_role: str = Field(default="StudioAgent", description="Canonical agent persona identifier (e.g. 'LeadGameplaySWEAgent')")
    agent_name: str = Field(default="", description="Human-readable or short alias (e.g. 'GameplaySWE')")
    requirement_id: Optional[str] = Field(default=None, description="Associated requirement track ID if applicable (e.g. 'R1')")
    track_title: Optional[str] = Field(default=None, description="Associated requirement track title")
    task_type: str = Field(
        default="EXECUTION",
        description="Categorical task type: 'PLANNING', 'SPECIFICATION', 'ASSET_SYNTHESIS', 'CODE_IMPLEMENTATION', 'INSTRUMENTATION', 'QA_VERIFICATION', 'CHALLENGER_CRITIC'",
    )
    status: str = Field(default="COMPLETED", description="Task lifecycle status: 'PENDING', 'IN_PROGRESS', 'COMPLETED', 'REJECTED', 'FAILED'")
    start_time_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO 8601 UTC task initiation timestamp")
    end_time_utc: Optional[str] = Field(default=None, description="ISO 8601 UTC task completion timestamp")
    duration_ms: Optional[float] = Field(default=None, description="Execution duration in milliseconds")
    input_artifacts: List[Union[ArtifactMetadataEnvelope, Dict[str, Any], str]] = Field(default_factory=list, description="Handoff input artifacts consumed by this task")
    output_artifacts: List[Union[ArtifactMetadataEnvelope, Dict[str, Any], str]] = Field(default_factory=list, description="Handoff output artifacts produced and sealed by this task")
    verification_verdict: Optional[Dict[str, Any]] = Field(default=None, description="Verification outcome or QA gate check results")
    diagnostic_trace: Optional[Dict[str, Any]] = Field(default=None, description="Rejection diagnostic trace or error stack if failed/rejected")
    summary: Optional[str] = Field(default=None, description="Human-readable concise summary of task accomplishments")


class AgentTaskHistoryResponse(BaseModel):
    """Chronological history of agent and subagent tasks with reasoning and handoffs.
    Strictly declares deliberation as field index 0 (Rule 4).
    """
    deliberation: str = Field(..., min_length=1, description="Epistemic analysis of queried agent task history")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    total_tasks: int = Field(default=0, description="Total matching tasks in store")
    total_count: int = Field(default=0, description="Compatibility alias for total_tasks")
    limit: int = Field(default=50, description="Applied query limit")
    offset: int = Field(default=0, description="Applied query offset")
    tasks: List[AgentTaskRecord] = Field(default_factory=list, description="List of chronologically ordered task records")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary metrics (completed, in_progress, failed, active_agents counts)")


class RequirementsBacklogResponse(BaseModel):
    """Live requirements backlog and per-track implementation and verification status.
    Strictly declares deliberation as field index 0 (Rule 4).
    """
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale on requirements specification and backlog status")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    has_active_spec: bool = Field(default=False, description="True if a valid GameRequirementsSpec exists")
    has_spec: bool = Field(default=False, description="Compatibility alias for has_active_spec")
    spec_id: Optional[str] = Field(default=None, description="Unique requirements spec ID")
    game_title: str = Field(default="", description="Title of the conceived game")
    genre_paradigm: str = Field(default="", description="Genre classification or blend")
    target_episode: int = Field(default=0, description="Target build episode")
    total_tracks: int = Field(default=0, description="Total requirement tracks")
    total_requirements: int = Field(default=0, description="Compatibility alias for total_tracks")
    completed_tracks: int = Field(default=0, description="Tracks with status 'completed' or 'satisfied'")
    completed_requirements: int = Field(default=0, description="Compatibility alias for completed_tracks")
    in_progress_tracks: int = Field(default=0, description="Tracks currently 'in_progress'")
    pending_tracks: int = Field(default=0, description="Tracks awaiting implementation")
    rejected_tracks: int = Field(default=0, description="Tracks rejected by Challenger gate")
    progress_percentage: float = Field(default=0.0, description="Percentage of tracks satisfied (0.0 to 100.0)")
    requirements: List[Dict[str, Any]] = Field(default_factory=list, description="Serialized RequirementTrack objects")
    requirements_spec: Optional[Dict[str, Any]] = Field(default=None, description="Complete serialized GameRequirementsSpec object")
    asset_manifest: List[Dict[str, Any]] = Field(default_factory=list, description="Serialized AssetRequirement objects")
    last_updated_utc: Optional[str] = Field(default=None, description="Timestamp of latest status update")


class AgentTaskDetailResponse(BaseModel):
    """Detailed view of an individual agent task, full reasoning, and handoff documentation.
    Strictly declares deliberation as field index 0 (Rule 4).
    """
    deliberation: str = Field(..., min_length=1, description="Epistemic evaluation of task execution and handoff validity")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    task: Optional[AgentTaskRecord] = Field(default=None, description="Complete task execution record")
    task_id: Optional[str] = Field(default=None, description="Direct task_id for top-level convenience")
    handoff_documentation: Dict[str, Any] = Field(default_factory=dict, description="Detailed handoff documentation package")


ArtifactMetadataEnvelope.model_rebuild()
AgentTaskRecord.model_rebuild()
AgentTaskHistoryResponse.model_rebuild()
RequirementsBacklogResponse.model_rebuild()
AgentTaskDetailResponse.model_rebuild()


class VerifiedCheckpointRecord(BaseModel):
    """Immutable metadata record for a verified incremental cartridge checkpoint.
    Strictly declares deliberation as field index 0 (Rule 4).
    """
    deliberation: str = Field(..., min_length=1, description="Epistemic rationale describing the verified checkpoint")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    checkpoint_id: str = Field(default="checkpoint_default", description="Unique checkpoint ID (e.g. candidate_genesis_8993a2_R1)")
    candidate_id: str = Field(default="candidate_default", description="Candidate build ID")
    requirement_id: str = Field(default="R1", description="Requirement track ID verified (e.g. R1)")
    track_title: str = Field(default="", description="Title of the verified requirement track")
    created_at_utc: str = Field(default="2026-10-17T00:00:00Z", description="ISO 8601 UTC creation timestamp")
    verified_tracks: List[str] = Field(default_factory=list, description="All requirement IDs verified up to this checkpoint")
    cartridge_path: str = Field(default="", description="Filesystem path to checkpoint cartridge directory")
    is_active: bool = Field(default=False, description="Whether this checkpoint is currently mirrored to cartridges/active/")
    files: List[str] = Field(default_factory=list, description="List of filenames contained in checkpoint")
    sha256_manifest: Dict[str, str] = Field(default_factory=dict, description="Cryptographic SHA256 hashes of all checkpoint files")


class CheckpointsResponse(BaseModel):
    """List of all verified checkpoints available for live play.
    Strictly declares deliberation as field index 0 (Rule 4).
    """
    deliberation: str = Field(..., min_length=1, description="Epistemic summary of available verified checkpoints")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    total_checkpoints: int = Field(default=0, description="Total verified checkpoints available")
    latest_checkpoint_id: Optional[str] = Field(default=None, description="Identifier of latest verified checkpoint")
    checkpoints: List[VerifiedCheckpointRecord] = Field(default_factory=list, description="Chronological list of verified checkpoints")


VerifiedCheckpointRecord.model_rebuild()
CheckpointsResponse.model_rebuild()


class GreenlightRequest(BaseModel):
    """Payload to approve the Living Game Bible or inject Executive Producer steering feedback."""
    deliberation: str = Field(
        default="Executive Producer greenlight or steering feedback submitted from Mission Control.",
        min_length=1,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale for greenlight decision or steering note",
    )

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    approved: bool = Field(default=True, description="True to greenlight the Game Bible and proceed to code synthesis")
    feedback: str = Field(default="", description="Optional steering feedback to inject into the Writers' Room")


class StudioSettingsUpdateRequest(BaseModel):
    """Payload to update studio runtime settings such as pause_for_greenlight."""
    deliberation: str = Field(
        default="Updating Mission Control studio configuration.",
        min_length=1,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale for configuration update",
    )

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    pause_for_greenlight: bool = Field(
        default=False,
        description="When True, pause at AWAITING_GREENLIGHT after the Writers' Room locks the Game Bible.",
    )
    paused: Optional[bool] = Field(
        default=None,
        description="When True, pauses active build workers and Writers' Room turns until resumed.",
    )
    candidate_id: Optional[str] = Field(
        default=None,
        description="Optional explicit candidate cartridge ID to resume when unpausing an idle studio.",
    )
    auto_spawn_resume: bool = Field(
        default=True,
        description="When True (default), unpausing an idle studio automatically spawns a resume worker for the target cartridge.",
    )


class GameLaunchRequest(BaseModel):
    """Payload to launch a playable game cartridge in a native Godot 4.3 / game engine window.
    Strictly declares deliberation as field index 0 (Rule 4).
    """
    deliberation: str = Field(
        default="Launching current game cartridge from Mission Control UI.",
        min_length=1,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale for launching the game cartridge",
    )

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    cartridge_id: Optional[str] = Field(
        default=None,
        description="Optional cartridge or checkpoint identifier to launch (defaults to currently selected or active cartridge)",
    )


class GameLaunchResponse(BaseModel):
    """Response returned when launching a game cartridge from the Mission Control UI.
    Strictly declares deliberation as field index 0 (Rule 4).
    """
    deliberation: str = Field(..., min_length=1, description="Epistemic summary of the launched game session")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    status: str = Field(default="launched", description="Launch status ('launched' or 'error')")
    cartridge_id: str = Field(default="active", description="Identifier of the launched cartridge")
    cartridge_path: str = Field(default="", description="Resolved filesystem path of the launched cartridge")
    game_title: str = Field(default="", description="Human-readable game title")
    pid: Optional[int] = Field(default=None, description="Spawned game process PID")


class CoProducerCreateDraftRequest(BaseModel):
    """Payload to open a fresh pre-genesis Draft Game Concept thread in Mission Control."""

    deliberation: str = Field(
        default="Creating a new pre-genesis Co-Producer draft concept thread.",
        min_length=1,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale for creating draft thread",
    )

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    title: str = Field(default="New Game Concept", description="Optional initial working title")


class CoProducerSendMessageRequest(BaseModel):
    """Payload to send a user message to the interactive 1-on-1 Co-Producer."""

    deliberation: str = Field(
        default="Submitting user message to Studio Co-Producer.",
        min_length=1,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale for user message submission",
    )

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    message: str = Field(..., min_length=1, description="User message text")
    selected_sketch_paths: List[str] = Field(
        default_factory=list,
        description="Optional relative paths of user-selected sketches/mockups for visual reference",
    )


class CoProducerExecuteActionRequest(BaseModel):
    """Payload to confirm and execute a CoProducerActionProposal from Mission Control."""

    deliberation: str = Field(
        default="Confirming explicit Co-Producer studio action proposal.",
        min_length=1,
        description="MANDATORY FIRST PROPERTY: Epistemic rationale for executing studio action",
    )

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    thread_id: str = Field(..., min_length=1, description="Co-Producer thread identifier")
    message_id: str = Field(
        default="",
        description="Assistant message containing the proposal (optional when direct_action_kind is set)",
    )
    proposal_id: str = Field(
        default="",
        description="Proposal identifier to execute (optional when direct_action_kind is set)",
    )
    direct_action_kind: Optional[str] = Field(
        default=None,
        description="Optional direct producer-initiated action kind (e.g. 'commission_game') when launching straight from an active draft thread",
    )
    carry_over_asset_paths: List[str] = Field(
        default_factory=list,
        description="Optional sketch/mockup asset paths pinned by the producer for handoff",
    )
    override_brief: Optional[str] = Field(
        default=None,
        description="Optional user-edited steering or seed brief overriding the proposal default",
    )
    override_handoff_mode: Optional[str] = Field(
        default=None,
        description="Optional override for handoff_mode ('loose_spark' | 'refined_concept')",
    )
    override_pause_for_greenlight: Optional[bool] = Field(
        default=None,
        description="Optional override for pause_for_greenlight",
    )
    dismiss: bool = Field(
        default=False,
        description="When True, dismisses the proposal without executing any studio mutation",
    )


class CoProducerThreadDetailResponse(BaseModel):
    """Full conversation history and concept brief for a Co-Producer thread."""

    deliberation: str = Field(..., min_length=1, description="Epistemic summary of loaded Co-Producer thread")

    @field_validator("deliberation")
    @classmethod
    def validate_deliberation(cls, v: str) -> str:
        return _validate_deliberation(v)

    thread: Dict[str, Any] = Field(default_factory=dict, description="Thread metadata and evolving concept brief")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Ordered messages with cards and traces")
