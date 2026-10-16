"""engine/ui/models.py
Epistemic data contracts and response models for Studio Observability UI.
Strictly declares deliberation first across all models (field index 0).
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
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
    top_games: List[Dict[str, Any]] = Field(default_factory=list, description="Top games with playtime and appid")
    pacing_preference: str = Field(default="balanced", description="Inferred gameplay pacing preference")
    pacing_score: float = Field(default=0.0, description="Pacing continuum score (-1.0 tactical to +1.0 kinetic)")
    challenge_tolerance: float = Field(default=0.5, description="Inferred challenge tolerance score (0.0 to 1.0)")


TelemetryDashboardResponse.model_rebuild()
ArtifactInventoryResponse.model_rebuild()
SteamDNAResponse.model_rebuild()


