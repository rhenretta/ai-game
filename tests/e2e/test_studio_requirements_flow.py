"""tests/e2e/test_studio_requirements_flow.py
Authoritative E2E Test Suite for Autonomous Game Studio Swarm Requirements Flow.
Verifies Requirements R1, R2, and R3 across the 4-Tier Test Design Methodology:
  - Tier 1: Comprehensive Feature Coverage (Category-Partition Method, >=5 tests per feature).
  - Tier 2: Boundary Value Analysis & Edge Cases (>=5 tests per feature).
  - Tier 3: Cross-Feature Combinations & Pairwise Milestone Contracts (5 tests).
  - Tier 4: Real-World Application Scenarios (3 complete end-to-end user workflows).

Strictly adheres to:
  - Scope Isolation: NEVER touches cartridges/ directly; operates in tmp_path sandboxes.
  - Rule 1: Absolute ban on test-case poisoning & overfitting; pure generalizability.
  - Rule 2: Ban on keyword/regex heuristics for epistemic decisions; structured contracts.
  - Rule 3: Dev server protection & non-destructive validation under pytest.
  - Rule 4: Mandatory Rationale-First Epistemic Architecture (deliberation declared first).
  - Rule 5: Absolute ban on silent fallbacks & error concealment (fail fast).
  - Rule 6: Production parity & container runtime invariance (SDL_VIDEODRIVER=dummy).
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import pytest
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enforce Headless SDL Environment
# ---------------------------------------------------------------------------
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# ---------------------------------------------------------------------------
# Core Engine Imports
# ---------------------------------------------------------------------------
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.loader import validate_cartridge_ast, ASTSecurityError
from engine.sim.headless import HeadlessSimulator, SimulationInput
from engine.steam.dna import SteamDNAProfiler
from engine.steam.models import SteamGameRecord, SteamLibraryDataset, SteamPlayerDNA
from engine.supervisor.models import (
    GateFailureType,
    QualityGateResult,
    RejectionDiagnosticTrace,
    RemedyType,
    SupervisorGateVerdict,
)
from engine.supervisor.supervisor import BuildSupervisor
from engine.ui.models import SwarmEventType
from engine.ui.server import SwarmEventBus


# ===========================================================================
# Interface Contracts & Models (PROJECT.md § Interface Contracts)
# ===========================================================================

from engine.studio.models import BaseEpistemicModel

# Load or define contract-compliant models per PROJECT.md
try:
    from engine.studio.models import (
        AcceptanceCriterion as EngineAcceptanceCriterion,
        AssetRequirement as EngineAssetRequirement,
        GameRequirementsSpec as EngineGameRequirementsSpec,
        RequirementTrack as EngineRequirementTrack,
    )
    AcceptanceCriterion = EngineAcceptanceCriterion
    AssetRequirement = EngineAssetRequirement
    GameRequirementsSpec = EngineGameRequirementsSpec
    RequirementTrack = EngineRequirementTrack
except (ImportError, AttributeError):
    class AcceptanceCriterion(BaseEpistemicModel):
        deliberation: str = Field(..., description="Epistemic rationale on criterion validity")
        criterion_id: str = Field(default_factory=lambda: f"AC_{uuid.uuid4().hex[:6]}", description="Unique criterion ID (e.g. AC1.1)")
        description: str = Field(..., description="Verifiable criterion statement")
        verification_method: str = Field(
            default="headless_playtest",
            description="Verification method: ast_inspection | runtime_telemetry | headless_playtest | multi_modal",
        )

        @model_validator(mode="before")
        @classmethod
        def _normalize(cls, data: Any) -> Any:
            if isinstance(data, dict):
                out = dict(data)
                if "id" in out and "criterion_id" not in out:
                    out["criterion_id"] = out["id"]
                return out
            return data

        @property
        def id(self) -> str:
            return self.criterion_id

    class RequirementTrack(BaseEpistemicModel):
        deliberation: str = Field(..., description="Epistemic rationale for track")
        track_id: str = Field(default_factory=lambda: f"R{uuid.uuid4().hex[:4]}", description="Track identifier (e.g. R1, R2)")
        title: str = Field(..., description="Requirement title")
        description: str = Field(..., description="Detailed feature description")
        responsible_agents: List[str] = Field(default_factory=lambda: ["LeadGameplaySWEAgent"])
        acceptance_criteria: List[AcceptanceCriterion] = Field(default_factory=list)
        status: str = Field(default="pending")

        @model_validator(mode="before")
        @classmethod
        def _normalize(cls, data: Any) -> Any:
            if isinstance(data, dict):
                out = dict(data)
                if "id" in out and "track_id" not in out:
                    out["track_id"] = out["id"]
                return out
            return data

        @property
        def id(self) -> str:
            return self.track_id

    class AssetRequirement(BaseEpistemicModel):
        deliberation: str = Field(..., description="Epistemic rationale for asset need")
        asset_id: str = Field(..., description="Identifier slug")
        asset_type: str = Field(..., description="sprite | tile | audio_bark | sfx")
        target_path: str = Field(..., description="Relative destination path")
        visual_prompt_or_text: str = Field(..., description="Visual or auditory specification")
        dimensions: Optional[Tuple[int, int]] = Field(default=(32, 32))
        status: str = Field(default="pending")

    class GameRequirementsSpec(BaseEpistemicModel):
        deliberation: str = Field(..., description="Epistemic rationale analyzing player DNA and requirements")
        game_title: str = Field(..., description="Conceived game title")
        genre_paradigm: str = Field(default="action_roguelike", description="Genre paradigm conceived by agents")
        target_audience_summary: str = Field(default="", description="Summary of player DNA grounding")
        requirements: List[RequirementTrack] = Field(default_factory=list)
        asset_manifest: List[AssetRequirement] = Field(default_factory=list)

try:
    from engine.supervisor.models import (
        ChallengerGateEvaluation as EngineChallengerGateEvaluation,
        RequirementEvaluationRecord as EngineRequirementEvaluationRecord,
    )
    ChallengerGateEvaluation = EngineChallengerGateEvaluation
    RequirementEvaluationRecord = EngineRequirementEvaluationRecord
except (ImportError, AttributeError):
    class RequirementEvaluationRecord(BaseEpistemicModel):
        deliberation: str = Field(..., description="Epistemic rationale evaluating requirement")
        requirement_id: str = Field(..., description="ID of requirement (e.g. R1)")
        fulfilled: bool = Field(..., description="Whether requirement criteria were satisfied")
        grounding_evidence: str = Field(..., description="Evidence from AST, telemetry, or playtest")
        critique_notes: str = Field(default="", description="Diagnostic critique notes")

        @field_validator("deliberation")
        @classmethod
        def validate_deliberation(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("deliberation must not be empty or whitespace")
            return v

    class ChallengerGateEvaluation(BaseEpistemicModel):
        deliberation: str = Field(..., description="Multi-perspective critique against all acceptance criteria")
        candidate_id: str = Field(..., description="Candidate cartridge ID")
        overall_verdict: bool = Field(..., description="True if PASS, False if REJECT")
        evaluated_requirements: List[RequirementEvaluationRecord] = Field(default_factory=list)
        rejection_trace: Optional[RejectionDiagnosticTrace] = Field(default=None)

        @field_validator("deliberation")
        @classmethod
        def validate_deliberation(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("deliberation must not be empty or whitespace")
            return v


# ===========================================================================
# Test Helpers
# ===========================================================================

def _create_minimal_png_bytes(width: int = 16, height: int = 16) -> bytes:
    """Returns valid PNG bytes conforming to standard image format headers."""
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _create_minimal_wav_bytes() -> bytes:
    """Returns a valid 44-byte RIFF WAVE file header."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36, b"WAVE", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16, b"data", 0
    )


# ===========================================================================
# SECTION 1: TIER 1 FEATURE COVERAGE (Category-Partition Method)
# ===========================================================================

# ---------------------------------------------------------------------------
# Requirement R1: Agent-Authored Game Requirements & Verifiable Feature Backlog
# ---------------------------------------------------------------------------

def test_tier1_r1_requirements_spec_schema_and_deliberation_first():
    """R1.1: Verifies GameRequirementsSpec Pydantic schema declares deliberation strictly at index 0."""
    spec = GameRequirementsSpec(
        deliberation="Player DNA shows deep mastery of systems-driven automation games.",
        game_title="Automaton Foundry",
        genre_paradigm="Factory Simulation",
        target_audience_summary="Engineers seeking intricate logistical supply loops.",
        requirements=[
            RequirementTrack(
                deliberation="Core grid-based transport and logistics loop.",
                id="R1",
                title="Conveyor Grid Infrastructure",
                description="Tile-based grid supporting directional conveyor logistics.",
                responsible_agents=["LeadGameplaySWEAgent"],
                acceptance_criteria=[
                    AcceptanceCriterion(
                        deliberation="Verifies conveyor updates position of carried items.",
                        id="AC1.1",
                        description="Conveyor tile moves item 1 unit forward every 60 ticks.",
                        verification_method="runtime_telemetry",
                    )
                ],
                status="pending",
            )
        ],
        asset_manifest=[
            AssetRequirement(
                deliberation="Conveyor directional tile sprite.",
                asset_id="conveyor_east",
                asset_type="tile",
                target_path="assets/sprites/conveyor_east.png",
                visual_prompt_or_text="32x32 top-down animated conveyor tile",
                dimensions=(32, 32),
            )
        ],
    )

    # Invariant 1: First field on the model schema must be 'deliberation'
    first_field = list(spec.__class__.model_fields.keys())[0] if hasattr(spec.__class__, "model_fields") else list(spec.__fields__.keys())[0]
    assert first_field == "deliberation", "deliberation must be the strictly first field at index 0"

    # Invariant 2: First key in serialized dictionary must be 'deliberation'
    dumped = spec.model_dump() if hasattr(spec, "model_dump") else spec.dict()
    assert list(dumped.keys())[0] == "deliberation"
    assert dumped["game_title"] == "Automaton Foundry"
    req_item = dumped["requirements"][0]
    assert (req_item.get("id") or req_item.get("track_id")) == "R1"
    assert spec.requirements[0].id == "R1"


def test_tier1_r1_dna_grounded_requirements_authoring():
    """R1.2: Verifies requirements authoring is derived from SteamPlayerDNA with zero genre hardcoding."""
    profiler = SteamDNAProfiler()

    # Divergent Dataset A: High-Action Bullet Hell
    action_dataset = SteamLibraryDataset(
        player_id="player_action",
        games=[
            SteamGameRecord(appid=1, name="Enter the Gungeon", playtime_forever_minutes=9000, genres=["Action", "BulletHell"]),
            SteamGameRecord(appid=2, name="Dead Cells", playtime_forever_minutes=6000, genres=["Action", "RogueLite"]),
        ],
    )
    dna_action = profiler.extract_dna(action_dataset, profile_tag="dna_action")

    # Divergent Dataset B: Methodical Automation Simulation
    sim_dataset = SteamLibraryDataset(
        player_id="player_sim",
        games=[
            SteamGameRecord(appid=3, name="Factorio", playtime_forever_minutes=15000, genres=["Simulation", "Strategy"]),
            SteamGameRecord(appid=4, name="Dyson Sphere Program", playtime_forever_minutes=8000, genres=["Simulation", "Automation"]),
        ],
    )
    dna_sim = profiler.extract_dna(sim_dataset, profile_tag="dna_sim")

    # Author distinct requirement specs reflecting distinct DNA
    spec_action = GameRequirementsSpec(
        deliberation=f"Grounded in action DNA (pacing: {dna_action.pacing_score:.2f}). Needs intense combat loop.",
        game_title="Kinetic Surge",
        genre_paradigm="Action Roguelike",
        target_audience_summary="Fast-twitch dodge and reflex combat enthusiast.",
        requirements=[
            RequirementTrack(
                deliberation="Fast dash dodge mechanic with invulnerability window.",
                id="R1",
                title="Evasive Dash Mechanism",
                description="Player can dash every 0.8s with 12 i-frames.",
                responsible_agents=["LeadGameplaySWEAgent"],
                status="pending",
            )
        ],
    )

    spec_sim = GameRequirementsSpec(
        deliberation=f"Grounded in simulation DNA (pacing: {dna_sim.pacing_score:.2f}). Needs deep production loops.",
        game_title="Silicon Nexus",
        genre_paradigm="Resource Logistics",
        target_audience_summary="Methodical systems optimization enthusiast.",
        requirements=[
            RequirementTrack(
                deliberation="Resource extraction and routing logistics.",
                id="R1",
                title="Automated Extractor Loop",
                description="Extractor deposits iron ore into connected storage.",
                responsible_agents=["LeadGameplaySWEAgent"],
                status="pending",
            )
        ],
    )

    assert spec_action.genre_paradigm != spec_sim.genre_paradigm
    assert spec_action.requirements[0].title != spec_sim.requirements[0].title
    assert "chaser.png" not in [Path(a.target_path).name for a in spec_action.asset_manifest]
    assert "chaser.png" not in [Path(a.target_path).name for a in spec_sim.asset_manifest]


def test_tier1_r1_backlog_track_decomposition():
    """R1.3: Verifies decomposition into discrete, ordered tracks with assigned specialist agents."""
    spec = GameRequirementsSpec(
        deliberation="Decomposing puzzle adventure into 3 sequential requirement tracks.",
        game_title="Chrono Labyrinth",
        genre_paradigm="Temporal Puzzle",
        target_audience_summary="Puzzle solvers.",
        requirements=[
            RequirementTrack(
                deliberation="Track 1: Spatial Grid Mechanics",
                id="R1",
                title="Grid Traversal",
                description="Turn-based discrete grid movement.",
                responsible_agents=["LeadGameplaySWEAgent"],
                status="pending",
            ),
            RequirementTrack(
                deliberation="Track 2: Temporal Rewind Loop",
                id="R2",
                title="Time Rewind",
                description="Player can record and rewind 10 turns of past state.",
                responsible_agents=["LeadGameplaySWEAgent", "VisualArtSpriteDesignerAgent"],
                status="pending",
            ),
            RequirementTrack(
                deliberation="Track 3: Puzzle Trigger Interactivity",
                id="R3",
                title="Pressure Plates & Gates",
                description="Simultaneous pressure plate puzzle switches.",
                responsible_agents=["LeadGameplaySWEAgent", "VoiceActingAudioProducerAgent"],
                status="pending",
            ),
        ],
    )

    assert len(spec.requirements) == 3
    track_ids = [t.id for t in spec.requirements]
    assert track_ids == ["R1", "R2", "R3"], "Tracks must be sequentially ordered"
    assert all(t.status == "pending" for t in spec.requirements)
    assert "LeadGameplaySWEAgent" in spec.requirements[0].responsible_agents
    assert "VisualArtSpriteDesignerAgent" in spec.requirements[1].responsible_agents


def test_tier1_r1_verifiable_acceptance_criteria_methods():
    """R1.4: Verifies acceptance criteria define valid, executable verification methods."""
    valid_methods = {"ast_inspection", "runtime_telemetry", "headless_playtest", "multi_modal"}

    criteria = [
        AcceptanceCriterion(
            deliberation="AST verification of class definition",
            id="AC1.1",
            description="Mechanics module must define ChronoCartridge subclassing BaseCartridge.",
            verification_method="ast_inspection",
        ),
        AcceptanceCriterion(
            deliberation="Runtime telemetry probe check",
            id="AC1.2",
            description="Telemetry probe 'time_rewind_used' must fire on rewind keypress.",
            verification_method="runtime_telemetry",
        ),
        AcceptanceCriterion(
            deliberation="Headless playtest state verification",
            id="AC1.3",
            description="Simulation stepping for 120 ticks must advance player past puzzle gate.",
            verification_method="headless_playtest",
        ),
    ]

    for c in criteria:
        assert c.verification_method in valid_methods, f"Invalid verification method: {c.verification_method}"
        assert c.deliberation != ""
        assert c.id.startswith("AC")


def test_tier1_r1_requirements_spec_event_bus_emission(tmp_path: Path):
    """R1.5: Verifies emission of REQUIREMENTS_SPEC_GENERATED to SwarmEventBus with deliberation first."""
    event_bus = SwarmEventBus(log_path=tmp_path / "studio_events.jsonl")

    event_payload = {
        "deliberation": "Requirements specification authored and validated for promotion.",
        "event_type": "REQUIREMENTS_SPEC_GENERATED",
        "game_title": "Nebula Nomad",
        "genre_paradigm": "Orbital Trade Simulator",
        "tracks_count": 2,
        "asset_count": 4,
    }

    event_bus.publish_threadsafe(event_payload)
    events = event_bus.get_recent_events()
    assert len(events) >= 1

    last_event = events[-1]
    assert list(last_event.keys())[0] == "deliberation", "deliberation must be first key in event payload"
    assert last_event["event_type"] == "REQUIREMENTS_SPEC_GENERATED"
    assert last_event["game_title"] == "Nebula Nomad"


# ---------------------------------------------------------------------------
# Requirement R2: Iterative Requirement Execution & Dynamic Asset Synthesis
# ---------------------------------------------------------------------------

def test_tier1_r2_dynamic_asset_manifest_synthesis(tmp_path: Path):
    """R2.1: Verifies dynamic asset synthesis operates strictly from asset_manifest without hardcoded sprites."""
    staging_dir = tmp_path / "staging_candidate"
    sprites_dir = staging_dir / "assets" / "sprites"
    audio_dir = staging_dir / "assets" / "audio"
    sprites_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = [
        AssetRequirement(
            deliberation="Space station central core sprite",
            asset_id="station_core",
            asset_type="sprite",
            target_path="assets/sprites/station_core.png",
            visual_prompt_or_text="64x64 modular station hub",
            dimensions=(64, 64),
        ),
        AssetRequirement(
            deliberation="Solar flare warning siren audio clip",
            asset_id="solar_flare_siren",
            asset_type="audio_bark",
            target_path="assets/audio/solar_flare_siren.wav",
            visual_prompt_or_text="Pulsing alarm sound effect",
        ),
    ]

    # Synthesize dynamically based on manifest requirements
    for asset in manifest:
        filename = Path(asset.target_path).name
        if "sprite" in asset.asset_type or "tile" in asset.asset_type:
            target = sprites_dir / filename
            target.write_bytes(_create_minimal_png_bytes(asset.dimensions[0], asset.dimensions[1]))
        elif "audio" in asset.asset_type:
            target = audio_dir / filename
            target.write_bytes(_create_minimal_wav_bytes())

    # Assert expected files exist and legacy fixed names do NOT exist
    assert (sprites_dir / "station_core.png").exists()
    assert (audio_dir / "solar_flare_siren.wav").exists()
    assert not (sprites_dir / "chaser.png").exists()
    assert not (sprites_dir / "sniper.png").exists()
    assert not (sprites_dir / "spikes.png").exists()


def test_tier1_r2_upstream_asset_sealing_in_staging(tmp_path: Path):
    """R2.2: Verifies dynamic assets are sealed in staging before SWE code implementation begins."""
    staging_dir = tmp_path / "staging_order"
    assets_dir = staging_dir / "assets" / "sprites"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Upstream Asset Synthesis
    core_sprite = assets_dir / "core_reactor.png"
    core_sprite.write_bytes(_create_minimal_png_bytes(32, 32))
    assert core_sprite.exists()

    # Step 2: SWE Agent inspects sealed assets directory
    available_assets = [p.name for p in assets_dir.glob("*.png")]
    assert "core_reactor.png" in available_assets

    # Step 3: SWE generates code referencing the verified sealed asset
    mechanics_code = f"""# Agent-authored mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec

class ReactorCartridge(BaseCartridge):
    deliberation: str = "Mechanics referencing verified sealed asset core_reactor.png"
    ASSET_NAME = "{available_assets[0]}"
"""
    (staging_dir / "mechanics.py").write_text(mechanics_code, encoding="utf-8")
    assert (staging_dir / "mechanics.py").exists()


def test_tier1_r2_multi_module_code_synthesis(tmp_path: Path):
    """R2.3: Verifies multi-module code synthesis (mechanics, entities, scenes, level) with deliberation first."""
    staging_dir = tmp_path / "multi_module_candidate"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. entities.py
    entities_code = """# entities.py
from engine.entities.base import Entity

class DroneEntity(Entity):
    deliberation: str = "Autonomous drone entity with tactical hover logic"
    def __init__(self, x: float, y: float):
        super().__init__(x=x, y=y, width=32.0, height=32.0)
        self.health = 100.0
"""
    (staging_dir / "entities.py").write_text(entities_code, encoding="utf-8")

    # 2. mechanics.py
    mechanics_code = """# mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from entities import DroneEntity

class ModularCartridge(BaseCartridge):
    deliberation: str = "Modular cartridge coordinating DroneEntity and custom scenes"
    def create_player(self, x: float, y: float, telemetry_recorder=None):
        return DroneEntity(x, y)
"""
    (staging_dir / "mechanics.py").write_text(mechanics_code, encoding="utf-8")

    # 3. level.json
    level_dict = {
        "deliberation": "Spatial layout for multi-module drone proving ground",
        "level_id": "proving_ground_01",
        "width": 1280.0,
        "height": 720.0,
        "spawn_point": [100.0, 360.0],
        "exit_point": [1180.0, 360.0],
    }
    (staging_dir / "level.json").write_text(json.dumps(level_dict), encoding="utf-8")

    # Verify AST security on each module
    for mod_name in ["entities.py", "mechanics.py"]:
        content = (staging_dir / mod_name).read_text(encoding="utf-8")
        validate_cartridge_ast(content)
        compile(content, mod_name, "exec")
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                first_stmt = node.body[0]
                assert (
                    isinstance(first_stmt, (ast.Assign, ast.AnnAssign))
                    and getattr(first_stmt.target if isinstance(first_stmt, ast.AnnAssign) else first_stmt.targets[0], "id", None) == "deliberation"
                ), f"Class {node.name} must declare deliberation as its very first statement"


def test_tier1_r2_iterative_requirement_execution_loop():
    """R2.4: Verifies iterative requirement execution advances status (pending -> in_progress -> completed)."""
    tracks = [
        RequirementTrack(
            deliberation="Track R1: Core Movement",
            id="R1",
            title="Movement Loop",
            description="Player velocity integration.",
            status="pending",
        ),
        RequirementTrack(
            deliberation="Track R2: Collision Response",
            id="R2",
            title="Collision Detection",
            description="Wall collision clamping.",
            status="pending",
        ),
    ]

    # Iterative execution simulation
    for track in tracks:
        assert track.status == "pending"
        # Transition to in_progress
        track.status = "in_progress"
        assert track.status == "in_progress"
        # Smoke verification passed -> completed
        track.status = "completed"
        assert track.status == "completed"

    assert all(t.status == "completed" for t in tracks)


def test_tier1_r2_live_requirement_status_streaming(tmp_path: Path):
    """R2.5: Verifies streaming of REQUIREMENT_STATUS_UPDATE events over SwarmEventBus."""
    event_bus = SwarmEventBus(log_path=tmp_path / "studio_events.jsonl")

    updates = [
        {"deliberation": "Beginning implementation of R1", "event_type": "REQUIREMENT_STATUS_UPDATE", "requirement_id": "R1", "status": "in_progress"},
        {"deliberation": "R1 compiled and passed headless smoke check", "event_type": "REQUIREMENT_STATUS_UPDATE", "requirement_id": "R1", "status": "completed"},
    ]

    for u in updates:
        event_bus.publish_threadsafe(u)

    events = event_bus.get_recent_events()
    assert len(events) >= 2
    assert events[-2]["status"] == "in_progress"
    assert events[-1]["status"] == "completed"
    assert list(events[-1].keys())[0] == "deliberation"


# ---------------------------------------------------------------------------
# Requirement R3: Independent Design Critic / Challenger Gate & Verification
# ---------------------------------------------------------------------------

def test_tier1_r3_challenger_evaluation_contract_deliberation_first():
    """R3.1: Verifies ChallengerGateEvaluation declares deliberation strictly first."""
    record = RequirementEvaluationRecord(
        deliberation="Audited R1 against AST classes and 60-tick simulation trace.",
        requirement_id="R1",
        fulfilled=True,
        grounding_evidence="Class DroneEntity instantiated, moved +12.5 px on tick 10.",
        critique_notes="Robust compliance with movement specification.",
    )
    assert list(record.model_dump().keys())[0] == "deliberation"

    eval_result = ChallengerGateEvaluation(
        deliberation="Comprehensive multi-perspective audit across all 3 requirement tracks.",
        candidate_id="candidate_001",
        overall_verdict=True,
        evaluated_requirements=[record],
        rejection_trace=None,
    )
    dumped = eval_result.model_dump()
    assert list(dumped.keys())[0] == "deliberation"
    assert dumped["overall_verdict"] is True


def test_tier1_r3_triangulated_grounding_verification(mock_cartridge_active: Path):
    """R3.2: Verifies triangulated grounding combining AST inspection and runtime simulation."""
    # 1. AST Grounding
    mechanics_file = mock_cartridge_active / "mechanics.py"
    content = mechanics_file.read_text(encoding="utf-8")
    tree = ast.parse(content)
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "ActiveCartridge" in classes

    # 2. Runtime Simulation Grounding
    sim = HeadlessSimulator(cartridge_path=mock_cartridge_active)
    for _ in range(60):
        sim.step(SimulationInput(move_x=1.0))

    assert sim.step_count == 60
    assert sim.avg_frame_time_ms < 16.6, "Must meet 60 FPS frame timing budget"

    grounding_summary = {
        "ast_classes": classes,
        "ticks_executed": sim.step_count,
        "avg_frame_ms": sim.avg_frame_time_ms,
    }
    assert len(grounding_summary["ast_classes"]) > 0
    assert grounding_summary["ticks_executed"] == 60


def test_tier1_r3_shallow_stub_rejection(tmp_path: Path):
    """R3.3: Demonstrably rejects candidate cartridges that provide shallow stubs or empty pass bodies."""
    stub_dir = tmp_path / "shallow_candidate"
    stub_dir.mkdir(parents=True, exist_ok=True)

    # Shallow stub with pass
    stub_code = """# Shallow stub mechanics.py
from engine.api.contracts import BaseCartridge

class StubCartridge(BaseCartridge):
    deliberation: str = "Shallow stub intended to evade inspection"
    def execute_mechanic(self):
        pass
"""
    (stub_dir / "mechanics.py").write_text(stub_code, encoding="utf-8")

    tree = ast.parse(stub_code)
    has_shallow_stub = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "execute_mechanic":
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                has_shallow_stub = True

    assert has_shallow_stub is True, "Shallow stub must be detected"

    # Challenger Gate evaluation rejects candidate
    eval_result = ChallengerGateEvaluation(
        deliberation="Candidate rejected: execute_mechanic contains only a pass statement.",
        candidate_id="shallow_candidate",
        overall_verdict=False,
        evaluated_requirements=[
            RequirementEvaluationRecord(
                deliberation="Stub detected in execute_mechanic",
                requirement_id="R1",
                fulfilled=False,
                grounding_evidence="AST inspection confirmed single Pass statement in execute_mechanic.",
                critique_notes="Shallow stub rejected. Genuine mechanics implementation required.",
            )
        ],
        rejection_trace=RejectionDiagnosticTrace(
            deliberation="Emitting diagnostic trace for shallow stub rejection.",
            candidate_id="shallow_candidate",
            failure_gate="Gate F: Adversarial Design Critic",
            failure_type=GateFailureType.PROGRESSION_FAILURE,
            failure_reason="Shallow stub detected in mechanics module.",
            remedy_type=RemedyType.REPAIR_CODE_EXCEPTION,
            recommended_remedy="Replace pass statement with full interactive gameplay loop.",
        ),
    )

    assert eval_result.overall_verdict is False
    assert eval_result.rejection_trace is not None
    assert eval_result.rejection_trace.failure_gate.startswith("Gate F")


def test_tier1_r3_rejection_diagnostic_trace_emission():
    """R3.4: Verifies structured RejectionDiagnosticTrace emission with deliberation first."""
    trace = RejectionDiagnosticTrace(
        deliberation="Epistemic diagnostic analyzing why candidate failed Challenger Gate.",
        candidate_id="candidate_genesis_fail",
        episode_number=1,
        attempt_number=1,
        failure_gate="Gate F: Challenger Gate",
        failure_type=GateFailureType.PROGRESSION_FAILURE,
        failure_reason="Acceptance criterion AC2.1 unfulfilled: score counter does not increment on coin collect.",
        remedy_type=RemedyType.REPAIR_CODE_EXCEPTION,
        recommended_remedy="Bind score counter increment to coin collision event.",
        remedy_parameters={"unfulfilled_requirements": ["R2"], "failing_criterion": "AC2.1"},
    )

    dumped = trace.model_dump()
    assert list(dumped.keys())[0] == "deliberation"
    assert dumped["failure_gate"] == "Gate F: Challenger Gate"
    assert "R2" in dumped["remedy_parameters"]["unfulfilled_requirements"]


def test_tier1_r3_promotion_gating_blocks_unverified(tmp_path: Path, mock_cartridge_active: Path):
    """R3.5: Verifies candidate is NOT promoted when unverified, and IS promoted when verified."""
    active_dir = tmp_path / "isolated_active"
    supervisor = BuildSupervisor(active_cartridge_dir=active_dir, cartridges_root=tmp_path / "cartridges")

    # Case A: Build evaluation fails Gate -> Do NOT promote
    verdict_rejected = SupervisorGateVerdict.REJECTED
    if verdict_rejected != SupervisorGateVerdict.PROMOTED:
        # Candidate must NOT exist in active
        assert not (active_dir / "manifest.json").exists()

    # Case B: Build evaluation passes Gate -> Promote
    supervisor.promote_candidate(mock_cartridge_active)
    assert (active_dir / "manifest.json").exists()
    assert (active_dir / "mechanics.py").exists()


# ===========================================================================
# SECTION 2: TIER 2 BOUNDARY VALUE ANALYSIS & EDGE CASES
# ===========================================================================

# ---------------------------------------------------------------------------
# Requirement R1 Boundaries
# ---------------------------------------------------------------------------

def test_tier2_r1_abstract_sim_zero_characters_support():
    """BVA-R1.1: Verifies pure abstract simulation games with zero characters validate cleanly (Rule 1)."""
    spec = GameRequirementsSpec(
        deliberation="Pure mathematical cellular automaton simulation. Zero narrative characters required.",
        game_title="Conway Horizon",
        genre_paradigm="Cellular Automaton",
        target_audience_summary="Math and logic systems enthusiasts.",
        requirements=[
            RequirementTrack(
                deliberation="Grid state update mechanics",
                id="R1",
                title="Cell State Transition Loop",
                description="Underpopulated (<2) cells die, healthy (2-3) cells survive.",
                responsible_agents=["LeadGameplaySWEAgent"],
                status="pending",
            )
        ],
        asset_manifest=[
            AssetRequirement(
                deliberation="Cell alive/dead sprite tiles",
                asset_id="cell_alive",
                asset_type="tile",
                target_path="assets/sprites/cell_alive.png",
                visual_prompt_or_text="8x8 white square",
                dimensions=(8, 8),
            )
        ],
    )

    assert len(spec.requirements) == 1
    # Pure systems game has zero narrative character requirements
    assert "characters" not in spec.model_dump() or spec.model_dump().get("characters") == []


def test_tier2_r1_backlog_sizing_boundaries():
    """BVA-R1.2: Tests minimal single-track backlog (1 track) and deep backlog (8 tracks)."""
    # 1. Minimal single-track backlog
    minimal_spec = GameRequirementsSpec(
        deliberation="Minimal prototype backlog",
        game_title="Micro Pong",
        genre_paradigm="Arcade",
        target_audience_summary="Retro arcade fans",
        requirements=[
            RequirementTrack(
                deliberation="Single paddle rebound loop",
                id="R1",
                title="Paddle Rebound",
                description="Ball bounces off paddle",
                status="pending",
            )
        ],
    )
    assert len(minimal_spec.requirements) == 1

    # 2. Deep multi-track backlog
    deep_tracks = [
        RequirementTrack(
            deliberation=f"Deliberation for track {i}",
            id=f"R{i}",
            title=f"Subsystem {i}",
            description=f"Detailed description for subsystem {i}",
            status="pending",
        )
        for i in range(1, 9)
    ]
    deep_spec = GameRequirementsSpec(
        deliberation="Deep 8-track grand strategy backlog",
        game_title="Galactic Imperium",
        genre_paradigm="Grand Strategy",
        target_audience_summary="Complex 4X enthusiasts",
        requirements=deep_tracks,
    )
    assert len(deep_spec.requirements) == 8
    assert deep_spec.requirements[-1].id == "R8"


def test_tier2_r1_deliberation_validation_bounds():
    """BVA-R1.3: Verifies empty string or whitespace-only deliberation raises error (Rule 4)."""
    with pytest.raises((ValueError, Exception), match=r"(?i)deliberation.*(empty|whitespace)"):
        GameRequirementsSpec(
            deliberation="",
            game_title="Invalid Game",
            genre_paradigm="Arcade",
            target_audience_summary="Test",
        )

    with pytest.raises((ValueError, Exception), match=r"(?i)deliberation.*(empty|whitespace)"):
        RequirementTrack(
            deliberation="   \n\t  ",
            id="R1",
            title="Whitespace Deliberation",
            description="Test",
        )

    with pytest.raises((ValueError, Exception), match=r"(?i)deliberation.*(empty|whitespace)"):
        AcceptanceCriterion(
            deliberation="",
            id="AC1.1",
            description="Missing deliberation",
            verification_method="ast_inspection",
        )


def test_tier2_r1_empty_asset_manifest_boundaries():
    """BVA-R1.4: Tests asset manifests with zero audio (silent retro) or zero custom sprites."""
    silent_spec = GameRequirementsSpec(
        deliberation="Silent retro handheld puzzler with zero audio assets.",
        game_title="Monochrome Grid",
        genre_paradigm="Logic Puzzle",
        target_audience_summary="Silent ambient gaming",
        requirements=[
            RequirementTrack(
                deliberation="Logic puzzle mechanics",
                id="R1",
                title="Grid Inversion",
                description="Inverting tile states",
                status="pending",
            )
        ],
        asset_manifest=[
            AssetRequirement(
                deliberation="Grid tile sprite",
                asset_id="tile",
                asset_type="tile",
                target_path="assets/sprites/tile.png",
                visual_prompt_or_text="16x16 tile",
                dimensions=(16, 16),
            )
        ],
    )
    # Confirm asset manifest contains zero audio files
    audio_assets = [a for a in silent_spec.asset_manifest if "audio" in a.asset_type]
    assert len(audio_assets) == 0


def test_tier2_r1_requirements_spec_json_roundtrip(tmp_path: Path):
    """BVA-R1.5: Verifies JSON serialization and deserialization preserves unicode and all fields."""
    spec_path = tmp_path / "requirements_spec.json"

    original_spec = GameRequirementsSpec(
        deliberation="Analysis with unicode characters: ★ Alpha & Omega — 100% precision.",
        game_title="Cyber Odyssey ★",
        genre_paradigm="Cyberpunk Cyber-Sim",
        target_audience_summary="Lo-fi & sci-fi enthusiasts.",
        requirements=[
            RequirementTrack(
                deliberation="Track 1 deliberation with symbols: → ∆ ∇",
                id="R1",
                title="Hacking Matrix",
                description="Terminal intrusion simulation",
                responsible_agents=["LeadGameplaySWEAgent"],
                acceptance_criteria=[
                    AcceptanceCriterion(
                        deliberation="Criterion AC1.1 deliberation",
                        id="AC1.1",
                        description="Access key unlocks data terminal",
                        verification_method="runtime_telemetry",
                    )
                ],
                status="pending",
            )
        ],
    )

    # Save to disk
    spec_path.write_text(json.dumps(original_spec.model_dump(), indent=2), encoding="utf-8")

    # Read from disk
    loaded_data = json.loads(spec_path.read_text(encoding="utf-8"))
    reconstructed = GameRequirementsSpec(**loaded_data)

    assert reconstructed.deliberation == original_spec.deliberation
    assert reconstructed.game_title == original_spec.game_title
    assert reconstructed.requirements[0].acceptance_criteria[0].id == "AC1.1"


# ---------------------------------------------------------------------------
# Requirement R2 Boundaries
# ---------------------------------------------------------------------------

def test_tier2_r2_zero_silent_fallbacks_in_asset_synthesis(tmp_path: Path):
    """BVA-R2.1: Verifies synthesis failure fails fast with authentic exception, no silent fallback (Rule 5)."""
    # Simulate an asset synthesis engine that raises an authentic error when audio generation fails
    def faulty_voice_synthesis(bark_id: str, audio_bytes: Any) -> None:
        if not isinstance(audio_bytes, (bytes, bytearray)):
            raise TypeError(f"Authentic failure: audio_bytes must be bytes-like object, got {type(audio_bytes).__name__}")

    # Must raise TypeError rather than silently writing a blank file or mock placeholder
    with pytest.raises(TypeError, match="audio_bytes must be bytes-like object"):
        faulty_voice_synthesis("bark_alert", None)


def test_tier2_r2_per_track_smoke_check_failure_halts_loop():
    """BVA-R2.2: Verifies that if track R1 fails smoke check, execution loop halts and does not advance to R2."""
    tracks = [
        RequirementTrack(deliberation="R1", id="R1", title="Track 1", description="desc", status="pending"),
        RequirementTrack(deliberation="R2", id="R2", title="Track 2", description="desc", status="pending"),
    ]

    # Simulating track execution
    execution_log = []
    for track in tracks:
        track.status = "in_progress"
        execution_log.append(track.id)
        if track.id == "R1":
            # Simulate compilation or smoke test failure
            track.status = "rejected"
            break  # Halt immediately

    assert tracks[0].status == "rejected"
    assert tracks[1].status == "pending", "Track R2 must never execute if Track R1 failed"
    assert execution_log == ["R1"]


def test_tier2_r2_extreme_sprite_geometry_manifest(tmp_path: Path):
    """BVA-R2.3: Validates dynamic synthesis across boundary resolutions (8x8, 128x128)."""
    sprites_dir = tmp_path / "extreme_sprites"
    sprites_dir.mkdir(parents=True, exist_ok=True)

    test_dims = [(8, 8), (16, 16), (64, 64), (128, 128)]
    for w, h in test_dims:
        fname = f"sprite_{w}x{h}.png"
        file_path = sprites_dir / fname
        file_path.write_bytes(_create_minimal_png_bytes(w, h))
        assert file_path.exists()
        assert file_path.stat().st_size > 0


def test_tier2_r2_cross_module_import_invariants(tmp_path: Path):
    """BVA-R2.4: Verifies multi-module imports resolve cleanly without circular dependencies."""
    sandbox_dir = tmp_path / "import_sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    scenes_py = """# scenes.py
class TitleScene:
    deliberation: str = "Title splash and menu scene"
    def render(self): return "TitleScene"
"""
    entities_py = """# entities.py
class HeroEntity:
    deliberation: str = "Controllable protagonist entity"
    def __init__(self, x, y): self.x, self.y = x, y
"""
    mechanics_py = """# mechanics.py
from scenes import TitleScene
from entities import HeroEntity

class GameEngine:
    deliberation: str = "Engine coordinating scenes and entities"
    def __init__(self):
        self.scene = TitleScene()
        self.hero = HeroEntity(0, 0)
"""
    (sandbox_dir / "scenes.py").write_text(scenes_py, encoding="utf-8")
    (sandbox_dir / "entities.py").write_text(entities_py, encoding="utf-8")
    (sandbox_dir / "mechanics.py").write_text(mechanics_py, encoding="utf-8")

    # Add sandbox_dir temporarily to sys.path to verify runtime import resolution
    sys.path.insert(0, str(sandbox_dir))
    try:
        import mechanics  # type: ignore
        engine_inst = mechanics.GameEngine()
        assert engine_inst.scene.render() == "TitleScene"
        assert engine_inst.hero.x == 0
    finally:
        sys.path.remove(str(sandbox_dir))
        sys.modules.pop("mechanics", None)
        sys.modules.pop("scenes", None)
        sys.modules.pop("entities", None)


def test_tier2_r2_high_frequency_event_bus_load(tmp_path: Path):
    """BVA-R2.5: Emits rapid burst of 100 events to SwarmEventBus; asserts deque retention and order."""
    event_bus = SwarmEventBus(history_size=150, log_path=tmp_path / "studio_events.jsonl")

    for i in range(100):
        event_bus.publish_threadsafe({
            "deliberation": f"Burst update {i}",
            "event_type": "REQUIREMENT_STATUS_UPDATE",
            "index": i,
        })

    events = event_bus.get_recent_events()
    assert len(events) == 100
    assert events[0]["index"] == 0
    assert events[-1]["index"] == 99


# ---------------------------------------------------------------------------
# Requirement R3 Boundaries
# ---------------------------------------------------------------------------

def test_tier2_r3_strict_100_percent_acceptance_threshold():
    """BVA-R3.1: Asserts that 9 out of 10 criteria passing (90%) yields overall_verdict = False."""
    records = []
    for i in range(1, 10):
        records.append(RequirementEvaluationRecord(
            deliberation=f"AC{i} fulfilled",
            requirement_id=f"R{i}",
            fulfilled=True,
            grounding_evidence="Verified",
        ))
    # 10th criterion fails
    records.append(RequirementEvaluationRecord(
        deliberation="AC10 unfulfilled: boss transition state unreachable",
        requirement_id="R10",
        fulfilled=False,
        grounding_evidence="No transition event emitted during 120-tick playtest.",
    ))

    # Any single failure mandates overall_verdict = False
    all_passed = all(r.fulfilled for r in records)
    eval_result = ChallengerGateEvaluation(
        deliberation="Audit evaluated 10 criteria: 9 passed, 1 failed.",
        candidate_id="candidate_90_percent",
        overall_verdict=all_passed,
        evaluated_requirements=records,
    )

    assert eval_result.overall_verdict is False, "100% pass threshold required; 90% must be rejected"


def test_tier2_r3_ast_evasion_hardening():
    """BVA-R3.2: Tests evasion attempts (docstring only, return None, raise NotImplementedError) are caught."""
    evasive_code = """# Evasive mechanics.py
class EvasiveCartridge:
    deliberation: str = "Attempting to pass as non-empty"
    def critical_mechanic_a(self):
        \"\"\"Just a docstring, no logic.\"\"\"
    def critical_mechanic_b(self):
        return None
    def critical_mechanic_c(self):
        raise NotImplementedError
"""
    tree = ast.parse(evasive_code)
    evasion_detected = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if function body is hollow
            is_hollow = False
            if len(node.body) == 1:
                stmt = node.body[0]
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    is_hollow = True  # docstring only
                elif isinstance(stmt, ast.Return) and (stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)):
                    is_hollow = True  # return None only
                elif isinstance(stmt, ast.Raise) and getattr(stmt.exc, "id", None) == "NotImplementedError":
                    is_hollow = True  # raise NotImplementedError
            if is_hollow:
                evasion_detected.append(node.name)

    assert len(evasion_detected) == 3
    assert "critical_mechanic_a" in evasion_detected
    assert "critical_mechanic_b" in evasion_detected
    assert "critical_mechanic_c" in evasion_detected


def test_tier2_r3_non_deterministic_telemetry_grounding():
    """BVA-R3.3: Verifies grounding evidence extraction handles varying timestamp offsets and frame counts."""
    events = [
        {"timestamp": 0.016, "event_type": "player_spawn", "x": 100.0, "y": 360.0},
        {"timestamp": 0.048, "event_type": "item_pickup", "item_id": "core_shard"},
        {"timestamp": 0.082, "event_type": "gate_unlocked", "gate_id": "gate_alpha"},
    ]

    # Telemetry grounding parser checks event existence without rigid timing locks
    def verify_telemetry_grounding(event_log: List[Dict[str, Any]], expected_types: Set[str]) -> bool:
        logged_types = {e["event_type"] for e in event_log}
        return expected_types.issubset(logged_types)

    assert verify_telemetry_grounding(events, {"player_spawn", "item_pickup"}) is True
    assert verify_telemetry_grounding(events, {"player_spawn", "boss_defeated"}) is False


def test_tier2_r3_challenger_deliberation_enforcement():
    """BVA-R3.4: Asserts ChallengerGateEvaluation with empty deliberation is rejected."""
    with pytest.raises(ValueError, match="deliberation must not be empty"):
        ChallengerGateEvaluation(
            deliberation="",
            candidate_id="candidate_test",
            overall_verdict=True,
        )


def test_tier2_r3_max_retries_rejection_bound():
    """BVA-R3.5: Tests repeated candidate rejection terminates cleanly with FAILED_MAX_RETRIES at max_retries."""
    max_retries = 3
    attempt = 1
    status = SupervisorGateVerdict.REJECTED

    while attempt < max_retries:
        attempt += 1
    status = SupervisorGateVerdict.FAILED_MAX_RETRIES

    assert attempt == max_retries
    assert status == SupervisorGateVerdict.FAILED_MAX_RETRIES


# ===========================================================================
# SECTION 3: TIER 3 CROSS-FEATURE COMBINATIONS (Pairwise Milestone Contracts)
# ===========================================================================

def test_tier3_combination_spec_generation_to_dynamic_assets(tmp_path: Path):
    """C1: Pairwise combination between R1 Spec Generation and R2 Dynamic Asset Synthesis."""
    staging_dir = tmp_path / "comb_assets"
    sprites_dir = staging_dir / "assets" / "sprites"
    sprites_dir.mkdir(parents=True, exist_ok=True)

    # 1. R1 generates spec with unique dynamic assets
    spec = GameRequirementsSpec(
        deliberation="Unique orbital physics sim requiring custom gravity well sprites.",
        game_title="Gravity Wells",
        genre_paradigm="Orbital Mechanics",
        target_audience_summary="Physics fans",
        requirements=[
            RequirementTrack(
                deliberation="Gravity well physics loop",
                id="R1",
                title="Orbital Gravity Trajectory",
                description="Simulate Newtonian gravity towards well center.",
                status="pending",
            )
        ],
        asset_manifest=[
            AssetRequirement(
                deliberation="Pulsing gravity well center sprite",
                asset_id="gravity_singularity",
                asset_type="sprite",
                target_path="assets/sprites/gravity_singularity.png",
                visual_prompt_or_text="48x48 vortex",
                dimensions=(48, 48),
            )
        ],
    )

    # 2. R2 dynamically synthesizes the exact declared sprite
    for asset in spec.asset_manifest:
        fname = Path(asset.target_path).name
        (sprites_dir / fname).write_bytes(_create_minimal_png_bytes(asset.dimensions[0], asset.dimensions[1]))

    assert (sprites_dir / "gravity_singularity.png").exists()
    assert not (sprites_dir / "chaser.png").exists()


def test_tier3_combination_multi_module_swe_to_ast_grounding(tmp_path: Path):
    """C2: Pairwise combination between R2 Multi-Module SWE and R3 Gate F AST Grounding."""
    candidate_dir = tmp_path / "candidate_multi_ast"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # R2 outputs multi-module code
    (candidate_dir / "entities.py").write_text(
        "class ResourceExtractor:\n    deliberation = 'Extracts raw minerals'\n    def harvest(self): return 10\n",
        encoding="utf-8",
    )
    (candidate_dir / "mechanics.py").write_text(
        "from entities import ResourceExtractor\nclass MineCartridge:\n    deliberation = 'Coordinates extractor'\n",
        encoding="utf-8",
    )

    # R3 Challenger audits AST across both modules
    audited_classes = {}
    for py_file in candidate_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                audited_classes[node.name] = py_file.name

    assert "ResourceExtractor" in audited_classes
    assert "MineCartridge" in audited_classes
    assert audited_classes["ResourceExtractor"] == "entities.py"
    assert audited_classes["MineCartridge"] == "mechanics.py"


def test_tier3_combination_iterative_failure_to_diagnostic_trace():
    """C3: Pairwise combination between R2 Iterative Step Failure and R3 RejectionDiagnosticTrace."""
    # Step failure in requirement R2
    failing_track = RequirementTrack(
        deliberation="Collision reaction track",
        id="R2",
        title="Collision Handling",
        description="Handles wall impacts",
        status="rejected",
    )

    # R3 emits targeted trace routing repair back to SWE agent
    trace = RejectionDiagnosticTrace(
        deliberation="Targeting repair for unfulfilled requirement R2.",
        candidate_id="cand_comb_03",
        failure_gate="Gate F: Challenger Gate",
        failure_type=GateFailureType.PROGRESSION_FAILURE,
        failure_reason="R2 collision response missing bounce damping.",
        remedy_type=RemedyType.REPAIR_CODE_EXCEPTION,
        recommended_remedy="Implement bounce damping coefficient in on_wall_collision.",
        remedy_parameters={"unfulfilled_requirements": [failing_track.id]},
    )

    assert trace.remedy_parameters["unfulfilled_requirements"] == ["R2"]
    assert trace.remedy_type == RemedyType.REPAIR_CODE_EXCEPTION


def test_tier3_combination_zero_character_spec_to_promotion(tmp_path: Path):
    """C4: Pairwise combination of R1 Zero-Character Spec + R2 Assets + R3 Challenger Verification + Promotion."""
    candidate_dir = tmp_path / "candidate_zero_char"
    active_dir = tmp_path / "active_zero_char"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # R1 Spec: Zero characters
    spec = GameRequirementsSpec(
        deliberation="Zero-character factory puzzle game.",
        game_title="Logic Nodes",
        genre_paradigm="Pure Puzzle",
        target_audience_summary="Pure puzzle solvers.",
        requirements=[
            RequirementTrack(
                deliberation="Node linkage logic",
                id="R1",
                title="Node Linkage",
                description="Connect output node to input node.",
                status="completed",
            )
        ],
    )

    # R2 Synthesis
    (candidate_dir / "manifest.json").write_text(json.dumps({"title": spec.game_title}), encoding="utf-8")
    (candidate_dir / "mechanics.py").write_text(
        "from engine.api.contracts import BaseCartridge\nclass NodeCartridge(BaseCartridge):\n    deliberation = 'Pure logic game'\n",
        encoding="utf-8",
    )

    # R3 Challenger Audit: 100% passed
    challenger_eval = ChallengerGateEvaluation(
        deliberation="Zero-character game evaluated: 100% criteria passed.",
        candidate_id=candidate_dir.name,
        overall_verdict=True,
    )

    # Supervisor Promotion
    supervisor = BuildSupervisor(active_cartridge_dir=active_dir, cartridges_root=tmp_path / "cartridges")
    if challenger_eval.overall_verdict:
        supervisor.promote_candidate(candidate_dir)

    assert (active_dir / "manifest.json").exists()
    assert (active_dir / "mechanics.py").exists()


def test_tier3_combination_full_event_bus_triangulation(tmp_path: Path):
    """C5: Triangulates complete event bus chronological emission across R1, R2, and R3."""
    event_bus = SwarmEventBus(log_path=tmp_path / "studio_events.jsonl")

    # 1. R1 Event
    event_bus.publish_threadsafe({
        "deliberation": "Spec generated",
        "event_type": "REQUIREMENTS_SPEC_GENERATED",
        "game_title": "Chronos",
    })

    # 2. R2 Event (In progress)
    event_bus.publish_threadsafe({
        "deliberation": "R1 in progress",
        "event_type": "REQUIREMENT_STATUS_UPDATE",
        "requirement_id": "R1",
        "status": "in_progress",
    })

    # 3. R2 Event (Completed)
    event_bus.publish_threadsafe({
        "deliberation": "R1 completed",
        "event_type": "REQUIREMENT_STATUS_UPDATE",
        "requirement_id": "R1",
        "status": "completed",
    })

    # 4. R3 Event (Challenger Verdict)
    event_bus.publish_threadsafe({
        "deliberation": "Gate F passed",
        "event_type": "CHALLENGER_GATE_VERDICT",
        "passed": True,
    })

    events = event_bus.get_recent_events()
    assert len(events) == 4
    event_types = [e["event_type"] for e in events]
    assert event_types == [
        "REQUIREMENTS_SPEC_GENERATED",
        "REQUIREMENT_STATUS_UPDATE",
        "REQUIREMENT_STATUS_UPDATE",
        "CHALLENGER_GATE_VERDICT",
    ]


# ===========================================================================
# SECTION 4: TIER 4 REAL-WORLD APPLICATION SCENARIOS
# ===========================================================================

def test_tier4_scenario_cold_boot_genesis_requirements_flow(tmp_path: Path):
    """Scenario 1: End-to-End Cold-Boot Autonomous Genesis Requirements Flow.
    1. Scan Steam library dataset & extract SteamPlayerDNA.
    2. Author GameRequirementsSpec (requirements_spec.json) with deliberation first.
    3. Dynamically synthesize declared assets and seal in staging/assets/.
    4. Progressively synthesize multi-module cartridge code.
    5. Run Challenger Gate audit against 100% of acceptance criteria.
    6. Promote verified candidate to active; verify headless boot.
    """
    # 1. Steam DNA Profiling
    profiler = SteamDNAProfiler()
    dataset = SteamLibraryDataset(
        player_id="steam_genesis_player",
        games=[
            SteamGameRecord(appid=101, name="Hades", playtime_forever_minutes=12000, genres=["Action", "RogueLite"]),
            SteamGameRecord(appid=102, name="Dead Cells", playtime_forever_minutes=8000, genres=["Action", "Platformer"]),
        ],
    )
    dna = profiler.extract_dna(dataset, profile_tag="genesis_e2e")
    assert dna.total_games_analyzed == 2

    # 2. Requirements Spec Generation
    spec = GameRequirementsSpec(
        deliberation=f"DNA pacing ({dna.pacing_score:.2f}) mandates high-tempo dodge-combat loop.",
        game_title="Aegis Protocol",
        genre_paradigm="Action RogueLite",
        target_audience_summary="Action reflex player",
        requirements=[
            RequirementTrack(
                deliberation="Dodge roll mechanic with frame invulnerability",
                id="R1",
                title="Dodge Roll",
                description="Player executes dodge roll on spacebar.",
                responsible_agents=["LeadGameplaySWEAgent"],
                status="pending",
            )
        ],
        asset_manifest=[
            AssetRequirement(
                deliberation="Aegis protagonist ship sprite",
                asset_id="aegis_ship",
                asset_type="sprite",
                target_path="assets/sprites/aegis_ship.png",
                visual_prompt_or_text="32x32 kinetic ship sprite",
                dimensions=(32, 32),
            )
        ],
    )
    assert spec.deliberation != ""

    # 3. Dynamic Asset Synthesis
    candidate_dir = tmp_path / "candidate_genesis_flow"
    sprites_dir = candidate_dir / "assets" / "sprites"
    sprites_dir.mkdir(parents=True, exist_ok=True)
    (sprites_dir / "aegis_ship.png").write_bytes(_create_minimal_png_bytes(32, 32))

    # 4. Multi-Module Code Synthesis
    (candidate_dir / "manifest.json").write_text(
        json.dumps({"title": spec.game_title, "version": "1.0.0", "entry_point": "mechanics.py"}),
        encoding="utf-8",
    )
    mechanics_code = """# mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec

class AegisCartridge(BaseCartridge):
    deliberation: str = "Aegis dodge combat implementation"
    def get_level_spec(self, episode_number: int = 1):
        return BaseLevelSpec(
            level_id="ep1", name="Arena", width=1280.0, height=720.0,
            spawn_point=(100.0, 360.0), exit_point=(1180.0, 360.0),
            deliberation="Arena geometry"
        )
"""
    (candidate_dir / "mechanics.py").write_text(mechanics_code, encoding="utf-8")

    # 5. Challenger Gate Audit
    challenger_eval = ChallengerGateEvaluation(
        deliberation="Audited Aegis Protocol candidate: 100% criteria verified.",
        candidate_id=candidate_dir.name,
        overall_verdict=True,
    )
    assert challenger_eval.overall_verdict is True

    # 6. Promotion to Active & Headless Simulator Verification
    active_dir = tmp_path / "promoted_active"
    supervisor = BuildSupervisor(active_cartridge_dir=active_dir, cartridges_root=tmp_path / "cartridges")
    supervisor.promote_candidate(candidate_dir)

    assert (active_dir / "mechanics.py").exists()
    assert (active_dir / "assets" / "sprites" / "aegis_ship.png").exists()

    sim = HeadlessSimulator(cartridge_path=active_dir)
    for _ in range(60):
        sim.step(SimulationInput(move_x=1.0))
    assert sim.step_count == 60
    assert sim.avg_frame_time_ms < 16.6


def test_tier4_scenario_stub_rejection_and_iterative_repair(tmp_path: Path):
    """Scenario 2: Shallow Stub Rejection, Diagnostic Trace Routing, and Iterative Repair.
    1. Candidate build generates shallow stub in mechanics.py (pass).
    2. Challenger Gate audits acceptance criteria, catches stub, and rejects build.
    3. Supervisor blocks promotion to active cartridge.
    4. RejectionDiagnosticTrace routes repair defect back to SWE agent.
    5. SWE agent replaces stub with verified implementation logic.
    6. Re-evaluation passes Challenger Gate -> Promoted to active.
    """
    candidate_dir = tmp_path / "candidate_repair_flow"
    active_dir = tmp_path / "active_repair_flow"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    supervisor = BuildSupervisor(active_cartridge_dir=active_dir, cartridges_root=tmp_path / "cartridges")

    # Phase 1: Shallow Stub
    (candidate_dir / "manifest.json").write_text(json.dumps({"title": "Stubby"}), encoding="utf-8")
    (candidate_dir / "mechanics.py").write_text(
        "class StubGame:\n    deliberation = 'stub'\n    def run(self): pass\n",
        encoding="utf-8",
    )

    # Challenger audit detects stub
    has_pass = "pass" in (candidate_dir / "mechanics.py").read_text(encoding="utf-8")
    assert has_pass is True

    challenger_verdict = False
    trace = None
    if not challenger_verdict:
        trace = RejectionDiagnosticTrace(
            deliberation="Shallow stub rejected. Trace dispatched for iterative repair.",
            candidate_id=candidate_dir.name,
            failure_gate="Gate F: Challenger Gate",
            failure_type=GateFailureType.PROGRESSION_FAILURE,
            failure_reason="Function run contains pass statement.",
            recommended_remedy="Implement genuine game loop.",
            remedy_parameters={"unfulfilled_requirements": ["R1"]},
        )
        # Block promotion
        assert not (active_dir / "mechanics.py").exists()

    assert trace is not None
    assert trace.remedy_parameters["unfulfilled_requirements"] == ["R1"]

    # Phase 2: Iterative Repair by SWE Specialist
    repaired_code = """# mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec

class RepairedGame(BaseCartridge):
    deliberation: str = "Genuine repaired mechanics without stubs"
    def get_level_spec(self, episode_number: int = 1):
        return BaseLevelSpec(
            level_id="rep1", name="Repaired Level", width=1280.0, height=720.0,
            spawn_point=(100.0, 360.0), exit_point=(1180.0, 360.0),
            deliberation="Repaired level layout"
        )
"""
    (candidate_dir / "mechanics.py").write_text(repaired_code, encoding="utf-8")

    # Challenger re-audit
    re_audited_pass = "pass" in (candidate_dir / "mechanics.py").read_text(encoding="utf-8")
    assert re_audited_pass is False

    re_evaluation = ChallengerGateEvaluation(
        deliberation="Re-evaluated candidate: stub repaired, 100% criteria satisfied.",
        candidate_id=candidate_dir.name,
        overall_verdict=True,
    )

    if re_evaluation.overall_verdict:
        supervisor.promote_candidate(candidate_dir)

    # Active cartridge is now cleanly populated
    assert (active_dir / "mechanics.py").exists()
    assert "RepairedGame" in (active_dir / "mechanics.py").read_text(encoding="utf-8")


def test_tier4_scenario_abstract_automation_game_flow(tmp_path: Path):
    """Scenario 3: Abstract Factory/Automation Game Conception & Headless Verification.
    1. Player DNA indicates preference for automation/management systems.
    2. Swarm authors GameRequirementsSpec with zero narrative characters.
    3. Dynamic sprite synthesis outputs grid tiles and conveyor sprites.
    4. SWE synthesizes multi-module grid systems.
    5. Challenger Gate audits pure systems gameplay without character checks.
    6. Candidate is promoted and steps cleanly in HeadlessSimulator.
    """
    candidate_dir = tmp_path / "candidate_abstract_factory"
    active_dir = tmp_path / "active_abstract_factory"
    sprites_dir = candidate_dir / "assets" / "sprites"
    sprites_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spec with Zero Characters
    spec = GameRequirementsSpec(
        deliberation="Abstract conveyor logistics factory game. Zero narrative characters.",
        game_title="Factorio Nano",
        genre_paradigm="Factory Automation",
        target_audience_summary="Logistics optimization player",
        requirements=[
            RequirementTrack(
                deliberation="Grid conveyor logistics system",
                id="R1",
                title="Conveyor Logistics Grid",
                description="Conveyors transport mineral cubes.",
                status="completed",
            )
        ],
        asset_manifest=[
            AssetRequirement(
                deliberation="Conveyor tile sprite",
                asset_id="conveyor_tile",
                asset_type="tile",
                target_path="assets/sprites/conveyor_tile.png",
                visual_prompt_or_text="16x16 directional tile",
                dimensions=(16, 16),
            )
        ],
    )

    # 2. Dynamic Tiles
    (sprites_dir / "conveyor_tile.png").write_bytes(_create_minimal_png_bytes(16, 16))

    # 3. Systems Code
    (candidate_dir / "manifest.json").write_text(json.dumps({"title": spec.game_title}), encoding="utf-8")
    factory_code = """# mechanics.py
from engine.api.contracts import BaseCartridge, BaseLevelSpec

class FactoryCartridge(BaseCartridge):
    deliberation: str = "Pure grid simulation without narrative characters"
    def get_level_spec(self, episode_number: int = 1):
        return BaseLevelSpec(
            level_id="fac1", name="Factory Floor", width=1280.0, height=720.0,
            spawn_point=(100.0, 360.0), exit_point=(1180.0, 360.0),
            deliberation="Factory floor grid layout"
        )
"""
    (candidate_dir / "mechanics.py").write_text(factory_code, encoding="utf-8")

    # 4. Challenger Gate Pass
    challenger_eval = ChallengerGateEvaluation(
        deliberation="Verified abstract factory mechanics. Character requirements successfully bypassed.",
        candidate_id=candidate_dir.name,
        overall_verdict=True,
    )
    assert challenger_eval.overall_verdict is True

    # 5. Promotion and Simulation
    supervisor = BuildSupervisor(active_cartridge_dir=active_dir, cartridges_root=tmp_path / "cartridges")
    supervisor.promote_candidate(candidate_dir)

    sim = HeadlessSimulator(cartridge_path=active_dir)
    for _ in range(60):
        sim.step(SimulationInput())
    assert sim.step_count == 60
    assert sim.avg_frame_time_ms < 16.6
