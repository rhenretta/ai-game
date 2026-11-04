"""tests/test_coproducer.py
Comprehensive verification suite for the Interactive Studio Co-Producer (`CoProducerAgent`):
- Rule 4 Rationale-First validation (`deliberation` strictly at property/key index 0 across all CoProducer schemas and payloads)
- `CoProducerThreadStore` SQLite WAL + disk persistence for pre-genesis drafts and cartridge-bound threads
- `CoProducerAgent` multi-step tool loop (`inspect_studio_progress`, `inspect_playtest_sessions` with `replay.mp4` + frame PNGs,
  `consult_council` in `divergent_pitch_round` and `huddle` modes, `dispatch_specialist_probe` read-only `swe_engineer` probe,
  and `sketch_multimodal_concept` for 16:9 mockups and parametric DSP `.wav` audio)
- Interactive foreground execution while `.studio_paused.json` is active (`interactive_foreground_call_scope()`)
- Two-mode Council handoff: `refined_concept` (hydrates `GameBibleSpec`, approved mockups, and `GameMemoryStore` non-goals while
  skipping blind `_run_concept_genesis`) vs. `loose_spark` (enters `_run_concept_genesis`)
- 3-Tier Pivot Engine: `ProductionBoard.cancel_items_for_pivot` (cancels obsolete items, marks worker traces, rewires dependencies),
  `.pending_structural_pivot.json` consumption in `CreativeWritersRoom._consume_pending_steering`, and non-destructive
  `fork_cartridge_thread_and_assets`
- Mission Control `/api/coproducer/*` REST endpoints, `/api/cartridges` draft-folder exclusion, `record_milestone_notice`,
  and Rule 5 HTTP 503 failure fidelity when Gemini credentials are absent.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PIL import Image
from pydantic import ValidationError
from starlette.testclient import TestClient

from engine.studio.coproducer import (
    CoProducerAgent,
    CoProducerThreadStore,
    record_milestone_notice,
)
from engine.studio.game_memory import GameMemoryStore
from engine.studio.models import (
    CoProducerActionProposal,
    CoProducerAdvisorySpecialistTurn,
    CoProducerConceptBrief,
    CoProducerConsultationCard,
    CoProducerCouncilHuddleTurn,
    CoProducerDivergentPitchOption,
    CoProducerFinalTurn,
    CoProducerMediaCard,
    CoProducerMediaItem,
    CoProducerMemoryMutationSpec,
    CoProducerStepDecision,
    CoProducerToolCall,
    ConceptCandidate,
    ConceptCollisionProposal,
    ConceptFragment,
    ConceptSeedProposal,
    ConceptSelection,
    CoreLoopSpec,
    CreativeProvocation,
    CreativeProvocationSet,
    DirectorConvergenceCheck,
    EngagementPlan,
    ExperienceStoryboardBeat,
    GameBibleSpec,
    GenesisWorldSpec,
    GoalHierarchy,
    GoalSpec,
    LevelArchetypeSpec,
    LevelGenerationStrategy,
    NoveltyJudgement,
    OnboardingPlan,
    OnboardingStep,
    TrackCodeSynthesisPayload,
    WorkItemProposal,
    WritersRoomListenCheck,
    WritersRoomTurnProposal,
)
from engine.studio.production_board import ProductionBoard
from engine.studio.sfx_synth import SfxLayerSpec, SfxRecipe
from engine.studio.writers_room import SPECIALIST_ROSTER, CreativeWritersRoom
from engine.telemetry.db import TelemetryDB
from engine.ui.models import (
    CoProducerCreateDraftRequest,
    CoProducerExecuteActionRequest,
    CoProducerSendMessageRequest,
    CoProducerThreadDetailResponse,
)
from engine.ui.server import SwarmEventBus, create_app


def _make_png_bytes(width: int = 320, height: int = 180, rgba: tuple[int, int, int, int] = (24, 52, 88, 255)) -> bytes:
    img = Image.new("RGBA", (width, height), rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeUsageMetadata:
    prompt_token_count = 140
    candidates_token_count = 75
    thoughts_token_count = 20
    cached_content_token_count = 0
    total_token_count = 235


class _FakeGeminiResponse:
    def __init__(self, payload: Any):
        self.parsed = payload if hasattr(payload, "model_dump") else None
        if isinstance(payload, str):
            self.text = payload
        elif hasattr(payload, "model_dump_json"):
            self.text = payload.model_dump_json()
        else:
            self.text = json.dumps(payload)
        self.usage_metadata = _FakeUsageMetadata()


class _FakeCoProducerGeminiModels:
    """Deterministic schema-dispatching Gemini stub for Co-Producer, Council, Specialist Probes, and Media."""

    def __init__(self, step_decisions: List[CoProducerStepDecision] | None = None):
        self.calls: List[Dict[str, Any]] = []
        self._step_decisions = list(step_decisions or [])
        self._step_idx = 0
        self.concept_seed_calls = 0

    def generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
        schema = getattr(config, "response_schema", None) if config else None
        modalities = getattr(config, "response_modalities", None) if config else None
        schema_name = getattr(schema, "__name__", "") if schema else ""
        self.calls.append(
            {
                "model": model,
                "schema_name": schema_name,
                "modalities": modalities,
                "contents": contents,
            }
        )

        if modalities and "IMAGE" in modalities:
            png_bytes = _make_png_bytes(320, 180, (36, 82, 122, 255))

            class _InlineData:
                data = png_bytes
                mime_type = "image/png"

            class _Part:
                inline_data = _InlineData()

            class _Content:
                parts = [_Part()]

            class _Candidate:
                content = _Content()

            class _ImgResp:
                candidates = [_Candidate()]
                usage_metadata = _FakeUsageMetadata()

            return _ImgResp()

        if schema_name == "CoProducerStepDecision":
            if self._step_idx < len(self._step_decisions):
                dec = self._step_decisions[self._step_idx]
                self._step_idx += 1
                return _FakeGeminiResponse(dec)
            return _FakeGeminiResponse(
                CoProducerStepDecision(
                    deliberation="Sufficient studio evidence gathered; ready to synthesize user-facing response.",
                    next_step_intent="Synthesize grounded 1-on-1 Co-Producer reply.",
                    response_mode="studio_briefing",
                    tool_calls=[],
                    action="respond",
                )
            )

        if schema_name == "CoProducerFinalTurn":
            return _FakeGeminiResponse(
                CoProducerFinalTurn(
                    deliberation="Synthesizing high-agency 1-on-1 response grounded in live board telemetry, playtest video, Council pitches, and inline sketch.",
                    updated_thread_title="Kinetic Lattice",
                    updated_concept_brief=CoProducerConceptBrief(
                        deliberation="Capturing evolving tactile concept brief.",
                        working_title="Kinetic Lattice",
                        core_genre="systemic_spatial_puzzle",
                        elevator_pitch="A tactile spatial routing game where signal pulses snap through modular relays.",
                        core_fantasy="Conducting rhythmic energy cascades across reactive geometric machinery.",
                        core_player_verbs=["place_relay", "phase_pulse", "redirect_beam"],
                        first_60_seconds_feel="Immediate snap-placement of a relay completes a resonant circuit within 10 seconds.",
                        visual_style_direction="High-contrast blueprint amber and cyan over slate industrial plates.",
                        audio_mood_direction="Crisp electro-mechanical clicks layered over warm harmonic hums.",
                        color_palette_hex=["#0F172A", "#38BDF8", "#F59E0B"],
                        explicit_non_goals=["No text-heavy tutorial popups", "No timer-induced game-over"],
                        open_creative_questions=["Should relay overloads split beams or invert phase?"],
                    ),
                    action_proposals=[
                        CoProducerActionProposal(
                            deliberation="Offering explicit Tier 1 course correction action card for user approval.",
                            proposal_id="prop_steer_01",
                            action_kind="pivot_course_correction",
                            title="Steer Active Slice Focus",
                            summary="Queues tactile relay-snap polish note for the Council without cancelling active work.",
                            steering_or_seed_brief="Tighten relay snap feedback and resonant ring visibility.",
                        )
                    ],
                    spoken_response=(
                        "Here is where **Kinetic Lattice** stands right now: the board has active slice work underway, "
                        "the latest playtest replay is attached below alongside a fresh **16:9 concept mockup**, and I "
                        "checked with both the Council and Lead SWE on relay routing feasibility."
                    ),
                )
            )

        if schema_name == "CoProducerAdvisorySpecialistTurn":
            return _FakeGeminiResponse(
                CoProducerAdvisorySpecialistTurn(
                    deliberation="Evaluating specialist trade-offs for tactile spatial clarity.",
                    key_recommendation="Use diegetic pulse rings on relay nodes instead of HUD counters.",
                    creative_angle_or_risk="Keep beam intersections visually distinct at 3+ relays.",
                    spoken_quote="Pulse rings directly on the brass housing make timing legible at a glance.",
                )
            )

        if schema_name == "CreativeProvocationSet":
            return _FakeGeminiResponse(
                CreativeProvocationSet(
                    deliberation="Crafting orthogonal provocations across all Council specialists.",
                    provocations=[
                        CreativeProvocation(
                            deliberation=f"Provocation for {sp['role']}.",
                            specialist_role=sp["role"],
                            provocation=f"What single tactile interaction defines {sp['role']} in 10 seconds?",
                        )
                        for sp in SPECIALIST_ROSTER
                    ],
                )
            )

        if schema_name == "ConceptSeedProposal":
            self.concept_seed_calls += 1
            return _FakeGeminiResponse(
                ConceptSeedProposal(
                    deliberation="Generating divergent domain-agnostic mechanical and atmospheric fragments.",
                    fragments=[
                        ConceptFragment(
                            deliberation="Tactile verb fragment.",
                            fragment_id="frag_verb_1",
                            contributor_role="GameplayDesigner",
                            fragment_kind="verb",
                            fragment="Snap-rotating prism relays that split kinetic pulses on beat.",
                        ),
                        ConceptFragment(
                            deliberation="Spatial tension fragment.",
                            fragment_id="frag_tension_1",
                            contributor_role="VisualArtDirector",
                            fragment_kind="aesthetic",
                            fragment="Diegetic phosphor traces etched directly into dark slate plates.",
                        ),
                    ],
                )
            )

        if schema_name == "ConceptCollisionProposal":
            return _FakeGeminiResponse(
                ConceptCollisionProposal(
                    deliberation="Colliding fragments into two contrasting playable directions.",
                    candidates=[
                        ConceptCandidate(
                            deliberation="Candidate A emphasizes rhythmic routing.",
                            candidate_id="cand_a",
                            working_title="Prism Conduit",
                            premise="Route harmonic pulses across rotating brass prisms with zero HUD clutter.",
                            source_fragment_ids=["GameplayDesigner_0", "VisualArtDirector_0"],
                            core_tension="Balancing throughput against prism heat saturation.",
                        ),
                        ConceptCandidate(
                            deliberation="Candidate B emphasizes topological folding.",
                            candidate_id="cand_b",
                            working_title="Folded Circuit",
                            premise="Slide modular slate tiles to bridge broken signal traces in real time.",
                            source_fragment_ids=["StoryWorldArchitect_0", "VisualArtDirector_1"],
                            core_tension="Sliding one tile severs adjacent active traces.",
                        ),
                    ],
                )
            )

        if schema_name == "NoveltyJudgement":
            return _FakeGeminiResponse(
                NoveltyJudgement(
                    deliberation="Evaluating candidate freshness against prior studio concepts.",
                    candidate_id="cand_a",
                    overlapping_elements=[],
                    is_distinct_from_prior_concepts=True,
                )
            )

        if schema_name == "ConceptSelection":
            return _FakeGeminiResponse(
                ConceptSelection(
                    deliberation="Selecting Prism Conduit for strong first-60-second clarity.",
                    selected_candidate_id="cand_a",
                    opening_dialogue="Let's pursue 'Prism Conduit'—snap-rotating relays with phosphor trace feedback.",
                )
            )

        if schema_name == "WritersRoomTurnProposal":
            return _FakeGeminiResponse(
                WritersRoomTurnProposal(
                    deliberation="Completing game-design architecture sections faithful to the active concept.",
                    spoken_dialogue="Fleshing out level progression, core loop, onboarding, engagement, and goal hierarchy.",
                    addressed_to=["GameplayDesigner", "VisualArtDirector"],
                    updated_first_60_seconds_hook="Rotate one prism within 5 seconds to complete the starter circuit.",
                    updated_diegetic_ui_philosophy="Signal brightness and ring expansion communicate charge state in-world.",
                    added_or_updated_storyboard_beats=[
                        ExperienceStoryboardBeat(
                            deliberation="Opening tactile circuit beat.",
                            beat_index=1,
                            phase_title="Starter Circuit",
                            player_emotional_state="Curiosity into immediate click-satisfaction.",
                            sensory_and_camera_staging="Clean top-down slate table with glowing cyan trace.",
                            diegetic_teaching_moment="Dormant resonator hums when beam aligns.",
                            narrative_integration="First chamber awakens.",
                        )
                    ],
                    updated_level_generation_strategy=LevelGenerationStrategy(
                        deliberation="Structured chamber progression.",
                        generation_approach="hybrid",
                        world_structure="Linked relay chambers with escalating beam intersections.",
                        level_archetypes=[
                            LevelArchetypeSpec(
                                deliberation="Intro chamber.",
                                archetype_id="starter_chamber",
                                purpose="teach",
                                visual_identity="clean slate plate",
                                signature_challenge="single beam redirect",
                            )
                        ],
                        generation_rules=["Every chamber has at least one clean harmonic solution."],
                        difficulty_and_pacing_curve="Measured escalation across chambers.",
                    ),
                    updated_core_loop=CoreLoopSpec(
                        deliberation="Rotate, pulse, observe.",
                        player_verbs=["rotate_prism", "split_beam"],
                        moment_to_moment_loop="rotate prism and observe live beam path",
                        minute_to_minute_loop="complete multi-resonator chamber circuit",
                        session_loop="awaken linked observatory wings",
                    ),
                    updated_engagement_plan=EngagementPlan(
                        deliberation="Tactile experimentation flow.",
                        entertainment_pillars=["immediate optical feedback", "emergent beam routing"],
                        pacing_and_intensity_curve="calm puzzle focus punctuated by resonant payoffs",
                    ),
                    updated_onboarding_plan=OnboardingPlan(
                        deliberation="Zero-text teaching via beam alignment.",
                        world_introduction="Single misaligned prism beside a glowing emitter.",
                        steps=[
                            OnboardingStep(
                                deliberation="Step 1.",
                                step_index=0,
                                concept_introduced="prism rotation",
                                diegetic_method="live beam sweep",
                            ),
                            OnboardingStep(
                                deliberation="Step 2.",
                                step_index=1,
                                concept_introduced="beam splitting",
                                diegetic_method="dual resonator chime",
                            ),
                        ],
                    ),
                    updated_goal_hierarchy=GoalHierarchy(
                        deliberation="Clear multi-horizon goals.",
                        goals=[
                            GoalSpec(
                                deliberation="Short-term goal.",
                                goal_id="g_short",
                                horizon="short_term",
                                description="Ignite the active chamber resonator",
                            ),
                            GoalSpec(
                                deliberation="Long-term goal.",
                                goal_id="g_long",
                                horizon="long_term",
                                description="Restore full harmonic array",
                            ),
                        ],
                        end_state="Observatory array fully synchronized",
                    ),
                )
            )

        if schema_name == "WritersRoomListenCheck":
            return _FakeGeminiResponse(
                WritersRoomListenCheck(
                    deliberation="Listening check complete; no further specialist interjection required.",
                    should_interject=False,
                    urgency_score=0.1,
                )
            )

        if schema_name == "DirectorConvergenceCheck":
            return _FakeGeminiResponse(
                DirectorConvergenceCheck(
                    deliberation="Concept foundation is cohesive.",
                    experience_cohesion_score=0.92,
                    is_ready_to_lock_bible=True,
                )
            )

        if schema_name == "GenesisWorldSpec":
            return _FakeGeminiResponse(
                GenesisWorldSpec(
                    deliberation="Synthesizing final GenesisWorldSpec from Living Game Bible.",
                    game_title="Kinetic Lattice",
                    core_genre="systemic_spatial_puzzle",
                    thematic_premise="Modular signal routing across reactive slate plates.",
                    world_lore="An acoustic relay observatory suspended in a quiet cavern.",
                    conflict_tone="Focused, tactile systemic mastery",
                    conflict_narrative="Restore harmonic flow across dormant relay chambers.",
                    aesthetic_style="Crisp geometric vector silhouettes with warm phosphor glow",
                    palette_mood="Deep slate, cyan signal, warm brass",
                    gameplay_pacing="measured_tactile",
                    pacing_score=0.6,
                    target_difficulty=0.5,
                )
            )

        if schema_name == "TrackCodeSynthesisPayload":
            return _FakeGeminiResponse(
                TrackCodeSynthesisPayload(
                    deliberation="Inspecting `main.gd` in read-only mode to assess relay node architecture without mutating files.",
                    operation_mode="inspect_only",
                    inspection_report="`main.gd` uses clean modular Node2D signal emitters; adding phase-splitting relays requires zero scene-tree rewrite.",
                    recommended_followups=["Add `RelayNode` scene hook in `scenes.gd` when ready."],
                )
            )

        if schema_name == "SfxRecipe":
            return _FakeGeminiResponse(
                SfxRecipe(
                    deliberation="Designing crisp mechanical click + warm harmonic chime for relay snap prototype.",
                    total_duration_s=0.35,
                    loop_seamlessly=False,
                    master_gain=0.75,
                    layers=[
                        SfxLayerSpec(
                            deliberation="Fundamental harmonic ping.",
                            source="triangle",
                            start_s=0.0,
                            duration_s=0.32,
                            freq_start_hz=520.0,
                            freq_end_hz=780.0,
                            attack_s=0.005,
                            release_s=0.12,
                            gain=0.6,
                        )
                    ],
                )
            )

        raise AssertionError(f"Unhandled schema in _FakeCoProducerGeminiModels: {schema_name!r}")


class _FakeCoProducerClient:
    def __init__(self, step_decisions: List[CoProducerStepDecision] | None = None):
        self.models = _FakeCoProducerGeminiModels(step_decisions=step_decisions)


def _seed_synthetic_cartridge(cartridges_root: Path, db: TelemetryDB, cartridge_id: str = "candidate_001") -> Path:
    c_dir = cartridges_root / cartridge_id
    (c_dir / "assets" / "sprites").mkdir(parents=True, exist_ok=True)
    (c_dir / "assets" / "audio").mkdir(parents=True, exist_ok=True)
    (c_dir / "assets" / "mockups").mkdir(parents=True, exist_ok=True)

    (c_dir / "main.gd").write_text(
        "extends Node2D\n\nfunc _ready() -> void:\n\tpass\n",
        encoding="utf-8",
    )
    (c_dir / "assets" / "sprites" / "relay_node.png").write_bytes(_make_png_bytes(64, 64, (56, 189, 248, 255)))
    (c_dir / "assets" / "mockups" / "board_01.png").write_bytes(_make_png_bytes(320, 180, (15, 23, 42, 255)))

    bible = GameBibleSpec(
        deliberation="Synthetic cartridge bible for Co-Producer verification.",
        cartridge_id=cartridge_id,
        status="in_production",
        game_title="Synthetic Build 001",
        core_genre="systemic_spatial_puzzle",
        elevator_pitch="Route kinetic signal pulses across modular relays.",
        core_fantasy="Tactile circuit mastery.",
    )
    (c_dir / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    board = ProductionBoard(path=c_dir / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="Core relay placement mechanic.",
                item_id="item_relay_core",
                kind="mechanic_code",
                title="Implement relay snap placement",
                spec="Snap modular relays cleanly onto slate socket coordinates.",
                proof_spec="Headless test verifying relay placement and signal emission.",
            )
        ],
        raised_by_role="CreativeDirector",
        slice_id="slice_0",
    )

    pt_id = "playtest_synth_001"
    pt_dir = c_dir / "artifacts" / "playtests" / pt_id
    (pt_dir / "screenshots").mkdir(parents=True, exist_ok=True)
    replay_mp4 = pt_dir / "replay.mp4"
    replay_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42synthetic_video")
    shot_png = pt_dir / "screenshots" / "turn_000.png"
    shot_png.write_bytes(_make_png_bytes(320, 180, (30, 64, 110, 255)))

    db.upsert_playtest_session(
        {
            "deliberation": "Synthetic playtest session telemetry.",
            "session_id": pt_id,
            "checkpoint_id": cartridge_id,
            "candidate_id": cartridge_id,
            "game_title": "Synthetic Build 001",
            "status": "completed",
            "turn_count": 1,
            "sim_duration_sec": 3.2,
            "video_path": str(replay_mp4),
            "session_report": {
                "deliberation": "Playtester session summary.",
                "overall_fun_rating": 8,
                "overall_experience_summary": "Relay snapping feels immediate and legible.",
            },
        }
    )
    db.insert_playtest_turn(
        {
            "deliberation": "Turn 0 observation.",
            "session_id": pt_id,
            "turn_index": 0,
            "mood_state": "engaged",
            "append_log_line": "Placed first relay and watched cyan pulse complete circuit.",
            "screenshot_path": str(shot_png),
        }
    )
    return c_dir


def test_rule4_rationale_first_schema_invariants():
    """All CoProducer* studio and UI schemas enforce `deliberation` strictly at index 0 and reject blank rationale."""
    models_to_check = [
        CoProducerMediaItem,
        CoProducerMediaCard,
        CoProducerCouncilHuddleTurn,
        CoProducerDivergentPitchOption,
        CoProducerConsultationCard,
        CoProducerConceptBrief,
        CoProducerMemoryMutationSpec,
        CoProducerActionProposal,
        CoProducerToolCall,
        CoProducerStepDecision,
        CoProducerFinalTurn,
        CoProducerAdvisorySpecialistTurn,
        CoProducerCreateDraftRequest,
        CoProducerSendMessageRequest,
        CoProducerExecuteActionRequest,
        CoProducerThreadDetailResponse,
    ]
    for cls in models_to_check:
        field_names = list(cls.model_fields.keys())
        assert field_names[0] == "deliberation", f"{cls.__name__} must declare `deliberation` at index 0"

    with pytest.raises(ValidationError):
        CoProducerConceptBrief(deliberation="   ", working_title="Invalid Blank Rationale")

    dumped = CoProducerConceptBrief(
        deliberation="Valid first-position rationale.",
        working_title="Valid Concept",
    ).model_dump()
    assert list(dumped.keys())[0] == "deliberation"


def test_thread_store_draft_and_cartridge_persistence(tmp_path: Path):
    """CoProducerThreadStore persists pre-genesis drafts and cartridge threads across SQLite WAL + disk JSON."""
    db = TelemetryDB(tmp_path / "telemetry.db")
    cartridges_root = tmp_path / "cartridges"
    cartridges_root.mkdir()
    (cartridges_root / "candidate_001").mkdir()

    store = CoProducerThreadStore(cartridges_root, db=db)
    draft = store.create_draft_thread(title="Untethered Concept Sandbox")
    assert list(draft.keys())[0] == "deliberation"
    assert draft["scope_kind"] == "draft"
    assert draft["thread_id"].startswith("draft_")

    cart_thread = store.get_or_create_cartridge_thread("candidate_001", game_title="Synthetic Build 001")
    assert cart_thread["scope_kind"] == "cartridge"
    assert cart_thread["thread_id"] == "thread_candidate_001"

    _, msg = store.append_message(
        draft,
        {
            "deliberation": "Persisting initial assistant turn.",
            "sender_role": "coproducer",
            "content_markdown": "Let's explore two distinct mechanical hooks.",
        },
    )
    assert list(msg.keys())[0] == "deliberation"

    detail = store.get_thread_detail(draft["thread_id"])
    assert list(detail.keys())[0] == "deliberation"
    assert len(detail["messages"]) == 1
    assert detail["messages"][0]["content_markdown"] == "Let's explore two distinct mechanical hooks."

    listing = store.list_threads()
    assert list(listing.keys())[0] == "deliberation"
    tids = {t["thread_id"] for t in listing["threads"]}
    assert draft["thread_id"] in tids
    assert cart_thread["thread_id"] in tids


def test_coproducer_agent_multistep_tools_and_paused_foreground_execution(tmp_path: Path):
    """CoProducerAgent executes progress, playtest video, Council, SWE probe, and multimodal sketch tools even while studio build is paused."""
    db = TelemetryDB(tmp_path / "telemetry.db")
    cartridges_root = tmp_path / "cartridges"
    c_dir = _seed_synthetic_cartridge(cartridges_root, db, "candidate_001")

    # Activate studio pause flag to verify interactive_foreground_call_scope() bypasses background worker pause
    (cartridges_root / ".studio_paused.json").write_text(
        json.dumps({"paused": True, "reason": "User paused background workers"}),
        encoding="utf-8",
    )

    step_decisions = [
        CoProducerStepDecision(
            deliberation="First gather live board progress, playtest video replay, and divergent Council pitches.",
            next_step_intent="Inspect studio progress, playtests, and consult Council.",
            response_mode="studio_briefing",
            action="call_tools",
            tool_calls=[
                CoProducerToolCall(
                    deliberation="Fetch live ProductionBoard counts.",
                    tool_name="inspect_studio_progress",
                ),
                CoProducerToolCall(
                    deliberation="Fetch latest playtest video and screenshots.",
                    tool_name="inspect_playtest_sessions",
                ),
                CoProducerToolCall(
                    deliberation="Ask Council for divergent pitch collisions.",
                    tool_name="consult_council",
                    council_mode="divergent_pitch_round",
                    query_or_question="Give me two contrasting overload mechanics for modular signal relays.",
                ),
            ],
        ),
        CoProducerStepDecision(
            deliberation="Now run a read-only Lead SWE feasibility check on `main.gd` and sketch a 16:9 concept mockup.",
            next_step_intent="Probe Lead SWE and sketch 16:9 mockup.",
            response_mode="creative_ideation",
            action="call_tools",
            tool_calls=[
                CoProducerToolCall(
                    deliberation="Check `main.gd` structure non-destructively.",
                    tool_name="dispatch_specialist_probe",
                    probe_persona="swe_engineer",
                    query_or_question="Can `main.gd` support phase-splitting relays cleanly?",
                    target_paths=["main.gd"],
                ),
                CoProducerToolCall(
                    deliberation="Render a 16:9 visual mockup to anchor our conversation.",
                    tool_name="sketch_multimodal_concept",
                    sketch_kind="concept_mockup_16x9",
                    sketch_title="Phase Relay Observatory",
                    query_or_question="16:9 widescreen top-down slate relay chamber with glowing cyan signal beams and amber brass rings.",
                ),
            ],
        ),
        CoProducerStepDecision(
            deliberation="All live telemetry, consultations, and visual sketches are staged; ready to respond.",
            next_step_intent="Deliver grounded 1-on-1 response.",
            response_mode="creative_ideation",
            action="respond",
            tool_calls=[],
        ),
    ]

    fake_client = _FakeCoProducerClient(step_decisions=step_decisions)
    emitted: List[tuple[str, Dict[str, Any]]] = []
    store = CoProducerThreadStore(cartridges_root, db=db)
    cart_thread = store.get_or_create_cartridge_thread("candidate_001", game_title="Synthetic Build 001")

    main_gd_before = (c_dir / "main.gd").read_text(encoding="utf-8")

    agent = CoProducerAgent(
        client=fake_client,
        cartridges_root=cartridges_root,
        db=db,
        event_callback=lambda etype, payload: emitted.append((etype, payload)),
        max_tool_hops=6,
    )
    result = agent.run_turn(
        thread_id=cart_thread["thread_id"],
        user_message="Show me current progress, our latest playtest video, consult the Council & SWE, and sketch a 16:9 mockup.",
    )

    assert list(result.keys())[0] == "deliberation"
    asst = result["assistant_message"]
    assert asst["sender_role"] == "coproducer"
    assert "Kinetic Lattice" in asst["content_markdown"]

    # Verify read-only SWE probe never mutated main.gd
    assert (c_dir / "main.gd").read_text(encoding="utf-8") == main_gd_before

    # Verify staged media cards include progress_summary, playtest_video (with replay.mp4), and sketch_showcase (16:9 PNG)
    card_kinds = [c["card_kind"] for c in asst["media_cards"]]
    assert "progress_summary" in card_kinds
    assert "playtest_video" in card_kinds
    assert "sketch_showcase" in card_kinds

    pt_card = next(c for c in asst["media_cards"] if c["card_kind"] == "playtest_video")
    assert any(
        it["media_kind"] == "video" and "/api/playtest/sessions/playtest_synth_001/video" in it["artifact_path"]
        for it in pt_card["items"]
    )

    # Verify collapsible consultation cards for both Council divergent pitch round and Lead SWE probe
    consult_kinds = [c["consultation_kind"] for c in asst["consultation_cards"]]
    assert "divergent_pitch_round" in consult_kinds
    assert "specialist_probe" in consult_kinds

    # Verify studio-builder observability trace + token usage + action proposal
    assert len(asst["prompt_trace"]) >= 3
    assert asst["token_usage"]["total_tokens"] > 0
    assert len(asst["action_proposals"]) == 1
    assert asst["action_proposals"][0]["action_kind"] == "pivot_course_correction"

    # Also verify parametric DSP `.wav` audio sketch generation directly
    audio_cards: List[CoProducerMediaCard] = []
    summary = agent._tool_sketch_multimodal_concept(
        CoProducerToolCall(
            deliberation="Prototype crisp relay snap SFX via parametric DSP.",
            tool_name="sketch_multimodal_concept",
            sketch_kind="sfx_or_ambience",
            sketch_title="Relay Snap Ping",
            query_or_question="Crisp mechanical brass latch click followed by warm harmonic chime",
            loop_audio=False,
        ),
        result["thread"],
        [],
        audio_cards,
    )
    assert ".wav" in summary
    assert len(audio_cards) == 1
    wav_rel = audio_cards[0].items[0].artifact_path
    assert (c_dir / wav_rel).is_file()
    assert (c_dir / wav_rel).stat().st_size > 44


def test_two_mode_council_handoff_refined_concept_vs_loose_spark(tmp_path: Path):
    """`refined_concept` hydrates GameBibleSpec + mockups + non-goal memories and skips `_run_concept_genesis`; `loose_spark` runs genesis."""
    db = TelemetryDB(tmp_path / "telemetry.db")
    cartridges_root = tmp_path / "cartridges"
    cartridges_root.mkdir()
    store = CoProducerThreadStore(cartridges_root, db=db)

    # 1. Create a draft with an approved 16:9 sketch and graduate in `refined_concept` mode
    draft = store.create_draft_thread(title="Refined Draft")
    draft_dir = store.resolve_thread_dir(draft)
    sketch_rel = "artifacts/coproducer_sketches/mockup_approved.png"
    sketch_abs = draft_dir / sketch_rel
    sketch_abs.parent.mkdir(parents=True, exist_ok=True)
    sketch_abs.write_bytes(_make_png_bytes(320, 180, (20, 90, 130, 255)))

    brief = CoProducerConceptBrief(
        deliberation="Producer-approved refined concept brief.",
        working_title="Harmonic Conduit",
        core_genre="tactile_circuit_puzzle",
        elevator_pitch="Snap-rotate brass prisms to route resonant light pulses across dark slate chambers.",
        core_fantasy="Hands-on optical engineering with immediate visual/acoustic feedback.",
        core_player_verbs=["rotate_prism", "split_beam"],
        first_60_seconds_feel="Rotate one prism to ignite the central resonator within 8 seconds.",
        visual_style_direction="Slate stone, cyan beams, warm brass housings.",
        audio_mood_direction="Tactile mechanical detents and glass harmonics.",
        color_palette_hex=["#0F172A", "#22D3EE", "#F59E0B"],
        explicit_non_goals=["Zero countdown timers", "Zero floating damage numbers"],
        selected_mockup_paths=[sketch_rel],
    )

    refined_dir = cartridges_root / "candidate_refined_01"
    grad_meta = store.graduate_draft_to_cartridge(
        draft_thread_id=draft["thread_id"],
        new_cartridge_dir=refined_dir,
        handoff_mode="refined_concept",
        steering_or_seed_brief=brief.elevator_pitch,
        concept_brief=brief.model_dump(),
    )
    assert grad_meta["cartridge_id"] == "candidate_refined_01"
    assert (refined_dir / "co_producer_brief.json").is_file()
    assert (refined_dir / "assets" / "mockups" / "mockup_approved.png").is_file()

    refined_client = _FakeCoProducerClient()
    room_refined = CreativeWritersRoom(client=refined_client, telemetry_db=db, max_turns=2)
    room_refined.max_turns = 0  # Isolate opening turn hydration vs genesis
    bible_refined = room_refined.run_writers_room(
        candidate_dir=refined_dir,
        cartridge_id="candidate_refined_01",
    )

    assert refined_client.models.concept_seed_calls == 0, "refined_concept MUST skip blind _run_concept_genesis"
    assert bible_refined.game_title == "Harmonic Conduit"
    assert any(m.image_path == "assets/mockups/mockup_approved.png" for m in bible_refined.visual_mockups)
    assert bible_refined.writers_room_transcript[0].speaker_role == "ExecutiveProducer"

    mem_store = GameMemoryStore(refined_dir)
    council_memories = mem_store.list_entries(bank_filter="council")
    assert any("Zero countdown timers" in e.content for e in council_memories)

    # 2. Contrast with `loose_spark` handoff -> MUST invoke `_run_concept_genesis`
    spark_draft = store.create_draft_thread(title="Spark Draft")
    spark_dir = cartridges_root / "candidate_spark_01"
    store.graduate_draft_to_cartridge(
        draft_thread_id=spark_draft["thread_id"],
        new_cartridge_dir=spark_dir,
        handoff_mode="loose_spark",
        steering_or_seed_brief="Explore something surprising around magnetic tether rebounds.",
    )
    spark_client = _FakeCoProducerClient()
    room_spark = CreativeWritersRoom(client=spark_client, telemetry_db=db, max_turns=2)
    room_spark.max_turns = 0
    bible_spark = room_spark.run_writers_room(
        candidate_dir=spark_dir,
        cartridge_id="candidate_spark_01",
        creative_prompt="Explore something surprising around magnetic tether rebounds.",
    )
    assert spark_client.models.concept_seed_calls >= 1, "loose_spark MUST invoke _run_concept_genesis"
    assert bible_spark.game_title == "Prism Conduit"


def test_three_tier_pivot_engine_board_cancellation_and_cartridge_fork(tmp_path: Path):
    """Tier 2 prunes obsolete board tasks + marks worker traces + rewires dependencies; Tier 3 forks carryover assets non-destructively."""
    db = TelemetryDB(tmp_path / "telemetry.db")
    cartridges_root = tmp_path / "cartridges"
    c_dir = _seed_synthetic_cartridge(cartridges_root, db, "candidate_001")

    board = ProductionBoard(path=c_dir / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="Obsolete timer hazard task to be pruned during structural pivot.",
                item_id="item_obsolete_timer",
                kind="mechanic_code",
                title="Add countdown hazard",
                spec="Implement countdown failure timer.",
                proof_spec="Headless test verifying timer expiration.",
            ),
            WorkItemProposal(
                deliberation="Downstream task initially blocked on obsolete timer task.",
                item_id="item_surviving_polish",
                kind="mechanic_code",
                title="Polish relay pulse glow",
                spec="Scale pulse glow with harmonic resonance.",
                proof_spec="Visual proof verifying pulse glow.",
                depends_on=["item_obsolete_timer"],
            ),
        ],
        raised_by_role="GameplayDesigner",
        slice_id="slice_0",
    )
    assert board.get("item_surviving_polish").state == "blocked"

    # Seed an in-flight worker trace file for `item_obsolete_timer`
    trace_file = c_dir / "artifacts" / "work_items" / "item_obsolete_timer" / "lead_trace_rev0.json"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(json.dumps({"deliberation": "In-flight trace", "status": "running"}), encoding="utf-8")

    cancelled = board.cancel_items_for_pivot(["item_obsolete_timer"], reason="Producer removed countdown timers")
    assert [it.item_id for it in cancelled] == ["item_obsolete_timer"]
    assert board.get("item_obsolete_timer").state == "cancelled"
    assert board.get("item_surviving_polish").depends_on == []
    assert board.get("item_surviving_polish").state == "ready"

    trace_after = json.loads(trace_file.read_text(encoding="utf-8"))
    assert trace_after["status"] == "cancelled_by_pivot"

    # Verify `.pending_structural_pivot.json` consumption by CreativeWritersRoom._consume_pending_steering
    (c_dir / ".pending_structural_pivot.json").write_text(
        json.dumps(
            [
                {
                    "deliberation": "Pivot away from timers toward pure spatial routing.",
                    "steering_brief": "Replace time pressure with multi-frequency beam harmonics.",
                    "cancel_item_ids": [],
                    "memory_mutations": [
                        {
                            "deliberation": "Record non-goal in council memory.",
                            "operation": "record",
                            "memory_bank": "council",
                            "kind": "architectural_decision",
                            "title": "No Countdown Timers",
                            "content": "Never add countdown failure clocks.",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    room = CreativeWritersRoom(client=_FakeCoProducerClient(), board=board, telemetry_db=db)
    bible = GameBibleSpec.model_validate_json((c_dir / "game_bible.json").read_text(encoding="utf-8"))
    steered_notes = room._consume_pending_steering(bible, candidate_dir=c_dir)
    assert any("STRUCTURAL PIVOT" in s for s in steered_notes)
    assert not (c_dir / ".pending_structural_pivot.json").exists()
    assert any("Never add countdown failure clocks" in m.content for m in GameMemoryStore(c_dir).list_entries(bank_filter="council"))

    # Tier 3: Non-destructive fork into a fresh cartridge carrying over selected sprite asset
    store = CoProducerThreadStore(cartridges_root, db=db)
    store.get_or_create_cartridge_thread("candidate_001", game_title="Synthetic Build 001")
    fork_dir = cartridges_root / "candidate_genesis_fork01"
    fork_meta = store.fork_cartridge_thread_and_assets(
        source_cartridge_dir=c_dir,
        new_cartridge_dir=fork_dir,
        handoff_mode="refined_concept",
        steering_or_seed_brief="Reimagine as a calm acoustic observatory using the existing relay sprite.",
        concept_brief={"deliberation": "Forked concept brief.", "working_title": "Acoustic Observatory"},
        carry_over_asset_paths=["assets/sprites/relay_node.png"],
    )
    assert fork_meta["cartridge_id"] == "candidate_genesis_fork01"
    assert (fork_dir / "assets" / "sprites" / "relay_node.png").is_file()
    assert (fork_dir / "co_producer_brief.json").is_file()
    # Original cartridge remains completely untouched
    assert (c_dir / "game_bible.json").is_file()


def test_coproducer_rest_endpoints_cartridge_filtering_and_rule5_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """REST endpoints serve threads/messages/actions, hide `_coproducer_drafts` from `/api/cartridges`, and return HTTP 503 without Gemini credentials."""
    db_file = tmp_path / "telemetry.db"
    db = TelemetryDB(db_file)
    cartridges_root = tmp_path / "cartridges"
    c_dir = _seed_synthetic_cartridge(cartridges_root, db, "candidate_001")

    fake_client = _FakeCoProducerClient()
    app = create_app(
        db_path=db_file,
        event_bus=SwarmEventBus(),
        cartridges_root=cartridges_root,
        genai_client=fake_client,
    )
    client = TestClient(app)

    # Create a draft thread -> creates `cartridges/_coproducer_drafts/draft_...` on disk
    r_draft = client.post("/api/coproducer/threads/draft", json={"title": "Fresh Sandbox"})
    assert r_draft.status_code == 200
    draft_payload = r_draft.json()
    assert list(draft_payload.keys())[0] == "deliberation"
    draft_id = draft_payload["thread"]["thread_id"]

    # `/api/cartridges` MUST NOT list `_coproducer_drafts` as a playable cartridge
    r_carts = client.get("/api/cartridges")
    assert r_carts.status_code == 200
    raw_carts = r_carts.json()["cartridges"]
    cart_ids = [c["cartridge_id"] if isinstance(c, dict) else str(c) for c in raw_carts]
    assert "candidate_001" in cart_ids
    assert "_coproducer_drafts" not in cart_ids

    # Post message to `candidate_001` (auto-resolves to `thread_candidate_001`)
    r_msg = client.post(
        "/api/coproducer/threads/candidate_001/messages",
        json={"message": "Give me a quick status check and steering option."},
    )
    assert r_msg.status_code == 200
    turn_json = r_msg.json()
    assert list(turn_json.keys())[0] == "deliberation"
    asst_msg = turn_json["assistant_message"]
    prop_id = asst_msg["action_proposals"][0]["proposal_id"]

    # Dismiss the action proposal cleanly via `/api/coproducer/actions/execute`
    r_dismiss = client.post(
        "/api/coproducer/actions/execute",
        json={
            "deliberation": "User opted to keep chatting before steering.",
            "thread_id": turn_json["thread"]["thread_id"],
            "message_id": asst_msg["message_id"],
            "proposal_id": prop_id,
            "dismiss": True,
        },
    )
    assert r_dismiss.status_code == 200
    assert r_dismiss.json()["status"] == "dismissed"

    # Record a milestone notice and execute a `pivot_structural` proposal via REST
    notice = record_milestone_notice(
        cartridges_root=cartridges_root,
        cartridge_id="candidate_001",
        deliberation="Playtest milestone completed; offering structural pivot test card.",
        content_markdown="**Slice 0 Playtest Ready** — review replay or prune tasks.",
        db=db,
        action_proposals=[
            CoProducerActionProposal(
                deliberation="Offer structural pivot pruning `item_relay_core`.",
                proposal_id="prop_struct_01",
                action_kind="pivot_structural",
                title="Prune & Replan Relay Loop",
                summary="Cancels `item_relay_core` and queues Council structural pivot.",
                steering_or_seed_brief="Pivot to continuous phase beams instead of discrete grid snapping.",
                cancel_item_ids=["item_relay_core"],
                memory_mutations=[
                    CoProducerMemoryMutationSpec(
                        deliberation="Store new continuous-beam direction.",
                        operation="record",
                        memory_bank="council",
                        kind="architectural_decision",
                        title="Continuous Phase Beams",
                        content="Use analog angle rotation rather than fixed grid cells.",
                    )
                ],
            ).model_dump()
        ],
    )
    assert notice is not None
    r_exec = client.post(
        "/api/coproducer/actions/execute",
        json={
            "deliberation": "User approved structural pivot action card.",
            "thread_id": "thread_candidate_001",
            "message_id": notice["message_id"],
            "proposal_id": "prop_struct_01",
        },
    )
    assert r_exec.status_code == 200
    exec_json = r_exec.json()
    assert exec_json["status"] == "executed"
    assert "item_relay_core" in exec_json["result"]["cancelled_item_ids"]
    assert (c_dir / ".pending_structural_pivot.json").is_file()

    # Verify direct producer-initiated commissioning (`direct_action_kind='commission_game'`) from an active draft
    store = CoProducerThreadStore(cartridges_root, db=db)
    draft_meta = store.get_thread_detail(draft_id)["thread"]
    draft_meta["concept_brief"] = {
        "deliberation": "Shaped concept pillars during live dialogue.",
        "working_title": "Resonant Orbit",
        "core_genre": "Spatial Rhythm",
        "elevator_pitch": "Tune orbital relays on a single screen to harmonize signal rings.",
        "core_player_verbs": ["place_relay", "phase_shift"],
    }
    store.append_message(
        draft_meta,
        {
            "deliberation": "Producer refined orbital concept.",
            "sender_role": "user",
            "content_markdown": "Let's tune orbital relays on a single screen.",
        },
    )

    r_direct_comm = client.post(
        "/api/coproducer/actions/execute",
        json={
            "deliberation": "Producer clicked Start Game Build directly from active Concept Lab header.",
            "thread_id": draft_id,
            "direct_action_kind": "commission_game",
            "override_handoff_mode": "refined_concept",
            "override_pause_for_greenlight": True,
        },
    )
    assert r_direct_comm.status_code == 200
    comm_json = r_direct_comm.json()
    assert comm_json["status"] == "executed"
    assert comm_json["action_kind"] == "commission_game"
    new_cid = comm_json["result"]["target_cartridge_id"]
    assert new_cid.startswith("candidate_genesis_")
    assert comm_json["result"]["handoff_mode"] == "refined_concept"
    assert comm_json["result"]["job_id"]
    assert (cartridges_root / new_cid / "co_producer_brief.json").is_file()

    # Rule 5: Zero Silent Fallbacks — missing Gemini client & unset GEMINI_API_KEY surfaces HTTP 503
    r_draft_2 = client.post("/api/coproducer/threads/draft", json={"title": "Unauthenticated Check"})
    draft_id_2 = r_draft_2.json()["thread"]["thread_id"]
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    app_no_key = create_app(
        db_path=db_file,
        event_bus=SwarmEventBus(),
        cartridges_root=cartridges_root,
        genai_client=None,
    )
    client_no_key = TestClient(app_no_key)
    r_503 = client_no_key.post(
        f"/api/coproducer/threads/{draft_id_2}/messages",
        json={"message": "Can we brainstorm without an API key?"},
    )
    assert r_503.status_code == 503
    assert "GEMINI_API_KEY" in r_503.json()["error"]


def test_structural_pivot_module_path_cancellation_refinement_cancel_and_report_archival(tmp_path: Path):
    from engine.studio.models import GameSliceSpec, WorkItemRefinement
    from engine.studio.production_board import ProductionBoard
    from engine.studio.writers_room import CreativeWritersRoom

    c_dir = tmp_path / "candidate_pivot_race"
    (c_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    board = ProductionBoard(path=c_dir / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="Legacy mechanic module task queued before structural pivot.",
                item_id="mechanic_legacy_module_fix",
                kind="mechanic_code",
                title="Fix legacy module dispatch",
                spec="Patch legacy module.",
                proof_spec="Headless test.",
                target_modules=["scenes/mechanics/legacy_controller.gd"],
            ),
            WorkItemProposal(
                deliberation="Legacy UI remediation task spawned by superseded gate.",
                item_id="ui_legacy_panel_fix",
                kind="ui",
                title="Fix legacy panel spacing",
                spec="Patch legacy HUD panel.",
                proof_spec="Visual proof.",
                target_modules=["scenes/ui/legacy_hud.gd"],
            ),
        ],
        raised_by_role="GameplayDesigner",
        slice_id="slice_0_proto",
    )

    # Seed pre-pivot pass0..pass3 playtest reports on disk
    for p_idx in range(4):
        (c_dir / "artifacts" / f"blind_playtest_report_slice_0_proto_pass{p_idx}.json").write_text(
            json.dumps({"council_verdict": "iterate_current_slice", "pass_index": p_idx}),
            encoding="utf-8",
        )

    (c_dir / ".pending_structural_pivot.json").write_text(
        json.dumps(
            [
                {
                    "deliberation": "Operator confirmed structural pivot mid-gate.",
                    "steering_brief": "Replace legacy prototype loop with modular structural assembly.",
                    "cancel_item_ids": ["res://scenes/mechanics/legacy_controller.gd"],
                    "memory_mutations": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    bible = GameBibleSpec(
        deliberation="Test bible with active slice_0_proto.",
        game_title="Prototype Fixture",
        core_genre="Simulation",
        elevator_pitch="A modular systems fixture.",
        slice_roadmap=[
            GameSliceSpec(
                deliberation="Initial slice.",
                slice_id="slice_0_proto",
                slice_index=0,
                title="Slice 0 Prototype",
                slice_objective="Test initial prototype.",
                player_acclimation_and_mechanics_focus="Core controls.",
                status="active",
            )
        ],
        active_slice_index=0,
    )
    room = CreativeWritersRoom(client=_FakeCoProducerClient(), board=board)
    assert room._has_unconsumed_operator_steering(c_dir) is True

    steered = room._consume_pending_steering(bible, candidate_dir=c_dir)
    assert any("STRUCTURAL PIVOT" in s for s in steered)
    assert room._has_unconsumed_operator_steering(c_dir) is False
    # Module path in cancel_item_ids matched target_modules on mechanic_legacy_module_fix
    assert board.get("mechanic_legacy_module_fix").state == "cancelled"
    # Pre-pivot pass0..pass3 reports were archived so the pivoted slice gets a fresh pass0 budget
    assert not (c_dir / "artifacts" / "blind_playtest_report_slice_0_proto_pass0.json").exists()
    archived = list((c_dir / "artifacts").glob("blind_playtest_report_slice_0_proto_pre_pivot_*_pass*.json"))
    assert len(archived) == 4

    # Council specialist refinement with action='cancel' retires remaining unstarted legacy item
    board.apply_refinements(
        [
            WorkItemRefinement(
                deliberation="Retire legacy UI fix superseded by structural pivot.",
                item_id="ui_legacy_panel_fix",
                action="cancel",
            )
        ],
        by_role="UXExperienceDesigner",
    )
    assert board.get("ui_legacy_panel_fix").state == "cancelled"


def test_coproducer_presented_card_curation_and_multi_session_sync(tmp_path: Path):
    """CoProducerFinalTurn.presented_card_ids demotes background inspection cards to show_by_default=False while keeping prototypes & multi-cartridge session sync intact."""
    db = TelemetryDB(tmp_path / "telemetry.db")
    cartridges_root = tmp_path / "cartridges"
    c1_dir = _seed_synthetic_cartridge(cartridges_root, db, "candidate_alpha_01")
    _seed_synthetic_cartridge(cartridges_root, db, "candidate_beta_02")

    # Seed persisted co_producer.json snapshots in two separate game sessions + one draft
    store_seed = CoProducerThreadStore(cartridges_root, db=db)
    t_alpha = store_seed.get_or_create_cartridge_thread("candidate_alpha_01")
    t_beta = store_seed.get_or_create_cartridge_thread("candidate_beta_02")
    t_draft = store_seed.create_draft_thread(title="Fresh Session Concept")

    # Even a fresh disk-only store (db=None) discovers all cartridge + draft sessions from disk
    disk_only_store = CoProducerThreadStore(cartridges_root, db=None)
    listed = disk_only_store.list_threads()
    listed_ids = {t["thread_id"] for t in listed["threads"]}
    assert t_alpha["thread_id"] in listed_ids
    assert t_beta["thread_id"] in listed_ids
    assert t_draft["thread_id"] in listed_ids

    # Run a turn where Co-Producer inspects studio progress in the background AND sketches a prototype,
    # then explicitly sets presented_card_ids=[] so only the prototype is shown inline by default.
    class _CuratingModels(_FakeCoProducerGeminiModels):
        def generate_content(self, *, model, contents, config):
            schema = getattr(config, "response_schema", None)
            schema_name = getattr(schema, "__name__", "")
            if schema_name == "CoProducerFinalTurn":
                return _FakeGeminiResponse(
                    CoProducerFinalTurn(
                        deliberation="Inspected board metrics internally to ground reply; only presenting the requested sketch inline.",
                        presented_card_ids=[],
                        spoken_response="Board progress is healthy—here is the requested concept sketch.",
                    )
                )
            return super().generate_content(model=model, contents=contents, config=config)

    client = _FakeCoProducerClient(
        step_decisions=[
            CoProducerStepDecision(
                deliberation="Check board metrics internally and render one sketch.",
                next_step_intent="Run progress inspection + sketch prototype.",
                response_mode="creative_ideation",
                tool_calls=[
                    CoProducerToolCall(
                        deliberation="Check current slice status silently.",
                        tool_name="inspect_studio_progress",
                    ),
                    CoProducerToolCall(
                        deliberation="Render requested concept sketch.",
                        tool_name="sketch_multimodal_concept",
                        sketch_kind="concept_mockup_16x9",
                        sketch_title="Chamber Plate",
                        query_or_question="Clean orthographic chamber plate",
                    ),
                ],
                action="call_tools",
            )
        ]
    )
    client.models = _CuratingModels(client.models._step_decisions)

    agent = CoProducerAgent(client=client, cartridges_root=cartridges_root, db=db)
    turn_out = agent.run_turn(
        thread_id=t_alpha["thread_id"],
        user_message="How is the build going, and can you sketch a chamber plate?",
    )
    cards = turn_out["assistant_message"]["media_cards"]
    by_kind = {c["card_kind"]: c for c in cards}
    assert "progress_summary" in by_kind
    assert "sketch_showcase" in by_kind
    # Background inspection card is retained for Behind-the-scenes drill-down, hidden by default
    assert by_kind["progress_summary"]["show_by_default"] is False
    # Explicitly requested prototype sketch stays visible inline by default
    assert by_kind["sketch_showcase"]["show_by_default"] is True
    assert (c1_dir / "co_producer.json").is_file()


def test_coproducer_council_parallel_ignores_paused_background_build_and_unfiltered_threads_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that Co-Producer `consult_council` (which fans out via `CreativeWritersRoom._map_parallel`)
    propagates `interactive_foreground_call_scope` to child threads so paused background builds never block
    interactive Co-Producer turns, and `GET /api/coproducer/threads` without `?cartridge=` returns 200 OK."""
    cartridges_root = tmp_path / "cartridges"
    db_path = tmp_path / "telemetry.db"
    db = TelemetryDB(db_path)
    c_dir = _seed_synthetic_cartridge(cartridges_root, db, "candidate_001")
    # Simulate global + candidate background build pause
    (cartridges_root / ".studio_paused.json").write_text('{"paused": true}', encoding="utf-8")
    (c_dir / ".studio_paused.json").write_text('{"paused": true}', encoding="utf-8")
    monkeypatch.setenv("STUDIO_ENFORCE_GLOBAL_PAUSE", "1")
    monkeypatch.setenv("STUDIO_FAIL_FAST_ON_PAUSE", "1")

    step_decisions = [
        CoProducerStepDecision(
            deliberation="Convene Council huddle across multiple specialists in parallel.",
            next_step_intent="Consult Council specialists.",
            response_mode="creative_ideation",
            action="call_tools",
            tool_calls=[
                CoProducerToolCall(
                    deliberation="Consult CreativeDirector, GameplayDesigner, and VisualArtDirector in parallel.",
                    tool_name="consult_council",
                    council_mode="huddle",
                    specialist_roles=["CreativeDirector", "GameplayDesigner", "VisualArtDirector"],
                    query_or_question="Propose contrasting core loop hooks.",
                )
            ],
        ),
        CoProducerStepDecision(
            deliberation="Synthesize concise response.",
            next_step_intent="Reply to user.",
            response_mode="creative_ideation",
            action="respond",
            tool_calls=[],
        ),
    ]
    fake_client = _FakeCoProducerClient(step_decisions=step_decisions)
    store = CoProducerThreadStore(cartridges_root, db=db)
    draft = store.create_draft_thread(title="New Game Concept")

    agent = CoProducerAgent(client=fake_client, cartridges_root=cartridges_root, db=db)
    res = agent.run_turn(thread_id=draft["thread_id"], user_message="Pitch a new game concept.")
    assert res["assistant_message"]["sender_role"] == "coproducer"
    assert len(res["assistant_message"]["consultation_cards"]) == 1

    # Also verify retry deduplication when user message is already trailing on an interrupted draft
    store.append_message(
        res["thread"],
        {
            "deliberation": "User submitted interactive message to Studio Co-Producer.",
            "sender_role": "user",
            "content_markdown": "Follow-up prompt after reload",
        },
    )
    res_retry = agent.run_turn(thread_id=draft["thread_id"], user_message="Follow-up prompt after reload")
    user_msgs = [
        m
        for m in store.get_thread_detail(draft["thread_id"])["messages"]
        if m["sender_role"] == "user" and m["content_markdown"] == "Follow-up prompt after reload"
    ]
    assert len(user_msgs) == 1
    assert res_retry["assistant_message"]["sender_role"] == "coproducer"

    app = create_app(
        db_path=db_path,
        event_bus=SwarmEventBus(),
        cartridges_root=cartridges_root,
        genai_client=fake_client,
    )
    with TestClient(app) as http:
        resp = http.get("/api/coproducer/threads")
        assert resp.status_code == 200
        body = resp.json()
        assert any(t["thread_id"] == draft["thread_id"] for t in body["draft_threads"])


def test_coproducer_is_active_council_member_and_leads_council_deliberation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that CoProducer sits in CORE_COUNCIL_ROSTER & SPECIALIST_ROSTER as Council Chair,
    carries 1-on-1 Co-Producer session history (`co_producer.json` + `co_producer_brief.json`) into
    Council prompts, actively participates in `_select_next_speaker` + subagent delegation, chairs
    proof reviews (`_reviewer_for`) and convergence, and syncs Blind Playtest verdicts back to its thread."""
    from types import SimpleNamespace

    from engine.studio.game_memory import normalize_agent_role
    from engine.studio.models import (
        CANONICAL_MEMORY_BANKS,
        DirectorConvergenceCheck,
        GenesisWorldSpec,
        WorkItem,
        WritersRoomListenCheck,
        WritersRoomTurnProposal,
    )
    from engine.studio.writers_room import (
        CORE_COUNCIL_ROSTER,
        COUNCIL_CHAIR_DISPLAY_NAME,
        COUNCIL_CHAIR_ROLE,
        COUNCIL_HIERARCHY,
        SPECIALIST_ROSTER,
        CreativeWritersRoom,
        _resolve_council_specialist,
    )

    # 1. Roster, hierarchy, memory bank, and backward-compatibility alias checks
    assert COUNCIL_CHAIR_ROLE == "CoProducer"
    assert CORE_COUNCIL_ROSTER[0]["role"] == "CoProducer"
    assert CORE_COUNCIL_ROSTER[0]["display_name"] == COUNCIL_CHAIR_DISPLAY_NAME
    assert any(s["role"] == "CoProducer" for s in SPECIALIST_ROSTER)
    assert "GameplaySystemsEngineer" in COUNCIL_HIERARCHY["CoProducer"]
    assert "UXExperienceDesigner" in COUNCIL_HIERARCHY["CoProducer"]
    assert "CoProducer" in CANONICAL_MEMORY_BANKS
    assert normalize_agent_role("co-producer") == "CoProducer"
    assert normalize_agent_role("council_chair") == "CoProducer"
    assert _resolve_council_specialist("CoProducer")["role"] == "CoProducer"  # type: ignore[index]
    assert _resolve_council_specialist("CreativeDirector")["role"] == "CoProducer"  # type: ignore[index]

    # 2. Reviewer routing chairs integration, level_map, and legacy CreativeDirector items with CoProducer
    integ_item = WorkItem(
        deliberation="Integration task check.",
        item_id="integ_01",
        kind="integration",
        title="Scene integration",
        spec="Wire systems.",
        proof_spec="Headless boot.",
        raised_by_role="",
    )
    legacy_dir_item = WorkItem(
        deliberation="Legacy raised item check.",
        item_id="legacy_01",
        kind="mechanic_code",
        title="Core mechanic",
        spec="Implement loop.",
        proof_spec="Headless test.",
        raised_by_role="CreativeDirector",
    )
    assert CreativeWritersRoom._reviewer_for(integ_item) == "CoProducer"
    assert CreativeWritersRoom._reviewer_for(legacy_dir_item) == "CoProducer"

    # 3. Seed 1-on-1 Co-Producer conversation + refined handoff brief and verify CoProducer brings that
    #    session memory into the Writers' Room, leads Turn 0, interjects mid-deliberation on Turn 2,
    #    and locks the Game Bible on convergence.
    cartridges_root = tmp_path / "cartridges"
    cand_dir = cartridges_root / "candidate_chair_001"
    cand_dir.mkdir(parents=True)
    store = CoProducerThreadStore(cartridges_root)
    t_meta = store.get_or_create_cartridge_thread(cand_dir.name, game_title="Harmonic Foundry")
    t_meta, _ = store.append_message(
        t_meta,
        {
            "deliberation": "Executive Producer alignment message.",
            "sender_role": "user",
            "content_markdown": "Keep placement strictly manual where the cursor clicks—zero auto-routing.",
        },
    )
    store.append_message(
        t_meta,
        {
            "deliberation": "Co-Producer session commitment.",
            "sender_role": "coproducer",
            "content_markdown": "Locked in: manual cursor-exact placement and zero passive resource trickle at t=0.",
        },
    )
    (cand_dir / "co_producer_brief.json").write_text(
        json.dumps(
            {
                "deliberation": "Refined concept handoff from Co-Producer session.",
                "handoff_mode": "refined_concept",
                "cartridge_id": cand_dir.name,
                "steering_or_seed_brief": "Build Harmonic Foundry with manual cursor-exact placement.",
                "concept_brief": {
                    "deliberation": "Co-designed brief.",
                    "working_title": "Harmonic Foundry",
                    "core_genre": "tactile_spatial_puzzle",
                    "elevator_pitch": "Align acoustic resonators by hand across a cavern grid.",
                    "core_player_verbs": ["Place resonator", "Tune pitch"],
                    "first_60_seconds_feel": "Silent dark cavern; player manually places the first resonator to reveal terrain.",
                    "explicit_non_goals": ["Zero passive resource generation", "Zero auto-placement"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    captured_coproducer_turn_prompts: List[str] = []

    class _CouncilChairTestClient:
        def __init__(self) -> None:
            self.turn_counter = 0
            outer = self

            class _Models:
                def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
                    schema = getattr(config, "response_schema", None) if config else None
                    s_name = getattr(schema, "__name__", "")
                    p_txt = str(contents)
                    if s_name == "GenreCraftFoundation":
                        from engine.studio.models import GenreCraftFoundation, GenrePillarSpec

                        gcf = GenreCraftFoundation(
                            deliberation="Synthesizing tactile spatial puzzle foundation.",
                            genre_and_subgenre_archetype="Tactile Spatial Puzzle",
                            core_fun_thesis="Earned spatial reasoning from exact manual placement.",
                            player_fantasy_and_emotional_payoff="Turning an inert cavern into a resonant network you built yourself.",
                            foundational_pillars=[
                                GenrePillarSpec(
                                    deliberation="Manual spatial placement pillar.",
                                    pillar_id="manual_placement",
                                    title="Cursor-Exact Placement",
                                    why_it_creates_fun="Every tile coordinate changes wave bounce geometry.",
                                    concrete_player_decisions="Choose exact grid cell and angle under tight resonator count.",
                                    anti_pattern_to_avoid="Snapping pieces to arbitrary positions away from the player's click.",
                                ),
                                GenrePillarSpec(
                                    deliberation="Scarcity bootstrap pillar.",
                                    pillar_id="manual_bootstrap",
                                    title="Scarcity Bootstrap",
                                    why_it_creates_fun="Every unlocked node feels earned.",
                                    concrete_player_decisions="Recover dormant nodes in the cavern before expanding.",
                                    anti_pattern_to_avoid="Giving free infinite resources at t=0.",
                                ),
                                GenrePillarSpec(
                                    deliberation="Legible cause-and-effect pillar.",
                                    pillar_id="legible_waves",
                                    title="Legible Acoustic Feedback",
                                    why_it_creates_fun="Players immediately diagnose why a circuit succeeded or missed.",
                                    concrete_player_decisions="Inspect wave reflection lines to re-angle resonators.",
                                    anti_pattern_to_avoid="Low-res ambiguous icons with unclear function.",
                                ),
                            ],
                            bootstrap_to_mastery_arc="t=0 starts completely inert with 1 resonator in hand; player scouts and expands manually.",
                            spatial_and_ux_precision_contracts=[
                                "Clicks map 1:1 via get_global_mouse_position() to the exact hovered grid cell."
                            ],
                            genre_anti_patterns_to_never_commit=[
                                "Self-completing loops that progress with zero player input."
                            ],
                        )
                        return SimpleNamespace(text=gcf.model_dump_json(), usage_metadata=None)
                    if s_name == "WritersRoomListenCheck":
                        # Turn 1: GameplayDesigner speaks; Turn 2: CoProducer interjects mid-deliberation!
                        if outer.turn_counter == 0:
                            active = "You are the Gameplay Designer (GameplayDesigner)" in p_txt
                        else:
                            active = "You are the Co-Producer (Council Chair) (CoProducer)" in p_txt
                        lc = WritersRoomListenCheck(
                            deliberation="Selecting speaker for Council deliberation.",
                            should_interject=active,
                            urgency_score=0.96 if active else 0.12,
                            intended_contribution_summary="Align mechanics with Executive Producer session commitments.",
                        )
                        return SimpleNamespace(text=lc.model_dump_json(), usage_metadata=None)
                    if s_name == "WritersRoomTurnProposal":
                        idx = outer.turn_counter
                        outer.turn_counter += 1
                        if "You are the Co-Producer (Council Chair) (CoProducer)" in p_txt:
                            captured_coproducer_turn_prompts.append(p_txt)
                        prop = WritersRoomTurnProposal(
                            deliberation="Advancing Council specification with cursor-exact placement and manual bootstrap.",
                            spoken_dialogue=(
                                "Enforcing our session commitment: clicks map 1:1 to grid coordinates and t=0 starts inert."
                                if idx >= 1
                                else "Defining core resonator placement loop."
                            ),
                            addressed_to=["CoProducer"] if idx == 0 else ["GameplaySystemsEngineer"],
                            updated_diegetic_ui_philosophy="Crisp hovered-cell ghost preview directly under cursor.",
                            added_or_updated_storyboard_beats=[
                                {
                                    "deliberation": "Opening beat.",
                                    "beat_index": 0,
                                    "phase_title": "Silent Cavern",
                                    "player_emotional_state": "Curious and focused",
                                    "sensory_and_camera_staging": "Dark cavern grid with crisp hovered-cell highlight.",
                                    "diegetic_teaching_moment": "Cursor ghost outline lights up valid cavern sockets.",
                                    "narrative_integration": "Quiet cave drip until the first resonator hums.",
                                }
                            ],
                        )
                        return SimpleNamespace(text=prop.model_dump_json(), usage_metadata=None)
                    if s_name == "DirectorConvergenceCheck":
                        conv = DirectorConvergenceCheck(
                            deliberation="All pillars and producer session commitments are unified.",
                            experience_cohesion_score=0.94,
                            missing_or_weak_elements=[],
                            director_steering_prompt="",
                            is_ready_to_lock_bible=True,
                        )
                        return SimpleNamespace(text=conv.model_dump_json(), usage_metadata=None)
                    if s_name == "GenesisWorldSpec":
                        ws = GenesisWorldSpec(
                            deliberation="Synthesizing final world spec as Co-Producer Council Chair.",
                            game_title="Harmonic Foundry",
                            core_genre="tactile_spatial_puzzle",
                            thematic_premise="Align acoustic resonators by hand across a cavern grid.",
                            world_lore="Subterranean acoustic chambers.",
                            conflict_tone="focused_methodical",
                            conflict_narrative="Route clean harmonic waves across cavern obstacles.",
                            aesthetic_style="High-contrast slate grid and brass resonators.",
                            palette_mood="deep_slate_brass",
                            gameplay_pacing="methodical_tactical",
                            pacing_score=0.5,
                            target_difficulty=0.6,
                            camera_style="fixed_orthographic",
                            mechanics_guidelines=["1:1 cursor-to-cell placement", "Inert t=0 bootstrap"],
                        )
                        return SimpleNamespace(text=ws.model_dump_json(), usage_metadata=None)
                    raise AssertionError(f"Unexpected schema in Council Chair test: {s_name}")

            self.models = _Models()

    chair_client = _CouncilChairTestClient()
    room = CreativeWritersRoom(client=chair_client, max_turns=4, min_turns_before_convergence=2)
    monkeypatch.setattr(room, "_complete_missing_design_sections", lambda *a, **kw: None)
    monkeypatch.setattr("engine.studio.writers_room.validate_game_design", lambda *a, **kw: [])
    monkeypatch.setattr(room, "_missing_required_board_specialists", lambda *a, **kw: [])

    bible = room.run_writers_room(
        dna=None,
        creative_prompt="Build Harmonic Foundry.",
        candidate_dir=cand_dir,
    )

    roles = [u.speaker_role for u in bible.writers_room_transcript]
    # ExecutiveProducer handoff -> CoProducer opening -> GameplayDesigner ->
    # CoProducer active mid-room interjection -> CoProducer convergence lock!
    assert roles[:5] == [
        "ExecutiveProducer",
        "CoProducer",
        "GameplayDesigner",
        "CoProducer",
        "CoProducer",
    ]
    assert "Locked Living Game Bible for production" in (bible.writers_room_transcript[4].bible_updates_summary or [])
    # Verify CoProducer's Council turn received the 1-on-1 Co-Producer conversation history!
    assert len(captured_coproducer_turn_prompts) >= 1
    assert "YOUR 1-ON-1 CO-PRODUCER SESSION MEMORY WITH THE EXECUTIVE PRODUCER" in captured_coproducer_turn_prompts[0]
    assert "Keep placement strictly manual where the cursor clicks" in captured_coproducer_turn_prompts[0]
    assert "Zero passive resource generation" in captured_coproducer_turn_prompts[0]
