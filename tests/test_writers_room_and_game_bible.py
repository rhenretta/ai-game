"""tests/test_writers_room_and_game_bible.py
Verification suite for the Multi-Agent Creative Writers' Room & Living Game Bible:
- Rationale-First (Rule 4) schema validation for all Writers' Room and Game Bible models
- Open-ended conversational turn-taking (WritersRoomListenCheck -> WritersRoomTurnProposal -> DirectorConvergenceCheck)
- Mid-conversation 16:9 concept & diegetic UI mockup synthesis (VisualArtisanAgent.generate_concept_mockup)
- SQLite WAL studio_game_bibles persistence & Mission Control REST endpoints (/api/studio/game-bible, /api/studio/greenlight, /api/studio/settings)
- Downstream grounding in GameConceptDirectorAgent, LeadGameplaySWEAgent, and ChallengerGate
- Rule 1 domain-agnostic verification (zero hardcoded game titles/entities in prompts)
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, List
import pytest
from PIL import Image
from starlette.testclient import TestClient

from engine.steam.models import SteamPlayerDNA
from engine.studio.director import GameConceptDirectorAgent
from engine.studio.models import (
    CharacterSheet,
    ConceptEvolutionEntry,
    CoreLoopSpec,
    EngagementPlan,
    GoalHierarchy,
    GoalSpec,
    LevelArchetypeSpec,
    LevelGenerationStrategy,
    OnboardingPlan,
    OnboardingStep,
    CutsceneBeat,
    CutsceneScript,
    DialogueNode,
    DialogueTreeSpec,
    DirectorConvergenceCheck,
    ExperienceStoryboardBeat,
    GameBibleSpec,
    GameRequirementsSpec,
    GenesisWorldSpec,
    NarrativeBible,
    StudioBuildState,
    WritersRoomListenCheck,
    WritersRoomTurnProposal,
    WritersRoomUtterance,
    VisualMockupArtifact,
)
from engine.studio.orchestrator import StudioOrchestrator
from engine.studio.swe import LeadGameplaySWEAgent
from engine.studio.visual import VisualArtisanAgent
from engine.studio.writers_room import CreativeWritersRoom
from engine.supervisor.challenger import ChallengerGate
from engine.telemetry.db import TelemetryDB
from engine.ui.models import GreenlightRequest, StudioSettingsUpdateRequest
from engine.ui.server import SwarmEventBus, create_app


class _FakeGeminiResponse:
    def __init__(self, payload: Any):
        if isinstance(payload, str):
            self.text = payload
        elif hasattr(payload, "model_dump_json"):
            self.text = payload.model_dump_json()
        else:
            self.text = json.dumps(payload)


class _SchemaDispatchGeminiModels:
    """Fake Gemini models endpoint that inspects response_schema to return valid structured JSON."""

    def __init__(self):
        self.calls: List[dict] = []
        self.listen_call_count = 0
        self.turn_call_count = 0
        self.convergence_call_count = 0

    def generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
        schema = getattr(config, "response_schema", None) if config else None
        modalities = getattr(config, "response_modalities", None) if config else None
        self.calls.append({"model": model, "schema": schema, "modalities": modalities})

        # Multimodal image generation call (16:9 mockup or sprite)
        if modalities and "IMAGE" in modalities:
            img = Image.new("RGBA", (320, 180), (29, 43, 83, 255))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            class _InlineData:
                data = png_bytes

            class _Part:
                inline_data = _InlineData()

            class _Content:
                parts = [_Part()]

            class _Candidate:
                content = _Content()

            class _ImgResp:
                candidates = [_Candidate()]

            return _ImgResp()

        schema_name = getattr(schema, "__name__", "") if schema else ""

        if schema_name == "WritersRoomListenCheck":
            self.listen_call_count += 1
            should_speak = self.listen_call_count in (1, 3, 5)
            urgency = 0.95 if self.listen_call_count == 1 else (0.85 if should_speak else 0.2)
            return _FakeGeminiResponse(
                WritersRoomListenCheck(
                    deliberation=f"Evaluating conversation relevance on listen check #{self.listen_call_count}.",
                    should_interject=should_speak,
                    urgency_score=urgency,
                    intended_contribution_summary="Proposing diegetic world hook and 16:9 visual mockup.",
                )
            )

        if schema_name == "WritersRoomTurnProposal":
            self.turn_call_count += 1
            if self.turn_call_count == 1:
                return _FakeGeminiResponse(
                    WritersRoomTurnProposal(
                        deliberation="Opening pitch: establishing a kinetic atmospheric world with zero static HUD text.",
                        spoken_dialogue="Let's build 'Chrono Canopy', an atmospheric momentum-based canopy glider where wind currents and lantern glow communicate speed and danger directly in the world.",
                        addressed_to=["GameplayDesigner", "UXExperienceDesigner", "VisualArtDirector"],
                        updated_game_title="Chrono Canopy",
                        updated_elevator_pitch="An atmospheric momentum-based canopy glider where wind currents and lantern glow communicate speed and danger directly in the world.",
                        updated_core_fantasy="Glide through bioluminescent wind canyons using rhythmic wing-tuck dives and thermal updrafts.",
                        updated_first_60_seconds_hook="Player wakes on a crumbling wind-spire as a storm wall approaches; diving off the ledge immediately catches an updraft.",
                        updated_diegetic_ui_philosophy="Scarf trail length and lantern color shift indicate stamina and altitude; zero on-screen text instructions.",
                        concept_evolution_milestone=ConceptEvolutionEntry(
                            deliberation="Initial pitch establishing momentum-driven canopy glider with diegetic scarf & lantern telemetry.",
                            author_role="CreativeDirector",
                            previous_premise="Untitled Concept",
                            evolved_premise="v1: Momentum-driven canopy glider with diegetic scarf & lantern telemetry.",
                            pivot_reason="Initial pitch replacing generic platforming with tactile wind physics.",
                        ),
                        added_or_updated_storyboard_beats=[
                            ExperienceStoryboardBeat(
                                deliberation="Beat 1 teaches diving without text prompts.",
                                beat_index=1,
                                phase_title="Opening Hook",
                                player_emotional_state="Vertigo transforming into exhilarating flight.",
                                sensory_and_camera_staging="Wide 16:9 pan revealing the storm wall behind the spire.",
                                diegetic_teaching_moment="Wind chimes and glowing thermal ribbons pull the camera downward.",
                                narrative_integration="Leap from the crumbling wind-spire into the canyon updraft as the storm rises.",
                            ),
                            ExperienceStoryboardBeat(
                                deliberation="Beat 2 introduces thermal charging.",
                                beat_index=2,
                                phase_title="First Discovery",
                                player_emotional_state="Wonder and harmonic flow.",
                                sensory_and_camera_staging="Warm amber rim-lighting contrasting against deep indigo clouds.",
                                diegetic_teaching_moment="Chime-towers emit expanding light rings when approached.",
                                narrative_integration="Skim across ancient chime-towers to resonantly charge the glider cape.",
                            ),
                        ],
                        gameplay_pillars_additions=[
                            "Momentum-conserving dive-and-soar flight physics",
                            "Diegetic scarf and lantern feedback (zero HUD text manuals)",
                            "Resonant chime-tower harmonic chain reactions",
                        ],
                    )
                )
            return _FakeGeminiResponse(
                WritersRoomTurnProposal(
                    deliberation="VisualArtDirector & Narrative synthesis: adding 16:9 gameplay mockup, protagonist Lyra, and intro cutscene.",
                    spoken_dialogue="I'm generating a 16:9 gameplay mockup showing the spire leap, glowing thermal ribbons, and scarf-trail stamina indicator, alongside Lyra's character sheet and opening spire cutscene.",
                    addressed_to=["CreativeDirector", "UXExperienceDesigner"],
                    concept_evolution_milestone=ConceptEvolutionEntry(
                        deliberation="Grounded the flight loop in visible thermal ribbons and diegetic cape glow.",
                        author_role="VisualArtDirector",
                        previous_premise="v1: Momentum-driven canopy glider",
                        evolved_premise="v2: Added harmonic chime-tower resonance, Lyra character arc, and 16:9 visual mockup board.",
                        pivot_reason="Grounded the flight loop in visible thermal ribbons and diegetic cape glow.",
                    ),
                    mockup_to_generate=VisualMockupArtifact(
                        deliberation="16:9 widescreen gameplay mockup demonstrating diegetic scarf stamina and thermal ribbons.",
                        mockup_id="spire_leap",
                        title="Spire Leap & Thermal Ribbons 16:9 Mockup",
                        scene_description="16:9 widescreen indie game screenshot of a glider leaping off a stone wind-spire into glowing cyan thermal ribbons, scarf glowing amber, zero text HUD.",
                        image_path="assets/mockups/mockup_1_spire_leap.png",
                        ui_callout_notes=["Scarf length and glow intensity show boost charge directly on the character."],
                    ),
                    added_or_updated_characters=[
                        CharacterSheet(
                            deliberation="Protagonist wind-weaver with expressive scarf silhouette.",
                            character_id="lyra",
                            name="Lyra",
                            role="protagonist",
                            personality_traits=["daring", "observant"],
                            backstory="Last apprentice of the High Spire.",
                            voice_timbre="Warm contralto with breathless urgency",
                            gemini_voice_name="Kore",
                        )
                    ],
                    added_or_updated_cutscenes=[
                        CutsceneScript(
                            deliberation="Opening spire awakening cutscene.",
                            cutscene_id="intro_spire",
                            title="The Storm Wall Rises",
                            trigger="episode_start",
                            beats=[
                                CutsceneBeat(
                                    deliberation="Opening beat alerting player to the storm.",
                                    beat_number=1,
                                    action_description="Storm lightning shatters the upper spire chimes.",
                                    speaker_id="lyra",
                                    dialogue_line="The upper chimes have shattered... I have to catch the canyon draft now!",
                                )
                            ],
                        )
                    ],
                    added_or_updated_dialogue_trees=[
                        DialogueTreeSpec(
                            deliberation="Root dialogue tree at the first beacon.",
                            root_node_id="beacon_1",
                            nodes={
                                "beacon_1": DialogueNode(
                                    deliberation="Root node at the first beacon.",
                                    node_id="beacon_1",
                                    speaker_id="lyra",
                                    text="The resonance core is still warm.",
                                    choices=[],
                                )
                            },
                        )
                    ],
                    visual_style_guide_update="Cyber-neon bioluminescent wind canyons with side-scrolling pan camera.",
                    audio_direction_update="Aeolian harp swells synchronized with airspeed.",
                    updated_level_generation_strategy=LevelGenerationStrategy(
                        deliberation="Hybrid generation keeps canyons varied yet fair.", generation_approach="hybrid",
                        world_structure="Five linked canyon regions around a central roost.",
                        level_archetypes=[LevelArchetypeSpec(
                            deliberation="Teaching canyon.", archetype_id="teaching_canyon", purpose="teach",
                            visual_identity="warm wide canyon", signature_challenge="first thermal climb")],
                        generation_rules=["Every canyon has a solvable thermal route."],
                        difficulty_and_pacing_curve="Ramp then relax per region."),
                    updated_core_loop=CoreLoopSpec(
                        deliberation="Dive, soar, repeat.", player_verbs=["dive", "soar"],
                        moment_to_moment_loop="dive then catch thermal", minute_to_minute_loop="chain thermals to a beacon",
                        session_loop="relight beacons then return to roost"),
                    updated_engagement_plan=EngagementPlan(
                        deliberation="Flow state.", entertainment_pillars=["momentum expression"],
                        pacing_and_intensity_curve="surge then glide"),
                    updated_onboarding_plan=OnboardingPlan(
                        deliberation="Learn by doing.", world_introduction="Wake on a spire as the storm approaches.",
                        steps=[OnboardingStep(deliberation="d", step_index=i, concept_introduced=c, diegetic_method="level layout")
                               for i, c in enumerate(["dive", "thermal", "beacon"])]),
                    updated_goal_hierarchy=GoalHierarchy(
                        deliberation="Nested goals.", goals=[
                            GoalSpec(deliberation="d", goal_id="g1", horizon="short_term", description="reach the next thermal"),
                            GoalSpec(deliberation="d", goal_id="g2", horizon="long_term", description="relight every beacon")],
                        end_state="Canopy glows"),
                )
            )

        if schema_name == "DirectorConvergenceCheck":
            self.convergence_call_count += 1
            ready = self.convergence_call_count >= 1
            return _FakeGeminiResponse(
                DirectorConvergenceCheck(
                    deliberation="All experience pillars, storyboard beats, and 16:9 visual mockups are concrete and cohesive.",
                    is_ready_to_lock_bible=ready,
                    experience_cohesion_score=0.94,
                    missing_or_weak_elements=[],
                    director_steering_prompt="",
                )
            )

        if schema_name == "GenesisWorldSpec":
            return _FakeGeminiResponse(
                GenesisWorldSpec(
                    deliberation="Translating locked Chrono Canopy Game Bible into GenesisWorldSpec.",
                    game_title="Chrono Canopy",
                    core_genre="Atmospheric Momentum Glider",
                    thematic_premise="Outrun the storm wall across the resonant wind canyons.",
                    world_lore="An ancient sky-archipelago where wind-weavers keep the storm at bay.",
                    conflict_tone="heroic_hopeful",
                    conflict_narrative="Lyra must ignite the three canyon beacon spires before the storm swallows the High Spire.",
                    aesthetic_style="Bioluminescent wind canyons with expressive scarf trails and glowing thermal ribbons.",
                    palette_mood="neon_cyberpunk",
                    gameplay_pacing="kinetic_high_mobility",
                    pacing_score=0.6,
                    target_difficulty=0.8,
                    camera_style="scrolling",
                    mechanics_guidelines=[
                        "Dive to build speed, pull up into cyan thermal ribbons to soar.",
                        "Scarf length and glow communicate boost stamina diegetically.",
                    ],
                )
            )

        if schema_name == "GameRequirementsSpec":
            return _FakeGeminiResponse(
                {
                    "deliberation": "Derived verifiable tracks from Living Game Bible.",
                    "spec_id": "spec_chrono_01",
                    "game_title": "Chrono Canopy",
                    "genre_paradigm": "Atmospheric Momentum Glider",
                    "target_episode": 0,
                    "requirements": [
                        {
                            "deliberation": "Core momentum flight track.",
                            "id": "R1",
                            "title": "Momentum Glider Physics & Diegetic Scarf",
                            "description": "Implement dive-and-soar physics and scarf trail telemetry.",
                            "assigned_agent": "LeadGameplaySWEAgent",
                            "acceptance_criteria": [
                                {
                                    "deliberation": "Verify player glider physics.",
                                    "id": "AC1.1",
                                    "description": "Player glider responds to dive and updraft forces.",
                                    "verification_method": "headless_playtest",
                                }
                            ],
                        }
                    ],
                }
            )

        if schema_name == "CreativeProvocationSet":
            from engine.studio.writers_room import SPECIALIST_ROSTER
            return _FakeGeminiResponse({
                "deliberation": "Orthogonal provocations.",
                "provocations": [
                    {"deliberation": f"axis {i}", "specialist_role": s_["role"], "provocation": f"What if axis {i} were the whole game?"}
                    for i, s_ in enumerate(SPECIALIST_ROSTER)
                ],
            })

        if schema_name == "ConceptSeedProposal":
            self.seed_call_count = getattr(self, "seed_call_count", 0) + 1
            return _FakeGeminiResponse({
                "deliberation": "Independent fragments.",
                "fragments": [
                    {"deliberation": "d", "fragment_id": "x", "fragment_kind": "interaction verb", "fragment": f"verb fragment {self.seed_call_count}a"},
                    {"deliberation": "d", "fragment_id": "y", "fragment_kind": "world tension", "fragment": f"tension fragment {self.seed_call_count}b"},
                ],
            })

        if schema_name == "ConceptCollisionProposal":
            self.collision_call_count = getattr(self, "collision_call_count", 0) + 1
            return _FakeGeminiResponse({
                "deliberation": "Collide.",
                "candidates": [
                    {"deliberation": "d", "candidate_id": f"c{i}", "working_title": f"Candidate {self.collision_call_count}.{i}",
                     "premise": f"Premise {self.collision_call_count}.{i}", "core_tension": "tension",
                     "source_fragment_ids": ["GameplayDesigner_0", "StoryWorldArchitect_1"]}
                    for i in range(4)
                ],
            })

        if schema_name == "NoveltyJudgement":
            self.novelty_call_count = getattr(self, "novelty_call_count", 0) + 1
            distinct = self.novelty_call_count > getattr(self, "novelty_reject_first_n", 0)
            return _FakeGeminiResponse({
                "deliberation": "Compared with each prior concept.", "candidate_id": "c0",
                "overlapping_elements": [] if distinct else ["same core interaction"],
                "is_distinct_from_prior_concepts": distinct,
            })

        if schema_name == "ConceptSelection":
            return _FakeGeminiResponse({
                "deliberation": "Chose for evolvability.", "opening_dialogue": "Here is a draft premise - tear it apart.",
                "selected_candidate_id": "c1",
            })

        raise ValueError(f"Unexpected schema in test fake: {schema_name}")


class _FakeGeminiClient:
    def __init__(self):
        self.models = _SchemaDispatchGeminiModels()


def _sample_dna() -> SteamPlayerDNA:
    return SteamPlayerDNA(
        deliberation="Player enjoys atmospheric movement and expressive indie action games.",
        player_id="test_player",
        total_games_analyzed=12,
        total_playtime_hours=420.0,
        top_genres=["Action", "Indie", "Adventure"],
        genre_weights={"Action": 0.5, "Indie": 0.3, "Adventure": 0.2},
        pacing_score=0.4,
        challenge_tolerance=0.6,
        thematic_biases=["atmospheric", "flight"],
        preferred_palette_mood="cyber_neon",
    )


def test_writers_room_models_enforce_rationale_first_index_zero():
    """Rule 4: Verify all new Writers' Room and Game Bible Pydantic models declare 'deliberation' at field index 0."""
    models_to_check = [
        WritersRoomListenCheck,
        ConceptEvolutionEntry,
        ExperienceStoryboardBeat,
        VisualMockupArtifact,
        WritersRoomUtterance,
        WritersRoomTurnProposal,
        DirectorConvergenceCheck,
        GameBibleSpec,
        GreenlightRequest,
        StudioSettingsUpdateRequest,
    ]
    for model_cls in models_to_check:
        field_names = list(model_cls.model_fields.keys())
        assert field_names[0] == "deliberation", f"{model_cls.__name__} must have 'deliberation' at field index 0, got {field_names}"


def test_visual_artisan_generate_concept_mockup_writes_16_9_png(tmp_path: Path):
    """Verify VisualArtisanAgent.generate_concept_mockup synthesizes a 16:9 PNG under assets/mockups/."""
    fake_client = _FakeGeminiClient()
    visual_agent = VisualArtisanAgent(client=fake_client)
    mockup = VisualMockupArtifact(
        deliberation="16:9 concept art board showing spire leap and thermal ribbons.",
        mockup_id="spire_leap",
        title="Spire Leap 16:9 Mockup",
        scene_description="Glider diving into cyan thermal ribbons",
        image_path="assets/mockups/spire_leap.png",
        ui_callout_notes=["Scarf glow shows stamina"],
    )
    out_path = visual_agent.generate_concept_mockup(
        mockup=mockup,
        output_path=tmp_path / "assets" / "mockups" / "spire_leap.png",
        style_guide="cyber_neon",
        dimensions=(640, 360),
    )
    assert out_path.is_file()
    assert out_path.parent.name == "mockups"
    with Image.open(out_path) as img:
        assert img.size == (640, 360)


def test_creative_writers_room_runs_open_ended_conversation_and_persists_bible(tmp_path: Path):
    """Verify CreativeWritersRoom runs open-ended multi-agent brainstorming, logs concept evolution,
    synthesizes 16:9 mockups, and persists game_bible.json + SQLite studio_game_bibles.
    """
    db_path = tmp_path / "telemetry.db"
    tdb = TelemetryDB(db_path)
    fake_client = _FakeGeminiClient()
    visual_agent = VisualArtisanAgent(client=fake_client)
    emitted_events: List[tuple] = []

    room = CreativeWritersRoom(
        client=fake_client,
        visual_agent=visual_agent,
        telemetry_db=tdb,
        event_callback=lambda etype, payload: emitted_events.append((etype, payload)),
        max_turns=4,
        min_turns_before_convergence=2,
    )

    candidate_dir = tmp_path / "candidate_genesis_test"
    bible = room.run_writers_room(
        dna=_sample_dna(),
        candidate_dir=candidate_dir,
        target_episode=0,
        cartridge_id="candidate_genesis_test",
    )

    assert isinstance(bible, GameBibleSpec)
    assert bible.status == "locked"
    assert bible.game_title == "Chrono Canopy"
    assert len(bible.writers_room_transcript) >= 2
    assert len(bible.concept_evolution) >= 2
    assert len(bible.experience_storyboard) >= 2
    assert len(bible.visual_mockups) >= 1
    assert bible.world_spec is not None
    assert bible.narrative_bible is not None

    # Verify game_bible.json, world_spec.json, and narrative_bible.json on disk
    assert (candidate_dir / "game_bible.json").is_file()
    assert (candidate_dir / "world_spec.json").is_file()
    assert (candidate_dir / "narrative_bible.json").is_file()

    # Verify SQLite studio_game_bibles persistence
    db_rec = tdb.get_latest_game_bible()
    assert db_rec is not None
    assert db_rec["game_title"] == "Chrono Canopy"
    assert db_rec["status"] == "locked"

    # Verify emitted real-time events
    event_types = [e[0] for e in emitted_events]
    assert "writers_room_utterance" in event_types
    assert "concept_evolution_logged" in event_types
    assert "bible_mockup_generated" in event_types
    assert "game_bible_updated" in event_types

    # Test Executive Producer steering injection
    updated_bible = room.inject_user_steering_and_continue(
        game_bible=bible,
        user_feedback="Add a co-op wind-tether mechanic between two gliders.",
        dna=_sample_dna(),
        candidate_dir=candidate_dir,
        additional_turns=1,
    )
    assert any(u.speaker_role == "UserSteering" for u in updated_bible.writers_room_transcript)


def test_downstream_director_swe_and_challenger_grounding_in_game_bible(tmp_path: Path):
    """Verify GameConceptDirectorAgent, LeadGameplaySWEAgent, and ChallengerGate consume game_bible.json and 16:9 mockups,
    and verify Rule 1 compliance (zero hardcoded Tellurium/Geocore/Tar-Phage strings in swe.py).
    """
    fake_client = _FakeGeminiClient()
    visual_agent = VisualArtisanAgent(client=fake_client)
    room = CreativeWritersRoom(client=fake_client, visual_agent=visual_agent, max_turns=3, min_turns_before_convergence=2)
    candidate_dir = tmp_path / "candidate_grounding"
    bible = room.run_writers_room(dna=_sample_dna(), candidate_dir=candidate_dir)

    # 1. Director generate_requirements_spec with game_bible
    director = GameConceptDirectorAgent(client=fake_client)
    req_spec = director.generate_requirements_spec(
        dna=_sample_dna(),
        world_spec=bible.world_spec,
        narrative_bible=bible.narrative_bible,
        game_bible=bible,
    )
    assert isinstance(req_spec, GameRequirementsSpec)
    assert req_spec.game_title == "Chrono Canopy"

    # 2. LeadGameplaySWEAgent._build_bible_and_narrative_context includes game_bible, cutscenes, dialogue trees, and mockup parts
    swe = LeadGameplaySWEAgent(client=fake_client)
    narrative_ctx, mockup_parts = swe._build_bible_and_narrative_context(candidate_dir)
    assert "LIVING GAME BIBLE" in narrative_ctx
    assert "UI, HUD & Onboarding Philosophy" in narrative_ctx
    assert "The Storm Wall Rises" in narrative_ctx
    assert len(mockup_parts) >= 1

    # Verify Rule 1 compliance: no hardcoded domain strings in engine/studio/swe.py
    swe_source = Path(swe.__class__.__module__.replace(".", "/") + ".py")
    if swe_source.is_file():
        text = swe_source.read_text(encoding="utf-8")
        for forbidden in ("Tellurium", "Geocore", "Tar-Phage", "MissionOnboardingPanel"):
            assert forbidden not in text, f"Rule 1 violation: forbidden hardcoded term '{forbidden}' found in swe.py"

    # 3. ChallengerGate._collect_asset_grounding and _build_evaluation_prompt load game_bible.json and 16:9 mockups
    challenger = ChallengerGate(client=fake_client)
    asset_grounding = challenger._collect_asset_grounding(candidate_dir, req_spec)
    assert len(asset_grounding["mockups_found"]) >= 1
    eval_prompt = challenger._build_evaluation_prompt(
        candidate_id="candidate_grounding",
        spec=req_spec,
        ast_grounding={},
        telemetry_grounding={},
        visual_grounding={},
        asset_grounding=asset_grounding,
        candidate_dir=candidate_dir,
    )
    assert "LIVING GAME BIBLE" in eval_prompt
    assert "Scarf trail length and lantern color shift" in eval_prompt


def test_mission_control_game_bible_and_greenlight_endpoints(tmp_path: Path):
    """Verify /api/studio/game-bible, /api/studio/greenlight, /api/studio/settings, and /api/assets/mockups/{filename}."""
    db_path = tmp_path / "telemetry.db"
    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    active_dir.mkdir(parents=True, exist_ok=True)

    tdb = TelemetryDB(db_path)
    bus = SwarmEventBus(db_path=db_path, log_path=tmp_path / "events.jsonl")
    fake_client = _FakeGeminiClient()
    visual_agent = VisualArtisanAgent(client=fake_client)
    room = CreativeWritersRoom(
        client=fake_client,
        visual_agent=visual_agent,
        telemetry_db=tdb,
        max_turns=3,
        min_turns_before_convergence=2,
    )
    bible = room.run_writers_room(dna=_sample_dna(), candidate_dir=active_dir, cartridge_id="active")

    app = create_app(
        db_path=db_path,
        cartridges_root=cartridges_root,
        active_cartridge_dir=active_dir,
        event_bus=bus,
    )
    client = TestClient(app)

    # 1. GET /api/studio/game-bible
    res = client.get("/api/studio/game-bible")
    assert res.status_code == 200
    data = res.json()
    assert list(data.keys())[0] == "deliberation"
    assert data["has_bible"] is True
    assert data["game_title"] == "Chrono Canopy"
    assert len(data["writers_room_transcript"]) >= 2
    assert len(data["visual_mockups"]) >= 1

    # 2. GET /api/assets/mockups/{filename}
    mockup_fname = Path(bible.visual_mockups[0].image_path).name
    m_res = client.get(f"/api/assets/mockups/{mockup_fname}")
    assert m_res.status_code == 200
    assert m_res.headers["content-type"] == "image/png"

    # 3. POST /api/studio/settings (toggle pause_for_greenlight)
    s_res = client.post(
        "/api/studio/settings",
        json={"deliberation": "Enable greenlight pause before coding", "pause_for_greenlight": True},
    )
    assert s_res.status_code == 200
    assert s_res.json()["pause_for_greenlight"] is True

    # 4. POST /api/studio/greenlight (inject steering feedback, then approve)
    g_res = client.post(
        "/api/studio/greenlight",
        json={
            "deliberation": "Executive Producer steering note",
            "approved": False,
            "feedback": "Emphasize harmonic wind chimes during high-speed dives.",
        },
    )
    assert g_res.status_code == 200
    assert g_res.json()["status"] == "steering_injected"

    g_approve = client.post(
        "/api/studio/greenlight",
        json={"deliberation": "Approve Living Game Bible for production", "approved": True, "feedback": ""},
    )
    assert g_approve.status_code == 200
    assert g_approve.json()["status"] == "greenlit"

    # 5. Verify React studio-web Writers' Room, Living Game Bible, and ModalDialog components are present
    idx_res = client.get("/")
    assert idx_res.status_code == 200
    studio_web_src = Path(__file__).resolve().parent.parent / "studio-web" / "src"
    assert (studio_web_src / "features" / "writers-room" / "WritersRoomView.tsx").is_file()
    assert (studio_web_src / "features" / "bible" / "GameBibleView.tsx").is_file()
    assert (studio_web_src / "components" / "ModalDialog.tsx").is_file()

    # 6. Verify reset_for_new_genesis_build clears old cartridge bible and initializes fresh brainstorming state
    fresh_bible = bus.reset_for_new_genesis_build(
        build_id="build_scratch_01",
        creative_prompt="A cozy botanical clockwork observatory mystery",
    )
    assert fresh_bible["status"] == "brainstorming"
    assert fresh_bible["writers_room_transcript"] == []
    assert "A cozy botanical clockwork observatory mystery" in fresh_bible["creative_vision_summary"]

    res_after_reset = client.get("/api/studio/game-bible")
    assert res_after_reset.status_code == 200
    after_data = res_after_reset.json()
    assert after_data["status"] == "brainstorming"
    assert after_data["writers_room_transcript"] == []


def test_character_deduplication_and_portrait_serving(tmp_path: Path):
    """Verify consolidate_character_list merges character revisions with differing character_ids or
    expanded names while keeping distinct characters separate, and /api/assets/sprites/ serves assets/portraits/.
    """
    from engine.studio.models import CharacterSheet
    from engine.studio.writers_room import consolidate_character_list

    c1 = CharacterSheet(
        deliberation="Initial contact character sheet.",
        character_id="npc_navigator_kestrel",
        name="Navigator Kestrel",
        role="companion",
        backstory="First draft backstory.",
        portrait_asset="assets/portraits/navigator_kestrel.png",
    )
    c2 = CharacterSheet(
        deliberation="Expanded full name on subsequent turn.",
        character_id="navigator_kestrel",
        name="Navigator Lyra Kestrel",
        role="companion",
        backstory="Expanded backstory with starlight cartography.",
        portrait_asset="assets/portraits/navigator_kestrel.png",
    )
    c3 = CharacterSheet(
        deliberation="Third refinement with full name in ID.",
        character_id="navigator_lyra_kestrel",
        name="Navigator Lyra Kestrel",
        role="companion",
        backstory="Final backstory with voice cadence.",
        portrait_asset="assets/portraits/navigator_kestrel.png",
    )
    distinct_rival = CharacterSheet(
        deliberation="Distinct rival character.",
        character_id="archivist_thorne",
        name="Archivist Thorne",
        role="rival",
        backstory="Keeper of the brass astrolabe.",
        portrait_asset="assets/portraits/archivist_thorne.png",
    )

    merged, alias_map = consolidate_character_list([c1], [c2, c3, distinct_rival])
    assert len(merged) == 2
    assert merged[0].character_id == "npc_navigator_kestrel"
    assert merged[0].name == "Navigator Lyra Kestrel"
    assert merged[0].backstory == "Final backstory with voice cadence."
    assert merged[1].character_id == "archivist_thorne"
    assert alias_map["navigator_kestrel"] == "npc_navigator_kestrel"
    assert alias_map["navigator_lyra_kestrel"] == "npc_navigator_kestrel"


def test_game_blueprint_dashboard_synchronizes_with_latest_writers_room_bible(tmp_path: Path):
    """Verify GET /api/telemetry/dashboard (Game Blueprint panel) and GET /api/studio/game-bible
    always reflect the latest Writers' Room Living Game Bible in TelemetryDB rather than a stale
    active cartridge.
    """
    db_path = tmp_path / "telemetry.db"
    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    active_dir.mkdir(parents=True, exist_ok=True)

    # Seed an older active cartridge on disk
    (active_dir / "world_spec.json").write_text(
        json.dumps(
            {
                "deliberation": "Old cartridge world spec.",
                "game_title": "Legacy Cartridge Title",
                "core_genre": "Legacy Genre",
                "thematic_premise": "Old premise from previous cartridge.",
                "world_lore": "Old lore.",
                "aesthetic_style": "Old style",
                "palette_mood": "dark_gritty",
                "mechanics_guidelines": ["Old rule 1"],
            }
        ),
        encoding="utf-8",
    )

    tdb = TelemetryDB(db_path)
    bus = SwarmEventBus(db_path=db_path, log_path=tmp_path / "events.jsonl")
    bus.reset_for_new_genesis_build(build_id="build_genesis_new", creative_prompt="New Writers Room pitch")

    # Simulate Writers' Room locking a new Game Bible into candidate_genesis_new
    cand_dir = cartridges_root / "candidate_genesis_new"
    fake_client = _FakeGeminiClient()
    visual_agent = VisualArtisanAgent(client=fake_client)
    room = CreativeWritersRoom(
        client=fake_client,
        visual_agent=visual_agent,
        telemetry_db=tdb,
        max_turns=3,
        min_turns_before_convergence=2,
    )
    room.run_writers_room(
        dna=_sample_dna(),
        candidate_dir=cand_dir,
        cartridge_id="candidate_genesis_new",
    )

    app = create_app(
        db_path=db_path,
        cartridges_root=cartridges_root,
        active_cartridge_dir=active_dir,
        event_bus=bus,
    )
    client = TestClient(app)

    bible_res = client.get("/api/studio/game-bible")
    assert bible_res.status_code == 200
    assert bible_res.json()["game_title"] == "Chrono Canopy"

    dash_res = client.get("/api/telemetry/dashboard")
    assert dash_res.status_code == 200
    dash_data = dash_res.json()
    assert dash_data["game_title"] == "Chrono Canopy"
    assert dash_data["genre"] == "Atmospheric Momentum Glider"
    cd = dash_data["manifest"]["creative_direction"]
    assert "canopy glider" in cd["narrative_premise"].lower()


def test_gameplay_systems_engineer_in_writers_room_and_downstream_pipeline(tmp_path: Path):
    """Verify GameplaySystemsEngineer is in SPECIALIST_ROSTER, MechanicEngineeringSpec enforces Rule 4,
    and technical_architecture_blueprint flows into CartridgeAssetLibrary, Director, and LeadGameplaySWEAgent.
    """
    from engine.studio.asset_library import CartridgeAssetLibrary
    from engine.studio.models import MechanicEngineeringSpec
    from engine.studio.writers_room import SPECIALIST_ROSTER

    roles = [s["role"] for s in SPECIALIST_ROSTER]
    assert "GameplaySystemsEngineer" in roles

    espec = MechanicEngineeringSpec(
        deliberation="Translate canopy thermal updrafts into Area2D wind volumes and CharacterBody2D velocity impulses.",
        mechanic_id="thermal_updraft_flight",
        title="Thermal Updraft Flight & Chime Resonance",
        target_gd_modules=["mechanics.gd", "scenes.gd", "entities.gd"],
        godot_node_and_collider_architecture=(
            "CharacterBody2D player with StaticBody2D cliff boundaries and Area2D thermal columns."
        ),
        state_machine_and_input_contract=(
            "Starts UNSOLVED at t=0 (chime_resonated=false); diving through Area2D thermal column sets velocity.y < -280."
        ),
        required_sprite_and_tile_assets=[
            "wind_canyon_stone_tile:tile:Weathered cyan-veined canyon cliff stone surface",
            "resonance_chime_pylon:prop:Ancient bronze wind-chime pylon with glowing core",
            "thermal_updraft_plume:sprite_sheet:Swirling bioluminescent wind current plume",
        ],
        falsifiable_test_invariants=[
            "unsolved_baseline: chime_resonated == false at t=0 during passive ticks",
            "spatial_boundary: StaticBody2D canyon floor blocks descent below y=680",
            "positive_causal: entering thermal Area2D transitions chime_resonated to true",
        ],
    )
    assert list(espec.model_dump().keys())[0] == "deliberation"

    fake_client = _FakeGeminiClient()
    room = CreativeWritersRoom(client=fake_client, max_turns=3, min_turns_before_convergence=2)
    bible = room.run_writers_room(dna=_sample_dna(), candidate_dir=tmp_path, cartridge_id="test_swe_room")

    # Apply a turn proposal from GameplaySystemsEngineer
    swe_proposal = WritersRoomTurnProposal(
        deliberation="Engineering consultation locking Godot 4 node hierarchy, assets, and test invariants.",
        spoken_dialogue="I've mapped the canopy dive and chime resonance into Area2D thermal volumes and StaticBody2D cliff colliders.",
        addressed_to=["GameplayDesigner", "VisualArtDirector"],
        added_or_updated_engineering_specs=[espec],
    )
    room._apply_turn_proposal(
        game_bible=bible,
        speaker_role="GameplaySystemsEngineer",
        speaker_display_name="Lead Gameplay & Godot Systems Engineer",
        proposal=swe_proposal,
        candidate_dir=tmp_path,
    )

    assert len(bible.technical_architecture_blueprint) == 1
    assert bible.technical_architecture_blueprint[0].mechanic_id == "thermal_updraft_flight"

    # Verify CartridgeAssetLibrary extracts the SWE's required_sprite_and_tile_assets
    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path)
    derived_assets = lib.derive_baseline_assets_from_bible(bible)
    derived_ids = {a.asset_id for a in derived_assets}
    assert "wind_canyon_stone_tile" in derived_ids
    assert "resonance_chime_pylon" in derived_ids
    assert "thermal_updraft_plume" in derived_ids

    # Verify LeadGameplaySWEAgent includes technical_architecture_blueprint in prompt context
    swe = LeadGameplaySWEAgent(client=fake_client)
    ctx_str, _ = swe._build_bible_and_narrative_context(staging_dir=tmp_path, game_bible=bible)
    assert "Technical Architecture Blueprint (GameplaySystemsEngineer" in ctx_str
    assert "thermal_updraft_flight" in ctx_str
    assert "CharacterBody2D player with StaticBody2D cliff boundaries" in ctx_str







# ---------------------------------------------------------------------------
# Concept genesis: the idea emerges and evolves instead of being pitched up front
# ---------------------------------------------------------------------------
def _make_room(tmp_path: Path, **kw):
    fake_client = _FakeGeminiClient()
    room = CreativeWritersRoom(
        client=fake_client, visual_agent=VisualArtisanAgent(client=fake_client),
        max_turns=2, min_turns_before_convergence=1, **kw,
    )
    return room, fake_client


def test_concept_genesis_models_rationale_first():
    from engine.studio.models import (
        ConceptCandidate, ConceptCollisionProposal, ConceptFragment, ConceptSeedProposal,
        ConceptSelection, CreativeProvocationSet, NoveltyJudgement,
    )
    for cls in (ConceptCandidate, ConceptCollisionProposal, ConceptFragment, ConceptSeedProposal,
                ConceptSelection, CreativeProvocationSet, NoveltyJudgement):
        assert list(cls.model_fields)[0] == "deliberation", cls.__name__
    assert list(NoveltyJudgement.model_fields)[-1] == "is_distinct_from_prior_concepts"
    assert list(ConceptSelection.model_fields)[-1] == "selected_candidate_id"


def test_room_opens_with_genesis_not_a_pitch(tmp_path: Path):
    from engine.studio.writers_room import SPECIALIST_ROSTER
    room, client = _make_room(tmp_path)
    candidate_dir = tmp_path / "cartridges" / "candidate_new"
    bible = room.run_writers_room(dna=_sample_dna(), candidate_dir=candidate_dir)
    expected_fragments = 2 * len(SPECIALIST_ROSTER)
    assert len(bible.concept_fragments) == expected_fragments      # 2 fragments x 9 specialists (incl. ScriptwriterAgent)
    assert {f.contributor_role for f in bible.concept_fragments} >= {"GameplayDesigner", "StoryWorldArchitect", "ScriptwriterAgent"}
    assert len({f.fragment_id for f in bible.concept_fragments}) == expected_fragments
    assert len(bible.concept_candidates) == 4
    assert bible.writers_room_transcript[0].speaker_role == "CreativeDirector"
    assert "draft premise" in bible.writers_room_transcript[0].spoken_dialogue
    # the selected candidate (c1) seeded the room, via the genesis path rather than a one-shot pitch
    first_title_update = bible.writers_room_transcript[0].bible_updates_summary
    assert any("Updated title" in u for u in first_title_update)
    assert client.models.calls[0]["schema"].__name__ == "CreativeProvocationSet"


def test_novelty_judge_filters_and_regenerates_when_all_overlap(tmp_path: Path):
    cartridges = tmp_path / "cartridges"
    prior_dir = cartridges / "candidate_prior"
    prior_dir.mkdir(parents=True)
    (prior_dir / "game_bible.json").write_text(json.dumps({
        "game_title": "Old Game", "elevator_pitch": "An older pitch about something.",
        "gameplay_pillars": ["p1"], "camera_and_spatial_paradigm": "side view",
    }))
    room, client = _make_room(tmp_path)
    client.models.novelty_reject_first_n = 4          # first collision round entirely rejected as overlapping
    bible = room.run_writers_room(dna=_sample_dna(), candidate_dir=cartridges / "candidate_new")
    assert client.models.collision_call_count == 2
    assert client.models.novelty_call_count == 8
    assert bible.concept_candidates[0].working_title.startswith("Candidate 2.")
    # the prior concept digest was handed to the room
    assert "Old Game" in room._load_prior_concepts(cartridges / "candidate_new")


def test_genesis_raises_when_nothing_is_distinct(tmp_path: Path):
    from engine.studio.models import StudioAgentError
    cartridges = tmp_path / "cartridges"
    (cartridges / "candidate_prior").mkdir(parents=True)
    (cartridges / "candidate_prior" / "game_bible.json").write_text(json.dumps(
        {"game_title": "Old Game", "elevator_pitch": "pitch"}))
    room, client = _make_room(tmp_path)
    client.models.novelty_reject_first_n = 10_000
    with pytest.raises(StudioAgentError):
        room.run_writers_room(dna=_sample_dna(), candidate_dir=cartridges / "candidate_new")


def test_specialist_prompt_marks_concept_as_provisional_and_dna_not_a_template(tmp_path: Path):
    room, client = _make_room(tmp_path)
    assert room.creative_temperature == pytest.approx(0.7)
    prompts = []
    temps = []
    orig = room._call_structured_llm
    room._call_structured_llm = lambda prompt, schema_cls, temperature=None, parts=None: (
        prompts.append(prompt),
        temps.append((schema_cls.__name__, room.creative_temperature if temperature is None else float(temperature))),
        orig(prompt, schema_cls, temperature, parts),
    )[2]
    room.run_writers_room(dna=_sample_dna(), candidate_dir=tmp_path / "cartridges" / "candidate_new")
    assert any("THE CONCEPT IS STILL EVOLVING" in p for p in prompts)
    assert any("CLARITY & CONCRETE GAME DESIGN" in p for p in prompts)
    assert all(t <= 0.75 for _, t in temps)
    assert any(name == "WritersRoomListenCheck" and t == pytest.approx(0.4) for name, t in temps)
    assert any(name == "DirectorConvergenceCheck" and t == pytest.approx(0.4) for name, t in temps)
    summary = room._format_dna_summary(_sample_dna())
    assert "Session Creative Exploration Lens" not in summary


def test_independent_room_calls_run_concurrently_and_keep_order(tmp_path: Path):
    """Listen checks, concept seeds and novelty judgements are independent and must overlap in time."""
    import threading
    import time as _time

    room, client = _make_room(tmp_path, max_parallel_calls=7)
    state = {"active": 0, "peak": 0}
    lock = threading.Lock()
    orig = client.models.generate_content

    def slow(model, contents, config=None):
        name = getattr(getattr(config, "response_schema", None), "__name__", "")
        if name in ("ConceptSeedProposal", "WritersRoomListenCheck"):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            _time.sleep(0.05)
            with lock:
                state["active"] -= 1
        return orig(model, contents, config)

    client.models.generate_content = slow
    bible = room.run_writers_room(dna=_sample_dna(), candidate_dir=tmp_path / "cartridges" / "candidate_new")
    assert state["peak"] >= 4                                   # genuinely concurrent
    roles = [f.contributor_role for f in bible.concept_fragments]
    from engine.studio.writers_room import SPECIALIST_ROSTER
    assert roles == [s_["role"] for s_ in SPECIALIST_ROSTER for _ in range(2)]   # deterministic order preserved


# ---------------------------------------------------------------------------
# Game-design sections: levels, loop, engagement, onboarding, goals
# ---------------------------------------------------------------------------
def _complete_design_bible() -> GameBibleSpec:
    bible = GameBibleSpec(deliberation="d")
    bible.level_generation_strategy = LevelGenerationStrategy(
        deliberation="d", generation_approach="procedural", world_structure="ring of regions",
        level_archetypes=[LevelArchetypeSpec(deliberation="d", archetype_id="a", purpose="teach", visual_identity="v", signature_challenge="c")],
        generation_rules=["always solvable"], difficulty_and_pacing_curve="ramp")
    bible.core_loop = CoreLoopSpec(deliberation="d", player_verbs=["move"], moment_to_moment_loop="m", minute_to_minute_loop="n", session_loop="s")
    bible.engagement_plan = EngagementPlan(deliberation="d", entertainment_pillars=["p"], pacing_and_intensity_curve="c")
    bible.onboarding_plan = OnboardingPlan(
        deliberation="d", world_introduction="w",
        steps=[OnboardingStep(deliberation="d", step_index=i, concept_introduced=f"c{i}", diegetic_method="m") for i in range(3)])
    bible.goal_hierarchy = GoalHierarchy(deliberation="d", goals=[
        GoalSpec(deliberation="d", goal_id="s", horizon="short_term", description="x"),
        GoalSpec(deliberation="d", goal_id="l", horizon="long_term", description="y")])
    return bible


def test_game_design_models_rationale_first_and_no_free_form_maps():
    from engine.studio.models import GameBibleSpec as GB
    for cls in (LevelGenerationStrategy, LevelArchetypeSpec, CoreLoopSpec, EngagementPlan, OnboardingPlan,
                OnboardingStep, GoalHierarchy, GoalSpec):
        assert list(cls.model_fields)[0] == "deliberation", cls.__name__
    assert "additionalProperties" not in json.dumps(WritersRoomTurnProposal.model_json_schema())
    assert "additionalProperties" not in json.dumps(GB.model_json_schema())


def test_validate_game_design_flags_every_gap_and_accepts_complete():
    from engine.studio.models import validate_game_design
    assert validate_game_design(_complete_design_bible()) == []
    issues = validate_game_design(GameBibleSpec(deliberation="d"))
    for section in ("level_generation_strategy", "core_loop", "engagement_plan", "onboarding_plan", "goal_hierarchy"):
        assert any(i.startswith(section) for i in issues), section
    b = _complete_design_bible()
    b.goal_hierarchy.goals = [b.goal_hierarchy.goals[0]]
    b.onboarding_plan.steps = b.onboarding_plan.steps[:1]
    issues = validate_game_design(b)
    assert any("long_term" in i for i in issues) and any("at least 3 steps" in i for i in issues)


def test_room_has_level_progression_specialist_and_locks_with_complete_design(tmp_path: Path):
    from engine.studio.models import validate_game_design
    from engine.studio.writers_room import SPECIALIST_ROSTER
    assert any(s_["role"] == "LevelAndProgressionDesigner" for s_ in SPECIALIST_ROSTER)
    room, _ = _make_room(tmp_path)
    room.max_turns, room.min_turns_before_convergence = 3, 2
    bible = room.run_writers_room(dna=_sample_dna(), candidate_dir=tmp_path / "cartridges" / "candidate_new")
    assert bible.status == "locked" and validate_game_design(bible) == []


def test_forced_completion_turns_fill_missing_sections_by_owner(tmp_path: Path):
    room, _ = _make_room(tmp_path)
    bible = GameBibleSpec(deliberation="d")
    called = []
    complete = _complete_design_bible()

    def fake_turn(specialist, listen_check, game_bible, dna_summary):
        called.append(specialist["role"])
        return WritersRoomTurnProposal(
            deliberation="d", spoken_dialogue="filling the gap",
            updated_level_generation_strategy=complete.level_generation_strategy,
            updated_goal_hierarchy=complete.goal_hierarchy, updated_core_loop=complete.core_loop,
            updated_engagement_plan=complete.engagement_plan, updated_onboarding_plan=complete.onboarding_plan)

    room._execute_specialist_turn = fake_turn
    room._complete_missing_design_sections(bible, "dna", None)
    assert set(called) == {"LevelAndProgressionDesigner", "GameplayDesigner", "UXExperienceDesigner"}
    assert bible.core_loop is not None and bible.goal_hierarchy is not None


def test_room_refuses_to_lock_when_design_cannot_be_completed(tmp_path: Path):
    from engine.studio.models import StudioAgentError
    room, _ = _make_room(tmp_path)
    room._execute_specialist_turn = lambda **k: WritersRoomTurnProposal(deliberation="d", spoken_dialogue="nothing")
    with pytest.raises(StudioAgentError):
        room._complete_missing_design_sections(GameBibleSpec(deliberation="d"), "dna", None)


def test_swe_context_and_beat_plan_consume_design_sections(tmp_path: Path):
    from types import SimpleNamespace
    agent = LeadGameplaySWEAgent.__new__(LeadGameplaySWEAgent)
    agent.client = None
    bible = _complete_design_bible()
    ctx, _ = agent._build_bible_and_narrative_context(tmp_path, game_bible=bible)
    assert "GAME DESIGN BLUEPRINT" in ctx and "always solvable" in ctx
    assert "LEVEL GENERATION STRATEGY" in ctx and "GOAL HIERARCHY" in ctx and "ONBOARDING PLAN" in ctx and "CORE LOOP" in ctx


def test_beat_plan_is_generated_from_onboarding_steps_even_without_storyboard(tmp_path: Path):
    from engine.studio.models import BeatPlanSpec
    from types import SimpleNamespace
    bible = _complete_design_bible()
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json())
    agent = LeadGameplaySWEAgent.__new__(LeadGameplaySWEAgent)
    agent._is_live_genai_client = lambda: True
    seen = {}

    class _Models:
        def generate_content(self, model, contents, config=None):
            seen["prompt"] = contents
            return SimpleNamespace(text=BeatPlanSpec(deliberation="d", beats=[]).model_dump_json())

    agent.client = SimpleNamespace(models=_Models())
    agent.model_name = "m"
    layout = SimpleNamespace(model_dump_json=lambda indent=2: "{}")
    plan = agent.formulate_beat_plan(tmp_path, layout)
    assert plan is not None
    assert "ONBOARDING PLAN" in seen["prompt"] and "c0" in seen["prompt"] and "GOALS" in seen["prompt"]


# ---------------------------------------------------------------------------
# Hierarchical Council, ScriptwriterAgent & Real-Time Live Agent Observability
# ---------------------------------------------------------------------------
def test_hierarchical_council_roster_and_subagent_handoff(tmp_path: Path):
    from engine.studio.writers_room import (
        CORE_COUNCIL_ROSTER,
        COUNCIL_HIERARCHY,
        SPECIALIST_ROSTER,
        SUBAGENT_ROSTER,
    )

    lead_roles = [s["role"] for s in CORE_COUNCIL_ROSTER]
    sub_roles = [s["role"] for s in SUBAGENT_ROSTER]
    assert lead_roles == ["GameplayDesigner", "StoryWorldArchitect", "VisualArtDirector"]
    assert "ScriptwriterAgent" in sub_roles
    assert "ScriptwriterAgent" in COUNCIL_HIERARCHY["StoryWorldArchitect"]
    assert len(SPECIALIST_ROSTER) == 9

    events = []
    room, _ = _make_room(tmp_path, event_callback=lambda ev_type, payload: events.append({"event_type": ev_type, "payload": payload}))
    bible = GameBibleSpec(deliberation="d", game_title="Echoes of the Spindle")

    def fake_eval_listen(specialist, game_bible, dna_summary):
        role = specialist["role"]
        if role == "StoryWorldArchitect":
            return WritersRoomListenCheck(
                deliberation="We need sharp dramatic dialogue and scene staging for the Spindle awakening.",
                should_interject=True,
                urgency_score=0.92,
                intended_contribution_summary="Establish Spindle lore and call ScriptwriterAgent for opening scene.",
                consult_subagent_role="ScriptwriterAgent",
                subagent_brief="Write the opening dramatic confrontation and dialogue tree at the Spindle gate.",
            )
        return WritersRoomListenCheck(
            deliberation="Holding back.",
            should_interject=False,
            urgency_score=0.1,
            intended_contribution_summary="",
        )

    room._evaluate_specialist_listen = fake_eval_listen
    chosen_spec, chosen_check = room._select_next_speaker(
        last_speaker_role="CreativeDirector",
        addressed_to=[],
        game_bible=bible,
        dna_summary="dna",
        spoken_counts={},
    )
    assert chosen_spec is not None
    assert chosen_spec["role"] == "ScriptwriterAgent"
    assert "[Called in by StoryWorldArchitect]" in chosen_check.intended_contribution_summary
    assert any(
        e.get("event_type") == "writers_room_activity"
        and e.get("payload", {}).get("phase") == "subagent_handoff"
        and e.get("payload", {}).get("subagent_role") == "ScriptwriterAgent"
        and e.get("payload", {}).get("called_by_lead") == "StoryWorldArchitect"
        for e in events
    )


def test_script_writer_worker_specialist_and_proof_plate(tmp_path: Path):
    from types import SimpleNamespace
    from engine.studio.models import (
        CutsceneBeat,
        CutsceneScript,
        DialogueBarkSpec,
        DialogueNode,
        DialogueTreeSpec,
        ScriptWritingDeliverable,
        WorkItem,
    )
    from engine.studio.workers import WorkerContext, available_specialists, run_script_writer_specialist

    deliverable = ScriptWritingDeliverable(
        deliberation="Crafting subtext-rich opening cutscene, branching dialogue tree, and reactive barks.",
        narrative_notes="Authored 'The Spindle Fracture' cutscene, 1 branching dialogue tree, and 1 reactive bark.",
        added_or_updated_cutscenes=[
            CutsceneScript(
                deliberation="Opening fracture scene.",
                cutscene_id="spindle_fracture",
                title="The Spindle Fracture",
                trigger="on_level_start",
                beats=[
                    CutsceneBeat(
                        deliberation="Vesper warns the player.",
                        beat_number=1,
                        action_description="Sparks cascade down the iron loom.",
                        speaker_id="vesper",
                        dialogue_line="Hold the thread! If the spindle snaps, the whole ward goes dark.",
                    )
                ],
            )
        ],
        added_or_updated_dialogue_trees=[
            DialogueTreeSpec(
                deliberation="Branching conversation with Vesper.",
                root_node_id="vesper_intro",
                nodes={
                    "vesper_intro": DialogueNode(
                        deliberation="Root greeting.",
                        node_id="vesper_intro",
                        speaker_id="vesper",
                        text="You brought the tuning fork?",
                        choices=[],
                    )
                },
            )
        ],
        added_or_updated_barks=[
            DialogueBarkSpec(
                deliberation="Low stamina reactive bark.",
                bark_id="bark_thread_fray",
                speaker_id="vesper",
                speaker_name="Vesper",
                trigger_type="on_thread_fray",
                text="The tension's too high—slacken the line!",
                voice_clip="assets/audio/bark_thread_fray.wav",
            )
        ],
    )

    class _ScriptFakeModels:
        def generate_content(self, model, contents, config=None):
            return SimpleNamespace(text=deliverable.model_dump_json())

    class _ScriptFakeClient:
        __module__ = "google.genai.fake"

        def __init__(self):
            self.models = _ScriptFakeModels()

    fake_client = _ScriptFakeClient()
    bible = GameBibleSpec(deliberation="d", game_title="Echoes of the Spindle")
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")
    item = WorkItem(
        deliberation="d",
        item_id="narrative_spindle_script",
        title="Spindle Opening Script & Barks",
        kind="narrative",
        spec="Write the opening cutscene, dialogue tree, and reactive barks.",
        proof_spec="Visual script sheet proof plate.",
        raised_by_role="ScriptwriterAgent",
        owner_role="ScriptwriterAgent",
    )
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=fake_client,
    )
    assert "script_writer" in available_specialists(ctx, item)
    report = run_script_writer_specialist(ctx, item, "Write the opening Spindle script and barks.", turn_index=1)
    assert report.specialist_agent == "script_writer"
    assert any("script_sheet_narrative_spindle_script_t1.png" in p for p in report.proof_images)
    assert (tmp_path / "narrative.json").exists()
    persisted_bible = GameBibleSpec.model_validate_json((tmp_path / "game_bible.json").read_text(encoding="utf-8"))
    assert len(persisted_bible.narrative_bible.cutscenes) == 1
    assert len(persisted_bible.narrative_bible.dialogue_barks) == 1


def test_live_agent_observability_events_emitted(tmp_path: Path):
    events = []
    room, _ = _make_room(tmp_path, event_callback=lambda ev_type, payload: events.append({"event_type": ev_type, "payload": payload}))
    candidate_dir = tmp_path / "cartridges" / "candidate_obs"
    room.run_writers_room(dna=_sample_dna(), candidate_dir=candidate_dir)
    activity_events = [e for e in events if e.get("event_type") == "writers_room_activity"]
    phases = {e.get("payload", {}).get("phase") for e in activity_events}
    assert {"genesis_provocations", "genesis_fragments", "genesis_collision", "genesis_selection", "listening", "speaking", "convergence_check"} <= phases
    assert all(isinstance(e.get("payload", {}).get("active_agents"), list) for e in activity_events)


def test_council_shaping_and_review_prompts_enforce_systemic_invariants_over_coordinate_patches(tmp_path: Path):
    from engine.studio.models import (
        ReviewBatch,
        ReviewDecision,
        TaskShapingBatch,
        WorkItemProposal,
        WorkItemRefinement,
    )
    from engine.studio.production_board import ProductionBoard

    room, _ = _make_room(tmp_path)
    board = ProductionBoard(tmp_path / "backlog.json")
    added = board.add_items(
        [
            WorkItemProposal(
                deliberation="d",
                item_id="mechanic_placement_sync",
                title="Synchronize Placement Tool With World Geometry",
                kind="mechanic_code",
                spec="Allow placing structures across valid terrain polygons.",
                proof_spec="Verify placement across multiple regions.",
                target_modules=["scenes/mechanics/placement_tool.gd"],
            )
        ],
        raised_by_role="GameplayDesigner",
        slice_id="slice_0",
    )
    room.board = board

    # 1. Role review mandate teaches systemic spatial contracts over coordinate whitelisting
    mandate = room._role_review_mandate("LevelAndProgressionDesigner")
    assert "never instruct workers to whitelist a specific `(x, y)` coordinate" in mandate

    # 2. Stage 2 Council task-shaping prompt instructs specialists to specify generalizable world-wide invariants
    shaping_prompts = []
    room._structured_with_validation = lambda prompt, schema_cls, validator, temperature=None, parts=None, **kwargs: (
        shaping_prompts.append(prompt),
        TaskShapingBatch(
            deliberation="d",
            backlog_refinements=[
                WorkItemRefinement(
                    deliberation="d",
                    item_id="mechanic_placement_sync",
                    action="no_input",
                )
            ],
        ),
    )[1]
    room._run_full_council_direction_pass(
        game_bible=_complete_design_bible(),
        dna_summary="dna",
        bible_snapshot="bible",
        candidate_dir=tmp_path,
    )
    assert shaping_prompts
    assert all("Specify generalizable, world-wide invariants" in p for p in shaping_prompts)
    assert all("never prescribe hardcoded `(x, y)` coordinate whitelists" in p for p in shaping_prompts)

    # 3. Council proof review prompt enforces systemic root-cause diagnosis over coordinate point-patches
    review_prompts = []
    room._structured_with_validation = lambda prompt, schema_cls, validator, temperature=None, parts=None, **kwargs: (
        review_prompts.append(prompt),
        ReviewBatch(
            deliberation="d",
            spoken_dialogue="Checking systemic spatial rules.",
            review_decisions=[
                ReviewDecision(
                    deliberation="d",
                    validation_criteria_for_my_role="Placement works across all valid world regions.",
                    item_id="mechanic_placement_sync",
                    verdict="approve",
                )
            ],
        ),
    )[1]
    room._review_as(
        role="UXExperienceDesigner",
        awaiting=added,
        deciding=[],
        game_bible=_complete_design_bible(),
        dna_summary="dna",
        candidate_dir=tmp_path,
    )
    assert len(review_prompts) == 1
    assert "SYSTEMIC INVARIANTS OVER COORDINATE POINT-PATCHES" in review_prompts[0]


