"""Tests for GenreCraftFoundation synthesis, Co-Producer integration, Game Bible genesis seeding,
and multi-turn / slice pillar evolution (Rules 1, 2, 4, 5)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from engine.studio.context_curator import (
    CartridgeSemanticIndex,
    ContextCuratorAgent,
    format_genre_craft_foundation_briefing,
)
from engine.studio.coproducer import CoProducerAgent
from engine.studio.models import (
    CoProducerActionProposal,
    CoProducerConceptBrief,
    CoProducerFinalTurn,
    CoProducerStepDecision,
    CoProducerToolCall,
    GameBibleSpec,
    GenesisWorldSpec,
    GenreCraftFoundation,
    GenrePillarSpec,
    StudioAgentConfigurationError,
    WritersRoomTurnProposal,
)
from engine.studio.writers_room import SPECIALIST_ROSTER, CreativeWritersRoom


def _sample_foundation(
    archetype: str = "Tactile Celestial Navigation & Sail-Trimming Voyage",
    extra_pillar_title: str = "",
) -> GenreCraftFoundation:
    pillars = [
        GenrePillarSpec(
            deliberation="Manual sextant sightings and sail trim give players tangible agency over wind and currents.",
            pillar_id="manual_celestial_reckoning",
            title="Manual Celestial Reckoning",
            why_it_creates_fun="Reading star angles and currents under shifting weather makes every plotted course feel earned.",
            systemic_mechanics_expression="Player aligns optical sightings and adjusts rudder/sail tension against live wind vectors.",
            anti_pattern_to_avoid="Never auto-pilot the vessel to waypoints or reveal full fog-of-war charts at t=0.",
        ),
        GenrePillarSpec(
            deliberation="Provision scarcity forces meaningful risk-reward route choices between safe shallows and fast gales.",
            pillar_id="provision_and_hull_tradeoffs",
            title="Provision & Hull Trade-Offs",
            why_it_creates_fun="Scarce fresh water and sail durability turn every storm detour into a tense strategic dilemma.",
            systemic_mechanics_expression="Starts with minimal rations at t=0; players chart landings to forage and repair.",
            anti_pattern_to_avoid="Never grant infinite hull/provisions or passive resource regeneration at t=0.",
        ),
        GenrePillarSpec(
            deliberation="Exact spatial chart plotting and clear wind silhouettes ensure player clicks map 1:1 to course headings.",
            pillar_id="crisp_chart_and_wind_legibility",
            title="Crisp Chart & Wind Legibility",
            why_it_creates_fun="Players can read gusts, shoals, and heading vectors at a glance and steer with precision.",
            systemic_mechanics_expression="1:1 world-coordinate heading markers, distinct shoal/current silhouettes, and immediate wake feedback.",
            anti_pattern_to_avoid="Never snap headings to arbitrary points away from the cursor or obscure hazards in muddy noise.",
        ),
    ]
    if extra_pillar_title:
        pillars.append(
            GenrePillarSpec(
                deliberation="Evolved mid-voyage harmonic beacon triangulation deepens multi-island routing.",
                pillar_id="beacon_triangulation_network",
                title=extra_pillar_title,
                why_it_creates_fun="Cross-referencing player-lit coastal beacons rewards earlier exploration with faster night passages.",
                systemic_mechanics_expression="Beacons remain dark until manually reached and fueled by the player.",
                anti_pattern_to_avoid="Never pre-light coastal beacons at t=0.",
            )
        )
    return GenreCraftFoundation(
        deliberation="Deconstructing why celestial voyage games are engaging: unsolved sea charts, tactile trim, and earned range.",
        genre_and_subgenre_archetype=archetype,
        core_fun_thesis=(
            "Curiosity pulls the player into uncharted waters while wind physics and scarce supplies demand skillful "
            "navigation, rewarding mastery with newly charted sea lanes."
        ),
        bootstrap_to_mastery_arc=(
            "At t=0 the player commands a bare skiff in unknown waters with zero pre-lit beacons or free supplies; "
            "each charted island and sail upgrade is earned through manual voyages."
        ),
        essential_spatial_and_ux_contracts=[
            "Course markers place at exact clicked world coordinates with live heading preview lines.",
            "Wind streaks, shoals, and collectible driftwood use unmistakable high-contrast silhouettes.",
        ],
        genre_killing_anti_patterns=[
            "Auto-sailing to objectives without player steering or sail trim.",
            "Granting free starting supplies or pre-charted maps at t=0.",
        ],
        foundational_pillars=pillars,
    )


class _QueuedMockModels:
    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self._responses:
            raise AssertionError("Mock Gemini queue exhausted unexpectedly.")
        text = self._responses.pop(0)
        return SimpleNamespace(text=text, usage_metadata=None)


class _QueuedMockClient:
    def __init__(self, responses: List[str]) -> None:
        self.models = _QueuedMockModels(responses)


def test_genre_craft_foundation_rule4_ordering_and_role_schema_budgets() -> None:
    assert list(GenrePillarSpec.model_fields.keys())[0] == "deliberation"
    assert list(GenreCraftFoundation.model_fields.keys())[0] == "deliberation"

    expected_with_foundation = {
        "CoProducer",
        "CreativeDirector",
        "GameplayDesigner",
        "LevelAndProgressionDesigner",
        "GameplaySystemsEngineer",
        "UXExperienceDesigner",
    }
    for role in ["CreativeDirector"] + [sp["role"] for sp in SPECIALIST_ROSTER]:
        scoped_cls = WritersRoomTurnProposal.for_role(role)
        assert list(scoped_cls.model_fields.keys())[0] == "deliberation"
        if role in expected_with_foundation:
            assert "updated_genre_craft_foundation" in scoped_cls.model_fields
        else:
            assert "updated_genre_craft_foundation" not in scoped_cls.model_fields
        js = scoped_cls.model_json_schema()
        defs = js.get("$defs", {})
        total_props = len(js.get("properties", {})) + sum(len(v.get("properties", {})) for v in defs.values())
        assert len(defs) <= 12
        assert total_props <= 95


def test_context_curator_synthesizes_persists_and_indexes_genre_foundation(tmp_path: Path) -> None:
    cart_dir = tmp_path / "candidate_voyage"
    cart_dir.mkdir(parents=True)

    # Rule 5 fail-fast when client is missing
    curator_no_client = ContextCuratorAgent(cartridge_dir=cart_dir, client=None)
    with pytest.raises(StudioAgentConfigurationError):
        curator_no_client.synthesize_genre_craft_foundation(genre_hint="Celestial navigation voyage")

    sample = _sample_foundation()
    events: List[tuple[str, Dict[str, Any]]] = []
    mock_client = _QueuedMockClient([sample.model_dump_json()])
    curator = ContextCuratorAgent(
        cartridge_dir=cart_dir,
        client=mock_client,
        event_callback=lambda et, payload: events.append((et, payload)),
    )
    result = curator.synthesize_genre_craft_foundation(
        genre_hint="Celestial navigation voyage",
        concept_or_seed="Charting storm-swept archipelagos by starlight.",
        player_dna_summary="Enjoys systemic exploration and spatial mastery.",
    )
    assert result.genre_and_subgenre_archetype == sample.genre_and_subgenre_archetype
    assert len(result.foundational_pillars) == 3

    # Persisted to artifacts/genre_craft_foundation.json & emitted reasoning_rationale
    art_file = cart_dir / "artifacts" / "genre_craft_foundation.json"
    assert art_file.is_file()
    synth_events = [p for et, p in events if et == "genre_craft_foundation_synthesized"]
    assert len(synth_events) == 1
    assert synth_events[0]["reasoning_rationale"] == sample.deliberation

    # Indexed as design:genre_craft_foundation even before game_bible.json exists
    sem_idx = CartridgeSemanticIndex(cartridge_dir=cart_dir, client=None)
    pre_bible_chunks = sem_idx.extract_corpus_chunks()
    gcf_chunk = next((c for c in pre_bible_chunks if c.chunk_id == "design:genre_craft_foundation"), None)
    assert gcf_chunk is not None
    assert "Manual Celestial Reckoning" in gcf_chunk.content
    assert "Genre-Killing Anti-Patterns" in gcf_chunk.content


def test_coproducer_curates_genre_foundation_and_hands_off_to_cartridge_brief(tmp_path: Path) -> None:
    cartridges_root = tmp_path / "cartridges"
    cartridges_root.mkdir(parents=True)
    sample = _sample_foundation()

    step_call = CoProducerStepDecision(
        deliberation="Need ContextCurator to synthesize what makes celestial navigation games genuinely fun first.",
        next_step_intent="Synthesizing Genre Craft & Fun Foundation...",
        response_mode="creative_ideation",
        tool_calls=[
            CoProducerToolCall(
                deliberation="Establish foundational fun pillars and anti-patterns for this voyage genre.",
                tool_name="curate_genre_fun_foundation",
                query_or_question="Tactile celestial navigation & archipelago charting",
            )
        ],
        action="call_tools",
    )
    step_done = CoProducerStepDecision(
        deliberation="Genre Craft & Fun Foundation is now ready; respond to the user.",
        next_step_intent="Synthesizing pitch options...",
        response_mode="creative_ideation",
        tool_calls=[],
        action="respond",
    )
    # Deliberately omit genre_craft_foundation in updated_concept_brief to verify CoProducerAgent preserves it!
    final_turn = CoProducerFinalTurn(
        deliberation="Grounding pitch and commissioning proposal in the curated genre foundation.",
        updated_thread_title="Starlight Reckoning",
        updated_concept_brief=CoProducerConceptBrief(
            deliberation="Converged on tactile star-sighting voyage concept.",
            working_title="Starlight Reckoning",
            core_genre="Tactile Celestial Navigation Voyage",
            elevator_pitch="Plot star-sighted courses across uncharted archipelagos under scarce provisions.",
            core_player_verbs=["Sight stars", "Trim sails", "Plot course"],
        ),
        action_proposals=[
            CoProducerActionProposal(
                deliberation="User asked to commission the concept.",
                proposal_id="act_comm_1",
                action_kind="commission_game",
                handoff_mode="refined_concept",
                title="Commission Starlight Reckoning",
                summary="Launch Writers' Room with curated Genre Craft & Fun Foundation.",
                steering_or_seed_brief="Build Starlight Reckoning honoring tactile navigation and scarce provisions.",
            )
        ],
        spoken_response="Here is our **Starlight Reckoning** foundation and commissioning card.",
    )

    mock_client = _QueuedMockClient(
        [
            step_call.model_dump_json(),
            sample.model_dump_json(),
            step_done.model_dump_json(),
            final_turn.model_dump_json(),
        ]
    )
    agent = CoProducerAgent(client=mock_client, cartridges_root=cartridges_root)
    draft_meta = agent.store.create_draft_thread(title="New Game Concept")

    turn_res = agent.run_turn(
        thread_id=draft_meta["thread_id"],
        user_message="Let's design a celestial navigation voyage game that is genuinely fun.",
    )
    saved_brief = turn_res["thread"]["concept_brief"]
    assert saved_brief["genre_craft_foundation"]["genre_and_subgenre_archetype"] == sample.genre_and_subgenre_archetype
    cards = turn_res["assistant_message"]["consultation_cards"]
    assert any(c["consultation_kind"] == "genre_foundation" for c in cards)

    # Graduating draft writes co_producer_brief.json with genre_craft_foundation intact
    cand_dir = cartridges_root / "candidate_starlight"
    agent.store.graduate_draft_to_cartridge(
        draft_thread_id=draft_meta["thread_id"],
        new_cartridge_dir=cand_dir,
        handoff_mode="refined_concept",
        steering_or_seed_brief="Build Starlight Reckoning.",
    )
    brief_on_disk = json.loads((cand_dir / "co_producer_brief.json").read_text(encoding="utf-8"))
    assert (
        brief_on_disk["concept_brief"]["genre_craft_foundation"]["genre_and_subgenre_archetype"]
        == sample.genre_and_subgenre_archetype
    )


def test_writers_room_starts_game_bible_from_genre_foundation_and_evolves_pillars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cand_dir = tmp_path / "candidate_starlight"
    cand_dir.mkdir(parents=True)
    initial_foundation = _sample_foundation()

    # Write Co-Producer handoff brief (works identically for loose_spark or refined_concept)
    (cand_dir / "co_producer_brief.json").write_text(
        json.dumps(
            {
                "deliberation": "Handing off curated Genre Craft & Fun Foundation to Writers' Room.",
                "handoff_mode": "refined_concept",
                "cartridge_id": cand_dir.name,
                "steering_or_seed_brief": "Build Starlight Reckoning.",
                "concept_brief": {
                    "deliberation": "Refined concept brief.",
                    "working_title": "Starlight Reckoning",
                    "core_genre": initial_foundation.genre_and_subgenre_archetype,
                    "elevator_pitch": "Chart storm-tossed islands by manual sextant sightings.",
                    "core_player_verbs": ["Sight stars", "Trim sails"],
                    "genre_craft_foundation": initial_foundation.model_dump(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    evolved_foundation = _sample_foundation(extra_pillar_title="Harmonic Beacon Triangulation")
    evolved_foundation.foundational_pillars[0].why_it_creates_fun = (
        "Cross-wind drift compensation makes manual star sightings a high-skill navigation puzzle."
    )
    specialist_turn = WritersRoomTurnProposal(
        deliberation="Deepening navigation skill expression and adding mid-voyage beacon triangulation.",
        spoken_dialogue="Let's sharpen manual star sightings around cross-wind drift and add player-lit coastal beacons.",
        addressed_to=["LevelAndProgressionDesigner"],
        updated_genre_craft_foundation=evolved_foundation,
    )

    class _SchemaAwareRoomClient:
        def __init__(self) -> None:
            self.specialist_turn_prompts: List[str] = []
            outer = self

            class _Models:
                def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
                    schema = getattr(config, "response_schema", None) if config else None
                    s_name = getattr(schema, "__name__", "")
                    prompt_str = str(contents)
                    if s_name == "WritersRoomListenCheck":
                        is_gd = "(GameplayDesigner)" in prompt_str
                        payload = {
                            "deliberation": "Evaluating whether to evolve genre pillars on Turn 1.",
                            "should_interject": is_gd,
                            "urgency_score": 0.95 if is_gd else 0.1,
                            "intended_contribution_summary": "Sharpen celestial navigation pillars.",
                        }
                        return SimpleNamespace(text=json.dumps(payload), usage_metadata=None)
                    if s_name == "WritersRoomTurnProposal":
                        outer.specialist_turn_prompts.append(prompt_str)
                        return SimpleNamespace(text=specialist_turn.model_dump_json(), usage_metadata=None)
                    if s_name == "GenesisWorldSpec":
                        ws = GenesisWorldSpec(
                            deliberation="Synthesizing GenesisWorldSpec from Starlight Reckoning Game Bible.",
                            game_title="Starlight Reckoning",
                            core_genre=evolved_foundation.genre_and_subgenre_archetype,
                            thematic_premise="Chart storm-tossed islands by manual sextant sightings.",
                            world_lore="An uncharted archipelago navigated by star angles and coastal beacons.",
                            conflict_tone="mysterious_exploratory",
                            conflict_narrative="Plot safe sea lanes across gale-swept shoals before rations run dry.",
                            aesthetic_style="Crisp nautical star charts with high-contrast shoal silhouettes.",
                            palette_mood="deep_oceanic_brass",
                            gameplay_pacing="methodical_tactical",
                            pacing_score=0.5,
                            target_difficulty=0.7,
                            camera_style="scrolling",
                            mechanics_guidelines=[
                                "Manual star sightings and sail trim govern heading and speed.",
                                "Coastal beacons remain dark until reached and fueled by the player.",
                            ],
                        )
                        return SimpleNamespace(text=ws.model_dump_json(), usage_metadata=None)
                    raise AssertionError(f"Unexpected schema in test: {s_name}")

            self.models = _Models()

    room_client = _SchemaAwareRoomClient()
    room = CreativeWritersRoom(client=room_client, max_turns=1, min_turns_before_convergence=5)
    monkeypatch.setattr(room, "_complete_missing_design_sections", lambda *a, **kw: None)
    bible = room.run_writers_room(
        dna=None,
        creative_prompt="Build Starlight Reckoning.",
        candidate_dir=cand_dir,
    )

    # 1. Game Bible started at Turn 0 with ContextCurator's Genre Craft & Fun Foundation, followed by handoff, CoProducer (Council Chair) & GameplayDesigner
    assert [u.speaker_role for u in bible.writers_room_transcript[:4]] == [
        "ContextCurator",
        "ExecutiveProducer",
        "CoProducer",
        "GameplayDesigner",
    ]
    assert [u.turn_index for u in bible.writers_room_transcript[:4]] == [0, 1, 2, 3]
    assert "Sight stars" in bible.gameplay_pillars

    # 2. GameplayDesigner received the Turn 0 Genre Craft & Fun Foundation directly in its Writers' Room prompt
    assert len(room_client.specialist_turn_prompts) >= 1
    assert "Genre Craft & Fun Foundation" in room_client.specialist_turn_prompts[0]
    assert "Manual Celestial Reckoning" in room_client.specialist_turn_prompts[0]
    assert "Auto-sailing to objectives without player steering" in room_client.specialist_turn_prompts[0]

    # 3. GameplayDesigner's Turn 1 evolved GenreCraftFoundation pillars in-place without duplicating old pillar entries
    assert bible.genre_craft_foundation is not None
    assert len(bible.genre_craft_foundation.foundational_pillars) == 4
    manual_entries = [p for p in bible.gameplay_pillars if p.startswith("Manual Celestial Reckoning:")]
    assert len(manual_entries) == 1
    assert "Cross-wind drift compensation" in manual_entries[0]
    assert any(p.startswith("Harmonic Beacon Triangulation:") for p in bible.gameplay_pillars)
    snap = room._format_bible_snapshot(bible)
    assert "Harmonic Beacon Triangulation" in snap
    assert format_genre_craft_foundation_briefing(bible.genre_craft_foundation) != ""

    # 4. Tier 2 structural pivot with an updated concept_brief.genre_craft_foundation updates the Game Bible
    pivoted_foundation = _sample_foundation(
        archetype="Deep-Trench Bathysphere Acoustic Charting",
        extra_pillar_title="Thermal Vent Acoustic Ranging",
    )
    (cand_dir / ".pending_structural_pivot.json").write_text(
        json.dumps(
            [
                {
                    "deliberation": "Pivoting to deep-sea acoustic charting.",
                    "steering_brief": "Pivot to bathysphere acoustic charting.",
                    "concept_brief": {
                        "deliberation": "Updated concept brief.",
                        "working_title": "Abyssal Reckoning",
                        "genre_craft_foundation": pivoted_foundation.model_dump(),
                    },
                    "cancel_item_ids": [],
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    consumed = room._consume_pending_steering(bible, candidate_dir=cand_dir)
    assert any("STRUCTURAL PIVOT" in note for note in consumed)
    assert bible.game_title == "Abyssal Reckoning"
    assert bible.genre_craft_foundation.genre_and_subgenre_archetype == "Deep-Trench Bathysphere Acoustic Charting"
    assert any(p.startswith("Thermal Vent Acoustic Ranging:") for p in bible.gameplay_pillars)
