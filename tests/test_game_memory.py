"""End-to-end unit and integration tests for the per-game multi-bank long-term memory system."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

from engine.studio.context_curator import ContextCuratorAgent, ContextCuratorDecision
from engine.studio.daemon import rehydrate_resumable_candidate_from_db
from engine.studio.game_memory import (
    CANONICAL_MEMORY_BANKS,
    GameMemoryStore,
    detect_active_slice_id,
)
from engine.studio.models import (
    AcceptanceCriterion,
    FilmstripPlan,
    GameBibleSpec,
    GameMemoryEntry,
    GameMemoryLedger,
    GameSliceSpec,
    MemoryMutationProposal,
    NextSliceStrategyPlan,
    RenderingStrategySpec,
    RequirementTrack,
    SliceStateReflectionTurn,
    StudioAgentError,
    SubjectRenderTreatment,
    TaskLeadDecision,
    TrackCodeSynthesisPayload,
    WorkItem,
    WorkItemProposal,
    WritersRoomTurnProposal,
)
from engine.studio.swe import LeadGameplaySWEAgent
from engine.studio.workers import WorkerContext, plan_filmstrip, plan_task_lead_turn
from engine.studio.writers_room import SPECIALIST_ROSTER, CreativeWritersRoom
from engine.telemetry.db import TelemetryDB
from engine.ui.server import SwarmEventBus, create_app


class _DirectionalEmbedClient:
    """Deterministic embedding client mapping distinct orthogonal semantic axes for vector tests."""

    __module__ = "google.genai.client"

    def __init__(self) -> None:
        self.models = self
        self.embed_calls: List[List[str]] = []

    @staticmethod
    def _vector_for(text: str) -> List[float]:
        import hashlib
        import re

        low = text.lower()
        vec = [0.005] * 256
        if "economy" in low or "progression" in low or "tier" in low:
            vec[0] = 1.0
        if "harness" in low or "capture" in low or "telemetry" in low:
            vec[1] = 1.0
        if "shader" in low or "palette" in low or "silhouette" in low:
            vec[2] = 1.0
        if "acoustic" in low or "envelope" in low:
            vec[3] = 1.0
        if "canvaslayer" in low or "hud" in low or "overlay" in low:
            vec[4] = 1.2
        if "camera2d" in low or "bounds" in low or "panning" in low:
            vec[5] = 1.2
        # Add fine-grained hashed token components so distinct notes within a domain stay separable
        # while paraphrased near-duplicates sharing vocabulary + domain exceed cosine >= 0.86
        stop = {"memory", "bank", "kind", "status", "author", "title", "content", "rationale", "the", "and", "for", "with", "into", "onto", "across", "keep", "ensure", "use", "record"}
        tokens = [tok for tok in re.findall(r"[a-z0-9_]{3,}", low) if tok not in stop]
        for tok in set(tokens):
            slot = 16 + (int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16) % 240)
            vec[slot] += 0.22
        return vec

    def embed_content(self, *, model: str, contents: List[str], config: Any = None) -> Any:
        batch = [str(c) for c in contents]
        self.embed_calls.append(batch)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=self._vector_for(c)) for c in batch]
        )


def test_memory_models_rationale_first_and_writers_room_schema_budget() -> None:
    for cls in (GameMemoryEntry, MemoryMutationProposal, GameMemoryLedger):
        field_names = list(cls.model_fields.keys())
        assert field_names[0] == "deliberation", f"{cls.__name__} must declare deliberation first"

    assert "council" in CANONICAL_MEMORY_BANKS
    assert "task_lead" in CANONICAL_MEMORY_BANKS
    assert "swe_engineer" in CANONICAL_MEMORY_BANKS
    assert "qa_test_engineer" in CANONICAL_MEMORY_BANKS

    for spec in SPECIALIST_ROSTER:
        role = spec["role"]
        dyn_cls = WritersRoomTurnProposal.for_role(role)
        schema = dyn_cls.model_json_schema()
        defs = schema.get("$defs", {})
        total_props = len(schema.get("properties", {})) + sum(
            len(d.get("properties", {})) for d in defs.values() if isinstance(d, dict)
        )
        assert len(defs) <= 12, f"{role} exceeded $defs ceiling: {len(defs)}"
        assert total_props <= 95, f"{role} exceeded property ceiling: {total_props}"


def test_multi_bank_lifecycle_dedup_compact_notes_and_slice_detection(tmp_path: Path) -> None:
    bible = GameBibleSpec(
        deliberation="Establish multi-slice roadmap.",
        bible_id="b_mem",
        cartridge_id=tmp_path.name,
        game_title="Chrono Foundry",
        core_genre="simulation",
        active_slice_index=0,
        slice_roadmap=[
            GameSliceSpec(
                deliberation="Initial slice.",
                slice_id="slice_0_foundation",
                slice_index=0,
                title="Foundation Loop",
                slice_objective="Deliver complete core resource loop.",
                status="active",
            ),
            GameSliceSpec(
                deliberation="Expansion slice.",
                slice_id="slice_1_expansion",
                slice_index=1,
                title="Multi-Region Network",
                slice_objective="Expand logistics across interconnected sectors.",
                status="planned",
            ),
        ],
    )
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")
    assert detect_active_slice_id(tmp_path) == "slice_0_foundation"

    events: List[tuple[str, Dict[str, Any]]] = []
    store = GameMemoryStore(tmp_path, event_callback=lambda et, p: events.append((et, p)))

    # 1. Record into shared 'council' bank and specialist's own 'self' bank
    created = store.apply_mutations(
        author_role="CreativeDirector",
        proposals=[
            MemoryMutationProposal(
                deliberation="Defer multi-sector logistics routing to Slice 1 after core loop stabilizes.",
                operation="record",
                target_bank="council",
                kind="strategic_goal",
                title="Multi-sector logistics expansion",
                content="Introduce interconnected sector routing in Slice 1 once single-sector throughput is verified.",
                applicable_roles=["LevelAndProgressionDesigner", "task_lead"],
            ),
            MemoryMutationProposal(
                deliberation="Keep pacing notes in director's own bank.",
                operation="record",
                target_bank="self",
                kind="domain_note",
                title="Escalation cadence target",
                content="Keep early-session cognitive load focused on two primary resource transforms.",
            ),
        ],
    )
    assert len(created) == 2
    goal_entry, dir_entry = created[0], created[1]
    assert goal_entry.memory_bank == "council"
    assert goal_entry.slice_id == "slice_0_foundation"
    assert dir_entry.memory_bank == "CreativeDirector"

    # 2. Recording duplicate title in same active bank updates in place instead of duplicating
    deduped = store.apply_mutations(
        author_role="CreativeDirector",
        proposals=[
            MemoryMutationProposal(
                deliberation="Refine Slice 1 logistics target.",
                operation="record",
                target_bank="council",
                kind="strategic_goal",
                title="Multi-sector logistics expansion",
                content="Introduce three-sector routing and throughput balancing in Slice 1.",
                applicable_roles=["swe_engineer"],
            )
        ],
    )
    assert deduped[0].memory_id == goal_entry.memory_id
    assert set(deduped[0].applicable_roles) == {
        "LevelAndProgressionDesigner",
        "task_lead",
        "swe_engineer",
    }
    assert len(store.list_entries(bank_filter="council")) == 1

    # 3. Compact Writers' Room note syntax
    compact_entry = store.parse_and_apply_compact_note(
        author_role="qa_test_engineer",
        raw_note="self:lesson_learned:Deterministic state telemetry::Expose state keys via get_state_snapshot for verification.",
        deliberation="Record deterministic QA verification lesson.",
        item_id="verify_loop",
    )
    assert compact_entry is not None
    assert compact_entry.memory_bank == "qa_test_engineer"
    assert compact_entry.kind == "lesson_learned"
    assert compact_entry.title == "Deterministic state telemetry"

    # 4. Fulfill, reactivate via update (preserving strategic_goal kind), then supersede
    store.apply_mutations(
        author_role="CreativeDirector",
        proposals=[
            MemoryMutationProposal(
                deliberation="Delivered multi-sector routing in Slice 1.",
                operation="fulfill",
                target_memory_id=goal_entry.memory_id,
                resolution_note="Verified across three active sectors.",
            )
        ],
        slice_id="slice_1_expansion",
    )
    fulfilled = store.list_entries(status_filter="fulfilled")
    assert len(fulfilled) == 1
    assert fulfilled[0].resolution_note == "Verified across three active sectors."

    # Update reactivates without clobbering kind when kind is omitted
    store.apply_mutations(
        author_role="CreativeDirector",
        proposals=[
            MemoryMutationProposal(
                deliberation="Reopen goal for cross-sector hazard balancing.",
                operation="update",
                target_memory_id=goal_entry.memory_id,
                content="Extend three-sector routing with dynamic bottleneck recovery.",
            )
        ],
    )
    reopened = [e for e in store.list_entries(status_filter="active") if e.memory_id == goal_entry.memory_id][0]
    assert reopened.kind == "strategic_goal"

    # Supersede replaces old entry and links superseded_by_id
    sup_touched = store.apply_mutations(
        author_role="GameplayDesigner",
        proposals=[
            MemoryMutationProposal(
                deliberation="Replace linear sector chain with hub-and-spoke topology.",
                operation="supersede",
                target_memory_id=goal_entry.memory_id,
                target_bank="council",
                kind="architectural_decision",
                title="Hub-and-spoke sector topology",
                content="Connect peripheral resource sectors through a central dispatch hub.",
            )
        ],
    )
    replacement = [e for e in sup_touched if e.status == "active"][0]
    superseded = [e for e in sup_touched if e.status == "superseded"][0]
    assert superseded.superseded_by_id == replacement.memory_id
    assert any(et == "game_memory_updated" and list(p.keys())[0] == "deliberation" for et, p in events)


def test_sqlite_wal_sync_and_daemon_rehydration(tmp_path: Path) -> None:
    db_path = tmp_path / "telemetry.db"
    db = TelemetryDB(str(db_path))
    cartridges_root = tmp_path / "cartridges"
    cand_dir = cartridges_root / "candidate_genesis_memhydrate"
    cand_dir.mkdir(parents=True)

    bible = GameBibleSpec(
        deliberation="Seed bible for rehydration.",
        bible_id="b_rehydrate",
        cartridge_id=cand_dir.name,
        game_title="Aether Relay",
        core_genre="strategy",
    )
    (cand_dir / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")
    db.upsert_game_bible(
        bible,
        build_id=f"build_{cand_dir.name}",
        cartridge_id=cand_dir.name,
        status="locked",
    )

    store = GameMemoryStore(cand_dir, db=db)
    store.apply_mutations(
        author_role="swe_engineer",
        proposals=[
            MemoryMutationProposal(
                deliberation="Persist modular signal bus invariant to WAL.",
                operation="record",
                target_bank="swe_engineer",
                kind="architectural_decision",
                title="Decoupled signal dispatch",
                content="Route cross-entity state transitions through mechanics.gd signals.",
            )
        ],
    )
    rows = db.list_game_memories(cand_dir.name)
    assert len(rows) == 1
    assert rows[0]["title"] == "Decoupled signal dispatch"

    # Delete on-disk game_memory.json and verify daemon rehydration restores it from SQLite WAL
    (cand_dir / "game_memory.json").unlink()
    rehydrated = rehydrate_resumable_candidate_from_db(
        db_path=db_path,
        cartridges_root=cartridges_root,
        preferred_candidate_id=cand_dir.name,
    )
    assert rehydrated == cand_dir
    assert (cand_dir / "game_memory.json").is_file()
    restored = GameMemoryStore(cand_dir, db=db).list_entries()
    assert len(restored) == 1
    assert restored[0].memory_bank == "swe_engineer"


def test_semantic_vector_query_cache_and_rule5_fail_fast(tmp_path: Path) -> None:
    store = GameMemoryStore(tmp_path)
    store.apply_mutations(
        author_role="Council",
        proposals=[
            MemoryMutationProposal(
                deliberation="Record economy pacing goal.",
                operation="record",
                target_bank="council",
                kind="strategic_goal",
                title="Multi-tier progression economy",
                content="Introduce mid-game progression tiers once base economy loop stabilizes.",
            ),
            MemoryMutationProposal(
                deliberation="Record QA harness capture constraint.",
                operation="record",
                target_bank="qa_test_engineer",
                kind="lesson_learned",
                title="Harness capture window sizing",
                content="Keep automated harness capture ticks compact and telemetry assertions deterministic.",
            ),
            MemoryMutationProposal(
                deliberation="Record shader palette invariant.",
                operation="record",
                target_bank="visual_artist",
                kind="domain_note",
                title="High-contrast silhouette palette",
                content="Maintain foreground actor silhouette contrast against mid-tone shader strata.",
            ),
        ],
    )

    # Rule 5: non-empty semantic query_text without an embedding client MUST fail fast
    with pytest.raises(StudioAgentError, match="embed_content"):
        store.query_memories(caller_role="qa_test_engineer", query_text="harness capture telemetry", client=None)

    embed_client = _DirectionalEmbedClient()
    hits = store.query_memories(
        caller_role="qa_test_engineer",
        query_text="harness capture telemetry",
        client=embed_client,
        top_k=3,
    )
    assert hits[0][0].title == "Harness capture window sizing"
    assert len(embed_client.embed_calls) == 1
    assert len(embed_client.embed_calls[0]) == 4

    # Second query reuses SHA-256 disk cache for all 3 entries -> embeds ONLY the 1 query string
    hits_2 = store.query_memories(
        caller_role="CreativeDirector",
        query_text="tier economy progression",
        client=embed_client,
        top_k=3,
    )
    assert hits_2[0][0].title == "Multi-tier progression economy"
    assert len(embed_client.embed_calls) == 2
    assert len(embed_client.embed_calls[1]) == 1


def test_context_curator_indexes_game_memory_and_invalidates_cache(tmp_path: Path) -> None:
    bible = GameBibleSpec(
        deliberation="Curator test bible.",
        bible_id="b_cur",
        cartridge_id=tmp_path.name,
        game_title="Signal Spire",
        core_genre="puzzle",
    )
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    store = GameMemoryStore(tmp_path)
    store.apply_mutations(
        author_role="CreativeDirector",
        proposals=[
            MemoryMutationProposal(
                deliberation="Shared council goal relevant to all roles.",
                operation="record",
                target_bank="council",
                kind="strategic_goal",
                title="Expand multi-spire relay network",
                content="Scale single-spire prototype into interconnected multi-spire network in Slice 1.",
            ),
            MemoryMutationProposal(
                deliberation="QA-specific operational lesson.",
                operation="record",
                target_bank="qa_test_engineer",
                kind="lesson_learned",
                title="Snapshot state assertions",
                content="Assert causal state keys in get_state_snapshot across capture checkpoints.",
            ),
            MemoryMutationProposal(
                deliberation="Unrelated audio specialist private note.",
                operation="record",
                target_bank="audio_designer",
                kind="domain_note",
                title="Low-pass acoustic tail",
                content="Use exponential decay envelopes below 400Hz for ambient resonance.",
            ),
        ],
    )

    curator_calls = {"count": 0}

    class _CuratorClient(_DirectionalEmbedClient):
        __module__ = "google.genai.client"

        def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
            sname = getattr(getattr(config, "response_schema", None), "__name__", "")
            if sname == "ContextCuratorDecision":
                curator_calls["count"] += 1
                prompt_str = str(contents)
                assert "Expand multi-spire relay network" in prompt_str
                assert "Snapshot state assertions" in prompt_str
                return SimpleNamespace(
                    text=ContextCuratorDecision(
                        deliberation="Select council goal and QA lesson for qa_test_engineer.",
                        search_queries_used=["harness capture telemetry"],
                        role_tailored_synthesis="Apply deterministic snapshot telemetry assertions.",
                        selected_chunk_ids=[],
                        selected_image_paths=[],
                    ).model_dump_json()
                )
            # RenderingStrategy fallback schema when rendering_strategy.json is first initialized
            schema_cls = getattr(config, "response_schema", None)
            if schema_cls is not None:
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "deliberation": "Default rendering strategy.",
                            "camera_projection": "orthographic_2d",
                            "viewport_width": 1280,
                            "viewport_height": 720,
                        }
                    )
                )
            raise AssertionError(f"Unexpected schema {sname}")

    client = _CuratorClient()
    curator = ContextCuratorAgent(cartridge_dir=tmp_path, client=client)
    pack_1 = curator.curate_context(
        target_agent_role="qa_test_engineer",
        task_request="Capture verification filmstrip for relay activation",
        item_id="verify_relay",
        item_kind="integration",
    )
    assert curator_calls["count"] == 1
    assert "Expand multi-spire relay network" in pack_1.prompt_block
    assert "Snapshot state assertions" in pack_1.prompt_block
    assert "Low-pass acoustic tail" not in pack_1.prompt_block

    # Identical call without memory mutations hits disk cache (0 additional LLM calls)
    curator_cached = ContextCuratorAgent(cartridge_dir=tmp_path, client=client)
    _ = curator_cached.curate_context(
        target_agent_role="qa_test_engineer",
        task_request="Capture verification filmstrip for relay activation",
        item_id="verify_relay",
        item_kind="integration",
    )
    assert curator_calls["count"] == 1

    # Updating game_memory.json changes memory_sig -> invalidates cache and runs fresh curation
    store.apply_mutations(
        author_role="qa_test_engineer",
        proposals=[
            MemoryMutationProposal(
                deliberation="Add camera framing lesson for QA.",
                operation="record",
                target_bank="qa_test_engineer",
                kind="lesson_learned",
                title="Viewport target framing",
                content="Center diagnostic Camera2D on active relay node during input segments.",
            )
        ],
    )
    curator_after_update = ContextCuratorAgent(cartridge_dir=tmp_path, client=client)
    pack_2 = curator_after_update.curate_context(
        target_agent_role="qa_test_engineer",
        task_request="Capture verification filmstrip for relay activation",
        item_id="verify_relay",
        item_kind="integration",
    )
    assert curator_calls["count"] == 2
    assert "Viewport target framing" in pack_2.prompt_block


def test_direct_agent_memory_query_and_write_across_task_lead_swe_and_qa(tmp_path: Path) -> None:
    bible = GameBibleSpec(
        deliberation="Worker memory loop test.",
        bible_id="b_wrk",
        cartridge_id=tmp_path.name,
        game_title="Flux Harbor",
        core_genre="simulation",
    )
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")
    (tmp_path / "rendering_strategy.json").write_text(
        RenderingStrategySpec(
            deliberation="Layered 2D rendering strategy.",
            rendering_paradigm="layered_2d",
            renderer="compatibility",
            uses_environment_glow=False,
            treatments=[
                SubjectRenderTreatment(
                    deliberation="Harbor structures.",
                    subject_id="harbor_node",
                    technique="static_sprite",
                    target_resolution=(128, 128),
                )
            ],
            lighting_and_atmosphere="Warm harbor directional lighting.",
            effects_stack=["GPUParticles2D"],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (tmp_path / "project.godot").write_text('[application]\nconfig/name="Flux Harbor"\n[input]\ninteract={}\n', encoding="utf-8")
    (tmp_path / "main.gd").write_text("extends Node2D\nfunc get_state_snapshot() -> Dictionary:\n\treturn {}\n", encoding="utf-8")

    store = GameMemoryStore(tmp_path)
    store.apply_mutations(
        author_role="Council",
        proposals=[
            MemoryMutationProposal(
                deliberation="Seed council economy architecture memory.",
                operation="record",
                target_bank="council",
                kind="architectural_decision",
                title="Economy tick accumulator contract",
                content="All harbor resource rates must integrate via integer tick accumulators in mechanics.gd.",
            ),
            MemoryMutationProposal(
                deliberation="Seed QA harness lesson.",
                operation="record",
                target_bank="qa_test_engineer",
                kind="lesson_learned",
                title="Compact harness checkpoint cadence",
                content="Space capture ticks across distinct pre-action and post-action intervals.",
            ),
        ],
    )

    # 1. Task Lead issues memory_query on call 1, then delegates + writes memory_updates on call 2
    lead_prompts: List[str] = []
    lead_responses = iter(
        [
            TaskLeadDecision(
                deliberation="Query long-term memory for economy architecture contracts before briefing SWE.",
                current_strategy="Look up economy accumulator conventions.",
                memory_query="economy progression tier",
                memory_query_bank="council",
                action="delegate",
                specialist_agent="swe_engineer",
                brief="",
            ).model_dump_json(),
            TaskLeadDecision(
                deliberation="Received economy accumulator contract; delegate implementation and record lead note.",
                current_strategy="Delegate integer accumulator implementation to swe_engineer.",
                memory_query_answer="Use integer tick accumulators in mechanics.gd.",
                memory_updates=[
                    MemoryMutationProposal(
                        deliberation="Record task decomposition lesson for harbor mechanics.",
                        operation="record",
                        target_bank="self",
                        kind="lesson_learned",
                        title="Pair accumulator code with state snapshot key",
                        content="Always brief swe_engineer to expose accumulator totals in get_state_snapshot.",
                    )
                ],
                action="delegate",
                specialist_agent="swe_engineer",
                brief="Implement integer tick accumulator in mechanics.gd and expose harbour_flux in get_state_snapshot().",
                target_paths=["mechanics.gd"],
            ).model_dump_json(),
        ]
    )

    class _LeadMemClient(_DirectionalEmbedClient):
        __module__ = "google.genai.client"

        def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
            sname = getattr(getattr(config, "response_schema", None), "__name__", "")
            if sname == "ContextCuratorDecision":
                return SimpleNamespace(
                    text=ContextCuratorDecision(
                        deliberation="Curate task lead context.",
                        search_queries_used=["economy"],
                        role_tailored_synthesis="Harbor mechanics context.",
                        selected_chunk_ids=[],
                        selected_image_paths=[],
                    ).model_dump_json()
                )
            if sname == "RenderingStrategySpec":
                return SimpleNamespace(text=json.dumps({"deliberation": "Default rendering strategy."}))
            lead_prompts.append(str(contents if not isinstance(contents, list) else contents[0]))
            return SimpleNamespace(text=next(lead_responses))

    work_item = WorkItem(
        deliberation="Implement harbor economy.",
        item_id="harbor_econ",
        title="Harbor Flux Economy",
        kind="mechanic_code",
        spec="Implement resource tick accumulator.",
        proof_spec="Show harbor_flux increasing on interact.",
    )
    lead_client = _LeadMemClient()
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=lead_client,
        swe_agent=LeadGameplaySWEAgent(client=lead_client),
    )
    decision = plan_task_lead_turn(ctx, work_item, turn_history=[], collected_proofs=[])
    assert decision.action == "delegate"
    assert "integer tick accumulator" in decision.brief
    assert len(lead_prompts) == 2
    assert "Economy tick accumulator contract" in lead_prompts[1]
    prefix_0 = lead_prompts[0].split("CARTRIDGE SYMBOLS & FILES:")[0]
    prefix_1 = lead_prompts[1].split("CARTRIDGE SYMBOLS & FILES:")[0]
    assert prefix_0 == prefix_1
    assert any(e.title == "Pair accumulator code with state snapshot key" for e in store.list_entries(bank_filter="task_lead"))

    # 2. LeadGameplaySWEAgent issues memory_query then synthesizes code + records swe_engineer memory
    swe_prompts: List[str] = []
    swe_responses = iter(
        [
            TrackCodeSynthesisPayload(
                deliberation="Check council economy accumulator convention first.",
                memory_query="economy progression",
                memory_query_bank="council",
                files=[],
            ).model_dump_json(),
            TrackCodeSynthesisPayload(
                deliberation="Implement mechanics.gd using integer tick accumulator and record SWE note.",
                memory_query_answer="Council requires integer tick accumulators in mechanics.gd.",
                memory_updates=[
                    MemoryMutationProposal(
                        deliberation="Record deterministic integer state update pattern.",
                        operation="record",
                        target_bank="self",
                        kind="domain_note",
                        title="Integer flux step determinism",
                        content="Advance flux counters using fixed integer increments per physics tick.",
                    )
                ],
                files=[
                    {
                        "deliberation": "Create mechanics.gd with integer accumulator.",
                        "filename": "mechanics.gd",
                        "content": "class_name HarborMechanics\nextends RefCounted\nvar flux: int = 0\nfunc step() -> void:\n\tflux += 1\n",
                    }
                ],
            ).model_dump_json(),
        ]
    )

    class _SWEMemClient(_DirectionalEmbedClient):
        __module__ = "google.genai.client"

        def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
            sname = getattr(getattr(config, "response_schema", None), "__name__", "")
            if sname == "ContextCuratorDecision":
                return SimpleNamespace(
                    text=ContextCuratorDecision(
                        deliberation="Curate SWE context.",
                        search_queries_used=["economy"],
                        role_tailored_synthesis="SWE harbor context.",
                        selected_chunk_ids=[],
                        selected_image_paths=[],
                    ).model_dump_json()
                )
            if sname == "RenderingStrategySpec":
                return SimpleNamespace(text=json.dumps({"deliberation": "Default rendering strategy."}))
            swe_prompts.append(str(contents))
            return SimpleNamespace(text=next(swe_responses))

    swe_agent = LeadGameplaySWEAgent(client=_SWEMemClient())
    track = RequirementTrack(
        deliberation="Harbor economy track.",
        id="R_ECON",
        title="Harbor Economy",
        description="Implement HarborMechanics integer accumulator.",
        target_modules=["mechanics.gd"],
        acceptance_criteria=[
            AcceptanceCriterion(
                deliberation="Verify mechanics.gd compiles.",
                id="AC1",
                description="HarborMechanics exists.",
                verification_method="ast_inspection",
            )
        ],
    )
    payload = swe_agent.implement_requirement_track(
        track=track,
        world_spec={"game_title": "Flux Harbor", "core_genre": "simulation"},
        staging_dir=tmp_path,
    )
    assert isinstance(payload, dict)
    assert (tmp_path / "mechanics.gd").is_file()
    assert len(swe_prompts) == 2
    assert "Economy tick accumulator contract" in swe_prompts[1]
    assert any(e.title == "Integer flux step determinism" for e in store.list_entries(bank_filter="swe_engineer"))

    # 3. QA plan_filmstrip queries QA memory and records lesson
    qa_prompts: List[str] = []
    qa_responses = iter(
        [
            FilmstripPlan(
                deliberation="Query QA memory for harness capture cadence lessons.",
                memory_query="harness capture telemetry",
                memory_query_bank="self",
                capture_ticks=[10, 30],
            ).model_dump_json(),
            FilmstripPlan(
                deliberation="Apply compact harness checkpoint cadence and record interact assertion note.",
                memory_query_answer="Space capture ticks across pre-action and post-action intervals.",
                memory_updates=[
                    MemoryMutationProposal(
                        deliberation="Record interact input verification note.",
                        operation="record",
                        target_bank="self",
                        kind="lesson_learned",
                        title="Pre/post interact tick spacing",
                        content="Capture baseline at tick 5 before triggering interact segment at tick 10..25.",
                    )
                ],
                capture_ticks=[5, 30],
                input_segments=[{"deliberation": "Trigger interact", "from_tick": 10, "to_tick": 25, "actions": ["interact"]}],
                council_test_scene_summary="Instantiates main.tscn, captures baseline at tick 5, triggers interact across ticks 10..25, and captures post-action state at tick 30.",
            ).model_dump_json(),
        ]
    )

    class _QAMemClient(_DirectionalEmbedClient):
        __module__ = "google.genai.client"

        def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
            sname = getattr(getattr(config, "response_schema", None), "__name__", "")
            if sname == "ContextCuratorDecision":
                return SimpleNamespace(
                    text=ContextCuratorDecision(
                        deliberation="Curate QA context.",
                        search_queries_used=["harness"],
                        role_tailored_synthesis="QA context.",
                        selected_chunk_ids=[],
                        selected_image_paths=[],
                    ).model_dump_json()
                )
            if sname == "RenderingStrategySpec":
                return SimpleNamespace(text=json.dumps({"deliberation": "Default rendering strategy."}))
            qa_prompts.append(str(contents if not isinstance(contents, list) else contents[0]))
            return SimpleNamespace(text=next(qa_responses))

    qa_ctx = WorkerContext(cartridge_dir=tmp_path, client=_QAMemClient())
    fplan = plan_filmstrip(qa_ctx, work_item)
    assert fplan.capture_ticks == [5, 30]
    assert len(qa_prompts) == 2
    assert "Compact harness checkpoint cadence" in qa_prompts[1]
    assert any(e.title == "Pre/post interact tick spacing" for e in store.list_entries(bank_filter="qa_test_engineer"))


def test_cross_slice_council_reflection_fulfills_and_carries_strategic_goals(tmp_path: Path) -> None:
    bible = GameBibleSpec(
        deliberation="Cross-slice memory persistence test.",
        bible_id="b_slice",
        cartridge_id=tmp_path.name,
        game_title="Starlight Conduit",
        core_genre="strategy",
        active_slice_index=0,
        slice_roadmap=[
            GameSliceSpec(
                deliberation="Slice 0 foundation.",
                slice_id="slice_0_core",
                slice_index=0,
                title="Core Relay Loop",
                slice_objective="Deliver single-sector relay routing.",
                status="active",
            )
        ],
    )
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    store = GameMemoryStore(tmp_path)
    seeded = store.apply_mutations(
        author_role="CreativeDirector",
        proposals=[
            MemoryMutationProposal(
                deliberation="Track Slice 0 core loop milestone.",
                operation="record",
                target_bank="council",
                kind="strategic_goal",
                title="Single-sector relay loop stability",
                content="Prove deterministic relay power transfer in Slice 0.",
            ),
            MemoryMutationProposal(
                deliberation="Track multi-slice network expansion goal across slices.",
                operation="record",
                target_bank="council",
                kind="strategic_goal",
                title="Dynamic multi-biome relay expansion",
                content="Once Slice 0 single-sector routing is proven, expand into dynamic multi-biome topology in Slice 1.",
            ),
        ],
    )
    slice0_goal_id = seeded[0].memory_id
    multi_slice_goal_id = seeded[1].memory_id

    reflection_prompts_seen: List[str] = []

    class _SliceCouncilClient(_DirectionalEmbedClient):
        __module__ = "google.genai.client"

        def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
            schema = getattr(config, "response_schema", None)
            sname = getattr(schema, "__name__", "")
            prompt_txt = str(contents if not isinstance(contents, list) else contents[0])
            reflection_prompts_seen.append(prompt_txt)
            if sname == "ContextCuratorDecision":
                return SimpleNamespace(
                    text=ContextCuratorDecision(
                        deliberation="Curate slice reflection context.",
                        search_queries_used=["economy"],
                        role_tailored_synthesis="Slice reflection context.",
                        selected_chunk_ids=[],
                        selected_image_paths=[],
                    ).model_dump_json()
                )
            if sname == "SliceStateReflectionTurn":
                return SimpleNamespace(
                    text=SliceStateReflectionTurn(
                        deliberation="Reflect on Slice 0 and check carried strategic goals.",
                        spoken_dialogue="Single-sector relay loop is working; let us build multi-biome routing next.",
                        honest_current_state_critique="Single-sector relay loop is working cleanly.",
                        how_to_play_and_intuitiveness_gaps="Controls are legible.",
                        depth_and_fun_gaps="Dynamic multi-biome relay topology remains for Slice 1.",
                        accomplishment_and_goals_gaps="Add cross-biome milestones.",
                        strategy_for_next_slice="Deliver dynamic multi-biome relay generation.",
                        memory_updates=[
                            MemoryMutationProposal(
                                deliberation="Record specialist transition lesson.",
                                operation="record",
                                target_bank="self",
                                kind="domain_note",
                                title="Biome transition pacing",
                                content="Introduce hazard modifiers gradually across biome borders.",
                            )
                        ],
                        current_slice_quality_score=0.75,
                    ).model_dump_json()
                )
            if sname == "NextSliceStrategyPlan":
                return SimpleNamespace(
                    text=NextSliceStrategyPlan(
                        deliberation="Fulfill Slice 0 goal, keep multi-biome goal active, and record Slice 2 horizon goal.",
                        spoken_dialogue="Slice 0 proved single-sector relay routing; next we expand across multiple biomes.",
                        current_slice_honest_diagnosis="Single-sector loop works; multi-biome scale is next.",
                        cohesion_and_world_integration_strategy="Connect sectors seamlessly.",
                        how_to_play_and_onboarding_strategy="Preserve clear diegetic relay indicators.",
                        depth_and_interesting_decisions_strategy="Add cross-biome routing trade-offs.",
                        short_term_and_long_term_goals_strategy="Track per-biome activation and full network synchronization.",
                        sense_of_accomplishment_and_reward_strategy="Illuminate synchronized biomes.",
                        next_slice_spec=GameSliceSpec(
                            deliberation="Plan Slice 1 around carried multi-biome strategic goal.",
                            slice_id="slice_1_biomes",
                            slice_index=1,
                            title="Multi-Biome Relay Network",
                            slice_objective="Expand into dynamic multi-biome relay topology.",
                            status="active",
                        ),
                        backlog_additions=[
                            WorkItemProposal(
                                deliberation="Create Slice 1 biome generator task.",
                                item_id="biome_topology",
                                title="Multi-Biome Relay Topology",
                                kind="level_map",
                                spec="Generate interconnected relay biomes.",
                                proof_spec="Show player traversing across two distinct relay biomes.",
                            )
                        ],
                        memory_updates=[
                            MemoryMutationProposal(
                                deliberation="Slice 0 single-sector relay stability is now complete.",
                                operation="fulfill",
                                target_memory_id=slice0_goal_id,
                                resolution_note="Verified in Slice 0 assembly check.",
                            ),
                            MemoryMutationProposal(
                                deliberation="Queue long-horizon endgame mastery goal for Slice 2.",
                                operation="record",
                                target_bank="council",
                                kind="strategic_goal",
                                title="Endgame harmonic cascade challenge",
                                content="Add multi-relay harmonic cascade challenge after multi-biome expansion.",
                            ),
                        ],
                    ).model_dump_json()
                )
            raise AssertionError(f"Unexpected schema {sname}")

    room = CreativeWritersRoom(client=_SliceCouncilClient(), target_slices=3)
    room._active_staging_dir = tmp_path

    plan = room._run_slice_reflection_and_strategy(
        game_bible=bible,
        dna_summary="Strategy player DNA",
        prev_slice=bible.slice_roadmap[0],
        next_idx=1,
        candidate_dir=tmp_path,
    )
    assert plan is not None
    assert any("Dynamic multi-biome relay expansion" in p for p in reflection_prompts_seen)

    by_id = {e.memory_id: e for e in store.list_entries()}
    assert by_id[slice0_goal_id].status == "fulfilled"
    assert by_id[multi_slice_goal_id].status == "active"
    active_titles = {e.title for e in store.list_entries(bank_filter="council", status_filter="active")}
    assert "Dynamic multi-biome relay expansion" in active_titles
    assert "Endgame harmonic cascade challenge" in active_titles


def test_checkpoint_copy_and_mission_control_memories_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    cand_dir = cartridges_root / "candidate_genesis_api"
    cand_dir.mkdir(parents=True)
    active_dir.mkdir(parents=True)
    db_path = tmp_path / "telemetry.db"

    (cand_dir / "game_bible.json").write_text(
        json.dumps(
            {
                "deliberation": "API test bible.",
                "bible_id": "b_api",
                "cartridge_id": cand_dir.name,
                "game_title": "Prism Citadel",
                "core_genre": "adventure",
                "active_slice_index": 0,
                "slice_roadmap": [
                    {
                        "deliberation": "Slice 0",
                        "slice_id": "slice_0_citadel",
                        "slice_index": 0,
                        "title": "Citadel Courtyard",
                        "slice_objective": "Core prism refraction puzzle.",
                        "status": "active",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bus = SwarmEventBus()
    app = create_app(
        db_path=db_path,
        active_cartridge_dir=active_dir,
        cartridges_root=cartridges_root,
        event_bus=bus,
    )
    client = TestClient(app)

    # 1. Empty GET works cleanly with deliberation first
    r0 = client.get(f"/api/studio/memories?cartridge={cand_dir.name}")
    assert r0.status_code == 200
    body0 = r0.json()
    assert list(body0.keys())[0] == "deliberation"
    assert body0["total_count"] == 0
    assert "council" in body0["banks"]

    # 2. Missing deliberation fails fast with HTTP 400 (Rule 4 & Rule 5)
    r_bad = client.post(
        "/api/studio/memories",
        json={
            "deliberation": "",
            "operation": "record",
            "target_bank": "council",
            "title": "Invalid",
            "content": "Missing rationale",
        },
    )
    assert r_bad.status_code == 400

    # 3. Record Council strategic goal + specialist lesson via POST /api/studio/memories
    r_rec1 = client.post(
        "/api/studio/memories",
        json={
            "deliberation": "Capture multi-slice spire ascension goal for Slice 1.",
            "cartridge": cand_dir.name,
            "author_role": "CreativeDirector",
            "operation": "record",
            "target_bank": "council",
            "kind": "strategic_goal",
            "title": "Upper spire ascension wing",
            "content": "Expand courtyard refraction mechanics into multi-floor spire wing in Slice 1.",
            "applicable_roles": ["LevelAndProgressionDesigner", "task_lead"],
        },
    )
    assert r_rec1.status_code == 200
    rec1_data = r_rec1.json()
    assert rec1_data["total_count"] == 1
    assert rec1_data["active_count"] == 1
    mem_id = rec1_data["entries"][0]["memory_id"]
    assert rec1_data["entries"][0]["slice_id"] == "slice_0_citadel"

    r_rec2 = client.post(
        "/api/studio/memories",
        json={
            "deliberation": "Record SWE prism raycast caching note.",
            "cartridge": cand_dir.name,
            "author_role": "swe_engineer",
            "operation": "record",
            "target_bank": "swe_engineer",
            "kind": "architectural_decision",
            "title": "Deterministic ray bounce cap",
            "content": "Bound recursive beam reflections to 8 bounces per tick.",
        },
    )
    assert r_rec2.status_code == 200
    assert r_rec2.json()["total_count"] == 2

    # 4. Structured filtering by bank and kind
    r_swe = client.get(f"/api/studio/memories?cartridge={cand_dir.name}&bank=swe_engineer")
    assert r_swe.status_code == 200
    assert len(r_swe.json()["entries"]) == 1
    assert r_swe.json()["entries"][0]["title"] == "Deterministic ray bounce cap"

    # 5. Semantic query ?q=... without GEMINI_API_KEY fails fast with 503 (Rule 2 & Rule 5: no substring fallback)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r_sem_err = client.get(f"/api/studio/memories?cartridge={cand_dir.name}&q=spire+ascension")
    assert r_sem_err.status_code == 503

    # 6. Fulfill strategic goal via POST /api/studio/memories
    r_ful = client.post(
        "/api/studio/memories",
        json={
            "deliberation": "Upper spire wing shipped and verified.",
            "cartridge": cand_dir.name,
            "author_role": "CreativeDirector",
            "operation": "fulfill",
            "target_memory_id": mem_id,
            "resolution_note": "Delivered in Slice 1 checkpoint.",
        },
    )
    assert r_ful.status_code == 200
    ful_data = r_ful.json()
    assert ful_data["active_count"] == 1
    fulfilled_entries = [e for e in ful_data["entries"] if e["memory_id"] == mem_id]
    assert fulfilled_entries[0]["status"] == "fulfilled"

    # 7. Verify checkpoint persistence copies game_memory.json
    import engine.studio.orchestrator as orch_mod
    from engine.studio.orchestrator import StudioOrchestrator

    monkeypatch.setattr(orch_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    orch = StudioOrchestrator.__new__(StudioOrchestrator)
    orch.event_bus = None
    orch._copy2_unlocked = lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes())
    orch._rmtree_force = lambda p: None
    orch._get_telemetry_db = lambda: TelemetryDB(str(db_path))
    orch.active_build_id = f"build_{cand_dir.name}"
    chk_track = RequirementTrack(
        deliberation="Checkpoint track",
        id="slice_0_complete",
        title="Slice 0 Complete",
        description="Verified Slice 0",
    )
    chk_dir = orch._persist_verified_checkpoint(
        staging_dir=cand_dir,
        track=chk_track,
        verified_tracks=["slice_0_complete"],
        mirror_to_active=False,
    )
    assert (chk_dir / "game_memory.json").is_file()
    chk_meta = json.loads((chk_dir / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    assert "game_memory.json" in chk_meta["sha256_manifest"]


def test_vector_similarity_deduplication_and_default_empty_selectivity_contracts(tmp_path: Path) -> None:
    """Verifies that near-duplicate paraphrased memories in the same bank merge in place via vector similarity,
    distinct memories in the same bank remain separate, cross-bank ownership stays isolated, and all in-slice
    memory schemas enforce default-empty selectivity."""
    from engine.studio.models import (
        AssetRequirement,
        AudioAssetPlanBatch,
        BlindPlaytestCouncilVerdict,
        ReviewBatch,
        ScriptWritingDeliverable,
        TaskShapingBatch,
    )

    # 1. Verify default-empty negative gating across in-slice schemas
    for schema_cls in (
        AssetRequirement,
        AudioAssetPlanBatch,
        TrackCodeSynthesisPayload,
        FilmstripPlan,
        TaskShapingBatch,
        ReviewBatch,
        BlindPlaytestCouncilVerdict,
        ScriptWritingDeliverable,
        TaskLeadDecision,
    ):
        desc = schema_cls.model_fields["memory_updates"].description or ""
        assert "DEFAULT TO []" in desc, f"{schema_cls.__name__}.memory_updates missing DEFAULT TO [] rubric"

    wr_note_desc = WritersRoomTurnProposal.model_fields["memory_note"].description or ""
    assert "DEFAULT TO ''" in wr_note_desc
    prop_delib_desc = MemoryMutationProposal.model_fields["deliberation"].description or ""
    assert "Counterfactual durability" in prop_delib_desc

    # 2. Vector similarity deduplication within the same bank (different titles, equivalent meaning)
    events: List[tuple[str, Dict[str, Any]]] = []
    embed_client = _DirectionalEmbedClient()
    store = GameMemoryStore(
        tmp_path,
        event_callback=lambda et, payload: events.append((et, payload)),
        client=embed_client,
    )

    first = store.apply_mutations(
        author_role="swe_engineer",
        proposals=[
            MemoryMutationProposal(
                deliberation="HUD widgets drifted with Camera2D scrolling until isolated onto a CanvasLayer.",
                operation="record",
                target_bank="self",
                kind="architectural_decision",
                title="Isolate HUD widgets onto CanvasLayer",
                content="Mount all HUD overlay widgets inside a dedicated CanvasLayer decoupled from Camera2D transforms.",
                applicable_roles=["swe_engineer"],
            )
        ],
    )
    assert len(first) == 1
    orig_id = first[0].memory_id
    orig_created = first[0].created_at_utc

    # Paraphrased duplicate with a DIFFERENT title in the same bank -> merges into `orig_id`
    second = store.apply_mutations(
        author_role="swe_engineer",
        proposals=[
            MemoryMutationProposal(
                deliberation="Prevent HUD overlay drift during Camera2D world scrolling via CanvasLayer.",
                operation="record",
                target_bank="swe_engineer",
                kind="architectural_decision",
                title="Dedicated CanvasLayer for HUD overlays",
                content="Keep all HUD overlay widgets on a dedicated CanvasLayer decoupled from Camera2D world transforms.",
                applicable_roles=["ui_designer", "task_lead"],
            )
        ],
    )
    assert len(second) == 1
    assert second[0].memory_id == orig_id
    assert second[0].created_at_utc == orig_created
    assert second[0].title == "Dedicated CanvasLayer for HUD overlays"
    assert "swe_engineer" in second[0].applicable_roles
    assert "task_lead" in second[0].applicable_roles
    assert len(store.list_entries(bank_filter="swe_engineer", status_filter="active")) == 1
    assert orig_id in events[-1][1]["deduplicated_memory_ids"]

    # 3. Genuinely distinct memory in the SAME bank -> kept as a separate entry
    third = store.apply_mutations(
        author_role="swe_engineer",
        proposals=[
            MemoryMutationProposal(
                deliberation="Integer tick counters avoid floating-point drift across headless simulation runs.",
                operation="record",
                target_bank="swe_engineer",
                kind="lesson_learned",
                title="Integer economy tick accumulators",
                content="Advance economy progression counters with integer physics ticks instead of frame delta floats.",
            )
        ],
    )
    assert len(third) == 1
    assert third[0].memory_id != orig_id
    assert len(store.list_entries(bank_filter="swe_engineer", status_filter="active")) == 2

    # 4. Same semantic topic recorded in a DIFFERENT bank ('council') stays bank-isolated
    council_entry = store.apply_mutations(
        author_role="UXExperienceDesigner",
        proposals=[
            MemoryMutationProposal(
                deliberation="Studio-wide HUD CanvasLayer standard shared with Council.",
                operation="record",
                target_bank="council",
                kind="architectural_decision",
                title="CanvasLayer HUD standard",
                content="Mount all HUD overlay widgets inside a dedicated CanvasLayer decoupled from Camera2D transforms.",
            )
        ],
    )
    assert len(council_entry) == 1
    assert council_entry[0].memory_id != orig_id
    assert len(store.list_entries(bank_filter="council", status_filter="active")) == 1

