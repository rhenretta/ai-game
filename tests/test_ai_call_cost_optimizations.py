"""Regression tests for AI Call & Token Cost Optimizations (P0 + P1 + P2).

Verifies:
1. SWE graph & context scoping (`exclude_prefixes=('tests/',)`, `max_imported_by_hops=1` preventing
   hub `main.gd` reverse fan-out, compact SWE asset summaries, and non-target `.gd` summarization).
2. Byte-0 static prefix canonicalization (`>= 4,096`-token implicit cache invariance) across:
   - `plan_task_lead_turn` (`t0 -> t1 -> t2`, keeping changing proofs & cartridge symbol summaries at tail)
   - `plan_filmstrip` (`pass 1 -> pass 2`, keeping `filmstrip_pass1.png` at tail)
   - `CreativeWritersRoom._run_review_round` (1 shared `curate_context` call + identical byte-0 `parts`
     across all 4 Council reviewers)
3. Retry-storm safeguards:
   - `_archive_and_extract_broken_qa_script` preserves `last_failed_test_scene.gd` + numbered error window
   - `_format_failed_attempts_postmortem` injects prior failed strategies & turn summaries on auto-retry
4. Surgical QA test-scene patching (`FilmstripPlan.test_scene_patches` & zero-rewrite reuse of existing
   `tests/test_<safe_id>.gd`) and `PlaytestSessionConfig.compaction_interval_turns == 75`.
5. Unified USD + media/embedding/playtest accounting in `TokenUsageTracker`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from PIL import Image

from engine.playtest.agent import BlindPlaytesterAgent, TurnObservation
from engine.playtest.models import BlindStoreCard, PlaytestSessionConfig
from engine.studio.context_curator import CartridgeSemanticIndex, CuratedContextPack, ContextCuratorDecision
from engine.studio.models import (
    FilmstripPlan,
    GameBibleSpec,
    ReviewBatch,
    ReviewDecision,
    SpecialistReport,
    SurgicalCodePatch,
    TaskLeadDecision,
    TokenUsageTracker,
    VisualMockupArtifact,
)
from engine.studio.production_board import ProductionBoard, WorkerResult, WorkItemProposal
from engine.studio.workers import (
    WorkerContext,
    plan_filmstrip,
    plan_task_lead_turn,
    run_qa_test_engineer_specialist,
    validate_filmstrip_plan,
)
from engine.studio.writers_room import CreativeWritersRoom


def _make_png(path: Path, color=(40, 90, 140)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 36), color).save(path)
    return path


def test_swe_graph_scoping_excludes_tests_and_bounds_reverse_hub_fanout(tmp_path: Path) -> None:
    """`get_transitive_related_modules` must exclude `tests/*` and stop reverse `imported_by`
    hub files (`main.gd`) from fanning out into unrelated sibling modules."""
    (tmp_path / "project.godot").write_text("[application]\nconfig/name=\"ScopeTest\"\n", encoding="utf-8")
    (tmp_path / "main.gd").write_text(
        'extends Node2D\n'
        'const Builder = preload("res://scenes/mechanics/builder_tool.gd")\n'
        'const UnrelatedWeather = preload("res://scenes/world/weather_fx.gd")\n'
        'const UnrelatedAudio = preload("res://scenes/audio/ambience.gd")\n',
        encoding="utf-8",
    )
    (tmp_path / "scenes" / "mechanics").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scenes" / "world").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scenes" / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)

    (tmp_path / "scenes" / "mechanics" / "builder_tool.gd").write_text(
        'extends Node2D\nconst SharedGrid = preload("res://scenes/mechanics/grid_rules.gd")\n',
        encoding="utf-8",
    )
    (tmp_path / "scenes" / "mechanics" / "grid_rules.gd").write_text(
        "extends RefCounted\nfunc is_cell_open(_x: int, _y: int) -> bool:\n\treturn true\n",
        encoding="utf-8",
    )
    (tmp_path / "scenes" / "world" / "weather_fx.gd").write_text("extends Node2D\n", encoding="utf-8")
    (tmp_path / "scenes" / "audio" / "ambience.gd").write_text("extends Node\n", encoding="utf-8")
    (tmp_path / "tests" / "test_old_item.gd").write_text(
        'extends Node2D\nconst Builder = preload("res://scenes/mechanics/builder_tool.gd")\n',
        encoding="utf-8",
    )

    idx = CartridgeSemanticIndex(cartridge_dir=tmp_path, client=None)
    related = idx.get_transitive_related_modules(
        ["scenes/mechanics/builder_tool.gd"],
        max_hops=2,
        include_imported_by=True,
        max_imported_by_hops=1,
        exclude_prefixes=("tests/",),
    )

    # Target module, its forward dependency (`grid_rules.gd`), and direct caller (`main.gd`) are included:
    assert "scenes/mechanics/builder_tool.gd" in related
    assert "scenes/mechanics/grid_rules.gd" in related
    assert "main.gd" in related
    # QA `tests/*` callers and unrelated siblings imported by hub `main.gd` MUST NOT leak in:
    assert "tests/test_old_item.gd" not in related
    assert "scenes/world/weather_fx.gd" not in related
    assert "scenes/audio/ambience.gd" not in related


def test_task_lead_byte0_static_prefix_invariance_and_failed_postmortem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Across `t0 -> t1 -> t2`, `plan_task_lead_turn` keeps `contents[0]` (charter + slice + curated pack
    + postmortem) 100% byte-for-byte invariant while dynamic turn history, changing cartridge symbols,
    and captured proof images stay at the tail (and stale filmstrips are omitted right after SWE edits)."""
    cart_dir = tmp_path / "cartridge"
    cart_dir.mkdir(parents=True, exist_ok=True)
    (cart_dir / "project.godot").write_text("[application]\nconfig/name=\"PrefixTest\"\n", encoding="utf-8")
    (cart_dir / "main.gd").write_text("extends Node2D\nfunc _ready() -> void:\n\tpass\n", encoding="utf-8")
    mockup_p = _make_png(cart_dir / "assets" / "mockups" / "mockup_01.png", (90, 50, 120))
    long_dna = "Player loves deep systemic optimization.\n" * 60
    (cart_dir / "game_bible.json").write_text(
        json.dumps(
            {
                "game_title": "Conduit Foundry",
                "target_player_dna_summary": long_dna,
                "visual_mockups": [{"title": "Foundry", "image_path": "assets/mockups/mockup_01.png"}],
                "core_loop": {
                    "moment_to_moment_loop": "Route conduits",
                    "minute_to_minute_loop": "Balance pressure",
                    "session_loop": "Expand network",
                    "player_agency_and_optimization_depth": "Multiple routing topologies",
                },
            }
        ),
        encoding="utf-8",
    )

    board = ProductionBoard(cart_dir / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="Add mechanic task",
                item_id="mechanic_router",
                kind="mechanic_code",
                title="Implement Valve Router",
                spec="Build interactive valve routing.",
                proof_spec="Show routing telemetry in filmstrip.",
                target_modules=["scenes/mechanics/valve_router.gd"],
                depends_on=[],
            )
        ],
        raised_by_role="GameplaySystemsEngineer",
    )
    item = board.get("mechanic_router")
    item.failure = "Runtime script error on prior attempt"

    # Seed an archived failed trace to verify postmortem injection
    item_trace_dir = cart_dir / "artifacts" / "work_items" / "mechanic_router"
    item_trace_dir.mkdir(parents=True, exist_ok=True)
    (item_trace_dir / "lead_trace_rev0_failed1.json").write_text(
        json.dumps(
            {
                "archived_failed_index": 1,
                "current_strategy": "Mutate valve state directly from _ready()",
                "turns": [
                    {
                        "turn_index": 0,
                        "specialist_agent": "swe_engineer",
                        "verification_passed": True,
                        "touched_files": ["scenes/mechanics/valve_router.gd"],
                        "summary": "Implemented initial valve_router.gd",
                    },
                    {
                        "turn_index": 1,
                        "specialist_agent": "qa_test_engineer",
                        "verification_passed": False,
                        "touched_files": [],
                        "summary": "FAILED: Counterfactual zero-input check failed because valve started open",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("engine.studio.workers._is_live", lambda _c: True)
    monkeypatch.setattr("engine.studio.context_curator.is_real_genai_client", lambda _c: True)

    def _fake_curate(self, **kwargs: Any) -> CuratedContextPack:
        from google.genai import types

        return CuratedContextPack(
            prompt_block="=== CURATED CONTEXT PACK (INVARIANT FOR REVISION) ===\n- Chunk design:core_loop\n",
            multimodal_parts=[
                "REFERENCE 16:9 CONCEPT ART MOCKUP (mockup_01.png):",
                types.Part.from_bytes(data=mockup_p.read_bytes(), mime_type="image/png"),
            ],
            decision=ContextCuratorDecision(
                deliberation="Selected core loop chunk.",
                role_tailored_synthesis="Core loop synthesis.",
                selected_chunk_ids=["design:core_loop"],
                selected_media_paths=[str(mockup_p)],
            ),
            retrieved_chunks=[],
            selected_media_paths=[str(mockup_p)],
        )

    monkeypatch.setattr("engine.studio.context_curator.ContextCuratorAgent.curate_context", _fake_curate)

    captured_calls: List[Any] = []

    class _RecordingModels:
        def generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
            captured_calls.append(contents)
            dec = TaskLeadDecision(
                deliberation="Delegate next step cleanly.",
                current_strategy="Pivot away from _ready() pre-solve; wire _unhandled_input click toggle.",
                proof_strategy="Capture causal click toggle filmstrip.",
                specialist_agent="swe_engineer",
                brief="Implement unsolved t=0 valve and click handler.",
                target_paths=["scenes/mechanics/valve_router.gd"],
                action="delegate",
            )
            return SimpleNamespace(text=dec.model_dump_json())

    client = SimpleNamespace(models=_RecordingModels())
    ctx = WorkerContext(
        cartridge_dir=cart_dir,
        client=client,
        board=board,
        swe_agent=SimpleNamespace(implement_requirement_track=lambda *a, **kw: {}),
    )

    # Turn 0
    plan_task_lead_turn(ctx, item, turn_history=[], collected_proofs=[])

    # Simulate Turn 0 -> Turn 1: disk changes + old filmstrip exists + SWE edits production file
    old_strip = _make_png(item_trace_dir / "filmstrip_mechanic_router_old.png", (20, 120, 60))
    (cart_dir / "scenes" / "mechanics").mkdir(parents=True, exist_ok=True)
    (cart_dir / "scenes" / "mechanics" / "valve_router.gd").write_text(
        "extends Node2D\nvar is_open: bool = false\n", encoding="utf-8"
    )
    t0_report = SpecialistReport(
        deliberation="SWE updated valve_router.gd",
        turn_index=0,
        specialist_agent="swe_engineer",
        brief="Fix valve",
        summary="Implemented valve_router.gd",
        touched_files=["scenes/mechanics/valve_router.gd"],
        proof_images=[],
        verification_passed=True,
    )
    plan_task_lead_turn(ctx, item, turn_history=[t0_report], collected_proofs=[str(old_strip)])

    # Simulate Turn 1 -> Turn 2: QA captures fresh filmstrip
    new_strip = _make_png(item_trace_dir / "filmstrip_mechanic_router_new.png", (180, 140, 30))
    t1_report = SpecialistReport(
        deliberation="QA captured filmstrip",
        turn_index=1,
        specialist_agent="qa_test_engineer",
        brief="Capture causal filmstrip",
        summary="Captured passing causal filmstrip",
        touched_files=["tests/test_mechanic_router.gd"],
        proof_images=[str(new_strip)],
        verification_passed=True,
    )
    plan_task_lead_turn(
        ctx,
        item,
        turn_history=[t0_report, t1_report],
        collected_proofs=[str(old_strip), str(new_strip)],
    )

    assert len(captured_calls) == 3
    c0, c1, c2 = captured_calls
    assert isinstance(c0, list) and isinstance(c1, list) and isinstance(c2, list)

    # Byte-0 static text prefix + static mockup binary MUST be 100% identical across t0, t1, and t2!
    assert c0[0] == c1[0] == c2[0]
    assert c0[1] == c1[1] == c2[1]
    assert "PRIOR FAILED ATTEMPTS POSTMORTEM FOR THIS REVISION" in c0[0]
    assert "Mutate valve state directly from _ready()" in c0[0]
    assert "[condensed on technical revision]" in c0[0]

    # On Turn 1 (immediately after SWE edited production .gd), stale filmstrip binary is omitted!
    c1_strs = "\n".join(x for x in c1 if isinstance(x, str))
    assert "stale pre-SWE filmstrips omitted" in c1_strs
    assert not any(isinstance(x, str) and "CAPTURED PROOF IMAGE" in x for x in c1)

    # On Turn 2 (after QA re-captured), fresh filmstrip binary is attached at the tail!
    assert any(isinstance(x, str) and "filmstrip_mechanic_router_new.png" in x for x in c2[3:])


def test_council_review_round_shares_single_curate_call_and_identical_byte0_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_run_review_round` executes `curate_context` ONCE per round and passes byte-for-byte identical
    `parts` (including the 16:9 concept art mockup at byte 0) across all 4 Council reviewers."""
    cart_dir = tmp_path / "cartridge"
    cart_dir.mkdir(parents=True, exist_ok=True)
    mockup_p = _make_png(cart_dir / "assets" / "mockups" / "concept.png", (70, 90, 160))
    proof_p = _make_png(cart_dir / "artifacts" / "filmstrip_mech_01.png", (40, 150, 90))

    board = ProductionBoard(cart_dir / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="Add mechanic item",
                item_id="mech_01",
                kind="mechanic_code",
                title="Conduit Pressure Valve",
                spec="Player clicks valve to route steam.",
                proof_spec="Filmstrip proving causal steam routing.",
                target_modules=["scenes/mechanics/valve.gd"],
                depends_on=[],
            )
        ],
        raised_by_role="GameplaySystemsEngineer",
    )
    board.claim_ready()
    board.submit_result(
        "mech_01",
        WorkerResult(proof_images=[str(proof_p)], notes="[swe_engineer] built; [qa_test_engineer] verified"),
    )

    bible = GameBibleSpec(
        deliberation="Test bible",
        game_title="Steam Lattice",
        elevator_pitch="Build pressurized conduit networks.",
        core_genre="Systems Puzzle",
        visual_mockups=[
            VisualMockupArtifact(
                deliberation="Primary mockup",
                mockup_id="m1",
                title="Main Lattice View",
                scene_description="16:9 isometric conduit foundry",
                image_path="assets/mockups/concept.png",
            )
        ],
    )

    curate_calls: List[str] = []
    monkeypatch.setattr("engine.studio.workers._is_live", lambda _c: True)
    monkeypatch.setattr("engine.studio.context_curator.is_real_genai_client", lambda _c: True)

    def _counting_curate(self, **kwargs: Any) -> CuratedContextPack:
        curate_calls.append(str(kwargs.get("target_agent_role")))
        return CuratedContextPack(
            prompt_block="=== SHARED COUNCIL CURATED PACK ===",
            multimodal_parts=[],
            decision=ContextCuratorDecision(
                deliberation="Curated once for all panelists.",
                role_tailored_synthesis="Shared synthesis.",
                selected_chunk_ids=[],
                selected_media_paths=[],
            ),
            retrieved_chunks=[],
            selected_media_paths=[],
        )

    monkeypatch.setattr("engine.studio.context_curator.ContextCuratorAgent.curate_context", _counting_curate)

    panelist_parts: List[List[Any]] = []
    panelist_roles = CreativeWritersRoom._reviewers_for(board.get("mech_01"))
    assert len(panelist_roles) == 4
    assert "VisualArtDirector" in panelist_roles

    room = CreativeWritersRoom(client=SimpleNamespace(), board=board)

    def _fake_structured(prompt: str, schema_cls: Any, validate_fn: Any, **kwargs: Any) -> Any:
        panelist_parts.append(list(kwargs.get("parts") or []))
        return ReviewBatch(
            deliberation="Audited proof carefully.",
            spoken_dialogue=" Telemetry confirms unsolved t=0 and causal click routing.",
            addressed_to=["CreativeDirector"],
            review_decisions=[
                ReviewDecision(
                    deliberation="Meets slice spec.",
                    item_id="mech_01",
                    validation_criteria_for_my_role="Causal input -> pressure delta.",
                    feedback="",
                    verdict="approve",
                )
            ],
            cap_decisions=[],
        )

    monkeypatch.setattr(room, "_structured_with_validation", _fake_structured)
    applied = room._run_review_round(bible, "Player DNA", cart_dir)

    assert applied == 1
    # Only ONE ContextCurator call for the entire 4-panelist review round!
    assert curate_calls == ["council_reviewer"]
    assert len(panelist_parts) == 4
    # Every panelist received the exact same byte-0 `parts` prefix with contiguous UTF-8 text strictly at index 0!
    assert isinstance(panelist_parts[0][0], str)
    assert "=== SHARED COUNCIL CURATED PACK ===" in panelist_parts[0][0]
    for idx in range(1, len(panelist_parts)):
        assert panelist_parts[idx] == panelist_parts[0]
    assert any(isinstance(p, str) and "REFERENCE 16:9 CONCEPT ART MOCKUP" in p for p in panelist_parts[0])


def test_qa_broken_script_archival_surgical_patches_and_plan_filmstrip_byte0_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1. `_archive_and_extract_broken_qa_script` archives broken `tests/test_*.gd` to `last_failed_test_scene.gd`
          with numbered `+/- 12` error-line windows before unlinking.
       2. `validate_filmstrip_plan` accepts surgical `test_scene_patches` (and zero-rewrite reuse) when
          `tests/test_<item>.gd` already exists on disk.
       3. `plan_filmstrip` keeps `contents[0]` byte-for-byte invariant between Pass 1 and Pass 2."""
    cart_dir = tmp_path / "cartridge"
    (cart_dir / "tests").mkdir(parents=True, exist_ok=True)
    (cart_dir / "project.godot").write_text("[application]\nconfig/name=\"QATest\"\n", encoding="utf-8")
    (cart_dir / "main.gd").write_text("extends Node2D\n", encoding="utf-8")

    board = ProductionBoard(cart_dir / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="Add item",
                item_id="mech_x",
                kind="mechanic_code",
                title="Test Mech X",
                spec="Spec X",
                proof_spec="Proof X",
                target_modules=["main.gd"],
                depends_on=[],
            )
        ],
        raised_by_role="GameplaySystemsEngineer",
    )
    w_item = board.get("mech_x")
    item_dir = cart_dir / "artifacts" / "work_items" / "mech_x"

    # 1. Broken QA script archival + error window extraction via `run_qa_test_engineer_specialist`
    broken_lines = [
        "extends Node2D",
        "var _game: Node = null",
        "func _ready() -> void:",
        "\tvar packed := load(\"res://main.tscn\") as PackedScene",
        "\t_game = packed.instantiate()",
        "\tadd_child(_game)",
    ] + [f"var line_{i}: int = {i}" for i in range(7, 45)]
    broken_lines[27] = "var broken_syntax = ("
    broken_plan = FilmstripPlan(
        deliberation="Draft QA script with syntax error at line 28.",
        verification_mode="exploratory_diagnostic",
        target_scene="res://tests/test_mech_x.tscn",
        test_scene_gd_content="\n".join(broken_lines) + "\n",
        council_test_scene_summary="Diagnostic test scene.",
        capture_ticks=[1, 30, 60, 90],
    )

    class _BrokenPlanModels:
        def generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
            return SimpleNamespace(text=broken_plan.model_dump_json())

    from engine.studio.models import StudioAgentError

    monkeypatch.setattr("engine.studio.workers._is_live", lambda _c: True)
    monkeypatch.setattr("engine.studio.context_curator.is_real_genai_client", lambda _c: False)

    def _failing_filmstrip_fn(c_dir: Path, **kw: Any) -> Path:
        raise StudioAgentError(
            "Godot headless smoke check failed in res://tests/test_mech_x.gd:28: Parse Error"
        )

    broken_ctx = WorkerContext(
        cartridge_dir=cart_dir,
        client=SimpleNamespace(models=_BrokenPlanModels()),
        board=board,
        filmstrip_fn=_failing_filmstrip_fn,
    )
    with pytest.raises(StudioAgentError) as exc_info:
        run_qa_test_engineer_specialist(broken_ctx, w_item, brief="Run exploratory diagnostic", target_paths=[])

    qa_gd = cart_dir / "tests" / "test_mech_x.gd"
    assert not qa_gd.exists()
    assert (item_dir / "last_failed_test_scene.gd").is_file()
    assert "BROKEN QA SCRIPT WINDOW (`res://tests/test_mech_x.gd`" in str(exc_info.value)
    assert "  28: var broken_syntax = (" in str(exc_info.value)

    # 2. Surgical QA patch validation + zero-rewrite reuse when valid test script is on disk
    valid_qa_gd = (
        "extends Node2D\n"
        "var _game: Node = null\n"
        "func _ready() -> void:\n"
        "\tvar packed := load(\"res://main.tscn\") as PackedScene\n"
        "\t_game = packed.instantiate()\n"
        "\tadd_child(_game)\n"
        "func get_state_snapshot() -> Dictionary:\n"
        "\treturn {\"phase\": \"old\"}\n"
    )
    qa_gd.write_text(valid_qa_gd, encoding="utf-8")

    reuse_plan = FilmstripPlan(
        deliberation="Reuse valid on-disk QA scene unchanged and update input segments.",
        verification_mode="exploratory_diagnostic",
        target_scene="res://tests/test_mech_x.tscn",
        test_scene_gd_content="",
        test_scene_patches=[],
        council_test_scene_summary="Reuses existing diagnostic harness.",
        capture_ticks=[1, 30, 60, 90],
    )
    assert validate_filmstrip_plan(reuse_plan, actions=[], existing_test_scene_on_disk=True) == []

    patch_plan = FilmstripPlan(
        deliberation="Patch snapshot dictionary in existing QA test script.",
        verification_mode="exploratory_diagnostic",
        target_scene="res://tests/test_mech_x.tscn",
        test_scene_gd_content="",
        test_scene_patches=[
            SurgicalCodePatch(
                deliberation="Add tick telemetry key.",
                filename="tests/test_mech_x.gd",
                start_line=8,
                end_line=8,
                replacement_code='\treturn {"phase": "patched"}',
            )
        ],
        council_test_scene_summary="Patched telemetry readout.",
        capture_ticks=[1, 30, 60, 90],
    )
    assert validate_filmstrip_plan(patch_plan, actions=[], existing_test_scene_on_disk=True) == []

    # 3. `plan_filmstrip` Pass 1 vs Pass 2 byte-0 static prefix invariance
    qa_mockup = _make_png(cart_dir / "assets" / "mockups" / "qa_ref.png", (60, 80, 120))
    monkeypatch.setattr("engine.studio.context_curator.is_real_genai_client", lambda _c: True)
    monkeypatch.setattr("engine.studio.workers.is_real_genai_client", lambda _c: True)

    def _qa_curate(self, **kw: Any) -> CuratedContextPack:
        from google.genai import types

        return CuratedContextPack(
            prompt_block="=== STATIC QA CURATED PACK ===",
            multimodal_parts=[
                "REFERENCE 16:9 CONCEPT ART MOCKUP (qa_ref.png):",
                types.Part.from_bytes(data=qa_mockup.read_bytes(), mime_type="image/png"),
            ],
            decision=ContextCuratorDecision(
                deliberation="QA pack",
                role_tailored_synthesis="QA synthesis",
                selected_chunk_ids=[],
                selected_media_paths=[str(qa_mockup)],
            ),
            retrieved_chunks=[],
            selected_media_paths=[str(qa_mockup)],
        )

    monkeypatch.setattr("engine.studio.context_curator.ContextCuratorAgent.curate_context", _qa_curate)

    qa_calls: List[Any] = []

    class _QAPlanModels:
        def generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
            qa_calls.append(contents)
            return SimpleNamespace(text=reuse_plan.model_dump_json())

    ctx = WorkerContext(cartridge_dir=cart_dir, client=SimpleNamespace(models=_QAPlanModels()), board=board)
    pass1_img = _make_png(item_dir / "filmstrip_pass1.png", (100, 120, 140))

    plan_filmstrip(ctx, w_item)
    plan_filmstrip(
        ctx,
        w_item,
        prior_capture_feedback="Observed entity at (320, 240); assertion failed at tick 90",
        prior_strip_path=pass1_img,
    )

    assert len(qa_calls) == 2
    assert isinstance(qa_calls[0], list) and isinstance(qa_calls[1], list)
    # Exact byte-0 prefix + static mockup binary match between Pass 1 and Pass 2!
    assert qa_calls[0][0] == qa_calls[1][0]
    assert qa_calls[0][1] == qa_calls[1][1]
    assert qa_calls[0][2] == qa_calls[1][2]
    assert "EXISTING ON-DISK QA TEST SCENE (`res://tests/test_mech_x.gd`" in qa_calls[0][3]
    # Pass 1 filmstrip is attached only at the tail of Pass 2!
    assert len(qa_calls[0]) == 4
    assert len(qa_calls[1]) == 6
    assert "PASS 1 CAPTURED FILMSTRIP THAT FAILED STATE ASSERTIONS" in qa_calls[1][4]


def test_token_usage_tracker_unified_usd_media_embedding_and_playtest_accounting(tmp_path: Path) -> None:
    """`TokenUsageTracker` accounts for cached vs uncached text tokens, `generate_images`,
    `embed_content`, TTS audio bytes, and unguarded `BlindPlaytesterAgent` turns, plus
    `PlaytestSessionConfig.compaction_interval_turns` defaults to 75."""
    cfg = PlaytestSessionConfig()
    assert cfg.compaction_interval_turns == 75

    tracker = TokenUsageTracker()

    # 1. Text call with 80% cache hit: 20,000 prompt (16,000 cached, 4,000 uncached) + 1,000 output + 1,000 thought
    text_resp = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=20_000,
            cached_content_token_count=16_000,
            candidates_token_count=1_000,
            thoughts_token_count=1_000,
            total_token_count=22_000,
        )
    )
    s1 = tracker.record_response(text_resp, label="TaskLead.plan_turn[item_1]")
    assert s1 is not None
    # Expected 2026 cost: 4k * $0.75/1M ($0.003) + 16k * $0.075/1M ($0.0012) + 2k * $3.75/1M ($0.0075) = $0.0117
    assert s1["estimated_call_cost_usd"] == pytest.approx(0.0117, abs=1e-5)

    # 2. Image synthesis response (`generated_images` without `usage_metadata`, 2 * $0.039 = $0.078)
    img_resp = SimpleNamespace(generated_images=[SimpleNamespace(), SimpleNamespace()], usage_metadata=None)
    s2 = tracker.record_response(img_resp, label="AssetFabricator.generate_images[hero]", method_name="generate_images")
    assert s2 is not None
    assert s2["image_count"] == 2
    assert s2["estimated_call_cost_usd"] == pytest.approx(0.078, abs=1e-5)

    # 3. Embedding response (`embeddings` without `usage_metadata`, 4,000 chars -> ~1,000 tokens)
    emb_resp = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])], usage_metadata=None)
    s3 = tracker.record_response(
        emb_resp,
        label="embed_content[batch=1]",
        input_chars_hint=4_000,
        method_name="embed_content",
    )
    assert s3 is not None
    assert s3["embedding_estimated_tokens"] == 1_000

    # 4. TTS inline audio response (48,000 * 60 bytes -> 60.0s audio @ $0.015/min = $0.015)
    tts_resp = SimpleNamespace(
        usage_metadata=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            inline_data=SimpleNamespace(mime_type="audio/pcm", data=b"\x00" * (48_000 * 60))
                        )
                    ]
                )
            )
        ],
    )
    s4 = tracker.record_response(tts_resp, label="GeminiAudioBackend.tts[bark_01]")
    assert s4 is not None
    assert s4["tts_audio_seconds"] == pytest.approx(60.0, abs=1e-3)
    assert s4["estimated_call_cost_usd"] == pytest.approx(0.015, abs=1e-4)

    summary = tracker.get_summary()
    assert summary["total_calls"] == 4
    assert summary["image_generations"] == 2
    assert summary["tts_generations"] == 1
    assert summary["embedding_calls"] == 1
    assert summary["embedding_estimated_tokens"] == 1_000
    assert summary["cached_token_share_pct"] == 80.0
    assert summary["estimated_cost_usd"] == pytest.approx(0.0117 + 0.078 + 0.00015 + 0.015, abs=1e-4)
    assert summary["estimated_cost_usd_2027_std"] > summary["estimated_cost_usd"]

    # 5. Verify BlindPlaytesterAgent mirrors unguarded client calls into global TOKEN_USAGE_TRACKER
    #    playtest counters + USD while keeping `total_calls` isolated to Studio build pipeline calls!
    from engine.studio.models import TOKEN_USAGE_TRACKER

    before_calls = TOKEN_USAGE_TRACKER.total_calls
    before_playtest_calls = TOKEN_USAGE_TRACKER.playtest_calls
    obs_png = _make_png(tmp_path / "obs.png")
    turn_json = {
        "deliberation": "Inspect screen.",
        "turn_log_entry": "Observed title screen.",
        "field_note_updates": [],
        "question_updates": [],
        "rulebook_updates": [],
        "bug_reports": [],
        "impression_updates": [],
        "current_mood": "curious",
        "mood_shift": None,
        "current_plan": "Click start button.",
        "motor_action": {
            "deliberation": "Move cursor to center.",
            "hold_keys": [],
            "tap_keys": [],
            "mouse_x": 500,
            "mouse_y": 500,
            "drag_from_x": None,
            "drag_from_y": None,
            "mouse_button_action": "left_click",
            "duration_ms": 500,
            "inspect_high_res_next": False,
            "request_audio_evaluation": None,
        },
        "submit_wish": None,
        "end_playtest": None,
    }

    class _UnguardedPlaytestClient:
        class models:
            @staticmethod
            def generate_content(model: str, contents: Any, config: Any = None) -> Any:
                return SimpleNamespace(
                    text=json.dumps(turn_json),
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=4_500,
                        cached_content_token_count=4_100,
                        candidates_token_count=220,
                        thoughts_token_count=0,
                        total_token_count=4_720,
                    ),
                )

    agent = BlindPlaytesterAgent(
        store_card=BlindStoreCard(
            deliberation="Card",
            checkpoint_id="cp_1",
            game_title="Signal Relay",
            game_description="Connect relays.",
        ),
        client=_UnguardedPlaytestClient(),
    )
    agent.decide_turn(
        1,
        TurnObservation(
            observation_index=1,
            tick=15,
            sim_time_sec=0.25,
            cursor_x=640.0,
            cursor_y=360.0,
            screenshot_path=obs_png,
            preview_path=obs_png,
            wav_path=None,
            audio_rms=0.0,
            audio_peak=0.0,
            audio_is_silent=True,
            interrupted_early=False,
        ),
    )
    assert TOKEN_USAGE_TRACKER.total_calls == before_calls
    assert TOKEN_USAGE_TRACKER.playtest_calls == before_playtest_calls + 1
    assert TOKEN_USAGE_TRACKER.get_summary()["playtest_estimated_cost_usd"] > 0.0


def test_visual_evaluation_report_coerces_sanitized_json_string_dicts() -> None:
    """`_sanitize_gemini_json_schema` converts `Dict[str, str]` into a JSON-encoded string field for
    the Interactions API; `VisualEvaluationReport` must parse JSON strings, fenced JSON strings, and
    list-of-entry payloads without raising a Pydantic ValidationError."""
    from engine.studio.models import VisualEvaluationReport, _sanitize_gemini_json_schema

    schema = _sanitize_gemini_json_schema(VisualEvaluationReport.model_json_schema())
    assert schema["properties"]["perceptual_descriptions_by_path"]["type"] == "string"

    raw_json_str_payload = json.dumps(
        {
            "deliberation": "Inspected 2x2 sprite sheet alpha and cell bounds cleanly.",
            "observed_visual_properties": "2x2 grid on transparent RGBA background.",
            "perceptual_descriptions_by_path": json.dumps(
                {"assets/sprites/forge_rig.png": "2x2 state sheet: idle, warming, active, cooldown."}
            ),
            "answers_brief_satisfactorily": True,
        }
    )
    rep1 = VisualEvaluationReport.model_validate_json(raw_json_str_payload)
    assert rep1.perceptual_descriptions_by_path == {
        "assets/sprites/forge_rig.png": "2x2 state sheet: idle, warming, active, cooldown."
    }

    fenced_list_payload = json.dumps(
        {
            "deliberation": "Inspected multi-asset contact plate.",
            "observed_visual_properties": "Crisp silhouettes.",
            "perceptual_descriptions_by_path": (
                "```json\n"
                + json.dumps([{"path": "assets/sprites/gear.png", "description": "Brass cog icon"}])
                + "\n```"
            ),
            "answers_brief_satisfactorily": True,
        }
    )
    rep2 = VisualEvaluationReport.model_validate_json(fenced_list_payload)
    assert rep2.perceptual_descriptions_by_path == {"assets/sprites/gear.png": "Brass cog icon"}


def test_task_lead_revision_prefix_cache_survives_target_modules_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `item.target_modules` expands between Turn 0 and Turn 1 of the same revision, passing
    `_revision_prefix_cache` keeps `static_lead_prefix_text` 100% byte-identical and runs `curate_context`
    only once while reporting `UPDATED LIVE TARGET MODULES ON BOARD` in the dynamic tail."""
    cart_dir = tmp_path / "cartridge"
    cart_dir.mkdir(parents=True, exist_ok=True)
    mockup_p = _make_png(cart_dir / "assets" / "mockups" / "concept.png", (50, 80, 130))

    board = ProductionBoard(cart_dir / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="Add mechanic task",
                item_id="mech_expand",
                kind="mechanic_code",
                title="Relay Coupling",
                spec="Wire relay signal to receiver.",
                proof_spec="Filmstrip showing signal coupling.",
                target_modules=["scenes/mechanics/relay.gd"],
                depends_on=[],
            )
        ],
        raised_by_role="GameplaySystemsEngineer",
    )
    item = board.get("mech_expand")

    curate_count = 0
    monkeypatch.setattr("engine.studio.workers._is_live", lambda _c: True)
    monkeypatch.setattr("engine.studio.context_curator.is_real_genai_client", lambda _c: True)

    def _fake_curate(self, **kwargs: Any) -> CuratedContextPack:
        nonlocal curate_count
        curate_count += 1
        from google.genai import types

        return CuratedContextPack(
            prompt_block="=== FROZEN LEAD CURATED CONTEXT ===",
            multimodal_parts=[
                "CONCEPT MOCKUP:",
                types.Part.from_bytes(data=mockup_p.read_bytes(), mime_type="image/png"),
            ],
            decision=ContextCuratorDecision(
                deliberation="Curated once for revision.",
                role_tailored_synthesis="Relay overview.",
                selected_chunk_ids=[],
                selected_media_paths=[],
            ),
            retrieved_chunks=[],
            selected_media_paths=[],
        )

    monkeypatch.setattr("engine.studio.context_curator.ContextCuratorAgent.curate_context", _fake_curate)

    captured_contents: List[Any] = []

    class _FakeModels:
        def generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
            captured_contents.append(contents)
            dec = TaskLeadDecision(
                deliberation="Delegate next step.",
                current_strategy="Implement and verify relay.",
                proof_strategy="Capture filmstrip.",
                specialist_agent="qa_test_engineer",
                brief="Capture causal relay filmstrip.",
                target_paths=[],
                action="delegate",
            )
            return SimpleNamespace(text=dec.model_dump_json())

    (cart_dir / "project.godot").write_text("[application]\nconfig/name=\"PrefixCacheTest\"\n", encoding="utf-8")
    ctx = WorkerContext(
        cartridge_dir=cart_dir,
        client=SimpleNamespace(models=_FakeModels()),
        board=board,
        swe_agent=SimpleNamespace(implement_requirement_track=lambda *a, **kw: {}),
        filmstrip_fn=lambda *a, **kw: cart_dir / "dummy.png",
    )

    rev_cache: dict[str, Any] = {}
    plan_task_lead_turn(ctx, item, turn_history=[], collected_proofs=[], _revision_prefix_cache=rev_cache)

    # Simulate SWE expanding `item.target_modules` during Turn 0 execution
    item.target_modules = ["scenes/mechanics/relay.gd", "scenes/entities/receiver.gd", "main.gd"]
    t0_rep = SpecialistReport(
        deliberation="Wired relay and receiver.",
        turn_index=0,
        specialist_agent="swe_engineer",
        brief="Wire relay",
        summary="Patched relay.gd and receiver.gd",
        touched_files=["scenes/mechanics/relay.gd", "scenes/entities/receiver.gd"],
        proof_images=[],
        verification_passed=True,
    )
    plan_task_lead_turn(ctx, item, turn_history=[t0_rep], collected_proofs=[], _revision_prefix_cache=rev_cache)

    assert curate_count == 1
    assert len(captured_contents) == 2
    t0_parts, t1_parts = captured_contents
    assert t0_parts[0] == t1_parts[0]
    assert t0_parts[1] == t1_parts[1]
    assert "UPDATED LIVE TARGET MODULES ON BOARD: ['scenes/mechanics/relay.gd', 'scenes/entities/receiver.gd', 'main.gd']" in t1_parts[3]


def test_swe_bounds_hub_fanout_and_dedupes_curated_code_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a target hub script preloads 20 unrelated leaf scripts, `implement_requirement_track` bounds
    non-target graph fan-out and omits duplicate `[code:module:...]` chunks already shown in staging code."""
    from engine.studio.context_curator import ContextChunk
    from engine.studio.models import RequirementTrack, SurgicalCodePatch, TrackCodeSynthesisPayload
    from engine.studio.swe import LeadGameplaySWEAgent

    cart_dir = tmp_path / "cartridge"
    (cart_dir / "scenes" / "world").mkdir(parents=True, exist_ok=True)
    (cart_dir / "scenes" / "mechanics").mkdir(parents=True, exist_ok=True)
    (cart_dir / "project.godot").write_text("[application]\nconfig/name=\"Test\"\n", encoding="utf-8")

    preload_lines = []
    for idx in range(20):
        leaf_rel = f"scenes/world/leaf_mod_{idx:02d}.gd"
        leaf_body = "extends Node2D\n" + "\n".join(f"var prop_{idx}_{k}: int = {k}" for k in range(45)) + "\n"
        (cart_dir / leaf_rel).write_text(leaf_body, encoding="utf-8")
        preload_lines.append(f"const Leaf{idx:02d} = preload('res://{leaf_rel}')")

    hub_code = "extends Node2D\n" + "\n".join(preload_lines) + "\nfunc build_world() -> void:\n\tpass\n"
    (cart_dir / "scenes" / "world" / "world_hub.gd").write_text(hub_code, encoding="utf-8")
    target_mech = "extends Node2D\n" + "\n".join(f"var step_{k}: int = {k}" for k in range(45)) + "\n"
    (cart_dir / "scenes" / "mechanics" / "core_mech.gd").write_text(target_mech, encoding="utf-8")

    monkeypatch.setattr("engine.studio.context_curator.is_real_genai_client", lambda _c: True)

    def _fake_curate(self, **kwargs: Any) -> CuratedContextPack:
        dup_chunk = ContextChunk(
            chunk_id="code:module:scenes/mechanics/core_mech.gd",
            category="code_module",
            title="Code Module scenes/mechanics/core_mech.gd",
            content="DUPLICATE_CORE_MECH_CONTRACT_MARKER",
        )
        design_chunk = ContextChunk(
            chunk_id="design:core_loop",
            category="design_spec",
            title="Core Loop",
            content="CORE_LOOP_DESIGN_MARKER",
        )
        return CuratedContextPack(
            prompt_block="[code:module:scenes/mechanics/core_mech.gd]\nDUPLICATE_CORE_MECH_CONTRACT_MARKER\n[design:core_loop]\nCORE_LOOP_DESIGN_MARKER",
            multimodal_parts=[],
            decision=ContextCuratorDecision(
                deliberation="Selected chunks.",
                role_tailored_synthesis="Synthesis map.",
                selected_chunk_ids=[dup_chunk.chunk_id, design_chunk.chunk_id],
                selected_media_paths=[],
            ),
            retrieved_chunks=[dup_chunk, design_chunk],
            selected_media_paths=[],
        )

    monkeypatch.setattr("engine.studio.context_curator.ContextCuratorAgent.curate_context", _fake_curate)
    monkeypatch.setattr(
        TrackCodeSynthesisPayload,
        "assemble_cartridge_directory",
        lambda self, *a, **kw: {},
    )

    captured_swe_prompts: List[str] = []

    class _FakeSWEModels:
        def generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
            prompt_str = contents if isinstance(contents, str) else "\n".join(str(x) for x in contents if isinstance(x, str))
            captured_swe_prompts.append(prompt_str)
            payload = TrackCodeSynthesisPayload(
                deliberation="Apply surgical edit to core_mech.gd.",
                patches=[
                    SurgicalCodePatch(
                        deliberation="Update step_0",
                        filename="scenes/mechanics/core_mech.gd",
                        start_line=2,
                        end_line=2,
                        target_snippet="var step_0: int = 0",
                        replacement_code="var step_0: int = 1",
                    )
                ],
                files=[],
            )
            return SimpleNamespace(text=payload.model_dump_json())

    swe = LeadGameplaySWEAgent(client=SimpleNamespace(models=_FakeSWEModels()))
    track = RequirementTrack(
        deliberation="Implement core_mech",
        id="mech_track",
        title="Core Mechanic",
        description="Update step_0",
        target_modules=["scenes/mechanics/core_mech.gd", "scenes/world/world_hub.gd"],
        acceptance_criteria=[],
    )
    swe.implement_requirement_track(track=track, staging_dir=cart_dir)

    assert len(captured_swe_prompts) == 1
    p_txt = captured_swe_prompts[0]
    assert "CORE_LOOP_DESIGN_MARKER" in p_txt
    assert "DUPLICATE_CORE_MECH_CONTRACT_MARKER" not in p_txt
    leaf_sig_hits = sum(1 for idx in range(20) if f"scenes/world/leaf_mod_{idx:02d}.gd (Cross-Module API Signature" in p_txt)
    assert leaf_sig_hits <= 8


def test_visual_sprite_synthesis_records_explicit_label_in_token_tracker(tmp_path: Path) -> None:
    """`_generate_gemini_sprite_image` routes through `call_gemini_with_transient_retry` with an explicit
    `synthesize_sprite_image` label so image calls do not collapse under `genai.models.generate_content`."""
    import io
    from PIL import Image
    from engine.studio.models import TOKEN_USAGE_TRACKER
    from engine.studio.visual import VisualArtSpriteDesignerAgent

    buf = io.BytesIO()
    img = Image.new("RGBA", (64, 64), (255, 0, 255, 255))
    for y in range(20, 44):
        for x in range(20, 44):
            img.putpixel((x, y), (40, 120, 200, 255))
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    class _FakeImgModels:
        def generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
            part = SimpleNamespace(inline_data=SimpleNamespace(data=png_bytes, mime_type="image/png"))
            cand = SimpleNamespace(content=SimpleNamespace(parts=[part]))
            return SimpleNamespace(
                candidates=[cand],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=120,
                    cached_content_token_count=0,
                    candidates_token_count=1290,
                    thoughts_token_count=0,
                    total_token_count=1410,
                ),
            )

    before_lbl = dict(TOKEN_USAGE_TRACKER.get_summary().get("by_label", {}).get("synthesize_sprite_image", {}))
    before_calls = int(before_lbl.get("calls", 0))
    before_imgs = int(before_lbl.get("image_generations", 0))

    designer = VisualArtSpriteDesignerAgent(client=SimpleNamespace(models=_FakeImgModels()))
    out_img = designer._generate_gemini_sprite_image("Brass relay valve", asset_type="sprite", dimensions=(64, 64))
    assert out_img.size == (64, 64)

    after_lbl = TOKEN_USAGE_TRACKER.get_summary()["by_label"]["synthesize_sprite_image"]
    assert after_lbl["calls"] == before_calls + 1
    assert after_lbl["image_generations"] == before_imgs + 1
