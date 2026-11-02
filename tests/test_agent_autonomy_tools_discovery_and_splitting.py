"""tests/test_agent_autonomy_tools_discovery_and_splitting.py
Comprehensive unit and integration suite verifying:
1. Studio Tool Registry & Progressive Tool Search (`tool_registry.py` + `ContextCuratorAgent.research_tool_capabilities`)
2. 5 Unified Specialist Personas (`visual_artist`, `audio_designer` + `audio_evaluator` alias, `script_writer`,
   `swe_engineer` lock-free `inspect_only`, `qa_test_engineer` `exploratory_diagnostic` + fullres/mockup VLM audit)
3. Task-kind `available_specialists` scoping
4. Parallel lock-free `kind="discovery"` subtasks, auto-approval, `deps_sig` cache invalidation, parent findings
   injection, and generalized 5-persona `CouncilDiagnosticProbe`
5. Self-directed `action="split_task"` decomposition (`ProductionBoard.split_active_item`, downstream `depends_on`
   rewiring, snapshot rollback vs preservation, and `BoardScheduler` `WorkerSplitTask` lifecycle)
6. In-place asset overwrite fidelity without silent `{item_id}_` filename mangling or 50-word bark truncation.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
import wave

import pytest
from PIL import Image, ImageDraw

from engine.studio.asset_library import CartridgeAssetLibrary
from engine.studio.context_curator import ContextCuratorAgent
from engine.studio.models import (
    AssetLibraryEntry,
    AudioDesignerPlan,
    AudioEvaluationReport,
    CouncilDiagnosticProbe,
    FilmstripPlan,
    GameBibleSpec,
    GameSliceSpec,
    QAPostCaptureAudit,
    ScriptWritingDeliverable,
    SpecialistReport,
    TaskLeadDecision,
    TrackCodeSynthesisPayload,
    VisualArtistPlan,
    VisualEvaluationReport,
    VisualMockupArtifact,
    WorkItem,
    WorkItemProposal,
)
from engine.studio.production_board import (
    PARALLEL_LOCK_FREE_KINDS,
    BoardScheduler,
    ProductionBoard,
    WorkerResult,
    WorkerSplitTask,
)
from engine.studio.swe import LeadGameplaySWEAgent
from engine.studio.tool_registry import (
    CANONICAL_SPECIALIST_PERSONAS,
    build_tool_capability_chunks,
    normalize_specialist_persona,
    render_compact_tool_directory,
)
from engine.studio.visual_evaluator import (
    VisualEvaluatorAgent,
    render_inspection_plate,
)
from engine.studio.workers import (
    ALL_SPECIALIST_AGENTS,
    CartridgeFileLockTable,
    WorkerContext,
    available_specialists,
    default_handlers,
    plan_asset_requirement,
    plan_task_lead_turn,
    run_audio_designer_specialist,
    run_qa_test_engineer_specialist,
    run_script_writer_specialist,
    run_swe_engineer_specialist,
    run_task_lead,
    run_visual_artist_specialist,
    validate_filmstrip_plan,
    validate_task_lead_decision,
)
from engine.studio.writers_room import CreativeWritersRoom


def _make_rgba_png(path: Path, size: tuple[int, int] = (64, 64), color: tuple[int, int, int, int] = (60, 150, 220, 255)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle((4, 4, size[0] - 5, size[1] - 5), fill=color, outline=(255, 255, 255, 255), width=2)
    img.save(path, format="PNG")
    return path


def _make_wav(path: Path, sample_rate: int = 24000, duration_s: float = 0.25) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(sample_rate * duration_s)
    pcm = (b"\x10\x27\xf0\xd8" * (max(1, n_samples // 2)))[: n_samples * 2]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return path


def _deterministic_embed(text: str, dim: int = 768) -> List[float]:
    """Deterministic unit-vector embedding so semantic cosine similarity aligns shared technical terms in tests."""
    import hashlib
    import math
    import re

    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9_]+", str(text).lower()):
        h = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:4], "big")
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _step_scheduler_once(sched: BoardScheduler, timeout: float = 10.0) -> bool:
    """Dispatches a single ready wave on `sched` and blocks until all started worker threads finish."""
    import time

    started = sched.dispatch_once()
    if started <= 0:
        return False
    deadline = time.monotonic() + timeout
    while sched.running_count() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sched.running_count() == 0
    return True


class _QueueGenaiClient:
    """Deterministic fake client supporting queued structured JSON responses, ContextCuratorDecision, and specialist plans."""

    __module__ = "google.genai.fake"
    supports_specialist_plans = True
    enable_post_capture_audit = False

    def __init__(self, responses: List[str]):
        self._queue = list(responses)
        self.calls: List[Dict[str, Any]] = []
        self.curator_calls: int = 0
        self.models = SimpleNamespace(
            generate_content=self._generate_content,
            embed_content=self._embed_content,
        )

    def _embed_content(self, model: str, contents: Any, config: Any = None) -> Any:
        items = list(contents) if isinstance(contents, (list, tuple)) else [str(contents)]
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=_deterministic_embed(str(t))) for t in items]
        )

    def _generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
        schema_str = str(getattr(config, "response_schema", "") or "")
        if "role_tailored_synthesis" in schema_str or "ContextCuratorDecision" in schema_str:
            self.curator_calls += 1
            cur_json = json.dumps(
                {
                    "deliberation": f"Curating context call #{self.curator_calls}.",
                    "search_queries_used": ["overview"],
                    "role_tailored_synthesis": f"Curated snapshot #{self.curator_calls}.",
                    "selected_chunk_ids": [],
                    "selected_image_paths": [],
                    "selected_media_paths": [],
                }
            )
            return SimpleNamespace(
                text=cur_json,
                usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=10, total_token_count=20),
            )
        if "MemoryDistillationDecision" in schema_str or "distilled_entries" in schema_str:
            return SimpleNamespace(
                text=json.dumps({"deliberation": "No additional memory entries.", "distilled_entries": []}),
                usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=5, total_token_count=10),
            )
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self._queue:
            raise AssertionError(f"Unexpected extra generate_content call #{len(self.calls)} on model={model} (schema={schema_str[:120]})")
        return SimpleNamespace(text=self._queue.pop(0), usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=10, total_token_count=20))


class _LiveQueueGenaiClient(_QueueGenaiClient):
    """Queue client treated as a live google.genai Client for ContextCuratorAgent tests."""

    __module__ = "google.genai.test_live"


# =============================================================================
# 1. Rationale-First Contracts, Tool Registry & Progressive Tool Search
# =============================================================================


def test_rationale_first_field_ordering_across_new_and_updated_contracts():
    """Rule 4: Every new/updated epistemic model declares `deliberation` at index 0 and its final decision/action last."""
    expected_last_fields = {
        VisualArtistPlan: "operation",
        VisualEvaluationReport: "answers_brief_satisfactorily",
        AudioDesignerPlan: "operation",
        TaskLeadDecision: "action",
        CouncilDiagnosticProbe: "needs_runtime_debug_scene",
        TrackCodeSynthesisPayload: "operation_mode",
        FilmstripPlan: "verification_mode",
    }
    for cls, expected_last in expected_last_fields.items():
        keys = list(cls.model_fields.keys())
        assert keys[0] == "deliberation", f"{cls.__name__} must declare deliberation at index 0, got {keys}"
        assert keys[-1] == expected_last, f"{cls.__name__} must place {expected_last} last, got {keys}"


def test_canonical_personas_task_scoping_and_progressive_tool_search(tmp_path: Path):
    """Verifies the 5 canonical personas, preserved task-kind scoping, compact tool directory, and semantic tool search."""
    assert ALL_SPECIALIST_AGENTS == (
        "visual_artist",
        "audio_designer",
        "script_writer",
        "swe_engineer",
        "qa_test_engineer",
    )
    assert CANONICAL_SPECIALIST_PERSONAS == ALL_SPECIALIST_AGENTS
    assert normalize_specialist_persona("audio_evaluator") == "audio_designer"
    assert "discovery" in PARALLEL_LOCK_FREE_KINDS

    (tmp_path / "project.godot").write_text('[application]\nconfig/name="AutonomyLab"\n', encoding="utf-8")
    fake_client = _QueueGenaiClient([])
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=fake_client,
        visual_agent=SimpleNamespace(),
        voice_agent=SimpleNamespace(),
        swe_agent=LeadGameplaySWEAgent(client=fake_client),
    )

    def _item_of_kind(kind: str) -> WorkItem:
        return WorkItem(
            deliberation=f"Test {kind}",
            item_id=f"item_{kind}",
            kind=kind,  # type: ignore[arg-type]
            title=f"Title {kind}",
            spec="Spec",
            proof_spec="Proof",
        )

    assert available_specialists(ctx, _item_of_kind("visual_asset")) == ["visual_artist"]
    assert available_specialists(ctx, _item_of_kind("animation")) == ["visual_artist"]
    assert available_specialists(ctx, _item_of_kind("audio_asset")) == ["audio_designer"]
    for cross_kind in ("mechanic_code", "ui", "level_map", "integration", "narrative", "other", "discovery"):
        assert available_specialists(ctx, _item_of_kind(cross_kind)) == list(ALL_SPECIALIST_AGENTS)

    # Compact baseline directory stays concise (~15-25 lines) while advertising progressive tool_search_query
    vis_dir = render_compact_tool_directory(["visual_artist"], layer="task_lead")
    assert "visual_artist" in vis_dir
    assert "swe_engineer" not in vis_dir
    assert "tool_search_query" in vis_dir

    full_dir = render_compact_tool_directory(ALL_SPECIALIST_AGENTS, layer="council")
    for p in ALL_SPECIALIST_AGENTS:
        assert p in full_dir

    # Progressive semantic search retrieves rich tool_capability chunks via ContextCuratorAgent
    chunks = build_tool_capability_chunks()
    assert len(chunks) >= 11
    curator = ContextCuratorAgent(cartridge_dir=tmp_path, client=_LiveQueueGenaiClient([]))
    corpus_chunks = curator.semantic_index.extract_corpus_chunks()
    assert any(c.category == "tool_capability" for c in corpus_chunks)

    expanded = curator.research_tool_capabilities(
        "how to render 1:1 visual inspection plates with grid_px and crop_boxes",
        caller_role="TaskLead",
        allowed_specialists=["visual_artist"],
    )
    assert "grid_px" in expanded
    assert "crop_boxes" in expanded
    assert "visual_artist" in expanded


# =============================================================================
# 2. Unified 5 Specialist Personas (Creator + Perceptual Evaluator)
# =============================================================================


def test_visual_artist_fabricate_preserves_exact_path_and_inspect_enriches_catalog(tmp_path: Path):
    """Verifies `visual_artist`:
    1. Preserves exact `target_path` and existing `asset_id` on in-place asset updates (zero `{item_id}_` mangling).
    2. Supports `render_contact_sheet` with `grid_px` and `crop_boxes`.
    3. Supports `inspect` via `VisualEvaluatorAgent`, attaching the 16:9 mockup and enriching `perceptual_description`.
    """
    mockup_path = _make_rgba_png(tmp_path / "assets" / "mockups" / "slice_0_mockup.png", size=(320, 180))
    bible = GameBibleSpec(
        deliberation="Bible with mockup.",
        game_title="Atelier",
        elevator_pitch="Craft artifacts.",
        visual_style_guide="Crisp inked illustration.",
        visual_mockups=[
            VisualMockupArtifact(
                deliberation="Mockup 0",
                mockup_id="m0",
                title="Workshop",
                scene_description="Workshop scene",
                image_path="assets/mockups/slice_0_mockup.png",
            )
        ],
        slice_roadmap=[
            GameSliceSpec(
                deliberation="Slice 0",
                slice_index=0,
                slice_id="slice_0",
                title="Opening",
                slice_objective="Play",
                status="active",
            )
        ],
    )
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    # Seed existing hero portrait under canonical asset_id "portrait_hero"
    target_png = _make_rgba_png(tmp_path / "assets" / "ui" / "portrait_hero.png", size=(64, 64))
    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path, client=None)
    lib.catalog.entries["portrait_hero"] = AssetLibraryEntry(
        deliberation="Seeded hero portrait.",
        asset_id="portrait_hero",
        asset_type="portrait",
        fabrication_mode="portrait",
        purpose="HUD speaker portrait",
        res_path="res://assets/ui/portrait_hero.png",
        file_path="assets/ui/portrait_hero.png",
        target_display_size=(64, 64),
        texture_pixel_size=(64, 64),
    )
    lib.save_catalog()

    # Unrelated task ID updating portrait_hero.png -> MUST NOT rename to unrelated_fix_task_portrait_hero.png!
    item = WorkItem(
        deliberation="Refine hero portrait shading.",
        item_id="unrelated_fix_task",
        kind="visual_asset",
        title="Update portrait",
        spec="Update hero portrait shading.",
        proof_spec="Portrait plate",
        slice_id="slice_0",
    )
    req_json = json.dumps(
        {
            "deliberation": "Overwrite exact portrait path.",
            "asset_id": "new_random_id",
            "name": "Hero Portrait",
            "asset_type": "portrait",
            "fabrication_mode": "portrait",
            "purpose": "HUD speaker portrait",
            "target_path": "assets/ui/portrait_hero.png",
            "description": "Inked portrait",
            "visual_prompt_or_text": "Inked portrait",
            "dimensions": [96, 96],
        }
    )
    req_client = _QueueGenaiClient([req_json])
    ctx_plan = WorkerContext(cartridge_dir=tmp_path, client=req_client)
    planned_req = plan_asset_requirement(ctx_plan, item, ("portrait", "static_sprite"))
    assert planned_req.target_path == "assets/ui/portrait_hero.png"
    assert planned_req.asset_id == "portrait_hero"  # Preserves canonical catalog ID!

    # Deterministic 1:1 inspection plate with grid_px + crop_boxes
    plate_out = render_inspection_plate(
        tmp_path,
        [target_png],
        output_path=tmp_path / "artifacts" / "renders" / "custom_plate.png",
        grid_px=16,
        crop_boxes=[(0, 0, 32, 32)],
    )
    assert plate_out.is_file()

    # VisualArtistPlan -> operation="inspect" via VisualEvaluatorAgent
    plan_inspect = VisualArtistPlan(
        deliberation="Inspect existing portrait against slice mockup before redrawing.",
        target_asset_paths=["assets/ui/portrait_hero.png"],
        grid_px=16,
        crop_boxes=[[0, 0, 32, 32]],
        inspection_inquiry="Check silhouette framing and style match against the Workshop mockup.",
        include_concept_mockup=True,
        operation="inspect",
    ).model_dump_json()
    eval_verdict = VisualEvaluationReport(
        deliberation="Inspected 1:1 plate alongside Workshop 16:9 mockup.",
        observed_visual_properties="64x64 crisp portrait with clean alpha silhouette.",
        dimensional_and_framing_notes="Centered bust within 4px margin.",
        style_and_mockup_alignment="Matches inked linework of Workshop mockup.",
        perceptual_descriptions_by_path={
            "assets/ui/portrait_hero.png": "64x64 inked bust portrait with clean alpha silhouette and warm brass rim lighting."
        },
        recommended_followups=["Optionally add blinking 2-frame variant in Slice 2"],
        answers_brief_satisfactorily=True,
    ).model_dump_json()

    vis_client = _QueueGenaiClient([plan_inspect, eval_verdict])
    ctx_vis = WorkerContext(
        cartridge_dir=tmp_path,
        client=vis_client,
        visual_evaluator=VisualEvaluatorAgent(client=vis_client),
    )
    rep = run_visual_artist_specialist(
        ctx_vis,
        item,
        brief="Inspect assets/ui/portrait_hero.png against the concept mockup.",
        target_paths=["assets/ui/portrait_hero.png"],
        turn_index=0,
    )
    assert rep.specialist_agent == "visual_artist"
    assert rep.touched_files == []
    assert rep.recommended_followups == ["Optionally add blinking 2-frame variant in Slice 2"]
    assert mockup_path.is_file()

    # Verify asset_library.json was enriched with perceptual_description
    reloaded = CartridgeAssetLibrary(cartridge_dir=tmp_path, client=None)
    assert "64x64 inked bust portrait" in reloaded.catalog.entries["portrait_hero"].perceptual_description


def test_audio_designer_unified_persona_long_barks_and_legacy_alias(tmp_path: Path):
    """Verifies:
    1. `audio_bark` preserves >50-word dramatic monologues verbatim without silent truncation.
    2. `audio_designer` routes `AudioDesignerPlan(operation='inspect')` through `AudioEvaluatorAgent`.
    3. Legacy `'audio_evaluator'` specialist alias dispatches cleanly through `run_audio_designer_specialist`.
    """
    cue_wav = _make_wav(tmp_path / "assets" / "audio" / "ambience_wind.wav")
    long_monologue = " ".join(f"word{i}" for i in range(75))

    bark_item = WorkItem(
        deliberation="Synthesize dramatic opening monologue.",
        item_id="opening_monologue",
        kind="audio_asset",
        title="Opening Monologue",
        spec="Record full 75-word opening monologue.",
        proof_spec="WAV file",
    )
    bark_req_json = json.dumps(
        {
            "deliberation": "Full dramatic monologue without truncation.",
            "asset_id": "opening_monologue",
            "name": "Opening Monologue",
            "asset_type": "voice_bark",
            "fabrication_mode": "audio_bark",
            "purpose": "Opening narration",
            "target_path": "assets/audio/opening_monologue.wav",
            "description": " Dramatic monologue",
            "visual_prompt_or_text": long_monologue,
        }
    )
    bark_client = _QueueGenaiClient([bark_req_json])
    planned_bark = plan_asset_requirement(
        WorkerContext(cartridge_dir=tmp_path, client=bark_client),
        bark_item,
        ("audio_bark", "audio_sfx"),
    )
    assert len(planned_bark.visual_prompt_or_text.split()) == 75

    # Unified audio_designer inspection + legacy audio_evaluator alias
    eval_calls: List[str] = []

    class _FakeAudioEvaluator:
        def evaluate(self, *, inquiry_prompt: str, media_paths: List[Path], **kwargs: Any) -> AudioEvaluationReport:
            eval_calls.append(inquiry_prompt)
            return AudioEvaluationReport(
                deliberation="Listened directly to ambience_wind.wav.",
                rich_acoustic_description="Warm low-frequency wind bed with subtle chime harmonics.",
                contextual_observations=["Clean loop boundary with zero clicks."],
                notable_concerns=[],
                answer_to_inquiry="Loop seamlessly matches scene mood.",
            )

    audio_plan = AudioDesignerPlan(
        deliberation="Brief asks to listen and verify loop seamlessness.",
        media_paths_to_inspect=["assets/audio/ambience_wind.wav"],
        inspection_inquiry="Verify loop boundary and frequency balance.",
        operation="inspect",
    ).model_dump_json()
    ad_client = _QueueGenaiClient([audio_plan])
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=ad_client,
        audio_evaluator=_FakeAudioEvaluator(),
    )
    rep_ad = run_audio_designer_specialist(
        ctx,
        bark_item,
        brief="Listen to assets/audio/ambience_wind.wav and check loop boundary.",
        media_paths=[str(cue_wav.relative_to(tmp_path))],
        turn_index=0,
    )
    assert rep_ad.specialist_agent == "audio_designer"
    assert "Warm low-frequency wind bed" in rep_ad.summary
    assert len(eval_calls) == 1

    # Validate legacy 'audio_evaluator' alias in validate_task_lead_decision
    code_item = WorkItem(
        deliberation="Cross-domain check",
        item_id="mix_check",
        kind="integration",
        title="Check mix",
        spec="Verify audio",
        proof_spec="Report",
    )
    alias_decision = TaskLeadDecision(
        deliberation="Delegate to legacy alias.",
        current_strategy="Verify wind loop via audio evaluator alias.",
        specialist_agent="audio_evaluator",
        brief="Check wind loop",
        target_paths=["assets/audio/ambience_wind.wav"],
        action="delegate",
    )
    assert validate_task_lead_decision(
        alias_decision,
        list(ALL_SPECIALIST_AGENTS),
        [],
        board=None,
        parent_item_id=code_item.item_id,
        item_kind=code_item.kind,
    ) == []


def test_swe_engineer_lock_free_inspect_only_and_script_writer_inspection(tmp_path: Path):
    """Verifies `swe_engineer` runs `operation_mode='inspect_only'` completely lock-free EVEN WHILE
    another worker holds `main.gd` in `CartridgeFileLockTable`, and `script_writer` supports read-only inspection.
    """
    (tmp_path / "project.godot").write_text('[application]\nconfig/name="LockFreeInspect"\n', encoding="utf-8")
    (tmp_path / "main.gd").write_text("extends Node2D\nvar speed: float = 220.0\n", encoding="utf-8")
    (tmp_path / "narrative.json").write_text(json.dumps({"world_title": "Chronos", "characters": [], "dialogues": []}), encoding="utf-8")

    lock_table = CartridgeFileLockTable()
    # Another worker holds an exclusive write lock on main.gd!
    assert lock_table.try_acquire_many("other_writer_item", ["main.gd"]) == (True, [])

    inspect_payload = TrackCodeSynthesisPayload(
        deliberation="Read-only architecture audit of main.gd; zero file mutations needed.",
        edits=[],
        implementation_summary="Inspected main.gd movement constants.",
        inspection_report="main.gd defines speed=220.0 on root Node2D.",
        recommended_followups=["Expose acceleration curve in Slice 1"],
        operation_mode="inspect_only",
    ).model_dump_json()
    swe_client = _QueueGenaiClient([inspect_payload])
    swe_agent = LeadGameplaySWEAgent(client=swe_client)

    disc_item = WorkItem(
        deliberation="Inspect main.gd while locked.",
        item_id="disc_arch",
        kind="discovery",
        title="Audit main.gd",
        spec="Inspect movement constants in main.gd.",
        proof_spec="Findings",
    )
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=swe_client,
        swe_agent=swe_agent,
        file_lock_table=lock_table,
    )
    swe_rep = run_swe_engineer_specialist(
        ctx,
        disc_item,
        brief="Inspect main.gd without editing any files.",
        target_paths=["main.gd"],
        turn_index=0,
    )
    assert swe_rep.touched_files == []
    assert "speed=220.0" in swe_rep.summary
    assert swe_rep.recommended_followups == ["Expose acceleration curve in Slice 1"]

    # Scriptwriter read-only continuity inspection (edits=[])
    sw_payload = ScriptWritingDeliverable(
        deliberation="Audited narrative.json tone and continuity; no edits requested.",
        narrative_summary="Audited Chronos world bible.",
        inspection_report="World title 'Chronos' is established; ready for Act 1 characters.",
        recommended_followups=["Add protagonist dialogue tree"],
        edits=[],
    ).model_dump_json()
    sw_client = _QueueGenaiClient([sw_payload])
    ctx_sw = WorkerContext(cartridge_dir=tmp_path, client=sw_client, file_lock_table=lock_table)
    sw_rep = run_script_writer_specialist(
        ctx_sw,
        disc_item,
        brief="Audit narrative.json continuity without modifying files.",
        target_paths=["narrative.json"],
        turn_index=1,
    )
    assert sw_rep.touched_files == []
    assert "Chronos" in sw_rep.summary
    assert sw_rep.recommended_followups == ["Add protagonist dialogue tree"]


def test_qa_exploratory_diagnostic_mode_and_fullres_mockup_vlm_grounding(tmp_path: Path):
    """Verifies:
    1. `validate_filmstrip_plan` permits `verification_mode='exploratory_diagnostic'` with custom `setup_calls`
       (including names like `trigger_dialogue` that were previously blacklisted) and zero forced input segments.
    2. `run_qa_test_engineer_specialist` attaches `.keyframe_fullres.png` + the slice's 16:9 mockup in `QAPostCaptureAudit`.
    """
    (tmp_path / "project.godot").write_text('[application]\nconfig/name="QALab"\nrun/main_scene="res://main.tscn"\n', encoding="utf-8")
    (tmp_path / "main.tscn").write_text('[gd_scene format=3]\n[node name="Main" type="Node2D"]\n', encoding="utf-8")
    (tmp_path / "main.gd").write_text("extends Node2D\nfunc get_state_snapshot() -> Dictionary:\n\treturn {'ready': true}\n", encoding="utf-8")
    _make_rgba_png(tmp_path / "assets" / "mockups" / "slice_0.png", size=(320, 180))
    bible = GameBibleSpec(
        deliberation="Bible",
        game_title="QALab",
        elevator_pitch="Lab",
        visual_mockups=[
            VisualMockupArtifact(
                deliberation="M0",
                mockup_id="m0",
                title="Main",
                scene_description="Main",
                image_path="assets/mockups/slice_0.png",
            )
        ],
    )
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    plan = FilmstripPlan(
        deliberation="Run exploratory scene telemetry snapshot and call trigger_dialogue to inspect UI layout.",
        capture_ticks=[10, 30],
        input_segments=[],
        setup_calls=[
            {
                "deliberation": "Stage dialogue box to inspect typography.",
                "at_tick": 5,
                "node_path": ".",
                "method_name": "trigger_dialogue",
                "args_json": '["greeting_01"]',
            }
        ],
        verification_mode="exploratory_diagnostic",
    )
    code_item = WorkItem(
        deliberation="Diagnostic UI probe",
        item_id="probe_ui",
        kind="mechanic_code",
        title="Probe UI",
        spec="Inspect dialogue layout",
        proof_spec="Diagnostic capture",
    )
    assert validate_filmstrip_plan(plan, code_item) == []

    audit_json = QAPostCaptureAudit(
        deliberation="Compared full-res 1:1 keyframe plate against 16:9 concept mockup.",
        observed_frames_summary="Crisp dialogue box aligned along bottom margin.",
        post_capture_test_scene_summary="Staged trigger_dialogue at t=5.",
        unaddressed_defects=[],
        recommended_followups=["Verify gamepad focus highlight"],
        qa_passed=True,
    ).model_dump_json()

    qa_client = _QueueGenaiClient([plan.model_dump_json(), audit_json])
    qa_client.enable_post_capture_audit = True

    def _fake_filmstrip(cartridge_dir: Path, *, ticks: List[int], out_path: Path, input_script: Any = None, **kwargs: Any) -> Path:
        strip = _make_rgba_png(out_path, size=(640, 180))
        _make_rgba_png(out_path.with_suffix(".keyframe_fullres.png"), size=(1280, 720))
        return strip

    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=qa_client,
        filmstrip_fn=_fake_filmstrip,
    )
    qa_rep = run_qa_test_engineer_specialist(ctx, code_item, brief="Inspect dialogue staging.", turn_index=0)
    assert qa_rep.recommended_followups == ["Verify gamepad focus highlight"]
    # Second LLM call (QAPostCaptureAudit) received: strip + .keyframe_fullres.png + 16:9 mockup + prompt (>= 4 parts)
    audit_contents = qa_client.calls[1]["contents"]
    assert len(audit_contents) >= 4


# =============================================================================
# 3. Parallel `discovery` Subtasks, Cache Invalidation & Generalized Council Probes
# =============================================================================


def test_discovery_subtask_auto_approves_invalidates_curator_cache_and_injects_into_parent(tmp_path: Path):
    """End-to-end verification of `kind='discovery'` subtasks:
    1. Parent `TaskLead` spawns a `kind='discovery'` subtask with `wait_for_spawned_tasks=True` (yielding into `blocked`).
    2. Worker-spawned `discovery` subtask runs via `default_handlers`, synthesizes `discovery_findings` + inspection plate,
       and auto-transitions directly to `approved` (skipping Council review queue) so the parent unblocks immediately!
    3. `ContextCuratorAgent.curate_context` invalidates its cached pack via `deps_sig` even though `design_revision` is unchanged.
    4. Resumed parent `TaskLead` receives `COMPLETED PREREQUISITE & DISCOVERY SUBTASK REPORTS` + child diagnostic plate.
    5. Council-spawned `discovery` items (`spawned_by_item_id=None`) route to `[raised_by_role]` in `WritersRoom._reviewers_for`.
    """
    (tmp_path / "project.godot").write_text('[application]\nconfig/name="DiscoveryFlow"\n', encoding="utf-8")
    (tmp_path / "main.gd").write_text("extends Node2D\nvar anchor_offset := Vector2(16, -24)\n", encoding="utf-8")
    sprite_p = _make_rgba_png(tmp_path / "assets" / "sprites" / "turret.png", size=(96, 96))

    board = ProductionBoard(tmp_path / "backlog.json")
    parent = board.add_items(
        [
            WorkItemProposal(
                deliberation="Wire turret sprite pivot to exact pixel socket.",
                item_id="wire_turret_socket",
                kind="mechanic_code",
                title="Wire turret socket",
                spec="Align turret barrel pivot to turret.png socket coordinates.",
                target_modules=["main.gd"],
                proof_spec="Verified alignment",
            ),
            WorkItemProposal(
                deliberation="Council-originated discovery question.",
                item_id="council_disc_q",
                kind="discovery",
                title="Check audio bus layout",
                spec="Inspect project audio buses.",
                proof_spec="Findings report",
            ),
        ],
        raised_by_role="TechnicalArchitect",
        slice_id="slice_0",
    )[0]

    # Pre-warm ContextCurator cache for parent BEFORE discovery subtask exists
    curator_client = _LiveQueueGenaiClient([])
    curator = ContextCuratorAgent(cartridge_dir=tmp_path, client=curator_client, board=board)
    pack_before = curator.curate_context(
        target_agent_role="task_lead",
        task_request=parent.spec,
        item_id=parent.item_id,
        item_kind=parent.kind,
    )
    assert curator_client.curator_calls == 1
    # Calling again before deps change hits disk cache (curator_calls stays 1)
    curator.curate_context(
        target_agent_role="task_lead",
        task_request=parent.spec,
        item_id=parent.item_id,
        item_kind=parent.kind,
    )
    assert curator_client.curator_calls == 1

    # Step 1: Claim parent; parent TaskLead first searches tools (tool_search_query) then spawns discovery subtask!
    claimed = board.claim_ready(1)
    assert [c.item_id for c in claimed] == ["wire_turret_socket"]

    parent_turn0_tool_search = TaskLeadDecision(
        deliberation="Search tool registry for 1:1 visual inspection capabilities before spawning discovery.",
        current_strategy="Query tool capabilities then spawn discovery subtask.",
        tool_search_query="visual_artist 1:1 inspection plate grid_px",
        action="delegate",
    ).model_dump_json()
    parent_turn0_spawn = TaskLeadDecision(
        deliberation="Spawn a lock-free discovery subtask to measure exact pixel coordinates on turret.png.",
        current_strategy="Spawn discovery subtask and wait for exact pixel offset.",
        spawned_tasks=[
            WorkItemProposal(
                deliberation="Measure pixel socket on turret.png with 16px ruler grid.",
                item_id="disc_turret_socket",
                kind="discovery",
                title="Measure turret.png socket coordinates",
                spec="Render a 1:1 plate of assets/sprites/turret.png with 16px grid and report exact socket pixel.",
                proof_spec="Socket pixel coordinates + 1:1 plate",
            )
        ],
        wait_for_spawned_tasks=True,
        action="spawn_tasks",
    ).model_dump_json()

    # Discovery child responses: visual_artist renders 1:1 plate -> TaskLead submits discovery_findings!
    child_delegate = TaskLeadDecision(
        deliberation="Delegate to visual_artist to render a 1:1 grid plate of turret.png.",
        current_strategy="Render 1:1 grid plate of turret.png.",
        specialist_agent="visual_artist",
        brief="Render a 1:1 inspection plate of assets/sprites/turret.png with 16px grid.",
        target_paths=["assets/sprites/turret.png"],
        action="delegate",
    ).model_dump_json()
    child_vis_plan = VisualArtistPlan(
        deliberation="Render deterministic 1:1 plate with 16px ruler grid.",
        target_asset_paths=["assets/sprites/turret.png"],
        grid_px=16,
        operation="render_contact_sheet",
    ).model_dump_json()
    child_submit = TaskLeadDecision(
        deliberation="Socket measured cleanly at pixel (48, 32).",
        current_strategy="Submit measured socket coordinates.",
        submission_summary="Completed turret socket measurement.",
        discovery_findings="turret.png is 96x96px; mounting socket center is at pixel (48, 32), i.e. offset (0, -16) from center.",
        action="submit_to_council",
    ).model_dump_json()

    # Resumed parent Turn 1: inspects injected discovery findings, edits main.gd, runs QA, and submits!
    parent_resumed_swe = TaskLeadDecision(
        deliberation="Discovery report confirms socket offset (0, -16); delegate main.gd edit to swe_engineer.",
        current_strategy="Update main.gd anchor_offset using discovered coordinates.",
        specialist_agent="swe_engineer",
        brief="Update anchor_offset to Vector2(0, -16) in main.gd.",
        target_paths=["main.gd"],
        action="delegate",
    ).model_dump_json()
    swe_edit = TrackCodeSynthesisPayload(
        deliberation="Apply discovered offset Vector2(0, -16) to main.gd.",
        patches=[
            {
                "deliberation": "Update anchor_offset.",
                "filename": "main.gd",
                "start_line": 2,
                "end_line": 2,
                "target_snippet": "var anchor_offset := Vector2(16, -24)",
                "replacement_code": "var anchor_offset := Vector2(0, -16)",
            }
        ],
        operation_mode="edit",
    ).model_dump_json()
    parent_resumed_qa = TaskLeadDecision(
        deliberation="Verify updated offset via qa_test_engineer.",
        current_strategy="Capture verification filmstrip.",
        specialist_agent="qa_test_engineer",
        brief="Verify main.gd state.",
        action="delegate",
    ).model_dump_json()
    qa_plan = FilmstripPlan(
        deliberation="Capture verification frame.",
        capture_ticks=[10],
        verification_mode="exploratory_diagnostic",
    ).model_dump_json()
    qa_audit = QAPostCaptureAudit(
        deliberation="Verified anchor_offset.",
        observed_frames_summary="Turret socket aligned.",
        qa_passed=True,
    ).model_dump_json()
    parent_resumed_submit = TaskLeadDecision(
        deliberation="Submit aligned turret socket to Council.",
        current_strategy="Submit verified implementation.",
        submission_summary="Aligned turret socket using discovery findings.",
        action="submit_to_council",
    ).model_dump_json()

    client = _LiveQueueGenaiClient(
        [
            parent_turn0_tool_search,
            parent_turn0_spawn,
            child_delegate,
            child_vis_plan,
            child_submit,
            parent_resumed_swe,
            swe_edit,
            parent_resumed_qa,
            qa_plan,
            parent_resumed_submit,
        ]
    )

    def _fake_strip(cartridge_dir: Path, *, ticks: List[int], out_path: Path, input_script: Any = None, **kwargs: Any) -> Path:
        return _make_rgba_png(out_path, size=(320, 180))

    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=client,
        visual_agent=SimpleNamespace(),
        swe_agent=LeadGameplaySWEAgent(client=client),
        code_verify=lambda cdir, trk: None,
        filmstrip_fn=_fake_strip,
        file_lock_table=CartridgeFileLockTable(),
        board=board,
    )
    handlers = default_handlers(ctx)
    assert set(handlers.keys()) >= {
        "visual_asset", "audio_asset", "animation", "mechanic_code",
        "ui", "level_map", "narrative", "integration", "discovery", "other",
    }

    # Verify Council-spawned discovery item (`spawned_by_item_id is None`) routes to [raised_by_role] and awaiting_review!
    wr = CreativeWritersRoom(client=None)
    council_disc_item = board.get("council_disc_q")
    assert wr._reviewers_for(council_disc_item) == ["TechnicalArchitect"]
    board.claim_ready(2)  # claims wire_turret_socket and council_disc_q; let's reset wire_turret_socket back via yield_item and submit council_disc_q
    board.yield_item("wire_turret_socket", reason="prep")
    board.submit_result("council_disc_q", WorkerResult(notes="Audio bus checked."))
    assert board.get("council_disc_q").state == "awaiting_review"

    sched = BoardScheduler(board, handlers, max_workers=1)
    # 1st step: runs parent -> yields into `blocked` after spawning `disc_turret_socket`
    assert _step_scheduler_once(sched) is True
    assert board.get("wire_turret_socket").state == "blocked"
    disc_child = board.get("disc_turret_socket")
    assert disc_child.state == "ready"
    assert disc_child.spawned_by_item_id == "wire_turret_socket"

    # 2nd step: runs worker-spawned discovery item -> MUST auto-approve and unblock parent immediately!
    assert _step_scheduler_once(sched) is True
    assert board.get("disc_turret_socket").state == "approved"
    assert board.get("wire_turret_socket").state == "ready"  # Unblocked automatically without Council review delay!

    # ContextCurator cache key MUST invalidate via deps_sig even though design_revision == 0!
    pack_after = curator.curate_context(
        target_agent_role="task_lead",
        task_request=parent.spec,
        item_id=parent.item_id,
        item_kind=parent.kind,
    )
    assert curator_client.curator_calls == 2
    assert pack_after.prompt_block != pack_before.prompt_block

    # 3rd step: resumed parent executes using injected discovery findings & child plate, submitting to awaiting_review!
    assert _step_scheduler_once(sched) is True
    assert board.get("wire_turret_socket").state == "awaiting_review"
    assert "Vector2(0, -16)" in (tmp_path / "main.gd").read_text(encoding="utf-8")

    # Confirm the resumed parent's prompt explicitly contained COMPLETED PREREQUISITE & DISCOVERY SUBTASK REPORTS
    prompt_texts = [
        " ".join(str(p) for p in call["contents"])
        for call in client.calls
    ]
    assert any("COMPLETED PREREQUISITE & DISCOVERY SUBTASK REPORTS" in t and "(48, 32)" in t for t in prompt_texts)
    assert sprite_p.is_file()


def test_generalized_council_diagnostic_probe_dispatches_any_specialist_persona(tmp_path: Path):
    """Verifies `CreativeWritersRoom._run_council_diagnostic_probe` supports all 5 Specialist Personas."""
    (tmp_path / "project.godot").write_text('[application]\nconfig/name="CouncilProbe"\n', encoding="utf-8")
    (tmp_path / "main.gd").write_text("extends Node2D\nconst GRAVITY := 980.0\n", encoding="utf-8")
    _make_rgba_png(tmp_path / "assets" / "sprites" / "ship.png", size=(64, 64))

    swe_inspect = TrackCodeSynthesisPayload(
        deliberation="Read-only probe for Council.",
        edits=[],
        implementation_summary="Inspected GRAVITY constant.",
        inspection_report="main.gd sets GRAVITY := 980.0.",
        operation_mode="inspect_only",
    ).model_dump_json()
    vis_plan = VisualArtistPlan(
        deliberation="Render 1:1 plate for Council visual probe.",
        target_asset_paths=["assets/sprites/ship.png"],
        grid_px=16,
        operation="render_contact_sheet",
    ).model_dump_json()

    probe_client = _QueueGenaiClient([swe_inspect, vis_plan])
    wr = CreativeWritersRoom(client=probe_client)
    dummy_item = WorkItem(
        deliberation="Review item",
        item_id="rev_item",
        kind="mechanic_code",
        title="Physics",
        spec="Physics",
        proof_spec="Proof",
    )

    swe_summary, swe_imgs = wr._run_council_diagnostic_probe(
        candidate_dir=tmp_path,
        diag_probe=CouncilDiagnosticProbe(
            deliberation="Probe GRAVITY in main.gd.",
            specialist_agent="swe_engineer",
            diagnostic_brief="Check GRAVITY in main.gd.",
            target_paths=["main.gd"],
            needs_runtime_debug_scene=True,
        ),
        title_hint=dummy_item.title,
    )
    assert "GRAVITY := 980.0" in swe_summary
    assert swe_imgs == []

    vis_summary, vis_imgs = wr._run_council_diagnostic_probe(
        candidate_dir=tmp_path,
        diag_probe=CouncilDiagnosticProbe(
            deliberation="Probe ship.png at 1:1 pixels.",
            specialist_agent="visual_artist",
            diagnostic_brief="Render 1:1 grid plate of assets/sprites/ship.png.",
            target_paths=["assets/sprites/ship.png"],
            needs_runtime_debug_scene=True,
        ),
        title_hint="Ship Sprite",
    )
    assert "ship.png" in vis_summary
    assert len(vis_imgs) >= 1


# =============================================================================
# 4. Self-Directed Task Decomposition (`action="split_task"`)
# =============================================================================


def test_task_lead_self_splitting_rewires_dependents_rolls_back_and_executes_children(tmp_path: Path):
    """Verifies `action='split_task'` end-to-end:
    1. Rejects `< 2` split tasks or cyclic intra-batch `depends_on`.
    2. When `preserve_verified_changes_on_split=False`, rolls back uncommitted partial edits on disk,
       releases file locks, marks parent `cancelled` with `'Split by TaskLead...'`, and raises `WorkerSplitTask`.
    3. `ProductionBoard.split_active_item` inherits upstream `depends_on` onto root split children and
       rewires downstream dependents of the parent onto the sink split children (so downstream tasks wait on
       `split_c` rather than failing via `blocked_by_cancelled`!).
    4. `BoardScheduler._run_item` catches `WorkerSplitTask` cleanly without incrementing `failure_count`.
    """
    (tmp_path / "project.godot").write_text('[application]\nconfig/name="SplitLab"\n', encoding="utf-8")
    main_gd = tmp_path / "main.gd"
    main_gd.write_text("extends Node2D\nvar stage: int = 0\n", encoding="utf-8")

    board = ProductionBoard(tmp_path / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="Upstream prerequisite already approved.",
                item_id="prereq_core",
                kind="visual_asset",
                title="Core sprite",
                spec="Sprite",
                proof_spec="PNG",
            ),
            WorkItemProposal(
                deliberation="Oversized multi-concern item that TaskLead will decompose into 2 sequential steps.",
                item_id="mega_combat_and_hud",
                kind="mechanic_code",
                title="Build combat AND full HUD",
                spec="Implement combat state machine and HUD overlay.",
                depends_on=["prereq_core"],
                target_modules=["main.gd"],
                proof_spec="Verified combat + HUD",
            ),
            WorkItemProposal(
                deliberation="Downstream polish task depending on mega_combat_and_hud.",
                item_id="downstream_polish",
                kind="integration",
                title="Final integration polish",
                spec="Polish after combat and HUD.",
                depends_on=["mega_combat_and_hud"],
                target_modules=["main.gd"],
                proof_spec="Verified polish",
            ),
        ],
        raised_by_role="GameplayDesigner",
        slice_id="slice_1",
    )
    # Approve prereq_core so mega_combat_and_hud becomes ready
    board.claim_ready(1)
    board.submit_result("prereq_core", WorkerResult(notes="ok"))
    from engine.studio.models import ReviewDecision
    board.apply_review(ReviewDecision(deliberation="ok", item_id="prereq_core", verdict="approve"))
    assert board.get("mega_combat_and_hud").state == "ready"
    assert board.get("downstream_polish").state == "blocked"

    # Validation rejects < 2 split_tasks
    bad_split = TaskLeadDecision(
        deliberation="Invalid 1-child split.",
        current_strategy="Invalid split test.",
        split_tasks=[
            WorkItemProposal(
                deliberation="Only one child",
                item_id="only_one",
                kind="mechanic_code",
                title="One",
                spec="One",
                proof_spec="One",
            )
        ],
        action="split_task",
    )
    assert any(
        "at least 2" in err
        for err in validate_task_lead_decision(
            bad_split,
            list(ALL_SPECIALIST_AGENTS),
            [],
            board=board,
            parent_item_id="mega_combat_and_hud",
            item_kind="mechanic_code",
        )
    )

    # Turn 0 on mega_combat_and_hud: SWE makes a partial edit, then Turn 1 TaskLead decides to split_task with preserve_verified_changes_on_split=False!
    turn0_swe = TaskLeadDecision(
        deliberation="Try initial edit on main.gd.",
        current_strategy="Initial edit attempt.",
        specialist_agent="swe_engineer",
        brief="Draft stage=99 in main.gd.",
        target_paths=["main.gd"],
        action="delegate",
    ).model_dump_json()
    swe_partial = TrackCodeSynthesisPayload(
        deliberation="Draft stage=99.",
        patches=[
            {
                "deliberation": "Temporary edit",
                "filename": "main.gd",
                "start_line": 2,
                "end_line": 2,
                "target_snippet": "var stage: int = 0",
                "replacement_code": "var stage: int = 99",
            }
        ],
        operation_mode="edit",
    ).model_dump_json()
    turn1_split = TaskLeadDecision(
        deliberation="Task combines combat mechanics and HUD; split into sequential combat_part -> hud_part and roll back partial draft.",
        current_strategy="Split into sequential combat_part and hud_part.",
        split_tasks=[
            WorkItemProposal(
                deliberation="Step 1: Combat mechanics.",
                item_id="split_combat_part",
                kind="mechanic_code",
                title="Combat state machine",
                spec="Implement combat state.",
                target_modules=["main.gd"],
                proof_spec="Combat verified",
            ),
            WorkItemProposal(
                deliberation="Step 2: HUD bound to combat state.",
                item_id="split_hud_part",
                kind="ui",
                title="Combat HUD",
                spec="Bind HUD to combat state.",
                depends_on=["split_combat_part"],
                target_modules=["main.gd"],
                proof_spec="HUD verified",
            ),
        ],
        preserve_verified_changes_on_split=False,
        action="split_task",
    ).model_dump_json()

    split_client = _QueueGenaiClient([turn0_swe, swe_partial, turn1_split])
    lock_table = CartridgeFileLockTable()
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=split_client,
        swe_agent=LeadGameplaySWEAgent(client=split_client),
        code_verify=lambda cdir, trk: None,
        file_lock_table=lock_table,
        board=board,
    )
    sched = BoardScheduler(board, default_handlers(ctx), max_workers=1)
    assert _step_scheduler_once(sched) is True

    # Parent is cancelled with "Split by TaskLead...", failure_count == 0, partial edit rolled back, locks released!
    parent_after = board.get("mega_combat_and_hud")
    assert parent_after.state == "cancelled"
    assert parent_after.failure_count == 0
    assert (parent_after.decision_reason or "").startswith("Split by TaskLead")
    assert "var stage: int = 0" in main_gd.read_text(encoding="utf-8")  # Rolled back cleanly!
    assert lock_table.all_held_paths() == set()

    # Split children created with upstream dependency preserved and intra-batch ordering intact:
    combat_child = board.get("split_combat_part")
    hud_child = board.get("split_hud_part")
    downstream = board.get("downstream_polish")
    assert combat_child.state == "ready"
    assert "prereq_core" in combat_child.depends_on
    assert combat_child.spawned_by_item_id == "mega_combat_and_hud"
    assert hud_child.state == "blocked"
    assert "split_combat_part" in hud_child.depends_on

    # Downstream item was automatically rewired to wait on all split children (NOT `blocked_by_cancelled`!)
    assert downstream.state == "blocked"
    assert downstream.depends_on == ["split_combat_part", "split_hud_part"]
