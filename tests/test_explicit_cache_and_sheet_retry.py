"""Unit tests for deterministic explicit Gemini prefix caching (`ExplicitPrefixCacheScope`),
surgical per-cell multi-cell sprite-sheet retry & `cell_prompts` refinement, per-cell `target_display_size`,
inanimate prop `fabricate_sprite_sheet` defaults, and image token cost non-double-counting.
"""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
from PIL import Image, ImageDraw

from engine.studio.asset_library import CartridgeAssetLibrary
from engine.studio.models import (
    AssetRequirement,
    AssetVisualValidationVerdict,
    TokenUsageTracker,
    explicit_prefix_cache_scope,
    inject_gemini_dataset_config,
    resolve_explicit_cached_request,
)
from engine.studio.visual import VisualArtSpriteDesignerAgent


class _SpyCachesAPI:
    def __init__(self) -> None:
        self.created: List[Dict[str, Any]] = []
        self.deleted: List[str] = []
        self._counter = 0

    def create(self, *, model: str, config: Any) -> Any:
        self._counter += 1
        name = f"cachedContents/spy-cache-{self._counter}"
        self.created.append({"name": name, "model": model, "config": config})
        return SimpleNamespace(
            name=name,
            usage_metadata=SimpleNamespace(total_token_count=8192),
        )

    def delete(self, *, name: str) -> None:
        self.deleted.append(name)


class _SpyGenAIClient:
    def __init__(self) -> None:
        self._allow_explicit_prefix_cache = True
        self.caches = _SpyCachesAPI()


def test_rule_4_asset_visual_validation_verdict_field_ordering() -> None:
    fields = list(AssetVisualValidationVerdict.model_fields.keys())
    assert fields[0] == "deliberation"
    assert "defective_cell_indices" in fields
    assert fields[-1] == "accepted"
    assert fields.index("defective_cell_indices") < fields.index("accepted")


def test_explicit_prefix_cache_scope_eager_and_cleanup_on_exception() -> None:
    from google.genai import types

    client = _SpyGenAIClient()
    large_prefix = "SHARED_COUNCIL_REVIEW_PREFIX " * 900  # > 25,000 chars (> 4,500 est tokens)

    # 1. Eager mode (`min_calls_before_create=1`) creates on Call 1, reuses across panelists, and deletes on exit
    with explicit_prefix_cache_scope(client, ttl_seconds=300, reuse_outer=False):
        c1, name1 = resolve_explicit_cached_request(
            client,
            model="gemini-3.8-flash",
            prefix_parts=[large_prefix],
            tail_parts=["Panelist 1 mandate and prompt"],
            unsplit_contents=[large_prefix, "Panelist 1 mandate and prompt"],
            min_calls_before_create=1,
            display_name="council_ReviewBatch",
        )
        c2, name2 = resolve_explicit_cached_request(
            client,
            model="gemini-3.8-flash",
            prefix_parts=[large_prefix],
            tail_parts=["Panelist 2 mandate and prompt"],
            unsplit_contents=[large_prefix, "Panelist 2 mandate and prompt"],
            min_calls_before_create=1,
            display_name="council_ReviewBatch",
        )
        assert name1 == "cachedContents/spy-cache-1"
        assert name2 == "cachedContents/spy-cache-1"
        assert c1 == ["Panelist 1 mandate and prompt"]
        assert c2 == ["Panelist 2 mandate and prompt"]
        assert len(client.caches.created) == 1
        assert client.caches.deleted == []

        # Verify `inject_gemini_dataset_config` preserves `cached_content` alongside `labels` and `store=True`
        cfg = types.GenerateContentConfig(
            cached_content=name1,
            response_mime_type="application/json",
            response_schema=AssetVisualValidationVerdict,
        )
        injected = inject_gemini_dataset_config(
            {"model": "gemini-3.8-flash", "contents": c1, "config": cfg},
            label="CreativeWritersRoom[ReviewBatch]",
        )
        assert injected["config"].cached_content == name1
        assert injected["config"].labels.get("agent") == "creativewritersroom"

    assert client.caches.deleted == ["cachedContents/spy-cache-1"]

    # 2. Cleanup also runs deterministically when an exception aborts the scope
    with pytest.raises(RuntimeError, match="simulated worker yield"):
        with explicit_prefix_cache_scope(client, ttl_seconds=420, reuse_outer=False):
            _, name_err = resolve_explicit_cached_request(
                client,
                model="gemini-3.8-flash",
                prefix_parts=[large_prefix],
                tail_parts=["Turn 0 dynamic tail"],
                min_calls_before_create=1,
                display_name="lead_item_1",
            )
            assert name_err == "cachedContents/spy-cache-2"
            raise RuntimeError("simulated worker yield")

    assert client.caches.deleted == [
        "cachedContents/spy-cache-1",
        "cachedContents/spy-cache-2",
    ]


def test_explicit_prefix_cache_scope_lazy_second_call_and_small_prefix_bypass() -> None:
    client = _SpyGenAIClient()
    swe_prefix = "STATIC_SWE_SYSTEM_AND_MODULE_CONTEXT " * 800  # ~29,600 chars

    # 1. Single-shot call (`min_calls_before_create=2`) incurs zero `caches.create` calls
    with explicit_prefix_cache_scope(client, ttl_seconds=360, reuse_outer=False):
        c_only, name_only = resolve_explicit_cached_request(
            client,
            model="gemini-3.8-flash",
            prefix_parts=[swe_prefix],
            tail_parts=["Pass 1 track brief"],
            unsplit_contents=f"{swe_prefix}\n\nPass 1 track brief",
            min_calls_before_create=2,
            display_name="swe_track_single",
        )
        assert name_only is None
        assert c_only == f"{swe_prefix}\n\nPass 1 track brief"
    assert client.caches.created == []
    assert client.caches.deleted == []

    # 2. Multi-call (`Pass 1 -> Pass 2` / self-inquiry) creates on Call 2 and reuses on Call 3
    with explicit_prefix_cache_scope(client, ttl_seconds=360, reuse_outer=False):
        _, n1 = resolve_explicit_cached_request(
            client,
            model="gemini-3.8-flash",
            prefix_parts=[swe_prefix],
            tail_parts=["Self-inquiry turn 0"],
            unsplit_contents=f"{swe_prefix}\n\nSelf-inquiry turn 0",
            min_calls_before_create=2,
            display_name="swe_track_multi",
        )
        assert n1 is None
        assert len(client.caches.created) == 0

        # Nested scope with `reuse_outer=True` shares the outer scope's seen counts and cache map
        with explicit_prefix_cache_scope(client, ttl_seconds=300, reuse_outer=True):
            c_hit2, n2 = resolve_explicit_cached_request(
                client,
                model="gemini-3.8-flash",
                prefix_parts=[swe_prefix],
                tail_parts=["Self-inquiry turn 1 with answer"],
                unsplit_contents=f"{swe_prefix}\n\nSelf-inquiry turn 1 with answer",
                min_calls_before_create=2,
                display_name="swe_track_multi",
            )
            assert n2 == "cachedContents/spy-cache-1"
            assert c_hit2 == ["Self-inquiry turn 1 with answer"]
            assert client.caches.deleted == []  # Inner `reuse_outer=True` must NOT close outer scope early

        c_hit3, n3 = resolve_explicit_cached_request(
            client,
            model="gemini-3.8-flash",
            prefix_parts=[swe_prefix],
            tail_parts=["Validation retry turn 2"],
            unsplit_contents=f"{swe_prefix}\n\nValidation retry turn 2",
            min_calls_before_create=2,
            display_name="swe_track_multi",
        )
        assert n3 == "cachedContents/spy-cache-1"
        assert c_hit3 == ["Validation retry turn 2"]
        assert len(client.caches.created) == 1

    assert client.caches.deleted == ["cachedContents/spy-cache-1"]

    # 3. Prefix below conservative token floor (< 4,500 est tokens) bypasses `caches.create`
    with explicit_prefix_cache_scope(client, ttl_seconds=300, reuse_outer=False):
        c_small, n_small = resolve_explicit_cached_request(
            client,
            model="gemini-3.8-flash",
            prefix_parts=["short prefix"],
            tail_parts=["short tail"],
            unsplit_contents="short prefix\nshort tail",
            min_calls_before_create=1,
        )
        assert n_small is None
        assert c_small == "short prefix\nshort tail"
    assert len(client.caches.created) == 1


def test_surgical_per_cell_sprite_sheet_retry_and_cell_display_size(tmp_path: Path) -> None:
    """Verifies that when 1 cell of a 2x2 multi-cell sheet fails validation on Attempt 1:
    1. Only the defective cell (`idx=2`) is re-synthesized on Attempt 2 (`4 + 1 = 5` calls instead of `8`),
    2. `verdict.prompt_refinement_directive` is appended specifically to `req.cell_prompts[2]`,
    3. Clean cells `0, 1, 3` keep their exact Attempt 1 pixels, and
    4. `AssetLibraryEntry.target_display_size` records per-cell `(64, 64)` rather than whole-sheet `(128, 128)`."""
    agent = VisualArtSpriteDesignerAgent(client=None, allow_offline_procedural=True)
    agent._is_live_genai_client = lambda: True  # type: ignore[method-assign]

    synth_calls: List[str] = []
    cell_colors = [
        (220, 70, 70, 255),
        (70, 200, 90, 255),
        (70, 120, 230, 255),
        (230, 190, 60, 255),
    ]

    def _fake_generate_sprite_image(prompt: str, dimensions=(64, 64), **kwargs: Any) -> Image.Image:
        call_no = len(synth_calls)
        synth_calls.append(prompt)
        im = Image.new("RGBA", dimensions, (0, 0, 0, 0))
        # On Attempt 1 (first 4 calls), make cell index 2 (call_no == 2) completely blank so it fails!
        if call_no == 2:
            return im
        color = cell_colors[call_no % 4] if call_no < 4 else (180, 90, 240, 255)
        draw = ImageDraw.Draw(im)
        draw.ellipse([12, 12, dimensions[0] - 12, dimensions[1] - 12], fill=color)
        return im

    agent._generate_gemini_sprite_image = _fake_generate_sprite_image  # type: ignore[method-assign]

    verify_calls = 0

    def _fake_verify(asset_req: Any, rendered_asset_path: Path, **kwargs: Any) -> AssetVisualValidationVerdict:
        nonlocal verify_calls
        verify_calls += 1
        with Image.open(rendered_asset_path) as sheet_im:
            arr = np.asarray(sheet_im.convert("RGBA"))
        cell2_alpha = arr[64:128, 0:64, 3]
        if int(np.count_nonzero(cell2_alpha > 24)) < 16:
            return AssetVisualValidationVerdict(
                deliberation="Cell 2 (row=1, col=0) is blank while cells 0, 1, 3 are clean.",
                asset_id=asset_req.asset_id,
                subject_framing_and_completeness_passed=False,
                visual_defects=["Multi-cell grid sheet (2x2) has empty/blank cell(s): (row=1, col=0)."],
                defective_cell_indices=[2],
                prompt_refinement_directive="Render centered solid silhouette inside cell bounds",
                accepted=False,
            )
        return AssetVisualValidationVerdict(
            deliberation="All 4 cells in the 2x2 grid are now populated and clean.",
            asset_id=asset_req.asset_id,
            accepted=True,
        )

    agent.verify_rendered_asset = _fake_verify  # type: ignore[method-assign]

    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path, visual_agent=agent, allow_offline_procedural=True)
    req = AssetRequirement(
        deliberation="2x2 status indicator sheet with 4 distinct states.",
        asset_id="status_quad_sheet",
        asset_type="prop",
        fabrication_mode="static_sprite",
        purpose="4-state world indicator",
        description="4-state world indicator",
        visual_prompt_or_text="4-state world indicator",
        target_path="assets/sprites/status_quad_sheet.png",
        dimensions=(128, 128),
        grid_cols=2,
        grid_rows=2,
        cell_prompts=[
            "State 0 dormant emblem",
            "State 1 charging emblem",
            "State 2 active emblem",
            "State 3 overloaded emblem",
        ],
    )

    out_p = lib.fabricate_single_asset(req, force_regenerate=True)
    assert out_p.is_file()
    assert verify_calls == 2
    # 4 cells synthesized on Attempt 1 + ONLY 1 defective cell (idx=2) on Attempt 2 = 5 calls (not 8!)
    assert len(synth_calls) == 5
    assert "Render centered solid silhouette inside cell bounds" in synth_calls[4]
    assert "Render centered solid silhouette inside cell bounds" in req.cell_prompts[2]
    assert "Render centered solid silhouette inside cell bounds" not in req.cell_prompts[0]

    # Verify clean cells 0, 1, 3 kept their Attempt 1 colors and cell 2 now has the Attempt 2 color
    with Image.open(out_p) as final_im:
        final_arr = np.asarray(final_im.convert("RGBA"))
    assert tuple(int(x) for x in final_arr[32, 32]) == (220, 70, 70, 255)   # Cell 0 preserved
    assert tuple(int(x) for x in final_arr[32, 96]) == (70, 200, 90, 255)   # Cell 1 preserved
    assert tuple(int(x) for x in final_arr[96, 32]) == (180, 90, 240, 255)  # Cell 2 surgically repaired
    assert tuple(int(x) for x in final_arr[96, 96]) == (230, 190, 60, 255)  # Cell 3 preserved

    entry = lib.catalog.entries["status_quad_sheet"]
    assert entry.texture_pixel_size == (128, 128)
    assert entry.target_display_size == (64, 64)
    assert (entry.hframes, entry.vframes) == (2, 2)


def test_inanimate_prop_default_sprite_sheet_vs_actor_full_grid(tmp_path: Path) -> None:
    agent = VisualArtSpriteDesignerAgent(client=None, allow_offline_procedural=True)

    # 1. Inanimate prop without `animation_states` defaults to compact 2x1 sheet (not 4x4 character locomotion)
    prop_req = AssetRequirement(
        deliberation="Ambient world prop sheet.",
        asset_id="beacon_prop",
        asset_type="prop",
        fabrication_mode="sprite_sheet",
        dimensions=(64, 64),
        description="Glowing signal marker prop",
    )
    prop_sheet, prop_meta = agent.fabricate_sprite_sheet(prop_req, tmp_path / "beacon_sheet.png")
    assert prop_sheet.is_file()
    assert (prop_meta["hframes"], prop_meta["vframes"]) == (2, 1)
    assert prop_meta["texture_pixel_size"] == (128, 64)

    # 2. Animated actor without `animation_states` defaults to 4x4 with all 4 frames populated in every row
    actor_req = AssetRequirement(
        deliberation="Animated protagonist sheet.",
        asset_id="hero_actor",
        asset_type="sprite_sheet",
        fabrication_mode="sprite_sheet",
        dimensions=(64, 64),
        description="Explorer protagonist",
    )
    actor_sheet, actor_meta = agent.fabricate_sprite_sheet(actor_req, tmp_path / "hero_sheet.png")
    assert (actor_meta["hframes"], actor_meta["vframes"]) == (4, 4)
    assert actor_meta["animation_states"]["jump"]["frame_count"] == 4
    with Image.open(actor_sheet) as im:
        arr = np.asarray(im.convert("RGBA"))
    # Previously (row=2, col=3) was blank because jump had frame_count=3; now it must be populated
    jump_col3_alpha = arr[2 * 64:3 * 64, 3 * 64:4 * 64, 3]
    assert int(np.count_nonzero(jump_col3_alpha > 24)) > 20


def test_token_usage_tracker_image_output_tokens_not_double_charged() -> None:
    tracker = TokenUsageTracker()
    buf = io.BytesIO()
    Image.new("RGBA", (32, 32), (100, 150, 200, 255)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    fake_img_resp = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=400,
            cached_content_token_count=0,
            candidates_token_count=1290,  # Image token representation already covered by $0.039/image
            thoughts_token_count=0,
            total_token_count=1690,
        ),
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            inline_data=SimpleNamespace(mime_type="image/png", data=png_bytes)
                        )
                    ]
                )
            )
        ],
    )
    stats = tracker.record_response(fake_img_resp, label="generate_gemini_sprite_image")
    assert stats is not None
    assert stats["image_count"] == 1
    # Expected cost: 400 * $0.75 / 1M ($0.0003) + 1 * $0.039 = $0.0393 (NOT $0.044138 with double-counted 1,290 output tokens)
    assert stats["estimated_call_cost_usd"] == pytest.approx(0.0393, abs=1e-5)

    # Also verify explicit cache storage accounting: 36,000 tokens * 100s / 3600s = 1,000.0 token-hours (+$0.001)
    tracker.record_explicit_cache_lifecycle(cached_tokens=36_000, lifetime_seconds=100.0)
    summary = tracker.get_summary()
    assert summary["explicit_caches_created"] == 1
    assert summary["cache_storage_token_hours"] == pytest.approx(1000.0, abs=1e-2)
    assert summary["estimated_cost_usd"] == pytest.approx(0.0403, abs=1e-5)


def test_plan_filmstrip_pass1_to_pass2_wires_explicit_cache_on_second_call(tmp_path: Path) -> None:
    """Verifies `plan_filmstrip` inside an `explicit_prefix_cache_scope` makes 0 `caches.create` calls on Pass 1
    and automatically creates & attaches `cached_content` on Pass 2 while sending only the dynamic Pass 1 tail."""
    from engine.studio.models import FilmstripPlan, InputSegment, WorkItem
    from engine.studio.workers import WorkerContext, plan_filmstrip

    (tmp_path / "project.godot").write_text(
        '[input]\nmove_right={"events":[]}\n',
        encoding="utf-8",
    )
    # Create realistic GDScript symbol signatures + task spec so `static_qa_prefix_text` exceeds the 4,500-token floor
    big_vars = "\n".join(f"var subsystem_metric_{i:03d}: float = {i}.0" for i in range(260))
    (tmp_path / "main.gd").write_text(
        f"extends Node2D\n{big_vars}\nfunc get_state_snapshot() -> Dictionary:\n\treturn {{'score': 0}}\n",
        encoding="utf-8",
    )
    detailed_spec = "Implement responsive horizontal movement. " * 280
    p1_strip = tmp_path / "filmstrip_p1.png"
    Image.new("RGBA", (64, 36), (20, 30, 40, 255)).save(p1_strip, format="PNG")

    valid_plan_json = FilmstripPlan(
        deliberation="Test player movement across 60 ticks.",
        target_scene="res://tests/test_move_item.tscn",
        test_scene_gd_content="extends Node2D\nfunc get_state_snapshot() -> Dictionary:\n\treturn {'score': 1}\n",
        overlays_and_telemetry_strategy="Render velocity vector and snapshot score.",
        council_test_scene_summary="Instantiates movement test scene and holds move_right.",
        capture_ticks=[1, 30, 60],
        input_segments=[InputSegment(deliberation="Hold right", from_tick=2, to_tick=55, actions=["move_right"])],
        verification_mode="causal_gameplay",
    ).model_dump_json()

    recorded_calls: List[Dict[str, Any]] = []

    class _SpyModelsAPI:
        def generate_content(self, *, model: str, contents: Any, config: Any = None, **kwargs: Any) -> Any:
            recorded_calls.append({"model": model, "contents": contents, "config": config})
            return SimpleNamespace(
                text=valid_plan_json,
                usage_metadata=SimpleNamespace(
                    prompt_token_count=6200,
                    cached_content_token_count=5800 if getattr(config, "cached_content", None) else 0,
                    candidates_token_count=180,
                    thoughts_token_count=90,
                    total_token_count=6470,
                ),
            )

    class _FullSpyClient(_SpyGenAIClient):
        __module__ = "google.genai.fake"

        def __init__(self) -> None:
            super().__init__()
            self.models = _SpyModelsAPI()

    spy_client = _FullSpyClient()
    ctx = WorkerContext(cartridge_dir=tmp_path, client=spy_client, model_name="gemini-3.8-flash")
    item = WorkItem(
        deliberation="Core movement task.",
        item_id="move_item",
        kind="mechanic_code",
        title="Responsive movement",
        spec=detailed_spec,
        proof_spec="Show movement progression across 60 ticks.",
        raised_by_role="GameplayDesigner",
        target_modules=["main.gd"],
    )

    with explicit_prefix_cache_scope(spy_client, ttl_seconds=300, reuse_outer=False):
        # Pass 1: seen == 1 (< min_calls_before_create=2) -> zero caches.create overhead
        plan_filmstrip(ctx, item)
        assert len(spy_client.caches.created) == 0
        assert getattr(recorded_calls[0]["config"], "cached_content", None) is None

        # Pass 2: seen == 2 -> creates explicit cache for static QA prefix and attaches cached_content!
        plan_filmstrip(
            ctx,
            item,
            prior_capture_feedback="Assertion failed at tick 60: score stayed 0",
            prior_strip_path=p1_strip,
        )
        assert len(spy_client.caches.created) == 1
        assert getattr(recorded_calls[1]["config"], "cached_content", None) == "cachedContents/spy-cache-1"
        assert "Assertion failed at tick 60" in str(recorded_calls[1]["contents"])

    # Scope teardown deterministically deletes the created cache
    assert spy_client.caches.deleted == ["cachedContents/spy-cache-1"]


def test_explicit_prefix_cache_evicts_expired_ttl_and_deduplicates_parallel_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """1. Prevents 403 `CachedContent not found` by evicting & recreating when monotonic age crosses `ttl_seconds - safety_margin`.
    2. Single-flight deduplication + `bind_explicit_prefix_cache_scope` lets concurrent worker threads share 1 `caches.create`."""
    import concurrent.futures
    import time as _time
    from engine.studio.models import bind_explicit_prefix_cache_scope

    client = _SpyGenAIClient()
    fake_clock = [1000.0]
    monkeypatch.setattr(_time, "monotonic", lambda: fake_clock[0])
    big_prefix = ["SHARED BIBLE AND BOARD SNAPSHOT\n" + ("Entity contract invariant line.\n" * 800)]

    with explicit_prefix_cache_scope(client, ttl_seconds=120, reuse_outer=False) as scope:
        # Concurrent calls across 4 worker threads -> single-flight creates ONLY 1 CachedContent and all 4 attach it!
        def _worker(idx: int) -> Tuple[Any, Any]:
            with bind_explicit_prefix_cache_scope(scope):
                return resolve_explicit_cached_request(
                    client,
                    model="gemini-3.8-flash",
                    prefix_parts=big_prefix,
                    tail_parts=[f"Role #{idx} instructions"],
                    min_calls_before_create=1,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            out = list(pool.map(_worker, range(4)))
        assert len(client.caches.created) == 1
        assert all(cname == "cachedContents/spy-cache-1" for _, cname in out)

        # Advance fake monotonic clock past ttl_seconds - safety_margin (120 - 18 = 102s -> advance +110s)
        fake_clock[0] += 110.0
        tail_after, cname_after = resolve_explicit_cached_request(
            client,
            model="gemini-3.8-flash",
            prefix_parts=big_prefix,
            tail_parts=["Turn t1 after long specialist run"],
            min_calls_before_create=1,
        )
        assert cname_after == "cachedContents/spy-cache-2"
        assert tail_after == ["Turn t1 after long specialist run"]
        assert "cachedContents/spy-cache-1" in client.caches.deleted


def test_board_build_bypasses_hardcoded_5_gate_even_with_default_supervisor_assembled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for live bug where `_assemble_agents()` populated `self.agents['QAPlaytesterBuildSupervisor']`
    with default `BuildSupervisor`, causing `custom_supervisor is None` to be False and running hardcoded Gate B."""
    from engine.studio.orchestrator import StudioOrchestrator, StudioOrchestratorConfig
    from engine.supervisor.models import GateFailureType, QualityGateResult
    from engine.supervisor.supervisor import BuildSupervisor

    orch = StudioOrchestrator(
        api_key="test-dummy-key",
        config_obj=StudioOrchestratorConfig(
            deliberation="Test default assembled supervisor on board build",
            gemini_api_key="test-dummy-key",
            continuous_production=True,
        ),
        enable_writers_room=False,
    )
    # Confirm _assemble_agents populated QAPlaytesterBuildSupervisor with default BuildSupervisor (NOT None!)
    assert isinstance(orch.agents["QAPlaytesterBuildSupervisor"], BuildSupervisor)
    assert orch.agents["QAPlaytesterBuildSupervisor"] is orch._default_supervisor_agent

    cand_dir = tmp_path / "candidate_genesis_test"
    cand_dir.mkdir(parents=True)
    (cand_dir / "project.godot").write_text("[application]\nconfig/name=\"Test\"\nrun/main_scene=\"res://main.tscn\"\n", encoding="utf-8")
    (cand_dir / "main.tscn").write_text("[gd_scene format=3]\n[node name=\"Main\" type=\"Node2D\"]\n", encoding="utf-8")
    (cand_dir / "backlog.json").write_text('{"items": []}', encoding="utf-8")

    eval_heal_called = False

    def _forbid_evaluate_and_heal(self_sup: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal eval_heal_called
        eval_heal_called = True
        raise AssertionError("Hardcoded 5-Gate evaluate_and_heal MUST NOT run on agent-verified board builds!")

    def _fast_gate_a(self_sup: Any, target_dir: Path) -> QualityGateResult:
        return QualityGateResult(
            deliberation="Boot check passed.",
            gate_name="Gate A",
            passed=True,
            failure_type=GateFailureType.NONE,
            details="OK",
        )

    monkeypatch.setattr(BuildSupervisor, "evaluate_and_heal", _forbid_evaluate_and_heal)
    monkeypatch.setattr(BuildSupervisor, "verify_gate_a_syntax_and_entrypoint", _fast_gate_a)

    from engine.studio.models import StudioSwarmPlan

    plan = StudioSwarmPlan(
        deliberation="Minimal verification plan for board-managed cartridge promotion.",
        plan_id="plan_board_promo",
        game_title="Test Board Promotion",
        genre_verdict="Automation",
        target_episode=0,
        specialist_agent_roles=["Supervisor"],
        dag_edges=[],
        task_manifest='{"Supervisor": "Promote agent-verified cartridge"}',
    )
    import asyncio

    report = asyncio.run(orch.execute_swarm_plan(plan, staging_dir=cand_dir))
    assert not eval_heal_called
    assert report.promoted_to_active is True
    assert (tmp_path / "active" / "project.godot").is_file()


