"""Asset / animation / audio workers for the Production Board (no network: fake LLM + fake fabrication library)."""
from __future__ import annotations

import json
import math
import struct
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from engine.studio.models import AssetRequirement, StudioAgentError, WorkItem, WorkItemProposal
from engine.studio.production_board import BoardScheduler, ProductionBoard
from engine.studio.workers import (
    WorkerContext,
    animation_handler,
    audio_asset_handler,
    default_handlers,
    plan_asset_requirement,
    render_waveform_image,
    visual_asset_handler,
)


class _GenaiClient:
    __module__ = "google.genai.fake"


class _Models:
    def __init__(self, responder):
        self.responder = responder
        self.prompts = []

    def generate_content(self, model, contents, config=None):
        self.prompts.append(contents)
        return SimpleNamespace(text=self.responder(contents))


def make_client(responder):
    c = _GenaiClient()
    c.models = _Models(responder)
    return c


def req_json(item_id, mode="static_sprite", target=None, **kw):
    data = dict(
        deliberation="planned", asset_id="ignored_by_worker", name="", asset_type="prop", fabrication_mode=mode,
        purpose="p", target_path=target if target is not None else f"assets/sprites/{item_id}.png",
        description="d", visual_prompt_or_text="vp", dimensions=[512, 512],
    )
    data.update(kw)
    return json.dumps(data)


class FakeLibrary:
    """Stands in for CartridgeAssetLibrary: writes the asset + an inspection plate, persists a JSON catalog."""

    def __init__(self, ctx):
        self.dir = ctx.cartridge_dir
        self.catalog_path = self.dir / "catalog.json"
        entries = json.loads(self.catalog_path.read_text()) if self.catalog_path.is_file() else {}
        self.catalog = SimpleNamespace(entries=dict(entries))
        self.calls = []

    def fabricate_single_asset(self, req, force_regenerate=False):
        self.calls.append((req.asset_id, force_regenerate))
        time.sleep(0.01)
        dest = self.dir / req.target_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if req.fabrication_mode in ("audio_bark", "sfx", "ambient_loop"):
            with wave.open(str(dest), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(8000)
                wf.writeframes(b"".join(struct.pack("<h", int(12000 * math.sin(i / 8.0))) for i in range(4000)))
        else:
            Image.new("RGBA", (16, 16), (200, 50, 50, 255)).save(dest)
            renders = self.dir / "artifacts" / "renders"
            renders.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (10, 10, 10)).save(renders / f"plate_{req.asset_id}.png")
            if req.fabrication_mode == "sprite_sheet":
                Image.new("RGB", (64, 32), (30, 30, 90)).save(renders / f"anim_sheet_{req.asset_id}.png")
        self.catalog.entries[req.asset_id] = {"file": req.target_path, "mode": req.fabrication_mode}
        return dest

    def save_catalog(self):
        self.catalog_path.write_text(json.dumps(self.catalog.entries))


def ctx_for(tmp_path, responder, voice=None):
    (tmp_path / "game_bible.json").write_text(json.dumps({"game_title": "t"}))
    return WorkerContext(cartridge_dir=tmp_path, client=make_client(responder), voice_agent=voice,
                         library_factory=FakeLibrary)


def item(item_id="hero", kind="visual_asset", revisions=None):
    return WorkItem(deliberation="d", item_id=item_id, kind=kind, title=item_id, spec="a heroic skiff",
                    proof_spec="front view", state="in_progress", revisions=revisions or [])


def test_visual_worker_plans_fabricates_and_returns_engine_plates_as_proof(tmp_path):
    ctx = ctx_for(tmp_path, lambda c: req_json("hero"))
    result = visual_asset_handler(ctx)(item())
    names = {Path(p).name for p in result.proof_images}
    assert "plate_hero.png" in names and "hero.png" in names
    assert all(Path(p).is_file() for p in result.proof_images)
    assert "Fabricated 'hero'" in result.notes
    assert json.loads((tmp_path / "catalog.json").read_text())["hero"]["file"] == "assets/sprites/hero.png"
    assert (tmp_path / "artifacts" / "work_items" / "hero" / "requirement.json").is_file()


def test_planner_forces_item_id_and_feeds_back_validation_then_succeeds(tmp_path):
    answers = iter([req_json("x", mode="tile_surface"), req_json("x", mode="static_sprite")])
    ctx = ctx_for(tmp_path, lambda c: next(answers))
    req = plan_asset_requirement(ctx, item("hero"), ("static_sprite", "sprite_sheet"))
    assert req.asset_id == "hero" and req.fabrication_mode == "static_sprite"
    assert "FAILED VALIDATION" in ctx.client.models.prompts[1] and "must be one of" in ctx.client.models.prompts[1]


def test_planner_rejects_disallowed_mode_and_path_escape_and_raises_after_attempts(tmp_path):
    ctx = ctx_for(tmp_path, lambda c: req_json("x", mode="tile_surface", target="../../etc/passwd"))
    with pytest.raises(StudioAgentError) as ei:
        plan_asset_requirement(ctx, item("hero"), ("sprite_sheet", "articulated_parts"), attempts=2)
    assert "must be one of" in str(ei.value) and "under assets/" in str(ei.value)


def test_animation_worker_requires_animated_modes_and_returns_contact_sheet(tmp_path):
    ctx = ctx_for(tmp_path, lambda c: req_json("walk", mode="sprite_sheet", asset_type="sprite_sheet"))
    result = animation_handler(ctx)(item("walk", kind="animation"))
    assert any(Path(p).name == "anim_sheet_walk.png" for p in result.proof_images)
    ctx2 = ctx_for(tmp_path / "b", lambda c: req_json("walk", mode="static_sprite")) if (tmp_path / "b").mkdir() is None else None
    with pytest.raises(StudioAgentError):
        animation_handler(ctx2)(item("walk", kind="animation"))


def test_revision_feedback_and_previous_requirement_reach_the_planner_and_force_regeneration(tmp_path):
    from engine.studio.models import RevisionRecord
    ctx = ctx_for(tmp_path, lambda c: req_json("hero"))
    visual_asset_handler(ctx)(item())                                    # first build stores requirement.json
    rev = RevisionRecord(revision_index=0, feedback="make the sails larger", verdict="request_changes")
    calls = []
    orig = FakeLibrary.fabricate_single_asset
    FakeLibrary.fabricate_single_asset = lambda self, req, force_regenerate=False: (calls.append(force_regenerate), orig(self, req, force_regenerate))[1]
    try:
        visual_asset_handler(ctx)(item(revisions=[rev]))
    finally:
        FakeLibrary.fabricate_single_asset = orig
    prompt = ctx.client.models.prompts[-1]
    assert "make the sails larger" in prompt and "PREVIOUS REQUIREMENT" in prompt
    assert calls == [True]


def test_fabrication_without_output_or_proof_raises(tmp_path):
    class Empty(FakeLibrary):
        def fabricate_single_asset(self, req, force_regenerate=False):
            return None

    ctx = ctx_for(tmp_path, lambda c: req_json("hero"))
    ctx.library_factory = Empty
    with pytest.raises(StudioAgentError) as ei:
        visual_asset_handler(ctx)(item())
    assert "produced no file" in str(ei.value)


def test_audio_worker_returns_waveform_image_and_requires_voice_agent(tmp_path):
    ctx = ctx_for(tmp_path, lambda c: req_json("chime", mode="sfx", asset_type="sfx", target="assets/audio/chime.wav"), voice=object())
    result = audio_asset_handler(ctx)(item("chime", kind="audio_asset"))
    assert Path(result.proof_images[0]).is_file() and "waveform" in Path(result.proof_images[0]).name
    with Image.open(result.proof_images[0]) as im:
        assert im.size == (900, 220)
    ctx.voice_agent = None
    with pytest.raises(StudioAgentError):
        audio_asset_handler(ctx)(item("chime", kind="audio_asset"))


def test_waveform_image_shows_signal_not_silence(tmp_path):
    wav = tmp_path / "t.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(8000)
        wf.writeframes(b"".join(struct.pack("<h", int(20000 * math.sin(i / 5.0))) for i in range(8000)))
    img = Image.open(render_waveform_image(wav, tmp_path / "w.png")).convert("RGB")
    import numpy as np
    arr = np.asarray(img)
    cyan = int(((arr[:, :, 0] == 120) & (arr[:, :, 1] == 220) & (arr[:, :, 2] == 255)).sum())
    assert cyan > 300


def test_concurrent_workers_do_not_lose_catalog_entries(tmp_path):
    answers = lambda c: req_json(c.split("[")[1].split("]")[0] if isinstance(c, str) else "x")
    ctx = ctx_for(tmp_path, answers)
    board = ProductionBoard()
    ids = [f"asset_{i}" for i in range(8)]
    board.add_items([WorkItemProposal(deliberation="d", item_id=i, kind="visual_asset", title=i, spec="s", proof_spec="p") for i in ids], "Visual")
    sched = BoardScheduler(board, default_handlers(ctx), max_workers=8, poll_interval=0.005)
    try:
        assert sched.run_until_idle(20)
    finally:
        sched.stop()
    assert sched.peak_running >= 3
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert set(catalog) == set(ids)                                    # no lost updates under concurrency
    assert all(board.get(i).state == "awaiting_review" and board.get(i).revisions[0].proof_paths for i in ids)


def test_room_rejects_kinds_without_a_worker(tmp_path):
    import tests.test_council_backlog as cb
    from engine.studio.models import WorkItemProposal as W
    room, client, board, sched, candidate, _ = cb.build(
        tmp_path, {"all": [W(deliberation="d", item_id="story", kind="narrative", title="t", spec="s", proof_spec="p")]}, cb.approve_all)
    try:
        with pytest.raises(StudioAgentError) as ei:
            room.run_writers_room(dna=cb.base._sample_dna(), candidate_dir=candidate)
    finally:
        sched.stop()
    assert "no worker can build" in str(ei.value)


def test_audio_worker_fabricates_multi_asset_soundbank_and_stacked_waveform_proof(tmp_path):
    from engine.studio.models import AudioAssetPlanBatch

    batch_payload = AudioAssetPlanBatch(
        deliberation="The hazard soundbank spec requests 4 distinct .wav files across low scrape, high scrape, wind shear loop, and thermal plume hiss.",
        assets=[
            AssetRequirement(
                deliberation="Looping convective wind shear.",
                asset_id="wind_gust_convective_loop",
                asset_type="ambient_loop",
                fabrication_mode="ambient_loop",
                target_path="assets/audio/wind_gust_convective_loop.wav",
                visual_prompt_or_text="Seamless brown noise wind shear with sweeping notch filter 200-1800Hz.",
            ),
            AssetRequirement(
                deliberation="Low frequency metallic scrape.",
                asset_id="gantry_scrape_low",
                asset_type="sfx",
                fabrication_mode="sfx",
                target_path="assets/audio/gantry_scrape_low.wav",
                visual_prompt_or_text="Deep 100-350Hz iron girder friction scrape.",
            ),
            AssetRequirement(
                deliberation="High frequency metallic screech.",
                asset_id="gantry_scrape_high",
                asset_type="sfx",
                fabrication_mode="sfx",
                target_path="assets/audio/gantry_scrape_high.wav",
                visual_prompt_or_text="Sharp 1.2-4kHz rivet screech chatter.",
            ),
            AssetRequirement(
                deliberation="High-pressure thermal exhaust hiss.",
                asset_id="thermal_plume_hiss",
                asset_type="sfx",
                fabrication_mode="sfx",
                target_path="assets/audio/thermal_plume_hiss.wav",
                visual_prompt_or_text="High-pressure 3-10kHz vapor sizzle.",
            ),
        ],
    ).model_dump_json()

    ctx = ctx_for(tmp_path, lambda c: batch_payload, voice=object())
    work_item = item("audio_environmental_wind_and_gantry_hazards", kind="audio_asset")
    result = audio_asset_handler(ctx)(work_item)

    # First proof is the composite stacked waveform PNG; remaining 4 entries are the 4 synthesized .wav files
    assert len(result.proof_images) == 5
    waveform_png = Path(result.proof_images[0])
    assert waveform_png.is_file() and waveform_png.suffix == ".png"
    with Image.open(waveform_png) as im:
        assert im.size[0] == 900
        assert im.size[1] >= 165 * 4

    wav_files = [Path(p).name for p in result.proof_images[1:]]
    assert wav_files == [
        "wind_gust_convective_loop.wav",
        "gantry_scrape_low.wav",
        "gantry_scrape_high.wav",
        "thermal_plume_hiss.wav",
    ]
    for p in result.proof_images[1:]:
        assert Path(p).is_file()

    catalog = json.loads((tmp_path / "catalog.json").read_text())
    for expected_id in (
        "wind_gust_convective_loop",
        "gantry_scrape_low",
        "gantry_scrape_high",
        "thermal_plume_hiss",
        "audio_environmental_wind_and_gantry_hazards",
    ):
        assert expected_id in catalog


def test_iter_cartridge_gd_files_and_2hop_transitive_imports(tmp_path):
    """Verifies recursive .gd discovery across subdirectories (`scenes/`, `scripts/`) and 2-hop transitive
    reference resolution in `CartridgeSemanticIndex`."""
    from engine.sim.godot_verifier import iter_cartridge_gd_files
    from engine.studio.context_curator import CartridgeSemanticIndex
    from engine.studio.models import ModuleFileSpec

    # Preserve subdirectory paths for encapsulated feature scripts, while normalizing legacy root names
    mod_nested = ModuleFileSpec(
        deliberation="Encapsulated feature module.",
        filename="res://scenes/mechanics/rcs_attitude_trim.gd",
        content="extends Node2D\n",
    )
    assert mod_nested.filename == "scenes/mechanics/rcs_attitude_trim.gd"
    mod_root = ModuleFileSpec(
        deliberation="Root main script.",
        filename="res://scripts/main.gd",
        content="extends Node2D\n",
    )
    assert mod_root.filename == "main.gd"

    (tmp_path / "main.gd").write_text(
        'extends Node2D\nconst UmbilicalHose = preload("res://scripts/mechanics/umbilical_hose.gd")\n',
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "mechanics").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "mechanics" / "umbilical_hose.gd").write_text(
        'extends Node2D\nconst HoseJoint = preload("res://scenes/entities/hose_joint.gd")\nvar is_hose_connected: bool = false\n',
        encoding="utf-8",
    )
    (tmp_path / "scenes" / "entities").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scenes" / "entities" / "hose_joint.gd").write_text(
        "extends Node2D\nvar joint_tension: float = 0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "studio_rig_and_collider.gd").write_text("class_name StudioRigHelper\n", encoding="utf-8")
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_hose.gd").write_text("extends Node\n", encoding="utf-8")

    discovered = [
        p.relative_to(tmp_path).as_posix()
        for p in iter_cartridge_gd_files(tmp_path, include_tests=False, exclude_helper_libs=True)
    ]
    assert discovered == [
        "main.gd",
        "scenes/entities/hose_joint.gd",
        "scripts/mechanics/umbilical_hose.gd",
    ]

    sem_idx = CartridgeSemanticIndex(tmp_path)
    related_1hop = sem_idx.get_transitive_related_modules(["main.gd"], max_hops=1)
    assert "scripts/mechanics/umbilical_hose.gd" in related_1hop

    related_2hop = sem_idx.get_transitive_related_modules(["main.gd"], max_hops=2)
    assert "scripts/mechanics/umbilical_hose.gd" in related_2hop
    assert "scenes/entities/hose_joint.gd" in related_2hop


def test_cartridge_file_lock_table_and_scoped_rollback(tmp_path):
    """Verifies deadlock-free per-file lock acquisition, conflict reporting, and scoped rollbacks that never
    delete concurrent workers' files."""
    from engine.studio.workers import CartridgeFileLockTable, _restore_modules, _snapshot_modules

    table = CartridgeFileLockTable()
    ok1, conf1 = table.try_acquire_many("task_a", ["scenes/mechanics/rcs.gd", "res://scenes/ui/trim_hud.gd"])
    assert ok1 is True and conf1 == []
    assert table.held_paths("task_a") == {"scenes/mechanics/rcs.gd", "scenes/ui/trim_hud.gd"}

    # Disjoint task_b succeeds in parallel
    ok2, conf2 = table.try_acquire_many("task_b", ["scenes/mechanics/umbilical.gd"])
    assert ok2 is True and conf2 == []

    # Overlapping task_c fails atomically without stealing task_b's or task_a's locks
    ok3, conf3 = table.try_acquire_many("task_c", ["scenes/mechanics/extra.gd", "scenes/mechanics/rcs.gd"])
    assert ok3 is False
    assert conf3 == ["scenes/mechanics/rcs.gd"]
    assert table.held_paths("task_c") == set()

    # Scoped snapshot & rollback for task_a must NOT delete task_b's newly created file
    (tmp_path / "scenes" / "mechanics").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scenes" / "mechanics" / "rcs.gd").write_text("# v1 rcs\n", encoding="utf-8")
    snap_a = _snapshot_modules(tmp_path, ["scenes/mechanics/rcs.gd"], scoped_only=True)

    # Concurrent task_b creates scenes/mechanics/umbilical.gd while task_a modifies rcs.gd
    (tmp_path / "scenes" / "mechanics" / "umbilical.gd").write_text("# task_b file\n", encoding="utf-8")
    (tmp_path / "scenes" / "mechanics" / "rcs.gd").write_text("# broken v2 rcs\n", encoding="utf-8")

    # Rollback task_a with scoped_only=True
    _restore_modules(tmp_path, snap_a, scoped_only=True)
    assert (tmp_path / "scenes" / "mechanics" / "rcs.gd").read_text(encoding="utf-8") == "# v1 rcs\n"
    assert (tmp_path / "scenes" / "mechanics" / "umbilical.gd").read_text(encoding="utf-8") == "# task_b file\n"


def test_dynamic_mid_task_lock_upgrade_stash_yield_and_replay(tmp_path):
    """Verifies that when an SWE worker starts with a lock on `scenes/mechanics/thruster.gd` and discovers mid-synthesis
    that it also needs to write `main.gd`:
    1. If `main.gd` is locked by another worker, it stashes the synthesized payload with SHA-256 base hashes, expands
       `item.target_modules` on the board, and raises `WorkerYielded('file_lock_busy:main.gd')` before writing any file.
    2. When resumed after `main.gd` is released (and `main.gd` was not modified), it replays the stashed payload with
       ZERO extra LLM synthesis calls.
    3. If `main.gd` WAS modified by the other worker while yielded, the SHA-256 base-hash check invalidates the stale
       stash and re-synthesizes against the fresh `main.gd`."""
    from engine.studio.models import ModuleFileSpec, TrackCodeSynthesisPayload
    from engine.studio.production_board import ProductionBoard, WorkItemProposal, WorkerYielded
    from engine.studio.workers import CartridgeFileLockTable, WorkerContext, run_swe_engineer_specialist

    (tmp_path / "main.gd").write_text("extends Node2D\n# initial main\n", encoding="utf-8")
    (tmp_path / "project.godot").write_text("config_version=5\n", encoding="utf-8")
    (tmp_path / "main.tscn").write_text("[gd_scene format=3]\n", encoding="utf-8")

    board = ProductionBoard(tmp_path / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="Encapsulated thruster mechanic.",
                item_id="mech_thruster",
                kind="mechanic_code",
                title="Thruster Mechanic",
                spec="Build thruster module and register in main.gd.",
                proof_spec="Show thruster firing.",
                target_modules=["scenes/mechanics/thruster.gd"],
            )
        ]
    )
    work_item = board.claim_ready(limit=1)[0]
    lock_table = CartridgeFileLockTable()
    # Acquire mech_thruster's initial declared lock
    lock_table.try_acquire_many("mech_thruster", ["scenes/mechanics/thruster.gd"])
    # Simulate another worker ('other_task') holding 'main.gd'
    lock_table.try_acquire_many("other_task", ["main.gd"])

    class _FakeModularSWE:
        def __init__(self):
            self.llm_calls = 0

        def implement_requirement_track(
            self,
            track,
            staging_dir,
            requirements_spec=None,
            prior_error=None,
            extra_image_paths=None,
            board=None,
            pre_write_hook=None,
            commit_gate=None,
            allowed_files=None,
            stashed_bundle=None,
        ):
            if stashed_bundle is not None:
                bundle = stashed_bundle
            else:
                self.llm_calls += 1
                main_now = (staging_dir / "main.gd").read_text(encoding="utf-8")
                bundle = TrackCodeSynthesisPayload(
                    deliberation="Synthesized thruster module and hooked into main.gd.",
                    files=[
                        ModuleFileSpec(
                            deliberation="Encapsulated thruster script.",
                            filename="scenes/mechanics/thruster.gd",
                            content="extends Node2D\nvar thrust: float = 10.0\n",
                        ),
                        ModuleFileSpec(
                            deliberation="Hook thruster in main.gd.",
                            filename="main.gd",
                            content=main_now + "# thruster hooked\n",
                        ),
                    ],
                )
            return bundle.assemble_cartridge_directory(
                staging_dir,
                swe_agent=None,
                pre_write_hook=pre_write_hook,
                commit_gate=commit_gate,
                allowed_files=allowed_files,
            )

    swe = _FakeModularSWE()
    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        swe_agent=swe,
        file_lock_table=lock_table,
        board=board,
    )

    # Attempt 1: 'main.gd' is held by 'other_task' -> stashes payload and raises WorkerYielded before writing
    with pytest.raises(WorkerYielded) as exc_info:
        run_swe_engineer_specialist(
            ctx,
            work_item,
            brief="Implement thruster in scenes/mechanics/thruster.gd and wire into main.gd",
            target_paths=["scenes/mechanics/thruster.gd"],
            turn_index=0,
            _lock_already_held=True,
        )
    assert "file_lock_busy:main.gd" in str(exc_info.value)
    assert swe.llm_calls == 1
    # Verify NO files were written before yielding
    assert not (tmp_path / "scenes" / "mechanics" / "thruster.gd").exists()
    assert "main.gd" in board.get("mech_thruster").target_modules

    # Attempt 2: 'other_task' releases 'main.gd' WITHOUT modifying it -> replays stashed bundle with 0 new LLM calls!
    lock_table.release_all("other_task")
    report = run_swe_engineer_specialist(
        ctx,
        board.get("mech_thruster"),
        brief="Implement thruster in scenes/mechanics/thruster.gd and wire into main.gd",
        target_paths=["scenes/mechanics/thruster.gd", "main.gd"],
        turn_index=0,
        _lock_already_held=True,
    )
    assert report.verification_passed is True
    assert swe.llm_calls == 1  # Replayed from stash without another LLM call!
    assert (tmp_path / "scenes" / "mechanics" / "thruster.gd").is_file()
    assert "# thruster hooked" in (tmp_path / "main.gd").read_text(encoding="utf-8")

    # Attempt 3: Verify SHA-256 base-hash invalidation when 'main.gd' changes while yielded
    lock_table.release_all("mech_thruster")
    lock_table.try_acquire_many("mech_thruster", ["scenes/mechanics/thruster.gd"])
    lock_table.try_acquire_many("other_task", ["main.gd"])
    with pytest.raises(WorkerYielded):
        run_swe_engineer_specialist(
            ctx,
            work_item,
            brief="Add afterburner in scenes/mechanics/thruster.gd and main.gd",
            target_paths=["scenes/mechanics/thruster.gd"],
            turn_index=1,
            _lock_already_held=True,
        )
    assert swe.llm_calls == 2
    # Now 'other_task' modifies 'main.gd' before releasing lock
    (tmp_path / "main.gd").write_text("extends Node2D\n# concurrent edit by other_task\n", encoding="utf-8")
    lock_table.release_all("other_task")
    report2 = run_swe_engineer_specialist(
        ctx,
        board.get("mech_thruster"),
        brief="Add afterburner in scenes/mechanics/thruster.gd and main.gd",
        target_paths=["scenes/mechanics/thruster.gd", "main.gd"],
        turn_index=1,
        _lock_already_held=True,
    )
    assert report2.verification_passed is True
    assert swe.llm_calls == 3  # Stale stash was invalidated and re-synthesized against fresh main.gd!
    main_final = (tmp_path / "main.gd").read_text(encoding="utf-8")
    assert "# concurrent edit by other_task" in main_final
    assert "# thruster hooked" in main_final


def test_per_turn_lock_release_and_parallel_claim_with_shared_historical_modules(tmp_path):
    """Verifies that:
    1. Two tasks with distinct primary modules (`scenes/mechanics/a.gd` vs `scenes/mechanics/b.gd`) that both have
       `main.gd` in their historical `target_modules` are claimed concurrently when `locked_modules` is passed.
    2. `run_task_lead` locks only `decision.target_paths` (not historical `item.target_modules`) and releases its
       per-file locks immediately when `swe_engineer` finishes so `qa_test_engineer` runs with zero held file locks.
    """
    from PIL import Image
    from engine.studio.models import SpecialistReport, TaskLeadDecision, WorkItemProposal
    from engine.studio.production_board import ProductionBoard
    from engine.studio.workers import CartridgeFileLockTable, WorkerContext, run_task_lead
    import engine.studio.workers as workers_mod

    lock_table = CartridgeFileLockTable()
    board = ProductionBoard(tmp_path / "backlog.json")
    board.add_items(
        [
            WorkItemProposal(
                deliberation="d",
                item_id="mech_a",
                kind="mechanic_code",
                title="Mech A",
                spec="Spec A",
                proof_spec="Proof A",
                target_modules=["scenes/mechanics/a.gd", "main.gd"],
            ),
            WorkItemProposal(
                deliberation="d",
                item_id="mech_b",
                kind="mechanic_code",
                title="Mech B",
                spec="Spec B",
                proof_spec="Proof B",
                target_modules=["scenes/mechanics/b.gd", "main.gd"],
            ),
        ],
        raised_by_role="GameplaySystemsEngineer",
    )

    # Both tasks can be claimed concurrently because their primary modules are distinct and main.gd is not locked
    claimed = board.claim_ready(limit=2, locked_modules=lock_table.all_held_paths())
    assert {c.item_id for c in claimed} == {"mech_a", "mech_b"}

    (tmp_path / "main.gd").write_text("extends Node2D\n", encoding="utf-8")
    (tmp_path / "scenes" / "mechanics").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scenes" / "mechanics" / "a.gd").write_text("extends Node2D\n", encoding="utf-8")

    class _LeadModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, model, contents, config=None):
            schema = getattr(config, "response_schema", None)
            if schema is not None and getattr(schema, "__name__", "") != "TaskLeadDecision":
                from engine.studio.context_curator import ContextSelectionBatch
                batch = ContextSelectionBatch(deliberation="d", selected_chunk_ids=[])

                class _CurResp:
                    text = batch.model_dump_json()

                return _CurResp()
            self.calls += 1
            if self.calls == 1:
                dec = TaskLeadDecision(
                    deliberation="Delegate to swe_engineer targeting only scenes/mechanics/a.gd.",
                    current_strategy="Edit only scenes/mechanics/a.gd.",
                    proof_strategy="Capture filmstrip.",
                    specialist_agent="swe_engineer",
                    brief="Edit scenes/mechanics/a.gd.",
                    target_paths=["scenes/mechanics/a.gd"],
                    action="delegate",
                )
            elif self.calls == 2:
                dec = TaskLeadDecision(
                    deliberation="Delegate to qa_test_engineer.",
                    current_strategy="Capture filmstrip.",
                    proof_strategy="Capture filmstrip.",
                    specialist_agent="qa_test_engineer",
                    brief="Capture filmstrip.",
                    action="delegate",
                )
            else:
                dec = TaskLeadDecision(
                    deliberation="Submit proof.",
                    current_strategy="Submit proof.",
                    proof_strategy="Submit proof.",
                    submission_summary="Done.",
                    action="submit_to_council",
                )

            class _Resp:
                text = dec.model_dump_json()

            return _Resp()

    class _LeadClient:
        __module__ = "google.genai.fake"

        def __init__(self):
            self.models = _LeadModels()

    class _FakeSwe:
        def implement_requirement_track(self, *args, **kwargs):
            return {}

    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=_LeadClient(),
        swe_agent=_FakeSwe(),
        file_lock_table=lock_table,
        board=board,
    )

    observed_swe_locks = []
    observed_qa_locks = []
    orig_dispatch = workers_mod._dispatch_to_specialist

    def _fake_dispatch(w_ctx, w_item, decision, turn_index, _lock_already_held=False):
        if decision.specialist_agent == "swe_engineer":
            observed_swe_locks.append(set(w_ctx.file_lock_table.all_held_paths()))
            return SpecialistReport(
                turn_index=turn_index,
                specialist_agent="swe_engineer",
                brief=decision.brief,
                summary="SWE done.",
                touched_files=["scenes/mechanics/a.gd"],
                proof_images=[],
                verification_passed=True,
            )
        observed_qa_locks.append(set(w_ctx.file_lock_table.all_held_paths()))
        strip_p = tmp_path / "filmstrip_mech_a_rev0.png"
        Image.new("RGB", (64, 36), (10, 20, 30)).save(strip_p)
        return SpecialistReport(
            turn_index=turn_index,
            specialist_agent="qa_test_engineer",
            brief=decision.brief,
            summary="QA done.",
            touched_files=[],
            proof_images=[str(strip_p)],
            verification_passed=True,
        )

    workers_mod._dispatch_to_specialist = _fake_dispatch
    try:
        res = run_task_lead(ctx, claimed[0], max_turns=3)
    finally:
        workers_mod._dispatch_to_specialist = orig_dispatch

    assert len(res.proof_images) == 1
    # During swe_engineer, ONLY scenes/mechanics/a.gd was locked (NOT historical main.gd!)
    assert observed_swe_locks == [{"scenes/mechanics/a.gd"}]
    # During qa_test_engineer, all file locks had already been released!
    assert observed_qa_locks == [set()]


def test_in_progress_worker_yields_at_turn_boundary_when_council_attaches_prerequisite(tmp_path):
    from engine.studio.models import ReviewDecision
    from engine.studio.production_board import WorkerResult, WorkerYielded
    from engine.studio.workers import CartridgeFileLockTable, run_task_lead

    board = ProductionBoard(tmp_path / "backlog.json")
    lock_table = CartridgeFileLockTable()
    board.add_items(
        [
            WorkItemProposal(
                deliberation="d",
                item_id="mech_in_flight",
                kind="mechanic_code",
                title="In-Flight Mechanic",
                spec="Build mechanic",
                proof_spec="Filmstrip",
                target_modules=["scenes/mechanics/in_flight.gd", "scenes.gd"],
            ),
            WorkItemProposal(
                deliberation="d",
                item_id="mech_peer",
                kind="mechanic_code",
                title="Peer Mechanic",
                spec="Build peer",
                proof_spec="Filmstrip",
                target_modules=["scenes/mechanics/peer.gd"],
            ),
        ]
    )
    claimed_in_flight = board.claim_ready(limit=1)[0]
    assert claimed_in_flight.item_id == "mech_in_flight"
    claimed_peer = board.claim_ready(limit=1)[0]
    assert claimed_peer.item_id == "mech_peer"

    proof_img = tmp_path / "peer_proof.png"
    Image.new("RGB", (64, 36), (10, 20, 30)).save(proof_img)
    board.submit_result("mech_peer", WorkerResult(proof_images=[str(proof_img)], notes="peer done"))

    # Council reviews mech_peer and spawns a shared fix task that also blocks `mech_in_flight` and prunes `scenes.gd`
    board.apply_review(
        ReviewDecision(
            deliberation="Shared backdrop defect in scenes.gd affects both mech_peer and mech_in_flight.",
            item_id="mech_peer",
            feedback="Wait for shared backdrop fix.",
            spawned_prerequisite_tasks=[
                WorkItemProposal(
                    deliberation="Fix scenes.gd backdrop.",
                    item_id="fix_backdrop",
                    kind="level_map",
                    title="Fix Backdrop",
                    spec="Fix backdrop seams in scenes.gd.",
                    proof_spec="Seamless backdrop filmstrip.",
                    target_modules=["scenes.gd"],
                    blocks_items=["mech_peer", "mech_in_flight"],
                )
            ],
            out_of_scope_modules_to_prune=["scenes.gd"],
            verdict="request_changes",
        )
    )

    class _DummyClient:
        __module__ = "google.genai.fake"

    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=_DummyClient(),
        file_lock_table=lock_table,
        board=board,
    )

    # When `mech_in_flight` hits its next turn boundary in `run_task_lead`, it detects the newly attached
    # unapproved prerequisite `fix_backdrop`, syncs its pruned `target_modules`, and yields immediately!
    with pytest.raises(WorkerYielded) as exc_info:
        run_task_lead(ctx, claimed_in_flight, max_turns=3)
    assert "spawned_prerequisite_tasks" in str(exc_info.value)
    assert claimed_in_flight.target_modules == ["scenes/mechanics/in_flight.gd"]
    assert "scenes.gd" in claimed_in_flight.excluded_modules
    board.yield_item("mech_in_flight", reason=exc_info.value.reason)
    assert board.get("mech_in_flight").state == "blocked"


def test_track_code_synthesis_payload_applies_surgical_patches_without_full_rewrite(tmp_path):
    """Verifies that `TrackCodeSynthesisPayload` applies surgical `patches` (`SurgicalCodePatch`) to existing
    cartridge files (anchoring via `target_snippet` even when line numbers are slightly off) while preserving
    all untouched lines, and creates new files from `files`."""
    from engine.studio.models import ModuleFileSpec, SurgicalCodePatch, TrackCodeSynthesisPayload

    existing_cam = tmp_path / "scenes" / "camera" / "train_camera.gd"
    existing_cam.parent.mkdir(parents=True, exist_ok=True)
    existing_cam.write_text(
        "extends Camera2D\n"
        "\n"
        "var min_zoom: float = 0.40\n"
        "var limit_bottom_val: int = 2160\n"
        "\n"
        "func _ready() -> void:\n"
        "\tlimit_bottom = limit_bottom_val\n"
        "\tmake_current()\n",
        encoding="utf-8",
    )

    payload = TrackCodeSynthesisPayload(
        deliberation="Surgically patch train_camera.gd and add a new helper script.",
        patches=[
            SurgicalCodePatch(
                deliberation="Expand limit_bottom_val so diving tug stays in frame.",
                filename="res://scenes/camera/train_camera.gd",
                start_line=10,  # Intentionally shifted; target_snippet anchors it to lines 3..4
                end_line=11,
                target_snippet="   3: var min_zoom: float = 0.40\n   4: var limit_bottom_val: int = 2160",
                replacement_code="   3: var min_zoom: float = 0.25\n   4: var limit_bottom_val: int = 50000",
            )
        ],
        files=[
            ModuleFileSpec(
                deliberation="Create new isolated helper module.",
                filename="res://scripts/mechanics/new_helper.gd",
                content="extends Node\nfunc ping() -> bool:\n\treturn true\n",
            )
        ],
    )

    hook_seen: list[str] = []
    res = payload.assemble_cartridge_directory(
        target_dir=tmp_path,
        pre_write_hook=lambda files, obj: hook_seen.extend(files),
    )

    assert "scenes/camera/train_camera.gd" in res
    assert "scripts/mechanics/new_helper.gd" in res
    assert set(hook_seen) == {"scenes/camera/train_camera.gd", "scripts/mechanics/new_helper.gd"}

    updated_cam = existing_cam.read_text(encoding="utf-8")
    assert "var min_zoom: float = 0.25" in updated_cam
    assert "var limit_bottom_val: int = 50000" in updated_cam
    assert "func _ready() -> void:\n\tlimit_bottom = limit_bottom_val\n\tmake_current()" in updated_cam
    assert (tmp_path / "scripts" / "mechanics" / "new_helper.gd").is_file()


def test_merge_level_layout_and_skip_redundant_layout_regen(tmp_path):
    """Verifies that:
    1. `merge_level_layout` incrementally merges newly added/updated regions, entities, and triggers
       by ID onto an existing `LevelLayoutSpec` without losing unchanged regions/entities.
    2. `inject_gemini_dataset_config` enforces a default bounded `thinking_config` on text/code
       calls that omit `thinking_config`.
    """
    from engine.studio.models import (
        LevelEntitySpec,
        LevelLayoutSpec,
        LevelRegionSpec,
        LevelTriggerSpec,
        inject_gemini_dataset_config,
        merge_level_layout,
    )

    base = LevelLayoutSpec(
        deliberation="Initial world layout.",
        viewport_size=(1280.0, 720.0),
        intended_world_scale_in_viewports=4.0,
        world_bounds=(0.0, 0.0, 2560.0, 1440.0),
        world_regions=[
            LevelRegionSpec(
                deliberation="Opening bay.",
                region_id="region_a",
                bounds=(0.0, 0.0, 1280.0, 1440.0),
                progression_order=0,
            ),
            LevelRegionSpec(
                deliberation="Gantry corridor.",
                region_id="region_b",
                bounds=(1280.0, 0.0, 1280.0, 1440.0),
                progression_order=1,
            ),
        ],
        camera_framing="Follow player",
        entities=[
            LevelEntitySpec(
                deliberation="Spawn point.",
                entity_id="spawn_0",
                entity_kind="spawn",
                position=(200.0, 400.0),
            ),
            LevelEntitySpec(
                deliberation="Gate 1.",
                entity_id="gate_1",
                entity_kind="mechanism",
                position=(1400.0, 400.0),
                initial_state="locked",
            ),
        ],
        triggers=[
            LevelTriggerSpec(
                deliberation="Unlock gate.",
                trigger_id="trig_1",
                source_entity_id="spawn_0",
                condition="overlap",
                target_entity_id="gate_1",
                resulting_state="open",
            )
        ],
    )

    # Partial patch only adds one new entity in region_b without repeating region_a or spawn_0
    patch = LevelLayoutSpec(
        deliberation="Add crane entity in region_b.",
        viewport_size=(1280.0, 720.0),
        intended_world_scale_in_viewports=4.0,
        world_bounds=(0.0, 0.0, 2560.0, 1440.0),
        world_regions=[],
        camera_framing="Follow player",
        entities=[
            LevelEntitySpec(
                deliberation="Crane hazard.",
                entity_id="crane_2",
                entity_kind="hazard",
                position=(1800.0, 500.0),
            )
        ],
        triggers=[],
    )

    merged = merge_level_layout(base, patch)
    assert [r.region_id for r in merged.world_regions] == ["region_a", "region_b"]
    assert {e.entity_id for e in merged.entities} == {"spawn_0", "gate_1", "crane_2"}
    assert [t.trigger_id for t in merged.triggers] == ["trig_1"]

    # Default bounded thinking_config injection on non-media calls
    kw = inject_gemini_dataset_config(
        {"model": "gemini-3.8-flash", "contents": "Hello"},
        label="VisualArtSpriteDesignerAgent.verify_rendered_asset",
        candidate_dir=tmp_path,
    )
    assert getattr(kw["config"], "thinking_config", None) is not None
    assert kw["config"].thinking_config.thinking_budget == 1024


def test_qa_cartridge_summary_input_semantics_and_filmstrip_telemetry(tmp_path):
    """Verifies that:
    1. `_summarize_cartridge_for_qa` and `LeadGameplaySWEAgent._summarize_non_target_gd_module`
       expose exact input-to-behavior bindings (`Input.*` / `KEY_*`), physics/lighting layers
       (`collision_layer`, `collision_mask`, `CanvasModulate`), and `level_layout.json` `world_bounds`.
    2. `format_filmstrip_telemetry_progression` formats `.telemetry.json` sidecar data into a clear
       per-frame coordinate and input trajectory string.
    """
    from engine.studio.proofs import format_filmstrip_telemetry_progression
    from engine.studio.swe import LeadGameplaySWEAgent
    from engine.studio.workers import _summarize_cartridge_for_qa

    (tmp_path / "project.godot").write_text("[application]\nconfig/name=\"Test\"\n", encoding="utf-8")
    (tmp_path / "level_layout.json").write_text(
        json.dumps(
            {
                "world_bounds": [0.0, 0.0, 9600.0, 4200.0],
                "world_regions": [{"region_id": "apron", "bounds": [0.0, 0.0, 3200.0, 4200.0]}],
            }
        ),
        encoding="utf-8",
    )
    controller_src = (
        "extends RigidBody2D\n"
        "var thrust_cmd: float = 0.0\n"
        "var steer_cmd: float = 0.0\n"
        "func _ready() -> void:\n"
        "\tcollision_layer = 2\n"
        "\tcollision_mask = 5\n"
        "func _physics_process(delta: float) -> void:\n"
        "\tif Input.is_key_pressed(KEY_W):\n"
        "\t\tthrust_cmd += 1.0\n"
        "\tif Input.is_key_pressed(KEY_D):\n"
        "\t\tsteer_cmd += 1.0\n"
    )
    (tmp_path / "controller.gd").write_text(controller_src, encoding="utf-8")

    qa_sum = _summarize_cartridge_for_qa(tmp_path)
    assert "world_bounds=[0.0, 0.0, 9600.0, 4200.0]" in qa_sum
    assert "KEY_W" in qa_sum and "thrust_cmd += 1.0" in qa_sum
    assert "KEY_D" in qa_sum and "steer_cmd += 1.0" in qa_sum
    assert "collision_mask = 5" in qa_sum

    peer_sum = LeadGameplaySWEAgent._summarize_non_target_gd_module("controller.gd", controller_src)
    assert "steer_cmd += 1.0" in peer_sum
    assert "collision_mask = 5" in peer_sum

    strip_png = tmp_path / "filmstrip_test_rev0.png"
    strip_png.write_bytes(b"png")
    strip_png.with_suffix(".telemetry.json").write_text(
        json.dumps(
            {
                "ticks": [30, 160],
                "frames": [
                    {"tick": 30, "active_inputs": "in:[KEY_W]", "snapshot_summary": "pos=(440,1420)"},
                    {"tick": 160, "active_inputs": "in:[KEY_W]", "snapshot_summary": "pos=(1120,1420)"},
                ],
            }
        ),
        encoding="utf-8",
    )
    prog = format_filmstrip_telemetry_progression(strip_png)
    assert "tick 30 (in:[KEY_W]): pos=(440,1420) -> tick 160 (in:[KEY_W]): pos=(1120,1420)" in prog


def test_context_curator_indexes_build_traces_code_diffs_and_input_semantics(tmp_path):
    """Verifies that `CartridgeSemanticIndex` indexes `board:trace:<item_id>` from `lead_trace_rev*.json`
    (including specialist code diffs and `.telemetry.json` frame trajectories) and enriches `code:module:*`
    chunks with constants, input bindings, and collision/lighting layers."""
    from engine.studio.context_curator import CartridgeSemanticIndex

    (tmp_path / "project.godot").write_text("[application]\nconfig/name=\"Test\"\n", encoding="utf-8")
    (tmp_path / "backlog.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "item_id": "task_camera_fix",
                        "kind": "mechanic_code",
                        "state": "in_progress",
                        "title": "Fix Camera Tracking",
                        "spec": "Keep vehicle framed.",
                        "proof_spec": "5-frame filmstrip.",
                        "target_modules": ["camera.gd"],
                        "revisions": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "camera.gd").write_text(
        "extends Camera2D\n"
        "const DEFAULT_ZOOM: float = 0.75\n"
        "var target_zoom: float = 0.75\n"
        "func _process(delta: float) -> void:\n"
        "\tif Input.is_key_pressed(KEY_Z):\n"
        "\t\ttarget_zoom = 0.55\n",
        encoding="utf-8",
    )
    item_dir = tmp_path / "artifacts" / "work_items" / "task_camera_fix"
    item_dir.mkdir(parents=True, exist_ok=True)
    strip_p = item_dir / "filmstrip_task_camera_fix_rev0.png"
    strip_p.write_bytes(b"png")
    strip_p.with_suffix(".telemetry.json").write_text(
        json.dumps(
            {
                "ticks": [30, 90],
                "frames": [
                    {"tick": 30, "active_inputs": "in:[KEY_W]", "snapshot_summary": "pos=(440,1420)"},
                    {"tick": 90, "active_inputs": "in:[KEY_W]", "snapshot_summary": "pos=(820,1420)"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (item_dir / "lead_trace_rev0.json").write_text(
        json.dumps(
            {
                "item_id": "task_camera_fix",
                "revision": 0,
                "status": "in_progress",
                "current_strategy": "Adjust min_zoom and verify horizontal flight.",
                "turns": [
                    {
                        "turn_index": 0,
                        "specialist_agent": "swe_engineer",
                        "brief": "Raise min_zoom from 0.15 to 0.55.",
                        "summary": "Patched camera.gd",
                        "touched_files": ["camera.gd"],
                        "code_diffs": {
                            "camera.gd": "--- a/camera.gd\n+++ b/camera.gd\n-var min_zoom = 0.15\n+var min_zoom = 0.55\n"
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    sem_index = CartridgeSemanticIndex(cartridge_dir=tmp_path, client=None)
    chunks = sem_index.extract_corpus_chunks(ensure_rendering=False)
    by_id = {c.chunk_id: c for c in chunks}

    assert "board:trace:task_camera_fix" in by_id
    trace_chunk = by_id["board:trace:task_camera_fix"]
    assert "+var min_zoom = 0.55" in trace_chunk.content
    assert "tick 30 (in:[KEY_W]): pos=(440,1420) -> tick 90 (in:[KEY_W]): pos=(820,1420)" in trace_chunk.content

    assert "code:module:camera.gd" in by_id
    mod_chunk = by_id["code:module:camera.gd"]
    assert "DEFAULT_ZOOM" in mod_chunk.content
    assert "KEY_Z" in mod_chunk.content and "target_zoom = 0.55" in mod_chunk.content


def test_qa_specialist_authors_isolated_test_scene(tmp_path):
    """Verifies that `run_qa_test_engineer_specialist` writes `tests/test_<item_id>.gd` and
    `tests/test_<item_id>.tscn` when `FilmstripPlan.test_scene_gd_content` is populated, and reports
    those files in `SpecialistReport.touched_files` so `main.gd` stays free of `func test_*()` helpers."""
    from engine.studio.models import FilmstripPlan, InputSegment, WorkItem
    from engine.studio.workers import WorkerContext, run_qa_test_engineer_specialist

    (tmp_path / "project.godot").write_text("[application]\nconfig/name=\"Test\"\n", encoding="utf-8")
    plan = FilmstripPlan(
        deliberation="Author an isolated test scene in res://tests/test_dock_latch.tscn to test docking without polluting main.gd.",
        target_scene="res://tests/test_dock_latch.tscn",
        test_scene_gd_content=(
            "extends Node2D\n"
            "var latch_engaged: bool = false\n"
            "func _ready() -> void:\n"
            "\tpass\n"
            "func get_state_snapshot() -> Dictionary:\n"
            "\treturn {\"latch_engaged\": latch_engaged}\n"
        ),
        capture_ticks=[10, 40],
        input_segments=[InputSegment(deliberation="Hold accept during docking window.", from_tick=5, to_tick=35, actions=["ui_accept"])],
        frame_expectations=["Tick 10: Approaching dock", "Tick 40: Latched"],
    )

    def _fake_filmstrip(c_dir, ticks, out_path, input_script=None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"png")
        return out_path

    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=make_client(lambda _: plan.model_dump_json()),
        filmstrip_fn=_fake_filmstrip,
    )
    item = WorkItem(
        deliberation="Test dock latch mechanic.",
        item_id="mechanic_dock_latch",
        kind="mechanic_code",
        title="Dock Latch Mechanic",
        spec="Implement magnetic dock latch.",
        proof_spec="2-frame filmstrip showing latch engagement.",
        raised_by_role="GameplaySystemsEngineer",
    )
    report = run_qa_test_engineer_specialist(ctx, item, brief="Verify dock latch in isolated test scene.")
    assert (tmp_path / "tests" / "test_dock_latch.gd").is_file()
    assert (tmp_path / "tests" / "test_dock_latch.tscn").is_file()
    tscn_text = (tmp_path / "tests" / "test_dock_latch.tscn").read_text(encoding="utf-8")
    assert 'path="res://tests/test_dock_latch.gd"' in tscn_text
    assert "tests/test_dock_latch.gd" in report.touched_files
    assert "tests/test_dock_latch.tscn" in report.touched_files
    assert "Target scene: res://tests/test_dock_latch.tscn." in report.summary


def test_compute_entrypoint_reachability_and_topology_chunk(tmp_path):
    """Verifies that `compute_entrypoint_reachability` traces `project.godot` -> `main.tscn` -> `main.gd` -> `scenes.gd`
    and distinguishes reachable production modules from unreferenced modules on disk while excluding `tests/`."""
    from engine.studio.context_curator import CartridgeSemanticIndex, compute_entrypoint_reachability

    (tmp_path / "project.godot").write_text(
        '[application]\nconfig/name="Test"\nrun/main_scene="res://main.tscn"\n',
        encoding="utf-8",
    )
    (tmp_path / "main.tscn").write_text(
        '[gd_scene load_steps=2 format=3]\n'
        '[ext_resource type="Script" path="res://main.gd" id="1_main"]\n'
        '[node name="Main" type="Node2D"]\n'
        'script = ExtResource("1_main")\n',
        encoding="utf-8",
    )
    (tmp_path / "main.gd").write_text(
        'extends Node2D\nconst Scenes = preload("res://scenes.gd")\n',
        encoding="utf-8",
    )
    (tmp_path / "scenes.gd").write_text('extends Node2D\n', encoding="utf-8")
    (tmp_path / "scenes" / "ui").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scenes" / "ui" / "cockpit_hud.gd").write_text('extends CanvasLayer\n', encoding="utf-8")
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_cockpit_hud.gd").write_text(
        'extends Node2D\nconst Hud = preload("res://scenes/ui/cockpit_hud.gd")\n',
        encoding="utf-8",
    )

    reach = compute_entrypoint_reachability(tmp_path)
    assert reach["main_scene"] == "res://main.tscn"
    assert "main.tscn" in reach["reachable_from_entrypoint"]
    assert "main.gd" in reach["reachable_from_entrypoint"]
    assert "scenes.gd" in reach["reachable_from_entrypoint"]
    assert "scenes/ui/cockpit_hud.gd" in reach["unreferenced_from_entrypoint"]
    assert "tests/test_cockpit_hud.gd" not in reach["unreferenced_from_entrypoint"]

    idx = CartridgeSemanticIndex(cartridge_dir=tmp_path, client=None)
    chunks = idx.extract_corpus_chunks()
    topo_chunk = next((c for c in chunks if c.chunk_id == "code:entrypoint_topology"), None)
    assert topo_chunk is not None
    assert "scenes/ui/cockpit_hud.gd" in topo_chunk.content



