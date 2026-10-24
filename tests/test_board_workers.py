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
