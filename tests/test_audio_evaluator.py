"""tests/test_audio_evaluator.py
Comprehensive test suite for the general-purpose multimodal `AudioEvaluatorAgent`,
`ContextCuratorAgent` audio/video curation, `TaskLead` / `qa_test_engineer` / Council
audiovisual wiring, `interactions.create` guardrails & token accounting, and strict
Opaque-Box `BlindPlaytesterAgent` on-demand audio evaluation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import wave

import imageio_ffmpeg
from PIL import Image
import pytest

from engine.playtest.agent import BlindPlaytesterAgent
from engine.playtest.godot_bridge import (
    TurnObservation,
    extract_recent_gameplay_clip_mp4,
    write_pcm16_to_wav,
)
from engine.playtest.models import (
    BlindStoreCard,
    PlaytestAudioEvaluationRequest,
    PlaytestMotorAction,
    PlaytestRunnerError,
    PlaytestSessionConfig,
    PlaytestSessionStartRequest,
    PlaytestTokenTracker,
    PlaytestTurnDecision,
)
from engine.playtest.session_manager import PlaytestSessionManager
from engine.sim.godot_verifier import mux_raw_rgb_and_pcm_to_mp4
from engine.studio.audio_evaluator import (
    AudioEvaluatorAgent,
    classify_media_path,
    enrich_catalog_audio_entries,
)
from engine.studio.context_curator import ContextCuratorAgent, ContextCuratorDecision
from engine.studio.models import (
    AssetLibraryCatalog,
    AssetLibraryEntry,
    AssetRequirement,
    AudioAssetPlanBatch,
    AudioEvaluationReport,
    FilmstripPlan,
    RevisionRecord,
    StudioAgentConfigurationError,
    StudioAgentError,
    StudioPausedError,
    TOKEN_USAGE_TRACKER,
    TaskLeadDecision,
    TokenUsageTracker,
    WorkItem,
    install_studio_gemini_guardrails,
    set_active_gemini_dataset_context,
)
from engine.studio.workers import (
    WorkerContext,
    available_specialists,
    run_qa_test_engineer_specialist,
    run_task_lead,
)
from engine.studio.writers_room import CreativeWritersRoom


def _write_sine_wav(path: Path, *, duration_sec: float = 0.25, freq_hz: float = 440.0, rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = max(64, int(round(duration_sec * rate)))
    samples = [
        int(14000.0 * math.sin(2.0 * math.pi * freq_hz * (i / float(rate))))
        for i in range(n_samples)
    ]
    raw_pcm = struct.pack("<" + "h" * len(samples), *samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(raw_pcm)
    return path


def _write_png(path: Path, color: tuple = (40, 80, 140), size: tuple = (64, 36)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _make_sample_mp4(path: Path, *, frames: int = 8, fps: int = 8) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_rgb = path.with_name(f".raw_{path.stem}.rgb")
    raw_pcm = path.with_name(f".raw_{path.stem}.pcm")
    w, h = 64, 36
    frame_bytes = bytes([30, 120, 200]) * (w * h)
    raw_rgb.write_bytes(frame_bytes * frames)
    dur_sec = frames / float(fps)
    n_samples = int(round(dur_sec * 16000))
    samples = [int(9000.0 * math.sin(2.0 * math.pi * 330.0 * (i / 16000.0))) for i in range(n_samples)]
    raw_pcm.write_bytes(struct.pack("<" + "h" * len(samples), *samples))
    try:
        return mux_raw_rgb_and_pcm_to_mp4(
            raw_rgb_path=raw_rgb,
            pcm_path=raw_pcm,
            out_mp4_path=path,
            width=w,
            height=h,
            fps=fps,
            sample_rate=16000,
        )
    finally:
        raw_rgb.unlink(missing_ok=True)
        raw_pcm.unlink(missing_ok=True)


def _sample_report(answer: str = "Crisp transient chime aligned with intended cue.") -> AudioEvaluationReport:
    return AudioEvaluationReport(
        deliberation="Listened to the full audio waveform and spectral decay before forming observations.",
        rich_acoustic_description="Clean 440 Hz tonal chime lasting ~0.25s with smooth exponential decay and zero clipping.",
        contextual_observations=["Matches a lightweight UI confirmation cue without masking background ambience."],
        notable_concerns=[],
        answer_to_inquiry=answer,
    )


class _FakeInteractionsAPI:
    def __init__(self, responder):
        self.responder = responder
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responder(kwargs)


# ---------------------------------------------------------------------------
# 1. Rationale-First Schema Invariants (Rule 4) & Fail-Fast Contracts (Rule 5)
# ---------------------------------------------------------------------------


def test_rationale_first_schema_ordering_for_audio_evaluation_models():
    for model_cls in (AudioEvaluationReport, PlaytestAudioEvaluationRequest, ContextCuratorDecision):
        field_names = list(model_cls.model_fields.keys())
        assert field_names[0] == "deliberation", f"{model_cls.__name__} must place deliberation at index 0"
        schema_props = list((model_cls.model_json_schema().get("properties") or {}).keys())
        assert schema_props[0] == "deliberation", f"{model_cls.__name__} JSON schema must emit deliberation first"

    req = PlaytestAudioEvaluationRequest(
        deliberation="Checking whether recent sound effects matched screen movement.",
        lookback_seconds=99.0,
        question="Did the action emit synchronized audio feedback?",
    )
    assert req.lookback_seconds == 20.0
    req_short = PlaytestAudioEvaluationRequest(
        deliberation="Short check.",
        lookback_seconds=0.2,
        question="Was any sound audible?",
    )
    assert req_short.lookback_seconds == 2.0


def test_audio_evaluator_inline_multimodal_payload_catalog_enrichment_and_rationale_logging(tmp_path):
    cdir = tmp_path / "cartridge"
    wav_path = _write_sine_wav(cdir / "assets" / "audio" / "cue_confirm.wav", duration_sec=0.35)
    mp4_path = _make_sample_mp4(cdir / "artifacts" / "clip_sample.mp4")
    ref_png = _write_png(cdir / "assets" / "ui_mockup.png")

    catalog = AssetLibraryCatalog(
        deliberation="Test audio catalog.",
        entries={
            "cue_confirm": AssetLibraryEntry(
                deliberation="Initial synthesis entry.",
                asset_id="cue_confirm",
                category="sfx",
                fabrication_mode="procedural_sfx",
                description="Short confirmation chime",
                file_path="assets/audio/cue_confirm.wav",
                res_path="res://assets/audio/cue_confirm.wav",
            )
        }
    )
    cat_file = cdir / "assets" / "asset_library.json"
    cat_file.parent.mkdir(parents=True, exist_ok=True)
    cat_file.write_text(catalog.model_dump_json(indent=2), encoding="utf-8")

    expected = _sample_report()

    def _respond(kwargs):
        return SimpleNamespace(
            id="inter_001",
            output_text=expected.model_dump_json(),
            usage=SimpleNamespace(
                total_input_tokens=420,
                total_cached_tokens=100,
                total_output_tokens=85,
                total_thought_tokens=20,
                total_tokens=525,
            ),
        )

    interactions_api = _FakeInteractionsAPI(_respond)
    fake_client = SimpleNamespace(interactions=interactions_api)

    trace_events: List[tuple] = []
    cb_events: List[tuple] = []
    fake_trace = SimpleNamespace(log_event=lambda name, payload: trace_events.append((name, payload)))

    agent = AudioEvaluatorAgent(
        client=fake_client,
        model_name="gemini-3.8-flash",
        trace_logger=fake_trace,
        event_callback=lambda et, pl: cb_events.append((et, pl)),
    )
    report = agent.evaluate(
        inquiry_prompt="Evaluate whether the confirmation cue and gameplay clip sound clean and balanced.",
        media_paths=[wav_path, mp4_path, ref_png],
        curate_studio_context=True,
        cartridge_dir=cdir,
        item_id="sfx_item_1",
        item_kind="audio_asset",
        enrich_catalog=True,
    )

    assert report.answer_to_inquiry == expected.answer_to_inquiry
    assert agent.last_usage["prompt_tokens"] == 420
    assert agent.last_usage["cached_tokens"] == 100
    assert agent.last_usage["output_tokens"] == 85
    assert agent.last_usage["thought_tokens"] == 20

    assert len(interactions_api.calls) == 1
    sent = interactions_api.calls[0]
    assert sent["model"] == "gemini-3.8-flash"
    assert sent["response_format"]["type"] == "text"
    assert sent["response_format"]["mime_type"] == "application/json"
    assert list(sent["response_format"]["schema"]["properties"].keys())[0] == "deliberation"

    input_types = [part.get("type") for part in sent["input"] if isinstance(part, dict)]
    assert "audio" in input_types
    assert "video" in input_types
    assert "image" in input_types

    # Verify Rule 4 reasoning_rationale observability logging
    assert trace_events and trace_events[0][0] == "audio_evaluation_completed"
    assert trace_events[0][1]["reasoning_rationale"] == expected.deliberation
    assert cb_events and cb_events[0][1]["reasoning_rationale"] == expected.deliberation

    # Verify asset catalog enrichment wrote perceptual_description & duration_seconds
    updated_cat = AssetLibraryCatalog.model_validate_json(cat_file.read_text(encoding="utf-8"))
    entry = updated_cat.entries["cue_confirm"]
    assert entry.perceptual_description == expected.rich_acoustic_description
    assert entry.duration_seconds == pytest.approx(0.35, abs=0.03)


def test_audio_evaluator_files_upload_fallback_for_large_media(tmp_path):
    mp4_path = _make_sample_mp4(tmp_path / "large_clip.mp4")
    upload_calls: List[Dict[str, Any]] = []
    get_calls: List[str] = []
    deleted_names: List[str] = []

    states = iter(["PROCESSING", "ACTIVE"])

    class _FakeFilesAPI:
        def upload(self, *, file, config):
            upload_calls.append({"file": file, "config": config})
            return SimpleNamespace(name="files/temp_clip_99", state=next(states), uri="https://generativelanguage.googleapis.com/v1beta/files/temp_clip_99")

        def get(self, *, name):
            get_calls.append(name)
            return SimpleNamespace(name=name, state=next(states), uri="https://generativelanguage.googleapis.com/v1beta/files/temp_clip_99")

        def delete(self, *, name):
            deleted_names.append(name)

    interactions_api = _FakeInteractionsAPI(
        lambda _kw: SimpleNamespace(output_text=_sample_report("Uploaded URI evaluated.").model_dump_json(), usage=None)
    )
    fake_client = SimpleNamespace(interactions=interactions_api, files=_FakeFilesAPI())

    # Set inline_max_bytes smaller than mp4_path size so Files API upload path is exercised
    agent = AudioEvaluatorAgent(client=fake_client, inline_max_bytes=1024)
    # Patch time.sleep locally via monkeypatch-free fast path? Wait: 1.5s once is fast, let's keep test snappy
    orig_sleep = time.sleep
    time.sleep = lambda _s: None
    try:
        rep = agent.evaluate(
            inquiry_prompt="Check large clip audio alignment.",
            media_paths=[mp4_path],
            curate_studio_context=False,
        )
    finally:
        time.sleep = orig_sleep

    assert rep.answer_to_inquiry == "Uploaded URI evaluated."
    assert len(upload_calls) == 1
    assert get_calls == ["files/temp_clip_99"]
    assert deleted_names == ["files/temp_clip_99"]
    vid_parts = [p for p in interactions_api.calls[0]["input"] if p.get("type") == "video"]
    assert len(vid_parts) == 1
    assert vid_parts[0]["uri"].endswith("files/temp_clip_99")
    assert "data" not in vid_parts[0]


def test_audio_evaluator_rule_5_fail_fast_errors(tmp_path):
    valid_wav = _write_sine_wav(tmp_path / "ok.wav")

    # 1. Missing client
    with pytest.raises(StudioAgentConfigurationError):
        AudioEvaluatorAgent(client=None).evaluate(inquiry_prompt="Test", media_paths=[valid_wav], curate_studio_context=False)

    # 2. Client missing interactions.create
    with pytest.raises(StudioAgentConfigurationError):
        AudioEvaluatorAgent(client=SimpleNamespace()).evaluate(
            inquiry_prompt="Test", media_paths=[valid_wav], curate_studio_context=False
        )

    good_client = SimpleNamespace(
        interactions=_FakeInteractionsAPI(
            lambda _kw: SimpleNamespace(output_text=_sample_report().model_dump_json(), usage=None)
        )
    )
    evaluator = AudioEvaluatorAgent(client=good_client)

    # 3. Missing file
    with pytest.raises(StudioAgentError, match="does not exist"):
        evaluator.evaluate(inquiry_prompt="Test", media_paths=[tmp_path / "missing.wav"], curate_studio_context=False)

    # 4. Empty 0-byte file
    zero_file = tmp_path / "empty.wav"
    zero_file.write_bytes(b"")
    with pytest.raises(StudioAgentError, match="empty"):
        evaluator.evaluate(inquiry_prompt="Test", media_paths=[zero_file], curate_studio_context=False)

    # 5. Corrupt WAV file
    corrupt_wav = tmp_path / "corrupt.wav"
    corrupt_wav.write_bytes(b"NOT_A_RIFF_WAV_HEADER_1234567890")
    with pytest.raises(StudioAgentError, match="invalid WAV"):
        evaluator.evaluate(inquiry_prompt="Test", media_paths=[corrupt_wav], curate_studio_context=False)

    # 6. Unsupported file extension
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hello", encoding="utf-8")
    with pytest.raises(StudioAgentError, match=r"(?i)unsupported media.*extension"):
        classify_media_path(txt_file)

    # 7. Empty output_text from model
    empty_client = SimpleNamespace(
        interactions=_FakeInteractionsAPI(lambda _kw: SimpleNamespace(output_text="", outputs=[], usage=None))
    )
    with pytest.raises(StudioAgentError, match="empty `output_text`"):
        AudioEvaluatorAgent(client=empty_client).evaluate(
            inquiry_prompt="Test", media_paths=[valid_wav], curate_studio_context=False
        )


# ---------------------------------------------------------------------------
# 2. Guardrails & Token Accounting on `client.interactions.create`
# ---------------------------------------------------------------------------


def test_studio_guardrails_and_token_trackers_wrap_interactions_create(tmp_path, monkeypatch):
    interactions_api = _FakeInteractionsAPI(
        lambda kwargs: SimpleNamespace(
            id="inter_guard_1",
            output_text=_sample_report().model_dump_json(),
            usage=SimpleNamespace(
                total_input_tokens=310,
                total_cached_tokens=60,
                total_output_tokens=90,
                total_thought_tokens=15,
                total_tokens=415,
            ),
        )
    )
    class _RealLikeGenaiClient:
        __module__ = "google.genai.client"

        def __init__(self, models, interactions):
            self.models = models
            self.interactions = interactions

    client = _RealLikeGenaiClient(
        models=SimpleNamespace(generate_content=lambda *a, **k: None),
        interactions=interactions_api,
    )
    cdir = tmp_path / "candidates" / "cand_01"
    cdir.mkdir(parents=True, exist_ok=True)
    pause_file = cdir.parent / ".studio_paused.json"
    try:
        install_studio_gemini_guardrails(
            client,
            candidate_dir=cdir,
            store_logs=True,
        )

        before_snap = TOKEN_USAGE_TRACKER.get_summary()
        resp = client.interactions.create(model="gemini-3.8-flash", input=[{"type": "text", "text": "ping"}])
        after_snap = TOKEN_USAGE_TRACKER.get_summary()

        # Injected labels + store=True at top level (never `config=...`)
        sent_kw = interactions_api.calls[0]
        assert "config" not in sent_kw
        assert sent_kw.get("store") is True
        assert isinstance(sent_kw.get("labels"), dict)

        assert after_snap["total_prompt_tokens"] - before_snap["total_prompt_tokens"] == 310
        assert after_snap["total_cached_tokens"] - before_snap["total_cached_tokens"] == 60
        assert after_snap["total_output_tokens"] - before_snap["total_output_tokens"] == 90
        assert after_snap["total_thought_tokens"] - before_snap["total_thought_tokens"] == 15

        # Re-recording the exact same response on a tracker must not double-count
        isolated_tracker = TokenUsageTracker()
        resp_dup = SimpleNamespace(
            id="inter_guard_dup",
            usage=SimpleNamespace(total_input_tokens=100, total_cached_tokens=0, total_output_tokens=20, total_thought_tokens=5, total_tokens=125),
        )
        isolated_tracker.record_response(resp_dup, label="inter_test")
        isolated_tracker.record_response(resp_dup, label="inter_test")
        assert isolated_tracker.get_summary()["total_calls"] == 1

        # PlaytestTokenTracker also parses interactions.Usage cleanly
        pt_tracker = PlaytestTokenTracker()
        pt_delta = pt_tracker.record_response(resp)
        assert pt_delta == {
            "prompt_tokens": 310,
            "cached_tokens": 60,
            "output_tokens": 90,
            "thought_tokens": 15,
        }

        # Verify pause gate halts interactions.create when .studio_paused.json + STUDIO_FAIL_FAST_ON_PAUSE=1
        monkeypatch.setenv("STUDIO_FAIL_FAST_ON_PAUSE", "1")
        pause_file.write_text(json.dumps({"paused": True}), encoding="utf-8")
        with pytest.raises(StudioPausedError):
            client.interactions.create(model="gemini-3.8-flash", input=[{"type": "text", "text": "blocked"}])
    finally:
        pause_file.unlink(missing_ok=True)
        set_active_gemini_dataset_context(dataset_name="", candidate_dir=None)


# ---------------------------------------------------------------------------
# 3. ContextCuratorAgent Multimodal Curation (Mockups + WAV + MP4, Zero Substring Blocks)
# ---------------------------------------------------------------------------


def test_context_curator_attaches_mockup_wav_and_mp4_for_audio_roles_without_substring_blocks(tmp_path):
    cdir = tmp_path / "cartridge"
    mockup_png = _write_png(cdir / "artifacts" / "concept_mockup.png")
    wav_cue = _write_sine_wav(cdir / "assets" / "audio" / "ambient_wind.wav", duration_sec=0.4)
    clip_mp4 = _make_sample_mp4(cdir / "artifacts" / "gameplay_clip_test.mp4")

    (cdir / "game_bible.json").write_text(
        json.dumps(
            {
                "game_title": "Acoustic Horizon",
                "visual_style": {"direction": "Crisp atmospheric minimalism"},
                "audio_direction": {"mood": "Sparse resonant acoustic textures"},
                "visual_mockups": [{"image_path": "artifacts/concept_mockup.png", "caption": "16:9 World Mockup"}],
            }
        ),
        encoding="utf-8",
    )
    catalog = AssetLibraryCatalog(
        deliberation="Test ambient catalog.",
        entries={
            "ambient_wind": AssetLibraryEntry(
                deliberation="Catalogued ambient cue.",
                asset_id="ambient_wind",
                category="ambient",
                fabrication_mode="procedural_sfx",
                description="Gentle resonant wind bed",
                perceptual_description="Soft airy low-mid shimmer with gentle harmonic swell.",
                duration_seconds=0.4,
                file_path="assets/audio/ambient_wind.wav",
                res_path="res://assets/audio/ambient_wind.wav",
            )
        }
    )
    (cdir / "assets").mkdir(parents=True, exist_ok=True)
    (cdir / "assets" / "asset_library.json").write_text(catalog.model_dump_json(indent=2), encoding="utf-8")
    (cdir / "main.gd").write_text(
        'extends Node2D\nfunc _ready():\n\tvar s = load("res://assets/audio/ambient_wind.wav")\n',
        encoding="utf-8",
    )

    curator = ContextCuratorAgent(cartridge_dir=cdir, client=None)
    chunks = curator.semantic_index.extract_corpus_chunks(ensure_rendering=False)
    aud_chunk = next(ch for ch in chunks if ch.chunk_id == "asset:catalog:ambient_wind")
    assert "Soft airy low-mid shimmer" in aud_chunk.content
    assert "duration=0.40s" in aud_chunk.title
    assert aud_chunk.associated_image_path == str(wav_cue)

    # LLM curator explicitly chooses the concept mockup + .wav + .mp4 for 'audio_evaluator'
    dec = ContextCuratorDecision(
        deliberation="Attaching world concept mockup, synthesized wind WAV, and gameplay MP4 clip so audio_evaluator can judge world cohesion.",
        role_tailored_synthesis="Complete multimodal context selected for acoustic world-cohesion review.",
        selected_chunk_ids=["bible:audio_direction", "asset:catalog:ambient_wind"],
        selected_media_paths=[
            "artifacts/concept_mockup.png",
            "assets/audio/ambient_wind.wav",
            str(clip_mp4),
        ],
    )
    selected_paths = curator._select_curated_media_paths(
        target_agent_role="audio_evaluator",
        item_kind="audio_asset",
        curator_selected_paths=dec.selected_media_paths,
        extra_image_paths=[str(clip_mp4)],
        cartridge_dir=cdir,
        max_selected_images=4,
    )
    assert len(selected_paths) == 3
    parts = curator._build_multimodal_parts_from_paths(selected_paths)
    mime_types = [
        getattr(getattr(p, "inline_data", None), "mime_type", None)
        for p in parts
        if not isinstance(p, str)
    ]
    assert "image/png" in mime_types
    assert "audio/wav" in mime_types
    assert "video/mp4" in mime_types


# ---------------------------------------------------------------------------
# 4. TaskLead Delegation, QA Audiovisual Capture Forwarding & Council `_proof_parts`
# ---------------------------------------------------------------------------


def test_task_lead_delegates_to_audio_designer_then_audio_evaluator_and_council_attaches_av_proofs(tmp_path):
    cdir = tmp_path / "cartridge"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "project.godot").write_text('[application]\nconfig/name="Tone Lab"\n', encoding="utf-8")

    class _FakeVoiceAgent:
        def generate_sfx(self, *, prompt: str, output_path: Path, kind: str = "sfx", purpose: str = "", loop: bool = False) -> Path:
            return _write_sine_wav(Path(output_path), duration_sec=0.3, freq_hz=520.0)

    interactions_api = _FakeInteractionsAPI(
        lambda _kw: SimpleNamespace(
            id="inter_lead_1",
            output_text=_sample_report("Synthesized cue is crisp, resonant, and ready for Council review.").model_dump_json(),
            usage=SimpleNamespace(total_input_tokens=200, total_cached_tokens=0, total_output_tokens=60, total_thought_tokens=10, total_tokens=270),
        )
    )

    lead_step = {"n": 0}

    class _FakeModelsAPI:
        def generate_content(self, *, model, contents, config=None):
            schema = getattr(config, "response_schema", None)
            schema_name = getattr(schema, "__name__", "")
            if schema_name == "AudioAssetPlanBatch":
                batch = AudioAssetPlanBatch(
                    deliberation="Plan single procedural cue.",
                    assets=[
                        AssetRequirement(
                            deliberation="Synthesize chime.",
                            asset_id="ui_pulse",
                            asset_type="sfx",
                            fabrication_mode="sfx",
                            target_path="assets/audio/ui_pulse.wav",
                            description="Short resonant pulse",
                            visual_prompt_or_text="clean resonant pulse chime",
                        )
                    ],
                )
                return SimpleNamespace(text=batch.model_dump_json())
            lead_step["n"] += 1
            if lead_step["n"] == 1:
                dec = TaskLeadDecision(
                    deliberation="First synthesize the requested audio asset via audio_designer.",
                    current_strategy="Synthesize ui_pulse.wav.",
                    proof_strategy="Render waveform and then perceptually evaluate the WAV.",
                    action="delegate",
                    specialist_agent="audio_designer",
                    brief="Synthesize a clean 0.3s resonant UI pulse at assets/audio/ui_pulse.wav.",
                    target_paths=["assets/audio/ui_pulse.wav"],
                )
            elif lead_step["n"] == 2:
                dec = TaskLeadDecision(
                    deliberation="Now ask audio_evaluator to listen directly to ui_pulse.wav and check timbral clarity.",
                    current_strategy="Perceptually evaluate ui_pulse.wav with audio_evaluator.",
                    proof_strategy="Confirm acoustic quality before submitting to Council.",
                    action="delegate",
                    specialist_agent="audio_evaluator",
                    brief="Does ui_pulse.wav sound crisp and free of harsh distortion or dead air?",
                    target_paths=["assets/audio/ui_pulse.wav"],
                )
            else:
                dec = TaskLeadDecision(
                    deliberation="Both waveform proof and perceptual audio evaluation confirm the cue is ready.",
                    current_strategy="Submit verified audio cue to Council.",
                    proof_strategy="Attach waveform PNG and synthesized WAV.",
                    action="submit_to_council",
                    submission_summary="Synthesized and perceptually verified ui_pulse.wav.",
                )
            return SimpleNamespace(text=dec.model_dump_json())

    class _FakeGenaiClient:
        __module__ = "google.genai.fake"

        def __init__(self, models, interactions):
            self.models = models
            self.interactions = interactions

    fake_client = _FakeGenaiClient(models=_FakeModelsAPI(), interactions=interactions_api)
    ctx = WorkerContext(
        cartridge_dir=cdir,
        client=fake_client,
        voice_agent=_FakeVoiceAgent(),
    )

    aud_item = WorkItem(
        deliberation="Create UI pulse cue.",
        item_id="ui_pulse",
        kind="audio_asset",
        title="UI Pulse Audio Asset",
        spec="Create a short resonant pulse cue.",
        proof_spec="Waveform plate + synthesized WAV.",
    )
    assert available_specialists(ctx, aud_item) == ["audio_designer"]

    res = run_task_lead(ctx, aud_item, max_turns=5)
    assert "audio_designer" in res.notes or "audio_evaluator" in res.notes
    assert "Synthesized cue is crisp" in res.notes
    assert any(p.endswith(".png") for p in res.proof_images)
    assert any(p.endswith(".wav") for p in res.proof_images)

    # Verify persisted audio_eval_turn1_rev0.json artifact and enriched asset catalog
    eval_json = cdir / "artifacts" / "work_items" / "ui_pulse" / "audio_eval_turn1_rev0.json"
    assert eval_json.is_file()
    cat_data = AssetLibraryCatalog.model_validate_json((cdir / "assets" / "asset_library.json").read_text(encoding="utf-8"))
    assert "440 Hz tonal chime" in cat_data.entries["ui_pulse"].perceptual_description

    # Verify QA Test Engineer forwards `audiovisual_mp4_out_path` when `record_audiovisual_clip=True`
    qa_item = WorkItem(
        deliberation="Verify live soundscape.",
        item_id="sound_sync_check",
        kind="integration",
        title="Verify Gameplay Audio Sync",
        spec="Ensure action emits synchronized sound.",
        proof_spec="Filmstrip + synchronized gameplay MP4 clip.",
    )
    captured_av_arg: Dict[str, Any] = {}

    def _fake_filmstrip_fn(cartridge_dir, ticks, out_path, input_script=None, audiovisual_mp4_out_path=None):
        _write_png(Path(out_path))
        captured_av_arg["audiovisual_mp4_out_path"] = audiovisual_mp4_out_path
        if audiovisual_mp4_out_path is not None:
            _make_sample_mp4(Path(audiovisual_mp4_out_path), frames=6, fps=6)
        return Path(out_path)

    class _FakeQAModels:
        def generate_content(self, *, model, contents, config=None):
            plan = FilmstripPlan(
                deliberation="Record synchronized audiovisual clip alongside contact sheet.",
                council_test_scene_summary="Runs gameplay action while recording Bus 0 audio.",
                capture_ticks=[1, 15, 30],
                record_audiovisual_clip=True,
            )
            return SimpleNamespace(text=plan.model_dump_json())

    qa_ctx = WorkerContext(
        cartridge_dir=cdir,
        client=_FakeGenaiClient(models=_FakeQAModels(), interactions=interactions_api),
        filmstrip_fn=_fake_filmstrip_fn,
    )
    qa_rep = run_qa_test_engineer_specialist(qa_ctx, qa_item, brief="Capture contact sheet and synchronized MP4 clip.")
    assert captured_av_arg["audiovisual_mp4_out_path"] is not None
    assert any(p.endswith(".mp4") for p in qa_rep.proof_images)

    # Verify WritersRoomCouncil._proof_parts attaches image/png + audio/wav + video/mp4
    rev_item = WorkItem(
        deliberation="Review item with multimodal proofs.",
        item_id="sound_sync_check",
        kind="integration",
        title="Verify Gameplay Audio Sync",
        spec="Spec",
        proof_spec="Proof",
        revisions=[
            RevisionRecord(
                deliberation="Submitted proofs.",
                revision_index=0,
                worker_notes="Captured strip, WAV, and MP4.",
                proof_paths=list(qa_rep.proof_images) + [str(cdir / "assets" / "audio" / "ui_pulse.wav")],
            )
        ],
    )
    council = object.__new__(CreativeWritersRoom)
    c_parts = council._proof_parts(rev_item, cdir)
    c_mimes = [
        getattr(getattr(p, "inline_data", None), "mime_type", None)
        for p in c_parts
        if not isinstance(p, str)
    ]
    assert "image/png" in c_mimes
    assert "audio/wav" in c_mimes
    assert "video/mp4" in c_mimes


# ---------------------------------------------------------------------------
# 5. Opaque-Box Blind Playtester On-Demand Audio Evaluation (`request_audio_evaluation`)
# ---------------------------------------------------------------------------


def test_extract_recent_gameplay_clip_mp4_and_opaque_box_playtester_audio_evaluation(tmp_path, monkeypatch):
    # 1. Verify `extract_recent_gameplay_clip_mp4` slices a pure 1280x720 MKV + 16kHz PCM into a valid MP4
    media_dir = tmp_path / "raw_stream_test"
    media_dir.mkdir(parents=True, exist_ok=True)
    raw_mkv = media_dir / "raw_game_stream.mkv"
    sess_pcm = media_dir / "session_audio.pcm"

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    res_mkv = subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-f", "lavfi",
            "-i", "color=c=navy:s=1280x720:r=60:d=3.0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            str(raw_mkv),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res_mkv.returncode == 0 and raw_mkv.is_file()

    n_pcm = 3 * 16000
    pcm_samples = [int(10000.0 * math.sin(2.0 * math.pi * 400.0 * (i / 16000.0))) for i in range(n_pcm)]
    sess_pcm.write_bytes(struct.pack("<" + "h" * len(pcm_samples), *pcm_samples))

    sliced_mp4 = extract_recent_gameplay_clip_mp4(
        raw_game_mkv=raw_mkv,
        session_pcm_path=sess_pcm,
        end_tick=180,  # 3.0s
        lookback_seconds=2.0,
        output_mp4_path=media_dir / "lookback_slice.mp4",
    )
    assert sliced_mp4.is_file() and sliced_mp4.stat().st_size > 512

    with pytest.raises(PlaytestRunnerError):
        extract_recent_gameplay_clip_mp4(
            raw_game_mkv=media_dir / "nonexistent.mkv",
            session_pcm_path=sess_pcm,
            end_tick=60,
            lookback_seconds=2.0,
            output_mp4_path=media_dir / "fail.mp4",
        )

    # 2. Enforce strict Opaque-Box Isolation: if ContextCuratorAgent.curate_context is ever touched during playtesting, fail!
    def _forbidden_curate_context(*args, **kwargs):
        raise AssertionError("OPAQUE-BOX VIOLATION: ContextCuratorAgent.curate_context must NEVER run during BlindPlaytester!")

    monkeypatch.setattr(ContextCuratorAgent, "curate_context", _forbidden_curate_context)

    cartridges_root = tmp_path / "cartridges"
    cp_dir = cartridges_root / "checkpoints" / "cp_slice_1"
    cp_dir.mkdir(parents=True, exist_ok=True)
    (cp_dir / "project.godot").write_text('[application]\nconfig/name="Mystery Signal"\n', encoding="utf-8")
    (cp_dir / "manifest.json").write_text(
        json.dumps(
            {
                "title": "Mystery Signal",
                "narrative_premise": "Tune mysterious receivers across a quiet observatory.",
                "internal_secret_spoiler": "DO_NOT_LEAK_INTERNAL_BIBLE_SECRET_999",
            }
        ),
        encoding="utf-8",
    )
    (cp_dir / "game_bible.json").write_text(
        json.dumps({"secret_design_notes": "DO_NOT_LEAK_INTERNAL_BIBLE_SECRET_999"}),
        encoding="utf-8",
    )

    turn_prompts_seen: List[str] = []
    interaction_Payloads: List[Dict[str, Any]] = []

    def _on_interaction(kwargs: Dict[str, Any]):
        interaction_Payloads.append(kwargs)
        rep = AudioEvaluationReport(
            deliberation="Listened to the 3.0s gameplay lookback clip from an external player's perspective.",
            rich_acoustic_description="Warm analog hum followed by a distinct metallic click at 1.4s when the dial moved.",
            contextual_observations=["The click aligns tightly with the cursor interaction and fits the quiet observatory mood."],
            notable_concerns=[],
            answer_to_inquiry="Yes—interacting with the dial produced a crisp, well-timed click over a subtle ambient hum.",
        )
        return SimpleNamespace(
            id="inter_pt_01",
            output_text=rep.model_dump_json(),
            usage=SimpleNamespace(
                total_input_tokens=260,
                total_cached_tokens=0,
                total_output_tokens=75,
                total_thought_tokens=12,
                total_tokens=347,
            ),
        )

    turn_counter = {"n": 0}

    class _FakePlaytestModels:
        def generate_content(self, *, model, contents, config=None):
            # Flatten text parts so we can inspect what BlindPlaytesterAgent received on Turn 2
            flat_txt = []
            if isinstance(contents, list):
                for item in contents:
                    for p in getattr(item, "parts", []) or []:
                        if getattr(p, "text", None):
                            flat_txt.append(p.text)
            joined_txt = "\n".join(flat_txt)
            schema = getattr(config, "response_schema", None)
            if isinstance(schema, dict) and "overall_fun_rating" in (schema.get("properties") or {}):
                from engine.playtest.models import PlaytestExperienceReport

                exp = PlaytestExperienceReport(
                    deliberation="Synthesizing session report.",
                    overall_experience_summary="Tested receiver dial and verified audio feedback.",
                    visual_and_ui_defects_observed=[],
                    graphics_and_visuals="Clear layout.",
                    gameplay_and_feel="Responsive dial interaction.",
                    sound_and_audio="Warm hum with crisp dial clicks.",
                    intuitiveness_and_onboarding="Intuitive.",
                    goals_pacing_and_accomplishment="Good pacing.",
                    world_and_atmosphere="Cohesive observatory mood.",
                    overall_fun_rating=7,
                )
                return SimpleNamespace(text=exp.model_dump_json(), usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=50))

            turn_counter["n"] += 1
            turn_prompts_seen.append(joined_txt)
            if turn_counter["n"] == 1:
                dec = PlaytestTurnDecision(
                    deliberation="Click the center dial and request a focused audio evaluation on the last 3 seconds.",
                    turn_log_entry="Observed opening observatory screen.",
                    current_mood="curious",
                    current_plan="Click the dial and check whether the audio matches the interaction.",
                    motor_action=PlaytestMotorAction(
                        deliberation="Click center.",
                        mouse_x=500,
                        mouse_y=500,
                        mouse_button_action="left_click",
                        duration_ms=500,
                    ),
                    request_audio_evaluation=PlaytestAudioEvaluationRequest(
                        deliberation="Want to verify whether clicking the dial produces synchronized sound.",
                        lookback_seconds=3.0,
                        question="Did clicking the dial produce a matching sound and how does the ambience feel?",
                    ),
                )
            else:
                from engine.playtest.models import PlaytestEndDecision

                dec = PlaytestTurnDecision(
                    deliberation="Received focused audio evaluation confirming the dial click; wrapping up session.",
                    turn_log_entry="Confirmed dial click and soundscape response.",
                    current_mood="satisfied",
                    current_plan="Finish playtest.",
                    motor_action=PlaytestMotorAction(deliberation="Idle.", duration_ms=350),
                    end_playtest=PlaytestEndDecision(
                        deliberation="Completed check.",
                        outcome="finished_exploring",
                        closing_thoughts="Enjoyed the responsive audio feedback.",
                    ),
                )
            return SimpleNamespace(
                text=dec.model_dump_json(),
                usage_metadata=SimpleNamespace(prompt_token_count=150, candidates_token_count=60),
            )

    class _FakeBridge:
        def __init__(self, *, checkpoint_dir, session_dir, reaction_ticks=15, on_live_frame=None):
            self.media_dir = Path(session_dir) / "media"
            self.media_dir.mkdir(parents=True, exist_ok=True)
            self.raw_game_mkv = self.media_dir / "raw_game_stream.mkv"
            self.session_pcm_path = self.media_dir / "session_audio.pcm"
            # Copy prebuilt valid MKV + PCM into session media_dir so live MP4 lookback slicing runs end-to-end!
            self.raw_game_mkv.write_bytes(raw_mkv.read_bytes())
            self.session_pcm_path.write_bytes(sess_pcm.read_bytes())
            self.last_tick = 0

        def start(self) -> TurnObservation:
            self.last_tick = 30
            s0 = _write_png(self.media_dir / "turn_0000.png", color=(20, 30, 60), size=(1280, 720))
            w0 = _write_sine_wav(self.media_dir / "turn_0000.wav", duration_sec=0.5)
            return TurnObservation(
                observation_index=0,
                tick=30,
                sim_time_sec=0.5,
                cursor_x=640.0,
                cursor_y=360.0,
                screenshot_path=s0,
                preview_path=s0,
                wav_path=w0,
                audio_rms=0.2,
                audio_peak=0.45,
                audio_is_silent=False,
                interrupted_early=False,
            )

        def step_action(self, action) -> TurnObservation:
            self.last_tick = 120
            s1 = _write_png(self.media_dir / "turn_0001.png", color=(45, 70, 110), size=(1280, 720))
            w1 = _write_sine_wav(self.media_dir / "turn_0001.wav", duration_sec=0.5, freq_hz=660.0)
            return TurnObservation(
                observation_index=1,
                tick=120,
                sim_time_sec=2.0,
                cursor_x=640.0,
                cursor_y=360.0,
                screenshot_path=s1,
                preview_path=s1,
                wav_path=w1,
                audio_rms=0.25,
                audio_peak=0.5,
                audio_is_silent=False,
                interrupted_early=False,
            )

        def close(self) -> None:
            pass

    bus_events: List[Dict[str, Any]] = []
    fake_bus = SimpleNamespace(publish_threadsafe=lambda pkt: bus_events.append(pkt))
    fake_playtest_client = SimpleNamespace(
        models=_FakePlaytestModels(),
        interactions=_FakeInteractionsAPI(_on_interaction),
    )

    mgr = PlaytestSessionManager(
        db_path=tmp_path / "telemetry.db",
        cartridges_root=cartridges_root,
        sessions_root=tmp_path / "sessions",
        event_bus=fake_bus,
        client_factory=lambda: fake_playtest_client,
        bridge_factory=_FakeBridge,
    )
    started = mgr.start_session(
        PlaytestSessionStartRequest(
            deliberation="Launch opaque-box playtest with audio evaluation.",
            checkpoint_id="cp_slice_1",
            max_turns=5,
        )
    )
    sid = started["session_id"]

    deadline = time.monotonic() + 20.0
    detail = None
    while time.monotonic() < deadline:
        detail = mgr.get_session_detail(sid)
        if detail and detail["session"]["status"] in ("completed", "error", "stopped"):
            break
        time.sleep(0.1)

    assert detail is not None
    assert detail["session"]["status"] == "completed", f"Unexpected session error: {detail['session'].get('error_message')}"

    # Verify AudioEvaluatorAgent received ONLY BlindStoreCard title + public description + sliced MP4 (zero internal spoilers!)
    assert len(interaction_Payloads) == 1
    sent_text_parts = [p["text"] for p in interaction_Payloads[0]["input"] if p.get("type") == "text"]
    combined_eval_prompt = "\n".join(sent_text_parts)
    assert "Mystery Signal" in combined_eval_prompt
    assert "Tune mysterious receivers across a quiet observatory." in combined_eval_prompt
    assert "DO_NOT_LEAK_INTERNAL_BIBLE_SECRET_999" not in combined_eval_prompt
    assert any(p.get("type") == "video" and p.get("mime_type") == "video/mp4" for p in interaction_Payloads[0]["input"])

    # Verify Turn 2 prompt received the queued Focused Audio Evaluation Result!
    assert len(turn_prompts_seen) >= 2
    assert "Focused Audio Evaluation Result (from your prior turn request):" in turn_prompts_seen[1]
    assert "Warm analog hum followed by a distinct metallic click" in turn_prompts_seen[1]

    # Verify `playtest_audio_evaluated` event was published to SwarmEventBus AND token usage included interaction tokens
    audio_evts = [e for e in bus_events if e.get("event_type") == "playtest_audio_evaluated"]
    assert len(audio_evts) == 1
    assert audio_evts[0]["payload"]["clip_name"] == "audio_eval_turn_0001.mp4"
    assert detail["session"]["prompt_tokens"] >= 150 + 260 + 150
