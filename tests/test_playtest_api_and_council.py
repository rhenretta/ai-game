"""tests/test_playtest_api_and_council.py
Integration tests for `/api/playtest/*` Mission Control endpoints, HTTP 206 Range video/WAV
streaming, single-item & batched Wish-to-Council dispatch, and multimodal screenshot +
slice-provenance ingestion inside `CreativeWritersRoom._execute_steering_pivot_turns`.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from PIL import Image
from starlette.testclient import TestClient

from engine.playtest.godot_bridge import TurnObservation, write_pcm16_to_wav
from engine.playtest.session_manager import PlaytestSessionManager
from engine.studio.models import GameBibleSpec, GameSliceSpec
from engine.studio.production_board import ProductionBoard
from engine.studio.writers_room import CreativeWritersRoom, WritersRoomTurnProposal
from engine.telemetry.db import TelemetryDB
from engine.ui.server import SwarmEventBus, create_app


class _FakeBridge:
    """Fast in-process bridge double for API route & lifecycle testing."""

    def __init__(self, *, checkpoint_dir: Path, session_dir: Path, reaction_ticks: int = 15, on_live_frame: Any = None) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.session_dir = session_dir
        self.media_dir = session_dir / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.raw_game_mkv = self.media_dir / "raw_game_stream.mkv"
        self.session_pcm_path = self.media_dir / "session_audio.pcm"
        self.session_pcm_path.write_bytes(b"\x00\x10" * 1600)
        self.last_tick = 0
        self._obs_idx = 0

    def _emit_obs(self, tick: int) -> TurnObservation:
        self.last_tick = tick
        png_path = self.media_dir / f"obs_{self._obs_idx:04d}.png"
        Image.new("RGB", (1280, 720), (30, 45, 70)).save(png_path, format="PNG")
        wav_path = write_pcm16_to_wav(b"\x00\x18\x00\xe8" * 800, self.media_dir / f"turn_{self._obs_idx:04d}.wav")
        obs = TurnObservation(
            observation_index=self._obs_idx,
            tick=tick,
            sim_time_sec=round(tick / 60.0, 2),
            cursor_x=640.0 + self._obs_idx * 25.0,
            cursor_y=360.0,
            screenshot_path=png_path,
            preview_path=png_path,
            wav_path=wav_path,
            audio_rms=0.12,
            audio_peak=0.22,
            audio_is_silent=False,
            interrupted_early=False,
        )
        self._obs_idx += 1
        return obs

    def start(self) -> TurnObservation:
        return self._emit_obs(30)

    def step_action(self, action: Any) -> TurnObservation:
        return self._emit_obs(self.last_tick + 45)

    def close(self) -> None:
        return None


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    db_path = tmp_path / "telemetry.db"
    cartridges_root = tmp_path / "cartridges"
    active_dir = cartridges_root / "active"
    cand_dir = cartridges_root / "candidate_001"
    cp_dir = cartridges_root / "checkpoints" / "candidate_001_slice_0"

    for d in (active_dir, cand_dir, cp_dir):
        d.mkdir(parents=True, exist_ok=True)
        (d / "project.godot").write_text('[application]\nconfig/name="API Fixture Game"\n', encoding="utf-8")
        (d / "manifest.json").write_text(
            json.dumps(
                {
                    "deliberation": "Fixture manifest.",
                    "title": "API Fixture Game",
                    "narrative_premise": "Explore a modular observatory.",
                    "candidate_id": "candidate_001",
                    "slice_id": "slice_0",
                }
            ),
            encoding="utf-8",
        )

    (cp_dir / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "candidate_001_slice_0",
                "candidate_id": "candidate_001",
                "requirement_id": "item_core_1",
                "verified_tracks": ["item_core_1"],
            }
        ),
        encoding="utf-8",
    )

    bible = GameBibleSpec(
        deliberation="Fixture Game Bible.",
        cartridge_id="candidate_001",
        target_episode=1,
        status="in_production",
        game_title="API Fixture Game",
        core_genre="exploration",
        elevator_pitch="Explore a modular observatory.",
        slice_roadmap=[
            GameSliceSpec(
                deliberation="Completed initial slice.",
                slice_id="slice_0",
                slice_index=0,
                title="Opening Courtyard",
                status="completed",
                slice_objective="Reach the central lift.",
                player_acclimation_and_mechanics_focus="Basic movement and switch activation.",
                expansion_from_prior_slice="Initial playable slice.",
                scope_boundaries=["Courtyard and lift."],
            )
        ],
    )
    (cand_dir / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")
    return db_path, cartridges_root, active_dir, cand_dir


def test_playtest_ui_markup_rest_endpoints_and_range_streaming(tmp_path: Path):
    db_path, cartridges_root, active_dir, _ = _setup_workspace(tmp_path)
    bus = SwarmEventBus()

    def _fake_llm_factory() -> Any:
        def _gen(*, model: str, contents: Any, config: Any) -> Any:
            schema = getattr(config, "response_schema", None)
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            usage = SimpleNamespace(
                prompt_token_count=1000,
                cached_content_token_count=750,
                candidates_token_count=120,
                thoughts_token_count=35,
            )
            if "overall_experience_summary" in props:
                report_payload = {
                    "deliberation": "Synthesizing end-of-session player report for manual operator run.",
                    "overall_experience_summary": "Atmospheric opening courtyard, though the wall switch lacked a hover cue.",
                    "graphics_and_visuals": "Clean cohesive architectural silhouettes; small interactive panels blend into masonry.",
                    "gameplay_and_feel": "Responsive cursor interaction once the switch was located.",
                    "sound_and_audio": "Pleasant ambient hum and clear lift activation tone.",
                    "intuitiveness_and_onboarding": "Took exploratory clicking to discover which wall panel was interactive.",
                    "goals_pacing_and_accomplishment": "Lighting up the central lift felt rewarding.",
                    "world_and_atmosphere": "Quiet mysterious observatory mood.",
                    "key_highlights": ["Lift activation lighting and audio cue"],
                    "key_pain_points": ["Interactive switch lacked hover highlight"],
                    "overall_fun_rating": 7,
                }
                return SimpleNamespace(text=json.dumps(report_payload), usage_metadata=usage)

            payload = {
                "deliberation": "Observed opening screen and submitting a player wish before finishing.",
                "current_plan": "Test courtyard switch and wrap up.",
                "turn_log_entry": "Clicked switch and noted missing hover cue.",
                "rulebook_updates": [
                    {"deliberation": "Left click activates switch", "key_or_topic": "Left Click", "behavior": "Toggles switch"}
                ],
                "impression_updates": [
                    {
                        "deliberation": "First-time discovery of interactive switch lacked hover affordance.",
                        "pillar": "intuitiveness_and_controls",
                        "impression": "Interactive switches work on left-click but have no visual hover cue yet.",
                    }
                ],
                "motor_action": {
                    "deliberation": "Click switch",
                    "mouse_x": 755.0,
                    "mouse_y": 390.0,
                    "mouse_button_action": "left_click",
                    "drag_from_x": None,
                    "drag_from_y": None,
                    "hold_keys": [],
                    "tap_keys": [],
                    "duration_ms": 500,
                    "inspect_high_res_next": False,
                },
                "mood_shift": {
                    "deliberation": "Switch worked once found.",
                    "mood": "intrigued",
                    "note": "Took a moment to spot the switch, then the lift lit up nicely.",
                },
                "submit_wish": {
                    "deliberation": "Switch blended into wall.",
                    "category": "visual_clarity",
                    "summary": "Highlight interactive switches on hover",
                    "rationale": "I clicked around the wall before realizing the small panel was interactive.",
                },
                "end_playtest": {
                    "deliberation": "Reached end of slice.",
                    "outcome": "completed",
                    "closing_thoughts": "Enjoyed the atmosphere; clearer switch affordance would help.",
                },
            }
            return SimpleNamespace(text=json.dumps(payload), usage_metadata=usage)

        return SimpleNamespace(models=SimpleNamespace(generate_content=_gen))

    pt_mgr = PlaytestSessionManager(
        db_path=db_path,
        cartridges_root=cartridges_root,
        sessions_root=tmp_path / "sessions",
        event_bus=bus,
        client_factory=_fake_llm_factory,
        bridge_factory=_FakeBridge,
    )
    app = create_app(
        db_path=db_path,
        active_cartridge_dir=active_dir,
        cartridges_root=cartridges_root,
        event_bus=bus,
        playtest_manager=pt_mgr,
    )

    with TestClient(app) as client:
        # 1. Verify HTML workspace tab, accessible tablist/label markup, & CLS-safe <video width="1600" height="720" preload="metadata">
        idx_res = client.get("/")
        assert idx_res.status_code == 200
        html = idx_res.text
        assert 'id="ws-tab-playtester"' in html
        assert 'id="pt-obs-view-report"' in html
        assert 'role="tablist"' in html
        assert 'for="pt-producer-note"' in html
        assert 'id="pt-replay-video"' in html
        assert 'width="1600"' in html and 'height="720"' in html
        assert 'preload="metadata"' in html

        # 2. List playable targets (includes completed slice_0)
        list_res = client.get("/api/playtest/sessions")
        assert list_res.status_code == 200
        targets = list_res.json()["playable_targets"]
        assert any(t.get("slice_id") == "slice_0" for t in targets)

        # 3. Start manual operator session by completed slice_id
        start_res = client.post(
            "/api/playtest/sessions/start",
            json={"candidate_id": "candidate_001", "slice_id": "slice_0", "max_turns": 5},
        )
        assert start_res.status_code == 200
        session_id = start_res.json()["session"]["session_id"]

        # Wait briefly for 1-turn session to self-terminate via `end_playtest`
        import time
        for _ in range(40):
            det_res = client.get(f"/api/playtest/sessions/{session_id}")
            assert det_res.status_code == 200
            det_json = det_res.json()
            if det_json["session"]["status"] == "completed":
                break
            time.sleep(0.05)

        assert det_json["session"]["status"] == "completed"
        assert det_json["session"]["initiated_by"] == "operator"
        assert det_json["session"]["end_outcome"] == "completed"
        assert det_json["session"]["rulebook"] == {"Left Click": "Toggles switch"}
        assert "hover cue" in det_json["session"]["impressions"]["intuitiveness_and_controls"]
        assert det_json["session"]["session_report"]["overall_fun_rating"] == 7
        assert len(det_json["wishes"]) == 1
        # CRITICAL BOUNDARY: Manual operator runs MUST NOT auto-deliver wishes to the Council!
        assert det_json["wishes"][0]["status"] == "pending"

        # 4. Seed synthetic MP4 bytes in session media dir and verify HTTP 206 Partial Content Range streaming
        mp4_file = tmp_path / "sessions" / session_id / "media" / "replay.mp4"
        mp4_file.write_bytes(b"FAKE_MP4_STREAM_" * 256)
        range_res = client.get(
            f"/api/playtest/sessions/{session_id}/video",
            headers={"Range": "bytes=0-127"},
        )
        assert range_res.status_code == 206
        assert range_res.headers["accept-ranges"] == "bytes"
        assert range_res.headers["content-range"].startswith("bytes 0-127/")
        assert len(range_res.content) == 128

        # Path traversal on media endpoint must fail fast with 400
        trav_res = client.get(f"/api/playtest/sessions/{session_id}/media/..%2Fsecret.txt")
        assert trav_res.status_code in (400, 404)

        # 5. Test Wish Dismiss endpoint on a newly inserted wish
        extra_wish_id = TelemetryDB(db_path).insert_playtest_wish(
            {
                "session_id": session_id,
                "turn_index": 1,
                "candidate_id": "candidate_001",
                "checkpoint_id": "candidate_001_slice_0",
                "observed_on_slice_id": "slice_0",
                "category": "audio",
                "summary": "Optional ambient wind volume",
                "rationale": "Wind loop felt slightly loud.",
                "deliberation": "Testing wish dismissal.",
            }
        )
        dis_res = client.post(f"/api/playtest/wishes/{extra_wish_id}/dismiss")
        assert dis_res.status_code == 200
        assert dis_res.json()["wish"]["status"] == "dismissed"

        # 6. Resetting a stopped/completed session clears its data to 'idle' WITHOUT starting a new playtester
        reset_res = client.post(
            f"/api/playtest/sessions/{session_id}/control",
            json={"deliberation": "Reset completed session data without starting.", "action": "reset"},
        )
        assert reset_res.status_code == 200
        reset_sess = reset_res.json()["session"]
        assert reset_sess["session_id"] == session_id
        assert reset_sess["status"] == "idle"
        assert reset_sess["turn_count"] == 0
        assert reset_sess["rulebook"] == {}
        assert reset_sess["impressions"] == {}
        assert reset_sess["session_report"] == {}
        assert reset_sess["has_video"] is False
        assert not mp4_file.exists()

        det_after_reset = client.get(f"/api/playtest/sessions/{session_id}").json()
        assert det_after_reset["session"]["status"] == "idle"
        assert det_after_reset["turns"] == []
        assert det_after_reset["wishes"] == []
        assert det_after_reset["compactions"] == []
        assert pt_mgr._handles == {}


def test_single_and_batched_wish_dispatch_and_multimodal_council_ingestion(tmp_path: Path):
    db_path, cartridges_root, active_dir, cand_dir = _setup_workspace(tmp_path)
    bus = SwarmEventBus()
    pt_mgr = PlaytestSessionManager(
        db_path=db_path,
        cartridges_root=cartridges_root,
        sessions_root=tmp_path / "sessions",
        event_bus=bus,
    )

    # Create real PNG screenshots for Wish 1 (1-click send) and Wishes 2+3 (batched send)
    shot1 = tmp_path / "shot_wish_1.png"
    shot2 = tmp_path / "shot_wish_2.png"
    Image.new("RGB", (640, 360), (40, 90, 140)).save(shot1, format="PNG")
    Image.new("RGB", (640, 360), (140, 70, 50)).save(shot2, format="PNG")

    db = TelemetryDB(db_path)
    db.upsert_playtest_session(
        {
            "session_id": "pt_wish_test",
            "checkpoint_id": "candidate_001_slice_0",
            "candidate_id": "candidate_001",
            "slice_id": "slice_0",
            "game_title": "API Fixture Game",
            "status": "completed",
            "deliberation": "Seeded session for wish dispatch testing.",
        }
    )
    w1 = db.insert_playtest_wish(
        {
            "session_id": "pt_wish_test",
            "turn_index": 3,
            "candidate_id": "candidate_001",
            "checkpoint_id": "candidate_001_slice_0",
            "observed_on_slice_id": "slice_0",
            "category": "controls_ux",
            "summary": "Show subtle prompt near lift lever",
            "rationale": "I walked past the lever twice before realizing E interacted with it.",
            "mood_at_time": "confused",
            "screenshot_path": str(shot1),
            "video_timestamp_sec": 4.5,
            "deliberation": "Player missed lever interaction cue.",
        }
    )
    w2 = db.insert_playtest_wish(
        {
            "session_id": "pt_wish_test",
            "turn_index": 7,
            "candidate_id": "candidate_001",
            "checkpoint_id": "candidate_001_slice_0",
            "observed_on_slice_id": "slice_0",
            "category": "gameplay_feel",
            "summary": "Slightly faster lift ascent",
            "rationale": "Waiting on the platform felt a bit slow after pulling the lever.",
            "mood_at_time": "bored",
            "screenshot_path": str(shot2),
            "video_timestamp_sec": 11.2,
            "deliberation": "Lift pacing felt slow.",
        }
    )
    w3 = db.insert_playtest_wish(
        {
            "session_id": "pt_wish_test",
            "turn_index": 9,
            "candidate_id": "candidate_001",
            "checkpoint_id": "candidate_001_slice_0",
            "observed_on_slice_id": "slice_0",
            "category": "audio",
            "summary": "Add click sound when lever latches",
            "rationale": "An audio click would confirm the lever locked in.",
            "mood_at_time": "curious",
            "screenshot_path": str(shot2),
            "video_timestamp_sec": 14.8,
            "deliberation": "Audio feedback wish.",
        }
    )

    live_steered_messages: List[str] = []
    bus._active_orchestrator = SimpleNamespace(
        writers_room=SimpleNamespace(
            queue_live_steering=lambda msg, candidate_dir=None: live_steered_messages.append(msg)
        )
    )

    app = create_app(
        db_path=db_path,
        active_cartridge_dir=active_dir,
        cartridges_root=cartridges_root,
        event_bus=bus,
        playtest_manager=pt_mgr,
    )

    with TestClient(app) as client:
        # A. 1-Click single wish dispatch (`len(wish_ids) == 1`)
        res_single = client.post(
            "/api/playtest/wishes/send-to-council",
            json={
                "wish_ids": [w1],
                "producer_note": "Prioritize lever affordance in UX.",
            },
        )
        assert res_single.status_code == 200
        assert len(res_single.json()["wishes"]) == 1
        assert res_single.json()["wishes"][0]["status"] == "sent_to_council"
        assert len(live_steered_messages) == 1

        # B. Batched multi-wish dispatch (`len(wish_ids) == 2`)
        res_batch = client.post(
            "/api/playtest/wishes/send-to-council",
            json={
                "wish_ids": [w2, w3],
                "producer_note": "Tighten lift pacing and latch SFX together.",
            },
        )
        assert res_batch.status_code == 200
        assert len(res_batch.json()["wishes"]) == 2
        assert all(item["status"] == "sent_to_council" for item in res_batch.json()["wishes"])
        assert len(live_steered_messages) == 2

    # Both `.pending_playtest_wishes.json` (3 wishes total) and `.pending_steering.json` exist on cand_dir
    pending_wishes_file = cand_dir / ".pending_playtest_wishes.json"
    pending_steering_file = cand_dir / ".pending_steering.json"
    assert pending_wishes_file.is_file()
    assert pending_steering_file.is_file()
    assert len(json.loads(pending_wishes_file.read_text(encoding="utf-8"))) == 3

    # C. Verify `CreativeWritersRoom._execute_steering_pivot_turns` consumes `.pending_playtest_wishes.json`,
    #    injects `slice_0` provenance + HEAD reconciliation instructions, AND attaches screenshot PNG Parts!
    recorded_contents: List[Any] = []

    def _council_generate_content(*, model: str, contents: Any, config: Any) -> Any:
        recorded_contents.append(contents)
        proposal = WritersRoomTurnProposal(
            deliberation="Inspected Blind Playtester screenshots from slice_0 and queuing UX/audio refinements.",
            spoken_dialogue="We see the Blind Playtester missed the lever prompt on slice_0; adding hover highlight and latch SFX task.",
            addressed_to=["CreativeDirector"],
            backlog_additions=[
                {
                    "deliberation": "Add lever hover prompt and latch audio cue.",
                    "item_id": "ux_lever_feedback",
                    "title": "Lever Hover Affordance & Latch Feedback",
                    "kind": "ui",
                    "spec": "Display crisp interaction indicator when approaching interactive levers and trigger latch confirmation.",
                    "target_modules": ["ui_hud.gd"],
                    "depends_on": [],
                    "blocks_items": [],
                    "proof_spec": "Verify indicator appears within interaction radius in res://tests/test_ux_lever_feedback.tscn.",
                }
            ],
            ready_to_lock_bible=False,
        )
        return SimpleNamespace(
            text=proposal.model_dump_json(),
            usage_metadata=SimpleNamespace(
                prompt_token_count=1500,
                cached_content_token_count=0,
                candidates_token_count=220,
                thoughts_token_count=80,
            ),
        )

    fake_council_client = SimpleNamespace(models=SimpleNamespace(generate_content=_council_generate_content))
    board = ProductionBoard(path=cand_dir / "production_board.json")
    wr = CreativeWritersRoom(client=fake_council_client, board=board)
    loaded_bible = GameBibleSpec.model_validate_json((cand_dir / "game_bible.json").read_text(encoding="utf-8"))

    steered_msgs = wr._consume_pending_steering(loaded_bible, candidate_dir=cand_dir)
    assert len(steered_msgs) == 2
    wr._execute_steering_pivot_turns(
        loaded_bible,
        "Exploration player profile.",
        cand_dir,
        steered_msgs,
    )

    # `.pending_playtest_wishes.json` was atomically consumed
    assert not pending_wishes_file.exists()
    assert len(recorded_contents) >= 1

    first_call_contents = recorded_contents[0]
    assert isinstance(first_call_contents, list), "Expected multimodal list contents (screenshot Parts + prompt string)"
    prompt_str = first_call_contents[-1]
    assert isinstance(prompt_str, str)
    assert "APPROVED BLIND PLAYTESTER WISHES" in prompt_str
    assert "slice_0" in prompt_str
    assert "candidate_001_slice_0" in prompt_str

    inline_png_parts = [
        p for p in first_call_contents[:-1]
        if getattr(p, "inline_data", None) is not None and p.inline_data.mime_type == "image/png"
    ]
    assert len(inline_png_parts) == 2, f"Expected 2 unique playtest wish PNG screenshots attached, got {len(inline_png_parts)}"
    assert any(it.item_id == "ux_lever_feedback" for it in board.items())


def test_orphaned_or_reloaded_session_recovers_and_muxes_replay_mp4(tmp_path: Path):
    import subprocess
    import imageio_ffmpeg
    from engine.playtest.godot_bridge import render_turn_sidebar_png

    db_path, cartridges_root, active_dir, _ = _setup_workspace(tmp_path)
    sessions_root = tmp_path / "sessions"
    sid = "pt_reloaded_unmuxed"
    media_dir = sessions_root / sid / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Generate a real 1.5s (90 ticks @ 60fps) 1280x720 Matroska raw stream via portable imageio-ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    raw_mkv = media_dir / "raw_game_stream.mkv"
    gen_cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", "color=c=0x162447:s=1280x720:r=60:d=1.5",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(raw_mkv),
    ]
    subprocess.run(gen_cmd, check=True, capture_output=True)
    assert raw_mkv.is_file() and raw_mkv.stat().st_size > 0

    for t_idx in (0, 1, 2):
        render_turn_sidebar_png(
            media_dir / f"sidebar_turn_{t_idx:04d}.png",
            game_title="API Fixture Game",
            turn_index=t_idx,
            sim_time_sec=0.5 * (t_idx + 1),
            current_mood="curious",
            current_plan="Testing replay recovery after reload.",
            sticky_feedback_items=[],
            action_summary="Move (640,360)",
            cursor_xy=(640.0, 360.0),
        )
    (media_dir / "session_audio.pcm").write_bytes(b"\x00\x00" * 16000)
    (media_dir / "final_tail.pcm").write_bytes(b"\x00\x00" * 8000)

    db = TelemetryDB(db_path)
    db.upsert_playtest_session(
        {
            "session_id": sid,
            "checkpoint_id": "candidate_001_slice_0",
            "candidate_id": "candidate_001",
            "slice_id": "slice_0",
            "game_title": "API Fixture Game",
            "status": "running",  # Simulates server reload interrupting active worker before finally: mux
            "turn_count": 2,
            "sim_ticks_total": 90,
            "sim_duration_sec": 1.5,
            "video_path": None,
            "deliberation": "Active session interrupted by server reload.",
        }
    )
    for t_idx, (st, et) in ((1, (30, 60)), (2, (60, 90))):
        db.insert_playtest_turn(
            {
                "session_id": sid,
                "turn_index": t_idx,
                "sim_tick_start": st,
                "sim_tick_end": et,
                "video_timestamp_sec": round(st / 60.0, 3),
                "current_plan": "Testing recovery.",
                "deliberation": "Recorded turn prior to reload.",
                "action": {"deliberation": "No-op", "mouse_button_action": "none", "duration_ms": 500},
                "append_log_line": f"[Turn {t_idx}] test",
            }
        )

    bus = SwarmEventBus()
    pt_mgr = PlaytestSessionManager(
        db_path=db_path,
        cartridges_root=cartridges_root,
        sessions_root=sessions_root,
        event_bus=bus,
    )
    app = create_app(
        db_path=db_path,
        active_cartridge_dir=active_dir,
        cartridges_root=cartridges_root,
        event_bus=bus,
        playtest_manager=pt_mgr,
    )

    with TestClient(app) as client:
        det_res = client.get(f"/api/playtest/sessions/{sid}")
        assert det_res.status_code == 200
        sess_json = det_res.json()["session"]
        assert sess_json["status"] == "stopped"
        assert sess_json["end_outcome"] == "server_reloaded"
        assert sess_json["has_video"] is True
        assert not (media_dir / "final_tail.pcm").exists()
        assert (media_dir / "replay.mp4").is_file()

        vid_res = client.get(f"/api/playtest/sessions/{sid}/video", headers={"Range": "bytes=0-255"})
        assert vid_res.status_code == 206
        assert len(vid_res.content) == 256


def test_end_of_slice_council_blind_playtest_deliberation_iterates_then_approves(tmp_path: Path):
    """Verifies the full automated End-of-Slice Blind Playtester -> Council Roundtable -> Verdict loop:
    1. Pass #0 (`initiated_by="council"`): Blind Playtester evaluates the slice cold, records sparse pillar
       impressions + a wish, synthesizes a 6-pillar `PlaytestExperienceReport`, and auto-delivers all wishes
       (`status="sent_to_council"`).
    2. Council Roundtable (`UXExperienceDesigner`, `GameplayDesigner`, `VisualArtDirector`, `AudioAtmosphereDesigner`)
       receives the report, wishes, rulebook, and multimodal PNG screenshots, and Creative Director selects
       `verdict="iterate_current_slice"` spawning a targeted `ui` fix task onto `ProductionBoard`.
    3. `BoardScheduler` executes the spawned fix task and Council approves its proof.
    4. Pass #1 (`initiated_by="council"`): Blind Playtester re-plays the improved slice (`overall_fun_rating=9`),
       Council approves (`verdict="approve_slice"`), marks `slice_0` `completed`, writes
       `artifacts/blind_playtest_report_slice_0.json`, and feeds carryover notes into next-slice reflection.
    """
    from engine.studio.models import (
        BlindPlaytestCouncilTurn,
        BlindPlaytestCouncilVerdict,
        ReviewBatch,
        ReviewDecision,
        WorkItemProposal,
    )
    from engine.studio.production_board import BoardScheduler, WorkerResult

    db_path, cartridges_root, _, cand_dir = _setup_workspace(tmp_path)

    # Set slice_0 to active so `_drain_backlog` drives it to completion after end-of-slice playtest approval
    bible = GameBibleSpec(
        deliberation="Active initial slice ready for end-of-slice blind playtest.",
        cartridge_id="candidate_001",
        target_episode=1,
        status="in_production",
        game_title="API Fixture Game",
        core_genre="exploration",
        elevator_pitch="Explore a modular observatory.",
        active_slice_index=0,
        slice_roadmap=[
            GameSliceSpec(
                deliberation="Opening slice under Council verification.",
                slice_id="slice_0",
                slice_index=0,
                title="Opening Courtyard",
                status="active",
                slice_objective="Activate the courtyard switch and ride the lift.",
                player_acclimation_and_mechanics_focus="Cursor interaction and switch discovery.",
                expansion_from_prior_slice="Initial playable slice.",
                scope_boundaries=["Courtyard and lift."],
            )
        ],
    )
    (cand_dir / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    playtest_run_counter = {"count": 0}

    def _playtester_llm_factory() -> Any:
        run_num = playtest_run_counter["count"]
        playtest_run_counter["count"] += 1

        def _gen(*, model: str, contents: Any, config: Any) -> Any:
            schema = getattr(config, "response_schema", None)
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            usage = SimpleNamespace(
                prompt_token_count=900,
                cached_content_token_count=600,
                candidates_token_count=140,
                thoughts_token_count=40,
            )
            if "overall_experience_summary" in props:
                if run_num == 0:
                    rep = {
                        "deliberation": "Pass 0 report: switch lacked visual hover cue and audio click.",
                        "overall_experience_summary": "Promising observatory courtyard, but finding the interactive switch felt like guesswork.",
                        "graphics_and_visuals": "Cohesive stone palette, though interactive switch blends into background wall.",
                        "gameplay_and_feel": "Lift movement feels smooth once triggered.",
                        "sound_and_audio": "Ambient wind is pleasant; switch click itself was silent.",
                        "intuitiveness_and_onboarding": "Unclear at t=0 which courtyard element responds to left-click.",
                        "goals_pacing_and_accomplishment": "Activating the central lift gave a clear milestone.",
                        "world_and_atmosphere": "Quiet atmospheric courtyard.",
                        "key_highlights": ["Smooth lift activation"],
                        "key_pain_points": ["Switch blends into wall without hover highlight"],
                        "overall_fun_rating": 6,
                    }
                else:
                    rep = {
                        "deliberation": "Pass 1 report: switch hover highlight made onboarding immediately intuitive.",
                        "overall_experience_summary": "Intuitive, cohesive opening slice with clear hover feedback and rewarding lift activation.",
                        "graphics_and_visuals": "Crisp interactive silhouette and hover glow on courtyard switch.",
                        "gameplay_and_feel": "Snappy click response and smooth lift ride.",
                        "sound_and_audio": "Clear switch chime complements ambient courtyard hum.",
                        "intuitiveness_and_onboarding": "Immediately understood what to click without any text wall.",
                        "goals_pacing_and_accomplishment": "Clear short-term goal and satisfying lift ascent.",
                        "world_and_atmosphere": "Inviting observatory courtyard.",
                        "key_highlights": ["Immediate hover affordance", "Rewarding lift transition"],
                        "key_pain_points": [],
                        "overall_fun_rating": 9,
                    }
                return SimpleNamespace(text=json.dumps(rep), usage_metadata=usage)

            turn_payload = {
                "deliberation": f"Pass {run_num} blind playtest turn.",
                "current_plan": "Click courtyard switch and complete slice check.",
                "turn_log_entry": "Tested courtyard switch interaction.",
                "rulebook_updates": [
                    {"deliberation": "Left click toggles courtyard switch", "key_or_topic": "Left Click", "behavior": "Activates switch"}
                ],
                "impression_updates": [
                    {
                        "deliberation": "Recording sparse intuitiveness impression.",
                        "pillar": "intuitiveness_and_controls",
                        "impression": (
                            "Switch needs a hover highlight." if run_num == 0 else "Hover highlight makes switch obvious."
                        ),
                    }
                ],
                "motor_action": {
                    "deliberation": "Left click switch",
                    "mouse_x": 740.0,
                    "mouse_y": 380.0,
                    "mouse_button_action": "left_click",
                    "drag_from_x": None,
                    "drag_from_y": None,
                    "hold_keys": [],
                    "tap_keys": [],
                    "duration_ms": 450,
                    "inspect_high_res_next": False,
                },
                "mood_shift": None,
                "submit_wish": (
                    {
                        "deliberation": "Switch needs clearer visual hover cue.",
                        "category": "controls_ux",
                        "summary": "Add hover highlight to courtyard switch",
                        "rationale": "Hard to tell the switch apart from decorative stonework on first boot.",
                    }
                    if run_num == 0
                    else {
                        "deliberation": "Future expansion idea for next slice.",
                        "category": "world_exploration",
                        "summary": "Add harmonic telescope alignment puzzle on upper deck",
                        "rationale": "Would be exciting to align lenses once the lift reaches the upper observatory.",
                    }
                ),
                "end_playtest": {
                    "deliberation": "Finished evaluating opening courtyard.",
                    "outcome": "completed",
                    "closing_thoughts": "Completed opening courtyard evaluation.",
                },
            }
            return SimpleNamespace(text=json.dumps(turn_payload), usage_metadata=usage)

        return SimpleNamespace(models=SimpleNamespace(generate_content=_gen))

    bus = SwarmEventBus()
    emitted_events: List[tuple[str, Dict[str, Any]]] = []
    pt_mgr = PlaytestSessionManager(
        db_path=db_path,
        cartridges_root=cartridges_root,
        sessions_root=tmp_path / "sessions",
        event_bus=bus,
        client_factory=_playtester_llm_factory,
        bridge_factory=_FakeBridge,
    )

    board = ProductionBoard(path=cand_dir / "production_board.json")
    seed_added = board.add_items(
        [
            WorkItemProposal(
                deliberation="Initial core courtyard mechanic.",
                item_id="core_courtyard",
                kind="ui",
                title="Initial Courtyard HUD",
                spec="Mount courtyard HUD.",
                proof_spec="Verify courtyard HUD.",
                target_modules=["ui_hud.gd"],
            )
        ],
        raised_by_role="CreativeDirector",
        slice_id="slice_0",
    )
    assert len(seed_added) == 1
    claimed_seed = board.claim_ready(limit=1)
    assert len(claimed_seed) == 1
    seed_proof = tmp_path / "seed_proof.png"
    Image.new("RGB", (320, 180), (40, 80, 120)).save(seed_proof, format="PNG")
    board.submit_result(
        "core_courtyard",
        WorkerResult(
            proof_images=[str(seed_proof)],
            notes="Built initial courtyard HUD.",
        ),
    )
    board.apply_review(
        ReviewDecision(
            deliberation="Seed item approved.",
            item_id="core_courtyard",
            feedback="",
            verdict="approve",
        )
    )
    assert board.is_drained()

    executed_fix_ids: List[str] = []

    def _ui_worker(item: Any) -> WorkerResult:
        executed_fix_ids.append(item.item_id)
        (cand_dir / "ui_hud.gd").write_text("# Updated with switch hover highlight\n", encoding="utf-8")
        fix_proof = tmp_path / f"proof_{item.item_id}.png"
        Image.new("RGB", (320, 180), (60, 140, 90)).save(fix_proof, format="PNG")
        return WorkerResult(
            proof_images=[str(fix_proof)],
            notes="Added crisp hover outline and audio chime hook in ui_hud.gd.",
        )

    scheduler = BoardScheduler(board=board, handlers={"ui": _ui_worker}, poll_interval=0.02, max_workers=1)

    verdict_calls = {"count": 0}
    roundtable_multimodal_png_counts: List[int] = []

    def _council_generate_content(*, model: str, contents: Any, config: Any) -> Any:
        schema_cls = getattr(config, "response_schema", None)
        schema_name = getattr(schema_cls, "__name__", "")
        usage = SimpleNamespace(
            prompt_token_count=1400,
            cached_content_token_count=0,
            candidates_token_count=200,
            thoughts_token_count=60,
        )

        if schema_name == "BlindPlaytestCouncilTurn":
            if isinstance(contents, list):
                png_cnt = sum(
                    1
                    for p in contents[:-1]
                    if getattr(p, "inline_data", None) is not None and p.inline_data.mime_type == "image/png"
                )
                roundtable_multimodal_png_counts.append(png_cnt)
            is_second_pass = verdict_calls["count"] >= 1
            turn_obj = BlindPlaytestCouncilTurn(
                deliberation="Evaluated Blind Playtester report, sparse impressions, and screenshots.",
                domain_reaction_to_blind_player=(
                    "Switch affordance was unclear on Pass 0."
                    if not is_second_pass
                    else "Switch hover cue resolved first-time onboarding friction."
                ),
                spoken_dialogue=(
                    "Let's add a clear hover outline on the courtyard switch before closing slice_0."
                    if not is_second_pass
                    else "The blind player immediately spotted the switch and rated fun 9/10—ready to approve!"
                ),
                addressed_to=["CreativeDirector"],
                priority_fixes_or_refinements=(
                    ["Add hover outline on courtyard switch"] if not is_second_pass else []
                ),
                recommends_slice_approval=is_second_pass,
            )
            return SimpleNamespace(text=turn_obj.model_dump_json(), usage_metadata=usage)

        if schema_name == "BlindPlaytestCouncilVerdict":
            pass_num = verdict_calls["count"]
            verdict_calls["count"] += 1
            if pass_num == 0:
                verdict_obj = BlindPlaytestCouncilVerdict(
                    deliberation="Pass 0 revealed onboarding friction on the courtyard switch; iterating current slice.",
                    player_experience_synthesis="Atmosphere worked well, but the blind player had to guess which stone panel was clickable.",
                    spoken_dialogue="We are iterating on slice_0 to add a crisp switch hover affordance before approving.",
                    next_slice_carryover_notes=[],
                    backlog_additions=[
                        WorkItemProposal(
                            deliberation="Add switch hover highlight requested by blind playtester.",
                            item_id="ux_switch_hover_fix",
                            kind="ui",
                            title="Courtyard Switch Hover Affordance",
                            spec="Highlight interactive courtyard switches on cursor hover in ui_hud.gd.",
                            proof_spec="Verify hover highlight state in headless test.",
                            target_modules=["ui_hud.gd"],
                        )
                    ],
                    verdict="iterate_current_slice",
                )
            else:
                verdict_obj = BlindPlaytestCouncilVerdict(
                    deliberation="Pass 1 achieved 9/10 fun rating with immediate onboarding clarity; approving slice_0.",
                    player_experience_synthesis="Switch hover highlight worked immediately; telescope puzzle wish belongs in Slice 1.",
                    spoken_dialogue="Slice 0 plays intuitively and cohesively! Approving slice_0 and carrying the telescope idea into Slice 1.",
                    next_slice_carryover_notes=[
                        "Consider harmonic telescope alignment puzzle on the upper observatory deck in Slice 1."
                    ],
                    backlog_additions=[],
                    verdict="approve_slice",
                )
            return SimpleNamespace(text=verdict_obj.model_dump_json(), usage_metadata=usage)

        if schema_name == "ReviewBatch":
            awaiting = board.awaiting_review()
            batch_obj = ReviewBatch(
                deliberation="Inspected proof for spawned playtest iteration task.",
                spoken_dialogue="Hover highlight on courtyard switch looks clean and legible.",
                addressed_to=["CreativeDirector"],
                review_decisions=[
                    ReviewDecision(
                        deliberation="Approved switch hover fix.",
                        item_id=it.item_id,
                        feedback="",
                        verdict="approve",
                    )
                    for it in awaiting
                ],
                cap_decisions=[],
            )
            return SimpleNamespace(text=batch_obj.model_dump_json(), usage_metadata=usage)

        raise AssertionError(f"Unexpected schema requested in test: {schema_name}")

    fake_council_client = SimpleNamespace(models=SimpleNamespace(generate_content=_council_generate_content))
    wr = CreativeWritersRoom(
        client=fake_council_client,
        board=board,
        scheduler=scheduler,
        playtest_manager=pt_mgr,
        telemetry_db=TelemetryDB(db_path),
        event_callback=lambda et, payload: emitted_events.append((et, payload)),
        drain_poll_s=0.02,
        drain_timeout_s=15.0,
    )
    wr.max_assembly_passes = 0
    wr.enable_end_of_slice_blind_playtest = True
    wr.slice_playtest_max_turns = 3

    try:
        wr._drain_backlog(bible, "Exploration & puzzle player profile.", cand_dir)
    finally:
        scheduler.stop()

    # 1. Verify 2 Council playtest passes ran: Pass #0 iterated (`ux_switch_hover_fix`), Pass #1 approved `slice_0`
    assert playtest_run_counter["count"] == 2
    assert verdict_calls["count"] == 2
    assert executed_fix_ids == ["ux_switch_hover_fix"]
    assert bible.slice_roadmap[0].status == "completed"

    # 2. Verify all Council-initiated sessions & wishes were persisted and auto-marked `sent_to_council`
    db = TelemetryDB(db_path)
    all_sessions = db.list_playtest_sessions()
    assert len(all_sessions) == 2
    assert all(s["initiated_by"] == "council" for s in all_sessions)
    assert {s["session_report"]["overall_fun_rating"] for s in all_sessions} == {6, 9}

    for s in all_sessions:
        s_wishes = db.list_playtest_wishes(session_id=s["session_id"])
        assert len(s_wishes) == 1
        assert s_wishes[0]["status"] == "sent_to_council"

    # 3. Verify Council Roundtable specialists received multimodal PNG screenshots from the Blind Playtester
    assert len(roundtable_multimodal_png_counts) == 8  # 4 specialists × 2 passes
    assert all(cnt >= 1 for cnt in roundtable_multimodal_png_counts)

    # 4. Verify Writers' Room transcript includes BlindPlaytester report, 4 specialist turns, and CreativeDirector verdict
    roles_in_tx = [u.speaker_role for u in bible.writers_room_transcript]
    assert "BlindPlaytester" in roles_in_tx
    assert "UXExperienceDesigner" in roles_in_tx
    assert "AudioAtmosphereDesigner" in roles_in_tx
    assert "CreativeDirector" in roles_in_tx
    assert any(et == "council_blind_playtest_deliberation_completed" for et, _ in emitted_events)

    # 5. Verify `artifacts/blind_playtest_report_slice_0.json` was written and feeds into Next-Slice Reflection Dossier
    report_artifact_path = cand_dir / "artifacts" / "blind_playtest_report_slice_0.json"
    assert report_artifact_path.is_file()
    saved_art = json.loads(report_artifact_path.read_text(encoding="utf-8"))
    assert saved_art["council_verdict"] == "approve_slice"
    assert saved_art["session_report"]["overall_fun_rating"] == 9

    dossier_text, _, _ = wr._capture_current_slice_reflection_dossier(
        bible,
        bible.slice_roadmap[0],
        cand_dir,
    )
    assert "BLIND PLAYTESTER EXPERIENCE REPORT & CARRYOVER FROM `slice_0`" in dossier_text
    assert "Overall Fun Rating: 9/10" in dossier_text
    assert "harmonic telescope alignment puzzle" in dossier_text

