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
        # 1. Verify React PlaytestLabView component exists and API root responds cleanly
        idx_res = client.get("/")
        assert idx_res.status_code == 200
        pt_view_src = (
            Path(__file__).resolve().parent.parent
            / "studio-web"
            / "src"
            / "features"
            / "playtest"
            / "PlaytestLabView.tsx"
        ).read_text(encoding="utf-8")
        assert "<video" in pt_view_src
        assert "Experience Report" in pt_view_src

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
                        "bugs_and_glitches_observed": ["Switch click produced zero audio feedback"],
                        "unanswered_questions_and_mysteries": ["Why is the upper deck hatch sealed?"],
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
                        "bugs_and_glitches_observed": [],
                        "unanswered_questions_and_mysteries": [],
                        "overall_fun_rating": 9,
                    }
                return SimpleNamespace(text=json.dumps(rep), usage_metadata=usage)

            turn_payload = {
                "deliberation": f"Pass {run_num} blind playtest turn.",
                "current_plan": "Click courtyard switch and complete slice check.",
                "turn_log_entry": "Tested courtyard switch interaction.",
                "field_note_updates": [
                    {
                        "deliberation": "Observing stone switch near lift.",
                        "topic": "Courtyard Switch",
                        "observation": "Recessed stone panel next to the central lift.",
                    }
                ],
                "question_updates": [
                    {
                        "deliberation": "Asking what the stone panel controls.",
                        "question_key": "stone_panel_effect",
                        "question": "What does clicking the recessed stone panel do?",
                        "answer": "Starts the central lift ascent." if run_num == 1 else None,
                        "status": "answered" if run_num == 1 else "open",
                    }
                ],
                "rulebook_updates": [
                    {"deliberation": "Left click toggles courtyard switch", "key_or_topic": "Left Click", "behavior": "Activates switch"}
                ],
                "bug_reports": (
                    [
                        {
                            "deliberation": "Switch click lacked audio cue.",
                            "bug_key": "silent_switch_click",
                            "category": "audio_glitch",
                            "summary": "Switch click plays no sound effect",
                            "details": "Clicking the switch triggered lift motion silently.",
                            "severity": "minor",
                        }
                    ]
                    if run_num == 0
                    else []
                ),
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
    roundtable_prompts: List[str] = []

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
                if isinstance(contents[-1], str):
                    roundtable_prompts.append(contents[-1])
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

    # 3. Verify Council Roundtable specialists received multimodal PNG screenshots + full report, wishes, bugs, notes, and questions
    assert len(roundtable_multimodal_png_counts) == 8  # 4 specialists × 2 passes
    assert all(cnt >= 1 for cnt in roundtable_multimodal_png_counts)
    assert any("Switch click plays no sound effect" in p for p in roundtable_prompts)
    assert any("Courtyard Switch: Recessed stone panel" in p for p in roundtable_prompts)
    assert any("What does clicking the recessed stone panel do?" in p for p in roundtable_prompts)
    assert any("Add hover highlight to courtyard switch" in p for p in roundtable_prompts)

    # 4. Verify Writers' Room transcript includes BlindPlaytester report, 4 specialist turns, and CoProducer (Council Chair) verdict
    roles_in_tx = [u.speaker_role for u in bible.writers_room_transcript]
    assert "BlindPlaytester" in roles_in_tx
    assert "UXExperienceDesigner" in roles_in_tx
    assert "AudioAtmosphereDesigner" in roles_in_tx
    assert "CoProducer" in roles_in_tx
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


def test_async_council_initiated_session_auto_delivers_full_report_and_wishes_to_council(tmp_path: Path):
    """Verifies that a playtest session created with `initiated_by='council'` via `/api/playtest/sessions/start`
    automatically delivers the FULL final PlaytestExperienceReport + all wishes + bugs + questions + field notes
    to the Council's steering + multimodal report queue upon completion.
    """
    import time

    db_path, cartridges_root, active_dir, cand_dir = _setup_workspace(tmp_path)

    def _council_session_llm_factory() -> Any:
        def _gen(*, model: str, contents: Any, config: Any) -> Any:
            schema = getattr(config, "response_schema", None)
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            usage = SimpleNamespace(
                prompt_token_count=800,
                cached_content_token_count=400,
                candidates_token_count=150,
                thoughts_token_count=0,
            )
            if "overall_experience_summary" in props:
                rep = {
                    "deliberation": "Synthesizing full report for Council-initiated session.",
                    "overall_experience_summary": "Could inspect the HUD, but clicking the main valve did not advance the pressure gauge.",
                    "graphics_and_visuals": "Clean industrial sprites.",
                    "gameplay_and_feel": "Valve click lacked state response.",
                    "sound_and_audio": "Opening steam hiss stopped after Turn 1.",
                    "intuitiveness_and_onboarding": "Valve icon looked clickable, yet gauge stayed at 0.",
                    "goals_pacing_and_accomplishment": "Blocked at initial pressure check.",
                    "world_and_atmosphere": "Cohesive brass boiler-room aesthetic.",
                    "key_highlights": ["Readable gauge dial"],
                    "key_pain_points": ["Main valve click did not increase pressure"],
                    "bugs_and_glitches_observed": ["Clicking main valve never increments pressure gauge"],
                    "unanswered_questions_and_mysteries": ["Does the secondary lever need to be unlocked first?"],
                    "overall_fun_rating": 3,
                }
                return SimpleNamespace(text=json.dumps(rep), usage_metadata=usage)

            turn_payload = {
                "deliberation": "Testing main valve click and noting gauge state.",
                "current_plan": "Click main valve and watch pressure gauge.",
                "turn_log_entry": "Clicked main valve; gauge stayed at 0.",
                "field_note_updates": [
                    {"deliberation": "Noting gauge.", "topic": "Pressure Gauge", "observation": "Reads 0/10 PSI at top-left."}
                ],
                "question_updates": [
                    {
                        "deliberation": "Wondering about secondary lever.",
                        "question_key": "secondary_lever_lock",
                        "question": "Does the secondary lever need to be unlocked first?",
                        "answer": None,
                        "status": "open",
                    }
                ],
                "rulebook_updates": [],
                "bug_reports": [
                    {
                        "deliberation": "Valve click had zero effect.",
                        "bug_key": "valve_no_response",
                        "category": "progression_blocker",
                        "summary": "Main valve click never increments pressure gauge",
                        "details": "Left-clicking directly on the central valve left PSI at 0.",
                        "severity": "blocker",
                    }
                ],
                "impression_updates": [],
                "current_mood": "frustrated",
                "motor_action": {
                    "deliberation": "Click valve.",
                    "mouse_x": 500.0,
                    "mouse_y": 500.0,
                    "mouse_button_action": "left_click",
                    "drag_from_x": None,
                    "drag_from_y": None,
                    "hold_keys": [],
                    "tap_keys": [],
                    "duration_ms": 500,
                    "inspect_high_res_next": False,
                },
                "mood_shift": None,
                "submit_wish": {
                    "deliberation": "Need valve feedback.",
                    "category": "progression_blocker",
                    "summary": "Wire main valve click to pressure gauge or show locked tooltip",
                    "rationale": "Clicking the primary focal object gives zero feedback.",
                },
                "end_playtest": {
                    "deliberation": "Blocked on valve.",
                    "outcome": "stuck_cannot_progress",
                    "closing_thoughts": "Cannot advance past 0 PSI.",
                },
            }
            return SimpleNamespace(text=json.dumps(turn_payload), usage_metadata=usage)

        return SimpleNamespace(models=SimpleNamespace(generate_content=_gen))

    bus = SwarmEventBus()
    pt_mgr = PlaytestSessionManager(
        db_path=db_path,
        cartridges_root=cartridges_root,
        sessions_root=tmp_path / "sessions_council_async",
        event_bus=bus,
        client_factory=_council_session_llm_factory,
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
        start_res = client.post(
            "/api/playtest/sessions/start",
            json={
                "checkpoint_id": "candidate_001_slice_0",
                "candidate_id": "candidate_001",
                "slice_id": "slice_0",
                "initiated_by": "council",
                "max_turns": 5,
            },
        )
        assert start_res.status_code == 200
        sid = start_res.json()["session"]["session_id"]

        deadline = time.time() + 10.0
        det = {}
        while time.time() < deadline:
            r = client.get(f"/api/playtest/sessions/{sid}")
            det = r.json()
            if det["session"]["status"] in ("completed", "stopped", "error"):
                break
            time.sleep(0.05)

        assert det["session"]["status"] == "completed"
        assert det["session"]["current_mood"] == "frustrated"
        assert "valve_no_response" in det["session"]["bugs_found"]
        assert "Pressure Gauge" in det["session"]["field_notes"]
        assert "secondary_lever_lock" in det["session"]["questions"]
        assert det["wishes"][0]["status"] == "sent_to_council"

    reports_file = cand_dir / ".pending_council_playtest_reports.json"
    steering_file = cand_dir / ".pending_steering.json"
    assert reports_file.is_file()
    assert steering_file.is_file()

    # Verify CreativeWritersRoom consumes the full report + wishes + bugs + notes + questions + screenshots!
    captured_council_calls: List[Any] = []

    def _council_pivot_gen(*, model: str, contents: Any, config: Any) -> Any:
        captured_council_calls.append(contents)
        proposal = WritersRoomTurnProposal(
            deliberation="Reviewed full Council blind playtest report, bug report, and wish.",
            spoken_dialogue="Fixing the main valve click handler and steam loop right away.",
            addressed_to=["CreativeDirector"],
            backlog_additions=[
                {
                    "deliberation": "Wire main valve click to pressure gauge and loop steam audio.",
                    "item_id": "fix_valve_and_steam_loop",
                    "title": "Main Valve Click Response & Steam Loop",
                    "kind": "integration",
                    "spec": "Connect main valve click signal to increment pressure gauge and keep steam audio looping.",
                    "target_modules": ["main.gd"],
                    "depends_on": [],
                    "blocks_items": [],
                    "proof_spec": "Verify valve click increments PSI in res://tests/test_fix_valve_and_steam_loop.tscn.",
                }
            ],
            ready_to_lock_bible=False,
        )
        return SimpleNamespace(
            text=proposal.model_dump_json(),
            usage_metadata=SimpleNamespace(
                prompt_token_count=1200,
                cached_content_token_count=0,
                candidates_token_count=150,
                thoughts_token_count=40,
            ),
        )

    board = ProductionBoard(path=cand_dir / "production_board.json")
    wr = CreativeWritersRoom(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=_council_pivot_gen)),
        board=board,
    )
    loaded_bible = GameBibleSpec.model_validate_json((cand_dir / "game_bible.json").read_text(encoding="utf-8"))
    steered = wr._consume_pending_steering(loaded_bible, candidate_dir=cand_dir)
    assert len(steered) == 1
    assert "Fun Rating: 3/10" in steered[0]
    wr._execute_steering_pivot_turns(loaded_bible, "Puzzle player profile.", cand_dir, steered)

    assert not reports_file.exists()
    assert len(captured_council_calls) >= 1
    council_prompt_str = captured_council_calls[0][-1]
    assert "COUNCIL-INITIATED BLIND PLAYTEST FULL REPORT & WISHES" in council_prompt_str
    assert "Overall Fun Rating: 3/10" in council_prompt_str
    assert "Main valve click never increments pressure gauge" in council_prompt_str
    assert "Pressure Gauge: Reads 0/10 PSI" in council_prompt_str
    assert "Does the secondary lever need to be unlocked first?" in council_prompt_str
    assert "Wire main valve click to pressure gauge or show locked tooltip" in council_prompt_str


def test_council_blind_playtest_gate_includes_mid_session_screenshots_action_tape_and_skeptical_audit(tmp_path: Path):
    """Verifies that even when a fast playtester files 0 bugs, 0 wishes, and an optimistic 9/10 rating,
    `_run_end_of_slice_blind_playtest` equips the Council with:
    1) Up to 5 evenly spaced screenshots across the session (not just Turn 1 & Final),
    2) The full `BLIND PLAYER CHRONOLOGICAL ACTION TAPE` showing repeated same-spot clicks, and
    3) Explicit skeptical audit instructions against no-op loops and auto-playing shortcuts.
    """
    from engine.studio.models import (
        BlindPlaytestCouncilTurn,
        BlindPlaytestCouncilVerdict,
        WorkItemProposal,
    )

    db_path, cartridges_root, _active_dir, cand_dir = _setup_workspace(tmp_path)
    bible = GameBibleSpec.model_validate_json((cand_dir / "game_bible.json").read_text(encoding="utf-8"))
    bible.slice_roadmap[0].status = "in_progress"

    turn_counter = {"n": 0}

    def _optimistic_playtester_factory() -> Any:
        def _gen(*, model: str, contents: Any, config: Any) -> Any:
            schema = getattr(config, "response_schema", None)
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            usage = SimpleNamespace(
                prompt_token_count=900,
                cached_content_token_count=500,
                candidates_token_count=120,
                thoughts_token_count=0,
            )
            if "overall_experience_summary" in props:
                rep = {
                    "deliberation": "Optimistic summary despite repetitive clicking.",
                    "overall_experience_summary": "Clicked the decorative crest repeatedly and enjoyed the scenery.",
                    "graphics_and_visuals": "Bright clean shapes.",
                    "gameplay_and_feel": "Smooth cursor feel.",
                    "sound_and_audio": "Silent session.",
                    "intuitiveness_and_onboarding": "Easy to spot crest.",
                    "goals_pacing_and_accomplishment": "Clicked crest several times.",
                    "world_and_atmosphere": "Calm courtyard.",
                    "key_highlights": ["Clean colors"],
                    "key_pain_points": [],
                    "bugs_and_glitches_observed": [],
                    "unanswered_questions_and_mysteries": [],
                    "overall_fun_rating": 9,
                }
                return SimpleNamespace(text=json.dumps(rep), usage_metadata=usage)

            turn_counter["n"] += 1
            idx = turn_counter["n"]
            turn_payload = {
                "deliberation": f"Turn {idx} clicking static corner emblem.",
                "current_plan": "Keep clicking corner emblem.",
                "turn_log_entry": f"Clicked static emblem at (910, 210) on step {idx}.",
                "field_note_updates": [],
                "question_updates": [],
                "rulebook_updates": [],
                "bug_reports": [],
                "impression_updates": [],
                "current_mood": "curious",
                "motor_action": {
                    "deliberation": "Click corner emblem.",
                    "mouse_x": 910.0,
                    "mouse_y": 210.0,
                    "mouse_button_action": "left_click",
                    "drag_from_x": None,
                    "drag_from_y": None,
                    "hold_keys": [],
                    "tap_keys": [],
                    "duration_ms": 450,
                    "inspect_high_res_next": False,
                },
                "mood_shift": None,
                "submit_wish": None,
                "end_playtest": (
                    {
                        "deliberation": "Done clicking.",
                        "outcome": "completed",
                        "closing_thoughts": "Finished clicking.",
                    }
                    if idx >= 6
                    else None
                ),
            }
            return SimpleNamespace(text=json.dumps(turn_payload), usage_metadata=usage)

        return SimpleNamespace(models=SimpleNamespace(generate_content=_gen))

    pt_mgr = PlaytestSessionManager(
        db_path=db_path,
        cartridges_root=cartridges_root,
        sessions_root=tmp_path / "sessions_skeptical_audit",
        event_bus=SwarmEventBus(),
        client_factory=_optimistic_playtester_factory,
        bridge_factory=_FakeBridge,
    )
    board = ProductionBoard(path=cand_dir / "production_board.json")

    captured_png_counts: List[int] = []
    captured_labels: List[str] = []
    captured_prompts: List[str] = []

    def _skeptical_council_gen(*, model: str, contents: Any, config: Any) -> Any:
        schema_cls = getattr(config, "response_schema", None)
        schema_name = getattr(schema_cls, "__name__", "")
        usage = SimpleNamespace(prompt_token_count=1200, cached_content_token_count=0, candidates_token_count=150, thoughts_token_count=30)
        if isinstance(contents, list):
            png_cnt = sum(
                1
                for p in contents[:-1]
                if getattr(p, "inline_data", None) is not None and p.inline_data.mime_type == "image/png"
            )
            captured_png_counts.append(png_cnt)
            for p in contents[:-1]:
                if isinstance(p, str) and p.startswith("BLIND PLAYTESTER SCREENSHOT"):
                    captured_labels.append(p)
            if isinstance(contents[-1], str):
                captured_prompts.append(contents[-1])

        if schema_name == "BlindPlaytestCouncilTurn":
            t_obj = BlindPlaytestCouncilTurn(
                deliberation="Action tape shows 6 identical clicks at (910, 210) despite 9/10 rating; voting iterate.",
                domain_reaction_to_blind_player="Player clicked static corner emblem 6 times in a row without real progression.",
                spoken_dialogue="Ignore the 9/10—look at the action tape: 6 identical clicks at (910, 210) with zero progression.",
                addressed_to=["CreativeDirector"],
                priority_fixes_or_refinements=["Replace ambiguous decorative badge or wire real interactive feedback"],
                recommends_slice_approval=False,
            )
            return SimpleNamespace(text=t_obj.model_dump_json(), usage_metadata=usage)

        if schema_name == "BlindPlaytestCouncilVerdict":
            v_obj = BlindPlaytestCouncilVerdict(
                deliberation="Rejecting inflated 9/10 rating based on action tape and mid-session screenshots.",
                player_experience_synthesis="Playtester looped 6 times on a non-interactive emblem with silent audio.",
                spoken_dialogue="Iterating on slice_0 to resolve the non-interactive UI loop.",
                next_slice_carryover_notes=[],
                backlog_additions=[
                    WorkItemProposal(
                        deliberation="Resolve non-interactive UI confusion.",
                        item_id="fix_ui_affordance_clarity",
                        kind="ui",
                        title="Clarify Interactive vs Decorative HUD Affordances",
                        spec="Ensure HUD status badges do not mimic clickable buttons in ui_hud.gd.",
                        proof_spec="Verify HUD affordances in headless test.",
                        target_modules=["ui_hud.gd"],
                    )
                ],
                verdict="iterate_current_slice",
            )
            return SimpleNamespace(text=v_obj.model_dump_json(), usage_metadata=usage)

        raise AssertionError(f"Unexpected schema: {schema_name}")

    wr = CreativeWritersRoom(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=_skeptical_council_gen)),
        board=board,
        playtest_manager=pt_mgr,
        telemetry_db=TelemetryDB(db_path),
    )
    wr.enable_end_of_slice_blind_playtest = True
    wr.slice_playtest_max_turns = 10

    added_count = wr._run_end_of_slice_blind_playtest(
        bible,
        "Puzzle player profile.",
        cand_dir,
        pass_index=0,
    )
    assert added_count == 1
    assert [it.item_id for it in board.items()] == ["fix_ui_affordance_clarity"]

    # Council received all 5 screenshot slots (Opening + 3 Mid-Session + Final), not just 2!
    assert all(c == 5 for c in captured_png_counts)
    assert any("Mid-Session Progress" in lbl for lbl in captured_labels)
    # Council received the full 6-turn Chronological Action Tape and skeptical audit mandate!
    assert any("=== BLIND PLAYER CHRONOLOGICAL ACTION TAPE (6 turns) ===" in p for p in captured_prompts)
    assert any("move_to=(910,210) mouse=left_click" in p for p in captured_prompts)
    assert any("Never rubber-stamp an inflated `overall_fun_rating`" in p for p in captured_prompts)


def test_playtest_session_history_archive_multi_run_and_council_verdict_preservation(tmp_path: Path):
    """Verifies that every playtester session for a game/slice is permanently archived and watchable:
    - Multiple Council Gate passes on the same slice (`pass_index=0` iterate, `pass_index=1` approve)
      preserve distinct per-pass JSON reports and distinct `council_review` verdicts per session.
    - Manual operator runs on the same slice increment `game_run_number` and `slice_run_number` cleanly.
    - `GET /api/playtest/sessions` supports game & slice filtering and exposes `games_catalog` + run numbers.
    - `GET /api/playtest/sessions/{id}` denormalizes `[0..1000]` coordinates into `1280x720` `action_px`
      and keeps every run's turn screenshots watchable.
    - Disk `session_archive.json` snapshots restore historical sessions if SQLite is rebuilt.
    """
    import time
    from engine.studio.models import (
        BlindPlaytestCouncilTurn,
        BlindPlaytestCouncilVerdict,
        WorkItemProposal,
    )

    db_path, cartridges_root, active_dir, cand_dir = _setup_workspace(tmp_path)
    sessions_root = tmp_path / "sessions_archive_test"
    bible = GameBibleSpec.model_validate_json((cand_dir / "game_bible.json").read_text(encoding="utf-8"))
    bible.slice_roadmap[0].status = "in_progress"
    (cand_dir / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    run_seq = {"idx": 0}

    def _multi_run_playtester_factory() -> Any:
        run_seq["idx"] += 1
        current_run = run_seq["idx"]
        local_turn = {"n": 0}
        fun_by_run = {1: 5, 2: 9, 3: 8}

        def _gen(*, model: str, contents: Any, config: Any) -> Any:
            schema = getattr(config, "response_schema", None)
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            usage = SimpleNamespace(
                prompt_token_count=800,
                cached_content_token_count=400,
                candidates_token_count=110,
                thoughts_token_count=0,
            )
            if "overall_experience_summary" in props:
                rep = {
                    "deliberation": f"Synthesizing report for run #{current_run}.",
                    "overall_experience_summary": f"Completed playtest run #{current_run}.",
                    "graphics_and_visuals": "Clean silhouettes.",
                    "gameplay_and_feel": "Responsive controls.",
                    "sound_and_audio": "Subtle cues.",
                    "intuitiveness_and_onboarding": "Readable layout.",
                    "goals_pacing_and_accomplishment": "Clear objective.",
                    "world_and_atmosphere": "Cohesive mood.",
                    "key_highlights": [f"Highlight from run #{current_run}"],
                    "key_pain_points": ["Minor friction"] if current_run == 1 else [],
                    "bugs_and_glitches_observed": [],
                    "unanswered_questions_and_mysteries": [],
                    "overall_fun_rating": fun_by_run.get(current_run, 8),
                }
                return SimpleNamespace(text=json.dumps(rep), usage_metadata=usage)

            local_turn["n"] += 1
            t_n = local_turn["n"]
            norm_x = 500.0 if current_run == 1 else (750.0 if current_run == 2 else 250.0)
            norm_y = 500.0 if current_run == 1 else 250.0
            turn_payload = {
                "deliberation": f"Run {current_run} turn {t_n} testing interaction.",
                "current_plan": f"Execute run {current_run} plan.",
                "turn_log_entry": f"Run {current_run} clicked ({norm_x}, {norm_y}).",
                "field_note_updates": [],
                "question_updates": [],
                "rulebook_updates": [],
                "bug_reports": [],
                "impression_updates": [],
                "current_mood": "curious" if current_run == 1 else "engaged",
                "motor_action": {
                    "deliberation": "Click target.",
                    "mouse_x": norm_x,
                    "mouse_y": norm_y,
                    "mouse_button_action": "left_click",
                    "drag_from_x": None,
                    "drag_from_y": None,
                    "hold_keys": [],
                    "tap_keys": ["SPACE"] if t_n == 1 else [],
                    "duration_ms": 400,
                    "inspect_high_res_next": False,
                },
                "mood_shift": None,
                "submit_wish": None,
                "end_playtest": (
                    {
                        "deliberation": "Finished short run.",
                        "outcome": "completed",
                        "closing_thoughts": f"Done run {current_run}.",
                    }
                    if t_n >= 2
                    else None
                ),
            }
            return SimpleNamespace(text=json.dumps(turn_payload), usage_metadata=usage)

        return SimpleNamespace(models=SimpleNamespace(generate_content=_gen))

    bus = SwarmEventBus()
    pt_mgr = PlaytestSessionManager(
        db_path=db_path,
        cartridges_root=cartridges_root,
        sessions_root=sessions_root,
        event_bus=bus,
        client_factory=_multi_run_playtester_factory,
        bridge_factory=_FakeBridge,
    )
    board = ProductionBoard(path=cand_dir / "production_board.json")

    council_pass_state = {"pass_idx": 0}

    def _council_gen(*, model: str, contents: Any, config: Any) -> Any:
        schema_cls = getattr(config, "response_schema", None)
        schema_name = getattr(schema_cls, "__name__", "")
        usage = SimpleNamespace(prompt_token_count=900, cached_content_token_count=0, candidates_token_count=120, thoughts_token_count=20)
        is_first_pass = council_pass_state["pass_idx"] == 0

        if schema_name == "BlindPlaytestCouncilTurn":
            t_obj = BlindPlaytestCouncilTurn(
                deliberation="Reviewing slice run.",
                domain_reaction_to_blind_player="Observed player interaction.",
                spoken_dialogue="Let us iterate once." if is_first_pass else "Slice looks ready to ship.",
                addressed_to=["CreativeDirector"],
                priority_fixes_or_refinements=["Tighten feedback"] if is_first_pass else [],
                recommends_slice_approval=not is_first_pass,
            )
            return SimpleNamespace(text=t_obj.model_dump_json(), usage_metadata=usage)

        if schema_name == "BlindPlaytestCouncilVerdict":
            v_obj = BlindPlaytestCouncilVerdict(
                deliberation="Pass 0 needs polish." if is_first_pass else "Pass 1 verified all fixes.",
                player_experience_synthesis=(
                    "Pass 0 revealed minor onboarding friction."
                    if is_first_pass
                    else "Pass 1 played smoothly from start to finish."
                ),
                spoken_dialogue="Iterating slice_0." if is_first_pass else "Approving slice_0!",
                next_slice_carryover_notes=[] if is_first_pass else ["Carry momentum into slice_1"],
                backlog_additions=(
                    [
                        WorkItemProposal(
                            deliberation="Polish interaction feedback.",
                            item_id="polish_feedback_pass0",
                            kind="mechanic_code",
                            title="Polish Interaction Feedback",
                            spec="Improve feedback clarity in player.gd.",
                            proof_spec="Headless test passes.",
                            target_modules=["player.gd"],
                        )
                    ]
                    if is_first_pass
                    else []
                ),
                verdict="iterate_current_slice" if is_first_pass else "approve_slice",
            )
            return SimpleNamespace(text=v_obj.model_dump_json(), usage_metadata=usage)

        raise AssertionError(f"Unexpected schema: {schema_name}")

    wr = CreativeWritersRoom(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=_council_gen)),
        board=board,
        playtest_manager=pt_mgr,
        telemetry_db=TelemetryDB(db_path),
    )
    wr.enable_end_of_slice_blind_playtest = True
    wr.slice_playtest_max_turns = 5

    # Pass #0 -> iterate_current_slice (fun=5)
    council_pass_state["pass_idx"] = 0
    added_p0 = wr._run_end_of_slice_blind_playtest(bible, "Profile", cand_dir, pass_index=0)
    assert added_p0 == 1
    board.get("polish_feedback_pass0").state = "approved"
    board._persist()

    # Pass #1 -> approve_slice (fun=9)
    council_pass_state["pass_idx"] = 1
    added_p1 = wr._run_end_of_slice_blind_playtest(bible, "Profile", cand_dir, pass_index=1)
    assert added_p1 == 0
    bible.slice_roadmap[0].status = "completed"
    (cand_dir / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    # Both per-pass Council report files must exist without Pass #1 clobbering Pass #0!
    p0_report_path = cand_dir / "artifacts" / "blind_playtest_report_slice_0_pass0.json"
    p1_report_path = cand_dir / "artifacts" / "blind_playtest_report_slice_0_pass1.json"
    assert p0_report_path.is_file()
    assert p1_report_path.is_file()
    assert json.loads(p0_report_path.read_text(encoding="utf-8"))["council_verdict"] == "iterate_current_slice"
    assert json.loads(p1_report_path.read_text(encoding="utf-8"))["council_verdict"] == "approve_slice"

    app = create_app(
        db_path=db_path,
        active_cartridge_dir=active_dir,
        cartridges_root=cartridges_root,
        event_bus=bus,
        playtest_manager=pt_mgr,
    )

    with TestClient(app) as client:
        # Verify React PlaytestLabView exposes session archive, frame stage, and live screen modes
        pt_view_src = (
            Path(__file__).resolve().parent.parent
            / "studio-web"
            / "src"
            / "features"
            / "playtest"
            / "PlaytestLabView.tsx"
        ).read_text(encoding="utf-8")
        assert "type StageMode = 'video' | 'frames' | 'live'" in pt_view_src
        assert "Turn-by-Turn Replay" in pt_view_src

        # Trigger Run #3 as a manual operator run on the completed slice_0
        start_res = client.post(
            "/api/playtest/sessions/start",
            json={
                "deliberation": "Operator re-playing slice_0 to compare against Council runs.",
                "candidate_id": "candidate_001",
                "slice_id": "slice_0",
                "max_turns": 5,
            },
        )
        assert start_res.status_code == 200
        manual_sid = start_res.json()["session"]["session_id"]

        for _ in range(40):
            det = client.get(f"/api/playtest/sessions/{manual_sid}").json()
            if det["session"]["status"] == "completed":
                break
            time.sleep(0.05)
        assert det["session"]["status"] == "completed"

        # Query archive filtered by candidate_001 + slice_0
        list_res = client.get("/api/playtest/sessions?candidate_id=candidate_001&slice_id=slice_0")
        assert list_res.status_code == 200
        list_payload = list_res.json()
        sessions = list_payload["sessions"]
        assert len(sessions) == 3
        assert any(g["candidate_id"] == "candidate_001" for g in list_payload["games_catalog"])

        # Sessions are returned newest-first (Run #3, Run #2, Run #1) with chronological run numbers intact
        by_run_num = {s["game_run_number"]: s for s in sessions}
        assert set(by_run_num.keys()) == {1, 2, 3}
        assert [by_run_num[i]["slice_run_number"] for i in (1, 2, 3)] == [1, 2, 3]
        assert by_run_num[1]["initiated_by"] == "council"
        assert by_run_num[1]["overall_fun_rating"] == 5
        assert by_run_num[1]["council_review"]["verdict"] == "iterate_current_slice"
        assert by_run_num[1]["council_review"]["spawned_item_ids"] == ["polish_feedback_pass0"]
        assert by_run_num[1]["slice_index"] == 0

        assert by_run_num[2]["initiated_by"] == "council"
        assert by_run_num[2]["overall_fun_rating"] == 9
        assert by_run_num[2]["council_review"]["verdict"] == "approve_slice"

        assert by_run_num[3]["initiated_by"] == "operator"
        assert by_run_num[3]["overall_fun_rating"] == 8
        assert by_run_num[3]["council_review"] is None

        # Every single historical run can be opened and watched turn-by-turn with denormalized 1280x720 coordinates
        for run_idx, expected_px in ((1, (640.0, 360.0)), (2, (960.0, 180.0)), (3, (320.0, 180.0))):
            sid = by_run_num[run_idx]["session_id"]
            d_res = client.get(f"/api/playtest/sessions/{sid}")
            assert d_res.status_code == 200
            d_json = d_res.json()
            assert d_json["session"]["game_run_number"] == run_idx
            assert len(d_json["turns"]) == 2
            t0 = d_json["turns"][0]
            assert t0["action_px"]["target_x"] == expected_px[0]
            assert t0["action_px"]["target_y"] == expected_px[1]
            assert t0["action_px"]["mouse_button_action"] == "left_click"
            assert t0["screenshot_name"]
            scr_res = client.get(f"/api/playtest/sessions/{sid}/media/{t0['screenshot_name']}")
            assert scr_res.status_code == 200
            assert len(scr_res.content) > 0

            # Per-session immutable archive JSON exists in both sessions_root and cartridge artifacts
            assert (sessions_root / sid / "session_archive.json").is_file()
            assert (cand_dir / "artifacts" / "playtest_sessions" / f"{sid}.json").is_file()

    # Verify DB disaster recovery: fresh SQLite file restores all 3 archived sessions from disk snapshots!
    fresh_db_path = tmp_path / "fresh_restored_telemetry.db"
    restored_mgr = PlaytestSessionManager(
        db_path=fresh_db_path,
        cartridges_root=cartridges_root,
        sessions_root=sessions_root,
        event_bus=SwarmEventBus(),
    )
    restored_listing = restored_mgr.list_enriched_sessions(candidate_id="candidate_001", slice_id="slice_0")
    assert len(restored_listing["sessions"]) == 3
    restored_by_run = {s["game_run_number"]: s for s in restored_listing["sessions"]}
    assert restored_by_run[1]["council_review"]["verdict"] == "iterate_current_slice"
    assert restored_by_run[2]["council_review"]["verdict"] == "approve_slice"
    assert len(restored_mgr.get_session_detail(restored_by_run[1]["session_id"])["turns"]) == 2


def test_manual_playtest_send_full_report_to_council_and_multimodal_council_ingestion(tmp_path: Path):
    """Verifies that an operator can dispatch the full report of a finished manual Blind Playtest
    via `POST /api/playtest/sessions/{session_id}/send-report-to-council`, persisting dispatch status,
    copying player proof screenshots, updating `game_bible.json`, approving pending session wishes,
    and feeding the complete report + multimodal screenshots into `CreativeWritersRoom`."""
    import time

    db_path, cartridges_root, active_dir, cand_dir = _setup_workspace(tmp_path)
    # Mirror candidate_001 game_bible.json into active_dir so `checkpoint_id='active'` also resolves provenance cleanly
    (active_dir / "game_bible.json").write_text(
        (cand_dir / "game_bible.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    bus = SwarmEventBus()

    def _fake_llm_factory() -> Any:
        def _gen(*, model: str, contents: Any, config: Any) -> Any:
            schema = getattr(config, "response_schema", None)
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            usage = SimpleNamespace(
                prompt_token_count=900,
                cached_content_token_count=600,
                candidates_token_count=140,
                thoughts_token_count=30,
            )
            if "overall_experience_summary" in props:
                report_payload = {
                    "deliberation": "Synthesizing end-of-session player report for manual operator run.",
                    "overall_experience_summary": "Cohesive modular chamber, but interactive terminal lacked hover feedback and overlapped status text.",
                    "visual_and_ui_defects_observed": ["Bottom prompt label overlaps slot 1 border at 1280x720"],
                    "graphics_and_visuals": "Crisp geometric silhouettes with one overlapping HUD caption.",
                    "gameplay_and_feel": "Responsive switching once discovered.",
                    "sound_and_audio": "Clear confirmation chime.",
                    "intuitiveness_and_onboarding": "Unclear which terminal panel accepts initial input.",
                    "goals_pacing_and_accomplishment": "Activating the central conduit felt earned.",
                    "world_and_atmosphere": "Atmospheric chamber.",
                    "key_highlights": ["Conduit power-up visual sequence"],
                    "key_pain_points": ["Missing hover state on interactive terminal"],
                    "bugs_and_glitches_observed": ["HUD caption overlaps slot 1"],
                    "unanswered_questions_and_mysteries": ["How to reset a locked auxiliary valve?"],
                    "overall_fun_rating": 6,
                }
                return SimpleNamespace(text=json.dumps(report_payload), usage_metadata=usage)

            payload = {
                "deliberation": "Testing terminal interaction and logging UI overlap bug + player wish.",
                "current_plan": "Click central terminal and complete manual check.",
                "turn_log_entry": "Clicked central terminal; noticed caption overlap.",
                "field_note_updates": [
                    {"deliberation": "Status bar counter", "topic": "Conduit Meter", "observation": "Shows 1/3 active links"}
                ],
                "question_updates": [
                    {
                        "deliberation": "Valve reset check",
                        "question_key": "aux_valve_reset",
                        "question": "How to reset a locked auxiliary valve?",
                        "status": "open",
                        "answer": None,
                    }
                ],
                "rulebook_updates": [
                    {"deliberation": "Left click toggles terminal", "key_or_topic": "Left Click", "behavior": "Toggles terminal"}
                ],
                "bug_reports": [
                    {
                        "deliberation": "HUD overlap spotted on bottom bar.",
                        "bug_key": "hud_slot_overlap",
                        "severity": "major",
                        "category": "visual_glitch",
                        "summary": "Bottom prompt label overlaps slot 1 border",
                        "details": "Prompt box intersects top 6px of toolbar slot 1.",
                    }
                ],
                "motor_action": {
                    "deliberation": "Click terminal",
                    "mouse_x": 500.0,
                    "mouse_y": 500.0,
                    "mouse_button_action": "left_click",
                    "drag_from_x": None,
                    "drag_from_y": None,
                    "hold_keys": [],
                    "tap_keys": [],
                    "duration_ms": 500,
                    "inspect_high_res_next": False,
                },
                "submit_wish": {
                    "deliberation": "Terminal needs hover highlight.",
                    "category": "controls_ux",
                    "summary": "Add hover outline on interactive terminals",
                    "rationale": "First-time players cannot tell the terminal is clickable.",
                },
                "end_playtest": {
                    "deliberation": "Completed manual playtest pass.",
                    "outcome": "completed",
                    "closing_thoughts": "Solid core loop; fix HUD overlap and add hover outline.",
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
        start_res = client.post(
            "/api/playtest/sessions/start",
            json={
                "deliberation": "Start manual playtest on active cartridge.",
                "checkpoint_id": "active",
                "max_turns": 5,
            },
        )
        assert start_res.status_code == 200
        session_id = start_res.json()["session"]["session_id"]

        for _ in range(40):
            det_res = client.get(f"/api/playtest/sessions/{session_id}")
            assert det_res.status_code == 200
            det_json = det_res.json()
            if det_json["session"]["status"] == "completed":
                break
            time.sleep(0.05)

        assert det_json["session"]["status"] == "completed"
        assert det_json["session"]["initiated_by"] == "operator"
        assert det_json["session"]["report_sent_to_council_at_utc"] is None
        assert det_json["wishes"][0]["status"] == "pending"
        assert not (cand_dir / ".pending_council_playtest_reports.json").exists()

        # Dispatch full manual report + pending wishes + producer direction to Council
        dispatch_res = client.post(
            f"/api/playtest/sessions/{session_id}/send-report-to-council",
            json={
                "deliberation": "Dispatching manual playtest report to Specialist Council for remediation.",
                "include_pending_wishes": True,
                "producer_note": "Fix the bottom prompt overlap and add terminal hover cues before adding new rooms.",
            },
        )
        assert dispatch_res.status_code == 200
        disp_json = dispatch_res.json()
        assert disp_json["candidate_id"] == "candidate_001"
        assert disp_json["session"]["report_sent_to_council_at_utc"]
        assert disp_json["wishes"][0]["status"] == "sent_to_council"
        assert len(disp_json["proof_image_paths"]) >= 1
        for rel_p in disp_json["proof_image_paths"]:
            assert (cand_dir / rel_p).is_file()

        # Verify persisted session detail reflects Sent-to-Council timestamp
        det_after = client.get(f"/api/playtest/sessions/{session_id}").json()
        assert det_after["session"]["report_sent_to_council_at_utc"] == disp_json["session"]["report_sent_to_council_at_utc"]
        assert det_after["wishes"][0]["status"] == "sent_to_council"

        # Verify Writers' Room transcript in game_bible.json received immediate Executive Producer report utterance + screenshots
        gb_disk = json.loads((cand_dir / "game_bible.json").read_text(encoding="utf-8"))
        last_utt = gb_disk["writers_room_transcript"][-1]
        assert last_utt["speaker_role"] == "ExecutiveProducer"
        assert "Manual Playtest Report" in last_utt["speaker_display_name"]
        assert "Fix the bottom prompt overlap" in last_utt["spoken_dialogue"]
        assert last_utt["proof_image_paths"] == disp_json["proof_image_paths"]

        # Verify CreativeWritersRoom consumes both the full structured report dossier AND multimodal screenshot Parts
        wr = CreativeWritersRoom(client=SimpleNamespace(models=SimpleNamespace()))
        dossier_text, multimodal_parts = wr._consume_pending_playtest_wishes(cand_dir)
        assert "EXECUTIVE PRODUCER DISPATCHED BLIND PLAYTEST FULL REPORT & WISHES" in dossier_text
        assert "Fix the bottom prompt overlap and add terminal hover cues before adding new rooms." in dossier_text
        assert "Bottom prompt label overlaps slot 1 border at 1280x720" in dossier_text
        assert "Add hover outline on interactive terminals" in dossier_text
        assert "How to reset a locked auxiliary valve?" in dossier_text
        assert len(multimodal_parts) >= 2


def test_blind_playtest_final_pass_propagates_note_suspends_scheduler_and_resumes_cleanly(
    tmp_path: Path, monkeypatch
) -> None:
    """Verifies:
    1. Default `STUDIO_MAX_BLIND_PLAYTEST_PASSES` aligns to 3 (matching `max_assembly_passes`).
    2. `_drain_backlog` suspends `BoardScheduler` dispatch during end-of-slice gates so newly added
       remediation tasks are shaped before any worker can claim them.
    3. On `_final_verification_blind_playtest_pass=True`, all 4 `BlindPlaytestCouncilTurn` panelist prompts
       AND `BlindPlaytestCouncilVerdict` receive `POST-REMEDIATION FINAL VERIFICATION PASS` (never 'Pass #N+1 of N'),
       and when the Council rejects promotion (`iterate_current_slice`), neither task shaping nor worker execution
       runs before `StudioAgentError` halts cleanly with blockers persisted on `backlog.json`.
    4. Resuming `_drain_backlog` on the halted cartridge shapes and drains the persisted blockers, runs a
       post-remediation verification pass, and marks the slice `completed` once verified.
    """
    import os
    import pytest
    from engine.studio.models import (
        ReviewBatch,
        ReviewDecision,
        StudioAgentError,
        WorkItem,
        WorkItemProposal,
    )
    from engine.studio.production_board import BoardScheduler, WorkerResult
    from engine.studio.writers_room import BlindPlaytestCouncilTurn, BlindPlaytestCouncilVerdict

    monkeypatch.delenv("STUDIO_MAX_BLIND_PLAYTEST_PASSES", raising=False)
    db_path, cartridges_root, _active_dir, cand_dir = _setup_workspace(tmp_path)
    sessions_root = tmp_path / "playtest_sessions"
    proof_png = cand_dir / "proof.png"
    Image.new("RGB", (64, 64), (40, 90, 120)).save(proof_png, format="PNG")

    run_seq = {"idx": 0}

    def _playtester_factory() -> Any:
        run_seq["idx"] += 1
        current_run = run_seq["idx"]
        local_turn = {"n": 0}

        def _gen(*, model: str, contents: Any, config: Any) -> Any:
            schema = getattr(config, "response_schema", None)
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            usage = SimpleNamespace(
                prompt_token_count=400,
                cached_content_token_count=200,
                candidates_token_count=80,
                thoughts_token_count=0,
            )
            if "overall_experience_summary" in props:
                rep = {
                    "deliberation": f"Report for run #{current_run}.",
                    "overall_experience_summary": f"Summary #{current_run}.",
                    "graphics_and_visuals": "Clean.",
                    "gameplay_and_feel": "Responsive.",
                    "sound_and_audio": "Audible.",
                    "intuitiveness_and_onboarding": "Clear.",
                    "goals_pacing_and_accomplishment": "Structured.",
                    "world_and_atmosphere": "Cohesive.",
                    "key_highlights": ["Core interaction"],
                    "key_pain_points": [],
                    "bugs_and_glitches_observed": [],
                    "unanswered_questions_and_mysteries": [],
                    "overall_fun_rating": 8 if current_run >= 3 else 4,
                }
                return SimpleNamespace(text=json.dumps(rep), usage_metadata=usage)
            local_turn["n"] += 1
            t_n = local_turn["n"]
            turn_payload = {
                "deliberation": f"Turn {t_n}.",
                "current_plan": "Test controls.",
                "turn_log_entry": "Clicked center.",
                "field_note_updates": [],
                "question_updates": [],
                "rulebook_updates": [],
                "bug_reports": [],
                "impression_updates": [],
                "current_mood": "curious",
                "motor_action": {
                    "deliberation": "Click.",
                    "mouse_x": 500.0,
                    "mouse_y": 500.0,
                    "mouse_button_action": "left_click",
                    "drag_from_x": None,
                    "drag_from_y": None,
                    "hold_keys": [],
                    "tap_keys": [],
                    "duration_ms": 300,
                    "inspect_high_res_next": False,
                },
                "mood_shift": None,
                "submit_wish": None,
                "end_playtest": (
                    {"deliberation": "Done.", "outcome": "completed", "closing_thoughts": "Finished."}
                    if t_n >= 2
                    else None
                ),
            }
            return SimpleNamespace(text=json.dumps(turn_payload), usage_metadata=usage)

        return SimpleNamespace(models=SimpleNamespace(generate_content=_gen))

    bus = SwarmEventBus()
    pt_mgr = PlaytestSessionManager(
        db_path=db_path,
        cartridges_root=cartridges_root,
        sessions_root=sessions_root,
        event_bus=bus,
        client_factory=_playtester_factory,
        bridge_factory=_FakeBridge,
    )

    panel_prompts_by_pass: Dict[int, List[str]] = {}
    verdict_prompts_by_pass: Dict[int, str] = {}
    active_pass_holder = {"pass_index": 0, "approve_from_pass": 2}

    def _extract_prompt_text(contents: Any) -> str:
        if isinstance(contents, str):
            return contents
        if isinstance(contents, list):
            return "\n".join(x for x in contents if isinstance(x, str))
        return str(contents)

    def _council_gen(*, model: str, contents: Any, config: Any) -> Any:
        schema_cls = getattr(config, "response_schema", None)
        schema_name = getattr(schema_cls, "__name__", "")
        p_idx = active_pass_holder["pass_index"]
        should_approve = p_idx >= active_pass_holder["approve_from_pass"]
        p_text = _extract_prompt_text(contents)
        usage = SimpleNamespace(
            prompt_token_count=600,
            cached_content_token_count=300,
            candidates_token_count=90,
            thoughts_token_count=10,
        )
        if schema_name == "BlindPlaytestCouncilTurn":
            panel_prompts_by_pass.setdefault(p_idx, []).append(p_text)
            t_obj = BlindPlaytestCouncilTurn(
                deliberation="Auditing playtest session.",
                domain_reaction_to_blind_player="Checked progression.",
                spoken_dialogue="Ready to ship." if should_approve else f"Pass {p_idx} still has a blocking defect.",
                addressed_to=["CreativeDirector"],
                priority_fixes_or_refinements=[] if should_approve else [f"Fix blocker from pass {p_idx}"],
                recommends_slice_approval=should_approve,
            )
            return SimpleNamespace(text=t_obj.model_dump_json(), usage_metadata=usage)
        if schema_name == "BlindPlaytestCouncilVerdict":
            verdict_prompts_by_pass[p_idx] = p_text
            proposals = (
                []
                if should_approve
                else [
                    WorkItemProposal(
                        deliberation="Repair core blocker.",
                        item_id=f"remediation_p{p_idx}_a",
                        kind="mechanic_code",
                        title=f"Remediation {p_idx} A",
                        spec="Repair primary progression state.",
                        proof_spec="Headless verification passes.",
                        target_modules=["main.gd"],
                    ),
                    WorkItemProposal(
                        deliberation="Verify integrated loop.",
                        item_id=f"remediation_p{p_idx}_b",
                        kind="integration",
                        title=f"Remediation {p_idx} B",
                        spec="Verify end-to-end integration.",
                        proof_spec="Headless verification passes.",
                        depends_on=[f"remediation_p{p_idx}_a"],
                        target_modules=["main.gd"],
                    ),
                ]
            )
            v_obj = BlindPlaytestCouncilVerdict(
                deliberation="Synthesizing verdict.",
                player_experience_synthesis="Verified." if should_approve else f"Blocking defect on pass {p_idx}.",
                spoken_dialogue="Approving slice." if should_approve else f"Iterating slice on pass {p_idx}.",
                next_slice_carryover_notes=["Future expansion idea"],
                backlog_additions=proposals,
                verdict="approve_slice" if should_approve else "iterate_current_slice",
            )
            return SimpleNamespace(text=v_obj.model_dump_json(), usage_metadata=usage)
        raise AssertionError(f"Unexpected schema: {schema_name}")

    executed_item_ids: List[str] = []
    shaping_snapshots: List[List[tuple[str, str]]] = []

    def _worker(item: WorkItem) -> WorkerResult:
        executed_item_ids.append(item.item_id)
        return WorkerResult(proof_images=[str(proof_png)], notes=f"Completed {item.item_id}")

    board_1 = ProductionBoard(path=cand_dir / "backlog.json")
    sched_1 = BoardScheduler(
        board_1,
        {"mechanic_code": _worker, "integration": _worker},
        max_workers=2,
        poll_interval=0.005,
    )
    wr_1 = CreativeWritersRoom(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=_council_gen)),
        board=board_1,
        scheduler=sched_1,
        playtest_manager=pt_mgr,
        telemetry_db=TelemetryDB(db_path),
        drain_poll_s=0.005,
        drain_timeout_s=10.0,
    )
    # Verify default max_blind_playtest_passes aligns with max_assembly_passes == 3 when not overridden
    assert int(os.environ.get("STUDIO_MAX_BLIND_PLAYTEST_PASSES", "3")) == 3
    wr_1.max_assembly_passes = 0
    wr_1.max_qa_gate_passes = 0
    wr_1.max_blind_playtest_passes = 1
    wr_1.enable_end_of_slice_blind_playtest = True
    wr_1.enable_task_shaping_on_playtest_iteration = True
    wr_1.slice_playtest_max_turns = 3

    orig_playtest_1 = wr_1._run_end_of_slice_blind_playtest

    def _tracked_playtest_1(game_bible, dna_summary, candidate_dir, pass_index=0):
        active_pass_holder["pass_index"] = pass_index
        return orig_playtest_1(game_bible, dna_summary, candidate_dir, pass_index=pass_index)

    wr_1._run_end_of_slice_blind_playtest = _tracked_playtest_1

    def _fake_shaping_1(game_bible, dna_summary, candidate_dir=None):
        # Sleep briefly so a still-running BoardScheduler would immediately snatch ready items if not suspended!
        import time as _t

        _t.sleep(0.03)
        snap = [
            (it.item_id, it.state)
            for it in board_1.items()
            if it.state not in ("approved", "cancelled")
        ]
        shaping_snapshots.append(snap)
        for it_id, st in snap:
            if st in ("ready", "blocked"):
                cur = board_1.get(it_id)
                cur.shaping_notes.append("[GameplaySystemsEngineer Direction]: Shaped cleanly before worker claim.")
        board_1._persist()

    wr_1._run_council_task_shaping_round = _fake_shaping_1
    wr_1._review_as = lambda role, awaiting, needing, *a, **kw: ReviewBatch(
        deliberation="Approve proofs.",
        spoken_dialogue="Approved.",
        review_decisions=[
            ReviewDecision(deliberation="ok", item_id=it.item_id, verdict="approve", feedback="ok")
            for it in awaiting
        ],
        cap_decisions=[],
    )

    bible = GameBibleSpec(
        deliberation="Fixture Bible.",
        cartridge_id="candidate_001",
        status="in_production",
        game_title="API Fixture Game",
        core_genre="simulation",
        slice_roadmap=[
            GameSliceSpec(
                deliberation="Slice 0.",
                slice_id="slice_0",
                slice_index=0,
                title="Slice 0",
                slice_objective="Core loop",
                status="active",
            )
        ],
        active_slice_index=0,
    )

    try:
        with pytest.raises(StudioAgentError, match="post-remediation Blind Playtest verification after 1 iteration round"):
            wr_1._drain_backlog(bible, "DNA", cand_dir)
    finally:
        sched_1.stop()

    # Pass 0 was normal ("Pass #1 of 1"); Pass 1 was final verification: ALL 4 panelists + chair got the final-pass note!
    assert len(panel_prompts_by_pass[0]) == 4
    assert all("Pass #1 of 1" in p and "POST-REMEDIATION FINAL VERIFICATION PASS" not in p for p in panel_prompts_by_pass[0])
    assert len(panel_prompts_by_pass[1]) == 4
    assert all(
        "Post-Remediation Final Verification Pass after 1 iteration round(s)" in p
        and "POST-REMEDIATION FINAL VERIFICATION PASS" in p
        and "Pass #2 of 1" not in p
        for p in panel_prompts_by_pass[1]
    )
    assert "POST-REMEDIATION FINAL VERIFICATION PASS" in verdict_prompts_by_pass[1]
    assert "Pass #2 of 1" not in verdict_prompts_by_pass[1]

    # Task shaping ran ONLY on Pass 0 (while BoardScheduler dispatch was suspended so items stayed ready/blocked),
    # and did NOT run on terminal Pass 1; Pass 1's blockers were NEVER claimed by workers before circuit-breaker halt!
    assert shaping_snapshots == [[("remediation_p0_a", "ready"), ("remediation_p0_b", "blocked")]]
    assert executed_item_ids == ["remediation_p0_a", "remediation_p0_b"]
    assert board_1.get("remediation_p1_a").state == "ready"
    assert board_1.get("remediation_p1_b").state == "blocked"
    assert bible.slice_roadmap[0].status == "active"

    # Now simulate operator Resume on the halted cartridge: persisted Pass 1 tasks shape, execute, and Pass 2 verifies!
    board_2 = ProductionBoard(path=cand_dir / "backlog.json")
    sched_2 = BoardScheduler(
        board_2,
        {"mechanic_code": _worker, "integration": _worker},
        max_workers=2,
        poll_interval=0.005,
    )
    wr_2 = CreativeWritersRoom(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=_council_gen)),
        board=board_2,
        scheduler=sched_2,
        playtest_manager=pt_mgr,
        telemetry_db=TelemetryDB(db_path),
        drain_poll_s=0.005,
        drain_timeout_s=10.0,
    )
    wr_2.max_assembly_passes = 0
    wr_2.max_qa_gate_passes = 0
    wr_2.max_blind_playtest_passes = 1
    wr_2.enable_end_of_slice_blind_playtest = True
    wr_2.slice_playtest_max_turns = 3

    orig_playtest_2 = wr_2._run_end_of_slice_blind_playtest

    def _tracked_playtest_2(game_bible, dna_summary, candidate_dir, pass_index=0):
        active_pass_holder["pass_index"] = pass_index
        return orig_playtest_2(game_bible, dna_summary, candidate_dir, pass_index=pass_index)

    wr_2._run_end_of_slice_blind_playtest = _tracked_playtest_2
    resumed_shaped: List[str] = []

    def _fake_shaping_2(game_bible, dna_summary, candidate_dir=None):
        for it in board_2.items():
            if it.state in ("ready", "blocked") and not it.shaping_notes:
                resumed_shaped.append(it.item_id)
                it.shaping_notes.append("[GameplaySystemsEngineer Direction]: Shaped on resume.")
        board_2._persist()

    wr_2._run_council_task_shaping_round = _fake_shaping_2
    wr_2._review_as = wr_1._review_as

    try:
        wr_2._drain_backlog(bible, "DNA", cand_dir)
    finally:
        sched_2.stop()

    assert resumed_shaped == ["remediation_p1_a", "remediation_p1_b"]
    assert executed_item_ids == [
        "remediation_p0_a",
        "remediation_p0_b",
        "remediation_p1_a",
        "remediation_p1_b",
    ]
    assert len(panel_prompts_by_pass[2]) == 4
    assert all("POST-REMEDIATION FINAL VERIFICATION PASS" in p for p in panel_prompts_by_pass[2])
    assert bible.slice_roadmap[0].status == "completed"






