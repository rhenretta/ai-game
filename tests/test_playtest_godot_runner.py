"""tests/test_playtest_godot_runner.py
End-to-end integration tests for `godot_playtest_runner.gd`, `GodotPlaytestBridge`,
`mux_broadcast_replay_mp4` (`1600x720` H.264 + AAC), and `PlaytestSessionManager`
lifecycle (`Start -> Pause -> Resume -> Reset -> Stop`).

Verifies 1:1 desktop mouse & cursor parity in real Godot 4.3:
- `Node2D.get_global_mouse_position()` and `Viewport.get_mouse_position()` match `cursor_pos` (so custom in-game cursors/previews render natively),
- GUI `Button` hover & click consumption (does NOT leak to `_unhandled_input` or world `Area2D`),
- World `Area2D` (`input_pickable`) `mouse_entered` + `input_event` picking and `_unhandled_input` world clicks,
- Single-turn `drag_left` (`drag_from_x`/`drag_from_y` -> `mouse_x`/`mouse_y`) smooth trajectory,
- Standard desktop white pointer arrow overlay.

100% domain-agnostic synthetic Godot 4.3 test fixture (Rules 1, 5, 6).
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List
import wave

import imageio_ffmpeg
from PIL import Image

from engine.playtest.godot_bridge import (
    GodotPlaytestBridge,
    SidebarSegment,
    mux_broadcast_replay_mp4,
    render_turn_sidebar_png,
    stage_isolated_checkpoint_runtime,
)
from engine.playtest.models import PlaytestMotorAction, PlaytestSessionStartRequest
from engine.playtest.session_manager import PlaytestSessionManager


def _create_synthetic_godot_checkpoint(cp_dir: Path) -> Path:
    """Creates a minimal, domain-agnostic Godot 4.3 interactive project with:
    - Keyboard movement (`KEY_D` moves Avatar ColorRect right),
    - Clickable HUD Button spanning `(740, 320)..(920, 410)`,
    - `Node2D` world layer tracking `get_global_mouse_position()` + pickable `Area2D` at `(420, 520)`,
    - `_unhandled_input` world click counter and left-button drag motion counter,
    - Continuous 440Hz sine tone via `AudioStreamGenerator` so `AudioEffectCapture` records real game audio,
    - `user://probe_state.json` persistence to verify `user://` isolation and clean-slate Reset.
    """
    cp_dir.mkdir(parents=True, exist_ok=True)
    (cp_dir / "project.godot").write_text(
        '; Engine configuration file.\n'
        'config_version=5\n\n'
        '[application]\n'
        'config/name="Synthetic Playtest Fixture"\n'
        'run/main_scene="res://main.tscn"\n'
        'config/features=PackedStringArray("4.3", "GL Compatibility")\n\n'
        '[display]\n'
        'window/size/viewport_width=1280\n'
        'window/size/viewport_height=720\n',
        encoding="utf-8",
    )
    (cp_dir / "manifest.json").write_text(
        json.dumps(
            {
                "deliberation": "Synthetic fixture manifest.",
                "title": "Synthetic Playtest Fixture",
                "narrative_premise": "An interactive test chamber with a movable marker and button.",
                "core_rules": ["INTERNAL_RULE_EXCLUDED_FROM_RUNTIME"],
                "candidate_id": "candidate_synth",
                "slice_id": "slice_0",
            }
        ),
        encoding="utf-8",
    )
    (cp_dir / "game_bible.json").write_text('{"secret": "DO_NOT_COPY"}', encoding="utf-8")
    (cp_dir / "requirements_spec.json").write_text('{"secret": "DO_NOT_COPY"}', encoding="utf-8")

    (cp_dir / "main.gd").write_text(
        'extends Node2D\n\n'
        'var bg: ColorRect\n'
        'var avatar: ColorRect\n'
        'var action_btn: Button\n'
        'var world_area: Area2D\n'
        'var tone_player: AudioStreamPlayer\n'
        'var playback: AudioStreamGeneratorPlayback\n'
        'var phase: float = 0.0\n'
        'var button_clicks: int = 0\n'
        'var area_clicks: int = 0\n'
        'var area_hovered: bool = false\n'
        'var unhandled_world_clicks: int = 0\n'
        'var drag_motion_events: int = 0\n'
        'var world_mouse_x: float = 0.0\n'
        'var world_mouse_y: float = 0.0\n'
        'var prior_runs_loaded: int = 0\n\n'
        'func _ready() -> void:\n'
        '\tif FileAccess.file_exists("user://probe_state.json"):\n'
        '\t\tprior_runs_loaded = 1\n'
        '\tbg = ColorRect.new()\n'
        '\tbg.name = "Bg"\n'
        '\tbg.color = Color(0.10, 0.14, 0.22)\n'
        '\tbg.size = Vector2(1280, 720)\n'
        '\tbg.mouse_filter = Control.MOUSE_FILTER_IGNORE\n'
        '\tadd_child(bg)\n\n'
        '\tavatar = ColorRect.new()\n'
        '\tavatar.name = "Avatar"\n'
        '\tavatar.color = Color(0.25, 0.85, 0.55)\n'
        '\tavatar.position = Vector2(200, 330)\n'
        '\tavatar.size = Vector2(64, 64)\n'
        '\tavatar.mouse_filter = Control.MOUSE_FILTER_IGNORE\n'
        '\tadd_child(avatar)\n\n'
        '\tworld_area = Area2D.new()\n'
        '\tworld_area.name = "WorldTarget"\n'
        '\tworld_area.position = Vector2(420, 520)\n'
        '\tworld_area.input_pickable = true\n'
        '\tvar cs := CollisionShape2D.new()\n'
        '\tvar circle := CircleShape2D.new()\n'
        '\tcircle.radius = 45.0\n'
        '\tcs.shape = circle\n'
        '\tworld_area.add_child(cs)\n'
        '\tworld_area.mouse_entered.connect(func():\n'
        '\t\tarea_hovered = true\n'
        '\t\t_save_probe_state()\n'
        '\t)\n'
        '\tworld_area.input_event.connect(func(_vp, ev, _idx):\n'
        '\t\tif ev is InputEventMouseButton and ev.pressed and ev.button_index == MOUSE_BUTTON_LEFT:\n'
        '\t\t\tarea_clicks += 1\n'
        '\t\t\t_save_probe_state()\n'
        '\t)\n'
        '\tadd_child(world_area)\n\n'
        '\tvar hud := CanvasLayer.new()\n'
        '\tadd_child(hud)\n'
        '\taction_btn = Button.new()\n'
        '\taction_btn.name = "ActionButton"\n'
        '\taction_btn.text = "ACTIVATE"\n'
        '\taction_btn.position = Vector2(740, 320)\n'
        '\taction_btn.size = Vector2(180, 90)\n'
        '\taction_btn.pressed.connect(_on_btn_pressed)\n'
        '\thud.add_child(action_btn)\n\n'
        '\ttone_player = AudioStreamPlayer.new()\n'
        '\tvar gen := AudioStreamGenerator.new()\n'
        '\tgen.mix_rate = 44100.0\n'
        '\tgen.buffer_length = 0.5\n'
        '\ttone_player.stream = gen\n'
        '\tadd_child(tone_player)\n'
        '\ttone_player.play()\n'
        '\tplayback = tone_player.get_stream_playback() as AudioStreamGeneratorPlayback\n'
        '\t_fill_tone()\n'
        '\t_save_probe_state()\n\n'
        'func _on_btn_pressed() -> void:\n'
        '\tbutton_clicks += 1\n'
        '\tbg.color = Color(0.22, 0.34, 0.18)\n'
        '\t_save_probe_state()\n\n'
        'func _unhandled_input(event: InputEvent) -> void:\n'
        '\tif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:\n'
        '\t\tunhandled_world_clicks += 1\n'
        '\t\t_save_probe_state()\n'
        '\telif event is InputEventMouseMotion and (event.button_mask & MOUSE_BUTTON_MASK_LEFT) != 0:\n'
        '\t\tdrag_motion_events += 1\n'
        '\t\t_save_probe_state()\n\n'
        'func _draw() -> void:\n'
        '\t# Custom in-world reticle drawn at get_global_mouse_position() to verify 1:1 mouse parity\n'
        '\tdraw_arc(Vector2(world_mouse_x, world_mouse_y), 14.0, 0.0, TAU, 24, Color(0.3, 0.95, 0.75, 0.9), 2.0)\n\n'
        'func _process(delta: float) -> void:\n'
        '\t_fill_tone()\n'
        '\tvar gm := get_global_mouse_position()\n'
        '\tif absf(gm.x - world_mouse_x) > 0.1 or absf(gm.y - world_mouse_y) > 0.1:\n'
        '\t\tworld_mouse_x = gm.x\n'
        '\t\tworld_mouse_y = gm.y\n'
        '\t\tqueue_redraw()\n'
        '\t\t_save_probe_state()\n'
        '\tif Input.is_physical_key_pressed(KEY_D) or Input.is_key_pressed(KEY_D):\n'
        '\t\tavatar.position.x += 320.0 * delta\n'
        '\t\t_save_probe_state()\n\n'
        'func _fill_tone() -> void:\n'
        '\tif playback == null:\n'
        '\t\treturn\n'
        '\tvar avail := playback.get_frames_available()\n'
        '\tvar inc := 440.0 / 44100.0\n'
        '\tfor _i in range(min(avail, 2048)):\n'
        '\t\tvar s := sin(phase * TAU) * 0.25\n'
        '\t\tplayback.push_frame(Vector2(s, s))\n'
        '\t\tphase = fmod(phase + inc, 1.0)\n\n'
        'func _save_probe_state() -> void:\n'
        '\tvar f := FileAccess.open("user://probe_state.json", FileAccess.WRITE)\n'
        '\tif f != null:\n'
        '\t\tf.store_string(JSON.stringify({\n'
        '\t\t\t"avatar_x": avatar.position.x,\n'
        '\t\t\t"button_clicks": button_clicks,\n'
        '\t\t\t"area_clicks": area_clicks,\n'
        '\t\t\t"area_hovered": area_hovered,\n'
        '\t\t\t"unhandled_world_clicks": unhandled_world_clicks,\n'
        '\t\t\t"drag_motion_events": drag_motion_events,\n'
        '\t\t\t"world_mouse_x": world_mouse_x,\n'
        '\t\t\t"world_mouse_y": world_mouse_y,\n'
        '\t\t\t"prior_runs_loaded": prior_runs_loaded,\n'
        '\t\t}))\n'
        '\t\tf.close()\n',
        encoding="utf-8",
    )
    (cp_dir / "main.tscn").write_text(
        '[gd_scene load_steps=2 format=3]\n\n'
        '[ext_resource type="Script" path="res://main.gd" id="1"]\n\n'
        '[node name="Main" type="Node2D"]\n'
        'script = ExtResource("1")\n',
        encoding="utf-8",
    )
    return cp_dir


def _find_probe_state_file(user_home_dir: Path) -> Path:
    matches = list(user_home_dir.rglob("probe_state.json"))
    assert matches, f"Expected isolated user://probe_state.json inside {user_home_dir}"
    return matches[0]


def test_stage_isolated_checkpoint_runtime_excludes_studio_metadata(tmp_path: Path):
    cp_dir = _create_synthetic_godot_checkpoint(tmp_path / "cp_src")
    sess_dir = tmp_path / "session_01"
    runtime_dir, user_home_dir = stage_isolated_checkpoint_runtime(cp_dir, sess_dir)

    assert (runtime_dir / "project.godot").is_file()
    assert (runtime_dir / "main.tscn").is_file()
    assert (runtime_dir / "main.gd").is_file()
    assert user_home_dir.is_dir()

    for excluded in ("manifest.json", "game_bible.json", "requirements_spec.json"):
        assert not (runtime_dir / excluded).exists()


def test_godot_playtest_bridge_desktop_mouse_parity_audio_and_1600x720_mux(tmp_path: Path):
    cp_dir = _create_synthetic_godot_checkpoint(tmp_path / "cp_live")
    sess_dir = tmp_path / "sess_live"
    live_frames: List[Dict[str, Any]] = []

    bridge = GodotPlaytestBridge(
        checkpoint_dir=cp_dir,
        session_dir=sess_dir,
        reaction_ticks=15,
        initial_settle_ticks=20,
        record_video=True,
        on_live_frame=lambda pkt: live_frames.append(pkt),
    )
    sidebar_segments: List[SidebarSegment] = []
    try:
        obs0 = bridge.start()
        assert obs0.tick == 20
        assert (round(obs0.cursor_x), round(obs0.cursor_y)) == (640, 360)
        with Image.open(obs0.screenshot_path) as im0:
            assert im0.size == (1280, 720)
            # Verify standard white desktop pointer arrow interior pixel near (642, 366)
            arrow_px = im0.convert("RGB").getpixel((642, 366))
            assert arrow_px[0] > 235 and arrow_px[1] > 235 and arrow_px[2] > 235, (
                f"Expected crisp white desktop arrow cursor interior at (642, 366), got {arrow_px}"
            )

        # Verify in-world get_global_mouse_position() matched (640, 360) right from boot!
        assert bridge.user_home_dir is not None
        probe0 = json.loads(_find_probe_state_file(bridge.user_home_dir).read_text(encoding="utf-8"))
        assert abs(probe0["world_mouse_x"] - 640.0) < 1.0
        assert abs(probe0["world_mouse_y"] - 360.0) < 1.0

        # Verify silent offscreen AudioEffectCapture captured real 16kHz non-silent game audio
        assert obs0.audio_is_silent is False
        assert obs0.audio_rms > 0.01
        assert obs0.wav_path is not None and obs0.wav_path.is_file()
        with wave.open(str(obs0.wav_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() > 2000

        s0 = render_turn_sidebar_png(
            sess_dir / "media" / "sidebar_turn_0000.png",
            game_title="Synthetic Playtest Fixture",
            turn_index=0,
            sim_time_sec=obs0.sim_time_sec,
            current_mood="curious",
            current_plan="Test HUD button, world Area2D click, and drag.",
            sticky_feedback_items=[],
            action_summary="Observing opening screen",
            cursor_xy=(obs0.cursor_x, obs0.cursor_y),
        )
        sidebar_segments.append(SidebarSegment(turn_index=0, start_tick=0, end_tick=obs0.tick, sidebar_png_path=s0))

        # Action 1: Hold 'D' for 550ms (33 ticks) + move cursor to HUD ACTIVATE button
        # Normalized (640.625, 500.0) -> Pixel (820.0, 360.0) + left_click!
        act1 = PlaytestMotorAction(
            deliberation="Hold D to move avatar right and point cursor onto ACTIVATE HUD button at normalized (640.6, 500.0) and click.",
            mouse_x=640.625,
            mouse_y=500.0,
            mouse_button_action="left_click",
            hold_keys=["D"],
            duration_ms=550,
        )
        obs1 = bridge.step_action(act1)

        # Tick math: 20 (obs0) + 15 (reaction_ticks) + 33 (550ms @ 60fps) == 68!
        assert obs1.tick == 20 + 15 + 33
        assert abs(obs1.cursor_x - 820.0) < 1.5
        assert abs(obs1.cursor_y - 360.0) < 1.5
        assert len(live_frames) >= 1

        probe1 = json.loads(_find_probe_state_file(bridge.user_home_dir).read_text(encoding="utf-8"))
        assert probe1["avatar_x"] > 280.0, f"Expected D key to move avatar right from 200, got {probe1}"
        assert probe1["button_clicks"] == 1, f"Expected HUD Button click, got {probe1}"
        # HUD Button consumed the click -> unhandled_world_clicks must still be 0!
        assert probe1["unhandled_world_clicks"] == 0, f"HUD click should not leak to _unhandled_input: {probe1}"
        assert abs(probe1["world_mouse_x"] - 820.0) < 1.5
        assert probe1["prior_runs_loaded"] == 0

        # Action 2: Point cursor to world Area2D at normalized (328.125, 722.222) -> pixel (420, 520) and left_click!
        act2 = PlaytestMotorAction(
            deliberation="Point cursor to world Area2D at normalized (328.1, 722.2) and click it.",
            mouse_x=328.125,
            mouse_y=722.222,
            mouse_button_action="left_click",
            duration_ms=450,
        )
        obs2 = bridge.step_action(act2)
        assert abs(obs2.cursor_x - 420.0) < 1.5
        assert abs(obs2.cursor_y - 520.0) < 1.5

        probe2 = json.loads(_find_probe_state_file(bridge.user_home_dir).read_text(encoding="utf-8"))
        assert probe2["area_hovered"] is True, f"Expected Area2D mouse_entered on hover: {probe2}"
        assert probe2["area_clicks"] == 1, f"Expected Area2D input_event click: {probe2}"
        assert probe2["unhandled_world_clicks"] == 1, f"Expected world click in _unhandled_input: {probe2}"
        assert abs(probe2["world_mouse_x"] - 420.0) < 1.5
        assert abs(probe2["world_mouse_y"] - 520.0) < 1.5

        # Action 3: Single-turn drag_left from normalized (250, 300) [pixel (320, 216)] to (450, 400) [pixel (576, 288)]!
        act3 = PlaytestMotorAction(
            deliberation="Test single-turn drag gesture across world canvas.",
            drag_from_x=250.0,
            drag_from_y=300.0,
            mouse_x=450.0,
            mouse_y=400.0,
            mouse_button_action="drag_left",
            duration_ms=600,
        )
        obs3 = bridge.step_action(act3)
        assert abs(obs3.cursor_x - 576.0) < 1.5
        assert abs(obs3.cursor_y - 288.0) < 1.5

        probe3 = json.loads(_find_probe_state_file(bridge.user_home_dir).read_text(encoding="utf-8"))
        assert probe3["drag_motion_events"] >= 5, f"Expected smooth held drag motion events, got {probe3}"

        s1 = render_turn_sidebar_png(
            sess_dir / "media" / "sidebar_turn_0001.png",
            game_title="Synthetic Playtest Fixture",
            turn_index=1,
            sim_time_sec=obs3.sim_time_sec,
            current_mood="satisfied",
            current_plan="Verified button, world Area2D, and drag.",
            sticky_feedback_items=[
                {
                    "kind": "mood",
                    "turn_label": "T#1",
                    "badge": "MOOD -> SATISFIED",
                    "text": "HUD button, world Area2D, and drag responded cleanly.",
                }
            ],
            action_summary="Move (820,360) • Left Click",
            cursor_xy=(obs3.cursor_x, obs3.cursor_y),
        )
        sidebar_segments.append(
            SidebarSegment(turn_index=1, start_tick=obs0.tick, end_tick=obs3.tick, sidebar_png_path=s1)
        )
    finally:
        bridge.close()

    # Source checkpoint directory must remain 100% untouched (no .godot cache created in cp_dir)
    assert not (cp_dir / ".godot").exists()

    # Mux 1600x720 H.264 + AAC broadcast replay MP4 and verify stream geometry & codecs
    sidebar_segments[-1].end_tick = max(sidebar_segments[-1].end_tick, bridge.last_tick)
    mp4_path = mux_broadcast_replay_mp4(
        raw_game_mkv=bridge.raw_game_mkv,
        session_pcm_path=bridge.session_pcm_path,
        total_sim_ticks=bridge.last_tick,
        sidebar_segments=sidebar_segments,
        output_mp4_path=sess_dir / "media" / "replay.mp4",
    )
    assert mp4_path.is_file() and mp4_path.stat().st_size > 4096

    probe_res = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(mp4_path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    ff_info = probe_res.stderr
    assert "1600x720" in ff_info, f"Expected 1600x720 broadcast canvas in ffmpeg probe:\n{ff_info}"
    assert "h264" in ff_info.lower()
    assert "aac" in ff_info.lower()


def test_session_manager_pause_interim_video_resume_reset_and_stop(tmp_path: Path):
    cartridges_root = tmp_path / "cartridges"
    cp_dir = _create_synthetic_godot_checkpoint(cartridges_root / "checkpoints" / "cp_slice_0")
    assert cp_dir.is_dir()

    turn_gate = threading.Event()
    call_counter = {"n": 0}

    def _generate_content(*, model: str, contents: Any, config: Any) -> Any:
        schema = getattr(config, "response_schema", None) or {}
        props = schema.get("properties") if isinstance(schema, dict) else {}
        if isinstance(props, dict) and "overall_experience_summary" in props:
            rep_payload = {
                "deliberation": "Synthesizing session experience report after session stopped.",
                "overall_experience_summary": "Responsive movement and crisp button feedback during the stopped session.",
                "graphics_and_visuals": "Clear spatial contrast between HUD and world elements.",
                "gameplay_and_feel": "Holding D moved the avatar smoothly across the viewport.",
                "sound_and_audio": "Continuous synthesized tone confirmed active audio stream.",
                "intuitiveness_and_onboarding": "ACTIVATE button and world target were easy to test.",
                "goals_pacing_and_accomplishment": "Immediate movement and click feedback felt brisk.",
                "world_and_atmosphere": "Clean test chamber presentation.",
                "key_highlights": ["Smooth 60fps cursor and keyboard response"],
                "key_pain_points": [],
                "overall_fun_rating": 8,
            }
            usage = SimpleNamespace(
                prompt_token_count=950,
                cached_content_token_count=700,
                candidates_token_count=140,
                thoughts_token_count=40,
            )
            return SimpleNamespace(text=json.dumps(rep_payload), usage_metadata=usage)

        call_counter["n"] += 1
        n = call_counter["n"]
        if n >= 2:
            # Slow slightly after turn 1 so the test can deterministically test Pause -> interim MP4 -> Resume
            turn_gate.wait(timeout=4.0)
        payload = {
            "deliberation": f"Player turn {n} deliberation.",
            "current_plan": f"Testing chamber step {n}",
            "turn_log_entry": f"Moved cursor and held D on step {n}",
            "rulebook_updates": [
                {"deliberation": "Confirmed D", "key_or_topic": "Key D", "behavior": "Moves marker right"}
            ],
            "impression_updates": [
                {
                    "deliberation": "Noting responsive movement feel.",
                    "pillar": "gameplay_and_feel",
                    "impression": "Avatar movement feels crisp and immediate.",
                }
            ],
            "motor_action": {
                "deliberation": "Move right",
                "mouse_x": 531.25,
                "mouse_y": 500.0,
                "mouse_button_action": "none",
                "drag_from_x": None,
                "drag_from_y": None,
                "hold_keys": ["D"],
                "tap_keys": [],
                "duration_ms": 400,
                "inspect_high_res_next": False,
            },
            "mood_shift": None,
            "submit_wish": None,
            "end_playtest": None,
        }
        usage = SimpleNamespace(
            prompt_token_count=900,
            cached_content_token_count=600,
            candidates_token_count=90,
            thoughts_token_count=30,
        )
        return SimpleNamespace(text=json.dumps(payload), usage_metadata=usage)

    def _client_factory() -> Any:
        return SimpleNamespace(models=SimpleNamespace(generate_content=_generate_content))

    mgr = PlaytestSessionManager(
        db_path=tmp_path / "telemetry.db",
        cartridges_root=cartridges_root,
        sessions_root=tmp_path / "playtest_sessions",
        client_factory=_client_factory,
    )

    started = mgr.start_session(
        PlaytestSessionStartRequest(
            checkpoint_id="cp_slice_0",
            max_turns=12,
            reaction_ticks=10,
        )
    )
    sid1 = started["session_id"]

    # Wait until Turn 1 completes
    deadline = time.monotonic() + 25.0
    while time.monotonic() < deadline:
        det = mgr.get_session_detail(sid1)
        if det and det["session"]["turn_count"] >= 1:
            break
        time.sleep(0.15)
    assert det is not None and det["session"]["turn_count"] >= 1

    # Pause session and release turn_gate so worker hits the pause gate and muxes interim replay.mp4
    paused = mgr.control_session(sid1, "pause")
    assert paused["status"] == "paused"
    turn_gate.set()

    deadline = time.monotonic() + 15.0
    has_interim_video = False
    while time.monotonic() < deadline:
        det_p = mgr.get_session_detail(sid1)
        if det_p and det_p["session"].get("has_video"):
            has_interim_video = True
            break
        time.sleep(0.2)
    assert has_interim_video, "Expected interim replay.mp4 to be muxed while session is paused"

    # Resume session
    resumed = mgr.control_session(sid1, "resume")
    assert resumed["status"] == "running"

    # Reset session -> stops sid1 and launches a fresh clean-slate session sid2
    reset_sess = mgr.control_session(sid1, "reset")
    sid2 = reset_sess["session_id"]
    assert sid2 != sid1

    # Wait for sid2 to complete at least 1 turn, then Stop it cleanly
    deadline = time.monotonic() + 25.0
    while time.monotonic() < deadline:
        det2 = mgr.get_session_detail(sid2)
        if det2 and det2["session"]["turn_count"] >= 1:
            break
        time.sleep(0.15)

    stopped = mgr.control_session(sid2, "stop")
    assert stopped["status"] == "stopped"
    assert stopped["session_report"]["overall_fun_rating"] == 8
    assert stopped["impressions"]["gameplay_and_feel"].startswith("Avatar movement feels crisp")

    # Verify sid2 started with a completely isolated user:// directory (`prior_runs_loaded == 0`)
    sid2_user_home = tmp_path / "playtest_sessions" / sid2 / "user_home"
    probe2 = json.loads(_find_probe_state_file(sid2_user_home).read_text(encoding="utf-8"))
    assert probe2["prior_runs_loaded"] == 0

    # Verify sid1 was completely deleted when reset while running
    assert mgr.get_session_detail(sid1) is None

    # Resetting a STOPPED playtester wipes its data and user:// state to 'idle' WITHOUT starting it
    calls_before_stopped_reset = call_counter["n"]
    reset_stopped = mgr.control_session(sid2, "reset")
    assert reset_stopped["session_id"] == sid2
    assert reset_stopped["status"] == "idle"
    assert reset_stopped["turn_count"] == 0
    assert reset_stopped["rulebook"] == {}
    assert reset_stopped["impressions"] == {}
    assert reset_stopped["session_report"] == {}
    assert reset_stopped["has_video"] is False
    assert not sid2_user_home.exists()
    assert mgr._handles == {}
    assert call_counter["n"] == calls_before_stopped_reset
