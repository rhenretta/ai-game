"""engine/playtest/godot_bridge.py
Interactive Godot 4.3 Process Host, Isolated Runtime Stager, Audio WAV Encoder,
and 1600x720 Broadcast Replay Video Compositor (Rules 5 & 6).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import socket
import struct
import subprocess
import textwrap
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import wave

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

from engine.godot_runtime import ensure_godot_background_render_binary
from engine.playtest.models import (
    PlaytestMotorAction,
    PlaytestRunnerError,
    denormalize_screen_x,
    denormalize_screen_y,
)

logger = logging.getLogger("GodotPlaytestBridge")

RUNNER_SCRIPT_PATH = Path(__file__).resolve().parent / "godot_playtest_runner.gd"
TARGET_AUDIO_HZ = 16000
SILENCE_RMS_THRESHOLD = 0.0012

_EXCLUDED_CHECKPOINT_NAMES = {
    ".godot",
    "__pycache__",
    "artifacts",
    "tests",
    "manifest.json",
    "checkpoint_metadata.json",
    "task_history.json",
    "telemetry_manifest.json",
    "world_spec.json",
    "game_bible.json",
    "requirements_spec.json",
    "backlog.json",
    "production_board.json",
    ".pending_steering.json",
    ".pending_playtest_wishes.json",
    ".pending_council_playtest_reports.json",
}


@dataclass
class TurnObservation:
    """Single deterministic viewport + audio observation captured from Godot."""
    observation_index: int
    tick: int
    sim_time_sec: float
    cursor_x: float
    cursor_y: float
    screenshot_path: Path
    preview_path: Path
    wav_path: Optional[Path]
    audio_rms: float
    audio_peak: float
    audio_is_silent: bool
    interrupted_early: bool


@dataclass
class SidebarSegment:
    """Tick-accurate HUD sidebar card span for the 1600x720 broadcast replay video."""
    turn_index: int
    start_tick: int
    end_tick: int
    sidebar_png_path: Path


def _get_font(size: int) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _reserve_ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def stage_isolated_checkpoint_runtime(checkpoint_dir: Path, session_dir: Path) -> Tuple[Path, Path]:
    """Stages an isolated Godot project copy under `<session_dir>/runtime/` and `<session_dir>/user_home/`.
    Guarantees:
      1. `checkpoint_dir` is never mutated by runtime files or `.godot/` import caches.
      2. Studio metadata (`manifest.json`, `game_bible.json`, `requirements_spec.json`) is excluded.
      3. `user://` saves live inside `<session_dir>/user_home/` so Reset starts from a 100% clean slate.
    """
    src_dir = Path(checkpoint_dir).resolve()
    if not src_dir.is_dir() or not (src_dir / "project.godot").is_file():
        raise PlaytestRunnerError(
            f"Cannot stage playtest runtime: invalid Godot checkpoint directory '{src_dir}' (missing project.godot)."
        )

    runtime_dir = (Path(session_dir).resolve() / "runtime")
    user_home_dir = (Path(session_dir).resolve() / "user_home")
    for target in (runtime_dir, user_home_dir):
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    for entry in src_dir.iterdir():
        if entry.name in _EXCLUDED_CHECKPOINT_NAMES:
            continue
        dst_entry = runtime_dir / entry.name
        if entry.name == "assets" and entry.is_dir():
            dst_entry.mkdir(parents=True, exist_ok=True)
            for asset_child in entry.iterdir():
                if asset_child.name == "playtest_captures":
                    (dst_entry / "playtest_captures").mkdir(parents=True, exist_ok=True)
                    continue
                target_link = dst_entry / asset_child.name
                try:
                    os.symlink(asset_child.resolve(), target_link, target_is_directory=asset_child.is_dir())
                except OSError:
                    if asset_child.is_dir():
                        shutil.copytree(asset_child, target_link)
                    else:
                        shutil.copy2(asset_child, target_link)
        elif entry.is_dir():
            shutil.copytree(entry, dst_entry)
        elif entry.is_file():
            shutil.copy2(entry, dst_entry)

    return runtime_dir, user_home_dir


def write_pcm16_to_wav(pcm_bytes: bytes, wav_path: Path, sample_rate: int = TARGET_AUDIO_HZ) -> Path:
    """Encodes raw 16-bit signed mono little-endian PCM bytes into a standard `.wav` file."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_path


def render_turn_sidebar_png(
    output_path: Path,
    *,
    game_title: str,
    turn_index: int,
    sim_time_sec: float,
    current_mood: str,
    current_plan: str,
    sticky_feedback_items: List[Dict[str, str]],
    action_summary: str,
    cursor_xy: Tuple[float, float],
) -> Path:
    """Renders a 320x720 dark broadcast HUD sidebar PNG for `replay.mp4`.
    Maintains a sticky rolling stack of the latest 3 mood notes / wish items so player
    commentary stays legible across 15-45 seconds of gameplay video.
    """
    w, h = 320, 720
    img = Image.new("RGB", (w, h), (13, 17, 26))
    draw = ImageDraw.Draw(img)

    f_xs = _get_font(11)
    f_sm = _get_font(12)
    f_md = _get_font(14)
    f_lg = _get_font(17)

    # Left separator border between 1280x720 game feed and 320x720 AI Player HUD
    draw.line([(0, 0), (0, h)], fill=(45, 58, 84), width=2)

    # Header block
    draw.rectangle([(2, 0), (w, 66)], fill=(19, 25, 38))
    draw.text((14, 10), "BLIND AI PLAYTESTER", fill=(103, 232, 249), font=f_xs)
    short_title = (game_title or "Untitled Slice")[:28]
    draw.text((14, 26), short_title, fill=(241, 245, 249), font=f_md)
    mins = int(sim_time_sec // 60)
    secs = sim_time_sec - mins * 60
    draw.text((14, 46), f"Turn #{turn_index:03d}   •   {mins:02d}:{secs:04.1f}", fill=(148, 163, 184), font=f_xs)

    # Current Mood Badge
    y = 78
    draw.text((14, y), "CURRENT MOOD", fill=(148, 163, 184), font=f_xs)
    y += 16
    mood_display = (current_mood or "observing").strip().upper()[:22]
    draw.rounded_rectangle([(14, y), (w - 14, y + 36)], radius=8, fill=(24, 36, 56), outline=(56, 189, 248), width=1)
    draw.text((26, y + 9), mood_display, fill=(224, 242, 254), font=f_lg)
    y += 48

    # Active Plan ("What player is currently trying")
    draw.text((14, y), "ACTIVE INTENT", fill=(148, 163, 184), font=f_xs)
    y += 15
    plan_lines = textwrap.wrap(current_plan or "Exploring initial screen and testing controls.", width=36)[:3]
    plan_box_h = max(44, 14 + len(plan_lines) * 15)
    draw.rounded_rectangle([(14, y), (w - 14, y + plan_box_h)], radius=6, fill=(18, 24, 36), outline=(42, 52, 73), width=1)
    py = y + 7
    for line in plan_lines:
        draw.text((22, py), line, fill=(203, 213, 225), font=f_sm)
        py += 15
    y += plan_box_h + 14

    # Sticky Rolling Stack of Latest 3 Mood Shifts / Wishes
    draw.text((14, y), "PLAYER THOUGHTS & WISHES (RECENT)", fill=(148, 163, 184), font=f_xs)
    y += 16
    recent = list(sticky_feedback_items[-3:])[::-1]  # Newest at top
    if not recent:
        draw.rounded_rectangle([(14, y), (w - 14, y + 58)], radius=6, fill=(18, 23, 34), outline=(35, 45, 64), width=1)
        draw.text((22, y + 14), "Getting a feel for the game...", fill=(100, 116, 139), font=f_sm)
        draw.text((22, y + 31), "Notes appear when mood shifts.", fill=(100, 116, 139), font=f_xs)
    else:
        for idx, item in enumerate(recent):
            if y > h - 135:
                break
            is_newest = (idx == 0)
            kind = item.get("kind", "mood")
            badge = item.get("badge", "")
            body = item.get("text", "")
            turn_tag = item.get("turn_label", "")
            wrapped = textwrap.wrap(body, width=35)[:4]
            card_h = 26 + len(wrapped) * 14 + 8
            if kind == "wish":
                bg = (42, 30, 18) if is_newest else (28, 22, 17)
                border = (245, 158, 11) if is_newest else (120, 83, 28)
                hdr_col = (253, 224, 71) if is_newest else (217, 179, 84)
            elif kind == "impression":
                bg = (20, 38, 32) if is_newest else (16, 26, 23)
                border = (52, 211, 153) if is_newest else (34, 97, 74)
                hdr_col = (110, 231, 183) if is_newest else (134, 189, 168)
            else:
                bg = (21, 33, 48) if is_newest else (17, 24, 36)
                border = (56, 189, 248) if is_newest else (45, 62, 85)
                hdr_col = (125, 211, 252) if is_newest else (148, 163, 184)
            txt_col = (241, 245, 249) if is_newest else (165, 178, 196)

            draw.rounded_rectangle([(14, y), (w - 14, min(h - 96, y + card_h))], radius=6, fill=bg, outline=border, width=1)
            draw.text((22, y + 6), f"{turn_tag}  {badge}"[:36], fill=hdr_col, font=f_xs)
            ty = y + 22
            for wline in wrapped:
                if ty > h - 110:
                    break
                draw.text((22, ty), wline, fill=txt_col, font=f_sm)
                ty += 14
            y += card_h + 8

    # Controller Telemetry Footer
    footer_top = h - 86
    draw.rectangle([(2, footer_top), (w, h)], fill=(17, 23, 35))
    draw.line([(2, footer_top), (w, footer_top)], fill=(42, 54, 78), width=1)
    draw.text((14, footer_top + 8), "INPUT CONTROLLER", fill=(148, 163, 184), font=f_xs)
    act_lines = textwrap.wrap(action_summary or "Idle / Observing", width=36)[:2]
    ay = footer_top + 24
    for aline in act_lines:
        draw.text((14, ay), aline, fill=(167, 243, 208), font=f_sm)
        ay += 15
    draw.text(
        (14, h - 22),
        f"Cursor Reticle: ({int(round(cursor_xy[0]))}, {int(round(cursor_xy[1]))})",
        fill=(148, 163, 184),
        font=f_xs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return output_path


def mux_broadcast_replay_mp4(
    *,
    raw_game_mkv: Path,
    session_pcm_path: Path,
    total_sim_ticks: int,
    sidebar_segments: List[SidebarSegment],
    output_mp4_path: Path,
) -> Path:
    """Muxes the 1280x720 60 FPS game stream + 320x720 sticky AI Mood/Feedback sidebar +
    16kHz game audio into a browser-seekable 1600x720 H.264/AAC MP4 (`-movflags +faststart`).
    Uses atomic temp-file replacement so concurrent browser `<video>` readers never see partial writes.
    """
    if not raw_game_mkv.is_file() or raw_game_mkv.stat().st_size == 0:
        raise PlaytestRunnerError(f"Cannot mux replay MP4: missing or empty game video stream '{raw_game_mkv}'.")
    if not sidebar_segments:
        raise PlaytestRunnerError("Cannot mux replay MP4: no sidebar HUD segments recorded.")

    media_dir = output_mp4_path.parent
    media_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build tick-exact ffconcat manifest for the 320x720 sidebar stream
    concat_path = media_dir / "sidebar_concat.txt"
    concat_lines = ["ffconcat version 1.0"]
    for seg in sidebar_segments:
        dur_sec = max(1.0 / 60.0, float(max(1, seg.end_tick - seg.start_tick)) / 60.0)
        escaped_path = str(seg.sidebar_png_path.resolve()).replace("'", "'\\''")
        concat_lines.append(f"file '{escaped_path}'")
        concat_lines.append(f"duration {dur_sec:.6f}")
    # Repeat final image per ffmpeg concat demuxer specification
    last_escaped = str(sidebar_segments[-1].sidebar_png_path.resolve()).replace("'", "'\\''")
    concat_lines.append(f"file '{last_escaped}'")
    concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    # 2. Materialize cumulative session_audio.wav (pad silent samples if cartridge emitted no audio)
    expected_samples = max(int(round((max(1, total_sim_ticks) / 60.0) * TARGET_AUDIO_HZ)), 1600)
    raw_pcm = session_pcm_path.read_bytes() if session_pcm_path.is_file() else b""
    actual_samples = len(raw_pcm) // 2
    if actual_samples < expected_samples:
        raw_pcm = raw_pcm + (b"\x00\x00" * (expected_samples - actual_samples))
    session_wav_path = media_dir / "session_audio.wav"
    write_pcm16_to_wav(raw_pcm, session_wav_path, sample_rate=TARGET_AUDIO_HZ)

    # 3. Mux side-by-side 1600x720 H.264 + AAC MP4 via portable imageio-ffmpeg binary
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_mp4 = output_mp4_path.with_name(f".tmp_{output_mp4_path.name}")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-fflags", "+genpts",
        "-i", str(raw_game_mkv),
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_path),
        "-i", str(session_wav_path),
        "-filter_complex",
        "[0:v]scale=1280:720,fps=60,format=yuv420p[game];"
        "[1:v]scale=320:720,fps=60,format=yuv420p[hud];"
        "[game][hud]hstack=inputs=2[vout]",
        "-map", "[vout]",
        "-map", "2:a:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        str(tmp_mp4),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0 or not tmp_mp4.is_file():
        tmp_mp4.unlink(missing_ok=True)
        raise PlaytestRunnerError(
            f"ffmpeg broadcast replay muxing failed (exit {res.returncode}):\n{(res.stderr or '')[-1200:]}"
        )

    tmp_mp4.replace(output_mp4_path)
    return output_mp4_path


def extract_recent_gameplay_clip_mp4(
    *,
    raw_game_mkv: Path,
    session_pcm_path: Path,
    end_tick: int,
    lookback_seconds: float,
    output_mp4_path: Path,
) -> Path:
    """Slices the most recent `lookback_seconds` (clamped 2.0..20.0s) from the pure 1280x720
    `raw_game_mkv` stream (NO HUD sidebar — preserving strict opaque-box isolation) paired with
    its synchronized 16kHz mono PCM audio window into a compact 640x360 H.264/AAC `.mp4` clip.
    """
    if not raw_game_mkv.is_file() or raw_game_mkv.stat().st_size == 0:
        raise PlaytestRunnerError(
            f"Cannot extract recent gameplay clip: missing or empty game video stream '{raw_game_mkv}' (Rule 5)."
        )

    lookback_sec = max(2.0, min(20.0, float(lookback_seconds)))
    end_sec = max(1.0 / 60.0, float(max(1, int(end_tick))) / 60.0)
    start_sec = max(0.0, end_sec - lookback_sec)
    clip_dur = max(1.0 / 60.0, end_sec - start_sec)

    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    start_sample = max(0, int(round(start_sec * TARGET_AUDIO_HZ)))
    end_sample = max(start_sample + 160, int(round(end_sec * TARGET_AUDIO_HZ)))
    needed_samples = end_sample - start_sample

    raw_pcm = session_pcm_path.read_bytes() if session_pcm_path.is_file() else b""
    slice_pcm = raw_pcm[start_sample * 2 : end_sample * 2]
    actual_samples = len(slice_pcm) // 2
    if actual_samples < needed_samples:
        slice_pcm = slice_pcm + (b"\x00\x00" * (needed_samples - actual_samples))

    slice_wav_path = output_mp4_path.with_name(f".tmp_audio_{output_mp4_path.stem}.wav")
    tmp_mp4 = output_mp4_path.with_name(f".tmp_{output_mp4_path.name}")
    write_pcm16_to_wav(slice_pcm, slice_wav_path, sample_rate=TARGET_AUDIO_HZ)
    try:
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_bin,
            "-y",
            "-fflags", "+genpts",
            "-i", str(raw_game_mkv),
            "-i", str(slice_wav_path),
            "-filter_complex",
            f"[0:v]trim=start={start_sec:.3f}:end={end_sec:.3f},setpts=PTS-STARTPTS,scale=640:360,fps=15,format=yuv420p[vout]",
            "-map", "[vout]",
            "-map", "1:a:0",
            "-t", f"{clip_dur:.3f}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "24",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            str(tmp_mp4),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode != 0 or not tmp_mp4.is_file() or tmp_mp4.stat().st_size == 0:
            tmp_mp4.unlink(missing_ok=True)
            raise PlaytestRunnerError(
                f"ffmpeg recent gameplay clip extraction failed (exit {res.returncode}):\n{(res.stderr or '')[-1200:]}"
            )
        tmp_mp4.replace(output_mp4_path)
        return output_mp4_path
    finally:
        slice_wav_path.unlink(missing_ok=True)


class GodotPlaytestBridge:
    """Host controller managing a live, pauseable, silent background Godot 4.3 playtest process."""

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        session_dir: Path,
        reaction_ticks: int = 15,
        initial_settle_ticks: int = 30,
        record_video: bool = True,
        on_live_frame: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.session_dir = Path(session_dir).resolve()
        self.media_dir = self.session_dir / "media"
        self.reaction_ticks = max(1, int(reaction_ticks))
        self.initial_settle_ticks = max(2, int(initial_settle_ticks))
        self.record_video = bool(record_video)
        self.on_live_frame = on_live_frame

        self.runtime_dir: Optional[Path] = None
        self.user_home_dir: Optional[Path] = None
        self.raw_game_mkv: Path = self.media_dir / "raw_game_stream.mkv"
        self.session_pcm_path: Path = self.media_dir / "session_audio.pcm"

        self._ctrl_listener: Optional[socket.socket] = None
        self._ctrl_conn: Optional[socket.socket] = None
        self._godot_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._stdout_lines: List[str] = []
        self._stderr_lines: List[str] = []
        self._ffmpeg_stderr_lines: List[str] = []
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._ffmpeg_stderr_thread: Optional[threading.Thread] = None
        self._closed = False
        self.last_tick: int = 0

    def start(self, boot_timeout_sec: float = 45.0) -> TurnObservation:
        """Stages the isolated runtime, boots Godot + live H.264 stream capture, and returns Turn 0 observation."""
        if not RUNNER_SCRIPT_PATH.is_file():
            raise PlaytestRunnerError(f"Playtest GDScript runner missing at {RUNNER_SCRIPT_PATH}")

        sys_name = platform.system()
        has_display = (
            sys_name == "Darwin"
            or bool(os.environ.get("DISPLAY"))
            or bool(os.environ.get("WAYLAND_DISPLAY"))
        )
        xvfb_bin = shutil.which("xvfb-run") if (sys_name == "Linux" and not has_display) else None
        if not has_display and not xvfb_bin:
            raise PlaytestRunnerError(
                "Blind Playtester requires GPU/offscreen viewport rendering, "
                "but no DISPLAY/WAYLAND_DISPLAY or xvfb-run was found in this environment (Rule 5)."
            )

        if self.media_dir.exists():
            shutil.rmtree(self.media_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.session_pcm_path.write_bytes(b"")

        self.runtime_dir, self.user_home_dir = stage_isolated_checkpoint_runtime(
            self.checkpoint_dir, self.session_dir
        )

        # 1. Bind control TCP listener
        self._ctrl_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._ctrl_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._ctrl_listener.bind(("127.0.0.1", 0))
        self._ctrl_listener.listen(1)
        self._ctrl_listener.settimeout(boot_timeout_sec)
        ipc_port = int(self._ctrl_listener.getsockname()[1])

        # 2. Spawn live rawvideo -> H.264 MKV encoder if video recording is enabled
        video_port = 0
        if self.record_video:
            video_port = _reserve_ephemeral_port()
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            ff_cmd = [
                ffmpeg_bin,
                "-y",
                "-loglevel", "warning",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", "1280x720",
                "-r", "60",
                "-i", f"tcp://127.0.0.1:{video_port}?listen=1&listen_timeout={int(boot_timeout_sec * 1000)}",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "22",
                "-g", "30",
                "-pix_fmt", "yuv420p",
                "-flush_packets", "1",
                str(self.raw_game_mkv),
            ]
            self._ffmpeg_proc = subprocess.Popen(
                ff_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._ffmpeg_stderr_thread = threading.Thread(
                target=self._drain_ffmpeg_stderr_worker,
                name="FfmpegPlaytestStderr",
                daemon=True,
            )
            self._ffmpeg_stderr_thread.start()
            time.sleep(0.12)

        # 3. Launch silent background Godot 4.3 process
        godot_bin = ensure_godot_background_render_binary()
        silent_clock_avi = self.media_dir / "_silent_clock.avi"
        cmd: List[str] = []
        if xvfb_bin:
            cmd.extend([xvfb_bin, "-a"])
        cmd.append(str(godot_bin))
        if sys_name == "Darwin":
            cmd.extend(["-ApplePersistenceIgnoreState", "YES"])
        cmd.extend([
            "--rendering-method", "gl_compatibility",
            "--write-movie", str(silent_clock_avi),
            "--fixed-fps", "60",
            "--path", str(self.runtime_dir),
            "-s", str(RUNNER_SCRIPT_PATH),
            "--",
            "--ipc-port", str(ipc_port),
            "--video-port", str(video_port),
            "--capture-dir", str(self.media_dir),
            "--reaction-ticks", str(self.reaction_ticks),
            "--initial-settle-ticks", str(self.initial_settle_ticks),
        ])

        env = os.environ.copy()
        # Isolate Godot user:// data on both macOS (HOME/Library/...) and Linux (XDG_DATA_HOME)
        # while preserving X11 XAUTHORITY when present on Linux.
        orig_home = Path.home()
        if sys_name == "Linux" and "XAUTHORITY" not in env and (orig_home / ".Xauthority").is_file():
            env["XAUTHORITY"] = str(orig_home / ".Xauthority")
        env["HOME"] = str(self.user_home_dir)
        env["XDG_DATA_HOME"] = str(self.user_home_dir / "xdg_data")
        env["XDG_CONFIG_HOME"] = str(self.user_home_dir / "xdg_config")
        (self.user_home_dir / "xdg_data").mkdir(parents=True, exist_ok=True)
        (self.user_home_dir / "xdg_config").mkdir(parents=True, exist_ok=True)

        self._godot_proc = subprocess.Popen(
            cmd,
            cwd=str(self.runtime_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(
            target=self._drain_stdout_worker,
            name="GodotPlaytestStdout",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr_worker,
            name="GodotPlaytestStderr",
            daemon=True,
        )
        self._stderr_thread.start()

        try:
            conn, _ = self._ctrl_listener.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(boot_timeout_sec)
            self._ctrl_conn = conn

            ready_msg = self._recv_json(timeout_sec=boot_timeout_sec)
            if ready_msg.get("event") != "runner_ready":
                raise PlaytestRunnerError(f"Unexpected Godot handshake packet: {ready_msg}")

            return self._wait_for_observation(timeout_sec=boot_timeout_sec)
        except Exception as exc:
            stderr_tail = "\n".join(self._stderr_lines[-30:])
            ff_tail = "\n".join(self._ffmpeg_stderr_lines[-15:])
            diag_parts = []
            if stderr_tail:
                diag_parts.append(f"Godot stderr:\n{stderr_tail}")
            if ff_tail:
                diag_parts.append(f"ffmpeg stderr:\n{ff_tail}")
            diag_block = "\n".join(diag_parts)
            self.close()
            if isinstance(exc, PlaytestRunnerError):
                if diag_block and diag_block not in str(exc):
                    raise PlaytestRunnerError(f"{exc}\n{diag_block}") from exc
                raise
            raise PlaytestRunnerError(
                f"Failed booting Godot Blind Playtester runner: {exc}"
                + (f"\n{diag_block}" if diag_block else "")
            ) from exc

    def step_action(self, action: PlaytestMotorAction, timeout_sec: float = 30.0) -> TurnObservation:
        """Executes a motor burst in Godot, streams mid-burst live previews, and returns the post-burst observation."""
        if self._closed or self._ctrl_conn is None:
            raise PlaytestRunnerError("Cannot step playtest action: Godot runner is already closed.")
        payload = {
            "cmd": "step",
            "mouse_x": round(denormalize_screen_x(action.mouse_x), 2) if action.mouse_x is not None else None,
            "mouse_y": round(denormalize_screen_y(action.mouse_y), 2) if action.mouse_y is not None else None,
            "drag_from_x": round(denormalize_screen_x(action.drag_from_x), 2) if action.drag_from_x is not None else None,
            "drag_from_y": round(denormalize_screen_y(action.drag_from_y), 2) if action.drag_from_y is not None else None,
            "mouse_button_action": str(action.mouse_button_action),
            "hold_keys": list(action.hold_keys),
            "tap_keys": list(action.tap_keys),
            "duration_ms": int(action.duration_ms),
        }
        self._send_json(payload)
        return self._wait_for_observation(timeout_sec=timeout_sec)

    def close(self) -> None:
        """Cleanly stops Godot + ffmpeg video stream and flushes final audio samples."""
        if self._closed:
            return
        self._closed = True

        if self._ctrl_conn is not None:
            try:
                self._send_json({"cmd": "quit"})
                stop_msg = self._recv_json(timeout_sec=4.0)
                if stop_msg.get("event") == "runner_stopping":
                    self.last_tick = max(self.last_tick, int(stop_msg.get("tick", self.last_tick)))
                    tail_pcm = Path(str(stop_msg.get("tail_pcm_path", "")))
                    if tail_pcm.is_file():
                        with self.session_pcm_path.open("ab") as af:
                            af.write(tail_pcm.read_bytes())
                        tail_pcm.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                self._ctrl_conn.close()
            except Exception:
                pass
            self._ctrl_conn = None

        if self._ctrl_listener is not None:
            try:
                self._ctrl_listener.close()
            except Exception:
                pass
            self._ctrl_listener = None

        if self._godot_proc is not None:
            try:
                self._godot_proc.wait(timeout=6.0)
            except subprocess.TimeoutExpired:
                self._godot_proc.kill()
                try:
                    self._godot_proc.wait(timeout=3.0)
                except Exception:
                    pass
            self._godot_proc = None

        if self._ffmpeg_proc is not None:
            try:
                self._ffmpeg_proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._ffmpeg_proc.terminate()
                try:
                    self._ffmpeg_proc.wait(timeout=4.0)
                except Exception:
                    self._ffmpeg_proc.kill()
            self._ffmpeg_proc = None

        silent_clock_avi = self.media_dir / "_silent_clock.avi"
        if silent_clock_avi.exists():
            silent_clock_avi.unlink(missing_ok=True)

    def _drain_stdout_worker(self) -> None:
        proc = self._godot_proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                clean = line.rstrip("\r\n")
                if clean:
                    self._stdout_lines.append(clean)
                    if len(self._stdout_lines) > 100:
                        self._stdout_lines = self._stdout_lines[-100:]
        except Exception:
            pass

    def _drain_stderr_worker(self) -> None:
        proc = self._godot_proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                clean = line.rstrip("\r\n")
                if clean:
                    self._stderr_lines.append(clean)
                    if len(self._stderr_lines) > 200:
                        self._stderr_lines = self._stderr_lines[-200:]
        except Exception:
            pass

    def _drain_ffmpeg_stderr_worker(self) -> None:
        proc = self._ffmpeg_proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                clean = line.rstrip("\r\n")
                if clean:
                    self._ffmpeg_stderr_lines.append(clean)
                    if len(self._ffmpeg_stderr_lines) > 100:
                        self._ffmpeg_stderr_lines = self._ffmpeg_stderr_lines[-100:]
        except Exception:
            pass

    def _wait_for_observation(self, timeout_sec: float = 30.0) -> TurnObservation:
        while True:
            msg = self._recv_json(timeout_sec=timeout_sec)
            ev = msg.get("event")
            if ev == "live_frame":
                if self.on_live_frame is not None:
                    try:
                        self.on_live_frame(msg)
                    except Exception:
                        pass
                continue
            if ev == "observation_ready":
                obs_idx = int(msg.get("observation_index", 0))
                tick = int(msg.get("tick", 0))
                self.last_tick = max(self.last_tick, tick)
                png_path = Path(str(msg.get("png_path", "")))
                preview_path = Path(str(msg.get("preview_path", "")))
                pcm_path = Path(str(msg.get("pcm_path", "")))
                rms = float(msg.get("audio_rms", 0.0))
                peak = float(msg.get("audio_peak", 0.0))

                if not png_path.is_file():
                    raise PlaytestRunnerError(
                        f"Godot reported observation_ready at tick {tick} but screenshot '{png_path}' does not exist."
                    )

                pcm_bytes = pcm_path.read_bytes() if pcm_path.is_file() else b""
                if pcm_path.is_file():
                    pcm_path.unlink(missing_ok=True)
                if pcm_bytes:
                    with self.session_pcm_path.open("ab") as af:
                        af.write(pcm_bytes)

                is_silent = (len(pcm_bytes) == 0) or (rms < SILENCE_RMS_THRESHOLD)
                wav_path: Optional[Path] = None
                if pcm_bytes:
                    wav_candidate = self.media_dir / f"turn_{obs_idx:04d}.wav"
                    write_pcm16_to_wav(pcm_bytes, wav_candidate, sample_rate=TARGET_AUDIO_HZ)
                    wav_path = wav_candidate

                return TurnObservation(
                    observation_index=obs_idx,
                    tick=tick,
                    sim_time_sec=float(msg.get("sim_time_sec", round(tick / 60.0, 3))),
                    cursor_x=float(msg.get("cursor_x", 640.0)),
                    cursor_y=float(msg.get("cursor_y", 360.0)),
                    screenshot_path=png_path,
                    preview_path=preview_path if preview_path.is_file() else png_path,
                    wav_path=wav_path,
                    audio_rms=rms,
                    audio_peak=peak,
                    audio_is_silent=is_silent,
                    interrupted_early=bool(msg.get("interrupted_early", False)),
                )
            raise PlaytestRunnerError(f"Unexpected packet from Godot playtest runner: {msg}")

    def _send_json(self, obj: Dict[str, Any]) -> None:
        if self._ctrl_conn is None:
            raise PlaytestRunnerError("Control socket is disconnected.")
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        header = struct.pack(">I", len(raw))
        self._ctrl_conn.sendall(header + raw)

    def _recv_json(self, timeout_sec: float = 30.0) -> Dict[str, Any]:
        if self._ctrl_conn is None:
            raise PlaytestRunnerError("Control socket is disconnected.")
        self._ctrl_conn.settimeout(timeout_sec)
        hdr = self._recv_exact(4)
        if not hdr:
            stderr_tail = "\n".join(self._stderr_lines[-20:])
            raise PlaytestRunnerError(
                f"Godot runner closed control connection unexpectedly.\nGodot stderr:\n{stderr_tail}"
            )
        (msg_len,) = struct.unpack(">I", hdr)
        if msg_len <= 0 or msg_len > 2_097_152:
            raise PlaytestRunnerError(f"Invalid control frame length from Godot: {msg_len}")
        body = self._recv_exact(msg_len)
        if len(body) < msg_len:
            raise PlaytestRunnerError("Truncated JSON payload from Godot runner.")
        return json.loads(body.decode("utf-8"))

    def _recv_exact(self, n: int) -> bytes:
        assert self._ctrl_conn is not None
        chunks: List[bytes] = []
        rem = n
        while rem > 0:
            chunk = self._ctrl_conn.recv(rem)
            if not chunk:
                break
            chunks.append(chunk)
            rem -= len(chunk)
        return b"".join(chunks)
