"""engine/playtest/session_manager.py
Decoupled Blind Playtester Session Lifecycle Manager & Council Wish Dispatcher.

Runs completely independently of `studio.lock`, `.studio_paused.json`, and `StudioDaemon`.
Supports Start, Pause (with live interim MP4 muxing), Resume, Stop, and Reset,
persisting all turns, compactions, and wishes to SQLite WAL (`TelemetryDB`) and
streaming live observability events over `SwarmEventBus`.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from engine.playtest.agent import BlindPlaytesterAgent, extract_blind_store_card
from engine.playtest.godot_bridge import (
    TARGET_AUDIO_HZ,
    GodotPlaytestBridge,
    SidebarSegment,
    TurnObservation,
    mux_broadcast_replay_mp4,
    render_turn_sidebar_png,
)
from engine.playtest.models import (
    IMPRESSION_PILLAR_LABELS,
    BlindStoreCard,
    PlaytestBudgetExceededError,
    PlaytestExperienceReport,
    PlaytestMotorAction,
    PlaytestRunnerError,
    PlaytestSessionConfig,
    PlaytestSessionStartRequest,
    PlaytestTokenTracker,
    denormalize_screen_x,
    denormalize_screen_y,
)
from engine.studio.models import StudioAgentError
from engine.telemetry.db import TelemetryDB

logger = logging.getLogger("PlaytestSessionManager")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_motor_summary(act: PlaytestMotorAction) -> str:
    parts: List[str] = []
    if act.hold_keys:
        parts.append(f"Hold [{'+'.join(act.hold_keys)}]")
    if act.tap_keys:
        parts.append(f"Tap [{'+'.join(act.tap_keys)}]")
    if act.mouse_button_action == "drag_left" and (act.drag_from_x is not None or act.drag_from_y is not None):
        fx = f"{denormalize_screen_x(act.drag_from_x):.0f}" if act.drag_from_x is not None else "cur"
        fy = f"{denormalize_screen_y(act.drag_from_y):.0f}" if act.drag_from_y is not None else "cur"
        tx = f"{denormalize_screen_x(act.mouse_x):.0f}" if act.mouse_x is not None else "cur"
        ty = f"{denormalize_screen_y(act.mouse_y):.0f}" if act.mouse_y is not None else "cur"
        parts.append(f"Drag ({fx},{fy})→({tx},{ty})")
    elif act.mouse_x is not None or act.mouse_y is not None:
        tx = f"{denormalize_screen_x(act.mouse_x):.0f}" if act.mouse_x is not None else "cur"
        ty = f"{denormalize_screen_y(act.mouse_y):.0f}" if act.mouse_y is not None else "cur"
        parts.append(f"Move ({tx},{ty})")
    if act.mouse_button_action != "none":
        parts.append(act.mouse_button_action.replace("_", " ").title())
    parts.append(f"({act.duration_ms}ms)")
    return " • ".join(parts)


class _RunningSessionHandle:
    """Thread-safe runtime state for an active or paused playtest session."""

    def __init__(
        self,
        *,
        session_id: str,
        checkpoint_dir: Path,
        store_card: BlindStoreCard,
        config: PlaytestSessionConfig,
        session_dir: Path,
        initiated_by: str = "operator",
    ) -> None:
        self.session_id = session_id
        self.checkpoint_dir = checkpoint_dir
        self.store_card = store_card
        self.config = config
        self.session_dir = session_dir
        self.initiated_by = initiated_by if initiated_by in ("operator", "council") else "operator"
        self.pause_gate = threading.Event()
        self.pause_gate.set()  # Unpaused initially
        self.stop_event = threading.Event()
        self.discard_on_stop = False
        self.thread: Optional[threading.Thread] = None
        self.agent: Optional[BlindPlaytesterAgent] = None
        self.bridge: Optional[GodotPlaytestBridge] = None


class PlaytestSessionManager:
    """Manages Blind Playtester background sessions, media artifacts, and Council wish dispatch."""

    def __init__(
        self,
        *,
        db_path: Path,
        cartridges_root: Path,
        sessions_root: Optional[Path] = None,
        event_bus: Optional[Any] = None,
        client_factory: Optional[Callable[[], Any]] = None,
        bridge_factory: Optional[Callable[..., GodotPlaytestBridge]] = None,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.cartridges_root = Path(cartridges_root).resolve()
        self.sessions_root = (
            Path(sessions_root).resolve()
            if sessions_root is not None
            else (self.db_path.parent / "playtest_sessions").resolve()
        )
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.event_bus = event_bus
        self.client_factory = client_factory
        self.bridge_factory = bridge_factory

        self._lock = threading.Lock()
        self._mux_recovery_lock = threading.Lock()
        self._handles: Dict[str, _RunningSessionHandle] = {}
        self._reconcile_orphaned_sessions()

    def _reconcile_orphaned_sessions(self) -> None:
        """Marks any stale 'running' or 'paused' SQLite rows from a prior server process as 'stopped',
        and recovers/muxes `replay.mp4` for any finished session whose raw stream exists on disk.
        """
        try:
            db = self._db()
            for row in db.list_playtest_sessions(limit=50):
                sid = str(row.get("session_id") or "")
                if not sid:
                    continue
                with self._lock:
                    is_live = sid in self._handles
                if is_live:
                    continue
                if row.get("status") in ("running", "paused"):
                    row["status"] = "stopped"
                    row["end_outcome"] = row.get("end_outcome") or "server_reloaded"
                    row["ended_at_utc"] = row.get("ended_at_utc") or _utc_now()
                    row["updated_at_utc"] = _utc_now()
                    row["deliberation"] = f"Session {sid} closed cleanly following Mission Control reload."
                    db.upsert_playtest_session(row)
                if row.get("status") in ("stopped", "completed", "error"):
                    self._recover_unmuxed_session_video(row, db)
        except Exception as exc:
            logger.debug("Orphaned playtest session reconciliation skipped: %s", exc)

    def _recover_unmuxed_session_video(
        self,
        session_row: Dict[str, Any],
        db: TelemetryDB,
    ) -> Optional[str]:
        """Deterministically recovers and muxes `replay.mp4` for an inactive session whose
        `raw_game_stream.mkv` exists on disk (e.g., interrupted by server reload or mid-turn stop).
        """
        sid = str(session_row.get("session_id") or "")
        if not sid or ".." in sid or "/" in sid or "\\" in sid or "\0" in sid:
            return None
        with self._lock:
            if sid in self._handles:
                return None
        if session_row.get("status") not in ("stopped", "completed", "error"):
            return None

        media_dir = (self.sessions_root / sid / "media").resolve()
        try:
            media_dir.relative_to(self.sessions_root)
        except ValueError:
            return None

        raw_mkv = media_dir / "raw_game_stream.mkv"
        replay_mp4 = media_dir / "replay.mp4"
        recorded_vp = session_row.get("video_path")

        # Fast path: valid up-to-date replay.mp4 already persisted
        if (
            recorded_vp
            and Path(str(recorded_vp)).is_file()
            and Path(str(recorded_vp)).stat().st_size > 0
            and (not raw_mkv.is_file() or Path(str(recorded_vp)).stat().st_mtime >= raw_mkv.stat().st_mtime)
        ):
            return str(recorded_vp)

        if not raw_mkv.is_file() or raw_mkv.stat().st_size == 0:
            if replay_mp4.is_file() and replay_mp4.stat().st_size > 0:
                session_row["video_path"] = str(replay_mp4)
                session_row["updated_at_utc"] = _utc_now()
                db.upsert_playtest_session(session_row)
                return str(replay_mp4)
            return None

        with self._mux_recovery_lock:
            with self._lock:
                if sid in self._handles:
                    return None
            fresh = db.get_playtest_session(sid) or session_row
            fresh_vp = fresh.get("video_path")
            if (
                fresh_vp
                and Path(str(fresh_vp)).is_file()
                and Path(str(fresh_vp)).stat().st_size > 0
                and Path(str(fresh_vp)).stat().st_mtime >= raw_mkv.stat().st_mtime
            ):
                session_row["video_path"] = str(fresh_vp)
                return str(fresh_vp)

            try:
                session_pcm = media_dir / "session_audio.pcm"
                tail_pcm = media_dir / "final_tail.pcm"
                if tail_pcm.is_file():
                    tail_bytes = tail_pcm.read_bytes()
                    if tail_bytes:
                        with session_pcm.open("ab") as af:
                            af.write(tail_bytes)
                    tail_pcm.unlink(missing_ok=True)
                (media_dir / "_silent_clock.avi").unlink(missing_ok=True)

                turns = db.list_playtest_turns(sid)
                pcm_ticks = (
                    int(round(((session_pcm.stat().st_size // 2) / float(TARGET_AUDIO_HZ)) * 60.0))
                    if session_pcm.is_file()
                    else 0
                )
                total_ticks = max(
                    int(fresh.get("sim_ticks_total") or 0),
                    max((int(t.get("sim_tick_end") or 0) for t in turns), default=0),
                    pcm_ticks,
                    1,
                )

                sidebar_segments: List[SidebarSegment] = []
                boot_png = media_dir / "sidebar_turn_0000.png"
                if boot_png.is_file():
                    first_start = int(turns[0].get("sim_tick_start") or 30) if turns else total_ticks
                    sidebar_segments.append(
                        SidebarSegment(
                            turn_index=0,
                            start_tick=0,
                            end_tick=max(1, first_start),
                            sidebar_png_path=boot_png,
                        )
                    )

                for tr in turns:
                    t_idx = int(tr.get("turn_index") or 0)
                    if t_idx <= 0:
                        continue
                    t_png = media_dir / f"sidebar_turn_{t_idx:04d}.png"
                    if not t_png.is_file():
                        continue
                    s_tick = int(tr.get("sim_tick_start") or 0)
                    e_tick = max(s_tick + 1, int(tr.get("sim_tick_end") or (s_tick + 1)))
                    sidebar_segments.append(
                        SidebarSegment(
                            turn_index=t_idx,
                            start_tick=s_tick,
                            end_tick=e_tick,
                            sidebar_png_path=t_png,
                        )
                    )

                if not sidebar_segments:
                    disk_pngs = sorted(media_dir.glob("sidebar_turn_*.png"))
                    if disk_pngs:
                        step = max(1, total_ticks // len(disk_pngs))
                        for idx, png_p in enumerate(disk_pngs):
                            st = idx * step
                            et = total_ticks if idx == len(disk_pngs) - 1 else max(st + 1, (idx + 1) * step)
                            sidebar_segments.append(
                                SidebarSegment(turn_index=idx, start_tick=st, end_tick=et, sidebar_png_path=png_p)
                            )
                    else:
                        fallback_png = render_turn_sidebar_png(
                            boot_png,
                            game_title=str(fresh.get("game_title") or "Untitled"),
                            turn_index=int(fresh.get("turn_count") or 0),
                            sim_time_sec=round(total_ticks / 60.0, 2),
                            current_mood=str(fresh.get("current_mood") or "observing"),
                            current_plan=str(fresh.get("current_plan") or "Playtest session stopped."),
                            sticky_feedback_items=[],
                            action_summary="Session stopped",
                            cursor_xy=(640.0, 360.0),
                        )
                        sidebar_segments.append(
                            SidebarSegment(
                                turn_index=0,
                                start_tick=0,
                                end_tick=total_ticks,
                                sidebar_png_path=fallback_png,
                            )
                        )

                sidebar_segments[-1].end_tick = max(sidebar_segments[-1].end_tick, total_ticks)
                muxed = mux_broadcast_replay_mp4(
                    raw_game_mkv=raw_mkv,
                    session_pcm_path=session_pcm,
                    total_sim_ticks=total_ticks,
                    sidebar_segments=sidebar_segments,
                    output_mp4_path=replay_mp4,
                )
                muxed_str = str(muxed)
                fresh["video_path"] = muxed_str
                fresh["sim_ticks_total"] = max(int(fresh.get("sim_ticks_total") or 0), total_ticks)
                fresh["sim_duration_sec"] = round(fresh["sim_ticks_total"] / 60.0, 2)
                fresh["updated_at_utc"] = _utc_now()
                db.upsert_playtest_session(fresh)
                session_row["video_path"] = muxed_str
                self._emit_event(
                    "playtest_video_ready",
                    f"Recovered and muxed broadcast replay.mp4 for Blind Playtester session {sid}.",
                    {"session_id": sid, "has_video": True},
                )
                return muxed_str
            except Exception as exc:
                logger.warning("Could not recover unmuxed replay video for %s: %s", sid, exc)
                return None

    def _db(self) -> TelemetryDB:
        return TelemetryDB(self.db_path)

    def _emit_event(self, event_type: str, deliberation: str, payload: Dict[str, Any]) -> None:
        if self.event_bus is None or not hasattr(self.event_bus, "publish_threadsafe"):
            return
        try:
            self.event_bus.publish_threadsafe(
                {
                    "deliberation": deliberation,
                    "event_type": event_type,
                    "agent_name": "BlindPlaytesterAgent",
                    "timestamp_utc": _utc_now(),
                    "payload": payload,
                }
            )
        except Exception as exc:
            logger.debug("Playtest event publish skipped (%s): %s", event_type, exc)

    def resolve_playtest_target(
        self,
        *,
        checkpoint_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        slice_id: Optional[str] = None,
    ) -> Tuple[Path, str, Optional[str], Optional[str]]:
        """Resolves `(checkpoint_dir, resolved_checkpoint_id, resolved_candidate_id, resolved_slice_id)`
        with strict path-traversal containment inside `self.cartridges_root`.
        """
        for label, val in (("checkpoint_id", checkpoint_id), ("candidate_id", candidate_id), ("slice_id", slice_id)):
            if val and (".." in val or "/" in val or "\\" in val or "\0" in val):
                raise StudioAgentError(f"Invalid {label} '{val}': path traversal characters are forbidden.")

        checkpoints_dir = self.cartridges_root / "checkpoints"

        # 1. Explicit checkpoint_id
        if checkpoint_id:
            if checkpoint_id == "active":
                act = (self.cartridges_root / "active").resolve()
                if (act / "project.godot").is_file():
                    return act, "active", candidate_id, slice_id
                raise StudioAgentError(f"Active cartridge directory '{act}' has no playable Godot project.godot.")
            cp_candidate = (checkpoints_dir / checkpoint_id).resolve()
            cp_candidate.relative_to(self.cartridges_root)
            if cp_candidate.is_dir() and (cp_candidate / "project.godot").is_file():
                return cp_candidate, checkpoint_id, candidate_id, slice_id
            direct_cand = (self.cartridges_root / checkpoint_id).resolve()
            direct_cand.relative_to(self.cartridges_root)
            if direct_cand.is_dir() and (direct_cand / "project.godot").is_file():
                return direct_cand, checkpoint_id, candidate_id or checkpoint_id, slice_id
            raise StudioAgentError(f"Checkpoint '{checkpoint_id}' not found or missing project.godot.")

        # 2. Resolve from completed slice_id (and optional candidate_id)
        resolved_cand_dir: Optional[Path] = None
        if candidate_id:
            cand_p = (self.cartridges_root / candidate_id).resolve()
            cand_p.relative_to(self.cartridges_root)
            if cand_p.is_dir():
                resolved_cand_dir = cand_p
        elif slice_id and self.cartridges_root.is_dir():
            for d in sorted(
                [p for p in self.cartridges_root.glob("candidate_*") if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                gb_file = d / "game_bible.json"
                if gb_file.is_file():
                    try:
                        gb = json.loads(gb_file.read_text(encoding="utf-8"))
                        slices = gb.get("slice_roadmap") or gb.get("slices") or []
                        if any(isinstance(s, dict) and s.get("slice_id") == slice_id for s in slices):
                            resolved_cand_dir = d
                            candidate_id = d.name
                            break
                    except Exception:
                        continue

        if slice_id or candidate_id:
            target_item_ids: set[str] = set()
            if resolved_cand_dir is not None and slice_id:
                pb_file = resolved_cand_dir / "production_board.json"
                if pb_file.is_file():
                    try:
                        pb = json.loads(pb_file.read_text(encoding="utf-8"))
                        for it in pb.get("items") or []:
                            if isinstance(it, dict) and it.get("slice_id") == slice_id and it.get("item_id"):
                                target_item_ids.add(str(it["item_id"]))
                    except Exception:
                        pass

            # Search verified checkpoints matching candidate_id
            matching_cps: List[Tuple[float, Path, Dict[str, Any]]] = []
            if checkpoints_dir.is_dir():
                for cp_d in checkpoints_dir.iterdir():
                    if not cp_d.is_dir() or not (cp_d / "project.godot").is_file():
                        continue
                    meta_f = cp_d / "checkpoint_metadata.json"
                    meta: Dict[str, Any] = {}
                    if meta_f.is_file():
                        try:
                            meta = json.loads(meta_f.read_text(encoding="utf-8"))
                        except Exception:
                            meta = {}
                    cp_cand = str(meta.get("candidate_id") or "")
                    if candidate_id and cp_cand != candidate_id and not cp_d.name.startswith(f"{candidate_id}_"):
                        continue
                    matching_cps.append((cp_d.stat().st_mtime, cp_d, meta))

            matching_cps.sort(key=lambda item: item[0], reverse=True)
            if target_item_ids and matching_cps:
                for _, cp_d, meta in matching_cps:
                    req_id = str(meta.get("requirement_id") or "")
                    v_tracks = set(str(t) for t in (meta.get("verified_tracks") or []))
                    if req_id in target_item_ids or (target_item_ids & v_tracks):
                        return cp_d, cp_d.name, candidate_id, slice_id
            if matching_cps:
                best_cp = matching_cps[0][1]
                return best_cp, best_cp.name, candidate_id, slice_id
            if resolved_cand_dir is not None and (resolved_cand_dir / "project.godot").is_file():
                return resolved_cand_dir, resolved_cand_dir.name, candidate_id or resolved_cand_dir.name, slice_id

        # 3. Default to active cartridge or newest checkpoint
        act_dir = (self.cartridges_root / "active").resolve()
        if act_dir.is_dir() and (act_dir / "project.godot").is_file():
            return act_dir, "active", candidate_id, slice_id
        if checkpoints_dir.is_dir():
            valid_cps = sorted(
                [p for p in checkpoints_dir.iterdir() if p.is_dir() and (p / "project.godot").is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if valid_cps:
                return valid_cps[0], valid_cps[0].name, candidate_id, slice_id

        raise StudioAgentError("No playable Godot checkpoint or active slice found to playtest.")

    def list_playable_targets(self) -> List[Dict[str, Any]]:
        """Lists completed slices and verified checkpoints available for Blind Playtesting."""
        targets: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        act_dir = self.cartridges_root / "active"
        if act_dir.is_dir() and (act_dir / "project.godot").is_file():
            try:
                card = extract_blind_store_card(act_dir, checkpoint_id="active")
                targets.append({
                    "target_type": "active",
                    "checkpoint_id": "active",
                    "candidate_id": card.candidate_id,
                    "slice_id": card.slice_id,
                    "label": f"Active Cartridge — {card.game_title}",
                    "game_title": card.game_title,
                    "game_description": card.game_description,
                })
                seen_ids.add("active")
            except Exception:
                pass

        # Discover completed slices across candidates
        if self.cartridges_root.is_dir():
            for cand_d in sorted(
                [p for p in self.cartridges_root.glob("candidate_*") if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                gb_file = cand_d / "game_bible.json"
                if not gb_file.is_file():
                    continue
                try:
                    gb = json.loads(gb_file.read_text(encoding="utf-8"))
                    g_title = str(gb.get("game_title") or gb.get("title") or cand_d.name)
                    for s in (gb.get("slice_roadmap") or gb.get("slices") or []):
                        if not isinstance(s, dict):
                            continue
                        if str(s.get("status") or "").lower() == "completed":
                            sid = str(s.get("slice_id") or "")
                            if not sid:
                                continue
                            try:
                                cp_dir, cp_id, _, _ = self.resolve_playtest_target(
                                    candidate_id=cand_d.name, slice_id=sid
                                )
                                card = extract_blind_store_card(
                                    cp_dir, checkpoint_id=cp_id, candidate_id=cand_d.name, slice_id=sid
                                )
                                targets.append({
                                    "target_type": "completed_slice",
                                    "checkpoint_id": cp_id,
                                    "candidate_id": cand_d.name,
                                    "slice_id": sid,
                                    "label": f"{g_title} — Slice #{s.get('slice_index', '?')}: {s.get('title', sid)}",
                                    "game_title": card.game_title,
                                    "game_description": card.game_description,
                                })
                                seen_ids.add(cp_id)
                            except Exception:
                                continue
                except Exception:
                    continue

        checkpoints_dir = self.cartridges_root / "checkpoints"
        if checkpoints_dir.is_dir():
            for cp_d in sorted(
                [p for p in checkpoints_dir.iterdir() if p.is_dir() and (p / "project.godot").is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                if cp_d.name in seen_ids:
                    continue
                try:
                    card = extract_blind_store_card(cp_d, checkpoint_id=cp_d.name)
                    targets.append({
                        "target_type": "checkpoint",
                        "checkpoint_id": cp_d.name,
                        "candidate_id": card.candidate_id,
                        "slice_id": card.slice_id,
                        "label": f"{card.game_title} ({cp_d.name})",
                        "game_title": card.game_title,
                        "game_description": card.game_description,
                    })
                except Exception:
                    continue

        return targets

    def _wipe_session_disk_artifacts(self, session_id: str, *, keep_config: bool = True) -> None:
        """Safely deletes captured media, staged runtime, and Godot user:// state for `session_id`."""
        if not session_id or ".." in session_id or "/" in session_id or "\\" in session_id or "\0" in session_id:
            return
        s_dir = (self.sessions_root / session_id).resolve()
        try:
            s_dir.relative_to(self.sessions_root)
        except ValueError:
            return
        if not s_dir.is_dir():
            return
        if not keep_config:
            shutil.rmtree(s_dir, ignore_errors=True)
            return
        for child in s_dir.iterdir():
            if child.name == "session_config.json":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink(missing_ok=True)
                except OSError:
                    pass

    def start_session(self, req: PlaytestSessionStartRequest) -> Dict[str, Any]:
        """Stops any currently active playtester session and starts a fresh opaque-box session."""
        # Stop any currently active/paused handle so only one background Godot runner runs at a time
        self._stop_all_active_handles()

        cp_dir, resolved_cp_id, resolved_cand_id, resolved_slice_id = self.resolve_playtest_target(
            checkpoint_id=req.checkpoint_id,
            candidate_id=req.candidate_id,
            slice_id=req.slice_id,
        )
        store_card = extract_blind_store_card(
            cp_dir,
            checkpoint_id=resolved_cp_id,
            candidate_id=resolved_cand_id,
            slice_id=resolved_slice_id,
            title_override=req.store_title_override,
            description_override=req.store_description_override,
        )
        cfg = PlaytestSessionConfig(
            reaction_ticks=req.reaction_ticks,
            compaction_interval_turns=req.compaction_interval_turns,
            max_turns=req.max_turns,
            enable_audio_perception=req.enable_audio_perception,
        )

        # Clean up any empty `status='idle'` reset placeholder rows for the same target before starting
        db = self._db()
        for old_row in db.list_playtest_sessions(
            checkpoint_id=store_card.checkpoint_id,
            candidate_id=store_card.candidate_id,
            limit=30,
        ):
            old_sid = str(old_row.get("session_id") or "")
            if (
                old_sid
                and str(old_row.get("status") or "").lower() == "idle"
                and int(old_row.get("turn_count", 0) or 0) == 0
                and old_row.get("slice_id") == store_card.slice_id
            ):
                db.delete_playtest_session(old_sid)
                self._wipe_session_disk_artifacts(old_sid, keep_config=False)

        session_id = f"pt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        session_dir = (self.sessions_root / session_id).resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
        # Persist session config for Reset fidelity
        (session_dir / "session_config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

        initiated_by = req.initiated_by if req.initiated_by in ("operator", "council") else "operator"
        now_iso = _utc_now()
        init_record = {
            "session_id": session_id,
            "checkpoint_id": store_card.checkpoint_id,
            "candidate_id": store_card.candidate_id,
            "slice_id": store_card.slice_id,
            "initiated_by": initiated_by,
            "game_title": store_card.game_title,
            "game_description": store_card.game_description,
            "status": "running",
            "turn_count": 0,
            "sim_ticks_total": 0,
            "sim_duration_sec": 0.0,
            "compaction_count": 0,
            "rulebook": {},
            "impressions": {},
            "session_report": {},
            "compacted_journal": {},
            "current_plan": "Booting game and observing opening screen.",
            "current_mood": "curious",
            "started_at_utc": now_iso,
            "updated_at_utc": now_iso,
            "deliberation": req.deliberation,
        }
        db = self._db()
        db.upsert_playtest_session(init_record)

        handle = _RunningSessionHandle(
            session_id=session_id,
            checkpoint_dir=cp_dir,
            store_card=store_card,
            config=cfg,
            session_dir=session_dir,
            initiated_by=initiated_by,
        )
        worker = threading.Thread(
            target=self._run_session_worker,
            args=(handle,),
            name=f"BlindPlaytester-{session_id}",
            daemon=True,
        )
        handle.thread = worker
        with self._lock:
            self._handles[session_id] = handle
        worker.start()

        self._emit_event(
            "playtest_session_status",
            f"Started Blind Playtester session {session_id} for '{store_card.game_title}' (initiated_by={initiated_by}).",
            {
                "session_id": session_id,
                "status": "running",
                "initiated_by": initiated_by,
                "game_title": store_card.game_title,
            },
        )
        return db.get_playtest_session(session_id) or init_record

    def run_council_slice_playtest(
        self,
        *,
        candidate_dir: Path,
        slice_id: str,
        game_title: str,
        store_description: str,
        max_turns: int = 60,
        reaction_ticks: int = 15,
        compaction_interval_turns: int = 30,
        enable_audio_perception: bool = True,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Synchronously runs an end-of-slice Council-initiated Blind Playtester session on `candidate_dir`,
        streaming live frames and wishes to Mission Control UI, automatically synthesizing the end-of-session
        `PlaytestExperienceReport` (even if stopped early by operator or max_turns), and auto-delivering all
        wishes + the experience report back to the calling Council.
        """
        cand_resolved = Path(candidate_dir).resolve()
        if not (cand_resolved / "project.godot").is_file():
            raise StudioAgentError(
                f"Cannot run Council Blind Playtest: '{cand_resolved}' has no playable project.godot (Rule 5)."
            )
        req = PlaytestSessionStartRequest(
            deliberation=(
                f"Council initiated automatic end-of-slice Blind Playtest for slice '{slice_id}' "
                f"on candidate '{cand_resolved.name}'."
            ),
            checkpoint_id=cand_resolved.name,
            candidate_id=cand_resolved.name,
            slice_id=slice_id,
            initiated_by="council",
            store_title_override=game_title,
            store_description_override=store_description,
            max_turns=max_turns,
            reaction_ticks=reaction_ticks,
            compaction_interval_turns=compaction_interval_turns,
            enable_audio_perception=enable_audio_perception,
        )
        started = self.start_session(req)
        session_id = str(started["session_id"])

        with self._lock:
            handle = self._handles.get(session_id)
        if handle is not None and handle.thread is not None:
            while handle.thread.is_alive():
                handle.thread.join(timeout=0.5)
                if abort_check is not None:
                    try:
                        should_abort = bool(abort_check())
                    except Exception:
                        should_abort = False
                    if should_abort and not handle.stop_event.is_set():
                        handle.stop_event.set()
                        handle.pause_gate.set()

        detail = self.get_session_detail(session_id)
        if detail is None:
            raise StudioAgentError(
                f"Council Blind Playtest session '{session_id}' vanished from telemetry database (Rule 5)."
            )
        sess_row = detail.get("session") or {}
        if sess_row.get("status") == "error":
            err_detail = sess_row.get("error_message") or "Unknown Blind Playtester runtime failure."
            raise StudioAgentError(f"Council Blind Playtest failed for slice '{slice_id}': {err_detail}")
        return detail

    def control_session(self, session_id: str, action: str) -> Dict[str, Any]:
        """Executes `pause`, `resume`, `stop`, or `reset` on a playtest session."""
        db = self._db()
        existing = db.get_playtest_session(session_id)
        if existing is None:
            raise StudioAgentError(f"Playtest session '{session_id}' not found.")

        with self._lock:
            handle = self._handles.get(session_id)

        act = action.strip().lower()
        if act == "pause":
            if handle is not None and not handle.stop_event.is_set():
                handle.pause_gate.clear()
                existing["status"] = "paused"
                existing["updated_at_utc"] = _utc_now()
                existing["deliberation"] = f"Operator paused Blind Playtester session {session_id}."
                db.upsert_playtest_session(existing)
                self._emit_event(
                    "playtest_session_status",
                    existing["deliberation"],
                    {"session_id": session_id, "status": "paused"},
                )
            return db.get_playtest_session(session_id) or existing

        if act == "resume":
            if handle is not None and not handle.stop_event.is_set():
                handle.pause_gate.set()
                existing["status"] = "running"
                existing["updated_at_utc"] = _utc_now()
                existing["deliberation"] = f"Operator resumed Blind Playtester session {session_id}."
                db.upsert_playtest_session(existing)
                self._emit_event(
                    "playtest_session_status",
                    existing["deliberation"],
                    {"session_id": session_id, "status": "running"},
                )
            return db.get_playtest_session(session_id) or existing

        if act == "stop":
            if handle is not None:
                handle.stop_event.set()
                handle.pause_gate.set()
                if handle.thread is not None and handle.thread.is_alive():
                    handle.thread.join(timeout=45.0)
            else:
                if existing.get("status") in ("running", "paused"):
                    existing["status"] = "stopped"
                    existing["ended_at_utc"] = _utc_now()
                    existing["updated_at_utc"] = _utc_now()
                    existing["deliberation"] = f"Operator stopped Blind Playtester session {session_id}."
                    db.upsert_playtest_session(existing)
            latest = db.get_playtest_session(session_id) or existing
            with self._lock:
                still_live = session_id in self._handles
            if not still_live:
                self._recover_unmuxed_session_video(latest, db)
                latest = db.get_playtest_session(session_id) or latest
            latest["has_video"] = bool(latest.get("video_path") and Path(str(latest["video_path"])).is_file())
            return latest

        if act == "reset":
            # Only a RUNNING playtester resets and starts over automatically.
            # A stopped / completed / error / idle / paused playtester resets all session data to 'idle' WITHOUT starting.
            was_running = bool(
                handle is not None
                and not handle.stop_event.is_set()
                and handle.pause_gate.is_set()
                and str(existing.get("status") or "").lower() == "running"
            )
            if handle is not None:
                handle.discard_on_stop = True
                handle.stop_event.set()
                handle.pause_gate.set()
                if handle.thread is not None and handle.thread.is_alive():
                    handle.thread.join(timeout=45.0)

            cfg = PlaytestSessionConfig()
            cfg_file = (self.sessions_root / session_id / "session_config.json")
            if cfg_file.is_file():
                try:
                    cfg = PlaytestSessionConfig.model_validate_json(cfg_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            if not was_running:
                self._wipe_session_disk_artifacts(session_id, keep_config=True)
                reset_delib = (
                    f"Operator reset stopped playtest session {session_id}: cleared turns, wishes, "
                    f"impressions, report, media, and user:// state without starting."
                )
                reset_row = db.reset_playtest_session_data(session_id, deliberation=reset_delib) or existing
                reset_row["has_video"] = False
                reset_row["latest_frame_name"] = None
                self._emit_event(
                    "playtest_session_status",
                    reset_delib,
                    {
                        "session_id": session_id,
                        "initiated_by": reset_row.get("initiated_by", "operator"),
                        "status": "idle",
                        "has_video": False,
                        "has_report": False,
                    },
                )
                return reset_row

            # Session was actively running -> wipe aborted run data and immediately start over from Turn 0
            db.delete_playtest_session(session_id)
            self._wipe_session_disk_artifacts(session_id, keep_config=False)

            prev_initiated_by = str(existing.get("initiated_by") or "operator")
            restart_req = PlaytestSessionStartRequest(
                deliberation=f"Operator reset running playtester from {session_id} to start over from Turn 0 with clean memory and user:// state.",
                checkpoint_id=existing.get("checkpoint_id"),
                candidate_id=existing.get("candidate_id"),
                slice_id=existing.get("slice_id"),
                initiated_by="council" if prev_initiated_by == "council" else "operator",
                max_turns=cfg.max_turns,
                reaction_ticks=cfg.reaction_ticks,
                compaction_interval_turns=cfg.compaction_interval_turns,
                enable_audio_perception=cfg.enable_audio_perception,
            )
            return self.start_session(restart_req)

        raise StudioAgentError(f"Unsupported session control action: '{action}'")

    def get_session_detail(self, session_id: str) -> Optional[Dict[str, Any]]:
        db = self._db()
        sess = db.get_playtest_session(session_id)
        if sess is None:
            return None
        turns = db.list_playtest_turns(session_id)
        compactions = db.list_playtest_compactions(session_id)
        wishes = db.list_playtest_wishes(session_id=session_id, limit=200)

        p_tok = int(sess.get("prompt_tokens", 0) or 0)
        c_tok = int(sess.get("cached_tokens", 0) or 0)
        o_tok = int(sess.get("output_tokens", 0) or 0)
        t_tok = int(sess.get("thought_tokens", 0) or 0)
        cache_pct = round(100.0 * c_tok / p_tok, 1) if p_tok > 0 else 0.0

        with self._lock:
            live_handle = self._handles.get(session_id)
        epoch_preview = (
            live_handle.agent.epoch_header_text
            if (live_handle is not None and live_handle.agent is not None)
            else None
        )

        # Recover replay.mp4 lazily if session is finished/stopped on disk without muxed MP4
        if live_handle is None and sess.get("status") in ("stopped", "completed", "error"):
            self._recover_unmuxed_session_video(sess, db)

        # Convert filesystem paths into session-relative media filenames for clean UI URLs
        sess["has_video"] = bool(sess.get("video_path") and Path(str(sess["video_path"])).is_file())
        sess["latest_frame_name"] = (
            Path(str(sess["latest_frame_path"])).name if sess.get("latest_frame_path") else None
        )
        for tr in turns:
            tr["screenshot_name"] = Path(str(tr["screenshot_path"])).name if tr.get("screenshot_path") else None
            tr["audio_clip_name"] = Path(str(tr["audio_clip_path"])).name if tr.get("audio_clip_path") else None
        for w in wishes:
            w["screenshot_name"] = Path(str(w["screenshot_path"])).name if w.get("screenshot_path") else None

        return {
            "deliberation": f"Loaded observability snapshot for Blind Playtester session {session_id}.",
            "session": sess,
            "turns": turns,
            "compactions": compactions,
            "wishes": wishes,
            "epoch_header_preview": epoch_preview,
            "token_summary": {
                "prompt_tokens": p_tok,
                "cached_tokens": c_tok,
                "cache_hit_pct": cache_pct,
                "output_tokens": o_tok,
                "thought_tokens": t_tok,
                "total_tokens": p_tok + o_tok + t_tok,
            },
        }

    def resolve_session_media_file(self, session_id: str, filename: str) -> Path:
        """Safely resolves a media file inside `<sessions_root>/<session_id>/media/`."""
        for tok in (session_id, filename):
            if not tok or ".." in tok or "/" in tok or "\\" in tok or "\0" in tok:
                raise StudioAgentError("Invalid media path component.")
        media_dir = (self.sessions_root / session_id / "media").resolve()
        media_dir.relative_to(self.sessions_root)
        target = (media_dir / filename).resolve()
        target.relative_to(media_dir)
        if not target.is_file() and filename == "replay.mp4":
            with self._lock:
                is_live = session_id in self._handles
            if not is_live:
                db = self._db()
                sess = db.get_playtest_session(session_id)
                if sess is not None:
                    self._recover_unmuxed_session_video(sess, db)
        if not target.is_file():
            raise FileNotFoundError(f"Playtest media file not found: {filename}")
        return target

    def dispatch_wishes_to_council(
        self,
        wish_ids: List[str],
        producer_note: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Path, str]:
        """Approves 1 or N playtest wish items and queues them immediately into the Council's
        `.pending_playtest_wishes.json` + `.pending_steering.json` stream on the target candidate.
        """
        db = self._db()
        wishes = db.get_playtest_wishes_by_ids(wish_ids)
        if not wishes:
            raise StudioAgentError(f"No matching playtest wishes found for IDs: {wish_ids}")

        # Resolve target candidate directory
        cand_id = next((str(w.get("candidate_id") or "") for w in wishes if w.get("candidate_id")), "")
        cand_dir: Optional[Path] = None
        if cand_id:
            p = (self.cartridges_root / cand_id).resolve()
            p.relative_to(self.cartridges_root)
            if p.is_dir():
                cand_dir = p

        if cand_dir is None and self.cartridges_root.is_dir():
            cands = sorted(
                [p for p in self.cartridges_root.glob("candidate_*") if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if cands:
                cand_dir = cands[0]
            elif (self.cartridges_root / "active").is_dir():
                cand_dir = (self.cartridges_root / "active").resolve()

        if cand_dir is None:
            raise StudioAgentError("Cannot dispatch wishes to Council: no target candidate directory found.")

        sent_iso = _utc_now()
        updated_records: List[Dict[str, Any]] = []
        wish_payload_items: List[Dict[str, Any]] = []

        for w in wishes:
            wid = str(w["wish_id"])
            updated = db.update_playtest_wish_status(
                wish_id=wid,
                status="sent_to_council",
                producer_note=producer_note,
                sent_at_utc=sent_iso,
            )
            rec = updated or dict(w, status="sent_to_council", producer_note=producer_note, sent_at_utc=sent_iso)
            updated_records.append(rec)
            wish_payload_items.append({
                "wish_id": wid,
                "session_id": rec.get("session_id"),
                "turn_index": rec.get("turn_index"),
                "checkpoint_id": rec.get("checkpoint_id"),
                "observed_on_slice_id": rec.get("observed_on_slice_id") or "unspecified_slice",
                "category": rec.get("category"),
                "summary": rec.get("summary"),
                "rationale": rec.get("rationale"),
                "mood_at_time": rec.get("mood_at_time"),
                "screenshot_path": rec.get("screenshot_path"),
                "video_timestamp_sec": rec.get("video_timestamp_sec"),
                "producer_note": producer_note or "",
                "sent_at_utc": sent_iso,
            })

        # 1. Append to `.pending_playtest_wishes.json` for multimodal screenshot attachment in CreativeWritersRoom
        wishes_queue_path = cand_dir / ".pending_playtest_wishes.json"
        existing_queue: List[Dict[str, Any]] = []
        if wishes_queue_path.is_file():
            try:
                loaded_q = json.loads(wishes_queue_path.read_text(encoding="utf-8"))
                if isinstance(loaded_q, list):
                    existing_queue = loaded_q
            except Exception:
                existing_queue = []
        existing_queue.extend(wish_payload_items)
        wishes_queue_path.write_text(json.dumps(existing_queue, indent=2), encoding="utf-8")

        # 2. Also append structured steering directive to `.pending_steering.json` so active orchestrator/Council
        # loops wake and pivot immediately via the exact same fast-path as Executive Producer steering messages.
        steering_lines = [
            f"[BLIND PLAYTESTER FEEDBACK DISPATCH — {len(wish_payload_items)} Approved Player Wish Item(s)]"
        ]
        if producer_note and producer_note.strip():
            steering_lines.append(f"Executive Producer Note: {producer_note.strip()}")
        for idx, item in enumerate(wish_payload_items, 1):
            steering_lines.append(
                f"{idx}. [{item['category']}] {item['summary']} "
                f"(Observed by Blind Player on slice='{item['observed_on_slice_id']}', "
                f"checkpoint='{item['checkpoint_id']}', mood='{item.get('mood_at_time') or 'n/a'}'): "
                f"{item['rationale']}"
            )
        steering_lines.append(
            "Verify against current HEAD code whether any newer slice work already addressed this; "
            "otherwise incorporate concrete backlog tasks or slice refinements immediately."
        )
        formatted_msg = "\n".join(steering_lines)

        steering_path = cand_dir / ".pending_steering.json"
        s_list: List[str] = []
        if steering_path.is_file():
            try:
                raw_s = json.loads(steering_path.read_text(encoding="utf-8"))
                if isinstance(raw_s, list):
                    s_list = [str(x) for x in raw_s]
                elif isinstance(raw_s, dict) and raw_s.get("message"):
                    s_list = [str(raw_s["message"])]
            except Exception:
                s_list = []
        s_list.append(formatted_msg)
        steering_path.write_text(json.dumps(s_list, indent=2), encoding="utf-8")

        self._emit_event(
            "playtest_wishes_dispatched",
            f"Dispatched {len(updated_records)} Blind Playtester wish(es) immediately to Council on {cand_dir.name}.",
            {
                "candidate_id": cand_dir.name,
                "wish_ids": [w["wish_id"] for w in updated_records],
                "count": len(updated_records),
            },
        )
        return updated_records, cand_dir, formatted_msg

    def shutdown(self) -> None:
        """Stops all active Blind Playtester sessions cleanly, allowing `replay.mp4` muxing to finish."""
        self._stop_all_active_handles(timeout_sec=20.0)

    def _stop_all_active_handles(self, timeout_sec: float = 20.0) -> None:
        with self._lock:
            active_list = list(self._handles.values())
        for h in active_list:
            h.stop_event.set()
            h.pause_gate.set()
            if h.thread is not None and h.thread.is_alive():
                h.thread.join(timeout=timeout_sec)

    def _run_session_worker(self, handle: _RunningSessionHandle) -> None:
        db = self._db()
        session_id = handle.session_id
        store_card = handle.store_card
        cfg = handle.config
        media_dir = handle.session_dir / "media"
        replay_mp4_path = media_dir / "replay.mp4"

        sidebar_segments: List[SidebarSegment] = []
        sticky_feedback_items: List[Dict[str, str]] = []
        turn_index = 0
        final_status = "completed"
        end_outcome: Optional[str] = None
        end_summary: Optional[str] = None
        error_msg: Optional[str] = None
        latest_frame_path: Optional[str] = None
        last_obs_screenshot_path: Optional[Path] = None
        session_report_dict: Optional[Dict[str, Any]] = None

        def _on_live_frame(frame_pkt: Dict[str, Any]) -> None:
            p_path = str(frame_pkt.get("preview_path") or "")
            if p_path:
                self._emit_event(
                    "playtest_frame_update",
                    f"Mid-burst live preview frame at tick {frame_pkt.get('tick', 0)}.",
                    {
                        "session_id": session_id,
                        "tick": int(frame_pkt.get("tick", 0)),
                        "cursor_x": float(frame_pkt.get("cursor_x", 640.0)),
                        "cursor_y": float(frame_pkt.get("cursor_y", 360.0)),
                        "frame_name": Path(p_path).name,
                    },
                )

        try:
            client = self.client_factory() if self.client_factory is not None else None
            tracker = PlaytestTokenTracker(max_session_tokens=cfg.max_session_tokens)
            agent = BlindPlaytesterAgent(
                store_card=store_card,
                config=cfg,
                client=client,
                token_tracker=tracker,
            )
            handle.agent = agent

            if self.bridge_factory is not None:
                bridge = self.bridge_factory(
                    checkpoint_dir=handle.checkpoint_dir,
                    session_dir=handle.session_dir,
                    reaction_ticks=cfg.reaction_ticks,
                    on_live_frame=_on_live_frame,
                )
            else:
                bridge = GodotPlaytestBridge(
                    checkpoint_dir=handle.checkpoint_dir,
                    session_dir=handle.session_dir,
                    reaction_ticks=cfg.reaction_ticks,
                    on_live_frame=_on_live_frame,
                )
            handle.bridge = bridge

            obs: TurnObservation = bridge.start()
            latest_frame_path = str(obs.preview_path)
            last_obs_screenshot_path = obs.screenshot_path

            # Initial boot settle sidebar segment (ticks 0 .. obs.tick)
            boot_sidebar = render_turn_sidebar_png(
                media_dir / "sidebar_turn_0000.png",
                game_title=store_card.game_title,
                turn_index=0,
                sim_time_sec=obs.sim_time_sec,
                current_mood=agent.current_mood,
                current_plan=agent.current_plan,
                sticky_feedback_items=[],
                action_summary="Booting game & observing title screen",
                cursor_xy=(obs.cursor_x, obs.cursor_y),
            )
            sidebar_segments.append(
                SidebarSegment(turn_index=0, start_tick=0, end_tick=max(1, obs.tick), sidebar_png_path=boot_sidebar)
            )

            self._persist_session_progress(
                db=db,
                handle=handle,
                status="running",
                turn_count=0,
                sim_ticks=obs.tick,
                latest_frame_path=latest_frame_path,
                deliberation="Booted checkpoint and captured initial player screen.",
            )
            self._emit_event(
                "playtest_frame_update",
                "Captured initial opening screen.",
                {
                    "session_id": session_id,
                    "tick": obs.tick,
                    "cursor_x": obs.cursor_x,
                    "cursor_y": obs.cursor_y,
                    "frame_name": Path(latest_frame_path).name,
                },
            )

            while not handle.stop_event.is_set() and turn_index < cfg.max_turns:
                # Honor Pause gate; while paused, mux an interim replay.mp4 so operator can scrub video immediately!
                if not handle.pause_gate.is_set():
                    self._try_mux_interim_or_final_video(bridge, sidebar_segments, replay_mp4_path, db, handle)
                    while not handle.pause_gate.wait(timeout=0.25):
                        if handle.stop_event.is_set():
                            break
                    if handle.stop_event.is_set():
                        break

                turn_index += 1
                tick_start = obs.tick
                cursor_start = (obs.cursor_x, obs.cursor_y)
                obs_screenshot_path = str(obs.screenshot_path)
                obs_wav_path = str(obs.wav_path) if (obs.wav_path and not obs.audio_is_silent) else None
                vid_ts_sec = round(tick_start / 60.0, 3)

                decision, tape_line, turn_stats = agent.decide_turn(turn_index, obs)

                # Record sparse experience pillar impression updates if player formed/revised any
                if decision.impression_updates:
                    for imp_u in decision.impression_updates:
                        p_label = IMPRESSION_PILLAR_LABELS.get(imp_u.pillar, imp_u.pillar)
                        sticky_feedback_items.append({
                            "kind": "impression",
                            "turn_label": f"T#{turn_index}",
                            "badge": f"IMPRESSION ({p_label.upper()[:22]})",
                            "text": imp_u.impression.strip(),
                        })
                    self._emit_event(
                        "playtest_impression_updated",
                        decision.impression_updates[0].deliberation,
                        {
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "video_timestamp_sec": vid_ts_sec,
                            "updates": [u.model_dump() for u in decision.impression_updates],
                            "impressions": dict(agent.impressions),
                        },
                    )

                # Record mood shift if player reported one
                mood_label: Optional[str] = None
                mood_note_txt: Optional[str] = None
                if decision.mood_shift is not None:
                    mood_label = decision.mood_shift.mood.strip()
                    mood_note_txt = decision.mood_shift.note.strip()
                    sticky_feedback_items.append({
                        "kind": "mood",
                        "turn_label": f"T#{turn_index}",
                        "badge": f"MOOD -> {mood_label.upper()}",
                        "text": mood_note_txt,
                    })
                    self._emit_event(
                        "playtest_mood_shift",
                        decision.mood_shift.deliberation,
                        {
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "video_timestamp_sec": vid_ts_sec,
                            "mood": mood_label,
                            "note": mood_note_txt,
                        },
                    )

                # Record wish item if player explicitly triggered submit_wish
                if decision.submit_wish is not None:
                    w_draft = decision.submit_wish
                    wish_id = db.insert_playtest_wish({
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "candidate_id": store_card.candidate_id,
                        "checkpoint_id": store_card.checkpoint_id,
                        "observed_on_slice_id": store_card.slice_id,
                        "status": "pending",
                        "category": w_draft.category,
                        "summary": w_draft.summary,
                        "rationale": w_draft.rationale,
                        "mood_at_time": agent.current_mood,
                        "screenshot_path": obs_screenshot_path,
                        "video_timestamp_sec": vid_ts_sec,
                        "deliberation": w_draft.deliberation,
                    })
                    sticky_feedback_items.append({
                        "kind": "wish",
                        "turn_label": f"T#{turn_index}",
                        "badge": f"WISH ({w_draft.category})",
                        "text": f"{w_draft.summary}: {w_draft.rationale}",
                    })
                    self._emit_event(
                        "playtest_wish_created",
                        w_draft.deliberation,
                        {
                            "session_id": session_id,
                            "wish_id": wish_id,
                            "turn_index": turn_index,
                            "video_timestamp_sec": vid_ts_sec,
                            "category": w_draft.category,
                            "summary": w_draft.summary,
                            "rationale": w_draft.rationale,
                            "screenshot_name": Path(obs_screenshot_path).name,
                        },
                    )

                # Check if player decided to stop playing (`end_playtest`)
                if decision.end_playtest is not None:
                    end_outcome = decision.end_playtest.outcome
                    end_summary = decision.end_playtest.closing_thoughts
                    final_status = "completed"

                    db.insert_playtest_turn({
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "sim_tick_start": tick_start,
                        "sim_tick_end": tick_start,
                        "video_timestamp_sec": vid_ts_sec,
                        "cursor_start_x": cursor_start[0],
                        "cursor_start_y": cursor_start[1],
                        "cursor_end_x": cursor_start[0],
                        "cursor_end_y": cursor_start[1],
                        "current_plan": decision.current_plan,
                        "deliberation": decision.deliberation,
                        "action": decision.motor_action.model_dump(),
                        "rulebook_updates": [u.model_dump() for u in decision.rulebook_updates],
                        "impression_updates": [u.model_dump() for u in decision.impression_updates],
                        "append_log_line": tape_line,
                        "mood_state": mood_label,
                        "mood_note": mood_note_txt,
                        "screenshot_path": obs_screenshot_path,
                        "audio_clip_path": obs_wav_path,
                        "prompt_tokens": turn_stats.get("prompt_tokens", 0),
                        "cached_tokens": turn_stats.get("cached_tokens", 0),
                        "output_tokens": turn_stats.get("output_tokens", 0),
                        "thought_tokens": turn_stats.get("thought_tokens", 0),
                    })
                    self._emit_event(
                        "playtest_turn_completed",
                        decision.deliberation,
                        {
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "video_timestamp_sec": vid_ts_sec,
                            "append_log_line": tape_line,
                            "current_mood": agent.current_mood,
                            "current_plan": agent.current_plan,
                            "impressions": dict(agent.impressions),
                            "end_outcome": end_outcome,
                        },
                    )
                    break

                if handle.stop_event.is_set():
                    final_status = "stopped"
                    break

                # Execute motor action burst inside Godot
                next_obs = bridge.step_action(decision.motor_action)
                latest_frame_path = str(next_obs.preview_path)
                last_obs_screenshot_path = next_obs.screenshot_path

                # Render sticky HUD sidebar card spanning [tick_start .. next_obs.tick]
                sidebar_png = render_turn_sidebar_png(
                    media_dir / f"sidebar_turn_{turn_index:04d}.png",
                    game_title=store_card.game_title,
                    turn_index=turn_index,
                    sim_time_sec=next_obs.sim_time_sec,
                    current_mood=agent.current_mood,
                    current_plan=agent.current_plan,
                    sticky_feedback_items=sticky_feedback_items,
                    action_summary=_format_motor_summary(decision.motor_action),
                    cursor_xy=(next_obs.cursor_x, next_obs.cursor_y),
                )
                sidebar_segments.append(
                    SidebarSegment(
                        turn_index=turn_index,
                        start_tick=tick_start,
                        end_tick=max(tick_start + 1, next_obs.tick),
                        sidebar_png_path=sidebar_png,
                    )
                )

                db.insert_playtest_turn({
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "sim_tick_start": tick_start,
                    "sim_tick_end": next_obs.tick,
                    "video_timestamp_sec": vid_ts_sec,
                    "cursor_start_x": cursor_start[0],
                    "cursor_start_y": cursor_start[1],
                    "cursor_end_x": next_obs.cursor_x,
                    "cursor_end_y": next_obs.cursor_y,
                    "current_plan": decision.current_plan,
                    "deliberation": decision.deliberation,
                    "action": decision.motor_action.model_dump(),
                    "rulebook_updates": [u.model_dump() for u in decision.rulebook_updates],
                    "impression_updates": [u.model_dump() for u in decision.impression_updates],
                    "append_log_line": tape_line,
                    "mood_state": mood_label,
                    "mood_note": mood_note_txt,
                    "screenshot_path": obs_screenshot_path,
                    "audio_clip_path": obs_wav_path,
                    "prompt_tokens": turn_stats.get("prompt_tokens", 0),
                    "cached_tokens": turn_stats.get("cached_tokens", 0),
                    "output_tokens": turn_stats.get("output_tokens", 0),
                    "thought_tokens": turn_stats.get("thought_tokens", 0),
                })

                # Run 30-turn episodic compaction with 5-turn raw tail overlap if interval reached
                if agent.should_compact():
                    comp_journal, covered_turn, _ = agent.compact_journal()
                    db.insert_playtest_compaction({
                        "session_id": session_id,
                        "compaction_index": agent.compaction_count,
                        "covered_up_to_turn": covered_turn,
                        "carried_tail_turns": len(agent.epoch_turn_lines),
                        "deliberation": comp_journal.deliberation,
                        "rulebook_snapshot": dict(agent.rulebook),
                        "journal": comp_journal,
                    })
                    self._emit_event(
                        "playtest_compacted",
                        comp_journal.deliberation,
                        {
                            "session_id": session_id,
                            "compaction_index": agent.compaction_count,
                            "covered_up_to_turn": covered_turn,
                            "carried_tail_turns": len(agent.epoch_turn_lines),
                        },
                    )

                obs = next_obs
                cur_status = "paused" if not handle.pause_gate.is_set() else "running"
                self._persist_session_progress(
                    db=db,
                    handle=handle,
                    status=cur_status,
                    turn_count=turn_index,
                    sim_ticks=obs.tick,
                    latest_frame_path=latest_frame_path,
                    deliberation=decision.deliberation,
                )
                self._emit_event(
                    "playtest_turn_completed",
                    decision.deliberation,
                    {
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "video_timestamp_sec": vid_ts_sec,
                        "append_log_line": tape_line,
                        "current_mood": agent.current_mood,
                        "current_plan": agent.current_plan,
                        "impressions": dict(agent.impressions),
                        "frame_name": Path(latest_frame_path).name,
                        "token_snapshot": tracker.snapshot(),
                    },
                )

            if handle.stop_event.is_set():
                final_status = "stopped"
                if not end_outcome:
                    end_outcome = "operator_stopped"
            elif turn_index >= cfg.max_turns and not end_outcome:
                final_status = "completed"
                end_outcome = "max_turns_reached"
                end_summary = f"Reached configured safety ceiling of {cfg.max_turns} turns."

        except PlaytestBudgetExceededError as budget_exc:
            final_status = "completed"
            end_outcome = "token_budget_reached"
            end_summary = str(budget_exc)
            logger.info("Playtest session %s stopped cleanly at token ceiling: %s", session_id, budget_exc)
        except Exception as exc:
            final_status = "error"
            error_msg = f"{exc}\n{traceback.format_exc()[-1000:]}"
            logger.error("Playtest session %s failed: %s", session_id, error_msg)
        finally:
            if handle.bridge is not None:
                try:
                    handle.bridge.close()
                except Exception as close_exc:
                    logger.warning("Error closing GodotPlaytestBridge for %s: %s", session_id, close_exc)

            if handle.discard_on_stop:
                with self._lock:
                    self._handles.pop(session_id, None)
                return

            # Mux final 1600x720 H.264 + AAC broadcast replay MP4 if stream frames were captured
            video_out_str: Optional[str] = None
            if handle.bridge is not None and sidebar_segments and handle.bridge.raw_game_mkv.is_file():
                try:
                    total_ticks = max(handle.bridge.last_tick, sidebar_segments[-1].end_tick)
                    sidebar_segments[-1].end_tick = max(sidebar_segments[-1].end_tick, total_ticks)
                    muxed = mux_broadcast_replay_mp4(
                        raw_game_mkv=handle.bridge.raw_game_mkv,
                        session_pcm_path=handle.bridge.session_pcm_path,
                        total_sim_ticks=total_ticks,
                        sidebar_segments=sidebar_segments,
                        output_mp4_path=replay_mp4_path,
                    )
                    video_out_str = str(muxed)
                except Exception as mux_exc:
                    logger.error("Replay MP4 muxing failed for %s: %s", session_id, mux_exc)
                    if final_status != "error" and not replay_mp4_path.is_file():
                        error_msg = f"Session finished, but replay video muxing failed: {mux_exc}"

            ended_iso = _utc_now()

            # Synthesize end-of-session PlaytestExperienceReport whenever stopped or completed
            if (
                final_status in ("completed", "stopped")
                and handle.agent is not None
                and (turn_index > 0 or last_obs_screenshot_path is not None)
            ):
                try:
                    session_wishes = db.list_playtest_wishes(session_id=session_id, limit=200)
                    report_model, _ = handle.agent.generate_experience_report(
                        end_status=final_status,
                        end_outcome=end_outcome,
                        closing_thoughts=end_summary,
                        final_screenshot_path=last_obs_screenshot_path,
                        wishes_filed=session_wishes,
                    )
                    session_report_dict = report_model.model_dump()
                    if not end_summary:
                        end_summary = report_model.overall_experience_summary

                    # Strict delivery boundary: Council-initiated runs auto-deliver all wishes + report;
                    # manual operator runs NEVER auto-deliver wishes (they stay status='pending').
                    if handle.initiated_by == "council":
                        auto_sent_ids: List[str] = []
                        for w_row in session_wishes:
                            wid = str(w_row.get("wish_id") or "")
                            if wid and w_row.get("status") == "pending":
                                db.update_playtest_wish_status(
                                    wish_id=wid,
                                    status="sent_to_council",
                                    producer_note="Auto-delivered from Council end-of-slice Blind Playtest",
                                    sent_at_utc=ended_iso,
                                )
                                auto_sent_ids.append(wid)
                        if auto_sent_ids:
                            self._emit_event(
                                "playtest_wishes_dispatched",
                                (
                                    f"Automatically delivered {len(auto_sent_ids)} Blind Playtester wish(es) "
                                    f"from Council slice session {session_id} to Council deliberation."
                                ),
                                {
                                    "session_id": session_id,
                                    "candidate_id": store_card.candidate_id,
                                    "wish_ids": auto_sent_ids,
                                    "count": len(auto_sent_ids),
                                    "initiated_by": "council",
                                },
                            )

                    self._emit_event(
                        "playtest_report_ready",
                        report_model.deliberation,
                        {
                            "session_id": session_id,
                            "initiated_by": handle.initiated_by,
                            "overall_fun_rating": report_model.overall_fun_rating,
                            "session_report": session_report_dict,
                        },
                    )
                except Exception as rep_exc:
                    final_status = "error"
                    error_msg = f"Experience report generation failed: {rep_exc}\n{traceback.format_exc()[-800:]}"
                    logger.error("Playtest experience report generation failed for %s: %s", session_id, error_msg)

            ticks_final = handle.bridge.last_tick if handle.bridge is not None else 0
            self._persist_session_progress(
                db=db,
                handle=handle,
                status=final_status,
                turn_count=turn_index,
                sim_ticks=ticks_final,
                latest_frame_path=latest_frame_path,
                video_path=video_out_str,
                end_outcome=end_outcome,
                end_summary=end_summary,
                error_message=error_msg,
                ended_at_utc=ended_iso,
                session_report=session_report_dict,
                deliberation=(
                    error_msg
                    or end_summary
                    or f"Blind Playtester session {session_id} ended with status={final_status}."
                ),
            )
            self._emit_event(
                "playtest_session_status",
                f"Blind Playtester session {session_id} finished ({final_status}).",
                {
                    "session_id": session_id,
                    "initiated_by": handle.initiated_by,
                    "status": final_status,
                    "end_outcome": end_outcome,
                    "end_summary": end_summary,
                    "error_message": error_msg,
                    "has_video": bool(video_out_str),
                    "has_report": bool(session_report_dict),
                },
            )
            with self._lock:
                self._handles.pop(session_id, None)

    def _try_mux_interim_or_final_video(
        self,
        bridge: GodotPlaytestBridge,
        sidebar_segments: List[SidebarSegment],
        replay_mp4_path: Path,
        db: TelemetryDB,
        handle: _RunningSessionHandle,
    ) -> None:
        if not sidebar_segments or not bridge.raw_game_mkv.is_file() or bridge.raw_game_mkv.stat().st_size == 0:
            return
        try:
            muxed = mux_broadcast_replay_mp4(
                raw_game_mkv=bridge.raw_game_mkv,
                session_pcm_path=bridge.session_pcm_path,
                total_sim_ticks=max(1, bridge.last_tick),
                sidebar_segments=sidebar_segments,
                output_mp4_path=replay_mp4_path,
            )
            existing = db.get_playtest_session(handle.session_id)
            if existing is not None:
                existing["video_path"] = str(muxed)
                existing["updated_at_utc"] = _utc_now()
                db.upsert_playtest_session(existing)
            self._emit_event(
                "playtest_video_ready",
                f"Interim paused replay.mp4 muxed at tick {bridge.last_tick}.",
                {"session_id": handle.session_id, "has_video": True},
            )
        except Exception as exc:
            logger.debug("Interim pause video mux skipped: %s", exc)

    def _persist_session_progress(
        self,
        *,
        db: TelemetryDB,
        handle: _RunningSessionHandle,
        status: str,
        turn_count: int,
        sim_ticks: int,
        latest_frame_path: Optional[str],
        deliberation: str,
        video_path: Optional[str] = None,
        end_outcome: Optional[str] = None,
        end_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        ended_at_utc: Optional[str] = None,
        session_report: Optional[Dict[str, Any]] = None,
    ) -> None:
        agent = handle.agent
        tok = agent.token_tracker.snapshot() if agent is not None else {}
        existing = db.get_playtest_session(handle.session_id) or {}
        rec = {
            "session_id": handle.session_id,
            "checkpoint_id": handle.store_card.checkpoint_id,
            "candidate_id": handle.store_card.candidate_id,
            "slice_id": handle.store_card.slice_id,
            "initiated_by": handle.initiated_by,
            "game_title": handle.store_card.game_title,
            "game_description": handle.store_card.game_description,
            "status": status,
            "end_outcome": end_outcome if end_outcome is not None else existing.get("end_outcome"),
            "end_summary": end_summary if end_summary is not None else existing.get("end_summary"),
            "error_message": error_message if error_message is not None else existing.get("error_message"),
            "turn_count": turn_count,
            "sim_ticks_total": sim_ticks,
            "sim_duration_sec": round(sim_ticks / 60.0, 2),
            "compaction_count": agent.compaction_count if agent is not None else 0,
            "rulebook": dict(agent.rulebook) if agent is not None else {},
            "impressions": dict(agent.impressions) if agent is not None else (existing.get("impressions") or {}),
            "session_report": (
                session_report if session_report is not None else (existing.get("session_report") or {})
            ),
            "compacted_journal": (
                agent.compacted_journal.model_dump()
                if (agent is not None and agent.compacted_journal is not None)
                else {}
            ),
            "current_plan": agent.current_plan if agent is not None else "",
            "current_mood": agent.current_mood if agent is not None else "",
            "prompt_tokens": tok.get("prompt_tokens", 0),
            "cached_tokens": tok.get("cached_tokens", 0),
            "output_tokens": tok.get("output_tokens", 0),
            "thought_tokens": tok.get("thought_tokens", 0),
            "video_path": video_path or existing.get("video_path"),
            "latest_frame_path": latest_frame_path or existing.get("latest_frame_path"),
            "started_at_utc": existing.get("started_at_utc") or _utc_now(),
            "updated_at_utc": _utc_now(),
            "ended_at_utc": ended_at_utc or existing.get("ended_at_utc"),
            "deliberation": deliberation,
        }
        db.upsert_playtest_session(rec)
