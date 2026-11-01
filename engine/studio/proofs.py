"""Proof renderers for the Production Board: images the council can judge.

- `render_level_map`: an annotated top-down map of a LevelLayoutSpec (regions, entities, triggers, bounds).
- `capture_filmstrip`: frames captured from the real game at chosen ticks, optionally while scripted inputs
  drive it, composed into one labelled strip (proof of actions, mechanics and animation in context).
Failures raise (Rule 5); nothing is faked when capture is unavailable.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from engine.studio.models import LevelLayoutSpec, StudioAgentError

logger = logging.getLogger("BoardProofs")

_REGION_COLORS = [
    (70, 110, 200), (200, 120, 70), (80, 170, 110), (170, 90, 170), (190, 170, 70), (70, 170, 180),
]


def render_level_map(layout: LevelLayoutSpec, out_path: Path, width: int = 1400) -> Path:
    """Draws the world as a top-down map: region fills with labels, entity footprints with ids/states, and trigger arrows."""
    bx, by, bw, bh = layout.world_bounds
    if bw <= 0 or bh <= 0:
        raise StudioAgentError(f"Cannot render a level map for non-positive world_bounds {layout.world_bounds}")
    margin, header = 30, 40
    scale = (width - 2 * margin) / bw
    height = int(bh * scale) + 2 * margin + header
    img = Image.new("RGB", (width, height), (14, 16, 24))
    draw = ImageDraw.Draw(img, "RGBA")

    def to_px(x: float, y: float) -> Tuple[float, float]:
        return margin + (x - bx) * scale, header + margin + (y - by) * scale

    x0, y0 = to_px(bx, by)
    x1, y1 = to_px(bx + bw, by + bh)
    draw.rectangle([x0, y0, x1, y1], outline=(150, 150, 170, 255), width=2)
    draw.text((margin, 8), f"WORLD {int(bw)}x{int(bh)}  viewport {layout.viewport_size}  ~{layout.intended_world_scale_in_viewports:.1f} screens", fill=(235, 235, 235, 255))

    for idx, region in enumerate(sorted(layout.world_regions, key=lambda r: r.progression_order)):
        rx, ry, rw, rh = region.bounds
        a, b = to_px(rx, ry), to_px(rx + rw, ry + rh)
        color = _REGION_COLORS[idx % len(_REGION_COLORS)]
        draw.rectangle([a, b], fill=color + (46,), outline=color + (220,), width=2)
        draw.text((a[0] + 6, a[1] + 4), f"[{region.progression_order}] {region.region_id}", fill=color + (255,))

    centers: Dict[str, Tuple[float, float]] = {}
    for e in layout.entities:
        cx, cy = to_px(*e.position)
        centers[e.entity_id] = (cx, cy)
        w, h = e.size[0] * scale, e.size[1] * scale
        if w < 4 and h < 4:
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 220, 80, 255))
        else:
            draw.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=(255, 255, 255, 38), outline=(255, 255, 255, 200))
        draw.text((cx - w / 2 + 3, cy - 5), f"{e.entity_id}\n({e.initial_state})", fill=(255, 255, 255, 255))

    for t in layout.triggers:
        if t.source_entity_id not in centers or t.target_entity_id not in centers:
            continue
        (sx, sy), (tx, ty) = centers[t.source_entity_id], centers[t.target_entity_id]
        draw.line([sx, sy, tx, ty], fill=(255, 120, 120, 200), width=2)
        ang = math.atan2(ty - sy, tx - sx)
        for da in (2.6, -2.6):
            draw.line([tx, ty, tx + 12 * math.cos(ang + da), ty + 12 * math.sin(ang + da)], fill=(255, 120, 120, 220), width=2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def _active_inputs_at_tick(
    tick: int,
    input_script: Optional[List[Dict[str, Any]]],
    setup_script: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Formats a concise QA label of active input actions, clicks, and setup calls for `tick`."""
    parts: List[str] = []
    if setup_script:
        calls = [
            f"{s.get('method_name') or s.get('method', '')}"
            for s in setup_script
            if isinstance(s, dict) and int(s.get("at_tick", 1)) <= tick and (s.get("method_name") or s.get("method"))
        ]
        if calls:
            parts.append(f"setup:{','.join(calls[-2:])}")
    if input_script:
        acts: List[str] = []
        for seg in input_script:
            if not isinstance(seg, dict):
                continue
            f_t = int(seg.get("from", 0))
            t_t = int(seg.get("to", f_t))
            if f_t <= tick <= t_t:
                for a in seg.get("actions", []):
                    if str(a) not in acts:
                        acts.append(str(a))
                if seg.get("mouse_world_pos"):
                    acts.append(f"click_world@{seg['mouse_world_pos']}")
                elif seg.get("mouse_click_pos"):
                    acts.append(f"click@{seg['mouse_click_pos']}")
        if acts:
            parts.append(f"in:[{','.join(acts[:3])}]")
    return " ".join(parts) if parts else "in:[idle]"


def _format_snapshot_footer(snap: Optional[Dict[str, Any]], max_chars: int = 58) -> str:
    """Formats a compact 1-line external QA telemetry summary from a tick snapshot."""
    if not snap or not isinstance(snap, dict):
        return ""
    items: List[str] = []
    pos = snap.get("player_pos")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        items.append(f"pos=({int(float(pos[0]))},{int(float(pos[1]))})")
    skip_keys = {"tick", "player_pos", "player_vel", "node_count", "player_health"}
    for k, v in snap.items():
        if k in skip_keys:
            continue
        if k == "is_completed" and v is False:
            continue
        if isinstance(v, float):
            items.append(f"{k}={v:.2f}")
        elif isinstance(v, (int, bool, str)):
            sval = str(v)
            if len(sval) <= 18:
                items.append(f"{k}={sval}")
        if sum(len(x) for x in items) >= max_chars:
            break
    line = " ".join(items)
    return line[:max_chars]


def _resolve_snapshot_key(snap: Dict[str, Any], key_path: str) -> Tuple[bool, Any]:
    """Resolves a dotted or top-level key against a tick snapshot dictionary."""
    if not isinstance(snap, dict) or not key_path:
        return False, None
    clean = key_path.strip()
    if clean in snap:
        return True, snap[clean]
    cur: Any = snap
    for part in clean.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def _check_single_assertion_against_snap(
    snap: Dict[str, Any],
    key: str,
    exp: str,
) -> Tuple[Optional[bool], Any]:
    """Evaluates a single `(key, exp)` expectation against `snap`.
    Returns `(passed, actual_val)` where `passed` is `None` if `key` is absent or `exp` is not a structured comparison."""
    import re as _re

    found, actual_val = _resolve_snapshot_key(snap, key)
    if not found:
        return None, None
    cmp_re = _re.compile(r"^\s*(==|!=|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?|true|false|[A-Za-z0-9_]+)\s*$", _re.IGNORECASE)
    exact_re = _re.compile(r"^\s*(-?\d+(?:\.\d+)?|true|false)\s*$", _re.IGNORECASE)
    m_cmp = cmp_re.match(exp)
    m_exact = exact_re.match(exp) if not m_cmp else None
    op = m_cmp.group(1) if m_cmp else ("==" if m_exact else "")
    rhs_raw = m_cmp.group(2) if m_cmp else (m_exact.group(1) if m_exact else "")
    if not op:
        return None, actual_val
    try:
        if rhs_raw.lower() in ("true", "false") or isinstance(actual_val, bool):
            lhs_b = bool(actual_val) if isinstance(actual_val, bool) else str(actual_val).strip().lower() in ("true", "1")
            rhs_b = rhs_raw.lower() == "true"
            ok = (lhs_b == rhs_b) if op == "==" else (lhs_b != rhs_b)
        elif isinstance(actual_val, (int, float)):
            lhs_f = float(actual_val)
            rhs_f = float(rhs_raw)
            if op == "==":
                ok = abs(lhs_f - rhs_f) <= 0.05
            elif op == "!=":
                ok = abs(lhs_f - rhs_f) > 0.05
            elif op == ">=":
                ok = lhs_f >= rhs_f - 1e-4
            elif op == "<=":
                ok = lhs_f <= rhs_f + 1e-4
            elif op == ">":
                ok = lhs_f > rhs_f
            elif op == "<":
                ok = lhs_f < rhs_f
            else:
                ok = True
        else:
            lhs_s = str(actual_val).strip().lower()
            rhs_s = rhs_raw.strip().lower()
            ok = (lhs_s == rhs_s) if op == "==" else (lhs_s != rhs_s)
        return ok, actual_val
    except Exception:
        return None, actual_val


def _evaluate_qa_state_assertions(
    tick_snaps: Union[Dict[int, Dict[str, Any]], List[Dict[str, Any]]],
    state_assertions: Optional[List[Any]],
) -> None:
    """Evaluates explicit numeric/boolean/equality `QAStateAssertion` entries against `tick_snapshots`
    when the target key is present in the captured snapshot."""
    if not state_assertions or not tick_snaps:
        return
    if isinstance(tick_snaps, list):
        tick_snaps = {
            int(s.get("tick", idx + 1)): s
            for idx, s in enumerate(tick_snaps)
            if isinstance(s, dict)
        }
    failures: List[str] = []
    for raw_a in state_assertions:
        at_t = int(getattr(raw_a, "at_tick", 0) if not isinstance(raw_a, dict) else raw_a.get("at_tick", 0))
        key = str(getattr(raw_a, "snapshot_key", "") if not isinstance(raw_a, dict) else raw_a.get("snapshot_key", "")).strip()
        exp = str(
            getattr(raw_a, "expected_value_or_change", "")
            if not isinstance(raw_a, dict)
            else raw_a.get("expected_value_or_change", "")
        ).strip()
        if at_t not in tick_snaps or not key or not exp:
            continue
        ok, actual_val = _check_single_assertion_against_snap(tick_snaps[at_t], key, exp)
        if ok is False:
            failures.append(f"tick {at_t}: expected `{key}` {exp}, got {actual_val!r}")
    if failures:
        raise StudioAgentError(
            "QA filmstrip state assertions failed against live engine snapshots: " + "; ".join(failures)
        )


def format_filmstrip_telemetry_progression(strip_path: Path) -> str:
    """Formats the per-frame input, telemetry state progression, visible HUD text, audio state, and
    live entrypoint parity saved alongside `strip_path` (`.telemetry.json`) so TaskLead, SWE, QA,
    and Council reviewers observe ground-truth runtime state directly."""
    import json as _json

    sidecar = Path(strip_path).with_suffix(".telemetry.json")
    if not sidecar.is_file():
        return ""
    try:
        data = _json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return ""
        frames = data.get("frames")
        if not isinstance(frames, list) or not frames:
            return ""
        parts: List[str] = []
        for f in frames:
            if not isinstance(f, dict):
                continue
            t = f.get("tick")
            act = f.get("active_inputs") or "in:[idle]"
            snap = f.get("snapshot_summary") or ""
            if snap:
                parts.append(f"tick {t} ({act}): {snap}")
            else:
                parts.append(f"tick {t} ({act})")
        if not parts:
            return ""
        out_sections = ["Captured Frame Telemetry: " + " -> ".join(parts)]
        scene_obs = data.get("scene_audit")
        if isinstance(scene_obs, dict):
            ui_txts = scene_obs.get("visible_ui_texts")
            if isinstance(ui_txts, list) and ui_txts:
                out_sections.append(f"Visible HUD/UI Texts: {ui_txts[:16]}")
            aud_det = scene_obs.get("audio_players_detail")
            if isinstance(aud_det, list) and aud_det:
                out_sections.append(f"Audio Players Runtime State: {aud_det[:8]}")
        parity_obs = data.get("main_scene_parity")
        if isinstance(parity_obs, dict) and parity_obs:
            out_sections.append(f"Live Main Scene Parity: {parity_obs}")
        return " | ".join(out_sections)
    except Exception:
        return ""


def _resolve_cartridge_main_scene(cartridge_dir: Path) -> str:
    proj = Path(cartridge_dir) / "project.godot"
    if proj.is_file():
        try:
            for line in proj.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("run/main_scene="):
                    val = s.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
        except Exception:
            pass
    return "res://main.tscn"


def capture_filmstrip(
    cartridge_dir: Path,
    ticks: List[int],
    out_path: Path,
    input_script: Optional[List[Dict[str, Any]]] = None,
    cell: Tuple[int, int] = (640, 360),
    cols: int = 2,
    setup_script: Optional[List[Dict[str, Any]]] = None,
    verify_counterfactual: bool = False,
    state_assertions: Optional[List[Any]] = None,
    target_scene: str = "",
    verify_entrypoint_parity: bool = False,
    audiovisual_mp4_out_path: Optional[Path] = None,
) -> Path:
    """Runs the real game, captures the viewport at `ticks` (driven by `input_script` and `setup_script` if given),
    and composes a labelled QA proof strip with external input & state telemetry (plus optional `.mp4` video+audio clip)."""
    from engine.sim.godot_verifier import GodotHeadlessSimulator

    cartridge_dir = Path(cartridge_dir)
    out_path = Path(out_path)
    av_mp4_path = Path(audiovisual_mp4_out_path) if audiovisual_mp4_out_path is not None else None
    if not ticks:
        raise StudioAgentError("capture_filmstrip requires at least one tick")
    if out_path.is_file():
        out_path.unlink()
    if av_mp4_path is not None and av_mp4_path.is_file():
        av_mp4_path.unlink()
    cap_dir = cartridge_dir / "assets" / "playtest_captures"
    for stale in cap_dir.glob("filmstrip_*.png"):
        stale.unlink()
    sim = GodotHeadlessSimulator(
        cartridge_dir, capture_viewport=True, capture_ticks=ticks, input_script=input_script,
        setup_script=setup_script, max_ticks=max(ticks), telemetry_mode="in_memory",
        target_scene=target_scene,
        capture_audiovisual_mp4=av_mp4_path,
    )
    try:
        sim.run_until_complete()
    except RuntimeError as sim_err:
        raise StudioAgentError(
            f"Godot simulation failed or timed out in {target_scene or 'res://main.tscn'}: {sim_err}"
        ) from sim_err
    script_errs = [
        f"{d.file_path.name}:{d.line_number} - {d.error_message}"
        for d in (getattr(sim, "runtime_diagnostics", None) or [])
        if getattr(d, "file_path", None) is not None and str(d.file_path).endswith(".gd")
    ] + list(getattr(sim, "script_errors", None) or [])
    if script_errs:
        raise StudioAgentError(
            f"QA filmstrip capture encountered runtime GDScript error(s) in {target_scene or 'res://main.tscn'}: "
            + " | ".join(str(e) for e in script_errs[:5])
        )
    avg_ms = float(getattr(sim, "avg_frame_time_ms", 0.0) or 0.0)
    max_ms = float(getattr(sim, "max_frame_time_ms", 0.0) or 0.0)
    audit_dict = getattr(sim, "scene_audit", None) or {}
    sv_thrash = audit_dict.get("subviewport_thrashing_nodes") or []
    if sv_thrash:
        raise StudioAgentError(
            f"QA filmstrip capture rejected due to SubViewport GPU thrashing: {sv_thrash}. "
            "Set `render_target_update_mode = SubViewport.UPDATE_ONCE` on `CLEAR_MODE_NEVER` SubViewports and only trigger `UPDATE_ONCE` when painting new strokes."
        )
    if avg_ms > 33.3:
        raise StudioAgentError(
            f"QA filmstrip capture failed 60 FPS frame-budget gate: avg_frame_time_ms={avg_ms:.2f}ms ({1000.0 / max(0.1, avg_ms):.1f} FPS, max={max_ms:.2f}ms) exceeds 33.3ms (<30 FPS). "
            "Eliminate unconditional per-frame `queue_redraw()` on nodes whose `_draw()` iterates over 2D grids or unbounded stroke arrays, and use `SubViewport.UPDATE_ONCE` for accumulation masks."
        )
    tick_snaps = dict(getattr(sim, "tick_snapshots", {}) or {})

    frames: List[Tuple[int, Path]] = []
    for t in sorted(set(ticks)):
        p = cap_dir / f"filmstrip_{t:04d}.png"
        if p.is_file():
            frames.append((t, p))
    if not frames:
        raise StudioAgentError(
            "Filmstrip capture produced no frames. Viewport capture needs a display (macOS, DISPLAY/WAYLAND_DISPLAY, or xvfb-run); "
            "refusing to fabricate proof."
        )
    missing = sorted(set(ticks) - {t for t, _ in frames})
    if missing:
        raise StudioAgentError(f"Filmstrip is missing frames for ticks {missing}")
    rows = math.ceil(len(frames) / cols)
    label_h = 18
    footer_h = 18 if tick_snaps else 0
    row_h = cell[1] + label_h + footer_h
    strip = Image.new("RGB", (cols * cell[0], rows * row_h), (24, 28, 36))
    draw = ImageDraw.Draw(strip)
    frame_telemetry_records: List[Dict[str, Any]] = []
    for i, (t, p) in enumerate(frames):
        with Image.open(p) as im:
            thumb = im.convert("RGB").resize(cell, Image.Resampling.LANCZOS)
        x, y = (i % cols) * cell[0], (i // cols) * row_h
        strip.paste(thumb, (x, y + label_h))
        # Draw a crisp 1px frame outline around the 16:9 game viewport so reviewers clearly distinguish
        # the in-game viewport from the external QA contact-sheet margin strips above/below it.
        draw.rectangle(
            [x, y + label_h, x + cell[0] - 1, y + label_h + cell[1] - 1],
            outline=(90, 105, 130),
            width=1,
        )
        active_in = _active_inputs_at_tick(t, input_script, setup_script)
        header_txt = f"[QA] tick {t}  {active_in}"
        draw.text((x + 6, y + 3), header_txt[:62], fill=(240, 240, 240))
        snap_d = tick_snaps.get(t) if isinstance(tick_snaps.get(t), dict) else None
        snap_txt_full = _format_snapshot_footer(snap_d, max_chars=600)
        if footer_h > 0:
            snap_txt = _format_snapshot_footer(snap_d, max_chars=58)
            if snap_txt:
                draw.text((x + 6, y + label_h + cell[1] + 2), f"[QA] {snap_txt}"[:62], fill=(140, 215, 245))
        frame_telemetry_records.append(
            {
                "tick": t,
                "active_inputs": active_in,
                "snapshot_summary": snap_txt_full,
                "player_pos": snap_d.get("player_pos") if isinstance(snap_d, dict) else None,
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_path, "PNG")
    scene_audit_summary: Dict[str, Any] = {}
    if isinstance(audit_dict, dict):
        if audit_dict.get("visible_ui_texts"):
            scene_audit_summary["visible_ui_texts"] = list(audit_dict.get("visible_ui_texts") or [])[:24]
        if audit_dict.get("audio_players_detail"):
            scene_audit_summary["audio_players_detail"] = list(audit_dict.get("audio_players_detail") or [])[:12]

    def _write_telemetry_sidecar(parity_summary: Optional[Dict[str, Any]] = None) -> None:
        try:
            import json as _json

            payload: Dict[str, Any] = {
                "ticks": [t for t, _ in frames],
                "frames": frame_telemetry_records,
            }
            if scene_audit_summary:
                payload["scene_audit"] = scene_audit_summary
            if parity_summary:
                payload["main_scene_parity"] = parity_summary
            out_path.with_suffix(".telemetry.json").write_text(
                _json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    _write_telemetry_sidecar()

    _evaluate_qa_state_assertions(tick_snaps, state_assertions)

    passive_clock_keys = {
        "tick",
        "node_count",
        "elapsed_time",
        "time",
        "time_elapsed",
        "frame",
        "fps",
        "delta",
        "sim_time",
        "uptime",
    }

    def _norm_snap(val: Any) -> Any:
        if isinstance(val, dict):
            return {
                k: _norm_snap(v)
                for k, v in val.items()
                if str(k).lower() not in passive_clock_keys
            }
        if isinstance(val, (list, tuple)):
            return [_norm_snap(x) for x in val]
        if isinstance(val, float):
            return int(round(val))
        if isinstance(val, str) and val.startswith("(") and val.endswith(")") and "," in val:
            try:
                parts = [float(p.strip()) for p in val[1:-1].split(",")]
                return "(" + ", ".join(str(int(round(p))) for p in parts) + ")"
            except ValueError:
                return val
        return val

    if verify_counterfactual and (input_script or setup_script) and len(set(ticks)) >= 2:
        ctrl_sim = GodotHeadlessSimulator(
            cartridge_dir, capture_viewport=False, capture_ticks=ticks, input_script=[],
            setup_script=setup_script, max_ticks=max(ticks), telemetry_mode="null",
            target_scene=target_scene,
        )
        try:
            ctrl_sim.run_until_complete()
        except RuntimeError as sim_err:
            raise StudioAgentError(
                f"Godot zero-input control simulation failed or timed out in {target_scene or 'res://main.tscn'}: {sim_err}"
            ) from sim_err
        ctrl_snaps = dict(getattr(ctrl_sim, "tick_snapshots", {}) or {})
        sorted_t = sorted(set(ticks))
        t_first, t_last = sorted_t[0], sorted_t[-1]
        if t_last in ctrl_snaps and isinstance(ctrl_snaps[t_last], dict):
            c_last_raw = ctrl_snaps[t_last]
            for win_key in ("victory", "victory_state", "is_completed", "level_completed", "game_won", "stage_cleared"):
                if bool(c_last_raw.get(win_key, False)):
                    raise StudioAgentError(
                        f"Counterfactual QA check failed: zero-input control run (input_script=[]) achieved `{win_key}=True` "
                        f"by tick {t_last} with zero player input! Ensure objectives require active player agency and never auto-complete at start."
                    )
        if t_first in ctrl_snaps and t_last in ctrl_snaps and t_last in tick_snaps:
            c0 = _norm_snap(ctrl_snaps[t_first])
            c_end = _norm_snap(ctrl_snaps[t_last])
            a_end = _norm_snap(tick_snaps[t_last])
            common_ticks = [t for t in sorted_t if t in ctrl_snaps and t in tick_snaps]
            ctrl_traj = [_norm_snap(ctrl_snaps[t]) for t in common_ticks]
            act_traj = [_norm_snap(tick_snaps[t]) for t in common_ticks]
            # If the zero-input control run produced the exact same state trajectory across ALL captured ticks
            # as the scripted run (diverging neither at any intermediate tick nor at the final tick):
            # - either a self-playing demo timer ran (`any(s != c0 for s in ctrl_traj)`), OR
            # - `setup_script` did all the state mutation and `input_script` was a dead no-op!
            # Comparing the full `common_ticks` trajectory preserves legitimate round-trip mechanics
            # (`S0 -> S1 -> S0`, e.g., open & close UI, jump & land, select & cancel) where scripted player
            # inputs causally diverge from zero-input at intermediate tick(s) before returning to neutral.
            if input_script and ctrl_traj == act_traj:
                raise StudioAgentError(
                    f"Counterfactual QA check failed: running with zero player inputs (input_script=[]) produced the exact same state "
                    f"across all captured ticks {common_ticks} (end state at tick {t_last}: {c_end}) as the scripted run! "
                    f"Either player inputs in the game script are dead/unwired, or a self-playing timer/setup call bypassed real player agency."
                )
            if not input_script and any(s != c0 for s in ctrl_traj) and ctrl_traj == act_traj:
                raise StudioAgentError(
                    f"Counterfactual QA check failed: running with zero inputs (input_script=[]) produced the exact same state transition "
                    f"from tick {t_first} to tick {t_last} ({c_end}) as the scripted run. Remove self-playing sim_tick demo timers from main.gd/scenes.gd."
                )
            # Also check if post-baseline progress assertions (assertions at tick > t_first that were NOT satisfied at t_first)
            # were ALL satisfied during the zero-input control run (e.g. when pre-placed structures auto-solve the goal on a timer).
            if state_assertions and isinstance(ctrl_snaps.get(t_first), dict):
                passive_clock_lower = {k.lower() for k in passive_clock_keys}
                post_progress_assertions: List[Tuple[int, str, str, bool]] = []
                for raw_a in state_assertions:
                    at_t = int(getattr(raw_a, "at_tick", 0) if not isinstance(raw_a, dict) else raw_a.get("at_tick", 0))
                    key = str(getattr(raw_a, "snapshot_key", "") if not isinstance(raw_a, dict) else raw_a.get("snapshot_key", "")).strip()
                    exp = str(
                        getattr(raw_a, "expected_value_or_change", "")
                        if not isinstance(raw_a, dict)
                        else raw_a.get("expected_value_or_change", "")
                    ).strip()
                    if at_t <= t_first or at_t not in ctrl_snaps or not key or not exp or key.lower() in passive_clock_lower:
                        continue
                    ok_t0, _ = _check_single_assertion_against_snap(ctrl_snaps[t_first], key, exp)
                    ok_ctrl, _ = _check_single_assertion_against_snap(ctrl_snaps[at_t], key, exp)
                    if ok_t0 is False and ok_ctrl is not None:
                        post_progress_assertions.append((at_t, key, exp, ok_ctrl))
                if post_progress_assertions and all(ok_c for _, _, _, ok_c in post_progress_assertions):
                    auto_passed_desc = [f"tick {at_t}: `{key}` {exp}" for at_t, key, exp, _ in post_progress_assertions]
                    raise StudioAgentError(
                        "Counterfactual QA check failed: zero-input control run (input_script=[]) satisfied all post-baseline progress "
                        f"assertions ({', '.join(auto_passed_desc)}) with ZERO player input! Do not pre-solve or pre-place the player's "
                        "required construction/routing loop in _ready(); require active player input to achieve the target state."
                    )

    if verify_entrypoint_parity and len(set(ticks)) >= 2:
        main_scene_res = _resolve_cartridge_main_scene(cartridge_dir)
        main_rel = main_scene_res.removeprefix("res://").lstrip("/")
        norm_target = (target_scene or "").strip()
        sorted_t = sorted(set(ticks))
        t_first, t_last = sorted_t[0], sorted_t[-1]
        if (cartridge_dir / main_rel).is_file():
            if norm_target and norm_target != main_scene_res:
                main_sim = GodotHeadlessSimulator(
                    cartridge_dir,
                    capture_viewport=False,
                    capture_ticks=ticks,
                    input_script=input_script,
                    setup_script=None,
                    max_ticks=max(ticks),
                    telemetry_mode="null",
                    target_scene=main_scene_res,
                )
                try:
                    main_sim.run_until_complete()
                except RuntimeError as sim_err:
                    raise StudioAgentError(
                        f"Entrypoint Parity Gate failed: live playable entrypoint '{main_scene_res}' crashed or timed out: {sim_err}"
                    ) from sim_err
                main_errs = [
                    f"{d.file_path.name}:{d.line_number} - {d.error_message}"
                    for d in (getattr(main_sim, "runtime_diagnostics", None) or [])
                    if getattr(d, "file_path", None) is not None and str(d.file_path).endswith(".gd")
                ] + list(getattr(main_sim, "script_errors", None) or [])
                if main_errs:
                    raise StudioAgentError(
                        f"Entrypoint Parity Gate failed: live playable entrypoint '{main_scene_res}' raised runtime GDScript error(s): "
                        + " | ".join(str(e) for e in main_errs[:5])
                    )
                main_snaps = dict(getattr(main_sim, "tick_snapshots", {}) or {})
                m_t0 = _norm_snap(main_snaps.get(t_first, {}))
                w_t0 = _norm_snap(tick_snaps.get(t_first, {}))
                m_last = _norm_snap(main_snaps.get(t_last, {}))
                w_last = _norm_snap(tick_snaps.get(t_last, {}))

                t0_diverged_keys: List[str] = []
                if isinstance(m_t0, dict) and isinstance(w_t0, dict):
                    for k, mv0 in m_t0.items():
                        if k == "player_pos" or k not in w_t0:
                            continue
                        if mv0 != w_t0[k]:
                            t0_diverged_keys.append(f"{k} (main_t0={mv0!r} vs wrapper_t0={w_t0[k]!r})")

                diverged_last: List[str] = []
                if isinstance(m_last, dict) and isinstance(w_last, dict):
                    for k, mv in m_last.items():
                        if k not in w_last:
                            continue
                        if k == "player_pos" and isinstance(m_t0, dict) and isinstance(w_t0, dict) and m_t0.get(k) != w_t0.get(k):
                            continue
                        if mv != w_last[k]:
                            diverged_last.append(f"{k} (main={mv!r} vs test_wrapper={w_last[k]!r})")

                parity_record: Dict[str, Any] = {
                    "main_scene": main_scene_res,
                    "target_scene": norm_target,
                    "staged_distinct_t0": bool(t0_diverged_keys or setup_script),
                    "t0_diverged_keys": t0_diverged_keys[:8],
                    "t_last_diverged_keys": diverged_last[:8],
                    "main_t0": m_t0,
                    "main_t_last": m_last,
                }
                _write_telemetry_sidecar(parity_record)

                if not t0_diverged_keys and not setup_script:
                    if input_script and t_first in main_snaps and t_last in main_snaps:
                        if m_t0 == m_last:
                            raise StudioAgentError(
                                f"Entrypoint Parity Gate failed: scripted player inputs produced ZERO gameplay state change in live '{main_scene_res}' "
                                f"from tick {t_first} to tick {t_last} ({m_last}). Ensure player verbs and input handlers are wired in '{main_scene_res}', not only inside '{norm_target}'."
                            )
                    if state_assertions and main_snaps:
                        main_keys_any: set[str] = set()
                        for ms in main_snaps.values():
                            if isinstance(ms, dict):
                                main_keys_any.update(ms.keys())
                        shared_assertions = [
                            a for a in state_assertions
                            if (str(getattr(a, "snapshot_key", "") if not isinstance(a, dict) else a.get("snapshot_key", "")).strip() in main_keys_any)
                        ]
                        if shared_assertions:
                            try:
                                _evaluate_qa_state_assertions(main_snaps, shared_assertions)
                            except StudioAgentError as parity_assert_err:
                                raise StudioAgentError(
                                    f"Entrypoint Parity Gate failed: state assertions passed inside QA wrapper '{norm_target}' "
                                    f"but FAILED when running the exact same player inputs on live '{main_scene_res}': {parity_assert_err}"
                                ) from parity_assert_err
                    if diverged_last:
                        raise StudioAgentError(
                            f"Entrypoint Parity Gate failed: QA wrapper scene '{norm_target}' started at the same t={t_first} state as '{main_scene_res}' "
                            f"but mutated or diverged during execution on game state key(s): {', '.join(diverged_last[:8])}. "
                            f"Do not inject mid-run state mutations inside `tests/test_*.gd`; if the production game is deadlocked or broken, report it via `production_defect_report`."
                        )
            elif input_script and t_first in tick_snaps and t_last in tick_snaps:
                if _norm_snap(tick_snaps[t_first]) == _norm_snap(tick_snaps[t_last]):
                    raise StudioAgentError(
                        f"Entrypoint Parity Gate failed: scripted player inputs produced ZERO gameplay state change in live '{main_scene_res}' "
                        f"from tick {t_first} to tick {t_last} ({_norm_snap(tick_snaps[t_last])})."
                    )

    return out_path
