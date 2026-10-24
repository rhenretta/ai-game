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
                if seg.get("mouse_click_pos"):
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


def capture_filmstrip(
    cartridge_dir: Path,
    ticks: List[int],
    out_path: Path,
    input_script: Optional[List[Dict[str, Any]]] = None,
    cell: Tuple[int, int] = (426, 240),
    cols: int = 3,
    setup_script: Optional[List[Dict[str, Any]]] = None,
    verify_counterfactual: bool = False,
) -> Path:
    """Runs the real game, captures the viewport at `ticks` (driven by `input_script` and `setup_script` if given),
    and composes a labelled QA proof strip with external input & state telemetry."""
    from engine.sim.godot_verifier import GodotHeadlessSimulator

    cartridge_dir = Path(cartridge_dir)
    if not ticks:
        raise StudioAgentError("capture_filmstrip requires at least one tick")
    cap_dir = cartridge_dir / "assets" / "playtest_captures"
    for stale in cap_dir.glob("filmstrip_*.png"):
        stale.unlink()
    sim = GodotHeadlessSimulator(
        cartridge_dir, capture_viewport=True, capture_ticks=ticks, input_script=input_script,
        setup_script=setup_script, max_ticks=max(ticks), telemetry_mode="in_memory",
    )
    sim.run_until_complete()
    tick_snaps = dict(getattr(sim, "tick_snapshots", {}) or {})

    if verify_counterfactual and (input_script or setup_script) and len(set(ticks)) >= 2:
        ctrl_sim = GodotHeadlessSimulator(
            cartridge_dir, capture_viewport=False, capture_ticks=ticks, input_script=[],
            setup_script=[], max_ticks=max(ticks), telemetry_mode="null",
        )
        ctrl_sim.run_until_complete()
        ctrl_snaps = dict(getattr(ctrl_sim, "tick_snapshots", {}) or {})
        sorted_t = sorted(set(ticks))
        t_first, t_last = sorted_t[0], sorted_t[-1]
        if t_first in ctrl_snaps and t_last in ctrl_snaps and t_last in tick_snaps:
            ignore_keys = {"tick", "node_count"}
            c0 = {k: v for k, v in ctrl_snaps[t_first].items() if k not in ignore_keys}
            c_end = {k: v for k, v in ctrl_snaps[t_last].items() if k not in ignore_keys}
            a_end = {k: v for k, v in tick_snaps[t_last].items() if k not in ignore_keys}
            # If the zero-input control run changed state across ticks AND ended in the exact same state as the scripted run,
            # the game is running a self-playing demo timer rather than responding to the QA input script.
            if c0 != c_end and c_end == a_end:
                raise StudioAgentError(
                    f"Counterfactual QA check failed: running with zero inputs (input_script=[]) produced the exact same state transition "
                    f"from tick {t_first} to tick {t_last} ({c_end}) as the scripted run. Remove self-playing sim_tick demo timers from main.gd/scenes.gd."
                )

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
        header_txt = f"[QA] tick {t}  {_active_inputs_at_tick(t, input_script, setup_script)}"
        draw.text((x + 6, y + 3), header_txt[:62], fill=(240, 240, 240))
        if footer_h > 0:
            snap_txt = _format_snapshot_footer(tick_snaps.get(t))
            if snap_txt:
                draw.text((x + 6, y + label_h + cell[1] + 2), f"[QA] {snap_txt}"[:62], fill=(140, 215, 245))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_path, "PNG")
    return out_path
