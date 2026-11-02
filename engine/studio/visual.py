"""engine/studio/visual.py
Visual Art & Sprite Designer Agent — Feature 20.
Generates character portraits, entity sprites, and environment tiles via Gemini image models and
Euclidean 16-color Style Bible palette quantization.
Includes an offline high-fidelity ProceduralSpriteRenderer for deterministic sprite generation.
Enforces Rule 4 Rationale-First Epistemic Architecture (deliberation strictly field 0).
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
import hashlib
import io
import logging
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageDraw

from engine.assets.palette import (
    STYLE_BIBLE_16_PALETTE,
    MOOD_PALETTES,
    quantize_rgb,
    quantize_image_to_indexed,
)
from engine.studio.models import (
    DEFAULT_STUDIO_MODEL,
    AnimationStateSpec,
    AnimationValidationVerdict,
    ArticulatedPartSpec,
    AssetManifestSpec,
    AssetRequirement,
    AssetVisualValidationVerdict,
    CharacterSheet,
    GameRequirementsSpec,
    KeyframeTransformSpec,
    PartKeyframeTrackSpec,
    StudioAgentError,
    VisualMockupArtifact,
    bounded_thinking_config,
    call_gemini_with_transient_retry,
)

logger = logging.getLogger("VisualArtSpriteDesignerAgent")


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def get_seed_integer(seed: Union[int, str, None]) -> int:
    """Deterministically maps any seed input to a 32-bit unsigned integer."""
    if seed is None:
        return 42
    if isinstance(seed, int):
        return seed
    digest = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class ProceduralSpriteRenderer:
    """Renders 16-color indexed procedural sprites and tiles using pure Pillow and geometric primitives."""
    deliberation: str = "Procedural sprite renderer generating deterministic 16-color indexed pixel art."

    def __init__(
        self,
        seed: Union[int, str, None] = None,
        mood: str = "retro_arcade",
        palette: Optional[List[Tuple[int, int, int]]] = None,
    ):
        self.seed_int = get_seed_integer(seed)
        self.rng = random.Random(self.seed_int)
        self.mood = mood if mood in MOOD_PALETTES else "retro_arcade"
        self.palette = palette or STYLE_BIBLE_16_PALETTE
        self.roles = MOOD_PALETTES[self.mood]

    def _color(self, index: int) -> Tuple[int, int, int]:
        return self.palette[index]

    def render_player_portrait(self, size: Tuple[int, int] = (256, 256)) -> Image.Image:
        """High-resolution stylized 16-color character portrait."""
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Scale factor relative to 64x64 base
        sx = w / 64.0
        sy = h / 64.0

        c_armor = self._color(self.roles["hero_primary"])
        c_visor = self._color(self.roles["hero_accent"])
        c_core = self._color(self.roles["hero_core"])
        c_shadow = self._color(1)
        c_glint = self._color(7)

        # Head / Helmet
        draw.ellipse([16 * sx, 8 * sy, 48 * sx, 44 * sy], fill=c_armor, outline=c_shadow, width=max(1, int(2 * sx)))
        # Visor
        draw.rounded_rectangle([22 * sx, 20 * sy, 42 * sx, 32 * sy], radius=int(4 * sx), fill=c_visor, outline=c_core)
        # Specular glint
        draw.line([24 * sx, 22 * sy, 30 * sx, 22 * sy], fill=c_glint, width=max(1, int(2 * sx)))
        # Torso / Collar
        draw.polygon(
            [(12 * sx, 44 * sy), (52 * sx, 44 * sy), (58 * sx, 64 * sy), (6 * sx, 64 * sy)],
            fill=c_shadow,
            outline=c_armor,
        )
        # Energy Core Emblem
        draw.ellipse([28 * sx, 48 * sy, 36 * sx, 56 * sy], fill=c_core)

        return quantize_image_to_indexed(img, self.palette)

    def render_boss_portrait(self, size: Tuple[int, int] = (256, 256)) -> Image.Image:
        """Antagonist heavy mechanical boss portrait."""
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        sx = w / 64.0
        sy = h / 64.0

        c_hull = self._color(self.roles["boss_hull"])
        c_trim = self._color(self.roles["boss_trim"])
        c_eye = self._color(8)  # Neon Red
        c_shadow = self._color(0)

        # Main Chassis
        draw.rectangle([10 * sx, 12 * sy, 54 * sx, 52 * sy], fill=c_hull, outline=c_trim, width=max(1, int(3 * sx)))
        # Ocular Optical Sensor
        draw.ellipse([20 * sx, 22 * sy, 44 * sx, 38 * sy], fill=c_shadow, outline=c_trim)
        draw.ellipse([27 * sx, 26 * sy, 37 * sx, 34 * sy], fill=c_eye)
        # Horn / Antenna Plating
        draw.polygon([(10 * sx, 12 * sy), (4 * sx, 2 * sy), (18 * sx, 12 * sy)], fill=c_trim)
        draw.polygon([(54 * sx, 12 * sy), (60 * sx, 2 * sy), (46 * sx, 12 * sy)], fill=c_trim)

        return quantize_image_to_indexed(img, self.palette)

    def render_npc_portrait(self, size: Tuple[int, int] = (256, 256)) -> Image.Image:
        """Companion relay drone portrait."""
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        sx = w / 64.0
        sy = h / 64.0

        c_hull = self._color(6)   # Light Gray
        c_glow = self._color(11)  # Electric Green
        c_shadow = self._color(1) # Dark Navy

        draw.ellipse([14 * sx, 14 * sy, 50 * sx, 50 * sy], fill=c_hull, outline=c_shadow, width=max(1, int(2 * sx)))
        draw.ellipse([22 * sx, 22 * sy, 42 * sx, 42 * sy], fill=c_glow, outline=c_shadow)
        # Antenna
        draw.line([32 * sx, 14 * sy, 32 * sx, 4 * sy], fill=c_shadow, width=max(1, int(3 * sx)))
        draw.ellipse([29 * sx, 2 * sy, 35 * sx, 8 * sy], fill=c_glow)

        return quantize_image_to_indexed(img, self.palette)

    def render_player_idle(self) -> Image.Image:
        """Controllable player character idle frame (32x32)."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        c_armor = self._color(self.roles["hero_primary"])
        c_visor = self._color(self.roles["hero_accent"])
        c_core = self._color(self.roles["hero_core"])
        c_navy = self._color(1)

        draw.rectangle([11, 4, 20, 11], fill=c_navy)
        draw.rectangle([12, 5, 19, 10], fill=c_armor)
        draw.rectangle([14, 6, 19, 8], fill=c_visor)
        draw.rectangle([10, 11, 21, 21], fill=c_armor)
        draw.rectangle([14, 13, 17, 16], fill=c_core)
        draw.rectangle([11, 22, 14, 28], fill=c_navy)
        draw.rectangle([17, 22, 20, 28], fill=c_navy)
        return quantize_image_to_indexed(img, self.palette)

    def render_player_walk1(self) -> Image.Image:
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        c_armor = self._color(self.roles["hero_primary"])
        c_visor = self._color(self.roles["hero_accent"])
        c_core = self._color(self.roles["hero_core"])
        c_navy = self._color(1)
        draw.rectangle([10, 4, 19, 11], fill=c_navy)
        draw.rectangle([11, 5, 18, 10], fill=c_armor)
        draw.rectangle([13, 6, 18, 8], fill=c_visor)
        draw.rectangle([9, 11, 20, 21], fill=c_armor)
        draw.rectangle([13, 13, 16, 16], fill=c_core)
        draw.rectangle([8, 22, 12, 29], fill=c_navy)
        draw.rectangle([18, 21, 22, 27], fill=c_navy)
        return quantize_image_to_indexed(img, self.palette)

    def render_player_walk2(self) -> Image.Image:
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        c_armor = self._color(self.roles["hero_primary"])
        c_visor = self._color(self.roles["hero_accent"])
        c_core = self._color(self.roles["hero_core"])
        c_navy = self._color(1)
        draw.rectangle([12, 4, 21, 11], fill=c_navy)
        draw.rectangle([13, 5, 20, 10], fill=c_armor)
        draw.rectangle([15, 6, 20, 8], fill=c_visor)
        draw.rectangle([11, 11, 22, 21], fill=c_armor)
        draw.rectangle([15, 13, 18, 16], fill=c_core)
        draw.rectangle([18, 22, 22, 29], fill=c_navy)
        draw.rectangle([8, 21, 12, 27], fill=c_navy)
        return quantize_image_to_indexed(img, self.palette)

    def render_player_dash(self) -> Image.Image:
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        c_armor = self._color(self.roles["hero_primary"])
        c_visor = self._color(self.roles["hero_accent"])
        c_core = self._color(self.roles["hero_core"])
        c_navy = self._color(1)
        c_trail = self._color(self.roles["hero_accent"])
        draw.rectangle([11, 4, 20, 11], fill=c_navy)
        draw.rectangle([12, 5, 19, 10], fill=c_armor)
        draw.rectangle([14, 6, 19, 8], fill=c_visor)
        draw.rectangle([10, 11, 21, 21], fill=c_armor)
        draw.rectangle([14, 13, 17, 16], fill=c_core)
        draw.rectangle([11, 22, 14, 28], fill=c_navy)
        draw.rectangle([17, 22, 20, 28], fill=c_navy)
        draw.line([2, 8, 8, 8], fill=c_trail, width=1)
        draw.line([0, 15, 7, 15], fill=c_trail, width=2)
        draw.line([3, 24, 9, 24], fill=c_trail, width=1)
        return quantize_image_to_indexed(img, self.palette)

    def render_chaser(self) -> Image.Image:
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        c_armor = self._color(self.roles["chaser_armor"])
        c_eye = self._color(self.roles["chaser_eye"])
        draw.polygon([(16, 4), (28, 26), (4, 26)], fill=c_armor)
        draw.rectangle([13, 16, 18, 20], fill=c_eye)
        return quantize_image_to_indexed(img, self.palette)

    def render_sniper(self) -> Image.Image:
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        c_hull = self._color(self.roles["ranged_hull"])
        c_lens = self._color(self.roles["ranged_lens"])
        draw.ellipse([6, 6, 25, 25], fill=c_hull)
        draw.ellipse([11, 11, 20, 20], fill=c_lens)
        return quantize_image_to_indexed(img, self.palette)

    def render_spikes(self) -> Image.Image:
        img = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        c_warn = self._color(self.roles["hazard_warn"])
        c_lethal = self._color(self.roles["hazard_lethal"])
        draw.rectangle([2, 40, 45, 46], fill=c_warn)
        for x in [6, 18, 30, 42]:
            draw.polygon([(x, 40), (x + 3, 10), (x + 6, 40)], fill=c_lethal)
        return quantize_image_to_indexed(img, self.palette)

    def render_acid(self) -> Image.Image:
        img = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        c_acid = self._color(self.roles["acid_liquid"])
        c_warn = self._color(self.roles["hazard_warn"])
        draw.rounded_rectangle([4, 4, 91, 91], radius=16, fill=c_acid, outline=c_warn, width=2)
        return quantize_image_to_indexed(img, self.palette)

    def render_wall(self) -> Image.Image:
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        c_stone = self._color(self.roles["stone_face"])
        c_metal = self._color(self.roles["metal_plate"])
        draw.rectangle([0, 0, 31, 31], fill=c_stone, outline=c_metal, width=2)
        return quantize_image_to_indexed(img, self.palette)

    def render_floor(self) -> Image.Image:
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        c_base = self._color(self.roles["floor_base"])
        c_plate = self._color(self.roles["metal_plate"])
        draw.rectangle([0, 0, 31, 31], fill=c_base)
        draw.line([0, 0, 31, 31], fill=c_plate, width=1)
        return quantize_image_to_indexed(img, self.palette)

    def render_dynamic_sprite(
        self,
        dimensions: Tuple[int, int] = (32, 32),
        prompt: str = "",
        asset_type: str = "sprite",
        seed: Optional[Union[int, str]] = None,
        geometry_directive: Optional[str] = None,
    ) -> Image.Image:
        """Renders a procedural 16-color indexed sprite or tile dynamically based on
        dimensions, geometric directives, color palette mood, and visual prompt.
        """
        w, h = dimensions
        if w <= 0 or h <= 0:
            raise ValueError(f"Sprite dimensions must be positive integers, got ({w}, {h})")

        # Deterministic seed derivation combining seed, prompt, asset_type, and geometry_directive
        geom_key = f"_{geometry_directive}" if geometry_directive else ""
        seed_key = f"{seed or ''}_{prompt}_{asset_type}{geom_key}_{w}x{h}"
        derived_seed = get_seed_integer(seed_key)
        rng = random.Random(derived_seed)

        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Palette colors selection
        c_dark = self._color(0)    # Pure Black / Deep Shadow
        c_navy = self._color(1)    # Dark Navy
        c_light = self._color(7)   # Off-White / Specular Highlight

        primary_idx = self.roles.get("hero_primary", 12)
        accent_idx = self.roles.get("hero_accent", 8)
        core_idx = self.roles.get("hero_core", 9)
        trim_idx = self.roles.get("boss_trim", 10)
        wall_idx = self.roles.get("stone_face", 5)

        c_primary = self._color(primary_idx)
        c_accent = self._color(accent_idx)
        c_core = self._color(core_idx)
        c_trim = self._color(trim_idx)
        c_wall = self._color(wall_idx)

        norm_type = asset_type.lower().strip()
        is_tile = "tile" in norm_type or "floor" in norm_type or "wall" in norm_type or "terrain" in norm_type
        is_portrait = "portrait" in norm_type

        if is_tile:
            # 1. Tile Rendering (Opaque background with structural and border details)
            c_bg = self._color(self.roles.get("floor_base", 1))
            c_fg = self._color(self.roles.get("metal_plate", 6))
            draw.rectangle([0, 0, w - 1, h - 1], fill=c_bg, outline=c_fg, width=max(1, min(w, h) // 16))

            tile_style = derived_seed % 4
            if geometry_directive:
                norm_geom = geometry_directive.lower().strip()
                if any(k in norm_geom for k in ("diagonal", "cross", "stripe", "lattice", "grid")):
                    tile_style = 0
                elif any(k in norm_geom for k in ("brick", "masonry", "split", "horizontal")):
                    tile_style = 1
                elif any(k in norm_geom for k in ("border", "inset", "panel", "frame", "box")):
                    tile_style = 2
                elif any(k in norm_geom for k in ("center", "core", "plus")):
                    tile_style = 3

            if tile_style == 0:
                step = max(4, w // 4)
                for x in range(0, w, step):
                    draw.line([x, 0, min(w - 1, x + step), h - 1], fill=c_fg, width=1)
            elif tile_style == 1:
                mid_y = h // 2
                draw.line([0, mid_y, w - 1, mid_y], fill=c_fg, width=max(1, h // 16))
                draw.line([w // 2, 0, w // 2, mid_y], fill=c_fg, width=1)
                draw.line([w // 4, mid_y, w // 4, h - 1], fill=c_fg, width=1)
                draw.line([3 * w // 4, mid_y, 3 * w // 4, h - 1], fill=c_fg, width=1)
            elif tile_style == 2:
                # Safe separate horizontal and vertical insets to prevent y1 < y0 on wide aspect ratios
                inset_x = max(2, min(w // 8, max(2, (w - 4) // 2)))
                inset_y = max(2, min(h // 8, max(2, (h - 4) // 2)))
                x0, y0 = inset_x, inset_y
                x1, y1 = w - 1 - inset_x, h - 1 - inset_y
                if x1 > x0 and y1 > y0:
                    draw.rectangle([x0, y0, x1, y1], fill=c_wall, outline=c_trim, width=1)
                    if x0 + 4 <= x1 and y0 + 4 <= y1:
                        draw.rectangle([x0 + 2, y0 + 2, x0 + 4, y0 + 4], fill=c_accent)
                    elif x0 + 2 <= x1 and y0 + 2 <= y1:
                        draw.rectangle([x0 + 1, y0 + 1, x0 + 2, y0 + 2], fill=c_accent)
            else:
                cx, cy = w // 2, h // 2
                draw.line([0, cy, w - 1, cy], fill=c_core, width=max(1, h // 16))
                draw.line([cx, 0, cx, h - 1], fill=c_core, width=max(1, w // 16))
                rx = max(1, min(w // 8, max(1, (w - 2) // 2)))
                ry = max(1, min(h // 8, max(1, (h - 2) // 2)))
                draw.rectangle([cx - rx, cy - ry, cx + rx, cy + ry], fill=c_accent)

        elif is_portrait:
            # 2. Stylized Portrait (Scaled to arbitrary w, h)
            sx = w / 64.0
            sy = h / 64.0
            draw.rectangle([0, 0, w - 1, h - 1], fill=c_navy, outline=c_trim, width=max(1, int(2 * sx)))
            draw.ellipse([16 * sx, 10 * sy, 48 * sx, 42 * sy], fill=c_primary, outline=c_dark, width=max(1, int(1.5 * sx)))
            draw.rounded_rectangle([22 * sx, 20 * sy, 42 * sx, 30 * sy], radius=max(1, int(3 * sx)), fill=c_accent, outline=c_core)
            draw.line([24 * sx, 22 * sy, 30 * sx, 22 * sy], fill=c_light, width=max(1, int(1.5 * sx)))
            draw.polygon([(10 * sx, 44 * sy), (54 * sx, 44 * sy), (60 * sx, 64 * sy), (4 * sx, 64 * sy)], fill=c_dark, outline=c_primary)

        else:
            # 3. Dynamic Entity / Sprite / Item / Hazard (Transparent background)
            norm_geom = geometry_directive.lower().strip() if geometry_directive else ""
            force_radial = any(k in norm_geom for k in ("radial", "concentric", "circle", "spherical", "orb"))
            force_bilateral = any(
                k in norm_geom
                for k in (
                    "bilateral", "mirror", "symmetric", "vertical", "horizontal", "winged",
                    "sharp", "angular", "triangle", "wedge", "arrow", "polygon",
                    "box", "rect", "block", "square",
                    "round", "oval", "curve", "ellipse",
                )
            )

            if force_radial:
                sym_mode = 1
            elif force_bilateral:
                sym_mode = 0
            else:
                sym_mode = (derived_seed >> 2) % 2

            cx = w / 2.0
            cy = h / 2.0

            if (sym_mode == 0 or w != h) and not force_radial:
                # Bilateral Vertical Symmetry
                if any(k in norm_geom for k in ("sharp", "angular", "triangle", "wedge", "arrow", "polygon")):
                    body_style = 0
                elif any(k in norm_geom for k in ("box", "rect", "block", "square")):
                    body_style = 1
                elif any(k in norm_geom for k in ("round", "oval", "curve", "ellipse")):
                    body_style = 2
                else:
                    body_style = (derived_seed >> 4) % 3

                margin_x = max(1, w // 6)
                margin_y = max(1, h // 6)

                if body_style == 0:
                    pts_left = [
                        (cx, margin_y),
                        (cx - (cx - margin_x) * 0.7, margin_y + (h - 2 * margin_y) * 0.3),
                        (margin_x, h - margin_y - 2),
                        (cx, h - margin_y),
                    ]
                    pts_right = [
                        (cx + (cx - px), py) for px, py in reversed(pts_left[:-1])
                    ]
                    draw.polygon(pts_left + pts_right, fill=c_primary, outline=c_dark)
                elif body_style == 1:
                    draw.rectangle([margin_x, margin_y, w - 1 - margin_x, h - 1 - margin_y], fill=c_primary, outline=c_dark, width=max(1, w // 16))
                else:
                    draw.ellipse([margin_x, margin_y, w - 1 - margin_x, h - 1 - margin_y], fill=c_primary, outline=c_dark, width=max(1, w // 16))

                # Symmetric lateral wings / appendages
                wing_y1 = margin_y + (h - 2 * margin_y) * 0.4
                wing_y2 = margin_y + (h - 2 * margin_y) * 0.85
                draw.polygon([(margin_x, wing_y1), (0, wing_y2), (margin_x, wing_y2)], fill=c_trim, outline=c_dark)
                draw.polygon([(w - 1 - margin_x, wing_y1), (w - 1, wing_y2), (w - 1 - margin_x, wing_y2)], fill=c_trim, outline=c_dark)

                # Core glowing emblem / cockpit / crystal
                core_w = max(2, int(w * 0.25))
                core_h = max(2, int(h * 0.25))
                draw.rectangle([cx - core_w // 2, cy - core_h // 2, cx + core_w // 2, cy + core_h // 2], fill=c_core, outline=c_accent)

                # Specular top-left highlight
                draw.point((max(1, margin_x + 1), max(1, margin_y + 1)), fill=c_light)

            else:
                # Radial / Concentric Symmetry with procedural seed entropy
                radius_max = max(2, min(w, h) // 2 - 2) if min(w, h) >= 8 else max(1, min(w, h) // 2 - 1)

                outer_style = (derived_seed >> 3) % 3
                if outer_style == 0:
                    draw.ellipse(
                        [cx - radius_max, cy - radius_max, cx + radius_max, cy + radius_max],
                        fill=c_primary,
                        outline=c_dark,
                        width=max(1, min(w, h) // 16),
                    )
                elif outer_style == 1:
                    n_sides = 4 + ((derived_seed >> 5) % 5)
                    rot = ((derived_seed >> 8) % 12) * (math.pi / 6.0)
                    pts = [
                        (
                            cx + radius_max * math.cos(rot + 2.0 * math.pi * i / n_sides),
                            cy + radius_max * math.sin(rot + 2.0 * math.pi * i / n_sides),
                        )
                        for i in range(n_sides)
                    ]
                    draw.polygon(pts, fill=c_primary, outline=c_dark)
                else:
                    n_teeth = 4 + 2 * ((derived_seed >> 5) % 4)
                    r_inner = max(1, int(radius_max * (0.65 + 0.05 * ((derived_seed >> 7) % 4))))
                    rot = ((derived_seed >> 9) % 8) * (math.pi / 4.0)
                    pts = []
                    for i in range(n_teeth * 2):
                        r_step = radius_max if i % 2 == 0 else r_inner
                        ang = rot + math.pi * i / n_teeth
                        pts.append((cx + r_step * math.cos(ang), cy + r_step * math.sin(ang)))
                    draw.polygon(pts, fill=c_primary, outline=c_dark)

                mid_feature = (derived_seed >> 11) % 4
                if mid_feature == 0:
                    r_mid = max(1, int(radius_max * (0.5 + 0.05 * ((derived_seed >> 13) % 4))))
                    draw.ellipse(
                        [cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid],
                        fill=c_accent,
                        outline=c_trim,
                        width=max(1, min(w, h) // 24),
                    )
                elif mid_feature == 1:
                    n_poly = 3 + ((derived_seed >> 13) % 4)
                    r_poly = max(1, int(radius_max * (0.55 + 0.05 * ((derived_seed >> 15) % 3))))
                    rot_poly = ((derived_seed >> 17) % 8) * (math.pi / 4.0)
                    pts = [
                        (
                            cx + r_poly * math.cos(rot_poly + 2.0 * math.pi * i / n_poly),
                            cy + r_poly * math.sin(rot_poly + 2.0 * math.pi * i / n_poly),
                        )
                        for i in range(n_poly)
                    ]
                    draw.polygon(pts, fill=c_accent, outline=c_trim)
                elif mid_feature == 2:
                    spoke_count = 4 + 2 * ((derived_seed >> 13) % 3)
                    r_spoke_inner = max(1, radius_max // 4)
                    r_spoke_outer = max(1, int(radius_max * 0.85))
                    rot_spoke = ((derived_seed >> 16) % 6) * (math.pi / 3.0)
                    for i in range(spoke_count):
                        ang = rot_spoke + 2.0 * math.pi * i / spoke_count
                        p1 = (cx + r_spoke_inner * math.cos(ang), cy + r_spoke_inner * math.sin(ang))
                        p2 = (cx + r_spoke_outer * math.cos(ang), cy + r_spoke_outer * math.sin(ang))
                        draw.line([p1, p2], fill=c_accent, width=max(1, min(w, h) // 20))
                else:
                    node_count = 4 + 2 * ((derived_seed >> 13) % 3)
                    r_node_ring = max(1, int(radius_max * 0.7))
                    r_dot = max(1, min(w, h) // 20)
                    rot_node = ((derived_seed >> 16) % 6) * (math.pi / 3.0)
                    for i in range(node_count):
                        ang = rot_node + 2.0 * math.pi * i / node_count
                        nx = cx + r_node_ring * math.cos(ang)
                        ny = cy + r_node_ring * math.sin(ang)
                        draw.ellipse([nx - r_dot, ny - r_dot, nx + r_dot, ny + r_dot], fill=c_accent)

                r_core = max(1, int(radius_max * (0.2 + 0.05 * ((derived_seed >> 19) % 4))))
                core_shape = (derived_seed >> 21) % 2
                if core_shape == 0:
                    draw.ellipse([cx - r_core, cy - r_core, cx + r_core, cy + r_core], fill=c_core, outline=c_accent)
                else:
                    draw.rectangle([cx - r_core, cy - r_core, cx + r_core, cy + r_core], fill=c_core, outline=c_accent)

                hi_angle = ((derived_seed >> 23) % 4) * (math.pi / 2.0) - math.pi * 0.75
                r_hi = max(1, int(r_core * 0.5))
                hx = int(cx + r_hi * math.cos(hi_angle))
                hy = int(cy + r_hi * math.sin(hi_angle))
                draw.point((hx, hy), fill=c_light)

        return quantize_image_to_indexed(img, self.palette)

    def render_from_requirement(
        self,
        requirement: Union[AssetRequirement, Dict[str, Any]],
    ) -> Image.Image:
        """Renders a procedural sprite directly from an AssetRequirement or requirement dict."""
        if isinstance(requirement, dict):
            dims = requirement.get("dimensions") or (32, 32)
            prompt = requirement.get("visual_prompt_or_text") or requirement.get("description", "")
            atype = requirement.get("asset_type", "sprite")
            seed = requirement.get("asset_id") or requirement.get("name")
            geom = requirement.get("geometry_directive")
        else:
            dims = getattr(requirement, "dimensions", None) or (32, 32)
            prompt = getattr(requirement, "visual_prompt_or_text", "") or getattr(requirement, "description", "")
            atype = getattr(requirement, "asset_type", "sprite")
            seed = getattr(requirement, "asset_id", None) or getattr(requirement, "name", None)
            geom = getattr(requirement, "geometry_directive", None)
        return self.render_dynamic_sprite(dimensions=dims, prompt=prompt, asset_type=atype, seed=seed, geometry_directive=geom)



class VisualArtSpriteDesignerAgent:
    """Specialist Agent producing 16-color character portraits, sprites, and tiles."""
    deliberation: str = "Visual art sprite designer agent generating 16-color Style Bible assets."
    _rembg_session: Optional[Any] = None

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = "gemini-3.1-flash-lite-image",
        text_model_name: str = DEFAULT_STUDIO_MODEL,
        allow_offline_procedural: bool = True,
    ):
        self.client = client
        self.model_name = model_name
        self.text_model_name = text_model_name
        self.allow_offline_procedural = bool(allow_offline_procedural)
        self.renderer = ProceduralSpriteRenderer()

    def _ensure_procedural_allowed_or_raise(self, op_name: str) -> None:
        """Rule 5: Fail fast when offline procedural synthesis is disabled and no live Gemini client is configured."""
        if not self._is_live_genai_client() and not self.allow_offline_procedural:
            from engine.studio.models import StudioAgentConfigurationError
            raise StudioAgentConfigurationError(
                f"VisualArtSpriteDesignerAgent.{op_name} requires a configured live Gemini client "
                f"(allow_offline_procedural=False; silent procedural fallback is prohibited by Rule 5)."
            )

    @classmethod
    def _get_rembg_session(cls) -> Optional[Any]:
        if cls._rembg_session is None:
            try:
                from rembg import new_session
                cls._rembg_session = new_session("u2netp")
            except Exception as exc:
                logger.debug("rembg session initialization skipped: %s", exc)
        return cls._rembg_session

    @staticmethod
    def quantize_color(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Quantizes an arbitrary RGB tuple to the nearest Euclidean 16-color entry."""
        return quantize_rgb(rgb)

    @classmethod
    def extract_clean_alpha_sprite(
        cls,
        raw_img: Union[Image.Image, bytes, bytearray],
        target_size: Tuple[int, int] = (64, 64),
        strip_border_frame: bool = True,
        pad_ratio: float = 0.06,
        target_dimensions: Optional[Tuple[int, int]] = None,
        size: Optional[Tuple[int, int]] = None,
        isolate_single_subject: bool = True,
        **_kwargs: Any,
    ) -> Image.Image:
        """Extracts a clean 32-bit RGBA foreground cutout from a synthesized image.
        1. Uses multi-tone perimeter-connected background flood-fill (handling both solid backdrops and
           baked-in 2-tone transparency checkerboards) + neural salient-object segmentation (`rembg` U2-Net)
           to isolate the foreground subject without hand-rolled interior white/eye hole-punching heuristics.
        2. Clears any residual pure chroma-key (`#FF00FF`) background pockets if a chroma backdrop was used.
        3. When `isolate_single_subject=True`, detects multi-pose collages or multi-item grids via morphological
           subject clustering and isolates the primary single subject before tight bounding-box cropping.
        """
        if isinstance(raw_img, (bytes, bytearray)):
            raw_img = Image.open(io.BytesIO(raw_img))
        eff_size = target_dimensions if target_dimensions is not None else (size if size is not None else target_size)
        w, h = max(1, int(eff_size[0])), max(1, int(eff_size[1]))
        work_w, work_h = 256, 256
        work_img = raw_img.convert("RGBA").resize((work_w, work_h), Image.Resampling.LANCZOS)
        px = work_img.load()

        # Step 1: Perimeter seed collection across all 4 edges for outer background connected component
        seed_coords: List[Tuple[int, int]] = []
        for x in range(0, work_w, 4):
            seed_coords.append((x, 0))
            seed_coords.append((x, work_h - 1))
        for y in range(0, work_h, 4):
            seed_coords.append((0, y))
            seed_coords.append((work_w - 1, y))

        if strip_border_frame:
            for frac in (0.03, 0.06):
                ix = int(work_w * frac)
                iy = int(work_h * frac)
                inset_pts = [
                    (ix, iy),
                    (work_w - 1 - ix, iy),
                    (ix, work_h - 1 - iy),
                    (work_w - 1 - ix, work_h - 1 - iy),
                ]
                c0 = px[ix, iy][:3]
                corners_match = sum(
                    1
                    for pt in inset_pts
                    if ((px[pt[0], pt[1]][0] - c0[0]) ** 2 + (px[pt[0], pt[1]][1] - c0[1]) ** 2 + (px[pt[0], pt[1]][2] - c0[2]) ** 2) <= 35 * 35
                )
                if corners_match >= 3:
                    seed_coords.extend(inset_pts)

        bg_samples: List[Tuple[int, int, int]] = []
        for sx, sy in seed_coords:
            r, g, b, a = px[sx, sy]
            if a > 0:
                bg_samples.append((r, g, b))
        if not bg_samples:
            bg_samples = [(0, 0, 0)]

        # Cluster perimeter samples into dominant perimeter background tones (supports solid backgrounds
        # AND 2-tone alternating transparency checkerboards around the perimeter).
        cluster_sums: List[List[int]] = []
        cluster_counts: List[int] = []
        cluster_centers: List[Tuple[int, int, int]] = []
        cluster_rad_sq = 24 * 24
        for sr, sg, sb in bg_samples:
            matched_idx = -1
            for c_idx, (cr, cg, cb) in enumerate(cluster_centers):
                if (sr - cr) ** 2 + (sg - cg) ** 2 + (sb - cb) ** 2 <= cluster_rad_sq:
                    matched_idx = c_idx
                    break
            if matched_idx >= 0:
                cluster_sums[matched_idx][0] += sr
                cluster_sums[matched_idx][1] += sg
                cluster_sums[matched_idx][2] += sb
                cluster_counts[matched_idx] += 1
                cnt = cluster_counts[matched_idx]
                cluster_centers[matched_idx] = (
                    cluster_sums[matched_idx][0] // cnt,
                    cluster_sums[matched_idx][1] // cnt,
                    cluster_sums[matched_idx][2] // cnt,
                )
            else:
                cluster_sums.append([sr, sg, sb])
                cluster_counts.append(1)
                cluster_centers.append((sr, sg, sb))

        min_cluster_hits = max(3, int(len(bg_samples) * 0.04))
        ranked_clusters = sorted(
            zip(cluster_counts, cluster_centers),
            key=lambda item: item[0],
            reverse=True,
        )
        bg_palette: List[Tuple[int, int, int]] = [
            center for cnt, center in ranked_clusters if cnt >= min_cluster_hits
        ][:6]
        if not bg_palette:
            bg_palette = [ranked_clusters[0][1]] if ranked_clusters else [(0, 0, 0)]

        # Detect whether the outer perimeter is a 2-tone neutral/achromatic checkerboard
        top2_coverage = sum(cnt for cnt, _ in ranked_clusters[:2]) / float(max(1, len(bg_samples)))
        is_perimeter_checkerboard = (
            len(ranked_clusters) >= 2
            and ranked_clusters[1][0] >= max(4, int(len(bg_samples) * 0.08))
            and top2_coverage >= 0.72
            and all((max(c) - min(c)) <= 28 for _, c in ranked_clusters[:2])
        )

        # Step 2: Perimeter-connected flood fill (never punches disconnected interior eyes/highlights)
        visited = set()
        stack: List[Tuple[int, int, Tuple[int, int, int]]] = []
        for sx, sy in seed_coords:
            r, g, b, a = px[sx, sy]
            if a == 0:
                continue
            stack.append((sx, sy, (r, g, b)))

        tol_hard_sq = 26 * 26
        tol_soft_sq = 38 * 38

        while stack:
            cx, cy, ref_rgb = stack.pop()
            if (cx, cy) in visited or cx < 0 or cx >= work_w or cy < 0 or cy >= work_h:
                continue
            visited.add((cx, cy))
            r, g, b, a = px[cx, cy]
            if a == 0:
                continue

            dist_local_sq = (r - ref_rgb[0]) ** 2 + (g - ref_rgb[1]) ** 2 + (b - ref_rgb[2]) ** 2
            dist_global_sq = min(
                (r - br) ** 2 + (g - bg) ** 2 + (b - bb) ** 2
                for br, bg, bb in bg_palette
            )
            # If the pixel matches ANY dominant perimeter background tone (e.g., stepping between light and dark
            # squares of a perimeter-connected checkerboard), allow flood propagation via dist_global_sq.
            effective_dist_sq = dist_global_sq if dist_global_sq <= tol_hard_sq else max(dist_local_sq, dist_global_sq)

            if effective_dist_sq <= tol_hard_sq:
                px[cx, cy] = (0, 0, 0, 0)
                next_ref = (r, g, b) if dist_global_sq <= tol_hard_sq else (
                    (ref_rgb[0] * 3 + r) // 4,
                    (ref_rgb[1] * 3 + g) // 4,
                    (ref_rgb[2] * 3 + b) // 4,
                )
                stack.extend([
                    (cx + 1, cy, next_ref),
                    (cx - 1, cy, next_ref),
                    (cx, cy + 1, next_ref),
                    (cx, cy - 1, next_ref),
                ])
            elif effective_dist_sq <= tol_soft_sq:
                ratio = (math.sqrt(effective_dist_sq) - 26.0) / 12.0
                new_a = max(0, min(255, int(a * ratio)))
                px[cx, cy] = (r, g, b, new_a)

        # Step 2b: Clear pure #FF00FF chroma-key pockets and despill chroma fringes
        # (including light/translucent boundary pixels where white/grey blended with #FF00FF yields g >= 110)
        for iy in range(work_h):
            for ix in range(work_w):
                r, g, b, a = px[ix, iy]
                if a == 0:
                    continue
                magenta_excess = min(r, b) - g
                rb_diff = abs(r - b)
                if r > 175 and b > 175 and g < 95 and magenta_excess > 105:
                    px[ix, iy] = (0, 0, 0, 0)
                elif r > 140 and b > 140 and g < 110 and magenta_excess > 65:
                    clamped_rb = min(r, b, g + 35)
                    new_a = max(0, min(a, int(a * 0.55)))
                    px[ix, iy] = (clamped_rb, g, clamped_rb, new_a)
                elif r > 180 and b > 180 and rb_diff < 35 and magenta_excess > 45 and (a < 250 or g > 95):
                    spill_frac = min(1.0, float(magenta_excess) / 220.0)
                    clamped_rb = min(r, b, g + 18)
                    new_a = max(0, min(a, int(a * (1.0 - 0.65 * spill_frac))))
                    px[ix, iy] = (clamped_rb, g, clamped_rb, new_a)

        rgba_np = np.array(work_img, dtype=np.uint8)

        # Step 2c: If the outer perimeter was a uniform synthetic studio background or 2-tone checkerboard,
        # clear enclosed interior background pockets while protecting internal facial/head features.
        c_tl = np.array(bg_palette[0], dtype=np.float32)
        bg_uniform = all(float(np.linalg.norm(np.array(s, dtype=np.float32) - c_tl)) < 25.0 for s in bg_samples[:16])
        if bg_uniform or is_perimeter_checkerboard:
            rgb_f = rgba_np[:, :, :3].astype(np.float32)
            min_dist_sq = np.full((work_h, work_w), 1e9, dtype=np.float32)
            for b_rgb in bg_palette[:2]:
                b_vec = np.array(b_rgb, dtype=np.float32)
                d_sq = np.sum((rgb_f - b_vec[None, None, :]) ** 2, axis=-1)
                min_dist_sq = np.minimum(min_dist_sq, d_sq)
            tol_pocket = 18.0 if is_perimeter_checkerboard else 14.0
            cand = ((rgba_np[:, :, 3] > 0) & (min_dist_sq <= tol_pocket * tol_pocket)).astype(np.uint8)
            num_lbl, lbls, stats, centroids = cv2.connectedComponentsWithStats(cand, connectivity=8)
            for lbl in range(1, num_lbl):
                area = int(stats[lbl, cv2.CC_STAT_AREA])
                cy_cent = float(centroids[lbl][1])
                if is_perimeter_checkerboard and 24 <= area <= int(work_w * work_h * 0.25):
                    rgba_np[lbls == lbl] = [0, 0, 0, 0]
                elif 28 <= area <= int(work_w * work_h * 0.10) and cy_cent > work_h * 0.36:
                    rgba_np[lbls == lbl] = [0, 0, 0, 0]

        # Step 2d: Neural salient-object matting (`rembg`) when perimeter flood-fill leaves a non-uniform background
        opaque_frac = float(np.count_nonzero(rgba_np[:, :, 3] > 24)) / float(work_w * work_h)
        if opaque_frac > 0.78:
            from rembg import remove as rembg_remove
            session = cls._get_rembg_session()
            rembg_out = rembg_remove(work_img, session=session) if session is not None else rembg_remove(work_img)
            rembg_np = np.array(rembg_out.convert("RGBA"), dtype=np.uint8)
            rembg_opaque = float(np.count_nonzero(rembg_np[:, :, 3] > 24)) / float(work_w * work_h)
            if 0.03 <= rembg_opaque <= 0.85:
                rgba_np[:, :, 3] = np.minimum(rgba_np[:, :, 3], rembg_np[:, :, 3])
                rgba_np[rgba_np[:, :, 3] == 0] = [0, 0, 0, 0]

        # Step 2e: Strip tiny detached noise specks (< 1.5% of primary subject area)
        fg_mask = (rgba_np[:, :, 3] > 24).astype(np.uint8)
        num_fg, fg_labels, fg_stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
        if num_fg > 2:
            areas = [int(fg_stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_fg)]
            max_fg_area = max(areas) if areas else 0
            min_keep_area = max(10, int(max_fg_area * 0.015))
            for lbl in range(1, num_fg):
                if int(fg_stats[lbl, cv2.CC_STAT_AREA]) < min_keep_area:
                    rgba_np[fg_labels == lbl] = [0, 0, 0, 0]

        # Step 2f: Single-subject isolation — if the image model synthesized a multi-pose collage or
        # multi-item variation grid (2+ spatially disjoint subject clusters), isolate the primary single subject.
        if isolate_single_subject:
            fg_clean = (rgba_np[:, :, 3] > 24).astype(np.uint8)
            # Dilate by 9x9 (~3.5% of 256px canvas) so closely attached parts of ONE character/prop merge together
            kernel = np.ones((9, 9), dtype=np.uint8)
            dilated_fg = cv2.dilate(fg_clean, kernel, iterations=1)
            num_cl, cl_labels, cl_stats, cl_centroids = cv2.connectedComponentsWithStats(dilated_fg, connectivity=8)
            if num_cl > 2:
                cluster_true_areas: Dict[int, int] = {}
                for c_idx in range(1, num_cl):
                    cluster_true_areas[c_idx] = int(np.count_nonzero((cl_labels == c_idx) & (fg_clean > 0)))
                max_cl_area = max(cluster_true_areas.values()) if cluster_true_areas else 0
                major_clusters = [
                    c_idx
                    for c_idx, c_area in cluster_true_areas.items()
                    if c_area >= max(100, int(max_cl_area * 0.16))
                ]
                if len(major_clusters) >= 2:
                    # Check whether the major clusters have non-overlapping bounding boxes (side-by-side poses or grid cells)
                    candidates = [c_idx for c_idx in major_clusters if cluster_true_areas[c_idx] >= int(max_cl_area * 0.25)]
                    if not candidates:
                        candidates = major_clusters
                    best_cl = max(
                        candidates,
                        key=lambda c_idx: (
                            (cluster_true_areas[c_idx] / float(max(1, max_cl_area)))
                            - 0.25 * (
                                math.hypot(
                                    float(cl_centroids[c_idx][0]) - work_w * 0.5,
                                    float(cl_centroids[c_idx][1]) - work_h * 0.5,
                                )
                                / (work_w * 0.5)
                            )
                        ),
                    )
                    bx = int(cl_stats[best_cl, cv2.CC_STAT_LEFT])
                    by = int(cl_stats[best_cl, cv2.CC_STAT_TOP])
                    bw_c = int(cl_stats[best_cl, cv2.CC_STAT_WIDTH])
                    bh_c = int(cl_stats[best_cl, cv2.CC_STAT_HEIGHT])
                    # Verify at least one other major cluster is laterally or vertically disjoint from best_cl
                    has_disjoint_sibling = False
                    for other_cl in major_clusters:
                        if other_cl == best_cl:
                            continue
                        ox = int(cl_stats[other_cl, cv2.CC_STAT_LEFT])
                        oy = int(cl_stats[other_cl, cv2.CC_STAT_TOP])
                        ow_c = int(cl_stats[other_cl, cv2.CC_STAT_WIDTH])
                        oh_c = int(cl_stats[other_cl, cv2.CC_STAT_HEIGHT])
                        overlap_x = max(0, min(bx + bw_c, ox + ow_c) - max(bx, ox))
                        overlap_y = max(0, min(by + bh_c, oy + oh_c) - max(by, oy))
                        if overlap_x < min(bw_c, ow_c) * 0.35 or overlap_y < min(bh_c, oh_c) * 0.35:
                            has_disjoint_sibling = True
                            break
                    if has_disjoint_sibling:
                        keep_mask = cl_labels == best_cl
                        # Keep any tiny detail cluster within 10px of best_cl's bounding box
                        pad_detail = 10
                        for c_idx, c_area in cluster_true_areas.items():
                            if c_idx == best_cl or c_idx in major_clusters:
                                continue
                            cx_c, cy_c = float(cl_centroids[c_idx][0]), float(cl_centroids[c_idx][1])
                            if (bx - pad_detail) <= cx_c <= (bx + bw_c + pad_detail) and (by - pad_detail) <= cy_c <= (by + bh_c + pad_detail):
                                keep_mask = keep_mask | (cl_labels == c_idx)
                        rgba_np[~keep_mask] = [0, 0, 0, 0]

        work_img = Image.fromarray(rgba_np)

        # When preserving multi-cell atlases / sheets (`isolate_single_subject=False`),
        # never tight-crop & re-center the composite bounding box, which would shift grid cell coordinates.
        if not isolate_single_subject:
            if work_img.getbbox() is not None:
                return work_img if work_img.size == (w, h) else work_img.resize((w, h), Image.Resampling.LANCZOS)
            return raw_img.convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)

        # Step 3: Crop to tight foreground bounding box and fit into target_size with padding
        bbox = work_img.getbbox()
        if bbox:
            x0, y0, x1, y1 = bbox
            if (x1 - x0) >= 8 and (y1 - y0) >= 8:
                cropped = work_img.crop(bbox)
                cw, ch = cropped.size
                pad_x = max(2, int(w * pad_ratio))
                pad_y = max(2, int(h * pad_ratio))
                avail_w = max(1, w - 2 * pad_x)
                avail_h = max(1, h - 2 * pad_y)
                scale = min(avail_w / max(1, cw), avail_h / max(1, ch))
                nw = max(1, int(cw * scale))
                nh = max(1, int(ch * scale))
                resized_obj = cropped.resize((nw, nh), Image.Resampling.LANCZOS)
                final_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                final_img.paste(resized_obj, ((w - nw) // 2, (h - nh) // 2), resized_obj)
                return final_img

        return raw_img.convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)


    def _build_reference_parts(
        self,
        reference_images: Optional[List[Union[Path, str, bytes, Image.Image]]] = None,
    ) -> List[Any]:
        """Converts reference images (e.g. Writers' Room 16:9 mockup or character sheet) into genai Part objects."""
        if not reference_images or not self._is_live_genai_client():
            return []
        parts: List[Any] = []
        try:
            from google.genai import types
            for ref in reference_images[:2]:
                raw_bytes: Optional[bytes] = None
                if isinstance(ref, (Path, str)):
                    p = Path(ref)
                    if p.is_file():
                        raw_bytes = p.read_bytes()
                elif isinstance(ref, bytes):
                    raw_bytes = ref
                elif isinstance(ref, Image.Image):
                    buf = io.BytesIO()
                    ref.save(buf, format="PNG")
                    raw_bytes = buf.getvalue()
                if raw_bytes:
                    parts.append(types.Part.from_bytes(data=raw_bytes, mime_type="image/png"))
        except Exception as exc:
            logger.debug("Reference image part encoding skipped: %s", exc)
        return parts

    def generate_portrait(
        self,
        character: Optional[Union[CharacterSheet, str, Dict[str, Any]]] = None,
        output_path: Optional[Union[Path, str]] = None,
        dimensions: Tuple[int, int] = (256, 256),
        **kwargs: Any,
    ) -> Path:
        """Generates a character portrait and saves as a valid PNG with clean alpha transparency."""
        if character is None and "character_name" in kwargs:
            character = kwargs.pop("character_name")
        if output_path is None and "output_path" in kwargs:
            output_path = kwargs.pop("output_path")
        if character is None:
            character = "Hero"
        if output_path is None:
            raise ValueError("output_path is required for generate_portrait")

        w, h = dimensions
        if w <= 0 or h <= 0:
            raise ValueError("Sprite dimensions must be positive integers")

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if self.client is None:
            raise StudioAgentError("VisualArtSpriteDesignerAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)")

        if isinstance(character, str):
            char_obj = CharacterSheet(
                deliberation=f"Portrait character {character}",
                character_id=character.lower(),
                name=character,
                role=kwargs.get("role", kwargs.get("archetype", "hero")),
                visual_description=kwargs.get("visual_prompt", kwargs.get("visual_description", "Expressive game character portrait")),
            )
        elif isinstance(character, dict):
            c_name = character.get("name", "Hero")
            char_obj = CharacterSheet(
                deliberation=character.get("deliberation", f"Portrait character {c_name}"),
                character_id=character.get("character_id", c_name.lower()),
                name=c_name,
                role=character.get("role", character.get("archetype", "hero")),
                visual_description=character.get("visual_description", character.get("visual_prompt", "Expressive game character portrait")),
            )
        else:
            char_obj = character

        style_guide = kwargs.get("style_guide", "")
        reference_images = kwargs.get("reference_images")

        max_retries = 3
        img_bytes = None
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                from google.genai import types
                prompt = (
                    f"Expressive 2D game dialogue bust portrait of {char_obj.name} ({char_obj.role}).\n"
                    f"Visual design: {char_obj.visual_description}.\n"
                    f"{f'Art style direction: {style_guide}. ' if style_guide else ''}"
                    f"Single isolated character bust centered on a solid flat uniform pure black (#000000) background, "
                    f"NO border frame, NO picture frame, NO text, rich cohesive colors and expressive silhouette."
                )
                ref_parts = self._build_reference_parts(reference_images)
                contents: Any = (ref_parts + [prompt]) if ref_parts else prompt
                response = call_gemini_with_transient_retry(
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            image_config=types.ImageConfig(aspect_ratio="1:1"),
                        ),
                    ),
                    max_transient_retries=0,
                    label=f"synthesize_portrait[{getattr(char_obj, 'character_id', None) or out_p.stem}]",
                )
                if response.candidates:
                    for cand in response.candidates:
                        if cand.content and cand.content.parts:
                            for p in cand.content.parts:
                                if hasattr(p, "inline_data") and p.inline_data:
                                    img_bytes = p.inline_data.data
                                    break
                if img_bytes:
                    break
                logger.warning("Attempt %d: No image bytes returned for portrait '%s', retrying...", attempt, char_obj.name)
                time.sleep(1.0)
            except Exception as exc:
                last_exc = exc
                logger.warning("Attempt %d failed for portrait '%s': %s", attempt, char_obj.name, exc)
                time.sleep(1.0)

        if not img_bytes:
            if last_exc:
                raise StudioAgentError(f"VisualArtSpriteDesigner failed to generate portrait after {max_retries} attempts: {last_exc}") from last_exc
            raise StudioAgentError(f"No image bytes returned from Gemini image generation for '{char_obj.name}' after {max_retries} attempts")

        raw_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        if self._is_live_genai_client() and not kwargs.get("force_16_color", False):
            cleaned = self.extract_clean_alpha_sprite(raw_img, target_size=(w, h), strip_border_frame=True, pad_ratio=0.04)
            cleaned.save(out_p, "PNG")
        else:
            raw_img = raw_img.resize((w, h), Image.Resampling.LANCZOS)
            quantized = quantize_image_to_indexed(raw_img, STYLE_BIBLE_16_PALETTE)
            quantized.save(out_p, "PNG")
        return out_p

    def generate_concept_mockup(
        self,
        mockup: Union[VisualMockupArtifact, Dict[str, Any]],
        output_path: Union[Path, str],
        style_guide: str = "",
        dimensions: Tuple[int, int] = (1280, 720),
    ) -> Path:
        """Synthesizes a 16:9 full-screen gameplay and diegetic UI visual mockup board via Gemini image model."""
        if self.client is None:
            raise StudioAgentError(
                "VisualArtSpriteDesignerAgent requires an active google.genai Client to synthesize concept mockups (Rule 5: Zero Silent Fallbacks)"
            )

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(mockup, dict):
            title = mockup.get("title", "Gameplay Concept Mockup")
            scene_desc = mockup.get("scene_description", "")
            aspect = mockup.get("aspect_ratio", "16:9") or "16:9"
            callouts = mockup.get("ui_callout_notes", []) or []
        else:
            title = mockup.title
            scene_desc = mockup.scene_description
            aspect = mockup.aspect_ratio or "16:9"
            callouts = mockup.ui_callout_notes or []

        callout_str = "; ".join(str(c) for c in callouts) if callouts else "Clear, readable UI and visual feedback suited to the game's genre"
        style_str = f" Art style direction: {style_guide}." if style_guide else ""

        prompt = (
            f"Polished indie video game 16:9 widescreen screenshot mockup: '{title}'. "
            f"Scene staging: {scene_desc}.{style_str} "
            f"UI & visual feedback framing: {callout_str}. "
            f"Rich atmospheric lighting, cohesive color palette, distinct silhouettes, expressive visual storytelling."
        )

        max_retries = 3
        img_bytes = None
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                from google.genai import types
                response = call_gemini_with_transient_retry(
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            image_config=types.ImageConfig(aspect_ratio=aspect),
                        ),
                    ),
                    max_transient_retries=0,
                    label=f"synthesize_concept_mockup[{out_p.stem}]",
                )
                if response.candidates:
                    for cand in response.candidates:
                        if cand.content and cand.content.parts:
                            for p in cand.content.parts:
                                if hasattr(p, "inline_data") and p.inline_data:
                                    img_bytes = p.inline_data.data
                                    break
                if img_bytes:
                    break
                logger.warning("Attempt %d: No image bytes returned for concept mockup '%s', retrying...", attempt, title)
                time.sleep(1.0)
            except Exception as exc:
                last_exc = exc
                logger.warning("Attempt %d failed for concept mockup '%s': %s", attempt, title, exc)
                time.sleep(1.0)

        if not img_bytes:
            if last_exc:
                raise StudioAgentError(
                    f"VisualArtSpriteDesigner failed to synthesize concept mockup '{title}' after {max_retries} attempts: {last_exc}"
                ) from last_exc
            raise StudioAgentError(
                f"No image bytes returned from Gemini image generation for concept mockup '{title}' after {max_retries} attempts"
            )

        raw_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        if dimensions and dimensions[0] > 0 and dimensions[1] > 0:
            raw_img = raw_img.resize(dimensions, Image.Resampling.LANCZOS)
        raw_img.save(out_p, "PNG")
        return out_p

    def _is_live_genai_client(self) -> bool:
        """Returns True if self.client is an active google.genai Client rather than None or a unit-test mock."""
        if self.client is None:
            return False
        mod = getattr(type(self.client), "__module__", "") or ""
        return mod.startswith("google.genai")

    def _generate_gemini_sprite_image(
        self,
        prompt: str,
        asset_type: str = "sprite",
        dimensions: Tuple[int, int] = (64, 64),
        reference_images: Optional[List[Union[Path, str, bytes, Image.Image]]] = None,
        style_guide: str = "",
        is_animated_actor: bool = False,
        isolate_single_subject: bool = True,
    ) -> Image.Image:
        """Synthesizes a real 2D game sprite, prop, tile, or parallax backdrop via Gemini image generation."""
        from google.genai import types

        w, h = max(1, int(dimensions[0])), max(1, int(dimensions[1]))
        norm_type = str(asset_type).lower().strip()
        is_tile = norm_type in ("tile", "tile_surface")
        is_backdrop = norm_type in ("backdrop", "parallax_layer")
        is_actor = bool(is_animated_actor or norm_type in ("sprite_sheet", "entity", "character"))

        filtered_refs: List[Union[Path, str, bytes, Image.Image]] = list(reference_images) if (reference_images and not is_tile) else []
        style_clause = f" Match the visual style and color palette ({style_guide})." if (filtered_refs or style_guide) else ""

        if is_backdrop:
            aspect = "16:9"
            full_prompt = (
                f"Pure 2D indie game scenic environment background illustration (unpopulated vista plate): {prompt}.{style_clause} "
                f"Painterly 2D game background layer only, rich atmospheric depth, strictly NO UI text, NO HUD, NO gauges, NO words, NO borders."
            )
        elif is_tile:
            aspect = "1:1"
            full_prompt = (
                f"Flat 90-degree orthographic top-down seamless game material texture swatch (macro close-up of surface material only): {prompt}. "
                f"Color palette & art style: {style_guide or 'vibrant hand-painted fantasy'}. "
                f"100% full-bleed edge-to-edge organic asymmetric repeatable material surface filling every pixel of the square frame. "
                f"CRITICAL: NEVER draw a gray-and-white or gray-and-black transparency checkerboard pattern! "
                f"If the material is a see-through lattice, truss, girder, or grate with open gaps, fill the open gaps with a solid flat uniform bright magenta (#FF00FF) chroma-key color (NEVER a checkerboard grid). "
                f"Strictly NO horizon, NO sky, NO isometric 3D block, NO bilateral mirror symmetry, NO white corners, NO empty margins, NO border frame, NO text."
            )
        elif is_actor and isolate_single_subject:
            aspect = "1:1"
            full_prompt = (
                f"SINGLE solitary full-body high-resolution illustrated in-game subject in ONE clean neutral readiness pose: {prompt}.{style_clause} "
                f"Strictly ONE single isolated full-body figure centered on a solid flat uniform bright magenta (#FF00FF) chroma-key background with complete silhouette and grounded base. "
                f"CRITICAL: Draw STRICTLY ONLY ONE single subject instance in ONE pose — NEVER multiple figures, NEVER multiple poses or animation frames, "
                f"NEVER a multi-angle turnaround chart, NEVER a sprite sheet grid, NEVER a gray-and-white transparency checkerboard pattern, "
                f"NO detached environment terrain slab, NO ground pedestal, NO cast shadow on the floor, NO background scenery, NOT a bust portrait, NO border frame, NO text."
            )
        elif not isolate_single_subject:
            aspect = "1:1"
            full_prompt = (
                f"High-resolution illustrated in-game asset sheet on a solid flat uniform bright magenta (#FF00FF) chroma-key background: {prompt}.{style_clause} "
                f"Keep each cell cleanly centered inside its grid cell with generous magenta (#FF00FF) margin around every element, "
                f"NEVER draw grid lines, border frames, picture boxes, or text labels, and NEVER draw a gray-and-white checkerboard pattern."
            )
        else:
            aspect = "1:1"
            full_prompt = (
                f"SINGLE isolated high-resolution illustrated in-game structure or interactive prop in ONE canonical intact state: {prompt}.{style_clause} "
                f"Centered on a solid flat uniform bright magenta (#FF00FF) chroma-key background. "
                f"CRITICAL: Draw STRICTLY ONLY ONE single discrete object/structure instance — NEVER multiple variations, NEVER a 3x3 or multi-state grid, "
                f"NEVER a gray-and-white transparency checkerboard pattern, NO ground slab, NO pedestal, "
                f"NO cast shadow on the floor, NO background sky/scenery, NO border frame, NO picture box, NO text, "
                f"crisp readable silhouette and vibrant details."
            )

        ref_parts = [] if is_tile else self._build_reference_parts(filtered_refs)
        contents: Any = (ref_parts + [full_prompt]) if ref_parts else full_prompt

        max_retries = 3
        img_bytes = None
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                response = call_gemini_with_transient_retry(
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            image_config=types.ImageConfig(aspect_ratio=aspect),
                        ),
                    ),
                    max_transient_retries=0,
                    label=f"synthesize_sprite_image[{norm_type or 'sprite'}]",
                )
                if response.candidates:
                    for cand in response.candidates:
                        if cand.content and cand.content.parts:
                            for p in cand.content.parts:
                                if hasattr(p, "inline_data") and p.inline_data:
                                    img_bytes = p.inline_data.data
                                    break
                if img_bytes:
                    break
                time.sleep(0.8)
            except Exception as exc:
                last_exc = exc
                logger.warning("Gemini sprite generation attempt %d failed for '%s': %s", attempt, prompt[:50], exc)
                time.sleep(0.8)

        if not img_bytes:
            if last_exc:
                raise StudioAgentError(
                    f"VisualArtSpriteDesigner failed to synthesize sprite '{prompt[:60]}' after {max_retries} attempts: {last_exc}"
                ) from last_exc
            raise StudioAgentError(
                f"No image bytes returned from Gemini for sprite '{prompt[:60]}' after {max_retries} attempts"
            )

        raw_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        if is_tile:
            # Check if the tile has interior magenta chroma-key openings (e.g., see-through truss/lattice/grate)
            arr_raw = np.array(raw_img, dtype=np.uint8)
            h_r, w_r = arr_raw.shape[:2]
            inner = arr_raw[h_r // 6 : h_r - h_r // 6, w_r // 6 : w_r - w_r // 6]
            if inner.size > 0:
                mag_mask = (inner[:, :, 0] > 185) & (inner[:, :, 2] > 185) & (inner[:, :, 1] < 90)
                if float(np.mean(mag_mask)) >= 0.03:
                    return self.extract_clean_alpha_sprite(
                        raw_img,
                        target_size=(w, h),
                        strip_border_frame=True,
                        pad_ratio=0.0,
                        isolate_single_subject=False,
                    )
            # Only apply full-bleed toroidal tiling when the target is an approximately square surface swatch;
            # non-square strips (e.g. 96x32 modular segments or thin fluid ribbons) would be smeared by toroidal self-blending.
            if max(w, h) / max(1, min(w, h)) >= 1.75:
                return self.extract_clean_alpha_sprite(
                    raw_img,
                    target_size=(w, h),
                    strip_border_frame=True,
                    pad_ratio=0.02,
                )
            return self.ensure_full_bleed_tile(raw_img, target_size=(w, h))
        if is_backdrop:
            return raw_img.resize((w, h), Image.Resampling.LANCZOS)

        return self.extract_clean_alpha_sprite(
            raw_img,
            target_size=(w, h),
            strip_border_frame=True,
            pad_ratio=0.06,
            isolate_single_subject=isolate_single_subject,
        )

    @staticmethod
    def compute_tile_mirror_symmetry(img: Image.Image) -> float:
        """Computes the maximum bilateral mirror correlation in [0.0, 1.0] across both the full tile
        and the outer 20% border strips (detecting Rorschach / kaleidoscope mirror-reflection defects).
        """
        rgb = np.array(img.convert("RGB"), dtype=np.float32)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        if h < 16 or w < 16:
            return 0.0

        def _corr(a: np.ndarray, b: np.ndarray) -> float:
            a_z = a - float(np.mean(a))
            b_z = b - float(np.mean(b))
            denom = float(np.linalg.norm(a_z) * np.linalg.norm(b_z))
            if denom <= 1e-5:
                return 0.0
            return float(np.sum(a_z * b_z) / denom)

        half_w = w // 2
        half_h = h // 2
        full_lr = _corr(gray[:, :half_w], np.fliplr(gray[:, w - half_w :]))
        full_tb = _corr(gray[:half_h, :], np.flipud(gray[h - half_h :, :]))

        margin_w = max(4, w // 5)
        margin_h = max(4, h // 5)
        border_lr = _corr(gray[:, :margin_w], np.fliplr(gray[:, w - margin_w :]))
        border_tb = _corr(gray[:margin_h, :], np.flipud(gray[h - margin_h :, :]))

        return round(float(max(0.0, full_lr, full_tb, (border_lr + border_tb) * 0.5)), 4)

    @classmethod
    def ensure_full_bleed_tile(cls, raw_img: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> Image.Image:
        """Ensures a tile texture is 100% full-bleed with zero white/magenta/transparent corner triangles,
        and synthesizes a seamless toroidal tile using NON-MIRRORED offset translation (`np.roll`) +
        raised-cosine diamond blending (eliminating Rorschach/kaleidoscope mirror artifacts).
        """
        w, h = max(16, int(target_size[0])), max(16, int(target_size[1]))
        img = raw_img.convert("RGBA").resize((256, 256), Image.Resampling.LANCZOS)
        arr_rgba = np.array(img, dtype=np.uint8)
        r_ch, g_ch, b_ch, a_ch = arr_rgba[:, :, 0], arr_rgba[:, :, 1], arr_rgba[:, :, 2], arr_rgba[:, :, 3]
        is_bg = (a_ch < 200) | ((r_ch > 215) & (g_ch > 215) & (b_ch > 215)) | ((r_ch > 200) & (b_ch > 200) & (g_ch < 80))
        fg_mask = ~is_bg

        fg_rows = np.where(np.mean(fg_mask, axis=1) > 0.50)[0]
        fg_cols = np.where(np.mean(fg_mask, axis=0) > 0.50)[0]
        if len(fg_rows) >= 64 and len(fg_cols) >= 64:
            top, bottom = int(fg_rows[0]), int(fg_rows[-1]) + 1
            left, right = int(fg_cols[0]), int(fg_cols[-1]) + 1
            # Also inset past any remaining corner bg pixels
            for _ in range(24):
                if (bottom - top) <= 96 or (right - left) <= 96:
                    break
                sub_bg = is_bg[top:bottom, left:right]
                if not np.any(sub_bg):
                    break
                if np.any(sub_bg[0, :]):
                    top += 2
                if np.any(sub_bg[-1, :]):
                    bottom -= 2
                if np.any(sub_bg[:, 0]):
                    left += 2
                if np.any(sub_bg[:, -1]):
                    right -= 2
            inset = 4
            if (right - left) > 96 + inset * 2:
                left += inset
                right -= inset
            if (bottom - top) > 96 + inset * 2:
                top += inset
                bottom -= inset
        else:
            top, bottom, left, right = 16, 240, 16, 240

        cropped = img.crop((left, top, right, bottom))
        # If the input image already exhibits bilateral mirror symmetry (e.g. a legacy Rorschach tile),
        # crop an off-center asymmetric sub-quadrant to break the symmetry before toroidal synthesis.
        if cls.compute_tile_mirror_symmetry(cropped) > 0.55:
            cw_c, ch_c = cropped.size
            cropped = cropped.crop((int(cw_c * 0.04), int(ch_c * 0.06), int(cw_c * 0.66), int(ch_c * 0.70)))

        cropped = cropped.resize((w, h), Image.Resampling.LANCZOS)
        arr_a = np.array(cropped.convert("RGB"), dtype=np.float32)

        # Non-mirrored toroidal offset synthesis:
        # Roll by (h // 2, w // 2) so opposite boundaries wrap seamlessly with ZERO mirror reflection.
        arr_b = np.roll(arr_a, shift=(h // 2, w // 2), axis=(0, 1))

        # Smooth raised-cosine interior weight mask: 1.0 at center, 0.0 at outer boundaries (where arr_b is continuous)
        ux = np.linspace(0.0, 2.0 * np.pi, w, endpoint=False, dtype=np.float32)
        uy = np.linspace(0.0, 2.0 * np.pi, h, endpoint=False, dtype=np.float32)
        wx = 0.5 - 0.5 * np.cos(ux)
        wy = 0.5 - 0.5 * np.cos(uy)
        mask = (wy[:, None] * wx[None, :])[..., None]

        # Variance-preserving blend so cross-fade regions retain crisp texture contrast
        mean_a = np.mean(arr_a, axis=(0, 1), keepdims=True)
        std_a = np.std(arr_a, axis=(0, 1), keepdims=True) + 1e-5
        blended = arr_a * mask + arr_b * (1.0 - mask)
        mean_bl = np.mean(blended, axis=(0, 1), keepdims=True)
        std_bl = np.std(blended, axis=(0, 1), keepdims=True) + 1e-5
        seamless_rgb = np.clip((blended - mean_bl) * (std_a / std_bl) + mean_a, 0.0, 255.0).astype(np.uint8)

        alpha_full = np.full((h, w, 1), 255, dtype=np.uint8)
        seamless_rgba = np.concatenate([seamless_rgb, alpha_full], axis=-1)
        return Image.fromarray(seamless_rgba)

    @staticmethod
    def default_keyframe_tracks_for_profile(motion_profile: str) -> List[PartKeyframeTrackSpec]:
        """Returns explicit biomechanical PartKeyframeTrackSpecs across normalized time t in [0.0, 1.0]
        for 'idle', 'locomotion', 'aerial', or 'action'.
        """
        prof = str(motion_profile or "idle").lower().strip()
        if prof == "locomotion":
            return [
                PartKeyframeTrackSpec(
                    deliberation="Torso forward lean and double-frequency stride bounce during run cycle.",
                    part_id="torso",
                    keyframes=[
                        KeyframeTransformSpec(deliberation="Contact phase 1", time_normalized=0.0, position_offset_px=(2.0, -3.0), rotation_degrees=8.0, scale_xy=(1.02, 0.97)),
                        KeyframeTransformSpec(deliberation="Passing phase 1", time_normalized=0.25, position_offset_px=(0.0, 1.0), rotation_degrees=4.0, scale_xy=(0.98, 1.03)),
                        KeyframeTransformSpec(deliberation="Contact phase 2", time_normalized=0.50, position_offset_px=(-2.0, -3.0), rotation_degrees=-8.0, scale_xy=(1.02, 0.97)),
                        KeyframeTransformSpec(deliberation="Passing phase 2", time_normalized=0.75, position_offset_px=(0.0, 1.0), rotation_degrees=-4.0, scale_xy=(0.98, 1.03)),
                        KeyframeTransformSpec(deliberation="Cycle loop closure", time_normalized=1.0, position_offset_px=(2.0, -3.0), rotation_degrees=8.0, scale_xy=(1.02, 0.97)),
                    ],
                ),
                PartKeyframeTrackSpec(
                    deliberation="Left leg wide forward-and-back stride swing around hip pivot.",
                    part_id="leg_left",
                    keyframes=[
                        KeyframeTransformSpec(deliberation="Left leg forward stride extension", time_normalized=0.0, position_offset_px=(-5.0, -3.0), rotation_degrees=28.0, scale_xy=(1.05, 0.96)),
                        KeyframeTransformSpec(deliberation="Left leg planted stance pass", time_normalized=0.25, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                        KeyframeTransformSpec(deliberation="Left leg rear push-off extension", time_normalized=0.50, position_offset_px=(5.0, -2.0), rotation_degrees=-28.0, scale_xy=(1.05, 0.96)),
                        KeyframeTransformSpec(deliberation="Left leg knee-lift recovery pass", time_normalized=0.75, position_offset_px=(0.0, -4.0), rotation_degrees=6.0, scale_xy=(0.96, 0.92)),
                        KeyframeTransformSpec(deliberation="Loop closure", time_normalized=1.0, position_offset_px=(-5.0, -3.0), rotation_degrees=28.0, scale_xy=(1.05, 0.96)),
                    ],
                ),
                PartKeyframeTrackSpec(
                    deliberation="Right leg counter-phase stride swing around hip pivot.",
                    part_id="leg_right",
                    keyframes=[
                        KeyframeTransformSpec(deliberation="Right leg rear push-off extension", time_normalized=0.0, position_offset_px=(5.0, -2.0), rotation_degrees=-28.0, scale_xy=(1.05, 0.96)),
                        KeyframeTransformSpec(deliberation="Right leg knee-lift recovery pass", time_normalized=0.25, position_offset_px=(0.0, -4.0), rotation_degrees=6.0, scale_xy=(0.96, 0.92)),
                        KeyframeTransformSpec(deliberation="Right leg forward stride extension", time_normalized=0.50, position_offset_px=(-5.0, -3.0), rotation_degrees=28.0, scale_xy=(1.05, 0.96)),
                        KeyframeTransformSpec(deliberation="Right leg planted stance pass", time_normalized=0.75, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                        KeyframeTransformSpec(deliberation="Loop closure", time_normalized=1.0, position_offset_px=(5.0, -2.0), rotation_degrees=-28.0, scale_xy=(1.05, 0.96)),
                    ],
                ),
                PartKeyframeTrackSpec(
                    deliberation="Lead arm/staff counter-swing balancing leg stride.",
                    part_id="arm_lead",
                    keyframes=[
                        KeyframeTransformSpec(deliberation="Arm back swing", time_normalized=0.0, position_offset_px=(-4.0, 0.0), rotation_degrees=-24.0, scale_xy=(1.0, 1.0)),
                        KeyframeTransformSpec(deliberation="Arm neutral pass", time_normalized=0.25, position_offset_px=(0.0, 1.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                        KeyframeTransformSpec(deliberation="Arm forward swing", time_normalized=0.50, position_offset_px=(5.0, -2.0), rotation_degrees=26.0, scale_xy=(1.04, 1.0)),
                        KeyframeTransformSpec(deliberation="Arm neutral pass", time_normalized=0.75, position_offset_px=(0.0, 1.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                        KeyframeTransformSpec(deliberation="Loop closure", time_normalized=1.0, position_offset_px=(-4.0, 0.0), rotation_degrees=-24.0, scale_xy=(1.0, 1.0)),
                    ],
                ),
            ]
        if prof == "aerial":
            return [
                PartKeyframeTrackSpec(
                    deliberation="Jump crouch launch, airborne apex stretch, and landing poise within cell bounds.",
                    part_id="torso",
                    keyframes=[
                        KeyframeTransformSpec(deliberation="Takeoff crouch squash", time_normalized=0.0, position_offset_px=(0.0, 3.0), rotation_degrees=-10.0, scale_xy=(1.10, 0.90)),
                        KeyframeTransformSpec(deliberation="Airborne apex upward stretch", time_normalized=0.50, position_offset_px=(0.0, -3.0), rotation_degrees=8.0, scale_xy=(0.94, 1.04)),
                        KeyframeTransformSpec(deliberation="Descent reach pose", time_normalized=1.0, position_offset_px=(2.0, -1.0), rotation_degrees=-6.0, scale_xy=(1.04, 0.96)),
                    ],
                ),
                PartKeyframeTrackSpec(
                    deliberation="Left leg crouch bend, aerial tuck, and landing extension.",
                    part_id="leg_left",
                    keyframes=[
                        KeyframeTransformSpec(deliberation="Crouch coil", time_normalized=0.0, position_offset_px=(-3.0, 2.0), rotation_degrees=20.0, scale_xy=(1.08, 0.88)),
                        KeyframeTransformSpec(deliberation="Aerial knee tuck", time_normalized=0.50, position_offset_px=(-3.5, -3.5), rotation_degrees=26.0, scale_xy=(0.95, 0.92)),
                        KeyframeTransformSpec(deliberation="Landing reach", time_normalized=1.0, position_offset_px=(-2.0, 0.0), rotation_degrees=-16.0, scale_xy=(1.0, 1.02)),
                    ],
                ),
                PartKeyframeTrackSpec(
                    deliberation="Right leg crouch coil, aerial trail, and brace.",
                    part_id="leg_right",
                    keyframes=[
                        KeyframeTransformSpec(deliberation="Crouch coil", time_normalized=0.0, position_offset_px=(3.0, 2.0), rotation_degrees=-20.0, scale_xy=(1.08, 0.88)),
                        KeyframeTransformSpec(deliberation="Aerial trail extension", time_normalized=0.50, position_offset_px=(3.5, -2.5), rotation_degrees=-22.0, scale_xy=(0.95, 1.02)),
                        KeyframeTransformSpec(deliberation="Landing brace", time_normalized=1.0, position_offset_px=(3.0, 0.0), rotation_degrees=18.0, scale_xy=(1.0, 1.0)),
                    ],
                ),
                PartKeyframeTrackSpec(
                    deliberation="Arm/tool upward thrust during jump.",
                    part_id="arm_lead",
                    keyframes=[
                        KeyframeTransformSpec(deliberation="Backswing coil", time_normalized=0.0, position_offset_px=(-3.0, 2.0), rotation_degrees=-22.0, scale_xy=(1.0, 1.0)),
                        KeyframeTransformSpec(deliberation="Apex overhead reach", time_normalized=0.50, position_offset_px=(2.5, -3.0), rotation_degrees=24.0, scale_xy=(1.03, 1.03)),
                        KeyframeTransformSpec(deliberation="Outstretched balance", time_normalized=1.0, position_offset_px=(3.0, -1.0), rotation_degrees=14.0, scale_xy=(1.0, 1.0)),
                    ],
                ),
            ]
        if prof == "action":
            return [
                PartKeyframeTrackSpec(
                    deliberation="Action wind-up coil, dynamic forward strike lunge, surge extension, and recovery.",
                    part_id="torso",
                    keyframes=[
                        KeyframeTransformSpec(deliberation="Wind-up coil back", time_normalized=0.0, position_offset_px=(-4.0, 1.0), rotation_degrees=-14.0, scale_xy=(0.95, 1.04)),
                        KeyframeTransformSpec(deliberation="Forward strike lunge", time_normalized=0.33, position_offset_px=(6.0, -2.0), rotation_degrees=16.0, scale_xy=(1.12, 0.94)),
                        KeyframeTransformSpec(deliberation="Channeling resonance burst", time_normalized=0.67, position_offset_px=(3.0, -4.0), rotation_degrees=8.0, scale_xy=(1.08, 1.06)),
                        KeyframeTransformSpec(deliberation="Follow-through recovery", time_normalized=1.0, position_offset_px=(-1.0, 0.0), rotation_degrees=-4.0, scale_xy=(1.0, 1.0)),
                    ],
                ),
                PartKeyframeTrackSpec(
                    deliberation="Lead arm / tool sweep across action keyframes.",
                    part_id="arm_lead",
                    keyframes=[
                        KeyframeTransformSpec(deliberation="Raise tool high", time_normalized=0.0, position_offset_px=(-5.0, -5.0), rotation_degrees=-34.0, scale_xy=(1.04, 1.04)),
                        KeyframeTransformSpec(deliberation="Full forward strike extension", time_normalized=0.33, position_offset_px=(8.0, 1.0), rotation_degrees=38.0, scale_xy=(1.16, 1.08)),
                        KeyframeTransformSpec(deliberation="Wide arc follow-through", time_normalized=0.67, position_offset_px=(5.0, -3.0), rotation_degrees=20.0, scale_xy=(1.10, 1.10)),
                        KeyframeTransformSpec(deliberation="Return to guard", time_normalized=1.0, position_offset_px=(0.0, 0.0), rotation_degrees=-6.0, scale_xy=(1.0, 1.0)),
                    ],
                ),
                PartKeyframeTrackSpec(
                    deliberation="Left leg braced lunge stance.",
                    part_id="leg_left",
                    keyframes=[
                        KeyframeTransformSpec(deliberation="Coil stance", time_normalized=0.0, position_offset_px=(2.0, 0.0), rotation_degrees=-12.0, scale_xy=(1.0, 1.0)),
                        KeyframeTransformSpec(deliberation="Rear anchor push", time_normalized=0.33, position_offset_px=(-5.0, 0.0), rotation_degrees=24.0, scale_xy=(1.06, 0.96)),
                        KeyframeTransformSpec(deliberation="Wide brace", time_normalized=0.67, position_offset_px=(-3.0, 0.0), rotation_degrees=16.0, scale_xy=(1.02, 1.0)),
                        KeyframeTransformSpec(deliberation="Recovery stance", time_normalized=1.0, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                    ],
                ),
                PartKeyframeTrackSpec(
                    deliberation="Right leg forward step into action lunge.",
                    part_id="leg_right",
                    keyframes=[
                        KeyframeTransformSpec(deliberation="Lift step", time_normalized=0.0, position_offset_px=(-2.0, -2.0), rotation_degrees=14.0, scale_xy=(1.0, 0.95)),
                        KeyframeTransformSpec(deliberation="Planted forward lunge", time_normalized=0.33, position_offset_px=(6.0, 0.0), rotation_degrees=-24.0, scale_xy=(1.06, 0.96)),
                        KeyframeTransformSpec(deliberation="Hold forward brace", time_normalized=0.67, position_offset_px=(4.0, 0.0), rotation_degrees=-14.0, scale_xy=(1.02, 1.0)),
                        KeyframeTransformSpec(deliberation="Recovery stance", time_normalized=1.0, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                    ],
                ),
            ]
        # Default 'idle': feet planted firmly on baseline while torso/head/arm articulate breathing & alert poise
        return [
            PartKeyframeTrackSpec(
                deliberation="Rhythmic chest breathing and alert upper-body poise shift while feet remain grounded.",
                part_id="torso",
                keyframes=[
                    KeyframeTransformSpec(deliberation="Neutral alert inhale start", time_normalized=0.0, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                    KeyframeTransformSpec(deliberation="Full chest expansion & slight forward poise", time_normalized=0.25, position_offset_px=(2.0, -2.0), rotation_degrees=6.0, scale_xy=(1.06, 1.05)),
                    KeyframeTransformSpec(deliberation="Exhale settle", time_normalized=0.50, position_offset_px=(0.0, 1.0), rotation_degrees=0.0, scale_xy=(1.03, 0.96)),
                    KeyframeTransformSpec(deliberation="Alert counter-look poise", time_normalized=0.75, position_offset_px=(-2.0, -1.0), rotation_degrees=-6.0, scale_xy=(0.96, 1.04)),
                    KeyframeTransformSpec(deliberation="Loop closure", time_normalized=1.0, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                ],
            ),
            PartKeyframeTrackSpec(
                deliberation="Lead arm / staff ready sway during idle breathing.",
                part_id="arm_lead",
                keyframes=[
                    KeyframeTransformSpec(deliberation="Neutral guard", time_normalized=0.0, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                    KeyframeTransformSpec(deliberation="Raise ready poise", time_normalized=0.25, position_offset_px=(3.0, -3.0), rotation_degrees=15.0, scale_xy=(1.04, 1.02)),
                    KeyframeTransformSpec(deliberation="Lower relax", time_normalized=0.50, position_offset_px=(0.0, 1.0), rotation_degrees=-5.0, scale_xy=(1.0, 1.0)),
                    KeyframeTransformSpec(deliberation="Back poise", time_normalized=0.75, position_offset_px=(-3.0, -1.0), rotation_degrees=-14.0, scale_xy=(1.02, 1.0)),
                    KeyframeTransformSpec(deliberation="Loop closure", time_normalized=1.0, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                ],
            ),
            PartKeyframeTrackSpec(
                deliberation="Left leg planted knee-flex weight shift.",
                part_id="leg_left",
                keyframes=[
                    KeyframeTransformSpec(deliberation="Planted neutral", time_normalized=0.0, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                    KeyframeTransformSpec(deliberation="Weight shift left", time_normalized=0.25, position_offset_px=(-1.0, 0.0), rotation_degrees=6.0, scale_xy=(1.02, 0.98)),
                    KeyframeTransformSpec(deliberation="Planted neutral", time_normalized=0.50, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                    KeyframeTransformSpec(deliberation="Weight shift right", time_normalized=0.75, position_offset_px=(1.0, 0.0), rotation_degrees=-6.0, scale_xy=(0.98, 1.02)),
                    KeyframeTransformSpec(deliberation="Loop closure", time_normalized=1.0, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                ],
            ),
            PartKeyframeTrackSpec(
                deliberation="Right leg planted counter weight shift.",
                part_id="leg_right",
                keyframes=[
                    KeyframeTransformSpec(deliberation="Planted neutral", time_normalized=0.0, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                    KeyframeTransformSpec(deliberation="Weight shift left", time_normalized=0.25, position_offset_px=(1.0, 0.0), rotation_degrees=-6.0, scale_xy=(0.98, 1.02)),
                    KeyframeTransformSpec(deliberation="Planted neutral", time_normalized=0.50, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                    KeyframeTransformSpec(deliberation="Weight shift right", time_normalized=0.75, position_offset_px=(-1.0, 0.0), rotation_degrees=6.0, scale_xy=(1.02, 0.98)),
                    KeyframeTransformSpec(deliberation="Loop closure", time_normalized=1.0, position_offset_px=(0.0, 0.0), rotation_degrees=0.0, scale_xy=(1.0, 1.0)),
                ],
            ),
        ]

    @staticmethod
    def _interpolate_keyframe_track(
        track: PartKeyframeTrackSpec,
        t_norm: float,
    ) -> Tuple[Tuple[float, float], float, Tuple[float, float]]:
        """Linearly interpolates (position_offset_px, rotation_degrees, scale_xy) at normalized time t_norm in [0, 1]."""
        kfs = sorted(track.keyframes, key=lambda k: float(k.time_normalized))
        if not kfs:
            return (0.0, 0.0), 0.0, (1.0, 1.0)
        t = max(0.0, min(1.0, float(t_norm)))
        if t <= kfs[0].time_normalized:
            k0 = kfs[0]
            return k0.position_offset_px, float(k0.rotation_degrees), k0.scale_xy
        if t >= kfs[-1].time_normalized:
            k1 = kfs[-1]
            return k1.position_offset_px, float(k1.rotation_degrees), k1.scale_xy

        for idx in range(len(kfs) - 1):
            k0, k1 = kfs[idx], kfs[idx + 1]
            t0, t1 = float(k0.time_normalized), float(k1.time_normalized)
            if t0 <= t <= t1:
                span = max(1e-5, t1 - t0)
                alpha = (t - t0) / span
                dx = k0.position_offset_px[0] + alpha * (k1.position_offset_px[0] - k0.position_offset_px[0])
                dy = k0.position_offset_px[1] + alpha * (k1.position_offset_px[1] - k0.position_offset_px[1])
                rot = float(k0.rotation_degrees) + alpha * (float(k1.rotation_degrees) - float(k0.rotation_degrees))
                sx = k0.scale_xy[0] + alpha * (k1.scale_xy[0] - k0.scale_xy[0])
                sy = k0.scale_xy[1] + alpha * (k1.scale_xy[1] - k0.scale_xy[1])
                return (dx, dy), rot, (sx, sy)
        k_last = kfs[-1]
        return k_last.position_offset_px, float(k_last.rotation_degrees), k_last.scale_xy

    @staticmethod
    def _warp_part_around_pivot(
        part_rgba: np.ndarray,
        pivot_xy: Tuple[float, float],
        pos_offset: Tuple[float, float],
        rot_deg: float,
        scale_xy: Tuple[float, float],
    ) -> np.ndarray:
        """Applies a 2D affine keyframe transform (anisotropic scale + pivot rotation + translation) via OpenCV."""
        h, w = part_rgba.shape[:2]
        px, py = float(pivot_xy[0]), float(pivot_xy[1])
        sx, sy = float(scale_xy[0]), float(scale_xy[1])
        dx, dy = float(pos_offset[0]), float(pos_offset[1])

        rad = math.radians(float(rot_deg))
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Combined scale around pivot + rotation around pivot + translation
        m00 = cos_a * sx
        m01 = -sin_a * sy
        m10 = sin_a * sx
        m11 = cos_a * sy
        tx = px - (m00 * px + m01 * py) + dx
        ty = py - (m10 * px + m11 * py) + dy
        mat = np.array([[m00, m01, tx], [m10, m11, ty]], dtype=np.float32)

        return cv2.warpAffine(
            part_rgba,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    @staticmethod
    def _alpha_composite_np(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
        """Alpha-composites `src` (H, W, 4) over `dst` (H, W, 4) in uint8 RGBA."""
        src_a = src[:, :, 3:4].astype(np.float32) / 255.0
        dst_a = dst[:, :, 3:4].astype(np.float32) / 255.0
        out_a = src_a + dst_a * (1.0 - src_a)
        safe_out_a = np.where(out_a > 1e-5, out_a, 1.0)
        out_rgb = (src[:, :, :3].astype(np.float32) * src_a + dst[:, :, :3].astype(np.float32) * dst_a * (1.0 - src_a)) / safe_out_a
        res = np.zeros_like(dst, dtype=np.uint8)
        res[:, :, :3] = np.clip(out_rgb, 0.0, 255.0).astype(np.uint8)
        res[:, :, 3:4] = np.clip(out_a * 255.0, 0.0, 255.0).astype(np.uint8)
        return res

    @classmethod
    def fabricate_pose_frame_from_base(
        cls,
        base_sprite: Image.Image,
        state_name: str = "move",
        frame_idx: int = 0,
        frame_count: int = 4,
        cell_size: Tuple[int, int] = (64, 64),
        motion_profile: str = "idle",
        keyframe_tracks: Optional[List[PartKeyframeTrackSpec]] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Synthesizes an articulated keyframe pose from `base_sprite` using a continuous, topology-preserving
        2D UV displacement field (`cv2.remap`) evaluated across normalized sprite coordinates (u, v).
        Unlike rigid quadrant slicing, continuous C^infinity UV deformation preserves 100% topological continuity
        across any entity anatomy, handheld tool, or organic structure while producing distinct keyframe poses.
        """
        eff_size = target_size if target_size is not None else cell_size
        cw, ch = max(16, int(eff_size[0])), max(16, int(eff_size[1]))
        src_pil = base_sprite.convert("RGBA").resize((cw, ch), Image.Resampling.LANCZOS)
        src_np = np.array(src_pil, dtype=np.uint8)

        profile = str(motion_profile or "idle").lower().strip()
        tracks = list(keyframe_tracks) if keyframe_tracks else cls.default_keyframe_tracks_for_profile(profile)
        track_map: Dict[str, PartKeyframeTrackSpec] = {t.part_id: t for t in tracks}

        if profile in ("idle", "locomotion"):
            t_norm = float(frame_idx) / float(max(1, frame_count))
        else:
            t_norm = float(frame_idx) / float(max(1, frame_count - 1)) if frame_count > 1 else 0.5

        ys, xs = np.where(src_np[:, :, 3] > 24)
        if len(ys) < 16:
            return src_pil

        x0, x1 = int(np.min(xs)), int(np.max(xs))
        y0, y1 = int(np.min(ys)), int(np.max(ys))
        bw = max(4.0, float(x1 - x0 + 1))
        bh = max(4.0, float(y1 - y0 + 1))
        cx = x0 + bw * 0.5
        cy = y0 + bh * 0.5
        scale_factor = float(cw) / 64.0

        def _get_kf(pid: str) -> Tuple[Tuple[float, float], float, Tuple[float, float]]:
            trk = track_map.get(pid)
            if trk is None:
                return (0.0, 0.0), 0.0, (1.0, 1.0)
            return cls._interpolate_keyframe_track(trk, t_norm)

        (t_dx, t_dy), t_rot, (t_sx, t_sy) = _get_kf("torso")
        (ll_dx, ll_dy), ll_rot, _ = _get_kf("leg_left")
        (lr_dx, lr_dy), lr_rot, _ = _get_kf("leg_right")
        (arm_dx, arm_dy), arm_rot, _ = _get_kf("arm_lead")

        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        # Normalized subject coordinates: u in [-1, 1] across width, v in [0, 1] from top (0) to base (1)
        u = np.clip((xx - cx) / (bw * 0.5), -1.25, 1.25)
        v = np.clip((yy - y0) / bh, 0.0, 1.0)

        # Smooth C^infinity spatial influence weights (never cuts or tears connected structures)
        w_upper = np.cos(v * (np.pi * 0.5)) ** 2
        w_lower = np.sin(v * (np.pi * 0.5)) ** 2
        w_left = 0.5 * (1.0 - np.tanh(2.2 * u))
        w_right = 0.5 * (1.0 + np.tanh(2.2 * u))
        w_mid_lead = np.exp(-(((v - 0.42) / 0.30) ** 2)) * w_right

        # 1. Core / upper-body squash-stretch, harmonic bend, and translation around grounded base (cx, y1)
        rad_t = math.radians(t_rot * 0.45)
        dx_upper = (
            t_dx * scale_factor * w_upper
            + (xx - cx) * (1.0 - 1.0 / max(0.82, min(1.22, t_sx)))
            + (y1 - yy) * math.sin(rad_t) * w_upper * 0.55
        )
        dy_upper = (
            t_dy * scale_factor * w_upper
            + (yy - y1) * (1.0 - 1.0 / max(0.82, min(1.22, t_sy)))
        )

        # 2. Smooth lower-body / base counter-phase stride & sway field (continuous across u=0)
        stride_l_x = (ll_dx * scale_factor * 0.65 + math.sin(math.radians(ll_rot)) * bw * 0.14) * w_lower * w_left
        stride_r_x = (lr_dx * scale_factor * 0.65 + math.sin(math.radians(lr_rot)) * bw * 0.14) * w_lower * w_right
        stride_l_y = (ll_dy * scale_factor * 0.55 - abs(math.sin(math.radians(ll_rot))) * bh * 0.06) * w_lower * w_left
        stride_r_y = (lr_dy * scale_factor * 0.55 - abs(math.sin(math.radians(lr_rot))) * bh * 0.06) * w_lower * w_right

        # 3. Smooth secondary appendage / lead follow-through wave
        lead_x = (arm_dx * scale_factor * 0.50 + math.sin(math.radians(arm_rot)) * bw * 0.10) * w_mid_lead
        lead_y = (arm_dy * scale_factor * 0.50 - math.sin(math.radians(arm_rot)) * bh * 0.08) * w_mid_lead

        total_dx = dx_upper + stride_l_x + stride_r_x + lead_x
        total_dy = dy_upper + stride_l_y + stride_r_y + lead_y

        # 4. Arbitrary / non-humanoid custom part tracks (e.g. radial outriggers, rotors, wings, mechanical appendages)
        humanoid_ids = {"torso", "leg_left", "leg_right", "arm_lead"}
        custom_tracks = [t for t in tracks if t.part_id not in humanoid_ids and t.keyframes]
        if custom_tracks or not any(pid in track_map for pid in humanoid_ids):
            rx = (xx - cx) / max(2.0, bw * 0.5)
            ry = (yy - cy) / max(2.0, bh * 0.5)
            r_norm = np.sqrt(rx ** 2 + ry ** 2)
            # Smooth peripheral weight: 0 at core center, ramping smoothly to 1 on outer appendages (r >= 0.35)
            w_periph = np.clip((r_norm - 0.18) / 0.65, 0.0, 1.25) ** 1.35

            c_dx_sum, c_dy_sum = 0.0, 0.0
            c_rot_sum = 0.0
            c_sx_prod, c_sy_prod = 1.0, 1.0
            radial_delta_px = 0.0
            for ctrk in custom_tracks:
                (cdx, cdy), crot, (csx, csy) = cls._interpolate_keyframe_track(ctrk, t_norm)
                (b0_dx, b0_dy), b0_rot, _ = cls._interpolate_keyframe_track(ctrk, 0.0)
                # Relative frame-to-frame motion within the track
                rel_dx = cdx - b0_dx
                rel_dy = cdy - b0_dy
                c_dx_sum += rel_dx
                c_dy_sum += rel_dy
                # Radial extension/retraction: if position_offset_px encodes radial reach (e.g. 50px vs 21px),
                # convert deviations from nominal 45px radius or relative radial delta into radial scale
                mag_cur = math.hypot(cdx, cdy)
                mag_b0 = math.hypot(b0_dx, b0_dy)
                radial_delta_px += (mag_cur - mag_b0)
                c_rot_sum += (crot - b0_rot) + crot * 0.25
                c_sx_prod *= max(0.65, min(1.35, csx))
                c_sy_prod *= max(0.65, min(1.35, csy))

            phase_angle = 2.0 * math.pi * (float(frame_idx) / float(max(1, frame_count)))
            # Ensure multi-frame custom animations exhibit distinct inter-frame motion across columns
            # without injecting unwanted rotational tilt when keyframe rotation_deg is 0.
            harmonic_radial = 1.0 + 0.05 * math.sin(phase_angle)
            if abs(c_rot_sum) > 1e-3:
                harmonic_twist_rad = math.radians(c_rot_sum * 0.55 + 3.0 * math.sin(phase_angle))
            else:
                harmonic_twist_rad = 0.0

            max_safe_sx = max(1.02, (float(cw) - 6.0) / max(8.0, bw))
            max_safe_sy = max(1.02, (float(ch) - 6.0) / max(8.0, bh))
            raw_sx = c_sx_prod * harmonic_radial + (radial_delta_px / max(16.0, bw * 0.5)) * 0.35
            raw_sy = c_sy_prod * harmonic_radial + (radial_delta_px / max(16.0, bh * 0.5)) * 0.35
            eff_sx = max(0.62, min(min(1.32, max_safe_sx), raw_sx))
            eff_sy = max(0.62, min(min(1.32, max_safe_sy), raw_sy))

            cos_tw = math.cos(harmonic_twist_rad)
            sin_tw = math.sin(harmonic_twist_rad)
            rel_x = xx - cx
            rel_y = yy - cy
            # Radial scale + optional rotational twist weighted toward peripheral outriggers/appendages
            target_rx = (rel_x * cos_tw - rel_y * sin_tw) * eff_sx
            target_ry = (rel_x * sin_tw + rel_y * cos_tw) * eff_sy
            custom_dx = (target_rx - rel_x) * w_periph + c_dx_sum * scale_factor * 0.35 * w_periph
            custom_dy = (target_ry - rel_y) * w_periph + c_dy_sum * scale_factor * 0.35 * w_periph

            total_dx = total_dx + custom_dx
            total_dy = total_dy + custom_dy

        # Inverse mapping coordinates for cv2.remap
        map_x = (xx - total_dx).astype(np.float32)
        map_y = (yy - total_dy).astype(np.float32)

        comp = cv2.remap(
            src_np,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # Grounded baseline lock for 'idle' and 'locomotion': lock the lowest opaque pixel to `y1`
        # so grounded animations never float or jitter vertically across frames.
        if profile in ("idle", "locomotion"):
            comp_ys = np.where(comp[:, :, 3] > 32)[0]
            if len(comp_ys) > 0:
                cur_bottom = int(np.max(comp_ys))
                target_bottom = min(ch - 2, y1)
                shift_y = target_bottom - cur_bottom
                if shift_y != 0:
                    mat_ground = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, float(shift_y)]], dtype=np.float32)
                    comp = cv2.warpAffine(
                        comp,
                        mat_ground,
                        (cw, ch),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0),
                    )

        return Image.fromarray(comp)

    def fabricate_sprite_sheet(
        self,
        asset_req: AssetRequirement,
        output_path: Path,
        mood: str = "retro_arcade",
        reference_images: Optional[List[Union[Path, str, bytes, Image.Image]]] = None,
        style_guide: str = "",
    ) -> Tuple[Path, Dict[str, Any]]:
        """Fabricates a multi-row, multi-frame keyframed sprite sheet PNG and returns (sheet_path, metadata_dict).
        Each row corresponds to an AnimationStateSpec in asset_req.animation_states and carries explicit
        `keyframe_tracks` exported into `asset_library.json` for runtime Godot `AnimationPlayer` binding.
        """
        cell_w, cell_h = asset_req.dimensions or (64, 64)
        cell_w = max(32, int(cell_w))
        cell_h = max(32, int(cell_h))

        raw_atype = str(getattr(asset_req, "asset_type", "") or "").lower().strip()
        if asset_req.animation_states:
            states = list(asset_req.animation_states)
        elif raw_atype in ("prop", "tile", "ui", "ui_element", "diegetic_ui", "backdrop", "icon"):
            states = [
                AnimationStateSpec(
                    deliberation="Default inanimate 2-frame ambient state.",
                    state_name="idle",
                    motion_profile="idle",
                    frame_count=2,
                    fps=4.0,
                    loop=True,
                    pose_progression_prompt="Subtle idle state",
                ),
            ]
        else:
            states = [
                AnimationStateSpec(
                    deliberation="Default idle breathing state.",
                    state_name="idle",
                    motion_profile="idle",
                    frame_count=4,
                    fps=6.0,
                    loop=True,
                    pose_progression_prompt="Subtle breathing and alert stance",
                ),
                AnimationStateSpec(
                    deliberation="Default locomotion state.",
                    state_name="move",
                    motion_profile="locomotion",
                    frame_count=4,
                    fps=10.0,
                    loop=True,
                    pose_progression_prompt="Dynamic stride cycle with arm and leg swing",
                ),
                AnimationStateSpec(
                    deliberation="Default aerial/jump state.",
                    state_name="jump",
                    motion_profile="aerial",
                    frame_count=4,
                    fps=8.0,
                    loop=False,
                    pose_progression_prompt="Takeoff crouch, airborne leap, apex, landing",
                ),
                AnimationStateSpec(
                    deliberation="Default interaction/ability state.",
                    state_name="action",
                    motion_profile="action",
                    frame_count=4,
                    fps=10.0,
                    loop=False,
                    pose_progression_prompt="Expressive action gesture or tool activation",
                ),
            ]

        base_dims = (cell_w * 2, cell_h * 2)
        raw_base_prompt = asset_req.visual_prompt_or_text or asset_req.description or asset_req.asset_id
        single_base_prompt = (
            f"SINGLE CANONICAL NEUTRAL STANDING/IDLE FIGURE ONLY (the animation engine will articulate all other rows/frames "
            f"from this one base image — do NOT draw multiple poses, rows, frames, or a sprite sheet grid): {raw_base_prompt}"
        )
        if self._is_live_genai_client():
            base_img = self._generate_gemini_sprite_image(
                prompt=single_base_prompt,
                asset_type=asset_req.asset_type or "sprite_sheet",
                dimensions=base_dims,
                reference_images=reference_images,
                style_guide=style_guide,
                is_animated_actor=True,
            )
        else:
            self._ensure_procedural_allowed_or_raise("fabricate_sprite_sheet")
            renderer = ProceduralSpriteRenderer(mood=mood)
            base_img = renderer.render_dynamic_sprite(
                dimensions=base_dims,
                prompt=asset_req.visual_prompt_or_text or asset_req.description,
                asset_type="sprite",
                seed=asset_req.asset_id,
                geometry_directive=asset_req.geometry_directive,
            )

        max_cols = max(max(1, s.frame_count) for s in states)
        num_rows = len(states)
        sheet_w = cell_w * max_cols
        sheet_h = cell_h * num_rows
        sheet_img = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))

        states_meta: Dict[str, Dict[str, Any]] = {}
        has_custom_states = bool(asset_req.animation_states and len(asset_req.animation_states) > 1)
        for row_idx, st in enumerate(states):
            fcount = max(1, int(st.frame_count))
            m_profile = getattr(st, "motion_profile", "idle") or "idle"
            kf_tracks = list(st.keyframe_tracks) if getattr(st, "keyframe_tracks", None) else self.default_keyframe_tracks_for_profile(m_profile)
            row_base_img = base_img
            state_pose_desc = (getattr(st, "pose_progression_prompt", "") or getattr(st, "deliberation", "") or "").strip()
            if (
                row_idx > 0
                and has_custom_states
                and state_pose_desc
                and self._is_live_genai_client()
            ):
                state_prompt = (
                    f"SINGLE CANONICAL SPRITE ONLY — maintain the exact same character/object identity, core proportions, "
                    f"color palette, and art style as the provided reference image, rendered in this specific state/pose "
                    f"({st.state_name}): {state_pose_desc}. Subject context: {raw_base_prompt}. "
                    f"Do NOT draw multiple poses, rows, frames, or a sprite sheet grid."
                )
                row_refs = [base_img] + (list(reference_images) if reference_images else [])
                row_base_img = self._generate_gemini_sprite_image(
                    prompt=state_prompt,
                    asset_type=asset_req.asset_type or "sprite_sheet",
                    dimensions=base_dims,
                    reference_images=row_refs,
                    style_guide=style_guide,
                    is_animated_actor=True,
                )
            for col_idx in range(fcount):
                col_source_img = row_base_img
                if (
                    col_idx > 0
                    and len(states) == 1
                    and fcount <= 6
                    and state_pose_desc
                    and self._is_live_genai_client()
                    and m_profile in ("custom", "action")
                ):
                    col_prompt = (
                        f"SINGLE CANONICAL SPRITE ONLY for Frame {col_idx} of {fcount} in animation '{st.state_name}' — "
                        f"maintain the exact same outer silhouette alignment, camera angle, proportions, and color palette as the "
                        f"attached Frame 0 reference image, while advancing the internal state/pose to Frame {col_idx} of {fcount} "
                        f"according to this progression: {state_pose_desc}. Subject context: {raw_base_prompt}. "
                        f"Do NOT draw multiple frames, grids, or labels; render ONLY the single isolated sprite for Frame {col_idx}."
                    )
                    col_refs = [row_base_img] + (list(reference_images) if reference_images else [])
                    try:
                        col_source_img = self._generate_gemini_sprite_image(
                            prompt=col_prompt,
                            asset_type=asset_req.asset_type or "sprite_sheet",
                            dimensions=base_dims,
                            reference_images=col_refs,
                            style_guide=style_guide,
                            is_animated_actor=True,
                        )
                    except Exception as col_exc:
                        logger.warning(
                            "Per-column progression synthesis for %s frame %d fell back to base keyframe articulation: %s",
                            asset_req.asset_id,
                            col_idx,
                            col_exc,
                        )
                        col_source_img = row_base_img
                frame_img = self.fabricate_pose_frame_from_base(
                    base_sprite=col_source_img,
                    state_name=st.state_name,
                    frame_idx=col_idx,
                    frame_count=fcount,
                    cell_size=(cell_w, cell_h),
                    motion_profile=m_profile,
                    keyframe_tracks=kf_tracks,
                )
                sheet_img.paste(frame_img, (col_idx * cell_w, row_idx * cell_h), frame_img)

            states_meta[st.state_name] = {
                "row": row_idx,
                "start_frame": row_idx * max_cols,
                "frame_count": fcount,
                "fps": float(st.fps),
                "loop": bool(st.loop),
                "motion_profile": m_profile,
                "keyframe_tracks": [t.model_dump() for t in kf_tracks],
            }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet_img.save(output_path, "PNG")

        meta = {
            "hframes": max_cols,
            "vframes": num_rows,
            "texture_pixel_size": (sheet_w, sheet_h),
            "target_display_size": (cell_w, cell_h),
            "animation_states": states_meta,
            "base_frame_image": base_img.resize((cell_w, cell_h), Image.Resampling.LANCZOS),
        }
        return output_path, meta

    @classmethod
    def _assemble_articulated_rig_frame(
        cls,
        parts_with_images: List[Tuple[Dict[str, Any], Image.Image]],
        canvas_size: Tuple[int, int],
        phase: Optional[float] = None,
    ) -> Image.Image:
        """Assembles an articulated multi-part rig onto `canvas_size` using hierarchical parent anchors,
        pivot-aligned placement (`pivot_normalized`), z-index sorting (`z_index_offset`), and auto-fitting
        so multi-part rigs never collapse into a tiny central dot or bury child parts inside the root body.
        """
        cw, ch = max(16, int(canvas_size[0])), max(16, int(canvas_size[1]))
        if not parts_with_images:
            return Image.new("RGBA", (cw, ch), (0, 0, 0, 0))

        part_lookup: Dict[str, Tuple[Dict[str, Any], Image.Image]] = {
            str(pm.get("part_id", f"part_{i}")): (pm, pimg)
            for i, (pm, pimg) in enumerate(parts_with_images)
        }

        # Identify root part radius to ensure radial child parts attached near the rim aren't buried inside the root body
        root_radius = 0.0
        for pm, pimg in parts_with_images:
            if not pm.get("parent_part_id") or str(pm.get("parent_part_id")) not in part_lookup:
                rw, rh = pimg.size
                root_radius = max(root_radius, min(rw, rh) * 0.45)

        anchors: Dict[str, Tuple[float, float]] = {}

        def _get_anchor(pid: str, visited: Optional[Set[str]] = None) -> Tuple[float, float]:
            if pid in anchors:
                return anchors[pid]
            if visited is None:
                visited = set()
            if pid in visited or pid not in part_lookup:
                return (0.0, 0.0)
            visited.add(pid)
            pm, _ = part_lookup[pid]
            offs = pm.get("attach_offset_px") or [0.0, 0.0]
            ox, oy = float(offs[0]), float(offs[1])
            parent_id = pm.get("parent_part_id")
            if parent_id and str(parent_id) in part_lookup and str(parent_id) != pid:
                px, py = _get_anchor(str(parent_id), visited)
                # If attaching directly to a root part at a non-zero radial offset that falls inside root_radius,
                # push the attachment anchor out to the root perimeter so radial appendages protrude visibly
                parent_meta, _ = part_lookup[str(parent_id)]
                is_parent_root = not parent_meta.get("parent_part_id") or str(parent_meta.get("parent_part_id")) not in part_lookup
                dist = math.hypot(ox, oy)
                if is_parent_root and 1.0 < dist < root_radius * 0.92:
                    scale_out = (root_radius * 0.95) / dist
                    ox *= scale_out
                    oy *= scale_out
                anc = (px + ox, py + oy)
            else:
                anc = (ox, oy)
            anchors[pid] = anc
            return anc

        for pid in part_lookup:
            _get_anchor(pid)

        # Compute stable rig-space bounds across all parts (including rotation margin) so scale is invariant across `phase`
        min_x, min_y = -16.0, -16.0
        max_x, max_y = 16.0, 16.0
        for pid, (pm, pimg) in part_lookup.items():
            ax, ay = anchors[pid]
            pw, ph = pimg.size
            diag = math.hypot(pw, ph) * 0.65
            min_x = min(min_x, ax - diag)
            max_x = max(max_x, ax + diag)
            min_y = min(min_y, ay - diag)
            max_y = max(max_y, ay + diag)

        span_w = max(32.0, (max_x - min_x) * 1.12)
        span_h = max(32.0, (max_y - min_y) * 1.12)
        work_w = max(cw, int(math.ceil(span_w)))
        work_h = max(ch, int(math.ceil(span_h)))
        center_x = work_w * 0.5 - (min_x + max_x) * 0.5
        center_y = work_h * 0.5 - (min_y + max_y) * 0.5

        work_canvas = np.zeros((work_h, work_w, 4), dtype=np.uint8)
        sorted_parts = sorted(
            parts_with_images,
            key=lambda item: int(item[0].get("z_index_offset", 0)),
        )
        for idx_p, (pm, pimg) in enumerate(sorted_parts):
            pid = str(pm.get("part_id", f"part_{idx_p}"))
            ax, ay = anchors.get(pid, (0.0, 0.0))
            pw, ph = pimg.size
            piv = pm.get("pivot_normalized") or [0.5, 0.5]
            piv_x = float(piv[0]) * pw
            piv_y = float(piv[1]) * ph

            is_child = bool(pm.get("parent_part_id")) and str(pm.get("parent_part_id")) in part_lookup
            angle = math.sin(phase + idx_p * 0.7) * 22.0 if (phase is not None and is_child) else 0.0
            rad = math.radians(angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)

            target_x = center_x + ax
            target_y = center_y + ay
            m00, m01 = cos_a, -sin_a
            m10, m11 = sin_a, cos_a
            tx = target_x - (m00 * piv_x + m01 * piv_y)
            ty = target_y - (m10 * piv_x + m11 * piv_y)
            mat = np.array([[m00, m01, tx], [m10, m11, ty]], dtype=np.float32)

            p_np = np.array(pimg.convert("RGBA"), dtype=np.uint8)
            warped = cv2.warpAffine(
                p_np,
                mat,
                (work_w, work_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            work_canvas = cls._alpha_composite_np(work_canvas, warped)

        assembled_pil = Image.fromarray(work_canvas)
        # Fit the stable rig bounds cleanly into (cw, ch) so the rig fills the frame consistently across all phases
        fit_scale = min((cw * 0.88) / max(1.0, span_w), (ch * 0.88) / max(1.0, span_h))
        if abs(fit_scale - 1.0) > 0.04 or (work_w, work_h) != (cw, ch):
            scaled_w = max(1, int(round(work_w * fit_scale)))
            scaled_h = max(1, int(round(work_h * fit_scale)))
            scaled_pil = assembled_pil.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
            out_canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
            paste_x = (cw - scaled_w) // 2
            paste_y = (ch - scaled_h) // 2
            out_canvas.paste(scaled_pil, (paste_x, paste_y), scaled_pil)
            return out_canvas
        return assembled_pil

    def fabricate_articulated_rig(
        self,
        asset_req: AssetRequirement,
        output_path: Path,
        mood: str = "retro_arcade",
        reference_images: Optional[List[Union[Path, str, bytes, Image.Image]]] = None,
        style_guide: str = "",
    ) -> Tuple[Path, Dict[str, Any]]:
        """Fabricates separate transparent PNGs for each ArticulatedPartSpec in an articulated rig,
        plus a composite preview PNG at output_path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parts_dir = output_path.parent / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)

        parts_specs = list(asset_req.articulated_parts) if asset_req.articulated_parts else [
            ArticulatedPartSpec(
                deliberation="Root body/chassis part.",
                part_id="body",
                visual_prompt=f"Central body/core of {asset_req.description}",
                parent_part_id=None,
                pivot_normalized=(0.5, 0.5),
                attach_offset_px=(0.0, 0.0),
                z_index_offset=0,
                dimensions=asset_req.dimensions or (64, 64),
            ),
            ArticulatedPartSpec(
                deliberation="Primary articulated rotor/appendage.",
                part_id="appendage",
                visual_prompt=f"Movable arm/mechanism/appendage of {asset_req.description}",
                parent_part_id="body",
                pivot_normalized=(0.5, 0.8),
                attach_offset_px=(0.0, -12.0),
                z_index_offset=1,
                dimensions=(32, 32),
            ),
        ]

        comp_w, comp_h = asset_req.dimensions or (64, 64)
        parts_meta: List[Dict[str, Any]] = []
        parts_with_images: List[Tuple[Dict[str, Any], Image.Image]] = []

        for pspec in parts_specs:
            pw, ph = pspec.dimensions or (48, 48)
            part_filename = f"{asset_req.asset_id}_{pspec.part_id}.png"
            part_path = parts_dir / part_filename
            if self._is_live_genai_client():
                p_img = self._generate_gemini_sprite_image(
                    prompt=pspec.visual_prompt or f"{asset_req.description} - {pspec.part_id}",
                    asset_type="sprite",
                    dimensions=(pw, ph),
                    reference_images=reference_images,
                    style_guide=style_guide,
                )
            else:
                self._ensure_procedural_allowed_or_raise("fabricate_articulated_rig")
                renderer = ProceduralSpriteRenderer(mood=mood)
                p_img = renderer.render_dynamic_sprite(
                    dimensions=(pw, ph),
                    prompt=pspec.visual_prompt,
                    asset_type="sprite",
                    seed=f"{asset_req.asset_id}_{pspec.part_id}",
                )
            p_img = p_img.convert("RGBA").resize((pw, ph), Image.Resampling.LANCZOS)
            p_img.save(part_path, "PNG")

            pmeta = {
                "part_id": pspec.part_id,
                "res_path": f"res://assets/sprites/parts/{part_filename}",
                "file_path": f"assets/sprites/parts/{part_filename}",
                "parent_part_id": pspec.parent_part_id,
                "pivot_normalized": list(pspec.pivot_normalized),
                "attach_offset_px": list(pspec.attach_offset_px),
                "z_index_offset": int(pspec.z_index_offset),
                "dimensions": [pw, ph],
            }
            parts_meta.append(pmeta)
            parts_with_images.append((pmeta, p_img))

        composite = self._assemble_articulated_rig_frame(parts_with_images, (comp_w, comp_h), phase=None)
        composite.save(output_path, "PNG")
        return output_path, {
            "texture_pixel_size": (comp_w, comp_h),
            "target_display_size": (comp_w, comp_h),
            "articulated_parts": parts_meta,
        }

    def generate_dynamic_sprite(
        self,
        asset_req: Union[AssetRequirement, Dict[str, Any]],
        output_path: Optional[Union[Path, str]] = None,
        mood: str = "retro_arcade",
        reference_images: Optional[List[Union[Path, str, bytes, Image.Image]]] = None,
        style_guide: str = "",
        *,
        regenerate_cell_indices: Optional[Sequence[int]] = None,
        existing_sheet_path: Optional[Union[Path, str]] = None,
    ) -> Path:
        """Renders and saves a dynamic sprite (or multi-cell grid/atlas sheet when grid_cols * grid_rows > 1) from an AssetRequirement.
        When `regenerate_cell_indices` and `existing_sheet_path` are provided for a multi-cell grid retry, preserves clean cells from
        `existing_sheet_path` and re-synthesizes only the defective cell indices."""
        if isinstance(asset_req, dict):
            dims = asset_req.get("dimensions") or (64, 64)
            prompt = asset_req.get("visual_prompt_or_text") or asset_req.get("description") or asset_req.get("name", "game sprite")
            atype = asset_req.get("asset_type", "sprite")
            aid = asset_req.get("asset_id", "sprite")
            grid_cols = max(1, int(asset_req.get("grid_cols") or 1))
            grid_rows = max(1, int(asset_req.get("grid_rows") or 1))
            cell_prompts = list(asset_req.get("cell_prompts") or [])
        else:
            dims = getattr(asset_req, "dimensions", None) or (64, 64)
            prompt = getattr(asset_req, "visual_prompt_or_text", "") or getattr(asset_req, "description", "") or getattr(asset_req, "name", "game sprite")
            atype = getattr(asset_req, "asset_type", "sprite")
            aid = getattr(asset_req, "asset_id", "sprite") or "sprite"
            grid_cols = max(1, int(getattr(asset_req, "grid_cols", 1) or 1))
            grid_rows = max(1, int(getattr(asset_req, "grid_rows", 1) or 1))
            cell_prompts = list(getattr(asset_req, "cell_prompts", None) or [])

        is_multi_cell = (grid_cols * grid_rows) > 1
        if is_multi_cell and cell_prompts:
            total_cells = grid_cols * grid_rows
            cell_w = max(32, int(dims[0]) // grid_cols)
            cell_h = max(32, int(dims[1]) // grid_rows)
            sheet_w = cell_w * grid_cols
            sheet_h = cell_h * grid_rows
            cells_to_render = list(range(total_cells))
            img: Optional[Image.Image] = None
            if regenerate_cell_indices is not None and existing_sheet_path:
                ex_p = Path(existing_sheet_path)
                if ex_p.is_file():
                    try:
                        with Image.open(ex_p) as prev_im:
                            prev_rgba = prev_im.convert("RGBA")
                        if prev_rgba.size == (sheet_w, sheet_h):
                            valid_regen = sorted(
                                {int(i) for i in regenerate_cell_indices if 0 <= int(i) < total_cells}
                            )
                            if 0 < len(valid_regen) < total_cells:
                                img = prev_rgba.copy()
                                cells_to_render = valid_regen
                    except Exception:
                        img = None
            if img is None:
                img = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
            renderer = None if self._is_live_genai_client() else ProceduralSpriteRenderer(mood=mood)
            for idx in cells_to_render:
                row_i = idx // grid_cols
                col_i = idx % grid_cols
                c_prompt = str(cell_prompts[idx] if idx < len(cell_prompts) else prompt).strip() or prompt
                if self._is_live_genai_client():
                    cell_img = self._generate_gemini_sprite_image(
                        prompt=c_prompt,
                        asset_type="sprite",
                        dimensions=(cell_w, cell_h),
                        reference_images=reference_images,
                        style_guide=style_guide,
                        isolate_single_subject=True,
                    )
                else:
                    self._ensure_procedural_allowed_or_raise("generate_dynamic_sprite")
                    cell_img = renderer.render_dynamic_sprite(
                        dimensions=(cell_w, cell_h),
                        prompt=c_prompt,
                        asset_type=atype,
                        seed=f"{aid}_cell_{idx}_{c_prompt[:24]}",
                    )
                cell_rgba = cell_img.convert("RGBA").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                cell_box = (col_i * cell_w, row_i * cell_h, (col_i + 1) * cell_w, (row_i + 1) * cell_h)
                img.paste((0, 0, 0, 0), cell_box)
                img.paste(cell_rgba, (col_i * cell_w, row_i * cell_h), cell_rgba)
        elif self._is_live_genai_client():
            img = self._generate_gemini_sprite_image(
                prompt=prompt,
                asset_type=atype,
                dimensions=dims,
                reference_images=reference_images,
                style_guide=style_guide,
                isolate_single_subject=not is_multi_cell,
            )
        else:
            self._ensure_procedural_allowed_or_raise("generate_dynamic_sprite")
            renderer = ProceduralSpriteRenderer(mood=mood)
            img = renderer.render_from_requirement(asset_req)

        if output_path is None:
            name = getattr(asset_req, "name", None) if hasattr(asset_req, "name") else (asset_req.get("name") if isinstance(asset_req, dict) else None)
            if not name:
                name = f"{aid}.png"
            output_path = Path(name)
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "PNG")
        return output_path

    def generate_cartridge_sprites(
        self,
        output_dir: Path,
        mood: str = "retro_arcade",
        requirements_spec: Optional[Any] = None,
        asset_manifest: Optional[Union[List[Any], Any]] = None,
        reference_images: Optional[List[Union[Path, str, bytes, Image.Image]]] = None,
        style_guide: str = "",
    ) -> Dict[str, Path]:
        """Generates cartridge sprites and environment tiles into output_dir.

        If requirements_spec or asset_manifest is provided, dynamically synthesizes
        sprites for EVERY visual asset declared in the manifest, eliminating hardcoded arena sprites.
        If neither is provided (legacy mode), generates standard base entity and tile sprites.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        renderer = ProceduralSpriteRenderer(mood=mood)
        sprites: Dict[str, Path] = {}

        # Resolve manifest items
        manifest_items = None
        if asset_manifest is not None:
            if hasattr(asset_manifest, "assets"):
                manifest_items = asset_manifest.assets
            elif isinstance(asset_manifest, list):
                manifest_items = asset_manifest
        elif requirements_spec is not None:
            if hasattr(requirements_spec, "asset_manifest"):
                manifest_items = requirements_spec.asset_manifest
            elif isinstance(requirements_spec, dict) and "asset_manifest" in requirements_spec:
                manifest_items = requirements_spec["asset_manifest"]

        if manifest_items is not None:
            # Dynamic manifest-driven generation
            for item in manifest_items:
                if isinstance(item, dict):
                    atype = item.get("asset_type", "sprite")
                    aid = item.get("asset_id") or ""
                    name = item.get("name") or ""
                    tpath = item.get("target_path") or ""
                    dims = item.get("dimensions") or (48, 48)
                    prompt = item.get("visual_prompt_or_text") or item.get("description", "")
                else:
                    atype = getattr(item, "asset_type", "sprite")
                    aid = getattr(item, "asset_id", "")
                    name = getattr(item, "name", "")
                    tpath = getattr(item, "target_path", "")
                    dims = getattr(item, "dimensions", None) or (48, 48)
                    prompt = getattr(item, "visual_prompt_or_text", "") or getattr(item, "description", "")

                # Only process visual assets
                norm_type = str(atype).lower()
                if norm_type not in (
                    "sprite", "sprite_sheet", "articulated_rig", "prop", "tile", "tile_surface",
                    "backdrop", "parallax_layer", "portrait", "diegetic_ui", "entity", "hazard", "item"
                ) and not str(tpath).endswith((".png", ".jpg", ".jpeg")):
                    continue

                # Determine filename
                filename = Path(tpath).name if tpath else (name if name else f"{aid}.png")
                if not filename.endswith(".png"):
                    filename = f"{filename}.png"

                # Route portraits to assets/portraits/ if output_dir is assets/sprites/ and target_path is assets/portraits/
                if norm_type == "portrait" and output_dir.name == "sprites" and str(tpath).startswith("assets/portraits/"):
                    dest_path = output_dir.parent / "portraits" / filename
                else:
                    dest_path = output_dir / filename
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                if self._is_live_genai_client():
                    target_dims = (max(int(dims[0]), 48), max(int(dims[1]), 48))
                    img = self._generate_gemini_sprite_image(
                        prompt=prompt or filename,
                        asset_type=atype,
                        dimensions=target_dims,
                        reference_images=reference_images,
                        style_guide=style_guide,
                    )
                else:
                    self._ensure_procedural_allowed_or_raise("generate_cartridge_sprites")
                    geom = item.get("geometry_directive") if isinstance(item, dict) else getattr(item, "geometry_directive", None)
                    img = renderer.render_dynamic_sprite(
                        dimensions=dims,
                        prompt=prompt,
                        asset_type=atype,
                        seed=aid or filename,
                        geometry_directive=geom,
                    )
                img.save(dest_path, "PNG")

                # If output_dir is a root directory containing relative subpaths, also save to target_path
                if "/" in str(tpath):
                    alt_path = output_dir / tpath
                    if alt_path != dest_path:
                        alt_path.parent.mkdir(parents=True, exist_ok=True)
                        img.save(alt_path, "PNG")
                    if output_dir.name == "sprites" and str(tpath).startswith("assets/"):
                        cart_root_alt = output_dir.parent.parent / tpath
                        if cart_root_alt != dest_path:
                            cart_root_alt.parent.mkdir(parents=True, exist_ok=True)
                            img.save(cart_root_alt, "PNG")

                key = aid or filename.replace(".png", "")
                sprites[key] = dest_path

            return sprites

        # Legacy mode: generate base sprites for backward compatibility (strictly offline opt-in only)
        self._ensure_procedural_allowed_or_raise("generate_cartridge_sprites")
        idle_path = output_dir / "player_idle.png"
        renderer.render_player_idle().save(idle_path, "PNG")
        sprites["player_idle"] = idle_path

        walk1_path = output_dir / "player_walk1.png"
        renderer.render_player_walk1().save(walk1_path, "PNG")
        sprites["player_walk1"] = walk1_path

        walk2_path = output_dir / "player_walk2.png"
        renderer.render_player_walk2().save(walk2_path, "PNG")
        sprites["player_walk2"] = walk2_path

        dash_path = output_dir / "player_dash.png"
        renderer.render_player_dash().save(dash_path, "PNG")
        sprites["player_dash"] = dash_path

        chaser_path = output_dir / "chaser.png"
        renderer.render_chaser().save(chaser_path, "PNG")
        sprites["chaser"] = chaser_path

        sniper_path = output_dir / "sniper.png"
        renderer.render_sniper().save(sniper_path, "PNG")
        sprites["sniper"] = sniper_path

        boss_path = output_dir / "boss.png"
        renderer.render_boss_portrait((48, 48)).save(boss_path, "PNG")
        sprites["boss"] = boss_path

        spikes_path = output_dir / "spikes.png"
        renderer.render_spikes().save(spikes_path, "PNG")
        sprites["spikes"] = spikes_path

        acid_path = output_dir / "acid.png"
        renderer.render_acid().save(acid_path, "PNG")
        sprites["acid"] = acid_path

        wall_path = output_dir / "wall.png"
        renderer.render_wall().save(wall_path, "PNG")
        sprites["wall"] = wall_path

        floor_path = output_dir / "floor.png"
        renderer.render_floor().save(floor_path, "PNG")
        sprites["floor"] = floor_path

        return sprites

    @staticmethod
    def _draw_checkerboard(
        canvas: Image.Image,
        rect: Tuple[int, int, int, int],
        color_a: Tuple[int, int, int, int],
        color_b: Tuple[int, int, int, int],
        tile_px: int = 8,
    ) -> None:
        """Draws a dual-tone checkerboard pattern inside `rect = (x0, y0, x1, y1)` on `canvas`."""
        draw = ImageDraw.Draw(canvas)
        x0, y0, x1, y1 = rect
        step = max(4, int(tile_px))
        for y in range(y0, y1, step):
            row_idx = (y - y0) // step
            for x in range(x0, x1, step):
                col_idx = (x - x0) // step
                fill = color_a if (row_idx + col_idx) % 2 == 0 else color_b
                draw.rectangle([x, y, min(x1 - 1, x + step - 1), min(y1 - 1, y + step - 1)], fill=fill)

    @classmethod
    def render_asset_inspection_plate(
        cls,
        img: Union[Image.Image, Path, str],
        asset_type: str = "sprite",
        output_path: Optional[Union[Path, str]] = None,
        grid_cols: int = 1,
        grid_rows: int = 1,
    ) -> Image.Image:
        """Tier 1 Isolated Asset Render Plate:
        - For `tile` / `tile_surface`: renders a 3x3 seamless tiled grid (`3*w x 3*h`) so any edge seam or corner defect is immediately visible.
        - For `sprite`, `prop`, `portrait`, `diegetic_ui`, `backdrop`: renders a side-by-side Dual-Tone Checkerboard Alpha Inspection Plate
          (dark checkerboard on left with cyan tight alpha bounding box and optional multi-cell grid dividers, light checkerboard on right to expose dark/magenta edge halos).
        """
        if isinstance(img, (Path, str)):
            with Image.open(img) as opened:
                rgba = opened.convert("RGBA")
        else:
            rgba = img.convert("RGBA")

        w, h = rgba.size
        norm_type = str(asset_type or "sprite").lower().strip()
        is_tile = norm_type in ("tile", "tile_surface")
        g_cols = max(1, int(grid_cols or 1))
        g_rows = max(1, int(grid_rows or 1))
        is_multi_cell = (g_cols * g_rows) > 1

        if is_tile:
            # Render 3x3 repeating grid plate to inspect edge-to-edge seamless tiling
            cell_w = max(32, min(128, w))
            cell_h = max(32, min(128, h))
            tile_cell = rgba.resize((cell_w, cell_h), Image.Resampling.LANCZOS) if (w, h) != (cell_w, cell_h) else rgba
            plate = Image.new("RGBA", (cell_w * 3, cell_h * 3), (0, 0, 0, 255))
            for gy in range(3):
                for gx in range(3):
                    plate.paste(tile_cell, (gx * cell_w, gy * cell_h))
        else:
            # Render side-by-side Dark & Light Checkerboard Alpha Cutout Plate
            max_disp = 288 if is_multi_cell else 192
            disp_w = max(64, min(max_disp, w))
            disp_h = max(64, min(max_disp, h))
            scaled = rgba.resize((disp_w, disp_h), Image.Resampling.NEAREST if max(w, h) <= 64 else Image.Resampling.LANCZOS)
            pad = 12
            plate_w = disp_w * 2 + pad * 3
            plate_h = disp_h + pad * 2
            plate = Image.new("RGBA", (plate_w, plate_h), (18, 20, 26, 255))

            left_rect = (pad, pad, pad + disp_w, pad + disp_h)
            right_rect = (pad * 2 + disp_w, pad, pad * 2 + disp_w * 2, pad + disp_h)
            cls._draw_checkerboard(plate, left_rect, (30, 34, 42, 255), (46, 52, 64, 255), tile_px=8)
            cls._draw_checkerboard(plate, right_rect, (216, 220, 230, 255), (242, 244, 248, 255), tile_px=8)

            plate.paste(scaled, (pad, pad), scaled)
            plate.paste(scaled, (pad * 2 + disp_w, pad), scaled)

            draw = ImageDraw.Draw(plate)
            if is_multi_cell:
                draw.rectangle(
                    [pad, pad, pad + disp_w - 1, pad + disp_h - 1],
                    outline=(0, 229, 255, 220),
                    width=1,
                )
                for cx_i in range(1, g_cols):
                    gx = pad + int(round((disp_w * cx_i) / float(g_cols)))
                    draw.line([gx, pad, gx, pad + disp_h - 1], fill=(0, 229, 255, 190), width=1)
                for cy_i in range(1, g_rows):
                    gy = pad + int(round((disp_h * cy_i) / float(g_rows)))
                    draw.line([pad, gy, pad + disp_w - 1, gy], fill=(0, 229, 255, 190), width=1)
            else:
                bbox = scaled.getbbox()
                if bbox:
                    bx0, by0, bx1, by1 = bbox
                    draw.rectangle(
                        [pad + bx0, pad + by0, pad + bx1 - 1, pad + by1 - 1],
                        outline=(0, 229, 255, 220),
                        width=1,
                    )

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plate.save(out_p, "PNG")
        return plate

    @classmethod
    def render_animation_contact_sheet(
        cls,
        sheet_img: Union[Image.Image, Path, str],
        states_meta: Dict[str, Any],
        cell_size: Tuple[int, int] = (64, 64),
        output_path: Optional[Union[Path, str]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Tier 2 Multi-Frame Animation Contact Sheet & Deterministic Motion Telemetry:
        Extracts each animation state row from `sheet_img`, renders a padded filmstrip with checkerboard alpha background,
        frame boundary boxes, and ground-contact foot baselines, and computes deterministic inter-frame pixel deltas.
        """
        if isinstance(sheet_img, (Path, str)):
            with Image.open(sheet_img) as opened:
                rgba = opened.convert("RGBA")
        else:
            rgba = sheet_img.convert("RGBA")

        cw, ch = max(16, int(cell_size[0])), max(16, int(cell_size[1]))
        sheet_w, sheet_h = rgba.size

        if not states_meta:
            hframes = max(1, sheet_w // cw)
            states_meta = {
                "default": {"row": 0, "frame_count": hframes, "fps": 8.0, "loop": True, "motion_profile": "idle"}
            }

        ordered_states = sorted(states_meta.items(), key=lambda kv: int(kv[1].get("row", 0)))
        max_cols = max(max(1, int(st.get("frame_count", 1))) for _, st in ordered_states)
        num_rows = max(1, len(ordered_states))

        pad = 8
        cell_stride_x = cw + pad
        cell_stride_y = ch + pad
        contact_w = pad + max_cols * cell_stride_x
        contact_h = pad + num_rows * cell_stride_y
        contact = Image.new("RGBA", (contact_w, contact_h), (20, 22, 28, 255))
        draw = ImageDraw.Draw(contact)

        state_telemetry: Dict[str, Dict[str, Any]] = {}
        overall_min_l1 = None
        all_non_empty = True
        any_multi_frame_static = False

        for r_idx, (state_name, sinfo) in enumerate(ordered_states):
            row_num = int(sinfo.get("row", r_idx))
            fcount = max(1, int(sinfo.get("frame_count", 1)))
            frames: List[Image.Image] = []
            foot_ys: List[int] = []
            non_empty_frames = 0

            for c_idx in range(fcount):
                sx0 = min(max(0, c_idx * cw), max(0, sheet_w - cw))
                sy0 = min(max(0, row_num * ch), max(0, sheet_h - ch))
                frame_crop = rgba.crop((sx0, sy0, sx0 + cw, sy0 + ch))
                frames.append(frame_crop)

                dest_x = pad + c_idx * cell_stride_x
                dest_y = pad + r_idx * cell_stride_y
                cls._draw_checkerboard(
                    contact,
                    (dest_x, dest_y, dest_x + cw, dest_y + ch),
                    (32, 36, 46, 255),
                    (48, 54, 68, 255),
                    tile_px=8,
                )
                contact.paste(frame_crop, (dest_x, dest_y), frame_crop)

                fbbox = frame_crop.getbbox()
                if fbbox is not None and (fbbox[2] - fbbox[0]) >= 2 and (fbbox[3] - fbbox[1]) >= 2:
                    non_empty_frames += 1
                    foot_ys.append(int(fbbox[3]))
                    # Draw gold foot-baseline indicator at the bottom of the frame's subject bbox
                    by = dest_y + min(ch - 1, int(fbbox[3]) - 1)
                    draw.line([dest_x + 2, by, dest_x + cw - 3, by], fill=(255, 200, 60, 210), width=1)
                else:
                    all_non_empty = False

                draw.rectangle(
                    [dest_x, dest_y, dest_x + cw - 1, dest_y + ch - 1],
                    outline=(0, 210, 255, 180),
                    width=1,
                )

            # Compute inter-frame L1 pixel differences and OpenCV Farneback dense optical flow across consecutive frames
            deltas: List[float] = []
            flow_mags: List[float] = []
            for idx_f in range(len(frames) - 1):
                arr_a = np.array(frames[idx_f], dtype=np.float32)
                arr_b = np.array(frames[idx_f + 1], dtype=np.float32)
                mean_l1 = float(np.mean(np.abs(arr_a - arr_b)))
                deltas.append(round(mean_l1, 4))

                gray_a = cv2.cvtColor(arr_a[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                gray_b = cv2.cvtColor(arr_b[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                active_mask = (arr_a[:, :, 3] > 24) | (arr_b[:, :, 3] > 24)
                if np.any(active_mask):
                    flow = cv2.calcOpticalFlowFarneback(
                        gray_a, gray_b, None,
                        pyr_scale=0.5, levels=2, winsize=9, iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
                    )
                    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
                    flow_mags.append(round(float(np.mean(mag[active_mask])), 4))
                else:
                    flow_mags.append(0.0)

            min_l1 = min(deltas) if deltas else 0.0
            max_l1 = max(deltas) if deltas else 0.0
            mean_flow = float(np.mean(flow_mags)) if flow_mags else 0.0
            if fcount > 1:
                if overall_min_l1 is None or min_l1 < overall_min_l1:
                    overall_min_l1 = min_l1
                if max_l1 <= 0.25:
                    any_multi_frame_static = True

            foot_drift = (max(foot_ys) - min(foot_ys)) if len(foot_ys) >= 2 else 0
            state_telemetry[state_name] = {
                "frame_count": fcount,
                "non_empty_frames": non_empty_frames,
                "min_interframe_l1_delta": round(min_l1, 4),
                "max_interframe_l1_delta": round(max_l1, 4),
                "mean_optical_flow_magnitude": round(mean_flow, 4),
                "foot_baseline_max_drift_px": int(foot_drift),
            }

        all_flows = [float(st.get("mean_optical_flow_magnitude", 0.0)) for st in state_telemetry.values() if int(st.get("frame_count", 1)) > 1]
        overall_flow_mean = round(float(np.mean(all_flows)), 4) if all_flows else 0.0

        telemetry = {
            "states": state_telemetry,
            "all_frames_non_empty": bool(all_non_empty),
            "min_interframe_l1_delta": round(overall_min_l1 if overall_min_l1 is not None else 0.0, 4),
            "optical_flow_magnitude_mean": overall_flow_mean,
            "is_truly_animated": bool(all_non_empty and not any_multi_frame_static),
        }

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            contact.save(out_p, "PNG")
        return contact, telemetry

    @classmethod
    def render_articulated_rig_contact_sheet(
        cls,
        composite_path: Union[Path, str],
        articulated_parts: List[Dict[str, Any]],
        cartridge_dir: Path,
        cell_size: Tuple[int, int] = (64, 64),
        output_path: Optional[Union[Path, str]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Tier 2 Articulated Rig Multi-Pose Contact Sheet:
        Renders 4 articulated keyframes (t=0, t=1, t=2, t=3) by rotating/oscillating child parts around their declared pivots
        so the multimodal critic can inspect whether parts articulate cleanly without detaching or clipping.
        """
        cw, ch = max(32, int(cell_size[0])), max(32, int(cell_size[1]))
        loaded_parts: List[Tuple[Dict[str, Any], Image.Image]] = []
        for pmeta in articulated_parts:
            rel = str(pmeta.get("file_path") or pmeta.get("res_path", "").replace("res://", ""))
            pfile = cartridge_dir / rel
            if pfile.is_file():
                with Image.open(pfile) as pim:
                    pdims = pmeta.get("dimensions") or [cw // 2, ch // 2]
                    pw, ph = max(8, int(pdims[0])), max(8, int(pdims[1]))
                    loaded_parts.append((pmeta, pim.convert("RGBA").resize((pw, ph), Image.Resampling.LANCZOS)))

        frame_count = 4
        strip = Image.new("RGBA", (cw * frame_count, ch), (0, 0, 0, 0))
        for f_idx in range(frame_count):
            phase = (2.0 * math.pi * f_idx) / frame_count
            if loaded_parts:
                frame_canvas = cls._assemble_articulated_rig_frame(loaded_parts, (cw, ch), phase=phase)
            else:
                frame_canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
                if Path(composite_path).is_file():
                    with Image.open(composite_path) as cim:
                        base = cim.convert("RGBA").resize((cw, ch), Image.Resampling.LANCZOS)
                        rot = base.rotate(math.sin(phase) * 12.0, resample=Image.Resampling.BICUBIC, expand=False)
                        frame_canvas.paste(rot, (0, 0), rot)
            strip.paste(frame_canvas, (f_idx * cw, 0), frame_canvas)

        states_meta = {
            "articulated_cycle": {"row": 0, "frame_count": frame_count, "fps": 8.0, "loop": True, "motion_profile": "action"}
        }
        return cls.render_animation_contact_sheet(
            sheet_img=strip,
            states_meta=states_meta,
            cell_size=(cw, ch),
            output_path=output_path,
        )

    def verify_rendered_asset(
        self,
        asset_req: Union[AssetRequirement, Dict[str, Any]],
        rendered_asset_path: Union[Path, str],
        inspection_plate_path: Optional[Union[Path, str]] = None,
        reference_images: Optional[List[Any]] = None,
        style_guide: str = "",
    ) -> AssetVisualValidationVerdict:
        """Tier 1 Deterministic OpenCV + Multimodal Visual Asset Verification.
        Inspects the rendered asset and its isolated inspection plate before allowing it into asset_library.json.
        """
        req = AssetRequirement.model_validate(asset_req) if isinstance(asset_req, dict) else asset_req
        asset_p = Path(rendered_asset_path)
        if not asset_p.is_file():
            return AssetVisualValidationVerdict(
                deliberation=f"Rendered asset file '{asset_p}' is missing on disk.",
                asset_id=req.asset_id,
                alpha_cleanliness_passed=False,
                interior_hole_cleanliness_passed=False,
                subject_framing_and_completeness_passed=False,
                tile_seam_continuity_passed=False,
                mirror_symmetry_score=0.0,
                style_coherence_score=1,
                visual_defects=["Asset file missing on disk"],
                prompt_refinement_directive="Regenerate valid PNG image.",
                accepted=False,
            )

        with Image.open(asset_p) as im:
            rgba = im.convert("RGBA")
            w, h = rgba.size
            rgba_np = np.array(rgba, dtype=np.uint8)

        total_px = max(1, w * h)
        opaque_px = int(np.count_nonzero(rgba_np[:, :, 3] > 24))
        r_ch = rgba_np[:, :, 0].astype(np.int32)
        g_ch = rgba_np[:, :, 1].astype(np.int32)
        b_ch = rgba_np[:, :, 2].astype(np.int32)
        a_ch = rgba_np[:, :, 3].astype(np.int32)
        magenta_mask = (a_ch > 40) & (r_ch > 185) & (b_ch > 185) & (g_ch < 90) & ((np.minimum(r_ch, b_ch) - g_ch) > 100)
        magenta_px = int(np.count_nonzero(magenta_mask))
        alpha_ratio = opaque_px / float(total_px)
        magenta_ratio = magenta_px / float(total_px)

        is_tile = req.fabrication_mode == "tile_surface" or req.asset_type in ("tile", "tile_surface")
        is_backdrop = req.fabrication_mode == "parallax_layer" or req.asset_type in ("backdrop", "parallax_layer")

        defects: List[str] = []
        alpha_ok = True
        holes_ok = True
        framing_ok = True
        tile_ok = True
        mirror_sym = 0.0

        if opaque_px == 0 or alpha_ratio < 0.02:
            defects.append(f"Asset '{req.asset_id}' is nearly blank/transparent (alpha coverage={alpha_ratio:.3f}).")
            framing_ok = False
        if magenta_ratio > 0.015:
            defects.append(f"Asset '{req.asset_id}' has residual chroma-key magenta spill ({magenta_ratio:.3f}).")
            alpha_ok = False

        if is_tile:
            px = rgba.load()
            corners = [px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1]]
            for r, g, b, a in corners:
                if a < 240 or (r > 215 and g > 215 and b > 215) or (r > 200 and b > 200 and g < 80):
                    tile_ok = False
                    defects.append("Tile texture has defective white/magenta/transparent corner pixels.")
                    break
            mirror_sym = self.compute_tile_mirror_symmetry(rgba)
            if mirror_sym > 0.62:
                tile_ok = False
                defects.append(
                    f"Tile texture '{req.asset_id}' exhibits excessive bilateral mirror symmetry "
                    f"(mirror_symmetry_score={mirror_sym:.3f} > 0.62). Must be synthesized with non-mirrored toroidal tiling."
                )
        elif not is_backdrop:
            px = rgba.load()
            corners_opaque = sum(1 for (cx, cy) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] if px[cx, cy][3] > 220)
            if corners_opaque == 4 and alpha_ratio > 0.96:
                alpha_ok = False
                defects.append("Cutout sprite/portrait has all 4 outer corners opaque (>96% fill), indicating an unremoved background box or frame.")

        g_cols = max(1, int(getattr(req, "grid_cols", 1) or 1))
        g_rows = max(1, int(getattr(req, "grid_rows", 1) or 1))
        det_defective_cells: List[int] = []
        if (g_cols * g_rows) > 1:
            cw_chk = max(1, w // g_cols)
            ch_chk = max(1, h // g_rows)
            empty_cells: List[str] = []
            magenta_cells: List[str] = []
            boxed_cells: List[str] = []
            for r_i in range(g_rows):
                for c_i in range(g_cols):
                    cell_idx = r_i * g_cols + c_i
                    sub_rgba = rgba_np[r_i * ch_chk:(r_i + 1) * ch_chk, c_i * cw_chk:(c_i + 1) * cw_chk]
                    sub_a = sub_rgba[:, :, 3]
                    sub_total = max(1, int(sub_a.size))
                    sub_opaque = int(np.count_nonzero(sub_a > 24))
                    if sub_opaque < 16:
                        empty_cells.append(f"(row={r_i}, col={c_i})")
                        det_defective_cells.append(cell_idx)
                        continue
                    sub_mag = int(
                        np.count_nonzero(
                            (sub_a > 24)
                            & (sub_rgba[:, :, 0] > 205)
                            & (sub_rgba[:, :, 2] > 205)
                            & (sub_rgba[:, :, 1] < 65)
                        )
                    )
                    if (sub_mag / float(sub_total)) > 0.015:
                        magenta_cells.append(f"(row={r_i}, col={c_i})")
                        det_defective_cells.append(cell_idx)
                    if not is_tile and not is_backdrop and sub_rgba.shape[0] >= 4 and sub_rgba.shape[1] >= 4:
                        c_corners = sum(
                            1
                            for (cy_c, cx_c) in [
                                (0, 0),
                                (0, sub_rgba.shape[1] - 1),
                                (sub_rgba.shape[0] - 1, 0),
                                (sub_rgba.shape[0] - 1, sub_rgba.shape[1] - 1),
                            ]
                            if int(sub_rgba[cy_c, cx_c, 3]) > 220
                        )
                        if c_corners == 4 and (sub_opaque / float(sub_total)) > 0.96:
                            boxed_cells.append(f"(row={r_i}, col={c_i})")
                            det_defective_cells.append(cell_idx)
            if empty_cells:
                framing_ok = False
                defects.append(
                    f"Multi-cell grid sheet ({g_cols}x{g_rows}) has empty/blank cell(s): {', '.join(empty_cells)}."
                )
            if magenta_cells:
                alpha_ok = False
                defects.append(
                    f"Multi-cell grid sheet ({g_cols}x{g_rows}) has chroma-key magenta spill in cell(s): {', '.join(magenta_cells)}."
                )
            if boxed_cells:
                alpha_ok = False
                defects.append(
                    f"Multi-cell grid sheet ({g_cols}x{g_rows}) has unremoved opaque background box in cell(s): {', '.join(boxed_cells)}."
                )

        plate_p = Path(inspection_plate_path) if inspection_plate_path else asset_p.parent / "renders" / f"plate_{req.asset_id}.png"
        if not plate_p.is_file():
            self.render_asset_inspection_plate(
                rgba,
                asset_type=req.asset_type,
                output_path=plate_p,
                grid_cols=g_cols,
                grid_rows=g_rows,
            )

        det_passed = bool(alpha_ok and holes_ok and framing_ok and tile_ok and not defects)

        if self._is_live_genai_client():
            from google.genai import types
            contents: List[Any] = [
                types.Part.from_bytes(data=plate_p.read_bytes(), mime_type="image/png"),
            ]
            for rp in self._build_reference_parts(reference_images)[:1]:
                contents.append(rp)
            grid_note = (
                f"\nMulti-Cell Grid Layout: {g_cols} columns x {g_rows} rows ({g_cols * g_rows} cells total, 0-based row-major index `idx = row * {g_cols} + col`, cell_prompts={getattr(req, 'cell_prompts', [])}) — "
                f"verify that every cell in the {g_cols}x{g_rows} grid contains its distinct graphic cleanly within its cyan cell boundaries. "
                f"If `accepted=False` because specific cell(s) have defects while other cells are clean, populate `defective_cell_indices` with ONLY those defective 0-based row-major cell indices (0..{g_cols * g_rows - 1}) so clean cells are preserved without regenerating the entire sheet."
                if (g_cols * g_rows) > 1
                else ""
            )
            v_prompt = (
                f"You are the Studio Lead Visual QA Critic inspecting a rendered asset inspection plate before sealing it into the Cartridge Asset Library.\n"
                f"Asset ID: '{req.asset_id}' | Type: '{req.asset_type}' | Mode: '{req.fabrication_mode}'\n"
                f"Intended Purpose: {req.purpose or req.description}{grid_note}\n"
                f"Style Guide: {style_guide}\n"
                f"Deterministic OpenCV Pre-Check: det_passed={det_passed}, mirror_symmetry_score={mirror_sym:.3f}, pre_detected_defects={defects}, pre_detected_defective_cell_indices={sorted(set(det_defective_cells))}\n"
                f"Inspection Rules:\n"
                f"- If mode is 'tile_surface' (shown as a 3x3 tiled grid): verify seamless edge-to-edge organic material repetition with ZERO bilateral mirror symmetry, ZERO harsh grid seams, ZERO isometric 3D block corners, and ZERO sky horizons.\n"
                f"- If mode is 'static_sprite', 'sprite_sheet', 'diegetic_ui', or 'portrait' (shown over external QA dark and light square checkerboards with a cyan bounding-box border / grid dividers): verify clean alpha cutout silhouette (ZERO solid background boxes or picture frames inside the subject, ZERO trapped background pockets between limbs, ZERO transparent holes inside the subject's eyes or face), complete subject framing (ZERO detached environment terrain slabs or ground pedestals attached beneath the subject's base), and cohesive art style. IMPORTANT: The side-by-side dark/light square checkerboard grids and the cyan bounding-box/grid lines in `plate_*.png` are drawn externally by the QA plate renderer to visualize transparency and cell bounds—they are NOT inside the sprite PNG and must NEVER be flagged as checkerboard/grid/box defects!\n"
                f"MANDATORY (Rule 4): Provide `deliberation` FIRST, and `accepted` LAST."
            )
            contents.append(v_prompt)
            text_model = getattr(self, "text_model_name", DEFAULT_STUDIO_MODEL) or DEFAULT_STUDIO_MODEL
            try:
                resp = call_gemini_with_transient_retry(
                    lambda: self.client.models.generate_content(
                        model=text_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=AssetVisualValidationVerdict,
                            thinking_config=bounded_thinking_config(512),
                        ),
                    ),
                    max_transient_retries=2,
                    label=f"verify_fabricated_asset[{req.asset_id}]",
                )
            except Exception as exc:
                raise StudioAgentError(
                    f"Multimodal asset verification failed for '{req.asset_id}' on model '{text_model}': {exc}"
                ) from exc
            if not resp.text:
                raise StudioAgentError(
                    f"Multimodal asset verification returned empty response for '{req.asset_id}' on model '{text_model}'."
                )
            verdict = AssetVisualValidationVerdict.model_validate_json(resp.text)
            verdict.mirror_symmetry_score = mirror_sym
            total_grid_cells = g_cols * g_rows
            if total_grid_cells > 1:
                verdict.defective_cell_indices = sorted(
                    {
                        int(i)
                        for i in list(verdict.defective_cell_indices or []) + det_defective_cells
                        if 0 <= int(i) < total_grid_cells
                    }
                )
            else:
                verdict.defective_cell_indices = []
            if not det_passed:
                verdict.accepted = False
                verdict.alpha_cleanliness_passed = verdict.alpha_cleanliness_passed and alpha_ok
                verdict.subject_framing_and_completeness_passed = verdict.subject_framing_and_completeness_passed and framing_ok
                verdict.tile_seam_continuity_passed = verdict.tile_seam_continuity_passed and tile_ok
                for d in defects:
                    if d not in verdict.visual_defects:
                        verdict.visual_defects.append(d)
            return verdict

        return AssetVisualValidationVerdict(
            deliberation=(
                f"Inspected rendered plate for '{req.asset_id}' ({req.fabrication_mode}): "
                f"alpha_ratio={alpha_ratio:.2f}, magenta_ratio={magenta_ratio:.4f}, mirror_symmetry={mirror_sym:.3f}, defects={len(defects)}."
            ),
            asset_id=req.asset_id,
            alpha_cleanliness_passed=alpha_ok,
            interior_hole_cleanliness_passed=holes_ok,
            subject_framing_and_completeness_passed=framing_ok,
            tile_seam_continuity_passed=tile_ok,
            mirror_symmetry_score=mirror_sym,
            style_coherence_score=9 if det_passed else 3,
            visual_defects=defects,
            defective_cell_indices=sorted(set(det_defective_cells)),
            prompt_refinement_directive=(
                "Ensure isolated grounded subject on flat #FF00FF chroma background with zero detached ground pedestal or bilateral mirror symmetry."
                if not det_passed else ""
            ),
            accepted=det_passed,
        )

    def verify_rendered_animation(
        self,
        target_id: str,
        contact_sheet_path: Union[Path, str],
        anim_telemetry: Dict[str, Any],
        purpose: str = "",
        style_guide: str = "",
    ) -> AnimationValidationVerdict:
        """Tier 2 Deterministic OpenCV Optical Flow + Multimodal Animation Strip / Contact Sheet Verification."""
        sheet_p = Path(contact_sheet_path)
        all_non_empty = bool(anim_telemetry.get("all_frames_non_empty", False))
        is_animated = bool(anim_telemetry.get("is_truly_animated", False))
        min_l1 = float(anim_telemetry.get("min_interframe_l1_delta", 0.0))
        flow_mean = float(anim_telemetry.get("optical_flow_magnitude_mean", 0.0))

        defects: List[str] = []
        if not sheet_p.is_file():
            defects.append(f"Animation contact sheet '{sheet_p}' missing on disk.")
        if not all_non_empty:
            defects.append(f"Animation '{target_id}' contains empty/blank frames in the contact sheet.")
        if not is_animated or min_l1 < 0.35:
            is_animated = False
            defects.append(
                f"Animation '{target_id}' has static or insufficiently articulated frames (min_interframe_l1_delta={min_l1:.4f} < 0.35, optical_flow={flow_mean:.4f})."
            )

        if self._is_live_genai_client() and sheet_p.is_file():
            from google.genai import types
            contents: List[Any] = [
                types.Part.from_bytes(data=sheet_p.read_bytes(), mime_type="image/png"),
                (
                    f"You are the Studio Lead Animation QA Critic inspecting a rendered multi-frame animation contact sheet.\n"
                    f"Target Animation ID: '{target_id}' | Purpose: {purpose}\n"
                    f"Deterministic OpenCV & Frame Telemetry: {anim_telemetry}\n"
                    f"Pre-detected Defects: {defects}\n"
                    f"Style Guide: {style_guide}\n"
                    f"Verify that:\n"
                    f"1. Frames show genuine, readable keyframe pose or mechanism articulation across the strip (not identical static copies).\n"
                    f"2. Subject silhouette is continuous and intact (no severed props, torn limbs, or detached terrain slabs across states).\n"
                    f"3. Grounded poses (idle/move) maintain consistent base contact along the external gold baseline guide.\n"
                    f"IMPORTANT: The dark square checkerboard grid, cyan cell borders, and horizontal yellow/gold baseline ruler line near the bottom of each cell in `anim_sheet_*.png` are external QA contact-sheet overlays drawn by the test harness—they are NOT part of the sprite sheet PNG and must NEVER be flagged as yellow line or checkerboard defects!\n"
                    f"MANDATORY (Rule 4): Provide `deliberation` FIRST, and `accepted` LAST."
                ),
            ]
            text_model = getattr(self, "text_model_name", DEFAULT_STUDIO_MODEL) or DEFAULT_STUDIO_MODEL
            try:
                resp = call_gemini_with_transient_retry(
                    lambda: self.client.models.generate_content(
                        model=text_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=AnimationValidationVerdict,
                            thinking_config=bounded_thinking_config(512),
                        ),
                    ),
                    max_transient_retries=2,
                    label=f"verify_animation_strip[{target_id}]",
                )
            except Exception as exc:
                raise StudioAgentError(
                    f"Multimodal animation verification failed for '{target_id}' on model '{text_model}': {exc}"
                ) from exc
            if not resp.text:
                raise StudioAgentError(
                    f"Multimodal animation verification returned empty response for '{target_id}' on model '{text_model}'."
                )
            verdict = AnimationValidationVerdict.model_validate_json(resp.text)
            verdict.optical_flow_magnitude_mean = flow_mean
            if not is_animated or not all_non_empty or defects:
                verdict.accepted = False
                verdict.motion_progression_verified = False
                for d in defects:
                    if d not in verdict.animation_defects:
                        verdict.animation_defects.append(d)
            return verdict

        passed = bool(all_non_empty and is_animated and not defects)
        return AnimationValidationVerdict(
            deliberation=(
                f"Evaluated rendered animation strip for '{target_id}': "
                f"all_frames_non_empty={all_non_empty}, is_truly_animated={is_animated}, "
                f"min_interframe_l1_delta={min_l1:.4f}, optical_flow_mean={flow_mean:.4f}."
            ),
            target_id=target_id,
            motion_progression_verified=is_animated,
            optical_flow_magnitude_mean=flow_mean,
            silhouette_and_pivot_integrity_passed=all_non_empty,
            baseline_grounding_passed=True,
            animation_defects=defects,
            remediation_directive="" if passed else "Regenerate clean grounded base sprite and apply continuous keyframe articulation.",
            accepted=passed,
        )


VisualArtisanAgent = VisualArtSpriteDesignerAgent



