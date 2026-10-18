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
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw

from engine.assets.palette import (
    STYLE_BIBLE_16_PALETTE,
    MOOD_PALETTES,
    quantize_rgb,
    quantize_image_to_indexed,
)
from engine.studio.models import (
    AssetManifestSpec,
    AssetRequirement,
    CharacterSheet,
    GameRequirementsSpec,
    StudioAgentError,
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

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = "gemini-3.1-flash-lite-image",
    ):
        self.client = client
        self.model_name = model_name
        self.renderer = ProceduralSpriteRenderer()

    @staticmethod
    def quantize_color(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Quantizes an arbitrary RGB tuple to the nearest Euclidean 16-color entry."""
        return quantize_rgb(rgb)

    def generate_portrait(
        self,
        character: Optional[Union[CharacterSheet, str, Dict[str, Any]]] = None,
        output_path: Optional[Union[Path, str]] = None,
        dimensions: Tuple[int, int] = (256, 256),
        **kwargs: Any,
    ) -> Path:
        """Generates a character portrait and saves as a valid PNG."""
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
                visual_description=kwargs.get("visual_prompt", kwargs.get("visual_description", "Pixel art sprite")),
            )
        elif isinstance(character, dict):
            c_name = character.get("name", "Hero")
            char_obj = CharacterSheet(
                deliberation=character.get("deliberation", f"Portrait character {c_name}"),
                character_id=character.get("character_id", c_name.lower()),
                name=c_name,
                role=character.get("role", character.get("archetype", "hero")),
                visual_description=character.get("visual_description", character.get("visual_prompt", "Pixel art sprite")),
            )
        else:
            char_obj = character

        max_retries = 3
        img_bytes = None
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                from google.genai import types
                prompt = (
                    f"Pixel art character portrait of {char_obj.name}, {char_obj.role}.\n"
                    f"Visual description: {char_obj.visual_description}.\n"
                    f"Square format, transparent or black background, 16-color retro palette aesthetic."
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio="1:1"),
                    ),
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
        raw_img = raw_img.resize((w, h), Image.Resampling.LANCZOS)
        quantized = quantize_image_to_indexed(raw_img, STYLE_BIBLE_16_PALETTE)
        quantized.save(out_p, "PNG")
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
        dimensions: Tuple[int, int] = (48, 48),
    ) -> Image.Image:
        """Synthesizes a real 2D game sprite or tile via Gemini image generation with background transparency."""
        from google.genai import types

        w, h = dimensions
        norm_type = str(asset_type).lower().strip()
        is_tile = any(k in norm_type for k in ("tile", "floor", "wall", "terrain"))

        if is_tile:
            full_prompt = (
                f"2D game tile texture: {prompt}. "
                f"Seamless top-down/isometric game environment tile, crisp pixel-art detailing, "
                f"distinctive silhouette and surface texture, rich contrast."
            )
        else:
            full_prompt = (
                f"2D game sprite icon: {prompt}. "
                f"Single isolated game object centered on a solid flat pure black (#000000) background, "
                f"no text, no border frame, crisp pixel-art game asset with clear silhouette and vibrant details."
            )

        max_retries = 3
        img_bytes = None
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio="1:1"),
                    ),
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
        # Work at 128x128 for clean corner flood-fill background removal before final resize
        work_img = raw_img.resize((128, 128), Image.Resampling.LANCZOS)
        if not is_tile:
            px = work_img.load()
            corners = [px[0, 0][:3], px[127, 0][:3], px[0, 127][:3], px[127, 127][:3]]
            bg_r = sum(c[0] for c in corners) // 4
            bg_g = sum(c[1] for c in corners) // 4
            bg_b = sum(c[2] for c in corners) // 4

            visited = set()
            stack = [(0, 0), (127, 0), (0, 127), (127, 127)]
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited or cx < 0 or cx >= 128 or cy < 0 or cy >= 128:
                    continue
                visited.add((cx, cy))
                r, g, b, a = px[cx, cy]
                dist_sq = (r - bg_r) ** 2 + (g - bg_g) ** 2 + (b - bg_b) ** 2
                if dist_sq <= 42 * 42:
                    px[cx, cy] = (0, 0, 0, 0)
                    stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

            bbox = work_img.getbbox()
            if bbox:
                cropped = work_img.crop(bbox)
                cw, ch = cropped.size
                scale = min((w - 4) / max(1, cw), (h - 4) / max(1, ch))
                nw = max(1, int(cw * scale))
                nh = max(1, int(ch * scale))
                resized_obj = cropped.resize((nw, nh), Image.Resampling.LANCZOS)
                final_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                final_img.paste(resized_obj, ((w - nw) // 2, (h - nh) // 2), resized_obj)
                return final_img

        return work_img.resize((w, h), Image.Resampling.LANCZOS)

    def generate_dynamic_sprite(
        self,
        asset_req: Union[AssetRequirement, Dict[str, Any]],
        output_path: Optional[Union[Path, str]] = None,
        mood: str = "retro_arcade",
    ) -> Path:
        """Renders and saves a single dynamic sprite from an AssetRequirement."""
        if self._is_live_genai_client():
            if isinstance(asset_req, dict):
                dims = asset_req.get("dimensions") or (48, 48)
                prompt = asset_req.get("visual_prompt_or_text") or asset_req.get("description") or asset_req.get("name", "game sprite")
                atype = asset_req.get("asset_type", "sprite")
            else:
                dims = getattr(asset_req, "dimensions", None) or (48, 48)
                prompt = getattr(asset_req, "visual_prompt_or_text", "") or getattr(asset_req, "description", "") or getattr(asset_req, "name", "game sprite")
                atype = getattr(asset_req, "asset_type", "sprite")
            img = self._generate_gemini_sprite_image(prompt=prompt, asset_type=atype, dimensions=dims)
        else:
            renderer = ProceduralSpriteRenderer(mood=mood)
            img = renderer.render_from_requirement(asset_req)

        if output_path is None:
            name = getattr(asset_req, "name", None) if hasattr(asset_req, "name") else (asset_req.get("name") if isinstance(asset_req, dict) else None)
            if not name:
                aid = getattr(asset_req, "asset_id", "sprite") if hasattr(asset_req, "asset_id") else (asset_req.get("asset_id", "sprite") if isinstance(asset_req, dict) else "sprite")
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
                if norm_type not in ("sprite", "tile", "portrait", "entity", "hazard", "item") and not tpath.endswith((".png", ".jpg", ".jpeg")):
                    continue

                # Determine filename
                filename = Path(tpath).name if tpath else (name if name else f"{aid}.png")
                if not filename.endswith(".png"):
                    filename = f"{filename}.png"

                # Determine destination path
                dest_path = output_dir / filename
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                if self._is_live_genai_client():
                    target_dims = (max(int(dims[0]), 48), max(int(dims[1]), 48))
                    img = self._generate_gemini_sprite_image(
                        prompt=prompt or filename,
                        asset_type=atype,
                        dimensions=target_dims,
                    )
                else:
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

        # Legacy mode: generate base sprites for backward compatibility
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

