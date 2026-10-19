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
    AnimationStateSpec,
    ArticulatedPartSpec,
    AssetManifestSpec,
    AssetRequirement,
    CharacterSheet,
    GameRequirementsSpec,
    StudioAgentError,
    VisualMockupArtifact,
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

    @staticmethod
    def extract_clean_alpha_sprite(
        raw_img: Union[Image.Image, bytes, bytearray],
        target_size: Tuple[int, int] = (64, 64),
        strip_border_frame: bool = True,
        pad_ratio: float = 0.06,
        target_dimensions: Optional[Tuple[int, int]] = None,
        **_kwargs: Any,
    ) -> Image.Image:
        """Extracts a clean 32-bit RGBA sprite or portrait from a raw synthesized image.
        1. Detects and strips outer rectangular picture-frame borders (e.g. beige/dark frames around portraits).
        2. Runs multi-seed perimeter + inset-corner flood-fill background removal with soft alpha feathering.
        3. Auto-crops to the tight non-transparent subject bounding box and centers inside target_size.
        """
        if isinstance(raw_img, (bytes, bytearray)):
            raw_img = Image.open(io.BytesIO(raw_img))
        eff_size = target_dimensions if target_dimensions is not None else target_size
        w, h = max(1, int(eff_size[0])), max(1, int(eff_size[1]))
        work_w, work_h = 256, 256
        work_img = raw_img.convert("RGBA").resize((work_w, work_h), Image.Resampling.LANCZOS)
        px = work_img.load()

        # Step 1: Detect if there is an outer decorative frame/border and determine interior background seeds
        seed_coords: List[Tuple[int, int]] = []
        for x in range(0, work_w, 4):
            seed_coords.append((x, 0))
            seed_coords.append((x, work_h - 1))
        for y in range(0, work_h, 4):
            seed_coords.append((0, y))
            seed_coords.append((work_w - 1, y))

        if strip_border_frame:
            # Sample inset rings (3%, 6%, 10% inward from corners/edges) in case an outer border frame encloses a solid backdrop
            for frac in (0.03, 0.06, 0.10):
                ix = int(work_w * frac)
                iy = int(work_h * frac)
                inset_pts = [
                    (ix, iy),
                    (work_w - 1 - ix, iy),
                    (ix, work_h - 1 - iy),
                    (work_w - 1 - ix, work_h - 1 - iy),
                    (work_w // 2, iy),
                    (ix, work_h // 2),
                    (work_w - 1 - ix, work_h // 2),
                ]
                # Check if the four inset corners share a roughly consistent backdrop color
                c0 = px[ix, iy][:3]
                corners_match = sum(
                    1
                    for pt in inset_pts[:4]
                    if ((px[pt[0], pt[1]][0] - c0[0]) ** 2 + (px[pt[0], pt[1]][1] - c0[1]) ** 2 + (px[pt[0], pt[1]][2] - c0[2]) ** 2) <= 45 * 45
                )
                if corners_match >= 3:
                    seed_coords.extend(inset_pts)
                    # Also clear the thin outer frame band outside (ix, iy) along the 4 edges if corners matched
                    for bx in range(work_w):
                        for by in range(iy):
                            seed_coords.append((bx, by))
                        for by in range(work_h - iy, work_h):
                            seed_coords.append((bx, by))
                    for by in range(work_h):
                        for bx in range(ix):
                            seed_coords.append((bx, by))
                        for bx in range(work_w - ix, work_w):
                            seed_coords.append((bx, by))

        # Collect dominant background reference colors from the seeds
        bg_samples: List[Tuple[int, int, int]] = []
        for sx, sy in seed_coords[:32]:
            r, g, b, a = px[sx, sy]
            if a > 0:
                bg_samples.append((r, g, b))
        if not bg_samples:
            bg_samples = [(0, 0, 0)]

        # Step 2: Multi-seed flood-fill background removal
        visited = set()
        stack: List[Tuple[int, int, Tuple[int, int, int]]] = []
        for sx, sy in seed_coords:
            r, g, b, a = px[sx, sy]
            if a == 0:
                continue
            stack.append((sx, sy, (r, g, b)))

        tol_hard_sq = 28 * 28
        tol_soft_sq = 40 * 40

        while stack:
            cx, cy, ref_rgb = stack.pop()
            if (cx, cy) in visited or cx < 0 or cx >= work_w or cy < 0 or cy >= work_h:
                continue
            visited.add((cx, cy))
            r, g, b, a = px[cx, cy]
            if a == 0:
                continue

            # Compare against BOTH local neighbor reference AND global corner background samples
            # so flood-fill never drifts across gradients into character clothing or interiors.
            dist_local_sq = (r - ref_rgb[0]) ** 2 + (g - ref_rgb[1]) ** 2 + (b - ref_rgb[2]) ** 2
            dist_global_sq = min(
                (r - br) ** 2 + (g - bg) ** 2 + (b - bb) ** 2
                for br, bg, bb in bg_samples[:8]
            )
            effective_dist_sq = max(dist_local_sq, dist_global_sq)

            if effective_dist_sq <= tol_hard_sq:
                px[cx, cy] = (0, 0, 0, 0)
                next_ref = (
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
                # Feather edge alpha for clean anti-aliased silhouette
                ratio = (math.sqrt(effective_dist_sq) - 28.0) / 12.0
                new_a = max(0, min(255, int(a * ratio)))
                px[cx, cy] = (r, g, b, new_a)

        # Step 2b: Clear enclosed chroma-key magenta pockets (e.g., between character legs/arms) and despill fringes
        for iy in range(work_h):
            for ix in range(work_w):
                r, g, b, a = px[ix, iy]
                if a == 0:
                    continue
                magenta_excess = min(r, b) - g
                if r > 175 and b > 175 and g < 95 and magenta_excess > 105:
                    px[ix, iy] = (0, 0, 0, 0)
                elif r > 140 and b > 140 and g < 110 and magenta_excess > 65:
                    # Despill magenta fringe along anti-aliased silhouette edges
                    clamped_rb = min(r, b, g + 35)
                    new_a = max(0, min(a, int(a * 0.55)))
                    px[ix, iy] = (clamped_rb, g, clamped_rb, new_a)

        # Step 3: Crop to tight foreground bounding box and fit into target_size with padding
        bbox = work_img.getbbox()
        if bbox:
            x0, y0, x1, y1 = bbox
            # Guard against complete erasure if subject color matched background
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
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
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

        callout_str = "; ".join(str(c) for c in callouts) if callouts else "Clean diegetic in-world visual feedback, zero instruction-manual text walls"
        style_str = f" Art style direction: {style_guide}." if style_guide else ""

        prompt = (
            f"Polished indie video game 16:9 widescreen screenshot mockup: '{title}'. "
            f"Scene staging: {scene_desc}.{style_str} "
            f"UI & visual feedback framing: {callout_str}. "
            f"Rich atmospheric lighting, cohesive color palette, distinct silhouettes, expressive visual storytelling, "
            f"no wall-of-text tutorial overlays."
        )

        max_retries = 3
        img_bytes = None
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                from google.genai import types
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio=aspect),
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
    ) -> Image.Image:
        """Synthesizes a real 2D game sprite, prop, tile, or parallax backdrop via Gemini image generation."""
        from google.genai import types

        w, h = max(1, int(dimensions[0])), max(1, int(dimensions[1]))
        norm_type = str(asset_type).lower().strip()
        is_tile = norm_type in ("tile", "tile_surface")
        is_backdrop = norm_type in ("backdrop", "parallax_layer")
        style_clause = f" Match the visual style and color palette of the attached reference concept art ({style_guide})." if (reference_images or style_guide) else ""

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
                f"100% full-bleed edge-to-edge uniform repeatable material surface filling every pixel of the square frame. "
                f"Strictly NO rivers, NO water channels, NO cliffs, NO sky, NO horizon, NO isometric 3D cube or diorama block, "
                f"NO diagonal pipes, NO white corners, NO empty margins, NO border frame, NO text."
            )
        else:
            aspect = "1:1"
            full_prompt = (
                f"Full-body 2D in-game sprite asset: {prompt}.{style_clause} "
                f"Single isolated full-body game sprite or prop centered on a solid flat bright magenta (#FF00FF) chroma-key background. "
                f"STRICTLY ONLY the isolated subject itself — NO ground slab, NO pedestal, NO rock/grass platform under the feet, "
                f"NO cast shadow on the floor, NO background scenery, NOT a bust portrait, NO border frame, NO picture box, NO text, "
                f"crisp readable silhouette and vibrant details."
            )

        # Do not attach full-scene 16:9 landscape mockups when generating a macro surface material tile,
        # otherwise the image model copies the full landscape scene (rivers/islands) into the tile.
        ref_parts = [] if is_tile else self._build_reference_parts(reference_images)
        contents: Any = (ref_parts + [full_prompt]) if ref_parts else full_prompt

        max_retries = 3
        img_bytes = None
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio=aspect),
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
        if is_tile:
            return self.ensure_full_bleed_tile(raw_img, target_size=(w, h))
        if is_backdrop:
            return raw_img.resize((w, h), Image.Resampling.LANCZOS)

        return self.extract_clean_alpha_sprite(raw_img, target_size=(w, h), strip_border_frame=True, pad_ratio=0.06)

    @staticmethod
    def ensure_full_bleed_tile(raw_img: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> Image.Image:
        """Ensures a tile texture is 100% full-bleed with zero white/magenta/transparent isometric corner triangles,
        and applies toroidal border blending so repeating tiles have zero harsh grid seams.
        """
        w, h = max(16, int(target_size[0])), max(16, int(target_size[1]))
        img = raw_img.convert("RGBA").resize((256, 256), Image.Resampling.LANCZOS)
        px = img.load()

        def _is_bg_pixel(r: int, g: int, b: int, a: int) -> bool:
            if a < 200:
                return True
            if r > 215 and g > 215 and b > 215:
                return True
            if r > 200 and b > 200 and g < 80:
                return True
            return False

        # Shrink crop bounds inward from any edge/corner that contains background triangles
        top, bottom, left, right = 0, 256, 0, 256
        while bottom - top > 96:
            row_bg = sum(1 for x in range(left, right, 2) if _is_bg_pixel(*px[x, bottom - 1]))
            if row_bg > 0:
                bottom -= 2
            else:
                break
        while bottom - top > 96:
            row_bg = sum(1 for x in range(left, right, 2) if _is_bg_pixel(*px[x, top]))
            if row_bg > 0:
                top += 2
            else:
                break
        while right - left > 96:
            col_bg = sum(1 for y in range(top, bottom, 2) if _is_bg_pixel(*px[left, y]))
            if col_bg > 0:
                left += 2
            else:
                break
        while right - left > 96:
            col_bg = sum(1 for y in range(top, bottom, 2) if _is_bg_pixel(*px[right - 1, y]))
            if col_bg > 0:
                right -= 2
            else:
                break

        cropped = img.crop((left, top, right, bottom)).resize((w, h), Image.Resampling.LANCZOS)
        # Apply smooth toroidal edge wrap across outer 18% margin so repeating UV tiles have zero visible seams
        src_px = cropped.load()
        seamless = cropped.copy()
        dst_px = seamless.load()
        margin_x = max(4, w // 5)
        margin_y = max(4, h // 5)
        for y in range(h):
            for x in range(margin_x):
                alpha_x = 0.5 * (1.0 - (x / float(margin_x)))
                r1, g1, b1, _ = dst_px[x, y]
                r2, g2, b2, _ = src_px[w - 1 - x, y]
                rb = int(r1 * (1.0 - alpha_x) + r2 * alpha_x)
                gb = int(g1 * (1.0 - alpha_x) + g2 * alpha_x)
                bb = int(b1 * (1.0 - alpha_x) + b2 * alpha_x)
                dst_px[x, y] = (rb, gb, bb, 255)
                dst_px[w - 1 - x, y] = (
                    int(r2 * (1.0 - alpha_x) + r1 * alpha_x),
                    int(g2 * (1.0 - alpha_x) + g1 * alpha_x),
                    int(b2 * (1.0 - alpha_x) + b1 * alpha_x),
                    255,
                )
        temp_px = seamless.copy().load()
        for x in range(w):
            for y in range(margin_y):
                alpha_y = 0.5 * (1.0 - (y / float(margin_y)))
                r1, g1, b1, _ = temp_px[x, y]
                r2, g2, b2, _ = temp_px[x, h - 1 - y]
                dst_px[x, y] = (
                    int(r1 * (1.0 - alpha_y) + r2 * alpha_y),
                    int(g1 * (1.0 - alpha_y) + g2 * alpha_y),
                    int(b1 * (1.0 - alpha_y) + b2 * alpha_y),
                    255,
                )
                dst_px[x, h - 1 - y] = (
                    int(r2 * (1.0 - alpha_y) + r1 * alpha_y),
                    int(g2 * (1.0 - alpha_y) + g1 * alpha_y),
                    int(b2 * (1.0 - alpha_y) + b1 * alpha_y),
                    255,
                )
        r_ch, g_ch, b_ch, _ = seamless.split()
        full_a = Image.new("L", (w, h), 255)
        return Image.merge("RGBA", (r_ch, g_ch, b_ch, full_a))

    def fabricate_pose_frame_from_base(
        self,
        base_sprite: Image.Image,
        state_name: str,
        frame_idx: int,
        frame_count: int,
        cell_size: Tuple[int, int] = (64, 64),
        motion_profile: str = "idle",
    ) -> Image.Image:
        """Generates an articulated animation frame from a clean base sprite using the explicit motion_profile
        from AnimationStateSpec ('idle' | 'locomotion' | 'aerial' | 'action').
        """
        cw, ch = max(16, int(cell_size[0])), max(16, int(cell_size[1]))
        src = base_sprite.convert("RGBA").resize((cw, ch), Image.Resampling.LANCZOS)
        phase = (2.0 * math.pi * frame_idx) / max(1, frame_count)
        profile = str(motion_profile or "idle").lower().strip()

        canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        mid_y = int(ch * 0.56)
        upper = src.crop((0, 0, cw, mid_y))
        lower = src.crop((0, mid_y, cw, ch))

        if profile == "locomotion":
            # Stride cycle: alternating leg shear/shift + torso bob and forward lean
            bob_y = int(round(-abs(math.sin(phase)) * max(2.0, ch * 0.05)))
            lean_deg = math.sin(phase) * 5.0
            stride_x = int(round(math.sin(phase) * max(2.0, cw * 0.06)))
            upper_rot = upper.rotate(-lean_deg, resample=Image.Resampling.BICUBIC, expand=False)
            half_x = cw // 2
            leg_l = lower.crop((0, 0, half_x, ch - mid_y))
            leg_r = lower.crop((half_x, 0, cw, ch - mid_y))
            canvas.paste(leg_l, (max(0, stride_x), max(0, mid_y + bob_y - (1 if stride_x > 0 else 0))), leg_l)
            canvas.paste(leg_r, (min(cw - half_x, half_x - stride_x), max(0, mid_y + bob_y - (1 if stride_x < 0 else 0))), leg_r)
            canvas.paste(upper_rot, (0, max(0, bob_y)), upper_rot)
        elif profile == "aerial":
            # Jump arc: crouch squash -> upward stretch -> apex tuck -> landing
            t = frame_idx / max(1, frame_count - 1) if frame_count > 1 else 0.5
            if t < 0.25:
                sx, sy, dy = 1.10, 0.88, int(ch * 0.06)
            elif t < 0.75:
                sx, sy, dy = 0.92, 1.10, -int(ch * 0.06)
            else:
                sx, sy, dy = 1.04, 0.95, 0
            nw = max(4, min(cw, int(cw * sx)))
            nh = max(4, min(ch, int(ch * sy)))
            morphed = src.resize((nw, nh), Image.Resampling.LANCZOS)
            ox = (cw - nw) // 2
            oy = max(0, min(ch - nh, (ch - nh) // 2 + dy))
            canvas.paste(morphed, (ox, oy), morphed)
        elif profile == "action":
            # Action / resonance state: dynamic lunge + scale pulse
            lunge_x = int(round(math.sin(phase) * max(2.0, cw * 0.06)))
            rot_deg = math.sin(phase) * 8.0
            rotated = src.rotate(rot_deg, resample=Image.Resampling.BICUBIC, expand=False)
            canvas.paste(rotated, (max(0, min(cw - 1, lunge_x)), 0), rotated)
        else:
            # Idle / breathing cycle: subtle rhythmic torso breathing and vertical sway
            scale_y = 1.0 + 0.035 * math.sin(phase)
            scale_x = 1.0 - 0.02 * math.sin(phase)
            nw = max(4, min(cw, int(round(cw * scale_x))))
            nh = max(4, min(ch, int(round(ch * scale_y))))
            morphed = src.resize((nw, nh), Image.Resampling.LANCZOS)
            ox = (cw - nw) // 2
            oy = ch - nh
            canvas.paste(morphed, (ox, max(0, oy)), morphed)

        return canvas

    def fabricate_sprite_sheet(
        self,
        asset_req: AssetRequirement,
        output_path: Path,
        mood: str = "retro_arcade",
        reference_images: Optional[List[Union[Path, str, bytes, Image.Image]]] = None,
        style_guide: str = "",
    ) -> Tuple[Path, Dict[str, Any]]:
        """Fabricates a multi-row, multi-frame sprite sheet PNG and returns (sheet_path, metadata_dict).
        Each row corresponds to an AnimationStateSpec in asset_req.animation_states.
        Also saves a clean single-frame default sprite at the base sprite path if needed.
        """
        cell_w, cell_h = asset_req.dimensions or (64, 64)
        cell_w = max(32, int(cell_w))
        cell_h = max(32, int(cell_h))

        states = list(asset_req.animation_states) if asset_req.animation_states else [
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
                frame_count=3,
                fps=8.0,
                loop=False,
                pose_progression_prompt="Takeoff crouch, airborne leap, landing",
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

        # Synthesize the master base character/object sprite at 2x cell resolution for crisp pose articulation
        base_dims = (cell_w * 2, cell_h * 2)
        if self._is_live_genai_client():
            base_img = self._generate_gemini_sprite_image(
                prompt=asset_req.visual_prompt_or_text or asset_req.description or asset_req.asset_id,
                asset_type="sprite",
                dimensions=base_dims,
                reference_images=reference_images,
                style_guide=style_guide,
            )
        else:
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
        for row_idx, st in enumerate(states):
            fcount = max(1, int(st.frame_count))
            m_profile = getattr(st, "motion_profile", "idle") or "idle"
            for col_idx in range(fcount):
                frame_img = self.fabricate_pose_frame_from_base(
                    base_sprite=base_img,
                    state_name=st.state_name,
                    frame_idx=col_idx,
                    frame_count=fcount,
                    cell_size=(cell_w, cell_h),
                    motion_profile=m_profile,
                )
                sheet_img.paste(frame_img, (col_idx * cell_w, row_idx * cell_h), frame_img)

            states_meta[st.state_name] = {
                "row": row_idx,
                "start_frame": row_idx * max_cols,
                "frame_count": fcount,
                "fps": float(st.fps),
                "loop": bool(st.loop),
                "motion_profile": m_profile,
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
        composite = Image.new("RGBA", (comp_w, comp_h), (0, 0, 0, 0))
        parts_meta: List[Dict[str, Any]] = []

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
                renderer = ProceduralSpriteRenderer(mood=mood)
                p_img = renderer.render_dynamic_sprite(
                    dimensions=(pw, ph),
                    prompt=pspec.visual_prompt,
                    asset_type="sprite",
                    seed=f"{asset_req.asset_id}_{pspec.part_id}",
                )
            p_img = p_img.convert("RGBA").resize((pw, ph), Image.Resampling.LANCZOS)
            p_img.save(part_path, "PNG")

            # Paste into composite preview
            ox = int((comp_w - pw) // 2 + pspec.attach_offset_px[0])
            oy = int((comp_h - ph) // 2 + pspec.attach_offset_px[1])
            composite.paste(p_img, (max(0, min(comp_w - pw, ox)), max(0, min(comp_h - ph, oy))), p_img)

            parts_meta.append({
                "part_id": pspec.part_id,
                "res_path": f"res://assets/sprites/parts/{part_filename}",
                "file_path": f"assets/sprites/parts/{part_filename}",
                "parent_part_id": pspec.parent_part_id,
                "pivot_normalized": list(pspec.pivot_normalized),
                "attach_offset_px": list(pspec.attach_offset_px),
                "z_index_offset": int(pspec.z_index_offset),
                "dimensions": [pw, ph],
            })

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
    ) -> Path:
        """Renders and saves a single dynamic sprite from an AssetRequirement."""
        if self._is_live_genai_client():
            if isinstance(asset_req, dict):
                dims = asset_req.get("dimensions") or (64, 64)
                prompt = asset_req.get("visual_prompt_or_text") or asset_req.get("description") or asset_req.get("name", "game sprite")
                atype = asset_req.get("asset_type", "sprite")
            else:
                dims = getattr(asset_req, "dimensions", None) or (64, 64)
                prompt = getattr(asset_req, "visual_prompt_or_text", "") or getattr(asset_req, "description", "") or getattr(asset_req, "name", "game sprite")
                atype = getattr(asset_req, "asset_type", "sprite")
            img = self._generate_gemini_sprite_image(
                prompt=prompt,
                asset_type=atype,
                dimensions=dims,
                reference_images=reference_images,
                style_guide=style_guide,
            )
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


VisualArtisanAgent = VisualArtSpriteDesignerAgent

