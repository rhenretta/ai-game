"""engine/assets/palette.py
Authoritative Style Bible 16-Color Palette and Euclidean Color Quantization.
Enforces Rationale-First Epistemic Architecture (deliberation declared first).
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from PIL import Image

# ---------------------------------------------------------------------------
# Authoritative Style Bible 16-Color Retro Palette (Canonical PICO-8 / EGA)
# ---------------------------------------------------------------------------
STYLE_BIBLE_16_PALETTE: List[Tuple[int, int, int]] = [
    (0, 0, 0),        # 0: Pure Black (Transparent key / deep shadow)
    (29, 43, 83),     # 1: Dark Navy / Midnight Blue
    (126, 37, 83),    # 2: Deep Crimson / Violet
    (0, 135, 81),     # 3: Dark Green / Forest
    (171, 82, 54),    # 4: Rust Brown / Ochre
    (95, 87, 79),     # 5: Muted Slate Gray / Stone Shadow
    (194, 195, 199),  # 6: Light Gray / Steel Highlight
    (255, 241, 232),  # 7: Off-White / Pearl / Specular Glint
    (255, 0, 77),     # 8: Neon Red / Threat / Weakspot
    (255, 163, 0),    # 9: Amber Orange / Caution / Warning
    (255, 236, 39),   # 10: Bright Yellow / Kinetic Spark / Visor
    (0, 228, 54),     # 11: Electric Green / Acid / Bio-hazard
    (41, 173, 255),   # 12: Cyan Sky Blue / Player Shield / Core
    (131, 118, 156),  # 13: Lavender Purple / Drone Sniper
    (255, 119, 168),  # 14: Pastel Rose Pink / Energy Flash
    (255, 204, 170),  # 15: Peach Flesh Tint / Warm Accent
]

PALETTE_COLOR_NAMES: Dict[str, Tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "dark_navy": (29, 43, 83),
    "deep_crimson": (126, 37, 83),
    "dark_green": (0, 135, 81),
    "rust_brown": (171, 82, 54),
    "slate_gray": (95, 87, 79),
    "light_gray": (194, 195, 199),
    "white": (255, 241, 232),
    "neon_red": (255, 0, 77),
    "amber_orange": (255, 163, 0),
    "bright_yellow": (255, 236, 39),
    "electric_green": (0, 228, 54),
    "cyan": (41, 173, 255),
    "lavender_purple": (131, 118, 156),
    "pink": (255, 119, 168),
    "peach": (255, 204, 170),
}

# Thematic color role mappings for visual moods while strictly using indices 0..15
MOOD_PALETTES: Dict[str, Dict[str, int]] = {
    "retro_arcade": {
        "hero_primary": 12,      # Cyan
        "hero_accent": 10,       # Yellow
        "hero_core": 8,          # Neon Red
        "chaser_armor": 2,       # Deep Crimson
        "chaser_eye": 8,         # Neon Red
        "ranged_hull": 13,       # Lavender Purple
        "ranged_lens": 12,       # Cyan
        "boss_hull": 1,          # Dark Navy
        "boss_trim": 9,          # Amber Orange
        "hazard_warn": 9,        # Amber Orange
        "hazard_lethal": 8,      # Neon Red
        "acid_liquid": 11,       # Electric Green
        "stone_face": 5,         # Slate Gray
        "metal_plate": 6,        # Light Gray
        "floor_base": 1,         # Dark Navy
        "portal_ring": 12,       # Cyan
    },
    "neon_cyberpunk": {
        "hero_primary": 14,      # Pastel Pink
        "hero_accent": 12,       # Cyan
        "hero_core": 10,         # Bright Yellow
        "chaser_armor": 1,       # Dark Navy
        "chaser_eye": 14,        # Pastel Pink
        "ranged_hull": 2,        # Deep Crimson
        "ranged_lens": 11,       # Electric Green
        "boss_hull": 2,          # Deep Crimson
        "boss_trim": 12,         # Cyan
        "hazard_warn": 10,       # Bright Yellow
        "hazard_lethal": 14,     # Pastel Pink
        "acid_liquid": 11,       # Electric Green
        "stone_face": 1,         # Dark Navy
        "metal_plate": 13,       # Lavender Purple
        "floor_base": 0,         # Pure Black
        "portal_ring": 14,       # Pastel Pink
    },
    "grim_dark": {
        "hero_primary": 5,       # Slate Gray
        "hero_accent": 9,        # Amber Orange
        "hero_core": 8,          # Neon Red
        "chaser_armor": 4,       # Rust Brown
        "chaser_eye": 8,         # Neon Red
        "ranged_hull": 5,        # Slate Gray
        "ranged_lens": 9,        # Amber Orange
        "boss_hull": 4,          # Rust Brown
        "boss_trim": 8,          # Neon Red
        "hazard_warn": 9,        # Amber Orange
        "hazard_lethal": 8,      # Neon Red
        "acid_liquid": 3,        # Dark Green
        "stone_face": 5,         # Slate Gray
        "metal_plate": 4,        # Rust Brown
        "floor_base": 0,         # Pure Black
        "portal_ring": 9,        # Amber Orange
    },
    "vibrant_fantasy": {
        "hero_primary": 11,      # Electric Green
        "hero_accent": 15,       # Peach
        "hero_core": 12,         # Cyan
        "chaser_armor": 4,       # Rust Brown
        "chaser_eye": 9,         # Amber Orange
        "ranged_hull": 3,        # Dark Green
        "ranged_lens": 10,       # Bright Yellow
        "boss_hull": 3,          # Dark Green
        "boss_trim": 15,         # Peach
        "hazard_warn": 10,       # Bright Yellow
        "hazard_lethal": 2,      # Deep Crimson
        "acid_liquid": 11,       # Electric Green
        "stone_face": 3,         # Dark Green
        "metal_plate": 6,        # Light Gray
        "floor_base": 1,         # Dark Navy
        "portal_ring": 11,       # Electric Green
    },
}


class StyleBibleManifest(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared first!"""
    deliberation: str = Field(
        default="",
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining palette constraints and aesthetic coherence."
    )
    name: str = Field(default="StyleBible16")
    mood: str = Field(default="retro_arcade")
    colors_rgb: List[Tuple[int, int, int]] = Field(default_factory=lambda: list(STYLE_BIBLE_16_PALETTE))
    color_names: Dict[str, Tuple[int, int, int]] = Field(default_factory=lambda: dict(PALETTE_COLOR_NAMES))
    transparent_index: int = Field(default=0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "name": self.name,
            "mood": self.mood,
            "colors_rgb": [list(c) for c in self.colors_rgb],
            "color_names": {k: list(v) for k, v in self.color_names.items()},
            "transparent_index": self.transparent_index,
        }


def euclidean_distance_sq(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> int:
    """Calculates squared Euclidean distance between two RGB colors."""
    return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2


def quantize_rgb(
    rgb: Tuple[int, int, int],
    palette: Optional[List[Tuple[int, int, int]]] = None
) -> Tuple[int, int, int]:
    """Maps an arbitrary RGB color to the closest entry in the 16-color Style Bible palette."""
    pal = palette or STYLE_BIBLE_16_PALETTE
    best_color = pal[0]
    min_dist = float("inf")
    r, g, b = rgb[:3]
    for p in pal:
        dist = (r - p[0]) ** 2 + (g - p[1]) ** 2 + (b - p[2]) ** 2
        if dist < min_dist:
            min_dist = dist
            best_color = p
            if dist == 0:
                break
    return best_color


def get_palette_index(
    rgb: Tuple[int, int, int],
    palette: Optional[List[Tuple[int, int, int]]] = None
) -> int:
    """Returns the integer index (0..15) of the closest palette color."""
    pal = palette or STYLE_BIBLE_16_PALETTE
    best_idx = 0
    min_dist = float("inf")
    r, g, b = rgb[:3]
    for idx, p in enumerate(pal):
        dist = (r - p[0]) ** 2 + (g - p[1]) ** 2 + (b - p[2]) ** 2
        if dist < min_dist:
            min_dist = dist
            best_idx = idx
            if dist == 0:
                break
    return best_idx


def get_flat_palette_bytes(palette: Optional[List[Tuple[int, int, int]]] = None) -> List[int]:
    """Builds a 768-element flat list of integers for Pillow palette initialization (256 colors * 3)."""
    pal = palette or STYLE_BIBLE_16_PALETTE
    flat: List[int] = []
    for r, g, b in pal:
        flat.extend([int(r), int(g), int(b)])
    flat.extend([0, 0, 0] * (256 - len(pal)))
    return flat


def create_palette_reference_image(palette: Optional[List[Tuple[int, int, int]]] = None) -> Image.Image:
    """Creates a 1x1 Pillow Image in mode 'P' holding the authoritative 16-color palette."""
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(get_flat_palette_bytes(palette))
    return pal_img


def quantize_image_to_indexed(
    img: Image.Image,
    palette: Optional[List[Tuple[int, int, int]]] = None,
    transparent_index: int = 0
) -> Image.Image:
    """Quantizes any PIL Image (RGBA or RGB) to strict mode 'P' with 16-color palette and transparency."""
    pal = palette or STYLE_BIBLE_16_PALETTE
    pal_img = create_palette_reference_image(pal)
    flat_pal = get_flat_palette_bytes(pal)

    if img.mode == "RGBA":
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        p_img = rgb.quantize(palette=pal_img, dither=Image.Dither.NONE)

        # Transparent mask for pixels with alpha < 128
        mask = a.point(lambda p: 255 if p < 128 else 0, mode="1")
        zero_fill = Image.new("P", img.size, transparent_index)
        zero_fill.putpalette(flat_pal)
        p_img.paste(zero_fill, mask=mask)
        p_img.putpalette(flat_pal)
        p_img.info["transparency"] = transparent_index
        return p_img
    elif img.mode == "RGB":
        p_img = img.quantize(palette=pal_img, dither=Image.Dither.NONE)
        p_img.putpalette(flat_pal)
        return p_img
    elif img.mode == "P":
        rgb = img.convert("RGB")
        return quantize_image_to_indexed(rgb, pal, transparent_index)
    else:
        rgba = img.convert("RGBA")
        return quantize_image_to_indexed(rgba, pal, transparent_index)
