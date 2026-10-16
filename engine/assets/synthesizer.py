"""engine/assets/synthesizer.py
Deterministic Local Procedural Pixel Art Sprite & Tile Synthesizer via Pillow.
Generates player, enemies, hazards, and wall tiles with 16-color retro indexing.
Zero external network calls; fully deterministic via seed hashing.
"""
from __future__ import annotations
import hashlib
import math
import random
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw
from engine.assets.palette import (
    STYLE_BIBLE_16_PALETTE,
    MOOD_PALETTES,
    quantize_image_to_indexed,
)


def get_seed_integer(seed: Union[int, str, None]) -> int:
    """Deterministically maps any seed input to a 32-bit unsigned integer."""
    if seed is None:
        return 42
    if isinstance(seed, int):
        return seed
    digest = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class DeterministicSpriteSynthesizer:
    """Synthesizes pixel-art sprites and environment tiles using pure Pillow and procedural geometry."""
    deliberation: str = "Deterministic local procedural pixel art sprite and tile synthesizer."

    def __init__(
        self,
        seed: Union[int, str, None] = None,
        mood: str = "retro_arcade",
        palette: Optional[List[Tuple[int, int, int]]] = None
    ):
        self.seed_int = get_seed_integer(seed)
        self.rng = random.Random(self.seed_int)
        self.mood = mood if mood in MOOD_PALETTES else "retro_arcade"
        self.palette = palette or STYLE_BIBLE_16_PALETTE
        self.roles = MOOD_PALETTES[self.mood]

    def _color(self, index: int) -> Tuple[int, int, int]:
        return self.palette[index]

    # -----------------------------------------------------------------------
    # Player Sprites (32x32)
    # -----------------------------------------------------------------------
    def synthesize_player_idle(self) -> Image.Image:
        """Controllable player character idle frame (32x32)."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        c_navy = self._color(1)
        c_armor = self._color(self.roles["hero_primary"])
        c_visor = self._color(self.roles["hero_accent"])
        c_core = self._color(self.roles["hero_core"])
        c_slate = self._color(5)
        c_white = self._color(7)

        # Helmet / Head
        draw.rectangle([11, 4, 20, 11], fill=c_navy)
        draw.rectangle([12, 6, 19, 9], fill=c_visor)
        draw.line([(13, 7), (16, 7)], fill=c_white)  # Visor glint

        # Torso & Breastplate
        draw.rectangle([9, 12, 22, 21], fill=c_navy)
        draw.rectangle([11, 13, 20, 20], fill=c_armor)
        draw.rectangle([14, 15, 17, 18], fill=c_core)  # Reactor core
        draw.point((15, 16), fill=c_white)

        # Pauldrons / Shoulders
        draw.rectangle([7, 13, 9, 18], fill=c_armor)
        draw.rectangle([22, 13, 24, 18], fill=c_armor)

        # Legs & Boots
        draw.rectangle([10, 22, 14, 28], fill=c_slate)
        draw.rectangle([17, 22, 21, 28], fill=c_slate)
        draw.rectangle([9, 27, 14, 29], fill=c_navy)
        draw.rectangle([17, 27, 22, 29], fill=c_navy)

        return quantize_image_to_indexed(img, self.palette)

    def synthesize_player_walk_1(self) -> Image.Image:
        """Player walk cycle frame 1 (32x32) — Left foot forward, right back."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        c_navy = self._color(1)
        c_armor = self._color(self.roles["hero_primary"])
        c_visor = self._color(self.roles["hero_accent"])
        c_core = self._color(self.roles["hero_core"])
        c_slate = self._color(5)
        c_spark = self._color(10)

        # Torso shifted 1px left
        draw.rectangle([10, 4, 19, 11], fill=c_navy)
        draw.rectangle([11, 6, 18, 9], fill=c_visor)
        draw.rectangle([8, 12, 21, 21], fill=c_navy)
        draw.rectangle([10, 13, 19, 20], fill=c_armor)
        draw.rectangle([13, 15, 16, 18], fill=c_core)

        # Dynamic leg stride
        draw.rectangle([8, 22, 12, 29], fill=c_slate)    # Forward boot
        draw.rectangle([18, 21, 22, 27], fill=c_slate)   # Raised trailing boot
        draw.point((19, 28), fill=c_spark)              # Kinetic stride puff

        return quantize_image_to_indexed(img, self.palette)

    def synthesize_player_walk_2(self) -> Image.Image:
        """Player walk cycle frame 2 (32x32) — Right foot forward, left back."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        c_navy = self._color(1)
        c_armor = self._color(self.roles["hero_primary"])
        c_visor = self._color(self.roles["hero_accent"])
        c_core = self._color(self.roles["hero_core"])
        c_slate = self._color(5)
        c_spark = self._color(10)

        # Torso shifted 1px right
        draw.rectangle([12, 4, 21, 11], fill=c_navy)
        draw.rectangle([13, 6, 20, 9], fill=c_visor)
        draw.rectangle([10, 12, 23, 21], fill=c_navy)
        draw.rectangle([12, 13, 21, 20], fill=c_armor)
        draw.rectangle([15, 15, 18, 18], fill=c_core)

        # Dynamic leg stride
        draw.rectangle([9, 21, 13, 27], fill=c_slate)    # Raised trailing boot
        draw.rectangle([19, 22, 23, 29], fill=c_slate)   # Forward boot
        draw.point((10, 28), fill=c_spark)              # Kinetic stride puff

        return quantize_image_to_indexed(img, self.palette)

    def synthesize_player_dash(self) -> Image.Image:
        """High-mobility horizontal dash frame with speed lines (32x32)."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        c_armor = self._color(self.roles["hero_primary"])
        c_core = self._color(self.roles["hero_core"])
        c_yellow = self._color(10)
        c_red = self._color(8)
        c_white = self._color(7)

        # Trailing speed lines / motion blur
        draw.line([(0, 12), (9, 12)], fill=c_yellow, width=1)
        draw.line([(2, 16), (11, 16)], fill=c_red, width=2)
        draw.line([(0, 20), (8, 20)], fill=c_white, width=1)
        draw.point((4, 14), fill=c_yellow)
        draw.point((6, 18), fill=c_white)

        # Compressed aerodynamic hero body
        draw.rectangle([10, 10, 28, 22], fill=c_armor)
        draw.rectangle([14, 12, 25, 20], fill=c_core)
        draw.polygon([(26, 11), (31, 16), (26, 21)], fill=c_yellow)  # Forward cutting wedge

        return quantize_image_to_indexed(img, self.palette)

    # -----------------------------------------------------------------------
    # Enemy Sprites (32x32 and 48x48 Boss)
    # -----------------------------------------------------------------------
    def synthesize_enemy_chaser(self) -> Image.Image:
        """Melee brawler enemy sprite with aggressive chitinous horns (32x32)."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        c_armor = self._color(self.roles["chaser_armor"])
        c_eye = self._color(self.roles["chaser_eye"])
        c_rust = self._color(4)
        c_white = self._color(7)

        # Horns / Mandibles
        draw.polygon([(6, 10), (2, 2), (10, 6)], fill=c_rust)
        draw.polygon([(25, 10), (29, 2), (21, 6)], fill=c_rust)

        # Chitinous Carapace
        draw.rectangle([6, 8, 25, 25], fill=c_armor)
        draw.rectangle([9, 11, 22, 22], fill=c_rust)

        # Menacing Glowing Eye Slit
        draw.rectangle([10, 14, 21, 17], fill=c_eye)
        draw.point((14, 15), fill=c_white)
        draw.point((17, 15), fill=c_white)

        # Claws / Spikes
        draw.polygon([(8, 25), (4, 30), (12, 27)], fill=c_armor)
        draw.polygon([(23, 25), (27, 30), (19, 27)], fill=c_armor)

        return quantize_image_to_indexed(img, self.palette)

    def synthesize_enemy_ranged(self) -> Image.Image:
        """Hovering sniper drone with optic barrel and telemetric sensor (32x32)."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        c_hull = self._color(self.roles["ranged_hull"])
        c_lens = self._color(self.roles["ranged_lens"])
        c_navy = self._color(1)
        c_spark = self._color(10)
        c_white = self._color(7)

        # Floating Diamond Chassis
        draw.polygon([(16, 4), (27, 16), (16, 26), (5, 16)], fill=c_hull)
        draw.polygon([(16, 8), (23, 16), (16, 23), (9, 16)], fill=c_navy)

        # Centered Optic Sensor
        draw.ellipse([12, 12, 19, 19], fill=c_lens)
        draw.point((15, 15), fill=c_white)

        # Directional Sniper Barrel / Emitter
        draw.rectangle([25, 14, 30, 17], fill=c_spark)
        draw.point((31, 15), fill=c_white)

        # Dual Stabilization Hover Thrusters
        draw.polygon([(9, 25), (12, 29), (14, 25)], fill=c_spark)
        draw.polygon([(18, 25), (20, 29), (23, 25)], fill=c_spark)

        return quantize_image_to_indexed(img, self.palette)

    def synthesize_enemy_boss(self) -> Image.Image:
        """Fortified Guardian Champion Boss (48x48)."""
        img = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        c_hull = self._color(self.roles["boss_hull"])
        c_trim = self._color(self.roles["boss_trim"])
        c_eye = self._color(self.roles["chaser_eye"])
        c_core = self._color(self.roles["hero_core"])
        c_white = self._color(7)

        # Crown Pauldrons / Spikes
        draw.polygon([(6, 16), (2, 4), (14, 10)], fill=c_trim)
        draw.polygon([(41, 16), (45, 4), (33, 10)], fill=c_trim)
        draw.polygon([(20, 6), (24, 1), (28, 6)], fill=c_trim)

        # Heavy Armored Carapace (48x48)
        draw.rectangle([8, 10, 39, 39], fill=c_hull)
        draw.rectangle([12, 14, 35, 35], fill=c_trim)

        # Dual Ocular Visors
        draw.rectangle([14, 18, 21, 22], fill=c_eye)
        draw.rectangle([26, 18, 33, 22], fill=c_eye)
        draw.point((17, 20), fill=c_white)
        draw.point((29, 20), fill=c_white)

        # Heavy Central Energy Reactor
        draw.ellipse([18, 25, 29, 34], fill=c_core)
        draw.ellipse([21, 27, 26, 32], fill=c_white)

        # Base Armor Treads / Cleats
        draw.rectangle([10, 40, 18, 45], fill=c_hull)
        draw.rectangle([29, 40, 37, 45], fill=c_hull)

        return quantize_image_to_indexed(img, self.palette)

    # -----------------------------------------------------------------------
    # Environmental Hazards (48x48 and 96x96 Acid Pool)
    # -----------------------------------------------------------------------
    def synthesize_hazard_spikes(self) -> Dict[str, Image.Image]:
        """Returns 3 states for SpikeHazard: retracted, warning, extended (48x48)."""
        c_slate = self._color(5)
        c_navy = self._color(1)
        c_warn = self._color(self.roles["hazard_warn"])
        c_lethal = self._color(self.roles["hazard_lethal"])
        c_gray = self._color(6)
        c_white = self._color(7)

        # 1. Retracted Grate
        img_ret = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
        d_ret = ImageDraw.Draw(img_ret)
        d_ret.rectangle([2, 2, 45, 45], fill=c_slate, outline=c_navy, width=2)
        # Floor slots where spikes hide
        for i in range(4):
            sx = 8 + i * 9
            d_ret.rectangle([sx, 8, sx + 5, 39], fill=c_navy)

        # 2. Warning State (Yellow chevron trim + emerging tips)
        img_warn = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
        d_warn = ImageDraw.Draw(img_warn)
        d_warn.rectangle([2, 2, 45, 45], fill=c_slate, outline=c_warn, width=2)
        for i in range(4):
            sx = 8 + i * 9
            d_warn.rectangle([sx, 8, sx + 5, 39], fill=c_navy)
            d_warn.polygon([(sx, 20), (sx + 2, 13), (sx + 5, 20)], fill=c_warn)

        # 3. Extended Lethal State (Full chrome blades with red tips)
        img_ext = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
        d_ext = ImageDraw.Draw(img_ext)
        d_ext.rectangle([2, 2, 45, 45], fill=c_slate, outline=c_navy, width=2)
        for i in range(4):
            sx = 8 + i * 9
            # Full steel blade
            d_ext.polygon([(sx, 40), (sx + 2, 6), (sx + 5, 40)], fill=c_gray)
            d_ext.line([(sx + 2, 6), (sx + 2, 40)], fill=c_white)
            d_ext.polygon([(sx + 1, 10), (sx + 2, 6), (sx + 4, 10)], fill=c_lethal)

        return {
            "hazard_spikes_retracted": quantize_image_to_indexed(img_ret, self.palette),
            "hazard_spikes_warning": quantize_image_to_indexed(img_warn, self.palette),
            "hazard_spikes_extended": quantize_image_to_indexed(img_ext, self.palette),
        }

    def synthesize_hazard_acid_pool(self) -> Image.Image:
        """Corrosive bubbling caustic acid pool (96x96)."""
        img = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        c_acid = self._color(self.roles["acid_liquid"])
        c_dark_green = self._color(3)
        c_yellow = self._color(10)
        c_white = self._color(7)

        # Outer murky basin
        draw.ellipse([4, 6, 91, 89], fill=c_dark_green)
        draw.ellipse([8, 10, 87, 85], fill=c_acid)

        # Undulating ripples
        draw.ellipse([18, 20, 77, 75], fill=c_dark_green)
        draw.ellipse([24, 26, 71, 69], fill=c_acid)

        # Bubbling toxic spores
        bubbles = [(32, 38, 6), (58, 45, 8), (44, 60, 5), (65, 30, 4), (28, 55, 5)]
        for bx, by, br in bubbles:
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=c_yellow)
            draw.point((bx - 1, by - 1), fill=c_white)

        return quantize_image_to_indexed(img, self.palette)

    def synthesize_hazard_laser_emitter(self) -> Image.Image:
        """Armored industrial turret housing with focused crystal aperture (48x48)."""
        img = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        c_navy = self._color(1)
        c_slate = self._color(5)
        c_warn = self._color(self.roles["hazard_warn"])
        c_lethal = self._color(self.roles["hazard_lethal"])
        c_white = self._color(7)

        # Octagonal turret housing
        draw.polygon([
            (14, 4), (33, 4), (43, 14), (43, 33),
            (33, 43), (14, 43), (4, 33), (4, 14)
        ], fill=c_navy, outline=c_slate)

        # Hazard warning stripes on perimeter
        draw.line([(10, 8), (14, 4)], fill=c_warn, width=2)
        draw.line([(33, 4), (37, 8)], fill=c_warn, width=2)
        draw.line([(10, 39), (14, 43)], fill=c_warn, width=2)
        draw.line([(33, 43), (37, 39)], fill=c_warn, width=2)

        # Centered focusing lens
        draw.ellipse([14, 14, 33, 33], fill=c_slate)
        draw.ellipse([18, 18, 29, 29], fill=c_lethal)
        draw.ellipse([21, 21, 26, 26], fill=c_white)

        return quantize_image_to_indexed(img, self.palette)

    # -----------------------------------------------------------------------
    # Environment & Wall Tiles (32x32)
    # -----------------------------------------------------------------------
    def synthesize_tile_stone_block(self) -> Image.Image:
        """Chiseled masonry stone wall block with light bevel and mortar joints (32x32)."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)

        c_face = self._color(self.roles["stone_face"])
        c_light = self._color(6)   # Light gray highlight
        c_dark = self._color(1)    # Dark navy shadow
        c_mortar = self._color(0)  # Pure black seam

        # Base stone surface
        draw.rectangle([0, 0, 31, 31], fill=c_face)

        # Bevel lighting (light source top-left)
        draw.line([(0, 0), (31, 0)], fill=c_light)
        draw.line([(0, 0), (0, 31)], fill=c_light)
        draw.line([(31, 0), (31, 31)], fill=c_dark)
        draw.line([(0, 31), (31, 31)], fill=c_dark)

        # Mortar course line across center
        draw.line([(1, 15), (30, 15)], fill=c_mortar)
        # Vertical mortar joints
        draw.line([(15, 1), (15, 14)], fill=c_mortar)
        draw.line([(8, 16), (8, 30)], fill=c_mortar)
        draw.line([(24, 16), (24, 30)], fill=c_mortar)

        return quantize_image_to_indexed(img, self.palette)

    def synthesize_tile_metal_barrier(self) -> Image.Image:
        """Reinforced blast barrier with corner bolts and cross-ribbing (32x32)."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)

        c_plate = self._color(self.roles["metal_plate"])
        c_navy = self._color(1)
        c_white = self._color(7)

        # Heavy steel plate
        draw.rectangle([0, 0, 31, 31], fill=c_navy)
        draw.rectangle([2, 2, 29, 29], fill=c_plate)

        # Embossed diagonal "X" cross-ribbing
        draw.line([(3, 3), (28, 28)], fill=c_navy, width=2)
        draw.line([(28, 3), (3, 28)], fill=c_navy, width=2)

        # Corner Rivets / Hex Bolts
        bolts = [(5, 5), (26, 5), (5, 26), (26, 26)]
        for bx, by in bolts:
            draw.rectangle([bx - 1, by - 1, bx + 1, by + 1], fill=c_navy)
            draw.point((bx, by), fill=c_white)

        return quantize_image_to_indexed(img, self.palette)

    def synthesize_tile_floor(self) -> Image.Image:
        """Subtle dark arena floor tile (32x32) with high contrast for sprites."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)

        c_base = self._color(self.roles["floor_base"])
        c_slate = self._color(5)
        c_black = self._color(0)

        draw.rectangle([0, 0, 31, 31], fill=c_base)
        draw.rectangle([0, 0, 31, 31], outline=c_black, width=1)
        # Center muted accent
        draw.point((15, 15), fill=c_slate)
        draw.point((16, 16), fill=c_slate)

        return quantize_image_to_indexed(img, self.palette)

    def synthesize_tile_exit_portal(self) -> Image.Image:
        """Swirling mystic gateway portal (32x32)."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        c_ring = self._color(self.roles["portal_ring"])
        c_purple = self._color(13)
        c_navy = self._color(1)
        c_white = self._color(7)

        # Swirling concentric portal rings
        draw.ellipse([1, 1, 30, 30], fill=c_navy)
        draw.ellipse([4, 4, 27, 27], fill=c_purple)
        draw.ellipse([7, 7, 24, 24], fill=c_ring)
        draw.ellipse([11, 11, 20, 20], fill=c_white)

        return quantize_image_to_indexed(img, self.palette)


def generate_all_sprites(
    seed: Union[int, str, None] = None,
    mood: str = "retro_arcade",
    palette: Optional[List[Tuple[int, int, int]]] = None
) -> Tuple[Dict[str, Image.Image], Dict[str, Image.Image]]:
    """Synthesizes all required sprites and tiles, returning (sprites_dict, tiles_dict)."""
    synth = DeterministicSpriteSynthesizer(seed=seed, mood=mood, palette=palette)

    # Sprites
    spike_states = synth.synthesize_hazard_spikes()
    sprites = {
        "player_idle": synth.synthesize_player_idle(),
        "player_walk_1": synth.synthesize_player_walk_1(),
        "player_walk_2": synth.synthesize_player_walk_2(),
        "player_dash": synth.synthesize_player_dash(),
        "enemy_chaser": synth.synthesize_enemy_chaser(),
        "enemy_ranged": synth.synthesize_enemy_ranged(),
        "enemy_boss": synth.synthesize_enemy_boss(),
        "hazard_spikes_retracted": spike_states["hazard_spikes_retracted"],
        "hazard_spikes_warning": spike_states["hazard_spikes_warning"],
        "hazard_spikes_extended": spike_states["hazard_spikes_extended"],
        "hazard_acid_pool": synth.synthesize_hazard_acid_pool(),
        "hazard_laser_emitter": synth.synthesize_hazard_laser_emitter(),
    }

    # Tiles
    tiles = {
        "tile_stone_block": synth.synthesize_tile_stone_block(),
        "tile_metal_barrier": synth.synthesize_tile_metal_barrier(),
        "tile_floor": synth.synthesize_tile_floor(),
        "tile_exit_portal": synth.synthesize_tile_exit_portal(),
    }

    return sprites, tiles
