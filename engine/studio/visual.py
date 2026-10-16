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
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw

from engine.assets.palette import (
    STYLE_BIBLE_16_PALETTE,
    MOOD_PALETTES,
    quantize_rgb,
    quantize_image_to_indexed,
)
from engine.studio.models import CharacterSheet, StudioAgentError

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
            img_bytes = None
            if response.candidates:
                for cand in response.candidates:
                    if cand.content and cand.content.parts:
                        for p in cand.content.parts:
                            if hasattr(p, "inline_data") and p.inline_data:
                                img_bytes = p.inline_data.data
                                break
            if not img_bytes:
                raise StudioAgentError(f"No image bytes returned from Gemini image generation for '{char_obj.name}'")

            raw_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            raw_img = raw_img.resize((w, h), Image.Resampling.LANCZOS)
            quantized = quantize_image_to_indexed(raw_img, STYLE_BIBLE_16_PALETTE)
            quantized.save(out_p, "PNG")
            return out_p
        except Exception as exc:
            logger.error("Gemini image generation failed: %s", exc)
            raise StudioAgentError(f"VisualArtSpriteDesigner failed to generate portrait: {exc}") from exc

    def generate_cartridge_sprites(
        self,
        output_dir: Path,
        mood: str = "retro_arcade",
    ) -> Dict[str, Path]:
        """Generates all standard entity sprites and environment tiles into output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        renderer = ProceduralSpriteRenderer(mood=mood)
        sprites: Dict[str, Path] = {}

        # Player animations
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

        # Enemies
        chaser_path = output_dir / "chaser.png"
        renderer.render_chaser().save(chaser_path, "PNG")
        sprites["chaser"] = chaser_path

        sniper_path = output_dir / "sniper.png"
        renderer.render_sniper().save(sniper_path, "PNG")
        sprites["sniper"] = sniper_path

        boss_path = output_dir / "boss.png"
        renderer.render_boss_portrait((48, 48)).save(boss_path, "PNG")
        sprites["boss"] = boss_path

        # Hazards
        spikes_path = output_dir / "spikes.png"
        renderer.render_spikes().save(spikes_path, "PNG")
        sprites["spikes"] = spikes_path

        acid_path = output_dir / "acid.png"
        renderer.render_acid().save(acid_path, "PNG")
        sprites["acid"] = acid_path

        # Tiles
        wall_path = output_dir / "wall.png"
        renderer.render_wall().save(wall_path, "PNG")
        sprites["wall"] = wall_path

        floor_path = output_dir / "floor.png"
        renderer.render_floor().save(floor_path, "PNG")
        sprites["floor"] = floor_path

        return sprites
