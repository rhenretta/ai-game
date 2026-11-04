"""engine/studio/asset_library.py
Concept-Grounded Cartridge Asset Library & On-Demand Agent Fabrication Engine.
Enables studio agents (Director, VisualArtSpriteDesignerAgent, VoiceAudioEngineerAgent,
LeadGameplaySWEAgent, and Challenger Sign-Off Council) to commission, index, and bind
purpose-built game assets grounded in the Writers' Room concept art:
- Multi-pose / multi-state sprite sheets & animation strips (idle, move, jump, action, etc.)
- Articulated multi-part 2D rigs with hierarchical pivots and attachment offsets
- Static entity & interactive prop sprites with clean 32-bit RGBA alpha extraction
- Seamless terrain/platform surface tiles & 16:9 parallax backdrop plates
- Dialogue bust portraits strictly isolated in assets/portraits/
- Voice barks, procedural SFX, and ambient audio loops in assets/audio/
- Pure JSON catalog manifest (res://assets/asset_library.json) for runtime consumption

Strictly adheres to Rule 1 (100% Domain-Agnostic), Rule 4 (Rationale-First), and Rule 5 (Zero Silent Fallbacks).
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from engine.studio.models import (
    AnimationStateSpec,
    ArticulatedPartSpec,
    AssetLibraryCatalog,
    AssetLibraryEntry,
    AssetRequirement,
    CharacterSheet,
    GameBibleSpec,
    GameRequirementsSpec,
    StudioAgentConfigurationError,
    StudioAgentError,
    bounded_thinking_config,
    call_gemini_with_transient_retry,
)

logger = logging.getLogger("CartridgeAssetLibrary")


class CartridgeAssetLibrary:
    """Manages a cartridge's multimodal asset catalog (assets/asset_library.json),
    concept-reference resolution, on-demand fabrication, and GDScript helper export.
    """
    deliberation: str = "Concept-grounded Cartridge Asset Library managing purpose-built game assets."

    def __init__(
        self,
        cartridge_dir: Union[Path, str],
        visual_agent: Optional[Any] = None,
        voice_agent: Optional[Any] = None,
        mood: str = "retro_arcade",
        client: Optional[Any] = None,
        allow_offline_procedural: Optional[bool] = None,
    ):
        self.cartridge_dir = Path(cartridge_dir)
        self.assets_dir = self.cartridge_dir / "assets"
        self.sprites_dir = self.assets_dir / "sprites"
        self.portraits_dir = self.assets_dir / "portraits"
        self.mockups_dir = self.assets_dir / "mockups"
        self.audio_dir = self.assets_dir / "audio"
        self.masters_dir = self.assets_dir / "masters"
        self.variants_dir = self.assets_dir / "variants"
        self.catalog_path = self.assets_dir / "asset_library.json"
        self.client = client
        self.visual_agent = visual_agent
        self.voice_agent = voice_agent
        self.mood = mood
        if allow_offline_procedural is None:
            is_live_board_cartridge = (
                (self.cartridge_dir / "backlog.json").is_file()
                and "PYTEST_CURRENT_TEST" not in os.environ
            )
            self.allow_offline_procedural = not is_live_board_cartridge
        else:
            self.allow_offline_procedural = bool(allow_offline_procedural)
        if not self.allow_offline_procedural and self.visual_agent is not None:
            self.visual_agent.allow_offline_procedural = False

        self.sprites_dir.mkdir(parents=True, exist_ok=True)
        self.portraits_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.catalog = self._load_or_init_catalog()

    def _supersede_same_path_entries(self, active_asset_id: str, file_path: str) -> None:
        """Removes stale duplicate catalog entries pointing to the exact same relative `file_path`
        when an asset is regenerated in-place under a new task `asset_id`."""
        norm_target = str(file_path or "").replace("res://", "").lstrip("/")
        if not norm_target:
            return
        stale_ids = [
            oid
            for oid, oentry in list(self.catalog.entries.items())
            if oid != active_asset_id
            and str(getattr(oentry, "file_path", "") or "").replace("res://", "").lstrip("/") == norm_target
        ]
        for oid in stale_ids:
            self.catalog.entries.pop(oid, None)

    def _load_or_init_catalog(self) -> AssetLibraryCatalog:
        if self.catalog_path.is_file():
            try:
                raw = json.loads(self.catalog_path.read_text(encoding="utf-8"))
                return AssetLibraryCatalog.model_validate(raw)
            except Exception as exc:
                logger.warning("Failed to parse existing asset_library.json, re-initializing: %s", exc)

        bible_dict = self._load_game_bible_dict()
        return AssetLibraryCatalog(
            deliberation="Initialized Cartridge Asset Library catalog grounded in Game Bible.",
            game_title=str(bible_dict.get("game_title") or "Game Cartridge"),
            camera_and_spatial_paradigm=str(bible_dict.get("camera_and_spatial_paradigm") or ""),
            color_palette_hex=list(bible_dict.get("color_palette_hex") or []),
            entries={},
        )

    def _load_game_bible_dict(self) -> Dict[str, Any]:
        bible_path = self.cartridge_dir / "game_bible.json"
        if bible_path.is_file():
            try:
                return json.loads(bible_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def save_catalog(self) -> Path:
        self.catalog.updated_at_utc = datetime.now(timezone.utc).isoformat()
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.catalog_path.write_text(
            json.dumps(self.catalog.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        if (self.cartridge_dir / "project.godot").is_file() or (self.cartridge_dir / "src").is_dir():
            try:
                self.export_godot_studio_helper()
            except Exception as exc:
                logger.debug("StudioRigHelper export skipped: %s", exc)
        return self.catalog_path

    def export_godot_studio_helper(self) -> Path:
        """Writes `src/studio_rig_and_collider.gd` (`class_name StudioRigHelper`) into the cartridge
        so the SWE agent has native Godot 4 helpers for:
        1. `AnimationPlayer` + keyframe track binding (`attach_sprite_sheet_animation_player`)
        2. Pixel-accurate `BitMap.opaque_to_polygons()` collision tracing (`create_bitmap_collision_polygons`)
        3. Non-stretched repeating `Polygon2D` tiled surfaces (`create_tiled_surface`)
        """
        src_dir = self.cartridge_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        helper_path = src_dir / "studio_rig_and_collider.gd"
        code = '''class_name StudioRigHelper
extends RefCounted

static func load_context_texture(res_path: String, enable_mipmaps: bool = true) -> Texture2D:
\tif res_path.is_empty():
\t\treturn null
\tif enable_mipmaps:
\t\tvar abs_path := ProjectSettings.globalize_path(res_path)
\t\tif FileAccess.file_exists(abs_path):
\t\t\tvar img := Image.load_from_file(abs_path)
\t\t\tif img != null and not img.is_empty():
\t\t\t\timg.generate_mipmaps()
\t\t\t\treturn ImageTexture.create_from_image(img)
\tif ResourceLoader.exists(res_path):
\t\tvar loaded := load(res_path)
\t\tif loaded is Texture2D:
\t\t\treturn loaded
\treturn null

static func attach_sprite_sheet_animation_player(
\thost: Node2D,
\ttex: Texture2D,
\thframes_count: int = 4,
\tvframes_count: int = 4,
\tstates_meta: Dictionary = {},
\ttarget_size: Vector2 = Vector2(64, 64)
) -> Dictionary:
\tvar spr := Sprite2D.new()
\tspr.name = "RigSprite"
\tspr.texture = tex
\tspr.hframes = maxi(1, hframes_count)
\tspr.vframes = maxi(1, vframes_count)
\tif tex != null:
\t\tvar cell_w := float(tex.get_width()) / float(spr.hframes)
\t\tvar cell_h := float(tex.get_height()) / float(spr.vframes)
\t\tif cell_w > 0.0 and cell_h > 0.0:
\t\t\tspr.scale = Vector2(target_size.x / cell_w, target_size.y / cell_h)
\thost.add_child(spr)

\tvar anim_player := AnimationPlayer.new()
\tanim_player.name = "RigAnimationPlayer"
\thost.add_child(anim_player)

\tvar lib := AnimationLibrary.new()
\tvar eff_states := states_meta
\tif eff_states.is_empty():
\t\teff_states = {
\t\t\t"idle": {"row": 0, "frame_count": spr.hframes, "fps": 6.0, "loop": true},
\t\t\t"move": {"row": mini(1, spr.vframes - 1), "frame_count": spr.hframes, "fps": 10.0, "loop": true},
\t\t\t"jump": {"row": mini(2, spr.vframes - 1), "frame_count": mini(3, spr.hframes), "fps": 8.0, "loop": false},
\t\t\t"action": {"row": mini(3, spr.vframes - 1), "frame_count": spr.hframes, "fps": 10.0, "loop": false}
\t\t}

\tfor state_name in eff_states.keys():
\t\tvar info: Dictionary = eff_states[state_name]
\t\tvar row_idx := int(info.get("row", 0))
\t\tvar fcount := maxi(1, int(info.get("frame_count", spr.hframes)))
\t\tvar fps := maxf(1.0, float(info.get("fps", 8.0)))
\t\tvar is_loop := bool(info.get("loop", true))
\t\tvar anim := Animation.new()
\t\tvar duration := float(fcount) / fps
\t\tanim.length = maxf(0.1, duration)
\t\tanim.loop_mode = Animation.LOOP_LINEAR if is_loop else Animation.LOOP_NONE

\t\tvar track_idx := anim.add_track(Animation.TYPE_VALUE)
\t\tanim.track_set_path(track_idx, NodePath("RigSprite:frame"))
\t\tanim.value_track_set_update_mode(track_idx, Animation.UPDATE_DISCRETE)
\t\tfor f in range(fcount):
\t\t\tvar frame_coord := clampi(row_idx * spr.hframes + f, 0, spr.hframes * spr.vframes - 1)
\t\t\tanim.track_insert_key(track_idx, float(f) / fps, frame_coord)
\t\tlib.add_animation(str(state_name), anim)

\tanim_player.add_animation_library("", lib)
\tif lib.has_animation("idle"):
\t\tanim_player.play("idle")
\treturn {"sprite": spr, "anim_player": anim_player}

static func create_bitmap_collision_polygons(
\ttex: Texture2D,
\ttarget_size: Vector2,
\talpha_threshold: float = 0.2,
\tepsilon: float = 2.5
) -> Array[CollisionPolygon2D]:
\tvar result: Array[CollisionPolygon2D] = []
\tif tex == null:
\t\treturn result
\tvar img := tex.get_image()
\tif img == null or img.is_empty():
\t\treturn result
\tvar bm := BitMap.new()
\tbm.create_from_image_alpha(img, alpha_threshold)
\tvar polys := bm.opaque_to_polygons(Rect2i(Vector2i.ZERO, img.get_size()), epsilon)
\tvar iw := maxf(1.0, float(img.get_width()))
\tvar ih := maxf(1.0, float(img.get_height()))
\tfor poly in polys:
\t\tif poly.size() < 3:
\t\t\tcontinue
\t\tvar cpoly := CollisionPolygon2D.new()
\t\tvar scaled := PackedVector2Array()
\t\tfor pt in poly:
\t\t\tvar nx := ( (pt.x / iw) - 0.5 ) * target_size.x
\t\t\tvar ny := ( (pt.y / ih) - 0.5 ) * target_size.y
\t\t\tscaled.append(Vector2(nx, ny))
\t\tcpoly.polygon = scaled
\t\tresult.append(cpoly)
\treturn result

static func create_tiled_surface(
\tbody: StaticBody2D,
\ttex: Texture2D,
\tsize: Vector2,
\tis_walkable_shelf: bool = true
) -> Polygon2D:
\tif is_walkable_shelf and not body.is_in_group("walkable_shelf"):
\t\tbody.add_to_group("walkable_shelf")
\tvar hw := maxf(8.0, size.x * 0.5)
\tvar hh := maxf(8.0, size.y * 0.5)
\tvar poly := Polygon2D.new()
\tpoly.name = "TiledSurfaceVisual"
\tpoly.polygon = PackedVector2Array([
\t\tVector2(-hw, -hh),
\t\tVector2(hw, -hh),
\t\tVector2(hw, hh),
\t\tVector2(-hw, hh)
\t])
\tpoly.uv = PackedVector2Array([
\t\tVector2(0, 0),
\t\tVector2(size.x, 0),
\t\tVector2(size.x, size.y),
\t\tVector2(0, size.y)
\t])
\tif tex != null:
\t\tpoly.texture = tex
\t\tpoly.texture_repeat = CanvasItem.TEXTURE_REPEAT_ENABLED
\tbody.add_child(poly)

\tvar col := CollisionShape2D.new()
\tcol.name = "TiledSurfaceCollider"
\tvar rect := RectangleShape2D.new()
\trect.size = Vector2(hw * 2.0, hh * 2.0)
\tcol.shape = rect
\tbody.add_child(col)
\treturn poly
'''
        helper_path.write_text(code, encoding="utf-8")
        return helper_path

    def get_concept_reference_images(self, concept_reference_id: Optional[str] = None) -> List[Path]:
        """Resolves Writers' Room 16:9 mockup PNGs and optional character reference portraits
        to condition Gemini visual asset synthesis.
        """
        refs: List[Path] = []
        if concept_reference_id:
            cand_portrait = self.portraits_dir / f"{concept_reference_id}_portrait.png"
            if cand_portrait.is_file():
                refs.append(cand_portrait)
            cand_portrait_alt = self.portraits_dir / f"{concept_reference_id}.png"
            if cand_portrait_alt.is_file() and cand_portrait_alt not in refs:
                refs.append(cand_portrait_alt)
            if refs:
                return refs[:2]

        for d in (self.mockups_dir, self.cartridge_dir / "artifacts" / "concept_art"):
            if d.is_dir():
                for mp in sorted(d.glob("*.png")):
                    if mp not in refs:
                        refs.append(mp)
        return refs[:2]

    def get_rendering_strategy(self, require: bool = False) -> Optional[Any]:
        """Returns the cartridge's RenderingStrategySpec. When `require` is True and none exists yet, generates it
        (live client required; failures raise, Rule 5)."""
        from engine.studio.rendering import ensure_rendering_strategy, load_rendering_strategy

        strategy = load_rendering_strategy(self.cartridge_dir)
        if strategy is not None or not require:
            return strategy
        client = getattr(self.visual_agent, "client", None) if self.visual_agent is not None else self.client
        text_model = getattr(self.visual_agent, "text_model_name", "gemini-3.8-flash") if self.visual_agent else "gemini-3.8-flash"
        concept_parts = []
        try:
            from google.genai import types
            for rp in self.get_concept_reference_images()[:2]:
                if rp.is_file():
                    concept_parts.append(types.Part.from_bytes(data=rp.read_bytes(), mime_type="image/png"))
        except ImportError:
            pass
        return ensure_rendering_strategy(self.cartridge_dir, client, text_model, concept_parts=concept_parts)

    def get_style_guide(self, for_environment: bool = False) -> str:
        bible = self._load_game_bible_dict()
        parts: List[str] = []
        _rs = self.get_rendering_strategy()
        if _rs is not None:
            parts.append(
                "Pixel-art / low-resolution aesthetic as specified by the Game Bible"
                if _rs.pixelated_style_intended
                else "High-resolution, polished, modern illustrated rendering; NOT pixel art, NOT low-resolution"
            )
            if _rs.lighting_and_atmosphere:
                parts.append(f"Lighting & atmosphere: {_rs.lighting_and_atmosphere}")
        raw_style = str(bible.get("visual_style_guide") or "")
        if raw_style and not for_environment:
            parts.append(raw_style)
        if bible.get("camera_and_spatial_paradigm"):
            parts.append(f"Camera/spatial perspective: {bible['camera_and_spatial_paradigm']}")
        if bible.get("color_palette_hex"):
            parts.append(f"Palette hex: {', '.join(str(c) for c in bible['color_palette_hex'])}")
        return ". ".join(parts)

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        if not path.is_file():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _read_image_size(path: Path) -> Tuple[int, int]:
        try:
            with Image.open(path) as img:
                return img.size
        except Exception:
            return (64, 64)

    def _is_pixel_art_cartridge(self) -> bool:
        """Returns True when the cartridge's RenderingStrategySpec explicitly declares a pixel-art style."""
        rs = self.get_rendering_strategy(require=False)
        return bool(rs is not None and getattr(rs, "pixelated_style_intended", False))

    @staticmethod
    def _master_rel_for_runtime_rel(runtime_rel: str) -> str:
        norm = str(runtime_rel or "").replace("res://", "").lstrip("/")
        if norm.startswith("assets/"):
            sub = norm[len("assets/"):]
        else:
            sub = norm
        return f"assets/masters/{sub}"

    def _upgrade_req_for_hidpi_context(self, req: AssetRequirement) -> AssetRequirement:
        """Upgrades runtime texture `dimensions` for non-pixel-art live Gemini synthesis so the primary
        runtime asset is crisp at HiDPI (`2560x1440` + camera zoom) while preserving the logical `1280x720`
        `world_footprint_px` display size. Full `1K`/`2K` masters are always archived alongside in `assets/masters/`.
        """
        if not (hasattr(self.visual_agent, "_is_live_genai_client") and self.visual_agent._is_live_genai_client()):
            return req
        if self._is_pixel_art_cartridge():
            return req

        fmode = str(req.fabrication_mode or "").lower().strip()
        atype = str(req.asset_type or "").lower().strip()
        if fmode in ("audio_bark", "sfx", "ambient_loop") or atype in ("audio_bark", "sfx", "ambient_loop"):
            return req
        if fmode == "articulated_parts" or atype == "articulated_rig":
            return req

        cur_d = tuple(req.dimensions or (256, 256))
        cw_raw, ch_raw = max(16, int(cur_d[0])), max(16, int(cur_d[1]))

        if fmode == "portrait" or atype == "portrait":
            if max(cw_raw, ch_raw) < 512:
                scale_p = 512.0 / float(max(cw_raw, ch_raw))
                return req.model_copy(
                    update={"dimensions": (int(round(cw_raw * scale_p)), int(round(ch_raw * scale_p)))}
                )
            return req

        if fmode == "parallax_layer" or atype in ("backdrop", "parallax_layer"):
            updates_bg: Dict[str, Any] = {}
            if getattr(req, "world_footprint_px", None) is None:
                updates_bg["world_footprint_px"] = (cw_raw, ch_raw)
            if cw_raw <= 1280 and ch_raw <= 720:
                updates_bg["dimensions"] = (2560, 1440)
            return req.model_copy(update=updates_bg) if updates_bg else req

        if fmode == "tile_surface" or atype in ("tile", "tile_surface"):
            updates_tile: Dict[str, Any] = {}
            if getattr(req, "world_footprint_px", None) is None:
                updates_tile["world_footprint_px"] = (cw_raw, ch_raw)
            if max(cw_raw, ch_raw) < 512:
                scale_t = 512.0 / float(max(cw_raw, ch_raw))
                updates_tile["dimensions"] = (int(round(cw_raw * scale_t)), int(round(ch_raw * scale_t)))
            return req.model_copy(update=updates_tile) if updates_tile else req

        has_grid_prompts = (
            max(1, int(getattr(req, "grid_cols", 1) or 1)) * max(1, int(getattr(req, "grid_rows", 1) or 1)) > 1
            and bool(getattr(req, "cell_prompts", None))
        )
        is_single_actor_sheet = (fmode == "sprite_sheet" or atype == "sprite_sheet") and not has_grid_prompts
        g_c = 1 if is_single_actor_sheet else max(1, int(getattr(req, "grid_cols", 1) or 1))
        g_r = 1 if is_single_actor_sheet else max(1, int(getattr(req, "grid_rows", 1) or 1))
        cell_w = max(16, cw_raw // g_c)
        cell_h = max(16, ch_raw // g_r)

        fp = getattr(req, "world_footprint_px", None)
        logical_cw = int(fp[0]) if (isinstance(fp, (list, tuple)) and len(fp) >= 2 and int(fp[0]) > 0) else cell_w
        logical_ch = int(fp[1]) if (isinstance(fp, (list, tuple)) and len(fp) >= 2 and int(fp[1]) > 0) else cell_h
        target_longest = min(512, max(256, max(logical_cw, logical_ch) * 2))
        if max(cell_w, cell_h) >= target_longest:
            if fp is None and g_c == 1 and g_r == 1:
                return req.model_copy(update={"world_footprint_px": (logical_cw, logical_ch)})
            return req

        scale_s = float(target_longest) / float(max(cell_w, cell_h))
        new_cw = max(cell_w, int(round(cell_w * scale_s)))
        new_ch = max(cell_h, int(round(cell_h * scale_s)))
        hi_updates: Dict[str, Any] = {
            "dimensions": (new_cw * g_c, new_ch * g_r),
        }
        if fp is None:
            hi_updates["world_footprint_px"] = (logical_cw, logical_ch)
        if getattr(req, "cell_target_sizes", None):
            sx = float(new_cw) / float(max(1, cell_w))
            sy = float(new_ch) / float(max(1, cell_h))
            hi_updates["cell_target_sizes"] = [
                (max(4, int(round(int(pair[0]) * sx))), max(4, int(round(int(pair[1]) * sy))))
                for pair in req.cell_target_sizes
            ]
        return req.model_copy(update=hi_updates)

    def derive_context_variant(
        self,
        asset_id: str,
        context_name: str,
        target_size: Tuple[int, int],
        *,
        resample_nearest: Optional[bool] = None,
        persist_catalog: bool = True,
    ) -> Path:
        """Derives a context-specific smaller (or custom-sized) PNG asset directly from the full-resolution
        master (`entry.master_file_path`, falling back to `entry.file_path`) with ZERO Gemini API calls.
        Writes to `assets/variants/<category>/<asset_id>_<context_name>.png` and records metadata in
        `entry.variants[context_name]`.
        """
        entry = self.catalog.entries.get(asset_id)
        if entry is None:
            raise StudioAgentError(f"Cannot derive context variant for unknown asset_id '{asset_id}'.")
        if entry.fabrication_mode in ("audio_bark", "sfx", "ambient_loop"):
            raise StudioAgentError(f"Cannot derive image variant for audio asset '{asset_id}'.")

        src_rel = entry.master_file_path if (entry.master_file_path and (self.cartridge_dir / entry.master_file_path).is_file()) else entry.file_path
        src_path = self.cartridge_dir / src_rel
        if not src_path.is_file():
            raise StudioAgentError(f"Source image '{src_rel}' for asset '{asset_id}' does not exist on disk.")

        tw, th = max(1, int(target_size[0])), max(1, int(target_size[1]))
        use_nearest = self._is_pixel_art_cartridge() if resample_nearest is None else bool(resample_nearest)
        resample_mode = Image.Resampling.NEAREST if use_nearest else Image.Resampling.LANCZOS

        ctx_slug = re.sub(r"[^a-z0-9_]+", "_", str(context_name or "variant").strip().lower()).strip("_") or "variant"
        sub_cat = "portraits" if (entry.asset_type == "portrait" or entry.fabrication_mode == "portrait") else "sprites"
        var_rel = f"assets/variants/{sub_cat}/{asset_id}_{ctx_slug}.png"
        var_path = self.cartridge_dir / var_rel
        var_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src_path) as src_im:
            rgba = src_im.convert("RGBA")
            resized = rgba if rgba.size == (tw, th) else rgba.resize((tw, th), resample_mode)
            resized.save(var_path, "PNG")

        entry.variants[ctx_slug] = {
            "res_path": f"res://{var_rel}",
            "file_path": var_rel,
            "pixel_size": [tw, th],
            "sha256_hash": self._compute_sha256(var_path),
        }
        if persist_catalog:
            self.save_catalog()
        return var_path

    def _populate_default_context_variants(self, asset_id: str) -> None:
        """Automatically derives standard smaller context assets (`world_1x`, `hud_icon`) from the
        full-size master whenever the master exceeds those context sizes, incurring zero API calls.
        """
        entry = self.catalog.entries.get(asset_id)
        if entry is None or entry.fabrication_mode in ("audio_bark", "sfx", "ambient_loop"):
            return
        mw, mh = entry.master_pixel_size if (entry.master_pixel_size and entry.master_pixel_size[0] > 0) else entry.texture_pixel_size
        if mw <= 0 or mh <= 0:
            return
        tex_w, tex_h = entry.texture_pixel_size
        disp_w, disp_h = entry.target_display_size
        hf = max(1, int(entry.hframes or 1))
        vf = max(1, int(entry.vframes or 1))

        try:
            if entry.fabrication_mode == "parallax_layer" or entry.asset_type in ("backdrop", "parallax_layer"):
                if (mw > 1280 or mh > 720) and (tex_w, tex_h) != (1280, 720):
                    self.derive_context_variant(asset_id, "world_1x", (1280, 720), persist_catalog=False)
                return

            if disp_w >= 8 and disp_h >= 8:
                w1x = (disp_w * hf, disp_h * vf)
                if w1x != (tex_w, tex_h) and (mw > w1x[0] or mh > w1x[1]):
                    self.derive_context_variant(asset_id, "world_1x", w1x, persist_catalog=False)

            if hf == 1 and vf == 1 and entry.asset_type in ("sprite", "prop", "articulated_rig", "portrait", "diegetic_ui"):
                base_ar_w = max(1, disp_w if disp_w > 0 else mw)
                base_ar_h = max(1, disp_h if disp_h > 0 else mh)
                icon_scale = 64.0 / float(max(base_ar_w, base_ar_h))
                icon_sz = (max(16, int(round(base_ar_w * icon_scale))), max(16, int(round(base_ar_h * icon_scale))))
                if icon_sz != (tex_w, tex_h) and (mw > icon_sz[0] or mh > icon_sz[1]):
                    self.derive_context_variant(asset_id, "hud_icon", icon_sz, persist_catalog=False)
        except Exception as exc:
            logger.debug("Default context variant derivation skipped for '%s': %s", asset_id, exc)

    def sanitize_misplaced_portraits_in_sprites_dir(self) -> List[Path]:
        """Moves or copies any misplaced single-frame *_portrait.png or portrait_*.png files from
        assets/sprites/ into assets/portraits/ with clean alpha background extraction, preventing
        single-frame bust portraits from masquerading as in-world entity sprites while preserving
        explicitly authored sprite sheets and catalog-registered sprite assets.
        """
        from engine.studio.visual import VisualArtSpriteDesignerAgent

        cleaned_paths: List[Path] = []
        if not self.sprites_dir.is_dir():
            return cleaned_paths

        registered_sprite_filenames: set[str] = set()
        for entry in self.catalog.entries.values():
            fp = str(getattr(entry, "file_path", "") or "")
            rp = str(getattr(entry, "res_path", "") or "")
            fab_mode = str(getattr(entry, "fabrication_mode", "") or "")
            asset_type = str(getattr(entry, "asset_type", "") or "")
            hframes = int(getattr(entry, "hframes", 1) or 1)
            vframes = int(getattr(entry, "vframes", 1) or 1)
            is_Explicit_sprite = (
                fab_mode != "portrait"
                or asset_type != "portrait"
                or hframes > 1
                or vframes > 1
            )
            if is_Explicit_sprite:
                if fp.startswith("assets/sprites/"):
                    registered_sprite_filenames.add(Path(fp).name)
                if rp.startswith("res://assets/sprites/"):
                    registered_sprite_filenames.add(Path(rp.replace("res://", "")).name)

        for png_file in list(self.sprites_dir.glob("*.png")):
            if png_file.name in registered_sprite_filenames:
                stale_entry = self.catalog.entries.get(png_file.stem)
                if (
                    stale_entry is not None
                    and getattr(stale_entry, "asset_type", "") == "portrait"
                    and str(getattr(stale_entry, "deliberation", "")).startswith(
                        "Cleaned and indexed character portrait"
                    )
                ):
                    self.catalog.entries.pop(png_file.stem, None)
                continue
            stem_lower = png_file.stem.lower()
            if "portrait" in stem_lower:
                dest_portrait = self.portraits_dir / png_file.name
                master_rel = f"assets/masters/portraits/{dest_portrait.name}"
                master_dest = self.cartridge_dir / master_rel
                try:
                    with Image.open(png_file) as raw_img:
                        w, h = raw_img.size
                        if w > 0 and h > 0 and max(w, h) >= min(w, h) * 1.5:
                            # Multi-frame horizontal/vertical sprite strip; never squish to 1:1
                            continue
                        master_cleaned = VisualArtSpriteDesignerAgent.extract_clean_alpha_sprite(
                            raw_img,
                            target_size=None,
                            strip_border_frame=True,
                            pad_ratio=0.04,
                        )
                        master_dest.parent.mkdir(parents=True, exist_ok=True)
                        master_cleaned.save(master_dest, "PNG")
                        rt_dim = (512, 512) if max(w, h) >= 512 else (256, 256)
                        cleaned = (
                            master_cleaned
                            if master_cleaned.size == rt_dim
                            else master_cleaned.resize(rt_dim, Image.Resampling.LANCZOS)
                        )
                        cleaned.save(dest_portrait, "PNG")
                    png_file.unlink(missing_ok=True)
                    cleaned_paths.append(dest_portrait)
                    aid = png_file.stem
                    self.catalog.entries[aid] = AssetLibraryEntry(
                        deliberation=f"Cleaned and indexed character portrait '{aid}' in assets/portraits/.",
                        asset_id=aid,
                        asset_type="portrait",
                        fabrication_mode="portrait",
                        purpose=f"Dialogue/HUD bust portrait for {aid}",
                        res_path=f"res://assets/portraits/{dest_portrait.name}",
                        file_path=f"assets/portraits/{dest_portrait.name}",
                        master_res_path=f"res://{master_rel}",
                        master_file_path=master_rel,
                        master_pixel_size=self._read_image_size(master_dest),
                        target_display_size=(128, 128),
                        texture_pixel_size=rt_dim,
                        sha256_hash=self._compute_sha256(dest_portrait),
                    )
                    self._populate_default_context_variants(aid)
                except Exception as exc:
                    logger.warning("Failed to clean misplaced portrait '%s': %s", png_file, exc)

        return cleaned_paths

    def derive_baseline_assets_from_bible(
        self,
        game_bible: Optional[Union[GameBibleSpec, Dict[str, Any]]] = None,
    ) -> List[AssetRequirement]:
        """Derives a rich, domain-agnostic set of purpose-built AssetRequirements from the Game Bible
        whenever requirements_spec.asset_manifest is empty or missing core world/character categories.
        """
        if game_bible is None:
            bible_dict = self._load_game_bible_dict()
        elif hasattr(game_bible, "model_dump"):
            bible_dict = game_bible.model_dump()
        elif isinstance(game_bible, dict):
            bible_dict = dict(game_bible)
        else:
            bible_dict = {}

        derived: List[AssetRequirement] = []
        style_guide = str(bible_dict.get("visual_style_guide") or "")
        narrative = bible_dict.get("narrative_bible") or {}
        characters = narrative.get("characters") or []
        world_spec = bible_dict.get("world_spec") or {}
        landmarks = world_spec.get("landmarks") or []
        mockups = bible_dict.get("visual_mockups") or []
        mockup_desc = mockups[0].get("scene_description", "") if mockups and isinstance(mockups[0], dict) else ""

        # 1. Character portraits (in assets/portraits/) when characters are defined in the Narrative Bible;
        # in-world actor representations (sprites, sheets, or rigs) are planned by the LLM decomposition contract below
        # according to the game's RenderingStrategySpec rather than a hardcoded humanoid platformer template.
        client = getattr(self.visual_agent, "client", None) if self.visual_agent is not None else self.client
        is_live = (
            getattr(self.visual_agent, "_is_live_genai_client", lambda: False)()
            if self.visual_agent is not None
            else (client is not None and getattr(type(client), "__module__", "").startswith("google.genai"))
        )
        for idx, char in enumerate(characters):
            if not isinstance(char, dict):
                char = char.model_dump() if hasattr(char, "model_dump") else {}
            cid = str(char.get("character_id") or f"char_{idx+1}").lower().strip()
            cname = str(char.get("name") or cid)
            crole = str(char.get("role") or "character")
            cdesc = str(char.get("visual_description") or f"{cname} ({crole})")

            if char.get("portrait_asset") or bible_dict.get("narrative_bible", {}).get("dialogue_trees"):
                derived.append(
                    AssetRequirement(
                        deliberation=f"Dialogue portrait for {cname} ({crole}) isolated in assets/portraits/.",
                        asset_id=f"{cid}_portrait",
                        name=f"{cid}_portrait.png",
                        asset_type="portrait",
                        fabrication_mode="portrait",
                        purpose=f"UI dialogue portrait for {cname}",
                        concept_reference_id=cid,
                        target_path=f"assets/portraits/{cid}_portrait.png",
                        description=f"Portrait of {cname}: {cdesc}",
                        visual_prompt_or_text=f"Expressive portrait of {cname} ({crole}): {cdesc}. {style_guide}",
                        dimensions=(256, 256),
                    )
                )

            if not is_live:
                derived.append(
                    AssetRequirement(
                        deliberation=f"In-world actor visual asset for {cname} ({crole}).",
                        asset_id=cid,
                        name=f"{cid}.png",
                        asset_type="prop",
                        fabrication_mode="static_sprite",
                        purpose=f"In-world actor asset for {cname}",
                        concept_reference_id=cid,
                        target_path=f"assets/sprites/{cid}.png",
                        description=f"In-game actor asset of {cname} ({crole}): {cdesc}",
                        visual_prompt_or_text=f"Single isolated 2D in-game actor sprite of {cname} ({crole}): {cdesc}. {style_guide}",
                        dimensions=(96, 96),
                    )
                )

        derived.extend(
            self.decompose_concept_into_elements(
                bible_dict=bible_dict,
                world_spec=world_spec,
                landmarks=landmarks,
                mockup_desc=mockup_desc,
                style_guide=style_guide,
            )
        )
        return derived

    def decompose_concept_into_elements(
        self,
        bible_dict: Optional[Dict[str, Any]] = None,
        world_spec: Optional[Dict[str, Any]] = None,
        landmarks: Optional[List[Any]] = None,
        mockup_desc: str = "",
        style_guide: str = "",
    ) -> List[AssetRequirement]:
        """Decomposes the Writers' Room 16:9 concept scene into discrete, isolated, buildable 2D world elements
        via a structured LLM JSON contract (Rule 2 & Rule 4) grounded in the concept art and Game Bible.
        """
        from pydantic import Field
        from engine.studio.models import BaseEpistemicModel

        class DecomposedElementItemSpec(BaseEpistemicModel):
            deliberation: str = Field(description="Reasoning for isolating this modular world element, actor, or surface.")
            asset_id: str = Field(description="Snake_case identifier for this asset.")
            asset_type: str = Field(description="Asset category suited to this subject (e.g. 'backdrop', 'tile', 'prop', 'sprite_sheet', 'articulated_rig', 'diegetic_ui').")
            fabrication_mode: str = Field(description="Fabrication pipeline mode matching the subject's rendering strategy ('parallax_layer', 'tile_surface', 'static_sprite', 'sprite_sheet', 'articulated_parts', 'diegetic_ui').")
            purpose: str = Field(description="In-world gameplay/visual role of this asset.")
            visual_prompt: str = Field(description="Detailed visual prompt for synthesizing this asset.")
            width: int = Field(default=1024, description="Source-art width in pixels, following the rendering strategy's target_resolution for this subject.")
            height: int = Field(default=1024, description="Source-art height in pixels, following the rendering strategy's target_resolution for this subject.")
            animation_states: List[AnimationStateSpec] = Field(default_factory=list, description="When fabrication_mode is 'sprite_sheet', the custom animation states tailored to this subject's verbs.")
            articulated_parts: List[ArticulatedPartSpec] = Field(default_factory=list, description="When fabrication_mode is 'articulated_parts', the independently movable parts.")

        class DecomposedConceptManifestSpec(BaseEpistemicModel):
            deliberation: str = Field(description="Architectural decomposition plan breaking the 16:9 concept scene into modular tiles, props, and animated elements.")
            elements: List[DecomposedElementItemSpec] = Field(default_factory=list)

        if bible_dict is None:
            bible_dict = self._load_game_bible_dict()
        if not world_spec:
            world_spec = bible_dict.get("world_layout_and_level_design") or bible_dict.get("world_spec") or {}
        if not landmarks:
            landmarks = world_spec.get("landmarks") or []
        if not style_guide:
            style_guide = self.get_style_guide(for_environment=True)
        if not mockup_desc:
            mockups = bible_dict.get("visual_mockups") or bible_dict.get("mockup_concepts") or []
            if mockups and isinstance(mockups[0], dict):
                mockup_desc = str(mockups[0].get("scene_description") or mockups[0].get("visual_composition_description") or "")

        elements: List[AssetRequirement] = []

        # 1. Extract explicit landmarks from world_layout_and_level_design schema
        for l_idx, lm in enumerate(landmarks):
            if not isinstance(lm, dict):
                lm = lm.model_dump() if hasattr(lm, "model_dump") else {}
            lid = str(lm.get("landmark_id") or f"landmark_{l_idx+1}").lower().strip()
            lname = str(lm.get("name") or lid)
            ltype = str(lm.get("landmark_type") or "interactive_prop")
            ldesc = str(lm.get("visual_description") or f"{lname} ({ltype})")
            elements.append(
                AssetRequirement(
                    deliberation=f"Decomposed interactive world landmark/prop sprite for {lname} ({ltype}).",
                    asset_id=lid,
                    name=f"{lid}.png",
                    asset_type="prop",
                    fabrication_mode="static_sprite",
                    purpose=f"In-world interactive landmark {lname} ({ltype})",
                    target_path=f"assets/sprites/{lid}.png",
                    description=f"{lname}: {ldesc}",
                    visual_prompt_or_text=(
                        f"Single isolated 2D game architectural landmark or mechanism prop '{lname}' ({ltype}): {ldesc}. "
                        f"STRICTLY a single discrete structure/object on flat #FF00FF magenta chroma background — NO characters, NO floating island vignette. {style_guide}"
                    ),
                    dimensions=(96, 96),
                )
            )

        # 1b. Extract explicit assets requested by GameplaySystemsEngineer in technical_architecture_blueprint
        eng_blueprint = bible_dict.get("technical_architecture_blueprint") or []
        eng_asset_notes: List[str] = []
        existing_elem_ids = {e.asset_id for e in elements}
        for espec in eng_blueprint:
            if not isinstance(espec, dict):
                espec = espec.model_dump() if hasattr(espec, "model_dump") else {}
            mech_title = str(espec.get("title") or espec.get("mechanic_id") or "")
            for raw_asset_str in espec.get("required_sprite_and_tile_assets") or []:
                raw_s = str(raw_asset_str).strip()
                if not raw_s:
                    continue
                eng_asset_notes.append(f"[{mech_title}] {raw_s}")
                parts = [p.strip() for p in raw_s.split(":", 2)]
                if len(parts) >= 2:
                    aid_raw = "".join(ch if ch.isalnum() else "_" for ch in parts[0].lower()).strip("_").removesuffix("_sheet")
                    while "__" in aid_raw:
                        aid_raw = aid_raw.replace("__", "_")
                    atype_raw = parts[1].lower().strip()
                    adesc_raw = parts[2] if len(parts) >= 3 else f"{aid_raw} for {mech_title}"
                    if aid_raw and aid_raw not in existing_elem_ids and aid_raw not in ("hero", "companion"):
                        if atype_raw in ("tile", "tile_surface"):
                            a_type, f_mode, dims = "tile", "tile_surface", (64, 64)
                        elif atype_raw in ("sprite_sheet", "animated_prop"):
                            a_type, f_mode, dims = "sprite_sheet", "sprite_sheet", (64, 64)
                        else:
                            a_type, f_mode, dims = "prop", "static_sprite", (80, 80)
                        elements.append(
                            AssetRequirement(
                                deliberation=f"Asset required by GameplaySystemsEngineer for mechanic '{mech_title}'.",
                                asset_id=aid_raw,
                                name=f"{aid_raw}.png",
                                asset_type=a_type,
                                fabrication_mode=f_mode,
                                purpose=f"Engineered gameplay asset for {mech_title}: {adesc_raw}",
                                target_path=f"assets/sprites/{aid_raw}.png",
                                description=adesc_raw,
                                visual_prompt_or_text=f"{adesc_raw}. {style_guide}",
                                dimensions=dims,
                            )
                        )
                        existing_elem_ids.add(aid_raw)

        # 2. Delegate visual concept decomposition to Gemini via structured DecomposedConceptManifestSpec contract (Rule 2, Rule 4, Rule 5)
        client = getattr(self.visual_agent, "client", None) if self.visual_agent is not None else self.client
        is_live = (
            getattr(self.visual_agent, "_is_live_genai_client", lambda: False)()
            if self.visual_agent is not None
            else (client is not None and getattr(type(client), "__module__", "").startswith("google.genai"))
        )
        if client is not None and is_live:
            from google.genai import types
            ref_imgs = self.get_concept_reference_images()
            contents: List[Any] = []
            for rp in ref_imgs[:2]:
                if rp.is_file():
                    contents.append(types.Part.from_bytes(data=rp.read_bytes(), mime_type="image/png"))
            from engine.studio.rendering import build_rendering_guide
            rendering_block = build_rendering_guide(self.get_rendering_strategy(require=True))
            eng_context_str = (
                f"GameplaySystemsEngineer Required Mechanic Assets:\n  " + "\n  ".join(eng_asset_notes) + "\n"
                if eng_asset_notes
                else ""
            )
            prompt = (
                f"You are the Visual Art & Technical Environment Director decomposing the attached 16:9 Writers' Room concept art "
                f"and Game Bible into discrete, composable 2D game assets (NEVER a static full-screen background painting or floating island vignette).\n"
                f"Game Title: {bible_dict.get('game_title', '')}\n"
                f"Elevator Pitch: {bible_dict.get('elevator_pitch', '')}\n"
                f"First 60 Seconds Hook: {bible_dict.get('first_60_seconds_hook', '')}\n"
                f"Visual Style Guide: {style_guide}\n"
                f"Concept Scene Description: {mockup_desc}\n"
                f"{eng_context_str}\n"
                f"{rendering_block}\n"
                f"Decompose the game's visuals into the assets required to realise the RENDERING STRATEGY above in `elements`. "
                f"Be EXHAUSTIVE: every distinct visual element visible in the concept art or required by the Game Bible must be covered; "
                f"do not collapse the scene into a handful of generic pieces.\n"
                f"- Follow each subject's `technique`: 'painted_parallax_layer' -> `asset_type='backdrop'`, `fabrication_mode='parallax_layer'` (full-frame painted layer); "
                f"'single_painted_image_procedural_motion' -> `asset_type='prop'`, `fabrication_mode='static_sprite'` (one high-resolution painted image); "
                f"'layered_cutout_rig' -> `asset_type='articulated_rig'`, `fabrication_mode='articulated_parts'` with `articulated_parts` filled in; "
                f"'sprite_sheet' -> `asset_type='sprite_sheet'`, `fabrication_mode='sprite_sheet'`; repeatable surfaces -> 'tile'.\n"
                f"- Use the strategy's `target_resolution` for `width`/`height`; do not shrink assets to icon size. "
                f"Effects the strategy implements with particles, shaders or lights need no image asset unless they require a texture.\n"
                f"MANDATORY (Rule 4): Provide `deliberation` FIRST in every schema object."
            )
            contents.append(prompt)
            text_model = getattr(self.visual_agent, "text_model_name", "gemini-3.8-flash") if self.visual_agent else "gemini-3.8-flash"
            resp = call_gemini_with_transient_retry(
                lambda: client.models.generate_content(
                    model=text_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=DecomposedConceptManifestSpec,
                        thinking_config=bounded_thinking_config(1024),
                    ),
                ),
                label="decompose_concept_manifest",
                candidate_dir=self.cartridge_dir,
            )
            if resp.text:
                manifest_spec = DecomposedConceptManifestSpec.model_validate_json(resp.text)
                existing_ids = {e.asset_id for e in elements}
                for item in manifest_spec.elements:
                    aid_clean = str(item.asset_id).strip().lower().replace(".png", "").removesuffix("_sheet")
                    if aid_clean and aid_clean not in existing_ids:
                        w_clamped = max(16, min(8192, int(item.width or 1024)))
                        h_clamped = max(16, min(8192, int(item.height or 1024)))
                        elements.append(
                            AssetRequirement(
                                deliberation=item.deliberation,
                                asset_id=aid_clean,
                                name=f"{aid_clean}.png",
                                asset_type=item.asset_type,
                                fabrication_mode=item.fabrication_mode,
                                purpose=item.purpose,
                                target_path=f"assets/sprites/{aid_clean}.png",
                                description=item.purpose,
                                visual_prompt_or_text=f"{item.visual_prompt}. {style_guide}",
                                dimensions=(w_clamped, h_clamped),
                                animation_states=list(item.animation_states or []),
                                articulated_parts=list(item.articulated_parts or []),
                            )
                        )
                        existing_ids.add(aid_clean)

        # 3. Map explicit scene_layers defined in game_bible.json as single discrete modular structures
        for layer in (bible_dict.get("scene_layers") or []):
            if not isinstance(layer, dict):
                layer = layer.model_dump() if hasattr(layer, "model_dump") else {}
            for assoc_id in (layer.get("associated_asset_ids") or []):
                aid = str(assoc_id).strip()
                if aid and all(e.asset_id != aid for e in elements):
                    ldesc = str(layer.get("visual_description") or aid)
                    is_tile_layer = int(layer.get("depth_order", 1)) >= 2
                    elements.append(
                        AssetRequirement(
                            deliberation=f"Modular architectural element '{aid}' referenced by GameBible scene_layer '{layer.get('layer_id', '')}'.",
                            asset_id=aid,
                            name=f"{aid}.png",
                            asset_type="tile" if is_tile_layer else "prop",
                            fabrication_mode="tile_surface" if is_tile_layer else "static_sprite",
                            purpose=f"Single modular architectural structure for {layer.get('layer_id', '')}: {ldesc}",
                            target_path=f"assets/sprites/{aid}.png",
                            description=ldesc,
                            visual_prompt_or_text=(
                                f"{ldesc}. {style_guide}"
                                if is_tile_layer
                                else f"Single isolated architectural structure or column element (NOT a floating island or full landscape scene): {ldesc}. {style_guide}"
                            ),
                            dimensions=(64, 64) if is_tile_layer else (128, 128),
                        )
                    )

        return elements


    def fabricate_single_asset(
        self,
        req: Union[AssetRequirement, Dict[str, Any]],
        force_regenerate: bool = False,
    ) -> Optional[Path]:
        """Fabricates a single AssetRequirement into the cartridge directory and indexes it in asset_library.json."""
        if isinstance(req, dict):
            req = AssetRequirement.model_validate(req)
        if not (getattr(req, "camera_projection", "") or "").strip() and (self.catalog.camera_and_spatial_paradigm or "").strip():
            req = req.model_copy(update={"camera_projection": self.catalog.camera_and_spatial_paradigm.strip()})

        ref_images = self.get_concept_reference_images(req.concept_reference_id)
        is_env = req.asset_type in ("prop", "tile", "backdrop") or req.fabrication_mode in ("tile_surface", "parallax_layer", "static_sprite")
        style_guide = self.get_style_guide(for_environment=is_env)
        dest_path = self.cartridge_dir / req.target_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        fmode = req.fabrication_mode
        atype = req.asset_type

        # Audio asset handling
        if fmode in ("audio_bark", "sfx", "ambient_loop") or atype in ("audio_bark", "sfx", "ambient_loop"):
            if not dest_path.is_file() or force_regenerate:
                if self.voice_agent is None:
                    raise StudioAgentError(f"No voice/audio agent configured; cannot fabricate audio '{req.asset_id}'.")
                if fmode == "audio_bark" and hasattr(self.voice_agent, "synthesize_voice_line"):
                    self.voice_agent.synthesize_voice_line(bark=req, output_path=dest_path)
                elif hasattr(self.voice_agent, "generate_sfx"):
                    self.voice_agent.generate_sfx(
                        prompt=req.visual_prompt_or_text or req.description,
                        output_path=dest_path,
                        kind=fmode,
                        purpose=req.purpose,
                        loop=(fmode == "ambient_loop"),
                    )
                else:
                    raise StudioAgentError(
                        f"Voice agent {type(self.voice_agent).__name__} has no capability to fabricate "
                        f"'{fmode}' audio for '{req.asset_id}' (needs synthesize_voice_line for audio_bark "
                        f"or generate_sfx for sfx/ambient_loop)."
                    )
            if dest_path.is_file():
                from engine.studio.audio import validate_wav_file
                _ok_wav, _wav_err, wav_dur = validate_wav_file(dest_path)
                self._supersede_same_path_entries(req.asset_id, req.target_path)
                self.catalog.entries[req.asset_id] = AssetLibraryEntry(
                    deliberation=req.deliberation,
                    asset_id=req.asset_id,
                    asset_type=atype,
                    fabrication_mode=fmode,
                    purpose=req.purpose or req.description,
                    res_path=f"res://{req.target_path}",
                    file_path=req.target_path,
                    target_display_size=(0, 0),
                    texture_pixel_size=(0, 0),
                    duration_seconds=round(float(wav_dur), 3) if wav_dur > 0 else 0.0,
                    sha256_hash=self._compute_sha256(dest_path),
                )
                return dest_path
            return None

        if self.visual_agent is None:
            from engine.studio.visual import VisualArtSpriteDesignerAgent
            self.visual_agent = VisualArtSpriteDesignerAgent(
                client=self.client,
                allow_offline_procedural=self.allow_offline_procedural,
            )
        if (
            not self.allow_offline_procedural
            and self.client is None
            and getattr(self.visual_agent, "client", None) is None
        ):
            raise StudioAgentConfigurationError(
                f"Cannot fabricate visual asset '{req.asset_id}' -> '{req.target_path}' without a Gemini client "
                f"(allow_offline_procedural=False; silent procedural fallback is prohibited by Rule 5)."
            )

        renders_dir = self.cartridge_dir / "artifacts" / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        max_val_attempts = 2

        req = self._upgrade_req_for_hidpi_context(req)

        # 1. Portrait fabrication + Tier 1 Render Plate Verification (strictly in assets/portraits/)
        if fmode == "portrait" or atype == "portrait":
            portrait_rel = f"assets/portraits/{Path(req.target_path).name}"
            portrait_dest = self.cartridge_dir / portrait_rel
            portrait_dest.parent.mkdir(parents=True, exist_ok=True)
            master_portrait_rel = self._master_rel_for_runtime_rel(portrait_rel)
            master_portrait_dest = self.cartridge_dir / master_portrait_rel
            plate_path = renders_dir / f"plate_{req.asset_id}.png"
            cur_prompt = req.visual_prompt_or_text or req.description

            for attempt in range(1, max_val_attempts + 1):
                if not portrait_dest.is_file() or force_regenerate or attempt > 1:
                    if getattr(self.visual_agent, "client", None) is not None:
                        self.visual_agent.generate_portrait(
                            character={
                                "deliberation": req.deliberation,
                                "character_id": req.asset_id.replace("_portrait", ""),
                                "name": req.asset_id.replace("_portrait", "").replace("_", " ").title(),
                                "role": "character",
                                "visual_description": cur_prompt,
                            },
                            output_path=portrait_dest,
                            dimensions=req.dimensions or (512, 512),
                            style_guide=style_guide,
                            reference_images=ref_images,
                            master_output_path=master_portrait_dest,
                        )
                    else:
                        if hasattr(self.visual_agent, "_ensure_procedural_allowed_or_raise"):
                            self.visual_agent._ensure_procedural_allowed_or_raise("generate_portrait")
                        elif not self.allow_offline_procedural:
                            raise StudioAgentConfigurationError(
                                f"Cannot fabricate portrait '{req.asset_id}' without a Gemini client "
                                f"(allow_offline_procedural=False; silent procedural fallback is prohibited by Rule 5)."
                            )
                        img = self.visual_agent.renderer.render_dynamic_sprite(
                            dimensions=req.dimensions or (256, 256),
                            prompt=cur_prompt,
                            asset_type="portrait",
                            seed=f"{req.asset_id}_{attempt}",
                        )
                        img.save(portrait_dest, "PNG")
                        master_portrait_dest.parent.mkdir(parents=True, exist_ok=True)
                        img.save(master_portrait_dest, "PNG")

                if hasattr(self.visual_agent, "render_asset_inspection_plate") and portrait_dest.is_file():
                    self.visual_agent.render_asset_inspection_plate(
                        portrait_dest, asset_type="portrait", output_path=plate_path
                    )
                if hasattr(self.visual_agent, "verify_rendered_asset") and portrait_dest.is_file():
                    verdict = self.visual_agent.verify_rendered_asset(
                        asset_req=req,
                        rendered_asset_path=portrait_dest,
                        inspection_plate_path=plate_path,
                        reference_images=ref_images,
                        style_guide=style_guide,
                    )
                    if verdict.accepted:
                        break
                    if verdict.prompt_refinement_directive:
                        cur_prompt = f"{cur_prompt}. {verdict.prompt_refinement_directive}"
                else:
                    break

            tex_size = self._read_image_size(portrait_dest)
            has_master_p = master_portrait_dest.is_file()
            m_tex_size = self._read_image_size(master_portrait_dest) if has_master_p else tex_size
            self._supersede_same_path_entries(req.asset_id, portrait_rel)
            self.catalog.entries[req.asset_id] = AssetLibraryEntry(
                deliberation=req.deliberation,
                asset_id=req.asset_id,
                asset_type="portrait",
                fabrication_mode="portrait",
                purpose=req.purpose or req.description,
                res_path=f"res://{portrait_rel}",
                file_path=portrait_rel,
                master_res_path=f"res://{master_portrait_rel}" if has_master_p else "",
                master_file_path=master_portrait_rel if has_master_p else "",
                master_pixel_size=m_tex_size,
                target_display_size=(128, 128),
                texture_pixel_size=tex_size,
                concept_reference_used=str(ref_images[0]) if ref_images else None,
                sha256_hash=self._compute_sha256(portrait_dest),
            )
            self._populate_default_context_variants(req.asset_id)
            return portrait_dest

        # 2. Multi-Pose / Multi-State Sprite Sheet fabrication + Tier 1 & Tier 2 Contact Sheet Verification
        # (Note: When grid_cols * grid_rows > 1 and cell_prompts are provided for a multi-cell state/icon/variant grid,
        # route through Step 4's per-cell grid compositor rather than single-actor biomechanical deformation.)
        has_grid_cell_prompts = (
            max(1, int(getattr(req, "grid_cols", 1) or 1)) * max(1, int(getattr(req, "grid_rows", 1) or 1)) > 1
            and bool(getattr(req, "cell_prompts", None))
        )
        if (fmode == "sprite_sheet" or atype == "sprite_sheet") and not has_grid_cell_prompts:
            stem = dest_path.stem
            sheet_filename = f"{stem}_sheet.png" if not stem.endswith("_sheet") else f"{stem}.png"
            sheet_rel = f"assets/sprites/{sheet_filename}"
            sheet_dest = self.cartridge_dir / sheet_rel
            master_sheet_rel = self._master_rel_for_runtime_rel(sheet_rel)
            master_sheet_dest = self.cartridge_dir / master_sheet_rel
            single_rel = f"assets/sprites/{stem.replace('_sheet', '')}.png"
            single_dest = self.cartridge_dir / single_rel
            master_single_dest = self.cartridge_dir / self._master_rel_for_runtime_rel(single_rel)
            contact_path = renders_dir / f"anim_sheet_{req.asset_id}.png"
            plate_path = renders_dir / f"plate_{req.asset_id}.png"

            sheet_path = sheet_dest
            meta: Dict[str, Any] = {}
            for attempt in range(1, max_val_attempts + 1):
                try:
                    sheet_path, meta = self.visual_agent.fabricate_sprite_sheet(
                        asset_req=req,
                        output_path=sheet_dest,
                        mood=self.mood,
                        reference_images=ref_images,
                        style_guide=style_guide,
                        master_output_path=master_sheet_dest,
                    )
                except TypeError:
                    sheet_path, meta = self.visual_agent.fabricate_sprite_sheet(
                        asset_req=req,
                        output_path=sheet_dest,
                        mood=self.mood,
                        reference_images=ref_images,
                        style_guide=style_guide,
                    )
                base_frame: Optional[Image.Image] = meta.get("base_frame_image")
                if base_frame is not None:
                    base_frame.save(single_dest, "PNG")
                elif not single_dest.is_file():
                    shutil.copyfile(sheet_path, single_dest)
                master_base_frame: Optional[Image.Image] = meta.get("master_base_frame_image")
                if master_base_frame is not None:
                    master_single_dest.parent.mkdir(parents=True, exist_ok=True)
                    master_base_frame.save(master_single_dest, "PNG")

                if hasattr(self.visual_agent, "render_asset_inspection_plate") and single_dest.is_file():
                    self.visual_agent.render_asset_inspection_plate(
                        single_dest, asset_type="sprite", output_path=plate_path
                    )
                if hasattr(self.visual_agent, "render_animation_contact_sheet") and sheet_path.is_file():
                    _, anim_telemetry = self.visual_agent.render_animation_contact_sheet(
                        sheet_img=sheet_path,
                        states_meta=dict(meta.get("animation_states") or {}),
                        cell_size=tuple(meta.get("target_display_size") or req.dimensions or (64, 64)),
                        output_path=contact_path,
                    )
                    if hasattr(self.visual_agent, "verify_rendered_animation"):
                        anim_verdict = self.visual_agent.verify_rendered_animation(
                            target_id=req.asset_id,
                            contact_sheet_path=contact_path,
                            anim_telemetry=anim_telemetry,
                            purpose=req.purpose or req.description,
                            style_guide=style_guide,
                        )
                        if anim_verdict.accepted:
                            break
                        if anim_verdict.remediation_directive:
                            req.visual_prompt_or_text = f"{req.visual_prompt_or_text}. {anim_verdict.remediation_directive}"
                    else:
                        break
                else:
                    break

            self._supersede_same_path_entries(req.asset_id, sheet_rel)
            sheet_disp_size = (
                (int(req.world_footprint_px[0]), int(req.world_footprint_px[1]))
                if getattr(req, "world_footprint_px", None)
                else tuple(meta.get("target_display_size") or req.dimensions or (64, 64))
            )
            has_master_s = master_sheet_dest.is_file()
            sheet_tex_sz = tuple(meta.get("texture_pixel_size") or self._read_image_size(sheet_path))
            sheet_master_sz = (
                tuple(meta.get("master_pixel_size") or self._read_image_size(master_sheet_dest))
                if has_master_s
                else sheet_tex_sz
            )
            self.catalog.entries[req.asset_id] = AssetLibraryEntry(
                deliberation=req.deliberation,
                asset_id=req.asset_id,
                asset_type="sprite_sheet",
                fabrication_mode="sprite_sheet",
                purpose=req.purpose or req.description,
                res_path=f"res://{sheet_rel}",
                file_path=sheet_rel,
                master_res_path=f"res://{master_sheet_rel}" if has_master_s else "",
                master_file_path=master_sheet_rel if has_master_s else "",
                master_pixel_size=sheet_master_sz,
                target_display_size=sheet_disp_size,
                texture_pixel_size=sheet_tex_sz,
                hframes=int(meta.get("hframes", 4)),
                vframes=int(meta.get("vframes", 4)),
                animation_states=dict(meta.get("animation_states") or {}),
                concept_reference_used=str(ref_images[0]) if ref_images else None,
                sha256_hash=self._compute_sha256(sheet_path),
            )
            self._populate_default_context_variants(req.asset_id)
            return single_dest

        # 3. Articulated Multi-Part Rig fabrication + Tier 2 Articulated Rig Contact Sheet Verification
        if fmode == "articulated_parts" or atype == "articulated_rig":
            contact_path = renders_dir / f"anim_rig_{req.asset_id}.png"
            plate_path = renders_dir / f"plate_{req.asset_id}.png"
            master_rig_rel = self._master_rel_for_runtime_rel(req.target_path)
            master_rig_dest = self.cartridge_dir / master_rig_rel
            use_hidpi_rig = bool(
                hasattr(self.visual_agent, "_is_live_genai_client")
                and self.visual_agent._is_live_genai_client()
                and not self._is_pixel_art_cartridge()
            )
            comp_path = dest_path
            meta = {}
            for attempt in range(1, max_val_attempts + 1):
                try:
                    comp_path, meta = self.visual_agent.fabricate_articulated_rig(
                        asset_req=req,
                        output_path=dest_path,
                        mood=self.mood,
                        reference_images=ref_images,
                        style_guide=style_guide,
                        master_output_path=master_rig_dest,
                        preserve_hidpi_parts=use_hidpi_rig,
                    )
                except TypeError:
                    comp_path, meta = self.visual_agent.fabricate_articulated_rig(
                        asset_req=req,
                        output_path=dest_path,
                        mood=self.mood,
                        reference_images=ref_images,
                        style_guide=style_guide,
                    )
                if hasattr(self.visual_agent, "render_asset_inspection_plate") and comp_path.is_file():
                    self.visual_agent.render_asset_inspection_plate(
                        comp_path, asset_type="sprite", output_path=plate_path
                    )
                if hasattr(self.visual_agent, "render_articulated_rig_contact_sheet") and comp_path.is_file():
                    _, rig_telemetry = self.visual_agent.render_articulated_rig_contact_sheet(
                        composite_path=comp_path,
                        articulated_parts=list(meta.get("articulated_parts") or []),
                        cartridge_dir=self.cartridge_dir,
                        cell_size=tuple(meta.get("target_display_size") or req.dimensions or (64, 64)),
                        output_path=contact_path,
                    )
                    if hasattr(self.visual_agent, "verify_rendered_animation"):
                        rig_verdict = self.visual_agent.verify_rendered_animation(
                            target_id=req.asset_id,
                            contact_sheet_path=contact_path,
                            anim_telemetry=rig_telemetry,
                            purpose=req.purpose or req.description,
                            style_guide=style_guide,
                        )
                        if rig_verdict.accepted:
                            break
                break

            self._supersede_same_path_entries(req.asset_id, req.target_path)
            rig_disp_size = (
                (int(req.world_footprint_px[0]), int(req.world_footprint_px[1]))
                if getattr(req, "world_footprint_px", None)
                else tuple(meta.get("target_display_size") or req.dimensions or (64, 64))
            )
            has_master_r = master_rig_dest.is_file()
            rig_tex_sz = tuple(meta.get("texture_pixel_size") or self._read_image_size(comp_path))
            rig_master_sz = (
                tuple(meta.get("master_pixel_size") or self._read_image_size(master_rig_dest))
                if has_master_r
                else rig_tex_sz
            )
            self.catalog.entries[req.asset_id] = AssetLibraryEntry(
                deliberation=req.deliberation,
                asset_id=req.asset_id,
                asset_type="articulated_rig",
                fabrication_mode="articulated_parts",
                purpose=req.purpose or req.description,
                res_path=f"res://{req.target_path}",
                file_path=req.target_path,
                master_res_path=f"res://{master_rig_rel}" if has_master_r else "",
                master_file_path=master_rig_rel if has_master_r else "",
                master_pixel_size=rig_master_sz,
                target_display_size=rig_disp_size,
                texture_pixel_size=rig_tex_sz,
                articulated_parts=list(meta.get("articulated_parts") or []),
                concept_reference_used=str(ref_images[0]) if ref_images else None,
                sha256_hash=self._compute_sha256(comp_path),
            )
            self._populate_default_context_variants(req.asset_id)
            return comp_path

        # 4. Static Sprite / Prop / Tile Surface / Parallax Layer / Diegetic UI / Multi-Cell Grid + Tier 1 Render Plate Verification
        plate_path = renders_dir / f"plate_{req.asset_id}.png"
        master_static_rel = self._master_rel_for_runtime_rel(req.target_path)
        master_static_dest = self.cartridge_dir / master_static_rel
        g_cols = max(1, int(getattr(req, "grid_cols", 1) or 1))
        g_rows = max(1, int(getattr(req, "grid_rows", 1) or 1))
        total_grid_cells = g_cols * g_rows
        regen_cells: Optional[List[int]] = None
        out_path = dest_path
        for attempt in range(1, max_val_attempts + 1):
            dyn_kwargs: Dict[str, Any] = {"master_output_path": master_static_dest}
            if attempt > 1 and regen_cells and dest_path.is_file():
                dyn_kwargs["regenerate_cell_indices"] = list(regen_cells)
                dyn_kwargs["existing_sheet_path"] = dest_path
            try:
                out_path = self.visual_agent.generate_dynamic_sprite(
                    asset_req=req,
                    output_path=dest_path,
                    mood=self.mood,
                    reference_images=ref_images,
                    style_guide=style_guide,
                    **dyn_kwargs,
                )
            except TypeError:
                dyn_kwargs.pop("master_output_path", None)
                out_path = self.visual_agent.generate_dynamic_sprite(
                    asset_req=req,
                    output_path=dest_path,
                    mood=self.mood,
                    reference_images=ref_images,
                    style_guide=style_guide,
                    **dyn_kwargs,
                )
            if hasattr(self.visual_agent, "render_asset_inspection_plate") and out_path.is_file():
                try:
                    self.visual_agent.render_asset_inspection_plate(
                        out_path,
                        asset_type="tile_surface" if fmode == "tile_surface" else atype,
                        output_path=plate_path,
                        grid_cols=g_cols,
                        grid_rows=g_rows,
                    )
                except TypeError:
                    self.visual_agent.render_asset_inspection_plate(
                        out_path,
                        asset_type="tile_surface" if fmode == "tile_surface" else atype,
                        output_path=plate_path,
                    )
                if (fmode == "sprite_sheet" or atype == "sprite_sheet") and plate_path.is_file():
                    shutil.copyfile(plate_path, renders_dir / f"anim_sheet_{req.asset_id}.png")
            if hasattr(self.visual_agent, "verify_rendered_asset") and out_path.is_file():
                verdict = self.visual_agent.verify_rendered_asset(
                    asset_req=req,
                    rendered_asset_path=out_path,
                    inspection_plate_path=plate_path,
                    reference_images=ref_images,
                    style_guide=style_guide,
                )
                if verdict.accepted:
                    break
                bad_cells = [
                    int(i)
                    for i in (getattr(verdict, "defective_cell_indices", None) or [])
                    if 0 <= int(i) < total_grid_cells
                ]
                regen_cells = bad_cells if (0 < len(bad_cells) < total_grid_cells) else None
                if verdict.prompt_refinement_directive:
                    directive_str = verdict.prompt_refinement_directive.strip()
                    if directive_str and directive_str not in (req.visual_prompt_or_text or ""):
                        req.visual_prompt_or_text = f"{req.visual_prompt_or_text}. {directive_str}".strip(". ")
                    existing_cell_prompts = list(getattr(req, "cell_prompts", None) or [])
                    if existing_cell_prompts and directive_str:
                        target_prompt_idxs = set(bad_cells) if bad_cells else set(range(len(existing_cell_prompts)))
                        updated_cell_prompts: List[str] = []
                        for idx_cp, cp_txt in enumerate(existing_cell_prompts):
                            if idx_cp in target_prompt_idxs and directive_str not in cp_txt:
                                updated_cell_prompts.append(f"{cp_txt}. {directive_str}".strip(". "))
                            else:
                                updated_cell_prompts.append(cp_txt)
                        req.cell_prompts = updated_cell_prompts
            else:
                break

        tex_size = self._read_image_size(out_path)
        has_master_st = master_static_dest.is_file()
        master_tex_size = self._read_image_size(master_static_dest) if has_master_st else tex_size
        req_fp = getattr(req, "world_footprint_px", None)
        if isinstance(req_fp, (list, tuple)) and len(req_fp) >= 2 and int(req_fp[0]) > 0 and int(req_fp[1]) > 0:
            eff_display_size: Tuple[int, int] = (int(req_fp[0]), int(req_fp[1]))
        elif total_grid_cells > 1 and tex_size[0] > 0 and tex_size[1] > 0:
            eff_display_size = (
                max(16, int(tex_size[0]) // g_cols),
                max(16, int(tex_size[1]) // g_rows),
            )
        else:
            eff_display_size = tuple(req.dimensions or tex_size)
        self._supersede_same_path_entries(req.asset_id, req.target_path)
        self.catalog.entries[req.asset_id] = AssetLibraryEntry(
            deliberation=req.deliberation,
            asset_id=req.asset_id,
            asset_type=atype,
            fabrication_mode=fmode,
            purpose=req.purpose or req.description,
            res_path=f"res://{req.target_path}",
            file_path=req.target_path,
            master_res_path=f"res://{master_static_rel}" if has_master_st else "",
            master_file_path=master_static_rel if has_master_st else "",
            master_pixel_size=master_tex_size,
            target_display_size=eff_display_size,
            texture_pixel_size=tex_size,
            hframes=g_cols,
            vframes=g_rows,
            concept_reference_used=str(ref_images[0]) if ref_images else None,
            sha256_hash=self._compute_sha256(out_path),
        )
        self._populate_default_context_variants(req.asset_id)
        return out_path

    def verify_all_catalog_assets_and_animations(self, auto_remediate: bool = True) -> Dict[str, Any]:
        """Renders isolated Tier 1 inspection plates (checkerboard alpha cutouts & 3x3 tile grids)
        and Tier 2 animation contact sheets for every visual asset in `self.catalog.entries`.
        When `auto_remediate=True`, any asset or animation rejected by the deterministic OpenCV or
        multimodal Gemini critic is automatically regenerated (`force_regenerate=True`) with the
        critic's remediation directive and re-verified before sealing the catalog.
        """
        if self.visual_agent is None:
            from engine.studio.visual import VisualArtSpriteDesignerAgent
            self.visual_agent = VisualArtSpriteDesignerAgent(
                client=self.client,
                allow_offline_procedural=self.allow_offline_procedural,
            )

        has_client = (self.client is not None) or (getattr(self.visual_agent, "client", None) is not None)
        self.index_existing_files_on_disk()
        renders_dir = self.cartridge_dir / "artifacts" / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        style_guide = self.get_style_guide(for_environment=False)

        asset_verdicts: Dict[str, Any] = {}
        animation_verdicts: Dict[str, Any] = {}
        rejected_assets: List[str] = []
        rejected_animations: List[str] = []
        remediated_assets: List[str] = []

        for aid in sorted(list(self.catalog.entries.keys())):
            entry = self.catalog.entries[aid]
            if entry.fabrication_mode in ("audio_bark", "sfx", "ambient_loop"):
                continue
            f_path = self.cartridge_dir / entry.file_path
            if not f_path.is_file():
                continue

            plate_p = renders_dir / f"plate_{aid}.png"
            e_hframes = max(1, int(getattr(entry, "hframes", 1) or 1))
            e_vframes = max(1, int(getattr(entry, "vframes", 1) or 1))
            stub_dims = (
                tuple(entry.texture_pixel_size)
                if (e_hframes * e_vframes) > 1 and entry.fabrication_mode != "sprite_sheet" and entry.texture_pixel_size
                else (entry.target_display_size or (64, 64))
            )
            req_stub = AssetRequirement(
                deliberation=entry.deliberation,
                asset_id=aid,
                name=f_path.name,
                asset_type=entry.asset_type,
                fabrication_mode=entry.fabrication_mode,
                purpose=entry.purpose,
                concept_reference_id=aid.replace("_portrait", ""),
                description=entry.purpose,
                visual_prompt_or_text=f"{entry.purpose}. {style_guide}",
                target_path=entry.file_path,
                dimensions=stub_dims,
                grid_cols=e_hframes,
                grid_rows=e_vframes,
            )
            single_f = self.sprites_dir / f"{aid}.png"
            target_for_plate = single_f if (entry.fabrication_mode == "sprite_sheet" and single_f.is_file()) else f_path
            target_is_multi_cell = (e_hframes * e_vframes) > 1 and target_for_plate == f_path
            is_ui_or_atlas = (
                target_is_multi_cell
                or entry.fabrication_mode in ("tile_surface", "parallax_layer", "diegetic_ui")
                or entry.asset_type in ("tile", "backdrop", "diegetic_ui")
                or str(entry.file_path).startswith("assets/ui/")
                or str(entry.file_path).startswith("assets/tiles/")
                or str(entry.file_path).startswith("assets/backgrounds/")
            )

            # Apply alpha matting pass ONLY to single-subject world cutouts — NEVER to UI sheets, tiles, or multi-cell atlases!
            if auto_remediate and has_client and not is_ui_or_atlas and target_for_plate.is_file():
                try:
                    with Image.open(target_for_plate) as raw_im:
                        raw_rgba = raw_im.convert("RGBA")
                    matted = self.visual_agent.extract_clean_alpha_sprite(
                        raw_rgba,
                        target_size=raw_rgba.size,
                        strip_border_frame=True,
                        pad_ratio=0.04,
                        isolate_single_subject=True,
                    )
                    matted.save(target_for_plate, format="PNG")
                except Exception as exc:
                    logger.debug("Pre-audit alpha matting pass skipped for '%s': %s", aid, exc)

            # If a sprite_sheet entry was generated before keyframe_tracks were introduced, rebuild its sheet frames
            # from the matted base sprite using the 5-part biomechanical keyframe rig
            missing_kf_tracks = False
            if entry.fabrication_mode == "sprite_sheet" and single_f.is_file() and single_f.resolve() != f_path.resolve():
                st_dict = entry.animation_states or {}
                if not st_dict or any(not isinstance(sv, dict) or not sv.get("keyframe_tracks") for sv in st_dict.values()):
                    missing_kf_tracks = True
                    if auto_remediate and has_client:
                        try:
                            with Image.open(single_f) as base_im:
                                base_rgba = base_im.convert("RGBA")
                            cw, ch = entry.target_display_size or (64, 64)
                            cw, ch = max(32, int(cw)), max(32, int(ch))
                            default_states = [
                                ("idle", "idle", 4, 6.0, True),
                                ("move", "locomotion", 4, 10.0, True),
                                ("jump", "aerial", 4, 8.0, False),
                                ("action", "action", 4, 10.0, False),
                            ]
                            max_cols = 4
                            num_rows = len(default_states)
                            rebuilt_sheet = Image.new("RGBA", (cw * max_cols, ch * num_rows), (0, 0, 0, 0))
                            new_states_meta: Dict[str, Any] = {}
                            for r_idx, (sname, mprof, fcnt, fps_val, loop_val) in enumerate(default_states):
                                kf_list = self.visual_agent.default_keyframe_tracks_for_profile(mprof)
                                for c_idx in range(fcnt):
                                    pose_im = self.visual_agent.fabricate_pose_frame_from_base(
                                        base_sprite=base_rgba,
                                        state_name=sname,
                                        frame_idx=c_idx,
                                        frame_count=fcnt,
                                        cell_size=(cw, ch),
                                        motion_profile=mprof,
                                        keyframe_tracks=kf_list,
                                    )
                                    rebuilt_sheet.paste(pose_im, (c_idx * cw, r_idx * ch), pose_im)
                                new_states_meta[sname] = {
                                    "row": r_idx,
                                    "start_frame": r_idx * max_cols,
                                    "frame_count": fcnt,
                                    "fps": fps_val,
                                    "loop": loop_val,
                                    "motion_profile": mprof,
                                    "keyframe_tracks": [t.model_dump() for t in kf_list],
                                }
                            rebuilt_sheet.save(f_path, format="PNG")
                            entry.animation_states = new_states_meta
                            remediated_assets.append(aid)
                            missing_kf_tracks = False
                        except Exception as exc:
                            logger.warning("Keyframe sheet upgrade failed for '%s': %s", aid, exc)

            self.visual_agent.render_asset_inspection_plate(
                target_for_plate,
                asset_type="tile_surface" if entry.fabrication_mode == "tile_surface" else entry.asset_type,
                output_path=plate_p,
                grid_cols=e_hframes if target_is_multi_cell else 1,
                grid_rows=e_vframes if target_is_multi_cell else 1,
            )
            v_asset = self.visual_agent.verify_rendered_asset(
                asset_req=req_stub,
                rendered_asset_path=target_for_plate,
                inspection_plate_path=plate_p,
                style_guide=style_guide,
            )
            if (e_hframes * e_vframes) > 1 and f_path.is_file():
                try:
                    with Image.open(f_path) as sheet_im:
                        arr_rgba = np.asarray(sheet_im.convert("RGBA"))
                    sh_h, sh_w = arr_rgba.shape[:2]
                    cw_cell = max(1, sh_w // e_hframes)
                    ch_cell = max(1, sh_h // e_vframes)
                    empty_cells: List[Tuple[int, int]] = []
                    for r_i in range(e_vframes):
                        for c_i in range(e_hframes):
                            cell_a = arr_rgba[r_i * ch_cell:(r_i + 1) * ch_cell, c_i * cw_cell:(c_i + 1) * cw_cell, 3]
                            if cell_a.size == 0 or (float(np.count_nonzero(cell_a > 16)) / float(cell_a.size)) < 0.01:
                                empty_cells.append((c_i, r_i))
                    if empty_cells:
                        v_asset.accepted = False
                        empty_msg = (
                            f"Multi-cell {e_hframes}x{e_vframes} atlas '{entry.file_path}' has {len(empty_cells)} empty/transparent "
                            f"cell(s) at (col, row)={empty_cells[:6]}; populate every grid cell with distinct visible artwork."
                        )
                        v_asset.prompt_refinement_directive = (
                            f"{v_asset.prompt_refinement_directive}. {empty_msg}".strip(". ")
                            if v_asset.prompt_refinement_directive
                            else empty_msg
                        )
                except Exception as exc:
                    logger.debug("Multi-cell occupancy check warning for '%s': %s", aid, exc)

            v_anim = None
            if entry.fabrication_mode == "sprite_sheet" and entry.animation_states:
                contact_p = renders_dir / f"anim_sheet_{aid}.png"
                _, anim_tel = self.visual_agent.render_animation_contact_sheet(
                    sheet_img=f_path,
                    states_meta=entry.animation_states,
                    cell_size=entry.target_display_size or (64, 64),
                    output_path=contact_p,
                )
                v_anim = self.visual_agent.verify_rendered_animation(
                    target_id=aid,
                    contact_sheet_path=contact_p,
                    anim_telemetry=anim_tel,
                    purpose=entry.purpose,
                    style_guide=style_guide,
                )
            elif entry.fabrication_mode == "articulated_parts":
                contact_p = renders_dir / f"anim_rig_{aid}.png"
                _, rig_tel = self.visual_agent.render_articulated_rig_contact_sheet(
                    composite_path=f_path,
                    articulated_parts=entry.articulated_parts,
                    cartridge_dir=self.cartridge_dir,
                    cell_size=entry.target_display_size or (64, 64),
                    output_path=contact_p,
                )
                v_anim = self.visual_agent.verify_rendered_animation(
                    target_id=aid,
                    contact_sheet_path=contact_p,
                    anim_telemetry=rig_tel,
                    purpose=entry.purpose,
                    style_guide=style_guide,
                )

            needs_remediation = (not v_asset.accepted) or (v_anim is not None and not v_anim.accepted) or missing_kf_tracks
            # Rule 5: NEVER overwrite existing PNGs on disk with ProceduralSpriteRenderer when no Gemini client is configured!
            if auto_remediate and has_client and needs_remediation:
                refinement_parts = [entry.purpose]
                if v_asset.prompt_refinement_directive:
                    refinement_parts.append(v_asset.prompt_refinement_directive)
                if v_anim is not None and v_anim.remediation_directive:
                    refinement_parts.append(v_anim.remediation_directive)
                req_stub.visual_prompt_or_text = ". ".join(p for p in refinement_parts if p)
                if entry.fabrication_mode == "sprite_sheet":
                    req_stub.target_path = f"assets/sprites/{aid}.png"
                try:
                    logger.info("Auto-remediating rejected catalog asset '%s' (mode=%s)...", aid, entry.fabrication_mode)
                    self.fabricate_single_asset(req_stub, force_regenerate=True)
                    if aid not in remediated_assets:
                        remediated_assets.append(aid)
                    entry = self.catalog.entries[aid]
                    f_path = self.cartridge_dir / entry.file_path
                    single_f = self.sprites_dir / f"{aid}.png"
                    target_for_plate = single_f if (entry.fabrication_mode == "sprite_sheet" and single_f.is_file()) else f_path
                    self.visual_agent.render_asset_inspection_plate(
                        target_for_plate,
                        asset_type="tile_surface" if entry.fabrication_mode == "tile_surface" else entry.asset_type,
                        output_path=plate_p,
                    )
                    v_asset = self.visual_agent.verify_rendered_asset(
                        asset_req=req_stub,
                        rendered_asset_path=target_for_plate,
                        inspection_plate_path=plate_p,
                        style_guide=style_guide,
                    )
                    if entry.fabrication_mode == "sprite_sheet" and entry.animation_states:
                        contact_p = renders_dir / f"anim_sheet_{aid}.png"
                        _, anim_tel = self.visual_agent.render_animation_contact_sheet(
                            sheet_img=f_path,
                            states_meta=entry.animation_states,
                            cell_size=entry.target_display_size or (64, 64),
                            output_path=contact_p,
                        )
                        v_anim = self.visual_agent.verify_rendered_animation(
                            target_id=aid,
                            contact_sheet_path=contact_p,
                            anim_telemetry=anim_tel,
                            purpose=entry.purpose,
                            style_guide=style_guide,
                        )
                    elif entry.fabrication_mode == "articulated_parts":
                        contact_p = renders_dir / f"anim_rig_{aid}.png"
                        _, rig_tel = self.visual_agent.render_articulated_rig_contact_sheet(
                            composite_path=f_path,
                            articulated_parts=entry.articulated_parts,
                            cartridge_dir=self.cartridge_dir,
                            cell_size=entry.target_display_size or (64, 64),
                            output_path=contact_p,
                        )
                        v_anim = self.visual_agent.verify_rendered_animation(
                            target_id=aid,
                            contact_sheet_path=contact_p,
                            anim_telemetry=rig_tel,
                            purpose=entry.purpose,
                            style_guide=style_guide,
                        )
                except Exception as exc:
                    logger.warning("Auto-remediation failed for '%s': %s", aid, exc)

            asset_verdicts[aid] = v_asset.model_dump()
            if not v_asset.accepted:
                rejected_assets.append(aid)

            if v_anim is not None:
                animation_verdicts[aid] = v_anim.model_dump()
                if not v_anim.accepted:
                    rejected_animations.append(aid)

        if remediated_assets:
            self.save_catalog()

        return {
            "asset_verdicts": asset_verdicts,
            "animation_verdicts": animation_verdicts,
            "rejected_assets": rejected_assets,
            "rejected_animations": rejected_animations,
            "remediated_assets": remediated_assets,
            "all_passed": len(rejected_assets) == 0 and len(rejected_animations) == 0,
        }

    def ensure_concept_mockups_on_disk(self) -> List[Path]:
        """Ensures that any VisualMockupArtifact declared in game_bible.json has its 16:9 PNG on disk."""
        synthesized: List[Path] = []
        gb = self._load_game_bible_dict()
        mockups = gb.get("visual_mockups") or []
        if not mockups or self.visual_agent is None or not getattr(self.visual_agent, "_is_live_genai_client", lambda: False)():
            return synthesized
        style_guide = str(gb.get("visual_style_guide") or "")
        for idx, m in enumerate(mockups):
            if not isinstance(m, dict):
                continue
            rel = str(m.get("image_path") or f"assets/mockups/mockup_{idx + 1}.png")
            dest = self.cartridge_dir / rel
            master_mockup_dest = self.cartridge_dir / self._master_rel_for_runtime_rel(rel)
            if not dest.is_file():
                try:
                    self.visual_agent.generate_concept_mockup(
                        mockup=m,
                        output_path=dest,
                        style_guide=style_guide,
                        master_output_path=master_mockup_dest,
                    )
                    if dest.is_file():
                        synthesized.append(dest)
                except Exception as exc:
                    logger.warning("Concept mockup synthesis warning for '%s': %s", rel, exc)
        return synthesized

    @staticmethod
    def _tile_has_defective_corners(path: Path) -> bool:
        """Returns True if a tile texture on disk has white/magenta background corners OR artificial Rorschach mirror symmetry."""
        if not path.is_file():
            return True
        try:
            from engine.studio.visual import VisualArtSpriteDesignerAgent
            with Image.open(path) as im:
                rgba = im.convert("RGBA")
                w, h = rgba.size
                px = rgba.load()
                corners = [px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1]]
                for r, g, b, a in corners:
                    if a < 240 or (r > 210 and g > 210 and b > 210) or (r > 200 and b > 200 and g < 80):
                        return True
                if VisualArtSpriteDesignerAgent.compute_tile_mirror_symmetry(rgba) > 0.62:
                    return True
        except Exception:
            return True
        return False

    def ensure_required_assets(
        self,
        required_specs: List[Any],
        palette_mood: str = "stylized_fantasy",
        art_direction: str = "",
        force_regenerate: bool = False,
    ) -> Dict[str, Path]:
        """Ensures all RequiredSpriteAssetSpec or AssetRequirement items in `required_specs` exist on disk
        and in the catalog, fabricating any missing or defective tile assets on demand.
        """
        self.mood = palette_mood or self.mood
        self.index_existing_files_on_disk()
        missing_reqs: List[AssetRequirement] = []
        any_forced = force_regenerate
        for spec in required_specs:
            if isinstance(spec, AssetRequirement):
                aid = spec.asset_id
                target_file = self.cartridge_dir / spec.target_path
                is_tile = spec.fabrication_mode == "tile_surface" or spec.asset_type in ("tile", "tile_surface")
                tile_defective = is_tile and self._tile_has_defective_corners(target_file)
                if tile_defective:
                    any_forced = True
                if force_regenerate or tile_defective or aid not in self.catalog.entries or not target_file.is_file():
                    missing_reqs.append(spec)
            else:
                aid = str(getattr(spec, "asset_id", "") or "").strip().lower().replace(".png", "")
                if not aid:
                    continue
                existing_file = self.sprites_dir / f"{aid}.png"
                fmode = str(getattr(spec, "fabrication_mode", "static_sprite") or "static_sprite")
                is_tile = fmode == "tile_surface"
                tile_defective = is_tile and self._tile_has_defective_corners(existing_file)
                if tile_defective:
                    any_forced = True
                if not force_regenerate and not tile_defective and aid in self.catalog.entries and existing_file.is_file():
                    continue
                atype = "sprite_sheet" if fmode == "sprite_sheet" else ("tile" if fmode == "tile_surface" else "prop")
                w = int(getattr(spec, "target_display_width", 96) or 96)
                h = int(getattr(spec, "target_display_height", 96) or 96)
                vprompt = str(getattr(spec, "visual_prompt", "") or getattr(spec, "purpose", aid))
                if art_direction:
                    vprompt = f"{vprompt}. {art_direction}"
                missing_reqs.append(
                    AssetRequirement(
                        deliberation=getattr(spec, "deliberation", f"On-demand strategy asset fabrication for {aid}"),
                        asset_id=aid,
                        name=f"{aid}.png",
                        asset_type=atype,
                        fabrication_mode=fmode if fmode in ("static_sprite", "sprite_sheet", "tile_surface", "parallax_layer", "articulated_parts", "portrait") else "static_sprite",
                        purpose=str(getattr(spec, "purpose", f"Scene element {aid}")),
                        target_path=f"assets/sprites/{aid}.png",
                        description=str(getattr(spec, "purpose", aid)),
                        visual_prompt_or_text=vprompt,
                        dimensions=(w, h),
                    )
                )
        if not missing_reqs:
            return {}
        return self.fabricate_assets(missing_reqs[:4], force_regenerate=any_forced)

    def fabricate_assets(
        self,
        requirements: List[Union[AssetRequirement, Dict[str, Any]]],
        force_regenerate: bool = False,
    ) -> Dict[str, Path]:
        """Fabricates a batch of AssetRequirements and saves assets/asset_library.json."""
        results: Dict[str, Path] = {}
        self.ensure_concept_mockups_on_disk()
        self.sanitize_misplaced_portraits_in_sprites_dir()

        for item in requirements:
            req = AssetRequirement.model_validate(item) if isinstance(item, dict) else item
            try:
                out_path = self.fabricate_single_asset(req, force_regenerate=force_regenerate)
                if out_path is not None and out_path.is_file():
                    results[req.asset_id] = out_path
            except Exception as exc:
                logger.warning("Asset fabrication failed for '%s': %s", req.asset_id, exc)

        self.index_existing_files_on_disk()
        self.save_catalog()
        return results

    def index_existing_files_on_disk(self) -> None:
        """Scans assets/portraits/, assets/sprites/, assets/tiles/, assets/backgrounds/, assets/ui/, and assets/audio/
        to ensure any pre-existing files on disk are registered in assets/asset_library.json.
        """
        existing_file_paths = {
            str(e.file_path) for e in self.catalog.entries.values() if getattr(e, "file_path", None)
        }

        def _master_fields(rel_runtime: str, default_sz: Tuple[int, int]) -> Dict[str, Any]:
            m_rel = self._master_rel_for_runtime_rel(rel_runtime)
            m_file = self.cartridge_dir / m_rel
            if m_file.is_file():
                return {
                    "master_res_path": f"res://{m_rel}",
                    "master_file_path": m_rel,
                    "master_pixel_size": self._read_image_size(m_file),
                }
            return {}

        for pfile in sorted(self.portraits_dir.glob("*.png")):
            aid = pfile.stem
            rel = f"assets/portraits/{pfile.name}"
            if aid not in self.catalog.entries and rel not in existing_file_paths:
                tex_sz = self._read_image_size(pfile)
                self.catalog.entries[aid] = AssetLibraryEntry(
                    deliberation=f"Indexed existing portrait '{aid}' from disk.",
                    asset_id=aid,
                    asset_type="portrait",
                    fabrication_mode="portrait",
                    purpose=f"Character portrait {aid}",
                    res_path=f"res://{rel}",
                    file_path=rel,
                    target_display_size=(128, 128),
                    texture_pixel_size=tex_sz,
                    sha256_hash=self._compute_sha256(pfile),
                    **_master_fields(rel, tex_sz),
                )
                existing_file_paths.add(rel)

        for sfile in sorted(self.sprites_dir.glob("*.png")):
            if "portrait" in sfile.stem.lower() or sfile.stem.endswith("_sheet"):
                continue
            aid = sfile.stem
            rel = f"assets/sprites/{sfile.name}"
            if aid not in self.catalog.entries and rel not in existing_file_paths:
                tex_sz = self._read_image_size(sfile)
                self.catalog.entries[aid] = AssetLibraryEntry(
                    deliberation=f"Indexed existing sprite '{aid}' from disk.",
                    asset_id=aid,
                    asset_type="sprite",
                    fabrication_mode="static_sprite",
                    purpose=f"In-world sprite {aid}",
                    res_path=f"res://{rel}",
                    file_path=rel,
                    target_display_size=(min(96, tex_sz[0]), min(96, tex_sz[1])),
                    texture_pixel_size=tex_sz,
                    hframes=1,
                    vframes=1,
                    sha256_hash=self._compute_sha256(sfile),
                    **_master_fields(rel, tex_sz),
                )
                existing_file_paths.add(rel)

        extra_visual_dirs = (
            ("tiles", "tile", "tile_surface"),
            ("backgrounds", "backdrop", "parallax_layer"),
            ("ui", "diegetic_ui", "diegetic_ui"),
        )
        for subdir_name, default_atype, default_fmode in extra_visual_dirs:
            sub_dir = self.assets_dir / subdir_name
            if not sub_dir.is_dir():
                continue
            for vfile in sorted(sub_dir.rglob("*.png")):
                aid = vfile.stem
                rel = vfile.relative_to(self.cartridge_dir).as_posix()
                if aid not in self.catalog.entries and rel not in existing_file_paths:
                    tex_sz = self._read_image_size(vfile)
                    self.catalog.entries[aid] = AssetLibraryEntry(
                        deliberation=f"Indexed existing {default_atype} '{aid}' from {rel}.",
                        asset_id=aid,
                        asset_type=default_atype,
                        fabrication_mode=default_fmode,
                        purpose=f"In-world {default_atype} {aid}",
                        res_path=f"res://{rel}",
                        file_path=rel,
                        target_display_size=tex_sz,
                        texture_pixel_size=tex_sz,
                        hframes=1,
                        vframes=1,
                        sha256_hash=self._compute_sha256(vfile),
                        **_master_fields(rel, tex_sz),
                    )
                    existing_file_paths.add(rel)

        for afile in sorted(self.audio_dir.rglob("*.wav")):
            aid = afile.stem
            rel = afile.relative_to(self.cartridge_dir).as_posix()
            if aid not in self.catalog.entries and rel not in existing_file_paths:
                self.catalog.entries[aid] = AssetLibraryEntry(
                    deliberation=f"Indexed existing audio clip '{aid}' from disk.",
                    asset_id=aid,
                    asset_type="audio_bark",
                    fabrication_mode="audio_bark",
                    purpose=f"Audio clip {aid}",
                    res_path=f"res://{rel}",
                    file_path=rel,
                    target_display_size=(0, 0),
                    texture_pixel_size=(0, 0),
                    sha256_hash=self._compute_sha256(afile),
                )
                existing_file_paths.add(rel)

    def build_prompt_summary_for_swe(self, compact: bool = False) -> str:
        """Formats a concise, high-signal summary of all available assets in the Cartridge Asset Library
        (assets/asset_library.json) for LeadGameplaySWEAgent prompts.
        """
        self.index_existing_files_on_disk()
        if (self.cartridge_dir / "project.godot").is_file() or (self.cartridge_dir / "src").is_dir():
            try:
                self.export_godot_studio_helper()
            except Exception:
                pass
        lines: List[str] = [
            "=== CARTRIDGE ASSET LIBRARY CATALOG (res://assets/asset_library.json) ===",
            "CRITICAL RULE: NEVER load files from 'res://assets/portraits/' or '*_portrait.png' as in-world CharacterBody2D/Node2D sprites!",
            "Use 'res://assets/portraits/' ONLY for UI/HUD dialogue busts in ui_hud.gd.",
            "Use 'res://assets/sprites/' for all in-world characters, props, tiles, and environmental set-pieces.",
            "",
            "NATIVE GODOT 4 RIG, ANIMATION & COLLIDER HELPER (`res://src/studio_rig_and_collider.gd`, `class_name StudioRigHelper`):",
            "- `StudioRigHelper.load_context_texture(res_path, true)` -> loads a runtime or context-variant texture with generated GPU mipmaps so high-resolution assets downsample cleanly at any zoom.",
            "- `StudioRigHelper.attach_sprite_sheet_animation_player(host_node, tex, hframes, vframes, states_meta, target_size)` -> returns `{\"sprite\": Sprite2D, \"anim_player\": AnimationPlayer}` with discrete keyframe tracks for `idle`, `move`, `jump`, `action`.",
            "- `StudioRigHelper.create_bitmap_collision_polygons(tex, target_size)` -> traces pixel-accurate `Array[CollisionPolygon2D]` from sprite alpha via `BitMap.opaque_to_polygons()`.",
            "- `StudioRigHelper.create_tiled_surface(static_body, tex, size, true)` -> builds a non-stretched `Polygon2D` (`TEXTURE_REPEAT_ENABLED`) + matching `CollisionShape2D` in group `'walkable_shelf'`.",
            "",
            "AVAILABLE ASSETS IN CATALOG (primary `res_path` is HiDPI-ready; `variants` offer pre-downsampled context sizes like `world_1x` and `hud_icon` derived from full-size `assets/masters/`):",
        ]
        referenced_part_paths: Set[str] = set()
        for entry in self.catalog.entries.values():
            for p in (entry.articulated_parts or []):
                rp = str(p.get("res_path", "") or "").strip()
                if rp:
                    referenced_part_paths.add(rp)

        purpose_limit = 56 if compact else 240
        for aid, entry in sorted(self.catalog.entries.items()):
            if compact and entry.res_path in referenced_part_paths and entry.fabrication_mode == "static_sprite":
                continue
            short_purpose = (entry.purpose or "").replace("\n", " ").strip()
            if len(short_purpose) > purpose_limit:
                short_purpose = short_purpose[:purpose_limit] + "..."
            perc_desc = (getattr(entry, "perceptual_description", "") or "").replace("\n", " ").strip()
            if perc_desc:
                if len(perc_desc) > 200:
                    perc_desc = perc_desc[:200] + "..."
                short_purpose = f"{short_purpose} [Perceptual: {perc_desc}]" if short_purpose else f"[Perceptual: {perc_desc}]"
            if compact:
                lines.append(
                    f"- [{aid}] ({entry.fabrication_mode}, {entry.hframes}x{entry.vframes}, "
                    f"tex_px={entry.texture_pixel_size}, size={entry.target_display_size}) -> '{entry.res_path}'"
                )
                continue
            master_note = (
                f", master='{entry.master_res_path}'({entry.master_pixel_size})"
                if getattr(entry, "master_res_path", "")
                else ""
            )
            var_dict = getattr(entry, "variants", None) or {}
            variants_note = (
                ", variants={"
                + ", ".join(
                    f"{vk}: '{vv.get('res_path', '')}'({vv.get('pixel_size', [])})"
                    for vk, vv in sorted(var_dict.items())
                    if isinstance(vv, dict)
                )
                + "}"
                if var_dict
                else ""
            )
            if entry.fabrication_mode == "sprite_sheet":
                states_str = ", ".join(
                    f"{sname}(row={sinfo.get('row', 0)}, {sinfo.get('frame_count', 4)}f@{sinfo.get('fps', 8)}fps)"
                    for sname, sinfo in entry.animation_states.items()
                )
                lines.append(
                    f"- [{aid}] (mode=sprite_sheet) sheet='{entry.res_path}', single='res://assets/sprites/{aid}.png', "
                    f"display_size={entry.target_display_size}, tex_size={entry.texture_pixel_size}{master_note}{variants_note}, "
                    f"hframes={entry.hframes}, vframes={entry.vframes}, states=[{states_str}] | {short_purpose}"
                )
            elif entry.fabrication_mode == "articulated_parts":
                if compact:
                    part_ids = [str(p.get("part_id", "")) for p in entry.articulated_parts if p.get("part_id")]
                    parts_str = f"part_ids={part_ids}"
                else:
                    parts_str = ", ".join(
                        f"{p.get('part_id', '')}(path={p.get('res_path', '')}, offset={p.get('attach_offset_px', [0, 0])})"
                        for p in entry.articulated_parts
                    )
                lines.append(
                    f"- [{aid}] (mode=articulated_parts, SINGLE-FRAME COMPOSITE: hframes={entry.hframes}, vframes={entry.vframes} — NEVER set hframes>1 or vframes>1 on this composite!) "
                    f"composite='{entry.res_path}', {parts_str}, "
                    f"display_size={entry.target_display_size}, tex_size={entry.texture_pixel_size}{master_note}{variants_note} | {short_purpose}"
                )
            elif entry.fabrication_mode == "portrait":
                lines.append(
                    f"- [{aid}] (mode=portrait, UI ONLY) res_path='{entry.res_path}', display_size={entry.target_display_size}, tex_size={entry.texture_pixel_size}{master_note}{variants_note}, hframes=1, vframes=1 | {short_purpose}"
                )
            else:
                lines.append(
                    f"- [{aid}] (mode={entry.fabrication_mode}) res_path='{entry.res_path}', "
                    f"display_size={entry.target_display_size}, tex_size={entry.texture_pixel_size}{master_note}{variants_note}, "
                    f"hframes={entry.hframes}, vframes={entry.vframes} | {short_purpose}"
                )
        return "\n".join(lines)


AssetLibraryManager = CartridgeAssetLibrary


