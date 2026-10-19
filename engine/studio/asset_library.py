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
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

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
    ):
        self.cartridge_dir = Path(cartridge_dir)
        self.assets_dir = self.cartridge_dir / "assets"
        self.sprites_dir = self.assets_dir / "sprites"
        self.portraits_dir = self.assets_dir / "portraits"
        self.mockups_dir = self.assets_dir / "mockups"
        self.audio_dir = self.assets_dir / "audio"
        self.catalog_path = self.assets_dir / "asset_library.json"
        self.client = client
        self.visual_agent = visual_agent
        self.voice_agent = voice_agent
        self.mood = mood

        self.sprites_dir.mkdir(parents=True, exist_ok=True)
        self.portraits_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.catalog = self._load_or_init_catalog()

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
        return self.catalog_path

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

        if self.mockups_dir.is_dir():
            for mp in sorted(self.mockups_dir.glob("*.png")):
                if mp not in refs:
                    refs.append(mp)
        return refs[:2]

    def get_style_guide(self, for_environment: bool = False) -> str:
        bible = self._load_game_bible_dict()
        parts: List[str] = []
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

    def sanitize_misplaced_portraits_in_sprites_dir(self) -> List[Path]:
        """Moves or copies any *_portrait.png or portrait_*.png files from assets/sprites/ into
        assets/portraits/ with clean alpha background extraction, preventing portraits from
        masquerading as in-world entity sprites.
        """
        from engine.studio.visual import VisualArtSpriteDesignerAgent

        cleaned_paths: List[Path] = []
        if not self.sprites_dir.is_dir():
            return cleaned_paths

        for png_file in list(self.sprites_dir.glob("*.png")):
            stem_lower = png_file.stem.lower()
            if "portrait" in stem_lower:
                dest_portrait = self.portraits_dir / png_file.name
                try:
                    with Image.open(png_file) as raw_img:
                        cleaned = VisualArtSpriteDesignerAgent.extract_clean_alpha_sprite(
                            raw_img,
                            target_size=(256, 256),
                            strip_border_frame=True,
                            pad_ratio=0.04,
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
                        target_display_size=(128, 128),
                        texture_pixel_size=(256, 256),
                        sha256_hash=self._compute_sha256(dest_portrait),
                    )
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

        # 1. Character portraits (in assets/portraits/) AND full-body animated sprite sheets (in assets/sprites/)
        for idx, char in enumerate(characters):
            if not isinstance(char, dict):
                char = char.model_dump() if hasattr(char, "model_dump") else {}
            cid = str(char.get("character_id") or f"char_{idx+1}").lower().strip()
            cname = str(char.get("name") or cid)
            crole = str(char.get("role") or "character")
            cdesc = str(char.get("visual_description") or f"{cname} ({crole})")

            # Portrait requirement
            derived.append(
                AssetRequirement(
                    deliberation=f"Dialogue bust portrait for {cname} ({crole}) isolated in assets/portraits/.",
                    asset_id=f"{cid}_portrait",
                    name=f"{cid}_portrait.png",
                    asset_type="portrait",
                    fabrication_mode="portrait",
                    purpose=f"UI dialogue bust portrait for {cname}",
                    concept_reference_id=cid,
                    target_path=f"assets/portraits/{cid}_portrait.png",
                    description=f"Dialogue bust portrait of {cname}: {cdesc}",
                    visual_prompt_or_text=f"Expressive bust portrait of {cname} ({crole}): {cdesc}. {style_guide}",
                    dimensions=(256, 256),
                )
            )

            # Full-body in-world animated sprite sheet requirement
            states = [
                AnimationStateSpec(
                    deliberation=f"Idle breathing animation for {cname}.",
                    state_name="idle",
                    frame_count=4,
                    fps=6.0,
                    loop=True,
                    pose_progression_prompt=f"Full-body {cname} standing alert with rhythmic breathing",
                ),
                AnimationStateSpec(
                    deliberation=f"Locomotion stride animation for {cname}.",
                    state_name="move",
                    frame_count=4,
                    fps=10.0,
                    loop=True,
                    pose_progression_prompt=f"Full-body {cname} running/moving with dynamic limb stride",
                ),
                AnimationStateSpec(
                    deliberation=f"Jump / aerial traversal animation for {cname}.",
                    state_name="jump",
                    frame_count=3,
                    fps=8.0,
                    loop=False,
                    pose_progression_prompt=f"Full-body {cname} leaping mid-air with dynamic silhouette",
                ),
                AnimationStateSpec(
                    deliberation=f"Primary interaction / ability animation for {cname}.",
                    state_name="action",
                    frame_count=4,
                    fps=10.0,
                    loop=False,
                    pose_progression_prompt=f"Full-body {cname} activating signature ability or tool",
                ),
            ]
            derived.append(
                AssetRequirement(
                    deliberation=f"Full-body in-world animated sprite sheet for {cname} ({crole}).",
                    asset_id=cid,
                    name=f"{cid}.png",
                    asset_type="sprite_sheet",
                    fabrication_mode="sprite_sheet",
                    purpose=f"Playable/NPC in-world animated actor sprite for {cname}",
                    concept_reference_id=cid,
                    target_path=f"assets/sprites/{cid}.png",
                    description=f"Full-body in-world 2D game character sprite of {cname} ({crole}): {cdesc}",
                    visual_prompt_or_text=(
                        f"Full-body 2D in-game character sprite (head to toe, NOT a bust portrait) of {cname} ({crole}): "
                        f"{cdesc}. {style_guide}"
                    ),
                    dimensions=(72, 72) if idx == 0 else (64, 64),
                    animation_states=states,
                )
            )

            if idx == 0 and cid != "hero":
                derived.append(
                    AssetRequirement(
                        deliberation=f"Canonical 'hero' alias full-body animated sprite sheet for protagonist {cname}.",
                        asset_id="hero",
                        name="hero.png",
                        asset_type="sprite_sheet",
                        fabrication_mode="sprite_sheet",
                        purpose=f"Primary player CharacterBody2D sprite for protagonist {cname}",
                        concept_reference_id=cid,
                        target_path="assets/sprites/hero.png",
                        description=f"Full-body in-world 2D protagonist sprite of {cname}: {cdesc}",
                        visual_prompt_or_text=(
                            f"Full-body 2D in-game protagonist sprite (head to toe, NOT a bust portrait) of {cname}: "
                            f"{cdesc}. {style_guide}"
                        ),
                        dimensions=(72, 72),
                        animation_states=states,
                    )
                )

        # 2 & 3. Decompose the Writers' Room 16:9 concept art & world layout into modular,
        # isolated, composable world-building elements (tiles, horizon set-pieces, midground
        # structures, and interactive landmarks/mechanisms)—never a static full-screen background image.
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
        via a structured LLM JSON contract (AssetManifestSpec, Rule 2 & Rule 4) grounded in the concept art and Game Bible.
        """
        from engine.studio.models import AssetManifestSpec

        if bible_dict is None:
            bible_dict = self._load_game_bible_dict()
        if world_spec is None:
            world_spec = bible_dict.get("world_layout_and_level_design") or {}
        if landmarks is None:
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
                        f"Single isolated 2D game interactive world landmark mechanism/prop '{lname}' ({ltype}): {ldesc}. "
                        f"STRICTLY an inanimate environment/architectural structure — NO characters, NO creatures. {style_guide}"
                    ),
                    dimensions=(96, 96),
                )
            )

        # 2. Delegate visual concept decomposition to Gemini via structured AssetManifestSpec contract (Rule 2 & Rule 4)
        client = getattr(self.visual_agent, "client", None) if self.visual_agent is not None else None
        is_live = getattr(self.visual_agent, "_is_live_genai_client", lambda: False)() if self.visual_agent is not None else False
        if client is not None and is_live:
            try:
                from google.genai import types
                ref_imgs = self.get_concept_reference_images()
                contents: List[Any] = []
                for rp in ref_imgs:
                    if rp.is_file():
                        contents.append(types.Part.from_bytes(data=rp.read_bytes(), mime_type="image/png"))
                prompt = (
                    f"You are the Visual Art & Technical Environment Director decomposing the attached 16:9 Writers' Room concept art "
                    f"and Game Bible into discrete, composable 2D game assets (NEVER a static full-screen background painting).\n"
                    f"Game Title: {bible_dict.get('game_title', '')}\n"
                    f"Camera & Spatial Paradigm: {bible_dict.get('camera_and_spatial_paradigm', '')}\n"
                    f"Visual Style Guide: {style_guide}\n"
                    f"Concept Scene Description: {mockup_desc}\n"
                    f"Scene Layers in Bible: {json.dumps(bible_dict.get('scene_layers') or [])}\n\n"
                    f"Decompose the environment into 4 to 6 modular, reusable visual assets in `assets`:\n"
                    f"- Repeatable surface/material tiles (`asset_type='tile'`, `fabrication_mode='tile_surface'`) for walkable terrain and architectural structures.\n"
                    f"- Isolated alpha-transparent horizon and midground environmental set-piece sprites (`asset_type='prop'`, `fabrication_mode='static_sprite'`) that can be placed across depth layers.\n"
                    f"- Any articulated environmental mechanisms (`asset_type='articulated_rig'`, `fabrication_mode='articulated_parts'`) or animated props (`asset_type='sprite_sheet'`, `fabrication_mode='sprite_sheet'`).\n"
                    f"MANDATORY (Rule 4): Provide `deliberation` FIRST in every schema object."
                )
                contents.append(prompt)
                text_model = getattr(self.visual_agent, "text_model_name", "gemini-3.8-flash")
                resp = client.models.generate_content(
                    model=text_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=AssetManifestSpec,
                    ),
                )
                if resp.text:
                    manifest_spec = AssetManifestSpec.model_validate_json(resp.text)
                    existing_ids = {e.asset_id for e in elements}
                    for req in manifest_spec.assets:
                        if req.asset_id not in existing_ids:
                            elements.append(req)
                            existing_ids.add(req.asset_id)
            except Exception as exc:
                logger.warning("LLM concept decomposition encountered error: %s", exc)

        # 3. Also map any explicit scene_layers defined in game_bible.json
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
                            deliberation=f"Asset '{aid}' referenced by GameBible scene_layer '{layer.get('layer_id', '')}'.",
                            asset_id=aid,
                            name=f"{aid}.png",
                            asset_type="tile" if is_tile_layer else "prop",
                            fabrication_mode="tile_surface" if is_tile_layer else "static_sprite",
                            purpose=f"Scene layer element for {layer.get('layer_id', '')}: {ldesc}",
                            target_path=f"assets/sprites/{aid}.png",
                            description=ldesc,
                            visual_prompt_or_text=f"{ldesc}. {style_guide}",
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
                if self.voice_agent is not None:
                    try:
                        if fmode == "audio_bark" and hasattr(self.voice_agent, "synthesize_bark"):
                            self.voice_agent.synthesize_bark(
                                text=req.visual_prompt_or_text or req.description,
                                output_path=dest_path,
                            )
                        elif hasattr(self.voice_agent, "generate_sfx"):
                            self.voice_agent.generate_sfx(
                                prompt=req.visual_prompt_or_text or req.description,
                                output_path=dest_path,
                            )
                    except Exception as exc:
                        logger.warning("Audio fabrication warning for '%s': %s", req.asset_id, exc)
            if dest_path.is_file():
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
                    sha256_hash=self._compute_sha256(dest_path),
                )
                return dest_path
            return None

        if self.visual_agent is None:
            from engine.studio.visual import VisualArtSpriteDesignerAgent
            self.visual_agent = VisualArtSpriteDesignerAgent(client=self.client)

        # 1. Portrait fabrication (strictly in assets/portraits/)
        if fmode == "portrait" or atype == "portrait":
            portrait_rel = f"assets/portraits/{Path(req.target_path).name}"
            portrait_dest = self.cartridge_dir / portrait_rel
            portrait_dest.parent.mkdir(parents=True, exist_ok=True)
            if not portrait_dest.is_file() or force_regenerate:
                if getattr(self.visual_agent, "client", None) is not None:
                    self.visual_agent.generate_portrait(
                        character={
                            "deliberation": req.deliberation,
                            "character_id": req.asset_id.replace("_portrait", ""),
                            "name": req.asset_id.replace("_portrait", "").replace("_", " ").title(),
                            "role": "character",
                            "visual_description": req.visual_prompt_or_text or req.description,
                        },
                        output_path=portrait_dest,
                        dimensions=req.dimensions or (256, 256),
                        style_guide=style_guide,
                        reference_images=ref_images,
                    )
                else:
                    img = self.visual_agent.renderer.render_dynamic_sprite(
                        dimensions=req.dimensions or (256, 256),
                        prompt=req.visual_prompt_or_text or req.description,
                        asset_type="portrait",
                        seed=req.asset_id,
                    )
                    img.save(portrait_dest, "PNG")

            tex_size = self._read_image_size(portrait_dest)
            self.catalog.entries[req.asset_id] = AssetLibraryEntry(
                deliberation=req.deliberation,
                asset_id=req.asset_id,
                asset_type="portrait",
                fabrication_mode="portrait",
                purpose=req.purpose or req.description,
                res_path=f"res://{portrait_rel}",
                file_path=portrait_rel,
                target_display_size=(128, 128),
                texture_pixel_size=tex_size,
                concept_reference_used=str(ref_images[0]) if ref_images else None,
                sha256_hash=self._compute_sha256(portrait_dest),
            )
            return portrait_dest

        # 2. Multi-Pose / Multi-State Sprite Sheet fabrication
        if fmode == "sprite_sheet" or atype == "sprite_sheet":
            stem = dest_path.stem
            sheet_filename = f"{stem}_sheet.png" if not stem.endswith("_sheet") else f"{stem}.png"
            sheet_rel = f"assets/sprites/{sheet_filename}"
            sheet_dest = self.cartridge_dir / sheet_rel
            single_rel = f"assets/sprites/{stem.replace('_sheet', '')}.png"
            single_dest = self.cartridge_dir / single_rel

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

            self.catalog.entries[req.asset_id] = AssetLibraryEntry(
                deliberation=req.deliberation,
                asset_id=req.asset_id,
                asset_type="sprite_sheet",
                fabrication_mode="sprite_sheet",
                purpose=req.purpose or req.description,
                res_path=f"res://{sheet_rel}",
                file_path=sheet_rel,
                target_display_size=tuple(meta.get("target_display_size") or req.dimensions or (64, 64)),
                texture_pixel_size=tuple(meta.get("texture_pixel_size") or self._read_image_size(sheet_path)),
                hframes=int(meta.get("hframes", 4)),
                vframes=int(meta.get("vframes", 4)),
                animation_states=dict(meta.get("animation_states") or {}),
                concept_reference_used=str(ref_images[0]) if ref_images else None,
                sha256_hash=self._compute_sha256(sheet_path),
            )
            return single_dest

        # 3. Articulated Multi-Part Rig fabrication
        if fmode == "articulated_parts" or atype == "articulated_rig":
            comp_path, meta = self.visual_agent.fabricate_articulated_rig(
                asset_req=req,
                output_path=dest_path,
                mood=self.mood,
                reference_images=ref_images,
                style_guide=style_guide,
            )
            self.catalog.entries[req.asset_id] = AssetLibraryEntry(
                deliberation=req.deliberation,
                asset_id=req.asset_id,
                asset_type="articulated_rig",
                fabrication_mode="articulated_parts",
                purpose=req.purpose or req.description,
                res_path=f"res://{req.target_path}",
                file_path=req.target_path,
                target_display_size=tuple(meta.get("target_display_size") or req.dimensions or (64, 64)),
                texture_pixel_size=tuple(meta.get("texture_pixel_size") or self._read_image_size(comp_path)),
                articulated_parts=list(meta.get("articulated_parts") or []),
                concept_reference_used=str(ref_images[0]) if ref_images else None,
                sha256_hash=self._compute_sha256(comp_path),
            )
            return comp_path

        # 4. Static Sprite / Prop / Tile Surface / Parallax Layer / Diegetic UI
        out_path = self.visual_agent.generate_dynamic_sprite(
            asset_req=req,
            output_path=dest_path,
            mood=self.mood,
            reference_images=ref_images,
            style_guide=style_guide,
        )
        tex_size = self._read_image_size(out_path)
        self.catalog.entries[req.asset_id] = AssetLibraryEntry(
            deliberation=req.deliberation,
            asset_id=req.asset_id,
            asset_type=atype,
            fabrication_mode=fmode,
            purpose=req.purpose or req.description,
            res_path=f"res://{req.target_path}",
            file_path=req.target_path,
            target_display_size=tuple(req.dimensions or tex_size),
            texture_pixel_size=tex_size,
            concept_reference_used=str(ref_images[0]) if ref_images else None,
            sha256_hash=self._compute_sha256(out_path),
        )
        return out_path

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
            if not dest.is_file():
                try:
                    self.visual_agent.generate_concept_mockup(
                        mockup=m,
                        output_path=dest,
                        style_guide=style_guide,
                    )
                    if dest.is_file():
                        synthesized.append(dest)
                except Exception as exc:
                    logger.warning("Concept mockup synthesis warning for '%s': %s", rel, exc)
        return synthesized

    @staticmethod
    def _tile_has_defective_corners(path: Path) -> bool:
        """Returns True if a tile texture on disk has white/magenta background corners from an isometric cube render."""
        if not path.is_file():
            return True
        try:
            with Image.open(path) as im:
                rgba = im.convert("RGBA")
                w, h = rgba.size
                px = rgba.load()
                corners = [px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1]]
                for r, g, b, a in corners:
                    if a < 240 or (r > 210 and g > 210 and b > 210) or (r > 200 and b > 200 and g < 80):
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
        """Scans assets/portraits/, assets/sprites/, and assets/audio/ to ensure any pre-existing files
        are registered in assets/asset_library.json.
        """
        for pfile in sorted(self.portraits_dir.glob("*.png")):
            aid = pfile.stem
            if aid not in self.catalog.entries:
                rel = f"assets/portraits/{pfile.name}"
                self.catalog.entries[aid] = AssetLibraryEntry(
                    deliberation=f"Indexed existing portrait '{aid}' from disk.",
                    asset_id=aid,
                    asset_type="portrait",
                    fabrication_mode="portrait",
                    purpose=f"Character portrait {aid}",
                    res_path=f"res://{rel}",
                    file_path=rel,
                    target_display_size=(128, 128),
                    texture_pixel_size=self._read_image_size(pfile),
                    sha256_hash=self._compute_sha256(pfile),
                )

        for sfile in sorted(self.sprites_dir.glob("*.png")):
            if "portrait" in sfile.stem.lower() or sfile.stem.endswith("_sheet"):
                continue
            aid = sfile.stem
            if aid not in self.catalog.entries:
                rel = f"assets/sprites/{sfile.name}"
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
                    sha256_hash=self._compute_sha256(sfile),
                )

        for afile in sorted(self.audio_dir.glob("*.wav")):
            aid = afile.stem
            if aid not in self.catalog.entries:
                rel = f"assets/audio/{afile.name}"
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

    def build_prompt_summary_for_swe(self) -> str:
        """Formats a concise, high-signal summary of all available assets in the Cartridge Asset Library
        (assets/asset_library.json) for LeadGameplaySWEAgent prompts.
        """
        self.index_existing_files_on_disk()
        lines: List[str] = [
            "=== CARTRIDGE ASSET LIBRARY CATALOG (res://assets/asset_library.json) ===",
            "CRITICAL RULE: NEVER load files from 'res://assets/portraits/' or '*_portrait.png' as in-world CharacterBody2D/Node2D sprites!",
            "Use 'res://assets/portraits/' ONLY for UI/HUD dialogue busts in ui_hud.gd.",
            "Use 'res://assets/sprites/' for all in-world characters, props, tiles, and environmental set-pieces.",
            "",
            "AVAILABLE ASSETS IN CATALOG:",
        ]
        for aid, entry in sorted(self.catalog.entries.items()):
            if entry.fabrication_mode == "sprite_sheet":
                states_str = ", ".join(
                    f"{sname}(row={sinfo.get('row', 0)}, {sinfo.get('frame_count', 4)}f@{sinfo.get('fps', 8)}fps)"
                    for sname, sinfo in entry.animation_states.items()
                )
                lines.append(
                    f"- [{aid}] (mode=sprite_sheet) sheet='{entry.res_path}', single='res://assets/sprites/{aid}.png', "
                    f"display_size={entry.target_display_size}, tex_size={entry.texture_pixel_size}, "
                    f"hframes={entry.hframes}, vframes={entry.vframes}, states=[{states_str}] | {entry.purpose}"
                )
            elif entry.fabrication_mode == "articulated_parts":
                parts_str = ", ".join(
                    f"{p.get('part_id', '')}(path={p.get('res_path', '')}, offset={p.get('attach_offset_px', [0, 0])})"
                    for p in entry.articulated_parts
                )
                lines.append(
                    f"- [{aid}] (mode=articulated_parts) composite='{entry.res_path}', parts=[{parts_str}], "
                    f"display_size={entry.target_display_size} | {entry.purpose}"
                )
            elif entry.fabrication_mode == "portrait":
                lines.append(
                    f"- [{aid}] (mode=portrait, UI ONLY) res_path='{entry.res_path}', display_size={entry.target_display_size}, tex_size={entry.texture_pixel_size} | {entry.purpose}"
                )
            else:
                lines.append(
                    f"- [{aid}] (mode={entry.fabrication_mode}) res_path='{entry.res_path}', "
                    f"display_size={entry.target_display_size}, tex_size={entry.texture_pixel_size} | {entry.purpose}"
                )
        return "\n".join(lines)
