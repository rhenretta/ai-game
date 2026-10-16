"""engine/assets/liaise.py
Asset Liaison & Style Bible Pipeline Orchestrator.
Synthesizes, validates, and packages 16-color procedural asset bundles into game cartridges.
Enforces Rationale-First Epistemic Architecture (deliberation declared first).
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from pydantic import BaseModel, Field
from PIL import Image

from engine.assets.palette import STYLE_BIBLE_16_PALETTE
from engine.assets.synthesizer import generate_all_sprites
from engine.scheduler.artifact import ArtifactMetadataEnvelope, ArtifactType

logger = logging.getLogger("AssetLiaison")


class AssetMetadata(BaseModel):
    """Metadata for an individual generated sprite or tile asset."""
    deliberation: str = Field(
        default="",
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining the visual parameters of this asset."
    )
    name: str = Field(..., description="Asset identifier, e.g. 'player_idle'")
    category: str = Field(..., description="'sprite' or 'tile'")
    relative_path: str = Field(..., description="Path relative to cartridge root")
    width: int = Field(..., description="Width in pixels")
    height: int = Field(..., description="Height in pixels")
    sha256_hash: str = Field(..., description="SHA-256 hexadecimal digest of the PNG file")
    file_size_bytes: int = Field(..., description="File size in bytes")
    palette_indices_used: List[int] = Field(default_factory=list, description="Palette indices present in the image")
    is_indexed_png: bool = Field(default=True, description="True if saved in Pillow 'P' indexed mode")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "name": self.name,
            "category": self.category,
            "relative_path": self.relative_path,
            "width": self.width,
            "height": self.height,
            "sha256_hash": self.sha256_hash,
            "file_size_bytes": self.file_size_bytes,
            "palette_indices_used": self.palette_indices_used,
            "is_indexed_png": self.is_indexed_png,
        }


class AssetBundleManifest(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared first!
    Complete manifest for all procedural assets synthesized for a cartridge.
    """
    deliberation: str = Field(
        default="",
        description="MANDATORY FIRST PROPERTY: Epistemic rationale explaining asset synthesis strategy, thematic palette rules, and generated inventory."
    )
    bundle_id: str = Field(default_factory=lambda: f"asset_bundle_{uuid.uuid4().hex[:12]}")
    cartridge_name: str = Field(..., description="Target cartridge name")
    palette_mood: str = Field(default="retro_arcade")
    seed: str = Field(default="seed_42")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sprites: Dict[str, AssetMetadata] = Field(default_factory=dict)
    tiles: Dict[str, AssetMetadata] = Field(default_factory=dict)
    total_files: int = Field(default=0)
    total_bytes: int = Field(default=0)
    is_fully_style_bible_compliant: bool = Field(default=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "bundle_id": self.bundle_id,
            "cartridge_name": self.cartridge_name,
            "palette_mood": self.palette_mood,
            "seed": self.seed,
            "created_at_utc": self.created_at_utc,
            "sprites": {k: v.to_dict() for k, v in self.sprites.items()},
            "tiles": {k: v.to_dict() for k, v in self.tiles.items()},
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "is_fully_style_bible_compliant": self.is_fully_style_bible_compliant,
        }


class AssetLiaison:
    """Orchestrator for the Style Bible and Procedural 16-Color Asset Synthesis Pipeline."""
    deliberation: str = "Orchestrator for the Style Bible and Procedural 16-Color Asset Synthesis Pipeline."

    @staticmethod
    def synthesize_cartridge_assets(
        cartridge_dir: Union[str, Path],
        spec: Optional[Any] = None,
        seed: Optional[Union[int, str]] = None,
        mood: Optional[str] = None
    ) -> AssetBundleManifest:
        """Synthesizes all authentic 16-color procedural assets for the given cartridge directory."""
        cartridge_path = Path(cartridge_dir).resolve()
        assets_dir = cartridge_path / "assets"
        sprites_dir = assets_dir / "sprites"
        tiles_dir = assets_dir / "tiles"

        sprites_dir.mkdir(parents=True, exist_ok=True)
        tiles_dir.mkdir(parents=True, exist_ok=True)

        # Resolve mood and seed from spec if provided
        chosen_mood = mood or "retro_arcade"
        chosen_seed = str(seed or "seed_genesis")

        if spec is not None:
            if hasattr(spec, "palette_mood") and getattr(spec, "palette_mood"):
                chosen_mood = str(getattr(spec, "palette_mood"))
            elif isinstance(spec, dict) and "palette_mood" in spec and spec["palette_mood"]:
                chosen_mood = str(spec["palette_mood"])
            elif hasattr(spec, "thematic_style") and getattr(spec, "thematic_style"):
                chosen_mood = str(getattr(spec, "thematic_style"))
            elif isinstance(spec, dict) and "thematic_style" in spec and spec["thematic_style"]:
                chosen_mood = str(spec["thematic_style"])

            if hasattr(spec, "spec_id") and getattr(spec, "spec_id"):
                chosen_seed = str(getattr(spec, "spec_id"))
            elif isinstance(spec, dict) and "spec_id" in spec and spec["spec_id"]:
                chosen_seed = str(spec["spec_id"])

        deliberation_text = (
            f"Synthesizing 16-color Style Bible asset bundle for cartridge '{cartridge_path.name}'. "
            f"Thematic mood selected as '{chosen_mood}' seeded by '{chosen_seed}'. "
            f"Procedural generators enforce strict 16-color retro quantization via Euclidean distance, "
            f"pixel-perfect bounding dimensions (32x32 player/chaser/ranged/tiles, 48x48 boss/spikes/emitter, 96x96 acid), "
            f"and zero external network dependencies."
        )

        sprites_dict, tiles_dict = generate_all_sprites(seed=chosen_seed, mood=chosen_mood)

        manifest = AssetBundleManifest(
            deliberation=deliberation_text,
            cartridge_name=cartridge_path.name,
            palette_mood=chosen_mood,
            seed=chosen_seed,
        )

        total_bytes = 0
        total_files = 0

        # Save and catalog sprites
        for name, img in sprites_dict.items():
            file_name = f"{name}.png"
            file_path = sprites_dir / file_name
            # Save as authentic indexed PNG with transparency key 0
            img.save(file_path, format="PNG", transparency=0)

            data = file_path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            size = len(data)
            total_bytes += size
            total_files += 1

            colors_used = [c[1] for c in (img.getcolors() or [])]
            manifest.sprites[name] = AssetMetadata(
                deliberation=f"Procedurally synthesized sprite for '{name}' under '{chosen_mood}' style bible.",
                name=name,
                category="sprite",
                relative_path=f"assets/sprites/{file_name}",
                width=img.width,
                height=img.height,
                sha256_hash=sha,
                file_size_bytes=size,
                palette_indices_used=sorted(list(set(colors_used))),
                is_indexed_png=(img.mode == "P"),
            )

        # Save and catalog tiles
        for name, img in tiles_dict.items():
            file_name = f"{name}.png"
            file_path = tiles_dir / file_name
            img.save(file_path, format="PNG")

            data = file_path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            size = len(data)
            total_bytes += size
            total_files += 1

            colors_used = [c[1] for c in (img.getcolors() or [])]
            manifest.tiles[name] = AssetMetadata(
                deliberation=f"Procedurally synthesized environment tile for '{name}' with 16-color masonry bevels.",
                name=name,
                category="tile",
                relative_path=f"assets/tiles/{file_name}",
                width=img.width,
                height=img.height,
                sha256_hash=sha,
                file_size_bytes=size,
                palette_indices_used=sorted(list(set(colors_used))),
                is_indexed_png=(img.mode == "P"),
            )

        manifest.total_files = total_files
        manifest.total_bytes = total_bytes

        # Write manifest.json with deliberation as the first key
        manifest_path = assets_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        logger.info(
            "Successfully synthesized %d assets (%d bytes) for cartridge '%s'",
            total_files, total_bytes, cartridge_path.name
        )
        return manifest

    @staticmethod
    def validate_asset_bundle(manifest: AssetBundleManifest, cartridge_dir: Union[str, Path]) -> List[str]:
        """Validates all assets in a bundle against Style Bible rules, returning any error diagnostics."""
        base_path = Path(cartridge_dir).resolve()
        errors: List[str] = []

        all_assets = {**manifest.sprites, **manifest.tiles}
        for name, meta in all_assets.items():
            file_path = base_path / meta.relative_path
            if not file_path.exists():
                errors.append(f"Missing asset file on disk: {meta.relative_path}")
                continue

            data = file_path.read_bytes()
            # PNG Magic Header check
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                errors.append(f"Invalid PNG magic bytes for {meta.relative_path}")

            # SHA-256 integrity check
            current_sha = hashlib.sha256(data).hexdigest()
            if current_sha != meta.sha256_hash:
                errors.append(f"SHA-256 mismatch for {meta.relative_path}: expected {meta.sha256_hash}, got {current_sha}")

            # Pillow image inspection
            try:
                with Image.open(file_path) as img:
                    if img.width != meta.width or img.height != meta.height:
                        errors.append(
                            f"Dimension mismatch for {meta.relative_path}: expected {meta.width}x{meta.height}, got {img.width}x{img.height}"
                        )
                    if img.mode != "P":
                        errors.append(f"Mode mismatch for {meta.relative_path}: expected 'P', got '{img.mode}'")
                    # Check palette indices
                    colors = img.getcolors() or []
                    for count, idx in colors:
                        if idx >= 16:
                            errors.append(f"Palette violation in {meta.relative_path}: color index {idx} >= 16")
            except Exception as exc:
                errors.append(f"Failed to load image {meta.relative_path}: {exc}")

        return errors

    @staticmethod
    def seal_asset_envelope(
        manifest: AssetBundleManifest,
        cartridge_dir: Union[str, Path]
    ) -> ArtifactMetadataEnvelope:
        """Packs and seals the asset bundle manifest into an ArtifactMetadataEnvelope."""
        return ArtifactMetadataEnvelope.seal(
            deliberation=manifest.deliberation,
            artifact_type=ArtifactType.ASSET_BUNDLE,
            source_agent="AssetLiaison",
            target_agent="BuildSupervisor",
            payload=manifest.to_dict(),
            metadata={
                "cartridge_name": manifest.cartridge_name,
                "palette_mood": manifest.palette_mood,
                "total_files": manifest.total_files,
                "total_bytes": manifest.total_bytes,
            }
        )

    @staticmethod
    def load_sprite_surface(asset_path: Union[str, Path]) -> Optional[Any]:
        """Safely loads a PNG asset into a Pygame Surface if Pygame is available and initialized."""
        try:
            import pygame
            path = Path(asset_path)
            if path.exists():
                surf = pygame.image.load(str(path))
                try:
                    return surf.convert_alpha()
                except Exception:
                    return surf
        except Exception as exc:
            logger.debug("Pygame load_sprite_surface skipped or failed: %s", exc)
        return None
