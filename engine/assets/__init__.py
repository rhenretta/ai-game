"""engine/assets
Style Bible and Procedural 16-Color Asset Liaison package.
"""
from engine.assets.palette import (
    STYLE_BIBLE_16_PALETTE,
    PALETTE_COLOR_NAMES,
    StyleBibleManifest,
    quantize_rgb,
    get_palette_index,
    quantize_image_to_indexed,
)
from engine.assets.synthesizer import (
    DeterministicSpriteSynthesizer,
    generate_all_sprites,
)
from engine.assets.liaise import (
    AssetLiaison,
    AssetMetadata,
    AssetBundleManifest,
)

__all__ = [
    "STYLE_BIBLE_16_PALETTE",
    "PALETTE_COLOR_NAMES",
    "StyleBibleManifest",
    "quantize_rgb",
    "get_palette_index",
    "quantize_image_to_indexed",
    "DeterministicSpriteSynthesizer",
    "generate_all_sprites",
    "AssetLiaison",
    "AssetMetadata",
    "AssetBundleManifest",
]
