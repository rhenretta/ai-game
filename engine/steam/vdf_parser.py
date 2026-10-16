"""engine/steam/vdf_parser.py
Alias and re-export module for engine.steam.vdf.
Zero external C-extensions or external pip dependencies.
"""
from __future__ import annotations

from engine.steam.vdf import (
    VDFParseError,
    load_vdf,
    parse_vdf,
    parse_vdf_tokens,
    tokenize_vdf,
)

__all__ = [
    "VDFParseError",
    "load_vdf",
    "parse_vdf",
    "parse_vdf_tokens",
    "tokenize_vdf",
]
