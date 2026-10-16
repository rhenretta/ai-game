"""engine/steam/vdf.py
Pure-Python Valve Data Format (VDF / ACF KeyValues 1) parser.
Zero external C-extensions or external pip dependencies.
Handles:
  - Single-line comments (// ...) outside of quoted strings
  - Preservation of URLs and slashes inside quoted strings
  - Quoted tokens with escape sequences (\\", \\\\, \\n, \\t)
  - Unquoted alphanumeric keys/values
  - Nested dictionary blocks delimited by { and }
  - Multiple top-level blocks
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Dict, List, Union


class VDFParseError(Exception):
    """Raised when VDF/ACF text contains syntax or tokenization errors."""
    pass


def tokenize_vdf(text: str) -> List[str]:
    """Tokenizes raw VDF text into a sequence of string tokens and braces.
    
    Preserves comments inside quoted strings while discarding comments outside quotes.
    Handles escaped backslashes and quotes reliably.
    """
    master_pattern = re.compile(r"(//[^\r\n]*)|(\"((?:[^\"\\]|\\.)*)\")|([{}])|([^\s\"{}]+)")
    tokens: List[str] = []
    
    for match in master_pattern.finditer(text):
        comment, quoted_full, quoted_content, brace, unquoted = match.groups()
        if comment is not None:
            # Skip comments outside quotes
            continue
        elif quoted_content is not None:
            # Unescape \" and \\
            unescaped = (
                quoted_content.replace('\\"', '"')
                .replace('\\\\', '\\')
                .replace('\\n', '\n')
                .replace('\\t', '\t')
            )
            tokens.append(unescaped)
        elif brace is not None:
            tokens.append(brace)
        elif unquoted is not None:
            tokens.append(unquoted)
            
    return tokens


def parse_vdf_tokens(tokens: List[str]) -> Dict[str, Any]:
    """Parses a flat token stream into a nested dictionary structure."""
    idx = 0
    n = len(tokens)

    def parse_dict() -> Dict[str, Any]:
        nonlocal idx
        res: Dict[str, Any] = {}
        while idx < n:
            tok = tokens[idx]
            if tok == "}":
                idx += 1
                return res
            key = tok
            idx += 1
            if idx >= n:
                break
            next_tok = tokens[idx]
            if next_tok == "{":
                idx += 1
                sub_dict = parse_dict()
                res[key] = sub_dict
            elif next_tok == "}":
                idx += 1
                return res
            else:
                res[key] = next_tok
                idx += 1
        return res

    result: Dict[str, Any] = {}
    while idx < n:
        tok = tokens[idx]
        if tok == "}":
            idx += 1
            continue
        key = tok
        idx += 1
        if idx >= n:
            break
        next_tok = tokens[idx]
        if next_tok == "{":
            idx += 1
            sub_dict = parse_dict()
            result[key] = sub_dict
        elif next_tok == "}":
            idx += 1
            continue
        else:
            result[key] = tokens[idx]
            idx += 1

    return result


def parse_vdf(text: str) -> Dict[str, Any]:
    """Parses a VDF/ACF string into a nested Python dictionary."""
    if not text or not text.strip():
        return {}
    tokens = tokenize_vdf(text)
    return parse_vdf_tokens(tokens)


def load_vdf(file_path: Union[str, Path], encoding: str = "utf-8", errors: str = "replace") -> Dict[str, Any]:
    """Loads and parses a VDF/ACF file from disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"VDF file not found: {path}")
    text = path.read_text(encoding=encoding, errors=errors)
    return parse_vdf(text)
