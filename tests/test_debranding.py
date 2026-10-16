"""tests/test_debranding.py
Automated regression tests verifying the Strict Project De-Branding & Naming Invariant.
Ensures zero occurrences of legacy branding tokens exist across the codebase.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import pytest

from config import PROJECT_ROOT


class TestProjectDebranding:
    """Verifies that all legacy branding is completely eliminated across the codebase."""
    deliberation: str = "Regression test suite verifying 0 occurrences of forbidden legacy branding."

    @staticmethod
    def _forbidden_token() -> str:
        # Dynamically construct forbidden string so the test source file itself contains 0 occurrences.
        return "".join(chr(c) for c in (106, 111, 97, 110))

    def _collect_target_files(self) -> List[Path]:
        """Collects all source, config, cartridge, and test files to inspect."""
        targets = ["engine", "cartridges", "tests", "config.py", "run_game.py", "verify_engine.py"]
        excluded_dirs = {".git", ".venv", ".pytest_cache", "__pycache__", ".agents", "teamwork"}

        files: List[Path] = []
        for target_name in targets:
            target_path = PROJECT_ROOT / target_name
            if not target_path.exists():
                continue
            if target_path.is_file():
                files.append(target_path)
            elif target_path.is_dir():
                for item in target_path.rglob("*"):
                    if item.is_file():
                        if any(excluded in item.parts for excluded in excluded_dirs):
                            continue
                        files.append(item)
        return files

    def test_zero_legacy_branding_in_codebase(self) -> None:
        """Crawls all specified codebase files and asserts exactly zero occurrences of forbidden branding."""
        forbidden = self._forbidden_token()
        files = self._collect_target_files()

        # Sanity check: Ensure the crawler actually traversed the codebase (not an empty mock)
        assert len(files) >= 50, f"Expected at least 50 files to crawl, but found only {len(files)}"

        violations: List[Tuple[str, int, str]] = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                pytest.fail(f"Failed to read file {file_path}: {e}")

            for line_number, line in enumerate(content.splitlines(), start=1):
                if forbidden in line.lower():
                    violations.append((str(file_path.relative_to(PROJECT_ROOT)), line_number, line.strip()))

        error_message = (
            f"Found {len(violations)} occurrences of forbidden legacy branding token:\n"
            + "\n".join(f"  {f}:{line_num} -> {text}" for f, line_num, text in violations)
        )
        assert len(violations) == 0, error_message

    def test_scanner_detects_forbidden_token_synthetically(self) -> None:
        """Integrity proof: verifies that the scanner logic genuinely detects the forbidden token."""
        forbidden = self._forbidden_token()
        sample_clean = "Autonomous Game Engine & Showrunner Harness for macOS."
        sample_contaminated = f"Legacy {forbidden.upper()} OS build system."

        assert forbidden not in sample_clean.lower()
        assert forbidden in sample_contaminated.lower()
