"""tests/e2e/runner.py
Authoritative E2E Test Suite Runner for Autonomous Game Engine & Studio Harness.
Supports granular execution filtering via:
  --tier [1|2|3|4]
  --milestone [M1|M2|M3|M4|M5|M6]
  --feature [F01..F33]
  --all
Strictly enforces Rationale-First Epistemic Architecture and project debranding invariants.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Milestone -> Feature Inventory Mapping (Features 1 to 33)
MILESTONE_FEATURE_MAP: Dict[str, List[int]] = {
    "M1": [1, 2, 3, 4, 5, 6],
    "M2": [7, 8, 9, 10],
    "M3": [11, 12, 13, 14, 15],
    "M4": [16, 17, 18, 19, 20],
    "M5": [21, 22, 23, 24, 25],
    "M6": [26, 27, 28, 29, 30, 31, 32, 33],
}

TIER_FILES: Dict[int, str] = {
    1: "tests/e2e/test_tier1_features.py",
    2: "tests/e2e/test_tier2_boundaries.py",
    3: "tests/e2e/test_tier3_combinations.py",
    4: "tests/e2e/test_tier4_scenarios.py",
}


def parse_feature_id(feature_str: str) -> Optional[int]:
    """Normalizes feature input like 'F01', 'f1', '1', 'F33' to integer 1..33."""
    clean = feature_str.upper().lstrip("F").strip()
    try:
        val = int(clean)
        if 1 <= val <= 33:
            return val
    except ValueError:
        pass
    return None


def build_pytest_args(
    tier: Optional[int] = None,
    milestone: Optional[str] = None,
    feature: Optional[str] = None,
    run_all: bool = False,
    verbose: bool = False,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Constructs authoritative pytest arguments based on CLI filtering flags."""
    args = []
    if verbose:
        args.append("-v")

    if extra_args:
        args.extend(extra_args)

    # 1. Feature filter specified
    if feature:
        feat_num = parse_feature_id(feature)
        if feat_num is None:
            raise ValueError(f"Invalid feature specifier: {feature}. Expected F01 to F33.")
        feat_tag = f"test_f{feat_num:02d}_"
        args.extend(["-k", feat_tag])
        if tier and tier in TIER_FILES:
            args.append(TIER_FILES[tier])
        else:
            args.extend([TIER_FILES[1], TIER_FILES[2]])
        return args

    # 2. Milestone filter specified
    if milestone:
        ms_key = milestone.upper()
        if ms_key not in MILESTONE_FEATURE_MAP:
            raise ValueError(f"Invalid milestone specifier: {milestone}. Expected M1 to M6.")
        feat_nums = MILESTONE_FEATURE_MAP[ms_key]
        patterns = [f"test_f{n:02d}_" for n in feat_nums]
        # Also include cross-milestone test patterns (e.g. test_m1_)
        ms_tag = f"test_{ms_key.lower()}_"
        k_expr = " or ".join(patterns + [ms_tag])
        args.extend(["-k", k_expr])

        if tier and tier in TIER_FILES:
            args.append(TIER_FILES[tier])
        else:
            args.append("tests/e2e/")
        return args

    # 3. Tier filter specified
    if tier:
        if tier not in TIER_FILES:
            raise ValueError(f"Invalid tier specifier: {tier}. Expected 1, 2, 3, or 4.")
        args.append(TIER_FILES[tier])
        return args

    # 4. Default or All
    args.append("tests/e2e/")
    return args


def run_e2e_suite(
    tier: Optional[int] = None,
    milestone: Optional[str] = None,
    feature: Optional[str] = None,
    run_all: bool = False,
    verbose: bool = False,
    json_output: bool = False,
    extra_args: Optional[List[str]] = None,
) -> int:
    """Executes the test suite via pytest and returns process exit code."""
    deliberation = "Executing authoritative E2E test runner across specified tiers and milestone boundaries."
    pytest_args = build_pytest_args(
        tier=tier,
        milestone=milestone,
        feature=feature,
        run_all=run_all,
        verbose=verbose,
        extra_args=extra_args,
    )

    header = f"=== Autonomous Game Engine & Studio Harness E2E Runner ==="
    print(header)
    print(f"Target: {'All Tiers' if run_all or not (tier or milestone or feature) else ''}"
          f"{f'Tier {tier}' if tier else ''}"
          f"{f'Milestone {milestone.upper()}' if milestone else ''}"
          f"{f'Feature {feature.upper()}' if feature else ''}")
    print(f"Pytest Arguments: {' '.join(pytest_args)}")
    print("=" * len(header))

    t_start = time.perf_counter()
    exit_code = pytest.main(pytest_args)
    duration = time.perf_counter() - t_start

    print("-" * len(header))
    print(f"E2E Execution Completed in {duration:.2f}s (Exit Code: {exit_code})")
    print(f"Verdict: {'PASSED (100% Quality Gate)' if exit_code == 0 else 'FAILED (Verification Halted)'}")
    print("-" * len(header))

    return int(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Game Engine & Studio Harness E2E Test Suite Runner"
    )
    parser.add_argument(
        "--tier", "-t",
        type=int,
        choices=[1, 2, 3, 4],
        help="Execute specific test tier (1: Features, 2: Boundaries, 3: Combinations, 4: Scenarios)",
    )
    parser.add_argument(
        "--milestone", "-m",
        type=str,
        choices=["M1", "m1", "M2", "m2", "M3", "m3", "M4", "m4", "M5", "m5", "M6", "m6"],
        help="Execute tests associated with milestone (M1 to M6)",
    )
    parser.add_argument(
        "--feature", "-f",
        type=str,
        help="Execute tests for specific feature (e.g. F01, F12, 33)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Execute entire E2E test suite across all 4 tiers",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose pytest output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON summary",
    )

    parsed, remaining = parser.parse_known_args()
    exit_code = run_e2e_suite(
        tier=parsed.tier,
        milestone=parsed.milestone,
        feature=parsed.feature,
        run_all=parsed.all,
        verbose=parsed.verbose,
        json_output=parsed.json,
        extra_args=remaining,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
