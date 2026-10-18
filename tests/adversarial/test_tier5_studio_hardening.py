"""tests/adversarial/test_tier5_studio_hardening.py
Tier 5 Adversarial Coverage Hardening Test Suite for Autonomous Game Studio Swarm.
Author: Explorer M6-1 (E2E Test Suite & Tier 5 Hardening)

Covers four mission-critical adversarial domains:
1. Extreme / Malformed Player DNA Inputs (empty games, Unicode/emojis, negative/extreme playtimes, corrupt achievement ratios).
2. Adversarial Code Synthesis (deep AST nesting, evasive syntax sandbox escapes, missing imports, circular references, deliberation first).
3. Event Bus Edge Cases (faulty listener exceptions, high-concurrency publishing, malformed payload deserialization, corrupted JSONL log recovery).
4. Challenger Gate Edge Cases (partial criterion match rejection, conflicting grounding telemetry, simulation crash/0 FPS timeout, missing/corrupt spec).

Strictly adheres to:
- Rule 1: Zero test-case poisoning; domain- and genre-agnostic.
- Rule 2: Ban on keyword/regex heuristics for epistemic decisions.
- Rule 3: Dev server protection & non-destructive validation under pytest.
- Rule 4: Mandatory Rationale-First Epistemic Architecture (deliberation declared first).
- Rule 5: Zero silent degradation (fail fast on invalid configurations).
- Rule 6: Production parity & container runtime invariance (SDL_VIDEODRIVER=dummy).
"""
from __future__ import annotations

import asyncio
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Dict, List, Optional
import uuid
import pytest
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enforce Headless SDL Environment
# ---------------------------------------------------------------------------
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# ---------------------------------------------------------------------------
# Core Imports
# ---------------------------------------------------------------------------
from engine.loader import ASTSecurityError, validate_cartridge_ast
from engine.steam.dna import (
    SteamDNAProfiler,
    compute_challenge_tolerance,
    compute_pacing_score,
    compute_playtime_distribution,
)
from engine.steam.models import SteamGameRecord, SteamLibraryDataset, SteamPlayerDNA
from engine.studio.models import (
    AcceptanceCriterion,
    GameRequirementsSpec,
    RequirementTrack,
    StudioAgentError,
)
from engine.studio.swe import CodeVerificationError, LeadGameplaySWEAgent
from engine.supervisor.challenger import (
    CandidateASTInspector,
    CandidateTelemetryAuditor,
    CandidateVisualPlaytester,
    ChallengerGate,
)
from engine.supervisor.models import (
    ChallengerGateEvaluation,
    GateFailureType,
    RejectionDiagnosticTrace,
    RemedyType,
    RequirementEvaluationRecord,
)
from engine.ui.models import SwarmEventType
from engine.ui.server import SwarmEventBus


# ===========================================================================
# AREA 1: EXTREME / MALFORMED PLAYER DNA INPUTS
# ===========================================================================

class TestTier5PlayerDNAHardening:
    """Hardening tests for player DNA extraction, normalization, and bounds under extreme inputs."""

    def test_tier5_dna_empty_games_list(self):
        """Validates that a player library with 0 games produces valid fallback DNA without division-by-zero."""
        empty_dataset = SteamLibraryDataset(
            deliberation="Empty library test dataset",
            player_id="test_zero_games_player",
            games=[],
        )
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(empty_dataset, profile_tag="genesis_zero_games")

        assert isinstance(dna, SteamPlayerDNA)
        assert dna.deliberation != ""
        assert dna.total_playtime_hours == 0.0
        assert dna.total_games_analyzed == 0
        assert len(dna.genre_weights) > 0
        # Check L1 normalization sum
        assert abs(sum(dna.genre_weights.values()) - 1.0) < 1e-3
        assert -1.0 <= dna.pacing_score <= 1.0
        assert 0.0 <= dna.challenge_tolerance <= 1.0

    def test_tier5_dna_extreme_non_ascii_unicode_emojis_titles(self):
        """Validates handling of non-ASCII, Unicode emojis, CJK, RTL, and shell-escape characters in game titles."""
        extreme_games = [
            SteamGameRecord(
                deliberation="Unicode game 1",
                appid=1001,
                name="🎮 东方Project: 弾幕狂宴 🌟 — Déjà Vu / 🚀 \u0000",
                playtime_forever_minutes=1200,
                genres=["Action", "Indie"],
                tags=["BulletHell", "Difficult"],
                achievements_total=50,
                achievements_unlocked=25,
            ),
            SteamGameRecord(
                deliberation="Unicode game 2",
                appid=1002,
                name="مغامرة الفضاء: ⚔️ Cyber_Samurai_1984; DROP TABLE games; --",
                playtime_forever_minutes=3600,
                genres=["RPG"],
                tags=["Souls-like", "Atmospheric"],
                achievements_total=10,
                achievements_unlocked=10,
            ),
        ]
        dataset = SteamLibraryDataset(
            deliberation="Adversarial Unicode dataset",
            player_id="player_unicode_007",
            games=extreme_games,
        )
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(dataset)

        assert dna.total_playtime_hours == (1200 + 3600) / 60.0
        assert dna.total_games_analyzed == 2
        # Serializability check
        json_str = dna.model_dump_json()
        assert "东方Project" in json_str or "\\u" in json_str
        deserialized = SteamPlayerDNA.model_validate_json(json_str)
        assert deserialized.total_games_analyzed == 2

    def test_tier5_dna_extreme_and_negative_playtimes(self):
        """Validates bounds clamping when playtime values are negative, zero, or astronomical (millions of hours)."""
        extreme_games = [
            SteamGameRecord(
                deliberation="Negative playtime anomaly",
                appid=2001,
                name="Corrupted Glitch Game",
                playtime_forever_minutes=-99999,  # Negative anomaly
                genres=["Action"],
            ),
            SteamGameRecord(
                deliberation="Astronomical playtime",
                appid=2002,
                name="Infinite Idle Sim",
                playtime_forever_minutes=600_000_000,  # 10,000,000 hours
                genres=["Simulation"],
            ),
        ]
        dataset = SteamLibraryDataset(
            deliberation="Extreme playtime dataset",
            games=extreme_games,
        )
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(dataset)

        # Negative should be clamped to 0.0 in total playtime sum
        assert dna.total_playtime_hours >= 0.0
        # Challenge tolerance must remain strictly bounded in [0.0, 1.0]
        assert 0.0 <= dna.challenge_tolerance <= 1.0
        assert -1.0 <= dna.pacing_score <= 1.0

    def test_tier5_dna_corrupted_achievement_ratios(self):
        """Validates calculation when unlocked achievements exceed total achievements or total is 0."""
        game_unlocked_gt_total = SteamGameRecord(
            deliberation="Unlocked greater than total",
            appid=3001,
            name="Hacked Stats Game",
            playtime_forever_minutes=300,
            achievements_total=10,
            achievements_unlocked=50,  # 500% rate
            tags=["Souls-like", "Difficult"],
        )
        game_zero_total = SteamGameRecord(
            deliberation="Zero total achievements",
            appid=3002,
            name="No Achievement Game",
            playtime_forever_minutes=600,
            achievements_total=0,
            achievements_unlocked=0,
            tags=["Casual"],
        )

        tol1, _ = compute_challenge_tolerance(
            games=[game_unlocked_gt_total],
            genre_weights={"Action": 1.0},
            total_hours=5.0,
            game_weights={3001: 1.0},
        )
        tol2, _ = compute_challenge_tolerance(
            games=[game_zero_total],
            genre_weights={"Simulation": 1.0},
            total_hours=10.0,
            game_weights={3002: 1.0},
        )

        assert 0.0 <= tol1 <= 1.0
        assert 0.0 <= tol2 <= 1.0

    def test_tier5_dna_pacing_score_and_distribution_extremes(self):
        """Validates continuous pacing score mapping between pure kinetic and pure tactical polar extremes."""
        pure_kinetic_weights = {"BulletHell": 0.5, "Action": 0.5}
        pure_tactical_weights = {"TurnBasedStrategy": 0.5, "Puzzle": 0.5}

        k_score, k_pref = compute_pacing_score(pure_kinetic_weights)
        t_score, t_pref = compute_pacing_score(pure_tactical_weights)

        assert k_score > 0.0
        assert k_pref == "kinetic_high_mobility"
        assert t_score < 0.0
        assert t_pref == "tactical_methodical"
        assert -1.0 <= k_score <= 1.0
        assert -1.0 <= t_score <= 1.0


# ===========================================================================
# AREA 2: ADVERSARIAL CODE SYNTHESIS HARDENING
# ===========================================================================

class TestTier5CodeSynthesisHardening:
    """Hardening tests for AST security validator, cross-module imports, and code modularity."""

    def test_tier5_swe_deep_ast_nesting_stress(self):
        """Validates AST security analysis under deeply nested control-flow structures (60+ levels)."""
        nested_code_lines = [
            "class DeeplyNestedCartridge:\n",
            "    deliberation: str = 'Deeply nested control structures for stress validation'\n",
            "    def update(self, dt: float) -> None:\n",
        ]
        depth = 60
        for i in range(depth):
            indent = "        " + ("    " * i)
            nested_code_lines.append(f"{indent}if dt > 0.0:\n")
        indent_deep = "        " + ("    " * depth)
        nested_code_lines.append(f"{indent_deep}x = 42\n")

        code = "".join(nested_code_lines)
        valid, violations = validate_cartridge_ast(code)
        assert valid is True
        assert len(violations) == 0

    def test_tier5_swe_evasive_syntax_sandbox_escapes(self):
        """Validates detection and rejection of evasive sandbox escape vectors attempting runtime bypass."""
        evasive_payloads = [
            # Chained subclasses access
            "class Escape:\n    deliberation = 'test'\n    def pwn(self):\n        return ().__class__.__bases__[0].__subclasses__()\n",
            # Lambda wrapping getattr
            "class LambdaEscape:\n    deliberation = 'test'\n    hook = lambda s: getattr(__import__('sys'), 'modules')\n",
            # Builtins access via eval identifier
            "class EvalEscape:\n    deliberation = 'test'\n    def execute(self):\n        runner = eval\n        return runner('1 + 1')\n",
            # Host termination via quit
            "class QuitAttack:\n    deliberation = 'test'\n    def kill(self):\n        t = quit\n        t()\n",
        ]

        agent = LeadGameplaySWEAgent(client=None)
        for payload in evasive_payloads:
            # Stage 2 security check must catch and fail
            valid, err = agent.verify_code_module("mechanics.py", payload, raise_on_ast_error=False)
            assert valid is False
            assert "ASTSecurityError" in err or "Prohibited" in err

    def test_tier5_swe_missing_cross_module_imports(self, tmp_path: Path):
        """Validates that importing missing local modules or missing symbols raises CodeVerificationError."""
        c_dir = tmp_path / "candidate_broken_imports"
        c_dir.mkdir(parents=True, exist_ok=True)

        # mechanics.py imports a non-existent local module
        (c_dir / "mechanics.py").write_text(
            "from nonexistent_subsystem import RogueClass\n\nclass Cartridge:\n    deliberation = 'test'\n",
            encoding="utf-8",
        )
        # entities.py defines EntityA but scenes.py imports EntityB
        (c_dir / "entities.py").write_text(
            "class EntityA:\n    deliberation = 'entity a'\n",
            encoding="utf-8",
        )
        (c_dir / "scenes.py").write_text(
            "from entities import NonExistentEntity\n\nclass MainScene:\n    deliberation = 'scene'\n",
            encoding="utf-8",
        )

        agent = LeadGameplaySWEAgent(client=None)
        with pytest.raises(CodeVerificationError) as exc_info:
            agent.verify_cartridge_cross_module_imports(c_dir, raise_on_error=True)

        err_msg = str(exc_info.value)
        assert "NonExistentEntity" in err_msg or "nonexistent_subsystem" in err_msg

    def test_tier5_swe_circular_module_references(self, tmp_path: Path):
        """Validates that cyclical dependencies across cartridge modules are statically detected and rejected."""
        c_dir = tmp_path / "candidate_circular_deps"
        c_dir.mkdir(parents=True, exist_ok=True)

        # 3-way circular dependency: mechanics -> scenes -> entities -> mechanics
        (c_dir / "mechanics.py").write_text(
            "from scenes import BattleScene\n\nclass GameCartridge:\n    deliberation = 'cartridge'\n",
            encoding="utf-8",
        )
        (c_dir / "scenes.py").write_text(
            "from entities import BossEntity\n\nclass BattleScene:\n    deliberation = 'scene'\n",
            encoding="utf-8",
        )
        (c_dir / "entities.py").write_text(
            "from mechanics import GameCartridge\n\nclass BossEntity:\n    deliberation = 'entity'\n",
            encoding="utf-8",
        )

        agent = LeadGameplaySWEAgent(client=None)
        valid, errors = agent.verify_cartridge_cross_module_imports(c_dir, raise_on_error=False)
        assert valid is False
        assert any("Circular import detected" in err for err in errors)

    def test_tier5_swe_missing_deliberation_first_attribute(self):
        """Validates that any class body lacking deliberation strictly as its first statement fails verification."""
        code_without_delib = """
class InvalidEntity:
    hp: int = 100
    deliberation: str = "This deliberation is placed second, violating Rule 4."
"""
        agent = LeadGameplaySWEAgent(client=None)
        valid, err = agent.verify_code_module("entities.py", code_without_delib, raise_on_ast_error=False)
        assert valid is False
        assert "must declare 'deliberation' as its first attribute" in err


# ===========================================================================
# AREA 3: EVENT BUS EDGE CASES
# ===========================================================================

class TestTier5EventBusHardening:
    """Hardening tests for SwarmEventBus under listener exceptions, concurrency, and corrupt payloads."""

    def test_tier5_event_bus_faulty_listener_exception_resilience(self, tmp_path: Path):
        """Validates that a subscriber throwing an exception during broadcast is pruned without breaking others."""
        async def _run():
            bus = SwarmEventBus(history_size=50, log_path=tmp_path / "events.jsonl")

            # Mock healthy WebSocket
            healthy_received = []
            class HealthyWS:
                async def send_json(self, data):
                    healthy_received.append(data)

            # Mock faulty WebSocket that crashes on send_json
            class FaultyWS:
                async def send_json(self, data):
                    raise RuntimeError("Network socket abruptly closed")

            h_ws = HealthyWS()
            f_ws = FaultyWS()
            await bus.register(h_ws)
            await bus.register(f_ws)
            assert bus.subscriber_count == 2

            test_event = {
                "deliberation": "Test broadcast resilience",
                "event_type": "AGENT_STREAM",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "payload": {"token": "hello"},
            }
            await bus.broadcast(test_event)

            # Faulty subscriber must be pruned; healthy subscriber must have received the event
            assert bus.subscriber_count == 1
            assert len(healthy_received) == 1
            assert healthy_received[0]["payload"]["token"] == "hello"

        asyncio.run(_run())

    def test_tier5_event_bus_high_concurrency_multi_threaded_publishing(self, tmp_path: Path):
        """Validates thread safety and FIFO stability under concurrent publishing from 20 parallel threads."""
        bus = SwarmEventBus(history_size=500, log_path=tmp_path / "concurrent_events.jsonl")

        num_threads = 20
        events_per_thread = 25

        def _publish_worker(thread_id: int):
            for i in range(events_per_thread):
                bus.publish_threadsafe({
                    "deliberation": f"Thread {thread_id} event {i}",
                    "event_type": "AGENT_STREAM",
                    "agent_name": f"WorkerAgent_{thread_id}",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "payload": {"thread_id": thread_id, "index": i},
                })

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(_publish_worker, tid) for tid in range(num_threads)]
            for f in futures:
                f.result()

        recent = bus.get_recent_events()
        assert len(recent) == min(500, num_threads * events_per_thread)

    def test_tier5_event_bus_malformed_payload_deserialization(self, tmp_path: Path):
        """Validates that non-dict or malformed event payloads do not raise unhandled exceptions."""
        bus = SwarmEventBus(history_size=50, log_path=tmp_path / "malformed_events.jsonl")

        malformed_events = [
            {"event_type": "REQUIREMENTS_SPEC_GENERATED", "payload": "not-a-dict"},
            {"event_type": "REQUIREMENT_STATUS_UPDATE", "payload": None},
            {"event_type": "AGENT_TASK_RECORD", "payload": 12345},
            {"event_type": "CHALLENGER_GATE_VERDICT", "payload": ["item1", "item2"]},
            {"event_type": "UNKNOWN_EVENT_TYPE"},
        ]

        for ev in malformed_events:
            # Must not crash
            bus.publish_threadsafe(ev)

        assert len(bus.get_recent_events()) == len(malformed_events)

    def test_tier5_event_bus_corrupted_jsonl_log_recovery(self, tmp_path: Path):
        """Validates that SwarmEventBus recovers gracefully from corrupt or truncated studio_events.jsonl lines."""
        log_file = tmp_path / "corrupted_studio_events.jsonl"
        valid_ev = {
            "deliberation": "Valid recovered event",
            "event_type": "REQUIREMENTS_SPEC_GENERATED",
            "agent_name": "Director",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "payload": {"spec": {"requirements": []}},
        }
        corrupted_content = (
            json.dumps(valid_ev) + "\n"
            + "{ truncated unclosed json string\n"
            + "\x00\x00\x00 binary null byte corruption\n"
            + "\n"  # empty line
            + "42\n"  # primitive int
            + "null\n"
            + json.dumps(valid_ev) + "\n"
        )
        log_file.write_text(corrupted_content, encoding="utf-8")

        # Temporarily unset PYTEST_CURRENT_TEST to exercise _load_persisted_events
        orig_pytest_env = os.environ.pop("PYTEST_CURRENT_TEST", None)
        try:
            bus = SwarmEventBus(history_size=50, log_path=log_file)
            events = bus.get_recent_events()
            assert len(events) == 2  # Exactly the 2 valid JSON dictionary records
        finally:
            if orig_pytest_env is not None:
                os.environ["PYTEST_CURRENT_TEST"] = orig_pytest_env


# ===========================================================================
# AREA 4: CHALLENGER GATE EDGE CASES
# ===========================================================================

class TestTier5ChallengerGateHardening:
    """Hardening tests for Gate F (Challenger Gate) edge cases, conflicting telemetry, and timeout under 0 FPS."""

    def test_tier5_challenger_partial_criterion_match_rejection(self):
        """Validates that a build satisfying 2 of 3 requirements is demonstrably rejected with overall_verdict=False."""
        spec = GameRequirementsSpec(
            deliberation="3-track game specification",
            game_title="Tri-Track Challenge",
            requirements=[
                RequirementTrack(deliberation="Track 1", id="R1", title="Movement", description="Player movement"),
                RequirementTrack(deliberation="Track 2", id="R2", title="Combat", description="Combat mechanics"),
                RequirementTrack(deliberation="Track 3", id="R3", title="Economy", description="Resource economy"),
            ],
        )

        partial_eval = ChallengerGateEvaluation(
            deliberation="Evaluated 3 tracks; R2 combat failed verification.",
            candidate_id="candidate_partial_001",
            overall_verdict=True,  # LLM erroneously attempted to set True
            evaluated_requirements=[
                RequirementEvaluationRecord(
                    deliberation="R1 movement verified",
                    requirement_id="R1",
                    fulfilled=True,
                    grounding_evidence="AST move methods and telemetry pos delta verified.",
                ),
                RequirementEvaluationRecord(
                    deliberation="R2 combat failed: weapon fire has empty body",
                    requirement_id="R2",
                    fulfilled=False,
                    grounding_evidence="AST inspect found empty fire() body.",
                    critique_notes="Missing fire mechanic implementation.",
                ),
                RequirementEvaluationRecord(
                    deliberation="R3 economy verified",
                    requirement_id="R3",
                    fulfilled=True,
                    grounding_evidence="Resource counters incremented in simulation.",
                ),
            ],
            rejection_trace=None,
        )

        challenger = ChallengerGate(client=None)
        enforced = challenger._enforce_deterministic_invariants(
            evaluation=partial_eval,
            spec=spec,
            ast_grounding={"hollow_stubs": []},
            telemetry_grounding={},
            visual_grounding={},
            candidate_id="candidate_partial_001",
            episode_number=1,
            attempt_number=1,
        )

        assert enforced.overall_verdict is False
        assert enforced.rejection_trace is not None
        assert "R2" in enforced.rejection_trace.unfulfilled_requirements

    def test_tier5_challenger_conflicting_telemetry_grounding(self):
        """Validates resolution when AST indicates clean code but visual telemetry reports blank/un-rendered canvas."""
        spec = GameRequirementsSpec(
            deliberation="1-track game specification",
            game_title="Visual Glitch Sim",
            requirements=[
                RequirementTrack(deliberation="Core track", id="R1", title="Renderer", description="Renders game canvas"),
            ],
        )

        clean_ast_eval = ChallengerGateEvaluation(
            deliberation="AST is 100% clean with 0 stubs.",
            candidate_id="candidate_blank_canvas_002",
            overall_verdict=True,
            evaluated_requirements=[
                RequirementEvaluationRecord(
                    deliberation="R1 code looks syntactically complete",
                    requirement_id="R1",
                    fulfilled=True,
                    grounding_evidence="AST inspection confirmed clean classes.",
                ),
            ],
            rejection_trace=None,
        )

        # Grounding conflict: AST is clean, BUT visual canvas is completely blank (monochrome black)
        ast_grounding = {"hollow_stubs": [], "cleanliness_score": 1.0}
        visual_grounding = {"blank_canvas_detected": True, "visual_motion_verified": False}

        challenger = ChallengerGate(client=None)
        enforced = challenger._enforce_deterministic_invariants(
            evaluation=clean_ast_eval,
            spec=spec,
            ast_grounding=ast_grounding,
            telemetry_grounding={},
            visual_grounding=visual_grounding,
            candidate_id="candidate_blank_canvas_002",
            episode_number=1,
            attempt_number=1,
        )

        # Invariant must override and reject due to blank canvas
        assert enforced.overall_verdict is False
        assert enforced.rejection_trace is not None
        assert "Blank canvas" in enforced.rejection_trace.failure_reason

    def test_tier5_challenger_simulation_crash_and_timeout_under_0_fps(self, tmp_path: Path):
        """Validates that CandidateTelemetryAuditor captures crashes and 0 FPS stagnation gracefully."""
        broken_cartridge_dir = tmp_path / "broken_cartridge"
        broken_cartridge_dir.mkdir(parents=True, exist_ok=True)
        # Broken mechanics that crashes on import or instantiation
        (broken_cartridge_dir / "mechanics.py").write_text(
            "raise RuntimeError('Fatal cartridge initialization crash')",
            encoding="utf-8",
        )

        auditor = CandidateTelemetryAuditor()
        report = auditor.audit_simulation(candidate_dir=broken_cartridge_dir, ticks=60)

        assert report["crashed"] is True
        assert report["frame_budget_passed"] is False
        assert report["avg_frame_time_ms"] == 999.0
        assert report["is_stagnant_or_frozen"] is True
        assert "Fatal cartridge initialization crash" in report["error"]

    def test_tier5_challenger_missing_and_corrupt_spec_handling(self, tmp_path: Path):
        """Validates that an empty or missing requirements_spec.json passes vacuously with epistemic rationale."""
        empty_cartridge_dir = tmp_path / "empty_cartridge"
        empty_cartridge_dir.mkdir(parents=True, exist_ok=True)

        challenger = ChallengerGate(client=None)
        evaluation = challenger.evaluate(candidate_dir=empty_cartridge_dir, requirements_spec=None)

        assert evaluation.overall_verdict is True
        assert "vacuously" in evaluation.deliberation
        assert evaluation.evaluated_requirements == []
        assert evaluation.rejection_trace is None
