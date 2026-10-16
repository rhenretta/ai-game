"""tests/test_telemetry_architect.py
Tests for Feature 22: Telemetry Architect Agent.
Verifies AST code inspection, probe catalog registration in SQLite WAL,
safe instrumentation injection, and strict Rationale-First compliance.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import tempfile
import pytest

from engine.loader import validate_cartridge_ast
from engine.studio.models import TelemetryArchitecturePlan, TelemetryProbeDesignSpec
from engine.studio.telemetry_architect import TelemetryArchitectAgent
from engine.telemetry.db import TelemetryDB
from engine.telemetry.probes import AgentTelemetryProbe
from engine.telemetry.recorder import TelemetryRecorder


class TestTelemetryArchitectAgent:
    """Test suite for autonomous telemetry probe architecting and instrumentation."""

    def test_extract_ast_targets_finds_classes_and_methods(self, tmp_path: Path) -> None:
        db_path = tmp_path / "telemetry_test.db"
        architect = TelemetryArchitectAgent(db_path=db_path)

        code_modules = {
            "entities.py": (
                "class DynamicPlayer:\n"
                "    def __init__(self, x, y):\n"
                "        self.x = x\n"
                "    def dash(self):\n"
                "        pass\n"
                "class BossEnemy:\n"
                "    def trigger_phase_two(self):\n"
                "        pass\n"
            ),
            "mechanics.py": (
                "class GeneratedCartridge:\n"
                "    def create_player(self, x, y):\n"
                "        pass\n"
                "    def create_enemies(self, spec):\n"
                "        pass\n"
            ),
        }

        targets = architect._extract_ast_targets(code_modules)
        assert len(targets) >= 3
        assert any(t["class"] == "DynamicPlayer" and t["method"] == "dash" for t in targets)
        assert any(t["class"] == "BossEnemy" and t["method"] == "trigger_phase_two" for t in targets)
        assert any(t["class"] == "GeneratedCartridge" and t["method"] == "create_player" for t in targets)

    def test_plan_telemetry_architecture_rationale_first(self, tmp_path: Path, mock_gemini_client) -> None:
        db_path = tmp_path / "telemetry_test.db"
        mock_plan = {
            "deliberation": "Formulated telemetry architecture addressing hypothesis targets.",
            "probes": [
                {
                    "deliberation": "Instrumenting DashAbility.execute for mobility telemetry.",
                    "probe_name": "dashability_execute_probe",
                    "category": "movement",
                    "description": "Tracks dash mobility cadence.",
                    "hypothesis_target": "H_MOBILITY",
                    "payload_schema": {"type": "object", "properties": {"distance": {"type": "number"}}, "required": ["distance"]},
                    "target_module": "abilities.py",
                    "target_class": "DashAbility",
                    "target_method": "execute",
                    "max_event_rate_hz": 60.0,
                }
            ],
            "hypotheses_addressed": ["H_MOBILITY"],
            "estimated_total_throughput_hz": 60.0,
        }
        mock_gemini_client.set_json_response(mock_plan)
        architect = TelemetryArchitectAgent(db_path=db_path, gemini_client=mock_gemini_client)

        code_modules = {
            "abilities.py": (
                "class DashAbility:\n"
                "    def execute(self, user):\n"
                "        pass\n"
            )
        }

        plan = architect.inspect_code_and_plan(code_modules)
        assert isinstance(plan, TelemetryArchitecturePlan)
        assert plan.deliberation != ""
        assert list(plan.__dict__.keys())[0] == "deliberation"
        assert len(plan.probes) >= 1
        for probe in plan.probes:
            assert isinstance(probe, TelemetryProbeDesignSpec)
            assert probe.deliberation != ""
            assert list(probe.__dict__.keys())[0] == "deliberation"
            assert probe.probe_name != ""
            assert probe.category in {
                "combat", "movement", "dialogue", "ability", "risk_reward", "exploration"
            }

    def test_probe_registration_in_sqlite_wal(self, tmp_path: Path, mock_gemini_client) -> None:
        db_path = tmp_path / "telemetry_test.db"
        mock_plan = {
            "deliberation": "Formulated telemetry architecture for player mobility.",
            "probes": [
                {
                    "deliberation": "Instrumenting DynamicPlayer.dash for player evasion.",
                    "probe_name": "dynamicplayer_dash_evasion",
                    "category": "movement",
                    "description": "Tracks player dash activations.",
                    "hypothesis_target": "H_PLAYER_MOBILITY_AND_DASH_CADENCE",
                    "payload_schema": {"type": "object", "properties": {"distance": {"type": "number"}}, "required": ["distance"]},
                    "target_module": "entities.py",
                    "target_class": "DynamicPlayer",
                    "target_method": "dash",
                    "max_event_rate_hz": 60.0,
                }
            ],
            "hypotheses_addressed": ["H_PLAYER_MOBILITY_AND_DASH_CADENCE"],
            "estimated_total_throughput_hz": 60.0,
        }
        mock_gemini_client.set_json_response(mock_plan)
        architect = TelemetryArchitectAgent(db_path=db_path, gemini_client=mock_gemini_client)

        code_modules = {
            "entities.py": (
                "class DynamicPlayer:\n"
                "    def dash(self):\n"
                "        pass\n"
            )
        }

        plan = architect.inspect_code_and_plan(code_modules)
        count = architect.register_probes(plan)
        assert count == len(plan.probes)
        assert count >= 1

        db = TelemetryDB(db_path)
        for probe_spec in plan.probes:
            record = db.get_probe(probe_spec.probe_name)
            assert record is not None
            assert record["probe_name"] == probe_spec.probe_name
            assert record["category"] == probe_spec.category
            assert record["hypothesis_target"] == probe_spec.hypothesis_target

    def test_instrument_code_passes_ast_sandbox_security(self, tmp_path: Path, mock_gemini_client) -> None:
        db_path = tmp_path / "telemetry_test.db"
        mock_plan = {
            "deliberation": "Formulated telemetry architecture for player dashing.",
            "probes": [
                {
                    "deliberation": "Instrumenting DynamicPlayer.dash.",
                    "probe_name": "dynamicplayer_dash",
                    "category": "movement",
                    "description": "Tracks dash events.",
                    "hypothesis_target": "H_MOBILITY",
                    "payload_schema": {"type": "object", "properties": {"dist": {"type": "number"}}},
                    "target_module": "entities.py",
                    "target_class": "DynamicPlayer",
                    "target_method": "dash",
                    "max_event_rate_hz": 60.0,
                }
            ],
            "hypotheses_addressed": ["H_MOBILITY"],
            "estimated_total_throughput_hz": 60.0,
        }
        mock_gemini_client.set_json_response(mock_plan)
        architect = TelemetryArchitectAgent(db_path=db_path, gemini_client=mock_gemini_client)

        sample_entities_code = (
            "from __future__ import annotations\n"
            "from typing import Any, Optional\n"
            "from engine.entities.player import Player\n"
            "\n"
            "class DynamicPlayer(Player):\n"
            "    deliberation: str = 'Dynamic player actor.'\n"
            "    def __init__(self, x: float = 0.0, y: float = 0.0, telemetry_recorder: Optional[Any] = None):\n"
            "        super().__init__(x=x, y=y, telemetry_recorder=telemetry_recorder)\n"
            "    def dash(self) -> bool:\n"
            "        return True\n"
        )

        code_modules = {"entities.py": sample_entities_code}
        plan = architect.inspect_code_and_plan(code_modules)
        res = architect.instrument_code(code_modules, plan)

        assert res.deliberation != ""
        assert list(res.__dict__.keys())[0] == "deliberation"
        assert "entities.py" in res.instrumented_code
        instrumented_code = res.instrumented_code["entities.py"]

        # Validate that instrumented code parses cleanly into valid Python AST
        tree = ast.parse(instrumented_code)
        assert tree is not None
        # Validate AST security sandbox
        valid, violations = validate_cartridge_ast(instrumented_code)
        assert valid is True, f"AST violations: {violations}"

    def test_instrumented_probe_runtime_emission(self, tmp_path: Path) -> None:
        db_path = tmp_path / "telemetry_test.db"
        recorder = TelemetryRecorder(db_path=db_path, batch_size=1, flush_interval_sec=0.01)
        recorder.start_session("probe_session")
        recorder.start_episode(level_id="lvl_01", episode_index=1)

        probe = AgentTelemetryProbe(
            recorder=recorder,
            probe_name="test_probe_dash",
            category="movement",
            description="Testing probe emission",
            hypothesis_target="H_MOBILITY",
        )

        # Emit telemetry event
        emitted_payload = probe.emit(pos=(100.0, 200.0), data={"speed": 250.0})
        assert emitted_payload is not None
        recorder.flush()

        reader = recorder.db.get_reader_connection()
        cur = reader.cursor()
        cur.execute(
            "SELECT event_type, probe_name, pos_x, pos_y, payload_json FROM telemetry_events WHERE probe_name = ?;",
            ("test_probe_dash",)
        )
        row = cur.fetchone()
        assert row is not None
        assert row["event_type"] == "CUSTOM_PROBE"
        assert row["probe_name"] == "test_probe_dash"
        assert row["pos_x"] == 100.0
        assert row["pos_y"] == 200.0
        payload = json.loads(row["payload_json"])
        assert payload["category"] == "movement"
        assert payload["speed"] == 250.0

        reader.close()
        recorder.end_episode()
        recorder.end_session()
        recorder.close()
