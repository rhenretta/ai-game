"""tests/test_mcp_server.py
Comprehensive unit test suite for Local Model Context Protocol (MCP) Server (Feature 11).
Verifies:
  1. Strict Rationale-First Schema Invariants (deliberation declared first in schemas and results).
  2. Fail-fast error handling for missing deliberation (-32602).
  3. All 4 MCP tools:
     - get_steam_player_dna
     - get_telemetry_profile
     - query_spatial_map
     - get_engine_api_docs
  4. Standard JSON-RPC 2.0 protocol handling (initialize, ping, tools/list, tools/call).
  5. Line-delimited stdio transport message loop.
  6. Programmatic in-process invocation API.
"""
from __future__ import annotations
import io
import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

from engine.analyst.analyst import TelemetryAnalyst
from engine.api import contracts
from engine.mcp.server import LocalMCPServer
from engine.mcp.tools import (
    BaseMCPTool,
    EngineApiDocsTool,
    SpatialMapTool,
    SteamPlayerDNATool,
    TelemetryProfileTool,
)
from engine.steam.dna import SteamDNAPersistence, SteamDNAProfiler
from engine.steam.models import SteamPlayerDNA
from engine.telemetry.db import TelemetryDB
from tests.test_tier1_units import validate_rationale_first_ast, validate_rationale_first_dict


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_telemetry.db"


@pytest.fixture
def mcp_server(temp_db_path: Path) -> LocalMCPServer:
    analyst = TelemetryAnalyst(db_path=temp_db_path)
    steam_repo = SteamDNAPersistence(db_path=temp_db_path)
    return LocalMCPServer(telemetry_analyst=analyst, steam_repository=steam_repo)


# ============================================================================
# Category 1: Schema & Rationale Invariant Tests
# ============================================================================

class TestMCPSchemaAndRationaleInvariants:
    """Verifies that deliberation is strictly enforced first in schemas, requests, and results."""

    def test_tools_list_schema_order(self, mcp_server: LocalMCPServer):
        """All 4 tools must exist, with 'deliberation' as the first schema property and required."""
        tools = mcp_server.list_tools()
        assert len(tools) == 4
        expected_names = {
            "get_steam_player_dna",
            "get_telemetry_profile",
            "query_spatial_map",
            "get_engine_api_docs",
        }
        assert {t["name"] for t in tools} == expected_names

        for tool_def in tools:
            schema = tool_def["inputSchema"]
            properties = schema["properties"]
            first_prop = list(properties.keys())[0]
            assert first_prop == "deliberation", f"Tool {tool_def['name']} did not declare deliberation first!"
            assert "deliberation" in schema.get("required", []), f"Tool {tool_def['name']} deliberation not required!"

    def test_missing_deliberation_rejected(self, mcp_server: LocalMCPServer):
        """Tool invocation without deliberation must fail fast with code -32602."""
        call_params = {
            "tool_name": "get_engine_api_docs",
            "arguments": {"module_name": "engine.api.contracts"},
            "id": 42,
        }
        resp = mcp_server.handle_tool_call("tools/call", call_params)
        assert resp["jsonrpc"] == "2.0"
        assert "error" in resp
        assert resp["error"]["code"] == -32602
        assert "deliberation" in resp["error"]["message"].lower()

    def test_tool_execution_deliberation_first(self, mcp_server: LocalMCPServer):
        """Results from all 4 tools must have 'deliberation' as the very first dictionary key."""
        # 1. API docs
        res_docs = mcp_server.call_tool(
            "get_engine_api_docs",
            deliberation="Inspecting engine contracts for testing."
        )
        assert validate_rationale_first_dict(res_docs) is True

        # 2. Steam DNA
        res_dna = mcp_server.call_tool(
            "get_steam_player_dna",
            deliberation="Inspecting DNA for testing."
        )
        assert validate_rationale_first_dict(res_dna) is True

        # 3. Spatial Map (direct grid)
        res_spatial = mcp_server.call_tool(
            "query_spatial_map",
            arguments={"grid": [[0, 0], [0, 0]], "start": [0, 0], "goal": [1, 1]},
            deliberation="Inspecting spatial map for testing."
        )
        assert validate_rationale_first_dict(res_spatial) is True

        # 4. Telemetry profile (raw events)
        res_telemetry = mcp_server.call_tool(
            "get_telemetry_profile",
            arguments={"raw_events": []},
            deliberation="Inspecting telemetry profile for testing."
        )
        assert validate_rationale_first_dict(res_telemetry) is True

    def test_mcp_classes_ast_rationale_first(self):
        """AST verification: all classes in tools.py and server.py declare deliberation first."""
        tools_path = Path(__file__).resolve().parent.parent / "engine" / "mcp" / "tools.py"
        server_path = Path(__file__).resolve().parent.parent / "engine" / "mcp" / "server.py"

        assert validate_rationale_first_ast(tools_path.read_text("utf-8")) is True
        assert validate_rationale_first_ast(server_path.read_text("utf-8")) is True


# ============================================================================
# Category 2: Tool 1 `get_steam_player_dna` Tests
# ============================================================================

class TestMCPSteamPlayerDNATool:
    """Verifies Steam DNA retrieval, SQLite persistence, and default profile fallback."""

    def test_mcp_steam_dna_default_fallback(self, mcp_server: LocalMCPServer):
        """When SQLite table is empty, a default SteamPlayerDNA is created, persisted, and returned."""
        result = mcp_server.call_tool(
            "get_steam_player_dna",
            arguments={"player_id": "test_pilot_99", "profile_tag": "test_tag"},
            deliberation="Need fallback DNA profile for new player."
        )
        assert result["deliberation"] == "Need fallback DNA profile for new player."
        assert result["player_id"] == "test_pilot_99"
        assert result["pacing_preference"] == "balanced"
        assert result["pacing_score"] == 0.0
        assert "Action" in result["top_genres"]

        # Verify it was persisted to SQLite
        stored = mcp_server.steam_repo.get_latest_dna(player_id="test_pilot_99")
        assert stored is not None
        assert stored.player_id == "test_pilot_99"

    def test_mcp_steam_dna_persisted(self, mcp_server: LocalMCPServer):
        """Pre-seeded DNA in SQLite is retrieved faithfully."""
        custom_dna = SteamPlayerDNA(
            deliberation="Pre-seeded rusher DNA fixture.",
            player_id="persisted_player_01",
            profile_tag="genesis_rusher",
            total_playtime_hours=150.0,
            pacing_preference="kinetic_high_mobility",
            pacing_score=0.85,
            challenge_tolerance=0.90,
            top_genres=["Action", "Roguelike"],
        )
        mcp_server.steam_repo.save_dna(custom_dna)

        result = mcp_server.call_tool(
            "get_steam_player_dna",
            arguments={"player_id": "persisted_player_01"},
            deliberation="Querying persisted profile."
        )
        assert result["player_id"] == "persisted_player_01"
        assert result["pacing_preference"] == "kinetic_high_mobility"
        assert result["pacing_score"] == 0.85
        assert result["top_genres"] == ["Action", "Roguelike"]


# ============================================================================
# Category 3: Tool 2 `get_telemetry_profile` Tests
# ============================================================================

class TestMCPTelemetryProfileTool:
    """Verifies behavioral telemetry profile computation from raw events or SQLite WAL."""

    def test_mcp_telemetry_profile_raw_events(self, mcp_server: LocalMCPServer):
        """Passes in-memory events and computes profile without touching disk."""
        events = [
            {"event_type": "input", "timestamp": 1.0, "payload": {"move_x": 1.0, "move_y": 0.0}},
            {"event_type": "dash", "timestamp": 1.5, "payload": {"success": True}},
            {"event_type": "input", "timestamp": 2.0, "payload": {"move_x": 1.0, "move_y": 0.0}},
        ]
        result = mcp_server.call_tool(
            "get_telemetry_profile",
            arguments={"session_id": "test_session", "raw_events": events},
            deliberation="Analyzing synthetic event stream."
        )
        assert result["deliberation"] == "Analyzing synthetic event stream."
        assert "playstyle_tag" in result
        assert "kinetic" in result
        assert "evasion" in result
        assert "spatial" in result

    def test_mcp_telemetry_profile_from_db(self, mcp_server: LocalMCPServer, temp_db_path: Path):
        """Writes session to SQLite and queries through MCP tool."""
        # Initialize tables
        tdb = TelemetryDB(db_path=temp_db_path)

        conn = tdb.get_writer_connection()
        try:
            conn.execute(
                "INSERT INTO telemetry_sessions (session_id, player_profile_tag) VALUES ('sess_mcp_01', 'pilot_1');"
            )
            conn.execute(
                "INSERT INTO telemetry_episodes (episode_id, session_id, episode_number) VALUES ('ep_1', 'sess_mcp_01', 1);"
            )
            conn.execute(
                "INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y, payload_json) VALUES ('sess_mcp_01', 'ep_1', 1.0, 'dash', 100.0, 100.0, '{\"success\": true}');"
            )
            conn.commit()
        finally:
            conn.close()

        result = mcp_server.call_tool(
            "get_telemetry_profile",
            arguments={"session_id": "sess_mcp_01"},
            deliberation="Querying telemetry recorded in SQLite DB."
        )
        assert result["session_id"] == "sess_mcp_01"
        assert "playstyle_tag" in result


# ============================================================================
# Category 4: Tool 3 `query_spatial_map` Tests
# ============================================================================

class TestMCPSpatialMapTool:
    """Verifies NavMesh queries, wall bounds, and A* pathfinding via spatial map tool."""

    def test_mcp_query_spatial_map_direct_grid(self, mcp_server: LocalMCPServer):
        """Direct 2D grid matrix traversability test."""
        # 5x5 grid with obstacle at (2, 2)
        grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        result = mcp_server.call_tool(
            "query_spatial_map",
            arguments={"grid": grid, "start": [0, 0], "goal": [4, 4]},
            deliberation="Checking direct grid traversability around central obstacle."
        )
        assert result["is_traversable"] is True
        assert result["path_length"] > 0
        assert [0, 0] in result["path"]
        assert [4, 4] in result["path"]

    def test_mcp_query_spatial_map_blocked_grid(self, mcp_server: LocalMCPServer):
        """Completely blocked grid returns is_traversable=False."""
        grid = [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]
        result = mcp_server.call_tool(
            "query_spatial_map",
            arguments={"grid": grid, "start": [0, 0], "goal": [2, 2]},
            deliberation="Checking blocked grid wall."
        )
        assert result["is_traversable"] is False
        assert result["path_length"] == 0
        assert result["path"] == []

    def test_mcp_query_spatial_map_cartridge(self, mcp_server: LocalMCPServer, tmp_path: Path):
        """Loads cartridge level specification, builds NavMesh, verifies spawn-to-exit path."""
        cart_dir = tmp_path / "spatial_cartridge"
        cart_dir.mkdir(parents=True)
        (cart_dir / "manifest.json").write_text(json.dumps({
            "cartridge_id": "test_spatial",
            "entrypoint": "mechanics.py"
        }), encoding="utf-8")
        (cart_dir / "mechanics.py").write_text('''
from engine.api.contracts import BaseCartridge, BaseLevelSpec
class SpatialCartridge(BaseCartridge):
    def get_level_spec(self, episode_number=1):
        return BaseLevelSpec(
            level_id="lvl_test",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 100.0),
            exit_point=(1100.0, 600.0),
            walls=[(0, 0, 1280, 20), (0, 700, 1280, 20), (0, 0, 20, 720), (1260, 0, 20, 720)],
        )
''', encoding="utf-8")
        result = mcp_server.call_tool(
            "query_spatial_map",
            arguments={"cartridge_path": str(cart_dir)},
            deliberation="Checking cartridge geometry."
        )
        assert result["is_traversable"] is True
        assert result["path_length"] > 0
        assert len(result["walls"]) > 0
        assert result["width"] == 1280.0
        assert result["height"] == 720.0

    def test_mcp_query_spatial_map_point_query(self, mcp_server: LocalMCPServer, tmp_path: Path):
        """Checks point clearance query."""
        cart_dir = tmp_path / "spatial_cartridge_point"
        cart_dir.mkdir(parents=True)
        (cart_dir / "manifest.json").write_text(json.dumps({
            "cartridge_id": "test_spatial",
            "entrypoint": "mechanics.py"
        }), encoding="utf-8")
        (cart_dir / "mechanics.py").write_text('''
from engine.api.contracts import BaseCartridge, BaseLevelSpec
class SpatialCartridge(BaseCartridge):
    def get_level_spec(self, episode_number=1):
        return BaseLevelSpec(
            level_id="lvl_test",
            width=1280.0,
            height=720.0,
            spawn_point=(100.0, 100.0),
            exit_point=(1100.0, 600.0),
            walls=[(0, 0, 1280, 20)],
        )
''', encoding="utf-8")
        result = mcp_server.call_tool(
            "query_spatial_map",
            arguments={"cartridge_path": str(cart_dir), "point": [200.0, 200.0]},
            deliberation="Checking point walkability."
        )
        assert "point_valid" in result
        assert isinstance(result["point_valid"], bool)

    def test_mcp_query_spatial_map_missing_cartridge_returns_clean_error(self, mcp_server: LocalMCPServer):
        """Verifies missing active cartridge returns clean non-traversable error without seed fallback."""
        result = mcp_server.call_tool(
            "query_spatial_map",
            arguments={},
            deliberation="Checking missing active cartridge behavior."
        )
        assert result["is_traversable"] is False


# ============================================================================
# Category 5: Tool 4 `get_engine_api_docs` Tests
# ============================================================================

class TestMCPEngineApiDocsTool:
    """Verifies SDK documentation retrieval and backwards-compatible showrunner lookup."""

    def test_mcp_get_engine_api_docs_contracts(self, mcp_server: LocalMCPServer):
        """Retrieves authoritative contracts and verifies version 1.0.0."""
        result = mcp_server.call_tool(
            "get_engine_api_docs",
            arguments={"module_name": "engine.api.contracts"},
            deliberation="Querying engine contracts documentation."
        )
        assert result["api_version"] == contracts.ENGINE_API_VERSION
        assert "BaseCartridge" in result["contracts"]
        assert "BasePlayer" in result["contracts"]
        assert "BaseLevelSpec" in result["contracts"]
        assert any("get_level_spec" in k for k in result["contracts"]["BaseCartridge"]["methods"])

    def test_mcp_get_engine_api_docs_class_filter(self, mcp_server: LocalMCPServer):
        """Filtering by class_name returns specifically targeted definition."""
        result = mcp_server.call_tool(
            "get_engine_api_docs",
            arguments={"class_name": "BaseCartridge"},
            deliberation="Inspecting BaseCartridge contract."
        )
        assert "BaseCartridge" in result["contracts"]
        assert "BasePlayer" not in result["contracts"]

    def test_mcp_get_engine_api_docs_showrunner_compat(self, mcp_server: LocalMCPServer):
        """Ensures backwards compatibility with showrunner tests querying GameLoop and TelemetryRecorder."""
        result = mcp_server.call_tool(
            "get_engine_api_docs",
            arguments={"module_name": "engine.telemetry.TelemetryRecorder"},
            deliberation="Inspecting TelemetryRecorder."
        )
        assert "SQLite WAL logger" in result["docs"]
        assert result["module"] == "engine.telemetry.TelemetryRecorder"


# ============================================================================
# Category 6: JSON-RPC 2.0 Protocol & Stdio Transport Tests
# ============================================================================

class TestMCPProtocolAndStdio:
    """Verifies JSON-RPC 2.0 handshake, errors, and line-delimited stdio transport."""

    def test_jsonrpc_initialize(self, mcp_server: LocalMCPServer):
        """Sends initialize message and checks capabilities."""
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        resp = mcp_server.handle_jsonrpc(req)
        assert resp is not None
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert "tools" in resp["result"]["capabilities"]

    def test_jsonrpc_notifications_initialized(self, mcp_server: LocalMCPServer):
        """Notifications require no response."""
        req = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        resp = mcp_server.handle_jsonrpc(req)
        assert resp is None

    def test_jsonrpc_ping(self, mcp_server: LocalMCPServer):
        req = {"jsonrpc": "2.0", "id": 99, "method": "ping"}
        resp = mcp_server.handle_jsonrpc(req)
        assert resp == {"jsonrpc": "2.0", "result": {}, "id": 99}

    def test_jsonrpc_tools_list(self, mcp_server: LocalMCPServer):
        req = {"jsonrpc": "2.0", "id": 5, "method": "tools/list"}
        resp = mcp_server.handle_jsonrpc(req)
        assert resp is not None
        assert "tools" in resp["result"]
        assert len(resp["result"]["tools"]) == 4

    def test_jsonrpc_tools_call(self, mcp_server: LocalMCPServer):
        req = {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "get_engine_api_docs",
                "deliberation": "Testing JSON-RPC call.",
                "arguments": {"module_name": "engine.api.contracts"}
            }
        }
        resp = mcp_server.handle_jsonrpc(req)
        assert resp is not None
        assert resp["id"] == 10
        assert "result" in resp
        assert resp["result"]["deliberation"] == "Testing JSON-RPC call."

    def test_jsonrpc_invalid_method(self, mcp_server: LocalMCPServer):
        req = {"jsonrpc": "2.0", "id": 11, "method": "non_existent_method"}
        resp = mcp_server.handle_jsonrpc(req)
        assert resp is not None
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_jsonrpc_parse_error(self, mcp_server: LocalMCPServer):
        resp = mcp_server.handle_jsonrpc("INVALID_JSON_RAW_STRING")
        assert resp is not None
        assert resp["error"]["code"] == -32700

    def test_stdio_transport_pipe(self, mcp_server: LocalMCPServer):
        """Feeds string streams through run_stdio and parses response lines."""
        requests = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
            json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_engine_api_docs",
                    "deliberation": "Stdio test call.",
                    "arguments": {}
                }
            }),
        ]
        in_stream = io.StringIO("\n".join(requests) + "\n")
        out_stream = io.StringIO()

        mcp_server.run_stdio(input_stream=in_stream, output_stream=out_stream)

        output_lines = [line.strip() for line in out_stream.getvalue().splitlines() if line.strip()]
        assert len(output_lines) == 2

        res1 = json.loads(output_lines[0])
        assert res1["id"] == 1
        assert res1["result"] == {}

        res2 = json.loads(output_lines[1])
        assert res2["id"] == 2
        assert res2["result"]["deliberation"] == "Stdio test call."


# ============================================================================
# Category 7: Programmatic In-Process Invocation API Tests
# ============================================================================

class TestMCPProgrammaticInvocation:
    """Verifies call_tool in-process programmatic invocation API."""

    def test_programmatic_call_tool_success(self, mcp_server: LocalMCPServer):
        res = mcp_server.call_tool(
            "get_engine_api_docs",
            arguments={"class_name": "BaseCartridge"},
            deliberation="Programmatic in-process check."
        )
        assert res["deliberation"] == "Programmatic in-process check."
        assert "BaseCartridge" in res["contracts"]

    def test_programmatic_call_tool_missing_deliberation_raises(self, mcp_server: LocalMCPServer):
        with pytest.raises(ValueError, match="Missing mandatory deliberation"):
            mcp_server.call_tool("get_engine_api_docs", arguments={})

    def test_programmatic_call_tool_unknown_raises(self, mcp_server: LocalMCPServer):
        with pytest.raises(KeyError, match="Unknown tool 'fake_tool'"):
            mcp_server.call_tool("fake_tool", deliberation="Testing invalid tool.")
