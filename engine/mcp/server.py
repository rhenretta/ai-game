"""engine/mcp/server.py
Local Model Context Protocol (MCP) Server Data Bridge for Autonomous Game Engine & Showrunner.
Provides standard JSON-RPC 2.0 stdio protocol handler and programmatic in-process invocation API.
Enforces strict Rationale-First Epistemic Architecture.
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

from engine.analyst.analyst import TelemetryAnalyst
from engine.mcp.tools import (
    BaseMCPTool,
    EngineApiDocsTool,
    SpatialMapTool,
    SteamPlayerDNATool,
    TelemetryProfileTool,
)
from engine.steam.dna import SteamDNAPersistence

logger = logging.getLogger("LocalMCPServer")


class LocalMCPServer:
    """Local MCP Server hosting data bridge tools for Autonomous Game Engine & Showrunner."""
    deliberation: str = "Local Model Context Protocol (MCP) server hosting game data bridge tools."

    def __init__(
        self,
        telemetry_analyst: Optional[TelemetryAnalyst] = None,
        steam_repository: Optional[SteamDNAPersistence] = None,
        tile_size: float = 32.0,
    ):
        self.analyst = telemetry_analyst or TelemetryAnalyst()
        self.steam_repo = steam_repository or SteamDNAPersistence()

        # Initialize tools
        self.tools: Dict[str, BaseMCPTool] = {
            "get_steam_player_dna": SteamPlayerDNATool(repository=self.steam_repo),
            "get_telemetry_profile": TelemetryProfileTool(analyst=self.analyst),
            "query_spatial_map": SpatialMapTool(tile_size=tile_size),
            "get_engine_api_docs": EngineApiDocsTool(),
        }

    # ------------------------------------------------------------------------
    # Programmatic In-Process Invocation API
    # ------------------------------------------------------------------------

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        deliberation: Optional[str] = None
    ) -> Dict[str, Any]:
        """Programmatic in-process tool invocation.
        Requires deliberation parameter either explicitly or within arguments.
        """
        args = arguments or {}
        delib = deliberation or args.get("deliberation", "")
        if not delib:
            raise ValueError(
                f"Missing mandatory deliberation parameter for tool '{name}'. "
                "Agent must spend deliberation tokens first."
            )

        tool = self.tools.get(name)
        if not tool:
            raise KeyError(f"Unknown tool '{name}'. Registered: {list(self.tools.keys())}")

        return tool.execute(args, deliberation=delib)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns MCP tool definitions with deliberation strictly first in schema."""
        tools_list: List[Dict[str, Any]] = []
        for name, tool in self.tools.items():
            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.get_input_schema(),
            })
        return tools_list

    # ------------------------------------------------------------------------
    # JSON-RPC 2.0 Dispatcher (Compatible with test_tier3_showrunner.py & MCP)
    # ------------------------------------------------------------------------

    def handle_tool_call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches standard MCP JSON-RPC 2.0 tool requests.
        Directly matches the test_tier3_showrunner.py interface.
        """
        request_id = params.get("id", 1)

        # Enforce deliberation in tool call params
        deliberation = params.get("deliberation", "")
        args = params.get("arguments", {})
        if not deliberation and isinstance(args, dict):
            deliberation = args.get("deliberation", "")

        if not deliberation:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32602,
                    "message": "Missing mandatory deliberation parameter. Agent must deliberate first."
                },
                "id": request_id
            }

        tool_name = params.get("tool_name") or params.get("name")
        if not tool_name:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": "Missing tool name in params."},
                "id": request_id
            }

        tool = self.tools.get(tool_name)
        if not tool:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Unknown tool {tool_name}"},
                "id": request_id
            }

        try:
            result = tool.execute(args, deliberation=deliberation)
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }
        except Exception as ex:
            logger.exception("Error executing tool %s", tool_name)
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Internal tool execution error: {str(ex)}"},
                "id": request_id
            }

    def handle_jsonrpc(self, request: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Processes a standard JSON-RPC 2.0 message."""
        if isinstance(request, str):
            try:
                req_obj = json.loads(request)
            except Exception:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Parse error: Invalid JSON string."},
                    "id": None
                }
        else:
            req_obj = request

        if not isinstance(req_obj, dict):
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": "Invalid Request: Expected JSON object."},
                "id": None
            }

        req_id = req_obj.get("id")
        method = req_obj.get("method", "")
        params = req_obj.get("params", {})

        # Standard MCP / JSON-RPC Methods
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False}
                    },
                    "serverInfo": {
                        "name": "ai-game-engine-mcp",
                        "version": "1.0.0"
                    }
                },
                "id": req_id
            }
        elif method == "notifications/initialized":
            return None  # Notification: no response required
        elif method == "ping":
            return {"jsonrpc": "2.0", "result": {}, "id": req_id}
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "result": {"tools": self.list_tools()},
                "id": req_id
            }
        elif method == "tools/call":
            params_with_id = dict(params) if isinstance(params, dict) else {}
            params_with_id["id"] = req_id
            return self.handle_tool_call(method, params_with_id)
        else:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method '{method}' not found."},
                "id": req_id
            }

    # ------------------------------------------------------------------------
    # Stdio Loop for CLI and External Process Invocation
    # ------------------------------------------------------------------------

    def run_stdio(self, input_stream: Optional[TextIO] = None, output_stream: Optional[TextIO] = None) -> None:
        """Runs the standard line-delimited JSON-RPC loop over stdio."""
        in_stream = input_stream or sys.stdin
        out_stream = output_stream or sys.stdout

        for line in in_stream:
            line_str = line.strip()
            if not line_str:
                continue
            response = self.handle_jsonrpc(line_str)
            if response is not None:
                out_stream.write(json.dumps(response) + "\n")
                out_stream.flush()


def main() -> None:
    """CLI entry point for running MCP server as a standalone daemon."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server = LocalMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
