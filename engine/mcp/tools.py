"""engine/mcp/tools.py
Rationale-First MCP Tool Definitions and Handlers for Autonomous Game Engine & Showrunner.
Enforces that deliberation is declared first on all input schemas and output results.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config import DB_PATH, PROJECT_ROOT
from engine.analyst.analyst import TelemetryAnalyst
from engine.analyst.models import BehavioralProfile
from engine.api import contracts
from engine.loader import CartridgeLoader
from engine.spatial.aabb import AABB, Vector2
from engine.spatial.navmesh import NavMesh, astar_pathfind
from engine.steam.dna import SteamDNAPersistence, SteamDNAProfiler
from engine.steam.models import SteamPlayerDNA

logger = logging.getLogger("MCPTools")


class BaseMCPTool(ABC):
    """Abstract base class for all Local MCP Server tools."""
    deliberation: str = "Abstract base contract for local Model Context Protocol tools."
    name: str = ""
    description: str = ""

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Returns JSON schema with deliberation strictly declared first."""
        pass

    @abstractmethod
    def execute(self, arguments: Dict[str, Any], deliberation: str) -> Dict[str, Any]:
        """Executes tool logic and returns dictionary with deliberation strictly first."""
        pass


class SteamPlayerDNATool(BaseMCPTool):
    """MCP Tool: get_steam_player_dna
    Queries local Steam DNA SQLite table or fallback scanner.
    """
    deliberation: str = "Retrieves player Steam DNA profile from SQLite WAL storage."
    name: str = "get_steam_player_dna"
    description: str = (
        "Retrieves the player's Steam DNA profile (pacing preference, genre affinities, "
        "challenge tolerance, mechanical biases, and palette mood) from SQLite WAL storage."
    )

    def __init__(
        self,
        repository: Optional[SteamDNAPersistence] = None,
        profiler: Optional[SteamDNAProfiler] = None,
    ):
        self.repository = repository or SteamDNAPersistence()
        self.profiler = profiler or SteamDNAProfiler()

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "deliberation": {
                    "type": "string",
                    "description": "Epistemic rationale explaining why player DNA is requested."
                },
                "player_id": {
                    "type": "string",
                    "description": "Optional SteamID64 or local player identifier."
                },
                "profile_tag": {
                    "type": "string",
                    "description": "Optional profile archetype tag filter."
                }
            },
            "required": ["deliberation"]
        }

    def execute(self, arguments: Dict[str, Any], deliberation: str) -> Dict[str, Any]:
        player_id = arguments.get("player_id")
        profile_tag = arguments.get("profile_tag")

        dna = self.repository.get_latest_dna(player_id=player_id, profile_tag=profile_tag)
        if dna is None:
            dna = self.profiler._build_default_dna(
                player_id or "local_player",
                profile_tag or "genesis_default"
            )
            self.repository.save_dna(dna)

        dna_dict = dna.to_dict()
        # Guarantee deliberation is the first key in the returned dictionary
        result: Dict[str, Any] = {"deliberation": deliberation or dna_dict.get("deliberation", "")}
        for k, v in dna_dict.items():
            if k != "deliberation":
                result[k] = v
        return result


class TelemetryProfileTool(BaseMCPTool):
    """MCP Tool: get_telemetry_profile
    Queries SQLite WAL telemetry_events and runs TelemetryAnalyst.
    """
    deliberation: str = "Queries telemetry database and computes player behavioral profile."
    name: str = "get_telemetry_profile"
    description: str = (
        "Queries live SQLite WAL telemetry and computes the player's BehavioralProfile, "
        "including kinetic velocity, evasion metrics, 16x16 spatial occupancy heatmap, and retry curve."
    )

    def __init__(self, analyst: Optional[TelemetryAnalyst] = None):
        self.analyst = analyst or TelemetryAnalyst()

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "deliberation": {
                    "type": "string",
                    "description": "Epistemic rationale explaining telemetry analysis query intent."
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional target telemetry session ID."
                },
                "episode_id": {
                    "type": "string",
                    "description": "Optional target telemetry episode ID."
                },
                "raw_events": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional in-memory event stream for fast synthetic evaluation."
                }
            },
            "required": ["deliberation"]
        }

    def execute(self, arguments: Dict[str, Any], deliberation: str) -> Dict[str, Any]:
        session_id = arguments.get("session_id")
        episode_id = arguments.get("episode_id")
        raw_events = arguments.get("raw_events")

        if raw_events is not None:
            profile = self.analyst.analyze(
                session_id=session_id or "mem_session",
                episode_id=episode_id or "mem_episode",
                events=raw_events,
            )
        elif episode_id:
            profile = self.analyst.analyze_episode(episode_id)
        elif session_id:
            profile = self.analyst.analyze_session(session_id)
        else:
            # Query latest session from SQLite via reader connection
            conn = self.analyst._get_reader()
            try:
                cur = conn.cursor()
                cur.execute("SELECT session_id FROM telemetry_sessions ORDER BY created_at DESC LIMIT 1;")
                row = cur.fetchone()
                if row:
                    profile = self.analyst.analyze_session(row["session_id"])
                else:
                    profile = self.analyst.analyze_events([])
            finally:
                conn.close()

        prof_dict = profile.to_dict()
        result: Dict[str, Any] = {"deliberation": deliberation or prof_dict.get("deliberation", "")}
        for k, v in prof_dict.items():
            if k != "deliberation":
                result[k] = v
        return result


class SpatialMapTool(BaseMCPTool):
    """MCP Tool: query_spatial_map
    Queries active level geometry, wall bounds, and NavMesh traversability.
    """
    deliberation: str = "Evaluates spatial geometry, NavMesh, and traversability."
    name: str = "query_spatial_map"
    description: str = (
        "Queries active level geometry, wall bounding boxes, and validates A* NavMesh traversability "
        "between spawn and exit portals, or tests specific coordinates for clearance."
    )

    def __init__(self, tile_size: float = 32.0):
        self.tile_size = tile_size

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "deliberation": {
                    "type": "string",
                    "description": "Epistemic rationale explaining spatial geometry and clearance query."
                },
                "level_id": {
                    "type": "string",
                    "description": "Optional level identifier to inspect."
                },
                "cartridge_path": {
                    "type": "string",
                    "description": "Optional cartridge directory path (defaults to active)."
                },
                "start": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional [x, y] start coordinate for pathfinding."
                },
                "goal": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional [x, y] goal coordinate for pathfinding."
                },
                "point": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional [x, y] coordinate to test for walkability."
                },
                "grid": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}},
                    "description": "Optional direct 2D grid matrix for synthetic playtest validation."
                }
            },
            "required": ["deliberation"]
        }

    def execute(self, arguments: Dict[str, Any], deliberation: str) -> Dict[str, Any]:
        # Handle direct grid input (e.g. from tests or supervisor)
        if "grid" in arguments and arguments["grid"]:
            grid = arguments["grid"]
            start = tuple(arguments.get("start", (0, 0)))
            goal = tuple(arguments.get("goal", (0, 0)))
            path = astar_pathfind(grid, start, goal)
            is_trav = path is not None
            path_len = len(path) if path else 0
            return {
                "deliberation": deliberation or f"Direct grid evaluation: traversable={is_trav}, path_length={path_len}.",
                "is_traversable": is_trav,
                "path_length": path_len,
                "path": [list(p) for p in path] if path else [],
            }

        # Otherwise, load level specification from cartridge
        cartridge_path_val = arguments.get("cartridge_path")
        if cartridge_path_val:
            cartridge_path = Path(cartridge_path_val).resolve()
        else:
            active_p = PROJECT_ROOT / "cartridges" / "active"
            if not active_p.exists():
                return {
                    "deliberation": "No active cartridge exists in cartridges/active and no cartridge_path provided.",
                    "is_traversable": False,
                    "path_length": 0,
                    "path": [],
                    "error": "No active cartridge found in cartridges/active and no cartridge_path provided",
                }
            cartridge_path = active_p

        if not cartridge_path.exists():
            return {
                "deliberation": f"Cartridge directory not found: {cartridge_path}.",
                "is_traversable": False,
                "path_length": 0,
                "path": [],
                "error": f"Cartridge directory not found: {cartridge_path}",
            }
        cartridge = CartridgeLoader.load_cartridge(cartridge_path)
        spec = cartridge.get_level_spec(episode_number=1)

        width = spec.width
        height = spec.height
        tile_size = self.tile_size
        cols = max(1, int(width // tile_size))
        rows = max(1, int(height // tile_size))

        # Build occupancy grid from walls
        grid = [[0 for _ in range(cols)] for _ in range(rows)]
        for wx, wy, ww, wh in spec.walls:
            min_tx = max(0, int(wx // tile_size))
            max_tx = min(cols - 1, int((wx + ww - 1) // tile_size))
            min_ty = max(0, int(wy // tile_size))
            max_ty = min(rows - 1, int((wy + wh - 1) // tile_size))
            for r in range(min_ty, max_ty + 1):
                for c in range(min_tx, max_tx + 1):
                    grid[r][c] = 1

        navmesh = NavMesh(grid, tile_size=tile_size)

        # Determine start and goal coordinates
        raw_start = arguments.get("start") or spec.spawn_point
        raw_goal = arguments.get("goal") or spec.exit_point
        start_tile = navmesh.world_to_tile(raw_start[0], raw_start[1])
        goal_tile = navmesh.world_to_tile(raw_goal[0], raw_goal[1])

        path = navmesh.find_path(start_tile, goal_tile)
        is_traversable = path is not None
        path_length = len(path) if path else 0

        # Point query test if requested
        point_valid = None
        if "point" in arguments and arguments["point"]:
            px, py = arguments["point"]
            pt_tile = navmesh.world_to_tile(px, py)
            point_valid = navmesh.is_walkable(pt_tile[0], pt_tile[1])

        exec_delib = (
            deliberation
            or f"NavMesh evaluated level '{spec.level_id}' ({width}x{height}, {len(spec.walls)} walls). "
               f"Path from {raw_start} to {raw_goal}: traversable={is_traversable}, length={path_length}."
        )

        return {
            "deliberation": exec_delib,
            "is_traversable": is_traversable,
            "path_length": path_length,
            "path": path or [],
            "walls": [list(w) for w in spec.walls],
            "spawn_point": list(spec.spawn_point),
            "exit_point": list(spec.exit_point),
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "point_valid": point_valid,
            "enemies_count": len(spec.enemies),
            "hazards_count": len(spec.hazards),
        }


class EngineApiDocsTool(BaseMCPTool):
    """MCP Tool: get_engine_api_docs
    Returns version-pinned engine SDK signatures and class definitions.
    """
    deliberation: str = "Returns version-pinned signatures and class definitions for Engine SDK."
    name: str = "get_engine_api_docs"
    description: str = (
        "Returns version-pinned signatures, abstract methods, and class contracts for engine/api/contracts.py "
        "to prevent hallucinated imports or deprecated API usage."
    )

    PINNED_SHOWRUNNER_DOCS = {
        "engine.core.GameLoop": "Class GameLoop: Manages 60 FPS fixed-timestep Cocoa loop on macOS.",
        "engine.telemetry.TelemetryRecorder": "Class TelemetryRecorder: High-throughput non-blocking SQLite WAL logger.",
        "engine.spatial.NavMesh": "Class NavMesh: 2D traversability grid and A* pathfinder.",
        "engine.api.contracts": "Module contracts: Version 1.0.0 SDK interfaces (BaseCartridge, BasePlayer, BaseEnemy, BaseHazard, BaseWeapon, BaseLevelSpec).",
    }

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "deliberation": {
                    "type": "string",
                    "description": "Epistemic rationale explaining why API documentation is requested."
                },
                "module_name": {
                    "type": "string",
                    "description": "Optional module name (e.g. 'engine.api.contracts', 'engine.telemetry.TelemetryRecorder')."
                },
                "class_name": {
                    "type": "string",
                    "description": "Optional class name filter (e.g. 'BaseCartridge', 'BaseLevelSpec')."
                }
            },
            "required": ["deliberation"]
        }

    def execute(self, arguments: Dict[str, Any], deliberation: str) -> Dict[str, Any]:
        module_name = arguments.get("module_name")
        class_name = arguments.get("class_name")

        # Compatibility hook for legacy showrunner queries
        if module_name and module_name in self.PINNED_SHOWRUNNER_DOCS and module_name != "engine.api.contracts":
            doc_text = self.PINNED_SHOWRUNNER_DOCS[module_name]
            return {
                "deliberation": deliberation or f"Retrieved pinned documentation for {module_name}.",
                "module": module_name,
                "docs": doc_text,
                "api_version": contracts.ENGINE_API_VERSION,
            }

        api_contracts: Dict[str, Any] = {
            "ENGINE_API_VERSION": contracts.ENGINE_API_VERSION,
            "BaseCartridge": {
                "description": "Abstract interface for modular game cartridges.",
                "methods": {
                    "get_level_spec(episode_number: int = 1) -> BaseLevelSpec": "Returns level geometry, spawn point, exit point, enemies, and hazards.",
                    "create_player(x: float, y: float, telemetry_recorder: Optional[Any] = None) -> BasePlayer": "Instantiates the player entity.",
                    "create_enemies(spec: BaseLevelSpec) -> List[BaseEnemy]": "Instantiates enemies configured in the level spec.",
                    "create_hazards(spec: BaseLevelSpec) -> List[BaseHazard]": "Instantiates hazards configured in the level spec.",
                }
            },
            "BaseLevelSpec": {
                "description": "Canonical specification defining episodic spatial geometry, entity spawns, and hazards.",
                "fields": {
                    "deliberation": "str = '' (Epistemic rationale)",
                    "level_id": "str",
                    "name": "str",
                    "width": "float = 1280.0",
                    "height": "float = 720.0",
                    "spawn_point": "Tuple[float, float] = (120.0, 360.0)",
                    "exit_point": "Tuple[float, float] = (1160.0, 360.0)",
                    "walls": "List[Tuple[float, float, float, float]] (x, y, w, h)",
                    "enemies": "List[Dict[str, Any]] (id, type, x, y)",
                    "hazards": "List[Dict[str, Any]] (id, type, x, y, ...)",
                }
            },
            "BasePlayer": {
                "description": "Player entity base contract with kinematic movement, dodging, and combat hooks.",
                "methods": {
                    "update(dt: float, input_state: InputState, ...)": "Advances player kinematics and weapons.",
                    "take_damage(packet: DamagePacket) -> bool": "Applies incoming damage.",
                }
            },
            "BaseEnemy": {
                "description": "Enemy entity base contract for autonomous AI behaviors (chaser, ranged, etc.).",
                "methods": {
                    "update(dt: float, target_pos: Vector2, ...)": "Executes movement, state machines, and attacks.",
                }
            },
            "BaseHazard": {
                "description": "Environmental hazard contract (spikes, floor acid, laser sweep).",
                "methods": {
                    "update(dt: float)": "Advances hazard cycle / sweep.",
                }
            },
            "BaseWeapon": {
                "description": "Weapon interface with cooldown, hitbox generation, and projectile emission.",
            }
        }

        if class_name and class_name in api_contracts:
            target_data: Dict[str, Any] = {class_name: api_contracts[class_name]}
        else:
            target_data = api_contracts

        formatted_docs = f"Autonomous Game Engine SDK Contracts (v{contracts.ENGINE_API_VERSION}):\n"
        for name, details in target_data.items():
            if isinstance(details, dict):
                formatted_docs += f"\n### {name}\n{details.get('description', '')}\n"
                if "methods" in details:
                    formatted_docs += "Methods:\n"
                    for m, desc in details["methods"].items():
                        formatted_docs += f"  - `{m}`: {desc}\n"

        return {
            "deliberation": deliberation or f"Retrieved SDK contracts for version {contracts.ENGINE_API_VERSION}.",
            "api_version": contracts.ENGINE_API_VERSION,
            "target": module_name or class_name or "engine.api.contracts",
            "contracts": target_data,
            "docs": formatted_docs,
        }
