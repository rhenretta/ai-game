"""Tier 3 Showrunner Cognitive Swarm & Self-Healing Tests.

Verifies:
  1. Telemetry Analyst behavioral profiling, noise filtering, and kinetic classification.
  2. Cognitive Director Rationale-First update_spec.md generation with bounded +/-15% adaptation clamp.
  3. Local MCP Server data bridge tool endpoints (telemetry, spatial NavMesh, pinned API docs).
  4. Build Supervisor defect rejection and closed-loop self-healing repair.
  5. Dual synthetic profile behavioral and recommendation divergence.
"""

import json
import math
import pytest
from typing import Dict, List, Tuple, Any, Optional


# ============================================================================
# Reference Showrunner Studio Components (Matches PROJECT.md)
# ============================================================================

class BehavioralProfile:
    """Extracted behavioral metrics from raw SQLite telemetry."""
    def __init__(
        self,
        session_id: str,
        episode_id: str,
        total_events: int,
        filtered_events: int,
        mean_velocity: float,
        peak_velocity: float,
        dash_ratio: float,
        playstyle_tag: str,
        heatmap_grid: List[List[int]]
    ):
        self.session_id = session_id
        self.episode_id = episode_id
        self.total_events = total_events
        self.filtered_events = filtered_events
        self.mean_velocity = mean_velocity
        self.peak_velocity = peak_velocity
        self.dash_ratio = dash_ratio
        self.playstyle_tag = playstyle_tag
        self.heatmap_grid = heatmap_grid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "total_events": self.total_events,
            "filtered_events": self.filtered_events,
            "mean_velocity": round(self.mean_velocity, 2),
            "peak_velocity": round(self.peak_velocity, 2),
            "dash_ratio": round(self.dash_ratio, 3),
            "playstyle_tag": self.playstyle_tag
        }


class TelemetryAnalyst:
    """Parses raw SQLite event stream, discards stationary noise, and extracts BehavioralProfile."""
    def __init__(self, noise_velocity_threshold: float = 5.0):
        self.noise_threshold = noise_velocity_threshold

    def analyze(self, session_id: str, episode_id: str, events: List[Dict[str, Any]]) -> BehavioralProfile:
        total_events = len(events)
        filtered_events = []
        dashes = 0
        velocities = []
        
        # 8x8 discretized spatial occupancy heatmap
        heatmap = [[0 for _ in range(8)] for _ in range(8)]
        
        for ev in events:
            vx, vy = ev["vel"]
            v_mag = math.sqrt(vx * vx + vy * vy)
            
            # Discard stationary idle frames with near-zero velocity
            if v_mag < self.noise_threshold:
                continue
                
            filtered_events.append(ev)
            velocities.append(v_mag)
            
            if ev.get("event_type") == "DASH":
                dashes += 1
                
            # Discretize position into 8x8 heatmap (assuming 800x800 world space)
            px, py = ev["pos"]
            hx = min(7, max(0, int(px / 100.0)))
            hy = min(7, max(0, int(py / 100.0)))
            heatmap[hy][hx] += 1

        n_filtered = len(filtered_events)
        mean_v = sum(velocities) / n_filtered if n_filtered > 0 else 0.0
        peak_v = max(velocities) if velocities else 0.0
        dash_ratio = (dashes / n_filtered) if n_filtered > 0 else 0.0

        # Behavioral classification
        if dash_ratio >= 0.40 and mean_v >= 200.0:
            playstyle = "high_mobility_rusher"
        elif dash_ratio <= 0.15 and mean_v <= 150.0:
            playstyle = "methodical_cautious"
        else:
            playstyle = "balanced_skirmisher"

        return BehavioralProfile(
            session_id=session_id,
            episode_id=episode_id,
            total_events=total_events,
            filtered_events=n_filtered,
            mean_velocity=mean_v,
            peak_velocity=peak_v,
            dash_ratio=dash_ratio,
            playstyle_tag=playstyle,
            heatmap_grid=heatmap
        )


class CognitiveDirector:
    """Enforces Rationale-First Epistemic Architecture & bounded +/-15% adaptation clamp."""
    def __init__(self, max_delta_clamp: float = 0.15):
        self.max_delta = max_delta_clamp

    def generate_update_spec(self, profile: BehavioralProfile, base_difficulty: float = 1.0) -> Dict[str, Any]:
        # Formulate deliberation FIRST
        if profile.playstyle_tag == "high_mobility_rusher":
            deliberation = (
                f"Player demonstrated aggressive high-mobility engagement with mean velocity {profile.mean_velocity:.1f} px/s "
                f"and dash ratio {profile.dash_ratio:.1%}. To prevent trivialization of encounters, expanding arena dimensions, "
                f"introducing dynamic high-speed chasers, and clamping difficulty scaling within safety margins."
            )
            raw_delta = 0.35  # Desired increase
            recommended_features = ["expanded_combat_arenas", "high_speed_chasers", "momentum_rewards"]
        elif profile.playstyle_tag == "methodical_cautious":
            deliberation = (
                f"Player exhibited cautious, cover-centric engagement with mean velocity {profile.mean_velocity:.1f} px/s "
                f"and low dash frequency {profile.dash_ratio:.1%}. Providing tactical sightlines and environmental barricades "
                f"while carefully modulating ranged sniper enemy density within safety bounds."
            )
            raw_delta = -0.20  # Desired decrease
            recommended_features = ["tactical_cover_barricades", "ranged_sniper_sightlines", "hazard_timing_puzzles"]
        else:
            deliberation = "Player exhibited balanced skirmishing engagement; maintaining steady pacing."
            raw_delta = 0.05
            recommended_features = ["balanced_mob_mix"]

        # Enforce bounded adaptation clamp: delta in [-max_delta, +max_delta]
        clamped_delta = max(-self.max_delta, min(self.max_delta, raw_delta))
        new_difficulty = round(base_difficulty * (1.0 + clamped_delta), 3)

        # RATIONALE-FIRST INVARIANT: deliberation MUST BE the very first key
        spec_dict = {
            "deliberation": deliberation,
            "target_tier": "episodic_update",
            "player_profile_tag": profile.playstyle_tag,
            "difficulty": {
                "base": base_difficulty,
                "raw_delta": raw_delta,
                "clamped_delta": clamped_delta,
                "resulting_difficulty": new_difficulty
            },
            "recommended_features": recommended_features
        }
        return spec_dict

    def generate_update_spec_markdown(self, spec_dict: Dict[str, Any]) -> str:
        """Renders update_spec.md strictly starting with Rationale section."""
        md = (
            f"# Episodic Update Blueprint: Episode Next\n\n"
            f"## 1. Rationale & Deliberation\n"
            f"{spec_dict['deliberation']}\n\n"
            f"## 2. Mechanical Specifications\n"
            f"- **Target Profile**: `{spec_dict['player_profile_tag']}`\n"
            f"- **Difficulty Scale**: `{spec_dict['difficulty']['resulting_difficulty']}` "
            f"(Delta: `{spec_dict['difficulty']['clamped_delta']:+.1%}`)\n"
            f"- **Features**: {', '.join(spec_dict['recommended_features'])}\n"
        )
        return md


class LocalMCPServer:
    """Local Model Context Protocol (MCP) JSON-RPC 2.0 tool provider."""
    def __init__(self, telemetry_analyst: TelemetryAnalyst):
        self.analyst = telemetry_analyst
        self.pinned_api_docs = {
            "engine.core.GameLoop": "Class GameLoop: Manages 60 FPS fixed-timestep Cocoa loop on macOS.",
            "engine.telemetry.TelemetryRecorder": "Class TelemetryRecorder: High-throughput non-blocking SQLite WAL logger.",
            "engine.spatial.NavMesh": "Class NavMesh: 2D traversability grid and A* pathfinder."
        }

    def handle_tool_call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches standard MCP JSON-RPC 2.0 tool requests."""
        # Enforce that agent requests spend deliberation tokens first
        deliberation = params.get("deliberation", "")
        if not deliberation:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": "Missing mandatory deliberation parameter."},
                "id": params.get("id", 1)
            }

        tool_name = params.get("tool_name")
        args = params.get("arguments", {})

        if tool_name == "get_telemetry_profile":
            session_id = args.get("session_id", "default")
            episode_id = args.get("episode_id", "default")
            raw_events = args.get("raw_events", [])
            profile = self.analyst.analyze(session_id, episode_id, raw_events)
            return {
                "jsonrpc": "2.0",
                "result": profile.to_dict(),
                "id": params.get("id", 1)
            }

        elif tool_name == "query_spatial_map":
            grid = args.get("grid", [])
            start = tuple(args.get("start", (0, 0)))
            goal = tuple(args.get("goal", (0, 0)))
            from tests.test_tier2_playtest import astar_pathfind
            path = astar_pathfind(grid, start, goal)
            return {
                "jsonrpc": "2.0",
                "result": {
                    "is_traversable": path is not None,
                    "path_length": len(path) if path else 0
                },
                "id": params.get("id", 1)
            }

        elif tool_name == "get_engine_api_docs":
            module_name = args.get("module_name")
            doc = self.pinned_api_docs.get(module_name)
            if doc:
                return {"jsonrpc": "2.0", "result": {"module": module_name, "docs": doc}, "id": params.get("id", 1)}
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Module {module_name} not found."}, "id": params.get("id", 1)}

        return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Unknown tool {tool_name}"}, "id": params.get("id", 1)}


class BuildSupervisor:
    """Build Supervisor evaluating candidate updates, rejecting defects, and driving self-healing."""
    def __init__(self, tile_size: float = 32.0):
        from tests.test_tier2_playtest import SyntheticPlaytester
        self.playtester = SyntheticPlaytester(tile_size=tile_size)

    def evaluate_candidate(self, level_data: Dict[str, Any]) -> Dict[str, Any]:
        grid = level_data["grid"]
        spawn = level_data["spawn"]
        exit_tile = level_data["exit"]
        
        report = self.playtester.evaluate_level(grid, spawn, exit_tile)
        if not report["is_traversable"]:
            return {
                "deliberation": "Candidate episodic update failed traversability verification; critical path blocked.",
                "status": "REJECTED",
                "failure_type": "CriticalPathBlockedError",
                "diagnostic_trace": {
                    "spawn": spawn,
                    "exit": exit_tile,
                    "blocked_region": level_data.get("blocked_nodes", [])
                }
            }
        return {
            "deliberation": "Candidate episodic update successfully passed headless playtest; critical path verified.",
            "status": "PROMOTED",
            "failure_type": None,
            "diagnostic_trace": None
        }

    def repair_and_resubmit(self, level_data: Dict[str, Any], diagnostic_trace: Dict[str, Any]) -> Dict[str, Any]:
        """Self-healing loop: applies remediation based on diagnostic trace and re-evaluates."""
        repaired_grid = [row[:] for row in level_data["grid"]]
        # Unblock nodes identified in diagnostic trace
        for bx, by in diagnostic_trace.get("blocked_region", []):
            repaired_grid[by][bx] = 0  # Carve opening
            
        repaired_level = {
            "grid": repaired_grid,
            "spawn": level_data["spawn"],
            "exit": level_data["exit"]
        }
        return self.evaluate_candidate(repaired_level)


# ============================================================================
# Test Cases
# ============================================================================

class TestTelemetryAnalystProfiling:
    """Verifies event parsing, kinetic velocity extraction, and noise filtering."""

    def test_analyst_rusher_profiling(self, sample_rusher_telemetry_events):
        """Analyst classifies Aggressive Rusher with high velocity and high dash ratio."""
        analyst = TelemetryAnalyst(noise_velocity_threshold=5.0)
        profile = analyst.analyze("sess_01", "ep_01", sample_rusher_telemetry_events)
        
        assert profile.playstyle_tag == "high_mobility_rusher"
        assert profile.mean_velocity >= 200.0
        assert profile.dash_ratio >= 0.50
        assert profile.filtered_events == 20  # All events above noise threshold

    def test_analyst_cautious_profiling_and_noise_filtering(self, sample_cautious_telemetry_events):
        """Analyst filters out stationary noise and classifies Methodical Cautious."""
        analyst = TelemetryAnalyst(noise_velocity_threshold=5.0)
        profile = analyst.analyze("sess_02", "ep_01", sample_cautious_telemetry_events)
        
        assert profile.playstyle_tag == "methodical_cautious"
        assert profile.mean_velocity <= 150.0
        assert profile.dash_ratio <= 0.15
        # Out of 20 events, 10 stationary events (< 5 px/s) must be filtered out
        assert profile.filtered_events == 10
        assert profile.total_events == 20


class TestCognitiveDirectorSpecGeneration:
    """Verifies Rationale-First JSON/Markdown generation and bounded +/-15% adaptation clamp."""

    def test_director_rationale_first_spec_structure(self, sample_rusher_telemetry_events):
        analyst = TelemetryAnalyst()
        director = CognitiveDirector(max_delta_clamp=0.15)
        
        profile = analyst.analyze("sess_01", "ep_01", sample_rusher_telemetry_events)
        spec = director.generate_update_spec(profile, base_difficulty=1.0)
        
        # Invariant 1: Deliberation is the first key
        assert list(spec.keys())[0] == "deliberation"
        assert "high-mobility" in spec["deliberation"]
        assert "expanded_combat_arenas" in spec["recommended_features"]

    def test_director_bounded_adaptation_clamp(self):
        """Verifies delta is strictly clamped to +/-15%."""
        director = CognitiveDirector(max_delta_clamp=0.15)
        
        # Test Rusher (raw delta was +35% -> must clamp to +15%)
        rusher_profile = BehavioralProfile("s1", "e1", 100, 100, 260.0, 320.0, 0.65, "high_mobility_rusher", [])
        rusher_spec = director.generate_update_spec(rusher_profile, base_difficulty=1.0)
        assert rusher_spec["difficulty"]["raw_delta"] == 0.35
        assert rusher_spec["difficulty"]["clamped_delta"] == 0.15
        assert rusher_spec["difficulty"]["resulting_difficulty"] == 1.15

        # Test Cautious (raw delta was -20% -> must clamp to -15%)
        cautious_profile = BehavioralProfile("s2", "e1", 100, 50, 90.0, 120.0, 0.05, "methodical_cautious", [])
        cautious_spec = director.generate_update_spec(cautious_profile, base_difficulty=1.0)
        assert cautious_spec["difficulty"]["raw_delta"] == -0.20
        assert cautious_spec["difficulty"]["clamped_delta"] == -0.15
        assert cautious_spec["difficulty"]["resulting_difficulty"] == 0.85

    def test_director_markdown_starts_with_deliberation(self, sample_rusher_telemetry_events):
        analyst = TelemetryAnalyst()
        director = CognitiveDirector()
        profile = analyst.analyze("s1", "e1", sample_rusher_telemetry_events)
        spec = director.generate_update_spec(profile)
        
        md = director.generate_update_spec_markdown(spec)
        lines = [line.strip() for line in md.strip().split("\n") if line.strip()]
        
        # First section must be Rationale & Deliberation
        assert "Rationale & Deliberation" in lines[1]


class TestLocalMCPServerDataBridge:
    """Verifies local MCP JSON-RPC 2.0 tool endpoints."""

    def test_mcp_get_telemetry_profile_tool(self, sample_rusher_telemetry_events):
        analyst = TelemetryAnalyst()
        mcp = LocalMCPServer(analyst)
        
        response = mcp.handle_tool_call(
            method="tools/call",
            params={
                "deliberation": "Fetching player kinetic profile for level layout planning.",
                "tool_name": "get_telemetry_profile",
                "arguments": {
                    "session_id": "sess_test",
                    "episode_id": "ep_test",
                    "raw_events": sample_rusher_telemetry_events
                },
                "id": 42
            }
        )
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 42
        assert response["result"]["playstyle_tag"] == "high_mobility_rusher"

    def test_mcp_query_spatial_map_tool(self, sample_level_grid_16x16):
        analyst = TelemetryAnalyst()
        mcp = LocalMCPServer(analyst)
        data = sample_level_grid_16x16
        
        response = mcp.handle_tool_call(
            method="tools/call",
            params={
                "deliberation": "Checking spatial traversability between spawn and exit portal.",
                "tool_name": "query_spatial_map",
                "arguments": {
                    "grid": data["grid"],
                    "start": data["spawn"],
                    "goal": data["exit"]
                },
                "id": 43
            }
        )
        assert response["jsonrpc"] == "2.0"
        assert response["result"]["is_traversable"] is True
        assert response["result"]["path_length"] > 0

    def test_mcp_get_engine_api_docs_tool(self):
        analyst = TelemetryAnalyst()
        mcp = LocalMCPServer(analyst)
        
        response = mcp.handle_tool_call(
            method="tools/call",
            params={
                "deliberation": "Fetching pinned API contract for TelemetryRecorder to prevent hallucinated calls.",
                "tool_name": "get_engine_api_docs",
                "arguments": {"module_name": "engine.telemetry.TelemetryRecorder"},
                "id": 44
            }
        )
        assert response["jsonrpc"] == "2.0"
        assert "SQLite WAL logger" in response["result"]["docs"]

    def test_mcp_missing_deliberation_rejected(self):
        """MCP tools strictly require deliberation in tool call params."""
        analyst = TelemetryAnalyst()
        mcp = LocalMCPServer(analyst)
        
        response = mcp.handle_tool_call(
            method="tools/call",
            params={"tool_name": "get_engine_api_docs", "arguments": {}}
        )
        assert "error" in response
        assert response["error"]["code"] == -32602


class TestBuildSupervisorDefectRejectionAndSelfHealing:
    """Verifies automated rejection gate and closed-loop self-healing repair."""

    def test_supervisor_promotes_valid_level(self, sample_level_grid_16x16):
        supervisor = BuildSupervisor(tile_size=32.0)
        result = supervisor.evaluate_candidate(sample_level_grid_16x16)
        
        assert result["status"] == "PROMOTED"
        assert result["failure_type"] is None
        assert list(result.keys())[0] == "deliberation"

    def test_supervisor_rejects_blocked_level_with_diagnostics(self, blocked_level_grid_16x16):
        supervisor = BuildSupervisor(tile_size=32.0)
        result = supervisor.evaluate_candidate(blocked_level_grid_16x16)
        
        assert result["status"] == "REJECTED"
        assert result["failure_type"] == "CriticalPathBlockedError"
        assert result["diagnostic_trace"]["spawn"] == (1, 1)
        assert result["diagnostic_trace"]["exit"] == (14, 14)

    def test_supervisor_closed_loop_self_healing_repair(self, blocked_level_grid_16x16):
        """Complete closed-loop: Defect injected -> Rejected -> Repaired -> Promoted."""
        supervisor = BuildSupervisor(tile_size=32.0)
        
        # Step 1: Evaluate defective level
        rejection_result = supervisor.evaluate_candidate(blocked_level_grid_16x16)
        assert rejection_result["status"] == "REJECTED"
        
        # Step 2: Feed diagnostic trace to repair
        repair_result = supervisor.repair_and_resubmit(
            blocked_level_grid_16x16,
            rejection_result["diagnostic_trace"]
        )
        
        # Step 3: Assert repaired level is now successfully promoted
        assert repair_result["status"] == "PROMOTED"
        assert repair_result["failure_type"] is None


class TestDualSyntheticProfileDivergence:
    """Verifies that Rusher vs Cautious profiles diverge in telemetry and adaptation specs."""

    def test_behavioral_and_spec_divergence(
        self,
        sample_rusher_telemetry_events,
        sample_cautious_telemetry_events
    ):
        analyst = TelemetryAnalyst()
        director = CognitiveDirector(max_delta_clamp=0.15)
        
        rusher_prof = analyst.analyze("s_rusher", "ep_01", sample_rusher_telemetry_events)
        cautious_prof = analyst.analyze("s_cautious", "ep_01", sample_cautious_telemetry_events)
        
        # Behavioral divergence
        assert rusher_prof.playstyle_tag != cautious_prof.playstyle_tag
        assert rusher_prof.mean_velocity > (cautious_prof.mean_velocity * 1.5)
        assert rusher_prof.dash_ratio > (cautious_prof.dash_ratio * 3.0)
        
        # Director recommendation divergence
        rusher_spec = director.generate_update_spec(rusher_prof)
        cautious_spec = director.generate_update_spec(cautious_prof)
        
        assert rusher_spec["difficulty"]["clamped_delta"] > 0.0
        assert cautious_spec["difficulty"]["clamped_delta"] < 0.0
        assert rusher_spec["recommended_features"] != cautious_spec["recommended_features"]
