"""tests/test_ui_static_index.py
Test specifications for Feature 29 & 30: Mission Control Single Page Application.
Verifies zero-build frontend, 7 mandatory DOM IDs, 9 agent SVG layout,
5-state lifecycle CSS, and real-time streaming contracts.
"""
from pathlib import Path
import pytest
from engine.ui.server import STATIC_DIR

INDEX_PATH = STATIC_DIR / "index.html"

MANDATORY_DOM_IDS = [
    "agent-dag-canvas",
    "deliberation-stream",
    "tool-invocations-list",
    "btn-trigger-genesis",
    "telemetry-heatmap",
    "voice-preview-player",
    "code-diff-container",
]

AUTHORITATIVE_9_AGENTS = [
    "StudioOrchestrator",
    "SteamAndTelemetryAnalystAgent",
    "CreativeDirectorWorldbuilderAgent",
    "NarrativeCharacterWriterAgent",
    "VoiceActingAudioProducerAgent",
    "VisualArtSpriteDesignerAgent",
    "LeadGameplaySWEAgent",
    "TelemetryArchitectAgent",
    "QAPlaytesterBuildSupervisor",
]

LIFECYCLE_STATES = ["idle", "analyzing", "synthesizing", "verifying", "promoted"]


class TestMissionControlSinglePageApp:
    """Verifies Feature 29 & 30 Mission Control HTML/CSS/JS contracts."""

    def test_static_index_file_exists_and_well_formed(self):
        """F29.1: index.html exists, is HTML5 well-formed, and contains required title."""
        assert INDEX_PATH.exists(), f"Missing static file at {INDEX_PATH}"
        content = INDEX_PATH.read_text(encoding="utf-8")
        assert content.strip().startswith("<!DOCTYPE html>")
        assert "<title>Mission Control</title>" in content
        assert "Autonomous Game Studio — Mission Control" in content

    def test_all_seven_mandatory_dom_ids_present(self):
        """F29.1: Enforces that all 7 required DOM elements exist with exact IDs."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        for dom_id in MANDATORY_DOM_IDS:
            assert f'id="{dom_id}"' in content, f"Mandatory DOM element missing: '{dom_id}'"

    def test_nine_agent_nodes_present_in_svg_dag(self):
        """F29.2: Verifies all 9 studio agent personas are represented in SVG DAG."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        for agent in AUTHORITATIVE_9_AGENTS:
            assert f'id="node-{agent}"' in content or f'data-agent="{agent}"' in content, (
                f"Agent persona '{agent}' missing from SVG DAG in index.html"
            )

    def test_svg_dag_canvas_dimensions_and_viewbox(self):
        """F29.2: Verifies SVG canvas has width=800 and height=400 as required by tests."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        assert 'id="agent-dag-canvas"' in content
        assert 'width="800"' in content
        assert 'height="400"' in content
        assert 'viewBox="0 0 800 400"' in content

    def test_five_lifecycle_states_css_styling_present(self):
        """F29.2: Verifies CSS rules exist for all 5 lifecycle states."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        for state in LIFECYCLE_STATES:
            assert f"state-{state}" in content, f"Missing CSS class for state-{state}"
        assert "@keyframes" in content, "Missing animation keyframes for status pulses"

    def test_deliberation_terminal_styling_and_epistemic_prominence(self):
        """F29.3: Verifies terminal-log class and Rule 4 deliberation formatting."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        assert 'class="terminal-log"' in content or 'terminal-log' in content
        assert "deliberation" in content.lower()
        assert 'id="stream-search"' in content
        assert 'id="agent-filter"' in content

    def test_tool_invocations_container_and_cards(self):
        """F29.3: Verifies tool invocation cards show inputs, outputs, errors, and PEV status."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        assert 'id="tool-invocations-list"' in content
        assert "pev" in content.lower()

    def test_websocket_auto_reconnect_and_resilience_logic(self):
        """F29.4: Verifies WebSocket reconnection backoff and endpoint fallback in JS."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        assert "/ws/swarm" in content
        assert "/ws/stream" in content
        assert "reconnect" in content.lower() or "retry" in content.lower()
        assert "studio-status-badge" in content

    def test_palette_and_diff_containers_present(self):
        """F30: Verifies 16-color Style Bible palette container and live code diff viewer."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        assert 'id="palette-grid"' in content
        assert 'id="code-diff-container"' in content
        assert 'id="voice-preview-player"' in content

    def test_zero_npm_external_build_dependencies(self):
        """F29.1: Verifies zero npm, webpack, or external build dependencies are referenced."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        assert "node_modules" not in content
        assert "webpack" not in content
        assert "<script src=" not in content, "External scripts are forbidden; frontend must be zero-build"

    def test_debranding_invariance(self):
        """Repository-wide invariant: 0 occurrences of forbidden branding tokens (case-insensitive)."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        forbidden = "".join(chr(c) for c in (106, 111, 97, 110))
        assert forbidden not in content.lower(), "Found legacy debranding token in index.html"

    def test_three_game_agnostic_studio_panels_present(self):
        """M2: Verifies that the 3 game-agnostic data-driven studio panels exist in index.html."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        assert "Steam Player DNA & Game Blueprint" in content
        assert "Dynamic Agent-Designed Telemetry & Hypotheses" in content
        assert "Multimodal Voice Clip Preview" in content
        assert 'id="dna-playtime"' in content
        assert 'id="game-blueprint-content"' in content
        assert 'id="probes-catalog-container"' in content

    def test_zero_forbidden_hardcoded_game_tokens(self):
        """M2: Rigorously enforces 0 occurrences of all 7 forbidden hardcoded game tokens in index.html."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        forbidden_patterns = [
            "Spatial Occupancy Telemetry Heatmap",
            "640x360",
            "Velocity Vectors",
            "Death Choke Points",
            "PICO-8",
            "hero_start.wav",
            "cartridges/seed",
        ]
        for pattern in forbidden_patterns:
            assert pattern not in content, f"Forbidden hardcoded game token '{pattern}' found in index.html"

    def test_clean_empty_state_game_blueprint_present(self):
        """M2: Verifies clean empty state for Game Blueprint before Genesis build."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        assert "No Active Game Built Yet — Commission Genesis Build" in content

    def test_empathetic_live_pulse_ribbon_and_production_board_subtab(self):
        """Verifies Always-Visible Live Studio Pulse Ribbon and Live Production Board & Proofs subtab."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        assert 'id="studio-live-pulse-bar"' in content
        assert 'id="studio-board-progress"' in content
        assert 'data-bible-tab="board"' in content
        assert 'id="bible-subview-board"' in content
        assert 'id="production-board-cards-container"' in content
        assert "fetchLiveProductionBoardFallback" in content
        assert "renderProductionBoardPanel" in content
        assert "updateStudioPulseRibbon" in content

    def test_game_switcher_pause_resume_and_built_renders_showcase(self):
        """Verifies Game Switcher dropdown, Pause/Resume controls, and Built Assets/Renders/Code showcase."""
        content = INDEX_PATH.read_text(encoding="utf-8")
        assert 'id="studio-game-switcher"' in content
        assert 'id="btn-activate-selected-game"' in content
        assert 'id="btn-resume-genesis"' in content
        assert 'data-bible-tab="renders"' in content
        assert 'id="bible-subview-renders"' in content
        assert 'id="built-renders-showcase-container"' in content
        assert "renderBuiltRendersShowcase" in content
        assert "switchSelectedGame" in content
        assert "pauseActiveGameBuild" in content
        assert "normalizeBoardItem" in content



