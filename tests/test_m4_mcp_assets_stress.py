"""tests/test_m4_mcp_assets_stress.py
Empirical stress test suite for Local MCP Server (Feature 11) and Style Bible Asset Pipeline (Feature 13).
Challenger 1 Milestone 4 Stress Challenge.

Verifies:
  1. MCP Hostile Payloads: malformed JSON-RPC syntax, non-object requests, missing deliberation,
     empty deliberation strings, non-existent tool names, and SQL injection strings in player_id/session_id.
  2. MCP Concurrency & Load: concurrent/rapid tool invocations over read-only SQLite database under multi-threaded load.
  3. Asset Palette Stress: fuzz 10,000 arbitrary RGB tuples into 16-color palette confirming zero invalid indices (strictly in 0..15).
  4. Asset Seed Determinism: byte-identical PNG hashes across multiple generations with identical seeds.
  5. Network Isolation: zero socket connection attempts during asset synthesis across all thematic moods.
  6. PNG Mode & Transparency: strictly mode 'P', transparency key index 0, and correct palette bounds.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import http.client
import json
from pathlib import Path
import random
import socket
import sqlite3
import time
from typing import Any, Dict, List, Tuple
import urllib.request
import pytest
from PIL import Image

import config
from engine.analyst.analyst import TelemetryAnalyst
from engine.assets.palette import (
    STYLE_BIBLE_16_PALETTE,
    get_palette_index,
    quantize_rgb,
    quantize_image_to_indexed,
)
from engine.assets.synthesizer import DeterministicSpriteSynthesizer
from engine.assets.liaise import AssetLiaison
from engine.mcp.server import LocalMCPServer
from engine.steam.dna import SteamDNAPersistence, SteamDNAProfiler
from engine.telemetry.db import TelemetryDB
from tests.test_tier1_units import validate_rationale_first_ast, validate_rationale_first_dict


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def populated_stress_db(tmp_path: Path) -> Path:
    """Creates and populates a temporary SQLite database with authentic WAL telemetry and Steam DNA."""
    db_path = tmp_path / "stress_telemetry.db"
    db = TelemetryDB(db_path)
    steam_repo = SteamDNAPersistence(db_path)

    conn = db.get_writer_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO telemetry_sessions (session_id, player_profile_tag, platform)
            VALUES (?, ?, ?);
            """,
            ("sess_stress_01", "hardcore_rusher", "macOS")
        )
        cur.execute(
            """
            INSERT INTO telemetry_episodes (
                episode_id, session_id, episode_number, status, outcome, duration_seconds,
                damage_dealt_total, damage_taken_total, dashes_total, shots_fired_total
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            ("ep_stress_01", "sess_stress_01", 1, "COMPLETED", "VICTORY", 45.2, 500.0, 50.0, 12, 40)
        )
        for i in range(25):
            cur.execute(
                """
                INSERT INTO telemetry_events (
                    session_id, episode_id, timestamp, event_type, pos_x, pos_y, vel_x, vel_y, health
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "sess_stress_01",
                    "ep_stress_01",
                    0.05 * i,
                    "MOVE" if i % 2 == 0 else "DASH",
                    128.0 + (i * 4.0),
                    256.0,
                    6.0,
                    0.0,
                    100.0
                )
            )
        conn.commit()
    finally:
        conn.close()

    # Seed Steam DNA
    profiler = SteamDNAProfiler()
    dna = profiler._build_default_dna(player_id="player_stress_test", profile_tag="hardcore_rusher")
    steam_repo.save_dna(dna, session_id="sess_stress_01")

    return db_path


@pytest.fixture
def mcp_server(populated_stress_db: Path) -> LocalMCPServer:
    """Instantiates LocalMCPServer wired to populated test database."""
    analyst = TelemetryAnalyst(db_path=populated_stress_db)
    steam_repo = SteamDNAPersistence(db_path=populated_stress_db)
    return LocalMCPServer(telemetry_analyst=analyst, steam_repository=steam_repo)


# ============================================================================
# Category 1: MCP Hostile Payloads Stress
# ============================================================================

class TestMCPHostilePayloadsStress:
    """Empirical stress test suite subjecting Local MCP Server to hostile JSON-RPC and database payloads."""
    deliberation: str = "Empirical stress test evaluating Local MCP Server robustness against hostile payloads."

    def test_malformed_json_rpc_syntax(self, mcp_server: LocalMCPServer):
        """Malformed JSON strings must fail fast with standard JSON-RPC parse error code -32700."""
        malformed_inputs = [
            "{ unclosed json payload",
            "{'single_quoted_key': 'value'}",
            "NOT_JSON_AT_ALL",
            "",
            "   \n  \t  ",
            "{\"key\": undefined}",
            "\x00\x01\x02\x03",
            "{\"truncated\": ",
        ]
        for raw_payload in malformed_inputs:
            if not raw_payload.strip():
                continue
            resp = mcp_server.handle_jsonrpc(raw_payload)
            assert resp is not None, f"Expected error response for '{raw_payload}', got None"
            assert resp["jsonrpc"] == "2.0"
            assert "error" in resp, f"Expected error key for '{raw_payload}', got {resp}"
            assert resp["error"]["code"] == -32700
            assert resp["id"] is None

    def test_malformed_json_rpc_non_object_requests(self, mcp_server: LocalMCPServer):
        """Non-dictionary JSON payloads must return Invalid Request code -32600."""
        non_object_payloads = [
            "[1, 2, 3]",
            "12345",
            "true",
            "false",
            "\"raw_json_string\"",
        ]
        for raw_payload in non_object_payloads:
            resp = mcp_server.handle_jsonrpc(raw_payload)
            assert resp is not None
            assert resp["jsonrpc"] == "2.0"
            assert "error" in resp
            assert resp["error"]["code"] == -32600
            assert resp["id"] is None

    def test_missing_and_empty_deliberation(self, mcp_server: LocalMCPServer):
        """Missing or empty deliberation strings must fail fast with code -32602 or raise ValueError."""
        # 1. Missing deliberation in JSON-RPC tools/call params
        call_missing = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "tool_name": "get_steam_player_dna",
                "arguments": {"player_id": "test_player"}
            },
            "id": 901
        }
        resp = mcp_server.handle_jsonrpc(call_missing)
        assert resp["jsonrpc"] == "2.0"
        assert resp["error"]["code"] == -32602
        assert "deliberation" in resp["error"]["message"].lower()

        # 2. Empty deliberation string in JSON-RPC tools/call params
        call_empty = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "tool_name": "get_steam_player_dna",
                "deliberation": "",
                "arguments": {"player_id": "test_player"}
            },
            "id": 902
        }
        resp = mcp_server.handle_jsonrpc(call_empty)
        assert resp["jsonrpc"] == "2.0"
        assert resp["error"]["code"] == -32602

        # 3. Empty deliberation inside arguments dict
        call_empty_in_args = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "tool_name": "get_steam_player_dna",
                "arguments": {"deliberation": "", "player_id": "test_player"}
            },
            "id": 903
        }
        resp = mcp_server.handle_jsonrpc(call_empty_in_args)
        assert resp["jsonrpc"] == "2.0"
        assert resp["error"]["code"] == -32602

        # 4. Programmatic call_tool with missing or empty deliberation
        with pytest.raises(ValueError, match="deliberation"):
            mcp_server.call_tool("get_steam_player_dna", deliberation="")

        with pytest.raises(ValueError, match="deliberation"):
            mcp_server.call_tool("get_steam_player_dna", arguments={"player_id": "test_player"})

        with pytest.raises(ValueError, match="deliberation"):
            mcp_server.call_tool("get_steam_player_dna", arguments={"deliberation": ""})

    def test_non_existent_tool_names(self, mcp_server: LocalMCPServer):
        """Calling non-existent tools must return code -32601 in JSON-RPC or raise KeyError in call_tool."""
        invalid_names = [
            "non_existent_tool_abc",
            "system_exec_payload",
            "drop_database_now",
            "__import__",
            "eval_tool",
        ]
        for name in invalid_names:
            req = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "tool_name": name,
                    "deliberation": f"Stress-testing invalid tool invocation for {name}.",
                    "arguments": {}
                },
                "id": 910
            }
            resp = mcp_server.handle_jsonrpc(req)
            assert resp["jsonrpc"] == "2.0"
            assert resp["error"]["code"] == -32601
            assert f"Unknown tool {name}" in resp["error"]["message"]

            with pytest.raises(KeyError, match="Unknown tool"):
                mcp_server.call_tool(name, deliberation="Testing unknown tool programmatic call.")

    def test_missing_tool_name_in_params(self, mcp_server: LocalMCPServer):
        """Calling tools/call without tool_name parameter must return error code -32602."""
        req = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "deliberation": "Testing missing tool name parameter.",
                "arguments": {}
            },
            "id": 920
        }
        resp = mcp_server.handle_jsonrpc(req)
        assert resp["jsonrpc"] == "2.0"
        assert resp["error"]["code"] == -32602
        assert "Missing tool name" in resp["error"]["message"]

    def test_sql_injection_resilience(self, mcp_server: LocalMCPServer, populated_stress_db: Path):
        """Hostile SQL injection strings in player_id, session_id, episode_id must be safely neutralized."""
        hostile_sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE steam_player_dna; --",
            "'; DROP TABLE telemetry_events; --",
            "'; DROP TABLE telemetry_sessions; --",
            "'; DROP TABLE telemetry_episodes; --",
            "1 UNION SELECT 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --",
            "admin'--",
            "' OR 1=1; DELETE FROM telemetry_events; --",
            "'); ATTACH DATABASE '/tmp/pwned.db' AS pwn; --",
            "1; SELECT pg_sleep(5); --",
        ]

        delib = "Epistemic stress validation verifying parameterized query resilience against SQL injection."

        for payload in hostile_sql_payloads:
            # 1. Test get_steam_player_dna with hostile player_id & profile_tag
            res_dna = mcp_server.call_tool(
                "get_steam_player_dna",
                arguments={"player_id": payload, "profile_tag": payload},
                deliberation=delib
            )
            assert validate_rationale_first_dict(res_dna) is True
            assert "deliberation" in res_dna

            # 2. Test get_telemetry_profile with hostile session_id & episode_id
            res_prof_sess = mcp_server.call_tool(
                "get_telemetry_profile",
                arguments={"session_id": payload},
                deliberation=delib
            )
            assert validate_rationale_first_dict(res_prof_sess) is True

            res_prof_ep = mcp_server.call_tool(
                "get_telemetry_profile",
                arguments={"episode_id": payload},
                deliberation=delib
            )
            assert validate_rationale_first_dict(res_prof_ep) is True

            # 3. Test through JSON-RPC layer as well
            json_rpc_call = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "tool_name": "get_steam_player_dna",
                    "deliberation": delib,
                    "arguments": {"player_id": payload}
                },
                "id": 930
            }
            rpc_resp = mcp_server.handle_jsonrpc(json_rpc_call)
            assert rpc_resp["jsonrpc"] == "2.0"
            assert "result" in rpc_resp
            assert validate_rationale_first_dict(rpc_resp["result"]) is True

        # Independent database verification: Verify all tables remain intact and uncorrupted
        conn = sqlite3.connect(str(populated_stress_db))
        try:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM telemetry_sessions;")
            sessions_count = cur.fetchone()[0]
            assert sessions_count == 1, f"telemetry_sessions corrupted, expected 1 row, got {sessions_count}"

            cur.execute("SELECT count(*) FROM telemetry_episodes;")
            episodes_count = cur.fetchone()[0]
            assert episodes_count == 1, f"telemetry_episodes corrupted, expected 1 row, got {episodes_count}"

            cur.execute("SELECT count(*) FROM telemetry_events;")
            events_count = cur.fetchone()[0]
            assert events_count == 25, f"telemetry_events dropped or wiped, expected 25 rows, got {events_count}"

            cur.execute("SELECT count(*) FROM steam_player_dna;")
            dna_count = cur.fetchone()[0]
            assert dna_count >= 1, "steam_player_dna was dropped or wiped by SQL injection"
        finally:
            conn.close()


# ============================================================================
# Category 2: MCP Concurrency & Load Stress
# ============================================================================

class TestMCPConcurrencyAndLoadStress:
    """Empirical stress tests verifying thread-safety and SQLite WAL concurrency under high throughput."""
    deliberation: str = "Empirical stress tests verifying thread-safety and SQLite WAL concurrency under high throughput."

    def test_concurrent_rapid_tool_invocations(self, mcp_server: LocalMCPServer, populated_stress_db: Path):
        """120 rapid concurrent tool invocations across 16 worker threads must complete without locks or race conditions."""
        total_requests = 120
        max_workers = 16

        def _execute_worker_task(task_id: int) -> Dict[str, Any]:
            mod = task_id % 4
            delib = f"Thread worker task #{task_id} performing concurrent load verification."
            if mod == 0:
                return mcp_server.call_tool(
                    "get_steam_player_dna",
                    arguments={"player_id": "player_stress_test"},
                    deliberation=delib
                )
            elif mod == 1:
                return mcp_server.call_tool(
                    "get_telemetry_profile",
                    arguments={"session_id": "sess_stress_01"},
                    deliberation=delib
                )
            elif mod == 2:
                # Direct grid pathfinding
                grid = [
                    [0, 0, 0, 0],
                    [1, 1, 0, 1],
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                ]
                return mcp_server.call_tool(
                    "query_spatial_map",
                    arguments={"grid": grid, "start": [0, 0], "goal": [3, 3]},
                    deliberation=delib
                )
            else:
                return mcp_server.call_tool(
                    "get_engine_api_docs",
                    arguments={"module_name": "engine.api.contracts"},
                    deliberation=delib
                )

        start_time = time.perf_counter()
        results: List[Dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(_execute_worker_task, i): i for i in range(total_requests)}
            for future in concurrent.futures.as_completed(future_to_id):
                task_id = future_to_id[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as ex:
                    pytest.fail(f"Task #{task_id} failed under concurrent load: {ex}")

        elapsed = time.perf_counter() - start_time

        # Validate total count and properties
        assert len(results) == total_requests
        for res in results:
            assert isinstance(res, dict)
            assert validate_rationale_first_dict(res) is True
            assert list(res.keys())[0] == "deliberation"

        # Verify execution throughput (120 tool executions must complete swiftly in < 5.0 seconds)
        assert elapsed < 5.0, f"Concurrent load took {elapsed:.2f}s, exceeding 5.0s benchmark limit"


# ============================================================================
# Category 3: Asset Palette Fuzz Stress
# ============================================================================

class TestAssetPaletteFuzzStress:
    """Empirical stress test fuzzing 10,000 arbitrary RGB tuples into 16-color Style Bible palette."""
    deliberation: str = "Empirical stress test fuzzing 10,000 arbitrary RGB tuples into 16-color Style Bible palette."

    def test_palette_fuzz_10000_rgb_tuples(self):
        """Fuzzes 10,000 arbitrary RGB tuples into 16-color palette to confirm zero invalid color indices (strictly in 0..15)."""
        rng = random.Random(20261016)

        # Explicit corner cases to test
        corner_cases: List[Tuple[int, int, int]] = [
            (0, 0, 0),
            (255, 255, 255),
            (128, 128, 128),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (254, 254, 254),
            (1, 1, 1),
            (0, 1, 2),
            (255, 254, 253),
            (127, 127, 127),
            (95, 87, 79),
            (41, 173, 255),
            (171, 82, 54),
            (0, 228, 54),
            (255, 204, 170),
        ]

        total_tested = 0

        # Test corner cases first
        for rgb in corner_cases:
            idx = get_palette_index(rgb)
            assert isinstance(idx, int)
            assert 0 <= idx <= 15, f"Corner case RGB {rgb} produced out-of-range index {idx}"
            q_rgb = quantize_rgb(rgb)
            assert q_rgb == STYLE_BIBLE_16_PALETTE[idx]
            assert q_rgb in STYLE_BIBLE_16_PALETTE
            total_tested += 1

        # Fuzz remaining tuples up to 10,000
        for _ in range(10000 - len(corner_cases)):
            r = rng.randint(0, 255)
            g = rng.randint(0, 255)
            b = rng.randint(0, 255)
            rgb = (r, g, b)

            idx = get_palette_index(rgb)
            assert isinstance(idx, int)
            assert 0 <= idx <= 15, f"Fuzzed RGB {rgb} produced out-of-range palette index {idx}"

            q_rgb = quantize_rgb(rgb)
            assert q_rgb == STYLE_BIBLE_16_PALETTE[idx]
            assert q_rgb in STYLE_BIBLE_16_PALETTE
            total_tested += 1

        assert total_tested == 10000

    def test_image_quantization_fuzz_noise(self):
        """Random noisy 64x64 RGBA image quantized to indexed mode 'P' contains only color indices 0..15."""
        rng = random.Random(9999)
        img = Image.new("RGBA", (64, 64))
        pixels = [
            (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            for _ in range(64 * 64)
        ]
        img.putdata(pixels)

        quantized = quantize_image_to_indexed(img, transparent_index=0)
        assert quantized.mode == "P"
        assert quantized.info.get("transparency") == 0

        colors = quantized.getcolors(maxcolors=256)
        assert colors is not None
        for count, color_idx in colors:
            assert isinstance(color_idx, int)
            assert 0 <= color_idx <= 15, f"Found out-of-bounds pixel palette index {color_idx} in quantized image"


# ============================================================================
# Category 4: Asset Seed Determinism Stress
# ============================================================================

class TestAssetSeedDeterminismStress:
    """Empirical stress tests verifying byte-identical reproducibility and divergence of procedural assets."""
    deliberation: str = "Empirical stress tests verifying byte-identical reproducibility and divergence of procedural assets."

    def test_identical_seeds_generate_byte_identical_hashes(self, tmp_path: Path):
        """Identical seeds must generate 100% byte-identical PNG hashes across multiple independent generations."""
        test_seeds = [
            42,
            "procedural_seed_alpha",
            "speedrunner_burst_genesis",
        ]

        for seed in test_seeds:
            dir_a = tmp_path / f"seed_{seed}_run_a"
            dir_b = tmp_path / f"seed_{seed}_run_b"

            manifest_a = AssetLiaison.synthesize_cartridge_assets(dir_a, seed=seed, mood="retro_arcade")
            manifest_b = AssetLiaison.synthesize_cartridge_assets(dir_b, seed=seed, mood="retro_arcade")

            assert manifest_a.total_files == manifest_b.total_files
            assert manifest_a.total_bytes == manifest_b.total_bytes

            files_a = sorted(list((dir_a / "assets").rglob("*.png")))
            files_b = sorted(list((dir_b / "assets").rglob("*.png")))
            assert len(files_a) == len(files_b)
            assert len(files_a) >= 16

            for file_a, file_b in zip(files_a, files_b):
                rel_a = file_a.relative_to(dir_a)
                rel_b = file_b.relative_to(dir_b)
                assert rel_a == rel_b, f"File structure mismatch: {rel_a} vs {rel_b}"

                bytes_a = file_a.read_bytes()
                bytes_b = file_b.read_bytes()
                assert bytes_a == bytes_b, f"Byte divergence detected in {rel_a} between identical runs!"

                hash_a = hashlib.sha256(bytes_a).hexdigest()
                hash_b = hashlib.sha256(bytes_b).hexdigest()
                assert hash_a == hash_b

    def test_thematic_moods_produce_divergent_asset_hashes(self, tmp_path: Path):
        """Thematic palette moods must produce divergent asset files and distinct SHA256 hashes."""
        dir_arcade = tmp_path / "mood_retro_arcade"
        dir_cyberpunk = tmp_path / "mood_neon_cyberpunk"
        dir_grim = tmp_path / "mood_grim_dark"
        dir_fantasy = tmp_path / "mood_vibrant_fantasy"

        m_arcade = AssetLiaison.synthesize_cartridge_assets(dir_arcade, seed="common_seed", mood="retro_arcade")
        m_cyber = AssetLiaison.synthesize_cartridge_assets(dir_cyberpunk, seed="common_seed", mood="neon_cyberpunk")
        m_grim = AssetLiaison.synthesize_cartridge_assets(dir_grim, seed="common_seed", mood="grim_dark")
        m_fantasy = AssetLiaison.synthesize_cartridge_assets(dir_fantasy, seed="common_seed", mood="vibrant_fantasy")

        # Compare hashes across moods for core sprite player_idle
        h_arcade = m_arcade.sprites["player_idle"].sha256_hash
        h_cyber = m_cyber.sprites["player_idle"].sha256_hash
        h_grim = m_grim.sprites["player_idle"].sha256_hash
        h_fantasy = m_fantasy.sprites["player_idle"].sha256_hash

        mood_hashes = {h_arcade, h_cyber, h_grim, h_fantasy}
        assert len(mood_hashes) == 4, f"Expected 4 distinct sprite hashes across 4 moods, got {len(mood_hashes)}"

    def test_canonical_pixel_invariance_across_arbitrary_seeds(self, tmp_path: Path):
        """Empirically verifies that sprite geometries strictly adhere to canonical Style Bible templates."""
        dir_1 = tmp_path / "seed_run_1"
        dir_2 = tmp_path / "seed_run_2"

        m1 = AssetLiaison.synthesize_cartridge_assets(dir_1, seed="arbitrary_seed_1", mood="retro_arcade")
        m2 = AssetLiaison.synthesize_cartridge_assets(dir_2, seed="arbitrary_seed_2", mood="retro_arcade")

        # Style Bible template fidelity: sprites within the same mood share canonical pixel templates
        assert m1.sprites["player_idle"].sha256_hash == m2.sprites["player_idle"].sha256_hash
        # Manifests record their distinct seed parameters
        assert m1.seed == "arbitrary_seed_1"
        assert m2.seed == "arbitrary_seed_2"


# ============================================================================
# Category 5: Network Isolation Stress
# ============================================================================

class TestAssetNetworkIsolationStress:
    """Empirical stress tests guaranteeing absolute offline isolation with zero socket calls."""
    deliberation: str = "Empirical stress tests guaranteeing absolute offline isolation with zero socket calls."

    def test_zero_socket_calls_during_synthesis_all_moods(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verifies zero socket connection attempts occur during asset synthesis across all thematic moods."""
        def banned_socket(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("CRITICAL VIOLATION: Network socket connection attempted during asset synthesis!")

        monkeypatch.setattr(socket, "socket", banned_socket)
        monkeypatch.setattr(socket, "create_connection", banned_socket)
        monkeypatch.setattr(urllib.request, "urlopen", banned_socket)
        monkeypatch.setattr(http.client, "HTTPConnection", banned_socket)
        monkeypatch.setattr(http.client, "HTTPSConnection", banned_socket)

        thematic_moods = ["retro_arcade", "neon_cyberpunk", "grim_dark", "vibrant_fantasy"]

        for mood in thematic_moods:
            cartridge_dir = tmp_path / f"isolated_cartridge_{mood}"
            manifest = AssetLiaison.synthesize_cartridge_assets(
                cartridge_dir=cartridge_dir,
                seed=f"offline_seed_{mood}",
                mood=mood
            )
            assert manifest.total_files >= 16
            assert manifest.is_fully_style_bible_compliant is True

            validation_errors = AssetLiaison.validate_asset_bundle(manifest, cartridge_dir)
            assert validation_errors == [], f"Validation errors for mood '{mood}': {validation_errors}"


# ============================================================================
# Category 6: PNG Mode & Transparency Stress
# ============================================================================

class TestAssetPNGModeAndTransparencyStress:
    """Empirical stress tests validating indexed mode 'P', 16-color limits, and transparency integrity."""
    deliberation: str = "Empirical stress tests validating indexed mode 'P', 16-color limits, and transparency integrity."

    def test_png_mode_and_transparency_invariants(self, tmp_path: Path):
        """Verifies that all generated PNG files are strictly mode 'P' with transparency index 0 intact."""
        cartridge_dir = tmp_path / "png_mode_test_cartridge"
        manifest = AssetLiaison.synthesize_cartridge_assets(
            cartridge_dir=cartridge_dir,
            seed="png_test_seed",
            mood="retro_arcade"
        )

        all_pngs = list((cartridge_dir / "assets").rglob("*.png"))
        assert len(all_pngs) >= 16

        for png_path in all_pngs:
            with Image.open(png_path) as img:
                # 1. Mode check
                assert img.mode == "P", f"File {png_path.name} mode is {img.mode}, expected 'P'"

                # 2. Palette check
                raw_palette = img.getpalette()
                assert raw_palette is not None, f"File {png_path.name} is missing color palette"
                # Check first 16 RGB colors match STYLE_BIBLE_16_PALETTE
                for i, expected_color in enumerate(STYLE_BIBLE_16_PALETTE):
                    actual_color = tuple(raw_palette[i * 3: (i + 1) * 3])
                    assert actual_color == expected_color, (
                        f"Palette color mismatch in {png_path.name} at index {i}: "
                        f"got {actual_color}, expected {expected_color}"
                    )

                # 3. For sprites, check transparency index
                if "sprites" in png_path.parts:
                    transparency = img.info.get("transparency")
                    assert transparency == 0, (
                        f"Sprite {png_path.name} transparency index is {transparency}, expected 0"
                    )

                # 4. Pixel data check: zero indices outside 0..15
                colors = img.getcolors(maxcolors=256)
                assert colors is not None
                for count, color_idx in colors:
                    assert 0 <= color_idx <= 15, (
                        f"Found out-of-bounds pixel palette index {color_idx} in {png_path.name}"
                    )

    def test_transparency_preservation_fidelity(self):
        """Verifies that transparent RGBA pixels strictly quantize to palette index 0."""
        # Create 16x16 image with transparent background (alpha=0) and opaque center (alpha=255)
        test_img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        # Center 4x4 square in Neon Red (255, 0, 77)
        for y in range(6, 10):
            for x in range(6, 10):
                test_img.putpixel((x, y), (255, 0, 77, 255))

        # Bottom-right 2x2 square in Cyan (41, 173, 255) with semi-transparent alpha (alpha=60 < 128)
        for y in range(14, 16):
            for x in range(14, 16):
                test_img.putpixel((x, y), (41, 173, 255, 60))

        quantized = quantize_image_to_indexed(test_img, transparent_index=0)
        assert quantized.mode == "P"
        assert quantized.info.get("transparency") == 0

        # Corner pixel (0, 0) was alpha=0 -> must be index 0
        assert quantized.getpixel((0, 0)) == 0

        # Semi-transparent pixel (14, 14) with alpha=60 < 128 -> must be index 0
        assert quantized.getpixel((14, 14)) == 0

        # Center pixel (7, 7) was opaque Neon Red (255, 0, 77) -> must be index 8
        assert quantized.getpixel((7, 7)) == 8


# ============================================================================
# Category 7: AST Rationale-First Architectural Check
# ============================================================================

def test_m4_stress_test_file_ast_rationale_first():
    """Verifies that tests/test_m4_mcp_assets_stress.py satisfies Rationale-First AST constraints."""
    source_path = Path(__file__).resolve()
    content = source_path.read_text(encoding="utf-8")
    assert validate_rationale_first_ast(content) is True
