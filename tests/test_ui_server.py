"""tests/test_ui_server.py
Comprehensive test suite for Feature 28 & 30: Observability UI Server, REST API,
Multimodal Asset Streaming, Telemetry Heatmaps, and WebSocket Hub.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict
import uuid

import pytest
from starlette.testclient import TestClient

import config
from engine.ui.models import (
    ActiveCartridgeResponse,
    ArtifactInventoryResponse,
    CodeDiffResponse,
    DeathClusterPoint,
    GenesisTriggerRequest,
    GenesisTriggerResponse,
    HeatmapResponse,
    PaletteColorItem,
    PaletteInfoResponse,
    PEVCycleInfo,
    SteamDNAResponse,
    StudioStatusResponse,
    SwarmEventType,
    SwarmStreamEvent,
    TelemetryDashboardResponse,
    TelemetryEventsResponse,
    VelocityVector,
)
from engine.ui.server import create_app, resolve_asset_safe, swarm_bus



@pytest.fixture
def mock_wal_db(tmp_path: Path) -> Path:
    """Creates a temporary SQLite database initialized with the telemetry schema."""
    db_file = tmp_path / "telemetry_test.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode = WAL;")
    schema_path = Path(__file__).resolve().parent.parent / "engine" / "telemetry" / "schema.sql"
    if schema_path.exists():
        conn.executescript(schema_path.read_text(encoding="utf-8"))
    else:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                probe_name TEXT,
                pos_x REAL NOT NULL,
                pos_y REAL NOT NULL,
                vel_x REAL DEFAULT 0.0,
                vel_y REAL DEFAULT 0.0,
                health REAL,
                payload_json TEXT DEFAULT '{}',
                frame_index INTEGER DEFAULT 0,
                timestamp_utc TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                episode_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                outcome TEXT,
                ended_at REAL
            );
        """)
    conn.commit()
    conn.close()
    return db_file


@pytest.fixture
def mock_cartridges(tmp_path: Path) -> Tuple[Path, Path]:
    """Sets up temporary cartridges root with active and seed cartridges."""
    cart_root = tmp_path / "cartridges"
    active_dir = cart_root / "active"
    seed_dir = cart_root / "seed"
    active_dir.mkdir(parents=True)
    seed_dir.mkdir(parents=True)

    # Active cartridge files
    manifest = {
        "title": "Test Arena Odyssey",
        "episode": 1,
        "author": "AutonomousStudioSwarm",
    }
    (active_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (active_dir / "mechanics.py").write_text("class Mechanics:\n    speed = 120.0\n", encoding="utf-8")

    # Assets in active
    audio_dir = active_dir / "assets" / "audio"
    audio_dir.mkdir(parents=True)
    # Minimal 44-byte standard RIFF WAVE PCM header
    riff_wav = (
        b"RIFF\x2c\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x08\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00"
    )
    (audio_dir / "hero_start.wav").write_bytes(riff_wav)

    sprites_dir = active_dir / "assets" / "sprites"
    sprites_dir.mkdir(parents=True)
    # Minimal 8-byte PNG signature + empty chunk
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\x00\x00\x00\x1f\xf3\xffa"
    (sprites_dir / "hero.png").write_bytes(png_bytes)

    # Seed cartridge
    (seed_dir / "manifest.json").write_text('{"title": "Seed Arena", "episode": 0}', encoding="utf-8")
    (seed_dir / "mechanics.py").write_text("class Mechanics:\n    speed = 100.0\n", encoding="utf-8")

    return cart_root, active_dir


class TestServerAppAndHealth:
    """Verifies ASGI server instantiation, route registration, and health check."""

    def test_routes_registry(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        app = create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir)
        routes = [r.path for r in app.routes]

        expected_paths = [
            "/",
            "/health",
            "/api/status",
            "/api/studio/status",
            "/api/cartridges",
            "/api/cartridges/active",
            "/api/artifacts",
            "/api/steam/dna",
            "/api/info",
            "/api/telemetry/events",
            "/api/telemetry/dashboard",
            "/api/telemetry/heatmap",
            "/api/assets/audio/{filename}",
            "/api/assets/sprites/{filename}",
            "/api/assets/palette",
            "/api/diff",
            "/api/genesis/trigger",
            "/ws/stream",
            "/ws/swarm",
        ]
        for path in expected_paths:
            assert path in routes, f"Missing route: {path}"

    def test_health_check_response(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok", "service": "studio_observability"}

    def test_cors_headers_present(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert res.status_code == 200
        assert res.headers.get("access-control-allow-origin") == "*"


class TestStatusEndpoints:
    """Verifies /api/status and /api/studio/status endpoints."""

    def test_api_status_deliberation_first(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert list(data.keys())[0] == "deliberation"
        assert len(data["deliberation"]) > 0

    def test_api_status_db_event_count(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        conn = sqlite3.connect(str(mock_wal_db))
        for i in range(12):
            conn.execute(
                "INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y) "
                "VALUES ('s1', 'e1', ?, 'BASE_MOVE', 10.0, 20.0);",
                (100.0 + i,),
            )
        conn.commit()
        conn.close()

        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/status")
        assert res.status_code == 200
        assert res.json()["db_event_count"] == 12

    def test_api_studio_status_agent_states_and_pev(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/studio/status")
        assert res.status_code == 200
        data = res.json()

        # Check all 9 personas
        authoritative_agents = [
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
        agent_states = data["agent_state"]
        for agent in authoritative_agents:
            assert agent in agent_states, f"Agent '{agent}' missing from studio status"
            assert agent_states[agent] == "ONLINE"

        # Check PEV cycle info
        assert "pev_cycle" in data
        assert data["pev_cycle"]["phase"] == "IDLE"
        assert data["pev_cycle"]["max_retries"] == 3


class TestTelemetryEventsEndpoint:
    """Verifies SQLite WAL paginated queries and filtering."""

    def test_telemetry_events_empty_database(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/telemetry/events")
        assert res.status_code == 200
        data = res.json()
        assert data["total_count"] == 0
        assert data["events"] == []
        assert list(data.keys())[0] == "deliberation"

    def test_telemetry_events_populated_and_paginated(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        conn = sqlite3.connect(str(mock_wal_db))
        for i in range(25):
            conn.execute(
                "INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, probe_name, pos_x, pos_y, payload_json) "
                "VALUES ('sess_1', 'ep_1', ?, ?, ?, ?, ?, ?);",
                (
                    10.0 + i,
                    "CUSTOM_PROBE" if i % 2 == 0 else "BASE_MOVE",
                    "dash_probe" if i % 2 == 0 else "pos_probe",
                    float(i * 10),
                    float(i * 5),
                    json.dumps({"idx": i}),
                ),
            )
        conn.commit()
        conn.close()

        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))

        # Default query
        res = client.get("/api/telemetry/events?limit=10&offset=0")
        assert res.status_code == 200
        data = res.json()
        assert data["total_count"] == 25
        assert len(data["events"]) == 10
        assert data["events"][0]["payload"] == {"idx": 24}

        # Filter by event_type
        res_filter = client.get("/api/telemetry/events?event_type=CUSTOM_PROBE")
        assert res_filter.status_code == 200
        data_f = res_filter.json()
        assert data_f["total_count"] == 13
        assert all(e["event_type"] == "CUSTOM_PROBE" for e in data_f["events"])


class TestTelemetryHeatmapEndpoint:
    """Verifies 2D density aggregation, death clustering, and velocity vectors."""

    def test_heatmap_empty_database(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/telemetry/heatmap")
        assert res.status_code == 200
        data = res.json()
        assert len(data["heatmap"]) == 16
        assert len(data["heatmap"][0]) == 16
        assert data["total_samples"] == 0

    def test_heatmap_density_deaths_and_velocities(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        conn = sqlite3.connect(str(mock_wal_db))
        # Insert movement points
        for i in range(40):
            conn.execute(
                "INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y, vel_x, vel_y) "
                "VALUES ('s1', 'e1', ?, 'BASE_MOVE', 640.0, 360.0, 50.0, 0.0);",
                (10.0 + i,),
            )
        # Insert death choke points at (300, 300)
        for i in range(4):
            conn.execute(
                "INSERT INTO telemetry_events (session_id, episode_id, timestamp, event_type, pos_x, pos_y, payload_json) "
                "VALUES ('s1', 'e1', ?, 'PLAYER_DEATH', 300.0, 300.0, ?);",
                (50.0 + i, json.dumps({"cause": "corridor_trap"})),
            )
        conn.commit()
        conn.close()

        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/telemetry/heatmap")
        assert res.status_code == 200
        data = res.json()

        assert data["total_samples"] == 44
        assert data["heatmap"][8][8] == 40.0

        # Assert death clusters
        assert len(data["death_clusters"]) >= 1
        dc = data["death_clusters"][0]
        assert dc["death_count"] == 4
        assert dc["primary_cause"] == "corridor_trap"
        assert list(dc.keys())[0] == "deliberation"

        # Assert velocity vectors
        assert len(data["velocity_vectors"]) >= 1
        vv = data["velocity_vectors"][0]
        assert vv["speed"] > 0
        assert list(vv.keys())[0] == "deliberation"


class TestCartridgesEndpoints:
    """Verifies /api/cartridges and /api/cartridges/active endpoints."""

    def test_cartridges_list(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/cartridges")
        assert res.status_code == 200
        carts = res.json()["cartridges"]
        assert "active" in carts
        assert "seed" in carts

    def test_active_cartridge_details(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/cartridges/active")
        assert res.status_code == 200
        data = res.json()
        assert data["exists"] is True
        assert data["manifest"]["title"] == "Test Arena Odyssey"
        assert data["has_audio"] is True
        assert data["has_sprites"] is True
        assert "hero_start.wav" in data["assets"]["audio"]
        assert "hero.png" in data["assets"]["sprites"]
        assert list(data.keys())[0] == "deliberation"


class TestAssetStreamingAndSecurity:
    """Verifies audio/sprite streaming, directory traversal defense, and palette/diff endpoints."""

    def test_audio_asset_serving(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/assets/audio/hero_start.wav")
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"
        assert res.content.startswith(b"RIFF")

    def test_audio_asset_range_request(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/assets/audio/hero_start.wav", headers={"Range": "bytes=0-15"})
        assert res.status_code == 206
        assert res.headers.get("accept-ranges") == "bytes"
        assert "bytes 0-15/" in res.headers.get("content-range", "")
        assert len(res.content) == 16

    def test_sprites_asset_serving(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/assets/sprites/hero.png")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        assert res.content.startswith(b"\x89PNG")

    def test_missing_asset_returns_404(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/assets/audio/nonexistent_voice.wav")
        assert res.status_code == 404

    def test_directory_traversal_neutralization(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        adversarial_paths = [
            "/api/assets/audio/../../../../etc/passwd",
            "/api/assets/audio/..%2F..%2Fconfig.py",
            "/api/assets/sprites/../../../run_game.py",
        ]
        for path in adversarial_paths:
            res = client.get(path)
            assert res.status_code in [400, 404]

    def test_palette_endpoint(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/assets/palette")
        assert res.status_code == 200
        data = res.json()
        assert len(data["colors"]) == 16
        assert data["colors"][0]["name"] == "black"
        assert data["colors"][0]["rgb"] == [0, 0, 0]
        assert data["colors"][12]["name"] == "cyan"
        assert list(data.keys())[0] == "deliberation"

    def test_diff_endpoint(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/diff?target=seed&module=mechanics.py")
        assert res.status_code == 200
        data = res.json()
        assert "diff" in data
        assert list(data.keys())[0] == "deliberation"

    def test_cartridge_query_parameter_traversal_neutralized(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        traversal_attempts = [
            "/api/assets/audio/passwd?cartridge=/etc",
            "/api/assets/audio/.env?cartridge=.",
            "/api/assets/audio/test.wav?cartridge=..",
            "/api/assets/audio/test.wav?cartridge=../seed",
            "/api/assets/sprites/hero.png?cartridge=/var",
            "/api/assets/generic/data.json?cartridge=.&type=",
        ]
        for url in traversal_attempts:
            res = client.get(url)
            assert res.status_code in [400, 404]
            assert b"root:" not in res.content
            assert b"GEMINI_API_KEY" not in res.content

    def test_diff_endpoint_traversal_neutralized(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        traversal_diffs = [
            "/api/diff?target=..&module=.env",
            "/api/diff?target=../../../../../../etc&candidate=&module=passwd",
            "/api/diff?target=seed&candidate=..&module=mechanics.py",
            "/api/diff?target=seed&candidate=none&module=.env",
        ]
        for url in traversal_diffs:
            res = client.get(url)
            assert res.status_code in [400, 404]
            assert "GEMINI_API_KEY" not in res.text
            assert "root:" not in res.text

    def test_diff_endpoint_missing_cartridges_dir(self, tmp_path: Path):
        client = TestClient(create_app(cartridges_root=tmp_path / "nonexistent"))
        res = client.get("/api/diff?target=seed&module=mechanics.py")
        assert res.status_code == 200
        data = res.json()
        assert data["deliberation"] == "No cartridges directory present"
        assert data["has_changes"] is False
        assert data["diff"] == ""


class TestWebSocketStreamingHub:
    """Verifies WebSocket broadcast lifecycle, canonical helpers, and threadsafe bridge."""

    def test_websocket_connect_lifecycle(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        with client.websocket_connect("/ws/stream") as ws:
            packet = ws.receive_json()
            assert packet["event_type"] == "connected"
            assert list(packet.keys())[0] == "deliberation"

    def test_broadcast_canonical_helpers(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        app = create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir)
        client = TestClient(app)

        with client.websocket_connect("/ws/stream") as ws:
            init = ws.receive_json()
            assert init["event_type"] == "connected"

            # 1. Agent stream
            swarm_bus.publish_agent_stream(
                agent_name="LeadGameplaySWEAgent",
                deliberation="Synthesizing dynamic dash ability.",
                token="def dash(self):",
            )
            msg1 = ws.receive_json()
            assert msg1["event_type"] == "AGENT_STREAM"
            assert msg1["agent_name"] == "LeadGameplaySWEAgent"
            assert msg1["payload"]["token"] == "def dash(self):"

            # 2. Tool invocation
            swarm_bus.publish_tool_invocation(
                agent_name="TelemetryArchitectAgent",
                deliberation="Registering dash probe.",
                tool_name="register_probe",
                arguments={"probe": "dash_duration"},
                status="success",
            )
            msg2 = ws.receive_json()
            assert msg2["event_type"] == "TOOL_INVOCATION"
            assert msg2["payload"]["tool_name"] == "register_probe"

            # 3. Artifact emitted
            swarm_bus.publish_artifact_emitted(
                agent_name="VoiceActingAudioProducerAgent",
                deliberation="Emitting synthesized voice line.",
                artifact_type="audio",
                filename="hero_start.wav",
            )
            msg3 = ws.receive_json()
            assert msg3["event_type"] == "ARTIFACT_EMITTED"
            assert msg3["payload"]["filename"] == "hero_start.wav"

            # 4. QA gate result
            swarm_bus.publish_qa_gate_result(
                agent_name="QAPlaytesterBuildSupervisor",
                deliberation="Gate B NavMesh traversability verified.",
                gate="Gate B",
                passed=True,
            )
            msg4 = ws.receive_json()
            assert msg4["event_type"] == "QA_GATE_RESULT"
            assert msg4["payload"]["passed"] is True

            # 5. Cartridge promoted
            swarm_bus.publish_cartridge_promoted(
                agent_name="StudioOrchestrator",
                deliberation="Promoted update episode 1.",
                episode=1,
                cartridge_path="cartridges/active",
            )
            msg5 = ws.receive_json()
            assert msg5["event_type"] == "CARTRIDGE_PROMOTED"
            assert msg5["payload"]["episode"] == 1

        assert swarm_bus.subscriber_count == 0

    def test_threadsafe_bridge_from_worker_thread(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        app = create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir)
        client = TestClient(app)

        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()  # connected

            def _bg_publisher():
                time.sleep(0.05)
                swarm_bus.publish_threadsafe({
                    "deliberation": "Cross-thread background delivery.",
                    "event_type": "BACKGROUND_THREAD_EVENT",
                    "agent_name": "WorkerThread",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "payload": {"status": "ok"},
                })

            t = threading.Thread(target=_bg_publisher)
            t.start()
            t.join(timeout=2.0)

            msg = ws.receive_json()
            assert msg["event_type"] == "BACKGROUND_THREAD_EVENT"
            assert msg["deliberation"] == "Cross-thread background delivery."


class TestEpistemicModelsValidation:
    """Verifies Rule 4: deliberation is strictly at field index 0 on all models."""

    def test_all_models_deliberation_field_first(self):
        models = [
            StudioStatusResponse,
            TelemetryEventsResponse,
            ActiveCartridgeResponse,
            DeathClusterPoint,
            VelocityVector,
            HeatmapResponse,
            PaletteColorItem,
            PaletteInfoResponse,
            CodeDiffResponse,
            GenesisTriggerRequest,
            GenesisTriggerResponse,
            SwarmStreamEvent,
            PEVCycleInfo,
            TelemetryDashboardResponse,
            ArtifactInventoryResponse,
            SteamDNAResponse,
        ]
        for model in models:
            first_field = list(model.model_fields.keys())[0]
            assert first_field == "deliberation", f"Model {model.__name__} violates field 0: {first_field}"

    def test_serialized_json_key_order(self):
        pev = PEVCycleInfo(deliberation="Deliberation test on PEV cycle")
        dumped = pev.model_dump()
        assert list(dumped.keys())[0] == "deliberation"

    def test_omitted_deliberation_raises_validation_error(self):
        with pytest.raises(Exception):
            PEVCycleInfo()

    def test_whitespace_deliberation_strictly_rejected(self):
        from pydantic import ValidationError
        sample_args = {
            "ActiveCartridgeResponse": {"deliberation": "   "},
            "ArtifactInventoryResponse": {"deliberation": "   "},
            "CodeDiffResponse": {"deliberation": "   ", "source_cartridge": "a", "candidate_cartridge": "b", "module": "m.py", "diff": ""},
            "DeathClusterPoint": {"deliberation": "   ", "x": 0.0, "y": 0.0},
            "GenesisTriggerRequest": {"deliberation": "   "},
            "GenesisTriggerResponse": {"deliberation": "   ", "job_id": "j1"},
            "HeatmapResponse": {"deliberation": "   "},
            "PEVCycleInfo": {"deliberation": "   "},
            "PaletteColorItem": {"deliberation": "   ", "index": 0, "name": "c", "rgb": (0, 0, 0), "hex": "#000"},
            "PaletteInfoResponse": {"deliberation": "   "},
            "SteamDNAResponse": {"deliberation": "   "},
            "StudioStatusResponse": {"deliberation": "   "},
            "SwarmStreamEvent": {"deliberation": "   ", "event_type": "echo", "timestamp_utc": "2026-10-16T00:00:00Z"},
            "TelemetryDashboardResponse": {"deliberation": "   "},
            "TelemetryEventsResponse": {"deliberation": "   "},
            "VelocityVector": {"deliberation": "   ", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
        }
        for name, kwargs in sample_args.items():
            cls = globals()[name]
            with pytest.raises(ValidationError):
                cls(**kwargs)


class TestTelemetryDashboardEndpoint:
    """Verifies /api/telemetry/dashboard dynamic manifest and custom events."""

    def test_dashboard_empty_active_cartridge(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/telemetry/dashboard")
        assert res.status_code == 200
        data = res.json()
        assert list(data.keys())[0] == "deliberation"
        assert "has_active_game" in data
        assert "probes" in data
        assert "metrics" in data

    def test_dashboard_with_custom_manifest_and_events(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        manifest = {
            "game_title": "Void Raider",
            "genre": "Roguelike Action",
            "probes": [
                {"probe_name": "dash_evasion", "category": "movement", "hypothesis_target": "H_MOBILITY"}
            ],
            "visualizations": [
                {"id": "v1", "type": "metric_counter", "title": "Total Dashes"}
            ],
            "hypotheses_addressed": ["H_MOBILITY"]
        }
        (active_dir / "telemetry_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        conn = sqlite3.connect(str(mock_wal_db))
        conn.execute(
            "INSERT INTO custom_telemetry_events (session_id, episode_id, timestamp, probe_name, category, payload_json) "
            "VALUES ('s1', 'e1', 10.0, 'dash_evasion', 'movement', '{\"stamina\": 20}');"
        )
        conn.commit()
        conn.close()

        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/telemetry/dashboard")
        assert res.status_code == 200
        data = res.json()
        assert list(data.keys())[0] == "deliberation"
        assert data["has_active_game"] is True
        assert data["game_title"] == "Void Raider"
        assert data["genre"] == "Roguelike Action"
        assert len(data["probes"]) >= 1
        assert data["metrics"]["total_custom_events"] >= 1
        assert data["metrics"]["category_counts"]["movement"] >= 1
        assert len(data["hypotheses"]) >= 1


class TestArtifactsInventoryEndpoint:
    """Verifies /api/artifacts comprehensive scanning."""

    def test_artifacts_inventory(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        (active_dir / "world_spec.json").write_text('{"title": "Void Realm"}', encoding="utf-8")
        (active_dir / "narrative_bible.json").write_text('{"characters": []}', encoding="utf-8")

        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/artifacts")
        assert res.status_code == 200
        data = res.json()
        assert list(data.keys())[0] == "deliberation"
        assert data["exists"] is True
        assert data["world_spec"] == {"title": "Void Realm"}
        assert data["narrative_bible"] == {"characters": []}
        assert len(data["code_modules"]) >= 1
        assert "mechanics.py" in data["code_modules"]
        assert len(data["audio_clips"]) >= 1
        assert len(data["sprites"]) >= 1


class TestSteamDNAEndpoint:
    """Verifies /api/steam/dna endpoint."""

    def test_steam_dna_baseline_response(self, mock_wal_db: Path, mock_cartridges: Tuple[Path, Path]):
        cart_root, active_dir = mock_cartridges
        client = TestClient(create_app(db_path=mock_wal_db, cartridges_root=cart_root, active_cartridge_dir=active_dir))
        res = client.get("/api/steam/dna")
        assert res.status_code == 200
        data = res.json()
        assert list(data.keys())[0] == "deliberation"
        assert "pacing_score" in data
        assert "challenge_tolerance" in data
        assert "top_genres" in data


class TestServerForbiddenTokensInvariance:
    """M2: Invariance tests enforcing 0 occurrences of hardcoded 2D arena and forbidden tokens."""

    def test_zero_forbidden_tokens_in_server_and_models(self):
        from pathlib import Path
        root = Path(__file__).parent.parent
        server_py = (root / "engine" / "ui" / "server.py").read_text(encoding="utf-8")
        models_py = (root / "engine" / "ui" / "models.py").read_text(encoding="utf-8")

        forbidden_patterns = [
            "Spatial Occupancy Telemetry Heatmap",
            "640x360",
            "Velocity Vectors",
            "Death Choke Points",
            "PICO-8",
            "hero_start.wav",
            "cartridges/seed",
        ]

        for pat in forbidden_patterns:
            assert pat not in server_py, f"Forbidden pattern '{pat}' found in server.py"
            assert pat not in models_py, f"Forbidden pattern '{pat}' found in models.py"



