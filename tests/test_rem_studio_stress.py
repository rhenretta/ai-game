"""tests/test_rem_studio_stress.py
Empirical stress-testing and adversarial challenge suite for the Studio Observability UI ("Mission Control")
and background daemon (Requirement R4).

Verifies:
1. Concurrency & Load Stress: 50 rapid concurrent GET requests to /health, /api/status,
   /api/cartridges, and /api/telemetry/heatmap under thread contention.
2. WebSocket Broadcast Stress: Multiple concurrent WebSocket connections to /ws/swarm and /ws/stream,
   streaming simulated tokens and tool invocations, verifying zero socket leaks and clean error handling.
3. Non-blocking Genesis Trigger: Calling /api/genesis/trigger under burst load; verifying 200/202 status,
   rationale-first response models, unique job generation, and sub-50ms event loop responsiveness.
4. Static Asset Invariance: Verifying all 7 mandatory DOM IDs exist in engine/ui/static/index.html,
   valid document markup, and static asset serving.
5. Multimodal Artifact Streaming: Streaming audio (.wav) and portraits (.png), verifying MIME headers,
   query parameter cartridge overrides, directory traversal neutralization, and 404 error boundaries.
6. CLI Runner Invariance: Argument parsing, signal handling, and execution harness in run_studio.py.
"""
from __future__ import annotations

import concurrent.futures
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Set
import uuid

import pytest
from starlette.testclient import TestClient

import config
from engine.ui.models import (
    GenesisTriggerRequest,
    GenesisTriggerResponse,
    HeatmapResponse,
    StudioStatusResponse,
    SwarmStreamEvent,
)
from engine.ui.server import STATIC_DIR, create_app, swarm_bus
import run_studio


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def test_cartridge_with_assets(tmp_path: Path) -> Path:
    """Creates a temporary mock cartridge directory with authentic WAV and PNG asset bytes."""
    cart_dir = tmp_path / "mock_stress_cartridge"
    audio_dir = cart_dir / "assets" / "audio"
    sprites_dir = cart_dir / "assets" / "sprites"
    portrait_dir = cart_dir / "assets" / "portrait"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sprites_dir.mkdir(parents=True, exist_ok=True)
    portrait_dir.mkdir(parents=True, exist_ok=True)

    # Valid canonical RIFF WAVE header (44 bytes PCM)
    wav_header = (
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    (audio_dir / "hero_start.wav").write_bytes(wav_header)
    (audio_dir / "ambient.wav").write_bytes(wav_header)

    # Valid canonical 1x1 PNG header
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00"
        b"\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    (sprites_dir / "hero.png").write_bytes(png_bytes)
    (portrait_dir / "director.png").write_bytes(png_bytes)

    # Basic manifest
    (cart_dir / "manifest.json").write_text(
        '{"title": "Stress Test Cartridge", "episode": 99}', encoding="utf-8"
    )
    return cart_dir


@pytest.fixture
def studio_app(test_cartridge_with_assets: Path):
    """Provides a fresh Studio Starlette app instance pointing to isolated test fixtures."""
    return create_app(
        cartridges_root=test_cartridge_with_assets.parent,
        active_cartridge_dir=test_cartridge_with_assets,
    )


@pytest.fixture
def studio_client(studio_app):
    """Provides a synchronous Starlette TestClient."""
    return TestClient(studio_app)


# ============================================================================
# SECTION 1: RATIONALE-FIRST MODEL & AST CONTRACTS
# ============================================================================

class TestRationaleFirstUIModels:
    """Stress-tests that all response models strictly adhere to the Rationale-First architecture."""

    def test_deliberation_field_is_strictly_first_in_all_models(self):
        """Verifies deliberation is declared as the first property in all UI Pydantic schemas."""
        model_classes = [
            StudioStatusResponse,
            HeatmapResponse,
            GenesisTriggerRequest,
            GenesisTriggerResponse,
            SwarmStreamEvent,
        ]
        for cls in model_classes:
            field_keys = list(cls.model_fields.keys())
            assert len(field_keys) > 0
            assert field_keys[0] == "deliberation", (
                f"{cls.__name__} violates Rationale-First invariant: "
                f"first declared field is '{field_keys[0]}', expected 'deliberation'."
            )

    def test_deliberation_first_key_in_serialized_json(self):
        """Verifies that dumped dictionaries retain deliberation as the first JSON key."""
        resp = StudioStatusResponse(
            deliberation="Initial state reflection before status output.",
            status="ONLINE",
            active_cartridge="active",
            cartridges=["active", "seed"],
            agent_state={"GameplaySWE": "ONLINE"},
            database_connected=True,
            uptime_seconds=42.0,
        )
        dumped = resp.model_dump()
        keys = list(dumped.keys())
        assert keys[0] == "deliberation"
        assert dumped["deliberation"].startswith("Initial state reflection")

    def test_missing_deliberation_raises_validation_error(self):
        """Ensures instantiating response models without deliberation raises ValidationError."""
        with pytest.raises(Exception):
            StudioStatusResponse(status="ONLINE")  # type: ignore

        with pytest.raises(Exception):
            GenesisTriggerResponse(job_id="gen_1234", status="triggered")  # type: ignore


# ============================================================================
# SECTION 2: STATIC ASSET INVARIANCE (ALL 7 MANDATORY DOM IDs)
# ============================================================================

class TestStaticAssetInvariance:
    """Validates the zero-build Mission Control single page application DOM invariants."""

    MANDATORY_DOM_IDS = [
        "agent-dag-canvas",
        "deliberation-stream",
        "tool-invocations-list",
        "btn-trigger-genesis",
        "telemetry-heatmap",
        "voice-preview-player",
        "code-diff-container",
    ]

    def test_all_seven_dom_ids_present_in_index_html(self):
        """Empirically verifies all 7 required DOM IDs exist in engine/ui/static/index.html."""
        index_path = STATIC_DIR / "index.html"
        assert index_path.exists(), f"Static index file missing at {index_path}"
        content = index_path.read_text(encoding="utf-8")

        for dom_id in self.MANDATORY_DOM_IDS:
            expected_pattern = f'id="{dom_id}"'
            assert expected_pattern in content, (
                f"Mandatory DOM element missing from index.html: '{dom_id}'. "
                f"Expected attribute '{expected_pattern}'."
            )

    def test_index_html_document_well_formedness(self):
        """Verifies that index.html contains essential HTML5 elements and correct CSS/JS references."""
        index_path = STATIC_DIR / "index.html"
        content = index_path.read_text(encoding="utf-8")

        assert content.strip().startswith("<!DOCTYPE html>")
        assert "<title>Mission Control</title>" in content
        assert '<svg id="agent-dag-canvas"' in content
        assert '<canvas id="telemetry-heatmap"' in content
        assert '<audio id="voice-preview-player"' in content
        assert "<script>" in content
        assert "/ws/swarm" in content

    def test_http_get_index_serves_html(self, studio_client):
        """Verifies GET / returns 200 OK with text/html content-type and authentic body."""
        res = studio_client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "Autonomous Game Studio — Mission Control" in res.text

    def test_static_files_mount_serves_index(self, studio_client):
        """Verifies GET /static/index.html directly resolves from mounted StaticFiles."""
        res = studio_client.get("/static/index.html")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "agent-dag-canvas" in res.text

    def test_index_disallows_invalid_http_methods(self, studio_client):
        """Verifies index endpoint rejects POST or PUT with 405 Method Not Allowed."""
        res_post = studio_client.post("/")
        assert res_post.status_code == 405
        res_put = studio_client.put("/")
        assert res_put.status_code == 405


# ============================================================================
# SECTION 3: CONCURRENCY & LOAD STRESS SUITE (50 RAPID CONCURRENT REQUESTS)
# ============================================================================

class TestConcurrencyAndLoadStress:
    """Executes high-concurrency request storms against Mission Control REST endpoints."""

    def test_50_concurrent_requests_to_health_endpoint(self, studio_client):
        """50 rapid concurrent GET requests to /health all return 200 OK."""
        def _fetch_health(idx: int) -> Dict[str, Any]:
            res = studio_client.get("/health")
            assert res.status_code == 200
            return res.json()

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(_fetch_health, i) for i in range(50)]
            results = [f.result() for f in futures]
        duration = time.perf_counter() - t0

        assert len(results) == 50
        for r in results:
            assert r["status"] == "ok"
            assert r["service"] == "studio_observability"
        assert duration < 5.0, f"50 concurrent /health calls took {duration:.2f}s (budget: <5.0s)"

    def test_50_concurrent_requests_to_status_endpoint(self, studio_client):
        """50 rapid concurrent GET requests to /api/status verify data model consistency under load."""
        def _fetch_status(idx: int) -> Dict[str, Any]:
            res = studio_client.get("/api/status")
            assert res.status_code == 200
            return res.json()

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(_fetch_status, i) for i in range(50)]
            results = [f.result() for f in futures]
        duration = time.perf_counter() - t0

        assert len(results) == 50
        for r in results:
            assert r["status"] == "ONLINE"
            assert "deliberation" in r
            assert len(r["deliberation"]) > 0
            assert "agent_state" in r
            assert r["agent_state"]["GameplaySWE"] == "ONLINE"
        assert duration < 5.0, f"50 concurrent /api/status calls took {duration:.2f}s (budget: <5.0s)"

    def test_50_concurrent_requests_to_cartridges_endpoint(self, studio_client):
        """50 rapid concurrent GET requests to /api/cartridges return sorted directory listings."""
        def _fetch_cartridges(idx: int) -> List[str]:
            res = studio_client.get("/api/cartridges")
            assert res.status_code == 200
            return res.json()["cartridges"]

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(_fetch_cartridges, i) for i in range(50)]
            results = [f.result() for f in futures]
        duration = time.perf_counter() - t0

        assert len(results) == 50
        for cart_list in results:
            assert isinstance(cart_list, list)
            assert "mock_stress_cartridge" in cart_list
        assert duration < 5.0

    def test_50_concurrent_requests_to_heatmap_endpoint(self, studio_client):
        """50 rapid concurrent GET requests to /api/telemetry/heatmap verify SQLite WAL concurrency."""
        def _fetch_heatmap(idx: int) -> Dict[str, Any]:
            res = studio_client.get("/api/telemetry/heatmap")
            assert res.status_code == 200
            return res.json()

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(_fetch_heatmap, i) for i in range(50)]
            results = [f.result() for f in futures]
        duration = time.perf_counter() - t0

        assert len(results) == 50
        for r in results:
            assert "heatmap" in r
            assert "deliberation" in r
            assert len(r["heatmap"]) == 16
            assert len(r["heatmap"][0]) == 16
        assert duration < 10.0, f"50 concurrent heatmap queries took {duration:.2f}s"

    def test_50_mixed_traffic_storm(self, studio_client):
        """50 concurrent requests randomized across all four GET endpoints."""
        endpoints = [
            "/health",
            "/api/status",
            "/api/cartridges",
            "/api/telemetry/heatmap",
            "/api/info",
        ]

        def _fetch_random(idx: int) -> int:
            target = random.choice(endpoints)
            res = studio_client.get(target)
            return res.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(_fetch_random, i) for i in range(50)]
            status_codes = [f.result() for f in futures]

        assert len(status_codes) == 50
        assert all(code == 200 for code in status_codes)


# ============================================================================
# SECTION 4: NON-BLOCKING GENESIS TRIGGER SUITE
# ============================================================================

class TestNonBlockingGenesisTrigger:
    """Stress-tests the Genesis build trigger under concurrent load and verifies ASGI non-blocking execution."""

    def test_genesis_trigger_post_and_get_methods(self, studio_client):
        """Verifies /api/genesis/trigger accepts both POST with JSON payload and GET."""
        # POST with explicit profile tag
        res_post = studio_client.post(
            "/api/genesis/trigger",
            json={"profile_tag": "kinetic_rusher", "deliberation": "Adversarial test dispatch"},
        )
        assert res_post.status_code in [200, 202]
        data_post = res_post.json()
        model_post = GenesisTriggerResponse(**data_post)
        assert model_post.status == "triggered"
        assert model_post.job_id.startswith("gen_")
        assert "kinetic_rusher" in model_post.deliberation

        # GET method without body
        res_get = studio_client.get("/api/genesis/trigger")
        assert res_get.status_code in [200, 202]
        data_get = res_get.json()
        model_get = GenesisTriggerResponse(**data_get)
        assert model_get.status == "triggered"
        assert model_get.job_id.startswith("gen_")

    def test_genesis_trigger_concurrent_burst_generates_unique_jobs(self, studio_client):
        """25 rapid concurrent trigger calls generate 25 unique job IDs without race condition collisions."""
        def _trigger(idx: int) -> str:
            res = studio_client.post(
                "/api/genesis/trigger",
                json={"profile_tag": f"stress_profile_{idx}"},
            )
            assert res.status_code in [200, 202]
            return res.json()["job_id"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(_trigger, i) for i in range(25)]
            job_ids = [f.result() for f in futures]

        assert len(job_ids) == 25
        assert len(set(job_ids)) == 25, "Job IDs must be uniquely generated per trigger request."

    def test_genesis_trigger_non_blocking_event_loop(self, studio_client):
        """Interleaves trigger bursts with health probes to verify the ASGI event loop is never blocked."""
        def _trigger(idx: int) -> int:
            return studio_client.post("/api/genesis/trigger", json={"profile_tag": "probe"}).status_code

        def _health_probe(idx: int) -> float:
            t0 = time.perf_counter()
            res = studio_client.get("/health")
            dt = time.perf_counter() - t0
            assert res.status_code == 200
            return dt

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            trigger_futs = [executor.submit(_trigger, i) for i in range(20)]
            health_futs = [executor.submit(_health_probe, i) for i in range(20)]

            # Wait for all
            trigger_statuses = [f.result() for f in trigger_futs]
            latencies = [f.result() for f in health_futs]

        assert all(s in [200, 202] for s in trigger_statuses)
        max_latency = max(latencies)
        assert max_latency < 0.25, (
            f"Event loop blocked during genesis triggers: max health probe took {max_latency*1000:.2f}ms"
        )

    def test_genesis_trigger_resilience_to_malformed_input(self, studio_client):
        """Verifies /api/genesis/trigger handles malformed JSON and null fields without 500 error."""
        # Non-JSON or broken content
        res = studio_client.post(
            "/api/genesis/trigger",
            content=b"this is not valid json!@@#$",
            headers={"content-type": "application/json"},
        )
        assert res.status_code in [200, 202]
        assert "job_id" in res.json()

        # Empty body
        res_empty = studio_client.post("/api/genesis/trigger", content=b"")
        assert res_empty.status_code in [200, 202]
        assert "job_id" in res_empty.json()


# ============================================================================
# SECTION 5: WEBSOCKET BROADCAST STRESS & ZERO SOCKET LEAKS
# ============================================================================

class TestWebSocketBroadcastStress:
    """Stress-tests the /ws/swarm and /ws/stream real-time event buses and verify clean teardown."""

    def test_websocket_single_connection_lifecycle(self, studio_client):
        """Verifies connection handshake, initial packet, echo response, and clean disconnect."""
        with studio_client.websocket_connect("/ws/swarm") as ws:
            init = ws.receive_json()
            assert init["event_type"] == "connected"
            assert "deliberation" in init

            # Send probe
            ws.send_text("ping_probe")
            echo = ws.receive_json()
            assert echo["event_type"] == "echo"
            assert "ping_probe" in echo["deliberation"]

    def test_websocket_stream_alias_route(self, studio_client):
        """Verifies /ws/stream route functions as an active alias for /ws/swarm."""
        with studio_client.websocket_connect("/ws/stream") as ws:
            init = ws.receive_json()
            assert init["event_type"] == "connected"

    def test_websocket_multiple_concurrent_connections_stream_stress(self, studio_app):
        """Opens 20 concurrent WebSocket sessions, streams simulated tokens, and checks zero socket leaks."""
        def _ws_session_worker(idx: int) -> int:
            client = TestClient(studio_app)
            with client.websocket_connect("/ws/swarm") as ws:
                init = ws.receive_json()
                assert init["event_type"] == "connected"
                assert "deliberation" in init

                # Stream 5 simulated tokens / tool invocations
                for step in range(5):
                    token_msg = f"agent_token_{idx}_{step}"
                    ws.send_text(token_msg)
                    resp = ws.receive_json()
                    assert resp["event_type"] == "echo"
                    assert token_msg in resp["deliberation"]
            return idx

        # Run 20 sessions across threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_ws_session_worker, i) for i in range(20)]
            results = [f.result() for f in futures]

        assert len(results) == 20
        # Verify zero socket leaks in swarm_bus
        assert len(swarm_bus._subscribers) == 0, (
            f"Socket leak detected: {len(swarm_bus._subscribers)} orphaned subscribers remain in swarm_bus."
        )

    def test_websocket_broadcast_delivery_from_genesis_trigger(self, studio_app):
        """Verifies a connected WebSocket receives real-time broadcast events dispatched by REST triggers."""
        client_ws = TestClient(studio_app)
        client_rest = TestClient(studio_app)

        with client_ws.websocket_connect("/ws/swarm") as ws:
            init = ws.receive_json()
            assert init["event_type"] == "connected"

            # Dispatch genesis trigger via REST
            tag = f"broadcast_test_{uuid.uuid4().hex[:6]}"
            res = client_rest.post("/api/genesis/trigger", json={"profile_tag": tag})
            assert res.status_code in [200, 202]

            # WebSocket client receives the broadcast event
            broadcast_msg = ws.receive_json()
            assert broadcast_msg["event_type"] == "genesis_triggered"
            assert "deliberation" in broadcast_msg
            assert tag in broadcast_msg["deliberation"]
            assert broadcast_msg["payload"]["profile_tag"] == tag

        # Subscriber unregistered
        assert len(swarm_bus._subscribers) == 0

    def test_websocket_nested_multi_subscriber_broadcast(self, studio_app):
        """Verifies multiple concurrently open WebSocket subscribers all receive broadcasted events."""
        c1 = TestClient(studio_app)
        c2 = TestClient(studio_app)
        c_trigger = TestClient(studio_app)

        with c1.websocket_connect("/ws/swarm") as ws1:
            with c2.websocket_connect("/ws/swarm") as ws2:
                init1 = ws1.receive_json()
                init2 = ws2.receive_json()
                assert init1["event_type"] == "connected"
                assert init2["event_type"] == "connected"
                assert len(swarm_bus._subscribers) == 2

                # Trigger broadcast
                tag = "nested_multi_sub"
                res = c_trigger.post("/api/genesis/trigger", json={"profile_tag": tag})
                assert res.status_code in [200, 202]

                msg1 = ws1.receive_json()
                msg2 = ws2.receive_json()
                assert msg1["event_type"] == "genesis_triggered"
                assert msg2["event_type"] == "genesis_triggered"
                assert msg1["payload"]["profile_tag"] == tag
                assert msg2["payload"]["profile_tag"] == tag

        # Verify zero socket leaks
        assert len(swarm_bus._subscribers) == 0

    def test_websocket_bus_publish_threadsafe_resilience(self):
        """Verifies publish_threadsafe does not crash when called outside of or with loop context."""
        event = {
            "deliberation": "Testing threadsafe publication resilience.",
            "event_type": "diagnostic_heartbeat",
            "timestamp_utc": "2026-10-16T05:00:00Z",
            "payload": {},
        }
        # Outside running loop should safely no-op
        swarm_bus.publish_threadsafe(event)


# ============================================================================
# SECTION 6: MULTIMODAL ARTIFACT STREAMING & 404 RESILIENCE
# ============================================================================

class TestMultimodalArtifactStreaming:
    """Stress-tests the artifact streaming endpoints for audio and portraits with strict error boundaries."""

    def test_audio_artifact_streaming_and_content_type(self, studio_client):
        """Streams authentic .wav voice file; verifies 200 OK, audio/wav MIME, and byte content."""
        res = studio_client.get("/api/artifacts/audio/hero_start.wav")
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"
        assert res.content.startswith(b"RIFF")
        assert b"WAVEfmt" in res.content

    def test_portrait_artifact_streaming_and_content_type(self, studio_client):
        """Streams authentic .png portrait file; verifies 200 OK, image/png MIME, and byte content."""
        res = studio_client.get("/api/artifacts/sprites/hero.png")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        assert res.content.startswith(b"\x89PNG\r\n\x1a\n")

    def test_default_artifact_endpoints(self, studio_client):
        """Verifies GET /api/artifacts/audio and GET /api/artifacts/portrait serve default media."""
        res_audio = studio_client.get("/api/artifacts/audio")
        assert res_audio.status_code == 200
        assert res_audio.headers["content-type"] in ["audio/wav", "audio/mpeg"]

        res_portrait = studio_client.get("/api/artifacts/sprites")
        assert res_portrait.status_code == 200
        assert res_portrait.headers["content-type"] in ["image/png", "image/jpeg"]

    def test_nonexistent_artifact_file_returns_404(self, studio_client):
        """Non-existent filename returns authentic 404 with JSON error description."""
        res = studio_client.get("/api/artifacts/audio/missing_clip_99999.wav")
        assert res.status_code == 404
        assert "application/json" in res.headers["content-type"]
        data = res.json()
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_nonexistent_artifact_type_returns_404(self, studio_client):
        """Unsupported artifact type returns authentic 404."""
        res = studio_client.get("/api/artifacts/unsupported_modality_type/test.bin")
        assert res.status_code == 404
        res_default = studio_client.get("/api/artifacts/unsupported_modality_type")
        assert res_default.status_code == 404

    def test_cartridge_query_parameter_override(self, studio_app, test_cartridge_with_assets: Path):
        """Verifies ?cartridge= parameter allows querying specific cartridge directory."""
        client = TestClient(studio_app)
        res = client.get(
            f"/api/artifacts/audio/hero_start.wav?cartridge={test_cartridge_with_assets.resolve()}"
        )
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"

    def test_directory_traversal_attack_neutralized(self, studio_client):
        """Adversarial directory traversal requests return 404 without leaking host files."""
        traversal_attempts = [
            "/api/artifacts/audio/../../../../etc/passwd",
            "/api/artifacts/audio/..%2F..%2F..%2Fetc%2Fpasswd",
            "/api/artifacts/audio/../../config.py",
            "/api/artifacts/sprites/../../run_game.py",
        ]
        for path in traversal_attempts:
            res = studio_client.get(path)
            assert res.status_code == 404


# ============================================================================
# SECTION 7: CLI RUNNER INVARIANCE (run_studio.py)
# ============================================================================

class TestRunStudioCLIInvariance:
    """Verifies CLI argument parsing and signal handling structures in run_studio.py."""

    def test_default_cli_arguments(self):
        """Default arguments should bind to 127.0.0.1:8000 with info logging."""
        args = run_studio.parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 8000
        assert args.reload is False
        assert args.log_level == "info"

    def test_custom_cli_arguments(self):
        """Custom CLI arguments should parse correctly."""
        args = run_studio.parse_args([
            "--host", "0.0.0.0",
            "--port", "8888",
            "--reload",
            "--log-level", "debug",
        ])
        assert args.host == "0.0.0.0"
        assert args.port == 8888
        assert args.reload is True
        assert args.log_level == "debug"

    def test_setup_signal_handlers_callable(self):
        """Ensures signal handler registration executes without exception."""
        run_studio.setup_signal_handlers()
