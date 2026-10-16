"""tests/adversarial/test_tier5_ui_hardening.py
Tier 5 Adversarial Test Suite 1: UI Server OS NAME_MAX Fuzzing, Path Traversal & HTTP Range Hardening.
"""
from __future__ import annotations
from pathlib import Path
import pytest
from starlette.testclient import TestClient

import config
from engine.ui.server import create_app, resolve_asset_safe


@pytest.fixture
def ui_client(tmp_path: Path):
    """Creates a hermetic test client for the Studio Observability server."""
    db_file = tmp_path / "telemetry.db"
    cartridges_dir = tmp_path / "cartridges"
    active_dir = cartridges_dir / "active"
    seed_dir = cartridges_dir / "seed"
    (active_dir / "assets" / "audio").mkdir(parents=True, exist_ok=True)
    (active_dir / "assets" / "sprites").mkdir(parents=True, exist_ok=True)
    (seed_dir / "assets" / "audio").mkdir(parents=True, exist_ok=True)

    # Create dummy audio file for range testing (100 bytes)
    dummy_wav = active_dir / "assets" / "audio" / "test_tone.wav"
    dummy_wav.write_bytes(b"RIFF" + b"\x00" * 96)

    dummy_png = active_dir / "assets" / "sprites" / "test_icon.png"
    dummy_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

    app = create_app(
        db_path=db_file,
        cartridges_root=cartridges_dir,
        active_cartridge_dir=active_dir,
    )
    return TestClient(app, raise_server_exceptions=False)


class TestTier5UIServerHardening:
    """Verifies that the UI Server defends against OS NAME_MAX crashes, directory traversal, and malformed requests."""

    def test_ui_filename_exceeding_name_max_255(self, ui_client: TestClient):
        """Probes asset endpoints with filename parameters exceeding OS NAME_MAX (255 bytes).
        Must return HTTP 400 or 404 without raising unhandled OSError (HTTP 500).
        """
        oversized_wav = "a" * 300 + ".wav"
        oversized_png = "b" * 300 + ".png"
        oversized_json = "c" * 300 + ".json"

        # 1. Audio endpoint
        resp_audio = ui_client.get(f"/api/assets/audio/{oversized_wav}")
        assert resp_audio.status_code in (400, 404), f"Expected 400/404, got {resp_audio.status_code}"

        # 2. Sprites endpoint
        resp_sprites = ui_client.get(f"/api/assets/sprites/{oversized_png}")
        assert resp_sprites.status_code in (400, 404), f"Expected 400/404, got {resp_sprites.status_code}"

        # 3. Generic asset endpoint
        resp_generic = ui_client.get(f"/api/assets/generic/{oversized_json}")
        assert resp_generic.status_code in (400, 404), f"Expected 400/404, got {resp_generic.status_code}"

    def test_ui_cartridge_override_exceeding_name_max_255(self, ui_client: TestClient):
        """Probes query parameters with cartridge override exceeding 255 bytes.
        Must safely return HTTP 400 Bad Request instead of 500 OSError.
        """
        oversized_cart = "x" * 300
        resp = ui_client.get(f"/api/assets/audio/test_tone.wav?cartridge={oversized_cart}")
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "Invalid cartridge parameter" in data["error"] or "escape" in data["error"].lower()

    def test_ui_diff_parameters_exceeding_name_max_255(self, ui_client: TestClient):
        """Probes /api/diff with target, candidate, and module parameters exceeding 255 bytes.
        Must safely return HTTP 400 Bad Request instead of 500 OSError.
        """
        oversized = "d" * 300

        # Oversized target
        resp1 = ui_client.get(f"/api/diff?target={oversized}")
        assert resp1.status_code == 400
        assert "error" in resp1.json()

        # Oversized candidate
        resp2 = ui_client.get(f"/api/diff?candidate={oversized}")
        assert resp2.status_code == 400
        assert "error" in resp2.json()

        # Oversized module
        resp3 = ui_client.get(f"/api/diff?module={oversized}")
        assert resp3.status_code == 400
        assert "error" in resp3.json()

    def test_ui_path_traversal_unicode_evasion(self, ui_client: TestClient):
        """Fuzzes endpoints with path traversal vectors, null bytes, and traversal evasion patterns.
        Must strictly return 400 or 404 and NEVER leak parent files.
        """
        traversal_payloads = [
            "../../../../etc/passwd",
            "..%2f..%2f..%2fetc%2fpasswd",
            "..\\..\\..\\windows\\win.ini",
            "..%5c..%5c..%5cwindows",
            "test_tone%00.wav",
            ".hidden_file.wav",
            "assets/../../../run_game.py",
            "%2e%2e%2f%2e%2e%2frun_game.py",
        ]

        for payload in traversal_payloads:
            resp = ui_client.get(f"/api/assets/audio/{payload}")
            assert resp.status_code in (400, 404), f"Payload '{payload}' returned {resp.status_code}"

    def test_ui_range_request_overflow_and_negative_offsets(self, ui_client: TestClient):
        """Validates HTTP 206 Range streaming edge cases and overflows on audio assets.
        Must return 416 (Range Not Satisfiable) or full content without server crashes.
        """
        # 1. Valid range request
        valid_resp = ui_client.get("/api/assets/audio/test_tone.wav", headers={"Range": "bytes=0-15"})
        assert valid_resp.status_code == 206
        assert valid_resp.headers.get("Content-Range") == "bytes 0-15/100"
        assert len(valid_resp.content) == 16

        # 2. Offset beyond file length (overflow)
        overflow_resp = ui_client.get("/api/assets/audio/test_tone.wav", headers={"Range": "bytes=99999999-"})
        assert overflow_resp.status_code in (416, 200, 206)
        if overflow_resp.status_code == 416:
            assert "Content-Range" in overflow_resp.headers

        # 3. Inverted range (start > end)
        inverted_resp = ui_client.get("/api/assets/audio/test_tone.wav", headers={"Range": "bytes=80-20"})
        assert inverted_resp.status_code in (416, 200, 206)

        # 4. Negative / non-numeric range
        malformed_resp = ui_client.get("/api/assets/audio/test_tone.wav", headers={"Range": "bytes=-50-invalid"})
        assert malformed_resp.status_code in (200, 206, 416)

    def test_resolve_asset_safe_adversarial_inputs(self, tmp_path: Path):
        """Directly exercises resolve_asset_safe against adversarial inputs and non-existent roots."""
        search_roots = [tmp_path / "active", tmp_path / "seed"]

        # 1. Oversized filename (NAME_MAX)
        assert resolve_asset_safe("a" * 300, ["audio"], search_roots) is None

        # 2. Null byte injection
        assert resolve_asset_safe("track\0.wav", ["audio"], search_roots) is None

        # 3. Directory traversal tokens
        assert resolve_asset_safe("../../../etc/passwd", ["audio"], search_roots) is None
        assert resolve_asset_safe("..\\..\\config.py", ["audio"], search_roots) is None

        # 4. Empty or dot-prefixed filenames
        assert resolve_asset_safe("", ["audio"], search_roots) is None
        assert resolve_asset_safe(".hidden", ["audio"], search_roots) is None
