"""tests/test_ui_static_index.py
Test specifications for the Mission Control React Frontend (`studio-web/`) and API root contract,
and retirement of the legacy monolithic `engine/ui/static/index.html` UI.
"""
import inspect
from pathlib import Path
import pytest
from starlette.testclient import TestClient

import config
from engine.sim.godot_verifier import GodotHeadlessSimulator
import engine.ui.server as ui_server
from engine.ui.server import STUDIO_WEB_DIR, create_app


class TestStudioWebFrontendArchitecture:
    """Verifies legacy static HTML UI removal, studio-web React SPA structure, and root serving contracts."""

    def test_legacy_static_html_retired(self):
        """Ensures engine/ui/static and engine/ui/static/index.html are permanently removed."""
        legacy_static_dir = config.PROJECT_ROOT / "engine" / "ui" / "static"
        assert not legacy_static_dir.exists(), f"Legacy static UI directory must not exist: {legacy_static_dir}"

    def test_studio_web_entrypoint_and_workspaces_exist(self):
        """Verifies studio-web index.html and core React workspace views exist."""
        index_html = STUDIO_WEB_DIR / "index.html"
        assert index_html.is_file(), f"Missing studio-web index.html at {index_html}"
        html_text = index_html.read_text(encoding="utf-8")
        assert html_text.strip().lower().startswith("<!doctype html>")
        assert "<title>Autonomous Game Studio — Mission Control</title>" in html_text
        assert '<div id="root"></div>' in html_text
        assert 'src="/src/main.tsx"' in html_text

        required_views = [
            STUDIO_WEB_DIR / "src" / "App.tsx",
            STUDIO_WEB_DIR / "src" / "features" / "board" / "ProductionBoardView.tsx",
            STUDIO_WEB_DIR / "src" / "features" / "writers-room" / "WritersRoomView.tsx",
            STUDIO_WEB_DIR / "src" / "features" / "bible" / "GameBibleView.tsx",
            STUDIO_WEB_DIR / "src" / "features" / "memory" / "GameMemoryView.tsx",
            STUDIO_WEB_DIR / "src" / "features" / "assets" / "AssetsAndCodeView.tsx",
            STUDIO_WEB_DIR / "src" / "features" / "playtest" / "PlaytestLabView.tsx",
            STUDIO_WEB_DIR / "src" / "features" / "dna" / "PlayerDnaView.tsx",
            STUDIO_WEB_DIR / "src" / "realtime" / "useSwarmSocket.ts",
        ]
        for view_path in required_views:
            assert view_path.is_file(), f"Expected React view missing: {view_path}"

    def test_root_endpoint_dev_json_and_production_dist_serving(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verifies GET / returns Rationale-First JSON when unbuilt and serves dist/index.html when compiled."""
        empty_dist = tmp_path / "dist_missing"
        monkeypatch.setattr(ui_server, "STUDIO_WEB_DIST_DIR", empty_dist)
        client = TestClient(create_app(db_path=tmp_path / "t.db", cartridges_root=tmp_path / "carts"))
        res_dev = client.get("/")
        assert res_dev.status_code == 200
        body = res_dev.json()
        assert list(body.keys())[0] == "deliberation"
        assert body["status"] == "ok"
        assert "ui_url" in body

        built_dist = tmp_path / "dist_built"
        (built_dist / "assets").mkdir(parents=True, exist_ok=True)
        (built_dist / "index.html").write_text("<!doctype html><title>Autonomous Game Studio — Mission Control</title>", encoding="utf-8")
        (built_dist / "assets" / "app.js").write_text("console.log('ready');", encoding="utf-8")
        monkeypatch.setattr(ui_server, "STUDIO_WEB_DIST_DIR", built_dist)

        client_prod = TestClient(create_app(db_path=tmp_path / "t2.db", cartridges_root=tmp_path / "carts2"))
        res_prod = client_prod.get("/")
        assert res_prod.status_code == 200
        assert "text/html" in res_prod.headers["content-type"]
        assert "Autonomous Game Studio — Mission Control" in res_prod.text

        res_asset = client_prod.get("/assets/app.js")
        assert res_asset.status_code == 200
        assert "ready" in res_asset.text

    def test_debranding_and_zero_forbidden_game_tokens_across_studio_web(self):
        """Enforces zero occurrences of legacy branding or hardcoded prototype game tokens across studio-web/src."""
        forbidden_brand = "".join(chr(c) for c in (106, 111, 97, 110))
        forbidden_patterns = [
            "Spatial Occupancy Telemetry Heatmap",
            "640x360",
            "Velocity Vectors",
            "Death Choke Points",
            "PICO-8",
            "hero_start.wav",
            "cartridges/seed",
        ]
        src_files = [STUDIO_WEB_DIR / "index.html", *sorted((STUDIO_WEB_DIR / "src").rglob("*"))]
        checked = 0
        for fpath in src_files:
            if not fpath.is_file() or fpath.suffix not in {".html", ".ts", ".tsx", ".css"}:
                continue
            checked += 1
            content = fpath.read_text(encoding="utf-8")
            assert forbidden_brand not in content.lower(), f"Found legacy debranding token in {fpath}"
            for pattern in forbidden_patterns:
                assert pattern not in content, f"Forbidden hardcoded game token '{pattern}' found in {fpath}"
        assert checked >= 20

    def test_headless_godot_renders_contract(self):
        """Verifies Godot automated renders execute strictly with --headless and --capture-viewport."""
        sim_src = inspect.getsource(GodotHeadlessSimulator.run_until_complete)
        assert '"--headless"' in sim_src
        assert '"--capture-viewport"' in sim_src
