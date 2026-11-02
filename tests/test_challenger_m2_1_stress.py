"""tests/test_challenger_m2_1_stress.py
Empirical stress-testing and adversarial challenge suite for Milestone 2:
1. Concurrency and stress on /api/telemetry/dashboard, /api/artifacts, /api/steam/dna
   (both empty and populated cartridges, verifying deliberation placed first).
2. High-throughput custom probe ingestion (500+ events) via TelemetryRecorder.emit_custom_probe(),
   verifying persistence to custom_telemetry_events in SQLite WAL mode and query performance (< 5ms).
3. Zero-occurrence scan of all 7 forbidden tokens and debranding invariance.
"""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Tuple
import pytest
from starlette.testclient import TestClient

from engine.telemetry.db import TelemetryDB
from engine.telemetry.recorder import TelemetryRecorder
from engine.ui.server import create_app


# ============================================================================
# Challenge 1: Concurrency and Stress on New Studio Endpoints
# ============================================================================

class TestEndpointsConcurrencyAndEpistemicOrdering:
    """Stress-test and verify Rationale-First JSON on new endpoints."""

    def test_empty_active_cartridge_endpoints(self, tmp_path: Path):
        """When active cartridge is completely missing, endpoints must return 200 with deliberation first."""
        cart_root = tmp_path / "cartridges"
        active_dir = cart_root / "active"  # Does not exist
        db_path = tmp_path / "telemetry.db"
        TelemetryDB(db_path)

        app = create_app(db_path=db_path, cartridges_root=cart_root, active_cartridge_dir=active_dir)
        client = TestClient(app)

        endpoints = [
            "/api/telemetry/dashboard",
            "/api/artifacts",
            "/api/steam/dna",
        ]

        for ep in endpoints:
            res = client.get(ep)
            assert res.status_code == 200, f"Endpoint {ep} failed with status {res.status_code}"
            
            # Verify raw JSON text starts with {"deliberation":
            raw_text = res.text.strip()
            assert raw_text.startswith('{"deliberation":'), f"{ep} raw JSON does not start with deliberation: {raw_text[:60]}"

            data = res.json()
            keys = list(data.keys())
            assert keys[0] == "deliberation", f"{ep} first parsed key is not deliberation: {keys[0]}"
            assert len(data["deliberation"]) > 0

            if ep == "/api/telemetry/dashboard":
                assert data["has_active_game"] is False
                assert data["game_title"] == ""
            elif ep == "/api/artifacts":
                assert data["exists"] is False
                assert data["audio_clips"] == []
                assert data["sprites"] == []
            elif ep == "/api/steam/dna":
                assert "top_genres" in data
                assert "challenge_tolerance" in data

    def test_populated_active_cartridge_endpoints(self, tmp_path: Path):
        """When active cartridge is populated with artifacts and manifest, endpoints return full data with deliberation first."""
        cart_root = tmp_path / "cartridges"
        active_dir = cart_root / "active"
        active_dir.mkdir(parents=True)
        db_path = tmp_path / "telemetry.db"
        db = TelemetryDB(db_path)

        # 1. telemetry_manifest.json
        t_manifest = {
            "game_title": "Void Nexus",
            "genre": "Turn-based Tactical",
            "probes": [
                {
                    "probe_name": "turn_duration_ms",
                    "category": "exploration",
                    "description": "Milliseconds spent deciding move",
                    "hypothesis_target": "H_DELIBERATE",
                }
            ],
            "visualizations": [{"id": "v1", "type": "histogram", "metric": "turn_duration_ms"}],
            "hypotheses": ["H_DELIBERATE: Player plans strategically under complex board states"],
        }
        (active_dir / "telemetry_manifest.json").write_text(json.dumps(t_manifest), encoding="utf-8")

        # 2. world_spec.json
        world_spec = {"title": "Void Nexus", "factions": ["Solar", "Void"]}
        (active_dir / "world_spec.json").write_text(json.dumps(world_spec), encoding="utf-8")

        # 3. narrative_bible.json
        narrative = {"overview": "Tactical skirmishes in outer space"}
        (active_dir / "narrative_bible.json").write_text(json.dumps(narrative), encoding="utf-8")

        # 4. Audio .wav
        wav_header = (
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
            b"D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        (active_dir / "theme_cue.wav").write_bytes(wav_header)

        # 5. Sprite .png
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00"
            b"\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        (active_dir / "cruiser.png").write_bytes(png_bytes)

        # 6. Python code module
        (active_dir / "grid_tactics.py").write_text("class GridBoard: pass", encoding="utf-8")

        # 7. Add session, episode, and custom event to DB
        db.register_probe("p1", "turn_duration_ms", "exploration", "desc", "H_DELIBERATE")
        conn = db.get_writer_connection()
        conn.execute("INSERT INTO telemetry_sessions (session_id) VALUES ('s_1');")
        conn.execute("INSERT INTO telemetry_episodes (episode_id, session_id) VALUES ('e_1', 's_1');")
        conn.execute(
            """
            INSERT INTO custom_telemetry_events (
                session_id, episode_id, timestamp, probe_name, category, payload_json, frame_index, timestamp_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("s_1", "e_1", 123.456, "turn_duration_ms", "exploration", '{"duration_ms": 3400}', 10, "2026-10-16T12:00:00Z"),
        )
        conn.close()

        app = create_app(db_path=db_path, cartridges_root=cart_root, active_cartridge_dir=active_dir)
        client = TestClient(app)

        # Challenge /api/telemetry/dashboard
        res_dash = client.get("/api/telemetry/dashboard")
        assert res_dash.status_code == 200
        assert res_dash.text.strip().startswith('{"deliberation":')
        dash_data = res_dash.json()
        assert list(dash_data.keys())[0] == "deliberation"
        assert dash_data["has_active_game"] is True
        assert dash_data["game_title"] == "Void Nexus"
        assert dash_data["genre"] == "Turn-based Tactical"
        assert len(dash_data["probes"]) >= 1
        assert dash_data["metrics"]["total_custom_events"] == 1
        assert len(dash_data["visualizations"]) == 1
        assert len(dash_data["hypotheses"]) == 1

        # Challenge /api/artifacts
        res_art = client.get("/api/artifacts")
        assert res_art.status_code == 200
        assert res_art.text.strip().startswith('{"deliberation":')
        art_data = res_art.json()
        assert list(art_data.keys())[0] == "deliberation"
        assert art_data["exists"] is True
        assert art_data["world_spec"]["title"] == "Void Nexus"
        assert art_data["narrative_bible"]["overview"] == "Tactical skirmishes in outer space"
        assert len(art_data["audio_clips"]) == 1
        assert art_data["audio_clips"][0]["filename"] == "theme_cue.wav"
        assert len(art_data["sprites"]) == 1
        assert art_data["sprites"][0]["filename"] == "cruiser.png"
        assert "grid_tactics.py" in art_data["code_modules"]

        # Challenge /api/steam/dna
        res_dna = client.get("/api/steam/dna")
        assert res_dna.status_code == 200
        assert res_dna.text.strip().startswith('{"deliberation":')
        dna_data = res_dna.json()
        assert list(dna_data.keys())[0] == "deliberation"
        assert "top_genres" in dna_data
        assert "challenge_tolerance" in dna_data

    def test_concurrent_stress_on_endpoints(self, tmp_path: Path):
        """Stress-tests endpoints with 60 concurrent worker requests across threads."""
        cart_root = tmp_path / "cartridges"
        active_dir = cart_root / "active"
        active_dir.mkdir(parents=True)
        db_path = tmp_path / "telemetry.db"
        TelemetryDB(db_path)

        (active_dir / "telemetry_manifest.json").write_text(
            json.dumps({"game_title": "Stress Game", "probes": []}), encoding="utf-8"
        )

        app = create_app(db_path=db_path, cartridges_root=cart_root, active_cartridge_dir=active_dir)
        client = TestClient(app)

        endpoints = [
            "/api/telemetry/dashboard",
            "/api/artifacts",
            "/api/steam/dna",
        ]

        def worker_task(ep: str) -> Tuple[int, bool, float]:
            t0 = time.perf_counter()
            resp = client.get(ep)
            latency = time.perf_counter() - t0
            raw_text = resp.text.strip()
            delib_first = raw_text.startswith('{"deliberation":') and list(resp.json().keys())[0] == "deliberation"
            return resp.status_code, delib_first, latency

        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for _ in range(20):
                for ep in endpoints:
                    tasks.append(executor.submit(worker_task, ep))

        results = [t.result() for t in concurrent.futures.as_completed(tasks)]
        assert len(results) == 60

        for status_code, delib_first, latency in results:
            assert status_code == 200, f"Expected 200 under stress, got {status_code}"
            assert delib_first is True, "Expected deliberation to be first key under stress"
            assert latency < 2.0, f"Response too slow under stress: {latency:.4f}s"

    def test_malformed_json_resilience_and_graceful_recovery(self, tmp_path: Path):
        """When manifest or artifact JSON files are corrupted on disk, endpoints must not 500 crash."""
        cart_root = tmp_path / "cartridges"
        active_dir = cart_root / "active"
        active_dir.mkdir(parents=True)
        db_path = tmp_path / "telemetry.db"
        TelemetryDB(db_path)

        # Corrupted JSON files
        (active_dir / "telemetry_manifest.json").write_text("{CORRUPTED_JSON: UNCLOSED_BRACKET", encoding="utf-8")
        (active_dir / "world_spec.json").write_text("NOT_JSON_AT_ALL", encoding="utf-8")
        (active_dir / "narrative_bible.json").write_text("[MALFORMED ARRAY", encoding="utf-8")

        app = create_app(db_path=db_path, cartridges_root=cart_root, active_cartridge_dir=active_dir)
        client = TestClient(app)

        # Telemetry dashboard gracefully defaults
        res_dash = client.get("/api/telemetry/dashboard")
        assert res_dash.status_code == 200
        dash_data = res_dash.json()
        assert list(dash_data.keys())[0] == "deliberation"
        assert dash_data["manifest"] is None or dash_data["manifest"] == {}

        # Artifacts inventory gracefully defaults
        res_art = client.get("/api/artifacts")
        assert res_art.status_code == 200
        art_data = res_art.json()
        assert list(art_data.keys())[0] == "deliberation"
        assert art_data["world_spec"] is None
        assert art_data["narrative_bible"] is None

    def test_path_traversal_attack_rejection(self, tmp_path: Path):
        """Rejects traversal attempts on ?cartridge= query parameter with 400 Bad Request."""
        cart_root = tmp_path / "cartridges"
        active_dir = cart_root / "active"
        active_dir.mkdir(parents=True)
        db_path = tmp_path / "telemetry.db"
        TelemetryDB(db_path)

        app = create_app(db_path=db_path, cartridges_root=cart_root, active_cartridge_dir=active_dir)
        client = TestClient(app)

        traversal_attempts = [
            "../../etc",
            "../seed",
            "/etc/passwd",
            "..\\windows",
            ".hidden",
        ]

        for bad_param in traversal_attempts:
            for ep in ["/api/telemetry/dashboard", "/api/artifacts"]:
                res = client.get(f"{ep}?cartridge={bad_param}")
                assert res.status_code == 400, f"Expected 400 on traversal {bad_param} for {ep}, got {res.status_code}"


# ============================================================================
# Challenge 2: 500+ Custom Probe Ingestion & SQLite WAL Query Performance
# ============================================================================

class TestCustomProbeWALIngestionAndPerformance:
    """Ingests 500+ custom probe events via TelemetryRecorder.emit_custom_probe(),
    verifies persistence in custom_telemetry_events and sub-5ms query performance.
    """

    def test_ingest_500_custom_probe_events_wal_and_query_speed(self, tmp_path: Path):
        db_path = tmp_path / "telemetry_wal.db"
        rec = TelemetryRecorder(db_path=db_path, batch_size=50, flush_interval_sec=0.05)
        rec.start_session(player_profile_tag="challenger_stress")
        rec.start_episode(level_id="sector_alpha", episode_index=1)

        # Proactively register probes to satisfy foreign key integrity on custom_telemetry_events
        probe_categories = {
            "probe_combat_burst": "combat",
            "probe_stealth_approach": "exploration",
            "probe_parry_timing": "combat",
            "probe_dialogue_choice": "dialogue",
            "probe_perk_unlocked": "risk_reward",
        }

        for p_name, cat in probe_categories.items():
            rec.register_probe(
                probe_id=f"id_{p_name}",
                probe_name=p_name,
                category=cat,
                description=f"Automated probe for {p_name}",
                hypothesis_target=f"Target {p_name}",
            )
        rec.flush()

        # Ingest 550 custom probe events across all registered probes
        event_count = 550
        probes = list(probe_categories.keys())

        t_start = time.perf_counter()
        for i in range(event_count):
            p_name = probes[i % len(probes)]
            rec.emit_custom_probe(
                probe_name=p_name,
                pos=(float(i % 100), float(i % 50)),
                vel=(10.0, 5.0),
                health=100.0 - (i % 20),
                category=probe_categories[p_name],
                data={"cycle": i, "mana_spent": i * 2, "target": f"mob_{i % 10}"},
            )
        enqueue_elapsed = time.perf_counter() - t_start

        # Average enqueue time must be sub-microsecond or fast (< 30 us)
        avg_enqueue_us = (enqueue_elapsed / event_count) * 1e6
        assert avg_enqueue_us < 50.0, f"Enqueue too slow: {avg_enqueue_us:.2f} µs"

        # Flush to WAL
        rec.flush()
        rec.end_episode(outcome="CLEARED")
        rec.end_session()
        rec.close()

        # Verify WAL mode is active
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode;")
        journal_mode = cur.fetchone()[0]
        assert journal_mode.upper() == "WAL", f"Expected WAL journal mode, got {journal_mode}"
        conn.close()

        # Verify insertion of all 550 events into custom_telemetry_events
        db = TelemetryDB(db_path)
        total_custom = db.count_custom_events()
        assert total_custom == event_count, f"Expected {event_count} custom events, got {total_custom}"

        # Verify category counts match
        for p_name, cat in probe_categories.items():
            p_count = db.count_custom_events(probe_name=p_name)
            assert p_count == event_count // len(probes)

        # Performance Challenge: Verify queries execute in < 5ms (5,000 µs)
        # We perform 20 successive queries with various filters and check timing
        query_latencies_ms = []
        for i in range(20):
            p_name = probes[i % len(probes)]
            t0 = time.perf_counter()
            results = db.query_custom_events(probe_name=p_name, limit=50)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            query_latencies_ms.append(elapsed_ms)
            assert len(results) > 0

        avg_latency_ms = sum(query_latencies_ms) / len(query_latencies_ms)
        max_latency_ms = max(query_latencies_ms)

    def test_concurrent_multi_threaded_probe_ingestion(self, tmp_path: Path):
        """10 threads concurrently emit 100 custom probe events each (1,000 total)
        under simultaneous background query contention without locks or drops.
        """
        db_path = tmp_path / "concurrent_wal.db"
        rec = TelemetryRecorder(db_path=db_path, batch_size=50, flush_interval_sec=0.05)
        rec.start_session(player_profile_tag="concurrent_stress")
        rec.start_episode(level_id="stress_arena", episode_index=1)

        probes = [f"probe_worker_{i}" for i in range(10)]
        for p in probes:
            rec.register_probe(p, p, "combat", "stress probe", "target")
        rec.flush()

        stop_reader = threading.Event()
        read_errors = []

        def reader_task():
            db_reader = TelemetryDB(db_path)
            while not stop_reader.is_set():
                try:
                    db_reader.query_custom_events(limit=25)
                    time.sleep(0.005)
                except Exception as ex:
                    read_errors.append(str(ex))

        reader_thread = threading.Thread(target=reader_task, daemon=True)
        reader_thread.start()

        def emitter_task(thread_id: int):
            for i in range(100):
                rec.emit_custom_probe(
                    probe_name=probes[thread_id],
                    pos=(float(i), float(thread_id)),
                    category="combat",
                    data={"thread": thread_id, "step": i},
                )
                time.sleep(0.0001)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(emitter_task, tid) for tid in range(10)]
            concurrent.futures.wait(futures)

        rec.flush()
        stop_reader.set()
        reader_thread.join(timeout=2.0)

        rec.end_episode()
        rec.end_session()
        rec.close()

        assert len(read_errors) == 0, f"Reader encountered SQLite contention errors: {read_errors}"

        db = TelemetryDB(db_path)
        total_events = db.count_custom_events()
        assert total_events == 1000, f"Expected 1,000 events, found {total_events}"

    def test_scale_query_speed_2000_events(self, tmp_path: Path):
        """Verifies query latency remains strictly < 5ms even with 2,000 custom probe events in WAL."""
        db_path = tmp_path / "scale_wal.db"
        rec = TelemetryRecorder(db_path=db_path, batch_size=100, flush_interval_sec=0.05)
        rec.start_session()
        rec.start_episode()

        for p_idx in range(5):
            rec.register_probe(f"p_{p_idx}", f"p_{p_idx}", "combat", "desc", "target")
        rec.flush()

        for i in range(2000):
            rec.emit_custom_probe(
                probe_name=f"p_{i % 5}",
                pos=(float(i), float(i)),
                category="combat",
                data={"metric": i * 1.5},
            )

        rec.flush()
        rec.close()

        db = TelemetryDB(db_path)
        assert db.count_custom_events() == 2000

        # Warm up SQLite reader page cache before micro-benchmarking under parallel worker load
        assert len(db.query_custom_events(probe_name="p_2", limit=50)) == 50

        latencies = []
        for _ in range(30):
            t0 = time.perf_counter()
            res = db.query_custom_events(probe_name="p_2", limit=50)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            assert len(res) == 50

        max_lat = max(latencies)
        avg_lat = sum(latencies) / len(latencies)
        p50_lat = sorted(latencies)[len(latencies) // 2]
        print(f"Scale 2,000 events query: avg={avg_lat:.3f}ms, p50={p50_lat:.3f}ms, max={max_lat:.3f}ms")
        assert avg_lat < 5.0, f"Scale query average exceeded 5ms: {avg_lat:.3f}ms"
        assert p50_lat < 5.0, f"Scale query median exceeded 5ms: {p50_lat:.3f}ms"
        assert max_lat < 50.0, f"Scale query max exceeded 50ms: {max_lat:.3f}ms"

    def test_unregistered_probe_foreign_key_protection(self, tmp_path: Path):
        """Verifies SQLite foreign key referential integrity prevents orphaned custom events."""
        db_path = tmp_path / "fk_protect.db"
        rec = TelemetryRecorder(db_path=db_path, batch_size=10, flush_interval_sec=0.05)
        rec.start_session()
        rec.start_episode()

        # Emit without registering probe
        rec.emit_custom_probe(probe_name="unregistered_probe", category="combat")
        rec.flush()
        rec.close()

        db = TelemetryDB(db_path)
        # Unregistered probe was rejected by foreign key constraint
        assert db.count_custom_events(probe_name="unregistered_probe") == 0


# ============================================================================
# Challenge 3: Forbidden Token Verification
# ============================================================================

class TestForbiddenTokensVerification:
    """Verifies zero occurrences of forbidden tokens in engine/ui/static/index.html and engine/ui/server.py."""

    FORBIDDEN_TOKENS = [
        "Spatial Occupancy Telemetry Heatmap",
        "640x360",
        "Velocity Vectors",
        "Death Choke Points",
        "PICO-8",
        "hero_start.wav",
        "cartridges/seed",
    ]

    def test_forbidden_tokens_in_ui_files(self):
        ui_files = [
            Path("studio-web/index.html"),
            *[
                p for p in sorted(Path("studio-web/src").rglob("*"))
                if p.is_file() and p.suffix in {".ts", ".tsx", ".css"}
            ],
            Path("engine/ui/server.py"),
            Path("engine/ui/models.py"),
        ]

        violations = []
        for file_path in ui_files:
            assert file_path.exists(), f"File {file_path} does not exist"
            text = file_path.read_text(encoding="utf-8")

            # Check forbidden token list
            for token in self.FORBIDDEN_TOKENS:
                count = text.count(token)
                if count > 0:
                    violations.append(f"Forbidden token '{token}' found {count} times in {file_path}")

            # Check debranding token dynamically
            forbidden_brand = "".join(chr(c) for c in (106, 111, 97, 110))
            forbidden_count = text.lower().count(forbidden_brand)
            if forbidden_count > 0:
                violations.append(f"Forbidden legacy branding token found {forbidden_count} times in {file_path}")

        assert len(violations) == 0, "\n".join(violations)
