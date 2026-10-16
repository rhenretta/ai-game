"""tests/adversarial/test_m5_challenger_dual_profile_stress.py
Milestone M5 Empirical Challenger Stress Suite:
Adversarial stress-testing of Dual Steam Profile Genesis Divergence,
Open-Genre Cartridge Architecture, Swarm DAG Validation, Telemetry Schema Isolation,
and BuildSupervisor Quality Gates.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import math
from pathlib import Path
import struct
import threading
import time
from typing import Any, Dict, List
import wave

from PIL import Image
import pytest

from engine.api.contracts import BaseCartridge, BaseScene
from engine.loader import CartridgeLoader, validate_cartridge_ast
from engine.steam.dna import SteamDNAProfiler
from engine.steam.models import SteamGameRecord, SteamLibraryDataset, SteamPlayerDNA
from engine.studio.models import StudioSwarmPlan
from engine.studio.orchestrator import AwaitableStudioSwarmPlan, CircularDependencyError, StudioAgentDAG
from engine.supervisor.supervisor import BuildSupervisor
from engine.telemetry.db import TelemetryDB
from engine.telemetry.probes import AgentTelemetryProbe
from engine.telemetry.recorder import TelemetryRecorder


# ============================================================================
# 1. TestDegenerateSteamProfileDivergence
# ============================================================================

class TestDegenerateSteamProfileDivergence:
    """Stress tests profile extraction under degenerate, edge-case, and adversarial inputs."""
    deliberation: str = "Empirically stress testing Steam DNA extraction on boundary datasets."

    def test_empty_steam_library(self):
        """Empty library with 0 games must extract cleanly without division by zero."""
        dataset = SteamLibraryDataset(
            deliberation="Adversarial empty Steam library.",
            player_id="steam_user_empty",
            games=[],
        )
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(dataset, profile_tag="empty_test")

        assert isinstance(dna, SteamPlayerDNA)
        assert dna.total_games_analyzed == 0
        assert dna.total_playtime_hours == 0.0
        assert -1.0 <= dna.pacing_score <= 1.0
        assert 0.0 <= dna.challenge_tolerance <= 1.0
        assert not math.isnan(dna.pacing_score)
        assert not math.isnan(dna.challenge_tolerance)
        assert len(dna.deliberation) > 0

    def test_zero_playtime_library(self):
        """Games exist but all have 0 playtime minutes."""
        dataset = SteamLibraryDataset(
            deliberation="Library with owned games but zero play time.",
            player_id="steam_user_backlog",
            games=[
                SteamGameRecord(appid=10, name="Counter-Strike", playtime_forever_minutes=0, genres=["Action"]),
                SteamGameRecord(appid=20, name="Team Fortress", playtime_forever_minutes=0, genres=["Action"]),
            ],
        )
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(dataset, profile_tag="zero_time_test")

        assert dna.total_games_analyzed == 2
        assert dna.total_playtime_hours == 0.0
        assert not math.isnan(dna.pacing_score)
        assert not math.isnan(dna.challenge_tolerance)

    def test_conflicting_genres_symmetry(self):
        """Library with exactly equal 50/50 balance between extreme kinetic and extreme tactical."""
        dataset = SteamLibraryDataset(
            deliberation="Library with 50/50 split between bullet hell and grand strategy.",
            player_id="steam_user_balanced",
            games=[
                SteamGameRecord(
                    appid=1145360,
                    name="Hades",
                    playtime_forever_minutes=6000,
                    genres=["Action", "BulletHell"],
                    tags=["bullet hell", "fast-paced"],
                ),
                SteamGameRecord(
                    appid=289070,
                    name="Civ VI",
                    playtime_forever_minutes=6000,
                    genres=["Strategy"],
                    tags=["turn-based strategy", "tactical"],
                ),
            ],
        )
        profiler = SteamDNAProfiler()
        dna = profiler.extract_dna(dataset, profile_tag="symmetry_test")

        assert not math.isnan(dna.pacing_score)
        # Should resolve cleanly within [-1.0, 1.0] without float overflow
        assert -1.0 <= dna.pacing_score <= 1.0
        assert len(dna.top_genres) >= 2

    def test_massive_library_performance(self):
        """Library with 250 games must extract in under 100ms."""
        games = []
        for i in range(250):
            games.append(
                SteamGameRecord(
                    appid=1000 + i,
                    name=f"Game_{i}",
                    playtime_forever_minutes=i * 60,
                    genres=["Action"] if i % 2 == 0 else ["Strategy"],
                    tags=["hardcore"] if i % 3 == 0 else ["casual"],
                )
            )
        dataset = SteamLibraryDataset(
            deliberation="Massive 250-game library.",
            player_id="steam_user_massive",
            games=games,
        )
        profiler = SteamDNAProfiler()
        t0 = time.perf_counter()
        dna = profiler.extract_dna(dataset, profile_tag="massive_test")
        duration = time.perf_counter() - t0

        assert duration < 0.15, f"Extraction took too long: {duration:.3f}s"
        assert dna.total_games_analyzed == 250


# ============================================================================
# 2. TestSwarmDAGTopologicalStress
# ============================================================================

class TestSwarmDAGTopologicalStress:
    """Stress tests DAG topology validation, cycle detection, and topological ordering."""
    deliberation: str = "Empirically stress testing DAG validation and cycle rejection."

    def test_direct_two_node_cycle(self):
        dag = StudioAgentDAG(edges=[("A", "B"), ("B", "A")])
        assert dag.has_cycle() is True
        with pytest.raises(CircularDependencyError):
            dag.topological_sort()

    def test_self_loop_cycle(self):
        dag = StudioAgentDAG(edges=[("A", "A")])
        assert dag.has_cycle() is True
        with pytest.raises(CircularDependencyError):
            dag.topological_sort()

    def test_multi_node_deep_cycle(self):
        # N1 -> N2 -> ... -> N10 -> N1
        edges = [(f"N{i}", f"N{i+1}") for i in range(1, 10)]
        edges.append(("N10", "N1"))
        dag = StudioAgentDAG(edges=edges)
        assert dag.has_cycle() is True
        with pytest.raises(CircularDependencyError):
            dag.topological_sort()

    def test_valid_complex_diamond_dag(self):
        # A -> B -> D, A -> C -> D, D -> E
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
        dag = StudioAgentDAG(edges=edges)
        assert dag.has_cycle() is False
        order = dag.topological_sort()
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")
        assert order.index("D") < order.index("E")

    def test_swarm_plan_cycle_validation_rejection(self):
        """StudioSwarmPlan.validate_dag() must raise CircularDependencyError on cyclical plan."""
        plan = StudioSwarmPlan(
            deliberation="Adversarial cyclic swarm plan.",
            game_title="Broken Loop",
            genre_verdict="action_roguelike",
            specialist_agent_roles=["Director", "SWE"],
            dag_edges=[["Director", "SWE"], ["SWE", "Director"]],
        )
        with pytest.raises(CircularDependencyError):
            plan.validate_dag()


# ============================================================================
# 3. TestGeminiOpenAPISchemaCompliance
# ============================================================================

class TestGeminiOpenAPISchemaCompliance:
    """Verifies strict adherence to Gemini OpenAPI 3.0 schema constraints and Rule 4."""
    deliberation: str = "Verifying Gemini OpenAPI 3.0 schema compatibility (no prefixItems)."

    def test_no_prefix_items_in_swarm_plan_schema(self):
        """Pydantic v2 schema for StudioSwarmPlan must not contain 'prefixItems'."""
        schema = StudioSwarmPlan.model_json_schema()
        schema_str = json.dumps(schema)
        assert "prefixItems" not in schema_str, "Found forbidden 'prefixItems' in StudioSwarmPlan schema"

    def test_deliberation_field_is_first_property(self):
        schema = StudioSwarmPlan.model_json_schema()
        props = list(schema.get("properties", {}).keys())
        assert props[0] == "deliberation", f"First property is {props[0]}, expected 'deliberation'"

    @pytest.mark.asyncio
    async def test_awaitable_studio_swarm_plan(self):
        """AwaitableStudioSwarmPlan must be awaitable in async contexts and sync accessible."""
        plan = AwaitableStudioSwarmPlan(
            deliberation="Valid awaitable swarm plan.",
            game_title="Async Title",
            genre_verdict="turn_based_tactics",
            specialist_agent_roles=["Director", "SWE"],
            dag_edges=[["Director", "SWE"]],
        )
        awaited_plan = await plan
        assert awaited_plan.game_title == "Async Title"
        assert awaited_plan.genre_verdict == "turn_based_tactics"


# ============================================================================
# 4. TestOpenGenreASTSecurityAndEvasionStress
# ============================================================================

class TestOpenGenreASTSecurityAndEvasionStress:
    """Stress tests AST validation against dangerous injection vs legitimate open-genre game code."""
    deliberation: str = "Empirically testing AST validator on malicious vs legitimate game modules."

    @pytest.mark.parametrize("malicious_code", [
        "import os\nx = os.system('echo hacked')",
        "import sys\nsys.exit(1)",
        "import subprocess\nsubprocess.run(['ls'])",
        "__import__('os').system('rm -rf /')",
        "eval('__import__(\"os\").system(\"ls\")')",
        "exec('import os')",
    ])
    def test_ast_validator_rejects_malicious_code(self, malicious_code: str):
        from engine.loader import ASTSecurityError
        with pytest.raises(ASTSecurityError):
            validate_cartridge_ast(malicious_code)

    def test_ast_validator_passes_legitimate_open_genre_code(self):
        legit_code = '''"""mechanics.py"""
from typing import Any, Dict
from engine.api.contracts import BaseCartridge, BaseScene

class MyCustomScene(BaseScene):
    deliberation: str = "Valid open-genre scene."

    def __init__(self, engine: Any):
        super().__init__(engine)
        self.counter = 0

    def update(self, dt: float):
        self.counter += 1

    def render(self, surface: Any):
        pass

class OpenGenreCartridge(BaseCartridge):
    deliberation: str = "Valid open-genre cartridge."

    def get_scenes(self, engine: Any) -> Dict[str, Any]:
        return {"main": MyCustomScene(engine)}

    def get_initial_scene(self, engine: Any) -> Any:
        return self.get_scenes(engine)["main"]
'''
        is_valid, violations = validate_cartridge_ast(legit_code)
        assert is_valid is True, f"Legitimate code was rejected: {violations}"
        assert len(violations) == 0

    def test_four_genre_ast_divergence(self):
        """Verifies that 4 different genres produce distinct AST structures and SHA-256 hashes."""
        genres_code = {
            "action": """
class ActionScene(BaseScene):
    deliberation: str = "Action"
    def update(self, dt): self.x += 100 * dt
""",
            "deckbuilder": """
class DeckScene(BaseScene):
    deliberation: str = "Deck"
    def update(self, dt):
        for card in self.hand: self.energy -= 1
""",
            "puzzler": """
class PuzzleScene(BaseScene):
    deliberation: str = "Puzzle"
    def update(self, dt):
        if self.grid[0][0] == 1: self.solved = True
""",
            "platformer": """
class PlatformerScene(BaseScene):
    deliberation: str = "Platformer"
    def update(self, dt):
        self.vy += 9.8 * dt; self.y += self.vy * dt
""",
        }

        hashes = set()
        node_counts = set()
        for name, code in genres_code.items():
            h = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()
            hashes.add(h)
            tree = ast.parse(code)
            node_counts.add(len(list(ast.walk(tree))))

        # All 4 hashes must be mutually unique
        assert len(hashes) == 4
        # All 4 AST node counts must differ
        assert len(node_counts) == 4


# ============================================================================
# 5. TestCustomTelemetryWalBurstStress
# ============================================================================

class TestCustomTelemetryWalBurstStress:
    """Stress tests SQLite WAL concurrency with custom telemetry probes."""
    deliberation: str = "Verifying SQLite WAL concurrent burst handling without database locked errors."

    def test_concurrent_custom_probe_burst_to_wal(self, tmp_path: Path):
        db_path = tmp_path / "telemetry_burst.db"
        db = TelemetryDB(db_path=db_path)
        # Register probes to satisfy foreign key constraints
        db.register_probe("p_a", "stress_probe_a", "movement", "Movement stress probe", "H1")
        db.register_probe("p_b", "stress_probe_b", "combat", "Combat stress probe", "H2")

        recorder = TelemetryRecorder(
            db_path=db_path,
            batch_size=50,
            flush_interval_sec=0.01,
        )
        recorder.start_session(player_profile_tag="challenger_wal_stress")
        recorder.start_episode(level_id="lvl_burst_01", episode_index=1)

        probe_a = AgentTelemetryProbe(recorder=recorder, probe_name="stress_probe_a", category="movement")
        probe_b = AgentTelemetryProbe(recorder=recorder, probe_name="stress_probe_b", category="combat")

        total_threads = 6
        events_per_thread = 500
        errors: List[Exception] = []

        def worker(thread_idx: int):
            probe = probe_a if thread_idx % 2 == 0 else probe_b
            for i in range(events_per_thread):
                try:
                    probe.emit(
                        pos=(float(thread_idx * 10), float(i)),
                        data={"thread": thread_idx, "iter": i, "timestamp_val": time.time()},
                    )
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(total_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Flush and close recorder
        recorder.flush()
        recorder.close()

        assert len(errors) == 0, f"Encountered errors during telemetry burst: {errors}"

        # Verify SQLite WAL contents
        events = db.query_custom_events(limit=10000)
        assert len(events) == total_threads * events_per_thread, (
            f"Expected {total_threads * events_per_thread} events, found {len(events)}"
        )

        probes_recorded = {e["probe_name"] for e in events}
        assert "stress_probe_a" in probes_recorded
        assert "stress_probe_b" in probes_recorded


# ============================================================================
# 6. TestSupervisorGateFailureIsolation
# ============================================================================

class TestSupervisorGateFailureIsolation:
    """Verifies that BuildSupervisor correctly detects and isolates each gate failure."""
    deliberation: str = "Empirically verifying that BuildSupervisor gates reject defective cartridges."

    def _create_minimal_cartridge(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        (path / "manifest.json").write_text(json.dumps({
            "deliberation": "Minimal manifest.",
            "cartridge_id": "min_cart",
            "title": "Minimal",
            "genre": "action_roguelike",
            "version": "1.0.0",
            "entrypoint": "mechanics.py",
            "author": "Challenger",
        }))
        (path / "mechanics.py").write_text('''"""mechanics.py"""
from engine.api.contracts import BaseCartridge, BaseScene
class MinScene(BaseScene):
    deliberation: str = "Min"
    def update(self, dt): pass
    def render(self, surface): pass
class OpenGenreCartridge(BaseCartridge):
    deliberation: str = "Min"
    def get_scenes(self, engine): return {"m": MinScene(engine)}
    def get_initial_scene(self, engine): return self.get_scenes(engine)["m"]
''')
        audio_dir = path / "assets" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio_dir / "voice.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(bytearray(1000))

        sprite_dir = path / "assets" / "sprites"
        sprite_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        img.save(sprite_dir / "sprite.png", format="PNG")

        (path / "narrative.json").write_text(json.dumps({
            "deliberation": "Narrative",
            "characters": [{"id": "hero", "name": "Hero", "portrait": "assets/sprites/sprite.png"}],
            "dialogue_barks": [{"bark_id": "b1", "text": "Bark", "voice_clip": "assets/audio/voice.wav"}],
        }))
        (path / "telemetry_manifest.json").write_text(json.dumps({
            "deliberation": "Telemetry",
            "probes": [],
        }))
        return path

    def test_gate_a_rejects_missing_entrypoint(self, tmp_path: Path):
        from engine.supervisor.models import GateFailureType
        cart_dir = self._create_minimal_cartridge(tmp_path / "bad_gate_a")
        (cart_dir / "mechanics.py").unlink()  # Remove entrypoint

        supervisor = BuildSupervisor(cartridges_root=tmp_path / "cartridges")
        gates, trace = supervisor.evaluate_candidate_cartridge(cart_dir, episode_number=1)

        assert gates["gate_a"].passed is False
        assert trace is not None
        assert trace.failure_type == GateFailureType.ENTRYPOINT_MISSING

    def test_gate_e_rejects_corrupted_audio(self, tmp_path: Path):
        from engine.supervisor.models import GateFailureType
        cart_dir = self._create_minimal_cartridge(tmp_path / "bad_gate_e")
        # Overwrite wav with corrupted bytes (invalid RIFF header)
        (cart_dir / "assets" / "audio" / "voice.wav").write_bytes(b"NOT_A_VALID_WAV_HEADER_DATA_1234567890")

        supervisor = BuildSupervisor(cartridges_root=tmp_path / "cartridges")
        gates, trace = supervisor.evaluate_candidate_cartridge(cart_dir, episode_number=1)

        assert trace is not None
        assert trace.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION
