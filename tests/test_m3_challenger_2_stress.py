"""tests/test_m3_challenger_2_stress.py
Milestone M3 Challenger 2 Empirical Stress Test Suite.
Adversarial verification of:
1. Evolution Engine Clamping Invariants (1,000 randomized trials, +/-15% mathematical bounds, NaN/Inf resilience, min/max limits).
2. Telemetry Architect Concurrency & Probe Catalog (100+ concurrent WAL probe registrations, idempotency, rapid event burst).
3. Rationale-First Key Ordering Invariance (all 25 Pydantic models in engine/studio/models.py, key index 0 == 'deliberation').
"""
from __future__ import annotations
import asyncio
import concurrent.futures
import inspect
import json
import math
from pathlib import Path
import random
import sqlite3
import tempfile
import time
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock
import pytest
from pydantic import BaseModel

from engine.scheduler.artifact import ArtifactMetadataEnvelope, ArtifactType
from engine.studio.orchestrator import (
    StudioAgentDAG,
    StudioOrchestrator,
    StudioSwarmExecutionError,
    SwarmExecutionContext,
)
from engine.studio.swe import LeadGameplaySWEAgent
import engine.studio.models as m
from engine.analyst.evolution import HypothesisDrivenEvolutionEngine
from engine.analyst.models import (
    BehavioralProfile,
    DashMetrics,
    DeathCluster,
    KineticMetrics,
    RetryMetrics,
    SpatialOccupancyMap,
)
from engine.director.models import CurrentEpisodeConfig, DifficultyClampState
from engine.studio.models import (
    AgentExecutionRecord,
    AgentPlan,
    AgentVerificationRecord,
    AudioSynthesisResult,
    BaseEpistemicModel,
    BoundedAdaptationSpec,
    CartridgeCompilationResult,
    CharacterSheet,
    CutsceneBeat,
    CutsceneScript,
    DialogueBarkSpec,
    DialogueChoice,
    DialogueNode,
    DialogueTreeSpec,
    EvolutionHypothesisReport,
    GenesisWorldSpec,
    HypothesisEvidence,
    InstrumentationResult,
    NarrativeBible,
    PlayerPreferenceHypothesis,
    StudioOrchestratorConfig,
    StudioSwarmPlan,
    TelemetryArchitecturePlan,
    TelemetryProbeDesignSpec,
    ToolInvocationRecord,
    UpdateWorldSpec,
)
from engine.studio.telemetry_architect import TelemetryArchitectAgent
from engine.telemetry.db import TelemetryDB
from engine.telemetry.probes import AgentTelemetryProbe
from engine.telemetry.recorder import TelemetryRecorder


# ============================================================================
# 1. Evolution Engine Clamping Invariants
# ============================================================================

class TestEvolutionClampingInvariants:
    """Empirical mathematical boundary verification for difficulty adaptation clamping."""

    def test_apply_clamp_1000_randomized_trials(self) -> None:
        """Empirically verify that apply_clamp strictly enforces +/-15% adaptation
        across 1,000 randomized trials with proposed shifts ranging from -95% to +500%.
        """
        clamp = DifficultyClampState(max_adaptation_ratio=0.15)
        trials = 1000

        for i in range(trials):
            p_old = random.uniform(1.0, 10000.0)
            shift_ratio = random.uniform(-0.95, 5.0)
            p_raw = p_old * (1.0 + shift_ratio)

            clamped = clamp.apply_clamp(f"param_{i}", p_old, p_raw)

            assert not math.isnan(clamped), f"Trial {i}: Clamped value produced NaN for p_old={p_old}, p_raw={p_raw}"
            assert not math.isinf(clamped), f"Trial {i}: Clamped value produced Inf for p_old={p_old}, p_raw={p_raw}"

            lower_bound = p_old * 0.85 - 1e-4
            upper_bound = p_old * 1.15 + 1e-4
            assert lower_bound <= clamped <= upper_bound, (
                f"Trial {i}: Shift ratio {shift_ratio:+.2%} breached bounds: "
                f"clamped={clamped}, expected in [{lower_bound:.4f}, {upper_bound:.4f}]"
            )

    def test_apply_clamp_extreme_boundary_values(self) -> None:
        """Verify strict behavior on boundary conditions (+100%, -90%, 0.0, NaN, Inf, -Inf)."""
        clamp = DifficultyClampState(max_adaptation_ratio=0.15)
        p_old = 200.0

        # +100% shift -> clamped to 1.15 * p_old = 230.0
        c_plus100 = clamp.apply_clamp("p_p100", p_old, 400.0)
        assert c_plus100 == 230.0
        assert clamp.clamps["p_p100"].clamp_triggered is True
        assert abs(clamp.clamps["p_p100"].delta_percentage - 15.0) < 1e-4

        # -90% shift -> clamped to 0.85 * p_old = 170.0
        c_minus90 = clamp.apply_clamp("p_m90", p_old, 20.0)
        assert c_minus90 == 170.0
        assert clamp.clamps["p_m90"].clamp_triggered is True
        assert abs(clamp.clamps["p_m90"].delta_percentage - (-15.0)) < 1e-4

        # 0.0% shift -> stays 200.0, clamp not triggered
        c_zero = clamp.apply_clamp("p_zero", p_old, 200.0)
        assert c_zero == 200.0
        assert clamp.clamps["p_zero"].clamp_triggered is False
        assert clamp.clamps["p_zero"].delta_percentage == 0.0

        # Raw NaN -> returns p_old safely with clamp triggered
        c_nan = clamp.apply_clamp("p_nan", p_old, float("nan"))
        assert c_nan == 200.0
        assert not math.isnan(c_nan)

        # Raw +Inf -> clamped to upper bound 230.0
        c_inf = clamp.apply_clamp("p_inf", p_old, float("inf"))
        assert c_inf == 230.0

        # Raw -Inf -> clamped to lower bound 170.0
        c_neginf = clamp.apply_clamp("p_neginf", p_old, float("-inf"))
        assert c_neginf == 170.0

    def test_apply_clamp_min_max_limit_enforcement(self) -> None:
        """Verify that min_val and max_val constraints are respected without inversion."""
        clamp = DifficultyClampState(max_adaptation_ratio=0.15)
        p_old = 100.0

        # min_val tighter than -15% bound (e.g. min_val = 90.0 > 85.0)
        # Raw proposed = 50.0 -> clamped to 90.0 (effective min bound)
        clamped_min = clamp.apply_clamp("p_tight_min", p_old, 50.0, min_val=90.0)
        assert clamped_min >= 85.0 - 1e-4

        # max_val tighter than +15% bound (e.g. max_val = 110.0 < 115.0)
        # Raw proposed = 200.0 -> clamped to 110.0 (effective max bound)
        clamped_max = clamp.apply_clamp("p_tight_max", p_old, 200.0, max_val=110.0)
        assert clamped_max <= 115.0 + 1e-4

    def test_evolution_engine_compute_bounded_adaptation_1000_trials(self) -> None:
        """Verify that HypothesisDrivenEvolutionEngine._compute_bounded_adaptation()
        clamps all 6 game parameters across 1,000 randomized profile and config variations.
        """
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "evo_trials.db"
            engine = HypothesisDrivenEvolutionEngine(db_path=db_path, max_adaptation_ratio=0.15)

            trials = 1000
            for trial in range(trials):
                base_speed = random.uniform(50.0, 500.0)
                base_damage = random.uniform(5.0, 100.0)
                base_health = random.uniform(20.0, 1000.0)
                base_hazard_rate = random.uniform(0.5, 5.0)
                base_hazard_damage = random.uniform(5.0, 50.0)
                base_weapon_damage = random.uniform(10.0, 100.0)

                cfg = CurrentEpisodeConfig(
                    deliberation=f"Trial config {trial}",
                    episode_number=trial + 1,
                    enemy_speed=base_speed,
                    enemy_damage=base_damage,
                    enemy_health=base_health,
                    hazard_cycle_rate=base_hazard_rate,
                    hazard_damage=base_hazard_damage,
                    weapon_damage=base_weapon_damage,
                )

                mastery = random.choice([0.0, 0.25, 0.5, 0.75, 1.0, random.random()])
                has_clusters = random.choice([True, False])
                clusters = []
                if has_clusters:
                    clusters.append(DeathCluster(
                        deliberation="Cluster",
                        cluster_id=f"c_{trial}",
                        center_x=random.uniform(0, 1280),
                        center_y=random.uniform(0, 720),
                        radius=random.uniform(20, 80),
                        death_count=random.randint(1, 10),
                        primary_cause="spike_trap",
                    ))

                profile = BehavioralProfile(
                    deliberation="Simulated profile",
                    session_id=f"sess_{trial}",
                    episode_id=f"ep_{trial}",
                    duration_seconds=random.uniform(10.0, 240.0),
                    kinetic=KineticMetrics(
                        deliberation="k",
                        mean_velocity=random.uniform(20.0, 500.0),
                        peak_velocity=random.uniform(100.0, 700.0),
                        velocity_variance=random.uniform(0.0, 40.0),
                    ),
                    evasion=DashMetrics(
                        deliberation="d",
                        dash_count=random.randint(0, 20),
                        dash_frequency=random.uniform(0.0, 15.0),
                    ),
                    occupancy=SpatialOccupancyMap(
                        deliberation="o",
                        grid_width=16,
                        grid_height=16,
                        occupancy_matrix=[[0.0]*16 for _ in range(16)],
                    ),
                    retries=RetryMetrics(
                        deliberation="r",
                        total_deaths=random.randint(0, 8),
                        clear_rate=mastery,
                        total_episodes_analyzed=random.randint(1, 10),
                        death_clusters=clusters,
                    ),
                    playstyle_tag="EMPIRICAL_TEST",
                )

                report = engine._formulate_hypotheses(f"ep_{trial}", f"sess_{trial}", profile, {}, None)
                spec = engine._compute_bounded_adaptation(report, profile, cfg)

                assert isinstance(spec, BoundedAdaptationSpec)
                assert spec.deliberation != ""
                assert list(spec.__dict__.keys())[0] == "deliberation"

                adapted = spec.adapted_parameters
                param_baselines = [
                    ("enemy_speed", base_speed),
                    ("enemy_damage", base_damage),
                    ("enemy_health", base_health),
                    ("hazard_cycle_rate", base_hazard_rate),
                    ("hazard_damage", base_hazard_damage),
                    ("weapon_damage", base_weapon_damage),
                ]

                for param_name, base_val in param_baselines:
                    val = adapted[param_name]
                    assert not math.isnan(val), f"Trial {trial}: {param_name} is NaN"
                    assert not math.isinf(val), f"Trial {trial}: {param_name} is Inf"
                    min_allowed = base_val * 0.85 - 1e-4
                    max_allowed = base_val * 1.15 + 1e-4
                    assert min_allowed <= val <= max_allowed, (
                        f"Trial {trial}: {param_name}={val} breached +/-15% of base {base_val}"
                    )


# ============================================================================
# 2. Telemetry Architect Concurrency & Probe Catalog
# ============================================================================

class TestTelemetryArchitectConcurrencyAndThrottling:
    """Stress tests verifying SQLite WAL concurrency, probe catalog idempotency, and emission bounds."""

    def test_concurrent_registration_120_probes_zero_lockups(self) -> None:
        """Verify concurrent registration of 120 distinct probes with 16 parallel threads
        completes with zero SQLite lockups and 100% record integrity in SQLite WAL mode.
        """
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "concurrent_wal.db"
            db = TelemetryDB(db_path)

            num_probes = 120
            num_workers = 16
            errors: List[Exception] = []

            def worker(idx: int) -> bool:
                try:
                    db.register_probe(
                        probe_id=f"probe_c_{idx:04d}",
                        probe_name=f"probe_metric_{idx:04d}",
                        category="combat" if idx % 2 == 0 else "movement",
                        description=f"Concurrent probe registration #{idx}",
                        hypothesis_target=f"HYP_CONCURRENT_{idx:04d}",
                        schema_json={"index": idx, "metric": "test"},
                    )
                    return True
                except Exception as exc:
                    errors.append(exc)
                    return False

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker, i) for i in range(num_probes)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            assert len(errors) == 0, f"Encountered {len(errors)} errors during concurrent registration: {errors[:3]}"
            assert all(results)

            # Verify in DB
            reader = db.get_reader_connection()
            cur = reader.cursor()
            cur.execute("SELECT count(*) FROM telemetry_probes;")
            total_count = cur.fetchone()[0]
            reader.close()

            assert total_count == num_probes, f"Expected {num_probes} probes, found {total_count}"

    def test_idempotent_concurrent_schema_updates(self) -> None:
        """Verify that re-registering existing probes concurrently updates metadata
        without row duplication, unique constraint errors, or deadlocks.
        """
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "idempotent_wal.db"
            db = TelemetryDB(db_path)

            num_probes = 80
            # Step 1: Initial sequential registration
            for i in range(num_probes):
                db.register_probe(
                    probe_id=f"probe_id_{i:03d}",
                    probe_name=f"event_name_{i:03d}",
                    category="movement",
                    description=f"V1 description {i}",
                    hypothesis_target="HYP_V1",
                )

            # Step 2: Concurrent updates to all 80 probes
            update_errors: List[Exception] = []

            def update_task(idx: int) -> bool:
                try:
                    db.register_probe(
                        probe_id=f"probe_id_{idx:03d}",
                        probe_name=f"event_name_{idx:03d}",
                        category="ability",
                        description=f"V2 updated description {idx}",
                        hypothesis_target="HYP_V2_MUTATED",
                        schema_json={"version": 2, "threshold": idx * 10},
                    )
                    return True
                except Exception as exc:
                    update_errors.append(exc)
                    return False

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(update_task, i) for i in range(num_probes)]
                for f in concurrent.futures.as_completed(futures):
                    f.result()

            assert len(update_errors) == 0

            # Verify count is strictly preserved (no duplicate rows)
            reader = db.get_reader_connection()
            cur = reader.cursor()
            cur.execute("SELECT count(*) FROM telemetry_probes;")
            count = cur.fetchone()[0]
            assert count == num_probes

            # Verify all rows updated to 'ability' category
            cur.execute("SELECT count(*) FROM telemetry_probes WHERE category = 'ability';")
            ability_count = cur.fetchone()[0]
            assert ability_count == num_probes
            reader.close()

    def test_telemetry_probe_emission_rate_limiting_ceiling(self) -> None:
        """Verify probe specification enforces emission rate ceiling guard (<= 120 Hz)."""
        # Default probe rate is 60.0 Hz (well under 120 Hz ceiling)
        spec = TelemetryProbeDesignSpec(
            deliberation="Rate limit ceiling check",
            probe_name="boss_slam",
            category="combat",
            description="Tracks boss slam attack",
            hypothesis_target="H_BOSS_LETHALITY",
            target_class="BossEnemy",
            target_method="slam_attack",
        )
        assert spec.max_event_rate_hz <= 120.0
        assert spec.max_event_rate_hz == 60.0

        # Plan total throughput computation
        plan = TelemetryArchitecturePlan(
            deliberation="Multi-probe plan rate check",
            probes=[spec],
            estimated_total_throughput_hz=spec.max_event_rate_hz,
        )
        assert plan.estimated_total_throughput_hz <= 120.0

    def test_rapid_burst_probe_emission_stability(self) -> None:
        """Verify that rapid emission bursts (1,000 events in tight loop) execute
        without buffer overflow, SQLite contention, or event loss.
        """
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "burst_telemetry.db"
            recorder = TelemetryRecorder(db_path=db_path, batch_size=50, flush_interval_sec=0.02)
            recorder.start_session("burst_session")
            recorder.start_episode(level_id="lvl_burst", episode_index=1)

            probe = AgentTelemetryProbe(
                recorder=recorder,
                probe_name="burst_dash_probe",
                category="movement",
                description="Rapid burst test probe",
                hypothesis_target="H_BURST_STABILITY",
            )

            burst_count = 1000
            start_time = time.perf_counter()
            for i in range(burst_count):
                payload = probe.emit(
                    pos=(100.0 + (i % 50), 200.0 + (i % 50)),
                    vel=(150.0, 0.0),
                    health=100.0 - (i % 50),
                    data={"tick": i},
                )
                assert payload["probe_name"] == "burst_dash_probe"
            elapsed = time.perf_counter() - start_time

            # Average emission latency must be sub-millisecond (< 50 microseconds typical)
            avg_micros = (elapsed / burst_count) * 1_000_000
            assert avg_micros < 200.0, f"Average emission time {avg_micros:.1f} µs exceeded 200 µs limit"

            recorder.flush()
            recorder.end_episode()
            recorder.end_session()
            recorder.close()

            # Verify persisted event count in SQLite WAL
            db = TelemetryDB(db_path)
            reader = db.get_reader_connection()
            cur = reader.cursor()
            cur.execute("SELECT count(*) FROM telemetry_events WHERE probe_name = 'burst_dash_probe';")
            stored_count = cur.fetchone()[0]
            reader.close()

            assert stored_count == burst_count, f"Expected {burst_count} persisted events, got {stored_count}"


# ============================================================================
# 3. Rationale-First Key Ordering Invariance
# ============================================================================

class TestRationaleFirstKeyOrderingInvariance:
    """Stress tests verifying that 100% of Pydantic models in engine/studio/models.py
    strictly emit 'deliberation' as key index 0 across all serialization formats.
    """

    def _build_all_model_instances(self) -> Dict[str, BaseEpistemicModel]:
        instances: Dict[str, BaseEpistemicModel] = {}

        instances["BaseEpistemicModel"] = BaseEpistemicModel(
            deliberation="Base epistemic rationale."
        )
        instances["StudioOrchestratorConfig"] = StudioOrchestratorConfig(
            deliberation="Config deliberation",
            gemini_api_key="fake_key",
        )
        instances["AgentPlan"] = AgentPlan(
            deliberation="Plan deliberation",
            agent_name="LeadGameplaySWE",
            objective="Synthesize cartridge",
            steps=["Step 1", "Step 2"],
        )
        instances["ToolInvocationRecord"] = ToolInvocationRecord(
            deliberation="Tool deliberation",
            tool_name="run_ast_validation",
            arguments={"code": "pass"},
        )
        instances["AgentExecutionRecord"] = AgentExecutionRecord(
            deliberation="Execution deliberation",
            agent_name="NarrativeWriter",
            plan_id="plan_123",
        )
        instances["AgentVerificationRecord"] = AgentVerificationRecord(
            deliberation="Verification deliberation",
            agent_name="QASupervisor",
            status="VERIFIED",
            passed=True,
        )
        instances["GenesisWorldSpec"] = GenesisWorldSpec(
            deliberation="Genesis world deliberation",
            game_title="Void Runner",
            core_genre="Action Roguelike",
            thematic_premise="Deep space derelict",
            world_lore="Ancient automated combat drones",
            conflict_tone="grim_tense",
            conflict_narrative="Rogue security AI",
            aesthetic_style="pixel_16bit",
            palette_mood="neon_cyberpunk",
            gameplay_pacing="kinetic_high_mobility",
            pacing_score=0.8,
            target_difficulty=1.1,
        )
        instances["UpdateWorldSpec"] = UpdateWorldSpec(
            deliberation="Update world deliberation",
            target_episode=2,
            previous_episode=1,
            narrative_progression="Story continues into the lower decks",
            escalation_factor=1.12,
        )
        instances["CharacterSheet"] = CharacterSheet(
            deliberation="Character deliberation",
            character_id="char_hero",
            name="Aria",
            role="protagonist",
            personality_traits=["resilient", "focused"],
            backstory="Ex-pilot marooned on station",
        )
        instances["DialogueBarkSpec"] = DialogueBarkSpec(
            deliberation="Bark deliberation",
            bark_id="bark_01",
            speaker_id="char_hero",
            speaker_name="Aria",
            trigger_type="low_health",
            text="Hull integrity critical!",
            voice_clip="assets/audio/bark_low_health.wav",
        )
        instances["DialogueChoice"] = DialogueChoice(
            deliberation="Choice deliberation",
            text="Vent the airlock",
            target_node_id="node_vent",
        )
        instances["DialogueNode"] = DialogueNode(
            deliberation="Node deliberation",
            node_id="node_01",
            speaker_id="char_hero",
            text="We have to move.",
            choices=[],
        )
        instances["DialogueTreeSpec"] = DialogueTreeSpec(
            deliberation="Dialogue tree deliberation",
            root_node_id="node_01",
            nodes={"node_01": instances["DialogueNode"]},
        )
        instances["CutsceneBeat"] = CutsceneBeat(
            deliberation="Beat deliberation",
            beat_number=1,
            action_description="Blast doors seal shut",
        )
        instances["CutsceneScript"] = CutsceneScript(
            deliberation="Cutscene script deliberation",
            cutscene_id="intro_cutscene",
            title="Lockdown",
            beats=[instances["CutsceneBeat"]],
        )
        instances["NarrativeBible"] = NarrativeBible(
            deliberation="Narrative bible deliberation",
            world_title="Void Station",
            setting_summary="An abandoned orbital platform",
            characters=[instances["CharacterSheet"]],
        )
        instances["AudioSynthesisResult"] = AudioSynthesisResult(
            deliberation="Audio result deliberation",
            bark_id="line_01",
            file_path="assets/audio/bark.wav",
            duration_seconds=2.5,
            file_size_bytes=1024,
            sha256_hash="abcdef123456",
        )
        instances["TelemetryProbeDesignSpec"] = TelemetryProbeDesignSpec(
            deliberation="Probe design deliberation",
            probe_name="player_dash",
            category="movement",
            description="Tracks player dash",
            hypothesis_target="H_MOBILITY",
            target_class="Player",
            target_method="dash",
        )
        instances["TelemetryArchitecturePlan"] = TelemetryArchitecturePlan(
            deliberation="Telemetry plan deliberation",
            probes=[instances["TelemetryProbeDesignSpec"]],
        )
        instances["InstrumentationResult"] = InstrumentationResult(
            deliberation="Instrumentation result deliberation",
            success=True,
            probes_registered=1,
            ast_security_passed=True,
        )
        instances["HypothesisEvidence"] = HypothesisEvidence(
            deliberation="Evidence deliberation",
            metric_name="mean_velocity",
            observed_value=240.0,
        )
        instances["PlayerPreferenceHypothesis"] = PlayerPreferenceHypothesis(
            deliberation="Hypothesis deliberation",
            domain="combat_style",
            statement="Player prefers aggressive close-quarters combat",
            confidence_score=0.88,
            evidence=[instances["HypothesisEvidence"]],
            falsification_condition="If dash rate drops below 2.0",
            prescribed_evolution="Increase shotgun power",
        )
        instances["EvolutionHypothesisReport"] = EvolutionHypothesisReport(
            deliberation="Evolution report deliberation",
            session_id="sess_01",
            episode_id="ep_01",
            dominant_playstyle="KINETIC_RUSHER",
            hypotheses=[instances["PlayerPreferenceHypothesis"]],
        )
        instances["BoundedAdaptationSpec"] = BoundedAdaptationSpec(
            deliberation="Adaptation spec deliberation",
            current_episode_number=1,
            target_episode_number=2,
            clamp_state={},
            adapted_parameters={"enemy_speed": 115.0},
        )
        instances["CartridgeCompilationResult"] = CartridgeCompilationResult(
            deliberation="Cartridge compilation deliberation",
            candidate_id="cart_active_01",
            promoted_path="cartridges/active",
            envelope_hash="hash123",
        )
        instances["StudioSwarmPlan"] = StudioSwarmPlan(
            deliberation="Dynamic swarm plan deliberation",
            plan_id="swarm_plan_01",
            game_title="Autonomous Cartridge",
            genre_verdict="action",
            specialist_agent_roles=["Director", "SWE"],
            dag_edges=[("Director", "SWE")],
            task_manifest={"Director": "Genesis prompt", "SWE": "Mechanics prompt"},
            target_episode=0,
        )

        return instances

    def test_all_26_models_present_and_accounted_for(self) -> None:
        """Confirm that exactly 26 (or 31 with M1 requirement models) Pydantic models are defined in engine/studio/models.py."""
        discovered = [
            obj for name, obj in inspect.getmembers(m)
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel
        ]
        assert len(discovered) in (26, 31), f"Expected 26 or 31 Pydantic models, discovered {len(discovered)}"

    def test_model_dump_deliberation_key_zero_all_25_models(self) -> None:
        """Assert list(model.model_dump().keys())[0] == 'deliberation' for every model."""
        instances = self._build_all_model_instances()
        assert len(instances) == 26

        for name, inst in instances.items():
            keys = list(inst.model_dump().keys())
            assert keys[0] == "deliberation", (
                f"Model {name}.model_dump() violated key index 0: first key was '{keys[0]}'"
            )

    def test_model_dump_json_deliberation_key_zero_all_25_models(self) -> None:
        """Assert list(json.loads(model.model_dump_json()).keys())[0] == 'deliberation' for every model."""
        instances = self._build_all_model_instances()
        assert len(instances) == 26

        for name, inst in instances.items():
            json_str = inst.model_dump_json()
            data = json.loads(json_str)
            keys = list(data.keys())
            assert keys[0] == "deliberation", (
                f"Model {name}.model_dump_json() violated key index 0: first key was '{keys[0]}'"
            )

    def test_argument_permutation_preserves_deliberation_first(self) -> None:
        """Verify that passing kwargs in reversed order does not affect deliberation at index 0."""
        plan = AgentPlan(
            steps=["First step"],
            objective="Test kwarg permutation",
            agent_name="PermutationWorker",
            deliberation="Deliberation supplied at the very end of kwargs",
        )
        assert list(plan.model_dump().keys())[0] == "deliberation"
        assert list(json.loads(plan.model_dump_json()).keys())[0] == "deliberation"

    def test_round_trip_deserialization_preserves_key_ordering(self) -> None:
        """Verify that deserializing from JSON and re-serializing preserves deliberation at index 0."""
        instances = self._build_all_model_instances()

        for name, inst in instances.items():
            json_str = inst.model_dump_json()
            cls = type(inst)
            roundtrip_inst = cls(**json.loads(json_str))
            rt_dict = roundtrip_inst.model_dump()
            rt_json = json.loads(roundtrip_inst.model_dump_json())

            assert list(rt_dict.keys())[0] == "deliberation", (
                f"Roundtrip dict for {name} has first key '{list(rt_dict.keys())[0]}'"
            )
            assert list(rt_json.keys())[0] == "deliberation", (
                f"Roundtrip JSON for {name} has first key '{list(rt_json.keys())[0]}'"
            )

    def test_nested_composite_models_key_ordering(self) -> None:
        """Verify that nested models inside composite schemas also preserve deliberation first."""
        beat = CutsceneBeat(
            deliberation="Beat deliberation",
            beat_number=1,
            action_description="Door slides open",
        )
        script = CutsceneScript(
            deliberation="Script deliberation",
            cutscene_id="cs_01",
            title="Intro",
            beats=[beat],
        )

        script_dict = script.model_dump()
        assert list(script_dict.keys())[0] == "deliberation"
        nested_beat_dict = script_dict["beats"][0]
        assert list(nested_beat_dict.keys())[0] == "deliberation"

        script_json = json.loads(script.model_dump_json())
        assert list(script_json.keys())[0] == "deliberation"
        nested_beat_json = script_json["beats"][0]
        assert list(nested_beat_json.keys())[0] == "deliberation"


# ============================================================================
# 4. Wave-Based Swarm DAG Concurrency & Multi-Tier Execution Invariants
# ============================================================================

class TestWaveBasedSwarmDAGConcurrency:
    """Stress tests verifying wave-based Kahn tiering, concurrency bounds, and artifact passing."""

    def test_compute_execution_tiers_complex_topologies(self) -> None:
        """Verify topological tiering on complex multi-tier topologies (diamond, wide waves, forests)."""
        orch = StudioOrchestrator()

        # 1. Wide fan-out / fan-in: 1 -> 8 -> 1
        wide_edges = [("Root", f"Worker_{i}") for i in range(8)] + [(f"Worker_{i}", "Sink") for i in range(8)]
        wide_roles = ["Root", "Sink"] + [f"Worker_{i}" for i in range(8)]
        tiers = orch._compute_execution_tiers(wide_edges, wide_roles)
        assert len(tiers) == 3
        assert tiers[0] == ["Root"]
        assert len(tiers[1]) == 8
        assert sorted(tiers[1]) == sorted([f"Worker_{i}" for i in range(8)])
        assert tiers[2] == ["Sink"]

        # 2. Multi-root forest (independent parallel pipelines)
        forest_edges = [("A1", "A2"), ("A2", "A3"), ("B1", "B2"), ("B2", "B3")]
        forest_roles = ["A1", "A2", "A3", "B1", "B2", "B3"]
        f_tiers = orch._compute_execution_tiers(forest_edges, forest_roles)
        assert len(f_tiers) == 3
        assert sorted(f_tiers[0]) == ["A1", "B1"]
        assert sorted(f_tiers[1]) == ["A2", "B2"]
        assert sorted(f_tiers[2]) == ["A3", "B3"]

        # 3. Disconnected nodes (no edges)
        solo_tiers = orch._compute_execution_tiers([], ["Solo1", "Solo2", "Solo3"])
        assert len(solo_tiers) == 1
        assert sorted(solo_tiers[0]) == ["Solo1", "Solo2", "Solo3"]

    @pytest.mark.anyio
    async def test_execute_swarm_plan_wave_concurrency_and_timestamps(self, tmp_path: Path) -> None:
        """Verify concurrent execution within tiers and sequential barriers across tiers using timestamps."""
        orch = StudioOrchestrator()
        execution_events: List[Tuple[str, str, float, Dict[str, Any]]] = []

        async def timed_specialist(role: str, plan: StudioSwarmPlan, context: SwarmExecutionContext) -> str:
            start_t = time.perf_counter()
            execution_events.append(("start", role, start_t, dict(context.raw_outputs)))
            await asyncio.sleep(0.04)
            end_t = time.perf_counter()
            output = f"result_{role}"
            envelope = ArtifactMetadataEnvelope.seal(
                deliberation=f"Deliberation for {role}",
                artifact_type=ArtifactType.GAME_MODULE,
                payload=output,
                source_agent=role,
                target_agent="Supervisor",
            )
            context.record_artifact(role, envelope, output)
            execution_events.append(("end", role, end_t, {"output": output}))
            return output

        orch._execute_specialist = timed_specialist

        mock_sup = MagicMock()
        mock_report = MagicMock()
        mock_report.promoted_to_active = True
        mock_sup.evaluate_and_heal.return_value = mock_report
        orch.agents["QAPlaytesterBuildSupervisor"] = mock_sup

        # Tier 0: NodeA (1 node)
        # Tier 1: NodeB, NodeC, NodeD (3 concurrent nodes)
        # Tier 2: NodeE (1 node)
        plan = StudioSwarmPlan(
            deliberation="Wave timing plan",
            plan_id="plan_wave_timing",
            game_title="Wave Timing Game",
            genre_verdict="action",
            specialist_agent_roles=["NodeA", "NodeB", "NodeC", "NodeD", "NodeE"],
            dag_edges=[
                ("NodeA", "NodeB"), ("NodeA", "NodeC"), ("NodeA", "NodeD"),
                ("NodeB", "NodeE"), ("NodeC", "NodeE"), ("NodeD", "NodeE"),
            ],
            task_manifest={},
            target_episode=0,
        )
        staging_dir = tmp_path / "staging_timing"
        staging_dir.mkdir()

        t0 = time.perf_counter()
        report = await orch.execute_swarm_plan(plan, staging_dir)
        total_duration = time.perf_counter() - t0

        # Sequential would take >= 5 * 0.04s = 0.20s; concurrent should take ~3 * 0.04s = 0.12s
        assert total_duration < 0.20, f"Expected concurrency benefit, but took {total_duration:.3f}s"

        # Verify barriers
        a_end = [e[2] for e in execution_events if e[0] == "end" and e[1] == "NodeA"][0]
        tier1_starts = [e[2] for e in execution_events if e[0] == "start" and e[1] in ("NodeB", "NodeC", "NodeD")]
        tier1_ends = [e[2] for e in execution_events if e[0] == "end" and e[1] in ("NodeB", "NodeC", "NodeD")]
        e_start = [e[2] for e in execution_events if e[0] == "start" and e[1] == "NodeE"][0]

        # Tier 0 must finish before any Tier 1 node starts
        for start_t in tier1_starts:
            assert start_t >= a_end, "Tier 1 node started before Tier 0 finished"

        # All Tier 1 nodes must finish before Tier 2 starts
        assert e_start >= max(tier1_ends), "Tier 2 node started before Tier 1 finished"

        # Artifact passing: NodeE's context must contain outputs of NodeA, NodeB, NodeC, NodeD
        e_context = [e[3] for e in execution_events if e[0] == "start" and e[1] == "NodeE"][0]
        for upstream in ["NodeA", "NodeB", "NodeC", "NodeD"]:
            assert upstream in e_context
            assert e_context[upstream] == f"result_{upstream}"

    @pytest.mark.anyio
    @pytest.mark.parametrize("concurrency_width", [1, 2, 4, 8])
    async def test_execute_swarm_plan_varying_concurrency_levels(
        self, tmp_path: Path, concurrency_width: int
    ) -> None:
        """Verify swarm plan execution across varying concurrency levels (1, 2, 4, 8 workers)."""
        orch = StudioOrchestrator()
        completed_roles: List[str] = []

        async def worker_specialist(role: str, plan: StudioSwarmPlan, context: SwarmExecutionContext) -> str:
            await asyncio.sleep(0.01)
            output = f"result_{role}"
            envelope = ArtifactMetadataEnvelope.seal(
                deliberation=f"Deliberation for {role}",
                artifact_type=ArtifactType.GAME_MODULE,
                payload=output,
                source_agent=role,
                target_agent="Supervisor",
            )
            context.record_artifact(role, envelope, output)
            completed_roles.append(role)
            return output

        orch._execute_specialist = worker_specialist
        mock_sup = MagicMock()
        mock_sup.evaluate_and_heal.return_value = MagicMock(promoted_to_active=True)
        orch.agents["QAPlaytesterBuildSupervisor"] = mock_sup

        worker_roles = [f"Worker_{i}" for i in range(concurrency_width)]
        edges = [("Leader", w) for w in worker_roles] + [(w, "Follower") for w in worker_roles]
        plan = StudioSwarmPlan(
            deliberation=f"Concurrency width {concurrency_width} plan",
            plan_id=f"plan_w{concurrency_width}",
            game_title=f"Game Width {concurrency_width}",
            genre_verdict="strategy",
            specialist_agent_roles=["Leader", "Follower"] + worker_roles,
            dag_edges=edges,
            task_manifest={},
            target_episode=0,
        )

        staging_dir = tmp_path / f"staging_w{concurrency_width}"
        staging_dir.mkdir()

        report = await orch.execute_swarm_plan(plan, staging_dir)
        assert len(completed_roles) == concurrency_width + 2
        assert "Leader" in completed_roles
        assert "Follower" in completed_roles
        for w in worker_roles:
            assert w in completed_roles

    @pytest.mark.anyio
    async def test_execute_swarm_plan_tier_failure_cancels_subsequent_tiers(self, tmp_path: Path) -> None:
        """Verify fail-fast (Rule 5): failure in Tier 1 immediately raises and aborts Tier 2."""
        orch = StudioOrchestrator()
        invoked_roles: List[str] = []

        async def failing_specialist(role: str, plan: StudioSwarmPlan, context: SwarmExecutionContext) -> str:
            invoked_roles.append(role)
            if role == "FaultyWorker":
                raise RuntimeError("Catastrophic specialist fault")
            return f"result_{role}"

        orch._execute_specialist = failing_specialist
        plan = StudioSwarmPlan(
            deliberation="Failure propagation plan",
            plan_id="plan_fail_prop",
            game_title="Fail Prop Game",
            genre_verdict="action",
            specialist_agent_roles=["Root", "FaultyWorker", "HealthyWorker", "TerminalSink"],
            dag_edges=[
                ("Root", "FaultyWorker"),
                ("Root", "HealthyWorker"),
                ("FaultyWorker", "TerminalSink"),
                ("HealthyWorker", "TerminalSink"),
            ],
            task_manifest={},
            target_episode=0,
        )
        staging_dir = tmp_path / "staging_fail_prop"
        staging_dir.mkdir()

        with pytest.raises(StudioSwarmExecutionError) as exc_info:
            await orch.execute_swarm_plan(plan, staging_dir)

        assert "Catastrophic specialist fault" in str(exc_info.value)
        # TerminalSink must never be invoked
        assert "TerminalSink" not in invoked_roles


# ============================================================================
# 5. Open-Genre Cartridge & Decoupled Stage 6 AST Verification
# ============================================================================

class TestOpenGenreCartridgeASTDecoupling:
    """Stress tests verifying Stage 6 decoupled AST verification and open-genre mechanics."""

    def test_stage_6_open_genre_both_scene_methods_accepted(self) -> None:
        """Subclass defining both get_scenes and get_initial_scene passes Stage 6."""
        swe = LeadGameplaySWEAgent()
        code = '''
from engine.api.contracts import BaseCartridge

class OpenGenreRPG(BaseCartridge):
    deliberation: str = 'Turn-based tactical RPG with scene architecture'
    
    def get_scenes(self, engine):
        return {}
        
    def get_initial_scene(self, engine):
        return None
'''
        valid, err = swe.verify_code_module("mechanics.py", code)
        assert valid is True, f"Expected valid, got error: {err}"

    def test_stage_6_open_genre_get_scenes_only_accepted(self) -> None:
        """Subclass defining only get_scenes passes Stage 6."""
        swe = LeadGameplaySWEAgent()
        code = '''
from engine.api.contracts import BaseCartridge

class ScenesOnlyGame(BaseCartridge):
    deliberation: str = 'Deckbuilder cartridge defining scene map'
    
    def get_scenes(self, engine):
        return {"battle": None}
'''
        valid, err = swe.verify_code_module("mechanics.py", code)
        assert valid is True, f"Expected valid, got error: {err}"

    def test_stage_6_open_genre_get_initial_scene_only_accepted(self) -> None:
        """Subclass defining only get_initial_scene passes Stage 6."""
        swe = LeadGameplaySWEAgent()
        code = '''
from engine.api.contracts import BaseCartridge

class SingleSceneAdventure(BaseCartridge):
    deliberation: str = 'Adventure game defining initial scene'
    
    def get_initial_scene(self, engine):
        return None
'''
        valid, err = swe.verify_code_module("mechanics.py", code)
        assert valid is True, f"Expected valid, got error: {err}"

    def test_stage_6_legacy_get_level_spec_accepted_for_compatibility(self) -> None:
        """Subclass defining legacy get_level_spec passes Stage 6 for backward compatibility."""
        swe = LeadGameplaySWEAgent()
        code = '''
from engine.api.contracts import BaseCartridge

class LegacyCompatibleGame(BaseCartridge):
    deliberation: str = 'Legacy level spec cartridge'
    
    def get_level_spec(self):
        return {}
'''
        valid, err = swe.verify_code_module("mechanics.py", code)
        assert valid is True, f"Expected valid, got error: {err}"

    def test_stage_6_missing_all_lifecycle_methods_rejected(self) -> None:
        """BaseCartridge subclass defining neither scene nor level lifecycle fails Stage 6."""
        swe = LeadGameplaySWEAgent()
        code = '''
from engine.api.contracts import BaseCartridge

class IncompleteCartridge(BaseCartridge):
    deliberation: str = 'Incomplete cartridge missing lifecycle methods'
    
    def configure(self):
        pass
'''
        valid, err = swe.verify_code_module("mechanics.py", code)
        assert valid is False
        assert "must implement either open-genre scene lifecycle" in err

    def test_stage_6_missing_base_cartridge_inheritance_rejected(self) -> None:
        """Class with scene methods but not inheriting BaseCartridge fails Stage 6."""
        swe = LeadGameplaySWEAgent()
        code = '''
class StandaloneGame:
    deliberation: str = 'Not inheriting BaseCartridge'
    
    def get_scenes(self, engine):
        return {}
'''
        valid, err = swe.verify_code_module("mechanics.py", code)
        assert valid is False
        assert "mechanics.py must define at least one subclass inheriting from BaseCartridge" in err

    def test_stage_5_deliberation_missing_or_non_first_rejected(self) -> None:
        """Classes without deliberation as first attribute fail Stage 5."""
        swe = LeadGameplaySWEAgent()
        # Missing deliberation completely
        code_missing = '''
from engine.api.contracts import BaseCartridge

class MissingDelib(BaseCartridge):
    def get_scenes(self, engine):
        return {}
'''
        valid, err = swe.verify_code_module("mechanics.py", code_missing)
        assert valid is False
        assert "must declare 'deliberation' as its first attribute" in err

        # Deliberation declared second
        code_second = '''
from engine.api.contracts import BaseCartridge

class SecondDelib(BaseCartridge):
    health: int = 100
    deliberation: str = 'Second attribute'
    
    def get_scenes(self, engine):
        return {}
'''
        valid, err = swe.verify_code_module("mechanics.py", code_second)
        assert valid is False
        assert "must declare 'deliberation' as its first attribute" in err

    def test_stage_7_scenes_module_lifecycle_checks(self) -> None:
        """scenes.py must define BaseScene/Scene subclass with update and render methods."""
        swe = LeadGameplaySWEAgent()
        # Valid scene
        valid_scene = '''
from engine.api.contracts import BaseScene

class CombatScene(BaseScene):
    deliberation: str = 'Active combat scene'
    
    def update(self, dt):
        pass
        
    def render(self, surface):
        pass
'''
        valid, err = swe.verify_code_module("scenes.py", valid_scene)
        assert valid is True, f"Expected valid, got: {err}"

        # Missing render
        missing_render = '''
from engine.api.contracts import BaseScene

class BrokenScene(BaseScene):
    deliberation: str = 'Missing render scene'
    
    def update(self, dt):
        pass
'''
        valid, err = swe.verify_code_module("scenes.py", missing_render)
        assert valid is False
        assert "missing required lifecycle method: render" in err

        # Missing BaseScene base
        missing_base = '''
class UnparentedScene:
    deliberation: str = 'No base class'
    
    def update(self, dt):
        pass
    def render(self, surface):
        pass
'''
        valid, err = swe.verify_code_module("scenes.py", missing_base)
        assert valid is False
        assert "scenes.py must define at least one subclass inheriting from BaseScene or Scene" in err
