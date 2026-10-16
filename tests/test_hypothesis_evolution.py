"""tests/test_hypothesis_evolution.py
Tests for Feature 23: Hypothesis-Driven Evolution Engine.
Verifies SQLite WAL custom probe aggregation, falsifiable hypothesis synthesis,
strict +/-15% adaptation clamping, and Rationale-First compliance.
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import pytest

from engine.analyst.evolution import HypothesisDrivenEvolutionEngine
from engine.analyst.models import (
    BehavioralProfile,
    DeathCluster,
    DashMetrics,
    KineticMetrics,
    RetryMetrics,
    SpatialOccupancyMap,
)
from engine.director.models import CurrentEpisodeConfig
from engine.studio.models import (
    BoundedAdaptationSpec,
    EvolutionHypothesisReport,
    PlayerPreferenceHypothesis,
)
from engine.telemetry.db import TelemetryDB
from engine.telemetry.recorder import TelemetryRecorder


class TestHypothesisDrivenEvolutionEngine:
    """Test suite for Feature 23 Hypothesis-Driven Evolution Engine."""

    def test_evaluate_episode_generates_rationale_first_models(self, tmp_path: Path) -> None:
        db_path = tmp_path / "evolution_test.db"
        engine = HypothesisDrivenEvolutionEngine(db_path=db_path)

        report, adaptation = engine.evaluate_episode("test_ep_01")

        assert isinstance(report, EvolutionHypothesisReport)
        assert isinstance(adaptation, BoundedAdaptationSpec)

        # Rationale-First: deliberation strictly field index 0
        assert report.deliberation != ""
        assert list(report.__dict__.keys())[0] == "deliberation"

        assert adaptation.deliberation != ""
        assert list(adaptation.__dict__.keys())[0] == "deliberation"

        for hyp in report.hypotheses:
            assert isinstance(hyp, PlayerPreferenceHypothesis)
            assert hyp.deliberation != ""
            assert list(hyp.__dict__.keys())[0] == "deliberation"
            assert 0.0 <= hyp.confidence_score <= 1.0
            assert hyp.statement != ""
            assert hyp.falsification_condition != ""

    def test_bounded_adaptation_enforces_15_percent_clamp(self, tmp_path: Path) -> None:
        db_path = tmp_path / "evolution_clamp.db"
        engine = HypothesisDrivenEvolutionEngine(db_path=db_path, max_adaptation_ratio=0.15)

        base_cfg = CurrentEpisodeConfig(
            deliberation="Baseline config for testing clamp",
            episode_number=1,
            enemy_speed=200.0,
            enemy_health=100.0,
            enemy_damage=20.0,
            hazard_cycle_rate=1.0,
            hazard_damage=15.0,
            weapon_damage=25.0,
        )

        # High mastery scenario (clear_rate = 1.0) -> increases enemy difficulty
        high_mastery_profile = BehavioralProfile(
            deliberation="High mastery player profile",
            session_id="s1",
            episode_id="ep_01",
            duration_seconds=60.0,
            kinetic=KineticMetrics(deliberation="k", mean_velocity=250.0, peak_velocity=350.0, velocity_variance=30.0),
            evasion=DashMetrics(deliberation="d", dash_count=15, dash_frequency=15.0),
            occupancy=SpatialOccupancyMap(deliberation="o", grid_width=16, grid_height=16, occupancy_matrix=[[0.0]*16 for _ in range(16)]),
            retries=RetryMetrics(deliberation="r", total_deaths=0, clear_rate=1.0, total_episodes_analyzed=1),
            playstyle_tag="KINETIC_RUSHER",
        )

        report = engine._formulate_hypotheses("ep_01", "s1", high_mastery_profile, {}, None)
        adaptation = engine._compute_bounded_adaptation(report, high_mastery_profile, base_cfg)

        adapted = adaptation.adapted_parameters
        assert adapted["enemy_speed"] <= base_cfg.enemy_speed * 1.15 + 1e-4
        assert adapted["enemy_speed"] >= base_cfg.enemy_speed
        assert adapted["enemy_health"] <= base_cfg.enemy_health * 1.15 + 1e-4
        assert adapted["hazard_damage"] <= base_cfg.hazard_damage * 1.15 + 1e-4

        # Low mastery scenario with death clusters -> decreases enemy difficulty
        low_mastery_profile = BehavioralProfile(
            deliberation="Struggling player profile",
            session_id="s2",
            episode_id="ep_01",
            duration_seconds=120.0,
            kinetic=KineticMetrics(deliberation="k", mean_velocity=80.0, peak_velocity=120.0, velocity_variance=15.0),
            evasion=DashMetrics(deliberation="d", dash_count=1, dash_frequency=0.5),
            occupancy=SpatialOccupancyMap(deliberation="o", grid_width=16, grid_height=16, occupancy_matrix=[[0.0]*16 for _ in range(16)]),
            retries=RetryMetrics(
                deliberation="r",
                total_deaths=5,
                clear_rate=0.1,
                total_episodes_analyzed=6,
                death_clusters=[
                    DeathCluster(
                        deliberation="Cluster",
                        cluster_id="c1",
                        center_x=400.0,
                        center_y=300.0,
                        radius=45.0,
                        death_count=4,
                        primary_cause="spike_hazard",
                    )
                ]
            ),
            playstyle_tag="TACTICAL_SENTINEL",
        )

        report_low = engine._formulate_hypotheses("ep_01", "s2", low_mastery_profile, {}, None)
        adaptation_low = engine._compute_bounded_adaptation(report_low, low_mastery_profile, base_cfg)

        adapted_low = adaptation_low.adapted_parameters
        assert adapted_low["enemy_speed"] >= base_cfg.enemy_speed * 0.85 - 1e-4
        assert adapted_low["enemy_speed"] <= base_cfg.enemy_speed
        assert adapted_low["enemy_health"] >= base_cfg.enemy_health * 0.85 - 1e-4
        assert adapted_low["weapon_damage"] <= base_cfg.weapon_damage * 1.15 + 1e-4

    def test_query_custom_probes_from_sqlite_wal(self, tmp_path: Path) -> None:
        db_path = tmp_path / "custom_probes_wal.db"
        recorder = TelemetryRecorder(db_path=db_path, batch_size=1, flush_interval_sec=0.01)
        recorder.start_session("probe_wal_session")
        ep_id = recorder.start_episode(level_id="lvl_01", episode_index=1)

        # Emit custom probe events
        recorder.emit_custom_probe(
            probe_name="player_dash_mobility",
            pos=(120.0, 360.0),
            vel=(300.0, 0.0),
            health=100.0,
            category="movement",
            data={"dash_distance": 180.0},
        )
        recorder.emit_custom_probe(
            probe_name="player_dash_mobility",
            pos=(160.0, 360.0),
            vel=(280.0, 0.0),
            health=90.0,
            category="movement",
            data={"dash_distance": 170.0},
        )
        recorder.flush()

        engine = HypothesisDrivenEvolutionEngine(db_path=db_path)
        probe_summaries = engine._query_custom_probes(ep_id)

        assert "player_dash_mobility" in probe_summaries
        dash_summary = probe_summaries["player_dash_mobility"]
        assert dash_summary["count"] == 2
        assert dash_summary["mean_pos"][0] == 140.0
        assert dash_summary["mean_health"] == 95.0

        recorder.end_episode()
        recorder.end_session()
        recorder.close()

    def test_weapon_preference_hypothesis_deduction(self, tmp_path: Path) -> None:
        db_path = tmp_path / "weapon_hyp_test.db"
        engine = HypothesisDrivenEvolutionEngine(db_path=db_path)

        dummy_profile = BehavioralProfile(
            deliberation="Profile",
            session_id="s_wp",
            episode_id="ep_01",
            duration_seconds=90.0,
            kinetic=KineticMetrics(deliberation="k", mean_velocity=150.0, peak_velocity=200.0, velocity_variance=20.0),
            evasion=DashMetrics(deliberation="d", dash_count=5, dash_frequency=3.0),
            occupancy=SpatialOccupancyMap(deliberation="o", grid_width=16, grid_height=16, occupancy_matrix=[[0.0]*16 for _ in range(16)]),
            retries=RetryMetrics(deliberation="r", total_deaths=1, clear_rate=0.8, total_episodes_analyzed=2),
            playstyle_tag="BALANCED",
        )

        probe_data = {
            "weapon_railgun": {"count": 45, "mean_pos": (200.0, 300.0), "mean_vel": (0.0, 0.0), "mean_health": 80.0},
            "weapon_shotgun": {"count": 5, "mean_pos": (220.0, 310.0), "mean_vel": (0.0, 0.0), "mean_health": 80.0},
        }

        report = engine._formulate_hypotheses("ep_01", "s_wp", dummy_profile, probe_data, None)

        weapon_hyp = [h for h in report.hypotheses if h.domain == "weapon_preference"]
        assert len(weapon_hyp) == 1
        h = weapon_hyp[0]
        assert "weapon_railgun" in h.statement
        assert h.status == "CONFIRMED"
        assert h.confidence_score >= 0.80
        assert "weapon_railgun" in h.falsification_condition
        assert len(h.evidence) >= 1

    def test_death_cluster_hypothesis_deduction(self, tmp_path: Path) -> None:
        db_path = tmp_path / "death_cluster_test.db"
        engine = HypothesisDrivenEvolutionEngine(db_path=db_path)

        dc = DeathCluster(
            deliberation="Cluster near spike hazard",
            cluster_id="cluster_acid_01",
            center_x=640.0,
            center_y=360.0,
            radius=50.0,
            death_count=5,
            primary_cause="acid_pool",
        )
        struggling_profile = BehavioralProfile(
            deliberation="Profile with death cluster",
            session_id="s_dc",
            episode_id="ep_02",
            duration_seconds=120.0,
            kinetic=KineticMetrics(deliberation="k", mean_velocity=110.0, peak_velocity=150.0, velocity_variance=15.0),
            evasion=DashMetrics(deliberation="d", dash_count=2, dash_frequency=1.0),
            occupancy=SpatialOccupancyMap(deliberation="o", grid_width=16, grid_height=16, occupancy_matrix=[[0.0]*16 for _ in range(16)]),
            retries=RetryMetrics(deliberation="r", total_deaths=5, clear_rate=0.0, total_episodes_analyzed=5, death_clusters=[dc]),
            playstyle_tag="TACTICAL_SENTINEL",
        )

        report = engine._formulate_hypotheses("ep_02", "s_dc", struggling_profile, {}, None)
        hazard_hyp = [h for h in report.hypotheses if h.domain == "hazard_fatigue"]
        assert len(hazard_hyp) == 1
        assert "640" in hazard_hyp[0].statement
        assert "360" in hazard_hyp[0].statement
        assert hazard_hyp[0].confidence_score >= 0.90
        assert len(report.bottlenecks_detected) == 1
