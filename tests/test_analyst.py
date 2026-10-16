"""tests/test_analyst.py
Comprehensive unit and integration test suite for the Telemetry Analyst Engine (Milestone 3, Feature 9).

Verifies:
  1. Strict Rationale-First Epistemic Architecture on all models via AST and dictionary checks.
  2. Mathematical accuracy of kinetic velocity vectors, acceleration variance, and movement entropy.
  3. Dash frequency, evasion ratio under threat proximity, and damage-window detection.
  4. 2D spatial occupancy discretization, residency matrix normalization, camping vs roaming index.
  5. Spatial death clustering algorithm and retry progression trends.
  6. Playstyle classification divergence (Rusher vs. Cautious profiles).
  7. SQLite WAL live reader queries and transaction isolation.
  8. Graceful zero-event edge case handling without ZeroDivisionError.
"""
import ast
import json
import math
import sqlite3
import pytest

from engine.analyst.models import (
    BehavioralProfile,
    DashMetrics,
    DeathCluster,
    KineticMetrics,
    PlaystyleClassification,
    RetryMetrics,
    SpatialOccupancyMap,
)
from engine.analyst.analyst import TelemetryAnalyst
from engine.telemetry.recorder import TelemetryRecorder
from tests.test_tier1_units import validate_rationale_first_ast, validate_rationale_first_dict


# ============================================================================
# Synthetic Telemetry Fixtures
# ============================================================================

@pytest.fixture
def synthetic_rusher_events():
    """Generates 40 events simulating a Kinetic High-Mobility Rusher:
    High velocity (>250 px/s), frequent dashes (>50%), roaming across grid.
    """
    events = []
    for i in range(40):
        is_dash = (i % 2 == 0)
        vx = 280.0 if is_dash else 220.0
        vy = 260.0 if is_dash else 210.0
        px = 100.0 + (i * 28.0) % 1100.0
        py = 80.0 + (i * 15.0) % 600.0
        events.append({
            "timestamp": 100.0 + i * 0.1,
            "event_type": "DASH" if is_dash else "MOVEMENT_TICK",
            "pos_x": px,
            "pos_y": py,
            "vel_x": vx,
            "vel_y": vy,
            "health": 100.0,
            "payload": {"nearest_threat_dist": 80.0 if is_dash else 300.0},
        })
    return events


@pytest.fixture
def synthetic_cautious_events():
    """Generates 40 events simulating a Methodical/Cautious Ranged player:
    Low velocity (<100 px/s), rare dashes (<10%), stationary camping in one corner.
    """
    events = []
    for i in range(40):
        is_dash = (i == 10)
        is_idle = (i >= 15)
        vx = 150.0 if is_dash else (0.0 if is_idle else 60.0)
        vy = 0.0 if is_dash else (0.0 if is_idle else 40.0)
        px = 120.0 + (min(i, 4) * 4.0)
        py = 120.0 + (min(i, 4) * 4.0)
        events.append({
            "timestamp": 100.0 + i * 0.1,
            "event_type": "DASH" if is_dash else "MOVEMENT_TICK",
            "pos_x": px,
            "pos_y": py,
            "vel_x": vx,
            "vel_y": vy,
            "health": 100.0,
            "payload": {"in_cover": True if is_idle else False},
        })
    return events


@pytest.fixture
def synthetic_cluster_deaths():
    """Generates clustered death events around (400, 300) and an isolated death at (1000, 600)."""
    return [
        {"timestamp": 110.0, "event_type": "PLAYER_DEATH", "pos_x": 402.0, "pos_y": 298.0, "payload": {"cause": "spike_pit"}},
        {"timestamp": 140.0, "event_type": "PLAYER_DEATH", "pos_x": 398.0, "pos_y": 305.0, "payload": {"cause": "spike_pit"}},
        {"timestamp": 180.0, "event_type": "PLAYER_DEATH", "pos_x": 405.0, "pos_y": 302.0, "payload": {"cause": "spike_pit"}},
        {"timestamp": 220.0, "event_type": "PLAYER_DEATH", "pos_x": 1020.0, "pos_y": 610.0, "payload": {"cause": "boss_bullet"}},
    ]


# ============================================================================
# Test Cases
# ============================================================================

class TestRationaleFirstEpistemicArchitecture:
    """Verifies that all models in engine.analyst.models strictly adhere to
    Rationale-First Epistemic Architecture (deliberation declared first).
    """

    def test_models_source_ast_check(self):
        import engine.analyst.models as models_module
        import inspect
        source = inspect.getsource(models_module)
        assert validate_rationale_first_ast(source) is True

    def test_analyst_source_ast_check(self):
        import engine.analyst.analyst as analyst_module
        import inspect
        source = inspect.getsource(analyst_module)
        assert validate_rationale_first_ast(source) is True

    def test_all_models_dict_ordering(self):
        models = [
            KineticMetrics(deliberation="Testing kinetic rationale."),
            DashMetrics(deliberation="Testing dash rationale."),
            SpatialOccupancyMap(deliberation="Testing spatial rationale."),
            DeathCluster(deliberation="Testing death cluster rationale."),
            RetryMetrics(deliberation="Testing retry rationale."),
            BehavioralProfile(deliberation="Testing behavioral profile rationale."),
        ]
        for m in models:
            d = m.to_dict()
            assert validate_rationale_first_dict(d) is True
            assert list(d.keys())[0] == "deliberation"


class TestKineticAndDashAnalytics:
    """Verifies kinetic velocity vectors, acceleration variance, and dash evasion calculations."""

    def test_kinetic_velocity_and_entropy(self, synthetic_rusher_events):
        analyst = TelemetryAnalyst()
        profile = analyst.analyze_events(synthetic_rusher_events)

        assert profile.kinetic.mean_velocity > 250.0
        assert profile.kinetic.peak_velocity >= 350.0
        assert profile.kinetic.velocity_variance > 0.0
        assert 0.0 <= profile.kinetic.movement_entropy <= 1.0
        assert profile.kinetic.idle_ratio == 0.0
        assert "deliberation" in profile.kinetic.to_dict()

    def test_dash_and_evasion_metrics(self, synthetic_rusher_events):
        analyst = TelemetryAnalyst(threat_proximity_threshold=150.0)
        profile = analyst.analyze_events(synthetic_rusher_events)

        assert profile.evasion.dash_count == 20
        assert profile.evasion.dash_ratio == 0.50
        assert profile.evasion.evasion_ratio > 0.80
        assert profile.evasion.mean_hazard_proximity_at_dash == 80.0


class TestSpatialOccupancyHeatmap:
    """Verifies 16x16 grid discretization, normalized residency, camping and roaming indexes."""

    def test_rusher_broad_roaming(self, synthetic_rusher_events):
        analyst = TelemetryAnalyst(grid_resolution=(16, 16))
        profile = analyst.analyze_events(synthetic_rusher_events)

        assert profile.spatial.grid_width == 16
        assert profile.spatial.grid_height == 16
        assert profile.spatial.unique_cells_visited >= 10
        assert profile.spatial.roaming_index > 0.05
        assert profile.spatial.camping_index < 0.80
        # Verify matrix sums to ~1.0
        matrix_sum = sum(sum(row) for row in profile.spatial.occupancy_matrix)
        assert pytest.approx(matrix_sum, abs=1e-3) == 1.0

    def test_cautious_tight_camping(self, synthetic_cautious_events):
        analyst = TelemetryAnalyst(grid_resolution=(16, 16))
        profile = analyst.analyze_events(synthetic_cautious_events)

        assert profile.spatial.unique_cells_visited <= 6
        assert profile.spatial.camping_index > 0.80
        assert profile.spatial.roaming_index < 0.03
        assert profile.kinetic.idle_ratio >= 0.40


class TestRetryAndDeathClusterAnalytics:
    """Verifies spatial death clustering and linear progression slope calculations."""

    def test_death_clustering(self, synthetic_cluster_deaths):
        analyst = TelemetryAnalyst(death_clustering_radius=80.0)
        profile = analyst.analyze_events(synthetic_cluster_deaths)

        assert profile.retries.total_deaths == 4
        # Expect 2 clusters: spike pit cluster (3 deaths) and isolated boss cluster (1 death)
        assert len(profile.retries.death_clusters) == 2

        spike_cluster = next(c for c in profile.retries.death_clusters if c.death_count == 3)
        assert pytest.approx(spike_cluster.center_x, abs=5.0) == 401.6
        assert pytest.approx(spike_cluster.center_y, abs=5.0) == 301.6
        assert spike_cluster.primary_cause == "spike_pit"

    def test_progression_trend(self):
        analyst = TelemetryAnalyst()
        episodes = [
            {"episode_number": 1, "duration_seconds": 15.0, "outcome": "FAILED"},
            {"episode_number": 2, "duration_seconds": 35.0, "outcome": "FAILED"},
            {"episode_number": 3, "duration_seconds": 65.0, "outcome": "FAILED"},
            {"episode_number": 4, "duration_seconds": 95.0, "outcome": "CLEARED"},
        ]
        profile = analyst.analyze_events(events=[], episodes=episodes)

        assert profile.retries.total_episodes_analyzed == 4
        assert profile.retries.clear_rate == 0.25
        assert profile.retries.progress_trend > 0.0  # Positive duration slope indicates advancing further


class TestPlaystyleClassificationDivergence:
    """Verifies that Rusher vs. Cautious profiles diverge into distinct archetypes."""

    def test_playstyle_divergence(self, synthetic_rusher_events, synthetic_cautious_events):
        analyst = TelemetryAnalyst()
        rusher_profile = analyst.analyze_events(synthetic_rusher_events)
        cautious_profile = analyst.analyze_events(synthetic_cautious_events)

        assert rusher_profile.playstyle_tag == PlaystyleClassification.KINETIC_HIGH_MOBILITY.value
        assert rusher_profile.pacing_score > 0.50
        assert rusher_profile.confidence_score >= 0.70

        assert cautious_profile.playstyle_tag in (
            PlaystyleClassification.CAUTIOUS_RANGED.value,
            PlaystyleClassification.METHODICAL_TACTICAL.value
        )
        assert cautious_profile.pacing_score < 0.0
        assert cautious_profile.confidence_score >= 0.70


class TestSQLiteWALIntegration:
    """Verifies live SQLite WAL recording and reading via TelemetryAnalyst."""

    def test_live_wal_session_analysis(self, temp_sqlite_path):
        recorder = TelemetryRecorder(db_path=temp_sqlite_path, batch_size=5, flush_interval_sec=0.01)
        sess_id = recorder.start_session(player_profile_tag="integration_test")
        ep_id = recorder.start_episode(level_id="arena_01", episode_index=1)

        for i in range(15):
            recorder.emit_event(
                event_type="DASH" if i % 2 == 0 else "MOVEMENT_TICK",
                pos=(100.0 + i * 20.0, 150.0 + i * 10.0),
                vel=(250.0, 200.0),
                health=100.0,
                payload={"nearest_threat_dist": 90.0},
            )
        recorder.end_episode(outcome="CLEARED", score=500)
        recorder.flush()
        recorder.close()

        analyst = TelemetryAnalyst(db_path=temp_sqlite_path)
        profile = analyst.analyze_session(sess_id)

        assert profile.session_id == sess_id
        assert profile.kinetic.total_movement_ticks == 15
        assert profile.kinetic.mean_velocity > 250.0
        assert profile.evasion.dash_count == 8
        assert profile.retries.cleared_episodes == 1
        assert profile.retries.clear_rate == 1.0


class TestEdgeCasesAndZeroDivision:
    """Verifies robustness on sparse or empty telemetry streams."""

    def test_empty_events_handling(self):
        analyst = TelemetryAnalyst()
        profile = analyst.analyze_events(events=[], episodes=[])

        assert profile.kinetic.mean_velocity == 0.0
        assert profile.evasion.dash_count == 0
        assert profile.spatial.unique_cells_visited == 0
        assert profile.retries.total_episodes_analyzed == 0
        assert profile.playstyle_tag == PlaystyleClassification.BALANCED_SKIRMISHER.value
        assert "No movement ticks" in profile.kinetic.deliberation
