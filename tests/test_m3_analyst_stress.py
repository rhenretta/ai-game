"""tests/test_m3_analyst_stress.py
Empirical stress-test suite for Telemetry Analyst Engine and Spatial Heatmap generator (Milestone 3).

Stress testing dimensions:
1. Empty event streams (0 events, zero-division resilience, safe default metrics).
2. Single-tick event streams (locomotion, dash, death, stationary idle).
3. Massive event flood (10,000+ events, 25,000+ events) measuring latency and memory.
4. Extreme out-of-bounds spatial coordinates (negative coords, coords > 2000px, boundary crossings).
5. Directional movement entropy extremes (zero-entropy monodirectional/stationary vs maximum Shannon entropy jitter).
6. Spatial death clustering edge cases (collinear points, points at exact cluster threshold, sub-threshold vs super-threshold).
7. Repeated load and memory stability (no memory leaks across multiple analysis cycles).
"""
import math
import time
import tracemalloc
from typing import Any, Dict, List
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
from tests.test_tier1_units import validate_rationale_first_ast, validate_rationale_first_dict


# ============================================================================
# 1. Empty Event Stream & Zero-Division Resilience
# ============================================================================

class TestEmptyEventStreamStress:
    """Verifies that empty streams (0 events, 0 episodes) are resilient to zero-division
    and safely return robust default baseline metrics without crashing.
    """

    def test_completely_empty_events_and_episodes(self):
        analyst = TelemetryAnalyst()
        profile = analyst.analyze_events(events=[], episodes=[])

        assert isinstance(profile, BehavioralProfile)
        assert profile.deliberation != ""
        assert profile.playstyle_tag == PlaystyleClassification.BALANCED_SKIRMISHER.value
        assert 0.0 <= profile.confidence_score <= 1.0

        # Kinetic zero-division safety
        assert profile.kinetic.mean_velocity == 0.0
        assert profile.kinetic.peak_velocity == 0.0
        assert profile.kinetic.velocity_variance == 0.0
        assert profile.kinetic.acceleration_mean == 0.0
        assert profile.kinetic.acceleration_variance == 0.0
        assert profile.kinetic.movement_entropy == 0.0
        assert profile.kinetic.idle_ratio == 0.0
        assert profile.kinetic.active_movement_duration == 0.0
        assert profile.kinetic.total_movement_ticks == 0
        assert profile.kinetic.deliberation != ""

        # Dash / evasion zero-division safety
        assert profile.evasion.dash_count == 0
        assert profile.evasion.dash_frequency == 0.0
        assert profile.evasion.dash_ratio == 0.0
        assert profile.evasion.evasion_ratio == 0.0
        assert profile.evasion.mean_hazard_proximity_at_dash == 0.0
        assert profile.evasion.dash_under_damage_count == 0
        assert profile.evasion.deliberation != ""

        # Spatial heatmap zero-division safety
        assert profile.spatial.grid_width == 16
        assert profile.spatial.grid_height == 16
        assert profile.spatial.unique_cells_visited == 0
        assert profile.spatial.spatial_coverage_ratio == 0.0
        assert profile.spatial.residency_time_entropy == 0.0
        assert profile.spatial.camping_index == 0.0
        assert profile.spatial.roaming_index == 0.0
        assert profile.spatial.centroid_x == 0.0
        assert profile.spatial.centroid_y == 0.0
        assert len(profile.spatial.occupancy_matrix) == 16
        assert all(len(row) == 16 for row in profile.spatial.occupancy_matrix)
        assert sum(sum(row) for row in profile.spatial.occupancy_matrix) == 0.0
        assert profile.spatial.deliberation != ""

        # Retries zero-division safety
        assert profile.retries.total_episodes_analyzed == 0
        assert profile.retries.cleared_episodes == 0
        assert profile.retries.failed_episodes == 0
        assert profile.retries.clear_rate == 0.0
        assert profile.retries.mean_duration_seconds == 0.0
        assert profile.retries.duration_variance == 0.0
        assert profile.retries.total_deaths == 0
        assert profile.retries.death_rate_per_minute == 0.0
        assert len(profile.retries.death_clusters) == 0
        assert profile.retries.progress_trend == 0.0
        assert profile.retries.deliberation != ""

        # Convenience properties
        assert profile.mean_velocity == 0.0
        assert profile.peak_velocity == 0.0
        assert profile.dash_ratio == 0.0
        assert profile.camping_index == 0.0
        assert profile.roaming_index == 0.0
        assert profile.deaths == 0
        assert profile.cleared is False
        assert profile.death_clusters == []

        # Rationale-first ordering on exports
        d = profile.to_dict()
        assert validate_rationale_first_dict(d) is True
        assert list(d.keys())[0] == "deliberation"

    def test_events_without_movement_ticks_only_metadata(self):
        analyst = TelemetryAnalyst()
        metadata_events = [
            {"timestamp": 1.0, "event_type": "SESSION_START", "payload": {"user": "tester"}},
            {"timestamp": 2.0, "event_type": "LEVEL_START", "payload": {"level": "zone_1"}},
            {"timestamp": 3.0, "event_type": "INVENTORY_OPEN", "payload": {"slot": 1}},
        ]
        profile = analyst.analyze_events(metadata_events)
        assert profile.kinetic.total_movement_ticks == 0
        assert profile.kinetic.mean_velocity == 0.0
        assert profile.evasion.dash_count == 0

    def test_episodes_without_durations_or_outcomes(self):
        analyst = TelemetryAnalyst()
        episodes = [
            {"episode_id": "ep_1", "session_id": "s_1"},
            {"episode_id": "ep_2", "session_id": "s_1", "duration_seconds": None, "outcome": None},
        ]
        profile = analyst.analyze_events(events=[], episodes=episodes)
        assert profile.retries.total_episodes_analyzed == 2
        assert profile.retries.cleared_episodes == 0
        assert profile.retries.failed_episodes == 0
        assert profile.retries.clear_rate == 0.0
        assert profile.retries.mean_duration_seconds == 0.0
        assert profile.retries.progress_trend == 0.0


# ============================================================================
# 2. Single-Tick Event Stream Stress
# ============================================================================

class TestSingleTickEventStreamStress:
    """Verifies that single-tick streams calculate exact boundaries without off-by-one
    or divide-by-zero crashes when delta-t or sample count is 1.
    """

    def test_single_movement_tick(self):
        analyst = TelemetryAnalyst()
        single_event = [{
            "timestamp": 10.0,
            "event_type": "MOVEMENT_TICK",
            "pos_x": 400.0,
            "pos_y": 300.0,
            "vel_x": 120.0,
            "vel_y": 50.0,
            "health": 100.0,
        }]
        profile = analyst.analyze_events(single_event)

        expected_speed = math.sqrt(120.0 ** 2 + 50.0 ** 2)
        assert profile.kinetic.total_movement_ticks == 1
        assert pytest.approx(profile.kinetic.mean_velocity, abs=0.01) == expected_speed
        assert pytest.approx(profile.kinetic.peak_velocity, abs=0.01) == expected_speed
        assert profile.kinetic.velocity_variance == 0.0
        assert profile.kinetic.acceleration_mean == 0.0  # Cannot compute acceleration from single tick
        assert profile.kinetic.acceleration_variance == 0.0
        assert profile.kinetic.movement_entropy == 0.0  # Single direction has zero Shannon entropy
        assert profile.kinetic.idle_ratio == 0.0

        # Spatial
        assert profile.spatial.unique_cells_visited == 1
        assert pytest.approx(profile.spatial.spatial_coverage_ratio, abs=1e-4) == 1.0 / 256.0
        assert pytest.approx(profile.spatial.centroid_x, abs=0.01) == 400.0
        assert pytest.approx(profile.spatial.centroid_y, abs=0.01) == 300.0
        assert profile.spatial.camping_index == 1.0
        assert profile.spatial.residency_time_entropy == 0.0
        matrix_sum = sum(sum(row) for row in profile.spatial.occupancy_matrix)
        assert pytest.approx(matrix_sum, abs=1e-3) == 1.0

    def test_single_dash_event(self):
        analyst = TelemetryAnalyst(threat_proximity_threshold=150.0)
        single_dash = [{
            "timestamp": 5.0,
            "event_type": "DASH",
            "pos_x": 200.0,
            "pos_y": 200.0,
            "vel_x": 250.0,
            "vel_y": 0.0,
            "payload": {"hazard_proximity": 60.0},
        }]
        profile = analyst.analyze_events(single_dash)

        assert profile.evasion.dash_count == 1
        assert profile.evasion.dash_ratio == 1.0
        assert profile.evasion.evasion_ratio == 1.0
        assert profile.evasion.mean_hazard_proximity_at_dash == 60.0
        assert profile.evasion.dash_frequency > 0.0  # Uses clamped duration minimum

    def test_single_death_event(self):
        analyst = TelemetryAnalyst()
        single_death = [{
            "timestamp": 12.0,
            "event_type": "PLAYER_DEATH",
            "pos_x": 640.0,
            "pos_y": 360.0,
            "payload": {"cause": "acid_pool"},
        }]
        profile = analyst.analyze_events(single_death)

        assert profile.retries.total_deaths == 1
        assert len(profile.retries.death_clusters) == 1
        cluster = profile.retries.death_clusters[0]
        assert cluster.death_count == 1
        assert cluster.center_x == 640.0
        assert cluster.center_y == 360.0
        assert cluster.radius == 0.0
        assert cluster.primary_cause == "acid_pool"

    def test_single_stationary_idle_tick(self):
        analyst = TelemetryAnalyst(noise_velocity_threshold=5.0)
        single_idle = [{
            "timestamp": 1.0,
            "event_type": "MOVEMENT_TICK",
            "pos_x": 100.0,
            "pos_y": 100.0,
            "vel_x": 1.0,
            "vel_y": 1.0,  # speed = 1.414 < 5.0
        }]
        profile = analyst.analyze_events(single_idle)

        assert profile.kinetic.total_movement_ticks == 1
        assert profile.kinetic.idle_ratio == 1.0
        assert profile.kinetic.mean_velocity == 0.0
        assert profile.kinetic.active_movement_duration == 0.0


# ============================================================================
# 3. Massive Event Flood Stress (10,000+ Events)
# ============================================================================

class TestMassiveEventFloodStress:
    """Verifies that high-volume telemetry event floods (10,000+ to 25,000+ events)
    execute with sub-second latency and minimal memory overhead.
    """

    def test_event_flood_10k_latency_and_memory(self):
        analyst = TelemetryAnalyst()
        n_events = 10000

        events = []
        for i in range(n_events):
            if i % 10 == 0:
                ev_type = "DASH"
                payload = {"nearest_threat_dist": 90.0}
            elif i % 25 == 0:
                ev_type = "DAMAGE_TAKEN"
                payload = {"damage": 15.0}
            else:
                ev_type = "MOVEMENT_TICK"
                payload = {}

            events.append({
                "timestamp": 100.0 + i * 0.016,
                "event_type": ev_type,
                "pos_x": 100.0 + (i * 3.7) % 1000.0,
                "pos_y": 80.0 + (i * 2.3) % 550.0,
                "vel_x": 180.0 if i % 2 == 0 else 60.0,
                "vel_y": 120.0 if i % 4 == 0 else -40.0,
                "health": 100.0 - (i % 20),
                "payload": payload,
            })

        tracemalloc.start()
        t0 = time.perf_counter()
        profile = analyst.analyze_events(events)
        elapsed_sec = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_bytes / (1024 * 1024)

        # Performance invariants: must be sub-second for 10k events and under 25MB peak
        assert elapsed_sec < 1.0, f"Analysis took {elapsed_sec:.3f}s for 10k events (expected < 1.0s)"
        assert peak_mb < 25.0, f"Peak memory was {peak_mb:.2f}MB (expected < 25MB)"

        # Correctness checks under flood
        assert profile.kinetic.total_movement_ticks > 9000
        assert profile.evasion.dash_count == 1000
        assert profile.spatial.unique_cells_visited > 100
        matrix_sum = sum(sum(row) for row in profile.spatial.occupancy_matrix)
        assert pytest.approx(matrix_sum, abs=1e-3) == 1.0

    def test_event_flood_25k_scaling(self):
        analyst = TelemetryAnalyst()
        n_events = 25000

        events = [
            {
                "timestamp": i * 0.016,
                "event_type": "MOVEMENT_TICK",
                "pos_x": (i * 1.5) % 1200.0,
                "pos_y": (i * 0.9) % 680.0,
                "vel_x": 150.0,
                "vel_y": 100.0,
                "health": 100.0,
            }
            for i in range(n_events)
        ]

        t0 = time.perf_counter()
        profile = analyst.analyze_events(events)
        elapsed_sec = time.perf_counter() - t0

        assert elapsed_sec < 1.5, f"Analysis took {elapsed_sec:.3f}s for 25k events (expected < 1.5s)"
        assert profile.kinetic.total_movement_ticks == 25000
        assert profile.spatial.unique_cells_visited > 50

    def test_repeated_analysis_memory_stability(self):
        analyst = TelemetryAnalyst()
        events = [
            {
                "timestamp": i * 0.02,
                "event_type": "MOVEMENT_TICK",
                "pos_x": (i * 7) % 800,
                "pos_y": (i * 5) % 600,
                "vel_x": 120.0,
                "vel_y": 80.0,
            }
            for i in range(2000)
        ]

        # Warmup
        for _ in range(3):
            analyst.analyze_events(events)

        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()
        for _ in range(10):
            analyst.analyze_events(events)
        snapshot_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snapshot_end.compare_to(snapshot_start, "lineno")
        total_growth = sum(stat.size_diff for stat in top_stats)
        growth_mb = total_growth / (1024 * 1024)

        assert growth_mb < 2.0, f"Potential memory growth: {growth_mb:.2f}MB across 10 iterations"


# ============================================================================
# 4. Extreme Out-Of-Bounds Coordinates Stress
# ============================================================================

class TestExtremeOutOfBoundsCoordinatesStress:
    """Verifies that coordinates far beyond world bounds (-99999 to +99999)
    are safely clamped into valid grid cells [0..15] without IndexError or array corruption.
    """

    def test_negative_coordinates(self):
        analyst = TelemetryAnalyst(world_bounds=(0.0, 0.0, 1280.0, 720.0), grid_resolution=(16, 16))
        events = [
            {"timestamp": 1.0, "event_type": "MOVEMENT_TICK", "pos_x": -500.0, "pos_y": -1200.0, "vel_x": 0.0, "vel_y": 0.0},
            {"timestamp": 2.0, "event_type": "MOVEMENT_TICK", "pos_x": -99999.0, "pos_y": -0.01, "vel_x": 0.0, "vel_y": 0.0},
        ]
        profile = analyst.analyze_events(events)

        # Both negative points must clamp to grid[0][0]
        assert profile.spatial.unique_cells_visited == 1
        assert profile.spatial.occupancy_matrix[0][0] == 1.0
        assert profile.spatial.centroid_x < 0.0
        assert profile.spatial.centroid_y < 0.0

    def test_excessive_positive_coordinates(self):
        analyst = TelemetryAnalyst(world_bounds=(0.0, 0.0, 1280.0, 720.0), grid_resolution=(16, 16))
        events = [
            {"timestamp": 1.0, "event_type": "MOVEMENT_TICK", "pos_x": 2500.0, "pos_y": 50000.0, "vel_x": 0.0, "vel_y": 0.0},
            {"timestamp": 2.0, "event_type": "MOVEMENT_TICK", "pos_x": 10000.0, "pos_y": 10000.0, "vel_x": 0.0, "vel_y": 0.0},
        ]
        profile = analyst.analyze_events(events)

        # Both excessive points must clamp to grid[15][15]
        assert profile.spatial.unique_cells_visited == 1
        assert profile.spatial.occupancy_matrix[15][15] == 1.0
        assert profile.spatial.centroid_x > 2000.0
        assert profile.spatial.centroid_y > 2000.0

    def test_mixed_extreme_coordinates_crossings(self):
        analyst = TelemetryAnalyst(world_bounds=(0.0, 0.0, 1280.0, 720.0), grid_resolution=(16, 16))
        events = [
            {"timestamp": 1.0, "event_type": "MOVEMENT_TICK", "pos_x": -1000.0, "pos_y": -1000.0, "vel_x": 100.0, "vel_y": 100.0},
            {"timestamp": 2.0, "event_type": "MOVEMENT_TICK", "pos_x": 5000.0, "pos_y": -1000.0, "vel_x": 100.0, "vel_y": 100.0},
            {"timestamp": 3.0, "event_type": "MOVEMENT_TICK", "pos_x": -1000.0, "pos_y": 5000.0, "vel_x": 100.0, "vel_y": 100.0},
            {"timestamp": 4.0, "event_type": "MOVEMENT_TICK", "pos_x": 5000.0, "pos_y": 5000.0, "vel_x": 100.0, "vel_y": 100.0},
        ]
        profile = analyst.analyze_events(events)

        # Clamps to the 4 corners: (0,0), (0,15), (15,0), (15,15)
        assert profile.spatial.unique_cells_visited == 4
        assert profile.spatial.occupancy_matrix[0][0] == 0.25
        assert profile.spatial.occupancy_matrix[0][15] == 0.25
        assert profile.spatial.occupancy_matrix[15][0] == 0.25
        assert profile.spatial.occupancy_matrix[15][15] == 0.25
        matrix_sum = sum(sum(row) for row in profile.spatial.occupancy_matrix)
        assert pytest.approx(matrix_sum, abs=1e-3) == 1.0

    def test_degenerate_zero_span_world_bounds(self):
        # min == max world bounds: span = 0 guarded by max(span, 1.0)
        analyst = TelemetryAnalyst(world_bounds=(100.0, 100.0, 100.0, 100.0))
        events = [
            {"timestamp": 1.0, "event_type": "MOVEMENT_TICK", "pos_x": 100.0, "pos_y": 100.0, "vel_x": 50.0, "vel_y": 50.0},
        ]
        profile = analyst.analyze_events(events)
        assert profile.spatial.unique_cells_visited == 1
        assert sum(sum(r) for r in profile.spatial.occupancy_matrix) == 1.0


# ============================================================================
# 5. Directional Movement & Spatial Entropy Extremes
# ============================================================================

class TestEntropyExtremesStress:
    """Verifies normalized Shannon entropy at boundary extremes:
    - Zero entropy: pure monodirectional movement or pure stationary camping.
    - Maximum entropy (~1.0): uniform dispersion across all 8 octants or uniform grid coverage.
    """

    def test_zero_entropy_monodirectional_movement(self):
        analyst = TelemetryAnalyst()
        events = [
            {
                "timestamp": 10.0 + i * 0.1,
                "event_type": "MOVEMENT_TICK",
                "pos_x": 100.0 + i * 15.0,
                "pos_y": 200.0,
                "vel_x": 150.0,
                "vel_y": 0.0,  # Exactly 0 degrees angle
            }
            for i in range(50)
        ]
        profile = analyst.analyze_events(events)

        assert profile.kinetic.mean_velocity == 150.0
        assert profile.kinetic.movement_entropy == 0.0
        assert profile.kinetic.idle_ratio == 0.0

    def test_zero_entropy_stationary_idle(self):
        analyst = TelemetryAnalyst(noise_velocity_threshold=5.0)
        events = [
            {
                "timestamp": 10.0 + i * 0.1,
                "event_type": "MOVEMENT_TICK",
                "pos_x": 100.0,
                "pos_y": 200.0,
                "vel_x": 0.0,
                "vel_y": 0.0,
            }
            for i in range(50)
        ]
        profile = analyst.analyze_events(events)

        assert profile.kinetic.idle_ratio == 1.0
        assert profile.kinetic.movement_entropy == 0.0
        assert profile.kinetic.mean_velocity == 0.0

    def test_maximum_entropy_chaotic_jitter(self):
        analyst = TelemetryAnalyst()
        events = []
        # Distribute velocity vectors centered within each of the 8 octant bins
        for i in range(80):
            angle = (i % 8 + 0.5) * (2.0 * math.pi / 8.0) - math.pi
            vx = 120.0 * math.cos(angle)
            vy = 120.0 * math.sin(angle)
            events.append({
                "timestamp": 10.0 + i * 0.1,
                "event_type": "MOVEMENT_TICK",
                "pos_x": 300.0,
                "pos_y": 300.0,
                "vel_x": vx,
                "vel_y": vy,
            })
        profile = analyst.analyze_events(events)

        # 8 octants equally populated -> normalized Shannon entropy must be exactly 1.0
        assert pytest.approx(profile.kinetic.movement_entropy, abs=1e-3) == 1.0

    def test_spatial_residency_entropy_uniform_vs_camped(self):
        analyst = TelemetryAnalyst(world_bounds=(0.0, 0.0, 800.0, 800.0), grid_resolution=(16, 16))

        # Uniform visits across all 256 cells
        uniform_events = []
        for gy in range(16):
            for gx in range(16):
                px = gx * 50.0 + 25.0
                py = gy * 50.0 + 25.0
                uniform_events.append({
                    "timestamp": 1.0,
                    "event_type": "MOVEMENT_TICK",
                    "pos_x": px,
                    "pos_y": py,
                    "vel_x": 50.0,
                    "vel_y": 50.0,
                })
        profile_uniform = analyst.analyze_events(uniform_events)

        assert profile_uniform.spatial.unique_cells_visited == 256
        assert pytest.approx(profile_uniform.spatial.spatial_coverage_ratio, abs=1e-3) == 1.0
        assert pytest.approx(profile_uniform.spatial.residency_time_entropy, abs=1e-3) == 1.0
        assert pytest.approx(profile_uniform.spatial.camping_index, abs=0.02) == 0.10
        assert profile_uniform.spatial.roaming_index > 0.80

        # Highly camped: 100 ticks in single cell (100, 100)
        camped_events = [
            {"timestamp": i * 0.1, "event_type": "MOVEMENT_TICK", "pos_x": 100.0, "pos_y": 100.0, "vel_x": 0.0, "vel_y": 0.0}
            for i in range(100)
        ]
        profile_camped = analyst.analyze_events(camped_events)

        assert profile_camped.spatial.unique_cells_visited == 1
        assert profile_camped.spatial.residency_time_entropy == 0.0
        assert profile_camped.spatial.camping_index == 1.0
        assert profile_camped.spatial.roaming_index < 0.01


# ============================================================================
# 6. Spatial Death Clustering Edge Cases
# ============================================================================

class TestSpatialDeathClusteringStress:
    """Verifies spatial death clustering behavior on exact boundaries, collinear points,
    coincident deaths, and negative coordinates.
    """

    def test_deaths_at_exact_cluster_threshold(self):
        # death_clustering_radius = 80.0
        analyst = TelemetryAnalyst(death_clustering_radius=80.0)
        deaths = [
            {"timestamp": 1.0, "event_type": "PLAYER_DEATH", "pos_x": 200.0, "pos_y": 200.0, "payload": {"cause": "spikes"}},
            {"timestamp": 2.0, "event_type": "PLAYER_DEATH", "pos_x": 280.0, "pos_y": 200.0, "payload": {"cause": "spikes"}},
        ]
        profile = analyst.analyze_events(deaths)

        # Distance is exactly 80.0 px: dist <= 80.0 -> single cluster
        assert profile.retries.total_deaths == 2
        assert len(profile.retries.death_clusters) == 1
        c = profile.retries.death_clusters[0]
        assert c.death_count == 2
        assert pytest.approx(c.center_x, abs=0.1) == 240.0
        assert pytest.approx(c.center_y, abs=0.1) == 200.0
        assert pytest.approx(c.radius, abs=0.1) == 40.0

    def test_deaths_just_outside_cluster_threshold(self):
        analyst = TelemetryAnalyst(death_clustering_radius=80.0)
        deaths = [
            {"timestamp": 1.0, "event_type": "PLAYER_DEATH", "pos_x": 200.0, "pos_y": 200.0, "payload": {"cause": "spikes"}},
            {"timestamp": 2.0, "event_type": "PLAYER_DEATH", "pos_x": 280.01, "pos_y": 200.0, "payload": {"cause": "spikes"}},
        ]
        profile = analyst.analyze_events(deaths)

        # Distance is 80.01 px > 80.0 -> 2 distinct clusters
        assert profile.retries.total_deaths == 2
        assert len(profile.retries.death_clusters) == 2
        assert all(c.death_count == 1 for c in profile.retries.death_clusters)

    def test_collinear_death_points(self):
        analyst = TelemetryAnalyst(death_clustering_radius=80.0)
        # Points along horizontal line: P1=100, P2=150, P3=200
        # dist(P1, P2)=50 <= 80, dist(P1, P3)=100 > 80
        deaths_order_1 = [
            {"timestamp": 1.0, "event_type": "PLAYER_DEATH", "pos_x": 100.0, "pos_y": 100.0, "payload": {"cause": "mine"}},
            {"timestamp": 2.0, "event_type": "PLAYER_DEATH", "pos_x": 150.0, "pos_y": 100.0, "payload": {"cause": "mine"}},
            {"timestamp": 3.0, "event_type": "PLAYER_DEATH", "pos_x": 200.0, "pos_y": 100.0, "payload": {"cause": "mine"}},
        ]
        profile_1 = analyst.analyze_events(deaths_order_1)

        # P1 is pivot: groups P1 and P2 (count 2), leaving P3 (count 1)
        assert len(profile_1.retries.death_clusters) == 2
        c1 = profile_1.retries.death_clusters[0]
        c2 = profile_1.retries.death_clusters[1]
        assert c1.death_count == 2
        assert pytest.approx(c1.center_x, abs=0.1) == 125.0
        assert c2.death_count == 1
        assert pytest.approx(c2.center_x, abs=0.1) == 200.0

        # Now center point P2 is first: dist(P2, P1)=50 <= 80, dist(P2, P3)=50 <= 80
        deaths_order_2 = [
            {"timestamp": 2.0, "event_type": "PLAYER_DEATH", "pos_x": 150.0, "pos_y": 100.0, "payload": {"cause": "mine"}},
            {"timestamp": 1.0, "event_type": "PLAYER_DEATH", "pos_x": 100.0, "pos_y": 100.0, "payload": {"cause": "mine"}},
            {"timestamp": 3.0, "event_type": "PLAYER_DEATH", "pos_x": 200.0, "pos_y": 100.0, "payload": {"cause": "mine"}},
        ]
        profile_2 = analyst.analyze_events(deaths_order_2)

        # Single cluster encompassing all 3 collinear points
        assert len(profile_2.retries.death_clusters) == 1
        c_all = profile_2.retries.death_clusters[0]
        assert c_all.death_count == 3
        assert pytest.approx(c_all.center_x, abs=0.1) == 150.0
        assert pytest.approx(c_all.radius, abs=0.1) == 50.0

    def test_coincident_deaths(self):
        analyst = TelemetryAnalyst()
        deaths = [
            {"timestamp": float(i), "event_type": "PLAYER_DEATH", "pos_x": 450.0, "pos_y": 350.0, "payload": {"cause": "turret"}}
            for i in range(10)
        ]
        profile = analyst.analyze_events(deaths)

        assert profile.retries.total_deaths == 10
        assert len(profile.retries.death_clusters) == 1
        c = profile.retries.death_clusters[0]
        assert c.death_count == 10
        assert c.center_x == 450.0
        assert c.center_y == 350.0
        assert c.radius == 0.0

    def test_death_clustering_with_negative_coordinates(self):
        analyst = TelemetryAnalyst(death_clustering_radius=80.0)
        deaths = [
            {"timestamp": 1.0, "event_type": "PLAYER_DEATH", "pos_x": -100.0, "pos_y": -200.0, "payload": {"cause": "void"}},
            {"timestamp": 2.0, "event_type": "PLAYER_DEATH", "pos_x": -150.0, "pos_y": -200.0, "payload": {"cause": "void"}},
            {"timestamp": 3.0, "event_type": "PLAYER_DEATH", "pos_x": -180.0, "pos_y": -200.0, "payload": {"cause": "void"}},
        ]
        profile = analyst.analyze_events(deaths)

        # All within 80px of pivot (-100, -200)
        assert len(profile.retries.death_clusters) == 1
        c = profile.retries.death_clusters[0]
        assert c.death_count == 3
        expected_cx = (-100.0 - 150.0 - 180.0) / 3.0
        assert pytest.approx(c.center_x, abs=0.1) == expected_cx
        assert pytest.approx(c.center_y, abs=0.1) == -200.0
        assert c.radius > 0.0
