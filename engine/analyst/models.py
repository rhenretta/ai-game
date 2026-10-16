"""engine/analyst/models.py
Rationale-First data models for telemetry analysis, spatial occupancy, and behavioral profiling.
In accordance with Architectural Integrity Rule 4 and AGENTS.md, every model declares
deliberation as its very first property.
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class PlaystyleClassification(str, Enum):
    """Categorical classification of player behavioral playstyle."""
    KINETIC_HIGH_MOBILITY = "kinetic_high_mobility"
    CAUTIOUS_RANGED = "cautious_ranged"
    METHODICAL_TACTICAL = "methodical_tactical"
    BALANCED_SKIRMISHER = "balanced_skirmisher"
    AGGRESSIVE_MELEE = "aggressive_melee"


class KineticMetrics(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared first!
    Kinetic velocity vector, acceleration variance, and movement entropy metrics.
    """
    deliberation: str = Field(
        default="",
        description="Epistemic rationale analyzing velocity distributions, acceleration bursts, and movement entropy."
    )
    mean_velocity: float = Field(default=0.0, description="Mean speed (px/s) over non-idle active frames")
    peak_velocity: float = Field(default=0.0, description="Maximum observed instantaneous speed (px/s)")
    velocity_variance: float = Field(default=0.0, description="Variance of speed distribution across active frames")
    acceleration_mean: float = Field(default=0.0, description="Mean acceleration magnitude (px/s^2)")
    acceleration_variance: float = Field(default=0.0, description="Variance in acceleration magnitude (erratic vs smooth)")
    movement_entropy: float = Field(
        default=0.0,
        description="Normalized Shannon entropy [0.0, 1.0] across 8-direction velocity vectors"
    )
    idle_ratio: float = Field(default=0.0, description="Fraction of recorded frames with velocity below noise threshold")
    active_movement_duration: float = Field(default=0.0, description="Estimated total seconds spent actively moving")
    total_movement_ticks: int = Field(default=0, description="Total count of movement tick events analyzed")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "mean_velocity": round(self.mean_velocity, 2),
            "peak_velocity": round(self.peak_velocity, 2),
            "velocity_variance": round(self.velocity_variance, 2),
            "acceleration_mean": round(self.acceleration_mean, 2),
            "acceleration_variance": round(self.acceleration_variance, 2),
            "movement_entropy": round(self.movement_entropy, 3),
            "idle_ratio": round(self.idle_ratio, 3),
            "active_movement_duration": round(self.active_movement_duration, 2),
            "total_movement_ticks": self.total_movement_ticks,
        }

    @property
    def mean_speed(self) -> float:
        return self.mean_velocity


class DashMetrics(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared first!
    Dash cadence, evasive utility, and threat proximity dynamics.
    """
    deliberation: str = Field(
        default="",
        description="Epistemic rationale analyzing evasive maneuvers, dash cadence, and proximity to lethal threats."
    )
    dash_count: int = Field(default=0, description="Total number of dash events executed")
    dash_frequency: float = Field(default=0.0, description="Dashes executed per active minute of gameplay")
    dash_ratio: float = Field(
        default=0.0,
        description="Ratio of dashes executed relative to total movement actions (dashes / (dashes + active_moves))"
    )
    evasion_ratio: float = Field(
        default=0.0,
        description="Ratio of dashes executed in proximity to threats (<150px) or damage windows"
    )
    mean_hazard_proximity_at_dash: float = Field(
        default=0.0,
        description="Mean distance to nearest hazard/enemy when dash was initiated (0.0 if unrecorded)"
    )
    dash_under_damage_count: int = Field(
        default=0,
        description="Dashes executed within 0.5 seconds before or after taking damage"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "dash_count": self.dash_count,
            "dash_frequency": round(self.dash_frequency, 2),
            "dash_ratio": round(self.dash_ratio, 3),
            "evasion_ratio": round(self.evasion_ratio, 3),
            "mean_hazard_proximity_at_dash": round(self.mean_hazard_proximity_at_dash, 2),
            "dash_under_damage_count": self.dash_under_damage_count,
        }


class SpatialOccupancyMap(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared first!
    2D spatial grid occupancy, residency heatmaps, camping index, and roaming dispersion.
    """
    deliberation: str = Field(
        default="",
        description="Epistemic rationale analyzing spatial territory coverage, grid occupancy, and zoning behavior."
    )
    grid_width: int = Field(default=16, description="Horizontal cell count of discretized spatial grid")
    grid_height: int = Field(default=16, description="Vertical cell count of discretized spatial grid")
    world_bounds: Tuple[float, float, float, float] = Field(
        default=(0.0, 0.0, 1280.0, 720.0),
        description="(min_x, min_y, max_x, max_y) of spatial arena"
    )
    occupancy_matrix: List[List[float]] = Field(
        default_factory=list,
        description="Normalized 16x16 probability density matrix of residency time (sums to 1.0)"
    )
    unique_cells_visited: int = Field(default=0, description="Count of discrete grid cells visited at least once")
    spatial_coverage_ratio: float = Field(
        default=0.0,
        description="Fraction of total grid cells visited (unique_cells / (grid_width * grid_height))"
    )
    residency_time_entropy: float = Field(
        default=0.0,
        description="Normalized Shannon entropy of spatial occupancy [0.0 = camped single spot, 1.0 = uniform roaming]"
    )
    camping_index: float = Field(
        default=0.0,
        description="Proportion of total residency time spent in top 10% most frequented cells [0.10 to 1.0]"
    )
    roaming_index: float = Field(
        default=0.0,
        description="Dispersion index in [0.0, 1.0] factoring coverage ratio and non-camping distribution"
    )
    centroid_x: float = Field(default=0.0, description="Residency-weighted mean X coordinate")
    centroid_y: float = Field(default=0.0, description="Residency-weighted mean Y coordinate")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "world_bounds": self.world_bounds,
            "unique_cells_visited": self.unique_cells_visited,
            "spatial_coverage_ratio": round(self.spatial_coverage_ratio, 3),
            "residency_time_entropy": round(self.residency_time_entropy, 3),
            "camping_index": round(self.camping_index, 3),
            "roaming_index": round(self.roaming_index, 3),
            "centroid_x": round(self.centroid_x, 1),
            "centroid_y": round(self.centroid_y, 1),
            "occupancy_matrix": [[round(val, 4) for val in row] for row in self.occupancy_matrix],
        }


class DeathCluster(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared first!
    Spatial aggregation of player death coordinates identifying high-lethality choke points.
    """
    deliberation: str = Field(
        default="",
        description="Epistemic rationale analyzing spatial concentration of fatal incidents."
    )
    center_x: float = Field(default=0.0, description="Centroid X coordinate of death cluster")
    center_y: float = Field(default=0.0, description="Centroid Y coordinate of death cluster")
    radius: float = Field(default=0.0, description="Spatial spread radius (px) encompassing cluster deaths")
    death_count: int = Field(default=0, description="Total fatal incidents within this cluster")
    primary_cause: str = Field(default="unknown", description="Dominant hazard or killer entity type")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "center_x": round(self.center_x, 1),
            "center_y": round(self.center_y, 1),
            "radius": round(self.radius, 1),
            "death_count": self.death_count,
            "primary_cause": self.primary_cause,
        }


class RetryMetrics(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared first!
    Run attempt variance, survival curve, clear rate, and fatal death clusters.
    """
    deliberation: str = Field(
        default="",
        description="Epistemic rationale analyzing episode retry variance, death frequency, and progression curves."
    )
    total_episodes_analyzed: int = Field(default=0, description="Total episode attempts evaluated")
    cleared_episodes: int = Field(default=0, description="Count of successfully cleared episodes")
    failed_episodes: int = Field(default=0, description="Count of failed/aborted episodes")
    clear_rate: float = Field(default=0.0, description="Ratio of cleared episodes to total attempts [0.0, 1.0]")
    mean_duration_seconds: float = Field(default=0.0, description="Mean episode survival duration")
    duration_variance: float = Field(default=0.0, description="Variance of episode durations across retries")
    total_deaths: int = Field(default=0, description="Total death events recorded across episodes")
    death_rate_per_minute: float = Field(default=0.0, description="Normalized deaths per active minute")
    death_clusters: List[DeathCluster] = Field(
        default_factory=list,
        description="Spatially clustered lethal coordinates grouped within epsilon radius"
    )
    progress_trend: float = Field(
        default=0.0,
        description="Linear regression slope of run duration over episode index (positive = advancing further)"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "total_episodes_analyzed": self.total_episodes_analyzed,
            "cleared_episodes": self.cleared_episodes,
            "failed_episodes": self.failed_episodes,
            "clear_rate": round(self.clear_rate, 3),
            "mean_duration_seconds": round(self.mean_duration_seconds, 2),
            "duration_variance": round(self.duration_variance, 2),
            "total_deaths": self.total_deaths,
            "death_rate_per_minute": round(self.death_rate_per_minute, 2),
            "death_clusters": [dc.to_dict() for dc in self.death_clusters],
            "progress_trend": round(self.progress_trend, 3),
        }


class BehavioralProfile(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared first!
    Synthesized behavioral profile consumed by Cognitive Director and exposed via Local MCP Server.
    """
    deliberation: str = Field(
        default="",
        description="Epistemic rationale synthesizing kinetic velocity, evasion, spatial occupancy, and retry metrics."
    )
    session_id: str = Field(default="", description="Target session identifier")
    episode_id: Optional[str] = Field(default=None, description="Target episode identifier, or None if session aggregate")
    playstyle_tag: str = Field(
        default=PlaystyleClassification.BALANCED_SKIRMISHER.value,
        description="Categorical playstyle archetype from PlaystyleClassification"
    )
    confidence_score: float = Field(
        default=0.5,
        description="Empirical classification confidence score in [0.0, 1.0]"
    )
    pacing_score: float = Field(
        default=0.0,
        description="Continuous kinetic-tactical metric from -1.0 (pure tactical) to +1.0 (pure kinetic)"
    )
    risk_tolerance_score: float = Field(
        default=0.5,
        description="Normalized risk tolerance metric from 0.0 (extreme cautious) to 1.0 (reckless rusher)"
    )
    kinetic: KineticMetrics = Field(default_factory=KineticMetrics)
    evasion: DashMetrics = Field(default_factory=DashMetrics)
    spatial: SpatialOccupancyMap = Field(default_factory=SpatialOccupancyMap)
    retries: RetryMetrics = Field(default_factory=RetryMetrics)
    summary_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Key numerical feature vector for downstream model conditioning"
    )
    analyzed_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Downstream convenience properties
    @property
    def mean_velocity(self) -> float:
        return self.kinetic.mean_velocity

    @property
    def peak_velocity(self) -> float:
        return self.kinetic.peak_velocity

    @property
    def dash_ratio(self) -> float:
        return self.evasion.dash_ratio

    @property
    def camping_index(self) -> float:
        return self.spatial.camping_index

    @property
    def roaming_index(self) -> float:
        return self.spatial.roaming_index

    @property
    def deaths(self) -> int:
        return self.retries.total_deaths

    @property
    def cleared(self) -> bool:
        return self.retries.cleared_episodes > 0

    @property
    def death_clusters(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": f"cluster_{i+1}",
                "x": c.center_x,
                "y": c.center_y,
                "radius": c.radius,
                "count": c.death_count,
                "cause": c.primary_cause,
            }
            for i, c in enumerate(self.retries.death_clusters)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "playstyle_tag": self.playstyle_tag,
            "confidence_score": round(self.confidence_score, 3),
            "pacing_score": round(self.pacing_score, 3),
            "risk_tolerance_score": round(self.risk_tolerance_score, 3),
            "kinetic": self.kinetic.to_dict(),
            "evasion": self.evasion.to_dict(),
            "spatial": self.spatial.to_dict(),
            "retries": self.retries.to_dict(),
            "summary_features": {k: round(v, 4) for k, v in self.summary_features.items()},
            "analyzed_at_utc": self.analyzed_at_utc,
        }
