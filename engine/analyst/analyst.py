"""engine/analyst/analyst.py
Telemetry Analyst Engine — parses raw SQLite WAL event streams and computes
kinetic vectors, evasion metrics, 2D spatial heatmaps, retry curves, and behavioral profiles.
Strictly adheres to Rationale-First Epistemic Architecture.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from config import DB_PATH
from engine.analyst.models import (
    BehavioralProfile,
    DashMetrics,
    DeathCluster,
    KineticMetrics,
    PlaystyleClassification,
    RetryMetrics,
    SpatialOccupancyMap,
)
from engine.telemetry.db import TelemetryDB


class TelemetryAnalyst:
    """Parses raw SQLite telemetry events into a structured BehavioralProfile."""
    deliberation: str = "Analyzing player telemetry streams for kinetic velocity, spatial occupancy, and behavioral playstyle."

    def __init__(
        self,
        db_path: Union[str, Path] = DB_PATH,
        noise_velocity_threshold: float = 5.0,
        grid_resolution: Tuple[int, int] = (16, 16),
        world_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 1280.0, 720.0),
        threat_proximity_threshold: float = 150.0,
        death_clustering_radius: float = 80.0,
    ):
        self.db_path = Path(db_path)
        self.noise_threshold = float(noise_velocity_threshold)
        self.grid_w, self.grid_h = grid_resolution
        self.world_bounds = world_bounds
        self.threat_threshold = float(threat_proximity_threshold)
        self.death_radius = float(death_clustering_radius)

    def _get_reader(self) -> sqlite3.Connection:
        if self.db_path.exists():
            tdb = TelemetryDB(self.db_path)
            return tdb.get_reader_connection()
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------------
    # Public Ingress Queries
    # ------------------------------------------------------------------------

    def analyze(
        self,
        session_id: str,
        episode_id: str,
        events: List[Dict[str, Any]],
        episodes: Optional[List[Dict[str, Any]]] = None,
    ) -> BehavioralProfile:
        """Convenience method matching showrunner harness caller interface."""
        return self.analyze_events(
            events=events,
            session_id=session_id,
            episode_id=episode_id,
            episodes=episodes,
        )

    def analyze_session(self, session_id: str) -> BehavioralProfile:
        """Queries all episodes and events for a session from SQLite WAL and returns profile."""
        conn = self._get_reader()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT session_id, episode_id, timestamp, event_type,
                       pos_x, pos_y, vel_x, vel_y, health, payload_json, frame_index
                FROM telemetry_events
                WHERE session_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (session_id,),
            )
            raw_events = [dict(row) for row in cur.fetchall()]

            cur.execute(
                """
                SELECT episode_id, session_id, episode_number, status,
                       duration_seconds, outcome, score, damage_dealt_total,
                       damage_taken_total, dashes_total, shots_fired_total
                FROM telemetry_episodes
                WHERE session_id = ?
                ORDER BY episode_number ASC
                """,
                (session_id,),
            )
            raw_episodes = [dict(row) for row in cur.fetchall()]

            return self.process_dataset(
                session_id=session_id,
                episode_id=None,
                events=raw_events,
                episodes=raw_episodes,
            )
        finally:
            conn.close()

    def analyze_episode(self, episode_id: str) -> BehavioralProfile:
        """Queries a single episode from SQLite WAL and returns profile."""
        conn = self._get_reader()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT session_id, episode_id, timestamp, event_type,
                       pos_x, pos_y, vel_x, vel_y, health, payload_json, frame_index
                FROM telemetry_events
                WHERE episode_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (episode_id,),
            )
            raw_events = [dict(row) for row in cur.fetchall()]

            cur.execute(
                """
                SELECT episode_id, session_id, episode_number, status,
                       duration_seconds, outcome, score, damage_dealt_total,
                       damage_taken_total, dashes_total, shots_fired_total
                FROM telemetry_episodes
                WHERE episode_id = ?
                """,
                (episode_id,),
            )
            raw_episodes = [dict(row) for row in cur.fetchall()]
            session_id = raw_episodes[0]["session_id"] if raw_episodes else (raw_events[0]["session_id"] if raw_events else "unknown")

            return self.process_dataset(
                session_id=session_id,
                episode_id=episode_id,
                events=raw_events,
                episodes=raw_episodes,
            )
        finally:
            conn.close()

    def analyze_events(
        self,
        events: List[Dict[str, Any]],
        session_id: str = "mem_session",
        episode_id: Optional[str] = "mem_ep_01",
        episodes: Optional[List[Dict[str, Any]]] = None,
    ) -> BehavioralProfile:
        """Direct in-memory analysis for synthetic tests and fast evaluations."""
        return self.process_dataset(
            session_id=session_id,
            episode_id=episode_id,
            events=events,
            episodes=episodes or [],
        )

    # ------------------------------------------------------------------------
    # Pipeline Processing Core
    # ------------------------------------------------------------------------

    def process_dataset(
        self,
        session_id: str,
        episode_id: Optional[str],
        events: List[Dict[str, Any]],
        episodes: List[Dict[str, Any]],
    ) -> BehavioralProfile:
        """Core mathematical processing pipeline."""
        normalized_events = self._normalize_events(events)

        kinetic = self._compute_kinetic_metrics(normalized_events)
        evasion = self._compute_dash_metrics(normalized_events)
        spatial = self._compute_spatial_metrics(normalized_events)
        retries = self._compute_retry_metrics(normalized_events, episodes)

        playstyle, confidence, pacing_score, risk_tolerance, deliberation = self._synthesize_classification(
            kinetic, evasion, spatial, retries, len(normalized_events), len(episodes)
        )

        summary = {
            "mean_velocity": kinetic.mean_velocity,
            "peak_velocity": kinetic.peak_velocity,
            "acceleration_variance": kinetic.acceleration_variance,
            "movement_entropy": kinetic.movement_entropy,
            "dash_frequency": evasion.dash_frequency,
            "evasion_ratio": evasion.evasion_ratio,
            "camping_index": spatial.camping_index,
            "roaming_index": spatial.roaming_index,
            "spatial_coverage_ratio": spatial.spatial_coverage_ratio,
            "pacing_score": pacing_score,
            "risk_tolerance_score": risk_tolerance,
        }

        return BehavioralProfile(
            deliberation=deliberation,
            session_id=session_id,
            episode_id=episode_id,
            playstyle_tag=playstyle.value,
            confidence_score=confidence,
            pacing_score=pacing_score,
            risk_tolerance_score=risk_tolerance,
            kinetic=kinetic,
            evasion=evasion,
            spatial=spatial,
            retries=retries,
            summary_features=summary,
        )

    # ------------------------------------------------------------------------
    # Mathematical Feature Extractors
    # ------------------------------------------------------------------------

    def _normalize_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for ev in events:
            ev_type = str(ev.get("event_type", "")).upper()
            pos = ev.get("pos")
            if pos and isinstance(pos, (list, tuple)) and len(pos) >= 2:
                pos_x, pos_y = float(pos[0]), float(pos[1])
            else:
                pos_x = float(ev.get("pos_x", 0.0))
                pos_y = float(ev.get("pos_y", 0.0))

            vel = ev.get("vel")
            if vel and isinstance(vel, (list, tuple)) and len(vel) >= 2:
                vel_x, vel_y = float(vel[0]), float(vel[1])
            else:
                vel_x = float(ev.get("vel_x", 0.0))
                vel_y = float(ev.get("vel_y", 0.0))

            payload = ev.get("payload")
            if payload is None:
                p_json = ev.get("payload_json")
                if p_json and isinstance(p_json, str):
                    try:
                        payload = json.loads(p_json)
                    except Exception:
                        payload = {}
                else:
                    payload = {}

            normalized.append({
                "timestamp": float(ev.get("timestamp", 0.0)),
                "event_type": ev_type,
                "pos_x": pos_x,
                "pos_y": pos_y,
                "vel_x": vel_x,
                "vel_y": vel_y,
                "health": ev.get("health"),
                "payload": payload,
            })
        return normalized

    def _compute_kinetic_metrics(self, events: List[Dict[str, Any]]) -> KineticMetrics:
        movement_events = [
            e for e in events
            if e["event_type"] in ("MOVEMENT_TICK", "PLAYER_MOVE", "DASH", "PLAYER_DASH")
        ]
        if not movement_events:
            return KineticMetrics(
                deliberation="No movement ticks recorded in telemetry stream. Yielding zero-velocity baseline."
            )

        active_speeds: List[float] = []
        accelerations: List[float] = []
        angle_bins = [0] * 8
        idle_ticks = 0

        prev_vx: Optional[float] = None
        prev_vy: Optional[float] = None
        prev_t: Optional[float] = None

        for ev in movement_events:
            vx, vy = ev["vel_x"], ev["vel_y"]
            speed = math.sqrt(vx * vx + vy * vy)
            t = ev["timestamp"]

            if speed < self.noise_threshold:
                idle_ticks += 1
            else:
                active_speeds.append(speed)

                angle = math.atan2(vy, vx)  # [-pi, pi]
                bin_idx = int(math.floor(((angle + math.pi) / (2.0 * math.pi)) * 8)) % 8
                angle_bins[bin_idx] += 1

                if prev_vx is not None and prev_vy is not None and prev_t is not None:
                    dt = t - prev_t
                    if 0.001 <= dt <= 1.0:
                        dvx = (vx - prev_vx) / dt
                        dvy = (vy - prev_vy) / dt
                        acc = math.sqrt(dvx * dvx + dvy * dvy)
                        accelerations.append(acc)

                prev_vx, prev_vy, prev_t = vx, vy, t

        n_active = len(active_speeds)
        mean_v = sum(active_speeds) / n_active if n_active > 0 else 0.0
        peak_v = max(active_speeds) if active_speeds else 0.0
        var_v = sum((s - mean_v) ** 2 for s in active_speeds) / n_active if n_active > 0 else 0.0

        n_acc = len(accelerations)
        mean_a = sum(accelerations) / n_acc if n_acc > 0 else 0.0
        var_a = sum((a - mean_a) ** 2 for a in accelerations) / n_acc if n_acc > 0 else 0.0

        entropy = 0.0
        if n_active > 0:
            for count in angle_bins:
                if count > 0:
                    p = count / n_active
                    entropy -= p * math.log2(p)
            entropy = entropy / 3.0

        idle_ratio = idle_ticks / len(movement_events) if movement_events else 0.0
        active_duration = n_active * 0.016667

        deliberation = (
            f"Evaluated {len(movement_events)} movement ticks ({n_active} active, {idle_ticks} idle). "
            f"Mean speed is {mean_v:.1f} px/s (peak {peak_v:.1f} px/s) with acceleration variance {var_a:.1f}. "
            f"Directional trajectory entropy is {entropy:.3f} across 8 octants."
        )

        return KineticMetrics(
            deliberation=deliberation,
            mean_velocity=mean_v,
            peak_velocity=peak_v,
            velocity_variance=var_v,
            acceleration_mean=mean_a,
            acceleration_variance=var_a,
            movement_entropy=entropy,
            idle_ratio=idle_ratio,
            active_movement_duration=active_duration,
            total_movement_ticks=len(movement_events),
        )

    def _compute_dash_metrics(self, events: List[Dict[str, Any]]) -> DashMetrics:
        dash_events = [
            e for e in events if e["event_type"] in ("DASH", "PLAYER_DASH")
        ]
        movement_events = [
            e for e in events
            if e["event_type"] in ("MOVEMENT_TICK", "PLAYER_MOVE", "DASH", "PLAYER_DASH")
        ]
        damage_events = [
            e for e in events
            if e["event_type"] in ("DAMAGE_TAKEN", "PLAYER_DAMAGE", "HAZARD_HIT")
        ]

        if not dash_events:
            return DashMetrics(
                deliberation="Zero dash events observed. Player relies purely on standard locomotion without active evasive burst."
            )

        t_min = min(e["timestamp"] for e in events) if events else 0.0
        t_max = max(e["timestamp"] for e in events) if events else 0.0
        duration_minutes = max((t_max - t_min) / 60.0, 0.01)

        dash_count = len(dash_events)
        dash_freq = dash_count / duration_minutes
        dash_ratio = dash_count / max(len(movement_events), 1)

        evasive_count = 0
        proximities: List[float] = []
        under_damage_count = 0

        damage_times = [d["timestamp"] for d in damage_events]

        for dash in dash_events:
            dt_dash = dash["timestamp"]
            p = dash.get("payload", {})
            prox = p.get("hazard_proximity") or p.get("nearest_enemy_dist") or p.get("nearest_threat_dist")
            if prox is not None:
                prox_val = float(prox)
                proximities.append(prox_val)
                if prox_val <= self.threat_threshold:
                    evasive_count += 1

            if any(abs(dt_dash - dmg_t) <= 0.5 for dmg_t in damage_times):
                under_damage_count += 1
                evasive_count = max(evasive_count, under_damage_count)

        mean_prox = sum(proximities) / len(proximities) if proximities else 0.0
        evasion_ratio = evasive_count / dash_count if dash_count > 0 else 0.0

        deliberation = (
            f"Detected {dash_count} dash maneuvers ({dash_freq:.1f} dashes/min, {dash_ratio:.1%} of locomotion). "
            f"Identified {evasive_count} evasive maneuvers under threat ({evasion_ratio:.1%} evasion ratio)."
        )

        return DashMetrics(
            deliberation=deliberation,
            dash_count=dash_count,
            dash_frequency=dash_freq,
            dash_ratio=dash_ratio,
            evasion_ratio=evasion_ratio,
            mean_hazard_proximity_at_dash=mean_prox,
            dash_under_damage_count=under_damage_count,
        )

    def _compute_spatial_metrics(self, events: List[Dict[str, Any]]) -> SpatialOccupancyMap:
        grid = [[0.0 for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        min_x, min_y, max_x, max_y = self.world_bounds
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)

        loc_events = [e for e in events if e.get("pos_x") is not None and e.get("pos_y") is not None]
        if not loc_events:
            return SpatialOccupancyMap(
                deliberation="No spatial coordinates found in telemetry stream.",
                occupancy_matrix=grid,
            )

        sum_x = 0.0
        sum_y = 0.0
        total_ticks = len(loc_events)

        for ev in loc_events:
            px, py = ev["pos_x"], ev["pos_y"]
            sum_x += px
            sum_y += py

            norm_x = max(0.0, min(0.9999, (px - min_x) / span_x))
            norm_y = max(0.0, min(0.9999, (py - min_y) / span_y))
            gx = int(norm_x * self.grid_w)
            gy = int(norm_y * self.grid_h)
            grid[gy][gx] += 1.0

        unique_cells = 0
        flat_occupancies = []
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                if grid[gy][gx] > 0:
                    unique_cells += 1
                grid[gy][gx] /= total_ticks
                flat_occupancies.append(grid[gy][gx])

        total_cells = self.grid_w * self.grid_h
        coverage_ratio = unique_cells / total_cells if total_cells > 0 else 0.0

        entropy = 0.0
        for p in flat_occupancies:
            if p > 0.0:
                entropy -= p * math.log2(p)
        entropy = entropy / math.log2(total_cells) if total_cells > 1 else 0.0

        top_k = max(1, int(math.ceil(0.10 * unique_cells))) if unique_cells > 0 else 1
        sorted_occ = sorted(flat_occupancies, reverse=True)
        camping_index = sum(sorted_occ[:top_k])
        roaming_index = min(1.0, max(0.0, coverage_ratio * (1.0 - (camping_index - 0.10))))

        centroid_x = sum_x / total_ticks
        centroid_y = sum_y / total_ticks

        deliberation = (
            f"Mapped {total_ticks} spatial samples into {self.grid_w}x{self.grid_h} grid. "
            f"Player covered {unique_cells}/{total_cells} cells ({coverage_ratio:.1%} coverage). "
            f"Camping index is {camping_index:.2f} (top 10% cells), roaming index is {roaming_index:.2f}, "
            f"with spatial entropy {entropy:.3f}."
        )

        return SpatialOccupancyMap(
            deliberation=deliberation,
            grid_width=self.grid_w,
            grid_height=self.grid_h,
            world_bounds=self.world_bounds,
            occupancy_matrix=grid,
            unique_cells_visited=unique_cells,
            spatial_coverage_ratio=coverage_ratio,
            residency_time_entropy=entropy,
            camping_index=camping_index,
            roaming_index=roaming_index,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
        )

    def _compute_retry_metrics(
        self,
        events: List[Dict[str, Any]],
        episodes: List[Dict[str, Any]],
    ) -> RetryMetrics:
        death_events = [
            e for e in events
            if e["event_type"] in ("PLAYER_DEATH", "DEATH")
        ]
        total_deaths = len(death_events)

        death_clusters: List[DeathCluster] = []
        unclustered = list(death_events)

        while unclustered:
            pivot = unclustered.pop(0)
            cluster_members = [pivot]
            px, py = pivot["pos_x"], pivot["pos_y"]

            remaining = []
            for other in unclustered:
                ox, oy = other["pos_x"], other["pos_y"]
                dist = math.sqrt((px - ox) ** 2 + (py - oy) ** 2)
                if dist <= self.death_radius:
                    cluster_members.append(other)
                else:
                    remaining.append(other)
            unclustered = remaining

            cx = sum(m["pos_x"] for m in cluster_members) / len(cluster_members)
            cy = sum(m["pos_y"] for m in cluster_members) / len(cluster_members)
            max_r = max(math.sqrt((m["pos_x"] - cx) ** 2 + (m["pos_y"] - cy) ** 2) for m in cluster_members)

            causes = [m.get("payload", {}).get("cause", "hazard") for m in cluster_members]
            prim_cause = max(set(causes), key=causes.count) if causes else "hazard"

            cluster_delib = (
                f"Cluster of {len(cluster_members)} deaths localized at ({cx:.1f}, {cy:.1f}) "
                f"within {max_r:.1f}px radius. Primary cause: {prim_cause}."
            )
            death_clusters.append(
                DeathCluster(
                    deliberation=cluster_delib,
                    center_x=cx,
                    center_y=cy,
                    radius=max_r,
                    death_count=len(cluster_members),
                    primary_cause=prim_cause,
                )
            )

        n_episodes = len(episodes)
        cleared = sum(1 for ep in episodes if str(ep.get("outcome", "")).upper() in ("CLEARED", "SUCCESS", "VICTORY"))
        failed = sum(1 for ep in episodes if str(ep.get("outcome", "")).upper() in ("FAILED", "DEFEAT", "DIED", "ABORTED"))
        clear_rate = cleared / n_episodes if n_episodes > 0 else 0.0

        durations = [float(ep.get("duration_seconds", 0.0)) for ep in episodes if ep.get("duration_seconds")]
        mean_dur = sum(durations) / len(durations) if durations else 0.0
        var_dur = sum((d - mean_dur) ** 2 for d in durations) / len(durations) if durations else 0.0

        trend = 0.0
        if len(durations) >= 2:
            n = len(durations)
            x_vals = list(range(n))
            mean_x = (n - 1) / 2.0
            cov_xy = sum((x - mean_x) * (y - mean_dur) for x, y in zip(x_vals, durations))
            var_x = sum((x - mean_x) ** 2 for x in x_vals)
            trend = cov_xy / var_x if var_x > 0 else 0.0

        total_time_min = sum(durations) / 60.0 if durations else 1.0
        death_rate = total_deaths / max(total_time_min, 0.1)

        deliberation = (
            f"Analyzed {n_episodes} episodes ({cleared} cleared, {failed} failed). "
            f"Mean duration is {mean_dur:.1f}s (trend slope {trend:+.2f}). "
            f"Recorded {total_deaths} deaths across {len(death_clusters)} spatial clusters."
        )

        return RetryMetrics(
            deliberation=deliberation,
            total_episodes_analyzed=n_episodes,
            cleared_episodes=cleared,
            failed_episodes=failed,
            clear_rate=clear_rate,
            mean_duration_seconds=mean_dur,
            duration_variance=var_dur,
            total_deaths=total_deaths,
            death_rate_per_minute=death_rate,
            death_clusters=death_clusters,
            progress_trend=trend,
        )

    # ------------------------------------------------------------------------
    # Behavioral Playstyle Synthesis
    # ------------------------------------------------------------------------

    def _synthesize_classification(
        self,
        kinetic: KineticMetrics,
        evasion: DashMetrics,
        spatial: SpatialOccupancyMap,
        retries: RetryMetrics,
        n_events: int,
        n_episodes: int,
    ) -> Tuple[PlaystyleClassification, float, float, float, str]:
        """Synthesizes high-level playstyle archetype and deliberation rationale."""
        v_norm = min(1.0, kinetic.mean_velocity / 260.0)
        dash_norm = min(1.0, evasion.dash_ratio / 0.40)
        roam_norm = spatial.roaming_index
        mobility_score = 0.45 * v_norm + 0.35 * dash_norm + 0.20 * roam_norm

        tactical_score = 0.50 * spatial.camping_index + 0.30 * kinetic.idle_ratio + 0.20 * (1.0 - v_norm)

        pacing_score = max(-1.0, min(1.0, (mobility_score - tactical_score) * 1.5))
        risk_tolerance = max(0.0, min(1.0, 0.4 * v_norm + 0.3 * evasion.evasion_ratio + 0.3 * (1.0 - spatial.camping_index)))

        if mobility_score >= 0.60 and evasion.dash_ratio >= 0.25:
            classification = PlaystyleClassification.KINETIC_HIGH_MOBILITY
            confidence = min(0.95, 0.60 + 0.35 * (mobility_score - 0.60) / 0.40)
        elif tactical_score >= 0.55 and kinetic.mean_velocity <= 140.0:
            classification = PlaystyleClassification.CAUTIOUS_RANGED
            confidence = min(0.92, 0.55 + 0.35 * (tactical_score - 0.55) / 0.45)
        elif spatial.camping_index >= 0.45 or kinetic.idle_ratio >= 0.25:
            classification = PlaystyleClassification.METHODICAL_TACTICAL
            confidence = 0.78
        else:
            classification = PlaystyleClassification.BALANCED_SKIRMISHER
            confidence = 0.70

        deliberation = (
            f"Epistemic synthesis of {n_events} raw telemetry events across {n_episodes} episodes: "
            f"Player exhibits mean velocity {kinetic.mean_velocity:.1f} px/s, dash ratio {evasion.dash_ratio:.1%}, "
            f"camping index {spatial.camping_index:.2f}, and roaming index {spatial.roaming_index:.2f}. "
            f"Composite mobility score is {mobility_score:.2f} vs tactical score {tactical_score:.2f}, "
            f"yielding continuous pacing score {pacing_score:+.2f} and risk tolerance {risk_tolerance:.2f}. "
            f"Behavioral evidence firmly classifies playstyle as '{classification.value}' (confidence: {confidence:.0%})."
        )

        return classification, confidence, pacing_score, risk_tolerance, deliberation
