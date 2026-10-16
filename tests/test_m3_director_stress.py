"""tests/test_m3_director_stress.py
Empirical Challenger Stress & Adversarial Test Suite for Milestone 3:
  1. Extreme Input Values Stress:
     - NaN / Infinity in SteamPlayerDNA challenge_tolerance & pacing_score (ValueError / OverflowError detection).
     - Extreme DNA pacing scores (<-1.0, >1.0, massive floats) strictly bounded.
     - Extreme BehavioralProfile metrics (negative velocities, NaN, inf, extreme deaths, negative damage).
     - DifficultyClampState with NaN, Infinity, negative and zero baseline values.
  2. Multi-Episode Clamp Drift & Boundary Precision Jitter:
     - 50 consecutive episodes of max difficulty scaling (+15% each step).
     - 50 consecutive episodes of max difficulty reduction (-15% each step) and lower-bound violations.
     - Floating point boundary precision jitter (115.0 vs 115.0000001, recurring fractions).
  3. NavMesh Obstruction & Traversability Invariant Stress:
     - Rejection of critical path severing geometry across mid-arena.
     - Obstacle directly placed on player spawn point (120, 360) and mask detection.
     - Obstacle directly placed on exit point (1160, 360) and mask detection.
     - Direct verification of _verify_navmesh_traversability grid clearing defect.
  4. Cryptographic Envelope Verification & Tampering Stress:
     - Payload parameter mutation detection via SHA-256 integrity check.
     - Envelope hash bit flip detection.
     - File-backed disk payload mutation detection.
"""
from __future__ import annotations
import math
from pathlib import Path
import pytest

from engine.director.episodic import EpisodicDirector
from engine.director.genesis import GenesisDirector
from engine.director.models import (
    CurrentEpisodeConfig,
    DifficultyClampState,
    DirectiveAction,
    EpisodicUpdateSpec,
)
from engine.scheduler.artifact import (
    ArtifactIntegrityError,
    ArtifactMetadataEnvelope,
    ArtifactType,
)
from engine.steam.models import SteamPlayerDNA


# ============================================================================
# Test Fixtures & Mock Behavioral Profiles
# ============================================================================

class MockBehavioralProfile:
    deliberation: str = "Mock Behavioral Profile for empirical stress testing."

    def __init__(
        self,
        playstyle_tag: str = "kinetic_high_mobility",
        mean_velocity: float = 245.0,
        dash_ratio: float = 0.60,
        mastery_score: float = 0.85,
        deaths: int = 0,
        cleared: bool = True,
        death_clusters: list = None,
        camping_index: float = 0.10,
        damage_taken_total: float = 20.0,
    ):
        self.playstyle_tag = playstyle_tag
        self.mean_velocity = mean_velocity
        self.dash_ratio = dash_ratio
        self.mastery_score = mastery_score
        self.deaths = deaths
        self.cleared = cleared
        self.death_clusters = death_clusters or []
        self.camping_index = camping_index
        self.damage_taken_total = damage_taken_total


# ============================================================================
# 1. Extreme Input Values Stress Tests
# ============================================================================

class TestExtremeInputValuesStress:
    """Stress tests verifying engine behavior under extreme, non-standard, or malicious inputs."""
    deliberation: str = "Empirical stress testing of Cognitive Director Swarm with extreme, NaN, and infinity inputs."

    def test_genesis_dna_nan_challenge_tolerance_handling(self):
        """Stress: SteamPlayerDNA with NaN challenge_tolerance must be handled safely without unhandled exceptions."""
        dna = SteamPlayerDNA(
            deliberation="Testing NaN challenge tolerance resilience.",
            player_id="nan_tester",
            profile_tag="nan_profile",
            total_games_analyzed=10,
            total_playtime_hours=50.0,
            top_genres=["Action"],
            pacing_score=0.5,
            pacing_preference="kinetic_high_mobility",
            challenge_tolerance=float("nan"),
            mechanical_biases={},
            preferred_palette_mood="retro_arcade",
        )
        gd = GenesisDirector()
        # Must not raise unhandled ValueError: cannot convert float NaN to integer
        spec = gd.generate_genesis_spec(dna)
        assert spec is not None
        assert not math.isnan(spec.target_difficulty)
        assert len(spec.enemy_specs) > 0

    def test_genesis_dna_inf_challenge_tolerance_handling(self):
        """Stress: SteamPlayerDNA with infinity challenge_tolerance must be clamped safely without OverflowError."""
        dna = SteamPlayerDNA(
            deliberation="Testing infinity challenge tolerance resilience.",
            player_id="inf_tester",
            profile_tag="inf_profile",
            total_games_analyzed=10,
            total_playtime_hours=50.0,
            top_genres=["Action"],
            pacing_score=0.5,
            pacing_preference="kinetic_high_mobility",
            challenge_tolerance=float("inf"),
            mechanical_biases={},
            preferred_palette_mood="retro_arcade",
        )
        gd = GenesisDirector()
        # Must not raise unhandled OverflowError: cannot convert float infinity to integer
        spec = gd.generate_genesis_spec(dna)
        assert spec is not None
        assert spec.target_difficulty <= 2.0
        for e in spec.enemy_specs:
            assert e.count <= 20
            assert e.max_health <= 500.0

    def test_genesis_dna_extreme_pacing_scores_bounded(self):
        """Stress: Pacing scores beyond [-1.0, 1.0] must remain strictly bounded without arithmetic overflow."""
        gd = GenesisDirector()
        for extreme_score in [-100.0, -10.0, -2.5, 2.5, 10.0, 100.0]:
            dna = SteamPlayerDNA(
                deliberation=f"Testing extreme pacing score {extreme_score}.",
                player_id="pacing_extreme",
                profile_tag="pacing_extreme",
                total_games_analyzed=5,
                total_playtime_hours=100.0,
                top_genres=["Roguelike"],
                pacing_score=extreme_score,
                pacing_preference="kinetic_high_mobility" if extreme_score > 0 else "tactical_methodical",
                challenge_tolerance=0.5,
                mechanical_biases={},
                preferred_palette_mood="neon_cyberpunk",
            )
            spec = gd.generate_genesis_spec(dna)
            p = spec.player_params

            # Player speed and dash parameters must be bounded
            assert 200.0 <= p.walk_speed <= 360.0, f"walk_speed {p.walk_speed} out of bounds for score {extreme_score}"
            assert 450.0 <= p.dash_speed <= 850.0, f"dash_speed {p.dash_speed} out of bounds for score {extreme_score}"
            assert 0.35 <= p.dash_cooldown <= 1.00, f"dash_cooldown {p.dash_cooldown} out of bounds for score {extreme_score}"
            assert p.dash_cost > 0.0
            assert p.max_health > 0.0

    def test_episodic_director_extreme_profile_metrics(self):
        """Stress: EpisodicDirector given negative, infinite, or corrupted profile metrics produces a valid spec."""
        director = EpisodicDirector()
        corrupted_profile = MockBehavioralProfile(
            playstyle_tag="glitched_profile",
            mean_velocity=-999.0,
            dash_ratio=-5.0,
            mastery_score=-2.0,
            deaths=-10,
            cleared=False,
            damage_taken_total=-500.0,
            camping_index=-1.0,
        )
        spec = director.evaluate_and_update(corrupted_profile)
        assert spec.target_episode_id == "level_02"
        assert spec.updated_episode_config.enemy_speed > 0.0
        assert spec.updated_episode_config.enemy_damage > 0.0
        assert len(spec.deliberation) > 20

    def test_clamp_extreme_raw_values(self):
        """Stress: DifficultyClampState.apply_clamp with NaN, Inf, and massive raw values."""
        clamp = DifficultyClampState(max_adaptation_ratio=0.15)
        old_val = 100.0

        # +Inf raw value must be clamped to upper bound 115.0
        clamped_inf = clamp.apply_clamp("speed_inf", old_val, float("inf"))
        assert clamped_inf == 115.0
        assert clamp.clamps["speed_inf"].clamp_triggered is True

        # -Inf raw value must be clamped to lower bound 85.0
        clamped_neg_inf = clamp.apply_clamp("speed_neg_inf", old_val, float("-inf"))
        assert clamped_neg_inf == 85.0
        assert clamp.clamps["speed_neg_inf"].clamp_triggered is True

        # Massive positive float
        clamped_huge = clamp.apply_clamp("speed_huge", old_val, 1e12)
        assert clamped_huge == 115.0

        # Massive negative float
        clamped_neg_huge = clamp.apply_clamp("speed_neg_huge", old_val, -1e12)
        assert clamped_neg_huge == 85.0

    def test_clamp_zero_and_negative_p_old(self):
        """Stress: DifficultyClampState with non-positive baseline values."""
        clamp = DifficultyClampState(max_adaptation_ratio=0.15)

        # Zero old value: cannot calculate ratio, returns raw value without division by zero
        res_zero = clamp.apply_clamp("zero_param", 0.0, 50.0)
        assert res_zero == 50.0
        assert clamp.clamps["zero_param"].delta_percentage == 0.0

        # Negative old value
        res_neg = clamp.apply_clamp("neg_param", -10.0, 20.0)
        assert res_neg == 20.0


# ============================================================================
# 2. Multi-Episode Clamp Drift & Boundary Precision Jitter
# ============================================================================

class TestMultiEpisodeClampDriftAndPrecisionJitter:
    """Stress tests verifying mathematical stability across 50 consecutive episodes."""
    deliberation: str = "Empirical stress testing of bounded +/-15% adaptation clamp over 50 consecutive episodes."

    def test_50_episode_max_difficulty_scaling_up(self):
        """Multi-episode drift: 50 consecutive episodes of max mastery scaling up.
        Every single step must strictly obey p_new in [p_old, 1.15 * p_old] without negative or unbounded jumps.
        """
        director = EpisodicDirector(max_adaptation_ratio=0.15)
        cfg = CurrentEpisodeConfig(enemy_speed=100.0, enemy_damage=10.0, enemy_health=50.0)
        rusher = MockBehavioralProfile(
            playstyle_tag="master_rusher",
            mean_velocity=300.0,
            dash_ratio=0.70,
            mastery_score=1.0,
            deaths=0,
            cleared=True,
        )

        for ep in range(1, 51):
            spec = director.evaluate_and_update(rusher, cfg)
            new_cfg = spec.updated_episode_config

            for param_name, record in spec.clamp_state.clamps.items():
                p_old = record.previous_value
                p_new = record.clamped_value
                assert p_new > 0.0, f"Episode {ep} param {param_name} became non-positive: {p_new}"
                assert p_new >= p_old, f"Episode {ep} param {param_name} decreased on master player: {p_old} -> {p_new}"

                ratio = (p_new - p_old) / p_old
                assert ratio <= 0.150001, f"Episode {ep} param {param_name} exceeded +15% upper clamp: ratio {ratio:.6f}"

            cfg = new_cfg

        assert cfg.episode_number == 51
        expected_speed = 100.0 * (1.15 ** 50)
        assert pytest.approx(cfg.enemy_speed, rel=1e-2) == expected_speed

    def test_50_episode_max_difficulty_scaling_down_strict_clamp_adherence(self):
        """Multi-episode drift: 50 consecutive episodes of struggling failure scaling down.
        Every single step must strictly obey p_new in [0.85 * p_old, p_old].
        Detects rounding drift defect where round(p_old * 0.85, 6) causes ratio < -15.0%.
        """
        director = EpisodicDirector(max_adaptation_ratio=0.15)
        cfg = CurrentEpisodeConfig(enemy_speed=100.0, enemy_damage=10.0, enemy_health=50.0)
        failing_player = MockBehavioralProfile(
            playstyle_tag="struggling_player",
            mean_velocity=40.0,
            dash_ratio=0.05,
            mastery_score=0.0,
            deaths=10,
            cleared=False,
            death_clusters=[],
            camping_index=0.85,
        )

        violations = []
        for ep in range(1, 51):
            spec = director.evaluate_and_update(failing_player, cfg)
            new_cfg = spec.updated_episode_config

            for param_name, record in spec.clamp_state.clamps.items():
                p_old = record.previous_value
                p_new = record.clamped_value
                assert p_new > 0.0, f"Episode {ep} param {param_name} dropped to non-positive: {p_new}"

                if p_old > 0.0:
                    ratio = (p_new - p_old) / p_old
                    # Invariant: p_new >= 0.85 * p_old strictly
                    if ratio < -0.15000001:
                        violations.append((ep, param_name, p_old, p_new, ratio))

            cfg = new_cfg

        assert not violations, (
            f"Encountered {len(violations)} clamp violations below -15% over 50 episodes. "
            f"First violation: Ep {violations[0][0]}, param '{violations[0][1]}', "
            f"p_old={violations[0][2]}, p_new={violations[0][3]}, delta ratio={violations[0][4]:.8f}"
        )

    def test_clamp_floating_point_boundary_precision_jitter(self):
        """Precision jitter: Boundary testing around exact 15% thresholds."""
        clamp = DifficultyClampState(max_adaptation_ratio=0.15)
        p_old = 100.0

        # Exact boundary: 100.0 * 1.15 = 115.0
        val_exact = clamp.apply_clamp("speed_exact", p_old, 115.0)
        assert val_exact == 115.0
        assert clamp.clamps["speed_exact"].clamp_triggered is False

        # Strictly above boundary: 115.00001 must trigger clamp
        val_above = clamp.apply_clamp("speed_above", p_old, 115.00001)
        assert val_above == 115.0
        assert clamp.clamps["speed_above"].clamp_triggered is True

        # Strictly below lower boundary: 84.99999 must trigger clamp
        val_below = clamp.apply_clamp("speed_below", p_old, 84.99999)
        assert val_below == 85.0
        assert clamp.clamps["speed_below"].clamp_triggered is True

        # Micro jitter: 115.0000001 exceeds upper bound, clamp must be triggered
        clamp.apply_clamp("speed_micro", p_old, 115.0000001)
        assert clamp.clamps["speed_micro"].clamp_triggered is True, (
            "Micro-increment 115.0000001 failed to register as clamp_triggered due to 6-decimal rounding!"
        )


# ============================================================================
# 3. NavMesh Obstruction & Traversability Invariant Stress Tests
# ============================================================================

class TestNavMeshObstructionAndTraversabilityStress:
    """Stress tests verifying critical-path preservation and obstacle placement safety."""
    deliberation: str = "Empirical stress testing of A* NavMesh traversability and obstacle placement/relocation."

    def test_critical_path_severing_wall_rejected(self):
        """NavMesh Stress: A proposed obstacle severing the arena from top to bottom must be rejected."""
        director = EpisodicDirector()
        world_bounds = (1280.0, 720.0)
        spawn_pt = (120.0, 360.0)
        exit_pt = (1160.0, 360.0)

        # Baseline walls + a vertical barrier splitting the arena across x=640
        severing_wall = [(640.0, 0.0, 32.0, 720.0)]
        is_traversable = director._verify_navmesh_traversability(
            severing_wall,
            world_bounds,
            spawn_pt,
            exit_pt
        )
        assert is_traversable is False, "A wall fully severing the arena was falsely verified as traversable!"

    def test_navmesh_traversability_masks_spawn_wall_defect(self):
        """NavMesh Defect: _verify_navmesh_traversability must NOT report True if spawn tile is inside a wall.
        Tests whether grid[sy][sx] = 0 incorrectly clears solid obstacles placed on spawn.
        """
        director = EpisodicDirector()
        world_bounds = (1280.0, 720.0)
        spawn_pt = (120.0, 360.0)
        exit_pt = (1160.0, 360.0)

        # Wall placed directly on spawn coordinates
        wall_on_spawn = [(120.0, 360.0, 32.0, 32.0)]
        is_traversable = director._verify_navmesh_traversability(
            wall_on_spawn,
            world_bounds,
            spawn_pt,
            exit_pt
        )
        assert is_traversable is False, (
            "Defect: _verify_navmesh_traversability reported True when an obstacle directly covers the spawn point!"
        )

    def test_navmesh_traversability_masks_exit_wall_defect(self):
        """NavMesh Defect: _verify_navmesh_traversability must NOT report True if exit tile is inside a wall.
        Tests whether grid[ey][ex] = 0 incorrectly clears solid obstacles placed on exit.
        """
        director = EpisodicDirector()
        world_bounds = (1280.0, 720.0)
        spawn_pt = (120.0, 360.0)
        exit_pt = (1160.0, 360.0)

        # Wall placed directly on exit coordinates
        wall_on_exit = [(1160.0, 360.0, 32.0, 32.0)]
        is_traversable = director._verify_navmesh_traversability(
            wall_on_exit,
            world_bounds,
            spawn_pt,
            exit_pt
        )
        assert is_traversable is False, (
            "Defect: _verify_navmesh_traversability reported True when an obstacle directly covers the exit portal!"
        )

    def test_obstacle_directly_on_spawn_point_must_be_rejected_or_relocated(self):
        """NavMesh Stress: When a player dies repeatedly at the spawn point (120, 360),
        EpisodicDirector MUST NOT place a cover obstacle overlapping the spawn point.
        """
        director = EpisodicDirector()
        profile = MockBehavioralProfile(
            playstyle_tag="struggling_rusher",
            mastery_score=0.20,
            deaths=5,
            cleared=False,
            death_clusters=[{"id": "spawn_cluster", "x": 120.0, "y": 360.0, "count": 5}],
        )

        spec = director.evaluate_and_update(profile)

        # Verify no spatial directive overlaps the spawn point (120, 360)
        spawn_x, spawn_y = 120.0, 360.0
        for directive in spec.spatial_directives:
            bx, by, bw, bh = directive.bounds
            overlaps_spawn = (bx <= spawn_x <= bx + bw) and (by <= spawn_y <= by + bh)
            assert not overlaps_spawn, (
                f"Defect: Proposed spatial obstacle at ({bx:.1f}, {by:.1f}, {bw:.1f}, {bh:.1f}) "
                f"directly covers player spawn point ({spawn_x}, {spawn_y}) with navmesh_verified={directive.navmesh_verified}!"
            )

    def test_obstacle_directly_on_exit_point_must_be_rejected_or_relocated(self):
        """NavMesh Stress: When a player dies repeatedly at the exit portal (1160, 360),
        EpisodicDirector MUST NOT place a cover obstacle overlapping the exit point.
        """
        director = EpisodicDirector()
        profile = MockBehavioralProfile(
            playstyle_tag="struggling_rusher",
            mastery_score=0.20,
            deaths=5,
            cleared=False,
            death_clusters=[{"id": "exit_cluster", "x": 1160.0, "y": 360.0, "count": 5}],
        )

        spec = director.evaluate_and_update(profile)

        # Verify no spatial directive overlaps the exit point (1160, 360)
        exit_x, exit_y = 1160.0, 360.0
        for directive in spec.spatial_directives:
            bx, by, bw, bh = directive.bounds
            overlaps_exit = (bx <= exit_x <= bx + bw) and (by <= exit_y <= by + bh)
            assert not overlaps_exit, (
                f"Defect: Proposed spatial obstacle at ({bx:.1f}, {by:.1f}, {bw:.1f}, {bh:.1f}) "
                f"directly covers exit point ({exit_x}, {exit_y}) with navmesh_verified={directive.navmesh_verified}!"
            )


# ============================================================================
# 4. Cryptographic Envelope Verification & Tampering Stress Tests
# ============================================================================

class TestCryptographicEnvelopeTamperingStress:
    """Stress tests verifying tamper-detection of sealed episodic artifacts."""
    deliberation: str = "Empirical stress testing of ArtifactMetadataEnvelope cryptographic tamper detection."

    def test_sealed_update_spec_payload_mutation_detected(self):
        """Tampering Stress: Mutating any field within a sealed spec payload must invalidate SHA256 integrity."""
        director = EpisodicDirector()
        profile = MockBehavioralProfile()
        spec = director.evaluate_and_update(profile)

        envelope = director.seal_update_artifact(spec, run_id="stress_tamper_01")
        assert envelope.verify_integrity(raise_on_error=True) is True

        # Malicious modification: modify enemy_speed in in-memory payload
        envelope.payload["updated_episode_config"]["enemy_speed"] += 5.0

        assert envelope.verify_integrity(raise_on_error=False) is False
        with pytest.raises(ArtifactIntegrityError) as exc_info:
            envelope.verify_integrity(raise_on_error=True)
        assert "integrity verification failed" in str(exc_info.value).lower()

    def test_sealed_update_spec_hash_tampering_detected(self):
        """Tampering Stress: Altering the recorded sha256_hash must fail verification."""
        director = EpisodicDirector()
        profile = MockBehavioralProfile()
        spec = director.evaluate_and_update(profile)

        envelope = director.seal_update_artifact(spec, run_id="stress_hash_tamper")
        original_hash = envelope.sha256_hash

        # Flip last hex character
        tampered_hash = original_hash[:-1] + ("0" if original_hash[-1] != "0" else "1")
        envelope.sha256_hash = tampered_hash

        assert envelope.verify_integrity(raise_on_error=False) is False
        with pytest.raises(ArtifactIntegrityError):
            envelope.verify_integrity(raise_on_error=True)

    def test_file_backed_artifact_disk_tampering_detected(self, tmp_path: Path):
        """Tampering Stress: External file payload modified on disk after sealing must fail verification."""
        director = EpisodicDirector(specs_dir=tmp_path / "specs")
        profile = MockBehavioralProfile()
        spec = director.evaluate_and_update(profile)

        # Write payload to disk and seal file-backed envelope
        spec_path = director.write_update_spec(spec)
        envelope = ArtifactMetadataEnvelope.seal(
            deliberation=spec.deliberation,
            artifact_type=ArtifactType.UPDATE_SPEC,
            payload_path=spec_path,
            source_agent="EpisodicDirector",
            target_agent="SWEWorkerCompiler",
        )
        assert envelope.verify_integrity(raise_on_error=True) is True

        # Mutate single character in the file on disk
        tampered_content = spec_path.read_text(encoding="utf-8").replace("Episode 2", "Episode 99")
        spec_path.write_text(tampered_content, encoding="utf-8")

        assert envelope.verify_integrity(raise_on_error=False) is False
        with pytest.raises(ArtifactIntegrityError):
            envelope.verify_integrity(raise_on_error=True)
