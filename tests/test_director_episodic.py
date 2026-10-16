"""tests/test_director_episodic.py
Comprehensive unit test suite verifying:
  1. Strict Rationale-First Epistemic Architecture (deliberation declared first in AST and dicts).
  2. Bounded Between-Run Adaptation Clamp (+/-15% max adaptation invariant).
  3. Multi-Episode Stability & Continuous Geometric Progression.
  4. Spatial Directives & NavMesh Critical-Path Traversability Preservation.
  5. Cryptographic ArtifactMetadataEnvelope Integration and Markdown spec format.
"""
import ast
from pathlib import Path
import random
import pytest

from engine.director.models import (
    CurrentEpisodeConfig,
    DifficultyClampRecord,
    DifficultyClampState,
    DirectiveAction,
    EpisodicUpdateSpec,
    MechanicalUpdateDirective,
    SpatialUpdateDirective,
)
from engine.director.episodic import EpisodicDirector
from engine.scheduler.artifact import ArtifactType
from tests.test_tier1_units import validate_rationale_first_ast, validate_rationale_first_dict


# ============================================================================
# Test Fixtures: Synthetic Behavioral Profiles
# ============================================================================

class MockBehavioralProfile:
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
    ):
        self.playstyle_tag = playstyle_tag
        self.mean_velocity = mean_velocity
        self.dash_ratio = dash_ratio
        self.mastery_score = mastery_score
        self.deaths = deaths
        self.cleared = cleared
        self.death_clusters = death_clusters or []
        self.camping_index = camping_index


@pytest.fixture
def rusher_profile():
    return MockBehavioralProfile(
        playstyle_tag="kinetic_high_mobility",
        mean_velocity=260.0,
        dash_ratio=0.65,
        mastery_score=0.90,
        deaths=0,
        cleared=True,
    )


@pytest.fixture
def struggling_bottleneck_profile():
    return MockBehavioralProfile(
        playstyle_tag="struggling_rusher",
        mean_velocity=190.0,
        dash_ratio=0.45,
        mastery_score=0.20,
        deaths=4,
        cleared=False,
        death_clusters=[{"id": "laser_cluster", "x": 780.0, "y": 360.0, "count": 4}],
    )


@pytest.fixture
def cautious_camper_profile():
    return MockBehavioralProfile(
        playstyle_tag="cautious_ranged",
        mean_velocity=110.0,
        dash_ratio=0.08,
        mastery_score=0.55,
        deaths=1,
        cleared=True,
        camping_index=0.72,
    )


# ============================================================================
# 1. Rationale-First Epistemic Architecture Tests
# ============================================================================

class TestRationaleFirstEpistemicCompliance:
    """Verifies that deliberation is strictly the first attribute across models and serializations."""

    def test_models_declare_deliberation_first_via_ast(self):
        root = Path(__file__).parent.parent
        models_file = root / "engine" / "director" / "models.py"
        assert models_file.exists()
        assert validate_rationale_first_ast(models_file.read_text(encoding="utf-8")) is True

    def test_episodic_director_declares_deliberation_first_via_ast(self):
        root = Path(__file__).parent.parent
        episodic_file = root / "engine" / "director" / "episodic.py"
        assert episodic_file.exists()
        assert validate_rationale_first_ast(episodic_file.read_text(encoding="utf-8")) is True

    def test_spec_dict_starts_with_deliberation(self, rusher_profile):
        director = EpisodicDirector()
        spec = director.evaluate_and_update(rusher_profile)
        spec_dict = spec.to_dict()

        assert list(spec_dict.keys())[0] == "deliberation"
        assert validate_rationale_first_dict(spec_dict) is True
        assert len(spec.deliberation) > 40
        assert "kinetic velocity" in spec.deliberation.lower()

    def test_markdown_starts_with_deliberation_header(self, rusher_profile):
        director = EpisodicDirector()
        spec = director.evaluate_and_update(rusher_profile)
        md = spec.to_markdown()

        lines = [line.strip() for line in md.strip().split("\n") if line.strip()]
        assert lines[0].startswith("# Episodic Update Specification")
        assert lines[1] == "## Deliberation"


# ============================================================================
# 2. Bounded Adaptation Clamp (+/-15% Max) Invariant Tests
# ============================================================================

class TestBoundedAdaptationClamp:
    """Rigorous verification of the +/-15% adaptation clamp invariant."""

    def test_upward_adaptation_strictly_clamped_at_positive_15_percent(self):
        """Dominating player requesting +50% difficulty must be clamped to exactly +15%."""
        clamp = DifficultyClampState(max_adaptation_ratio=0.15)
        old_val = 100.0
        raw_val = 150.0  # +50% request

        clamped = clamp.apply_clamp("enemy_speed", old_val, raw_val)
        record = clamp.clamps["enemy_speed"]

        assert clamped == 115.0
        assert record.clamp_triggered is True
        assert record.delta_percentage == pytest.approx(15.0, abs=1e-3)
        assert record.min_bound == 85.0
        assert record.max_bound == 115.0

    def test_downward_adaptation_strictly_clamped_at_negative_15_percent(self):
        """Struggling player requesting -40% difficulty must be clamped to exactly -15%."""
        clamp = DifficultyClampState(max_adaptation_ratio=0.15)
        old_val = 200.0
        raw_val = 120.0  # -40% request

        clamped = clamp.apply_clamp("enemy_speed", old_val, raw_val)
        record = clamp.clamps["enemy_speed"]

        assert clamped == 170.0  # 200 * 0.85
        assert record.clamp_triggered is True
        assert record.delta_percentage == pytest.approx(-15.0, abs=1e-3)

    def test_in_bounds_adaptation_preserved_unclamped(self):
        """Adaptation of +8% within [-15%, +15%] must remain untouched."""
        clamp = DifficultyClampState(max_adaptation_ratio=0.15)
        old_val = 100.0
        raw_val = 108.0  # +8%

        clamped = clamp.apply_clamp("enemy_speed", old_val, raw_val)
        record = clamp.clamps["enemy_speed"]

        assert clamped == 108.0
        assert record.clamp_triggered is False
        assert record.delta_percentage == pytest.approx(8.0, abs=1e-3)

    def test_randomized_fuzz_clamp_invariant(self):
        """Fuzz testing: 100 randomized inputs must NEVER violate the [-15%, +15%] bound."""
        clamp = DifficultyClampState(max_adaptation_ratio=0.15)
        random.seed(42)

        for _ in range(100):
            old_val = random.uniform(10.0, 1000.0)
            raw_val = random.uniform(0.1, 3000.0)

            clamped = clamp.apply_clamp("param_fuzz", old_val, raw_val)
            delta_ratio = (clamped - old_val) / old_val

            assert delta_ratio >= -0.150001, f"Failed lower clamp: {delta_ratio} for old={old_val}, raw={raw_val}"
            assert delta_ratio <= +0.150001, f"Failed upper clamp: {delta_ratio} for old={old_val}, raw={raw_val}"


# ============================================================================
# 3. Multi-Episode Stability Tests
# ============================================================================

class TestMultiEpisodeProgressionStability:
    """Verifies that consecutive episodes scale smoothly without sudden jumps."""

    def test_10_episode_dominating_progression(self, rusher_profile):
        """Over 10 mastered episodes, difficulty scales strictly by (1.15)^n without runaway jumps."""
        director = EpisodicDirector(max_adaptation_ratio=0.15)
        current_cfg = CurrentEpisodeConfig(enemy_speed=100.0, enemy_damage=10.0)

        for ep in range(1, 11):
            spec = director.evaluate_and_update(rusher_profile, current_cfg)
            new_speed = spec.updated_episode_config.enemy_speed

            # Consecutive step check
            ratio = (new_speed - current_cfg.enemy_speed) / current_cfg.enemy_speed
            assert ratio <= 0.150001
            assert ratio >= 0.0

            current_cfg = spec.updated_episode_config

        expected_max = 100.0 * (1.15 ** 10)
        assert pytest.approx(current_cfg.enemy_speed, rel=1e-3) == expected_max

    def test_10_episode_struggling_reduction(self, struggling_bottleneck_profile):
        """Over 10 failed episodes, difficulty eases strictly by (0.85)^n smoothly."""
        director = EpisodicDirector(max_adaptation_ratio=0.15)
        current_cfg = CurrentEpisodeConfig(enemy_speed=200.0, enemy_damage=20.0)

        for ep in range(1, 11):
            spec = director.evaluate_and_update(struggling_bottleneck_profile, current_cfg)
            new_speed = spec.updated_episode_config.enemy_speed

            ratio = (new_speed - current_cfg.enemy_speed) / current_cfg.enemy_speed
            assert ratio >= -0.150001
            assert ratio <= 0.0

            current_cfg = spec.updated_episode_config

        expected_min = 200.0 * (0.85 ** 10)
        assert pytest.approx(current_cfg.enemy_speed, rel=1e-3) == expected_min


# ============================================================================
# 4. Spatial Directives & NavMesh Traversability Invariant
# ============================================================================

class TestSpatialDirectivesAndNavMeshVerification:
    """Verifies spatial obstacle placement and critical-path validation."""

    def test_death_cluster_triggers_cover_obstacle(self, struggling_bottleneck_profile):
        director = EpisodicDirector()
        spec = director.evaluate_and_update(struggling_bottleneck_profile)

        assert spec.bottleneck_detected is True
        assert len(spec.spatial_directives) >= 1

        dir1 = spec.spatial_directives[0]
        assert dir1.action == DirectiveAction.ADD_OBSTACLE
        assert dir1.navmesh_verified is True
        assert dir1.death_count_addressed == 4

    def test_spatial_directives_never_block_critical_path(self, struggling_bottleneck_profile):
        """Asserts that modified walls in updated config still yield a traversable A* path."""
        from engine.spatial.navmesh import astar_pathfind
        director = EpisodicDirector()
        spec = director.evaluate_and_update(struggling_bottleneck_profile)

        walls = spec.updated_episode_config.walls
        grid = [[0 for _ in range(40)] for _ in range(22)]
        for wx, wy, ww, wh in walls:
            c1, c2 = max(0, int(wx // 32)), min(39, int((wx + ww) // 32))
            r1, r2 = max(0, int(wy // 32)), min(21, int((wy + wh) // 32))
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    grid[r][c] = 1

        # Spawn (120, 360) -> (3, 11); Exit (1160, 360) -> (36, 11)
        grid[11][3] = 0
        grid[11][36] = 0
        path = astar_pathfind(grid, (3, 11), (36, 11), allow_diagonal=True)
        assert path is not None, "Critical path was blocked by spatial directive!"


# ============================================================================
# 5. Artifact Envelope Sealing & File Output Tests
# ============================================================================

class TestArtifactEnvelopeSealing:
    """Verifies cryptographic envelope sealing, provenance tracking, and file writing."""

    def test_seal_episodic_update_artifact(self, rusher_profile):
        director = EpisodicDirector()
        spec = director.evaluate_and_update(rusher_profile)

        envelope = director.seal_update_artifact(spec, run_id="run_test_01")
        assert envelope.artifact_type == ArtifactType.UPDATE_SPEC
        assert envelope.verify_integrity(raise_on_error=True) is True
        assert envelope.payload["target_episode_id"] == "level_02"
        assert list(envelope.payload.keys())[0] == "deliberation"
        assert len(envelope.sha256_hash) == 64

    def test_write_update_spec_to_disk(self, rusher_profile, tmp_path):
        specs_dir = tmp_path / "specs"
        director = EpisodicDirector(specs_dir=specs_dir)
        spec = director.evaluate_and_update(rusher_profile)

        md_path = director.write_update_spec(spec)
        assert md_path.exists()
        assert md_path.name == "update_spec.md"

        content = md_path.read_text(encoding="utf-8")
        assert content.startswith("# Episodic Update Specification")
        assert "## Deliberation" in content

        json_path = md_path.with_suffix(".json")
        assert json_path.exists()
        assert "target_episode_id" in json_path.read_text(encoding="utf-8")
