"""tests/test_director_genesis.py
Comprehensive unit test suite for Rationale-First Steam Genesis Spec Generator.
Verifies:
  1. AST Rationale-First compliance for all models and classes.
  2. Dict serialization key order (deliberation strictly first).
  3. Markdown rendering structure (## Deliberation strictly first).
  4. Behavioral and parameter divergence between synthetic Rusher and Tactician DNA fixtures.
  5. Default / empty DNA graceful fallback.
  6. Disk persistence of genesis_spec.md and companion JSON.
  7. SQLite WAL audit persistence.
"""
import json
from pathlib import Path
import pytest

from engine.director.genesis import GenesisDirector
from engine.director.models import (
    EnemyArchetypeSpec,
    GenesisSpec,
    HazardArchetypeSpec,
    LevelDimensionsSpec,
    PlayerParametersSpec,
    WeaponArchetypeSpec,
)
from engine.steam.models import SteamPlayerDNA
from tests.test_tier1_units import validate_rationale_first_ast


@pytest.fixture
def rusher_dna() -> SteamPlayerDNA:
    """Synthetic DNA fixture for an aggressive high-mobility rusher."""
    return SteamPlayerDNA(
        deliberation="Simulated Rusher: High playtime in Hades, Dead Cells, Celeste. High kinetic pacing.",
        player_id="steam_rusher_001",
        profile_tag="genesis_rusher",
        total_playtime_hours=320.0,
        total_games_analyzed=5,
        genre_weights={
            "Action": 0.45,
            "Roguelike": 0.30,
            "PrecisionPlatformer": 0.15,
            "BulletHell": 0.10,
            "Strategy": 0.0,
            "RPG": 0.0,
            "Shooter": 0.0,
            "Simulation": 0.0,
            "Puzzle": 0.0,
            "Adventure": 0.0,
        },
        pacing_preference="kinetic_high_mobility",
        pacing_score=0.65,
        challenge_tolerance=0.78,
        mechanical_biases={
            "dodge_roll": 0.85,
            "ranged_aim": 0.30,
            "close_combat": 0.70,
            "speedrun": 0.80,
        },
        preferred_palette_mood="neon_cyberpunk",
        palette_weights={
            "neon_cyberpunk": 0.60,
            "grim_dark": 0.20,
            "retro_arcade": 0.15,
            "vibrant_fantasy": 0.05,
        },
        top_genres=["Action", "Roguelike", "PrecisionPlatformer"],
    )


@pytest.fixture
def tactician_dna() -> SteamPlayerDNA:
    """Synthetic DNA fixture for a methodical tactical / strategic player."""
    return SteamPlayerDNA(
        deliberation="Simulated Tactician: High playtime in Civ VI, Slay the Spire, Into the Breach. Negative pacing.",
        player_id="steam_tactician_002",
        profile_tag="genesis_tactician",
        total_playtime_hours=410.0,
        total_games_analyzed=4,
        genre_weights={
            "Strategy": 0.60,
            "Puzzle": 0.25,
            "Simulation": 0.10,
            "RPG": 0.05,
            "Action": 0.0,
            "Roguelike": 0.0,
            "BulletHell": 0.0,
            "PrecisionPlatformer": 0.0,
            "Shooter": 0.0,
            "Adventure": 0.0,
        },
        pacing_preference="tactical_methodical",
        pacing_score=-0.55,
        challenge_tolerance=0.45,
        mechanical_biases={
            "dodge_roll": 0.25,
            "ranged_aim": 0.70,
            "close_combat": 0.20,
            "speedrun": 0.15,
        },
        preferred_palette_mood="grim_dark",
        palette_weights={
            "grim_dark": 0.55,
            "neon_cyberpunk": 0.10,
            "retro_arcade": 0.15,
            "vibrant_fantasy": 0.20,
        },
        top_genres=["Strategy", "Puzzle", "Simulation"],
    )


@pytest.fixture
def default_dna() -> SteamPlayerDNA:
    """Baseline fallback DNA."""
    return SteamPlayerDNA(
        deliberation="Simulated Default: Baseline uniform distribution.",
        player_id="steam_default_000",
        profile_tag="genesis_default",
        total_playtime_hours=0.0,
        total_games_analyzed=0,
        pacing_preference="balanced",
        pacing_score=0.0,
        challenge_tolerance=0.50,
        mechanical_biases={
            "dodge_roll": 0.5,
            "ranged_aim": 0.5,
            "close_combat": 0.5,
            "speedrun": 0.5,
        },
        preferred_palette_mood="retro_arcade",
        top_genres=["Action", "Adventure"],
    )


class TestGenesisDirectorASTCompliance:
    """Verifies that all classes declare deliberation as their very first attribute."""

    def test_models_ast_compliance(self):
        root = Path(__file__).parent.parent
        models_file = root / "engine" / "director" / "models.py"
        assert models_file.exists(), "engine/director/models.py must exist"
        assert validate_rationale_first_ast(models_file.read_text(encoding="utf-8")) is True

    def test_genesis_engine_ast_compliance(self):
        root = Path(__file__).parent.parent
        genesis_file = root / "engine" / "director" / "genesis.py"
        assert genesis_file.exists(), "engine/director/genesis.py must exist"
        assert validate_rationale_first_ast(genesis_file.read_text(encoding="utf-8")) is True


class TestGenesisDirectorSerialization:
    """Verifies that dictionary outputs declare deliberation first."""

    def test_genesis_spec_dict_keys_order(self, rusher_dna):
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)

        d = spec.to_dict()
        assert list(d.keys())[0] == "deliberation"
        assert list(spec.model_dump().keys())[0] == "deliberation"
        assert list(spec.player_params.to_dict().keys())[0] == "deliberation"
        assert list(spec.weapon_spec.to_dict().keys())[0] == "deliberation"
        assert list(spec.level_dimensions.to_dict().keys())[0] == "deliberation"
        if spec.enemy_specs:
            assert list(spec.enemy_specs[0].to_dict().keys())[0] == "deliberation"


class TestGenesisMarkdownRendering:
    """Verifies that genesis_spec.md starts with ## Deliberation."""

    def test_markdown_deliberation_header_order(self, rusher_dna):
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        md = director.render_genesis_markdown(spec)

        lines = [line.strip() for line in md.splitlines() if line.strip().startswith("## ")]
        assert len(lines) >= 5
        assert lines[0] == "## Deliberation"
        assert "## Genesis Metadata" in lines
        assert "## Player Dynamics & Physics" in lines
        assert "## Starter Weapon Archetype" in lines
        assert "## Enemy Archetypes & Spawns" in lines

    def test_markdown_deliberation_content_depth(self, rusher_dna):
        director = GenesisDirector()
        spec = director.generate_genesis_spec(rusher_dna)
        md = director.render_genesis_markdown(spec)

        delib = spec.deliberation
        assert len(delib.split(".")) >= 4
        assert "320.0 hours" in delib
        assert "Action" in delib


class TestGenesisParameterDivergence:
    """Verifies distinct gameplay mechanics for Rusher vs Tactician profiles."""

    def test_rusher_vs_tactician_divergence(self, rusher_dna, tactician_dna):
        director = GenesisDirector()
        rusher_spec = director.generate_genesis_spec(rusher_dna)
        tactician_spec = director.generate_genesis_spec(tactician_dna)

        # 1. Player Speed divergence: Rusher must be significantly faster
        assert rusher_spec.player_params.walk_speed > tactician_spec.player_params.walk_speed + 50.0
        assert rusher_spec.player_params.walk_speed >= 300.0
        assert tactician_spec.player_params.walk_speed <= 250.0

        # 2. Dash Cooldown divergence: Rusher must dash more frequently
        assert rusher_spec.player_params.dash_cooldown < tactician_spec.player_params.dash_cooldown - 0.20
        assert rusher_spec.player_params.dash_cooldown <= 0.45
        assert tactician_spec.player_params.dash_cooldown >= 0.70

        # 3. Dash Speed divergence
        assert rusher_spec.player_params.dash_speed > tactician_spec.player_params.dash_speed + 150.0

        # 4. Weapon Archetype divergence
        assert rusher_spec.weapon_spec.archetype == "spread"
        assert rusher_spec.weapon_spec.spread_count == 3
        assert tactician_spec.weapon_spec.archetype == "sniper"
        assert tactician_spec.weapon_spec.spread_count == 1
        assert tactician_spec.weapon_spec.projectile_speed > rusher_spec.weapon_spec.projectile_speed

        # 5. Enemy Archetype mix divergence
        rusher_chasers = next(e for e in rusher_spec.enemy_specs if e.archetype == "chaser")
        tactician_chasers = next(e for e in tactician_spec.enemy_specs if e.archetype == "chaser")
        assert rusher_chasers.count >= 3
        assert rusher_chasers.speed > tactician_chasers.speed + 20.0

        # 6. Palette mood divergence
        assert rusher_spec.palette_mood == "neon_cyberpunk"
        assert tactician_spec.palette_mood == "grim_dark"

        # 7. Level Dimensions divergence
        assert rusher_spec.level_dimensions.arena_layout_type == "wide_expanse"
        assert rusher_spec.level_dimensions.width >= 1400.0
        assert tactician_spec.level_dimensions.arena_layout_type == "tactical_corridors"
        assert tactician_spec.level_dimensions.pillar_count > rusher_spec.level_dimensions.pillar_count

    def test_default_dna_baseline(self, default_dna):
        director = GenesisDirector()
        spec = director.generate_genesis_spec(default_dna)

        assert spec.player_params.walk_speed == pytest.approx(260.0, abs=5.0)
        assert spec.player_params.dash_cooldown == pytest.approx(0.60, abs=0.05)
        assert spec.weapon_spec.archetype == "blaster"
        assert spec.palette_mood == "retro_arcade"
        assert spec.level_dimensions.width == 1280.0
        assert spec.level_dimensions.height == 720.0


class TestGenesisFilePersistence:
    """Verifies writing genesis_spec.md and JSON to disk and SQLite."""

    def test_write_genesis_spec_to_disk(self, rusher_dna, tmp_path):
        specs_dir = tmp_path / "specs"
        director = GenesisDirector(specs_dir=specs_dir)
        spec = director.generate_genesis_spec(rusher_dna)

        md_path = director.write_genesis_spec(spec)
        assert md_path.exists()
        assert md_path.name == "genesis_spec.md"

        content = md_path.read_text(encoding="utf-8")
        assert content.startswith("# Genesis Specification: Episode 0")
        assert "## Deliberation" in content

        json_path = md_path.with_suffix(".json")
        assert json_path.exists()
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        assert parsed["spec_id"] == spec.spec_id
        assert list(parsed.keys())[0] == "deliberation"

    def test_save_spec_to_sqlite(self, rusher_dna, tmp_path):
        db_path = tmp_path / "test_telemetry.db"
        director = GenesisDirector(db_path=db_path)
        spec = director.generate_genesis_spec(rusher_dna)

        spec_id = director.save_spec_to_db(spec)
        assert spec_id == spec.spec_id
