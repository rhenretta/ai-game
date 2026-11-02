"""tests/test_m1_challenger_2_decoupling.py
Empirical challenge suite for Milestone 1:
1. PlayingScene decoupling with cartridge = None (60 ticks update/render, no crashes, no hardcoded entities, safe portal=None).
2. PlayingScene with minimal cartridge (empty walls, enemies, hazards) checking victory and defeat triggers.
3. GenericCartridge in engine/loader.py: verify it does not import or instantiate ChaserEnemy, RangedEnemy, or SpikeHazard.
4. run_game.sh execution when cartridges/active is absent: verify exit code 0 and guidance banner.
"""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional
import pytest
import pygame

import config
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.core.engine import Engine
from engine.core.scenes.game_over_scene import GameOverScene
from engine.core.scenes.playing_scene import PlayingScene
from engine.core.scenes.victory_scene import EpisodeVictoryScene
from engine.loader import GenericCartridge
from engine.telemetry.recorder import TelemetryRecorder


class MinimalEmptyCartridge(BaseCartridge):
    """Minimal open-genre cartridge with empty entities."""

    def __init__(self, directory: Path, exit_point: Optional[tuple[float, float]] = None):
        super().__init__(cartridge_dir=directory)
        self._exit_point = exit_point

    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            level_id=f"level_{episode_number:02d}",
            name=f"Minimal Level {episode_number}",
            spawn_point=(120.0, 360.0),
            exit_point=self._exit_point,
            walls=[],
            enemies=[],
            hazards=[],
        )

    def create_enemies(self, spec: Optional[BaseLevelSpec] = None) -> List[Any]:
        return []

    def create_hazards(self, spec: Optional[BaseLevelSpec] = None) -> List[Any]:
        return []


@pytest.fixture
def headless_engine(tmp_path: Path):
    """Initializes a headless Engine with a temporary SQLite telemetry recorder."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    db_path = tmp_path / "telemetry_test.db"
    recorder = TelemetryRecorder(db_path=db_path, batch_size=10, flush_interval_sec=0.05)
    recorder.start_session(player_profile_tag="challenger_test")

    engine = Engine(headless=True)
    engine.set_telemetry_recorder(recorder)
    engine.active_cartridge = None

    # Register standard scenes
    engine.scene_manager.register_scene("playing", PlayingScene(engine))
    engine.scene_manager.register_scene("game_over", GameOverScene(engine))
    engine.scene_manager.register_scene("victory", EpisodeVictoryScene(engine))

    yield engine

    if recorder:
        try:
            recorder.close()
        except Exception:
            pass


class TestPlayingSceneDecoupling:
    """Empirical challenge for PlayingScene decoupling and safety."""

    def test_playing_scene_cartridge_none_60_ticks(self, headless_engine: Engine):
        """
        Challenge 1.1:
        Initialize PlayingScene with cartridge = None, enter scene,
        run 60 ticks of update() and render().
        Verify:
        - Does NOT crash.
        - Does NOT spawn hardcoded walls (walls == []).
        - Does NOT spawn enemies (enemies == []).
        - Does NOT spawn hazards (hazards == []).
        - self.portal is None.
        - update() and render() safely tolerate self.portal is None.
        """
        headless_engine.active_cartridge = None
        scene = PlayingScene(headless_engine)
        headless_engine.scene_manager.register_scene("playing_test", scene)
        headless_engine.scene_manager.switch_to("playing_test")

        # Invariant checks immediately upon enter
        assert scene.walls == [], f"Expected empty walls, found {len(scene.walls)}"
        assert scene.enemies == [], f"Expected empty enemies, found {len(scene.enemies)}"
        assert scene.hazards == [], f"Expected empty hazards, found {len(scene.hazards)}"
        assert scene.portal is None, f"Expected portal to be None, found {scene.portal}"

        # Surface for rendering
        surface = headless_engine.logical_surface

        # Run 60 ticks (1 second of simulated 60 FPS gameplay)
        dt = 1.0 / 60.0
        for tick in range(60):
            scene.update(dt)
            scene.render(surface)

        # Invariant checks after 60 ticks
        assert scene.walls == [], f"Walls spawned during ticks: {scene.walls}"
        assert scene.enemies == [], f"Enemies spawned during ticks: {scene.enemies}"
        assert scene.hazards == [], f"Hazards spawned during ticks: {scene.hazards}"
        assert scene.portal is None, f"Portal unexpectedly materialized: {scene.portal}"
        assert scene.stats["enemies_killed"] == 0
        assert scene.stats["damage_taken"] == 0.0

    def test_playing_scene_portal_none_without_on_enter(self, headless_engine: Engine):
        """
        Challenge 1.2:
        Test raw PlayingScene instance where portal is explicitly None
        to ensure update() and render() never assume portal is non-null.
        """
        scene = PlayingScene(headless_engine)
        scene.portal = None
        scene.walls = []
        scene.enemies = []
        scene.hazards = []

        surface = headless_engine.logical_surface
        for _ in range(10):
            scene.update(1.0 / 60.0)
            scene.render(surface)

        assert scene.portal is None

    def test_minimal_cartridge_victory_and_defeat_triggers(self, headless_engine: Engine, tmp_path: Path):
        """
        Challenge 1.3:
        Initialize PlayingScene with a minimal cartridge that has empty walls and empty enemies.
        Verify victory/defeat triggers do not raise exceptions and transition scene state cleanly.
        """
        cart = MinimalEmptyCartridge(directory=tmp_path, exit_point=None)
        headless_engine.active_cartridge = cart

        # Case A: Defeat trigger
        scene_defeat = PlayingScene(headless_engine)
        headless_engine.scene_manager.register_scene("scene_defeat", scene_defeat)
        headless_engine.scene_manager.switch_to("scene_defeat")

        assert scene_defeat.walls == []
        assert scene_defeat.enemies == []
        assert scene_defeat.hazards == []
        assert scene_defeat.portal is None

        # Trigger defeat
        scene_defeat.trigger_defeat()
        assert headless_engine.scene_manager.current_scene_name == "game_over"

        # Case B: Victory trigger without portal (exit_point is None)
        scene_victory = PlayingScene(headless_engine)
        headless_engine.scene_manager.register_scene("scene_victory", scene_victory)
        headless_engine.scene_manager.switch_to("scene_victory")
        assert scene_victory.portal is None

        scene_victory.trigger_victory()
        assert headless_engine.scene_manager.current_scene_name == "victory"

        # Case C: Victory trigger with portal (exit_point defined)
        cart_with_portal = MinimalEmptyCartridge(directory=tmp_path, exit_point=(500.0, 300.0))
        headless_engine.active_cartridge = cart_with_portal

        scene_portal = PlayingScene(headless_engine)
        headless_engine.scene_manager.register_scene("scene_portal", scene_portal)
        headless_engine.scene_manager.switch_to("scene_portal")
        assert scene_portal.portal is not None
        assert scene_portal.portal.circle.x == 500.0
        assert scene_portal.portal.circle.y == 300.0

        scene_portal.trigger_victory()
        assert headless_engine.scene_manager.current_scene_name == "victory"


class TestGenericCartridgeDecoupling:
    """Empirical challenge for GenericCartridge in engine/loader.py."""

    def test_generic_cartridge_does_not_import_or_instantiate_arena_entities(self, tmp_path: Path):
        """
        Challenge 2.1:
        Verify that GenericCartridge does not import or instantiate:
        - ChaserEnemy
        - RangedEnemy
        - SpikeHazard
        - FloorAcidHazard
        - LaserSweepHazard
        """
        import engine.loader as loader_mod

        # Module attribute inspection
        forbidden_symbols = [
            "ChaserEnemy",
            "RangedEnemy",
            "SpikeHazard",
            "FloorAcidHazard",
            "LaserSweepHazard",
        ]
        for sym in forbidden_symbols:
            assert sym not in loader_mod.__dict__, f"loader.py unexpectedly exports or imports {sym}"

        # Instantiation inspection
        dummy_dir = tmp_path / "generic_cart"
        dummy_dir.mkdir()
        cartridge = GenericCartridge(directory=dummy_dir)

        # Entity factories must return empty lists by default
        enemies = cartridge.create_enemies()
        assert enemies == [], f"Expected empty enemies, got {enemies}"

        hazards = cartridge.create_hazards()
        assert hazards == [], f"Expected empty hazards, got {hazards}"

        spec = cartridge.get_level_spec(1)
        assert spec.walls == [], f"Expected empty walls in default spec, got {spec.walls}"
        assert spec.enemies == [], f"Expected empty enemies in default spec, got {spec.enemies}"
        assert spec.hazards == [], f"Expected empty hazards in default spec, got {spec.hazards}"


class TestRunGameShellScript:
    """Empirical challenge for run_game.sh behavior when cartridges/active is absent."""

    def test_run_game_sh_when_active_cartridge_absent(self, stash_active_cartridge):
        """
        Challenge 3.1:
        Execute run_game.sh when cartridges/active is absent.
        Verify:
        - Exit code is 0 (clean shutdown).
        - Emits guidance banner instructing user to run run_studio.py.
        - Does NOT fall back to seed or raise uncaught exceptions.
        """
        active_cartridge_dir = Path(stash_active_cartridge)
        assert not active_cartridge_dir.exists(), "cartridges/active should not exist before Studio builds a game"

        # Run bash run_game.sh --headless
        cmd = ["bash", "run_game.sh", "--headless"]
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )

        assert res.returncode == 0, f"run_game.sh failed with code {res.returncode}. Stderr: {res.stderr}"
        stdout = res.stdout

        # Verify guidance banner
        assert "No Active Game Cartridge Found in cartridges/active/" in stdout
        assert "python run_studio.py" in stdout
        assert "Autonomous Game Engine & Showrunner Harness" in stdout

        # Verify clean telemetry shutdown
        assert "Shutdown completed cleanly" in stdout or "Game and telemetry shutdown completed cleanly." in stdout

        # Verify zero fallback to seed
        assert "cartridges/seed" not in stdout.lower()


class TestAdversarialDecouplingEdgeCases:
    """Stress testing edge cases, boundary conditions, and corruptions."""

    def test_playing_scene_low_health_trigger_without_barks(self, headless_engine: Engine):
        """Verify update() does not crash when low_health trigger fires with 0 barks."""
        headless_engine.active_cartridge = None
        scene = PlayingScene(headless_engine)
        headless_engine.scene_manager.register_scene("low_health_test", scene)
        headless_engine.scene_manager.switch_to("low_health_test")

        # Manually lower player health to trigger reactive bark
        scene.player.health = 25.0
        assert scene.dialogue_runtime.barks == []

        for _ in range(10):
            scene.update(1.0 / 60.0)

        # Player remains alive at low health without crash
        assert scene.player.health == 25.0
        assert scene.player.is_alive

    def test_playing_scene_defeat_on_player_death_without_cartridge(self, headless_engine: Engine):
        """Verify update() transitions to game_over scene when player health reaches 0 with cartridge=None."""
        headless_engine.active_cartridge = None
        scene = PlayingScene(headless_engine)
        headless_engine.scene_manager.register_scene("death_test", scene)
        headless_engine.scene_manager.switch_to("death_test")

        from engine.combat.damage import DamagePacket, Faction
        # Apply fatal damage to player via DamagePacket
        packet = DamagePacket(
            amount=150.0,
            source_faction=Faction.ENEMY,
            iframes_granted=0.0,
            source_id="test_hazard",
        )
        scene.player.take_damage(packet)
        assert not scene.player.is_alive
        assert scene.player.health == 0.0

        # Update should catch death and transition to game_over
        scene.update(1.0 / 60.0)
        assert headless_engine.scene_manager.current_scene_name == "game_over"

    def test_multi_episode_switch_without_cartridge(self, headless_engine: Engine):
        """Verify switching across multiple episodes with cartridge=None cleanly resets state without leaks."""
        headless_engine.active_cartridge = None
        scene = PlayingScene(headless_engine)
        headless_engine.scene_manager.register_scene("multi_ep_test", scene)

        for ep in range(1, 5):
            headless_engine.scene_manager.switch_to("multi_ep_test", episode_number=ep)
            assert scene.episode_number == ep
            assert scene.walls == []
            assert scene.enemies == []
            assert scene.hazards == []
            assert scene.portal is None

            for _ in range(15):
                scene.update(1.0 / 60.0)
                scene.render(headless_engine.logical_surface)

    def test_open_genre_cartridge_custom_scene_routing(self, headless_engine: Engine, tmp_path: Path):
        """Verify that an open-genre cartridge can mount custom non-arena scenes and set initial scene."""
        from engine.core.scene import Scene

        class TacticalDeckScene(Scene):
            def __init__(self, engine: Any):
                super().__init__(engine)
                self.cards_played = 0

            def handle_event(self, event: Any) -> None:
                pass

            def on_enter(self, context: Dict[str, Any]) -> None:
                pass

            def on_exit(self) -> Dict[str, Any]:
                return {}

            def update(self, dt: float) -> None:
                self.cards_played += 1

            def render(self, surface: Any) -> None:
                pass

        class TacticalDeckCartridge(BaseCartridge):
            def __init__(self, directory: Path):
                super().__init__(cartridge_dir=directory)
                self.custom_scene = TacticalDeckScene(headless_engine)

            def get_scenes(self, engine: Any) -> Dict[str, Any]:
                return {"TacticalDeckScene": self.custom_scene}

            def get_initial_scene(self, engine: Any) -> Optional[Any]:
                return self.custom_scene

        cart = TacticalDeckCartridge(tmp_path)
        headless_engine.active_cartridge = cart

        # Emulate launcher scene registration
        custom_scenes = cart.get_scenes(headless_engine)
        for s_name, s_inst in custom_scenes.items():
            headless_engine.scene_manager.register_scene(s_name, s_inst)

        init_scene = cart.get_initial_scene(headless_engine)
        assert init_scene is not None
        headless_engine.scene_manager.register_scene("cartridge_initial", init_scene)
        headless_engine.scene_manager.switch_to("cartridge_initial")

        assert headless_engine.scene_manager.current_scene_name == "cartridge_initial"
        current_scene = headless_engine.scene_manager.current_scene
        assert isinstance(current_scene, TacticalDeckScene)

        # Tick custom scene
        for _ in range(10):
            current_scene.update(1.0 / 60.0)
            current_scene.render(headless_engine.logical_surface)

        assert current_scene.cards_played == 10

    def test_resolve_cartridge_corrupted_active_directory_behavior(self, tmp_path: Path, monkeypatch, capsys):
        """Verify resolve_and_load_cartridge handles corrupted active directory safely with banner and returns None."""
        from run_game import resolve_and_load_cartridge, NO_ACTIVE_GAME_BANNER
        engine = Engine(headless=True)

        # Hermetic test isolation: create corrupted cartridge in an isolated temp directory
        fake_root = tmp_path / "corrupted_workspace"
        active_dir = fake_root / "cartridges" / "active"
        active_dir.mkdir(parents=True, exist_ok=True)
        corrupted_manifest = active_dir / "manifest.json"
        corrupted_manifest.write_text("CORRUPTED JSON {{{{", encoding="utf-8")

        monkeypatch.setattr(config, "PROJECT_ROOT", fake_root)

        booted = resolve_and_load_cartridge(engine)
        assert booted is None, f"Expected None on corrupted active cartridge, got {booted}"
        out, _ = capsys.readouterr()
        assert "No Active Game Cartridge Found in cartridges/active/" in out

