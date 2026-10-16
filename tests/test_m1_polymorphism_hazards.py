"""tests/test_m1_polymorphism_hazards.py
Regression tests for:
1. Dynamic enemy polymorphism in PlayingScene (Sentinel Mandate).
2. FloorAcidHazard movement slowdown verification.
3. Fallback wall geometry rendering without crash.
"""
from typing import Optional, Any
import pytest
from engine.api.contracts import BaseCartridge, BaseLevelSpec, BaseEnemy
from engine.core.engine import Engine
from engine.core.scenes.playing_scene import PlayingScene
from engine.entities.hazard import FloorAcidHazard
from engine.entities.player import Player
from engine.spatial.aabb import Vector2, AABB


class CustomCartridgeEnemy(BaseEnemy):
    """Dynamically authored enemy class simulating SWE compiler output."""
    def __init__(self, eid: str, x: float, y: float):
        super().__init__(eid, x, y, 32.0, 32.0, max_health=50.0, speed=100.0)
        self.ai_updated = False
        self.received_projectile_mgr = False

    def update_ai(self, dt: float, player: Any, walls: Any, projectile_manager: Optional[Any] = None, telemetry_recorder: Optional[Any] = None) -> None:
        self.ai_updated = True
        if projectile_manager is not None:
            self.received_projectile_mgr = True


class CustomEnemyCartridge(BaseCartridge):
    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            level_id="lvl_poly",
            name="Polymorphism Test",
            enemies=[{"id": "custom_1", "type": "custom", "x": 500.0, "y": 300.0}]
        )

    def create_player(self, x: float, y: float, telemetry_recorder: Optional[Any] = None) -> Player:
        return Player(x, y, telemetry_recorder)

    def create_enemies(self, spec: BaseLevelSpec):
        return [CustomCartridgeEnemy("custom_1", 500.0, 300.0)]

    def create_hazards(self, spec: BaseLevelSpec):
        return []


def test_custom_enemy_polymorphic_dispatch_in_playing_scene():
    """Verifies that custom enemy types extending BaseEnemy have their update_ai invoked."""
    engine = Engine(headless=True)
    engine.load_cartridge(CustomEnemyCartridge())
    scene = PlayingScene(engine)
    engine.scene_manager.register_scene("playing", scene)
    engine.scene_manager.switch_to("playing")

    enemy = scene.enemies[0]
    assert isinstance(enemy, CustomCartridgeEnemy)
    assert enemy.ai_updated is False

    engine.game_loop.step()

    assert enemy.ai_updated is True, "Custom enemy AI must be invoked by PlayingScene.update()"
    assert enemy.received_projectile_mgr is True, "Custom enemy should receive projectile_manager"


def test_floor_acid_hazard_genuinely_slows_player_movement():
    """Verifies that standing in FloorAcidHazard reduces player displacement."""
    engine = Engine(headless=True)
    scene = PlayingScene(engine)
    engine.scene_manager.register_scene("playing", scene)
    engine.scene_manager.switch_to("playing")

    player = scene.player
    acid = FloorAcidHazard("acid_01", 100.0, 100.0, 96.0, 96.0)
    scene.hazards.append(acid)

    # Place player inside acid
    player.pos = Vector2(acid.pos.x + 10.0, acid.pos.y + 10.0)
    engine.input_manager.get_movement_vector = lambda: (1.0, 0.0)

    initial_x = player.pos.x
    scene.update(0.1)
    displacement = player.pos.x - initial_x

    expected_normal = player.walk_speed * 0.1
    # Slowdown should reduce displacement substantially (< 80% of normal)
    assert displacement < expected_normal * 0.8, (
        f"Acid failed to slow player: displacement={displacement:.2f}, normal={expected_normal:.2f}"
    )


def test_fallback_geometry_rendering_no_cartridge():
    """Verifies that PlayingScene renders without crash when no cartridge is loaded."""
    engine = Engine(headless=True)
    scene = PlayingScene(engine)
    engine.scene_manager.register_scene("playing", scene)
    engine.scene_manager.switch_to("playing")

    # Step engine (includes update and render_and_present)
    engine.game_loop.step()
    for wall in scene.walls:
        assert isinstance(wall, AABB), f"Wall geometry must be AABB: {wall}"
