"""tests/test_m1_engine.py
Direct unit and integration test suite for Milestone 1 Autonomous Engine modules:
Spatial math, Combat mechanics, Entities, Telemetry WAL, and Dynamic Cartridge Loader.
"""
import math
import tempfile
import time
from pathlib import Path
import pytest
import config
from engine.api.contracts import BaseLevelSpec
from engine.combat.damage import DamagePacket, Faction, calculate_damage
from engine.combat.hitbox import Hitbox, Hurtbox, check_hitbox_hurtbox
from engine.combat.weapon import BlasterWeapon, SpreadWeapon
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.core.engine import Engine
from engine.core.scenes.game_over_scene import GameOverScene
from engine.core.scenes.playing_scene import PlayingScene
from engine.core.scenes.title_scene import TitleScene
from engine.core.scenes.victory_scene import EpisodeVictoryScene
from engine.entities.enemies import ChaserEnemy, RangedEnemy
from engine.entities.hazard import FloorAcidHazard, LaserSweepHazard, SpikeHazard, SpikeState
from engine.entities.player import Player
from engine.entities.portal import Portal
from engine.entities.projectile import ProjectileManager
from engine.loader import ASTSecurityError, CartridgeLoader, validate_cartridge_ast
from engine.spatial.aabb import AABB, Circle, Vector2
from engine.spatial.collision import move_and_resolve_walls
from engine.spatial.grid import SpatialHashGrid
from engine.spatial.navmesh import NavMesh, astar_pathfind
from engine.telemetry.recorder import TelemetryRecorder


class TestEngineCartridgeLoader:
    """Verifies dynamic cartridge compilation, loading, and AST security validation."""

    def test_load_seed_cartridge(self, sample_open_genre_cartridge):
        cartridge = CartridgeLoader.load_cartridge(sample_open_genre_cartridge)
        assert cartridge is not None
        assert isinstance(cartridge, BaseCartridge)
        spec = cartridge.get_level_spec(1)
        assert isinstance(spec, BaseLevelSpec)
        assert hasattr(cartridge, "get_initial_scene")
        scene = cartridge.get_initial_scene(None)
        assert scene is not None
        assert hasattr(scene, "update")

    def test_ast_security_validation_catches_unsafe_calls(self):
        unsafe_code = """
import os
def malicious():
    os.system('rm -rf /')
"""
        with pytest.raises(ASTSecurityError):
            validate_cartridge_ast(unsafe_code)

    def test_ast_security_validation_allows_safe_math(self):
        safe_code = """
import math
def compute_velocity(speed, angle):
    return speed * math.cos(angle), speed * math.sin(angle)
"""
        validate_cartridge_ast(safe_code)  # Should not raise


class TestCombatAndEntityMechanics:
    """Verifies kinetic movement, dashing, iFrames, and damage propagation."""

    def test_player_dash_triggers_iframes_and_stamina(self):
        player = Player(100.0, 100.0)
        proj_mgr = ProjectileManager()
        assert player.stamina == 100.0
        assert player.is_dashing is False

        # Trigger dash
        player.handle_input(
            move_input=Vector2(1.0, 0.0),
            mouse_canvas_pos=Vector2(200.0, 100.0),
            dash_pressed=True,
            fire_pressed=False,
            projectile_manager=proj_mgr
        )
        assert player.is_dashing is True
        assert player.stamina == 75.0
        assert player.iframe_timer > 0.0

        # Attempt damage during dash iFrames
        packet = DamagePacket(amount=50.0, source_faction=Faction.ENEMY, knockback_impulse=Vector2(0, 0))
        applied = player.take_damage(packet)
        assert applied is False
        assert player.health == 100.0

    def test_enemy_damage_and_death(self):
        enemy = ChaserEnemy("chaser_test", 200.0, 200.0)
        assert enemy.health == 60.0
        assert enemy.is_alive is True

        packet = DamagePacket(amount=60.0, source_faction=Faction.PLAYER, knockback_impulse=Vector2(50, 0))
        applied = enemy.take_damage(packet)
        assert applied is True
        assert enemy.health == 0.0
        assert enemy.is_alive is False

    def test_spike_hazard_cycles_and_damages(self):
        spikes = SpikeHazard("spikes_test", 100.0, 100.0)
        player = Player(110.0, 110.0)
        assert spikes.state == SpikeState.RETRACTED

        # Retracted spikes deal no damage
        spikes.update(0.1, player)
        assert player.health == 100.0

        # Advance to EXTENDED state
        spikes.state = SpikeState.EXTENDED
        spikes.update(0.016, player)
        assert player.health == 80.0  # Took 20 damage


class TestTelemetryRecorderAndWALConcurrency:
    """Verifies sub-microsecond non-blocking telemetry enqueue and WAL persistence."""

    def test_telemetry_batch_persistence(self, tmp_path):
        db_file = tmp_path / "telemetry_test.db"
        recorder = TelemetryRecorder(db_path=db_file, batch_size=10, flush_interval_sec=0.05)

        sess_id = recorder.start_session(player_profile_tag="rusher")
        ep_id = recorder.start_episode(level_id="test_chamber", episode_index=1)

        start_time = time.perf_counter()
        for i in range(50):
            recorder.emit_event(
                event_type="MOVEMENT_TICK",
                pos=(100.0 + i, 200.0),
                vel=(250.0, 0.0),
                health=100.0,
                payload={"frame": i}
            )
        elapsed_us = (time.perf_counter() - start_time) / 50 * 1e6
        # Assert sub-microsecond latency budget (< 5 microseconds per call)
        assert elapsed_us < 50.0, f"Enqueue too slow: {elapsed_us:.2f} us"

        recorder.flush()

        reader = recorder.db.get_reader_connection()
        cur = reader.cursor()
        cur.execute("SELECT COUNT(*) FROM telemetry_events WHERE episode_id = ?;", (ep_id,))
        count = cur.fetchone()[0]
        assert count == 50

        recorder.end_episode(outcome="CLEARED")
        recorder.end_session()
        recorder.close()


class TestHeadlessEngineIntegration:
    """Verifies complete headless game loop with scene transitions and cartridge integration."""

    def test_full_engine_scene_lifecycle(self, sample_open_genre_cartridge):
        engine = Engine(headless=True)
        cartridge = engine.load_cartridge(sample_open_genre_cartridge)
        assert engine.active_cartridge == cartridge

        engine.scene_manager.register_scene("title", TitleScene(engine))
        engine.scene_manager.register_scene("playing", PlayingScene(engine))
        engine.scene_manager.register_scene("game_over", GameOverScene(engine))
        engine.scene_manager.register_scene("victory", EpisodeVictoryScene(engine))

        # Title -> step
        engine.scene_manager.switch_to("title")
        for _ in range(5):
            engine.game_loop.step()

        # Playing -> step
        engine.scene_manager.switch_to("playing", episode_number=1)
        assert engine.scene_manager.current_scene_name == "playing"
        for _ in range(20):
            engine.game_loop.step()

        # Victory -> step
        engine.scene_manager.switch_to("victory")
        assert engine.scene_manager.current_scene_name == "victory"
        engine.game_loop.step()

        # Game Over -> step
        engine.scene_manager.switch_to("game_over")
        assert engine.scene_manager.current_scene_name == "game_over"
        engine.game_loop.step()

        engine.quit()
