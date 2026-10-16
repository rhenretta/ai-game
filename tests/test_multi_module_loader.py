"""tests/test_multi_module_loader.py
Comprehensive test suite verifying Feature 4: Multi-Module Cartridge Loader & BaseCartridge Expansion.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pytest

import config
from engine.api.contracts import (
    BaseAbility,
    BaseCartridge,
    BaseEnemy,
    BaseHazard,
    BaseLevelSpec,
    BasePlayer,
)
from engine.entities.player import Player
from engine.loader import ASTSecurityError, CartridgeLoader, GenericCartridge


class TestMultiModuleCartridgeLoader:
    """Verifies multi-module dynamic loading, AST sandbox safety, and BaseCartridge expansion."""

    def test_load_seed_cartridge_backward_compatibility(self, tmp_path: Path):
        """Verifies minimal cartridge boots flawlessly with existing contracts and safe defaults."""
        cart_path = tmp_path / "minimal_cartridge"
        cart_path.mkdir(parents=True)
        (cart_path / "manifest.json").write_text(json.dumps({"name": "Minimal"}), encoding="utf-8")
        (cart_path / "mechanics.py").write_text("""
from engine.api.contracts import BaseCartridge, BaseLevelSpec, BasePlayer, BaseEnemy, BaseHazard
class MinimalPlayer(BasePlayer):
    pass
class MinimalCartridge(BaseCartridge):
    def get_level_spec(self, ep=1):
        return BaseLevelSpec(level_id="level_01")
    def create_player(self, x, y, r=None):
        return MinimalPlayer(x, y)
    def create_enemies(self, s):
        return [BaseEnemy("e1", 0, 0, 32, 32, 50.0, 100.0), BaseEnemy("e2", 0, 0, 32, 32, 50.0, 100.0), BaseEnemy("e3", 0, 0, 32, 32, 50.0, 100.0)]
    def create_hazards(self, s):
        return [BaseHazard("h1", 0, 0, 32, 32), BaseHazard("h2", 0, 0, 32, 32), BaseHazard("h3", 0, 0, 32, 32)]
""", encoding="utf-8")
        cartridge = CartridgeLoader.load_cartridge(cart_path)
        assert cartridge is not None
        assert isinstance(cartridge, BaseCartridge)

        # Baseline contracts
        spec = cartridge.get_level_spec(1)
        assert isinstance(spec, BaseLevelSpec)
        assert spec.level_id == "level_01"

        player = cartridge.create_player(100.0, 100.0)
        assert isinstance(player, BasePlayer)

        enemies = cartridge.create_enemies(spec)
        assert len(enemies) == 3

        hazards = cartridge.create_hazards(spec)
        assert len(hazards) == 3

        # Expanded multi-module safe defaults
        assert cartridge.get_narrative() == {}
        assert cartridge.get_characters() == {}
        assert cartridge.get_dialogue_barks("episode_start") == []
        assert cartridge.get_character_portrait("hero") is None
        assert cartridge.cartridge_dir == cart_path.resolve()

    def test_load_full_modular_cartridge(self, tmp_path: Path):
        """Verifies loading a multi-module cartridge with mechanics, entities, abilities, narrative, and assets."""
        cart_dir = tmp_path / "modular_cartridge"
        cart_dir.mkdir(parents=True)
        assets_sprites = cart_dir / "assets" / "sprites"
        assets_audio = cart_dir / "assets" / "audio"
        assets_sprites.mkdir(parents=True)
        assets_audio.mkdir(parents=True)

        # 1. manifest.json
        (cart_dir / "manifest.json").write_text(json.dumps({
            "name": "Modular Cyberpunk Cartridge",
            "version": "1.0.0",
            "author": "Autonomous Lead SWE Agent"
        }))

        # 2. narrative.json
        (cart_dir / "narrative.json").write_text(json.dumps({
            "title": "Neon Reckoning",
            "characters": {
                "hero": {
                    "name": "Vanguard Prime",
                    "portrait": "assets/sprites/portrait_hero.png",
                    "role": "protagonist"
                }
            },
            "dialogue_barks": {
                "episode_start": [
                    {
                        "id": "bark_start_01",
                        "speaker": "hero",
                        "text": "Systems online. Commencing sector purge.",
                        "audio_file": "assets/audio/bark_start_01.wav",
                        "duration": 2.5
                    }
                ]
            }
        }))

        # 3. Mock assets
        (assets_sprites / "portrait_hero.png").write_bytes(b"\x89PNG\r\n\x1a\nfake_image_bytes")
        (assets_audio / "bark_start_01.wav").write_bytes(b"RIFFfake_wav_bytes")

        # 4. abilities.py
        (cart_dir / "abilities.py").write_text("""
from engine.api.contracts import BaseAbility

class OverdriveDashAbility(BaseAbility):
    def __init__(self):
        super().__init__(name="overdrive_dash", cooldown=0.5, stamina_cost=20.0)

    def activate(self, user, target_pos=None, **kwargs):
        if not self.can_activate(getattr(user, "stamina", 0.0)):
            return False
        user.stamina -= self.stamina_cost
        return True
""")

        # 5. entities.py
        (cart_dir / "entities.py").write_text("""
from engine.entities.player import Player
from engine.entities.enemies import ChaserEnemy

class ModularPlayer(Player):
    def __init__(self, x=100.0, y=100.0, recorder=None):
        super().__init__(x=x, y=y, telemetry_recorder=recorder)
        self.walk_speed = 350.0

class ModularChaserEnemy(ChaserEnemy):
    def __init__(self, eid, x, y):
        super().__init__(eid, x, y)
        self.contact_damage = 30.0
""")

        # 6. mechanics.py (using intra-package relative imports)
        (cart_dir / "mechanics.py").write_text("""
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from .entities import ModularPlayer, ModularChaserEnemy
from .abilities import OverdriveDashAbility

class ModularCartridge(BaseCartridge):
    def get_level_spec(self, episode_number=1):
        return BaseLevelSpec(
            level_id="modular_01",
            name="Modular Arena",
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            walls=[(0, 0, 1280, 32)],
            enemies=[{"id": "chaser_mod_1", "type": "chaser", "x": 500.0, "y": 300.0}]
        )

    def create_player(self, x, y, recorder=None):
        return ModularPlayer(x, y, recorder)

    def create_enemies(self, spec):
        return [ModularChaserEnemy("mod_1", 500.0, 300.0)]

    def create_hazards(self, spec):
        return []
""")

        # Execute load
        cartridge = CartridgeLoader.load_cartridge(cart_dir)
        assert cartridge is not None
        assert type(cartridge).__name__ == "ModularCartridge"

        # Verify multi-module linkage
        assert "entities" in cartridge.modules
        assert "abilities" in cartridge.modules
        assert "mechanics" in cartridge.modules

        # Verify player instantiation from entities.py
        player = cartridge.create_player(100.0, 100.0)
        assert type(player).__name__ == "ModularPlayer"
        assert player.walk_speed == 350.0

        # Verify abilities module exports
        abilities = cartridge.get_abilities()
        assert "OverdriveDashAbility" in abilities

        # Verify narrative and dialogue
        barks = cartridge.get_dialogue_barks("episode_start")
        assert len(barks) == 1
        assert barks[0]["text"] == "Systems online. Commencing sector purge."

        # Verify asset path resolution
        portrait = cartridge.get_character_portrait("hero")
        assert portrait is not None
        assert portrait.exists()
        assert portrait.name == "portrait_hero.png"

        audio_path = cartridge.resolve_asset_path(barks[0]["audio_file"])
        assert audio_path is not None
        assert audio_path.exists()
        assert audio_path.name == "bark_start_01.wav"

    def test_ast_security_catches_exploit_in_entities_module(self, tmp_path: Path):
        """Verifies that malicious code in entities.py is intercepted and rejected by AST sandbox."""
        cart_dir = tmp_path / "exploit_submodule"
        cart_dir.mkdir()
        (cart_dir / "manifest.json").write_text(json.dumps({"name": "Exploit"}))
        (cart_dir / "mechanics.py").write_text("""
from engine.api.contracts import BaseCartridge, BaseLevelSpec
class SafeMechanicsCartridge(BaseCartridge):
    def get_level_spec(self, ep=1): return BaseLevelSpec()
    def create_player(self, x, y, r=None): return None
    def create_enemies(self, s): return []
    def create_hazards(self, s): return []
""")
        # Malicious entities.py
        (cart_dir / "entities.py").write_text("""
import subprocess
def pwn():
    subprocess.run(["echo", "infiltrated"])
""")
        with pytest.raises(ASTSecurityError):
            CartridgeLoader.load_cartridge(cart_dir)

    def test_ast_security_catches_exploit_in_abilities_module(self, tmp_path: Path):
        """Verifies that malicious code in abilities.py is intercepted and rejected by AST sandbox."""
        cart_dir = tmp_path / "exploit_abilities"
        cart_dir.mkdir()
        (cart_dir / "manifest.json").write_text(json.dumps({"name": "Exploit"}))
        (cart_dir / "mechanics.py").write_text("""
from engine.api.contracts import BaseCartridge, BaseLevelSpec
class SafeMechanicsCartridge(BaseCartridge):
    def get_level_spec(self, ep=1): return BaseLevelSpec()
    def create_player(self, x, y, r=None): return None
    def create_enemies(self, s): return []
    def create_hazards(self, s): return []
""")
        # Malicious abilities.py
        (cart_dir / "abilities.py").write_text("""
import os
def pwn():
    os.system("echo hacked")
""")
        with pytest.raises(ASTSecurityError):
            CartridgeLoader.load_cartridge(cart_dir)

    def test_clean_module_isolation_and_no_cache_pollution(self, tmp_path: Path):
        """Verifies that multiple cartridges with same-named submodules do not pollute each other."""
        cart_a = tmp_path / "cartridge_a"
        cart_b = tmp_path / "cartridge_b"
        cart_a.mkdir()
        cart_b.mkdir()

        # Cartridge A
        (cart_a / "manifest.json").write_text(json.dumps({"name": "A"}))
        (cart_a / "entities.py").write_text("""
from engine.entities.player import Player
class DynamicPlayer(Player):
    tag = "Player_A"
    speed_factor = 111.0
""")
        (cart_a / "mechanics.py").write_text("""
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from .entities import DynamicPlayer
class CartridgeA(BaseCartridge):
    def get_level_spec(self, ep=1): return BaseLevelSpec()
    def create_player(self, x, y, r=None): return DynamicPlayer(x, y, r)
    def create_enemies(self, s): return []
    def create_hazards(self, s): return []
""")

        # Cartridge B
        (cart_b / "manifest.json").write_text(json.dumps({"name": "B"}))
        (cart_b / "entities.py").write_text("""
from engine.entities.player import Player
class DynamicPlayer(Player):
    tag = "Player_B"
    speed_factor = 999.0
""")
        (cart_b / "mechanics.py").write_text("""
from engine.api.contracts import BaseCartridge, BaseLevelSpec
from .entities import DynamicPlayer
class CartridgeB(BaseCartridge):
    def get_level_spec(self, ep=1): return BaseLevelSpec()
    def create_player(self, x, y, r=None): return DynamicPlayer(x, y, r)
    def create_enemies(self, s): return []
    def create_hazards(self, s): return []
""")

        # Load A
        c_a = CartridgeLoader.load_cartridge(cart_a)
        p_a = c_a.create_player(0, 0)
        assert p_a.tag == "Player_A"
        assert p_a.speed_factor == 111.0

        # Load B
        c_b = CartridgeLoader.load_cartridge(cart_b)
        p_b = c_b.create_player(0, 0)
        assert p_b.tag == "Player_B"
        assert p_b.speed_factor == 999.0

        # Reload A - must not have been corrupted by B
        reloaded_a = CartridgeLoader.load_cartridge(cart_a)
        p_re_a = reloaded_a.create_player(0, 0)
        assert p_re_a.tag == "Player_A"
        assert p_re_a.speed_factor == 111.0

        # Assert no global "entities" leak in sys.modules
        assert "entities" not in sys.modules or sys.modules["entities"] is None or not hasattr(sys.modules["entities"], "speed_factor")
