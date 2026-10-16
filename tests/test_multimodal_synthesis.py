"""tests/test_multimodal_synthesis.py
End-to-End Multimodal Game Synthesis Integration Test Suite.
Verifies the complete flow from SteamPlayerDNA -> WorldSpec -> NarrativeBible ->
Voice Audio (.wav) -> Sprites (.png) -> SWE Code -> Cartridge Loading & Execution.
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import tempfile
from PIL import Image
import pytest

from engine.loader import CartridgeLoader
from engine.steam.models import SteamPlayerDNA
from engine.studio.audio import OfflineMockAudioBackend, VoiceActingAudioProducerAgent, validate_wav_file
from engine.studio.director import CreativeDirectorWorldbuilderAgent
from engine.studio.narrative import NarrativeCharacterWriterAgent
from engine.studio.swe import LeadGameplaySWEAgent
from engine.studio.visual import VisualArtSpriteDesignerAgent, PNG_SIGNATURE


class TestMultimodalSynthesisPipeline:
    """Verifies end-to-end multimodal game compilation and runtime execution."""
    deliberation: str = "Test suite validating end-to-end multimodal synthesis pipeline."

    def test_end_to_end_multimodal_genesis_pipeline(self, mock_gemini_client) -> None:
        """Full pipeline: DNA -> World -> Narrative -> Audio -> Visuals -> SWE -> Loader -> Runtime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cartridge_dir = Path(tmpdir) / "candidate_cartridge"
            cartridge_dir.mkdir(parents=True, exist_ok=True)
            assets_dir = cartridge_dir / "assets"
            audio_dir = assets_dir / "audio"
            sprites_dir = assets_dir / "sprites"
            audio_dir.mkdir(parents=True, exist_ok=True)
            sprites_dir.mkdir(parents=True, exist_ok=True)

            # 1. Player DNA Ingestion
            dna = SteamPlayerDNA(
                deliberation="High-mobility action roguelike profile.",
                player_id="steam_test_player",
                profile_tag="RUSHER_PROFILE",
                total_playtime_hours=450.0,
                pacing_score=0.8,
                challenge_tolerance=0.75,
                top_genres=["Roguelike", "Action"],
                preferred_palette_mood="neon_cyberpunk",
            )

            # 2. Creative Director World Conception
            mock_gemini_client.set_json_response({
                "deliberation": "Genesis world conception for rusher player.",
                "game_title": "Neon Breach",
                "core_genre": "Action Roguelike",
                "thematic_premise": "Deep space neon citadel",
                "world_lore": "Autonomous defense machines in revolt.",
                "conflict_tone": "unforgiving_apocalyptic",
                "conflict_narrative": "Breaching reactor core.",
                "aesthetic_style": "Pixel Art",
                "palette_mood": "neon_cyberpunk",
                "gameplay_pacing": "kinetic_high_mobility",
                "pacing_score": 0.8,
                "target_difficulty": 1.0,
            })
            director = CreativeDirectorWorldbuilderAgent(client=mock_gemini_client)
            world_spec = director.generate_genesis_world(dna)
            assert world_spec.game_title != ""
            assert world_spec.gameplay_pacing == "kinetic_high_mobility"

            # 3. Narrative & Character Writer
            mock_gemini_client.set_json_response({
                "deliberation": "Narrative bible for Neon Breach.",
                "world_title": "Neon Breach",
                "setting_summary": "Cyberpunk citadel deep space station.",
                "characters": [
                    {
                        "deliberation": "Protagonist scout.",
                        "character_id": "hero",
                        "name": "Kaelen",
                        "role": "protagonist",
                        "personality_traits": ["swift", "focused"],
                        "voice_timbre": "focused",
                        "gemini_voice_name": "Aoede",
                        "visual_description": "Neon armored scout",
                        "portrait_asset": "assets/sprites/hero.png",
                    },
                    {
                        "deliberation": "Boss core sentinel.",
                        "character_id": "boss",
                        "name": "Echo",
                        "role": "boss",
                        "personality_traits": ["unrelenting"],
                        "voice_timbre": "deep",
                        "gemini_voice_name": "Fenrir",
                        "visual_description": "Heavy core sentinel",
                        "portrait_asset": "assets/sprites/boss.png",
                    },
                ],
                "cutscenes": [],
                "dialogue_trees": [],
                "dialogue_barks": [
                    {
                        "deliberation": "Start mission bark.",
                        "bark_id": "bark_start",
                        "speaker_id": "hero",
                        "speaker_name": "Kaelen",
                        "trigger_type": "episode_start",
                        "text": "Thrusters online. Entering hostile territory.",
                        "voice_clip": "assets/audio/hero_start.wav",
                        "portrait": "assets/sprites/hero.png",
                    }
                ],
            })
            narrative_agent = NarrativeCharacterWriterAgent(client=mock_gemini_client)
            bible = narrative_agent.generate_narrative_bible(world_spec)
            narrative_json_path = cartridge_dir / "narrative.json"
            narrative_agent.export_narrative_json(bible, narrative_json_path)
            assert narrative_json_path.exists()

            # 4. Voice Acting Audio Producer (Offline mock backend for genuine WAV generation)
            audio_producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
            for bark in bible.dialogue_barks:
                wav_filename = Path(bark.voice_clip).name
                target_wav = audio_dir / wav_filename
                result = audio_producer.synthesize_voice_line(bark, target_wav)
                assert target_wav.exists()
                valid, err, duration = validate_wav_file(target_wav)
                assert valid is True
                assert 0.1 <= duration <= 30.0

            # 5. Visual Art & Sprite Designer
            buf = io.BytesIO()
            Image.new("RGBA", (16, 16), (255, 0, 77, 255)).save(buf, format="PNG")
            sample_png = buf.getvalue()
            # Queue image responses for character portraits and default fallback
            for _ in range(len(bible.characters) + 1):
                mock_gemini_client.set_image_response(sample_png)

            visual_designer = VisualArtSpriteDesignerAgent(client=mock_gemini_client)
            for char in bible.characters:
                portrait_filename = Path(char.portrait_asset).name
                target_portrait = sprites_dir / portrait_filename
                visual_designer.generate_portrait(char, target_portrait, dimensions=(128, 128))
                assert target_portrait.exists()
                assert target_portrait.read_bytes()[:8] == PNG_SIGNATURE

            # Generate entity sprites and tiles
            sprites = visual_designer.generate_cartridge_sprites(sprites_dir, mood=world_spec.palette_mood)
            assert len(sprites) >= 6

            # Copy a default portrait fallback
            default_portrait = sprites_dir / "default_portrait.png"
            if not default_portrait.exists():
                visual_designer.generate_portrait(bible.characters[0], default_portrait)

            # 6. Lead Gameplay SWE (Zero Templates)
            mechanics_code = '''
from typing import Any, Dict, List, Optional
from engine.api.contracts import BaseCartridge, BaseEnemy, BaseHazard, BaseLevelSpec, BasePlayer
from engine.entities.player import Player
from entities import BasicEnemy, SpikeHazard

class KineticActionCartridge(BaseCartridge):
    deliberation = "Complete kinetic action roguelike cartridge implementation."

    def __init__(self, manifest: Optional[Dict[str, Any]] = None, cartridge_dir: Any = None, narrative: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest, cartridge_dir=cartridge_dir, narrative=narrative)

    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            deliberation="Level specification for 1280x720 canvas.",
            level_id=f"level_{episode_number}",
            width=1280.0,
            height=720.0,
            name=f"Level {episode_number}",
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            walls=[],
            enemies=[],
            hazards=[]
        )

    def create_player(self, x: float, y: float, telemetry_recorder: Optional[Any] = None) -> Optional[BasePlayer]:
        return Player(x, y, telemetry_recorder)

    def create_enemies(self, spec: Optional[BaseLevelSpec] = None) -> List[BaseEnemy]:
        return [BasicEnemy(200.0, 200.0)]

    def create_hazards(self, spec: Optional[BaseLevelSpec] = None) -> List[BaseHazard]:
        return [SpikeHazard(300.0, 300.0)]

    def get_scenes(self, engine: Any) -> Dict[str, Any]:
        return {}

    def get_initial_scene(self, engine: Any) -> Optional[Any]:
        return None
'''
            entities_code = '''
from typing import Any
from engine.api.contracts import BaseEnemy, BaseHazard

class BasicEnemy(BaseEnemy):
    deliberation = "Basic roaming enemy."
    def __init__(self, x: float = 0.0, y: float = 0.0):
        super().__init__("basic_enemy", x, y, 32.0, 32.0, max_health=50.0, speed=100.0)

    def update(self, dt: float) -> None:
        pass

    def render(self, surface: Any) -> None:
        pass

class SpikeHazard(BaseHazard):
    deliberation = "Spike hazard obstacle."
    def __init__(self, x: float = 0.0, y: float = 0.0):
        super().__init__("spike_hazard", x, y, 32.0, 32.0)

    def update(self, dt: float) -> None:
        pass

    def render(self, surface: Any) -> None:
        pass
'''
            cartridge_bundle = {
                "deliberation": "Open-genre cartridge bundle for Neon Breach.",
                "manifest_data": {
                    "title": "Neon Breach",
                    "version": "1.0.0",
                    "genre": "Action Roguelike",
                },
                "mechanics_code": mechanics_code,
                "scenes_code": None,
                "entities_code": entities_code,
                "abilities_code": None,
                "level_data": {"grid": [], "width": 1280, "height": 720},
            }
            mock_gemini_client.set_json_response(cartridge_bundle)
            swe_agent = LeadGameplaySWEAgent(client=mock_gemini_client)
            code_files = swe_agent.synthesize_cartridge_code(world_spec, bible, output_dir=cartridge_dir)
            assert "mechanics.py" in code_files
            assert "entities.py" in code_files
            assert "level.json" in code_files

            # 7. Multi-Module Cartridge Loader
            loader = CartridgeLoader()
            cartridge = loader.load_cartridge(cartridge_dir)
            assert cartridge is not None

            level_spec = cartridge.get_level_spec(1)
            assert level_spec.width == 1280.0
            assert level_spec.height == 720.0

            player = cartridge.create_player(120.0, 360.0)
            assert player is not None
            assert player.x == 120.0
            assert player.y == 360.0
            assert player.health == 100.0

            enemies = cartridge.create_enemies(level_spec)
            assert len(enemies) >= 1

            hazards = cartridge.create_hazards(level_spec)
            assert len(hazards) >= 1

            # 9. Verify Narrative and Assets Integration
            narrative_data = cartridge.get_narrative()
            assert "characters" in narrative_data
            assert "dialogue_barks" in narrative_data

    def test_multimodal_episodic_update_pipeline(self, mock_gemini_client) -> None:
        """Verifies episodic update evolution cycle with clamped adaptation and story continuation."""
        mock_gemini_client.set_json_response({
            "deliberation": "Genesis world conception.",
            "game_title": "Sector Genesis",
            "core_genre": "Action",
            "thematic_premise": "Outer rim research node",
            "world_lore": "Ancient orbital station",
            "conflict_tone": "grim_tense",
            "conflict_narrative": "Containment breach",
            "aesthetic_style": "Pixel Art",
            "palette_mood": "retro_arcade",
            "gameplay_pacing": "balanced",
            "pacing_score": 0.0,
            "target_difficulty": 1.0,
        })
        mock_gemini_client.set_json_response({
            "deliberation": "Episodic update adapting to player mobility.",
            "target_episode": 2,
            "previous_episode": 0,
            "narrative_progression": "Core containment collapsed into sector 2.",
            "escalation_factor": 0.12,
            "pacing_adjustment": 0.05,
            "new_environmental_threats": ["Plasma venting"],
            "mechanics_evolution_directives": ["Enhanced mobility traps"],
            "hypotheses_addressed": ["H_PLAYER_MOBILITY_AND_DASH_CADENCE", "H_HAZARD_AND_THREAT_LETHALITY_FATIGUE"],
        })
        director = CreativeDirectorWorldbuilderAgent(client=mock_gemini_client)
        dna = SteamPlayerDNA(deliberation="DNA", challenge_tolerance=0.7)
        genesis_world = director.generate_genesis_world(dna)

        update_spec = director.generate_episodic_update_world(
            dna=dna,
            hypothesis_report={"hypotheses": ["H_PLAYER_MOBILITY_AND_DASH_CADENCE", "H_HAZARD_AND_THREAT_LETHALITY_FATIGUE"]},
            previous_spec=genesis_world,
            target_episode=2,
        )
        assert update_spec.target_episode == 2
        assert update_spec.previous_episode == 0
        assert len(update_spec.new_environmental_threats) >= 1
        assert abs(update_spec.escalation_factor) <= 0.15
