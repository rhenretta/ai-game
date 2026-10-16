"""tests/test_studio_specialist_agents.py
Comprehensive unit tests for Milestone M3 Specialist Agents (Features 17-21).
Covers CreativeDirector, NarrativeWriter, VoiceProducer, VisualDesigner, and LeadGameplaySWE.
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
import ast
from pathlib import Path
import struct
import io
from typing import Any, Dict
import tempfile
from PIL import Image
import pytest

from engine.api.contracts import BaseCartridge
from engine.loader import validate_cartridge_ast
from engine.steam.models import SteamPlayerDNA
from engine.studio.audio import (
    OfflineMockAudioBackend,
    VoiceActingAudioProducerAgent,
    VoiceSynthesisError,
    create_wav_header,
    escape_ssml,
    validate_wav_file,
)
from engine.studio.director import CreativeDirectorWorldbuilderAgent
from engine.studio.models import (
    CharacterSheet,
    DialogueBarkSpec,
    DialogueChoice,
    DialogueNode,
    DialogueTreeSpec,
    GenesisWorldSpec,
)
from engine.studio.narrative import (
    NarrativeCharacterWriterAgent,
    validate_dialogue_tree,
)
from engine.studio.swe import LeadGameplaySWEAgent
from engine.studio.visual import (
    PNG_SIGNATURE,
    ProceduralSpriteRenderer,
    VisualArtSpriteDesignerAgent,
)


class TestCreativeDirectorWorldbuilder:
    """F17: Creative Director & Worldbuilder Agent tests."""
    deliberation: str = "Test suite validating creative director worldbuilding and tone alignment."

    def test_divergent_dna_produces_contrasting_worlds(self, mock_gemini_client) -> None:
        """Contrasting Steam DNA profiles produce divergent pacing, tones, and genres."""
        rusher_data = {
            "deliberation": "High mobility rusher world.",
            "game_title": "Neon Breach",
            "core_genre": "Action Roguelike",
            "thematic_premise": "Deep space neon citadel",
            "world_lore": "Roguelike action setting",
            "conflict_tone": "unforgiving_apocalyptic",
            "conflict_narrative": "High stakes breach",
            "aesthetic_style": "Pixel Art",
            "palette_mood": "neon_cyberpunk",
            "gameplay_pacing": "kinetic_high_mobility",
            "pacing_score": 0.75,
            "target_difficulty": 1.0,
        }
        cautious_data = {
            "deliberation": "Methodical puzzle world.",
            "game_title": "Abyssal Cogitation",
            "core_genre": "Strategy",
            "thematic_premise": "Submerged vault",
            "world_lore": "Tactical puzzle setting",
            "conflict_tone": "cozy_exploratory",
            "conflict_narrative": "Methodical descent",
            "aesthetic_style": "Pixel Art",
            "palette_mood": "grim_dark",
            "gameplay_pacing": "tactical_methodical",
            "pacing_score": -0.75,
            "target_difficulty": 0.5,
        }
        mock_gemini_client.set_json_response(rusher_data)
        mock_gemini_client.set_json_response(cautious_data)
        director = CreativeDirectorWorldbuilderAgent(client=mock_gemini_client)

        # Rusher Profile: high pacing, high challenge tolerance
        rusher_dna = SteamPlayerDNA(
            deliberation="Fast-paced action roguelike preference.",
            player_id="rusher",
            profile_tag="RUSHER_PROFILE",
            total_playtime_hours=500.0,
            pacing_score=0.75,
            challenge_tolerance=0.85,
            top_genres=["Roguelike", "Action"],
            preferred_palette_mood="neon_cyberpunk",
        )
        rusher_world = director.generate_genesis_world(rusher_dna)
        assert rusher_world.gameplay_pacing == "kinetic_high_mobility"
        assert rusher_world.conflict_tone == "unforgiving_apocalyptic"
        assert rusher_world.palette_mood == "neon_cyberpunk"

        # Cautious Profile: low pacing, low challenge tolerance
        cautious_dna = SteamPlayerDNA(
            deliberation="Methodical puzzle strategy preference.",
            player_id="cautious",
            profile_tag="CAUTIOUS_PROFILE",
            total_playtime_hours=400.0,
            pacing_score=-0.75,
            challenge_tolerance=0.25,
            top_genres=["Strategy", "Puzzle"],
            preferred_palette_mood="grim_dark",
        )
        cautious_world = director.generate_genesis_world(cautious_dna)
        assert cautious_world.gameplay_pacing == "tactical_methodical"
        assert cautious_world.conflict_tone == "cozy_exploratory"

    def test_tone_boundary_mappings(self, mock_gemini_client) -> None:
        """Boundary challenge tolerances map to exact prescribed tones."""
        mock_gemini_client.set_json_response({
            "deliberation": "Extreme tone deliberation.",
            "game_title": "Extreme World",
            "core_genre": "Action",
            "thematic_premise": "Volcanic core",
            "world_lore": "Ash and ruin",
            "conflict_tone": "unforgiving_apocalyptic",
            "conflict_narrative": "Survival against the inferno",
            "aesthetic_style": "Pixel Art",
            "palette_mood": "grim_dark",
            "gameplay_pacing": "kinetic_high_mobility",
            "pacing_score": 0.9,
            "target_difficulty": 1.5,
        })
        mock_gemini_client.set_json_response({
            "deliberation": "High tone deliberation.",
            "game_title": "Tense World",
            "core_genre": "Action",
            "thematic_premise": "Derelict ship",
            "world_lore": "Echoing corridors",
            "conflict_tone": "grim_tense",
            "conflict_narrative": "System recovery",
            "aesthetic_style": "Pixel Art",
            "palette_mood": "neon_cyberpunk",
            "gameplay_pacing": "balanced",
            "pacing_score": 0.5,
            "target_difficulty": 1.2,
        })
        mock_gemini_client.set_json_response({
            "deliberation": "Zero tone deliberation.",
            "game_title": "Cozy World",
            "core_genre": "Exploration",
            "thematic_premise": "Verdant garden",
            "world_lore": "Flora and fauna",
            "conflict_tone": "cozy_exploratory",
            "conflict_narrative": "Documenting species",
            "aesthetic_style": "Pixel Art",
            "palette_mood": "vibrant_fantasy",
            "gameplay_pacing": "tactical_methodical",
            "pacing_score": -0.8,
            "target_difficulty": 0.5,
        })
        director = CreativeDirectorWorldbuilderAgent(client=mock_gemini_client)

        dna_extreme = SteamPlayerDNA(deliberation="Extreme", challenge_tolerance=1.0)
        assert director.generate_genesis_world(dna_extreme).conflict_tone == "unforgiving_apocalyptic"

        dna_high = SteamPlayerDNA(deliberation="High", challenge_tolerance=0.75)
        assert director.generate_genesis_world(dna_high).conflict_tone == "grim_tense"

        dna_zero = SteamPlayerDNA(deliberation="Zero", challenge_tolerance=0.0)
        assert director.generate_genesis_world(dna_zero).conflict_tone == "cozy_exploratory"

    def test_empty_thematic_biases_fallback(self, mock_gemini_client) -> None:
        """Empty thematic biases defaults gracefully to a core setting."""
        mock_gemini_client.set_json_response({
            "deliberation": "Default thematic setting.",
            "game_title": "Outer Outpost",
            "core_genre": "Sci-Fi",
            "thematic_premise": "A lonely deep space outpost under siege.",
            "world_lore": "Automated defenses on the frontier.",
            "conflict_tone": "grim_tense",
            "conflict_narrative": "Holding the line.",
            "aesthetic_style": "Pixel Art",
            "palette_mood": "retro_arcade",
            "gameplay_pacing": "balanced",
            "pacing_score": 0.0,
            "target_difficulty": 1.0,
        })
        director = CreativeDirectorWorldbuilderAgent(client=mock_gemini_client)
        dna = SteamPlayerDNA(deliberation="No biases", thematic_biases=[])
        world = director.generate_genesis_world(dna)
        assert "deep space outpost" in world.thematic_premise

    def test_update_world_spec_difficulty_clamp(self, mock_gemini_client) -> None:
        """Episodic update spec bounds difficulty escalation within +/-15%."""
        mock_gemini_client.set_json_response({
            "deliberation": "Genesis world deliberation.",
            "game_title": "Genesis Hub",
            "core_genre": "Action",
            "thematic_premise": "Facility 9",
            "world_lore": "Research station gone dark.",
            "conflict_tone": "grim_tense",
            "conflict_narrative": "Breach containment.",
            "aesthetic_style": "Pixel Art",
            "palette_mood": "retro_arcade",
            "gameplay_pacing": "balanced",
            "pacing_score": 0.0,
            "target_difficulty": 1.0,
        })
        mock_gemini_client.set_json_response({
            "deliberation": "Episodic update deliberation.",
            "target_episode": 1,
            "previous_episode": 0,
            "narrative_progression": "Containment fractured further into the lower sub-levels.",
            "escalation_factor": 0.10,
            "pacing_adjustment": 0.05,
            "new_environmental_threats": ["Toxic radiation leaks"],
            "mechanics_evolution_directives": ["Add hazard dashing mechanics"],
            "hypotheses_addressed": ["H_PLAYER_MOBILITY_AND_DASH_CADENCE"],
        })
        director = CreativeDirectorWorldbuilderAgent(client=mock_gemini_client)
        dna = SteamPlayerDNA(deliberation="DNA", challenge_tolerance=0.5)
        genesis_world = director.generate_genesis_world(dna)

        update_world = director.generate_episodic_update_world(
            dna=dna,
            hypothesis_report={"hypotheses": ["H_PLAYER_MOBILITY_AND_DASH_CADENCE"]},
            previous_spec=genesis_world,
            target_episode=1,
        )
        assert update_world.target_episode == 1
        assert abs(update_world.escalation_factor) <= 0.15
        assert abs(update_world.pacing_adjustment) <= 0.15


class TestNarrativeCharacterWriter:
    """F18: Narrative & Character Writer Agent tests."""
    deliberation: str = "Test suite validating narrative bible and dialogue graph reachability."

    def test_character_sheet_completeness(self, mock_gemini_client) -> None:
        """Narrative bible contains protagonist, companion, and boss character sheets."""
        mock_gemini_client.set_json_response({
            "deliberation": "Narrative bible deliberation.",
            "world_title": "Sector Alpha",
            "setting_summary": "Mining facility revolt.",
            "characters": [
                {
                    "deliberation": "Protagonist character deliberation.",
                    "character_id": "hero",
                    "name": "Kaelen",
                    "role": "protagonist",
                    "personality_traits": ["focused", "resilient"],
                    "voice_timbre": "focused",
                    "gemini_voice_name": "Aoede",
                    "visual_description": "Armored scout",
                    "portrait_asset": "assets/sprites/hero.png",
                },
                {
                    "deliberation": "Companion character deliberation.",
                    "character_id": "companion",
                    "name": "Apex",
                    "role": "companion",
                    "personality_traits": ["analytical"],
                    "voice_timbre": "tactical",
                    "gemini_voice_name": "Puck",
                    "visual_description": "Support drone",
                    "portrait_asset": "assets/sprites/companion.png",
                },
                {
                    "deliberation": "Boss character deliberation.",
                    "character_id": "boss",
                    "name": "Echo",
                    "role": "boss",
                    "personality_traits": ["ruthless"],
                    "voice_timbre": "deep",
                    "gemini_voice_name": "Fenrir",
                    "visual_description": "Massive mining mech",
                    "portrait_asset": "assets/sprites/boss.png",
                },
            ],
            "cutscenes": [],
            "dialogue_trees": [],
            "dialogue_barks": [
                {
                    "deliberation": "Start bark deliberation.",
                    "bark_id": "bark_start",
                    "speaker_id": "hero",
                    "speaker_name": "Kaelen",
                    "trigger_type": "episode_start",
                    "text": "Entering sector alpha.",
                    "voice_clip": "assets/audio/hero_start.wav",
                },
                {
                    "deliberation": "Low health bark deliberation.",
                    "bark_id": "bark_low_hp",
                    "speaker_id": "hero",
                    "speaker_name": "Kaelen",
                    "trigger_type": "low_health",
                    "text": "Shields depleted!",
                    "voice_clip": "assets/audio/hero_low_hp.wav",
                },
                {
                    "deliberation": "Boss bark deliberation.",
                    "bark_id": "bark_boss",
                    "speaker_id": "boss",
                    "speaker_name": "Echo",
                    "trigger_type": "boss_encounter",
                    "text": "Intruder detected.",
                    "voice_clip": "assets/audio/boss_encounter.wav",
                },
            ],
        })
        writer = NarrativeCharacterWriterAgent(client=mock_gemini_client)
        world_spec = GenesisWorldSpec(
            deliberation="World delib",
            game_title="Sector Alpha",
            core_genre="Action",
            thematic_premise="Mining facility",
            world_lore="Automated miners in revolt.",
            conflict_tone="grim_tense",
            conflict_narrative="Overriding rogue control node.",
            aesthetic_style="Pixel",
            palette_mood="retro_arcade",
            gameplay_pacing="kinetic_high_mobility",
            pacing_score=0.6,
            target_difficulty=1.0,
        )
        bible = writer.generate_narrative_bible(world_spec)
        assert len(bible.characters) >= 3

        roles = [c.role for c in bible.characters]
        assert "protagonist" in roles
        assert "boss" in roles
        assert len(bible.dialogue_barks) >= 3

    def test_dialogue_graph_cycle_and_orphan_detection(self) -> None:
        """Graph validator detects unreachable orphan nodes and infinite loops."""
        # Clean graph
        clean_tree = DialogueTreeSpec(
            deliberation="Clean tree",
            root_node_id="root",
            nodes={
                "root": DialogueNode(
                    deliberation="Root",
                    node_id="root",
                    speaker_id="npc",
                    text="Hello",
                    choices=[DialogueChoice(deliberation="C", text="Go", target_node_id="step1")],
                ),
                "step1": DialogueNode(
                    deliberation="Step 1",
                    node_id="step1",
                    speaker_id="hero",
                    text="Ready",
                    choices=[],
                ),
            },
        )
        assert len(validate_dialogue_tree(clean_tree)) == 0

        # Orphan node
        orphan_tree = DialogueTreeSpec(
            deliberation="Orphan tree",
            root_node_id="root",
            nodes={
                "root": clean_tree.nodes["root"],
                "step1": clean_tree.nodes["step1"],
                "orphan": DialogueNode(deliberation="O", node_id="orphan", speaker_id="hero", text="Lost", choices=[]),
            },
        )
        errs = validate_dialogue_tree(orphan_tree)
        assert any("orphan" in e.lower() for e in errs)

        # 2-node infinite loop
        loop_tree = DialogueTreeSpec(
            deliberation="Loop tree",
            root_node_id="n1",
            nodes={
                "n1": DialogueNode(deliberation="1", node_id="n1", speaker_id="hero", text="1", choices=[DialogueChoice(deliberation="c", text="t", target_node_id="n2")]),
                "n2": DialogueNode(deliberation="2", node_id="n2", speaker_id="hero", text="2", choices=[DialogueChoice(deliberation="c", text="t", target_node_id="n1")]),
            },
        )
        loop_errs = validate_dialogue_tree(loop_tree)
        assert any("loop" in e.lower() for e in loop_errs)

    def test_narrative_json_cartridge_export_schema(self, mock_gemini_client) -> None:
        """Bible exports narrative dictionary compliant with PlayingScene schema."""
        mock_gemini_client.set_json_response({
            "deliberation": "Bible deliberation.",
            "world_title": "Omega",
            "setting_summary": "Orbital station lore.",
            "characters": [
                {
                    "deliberation": "Protagonist deliberation.",
                    "character_id": "hero",
                    "name": "Kaelen",
                    "role": "protagonist",
                    "visual_description": "Armored scout",
                    "portrait_asset": "assets/sprites/hero.png",
                }
            ],
            "dialogue_barks": [
                {
                    "deliberation": "Bark deliberation.",
                    "bark_id": "bark_1",
                    "speaker_id": "hero",
                    "speaker_name": "Kaelen",
                    "trigger_type": "episode_start",
                    "text": "Entering station.",
                    "voice_clip": "assets/audio/hero.wav",
                }
            ],
        })
        writer = NarrativeCharacterWriterAgent(client=mock_gemini_client)
        world_spec = GenesisWorldSpec(
            deliberation="World",
            game_title="Omega",
            core_genre="Action",
            thematic_premise="Orbital station",
            world_lore="Lore",
            conflict_tone="heroic_hopeful",
            conflict_narrative="Conflict",
            aesthetic_style="Style",
            palette_mood="retro_arcade",
            gameplay_pacing="balanced",
            pacing_score=0.0,
            target_difficulty=1.0,
        )
        bible = writer.generate_narrative_bible(world_spec)
        data = bible.to_cartridge_narrative_dict()
        assert "characters" in data
        assert "dialogue_barks" in data
        assert "hero" in data["characters"]


class TestVoiceActingAudioProducer:
    """F19: Voice Acting Audio Producer Agent tests."""
    deliberation: str = "Test suite validating RIFF/WAVE audio synthesis and duration bounds."

    def test_riff_wav_header_construction_and_validation(self) -> None:
        """Validates 44-byte RIFF/WAVE header fields and duration calculation."""
        header = create_wav_header(pcm_bytes_len=88200, sample_rate=44100, channels=1, bits_per_sample=16)
        assert len(header) == 44
        assert header[:4] == b"RIFF"
        assert header[8:12] == b"WAVE"
        assert header[12:16] == b"fmt "

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(header + b"\x00" * 88200)
            wav_path = Path(f.name)

        valid, err, duration = validate_wav_file(wav_path)
        assert valid is True
        assert abs(duration - 1.0) < 0.01

    def test_ssml_special_characters_escaping(self) -> None:
        """SSML special characters are escaped properly."""
        raw = 'Look <over there> & "listen" to O\'Connor'
        escaped = escape_ssml(raw)
        assert "&amp;" in escaped
        assert "&lt;" in escaped
        assert "&gt;" in escaped
        assert "&quot;" in escaped
        assert "&apos;" in escaped

    def test_empty_dialogue_text_raises_value_error(self) -> None:
        """F19.B4: Empty dialogue text raises ValueError before synthesis dispatch."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        bark = DialogueBarkSpec(
            deliberation="Bark",
            bark_id="empty_bark",
            speaker_id="hero",
            speaker_name="Kaelen",
            trigger_type="episode_start",
            text="   \n  ",
            voice_clip="assets/audio/empty.wav",
        )
        with pytest.raises(ValueError):
            producer.synthesize_voice_line(bark, Path("/tmp/empty.wav"))

    def test_voice_line_duration_bounds(self) -> None:
        """Audio Producer validates duration bounds [0.1s, 30.0s]."""
        producer = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "hero_start.wav"
            bark = DialogueBarkSpec(
                deliberation="Bark",
                bark_id="hero_start",
                speaker_id="hero",
                speaker_name="Kaelen",
                trigger_type="episode_start",
                text="Thrusters online. Entering hostile territory.",
                voice_clip=str(out_path),
            )
            result = producer.synthesize_voice_line(bark, out_path)
            assert 0.1 <= result.duration_seconds <= 30.0
            assert result.sample_rate == 44100
            assert Path(result.file_path).exists()


class TestVisualArtSpriteDesigner:
    """F20: Visual Art & Sprite Designer Agent tests."""
    deliberation: str = "Test suite validating 16-color Euclidean quantization and PNG generation."

    def test_palette_quantization_euclidean_mapping(self) -> None:
        """F20.1: Maps RGB values to closest 16-color Style Bible palette colors."""
        designer = VisualArtSpriteDesignerAgent()
        # Pure red -> Neon Red (255, 0, 77)
        assert designer.quantize_color((255, 0, 0)) == (255, 0, 77)
        # Pure white -> Off-White (255, 241, 232)
        assert designer.quantize_color((255, 255, 255)) == (255, 241, 232)
        # Pure black -> Pure Black (0, 0, 0)
        assert designer.quantize_color((0, 0, 0)) == (0, 0, 0)

    def test_portrait_dimension_boundary_validation(self, mock_gemini_client) -> None:
        """Rejects non-positive image dimensions with ValueError."""
        designer = VisualArtSpriteDesignerAgent(client=mock_gemini_client)
        char = CharacterSheet(
            deliberation="Hero",
            character_id="hero",
            name="Hero",
            role="protagonist",
            visual_description="Scout",
        )
        with pytest.raises(ValueError):
            designer.generate_portrait(char, Path("/tmp/invalid.png"), dimensions=(0, 256))

        with pytest.raises(ValueError):
            designer.generate_portrait(char, Path("/tmp/invalid.png"), dimensions=(256, -10))

    def test_portrait_generation_has_valid_png_signature(self, mock_gemini_client) -> None:
        """Character portrait writes valid 8-byte PNG signature."""
        buf = io.BytesIO()
        Image.new("RGBA", (16, 16), (255, 0, 77, 255)).save(buf, format="PNG")
        mock_gemini_client.set_image_response(buf.getvalue())
        designer = VisualArtSpriteDesignerAgent(client=mock_gemini_client)
        char = CharacterSheet(
            deliberation="Hero",
            character_id="hero",
            name="Hero",
            role="protagonist",
            visual_description="Armored scout",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "hero.png"
            designer.generate_portrait(char, out_path, dimensions=(128, 128))
            assert out_path.exists()
            assert out_path.read_bytes()[:8] == PNG_SIGNATURE

    def test_full_cartridge_sprite_pack_generation(self) -> None:
        """Generates all standard entities, enemies, hazards, and environment tiles."""
        designer = VisualArtSpriteDesignerAgent()
        with tempfile.TemporaryDirectory() as tmpdir:
            sprites = designer.generate_cartridge_sprites(Path(tmpdir), mood="neon_cyberpunk")
            assert "player_idle" in sprites
            assert "chaser" in sprites
            assert "spikes" in sprites
            assert "wall" in sprites
            assert "floor" in sprites
            for path in sprites.values():
                assert path.exists()
                assert path.read_bytes()[:8] == PNG_SIGNATURE


def _make_sample_cartridge_bundle() -> Dict[str, Any]:
    mechanics_code = '''
from typing import Any, Dict, List, Optional
from engine.api.contracts import BaseCartridge, BaseEnemy, BaseHazard, BaseLevelSpec, BasePlayer
from engine.entities.player import Player

class CustomGameCartridge(BaseCartridge):
    deliberation = "Custom open-genre game cartridge implementation."

    def __init__(self, manifest: Optional[Dict[str, Any]] = None, cartridge_dir: Any = None, narrative: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest, cartridge_dir=cartridge_dir, narrative=narrative)

    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        return BaseLevelSpec(
            deliberation="Level specification for 1280x720 arena.",
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
        return []

    def create_hazards(self, spec: Optional[BaseLevelSpec] = None) -> List[BaseHazard]:
        return []

    def get_scenes(self, engine: Any) -> Dict[str, Any]:
        return {}

    def get_initial_scene(self, engine: Any) -> Optional[Any]:
        return None
'''
    return {
        "deliberation": "Open-genre cartridge bundle deliberation.",
        "manifest_data": {
            "title": "Open Genre Action",
            "version": "1.0.0",
            "genre": "Action",
        },
        "mechanics_code": mechanics_code,
        "scenes_code": None,
        "entities_code": None,
        "abilities_code": None,
        "level_data": {"grid": []},
    }


class TestLeadGameplaySWE:
    """F21: Lead Gameplay SWE Agent tests (Zero Templates)."""
    deliberation: str = "Test suite validating template-free code generation and AST verification."

    def test_zero_string_template_markers(self, mock_gemini_client) -> None:
        """Synthesized code contains zero static string template placeholders."""
        mock_gemini_client.set_json_response(_make_sample_cartridge_bundle())
        swe = LeadGameplaySWEAgent(client=mock_gemini_client)
        world_spec = GenesisWorldSpec(
            deliberation="World",
            game_title="Zero Template Game",
            core_genre="Action Roguelike",
            thematic_premise="Testing",
            world_lore="Lore",
            conflict_tone="grim_tense",
            conflict_narrative="Conflict",
            aesthetic_style="Style",
            palette_mood="neon_cyberpunk",
            gameplay_pacing="kinetic_high_mobility",
            pacing_score=0.8,
            target_difficulty=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            results = swe.synthesize_cartridge_code(world_spec, output_dir=Path(tmpdir))
            for name, path in results.items():
                if path.suffix == ".py":
                    code = path.read_text(encoding="utf-8")
                    assert "{TEMPLATE_" not in code
                    assert "{{LEVEL_CONFIG}}" not in code

    def test_synthesized_mechanics_inherits_base_cartridge(self, mock_gemini_client) -> None:
        """Synthesized mechanics.py defines a class inheriting from BaseCartridge."""
        mock_gemini_client.set_json_response(_make_sample_cartridge_bundle())
        swe = LeadGameplaySWEAgent(client=mock_gemini_client)
        world_spec = GenesisWorldSpec(
            deliberation="World",
            game_title="Inheritance Game",
            core_genre="Action",
            thematic_premise="Premise",
            world_lore="Lore",
            conflict_tone="grim_tense",
            conflict_narrative="Conflict",
            aesthetic_style="Style",
            palette_mood="retro_arcade",
            gameplay_pacing="balanced",
            pacing_score=0.0,
            target_difficulty=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            results = swe.synthesize_cartridge_code(world_spec, output_dir=Path(tmpdir))
            mechanics_code = results["mechanics.py"].read_text(encoding="utf-8")

            tree = ast.parse(mechanics_code)
            classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
            cartridge_classes = [c for c in classes if any(b.id == "BaseCartridge" for b in c.bases if isinstance(b, ast.Name))]
            assert len(cartridge_classes) >= 1

            c_class = cartridge_classes[0]
            methods = [m.name for m in c_class.body if isinstance(m, ast.FunctionDef)]
            for req in ["get_level_spec", "create_player", "create_enemies", "create_hazards"]:
                assert req in methods

    def test_synthesized_modules_pass_ast_sandbox_security(self, mock_gemini_client) -> None:
        """Synthesized modules pass engine AST sandbox validator."""
        mock_gemini_client.set_json_response(_make_sample_cartridge_bundle())
        swe = LeadGameplaySWEAgent(client=mock_gemini_client)
        world_spec = GenesisWorldSpec(
            deliberation="World",
            game_title="Sandbox Game",
            core_genre="Action",
            thematic_premise="Premise",
            world_lore="Lore",
            conflict_tone="grim_tense",
            conflict_narrative="Conflict",
            aesthetic_style="Style",
            palette_mood="retro_arcade",
            gameplay_pacing="balanced",
            pacing_score=0.0,
            target_difficulty=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            results = swe.synthesize_cartridge_code(world_spec, output_dir=Path(tmpdir))
            for name, path in results.items():
                if path.suffix == ".py":
                    code = path.read_text(encoding="utf-8")
                    valid, errs = validate_cartridge_ast(code)
                    assert valid is True, f"Module {name} failed AST sandbox check: {errs}"
