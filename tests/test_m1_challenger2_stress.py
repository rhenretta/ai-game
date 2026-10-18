"""tests/test_m1_challenger2_stress.py
Empirical Challenger 2 stress test suite for Milestone 1.
Validates:
1. AST Verification: deliberation strictly first attribute of every single class in engine/studio/models.py and engine/ui/models.py.
2. Cross-Genre Generalizability (Rule 1): Synthetic DNA profiles for abstract factory automation, turn-based deckbuilder, and physics simulation puzzler.
   Verify generate_requirements_spec produces valid specifications without character or arena constraints.
3. Event bus serialization: REQUIREMENTS_SPEC_GENERATED and REQUIREMENT_STATUS_UPDATE serialize with deliberation first and load into SwarmEventBus without schema degradation.
"""
from __future__ import annotations
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import threading
from typing import Any, Dict, List
from unittest.mock import MagicMock
import pytest
from pydantic import BaseModel, ValidationError

import engine.studio.models as sm
from engine.studio.models import (
    AcceptanceCriterion,
    AssetManifestSpec,
    AssetRequirement,
    BaseEpistemicModel,
    GameRequirementsSpec,
    GenesisWorldSpec,
    NarrativeBible,
    RequirementTrack,
    StudioAgentError,
)
from engine.studio.director import GameConceptDirectorAgent
from engine.steam.models import SteamPlayerDNA
import engine.ui.models as uim
from engine.ui.models import (
    RequirementsSpecPayload,
    RequirementStatusPayload,
    SwarmEventType,
    SwarmStreamEvent,
)
from engine.ui.server import SwarmEventBus


class TestASTDeliberationOrdering:
    """Stress tests AST and runtime attribute order across 100% of classes."""

    def test_ast_studio_models_all_classes_deliberation_first(self) -> None:
        """Assert that AST inspection confirms 'deliberation' is strictly the first attribute of every non-enum class in engine/studio/models.py."""
        studio_models_path = Path(sm.__file__)
        tree = ast.parse(studio_models_path.read_text(encoding="utf-8"))
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]

        assert len(classes) >= 30, f"Expected >=30 classes in studio models, found {len(classes)}"

        for c_node in classes:
            base_names = [getattr(b, "id", getattr(b, "attr", "")) for b in c_node.bases]
            if "Enum" in base_names:
                continue

            first_stmt = c_node.body[0]
            if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                # Skip docstring if present
                first_stmt = c_node.body[1] if len(c_node.body) > 1 else None

            assert first_stmt is not None, f"Class {c_node.name} has no body statements"

            attr_name = None
            if isinstance(first_stmt, ast.AnnAssign) and isinstance(first_stmt.target, ast.Name):
                attr_name = first_stmt.target.id
            elif isinstance(first_stmt, ast.Assign):
                for target in first_stmt.targets:
                    if isinstance(target, ast.Name):
                        attr_name = target.id
                        break

            assert attr_name == "deliberation", (
                f"Class {c_node.name} in engine/studio/models.py violates Rule 4: "
                f"first statement attribute is '{attr_name}', expected 'deliberation'"
            )

    def test_ast_ui_models_all_classes_deliberation_first(self) -> None:
        """Assert that AST inspection confirms 'deliberation' is strictly the first attribute of every non-enum class in engine/ui/models.py."""
        ui_models_path = Path(uim.__file__)
        tree = ast.parse(ui_models_path.read_text(encoding="utf-8"))
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]

        assert len(classes) >= 15, f"Expected >=15 classes in UI models, found {len(classes)}"

        for c_node in classes:
            base_names = [getattr(b, "id", getattr(b, "attr", "")) for b in c_node.bases]
            if "Enum" in base_names:
                continue

            first_stmt = c_node.body[0]
            if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                first_stmt = c_node.body[1] if len(c_node.body) > 1 else None

            assert first_stmt is not None, f"Class {c_node.name} has no body statements"

            attr_name = None
            if isinstance(first_stmt, ast.AnnAssign) and isinstance(first_stmt.target, ast.Name):
                attr_name = first_stmt.target.id
            elif isinstance(first_stmt, ast.Assign):
                for target in first_stmt.targets:
                    if isinstance(target, ast.Name):
                        attr_name = target.id
                        break

            assert attr_name == "deliberation", (
                f"Class {c_node.name} in engine/ui/models.py violates Rule 4: "
                f"first statement attribute is '{attr_name}', expected 'deliberation'"
            )

    def test_runtime_model_fields_index_zero_is_deliberation(self) -> None:
        """Verifies that for every Pydantic model in studio and UI models, model_fields[0] == 'deliberation'."""
        all_models: List[type[BaseModel]] = []
        for mod in (sm, uim):
            for name, obj in inspect.getmembers(mod):
                if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj not in (BaseModel, BaseEpistemicModel):
                    all_models.append(obj)

        assert len(all_models) >= 40, f"Expected >=40 Pydantic models across studio and ui, got {len(all_models)}"

        for model_cls in all_models:
            field_names = list(model_cls.model_fields.keys())
            assert len(field_names) > 0, f"Model {model_cls.__name__} has no fields"
            assert field_names[0] == "deliberation", (
                f"Model {model_cls.__name__} has '{field_names[0]}' at index 0 instead of 'deliberation'"
            )


class TestCrossGenreGeneralizability:
    """Stress tests Rule 1: Pure Generalizability across diverse genres with zero hardcoded character/arena constraints."""

    @pytest.fixture
    def factory_automation_dna(self) -> SteamPlayerDNA:
        """Synthetic DNA for an abstract factory automation enthusiast."""
        return SteamPlayerDNA(
            deliberation="Player is obsessed with throughput optimization, conveyor balancing, and zero-narrative systems.",
            player_id="factory_architect_99",
            profile_tag="hardcore_simulation",
            total_playtime_hours=1530.0,
            total_games_analyzed=42,
            top_genres=["Simulation", "Automation", "Base Building", "Management", "Logistics"],
            genre_weights={"Simulation": 0.45, "Automation": 0.35, "Strategy": 0.20},
            top_games=[
                {"name": "Factorio", "playtime_hours": 850.0, "genres": ["Automation", "Simulation"]},
                {"name": "Satisfactory", "playtime_hours": 420.0, "genres": ["Automation", "Open World"]},
                {"name": "Dyson Sphere Program", "playtime_hours": 260.0, "genres": ["Automation", "Sci-Fi"]},
            ],
            thematic_biases=["Industrial", "Mechanical", "Optimization", "Deterministic"],
            preferred_palette_mood="retro_arcade",
            pacing_preference="tactical_methodical",
            pacing_score=-0.85,
            challenge_tolerance=0.75,
        )

    @pytest.fixture
    def turn_based_deckbuilder_dna(self) -> SteamPlayerDNA:
        """Synthetic DNA for a tactical roguelike deckbuilder specialist."""
        return SteamPlayerDNA(
            deliberation="Player focuses purely on card synergies, hand math, relic compounding, and strategic probability.",
            player_id="deckbuilder_master_01",
            profile_tag="tactical_card_strategist",
            total_playtime_hours=1150.0,
            total_games_analyzed=28,
            top_genres=["Card Game", "Roguelike Deckbuilder", "Turn-Based Strategy"],
            genre_weights={"Card Game": 0.50, "Roguelike": 0.30, "Strategy": 0.20},
            top_games=[
                {"name": "Slay the Spire", "playtime_hours": 620.0, "genres": ["Roguelike Deckbuilder"]},
                {"name": "Balatro", "playtime_hours": 310.0, "genres": ["Card Game", "Roguelike"]},
                {"name": "Monster Train", "playtime_hours": 220.0, "genres": ["Strategy", "Card Game"]},
            ],
            thematic_biases=["Tactical", "Synergy", "Card Manipulation", "Probability"],
            preferred_palette_mood="grim_dark",
            pacing_preference="tactical_methodical",
            pacing_score=-0.60,
            challenge_tolerance=0.85,
        )

    @pytest.fixture
    def physics_puzzler_dna(self) -> SteamPlayerDNA:
        """Synthetic DNA for a zero-combat physics simulation puzzler player."""
        return SteamPlayerDNA(
            deliberation="Player seeks emergent physical puzzles, momentum conservation, and mechanical logic without enemies.",
            player_id="physics_solver_42",
            profile_tag="abstract_puzzle_solver",
            total_playtime_hours=570.0,
            total_games_analyzed=35,
            top_genres=["Puzzle", "Physics Simulation", "Abstract", "Logic"],
            genre_weights={"Puzzle": 0.55, "Physics": 0.30, "Simulation": 0.15},
            top_games=[
                {"name": "Baba Is You", "playtime_hours": 180.0, "genres": ["Puzzle", "Logic"]},
                {"name": "Besiege", "playtime_hours": 240.0, "genres": ["Physics", "Sandbox"]},
                {"name": "Portal 2", "playtime_hours": 150.0, "genres": ["Puzzle", "Physics"]},
            ],
            thematic_biases=["Mechanical Logic", "Gravitational Physics", "Spatial Constraints"],
            preferred_palette_mood="vibrant_fantasy",
            pacing_preference="tactical_methodical",
            pacing_score=-0.40,
            challenge_tolerance=0.60,
        )

    def test_director_fails_fast_without_client_rule5(self, factory_automation_dna: SteamPlayerDNA) -> None:
        """Rule 5: generate_requirements_spec fails fast with StudioAgentError when client is None."""
        director = GameConceptDirectorAgent(client=None)
        with pytest.raises(StudioAgentError) as exc_info:
            director.generate_requirements_spec(dna=factory_automation_dna)
        assert "Rule 5" in str(exc_info.value) or "Client" in str(exc_info.value)

    def test_factory_automation_generates_valid_spec_without_characters_or_arena(
        self,
        factory_automation_dna: SteamPlayerDNA,
    ) -> None:
        """Verifies an abstract factory automation game generates a valid spec with zero character or arena constraints."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "deliberation": "Player DNA demonstrates deep dedication to deterministic industrial automation. Conceived 'Automata Grid': an isometric logistics conveyor simulator with zero characters, pure systems focus, and discrete throughput metrics.",
            "spec_id": "req_factory_001",
            "game_title": "Automata Grid",
            "genre_paradigm": "Automation Simulation",
            "target_audience_summary": "Players who love logistics, conveyor balancing, and deterministic supply chain optimization.",
            "high_concept_summary": "Top-down 2D grid where miners extract raw materials and conveyor networks route them to assemblers.",
            "target_episode": 0,
            "requirements": [
                {
                    "deliberation": "Core 2D grid matrix and directional conveyor routing state machine.",
                    "id": "R1",
                    "title": "Conveyor Grid Engine & Item Transport Matrix",
                    "description": "Implements discrete cell occupancy grid and directional conveyor transfers without entity physics.",
                    "responsible_agents": ["LeadGameplaySWEAgent"],
                    "target_modules": ["mechanics.py", "grid.py"],
                    "acceptance_criteria": [
                        {
                            "deliberation": "Verify items translate between adjacent conveyor cells at constant 1 cell/second rate.",
                            "id": "AC1.1",
                            "description": "Item transit across 10-cell conveyor line completes within 10.0s +/- 0.1s in headless simulation.",
                            "verification_method": "headless_playtest",
                        }
                    ],
                    "status": "pending",
                },
                {
                    "deliberation": "Multi-tier assembling machine consuming inputs and emitting crafted products.",
                    "id": "R2",
                    "title": "Assembler Recipes & Sink Logic",
                    "description": "Implements deterministic recipe conversion timers and buffer overflow prevention.",
                    "responsible_agents": ["LeadGameplaySWEAgent"],
                    "target_modules": ["mechanics.py", "recipes.py"],
                    "acceptance_criteria": [
                        {
                            "deliberation": "Verify recipe cycle completes and produces exact ratio.",
                            "id": "AC2.1",
                            "description": "Feeding 2 iron plates produces 1 gear every 2.0s without item loss.",
                            "verification_method": "runtime_telemetry",
                        }
                    ],
                    "status": "pending",
                },
            ],
            "asset_manifest": [
                {
                    "deliberation": "Directional conveyor belt straight tile (32x32).",
                    "asset_id": "conveyor_straight",
                    "asset_type": "tile",
                    "target_path": "assets/sprites/conveyor_straight.png",
                    "visual_prompt_or_text": "32x32 pixel industrial yellow directional conveyor belt",
                },
                {
                    "deliberation": "Iron plate intermediate resource item icon (16x16).",
                    "asset_id": "item_iron_plate",
                    "asset_type": "sprite",
                    "target_path": "assets/sprites/item_iron_plate.png",
                    "visual_prompt_or_text": "16x16 metallic iron sheet icon",
                },
            ],
        })
        mock_client.models.generate_content.return_value = mock_response

        director = GameConceptDirectorAgent(client=mock_client)
        spec = director.generate_requirements_spec(dna=factory_automation_dna)

        # 1. Verify schema and deliberation
        assert isinstance(spec, GameRequirementsSpec)
        assert spec.deliberation.startswith("Player DNA demonstrates")
        assert list(spec.model_dump().keys())[0] == "deliberation"

        # 2. Verify genre and features
        assert spec.game_title == "Automata Grid"
        assert spec.genre_paradigm == "Automation Simulation"
        assert len(spec.requirements) == 2
        assert len(spec.asset_manifest) == 2

        # 3. Verify zero character constraints: NarrativeBible can be empty
        bible = NarrativeBible(
            deliberation="Abstract automation simulation requires zero narrative characters or voice barks.",
            world_title="Automata Grid",
            setting_summary="Silent planetary factory outpost.",
            characters=[],
            dialogue_barks=[],
        )
        assert len(bible.characters) == 0
        narrative_export = bible.to_cartridge_narrative_dict()
        assert len(narrative_export["characters"]) == 0
        assert narrative_export["deliberation"].startswith("Abstract automation")

        # 4. Verify GenesisWorldSpec without arena constraints
        world_spec = GenesisWorldSpec(
            deliberation="World spec for infinite factory grid with agent-directed camera.",
            game_title="Automata Grid",
            core_genre="Automation Simulation",
            thematic_premise="Planetary resource exploitation",
            world_lore="Autonomous Von Neumann probe setting up orbital manufacturing.",
            conflict_tone="cozy_exploratory",
            conflict_narrative="Overcoming logistics bottlenecks and entropy.",
            aesthetic_style="Crisp industrial pixel art with visible gear mechanisms",
            palette_mood="retro_arcade",
            gameplay_pacing="tactical_methodical",
            pacing_score=-0.85,
            target_difficulty=0.9,
            camera_style="static_view",
            arena_layout_directive="procedural_grid",
        )
        assert world_spec.camera_style == "static_view"
        assert world_spec.arena_layout_directive == "procedural_grid"

    def test_turn_based_deckbuilder_generates_valid_spec(
        self,
        turn_based_deckbuilder_dna: SteamPlayerDNA,
    ) -> None:
        """Verifies turn-based deckbuilder produces valid requirements without movement/arena assumptions."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "deliberation": "Player DNA prefers synergistic turn-based deck manipulation. Conceived 'Spire of Runes': a lane-based card battle system with draw/discard mechanics and artifact passive buffs.",
            "spec_id": "req_deck_002",
            "game_title": "Spire of Runes",
            "genre_paradigm": "Roguelike Deckbuilder",
            "target_audience_summary": "Players who enjoy calculating synergy chains, drafting card pools, and mitigating RNG variance.",
            "high_concept_summary": "Turn-based card battle game where player manages energy to cast attacks, shields, and persistent enchantments.",
            "target_episode": 0,
            "requirements": [
                {
                    "deliberation": "Card draw pile, hand limit, and turn cycle state machine.",
                    "id": "R1",
                    "title": "Deck Engine & Turn Cycle Manager",
                    "description": "Manages draw pile, hand array, discard pile, and turn-end discard phase.",
                    "responsible_agents": ["LeadGameplaySWEAgent"],
                    "target_modules": ["mechanics.py", "deck.py"],
                    "acceptance_criteria": [
                        {
                            "deliberation": "Verify turn start draws 5 cards from draw pile.",
                            "id": "AC1.1",
                            "description": "Turn start event fills hand to 5 cards; reshuffles discard pile when draw pile is empty.",
                            "verification_method": "runtime_telemetry",
                        }
                    ],
                    "status": "pending",
                }
            ],
            "asset_manifest": [
                {
                    "deliberation": "Card frame background sprite (64x96).",
                    "asset_id": "card_frame_attack",
                    "asset_type": "sprite",
                    "target_path": "assets/sprites/card_frame_attack.png",
                    "visual_prompt_or_text": "64x96 ornate stone card frame with ruby rune inset",
                }
            ],
        })
        mock_client.models.generate_content.return_value = mock_response

        director = GameConceptDirectorAgent(client=mock_client)
        spec = director.generate_requirements_spec(dna=turn_based_deckbuilder_dna)

        assert isinstance(spec, GameRequirementsSpec)
        assert spec.genre_paradigm == "Roguelike Deckbuilder"
        assert len(spec.requirements) == 1
        assert spec.requirements[0].id == "R1"
        assert len(spec.asset_manifest) == 1

        # Check prompt captured from client call respects Rule 1
        called_args, called_kwargs = mock_client.models.generate_content.call_args
        prompt_text = called_kwargs["contents"]
        assert "PURE GENERALIZABILITY MANDATE (Rule 1)" in prompt_text
        assert "Zero hardcoded genre, mechanic, character count, or arena prescriptions" in prompt_text

    def test_physics_puzzler_generates_valid_spec_zero_combat(
        self,
        physics_puzzler_dna: SteamPlayerDNA,
    ) -> None:
        """Verifies physics puzzler produces valid requirements with zero combat, enemies, or hazard barks."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "deliberation": "Conceived 'Prismatica': an optic laser refraction puzzle game with zero enemies, zero combat, and continuous raycast reflection logic.",
            "spec_id": "req_puzzle_003",
            "game_title": "Prismatica",
            "genre_paradigm": "Physics Simulation Puzzler",
            "target_audience_summary": "Players who love optical physics, laser alignment, and eureka moments without timing pressure.",
            "high_concept_summary": "Rotate optical prisms to redirect laser beams onto sensor receptors to open the chamber portal.",
            "target_episode": 0,
            "requirements": [
                {
                    "deliberation": "Laser raycast propagation and mirror reflection calculus.",
                    "id": "R1",
                    "title": "Optic Beam Propagation & Reflection Solver",
                    "description": "Calculates recursive 2D ray segments reflected by mirrored prism surfaces.",
                    "responsible_agents": ["LeadGameplaySWEAgent"],
                    "target_modules": ["mechanics.py", "optics.py"],
                    "acceptance_criteria": [
                        {
                            "deliberation": "Verify laser beam reflects at angle equal to angle of incidence.",
                            "id": "AC1.1",
                            "description": "A 45-degree ray hitting a vertical mirror emerges at 135 degrees within 0.01 deg tolerance.",
                            "verification_method": "headless_playtest",
                        }
                    ],
                    "status": "pending",
                }
            ],
            "asset_manifest": [
                {
                    "deliberation": "Glass prism rotator sprite (32x32).",
                    "asset_id": "prism_rotator",
                    "asset_type": "sprite",
                    "target_path": "assets/sprites/prism_rotator.png",
                    "visual_prompt_or_text": "32x32 triangular glass prism reflecting light",
                }
            ],
        })
        mock_client.models.generate_content.return_value = mock_response

        director = GameConceptDirectorAgent(client=mock_client)
        spec = director.generate_requirements_spec(dna=physics_puzzler_dna)

        assert isinstance(spec, GameRequirementsSpec)
        assert spec.game_title == "Prismatica"
        assert spec.genre_paradigm == "Physics Simulation Puzzler"
        assert len(spec.requirements) == 1
        assert spec.requirements[0].acceptance_criteria[0].verification_method == "headless_playtest"


class TestEventBusSerialization:
    """Stress tests event bus serialization with deliberation first and schema preservation."""

    def test_requirements_spec_generated_event_serialization_and_bus_ingestion(self, tmp_path: Path) -> None:
        """Verify REQUIREMENTS_SPEC_GENERATED serializes with deliberation strictly first and loads into SwarmEventBus."""
        log_file = tmp_path / "studio_events.jsonl"
        bus = SwarmEventBus(log_path=log_file)

        spec = GameRequirementsSpec(
            deliberation="Requirements specification deliberated from Steam player DNA profile.",
            spec_id="req_test_99",
            game_title="Quantum Conveyor",
            genre_paradigm="Factory Simulator",
            target_audience_summary="Automation lovers",
            requirements=[
                RequirementTrack(
                    deliberation="Conveyor grid system architecture.",
                    id="R1",
                    title="Grid Logistics",
                    description="2D transport grid.",
                    responsible_agents=["LeadGameplaySWEAgent"],
                    acceptance_criteria=[
                        AcceptanceCriterion(
                            deliberation="Verify item transfer throughput.",
                            id="AC1.1",
                            description="Transfer 100 items without blockage.",
                            verification_method="runtime_telemetry",
                        )
                    ],
                )
            ],
            asset_manifest=[
                AssetRequirement(
                    deliberation="Conveyor straight belt sprite.",
                    asset_id="belt_straight",
                    asset_type="sprite",
                    target_path="assets/sprites/belt.png",
                    visual_prompt_or_text="32x32 yellow belt",
                )
            ],
        )

        # 1. Payload validation
        payload = RequirementsSpecPayload(
            deliberation="Event payload wrapping generated requirements spec.",
            spec=spec.model_dump(),
            spec_id=spec.spec_id,
            game_title=spec.game_title,
            total_requirements=len(spec.requirements),
        )
        assert list(payload.model_dump().keys())[0] == "deliberation"
        assert list(json.loads(payload.model_dump_json()).keys())[0] == "deliberation"

        # 2. Event serialization
        event_dict = {
            "deliberation": "Broadcasting newly authored requirements spec to swarm subscribers.",
            "event_type": SwarmEventType.REQUIREMENTS_SPEC_GENERATED.value,
            "agent_name": "GameConceptDirectorAgent",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "payload": payload.model_dump(),
        }
        json_str = json.dumps(event_dict)
        assert json_str.startswith('{"deliberation":'), f"JSON must start with 'deliberation', got: {json_str[:30]}"

        # 3. Bus ingestion
        bus.publish_threadsafe(event_dict)
        events = bus.get_recent_events()
        assert len(events) == 1
        ingested = events[0]
        assert list(ingested.keys())[0] == "deliberation"
        assert ingested["event_type"] == "REQUIREMENTS_SPEC_GENERATED"
        assert ingested["payload"]["spec_id"] == "req_test_99"

        # 4. SwarmStreamEvent schema validation
        stream_event = SwarmStreamEvent(**ingested)
        assert stream_event.deliberation == event_dict["deliberation"]
        assert stream_event.event_type == "REQUIREMENTS_SPEC_GENERATED"

        # 5. Disk persistence verification (if written)
        if log_file.exists():
            lines = log_file.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) >= 1
            disk_event = json.loads(lines[-1])
            assert list(disk_event.keys())[0] == "deliberation"
            assert disk_event["event_type"] == "REQUIREMENTS_SPEC_GENERATED"

    def test_requirement_status_update_event_serialization_and_bus_ingestion(self, tmp_path: Path) -> None:
        """Verify REQUIREMENT_STATUS_UPDATE serializes with deliberation first and loads into SwarmEventBus."""
        log_file = tmp_path / "studio_events.jsonl"
        bus = SwarmEventBus(log_path=log_file)

        status_payload = RequirementStatusPayload(
            deliberation="Requirement track R1 verified through 60-tick headless smoke test.",
            requirement_id="R1",
            status="completed",
            metrics={"compilation_pass": True, "frame_rate": 60.0, "exceptions": 0},
            track_title="Grid Logistics",
        )
        assert list(status_payload.model_dump().keys())[0] == "deliberation"
        assert list(json.loads(status_payload.model_dump_json()).keys())[0] == "deliberation"

        event_dict = {
            "deliberation": "Live streaming requirement R1 completion status to UI.",
            "event_type": SwarmEventType.REQUIREMENT_STATUS_UPDATE.value,
            "agent_name": "LeadGameplaySWEAgent",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "payload": status_payload.model_dump(),
        }
        json_str = json.dumps(event_dict)
        assert json_str.startswith('{"deliberation":')

        bus.publish_threadsafe(event_dict)
        recent = bus.get_recent_events()
        assert len(recent) == 1
        ingested = recent[0]
        assert list(ingested.keys())[0] == "deliberation"
        assert ingested["payload"]["requirement_id"] == "R1"
        assert ingested["payload"]["status"] == "completed"

        stream_event = SwarmStreamEvent(**ingested)
        assert stream_event.deliberation == event_dict["deliberation"]

    def test_concurrent_event_bus_publishing_stress(self, tmp_path: Path) -> None:
        """Stress-test SwarmEventBus with concurrent worker threads publishing requirements events in rapid bursts."""
        log_file = tmp_path / "concurrent_studio_events.jsonl"
        bus = SwarmEventBus(history_size=500, log_path=log_file)

        worker_count = 10
        events_per_worker = 25
        total_expected = worker_count * events_per_worker

        def worker_task(worker_id: int):
            for i in range(events_per_worker):
                req_id = f"R{worker_id}_{i}"
                event = {
                    "deliberation": f"Worker {worker_id} requirement {req_id} progress update.",
                    "event_type": SwarmEventType.REQUIREMENT_STATUS_UPDATE.value,
                    "agent_name": f"SWEWorker_{worker_id}",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "payload": {
                        "deliberation": f"Telemetry verified for {req_id}.",
                        "requirement_id": req_id,
                        "status": "in_progress" if i < 15 else "completed",
                        "metrics": {"batch": i, "worker": worker_id},
                    },
                }
                bus.publish_threadsafe(event)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(worker_task, wid) for wid in range(worker_count)]
            for f in futures:
                f.result()

        recent = bus.get_recent_events()
        assert len(recent) == min(total_expected, 500)

        # Verify every recorded event has deliberation strictly first
        for ev in recent:
            assert list(ev.keys())[0] == "deliberation"
            assert list(ev["payload"].keys())[0] == "deliberation"
            assert ev["event_type"] == "REQUIREMENT_STATUS_UPDATE"

        # Verify all events can be deserialized into SwarmStreamEvent
        for ev in recent:
            parsed = SwarmStreamEvent(**ev)
            assert parsed.deliberation.startswith("Worker ")
