"""tests/test_m3_iterative_loop.py
Milestone 3 Dedicated Test Suite for Autonomous Game Studio SWE Iterative Loop & Multi-Module Assembly.

Verifies:
1. SwarmEventBus requirements spec and status update helper methods (Rule 4: deliberation first).
2. OpenGenreCartridgeBundle.assemble_cartridge_directory modular output.
3. LeadGameplaySWEAgent.verify_cartridge_cross_module_imports (passing valid imports, rejecting broken imports).
4. LeadGameplaySWEAgent._discover_sealed_assets discovery of sealed visual and audio staging assets.
5. StudioOrchestrator._execute_iterative_swe_loop status progression (pending -> in_progress -> completed).
6. StudioOrchestrator._execute_iterative_swe_loop immediate fail-fast halting on AST / smoke verification failure.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock
import pytest

from engine.api.contracts import BaseCartridge, BaseScene
from engine.studio.models import (
    AcceptanceCriterion,
    GameRequirementsSpec,
    GenesisWorldSpec,
    OpenGenreCartridgeBundle,
    RequirementTrack,
    StudioSwarmExecutionError,
    StudioSwarmPlan,
)
from engine.studio.orchestrator import StudioOrchestrator
from engine.studio.swe import CodeVerificationError, LeadGameplaySWEAgent
from engine.ui.models import SwarmEventType
from engine.ui.server import SwarmEventBus
from tests.conftest import MockGeminiClient


def _make_dummy_req_spec() -> GameRequirementsSpec:
    return GameRequirementsSpec(
        deliberation="Requirements specification for testing iterative execution loop.",
        game_title="Neon Grid Survival",
        genre_paradigm="Action Roguelike",
        target_audience_summary="Pacing index 0.5 fast kinetic gameplay",
        target_episode=0,
        requirements=[
            RequirementTrack(
                deliberation="Track 1: Player character and core simulation lifecycle",
                id="R1",
                title="Player Entity Simulation",
                description="Simulates player coordinates and tick counter.",
                responsible_agents=["LeadGameplaySWEAgent"],
                acceptance_criteria=[
                    AcceptanceCriterion(
                        deliberation="AC1.1: Increments simulation ticks headlessly",
                        id="AC1.1",
                        description="Executes 10 headless steps without throwing exceptions.",
                        verification_method="headless_playtest",
                    )
                ],
                status="pending",
            ),
            RequirementTrack(
                deliberation="Track 2: Directional projectile combat system",
                id="R2",
                title="Combat Projectiles",
                description="Synthesizes combat projectile entities and physics.",
                responsible_agents=["LeadGameplaySWEAgent"],
                acceptance_criteria=[
                    AcceptanceCriterion(
                        deliberation="AC2.1: AST check for combat loop",
                        id="AC2.1",
                        description="Passes AST static check and does not import banned modules.",
                        verification_method="ast_inspection",
                    )
                ],
                status="pending",
            ),
        ],
        asset_manifest=[],
    )


def test_swarm_event_bus_requirements_spec_and_status_helpers(tmp_path: Path) -> None:
    """Verifies that SwarmEventBus helper methods emit records conforming to Rule 4 (deliberation first)."""
    log_file = tmp_path / "test_swarm_events.jsonl"
    bus = SwarmEventBus(log_path=log_file)

    # 1. Test publish_requirements_spec_generated
    bus.publish_requirements_spec_generated(
        agent_name="CreativeDirectorWorldbuilderAgent",
        deliberation="Decomposing player DNA into 2 requirement tracks.",
        spec={"spec_id": "spec_123", "game_title": "Neon Grid Survival"},
        spec_id="spec_123",
        game_title="Neon Grid Survival",
        total_requirements=2,
    )

    # 2. Test publish_requirement_status_update
    bus.publish_requirement_status_update(
        agent_name="LeadGameplaySWEAgent",
        deliberation="Track R1 passed headless smoke check.",
        requirement_id="R1",
        status="completed",
        metrics={"compilation_pass": True},
        track_title="Player Entity Simulation",
    )

    # Inspect persistent JSONL records
    records = bus.get_recent_events()
    assert len(records) >= 2

    # Verify event types and deliberation priority
    ev_types = [r.get("event_type") for r in records]
    assert "REQUIREMENTS_SPEC_GENERATED" in ev_types
    assert "REQUIREMENT_STATUS_UPDATE" in ev_types

    req_update_rec = next(r for r in records if r.get("event_type") == "REQUIREMENT_STATUS_UPDATE")
    assert req_update_rec["deliberation"] == "Track R1 passed headless smoke check."
    assert req_update_rec["payload"]["requirement_id"] == "R1"
    assert req_update_rec["payload"]["status"] == "completed"


def test_open_genre_cartridge_bundle_assemble_directory(tmp_path: Path) -> None:
    """Verifies that OpenGenreCartridgeBundle.assemble_cartridge_directory decomposes modular code."""
    dest_dir = tmp_path / "modular_cartridge"

    bundle = OpenGenreCartridgeBundle(
        deliberation="Decomposed modular code bundle.",
        manifest_data={
            "deliberation": "Manifest for modular cartridge",
            "name": "Modular Cyber Run",
            "version": "1.0.0",
            "entrypoint": "mechanics.py",
        },
        entities_code=(
            '"""entities.py"""\n'
            'class PlayerEntity:\n'
            '    deliberation: str = "Player avatar entity."\n'
            '    def __init__(self, x=100.0, y=100.0):\n'
            '        self.x = x\n'
            '        self.y = y\n'
        ),
        scenes_code=(
            '"""scenes.py"""\n'
            'from engine.api.contracts import BaseScene\n'
            'from entities import PlayerEntity\n'
            'class CyberScene(BaseScene):\n'
            '    deliberation: str = "Primary cyber world scene."\n'
            '    def __init__(self, engine=None):\n'
            '        super().__init__(engine)\n'
            '        self.player = PlayerEntity()\n'
            '        self.state_counter = 0\n'
            '    def update(self, dt):\n'
            '        self.state_counter += 1\n'
            '    def render(self, surface):\n'
            '        pass\n'
        ),
        mechanics_code=(
            '"""mechanics.py"""\n'
            'from engine.api.contracts import BaseCartridge\n'
            'from scenes import CyberScene\n'
            'class CyberCartridge(BaseCartridge):\n'
            '    deliberation: str = "Entrypoint cartridge."\n'
            '    def get_scenes(self, engine):\n'
            '        return {"main": CyberScene(engine)}\n'
            '    def get_initial_scene(self, engine):\n'
            '        return self.get_scenes(engine)["main"]\n'
        ),
        level_data={
            "deliberation": "Level layout for Sector 1",
            "level_id": "sector_01",
            "width": 1280.0,
            "height": 720.0,
        },
    )

    written_files = bundle.assemble_cartridge_directory(dest_dir)
    assert (dest_dir / "manifest.json").is_file()
    assert (dest_dir / "entities.py").is_file()
    assert (dest_dir / "scenes.py").is_file()
    assert (dest_dir / "mechanics.py").is_file()
    assert (dest_dir / "level.json").is_file()

    assert "mechanics.py" in written_files
    assert "entities.py" in written_files
    assert "scenes.py" in written_files
    assert "level.json" in written_files


def test_cross_module_import_verification_passes_and_fails(tmp_path: Path) -> None:
    """Verifies that LeadGameplaySWEAgent.verify_cartridge_cross_module_imports detects missing symbols."""
    swe = LeadGameplaySWEAgent(client=MagicMock())
    test_dir = tmp_path / "import_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # 1. Valid cross-module setup
    (test_dir / "entities.py").write_text(
        'class Drone:\n    deliberation: str = "Autonomous drone."\n',
        encoding="utf-8",
    )
    (test_dir / "scenes.py").write_text(
        'from entities import Drone\nclass BattleScene:\n    deliberation: str = "Battle."\n',
        encoding="utf-8",
    )
    (test_dir / "mechanics.py").write_text(
        'from scenes import BattleScene\nclass Cartridge:\n    deliberation: str = "Cartridge."\n',
        encoding="utf-8",
    )

    passed, errors = swe.verify_cartridge_cross_module_imports(test_dir)
    assert passed is True
    assert len(errors) == 0

    # 2. Introduce broken import: import NonExistentEntity from entities
    (test_dir / "scenes.py").write_text(
        'from entities import NonExistentEntity\nclass BattleScene:\n    deliberation: str = "Battle."\n',
        encoding="utf-8",
    )
    # Test non-raising mode returns (False, errors)
    passed_fail, errors_fail = swe.verify_cartridge_cross_module_imports(test_dir, raise_on_error=False)
    assert passed_fail is False
    assert any("NonExistentEntity" in err for err in errors_fail)

    # Test raising mode raises CodeVerificationError
    with pytest.raises(CodeVerificationError) as exc_info:
        swe.verify_cartridge_cross_module_imports(test_dir, raise_on_error=True)
    assert "NonExistentEntity" in str(exc_info.value)


def test_swe_agent_sealed_asset_discovery(tmp_path: Path) -> None:
    """Verifies that LeadGameplaySWEAgent._discover_sealed_assets catalogs sprite and audio assets."""
    swe = LeadGameplaySWEAgent(client=MagicMock())
    staging_dir = tmp_path / "staging_assets"
    sprites_dir = staging_dir / "assets" / "sprites"
    audio_dir = staging_dir / "assets" / "audio"
    sprites_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    (sprites_dir / "hero.png").write_bytes(b"\x89PNG\r\n\x1a\nfake_png")
    (sprites_dir / "boss.png").write_bytes(b"\x89PNG\r\n\x1a\nfake_png")
    (audio_dir / "bark_start.wav").write_bytes(b"RIFFfake_wav")

    assets = swe._discover_sealed_assets(staging_dir)
    assert "assets/sprites/hero.png" in assets["sprites"]
    assert "assets/sprites/boss.png" in assets["sprites"]
    assert "assets/audio/bark_start.wav" in assets["audio"]


@pytest.mark.anyio
async def test_orchestrator_iterative_swe_loop_lifecycle(tmp_path: Path) -> None:
    """Verifies status progression from pending -> in_progress -> completed across tracks."""
    client = MockGeminiClient()
    cartridge_bundle_data_r1 = {
        "deliberation": "Formulating complete open-genre cartridge code bundle implementing BaseCartridge for R1.",
        "manifest_data": {
            "title": "Neon Grid Survival",
            "genre": "Action Roguelike",
            "version": "1.0.0",
            "entrypoint": "mechanics.py",
        },
        "mechanics_code": (
            '"""mechanics.py"""\n'
            'from engine.api.contracts import BaseCartridge, BaseScene\n'
            'class GameScene(BaseScene):\n'
            '    deliberation: str = "Lifecycle scene."\n'
            '    def __init__(self, engine=None):\n'
            '        super().__init__(engine)\n'
            '        self.counter = 0\n'
            '    def update(self, dt):\n'
            '        self.counter += 1\n'
            '    def render(self, surface):\n'
            '        pass\n'
            'class GridCartridge(BaseCartridge):\n'
            '    deliberation: str = "Cartridge."\n'
            '    def get_scenes(self, engine):\n'
            '        return {"main": GameScene(engine)}\n'
            '    def get_initial_scene(self, engine):\n'
            '        return self.get_scenes(engine)["main"]\n'
        ),
        "scenes_code": None,
        "entities_code": None,
        "abilities_code": None,
        "level_data": None,
    }
    cartridge_bundle_data_r2 = {
        "deliberation": "Expanding cartridge bundle with combat projectiles implementation for R2.",
        "manifest_data": {
            "title": "Neon Grid Survival",
            "genre": "Action Roguelike",
            "version": "1.0.0",
            "entrypoint": "mechanics.py",
        },
        "mechanics_code": (
            '"""mechanics.py"""\n'
            'from engine.api.contracts import BaseCartridge, BaseScene\n'
            'class GameScene(BaseScene):\n'
            '    deliberation: str = "Lifecycle scene with projectile combat."\n'
            '    def __init__(self, engine=None):\n'
            '        super().__init__(engine)\n'
            '        self.counter = 0\n'
            '        self.projectiles = []\n'
            '    def update(self, dt):\n'
            '        self.counter += 1\n'
            '    def render(self, surface):\n'
            '        pass\n'
            'class GridCartridge(BaseCartridge):\n'
            '    deliberation: str = "Cartridge."\n'
            '    def get_scenes(self, engine):\n'
            '        return {"main": GameScene(engine)}\n'
            '    def get_initial_scene(self, engine):\n'
            '        return self.get_scenes(engine)["main"]\n'
        ),
        "scenes_code": None,
        "entities_code": None,
        "abilities_code": None,
        "level_data": None,
    }
    client.set_json_response(cartridge_bundle_data_r1)
    client.set_json_response(cartridge_bundle_data_r2)
    orch = StudioOrchestrator(client=client)

    staging_dir = tmp_path / "staging_iterative_loop"
    staging_dir.mkdir(parents=True, exist_ok=True)

    req_spec = _make_dummy_req_spec()
    (staging_dir / "requirements_spec.json").write_text(
        req_spec.model_dump_json(indent=2), encoding="utf-8"
    )

    swe = orch.agents["LeadGameplaySWEAgent"]
    world_spec = GenesisWorldSpec(
        deliberation="World spec for lifecycle testing",
        game_title="Neon Grid Survival",
        core_genre="Action Roguelike",
        thematic_premise="Cyber survival",
        world_lore="Grid world lore",
        conflict_tone="tense",
        conflict_narrative="survive against defensive countermeasures",
        aesthetic_style="retro_arcade",
        palette_mood="retro_arcade",
        gameplay_pacing="fast",
        pacing_score=0.5,
        target_difficulty=1.0,
    )

    results = await orch._execute_iterative_swe_loop(
        agent=swe,
        world_spec=world_spec,
        narrative_bible=None,
        req_spec=req_spec,
        staging_dir=staging_dir,
    )

    assert len(results) >= 1
    # Verify both tracks are marked completed
    assert all(t.status == "completed" for t in req_spec.requirements)

    # Verify disk persistence
    saved_spec = GameRequirementsSpec.model_validate_json(
        (staging_dir / "requirements_spec.json").read_text(encoding="utf-8")
    )
    assert all(t.status == "completed" for t in saved_spec.requirements)


@pytest.mark.anyio
async def test_orchestrator_iterative_swe_loop_halts_on_smoke_failure(tmp_path: Path) -> None:
    """Verifies that an AST or smoke validation failure halts the loop immediately with status 'rejected'."""
    client = MockGeminiClient()
    orch = StudioOrchestrator(client=client)

    staging_dir = tmp_path / "staging_broken_track"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Write broken syntax into mechanics.py
    (staging_dir / "mechanics.py").write_text(
        'def broken_syntax(:\n    pass\n',
        encoding="utf-8",
    )

    req_spec = _make_dummy_req_spec()
    (staging_dir / "requirements_spec.json").write_text(
        req_spec.model_dump_json(indent=2), encoding="utf-8"
    )

    swe = orch.agents["LeadGameplaySWEAgent"]
    world_spec = GenesisWorldSpec(
        deliberation="World spec for failure testing",
        game_title="Neon Grid Survival",
        core_genre="Action Roguelike",
        thematic_premise="Cyber survival",
        world_lore="Grid world lore",
        conflict_tone="tense",
        conflict_narrative="survive against defensive countermeasures",
        aesthetic_style="retro_arcade",
        palette_mood="retro_arcade",
        gameplay_pacing="fast",
        pacing_score=0.5,
        target_difficulty=1.0,
    )

    with pytest.raises(Exception):
        await orch._execute_iterative_swe_loop(
            agent=swe,
            world_spec=world_spec,
            narrative_bible=None,
            req_spec=req_spec,
            staging_dir=staging_dir,
        )

    # Track R1 must be rejected and Track R2 must remain pending (loop halted)
    assert req_spec.requirements[0].status == "rejected"
    assert req_spec.requirements[1].status == "pending"

    # Verify disk persistence reflects the failure
    saved_spec = GameRequirementsSpec.model_validate_json(
        (staging_dir / "requirements_spec.json").read_text(encoding="utf-8")
    )
    assert saved_spec.requirements[0].status == "rejected"
    assert saved_spec.requirements[1].status == "pending"


@pytest.mark.anyio
async def test_orchestrator_iterative_swe_loop_fails_fast_when_mock_responses_exhausted(tmp_path: Path) -> None:
    """Verifies that if responses are exhausted for track R2, the loop fails fast and rejects R2 (no self-certification)."""
    client = MockGeminiClient()
    cartridge_bundle_data_r1 = {
        "deliberation": "Initial bundle for R1 only.",
        "manifest_data": {
            "title": "Neon Grid Survival",
            "genre": "Action Roguelike",
            "version": "1.0.0",
            "entrypoint": "mechanics.py",
        },
        "mechanics_code": (
            '"""mechanics.py"""\n'
            'from engine.api.contracts import BaseCartridge, BaseScene\n'
            'class GameScene(BaseScene):\n'
            '    deliberation: str = "Lifecycle scene."\n'
            '    def __init__(self, engine=None):\n'
            '        super().__init__(engine)\n'
            '        self.counter = 0\n'
            '    def update(self, dt):\n'
            '        self.counter += 1\n'
            '    def render(self, surface):\n'
            '        pass\n'
            'class GridCartridge(BaseCartridge):\n'
            '    deliberation: str = "Cartridge."\n'
            '    def get_scenes(self, engine):\n'
            '        return {"main": GameScene(engine)}\n'
            '    def get_initial_scene(self, engine):\n'
            '        return self.get_scenes(engine)["main"]\n'
        ),
        "scenes_code": None,
        "entities_code": None,
        "abilities_code": None,
        "level_data": None,
    }
    # Only queue response for R1; do NOT queue response for R2
    client.set_json_response(cartridge_bundle_data_r1)
    orch = StudioOrchestrator(client=client)

    staging_dir = tmp_path / "staging_exhausted_responses"
    staging_dir.mkdir(parents=True, exist_ok=True)

    req_spec = _make_dummy_req_spec()
    (staging_dir / "requirements_spec.json").write_text(
        req_spec.model_dump_json(indent=2), encoding="utf-8"
    )

    swe = orch.agents["LeadGameplaySWEAgent"]
    world_spec = GenesisWorldSpec(
        deliberation="World spec for exhausted response test",
        game_title="Neon Grid Survival",
        core_genre="Action Roguelike",
        thematic_premise="Cyber survival",
        world_lore="Grid world lore",
        conflict_tone="tense",
        conflict_narrative="survive against defensive countermeasures",
        aesthetic_style="retro_arcade",
        palette_mood="retro_arcade",
        gameplay_pacing="fast",
        pacing_score=0.5,
        target_difficulty=1.0,
    )

    with pytest.raises(StudioSwarmExecutionError) as exc_info:
        await orch._execute_iterative_swe_loop(
            agent=swe,
            world_spec=world_spec,
            narrative_bible=None,
            req_spec=req_spec,
            staging_dir=staging_dir,
        )

    assert "Requirement R2 ('Combat Projectiles') failed verification" in str(exc_info.value)
    # Track R1 completed, but Track R2 must be rejected
    assert req_spec.requirements[0].status == "completed"
    assert req_spec.requirements[1].status == "rejected"

    # Verify on-disk persistence
    saved_spec = GameRequirementsSpec.model_validate_json(
        (staging_dir / "requirements_spec.json").read_text(encoding="utf-8")
    )
    assert saved_spec.requirements[0].status == "completed"
    assert saved_spec.requirements[1].status == "rejected"

