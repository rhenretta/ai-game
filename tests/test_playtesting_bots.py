"""tests/test_playtesting_bots.py
Authoritative unit and integration test suite for Dual Synthetic Playtesting Bots (Feature 26).
Verifies RusherBot aggressive kinetic mechanics, CautiousBot tactical kiting & standoff,
boundary stress testing, DualPlaytester orchestration, and backward compatibility.
Strictly 0 occurrences of forbidden legacy branding tokens.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import pytest

from engine.api.contracts import BaseCartridge, BaseLevelSpec
from engine.entities.enemies import ChaserEnemy
from engine.entities.hazard import FloorAcidHazard, SpikeHazard
from engine.entities.player import Player
from engine.loader import CartridgeLoader, GenericCartridge
from engine.sim.simulator import HeadlessSimulator, SimulationInput
from engine.spatial.aabb import AABB, Vector2
from qa.bots import BasePlaytestingBot, CautiousBot, RusherBot
from qa.playtester import (
    BotAction,
    DualPlaytestReport,
    DualPlaytester,
    PlaytestResult,
    SyntheticPlaytester,
    SyntheticPlaytesterBot,
    run_playtest_evaluation,
)
from qa.profiles import CAUTIOUS_PROFILE, RUSHER_PROFILE, PlaytesterProfile
from tests.test_tier1_units import validate_rationale_first_ast


class TestBotEpistemicSchemaContracts:
    """Verifies that all QA bot schemas and actions adhere strictly to Rationale-First architecture."""
    deliberation: str = "Verifying Rationale-First field ordering on QA bot models."

    def test_bot_action_deliberation_first(self):
        action = BotAction(
            deliberation="Dashing forward toward enemy target.",
            move_x=1.0,
            move_y=0.0,
            dash=True,
            fire=True,
        )
        assert list(action.to_dict().keys())[0] == "deliberation"

    def test_playtest_result_deliberation_first(self):
        result = PlaytestResult(
            deliberation="RusherBot cleared sector with zero boundary violations.",
            bot_name="RusherBot",
            outcome="CLEARED",
            cleared=True,
            ticks=600,
            duration_seconds=10.0,
        )
        assert list(result.to_dict().keys())[0] == "deliberation"

    def test_dual_playtest_report_deliberation_first(self):
        r_res = PlaytestResult(
            deliberation="RusherBot clear.",
            bot_name="RusherBot",
            outcome="CLEARED",
            cleared=True,
            ticks=400,
            duration_seconds=6.6,
        )
        c_res = PlaytestResult(
            deliberation="CautiousBot clear.",
            bot_name="CautiousBot",
            outcome="CLEARED",
            cleared=True,
            ticks=700,
            duration_seconds=11.6,
        )
        report = DualPlaytestReport(
            deliberation="Dual playtest passed with high kinetic divergence.",
            rusher_result=r_res,
            cautious_result=c_res,
            both_cleared=True,
            pacing_divergence=5.0,
            status="PROMOTED",
        )
        assert list(report.to_dict().keys())[0] == "deliberation"

    def test_ast_inspection_of_qa_models(self):
        for path_str in ["qa/bots.py", "qa/playtester.py"]:
            code = Path(path_str).read_text(encoding="utf-8")
            assert validate_rationale_first_ast(code) is True


class TestRusherBotKineticMechanics:
    """Verifies RusherBot aggressive forward velocity, dash execution, and firing in range."""
    deliberation: str = "Verifying RusherBot kinetic mechanics and high dash cadence."

    def test_rusher_dashes_when_ready(self):
        bot = RusherBot()
        player = Player(x=120.0, y=360.0)
        player.stamina = 100.0
        player.dash_cooldown = 0.0

        enemy = ChaserEnemy(enemy_id="chaser_1", x=300.0, y=360.0)
        action = bot.decide_action(
            player=player,
            enemies=[enemy],
            hazards=[],
            exit_pos=Vector2(1160.0, 360.0),
            walls=[],
        )
        assert action.dash is True
        assert action.move_x > 0.0  # Moving toward enemy
        assert list(action.to_dict().keys())[0] == "deliberation"

    def test_rusher_fires_in_range(self):
        bot = RusherBot()
        player = Player(x=120.0, y=360.0)
        # Enemy in weapon range (<= 320px)
        enemy = ChaserEnemy(enemy_id="chaser_1", x=250.0, y=360.0)
        action = bot.decide_action(
            player=player,
            enemies=[enemy],
            hazards=[],
            exit_pos=Vector2(1160.0, 360.0),
            walls=[],
        )
        assert action.fire is True


class TestRusherBotBoundaryStressTesting:
    """Verifies collision checking and wall clipping incident detection."""
    deliberation: str = "Verifying boundary integrity and wall collision stress testing."

    def test_bot_tracks_collision_integrity(self):
        bot = RusherBot()
        wall = AABB(Vector2(640.0, 0.0), Vector2(32.0, 720.0))
        # Player inside wall (penetration)
        penetrating_player = Player(x=645.0, y=360.0)
        bot.check_collision_boundary_integrity(penetrating_player, [wall])
        assert bot.clipping_incidents > 0
        assert len(bot.clipping_locations) > 0


class TestCautiousBotStandoffAndKiting:
    """Verifies CautiousBot maintains distance and kites away from close threats."""
    deliberation: str = "Verifying CautiousBot standoff maintenance and kiting behavior."

    def test_cautious_bot_kites_when_too_close(self):
        bot = CautiousBot()
        player = Player(x=200.0, y=360.0)
        # Enemy is only 100px away, which is well inside standoff distance (260px)
        close_enemy = ChaserEnemy(enemy_id="chaser_1", x=300.0, y=360.0)
        action = bot.decide_action(
            player=player,
            enemies=[close_enemy],
            hazards=[],
            exit_pos=Vector2(1160.0, 360.0),
            walls=[],
        )
        # Kiting should reverse or push direction away from enemy (negative move_x)
        assert action.move_x < 0.0 or action.move_y != 0.0

    def test_cautious_bot_aims_at_enemy(self):
        bot = CautiousBot()
        player = Player(x=200.0, y=360.0)
        enemy = ChaserEnemy(enemy_id="chaser_1", x=400.0, y=360.0)
        action = bot.decide_action(
            player=player,
            enemies=[enemy],
            hazards=[],
            exit_pos=Vector2(1160.0, 360.0),
            walls=[],
        )
        assert action.aim_x == enemy.center.x
        assert action.aim_y == enemy.center.y
        assert action.fire is True


class TestCautiousBotHazardSafetyMargins:
    """Verifies CautiousBot avoids hazards at a safe buffer distance."""
    deliberation: str = "Verifying CautiousBot hazard evasion and safety buffer."

    def test_cautious_bot_evades_spikes(self):
        bot = CautiousBot()
        player = Player(x=200.0, y=360.0)
        # Spike hazard directly ahead
        spike = SpikeHazard(hazard_id="spike_1", x=240.0, y=360.0, w=48.0, h=48.0)
        from engine.entities.hazard import SpikeState
        spike.state = SpikeState.EXTENDED

        action = bot.decide_action(
            player=player,
            enemies=[],
            hazards=[spike],
            exit_pos=Vector2(1160.0, 360.0),
            walls=[],
        )
        # Lateral or escape movement
        assert action.move_y != 0.0 or action.move_x <= 0.0


@pytest.fixture
def sample_cartridge(tmp_path: Path) -> BaseCartridge:
    spec = BaseLevelSpec(
        level_id="mock_01",
        width=1280.0,
        height=720.0,
        spawn_point=(120.0, 360.0),
        exit_point=(1160.0, 360.0),
        walls=[
            (0.0, 0.0, 1280.0, 32.0),
            (0.0, 688.0, 1280.0, 32.0),
            (0.0, 0.0, 32.0, 720.0),
            (1248.0, 0.0, 32.0, 720.0),
        ],
        enemies=[{"id": "enemy_1", "type": "chaser", "x": 600.0, "y": 360.0}],
        hazards=[{"id": "hazard_1", "type": "spike", "x": 400.0, "y": 360.0, "w": 32.0, "h": 32.0}],
        deliberation="Mock level spec for playtesting bots test.",
    )
    cart = GenericCartridge(tmp_path)
    cart.level_specs[1] = spec
    return cart


class TestHeadlessSimulatorIntegration:
    """Verifies bot driving HeadlessSimulator over tick cycles."""
    deliberation: str = "Verifying HeadlessSimulator stepping driven by synthetic bot."

    def test_bot_runs_in_headless_simulator(self, sample_cartridge):
        cartridge = sample_cartridge
        level_spec = cartridge.get_level_spec(1)
        sim = HeadlessSimulator(cartridge=cartridge, level_spec=level_spec, max_steps=120)
        bot = RusherBot()
        result = bot.run_playtest(sim, max_ticks=120)
        assert result.ticks > 0
        assert list(result.to_dict().keys())[0] == "deliberation"


class TestDualBotBehavioralDivergence:
    """Verifies observable behavioral divergence between RusherBot and CautiousBot."""
    deliberation: str = "Verifying divergence in dashes and engagement between bot styles."

    def test_rusher_uses_more_dashes_than_cautious(self, sample_cartridge):
        cartridge = sample_cartridge
        spec = cartridge.get_level_spec(1)

        sim_rusher = HeadlessSimulator(cartridge=cartridge, level_spec=spec, max_steps=300)
        rusher = RusherBot()
        res_rusher = rusher.run_playtest(sim_rusher, max_ticks=300)

        sim_cautious = HeadlessSimulator(cartridge=cartridge, level_spec=spec, max_steps=300)
        cautious = CautiousBot()
        res_cautious = cautious.run_playtest(sim_cautious, max_ticks=300)

        assert res_rusher.dashes_used >= res_cautious.dashes_used


class TestDualPlaytesterOrchestration:
    """Verifies DualPlaytester orchestrates both bots and produces DualPlaytestReport."""
    deliberation: str = "Verifying DualPlaytester multi-bot evaluation and report synthesis."

    def test_dual_playtester_evaluates_cartridge(self, sample_cartridge):
        cartridge = sample_cartridge
        playtester = DualPlaytester(tile_size=32.0, max_ticks=600)
        report = playtester.evaluate_cartridge_dual(cartridge, episode_number=1)

        assert isinstance(report, DualPlaytestReport)
        assert report.rusher_result.bot_name == "RusherBot"
        assert report.cautious_result.bot_name == "CautiousBot"
        assert list(report.to_dict().keys())[0] == "deliberation"


class TestBackwardCompatibility:
    """Verifies existing SyntheticPlaytesterBot and run_playtest_evaluation APIs remain unbroken."""
    deliberation: str = "Verifying backward compatibility with legacy playtester interfaces."

    def test_synthetic_playtester_bot_alias(self):
        bot = SyntheticPlaytesterBot()
        assert isinstance(bot, BasePlaytestingBot)

    def test_run_playtest_evaluation_legacy_call(self, sample_cartridge):
        cartridge = sample_cartridge
        spec = cartridge.get_level_spec(1)
        result = run_playtest_evaluation(cartridge, spec, max_ticks=100)
        assert result is not None
        assert "ticks" in result or result.completion_ticks > 0
