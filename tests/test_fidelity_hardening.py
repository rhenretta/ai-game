"""Regression tests for fidelity / silent-failure hardening of the studio pipeline (Rule 5)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from engine.studio.models import (
    LevelLayoutSpec,
    SceneCritiqueAndStrategySpec,
    SceneFidelityRejection,
    SceneVerificationError,
    StudioAgentError,
    TrackCodeSynthesisPayload,
    VisualSceneLayerSpec,
)
from engine.studio.swe import LeadGameplaySWEAgent


def _agent() -> LeadGameplaySWEAgent:
    agent = LeadGameplaySWEAgent.__new__(LeadGameplaySWEAgent)
    agent.client = MagicMock()
    agent.model_name = "test-model"
    return agent


def test_layer_handoff_uses_real_fields(tmp_path: Path):
    layer = VisualSceneLayerSpec(
        deliberation="d", layer_id="mid", depth_order=3,
        scroll_parallax_factor=(0.5, 0.25), visual_description="hills",
        associated_asset_ids=["a1"],
    )
    bible = SimpleNamespace(scene_layers=[layer])
    ctx, _ = _agent()._build_bible_and_narrative_context(tmp_path, game_bible=bible)
    assert "depth_order=3" in ctx
    assert "(0.5, 0.25)" in ctx
    assert "a1" in ctx


def test_layer_handoff_fails_loudly_on_schema_drift(tmp_path: Path):
    bad_layer = SimpleNamespace(layer_id="x", visual_description="y")  # missing depth_order etc.
    bible = SimpleNamespace(scene_layers=[bad_layer])
    with pytest.raises(AttributeError):
        _agent()._build_bible_and_narrative_context(tmp_path, game_bible=bible)


def test_full_text_never_truncated_but_ceiling_raises():
    text = "x" * 50_000
    assert LeadGameplaySWEAgent._require_full_text(text, "bible") == text
    with pytest.raises(StudioAgentError):
        LeadGameplaySWEAgent._require_full_text("x" * (LeadGameplaySWEAgent.MAX_PROMPT_DOCUMENT_CHARS + 1), "bible")


def test_asset_fabrication_failure_propagates(tmp_path: Path, monkeypatch):
    import engine.studio.asset_library as al

    class Boom:
        def __init__(self, *a, **k): ...
        def fabricate_assets(self, *_a, **_k):
            raise RuntimeError("provider down")
        def save_catalog(self): ...

    monkeypatch.setattr(al, "CartridgeAssetLibrary", Boom)
    payload = TrackCodeSynthesisPayload.model_construct(
        requested_assets=[object()], modules=[], deliberation="d",
    )
    with pytest.raises((StudioAgentError, Exception)) as ei:
        payload.assemble_cartridge_directory(tmp_path, swe_agent=None)
    assert "provider down" in str(ei.value) or "provider down" in repr(ei.value.__cause__)


def _critique(ok: bool) -> SceneCritiqueAndStrategySpec:
    return SceneCritiqueAndStrategySpec(
        deliberation="reasoning", environment_fidelity_critique="e", ui_and_experience_critique="u",
        deficiencies=[], new_assets_needed=[], scene_matches_game_bible=ok,
    )


def _prime_loop_agent(tmp_path: Path, monkeypatch, verdicts, beat_reports):
    agent = _agent()
    agent._is_live_genai_client = lambda: True
    agent.formulate_production_strategy = lambda c_dir: SimpleNamespace(
        environment_topology_strategy="", camera_and_perspective_coherence_strategy="",
        psychoacoustic_audio_strategy="",
    )
    calls = {"analyze": 0, "patch": 0}

    def analyze(c_dir, strategy=None):
        v = verdicts[min(calls["analyze"], len(verdicts) - 1)]
        calls["analyze"] += 1
        return _critique(v)

    agent.analyze_scene_and_restrategize = analyze
    beat_iter = iter(beat_reports)
    agent.verify_authored_beats = lambda c_dir: next(beat_iter, beat_reports[-1])

    def patch(**kw):
        calls["patch"] += 1
        return {}

    agent.debug_and_patch_pev_cycle = patch
    agent._verify_and_heal_godot_syntax_and_probes = lambda c_dir: {}

    import engine.sim.godot_verifier as gv
    monkeypatch.setattr(gv, "verify_godot_cartridge_syntax", lambda c: (True, []))
    monkeypatch.setattr(gv, "GodotHeadlessSimulator",
                        lambda *a, **k: SimpleNamespace(run_until_complete=lambda: None))
    import engine.studio.asset_library as al
    monkeypatch.setattr(al, "CartridgeAssetLibrary", lambda **k: MagicMock())
    return agent, calls


def test_scene_loop_iterates_until_success(tmp_path: Path, monkeypatch):
    agent, calls = _prime_loop_agent(
        tmp_path, monkeypatch,
        verdicts=[False, False, False, True],
        beat_reports=[{"passed": True, "failures": []}],
    )
    agent.execute_scene_strategy_and_refine_loop(tmp_path)  # no ceiling
    assert calls["analyze"] == 4
    assert calls["patch"] == 3
    trace = (tmp_path / "scene_loop_trace.jsonl").read_text().strip().splitlines()
    assert len(trace) == 4
    assert json.loads(trace[0])["reasoning_rationale"] == "reasoning"


def test_scene_loop_beat_failure_blocks_acceptance(tmp_path: Path, monkeypatch):
    agent, calls = _prime_loop_agent(
        tmp_path, monkeypatch,
        verdicts=[True],
        beat_reports=[{"passed": False, "failures": ["b1: nope"]}, {"passed": True, "failures": []}],
    )
    agent.execute_scene_strategy_and_refine_loop(tmp_path)
    assert calls["analyze"] == 2 and calls["patch"] == 1


def test_scene_loop_explicit_ceiling_raises(tmp_path: Path, monkeypatch):
    agent, _ = _prime_loop_agent(
        tmp_path, monkeypatch, verdicts=[False], beat_reports=[{"passed": True, "failures": []}],
    )
    with pytest.raises(SceneFidelityRejection):
        agent.execute_scene_strategy_and_refine_loop(tmp_path, max_scene_iterations=2)


def test_visual_verification_exception_is_not_a_pass(tmp_path: Path, monkeypatch):
    agent = _agent()
    agent._is_live_genai_client = lambda: True
    clean = tmp_path / "clean.png"
    clean.write_bytes(b"\x89PNG\r\n")
    import engine.sim.atomic_runner as ar
    monkeypatch.setattr(
        ar.GodotAtomicTestRunner, "render_scene_and_collision_overlay",
        staticmethod(lambda **k: {"clean_render_path": str(clean), "scene_audit": {}}),
    )
    agent.client.models.generate_content.side_effect = RuntimeError("model down")
    unit = SimpleNamespace(unit_id="u1", title="t", single_responsibility_scope="s",
                           verification_modality="behavioral_only")
    with pytest.raises(SceneVerificationError):
        agent.verify_scene_unit_visually(tmp_path, unit)


def test_level_layout_spec_is_rationale_first():
    assert list(LevelLayoutSpec.model_fields)[0] == "deliberation"


# ---------------------------------------------------------------------------
# Level layout structural validation / world scale
# ---------------------------------------------------------------------------
from engine.studio.models import LevelEntitySpec, LevelRegionSpec, LevelTriggerSpec, validate_level_layout


def _entity(eid, pos=(100.0, 50.0), size=(10.0, 10.0)):
    return LevelEntitySpec(deliberation="d", entity_id=eid, entity_kind="k", position=pos, size=size)


def _layout(**over):
    base = dict(
        deliberation="d", viewport_size=(100.0, 100.0), intended_world_scale_in_viewports=10.0,
        world_bounds=(0.0, 0.0, 1000.0, 100.0), camera_framing="c",
        world_regions=[LevelRegionSpec(deliberation="d", region_id="r0", bounds=(0, 0, 1000, 100))],
        entities=[_entity("a"), _entity("b", (500.0, 50.0))],
        triggers=[LevelTriggerSpec(deliberation="d", trigger_id="t", source_entity_id="a",
                                   condition="c", target_entity_id="b", resulting_state="s")],
    )
    base.update(over)
    return LevelLayoutSpec(**base)


def test_valid_layout_has_no_issues():
    assert validate_level_layout(_layout()) == []


def test_tiny_world_vs_declared_scale_is_flagged():
    issues = validate_level_layout(_layout(world_bounds=(0.0, 0.0, 150.0, 100.0)))
    assert any("viewports" in i for i in issues)


def test_undeclared_scale_and_missing_regions_flagged():
    issues = validate_level_layout(_layout(intended_world_scale_in_viewports=0.0, world_regions=[]))
    assert any("intended_world_scale" in i for i in issues)
    assert any("world_regions" in i for i in issues)


def test_out_of_bounds_entity_unknown_and_self_trigger_flagged():
    bad_trigger = LevelTriggerSpec(deliberation="d", trigger_id="t2", source_entity_id="a",
                                   condition="c", target_entity_id="a", resulting_state="s")
    ghost = LevelTriggerSpec(deliberation="d", trigger_id="t3", source_entity_id="a",
                             condition="c", target_entity_id="ghost", resulting_state="s")
    issues = validate_level_layout(_layout(
        entities=[_entity("a"), _entity("b", (5000.0, 50.0))], triggers=[bad_trigger, ghost]))
    assert any("Entity 'b'" in i for i in issues)
    assert any("own source" in i for i in issues)
    assert any("unknown entity 'ghost'" in i for i in issues)


def test_layout_generation_regenerates_with_feedback(tmp_path: Path):
    agent = _agent()
    agent._is_live_genai_client = lambda: True
    agent._collect_concept_art_parts = lambda c: []
    (tmp_path / "game_bible.json").write_text("{}")
    bad = _layout(world_bounds=(0.0, 0.0, 150.0, 100.0))
    good = _layout()
    prompts = []

    def gen(model, contents, config):
        prompts.append(contents)
        return SimpleNamespace(text=(bad if len(prompts) == 1 else good).model_dump_json())

    agent.client.models.generate_content.side_effect = gen
    out = agent.formulate_level_layout(tmp_path)
    assert out.world_bounds == good.world_bounds
    assert len(prompts) == 2 and "FAILED STRUCTURAL VALIDATION" in prompts[1]
    assert (tmp_path / "level_layout.json").is_file()


def test_beat_plan_schema_has_no_free_form_dict_and_roundtrips():
    from engine.studio.models import BeatPlanSpec

    schema = json.dumps(BeatPlanSpec.model_json_schema())
    assert "additionalProperties" not in schema  # structured-output schemas reject free-form maps
    plan = BeatPlanSpec.model_validate_json(json.dumps({
        "deliberation": "d",
        "beats": [{"deliberation": "d", "beat_id": "b1", "steps": [{
            "deliberation": "d", "step_id": "s1", "scripted_input": "hold up",
            "expected_entity_states": [{"deliberation": "d", "entity_id": "e", "expected_state": "active"}],
        }]}],
    }))
    assert plan.beats[0].steps[0].expected_entity_states[0].entity_id == "e"


def test_layout_schema_has_no_free_form_dict():
    assert "additionalProperties" not in json.dumps(LevelLayoutSpec.model_json_schema())


def test_compact_bible_json_for_prompt_strips_process_bloat_and_preserves_specs():
    from engine.studio.models import compact_bible_json_for_prompt

    raw_bible = json.dumps({
        "deliberation": "top-level internal monologue",
        "game_title": "Test Title",
        "core_genre": "puzzle",
        "camera_and_spatial_paradigm": "2D side view",
        "color_palette_hex": ["#112233", "#445566"],
        "visual_style_guide": "Hand-painted gouache layers",
        "audio_direction_guide": "Warm acoustic marimba",
        "writers_room_transcript": [{"turn_index": 0, "spoken_dialogue": "chat " * 500}],
        "concept_fragments": [{"fragment_id": "f1", "fragment": "unused seed " * 200}],
        "concept_candidates": [{"candidate_id": "c1", "premise": "rejected"}],
        "concept_evolution": [{"entry_id": "e1", "evolved_premise": "pivot"}],
        "technical_architecture_blueprint": [
            {"deliberation": "nested reasoning", "mechanic_id": "m1", "title": "Core Mechanic"}
        ],
        "narrative_bible": {
            "deliberation": "narrative reasoning",
            "characters": [
                {
                    "deliberation": "char reasoning",
                    "character_id": "hero",
                    "name": "Hero",
                    "role": "protagonist",
                    "visual_description": "tall silhouette",
                    "voice_profile": "calm",
                }
            ],
            "dialogue_trees": [{"root_node_id": "root", "nodes": ["huge branching tree"]}],
        },
    })

    full_compact = json.loads(compact_bible_json_for_prompt(raw_bible, mode="full"))
    assert "writers_room_transcript" not in full_compact
    assert "concept_fragments" not in full_compact
    assert "concept_candidates" not in full_compact
    assert "concept_evolution" not in full_compact
    assert "deliberation" not in full_compact
    assert full_compact["game_title"] == "Test Title"
    assert "deliberation" not in full_compact["technical_architecture_blueprint"][0]
    assert full_compact["technical_architecture_blueprint"][0]["mechanic_id"] == "m1"

    visual_compact = json.loads(compact_bible_json_for_prompt(raw_bible, mode="visual_asset"))
    assert "technical_architecture_blueprint" not in visual_compact
    assert "audio_direction_guide" not in visual_compact
    assert visual_compact["visual_style_guide"] == "Hand-painted gouache layers"
    assert visual_compact["characters"][0]["visual_description"] == "tall silhouette"

    audio_compact = json.loads(compact_bible_json_for_prompt(raw_bible, mode="audio_asset"))
    assert "visual_style_guide" not in audio_compact
    assert audio_compact["audio_direction_guide"] == "Warm acoustic marimba"
    assert audio_compact["characters"][0]["voice_profile"] == "calm"


def test_call_gemini_with_transient_retry_retries_504_and_fails_fast_on_non_transient():
    from engine.studio.models import call_gemini_with_transient_retry

    calls = {"n": 0}

    def transient_then_ok():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("504 DEADLINE_EXCEEDED: Deadline expired before operation could complete.")
        return "ok"

    assert call_gemini_with_transient_retry(transient_then_ok, base_delay_s=0.01) == "ok"
    assert calls["n"] == 2

    non_transient_calls = {"n": 0}

    def permanent_error():
        non_transient_calls["n"] += 1
        raise ValueError("Invalid schema argument")

    with pytest.raises(ValueError, match="Invalid schema argument"):
        call_gemini_with_transient_retry(permanent_error, base_delay_s=0.01)
    assert non_transient_calls["n"] == 1

