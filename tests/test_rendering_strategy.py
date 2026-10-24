"""Tests for the per-game rendering strategy (modern rendering pipeline)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from engine.studio.models import RenderingStrategySpec, StudioAgentError, SubjectRenderTreatment
from engine.studio.rendering import (
    build_rendering_guide,
    check_project_renderer,
    ensure_rendering_strategy,
    load_rendering_strategy,
    validate_rendering_strategy,
)


def _treatment(sid="sky", res=(1920, 1080), tech="painted_parallax_layer"):
    return SubjectRenderTreatment(deliberation="d", subject_id=sid, technique=tech, target_resolution=res)


def _strategy(**over):
    base = dict(
        deliberation="d", rendering_paradigm="painted_2d_layered", renderer="compatibility",
        uses_environment_glow=False, pixelated_style_intended=False,
        treatments=[_treatment()], lighting_and_atmosphere="warm rim light, mist",
        effects_stack=["GPUParticles2D", "PointLight2D"], asset_resolution_policy="2x on-screen size",
    )
    base.update(over)
    return RenderingStrategySpec(**base)


def test_schema_rationale_first_and_no_free_form_maps():
    assert list(RenderingStrategySpec.model_fields)[0] == "deliberation"
    assert list(SubjectRenderTreatment.model_fields)[0] == "deliberation"
    assert "additionalProperties" not in json.dumps(RenderingStrategySpec.model_json_schema())


def test_valid_strategy_has_no_issues():
    assert validate_rendering_strategy(_strategy()) == []


def test_glow_with_compatibility_renderer_flagged():
    issues = validate_rendering_strategy(_strategy(uses_environment_glow=True))
    assert any("glow" in i for i in issues)
    assert validate_rendering_strategy(_strategy(uses_environment_glow=True, renderer="forward_plus")) == []


def test_empty_treatments_bad_resolution_duplicates_and_missing_plans_flagged():
    issues = validate_rendering_strategy(_strategy(treatments=[], lighting_and_atmosphere="", effects_stack=[]))
    assert any("treatments" in i for i in issues)
    assert any("lighting_and_atmosphere" in i for i in issues)
    assert any("effects_stack" in i for i in issues)
    bad = validate_rendering_strategy(_strategy(treatments=[_treatment(res=(4, 99999)), _treatment()]))
    assert any("target_resolution" in i for i in bad)
    dup = validate_rendering_strategy(_strategy(treatments=[_treatment(), _treatment()]))
    assert any("Duplicate" in i for i in dup)


def test_guide_reflects_pixelation_and_glow_choices():
    modern = build_rendering_guide(_strategy())
    assert "NOT intended" in modern and "Do not rely on WorldEnvironment glow" in modern and "gl_compatibility" in modern
    pixel = build_rendering_guide(_strategy(pixelated_style_intended=True, renderer="forward_plus", uses_environment_glow=True))
    assert "nearest-neighbour" in pixel and "forward_plus" in pixel and "glow is part of the plan" in pixel


def test_project_renderer_check(tmp_path: Path):
    s = _strategy()
    assert check_project_renderer(tmp_path, s)  # missing project.godot
    (tmp_path / "project.godot").write_text('[application]\nconfig/features=PackedStringArray("4.3", "Forward Plus")\n')
    assert len(check_project_renderer(tmp_path, s)) == 2
    (tmp_path / "project.godot").write_text(
        '[application]\nconfig/features=PackedStringArray("4.3", "GL Compatibility")\n'
        '[rendering]\nrenderer/rendering_method="gl_compatibility"\n')
    assert check_project_renderer(tmp_path, s) == []


class _FakeGenaiClient:
    __module__ = "google.genai.fake"


def _live_client(responses):
    client = _FakeGenaiClient()
    client.models = MagicMock()
    it = iter(responses)
    client.models.generate_content.side_effect = lambda **k: SimpleNamespace(text=next(it).model_dump_json())
    return client


def test_ensure_regenerates_with_feedback_and_persists(tmp_path: Path):
    (tmp_path / "game_bible.json").write_text("{}")
    bad = _strategy(uses_environment_glow=True)  # invalid: glow + compatibility
    good = _strategy()
    client = _live_client([bad, good])
    out = ensure_rendering_strategy(tmp_path, client, "m")
    assert out.renderer == "compatibility" and client.models.generate_content.call_count == 2
    second_prompt = client.models.generate_content.call_args_list[1].kwargs["contents"]
    assert "FAILED VALIDATION" in second_prompt
    assert load_rendering_strategy(tmp_path) is not None
    # cached on second call: no further generation
    ensure_rendering_strategy(tmp_path, client, "m")
    assert client.models.generate_content.call_count == 2


def test_ensure_raises_after_exhausting_attempts(tmp_path: Path):
    (tmp_path / "game_bible.json").write_text("{}")
    client = _live_client([_strategy(treatments=[])] * 4)
    with pytest.raises(StudioAgentError):
        ensure_rendering_strategy(tmp_path, client, "m", max_attempts=4)


def test_ensure_requires_live_client(tmp_path: Path):
    (tmp_path / "game_bible.json").write_text("{}")
    with pytest.raises(StudioAgentError):
        ensure_rendering_strategy(tmp_path, MagicMock(), "m")


def test_decomposition_prompt_is_strategy_driven_and_unclamped(tmp_path: Path, monkeypatch):
    """Decomposition must embed the rendering strategy and honour large resolutions (no 48-128px clamp)."""
    from engine.studio import asset_library as al

    (tmp_path / "game_bible.json").write_text(json.dumps({"game_title": "t"}))
    (tmp_path / "rendering_strategy.json").write_text(_strategy().model_dump_json())
    captured = {}

    class _Item(SimpleNamespace):
        pass

    def fake_generate(**k):
        captured["contents"] = k["contents"]
        spec_cls = k["config"].response_schema
        item = {
            "deliberation": "d", "asset_id": "sky_layer", "asset_type": "backdrop", "fabrication_mode": "parallax_layer",
            "purpose": "p", "visual_prompt": "vp", "width": 3840, "height": 2160,
        }
        return SimpleNamespace(text=json.dumps({"deliberation": "d", "elements": [item]}))

    client = _FakeGenaiClient()
    client.models = MagicMock()
    client.models.generate_content.side_effect = fake_generate
    lib = al.CartridgeAssetLibrary(cartridge_dir=tmp_path, client=client)
    lib.get_concept_reference_images = lambda *a, **k: []
    elements = lib.decompose_concept_into_elements(bible_dict={"game_title": "t"}, world_spec={}, landmarks=[], mockup_desc="m", style_guide="s")
    prompt = captured["contents"][-1]
    assert "RENDERING STRATEGY" in prompt and "painted_parallax_layer" in prompt
    layer = next(e for e in elements if e.asset_id == "sky_layer")
    assert layer.dimensions == (3840, 2160) and layer.fabrication_mode == "parallax_layer"


def test_style_guide_states_non_pixel_unless_intended(tmp_path: Path):
    from engine.studio import asset_library as al

    (tmp_path / "game_bible.json").write_text("{}")
    (tmp_path / "rendering_strategy.json").write_text(_strategy().model_dump_json())
    assert "NOT pixel art" in al.CartridgeAssetLibrary(cartridge_dir=tmp_path).get_style_guide()
    (tmp_path / "rendering_strategy.json").write_text(_strategy(pixelated_style_intended=True).model_dump_json())
    assert "Pixel-art" in al.CartridgeAssetLibrary(cartridge_dir=tmp_path).get_style_guide()
