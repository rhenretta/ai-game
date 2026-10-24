"""Rendering strategy: a per-game, model-decided rendering approach (painted layered 2D, stylized 3D, or hybrid).

The strategy is produced by a structured, rationale-first LLM contract (Rules 2 & 4), validated structurally,
persisted as `rendering_strategy.json`, and consumed by asset decomposition, SWE code generation and the scene critic.
Nothing here decides rendering by keyword heuristics.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from engine.studio.models import (
    RenderingStrategySpec,
    StudioAgentError,
    bounded_thinking_config,
    call_gemini_with_transient_retry,
    compact_bible_json_for_prompt,
)

logger = logging.getLogger("RenderingStrategy")

STRATEGY_FILENAME = "rendering_strategy.json"
MIN_SOURCE_DIMENSION = 16
MAX_SOURCE_DIMENSION = 8192

# Godot project feature tag / rendering_method value per renderer (project.godot).
RENDERER_PROJECT_SETTINGS = {
    "compatibility": ("GL Compatibility", "gl_compatibility"),
    "mobile": ("Mobile", "mobile"),
    "forward_plus": ("Forward Plus", "forward_plus"),
}


def validate_rendering_strategy(strategy: RenderingStrategySpec) -> List[str]:
    """Structural validation only; returns human-readable issues (empty = valid)."""
    issues: List[str] = []
    if not strategy.treatments:
        issues.append("treatments is empty; every significant subject needs a rendering treatment.")
    seen = set()
    for t in strategy.treatments:
        if t.subject_id in seen:
            issues.append(f"Duplicate treatment for subject '{t.subject_id}'.")
        seen.add(t.subject_id)
        w, h = t.target_resolution
        if not (MIN_SOURCE_DIMENSION <= w <= MAX_SOURCE_DIMENSION and MIN_SOURCE_DIMENSION <= h <= MAX_SOURCE_DIMENSION):
            issues.append(
                f"Treatment '{t.subject_id}' has target_resolution {t.target_resolution} outside "
                f"[{MIN_SOURCE_DIMENSION}, {MAX_SOURCE_DIMENSION}]."
            )
    if strategy.uses_environment_glow and strategy.renderer == "compatibility":
        issues.append("uses_environment_glow=True is unsupported by the 'compatibility' renderer; choose another renderer or set it False.")
    if strategy.pixelated_style_intended and strategy.rendering_paradigm == "stylized_3d":
        issues.append("pixelated_style_intended with 'stylized_3d' is contradictory; clarify the paradigm.")
    if not strategy.lighting_and_atmosphere.strip():
        issues.append("lighting_and_atmosphere must describe the lighting/atmosphere plan.")
    if not strategy.effects_stack:
        issues.append("effects_stack must list the concrete Godot features to be used.")
    return issues


def load_rendering_strategy(c_dir: Path) -> Optional[RenderingStrategySpec]:
    """Loads and returns the persisted strategy, or None when absent. Invalid files raise (Rule 5)."""
    path = Path(c_dir) / STRATEGY_FILENAME
    if not path.is_file():
        return None
    strategy = RenderingStrategySpec.model_validate_json(path.read_text(encoding="utf-8"))
    issues = validate_rendering_strategy(strategy)
    if issues:
        raise StudioAgentError(f"{path} is invalid: {issues}")
    return strategy


def ensure_rendering_strategy(
    c_dir: Path,
    client: Any,
    model_name: str,
    concept_parts: Optional[List[Any]] = None,
    max_attempts: int = 4,
    force_refresh: bool = False,
) -> RenderingStrategySpec:
    """Returns the cartridge's rendering strategy, generating (and persisting) it from the full Game Bible if needed.
    Validation issues are fed back into regeneration; exhausting attempts raises (Rule 5).
    """
    c_dir = Path(c_dir)
    prior_issues: List[str] = []
    path = c_dir / STRATEGY_FILENAME
    if path.is_file() and not force_refresh:
        try:
            cached = load_rendering_strategy(c_dir)
            if cached is not None:
                return cached
        except StudioAgentError as exc:
            prior_issues = [str(exc)]
            logger.info("Cached rendering strategy invalid; regenerating: %s", exc)

    if client is None or not getattr(type(client), "__module__", "").startswith("google.genai"):
        raise StudioAgentError("ensure_rendering_strategy requires an active google.genai Client (Rule 5)")
    bible_path = c_dir / "game_bible.json"
    if not bible_path.is_file():
        raise StudioAgentError(f"ensure_rendering_strategy requires {bible_path}")
    from google.genai import types

    bible_text = compact_bible_json_for_prompt(
        bible_path.read_text(encoding="utf-8"),
        mode="rendering",
    )
    for attempt in range(1, max_attempts + 1):
        feedback = (
            "YOUR PREVIOUS STRATEGY FAILED VALIDATION; FIX EVERY ISSUE:\n- " + "\n- ".join(prior_issues) + "\n\n"
            if prior_issues else ""
        )
        prompt = (
            "You are the Principal Technical Artist deciding the RENDERING STRATEGY (`RenderingStrategySpec`) for a game, "
            "using Godot 4.3. Read the full Game Bible and study any attached concept art.\n"
            "Devise a bespoke rendering paradigm and per-subject technical strategy tailored specifically to THIS game's "
            "camera, spatial structure, tone, mechanics, and concept art with a cohesive, high-fidelity look. Do not default "
            "to small low-resolution sprites; use `pixelated_style_intended=True` only if the Bible explicitly calls for a "
            "pixel-art/low-resolution aesthetic.\n"
            "For every significant subject (environment/stage elements, landmarks, mechanisms, actors/entities, effects, UI) "
            "specify the rendering `technique`, `target_resolution`, and construction `notes` that suit this specific game "
            "(for example: multi-plane backdrops, top-down/isometric tactical surfaces, single-screen arena stages, layered "
            "cutout rigs, frame-animated sprite sheets, procedural/shader geometry, particle systems, or 3D viewports).\n"
            "Plan lighting, atmosphere and post-processing concretely. The 'compatibility' renderer is the most portable but "
            "cannot use WorldEnvironment glow (use additive-blend light sprites and shaders instead); 'mobile' and "
            "'forward_plus' support glow.\n"
            "Rules: provide `deliberation` FIRST in every schema object (Rule 4).\n\n"
            f"{feedback}--- GAME BIBLE ---\n{bible_text}\n"
        )
        contents: Any = list(concept_parts or []) + [prompt] if concept_parts else prompt
        resp = call_gemini_with_transient_retry(
            lambda: client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RenderingStrategySpec,
                    thinking_config=bounded_thinking_config(2048),
                ),
            ),
            label="ensure_rendering_strategy",
        )
        if not resp.text:
            raise StudioAgentError("Rendering strategy generation returned an empty response")
        strategy = RenderingStrategySpec.model_validate_json(resp.text)
        prior_issues = validate_rendering_strategy(strategy)
        if not prior_issues:
            path.write_text(strategy.model_dump_json(indent=2), encoding="utf-8")
            return strategy
        logger.info("Rendering strategy attempt %d/%d failed validation: %s", attempt, max_attempts, prior_issues)
    raise StudioAgentError(f"Rendering strategy still invalid after {max_attempts} attempts: {prior_issues}")


_COMMON_GUIDE = (
    "GODOT 4.3 RENDERING EXECUTION GUIDE (realize the game's specific `RenderingStrategySpec`; never substitute flat untextured programmer-art rectangles for authored visuals):\n"
    "- World & Stage Surfaces: construct the environment according to `rendering_paradigm` and each subject's `technique` in `treatments` "
    "(e.g. multi-plane layers, top-down/isometric surfaces, single-screen stage compositions, or procedural/shader canvases); use `Sprite2D` "
    "with `texture_filter = TEXTURE_FILTER_LINEAR` (smooth, not pixelated) unless `pixelated_style_intended` is True.\n"
    "- Lighting: `CanvasModulate` for ambient tone, `PointLight2D`/`DirectionalLight2D` with soft gradient textures for emissive sources.\n"
    "- Effects: `GPUParticles2D` + `ParticleProcessMaterial`, `Line2D` trails, `CanvasItem` shaders, additive blending via `CanvasItemMaterial.blend_mode = BLEND_MODE_ADD`.\n"
    "- Actors & Entities: follow the per-subject `treatment` in `rendering_strategy.json` — sprite sheets, articulated multi-part rigs, or procedural/shader motion.\n"
    "- Camera: configure `Camera2D` (or fixed stage framing) to match the Game Bible's `camera_and_spatial_paradigm` and `world_bounds`.\n"
    "- 3D (when the strategy uses 3D): `SubViewport`/`Node3D` scenes with `MeshInstance3D`, `DirectionalLight3D`, `Environment`.\n"
)

_RENDERER_GUIDE = {
    "compatibility": (
        "RENDERER: compatibility. Do NOT enable WorldEnvironment glow (unsupported). Emulate glow with additive-blend soft light sprites, "
        "PointLight2D and shaders. project.godot must use config/features \"GL Compatibility\" and renderer/rendering_method=\"gl_compatibility\".\n"
    ),
    "mobile": (
        "RENDERER: mobile. WorldEnvironment glow is available. project.godot must use config/features \"Mobile\" and renderer/rendering_method=\"mobile\".\n"
    ),
    "forward_plus": (
        "RENDERER: forward_plus. WorldEnvironment glow is available. project.godot must use config/features \"Forward Plus\" and renderer/rendering_method=\"forward_plus\".\n"
    ),
}


def build_rendering_guide(strategy: RenderingStrategySpec) -> str:
    """Prompt block describing the mandatory rendering strategy plus a generic Godot technique guide."""
    glow = (
        "WorldEnvironment glow is part of the plan.\n" if strategy.uses_environment_glow else "Do not rely on WorldEnvironment glow.\n"
    )
    pix = (
        "A pixelated/low-resolution style IS intended: use nearest-neighbour filtering.\n"
        if strategy.pixelated_style_intended
        else "A pixelated/low-resolution look is NOT intended: use smooth filtering and high-resolution source art.\n"
    )
    return (
        "RENDERING STRATEGY (`rendering_strategy.json` — MANDATORY; realise it exactly):\n"
        + strategy.model_dump_json(indent=2)
        + "\n"
        + _RENDERER_GUIDE[strategy.renderer]
        + glow
        + pix
        + _COMMON_GUIDE
    )


def project_godot_renderer_settings(strategy: RenderingStrategySpec) -> tuple:
    """Returns (config/features tag, rendering_method) for project.godot."""
    return RENDERER_PROJECT_SETTINGS[strategy.renderer]


def check_project_renderer(c_dir: Path, strategy: RenderingStrategySpec) -> List[str]:
    """Structural check that project.godot declares the renderer the strategy requires. Returns issues (empty = OK)."""
    project = Path(c_dir) / "project.godot"
    if not project.is_file():
        return ["project.godot is missing."]
    text = project.read_text(encoding="utf-8")
    feature_tag, method = RENDERER_PROJECT_SETTINGS[strategy.renderer]
    issues: List[str] = []
    if f'"{feature_tag}"' not in text:
        issues.append(f'project.godot config/features must include "{feature_tag}" for the {strategy.renderer} renderer.')
    if f'renderer/rendering_method="{method}"' not in text:
        issues.append(f'project.godot [rendering] must set renderer/rendering_method="{method}".')
    return issues
