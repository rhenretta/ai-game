"""tests/test_asset_library.py
Unit and integration tests for the Cartridge Asset Library, On-Demand Agent Fabrication Toolkit,
Alpha/Border Sprite Extraction, Portrait Isolation, and Visual Craft Challenger Gate.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from engine.studio.asset_library import CartridgeAssetLibrary
from engine.studio.models import (
    AnimationStateSpec,
    ArticulatedPartSpec,
    AssetLibraryCatalog,
    AssetLibraryEntry,
    AssetRequirement,
    GameRequirementsSpec,
    ModuleFileSpec,
    RequirementTrack,
    TrackCodeSynthesisPayload,
    VisualSceneLayerSpec,
)
from engine.studio.visual import VisualArtSpriteDesignerAgent
from engine.supervisor.challenger import CandidateASTInspector, MultiAgentSignOffCouncil
from engine.telemetry.db import TelemetryDB


def test_rationale_first_schema_ordering():
    """Verifies Rule 4: deliberation is strictly the first field in all new Asset Library models."""
    for model_cls in (
        AnimationStateSpec,
        ArticulatedPartSpec,
        VisualSceneLayerSpec,
        AssetLibraryEntry,
        AssetLibraryCatalog,
        AssetRequirement,
    ):
        first_field = next(iter(model_cls.model_fields.keys()))
        assert first_field == "deliberation", f"{model_cls.__name__} must declare 'deliberation' first, got '{first_field}'"


def test_portrait_vs_sprite_directory_isolation(tmp_path: Path):
    """Verifies portrait assets route to assets/portraits/ and misplaced portraits in assets/sprites/ are sanitized."""
    req = AssetRequirement(
        deliberation="Synthesize dialogue portrait for companion.",
        asset_id="pip_portrait",
        asset_type="portrait",
        fabrication_mode="portrait",
        target_path="assets/sprites/pip_portrait.png",
        description="A glowing companion portrait.",
    )
    assert req.target_path == "assets/portraits/pip_portrait.png"

    # Place a misplaced portrait in assets/sprites/ and verify sanitize_misplaced_portraits_in_sprites_dir moves it
    sprites_dir = tmp_path / "assets" / "sprites"
    sprites_dir.mkdir(parents=True, exist_ok=True)
    misplaced = sprites_dir / "kaelen_portrait.png"
    img = Image.new("RGBA", (128, 128), (240, 240, 240, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((32, 32, 96, 96), fill=(40, 120, 220, 255))
    img.save(misplaced, format="PNG")

    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path)
    moved = lib.sanitize_misplaced_portraits_in_sprites_dir()
    assert len(moved) == 1
    assert not misplaced.exists()
    assert (tmp_path / "assets" / "portraits" / "kaelen_portrait.png").is_file()


def test_extract_clean_alpha_sprite_strips_border_and_background():
    """Verifies extract_clean_alpha_sprite strips outer decorative frames and flood-fills opaque backgrounds to alpha=0."""
    # Create a 256x256 image with a dark border frame, solid light-gray background, and a colored character in the center
    img = Image.new("RGBA", (256, 256), (235, 235, 240, 255))
    draw = ImageDraw.Draw(img)
    # Outer decorative frame
    draw.rectangle((0, 0, 255, 255), outline=(20, 20, 25, 255), width=6)
    draw.rectangle((10, 10, 245, 245), outline=(40, 40, 50, 255), width=3)
    # Central subject
    draw.ellipse((88, 72, 168, 184), fill=(30, 180, 230, 255), outline=(15, 60, 95, 255), width=3)

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    cleaned = VisualArtSpriteDesignerAgent.extract_clean_alpha_sprite(
        buf.getvalue(),
        target_dimensions=(64, 64),
        preserve_aspect=True,
    )
    assert cleaned.mode == "RGBA"
    assert cleaned.size == (64, 64)
    # Corners of the cleaned sprite must be transparent (alpha == 0)
    assert cleaned.getpixel((0, 0))[3] == 0
    assert cleaned.getpixel((63, 0))[3] == 0
    assert cleaned.getpixel((0, 63))[3] == 0
    assert cleaned.getpixel((63, 63))[3] == 0
    # Center of the cleaned sprite must be opaque subject
    assert cleaned.getpixel((32, 32))[3] > 200


def test_asset_library_fabricates_sprite_sheet_and_articulated_rig_and_saves_catalog(tmp_path: Path):
    """Verifies CartridgeAssetLibrary fabricates static sprites, multi-state sprite sheets,
    and articulated multi-part rigs, and persists assets/asset_library.json without generating any .gd files.
    """
    visual_agent = VisualArtSpriteDesignerAgent(client=None)
    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path, visual_agent=visual_agent)

    reqs = [
        AssetRequirement(
            deliberation="Animated protagonist sprite sheet with idle, move, and jump states.",
            asset_id="pioneer_avatar",
            asset_type="sprite_sheet",
            fabrication_mode="sprite_sheet",
            dimensions=(64, 64),
            description="Side-view crystal explorer character.",
            animation_states=[
                AnimationStateSpec(deliberation="Idle breathing", state_name="idle", motion_profile="idle", frame_count=4, fps=8.0, motion_description="breathing"),
                AnimationStateSpec(deliberation="Running stride", state_name="move", motion_profile="locomotion", frame_count=4, fps=10.0, motion_description="running"),
                AnimationStateSpec(deliberation="Mid-air jump", state_name="jump", motion_profile="aerial", frame_count=4, fps=10.0, motion_description="jumping"),
            ],
        ),
        AssetRequirement(
            deliberation="Articulated crystal golem with rotating arm.",
            asset_id="crystal_guardian",
            asset_type="articulated_rig",
            fabrication_mode="articulated_parts",
            dimensions=(80, 80),
            description="Faceted crystal guardian construct.",
            articulated_parts=[
                ArticulatedPartSpec(
                    deliberation="Core torso",
                    part_id="torso",
                    part_description="Faceted crystal torso",
                    dimensions=(48, 56),
                ),
                ArticulatedPartSpec(
                    deliberation="Swinging prism arm",
                    part_id="prism_arm",
                    part_description="Glowing crystal arm",
                    dimensions=(28, 40),
                    attach_offset_px=(18.0, -6.0),
                    rotation_range_deg=(-35.0, 45.0),
                ),
            ],
        ),
    ]

    fabricated = lib.fabricate_assets(reqs)
    assert "pioneer_avatar" in fabricated
    assert "crystal_guardian" in fabricated

    sheet_path = tmp_path / "assets" / "sprites" / "pioneer_avatar_sheet.png"
    assert sheet_path.is_file()
    sheet_img = Image.open(sheet_path)
    assert sheet_img.size == (64 * 4, 64 * 3)

    part_torso = tmp_path / "assets" / "sprites" / "parts" / "crystal_guardian_torso.png"
    part_arm = tmp_path / "assets" / "sprites" / "parts" / "crystal_guardian_prism_arm.png"
    assert part_torso.is_file()
    assert part_arm.is_file()

    catalog_json = tmp_path / "assets" / "asset_library.json"
    assert catalog_json.is_file()
    cat_data = json.loads(catalog_json.read_text(encoding="utf-8"))
    assert "pioneer_avatar" in cat_data["entries"]
    assert cat_data["entries"]["pioneer_avatar"]["hframes"] == 4
    assert cat_data["entries"]["pioneer_avatar"]["vframes"] == 3
    assert not (tmp_path / "asset_library.gd").exists()


def test_on_demand_requested_assets_in_track_code_synthesis_payload(tmp_path: Path):
    """Verifies TrackCodeSynthesisPayload.assemble_cartridge_directory fabricates SWE-commissioned requested_assets."""
    payload = TrackCodeSynthesisPayload(
        deliberation="Synthesizing core scene with an on-demand glowing leyline pylon prop.",
        files=[
            ModuleFileSpec(
                deliberation="Root scene script.",
                filename="main.gd",
                content="extends Node2D\nfunc _ready() -> void:\n\tpass\n",
            )
        ],
        requested_assets=[
            AssetRequirement(
                deliberation="On-demand leyline pylon prop requested by SWE during track implementation.",
                asset_id="leyline_pylon",
                asset_type="sprite",
                fabrication_mode="static_sprite",
                dimensions=(48, 96),
                description="Glowing cyan crystal pylon.",
            )
        ],
    )
    written = payload.assemble_cartridge_directory(tmp_path)
    assert "main.gd" in written
    assert "asset_library.gd" not in written
    assert (tmp_path / "assets" / "sprites" / "leyline_pylon.png").is_file()
    assert (tmp_path / "assets" / "asset_library.json").is_file()


def test_telemetry_db_persists_and_rehydrates_full_requirements_spec(tmp_path: Path):
    """Verifies TelemetryDB persists full GameRequirementsSpec JSON including asset_manifest."""
    db = TelemetryDB(tmp_path / "telemetry.db")
    spec = GameRequirementsSpec(
        deliberation="Full requirements spec with asset_manifest.",
        game_title="Prismtide Test",
        genre_paradigm="2D Side-Scrolling Platformer",
        high_concept_summary="Restore shattered crystal leylines across floating islands.",
        requirements=[
            RequirementTrack(
                deliberation="Core track",
                id="REQ-1",
                title="Core Platforming",
                category="mechanics",
                description="Implement jump and beam redirect.",
            )
        ],
        asset_manifest=[
            AssetRequirement(
                deliberation="Hero sprite sheet",
                asset_id="hero_sheet",
                asset_type="sprite_sheet",
                fabrication_mode="sprite_sheet",
                dimensions=(64, 64),
                description="Animated pioneer sprite sheet.",
            )
        ],
    )
    db.record_studio_requirements("ep_test_1", spec)
    rehydrated = db.get_studio_requirements_spec("ep_test_1")
    assert rehydrated is not None
    assert rehydrated["game_title"] == "Prismtide Test"
    assert len(rehydrated["asset_manifest"]) == 1
    assert rehydrated["asset_manifest"][0]["asset_id"] == "hero_sheet"


def test_challenger_vetoes_portrait_bound_to_world_entity(tmp_path: Path):
    """Verifies MultiAgentSignOffCouncil surfaces telemetry grounding to the LLM council and preserves specialist vetoes."""
    from types import SimpleNamespace
    from engine.supervisor.challenger import SPECIALIST_COUNCIL_ROLES
    from engine.supervisor.models import MultiAgentSignOffReport, SpecialistSignOffVerdict

    (tmp_path / "project.godot").write_text('[application]\nconfig/name="Test"\n', encoding="utf-8")
    (tmp_path / "main.gd").write_text(
        "extends Node2D\nfunc _ready() -> void:\n\tvar x = 1\nfunc _process(delta: float) -> void:\n\tvar y = delta\n",
        encoding="utf-8",
    )
    (tmp_path / "entities.gd").write_text(
        'extends CharacterBody2D\nfunc _ready() -> void:\n\tvar tex = load("res://assets/portraits/hero_portrait.png")\n',
        encoding="utf-8",
    )
    (tmp_path / "scenes.gd").write_text(
        "extends Node2D\nfunc _draw() -> void:\n\tdraw_rect(Rect2(0, 0, 100, 100), Color.BLUE)\n",
        encoding="utf-8",
    )
    insp = CandidateASTInspector()
    ast_report = insp.inspect_directory(tmp_path)

    captured_prompts = []

    class _MockCouncilClient:
        __module__ = "google.genai.client"

        def __init__(self):
            self.models = self

        def generate_content(self, *, model, contents, config=None):
            captured_prompts.append(str(contents[0]))
            rep = MultiAgentSignOffReport(
                deliberation="VisualArtDirector rejected portrait texture bound onto world entity.",
                candidate_id=tmp_path.name,
                unanimous_approval=False,
                verdicts=[
                    SpecialistSignOffVerdict(
                        deliberation="Portrait texture bound on world entity." if role == "VisualArtDirector" else "OK",
                        specialist_role=role,
                        approved=(role != "VisualArtDirector"),
                        blocking_defects=["hero_portrait.png misused on world Sprite2D"] if role == "VisualArtDirector" else [],
                    )
                    for role in SPECIALIST_COUNCIL_ROLES
                ],
            )
            return SimpleNamespace(text=rep.model_dump_json())

    council = MultiAgentSignOffCouncil(client=_MockCouncilClient())
    sign_off = council.evaluate_sign_off(
        candidate_dir=tmp_path,
        ast_grounding=ast_report,
        telemetry_grounding={"scene_audit": {"portrait_misused_on_world_sprite": True, "portrait_misused_nodes": ["Player:hero_portrait.png"]}},
        visual_grounding={"blank_canvas_detected": False},
    )
    assert any("hero_portrait.png" in p for p in captured_prompts)
    assert not sign_off.unanimous_approval
    visual_verdict = next(v for v in sign_off.verdicts if v.specialist_role == "VisualArtDirector")
    assert not visual_verdict.approved


def test_decompose_concept_into_world_elements(tmp_path: Path):
    """Verifies CartridgeAssetLibrary.decompose_concept_into_elements decomposes landmarks and world_spec
    into discrete interactive elements without generating any static background wallpaper.
    """
    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path)
    elements = lib.decompose_concept_into_elements(
        bible_dict={"elevator_pitch": "Floating crystal terraces"},
        world_spec={"biome_theme": "Jade sky archipelago"},
        landmarks=[{"landmark_id": "siphon_gate", "name": "Siphon Gate", "landmark_type": "mechanism", "visual_description": "Brass valve"}],
    )
    ids = {e.asset_id for e in elements}
    assert "siphon_gate" in ids
    assert "backdrop_main" not in ids


def test_pev_surgical_debugging_slice_and_patch(tmp_path: Path):
    """Verifies LeadGameplaySWEAgent PEV symbol indexing, scoped slice extraction, and surgical patching."""
    from engine.studio.models import CodeSliceRequest, SurgicalCodePatch
    from engine.studio.swe import LeadGameplaySWEAgent

    sample_gd = tmp_path / "scenes.gd"
    sample_gd.write_text(
        "extends Node2D\n"
        "\n"
        "const CELL_SIZE: float = 24.0\n"
        "\n"
        "static func build_world() -> Node2D:\n"
        "\tvar root = Node2D.new()\n"
        "\treturn root\n"
        "\n"
        "func _draw() -> void:\n"
        "\tvar rect = Rect2(0.0, 60.0, 200.0, 500.0)\n"
        "\tdraw_rect(rect, Color.PURPLE)\n",
        encoding="utf-8",
    )

    swe = LeadGameplaySWEAgent(client=None)
    sym_idx = swe._build_cartridge_symbol_index(tmp_path)
    assert "scenes.gd (11 lines total)" in sym_idx
    assert "static func build_world() -> Node2D:" in sym_idx
    assert "func _draw() -> void:" in sym_idx

    slice_txt = swe._extract_code_slice(
        tmp_path,
        CodeSliceRequest(
            deliberation="Inspect _draw() method to fix shelf top y coordinate.",
            filename="scenes.gd",
            start_line=9,
            end_line=11,
        ),
    )
    assert "scenes.gd [Lines 9..11 of 11]" in slice_txt
    assert "   9: func _draw() -> void:" in slice_txt
    assert "static func build_world" not in slice_txt

    patched = swe._apply_surgical_patches(
        tmp_path,
        [
            SurgicalCodePatch(
                deliberation="Lower shelf top from y=60.0 to y=250.0 without touching build_world().",
                filename="scenes.gd",
                start_line=10,
                end_line=11,
                replacement_code="\tvar rect = Rect2(0.0, 250.0, 200.0, 310.0)\n\tdraw_rect(rect, Color.CYAN)",
            )
        ],
    )
    assert "scenes.gd" in patched
    updated_src = sample_gd.read_text(encoding="utf-8")
    assert "static func build_world() -> Node2D:" in updated_src
    assert "Rect2(0.0, 250.0, 200.0, 310.0)" in updated_src
    assert "Color.CYAN" in updated_src
    assert "Color.PURPLE" not in updated_src


def test_production_strategy_and_scene_critique_schemas_and_asset_fabrication(tmp_path: Path):
    """Verifies Rationale-First GameProductionStrategySpec, SceneCritiqueAndStrategySpec, and ensure_required_assets."""
    from engine.studio.models import (
        AssetRequirement,
        AudioCueStrategy,
        GameProductionStrategySpec,
        SceneCritiqueAndStrategySpec,
        SceneDeficiencyItem,
        SceneLayerStrategy,
    )

    strat = GameProductionStrategySpec(
        deliberation="Decompose floating island world into multi-plane parallax layers and sculpted cliff polygons.",
        environment_topology_strategy="Use curved cliff polygons with mossy rims and layered flora sprites.",
        scene_layers=[
            SceneLayerStrategy(
                deliberation="Mid-ground floating island terrace.",
                layer_name="PlayfieldTerrace",
                z_index=0,
                construction_technique="Sculpted stone cliff Polygon2D with cyan water canal.",
                required_asset_ids=["cliff_spire"],
            )
        ],
        required_visual_assets=[
            AssetRequirement(
                deliberation="Need a mossy cliff spire prop for terrace silhouette.",
                asset_id="cliff_spire",
                name="cliff_spire.png",
                asset_type="prop",
                fabrication_mode="static_sprite",
                purpose="Sculpted rocky spire along floating island edge",
                target_path="assets/sprites/cliff_spire.png",
                description="Mossy basalt cliff spire with cyan crystal veins",
                visual_prompt_or_text="2D game prop: mossy basalt cliff spire with cyan crystal veins",
                dimensions=(96, 128),
            )
        ],
        audio_cues=[
            AudioCueStrategy(
                deliberation="Resonance chime when Chime-Ferns bloom.",
                cue_id="chime_bloom",
                trigger_condition="fern_hydrated",
                synthesis_or_bark_spec="Harmonic chime at 528 Hz for 0.35s",
            )
        ],
        code_architecture_strategy="scenes.gd builds layered terrain; entities.gd handles player surf physics; ui_hud.gd renders diegetic crest.",
        diegetic_ui_strategy="Render warm parchment crest frame with cyan hydro-pressure gauge and concise objective banner.",
    )
    assert list(GameProductionStrategySpec.model_fields.keys())[0] == "deliberation"
    assert list(SceneCritiqueAndStrategySpec.model_fields.keys())[0] == "deliberation"

    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path)
    fabricated = lib.ensure_required_assets(strat.required_visual_assets)
    assert "cliff_spire" in fabricated
    assert (tmp_path / "assets" / "sprites" / "cliff_spire.png").is_file()

    critique = SceneCritiqueAndStrategySpec(
        deliberation="Evaluated playtest_t0.png against concept art: terrace is still a flat rectangle.",
        environment_fidelity_critique="Terrace needs organic stepped cliff contour and sprite flora.",
        ui_and_experience_critique="HUD displays raw debug telemetry text instead of diegetic crest.",
        deficiencies=[
            SceneDeficiencyItem(
                deliberation="TerraceRenderer._draw uses axis-aligned Rect2 instead of sculpted island polygon.",
                target_file="scenes.gd",
                target_symbol_or_lines="TerraceRenderer._draw",
                deficiency_description="Western and Eastern shelves are drawn as boxy rectangles.",
                remediation_strategy="Replace Rect2 slab with multi-vertex rocky island underside polygon and mossy top rim.",
            )
        ],
        scene_matches_game_bible=False,
    )
    assert not critique.scene_matches_game_bible
    assert critique.deficiencies[0].target_symbol_or_lines == "TerraceRenderer._draw"


def test_extract_clean_alpha_sprite_strips_two_tone_checkerboard_and_isolates_single_subject(tmp_path: Path):
    """Verifies that extract_clean_alpha_sprite:
    1. Floods out a baked-in 2-tone alternating gray/white transparency checkerboard around the perimeter.
    2. Isolates a single primary subject when the source image contains a 3x3 variation grid or 5-pose collage.
    3. Indexes assets in assets/tiles/, assets/backgrounds/, and assets/ui/ without duplicating catalog entries.
    """
    import cv2
    import numpy as np
    from PIL import ImageDraw

    # 1. Synthesize a 256x256 image with a 16px alternating gray/white checkerboard background
    # and a 3x3 grid of 9 separate vibrant cyan/gold crystal props
    raw = Image.new("RGBA", (256, 256), (255, 255, 255, 255))
    draw = ImageDraw.Draw(raw)
    sq = 16
    for y in range(0, 256, sq):
        for x in range(0, 256, sq):
            c = (236, 236, 236, 255) if ((x // sq) + (y // sq)) % 2 == 0 else (188, 188, 188, 255)
            draw.rectangle([x, y, x + sq - 1, y + sq - 1], fill=c)

    for gy in (52, 128, 204):
        for gx in (52, 128, 204):
            r = 18 if (gx == 128 and gy == 128) else 15
            draw.polygon(
                [(gx, gy - r), (gx + r, gy), (gx, gy + r), (gx - r, gy)],
                fill=(30, 210, 240, 255),
                outline=(245, 200, 50, 255),
            )

    cleaned = VisualArtSpriteDesignerAgent.extract_clean_alpha_sprite(raw, target_size=(64, 64))
    assert cleaned.size == (64, 64)
    arr = np.array(cleaned, dtype=np.uint8)
    # All 4 corners must be transparent (2-tone checkerboard stripped)
    assert arr[0, 0, 3] == 0
    assert arr[0, 63, 3] == 0
    assert arr[63, 0, 3] == 0
    assert arr[63, 63, 3] == 0
    # Center must be opaque (the single isolated subject scaled to fill the 64x64 cell)
    assert arr[32, 32, 3] > 200
    fg_mask = (arr[:, :, 3] > 24).astype(np.uint8)
    num_cc, _, _, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    # Exactly 1 foreground connected component (plus background label 0 -> num_cc == 2)
    assert num_cc == 2

    # 2. Synthesize a 5-pose horizontal collage on a magenta backdrop and verify single-subject isolation
    collage = Image.new("RGBA", (256, 256), (255, 0, 255, 255))
    cdraw = ImageDraw.Draw(collage)
    for idx, cx in enumerate((32, 80, 128, 176, 224)):
        h_half = 26 if idx == 2 else 22
        cdraw.ellipse([cx - 12, 128 - h_half, cx + 12, 128 + h_half], fill=(220, 70, 40, 255))

    cleaned_collage = VisualArtSpriteDesignerAgent.extract_clean_alpha_sprite(collage, target_size=(64, 64))
    carr = np.array(cleaned_collage, dtype=np.uint8)
    assert carr[0, 0, 3] == 0
    assert carr[32, 32, 3] > 200
    num_cc_collage, _, _, _ = cv2.connectedComponentsWithStats((carr[:, :, 3] > 24).astype(np.uint8), connectivity=8)
    assert num_cc_collage == 2

    # 3. Verify CartridgeAssetLibrary and LeadGameplaySWEAgent._discover_sealed_assets index tiles, backgrounds, and ui
    tiles_dir = tmp_path / "assets" / "tiles" / "sub"
    bg_dir = tmp_path / "assets" / "backgrounds"
    ui_dir = tmp_path / "assets" / "ui"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)
    ui_dir.mkdir(parents=True, exist_ok=True)

    cleaned.save(tiles_dir / "floor_strip.png", "PNG")
    cleaned.save(bg_dir / "vista_frame.png", "PNG")
    cleaned.save(ui_dir / "hud_gauge.png", "PNG")

    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path)
    summary = lib.build_prompt_summary_for_swe()
    assert "floor_strip" in lib.catalog.entries
    assert "vista_frame" in lib.catalog.entries
    assert "hud_gauge" in lib.catalog.entries
    assert "res://assets/tiles/sub/floor_strip.png" in summary

    from engine.studio.swe import LeadGameplaySWEAgent
    swe = LeadGameplaySWEAgent(client=None)
    discovered = swe._discover_sealed_assets(tmp_path)
    assert "assets/tiles/sub/floor_strip.png" in discovered["sprites"]
    assert "assets/backgrounds/vista_frame.png" in discovered["sprites"]
    assert "assets/ui/hud_gauge.png" in discovered["sprites"]


def test_full_size_masters_and_zero_api_call_context_variants(tmp_path: Path):
    """Verifies that live Gemini synthesis preserves full-resolution 1K/2K masters under assets/masters/,
    produces HiDPI runtime textures without 256x256 bottlenecks, and derives smaller context variants
    (world_1x, hud_icon, custom contexts) from the saved masters with zero extra Gemini calls.
    """
    import io
    from types import SimpleNamespace
    from PIL import ImageDraw
    from engine.studio.models import VisualMockupArtifact

    call_log: list[str] = []

    def _encode_png(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # Build synthetic 1024x1024 sprite, 1024x1024 tile, 1024x1024 portrait, and 2752x1536 2K mockup
    sprite_1k = Image.new("RGBA", (1024, 1024), (255, 0, 255, 255))
    s_draw = ImageDraw.Draw(sprite_1k)
    s_draw.ellipse([180, 160, 844, 864], fill=(35, 195, 235, 255), outline=(245, 210, 75, 255), width=18)
    s_draw.rectangle([420, 320, 604, 704], fill=(250, 140, 50, 255))
    sprite_1k_bytes = _encode_png(sprite_1k)

    tile_1k = Image.new("RGBA", (1024, 1024), (36, 48, 68, 255))
    t_draw = ImageDraw.Draw(tile_1k)
    for idx in range(0, 1024, 64):
        t_draw.line([(idx, 0), (min(1023, idx + 96), 1023)], fill=(68, 96, 132, 255), width=6)
        t_draw.ellipse([idx % 900, (idx * 3) % 900, (idx % 900) + 48, ((idx * 3) % 900) + 36], fill=(92, 130, 168, 255))
    tile_1k_bytes = _encode_png(tile_1k)

    portrait_1k = Image.new("RGBA", (1024, 1024), (0, 0, 0, 255))
    p_draw = ImageDraw.Draw(portrait_1k)
    p_draw.ellipse([220, 140, 804, 880], fill=(80, 175, 220, 255), outline=(240, 220, 150, 255), width=14)
    portrait_1k_bytes = _encode_png(portrait_1k)

    mockup_2k = Image.new("RGBA", (2752, 1536), (18, 24, 42, 255))
    m_draw = ImageDraw.Draw(mockup_2k)
    m_draw.rectangle([200, 180, 2552, 1356], outline=(70, 210, 245, 255), width=12)
    mockup_2k_bytes = _encode_png(mockup_2k)

    class _FakeGenaiModels:
        def generate_content(self, *, model, contents, config=None):
            prompt_str = str(contents)
            if config is not None and getattr(config, "response_mime_type", None) == "application/json":
                schema_name = getattr(getattr(config, "response_schema", None), "__name__", "")
                if "Animation" in schema_name:
                    return SimpleNamespace(
                        text=json.dumps(
                            {
                                "deliberation": "All articulated keyframes align cleanly.",
                                "target_id": "rotor_arm_rig",
                                "accepted": True,
                            }
                        )
                    )
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "deliberation": "Clean high-resolution asset with crisp silhouette.",
                            "asset_id": "test_asset",
                            "accepted": True,
                        }
                    )
                )

            call_log.append(prompt_str[:80])
            img_cfg = getattr(config, "image_config", None)
            if img_cfg is not None and getattr(img_cfg, "image_size", None) == "2K":
                data = mockup_2k_bytes
            elif "dialogue bust portrait" in prompt_str:
                data = portrait_1k_bytes
            elif "seamless game material texture swatch" in prompt_str:
                data = tile_1k_bytes
            else:
                data = sprite_1k_bytes

            part = SimpleNamespace(inline_data=SimpleNamespace(data=data))
            cand = SimpleNamespace(content=SimpleNamespace(parts=[part]))
            return SimpleNamespace(candidates=[cand], text="")

    class _FakeLiveClient:
        __module__ = "google.genai.client"

        def __init__(self):
            self.models = _FakeGenaiModels()

    live_client = _FakeLiveClient()
    visual_agent = VisualArtSpriteDesignerAgent(client=live_client)
    assert visual_agent._is_live_genai_client() is True

    # 1. Concept mockup preserves full native 2K (2752x1536) by default
    mockup_out = tmp_path / "assets" / "mockups" / "concept_1.png"
    mockup_master = tmp_path / "assets" / "masters" / "mockups" / "concept_1.png"
    visual_agent.generate_concept_mockup(
        mockup=VisualMockupArtifact(
            deliberation="Full 2K concept art verification.",
            mockup_id="concept_1",
            title="Concept Plate",
            scene_description="Wide architectural vista with central apparatus",
            image_path="assets/mockups/concept_1.png",
        ),
        output_path=mockup_out,
        master_output_path=mockup_master,
    )
    with Image.open(mockup_out) as im_m:
        assert im_m.size == (2752, 1536)
    with Image.open(mockup_master) as im_mm:
        assert im_mm.size == (2752, 1536)

    # 2. Fabricate sprite, seamless tile, portrait, and articulated rig via CartridgeAssetLibrary
    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path, visual_agent=visual_agent, client=live_client)
    reqs = [
        AssetRequirement(
            deliberation="World structure prop with 96x96 logical footprint.",
            asset_id="core_apparatus",
            asset_type="prop",
            fabrication_mode="static_sprite",
            target_path="assets/sprites/core_apparatus.png",
            dimensions=(128, 128),
            world_footprint_px=(96, 96),
            description="Precision brass and cyan glass apparatus",
            visual_prompt_or_text="Precision brass and cyan glass apparatus",
        ),
        AssetRequirement(
            deliberation="Seamless floor surface tile.",
            asset_id="deck_plating",
            asset_type="tile",
            fabrication_mode="tile_surface",
            target_path="assets/tiles/deck_plating.png",
            dimensions=(128, 128),
            description="Brushed slate deck plating",
            visual_prompt_or_text="Brushed slate deck plating",
        ),
        AssetRequirement(
            deliberation="Operator bust portrait.",
            asset_id="operator_portrait",
            asset_type="portrait",
            fabrication_mode="portrait",
            target_path="assets/portraits/operator_portrait.png",
            dimensions=(256, 256),
            description=" calm engineer bust",
            visual_prompt_or_text="Calm engineer bust",
        ),
        AssetRequirement(
            deliberation="Articulated rotor assembly with small logical part footprint.",
            asset_id="rotor_arm_rig",
            asset_type="articulated_rig",
            fabrication_mode="articulated_parts",
            target_path="assets/sprites/rotor_arm_rig.png",
            dimensions=(80, 80),
            description="Articulated rotor mechanism",
            articulated_parts=[
                ArticulatedPartSpec(
                    deliberation="Base hub",
                    part_id="hub",
                    visual_prompt="Circular hub mount",
                    dimensions=(48, 48),
                ),
                ArticulatedPartSpec(
                    deliberation="Rotating arm",
                    part_id="arm",
                    parent_part_id="hub",
                    visual_prompt="Narrow mechanical rotor arm",
                    attach_offset_px=(0.0, -14.0),
                    dimensions=(20, 40),
                ),
            ],
        ),
    ]
    lib.fabricate_assets(reqs, force_regenerate=True)

    # Verify full-size 1024x1024 masters exist on disk and runtime textures are HiDPI-ready
    prop_entry = lib.catalog.entries["core_apparatus"]
    assert prop_entry.master_file_path == "assets/masters/sprites/core_apparatus.png"
    assert prop_entry.master_pixel_size == (1024, 1024)
    assert prop_entry.target_display_size == (96, 96)
    assert prop_entry.texture_pixel_size == (256, 256)
    assert "world_1x" in prop_entry.variants
    assert prop_entry.variants["world_1x"]["pixel_size"] == [96, 96]
    assert "hud_icon" in prop_entry.variants
    assert prop_entry.variants["hud_icon"]["pixel_size"] == [64, 64]

    tile_entry = lib.catalog.entries["deck_plating"]
    assert tile_entry.master_file_path == "assets/masters/tiles/deck_plating.png"
    assert tile_entry.master_pixel_size == (1024, 1024)
    assert tile_entry.texture_pixel_size == (512, 512)
    assert tile_entry.target_display_size == (128, 128)
    assert tile_entry.variants["world_1x"]["pixel_size"] == [128, 128]

    portrait_entry = lib.catalog.entries["operator_portrait"]
    assert portrait_entry.master_file_path == "assets/masters/portraits/operator_portrait.png"
    assert portrait_entry.master_pixel_size == (1024, 1024)
    assert portrait_entry.texture_pixel_size == (512, 512)
    assert portrait_entry.variants["world_1x"]["pixel_size"] == [128, 128]
    assert portrait_entry.variants["hud_icon"]["pixel_size"] == [64, 64]

    rig_entry = lib.catalog.entries["rotor_arm_rig"]
    assert rig_entry.master_file_path == "assets/masters/sprites/rotor_arm_rig.png"
    assert rig_entry.master_pixel_size[0] >= 512
    assert rig_entry.target_display_size == (80, 80)
    assert rig_entry.texture_pixel_size[0] >= 256
    arm_meta = next(p for p in rig_entry.articulated_parts if p["part_id"] == "arm")
    assert arm_meta["dimensions"] == [20, 40]  # Logical rig coordinates preserved
    assert arm_meta["texture_pixel_size"] == [80, 160]  # 4x HiDPI runtime part texture (no longer tiny 20x40!)
    arm_master_path = tmp_path / arm_meta["master_file_path"]
    assert arm_master_path.is_file()
    with Image.open(arm_master_path) as arm_m_im:
        assert arm_m_im.size == (512, 1024)  # Full-res aspect-matched 1K part master

    # 3. Derive a new custom context variant on-demand with ZERO Gemini API calls
    calls_before = len(call_log)
    pip_variant = lib.derive_context_variant("core_apparatus", "minimap_pip", (32, 32))
    assert len(call_log) == calls_before  # Strictly 0 image generation calls!
    assert pip_variant.is_file()
    with Image.open(pip_variant) as pip_im:
        assert pip_im.size == (32, 32)
    assert lib.catalog.entries["core_apparatus"].variants["minimap_pip"]["pixel_size"] == [32, 32]

    # 4. StudioRigHelper exports load_context_texture with mipmap generation & SWE summary lists masters + variants
    (tmp_path / "project.godot").write_text("[application]\n", encoding="utf-8")
    swe_summary = lib.build_prompt_summary_for_swe()
    assert "StudioRigHelper.load_context_texture" in swe_summary
    assert "assets/masters/sprites/core_apparatus.png" in swe_summary
    assert "minimap_pip" in swe_summary
    helper_gd = (tmp_path / "src" / "studio_rig_and_collider.gd").read_text(encoding="utf-8")
    assert "static func load_context_texture(" in helper_gd
    assert "img.generate_mipmaps()" in helper_gd




