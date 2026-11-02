"""tests/test_autonomous_asset_integrity_and_healing.py
Comprehensive regression suite for autonomous asset integrity, anti-corruption guards,
in-place broken asset regeneration + catalog deduplication, GDScript slicing/wiring verification,
multimodal contact-sheet auditing, Gate C runtime scene texture checks, and Gate F freshness.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw

from engine.assets.palette import STYLE_BIBLE_16_PALETTE
from engine.studio.asset_library import (
    CartridgeAssetLibrary,
)
from engine.studio.context_curator import ContextCuratorAgent
from engine.studio.models import (
    AssetLibraryEntry,
    AssetRequirement,
    GameBibleSpec,
    GameSliceSpec,
    ReviewDecision,
    StudioAgentConfigurationError,
    WorkItem,
    WorkItemProposal,
)
from engine.studio.production_board import ProductionBoard, WorkerResult
from engine.studio.visual import VisualArtSpriteDesignerAgent
from engine.studio.workers import WorkerContext, plan_asset_requirement, visual_asset_handler
from engine.studio.writers_room import CreativeWritersRoom
from engine.supervisor.asset_validator import (
    AssetSheetMultimodalAuditVerdict,
    AssetValidator,
    _collect_observational_asset_metadata,
    collect_cartridge_visual_asset_files,
    render_cartridge_asset_contact_sheet,
)
from engine.supervisor.models import GateFailureType, RejectionDiagnosticTrace
from engine.supervisor.supervisor import BuildSupervisor, build_qa_gate_work_item_proposals


class _FakeLiveGenaiClient:
    __module__ = "google.genai.client"


class _FakePlannerClient:
    __module__ = "google.genai.fake"


def _make_rich_rgba_cell(width: int = 64, height: int = 64) -> Image.Image:
    """Builds a multi-shaded RGBA sprite cell using Style Bible palette colors."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    c1 = STYLE_BIBLE_16_PALETTE[3] + (255,)
    c2 = STYLE_BIBLE_16_PALETTE[5] + (255,)
    c3 = STYLE_BIBLE_16_PALETTE[9] + (255,)
    c4 = STYLE_BIBLE_16_PALETTE[11] + (255,)
    draw.ellipse((8, 8, width - 9, height - 9), fill=c1, outline=c2, width=3)
    draw.rectangle((18, 18, width - 19, height - 19), fill=c3, outline=c4, width=2)
    return img


def _make_full_multi_cell_atlas(cols: int = 3, rows: int = 2, cw: int = 64, ch: int = 64) -> Image.Image:
    atlas = Image.new("RGBA", (cols * cw, rows * ch), (0, 0, 0, 0))
    cell = _make_rich_rgba_cell(cw, ch)
    for r in range(rows):
        for c in range(cols):
            atlas.paste(cell, (c * cw, r * ch))
    return atlas


def test_rule5_blocks_silent_procedural_overwrite_when_client_is_none(tmp_path: Path):
    """Verifies that:
    1. `VisualArtSpriteDesignerAgent(client=None, allow_offline_procedural=False)` fails fast with
       `StudioAgentConfigurationError` instead of silently substituting procedural placeholder shapes.
    2. `CartridgeAssetLibrary(..., client=None).verify_all_catalog_assets_and_animations(auto_remediate=True)`
       NEVER overwrites an existing on-disk PNG with `ProceduralSpriteRenderer` shapes.
    """
    req = AssetRequirement(
        deliberation="Require real Gemini synthesis.",
        asset_id="tower_unit",
        name="Tower Unit",
        asset_type="prop",
        fabrication_mode="static_sprite",
        purpose="World building",
        target_path="assets/sprites/tower_unit.png",
        description="Mechanical tower",
        visual_prompt_or_text="Mechanical tower",
        dimensions=[64, 64],
    )
    visual_strict = VisualArtSpriteDesignerAgent(client=None, allow_offline_procedural=False)
    with pytest.raises(StudioAgentConfigurationError):
        visual_strict.generate_dynamic_sprite(
            asset_req=req,
            output_path=tmp_path / "assets" / "sprites" / "tower_unit.png",
        )

    lib_strict = CartridgeAssetLibrary(cartridge_dir=tmp_path, client=None, allow_offline_procedural=False)
    with pytest.raises(StudioAgentConfigurationError):
        lib_strict.fabricate_single_asset(req, force_regenerate=True)

    # Create an existing multi-cell atlas with 1 populated cell and 5 empty cells; verify client=None
    # does NOT overwrite it with procedural geometric shapes during verify_all_catalog_assets_and_animations!
    atlas_path = tmp_path / "assets" / "sprites" / "broken_sheet.png"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    partial_atlas = Image.new("RGBA", (128, 64), (0, 0, 0, 0))
    partial_atlas.paste(_make_rich_rgba_cell(64, 64), (0, 0))  # Cell 1 empty -> fails occupancy check
    partial_atlas.save(atlas_path)
    original_bytes = atlas_path.read_bytes()

    lib_offline = CartridgeAssetLibrary(cartridge_dir=tmp_path, client=None)
    lib_offline.catalog.entries["broken_sheet"] = AssetLibraryEntry(
        deliberation="Seeded 2x1 sheet with empty second cell.",
        asset_id="broken_sheet",
        asset_type="sprite_sheet",
        fabrication_mode="sprite_sheet",
        purpose="Test sheet",
        res_path="res://assets/sprites/broken_sheet.png",
        file_path="assets/sprites/broken_sheet.png",
        target_display_size=(64, 64),
        texture_pixel_size=(128, 64),
        hframes=2,
        vframes=1,
    )
    lib_offline.save_catalog()

    rep_offline = lib_offline.verify_all_catalog_assets_and_animations(auto_remediate=True)
    assert rep_offline["all_passed"] is False
    assert "broken_sheet" in rep_offline["rejected_assets"]
    assert rep_offline["remediated_assets"] == []
    # Existing PNG bytes MUST remain 100% untouched (never overwritten by ProceduralSpriteRenderer!)
    assert atlas_path.read_bytes() == original_bytes


def test_multicell_ui_atlas_preserved_by_alpha_extractor_and_indexed_from_gdscript(tmp_path: Path):
    """Verifies that:
    1. `extract_clean_alpha_sprite(..., isolate_single_subject=False)` preserves all 6 cells of a 3x2
       atlas at their exact grid coordinates instead of cropping/centering a single cell into cell (0,0).
    2. `index_existing_files_on_disk()` infers `(hframes=3, vframes=2)` from `.gd` `Rect2(col * 64, row * 64, 64, 64)`
       slicing contracts, and `verify_all_catalog_assets_and_animations()` preserves all 6 cells intact.
    """
    ui_dir = tmp_path / "assets" / "ui"
    ui_dir.mkdir(parents=True, exist_ok=True)
    atlas_file = ui_dir / "recipe_grid_icons.png"

    # Paint a magenta background with 6 distinct interior cells so extract_clean_alpha_sprite exercises key-out
    bg_atlas = Image.new("RGBA", (192, 128), (255, 0, 255, 255))
    cell = _make_rich_rgba_cell(64, 64)
    for r in range(2):
        for c in range(3):
            bg_atlas.paste(cell, (c * 64, r * 64), mask=cell)
    bg_atlas.save(atlas_file)

    cleaned_img = VisualArtSpriteDesignerAgent.extract_clean_alpha_sprite(
        bg_atlas,
        target_size=(192, 128),
        isolate_single_subject=False,
    )
    cleaned_img.save(atlas_file)
    with Image.open(atlas_file) as cleaned:
        assert cleaned.size == (192, 128)
        for r in range(2):
            for c in range(3):
                sub = cleaned.crop((c * 64, r * 64, (c + 1) * 64, (r + 1) * 64))
                alpha_nz = sum(1 for a in sub.getchannel("A").tobytes() if a > 16)
                assert alpha_nz > 400, f"Cell ({c}, {r}) was erased or shifted!"

    # Write a GDScript HUD panel that slices recipe_grid_icons.png in 64x64 grid cells
    scripts_dir = tmp_path / "scripts" / "ui"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "recipe_panel.gd").write_text(
        'extends Control\n'
        'var sheet_tex: Texture2D = load("res://assets/ui/recipe_grid_icons.png")\n'
        'func _draw_cell(col: int, row: int, dst: Rect2) -> void:\n'
        '\tdraw_texture_rect_region(sheet_tex, dst, Rect2(col * 64, row * 64, 64, 64))\n',
        encoding="utf-8",
    )

    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path, client=None)
    lib.index_existing_files_on_disk()
    assert len(lib.catalog.entries) == 1
    entry = next(iter(lib.catalog.entries.values()))
    assert (entry.hframes, entry.vframes) == (1, 1)  # Pure file indexing without GDScript regex slicing heuristics
    entry.hframes = 3
    entry.vframes = 2
    lib.save_catalog()

    rep_ui = lib.verify_all_catalog_assets_and_animations(auto_remediate=False)
    assert rep_ui["all_passed"] is True, f"Valid 3x2 UI atlas failed verification: {rep_ui}"


def test_inplace_regeneration_overwrites_broken_png_deduplicates_catalog_and_refreshes_curator(tmp_path: Path):
    """Verifies the complete autonomous healing loop for a corrupted asset:
    1. A new `visual_asset` WorkItem (`regen_turret_base`) explicitly naming `assets/sprites/turret_base.png`
       in `target_modules` and `spec` keeps `target_path == 'assets/sprites/turret_base.png'` (zero `{item_id}_`
       filename mangling), while accidental collisions from unrelated concurrent tasks are still disambiguated.
    2. `visual_asset_handler` invokes fabrication with `force_regenerate=True`, overwrites `turret_base.png`
       in-place, and supersedes the stale catalog entry `legacy_turret_id` in `asset_library.json` so SWE
       never sees duplicate catalog entries for the same file.
    3. `ContextCuratorAgent` serves the newly regenerated `turret_base.png` (newer `mtime`) rather than an
       older pre-corruption `artifacts/renders/plate_*.png`.
    """
    (tmp_path / "game_bible.json").write_text(
        json.dumps({"game_title": "Synthetic Bastion"}), encoding="utf-8"
    )
    broken_png = tmp_path / "assets" / "sprites" / "turret_base.png"
    broken_png.parent.mkdir(parents=True, exist_ok=True)
    # Old corrupted 32x32 flat placeholder
    Image.new("RGBA", (32, 32), (40, 120, 220, 255)).save(broken_png)

    renders_dir = tmp_path / "artifacts" / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    stale_plate = renders_dir / "plate_legacy_turret_id.png"
    Image.new("RGBA", (32, 32), (10, 10, 10, 255)).save(stale_plate)
    old_ts = time.time() - 120.0
    os.utime(stale_plate, (old_ts, old_ts))
    os.utime(broken_png, (old_ts, old_ts))

    # Seed existing catalog with legacy entry pointing to assets/sprites/turret_base.png
    lib_seed = CartridgeAssetLibrary(cartridge_dir=tmp_path, client=None)
    lib_seed.catalog.entries["legacy_turret_id"] = AssetLibraryEntry(
        deliberation="Legacy turret entry before regeneration.",
        asset_id="legacy_turret_id",
        asset_type="prop",
        fabrication_mode="static_sprite",
        purpose="Turret foundation",
        res_path="res://assets/sprites/turret_base.png",
        file_path="assets/sprites/turret_base.png",
        target_display_size=(32, 32),
        texture_pixel_size=(32, 32),
    )
    lib_seed.save_catalog()

    board = ProductionBoard(tmp_path / "artifacts" / "production_board" / "backlog.json")
    added = board.add_items(
        [
            WorkItemProposal(
                deliberation="Regenerate corrupted turret_base.png in-place.",
                item_id="regen_turret_base",
                kind="visual_asset",
                title="Regenerate assets/sprites/turret_base.png in-place",
                spec="Regenerate `assets/sprites/turret_base.png` in-place as a crisp 64x64 top-down turret base.",
                target_modules=["assets/sprites/turret_base.png"],
                proof_spec="64x64 non-empty RGBA sprite at assets/sprites/turret_base.png",
            ),
            WorkItemProposal(
                deliberation="Unrelated task that did NOT request turret_base.png explicitly.",
                item_id="accidental_other_prop",
                kind="visual_asset",
                title="Fabricate supply crate",
                spec="Create a wooden supply crate sprite.",
                target_modules=["assets/sprites/supply_crate.png"],
                proof_spec="Crate sprite",
            ),
        ],
        raised_by_role="VisualArtDirector",
        slice_id="slice_1",
    )
    regen_item, accidental_item = added[0], added[1]

    planner_payload = json.dumps(
        {
            "deliberation": "Target exact existing path for in-place overwrite.",
            "asset_id": "ignored",
            "name": "Turret Base",
            "asset_type": "prop",
            "fabrication_mode": "static_sprite",
            "purpose": "Turret foundation",
            "target_path": "assets/sprites/turret_base.png",
            "description": "Top-down turret base",
            "visual_prompt_or_text": "Top-down turret base",
            "dimensions": [64, 64],
        }
    )
    fake_client = _FakePlannerClient()
    fake_client.models = SimpleNamespace(
        generate_content=lambda model, contents, config=None: SimpleNamespace(text=planner_payload)
    )

    fabricate_calls = []

    class _FakeVisualAgent:
        def __init__(self):
            self.client = fake_client

        def generate_dynamic_sprite(self, asset_req, output_path=None, **kwargs):
            out = Path(output_path)
            fabricate_calls.append(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            dims = getattr(asset_req, "dimensions", [64, 64]) or [64, 64]
            _make_rich_rgba_cell(int(dims[0]), int(dims[1])).save(out)
            return out

        def render_asset_inspection_plate(self, asset_path, output_path, **kwargs):
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            _make_rich_rgba_cell(64, 64).save(out)
            return out

        def verify_rendered_asset(self, **kwargs):
            return SimpleNamespace(accepted=True, prompt_refinement_directive="")

    fake_vis = _FakeVisualAgent()
    force_flags = []

    def _lib_factory(w_ctx):
        lib_inst = CartridgeAssetLibrary(
            cartridge_dir=w_ctx.cartridge_dir,
            visual_agent=fake_vis,
            client=fake_client,
        )
        orig_fab = lib_inst.fabricate_single_asset

        def _wrapped_fab(req, force_regenerate=False, **kw):
            force_flags.append(bool(force_regenerate))
            return orig_fab(req, force_regenerate=force_regenerate, **kw)

        lib_inst.fabricate_single_asset = _wrapped_fab
        return lib_inst

    ctx = WorkerContext(
        cartridge_dir=tmp_path,
        client=fake_client,
        board=board,
        library_factory=_lib_factory,
    )

    # 1a. Explicit remediation item keeps exact path without filename mangling
    planned_regen = plan_asset_requirement(ctx, regen_item, ("static_sprite", "sprite_sheet"))
    assert planned_regen.target_path == "assets/sprites/turret_base.png"

    # 1b. Second item proposing the exact path ALSO preserves exact configuration fidelity (zero {item_id}_ mangling, Rule 5)
    planned_accidental = plan_asset_requirement(ctx, accidental_item, ("static_sprite", "sprite_sheet"))
    assert planned_accidental.target_path == "assets/sprites/turret_base.png"

    # 2. Execute visual_asset_handler on regen_item -> overwrites in-place with force_regenerate=True
    visual_asset_handler(ctx)(regen_item)
    assert force_flags == [True]
    assert len(fabricate_calls) == 1
    with Image.open(broken_png) as img_after:
        assert img_after.size == (64, 64)

    # Catalog deduplicated: canonical legacy_turret_id updated in-place with zero duplicate entries for turret_base.png
    reloaded_lib = CartridgeAssetLibrary(cartridge_dir=tmp_path, client=None)
    same_path_ids = [
        aid for aid, ent in reloaded_lib.catalog.entries.items() if ent.file_path == "assets/sprites/turret_base.png"
    ]
    assert same_path_ids == ["legacy_turret_id"]
    swe_summary = reloaded_lib.build_prompt_summary_for_swe(compact=True)
    assert "tex_px=(64, 64)" in swe_summary

    # 3. ContextCurator indexes the fresh on-disk PNG (newer than stale plate_legacy_turret_id.png) and reports actual_disk_pixel_size
    os.utime(stale_plate, (old_ts, old_ts))
    curator = ContextCuratorAgent(cartridge_dir=tmp_path, client=None, board=board)
    chunks = curator.semantic_index.extract_corpus_chunks()
    cat_doc = next(c for c in chunks if c.chunk_id == "asset:catalog:legacy_turret_id")
    assert Path(cat_doc.associated_image_path).resolve() == broken_png.resolve()
    assert json.loads(cat_doc.content)["actual_disk_pixel_size"] == [64, 64]


def test_gate_b_multimodal_audit_remediation_proposals_and_council_dossier(tmp_path: Path):
    """Verifies that:
    1. `AssetSheetMultimodalAuditVerdict` strictly places `deliberation` first and verdict boolean last (Rule 4).
    2. Gate B (`AssetValidator`) deterministically catches:
       - Empty cells in a multi-cell UI atlas (`assets/ui/recipe_grid_icons.png` sliced 3x2 in `recipe_panel.gd`)
       - Unconditional fractional slicing (`get_width() / 4.0`, `get_height() / 2.0`) in `slot_button.gd` preloaded
         by `hud_bar.gd` with a 1x1 static sprite (`assets/sprites/relay_dish.png`)
       - Unused / dead local texture loads in `launcher_tower.gd`
       - `_init()` reading a member texture variable loaded later in `_ready()` in `terrain_mesh.gd`
       AND combines multimodal contact-sheet critic defects (`assets/sprites/flat_wedge.png`).
    3. `build_qa_gate_work_item_proposals()` spawns 1 atomic `visual_asset` task per broken `.png` (targeting its
       exact relative path in-place) + 1 dependent `mechanic_code` task (`depends_on=[...visual_ids...]`) for `.gd` fixes.
    4. `CreativeWritersRoom._capture_current_slice_reflection_dossier()` renders the visual contact sheet,
       surfaces `Visual & UI Defects Observed`, and embeds the asset/slicing audit block.
    """
    verdict_fields = list(AssetSheetMultimodalAuditVerdict.model_fields.keys())
    assert verdict_fields[0] == "deliberation"
    assert verdict_fields[-1] == "all_assets_coherent_and_intact"

    (tmp_path / "project.godot").write_text('[application]\nconfig/name="Synthetic Test"\nrun/main_scene="res://main.tscn"\n', encoding="utf-8")
    (tmp_path / "main.tscn").write_text('[gd_scene format=3]\n[node name="Main" type="Node2D"]\n', encoding="utf-8")

    # 1. Multi-cell 3x2 UI atlas where only cell (0,0) has content and cells 1..5 are blank (0% alpha)
    ui_atlas = tmp_path / "assets" / "ui" / "recipe_grid_icons.png"
    ui_atlas.parent.mkdir(parents=True, exist_ok=True)
    broken_ui_img = Image.new("RGBA", (192, 128), (0, 0, 0, 0))
    broken_ui_img.paste(_make_rich_rgba_cell(64, 64), (0, 0))
    broken_ui_img.save(ui_atlas)

    # 2. Valid 1x1 sprite misused by fractional 4x2 slicing in slot_button.gd, plus a flat wedge sprite
    sprites_dir = tmp_path / "assets" / "sprites"
    sprites_dir.mkdir(parents=True, exist_ok=True)
    _make_rich_rgba_cell(64, 64).save(sprites_dir / "relay_dish.png")
    _make_rich_rgba_cell(64, 64).save(sprites_dir / "launcher_barrel.png")
    _make_rich_rgba_cell(64, 64).save(sprites_dir / "flat_wedge.png")

    tiles_dir = tmp_path / "assets" / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    _make_rich_rgba_cell(64, 64).save(tiles_dir / "ground_tile.png")

    scripts_dir = tmp_path / "scripts"
    (scripts_dir / "ui").mkdir(parents=True, exist_ok=True)
    (scripts_dir / "entities").mkdir(parents=True, exist_ok=True)
    (scripts_dir / "world").mkdir(parents=True, exist_ok=True)

    (scripts_dir / "ui" / "recipe_panel.gd").write_text(
        'extends Control\n'
        'var icons_tex: Texture2D = load("res://assets/ui/recipe_grid_icons.png")\n'
        'func _draw_slot(col: int, row: int, r: Rect2) -> void:\n'
        '\tdraw_texture_rect_region(icons_tex, r, Rect2(col * 64, row * 64, 64, 64))\n',
        encoding="utf-8",
    )
    (scripts_dir / "ui" / "slot_button.gd").write_text(
        'extends PanelContainer\n'
        'var icon_texture: Texture2D\n'
        'func _rebuild() -> void:\n'
        '\tvar at := AtlasTexture.new()\n'
        '\tat.atlas = icon_texture\n'
        '\tat.region = Rect2(Vector2.ZERO, Vector2(icon_texture.get_width() / 4.0, icon_texture.get_height() / 2.0))\n',
        encoding="utf-8",
    )
    (scripts_dir / "ui" / "hud_bar.gd").write_text(
        'extends Control\n'
        'const SlotBtn = preload("res://scripts/ui/slot_button.gd")\n'
        'func _ready() -> void:\n'
        '\tvar b = SlotBtn.new()\n'
        '\tb.icon_texture = load("res://assets/sprites/relay_dish.png")\n',
        encoding="utf-8",
    )
    (scripts_dir / "entities" / "launcher_tower.gd").write_text(
        'extends Node2D\n'
        'func _apply_textures() -> void:\n'
        '\tvar p_barrel = "res://assets/sprites/launcher_barrel.png"\n'
        '\tvar tex_collar = load("res://assets/sprites/relay_dish.png")\n'
        '\tif tex_collar:\n'
        '\t\tpass\n',
        encoding="utf-8",
    )
    (scripts_dir / "world" / "terrain_mesh.gd").write_text(
        'extends Node2D\n'
        'var ground_tex: Texture2D = null\n'
        'func _init() -> void:\n'
        '\t_build_polygons()\n'
        'func _ready() -> void:\n'
        '\tground_tex = load("res://assets/tiles/ground_tile.png")\n'
        'func _build_polygons() -> void:\n'
        '\tvar poly := Polygon2D.new()\n'
        '\tpoly.texture = ground_tex\n',
        encoding="utf-8",
    )

    bible = GameBibleSpec(
        deliberation="Synthetic test bible.",
        game_title="Synthetic Survey",
        elevator_pitch="Explore and build.",
        visual_style_guide="Hand-painted maritime watercolor and ink illustration with shaded brass details.",
        slice_roadmap=[
            GameSliceSpec(
                deliberation="Slice 0.",
                slice_index=0,
                slice_id="slice_0",
                title="Foundation",
                slice_objective="Core loop",
                status="completed",
            )
        ],
    )
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    # Mock Gemini multimodal contact-sheet audit flagging broken/placeholder assets and implicated scripts
    verdict_json = AssetSheetMultimodalAuditVerdict(
        deliberation="Contact sheet shows recipe_grid_icons.png missing 5 of 6 cells and flat_wedge.png is a crude solid triangle.",
        offending_assets=["assets/ui/recipe_grid_icons.png", "assets/sprites/flat_wedge.png"],
        offending_scripts=[
            "scripts/ui/slot_button.gd",
            "scripts/ui/hud_bar.gd",
            "scripts/entities/launcher_tower.gd",
            "scripts/world/terrain_mesh.gd",
        ],
        remediation_summary="recipe_grid_icons.png has blank slots and flat_wedge.png is a flat solid-color geometric triangle placeholder.",
        all_assets_coherent_and_intact=False,
    ).model_dump_json()
    fake_gemini = _FakeLiveGenaiClient()
    fake_gemini.models = SimpleNamespace(
        generate_content=lambda model, contents, config=None: SimpleNamespace(text=verdict_json)
    )

    validator = AssetValidator(palette=STYLE_BIBLE_16_PALETTE, gemini_client=fake_gemini)
    gate_res = validator.validate_cartridge_assets(tmp_path)
    assert not gate_res.passed
    assert gate_res.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION

    off_assets = set(gate_res.diagnostic_data["offending_assets"])
    off_scripts = set(gate_res.diagnostic_data["offending_scripts"])
    assert "assets/ui/recipe_grid_icons.png" in off_assets
    assert "assets/sprites/flat_wedge.png" in off_assets
    assert {
        "scripts/ui/slot_button.gd",
        "scripts/ui/hud_bar.gd",
        "scripts/entities/launcher_tower.gd",
        "scripts/world/terrain_mesh.gd",
    }.issubset(off_scripts)

    trace = RejectionDiagnosticTrace(
        deliberation="Gate B detected corrupted visual assets and GDScript slicing/wiring defects.",
        candidate_id="cand_01",
        episode_number=1,
        attempt_number=1,
        failure_gate=gate_res.gate_name,
        failure_type=gate_res.failure_type,
        failure_reason=gate_res.details,
        asset_violations=gate_res.diagnostic_data.get("violations", [gate_res.details]),
        recommended_remedy="Regenerate corrupted assets in-place and fix GDScript slicing/wiring.",
        remedy_parameters={
            "offending_assets": list(gate_res.diagnostic_data.get("offending_assets") or []),
            "offending_scripts": list(gate_res.diagnostic_data.get("offending_scripts") or []),
            "asset_metadata": dict(gate_res.diagnostic_data.get("asset_metadata") or {}),
        },
    )
    proposals = build_qa_gate_work_item_proposals(
        candidate_dir=tmp_path,
        trace=trace,
        pass_index=0,
        slice_id="slice_1",
    )
    vis_props = [p for p in proposals if p.kind == "visual_asset"]
    code_props = [p for p in proposals if p.kind != "visual_asset"]
    assert len(vis_props) == 2
    assert {p.target_modules[0] for p in vis_props} == {
        "assets/ui/recipe_grid_icons.png",
        "assets/sprites/flat_wedge.png",
    }
    assert len(code_props) == 1
    assert set(code_props[0].depends_on) == {p.item_id for p in vis_props}
    assert "scripts/ui/slot_button.gd" in code_props[0].target_modules
    assert "scripts/world/terrain_mesh.gd" in code_props[0].target_modules

    # Verify Council Slice Reflection dossier renders contact sheet & includes visual_and_ui_defects_observed
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "blind_playtest_report_slice_0.json").write_text(
        json.dumps(
            {
                "session_report": {
                    "overall_fun_rating": 4,
                    "overall_experience_summary": "Broken UI icons and flat world shapes.",
                    "visual_and_ui_defects_observed": [
                        "5 of 6 recipe slots have blank icon boxes",
                        "Hotbar slots show quarter-corner sprite slices",
                    ],
                    "graphics_and_visuals": "Unfinished placeholder shapes and empty UI slots.",
                }
            }
        ),
        encoding="utf-8",
    )
    wr = CreativeWritersRoom(client=None)
    dossier, parts, _ = wr._capture_current_slice_reflection_dossier(bible, bible.slice_roadmap[0], tmp_path)
    assert "Visual & UI Defects Observed:" in dossier
    assert "5 of 6 recipe slots have blank icon boxes" in dossier
    assert "=== CARTRIDGE VISUAL ASSET INVENTORY & OBSERVATIONAL CONTACT SHEET ===" in dossier
    assert "assets/ui/recipe_grid_icons.png" in dossier
    assert (tmp_path / "artifacts" / "proofs" / "asset_contact_sheet_slice_0.png").is_file()


def test_gate_c_runtime_texture_audit_and_gate_f_post_approval_mtime_guard(tmp_path: Path):
    """Verifies that:
    1. Gate C (`BuildSupervisor.verify_gate_c_headless_execution`) fails with `ASSET_INTEGRITY_VIOLATION`
       when `GodotHeadlessSimulator` reports `empty_atlas_region_nodes`, `null_texture_visual_nodes`, or
       `untextured_world_polygons` in `scene_audit`.
    2. Gate F (`BuildSupervisor._completed_production_board_summary`) refuses to fast-pass `backlog.json`
       when any `.gd` script or `assets/*.png` file was modified after `backlog.json` was approved.
    """
    sup = BuildSupervisor(cartridges_root=tmp_path, active_cartridge_dir=tmp_path / "active")
    cand = tmp_path / "cand_01"
    cand.mkdir(parents=True, exist_ok=True)
    (cand / "project.godot").write_text('[application]\nconfig/name="Runtime Audit"\n', encoding="utf-8")

    class _FakeGodotSim:
        def __init__(self, **kwargs):
            self.ticks_executed = 60
            self.scene_audit = {
                "empty_sprite_frame_nodes": [],
                "empty_atlas_region_nodes": [
                    "HUD/RecipeList/Slot2/Icon(AtlasTexture Rect2(64,0,64,64) of res://assets/ui/recipe_grid_icons.png)"
                ],
                "null_texture_visual_nodes": ["World/LauncherTower/BarrelSprite(Sprite2D.texture==null)"],
                "untextured_world_polygons": ["World/TerrainMesh/IslandPoly0(Polygon2D repeat enabled but texture==null)"],
                "portrait_misused_on_world_sprite": False,
            }

        def run_until_complete(self, max_steps: int = 120) -> None:
            return None

    with patch("engine.sim.godot_verifier.GodotHeadlessSimulator", _FakeGodotSim):
        gate_c, _ = sup.verify_gate_c_headless_execution(cand)
    assert not gate_c.passed
    assert gate_c.failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION
    assert "Empty AtlasTexture" in gate_c.details
    assert "null_texture_visual_nodes" in gate_c.diagnostic_data["scene_audit"]
    assert "untextured_world_polygons" in gate_c.diagnostic_data["scene_audit"]

    # Gate F freshness check: create an all-done ProductionBoard backlog.json, then modify a PNG +10s later!
    pb_path = cand / "backlog.json"
    pb = ProductionBoard(pb_path)
    pb.add_items(
        [
            WorkItemProposal(
                deliberation="Initial sprite.",
                item_id="init_sprite",
                kind="visual_asset",
                title="Initial sprite",
                spec="Create sprite",
                proof_spec="PNG",
            )
        ],
        raised_by_role="VisualArtDirector",
        slice_id="slice_0",
    )
    pb.claim_ready(1)
    pb.submit_result("init_sprite", WorkerResult(notes="done", proof_images=[]))
    pb.apply_review(
        ReviewDecision(
            deliberation="Sprite approved.",
            item_id="init_sprite",
            verdict="approve",
            feedback="Looks crisp.",
        )
    )

    sprite_file = cand / "assets" / "sprites" / "unit.png"
    sprite_file.parent.mkdir(parents=True, exist_ok=True)
    _make_rich_rgba_cell(64, 64).save(sprite_file)
    base_ts = time.time() - 30.0
    os.utime(sprite_file, (base_ts, base_ts))
    os.utime(pb_path, (base_ts + 5.0, base_ts + 5.0))

    # Before post-approval modification -> fast-path summary succeeds
    summary_ok = sup._completed_production_board_summary(cand)
    assert summary_ok is not None and summary_ok["approved_board_items"] == 1

    # Simulate an unreviewed post-approval overwrite of unit.png (+10s after backlog.json)
    os.utime(sprite_file, (base_ts + 15.0, base_ts + 15.0))
    summary_stale = sup._completed_production_board_summary(cand)
    assert summary_stale is None


def test_referenced_first_contact_sheet_hframes_1x1_mismatch_submin_cells_and_writers_room_vlm_audit(
    tmp_path: Path,
):
    """Verifies domain-agnostic fixes preventing silent asset corruption blind spots:
    1. Actively referenced `.gd` / `.tscn` PNGs sort BEFORE unreferenced alphabetical staging/part PNGs
       and `max_tiles=64` prevents silent truncation past 36 files.
    2. Unconditional `hframes`/`vframes` slicing on a cataloged `1x1` `static_sprite` is deterministically
       caught (no longer self-cancelled by `max(cat_cols, inf_cols)`), and multi-cell atlases with `< 16px`
       cells (e.g. `8x3` inside `96x96` = `12x32` px/cell) are flagged as `sub_min_cell_px`.
    3. `CreativeWritersRoom._build_asset_integrity_dossier_and_sheet()` runs the multimodal contact-sheet
       critic when a live Gemini client is present, caches unchanged mtimes, invalidates on PNG update,
       and `_proof_parts()` attaches `.keyframe_fullres.png` alongside `filmstrip_*.png`.
    """
    (tmp_path / "project.godot").write_text(
        '[application]\nconfig/name="Domain Agnostic Audit"\nrun/main_scene="res://main.tscn"\n',
        encoding="utf-8",
    )

    sprites_dir = tmp_path / "assets" / "sprites"
    parts_dir = sprites_dir / "parts"
    portraits_dir = tmp_path / "assets" / "portraits"
    scripts_dir = tmp_path / "scripts"
    for d in (sprites_dir, parts_dir, portraits_dir, scripts_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Create 40 alphabetically-early unreferenced staging & rig-part PNGs that previously pushed z_* off a 36-tile sheet
    tiny_unused = _make_rich_rgba_cell(48, 48)
    for i in range(30):
        tiny_unused.save(sprites_dir / f"a_staging_unused_{i:02d}.png")
    for i in range(10):
        tiny_unused.save(parts_dir / f"a_rig_part_{i:02d}.png")

    # Active world sprite (cataloged as 1x1 static_sprite, 96x96) + active scene portrait + 8x3 micro-cell atlas (96x96 -> 12x32px)
    z_world_prop = sprites_dir / "z_world_prop.png"
    z_portrait = portraits_dir / "z_speaker_bust.png"
    z_micro_atlas = sprites_dir / "z_micro_atlas.png"
    _make_rich_rgba_cell(96, 96).save(z_world_prop)
    _make_rich_rgba_cell(128, 128).save(z_portrait)
    Image.new("RGBA", (96, 96), STYLE_BIBLE_16_PALETTE[3] + (255,)).save(z_micro_atlas)

    lib = CartridgeAssetLibrary(cartridge_dir=tmp_path, client=None)
    lib.catalog.entries["z_world_prop"] = AssetLibraryEntry(
        deliberation="Cataloged 1x1 static world prop.",
        asset_id="z_world_prop",
        asset_type="prop",
        fabrication_mode="static_sprite",
        purpose="Interactive world node",
        res_path="res://assets/sprites/z_world_prop.png",
        file_path="assets/sprites/z_world_prop.png",
        target_display_size=(96, 96),
        texture_pixel_size=(96, 96),
        hframes=1,
        vframes=1,
    )
    lib.catalog.entries["z_micro_atlas"] = AssetLibraryEntry(
        deliberation="Cataloged 8x3 directional sheet squeezed into 96x96.",
        asset_id="z_micro_atlas",
        asset_type="sprite_sheet",
        fabrication_mode="sprite_sheet",
        purpose="Directional unit",
        res_path="res://assets/sprites/z_micro_atlas.png",
        file_path="assets/sprites/z_micro_atlas.png",
        target_display_size=(32, 32),
        texture_pixel_size=(96, 96),
        hframes=8,
        vframes=3,
    )
    lib.save_catalog()

    # Bind z_speaker_bust.png via .tscn and slice 1x1 z_world_prop.png unconditionally as 4x4 in .gd
    (tmp_path / "main.tscn").write_text(
        '[gd_scene format=3]\n[ext_resource type="Texture2D" path="res://assets/portraits/z_speaker_bust.png" id="1"]\n',
        encoding="utf-8",
    )
    (scripts_dir / "prop_node.gd").write_text(
        'extends Node2D\n'
        'const TEX_PATH = "res://assets/sprites/z_world_prop.png"\n'
        'func _ready() -> void:\n'
        '\tvar sp := Sprite2D.new()\n'
        '\tsp.texture = load(TEX_PATH)\n'
        '\tsp.hframes = 4\n'
        '\tsp.vframes = 4\n'
        '\tadd_child(sp)\n',
        encoding="utf-8",
    )

    all_vfiles = collect_cartridge_visual_asset_files(tmp_path)
    meta_map = _collect_observational_asset_metadata(tmp_path, all_vfiles)
    # Cataloged non-parts assets (z_world_prop.png, z_micro_atlas.png) are marked in_catalog=True with exact grid metadata
    assert meta_map["assets/sprites/z_world_prop.png"]["in_catalog"] is True
    assert meta_map["assets/sprites/z_world_prop.png"]["grid"] == [1, 1]
    assert meta_map["assets/sprites/z_micro_atlas.png"]["in_catalog"] is True
    assert meta_map["assets/sprites/z_micro_atlas.png"]["grid"] == [8, 3]
    sheet_check = render_cartridge_asset_contact_sheet(tmp_path, max_tiles=64)
    assert sheet_check is not None and sheet_check.is_file()

    # Verify WritersRoom _build_asset_integrity_dossier_and_sheet renders observational contact sheet, caches by mtime, and invalidates on edit
    bible = GameBibleSpec(
        deliberation="Synthetic test bible.",
        game_title="Synthetic Horizon",
        elevator_pitch="Build and explore.",
        visual_style_guide="Rich hand-painted illustration.",
        slice_roadmap=[
            GameSliceSpec(
                deliberation="Slice 0.",
                slice_index=0,
                slice_id="slice_0",
                title="Init",
                slice_objective="Playable loop",
                status="active",
            )
        ],
    )
    (tmp_path / "game_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")

    wr = CreativeWritersRoom(client=None)
    with patch(
        "engine.supervisor.asset_validator.render_cartridge_asset_contact_sheet",
        wraps=render_cartridge_asset_contact_sheet,
    ) as render_spy:
        dossier_1, sheet_p1, _ = wr._build_asset_integrity_dossier_and_sheet(tmp_path, "slice_0_p0")
        assert render_spy.call_count == 1
        assert "=== CARTRIDGE VISUAL ASSET INVENTORY & OBSERVATIONAL CONTACT SHEET ===" in dossier_1
        assert "assets/sprites/z_micro_atlas.png" in dossier_1 or "assets/sprites/a_staging_unused_00.png" in dossier_1
        assert sheet_p1 is not None and sheet_p1.is_file()

        # Second call with unchanged mtimes hits cache (no duplicate contact sheet render)
        dossier_2, _, _ = wr._build_asset_integrity_dossier_and_sheet(tmp_path, "slice_0_p0")
        assert render_spy.call_count == 1
        assert dossier_2 == dossier_1

        # Touching a PNG invalidates cache and re-renders contact sheet
        future_ts = time.time() + 10.0
        os.utime(z_world_prop, (future_ts, future_ts))
        wr._build_asset_integrity_dossier_and_sheet(tmp_path, "slice_0_p0")
        assert render_spy.call_count == 2

    # Verify _proof_parts attaches .keyframe_fullres.png sidecar alongside filmstrip_*.png
    proof_dir = tmp_path / "artifacts" / "work_items" / "demo_item"
    proof_dir.mkdir(parents=True, exist_ok=True)
    strip_file = proof_dir / "filmstrip_demo_item_rev0.png"
    fullres_file = proof_dir / "filmstrip_demo_item_rev0.keyframe_fullres.png"
    _make_rich_rgba_cell(640, 360).save(strip_file)
    _make_rich_rgba_cell(1280, 720).save(fullres_file)

    board = ProductionBoard(tmp_path / "backlog.json")
    added = board.add_items(
        [
            WorkItemProposal(
                deliberation="Demo item.",
                item_id="demo_item",
                kind="integration",
                title="Demo integration",
                spec="Wire scene",
                proof_spec="Filmstrip",
            )
        ],
        raised_by_role="CreativeDirector",
        slice_id="slice_0",
    )
    board.claim_ready(1)
    board.submit_result(
        "demo_item",
        WorkerResult(
            notes="Captured strip",
            proof_images=[str(strip_file)],
        ),
    )
    item_obj = board.get("demo_item")
    proof_parts = wr._proof_parts(item_obj, tmp_path)
    text_labels = [p for p in proof_parts if isinstance(p, str)]
    assert any("filmstrip_demo_item_rev0.png" in lbl for lbl in text_labels)
    assert any("FULL-RESOLUTION 1:1 KEYFRAME" in lbl for lbl in text_labels)


def test_rule5_zero_procedural_fallbacks_across_visual_swe_and_checkpoint_mirror(tmp_path: Path, monkeypatch):
    """Verifies that:
    1. `VisualArtSpriteDesignerAgent` fails fast on `generate_sprites_from_requirements` (manifest and legacy paths)
       and non-live stub clients when `allow_offline_procedural=False`.
    2. `CartridgeAssetLibrary(..., allow_offline_procedural=False)` propagates `allow_offline_procedural=False` onto
       a supplied `VisualArtSpriteDesignerAgent` and blocks procedural portrait fallback.
    3. `LeadGameplaySWEAgent` blocks `_write_valid_tone_wav` / `_write_valid_indexed_png` when `allow_offline_procedural=False`
       and never overwrites `.wav`/`.png` assets with procedural placeholders on `backlog.json` board cartridges.
    4. `StudioOrchestrator._persist_verified_checkpoint` copies all runtime subdirectories (`src/`, `shaders/`, root `.gdshader`).
    """
    from engine.studio.models import RequirementTrack
    from engine.studio.orchestrator import StudioOrchestrator
    from engine.studio.swe import LeadGameplaySWEAgent

    # 1. VisualArtSpriteDesignerAgent strict mode with non-live wrapper client
    non_live_stub = SimpleNamespace(models=SimpleNamespace())
    vis_strict = VisualArtSpriteDesignerAgent(client=non_live_stub, allow_offline_procedural=False)
    with pytest.raises(StudioAgentConfigurationError):
        vis_strict.generate_cartridge_sprites(
            output_dir=tmp_path / "sprites_manifest",
            asset_manifest=[
                AssetRequirement(
                    deliberation="Test prop.",
                    asset_id="prop_alpha",
                    name="prop_alpha.png",
                    asset_type="prop",
                    target_path="assets/sprites/prop_alpha.png",
                    description="Test prop",
                )
            ],
        )
    with pytest.raises(StudioAgentConfigurationError):
        vis_strict.generate_cartridge_sprites(output_dir=tmp_path / "sprites_legacy", asset_manifest=None)

    # 2. CartridgeAssetLibrary propagates allow_offline_procedural=False onto supplied visual_agent & blocks portrait fallback
    vis_default = VisualArtSpriteDesignerAgent(client=None)
    assert vis_default.allow_offline_procedural is True
    lib_strict = CartridgeAssetLibrary(
        cartridge_dir=tmp_path / "strict_cart",
        visual_agent=vis_default,
        allow_offline_procedural=False,
    )
    assert vis_default.allow_offline_procedural is False
    with pytest.raises(StudioAgentConfigurationError):
        lib_strict.fabricate_single_asset(
            AssetRequirement(
                deliberation="Portrait asset.",
                asset_id="mentor_portrait",
                name="mentor_portrait.png",
                asset_type="portrait",
                fabrication_mode="portrait",
                target_path="assets/portraits/mentor_portrait.png",
                description="Portrait",
            ),
            force_regenerate=True,
        )

    # 3. LeadGameplaySWEAgent strict mode & board-cartridge protection
    swe_strict = LeadGameplaySWEAgent(client=None, allow_offline_procedural=False)
    with pytest.raises(StudioAgentConfigurationError):
        swe_strict._write_valid_tone_wav(tmp_path / "forbidden.wav")
    with pytest.raises(StudioAgentConfigurationError):
        swe_strict._write_valid_indexed_png(tmp_path / "forbidden.png")

    board_cart = tmp_path / "board_cart"
    board_cart.mkdir(parents=True, exist_ok=True)
    (board_cart / "backlog.json").write_text('{"items": []}', encoding="utf-8")
    existing_wav = board_cart / "assets" / "audio" / "click.wav"
    existing_wav.parent.mkdir(parents=True, exist_ok=True)
    existing_wav.write_bytes(b"ORIGINAL_WAV_BYTES")
    (board_cart / "narrative.json").write_text(
        json.dumps({"dialogue_barks": [{"voice_clip": "assets/audio/missing_bark.wav"}]}),
        encoding="utf-8",
    )
    swe_default = LeadGameplaySWEAgent(client=None)
    swe_default.repair_cartridge_code(
        board_cart,
        RejectionDiagnosticTrace(
            deliberation="Asset check.",
            candidate_id="board_cart",
            failure_gate="Gate B",
            failure_type=GateFailureType.ASSET_INTEGRITY_VIOLATION,
            failure_reason="Audio check",
        ),
    )
    assert existing_wav.read_bytes() == b"ORIGINAL_WAV_BYTES"
    assert not (board_cart / "assets" / "audio" / "missing_bark.wav").exists()

    # 4. StudioOrchestrator._persist_verified_checkpoint copies src/, shaders/, and root .gdshader
    import engine.studio.orchestrator as orch_mod

    fake_proj_root = tmp_path / "proj_root"
    (fake_proj_root / "cartridges").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(orch_mod, "PROJECT_ROOT", fake_proj_root)

    cand_ckpt = fake_proj_root / "cartridges" / "candidate_genesis_ckpt01"
    (cand_ckpt / "src").mkdir(parents=True, exist_ok=True)
    (cand_ckpt / "shaders").mkdir(parents=True, exist_ok=True)
    (cand_ckpt / "assets" / "sprites").mkdir(parents=True, exist_ok=True)
    (cand_ckpt / "artifacts").mkdir(parents=True, exist_ok=True)
    (cand_ckpt / "project.godot").write_text('[application]\nrun/main_scene="res://main.tscn"\n', encoding="utf-8")
    (cand_ckpt / "main.gd").write_text("extends Node2D\n", encoding="utf-8")
    (cand_ckpt / "water.gdshader").write_text("shader_type canvas_item;\n", encoding="utf-8")
    (cand_ckpt / "src" / "studio_rig_and_collider.gd").write_text("class_name StudioRigHelper\nextends RefCounted\n", encoding="utf-8")
    (cand_ckpt / "shaders" / "outline.gdshader").write_text("shader_type canvas_item;\n", encoding="utf-8")
    (cand_ckpt / "artifacts" / "scratch.txt").write_text("skip", encoding="utf-8")

    orch = StudioOrchestrator(api_key="test-key", client=non_live_stub)
    ckpt_out = orch._persist_verified_checkpoint(
        staging_dir=cand_ckpt,
        track=RequirementTrack(
            deliberation="Slice 0 checkpoint.",
            id="slice_0_completed",
            title="Slice 0",
            description="Completed slice 0",
            target_modules=["main.gd"],
        ),
        verified_tracks=["item_01"],
        mirror_to_active=True,
    )
    active_out = fake_proj_root / "cartridges" / "active"
    for target_root in (ckpt_out, active_out):
        assert (target_root / "src" / "studio_rig_and_collider.gd").is_file()
        assert (target_root / "shaders" / "outline.gdshader").is_file()
        assert (target_root / "water.gdshader").is_file()
        assert not (target_root / "artifacts").exists()


def test_qa_proposals_no_8_asset_truncation_and_drain_backlog_post_remediation_verification(tmp_path: Path):
    """Verifies that:
    1. `build_qa_gate_work_item_proposals` emits `visual_asset` proposals for ALL >8 offending assets
       (12 alphabetically ordered assets) without truncating after index 7.
    2. `CreativeWritersRoom._drain_backlog` runs a mandatory post-remediation verification pass after
       `max_blind_playtest_passes` or `max_qa_gate_passes` remediation tasks drain, completing cleanly
       when the verification pass succeeds and raising `StudioAgentError` (leaving slice `active` and
       persisting blockers on `backlog.json`) when defects remain unresolved.
    """
    from engine.studio.models import ReviewBatch, StudioAgentError
    from engine.studio.production_board import BoardScheduler

    # 1. Verify all 12 offending assets receive qa_regen_* proposals (zero [:8] truncation)
    twelve_assets = [f"assets/sprites/asset_{idx:02d}.png" for idx in range(1, 13)]
    trace_12 = RejectionDiagnosticTrace(
        deliberation="12 placeholder assets flagged on contact sheet.",
        candidate_id="cand_12",
        episode_number=0,
        attempt_number=1,
        failure_gate="Gate B: Multimodal Asset Integrity",
        failure_type=GateFailureType.ASSET_INTEGRITY_VIOLATION,
        failure_reason="12 flat geometric placeholder PNGs detected.",
        remedy_parameters={"offending_assets": twelve_assets, "offending_scripts": ["main.gd"]},
    )
    props_12 = build_qa_gate_work_item_proposals(
        candidate_dir=tmp_path,
        trace=trace_12,
        pass_index=0,
        slice_id="slice_0",
    )
    vis_targets = [p.target_modules[0] for p in props_12 if p.kind == "visual_asset"]
    assert vis_targets == twelve_assets

    def _make_slice_env(sub_name: str):
        c_dir = tmp_path / sub_name
        c_dir.mkdir(parents=True, exist_ok=True)
        (c_dir / "project.godot").write_text('[application]\nrun/main_scene="res://main.tscn"\n', encoding="utf-8")
        (c_dir / "main.gd").write_text("extends Node2D\n", encoding="utf-8")
        b = ProductionBoard(c_dir / "backlog.json")
        p_img = c_dir / "proof.png"
        _make_rich_rgba_cell(64, 64).save(p_img)

        def _worker(item: WorkItem) -> WorkerResult:
            return WorkerResult(proof_images=[str(p_img)], notes=f"Executed {item.item_id}")

        sched = BoardScheduler(
            b,
            {"mechanic_code": _worker, "integration": _worker, "visual_asset": _worker},
            max_workers=1,
        )
        rm = CreativeWritersRoom(
            client=object(),
            board=b,
            scheduler=sched,
            drain_poll_s=0.01,
            drain_timeout_s=5.0,
        )
        rm._review_as = lambda role, awaiting, needing, *a, **kw: ReviewBatch(
            deliberation="Approve proofs.",
            spoken_dialogue="Approved.",
            review_decisions=[
                ReviewDecision(deliberation="ok", item_id=it.item_id, verdict="approve", feedback="ok")
                for it in awaiting
            ],
            cap_decisions=[],
        )
        gb = GameBibleSpec(
            deliberation="Bible.",
            cartridge_id=sub_name,
            status="in_production",
            game_title="Verification Test",
            core_genre="simulation",
            slice_roadmap=[
                GameSliceSpec(
                    deliberation="Slice 0.",
                    slice_id="slice_0",
                    slice_index=0,
                    title="Slice 0",
                    slice_objective="Core loop",
                    status="active",
                )
            ],
            active_slice_index=0,
        )
        return c_dir, b, sched, rm, gb

    # 2a. Blind Playtest: 2 iterate_current_slice rounds + 3rd post-remediation verification pass approving -> completed!
    c_pt_ok, b_pt_ok, sched_pt_ok, rm_pt_ok, gb_pt_ok = _make_slice_env("pt_verify_pass")
    rm_pt_ok.max_assembly_passes = 0
    rm_pt_ok.max_blind_playtest_passes = 2
    pt_passes_seen: list[tuple[int, bool]] = []

    def _fake_blind_pt_ok(game_bible, dna_summary, candidate_dir, pass_index=0):
        is_final = bool(getattr(rm_pt_ok, "_final_verification_blind_playtest_pass", False))
        pt_passes_seen.append((pass_index, is_final))
        if pass_index < 2:
            added_items = b_pt_ok.add_items(
                [
                    WorkItemProposal(
                        deliberation="Repair gameplay.",
                        item_id=f"pt_fix_p{pass_index}",
                        kind="mechanic_code",
                        title=f"PT Fix {pass_index}",
                        spec="Fix gameplay",
                        proof_spec="Verify",
                        target_modules=["main.gd"],
                    )
                ],
                raised_by_role="CreativeDirector",
                slice_id="slice_0",
            )
            return len(added_items)
        return 0

    rm_pt_ok._run_end_of_slice_blind_playtest = _fake_blind_pt_ok
    try:
        rm_pt_ok._drain_backlog(gb_pt_ok, "DNA", c_pt_ok)
    finally:
        sched_pt_ok.stop()
    assert pt_passes_seen == [(0, False), (1, False), (2, True)]
    assert gb_pt_ok.slice_roadmap[0].status == "completed"

    # 2b. Blind Playtest: post-remediation verification pass STILL rejects -> raises StudioAgentError, slice stays active!
    c_pt_fail, b_pt_fail, sched_pt_fail, rm_pt_fail, gb_pt_fail = _make_slice_env("pt_verify_fail")
    rm_pt_fail.max_assembly_passes = 0
    rm_pt_fail.max_blind_playtest_passes = 1

    def _fake_blind_pt_fail(game_bible, dna_summary, candidate_dir, pass_index=0):
        added_items = b_pt_fail.add_items(
            [
                WorkItemProposal(
                    deliberation="Still broken.",
                    item_id=f"pt_blocker_p{pass_index}",
                    kind="mechanic_code",
                    title=f"Blocker {pass_index}",
                    spec="Fix blocker",
                    proof_spec="Verify",
                    target_modules=["main.gd"],
                )
            ],
            raised_by_role="CreativeDirector",
            slice_id="slice_0",
        )
        return len(added_items)

    rm_pt_fail._run_end_of_slice_blind_playtest = _fake_blind_pt_fail
    try:
        with pytest.raises(StudioAgentError, match="post-remediation Blind Playtest verification"):
            rm_pt_fail._drain_backlog(gb_pt_fail, "DNA", c_pt_fail)
    finally:
        sched_pt_fail.stop()
    assert gb_pt_fail.slice_roadmap[0].status == "active"
    assert b_pt_fail.get("pt_blocker_p1") is not None

    # 2c. 5-Gate QA: 2 failing remediation passes + 3rd post-remediation verification pass STILL failing -> raises StudioAgentError!
    c_qa_fail, b_qa_fail, sched_qa_fail, rm_qa_fail, gb_qa_fail = _make_slice_env("qa_verify_fail")
    rm_qa_fail.max_assembly_passes = 0
    rm_qa_fail.max_blind_playtest_passes = 0
    rm_qa_fail.max_qa_gate_passes = 2
    qa_eval_passes: list[int] = []

    def _always_failing_qa(c_path: Path, pass_idx: int):
        qa_eval_passes.append(pass_idx)
        return (
            {},
            RejectionDiagnosticTrace(
                deliberation="Persistent crash.",
                candidate_id=c_path.name,
                episode_number=0,
                attempt_number=pass_idx + 1,
                failure_gate="Gate C: Headless Execution",
                failure_type=GateFailureType.UNHANDLED_EXCEPTION,
                failure_reason=f"Crash persists on pass {pass_idx}",
                file_path="main.gd",
            ),
        )

    rm_qa_fail.qa_gate_evaluator_fn = _always_failing_qa
    try:
        with pytest.raises(StudioAgentError, match="post-remediation 5-Gate QA verification"):
            rm_qa_fail._drain_backlog(gb_qa_fail, "DNA", c_qa_fail)
    finally:
        sched_qa_fail.stop()
    assert qa_eval_passes == [0, 1, 2]
    assert gb_qa_fail.slice_roadmap[0].status == "active"


