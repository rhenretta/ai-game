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
    infer_gdscript_asset_slicing_and_wiring,
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
    audit_cartridge_visual_and_slicing_integrity,
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
    assert entry.hframes == 3
    assert entry.vframes == 2

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

    # 1b. Unrelated item proposing the same path by mistake IS disambiguated
    planned_accidental = plan_asset_requirement(ctx, accidental_item, ("static_sprite", "sprite_sheet"))
    assert planned_accidental.target_path == "assets/sprites/accidental_other_prop_turret_base.png"

    # 2. Execute visual_asset_handler on regen_item -> overwrites in-place with force_regenerate=True
    visual_asset_handler(ctx)(regen_item)
    assert force_flags == [True]
    assert len(fabricate_calls) == 1
    with Image.open(broken_png) as img_after:
        assert img_after.size == (64, 64)

    # Catalog deduplicated: legacy_turret_id superseded by regen_turret_base
    reloaded_lib = CartridgeAssetLibrary(cartridge_dir=tmp_path, client=None)
    same_path_ids = [
        aid for aid, ent in reloaded_lib.catalog.entries.items() if ent.file_path == "assets/sprites/turret_base.png"
    ]
    assert same_path_ids == ["regen_turret_base"]
    swe_summary = reloaded_lib.build_prompt_summary_for_swe(compact=True)
    assert "tex_px=(64, 64)" in swe_summary

    # 3. ContextCurator indexes the fresh on-disk PNG (newer than stale plate) and reports actual_disk_pixel_size
    # Simulate an older plate for regen_turret_base too so mtime preference is directly exercised
    stale_regen_plate = renders_dir / "plate_regen_turret_base.png"
    Image.new("RGBA", (32, 32), (9, 9, 9, 255)).save(stale_regen_plate)
    os.utime(stale_regen_plate, (old_ts, old_ts))

    curator = ContextCuratorAgent(cartridge_dir=tmp_path, client=None, board=board)
    chunks = curator.semantic_index.extract_corpus_chunks()
    cat_doc = next(c for c in chunks if c.chunk_id == "asset:catalog:regen_turret_base")
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

    slicing_audit = infer_gdscript_asset_slicing_and_wiring(tmp_path)
    assert slicing_audit["asset_grids"]["assets/ui/recipe_grid_icons.png"] == (3, 2)
    assert len(slicing_audit["dead_texture_loads"]) >= 2
    assert len(slicing_audit["init_before_ready_texture_vars"]) == 1

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

    # Mock Gemini multimodal contact-sheet audit flagging flat_wedge.png as a placeholder geometric wedge
    verdict_json = AssetSheetMultimodalAuditVerdict(
        deliberation="Contact sheet shows recipe_grid_icons.png missing 5 of 6 cells and flat_wedge.png is a crude solid triangle.",
        offending_assets=["assets/sprites/flat_wedge.png"],
        remediation_summary="assets/sprites/flat_wedge.png is a flat solid-color geometric triangle placeholder.",
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
    assert "=== CARTRIDGE VISUAL ASSET & GDSCRIPT SLICING INTEGRITY AUDIT ===" in dossier
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
