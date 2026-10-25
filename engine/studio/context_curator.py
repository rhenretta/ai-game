"""Semantic Context Generator & Curator Agent (`ContextCuratorAgent`).

Rather than indiscriminately dumping the entire Game Bible, transcript, or asset catalog into every prompt
(or starving agents of visual/architectural context), `ContextCuratorAgent` indexes the cartridge's live
corpus across five semantic categories:
  1. `asset` — 16:9 Concept Art Mockups, fabricated sprites/rigs/sheets/tiles/audio in `asset_library.json`
     and `catalog.json`, and inspection plates (`artifacts/renders/*.png`).
  2. `council_utterance` — Writers' Room transcript utterances (`writers_room_transcript`), `concept_evolution`
     pivots, and Task Lead / Council revision traces.
  3. `design_spec` — Granular sections of `game_bible.json` (visual style guide, scene layers, camera/palette,
     technical architecture blueprints, level generation strategy, core loop, onboarding, narrative bible,
     slice roadmap) and individual treatments in `rendering_strategy.json` + `level_layout.json`.
  4. `board_item` — Active and completed `ProductionBoard` work items (with their full `spec`, `proof_spec`,
     `state`, `depends_on`, and revision history).
  5. `code_module` — On-disk `.gd` module symbols, methods, input actions, and `bound_assets`.

It embeds the corpus with `gemini-embedding-001` (cached by content SHA-256 in `artifacts/context_index.json`),
retrieves the highest-similarity chunks for the `(target_agent_role, task_request)` pair, and runs a structured
rationale-first LLM decision (`ContextCuratorDecision`, Rule 4) to curate the exact text chunks, role-tailored
synthesis, and multimodal reference/proof image attachments (`types.Part.from_bytes`) to inject into the target
agent's context window.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from engine.studio.models import (
    StudioAgentError,
    bounded_thinking_config,
    call_gemini_with_transient_retry,
)

logger = logging.getLogger("ContextCurator")

_INDEX_LOCK = threading.Lock()

ChunkCategory = Literal["asset", "council_utterance", "design_spec", "board_item", "code_module"]


class ContextChunk(BaseModel):
    """A discrete, semantically searchable unit of studio knowledge (Rule 4: deliberation first)."""

    deliberation: str = Field(
        default="Indexed studio context chunk for semantic vector retrieval.",
        description="Rationale for indexing this discrete knowledge chunk.",
    )
    chunk_id: str = Field(..., description="Unique stable identifier for this chunk within the cartridge.")
    category: ChunkCategory = Field(
        ...,
        description="Semantic category: 'asset', 'council_utterance', 'design_spec', 'board_item', or 'code_module'.",
    )
    title: str = Field(..., description="Short human-readable title summarizing the chunk.")
    content: str = Field(..., description="Full text payload of the chunk.")
    associated_image_path: str = Field(
        default="",
        description="Optional cartridge-relative or absolute image path (e.g. concept art mockup, sprite plate, or proof strip).",
    )


class ContextCuratorDecision(BaseModel):
    """Structured output of the Context Generator Agent (Rule 4: deliberation strictly first)."""

    deliberation: str = Field(
        ...,
        description="Step-by-step deliberation evaluating what council discussions, rendering treatments, assets, code symbols, and reference images the target agent role specifically needs to fulfill this request without context bloat.",
    )
    search_queries_used: List[str] = Field(
        default_factory=list,
        description="The semantic search facets prioritized for this agent role and request.",
    )
    role_tailored_synthesis: str = Field(
        ...,
        description="Focused, high-signal briefing tailored to `target_agent_role` that connects the Council's design intent, visual/rendering targets, relevant existing or queued assets, and technical contracts for this specific task.",
    )
    selected_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Exact `chunk_id` values from the candidate corpus chunks that must be injected into the agent's prompt.",
    )
    selected_image_paths: List[str] = Field(
        default_factory=list,
        description="Exact image paths from `AVAILABLE MULTIMODAL IMAGES` (such as the 16:9 Concept Art Mockup, fabricated sprite/rig plates, or proof filmstrips) to attach multimodally for this agent.",
    )


@dataclass
class CuratedContextPack:
    """The assembled text prompt block and multimodal image Parts produced by `ContextCuratorAgent`."""

    prompt_block: str
    multimodal_parts: List[Any] = field(default_factory=list)
    decision: Optional[ContextCuratorDecision] = None
    retrieved_chunks: List[ContextChunk] = field(default_factory=list)


def is_real_genai_client(client: Any) -> bool:
    """Returns True when `client` is a live `google.genai` Client (excluding unit-test `google.genai.fake` clients)."""
    if client is None:
        return False
    mod = getattr(type(client), "__module__", "") or ""
    return mod.startswith("google.genai") and mod != "google.genai.fake"


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CartridgeSemanticIndex:
    """Extracts and embeds all discrete knowledge chunks in a cartridge directory for semantic vector search."""

    EMBEDDING_MODEL = "gemini-embedding-001"
    EMBEDDING_DIM = 256
    BATCH_SIZE = 64

    def __init__(
        self,
        cartridge_dir: Path,
        client: Any = None,
        model_name: str = "gemini-3.8-flash",
    ) -> None:
        self.cartridge_dir = Path(cartridge_dir)
        self.client = client
        self.model_name = model_name
        self.index_path = self.cartridge_dir / "artifacts" / "context_index.json"

    def _ensure_rendering_strategy_if_needed(self) -> None:
        """Lazily synthesizes `rendering_strategy.json` once `game_bible.json` exists on disk so that even
        Turn 0 Task Leads and Visual Artists can semantically search the cartridge's rendering treatments."""
        rs_path = self.cartridge_dir / "rendering_strategy.json"
        bible_path = self.cartridge_dir / "game_bible.json"
        if rs_path.is_file() or not bible_path.is_file() or not is_real_genai_client(self.client):
            return
        try:
            from engine.studio.rendering import ensure_rendering_strategy

            ensure_rendering_strategy(self.cartridge_dir, self.client, self.model_name)
        except Exception as exc:
            logger.debug("Could not eagerly synthesize rendering_strategy.json during context indexing: %s", exc)

    def extract_corpus_chunks(
        self,
        board: Any = None,
        game_bible: Optional[Any] = None,
        ensure_rendering: bool = True,
    ) -> List[ContextChunk]:
        """Extracts granular `ContextChunk` objects across assets, council utterances, design specs,
        production board items, and GDScript code modules."""
        if ensure_rendering:
            self._ensure_rendering_strategy_if_needed()
        chunks: List[ContextChunk] = []
        seen_ids: set[str] = set()

        def _add(chunk: ContextChunk) -> None:
            if not chunk.content.strip() or chunk.chunk_id in seen_ids:
                return
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)

        # 1. Load game_bible.json (from passed object or disk)
        bible_dict: Dict[str, Any] = {}
        bible_path = self.cartridge_dir / "game_bible.json"
        if game_bible is not None and hasattr(game_bible, "model_dump"):
            try:
                bible_dict = game_bible.model_dump()
            except Exception:
                bible_dict = {}
        if not bible_dict and bible_path.is_file():
            try:
                raw = json.loads(bible_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    bible_dict = raw
            except Exception:
                bible_dict = {}

        if bible_dict:
            # 1a. Core identity & spatial/visual lock
            core_fields = {
                k: bible_dict.get(k)
                for k in (
                    "game_title",
                    "core_genre",
                    "elevator_pitch",
                    "core_fantasy",
                    "gameplay_pillars",
                    "camera_and_spatial_paradigm",
                    "color_palette_hex",
                    "first_60_seconds_hook",
                )
                if bible_dict.get(k)
            }
            if core_fields:
                _add(
                    ContextChunk(
                        chunk_id="design:core_identity",
                        category="design_spec",
                        title=f"Core Game Identity & Spatial/Palette Lock ({core_fields.get('game_title', 'Game')})",
                        content=json.dumps(core_fields, ensure_ascii=False, indent=2),
                    )
                )

            # 1b. Visual Style Guide & Multi-Layer Scene Architecture
            vis_fields = {
                k: bible_dict.get(k)
                for k in ("visual_style_guide", "camera_and_spatial_paradigm", "color_palette_hex", "scene_layers")
                if bible_dict.get(k)
            }
            if vis_fields:
                _add(
                    ContextChunk(
                        chunk_id="design:visual_style_and_layers",
                        category="design_spec",
                        title="Visual Style Guide & Multi-Layer Scene Architecture",
                        content=json.dumps(vis_fields, ensure_ascii=False, indent=2),
                    )
                )

            # 1c. 16:9 Concept Art Mockup Boards (indexed as `asset` chunks with `associated_image_path`)
            mockups = bible_dict.get("visual_mockups") or []
            if isinstance(mockups, list):
                for idx, m in enumerate(mockups):
                    if not isinstance(m, dict):
                        continue
                    rel_img = str(m.get("image_path") or "").strip()
                    resolved_img = ""
                    if rel_img:
                        cand = Path(rel_img) if Path(rel_img).is_absolute() else (self.cartridge_dir / rel_img)
                        if cand.is_file():
                            resolved_img = str(cand)
                    m_title = str(m.get("title") or f"16:9 Concept Art Mockup #{idx + 1}")
                    _add(
                        ContextChunk(
                            chunk_id=f"asset:concept_mockup:{idx}",
                            category="asset",
                            title=f"16:9 Concept Art Mockup — {m_title}",
                            content=json.dumps(
                                {
                                    "title": m_title,
                                    "image_path": rel_img,
                                    "scene_description": m.get("scene_description", ""),
                                    "camera_and_spatial_paradigm": m.get("camera_and_spatial_paradigm", ""),
                                    "color_palette_hex": m.get("color_palette_hex", []),
                                    "layered_composition_notes": m.get("layered_composition_notes", ""),
                                    "ui_callout_notes": m.get("ui_callout_notes", ""),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            associated_image_path=resolved_img,
                        )
                    )

            # 1d. Technical Architecture Blueprints (each mechanic indexed individually)
            tech_list = bible_dict.get("technical_architecture_blueprint") or []
            if isinstance(tech_list, list):
                for tech in tech_list:
                    if not isinstance(tech, dict):
                        continue
                    mech_id = str(tech.get("mechanic_id") or tech.get("title") or "mechanic").strip()
                    clean_tech = {k: v for k, v in tech.items() if k != "deliberation" and v}
                    _add(
                        ContextChunk(
                            chunk_id=f"design:tech_arch:{mech_id}",
                            category="design_spec",
                            title=f"Technical Architecture Blueprint — {tech.get('title', mech_id)}",
                            content=json.dumps(clean_tech, ensure_ascii=False, indent=2),
                        )
                    )

            # 1e. Level Generation Strategy, Core Loop, Engagement, Onboarding, Goal Hierarchy, Audio, UI
            for section_key, section_title in (
                ("level_generation_strategy", "Level Generation Strategy & World Construction Plan"),
                ("core_loop", "Core Gameplay Loop & System Interactions"),
                ("engagement_plan", "Player Engagement & Pacing Plan"),
                ("onboarding_plan", "Player Onboarding & World Introduction Plan"),
                ("goal_hierarchy", "Short-, Mid-, and Long-Term Goal Hierarchy"),
                ("diegetic_ui_philosophy", "Diegetic UI & HUD Philosophy"),
                ("audio_direction_guide", "Audio Direction & Soundscape Guide"),
            ):
                sec_val = bible_dict.get(section_key)
                if sec_val:
                    if isinstance(sec_val, dict):
                        sec_val = {k: v for k, v in sec_val.items() if k != "deliberation" and v}
                    _add(
                        ContextChunk(
                            chunk_id=f"design:{section_key}",
                            category="design_spec",
                            title=section_title,
                            content=json.dumps(sec_val, ensure_ascii=False, indent=2)
                            if not isinstance(sec_val, str)
                            else sec_val,
                        )
                    )

            # 1f. Active & Planned Slices
            slices = bible_dict.get("slice_roadmap") or []
            if isinstance(slices, list):
                for sl in slices:
                    if not isinstance(sl, dict):
                        continue
                    sl_id = str(sl.get("slice_id") or f"slice_{sl.get('slice_index', 0)}")
                    clean_sl = {k: v for k, v in sl.items() if k != "deliberation" and v is not None}
                    _add(
                        ContextChunk(
                            chunk_id=f"design:slice:{sl_id}",
                            category="design_spec",
                            title=f"Game Slice [{sl_id}] ({sl.get('status', '')}) — {sl.get('title', '')}",
                            content=json.dumps(clean_sl, ensure_ascii=False, indent=2),
                        )
                    )

            # 1g. Experience Storyboard Beats
            beats = bible_dict.get("experience_storyboard") or []
            if isinstance(beats, list) and beats:
                clean_beats = [
                    {k: v for k, v in b.items() if k != "deliberation" and v}
                    for b in beats
                    if isinstance(b, dict)
                ]
                _add(
                    ContextChunk(
                        chunk_id="design:experience_storyboard",
                        category="design_spec",
                        title="Experience Storyboard Beats (Emotional & Sensory Staging)",
                        content=json.dumps(clean_beats, ensure_ascii=False, indent=2),
                    )
                )

            # 1h. Narrative Bible (characters, cutscenes, dialogue trees, barks)
            nb = bible_dict.get("narrative_bible")
            if isinstance(nb, dict):
                chars = nb.get("characters") or []
                if isinstance(chars, list) and chars:
                    clean_chars = [
                        {k: v for k, v in c.items() if k != "deliberation" and v}
                        for c in chars
                        if isinstance(c, dict)
                    ]
                    _add(
                        ContextChunk(
                            chunk_id="design:narrative:characters",
                            category="design_spec",
                            title="Narrative Bible Characters, Visual Descriptions & Voice Profiles",
                            content=json.dumps(clean_chars, ensure_ascii=False, indent=2),
                        )
                    )
                for cs in (nb.get("cutscenes") or []):
                    if isinstance(cs, dict):
                        cs_id = str(cs.get("cutscene_id") or cs.get("title") or "cutscene")
                        _add(
                            ContextChunk(
                                chunk_id=f"design:narrative:cutscene:{cs_id}",
                                category="design_spec",
                                title=f"Narrative Cutscene — {cs.get('title', cs_id)}",
                                content=json.dumps(
                                    {k: v for k, v in cs.items() if k != "deliberation"},
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                            )
                        )
                for dt in (nb.get("dialogue_trees") or []):
                    if isinstance(dt, dict):
                        dt_id = str(dt.get("root_node_id") or "tree")
                        _add(
                            ContextChunk(
                                chunk_id=f"design:narrative:dialogue_tree:{dt_id}",
                                category="design_spec",
                                title=f"Interactive Dialogue Tree ({dt_id})",
                                content=json.dumps(
                                    {k: v for k, v in dt.items() if k != "deliberation"},
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                            )
                        )
                barks = nb.get("dialogue_barks") or []
                if isinstance(barks, list) and barks:
                    clean_barks = [
                        {k: v for k, v in b.items() if k != "deliberation" and v}
                        for b in barks
                        if isinstance(b, dict)
                    ]
                    _add(
                        ContextChunk(
                            chunk_id="design:narrative:barks",
                            category="design_spec",
                            title="Reactive Character Voice Barks",
                            content=json.dumps(clean_barks, ensure_ascii=False, indent=2),
                        )
                    )

            # 1i. Council Conversations (`writers_room_transcript` & `concept_evolution`)
            transcript = bible_dict.get("writers_room_transcript") or []
            if isinstance(transcript, list):
                for utt in transcript:
                    if not isinstance(utt, dict):
                        continue
                    t_idx = utt.get("turn_index", 0)
                    role = str(utt.get("speaker_role") or "Council")
                    disp = str(utt.get("speaker_display_name") or role)
                    spoken = str(utt.get("spoken_dialogue") or "").strip()
                    updates = utt.get("bible_updates_summary") or []
                    if not spoken:
                        continue
                    _add(
                        ContextChunk(
                            chunk_id=f"council:utterance:turn_{t_idx}_{role}",
                            category="council_utterance",
                            title=f"Writers' Room Turn #{t_idx} — {disp} ({role})",
                            content=(
                                f"Speaker: {disp} ({role})\n"
                                f"Addressed To: {utt.get('addressed_to', [])}\n"
                                f"Updates Summary: {updates}\n"
                                f"Dialogue: {spoken}"
                            ),
                        )
                    )

            evolutions = bible_dict.get("concept_evolution") or []
            if isinstance(evolutions, list):
                for evo in evolutions:
                    if not isinstance(evo, dict):
                        continue
                    t_idx = evo.get("turn_index", 0)
                    author = str(evo.get("author_role") or "Council")
                    _add(
                        ContextChunk(
                            chunk_id=f"council:evolution:turn_{t_idx}_{author}",
                            category="council_utterance",
                            title=f"Concept Evolution Pivot (Turn #{t_idx} by {author})",
                            content=(
                                f"Author: {author}\n"
                                f"Previous Premise: {evo.get('previous_premise', '')}\n"
                                f"Evolved Premise: {evo.get('evolved_premise', '')}\n"
                                f"Pivot Reason: {evo.get('pivot_reason', '')}"
                            ),
                        )
                    )

        # 2. Rendering Strategy (`rendering_strategy.json`) — overview + each individual subject treatment
        rs_path = self.cartridge_dir / "rendering_strategy.json"
        if rs_path.is_file():
            try:
                rs_data = json.loads(rs_path.read_text(encoding="utf-8"))
                if isinstance(rs_data, dict):
                    overview = {
                        k: rs_data.get(k)
                        for k in (
                            "rendering_paradigm",
                            "renderer",
                            "uses_environment_glow",
                            "pixelated_style_intended",
                            "lighting_and_atmosphere",
                            "effects_stack",
                            "asset_resolution_policy",
                        )
                        if rs_data.get(k) is not None
                    }
                    if overview:
                        _add(
                            ContextChunk(
                                chunk_id="design:rendering:overview",
                                category="design_spec",
                                title="Cartridge Rendering Strategy Overview (Renderer, Shaders, Lighting & Resolution Policy)",
                                content=json.dumps(overview, ensure_ascii=False, indent=2),
                            )
                        )
                    for tr in (rs_data.get("treatments") or []):
                        if not isinstance(tr, dict):
                            continue
                        subj_id = str(tr.get("subject_id") or "subject").strip()
                        _add(
                            ContextChunk(
                                chunk_id=f"design:rendering:treatment:{subj_id}",
                                category="design_spec",
                                title=f"Rendering Strategy Treatment — {subj_id} ({tr.get('technique', '')[:80]})",
                                content=json.dumps(tr, ensure_ascii=False, indent=2),
                            )
                        )
            except Exception:
                pass

        # 3. Authoritative Level Layout (`level_layout.json`)
        layout_path = self.cartridge_dir / "level_layout.json"
        if layout_path.is_file():
            try:
                layout_text = layout_path.read_text(encoding="utf-8")
                _add(
                    ContextChunk(
                        chunk_id="design:level_layout",
                        category="design_spec",
                        title="Authoritative Level Layout (`level_layout.json`)",
                        content=layout_text[:6000],
                    )
                )
            except Exception:
                pass

        # 4. On-Disk Asset Catalog (`assets/asset_library.json` and `catalog.json`) + Mockups & Render Plates
        catalog_entries: Dict[str, Dict[str, Any]] = {}
        for cat_file in (self.cartridge_dir / "assets" / "asset_library.json", self.cartridge_dir / "catalog.json"):
            if cat_file.is_file():
                try:
                    c_data = json.loads(cat_file.read_text(encoding="utf-8"))
                    entries = c_data.get("entries", c_data) if isinstance(c_data, dict) else {}
                    if isinstance(entries, dict):
                        for aid, aval in entries.items():
                            if isinstance(aval, dict):
                                catalog_entries.setdefault(str(aid), {}).update(aval)
                except Exception:
                    pass

        renders_dir = self.cartridge_dir / "artifacts" / "renders"
        for aid, aval in sorted(catalog_entries.items()):
            clean_entry = {k: v for k, v in aval.items() if k != "deliberation" and v not in (None, "", [], {})}
            rel_file = str(
                aval.get("res_path")
                or aval.get("file_path")
                or aval.get("file")
                or aval.get("target_path")
                or ""
            ).replace("res://", "")
            img_path = ""
            for cand_plate in (
                renders_dir / f"plate_{aid}.png",
                renders_dir / f"anim_rig_{aid}.png",
                renders_dir / f"anim_sheet_{aid}.png",
                (self.cartridge_dir / rel_file) if rel_file else None,
            ):
                if cand_plate is not None and cand_plate.is_file() and cand_plate.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    img_path = str(cand_plate)
                    break
            fmode_val = aval.get("fabrication_mode", aval.get("asset_type", "asset"))
            hf_val = int(aval.get("hframes") or 1)
            vf_val = int(aval.get("vframes") or 1)
            tex_sz_val = aval.get("texture_pixel_size") or aval.get("target_display_size") or []
            _add(
                ContextChunk(
                    chunk_id=f"asset:catalog:{aid}",
                    category="asset",
                    title=(
                        f"Fabricated Asset [{aid}] (mode={fmode_val}, path={rel_file}, "
                        f"tex_size={tex_sz_val}, hframes={hf_val}, vframes={vf_val})"
                    ),
                    content=json.dumps(clean_entry, ensure_ascii=False, indent=2),
                    associated_image_path=img_path,
                )
            )

        # Also index any mockup PNGs on disk if not already indexed from `visual_mockups`
        mockups_dir = self.cartridge_dir / "assets" / "mockups"
        if mockups_dir.is_dir():
            existing_mockup_imgs = {Path(c.associated_image_path).resolve() for c in chunks if c.associated_image_path}
            for mp in sorted(mockups_dir.glob("*.png")):
                if mp.resolve() not in existing_mockup_imgs:
                    _add(
                        ContextChunk(
                            chunk_id=f"asset:mockup_file:{mp.stem}",
                            category="asset",
                            title=f"16:9 Concept Art Mockup File ({mp.name})",
                            content=f"16:9 Concept Art Mockup image on disk at assets/mockups/{mp.name}.",
                            associated_image_path=str(mp),
                        )
                    )

        # 5. Production Board Items (`board.items()`, `backlog.json`, or `production_board.json`)
        raw_board_items: List[Dict[str, Any]] = []
        if board is not None and hasattr(board, "items"):
            try:
                for bit in board.items():
                    if hasattr(bit, "model_dump"):
                        raw_board_items.append(bit.model_dump())
            except Exception:
                pass
        if not raw_board_items:
            for pb_cand in (self.cartridge_dir / "backlog.json", self.cartridge_dir / "production_board.json"):
                if pb_cand.is_file():
                    try:
                        pb_data = json.loads(pb_cand.read_text(encoding="utf-8"))
                        if isinstance(pb_data, dict) and isinstance(pb_data.get("items"), list):
                            raw_board_items = [x for x in pb_data["items"] if isinstance(x, dict)]
                            break
                        elif isinstance(pb_data, list):
                            raw_board_items = [x for x in pb_data if isinstance(x, dict)]
                            break
                    except Exception:
                        pass

        for bit in raw_board_items:
            iid = str(bit.get("item_id") or "").strip()
            if not iid:
                continue
            revs = bit.get("revisions") or []
            latest_proof_img = ""
            rev_summaries: List[str] = []
            if isinstance(revs, list):
                for r in revs[-2:]:
                    if isinstance(r, dict):
                        rev_summaries.append(
                            f"rev {r.get('revision_index', 0) + 1}: verdict={r.get('verdict') or 'pending'} | "
                            f"notes={str(r.get('worker_notes') or '')[:300]} | feedback={str(r.get('feedback') or '')[:600]}"
                        )
                        for pp in (r.get("proof_paths") or []):
                            cand_p = Path(pp) if Path(pp).is_absolute() else (self.cartridge_dir / str(pp))
                            if cand_p.is_file():
                                latest_proof_img = str(cand_p)
            target_mods = bit.get("target_modules") or bit.get("owned_files") or []
            _add(
                ContextChunk(
                    chunk_id=f"board:item:{iid}",
                    category="board_item",
                    title=f"Board Task [{iid}] ({bit.get('kind', '')}, state={bit.get('state', '')}, raised_by={bit.get('raised_by_role', '')}): {bit.get('title', '')}",
                    content=(
                        f"Item ID: {iid}\n"
                        f"Kind: {bit.get('kind', '')} | State: {bit.get('state', '')} | Raised By: {bit.get('raised_by_role', '')} | Slice: {bit.get('slice_id', '')}\n"
                        f"Title: {bit.get('title', '')}\n"
                        f"Depends On: {bit.get('depends_on', [])}\n"
                        f"Target Modules: {target_mods}\n"
                        f"Spec: {bit.get('spec', '')}\n"
                        f"Proof Spec: {bit.get('proof_spec', '')}\n"
                        + (f"Recent Revisions:\n  " + "\n  ".join(rev_summaries) if rev_summaries else "")
                    ),
                    associated_image_path=latest_proof_img,
                )
            )

        # 6. On-Disk GDScript Modules & Bound Assets
        for gd_path in sorted(self.cartridge_dir.glob("*.gd")):
            if gd_path.name.startswith("test_"):
                continue
            try:
                src = gd_path.read_text(encoding="utf-8")
            except Exception:
                continue
            classes = re.findall(r"^\s*class\s+([A-Za-z0-9_]+)", src, flags=re.MULTILINE)
            funcs = re.findall(r"^\s*(?:static\s+)?func\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)", src, flags=re.MULTILINE)
            vars_top = re.findall(r"^var\s+([A-Za-z0-9_]+)", src, flags=re.MULTILINE)
            bound_assets = sorted(set(re.findall(r"(?:res://)?assets/[A-Za-z0-9_./-]+\.(?:png|wav|ogg|mp3)", src)))
            func_sig = ", ".join(f"{fn}({args.strip()})" for fn, args in funcs[:24])
            _add(
                ContextChunk(
                    chunk_id=f"code:module:{gd_path.name}",
                    category="code_module",
                    title=f"GDScript Module `{gd_path.name}` (bound_assets={bound_assets})",
                    content=(
                        f"Module: {gd_path.name} ({len(src)} chars)\n"
                        f"Inner Classes: {classes}\n"
                        f"Top-Level Vars: {vars_top[:20]}\n"
                        f"Bound Asset Paths Referenced in Code: {bound_assets}\n"
                        f"Function Signatures: [{func_sig}]"
                    ),
                )
            )

        return chunks

    def _load_embedding_cache(self) -> Dict[str, List[float]]:
        if not self.index_path.is_file():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("embeddings"), dict):
                return data["embeddings"]
        except Exception:
            pass
        return {}

    def _save_embedding_cache(self, cache: Dict[str, List[float]]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.EMBEDDING_MODEL, "dim": self.EMBEDDING_DIM, "embeddings": cache}
        self.index_path.write_text(json.dumps(payload), encoding="utf-8")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embeds `texts` using `client.models.embed_content(model='gemini-embedding-001')` with SHA-256 disk caching."""
        if not texts:
            return []
        if not is_real_genai_client(self.client) or not hasattr(getattr(self.client, "models", None), "embed_content"):
            raise StudioAgentError(
                "CartridgeSemanticIndex.embed_texts requires an active google.genai Client supporting models.embed_content (Rule 5)."
            )
        from google.genai import types

        hashes = [_sha256_text(t) for t in texts]
        with _INDEX_LOCK:
            cache = self._load_embedding_cache()
            missing_indices = [i for i, h in enumerate(hashes) if h not in cache]

        if missing_indices:
            new_vectors: Dict[str, List[float]] = {}
            for start in range(0, len(missing_indices), self.BATCH_SIZE):
                batch_idxs = missing_indices[start : start + self.BATCH_SIZE]
                batch_texts = [texts[i][:6000] for i in batch_idxs]
                resp = call_gemini_with_transient_retry(
                    lambda bt=batch_texts: self.client.models.embed_content(
                        model=self.EMBEDDING_MODEL,
                        contents=bt,
                        config=types.EmbedContentConfig(output_dimensionality=self.EMBEDDING_DIM),
                    ),
                    label=f"embed_content[batch={len(batch_texts)}]",
                )
                embeddings_list = getattr(resp, "embeddings", None) or []
                if len(embeddings_list) != len(batch_idxs):
                    raise StudioAgentError(
                        f"embed_content returned {len(embeddings_list)} embeddings for batch of {len(batch_idxs)} texts."
                    )
                for idx_in_texts, emb_obj in zip(batch_idxs, embeddings_list):
                    vals = list(getattr(emb_obj, "values", None) or [])
                    if not vals:
                        raise StudioAgentError("embed_content returned an empty embedding vector.")
                    new_vectors[hashes[idx_in_texts]] = vals

            with _INDEX_LOCK:
                cache = self._load_embedding_cache()
                cache.update(new_vectors)
                self._save_embedding_cache(cache)

        return [cache[h] for h in hashes]

    def semantic_search(
        self,
        queries: List[str],
        chunks: List[ContextChunk],
        top_k: int = 22,
        per_category_min: int = 3,
    ) -> List[Tuple[ContextChunk, float]]:
        """Ranks `chunks` by maximum cosine similarity across `queries`, guaranteeing representation across
        all five categories (`asset`, `council_utterance`, `design_spec`, `board_item`, `code_module`)
        before filling the remaining slots up to `top_k`."""
        if not chunks:
            return []
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries:
            return [(c, 1.0) for c in chunks[:top_k]]

        chunk_texts = [f"[{c.category}] {c.title}\n{c.content}" for c in chunks]
        all_embeddings = self.embed_texts(clean_queries + chunk_texts)
        query_vecs = all_embeddings[: len(clean_queries)]
        chunk_vecs = all_embeddings[len(clean_queries) :]

        scored: List[Tuple[ContextChunk, float]] = []
        for chunk, c_vec in zip(chunks, chunk_vecs):
            best_sim = max(_cosine_similarity(q_vec, c_vec) for q_vec in query_vecs)
            scored.append((chunk, best_sim))

        scored.sort(key=lambda pair: pair[1], reverse=True)

        selected: List[Tuple[ContextChunk, float]] = []
        selected_ids: set[str] = set()

        # Always guarantee the primary 16:9 Concept Art Mockup chunk is in the candidate pool if it exists
        for chunk, sim in scored:
            if chunk.chunk_id.startswith("asset:concept_mockup:") or chunk.chunk_id.startswith("asset:mockup_file:"):
                selected.append((chunk, sim))
                selected_ids.add(chunk.chunk_id)
                break

        # Guarantee top `per_category_min` chunks from each category so the LLM Curator can see assets,
        # council conversations, design/rendering specs, board items, and code symbols
        for cat in ("asset", "council_utterance", "design_spec", "board_item", "code_module"):
            cat_count = 0
            for chunk, sim in scored:
                if chunk.category != cat or chunk.chunk_id in selected_ids:
                    continue
                selected.append((chunk, sim))
                selected_ids.add(chunk.chunk_id)
                cat_count += 1
                if cat_count >= per_category_min:
                    break

        # Fill remaining slots up to `top_k` by global cosine similarity
        for chunk, sim in scored:
            if len(selected) >= top_k:
                break
            if chunk.chunk_id not in selected_ids:
                selected.append((chunk, sim))
                selected_ids.add(chunk.chunk_id)

        selected.sort(key=lambda pair: pair[1], reverse=True)
        return selected


_CURATOR_CACHE_LOCK = threading.RLock()


class ContextCuratorAgent:
    """Autonomous Context Generator Agent that semantically searches cartridge assets, council conversations,
    design/rendering specs, board items, and code modules, and decides what text and multimodal images to
    inject into a target agent's context window."""

    def __init__(
        self,
        cartridge_dir: Path,
        client: Any,
        model_name: str = "gemini-3.8-flash",
        board: Any = None,
        game_bible: Optional[Any] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.cartridge_dir = Path(cartridge_dir)
        self.client = client
        self.model_name = model_name
        self.board = board
        self.game_bible = game_bible
        self.event_callback = event_callback
        self.semantic_index = CartridgeSemanticIndex(
            cartridge_dir=self.cartridge_dir,
            client=self.client,
            model_name=self.model_name,
        )

    def _current_item_revision(self, item_id: str) -> int:
        if not item_id:
            return 0
        if self.board is not None and hasattr(self.board, "get"):
            try:
                it = self.board.get(item_id)
                if it is not None and hasattr(it, "revisions"):
                    return len(it.revisions or [])
            except Exception:
                pass
        bl_path = self.cartridge_dir / "backlog.json"
        if bl_path.is_file():
            try:
                raw = json.loads(bl_path.read_text(encoding="utf-8"))
                for entry in raw.get("items", []):
                    if isinstance(entry, dict) and entry.get("item_id") == item_id:
                        return len(entry.get("revisions") or [])
            except Exception:
                pass
        return 0

    @staticmethod
    def _normalized_cache_role(target_agent_role: str) -> str:
        role_clean = (target_agent_role or "").strip().lower()
        if role_clean.startswith("council_reviewer"):
            return "council_reviewer"
        return re.sub(r"[^a-z0-9_]+", "_", role_clean).strip("_") or "agent"

    def _build_role_queries(
        self,
        target_agent_role: str,
        task_request: str,
        item_id: str = "",
        item_kind: str = "",
    ) -> List[str]:
        """Constructs multi-faceted semantic search queries combining the request and the target agent's role."""
        base_query = f"Task [{item_id}] ({item_kind}): {task_request}".strip()
        role_lower = target_agent_role.lower()
        if "visual" in role_lower or "art" in role_lower:
            role_query = (
                f"Visual style guide, 16:9 concept art mockup, color palette, rendering strategy treatment, "
                f"sprite sheets, articulated rigs, and VisualArtDirector council discussions for: {task_request[:500]}"
            )
        elif "audio" in role_lower or "voice" in role_lower:
            role_query = (
                f"Audio direction guide, character voice profiles, reactive voice barks, sound effects, "
                f"and AudioAtmosphereDesigner council discussions for: {task_request[:500]}"
            )
        elif "swe" in role_lower or "engineer" in role_lower:
            role_query = (
                f"Godot 4 technical architecture blueprint, rendering strategy shader/sprite treatment, "
                f"on-disk asset catalog PNGs/rigs to bind, level layout, and GameplaySystemsEngineer council discussions for: {task_request[:500]}"
            )
        elif "qa" in role_lower or "test" in role_lower:
            role_query = (
                f"Falsifiable test invariants, input actions, GDScript state snapshot keys, setup methods, "
                f"and proof specifications for: {task_request[:500]}"
            )
        elif "lead" in role_lower:
            role_query = (
                f"16:9 concept art mockup, rendering strategy treatments, existing or queued visual/audio asset tasks "
                f"on the ProductionBoard, Council discussions, and active slice scope for: {task_request[:500]}"
            )
        else:
            role_query = (
                f"Council discussions, 16:9 concept art mockup, visual/gameplay design intent, bound assets, "
                f"and acceptance criteria for ({target_agent_role}): {task_request[:500]}"
            )
        return [base_query, role_query]

    def curate_context(
        self,
        target_agent_role: str,
        task_request: str,
        item_id: str = "",
        item_kind: str = "",
        extra_image_paths: Optional[List[str]] = None,
        max_selected_chunks: int = 12,
        max_selected_images: int = 4,
        reuse_revision_cache: bool = True,
    ) -> CuratedContextPack:
        """Semantically searches the cartridge corpus and runs the LLM Context Generator decision to assemble
        a tailored `CuratedContextPack` (`prompt_block` + `multimodal_parts`).
        Reuses the curated pack within the same `(item_id, revision, normalized_role)` so multi-turn Task Lead loops,
        SWE compiler-repair retries, and parallel Council review panelists share a stable byte-identical prefix."""
        if not is_real_genai_client(self.client):
            raise StudioAgentError(
                f"ContextCuratorAgent.curate_context requires an active google.genai Client for '{item_id}' (Rule 5)."
            )
        from google.genai import types

        safe_item = re.sub(r"[^a-z0-9_]+", "_", item_id.lower()).strip("_") if item_id else ""
        safe_role = re.sub(r"[^a-z0-9_]+", "_", target_agent_role.lower()).strip("_") or "agent"
        cache_role = self._normalized_cache_role(target_agent_role)
        current_rev = self._current_item_revision(item_id)
        trace_dir = (self.cartridge_dir / "artifacts" / "work_items" / safe_item) if safe_item else None
        cache_file = (trace_dir / f"context_pack_{cache_role}.json") if trace_dir is not None else None

        with _CURATOR_CACHE_LOCK:
            if reuse_revision_cache and cache_file is not None and cache_file.is_file():
                try:
                    cached = json.loads(cache_file.read_text(encoding="utf-8"))
                    if (
                        isinstance(cached, dict)
                        and cached.get("revision") == current_rev
                        and cached.get("prompt_block")
                        and isinstance(cached.get("decision"), dict)
                    ):
                        decision = ContextCuratorDecision.model_validate(cached["decision"])
                        all_chunks = self.semantic_index.extract_corpus_chunks(
                            board=self.board,
                            game_bible=self.game_bible,
                            ensure_rendering=False,
                        )
                        by_cid = {c.chunk_id: c for c in all_chunks}
                        chosen_chunks = [
                            by_cid[cid] for cid in (cached.get("selected_chunk_ids") or []) if cid in by_cid
                        ]
                        chosen_image_paths: List[str] = []
                        for ip in (cached.get("selected_image_paths") or []):
                            if ip and Path(ip).is_file() and str(ip) not in chosen_image_paths:
                                chosen_image_paths.append(str(ip))
                        for ep in list(extra_image_paths or [])[-3:]:
                            if ep and Path(ep).is_file() and str(ep) not in chosen_image_paths:
                                chosen_image_paths.append(str(ep))
                        multimodal_parts: List[Any] = []
                        for img_str in chosen_image_paths:
                            p = Path(img_str)
                            if not p.is_file():
                                continue
                            mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                            multimodal_parts.append(f"CURATED REFERENCE / PROOF IMAGE ({p.name}):")
                            multimodal_parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime))
                        if trace_dir is not None and safe_role != cache_role:
                            role_file = trace_dir / f"context_pack_{safe_role}.json"
                            if not role_file.is_file():
                                try:
                                    role_file.write_text(json.dumps(cached, indent=2), encoding="utf-8")
                                except Exception:
                                    pass
                        return CuratedContextPack(
                            prompt_block=str(cached["prompt_block"]),
                            multimodal_parts=multimodal_parts,
                            decision=decision,
                            retrieved_chunks=chosen_chunks,
                        )
                except Exception:
                    pass

            if self.event_callback is not None:
                try:
                    self.event_callback(
                        "studio_live_activity",
                        {
                            "deliberation": f"Context Curator Agent is semantically searching assets, council transcripts, and design specs for '{target_agent_role}' on '{item_id or 'task'}'.",
                            "agent_role": "ContextCurator",
                            "status_text": f"Context Curator: searching corpus & curating context for {target_agent_role} ({item_id or 'task'})...",
                            "active_agents": ["ContextCurator", target_agent_role],
                            "phase": "context_curation",
                        },
                    )
                except Exception:
                    pass

            all_chunks = self.semantic_index.extract_corpus_chunks(board=self.board, game_bible=self.game_bible)
            queries = self._build_role_queries(
                target_agent_role=target_agent_role,
                task_request=task_request,
                item_id=item_id,
                item_kind=item_kind,
            )
            ranked_pairs = self.semantic_index.semantic_search(queries=queries, chunks=all_chunks, top_k=22)
            candidate_chunks = [c for c, _ in ranked_pairs]
            chunks_by_id: Dict[str, ContextChunk] = {c.chunk_id: c for c in candidate_chunks}
            effective_extra_images: List[str] = list(extra_image_paths or [])
            if item_id:
                own_board_cid = f"board:item:{item_id}"
                for ch in all_chunks:
                    if ch.chunk_id == own_board_cid:
                        if own_board_cid not in chunks_by_id:
                            ranked_pairs.insert(0, (ch, 1.0))
                            candidate_chunks.insert(0, ch)
                            chunks_by_id[own_board_cid] = ch
                        if ch.associated_image_path and Path(ch.associated_image_path).is_file():
                            if ch.associated_image_path not in effective_extra_images:
                                effective_extra_images.insert(0, ch.associated_image_path)
                        break

            # Collect available multimodal images from candidate chunks + any explicit `extra_image_paths`
            available_images: Dict[str, str] = {}
            for c in candidate_chunks:
                if c.associated_image_path and Path(c.associated_image_path).is_file():
                    available_images[c.associated_image_path] = f"From {c.chunk_id} ({c.title})"
            for ep in effective_extra_images:
                if ep and Path(ep).is_file():
                    available_images[str(ep)] = f"Captured proof / prior revision image ({Path(ep).name})"

            candidate_lines: List[str] = []
            for c, sim in ranked_pairs:
                img_tag = f" | image={c.associated_image_path}" if c.associated_image_path else ""
                candidate_lines.append(
                    f"--- CHUNK id='{c.chunk_id}' category='{c.category}' similarity={sim:.3f}{img_tag} ---\n"
                    f"Title: {c.title}\n"
                    f"{c.content[:1800]}\n"
                )

            images_block = (
                "\n".join(f"  - path='{p}': {desc}" for p, desc in available_images.items())
                if available_images
                else "  (none on disk yet)"
            )

            prompt = (
                f"You are the Studio Context Generator & Curator Agent.\n"
                f"Your job is to inspect the semantically retrieved candidate chunks (spanning on-disk assets, Writers' Room council "
                f"conversations, Game Bible & Rendering Strategy treatments, Production Board items, and GDScript code modules) and "
                f"decide the exact context and multimodal images to inject for the target agent.\n\n"
                f"TARGET AGENT ROLE: {target_agent_role}\n"
                f"WORK ITEM: id='{item_id}' kind='{item_kind}'\n"
                f"TASK REQUEST / BRIEF:\n{task_request}\n\n"
                f"AVAILABLE MULTIMODAL IMAGES ON DISK (select up to {max_selected_images} relevant paths in `selected_image_paths`):\n"
                f"{images_block}\n\n"
                f"SEMANTICALLY RETRIEVED CANDIDATE CHUNKS (select up to {max_selected_chunks} relevant `chunk_id`s in `selected_chunk_ids`):\n"
                + "\n".join(candidate_lines)
                + "\nINSTRUCTIONS:\n"
                f"1. Select ONLY the chunks (`selected_chunk_ids`) that `{target_agent_role}` genuinely needs to execute or review `{item_id}` with high visual, mechanical, and architectural fidelity:\n"
                f"   - Include relevant Council utterances (`council:utterance:...`) where the Council discussed this feature, entity, level, or aesthetic.\n"
                f"   - Include the relevant `design:rendering:treatment:...` and `design:visual_style_and_layers` / `asset:concept_mockup:...` chunks whenever `{target_agent_role}` is planning, building, or reviewing visual assets, entities, levels (`level_map`), mechanics, or UI.\n"
                f"   - Include relevant `asset:catalog:...` and `board:item:...` chunks so `{target_agent_role}` knows which sprites/rigs/props already exist on disk vs. which are queued on the ProductionBoard.\n"
                f"2. In `selected_image_paths`, select the 16:9 Concept Art Mockup image path whenever `{target_agent_role}` is planning, implementing, fabricating, or reviewing visual, level, entity, or UI work, plus any relevant fabricated sprite plates or captured proof images.\n"
                f"3. In `role_tailored_synthesis`, write a concise, high-signal synthesis specifically for `{target_agent_role}` connecting the Council's discussion, the target visual/rendering aesthetic, and the exact asset/code contracts for `{item_id}`.\n"
                f"Provide `deliberation` FIRST."
            )

            resp = call_gemini_with_transient_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ContextCuratorDecision,
                        thinking_config=bounded_thinking_config(384),
                    ),
                ),
                label=f"curate_context[{cache_role}:{item_id}]",
            )
            if not resp.text:
                raise StudioAgentError(
                    f"ContextCuratorAgent returned an empty response for role='{target_agent_role}', item='{item_id}'."
                )
            decision = ContextCuratorDecision.model_validate_json(resp.text)

            # Resolve selected chunks (preserving curator order, falling back to top-ranked if none matched)
            chosen_chunks: List[ContextChunk] = []
            seen_cids: set[str] = set()
            for cid in decision.selected_chunk_ids:
                if cid in chunks_by_id and cid not in seen_cids:
                    chosen_chunks.append(chunks_by_id[cid])
                    seen_cids.add(cid)
                    if len(chosen_chunks) >= max_selected_chunks:
                        break
            if not chosen_chunks and candidate_chunks:
                chosen_chunks = candidate_chunks[: min(6, max_selected_chunks)]

            # Resolve multimodal image parts
            chosen_image_paths = []
            for ip in decision.selected_image_paths:
                cand_p = Path(ip) if Path(ip).is_absolute() else (self.cartridge_dir / ip)
                if cand_p.is_file() and str(cand_p) not in chosen_image_paths:
                    chosen_image_paths.append(str(cand_p))
                    if len(chosen_image_paths) >= max_selected_images:
                        break

            # Ensure any explicitly passed or prior-revision `effective_extra_images` (such as latest proof filmstrips) are attached
            for ep in effective_extra_images[-3:]:
                if ep and Path(ep).is_file() and str(ep) not in chosen_image_paths:
                    chosen_image_paths.append(str(ep))

            multimodal_parts = []
            for img_str in chosen_image_paths:
                p = Path(img_str)
                if not p.is_file():
                    continue
                mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                desc = available_images.get(img_str) or f"Attached image ({p.name})"
                multimodal_parts.append(f"CURATED REFERENCE / PROOF IMAGE ({p.name} — {desc}):")
                multimodal_parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime))

            chunk_sections = "\n\n".join(
                f"[{c.chunk_id}] ({c.category}) {c.title}:\n{c.content}" for c in chosen_chunks
            )
            prompt_block = (
                f"\n=== SEMANTICALLY CURATED STUDIO CONTEXT (for {cache_role} on '{item_id or 'task'}') ===\n"
                f"CURATOR SYNTHESIS:\n{decision.role_tailored_synthesis}\n\n"
                f"RETRIEVED & SELECTED STUDIO KNOWLEDGE CHUNKS ({len(chosen_chunks)} selected from {len(all_chunks)} indexed chunks):\n"
                f"{chunk_sections}\n"
                f"=== END CURATED STUDIO CONTEXT ===\n"
            )

            # Persist audit trace of what the Context Curator selected
            if trace_dir is not None and cache_file is not None:
                payload_to_save = {
                    "item_id": item_id,
                    "revision": current_rev,
                    "target_agent_role": target_agent_role,
                    "normalized_role": cache_role,
                    "total_indexed_chunks": len(all_chunks),
                    "candidate_chunk_ids": [c.chunk_id for c in candidate_chunks],
                    "selected_chunk_ids": [c.chunk_id for c in chosen_chunks],
                    "selected_image_paths": chosen_image_paths,
                    "prompt_block": prompt_block,
                    "decision": decision.model_dump(),
                }
                try:
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(payload_to_save, indent=2), encoding="utf-8")
                    if safe_role != cache_role:
                        (trace_dir / f"context_pack_{safe_role}.json").write_text(
                            json.dumps(payload_to_save, indent=2),
                            encoding="utf-8",
                        )
                except Exception:
                    pass

            if self.event_callback is not None:
                try:
                    self.event_callback(
                        "context_curator_packed",
                        {
                            "deliberation": decision.deliberation,
                            "item_id": item_id,
                            "target_agent_role": target_agent_role,
                            "role_tailored_synthesis": decision.role_tailored_synthesis,
                            "selected_chunk_ids": [c.chunk_id for c in chosen_chunks],
                            "selected_image_paths": [Path(p).name for p in chosen_image_paths],
                            "total_indexed_chunks": len(all_chunks),
                        },
                    )
                except Exception:
                    pass

            return CuratedContextPack(
                prompt_block=prompt_block,
                multimodal_parts=multimodal_parts,
                decision=decision,
                retrieved_chunks=chosen_chunks,
            )
