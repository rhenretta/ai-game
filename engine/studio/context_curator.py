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


class SelfInquiryQAPair(BaseModel):
    """A resolved Question -> Self-Answer pair produced by an agent's self-inquiry turn (Rule 4: deliberation first)."""

    deliberation: str = Field(
        default="Recorded self-inquiry question and agent-synthesized answer for tight context retention.",
        description="Rationale for recording this resolved self-inquiry Q&A pair.",
    )
    question: str = Field(..., description="The question the agent asked itself when it needed additional context.")
    answer: str = Field(
        ...,
        description="The concise, factual answer synthesized by the asking agent itself on the next turn after inspecting the chunks brought in by ContextCuratorAgent.",
    )
    researched_chunk_ids: List[str] = Field(
        default_factory=list,
        description="The chunk IDs brought in by ContextCuratorAgent to let the agent answer this question.",
    )


class QuestionContextResearchDecision(BaseModel):
    """Structured output of ContextCuratorAgent when researching an agent's `self_question` (Rule 4: deliberation strictly first).
    Crucially, ContextCuratorAgent NEVER answers the question itself; it only selects the chunks needed for the asking agent to answer its own question.
    """

    deliberation: str = Field(
        ...,
        description="Step-by-step deliberation identifying which candidate chunks (code modules, reference-graph neighbors, design specs, council utterances, or asset definitions) contain the raw evidence needed so the asking agent can answer its own question on the next turn.",
    )
    selected_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Exact `chunk_id` values from the candidate chunks to bring into the asking agent's next turn so the agent can answer its own question.",
    )


@dataclass
class QuestionContextPack:
    """Raw context chunks retrieved by `ContextCuratorAgent.research_question_context` for an agent's `self_question`."""

    question: str
    prompt_block: str
    selected_chunk_ids: List[str] = field(default_factory=list)
    retrieved_chunks: List[ContextChunk] = field(default_factory=list)
    decision: Optional[QuestionContextResearchDecision] = None


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

            # 1e. Target Player DNA, Level Generation Strategy, Core Loop, Engagement, Onboarding, Goal Hierarchy, Audio, UI
            for section_key, section_title in (
                ("target_player_dna_summary", "Target Player Steam DNA Profile & Optimization Mandate"),
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

        from engine.studio.proofs import format_filmstrip_telemetry_progression

        for bit in raw_board_items:
            iid = str(bit.get("item_id") or "").strip()
            if not iid:
                continue
            revs = bit.get("revisions") or []
            latest_proof_img = ""
            rev_summaries: List[str] = []
            telemetry_summaries: List[str] = []
            if isinstance(revs, list):
                for r in revs[-2:]:
                    if isinstance(r, dict):
                        rev_summaries.append(
                            f"rev {r.get('revision_index', 0) + 1}: verdict={r.get('verdict') or 'pending'} | "
                            f"notes={str(r.get('worker_notes') or '')[:1000]} | feedback={str(r.get('feedback') or '')[:900]}"
                        )
                        for pp in (r.get("proof_paths") or []):
                            cand_p = Path(pp) if Path(pp).is_absolute() else (self.cartridge_dir / str(pp))
                            if cand_p.is_file() and cand_p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                                latest_proof_img = str(cand_p)
                                t_prog = format_filmstrip_telemetry_progression(cand_p)
                                if t_prog and t_prog not in telemetry_summaries:
                                    telemetry_summaries.append(t_prog)
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
                        + (f"Recent Revisions:\n  " + "\n  ".join(rev_summaries) + "\n" if rev_summaries else "")
                        + ("\n".join(telemetry_summaries) if telemetry_summaries else "")
                    ),
                    associated_image_path=latest_proof_img,
                )
            )

            # 5b. Build Execution Traces (`lead_trace_rev*.json`), Specialist Code Diffs & QA Telemetry Sidecars
            safe_iid = re.sub(r"[^a-z0-9_]+", "_", iid.lower()).strip("_")
            item_trace_dir = self.cartridge_dir / "artifacts" / "work_items" / safe_iid
            if item_trace_dir.is_dir():
                trace_files = sorted(
                    item_trace_dir.glob("lead_trace_rev*.json"),
                    key=lambda p: (
                        int(m.group(1)) if (m := re.search(r"rev(\d+)", p.stem)) else -1,
                        p.name,
                    ),
                )
                trace_lines: List[str] = []
                for tf in trace_files[-2:]:
                    try:
                        t_data = json.loads(tf.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if not isinstance(t_data, dict):
                        continue
                    t_rev = t_data.get("revision", 0)
                    t_status = t_data.get("status", "")
                    t_strat = str(t_data.get("current_strategy") or "")[:300]
                    trace_lines.append(f"=== Execution Trace `{tf.name}` (rev={t_rev}, status={t_status}) ===")
                    if t_strat:
                        trace_lines.append(f"  Lead Strategy: {t_strat}")
                    for turn in (t_data.get("turns") or [])[-6:]:
                        if not isinstance(turn, dict):
                            continue
                        t_idx = turn.get("turn_index", 0)
                        t_agent = turn.get("specialist_agent", "")
                        t_brief = str(turn.get("brief") or "").replace("\n", " ")[:320]
                        t_sum = str(turn.get("summary") or "").replace("\n", " ")[:420]
                        t_files = turn.get("touched_files") or []
                        trace_lines.append(
                            f"  [Turn {t_idx} -> {t_agent}] touched={t_files}\n"
                            f"    Brief: {t_brief}\n"
                            f"    Outcome: {t_sum}"
                        )
                        diffs = turn.get("code_diffs")
                        if isinstance(diffs, dict) and diffs:
                            for d_file, d_text in list(diffs.items())[:4]:
                                changed_lines = [
                                    ln
                                    for ln in str(d_text or "").splitlines()
                                    if (ln.startswith("+") or ln.startswith("-"))
                                    and not ln.startswith("+++")
                                    and not ln.startswith("---")
                                ]
                                if changed_lines:
                                    snippet = "\n      ".join(changed_lines[:18])
                                    trace_lines.append(f"    Diff `{d_file}`:\n      {snippet}")
                for strip_png in sorted(item_trace_dir.glob("*filmstrip*.png"))[-2:]:
                    t_prog = format_filmstrip_telemetry_progression(strip_png)
                    if t_prog:
                        trace_lines.append(t_prog)
                        latest_proof_img = str(strip_png)
                if trace_lines:
                    _add(
                        ContextChunk(
                            chunk_id=f"board:trace:{iid}",
                            category="board_item",
                            title=f"Build Execution Trace, Specialist Code Diffs & QA Telemetry for [{iid}]",
                            content="\n".join(trace_lines)[:4800],
                            associated_image_path=latest_proof_img,
                        )
                    )

        # 6. On-Disk GDScript / Shader / Scene Modules & Bidirectional Reference Graph
        code_graph = self.build_code_reference_graph()
        for rel_mod, meta in sorted(code_graph.items()):
            _add(
                ContextChunk(
                    chunk_id=f"code:module:{rel_mod}",
                    category="code_module",
                    title=(
                        f"Code Module `{rel_mod}` "
                        f"(imports={meta['imports']}, imported_by={meta['imported_by']}, bound_assets={meta['bound_assets']})"
                    ),
                    content=meta["summary_content"],
                )
            )

        return chunks

    def build_code_reference_graph(self) -> Dict[str, Dict[str, Any]]:
        """Scans all `.gd`, `.gdshader`, and `.tscn` files in the cartridge (including `scripts/`, `shaders/`, `tests/`)
        and constructs a bidirectional reference graph (`imports` and `imported_by`) across `preload(...)`, `load(...)`,
        `[ext_resource ... path="res://..."]`, and global `class_name` identifiers, along with compact symbol & property contracts."""
        ignored_top_dirs = {".godot", "artifacts", "assets", "res:", "__pycache__"}
        module_files: Dict[str, Path] = {}
        for ext in ("*.gd", "*.gdshader", "*.tscn"):
            for p in sorted(self.cartridge_dir.rglob(ext)):
                if not p.is_file():
                    continue
                try:
                    rel = p.relative_to(self.cartridge_dir)
                except ValueError:
                    continue
                if rel.parts and rel.parts[0] in ignored_top_dirs:
                    continue
                rel_str = str(rel).replace("\\", "/")
                if rel_str == "src/studio_rig_and_collider.gd":
                    continue
                module_files[rel_str] = p

        raw_sources: Dict[str, str] = {}
        class_name_to_mod: Dict[str, str] = {}
        for rel_str, p in module_files.items():
            try:
                src = p.read_text(encoding="utf-8")
            except Exception:
                continue
            raw_sources[rel_str] = src
            if rel_str.endswith(".gd"):
                m_cn = re.search(r"^\s*class_name\s+([A-Za-z0-9_]+)", src, flags=re.MULTILINE)
                if m_cn:
                    class_name_to_mod[m_cn.group(1).strip()] = rel_str

        imports_map: Dict[str, List[str]] = {k: [] for k in raw_sources}
        imported_by_map: Dict[str, List[str]] = {k: [] for k in raw_sources}

        ref_re = re.compile(
            r'(?:preload|load)\(\s*["\'](?:res://)?([A-Za-z0-9_./-]+\.(?:gd|gdshader|tscn))["\']\s*\)'
            r'|path\s*=\s*["\'](?:res://)?([A-Za-z0-9_./-]+\.(?:gd|gdshader|tscn))["\']'
        )

        for rel_str, src in raw_sources.items():
            refs: List[str] = []
            for m in ref_re.finditer(src):
                target = (m.group(1) or m.group(2) or "").strip().lstrip("/")
                if target in raw_sources and target != rel_str and target not in refs:
                    refs.append(target)
            if rel_str.endswith(".gd"):
                for cn, cn_mod in class_name_to_mod.items():
                    if cn_mod != rel_str and cn_mod not in refs:
                        if re.search(rf"\b{re.escape(cn)}\b", src):
                            refs.append(cn_mod)
            imports_map[rel_str] = refs
            for r in refs:
                if rel_str not in imported_by_map[r]:
                    imported_by_map[r].append(rel_str)

        from engine.studio.workers import _extract_gd_input_and_layer_semantics

        graph: Dict[str, Dict[str, Any]] = {}
        for rel_str, src in raw_sources.items():
            classes = re.findall(r"^\s*class\s+([A-Za-z0-9_]+)", src, flags=re.MULTILINE)
            signals = re.findall(r"^\s*signal\s+([A-Za-z0-9_]+(?:\([^)]*\))?)", src, flags=re.MULTILINE)
            funcs = re.findall(r"^\s*(?:static\s+)?func\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)", src, flags=re.MULTILINE)
            const_lines: List[str] = []
            for m_const in re.finditer(
                r"^const\s+([A-Za-z0-9_]+(?:\s*:\s*[A-Za-z0-9_]+)?\s*=\s*[^\n#]+)",
                src,
                flags=re.MULTILINE,
            ):
                c_decl = m_const.group(1).strip()
                if len(c_decl) > 80:
                    c_decl = c_decl[:77] + "..."
                const_lines.append(c_decl)
            var_lines: List[str] = []
            for m_var in re.finditer(
                r"^(?:@onready\s+|@export\s+)?var\s+([A-Za-z0-9_]+(?:\s*:\s*[A-Za-z0-9_]+)?(?:\s*=\s*[^\n#]+)?)",
                src,
                flags=re.MULTILINE,
            ):
                v_decl = m_var.group(1).strip()
                if len(v_decl) > 80:
                    v_decl = v_decl[:77] + "..."
                var_lines.append(v_decl)
            input_bindings: List[str] = []
            layer_lines: List[str] = []
            if rel_str.endswith(".gd"):
                input_bindings, layer_lines = _extract_gd_input_and_layer_semantics(src)
            uniforms = re.findall(
                r"^\s*uniform\s+([A-Za-z0-9_]+\s+[A-Za-z0-9_]+(?:\s*:\s*[^=;\n]+)?(?:\s*=\s*[^;\n]+)?)",
                src,
                flags=re.MULTILINE,
            )
            bound_assets = sorted(
                set(re.findall(r"(?:res://)?assets/[A-Za-z0-9_./-]+\.(?:png|wav|ogg|mp3)", src))
            )
            func_sig = ", ".join(f"{fn}({args.strip()})" for fn, args in funcs[:28])
            summary_lines = [
                f"Module: {rel_str} ({len(src)} chars)",
                f"Imports / References: {imports_map[rel_str]}",
                f"Imported By: {imported_by_map[rel_str]}",
            ]
            if signals:
                summary_lines.append(f"Signals: {signals[:16]}")
            if classes:
                summary_lines.append(f"Inner Classes: {classes[:12]}")
            if const_lines:
                summary_lines.append(f"Constants: {const_lines[:18]}")
            if var_lines:
                summary_lines.append(f"Top-Level Vars & Defaults: {var_lines[:28]}")
            if input_bindings:
                summary_lines.append(f"Input Action / Key Bindings (`KEY_*` / `Input.* -> effect`): {input_bindings[:18]}")
            if layer_lines:
                summary_lines.append(f"Collision Layers/Masks, z_index & CanvasModulate: {layer_lines[:16]}")
            if uniforms:
                summary_lines.append(f"Shader Uniforms: {uniforms[:20]}")
            if bound_assets:
                summary_lines.append(f"Bound Asset Paths Referenced in Code: {bound_assets}")
            if func_sig:
                summary_lines.append(f"Function Signatures: [{func_sig}]")
            if rel_str.endswith((".tscn", ".gdshader")) and len(src) <= 1800:
                summary_lines.append(f"Source Preview:\n{src}")
            graph[rel_str] = {
                "rel_path": rel_str,
                "imports": imports_map[rel_str],
                "imported_by": imported_by_map[rel_str],
                "bound_assets": bound_assets,
                "summary_content": "\n".join(summary_lines),
            }
        return graph

    def get_transitive_related_modules(
        self,
        seed_modules: Sequence[str],
        max_hops: int = 2,
        code_graph: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[str]:
        """Walks `imports` and `imported_by` edges in `build_code_reference_graph()` up to `max_hops`
        starting from `seed_modules` (normalizing `res://` prefixes and bare basenames) and returns
        all discovered module relative paths in BFS order."""
        graph = code_graph if code_graph is not None else self.build_code_reference_graph()
        if not graph:
            return []
        by_basename: Dict[str, List[str]] = {}
        for k in graph:
            by_basename.setdefault(Path(k).name, []).append(k)

        queue: List[Tuple[str, int]] = []
        visited: List[str] = []
        for raw_seed in seed_modules:
            s = str(raw_seed or "").strip().replace("\\", "/")
            if s.startswith("res://"):
                s = s[len("res://"):]
            elif s.startswith("res:/"):
                s = s[len("res:/"):]
            s = s.lstrip("/")
            if not s:
                continue
            resolved_seeds = [s] if s in graph else by_basename.get(Path(s).name, [])
            for rs in resolved_seeds:
                if rs not in visited:
                    visited.append(rs)
                    queue.append((rs, 0))

        idx = 0
        while idx < len(queue):
            cur_mod, depth = queue[idx]
            idx += 1
            if depth >= max_hops:
                continue
            meta = graph.get(cur_mod) or {}
            neighbors = list(meta.get("imports") or []) + list(meta.get("imported_by") or [])
            for nb in neighbors:
                if nb in graph and nb not in visited:
                    visited.append(nb)
                    queue.append((nb, depth + 1))
        return visited

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

        intra_rev_sig = f"rev={current_rev}"
        if trace_dir is not None and trace_dir.is_dir():
            try:
                active_trace = trace_dir / f"lead_trace_rev{current_rev}.json"
                turn_count = 0
                if active_trace.is_file():
                    raw_t = json.loads(active_trace.read_text(encoding="utf-8"))
                    if isinstance(raw_t, dict):
                        turn_count = len(raw_t.get("turns") or [])
                strip_files = sorted(trace_dir.glob("*filmstrip*.png"))
                latest_strip_name = strip_files[-1].name if strip_files else ""
                intra_rev_sig = f"rev={current_rev}|turns={turn_count}|strip={latest_strip_name}"
            except Exception:
                pass

        with _CURATOR_CACHE_LOCK:
            if reuse_revision_cache and cache_file is not None and cache_file.is_file():
                try:
                    cached = json.loads(cache_file.read_text(encoding="utf-8"))
                    if (
                        isinstance(cached, dict)
                        and cached.get("revision") == current_rev
                        and cached.get("intra_rev_sig", f"rev={current_rev}") == intra_rev_sig
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
                            if (
                                ip
                                and Path(ip).is_file()
                                and Path(ip).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                                and str(ip) not in chosen_image_paths
                            ):
                                chosen_image_paths.append(str(ip))
                        for ep in list(extra_image_paths or [])[-3:]:
                            if (
                                ep
                                and Path(ep).is_file()
                                and Path(ep).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                                and str(ep) not in chosen_image_paths
                            ):
                                chosen_image_paths.append(str(ep))
                        multimodal_parts: List[Any] = []
                        for img_str in chosen_image_paths:
                            p = Path(img_str)
                            if not p.is_file() or p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                                continue
                            mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else ("image/webp" if p.suffix.lower() == ".webp" else "image/png")
                            n_low = p.name.lower()
                            qa_overlay_note = ""
                            if n_low.startswith("plate_"):
                                qa_overlay_note = " [QA ALPHA INSPECTION PLATE: dark/light checkerboard grids and cyan bounding box are external QA overlays, NOT inside the sprite PNG]"
                            elif n_low.startswith("anim_sheet_") or n_low.startswith("anim_rig_"):
                                qa_overlay_note = " [QA ANIMATION CONTACT SHEET: dark checkerboard grid, cyan cell borders, and horizontal yellow/gold baseline ruler lines at the bottom of frames are external QA alignment guides, NOT inside the sprite sheet PNG]"
                            elif "filmstrip" in n_low:
                                qa_overlay_note = " [QA CONTACT SHEET: top/bottom '[QA] tick ...' / '[QA] pos=...' strips are external QA margins, NOT inside the 16:9 viewport]"
                            multimodal_parts.append(f"CURATED REFERENCE / PROOF IMAGE ({p.name}){qa_overlay_note}:")
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
            effective_extra_images: List[str] = [
                str(ep)
                for ep in (extra_image_paths or [])
                if ep and Path(ep).is_file() and Path(ep).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            ]
            if trace_dir is not None and trace_dir.is_dir():
                for strip_p in sorted(trace_dir.glob("*filmstrip*.png"))[-2:]:
                    if str(strip_p) not in effective_extra_images:
                        effective_extra_images.append(str(strip_p))
            if item_id:
                for pinned_cid in (f"board:trace:{item_id}", f"board:item:{item_id}"):
                    for ch in all_chunks:
                        if ch.chunk_id == pinned_cid:
                            if pinned_cid not in chunks_by_id:
                                ranked_pairs.insert(0, (ch, 1.0))
                                candidate_chunks.insert(0, ch)
                                chunks_by_id[pinned_cid] = ch
                            if (
                                ch.associated_image_path
                                and Path(ch.associated_image_path).is_file()
                                and Path(ch.associated_image_path).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                            ):
                                if ch.associated_image_path not in effective_extra_images:
                                    effective_extra_images.insert(0, ch.associated_image_path)
                            break

            # Traverse up to 3 hops of the code reference graph (imports & imported_by) from target modules
            # so any file referenced by or referencing the task's files (including 2-3 hop actor/physics dependencies
            # like `main.gd -> entities.gd -> scenes/vehicles/tug_controller.gd`) is presented to the Curator LLM.
            code_graph = self.semantic_index.build_code_reference_graph()
            all_by_cid: Dict[str, ContextChunk] = {c.chunk_id: c for c in all_chunks}
            seed_mods: List[str] = []

            def _add_seed_mod(raw_m: str) -> None:
                s = str(raw_m or "").strip().replace("\\", "/")
                if s.startswith("res://"):
                    s = s[len("res://"):]
                elif s.startswith("res:/"):
                    s = s[len("res:/"):]
                s = s.lstrip("/")
                if s in code_graph and s not in seed_mods:
                    seed_mods.append(s)

            if item_id and self.board is not None and hasattr(self.board, "get"):
                try:
                    b_item = self.board.get(item_id)
                    for tm in (getattr(b_item, "target_modules", None) or []):
                        _add_seed_mod(tm)
                except Exception:
                    pass
            for m_path in re.findall(r"(?:res://)?([A-Za-z0-9_./-]+\.(?:gd|gdshader|tscn))", task_request or ""):
                _add_seed_mod(m_path)
            if item_kind in ("level_map", "ui", "integration", "mechanic_code"):
                _add_seed_mod("main.gd")

            graph_relations: Dict[str, str] = {}
            hop1_mods: List[str] = []
            for sm in seed_mods:
                graph_relations.setdefault(sm, "target/seed module")
                meta = code_graph.get(sm) or {}
                for imp in meta.get("imports", []):
                    if imp in code_graph and imp not in graph_relations:
                        graph_relations[imp] = f"imported by `{sm}`"
                        hop1_mods.append(imp)
                for imp_by in meta.get("imported_by", []):
                    if imp_by in code_graph and imp_by not in graph_relations:
                        graph_relations[imp_by] = f"imports `{sm}`"
                        hop1_mods.append(imp_by)
            hop2_mods: List[str] = []
            for h1 in hop1_mods:
                meta = code_graph.get(h1) or {}
                for imp in meta.get("imports", []):
                    if imp in code_graph and imp not in graph_relations:
                        graph_relations[imp] = f"2-hop import via `{h1}`"
                        hop2_mods.append(imp)
                for imp_by in meta.get("imported_by", []):
                    if imp_by in code_graph and imp_by not in graph_relations:
                        graph_relations[imp_by] = f"2-hop importer via `{h1}`"
                        hop2_mods.append(imp_by)
            for h2 in hop2_mods:
                meta = code_graph.get(h2) or {}
                for imp in meta.get("imports", []):
                    if imp in code_graph and imp not in graph_relations:
                        graph_relations[imp] = f"3-hop import via `{h2}`"

            for rel_mod, rel_reason in graph_relations.items():
                cid = f"code:module:{rel_mod}"
                if cid in all_by_cid and cid not in chunks_by_id:
                    ch_copy = all_by_cid[cid].model_copy(
                        update={
                            "title": f"{all_by_cid[cid].title} [GRAPH-LINKED: {rel_reason}]",
                        }
                    )
                    ranked_pairs.append((ch_copy, 0.92))
                    candidate_chunks.append(ch_copy)
                    chunks_by_id[cid] = ch_copy
                elif cid in chunks_by_id:
                    existing_ch = chunks_by_id[cid]
                    if "[GRAPH-LINKED:" not in existing_ch.title:
                        existing_ch.title = f"{existing_ch.title} [GRAPH-LINKED: {rel_reason}]"

            # Collect available multimodal images from candidate chunks + any explicit `extra_image_paths`
            available_images: Dict[str, str] = {}
            for c in candidate_chunks:
                if (
                    c.associated_image_path
                    and Path(c.associated_image_path).is_file()
                    and Path(c.associated_image_path).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                ):
                    available_images[c.associated_image_path] = f"From {c.chunk_id} ({c.title})"
            for ep in effective_extra_images:
                if ep and Path(ep).is_file() and Path(ep).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    available_images[str(ep)] = f"Captured proof / prior revision image ({Path(ep).name})"

            candidate_lines: List[str] = []
            for c, sim in ranked_pairs:
                img_tag = f" | image={c.associated_image_path}" if c.associated_image_path else ""
                candidate_lines.append(
                    f"--- CHUNK id='{c.chunk_id}' category='{c.category}' similarity={sim:.3f}{img_tag} ---\n"
                    f"Title: {c.title}\n"
                    f"{c.content[:2400]}\n"
                )

            images_block = (
                "\n".join(f"  - path='{p}': {desc}" for p, desc in available_images.items())
                if available_images
                else "  (none on disk yet)"
            )

            prompt = (
                f"You are the Studio Context Generator & Curator Agent.\n"
                f"Your job is to inspect the semantically retrieved candidate chunks (spanning on-disk assets, Writers' Room council "
                f"conversations, Game Bible & Rendering Strategy treatments, Production Board items, build execution traces `board:trace:*`, "
                f"and GDScript code modules) and decide the exact context and multimodal images to inject for the target agent.\n\n"
                f"TARGET AGENT ROLE: {target_agent_role}\n"
                f"WORK ITEM: id='{item_id}' kind='{item_kind}'\n"
                f"TASK REQUEST / BRIEF:\n{task_request}\n\n"
                f"AVAILABLE MULTIMODAL IMAGES ON DISK (select up to {max_selected_images} relevant paths in `selected_image_paths`):\n"
                f"{images_block}\n\n"
                f"SEMANTICALLY RETRIEVED CANDIDATE CHUNKS (select up to {max_selected_chunks} relevant `chunk_id`s in `selected_chunk_ids`):\n"
                + "\n".join(candidate_lines)
                + "\nINSTRUCTIONS:\n"
                f"1. Select ONLY the chunks (`selected_chunk_ids`) that `{target_agent_role}` genuinely needs to execute or review `{item_id}` with high visual, mechanical, and architectural fidelity:\n"
                f"   - Include `board:item:{item_id}`, `board:trace:...`, or other `board:item:...` chunks when `{target_agent_role}` needs to know the task spec, prior revision diffs, or captured frame telemetry.\n"
                f"   - Include relevant Council utterances (`council:utterance:...`) where the Council discussed this feature, entity, level, or aesthetic.\n"
                f"   - Include the relevant `design:rendering:treatment:...` and `design:visual_style_and_layers` / `asset:concept_mockup:...` chunks whenever `{target_agent_role}` is planning, building, or reviewing visual assets, entities, levels (`level_map`), mechanics, or UI.\n"
                f"   - Include relevant `asset:catalog:...` chunks so `{target_agent_role}` knows which sprites/rigs/props already exist on disk.\n"
                f"   - Inspect `code:module:...` chunks (especially `[GRAPH-LINKED: ...]` modules imported by or importing the task's target files, plus actor controllers whose input bindings, collision layers, or physics interact with this task) and include any referenced modules needed for cross-file root-cause diagnosis.\n"
                f"2. In `selected_image_paths`, select the 16:9 Concept Art Mockup image path whenever `{target_agent_role}` is planning, implementing, fabricating, or reviewing visual, level, entity, or UI work, plus any relevant fabricated sprite plates or captured proof filmstrip images.\n"
                f"3. In `role_tailored_synthesis`, write a concise, high-signal synthesis specifically for `{target_agent_role}` connecting the Council's discussion, prior build execution traces / frame telemetry coordinates, the target visual/rendering aesthetic, and the exact cross-file asset/code contracts (including input bindings and collision layers) for `{item_id}`.\n"
                f"Provide `deliberation` FIRST."
            )

            curator_multimodal_parts: List[Any] = []
            for ep in effective_extra_images[-2:]:
                ep_path = Path(ep)
                if ep_path.is_file() and ep_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    mime = "image/jpeg" if ep_path.suffix.lower() in (".jpg", ".jpeg") else ("image/webp" if ep_path.suffix.lower() == ".webp" else "image/png")
                    curator_multimodal_parts.append(f"ATTACHED PROOF / REFERENCE IMAGE ({ep_path.name}):")
                    curator_multimodal_parts.append(types.Part.from_bytes(data=ep_path.read_bytes(), mime_type=mime))

            curator_contents: Any = ([prompt] + curator_multimodal_parts) if curator_multimodal_parts else prompt
            resp = call_gemini_with_transient_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=curator_contents,
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

            # Resolve selected chunks strictly from the Curator LLM's `selected_chunk_ids` decision
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
                if (
                    cand_p.is_file()
                    and cand_p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                    and str(cand_p) not in chosen_image_paths
                ):
                    chosen_image_paths.append(str(cand_p))
                    if len(chosen_image_paths) >= max_selected_images:
                        break

            # Ensure any explicitly passed or prior-revision `effective_extra_images` (such as latest proof filmstrips) are attached
            for ep in effective_extra_images[-3:]:
                if (
                    ep
                    and Path(ep).is_file()
                    and Path(ep).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                    and str(ep) not in chosen_image_paths
                ):
                    chosen_image_paths.append(str(ep))

            multimodal_parts = []
            for img_str in chosen_image_paths:
                p = Path(img_str)
                if not p.is_file() or p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                    continue
                mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else ("image/webp" if p.suffix.lower() == ".webp" else "image/png")
                desc = available_images.get(img_str) or f"Attached image ({p.name})"
                n_low = p.name.lower()
                qa_overlay_note = ""
                if n_low.startswith("plate_"):
                    qa_overlay_note = " [QA ALPHA INSPECTION PLATE: dark/light checkerboard grids and cyan bounding box are external QA overlays, NOT inside the sprite PNG]"
                elif n_low.startswith("anim_sheet_") or n_low.startswith("anim_rig_"):
                    qa_overlay_note = " [QA ANIMATION CONTACT SHEET: dark checkerboard grid, cyan cell borders, and horizontal yellow/gold baseline ruler lines at the bottom of frames are external QA alignment guides, NOT inside the sprite sheet PNG]"
                elif "filmstrip" in n_low:
                    qa_overlay_note = " [QA CONTACT SHEET: top/bottom '[QA] tick ...' / '[QA] pos=...' strips are external QA margins, NOT inside the 16:9 viewport]"
                multimodal_parts.append(f"CURATED REFERENCE / PROOF IMAGE ({p.name} — {desc}){qa_overlay_note}:")
                multimodal_parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime))

            expanded_chosen_chunks: List[ContextChunk] = []
            for c in chosen_chunks:
                if c.chunk_id.startswith("code:module:"):
                    rel_mod = c.chunk_id[len("code:module:") :].strip().lstrip("/")
                    f_mod = self.cartridge_dir / rel_mod
                    if f_mod.is_file() and rel_mod.endswith(".gd"):
                        try:
                            from engine.studio.swe import LeadGameplaySWEAgent
                            raw_gd = f_mod.read_text(encoding="utf-8")
                            rich_contract = LeadGameplaySWEAgent._summarize_non_target_gd_module(rel_mod, raw_gd)
                            c = c.model_copy(
                                update={"content": f"{c.content}\n\n--- MODULE CONTRACT & SEMANTICS ---\n{rich_contract}"}
                            )
                        except Exception:
                            pass
                expanded_chosen_chunks.append(c)

            chunk_sections = "\n\n".join(
                f"[{c.chunk_id}] ({c.category}) {c.title}:\n{c.content}" for c in expanded_chosen_chunks
            )
            prompt_block = (
                f"\n=== SEMANTICALLY CURATED STUDIO CONTEXT (for {cache_role} on '{item_id or 'task'}') ===\n"
                f"CURATOR SYNTHESIS:\n{decision.role_tailored_synthesis}\n\n"
                f"RETRIEVED & SELECTED STUDIO KNOWLEDGE CHUNKS ({len(expanded_chosen_chunks)} selected from {len(all_chunks)} indexed chunks):\n"
                f"{chunk_sections}\n"
                f"=== END CURATED STUDIO CONTEXT ===\n"
            )

            # Persist audit trace of what the Context Curator selected
            if trace_dir is not None and cache_file is not None:
                payload_to_save = {
                    "item_id": item_id,
                    "revision": current_rev,
                    "intra_rev_sig": intra_rev_sig,
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

    def _expand_chunk_for_question(self, chunk: ContextChunk) -> ContextChunk:
        """When a `code:module:<rel_path>` chunk is selected to help an agent answer a `self_question`,
        expands the chunk content with the full on-disk source code of that module so the asking agent
        can inspect exact method bodies, signal definitions, and shader uniforms."""
        if not chunk.chunk_id.startswith("code:module:"):
            return chunk
        rel_path = chunk.chunk_id[len("code:module:") :].strip().lstrip("/")
        if not rel_path:
            return chunk
        f_path = self.cartridge_dir / rel_path
        if not f_path.is_file():
            return chunk
        try:
            raw_src = f_path.read_text(encoding="utf-8")
        except Exception:
            return chunk
        expanded_content = (
            f"{chunk.content}\n\n"
            f"--- FULL SOURCE OF `{rel_path}` ---\n"
            f"{raw_src[:6500]}"
        )
        return chunk.model_copy(update={"content": expanded_content})

    def research_question_context(
        self,
        target_agent_role: str,
        question: str,
        item_id: str = "",
        target_modules: Optional[Sequence[str]] = None,
        already_included_chunk_ids: Optional[Sequence[str]] = None,
        max_selected_chunks: int = 5,
    ) -> QuestionContextPack:
        """Researches an agent's `self_question` against the cartridge semantic index and code reference graph.

        IMPORTANT: `ContextCuratorAgent` does NOT answer the question itself. It selects and returns the
        necessary raw context chunks (including full module source for selected `code:module:*` chunks) so the
        asking agent can answer its own question on the next turn.
        """
        clean_q = (question or "").strip()
        if not clean_q:
            return QuestionContextPack(question="", prompt_block="")

        from google.genai import types

        all_chunks = self.semantic_index.extract_corpus_chunks(
            board=self.board,
            game_bible=self.game_bible,
        )
        if not all_chunks:
            return QuestionContextPack(
                question=clean_q,
                prompt_block=(
                    f"\n=== CONTEXT RESEARCHED FOR YOUR SELF-QUESTION ===\n"
                    f"YOUR PENDING QUESTION: {clean_q}\n"
                    f"(No indexed chunks exist in the cartridge yet.)\n"
                    f"=== END RESEARCHED QUESTION CONTEXT ===\n"
                ),
            )

        excluded = set(already_included_chunk_ids or [])
        ranked_pairs = self.semantic_index.semantic_search(
            queries=[
                clean_q,
                f"GDScript modules, signals, methods, shaders, and design specs for ({target_agent_role}): {clean_q}",
            ],
            chunks=all_chunks,
            top_k=min(14, len(all_chunks)),
        )
        candidate_chunks: List[ContextChunk] = []
        filtered_pairs: List[Tuple[ContextChunk, float]] = []
        chunks_by_id: Dict[str, ContextChunk] = {}
        all_by_id = {c.chunk_id: c for c in all_chunks}

        for ch, sim in ranked_pairs:
            if ch.chunk_id in excluded:
                continue
            candidate_chunks.append(ch)
            filtered_pairs.append((ch, sim))
            chunks_by_id[ch.chunk_id] = ch

        # Traverse the code reference graph from any module paths mentioned in `question` or `target_modules`
        code_graph = self.semantic_index.build_code_reference_graph()
        seed_mods: List[str] = []
        for tm in target_modules or []:
            norm_m = str(tm or "").strip().replace("\\", "/")
            if norm_m.startswith("res://"):
                norm_m = norm_m[len("res://") :]
            elif norm_m.startswith("res:/"):
                norm_m = norm_m[len("res:/") :]
            norm_m = norm_m.lstrip("/")
            if norm_m and norm_m not in seed_mods:
                seed_mods.append(norm_m)

        q_lower = clean_q.lower()
        for known_mod in code_graph:
            if known_mod.lower() in q_lower or Path(known_mod).name.lower() in q_lower:
                if known_mod not in seed_mods:
                    seed_mods.append(known_mod)
                cid = f"code:module:{known_mod}"
                if cid in all_by_id and cid not in chunks_by_id:
                    ch_direct = all_by_id[cid].model_copy(
                        update={"title": f"{all_by_id[cid].title} [MENTIONED IN QUESTION]"}
                    )
                    candidate_chunks.insert(0, ch_direct)
                    filtered_pairs.insert(0, (ch_direct, 0.98))
                    chunks_by_id[cid] = ch_direct

        for sm in seed_mods:
            entry = code_graph.get(sm) or {}
            for neighbor in list(entry.get("imports", [])) + list(entry.get("imported_by", [])):
                cid = f"code:module:{neighbor}"
                if cid in all_by_id and cid not in chunks_by_id and cid not in excluded:
                    ch_graph = all_by_id[cid].model_copy(
                        update={"title": f"{all_by_id[cid].title} [GRAPH-LINKED TO {sm}]"}
                    )
                    candidate_chunks.append(ch_graph)
                    filtered_pairs.append((ch_graph, 0.92))
                    chunks_by_id[cid] = ch_graph

        # If all top chunks were already in `excluded`, allow the top ranked chunks so the agent still gets evidence
        if not candidate_chunks:
            for ch, sim in ranked_pairs[: min(6, len(ranked_pairs))]:
                candidate_chunks.append(ch)
                filtered_pairs.append((ch, sim))
                chunks_by_id[ch.chunk_id] = ch

        decision: Optional[QuestionContextResearchDecision] = None
        chosen_chunks: List[ContextChunk] = []

        if is_real_genai_client(self.client):
            candidate_lines: List[str] = []
            for c, sim in filtered_pairs[:16]:
                candidate_lines.append(
                    f"--- CHUNK id='{c.chunk_id}' category='{c.category}' similarity={sim:.3f} ---\n"
                    f"Title: {c.title}\n"
                    f"{c.content[:1600]}\n"
                )
            prompt = (
                f"You are the Studio Context Generator & Curator Agent researching a self-inquiry question asked by `{target_agent_role}`.\n"
                f"CRITICAL RULE: Do NOT answer the question yourself! Your ONLY job is to select the candidate `chunk_id`s "
                f"(up to {max_selected_chunks}) that contain the raw code, signals, variables, council decisions, or design specs "
                f"so `{target_agent_role}` can read them and answer its own question on the next turn.\n\n"
                f"ASKING AGENT ROLE: {target_agent_role}\n"
                f"WORK ITEM: {item_id or '(general)'}\n"
                f"AGENT'S SELF-QUESTION:\n{clean_q}\n\n"
                f"CANDIDATE CHUNKS:\n"
                + "\n".join(candidate_lines)
                + "\nSelect the most relevant `selected_chunk_ids` and provide `deliberation` FIRST."
            )
            resp = call_gemini_with_transient_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=QuestionContextResearchDecision,
                        thinking_config=bounded_thinking_config(256),
                    ),
                ),
                label=f"research_question[{self._normalized_cache_role(target_agent_role)}:{item_id}]",
            )
            if resp.text:
                decision = QuestionContextResearchDecision.model_validate_json(resp.text)
                seen_ids: set[str] = set()
                for cid in decision.selected_chunk_ids:
                    if cid in chunks_by_id and cid not in seen_ids:
                        chosen_chunks.append(self._expand_chunk_for_question(chunks_by_id[cid]))
                        seen_ids.add(cid)
                        if len(chosen_chunks) >= max_selected_chunks:
                            break

        if not chosen_chunks and candidate_chunks:
            chosen_chunks = [
                self._expand_chunk_for_question(c) for c in candidate_chunks[: min(3, max_selected_chunks)]
            ]

        chunk_sections = "\n\n".join(
            f"[{c.chunk_id}] ({c.category}) {c.title}:\n{c.content}" for c in chosen_chunks
        )
        prompt_block = (
            f"\n=== CONTEXT RESEARCHED FOR YOUR SELF-QUESTION (for {target_agent_role} on '{item_id or 'task'}') ===\n"
            f"YOUR PENDING QUESTION FROM PREVIOUS TURN:\n{clean_q}\n\n"
            f"RETRIEVED CONTEXT CHUNKS ({len(chosen_chunks)} chunk(s) brought in by ContextCuratorAgent):\n"
            f"{chunk_sections}\n\n"
            f"MANDATORY NEXT-TURN INSTRUCTION:\n"
            f"1. Read the retrieved chunks above and write a concise, factual answer to your question in `self_question_answer` "
            f"(citing the exact function names, signal signatures, property names, or spec values you need).\n"
            f"2. Your `self_question_answer` will be saved in your persistent context going forward so you do not need to re-ask this question.\n"
            f"3. After filling `self_question_answer`, proceed to complete your task output in this same response (leave `self_question` empty unless a brand-new follow-up question is strictly required).\n"
            f"=== END RESEARCHED QUESTION CONTEXT ===\n"
        )

        if self.event_callback is not None:
            try:
                self.event_callback(
                    "context_curator_question_researched",
                    {
                        "deliberation": (
                            decision.deliberation
                            if decision is not None
                            else f"Researched context for {target_agent_role} self-question: {clean_q[:120]}"
                        ),
                        "item_id": item_id,
                        "target_agent_role": target_agent_role,
                        "question": clean_q,
                        "selected_chunk_ids": [c.chunk_id for c in chosen_chunks],
                    },
                )
            except Exception:
                pass

        return QuestionContextPack(
            question=clean_q,
            prompt_block=prompt_block,
            selected_chunk_ids=[c.chunk_id for c in chosen_chunks],
            retrieved_chunks=chosen_chunks,
            decision=decision,
        )

    def _self_inquiry_memory_path(self, item_id: str, target_agent_role: str) -> Optional[Path]:
        safe_item = re.sub(r"[^a-z0-9_]+", "_", (item_id or "").lower()).strip("_")
        if not safe_item:
            return None
        cache_role = self._normalized_cache_role(target_agent_role)
        trace_dir = self.cartridge_dir / "artifacts" / "work_items" / safe_item
        trace_dir.mkdir(parents=True, exist_ok=True)
        return trace_dir / f"self_inquiry_qa_{cache_role}.json"

    def load_self_inquiry_memory(self, item_id: str, target_agent_role: str) -> List[SelfInquiryQAPair]:
        """Loads previously answered `(question, answer)` pairs for `(item_id, target_agent_role)`."""
        p = self._self_inquiry_memory_path(item_id, target_agent_role)
        if p is None or not p.is_file():
            return []
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return [SelfInquiryQAPair.model_validate(item) for item in raw if isinstance(item, dict)]
        except Exception:
            pass
        return []

    def save_self_inquiry_memory(
        self,
        item_id: str,
        target_agent_role: str,
        pairs: Sequence[SelfInquiryQAPair],
    ) -> None:
        """Persists resolved `(question, answer)` pairs for `(item_id, target_agent_role)` so they remain in context across turns."""
        p = self._self_inquiry_memory_path(item_id, target_agent_role)
        if p is None:
            return
        try:
            p.write_text(
                json.dumps([pair.model_dump() for pair in pairs], indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    @staticmethod
    def format_self_inquiry_prompt_section(
        resolved_pairs: Sequence[SelfInquiryQAPair],
        pending_pack: Optional[QuestionContextPack] = None,
    ) -> str:
        """Formats the agent's accumulated `Question -> Self-Answer` memory plus any active `pending_pack`
        brought in by `ContextCuratorAgent` for the current turn."""
        sections: List[str] = []
        if resolved_pairs:
            qa_lines = [
                "=== SELF-INQUIRY MEMORY (QUESTIONS YOU ALREADY ASKED & ANSWERED) ===",
                "Use these verified facts that you extracted on earlier turns (do not re-ask these questions):",
            ]
            for idx, pair in enumerate(resolved_pairs, start=1):
                cids = f" (from {pair.researched_chunk_ids})" if pair.researched_chunk_ids else ""
                qa_lines.append(f"  Q{idx}{cids}: {pair.question}\n  A{idx}: {pair.answer}")
            qa_lines.append("=== END SELF-INQUIRY MEMORY ===")
            sections.append("\n".join(qa_lines))
        if pending_pack is not None and pending_pack.prompt_block:
            sections.append(pending_pack.prompt_block)
        return ("\n\n".join(sections) + "\n\n") if sections else ""

