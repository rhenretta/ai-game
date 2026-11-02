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

from pydantic import BaseModel, Field, model_validator

from engine.studio.models import (
    StudioAgentError,
    bounded_thinking_config,
    call_gemini_with_transient_retry,
    is_real_genai_client,
)

logger = logging.getLogger("ContextCurator")

_INDEX_LOCK = threading.Lock()

ChunkCategory = Literal[
    "asset",
    "council_utterance",
    "design_spec",
    "board_item",
    "code_module",
    "game_memory",
    "tool_capability",
]


class ContextChunk(BaseModel):
    """A discrete, semantically searchable unit of studio knowledge (Rule 4: deliberation first)."""

    deliberation: str = Field(
        default="Indexed studio context chunk for semantic vector retrieval.",
        description="Rationale for indexing this discrete knowledge chunk.",
    )
    chunk_id: str = Field(..., description="Unique stable identifier for this chunk within the cartridge.")
    category: ChunkCategory = Field(
        ...,
        description="Semantic category: 'asset', 'council_utterance', 'design_spec', 'board_item', 'code_module', or 'game_memory'.",
    )
    title: str = Field(..., description="Short human-readable title summarizing the chunk.")
    content: str = Field(..., description="Full text payload of the chunk.")
    associated_image_path: str = Field(
        default="",
        description="Optional cartridge-relative or absolute media path (e.g. concept art mockup, sprite plate, .wav cue, or .mp4 proof clip).",
    )


class ContextCuratorDecision(BaseModel):
    """Structured output of the Context Generator Agent (Rule 4: deliberation strictly first)."""

    deliberation: str = Field(
        ...,
        description="Step-by-step deliberation evaluating what council discussions, rendering treatments, assets, code symbols, and reference media (.png/.wav/.mp4) the target agent role specifically needs to fulfill this request without context bloat.",
    )
    search_queries_used: List[str] = Field(
        default_factory=list,
        description="The semantic search facets prioritized for this agent role and request.",
    )
    role_tailored_synthesis: str = Field(
        ...,
        description=(
            "Strictly neutral, factual map of the retrieved context for `target_agent_role`: list what files, modules, "
            "design specs, and artifacts are included and how they connect in the codebase. NEVER solve the problem, "
            "propose code changes, diagnose bugs, or claim that a proof/telemetry confirms or passes a requirement."
        ),
    )
    selected_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Exact `chunk_id` values from the candidate corpus chunks that must be injected into the agent's prompt.",
    )
    selected_image_paths: List[str] = Field(
        default_factory=list,
        description="Exact image/media paths from `AVAILABLE MULTIMODAL MEDIA` (such as the 16:9 Concept Art Mockup, sprite/rig plates, proof filmstrips, .wav cues, or .mp4 clips) to attach multimodally for this agent.",
    )
    selected_media_paths: List[str] = Field(
        default_factory=list,
        description="Exact multimodal paths (.png/.jpg/.webp images, .wav audio cues, or .mp4 gameplay clips) from `AVAILABLE MULTIMODAL MEDIA` to attach when seeing or hearing them genuinely helps `target_agent_role`.",
    )

    @model_validator(mode="after")
    def _sync_media_and_image_paths(self) -> "ContextCuratorDecision":
        merged: List[str] = []
        for raw_p in list(self.selected_media_paths or []) + list(self.selected_image_paths or []):
            s = str(raw_p or "").strip()
            if s and s not in merged:
                merged.append(s)
        self.selected_media_paths = list(merged)
        self.selected_image_paths = list(merged)
        return self


@dataclass
class CuratedContextPack:
    """The assembled text prompt block and multimodal Parts produced by `ContextCuratorAgent`."""

    prompt_block: str
    multimodal_parts: List[Any] = field(default_factory=list)
    decision: Optional[ContextCuratorDecision] = None
    retrieved_chunks: List[ContextChunk] = field(default_factory=list)
    selected_media_paths: List[str] = field(default_factory=list)


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
    _MEMORY_EMBEDDING_CACHE: Dict[str, Dict[str, List[float]]] = {}

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
            asset_disk_path = (self.cartridge_dir / rel_file) if rel_file else None
            actual_disk_px: Optional[List[int]] = None
            asset_disk_mtime = 0.0
            if asset_disk_path is not None and asset_disk_path.is_file() and asset_disk_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                try:
                    asset_disk_mtime = asset_disk_path.stat().st_mtime
                    from PIL import Image as _PILImage
                    with _PILImage.open(asset_disk_path) as _im:
                        actual_disk_px = [int(_im.size[0]), int(_im.size[1])]
                except Exception:
                    pass

            img_path = ""
            for cand_plate in (
                renders_dir / f"plate_{aid}.png",
                renders_dir / f"anim_rig_{aid}.png",
                renders_dir / f"anim_sheet_{aid}.png",
            ):
                if cand_plate.is_file() and cand_plate.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    try:
                        plate_mtime = cand_plate.stat().st_mtime
                    except Exception:
                        plate_mtime = 0.0
                    # Never serve a stale pre-overwrite render plate if the actual PNG on disk is newer!
                    if asset_disk_path is not None and asset_disk_mtime > plate_mtime + 0.5:
                        img_path = str(asset_disk_path)
                    else:
                        img_path = str(cand_plate)
                    break
            if not img_path and asset_disk_path is not None and asset_disk_path.is_file() and asset_disk_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                img_path = str(asset_disk_path)

            if actual_disk_px is not None:
                clean_entry["actual_disk_pixel_size"] = actual_disk_px
            fmode_val = str(aval.get("fabrication_mode", aval.get("asset_type", "asset")) or "asset")
            atype_val = str(aval.get("asset_type", "") or "")
            is_audio_entry = (
                fmode_val in ("audio_bark", "sfx", "ambient_loop")
                or atype_val in ("audio_bark", "sfx", "ambient_loop")
                or (
                    asset_disk_path is not None
                    and asset_disk_path.suffix.lower() in (".wav", ".mp3", ".ogg", ".flac", ".aac", ".opus")
                )
            )
            if is_audio_entry:
                dur_val = float(aval.get("duration_seconds") or 0.0)
                if dur_val <= 0.0 and asset_disk_path is not None and asset_disk_path.is_file() and asset_disk_path.suffix.lower() == ".wav":
                    try:
                        from engine.studio.audio import validate_wav_file
                        _ok_w, _err_w, computed_dur = validate_wav_file(asset_disk_path)
                        if _ok_w and computed_dur > 0:
                            dur_val = round(float(computed_dur), 3)
                            clean_entry["duration_seconds"] = dur_val
                    except Exception:
                        pass
                perc_desc = str(aval.get("perceptual_description") or "").strip()
                media_path_str = (
                    str(asset_disk_path)
                    if (asset_disk_path is not None and asset_disk_path.is_file())
                    else img_path
                )
                sound_tag = f", sound={perc_desc[:120]!r}" if perc_desc else ""
                chunk_title = (
                    f"Fabricated Audio Asset [{aid}] (mode={fmode_val}, path={rel_file}, "
                    f"duration={dur_val:.2f}s{sound_tag})"
                )
            else:
                media_path_str = img_path
                hf_val = int(aval.get("hframes") or 1)
                vf_val = int(aval.get("vframes") or 1)
                tex_sz_val = actual_disk_px or aval.get("texture_pixel_size") or aval.get("target_display_size") or []
                chunk_title = (
                    f"Fabricated Asset [{aid}] (mode={fmode_val}, path={rel_file}, "
                    f"tex_size={tex_sz_val}, hframes={hf_val}, vframes={vf_val})"
                )
            _add(
                ContextChunk(
                    chunk_id=f"asset:catalog:{aid}",
                    category="asset",
                    title=chunk_title,
                    content=json.dumps(clean_entry, ensure_ascii=False, indent=2),
                    associated_image_path=media_path_str,
                )
            )

        # Also index any mockup PNGs on disk if not already indexed from `visual_mockups`
        mockups_dir = self.cartridge_dir / "assets" / "mockups"
        if mockups_dir.is_dir():
            existing_mockup_imgs = {
                Path(c.associated_image_path).resolve()
                for c in chunks
                if c.associated_image_path and Path(c.associated_image_path).suffix.lower() == ".png"
            }
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
                            f"UNVERIFIED_WORKER_CLAIMS={str(r.get('worker_notes') or '')[:1000]} | feedback={str(r.get('feedback') or '')[:900]}"
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
                    t_turns = t_data.get("turns") or []
                    if t_status == "in_progress" and not t_turns:
                        continue
                    t_strat = str(t_data.get("current_strategy") or "")[:300]
                    trace_lines.append(f"=== Execution Trace `{tf.name}` (rev={t_rev}, status={t_status}) ===")
                    if t_strat:
                        trace_lines.append(f"  Lead Strategy: {t_strat}")
                    for turn in t_turns[-6:]:
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
                            content="\n".join(trace_lines)[:2800],
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
        if code_graph:
            reach = compute_entrypoint_reachability(self.cartridge_dir, code_graph=code_graph)
            _add(
                ContextChunk(
                    chunk_id="code:entrypoint_topology",
                    category="code_module",
                    title=(
                        f"Playable Entrypoint Reachability Map (`{reach['main_scene']}`: "
                        f"{len(reach['reachable_from_entrypoint'])} reachable, "
                        f"{len(reach['unreferenced_from_entrypoint'])} unreferenced on disk)"
                    ),
                    content=reach["summary_text"],
                )
            )

        # 7. Long-Term Per-Game & Per-Agent Memory Ledger (`game_memory.json`)
        mem_path = self.cartridge_dir / "game_memory.json"
        if mem_path.is_file():
            try:
                from engine.studio.game_memory import GameMemoryStore

                ledger = GameMemoryStore(self.cartridge_dir).load_ledger()
                for entry in ledger.entries:
                    if entry.status != "active":
                        continue
                    roles_line = (
                        f"Applicable Roles: {', '.join(entry.applicable_roles)}\n"
                        if entry.applicable_roles
                        else ""
                    )
                    prov_line = (
                        f"Provenance: author={entry.author_role}"
                        + (f", slice={entry.slice_id}" if entry.slice_id else "")
                        + (f", item={entry.item_id}" if entry.item_id else "")
                        + "\n"
                    )
                    _add(
                        ContextChunk(
                            deliberation=entry.deliberation,
                            chunk_id=f"memory:{entry.memory_bank}:{entry.memory_id}",
                            category="game_memory",
                            title=f"Long-Term Memory [{entry.memory_bank} | {entry.kind}] — {entry.title}",
                            content=(
                                f"Memory ID: {entry.memory_id}\n"
                                f"Memory Bank: {entry.memory_bank}\n"
                                f"Kind: {entry.kind}\n"
                                f"{prov_line}"
                                f"{roles_line}"
                                f"Guidance: {entry.content}\n"
                                f"Rationale: {entry.deliberation}"
                            ),
                        )
                    )
            except Exception:
                logger.debug("Could not index game_memory.json chunks in %s", self.cartridge_dir, exc_info=True)

        # 8. Progressive Studio Tool & Persona Capability Registry (`tool_registry.py`)
        try:
            from engine.studio.tool_registry import build_tool_capability_chunks

            for t_raw in build_tool_capability_chunks():
                _add(
                    ContextChunk(
                        chunk_id=str(t_raw["chunk_id"]),
                        category="tool_capability",
                        title=f"Studio Tool Capability: {t_raw['symbol_name']}",
                        content=str(t_raw["content"]),
                    )
                )
        except Exception:
            logger.debug("Could not index tool_registry chunks in %s", self.cartridge_dir, exc_info=True)

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
        *,
        include_imported_by: bool = True,
        max_imported_by_hops: Optional[int] = None,
        exclude_prefixes: Sequence[str] = (),
    ) -> List[str]:
        """Walks `imports` and (optionally bounded) `imported_by` edges in `build_code_reference_graph()` up to
        `max_hops` starting from `seed_modules` (normalizing `res://` prefixes and bare basenames) and returns
        all discovered module relative paths in BFS order.

        When `max_imported_by_hops` is set (e.g. `1`), reverse `imported_by` edges are traversed only up to that
        depth and callers reached strictly via `imported_by` (such as hub scripts like `main.gd`) do not fan back
        out across unrelated sibling `imports`. Modules matching `exclude_prefixes` (e.g. `('tests/',)`) are
        skipped unless explicitly present in `seed_modules`.
        """
        graph = code_graph if code_graph is not None else self.build_code_reference_graph()
        if not graph:
            return []
        excl = tuple(
            str(p).strip().replace("\\", "/").lstrip("/")
            for p in (exclude_prefixes or ())
            if str(p).strip()
        )
        by_basename: Dict[str, List[str]] = {}
        for k in graph:
            by_basename.setdefault(Path(k).name, []).append(k)

        # Queue entries: (module_rel_path, depth, reached_via_reverse_edge)
        queue: List[Tuple[str, int, bool]] = []
        visited: List[str] = []
        explicit_seeds: set[str] = set()
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
                if excl and "/" not in s and any(rs.startswith(px) for px in excl):
                    continue
                explicit_seeds.add(rs)
                if rs not in visited:
                    visited.append(rs)
                    queue.append((rs, 0, False))

        idx = 0
        while idx < len(queue):
            cur_mod, depth, via_reverse = queue[idx]
            idx += 1
            if depth >= max_hops:
                continue
            # When max_imported_by_hops bounds caller lookup, do not fan out from a hub caller into all its unrelated imports
            if via_reverse and max_imported_by_hops is not None and depth >= max_imported_by_hops:
                continue
            meta = graph.get(cur_mod) or {}
            forward_nbs = list(meta.get("imports") or [])
            reverse_nbs: List[str] = []
            if include_imported_by and (max_imported_by_hops is None or depth < max_imported_by_hops):
                reverse_nbs = list(meta.get("imported_by") or [])

            for nb in forward_nbs:
                if excl and nb not in explicit_seeds and any(nb.startswith(px) for px in excl):
                    continue
                if nb in graph and nb not in visited:
                    visited.append(nb)
                    queue.append((nb, depth + 1, via_reverse))
            for nb in reverse_nbs:
                if excl and nb not in explicit_seeds and any(nb.startswith(px) for px in excl):
                    continue
                if nb in graph and nb not in visited:
                    visited.append(nb)
                    queue.append((nb, depth + 1, True))
        return visited

    def _load_embedding_cache(self) -> Dict[str, List[float]]:
        cache_key = str(self.index_path.resolve())
        mem_cache = self._MEMORY_EMBEDDING_CACHE.get(cache_key)
        if mem_cache is not None and self.index_path.is_file():
            return dict(mem_cache)
        if not self.index_path.is_file():
            self._MEMORY_EMBEDDING_CACHE.pop(cache_key, None)
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("embeddings"), dict):
                loaded = data["embeddings"]
                self._MEMORY_EMBEDDING_CACHE[cache_key] = dict(loaded)
                return loaded
        except Exception:
            pass
        return {}

    def _save_embedding_cache(self, cache: Dict[str, List[float]]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.EMBEDDING_MODEL, "dim": self.EMBEDDING_DIM, "embeddings": cache}
        self.index_path.write_text(json.dumps(payload), encoding="utf-8")
        self._MEMORY_EMBEDDING_CACHE[str(self.index_path.resolve())] = dict(cache)

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
        top_k: int = 16,
        per_category_min: int = 2,
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

        # Guarantee top `per_category_min` chunks from each category so the LLM Curator can see long-term memories,
        # assets, council conversations, design/rendering specs, board items, and code symbols
        for cat in ("game_memory", "asset", "council_utterance", "design_spec", "board_item", "code_module"):
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

    _CURATOR_MEDIA_MIMES: Dict[str, str] = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }
    _CURATOR_INLINE_MAX_BYTES: int = 18 * 1024 * 1024

    @staticmethod
    def _should_attach_concept_mockup(target_agent_role: str, item_kind: str) -> bool:
        """Fallback helper for non-Curator callers checking whether a task kind has a visual/spatial mockup."""
        kind_low = (item_kind or "").strip().lower()
        return kind_low in ("visual_asset", "animation", "level_map", "ui", "integration", "")

    @classmethod
    def _select_curated_media_paths(
        cls,
        target_agent_role: str,
        item_kind: str,
        curator_selected_paths: Sequence[str],
        extra_image_paths: Optional[Sequence[str]],
        cartridge_dir: Path,
        max_selected_images: int = 3,
    ) -> List[str]:
        """Resolves and caps LLM-selected multimodal media attachments (.png/.jpg/.webp images,
        .wav/.mp3/.ogg/.flac audio cues, and .mp4/.webm gameplay clips) without hardcoded role blocks."""
        role_low = (target_agent_role or "").strip().lower()
        is_swe = "swe" in role_low or ("engineer" in role_low and "audio" not in role_low and "qa" not in role_low)
        cap = max(1, int(max_selected_images or 3))

        def _resolve_valid(raw_p: str) -> Optional[Path]:
            if not raw_p:
                return None
            p = Path(raw_p) if Path(raw_p).is_absolute() else (cartridge_dir / raw_p)
            if p.is_file() and p.suffix.lower() in cls._CURATOR_MEDIA_MIMES:
                try:
                    if p.stat().st_size > 0:
                        return p
                except Exception:
                    return None
            return None

        final_paths: List[str] = []
        seen_resolved: set[Path] = set()
        total_bytes = 0

        for raw_p in list(curator_selected_paths or []):
            p = _resolve_valid(raw_p)
            if p is None:
                continue
            rp = p.resolve()
            if rp in seen_resolved:
                continue
            n_low = p.name.lower()
            # Never feed QA sprite alpha-checkerboard diagnostic plates as SWE code-generation textures
            if is_swe and n_low.startswith(("plate_", "anim_sheet_", "anim_rig_")):
                continue
            f_bytes = p.stat().st_size
            if total_bytes + f_bytes > cls._CURATOR_INLINE_MAX_BYTES:
                continue
            seen_resolved.add(rp)
            total_bytes += f_bytes
            final_paths.append(str(p))
            if len(final_paths) >= cap:
                break

        # If the caller explicitly provided captured proof artifacts (`extra_image_paths`) and budget remains,
        # attach at most one latest non-pass1 proof strip when none was already included
        if len(final_paths) < cap and extra_image_paths:
            proof_strips = [
                p for raw_e in extra_image_paths
                if (p := _resolve_valid(raw_e)) is not None
                and "filmstrip" in p.name.lower()
                and "_pass1." not in p.name.lower()
            ]
            if proof_strips:
                cand_strip = proof_strips[-1]
                rp_s = cand_strip.resolve()
                s_bytes = cand_strip.stat().st_size
                if rp_s not in seen_resolved and total_bytes + s_bytes <= cls._CURATOR_INLINE_MAX_BYTES:
                    seen_resolved.add(rp_s)
                    total_bytes += s_bytes
                    final_paths.append(str(cand_strip))

        return final_paths[:cap]

    @classmethod
    def _select_role_tailored_image_paths(
        cls,
        target_agent_role: str,
        item_kind: str,
        curator_selected_paths: Sequence[str],
        extra_image_paths: Optional[Sequence[str]],
        cartridge_dir: Path,
        max_selected_images: int = 3,
    ) -> List[str]:
        return cls._select_curated_media_paths(
            target_agent_role=target_agent_role,
            item_kind=item_kind,
            curator_selected_paths=curator_selected_paths,
            extra_image_paths=extra_image_paths,
            cartridge_dir=cartridge_dir,
            max_selected_images=max_selected_images,
        )

    @classmethod
    def _build_multimodal_parts_from_paths(
        cls,
        media_paths: Sequence[str],
        available_descriptions: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        from google.genai import types

        parts: List[Any] = []
        total_bytes = 0
        desc_map = available_descriptions or {}
        for m_str in media_paths:
            p = Path(m_str)
            ext = p.suffix.lower()
            mime = cls._CURATOR_MEDIA_MIMES.get(ext)
            if mime is None or not p.is_file():
                continue
            try:
                f_size = p.stat().st_size
            except Exception:
                continue
            if f_size <= 0 or total_bytes + f_size > cls._CURATOR_INLINE_MAX_BYTES:
                continue
            total_bytes += f_size
            desc = desc_map.get(m_str) or ""
            desc_suffix = f" — {desc}" if desc else ""
            n_low = p.name.lower()
            if mime.startswith("image/"):
                qa_overlay_note = ""
                if n_low.startswith("plate_"):
                    qa_overlay_note = " [QA ALPHA INSPECTION PLATE: dark/light checkerboard grids and cyan bounding box are external QA overlays, NOT inside the sprite PNG]"
                elif n_low.startswith("anim_sheet_") or n_low.startswith("anim_rig_"):
                    qa_overlay_note = " [QA ANIMATION CONTACT SHEET: dark checkerboard grid, cyan cell borders, and horizontal yellow/gold baseline ruler lines at the bottom of frames are external QA alignment guides, NOT inside the sprite sheet PNG]"
                elif "filmstrip" in n_low:
                    qa_overlay_note = " [QA CONTACT SHEET: top/bottom '[QA] tick ...' / '[QA] pos=...' strips are external QA margins, NOT inside the 16:9 viewport]"
                parts.append(f"CURATED REFERENCE / PROOF IMAGE ({p.name}{desc_suffix}){qa_overlay_note}:")
            elif mime.startswith("audio/"):
                parts.append(f"CURATED REFERENCE / PROOF AUDIO ({p.name}{desc_suffix}):")
            else:
                parts.append(f"CURATED REFERENCE / PROOF GAMEPLAY VIDEO+AUDIO CLIP ({p.name}{desc_suffix}):")
            parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime))
        return parts

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
                f"perceptual audio descriptions in asset_library.json, GDScript AudioStreamPlayer and res://assets/audio/*.wav bindings, "
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
        queries = [base_query, role_query]
        if (self.cartridge_dir / "game_memory.json").is_file():
            from engine.studio.game_memory import normalize_agent_role

            norm_role = normalize_agent_role(target_agent_role)
            queries.append(
                f"Long-term game memory for bank '{norm_role}' and 'council' — strategic goals, architectural decisions, "
                f"and operational lessons learned relevant to ({norm_role}): {task_request[:420]}"
            )
        return queries

    def curate_context(
        self,
        target_agent_role: str,
        task_request: str,
        item_id: str = "",
        item_kind: str = "",
        extra_image_paths: Optional[List[str]] = None,
        max_selected_chunks: int = 10,
        max_selected_images: int = 3,
        reuse_revision_cache: bool = True,
    ) -> CuratedContextPack:
        """Semantically searches the cartridge corpus and runs the LLM Context Generator decision to assemble
        a tailored `CuratedContextPack` (`prompt_block` + `multimodal_parts`).
        Reuses the curated pack within the same `(item_id, revision)` so multi-turn Task Lead loops,
        SWE synthesis, QA filmstrip planning, and Council review panelists share a stable byte-identical prefix
        while receiving role-appropriate multimodal media attachments."""
        if not is_real_genai_client(self.client):
            raise StudioAgentError(
                f"ContextCuratorAgent.curate_context requires an active google.genai Client for '{item_id}' (Rule 5)."
            )
        from google.genai import types
        from engine.studio.game_memory import normalize_agent_role

        safe_item = re.sub(r"[^a-z0-9_]+", "_", item_id.lower()).strip("_") if item_id else ""
        safe_role = re.sub(r"[^a-z0-9_]+", "_", target_agent_role.lower()).strip("_") or "agent"
        cache_role = self._normalized_cache_role(target_agent_role)
        norm_target_role = normalize_agent_role(target_agent_role)
        current_rev = self._current_item_revision(item_id)
        trace_dir = (self.cartridge_dir / "artifacts" / "work_items" / safe_item) if safe_item else None
        cache_file = (trace_dir / f"context_pack_{cache_role}.json") if trace_dir is not None else None

        mem_file = self.cartridge_dir / "game_memory.json"
        mem_sig = ""
        if mem_file.is_file():
            try:
                mem_sig = _sha256_text(mem_file.read_text(encoding="utf-8"))[:12]
            except Exception:
                mem_sig = ""

        deps_sig = ""
        if item_id and self.board is not None and hasattr(self.board, "items"):
            try:
                all_b_items = list(self.board.items())
                by_id_map = {str(getattr(it, "item_id", "")): it for it in all_b_items}
                cur_it = by_id_map.get(item_id)
                sig_parts: List[str] = []
                if cur_it is not None:
                    for dep_id in sorted(set(getattr(cur_it, "depends_on", None) or [])):
                        dep_it = by_id_map.get(dep_id)
                        if dep_it is not None:
                            sig_parts.append(
                                f"dep:{dep_id}:{getattr(dep_it, 'state', '')}:{len(getattr(dep_it, 'revisions', None) or [])}"
                            )
                for ch_it in all_b_items:
                    if str(getattr(ch_it, "spawned_by_item_id", "") or "") == item_id:
                        sig_parts.append(
                            f"child:{getattr(ch_it, 'item_id', '')}:{getattr(ch_it, 'state', '')}:{len(getattr(ch_it, 'revisions', None) or [])}"
                        )
                if sig_parts:
                    deps_sig = _sha256_text("|".join(sorted(sig_parts)))[:12]
            except Exception:
                deps_sig = ""

        intra_rev_sig = f"rev={current_rev}"

        with _CURATOR_CACHE_LOCK:
            candidate_cache_files: List[Path] = []
            if cache_file is not None:
                candidate_cache_files.append(cache_file)
                if cache_role in ("qa_test_engineer", "swe_engineer", "council_reviewer") and trace_dir is not None:
                    lead_cf = trace_dir / "context_pack_task_lead.json"
                    if lead_cf not in candidate_cache_files:
                        candidate_cache_files.append(lead_cf)
            for cf in candidate_cache_files if reuse_revision_cache else []:
                if not cf.is_file():
                    continue
                try:
                    cached = json.loads(cf.read_text(encoding="utf-8"))
                    if (
                        isinstance(cached, dict)
                        and cached.get("revision") == current_rev
                        and str(cached.get("memory_sig") or "") == mem_sig
                        and str(cached.get("deps_sig") or "") == deps_sig
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
                        cached_media_raw = (
                            cached.get("selected_media_paths")
                            or cached.get("selected_image_paths")
                            or decision.selected_media_paths
                            or []
                        )
                        chosen_media_paths = self._select_curated_media_paths(
                            target_agent_role=target_agent_role,
                            item_kind=item_kind,
                            curator_selected_paths=cached_media_raw,
                            extra_image_paths=extra_image_paths,
                            cartridge_dir=self.cartridge_dir,
                            max_selected_images=max_selected_images,
                        )
                        multimodal_parts = self._build_multimodal_parts_from_paths(chosen_media_paths)
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
                            selected_media_paths=chosen_media_paths,
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

            def _is_role_eligible_memory_chunk(ch: ContextChunk) -> bool:
                if ch.category != "game_memory":
                    return True
                if norm_target_role == "council":
                    return True
                is_own_bank = ch.chunk_id.startswith(f"memory:{norm_target_role}:")
                is_council_bank = ch.chunk_id.startswith("memory:council:")
                is_applicable = "Applicable Roles: " in ch.content and norm_target_role in ch.content
                return is_own_bank or is_council_bank or is_applicable

            searchable_chunks = [
                ch
                for ch in all_chunks
                if ch.category != "tool_capability" and _is_role_eligible_memory_chunk(ch)
            ]
            queries = self._build_role_queries(
                target_agent_role=target_agent_role,
                task_request=task_request,
                item_id=item_id,
                item_kind=item_kind,
            )
            ranked_pairs = self.semantic_index.semantic_search(queries=queries, chunks=searchable_chunks, top_k=16)
            candidate_chunks = [c for c, _ in ranked_pairs]
            chunks_by_id: Dict[str, ContextChunk] = {c.chunk_id: c for c in candidate_chunks}

            # Ensure role-relevant long-term memories (target agent's own bank, 'council' bank, or matching applicable_roles)
            # are surfaced in the candidate pool so ContextCuratorDecision can inject them when relevant
            role_memory_candidates: List[ContextChunk] = [
                ch for ch in searchable_chunks if ch.category == "game_memory"
            ]
            for m_ch in role_memory_candidates[:6]:
                if m_ch.chunk_id not in chunks_by_id:
                    prior_score = 0.94 if m_ch.chunk_id.startswith(f"memory:{norm_target_role}:") else 0.90
                    ranked_pairs.insert(0, (m_ch, prior_score))
                    candidate_chunks.insert(0, m_ch)
                    chunks_by_id[m_ch.chunk_id] = m_ch
            effective_extra_media: List[str] = [
                str(ep)
                for ep in (extra_image_paths or [])
                if ep and Path(ep).is_file() and Path(ep).suffix.lower() in self._CURATOR_MEDIA_MIMES
            ]
            if trace_dir is not None and trace_dir.is_dir():
                strip_candidates = [
                    p for p in sorted(trace_dir.glob("*filmstrip*.png")) if "_pass1." not in p.name.lower()
                ] or sorted(trace_dir.glob("*filmstrip*.png"))
                for strip_p in strip_candidates[-1:]:
                    if str(strip_p) not in effective_extra_media:
                        effective_extra_media.append(str(strip_p))
                clip_candidates = sorted(trace_dir.glob("*gameplay_clip*.mp4"))
                for clip_p in clip_candidates[-1:]:
                    if clip_p.stat().st_size <= self._CURATOR_INLINE_MAX_BYTES and str(clip_p) not in effective_extra_media:
                        effective_extra_media.append(str(clip_p))
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
                                and Path(ch.associated_image_path).suffix.lower() in self._CURATOR_MEDIA_MIMES
                            ):
                                if ch.associated_image_path not in effective_extra_media:
                                    effective_extra_media.insert(0, ch.associated_image_path)
                            break

            # Traverse 1 hop of the code reference graph (direct imports & imported_by) from target modules
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
            if item_kind in ("level_map", "ui", "integration"):
                _add_seed_mod("main.gd")

            allow_test_graph_neighbors = cache_role == "qa_test_engineer" or "tests/" in (task_request or "")
            graph_relations: Dict[str, str] = {}
            for sm in seed_mods:
                graph_relations.setdefault(sm, "target/seed module")
                meta = code_graph.get(sm) or {}
                for imp in meta.get("imports", []):
                    if not allow_test_graph_neighbors and imp.startswith("tests/"):
                        continue
                    if imp in code_graph and imp not in graph_relations:
                        graph_relations[imp] = f"imported by `{sm}`"
                for imp_by in meta.get("imported_by", []):
                    if not allow_test_graph_neighbors and imp_by.startswith("tests/"):
                        continue
                    if imp_by in code_graph and imp_by not in graph_relations:
                        graph_relations[imp_by] = f"imports `{sm}`"

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

            # Collect available multimodal media (.png/.jpg/.webp images, .wav cues, .mp4 gameplay clips)
            available_media: Dict[str, str] = {}
            for c in candidate_chunks:
                if (
                    c.associated_image_path
                    and Path(c.associated_image_path).is_file()
                    and Path(c.associated_image_path).suffix.lower() in self._CURATOR_MEDIA_MIMES
                ):
                    available_media[c.associated_image_path] = f"From {c.chunk_id} ({c.title})"
            for ep in effective_extra_media:
                if ep and Path(ep).is_file() and Path(ep).suffix.lower() in self._CURATOR_MEDIA_MIMES:
                    available_media[str(ep)] = f"Captured proof / reference media ({Path(ep).name})"

            candidate_lines: List[str] = []
            for c, sim in ranked_pairs[:22]:
                media_tag = f" | media={c.associated_image_path}" if c.associated_image_path else ""
                candidate_lines.append(
                    f"--- CHUNK id='{c.chunk_id}' category='{c.category}' similarity={sim:.3f}{media_tag} ---\n"
                    f"Title: {c.title}\n"
                    f"{c.content[:650]}\n"
                )

            media_block = (
                "\n".join(f"  - path='{p}': {desc}" for p, desc in available_media.items())
                if available_media
                else "  (none on disk yet)"
            )
            multimodal_guidance = (
                "Attach visual mockups, proof filmstrips, audio cues (.wav), or gameplay clips (.mp4) ONLY when "
                f"`{target_agent_role}`'s specific task genuinely benefits from seeing or hearing them (for example: "
                "attaching the 16:9 Concept Art Mockup for visual/spatial/mood alignment, attaching reference `.wav` "
                "cues or `.mp4` clips for perceptual audio/gameplay evaluation, or attaching the latest proof strip/clip "
                "when inspecting or fixing a capture). Leave empty when text specs and code suffice."
            )

            prompt = (
                f"You are the Studio Context Generator & Curator Agent.\n"
                f"Your job is to inspect the semantically retrieved candidate chunk digests (spanning long-term game & agent memories "
                f"`memory:*`, on-disk assets, Writers' Room council conversations, Game Bible & Rendering Strategy treatments, "
                f"Production Board items, build execution traces `board:trace:*`, and GDScript code modules) and decide the exact "
                f"high-signal context and multimodal media to inject for the target agent.\n\n"
                f"TARGET AGENT ROLE: {target_agent_role} (canonical memory bank: '{norm_target_role}')\n"
                f"WORK ITEM: id='{item_id}' kind='{item_kind}'\n"
                f"TASK REQUEST / BRIEF:\n{task_request[:1800]}\n\n"
                f"AVAILABLE MULTIMODAL MEDIA ON DISK (select up to {max_selected_images} relevant paths in `selected_media_paths` / `selected_image_paths`):\n"
                f"{media_block}\n\n"
                f"SEMANTICALLY RETRIEVED CANDIDATE CHUNK DIGESTS (select up to {max_selected_chunks} relevant `chunk_id`s in `selected_chunk_ids`; full chunk contents will be expanded automatically):\n"
                + "\n".join(candidate_lines)
                + "\nINSTRUCTIONS:\n"
                f"1. Select ONLY the chunks (`selected_chunk_ids`, up to {max_selected_chunks}) that `{target_agent_role}` genuinely needs to execute or review `{item_id}` with high visual, acoustic, mechanical, and architectural fidelity:\n"
                f"   - Inspect `memory:...` chunks (`game_memory` category—especially memories in `'council'`, `'{norm_target_role}'`, or listing `{norm_target_role}` in `Applicable Roles`) and ALWAYS select any long-term strategic goals, architectural decisions, or operational/verification lessons learned that apply to what `{target_agent_role}` is doing!\n"
                f"   - Include `board:item:{item_id}` and `board:trace:{item_id}` when `{target_agent_role}` needs the task spec, prior revision diffs, or captured frame telemetry.\n"
                f"   - Include only directly relevant Council utterances (`council:utterance:...`) or `design:...` treatments that govern this specific feature.\n"
                f"   - Include relevant `asset:catalog:...` chunks when `{target_agent_role}` needs to know which sprites, rigs, props, or audio cues exist on disk.\n"
                f"   - Inspect `code:module:...` chunks (especially `[GRAPH-LINKED: ...]` modules directly imported by or importing the task's target files) and select only the modules that interact with this task.\n"
                f"2. SMART MULTIMODAL MEDIA SELECTION (`selected_media_paths`, max {max_selected_images}): {multimodal_guidance} Never attach sprite alpha-inspection plates (`plate_*.png`) as SWE code-generation textures.\n"
                f"3. STRICT EPISTEMIC NEUTRALITY IN `role_tailored_synthesis`: Your purpose is strictly to gather and organize context so `{target_agent_role}` can solve the problem—NOT to solve the problem, prescribe fixes, or judge proofs yourself! In `role_tailored_synthesis`, provide ONLY a concise factual inventory of the selected chunks, cross-file dependency links (`imports` / `imported_by`), and attached artifacts/telemetry. NEVER claim that a proof image, audio cue, or telemetry 'confirms', 'proves', or 'satisfies' a requirement, and NEVER tell `{target_agent_role}` what code to write or what verdict to reach.\n"
                f"Provide `deliberation` FIRST."
            )

            curator_contents: Any = prompt
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

            # Resolve LLM-selected multimodal media parts (.png/.jpg/.webp, .wav, .mp4)
            chosen_media_paths = self._select_curated_media_paths(
                target_agent_role=target_agent_role,
                item_kind=item_kind,
                curator_selected_paths=decision.selected_media_paths,
                extra_image_paths=effective_extra_media,
                cartridge_dir=self.cartridge_dir,
                max_selected_images=max_selected_images,
            )

            multimodal_parts = self._build_multimodal_parts_from_paths(
                chosen_media_paths,
                available_descriptions=available_media,
            )

            expanded_chosen_chunks: List[ContextChunk] = []
            for c in chosen_chunks:
                if c.chunk_id.startswith("code:module:") and cache_role not in ("task_lead", "swe_engineer"):
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
                f"NEUTRAL CONTEXT MAP (factual inventory of retrieved chunks & cross-module links — evaluate and solve independently):\n"
                f"{decision.role_tailored_synthesis}\n\n"
                f"RETRIEVED & SELECTED STUDIO KNOWLEDGE CHUNKS ({len(expanded_chosen_chunks)} selected from {len(all_chunks)} indexed chunks):\n"
                f"{chunk_sections}\n"
                f"=== END CURATED STUDIO CONTEXT ===\n"
            )

            # Persist audit trace of what the Context Curator selected
            if trace_dir is not None and cache_file is not None:
                payload_to_save = {
                    "item_id": item_id,
                    "revision": current_rev,
                    "memory_sig": mem_sig,
                    "deps_sig": deps_sig,
                    "intra_rev_sig": intra_rev_sig,
                    "target_agent_role": target_agent_role,
                    "normalized_role": cache_role,
                    "total_indexed_chunks": len(all_chunks),
                    "candidate_chunk_ids": [c.chunk_id for c in candidate_chunks],
                    "selected_chunk_ids": [c.chunk_id for c in chosen_chunks],
                    "selected_image_paths": chosen_media_paths,
                    "selected_media_paths": chosen_media_paths,
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
                            "selected_image_paths": [Path(p).name for p in chosen_media_paths],
                            "selected_media_paths": [Path(p).name for p in chosen_media_paths],
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
                selected_media_paths=chosen_media_paths,
            )

    def _expand_chunk_for_question(self, chunk: ContextChunk, *, target_agent_role: str = "") -> ContextChunk:
        """When a `code:module:<rel_path>` chunk is selected to help an agent answer a `self_question`,
        expands the chunk content with bounded, 1-indexed on-disk source code of that module so the asking agent
        can inspect exact method bodies, signal definitions, and shader uniforms without unbounded prompt bloat."""
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
        role_norm = self._normalized_cache_role(target_agent_role)
        max_src_chars = 3200 if role_norm == "task_lead" else 6000
        numbered_lines = [f"{ln:4d}: {line}" for ln, line in enumerate(raw_src.splitlines(), start=1)]
        numbered_src = "\n".join(numbered_lines)
        if len(numbered_src) > max_src_chars:
            excerpt = numbered_src[:max_src_chars] + f"\n# ... [{len(numbered_src) - max_src_chars} chars omitted; ask a narrower line/function self_question if needed]"
        else:
            excerpt = numbered_src
        expanded_content = (
            f"{chunk.content}\n\n"
            f"--- NUMBERED SOURCE EXCERPT OF `{rel_path}` ---\n"
            f"{excerpt}"
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

        eff_max_chunks = min(max_selected_chunks, 3) if self._normalized_cache_role(target_agent_role) == "task_lead" else max_selected_chunks
        if is_real_genai_client(self.client):
            candidate_lines: List[str] = []
            for c, sim in filtered_pairs[:16]:
                candidate_lines.append(
                    f"--- CHUNK id='{c.chunk_id}' category='{c.category}' similarity={sim:.3f} ---\n"
                    f"Title: {c.title}\n"
                    f"{c.content[:1200]}\n"
                )
            prompt = (
                f"You are the Studio Context Generator & Curator Agent researching a self-inquiry question asked by `{target_agent_role}`.\n"
                f"CRITICAL RULE: Do NOT answer the question yourself! Your ONLY job is to select the candidate `chunk_id`s "
                f"(up to {eff_max_chunks}) that contain the raw code, signals, variables, council decisions, or design specs "
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
                        chosen_chunks.append(self._expand_chunk_for_question(chunks_by_id[cid], target_agent_role=target_agent_role))
                        seen_ids.add(cid)
                        if len(chosen_chunks) >= eff_max_chunks:
                            break

        if not chosen_chunks and candidate_chunks:
            chosen_chunks = [
                self._expand_chunk_for_question(c, target_agent_role=target_agent_role)
                for c in candidate_chunks[: min(3, eff_max_chunks)]
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

    def research_tool_capabilities(
        self,
        tool_search_query: str,
        *,
        caller_role: str = "task_lead",
        item_id: str = "",
        allowed_specialists: Optional[Sequence[str]] = None,
        max_selected_chunks: int = 5,
        top_k: Optional[int] = None,
    ) -> str:
        """Progressively searches `STUDIO_CAPABILITIES` (`category='tool_capability'`) for detailed
        Specialist Persona schemas, internal sub-tools, parameters, and board orchestration recipes."""
        clean_q = str(tool_search_query or "").strip()
        if not clean_q:
            return ""
        eff_limit = int(top_k) if top_k is not None else int(max_selected_chunks)

        from engine.studio.tool_registry import build_tool_capability_chunks

        raw_tool_chunks = build_tool_capability_chunks(allowed_specialists=allowed_specialists)
        tool_chunks = [
            ContextChunk(
                chunk_id=str(tc["chunk_id"]),
                category="tool_capability",
                title=f"Studio Tool Capability: {tc['symbol_name']}",
                content=str(tc["content"]),
            )
            for tc in raw_tool_chunks
        ]
        if not tool_chunks:
            return ""

        ranked = self.semantic_index.semantic_search(
            queries=[
                clean_q,
                f"Studio tool capability, specialist persona, subtool, or board orchestration for ({caller_role}): {clean_q}",
            ],
            chunks=tool_chunks,
            top_k=min(max(1, eff_limit), len(tool_chunks)),
        )
        chosen = [ch for ch, _sim in ranked[:eff_limit]] or tool_chunks[: min(3, len(tool_chunks))]
        sections = "\n\n".join(f"[{c.chunk_id}] {c.title}:\n{c.content}" for c in chosen)
        block = (
            f"\n=== STUDIO TOOL CAPABILITIES RESEARCHED FOR YOUR QUERY (caller={caller_role}) ===\n"
            f"YOUR TOOL SEARCH QUERY: {clean_q}\n\n"
            f"{sections}\n\n"
            f"NEXT-TURN INSTRUCTION: Summarize the relevant tool modes/parameters in `tool_search_answer` and proceed with your action.\n"
            f"=== END RESEARCHED STUDIO TOOL CAPABILITIES ===\n"
        )
        if self.event_callback is not None:
            try:
                self.event_callback(
                    "context_curator_tool_search_completed",
                    {
                        "deliberation": f"Expanded {len(chosen)} Studio tool capability chunk(s) for '{caller_role}' query: {clean_q[:120]}",
                        "item_id": item_id,
                        "caller_role": caller_role,
                        "tool_search_query": clean_q,
                        "selected_chunk_ids": [c.chunk_id for c in chosen],
                    },
                )
            except Exception:
                pass
        return block

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


def compute_entrypoint_reachability(
    cartridge_dir: Path,
    code_graph: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Computes the factual forward module reachability graph starting from `project.godot`'s `run/main_scene`
    and any `[autoload]` singletons, partitioning production `.gd`/`.tscn`/`.gdshader` files (excluding `tests/**`
    and `src/studio_rig_and_collider.gd`) into `reachable_from_entrypoint` vs `unreferenced_from_entrypoint`."""
    c_dir = Path(cartridge_dir)
    if code_graph is None:
        idx = CartridgeSemanticIndex(cartridge_dir=c_dir, client=None)
        code_graph = idx.build_code_reference_graph()

    main_scene_res = "res://main.tscn"
    autoload_roots: List[str] = []
    proj_path = c_dir / "project.godot"
    if proj_path.is_file():
        try:
            proj_text = proj_path.read_text(encoding="utf-8", errors="replace")
            m_main = re.search(r'^\s*run/main_scene\s*=\s*"([^"]+)"', proj_text, flags=re.MULTILINE)
            if m_main and m_main.group(1).strip():
                main_scene_res = m_main.group(1).strip()
            in_autoload = False
            for raw_ln in proj_text.splitlines():
                ln = raw_ln.strip()
                if ln.startswith("[") and ln.endswith("]"):
                    in_autoload = ln.lower() == "[autoload]"
                    continue
                if in_autoload and "=" in ln:
                    m_auto = re.search(r'"\*?(?:res://)?([^"]+)"', ln)
                    if m_auto:
                        auto_rel = m_auto.group(1).strip().lstrip("/")
                        if auto_rel and auto_rel not in autoload_roots:
                            autoload_roots.append(auto_rel)
        except Exception:
            pass

    main_rel = main_scene_res.replace("res://", "").lstrip("/")
    seed_roots: List[str] = []
    if main_rel in code_graph:
        seed_roots.append(main_rel)
    elif "main.tscn" in code_graph:
        seed_roots.append("main.tscn")
    elif "main.gd" in code_graph:
        seed_roots.append("main.gd")
    for ar in autoload_roots:
        if ar in code_graph and ar not in seed_roots:
            seed_roots.append(ar)

    queue: List[str] = list(seed_roots)
    reachable_set: set[str] = set(seed_roots)
    q_idx = 0
    while q_idx < len(queue):
        cur = queue[q_idx]
        q_idx += 1
        meta = code_graph.get(cur) or {}
        for imp in meta.get("imports") or []:
            if imp in code_graph and imp not in reachable_set:
                reachable_set.add(imp)
                queue.append(imp)

    production_modules = sorted(
        k
        for k in code_graph
        if not k.startswith("tests/") and k != "src/studio_rig_and_collider.gd"
    )
    reachable_prod = [m for m in production_modules if m in reachable_set]
    unreferenced_prod = [m for m in production_modules if m not in reachable_set]

    lines = [
        f"Playable Entrypoint (`project.godot` -> `run/main_scene`): {main_scene_res}",
        f"Entrypoint Root Files: {seed_roots}",
        f"Autoload Singletons: {autoload_roots or 'none'}",
        f"Production Modules Reachable from Entrypoint ({len(reachable_prod)}): {reachable_prod}",
        f"Production Modules on Disk NOT Currently Referenced from Entrypoint ({len(unreferenced_prod)}): {unreferenced_prod}",
    ]
    if unreferenced_prod:
        lines.append("Unreferenced Module Summaries:")
        for u_mod in unreferenced_prod[:16]:
            u_meta = code_graph.get(u_mod) or {}
            u_sum = str(u_meta.get("summary_content") or "").splitlines()[:4]
            lines.append("  - " + " | ".join(u_sum))

    return {
        "main_scene": main_scene_res,
        "entrypoint_roots": seed_roots,
        "autoload_roots": autoload_roots,
        "reachable_from_entrypoint": reachable_prod,
        "unreferenced_from_entrypoint": unreferenced_prod,
        "summary_text": "\n".join(lines),
    }


