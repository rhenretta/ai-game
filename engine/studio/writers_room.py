"""engine/studio/writers_room.py
Experience-First Multi-Agent Creative Writers' Room & Living Game Bible Orchestrator.
Replaces single-shot concept generation with an open-ended, dynamic multi-specialist
brainstorming room where specialists listen to the conversation, interject critiques
and creative artifacts when relevant, synthesize 16:9 visual mockups, portraits, and
voice auditions mid-conversation, and iteratively build out a Living Game Bible
until the Creative Director is satisfied.

Enforces:
- Rule 1: Pure Generalizability (100% domain-agnostic across any genre/world)
- Rule 2: Zero Regex/Keyword Heuristics (all listening/interjection/convergence decisions via LLM JSON schemas)
- Rule 4: Mandatory Rationale-First Epistemic Architecture (deliberation strictly field index 0)
- Rule 5: Zero Silent Fallbacks (fails fast with StudioAgentError if client is missing or API fails)
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import os
import random
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.steam.models import SteamPlayerDNA
from engine.studio.models import (
    BlindPlaytestCouncilTurn,
    BlindPlaytestCouncilVerdict,
    CharacterSheet,
    ConceptCandidate,
    ConceptCollisionProposal,
    ConceptEvolutionEntry,
    ConceptFragment,
    ConceptSeedProposal,
    ConceptSelection,
    CouncilDiagnosticProbe,
    CouncilSteeringRoutingPlan,
    CreativeProvocationSet,
    validate_game_design,
    DirectorConvergenceCheck,
    ExperienceStoryboardBeat,
    GameBibleSpec,
    GameSliceSpec,
    GenesisWorldSpec,
    MechanicEngineeringSpec,
    NarrativeBible,
    NextSliceStrategyPlan,
    NoveltyJudgement,
    ReviewBatch,
    ReviewDecision,
    SliceAssemblyDecision,
    SlicePlaytestRoundtableTurn,
    SliceStateReflectionTurn,
    StudioAgentError,
    TaskShapingBatch,
    VisualMockupArtifact,
    WorkItemRefinement,
    WritersRoomListenCheck,
    WritersRoomTurnProposal,
    WritersRoomUtterance,
    StudioPausedError,
    bind_explicit_prefix_cache_scope,
    bounded_thinking_config,
    call_gemini_with_transient_retry,
    explicit_prefix_cache_scope,
    get_active_explicit_prefix_cache_scope,
    is_candidate_build_aborted,
    is_real_genai_client,
    resolve_explicit_cached_request,
)

from engine.studio.production_board import MAX_SPEC_RESETS, BoardScheduler, ProductionBoard
from engine.studio.tool_registry import render_compact_tool_directory

logger = logging.getLogger("CreativeWritersRoom")


CORE_COUNCIL_ROSTER: List[Dict[str, Any]] = [
    {
        "role": "GameplayDesigner",
        "display_name": "Lead Gameplay & Systems Designer",
        "tier": "lead",
        "subagents": ["LevelAndProgressionDesigner", "GameplaySystemsEngineer"],
        "focus": (
            "Tactile feel, core game loop, emergent mechanical depth, moment-to-moment player agency, "
            "and why the mechanics are genuinely fun and immediately playable in a 2D viewport. "
            "Can call on sub-agents LevelAndProgressionDesigner and GameplaySystemsEngineer whenever progression, "
            "world structure, or Godot 4 systems engineering strategy is needed."
        ),
    },
    {
        "role": "StoryWorldArchitect",
        "display_name": "Story & World Lore Architect",
        "tier": "lead",
        "subagents": ["ScriptwriterAgent", "CharacterDesigner"],
        "focus": (
            "Imaginative world-building, central dramatic conflict, environmental storytelling, "
            "narrative stakes, and narrative beats that pull the player into a vivid, living world. "
            "Can call on sub-agents ScriptwriterAgent (for scene scripts, dialogue writing, and dramatic pacing) "
            "and CharacterDesigner (for character sheets, motivations, and casting) whenever needed."
        ),
    },
    {
        "role": "VisualArtDirector",
        "display_name": "Visual Art & Mockup Director",
        "tier": "lead",
        "subagents": ["UXExperienceDesigner", "AudioAtmosphereDesigner"],
        "focus": (
            "Cohesive visual identity, color mood, lighting, camera staging, procedural/shader vs. sprite art direction, "
            "and authoring 16:9 full-screen VisualMockupArtifact boards that show an unobstructed, readable game world and UI. "
            "Can call on sub-agents UXExperienceDesigner and AudioAtmosphereDesigner whenever player experience, onboarding/UI, "
            "or audio/voice atmosphere strategy is needed."
        ),
    },
]

SUBAGENT_ROSTER: List[Dict[str, Any]] = [
    {
        "role": "LevelAndProgressionDesigner",
        "display_name": "Level, Progression & Goals Designer",
        "tier": "subagent",
        "parent_lead": "GameplayDesigner",
        "focus": (
            "Strategizing a bespoke world and level construction plan tailored specifically to this game (`generation_approach`, "
            "`world_structure`, and `runtime_construction_plan` in `scenes.gd` — devising whatever world structure, scale, or "
            "generation algorithm best serves the game's core loop), how difficulty and variety evolve, and the player's short-, mid- and long-term goals."
        ),
    },
    {
        "role": "GameplaySystemsEngineer",
        "display_name": "Lead Gameplay & Godot Systems Engineer",
        "tier": "subagent",
        "parent_lead": "GameplayDesigner",
        "focus": (
            "Translating creative gameplay pillars, spatial camera staging, and interactions into concrete "
            "Godot 4 node architectures (CharacterBody2D, StaticBody2D, Area2D, Polygon2D/Line2D/ShaderMaterial, AnimatedSprite2D), "
            "1:1 visual-to-physical collider alignment, deterministic state machines (unsolved at t=0), procedural rendering vs. discrete assets, "
            "and falsifiable headless test invariants."
        ),
    },
    {
        "role": "ScriptwriterAgent",
        "display_name": "Lead Scriptwriter & Narrative Dramatist",
        "tier": "subagent",
        "parent_lead": "StoryWorldArchitect",
        "focus": (
            "Dramatic scene writing, screenplay structure, sharp character dialogue, branching conversational trees, "
            "contextual/reactive barks, subtext, comedic or dramatic timing, and how story moments play out on screen "
            "when the game calls for scripted scenes, dialogue, or character voice."
        ),
    },
    {
        "role": "CharacterDesigner",
        "display_name": "Character & Dialogue Designer",
        "tier": "subagent",
        "parent_lead": "StoryWorldArchitect",
        "focus": (
            "Memorable characters or entities with distinct silhouettes, personalities, motivations, "
            "relationships, and reactive voice signatures when appropriate to the game."
        ),
    },
    {
        "role": "UXExperienceDesigner",
        "display_name": "Player Experience & UX Designer",
        "tier": "subagent",
        "parent_lead": "VisualArtDirector",
        "focus": (
            "Empathetic new-player experience (putting yourself in the shoes of a brand-new player who has zero prior context "
            "and must intuitively figure out who they are, how the world works, what can be interacted with, and what their goal is), "
            "the First 60 Seconds Hook, Experience Storyboard, and cohesive UI/HUD/feedback architecture."
        ),
    },
    {
        "role": "AudioAtmosphereDesigner",
        "display_name": "Audio & Voice Atmosphere Designer",
        "tier": "subagent",
        "parent_lead": "VisualArtDirector",
        "focus": (
            "Acoustic atmosphere, reactive sound design, character voice casting, and emotional cadence."
        ),
    },
]

SPECIALIST_ROSTER: List[Dict[str, Any]] = CORE_COUNCIL_ROSTER + SUBAGENT_ROSTER

COUNCIL_HIERARCHY: Dict[str, List[str]] = {
    lead["role"]: list(lead.get("subagents", []))
    for lead in CORE_COUNCIL_ROSTER
}


def _resolve_council_specialist(raw_role: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolves a Council specialist dictionary by exact (case-insensitive) role name in `SPECIALIST_ROSTER`."""
    cleaned = str(raw_role or "").strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    for spec in SPECIALIST_ROSTER:
        if spec["role"].lower() == lowered:
            return spec
    return None


def _parse_council_reroute(proposal: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """Extracts `(specialist_dict, handoff_reason)` from `proposal.route_to_council_role` (supporting
    both `RoleName` + `routing_handoff_reason` and compact `RoleName: handoff reason` strings)."""
    raw_val = str(getattr(proposal, "route_to_council_role", "") or "").strip()
    explicit_reason = str(getattr(proposal, "routing_handoff_reason", "") or "").strip()
    if not raw_val:
        return None, explicit_reason
    if ":" in raw_val:
        role_part, inline_reason = raw_val.split(":", 1)
        spec = _resolve_council_specialist(role_part.strip())
        if spec is not None:
            return spec, explicit_reason or inline_reason.strip()
    return _resolve_council_specialist(raw_val), explicit_reason

STUDIO_CRITICAL_DISCOURSE_CHARTER: str = (
    "STUDIO DISCOURSE, EXECUTIVE PRODUCER RESPECT & PRAGMATIC SLICE SCOPE CHARTER:\n"
    "1. SKEPTICAL, ZERO-SYCOPHANCY PEER DISCOURSE (NO CIRCLE-JERK): Never open `spoken_dialogue` with agreement filler, "
    "peer roll-calls, or flattery ('I completely agree with...', 'I echo your assessments...', 'I couldn't agree more...', "
    "'Spot on...', 'I am thrilled...', 'feels wonderful'). Never repeat points an earlier colleague already made. "
    "Scrutinize peer proposals, worker claims, rendered frames, telemetry, and code with healthy skepticism—open immediately "
    "with net-new domain critique, unverified assumptions, concrete flaws/trade-offs, or pushback against premature praise.\n"
    "2. FULL RESPECT & CATERING TO THE EXECUTIVE PRODUCER: Skeptical pushback applies ONLY to peer agents and our own builds—"
    "NEVER argue with, lecture, dismiss, make excuses to, or silently water down messages from the Executive Producer "
    "(`ExecutiveProducer`, `UserSteering`, Creative Seed, or Slice Directives). Executive Producer messages are always treated "
    "with full respect, and your job is to cater to them faithfully (Rule 5), applying critical rigor to why our build fell short "
    "and how to deliver what the Executive Producer asked for.\n"
    "3. PRAGMATIC SLICE SCOPE & CALLING THE SLICE DONE WHEN APPROPRIATE: You cannot build or fix everything in one go. "
    "Keep your diagnosis candid (never pretend rough visuals or UX are 'flawless' just to approve), while keeping your actions "
    "bounded to the active slice: cover the active slice's core scope, clean up a few high-impact problem areas if needed "
    "(1–3 surgical tasks rather than boiling the ocean), defer broader expansions/wishlist items to future slices, and call the "
    "slice done when its scope is genuinely covered and major problem areas are cleaned up."
)


def _normalize_char_tokens(text: str, drop_generic: bool = False) -> set[str]:
    """Extracts lowercase alphanumeric word tokens for domain-agnostic character identity matching."""
    import re
    generic = {"npc", "char", "character", "the", "and", "of"}
    tokens = {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) >= 2}
    if drop_generic:
        tokens -= generic
    return tokens


def consolidate_character_list(
    existing_chars: List[Any],
    incoming_chars: List[Any],
) -> Tuple[List[Any], Dict[str, str]]:
    """Merges incoming character sheets (Pydantic CharacterSheet or dict) into existing_chars
    without creating duplicate entries when specialists vary character_id or expand a character's name
    across turns. Returns (consolidated_list, alias_id_to_canonical_id).
    """
    consolidated: List[Any] = list(existing_chars)
    alias_map: Dict[str, str] = {}

    def _get(obj: Any, key: str, default: Any = "") -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _set(obj: Any, key: str, val: Any) -> None:
        if isinstance(obj, dict):
            obj[key] = val
        else:
            setattr(obj, key, val)

    for inc in incoming_chars:
        inc_id = str(_get(inc, "character_id", "")).strip()
        inc_name = str(_get(inc, "name", "")).strip()
        inc_portrait = str(_get(inc, "portrait_asset", "")).strip()
        inc_portrait_base = Path(inc_portrait).name if inc_portrait else ""
        inc_name_tokens = _normalize_char_tokens(inc_name, drop_generic=True)
        inc_id_tokens = _normalize_char_tokens(inc_id, drop_generic=True)

        matched_idx: Optional[int] = None
        for idx, cur in enumerate(consolidated):
            cur_id = str(_get(cur, "character_id", "")).strip()
            cur_name = str(_get(cur, "name", "")).strip()
            cur_portrait = str(_get(cur, "portrait_asset", "")).strip()
            cur_portrait_base = Path(cur_portrait).name if cur_portrait else ""

            # 1. Exact character_id match
            if inc_id and cur_id and inc_id.lower() == cur_id.lower():
                matched_idx = idx
                break

            # 2. Same non-default portrait asset filename
            if (
                inc_portrait_base
                and cur_portrait_base
                and inc_portrait_base == cur_portrait_base
                and not inc_portrait_base.startswith("portrait_char_")
            ):
                matched_idx = idx
                break

            # 3. Name token subset match (e.g., "Dispatcher Vance" vs "Dispatcher Corvin Vance")
            cur_name_tokens = _normalize_char_tokens(cur_name, drop_generic=True)
            if inc_name_tokens and cur_name_tokens:
                if inc_name_tokens.issubset(cur_name_tokens) or cur_name_tokens.issubset(inc_name_tokens):
                    matched_idx = idx
                    break

            # 4. ID token subset match
            cur_id_tokens = _normalize_char_tokens(cur_id, drop_generic=True)
            if inc_id_tokens and cur_id_tokens and len(inc_id_tokens & cur_id_tokens) >= 2:
                if inc_id_tokens.issubset(cur_id_tokens) or cur_id_tokens.issubset(inc_id_tokens):
                    matched_idx = idx
                    break

        if matched_idx is not None:
            cur = consolidated[matched_idx]
            canonical_id = str(_get(cur, "character_id", inc_id)).strip() or inc_id
            if inc_id:
                alias_map[inc_id] = canonical_id
            # Preserve the canonical character_id while adopting richer updated fields
            _set(inc, "character_id", canonical_id)
            if len(inc_name) < len(str(_get(cur, "name", ""))) and str(_get(cur, "name", "")):
                _set(inc, "name", _get(cur, "name"))
            if (not inc_portrait or inc_portrait.startswith(f"assets/sprites/portrait_{inc_id}")) and _get(cur, "portrait_asset"):
                _set(inc, "portrait_asset", _get(cur, "portrait_asset"))
            consolidated[matched_idx] = inc
        else:
            if inc_id:
                alias_map[inc_id] = inc_id
            consolidated.append(inc)

    return consolidated, alias_map


class CreativeWritersRoom:
    """Open-ended multi-agent conversational Writers' Room that iteratively constructs a Living Game Bible."""

    deliberation: str = "Open-ended multi-specialist creative brainstorming room for experience-first game design."

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = "gemini-3.8-flash",
        visual_agent: Optional[Any] = None,
        narrative_agent: Optional[Any] = None,
        audio_agent: Optional[Any] = None,
        telemetry_db: Optional[Any] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        max_turns: int = 12,
        min_turns_before_convergence: int = 4,
        creative_temperature: float = 0.7,
        max_parallel_calls: Optional[int] = None,
        board: Optional[ProductionBoard] = None,
        scheduler: Optional[BoardScheduler] = None,
        drain_timeout_s: float = 3600.0,
        drain_poll_s: float = 0.5,
        target_slices: Optional[int] = None,
        slice_checkpoint_callback: Optional[Callable[[GameBibleSpec, GameSliceSpec, Optional[Path]], None]] = None,
        playtest_manager: Optional[Any] = None,
    ):
        self.client = client
        self.model_name = model_name
        self.visual_agent = visual_agent
        self.narrative_agent = narrative_agent
        self.audio_agent = audio_agent
        self.telemetry_db = telemetry_db
        self.event_callback = event_callback
        self.max_turns = max(2, int(max_turns))
        self.min_turns_before_convergence = max(1, int(min_turns_before_convergence))
        self.creative_temperature = float(creative_temperature)
        self.board = board
        self.scheduler = scheduler
        self.drain_timeout_s = float(drain_timeout_s)
        self.drain_poll_s = float(drain_poll_s)
        self.target_slices = target_slices
        self.slice_checkpoint_callback = slice_checkpoint_callback
        self.playtest_manager = playtest_manager
        # Independent LLM calls (listen checks, concept seeds, novelty judgements) run concurrently.
        env_par = os.environ.get("STUDIO_ROOM_PARALLELISM")
        self.max_parallel_calls = max(1, int(max_parallel_calls if max_parallel_calls is not None else (env_par or 2)))
        self._steering_lock = threading.Lock()
        self._pending_steering: List[str] = []
        self._abort_requested: bool = False

    def _check_aborted(self, candidate_dir: Optional[Path] = None) -> None:
        if getattr(self, "_abort_requested", False) or is_candidate_build_aborted(candidate_dir):
            if self.scheduler is not None and hasattr(self.scheduler, "stop"):
                try:
                    self.scheduler.stop()
                except Exception:
                    pass
            raise StudioPausedError("Writers' Room build cancelled/preempted by a new game request.")

    def enqueue_user_steering(self, feedback: str, candidate_dir: Optional[Path] = None) -> None:
        """Thread-safely queues live Executive Producer steering feedback for immediate consumption by the Writers' Room."""
        clean = (feedback or "").strip()
        if not clean:
            return
        with self._steering_lock:
            if clean not in self._pending_steering:
                self._pending_steering.append(clean)
        if candidate_dir is not None:
            try:
                c_path = Path(candidate_dir)
                if c_path.is_dir():
                    sf = c_path / ".pending_steering.json"
                    existing: List[str] = []
                    if sf.is_file():
                        try:
                            raw = json.loads(sf.read_text(encoding="utf-8"))
                            if isinstance(raw, list):
                                existing = [str(x).strip() for x in raw if str(x).strip()]
                        except Exception:
                            existing = []
                    if clean not in existing:
                        existing.append(clean)
                    sf.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            except Exception as exc:
                logger.debug("Failed writing .pending_steering.json: %s", exc)

    def _consume_pending_steering(
        self,
        game_bible: GameBibleSpec,
        candidate_dir: Optional[Path] = None,
    ) -> List[str]:
        """Consumes any queued live Executive Producer steering notes from memory or disk and records them on `game_bible`."""
        queued: List[str] = []
        with self._steering_lock:
            if self._pending_steering:
                queued.extend(self._pending_steering)
                self._pending_steering.clear()
        if candidate_dir is not None:
            sf = Path(candidate_dir) / ".pending_steering.json"
            if sf.is_file():
                try:
                    raw = json.loads(sf.read_text(encoding="utf-8"))
                    if isinstance(raw, list):
                        for item in raw:
                            s = str(item or "").strip()
                            if s and s not in queued:
                                queued.append(s)
                    sf.unlink(missing_ok=True)
                except Exception as exc:
                    logger.debug("Failed reading .pending_steering.json: %s", exc)
            wf = Path(candidate_dir) / ".pending_playtest_wishes.json"
            if wf.is_file() and not queued:
                try:
                    raw_w = json.loads(wf.read_text(encoding="utf-8"))
                    if isinstance(raw_w, list) and raw_w:
                        summaries = [
                            f"[{w.get('category', 'wish')}] {w.get('summary', '')} (slice={w.get('observed_on_slice_id', 'unspecified')})"
                            for w in raw_w
                            if isinstance(w, dict)
                        ]
                        if summaries:
                            queued.append(
                                "[BLIND PLAYTESTER APPROVED WISHES]: " + "; ".join(summaries)
                            )
                except Exception as exc:
                    logger.debug("Failed peeking .pending_playtest_wishes.json: %s", exc)
            rf = Path(candidate_dir) / ".pending_council_playtest_reports.json"
            if rf.is_file() and not queued:
                try:
                    raw_r = json.loads(rf.read_text(encoding="utf-8"))
                    if isinstance(raw_r, list) and raw_r:
                        rep_summaries = []
                        for env in raw_r:
                            if not isinstance(env, dict):
                                continue
                            s_rep = env.get("session_report") or {}
                            rep_summaries.append(
                                f"Session {env.get('session_id', '?')} (slice={env.get('observed_on_slice_id', 'unspecified')}, "
                                f"fun={s_rep.get('overall_fun_rating', '?')}/10, wishes={len(env.get('wishes') or [])}): "
                                f"{s_rep.get('overall_experience_summary', '')}"
                            )
                        if rep_summaries:
                            queued.append("[COUNCIL BLIND PLAYTEST REPORT & WISHES]: " + " | ".join(rep_summaries))
                except Exception as exc:
                    logger.debug("Failed peeking .pending_council_playtest_reports.json: %s", exc)

        # Also check if any ExecutiveProducer utterance is in transcript without a matching ConceptEvolutionEntry
        pivoted_texts = {
            e.pivot_reason.strip()
            for e in game_bible.concept_evolution
            if e.author_role in ("ExecutiveProducer", "UserSteering") and e.pivot_reason
        }
        for u in game_bible.writers_room_transcript:
            if u.speaker_role in ("ExecutiveProducer", "UserSteering") and u.spoken_dialogue.strip():
                if u.spoken_dialogue.strip() not in pivoted_texts and u.spoken_dialogue.strip() not in queued:
                    queued.append(u.spoken_dialogue.strip())

        if not queued:
            return []

        existing_utt_texts = {
            u.spoken_dialogue.strip()
            for u in game_bible.writers_room_transcript
            if u.speaker_role in ("ExecutiveProducer", "UserSteering") and u.spoken_dialogue
        }
        consumed: List[str] = []
        for fb in queued:
            if not fb:
                continue
            consumed.append(fb)
            turn_index = len(game_bible.writers_room_transcript)
            if fb not in existing_utt_texts:
                user_utt = WritersRoomUtterance(
                    deliberation="Executive Producer injected live creative steering from Mission Control.",
                    turn_index=turn_index,
                    speaker_role="ExecutiveProducer",
                    speaker_display_name="Executive Producer (You)",
                    spoken_dialogue=fb,
                    addressed_to=["CreativeDirector", "GameplayDesigner", "GameplaySystemsEngineer", "UXExperienceDesigner"],
                    bible_updates_summary=["Executive Producer creative direction injected"],
                )
                game_bible.writers_room_transcript.append(user_utt)
                existing_utt_texts.add(fb)
                self._emit_event(
                    "writers_room_utterance",
                    {
                        "deliberation": user_utt.deliberation,
                        "utterance_id": user_utt.utterance_id,
                        "turn_index": user_utt.turn_index,
                        "speaker_role": user_utt.speaker_role,
                        "speaker_display_name": user_utt.speaker_display_name,
                        "spoken_dialogue": user_utt.spoken_dialogue,
                        "addressed_to": user_utt.addressed_to,
                        "bible_updates_summary": user_utt.bible_updates_summary,
                    },
                )
            if fb not in pivoted_texts:
                evo = ConceptEvolutionEntry(
                    deliberation="Concept pivot initiated by live Executive Producer steering feedback.",
                    turn_index=len(game_bible.writers_room_transcript),
                    author_role="ExecutiveProducer",
                    previous_premise=game_bible.elevator_pitch or game_bible.game_title,
                    evolved_premise=f"Incorporating Executive Producer steering: {fb[:160]}",
                    pivot_reason=fb,
                )
                game_bible.concept_evolution.append(evo)
                pivoted_texts.add(fb)

        self._persist_bible(game_bible, candidate_dir=candidate_dir)
        return consumed

    def _consume_pending_playtest_wishes(
        self,
        candidate_dir: Optional[Path] = None,
    ) -> Tuple[str, List[Any]]:
        """Consumes approved Blind Playtester wish items from `<candidate_dir>/.pending_playtest_wishes.json`
        AND any auto-delivered full Council playtest reports + wishes from
        `<candidate_dir>/.pending_council_playtest_reports.json`, returning
        `(formatted_provenance_block, multimodal_screenshot_parts)` so Council specialists inspect
        both the `observed_on_slice_id` provenance against current HEAD and the player's actual screenshots.
        """
        self._last_consumed_playtest_session_ids: Set[str] = set()
        if candidate_dir is None:
            return "", []
        wf = Path(candidate_dir) / ".pending_playtest_wishes.json"
        rf = Path(candidate_dir) / ".pending_council_playtest_reports.json"
        raw_items: List[Any] = []
        raw_reports: List[Any] = []
        if wf.is_file():
            try:
                loaded_w = json.loads(wf.read_text(encoding="utf-8"))
                wf.unlink(missing_ok=True)
                if isinstance(loaded_w, list):
                    raw_items = loaded_w
            except Exception as exc:
                logger.debug("Failed reading .pending_playtest_wishes.json: %s", exc)
        if rf.is_file():
            try:
                loaded_r = json.loads(rf.read_text(encoding="utf-8"))
                rf.unlink(missing_ok=True)
                if isinstance(loaded_r, list):
                    raw_reports = loaded_r
            except Exception as exc:
                logger.debug("Failed reading .pending_council_playtest_reports.json: %s", exc)

        if not raw_items and not raw_reports:
            return "", []

        for env_s in raw_reports:
            if isinstance(env_s, dict) and env_s.get("session_id"):
                self._last_consumed_playtest_session_ids.add(str(env_s["session_id"]).strip())

        from google.genai import types

        blocks: List[str] = []
        parts: List[Any] = []
        seen_screens: set[str] = set()

        def _attach_screen(label: str, raw_scr: Any) -> None:
            scr_str = str(raw_scr or "").strip()
            if not scr_str or scr_str in seen_screens or len(seen_screens) >= 4:
                return
            scr_p = Path(scr_str)
            if not scr_p.is_file():
                return
            try:
                img_bytes = scr_p.read_bytes()
                if img_bytes:
                    seen_screens.add(scr_str)
                    parts.append(types.Part.from_text(text=label))
                    parts.append(
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type="image/png",
                            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                        )
                    )
            except Exception as img_exc:
                logger.debug("Skipping unreadable playtest screenshot %s: %s", scr_p, img_exc)

        for r_idx, env in enumerate(raw_reports, 1):
            if not isinstance(env, dict):
                continue
            sid = str(env.get("session_id") or f"report_{r_idx}")
            obs_slice = str(env.get("observed_on_slice_id") or "unspecified_slice")
            cp_id = str(env.get("checkpoint_id") or "unknown_checkpoint")
            rep = env.get("session_report") or {}
            env_wishes = env.get("wishes") or []
            env_bugs = env.get("bugs_found") or {}
            env_questions = env.get("questions") or {}
            env_notes = env.get("field_notes") or {}
            env_rulebook = env.get("rulebook") or {}
            is_exec_dispatch = (
                str(env.get("dispatched_by") or "") == "executive_producer"
                or str(env.get("initiated_by") or "") == "operator"
            )
            header_title = (
                f"=== EXECUTIVE PRODUCER DISPATCHED BLIND PLAYTEST FULL REPORT & WISHES (`{sid}` | Slice `{obs_slice}` | Checkpoint `{cp_id}`) ==="
                if is_exec_dispatch
                else f"=== COUNCIL-INITIATED BLIND PLAYTEST FULL REPORT & WISHES (`{sid}` | Slice `{obs_slice}` | Checkpoint `{cp_id}`) ==="
            )

            r_lines: List[str] = [header_title]
            p_note_env = str(env.get("producer_note") or "").strip()
            if p_note_env:
                r_lines.append(f"Executive Producer Direction (HONOR AT FULL SCOPE): {p_note_env}")
            r_lines.extend([
                f"Overall Fun Rating: {rep.get('overall_fun_rating', '?')}/10 ({env.get('turn_count', '?')} turns, {env.get('sim_duration_sec', '?')}s)",
                f"Overall Experience Summary: {rep.get('overall_experience_summary', '')}",
                f"- Graphics & Visuals: {rep.get('graphics_and_visuals', '')}",
                f"- Gameplay & Feel: {rep.get('gameplay_and_feel', '')}",
                f"- Sound & Audio: {rep.get('sound_and_audio', '')}",
                f"- Intuitiveness & Onboarding: {rep.get('intuitiveness_and_onboarding', '')}",
                f"- Goals, Pacing & Accomplishment: {rep.get('goals_pacing_and_accomplishment', '')}",
                f"- World & Atmosphere: {rep.get('world_and_atmosphere', '')}",
                f"- Visual & UI Defects Observed: {json.dumps(rep.get('visual_and_ui_defects_observed') or [])}",
                f"- Key Highlights: {json.dumps(rep.get('key_highlights') or [])}",
                f"- Key Pain Points: {json.dumps(rep.get('key_pain_points') or [])}",
                f"- Bugs & Glitches Observed: {json.dumps(rep.get('bugs_and_glitches_observed') or [])}",
                f"- Unanswered Questions & Mysteries: {json.dumps(rep.get('unanswered_questions_and_mysteries') or [])}",
            ])
            if isinstance(env_bugs, dict) and env_bugs:
                r_lines.append("Spotted Bugs & Glitches During Play:")
                for bk, bv in env_bugs.items():
                    if isinstance(bv, dict):
                        r_lines.append(
                            f"  * [{bv.get('severity', 'major').upper()} | {bv.get('category', 'bug')}] "
                            f"({bk}) {bv.get('summary', '')}: {bv.get('details', '')}"
                        )
                        _attach_screen(
                            f"Blind Playtester Bug Screenshot ('{bv.get('summary', bk)}', slice '{obs_slice}'):",
                            bv.get("screenshot_path"),
                        )
            if isinstance(env_notes, dict) and env_notes:
                r_lines.append("Player Field Notes:")
                for nk, nv in env_notes.items():
                    r_lines.append(f"  * {nk}: {nv}")
            if isinstance(env_questions, dict) and env_questions:
                r_lines.append("Player Questions (Asked & Answered Along The Way):")
                for qk, qv in env_questions.items():
                    if isinstance(qv, dict):
                        st = str(qv.get("status") or "open").upper()
                        ans = f" -> {qv['answer']}" if qv.get("answer") else ""
                        r_lines.append(f"  * [{st}] ({qk}) {qv.get('question', '')}{ans}")
            if isinstance(env_rulebook, dict) and env_rulebook:
                r_lines.append("Confirmed Controls/Rules Figured Out:")
                for rk, rv in env_rulebook.items():
                    r_lines.append(f"  * {rk}: {rv}")
            if isinstance(env_wishes, list) and env_wishes:
                r_lines.append(f"Auto-Delivered Player Wishes ({len(env_wishes)}):")
                for w_i, w_item in enumerate(env_wishes, 1):
                    if isinstance(w_item, dict):
                        r_lines.append(
                            f"  {w_i}. [{w_item.get('category', 'gameplay_feel')}] {w_item.get('summary', '')} "
                            f"(mood={w_item.get('mood_at_time', 'n/a')}): {w_item.get('rationale', '')}"
                        )
                        _attach_screen(
                            f"Blind Playtester Wish Screenshot ('{w_item.get('summary', '')}', slice '{obs_slice}'):",
                            w_item.get("screenshot_path"),
                        )
            for rep_scr in env.get("representative_screenshots") or []:
                if isinstance(rep_scr, dict) and rep_scr.get("path"):
                    _attach_screen(
                        f"Blind Playtester Session Frame — {rep_scr.get('label', sid)} ('{sid}', slice '{obs_slice}'):",
                        rep_scr.get("path"),
                    )
            _attach_screen(
                f"Blind Playtester Final Session Screen ('{sid}', slice '{obs_slice}'):",
                env.get("final_screenshot_path"),
            )
            blocks.append("\n".join(r_lines))

        if raw_items:
            lines: List[str] = [
                "=== APPROVED BLIND PLAYTESTER WISHES (OPAQUE-BOX PLAYER FEEDBACK) ===",
                "PROVENANCE & HEAD RECONCILIATION RULE: Each wish below was filed by an autonomous Blind Playtester "
                "(who knew ONLY the game's title and short store description) while playing a specific completed slice checkpoint.",
                "Before adding tasks, check current HEAD code and recent ProductionBoard tasks to verify whether a later slice "
                "already resolved the symptom; if not, update your domain's Game Bible strategy and raise/refine concrete "
                "`backlog_additions` or `backlog_refinements` immediately:",
            ]
            for idx, item in enumerate(raw_items, 1):
                if not isinstance(item, dict):
                    continue
                cat = str(item.get("category") or "gameplay_feel")
                summary = str(item.get("summary") or "").strip()
                rationale = str(item.get("rationale") or "").strip()
                obs_slice = str(item.get("observed_on_slice_id") or "unspecified_slice")
                cp_id = str(item.get("checkpoint_id") or "unknown_checkpoint")
                mood = str(item.get("mood_at_time") or "unspecified")
                p_note = str(item.get("producer_note") or "").strip()
                lines.append(
                    f"  {idx}. [{cat}] {summary}\n"
                    f"     - Observed on Slice: `{obs_slice}` (Checkpoint `{cp_id}`, Player Mood: `{mood}`)\n"
                    f"     - Player Experience Rationale: {rationale}"
                    + (f"\n     - Executive Producer Direction: {p_note}" if p_note else "")
                )
                _attach_screen(
                    f"Blind Playtester Screenshot for Wish #{idx} ('{summary}', observed on slice '{obs_slice}'):",
                    item.get("screenshot_path"),
                )
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks), parts

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Emits a rationale-first telemetry event to Mission Control and TraceLogger."""
        if getattr(self, "_abort_requested", False):
            raise StudioPausedError(f"Writers' Room build aborted before emitting '{event_type}'.")
        ordered: Dict[str, Any] = {
            "deliberation": payload.get(
                "deliberation",
                f"CreativeWritersRoom event '{event_type}' emitted for live Mission Control observability.",
            )
        }
        for k, v in payload.items():
            if k != "deliberation":
                ordered[k] = v
        if self.event_callback is not None:
            try:
                self.event_callback(event_type, ordered)
            except Exception as exc:
                logger.warning("Event callback failed for '%s': %s", event_type, exc)

    def _emit_activity(
        self,
        agent_role: str,
        status_text: str,
        active_agents: Optional[List[str]] = None,
        phase: str = "brainstorming",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emits a live real-time observability event (`writers_room_activity`) so Mission Control
        displays exactly which Council Lead(s) or Sub-Agent(s) are running right now and what they are doing."""
        payload: Dict[str, Any] = {
            "deliberation": f"Live agent activity update ({phase}): {status_text}",
            "agent_role": agent_role,
            "status_text": status_text,
            "active_agents": list(active_agents) if active_agents is not None else [agent_role],
            "phase": phase,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        self._emit_event("writers_room_activity", payload)

    def _persist_bible(self, game_bible: GameBibleSpec, candidate_dir: Optional[Path] = None) -> None:
        """Saves game_bible.json (and world_spec.json / narrative_bible.json if populated) to disk and upserts to SQLite studio_game_bibles."""
        self._check_aborted(candidate_dir)
        game_bible.updated_at_utc = datetime.now(timezone.utc).isoformat()
        if candidate_dir is not None:
            candidate_dir = Path(candidate_dir)
            candidate_dir.mkdir(parents=True, exist_ok=True)
            bible_path = candidate_dir / "game_bible.json"
            # Preserve any ExecutiveProducer / UserSteering utterances written to disk concurrently by Mission Control
            if bible_path.is_file():
                try:
                    disk_data = json.loads(bible_path.read_text(encoding="utf-8"))
                    disk_tx = disk_data.get("writers_room_transcript") if isinstance(disk_data, dict) else []
                    if isinstance(disk_tx, list):
                        mem_exec_texts = {
                            u.spoken_dialogue.strip()
                            for u in game_bible.writers_room_transcript
                            if u.speaker_role in ("ExecutiveProducer", "UserSteering") and u.spoken_dialogue
                        }
                        for raw_u in disk_tx:
                            if isinstance(raw_u, dict) and raw_u.get("speaker_role") in ("ExecutiveProducer", "UserSteering"):
                                txt = str(raw_u.get("spoken_dialogue") or raw_u.get("message") or "").strip()
                                if txt and txt not in mem_exec_texts:
                                    self.enqueue_user_steering(txt, candidate_dir=candidate_dir)
                                    game_bible.writers_room_transcript.append(
                                        WritersRoomUtterance(
                                            deliberation=str(raw_u.get("deliberation") or "Executive Producer steering."),
                                            utterance_id=str(raw_u.get("utterance_id") or f"utt_exec_{uuid.uuid4().hex[:8]}"),
                                            turn_index=len(game_bible.writers_room_transcript),
                                            speaker_role="ExecutiveProducer",
                                            speaker_display_name=str(raw_u.get("speaker_display_name") or "Executive Producer (You)"),
                                            spoken_dialogue=txt,
                                            addressed_to=["CreativeDirector", "GameplayDesigner", "GameplaySystemsEngineer", "UXExperienceDesigner"],
                                            bible_updates_summary=["Executive Producer creative direction injected"],
                                        )
                                    )
                                    mem_exec_texts.add(txt)
                except Exception as exc:
                    logger.debug("Concurrent steering check on _persist_bible warning: %s", exc)
            bible_path.write_text(game_bible.model_dump_json(indent=2), encoding="utf-8")
            if game_bible.world_spec is not None:
                (candidate_dir / "world_spec.json").write_text(
                    game_bible.world_spec.model_dump_json(indent=2), encoding="utf-8"
                )
            if game_bible.narrative_bible is not None:
                (candidate_dir / "narrative_bible.json").write_text(
                    game_bible.narrative_bible.model_dump_json(indent=2), encoding="utf-8"
                )

        if self.telemetry_db is not None and hasattr(self.telemetry_db, "upsert_game_bible"):
            try:
                self.telemetry_db.upsert_game_bible(game_bible)
            except Exception as exc:
                logger.warning("Failed to upsert GameBibleSpec to telemetry_db: %s", exc)

        full_dump = game_bible.model_dump()
        event_payload: Dict[str, Any] = {
            "deliberation": (
                f"Persisted Living Game Bible '{game_bible.game_title}' "
                f"(status={game_bible.status}, turns={len(game_bible.writers_room_transcript)})."
            ),
            **{k: v for k, v in full_dump.items() if k != "deliberation"},
            "turn_count": len(game_bible.writers_room_transcript),
            "evolution_count": len(game_bible.concept_evolution),
            "storyboard_count": len(game_bible.experience_storyboard),
            "mockup_count": len(game_bible.visual_mockups),
        }
        self._emit_event("game_bible_updated", event_payload)

    def _call_structured_llm(
        self,
        prompt: str,
        schema_cls: Any,
        temperature: Optional[float] = None,
        parts: Optional[List[Any]] = None,
        *,
        cacheable_prefix: str = "",
        dynamic_tail: str = "",
        min_calls_before_create: int = 2,
    ) -> Any:
        """Invokes Gemini with structured JSON schema and high creative temperature, returning validated Pydantic instance.
        `parts` (e.g. proof images) are sent before the prompt text."""
        if self.client is None:
            raise StudioAgentError(
                "CreativeWritersRoom requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)"
            )
        self._check_aborted()
        from google.genai import types

        effective_temp = self.creative_temperature if temperature is None else float(temperature)
        schema_name = getattr(schema_cls, "__name__", str(schema_cls))
        thinking_budget = (
            384
            if schema_name == "WritersRoomListenCheck"
            else (1024 if schema_name == "NoveltyJudgement" else 2048)
        )
        prompt_body = (
            prompt[len(cacheable_prefix):]
            if (cacheable_prefix and prompt.startswith(cacheable_prefix))
            else prompt
        )
        tail_prompt = f"{prompt_body}{dynamic_tail}" if dynamic_tail else prompt_body
        combined_prompt = f"{cacheable_prefix}{tail_prompt}" if cacheable_prefix else tail_prompt
        unsplit_contents: Any = (list(parts) + [combined_prompt]) if parts else combined_prompt
        if cacheable_prefix:
            prefix_parts: List[Any] = (list(parts) + [cacheable_prefix]) if parts else [cacheable_prefix]
            tail_parts: List[Any] = [tail_prompt]
        else:
            prefix_parts = (list(parts) + [prompt_body]) if parts else [prompt_body]
            tail_parts = [dynamic_tail] if dynamic_tail else []
        eff_contents, cached_name = resolve_explicit_cached_request(
            self.client,
            model=self.model_name,
            prefix_parts=prefix_parts,
            tail_parts=tail_parts,
            unsplit_contents=unsplit_contents,
            min_calls_before_create=min_calls_before_create,
            display_name=f"council_{schema_name}",
        )
        gen_cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_cls,
            temperature=effective_temp,
            thinking_config=bounded_thinking_config(thinking_budget),
            **({"cached_content": cached_name} if cached_name else {}),
        )
        response = call_gemini_with_transient_retry(
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=eff_contents,
                config=gen_cfg,
            ),
            label=f"CreativeWritersRoom[{schema_name}]",
        )
        self._check_aborted()
        if not response.text:
            raise StudioAgentError(f"Empty response from Gemini for schema {schema_name}")
        raw_data = json.loads(response.text)
        target_cls = WritersRoomTurnProposal if schema_name == "WritersRoomTurnProposal" else schema_cls
        if isinstance(raw_data, target_cls):
            return raw_data
        return target_cls(**raw_data)

    def _format_dna_summary(self, dna: Optional[Any], hypothesis_report: Optional[Any] = None) -> str:
        lines: List[str] = []
        if dna is not None:
            def _dget(key: str, default: Any = None) -> Any:
                if isinstance(dna, dict):
                    return dna.get(key, default)
                return getattr(dna, key, default)

            raw_games = _dget("top_games") or _dget("games") or []
            top_games = [
                g for g in raw_games
                if isinstance(g, dict) and not str(g.get("name", "")).startswith("App ")
            ]
            sorted_by_hours = sorted(
                top_games,
                key=lambda g: float(g.get("playtime_hours", 0.0) or 0.0),
                reverse=True,
            )
            core_anchors = sorted_by_hours[:8]
            core_str = ", ".join(
                f"{g.get('name', 'Unknown')} ({float(g.get('playtime_hours', 0.0) or 0.0):.1f}h, genres: {g.get('genres', [])})"
                for g in core_anchors
            )
            # Also sample a varied spotlight subset for fresh creative cross-pollination
            if len(top_games) > 4:
                spotlight_games = random.sample(top_games[:12], k=min(4, len(top_games[:12])))
            else:
                spotlight_games = list(top_games)
            spotlight_str = ", ".join(
                f"{g.get('name', 'Unknown')} ({float(g.get('playtime_hours', 0.0) or 0.0):.1f}h, genres: {g.get('genres', [])})"
                for g in spotlight_games
            )
            top_genres = list(_dget("top_genres") or [])
            genre_weights = _dget("genre_weights") or {}
            playtime_dist = _dget("playtime_distribution") or {}
            total_hours = float(_dget("total_playtime_hours", 0.0) or 0.0)
            total_games = int(_dget("total_games_analyzed") or _dget("total_games") or 0)
            pacing_pref = _dget("pacing_preference", "balanced")
            pacing_score = float(_dget("pacing_score", 0.0) or 0.0)
            challenge_tol = float(_dget("challenge_tolerance", 0.5) or 0.5)
            dna_delib = str(_dget("deliberation") or "").strip()
            mech_biases = _dget("mechanical_biases") or {}
            thematic_biases = list(_dget("thematic_biases") or [])
            if dna_delib:
                lines.append(f"- Epistemic Player DNA Synthesis: {dna_delib}")
            lines.extend([
                f"- Target Player Steam Library Scale: {total_hours:.1f} total hours across {total_games} analyzed games",
                f"- Top Genres: {top_genres} (weights: {genre_weights})",
                f"- Engagement & Mastery Distribution: {playtime_dist}",
                f"- Pacing & Complexity Appetite: {pacing_pref} (pacing_score={pacing_score:.2f}, challenge_tolerance={challenge_tol:.2f})",
            ])
            if mech_biases:
                lines.append(f"- Core Mechanical Verbs & Affinities: {mech_biases}")
            if thematic_biases:
                lines.append(f"- Thematic & Setting Affinities: {thematic_biases}")
            palette_pref = _dget("preferred_palette_mood", None)
            if palette_pref:
                lines.append(f"- Preferred Palette Mood: {palette_pref}")
            if core_str:
                lines.append(
                    f"- Most-Played Core Anchor Games (what this specific player invests hundreds of hours mastering): {core_str}"
                )
            activity_windows = _dget("activity_windows") or {}
            if isinstance(activity_windows, dict) and activity_windows:
                for w_key, w_title in (
                    ("last_30_days", "Recent Gameplay (Last 30 Days)"),
                    ("last_90_days", "Medium-Term Gameplay (Last 90 Days)"),
                    ("last_1_year", "Past-Year Gameplay (Last 1 Year)"),
                ):
                    w_data = activity_windows.get(w_key)
                    if not isinstance(w_data, dict):
                        continue
                    w_games = [
                        g
                        for g in (w_data.get("games") or [])
                        if isinstance(g, dict) and not str(g.get("name", "")).startswith("App ")
                    ]
                    w_count = int(w_data.get("games_count", len(w_games)) or 0)
                    if w_count <= 0 or not w_games:
                        continue
                    w_str = ", ".join(
                        f"{g.get('name', 'Unknown')} ({float(g.get('playtime_hours', 0.0) or 0.0):.1f}h total"
                        + (
                            f", {float(g.get('playtime_2weeks_hours', 0.0) or 0.0):.1f}h in 2wk"
                            if g.get("playtime_2weeks_hours")
                            else ""
                        )
                        + (
                            f", last played {g.get('last_played_date')}"
                            if g.get("last_played_date")
                            else ""
                        )
                        + f", genres: {g.get('genres', [])})"
                        for g in w_games[:6]
                    )
                    lines.append(f"- {w_title} ({w_count} active titles): {w_str}")
            if spotlight_str:
                lines.append(
                    f"- Session Spotlight Cross-Pollination Subset (taste signals — NOT templates to copy or pastiche): {spotlight_str}"
                )
            lines.append(
                "- TARGET PLAYER OPTIMIZATION MANDATE: You are designing specifically for THIS human player based on their Steam library playtime "
                "and behavioral DNA above—not a generic anonymous casual player. Reason from what this player actually loves to DO in games "
                "(their mastery depth, systemic/construction/optimization appetite vs. kinetic/tactical expression, and session depth) combined with "
                "the active concept/genre. When the player's DNA or genre centers on building, routing, automation, engineering, or systemic mastery, "
                "the player MUST be the one who builds, places, connects, experiments, and optimizes their own layout—there is no single pre-scripted "
                "'correct' solution, and you must NEVER hand them a pre-built diorama or a trivial 1-step lock-and-key switch!"
            )
        if hypothesis_report is not None:
            hyps = getattr(hypothesis_report, "hypotheses", []) or []
            if hyps:
                lines.append(f"- Empirical Player Telemetry Hypotheses: {[getattr(h, 'statement', str(h)) for h in hyps]}")
        return "\n".join(lines) if lines else "- Open creative mandate: craft a deeply polished, unforgettable game experience."

    @staticmethod
    def _format_design_sections(game_bible: GameBibleSpec) -> str:
        """Compact view of the game-design sections so every specialist can build on / challenge them."""
        parts: List[str] = []
        lg = game_bible.level_generation_strategy
        parts.append(
            "level_generation_strategy=" + (
                f"[{lg.generation_approach}; {lg.world_structure[:180]}; runtime={getattr(lg, 'runtime_construction_plan', '')[:120]}; "
                f"archetypes={[a.archetype_id for a in lg.level_archetypes]}; rules={len(lg.generation_rules)}]" if lg else "(missing)"
            )
        )
        loop = game_bible.core_loop
        parts.append(
            "core_loop=" + (
                f"[verbs={loop.player_verbs}; agency={getattr(loop, 'player_agency_and_optimization_depth', '')[:140]}; "
                f"m2m={loop.moment_to_moment_loop[:140]}; session={loop.session_loop[:140]}]"
                if loop else "(missing)"
            )
        )
        eng = game_bible.engagement_plan
        parts.append("engagement_plan=" + (f"[pillars={eng.entertainment_pillars}]" if eng else "(missing)"))
        onb = game_bible.onboarding_plan
        parts.append(
            "onboarding_plan=" + (f"[world_intro={onb.world_introduction[:160]}; steps={[st.concept_introduced for st in onb.steps]}]" if onb else "(missing)")
        )
        gh = game_bible.goal_hierarchy
        parts.append(
            "goal_hierarchy=" + (f"[{[(g.horizon, g.description[:80]) for g in gh.goals]}; end_state={gh.end_state[:120]}]" if gh else "(missing)")
        )
        return " | ".join(parts)

    @staticmethod
    def _dna_header_if_absent(dna_summary: str, bible_snapshot: str) -> str:
        """Returns the `PLAYER DNA & STUDIO CONTEXT` header only when `dna_summary` is not already
        embedded inside `bible_snapshot` (`TARGET PLAYER STEAM DNA PROFILE`), avoiding ~1.7k-token duplication."""
        clean_dna = (dna_summary or "").strip()
        if not clean_dna or clean_dna in (bible_snapshot or ""):
            return ""
        return f"PLAYER DNA & STUDIO CONTEXT:\n{dna_summary}\n\n"

    @staticmethod
    def _compact_playtest_dispatch_text(raw_text: str, *, note_suffix: str) -> str:
        """Preserves the deterministic header + any human `Executive Producer Direction:` line on a
        formatted Blind Playtest report dispatch while omitting the multi-kilobyte auto-serialized
        report body when that report is already attached in full below or superseded by a newer slice."""
        txt = (raw_text or "").strip()
        if not txt.startswith("[EXECUTIVE PRODUCER MANUAL BLIND PLAYTEST FULL REPORT DISPATCH"):
            return txt
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        kept: List[str] = []
        if lines:
            kept.append(lines[0])
        for ln in lines[1:]:
            if ln.lstrip("- ").startswith("Executive Producer Direction"):
                kept.append(ln.lstrip("- "))
                break
        kept.append(f"({note_suffix})")
        return "\n    ".join(kept)

    def _format_bible_snapshot(
        self,
        game_bible: GameBibleSpec,
        *,
        compact_for_review: bool = False,
        omit_playtest_session_ids: Optional[Set[str]] = None,
    ) -> str:
        chars = game_bible.narrative_bible.characters if game_bible.narrative_bible else []
        char_names = [f"{c.name} [character_id={c.character_id}, role={c.role}]" for c in chars]
        beats = [f"Beat {b.beat_index}: {b.phase_title} ({b.player_emotional_state})" for b in game_bible.experience_storyboard]
        mockups = [f"{m.title} ({m.image_path})" for m in game_bible.visual_mockups]
        all_eng_specs = list(getattr(game_bible, "technical_architecture_blueprint", []) or [])
        shown_eng_specs = all_eng_specs[-10:] if compact_for_review else all_eng_specs
        eng_specs = [
            f"{e.mechanic_id} ({e.title}) -> nodes={e.godot_node_and_collider_architecture[:80]}, assets={e.required_sprite_and_tile_assets}"
            for e in shown_eng_specs
        ]
        all_slices = list(getattr(game_bible, "slice_roadmap", []) or [])
        shown_slices = all_slices[-4:] if compact_for_review else all_slices
        slice_summaries = [
            f"[{s.slice_id} idx={s.slice_index} status={s.status}] '{s.title}': objective={s.slice_objective[:120]} | focus={s.player_acclimation_and_mechanics_focus[:120]}"
            for s in shown_slices
        ]
        active_slice = game_bible.get_active_slice() if hasattr(game_bible, "get_active_slice") else None
        active_slice_str = (
            f"[{active_slice.slice_id} (Iteration #{active_slice.slice_index})] '{active_slice.title}' — "
            f"Objective: {active_slice.slice_objective} | Acclimation & Full-Loop Focus: {active_slice.player_acclimation_and_mechanics_focus} | "
            f"Expansion: {active_slice.expansion_from_prior_slice} | Required Systems & Scope: {active_slice.scope_boundaries}"
            if active_slice is not None
            else "(not yet strategized — Slice 0 MUST deliver the complete playable game loop realizing all 3 timescales of `core_loop`, `player_agency_and_optimization_depth`, multi-horizon `goal_hierarchy`, multi-region world progression, UX onboarding, and audio—never defer the core game loop to future slices!)"
        )
        tx_all = list(game_bible.writers_room_transcript or [])
        last_strategy_turn_idx = -1
        latest_pt_report_turn_idx = -1
        for u_scan in tx_all:
            u_upds = list(getattr(u_scan, "bible_updates_summary", None) or [])
            if any(str(s).startswith("next_slice_strategy:") for s in u_upds):
                last_strategy_turn_idx = max(last_strategy_turn_idx, int(u_scan.turn_index))
            u_art = getattr(u_scan, "artifact_emitted", None) or {}
            if (isinstance(u_art, dict) and u_art.get("type") == "playtest_report") or str(
                getattr(u_scan, "utterance_id", "") or ""
            ).startswith("utt_ptrep_"):
                latest_pt_report_turn_idx = max(latest_pt_report_turn_idx, int(u_scan.turn_index))

        omit_pt_sids = set(omit_playtest_session_ids or ())
        exec_entries: List[Tuple[str, str]] = []
        seen_directives: Set[str] = set()
        active_seed = (getattr(self, "_active_creative_prompt", None) or "").strip()
        if not active_seed and game_bible.concept_evolution:
            first_prev = (game_bible.concept_evolution[0].previous_premise or "").strip()
            if first_prev and first_prev != "Untitled Concept":
                active_seed = first_prev
        for u in tx_all:
            if u.speaker_role in ("ExecutiveProducer", "UserSteering") and u.spoken_dialogue and u.spoken_dialogue.strip():
                raw_d = u.spoken_dialogue.strip()
                if raw_d in seen_directives:
                    continue
                seen_directives.add(raw_d)
                u_art = getattr(u, "artifact_emitted", None) or {}
                is_pt_rep = (
                    (isinstance(u_art, dict) and u_art.get("type") == "playtest_report")
                    or str(getattr(u, "utterance_id", "") or "").startswith("utt_ptrep_")
                    or raw_d.startswith("[EXECUTIVE PRODUCER MANUAL BLIND PLAYTEST FULL REPORT DISPATCH")
                )
                pt_sid = str(u_art.get("session_id") or "") if isinstance(u_art, dict) else ""
                if is_pt_rep and (
                    (pt_sid and pt_sid in omit_pt_sids)
                    or (pt_sid == "" and any(f"`{sid}`" in raw_d for sid in omit_pt_sids))
                ):
                    d_txt = self._compact_playtest_dispatch_text(
                        raw_d,
                        note_suffix="full structured playtest report & screenshots attached in dossier below",
                    )
                elif is_pt_rep and (
                    int(u.turn_index) < last_strategy_turn_idx
                    or int(u.turn_index) < latest_pt_report_turn_idx
                ):
                    d_txt = self._compact_playtest_dispatch_text(
                        raw_d,
                        note_suffix="prior-slice playtest report archived after slice progression",
                    )
                else:
                    d_txt = raw_d
                exec_entries.append((f"Turn #{u.turn_index}", d_txt))
        if active_seed and active_seed not in seen_directives:
            seen_directives.add(active_seed)
            exec_entries.insert(0, ("Initial Seed", active_seed))

        recent_turns = tx_all[(-3 if compact_for_review else -6):]
        recent_lines: List[str] = []
        for u in recent_turns:
            raw_sp = u.spoken_dialogue or ""
            if (
                u.speaker_role in ("ExecutiveProducer", "UserSteering")
                and len(raw_sp) > 600
                and raw_sp.strip() in seen_directives
            ):
                sp_display = "(See MANDATORY EXECUTIVE PRODUCER CREATIVE STEERING banner above)"
            elif compact_for_review and len(raw_sp) > 480:
                sp_display = raw_sp[:480] + "..."
            else:
                sp_display = raw_sp
            recent_lines.append(
                f"  [Turn {u.turn_index}] {u.speaker_display_name} ({u.speaker_role}): {sp_display}"
            )
        transcript_text = "\n".join(recent_lines) or "  (Room just opened)"
        formatted_exec_lines: List[str] = []
        for idx_d, (turn_lbl, d_txt) in enumerate(exec_entries):
            if len(exec_entries) > 1 and idx_d == len(exec_entries) - 1:
                formatted_exec_lines.append(f'  - [LATEST / CURRENT DIRECTIVE — {turn_lbl}]: "{d_txt}"')
            elif len(exec_entries) > 1:
                formatted_exec_lines.append(f'  - [PRIOR FOUNDATIONAL DIRECTIVE — {turn_lbl}]: "{d_txt}"')
            else:
                formatted_exec_lines.append(f'  - "{d_txt}"')
        exec_banner = (
            "🚨 MANDATORY EXECUTIVE PRODUCER CREATIVE STEERING (HIGHEST PRIORITY — ALWAYS RESPECT & CATER TO THIS DIRECTION):\n"
            + "\n".join(formatted_exec_lines)
            + "\n  Executive Producer messages are ALWAYS treated with full respect—never push back against, lecture, make excuses to, or silently downgrade what the Executive Producer asked for (Rule 5). Every specialist's job is to cater directly to this Executive Producer direction across core loop, pacing, visual/UX craft, controls, and Production Board items (treating `[LATEST / CURRENT DIRECTIVE]` as what the Executive Producer just asked for right now, and `[PRIOR FOUNDATIONAL DIRECTIVE]` as earlier context already established in prior slices), while staying sharply critical of our own build and peer assumptions!\n\n"
            if formatted_exec_lines
            else ""
        )
        dna_block = (
            f"TARGET PLAYER STEAM DNA PROFILE (the specific human player this game is being optimized for):\n"
            f"{game_bible.target_player_dna_summary}\n\n"
            if getattr(game_bible, "target_player_dna_summary", "")
            else ""
        )
        return (
            f"{exec_banner}"
            f"{dna_block}"
            f"CURRENT LIVING GAME BIBLE STATE:\n"
            f"- Working Title: {game_bible.game_title}\n"
            f"- Core Genre: {game_bible.core_genre}\n"
            f"- Elevator Pitch: {game_bible.elevator_pitch or '(not yet defined)'}\n"
            f"- Concept Evolution So Far ({len(game_bible.concept_evolution)} milestones): "
            f"{[e.evolved_premise[:160] for e in game_bible.concept_evolution[-4:]]}\n"
            f"- Core Fantasy: {game_bible.core_fantasy or '(not yet defined)'}\n"
            f"- First 60 Seconds Hook: {game_bible.first_60_seconds_hook or '(not yet defined)'}\n"
            f"- Diegetic UI & Onboarding Philosophy: {game_bible.diegetic_ui_philosophy or '(not yet defined)'}\n"
            f"- Gameplay Pillars ({len(game_bible.gameplay_pillars)}): {game_bible.gameplay_pillars}\n"
            f"- Iterative Slice Roadmap ({len(slice_summaries)} slices): {slice_summaries}\n"
            f"- Active Slice right now: {active_slice_str}\n"
            f"- Technical Architecture Blueprint ({len(eng_specs)}): {eng_specs}\n"
            f"- Visual Style Guide: {(game_bible.visual_style_guide or '(not yet defined)')[:1200]}\n"
            f"- Audio Direction Guide: {(game_bible.audio_direction_guide or '(not yet defined)')[:1200]}\n"
            f"- Experience Storyboard Beats ({len(beats)}): {beats}\n"
            f"- Game Design Sections: {self._format_design_sections(game_bible)}\n"
            f"- Characters ({len(char_names)}): {char_names}\n"
            f"- Synthesized 16:9 Visual Mockups ({len(mockups)}): {mockups}\n"
            f"\nRECENT WRITERS' ROOM CONVERSATION:\n{transcript_text}\n"
        )

    def _apply_turn_proposal(
        self,
        game_bible: GameBibleSpec,
        speaker_role: str,
        speaker_display_name: str,
        proposal: WritersRoomTurnProposal,
        candidate_dir: Optional[Path] = None,
    ) -> WritersRoomUtterance:
        """Applies a specialist's structured turn proposal to the Living Game Bible and synthesizes any requested multimodal artifacts."""
        turn_index = len(game_bible.writers_room_transcript)
        updates_summary: List[str] = []

        prev_premise = game_bible.elevator_pitch or game_bible.core_fantasy or game_bible.game_title

        if proposal.updated_game_title and proposal.updated_game_title.strip():
            game_bible.game_title = proposal.updated_game_title.strip()
            updates_summary.append(f"Updated title to '{game_bible.game_title}'")

        if proposal.updated_elevator_pitch and proposal.updated_elevator_pitch.strip():
            game_bible.elevator_pitch = proposal.updated_elevator_pitch.strip()
            updates_summary.append("Refined elevator pitch")

        if proposal.updated_core_fantasy and proposal.updated_core_fantasy.strip():
            game_bible.core_fantasy = proposal.updated_core_fantasy.strip()
            updates_summary.append("Deepened core player fantasy")

        if proposal.updated_first_60_seconds_hook and proposal.updated_first_60_seconds_hook.strip():
            game_bible.first_60_seconds_hook = proposal.updated_first_60_seconds_hook.strip()
            updates_summary.append("Defined First 60 Seconds sensory hook")

        if proposal.updated_diegetic_ui_philosophy and proposal.updated_diegetic_ui_philosophy.strip():
            game_bible.diegetic_ui_philosophy = proposal.updated_diegetic_ui_philosophy.strip()
            updates_summary.append("Updated Diegetic UI & Onboarding philosophy")

        if proposal.visual_style_guide_update and proposal.visual_style_guide_update.strip():
            game_bible.visual_style_guide = proposal.visual_style_guide_update.strip()
            updates_summary.append("Updated Visual Style Guide")

        if proposal.updated_camera_and_spatial_paradigm and proposal.updated_camera_and_spatial_paradigm.strip():
            game_bible.camera_and_spatial_paradigm = proposal.updated_camera_and_spatial_paradigm.strip()
            updates_summary.append(f"Locked spatial/camera paradigm: {game_bible.camera_and_spatial_paradigm[:60]}")

        updated_palette = getattr(proposal, "updated_color_palette_hex", None)
        if updated_palette:
            game_bible.color_palette_hex = [str(c).strip() for c in updated_palette if str(c).strip()]
            updates_summary.append(f"Updated color palette ({len(game_bible.color_palette_hex)} hex swatches)")

        updated_layers = getattr(proposal, "added_or_updated_scene_layers", None)
        if updated_layers:
            layers_by_id = {l.layer_id: l for l in game_bible.scene_layers}
            for layer in updated_layers:
                layers_by_id[layer.layer_id] = layer
            game_bible.scene_layers = sorted(layers_by_id.values(), key=lambda x: x.depth_order)
            updates_summary.append(f"Defined {len(game_bible.scene_layers)} visual scene depth layers")

        if proposal.audio_direction_update and proposal.audio_direction_update.strip():
            game_bible.audio_direction_guide = proposal.audio_direction_update.strip()
            updates_summary.append("Updated Audio & Voice Atmosphere Guide")

        for pillar in proposal.gameplay_pillars_additions or []:
            if pillar and pillar.strip() and pillar.strip() not in game_bible.gameplay_pillars:
                game_bible.gameplay_pillars.append(pillar.strip())
                updates_summary.append(f"Added gameplay pillar: {pillar.strip()[:60]}")

        if proposal.added_or_updated_engineering_specs:
            existing_eng = {e.mechanic_id: e for e in game_bible.technical_architecture_blueprint}
            for espec in proposal.added_or_updated_engineering_specs:
                existing_eng[espec.mechanic_id] = espec
            game_bible.technical_architecture_blueprint = list(existing_eng.values())
            updates_summary.append(
                f"Engineered {len(proposal.added_or_updated_engineering_specs)} technical architecture spec(s)"
            )
            self._emit_event(
                "bible_engineering_blueprint_updated",
                {
                    "deliberation": proposal.deliberation,
                    "speaker_role": speaker_role,
                    "engineering_specs_count": len(game_bible.technical_architecture_blueprint),
                    "mechanic_ids": [e.mechanic_id for e in game_bible.technical_architecture_blueprint],
                },
            )

        # Concept evolution milestone
        if proposal.concept_evolution_milestone is not None:
            evo = proposal.concept_evolution_milestone
            evo.turn_index = turn_index
            evo.author_role = speaker_role
            if not evo.previous_premise:
                evo.previous_premise = prev_premise
            game_bible.concept_evolution.append(evo)
            updates_summary.append(f"Concept Evolution: {evo.evolved_premise[:70]}")
            self._emit_event(
                "concept_evolution_logged",
                {
                    "deliberation": evo.deliberation,
                    "entry_id": evo.entry_id,
                    "turn_index": evo.turn_index,
                    "author_role": evo.author_role,
                    "previous_premise": evo.previous_premise,
                    "evolved_premise": evo.evolved_premise,
                    "pivot_reason": evo.pivot_reason,
                },
            )
        elif proposal.updated_elevator_pitch or proposal.updated_core_fantasy:
            evo = ConceptEvolutionEntry(
                deliberation=proposal.deliberation,
                turn_index=turn_index,
                author_role=speaker_role,
                previous_premise=prev_premise,
                evolved_premise=proposal.updated_elevator_pitch or proposal.updated_core_fantasy or game_bible.game_title,
                pivot_reason=proposal.spoken_dialogue[:160],
            )
            game_bible.concept_evolution.append(evo)
            self._emit_event(
                "concept_evolution_logged",
                {
                    "deliberation": evo.deliberation,
                    "entry_id": evo.entry_id,
                    "turn_index": evo.turn_index,
                    "author_role": evo.author_role,
                    "previous_premise": evo.previous_premise,
                    "evolved_premise": evo.evolved_premise,
                    "pivot_reason": evo.pivot_reason,
                },
            )

        # Experience Storyboard Beats
        if proposal.added_or_updated_storyboard_beats:
            by_idx = {b.beat_index: b for b in game_bible.experience_storyboard}
            for beat in proposal.added_or_updated_storyboard_beats:
                by_idx[beat.beat_index] = beat
            game_bible.experience_storyboard = [by_idx[k] for k in sorted(by_idx.keys())]
            updates_summary.append(f"Authored {len(proposal.added_or_updated_storyboard_beats)} Experience Storyboard beat(s)")

        # Ensure NarrativeBible container exists on game_bible
        if game_bible.narrative_bible is None:
            game_bible.narrative_bible = NarrativeBible(
                deliberation=f"Living Narrative Bible initialized by {speaker_role}.",
                world_title=game_bible.game_title,
                setting_summary=game_bible.elevator_pitch or "World setting under active Writers' Room development.",
            )
        else:
            game_bible.narrative_bible.world_title = game_bible.game_title
            if game_bible.elevator_pitch:
                game_bible.narrative_bible.setting_summary = game_bible.elevator_pitch

        # Characters (stored as lightweight metadata in the Bible; portraits are fabricated just-in-time by slice tasks when needed)
        char_alias_map: Dict[str, str] = {}
        if proposal.added_or_updated_characters:
            for char_sheet in proposal.added_or_updated_characters:
                if (
                    not char_sheet.portrait_asset
                    or not char_sheet.portrait_asset.startswith("assets/portraits/")
                ):
                    char_sheet.portrait_asset = f"assets/portraits/{char_sheet.character_id}_portrait.png"
            merged_chars, char_alias_map = consolidate_character_list(
                game_bible.narrative_bible.characters,
                proposal.added_or_updated_characters,
            )
            game_bible.narrative_bible.characters = merged_chars
            updates_summary.append(
                f"Added/updated character(s): {[c.name for c in proposal.added_or_updated_characters]}"
            )

        # Cutscenes
        if proposal.added_or_updated_cutscenes:
            existing_cs = {c.cutscene_id: c for c in game_bible.narrative_bible.cutscenes}
            for cs in proposal.added_or_updated_cutscenes:
                existing_cs[cs.cutscene_id] = cs
            game_bible.narrative_bible.cutscenes = list(existing_cs.values())
            updates_summary.append(f"Authored {len(proposal.added_or_updated_cutscenes)} cutscene script(s)")

        # Dialogue trees
        if proposal.added_or_updated_dialogue_trees:
            existing_trees = {t.root_node_id: t for t in game_bible.narrative_bible.dialogue_trees}
            for dt in proposal.added_or_updated_dialogue_trees:
                existing_trees[dt.root_node_id] = dt
            game_bible.narrative_bible.dialogue_trees = list(existing_trees.values())
            updates_summary.append(f"Authored {len(proposal.added_or_updated_dialogue_trees)} dialogue tree(s)")

        # Dialogue barks (stored as lightweight metadata in the Bible; audio WAVs are fabricated just-in-time by slice tasks when needed)
        if proposal.added_or_updated_barks:
            existing_barks = {b.bark_id: b for b in game_bible.narrative_bible.dialogue_barks}
            chars_by_id = {c.character_id: c for c in game_bible.narrative_bible.characters}
            for bark in proposal.added_or_updated_barks:
                if bark.speaker_id in char_alias_map:
                    bark.speaker_id = char_alias_map[bark.speaker_id]
                if not bark.voice_clip or not bark.voice_clip.startswith("assets/"):
                    bark.voice_clip = f"assets/audio/{bark.bark_id}.wav"
                char_obj = chars_by_id.get(bark.speaker_id)
                if char_obj is None and game_bible.narrative_bible.characters:
                    spk_tokens = _normalize_char_tokens(bark.speaker_id, drop_generic=True)
                    for c_cand in game_bible.narrative_bible.characters:
                        c_tokens = _normalize_char_tokens(c_cand.name, drop_generic=True) | _normalize_char_tokens(c_cand.character_id, drop_generic=True)
                        if spk_tokens and spk_tokens.issubset(c_tokens):
                            char_obj = c_cand
                            bark.speaker_id = c_cand.character_id
                            break
                existing_barks[bark.bark_id] = bark
            game_bible.narrative_bible.dialogue_barks = list(existing_barks.values())
            updates_summary.append(f"Added {len(proposal.added_or_updated_barks)} voice bark(s)")

        # Iterative Slice Roadmap updates
        if proposal.added_or_updated_slices:
            by_key: Dict[str, GameSliceSpec] = {}
            for s in game_bible.slice_roadmap:
                by_key[s.slice_id or f"idx_{s.slice_index}"] = s
            for new_s in proposal.added_or_updated_slices:
                matched_key = None
                for k, cur_s in by_key.items():
                    if cur_s.slice_id == new_s.slice_id or cur_s.slice_index == new_s.slice_index:
                        matched_key = k
                        break
                if matched_key is not None:
                    by_key[matched_key] = new_s
                else:
                    by_key[new_s.slice_id or f"idx_{new_s.slice_index}"] = new_s
            merged_slices = sorted(by_key.values(), key=lambda x: x.slice_index)
            explicit_active = [new_s for new_s in proposal.added_or_updated_slices if new_s.status == "active"]
            if explicit_active:
                game_bible.active_slice_index = explicit_active[-1].slice_index
            if merged_slices and not any(s.status == "active" for s in merged_slices):
                active_idx = max(0, min(game_bible.active_slice_index, len(merged_slices) - 1))
                merged_slices[active_idx].status = "active"
            game_bible.slice_roadmap = merged_slices
            updates_summary.append(
                f"Strategized {len(proposal.added_or_updated_slices)} game slice(s): {[s.slice_id for s in proposal.added_or_updated_slices]}"
            )
            self._emit_event(
                "bible_slice_roadmap_updated",
                {
                    "deliberation": proposal.deliberation,
                    "speaker_role": speaker_role,
                    "slice_count": len(game_bible.slice_roadmap),
                    "active_slice_index": game_bible.active_slice_index,
                    "slices": [s.model_dump() for s in game_bible.slice_roadmap],
                },
            )

        active_slice_idx_update = getattr(proposal, "active_slice_index_update", None)
        if active_slice_idx_update is not None:
            game_bible.active_slice_index = max(0, int(active_slice_idx_update))
            for idx_s, s in enumerate(game_bible.slice_roadmap):
                if idx_s < game_bible.active_slice_index:
                    s.status = "completed"
                elif idx_s == game_bible.active_slice_index:
                    s.status = "active"
                else:
                    s.status = "planned"
            updates_summary.append(f"Set active_slice_index={game_bible.active_slice_index}")

        # Game-design sections (levels, loop, engagement, onboarding, goals)
        for attr, label in (
            ("updated_level_generation_strategy", "level_generation_strategy"),
            ("updated_core_loop", "core_loop"),
            ("updated_engagement_plan", "engagement_plan"),
            ("updated_onboarding_plan", "onboarding_plan"),
            ("updated_goal_hierarchy", "goal_hierarchy"),
        ):
            value = getattr(proposal, attr, None)
            if value is not None:
                setattr(game_bible, label, value)
                updates_summary.append(f"Updated {label.replace('_', ' ')}")

        # Backlog requirements raised for the implementation side (already validated against the board)
        mem_note_raw = str(getattr(proposal, "memory_note", "") or "").strip()
        if mem_note_raw:
            eff_mem_dir = self._resolve_memory_dir(candidate_dir)
            if eff_mem_dir is not None:
                from engine.studio.game_memory import GameMemoryStore

                active_sid = ""
                for s in (game_bible.slice_roadmap or []):
                    if getattr(s, "status", "") == "active" and getattr(s, "slice_id", ""):
                        active_sid = str(s.slice_id)
                        break
                saved_mem = GameMemoryStore(
                    eff_mem_dir,
                    event_callback=getattr(self, "event_callback", None),
                    client=self.client,
                ).parse_and_apply_compact_note(
                    author_role=speaker_role,
                    raw_note=mem_note_raw,
                    deliberation=proposal.deliberation,
                    slice_id=active_sid,
                    client=self.client,
                )
                if saved_mem is not None:
                    updates_summary.append(
                        f"Recorded long-term memory [{saved_mem.memory_bank}:{saved_mem.kind}] '{saved_mem.title}'"
                    )

        if proposal.backlog_additions:
            if self.board is None:
                raise StudioAgentError("A backlog_additions proposal arrived but no Production Board is attached.")
            active_slice_obj = game_bible.get_active_slice()
            default_slice_id = active_slice_obj.slice_id if active_slice_obj is not None else ""
            for prop in proposal.backlog_additions:  # level maps all rewrite the shared layout file: serialise them
                if prop.kind == "level_map" and "level_layout.json" not in prop.target_modules:
                    prop.target_modules.append("level_layout.json")
            added = self.board.add_items(
                proposal.backlog_additions,
                raised_by_role=speaker_role,
                slice_id=default_slice_id,
            )
            updates_summary.append(f"Raised {len(added)} backlog item(s): {[a.item_id for a in added]}")

        # Pre-execution Council shaping / enrichment / decomposition of existing unstarted board items
        refinements = getattr(proposal, "backlog_refinements", None) or []
        if refinements:
            if self.board is None:
                raise StudioAgentError("A backlog_refinements proposal arrived but no Production Board is attached.")
            shaped = self.board.apply_refinements(
                refinements,
                by_role=speaker_role,
            )
            updates_summary.append(
                f"Shaped/split {len(refinements)} backlog item(s) -> {[s.item_id for s in shaped]}"
            )

        # 16:9 Visual Mockup Board synthesis
        if proposal.mockup_to_generate is not None:
            mockup = proposal.mockup_to_generate
            idx = len(game_bible.visual_mockups) + 1
            if not mockup.image_path or not mockup.image_path.startswith("assets/mockups/"):
                mockup.image_path = f"assets/mockups/mockup_{idx}_{mockup.mockup_id}.png"
            if mockup.camera_and_spatial_paradigm and not game_bible.camera_and_spatial_paradigm:
                game_bible.camera_and_spatial_paradigm = mockup.camera_and_spatial_paradigm
            if mockup.color_palette_hex and not game_bible.color_palette_hex:
                game_bible.color_palette_hex = list(mockup.color_palette_hex)
            if mockup.scene_layers and not game_bible.scene_layers:
                game_bible.scene_layers = list(mockup.scene_layers)
            if self.visual_agent is not None and candidate_dir is not None and getattr(self.visual_agent, "client", None) is not None:
                self._emit_activity(
                    agent_role=speaker_role,
                    status_text=f"Synthesizing 16:9 Concept Art Mockup '{mockup.title}'...",
                    active_agents=[speaker_role, "VisualTasksAgent"],
                    phase="mockup_synthesis",
                )
                mockup_dest = Path(candidate_dir) / mockup.image_path
                self.visual_agent.generate_concept_mockup(
                    mockup=mockup,
                    output_path=mockup_dest,
                    style_guide=game_bible.visual_style_guide,
                )
            game_bible.visual_mockups.append(mockup)
            updates_summary.append(f"Synthesized 16:9 Visual Mockup: '{mockup.title}'")
            self._emit_event(
                "bible_mockup_generated",
                {
                    "deliberation": mockup.deliberation,
                    "mockup_id": mockup.mockup_id,
                    "title": mockup.title,
                    "image_path": mockup.image_path,
                    "ui_callout_notes": mockup.ui_callout_notes,
                },
            )

        routed_spec, handoff_reason = _parse_council_reroute(proposal)
        effective_addressed_to = list(proposal.addressed_to or [])
        if routed_spec is not None and routed_spec["role"] != speaker_role:
            routed_role = routed_spec["role"]
            effective_addressed_to = [routed_role] + [r for r in effective_addressed_to if r != routed_role]
            updates_summary.append(f"routed_to:{routed_role}")
            self._emit_event(
                "council_role_rerouted",
                {
                    "deliberation": proposal.deliberation,
                    "reasoning_rationale": proposal.deliberation,
                    "from_role": speaker_role,
                    "to_role": routed_role,
                    "handoff_reason": handoff_reason,
                },
            )

        if self.board is not None and updates_summary:
            self.board.note_design_revision()

        utterance = WritersRoomUtterance(
            deliberation=proposal.deliberation,
            turn_index=turn_index,
            speaker_role=speaker_role,
            speaker_display_name=speaker_display_name,
            spoken_dialogue=proposal.spoken_dialogue,
            addressed_to=effective_addressed_to,
            bible_updates_summary=updates_summary,
        )
        game_bible.writers_room_transcript.append(utterance)
        self._emit_event(
            "writers_room_utterance",
            {
                "deliberation": utterance.deliberation,
                "utterance_id": utterance.utterance_id,
                "turn_index": utterance.turn_index,
                "speaker_role": utterance.speaker_role,
                "speaker_display_name": utterance.speaker_display_name,
                "spoken_dialogue": utterance.spoken_dialogue,
                "addressed_to": utterance.addressed_to,
                "bible_updates_summary": utterance.bible_updates_summary,
            },
        )
        self._persist_bible(game_bible, candidate_dir=candidate_dir)
        return utterance

    def _evaluate_specialist_listen(
        self,
        specialist: Dict[str, Any],
        game_bible: GameBibleSpec,
        dna_summary: str,
    ) -> WritersRoomListenCheck:
        """Asks a listening Council Lead (or invited Sub-Agent) if they want to speak or call in a Sub-Agent right now."""
        role = specialist["role"]
        display_name = specialist["display_name"]
        focus = specialist["focus"]
        subagents = specialist.get("subagents") or []
        bible_snapshot = self._format_bible_snapshot(game_bible)

        subagent_clause = ""
        if subagents:
            sub_details = []
            for sr in subagents:
                sp_obj = next((s for s in SPECIALIST_ROSTER if s["role"] == sr), None)
                if sp_obj:
                    sub_details.append(f"- `{sr}` ({sp_obj['display_name']}): {sp_obj['focus']}")
            subagent_clause = (
                "SUB-AGENTS UNDER YOUR DOMAIN (call on them when their specialized craft will best advance this game):\n"
                + "\n".join(sub_details)
                + "\nIf you believe one of your Sub-Agents should chime in right now instead of (or to deepen) your domain, "
                f"set `consult_subagent_role` to one of {subagents} and provide `subagent_brief` explaining what you want them to address "
                "(or leave `consult_subagent_role = null` to speak yourself).\n\n"
            )

        dna_header = self._dna_header_if_absent(dna_summary, bible_snapshot)
        prompt = (
            f"{dna_header}"
            f"{bible_snapshot}\n\n"
            f"You are the {display_name} ({role}) sitting in an open-ended Game Studio Writers' Room.\n"
            f"Your specialty focus: {focus}\n\n"
            f"{STUDIO_CRITICAL_DISCOURSE_CHARTER}\n\n"
            f"{subagent_clause}"
            f"Listen to the conversation above and inspect the Living Game Bible.\n"
            f"Strategize what this specific game needs most right now from your domain (or from one of your Sub-Agents if applicable): "
            f"catering respectfully to any Executive Producer direction while skeptically challenging weak, generic, over-scoped, or ungrounded peer ideas "
            f"(never speak up just to echo or praise a colleague!), answering a colleague who called on you, or shaping the active slice's design, world, "
            f"script/dialogue, visuals, onboarding/UI, progression, audio, or technical architecture.\n"
            f"Return your structured WritersRoomListenCheck with `deliberation` FIRST."
        )
        return self._call_structured_llm(prompt, WritersRoomListenCheck, temperature=0.4)

    def _build_specialist_shared_prefix(
        self,
        dna_summary: str,
        bible_snapshot: str,
        extra_context_block: str = "",
    ) -> str:
        dna_header = self._dna_header_if_absent(dna_summary, bible_snapshot)
        extra_sec = (
            f"{extra_context_block.strip()}\n\n"
            if (extra_context_block and extra_context_block.strip())
            else ""
        )
        return (
            f"{dna_header}"
            f"{bible_snapshot}\n\n"
            f"{STUDIO_CRITICAL_DISCOURSE_CHARTER}\n\n"
            f"{extra_sec}"
        )

    @staticmethod
    def _format_utterance_delta(game_bible: GameBibleSpec, start_idx: int) -> str:
        tx = list(game_bible.writers_room_transcript or [])
        if start_idx >= len(tx):
            return ""
        return "\n".join(
            f"  [Turn {u.turn_index}] {u.speaker_display_name} ({u.speaker_role}): {u.spoken_dialogue}"
            for u in tx[start_idx:]
        )

    def _execute_specialist_turn(
        self,
        specialist: Dict[str, Any],
        listen_check: WritersRoomListenCheck,
        game_bible: GameBibleSpec,
        dna_summary: str,
        extra_context_block: str = "",
        require_backlog_addition: bool = False,
        extra_multimodal_parts: Optional[List[Any]] = None,
        allow_council_reroute_handoff: bool = False,
        visited_council_roles: Optional[set[str]] = None,
        *,
        frozen_bible_snapshot: Optional[str] = None,
        recent_turn_delta: str = "",
        cacheable_prefix: str = "",
        min_calls_before_create: int = 2,
        omit_playtest_session_ids: Optional[Set[str]] = None,
    ) -> WritersRoomTurnProposal:
        """Executes the chosen specialist's turn to speak to the room and update the Living Game Bible."""
        role = specialist["role"]
        display_name = specialist["display_name"]
        focus = specialist["focus"]
        subagents = specialist.get("subagents") or []
        parent_lead = specialist.get("parent_lead")
        bible_snapshot = (
            frozen_bible_snapshot
            if frozen_bible_snapshot is not None
            else self._format_bible_snapshot(
                game_bible,
                omit_playtest_session_ids=omit_playtest_session_ids,
            )
        )

        self._emit_activity(
            agent_role=role,
            status_text=(
                f"{display_name} ({role}) is formulating their Writers' Room contribution"
                + (f": {listen_check.intended_contribution_summary[:110]}" if listen_check.intended_contribution_summary else "...")
            ),
            active_agents=[role],
            phase="speaking",
            extra={
                "speaker_display_name": display_name,
                "tier": specialist.get("tier", "lead"),
                "parent_lead": parent_lead,
                "intended_contribution": listen_check.intended_contribution_summary,
            },
        )

        hierarchy_note = ""
        if subagents:
            hierarchy_note = (
                f"COUNCIL HIERARCHY: You are a Core Council Domain Lead. Your specialized Sub-Agents are {subagents}. "
                f"Whenever you want one of your Sub-Agents (or another Council member) to chime in next, include their role name in `addressed_to`.\n"
            )
        elif parent_lead:
            hierarchy_note = (
                f"COUNCIL HIERARCHY: You are a specialized Sub-Agent working with `{parent_lead}` and the Core Council. "
                f"Bring your specialized craft to bear on what is best for THIS game, and name `{parent_lead}` or any colleague in `addressed_to` when handing back.\n"
            )

        domain_instructions = ""
        if role == "UXExperienceDesigner":
            domain_instructions = (
                "Domain Craft & Bible Fields (UXExperienceDesigner): Put yourself empathetically in the shoes of a brand-new player booting this game "
                "for the first time with zero external documentation. Strategize how THIS specific game brings the player into the world so they "
                "understand what they are doing and why (`purpose_and_stakes_framing`), how their actions causally connect (`input -> entity response -> "
                "world consequence` in `causal_chain_and_feedback_strategy`), how every bound input key is made discoverable, and what feedback occurs "
                "when an action is blocked or attempted before its prerequisite—using whatever onboarding strategy (environmental cues, NPC barks, "
                "staged introductory interaction, or diegetic instruments) best fits THIS game's genre and tone. "
                "CRITICAL: Onboarding is ONLY the opening 60 seconds of play layered onto the real game—it is NEVER the entire game! Do NOT propose standalone "
                "'tutorial manager' or 'onboarding milestone map' tasks that reduce the game into a linear 4-step checklist that ends once the player learns the controls; "
                "instead, design HUD/toolbar/affordance UI (`ui` tasks) that empowers the player to play the full `core_loop` (moment-to-moment, minute-to-minute, and session loops) "
                "and track their progress across `goal_hierarchy`. Shape `updated_first_60_seconds_hook`, `updated_diegetic_ui_philosophy`, "
                "`updated_onboarding_plan`, and `added_or_updated_storyboard_beats`."
            )
        elif role == "VisualArtDirector":
            domain_instructions = (
                "Domain Craft & Bible Fields (VisualArtDirector): You can shape `visual_style_guide_update`, `updated_camera_and_spatial_paradigm`, "
                "`updated_color_palette_hex`, `added_or_updated_scene_layers`, and propose a `mockup_to_generate` "
                "(VisualMockupArtifact with a rich 16:9 `scene_description` showing a clear, readable gameplay viewport and `ui_callout_notes`). "
                "When strategizing visual rendering in Godot 4, champion the best of both worlds—combining procedural generation (noise/cellular-automata/spline "
                "geometry, custom `CanvasItem` `ShaderMaterial`, `GPUParticles2D`, `PointLight2D`) with generated visual assets (`tile_surface` seamless textures "
                "mapped onto procedural `Polygon2D`s via `StudioRigHelper.create_tiled_polygon` or `Line2D` strip textures, environmental decals/landmarks, "
                "and discrete `sprite_sheet`/`articulated_parts` rigs for characters and interactive structures) so the game never relies on flat untextured "
                "`Polygon2D` slabs or bare programmer-art `Line2D` strokes. "
                "You can also call on your Sub-Agents `UXExperienceDesigner` and `AudioAtmosphereDesigner` via `addressed_to`."
            )
        elif role == "ScriptwriterAgent":
            domain_instructions = (
                "Domain Craft & Bible Fields (ScriptwriterAgent): As Lead Scriptwriter & Narrative Dramatist, strategize whether and how scripted scenes, "
                "sharp dialogue, or contextual voice barks can elevate THIS game. When appropriate for the game, author or refine "
                "`added_or_updated_cutscenes` (scene scripts with dramatic staging and line-by-line dialogue), `added_or_updated_dialogue_trees` "
                "(branching conversations, player choices, and consequence flags), `added_or_updated_barks` (reactive in-gameplay lines), "
                "or enrich character voices in `added_or_updated_characters`."
            )
        elif role == "CharacterDesigner":
            domain_instructions = (
                "Domain Craft & Bible Fields (CharacterDesigner): Strategize the actors, entities, or personas that fit THIS game's setting and loop in "
                "`added_or_updated_characters` (each with a unique `character_id`, distinct visual silhouette, motivation, and voice signature; "
                "reuse existing `character_id`s when refining a character already in the Bible), plus `added_or_updated_dialogue_trees` or `added_or_updated_barks` where fitting."
            )
        elif role == "StoryWorldArchitect":
            domain_instructions = (
                "Domain Craft & Bible Fields (StoryWorldArchitect): Shape the world lore, dramatic stakes, and narrative framing via "
                "`updated_core_fantasy`, `updated_elevator_pitch`, `concept_evolution_milestone`, and `added_or_updated_cutscenes`. "
                "You can call on your Sub-Agents `ScriptwriterAgent` (to write scene scripts, dialogue trees, and barks) and `CharacterDesigner` "
                "(to design characters and personas) via `addressed_to` whenever their craft will strengthen the game."
            )
        elif role == "GameplayDesigner":
            domain_instructions = (
                "Domain Craft & Bible Fields (GameplayDesigner): Pressure-test the mechanics for genuine fun, player agency, and alignment with the "
                "Target Player's Steam DNA. Ask: 'What is the LARGER POINT of this game? Why is this genre compelling to play beyond the first 60 seconds?' "
                "NEVER allow the team to build 'the smallest slice possible' that reduces the game to a 1-step tutorial or a pre-solved diorama while deferring "
                "the actual game loop to unbuilt future slices! Shape `gameplay_pillars_additions`, `updated_core_loop` (player verbs, `player_agency_and_optimization_depth` "
                "explaining how the player builds/experiments/optimizes/strategizes their own approach with no single pre-scripted solution, plus moment-to-moment, "
                "minute-to-minute, and session loops, feedback, reward cadence, failure and recovery), `updated_engagement_plan`, `added_or_updated_slices` "
                "(ensuring Slice 0 includes the full minute-to-minute and session loop and player toolkit, NEVER deferring the core point of the game), and "
                "`concept_evolution_milestone`. When a Production Board is attached, raise `mechanic_code` tasks that build both the moment-to-moment verbs AND "
                "the minute-to-minute/session loop progression. You can call on your Sub-Agents `LevelAndProgressionDesigner` and `GameplaySystemsEngineer` via `addressed_to`."
            )
        elif role == "GameplaySystemsEngineer":
            domain_instructions = (
                "Domain Craft & Bible Fields (GameplaySystemsEngineer): Translate the game's full 3-timescale `core_loop` (moment-to-moment, minute-to-minute, "
                "and session loops), `player_agency_and_optimization_depth`, `goal_hierarchy`, camera/spatial paradigm, and visual direction into "
                "concrete Godot 4 engineering strategies in `added_or_updated_engineering_specs` (`MechanicEngineeringSpec` items covering node/collider architecture, "
                "hybrid procedural + generated texture/sprite/rig rendering, active `Camera2D` tracking across world bounds with HUD isolated on a `CanvasLayer`, "
                "deterministic state machines that start unbuilt/unsolved at t=0 and give the player a full interactive toolkit/sandbox to build, experiment, optimize, "
                "and progress through escalating goals, and falsifiable headless test invariants). Never design or accept a state machine that reduces the game to a "
                "4-step linear tutorial or pre-solves the player's work in `_ready()`."
            )
        elif role == "LevelAndProgressionDesigner":
            domain_instructions = (
                "Domain Craft & Bible Fields (LevelAndProgressionDesigner): Strategize the game's world and progression in `updated_level_generation_strategy` "
                "(bespoke `generation_approach`, `world_structure`, `runtime_construction_plan`, `level_archetypes`, `generation_rules`, and difficulty curve), "
                "`updated_goal_hierarchy` (goals across immediate, short-term, mid-term, and long-term horizons, progression structure, and end state), and "
                "`added_or_updated_slices` (where Slice 0 delivers the COMPLETE playable game loop across all 3 time scales of `core_loop` and multi-horizon `goal_hierarchy`—never "
                "shrinking the build into a tutorial-only room or deferring the real progression and systemic depth to future slices!). When a Production Board is attached, "
                "raise `level_map` and progression `mechanic_code` tasks that realize the multi-region world and escalating goals."
            )
        elif role == "AudioAtmosphereDesigner":
            domain_instructions = (
                "Domain Craft & Bible Fields (AudioAtmosphereDesigner): Shape the game's acoustic identity and reactive soundscape in "
                "`audio_direction_update`, contribute reactive or atmospheric `added_or_updated_barks` where suited to this game, and when a Production Board "
                "is attached, raise `audio_asset` `backlog_additions` so the game's ambient and mechanical soundscape is fabricated and wired."
            )

        shared_head = self._build_specialist_shared_prefix(
            dna_summary,
            bible_snapshot,
            extra_context_block,
        )
        delta_block = (
            f"NEW WRITERS' ROOM TURNS SINCE SNAPSHOT:\n{recent_turn_delta.strip()}\n\n"
            if (recent_turn_delta and recent_turn_delta.strip())
            else ""
        )
        prompt = (
            f"{shared_head}"
            f"{delta_block}"
            f"You are the {display_name} ({role}) speaking in the Game Studio Writers' Room.\n"
            f"Your specialty focus: {focus}\n"
            f"{hierarchy_note}"
            f"Context for your turn: {listen_check.intended_contribution_summary}\n\n"
            f"{domain_instructions}\n"
            f"STRATEGIZE WHAT IS BEST FOR THIS GAME (NO PRESCRIPTIONS): Do not follow rote templates. Reason from first principles about what mechanics, "
            f"world structure, empathetic new-player onboarding/UI, narrative/script moments, hybrid visual rendering (combining procedural geometry/shaders/particles "
            f"with generated surface textures, decals, and sprite/rig assets), and iterative vertical slices (`added_or_updated_slices`) best serve THIS specific game.\n"
            f"THE CONCEPT IS STILL EVOLVING: the premise above is provisional (except for Executive Producer directives, which you must always respect and cater to). "
            f"Do not merely decorate or rubber-stamp peer proposals. Pressure-test peer ideas skeptically from your domain; "
            f"if your contribution exposes a weakness, unplayable abstraction, or over-scoped tangle, challenge it directly and improve it via the `updated_*` "
            f"fields and a `concept_evolution_milestone` explaining what changed and why. Keep what is alive, replace what is generic.\n"
            f"CLARITY & CONCRETE GAME DESIGN (NO PSEUDO-SCIENTIFIC JARGON SALAD): Speak in clear, direct, plain-spoken game design language. "
            f"Always describe concretely what the player sees on screen, what they directly control or manipulate, what obstacle or challenge pushes back, "
            f"and why the moment-to-moment interaction is fun. Avoid obscure academic, physics, or materials-science buzzword mashups that obscure the actual game.\n"
            f"Speak candidly, skeptically, and directly to your fellow designers in `spoken_dialogue`—NEVER open with 'I completely agree with...', peer flattery, or "
            f"restatements of earlier speakers, and never recite raw telemetry floats—while treating any Executive Producer message with full respect and catering directly "
            f"to what they asked for. Keep Production Board tasks focused on covering the active slice's scope cleanly rather than trying to build everything at once. "
            f"Leave `memory_note` empty (`''`) on routine turns—never duplicate Game Bible updates or ordinary dialogue points into `memory_note`; use it ONLY when deferring an out-of-slice goal or recording a non-obvious cross-slice constraint. "
            f"DYNAMIC PEER ROUTING (`route_to_council_role`): If an issue, Steering directive, or design/engineering question belongs primarily to another Council specialist's domain—or needs a direct follow-up turn from another Council role (`GameplayDesigner`, `StoryWorldArchitect`, `VisualArtDirector`, `LevelAndProgressionDesigner`, `GameplaySystemsEngineer`, `ScriptwriterAgent`, `CharacterDesigner`, `UXExperienceDesigner`, `AudioAtmosphereDesigner`) to author their Bible fields or domain tasks—set `route_to_council_role` to that exact role name and specify what they should handle in `routing_handoff_reason`. "
            f"Call out specific colleagues or Sub-Agents in `addressed_to`, and provide `deliberation` FIRST."
        )
        turn_schema = WritersRoomTurnProposal.for_role(role) if is_real_genai_client(self.client) else WritersRoomTurnProposal
        eff_mem_dir = self._resolve_memory_dir(None)
        mem_briefing = ""
        if eff_mem_dir is not None and (eff_mem_dir / "game_memory.json").is_file():
            from engine.studio.game_memory import GameMemoryStore

            mem_briefing = GameMemoryStore(
                eff_mem_dir, event_callback=getattr(self, "event_callback", None)
            ).format_role_memory_briefing(role)
        mem_extra = mem_briefing.strip()

        if self.board is None:
            full_no_board_prompt = prompt + ("\n\n" + mem_extra if mem_extra else "")
            if eff_mem_dir is not None:
                return self._structured_with_validation(
                    full_no_board_prompt,
                    turn_schema,
                    lambda _p: [],
                    parts=extra_multimodal_parts,
                    caller_role=role,
                    candidate_dir=eff_mem_dir,
                    cacheable_prefix=cacheable_prefix,
                    min_calls_before_create=min_calls_before_create,
                )
            if getattr(self._call_structured_llm, "__func__", None) is CreativeWritersRoom._call_structured_llm:
                return self._call_structured_llm(
                    full_no_board_prompt,
                    turn_schema,
                    parts=extra_multimodal_parts,
                    cacheable_prefix=cacheable_prefix,
                    min_calls_before_create=min_calls_before_create,
                )
            return self._call_structured_llm(full_no_board_prompt, turn_schema)
        tool_dir_block = render_compact_tool_directory(layer="council")
        board_block = (
            f"{self.board.snapshot_text()}\n"
            f"{tool_dir_block}\n"
            f"BUILD BACKLOG (ACTIVE SLICE — COUNCIL TASK SHAPING, ENRICHMENT & DECOMPOSITION):\n"
            f"Before tasks are built by workers, the Council shapes what is needed so the team builds a cohesive, playable game rather than disconnected subsystems.\n"
            f"1. INSPECT EXISTING TASKS ON THE PRODUCTION BOARD (`backlog_refinements`):\n"
            f"   - Review every unstarted (`ready` or `blocked`) item currently on the Production Board from your specialist perspective.\n"
            f"   - ENRICH (`action='enrich'`): If an existing task needs direction from your domain—such as how it connects to the world/levels, what shared data contracts or node structures it must expose, how the player discovers and interacts with it in the HUD/UI without overlapping panels, or which visual rigs/audio cues it should bind—add a `WorkItemRefinement` in `backlog_refinements` with `action='enrich'`, `spec_addition_or_rewrite`, and any prerequisite `add_depends_on`.\n"
            f"   - SPLIT (`action='split'`): If an existing task is too monolithic (for example, bundling core simulation physics, player placement/building tools, world progression, and HUD into one giant script task, or bundling multiple distinct rigs into one asset task), decompose it via `WorkItemRefinement` with `action='split'` and `split_items=[...]` into focused, dependency-linked tasks! Any downstream tasks on the board that depended on the original task will automatically re-point to the split tasks.\n"
            f"2. RAISE NEW, DISCOVERY, OR COMPANION TASKS (`backlog_additions`):\n"
            f"   - Raise `backlog_additions` needed for the ACTIVE SLICE so Slice 0 ships as a complete, cohesive playable vertical slice (world/level traversal with proper camera tracking, core gameplay loop, empathetic new-player onboarding/HUD consolidated on a single authoritative CanvasLayer, visual assets/surfaces, and audio).\n"
            f"   - Whenever you want a read-only perceptual or architectural investigation before or alongside implementation (e.g., 1:1 sprite/sheet inspection, acoustic evaluation, codebase/signal inspection, or exploratory viewport diagnosis), you can raise a lock-free `kind='discovery'` task and list it in `depends_on` for downstream build tasks.\n"
            f"HYBRID PROCEDURAL + GENERATED ASSETS: Do NOT settle for flat untextured `Polygon2D` slabs or bare programmer-art `Line2D` strokes! "
            f"Raise `visual_asset` tasks (`tile_surface` seamless textures for procedural `Polygon2D`/`Line2D` terrain/conduits, environmental decals/landmarks, "
            f"and `sprite_sheet`/`articulated_parts` rigs for characters and interactive structures) alongside procedural shaders and particles.\n"
            f"ATOMIC 1-ENTITY-PER-`visual_asset` RULE: Each `visual_asset` item maps 1-to-1 to a SINGLE `.png` file (`assets/sprites/<id>.png`) and a single "
            f"`AssetRequirement` entry in `asset_library.json`. Every distinct character, creature, machine, or interactive structure rig (`sprite_sheet` or `articulated_parts`) "
            f"MUST be proposed as its own separate `visual_asset` item (1 entity/machine rig per `visual_asset` task)—NEVER bundle multiple distinct machines, characters, "
            f"or animated rigs (e.g. '1) Machine A; 2) Machine B; 3) Machine C') into a single umbrella `visual_asset` item!\n"
            f"DEPENDENCY & ASSET BLOCKING RULES (`depends_on` and `blocks_items`):\n"
            f"- Characters, avatars, creatures, and hero props in `mechanic_code` / `integration` tasks MUST bind actual fabricated sprites or articulated rigs—never primitive `draw_circle`/`draw_rect` stickmen.\n"
            f"- When you raise a `mechanic_code` or `integration` task that requires a character rig, surface texture, or discrete sprite, either include that `visual_asset` item in `depends_on` (proposing the `visual_asset` item in the same turn if it is not on the board yet), OR let the Task Lead spawn it on Turn 0 with `wait_for_spawned_tasks=True`.\n"
            f"- Conversely, when you raise a `visual_asset`, `animation`, or foundational system task that an ALREADY-PROPOSED task on the Production Board needs (for example, proposing the player's `visual_asset` character rig after a player controller `mechanic_code` task was already raised on an earlier turn), list those downstream task IDs in `blocks_items` (e.g. `blocks_items=['mechanic_...']`) or wire them via `backlog_refinements` (`add_depends_on`)!\n"
            f"- For each item you raise, provide a unique `item_id`, `slice_id` matching the active slice, a precise `spec`, `depends_on`, `blocks_items`, "
            f"`target_modules` (leave `target_modules=[]` for standalone `visual_asset`, `animation`, and `audio_asset` items; for `mechanic_code`, `ui`, `level_map`, and `integration` items, prefer encapsulated per-feature GDScript module paths such as `scenes/mechanics/<feature_id>.gd`, `scenes/entities/<entity_id>.gd`, `scenes/ui/<widget_id>.gd`, or `scenes/world/<region_id>.gd` that register in the `'state_providers'` node group so multiple SWE workers can build features in parallel with disjoint per-file locks rather than bottlenecking on a single monolithic file), and a `proof_spec`. "
            f"Do not duplicate existing items. The finished work will be shown to the Council panel for approval.\n"
            f"- SYSTEMIC CONTRACTS OVER HARDCODED COORDINATES: Define `spec` and `spec_addition_or_rewrite` (`enrich`) as generalizable, world-wide rules (querying authoritative world geometry/regions, node groups, and player-driven state across all valid locations)—never prescribe hardcoded `(x, y)` coordinate whitelists, bounding-box overrides, or fixed target coordinates for production mechanics (specific coordinates belong only as example test landmarks in `proof_spec`).\n"
            f"Buildable kinds right now: {self._supported_kinds() or 'all'}.\n"
        )

        def _validate(p: WritersRoomTurnProposal) -> List[str]:
            issues: List[str] = []
            routed_spec, _ = _parse_council_reroute(p)
            valid_handoff = bool(
                allow_council_reroute_handoff
                and routed_spec is not None
                and routed_spec["role"] != role
                and routed_spec["role"] not in (visited_council_roles or set())
            )
            has_board_action = bool(p.backlog_additions or getattr(p, "backlog_refinements", None))
            if require_backlog_addition and not has_board_action and not valid_handoff:
                issues.append(
                    "The Executive Producer reported a live issue or steering directive on a Production Board requiring action; "
                    "you MUST either populate `backlog_additions` / `backlog_refinements` with at least one concrete Production Board "
                    "task (`integration`, `mechanic_code`, `ui`, `visual_asset`, `audio_asset`, `level_map`, or `discovery`), OR set "
                    "`route_to_council_role` to an unvisited Council colleague whose domain owns creating that task."
                )
            if p.backlog_additions:
                issues.extend(self.board.validate_proposals(p.backlog_additions))
                issues.extend(self._unsupported_kind_issues(p.backlog_additions))
            refinements = getattr(p, "backlog_refinements", None) or []
            if refinements:
                issues.extend(self.board.validate_refinements(refinements, new_proposals=p.backlog_additions))
                for r in refinements:
                    if r.action == "split" and r.split_items:
                        issues.extend(self._unsupported_kind_issues(r.split_items))
            return issues

        full_turn_prompt = prompt + ("\n\n" + mem_extra if mem_extra else "") + "\n" + board_block
        return self._structured_with_validation(
            full_turn_prompt,
            turn_schema,
            _validate,
            parts=extra_multimodal_parts,
            caller_role=role,
            candidate_dir=eff_mem_dir,
            cacheable_prefix=cacheable_prefix,
            min_calls_before_create=min_calls_before_create,
        )

    # ------------------------------------------------------------------
    # Production Board: proof review, revision-cap decisions and backlog draining
    # ------------------------------------------------------------------
    def _supported_kinds(self) -> List[str]:
        return sorted(self.scheduler.handlers) if self.scheduler is not None else []

    def _unsupported_kind_issues(self, proposals: List[Any]) -> List[str]:
        supported = self._supported_kinds()
        if not supported:
            return []
        return [
            f"Item '{p.item_id}' has kind '{p.kind}', which no worker can build (buildable kinds: {supported})."
            for p in proposals
            if p.kind not in supported
        ]

    @staticmethod
    def _reviewer_for(item: Any) -> str:
        if item.kind in ("integration", "level_map") or not item.raised_by_role:
            return "CreativeDirector"
        return item.raised_by_role

    @classmethod
    def _reviewers_for(cls, item: Any) -> List[str]:
        """Returns the cross-functional Council review panel for `item`, incorporating both the item's domain
        and any autonomous specialists engaged by the Task Lead on the latest revision.
        All panelists must unanimously approve for the item to pass review."""
        primary = cls._reviewer_for(item)
        if item.kind == "discovery":
            return [primary] if primary else ["CreativeDirector"]
        if item.kind in ("mechanic_code", "ui", "integration"):
            panel = [primary, "GameplayDesigner", "UXExperienceDesigner", "VisualArtDirector"]
        elif item.kind == "level_map":
            panel = ["CreativeDirector", "LevelAndProgressionDesigner", "UXExperienceDesigner", "VisualArtDirector"]
        elif item.kind in ("visual_asset", "animation"):
            panel = [primary, "VisualArtDirector"]
        elif item.kind == "audio_asset":
            panel = [primary, "AudioAtmosphereDesigner"]
        else:
            panel = [primary]
        latest_notes = item.revisions[-1].worker_notes if getattr(item, "revisions", None) else ""
        if "visual_artist]" in latest_notes and "VisualArtDirector" not in panel:
            panel.append("VisualArtDirector")
        if "audio_designer]" in latest_notes and "AudioAtmosphereDesigner" not in panel:
            panel.append("AudioAtmosphereDesigner")
        if "swe_engineer]" in latest_notes:
            for r in ("GameplayDesigner", "UXExperienceDesigner", "VisualArtDirector"):
                if r not in panel:
                    panel.append(r)
        return list(dict.fromkeys(r for r in panel if r))

    def _concept_mockup_parts(self, game_bible: GameBibleSpec, candidate_dir: Optional[Path]) -> List[Any]:
        """Attaches the primary 16:9 Concept Art Mockup image so reviewers can compare proofs against the target visual direction."""
        from google.genai import types

        if candidate_dir is None or not getattr(game_bible, "visual_mockups", None):
            return []
        parts: List[Any] = []
        for mockup in game_bible.visual_mockups[:1]:
            rel = getattr(mockup, "image_path", "") or ""
            if not rel:
                continue
            path = Path(rel) if Path(rel).is_absolute() else (Path(candidate_dir) / rel)
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            parts.append(
                f"REFERENCE 16:9 CONCEPT ART MOCKUP '{mockup.title}' (file {path.name} — compare camera framing, character scale, "
                f"layered world art, and diegetic UI against this target):"
            )
            parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
        return parts

    def _load_image_part(self, path: Path) -> Optional[Any]:
        """Loads a rendered image file from disk into a multimodal Gemini Part for Council visual inspection."""
        from google.genai import types

        p = Path(path)
        if not p.is_file():
            return None
        suffix = p.suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
            return None
        mime = (
            "image/jpeg"
            if suffix in (".jpg", ".jpeg")
            else ("image/webp" if suffix == ".webp" else "image/png")
        )
        media_res = getattr(getattr(types, "MediaResolution", None), "MEDIA_RESOLUTION_HIGH", None)
        if media_res is not None:
            return types.Part.from_bytes(data=p.read_bytes(), mime_type=mime, media_resolution=media_res)
        return types.Part.from_bytes(data=p.read_bytes(), mime_type=mime)

    def _proof_parts(self, item: Any, candidate_dir: Optional[Path]) -> List[Any]:
        from engine.studio.audio_evaluator import AUDIO_MIME_BY_SUFFIX, INLINE_MEDIA_MAX_BYTES, VIDEO_MIME_BY_SUFFIX
        from google.genai import types

        if not item.revisions:
            return []
        latest = item.revisions[-1]
        parts: List[Any] = []
        resolved_paths: List[Path] = []
        audio_paths: List[Path] = []
        video_paths: List[Path] = []
        for rel in latest.proof_paths:
            path = Path(rel)
            if not path.is_absolute() and candidate_dir is not None:
                path = Path(candidate_dir) / rel
            if not path.is_file():
                raise StudioAgentError(f"Proof image for item '{item.item_id}' is missing on disk: {path}")
            suffix = path.suffix.lower()
            if suffix in (".png", ".jpg", ".jpeg", ".webp"):
                resolved_paths.append(path)
            elif suffix in AUDIO_MIME_BY_SUFFIX and 0 < path.stat().st_size <= INLINE_MEDIA_MAX_BYTES:
                if path not in audio_paths:
                    audio_paths.append(path)
            elif suffix in VIDEO_MIME_BY_SUFFIX and 0 < path.stat().st_size <= INLINE_MEDIA_MAX_BYTES:
                if path not in video_paths:
                    video_paths.append(path)
        if len(resolved_paths) > 1:
            strips = [p for p in resolved_paths if "filmstrip" in p.name.lower()]
            others = [p for p in resolved_paths if "filmstrip" not in p.name.lower()]
            if strips:
                non_pass1 = [p for p in strips if "_pass1." not in p.name.lower()]
                latest_strip = non_pass1[-1:] if non_pass1 else strips[-1:]
                resolved_paths = others[-2:] + latest_strip
            else:
                resolved_paths = others[-2:]
        for path in resolved_paths:
            suffix = path.suffix.lower()
            mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else ("image/webp" if suffix == ".webp" else "image/png")
            name_lower = path.name.lower()
            is_filmstrip = "filmstrip" in name_lower
            is_plate = name_lower.startswith("plate_")
            is_anim_sheet = name_lower.startswith("anim_sheet_") or name_lower.startswith("anim_rig_")
            if is_filmstrip:
                proof_note = (
                    " [QA CONTACT SHEET: Each cell shows the 16:9 game viewport inside a thin slate border; the top strip ('[QA] tick N in:[...]') "
                    "and bottom strip ('[QA] pos=(...) ...') outside each 16:9 frame are external QA test-harness annotations provided FOR THE COUNCIL "
                    "to verify simulation ticks, inputs, and live engine state—they are NOT inside the game viewport and must NEVER be flagged as in-game debug text]"
                )
            elif is_plate:
                proof_note = (
                    " [QA ALPHA INSPECTION PLATE: The side-by-side dark/light square checkerboard grids and cyan bounding-box rectangle are external QA "
                    "test-harness backgrounds drawn behind/around the transparent PNG to verify alpha cutout cleanliness—they are NOT inside the sprite PNG file "
                    "and must NEVER be flagged as cross-hatch/grid/box artifacts!]"
                )
            elif is_anim_sheet:
                proof_note = (
                    " [QA ANIMATION CONTACT SHEET: The dark square checkerboard grid, cyan cell borders, and horizontal yellow/gold baseline ruler lines "
                    "at the bottom of each frame are external QA test-harness alignment guides—they are NOT inside the sprite sheet PNG file and must NEVER "
                    "be flagged as yellow line or checkerboard artifacts!]"
                )
            else:
                proof_note = ""
            parts.append(
                f"PROOF IMAGE for item '{item.item_id}' (revision {latest.revision_index + 1}, file {path.name}){proof_note}:"
            )
            media_res = getattr(getattr(types, "MediaResolution", None), "MEDIA_RESOLUTION_HIGH", None)
            if media_res is not None:
                parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime, media_resolution=media_res))
            else:
                parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
            if is_filmstrip:
                fullres_kf = path.with_name(f"{path.stem}.keyframe_fullres.png")
                if fullres_kf.is_file():
                    parts.append(
                        f"FULL-RESOLUTION 1:1 KEYFRAME (native viewport pixels, final tick of {path.name}) for item '{item.item_id}' "
                        "[inspect HUD icons, character portraits, text legibility, world sprites, and shader/tile seams at 1:1 fidelity]:"
                    )
                    if media_res is not None:
                        parts.append(
                            types.Part.from_bytes(
                                data=fullres_kf.read_bytes(),
                                mime_type="image/png",
                                media_resolution=media_res,
                            )
                        )
                    else:
                        parts.append(types.Part.from_bytes(data=fullres_kf.read_bytes(), mime_type="image/png"))
        for ap in audio_paths[-2:]:
            a_mime = AUDIO_MIME_BY_SUFFIX[ap.suffix.lower()]
            parts.append(
                f"PROOF AUDIO for item '{item.item_id}' (revision {latest.revision_index + 1}, file {ap.name}):"
            )
            parts.append(types.Part.from_bytes(data=ap.read_bytes(), mime_type=a_mime))
        for vp in video_paths[-1:]:
            v_mime = VIDEO_MIME_BY_SUFFIX[vp.suffix.lower()]
            parts.append(
                f"PROOF AUDIOVISUAL CLIP for item '{item.item_id}' (revision {latest.revision_index + 1}, file {vp.name}):"
            )
            parts.append(types.Part.from_bytes(data=vp.read_bytes(), mime_type=v_mime))
        if not latest.proof_paths:
            parts.append(f"NOTE: item '{item.item_id}' returned NO proof images.")
        return parts

    @staticmethod
    def _format_item_for_review(item: Any, candidate_dir: Optional[Path] = None) -> str:
        def _clip_notes(txt: str) -> str:
            if len(txt) <= 3600:
                return txt
            return txt[:1200] + " ... [middle turns omitted] ... " + txt[-2400:]

        history = "\n".join(
            f"      rev {r.revision_index + 1}: UNVERIFIED_WORKER_CLAIMS={_clip_notes(r.worker_notes)!r} verdict={r.verdict or '(pending)'} feedback={r.feedback[:900]!r}"
            for r in item.revisions
        ) or "      (none)"
        failure_line = (
            f"\n    last_worker_failure (attempts={getattr(item, 'failure_count', 0)}): {item.failure[-800:]!r}"
            if getattr(item, "failure", "")
            else ""
        )
        excluded_line = (
            f"\n    excluded_modules (pruned by Council): {item.excluded_modules}"
            if getattr(item, "excluded_modules", None)
            else ""
        )
        ground_truth_qa_block = ""
        if candidate_dir is not None:
            from engine.studio.workers import _safe_id

            c_root = Path(candidate_dir)
            safe_item = _safe_id(str(item.item_id))
            test_gd_file = c_root / "tests" / f"test_{safe_item}.gd"
            if test_gd_file.is_file():
                raw_tgd = test_gd_file.read_text(encoding="utf-8", errors="replace").strip()
                if len(raw_tgd) > 2400:
                    raw_tgd = raw_tgd[:2400] + "\n# ... [truncated] ..."
                ground_truth_qa_block += (
                    f"\n    qa_test_script_source (`tests/test_{safe_item}.gd` on disk — verify whether this test observes honestly vs. mutates game state):\n"
                    + "\n".join(f"      {ln}" for ln in raw_tgd.splitlines())
                )
            if getattr(item, "revisions", None):
                latest_proofs = getattr(item.revisions[-1], "proof_images", None) or []
                if latest_proofs:
                    from engine.studio.proofs import format_filmstrip_telemetry_progression

                    p_cand = Path(latest_proofs[-1])
                    if not p_cand.is_absolute():
                        p_cand = c_root / p_cand
                    t_prog = format_filmstrip_telemetry_progression(p_cand)
                    if t_prog:
                        ground_truth_qa_block += f"\n    latest_proof_runtime_telemetry: {t_prog[:1600]}"
        return (
            f"  ITEM [{item.item_id}] kind={item.kind} title={item.title!r} state={item.state}\n"
            f"    spec: {item.spec}\n    proof_spec: {item.proof_spec}\n"
            f"    target_modules: {getattr(item, 'target_modules', [])} | depends_on: {item.depends_on}{excluded_line}\n"
            f"    revision_cap: {item.max_revisions} (spec rewrites so far: {item.spec_resets}/{MAX_SPEC_RESETS})\n"
            f"    decision_reason: {item.decision_reason or '-'}{failure_line}\n    history:\n{history}{ground_truth_qa_block}"
        )

    @staticmethod
    def _role_review_mandate(role: str) -> str:
        if role == "UXExperienceDesigner":
            return (
                "ROLE MANDATE (UXExperienceDesigner): You have VETO power over UI/UX and empathetic new-player legibility. Put yourself in the shoes of a "
                "brand-new player seeing this screen for the first time with zero external instructions: can they tell what they are doing and why "
                "(purpose/stakes), which objects are interactive, how their actions causally affect the world, and what is world vs. HUD? Evaluate the proof "
                "against the UI, HUD, and onboarding strategy (`onboarding_plan`) designed in the Game Bible. Immediately `request_changes` if the game "
                "relies on hidden/unexplained keys with no discoverable affordance, if onboarding steps can only be skipped via a fake key/setup shortcut "
                "instead of performing the actual in-world action, if a blocked/locked state freezes the player with no clear feedback on "
                "why or how to unblock it, if HUD widgets are placed in world-space where the player avatar can walk behind or collide with them (HUD must "
                "live on a `CanvasLayer`), if a proof shows internal QA test-runner strings (e.g. 'HEADLESS MECHANICS', 'ALL PASS') drawn INSIDE the 16:9 "
                "game viewport, or if UI panels occlude the playable center. "
                "IMPORTANT: The thin header bar ('[QA] tick N in:[...]') and footer bar ('[QA] pos=...') outside each 16:9 frame in `filmstrip_*.png`, "
                "as well as the dark/light checkerboard, cyan bounding boxes, and yellow baseline ruler lines in `plate_*.png` and `anim_sheet_*.png`, "
                "are external QA Test Harness overlays added FOR THE COUNCIL—do NOT flag those external QA overlays as in-game HUD or sprite defects.\n"
            )
        if role == "VisualArtDirector":
            return (
                "ROLE MANDATE (VisualArtDirector): You have VETO power over visual fidelity and spatial staging. Compare the proof directly against the 16:9 Concept Art "
                "Mockup, Game Bible Visual Style Guide, and the cartridge's fabricated assets. We demand hybrid procedural + generated visual craft: procedural "
                "geometry/contours (`Polygon2D`, `Line2D`) MUST be combined with fabricated surface textures (`tile_surface` via `StudioRigHelper.create_tiled_polygon` / "
                "`TEXTURE_REPEAT_ENABLED`, `Line2D` strip textures with `LINE_TEXTURE_TILE`), custom `ShaderMaterial`s, environmental decals, and `GPUParticles2D`—NEVER "
                "solid flat-colored untextured `Polygon2D` slabs, blank white/monochrome backgrounds, or bare 8-point `Line2D` programmer-art zigzags! "
                "Characters, avatars, creatures, and interactive world structures MUST bind actual fabricated sprites or articulated rigs from `res://assets/sprites/*.png` "
                "and be coherently layered into the environment (not crudely pasted on top of each other, and never duplicated across overlapping scripts). Immediately `request_changes` if a `level_map`, `mechanic_code`, "
                "or `integration` proof renders untextured flat polygons, bare programmer-art lines, duplicate overlapping structures, or primitive `draw_circle`/`draw_rect` stickmen/boxes. "
                "When a FULL-RESOLUTION 1:1 KEYFRAME is attached alongside a filmstrip, closely inspect every HUD icon, portrait frame, world sprite, and shader surface at 1:1 resolution for flat 16-color geometric placeholder polygons, "
                "squished/anamorphic or corner-clipped sprite slices, or unblended tile grid seams—if a broken `.png` asset or `.gd` consumer lies outside `target_modules`, emit atomic in-place `visual_asset` / `mechanic_code` proposals in `spawned_prerequisite_tasks` so it is repaired rather than ignored! "
                "IMPORTANT: In `plate_*.png` and `anim_sheet_*.png` / `anim_rig_*.png`, the dark/light square checkerboard grids, cyan bounding-box borders, and horizontal yellow/gold "
                "baseline ruler lines at the bottom of frames are external QA test-harness overlays drawn behind/around the transparent PNG—NEVER flag the QA checkerboard, cyan border, "
                "or yellow baseline guide line as a sprite artifact! Likewise, do NOT flag the external QA contact-sheet header/footer strips ('[QA] tick ...', '[QA] pos=...') outside each 16:9 frame as in-game text.\n"
            )
        if role in ("GameplayDesigner", "GameplaySystemsEngineer", "LevelAndProgressionDesigner", "CreativeDirector"):
            return (
                f"ROLE MANDATE ({role}): You have VETO power over gameplay authenticity, whole-game loop depth, unsolved t=0 baselines, and causal player agency tailored to the Target Player's Steam DNA. Immediately `request_changes` if:\n"
                "  1. A `mechanic_code`, `level_map`, or `integration` proof starts already pre-connected, pre-built, or pre-solved in Frame 1 (t=0..20) before any player input is applied (violating `test_unsolved_baseline_t0`), or reduces the game into a static pre-wired diorama or a single-solution 1-click lock-and-key tutorial instead of delivering the larger point of the game (`core_loop` across moment-to-moment, minute-to-minute, and session loops, `player_agency_and_optimization_depth`, and escalating `goal_hierarchy`).\n"
                "  2. State transitions across frames are not causally driven by player inputs (`[QA] tick ... in:[...]` or mouse clicks), or if an input action / setup call teleports or forces the world/onboarding step directly to a solved state by bypassing intermediate mechanics or prerequisites.\n"
                "  3. A multi-screen world (`world_bounds` larger than the 1280x720 viewport) lacks working camera navigation/panning or leaves core player verbs and mid/long-term progression from the Game Bible unimplemented.\n"
                "  4. A worker attempts to 'cheat' a failed playtest by inflating collision/catch radii into giant magnets, pre-spawning player-built routing/solution structures in `_ready()`, or moving the goal/receiver into the un-deflected/un-routed default trajectory of an emitter so the objective completes without genuine player construction.\n"
                "ABSOLUTE BAN ON COUNCIL-ORDERED SHORTCUTS & WRAPPER-SCRIPT AUTOMATION: When a QA playtest fails to complete a gameplay chain or launch/route an entity to the goal, NEVER write `feedback` telling workers to move the goal/receiver into an un-interacted path, widen hitboxes, OR call spawn/placement/mutation functions inside `_physics_process()` / `_process()` of `tests/test_<item_id>.gd`! The engine enforces a Counterfactual Zero-Input Gate (`input_script=[]`) AND an Entrypoint Parity Gate (replaying the exact same `input_segments` directly against live `res://main.tscn` without `tests/test_*.gd`), so scripting gameplay actions inside `tests/test_*.gd` immediately fails verification. Instead, inspect `qa_test_script_source`, `[QA] tick ... in:[...]` inputs, and `[QA] pos=...` telemetry: if `tests/test_<item_id>.gd` is mutating game state or bypassing `main.tscn` inputs, `request_changes` to remove that wrapper cheating; otherwise diagnose whether (a) a signal/input/entity wire is missing in the production game scripts (`target_modules` / `res://main.tscn`), or (b) `qa_test_engineer`'s external `input_segments` (`mouse_world_pos`, `mouse_click_pos`, tool/tuning keys) missed a required player step—and direct your `feedback` at fixing that exact root cause. Likewise, treat any `(x, y)` coordinate in QA telemetry strictly as diagnostic evidence of where a world-wide rule broke—never instruct workers to whitelist a specific `(x, y)` coordinate/rectangle or hardcode aiming/spawning toward a fixed coordinate in production scripts; always require fixing the general spatial/gameplay contract across all valid world regions.\n"
                "Use the external QA contact-sheet header/footer strips ('[QA] tick ... in:[...]', '[QA] pos=(...) ...') above and below each 16:9 frame "
                "as ground-truth engine telemetry to verify coordinates, inputs, and state changes—never flag those external QA margin strips (or the external checkerboard, cyan border, and yellow baseline guide line in `plate_*.png` / `anim_sheet_*.png`) as defects.\n"
            )
        return ""

    def _build_shared_review_parts(
        self,
        review_items: List[Any],
        game_bible: GameBibleSpec,
        candidate_dir: Optional[Path],
        *,
        include_concept_mockup: bool,
        frozen_bible_snapshot: Optional[str] = None,
    ) -> List[Any]:
        """Builds the byte-0 invariant text + multimodal prefix shared across all Council panelists in a review round.
        Places the large contiguous UTF-8 text block (`shared_review_context`) strictly FIRST at `parts[0]` ahead of
        binary proof/mockup `Part`s to maximize Gemini implicit prefix caching across back-to-back panelist calls."""
        from engine.studio.context_curator import ContextCuratorAgent, is_real_genai_client

        media_parts: List[Any] = []
        if include_concept_mockup:
            media_parts.extend(self._concept_mockup_parts(game_bible, candidate_dir))
        seen_item_ids: set[str] = set()
        for it in review_items:
            iid = str(getattr(it, "item_id", "") or "")
            if iid and iid in seen_item_ids:
                continue
            if iid:
                seen_item_ids.add(iid)
            media_parts.extend(self._proof_parts(it, candidate_dir))

        on_disk_sprites_block = ""
        curated_review_block = ""
        if candidate_dir is not None:
            try:
                from engine.studio.workers import _summarize_cartridge_for_qa

                qa_sum = _summarize_cartridge_for_qa(Path(candidate_dir), max_chars=2500)
                if qa_sum:
                    on_disk_sprites_block = (
                        "CARTRIDGE MODULES, BOUND ASSETS & ON-DISK ASSET LIBRARY (`bound_assets` shows which `res://assets/...` files each `.gd` script actually loads):\n"
                        f"{qa_sum}\n\n"
                    )
            except Exception:
                pass
            if is_real_genai_client(self.client):
                first_item = review_items[0] if review_items else None
                review_req = "\n".join(
                    f"[{it.item_id}] ({it.kind}) {it.title}: {it.spec} | proof: {it.proof_spec}"
                    for it in review_items
                )
                curator = ContextCuratorAgent(
                    cartridge_dir=Path(candidate_dir),
                    client=self.client,
                    model_name=self.model_name,
                    board=self.board,
                    game_bible=game_bible,
                    event_callback=self.event_callback,
                )
                pack = curator.curate_context(
                    target_agent_role="council_reviewer",
                    task_request=review_req,
                    item_id=(first_item.item_id if first_item is not None else "review_batch"),
                    item_kind=(first_item.kind if first_item is not None else "review"),
                )
                curated_review_block = f"{pack.prompt_block}\n"

        board_review_snapshot = ""
        if self.board is not None and hasattr(self.board, "snapshot_text"):
            try:
                board_review_snapshot = (
                    "FULL PRODUCTION BOARD STATE (inspect all active, changes_requested, blocked, and approved tasks and their `mods` before attributing defects):\n"
                    f"{self.board.snapshot_text(limit_chars=4500)}\n\n"
                )
            except Exception:
                pass
        bible_snap_str = (
            frozen_bible_snapshot
            if frozen_bible_snapshot is not None
            else self._format_bible_snapshot(game_bible, compact_for_review=True)
        )
        cross_task_triage_guide = (
            "CROSS-TASK ROOT-CAUSE ATTRIBUTION & SCOPE ISOLATION (PREVENTING MULTI-TASK PING-PONG):\n"
            "- Before writing `feedback` on an item, check `FULL PRODUCTION BOARD STATE` above to see whether any visual or behavioral defect visible in the proof belongs to THIS item's core responsibility (`target_modules`) OR to a shared/external subsystem (e.g. a shared camera controller, world backdrop/scene generator, or hub script) owned by another task.\n"
            "- NEVER instruct multiple downstream mechanic/feature tasks to directly patch the same shared camera, scene, or backdrop script in parallel (which causes workers to overwrite each other and burn revision attempts)!\n"
            "- Instead, when a proof fails partly or wholly because of a defect in a shared/external module outside this item's primary module:\n"
            "  1. If an unapproved task on the board (`in_progress`, `changes_requested`, `ready`, or `blocked`) ALREADY owns that shared module, add its `item_id` to `blocked_by_item_ids`, list the shared module path(s) in `out_of_scope_modules_to_prune`, and write `feedback` telling this item what remains in its own module once the blocker finishes.\n"
            "  2. If the task that originally built that shared module is already `approved` (or no open task owns fixing the shared bug), populate `spawned_prerequisite_tasks` with a focused `WorkItemProposal` targeting that shared module (and set `blocks_items` to include all affected open tasks on the board), and list the shared module path(s) in `out_of_scope_modules_to_prune` so this item is cleanly blocked on the dedicated fix without losing a revision attempt.\n\n"
        )
        shared_review_context = (
            f"{STUDIO_CRITICAL_DISCOURSE_CHARTER}\n\n"
            f"{cross_task_triage_guide}"
            f"{board_review_snapshot}"
            f"{on_disk_sprites_block}"
            f"{curated_review_block}"
            f"{bible_snap_str}\n"
        )
        return [shared_review_context] + media_parts

    def _review_as(
        self,
        role: str,
        awaiting: List[Any],
        deciding: List[Any],
        game_bible: GameBibleSpec,
        dna_summary: str,
        candidate_dir: Optional[Path],
        prior_roundtable_notes: Optional[List[str]] = None,
        frozen_bible_snapshot: Optional[str] = None,
        shared_round_parts: Optional[List[Any]] = None,
        *,
        _eager_explicit_cache: bool = False,
    ) -> ReviewBatch:
        from engine.studio.context_curator import is_real_genai_client

        spec = next((sp for sp in SPECIALIST_ROSTER if sp["role"] == role), None)
        who = f"the {spec['display_name']} ({role})" if spec else "the Creative Director"
        if shared_round_parts is not None:
            parts = list(shared_round_parts)
        else:
            include_mockup = role == "VisualArtDirector" or any(
                it.kind in ("level_map", "ui", "integration", "visual_asset", "animation")
                for it in awaiting + deciding
            )
            parts = self._build_shared_review_parts(
                awaiting + deciding,
                game_bible,
                candidate_dir,
                include_concept_mockup=include_mockup,
                frozen_bible_snapshot=frozen_bible_snapshot,
            )
        awaiting_ids = {i.item_id for i in awaiting}
        deciding_ids = {i.item_id for i in deciding}
        by_id = {i.item_id: i for i in awaiting + deciding}
        mandate = self._role_review_mandate(role)
        filmstrip_guide = (
            "HOW TO READ `filmstrip_*.png` QA CONTACT SHEETS & STRATEGIZE YOUR SPECIALIST VALIDATION:\n"
            "- STEP 1: DETERMINE WHAT MUST BE VALIDATED FOR YOUR ROLE (`validation_criteria_for_my_role`): Do NOT rubber-stamp items or follow a generic checklist. For each item, first determine from first principles what MUST be proven for this specific change to be genuinely playable, responsive, and cohesive from your specialist perspective.\n"
            "- STEP 2: INSPECT THE FILMSTRIP, TELEMETRY & HUMAN-SCALE RESPONSIVENESS:\n"
            "  * Above each 16:9 game frame, `[QA] tick N  in:[...]` shows the simulation tick (60 ticks ~ 1 second) and injected player inputs. Below each frame, `[QA] pos=(...) ...` shows the live state snapshot.\n"
            "  * These external margin strips above/below the 16:9 border are QA contact-sheet headers/footers—never reject those external margin strips (or the external checkerboard/cyan border/yellow baseline in `plate_*.png`).\n"
            "  * DEBUG OVERLAY POLICY (REQUIRED IN TEST SCENES, BANNED IN FINAL PRODUCTION GAME): In isolated or staged QA test scenes (`res://tests/test_<item_id>.tscn`), rich visual debug overlays (velocity/force vectors, target markers, state readouts) ARE expected and necessary to prove physics and state changes—if a proof is ambiguous without a visual vector/marker overlay, use `request_changes` to ask QA for that overlay in `res://tests/test_<item_id>.tscn`. Conversely, the final production game (`res://main.tscn`, `main.gd`, `ui_hud.gd`) must NEVER render raw developer debug text (`[DEBUG]`, telemetry dumps) on the player's HUD!\n"
            "  * HUMAN-SCALE PLAYABILITY & SUBSYSTEM HARMONY: Check the rate of change across ticks (`60 ticks = 1.0s`). If a player holds a movement, steering, rotation, or interaction input for 60–180 ticks (1–3 seconds) and the entity barely turns or moves (e.g., 1 degree per second because a damping/stabilizer subsystem is fighting the player's input, or `[idle]` with zero inputs), you MUST `request_changes`!\n"
            "  * CAUSAL-CHAIN ROOT-CAUSE DIAGNOSIS (NEVER ORDER GOALPOST SHIFTS, SHORTCUTS, OR `_physics_process` WRAPPER CALLS): Every QA capture must pass the Counterfactual Zero-Input Gate (`input_script=[]` produces zero progress) AND the Entrypoint Parity Gate (the external `input_segments` satisfy the exact same state assertions inside live `res://main.tscn` without `tests/test_<item_id>.gd`). NEVER instruct workers in `feedback` to invoke spawn/placement/action methods from `_physics_process()` inside `tests/test_<item_id>.gd`, and NEVER ask workers to move a goal/receiver into an un-deflected default trajectory or inflate a catch radius! Instead, read `[QA] tick N in:[...]` and `[QA] pos=...` to see WHICH link in the causal chain broke: require fixing missing signal/input/entity connections in the production `.gd` modules (`target_modules` / `res://main.tscn`) and driving real player inputs externally via `qa_test_engineer`'s `input_segments` (`mouse_world_pos`, `mouse_click_pos`, action keys), keeping `tests/test_<item_id>.gd` strictly limited to scene instantiation, read-only camera/diagnostic `_draw()` overlays, and `get_state_snapshot()` readouts!\n"
            "  * SYSTEMIC INVARIANTS OVER COORDINATE POINT-PATCHES: Treat any `(x, y)` coordinate, tick, or angle in `[QA]` telemetry strictly as a diagnostic sample showing WHERE a general game rule broke—NEVER as the boundary of the fix! Never write `feedback` (or `replacement_spec` / `spawned_prerequisite_tasks`) instructing production code to whitelist a specific `(x, y)` coordinate/bounding box or hardcode aiming/targeting toward a fixed coordinate; instead, diagnose the underlying systemic mismatch (e.g., UI/cursor validation disagreeing with authoritative world geometry across regions, or entity targeting/orientation not driven by player controls/state) so the mechanic works everywhere in the world.\n"
            "  * VISUAL & UI INTEGRITY: Inspect every frame from Frame 1 to the final frame. If active subjects drift off-screen, if world indicators are detached from their sprites, or if HUD panels/labels overlap or collide with each other, you MUST `request_changes`!\n\n"
        )
        eff_rev_dir = self._resolve_memory_dir(candidate_dir)
        rev_mem_block = ""
        if (
            eff_rev_dir is not None
            and not is_real_genai_client(self.client)
            and (eff_rev_dir / "game_memory.json").is_file()
        ):
            from engine.studio.game_memory import GameMemoryStore

            rev_mem_block = GameMemoryStore(
                eff_rev_dir, event_callback=getattr(self, "event_callback", None)
            ).format_role_memory_briefing(role)
        roundtable_block = ""
        if prior_roundtable_notes:
            roundtable_block = (
                "EARLIER COUNCIL PANELIST OBSERVATIONS IN THIS REVIEW ROUND (DO NOT open `spoken_dialogue` with 'I completely agree with...', 'I echo...', or peer roll-call flattery, and DO NOT repeat points already stated below—audit the proof independently from your own domain, push back on any premature peer approval if you spot a real defect, and push back on out-of-scope perfectionism if the item already satisfies its slice scope):\n"
                + "\n".join(f"  - {note}" for note in prior_roundtable_notes)
                + "\n\n"
            )
        prompt = (
            f"You are {who} reviewing work built by the implementation team for the game being designed.\n"
            f"{mandate}"
            f"{filmstrip_guide}"
            f"{rev_mem_block}"
            f"{roundtable_block}"
            f"ITEMS AWAITING YOUR REVIEW (proof images and Living Game Bible context are attached above):\n"
            + ("\n".join(self._format_item_for_review(i, candidate_dir=candidate_dir) for i in awaiting) or "  (none)")
            + f"\n\nITEMS THAT HIT THEIR REVISION CAP OR STALLED (you must decide):\n"
            + ("\n".join(self._format_item_for_review(i, candidate_dir=candidate_dir) for i in deciding) or "  (none)")
            + "\n\nIn `spoken_dialogue` and `addressed_to`, speak candidly and skeptically to your fellow Council reviewers (zero agreement preambles or cheerleading; full respect for any Executive Producer direction) about what you actually observe in the HIGH-resolution proof frames and telemetry versus `UNVERIFIED_WORKER_CLAIMS`. "
            "For EVERY awaiting item give one `review_decisions` entry: first state `validation_criteria_for_my_role`, verify the rendered proof skeptically against those criteria, the item's `spec`, `proof_spec`, and Executive Producer direction, then: "
            "(a) choose `request_changes` with concrete, surgical feedback when the item fails its own `spec`/`proof_spec`, exhibits real visual/UI craft glitches (cut-off/bleeding sprite slicing, squished icons, overflowing/overlapping HUD text, placeholder geometry), or breaks causal responsiveness (use `blocked_by_item_ids`, `spawned_prerequisite_tasks`, and `out_of_scope_modules_to_prune` when a defect belongs to a shared/external module); OR "
            "(b) choose `approve` once THIS item genuinely fulfills its own active-slice `spec` and visual/UX quality bar—do NOT invent out-of-scope feature creep or try to redesign the whole game inside one task's review! "
            "For EVERY capped/stalled item give one `cap_decisions` entry: `accept_as_is` (only if the result genuinely "
            "serves the active slice; state candidly what remains rough for future cleanup), `rewrite_spec` (populate a non-empty `replacement_spec` BEFORE setting `action='rewrite_spec'`; max "
            f"{MAX_SPEC_RESETS} rewrites), `split` (populate a non-empty `split_items` list of `WorkItemProposal` objects BEFORE setting `action='split'` — NEVER set `action='split'` if `split_items` is empty), or `cancel`. Provide `deliberation` FIRST and each verdict/action last."
        )

        def _validate(batch: ReviewBatch) -> List[str]:
            issues: List[str] = []
            seen_r = [d.item_id for d in batch.review_decisions]
            for iid in awaiting_ids - set(seen_r):
                issues.append(f"Missing review decision for item '{iid}'.")
            for iid in set(seen_r) - awaiting_ids:
                issues.append(f"review_decisions references '{iid}', which is not awaiting your review.")
            if len(seen_r) != len(set(seen_r)):
                issues.append("Duplicate review decisions for the same item.")
            for d in batch.review_decisions:
                if d.verdict == "request_changes" and not d.feedback.strip():
                    issues.append(f"request_changes for '{d.item_id}' needs concrete feedback.")
                if d.spawned_prerequisite_tasks:
                    issues.extend(self._unsupported_kind_issues(d.spawned_prerequisite_tasks))
                    for sp in d.spawned_prerequisite_tasks:
                        if not sp.item_id.strip():
                            issues.append(f"spawned_prerequisite_tasks for '{d.item_id}' has an empty item_id.")
                        if sp.item_id == d.item_id:
                            issues.append(f"spawned_prerequisite_tasks for '{d.item_id}' cannot use the reviewed item's own item_id.")
            seen_c = [c.item_id for c in batch.cap_decisions]
            for iid in deciding_ids - set(seen_c):
                issues.append(f"Missing cap decision for item '{iid}'.")
            for iid in set(seen_c) - deciding_ids:
                issues.append(f"cap_decisions references '{iid}', which is not awaiting a decision.")
            for c in batch.cap_decisions:
                item = by_id.get(c.item_id)
                if item is None:
                    continue
                if c.action == "split" and not c.split_items and c.replacement_spec.strip() and item.spec_resets < MAX_SPEC_RESETS:
                    c.action = "rewrite_spec"
                if c.action == "rewrite_spec" and not c.replacement_spec.strip() and c.deliberation.strip() and item.spec_resets < MAX_SPEC_RESETS:
                    c.replacement_spec = f"{item.spec}\n\nREVISED SPECIFICATION DIRECTIVE: {c.deliberation.strip()}"
                if c.action == "rewrite_spec":
                    if not c.replacement_spec.strip():
                        issues.append(f"rewrite_spec for '{c.item_id}' needs a replacement_spec.")
                    if item.spec_resets >= MAX_SPEC_RESETS:
                        issues.append(f"'{c.item_id}' has used all {MAX_SPEC_RESETS} spec rewrites; choose another action.")
                if c.action == "split":
                    if not c.split_items:
                        issues.append(
                            f"split for '{c.item_id}' has empty split_items=[]. Because split_items appears BEFORE action in JSON schema order, "
                            f"you MUST either populate split_items with WorkItemProposal objects BEFORE emitting action='split', OR choose action='rewrite_spec' (with replacement_spec) or action='accept_as_is'."
                        )
                    else:
                        issues.extend(self.board.validate_proposals(c.split_items))
                        issues.extend(self._unsupported_kind_issues(c.split_items))
            return issues

        return self._structured_with_validation(
            prompt,
            ReviewBatch,
            _validate,
            temperature=0.3,
            parts=parts,
            caller_role=role,
            candidate_dir=eff_rev_dir,
            min_calls_before_create=1 if _eager_explicit_cache else 2,
        )

    def _run_review_round(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        candidate_dir: Optional[Path],
    ) -> int:
        """Shows every finished item's proof to its Council review panel, requires unanimous approval across panelists,
        and applies consolidated decisions to the board. Returns number of item decisions applied."""
        if self.board is None:
            return 0
        groups: Dict[str, Tuple[List[Any], List[Any]]] = {}
        awaiting_items = self.board.awaiting_review()
        for it in awaiting_items:
            for r in self._reviewers_for(it):
                groups.setdefault(r, ([], []))[0].append(it)
        for it in self.board.needing_decision():
            groups.setdefault(self._reviewer_for(it), ([], []))[1].append(it)
        if not groups:
            return 0
        roles = list(groups)
        self._emit_activity(
            agent_role="CouncilReviewPanel",
            status_text=(
                f"Council review panel ({', '.join(roles)}) is inspecting proof images for "
                f"{[it.item_id for it in awaiting_items] or [it.item_id for it in self.board.needing_decision()]}..."
            ),
            active_agents=roles,
            phase="proof_review",
        )
        batches: List[ReviewBatch] = []
        prior_roundtable_notes: List[str] = []
        all_reviewed_items = list(awaiting_items) + list(self.board.needing_decision())
        round_proof_paths: List[str] = []
        for it in all_reviewed_items:
            if getattr(it, "revisions", None):
                for pp in (it.revisions[-1].proof_paths or []):
                    if pp and str(pp) not in round_proof_paths:
                        round_proof_paths.append(str(pp))

        round_bible_snapshot = self._format_bible_snapshot(game_bible, compact_for_review=True)
        include_round_mockup = ("VisualArtDirector" in roles) or any(
            it.kind in ("level_map", "ui", "integration", "visual_asset", "animation")
            for it in all_reviewed_items
        )
        shared_round_parts = self._build_shared_review_parts(
            all_reviewed_items,
            game_bible,
            candidate_dir,
            include_concept_mockup=include_round_mockup,
            frozen_bible_snapshot=round_bible_snapshot,
        )
        round_cache_cm = explicit_prefix_cache_scope(self.client, ttl_seconds=300, reuse_outer=False)
        round_cache_cm.__enter__()
        try:
            for idx_r, r in enumerate(roles):
                batch = self._review_as(
                    r,
                    groups[r][0],
                    groups[r][1],
                    game_bible,
                    dna_summary,
                    candidate_dir,
                    prior_roundtable_notes=prior_roundtable_notes,
                    frozen_bible_snapshot=round_bible_snapshot,
                    shared_round_parts=shared_round_parts,
                    _eager_explicit_cache=(len(roles) >= 2),
                )
                batches.append(batch)
                spec = next((sp for sp in SPECIALIST_ROSTER if sp["role"] == r), None)
                disp_name = spec["display_name"] if spec else r
                verdict_bits = [
                    f"`{d.item_id}` -> **{d.verdict}**"
                    + (f" (criteria: {d.validation_criteria_for_my_role})" if getattr(d, "validation_criteria_for_my_role", "") else "")
                    + (f": {d.feedback}" if d.feedback else "")
                    for d in batch.review_decisions
                ]
                spoken_text = (batch.spoken_dialogue or "").strip()
                if not spoken_text:
                    spoken_text = f"[Council Proof Review] {'; '.join(verdict_bits) if verdict_bits else batch.deliberation}"
                prior_roundtable_notes.append(f"[{disp_name} ({r})]: {spoken_text} | Verdicts: {'; '.join(verdict_bits)}")

                turn_idx = len(game_bible.writers_room_transcript)
                other_roles = [orole for orole in roles if orole != r] or ["CreativeDirector"]
                addressed = [a for a in (batch.addressed_to or []) if a] or other_roles
                updates_summary = [
                    f"review:{d.item_id}:{d.verdict}" for d in batch.review_decisions
                ] + [
                    f"cap:{c.item_id}:{c.action}" for c in batch.cap_decisions
                ]
                attach_proofs = round_proof_paths if idx_r == 0 else []
                art_emitted = (
                    {
                        "type": "filmstrip",
                        "path": round_proof_paths[0],
                        "item_ids": [it.item_id for it in all_reviewed_items],
                        "caption": f"Proof under Council Review: {', '.join(it.item_id for it in all_reviewed_items)}",
                    }
                    if idx_r == 0 and round_proof_paths
                    else None
                )
                rev_utt = WritersRoomUtterance(
                    deliberation=batch.deliberation,
                    utterance_id=f"utt_{turn_idx:02d}_{r}_review",
                    turn_index=turn_idx,
                    speaker_role=r,
                    speaker_display_name=disp_name,
                    spoken_dialogue=spoken_text,
                    addressed_to=addressed,
                    bible_updates_summary=updates_summary,
                    proof_image_paths=attach_proofs,
                    artifact_emitted=art_emitted,
                )
                game_bible.writers_room_transcript.append(rev_utt)
                self._emit_event(
                    "writers_room_utterance",
                    {
                        "deliberation": rev_utt.deliberation,
                        "utterance_id": rev_utt.utterance_id,
                        "turn_index": rev_utt.turn_index,
                        "speaker_role": rev_utt.speaker_role,
                        "speaker_display_name": rev_utt.speaker_display_name,
                        "spoken_dialogue": rev_utt.spoken_dialogue,
                        "addressed_to": rev_utt.addressed_to,
                        "bible_updates_summary": rev_utt.bible_updates_summary,
                        "proof_image_paths": rev_utt.proof_image_paths,
                        "artifact_emitted": rev_utt.artifact_emitted,
                    },
                )
        finally:
            round_cache_cm.__exit__(None, None, None)
        decisions_by_item: Dict[str, List[Tuple[str, ReviewDecision]]] = {}
        applied = 0
        for role, batch in zip(roles, batches):
            for d in batch.review_decisions:
                decisions_by_item.setdefault(d.item_id, []).append((role, d))
            for c in batch.cap_decisions:
                self.board.apply_cap_decision(c)
                applied += 1
            self._emit_event(
                "review_round_completed",
                {
                    "deliberation": batch.deliberation,
                    "reviewer_role": role,
                    "approved": [d.item_id for d in batch.review_decisions if d.verdict == "approve"],
                    "changes_requested": [d.item_id for d in batch.review_decisions if d.verdict == "request_changes"],
                    "cap_decisions": [(c.item_id, c.action) for c in batch.cap_decisions],
                },
            )
        for it in awaiting_items:
            panel_votes = decisions_by_item.get(it.item_id, [])
            if not panel_votes:
                continue
            dissenters = [(r, d) for r, d in panel_votes if d.verdict == "request_changes"]
            combined_delib = " | ".join(f"[{r}] {d.deliberation}" for r, d in panel_votes)
            if not dissenters:
                consolidated = ReviewDecision(
                    deliberation=combined_delib,
                    item_id=it.item_id,
                    feedback="",
                    verdict="approve",
                )
            else:
                distinct_fb: List[Tuple[str, str]] = []
                seen_fb: set = set()
                merged_blockers: List[str] = []
                merged_spawned: List[WorkItemProposal] = []
                seen_spawn_keys: set = set()
                merged_pruned: List[str] = []
                for r, d in dissenters:
                    fb_clean = d.feedback.strip()
                    if fb_clean and fb_clean not in seen_fb:
                        seen_fb.add(fb_clean)
                        distinct_fb.append((r, fb_clean))
                    for b_id in getattr(d, "blocked_by_item_ids", None) or []:
                        if b_id and b_id not in merged_blockers:
                            merged_blockers.append(b_id)
                    for sp in getattr(d, "spawned_prerequisite_tasks", None) or []:
                        sp_key = (sp.item_id, tuple(sorted(sp.target_modules or [])))
                        if sp_key not in seen_spawn_keys:
                            seen_spawn_keys.add(sp_key)
                            merged_spawned.append(sp)
                    for pm in getattr(d, "out_of_scope_modules_to_prune", None) or []:
                        if pm and pm not in merged_pruned:
                            merged_pruned.append(pm)
                if len(distinct_fb) == 1:
                    merged_fb = distinct_fb[0][1]
                else:
                    merged_fb = " | ".join(f"[{r}]: {fb}" for r, fb in distinct_fb)
                consolidated = ReviewDecision(
                    deliberation=combined_delib,
                    item_id=it.item_id,
                    feedback=merged_fb,
                    blocked_by_item_ids=merged_blockers,
                    spawned_prerequisite_tasks=merged_spawned,
                    out_of_scope_modules_to_prune=merged_pruned,
                    verdict="request_changes",
                )
            self.board.apply_review(consolidated)
            applied += 1
        return applied

    def _handle_failed_items(self) -> bool:
        """Re-queues a failed item once (visible via events/logs); on a second failure in a live Council build,
        escalates the item to `needs_decision` so the Council can rewrite its spec or cancel/re-point dependents,
        or raises the authentic error if spec rewrites are exhausted or in direct unit-test failure probes (Rule 5).
        Returns True if any failed item was re-queued or escalated."""
        if self.board is None:
            return False
        handled = False
        for it in self.board.items():
            if it.state != "failed":
                continue
            if it.failure_count >= 2:
                if is_real_genai_client(self.client) and it.failure_count <= (MAX_SPEC_RESETS + 2):
                    logger.warning(
                        "Escalating item %s to Council needs_decision after %d worker failures:\n%s",
                        it.item_id,
                        it.failure_count,
                        it.failure,
                    )
                    self.board.escalate_failed_to_decision(it.item_id)
                    handled = True
                    continue
                raise StudioAgentError(f"Backlog item '{it.item_id}' failed {it.failure_count} times:\n{it.failure}")
            logger.warning("Re-queueing failed item %s (failure %d):\n%s", it.item_id, it.failure_count, it.failure)
            self.board.requeue_failed(it.item_id)
            handled = True
        if handled and self.scheduler is not None:
            self.scheduler.start()
        return handled

    def _resolve_blocked_by_cancelled(self) -> bool:
        """Unblocks items stranded solely because a Council decision explicitly cancelled one of their `depends_on`
        items, pruning the cancelled dependency IDs and annotating the dependent's spec so the build never deadlocks."""
        if self.board is None:
            return False
        stranded = self.board.blocked_by_cancelled()
        if not stranded:
            return False
        all_by_id = {i.item_id: i for i in self.board.items()}
        resolved = False
        for it in stranded:
            cancelled_deps = [d for d in it.depends_on if d in all_by_id and all_by_id[d].state == "cancelled"]
            if not cancelled_deps:
                continue
            remaining_deps = [d for d in it.depends_on if d not in cancelled_deps]
            reasons = "; ".join(
                f"{d} ({all_by_id[d].decision_reason or 'cancelled by Council'})" for d in cancelled_deps
            )
            note = (
                f"[Council Dependency Update]: Prerequisite item(s) {cancelled_deps} were cancelled ({reasons}); "
                "implement this item cleanly without relying on cancelled prerequisites."
            )
            updated_spec = f"{it.spec.rstrip()}\n\n{note}" if note not in it.spec else it.spec
            self.board.update_item(it.item_id, spec=updated_spec, depends_on=remaining_deps)
            resolved = True
        if resolved and self.scheduler is not None:
            self.scheduler.start()
        return resolved

    def _is_paused(self, candidate_dir: Optional[Path] = None) -> bool:
        if self.scheduler is not None and hasattr(self.scheduler, "is_paused") and self.scheduler.is_paused():
            return True
        if candidate_dir is not None:
            pause_file = candidate_dir.parent / ".studio_paused.json"
            if pause_file.is_file():
                try:
                    data = json.loads(pause_file.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and data.get("paused") is True:
                        return True
                except Exception:
                    pass
        return False

    def _wait_if_paused(self, candidate_dir: Optional[Path] = None) -> float:
        """Waits while operator pause is active and returns elapsed pause duration in seconds."""
        self._check_aborted(candidate_dir)
        t0 = time.monotonic()
        while self._is_paused(candidate_dir):
            self._check_aborted(candidate_dir)
            time.sleep(0.25)
        self._check_aborted(candidate_dir)
        return max(0.0, time.monotonic() - t0)

    def _run_council_diagnostic_probe(
        self,
        candidate_dir: Path,
        diag_probe: CouncilDiagnosticProbe,
        title_hint: str,
    ) -> Tuple[str, List[Any]]:
        """Executes a read-only/diagnostic Council probe via ANY of the 5 Specialist Personas
        (`visual_artist`, `audio_designer`, `script_writer`, `swe_engineer`, `qa_test_engineer`),
        following up to 2 cycle-safe peer re-route hops if the probed specialist realizes another
        domain specialist is better suited to inspect."""
        from engine.studio.models import WorkItem
        from engine.studio.tool_registry import CANONICAL_SPECIALIST_PERSONAS, normalize_specialist_persona
        from engine.studio.workers import (
            WorkerContext,
            run_audio_designer_specialist,
            run_qa_test_engineer_specialist,
            run_script_writer_specialist,
            run_swe_engineer_specialist,
            run_visual_artist_specialist,
        )

        initial_brief = diag_probe.effective_brief
        if not initial_brief:
            return "", []
        current_persona = normalize_specialist_persona(diag_probe.specialist_agent or "qa_test_engineer")
        current_brief = initial_brief
        current_targets: Optional[List[str]] = list(diag_probe.target_paths) if diag_probe.target_paths else None

        swe_agent_ref = None
        if self.scheduler is not None:
            for h_fn in (getattr(self.scheduler, "handlers", None) or {}).values():
                w_ctx = getattr(h_fn, "_worker_ctx", None)
                if w_ctx is not None and getattr(w_ctx, "swe_agent", None) is not None:
                    swe_agent_ref = w_ctx.swe_agent
                    break

        diag_ctx = WorkerContext(
            cartridge_dir=Path(candidate_dir),
            client=self.client,
            model_name=self.model_name,
            swe_agent=swe_agent_ref,
            board=self.board,
            event_callback=self.event_callback,
        )

        max_probe_hops = 2
        visited_personas: set[str] = set()
        probe_sections: List[str] = []
        probe_parts: List[Any] = []

        for hop_idx in range(max_probe_hops + 1):
            visited_personas.add(current_persona)
            if current_persona == "swe_engineer" and diag_ctx.swe_agent is None:
                from engine.studio.swe import LeadGameplaySWEAgent

                diag_ctx.swe_agent = LeadGameplaySWEAgent(
                    client=self.client,
                    model_name=self.model_name,
                )

            self._emit_activity(
                agent_role="CouncilTable",
                status_text=(
                    f"Running Council diagnostic probe via `{current_persona}`"
                    + (f" (reroute hop #{hop_idx})" if hop_idx > 0 else "")
                    + f" to investigate '{title_hint[:70]}'..."
                ),
                active_agents=["CouncilTable", current_persona],
                phase="steering_diagnostic",
            )
            diag_item = WorkItem(
                deliberation=diag_probe.deliberation,
                item_id=f"council_diag_{hop_idx}",
                kind="discovery",
                title=f"Diagnostic investigation: {title_hint[:80]}",
                spec=current_brief,
                proof_spec=current_brief,
                target_modules=list(current_targets or ["main.gd"]),
            )
            if current_persona == "visual_artist":
                diag_report = run_visual_artist_specialist(
                    diag_ctx,
                    diag_item,
                    brief=current_brief,
                    target_paths=current_targets,
                    _read_only_probe=True,
                )
            elif current_persona == "audio_designer":
                diag_report = run_audio_designer_specialist(
                    diag_ctx,
                    diag_item,
                    brief=current_brief,
                    target_paths=current_targets,
                    _force_operation="inspect",
                )
            elif current_persona == "script_writer":
                diag_report = run_script_writer_specialist(
                    diag_ctx,
                    diag_item,
                    brief=current_brief,
                    target_paths=current_targets,
                    _force_operation="inspect_only",
                )
            elif current_persona == "swe_engineer":
                diag_report = run_swe_engineer_specialist(
                    diag_ctx,
                    diag_item,
                    brief=current_brief,
                    target_paths=current_targets,
                    _inspect_only=True,
                )
            else:
                diag_report = run_qa_test_engineer_specialist(
                    diag_ctx,
                    diag_item,
                    brief=f"[COUNCIL DIAGNOSTIC TEST SCENE REQUEST]:\n{current_brief}",
                    target_paths=current_targets,
                )

            for p_img in (diag_report.proof_images or [])[-2:]:
                p_path = Path(p_img)
                if p_path.is_file() and p_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    img_part = self._load_image_part(p_path)
                    if img_part is not None:
                        probe_parts.append(f"COUNCIL DIAGNOSTIC PROBE PLATE ({current_persona} — {p_path.name}):")
                        probe_parts.append(img_part)

            probe_sections.append(
                f"--- Probe Leg #{hop_idx + 1} (`{current_persona}`) ---\n"
                f"Specialist Summary & Findings: {diag_report.summary}\n"
                f"Diagnostic Proof Paths: {diag_report.proof_images}"
            )

            raw_reroute = str(getattr(diag_report, "reroute_to_specialist", "") or "").strip()
            if not raw_reroute or hop_idx >= max_probe_hops:
                break
            next_persona = normalize_specialist_persona(raw_reroute)
            if next_persona not in CANONICAL_SPECIALIST_PERSONAS or next_persona in visited_personas:
                break
            next_brief = str(getattr(diag_report, "reroute_brief", "") or "").strip() or current_brief
            next_targets = list(getattr(diag_report, "reroute_target_paths", None) or current_targets or []) or None
            self._emit_event(
                "council_diagnostic_probe_rerouted",
                {
                    "deliberation": diag_report.deliberation,
                    "reasoning_rationale": diag_report.deliberation,
                    "from_specialist": current_persona,
                    "to_specialist": next_persona,
                    "reroute_brief": next_brief,
                    "hop": hop_idx + 1,
                },
            )
            current_persona = next_persona
            current_brief = next_brief
            current_targets = next_targets

        probe_block = (
            f"\n=== LIVE COUNCIL SPECIALIST DIAGNOSTIC PROBE ===\n"
            f"Council Diagnostic Hypothesis: {diag_probe.deliberation}\n"
            + "\n".join(probe_sections)
            + "\n"
        )
        return probe_block, probe_parts

    def _execute_steering_pivot_turns(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        candidate_dir: Optional[Path],
        steered_notes: List[str],
    ) -> None:
        """Routes Executive Producer steering / build diagnostics dynamically across the Council roster
        (and optional Specialist Persona diagnostic probes), allowing any speaker to hand the floor to
        another Council specialist via `route_to_council_role`."""
        from engine.studio.context_curator import (
            ContextCuratorAgent,
            compute_entrypoint_reachability,
        )

        combined_steering = " | ".join(steered_notes)
        board_was_drained = bool(self.board is not None and self.board.is_drained())
        wish_context_block, wish_multimodal_parts = self._consume_pending_playtest_wishes(candidate_dir)
        consumed_pt_sids = set(getattr(self, "_last_consumed_playtest_session_ids", None) or ())
        steering_summary_for_prompt = (
            " | ".join(
                self._compact_playtest_dispatch_text(
                    n,
                    note_suffix="full structured playtest report & screenshots attached in dossier",
                )
                for n in steered_notes
            )
            if wish_context_block
            else combined_steering
        )
        curated_code_block = wish_context_block
        use_dynamic_steering_routing = bool(
            is_real_genai_client(self.client) or getattr(self.client, "supports_steering_routing", False)
        )
        routing_plan: Optional[CouncilSteeringRoutingPlan] = None

        if candidate_dir is not None and Path(candidate_dir).is_dir() and use_dynamic_steering_routing:
            try:
                c_dir = Path(candidate_dir)
                reach = compute_entrypoint_reachability(c_dir)
                asset_dossier_txt, _, asset_sheet_part = self._build_asset_integrity_dossier_and_sheet(
                    c_dir, "steering"
                )
                if asset_sheet_part is not None:
                    wish_multimodal_parts = list(wish_multimodal_parts or []) + [
                        "CARTRIDGE VISUAL ASSET CONTACT SHEET FOR STEERING TRIAGE "
                        "(inspect active in-game sprites vs `[UNREF]` orphaned files alongside the report):",
                        asset_sheet_part,
                    ]

                recent_board_history = ""
                if self.board is not None and hasattr(self.board, "items"):
                    try:
                        recent_items = [
                            it for it in self.board.items()
                            if getattr(it, "revisions", None)
                        ][-5:]
                        if recent_items:
                            hist_lines = []
                            for r_it in recent_items:
                                last_rev = r_it.revisions[-1]
                                hist_lines.append(
                                    f"  - [{r_it.item_id}] ({r_it.state}, mods={r_it.target_modules}): {r_it.title}\n"
                                    f"    spec: {r_it.spec[:260]}\n"
                                    f"    latest_worker_notes: {last_rev.worker_notes[:320]}"
                                )
                            recent_board_history = (
                                "\nRECENTLY EXECUTED PRODUCTION BOARD TASKS & WHAT THEY ALREADY CHANGED:\n"
                                + "\n".join(hist_lines)
                                + "\n"
                            )
                    except Exception:
                        pass

                roster_summary = "\n".join(
                    f"  - `{sp['role']}` ({sp['display_name']}): {sp['focus']}"
                    for sp in SPECIALIST_ROSTER
                )
                routing_prompt = (
                    f"You are the Writers' Room Council Table routing an incoming Executive Producer Steering / Build Report:\n"
                    f"REPORT: {steering_summary_for_prompt}\n\n"
                    f"AVAILABLE COUNCIL ROLES:\n{roster_summary}\n\n"
                    f"{render_compact_tool_directory(layer='council')}\n\n"
                    f"{wish_context_block}\n\n"
                    f"ENTRYPOINT REACHABILITY:\n{reach.get('summary_text', '')}\n"
                    f"{recent_board_history}\n"
                    f"{asset_dossier_txt}\n"
                    f"ROUTING & PRIOR-HYPOTHESIS FALSIFICATION INSTRUCTIONS:\n"
                    f"1. Choose `lead_council_role` (the primary domain specialist in `AVAILABLE COUNCIL ROLES` best suited to diagnose and own this report) "
                    f"and up to 3 `supporting_council_roles` whose domains are also affected (do NOT blindly route visual art, UX, audio, or narrative issues solely to `GameplaySystemsEngineer`).\n"
                    f"2. Check `RECENTLY EXECUTED PRODUCTION BOARD TASKS`: if a prior task already attempted to fix this symptom and the issue persists, treat that prior hypothesis as FALSIFIED/INCOMPLETE.\n"
                    f"3. Decide whether static context + asset contact sheet above are ALREADY sufficient (`needs_specialist_probe = False`), OR whether one of the 5 Execution Specialist Personas "
                    f"(`visual_artist`, `audio_designer`, `script_writer`, `swe_engineer`, `qa_test_engineer`) should run a read-only diagnostic probe first (`needs_specialist_probe = True`, setting `specialist_probe_agent`, `diagnostic_brief`, and optional `target_paths`).\n"
                    f"Provide `deliberation` FIRST and `needs_specialist_probe` LAST."
                )
                routing_plan = self._structured_with_validation(
                    routing_prompt,
                    CouncilSteeringRoutingPlan,
                    lambda p: (
                        ["`lead_council_role` must match a valid role in `AVAILABLE COUNCIL ROLES`."]
                        if _resolve_council_specialist(p.lead_council_role) is None
                        else []
                    )
                    + (
                        ["`diagnostic_brief` must be non-empty when `needs_specialist_probe=True`."]
                        if p.needs_specialist_probe and not str(p.diagnostic_brief or "").strip()
                        else []
                    ),
                    parts=wish_multimodal_parts or None,
                    caller_role="CouncilTable",
                    candidate_dir=c_dir,
                )
                lead_spec = _resolve_council_specialist(routing_plan.lead_council_role)
                lead_role_name = lead_spec["role"] if lead_spec is not None else "GameplaySystemsEngineer"
                self._emit_event(
                    "council_steering_routed",
                    {
                        "deliberation": routing_plan.deliberation,
                        "reasoning_rationale": routing_plan.deliberation,
                        "lead_council_role": lead_role_name,
                        "supporting_council_roles": list(routing_plan.supporting_council_roles or []),
                        "needs_specialist_probe": routing_plan.needs_specialist_probe,
                        "specialist_probe_agent": routing_plan.specialist_probe_agent,
                    },
                )

                pack_prompt_block = ""
                if is_real_genai_client(self.client):
                    curator = ContextCuratorAgent(
                        cartridge_dir=c_dir,
                        client=self.client,
                        model_name=self.model_name,
                        board=self.board,
                        game_bible=game_bible,
                        event_callback=self.event_callback,
                    )
                    pack = curator.curate_context(
                        target_agent_role=f"council_reviewer:{lead_role_name}",
                        task_request=(
                            f"Executive Producer Live Steering / Problem Report: {steering_summary_for_prompt}\n"
                            f"Council Routing Deliberation: {routing_plan.deliberation}\n"
                            f"Inspect the cartridge entrypoint (`run/main_scene`), relevant `.gd` scripts, and asset references "
                            f"to identify the root cause and raise/update the exact ProductionBoard task(s) needed to resolve it.\n"
                            f"{reach.get('summary_text', '')}"
                        ),
                        item_id="executive_producer_steering",
                        item_kind="integration",
                    )
                    pack_prompt_block = pack.prompt_block
                curated_code_block = (
                    (f"{wish_context_block}\n\n" if wish_context_block else "")
                    + f"=== CARTRIDGE CODEBASE, ASSET INVENTORY & ENTRYPOINT CONTEXT FOR STEERING ===\n"
                    f"{reach.get('summary_text', '')}\n"
                    f"{recent_board_history}\n"
                    f"{asset_dossier_txt}\n"
                    f"{pack_prompt_block}\n"
                )
                if routing_plan.needs_specialist_probe and str(routing_plan.diagnostic_brief or "").strip():
                    diag_probe = CouncilDiagnosticProbe(
                        deliberation=routing_plan.deliberation,
                        specialist_agent=routing_plan.specialist_probe_agent or "qa_test_engineer",
                        diagnostic_brief=routing_plan.diagnostic_brief,
                        target_paths=list(routing_plan.target_paths or []),
                        needs_runtime_debug_scene=True,
                    )
                    probe_block, probe_parts = self._run_council_diagnostic_probe(
                        c_dir,
                        diag_probe,
                        steering_summary_for_prompt,
                    )
                    curated_code_block += probe_block
                    if probe_parts:
                        wish_multimodal_parts = list(wish_multimodal_parts or []) + probe_parts
            except Exception as cur_exc:
                logger.warning("ContextCurator steering routing/curation skipped: %s", cur_exc)

        if use_dynamic_steering_routing and routing_plan is not None:
            raw_queue = [routing_plan.lead_council_role] + list(routing_plan.supporting_council_roles or [])
            pivot_queue: List[str] = []
            for r_cand in raw_queue:
                sp_cand = _resolve_council_specialist(r_cand)
                if sp_cand is not None and sp_cand["role"] not in pivot_queue:
                    pivot_queue.append(sp_cand["role"])
            if not pivot_queue:
                pivot_queue = ["GameplayDesigner", "VisualArtDirector", "GameplaySystemsEngineer"]
        elif use_dynamic_steering_routing:
            pivot_queue = [
                "GameplayDesigner",
                "LevelAndProgressionDesigner",
                "VisualArtDirector",
                "UXExperienceDesigner",
                "GameplaySystemsEngineer",
            ]
        else:
            pivot_queue = ["CreativeDirector", "GameplaySystemsEngineer"]

        max_pivot_turns = 6
        visited_roles: set[str] = set()
        handoff_reasons: Dict[str, Tuple[str, str]] = {}
        items_before_steering = len(self.board.items()) if self.board is not None else 0
        base_turn_idx = len(game_bible.writers_room_transcript)
        base_bible_snapshot = self._format_bible_snapshot(
            game_bible,
            omit_playtest_session_ids=consumed_pt_sids,
        )
        shared_steering_prefix = self._build_specialist_shared_prefix(
            dna_summary,
            base_bible_snapshot,
            curated_code_block,
        )
        steering_min_calls = 1 if len(pivot_queue) >= 2 else 2
        idx_q = 0
        with explicit_prefix_cache_scope(self.client, ttl_seconds=600, reuse_outer=False):
            while idx_q < len(pivot_queue) and len(visited_roles) < max_pivot_turns:
                pivot_role = pivot_queue[idx_q]
                idx_q += 1
                spec = _resolve_council_specialist(pivot_role)
                if spec is None or spec["role"] in visited_roles:
                    continue
                visited_roles.add(spec["role"])

                has_remaining_speakers = any(
                    (_resolve_council_specialist(rem) is not None and _resolve_council_specialist(rem)["role"] not in visited_roles)  # type: ignore[index]
                    for rem in pivot_queue[idx_q:]
                )
                items_now = len(self.board.items()) if self.board is not None else 0
                still_drained = bool(self.board is not None and self.board.is_drained())
                no_task_added_yet = (items_now <= items_before_steering)
                must_add_task = bool(
                    (not has_remaining_speakers)
                    and (
                        (board_was_drained and still_drained)
                        or (use_dynamic_steering_routing and no_task_added_yet)
                    )
                )
                can_handoff = len(visited_roles) < max_pivot_turns
                handoff_prefix = ""
                if spec["role"] in handoff_reasons:
                    from_role, h_reason = handoff_reasons[spec["role"]]
                    handoff_prefix = (
                        f"[Routed directly to you by {from_role}"
                        + (f": {h_reason}" if h_reason else "")
                        + "] "
                    )
                listen_check = WritersRoomListenCheck(
                    deliberation=f"Immediate Council discussion and slice task planning required to honor Executive Producer steering: {steering_summary_for_prompt}",
                    should_interject=True,
                    urgency_score=1.0,
                    intended_contribution_summary=(
                        f"{handoff_prefix}Treat the Executive Producer's message ('{steering_summary_for_prompt}') with full respect and cater directly to what they asked for "
                        f"at its full intended architectural scope without defensiveness, excuses, or silent downgrading (Rule 5). "
                        f"Be candid and self-critical about where our current build fell short of their expectations (and never open `spoken_dialogue` with "
                        f"peer flattery or 'I completely agree with...'), reason from first principles about how to integrate their direction cleanly into "
                        f"your domain, update your Living Game Bible sections, refine unstarted slice tasks in `backlog_refinements`, and raise focused, "
                        f"achievable production-code `backlog_additions` tasks that deliver their request cleanly without boiling the ocean on unrelated systems. "
                        f"If another Council specialist's domain should handle or co-own this request next, set `route_to_council_role` and `routing_handoff_reason`."
                    ),
                )
                extra_turn_kwargs: Dict[str, Any] = {}
                if getattr(self._execute_specialist_turn, "__func__", None) is CreativeWritersRoom._execute_specialist_turn:
                    extra_turn_kwargs = {
                        "frozen_bible_snapshot": base_bible_snapshot,
                        "recent_turn_delta": self._format_utterance_delta(game_bible, base_turn_idx),
                        "cacheable_prefix": shared_steering_prefix,
                        "min_calls_before_create": steering_min_calls,
                        "omit_playtest_session_ids": consumed_pt_sids,
                    }
                proposal = self._execute_specialist_turn(
                    specialist=spec,
                    listen_check=listen_check,
                    game_bible=game_bible,
                    dna_summary=dna_summary,
                    extra_context_block=curated_code_block,
                    require_backlog_addition=must_add_task,
                    extra_multimodal_parts=wish_multimodal_parts or None,
                    allow_council_reroute_handoff=can_handoff,
                    visited_council_roles=visited_roles,
                    **extra_turn_kwargs,
                )
                self._apply_turn_proposal(
                    game_bible=game_bible,
                    speaker_role=spec["role"],
                    speaker_display_name=spec["display_name"],
                    proposal=proposal,
                    candidate_dir=candidate_dir,
                )
                routed_spec, handoff_txt = _parse_council_reroute(proposal)
                if (
                    routed_spec is not None
                    and routed_spec["role"] not in visited_roles
                    and len(visited_roles) < max_pivot_turns
                ):
                    routed_name = routed_spec["role"]
                    pivot_queue = pivot_queue[:idx_q] + [routed_name] + [
                        r for r in pivot_queue[idx_q:] if r != routed_name
                    ]
                    handoff_reasons[routed_name] = (spec["role"], handoff_txt)
                    self._emit_activity(
                        agent_role=routed_name,
                        status_text=(
                            f"{spec['display_name']} ({spec['role']}) routed steering turn to "
                            f"{routed_spec['display_name']} ({routed_name})"
                            + (f": {handoff_txt[:100]}" if handoff_txt else "")
                        ),
                        active_agents=[spec["role"], routed_name],
                        phase="council_reroute",
                    )

    _SHAPING_COUNCIL_ROLES = (
        "GameplayDesigner",
        "LevelAndProgressionDesigner",
        "GameplaySystemsEngineer",
        "UXExperienceDesigner",
        "VisualArtDirector",
        "AudioAtmosphereDesigner",
    )

    def _has_started_execution(self, item: Any, candidate_dir: Optional[Path] = None) -> bool:
        """Returns True if `item` has already completed worker turns, revisions, or retries (so Council
        pre-execution Task Shaping must not re-shape or split it on pause/resume)."""
        if getattr(item, "revisions", None) or getattr(item, "failure_count", 0) > 0 or getattr(item, "spec_resets", 0) > 0:
            return True
        if self.board is not None and hasattr(self.board, "has_prior_execution_activity"):
            try:
                if self.board.has_prior_execution_activity(item):
                    return True
            except Exception:
                pass
        c_dir = candidate_dir or getattr(self, "_active_candidate_dir", None)
        if c_dir is not None:
            raw_id = str(getattr(item, "item_id", ""))
            safe_id = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in raw_id).strip("_") or "item"
            rev_idx = len(getattr(item, "revisions", None) or [])
            trace_path = Path(c_dir) / "artifacts" / "work_items" / safe_id / f"lead_trace_rev{rev_idx}.json"
            if trace_path.is_file():
                try:
                    t_data = json.loads(trace_path.read_text(encoding="utf-8"))
                    if isinstance(t_data, dict) and (t_data.get("turns") or t_data.get("pending_decision")):
                        return True
                except Exception:
                    pass
        return False

    def _items_needing_domain_direction(
        self,
        role_name: str,
        candidate_dir: Optional[Path] = None,
    ) -> List[Any]:
        """Returns unstarted board items that were raised by a different specialist (or split/added in Stage 1)
        and have not yet received `[<role_name> Direction]` input."""
        if self.board is None:
            return []
        role_prefix = f"[{role_name} Direction]:"
        needed: List[Any] = []
        for item in self.board.items():
            if item.state not in ("ready", "blocked"):
                continue
            if self._has_started_execution(item, candidate_dir=candidate_dir):
                continue
            if item.raised_by_role == role_name:
                continue
            if any(str(n).startswith(role_prefix) for n in (item.shaping_notes or [])):
                continue
            needed.append(item)
        return needed

    def _run_full_council_direction_pass(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        bible_snapshot: str,
        candidate_dir: Optional[Path] = None,
    ) -> int:
        """Stage 2 of Council Task Shaping: Runs all 6 Council domain leads in parallel over the complete
        Production Board (including any tasks added or split during Stage 1) so every task receives
        cross-functional Council direction before workers execute."""
        if self.board is None:
            return 0
        role_targets: List[Tuple[str, Dict[str, Any], List[Any]]] = []
        for role_name in self._SHAPING_COUNCIL_ROLES:
            spec = next((sp for sp in SPECIALIST_ROSTER if sp["role"] == role_name), None)
            if spec is None:
                continue
            targets = self._items_needing_domain_direction(role_name, candidate_dir=candidate_dir)
            if targets:
                role_targets.append((role_name, spec, targets))
        if not role_targets:
            return 0

        active_roles = [r for r, _, _ in role_targets]
        self._emit_activity(
            agent_role="CouncilTaskShaping",
            status_text=(
                f"Council specialists ({', '.join(active_roles)}) are contributing cross-domain direction "
                f"across all Production Board tasks..."
            ),
            active_agents=active_roles,
            phase="task_shaping",
        )
        board_snap = self.board.snapshot_text()
        dna_header_s2 = self._dna_header_if_absent(dna_summary, bible_snapshot)
        stage2_shared_prefix = (
            f"{dna_header_s2}"
            f"{bible_snapshot}\n\n"
            f"{board_snap}\n\n"
        )
        stage2_eager = len(role_targets) >= 2

        def _shape_for_role(entry: Tuple[str, Dict[str, Any], List[Any]]) -> Tuple[str, TaskShapingBatch]:
            role_name, spec, targets = entry
            display_name = spec["display_name"]
            required_ids = {t.item_id for t in targets}
            target_lines = "\n".join(
                f"  - [{t.item_id}] ({t.kind}, raised_by={t.raised_by_role}, deps={t.depends_on}, modules={t.target_modules}) "
                f"'{t.title}':\n    spec: {t.spec}\n    proof_spec: {t.proof_spec}"
                for t in targets
            )
            prompt = (
                f"You are the {display_name} ({role_name}) in Stage 2 of the Council Task-Shaping Session (Full-Council Cross-Domain Direction).\n"
                f"Your specialty focus: {spec['focus']}\n\n"
                f"All tasks and task decompositions are now on the Production Board. Before workers start building, "
                f"you must review EVERY unstarted task below from your domain's perspective and provide one `backlog_refinements` entry per task:\n"
                f"TASKS REQUIRING YOUR DOMAIN REVIEW ({len(targets)} tasks):\n"
                f"{target_lines}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. For EVERY task listed in `TASKS REQUIRING YOUR DOMAIN REVIEW` above, include one `WorkItemRefinement` entry in `backlog_refinements` with matching `item_id`:\n"
                f"   - Set `action='enrich'` and populate `spec_addition_or_rewrite` (plus optional `proof_spec_update`, `add_depends_on`, or `add_target_modules`—using encapsulated per-feature `.gd` paths like `scenes/mechanics/<feature>.gd`, `scenes/entities/<entity>.gd`, or `scenes/ui/<widget>.gd` when assigning or refining code targets so parallel SWE workers can lock disjoint files) whenever your domain has direction, integration contracts, UX affordances, non-overlapping HUD layout rules, world/progression hooks, visual rig/silhouette specs, or reactive audio cues for that task.\n"
                f"   - Specify generalizable, world-wide invariants (querying authoritative world geometry/regions, node groups, and player-driven controls/state across the entire map)—never prescribe hardcoded `(x, y)` coordinate whitelists, bounding-box overrides, or fixed aiming/spawn coordinates for production code (keep sample coordinates strictly in `proof_spec_update` if needed as QA test landmarks).\n"
                f"   - Set `action='no_input'` ONLY if, after deliberating, your domain genuinely has zero requirements or touchpoints with that specific task.\n"
                f"2. Leave `backlog_additions=[]` in Stage 2 (all companion tasks were already raised in Stage 1).\n"
                f"Return `TaskShapingBatch` with `deliberation` FIRST."
            )

            def _validate_stage2(batch: TaskShapingBatch) -> List[str]:
                issues: List[str] = []
                seen_ids = {r.item_id for r in (batch.backlog_refinements or [])}
                missing = sorted(required_ids - seen_ids)
                if missing:
                    if batch.backlog_refinements or (batch.deliberation or "").strip():
                        for m_id in missing:
                            batch.backlog_refinements.append(
                                WorkItemRefinement(
                                    deliberation="Reviewed in Stage 2; no additional domain requirements.",
                                    item_id=m_id,
                                    action="no_input",
                                )
                            )
                    else:
                        issues.append(
                            f"Missing `backlog_refinements` decision for required task(s): {missing}. "
                            f"Provide one entry per listed task with `action='enrich'` (and `spec_addition_or_rewrite`) or `action='no_input'`."
                        )
                for r in batch.backlog_refinements or []:
                    if r.action == "split":
                        r.action = "enrich" if (r.spec_addition_or_rewrite.strip() or r.deliberation.strip()) else "no_input"
                issues.extend(self.board.validate_refinements(batch.backlog_refinements or []))
                return issues

            eff_shape_dir = self._resolve_memory_dir(candidate_dir)
            shape_mem_block = ""
            if eff_shape_dir is not None and (eff_shape_dir / "game_memory.json").is_file():
                from engine.studio.game_memory import GameMemoryStore

                shape_mem_block = GameMemoryStore(
                    eff_shape_dir, event_callback=getattr(self, "event_callback", None)
                ).format_role_memory_briefing(role_name)
            if shape_mem_block:
                prompt = f"{prompt}\n\n{shape_mem_block}"
            return role_name, self._structured_with_validation(
                prompt,
                TaskShapingBatch,
                _validate_stage2,
                caller_role=role_name,
                candidate_dir=eff_shape_dir,
                cacheable_prefix=stage2_shared_prefix,
                min_calls_before_create=1 if stage2_eager else 2,
            )

        with explicit_prefix_cache_scope(self.client, ttl_seconds=360, reuse_outer=False):
            results = self._map_parallel(_shape_for_role, role_targets)
        applied_count = 0
        for role_name, batch in results:
            if batch.backlog_refinements:
                shaped = self.board.apply_refinements(
                    batch.backlog_refinements,
                    by_role=role_name,
                )
                applied_count += len(shaped)
        return applied_count

    def _run_council_task_shaping_round(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        candidate_dir: Optional[Path] = None,
    ) -> int:
        """Runs a two-stage Council task-shaping and decomposition pass over unstarted Production Board items
        before workers start claiming tasks:
        - Stage 1 (Decomposition & Missing Task Discovery): Specialists decompose monolithic tasks (`action='split'`)
          and raise any missing companion/prerequisite tasks (`backlog_additions`).
        - Stage 2 (Full-Council Cross-Domain Direction): Once the complete task list is on the board (including all
          newly added and split tasks), all 6 Council specialists run in parallel to review every relevant task and
          contribute domain direction (`action='enrich'`)."""
        if self.board is None or not is_real_genai_client(self.client):
            return 0
        unstarted = [
            i for i in self.board.items()
            if i.state in ("ready", "blocked") and not self._has_started_execution(i, candidate_dir=candidate_dir)
        ]
        if not unstarted:
            return 0
        bible_snapshot = self._format_bible_snapshot(game_bible)
        active_slice_obj = game_bible.get_active_slice()
        default_slice_id = active_slice_obj.slice_id if active_slice_obj is not None else ""
        total_changes = 0
        dna_header_s1 = self._dna_header_if_absent(dna_summary, bible_snapshot)
        stage1_shared_prefix = (
            f"{dna_header_s1}"
            f"{bible_snapshot}\n\n"
        )
        stage1_cache_cm = explicit_prefix_cache_scope(self.client, ttl_seconds=360, reuse_outer=False)
        stage1_cache_cm.__enter__()
        try:
            # Stage 1: Decomposition ('split') & Missing Companion Task Discovery ('backlog_additions')
            for role_name in self._SHAPING_COUNCIL_ROLES:
                unstarted_now = [
                    i for i in self.board.items()
                    if i.state in ("ready", "blocked") and not self._has_started_execution(i, candidate_dir=candidate_dir)
                ]
                if not unstarted_now:
                    break
                role_prefix = f"[{role_name} Direction]:"
                if any(str(n).startswith(role_prefix) for i in unstarted_now for n in (i.shaping_notes or [])):
                    continue
                spec = next((sp for sp in SPECIALIST_ROSTER if sp["role"] == role_name), None)
                if spec is None:
                    continue
                display_name = spec["display_name"]
                self._emit_activity(
                    agent_role=role_name,
                    status_text=f"{display_name} ({role_name}) is decomposing tasks & checking for missing companion tasks (Stage 1)...",
                    active_agents=[role_name],
                    phase="task_shaping",
                )
                unstarted_id_lines = "\n".join(
                    f"  - [{t.item_id}] ({t.kind}, state={t.state}, slice={t.slice_id or default_slice_id}, deps={t.depends_on}) '{t.title}'"
                    for t in unstarted_now
                )
                prompt = (
                    f"{self.board.snapshot_text()}\n\n"
                    f"UNSTARTED TASKS CURRENTLY ON THE BOARD ({len(unstarted_now)} tasks — use ONLY these exact `item_id` values when referencing existing unstarted tasks in `backlog_refinements`, `blocks_items`, or `add_depends_on`):\n"
                    f"{unstarted_id_lines}\n\n"
                    f"You are the {display_name} ({role_name}) in Stage 1 of the Council Task-Shaping & Decomposition Session.\n"
                    f"Your specialty focus: {spec['focus']}\n\n"
                    f"{STUDIO_CRITICAL_DISCOURSE_CHARTER}\n\n"
                    f"Before the full Council gives cross-domain direction in Stage 2, skeptically audit the unstarted (`ready` or `blocked`) tasks on the Production Board (catering faithfully to any Executive Producer directives while keeping task count bounded to THIS active slice's scope rather than trying to build every future slice at once):\n"
                    f"1. DECOMPOSE MONOLITHIC TASKS (`action='split'` in `backlog_refinements`): Should any task be split into smaller, focused tasks (for example, separating core simulation mechanics from interactive player placement/building tools, separating world/level progression from HUD telemetry, or splitting a multi-entity visual_asset bundle into 1-entity-per-item visual_asset tasks)? When you split a task, downstream dependents on the board automatically re-point to your split tasks.\n"
                    f"2. RAISE MISSING COMPANION TASKS (`backlog_additions`): Is an essential active-slice prerequisite asset, UI affordance, level region, or sub-mechanic task missing altogether? Propose it in `backlog_additions` (and use `blocks_items` or `add_depends_on` to link it), without bloating the board with out-of-slice wishlist features.\n"
                    f"3. CONTRIBUTE INITIAL DOMAIN DIRECTION (`action='enrich'` in `backlog_refinements`): You may also enrich existing tasks with `spec_addition_or_rewrite` and `add_depends_on` (define generalizable, world-wide system/geometry contracts rather than hardcoded `(x, y)` coordinate whitelists or fixed target coordinates for production code).\n"
                    f"Buildable kinds right now: {self._supported_kinds() or 'all'}.\n"
                    f"Return `TaskShapingBatch` with `deliberation` FIRST."
                )

                def _validate_shaping(batch: TaskShapingBatch) -> List[str]:
                    issues: List[str] = []
                    if batch.backlog_additions:
                        issues.extend(self.board.validate_proposals(batch.backlog_additions))
                        issues.extend(self._unsupported_kind_issues(batch.backlog_additions))
                    if batch.backlog_refinements:
                        issues.extend(
                            self.board.validate_refinements(
                                batch.backlog_refinements,
                                new_proposals=batch.backlog_additions,
                            )
                        )
                        for r in batch.backlog_refinements:
                            if r.action == "split" and r.split_items:
                                issues.extend(self._unsupported_kind_issues(r.split_items))
                    return issues

                batch = self._structured_with_validation(
                    prompt,
                    TaskShapingBatch,
                    _validate_shaping,
                    cacheable_prefix=stage1_shared_prefix,
                    min_calls_before_create=1 if len(self._SHAPING_COUNCIL_ROLES) >= 2 else 2,
                )
                added_this_turn: List[Any] = []
                shaped_this_turn: List[str] = []
                if batch.backlog_additions:
                    for prop in batch.backlog_additions:
                        if prop.kind == "level_map" and "level_layout.json" not in prop.target_modules:
                            prop.target_modules.append("level_layout.json")
                    added_this_turn = self.board.add_items(
                        batch.backlog_additions,
                        raised_by_role=role_name,
                        slice_id=default_slice_id,
                    )
                    total_changes += len(added_this_turn)
                if batch.backlog_refinements:
                    shaped_this_turn = self.board.apply_refinements(
                        batch.backlog_refinements,
                        by_role=role_name,
                    )
                    total_changes += len(shaped_this_turn)
                if added_this_turn or shaped_this_turn:
                    t_idx = len(game_bible.writers_room_transcript)
                    summary_badges: List[str] = []
                    spoken_parts: List[str] = []
                    if added_this_turn:
                        added_ids = [it.item_id for it in added_this_turn]
                        summary_badges.append(f"Raised {len(added_this_turn)} backlog item(s): {added_ids}")
                        spoken_parts.append(
                            "Added companion task(s): "
                            + ", ".join(f"`{it.item_id}` ({it.title})" for it in added_this_turn)
                            + "."
                        )
                    if shaped_this_turn:
                        summary_badges.append(f"Shaped/split {len(shaped_this_turn)} backlog item(s) -> {shaped_this_turn}")
                        spoken_parts.append(f"Shaped/refined board task(s): {', '.join(f'`{sid}`' for sid in shaped_this_turn)}.")
                    shaping_utt = WritersRoomUtterance(
                        deliberation=batch.deliberation,
                        utterance_id=f"utt_{t_idx:02d}_{role_name}_task_shaping",
                        turn_index=t_idx,
                        speaker_role=role_name,
                        speaker_display_name=display_name,
                        spoken_dialogue=f"{batch.deliberation} {' '.join(spoken_parts)}".strip(),
                        addressed_to=["CreativeDirector", "GameplayDesigner", "GameplaySystemsEngineer"],
                        bible_updates_summary=summary_badges,
                    )
                    game_bible.writers_room_transcript.append(shaping_utt)
                    self._persist_bible(game_bible, candidate_dir=candidate_dir)
                    self._emit_event(
                        "writers_room_utterance",
                        {
                            "deliberation": shaping_utt.deliberation,
                            "utterance_id": shaping_utt.utterance_id,
                            "turn_index": shaping_utt.turn_index,
                            "speaker_role": shaping_utt.speaker_role,
                            "speaker_display_name": shaping_utt.speaker_display_name,
                            "spoken_dialogue": shaping_utt.spoken_dialogue,
                            "addressed_to": shaping_utt.addressed_to,
                            "bible_updates_summary": shaping_utt.bible_updates_summary,
                        },
                    )
        finally:
            stage1_cache_cm.__exit__(None, None, None)

        # Stage 2: Parallel Full-Council Cross-Domain Direction over the complete board (including all newly added/split tasks)
        total_changes += self._run_full_council_direction_pass(
            game_bible=game_bible,
            dna_summary=dna_summary,
            bible_snapshot=bible_snapshot,
            candidate_dir=candidate_dir,
        )

        if candidate_dir is not None and total_changes > 0:
            self._persist_bible(game_bible, candidate_dir=candidate_dir)
        return total_changes

    def _is_end_of_slice_blind_playtest_enabled(self) -> bool:
        """Returns True when the interactive turn-by-turn Blind AI Playtester is enabled for slice completion."""
        explicit_flag = getattr(self, "enable_end_of_slice_blind_playtest", None)
        if explicit_flag is False:
            return False
        if explicit_flag is True:
            return True
        if os.environ.get("STUDIO_ENABLE_END_OF_SLICE_PLAYTEST", "1") == "0":
            return False
        return bool(is_real_genai_client(self.client))

    def _run_slice_assembly_check(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        candidate_dir: Optional[Path],
        pass_index: int = 0,
    ) -> int:
        """Runs entrypoint reachability & zero-input boot sanity before handing off directly to the Blind AI Playtester
        (or runs the legacy scripted QA playthrough + Council roundtable when explicitly requested via
        `enable_slice_assembly_check=True` or when the Blind AI Playtester is disabled)."""
        if (
            self.board is None
            or candidate_dir is None
            or not (Path(candidate_dir) / "project.godot").is_file()
        ):
            return 0
        # Only run automatic slice assembly checks on live Gemini builds unless explicitly enabled in tests
        if (
            not is_real_genai_client(self.client)
            and not getattr(self, "enable_slice_assembly_check", False)
            and not getattr(self, "enable_end_of_slice_blind_playtest", False)
        ):
            return 0

        c_dir = Path(candidate_dir)
        active_slice_obj = game_bible.get_active_slice()
        active_slice_id = active_slice_obj.slice_id if active_slice_obj is not None else "slice_1"
        active_slice_title = active_slice_obj.title if active_slice_obj is not None else "Slice 1"
        active_slice_idx = active_slice_obj.slice_index if active_slice_obj is not None else 1

        from engine.studio.context_curator import (
            ContextCuratorAgent,
            compute_entrypoint_reachability,
        )
        from engine.studio.proofs import format_filmstrip_telemetry_progression

        legacy_scripted_playtest_explicit = bool(
            getattr(self, "enable_slice_assembly_check", False)
            or getattr(self, "assembly_playtest_planner_fn", None) is not None
            or getattr(self, "assembly_filmstrip_fn", None) is not None
        )
        blind_playtest_takes_over = (
            self._is_end_of_slice_blind_playtest_enabled() and not legacy_scripted_playtest_explicit
        )

        reach = compute_entrypoint_reachability(c_dir)
        parts: List[Any] = []
        boot_telemetry_summary = ""
        playtest_plan_summary = ""
        playtest_test_gd_source = ""
        playtest_assertion_failure = ""
        proof_out = c_dir / "artifacts" / "proofs" / f"filmstrip_slice_assembly_check_pass{pass_index}.png"
        rel_proof_path = f"artifacts/proofs/filmstrip_slice_assembly_check_pass{pass_index}.png"

        if blind_playtest_takes_over:
            self._emit_activity(
                agent_role="GameplaySystemsEngineer",
                status_text=(
                    f"Verifying entrypoint reachability & zero-input boot sanity for '{active_slice_title}' "
                    f"before launching Blind AI Playtester..."
                ),
                active_agents=["GameplaySystemsEngineer"],
                phase="slice_assembly_check",
            )
            main_scene_res = reach.get("main_scene") or "res://main.tscn"
            boot_errs: List[str] = []
            try:
                from engine.sim.godot_verifier import GodotHeadlessSimulator

                boot_sim = GodotHeadlessSimulator(
                    c_dir,
                    capture_viewport=False,
                    capture_ticks=[1, 15],
                    input_script=[],
                    setup_script=None,
                    max_ticks=15,
                    telemetry_mode="null",
                    target_scene=main_scene_res,
                )
                boot_state = boot_sim.run_until_complete()
                boot_errs = [
                    f"{d.file_path.name}:{d.line_number} - {d.error_message}"
                    for d in (getattr(boot_sim, "runtime_diagnostics", None) or [])
                    if getattr(d, "file_path", None) is not None and str(d.file_path).endswith(".gd")
                ] + list(getattr(boot_sim, "script_errors", None) or [])
                if getattr(boot_state, "is_crashed", False) and not boot_errs:
                    boot_errs.append(f"Playable entrypoint '{main_scene_res}' exited abnormally during zero-input boot sanity check.")
            except Exception as boot_exc:
                boot_errs = [str(boot_exc)]

            unreferenced_mods = list(reach.get("unreferenced_from_entrypoint") or [])
            if not boot_errs and not unreferenced_mods:
                self._emit_event(
                    "slice_assembly_check_completed",
                    {
                        "deliberation": (
                            f"Static entrypoint reachability ({len(reach.get('reachable_from_entrypoint') or [])} reachable modules, "
                            f"0 unreferenced) and zero-input boot sanity passed on '{main_scene_res}'; handing off directly to Blind AI Playtester."
                        ),
                        "pass_index": pass_index,
                        "is_slice_assembled": True,
                        "mode": "static_boot_sanity",
                        "playability_and_fun_evaluation": "Handed off to interactive Blind AI Playtester.",
                        "unmounted_or_disconnected_systems": [],
                        "spawned_item_ids": [],
                    },
                )
                return 0

            if boot_errs:
                playtest_assertion_failure = (
                    f"Entrypoint Parity Gate failed: zero-input boot sanity check on '{main_scene_res}' raised runtime GDScript error(s): "
                    + " | ".join(str(e) for e in boot_errs[:5])
                )
            playtest_plan_summary = (
                f"Static Entrypoint Reachability & Zero-Input Boot Sanity Check on `{main_scene_res}` "
                f"(unreferenced_modules={unreferenced_mods}; boot_errors={len(boot_errs)}). "
                "Interactive gameplay evaluation is delegated to the Blind AI Playtester once entrypoint wiring and boot health are clean."
            )
        else:
            self._emit_activity(
                agent_role="QATestEngineer",
                status_text=(
                    f"QA is building & running an automated Slice Playtest Script for '{active_slice_title}' "
                    f"(Playtest & Assembly Check #{pass_index + 1})..."
                ),
                active_agents=["QATestEngineer", "CreativeDirector", "GameplaySystemsEngineer", "GameplayDesigner", "UXExperienceDesigner"],
                phase="slice_assembly_check",
            )

        if not blind_playtest_takes_over and (
            is_real_genai_client(self.client)
            or getattr(self, "assembly_filmstrip_fn", None) is not None
            or getattr(self, "assembly_playtest_planner_fn", None) is not None
        ):
            try:
                filmstrip_fn = getattr(self, "assembly_filmstrip_fn", None)
                playtest_planner_fn = getattr(self, "assembly_playtest_planner_fn", None)
                if filmstrip_fn is not None and playtest_planner_fn is None:
                    filmstrip_fn(c_dir, [15, 60, 120, 180], proof_out)
                else:
                    from engine.studio.production_board import WorkItem
                    from engine.studio.proofs import capture_filmstrip
                    from engine.studio.workers import WorkerContext, plan_filmstrip

                    gh_obj = getattr(game_bible, "goal_hierarchy", None)
                    if isinstance(gh_obj, list):
                        gh_list = gh_obj
                    elif gh_obj is not None:
                        gh_list = list(getattr(gh_obj, "immediate_goals", None) or []) + list(
                            getattr(gh_obj, "short_term_goals", None) or []
                        )
                    else:
                        gh_list = []
                    first_goals = (
                        "; ".join(str(g) for g in gh_list[:2])
                        if gh_list
                        else "Complete the opening interactive objective of the current slice."
                    )
                    raw_mechs = (
                        getattr(active_slice_obj, "core_mechanics", None)
                        or getattr(getattr(game_bible, "core_loop", None), "player_verbs", None)
                        or []
                    )
                    slice_mechs = (
                        ", ".join(str(m) for m in raw_mechs)
                        if raw_mechs
                        else "core player movement and primary interaction verbs"
                    )
                    slice_onboard = (
                        getattr(active_slice_obj, "player_onboarding_and_communication_plan", "")
                        if active_slice_obj is not None
                        else ""
                    )
                    playtest_item_id = f"slice_playtest_{active_slice_id}_pass{pass_index}"
                    playtest_item = WorkItem(
                        deliberation=f"Synthesize automated Slice Playtest work item for {active_slice_id} (pass {pass_index}).",
                        item_id=playtest_item_id,
                        title=f"Automated Slice Playtest: {active_slice_title} (Slice #{active_slice_idx})",
                        kind="integration",
                        spec=(
                            f"Build a Slice Playtest Observer Script (`res://tests/test_{playtest_item_id}.tscn`) that instantiates the live "
                            f"playable game entrypoint (`preload('{reach.get('main_scene') or 'res://main.tscn'}').instantiate()`) as a child node in `_ready()`, "
                            f"attaches diagnostic overlays (velocity/heading vectors, target/goal markers, and telemetry including `player_pos`), "
                            f"and scripts real player `input_segments` (`actions`, `mouse_world_pos`, `mouse_click_pos`) to play through at least the first goal / core loop segment of the current slice ({first_goals}) "
                            f"exercising `{slice_mechs}`. If inspecting the production code reveals that the live entrypoint is broken or deadlocked under valid player inputs, "
                            f"report the root cause honestly via `production_defect_report` instead of mutating state inside the test scene. Onboarding plan: {slice_onboard}"
                        ),
                        proof_spec=(
                            f"Multi-frame gameplay filmstrip across 180-360 ticks showing how a real player booting `{reach.get('main_scene') or 'res://main.tscn'}` at Slice #{active_slice_idx} (`{active_slice_id}`) "
                            f"experiences the opening state, HUD/orientation, and causal progression toward the first goal ({first_goals})."
                        ),
                        raised_by_role="QATestEngineer",
                        target_modules=[reach.get("main_script") or "main.gd"],
                        slice_id=active_slice_id,
                    )
                    w_ctx = WorkerContext(
                        cartridge_dir=c_dir,
                        client=self.client,
                        model_name=self.model_name,
                        event_callback=self.event_callback,
                        board=self.board,
                    )
                    if playtest_planner_fn is not None:
                        pt_plan = playtest_planner_fn(w_ctx, playtest_item)
                    else:
                        pt_plan = plan_filmstrip(w_ctx, playtest_item)

                    playtest_inputs_summary = [
                        f"{s.from_tick}-{s.to_tick}:{s.actions or getattr(s, 'mouse_world_pos', None) or s.mouse_click_pos}"
                        for s in (pt_plan.input_segments or [])
                    ]
                    playtest_plan_summary = (
                        f"QA Slice Playtest Scene (`{pt_plan.target_scene}`): {pt_plan.council_test_scene_summary} | "
                        f"Overlays & Telemetry Strategy: {pt_plan.overlays_and_telemetry_strategy} | "
                        f"Ticks: {pt_plan.capture_ticks} | "
                        f"Inputs: {playtest_inputs_summary} | "
                        f"SetupCalls: {[c.method_name for c in (getattr(pt_plan, 'setup_calls', None) or [])]} | "
                        f"Assertions: {[f't={a.at_tick}:{a.snapshot_key} {a.expected_value_or_change}' for a in (getattr(pt_plan, 'state_assertions', None) or [])]}"
                    )
                    playtest_test_gd_source = (pt_plan.test_scene_gd_content or "").strip()
                    qa_defect_rep = (getattr(pt_plan, "production_defect_report", "") or "").strip()
                    if qa_defect_rep:
                        playtest_assertion_failure = f"QA Diagnostic Production Defect Report: {qa_defect_rep}"
                        if filmstrip_fn is None:
                            capture_filmstrip(
                                cartridge_dir=c_dir,
                                ticks=[15, 60, 120, 180],
                                out_path=proof_out,
                                input_script=[],
                                target_scene=reach.get("main_scene") or "res://main.tscn",
                            )
                        else:
                            filmstrip_fn(c_dir, [15, 60, 120, 180], proof_out)
                    else:
                        if playtest_test_gd_source:
                            from engine.studio.workers import (
                                _serialize_input_segments,
                                _write_qa_test_scene_pair,
                            )
                            pt_plan.target_scene, _, _ = _write_qa_test_scene_pair(
                                c_dir, playtest_item_id, playtest_test_gd_source, pt_plan.target_scene
                            )
                        else:
                            from engine.studio.workers import _serialize_input_segments
                        proof_out.parent.mkdir(parents=True, exist_ok=True)
                        script_segs = _serialize_input_segments(pt_plan)
                        setup_calls_list = [
                            {"at_tick": c.at_tick, "method_name": c.method_name, "args": json.loads(c.args_json or "[]")}
                            for c in (getattr(pt_plan, "setup_calls", None) or [])
                            if c.method_name.strip()
                        ]
                        try:
                            if filmstrip_fn is not None:
                                filmstrip_fn(c_dir, pt_plan.capture_ticks, proof_out)
                            else:
                                capture_filmstrip(
                                    cartridge_dir=c_dir,
                                    ticks=list(pt_plan.capture_ticks),
                                    out_path=proof_out,
                                    input_script=script_segs,
                                    setup_script=setup_calls_list or None,
                                    verify_counterfactual=bool(script_segs),
                                    state_assertions=list(getattr(pt_plan, "state_assertions", None) or []),
                                    target_scene=pt_plan.target_scene,
                                    verify_entrypoint_parity=True,
                                )
                        except Exception as run_exc:
                            playtest_assertion_failure = str(run_exc)
                            logger.warning("Slice playtest script reported failure/assertion error: %s", run_exc)
                            if not proof_out.is_file() and filmstrip_fn is None:
                                capture_filmstrip(
                                    cartridge_dir=c_dir,
                                    ticks=[15, 60, 120, 180],
                                    out_path=proof_out,
                                    input_script=[],
                                    target_scene=reach.get("main_scene") or "res://main.tscn",
                                )
                if proof_out.is_file():
                    part = self._load_image_part(proof_out)
                    if part is not None:
                        parts.append(part)
                    boot_telemetry_summary = format_filmstrip_telemetry_progression(proof_out)
            except Exception as cap_exc:
                playtest_assertion_failure = str(cap_exc)
                logger.warning("Slice assembly playtest capture encountered error: %s", cap_exc)

        # If a playtest plan was executed (or in live mode when proof_out exists), emit the QA Playtester utterance with inline filmstrip
        qa_utt_emitted = False
        if proof_out.is_file() and (
            is_real_genai_client(self.client)
            or getattr(self, "assembly_playtest_planner_fn", None) is not None
            or getattr(self, "enable_slice_playtest_discussion", False)
        ):
            qa_turn_idx = len(game_bible.writers_room_transcript)
            qa_spoken = (
                f"[Automated Slice Playtest — {active_slice_title} (Pass #{pass_index + 1})] "
                f"{playtest_plan_summary or 'Executed scripted slice playthrough on playable entrypoint.'}"
            )
            if playtest_assertion_failure:
                qa_spoken += f" Playtest script flagged assertion/runtime failure: {playtest_assertion_failure}"
            if boot_telemetry_summary:
                qa_spoken += f" Telemetry progression: {boot_telemetry_summary[:360]}"
            qa_utt = WritersRoomUtterance(
                deliberation=playtest_plan_summary or "Captured automated slice playtest filmstrip for Council inspection.",
                utterance_id=f"utt_{qa_turn_idx:02d}_QATestEngineer_slice_playtest_{pass_index}",
                turn_index=qa_turn_idx,
                speaker_role="QATestEngineer",
                speaker_display_name="Autonomous QA Playtester",
                spoken_dialogue=qa_spoken,
                addressed_to=["GameplayDesigner", "UXExperienceDesigner", "VisualArtDirector", "GameplaySystemsEngineer", "CreativeDirector"],
                bible_updates_summary=[f"slice_playtest:{active_slice_id}:pass_{pass_index}"],
                proof_image_paths=[rel_proof_path],
                artifact_emitted={
                    "type": "filmstrip",
                    "path": rel_proof_path,
                    "slice_id": active_slice_id,
                    "caption": f"Automated Slice Playtest Filmstrip — {active_slice_title} (Pass #{pass_index + 1})",
                },
            )
            game_bible.writers_room_transcript.append(qa_utt)
            self._persist_bible(game_bible, candidate_dir=candidate_dir)
            self._emit_event(
                "writers_room_utterance",
                {
                    "deliberation": qa_utt.deliberation,
                    "utterance_id": qa_utt.utterance_id,
                    "turn_index": qa_utt.turn_index,
                    "speaker_role": qa_utt.speaker_role,
                    "speaker_display_name": qa_utt.speaker_display_name,
                    "spoken_dialogue": qa_utt.spoken_dialogue,
                    "addressed_to": qa_utt.addressed_to,
                    "bible_updates_summary": qa_utt.bible_updates_summary,
                    "proof_image_paths": qa_utt.proof_image_paths,
                    "artifact_emitted": qa_utt.artifact_emitted,
                },
            )
            qa_utt_emitted = True

        curated_assembly_block = ""
        if is_real_genai_client(self.client):
            try:
                curator = ContextCuratorAgent(
                    cartridge_dir=c_dir,
                    client=self.client,
                    model_name=self.model_name,
                    board=self.board,
                    game_bible=game_bible,
                    event_callback=self.event_callback,
                )
                pack = curator.curate_context(
                    target_agent_role="council_reviewer:SliceAssemblyCouncil",
                    task_request=(
                        f"Evaluate the Automated Slice Playtest of '{active_slice_title}' ({active_slice_id}): "
                        f"verify whether all built modules, UI instruments, and gameplay systems are mounted in {reach.get('main_scene')}, "
                        f"and whether playing through the first goal of the slice is responsive at human scale, free of production debug overlays, and fun.\n"
                        f"{reach.get('summary_text', '')}"
                    ),
                    item_id=f"slice_assembly_check_pass{pass_index}",
                    item_kind="integration",
                )
                curated_assembly_block = f"{pack.prompt_block}\n"
            except Exception as cur_exc:
                logger.warning("ContextCurator slice assembly curation skipped: %s", cur_exc)

        approved_summaries: List[str] = []
        for it in self.board.items():
            if it.state == "approved":
                last_notes = (it.revisions[-1].worker_notes if getattr(it, "revisions", None) else "") or ""
                approved_summaries.append(
                    f"  - [{it.item_id}] ({it.kind}) '{it.title}' | target_modules={it.target_modules} | UNVERIFIED_WORKER_CLAIMS={last_notes[:200]}"
                )

        bible_snapshot = self._format_bible_snapshot(game_bible)
        asm_dna_header = self._dna_header_if_absent(dna_summary, bible_snapshot)
        qa_script_dossier_block = (
            f"=== QA SLICE PLAYTEST SCRIPT SOURCE (`{pt_plan.target_scene if 'pt_plan' in locals() else 'tests/...'}`) ===\n"
            f"{playtest_test_gd_source[:2400]}\n\n"
            if playtest_test_gd_source
            else ""
        )
        shared_playtest_dossier = (
            f"{asm_dna_header}"
            f"{bible_snapshot}\n\n"
            f"{curated_assembly_block}"
            f"=== PLAYABLE ENTRYPOINT REACHABILITY GRAPH ===\n{reach.get('summary_text', '')}\n\n"
            f"=== AUTOMATED SLICE PLAYTEST PLAN & SCENE SUMMARY ===\n{playtest_plan_summary or '(boot filmstrip capture)'}\n"
            + qa_script_dossier_block
            + (f"=== PLAYTEST ASSERTION / RUNTIME FAILURE ===\n{playtest_assertion_failure}\n\n" if playtest_assertion_failure else "\n")
            + f"=== SLICE PLAYTEST TELEMETRY PROGRESSION ===\n{boot_telemetry_summary or '(not captured)'}\n\n"
            f"=== APPROVED PRODUCTION BOARD ITEMS ({len(approved_summaries)} items) ===\n"
            + ("\n".join(approved_summaries) or "  (none)")
        )
        assembly_shared_prefix = f"{shared_playtest_dossier}\n\n"
        is_final_assembly = bool(getattr(self, "_final_verification_assembly_pass", False))
        max_asm_passes = int(getattr(self, "max_assembly_passes", 3))
        assembly_pass_label = (
            f"Post-Remediation Final Verification Pass after {max_asm_passes} iteration round(s)"
            if is_final_assembly
            else f"Pass #{pass_index + 1} of {max_asm_passes}"
        )
        final_assembly_note = (
            "\nPOST-REMEDIATION FINAL VERIFICATION PASS: All current-slice assembly iteration rounds have been spent, "
            "and this check verifies the repaired build. Do NOT queue optional polish or feature-creep tasks; set "
            "`is_slice_assembled=True` (`backlog_additions=[]`) if the playable entrypoint boots cleanly, mounts active-slice "
            "systems, and passes automated playtest assertions, or set `is_slice_assembled=False` ONLY if a severe blocking "
            "assembly/boot defect still breaks the slice.\n"
            if is_final_assembly
            else ""
        )
        run_asm_roundtable = bool(
            not blind_playtest_takes_over
            and (is_real_genai_client(self.client) or getattr(self, "enable_slice_playtest_discussion", False))
        )

        roundtable_transcript_lines: List[str] = []
        with explicit_prefix_cache_scope(self.client, ttl_seconds=600, reuse_outer=False):
            if run_asm_roundtable:
                playtest_panel_roles = ["GameplayDesigner", "UXExperienceDesigner", "VisualArtDirector"]
                self._emit_activity(
                    agent_role="CouncilReviewPanel",
                    status_text=f"Council specialists ({', '.join(playtest_panel_roles)}) are conversing on the Slice Playtest results...",
                    active_agents=playtest_panel_roles,
                    phase="slice_assembly_check",
                )
                for p_role in playtest_panel_roles:
                    spec = next((sp for sp in SPECIALIST_ROSTER if sp["role"] == p_role), None)
                    disp_name = spec["display_name"] if spec else p_role
                    mandate = self._role_review_mandate(p_role)
                    prior_chat = (
                        "\n".join(f"  - {line}" for line in roundtable_transcript_lines)
                        if roundtable_transcript_lines
                        else "  (You are opening the Council's Slice Playtest critique after QA's playthrough presentation.)"
                    )
                    turn_prompt = (
                        f"{assembly_shared_prefix}"
                        f"=== SLICE PLAYTEST ROUNDTABLE CONVERSATION SO FAR ===\n{prior_chat}\n\n"
                        f"You are the {disp_name} ({p_role}) in the Writers' Room inspecting the Automated Slice Playtest filmstrip and telemetry for '{active_slice_title}' (`{active_slice_id}`, {assembly_pass_label}).\n"
                        f"{STUDIO_CRITICAL_DISCOURSE_CHARTER}\n"
                        f"{final_assembly_note}\n"
                        f"{mandate}\n"
                        "INSTRUCTIONS:\n"
                        "1. Determine `validation_criteria_for_my_role`: what must be true in this slice playtest for THIS active slice's scope (plus any Executive Producer direction) to be cohesive, intuitive (the player knows how to play), responsive, goal-driven, and genuinely playable from your specialist perspective?\n"
                        "2. Skeptically inspect every HIGH-resolution frame of the attached playtest filmstrip and telemetry progression:\n"
                        "   - COHESION & SPATIAL READABILITY: Is the playable world clearly visible and spatially staged around the player's actions, or is the screen dominated/occluded by a giant static panel, overlapping HUD widgets, or cut-off/bleeding sprite slicing?\n"
                        "   - INTUITIVENESS & KNOWING HOW TO PLAY: Would a human player opening `run/main_scene` immediately understand what controls to use, what each control does in the world, and what their immediate objective is without developer telemetry?\n"
                        "   - RESPONSIVENESS & DEPTH: Does holding player control inputs over 60–180 ticks (1–3 seconds) produce a responsive, human-playable rate of motion/turn/action and meaningful cause-and-effect in the world?\n"
                        "   - SHORT-TERM / LONG-TERM GOALS & ACCOMPLISHMENT: Did the scripted playthrough make real causal progress toward the slice's immediate/short-term goal while keeping the longer-term goal and progress milestones visible?\n"
                        "   - ZERO DEBUG OVERLAYS IN PRODUCTION: Are HUD instruments clean, non-overlapping, and free of developer debug text (`[DEBUG]`, `[QA ... PLAYTEST HARNESS]`, raw telemetry dumps) in the production game?\n"
                        "3. PRAGMATIC SLICE COMPLETION JUDGMENT: Never open `spoken_dialogue` with 'I completely agree with...' or peer flattery, and never pretend rough edges are 'flawless'. "
                        "Set `is_playable_and_fun_from_my_perspective=False` when active-slice scope is unmounted/broken, controls or causal progression fail, production debug overlays leak, high-impact visual/UI glitches (cut-off sprites, overflowing/overlapping HUD) hurt readability, or an Executive Producer directive is unmet. "
                        "Conversely, remember we cannot change everything in one go: once the active slice's scope is covered and major problem areas are cleaned up, call the slice assembled (`is_playable_and_fun_from_my_perspective=True`) while candidly naming any minor rough spots or broader ambitions to carry into future slices.\n"
                        "Provide `deliberation` FIRST and `is_playable_and_fun_from_my_perspective` strictly LAST."
                    )
                    turn_decision = self._structured_with_validation(
                        turn_prompt,
                        SlicePlaytestRoundtableTurn,
                        lambda _t: [],
                        temperature=0.35,
                        parts=parts,
                        cacheable_prefix=assembly_shared_prefix,
                        min_calls_before_create=1,
                    )
                    roundtable_transcript_lines.append(
                        f"[{disp_name} ({p_role})] (is_playable_and_fun={turn_decision.is_playable_and_fun_from_my_perspective} | criteria={turn_decision.validation_criteria_for_my_role}): "
                        f"{turn_decision.spoken_dialogue} | Issues: {turn_decision.observed_issues_or_gaps}"
                    )
                    t_idx = len(game_bible.writers_room_transcript)
                    p_utt = WritersRoomUtterance(
                        deliberation=turn_decision.deliberation,
                        utterance_id=f"utt_{t_idx:02d}_{p_role}_slice_playtest_{pass_index}",
                        turn_index=t_idx,
                        speaker_role=p_role,
                        speaker_display_name=disp_name,
                        spoken_dialogue=turn_decision.spoken_dialogue,
                        addressed_to=turn_decision.addressed_to or ["CreativeDirector", "GameplaySystemsEngineer"],
                        bible_updates_summary=[
                            f"slice_playtest_vote:{p_role}:{'pass' if turn_decision.is_playable_and_fun_from_my_perspective else 'needs_work'}"
                        ],
                    )
                    game_bible.writers_room_transcript.append(p_utt)
                    self._emit_event(
                        "writers_room_utterance",
                        {
                            "deliberation": p_utt.deliberation,
                            "utterance_id": p_utt.utterance_id,
                            "turn_index": p_utt.turn_index,
                            "speaker_role": p_utt.speaker_role,
                            "speaker_display_name": p_utt.speaker_display_name,
                            "spoken_dialogue": p_utt.spoken_dialogue,
                            "addressed_to": p_utt.addressed_to,
                            "bible_updates_summary": p_utt.bible_updates_summary,
                            "proof_image_paths": p_utt.proof_image_paths,
                            "artifact_emitted": p_utt.artifact_emitted,
                        },
                    )

            roundtable_summary_block = (
                "=== COUNCIL SLICE PLAYTEST ROUNDTABLE DISCUSSION ===\n"
                + "\n".join(f"  - {line}" for line in roundtable_transcript_lines)
                + "\n\n"
                if roundtable_transcript_lines
                else ""
            )
            prompt = (
                f"{assembly_shared_prefix}"
                f"{roundtable_summary_block}"
                f"You are the Council Chair & Gameplay Systems Engineer synthesizing the Slice Playtest & Whole-Game Readiness Check now that all currently queued Production Board items are approved ({assembly_pass_label}).\n"
                f"{STUDIO_CRITICAL_DISCOURSE_CHARTER}\n"
                f"{final_assembly_note}\n"
                "INSTRUCTIONS:\n"
                "1. Compare the Living Game Bible (including any Executive Producer directives), the Approved Production Board items, and the Council Slice Playtest Roundtable Discussion above against the `PLAYABLE ENTRYPOINT REACHABILITY GRAPH` "
                "and the attached HIGH-resolution Slice Playtest filmstrip/telemetry of the playable entrypoint (`run/main_scene`).\n"
                "2. Check whether any built gameplay system, progression loop, or HUD/UI widget that belongs in the active slice is still unmounted or unwired "
                "from the playable entrypoint scene tree.\n"
                "3. Skeptically evaluate `playability_and_fun_evaluation` across ALL frames of the attached Slice Playtest filmstrip:\n"
                "   - Verify that player inputs produce a responsive, human-playable rate of motion/turn/action over 1–3 seconds (never fighting a damping/stabilizer script at 1 deg/sec, and never plummeting/drifting off-screen).\n"
                "   - Verify that the playable game world is spatially legible and unobstructed (no giant static control box occluding the center of the viewport), and that a player immediately knows how to play and sees their short-term and long-term goals.\n"
                "   - Verify that the scripted playtest can actively progress through the first goal of the current slice without assertion failures.\n"
                "   - Verify that HUD widgets coexist cleanly without overlapping/overflowing text or cut-off/bleeding sprite slicing, and that there are ZERO developer debug overlays (`[DEBUG]`, `[QA ... PLAYTEST HARNESS]`, telemetry dumps) or `func test_*()` hacks in production `main.gd` / `ui_hud.gd`.\n"
                "4. In `spoken_dialogue`, address the Council candidly with zero agreement preambles or cheerleading—summarize what actually works, what remains rough, and whether we are calling the slice assembled or running a surgical cleanup pass.\n"
                "5. PRAGMATIC SLICE COMPLETION RULE: You cannot fix or expand everything in one go. If the active slice's scope (and any Executive Producer request) is assembled, responsive, playable through the first goal, and free of playtest assertion failures or glaring UI/visual breakage in `run/main_scene`, CALL THE SLICE ASSEMBLED (`backlog_additions=[]` and `is_slice_assembled=True`), leaving broader wishlist expansions for future slices.\n"
                "6. If an active-slice system is unmounted, if the playtest revealed sluggish/conflicting controls or assertion failures, if an Executive Producer directive is still unmet, or if `run/main_scene` has a high-impact visual/HUD/debug-overlay/onboarding defect, "
                "propose a small, surgical batch (1–3 tasks) of `integration` / `mechanic_code` / `ui` task(s) in `backlog_additions` BEFORE setting `is_slice_assembled=False`:\n"
                "   - Diagnose and specify systemic, world-wide root causes in `spec` (fixing authoritative world/UI/mechanic rules across all valid regions and player states)—treat any failed playtest `(x, y)` coordinate strictly as diagnostic evidence or a `proof_spec` test landmark, never as a coordinate whitelist or fixed target to hardcode in production scripts.\n"
                "   - When writing `proof_spec` for a fix task, require a QA test scene (`res://tests/test_<item_id>.tscn`) with appropriate test-only diagnostic overlays and real `input_segments` proving the fix over ~120-240 ticks.\n"
                f"Buildable kinds right now: {self._supported_kinds() or 'all'}.\n"
                "Return `SliceAssemblyDecision` with `deliberation` FIRST and `is_slice_assembled` strictly last."
            )

            hard_playtest_gate_failed = bool(playtest_assertion_failure) and (
                is_real_genai_client(self.client)
                or any(
                    tok in playtest_assertion_failure
                    for tok in (
                        "Entrypoint Parity Gate failed",
                        "Counterfactual QA check failed",
                        "QA filmstrip state assertions failed",
                        "QA Diagnostic Production Defect Report",
                    )
                )
            )

            def _validate_assembly(decision: SliceAssemblyDecision) -> List[str]:
                issues: List[str] = []
                if hard_playtest_gate_failed and decision.is_slice_assembled:
                    issues.append(
                        f"Cannot set `is_slice_assembled=True` while Automated Slice Playtest failed ({playtest_assertion_failure[:260]}). "
                        "Set `is_slice_assembled=False` and propose focused production-module fix task(s) in `backlog_additions` (targeting real `.gd` game scripts, NEVER `tests/test_*`)."
                    )
                if not decision.is_slice_assembled and not decision.backlog_additions:
                    issues.append(
                        "is_slice_assembled=False requires at least one WorkItemProposal in backlog_additions "
                        "(or set is_slice_assembled=True if no further integration tasks are needed)."
                    )
                if decision.is_slice_assembled and decision.backlog_additions:
                    decision.is_slice_assembled = False
                if decision.backlog_additions:
                    issues.extend(self.board.validate_proposals(decision.backlog_additions))
                    issues.extend(self._unsupported_kind_issues(decision.backlog_additions))
                return issues

            decision = self._structured_with_validation(
                prompt,
                SliceAssemblyDecision,
                _validate_assembly,
                temperature=0.3,
                parts=parts,
                cacheable_prefix=assembly_shared_prefix,
                min_calls_before_create=1 if run_asm_roundtable else 2,
            )
        self._emit_event(
            "slice_assembly_check_completed",
            {
                "deliberation": decision.deliberation,
                "pass_index": pass_index,
                "is_slice_assembled": decision.is_slice_assembled,
                "playability_and_fun_evaluation": getattr(decision, "playability_and_fun_evaluation", ""),
                "unmounted_or_disconnected_systems": decision.unmounted_or_disconnected_systems,
                "spawned_item_ids": [p.item_id for p in (decision.backlog_additions or [])],
            },
        )
        added: List[Any] = []
        if not decision.is_slice_assembled and decision.backlog_additions:
            default_slice_id = active_slice_obj.slice_id if active_slice_obj is not None else ""
            added = self.board.add_items(
                decision.backlog_additions,
                raised_by_role="GameplaySystemsEngineer",
                slice_id=default_slice_id,
            )

        turn_idx = len(game_bible.writers_room_transcript)
        chair_dialogue = (getattr(decision, "spoken_dialogue", "") or "").strip()
        if added:
            spawned_summary = ", ".join(f"`{it.item_id}` ({it.title})" for it in added)
            unmounted_note = (
                f" Unmounted/disconnected modules detected: {', '.join(decision.unmounted_or_disconnected_systems)}."
                if decision.unmounted_or_disconnected_systems
                else ""
            )
            play_note = (
                f" Playtest evaluation: {decision.playability_and_fun_evaluation}"
                if getattr(decision, "playability_and_fun_evaluation", "")
                else ""
            )
            spoken = (
                chair_dialogue
                or f"[Slice Playtest & Readiness Check #{pass_index + 1}] Audited scripted slice playthrough and entrypoint reachability.{unmounted_note}{play_note} "
                f"Spawning task(s): {spawned_summary}."
            )
            if chair_dialogue and spawned_summary not in chair_dialogue:
                spoken = f"{chair_dialogue} (Spawning: {spawned_summary})"
            updates_list = [f"backlog:+{it.item_id}" for it in added]
        else:
            spoken = (
                chair_dialogue
                or f"[Slice Playtest & Readiness Check #{pass_index + 1}] Verified scripted slice playthrough and entrypoint reachability—"
                f"the current slice is mounted, responsive at human scale, and playable through its opening goal."
            )
            updates_list = ["slice_assembly:verified"]

        attach_chair_proofs = [rel_proof_path] if (proof_out.is_file() and not qa_utt_emitted) else []
        chair_artifact = (
            {
                "type": "filmstrip",
                "path": rel_proof_path,
                "slice_id": active_slice_id,
                "caption": f"Slice Playtest & Assembly Check Filmstrip — {active_slice_title} (Pass #{pass_index + 1})",
            }
            if (proof_out.is_file() and not qa_utt_emitted)
            else None
        )
        assembly_utt = WritersRoomUtterance(
            deliberation=decision.deliberation,
            utterance_id=f"utt_{turn_idx:02d}_GameplaySystemsEngineer_assembly_{pass_index}",
            turn_index=turn_idx,
            speaker_role="GameplaySystemsEngineer",
            speaker_display_name="Gameplay Systems Engineer",
            spoken_dialogue=spoken,
            addressed_to=["CreativeDirector", "GameplayDesigner", "UXExperienceDesigner", "VisualArtDirector"],
            bible_updates_summary=updates_list,
            proof_image_paths=attach_chair_proofs,
            artifact_emitted=chair_artifact,
        )
        game_bible.writers_room_transcript.append(assembly_utt)
        self._emit_event(
            "writers_room_utterance",
            {
                "deliberation": assembly_utt.deliberation,
                "utterance_id": assembly_utt.utterance_id,
                "turn_index": assembly_utt.turn_index,
                "speaker_role": assembly_utt.speaker_role,
                "speaker_display_name": assembly_utt.speaker_display_name,
                "spoken_dialogue": assembly_utt.spoken_dialogue,
                "addressed_to": assembly_utt.addressed_to,
                "bible_updates_summary": assembly_utt.bible_updates_summary,
                "proof_image_paths": assembly_utt.proof_image_paths,
                "artifact_emitted": assembly_utt.artifact_emitted,
            },
        )
        self._persist_bible(game_bible, candidate_dir=candidate_dir)
        return len(added)

    def _run_end_of_slice_blind_playtest(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        candidate_dir: Optional[Path],
        pass_index: int = 0,
    ) -> int:
        """Runs an automated Council-initiated Blind AI Playtester session at the end of the current slice,
        auto-delivers the player's Experience Report and all generated wishes to the Council, convenes a
        multi-specialist Council deliberation roundtable, and decides whether to `iterate_current_slice`
        (spawning targeted backlog tasks) or `approve_slice` (closing the slice and carrying takeaways forward).
        Returns the number of new ProductionBoard items added.
        """
        if self.board is None or candidate_dir is None:
            return 0
        if not self._is_end_of_slice_blind_playtest_enabled():
            return 0

        c_dir = Path(candidate_dir).resolve()
        if not (c_dir / "project.godot").is_file():
            return 0

        self._check_aborted(c_dir)

        active_slice_obj = game_bible.get_active_slice()
        active_slice_id = active_slice_obj.slice_id if active_slice_obj is not None else "slice_0"
        active_slice_title = active_slice_obj.title if active_slice_obj is not None else "Active Slice"

        public_title = (game_bible.game_title or "Untitled").strip()
        public_pitch = (game_bible.elevator_pitch or game_bible.core_fantasy or public_title).strip()
        max_turns = int(
            getattr(self, "slice_playtest_max_turns", None)
            or os.environ.get("STUDIO_SLICE_PLAYTEST_MAX_TURNS", "45")
        )

        mgr = self.playtest_manager
        if mgr is None:
            from engine.playtest.session_manager import PlaytestSessionManager
            pt_db_path = (
                Path(self.telemetry_db.db_path)
                if (self.telemetry_db is not None and getattr(self.telemetry_db, "db_path", None) is not None)
                else (c_dir.parent.parent / "data" / "telemetry.db")
            )
            pt_db_path.parent.mkdir(parents=True, exist_ok=True)
            mgr = PlaytestSessionManager(
                db_path=pt_db_path,
                cartridges_root=c_dir.parent,
            )
            self.playtest_manager = mgr

        self._emit_activity(
            agent_role="BlindPlaytesterAgent",
            status_text=(
                f"Blind Playtester is playing '{active_slice_title}' (`{active_slice_id}`) "
                f"unguided to evaluate graphics, gameplay, sound, and intuitiveness (Pass #{pass_index + 1})..."
            ),
            active_agents=[
                "BlindPlaytesterAgent",
                "CreativeDirector",
                "UXExperienceDesigner",
                "GameplayDesigner",
                "VisualArtDirector",
                "AudioAtmosphereDesigner",
            ],
            phase="slice_blind_playtest",
        )

        detail = mgr.run_council_slice_playtest(
            candidate_dir=c_dir,
            slice_id=active_slice_id,
            game_title=public_title,
            store_description=public_pitch,
            max_turns=max_turns,
            abort_check=lambda: bool(getattr(self, "_abort_requested", False) or is_candidate_build_aborted(c_dir)),
        )
        self._check_aborted(c_dir)

        sess = detail.get("session") or {}
        session_id = str(sess.get("session_id") or "")
        report = sess.get("session_report") or {}
        impressions = sess.get("impressions") or {}
        rulebook = sess.get("rulebook") or {}
        field_notes = sess.get("field_notes") or {}
        questions = sess.get("questions") or {}
        bugs_found = sess.get("bugs_found") or {}
        wishes = detail.get("wishes") or []
        turns = detail.get("turns") or []

        # Copy up to 5 representative session/bug/wish screenshots into candidate proofs for UI & multimodal Council review
        proof_dir = c_dir / "artifacts" / "proofs"
        proof_dir.mkdir(parents=True, exist_ok=True)
        candidate_shot_sources: List[Tuple[str, Path]] = []
        seen_src: set[str] = set()

        def _push_shot(label: str, raw_p: Optional[str]) -> None:
            if not raw_p:
                return
            p_obj = Path(str(raw_p))
            if p_obj.is_file() and str(p_obj) not in seen_src:
                seen_src.add(str(p_obj))
                candidate_shot_sources.append((label, p_obj))

        if turns:
            _push_shot(f"Opening Screen (Turn #{turns[0].get('turn_index', 1)})", turns[0].get("screenshot_path"))
        if isinstance(bugs_found, dict):
            for b_item in list(bugs_found.values())[:2]:
                if isinstance(b_item, dict):
                    _push_shot(
                        f"Spotted Bug [{b_item.get('severity', 'major')}/{b_item.get('category', 'bug')}] '{b_item.get('summary', '')}'",
                        b_item.get("screenshot_path"),
                    )
        for w in wishes[:2]:
            _push_shot(
                f"Player Wish [{w.get('category', 'feedback')}] '{w.get('summary', '')}'",
                w.get("screenshot_path"),
            )
        final_reserved = 1 if len(turns) >= 2 else 0
        mid_slots = max(0, (5 - final_reserved) - len(candidate_shot_sources))
        if mid_slots > 0 and len(turns) > 2:
            for k in range(1, mid_slots + 1):
                mid_idx = int(round(k * (len(turns) - 1) / float(mid_slots + 1)))
                if 0 < mid_idx < len(turns) - 1:
                    mid_turn = turns[mid_idx]
                    _push_shot(
                        f"Mid-Session Progress (Turn #{mid_turn.get('turn_index', mid_idx + 1)})",
                        mid_turn.get("screenshot_path"),
                    )
        if len(turns) >= 2:
            _push_shot(f"Final Session Screen (Turn #{turns[-1].get('turn_index', len(turns))})", turns[-1].get("screenshot_path"))

        rel_proof_paths: List[str] = []
        multimodal_parts: List[Any] = []
        for idx, (shot_label, src_path) in enumerate(candidate_shot_sources[:5], start=1):
            dest_name = f"blind_playtest_{active_slice_id}_p{pass_index}_shot{idx}.png"
            dest_path = proof_dir / dest_name
            try:
                shutil.copy2(src_path, dest_path)
                rel_p = f"artifacts/proofs/{dest_name}"
                rel_proof_paths.append(rel_p)
                img_part = self._load_image_part(dest_path)
                if img_part is not None:
                    multimodal_parts.append(f"BLIND PLAYTESTER SCREENSHOT #{idx} — {shot_label}:")
                    multimodal_parts.append(img_part)
            except Exception as copy_exc:
                logger.debug("Skipping blind playtester screenshot copy %s: %s", src_path, copy_exc)

        fun_rating = report.get("overall_fun_rating", "?")
        overall_summary = str(report.get("overall_experience_summary") or sess.get("end_summary") or "Session completed.")
        wishes_bullet_lines = [
            f"- [{w.get('category', 'general')}] {w.get('summary', '')} (Turn #{w.get('turn_index', '?')}, mood={w.get('mood_at_time', 'n/a')}): {w.get('rationale', '')}"
            for w in wishes
        ]
        rep_visual_defects = report.get("visual_and_ui_defects_observed") or []
        rep_highlights = report.get("key_highlights") or []
        rep_pain_points = report.get("key_pain_points") or []
        rep_bugs_observed = report.get("bugs_and_glitches_observed") or []
        rep_unanswered_qs = report.get("unanswered_questions_and_mysteries") or []

        player_dialogue_lines = [
            f"[End-of-Slice Blind Playtest Report — Fun Rating: {fun_rating}/10 | {sess.get('turn_count', len(turns))} turns | status={sess.get('status')}]",
            overall_summary,
            f"• Graphics & Visuals: {report.get('graphics_and_visuals', '(not reported)')}",
            f"• Gameplay & Feel: {report.get('gameplay_and_feel', '(not reported)')}",
            f"• Sound & Audio: {report.get('sound_and_audio', '(not reported)')}",
            f"• Intuitiveness & Onboarding: {report.get('intuitiveness_and_onboarding', '(not reported)')}",
            f"• Goals & Pacing: {report.get('goals_pacing_and_accomplishment', '(not reported)')}",
            f"• World & Atmosphere: {report.get('world_and_atmosphere', '(not reported)')}",
        ]
        if rep_visual_defects:
            player_dialogue_lines.append("Visual & UI Defects Observed:")
            player_dialogue_lines.extend(f"- {vd}" for vd in rep_visual_defects)
        if rep_highlights:
            player_dialogue_lines.append("Key Highlights:")
            player_dialogue_lines.extend(f"- {h}" for h in rep_highlights)
        if rep_pain_points:
            player_dialogue_lines.append("Key Pain Points:")
            player_dialogue_lines.extend(f"- {p}" for p in rep_pain_points)
        if rep_bugs_observed:
            player_dialogue_lines.append("Bugs & Glitches Observed:")
            player_dialogue_lines.extend(f"- {b}" for b in rep_bugs_observed)
        elif isinstance(bugs_found, dict) and bugs_found:
            player_dialogue_lines.append("Bugs & Glitches Spotted Along The Way:")
            for bk, bv in bugs_found.items():
                if isinstance(bv, dict):
                    player_dialogue_lines.append(
                        f"- [{bv.get('severity', 'major').upper()}/{bv.get('category', 'bug')}] {bv.get('summary', bk)}: {bv.get('details', '')}"
                    )
        if rep_unanswered_qs:
            player_dialogue_lines.append("Unanswered Player Questions:")
            player_dialogue_lines.extend(f"- {q}" for q in rep_unanswered_qs)
        if wishes_bullet_lines:
            player_dialogue_lines.append(f"Auto-Delivered Player Wishes ({len(wishes_bullet_lines)}):")
            player_dialogue_lines.extend(wishes_bullet_lines)

        pt_idx = len(game_bible.writers_room_transcript)
        pt_utt = WritersRoomUtterance(
            deliberation=str(report.get("deliberation") or f"Blind Playtester completed session {session_id}."),
            utterance_id=f"utt_{pt_idx:02d}_BlindPlaytester_{active_slice_id}_{pass_index}",
            turn_index=pt_idx,
            speaker_role="BlindPlaytester",
            speaker_display_name="Blind AI Playtester (Session Report)",
            spoken_dialogue="\n".join(player_dialogue_lines),
            addressed_to=[
                "CreativeDirector",
                "UXExperienceDesigner",
                "GameplayDesigner",
                "VisualArtDirector",
                "AudioAtmosphereDesigner",
            ],
            bible_updates_summary=[
                f"blind_playtest:{session_id}:fun_{fun_rating}_of_10",
                f"blind_wishes_delivered:{len(wishes)}",
            ],
            proof_image_paths=rel_proof_paths,
        )
        game_bible.writers_room_transcript.append(pt_utt)
        self._emit_event(
            "writers_room_utterance",
            {
                "deliberation": pt_utt.deliberation,
                "utterance_id": pt_utt.utterance_id,
                "turn_index": pt_utt.turn_index,
                "speaker_role": pt_utt.speaker_role,
                "speaker_display_name": pt_utt.speaker_display_name,
                "spoken_dialogue": pt_utt.spoken_dialogue,
                "addressed_to": pt_utt.addressed_to,
                "bible_updates_summary": pt_utt.bible_updates_summary,
                "proof_image_paths": pt_utt.proof_image_paths,
                "artifact_emitted": pt_utt.artifact_emitted,
            },
        )

        # Build shared dossier for Council deliberation
        curated_code_block = ""
        if is_real_genai_client(self.client):
            try:
                from engine.studio.context_curator import ContextCuratorAgent
                curator = ContextCuratorAgent(
                    cartridge_dir=c_dir,
                    client=self.client,
                    model_name=self.model_name,
                    board=self.board,
                    game_bible=game_bible,
                    event_callback=self.event_callback,
                )
                pack = curator.curate_context(
                    target_agent_role="council_reviewer:BlindPlaytestDeliberation",
                    task_request=(
                        f"Inspect the active playable modules for '{active_slice_title}' (`{active_slice_id}`) "
                        f"so the Council can deliberate on the Blind Playtester's Experience Report "
                        f"(fun={fun_rating}/10) and {len(wishes)} auto-delivered player wish(es)."
                    ),
                    item_id=f"blind_playtest_{active_slice_id}_p{pass_index}",
                    item_kind="integration",
                )
                curated_code_block = f"{pack.prompt_block}\n"
            except Exception as cur_exc:
                logger.warning("ContextCurator blind playtest curation skipped: %s", cur_exc)

        rulebook_lines = (
            "\n".join(f"  - {k}: {v}" for k, v in rulebook.items())
            if rulebook
            else "  (The Blind Playtester could not confirm any controls/rules!)"
        )
        field_notes_lines = (
            "\n".join(f"  - {k}: {v}" for k, v in field_notes.items())
            if isinstance(field_notes, dict) and field_notes
            else "  (No separate field notes logged.)"
        )
        if isinstance(questions, dict) and questions:
            q_lines_list: List[str] = []
            for qk, qv in questions.items():
                if isinstance(qv, dict):
                    st = str(qv.get("status") or "open").upper()
                    ans = f" -> {qv['answer']}" if qv.get("answer") else ""
                    q_lines_list.append(f"  - [{st}] ({qk}) {qv.get('question', '')}{ans}")
            questions_lines = "\n".join(q_lines_list) if q_lines_list else "  (No player questions logged.)"
        else:
            questions_lines = "  (No player questions logged.)"
        if isinstance(bugs_found, dict) and bugs_found:
            b_lines_list: List[str] = []
            for bk, bv in bugs_found.items():
                if isinstance(bv, dict):
                    b_lines_list.append(
                        f"  - [{bv.get('severity', 'major').upper()} | {bv.get('category', 'bug')}] "
                        f"(Turn #{bv.get('turn_index', '?')}, `{bk}`) {bv.get('summary', '')}: {bv.get('details', '')}"
                    )
            bugs_found_lines = "\n".join(b_lines_list) if b_lines_list else "  (No live bugs logged during play.)"
        else:
            bugs_found_lines = "  (No live bugs logged during play.)"
        impressions_lines = (
            "\n".join(f"  - {k}: {v}" for k, v in impressions.items())
            if impressions
            else "  (Synthesized directly in end-of-session Experience Report below.)"
        )
        wishes_block = (
            "\n".join(
                f"  {idx}. [{w.get('category')}] {w.get('summary')} (Turn #{w.get('turn_index')}, mood={w.get('mood_at_time')}): {w.get('rationale')}"
                for idx, w in enumerate(wishes, start=1)
            )
            if wishes
            else "  (No explicit player wishes were filed during this session.)"
        )
        tape_entries: List[str] = []
        for t_row in turns:
            if isinstance(t_row, dict):
                line_txt = str(t_row.get("append_log_line") or "").strip()
                if line_txt:
                    tape_entries.append(f"  {line_txt}")
        action_tape_block = (
            "\n".join(tape_entries)
            if tape_entries
            else "  (No per-turn action tape recorded.)"
        )
        approved_board_lines = [
            f"  - [{it.item_id}] ({it.kind}) '{it.title}' | target_modules={it.target_modules}"
            for it in self.board.items()
            if it.state == "approved"
        ]

        blind_asset_block, _, blind_sheet_part = self._build_asset_integrity_dossier_and_sheet(
            c_dir, f"{active_slice_id}_p{pass_index}"
        )
        if blind_sheet_part is not None:
            multimodal_parts.append(
                f"CARTRIDGE VISUAL ASSET CONTACT SHEET FOR `{active_slice_id}` (Pass #{pass_index + 1} — "
                "active in-game textures listed first; red card borders mark deterministic or multimodal-critic-flagged "
                "broken/placeholder assets; inspect every tile against the Game Bible style guide and concept mockups):"
            )
            multimodal_parts.append(blind_sheet_part)

        blind_bible_snapshot = self._format_bible_snapshot(
            game_bible,
            omit_playtest_session_ids={str(session_id)} if session_id else None,
        )
        blind_dna_header = self._dna_header_if_absent(dna_summary, blind_bible_snapshot)
        blind_dossier = (
            f"{blind_dna_header}"
            f"{blind_bible_snapshot}\n\n"
            f"{curated_code_block}"
            f"=== BLIND AI PLAYTESTER END-OF-SLICE SESSION REPORT (`{session_id}` | Pass #{pass_index + 1}) ===\n"
            f"Active Slice Under Evaluation: `{active_slice_id}` — '{active_slice_title}'\n"
            f"Session Status: {sess.get('status')} (outcome={sess.get('end_outcome')}, turns={sess.get('turn_count')}, sim_duration={sess.get('sim_duration_sec')}s)\n"
            f"Overall Fun Rating: {fun_rating}/10\n"
            f"Overall Experience Summary: {overall_summary}\n"
            f"Visual & UI Defects Observed: {json.dumps(report.get('visual_and_ui_defects_observed') or [])}\n"
            f"Graphics & Visuals: {report.get('graphics_and_visuals', '')}\n"
            f"Gameplay & Feel: {report.get('gameplay_and_feel', '')}\n"
            f"Sound & Audio: {report.get('sound_and_audio', '')}\n"
            f"Intuitiveness & Onboarding: {report.get('intuitiveness_and_onboarding', '')}\n"
            f"Goals, Pacing & Accomplishment: {report.get('goals_pacing_and_accomplishment', '')}\n"
            f"World & Atmosphere: {report.get('world_and_atmosphere', '')}\n"
            f"Key Highlights: {json.dumps(report.get('key_highlights') or [])}\n"
            f"Key Pain Points: {json.dumps(report.get('key_pain_points') or [])}\n"
            f"Bugs & Glitches Observed: {json.dumps(report.get('bugs_and_glitches_observed') or [])}\n"
            f"Unanswered Questions & Mysteries: {json.dumps(report.get('unanswered_questions_and_mysteries') or [])}\n\n"
            f"=== BLIND PLAYER CHRONOLOGICAL ACTION TAPE ({len(tape_entries)} turns) ===\n{action_tape_block}\n\n"
            f"=== BUGS & GLITCHES SPOTTED ALONG THE WAY ===\n{bugs_found_lines}\n\n"
            f"=== PLAYER FIELD NOTES (SCREEN & WORLD OBSERVATIONS) ===\n{field_notes_lines}\n\n"
            f"=== PLAYER QUESTIONS & DISCOVERED ANSWERS ===\n{questions_lines}\n\n"
            f"=== CONTROLS & RULES THE BLIND PLAYER ACTUALLY FIGURED OUT ===\n{rulebook_lines}\n\n"
            f"=== SPARSE RUNNING IMPRESSIONS DURING PLAY ===\n{impressions_lines}\n\n"
            f"=== AUTO-DELIVERED BLIND PLAYER WISHES ({len(wishes)} items) ===\n{wishes_block}\n\n"
            f"{blind_asset_block}"
            f"=== CURRENTLY APPROVED PRODUCTION BOARD ITEMS ({len(approved_board_lines)}) ===\n"
            + ("\n".join(approved_board_lines) or "  (none)")
        )

        blind_shared_prefix = f"{blind_dossier}\n\n"
        is_final_playtest = bool(getattr(self, "_final_verification_blind_playtest_pass", False))
        max_pt_passes = int(
            getattr(self, "max_blind_playtest_passes", os.environ.get("STUDIO_MAX_BLIND_PLAYTEST_PASSES", "3"))
        )
        blind_pass_label = (
            f"Post-Remediation Final Verification Pass after {max_pt_passes} iteration round(s)"
            if is_final_playtest
            else f"Pass #{pass_index + 1} of {max_pt_passes}"
        )
        final_playtest_note = (
            "\nPOST-REMEDIATION FINAL VERIFICATION PASS: All current-slice iteration rounds have been spent, and this session "
            "verifies the repaired build. Do NOT queue speculative feature additions or minor polish—capture all non-blocking "
            "improvements in `next_slice_carryover_notes` and choose `approve_slice` (`recommends_slice_approval=True`) if the "
            "repaired slice boots cleanly, core player actions function, and active visual/audio assets are intact; choose "
            "`iterate_current_slice` (`recommends_slice_approval=False`) ONLY if a severe slice-blocking defect (crash, dead-end "
            "broken controls, or corrupted/placeholder active visual assets) still breaks playability.\n"
            if is_final_playtest
            else ""
        )
        roundtable_lines: List[str] = []
        run_roundtable = getattr(self, "enable_blind_playtest_roundtable", None)
        if run_roundtable is None:
            run_roundtable = True

        with explicit_prefix_cache_scope(self.client, ttl_seconds=600, reuse_outer=False):
            if run_roundtable:
                panel_roles = [
                    "UXExperienceDesigner",
                    "GameplayDesigner",
                    "VisualArtDirector",
                    "AudioAtmosphereDesigner",
                ]
                self._emit_activity(
                    agent_role="CouncilReviewPanel",
                    status_text=(
                        f"Council specialists ({', '.join(panel_roles)}) are deliberating on the Blind Playtester's "
                        f"Experience Report & {len(wishes)} wish(es)..."
                    ),
                    active_agents=panel_roles + ["CreativeDirector"],
                    phase="slice_blind_playtest",
                )
                for p_role in panel_roles:
                    spec = next((sp for sp in SPECIALIST_ROSTER if sp["role"] == p_role), None)
                    disp_name = spec["display_name"] if spec else p_role
                    mandate = self._role_review_mandate(p_role)
                    prior_chat = (
                        "\n".join(f"  - {line}" for line in roundtable_lines)
                        if roundtable_lines
                        else "  (You are opening the Council's deliberation right after reading the Blind Playtester's report.)"
                    )
                    turn_prompt = (
                        f"{blind_shared_prefix}"
                        f"=== COUNCIL BLIND PLAYTEST DELIBERATION SO FAR ===\n{prior_chat}\n\n"
                        f"You are the {disp_name} ({p_role}) deliberating in the Writers' Room on the unguided Blind AI Playtester's "
                        f"Experience Report and auto-delivered wishes for '{active_slice_title}' (`{active_slice_id}`, {blind_pass_label}).\n"
                        f"{STUDIO_CRITICAL_DISCOURSE_CHARTER}\n"
                        f"{final_playtest_note}\n"
                        f"{mandate}\n"
                        "INSTRUCTIONS:\n"
                        "1. Remember: the Blind Playtester is a fast non-thinking player model playing cold with ZERO internal instructions. "
                        "Do NOT take a high `overall_fun_rating` or optimistic report prose at face value! Independently audit the attached "
                        "HIGH-resolution screenshots, `BLIND PLAYER CHRONOLOGICAL ACTION TAPE`, and curated source code (and cater with full respect to any Executive Producer directives).\n"
                        "2. Specifically verify (`domain_reaction_to_blind_player`):\n"
                        "   - Did visible objectives/counters actually progress through genuine player decisions, or did the player get stuck in "
                        "repetitive no-op clicks, hallucinate that decorative UI was interactive, or leave core slice loops unexercised?\n"
                        "   - Did any mechanics 'play themselves' via hardcoded target coordinates, timer-driven inventory shortcuts, or fake "
                        "progression credits instead of real player agency?\n"
                        "   - Are there visual/UI craft defects in the HIGH-resolution screenshots—cut-off sprites or stray bleeding edge "
                        "slivers from wrong sprite-sheet slicing, squished/distorted UI icons, text overflowing panel borders or truncated "
                        "mid-word, overlapping/duplicate HUD widgets, harsh unblended rectangular texture boxes inside shapes, raw untextured "
                        "placeholder polygons clashing with detailed sprites, or silent/dropped audio?\n"
                        "3. PRAGMATIC SLICE SCOPE & COMPLETION DISCIPLINE — distinguish sharply between:\n"
                        "   - BOUNDED CURRENT-SLICE CLEANUPS (`priority_fixes_or_refinements` — keep tight, 0–3 surgical items): unmet active-slice scope or Executive Producer direction, "
                        "onboarding/control dead-ends, repetitive no-op traps, auto-playing/hardcoded shortcuts bypassing player agency, glaring sprite-slicing/UI-overflow/overlap glitches, or silent core audio.\n"
                        "   - NEXT-SLICE CARRYOVER WISHES: broader mechanics, deeper content tiers, secondary polish, or new world features that we CANNOT and SHOULD NOT try to cram into this slice in one go.\n"
                        "4. In `spoken_dialogue`, speak candidly and skeptically to your peers—NEVER open with 'I completely agree with...', 'I echo...', or peer roll-call flattery, and never repeat what an earlier speaker already caught. "
                        "Set `recommends_slice_approval=False` if targeted current-slice scope or high-impact problem cleanup is needed right now; "
                        "set `recommends_slice_approval=True` once the active slice's scope is covered and major problem areas are cleaned up (calling the slice done when appropriate while candidly naming what remains rough for future slices).\n"
                        "Provide `deliberation` FIRST and `recommends_slice_approval` strictly LAST."
                    )
                    turn_dec = self._structured_with_validation(
                        turn_prompt,
                        BlindPlaytestCouncilTurn,
                        lambda _t: [],
                        temperature=0.35,
                        parts=multimodal_parts,
                        cacheable_prefix=blind_shared_prefix,
                        min_calls_before_create=1,
                    )
                    roundtable_lines.append(
                        f"[{disp_name} ({p_role})] (recommends_approval={turn_dec.recommends_slice_approval}): "
                        f"{turn_dec.spoken_dialogue} | Priority fixes: {turn_dec.priority_fixes_or_refinements}"
                    )
                    t_idx = len(game_bible.writers_room_transcript)
                    c_utt = WritersRoomUtterance(
                        deliberation=turn_dec.deliberation,
                        utterance_id=f"utt_{t_idx:02d}_{p_role}_blind_playtest_{pass_index}",
                        turn_index=t_idx,
                        speaker_role=p_role,
                        speaker_display_name=disp_name,
                        spoken_dialogue=turn_dec.spoken_dialogue,
                        addressed_to=turn_dec.addressed_to or ["CreativeDirector"],
                        bible_updates_summary=[
                            f"blind_playtest_vote:{p_role}:{'approve' if turn_dec.recommends_slice_approval else 'iterate'}"
                        ],
                    )
                    game_bible.writers_room_transcript.append(c_utt)
                    self._emit_event(
                        "writers_room_utterance",
                        {
                            "deliberation": c_utt.deliberation,
                            "utterance_id": c_utt.utterance_id,
                            "turn_index": c_utt.turn_index,
                            "speaker_role": c_utt.speaker_role,
                            "speaker_display_name": c_utt.speaker_display_name,
                            "spoken_dialogue": c_utt.spoken_dialogue,
                            "addressed_to": c_utt.addressed_to,
                            "bible_updates_summary": c_utt.bible_updates_summary,
                            "proof_image_paths": c_utt.proof_image_paths,
                            "artifact_emitted": c_utt.artifact_emitted,
                        },
                    )

            roundtable_block = (
                "=== COUNCIL ROUNDTABLE DISCUSSION ON BLIND PLAYTEST ===\n"
                + "\n".join(f"  - {line}" for line in roundtable_lines)
                + "\n\n"
                if roundtable_lines
                else ""
            )
            verdict_prompt = (
                f"{blind_shared_prefix}"
                f"{roundtable_block}"
                f"You are the Creative Director chairing the Council's End-of-Slice Blind Playtest Verdict for "
                f"'{active_slice_title}' (`{active_slice_id}`, {blind_pass_label}).\n"
                f"{STUDIO_CRITICAL_DISCOURSE_CHARTER}\n"
                f"{final_playtest_note}\n"
                "INSTRUCTIONS:\n"
                "1. Independently and skeptically audit the Blind Playtester's HIGH-resolution screenshots, `BLIND PLAYER CHRONOLOGICAL ACTION TAPE`, curated source code, "
                "6-pillar Experience Report, `overall_fun_rating`, discovered controls rulebook, wishes, any Executive Producer directives, and the Council Roundtable Discussion. "
                "Never rubber-stamp an inflated `overall_fun_rating`, and never open `spoken_dialogue` with 'I completely agree with the Council' or cheerleading.\n"
                "2. APPLY PRAGMATIC SLICE COMPLETION JUDGMENT (WE CANNOT CHANGE EVERYTHING IN ONE GO) — decide between two clear moves:\n"
                "   - `iterate_current_slice`: Choose this when active-slice scope or an Executive Producer directive is unmet, OR when high-impact problem areas "
                "(onboarding dead-ends, repetitive no-op loops, stalled progression, auto-playing/shortcutted mechanics bypassing player agency, glaring sprite-bleed/UI-overflow/overlap glitches, "
                "corrupted/placeholder visual assets on the `CARTRIDGE VISUAL ASSET CONTACT SHEET`, or silent core audio) need surgical cleanup before closing this slice. Keep gameplay/UX `backlog_additions` "
                "tight and surgical (1–3 focused gameplay/UX `WorkItemProposal` tasks—never boil the ocean with a massive redesign list!), PLUS any atomic in-place `visual_asset` regeneration tasks and "
                "dependent `.gd` reintegration task required by `CARTRIDGE VISUAL ASSET & GDSCRIPT SLICING INTEGRITY AUDIT`, targeting real production modules with systemic, world-wide fixes (never single-coordinate point patches) "
                "and concrete `proof_spec` verification, and defer broader feature wishes to `next_slice_carryover_notes`.\n"
                "   - `approve_slice`: CALL THE SLICE DONE when the screenshots, action tape, and code confirm that THIS slice's intended scope (and any Executive Producer request) "
                "is genuinely covered and playable through real player decisions and major problem areas have been cleaned up—even if non-blocking rough edges or broader wishes remain! "
                "Be candid in `spoken_dialogue` about what is still rough, capture those deferred improvements and wishes cleanly in `next_slice_carryover_notes`, and set `backlog_additions=[]`.\n"
                f"Buildable kinds right now: {self._supported_kinds() or 'all'}.\n"
                "Provide `deliberation` FIRST and `verdict` (`approve_slice` or `iterate_current_slice`) strictly LAST."
            )

            def _validate_blind_verdict(v: BlindPlaytestCouncilVerdict) -> List[str]:
                issues: List[str] = []
                if v.verdict == "iterate_current_slice" and not v.backlog_additions:
                    issues.append(
                        "verdict='iterate_current_slice' requires at least one WorkItemProposal in backlog_additions "
                        "(or set verdict='approve_slice' if no current-slice tasks are needed)."
                    )
                if v.verdict == "approve_slice" and v.backlog_additions:
                    v.verdict = "iterate_current_slice"
                if v.backlog_additions:
                    issues.extend(self.board.validate_proposals(v.backlog_additions))
                    issues.extend(self._unsupported_kind_issues(v.backlog_additions))
                return issues

            self._emit_activity(
                agent_role="CreativeDirector",
                status_text=f"Creative Director synthesizing Council Blind Playtest Verdict for '{active_slice_title}'...",
                active_agents=["CreativeDirector"],
                phase="slice_blind_playtest",
            )
            verdict = self._structured_with_validation(
                verdict_prompt,
                BlindPlaytestCouncilVerdict,
                _validate_blind_verdict,
                temperature=0.3,
                parts=multimodal_parts,
                caller_role="CreativeDirector",
                candidate_dir=c_dir,
                slice_id=active_slice_id,
                cacheable_prefix=blind_shared_prefix,
                min_calls_before_create=1 if run_roundtable else 2,
            )

        added: List[Any] = []
        if verdict.verdict == "iterate_current_slice" and verdict.backlog_additions:
            added = self.board.add_items(
                verdict.backlog_additions,
                raised_by_role="CreativeDirector",
                slice_id=active_slice_id,
            )

        # Persist end-of-slice Blind Playtest artifact so next-slice reflection inherits player report + carryover notes,
        # AND preserve immutable per-pass + per-session records so every historical run and verdict remains watchable.
        artifacts_dir = c_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        saved_report_payload = {
            "session_id": session_id,
            "slice_id": active_slice_id,
            "slice_title": active_slice_title,
            "pass_index": pass_index,
            "initiated_by": "council",
            "session_report": report,
            "impressions": impressions,
            "rulebook": rulebook,
            "field_notes": field_notes,
            "questions": questions,
            "bugs_found": bugs_found,
            "wishes": wishes,
            "council_verdict": verdict.verdict,
            "player_experience_synthesis": verdict.player_experience_synthesis,
            "next_slice_carryover_notes": list(verdict.next_slice_carryover_notes or []),
            "spawned_item_ids": [it.item_id for it in added],
        }
        report_json_str = json.dumps(saved_report_payload, indent=2)
        (artifacts_dir / f"blind_playtest_report_{active_slice_id}.json").write_text(
            report_json_str,
            encoding="utf-8",
        )
        (artifacts_dir / f"blind_playtest_report_{active_slice_id}_pass{pass_index}.json").write_text(
            report_json_str,
            encoding="utf-8",
        )
        if session_id and hasattr(mgr, "record_council_verdict"):
            try:
                mgr.record_council_verdict(
                    session_id,
                    slice_id=active_slice_id,
                    slice_title=active_slice_title,
                    pass_index=pass_index,
                    verdict=verdict.verdict,
                    player_experience_synthesis=verdict.player_experience_synthesis,
                    spawned_item_ids=[it.item_id for it in added],
                    next_slice_carryover_notes=list(verdict.next_slice_carryover_notes or []),
                )
            except Exception as rec_exc:
                logger.warning("Could not persist Council Blind Playtest verdict onto session %s: %s", session_id, rec_exc)

        self._emit_event(
            "council_blind_playtest_deliberation_completed",
            {
                "deliberation": verdict.deliberation,
                "session_id": session_id,
                "slice_id": active_slice_id,
                "pass_index": pass_index,
                "overall_fun_rating": fun_rating,
                "wishes_count": len(wishes),
                "verdict": verdict.verdict,
                "spawned_item_ids": [it.item_id for it in added],
                "next_slice_carryover_notes": list(verdict.next_slice_carryover_notes or []),
            },
        )

        dir_idx = len(game_bible.writers_room_transcript)
        if added:
            spawned_str = ", ".join(f"`{it.item_id}` ({it.title})" for it in added)
            dir_spoken = verdict.spoken_dialogue.strip() or (
                f"[Council Blind Playtest Verdict: ITERATE CURRENT SLICE] {verdict.player_experience_synthesis}"
            )
            if spawned_str not in dir_spoken:
                dir_spoken = f"{dir_spoken} (Spawning: {spawned_str})"
            dir_updates = [f"backlog:+{it.item_id}" for it in added]
        else:
            dir_spoken = verdict.spoken_dialogue.strip() or (
                f"[Council Blind Playtest Verdict: APPROVE SLICE] {verdict.player_experience_synthesis}"
            )
            dir_updates = [f"blind_playtest_approved:{active_slice_id}"]

        dir_utt = WritersRoomUtterance(
            deliberation=verdict.deliberation,
            utterance_id=f"utt_{dir_idx:02d}_CreativeDirector_blind_verdict_{pass_index}",
            turn_index=dir_idx,
            speaker_role="CreativeDirector",
            speaker_display_name="Creative Director",
            spoken_dialogue=dir_spoken,
            addressed_to=[
                "GameplayDesigner",
                "UXExperienceDesigner",
                "VisualArtDirector",
                "AudioAtmosphereDesigner",
                "GameplaySystemsEngineer",
            ],
            bible_updates_summary=dir_updates,
        )
        game_bible.writers_room_transcript.append(dir_utt)
        self._emit_event(
            "writers_room_utterance",
            {
                "deliberation": dir_utt.deliberation,
                "utterance_id": dir_utt.utterance_id,
                "turn_index": dir_utt.turn_index,
                "speaker_role": dir_utt.speaker_role,
                "speaker_display_name": dir_utt.speaker_display_name,
                "spoken_dialogue": dir_utt.spoken_dialogue,
                "addressed_to": dir_utt.addressed_to,
                "bible_updates_summary": dir_utt.bible_updates_summary,
                "proof_image_paths": dir_utt.proof_image_paths,
                "artifact_emitted": dir_utt.artifact_emitted,
            },
        )
        self._persist_bible(game_bible, candidate_dir=candidate_dir)

        if (
            added
            and not is_final_playtest
            and (is_real_genai_client(self.client) or getattr(self, "enable_task_shaping_on_playtest_iteration", False))
        ):
            self._run_council_task_shaping_round(game_bible, dna_summary, candidate_dir=candidate_dir)

        return len(added)

    def _run_slice_qa_gate_check(
        self,
        game_bible: GameBibleSpec,
        candidate_dir: Optional[Path],
        pass_index: int = 0,
    ) -> int:
        """Runs 5-Gate QA verification (Gates A–E) on the slice candidate before marking the slice completed.
        When any gate fails, posts a tracked `WorkItemProposal` onto `self.board` so remediation executes
        visibly through Task Lead + ContextCurator + Specialist Council review rather than off-board loops.
        """
        if (
            self.board is None
            or candidate_dir is None
            or not (Path(candidate_dir) / "project.godot").is_file()
        ):
            return 0
        evaluator_override = getattr(self, "qa_gate_evaluator_fn", None)
        if (
            not getattr(self, "enable_slice_qa_gate_check", False)
            and evaluator_override is None
        ):
            return 0

        c_dir = Path(candidate_dir).resolve()
        active_slice_obj = game_bible.get_active_slice()
        active_slice_id = active_slice_obj.slice_id if active_slice_obj is not None else "slice_0"
        active_slice_title = active_slice_obj.title if active_slice_obj is not None else "Active Slice"

        self._emit_activity(
            agent_role="QATestEngineer",
            status_text=(
                f"Running 5-Gate QA verification (Syntax, Assets, 120-Tick Headless Sim, 60 FPS Frame Budget, Telemetry) "
                f"for '{active_slice_title}' (Pass #{pass_index + 1})..."
            ),
            active_agents=["QATestEngineer"],
            phase="slice_qa_gate_check",
        )

        from engine.supervisor.supervisor import BuildSupervisor, build_qa_gate_work_item_proposals

        if callable(evaluator_override):
            gate_results, trace = evaluator_override(c_dir, pass_index)
        else:
            supervisor = BuildSupervisor(
                cartridges_root=c_dir.parent,
                active_cartridge_dir=c_dir.parent / "active",
                swe_agent=None,
                gemini_client=self.client,
                event_bus=None,
            )
            gate_results, trace = supervisor.evaluate_candidate_cartridge(
                candidate_dir=c_dir,
                episode_number=getattr(game_bible, "target_episode", 0) or 0,
                attempt_number=pass_index + 1,
                include_challenger_gate=False,
            )

        if trace is None:
            self._emit_event(
                "slice_qa_gate_check_completed",
                {
                    "deliberation": (
                        f"All 5 Quality Gates (Syntax, Asset Integrity, Headless Simulation, Frame Budget, Telemetry) "
                        f"passed cleanly for '{active_slice_title}' ({active_slice_id}) on Pass #{pass_index + 1}."
                    ),
                    "slice_id": active_slice_id,
                    "pass_index": pass_index,
                    "passed": True,
                    "spawned_item_ids": [],
                },
            )
            return 0

        existing_ids = {it.item_id for it in self.board.items()}
        supported_set = set(self._supported_kinds())
        proposals = build_qa_gate_work_item_proposals(
            candidate_dir=c_dir,
            trace=trace,
            pass_index=pass_index,
            slice_id=active_slice_id,
            existing_item_ids=existing_ids,
            supported_kinds=supported_set if supported_set else None,
        )
        added = self.board.add_items(
            proposals,
            raised_by_role="QATestEngineer",
            slice_id=active_slice_id,
        )
        spawned_ids = [it.item_id for it in added]
        lead_proposal = proposals[0]
        combined_targets: List[str] = []
        for p in proposals:
            for tm in p.target_modules:
                if tm not in combined_targets:
                    combined_targets.append(tm)
        addressed_roles = ["GameplaySystemsEngineer", "GameplayDesigner", "CreativeDirector"]
        if any(p.kind == "visual_asset" for p in proposals):
            addressed_roles.insert(0, "VisualArtDirector")
        fail_type_str = trace.failure_type.value if hasattr(trace.failure_type, "value") else str(trace.failure_type)
        turn_idx = len(game_bible.writers_room_transcript)
        task_desc_str = ", ".join(f"`{p.item_id}` ({p.kind})" for p in proposals)
        spoken = (
            f"[5-Gate QA Verification — {active_slice_title} (Pass #{pass_index + 1})] "
            f"{trace.failure_gate} ({fail_type_str}) flagged a blocking defect: {trace.failure_reason} "
            f"Spawning {len(proposals)} Production Board remediation task(s) {task_desc_str} targeting "
            f"{', '.join(combined_targets[:6])} for Task Lead execution and Council sign-off."
        )
        qa_utt = WritersRoomUtterance(
            deliberation=lead_proposal.deliberation,
            utterance_id=f"utt_{turn_idx:02d}_QATestEngineer_qa_gate_{pass_index}",
            turn_index=turn_idx,
            speaker_role="QATestEngineer",
            speaker_display_name="QA & Build Supervisor",
            spoken_dialogue=spoken,
            addressed_to=addressed_roles,
            bible_updates_summary=[f"backlog:+{it_id}" for it_id in spawned_ids],
        )
        game_bible.writers_room_transcript.append(qa_utt)
        self._persist_bible(game_bible, candidate_dir=candidate_dir)
        self._emit_event(
            "writers_room_utterance",
            {
                "deliberation": qa_utt.deliberation,
                "utterance_id": qa_utt.utterance_id,
                "turn_index": qa_utt.turn_index,
                "speaker_role": qa_utt.speaker_role,
                "speaker_display_name": qa_utt.speaker_display_name,
                "spoken_dialogue": qa_utt.spoken_dialogue,
                "addressed_to": qa_utt.addressed_to,
                "bible_updates_summary": qa_utt.bible_updates_summary,
                "proof_image_paths": qa_utt.proof_image_paths,
                "artifact_emitted": qa_utt.artifact_emitted,
            },
        )
        self._emit_event(
            "slice_qa_gate_check_completed",
            {
                "deliberation": lead_proposal.deliberation,
                "slice_id": active_slice_id,
                "pass_index": pass_index,
                "passed": False,
                "failure_gate": trace.failure_gate,
                "failure_type": fail_type_str,
                "failure_reason": trace.failure_reason,
                "spawned_item_ids": spawned_ids,
            },
        )
        return len(added)

    def _drain_backlog(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        candidate_dir: Optional[Path],
    ) -> None:
        """Blocks until every backlog item is approved or cancelled, running review rounds as proofs arrive."""
        if self.board is None:
            return
        steered_before_shaping = self._consume_pending_steering(game_bible, candidate_dir=candidate_dir)
        if steered_before_shaping:
            self._execute_steering_pivot_turns(game_bible, dna_summary, candidate_dir, steered_before_shaping)
        has_active_flight = any(
            i.state in ("in_progress", "waiting_for_lock", "awaiting_review", "changes_requested")
            or (i.state in ("ready", "blocked") and self._has_started_execution(i, candidate_dir=candidate_dir))
            for i in self.board.items()
        )
        has_unshaped_unstarted = any(
            i.state in ("ready", "blocked")
            and not self._has_started_execution(i, candidate_dir=candidate_dir)
            and not (i.shaping_notes or [])
            for i in self.board.items()
        )
        if not has_active_flight and has_unshaped_unstarted:
            self._run_council_task_shaping_round(game_bible, dna_summary, candidate_dir=candidate_dir)
            steered_after_shaping = self._consume_pending_steering(game_bible, candidate_dir=candidate_dir)
            if steered_after_shaping:
                self._execute_steering_pivot_turns(game_bible, dna_summary, candidate_dir, steered_after_shaping)
        if self.scheduler is not None:
            self.scheduler.start()
        deadline = time.monotonic() + self.drain_timeout_s

        def _board_progress_sig() -> Tuple[int, int, int, int]:
            all_items = self.board.items()
            return (
                len(all_items),
                sum(len(i.revisions) for i in all_items),
                sum(1 for i in all_items if i.state in ("approved", "cancelled")),
                sum(i.failure_count + i.spec_resets for i in all_items),
            )

        last_progress_sig = _board_progress_sig()
        assembly_passes = 0
        max_assembly_passes = int(getattr(self, "max_assembly_passes", 3))
        assembly_verified = max_assembly_passes <= 0
        blind_playtest_passes = 0
        max_blind_playtest_passes = int(
            getattr(self, "max_blind_playtest_passes", os.environ.get("STUDIO_MAX_BLIND_PLAYTEST_PASSES", "3"))
        )
        blind_playtest_verified = max_blind_playtest_passes <= 0
        qa_gate_passes = 0
        max_qa_gate_passes = int(
            getattr(self, "max_qa_gate_passes", os.environ.get("STUDIO_MAX_QA_GATE_PASSES", "2"))
        )
        qa_gate_verified = max_qa_gate_passes <= 0
        active_slice_entry = game_bible.get_active_slice()
        if candidate_dir is not None and active_slice_entry is not None and max_blind_playtest_passes > 0:
            art_dir = Path(candidate_dir) / "artifacts"
            last_playtest_verdict: Optional[str] = None
            while (
                art_dir / f"blind_playtest_report_{active_slice_entry.slice_id}_pass{blind_playtest_passes}.json"
            ).is_file():
                p_file = art_dir / f"blind_playtest_report_{active_slice_entry.slice_id}_pass{blind_playtest_passes}.json"
                try:
                    p_data = json.loads(p_file.read_text(encoding="utf-8"))
                    last_playtest_verdict = str(p_data.get("council_verdict") or "")
                except Exception:
                    last_playtest_verdict = None
                blind_playtest_passes += 1
            if self.board.is_drained() and last_playtest_verdict == "approve_slice":
                blind_playtest_verified = True
        if self.board.is_drained() and game_bible.writers_room_transcript:
            last_utt = game_bible.writers_room_transcript[-1]
            if "slice_assembly:verified" in (last_utt.bible_updates_summary or []):
                assembly_passes = max_assembly_passes
                assembly_verified = True
        while True:
            paused_dur = self._wait_if_paused(candidate_dir)
            if paused_dur > 0:
                deadline += paused_dur
            steered = self._consume_pending_steering(game_bible, candidate_dir=candidate_dir)
            if steered:
                self._execute_steering_pivot_turns(game_bible, dna_summary, candidate_dir, steered)
                if not self.board.is_drained():
                    assembly_passes = 0
                    assembly_verified = max_assembly_passes <= 0
                    blind_playtest_verified = max_blind_playtest_passes <= 0
                    qa_gate_passes = 0
                    qa_gate_verified = max_qa_gate_passes <= 0
                    if self.scheduler is not None:
                        self.scheduler.start()
            handled_failed = self._handle_failed_items()
            unblocked_cancelled = self._resolve_blocked_by_cancelled()
            applied = self._run_review_round(game_bible, dna_summary, candidate_dir)
            # Re-run failure & cancelled-dependent resolution immediately after `_run_review_round`,
            # because background workers can fail an item while a multi-second Council review is in flight,
            # or a Council `CapDecision(action="cancel")` in `_run_review_round` can leave dependents stranded.
            handled_failed = self._handle_failed_items() or handled_failed
            unblocked_cancelled = self._resolve_blocked_by_cancelled() or unblocked_cancelled
            if self.board.is_drained():
                if self.scheduler is not None and hasattr(self.scheduler, "stop_dispatch"):
                    self.scheduler.stop_dispatch()
                if not assembly_verified:
                    is_final_assembly_verify = assembly_passes >= max_assembly_passes
                    self._final_verification_assembly_pass = is_final_assembly_verify
                    try:
                        added_assembly = self._run_slice_assembly_check(
                            game_bible=game_bible,
                            dna_summary=dna_summary,
                            candidate_dir=candidate_dir,
                            pass_index=assembly_passes,
                        )
                    finally:
                        self._final_verification_assembly_pass = False
                    assembly_passes += 1
                    if added_assembly > 0:
                        if is_final_assembly_verify:
                            cur_s = game_bible.get_active_slice()
                            raise StudioAgentError(
                                f"Slice '{cur_s.slice_id if cur_s is not None else 'active'}' failed "
                                f"post-remediation assembly verification after {max_assembly_passes} pass(es) "
                                f"(persisted {added_assembly} unresolved assembly blocker task(s) on backlog.json; "
                                f"refusing unverified promotion)."
                            )
                        blind_playtest_verified = max_blind_playtest_passes <= 0
                        qa_gate_passes = 0
                        qa_gate_verified = max_qa_gate_passes <= 0
                        if self.scheduler is not None:
                            self.scheduler.start()
                        last_progress_sig = _board_progress_sig()
                        deadline = max(deadline, time.monotonic() + self.drain_timeout_s)
                        continue
                    assembly_verified = True
                if not blind_playtest_verified:
                    is_final_playtest_verify = blind_playtest_passes >= max_blind_playtest_passes
                    self._final_verification_blind_playtest_pass = is_final_playtest_verify
                    try:
                        added_playtest = self._run_end_of_slice_blind_playtest(
                            game_bible=game_bible,
                            dna_summary=dna_summary,
                            candidate_dir=candidate_dir,
                            pass_index=blind_playtest_passes,
                        )
                    finally:
                        self._final_verification_blind_playtest_pass = False
                    blind_playtest_passes += 1
                    if added_playtest > 0:
                        if is_final_playtest_verify:
                            cur_s = game_bible.get_active_slice()
                            raise StudioAgentError(
                                f"Slice '{cur_s.slice_id if cur_s is not None else 'active'}' failed "
                                f"post-remediation Blind Playtest verification after {max_blind_playtest_passes} iteration round(s) "
                                f"(Council persisted {added_playtest} blocking task(s) on backlog.json; "
                                f"refusing unverified promotion)."
                            )
                        assembly_passes = 0
                        assembly_verified = max_assembly_passes <= 0
                        qa_gate_passes = 0
                        qa_gate_verified = max_qa_gate_passes <= 0
                        if self.scheduler is not None:
                            self.scheduler.start()
                        last_progress_sig = _board_progress_sig()
                        deadline = max(deadline, time.monotonic() + self.drain_timeout_s)
                        continue
                    blind_playtest_verified = True
                if not qa_gate_verified:
                    is_final_qa_verify = qa_gate_passes >= max_qa_gate_passes
                    self._final_verification_qa_gate_pass = is_final_qa_verify
                    try:
                        added_qa = self._run_slice_qa_gate_check(
                            game_bible=game_bible,
                            candidate_dir=candidate_dir,
                            pass_index=qa_gate_passes,
                        )
                    finally:
                        self._final_verification_qa_gate_pass = False
                    qa_gate_passes += 1
                    if added_qa > 0:
                        if is_final_qa_verify:
                            cur_s = game_bible.get_active_slice()
                            raise StudioAgentError(
                                f"Slice '{cur_s.slice_id if cur_s is not None else 'active'}' failed "
                                f"post-remediation 5-Gate QA verification after {max_qa_gate_passes} remediation pass(es) "
                                f"(persisted {added_qa} unresolved QA remediation task(s) on backlog.json; "
                                f"refusing unverified promotion)."
                            )
                        assembly_passes = 0
                        assembly_verified = max_assembly_passes <= 0
                        blind_playtest_verified = max_blind_playtest_passes <= 0
                        if self.scheduler is not None:
                            self.scheduler.start()
                        last_progress_sig = _board_progress_sig()
                        deadline = max(deadline, time.monotonic() + self.drain_timeout_s)
                        continue
                    qa_gate_verified = True
                active_slice_done = game_bible.get_active_slice()
                if active_slice_done is not None:
                    active_slice_done.status = "completed"
                    self._persist_bible(game_bible, candidate_dir=candidate_dir)
                    self._emit_event(
                        "slice_completed",
                        {
                            "deliberation": f"Completed vertical slice {active_slice_done.slice_id} (index {active_slice_done.slice_index}).",
                            "slice_id": active_slice_done.slice_id,
                            "slice_index": active_slice_done.slice_index,
                            "title": active_slice_done.title,
                        },
                    )
                    if self.slice_checkpoint_callback is not None:
                        try:
                            self.slice_checkpoint_callback(game_bible, active_slice_done, candidate_dir)
                        except Exception as cb_exc:
                            logger.warning("Slice checkpoint callback raised error: %s", cb_exc)
                return
            cur_progress_sig = _board_progress_sig()
            if applied > 0 or steered or handled_failed or unblocked_cancelled or cur_progress_sig != last_progress_sig:
                last_progress_sig = cur_progress_sig
                deadline = max(deadline, time.monotonic() + self.drain_timeout_s)
            board_items_now = self.board.items()
            if any(i.state == "failed" for i in board_items_now):
                # A background worker transitioned an item to `failed` asynchronously; loop immediately so
                # `_handle_failed_items()` requeues, escalates to `needs_decision`, or raises its authentic error.
                continue
            active = [
                i for i in board_items_now
                if i.state in ("ready", "in_progress", "waiting_for_lock", "changes_requested", "awaiting_review", "needs_decision")
            ]
            if not active and self.scheduler is not None and self.scheduler.running_count() == 0:
                # Re-snapshot atomically after `running_count() == 0` in case the last worker just finished/failed
                # between `board_items_now` and `running_count()`.
                recheck_items = self.board.items()
                if any(
                    i.state in ("failed", "ready", "in_progress", "waiting_for_lock", "changes_requested", "awaiting_review", "needs_decision")
                    for i in recheck_items
                ):
                    continue
                if self._resolve_blocked_by_cancelled():
                    continue
                raise StudioAgentError(
                    "Backlog cannot drain: remaining items are blocked with nothing in flight.\n" + self.board.snapshot_text()
                )
            if time.monotonic() > deadline:
                raise StudioAgentError(f"Backlog did not drain within {self.drain_timeout_s}s.\n" + self.board.snapshot_text())
            if applied == 0:
                time.sleep(self.drain_poll_s)

    _ESSENTIAL_DOMAIN_SPECIALISTS = (
        "LevelAndProgressionDesigner",
        "GameplaySystemsEngineer",
        "UXExperienceDesigner",
        "AudioAtmosphereDesigner",
    )
    _REQUIRED_BOARD_SPECIALISTS = (
        "GameplayDesigner",
        "LevelAndProgressionDesigner",
        "GameplaySystemsEngineer",
        "VisualArtDirector",
        "UXExperienceDesigner",
        "AudioAtmosphereDesigner",
    )

    def _missing_required_board_specialists(self, game_bible: GameBibleSpec) -> List[str]:
        """When a Production Board is attached in a full Writers' Room session, every core domain specialist
        must speak at least once so UX, Audio, Level, Systems, Art, and Gameplay all contribute their domain design and tasks."""
        if self.board is None or (self.max_turns < 6 and not is_real_genai_client(self.client)):
            return []
        spoken = {u.speaker_role for u in game_bible.writers_room_transcript}
        return [role for role in self._REQUIRED_BOARD_SPECIALISTS if role not in spoken]

    def _select_next_speaker(
        self,
        last_speaker_role: str,
        addressed_to: List[str],
        game_bible: GameBibleSpec,
        dna_summary: str,
        spoken_counts: Dict[str, int],
    ) -> Tuple[Dict[str, Any], WritersRoomListenCheck]:
        """Polls the Core Council Leads, any Sub-Agents invited via `addressed_to`, and (once the opening
        core exchange is underway) any essential domain specialists who have not yet spoken.
        Relies purely on LLM deliberation & urgency (zero hardcoded role score boosts)."""
        core_roles = {s["role"] for s in CORE_COUNCIL_ROSTER}
        invited_roles = set(addressed_to or []) - {last_speaker_role}
        total_spoken = sum(spoken_counts.values()) if spoken_counts else 0
        unheard_essential = (
            {r for r in self._ESSENTIAL_DOMAIN_SPECIALISTS if spoken_counts.get(r, 0) == 0}
            if total_spoken >= 2
            else set()
        )
        invited_specs = [
            spec
            for spec in SPECIALIST_ROSTER
            if spec["role"] != last_speaker_role and spec["role"] in invited_roles
        ]
        if is_real_genai_client(self.client) and invited_specs:
            listeners = list(invited_specs)
            if unheard_essential and not any(spoken_counts.get(sp["role"], 0) == 0 for sp in listeners):
                for spec in SPECIALIST_ROSTER:
                    if (
                        spec["role"] != last_speaker_role
                        and spec["role"] in unheard_essential
                        and spec not in listeners
                    ):
                        listeners.append(spec)
                        break
        else:
            listeners = [
                spec
                for spec in SPECIALIST_ROSTER
                if spec["role"] != last_speaker_role
                and (spec["role"] in core_roles or spec["role"] in invited_roles or spec["role"] in unheard_essential)
            ]
        if not listeners:
            listeners = [spec for spec in SPECIALIST_ROSTER if spec["role"] != last_speaker_role] or list(SPECIALIST_ROSTER)

        self._emit_activity(
            agent_role="CouncilTable",
            status_text=f"Council listening & deliberating next speaker ({', '.join(sp['role'] for sp in listeners)})...",
            active_agents=[sp["role"] for sp in listeners],
            phase="listening",
        )

        checks = self._map_parallel(
            lambda sp: self._evaluate_specialist_listen(sp, game_bible, dna_summary), listeners
        )
        candidates: List[Tuple[int, float, Dict[str, Any], WritersRoomListenCheck]] = []
        for spec, check in zip(listeners, checks):
            candidates.append((1 if check.should_interject else 0, float(check.urgency_score), spec, check))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _, _, chosen_spec, chosen_check = candidates[0]

        # If a Core Council Lead chose to call in a specialized Sub-Agent (e.g. StoryWorldArchitect -> ScriptwriterAgent),
        # route the microphone to that Sub-Agent with the Lead's brief.
        consult_role = (getattr(chosen_check, "consult_subagent_role", None) or "").strip()
        if consult_role and consult_role != last_speaker_role:
            sub_spec = next((sp for sp in SPECIALIST_ROSTER if sp["role"] == consult_role), None)
            if sub_spec is not None:
                sub_brief = (getattr(chosen_check, "subagent_brief", "") or chosen_check.intended_contribution_summary).strip()
                delegated_check = WritersRoomListenCheck(
                    deliberation=(
                        f"Called into the Writers' Room by {chosen_spec['display_name']} ({chosen_spec['role']}): "
                        f"{chosen_check.deliberation}"
                    ),
                    should_interject=True,
                    urgency_score=chosen_check.urgency_score,
                    intended_contribution_summary=(
                        f"[Called in by {chosen_spec['role']}] {sub_brief}"
                        if sub_brief
                        else f"Called in by {chosen_spec['role']} to contribute {sub_spec['display_name']} expertise."
                    ),
                )
                self._emit_activity(
                    agent_role=sub_spec["role"],
                    status_text=(
                        f"{chosen_spec['display_name']} ({chosen_spec['role']}) called in Sub-Agent "
                        f"{sub_spec['display_name']} ({sub_spec['role']})"
                        + (f": {sub_brief[:100]}" if sub_brief else "")
                    ),
                    active_agents=[chosen_spec["role"], sub_spec["role"]],
                    phase="subagent_handoff",
                    extra={
                        "called_by_lead": chosen_spec["role"],
                        "subagent_role": sub_spec["role"],
                        "subagent_brief": sub_brief,
                    },
                )
                return sub_spec, delegated_check

        return chosen_spec, chosen_check

    def _evaluate_convergence(self, game_bible: GameBibleSpec, dna_summary: str) -> DirectorConvergenceCheck:
        """CreativeDirector evaluates whether the Living Game Bible is cohesive and ready to lock."""
        self._emit_activity(
            agent_role="CreativeDirector",
            status_text="Creative Director is evaluating Game Bible cohesion & convergence readiness...",
            active_agents=["CreativeDirector"],
            phase="convergence_check",
        )
        bible_snapshot = self._format_bible_snapshot(game_bible)
        prompt = (
            f"You are the CreativeDirector of the Game Studio Writers' Room.\n"
            f"PLAYER DNA & STUDIO CONTEXT:\n{dna_summary}\n\n"
            f"{STUDIO_CRITICAL_DISCOURSE_CHARTER}\n\n"
            f"{bible_snapshot}\n"
            f"Skeptically evaluate whether the Living Game Bible is ready to lock and enter production for the active slice (catering faithfully to any Executive Producer creative seed/directives while ensuring the active slice is ambitious yet realistically buildable without trying to cram five sequels into Slice 0).\n"
            f"Do NOT rubber-stamp a shallow 'Hello World' prototype or polite peer hand-waving, and do NOT demand endless Writers' Room churn once the active slice's full core loop and specs are concrete. To be ready (`is_ready_to_lock_bible = true`), the Bible MUST have:\n"
            f"1. A deeply evocative premise, core fantasy, and a Concept Evolution history showing the premise was genuinely pressure-tested and reshaped by multiple specialists (do not lock a premise that is still its initial draft),\n"
            f"2. A compelling First 60 Seconds Hook and a clear UI, HUD & Onboarding Philosophy suited to this game,\n"
            f"3. An Experience Storyboard detailing the player's emotional & sensory arc,\n"
            f"4. Synthesized 16:9 Visual Mockup board(s),\n"
            f"5. Cohesive World Lore, Audio direction, and Characters/Entities as appropriate to the game concept,\n"
            f"6. A Technical Architecture Blueprint (`technical_architecture_blueprint`) from GameplaySystemsEngineer translating mechanics into Godot 4 nodes, procedural/shader or sprite rendering, colliders, and test invariants.\n"
            f"7. A COHESIVE WHOLE GAME (NEVER A TINY TUTORIAL SLICE): a Level Generation Strategy (how levels are produced and what they look like), a Core Loop at three time scales "
            f"(moment-to-moment, minute-to-minute, and session loops, plus `player_agency_and_optimization_depth`), an Engagement Plan, an Onboarding Plan (where onboarding is ONLY the opening 60 seconds, never the whole game), "
            f"and a Goal Hierarchy spanning short-, mid- and long-term goals. Do NOT lock the Bible if `slice_roadmap` or the Production Board items shrink the build into 'the smallest slice possible' "
            f"(e.g. a 4-step tutorial or a 1-step connection) while deferring the actual minute-to-minute/session loop and systemic depth to unbuilt future slices! "
            f"Current structural gaps: {validate_game_design(game_bible) or 'none'}.\n"
            f"Return your structured DirectorConvergenceCheck with `deliberation` FIRST."
        )
        return self._call_structured_llm(prompt, DirectorConvergenceCheck, temperature=0.4)

    def _synthesize_final_world_spec(self, game_bible: GameBibleSpec, dna: Optional[SteamPlayerDNA], target_episode: int) -> GenesisWorldSpec:
        """Synthesizes the authoritative GenesisWorldSpec grounded in the locked Living Game Bible."""
        self._emit_activity(
            agent_role="CreativeDirector",
            status_text=f"Creative Director is synthesizing authoritative GenesisWorldSpec for '{game_bible.game_title}'...",
            active_agents=["CreativeDirector"],
            phase="world_spec_synthesis",
        )
        bible_snapshot = self._format_bible_snapshot(game_bible)
        mood_hint = getattr(dna, "preferred_palette_mood", None) if dna is not None else None
        mood_line = f"Preferred Palette Mood Hint: {mood_hint}\n" if mood_hint else ""
        prompt = (
            f"You are the CreativeDirector finalizing the authoritative GenesisWorldSpec from the Writers' Room Living Game Bible.\n"
            f"Target Episode: {target_episode}\n"
            f"{mood_line}\n"
            f"{bible_snapshot}\n"
            f"Synthesize the complete GenesisWorldSpec honoring every single pillar, UI & onboarding rule, camera staging, "
            f"and story hook agreed upon by the Writers' Room. Provide `deliberation` FIRST."
        )
        world_spec: GenesisWorldSpec = self._call_structured_llm(prompt, GenesisWorldSpec)
        world_spec.target_episode = target_episode
        if game_bible.game_title and game_bible.game_title != "Untitled Concept":
            world_spec.game_title = game_bible.game_title
        return world_spec

    # ------------------------------------------------------------------
    # Concept genesis: the game idea is NOT given at the start; it emerges
    # from divergent fragments, collides, is judged for novelty against
    # prior concepts, and is then evolved by the room.
    # ------------------------------------------------------------------

    _DESIGN_SECTION_OWNERS = {
        "level_generation_strategy": "LevelAndProgressionDesigner",
        "goal_hierarchy": "LevelAndProgressionDesigner",
        "core_loop": "GameplayDesigner",
        "engagement_plan": "GameplayDesigner",
        "onboarding_plan": "UXExperienceDesigner",
    }

    def _complete_missing_design_sections(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        candidate_dir: Optional[Path],
        max_rounds: int = 2,
    ) -> None:
        """Gives the owning specialist a mandatory turn for every missing/incomplete game-design section
        and ensures every required domain specialist has contributed before locking a Production Board build."""
        for _round in range(max_rounds):
            issues = validate_game_design(game_bible)
            missing_specialists = self._missing_required_board_specialists(game_bible)
            if not issues and not missing_specialists:
                return
            owners: List[str] = []
            for issue in issues:
                for section, owner in self._DESIGN_SECTION_OWNERS.items():
                    if issue.startswith(section) and owner not in owners:
                        owners.append(owner)
            for role in missing_specialists:
                if role not in owners:
                    owners.append(role)
            for owner in owners:
                spec = next(sp for sp in SPECIALIST_ROSTER if sp["role"] == owner)
                gap_summary = "; ".join(issues) if issues else f"Contribute {spec['display_name']} domain strategy and Production Board tasks for Slice 0."
                check = WritersRoomListenCheck(
                    deliberation="The Game Bible cannot lock until all required game-design sections and specialist domain contributions are complete.",
                    should_interject=True,
                    urgency_score=1.0,
                    intended_contribution_summary="Complete these game-design and domain requirements: " + gap_summary,
                )
                proposal = self._execute_specialist_turn(
                    specialist=spec, listen_check=check, game_bible=game_bible, dna_summary=dna_summary
                )
                self._apply_turn_proposal(
                    game_bible=game_bible,
                    speaker_role=owner,
                    speaker_display_name=spec["display_name"],
                    proposal=proposal,
                    candidate_dir=candidate_dir,
                )
        remaining = validate_game_design(game_bible)
        if remaining:
            raise StudioAgentError(f"Game Bible still lacks required game-design content after completion turns: {remaining}")

    def _map_parallel(self, fn: Callable[[Any], Any], items: List[Any]) -> List[Any]:
        """Applies `fn` to `items` concurrently (bounded by max_parallel_calls), preserving order and propagating the first error."""
        workers = min(self.max_parallel_calls, len(items))
        if workers <= 1:
            return [fn(i) for i in items]
        parent_scope = get_active_explicit_prefix_cache_scope(self.client)

        def _wrapped_fn(item: Any) -> Any:
            with bind_explicit_prefix_cache_scope(parent_scope):
                return fn(item)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_wrapped_fn, items))

    def _resolve_memory_dir(self, candidate_dir: Optional[Path] = None) -> Optional[Path]:
        if candidate_dir is not None:
            return Path(candidate_dir)
        if self.board is not None and getattr(self.board, "path", None) is not None:
            return Path(self.board.path).parent
        return None

    def _structured_with_validation(
        self,
        prompt: str,
        schema_cls: Any,
        validate: Callable[[Any], List[str]],
        attempts: int = 3,
        temperature: Optional[float] = None,
        parts: Optional[List[Any]] = None,
        caller_role: Optional[str] = None,
        candidate_dir: Optional[Path] = None,
        slice_id: str = "",
        *,
        cacheable_prefix: str = "",
        min_calls_before_create: int = 2,
    ) -> Any:
        """Calls the LLM until `validate` returns no issues, feeding issues back; raises after `attempts` (Rule 5).
        Also supports optional bounded `memory_query` lookups and `memory_updates`/`memory_note` persistence
        when `candidate_dir` (or an attached board path) is present."""
        with explicit_prefix_cache_scope(self.client, ttl_seconds=300, reuse_outer=True):
            return self._structured_with_validation_inner(
                prompt=prompt,
                schema_cls=schema_cls,
                validate=validate,
                attempts=attempts,
                temperature=temperature,
                parts=parts,
                caller_role=caller_role,
                candidate_dir=candidate_dir,
                slice_id=slice_id,
                cacheable_prefix=cacheable_prefix,
                min_calls_before_create=min_calls_before_create,
            )

    def _structured_with_validation_inner(
        self,
        prompt: str,
        schema_cls: Any,
        validate: Callable[[Any], List[str]],
        attempts: int = 3,
        temperature: Optional[float] = None,
        parts: Optional[List[Any]] = None,
        caller_role: Optional[str] = None,
        candidate_dir: Optional[Path] = None,
        slice_id: str = "",
        *,
        cacheable_prefix: str = "",
        min_calls_before_create: int = 2,
    ) -> Any:
        issues: List[str] = []
        eff_dir = self._resolve_memory_dir(candidate_dir)
        has_embed_client = callable(getattr(getattr(self.client, "models", None), "embed_content", None))
        mem_query_blocks: List[str] = []
        asked_mem_queries: set[str] = set()
        mem_queries_used = 0
        max_mem_queries = 2 if (eff_dir is not None and has_embed_client) else 0
        attempt = 1

        eff_full_prompt = (
            prompt
            if (not cacheable_prefix or prompt.startswith(cacheable_prefix))
            else (cacheable_prefix + prompt)
        )
        while attempt <= attempts:
            feedback = (
                "\n\nYOUR PREVIOUS ANSWER FAILED VALIDATION; FIX EVERY ISSUE:\n- " + "\n- ".join(issues) if issues else ""
            )
            mem_q_block = ("\n\n" + "\n".join(mem_query_blocks)) if mem_query_blocks else ""
            if getattr(self._call_structured_llm, "__func__", None) is CreativeWritersRoom._call_structured_llm:
                result = self._call_structured_llm(
                    prompt,
                    schema_cls,
                    temperature=temperature,
                    parts=parts,
                    cacheable_prefix=cacheable_prefix,
                    dynamic_tail=(mem_q_block + feedback),
                    min_calls_before_create=min_calls_before_create,
                )
            else:
                result = self._call_structured_llm(
                    eff_full_prompt + mem_q_block + feedback,
                    schema_cls,
                    temperature=temperature,
                    parts=parts,
                )

            if eff_dir is not None:
                from engine.studio.game_memory import GameMemoryStore, format_memory_query_results

                mem_store = GameMemoryStore(
                    eff_dir,
                    event_callback=getattr(self, "event_callback", None),
                    client=self.client,
                )
                mem_updates = getattr(result, "memory_updates", None) or []
                if mem_updates:
                    mem_store.apply_mutations(
                        author_role=caller_role or "council",
                        proposals=mem_updates,
                        slice_id=slice_id,
                        client=self.client,
                    )
                mem_note = str(getattr(result, "memory_note", "") or "").strip()
                if mem_note:
                    mem_store.parse_and_apply_compact_note(
                        author_role=caller_role or "council",
                        raw_note=mem_note,
                        deliberation=str(getattr(result, "deliberation", "") or "Council memory note."),
                        slice_id=slice_id,
                        client=self.client,
                    )

                mem_q = str(getattr(result, "memory_query", "") or "").strip()
                mem_q_ans = str(getattr(result, "memory_query_answer", "") or "").strip()
                mem_q_bank = str(getattr(result, "memory_query_bank", "") or "").strip()
                mem_q_key = f"{mem_q_bank.lower()}::{mem_q.lower()}"
                if (
                    mem_q
                    and not mem_q_ans
                    and mem_queries_used < max_mem_queries
                    and mem_q_key not in asked_mem_queries
                ):
                    asked_mem_queries.add(mem_q_key)
                    mem_queries_used += 1
                    hits = mem_store.query_memories(
                        caller_role=caller_role or "council",
                        query_text=mem_q,
                        bank_filter=mem_q_bank,
                        client=self.client,
                    )
                    existing_ctx = f"{eff_full_prompt}\n" + "\n".join(mem_query_blocks)
                    has_unseen_hits = any(
                        f"[{entry.memory_id}]" not in existing_ctx
                        for entry, _ in hits
                    )
                    if has_unseen_hits:
                        mem_query_blocks.append(format_memory_query_results(mem_q, mem_q_bank, hits))
                        issues = []
                        continue

            issues = validate(result)
            if not issues:
                return result
            logger.info("%s attempt %d/%d failed validation: %s", schema_cls.__name__, attempt, attempts, issues)
            attempt += 1
        raise StudioAgentError(f"{schema_cls.__name__} still invalid after {attempts} attempts: {issues}")

    def _load_prior_concepts(self, candidate_dir: Optional[Path], limit: int = 12) -> str:
        """Digest of concepts from earlier builds (title, pitch, pillars, camera) so the room can avoid repeating them."""
        if candidate_dir is None:
            return "(none)"
        root = Path(candidate_dir).resolve().parent
        found: List[Tuple[float, str]] = []
        seen_titles = set()
        paths = list(root.glob("*/game_bible.json")) + list(root.glob("checkpoints/*/game_bible.json"))
        for path in paths:
            if path.parent.resolve() == Path(candidate_dir).resolve():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Skipping unreadable prior bible %s: %s", path, exc)
                continue
            title = str(data.get("game_title") or "").strip()
            pitch = str(data.get("elevator_pitch") or "").strip()
            if not title or title == "Untitled Concept" or not pitch or title in seen_titles:
                continue
            seen_titles.add(title)
            pillars = [str(p)[:140] for p in (data.get("gameplay_pillars") or [])[:3]]
            found.append((
                path.stat().st_mtime,
                f"- '{title}': {pitch[:400]} | pillars: {pillars} | camera: {str(data.get('camera_and_spatial_paradigm') or '')[:140]}",
            ))
        found.sort(key=lambda x: x[0], reverse=True)
        return "\n".join(line for _, line in found[:limit]) or "(none)"

    def _run_concept_genesis(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        creative_prompt: Optional[str],
        candidate_dir: Optional[Path],
        target_episode: int,
    ) -> WritersRoomTurnProposal:
        """Divergence -> collision -> novelty judgement -> selection. Returns the Creative Director's opening turn,
        which seeds a PROVISIONAL premise that the rest of the room then pressure-tests and evolves.
        """
        roles = [s["role"] for s in SPECIALIST_ROSTER]
        prior = self._load_prior_concepts(candidate_dir)
        seed_clause = (
            f"EXECUTIVE PRODUCER DIRECTION (every concept must honor this): {creative_prompt.strip()}\n"
            if (creative_prompt and creative_prompt.strip()) else ""
        )
        prior_clause = f"PRIOR CONCEPTS ALREADY PRODUCED (do NOT repeat or lightly vary any of them):\n{prior}\n"

        # 1. Divergent provocations (one per specialist, mutually orthogonal)
        self._emit_activity(
            agent_role="CreativeDirector",
            status_text=f"Concept Genesis (1/4): Creative Director is crafting orthogonal provocations for {len(roles)} specialists...",
            active_agents=["CreativeDirector"],
            phase="genesis_provocations",
        )
        def _validate_prov(ps: CreativeProvocationSet) -> List[str]:
            got = {p.specialist_role for p in ps.provocations}
            missing = [r for r in roles if r not in got]
            return [f"Missing provocation for roles: {missing}"] if missing else []

        prov_set = self._structured_with_validation(
            (
                f"You are the CreativeDirector preparing Episode {target_episode}'s Writers' Room. NO game has been chosen yet.\n"
                f"Write one creative provocation for each specialist role in {roles}. Provocations must be mutually orthogonal "
                f"(each opens a different creative axis), phrased as clear, concrete design questions or constraints — not genre labels, not existing "
                f"game names, and grounded in tangible, playable experiences rather than obscure academic jargon.\n"
                f"{prior_clause}{seed_clause}"
                f"PLAYER TASTE CONTEXT (signals of what they value; do not imitate):\n{dna_summary}\n"
                f"Provide `deliberation` FIRST."
            ),
            CreativeProvocationSet,
            _validate_prov,
        )
        provocation_by_role = {p.specialist_role: p.provocation for p in prov_set.provocations}

        # 2. Divergence: each specialist independently contributes fragments (no coordination, no whole game)
        self._emit_activity(
            agent_role="CouncilTable",
            status_text=f"Concept Genesis (2/4): All {len(SPECIALIST_ROSTER)} specialists are seeding divergent concept fragments in parallel...",
            active_agents=roles,
            phase="genesis_fragments",
        )
        def _seed_for(spec: Dict[str, str]) -> ConceptSeedProposal:
            role = spec["role"]
            return self._call_structured_llm(
                (
                    f"You are the {spec['display_name']} ({role}) at the very start of a Writers' Room. NO game has been chosen.\n"
                    f"Your specialty focus: {spec['focus']}\n"
                    f"YOUR PROVOCATION: {provocation_by_role[role]}\n"
                    f"{prior_clause}{seed_clause}"
                    f"PLAYER TASTE CONTEXT (signals of what they value; do not imitate or mash up titles):\n{dna_summary}\n\n"
                    f"Contribute 2 independent concept fragments from YOUR domain, each a clear, concrete, immediately understandable seed "
                    f"(a tactile interaction verb or system, a tangible world tension, an emotional experience, a structural constraint, or a sensory signature). "
                    f"Use plain, vivid language—avoid pretentious pseudo-scientific or academic buzzword salad. Do NOT pitch a complete game, do "
                    f"NOT coordinate with other specialists, and avoid anything resembling the prior concepts. Provide `deliberation` FIRST."
                ),
                ConceptSeedProposal,
            )

        fragments: List[ConceptFragment] = []
        for spec, seed in zip(SPECIALIST_ROSTER, self._map_parallel(_seed_for, list(SPECIALIST_ROSTER))):
            role = spec["role"]
            for idx, frag in enumerate(seed.fragments[:2]):
                frag.contributor_role = role
                frag.fragment_id = f"{role}_{idx}"
                fragments.append(frag)
        if len(fragments) < 2 or len({f.contributor_role for f in fragments}) < 2:
            raise StudioAgentError("Concept genesis produced too few fragments from distinct specialists")
        frag_ids = {f.fragment_id: f for f in fragments}
        frag_block = "\n".join(f"  [{f.fragment_id}] ({f.contributor_role}, {f.fragment_kind}): {f.fragment}" for f in fragments)

        # 3. Collision + novelty judgement (regenerate with feedback if nothing is distinct)
        overlap_feedback = ""
        distinct: List[ConceptCandidate] = []
        all_candidates: List[ConceptCandidate] = []

        def _validate_collision(cp: ConceptCollisionProposal) -> List[str]:
            issues: List[str] = []
            if len(cp.candidates) < 2:
                issues.append("Produce at least 2 candidates.")
            for c in cp.candidates:
                missing = [i for i in c.source_fragment_ids if i not in frag_ids]
                if missing:
                    issues.append(f"Candidate '{c.candidate_id}' cites unknown fragment ids {missing}.")
                if len({frag_ids[i].contributor_role for i in c.source_fragment_ids if i in frag_ids}) < 2:
                    issues.append(f"Candidate '{c.candidate_id}' must combine fragments from at least two different contributors.")
            if len({c.candidate_id for c in cp.candidates}) != len(cp.candidates):
                issues.append("Duplicate candidate_id values.")
            return issues

        for _round in range(3):
            self._emit_activity(
                agent_role="CreativeDirector",
                status_text=f"Concept Genesis (3/4): Creative Director is colliding {len(fragments)} fragments into candidate premises (round {_round + 1})...",
                active_agents=["CreativeDirector"],
                phase="genesis_collision",
            )
            collision = self._structured_with_validation(
                (
                    f"You are the CreativeDirector. The room has contributed these independent fragments:\n{frag_block}\n\n"
                    f"Synthesize fragments from DIFFERENT specialists into 4 candidate game premises that are distinct from one another and from the "
                    f"prior concepts. Each candidate must combine at least two fragments into a coherent, grounded, immediately playable game concept "
                    f"where a human player can instantly picture what they do on screen, what they build/control/fight, and why it is fun. "
                    f"Write titles and premises in clear, accessible language—NEVER string together obscure scientific or academic jargon into a word salad. "
                    f"Premises are provisional starting points the room will evolve, not finished designs.\n"
                    f"{prior_clause}{seed_clause}{overlap_feedback}"
                    f"Provide `deliberation` FIRST; each candidate needs a unique `candidate_id`."
                ),
                ConceptCollisionProposal,
                _validate_collision,
            )
            all_candidates = list(collision.candidates)
            overlaps: List[str] = []
            distinct = []

            def _judge(cand: ConceptCandidate) -> Optional[NoveltyJudgement]:
                if prior.strip() == "(none)":
                    return None
                return self._call_structured_llm(
                    (
                        f"You are the studio's novelty judge. Compare this candidate with every prior concept, element by element "
                        f"(core interaction, setting, tone, structure).\n"
                        f"CANDIDATE [{cand.candidate_id}] '{cand.working_title}': {cand.premise} | tension: {cand.core_tension}\n"
                        f"{prior_clause}"
                        f"Provide `deliberation` FIRST and the verdict last."
                    ),
                    NoveltyJudgement,
                    temperature=0.3,
                )

            for cand, verdict in zip(all_candidates, self._map_parallel(_judge, all_candidates)):
                if verdict is None or verdict.is_distinct_from_prior_concepts:
                    distinct.append(cand)
                else:
                    overlaps.append(f"'{cand.working_title}' overlaps: {verdict.overlapping_elements}")
            if distinct:
                break
            overlap_feedback = "REJECTED FOR OVERLAP WITH PRIOR CONCEPTS (move far away from these):\n- " + "\n- ".join(overlaps) + "\n"
        if not distinct:
            raise StudioAgentError("Concept genesis could not produce a candidate distinct from prior concepts after 3 rounds")

        # 4. Director selects (rationale first, decision last)
        distinct_ids = {c.candidate_id for c in distinct}

        def _validate_selection(sel: ConceptSelection) -> List[str]:
            return [] if sel.selected_candidate_id in distinct_ids else [f"selected_candidate_id must be one of {sorted(distinct_ids)}"]

        cand_block = "\n".join(
            f"  [{c.candidate_id}] '{c.working_title}': {c.premise} | tension: {c.core_tension}" for c in distinct
        )
        self._emit_activity(
            agent_role="CreativeDirector",
            status_text=f"Concept Genesis (4/4): Creative Director is selecting the provisional starting premise from {len(distinct)} candidates...",
            active_agents=["CreativeDirector"],
            phase="genesis_selection",
        )
        selection = self._structured_with_validation(
            (
                f"You are the CreativeDirector choosing the PROVISIONAL starting premise.\n{cand_block}\n\n"
                f"Choose the candidate that is clearest, most coherent, and has the strongest potential to become a genuinely fun, playable 2D game once the room "
                f"pressure-tests it (reject any candidate that sounds like abstract academic jargon or a confusing mashup of buzzwords). "
                f"Write `opening_dialogue` in plain, direct language presenting what the player actually does and why it is compelling as a draft to be challenged and reshaped, "
                f"and name which specialists should challenge it first. Provide `deliberation` FIRST and `selected_candidate_id` last."
            ),
            ConceptSelection,
            _validate_selection,
        )
        chosen = next(c for c in distinct if c.candidate_id == selection.selected_candidate_id)

        game_bible.concept_fragments = fragments
        game_bible.concept_candidates = all_candidates
        self._emit_event(
            "concept_genesis_completed",
            {
                "deliberation": selection.deliberation,
                "fragments": len(fragments),
                "candidates": len(all_candidates),
                "distinct_candidates": len(distinct),
                "selected_candidate_id": chosen.candidate_id,
            },
        )
        return WritersRoomTurnProposal(
            deliberation=selection.deliberation,
            spoken_dialogue=selection.opening_dialogue,
            addressed_to=["GameplayDesigner", "StoryWorldArchitect", "UXExperienceDesigner"],
            updated_game_title=chosen.working_title,
            updated_elevator_pitch=chosen.premise,
            updated_core_fantasy=chosen.core_tension or None,
        )

    def run_writers_room(
        self,
        dna: Optional[SteamPlayerDNA] = None,
        hypothesis_report: Optional[Any] = None,
        candidate_dir: Optional[Path] = None,
        target_episode: int = 0,
        cartridge_id: str = "active",
        existing_bible: Optional[GameBibleSpec] = None,
        creative_prompt: Optional[str] = None,
    ) -> GameBibleSpec:
        """Runs the open-ended multi-specialist Writers' Room conversation until the CreativeDirector locks the Living Game Bible."""
        if self.client is None:
            raise StudioAgentError(
                "CreativeWritersRoom requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)"
            )

        dna_summary = self._format_dna_summary(dna, hypothesis_report)
        if creative_prompt and creative_prompt.strip():
            self._active_creative_prompt = creative_prompt.strip()
        elif existing_bible is not None and existing_bible.concept_evolution:
            first_prev = (existing_bible.concept_evolution[0].previous_premise or "").strip()
            self._active_creative_prompt = first_prev if first_prev != "Untitled Concept" else ""
        else:
            self._active_creative_prompt = ""
        if creative_prompt and creative_prompt.strip():
            dna_summary = (
                f"{dna_summary}\n"
                f"- EXECUTIVE PRODUCER CREATIVE SEED / DIRECTION: {creative_prompt.strip()}"
            )

        already_in_production = False
        existing_status_was_locked = False
        if existing_bible is not None:
            game_bible = existing_bible
            existing_status_was_locked = str(getattr(game_bible, "status", "") or "").lower() == "locked"
            has_trailing_user_steering = bool(
                game_bible.writers_room_transcript
                and game_bible.writers_room_transcript[-1].speaker_role == "UserSteering"
            )
            board_has_execution_history = bool(
                self.board is not None
                and any(
                    i.state in ("in_progress", "awaiting_review", "changes_requested", "approved") or i.revisions
                    for i in self.board.items()
                )
            )
            transcript_already_converged = any(
                "Locked Living Game Bible for production" in (u.bible_updates_summary or [])
                for u in game_bible.writers_room_transcript
            )
            already_in_production = (
                not has_trailing_user_steering
                and (
                    game_bible.status in ("in_production", "locked")
                    or board_has_execution_history
                    or transcript_already_converged
                )
                and not validate_game_design(game_bible)
            )
            game_bible.status = "in_production" if already_in_production else "brainstorming"
            if dna is not None or not getattr(game_bible, "target_player_dna_summary", ""):
                game_bible.target_player_dna_summary = dna_summary
            elif getattr(game_bible, "target_player_dna_summary", ""):
                dna_summary = game_bible.target_player_dna_summary
        else:
            game_bible = GameBibleSpec(
                deliberation="Initializing Living Game Bible for open-ended multi-agent Writers' Room.",
                cartridge_id=cartridge_id,
                target_episode=target_episode,
                status="brainstorming",
                game_title="Untitled Concept",
                core_genre=(dna.top_genres[0] if (dna and getattr(dna, "top_genres", None)) else "indie_experience"),
                target_player_dna_summary=dna_summary,
                elevator_pitch=(creative_prompt.strip() if (creative_prompt and creative_prompt.strip()) else ""),
            )

        self._persist_bible(game_bible, candidate_dir=candidate_dir)

        spoken_counts: Dict[str, int] = {s["role"]: 0 for s in SPECIALIST_ROSTER}
        last_speaker_role = "CreativeDirector"
        last_addressed_to: List[str] = ["GameplayDesigner", "StoryWorldArchitect", "UXExperienceDesigner"]
        pending_invited_roles: List[str] = list(last_addressed_to)
        for existing_utt in game_bible.writers_room_transcript[1:]:
            if existing_utt.speaker_role in spoken_counts:
                spoken_counts[existing_utt.speaker_role] += 1
            if existing_utt.speaker_role in pending_invited_roles:
                pending_invited_roles.remove(existing_utt.speaker_role)
            last_speaker_role = existing_utt.speaker_role
            if existing_utt.addressed_to:
                last_addressed_to = list(existing_utt.addressed_to)
                for r in existing_utt.addressed_to:
                    if r not in pending_invited_roles and spoken_counts.get(r, 0) == 0:
                        pending_invited_roles.append(r)

        # Opening pitch/provocation by CreativeDirector if transcript is empty
        if not game_bible.writers_room_transcript:
            opening_proposal = self._run_concept_genesis(
                game_bible=game_bible,
                dna_summary=dna_summary,
                creative_prompt=creative_prompt,
                candidate_dir=candidate_dir,
                target_episode=target_episode,
            )
            opening_utt = self._apply_turn_proposal(
                game_bible=game_bible,
                speaker_role="CreativeDirector",
                speaker_display_name="Creative Director",
                proposal=opening_proposal,
                candidate_dir=candidate_dir,
            )
            last_addressed_to = opening_utt.addressed_to
            for r in opening_utt.addressed_to:
                if r not in pending_invited_roles:
                    pending_invited_roles.append(r)

        # Open-ended specialist conversation loop (skipped when resuming a build already in active production)
        turns_executed = sum(spoken_counts.values())
        pending_routed_role = ""
        pending_routed_from = ""
        pending_routed_reason = ""
        if self.board is not None and self.scheduler is not None and not is_real_genai_client(self.client):
            if turns_executed >= min(3, self.max_turns) or len(self.board.items()) >= 3:
                self.scheduler.start()
        while not already_in_production and turns_executed < self.max_turns:
            self._wait_if_paused(candidate_dir)
            steered = self._consume_pending_steering(game_bible, candidate_dir=candidate_dir)
            if steered:
                pending_routed_role = ""
                pending_routed_from = ""
                pending_routed_reason = ""
                last_speaker_role = "ExecutiveProducer"
                last_addressed_to = ["CreativeDirector", "GameplayDesigner", "GameplaySystemsEngineer", "UXExperienceDesigner"]
                for r in last_addressed_to:
                    if r not in pending_invited_roles:
                        pending_invited_roles.append(r)
                if self.max_turns - turns_executed < 3:
                    self.max_turns = turns_executed + 3
            effective_addressed_to = list(dict.fromkeys(list(last_addressed_to) + pending_invited_roles))
            routed_spec = _resolve_council_specialist(pending_routed_role) if pending_routed_role else None
            routed_from_snapshot = pending_routed_from
            routed_reason_snapshot = pending_routed_reason
            pending_routed_role = ""
            pending_routed_from = ""
            pending_routed_reason = ""
            if routed_spec is not None and routed_spec["role"] != last_speaker_role:
                chosen_spec = routed_spec
                listen_check = WritersRoomListenCheck(
                    deliberation=(
                        f"Routed directly into the Writers' Room by {routed_from_snapshot}: "
                        f"{routed_reason_snapshot or 'Domain specialist handoff requested.'}"
                    ),
                    should_interject=True,
                    urgency_score=1.0,
                    intended_contribution_summary=(
                        f"[Routed to you by {routed_from_snapshot}] {routed_reason_snapshot}"
                        if routed_reason_snapshot
                        else f"Routed in by {routed_from_snapshot} to contribute {routed_spec['display_name']} expertise."
                    ),
                )
                self._emit_activity(
                    agent_role=routed_spec["role"],
                    status_text=(
                        f"{routed_from_snapshot} routed the floor directly to "
                        f"{routed_spec['display_name']} ({routed_spec['role']})"
                        + (f": {routed_reason_snapshot[:100]}" if routed_reason_snapshot else "")
                    ),
                    active_agents=[routed_from_snapshot, routed_spec["role"]],
                    phase="council_reroute",
                )
            else:
                chosen_spec, listen_check = self._select_next_speaker(
                    last_speaker_role=last_speaker_role,
                    addressed_to=effective_addressed_to,
                    game_bible=game_bible,
                    dna_summary=dna_summary,
                    spoken_counts=spoken_counts,
                )
            role = chosen_spec["role"]
            display_name = chosen_spec["display_name"]

            proposal = self._execute_specialist_turn(
                specialist=chosen_spec,
                listen_check=listen_check,
                game_bible=game_bible,
                dna_summary=dna_summary,
            )
            utterance = self._apply_turn_proposal(
                game_bible=game_bible,
                speaker_role=role,
                speaker_display_name=display_name,
                proposal=proposal,
                candidate_dir=candidate_dir,
            )
            next_routed_spec, next_handoff_reason = _parse_council_reroute(proposal)
            if next_routed_spec is not None and next_routed_spec["role"] != role:
                pending_routed_role = next_routed_spec["role"]
                pending_routed_from = role
                pending_routed_reason = next_handoff_reason
            spoken_counts[role] = spoken_counts.get(role, 0) + 1
            if role in pending_invited_roles:
                pending_invited_roles.remove(role)
            last_speaker_role = role
            last_addressed_to = utterance.addressed_to
            for r in utterance.addressed_to:
                if r not in pending_invited_roles and spoken_counts.get(r, 0) == 0:
                    pending_invited_roles.append(r)
            turns_executed += 1
            if self.board is not None:
                # Wait to start the background worker scheduler until all required Council specialists have spoken
                # and had the chance to shape, enrich, wire dependencies, or split existing tasks on the board.
                if (
                    self.scheduler is not None
                    and turns_executed >= min(3, self.max_turns)
                    and not self._missing_required_board_specialists(game_bible)
                    and not is_real_genai_client(self.client)
                ):
                    self.scheduler.start()
                self._handle_failed_items()
                self._run_review_round(game_bible, dna_summary, candidate_dir)

            # Check if live Executive Producer steering arrived during the turn before evaluating convergence
            steered_post = self._consume_pending_steering(game_bible, candidate_dir=candidate_dir)
            if steered_post:
                last_speaker_role = "ExecutiveProducer"
                last_addressed_to = ["CreativeDirector", "GameplayDesigner", "GameplaySystemsEngineer", "UXExperienceDesigner"]
                for r in last_addressed_to:
                    if r not in pending_invited_roles:
                        pending_invited_roles.append(r)
                if self.max_turns - turns_executed < 3:
                    self.max_turns = turns_executed + 3
                continue

            # Check convergence once minimum turns have elapsed
            if turns_executed >= self.min_turns_before_convergence:
                convergence = self._evaluate_convergence(game_bible, dna_summary)
                missing_board_specialists = self._missing_required_board_specialists(game_bible)
                has_minimum_artifacts = (
                    bool(game_bible.experience_storyboard)
                    and bool(game_bible.first_60_seconds_hook)
                    and bool(game_bible.diegetic_ui_philosophy)
                    and not validate_game_design(game_bible)
                    and not missing_board_specialists
                )
                if convergence.is_ready_to_lock_bible and has_minimum_artifacts:
                    closing_utt = WritersRoomUtterance(
                        deliberation=convergence.deliberation,
                        turn_index=len(game_bible.writers_room_transcript),
                        speaker_role="CreativeDirector",
                        speaker_display_name="Creative Director",
                        spoken_dialogue=(
                            f"The Living Game Bible for '{game_bible.game_title}' is cohesive and ready "
                            f"(cohesion score: {convergence.experience_cohesion_score:.2f}). Locking the Game Bible!"
                        ),
                        addressed_to=[],
                        bible_updates_summary=["Locked Living Game Bible for production"],
                    )
                    game_bible.writers_room_transcript.append(closing_utt)
                    self._emit_event(
                        "writers_room_utterance",
                        {
                            "deliberation": closing_utt.deliberation,
                            "utterance_id": closing_utt.utterance_id,
                            "turn_index": closing_utt.turn_index,
                            "speaker_role": closing_utt.speaker_role,
                            "speaker_display_name": closing_utt.speaker_display_name,
                            "spoken_dialogue": closing_utt.spoken_dialogue,
                            "addressed_to": [],
                            "bible_updates_summary": closing_utt.bible_updates_summary,
                        },
                    )
                    break
                elif convergence.director_steering_prompt:
                    steering_utt = WritersRoomUtterance(
                        deliberation=convergence.deliberation,
                        turn_index=len(game_bible.writers_room_transcript),
                        speaker_role="CreativeDirector",
                        speaker_display_name="Creative Director",
                        spoken_dialogue=convergence.director_steering_prompt,
                        addressed_to=[],
                        bible_updates_summary=[
                            f"Director Steering Challenge: {', '.join(convergence.missing_or_weak_elements[:3])}"
                        ],
                    )
                    game_bible.writers_room_transcript.append(steering_utt)
                    last_speaker_role = "CreativeDirector"
                    self._emit_event(
                        "writers_room_utterance",
                        {
                            "deliberation": steering_utt.deliberation,
                            "utterance_id": steering_utt.utterance_id,
                            "turn_index": steering_utt.turn_index,
                            "speaker_role": steering_utt.speaker_role,
                            "speaker_display_name": steering_utt.speaker_display_name,
                            "spoken_dialogue": steering_utt.spoken_dialogue,
                            "addressed_to": [],
                            "bible_updates_summary": steering_utt.bible_updates_summary,
                        },
                    )

        # A locked Bible must describe a whole, cohesive game: force-complete any missing design section (Rule 5: never lock silently incomplete).
        if not already_in_production or validate_game_design(game_bible):
            self._complete_missing_design_sections(game_bible, dna_summary, candidate_dir)
        if not game_bible.slice_roadmap:
            onb_intro = (
                game_bible.onboarding_plan.world_introduction
                if game_bible.onboarding_plan and game_bible.onboarding_plan.world_introduction
                else (game_bible.first_60_seconds_hook or game_bible.elevator_pitch or game_bible.game_title)
            )
            verbs_str = (
                ", ".join(game_bible.core_loop.player_verbs)
                if game_bible.core_loop and game_bible.core_loop.player_verbs
                else "core interaction mechanics"
            )
            m2m = (
                game_bible.core_loop.moment_to_moment_loop
                if game_bible.core_loop and game_bible.core_loop.moment_to_moment_loop
                else "acclimating to the world and learning the core mechanics"
            )
            min2min = (
                game_bible.core_loop.minute_to_minute_loop
                if game_bible.core_loop and game_bible.core_loop.minute_to_minute_loop
                else ""
            )
            sess_loop = (
                game_bible.core_loop.session_loop
                if game_bible.core_loop and game_bible.core_loop.session_loop
                else ""
            )
            game_bible.slice_roadmap = [
                GameSliceSpec(
                    deliberation=(
                        "Strategizing Slice 0 as the complete playable game experience realizing all three timescales of the Core Loop "
                        "(moment-to-moment, minute-to-minute, and session loops), player agency/optimization depth, and multi-horizon Goal Hierarchy."
                    ),
                    slice_id="slice_0_full_core_loop",
                    slice_index=0,
                    title=f"{game_bible.game_title} — Full Core Loop & Progression Experience",
                    slice_objective=f"{onb_intro} | Minute-to-minute: {min2min} | Session: {sess_loop}".strip(" |"),
                    player_acclimation_and_mechanics_focus=f"Player verbs ({verbs_str}), moment-to-moment ({m2m}), minute-to-minute ({min2min}), and session loop ({sess_loop})",
                    expansion_from_prior_slice="Complete playable game loop establishing onboarding in the opening 60s and opening immediately into multi-stage systemic gameplay and goal progression.",
                    scope_boundaries=["Build and wire the complete core loop, player toolkit, multi-region world progression, and short/mid/long-term goals."],
                    status="active",
                )
            ]
            game_bible.active_slice_index = 0
        # Transition Living Game Bible to active production before draining the board so resumes skip initial seeding.
        active_s_initial = game_bible.get_active_slice()
        has_unfinished_explicit_slice = bool(
            not existing_status_was_locked
            and game_bible.slice_roadmap
            and active_s_initial is not None
            and str(getattr(active_s_initial, "status", "") or "").lower() in ("active", "in_progress")
        )
        resumed_drained_slice = bool(
            already_in_production
            and self.board is not None
            and self.board.items()
            and self.board.is_drained()
            and not has_unfinished_explicit_slice
        )
        game_bible.status = "in_production"
        self._persist_bible(game_bible, candidate_dir=candidate_dir)
        # ...and every requirement the council raised must be built and approved (or deliberately cancelled).
        if not resumed_drained_slice:
            self._drain_backlog(game_bible, dna_summary, candidate_dir)
        else:
            active_s = game_bible.get_active_slice()
            if active_s is not None and active_s.status != "completed":
                active_s.status = "completed"
                self._persist_bible(game_bible, candidate_dir=candidate_dir)

        if self.target_slices is not None:
            effective_target_slices = max(1, int(self.target_slices))
        elif is_real_genai_client(self.client):
            effective_target_slices = max(2, int(os.environ.get("STUDIO_TARGET_SLICES", "2")))
        else:
            effective_target_slices = 1

        has_pending_slice_feedback = bool(
            (creative_prompt and creative_prompt.strip())
            or self._pending_steering
            or (
                candidate_dir is not None
                and (
                    (Path(candidate_dir) / ".pending_steering.json").is_file()
                    or (Path(candidate_dir) / ".pending_playtest_wishes.json").is_file()
                    or (Path(candidate_dir) / ".pending_council_playtest_reports.json").is_file()
                )
            )
        )
        if resumed_drained_slice and (
            is_real_genai_client(self.client)
            or self.target_slices is not None
            or getattr(self, "enable_slice_reflection_roundtable", False)
            or has_pending_slice_feedback
        ):
            completed_now = sum(1 for s in game_bible.slice_roadmap if s.status == "completed")
            effective_target_slices = max(effective_target_slices, completed_now + 1)

        if resumed_drained_slice and creative_prompt and creative_prompt.strip():
            self.enqueue_user_steering(creative_prompt.strip(), candidate_dir=candidate_dir)

        while sum(1 for s in game_bible.slice_roadmap if s.status == "completed") < effective_target_slices:
            steered_notes = self._consume_pending_steering(game_bible, candidate_dir=candidate_dir)
            slice_feedback = " | ".join(steered_notes) if steered_notes else None
            self.run_slice_expansion(
                game_bible=game_bible,
                dna=dna,
                candidate_dir=candidate_dir,
                slice_directive=slice_feedback,
            )

        # Synthesize final GenesisWorldSpec on the locked GameBibleSpec
        if game_bible.world_spec is None or not already_in_production:
            game_bible.world_spec = self._synthesize_final_world_spec(game_bible, dna, target_episode)
        if game_bible.world_spec and game_bible.world_spec.core_genre:
            game_bible.core_genre = game_bible.world_spec.core_genre
        game_bible.status = "locked"
        self._persist_bible(game_bible, candidate_dir=candidate_dir)
        return game_bible

    def _capture_current_slice_reflection_dossier(
        self,
        game_bible: GameBibleSpec,
        prev_slice: Optional[GameSliceSpec],
        candidate_dir: Optional[Path],
        *,
        dispatched_playtest_session_ids: Optional[Set[str]] = None,
        has_extra_playtest_dossier: bool = False,
    ) -> Tuple[str, List[Any], str]:
        """Empirically inspects the current slice's rendered game state, scene tree, UI/HUD, and scripts
        so the Council can reflect on what is actually on screen before strategizing the next slice."""
        if candidate_dir is None:
            return "(No cartridge directory provided for empirical reflection.)", [], ""
        c_dir = Path(candidate_dir)
        if not c_dir.is_dir():
            return "(Cartridge directory does not exist on disk.)", [], ""

        prev_idx = prev_slice.slice_index if prev_slice is not None else max(0, game_bible.active_slice_index)
        prev_id = prev_slice.slice_id if prev_slice is not None else f"slice_{prev_idx}"
        prev_title = prev_slice.title if prev_slice is not None else f"Slice #{prev_idx}"

        from engine.studio.context_curator import (
            ContextCuratorAgent,
            compute_entrypoint_reachability,
        )
        from engine.studio.proofs import (
            capture_filmstrip,
            format_filmstrip_telemetry_progression,
        )
        from engine.studio.workers import _summarize_cartridge_for_qa

        reach = compute_entrypoint_reachability(c_dir)
        cartridge_summary = _summarize_cartridge_for_qa(c_dir)

        proof_dir = c_dir / "artifacts" / "proofs"
        proof_dir.mkdir(parents=True, exist_ok=True)
        reflection_filmstrip_path = proof_dir / f"filmstrip_slice_{prev_idx}_state_reflection.png"
        rel_reflection_path = f"artifacts/proofs/filmstrip_slice_{prev_idx}_state_reflection.png"

        parts: List[Any] = list(self._concept_mockup_parts(game_bible, c_dir))
        filmstrip_fn = getattr(self, "assembly_filmstrip_fn", None)

        if (c_dir / "project.godot").is_file() and (
            is_real_genai_client(self.client) or filmstrip_fn is not None
        ):
            try:
                if filmstrip_fn is not None:
                    filmstrip_fn(c_dir, [15, 60, 120, 180, 240], reflection_filmstrip_path)
                else:
                    capture_filmstrip(
                        cartridge_dir=c_dir,
                        ticks=[15, 60, 120, 180, 240],
                        out_path=reflection_filmstrip_path,
                        target_scene=reach.get("main_scene") or "res://main.tscn",
                    )
            except Exception as cap_exc:
                logger.warning("Slice reflection boot filmstrip capture fallback check: %s", cap_exc)

        if not reflection_filmstrip_path.is_file():
            fallback_candidates = [
                proof_dir / "filmstrip_slice_assembly_check_pass1.png",
                proof_dir / "filmstrip_slice_assembly_check_pass0.png",
            ] + sorted(proof_dir.glob("filmstrip_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            for fc in fallback_candidates:
                if fc.is_file():
                    reflection_filmstrip_path = fc
                    try:
                        rel_reflection_path = str(fc.relative_to(c_dir))
                    except ValueError:
                        rel_reflection_path = fc.name
                    break

        telemetry_prog = ""
        if reflection_filmstrip_path.is_file():
            img_part = self._load_image_part(reflection_filmstrip_path)
            if img_part is not None:
                parts.append(
                    f"CURRENT RENDERED GAME FILMSTRIP OF COMPLETED `{prev_id}` ('{prev_title}') "
                    f"(file `{reflection_filmstrip_path.name}` — inspect critically for cohesion, intuitive onboarding, "
                    f"visual framing, mechanical depth, and visible short/long-term goals):"
                )
                parts.append(img_part)
            try:
                telemetry_prog = format_filmstrip_telemetry_progression(reflection_filmstrip_path)
            except Exception:
                telemetry_prog = ""

        curated_code_block = ""
        if is_real_genai_client(self.client):
            try:
                curator = ContextCuratorAgent(
                    cartridge_dir=c_dir,
                    client=self.client,
                    model_name=self.model_name,
                    board=self.board,
                    game_bible=game_bible,
                    event_callback=self.event_callback,
                )
                pack = curator.curate_context(
                    target_agent_role="council_reviewer:SliceReflectionCouncil",
                    task_request=(
                        f"Inspect the current state of '{prev_title}' (`{prev_id}`) in `{reach.get('main_scene')}`, "
                        f"`main.gd`, `scenes.gd`, `ui_hud.gd`, and active gameplay modules to diagnose why or where the current "
                        f"build lacks cohesion, intuitive onboarding ('knowing how to play'), mechanical depth, short/long-term goals, "
                        f"or a rewarding sense of accomplishment, so the Council can strategize the next slice."
                    ),
                    item_id=f"slice_reflection_{prev_id}",
                    item_kind="integration",
                )
                curated_code_block = f"{pack.prompt_block}\n"
            except Exception as cur_exc:
                logger.warning("ContextCurator slice reflection curation skipped: %s", cur_exc)

        blind_report_block = ""
        blind_rep_file = c_dir / "artifacts" / f"blind_playtest_report_{prev_id}.json"
        if blind_rep_file.is_file():
            try:
                b_data = json.loads(blind_rep_file.read_text(encoding="utf-8"))
                b_sid = str(b_data.get("session_id") or "")
                b_rep = b_data.get("session_report") or {}
                b_carry = b_data.get("next_slice_carryover_notes") or []
                b_wishes = b_data.get("wishes") or []
                b_bugs_obs = b_rep.get("bugs_and_glitches_observed") or []
                b_vis_defects = b_rep.get("visual_and_ui_defects_observed") or []
                b_pain_pts = b_rep.get("key_pain_points") or []
                b_unans_qs = b_rep.get("unanswered_questions_and_mysteries") or []
                wish_lines = [
                    f"  - [{w.get('category', 'general')}] {w.get('summary', '')}: {w.get('rationale', '')}"
                    for w in b_wishes
                ]
                carry_lines = [f"  - {n}" for n in b_carry]
                if b_sid and b_sid in (dispatched_playtest_session_ids or set()):
                    blind_report_block = ""
                elif has_extra_playtest_dossier:
                    blind_report_block = (
                        f"=== PRIOR AUTOMATED BLIND PLAYTEST SUMMARY FROM `{prev_id}` (full newer playtest dossier attached above) ===\n"
                        f"Overall Fun Rating: {b_rep.get('overall_fun_rating', '?')}/10\n"
                        f"Council Synthesis: {b_data.get('player_experience_synthesis', '')}\n"
                        + ("Carryover Takeaways for Next Slice:\n" + "\n".join(carry_lines) + "\n" if carry_lines else "")
                        + "\n"
                    )
                else:
                    blind_report_block = (
                        f"=== BLIND PLAYTESTER EXPERIENCE REPORT & CARRYOVER FROM `{prev_id}` ===\n"
                        f"Overall Fun Rating: {b_rep.get('overall_fun_rating', '?')}/10\n"
                        f"Overall Experience: {b_rep.get('overall_experience_summary', '')}\n"
                        f"Graphics & Visuals: {b_rep.get('graphics_and_visuals', '')}\n"
                        + (f"Visual & UI Defects Observed: {json.dumps(b_vis_defects)}\n" if b_vis_defects else "")
                        + f"Gameplay & Feel: {b_rep.get('gameplay_and_feel', '')}\n"
                        f"Sound & Audio: {b_rep.get('sound_and_audio', '')}\n"
                        f"Intuitiveness & Onboarding: {b_rep.get('intuitiveness_and_onboarding', '')}\n"
                        f"Goals & Pacing: {b_rep.get('goals_pacing_and_accomplishment', '')}\n"
                        f"World & Atmosphere: {b_rep.get('world_and_atmosphere', '')}\n"
                        + (f"Key Pain Points: {json.dumps(b_pain_pts)}\n" if b_pain_pts else "")
                        + (f"Bugs & Glitches Observed: {json.dumps(b_bugs_obs)}\n" if b_bugs_obs else "")
                        + (f"Unanswered Player Questions: {json.dumps(b_unans_qs)}\n" if b_unans_qs else "")
                        + f"Council Synthesis: {b_data.get('player_experience_synthesis', '')}\n"
                        + ("Carryover Takeaways for Next Slice:\n" + "\n".join(carry_lines) + "\n" if carry_lines else "")
                        + ("Player Wishes Recorded:\n" + "\n".join(wish_lines) + "\n" if wish_lines else "")
                        + "\n"
                    )
            except Exception as br_exc:
                logger.debug("Could not read blind playtest report for %s: %s", prev_id, br_exc)

        asset_integrity_block, _, contact_sheet_part = self._build_asset_integrity_dossier_and_sheet(c_dir, prev_id)
        if contact_sheet_part is not None:
            parts.append(
                f"CARTRIDGE VISUAL ASSET CONTACT SHEET FOR `{prev_id}` "
                "(inspect every sprite, tile, and multi-cell UI/world atlas with cell grid overlays for flat geometric "
                "placeholder shapes, empty/blank atlas cells, or broken silhouettes):"
            )
            parts.append(contact_sheet_part)

        dossier = (
            f"=== EMPIRICAL STATE OF CURRENT GAME (`{prev_id}`: {prev_title}) ===\n"
            f"Playable Entrypoint Reachability:\n{reach.get('summary_text', '')}\n\n"
            f"Rendered Filmstrip Telemetry Progression:\n{telemetry_prog or '(no telemetry progression recorded)'}\n\n"
            f"{blind_report_block}"
            f"{asset_integrity_block}"
            f"Cartridge Scene & Script Architecture Summary:\n{cartridge_summary}\n\n"
            f"{curated_code_block}"
        )
        return dossier, parts, (rel_reflection_path if reflection_filmstrip_path.is_file() else "")

    def _build_asset_integrity_dossier_and_sheet(
        self,
        c_dir: Path,
        slice_tag: str,
    ) -> Tuple[str, Optional[Path], Any]:
        """Renders a neutral observational contact sheet of cartridge `assets/` PNGs and summarizes
        `assets/asset_library.json` metadata plus active in-game `.gd`/`.tscn` reference status so Council
        specialists can inspect every active game asset without unreferenced orphan files crowding them out,
        and clean up obsolete orphan files via `delete_paths` when desired.
        """
        import re
        from PIL import Image
        from engine.supervisor.asset_validator import (
            _collect_observational_asset_metadata,
            render_cartridge_asset_contact_sheet,
        )

        if not c_dir.is_dir():
            return "", None, None
        proof_dir = c_dir / "artifacts" / "proofs"
        proof_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(slice_tag or "current")).strip("_") or "current"
        sheet_out = proof_dir / f"asset_contact_sheet_{safe_tag}.png"

        assets_root = c_dir / "assets"
        visual_files: List[Path] = []
        if assets_root.is_dir():
            for p_png in sorted(assets_root.rglob("*.png")):
                rel_parts = p_png.relative_to(c_dir).parts
                if "mockups" in rel_parts or "playtest_captures" in rel_parts or "concepts" in rel_parts:
                    continue
                visual_files.append(p_png)

        latest_mtime = 0.0
        for p_chk in visual_files:
            try:
                latest_mtime = max(latest_mtime, p_chk.stat().st_mtime)
            except OSError:
                pass
        cat_path = c_dir / "assets" / "asset_library.json"
        if cat_path.is_file():
            try:
                latest_mtime = max(latest_mtime, cat_path.stat().st_mtime)
            except OSError:
                pass
        excluded_ref_dirs = {"tests", "artifacts", ".godot", "mockups", "concepts", "playtest_captures"}
        for ext in ("*.gd", "*.tscn", "*.tres", "*.gdshader"):
            for ref_src in c_dir.rglob(ext):
                try:
                    rel_parts = ref_src.relative_to(c_dir).parts
                    if any(part in excluded_ref_dirs for part in rel_parts):
                        continue
                    latest_mtime = max(latest_mtime, ref_src.stat().st_mtime)
                except OSError:
                    pass

        cache_dict: Dict[Tuple[str, str, float], Tuple[str, Optional[Path], Any]] = getattr(
            self, "_asset_sheet_audit_cache", None
        ) or {}
        self._asset_sheet_audit_cache = cache_dict
        cache_key = (str(c_dir.resolve()), safe_tag, round(latest_mtime, 3))
        if cache_key in cache_dict:
            cached_txt, cached_p, cached_part = cache_dict[cache_key]
            if cached_p is None or cached_p.is_file():
                return cached_txt, cached_p, cached_part

        obs_meta = _collect_observational_asset_metadata(c_dir, visual_files) if visual_files else {}
        active_files: List[Path] = []
        unreferenced_files: List[Path] = []
        for vf in visual_files:
            rel_vf = vf.relative_to(c_dir).as_posix()
            if (obs_meta.get(rel_vf) or {}).get("referenced"):
                active_files.append(vf)
            else:
                unreferenced_files.append(vf)
        unreferenced_files.sort(
            key=lambda p: (
                0 if (obs_meta.get(p.relative_to(c_dir).as_posix()) or {}).get("in_catalog") else 1,
                1 if "/parts/" in p.relative_to(c_dir).as_posix() else 0,
                p.relative_to(c_dir).as_posix(),
            )
        )

        sheet_path: Optional[Path] = None
        sheet_part: Any = None
        if visual_files:
            try:
                sheet_path = render_cartridge_asset_contact_sheet(
                    c_dir,
                    output_path=sheet_out,
                    active_only=bool(active_files and len(unreferenced_files) > 8),
                )
            except Exception as sh_exc:
                logger.debug("Asset contact sheet render skipped: %s", sh_exc)

        if sheet_path is not None and sheet_path.is_file():
            try:
                sheet_part = self._load_image_part(sheet_path)
            except Exception as lp_exc:
                logger.debug("Could not load contact sheet image part: %s", lp_exc)

        cat_entries: Dict[str, Dict[str, Any]] = {}
        if cat_path.is_file():
            try:
                raw_cat = json.loads(cat_path.read_text(encoding="utf-8"))
                raw_entries = raw_cat.get("entries", raw_cat) if isinstance(raw_cat, dict) else raw_cat
                iter_entries = (
                    raw_entries.values()
                    if isinstance(raw_entries, dict)
                    else (raw_entries if isinstance(raw_entries, list) else [])
                )
                for ent in iter_entries:
                    if isinstance(ent, dict):
                        fp = str(
                            ent.get("file_path") or ent.get("res_path") or ent.get("relative_path") or ""
                        ).replace("res://", "").lstrip("/")
                        if fp:
                            cat_entries[fp] = ent
            except Exception:
                pass

        if not visual_files and not cat_entries:
            res_empty = ("", sheet_path, sheet_part)
            cache_dict[cache_key] = res_empty
            return res_empty

        def _format_asset_line(vf: Path, status_tag: str) -> str:
            rel_vf = vf.relative_to(c_dir).as_posix()
            dims_txt = "?"
            try:
                with Image.open(vf) as im:
                    dims_txt = f"{im.width}x{im.height}px ({im.mode})"
            except Exception:
                pass
            ent = cat_entries.get(rel_vf) or {}
            meta = obs_meta.get(rel_vf) or {}
            hf = ent.get("hframes") or meta.get("hframes") or 1
            vf_cnt = ent.get("vframes") or meta.get("vframes") or 1
            perc = str(ent.get("perceptual_description") or "").strip()
            perc_suffix = f" | perceptual_notes: {perc[:180]}" if perc else ""
            return f"  - `{rel_vf}` [{status_tag}]: {dims_txt}, grid={hf}x{vf_cnt}{perc_suffix}"

        lines: List[str] = [
            "=== CARTRIDGE VISUAL ASSET INVENTORY & OBSERVATIONAL CONTACT SHEET ===",
            (
                f"Visual Asset Files on Disk: {len(visual_files)} "
                f"(active_in_game={len(active_files)}, unreferenced_on_disk={len(unreferenced_files)})"
            ),
        ]
        if active_files:
            lines.append(f"Active In-Game Referenced Visual Assets ({len(active_files)} — loaded by production .gd/.tscn):")
            for vf in active_files:
                lines.append(_format_asset_line(vf, "ACTIVE"))
        if unreferenced_files:
            lines.append(
                f"Unreferenced / Orphaned PNG Files on Disk ({len(unreferenced_files)} — NOT loaded by any production .gd/.tscn; "
                f"if obsolete, clean up via `delete_paths` on a specialist task; if intended for gameplay, wire into GDScript):"
            )
            for vf in unreferenced_files[:40]:
                cat_flag = "UNREFERENCED, cataloged" if (obs_meta.get(vf.relative_to(c_dir).as_posix()) or {}).get("in_catalog") else "UNREFERENCED, uncataloged"
                lines.append(_format_asset_line(vf, cat_flag))
            if len(unreferenced_files) > 40:
                lines.append(f"  ... ({len(unreferenced_files) - 40} additional unreferenced PNG file(s) omitted)")
        lines.append("")
        res_tuple = ("\n".join(lines), sheet_path, sheet_part)
        cache_dict[cache_key] = res_tuple
        return res_tuple

    def _run_slice_reflection_and_strategy(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        prev_slice: Optional[GameSliceSpec],
        next_idx: int,
        candidate_dir: Optional[Path],
        slice_directive: Optional[str] = None,
        extra_playtest_dossier: str = "",
        extra_multimodal_parts: Optional[List[Any]] = None,
        *,
        consumed_playtest_session_ids: Optional[Set[str]] = None,
    ) -> Optional[NextSliceStrategyPlan]:
        """Convenes the Council's Slice State Reflection Roundtable to critique the empirical state of the current
        slice from first principles of game design, then has the Creative Director synthesize a master `NextSliceStrategyPlan`
        for `Slice #{next_idx}` that updates the Living Game Bible and Slice Roadmap."""
        if not is_real_genai_client(self.client) and not getattr(self, "enable_slice_reflection_roundtable", False):
            return None

        prev_idx = prev_slice.slice_index if prev_slice is not None else max(0, next_idx - 1)
        prev_id = prev_slice.slice_id if prev_slice is not None else f"slice_{prev_idx}"
        prev_title = prev_slice.title if prev_slice is not None else f"Slice #{prev_idx}"

        self._emit_activity(
            agent_role="QATestEngineer",
            status_text=f"Capturing empirical game state of '{prev_title}' (`{prev_id}`) for Slice #{next_idx} Reflection & Strategy...",
            active_agents=["QATestEngineer", "CreativeDirector", "UXExperienceDesigner", "GameplayDesigner", "LevelAndProgressionDesigner", "VisualArtDirector", "GameplaySystemsEngineer"],
            phase="slice_reflection",
        )
        has_extra_dossier = bool(extra_playtest_dossier and extra_playtest_dossier.strip())
        empirical_dossier, reflection_parts, rel_proof_path = self._capture_current_slice_reflection_dossier(
            game_bible=game_bible,
            prev_slice=prev_slice,
            candidate_dir=candidate_dir,
            dispatched_playtest_session_ids=consumed_playtest_session_ids,
            has_extra_playtest_dossier=has_extra_dossier,
        )
        if extra_playtest_dossier and extra_playtest_dossier.strip():
            empirical_dossier = f"{extra_playtest_dossier.strip()}\n\n{empirical_dossier}"
        if extra_multimodal_parts:
            reflection_parts = list(reflection_parts) + list(extra_multimodal_parts)

        if rel_proof_path:
            qa_idx = len(game_bible.writers_room_transcript)
            qa_utt = WritersRoomUtterance(
                deliberation=f"Captured empirical state filmstrip and scene/code dossier for {prev_id} before strategizing Slice #{next_idx}.",
                utterance_id=f"utt_{qa_idx:02d}_QATestEngineer_slice_{prev_idx}_reflection",
                turn_index=qa_idx,
                speaker_role="QATestEngineer",
                speaker_display_name="Autonomous QA Playtester",
                spoken_dialogue=(
                    f"[Slice #{prev_idx} Checkpoint Empirical Inspection — '{prev_title}'] "
                    f"Presenting the rendered gameplay filmstrip and scene/script inspection of `{prev_id}` so the Council can "
                    f"honestly evaluate what is currently on screen and strategize Slice #{next_idx}."
                ),
                addressed_to=["UXExperienceDesigner", "GameplayDesigner", "LevelAndProgressionDesigner", "VisualArtDirector", "GameplaySystemsEngineer", "CreativeDirector"],
                bible_updates_summary=[f"slice_reflection_capture:{prev_id}"],
                proof_image_paths=[rel_proof_path],
                artifact_emitted={
                    "type": "filmstrip",
                    "path": rel_proof_path,
                    "slice_id": prev_id,
                    "caption": f"Slice #{prev_idx} Empirical State Inspection — {prev_title}",
                },
            )
            game_bible.writers_room_transcript.append(qa_utt)
            self._persist_bible(game_bible, candidate_dir=candidate_dir)
            self._emit_event(
                "writers_room_utterance",
                {
                    "deliberation": qa_utt.deliberation,
                    "utterance_id": qa_utt.utterance_id,
                    "turn_index": qa_utt.turn_index,
                    "speaker_role": qa_utt.speaker_role,
                    "speaker_display_name": qa_utt.speaker_display_name,
                    "spoken_dialogue": qa_utt.spoken_dialogue,
                    "addressed_to": qa_utt.addressed_to,
                    "bible_updates_summary": qa_utt.bible_updates_summary,
                    "proof_image_paths": qa_utt.proof_image_paths,
                    "artifact_emitted": qa_utt.artifact_emitted,
                },
            )

        reflection_roles = [
            "UXExperienceDesigner",
            "GameplayDesigner",
            "LevelAndProgressionDesigner",
            "VisualArtDirector",
            "GameplaySystemsEngineer",
        ]
        self._emit_activity(
            agent_role="CouncilReflectionPanel",
            status_text=f"Council specialists ({', '.join(reflection_roles)}) are reflecting on Slice #{prev_idx} and strategizing Slice #{next_idx}...",
            active_agents=reflection_roles,
            phase="slice_reflection",
        )

        bible_snapshot = self._format_bible_snapshot(
            game_bible,
            omit_playtest_session_ids=consumed_playtest_session_ids,
        )
        eff_slice_directive = (slice_directive or "").strip()
        if eff_slice_directive and (has_extra_dossier or consumed_playtest_session_ids):
            eff_slice_directive = self._compact_playtest_dispatch_text(
                eff_slice_directive,
                note_suffix="full structured playtest report & screenshots attached in dossier below",
            )
        directive_block = (
            f"\nEXECUTIVE PRODUCER / DIRECTOR SLICE DIRECTIVE:\n{eff_slice_directive}\n"
            if eff_slice_directive
            else ""
        )
        first_principles_framework = (
            "=== FIRST-PRINCIPLES GAME DESIGN & SYSTEMIC COHERENCE INQUIRY ===\n"
            "Examine the attached rendered filmstrip, runtime telemetry (`Visible HUD/UI Texts`, `Audio Players Runtime State`, `Live Main Scene Parity`), and code/scene dossier alongside the Target Player DNA and any Executive Producer directive. "
            "Do NOT rubber-stamp the current build or silently downgrade Executive Producer scope directives (Rule 5). Reason from first principles across four questions:\n"
            "1. FIRST-TIME PLAYER MENTAL MODEL (`t=0`): Imagine sitting down at `res://main.tscn` with zero external knowledge. What do you perceive, understand, and try to do in the first 60 seconds? Do every visible UI readout, prompt, and narrative message accurately reflect live game state and make sense in context without unlatched repetition?\n"
            "2. CAUSAL REACHABILITY & LOOP DEPTH: Tracing the dependency chain from the actual `t=0` starting state, is every player verb causally reachable via real inputs without circular prerequisites, and does the loop offer genuine decision/optimization depth?\n"
            "3. SYSTEMIC & EXPERIENTIAL COHERENCE: Do the control/embodiment model, camera, spatial mechanics, audio atmosphere, and world scale make logical sense together as one unified game—and if the Executive Producer gave a directive, does our architecture realize its full intent?\n"
            "4. GOALS & SENSE OF ACCOMPLISHMENT: Are short-, mid-, and long-term goals clear and motivating for this specific player (`SteamPlayerDNA`), and does progress visibly and audibly transform the world?"
        )

        dna_header = self._dna_header_if_absent(dna_summary, bible_snapshot)
        reflection_shared_prefix = (
            f"{dna_header}"
            f"{directive_block}\n"
            f"{STUDIO_CRITICAL_DISCOURSE_CHARTER}\n\n"
            f"{bible_snapshot}\n\n"
            f"{empirical_dossier}\n\n"
            f"{first_principles_framework}\n\n"
        )

        eff_refl_dir = self._resolve_memory_dir(candidate_dir)
        reflection_turns_log: List[str] = []
        with explicit_prefix_cache_scope(self.client, ttl_seconds=600, reuse_outer=False):
            for r_role in reflection_roles:
                spec = next((sp for sp in SPECIALIST_ROSTER if sp["role"] == r_role), None)
                disp_name = spec["display_name"] if spec else r_role
                role_focus = spec["focus"] if spec else ""
                role_mem_block = ""
                if eff_refl_dir is not None and (eff_refl_dir / "game_memory.json").is_file():
                    from engine.studio.game_memory import GameMemoryStore

                    role_mem_block = GameMemoryStore(
                        eff_refl_dir, event_callback=getattr(self, "event_callback", None)
                    ).format_role_memory_briefing(r_role, include_fulfilled_for_reflection=True)
                prior_reflections = (
                    "\n".join(f"  - {line}" for line in reflection_turns_log)
                    if reflection_turns_log
                    else "  (You are opening the Slice State Reflection Roundtable.)"
                )
                r_prompt = (
                    f"{reflection_shared_prefix}"
                    + (f"{role_mem_block}\n\n" if role_mem_block else "")
                    + f"=== SLICE REFLECTION ROUNDTABLE SO FAR ===\n{prior_reflections}\n\n"
                    f"You are the {disp_name} ({r_role}). Your specialty focus: {role_focus}\n"
                    f"Skeptically and candidly audit the completed Slice #{prev_idx} (`{prev_id}`: '{prev_title}')—never open `spoken_dialogue` with 'I completely agree with...' or repeat earlier colleagues' points, and treat any Executive Producer slice directive above with full respect as your primary mandate to cater to. "
                    f"Formulate a realistic, bounded domain strategy for Slice #{next_idx} (cleaning up the highest-impact problem areas from Slice #{prev_idx} + delivering Slice #{next_idx}'s coherent scope without trying to rewrite everything at once). "
                    f"Use `memory_updates` SPARINGLY (never write slice recap summaries or duplicate Bible fields): mark completed `[mem_...]` IDs as `'fulfill'`/`'archive'`, and record ONLY genuinely deferred multi-slice goals or hard-won systemic lessons in your bank (`target_bank='self'`) or `'council'`.\n"
                    f"Return `SliceStateReflectionTurn` with `deliberation` FIRST and `current_slice_quality_score` strictly LAST."
                )
                r_turn = self._structured_with_validation(
                    r_prompt,
                    SliceStateReflectionTurn,
                    lambda _t: [],
                    temperature=0.45,
                    parts=reflection_parts,
                    caller_role=r_role,
                    candidate_dir=eff_refl_dir,
                    slice_id=prev_id,
                    cacheable_prefix=reflection_shared_prefix,
                    min_calls_before_create=1,
                )
                reflection_turns_log.append(
                    f"[{disp_name} ({r_role})] (score={r_turn.current_slice_quality_score:.2f})\n"
                    f"    Critique: {r_turn.honest_current_state_critique}\n"
                    f"    How-to-Play Gaps: {r_turn.how_to_play_and_intuitiveness_gaps}\n"
                    f"    Depth/Fun Gaps: {r_turn.depth_and_fun_gaps}\n"
                    f"    Goals/Accomplishment Gaps: {r_turn.accomplishment_and_goals_gaps}\n"
                    f"    Strategy for Slice #{next_idx}: {r_turn.strategy_for_next_slice}\n"
                    f"    Spoken: {r_turn.spoken_dialogue}"
                )
                t_idx = len(game_bible.writers_room_transcript)
                r_utt = WritersRoomUtterance(
                    deliberation=r_turn.deliberation,
                    utterance_id=f"utt_{t_idx:02d}_{r_role}_slice_{next_idx}_reflection",
                    turn_index=t_idx,
                    speaker_role=r_role,
                    speaker_display_name=disp_name,
                    spoken_dialogue=r_turn.spoken_dialogue,
                    addressed_to=r_turn.addressed_to or ["CreativeDirector", "GameplayDesigner", "UXExperienceDesigner"],
                    bible_updates_summary=[
                        f"slice_reflection:{r_role}:score_{r_turn.current_slice_quality_score:.2f}"
                    ],
                )
                game_bible.writers_room_transcript.append(r_utt)
                self._persist_bible(game_bible, candidate_dir=candidate_dir)
                self._emit_event(
                    "writers_room_utterance",
                    {
                        "deliberation": r_utt.deliberation,
                        "utterance_id": r_utt.utterance_id,
                        "turn_index": r_utt.turn_index,
                        "speaker_role": r_utt.speaker_role,
                        "speaker_display_name": r_utt.speaker_display_name,
                        "spoken_dialogue": r_utt.spoken_dialogue,
                        "addressed_to": r_utt.addressed_to,
                        "bible_updates_summary": r_utt.bible_updates_summary,
                    },
                )

            self._emit_activity(
                agent_role="CreativeDirector",
                status_text=f"Creative Director is synthesizing the master NextSliceStrategyPlan for Slice #{next_idx}...",
                active_agents=["CreativeDirector"],
                phase="slice_reflection",
            )
            director_mem_block = ""
            if eff_refl_dir is not None and (eff_refl_dir / "game_memory.json").is_file():
                from engine.studio.game_memory import GameMemoryStore

                director_mem_block = GameMemoryStore(
                    eff_refl_dir, event_callback=getattr(self, "event_callback", None)
                ).format_role_memory_briefing("CreativeDirector", include_fulfilled_for_reflection=True, max_entries=18)

            director_prompt = (
                f"{reflection_shared_prefix}"
                + (f"{director_mem_block}\n\n" if director_mem_block else "")
                + f"=== COUNCIL SLICE #{prev_idx} REFLECTION ROUNDTABLE ===\n"
                + "\n".join(reflection_turns_log)
                + f"\n\nYou are the Creative Director synthesizing the Council's candid reflection on Slice #{prev_idx} (`{prev_id}`) into an actionable, bounded Master Strategy (`NextSliceStrategyPlan`) for Slice #{next_idx}.\n"
                f"REQUIREMENTS FOR `NextSliceStrategyPlan`:\n"
                f"1. Diagnose the completed slice honestly and unsentimentally in `current_slice_honest_diagnosis` (zero self-congratulation; zero 'I completely agree with everyone' in `spoken_dialogue`), while catering with full respect and fidelity to any Executive Producer slice directive.\n"
                f"2. Keep Slice #{next_idx} realistically scoped—clean up key problem areas carried over from Slice #{prev_idx} and deliver Slice #{next_idx}'s core expansion cleanly rather than trying to change everything in one go—across all five pillars (tailored to `{game_bible.game_title}` and the Target Player DNA):\n"
                f"   - `cohesion_and_world_integration_strategy`: How Slice #{next_idx} unifies visual framing, world space, and HUD instruments so the screen is an unobstructed, living game world (and eliminates any giant center-screen static panels or leaked debug/harness overlays).\n"
                f"   - `how_to_play_and_onboarding_strategy`: How Slice #{next_idx} teaches the player how to play intuitively in the first 60 seconds with diegetic control prompts, clear cause-and-effect, and immediate guidance.\n"
                f"   - `depth_and_interesting_decisions_strategy`: How Slice #{next_idx} deepens gameplay with meaningful decisions, trade-offs, hazards/events, and systemic depth tailored to this player.\n"
                f"   - `short_term_and_long_term_goals_strategy`: How Slice #{next_idx} makes short-term (15–60s), mid-term (3–5m), and long-term (session) goals visibly tracked and causally connected on screen.\n"
                f"   - `sense_of_accomplishment_and_reward_strategy`: How Slice #{next_idx} rewards goal completion with unmistakable visual/audio payoff, world progression, and capability/milestone unlocks.\n"
                f"3. Define `next_slice_spec` (`GameSliceSpec` with `slice_index={next_idx}`, `status='active'`, and a specific `slice_id` like `slice_{next_idx}_...`) and provide updated Bible sections (`updated_core_loop`, `updated_onboarding_plan`, `updated_goal_hierarchy`, `updated_engagement_plan`, `updated_diegetic_ui_philosophy`) whenever they should be strengthened to realize this strategy.\n"
                f"4. Optionally seed initial high-level `backlog_additions` for Slice #{next_idx} (buildable kinds: {self._supported_kinds() or 'all'}).\n"
                f"5. CURATE COUNCIL LONG-TERM MEMORY SPARINGLY (`memory_updates`): Review existing `[mem_...]` entries above, mark delivered/replaced goals as `operation='fulfill'`/`'supersede'`/`'archive'`, and record ONLY genuinely deferred future-slice strategic goals, cross-slice architectural invariants, or studio-wide lessons in `target_bank='council'` (never duplicate `next_slice_spec` or Bible updates).\n"
                f"Return `NextSliceStrategyPlan` with `deliberation` FIRST."
            )

            def _validate_strategy(plan: NextSliceStrategyPlan) -> List[str]:
                issues: List[str] = []
                plan.next_slice_spec.slice_index = next_idx
                plan.next_slice_spec.status = "active"
                if not (plan.next_slice_spec.slice_id or "").strip():
                    plan.next_slice_spec.slice_id = f"slice_{next_idx}_iterative_expansion"
                if plan.backlog_additions and self.board is not None:
                    issues.extend(self.board.validate_proposals(plan.backlog_additions))
                    issues.extend(self._unsupported_kind_issues(plan.backlog_additions))
                return issues

            strategy_plan = self._structured_with_validation(
                director_prompt,
                NextSliceStrategyPlan,
                _validate_strategy,
                temperature=0.4,
                parts=reflection_parts,
                caller_role="CreativeDirector",
                candidate_dir=eff_refl_dir,
                slice_id=prev_id,
                cacheable_prefix=reflection_shared_prefix,
                min_calls_before_create=1,
            )

        # Apply NextSliceStrategyPlan to GameBibleSpec
        updates_summary: List[str] = [f"next_slice_strategy:{strategy_plan.next_slice_spec.slice_id}"]
        if strategy_plan.memory_updates:
            updates_summary.append(f"memories_updated:{len(strategy_plan.memory_updates)}")
        replaced_slice = False
        for i_s, existing_s in enumerate(game_bible.slice_roadmap):
            if existing_s.slice_index == next_idx or existing_s.slice_id == strategy_plan.next_slice_spec.slice_id:
                game_bible.slice_roadmap[i_s] = strategy_plan.next_slice_spec
                replaced_slice = True
                break
        if not replaced_slice:
            game_bible.slice_roadmap.append(strategy_plan.next_slice_spec)
        game_bible.slice_roadmap.sort(key=lambda s: s.slice_index)
        game_bible.active_slice_index = next_idx

        if strategy_plan.updated_core_loop is not None:
            game_bible.core_loop = strategy_plan.updated_core_loop
            updates_summary.append("updated:core_loop")
        if strategy_plan.updated_onboarding_plan is not None:
            game_bible.onboarding_plan = strategy_plan.updated_onboarding_plan
            updates_summary.append("updated:onboarding_plan")
        if strategy_plan.updated_goal_hierarchy is not None:
            game_bible.goal_hierarchy = strategy_plan.updated_goal_hierarchy
            updates_summary.append("updated:goal_hierarchy")
        if strategy_plan.updated_engagement_plan is not None:
            game_bible.engagement_plan = strategy_plan.updated_engagement_plan
            updates_summary.append("updated:engagement_plan")
        if (strategy_plan.updated_diegetic_ui_philosophy or "").strip():
            game_bible.diegetic_ui_philosophy = strategy_plan.updated_diegetic_ui_philosophy.strip()
            updates_summary.append("updated:diegetic_ui_philosophy")

        game_bible.concept_evolution.append(
            ConceptEvolutionEntry(
                deliberation=strategy_plan.deliberation,
                turn_index=len(game_bible.writers_room_transcript),
                author_role="CreativeDirector",
                previous_premise=f"Slice #{prev_idx} ({prev_id}): {prev_title}",
                evolved_premise=f"Slice #{next_idx} ({strategy_plan.next_slice_spec.slice_id}): {strategy_plan.next_slice_spec.title}",
                pivot_reason=(
                    f"Diagnosis: {strategy_plan.current_slice_honest_diagnosis} | "
                    f"How-to-Play: {strategy_plan.how_to_play_and_onboarding_strategy} | "
                    f"Depth: {strategy_plan.depth_and_interesting_decisions_strategy} | "
                    f"Goals & Accomplishment: {strategy_plan.short_term_and_long_term_goals_strategy}; {strategy_plan.sense_of_accomplishment_and_reward_strategy}"
                ),
            )
        )

        if strategy_plan.backlog_additions and self.board is not None:
            for prop in strategy_plan.backlog_additions:
                if prop.kind == "level_map" and "level_layout.json" not in prop.target_modules:
                    prop.target_modules.append("level_layout.json")
            added_items = self.board.add_items(
                strategy_plan.backlog_additions,
                raised_by_role="CreativeDirector",
                slice_id=strategy_plan.next_slice_spec.slice_id,
            )
            updates_summary.extend(f"backlog:+{it.item_id}" for it in added_items)

        dir_idx = len(game_bible.writers_room_transcript)
        dir_utt = WritersRoomUtterance(
            deliberation=strategy_plan.deliberation,
            utterance_id=f"utt_{dir_idx:02d}_CreativeDirector_slice_{next_idx}_strategy",
            turn_index=dir_idx,
            speaker_role="CreativeDirector",
            speaker_display_name="Creative Director",
            spoken_dialogue=strategy_plan.spoken_dialogue,
            addressed_to=["LevelAndProgressionDesigner", "GameplayDesigner", "UXExperienceDesigner", "VisualArtDirector", "GameplaySystemsEngineer"],
            bible_updates_summary=updates_summary,
        )
        game_bible.writers_room_transcript.append(dir_utt)
        self._emit_event(
            "writers_room_utterance",
            {
                "deliberation": dir_utt.deliberation,
                "utterance_id": dir_utt.utterance_id,
                "turn_index": dir_utt.turn_index,
                "speaker_role": dir_utt.speaker_role,
                "speaker_display_name": dir_utt.speaker_display_name,
                "spoken_dialogue": dir_utt.spoken_dialogue,
                "addressed_to": dir_utt.addressed_to,
                "bible_updates_summary": dir_utt.bible_updates_summary,
            },
        )
        self._emit_event(
            "slice_strategy_synthesized",
            {
                "deliberation": strategy_plan.deliberation,
                "slice_index": next_idx,
                "slice_id": strategy_plan.next_slice_spec.slice_id,
                "title": strategy_plan.next_slice_spec.title,
                "current_slice_honest_diagnosis": strategy_plan.current_slice_honest_diagnosis,
                "cohesion_and_world_integration_strategy": strategy_plan.cohesion_and_world_integration_strategy,
                "how_to_play_and_onboarding_strategy": strategy_plan.how_to_play_and_onboarding_strategy,
                "depth_and_interesting_decisions_strategy": strategy_plan.depth_and_interesting_decisions_strategy,
                "short_term_and_long_term_goals_strategy": strategy_plan.short_term_and_long_term_goals_strategy,
                "sense_of_accomplishment_and_reward_strategy": strategy_plan.sense_of_accomplishment_and_reward_strategy,
            },
        )
        self._persist_bible(game_bible, candidate_dir=candidate_dir)
        return strategy_plan

    def run_slice_expansion(
        self,
        game_bible: GameBibleSpec,
        dna: Optional[SteamPlayerDNA] = None,
        candidate_dir: Optional[Path] = None,
        slice_directive: Optional[str] = None,
    ) -> GameBibleSpec:
        """Advances `game_bible` to the next iterative vertical slice:
        1. Marks the prior slice completed.
        2. Empirically inspects the current game state and convenes the Council's Slice State Reflection Roundtable
           + Creative Director `NextSliceStrategyPlan` synthesis (evaluating cohesion, knowing how to play, depth/fun
           for the target player, short/long-term goals, and sense of accomplishment).
        3. Converts the next-slice strategy into concrete specialist Bible updates and ProductionBoard tasks for
           Slice #N+1, shapes them with the full Council, and drains the board for that slice."""
        if self.client is None:
            raise StudioAgentError(
                "CreativeWritersRoom.run_slice_expansion requires an active google.genai Client (Rule 5)"
            )
        if self.scheduler is not None and hasattr(self.scheduler, "stop_dispatch"):
            self.scheduler.stop_dispatch()
        dna_summary = self._format_dna_summary(dna)
        if not dna and getattr(game_bible, "target_player_dna_summary", ""):
            dna_summary = game_bible.target_player_dna_summary

        prev_slice: Optional[GameSliceSpec] = game_bible.get_active_slice()
        for idx_s, s in enumerate(game_bible.slice_roadmap):
            if idx_s <= game_bible.active_slice_index and s.status == "active":
                s.status = "completed"
                prev_slice = s
        if prev_slice is None and game_bible.slice_roadmap:
            completed_slices = [s for s in game_bible.slice_roadmap if s.status == "completed"]
            prev_slice = completed_slices[-1] if completed_slices else game_bible.slice_roadmap[-1]

        next_idx = (
            (max(s.slice_index for s in game_bible.slice_roadmap if s.status == "completed") + 1)
            if any(s.status == "completed" for s in game_bible.slice_roadmap)
            else (len(game_bible.slice_roadmap) if game_bible.slice_roadmap else 0)
        )
        game_bible.active_slice_index = next_idx
        game_bible.status = "in_production"
        found_next_slice = False
        for s in game_bible.slice_roadmap:
            if s.slice_index == next_idx:
                s.status = "active"
                found_next_slice = True
                break
        if not found_next_slice:
            game_bible.slice_roadmap.append(
                GameSliceSpec(
                    deliberation=f"Initializing vertical slice #{next_idx} while Council reflects on Slice #{max(0, next_idx - 1)} and synthesizes strategy.",
                    slice_id=f"slice_{next_idx}_planning",
                    slice_index=next_idx,
                    title=f"Slice #{next_idx}: Strategizing Next Iteration...",
                    slice_objective=(
                        slice_directive.strip()
                        if (slice_directive and slice_directive.strip())
                        else f"Council is reflecting on Slice #{max(0, next_idx - 1)} and formulating the Slice #{next_idx} strategy..."
                    ),
                    player_acclimation_and_mechanics_focus="Council reflection & strategy synthesis in progress.",
                    expansion_from_prior_slice=f"Iterative expansion from Slice #{max(0, next_idx - 1)}.",
                    scope_boundaries=[],
                    status="active",
                )
            )
            game_bible.slice_roadmap.sort(key=lambda s: s.slice_index)
        self._persist_bible(game_bible, candidate_dir=candidate_dir)

        pre_wish_block, pre_wish_parts = self._consume_pending_playtest_wishes(candidate_dir)
        pre_pt_sids = set(getattr(self, "_last_consumed_playtest_session_ids", None) or ())

        strategy_plan = self._run_slice_reflection_and_strategy(
            game_bible=game_bible,
            dna_summary=dna_summary,
            prev_slice=prev_slice,
            next_idx=next_idx,
            candidate_dir=candidate_dir,
            slice_directive=slice_directive,
            extra_playtest_dossier=pre_wish_block,
            extra_multimodal_parts=pre_wish_parts or None,
            consumed_playtest_session_ids=pre_pt_sids,
        )

        steered_post_reflection = self._consume_pending_steering(game_bible, candidate_dir=candidate_dir)
        post_wish_block, post_wish_parts = self._consume_pending_playtest_wishes(candidate_dir)
        post_pt_sids = set(getattr(self, "_last_consumed_playtest_session_ids", None) or ())
        exp_consumed_pt_sids = pre_pt_sids | post_pt_sids
        exp_wish_block = "\n\n".join(b for b in (pre_wish_block, post_wish_block) if b)
        exp_wish_parts = list(pre_wish_parts) + list(post_wish_parts)
        raw_slice_directives = [
            s.strip()
            for s in ([slice_directive] if slice_directive and slice_directive.strip() else []) + list(steered_post_reflection)
            if s and s.strip()
        ]
        combined_slice_directives = [
            self._compact_playtest_dispatch_text(
                s,
                note_suffix="full structured playtest report & screenshots attached in dossier",
            )
            if (exp_wish_block or exp_consumed_pt_sids)
            else s
            for s in raw_slice_directives
        ]
        directive_note = (
            f" User/Director expansion directive (HONOR AT FULL ARCHITECTURAL SCOPE — ZERO SILENT DOWNGRADING): {' | '.join(combined_slice_directives)}"
            if combined_slice_directives
            else ""
        )
        exp_asset_block = ""
        if candidate_dir is not None:
            prev_tag = prev_slice.slice_id if prev_slice is not None else f"slice_{max(0, next_idx - 1)}"
            exp_asset_block, _, exp_sheet_part = self._build_asset_integrity_dossier_and_sheet(
                Path(candidate_dir), prev_tag
            )
            if is_real_genai_client(self.client) and exp_sheet_part is not None:
                exp_wish_parts.append(
                    f"CARTRIDGE VISUAL ASSET CONTACT SHEET FOR `{prev_tag}` "
                    "(inspect all sprites, tiles, and multi-cell UI/world atlases; raise in-place `visual_asset` "
                    "tasks for any placeholder/broken PNGs and dependent `.gd` reintegration tasks):"
                )
                exp_wish_parts.append(exp_sheet_part)

        strategy_context_block = exp_wish_block
        if strategy_plan is not None:
            strategy_context_block = (
                (f"{exp_wish_block}\n\n" if exp_wish_block else "")
                + (f"{exp_asset_block}\n" if exp_asset_block else "")
                + f"=== MASTER STRATEGY FOR SLICE #{next_idx} (`{strategy_plan.next_slice_spec.slice_id}`: {strategy_plan.next_slice_spec.title}) ===\n"
                f"- Honest Diagnosis of Prior Slice: {strategy_plan.current_slice_honest_diagnosis}\n"
                f"- Cohesion & World Integration Strategy: {strategy_plan.cohesion_and_world_integration_strategy}\n"
                f"- How-to-Play & Intuitive Onboarding Strategy: {strategy_plan.how_to_play_and_onboarding_strategy}\n"
                f"- Mechanical Depth & Interesting Decisions Strategy: {strategy_plan.depth_and_interesting_decisions_strategy}\n"
                f"- Short-Term & Long-Term Goals Strategy: {strategy_plan.short_term_and_long_term_goals_strategy}\n"
                f"- Sense of Accomplishment & Reward Strategy: {strategy_plan.sense_of_accomplishment_and_reward_strategy}\n"
                + (
                    f"- LIVE EXECUTIVE PRODUCER DIRECTIVE (MANDATORY — DO NOT SHRINK SCOPE): {' | '.join(combined_slice_directives)}\n"
                    if combined_slice_directives
                    else ""
                )
                + f"Ensure your `backlog_additions` and Bible updates directly implement this strategy in the playable game (`res://main.tscn`)!"
            )
        elif exp_asset_block:
            strategy_context_block = (
                f"{exp_wish_block}\n\n{exp_asset_block}" if exp_wish_block else exp_asset_block
            )

        expansion_roles: List[str] = ["LevelAndProgressionDesigner", "GameplayDesigner"]
        if is_real_genai_client(self.client):
            expansion_roles.extend(["VisualArtDirector", "UXExperienceDesigner"])
        expansion_roles.append("GameplaySystemsEngineer")

        exp_base_turn_idx = len(game_bible.writers_room_transcript)
        exp_base_bible_snapshot = self._format_bible_snapshot(
            game_bible,
            omit_playtest_session_ids=exp_consumed_pt_sids,
        )
        expansion_shared_prefix = self._build_specialist_shared_prefix(
            dna_summary,
            exp_base_bible_snapshot,
            strategy_context_block,
        )
        expansion_min_calls = 1 if len(expansion_roles) >= 2 else 2

        with explicit_prefix_cache_scope(self.client, ttl_seconds=600, reuse_outer=False):
            for idx_r, role_name in enumerate(expansion_roles):
                steered_mid_expansion = self._consume_pending_steering(game_bible, candidate_dir=candidate_dir)
                if steered_mid_expansion:
                    self._execute_steering_pivot_turns(game_bible, dna_summary, candidate_dir, steered_mid_expansion)
                    combined_slice_directives.extend(steered_mid_expansion)
                    directive_note = (
                        f" User/Director expansion directive (HONOR AT FULL ARCHITECTURAL SCOPE — ZERO SILENT DOWNGRADING): {' | '.join(combined_slice_directives)}"
                    )
                    exp_base_turn_idx = len(game_bible.writers_room_transcript)
                    exp_base_bible_snapshot = self._format_bible_snapshot(
                        game_bible,
                        omit_playtest_session_ids=exp_consumed_pt_sids,
                    )
                    expansion_shared_prefix = self._build_specialist_shared_prefix(
                        dna_summary,
                        exp_base_bible_snapshot,
                        strategy_context_block,
                    )
                spec = next((sp for sp in SPECIALIST_ROSTER if sp["role"] == role_name), None)
                if spec is None:
                    continue
                active_slice_now = game_bible.get_active_slice()
                active_slice_id_now = active_slice_now.slice_id if active_slice_now is not None else f"slice_{next_idx}"
                has_unstarted_for_slice = bool(
                    self.board is not None
                    and any(
                        i.state in ("ready", "blocked")
                        for i in self.board.items()
                    )
                )
                must_seed_on_last = (idx_r == len(expansion_roles) - 1) and not has_unstarted_for_slice
                check = WritersRoomListenCheck(
                    deliberation=(
                        f"Advancing to iterative Slice #{next_idx} (`{active_slice_id_now}`): translating the Council's "
                        f"reflection on Slice #{max(0, next_idx - 1)} and next-slice strategy (cohesion, intuitive onboarding, "
                        f"mechanical depth, short/long-term goals, and sense of accomplishment) into concrete Bible updates and "
                        f"ProductionBoard tasks for Slice #{next_idx}.{directive_note}"
                    ),
                    should_interject=True,
                    urgency_score=1.0,
                    intended_contribution_summary=(
                        f"Strategize Slice #{next_idx} in `added_or_updated_slices` (with `slice_index={next_idx}`, `status='active'`, "
                        f"and `expansion_from_prior_slice` explaining how it transforms Slice #{max(0, next_idx - 1)} into a more cohesive, "
                        f"intuitive, deep, goal-driven, and rewarding game) and raise `backlog_additions` ONLY for Slice #{next_idx}.{directive_note}"
                    ),
                )
                extra_exp_kwargs: Dict[str, Any] = {}
                if getattr(self._execute_specialist_turn, "__func__", None) is CreativeWritersRoom._execute_specialist_turn:
                    extra_exp_kwargs = {
                        "frozen_bible_snapshot": exp_base_bible_snapshot,
                        "recent_turn_delta": self._format_utterance_delta(game_bible, exp_base_turn_idx),
                        "cacheable_prefix": expansion_shared_prefix,
                        "min_calls_before_create": expansion_min_calls,
                        "omit_playtest_session_ids": exp_consumed_pt_sids,
                    }
                proposal = self._execute_specialist_turn(
                    specialist=spec,
                    listen_check=check,
                    game_bible=game_bible,
                    dna_summary=dna_summary,
                    require_backlog_addition=must_seed_on_last,
                    extra_context_block=strategy_context_block,
                    extra_multimodal_parts=exp_wish_parts or None,
                    **extra_exp_kwargs,
                )
                self._apply_turn_proposal(
                    game_bible=game_bible,
                    speaker_role=role_name,
                    speaker_display_name=spec["display_name"],
                    proposal=proposal,
                    candidate_dir=candidate_dir,
                )

        self._drain_backlog(game_bible, dna_summary, candidate_dir)
        game_bible.status = "locked"
        self._persist_bible(game_bible, candidate_dir=candidate_dir)
        return game_bible

    def inject_user_steering_and_continue(
        self,
        game_bible: GameBibleSpec,
        user_feedback: str,
        dna: Optional[SteamPlayerDNA] = None,
        candidate_dir: Optional[Path] = None,
        additional_turns: Optional[int] = None,
    ) -> GameBibleSpec:
        """Injects user steering notes into the Writers' Room transcript and runs additional refinement turns."""
        turn_index = len(game_bible.writers_room_transcript)
        user_utt = WritersRoomUtterance(
            deliberation="User injected creative steering feedback from Mission Control into the Writers' Room.",
            turn_index=turn_index,
            speaker_role="UserSteering",
            speaker_display_name="Executive Producer (User)",
            spoken_dialogue=user_feedback,
            addressed_to=["CreativeDirector", "GameplayDesigner", "UXExperienceDesigner", "VisualArtDirector"],
            bible_updates_summary=["Executive Producer steering note injected"],
        )
        game_bible.writers_room_transcript.append(user_utt)
        evo = ConceptEvolutionEntry(
            deliberation="Concept pivot initiated by Executive Producer steering feedback.",
            turn_index=turn_index,
            author_role="UserSteering",
            previous_premise=game_bible.elevator_pitch or game_bible.game_title,
            evolved_premise=f"Incorporating user steering: {user_feedback[:120]}",
            pivot_reason=user_feedback,
        )
        game_bible.concept_evolution.append(evo)
        self._emit_event(
            "writers_room_utterance",
            {
                "deliberation": user_utt.deliberation,
                "utterance_id": user_utt.utterance_id,
                "turn_index": user_utt.turn_index,
                "speaker_role": user_utt.speaker_role,
                "speaker_display_name": user_utt.speaker_display_name,
                "spoken_dialogue": user_utt.spoken_dialogue,
                "addressed_to": user_utt.addressed_to,
                "bible_updates_summary": user_utt.bible_updates_summary,
            },
        )
        prev_max = self.max_turns
        if additional_turns is not None:
            self.max_turns = max(1, int(additional_turns))
        try:
            return self.run_writers_room(
                dna=dna,
                candidate_dir=candidate_dir,
                target_episode=game_bible.target_episode,
                cartridge_id=game_bible.cartridge_id,
                existing_bible=game_bible,
            )
        finally:
            self.max_turns = prev_max

