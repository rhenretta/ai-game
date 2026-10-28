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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.steam.models import SteamPlayerDNA
from engine.studio.models import (
    CharacterSheet,
    ConceptCandidate,
    ConceptCollisionProposal,
    ConceptEvolutionEntry,
    ConceptFragment,
    ConceptSeedProposal,
    ConceptSelection,
    CreativeProvocationSet,
    validate_game_design,
    DirectorConvergenceCheck,
    ExperienceStoryboardBeat,
    GameBibleSpec,
    GameSliceSpec,
    GenesisWorldSpec,
    MechanicEngineeringSpec,
    NarrativeBible,
    NoveltyJudgement,
    ReviewBatch,
    ReviewDecision,
    SliceAssemblyDecision,
    StudioAgentError,
    TaskShapingBatch,
    VisualMockupArtifact,
    WorkItemRefinement,
    WritersRoomListenCheck,
    WritersRoomTurnProposal,
    WritersRoomUtterance,
    bounded_thinking_config,
    call_gemini_with_transient_retry,
    is_real_genai_client,
)

from engine.studio.production_board import MAX_SPEC_RESETS, BoardScheduler, ProductionBoard

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
        # Independent LLM calls (listen checks, concept seeds, novelty judgements) run concurrently.
        env_par = os.environ.get("STUDIO_ROOM_PARALLELISM")
        self.max_parallel_calls = max(1, int(max_parallel_calls if max_parallel_calls is not None else (env_par or 2)))
        self._steering_lock = threading.Lock()
        self._pending_steering: List[str] = []

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

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Emits a rationale-first telemetry event to Mission Control and TraceLogger."""
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
    ) -> Any:
        """Invokes Gemini with structured JSON schema and high creative temperature, returning validated Pydantic instance.
        `parts` (e.g. proof images) are sent before the prompt text."""
        if self.client is None:
            raise StudioAgentError(
                "CreativeWritersRoom requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)"
            )
        from google.genai import types

        effective_temp = self.creative_temperature if temperature is None else float(temperature)
        schema_name = getattr(schema_cls, "__name__", str(schema_cls))
        thinking_budget = (
            384
            if schema_name == "WritersRoomListenCheck"
            else (1024 if schema_name == "NoveltyJudgement" else 2048)
        )
        response = call_gemini_with_transient_retry(
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=(list(parts) + [prompt]) if parts else prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_cls,
                    temperature=effective_temp,
                    thinking_config=bounded_thinking_config(thinking_budget),
                ),
            ),
            label=f"CreativeWritersRoom[{schema_name}]",
        )
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

    def _format_bible_snapshot(self, game_bible: GameBibleSpec) -> str:
        recent_turns = game_bible.writers_room_transcript[-6:]
        transcript_text = "\n".join(
            f"  [Turn {u.turn_index}] {u.speaker_display_name} ({u.speaker_role}): {u.spoken_dialogue}"
            for u in recent_turns
        ) or "  (Room just opened)"
        chars = game_bible.narrative_bible.characters if game_bible.narrative_bible else []
        char_names = [f"{c.name} [character_id={c.character_id}, role={c.role}]" for c in chars]
        beats = [f"Beat {b.beat_index}: {b.phase_title} ({b.player_emotional_state})" for b in game_bible.experience_storyboard]
        mockups = [f"{m.title} ({m.image_path})" for m in game_bible.visual_mockups]
        eng_specs = [
            f"{e.mechanic_id} ({e.title}) -> nodes={e.godot_node_and_collider_architecture[:80]}, assets={e.required_sprite_and_tile_assets}"
            for e in getattr(game_bible, "technical_architecture_blueprint", [])
        ]
        slice_summaries = [
            f"[{s.slice_id} idx={s.slice_index} status={s.status}] '{s.title}': objective={s.slice_objective[:120]} | focus={s.player_acclimation_and_mechanics_focus[:120]}"
            for s in getattr(game_bible, "slice_roadmap", [])
        ]
        active_slice = game_bible.get_active_slice() if hasattr(game_bible, "get_active_slice") else None
        active_slice_str = (
            f"[{active_slice.slice_id} (Iteration #{active_slice.slice_index})] '{active_slice.title}' — "
            f"Objective: {active_slice.slice_objective} | Acclimation & Full-Loop Focus: {active_slice.player_acclimation_and_mechanics_focus} | "
            f"Expansion: {active_slice.expansion_from_prior_slice} | Required Systems & Scope: {active_slice.scope_boundaries}"
            if active_slice is not None
            else "(not yet strategized — Slice 0 MUST deliver the complete playable game loop realizing all 3 timescales of `core_loop`, `player_agency_and_optimization_depth`, multi-horizon `goal_hierarchy`, multi-region world progression, UX onboarding, and audio—never defer the core game loop to future slices!)"
        )
        exec_directives: List[str] = []
        active_seed = (getattr(self, "_active_creative_prompt", None) or "").strip()
        if not active_seed and game_bible.concept_evolution:
            first_prev = (game_bible.concept_evolution[0].previous_premise or "").strip()
            if first_prev and first_prev != "Untitled Concept":
                active_seed = first_prev
        if active_seed:
            exec_directives.append(active_seed)
        for u in game_bible.writers_room_transcript:
            if u.speaker_role in ("ExecutiveProducer", "UserSteering") and u.spoken_dialogue and u.spoken_dialogue.strip():
                d_txt = u.spoken_dialogue.strip()
                if d_txt not in exec_directives:
                    exec_directives.append(d_txt)
        exec_banner = (
            "🚨 MANDATORY EXECUTIVE PRODUCER CREATIVE STEERING (HIGHEST PRIORITY — OVERRIDES CONFLICTING PREMISES):\n"
            + "\n".join(f'  - "{d}"' for d in exec_directives)
            + "\n  Every specialist MUST adapt the game's core loop, pacing, fail/recovery rules, controls, and Production Board items to honor this Executive Producer direction!\n\n"
            if exec_directives
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
        if proposal.backlog_additions:
            if self.board is None:
                raise StudioAgentError("A backlog_additions proposal arrived but no Production Board is attached.")
            active_slice_obj = game_bible.get_active_slice()
            default_slice_id = active_slice_obj.slice_id if active_slice_obj is not None else ""
            for prop in proposal.backlog_additions:  # level maps all rewrite the shared layout file: serialise them
                if prop.kind == "level_map" and "level_layout.json" not in prop.target_modules:
                    prop.target_modules.append("level_layout.json")
                elif prop.kind in ("visual_asset", "animation", "audio_asset"):
                    prop.target_modules = [m for m in prop.target_modules if not str(m).endswith(".gd")]
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

        if self.board is not None and updates_summary:
            self.board.note_design_revision()

        utterance = WritersRoomUtterance(
            deliberation=proposal.deliberation,
            turn_index=turn_index,
            speaker_role=speaker_role,
            speaker_display_name=speaker_display_name,
            spoken_dialogue=proposal.spoken_dialogue,
            addressed_to=proposal.addressed_to,
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

        prompt = (
            f"PLAYER DNA & STUDIO CONTEXT:\n{dna_summary}\n\n"
            f"{bible_snapshot}\n\n"
            f"You are the {display_name} ({role}) sitting in an open-ended Game Studio Writers' Room.\n"
            f"Your specialty focus: {focus}\n\n"
            f"{subagent_clause}"
            f"Listen to the conversation above and inspect the Living Game Bible.\n"
            f"Strategize what this specific game needs most right now from your domain (or from one of your Sub-Agents if applicable): "
            f"whether challenging a weak or generic idea, answering a colleague who called on you, or shaping the game's design, world, "
            f"script/dialogue, visuals, onboarding/UI, progression, audio, or technical architecture.\n"
            f"Return your structured WritersRoomListenCheck with `deliberation` FIRST."
        )
        return self._call_structured_llm(prompt, WritersRoomListenCheck, temperature=0.4)

    def _execute_specialist_turn(
        self,
        specialist: Dict[str, Any],
        listen_check: WritersRoomListenCheck,
        game_bible: GameBibleSpec,
        dna_summary: str,
        extra_context_block: str = "",
        require_backlog_addition: bool = False,
    ) -> WritersRoomTurnProposal:
        """Executes the chosen specialist's turn to speak to the room and update the Living Game Bible."""
        role = specialist["role"]
        display_name = specialist["display_name"]
        focus = specialist["focus"]
        subagents = specialist.get("subagents") or []
        parent_lead = specialist.get("parent_lead")
        bible_snapshot = self._format_bible_snapshot(game_bible)

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

        prompt = (
            f"PLAYER DNA & STUDIO CONTEXT:\n{dna_summary}\n\n"
            f"{bible_snapshot}\n\n"
            f"You are the {display_name} ({role}) speaking in the Game Studio Writers' Room.\n"
            f"Your specialty focus: {focus}\n"
            f"{hierarchy_note}"
            f"Context for your turn: {listen_check.intended_contribution_summary}\n\n"
            f"{domain_instructions}\n"
            f"STRATEGIZE WHAT IS BEST FOR THIS GAME (NO PRESCRIPTIONS): Do not follow rote templates. Reason from first principles about what mechanics, "
            f"world structure, empathetic new-player onboarding/UI, narrative/script moments, hybrid visual rendering (combining procedural geometry/shaders/particles "
            f"with generated surface textures, decals, and sprite/rig assets), and iterative vertical slices (`added_or_updated_slices`) best serve THIS specific game.\n"
            f"THE CONCEPT IS STILL EVOLVING: the premise above is provisional. Do not merely decorate it. Pressure-test it from your domain; "
            f"if your contribution exposes a weakness or a better direction, change it (title, pitch, core fantasy, mechanics) via the `updated_*` "
            f"fields and log a `concept_evolution_milestone` explaining what changed and why. Keep what is alive, replace what is generic.\n"
            f"CLARITY & CONCRETE GAME DESIGN (NO PSEUDO-SCIENTIFIC JARGON SALAD): Speak in clear, direct, plain-spoken game design language. "
            f"Always describe concretely what the player sees on screen, what they directly control or manipulate, what obstacle or challenge pushes back, "
            f"and why the moment-to-moment interaction is fun. Avoid obscure academic, physics, or materials-science buzzword mashups that obscure the actual game.\n"
            f"Speak naturally, creatively, and passionately to your fellow designers in `spoken_dialogue` (never recite raw telemetry floats), "
            f"call out specific colleagues or Sub-Agents in `addressed_to`, and contribute your concrete updates to the Living Game Bible. Provide `deliberation` FIRST."
        )
        turn_schema = WritersRoomTurnProposal.for_role(role) if is_real_genai_client(self.client) else WritersRoomTurnProposal
        if self.board is None:
            return self._call_structured_llm(prompt, turn_schema)
        board_block = (
            f"{self.board.snapshot_text()}\n"
            f"BUILD BACKLOG (ACTIVE SLICE — COUNCIL TASK SHAPING, ENRICHMENT & DECOMPOSITION):\n"
            f"Before tasks are built by workers, the Council shapes what is needed so the team builds a cohesive, playable game rather than disconnected subsystems.\n"
            f"1. INSPECT EXISTING TASKS ON THE PRODUCTION BOARD (`backlog_refinements`):\n"
            f"   - Review every unstarted (`ready` or `blocked`) item currently on the Production Board from your specialist perspective.\n"
            f"   - ENRICH (`action='enrich'`): If an existing task needs direction from your domain—such as how it connects to the world/levels, what shared data contracts or node structures it must expose, how the player discovers and interacts with it in the HUD/UI without overlapping panels, or which visual rigs/audio cues it should bind—add a `WorkItemRefinement` in `backlog_refinements` with `action='enrich'`, `spec_addition_or_rewrite`, and any prerequisite `add_depends_on`.\n"
            f"   - SPLIT (`action='split'`): If an existing task is too monolithic (for example, bundling core simulation physics, player placement/building tools, world progression, and HUD into one giant script task, or bundling multiple distinct rigs into one asset task), decompose it via `WorkItemRefinement` with `action='split'` and `split_items=[...]` into focused, dependency-linked tasks! Any downstream tasks on the board that depended on the original task will automatically re-point to the split tasks.\n"
            f"2. RAISE NEW OR COMPANION TASKS (`backlog_additions`):\n"
            f"   - Raise `backlog_additions` needed for the ACTIVE SLICE so Slice 0 ships as a complete, cohesive playable vertical slice (world/level traversal with proper camera tracking, core gameplay loop, empathetic new-player onboarding/HUD consolidated on a single authoritative CanvasLayer, visual assets/surfaces, and audio).\n"
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
            f"Buildable kinds right now: {self._supported_kinds() or 'all'}.\n"
        )

        def _validate(p: WritersRoomTurnProposal) -> List[str]:
            issues: List[str] = []
            if require_backlog_addition and not p.backlog_additions:
                issues.append(
                    "The Executive Producer reported a live issue or steering directive on a drained Production Board; "
                    "you MUST populate `backlog_additions` with at least one concrete `WorkItemProposal` (`integration`, `mechanic_code`, or `ui`) "
                    "targeting the responsible script(s) so the worker pool executes the fix."
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

        full_turn_prompt = prompt + ("\n\n" + extra_context_block if extra_context_block else "") + "\n" + board_block
        return self._structured_with_validation(full_turn_prompt, turn_schema, _validate)

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

    def _proof_parts(self, item: Any, candidate_dir: Optional[Path]) -> List[Any]:
        from google.genai import types

        if not item.revisions:
            return []
        latest = item.revisions[-1]
        parts: List[Any] = []
        for rel in latest.proof_paths:
            path = Path(rel)
            if not path.is_absolute() and candidate_dir is not None:
                path = Path(candidate_dir) / rel
            if not path.is_file():
                raise StudioAgentError(f"Proof image for item '{item.item_id}' is missing on disk: {path}")
            suffix = path.suffix.lower()
            if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
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
            parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
        if not latest.proof_paths:
            parts.append(f"NOTE: item '{item.item_id}' returned NO proof images.")
        return parts

    @staticmethod
    def _format_item_for_review(item: Any) -> str:
        history = "\n".join(
            f"      rev {r.revision_index + 1}: UNVERIFIED_WORKER_CLAIMS={r.worker_notes[:1200]!r} verdict={r.verdict or '(pending)'} feedback={r.feedback[:600]!r}"
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
        return (
            f"  ITEM [{item.item_id}] kind={item.kind} title={item.title!r} state={item.state}\n"
            f"    spec: {item.spec}\n    proof_spec: {item.proof_spec}\n"
            f"    target_modules: {getattr(item, 'target_modules', [])} | depends_on: {item.depends_on}{excluded_line}\n"
            f"    revision_cap: {item.max_revisions} (spec rewrites so far: {item.spec_resets}/{MAX_SPEC_RESETS})\n"
            f"    decision_reason: {item.decision_reason or '-'}{failure_line}\n    history:\n{history}"
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
                "Use the external QA contact-sheet header/footer strips ('[QA] tick ... in:[...]', '[QA] pos=(...) ...') above and below each 16:9 frame "
                "as ground-truth engine telemetry to verify coordinates, inputs, and state changes—never flag those external QA margin strips (or the external checkerboard, cyan border, and yellow baseline guide line in `plate_*.png` / `anim_sheet_*.png`) as defects.\n"
            )
        return ""

    def _review_as(
        self,
        role: str,
        awaiting: List[Any],
        deciding: List[Any],
        game_bible: GameBibleSpec,
        dna_summary: str,
        candidate_dir: Optional[Path],
    ) -> ReviewBatch:
        spec = next((sp for sp in SPECIALIST_ROSTER if sp["role"] == role), None)
        who = f"the {spec['display_name']} ({role})" if spec else "the Creative Director"
        parts: List[Any] = []
        if any(it.kind in ("level_map", "mechanic_code", "ui", "integration", "visual_asset", "animation") for it in awaiting + deciding):
            parts.extend(self._concept_mockup_parts(game_bible, candidate_dir))
        for it in awaiting + deciding:
            parts.extend(self._proof_parts(it, candidate_dir))
        awaiting_ids = {i.item_id for i in awaiting}
        deciding_ids = {i.item_id for i in deciding}
        by_id = {i.item_id: i for i in awaiting + deciding}
        mandate = self._role_review_mandate(role)
        filmstrip_guide = (
            "HOW TO READ `filmstrip_*.png` QA CONTACT SHEETS & VERIFY WORKER CLAIMS:\n"
            "- Each `filmstrip_*.png` proof is a multi-frame grid captured by the external QA Test Harness (640x360 per frame).\n"
            "- Above each 16:9 game frame, the test harness prints the simulation tick and injected controller inputs (`[QA] tick N  in:[...]`).\n"
            "- Below each 16:9 game frame, the test harness prints the live engine state snapshot (`[QA] pos=(...) ...`) specifically FOR YOU (the Council) to verify coordinates and state variables.\n"
            "- These top and bottom margin bands outside the 16:9 frame border are external QA contact-sheet annotations—they are NOT rendered inside the game viewport and must NEVER be rejected as 'developer debug text overlays'.\n"
            "- SKEPTICAL VISUAL VERIFICATION OF `UNVERIFIED_WORKER_CLAIMS`: The `UNVERIFIED_WORKER_CLAIMS` in an item's revision history are self-reported worker notes and pre-capture expectations—NEVER approve an item because `UNVERIFIED_WORKER_CLAIMS` claims a fix was made! You MUST visually verify every single point of prior revision `feedback` directly in the pixels of the attached proof image (and check any `QA Post-Capture Visual Audit` notes). Immediately `request_changes` if the proof image shows clipped/invisible sprite fragments (from wrong `hframes`/`vframes` slicing), shrunk background rectangles that leave dark void around the play area, or crude procedural polygons where fabricated textures/sprites were requested.\n\n"
        )
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
            from engine.studio.context_curator import ContextCuratorAgent, is_real_genai_client
            if is_real_genai_client(self.client):
                review_items = awaiting + deciding
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
                    target_agent_role=f"council_reviewer:{role}",
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
        shared_review_context = (
            f"{board_review_snapshot}"
            f"{on_disk_sprites_block}"
            f"{curated_review_block}"
            f"{self._format_bible_snapshot(game_bible)}\n"
        )
        parts.append(shared_review_context)
        cross_task_triage_guide = (
            "CROSS-TASK ROOT-CAUSE ATTRIBUTION & SCOPE ISOLATION (PREVENTING MULTI-TASK PING-PONG):\n"
            "- Before writing `feedback` on an item, check `FULL PRODUCTION BOARD STATE` above to see whether any visual or behavioral defect visible in the proof belongs to THIS item's core responsibility (`target_modules`) OR to a shared/external subsystem (e.g. a shared camera controller, world backdrop/scene generator, or hub script) owned by another task.\n"
            "- NEVER instruct multiple downstream mechanic/feature tasks to directly patch the same shared camera, scene, or backdrop script in parallel (which causes workers to overwrite each other and burn revision attempts)!\n"
            "- Instead, when a proof fails partly or wholly because of a defect in a shared/external module outside this item's primary module:\n"
            "  1. If an unapproved task on the board (`in_progress`, `changes_requested`, `ready`, or `blocked`) ALREADY owns that shared module, add its `item_id` to `blocked_by_item_ids`, list the shared module path(s) in `out_of_scope_modules_to_prune`, and write `feedback` telling this item what remains in its own module once the blocker finishes.\n"
            "  2. If the task that originally built that shared module is already `approved` (or no open task owns fixing the shared bug), populate `spawned_prerequisite_tasks` with a focused `WorkItemProposal` targeting that shared module (and set `blocks_items` to include all affected open tasks on the board), and list the shared module path(s) in `out_of_scope_modules_to_prune` so this item is cleanly blocked on the dedicated fix without losing a revision attempt.\n\n"
        )
        prompt = (
            f"You are {who} reviewing work built by the implementation team for the game being designed.\n"
            f"{mandate}"
            f"{filmstrip_guide}"
            f"{cross_task_triage_guide}"
            f"ITEMS AWAITING YOUR REVIEW (proof images and Living Game Bible context are attached above):\n"
            + ("\n".join(self._format_item_for_review(i) for i in awaiting) or "  (none)")
            + f"\n\nITEMS THAT HIT THEIR REVISION CAP OR STALLED (you must decide):\n"
            + ("\n".join(self._format_item_for_review(i) for i in deciding) or "  (none)")
            + "\n\nFor EVERY awaiting item give one `review_decisions` entry: examine the proof carefully against the spec, the proof_spec and the "
            "game's design, then `approve`, or `request_changes` with concrete, actionable feedback (never repeat feedback that was already given without "
            "saying what is still wrong; use `blocked_by_item_ids`, `spawned_prerequisite_tasks`, and `out_of_scope_modules_to_prune` when a defect belongs to a shared/external module). "
            "For EVERY capped/stalled item give one `cap_decisions` entry: `accept_as_is` (only if the result genuinely "
            "serves the game; say what is compromised), `rewrite_spec` (populate a non-empty `replacement_spec` BEFORE setting `action='rewrite_spec'`; max "
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

        return self._structured_with_validation(prompt, ReviewBatch, _validate, temperature=0.3, parts=parts)

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
        batches = self._map_parallel(
            lambda r: self._review_as(r, groups[r][0], groups[r][1], game_bible, dna_summary, candidate_dir), roles
        )
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

    def _handle_failed_items(self) -> None:
        """Re-queues a failed item once (visible via events/logs); on a second failure in a live Council build,
        escalates the item to `needs_decision` so the Council can rewrite its spec or cancel/re-point dependents,
        or raises the authentic error if spec rewrites are exhausted or in direct unit-test failure probes (Rule 5)."""
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
                    continue
                raise StudioAgentError(f"Backlog item '{it.item_id}' failed {it.failure_count} times:\n{it.failure}")
            logger.warning("Re-queueing failed item %s (failure %d):\n%s", it.item_id, it.failure_count, it.failure)
            self.board.requeue_failed(it.item_id)

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
        t0 = time.monotonic()
        while self._is_paused(candidate_dir):
            time.sleep(0.25)
        return max(0.0, time.monotonic() - t0)

    def _execute_steering_pivot_turns(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        candidate_dir: Optional[Path],
        steered_notes: List[str],
    ) -> None:
        """Runs immediate CreativeDirector and GameplaySystemsEngineer turns to pivot the Game Bible and ProductionBoard
        when live Executive Producer steering arrives during _drain_backlog."""
        combined_steering = " | ".join(steered_notes)
        board_was_drained = bool(self.board is not None and self.board.is_drained())
        curated_code_block = ""
        if candidate_dir is not None and Path(candidate_dir).is_dir() and is_real_genai_client(self.client):
            try:
                reach = compute_entrypoint_reachability(Path(candidate_dir))
                curator = ContextCuratorAgent(
                    cartridge_dir=Path(candidate_dir),
                    client=self.client,
                    model_name=self.model_name,
                    board=self.board,
                    game_bible=game_bible,
                    event_callback=self.event_callback,
                )
                pack = curator.curate_context(
                    target_agent_role="council_reviewer:GameplaySystemsEngineer",
                    task_request=(
                        f"Executive Producer Live Steering / Problem Report: {combined_steering}\n"
                        f"Inspect the cartridge entrypoint (`run/main_scene`) and `.gd` scripts to identify the root cause "
                        f"and raise/update the exact ProductionBoard task(s) needed to resolve it.\n"
                        f"{reach.get('summary_text', '')}"
                    ),
                    item_id="executive_producer_steering",
                    item_kind="integration",
                )
                curated_code_block = (
                    f"=== CARTRIDGE CODEBASE & ENTRYPOINT CONTEXT FOR EXECUTIVE PRODUCER STEERING ===\n"
                    f"{reach.get('summary_text', '')}\n\n"
                    f"{pack.prompt_block}\n"
                )
            except Exception as cur_exc:
                logger.warning("ContextCurator steering curation skipped: %s", cur_exc)

        for pivot_role in ("CreativeDirector", "GameplaySystemsEngineer"):
            spec = next((s for s in SPECIALIST_ROSTER if s["role"] == pivot_role), None)
            if spec is None:
                continue
            still_drained = bool(self.board is not None and self.board.is_drained())
            must_add_task = bool(board_was_drained and still_drained and pivot_role == "GameplaySystemsEngineer")
            listen_check = WritersRoomListenCheck(
                deliberation=f"Immediate pivot required to honor Executive Producer steering: {combined_steering}",
                should_interject=True,
                urgency_score=1.0,
                intended_contribution_summary=(
                    f"Address Executive Producer steering ('{combined_steering}'): diagnose any reported runtime/gameplay/UX issue "
                    f"against the cartridge scripts and raise a concrete `backlog_additions` task (`integration`, `mechanic_code`, or `ui`) "
                    f"targeting the responsible `.gd` file(s) so the worker pool immediately fixes and verifies it."
                ),
            )
            proposal = self._execute_specialist_turn(
                specialist=spec,
                listen_check=listen_check,
                game_bible=game_bible,
                dna_summary=dna_summary,
                extra_context_block=curated_code_block,
                require_backlog_addition=must_add_task,
            )
            self._apply_turn_proposal(
                game_bible=game_bible,
                speaker_role=spec["role"],
                speaker_display_name=spec["display_name"],
                proposal=proposal,
                candidate_dir=candidate_dir,
            )

    _SHAPING_COUNCIL_ROLES = (
        "GameplayDesigner",
        "LevelAndProgressionDesigner",
        "GameplaySystemsEngineer",
        "UXExperienceDesigner",
        "VisualArtDirector",
        "AudioAtmosphereDesigner",
    )

    def _items_needing_domain_direction(self, role_name: str) -> List[Any]:
        """Returns unstarted board items that were raised by a different specialist (or split/added in Stage 1)
        and have not yet received `[<role_name> Direction]` input."""
        if self.board is None:
            return []
        role_prefix = f"[{role_name} Direction]:"
        needed: List[Any] = []
        for item in self.board.items():
            if item.state not in ("ready", "blocked"):
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
            targets = self._items_needing_domain_direction(role_name)
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
                f"PLAYER DNA & STUDIO CONTEXT:\n{dna_summary}\n\n"
                f"{bible_snapshot}\n\n"
                f"{board_snap}\n\n"
                f"You are the {display_name} ({role_name}) in Stage 2 of the Council Task-Shaping Session (Full-Council Cross-Domain Direction).\n"
                f"Your specialty focus: {spec['focus']}\n\n"
                f"All tasks and task decompositions are now on the Production Board. Before workers start building, "
                f"you must review EVERY unstarted task below from your domain's perspective and provide one `backlog_refinements` entry per task:\n"
                f"TASKS REQUIRING YOUR DOMAIN REVIEW ({len(targets)} tasks):\n"
                f"{target_lines}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. For EVERY task listed in `TASKS REQUIRING YOUR DOMAIN REVIEW` above, include one `WorkItemRefinement` entry in `backlog_refinements` with matching `item_id`:\n"
                f"   - Set `action='enrich'` and populate `spec_addition_or_rewrite` (plus optional `proof_spec_update`, `add_depends_on`, or `add_target_modules`—using encapsulated per-feature `.gd` paths like `scenes/mechanics/<feature>.gd`, `scenes/entities/<entity>.gd`, or `scenes/ui/<widget>.gd` when assigning or refining code targets so parallel SWE workers can lock disjoint files) whenever your domain has direction, integration contracts, UX affordances, non-overlapping HUD layout rules, world/progression hooks, visual rig/silhouette specs, or reactive audio cues for that task.\n"
                f"   - Set `action='no_input'` ONLY if, after deliberating, your domain genuinely has zero requirements or touchpoints with that specific task.\n"
                f"2. Leave `backlog_additions=[]` in Stage 2 (all companion tasks were already raised in Stage 1).\n"
                f"Return `TaskShapingBatch` with `deliberation` FIRST."
            )

            def _validate_stage2(batch: TaskShapingBatch) -> List[str]:
                issues: List[str] = []
                seen_ids = {r.item_id for r in (batch.backlog_refinements or [])}
                missing = sorted(required_ids - seen_ids)
                if missing:
                    issues.append(
                        f"Missing `backlog_refinements` decision for required task(s): {missing}. "
                        f"Provide one entry per listed task with `action='enrich'` (and `spec_addition_or_rewrite`) or `action='no_input'`."
                    )
                for r in batch.backlog_refinements or []:
                    if r.action == "split":
                        r.action = "enrich" if (r.spec_addition_or_rewrite.strip() or r.deliberation.strip()) else "no_input"
                issues.extend(self.board.validate_refinements(batch.backlog_refinements or []))
                return issues

            return role_name, self._structured_with_validation(prompt, TaskShapingBatch, _validate_stage2)

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
        unstarted = [i for i in self.board.items() if i.state in ("ready", "blocked")]
        if not unstarted:
            return 0
        bible_snapshot = self._format_bible_snapshot(game_bible)
        active_slice_obj = game_bible.get_active_slice()
        default_slice_id = active_slice_obj.slice_id if active_slice_obj is not None else ""
        total_changes = 0

        # Stage 1: Decomposition ('split') & Missing Companion Task Discovery ('backlog_additions')
        for role_name in self._SHAPING_COUNCIL_ROLES:
            unstarted_now = [i for i in self.board.items() if i.state in ("ready", "blocked")]
            if not unstarted_now:
                break
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
            prompt = (
                f"PLAYER DNA & STUDIO CONTEXT:\n{dna_summary}\n\n"
                f"{bible_snapshot}\n\n"
                f"{self.board.snapshot_text()}\n\n"
                f"You are the {display_name} ({role_name}) in Stage 1 of the Council Task-Shaping & Decomposition Session.\n"
                f"Your specialty focus: {spec['focus']}\n\n"
                f"Before the full Council gives cross-domain direction in Stage 2, review the unstarted (`ready` or `blocked`) tasks on the Production Board:\n"
                f"1. DECOMPOSE MONOLITHIC TASKS (`action='split'` in `backlog_refinements`): Should any task be split into smaller, focused tasks (for example, separating core simulation mechanics from interactive player placement/building tools, separating world/level progression from HUD telemetry, or splitting a multi-entity visual_asset bundle into 1-entity-per-item visual_asset tasks)? When you split a task, downstream dependents on the board automatically re-point to your split tasks.\n"
                f"2. RAISE MISSING COMPANION TASKS (`backlog_additions`): Is a prerequisite asset, UI affordance, level region, or sub-mechanic task missing altogether? Propose it in `backlog_additions` (and use `blocks_items` or `add_depends_on` to link it).\n"
                f"3. CONTRIBUTE INITIAL DOMAIN DIRECTION (`action='enrich'` in `backlog_refinements`): You may also enrich existing tasks with `spec_addition_or_rewrite` and `add_depends_on`.\n"
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

            batch = self._structured_with_validation(prompt, TaskShapingBatch, _validate_shaping)
            if batch.backlog_additions:
                for prop in batch.backlog_additions:
                    if prop.kind == "level_map" and "level_layout.json" not in prop.target_modules:
                        prop.target_modules.append("level_layout.json")
                    elif prop.kind in ("visual_asset", "animation", "audio_asset"):
                        prop.target_modules = [m for m in prop.target_modules if not str(m).endswith(".gd")]
                added = self.board.add_items(
                    batch.backlog_additions,
                    raised_by_role=role_name,
                    slice_id=default_slice_id,
                )
                total_changes += len(added)
            if batch.backlog_refinements:
                shaped = self.board.apply_refinements(
                    batch.backlog_refinements,
                    by_role=role_name,
                )
                total_changes += len(shaped)

        # Stage 2: Parallel Full-Council Cross-Domain Direction over the complete board (including all newly added/split tasks)
        total_changes += self._run_full_council_direction_pass(
            game_bible=game_bible,
            dna_summary=dna_summary,
            bible_snapshot=bible_snapshot,
        )

        if candidate_dir is not None and total_changes > 0:
            self._persist_bible(game_bible, candidate_dir=candidate_dir)
        return total_changes

    def _run_slice_assembly_check(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        candidate_dir: Optional[Path],
        pass_index: int = 0,
    ) -> int:
        """Convenes the Council when all current Production Board tasks for the active slice are approved,
        presenting the factual entrypoint reachability graph (`compute_entrypoint_reachability`) and a
        lightweight 3-frame boot filmstrip (`ticks=[15, 60, 120]`) of `project.godot`'s `run/main_scene`
        so the Council can decide whether all built systems are assembled in the playable entrypoint or
        whether to add integration/assembly task(s) to the Production Board."""
        if (
            self.board is None
            or candidate_dir is None
            or not (Path(candidate_dir) / "project.godot").is_file()
        ):
            return 0
        # Only run automatic slice assembly checks on live Gemini builds unless explicitly enabled in tests
        if not is_real_genai_client(self.client) and not getattr(self, "enable_slice_assembly_check", False):
            return 0

        c_dir = Path(candidate_dir)
        self._emit_activity(
            agent_role="CreativeDirector",
            status_text=f"Council is inspecting booted main scene & entrypoint module reachability (Assembly Check #{pass_index + 1})...",
            active_agents=["CreativeDirector", "GameplaySystemsEngineer", "GameplayDesigner", "UXExperienceDesigner"],
            phase="slice_assembly_check",
        )

        from engine.studio.context_curator import (
            ContextCuratorAgent,
            compute_entrypoint_reachability,
        )
        from engine.studio.proofs import format_filmstrip_telemetry_progression

        reach = compute_entrypoint_reachability(c_dir)
        parts: List[Any] = []
        boot_telemetry_summary = ""
        proof_out = c_dir / "artifacts" / "proofs" / f"filmstrip_slice_assembly_check_pass{pass_index}.png"
        if is_real_genai_client(self.client) or getattr(self, "assembly_filmstrip_fn", None) is not None:
            try:
                filmstrip_fn = getattr(self, "assembly_filmstrip_fn", None)
                if filmstrip_fn is not None:
                    filmstrip_fn(c_dir, [15, 60, 120], proof_out)
                else:
                    from engine.studio.proofs import capture_filmstrip
                    capture_filmstrip(
                        cartridge_dir=c_dir,
                        ticks=[15, 60, 120],
                        out_path=proof_out,
                        target_scene=reach.get("main_scene") or "res://main.tscn",
                    )
                if proof_out.is_file():
                    part = self._load_image_part(proof_out)
                    if part is not None:
                        parts.append(part)
                    boot_telemetry_summary = format_filmstrip_telemetry_progression(proof_out)
            except Exception as cap_exc:
                logger.warning("Slice assembly boot filmstrip capture skipped: %s", cap_exc)

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
                        f"Evaluate whether all built modules, UI instruments, and gameplay systems for the active slice "
                        f"are mounted and connected inside the playable entrypoint ({reach.get('main_scene')}).\n"
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
                    f"  - [{it.item_id}] ({it.kind}) '{it.title}' | target_modules={it.target_modules} | notes={last_notes[:200]}"
                )

        bible_snapshot = self._format_bible_snapshot(game_bible)
        prompt = (
            f"PLAYER DNA & STUDIO CONTEXT:\n{dna_summary}\n\n"
            f"{bible_snapshot}\n\n"
            f"{curated_assembly_block}"
            f"=== PLAYABLE ENTRYPOINT REACHABILITY GRAPH ===\n{reach.get('summary_text', '')}\n\n"
            f"=== MAIN SCENE OPENING BOOT TELEMETRY (ticks 15, 60, 120) ===\n{boot_telemetry_summary or '(not captured)'}\n\n"
            f"=== APPROVED PRODUCTION BOARD ITEMS ({len(approved_summaries)} items) ===\n"
            + ("\n".join(approved_summaries) or "  (none)")
            + "\n\nYou are the Council (CreativeDirector, GameplaySystemsEngineer, GameplayDesigner, and UXExperienceDesigner) "
            "performing the Slice Readiness & Whole-Game Assembly Check now that all currently queued Production Board items are approved.\n"
            "INSTRUCTIONS:\n"
            "1. Compare the Living Game Bible and the Approved Production Board items above against the `PLAYABLE ENTRYPOINT REACHABILITY GRAPH` "
            "and the attached opening boot filmstrip/telemetry of the playable entrypoint (`run/main_scene`).\n"
            "2. Because individual features were developed in modular `.gd` files and verified in isolated test scenes (`res://tests/test_*.tscn`), "
            "check whether any built gameplay system, progression loop, or HUD/UI widget that belongs in the playable game is still unmounted or unwired "
            "from the playable entrypoint scene tree (note: helper scripts or intentionally superseded prototypes do NOT need to be mounted—only systems that serve the actual game design).\n"
            "3. Also inspect the attached opening boot filmstrip of `run/main_scene` across ALL frames (`tick 15`, `tick 60`, `tick 120`) for whole-game visual/HUD cohesion AND opening temporal stability: "
            "verify that the player avatar and core interactive world entities remain stable and framed on screen across `t=15..120` (never plummeting/drifting out of the playfield under unintended gravity/forces or dragging the camera away from the player into empty background space), "
            "that HUD widgets coexist cleanly without overlapping or occluding the playfield, and that there are zero leftover test-staging hacks in `main.gd`.\n"
            "4. If everything needed for the active slice is already assembled and coherent in `run/main_scene`, set `backlog_additions=[]` and `is_slice_assembled=True`.\n"
            "5. If any built system or UI instrument needs to be mounted/wired into the playable entrypoint (or if `run/main_scene` has a whole-game boot/HUD/wiring issue), "
            "propose focused `integration` (or `mechanic_code`/`ui`) task(s) in `backlog_additions` BEFORE setting `is_slice_assembled=False`:\n"
            "   - IMPORTANT FOR QA SCOPE: Deep per-mechanic edge cases were ALREADY verified in isolated `res://tests/test_*.tscn` scenes! "
            "When writing `proof_spec` for an assembly/integration task on `run/main_scene`, keep the QA proof lightweight (verifying that `res://main.tscn` boots cleanly with all slice nodes/HUD widgets mounted in the scene tree and responsive to opening player controls over ~120-180 ticks—NEVER demand a multi-minute full-game playthrough in a single filmstrip).\n"
            f"Buildable kinds right now: {self._supported_kinds() or 'all'}.\n"
            "Return `SliceAssemblyDecision` with `deliberation` FIRST and `is_slice_assembled` strictly last."
        )

        def _validate_assembly(decision: SliceAssemblyDecision) -> List[str]:
            issues: List[str] = []
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
        )
        self._emit_event(
            "slice_assembly_check_completed",
            {
                "deliberation": decision.deliberation,
                "pass_index": pass_index,
                "is_slice_assembled": decision.is_slice_assembled,
                "unmounted_or_disconnected_systems": decision.unmounted_or_disconnected_systems,
                "spawned_item_ids": [p.item_id for p in (decision.backlog_additions or [])],
            },
        )
        added: List[Any] = []
        if not decision.is_slice_assembled and decision.backlog_additions:
            active_slice_obj = game_bible.get_active_slice()
            default_slice_id = active_slice_obj.slice_id if active_slice_obj is not None else ""
            added = self.board.add_items(
                decision.backlog_additions,
                raised_by_role="GameplaySystemsEngineer",
                slice_id=default_slice_id,
            )

        turn_idx = len(game_bible.writers_room_transcript)
        if added:
            spawned_summary = ", ".join(f"`{it.item_id}` ({it.title})" for it in added)
            unmounted_note = (
                f" Unmounted/disconnected modules detected: {', '.join(decision.unmounted_or_disconnected_systems)}."
                if decision.unmounted_or_disconnected_systems
                else ""
            )
            spoken = (
                f"[Slice Assembly Check #{pass_index + 1}] All individual task proofs passed, so we booted `run/main_scene` and audited "
                f"entrypoint reachability.{unmounted_note} Spawning whole-game assembly task(s): {spawned_summary}."
            )
            updates_list = [f"backlog:+{it.item_id}" for it in added]
        else:
            spoken = (
                f"[Slice Assembly Check #{pass_index + 1}] Verified `run/main_scene` opening boot filmstrip and entrypoint reachability—"
                f"all slice systems and HUD instruments are mounted and wired in the playable game."
            )
            updates_list = ["slice_assembly:verified"]

        assembly_utt = WritersRoomUtterance(
            deliberation=decision.deliberation,
            utterance_id=f"utt_{turn_idx:02d}_GameplaySystemsEngineer_assembly_{pass_index}",
            turn_index=turn_idx,
            speaker_role="GameplaySystemsEngineer",
            speaker_display_name="Gameplay Systems Engineer",
            spoken_dialogue=spoken,
            addressed_to=["CreativeDirector", "GameplayDesigner", "UXExperienceDesigner"],
            bible_updates_summary=updates_list,
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
            },
        )
        self._persist_bible(game_bible, candidate_dir=candidate_dir)
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
        already_executing = any(
            i.state in ("in_progress", "waiting_for_lock", "awaiting_review", "changes_requested", "approved") or i.revisions
            for i in self.board.items()
        )
        if not already_executing:
            self._run_council_task_shaping_round(game_bible, dna_summary, candidate_dir=candidate_dir)
        if self.scheduler is not None:
            self.scheduler.start()
        deadline = time.monotonic() + self.drain_timeout_s

        def _board_progress_sig() -> Tuple[int, int, int]:
            all_items = self.board.items()
            return (
                len(all_items),
                sum(len(i.revisions) for i in all_items),
                sum(1 for i in all_items if i.state in ("approved", "cancelled")),
            )

        last_progress_sig = _board_progress_sig()
        assembly_passes = 0
        max_assembly_passes = int(getattr(self, "max_assembly_passes", 2))
        while True:
            paused_dur = self._wait_if_paused(candidate_dir)
            if paused_dur > 0:
                deadline += paused_dur
            steered = self._consume_pending_steering(game_bible, candidate_dir=candidate_dir)
            if steered:
                self._execute_steering_pivot_turns(game_bible, dna_summary, candidate_dir, steered)
                if not self.board.is_drained():
                    assembly_passes = 0
                    if self.scheduler is not None:
                        self.scheduler.start()
            self._handle_failed_items()
            applied = self._run_review_round(game_bible, dna_summary, candidate_dir)
            if self.board.is_drained():
                if assembly_passes < max_assembly_passes:
                    added_assembly = self._run_slice_assembly_check(
                        game_bible=game_bible,
                        dna_summary=dna_summary,
                        candidate_dir=candidate_dir,
                        pass_index=assembly_passes,
                    )
                    assembly_passes += 1
                    if added_assembly > 0:
                        if self.scheduler is not None:
                            self.scheduler.start()
                        last_progress_sig = _board_progress_sig()
                        deadline = max(deadline, time.monotonic() + self.drain_timeout_s)
                        continue
                return
            cur_progress_sig = _board_progress_sig()
            if applied > 0 or steered or cur_progress_sig != last_progress_sig:
                last_progress_sig = cur_progress_sig
                deadline = max(deadline, time.monotonic() + self.drain_timeout_s)
            active = [
                i for i in self.board.items()
                if i.state in ("ready", "in_progress", "waiting_for_lock", "changes_requested", "awaiting_review", "needs_decision")
            ]
            if not active and self.scheduler is not None and self.scheduler.running_count() == 0:
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
            f"{bible_snapshot}\n"
            f"Critically evaluate whether the Living Game Bible is ready to lock and enter production.\n"
            f"Do NOT approve a shallow 'Hello World' prototype. To be ready (`is_ready_to_lock_bible = true`), the Bible MUST have:\n"
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
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))

    def _structured_with_validation(
        self,
        prompt: str,
        schema_cls: Any,
        validate: Callable[[Any], List[str]],
        attempts: int = 3,
        temperature: Optional[float] = None,
        parts: Optional[List[Any]] = None,
    ) -> Any:
        """Calls the LLM until `validate` returns no issues, feeding issues back; raises after `attempts` (Rule 5)."""
        issues: List[str] = []
        for attempt in range(1, attempts + 1):
            feedback = (
                "\n\nYOUR PREVIOUS ANSWER FAILED VALIDATION; FIX EVERY ISSUE:\n- " + "\n- ".join(issues) if issues else ""
            )
            result = self._call_structured_llm(prompt + feedback, schema_cls, temperature=temperature, parts=parts)
            issues = validate(result)
            if not issues:
                return result
            logger.info("%s attempt %d/%d failed validation: %s", schema_cls.__name__, attempt, attempts, issues)
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
        if existing_bible is not None:
            game_bible = existing_bible
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
        if self.board is not None and self.scheduler is not None and not is_real_genai_client(self.client):
            if turns_executed >= min(3, self.max_turns) or len(self.board.items()) >= 3:
                self.scheduler.start()
        while not already_in_production and turns_executed < self.max_turns:
            self._wait_if_paused(candidate_dir)
            steered = self._consume_pending_steering(game_bible, candidate_dir=candidate_dir)
            if steered:
                last_speaker_role = "ExecutiveProducer"
                last_addressed_to = ["CreativeDirector", "GameplayDesigner", "GameplaySystemsEngineer", "UXExperienceDesigner"]
                for r in last_addressed_to:
                    if r not in pending_invited_roles:
                        pending_invited_roles.append(r)
                if self.max_turns - turns_executed < 3:
                    self.max_turns = turns_executed + 3
            effective_addressed_to = list(dict.fromkeys(list(last_addressed_to) + pending_invited_roles))
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
        game_bible.status = "in_production"
        self._persist_bible(game_bible, candidate_dir=candidate_dir)
        # ...and every requirement the council raised must be built and approved (or deliberately cancelled).
        self._drain_backlog(game_bible, dna_summary, candidate_dir)

        # Synthesize final GenesisWorldSpec on the locked GameBibleSpec
        if game_bible.world_spec is None or not already_in_production:
            game_bible.world_spec = self._synthesize_final_world_spec(game_bible, dna, target_episode)
        if game_bible.world_spec and game_bible.world_spec.core_genre:
            game_bible.core_genre = game_bible.world_spec.core_genre
        game_bible.status = "locked"
        self._persist_bible(game_bible, candidate_dir=candidate_dir)
        return game_bible

    def run_slice_expansion(
        self,
        game_bible: GameBibleSpec,
        dna: Optional[SteamPlayerDNA] = None,
        candidate_dir: Optional[Path] = None,
        slice_directive: Optional[str] = None,
    ) -> GameBibleSpec:
        """Advances `game_bible` to the next iterative vertical slice (marking the prior slice completed,
        strategizing or activating the next slice that expands it, seeding only the new slice's high-level
        tasks on the ProductionBoard, and draining the board for that slice)."""
        if self.client is None:
            raise StudioAgentError(
                "CreativeWritersRoom.run_slice_expansion requires an active google.genai Client (Rule 5)"
            )
        dna_summary = self._format_dna_summary(dna)
        for idx_s, s in enumerate(game_bible.slice_roadmap):
            if idx_s <= game_bible.active_slice_index and s.status == "active":
                s.status = "completed"
        next_idx = (
            (max(s.slice_index for s in game_bible.slice_roadmap if s.status == "completed") + 1)
            if any(s.status == "completed" for s in game_bible.slice_roadmap)
            else (len(game_bible.slice_roadmap) if game_bible.slice_roadmap else 0)
        )
        game_bible.active_slice_index = next_idx
        for s in game_bible.slice_roadmap:
            if s.slice_index == next_idx:
                s.status = "active"
                break

        directive_note = (
            f" User/Director expansion directive: {slice_directive.strip()}"
            if slice_directive and slice_directive.strip()
            else ""
        )
        for role_name in ("LevelAndProgressionDesigner", "GameplayDesigner", "GameplaySystemsEngineer"):
            spec = next((sp for sp in SPECIALIST_ROSTER if sp["role"] == role_name), None)
            if spec is None:
                continue
            check = WritersRoomListenCheck(
                deliberation=(
                    f"Advancing to iterative Slice #{next_idx}: strategizing how this slice expands the prior slice "
                    f"and raising only the tasks needed for Slice #{next_idx}.{directive_note}"
                ),
                should_interject=True,
                urgency_score=1.0,
                intended_contribution_summary=(
                    f"Strategize Slice #{next_idx} in `added_or_updated_slices` (with `slice_index={next_idx}`, `status='active'`, "
                    f"and `expansion_from_prior_slice` explaining how it expands Slice #{max(0, next_idx - 1)}) and raise `backlog_additions` "
                    f"ONLY for Slice #{next_idx}.{directive_note}"
                ),
            )
            proposal = self._execute_specialist_turn(
                specialist=spec,
                listen_check=check,
                game_bible=game_bible,
                dna_summary=dna_summary,
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

