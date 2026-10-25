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
    StudioAgentError,
    VisualMockupArtifact,
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
        creative_temperature: float = 1.25,
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
        if isinstance(raw_data, schema_cls):
            return raw_data
        return schema_cls(**raw_data)

    def _format_dna_summary(self, dna: Optional[SteamPlayerDNA], hypothesis_report: Optional[Any] = None) -> str:
        lines: List[str] = []
        if dna is not None:
            top_games = [
                g for g in (getattr(dna, "top_games", []) or [])
                if isinstance(g, dict) and not str(g.get("name", "")).startswith("App ")
            ]
            # Dynamically sample a varied spotlight subset of the player's top games so each session
            # explores a fresh creative intersection of their library rather than repeating the same pitch.
            if len(top_games) > 4:
                spotlight_games = random.sample(top_games[:12], k=min(4, len(top_games[:12])))
            else:
                spotlight_games = list(top_games)
            spotlight_str = ", ".join(
                f"{g.get('name', 'Unknown')} (genres: {g.get('genres', [])})"
                for g in spotlight_games
            )
            top_genres = list(getattr(dna, "top_genres", []) or [])
            if len(top_genres) > 1:
                shuffled_genres = list(top_genres)
                random.shuffle(shuffled_genres)
            else:
                shuffled_genres = top_genres
            lines.extend([
                f"- Genre Affinities to Explore: {shuffled_genres}",
                f"- Pacing Preference: {getattr(dna, 'pacing_preference', 'balanced')}",
            ])
            palette_pref = getattr(dna, "preferred_palette_mood", None)
            if palette_pref:
                lines.append(f"- Preferred Palette Mood: {palette_pref}")
            if spotlight_str:
                lines.append(
                    f"- Taste signals from the player's library (what they value — NOT templates to imitate, mash up or pastiche): {spotlight_str}"
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
            "core_loop=" + (f"[verbs={loop.player_verbs}; m2m={loop.moment_to_moment_loop[:160]}; session={loop.session_loop[:160]}]" if loop else "(missing)")
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
            f"Objective: {active_slice.slice_objective} | Acclimation & Mechanics Focus: {active_slice.player_acclimation_and_mechanics_focus} | "
            f"Expansion: {active_slice.expansion_from_prior_slice} | Scope Boundaries: {active_slice.scope_boundaries}"
            if active_slice is not None
            else "(not yet strategized — Slice 0 MUST deliver a complete, cohesive playable vertical slice realizing the world's camera/scale, core loop, new-player onboarding, and audio)"
        )
        exec_directives = [
            u.spoken_dialogue.strip()
            for u in game_bible.writers_room_transcript
            if u.speaker_role in ("ExecutiveProducer", "UserSteering") and u.spoken_dialogue and u.spoken_dialogue.strip()
        ]
        exec_banner = (
            "🚨 MANDATORY EXECUTIVE PRODUCER CREATIVE STEERING (HIGHEST PRIORITY — OVERRIDES CONFLICTING PREMISES):\n"
            + "\n".join(f'  - "{d}"' for d in exec_directives)
            + "\n  Every specialist MUST adapt the game's core loop, pacing, fail/recovery rules, controls, and Production Board items to honor this Executive Producer direction!\n\n"
            if exec_directives
            else ""
        )
        return (
            f"{exec_banner}"
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
        return self._call_structured_llm(prompt, WritersRoomListenCheck)

    def _execute_specialist_turn(
        self,
        specialist: Dict[str, Any],
        listen_check: WritersRoomListenCheck,
        game_bible: GameBibleSpec,
        dna_summary: str,
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
                "Domain Craft & Bible Fields (UXExperienceDesigner): Put yourself in the shoes of a brand-new player booting this game for the first time "
                "with zero external documentation. Strategize how the player immediately understands their role in the world, what objects are interactive "
                "and how they connect, what their active objective is, and how the world responds to their actions—without cluttering the screen with "
                "generic debug labels. Shape `updated_first_60_seconds_hook`, `updated_diegetic_ui_philosophy` (identifying the exact HUD/CanvasLayer "
                "isolation, visual affordances, and feedback cues that fit THIS game), `updated_onboarding_plan` (how the player is introduced to the "
                "world, role, and stakes, plus ordered `OnboardingStep`s), and `added_or_updated_storyboard_beats`. When a Production Board is attached, "
                "raise concrete `ui` / onboarding `backlog_additions` so your new-player experience strategy is actually built in code."
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
                "Domain Craft & Bible Fields (GameplayDesigner): Pressure-test the mechanics for genuine fun and agency, and shape "
                "`gameplay_pillars_additions`, `updated_core_loop` (player verbs plus moment-to-moment, minute-to-minute, and session loops, feedback, reward cadence, "
                "failure and recovery), `updated_engagement_plan` (entertainment pillars, variety injectors, pacing curve, mastery), `added_or_updated_slices`, and "
                "`concept_evolution_milestone`. You can call on your Sub-Agents `LevelAndProgressionDesigner` and `GameplaySystemsEngineer` via `addressed_to`."
            )
        elif role == "GameplaySystemsEngineer":
            domain_instructions = (
                "Domain Craft & Bible Fields (GameplaySystemsEngineer): Translate the game's mechanics, camera/spatial paradigm, and visual direction into "
                "concrete Godot 4 engineering strategies in `added_or_updated_engineering_specs` (`MechanicEngineeringSpec` items covering node/collider architecture, "
                "hybrid procedural + generated texture/sprite/rig rendering, active `Camera2D` tracking across world bounds with HUD isolated on a `CanvasLayer`, "
                "deterministic state machines that start unsolved at t=0, and falsifiable headless test invariants)."
            )
        elif role == "LevelAndProgressionDesigner":
            domain_instructions = (
                "Domain Craft & Bible Fields (LevelAndProgressionDesigner): Strategize the game's world and progression in `updated_level_generation_strategy` "
                "(bespoke `generation_approach`, `world_structure`, `runtime_construction_plan`, `level_archetypes`, `generation_rules`, and difficulty curve), "
                "`updated_goal_hierarchy` (goals across multiple horizons, progression structure, and end state), and `added_or_updated_slices` "
                "(where Slice 0 delivers a complete, cohesive playable vertical slice across the world's intended scale, camera movement, and progression loop, "
                "rather than a cramped single-object test room)."
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
            f"Speak naturally, creatively, and passionately to your fellow designers in `spoken_dialogue` (never recite raw telemetry floats), "
            f"call out specific colleagues or Sub-Agents in `addressed_to`, and contribute your concrete updates to the Living Game Bible. Provide `deliberation` FIRST."
        )
        if self.board is None:
            return self._call_structured_llm(prompt, WritersRoomTurnProposal)
        board_block = (
            f"{self.board.snapshot_text()}\n"
            f"BUILD BACKLOG (ACTIVE SLICE — EMERGENT TASK & ASSET DISCOVERY): The implementation side builds in parallel while you design. "
            f"Raise `backlog_additions` needed for the ACTIVE SLICE right now so Slice 0 ships as a complete, cohesive playable vertical slice "
            f"(world/level traversal with proper camera tracking, core gameplay loop, empathetic new-player onboarding/HUD on a CanvasLayer, visual assets/surfaces, and audio).\n"
            f"HYBRID PROCEDURAL + GENERATED ASSETS: Do NOT settle for flat untextured `Polygon2D` slabs or bare programmer-art `Line2D` strokes! "
            f"Raise `visual_asset` tasks (`tile_surface` seamless textures for procedural `Polygon2D`/`Line2D` terrain/conduits, environmental decals/landmarks, "
            f"and `sprite_sheet`/`articulated_parts` rigs for characters and interactive structures) alongside procedural shaders and particles.\n"
            f"DEPENDENCY & ASSET BLOCKING RULES (`depends_on` and `blocks_items`):\n"
            f"- Characters, avatars, creatures, and hero props in `mechanic_code` / `integration` tasks MUST bind actual fabricated sprites or articulated rigs—never primitive `draw_circle`/`draw_rect` stickmen.\n"
            f"- When you raise a `mechanic_code` or `integration` task that requires a character rig, surface texture, or discrete sprite, either include that `visual_asset` item in `depends_on` (proposing the `visual_asset` item in the same turn if it is not on the board yet), OR let the Task Lead spawn it on Turn 0 with `wait_for_spawned_tasks=True`.\n"
            f"- Conversely, when you raise a `visual_asset`, `animation`, or foundational system task that an ALREADY-PROPOSED task on the Production Board needs (for example, proposing the player's `visual_asset` character rig after a player controller `mechanic_code` task was already raised on an earlier turn), list those downstream task IDs in `blocks_items` (e.g. `blocks_items=['mechanic_...']`)! Setting `blocks_items` automatically wires your new task into their `depends_on` list and blocks them until your asset/system is fabricated and approved.\n"
            f"- For each item you raise, provide a unique `item_id`, `slice_id` matching the active slice, a precise `spec`, `depends_on`, `blocks_items`, "
            f"`target_modules` (leave `target_modules=[]` for standalone `visual_asset`, `animation`, and `audio_asset` items), and a `proof_spec`. "
            f"Do not duplicate existing items. The finished work will be shown to the Council panel for approval.\n"
            f"Buildable kinds right now: {self._supported_kinds() or 'all'}.\n"
        )

        def _validate(p: WritersRoomTurnProposal) -> List[str]:
            if not p.backlog_additions:
                return []
            return self.board.validate_proposals(p.backlog_additions) + self._unsupported_kind_issues(p.backlog_additions)

        return self._structured_with_validation(prompt + "\n" + board_block, WritersRoomTurnProposal, _validate)

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
            mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            is_filmstrip = "filmstrip" in path.name.lower()
            filmstrip_note = (
                " [QA CONTACT SHEET: Each cell shows the 16:9 game viewport inside a thin slate border; the top strip ('[QA] tick N in:[...]') "
                "and bottom strip ('[QA] pos=(...) ...') outside each 16:9 frame are external QA test-harness annotations provided FOR THE COUNCIL "
                "to verify simulation ticks, inputs, and live engine state—they are NOT inside the game viewport and must NEVER be flagged as in-game debug text]"
                if is_filmstrip
                else ""
            )
            parts.append(
                f"PROOF IMAGE for item '{item.item_id}' (revision {latest.revision_index + 1}, file {path.name}){filmstrip_note}:"
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
        return (
            f"  ITEM [{item.item_id}] kind={item.kind} title={item.title!r} state={item.state}\n"
            f"    spec: {item.spec}\n    proof_spec: {item.proof_spec}\n    depends_on: {item.depends_on}\n"
            f"    revision_cap: {item.max_revisions} (spec rewrites so far: {item.spec_resets}/{MAX_SPEC_RESETS})\n"
            f"    decision_reason: {item.decision_reason or '-'}{failure_line}\n    history:\n{history}"
        )

    @staticmethod
    def _role_review_mandate(role: str) -> str:
        if role == "UXExperienceDesigner":
            return (
                "ROLE MANDATE (UXExperienceDesigner): You have VETO power over UI/UX and new-player legibility. Put yourself in the shoes of a brand-new "
                "player seeing this screen for the first time: can they tell what their role/goal is, which structures are interactive, and what is "
                "world vs. HUD? Evaluate the proof against the UI, HUD, and onboarding approach designed in the Game Bible. Immediately `request_changes` "
                "if HUD widgets are placed in world-space where the player avatar can walk behind or collide with them (HUD must live on a `CanvasLayer`), "
                "if a proof shows internal QA test-runner strings (e.g. 'HEADLESS MECHANICS', 'ALL PASS') drawn INSIDE the 16:9 game viewport, or if UI "
                "panels occlude the playable center. "
                "IMPORTANT: The thin header bar ('[QA] tick N in:[...]') and footer bar ('[QA] pos=...') outside each 16:9 frame in `filmstrip_*.png` "
                "are external QA Test Harness contact-sheet margins added FOR THE COUNCIL—do NOT flag those external frame caption bars as in-game HUD defects.\n"
            )
        if role == "VisualArtDirector":
            return (
                "ROLE MANDATE (VisualArtDirector): You have VETO power over visual fidelity and spatial staging. Compare the proof directly against the 16:9 Concept Art "
                "Mockup, Game Bible Visual Style Guide, and the cartridge's fabricated assets. We demand hybrid procedural + generated visual craft: procedural "
                "geometry/contours (`Polygon2D`, `Line2D`) MUST be combined with fabricated surface textures (`tile_surface` via `StudioRigHelper.create_tiled_polygon` / "
                "`TEXTURE_REPEAT_ENABLED`, `Line2D` strip textures with `LINE_TEXTURE_TILE`), custom `ShaderMaterial`s, environmental decals, and `GPUParticles2D`—NEVER "
                "solid flat-colored untextured `Polygon2D` slabs, blank white/monochrome backgrounds, or bare 8-point `Line2D` programmer-art zigzags! "
                "Characters, avatars, creatures, and interactive world structures MUST bind actual fabricated sprites or articulated rigs from `res://assets/sprites/*.png` "
                "and be coherently layered into the environment (not crudely pasted on top of each other). Immediately `request_changes` if a `level_map`, `mechanic_code`, "
                "or `integration` proof renders untextured flat polygons, bare programmer-art lines, or primitive `draw_circle`/`draw_rect` stickmen/boxes. "
                "Do NOT flag the external QA contact-sheet header/footer strips ('[QA] tick ...', '[QA] pos=...') outside each 16:9 frame as in-game text.\n"
            )
        if role in ("GameplayDesigner", "GameplaySystemsEngineer"):
            return (
                f"ROLE MANDATE ({role}): You have VETO power over gameplay authenticity. Immediately `request_changes` if the proof filmstrip shows "
                "static/non-interactive frames where an action should occur, or if state transitions are not causally driven by player inputs. "
                "Use the external QA contact-sheet header/footer strips ('[QA] tick ... in:[...]', '[QA] pos=(...) ...') above and below each 16:9 frame "
                "as ground-truth engine telemetry to verify coordinates and state changes—never flag those external QA margin strips as in-game debug text.\n"
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
        shared_review_context = (
            f"{on_disk_sprites_block}"
            f"{curated_review_block}"
            f"{self._format_bible_snapshot(game_bible)}\n"
        )
        parts.append(shared_review_context)
        prompt = (
            f"You are {who} reviewing work built by the implementation team for the game being designed.\n"
            f"{mandate}"
            f"{filmstrip_guide}"
            f"ITEMS AWAITING YOUR REVIEW (proof images and Living Game Bible context are attached above):\n"
            + ("\n".join(self._format_item_for_review(i) for i in awaiting) or "  (none)")
            + f"\n\nITEMS THAT HIT THEIR REVISION CAP OR STALLED (you must decide):\n"
            + ("\n".join(self._format_item_for_review(i) for i in deciding) or "  (none)")
            + "\n\nFor EVERY awaiting item give one `review_decisions` entry: examine the proof carefully against the spec, the proof_spec and the "
            "game's design, then `approve`, or `request_changes` with concrete, actionable feedback (never repeat feedback that was already given without "
            "saying what is still wrong). For EVERY capped/stalled item give one `cap_decisions` entry: `accept_as_is` (only if the result genuinely "
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

        return self._structured_with_validation(prompt, ReviewBatch, _validate, parts=parts)

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
                for r, d in dissenters:
                    fb_clean = d.feedback.strip()
                    if fb_clean and fb_clean not in seen_fb:
                        seen_fb.add(fb_clean)
                        distinct_fb.append((r, fb_clean))
                if len(distinct_fb) == 1:
                    merged_fb = distinct_fb[0][1]
                else:
                    merged_fb = " | ".join(f"[{r}]: {fb}" for r, fb in distinct_fb)
                consolidated = ReviewDecision(
                    deliberation=combined_delib,
                    item_id=it.item_id,
                    feedback=merged_fb,
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
                if is_real_genai_client(self.client) and it.spec_resets < MAX_SPEC_RESETS:
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
        for pivot_role in ("CreativeDirector", "GameplaySystemsEngineer"):
            spec = next((s for s in SPECIALIST_ROSTER if s["role"] == pivot_role), None)
            if spec is None:
                continue
            listen_check = WritersRoomListenCheck(
                deliberation=f"Immediate pivot required to honor Executive Producer steering: {combined_steering}",
                should_interject=True,
                urgency_score=1.0,
                intended_contribution_summary=f"Adapt the Game Bible and update/cancel/create ProductionBoard items to satisfy Executive Producer steering: {combined_steering}",
            )
            proposal = self._execute_specialist_turn(
                specialist=spec,
                listen_check=listen_check,
                game_bible=game_bible,
                dna_summary=dna_summary,
            )
            self._apply_turn_proposal(
                game_bible=game_bible,
                speaker_role=spec["role"],
                speaker_display_name=spec["display_name"],
                proposal=proposal,
                candidate_dir=candidate_dir,
            )

    def _drain_backlog(
        self,
        game_bible: GameBibleSpec,
        dna_summary: str,
        candidate_dir: Optional[Path],
    ) -> None:
        """Blocks until every backlog item is approved or cancelled, running review rounds as proofs arrive."""
        if self.board is None:
            return
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
        while True:
            paused_dur = self._wait_if_paused(candidate_dir)
            if paused_dur > 0:
                deadline += paused_dur
            steered = self._consume_pending_steering(game_bible, candidate_dir=candidate_dir)
            if steered:
                self._execute_steering_pivot_turns(game_bible, dna_summary, candidate_dir, steered)
            self._handle_failed_items()
            applied = self._run_review_round(game_bible, dna_summary, candidate_dir)
            if self.board.is_drained():
                return
            cur_progress_sig = _board_progress_sig()
            if applied > 0 or steered or cur_progress_sig != last_progress_sig:
                last_progress_sig = cur_progress_sig
                deadline = max(deadline, time.monotonic() + self.drain_timeout_s)
            active = [
                i for i in self.board.items()
                if i.state in ("ready", "in_progress", "changes_requested", "awaiting_review", "needs_decision")
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
            f"7. A COHESIVE WHOLE GAME, not a single scene: a Level Generation Strategy (how levels are produced and what they look like), a Core Loop at three time scales, "
            f"an Engagement Plan, an Onboarding Plan, and a Goal Hierarchy spanning short-, mid- and long-term goals. They must agree with one another: the onboarding must teach the "
            f"core loop's verbs, the level archetypes must exercise those verbs and escalate difficulty, and every short-term goal must serve a long-term goal. "
            f"Current structural gaps: {validate_game_design(game_bible) or 'none'}.\n"
            f"Return your structured DirectorConvergenceCheck with `deliberation` FIRST."
        )
        return self._call_structured_llm(prompt, DirectorConvergenceCheck, temperature=0.9)

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
                f"(each opens a different creative axis), phrased as open prompts/constraints/questions — not genre labels, not existing "
                f"game names, and deliberately outside conventional game-concept space.\n"
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
                    f"Contribute 2 independent concept fragments from YOUR domain, each a vivid, surprising seed (an interaction verb or system, "
                    f"a world tension, an emotional experience, a structural constraint, a sensory signature). Do NOT pitch a complete game, do "
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
                    f"Collide fragments from DIFFERENT specialists into 4 candidate premises that are distinct from one another and from the "
                    f"prior concepts. Each candidate must combine at least two fragments and yield something none implied alone. Premises are "
                    f"provisional starting points the room will evolve, not finished designs.\n"
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
                f"Choose the candidate with the greatest potential to evolve into something fresh, coherent and buildable once the room "
                f"pressure-tests it. Write `opening_dialogue` presenting it to the room as a draft to be challenged and reshaped, "
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
            dna_summary = (
                f"{dna_summary}\n"
                f"- EXECUTIVE PRODUCER CREATIVE SEED / DIRECTION: {creative_prompt.strip()}"
            )

        if existing_bible is not None:
            game_bible = existing_bible
            game_bible.status = "brainstorming"
        else:
            game_bible = GameBibleSpec(
                deliberation="Initializing Living Game Bible for open-ended multi-agent Writers' Room.",
                cartridge_id=cartridge_id,
                target_episode=target_episode,
                status="brainstorming",
                game_title="Untitled Concept",
                core_genre=(dna.top_genres[0] if (dna and getattr(dna, "top_genres", None)) else "indie_experience"),
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

        # Open-ended specialist conversation loop
        turns_executed = sum(spoken_counts.values())
        if self.board is not None and self.scheduler is not None:
            if turns_executed >= min(3, self.max_turns) or len(self.board.items()) >= 3:
                self.scheduler.start()
        while turns_executed < self.max_turns:
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
                # Wait to start the background worker scheduler until the initial Core Council pass (at least 3 turns
                # or max_turns if smaller) has completed so VisualArtDirector and fellow Leads can propose foundational
                # visual_asset rigs and wire `blocks_items` dependencies before mechanic_code workers claim tasks.
                if self.scheduler is not None and turns_executed >= min(3, self.max_turns):
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
            game_bible.slice_roadmap = [
                GameSliceSpec(
                    deliberation=(
                        "Strategizing Slice 0 from the locked Onboarding Plan and Core Loop so the first iteration "
                        "focuses on getting the player acclimated to the world and understanding the mechanics."
                    ),
                    slice_id="slice_0_acclimation",
                    slice_index=0,
                    title=f"{game_bible.game_title} — Slice 0: World Acclimation & Core Mechanics",
                    slice_objective=onb_intro,
                    player_acclimation_and_mechanics_focus=f"Player verbs ({verbs_str}) and moment-to-moment loop: {m2m}",
                    expansion_from_prior_slice="Foundational first slice establishing world acclimation and core mechanics.",
                    scope_boundaries=["Generate and wire ONLY the assets, animations, and systems required for Slice 0."],
                    status="active",
                )
            ]
            game_bible.active_slice_index = 0
        # ...and every requirement the council raised must be built and approved (or deliberately cancelled).
        self._drain_backlog(game_bible, dna_summary, candidate_dir)

        # Synthesize final GenesisWorldSpec on the locked GameBibleSpec
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

