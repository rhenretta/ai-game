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
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.steam.models import SteamPlayerDNA
from engine.studio.models import (
    CharacterSheet,
    ConceptEvolutionEntry,
    DirectorConvergenceCheck,
    ExperienceStoryboardBeat,
    GameBibleSpec,
    GenesisWorldSpec,
    NarrativeBible,
    StudioAgentError,
    VisualMockupArtifact,
    WritersRoomListenCheck,
    WritersRoomTurnProposal,
    WritersRoomUtterance,
)

logger = logging.getLogger("CreativeWritersRoom")


SPECIALIST_ROSTER: List[Dict[str, str]] = [
    {
        "role": "GameplayDesigner",
        "display_name": "Lead Gameplay & Systems Designer",
        "focus": (
            "Tactile feel, core game loop, emergent mechanical depth, moment-to-moment player agency, "
            "and why the mechanics are genuinely fun and immediately playable in a 2D viewport."
        ),
    },
    {
        "role": "StoryWorldArchitect",
        "display_name": "Story & World Lore Architect",
        "focus": (
            "Imaginative world-building, central dramatic conflict, environmental storytelling, "
            "narrative stakes, and cutscenes that pull the player into a vivid, living world."
        ),
    },
    {
        "role": "CharacterDesigner",
        "display_name": "Character & Dialogue Designer",
        "focus": (
            "Memorable characters or entities with distinct silhouettes, personalities, motivations, "
            "branching dialogue trees, and reactive voice barks."
        ),
    },
    {
        "role": "UXExperienceDesigner",
        "display_name": "Player Experience & UX Designer",
        "focus": (
            "The player's emotional and sensory journey (Experience Storyboard), the First 60 Seconds Hook, "
            "and Intuitive Onboarding & Readable UI—combining clean, legible HUD elements with responsive "
            "in-world visual cues so the player immediately understands their goal and controls."
        ),
    },
    {
        "role": "VisualArtDirector",
        "display_name": "Visual Art & Mockup Director",
        "focus": (
            "Cohesive visual identity, color mood, lighting, camera staging, and authoring 16:9 full-screen "
            "VisualMockupArtifact boards that show an unobstructed, vibrant, readable 2D game world and clean UI."
        ),
    },
    {
        "role": "AudioAtmosphereDesigner",
        "display_name": "Audio & Voice Atmosphere Designer",
        "focus": (
            "Acoustic atmosphere, reactive sound design, character voice casting, and emotional cadence."
        ),
    },
]


CREATIVE_EXPLORATION_LENSES: List[str] = [
    "Kinetic physics & expressive spatial momentum (inspired by high-energy movement, trajectories, and satisfying collisions)",
    "Clever spatial puzzle-adventure & environmental contraptions (inspired by witty discovery, chain reactions, and portal/circuit logic)",
    "Vibrant world-building & automated ecosystem harmony (inspired by building living networks, whimsical machinery, or thriving colonies)",
    "Tactical expedition & mystical frontier exploration (inspired by charting unknown realms, resource synergy, and colorful encounters)",
    "High-craft systemic sandbox & playful vehicle/gadget invention (inspired by modular tools, emergent physics, and clear visual feedback)",
]


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

    def _persist_bible(self, game_bible: GameBibleSpec, candidate_dir: Optional[Path] = None) -> None:
        """Saves game_bible.json (and world_spec.json / narrative_bible.json if populated) to disk and upserts to SQLite studio_game_bibles."""
        game_bible.updated_at_utc = datetime.now(timezone.utc).isoformat()
        if candidate_dir is not None:
            candidate_dir = Path(candidate_dir)
            candidate_dir.mkdir(parents=True, exist_ok=True)
            bible_path = candidate_dir / "game_bible.json"
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
    ) -> Any:
        """Invokes Gemini with structured JSON schema and high creative temperature, returning validated Pydantic instance."""
        if self.client is None:
            raise StudioAgentError(
                "CreativeWritersRoom requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)"
            )
        from google.genai import types

        effective_temp = self.creative_temperature if temperature is None else float(temperature)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_cls,
                temperature=effective_temp,
            ),
        )
        if not response.text:
            raise StudioAgentError(f"Empty response from Gemini for schema {schema_cls.__name__}")
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
            lens = random.choice(CREATIVE_EXPLORATION_LENSES)
            top_genres = list(getattr(dna, "top_genres", []) or [])
            if len(top_genres) > 1:
                shuffled_genres = list(top_genres)
                random.shuffle(shuffled_genres)
            else:
                shuffled_genres = top_genres
            lines.extend([
                f"- Genre Affinities to Explore: {shuffled_genres}",
                f"- Pacing Preference: {getattr(dna, 'pacing_preference', 'balanced')} (balance thoughtful strategy with satisfying, responsive moment-to-moment movement)",
                f"- Preferred Palette Mood: {getattr(dna, 'preferred_palette_mood', 'vibrant_fantasy')}",
                f"- Session Creative Exploration Lens: {lens}",
            ])
            if spotlight_str:
                lines.append(f"- Spotlight Titles from Player Library for This Session: {spotlight_str}")
        if hypothesis_report is not None:
            hyps = getattr(hypothesis_report, "hypotheses", []) or []
            if hyps:
                lines.append(f"- Empirical Player Telemetry Hypotheses: {[getattr(h, 'statement', str(h)) for h in hyps]}")
        return "\n".join(lines) if lines else "- Open creative mandate: craft a deeply polished, unforgettable game experience."

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
        return (
            f"CURRENT LIVING GAME BIBLE STATE:\n"
            f"- Working Title: {game_bible.game_title}\n"
            f"- Core Genre: {game_bible.core_genre}\n"
            f"- Elevator Pitch: {game_bible.elevator_pitch or '(not yet defined)'}\n"
            f"- Core Fantasy: {game_bible.core_fantasy or '(not yet defined)'}\n"
            f"- First 60 Seconds Hook: {game_bible.first_60_seconds_hook or '(not yet defined)'}\n"
            f"- Diegetic UI & Onboarding Philosophy: {game_bible.diegetic_ui_philosophy or '(not yet defined)'}\n"
            f"- Gameplay Pillars ({len(game_bible.gameplay_pillars)}): {game_bible.gameplay_pillars}\n"
            f"- Visual Style Guide: {game_bible.visual_style_guide or '(not yet defined)'}\n"
            f"- Audio Direction Guide: {game_bible.audio_direction_guide or '(not yet defined)'}\n"
            f"- Experience Storyboard Beats ({len(beats)}): {beats}\n"
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

        if proposal.updated_color_palette_hex:
            game_bible.color_palette_hex = [str(c).strip() for c in proposal.updated_color_palette_hex if str(c).strip()]
            updates_summary.append(f"Updated color palette ({len(game_bible.color_palette_hex)} hex swatches)")

        if proposal.added_or_updated_scene_layers:
            layers_by_id = {l.layer_id: l for l in game_bible.scene_layers}
            for layer in proposal.added_or_updated_scene_layers:
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

        # Characters & Portrait synthesis (with domain-agnostic deduplication across turns, isolated in assets/portraits/)
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
            for char_sheet in proposal.added_or_updated_characters:
                if self.visual_agent is not None and candidate_dir is not None and getattr(self.visual_agent, "client", None) is not None:
                    try:
                        portrait_dest = Path(candidate_dir) / char_sheet.portrait_asset
                        if not portrait_dest.is_file():
                            self.visual_agent.generate_portrait(
                                character=char_sheet,
                                output_path=portrait_dest,
                                dimensions=(256, 256),
                                style_guide=game_bible.visual_style_guide,
                            )
                    except Exception as exc:
                        logger.warning("Writers' Room portrait synthesis warning for '%s': %s", char_sheet.name, exc)
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

        # Dialogue barks & Voice Audition synthesis
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
                    # Match by normalized speaker tokens if speaker_id differed slightly
                    spk_tokens = _normalize_char_tokens(bark.speaker_id, drop_generic=True)
                    for c_cand in game_bible.narrative_bible.characters:
                        c_tokens = _normalize_char_tokens(c_cand.name, drop_generic=True) | _normalize_char_tokens(c_cand.character_id, drop_generic=True)
                        if spk_tokens and spk_tokens.issubset(c_tokens):
                            char_obj = c_cand
                            bark.speaker_id = c_cand.character_id
                            break
                v_name = char_obj.gemini_voice_name if char_obj else "Aoede"
                if candidate_dir is not None:
                    wav_dest = Path(candidate_dir) / bark.voice_clip
                    try:
                        if self.audio_agent is not None and hasattr(self.audio_agent, "synthesize_voice_line"):
                            self.audio_agent.synthesize_voice_line(
                                bark=bark,
                                output_path=wav_dest,
                                character=char_obj,
                                voice_name=v_name,
                            )
                        elif (
                            self.narrative_agent is not None
                            and getattr(self.narrative_agent, "client", None) is not None
                            and hasattr(self.narrative_agent, "synthesize_bark_audio")
                        ):
                            self.narrative_agent.synthesize_bark_audio(
                                bark=bark,
                                output_path=wav_dest,
                                voice_name=v_name,
                            )
                    except Exception as exc:
                        logger.warning("Writers' Room voice audition synthesis warning for '%s': %s", bark.bark_id, exc)
                existing_barks[bark.bark_id] = bark
            game_bible.narrative_bible.dialogue_barks = list(existing_barks.values())
            updates_summary.append(f"Added {len(proposal.added_or_updated_barks)} voice bark(s)")

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
        specialist: Dict[str, str],
        game_bible: GameBibleSpec,
        dna_summary: str,
    ) -> WritersRoomListenCheck:
        """Asks a listening specialist if they want to interject into the Writers' Room right now."""
        role = specialist["role"]
        display_name = specialist["display_name"]
        focus = specialist["focus"]
        bible_snapshot = self._format_bible_snapshot(game_bible)

        prompt = (
            f"You are the {display_name} ({role}) sitting in an open-ended Game Studio Writers' Room.\n"
            f"Your specialty focus: {focus}\n\n"
            f"PLAYER DNA & STUDIO CONTEXT:\n{dna_summary}\n\n"
            f"{bible_snapshot}\n"
            f"Listen to the conversation above and inspect the Living Game Bible.\n"
            f"Decide whether you should interject right now to:\n"
            f"1. Challenge a generic/boring 'Hello World' idea and pitch a much richer experiential hook,\n"
            f"2. Answer a colleague who called on you, or\n"
            f"3. Contribute missing artifacts in your domain (e.g., Experience Storyboard & Diegetic UI for UXExperienceDesigner, "
            f"16:9 Visual Mockup for VisualArtDirector, Characters/Dialogue/Barks for CharacterDesigner, World Lore/Cutscenes for StoryWorldArchitect).\n"
            f"Return your structured WritersRoomListenCheck with deliberation FIRST."
        )
        return self._call_structured_llm(prompt, WritersRoomListenCheck)

    def _execute_specialist_turn(
        self,
        specialist: Dict[str, str],
        listen_check: WritersRoomListenCheck,
        game_bible: GameBibleSpec,
        dna_summary: str,
    ) -> WritersRoomTurnProposal:
        """Executes the chosen specialist's turn to speak to the room and update the Living Game Bible."""
        role = specialist["role"]
        display_name = specialist["display_name"]
        focus = specialist["focus"]
        bible_snapshot = self._format_bible_snapshot(game_bible)

        domain_instructions = ""
        if role == "UXExperienceDesigner":
            domain_instructions = (
                "As UXExperienceDesigner, you MUST populate `updated_first_60_seconds_hook`, "
                "`updated_diegetic_ui_philosophy` (explaining how crisp, readable HUD indicators and responsive in-world visual cues work together without obscuring the playable 2D world), "
                "and `added_or_updated_storyboard_beats` (at least 3-4 ExperienceStoryboardBeat items covering the emotional & sensory arc)."
            )
        elif role == "VisualArtDirector":
            domain_instructions = (
                "As VisualArtDirector, you MUST populate `visual_style_guide_update` AND propose a `mockup_to_generate` "
                "(VisualMockupArtifact with a rich 16:9 `scene_description` showing a clear, unobstructed 2D gameplay viewport and specific `ui_callout_notes`) so the studio synthesizes a visual concept board right now."
            )
        elif role == "CharacterDesigner":
            domain_instructions = (
                "As CharacterDesigner, you MUST populate `added_or_updated_characters` with a diverse cast of distinct characters "
                "(e.g., protagonist, companion/contact, rival/antagonist—each with a unique `character_id`, distinct visual description, and voice timbre; "
                "when updating an existing character already listed in the Bible, reuse their exact `character_id` rather than creating a duplicate entry), "
                "`added_or_updated_dialogue_trees`, and `added_or_updated_barks`."
            )
        elif role == "StoryWorldArchitect":
            domain_instructions = (
                "As StoryWorldArchitect, you MUST deepen `updated_core_fantasy`, `updated_elevator_pitch`, "
                "log a `concept_evolution_milestone` showing how the narrative elevated the premise, and author `added_or_updated_cutscenes`."
            )
        elif role == "GameplayDesigner":
            domain_instructions = (
                "As GameplayDesigner, you MUST challenge any shallow prototype mechanics, add concrete tactile pillars in "
                "`gameplay_pillars_additions` that are genuinely fun to play in a 2D viewport, and log a `concept_evolution_milestone` explaining how the gameplay loop evolved."
            )
        elif role == "AudioAtmosphereDesigner":
            domain_instructions = (
                "As AudioAtmosphereDesigner, you MUST define `audio_direction_update` and contribute atmospheric or reactive `added_or_updated_barks`."
            )

        prompt = (
            f"You are the {display_name} ({role}) speaking in the Game Studio Writers' Room.\n"
            f"Your specialty focus: {focus}\n"
            f"Why you raised your hand to interject: {listen_check.intended_contribution_summary}\n\n"
            f"PLAYER DNA & STUDIO CONTEXT:\n{dna_summary}\n\n"
            f"{bible_snapshot}\n"
            f"{domain_instructions}\n"
            f"Speak naturally, creatively, and passionately to your fellow designers in `spoken_dialogue` (never recite raw telemetry floats), "
            f"call out specific colleagues in `addressed_to`, and contribute your concrete updates to the Living Game Bible. Provide `deliberation` FIRST."
        )
        return self._call_structured_llm(prompt, WritersRoomTurnProposal)

    def _select_next_speaker(
        self,
        last_speaker_role: str,
        addressed_to: List[str],
        game_bible: GameBibleSpec,
        dna_summary: str,
        spoken_counts: Dict[str, int],
    ) -> Tuple[Dict[str, str], WritersRoomListenCheck]:
        """Polls specialists to see who wants to interject next based on the conversation and Game Bible gaps."""
        candidates: List[Tuple[float, Dict[str, str], WritersRoomListenCheck]] = []

        for spec in SPECIALIST_ROSTER:
            role = spec["role"]
            if role == last_speaker_role:
                continue
            check = self._evaluate_specialist_listen(spec, game_bible, dna_summary)
            effective_score = float(check.urgency_score)
            if role in (addressed_to or []):
                effective_score += 0.35
            if spoken_counts.get(role, 0) == 0:
                effective_score += 0.25
            # Boost specialists whose core Bible sections are still empty
            if role == "UXExperienceDesigner" and not game_bible.experience_storyboard:
                effective_score += 0.4
            elif role == "VisualArtDirector" and not game_bible.visual_mockups:
                effective_score += 0.4
            elif role == "CharacterDesigner" and (not game_bible.narrative_bible or not game_bible.narrative_bible.characters):
                effective_score += 0.35
            elif role == "StoryWorldArchitect" and (not game_bible.narrative_bible or not game_bible.narrative_bible.cutscenes):
                effective_score += 0.3
            elif role == "GameplayDesigner" and len(game_bible.gameplay_pillars) < 2:
                effective_score += 0.3

            candidates.append((effective_score, spec, check))

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, chosen_spec, chosen_check = candidates[0]
        return chosen_spec, chosen_check

    def _evaluate_convergence(self, game_bible: GameBibleSpec, dna_summary: str) -> DirectorConvergenceCheck:
        """CreativeDirector evaluates whether the Living Game Bible is cohesive and ready to lock."""
        bible_snapshot = self._format_bible_snapshot(game_bible)
        prompt = (
            f"You are the CreativeDirector of the Game Studio Writers' Room.\n"
            f"PLAYER DNA & STUDIO CONTEXT:\n{dna_summary}\n\n"
            f"{bible_snapshot}\n"
            f"Critically evaluate whether the Living Game Bible is ready to lock and enter production.\n"
            f"Do NOT approve a shallow 'Hello World' prototype. To be ready (`is_ready_to_lock_bible = true`), the Bible MUST have:\n"
            f"1. A deeply evocative premise, core fantasy, and Concept Evolution history,\n"
            f"2. A compelling First 60 Seconds Hook and Intuitive UI & Onboarding Philosophy (clean, readable HUD paired with responsive world cues),\n"
            f"3. An Experience Storyboard detailing the player's emotional & sensory arc,\n"
            f"4. Synthesized 16:9 Visual Mockup board(s),\n"
            f"5. Cohesive World Lore, Characters/Entities, Cutscenes, and Audio/Voice direction.\n"
            f"Return your structured DirectorConvergenceCheck with `deliberation` FIRST."
        )
        return self._call_structured_llm(prompt, DirectorConvergenceCheck, temperature=0.9)

    def _synthesize_final_world_spec(self, game_bible: GameBibleSpec, dna: Optional[SteamPlayerDNA], target_episode: int) -> GenesisWorldSpec:
        """Synthesizes the authoritative GenesisWorldSpec grounded in the locked Living Game Bible."""
        bible_snapshot = self._format_bible_snapshot(game_bible)
        mood_hint = getattr(dna, "preferred_palette_mood", "vibrant_fantasy") if dna is not None else "vibrant_fantasy"
        prompt = (
            f"You are the CreativeDirector finalizing the authoritative GenesisWorldSpec from the Writers' Room Living Game Bible.\n"
            f"Target Episode: {target_episode}\n"
            f"Preferred Palette Mood Hint: {mood_hint}\n\n"
            f"{bible_snapshot}\n"
            f"Synthesize the complete GenesisWorldSpec honoring every single pillar, UI & onboarding rule, camera staging, "
            f"and story hook agreed upon by the Writers' Room. Provide `deliberation` FIRST."
        )
        world_spec: GenesisWorldSpec = self._call_structured_llm(prompt, GenesisWorldSpec)
        world_spec.target_episode = target_episode
        if game_bible.game_title and game_bible.game_title != "Untitled Concept":
            world_spec.game_title = game_bible.game_title
        return world_spec

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

        # Opening pitch/provocation by CreativeDirector if transcript is empty
        if not game_bible.writers_room_transcript:
            seed_directive = (
                f"\nEXECUTIVE PRODUCER SEED TO BUILD UPON: {creative_prompt.strip()}\n"
                if (creative_prompt and creative_prompt.strip())
                else ""
            )
            opening_prompt = (
                f"You are the CreativeDirector opening the Game Studio Writers' Room for Episode {target_episode}.\n"
                f"PLAYER DNA & STUDIO CONTEXT:\n{dna_summary}\n"
                f"{seed_directive}\n"
                f"Open the room with a bold, wildly imaginative, and fresh game concept inspired by the Session Creative Exploration Lens "
                f"and Spotlight Titles above. Do NOT recite raw telemetry numbers in your dialogue, and avoid repetitive dreary/claustrophobic "
                f"cockpit-gauge or rusted derelict-crawler clichés unless explicitly requested by the Executive Producer seed. "
                f"Pitch a vivid, visually readable, immediately playable 2D world with joyful or compelling mechanics, clear UI + world feedback, "
                f"an evocative initial title, and challenge specific specialists in `addressed_to` (using exact roster roles like "
                f"GameplayDesigner, StoryWorldArchitect, CharacterDesigner, UXExperienceDesigner, VisualArtDirector, AudioAtmosphereDesigner) "
                f"to push the concept further. Provide `deliberation` FIRST."
            )
            opening_proposal: WritersRoomTurnProposal = self._call_structured_llm(opening_prompt, WritersRoomTurnProposal)
            opening_utt = self._apply_turn_proposal(
                game_bible=game_bible,
                speaker_role="CreativeDirector",
                speaker_display_name="Creative Director",
                proposal=opening_proposal,
                candidate_dir=candidate_dir,
            )
            last_addressed_to = opening_utt.addressed_to

        # Open-ended specialist conversation loop
        turns_executed = 0
        while turns_executed < self.max_turns:
            chosen_spec, listen_check = self._select_next_speaker(
                last_speaker_role=last_speaker_role,
                addressed_to=last_addressed_to,
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
            last_speaker_role = role
            last_addressed_to = utterance.addressed_to
            turns_executed += 1

            # Check convergence once minimum turns have elapsed
            if turns_executed >= self.min_turns_before_convergence:
                convergence = self._evaluate_convergence(game_bible, dna_summary)
                has_minimum_artifacts = (
                    bool(game_bible.experience_storyboard)
                    and bool(game_bible.first_60_seconds_hook)
                    and bool(game_bible.diegetic_ui_philosophy)
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

        # Synthesize final GenesisWorldSpec on the locked GameBibleSpec
        game_bible.world_spec = self._synthesize_final_world_spec(game_bible, dna, target_episode)
        if game_bible.world_spec and game_bible.world_spec.core_genre:
            game_bible.core_genre = game_bible.world_spec.core_genre
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

