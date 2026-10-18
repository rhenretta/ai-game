"""engine/studio/director.py
Creative Director & Worldbuilder Agent — Feature 17 & Milestone 1.
Conceives original game world lore, premise, aesthetic direction, and pacing grounded in SteamPlayerDNA.
Authors comprehensive GameRequirementsSpec decomposing games into verifiable tracks and asset manifests.
Enforces Rule 4 Rationale-First Epistemic Architecture (deliberation strictly field 0).
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional
import uuid

from engine.steam.models import SteamPlayerDNA
from engine.studio.models import (
    GenesisWorldSpec,
    UpdateWorldSpec,
    GameRequirementsSpec,
    StudioAgentError,
)

logger = logging.getLogger("GameConceptDirectorAgent")


class GameConceptDirectorAgent:
    """Specialist Agent that conceives world lore, premise, aesthetic direction, pacing,
    and authors comprehensive, verifiable GameRequirementsSpec backlogs from SteamPlayerDNA and telemetry.
    """
    deliberation: str = "Cognitive game concept director and requirements architect driven by live Gemini reasoning."

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = "gemini-3.8-flash",
        creative_temperature: float = 1.2,
    ):
        self.client = client
        self.model_name = model_name
        self.creative_temperature = float(creative_temperature)

    def generate_genesis_world(self, dna: SteamPlayerDNA) -> GenesisWorldSpec:
        """Derives an original GenesisWorldSpec from the user's SteamPlayerDNA."""
        if self.client is None:
            raise StudioAgentError("GameConceptDirectorAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)")

        try:
            from google.genai import types
            top_games_list = getattr(dna, "top_games", []) or []
            top_games_lines = "\n".join(
                f"  * {g.get('name', 'Unknown')} ({g.get('playtime_hours', 0)} hrs) — genres: {g.get('genres', [])}"
                for g in top_games_list[:10]
            )
            prompt = (
                f"You are the Creative Director AI for an autonomous game studio.\n"
                f"Analyze this player's actual Steam Library & Player DNA:\n"
                f"- Steam Profile Summary: {getattr(dna, 'deliberation', '')}\n"
                f"- Total playtime: {getattr(dna, 'total_playtime_hours', 0.0):.1f} hours across {getattr(dna, 'total_games_analyzed', 0)} games\n"
                f"- Top Genres: {dna.top_genres} (weights: {getattr(dna, 'genre_weights', {})})\n"
                f"- Pacing score: {dna.pacing_score:.2f} (-1.0 methodical/simulation/strategy to +1.0 fast kinetic action)\n"
                f"- Challenge tolerance: {dna.challenge_tolerance:.2f} (0.0 cozy to 1.0 unforgiving)\n"
                f"- Thematic biases: {dna.thematic_biases}\n"
                f"- Preferred palette mood: {dna.preferred_palette_mood}\n"
                + (f"- Player's Most-Played & Installed Steam Titles:\n{top_games_lines}\n" if top_games_lines else "")
                + f"Conceive a wildly imaginative, original, and deeply engaging 2D game world, setting, overarching lore, conflict narrative, "
                f"and core mechanics guidelines tailored specifically to what this player genuinely loves playing. "
                f"The world can span any genre or paradigm (action, puzzle, strategy, simulation, automation, adventure) without hardcoded arena constraints. "
                f"You MUST provide reasoning/deliberation FIRST before any other field."
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GenesisWorldSpec,
                    temperature=self.creative_temperature,
                ),
            )
            if not response.text:
                raise StudioAgentError("Empty response returned from Gemini worldbuilding call")
            data = json.loads(response.text)
            return GenesisWorldSpec(**data)
        except Exception as exc:
            logger.error("Gemini live worldbuilding call failed: %s", exc)
            raise StudioAgentError(f"CreativeDirector failed to conceive world: {exc}") from exc

    def generate_episodic_update_world(
        self,
        dna: SteamPlayerDNA,
        hypothesis_report: Any,
        previous_spec: GenesisWorldSpec,
        target_episode: int,
    ) -> UpdateWorldSpec:
        """Derives an episodic UpdateWorldSpec adapting the world based on telemetry hypotheses."""
        if self.client is None:
            raise StudioAgentError("GameConceptDirectorAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)")

        try:
            from google.genai import types
            hypotheses_text = getattr(hypothesis_report, "hypotheses", [])
            prompt = (
                f"You are the Creative Director AI.\n"
                f"Design episodic update {target_episode} continuing previous world: '{previous_spec.game_title}'.\n"
                f"Player telemetry hypotheses:\n{hypotheses_text}\n"
                f"Provide deliberation first, followed by story progression, clamped difficulty adjustments, and mechanics directives."
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=UpdateWorldSpec,
                    temperature=self.creative_temperature,
                ),
            )
            if not response.text:
                raise StudioAgentError("Empty response returned from Gemini update worldbuilding call")
            data = json.loads(response.text)
            return UpdateWorldSpec(**data)
        except Exception as exc:
            logger.error("Gemini update worldbuilding failed: %s", exc)
            raise StudioAgentError(f"CreativeDirector failed to conceive episodic update: {exc}") from exc

    def generate_requirements_spec(
        self,
        dna: Optional[SteamPlayerDNA] = None,
        world_spec: Optional[GenesisWorldSpec] = None,
        hypothesis_report: Optional[Any] = None,
        target_episode: int = 0,
        game_bible: Optional[Any] = None,
        narrative_bible: Optional[Any] = None,
    ) -> GameRequirementsSpec:
        """Authors a comprehensive GameRequirementsSpec decomposing the game concept into
        discrete, ordered implementation tracks and verifiable acceptance criteria.
        Enforces Rule 4 (deliberation first) and Rule 1 (zero hardcoded genre/character assumptions).
        """
        if self.client is None:
            raise StudioAgentError("GameConceptDirectorAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)")

        try:
            from google.genai import types

            dna_context_lines = []
            if dna is not None:
                top_games_list = getattr(dna, "top_games", []) or []
                top_games_str = ", ".join(g.get("name", "Unknown") for g in top_games_list[:5])
                dna_context_lines.extend([
                    f"- Total playtime: {getattr(dna, 'total_playtime_hours', 0.0):.1f} hours",
                    f"- Top Genres: {getattr(dna, 'top_genres', [])}",
                    f"- Pacing score: {getattr(dna, 'pacing_score', 0.0):.2f}",
                    f"- Challenge tolerance: {getattr(dna, 'challenge_tolerance', 0.5):.2f}",
                    f"- Thematic biases: {getattr(dna, 'thematic_biases', [])}",
                    f"- Preferred palette mood: {getattr(dna, 'preferred_palette_mood', 'retro_arcade')}",
                ])
                if top_games_str:
                    dna_context_lines.append(f"- Favorite titles: {top_games_str}")

            world_context_lines = []
            if world_spec is not None:
                world_context_lines.extend([
                    f"- Title: {world_spec.game_title}",
                    f"- Genre: {world_spec.core_genre}",
                    f"- Premise: {world_spec.thematic_premise}",
                    f"- Lore: {world_spec.world_lore}",
                    f"- Aesthetic Style: {world_spec.aesthetic_style}",
                    f"- Gameplay Pacing: {world_spec.gameplay_pacing}",
                ])

            bible_context_lines = []
            if game_bible is not None:
                storyboard_str = "\n".join(
                    f"  * Beat {b.beat_index} ({b.phase_title}): Emotion={b.player_emotional_state} | Staging={b.sensory_and_camera_staging} | Diegetic Teaching={b.diegetic_teaching_moment} | Story={b.narrative_integration}"
                    for b in (getattr(game_bible, "experience_storyboard", []) or [])
                )
                mockup_notes = [
                    f"{m.title}: {m.ui_callout_notes}"
                    for m in (getattr(game_bible, "visual_mockups", []) or [])
                ]
                bible_context_lines.extend([
                    f"- Working Title: {getattr(game_bible, 'game_title', '')}",
                    f"- Elevator Pitch: {getattr(game_bible, 'elevator_pitch', '')}",
                    f"- Core Player Fantasy: {getattr(game_bible, 'core_fantasy', '')}",
                    f"- First 60 Seconds Hook: {getattr(game_bible, 'first_60_seconds_hook', '')}",
                    f"- Intuitive UI & Onboarding Philosophy (clean readable HUD + responsive world cues): {getattr(game_bible, 'diegetic_ui_philosophy', '')}",
                    f"- Gameplay Pillars: {getattr(game_bible, 'gameplay_pillars', [])}",
                    f"- Visual Style Guide: {getattr(game_bible, 'visual_style_guide', '')}",
                    f"- Audio Direction Guide: {getattr(game_bible, 'audio_direction_guide', '')}",
                    f"- Visual Mockup Callouts: {mockup_notes}",
                ])
                if storyboard_str:
                    bible_context_lines.append(f"- Experience Storyboard:\n{storyboard_str}")

            hypothesis_lines = []
            if hypothesis_report is not None:
                dominant_style = getattr(hypothesis_report, "dominant_playstyle", "unknown")
                hyp_list = getattr(hypothesis_report, "hypotheses", [])
                hypothesis_lines.extend([
                    f"- Target Episode: {target_episode}",
                    f"- Dominant Playstyle: {dominant_style}",
                    f"- Hypotheses: {hyp_list}",
                ])

            dna_block = "\n".join(dna_context_lines) if dna_context_lines else "None provided."
            world_block = "\n".join(world_context_lines) if world_context_lines else "None provided."
            bible_block = "\n".join(bible_context_lines) if bible_context_lines else "None provided."
            hyp_block = "\n".join(hypothesis_lines) if hypothesis_lines else "None provided."

            prompt = (
                f"You are the Game Concept Director and Lead Systems Architect AI.\n"
                f"Your task is to deliberate on the player's profile and the Writers' Room Living Game Bible to author a comprehensive, rigorous Game Requirements & Feature Specification (`GameRequirementsSpec`).\n"
                f"\n--- PLAYER STEAM DNA ---\n{dna_block}\n"
                f"\n--- CONCEIVED WORLD BLUEPRINT ---\n{world_block}\n"
                f"\n--- LIVING GAME BIBLE (WRITERS' ROOM CONSENSUS) ---\n{bible_block}\n"
                f"\n--- EVOLUTION HYPOTHESES ---\n{hyp_block}\n"
                f"\n--- ARCHITECTURAL & AGENTIC DIRECTIVES ---\n"
                f"1. RATIONALE-FIRST MANDATE (Rule 4): You MUST provide 'deliberation' FIRST before any decision tokens, spec details, or requirement lists.\n"
                f"2. PURE GENERALIZABILITY MANDATE (Rule 1): Zero hardcoded genre, mechanic, character count, or arena prescriptions. "
                f"The game can be an abstract puzzle, simulation, automation, turn-based tactics, bullet heaven, or narrative adventure depending on player preferences. "
                f"If the game concept is character-less, it requires 0 characters.\n"
                f"3. EXPERIENCE-FIRST FIDELITY: Your RequirementTrack items MUST translate the Living Game Bible's First 60 Seconds Hook, Experience Storyboard Beats, Story/Cutscenes/Dialogue, and UI Philosophy into concrete, playable systems (combining clean, readable HUD indicators with responsive world cues).\n"
                f"4. PLAYABLE-WORLD-FIRST DECOMPOSITION: Decompose the game into ordered, discrete RequirementTrack items ('R1', 'R2', ...). "
                f"Track 'R1' MUST ALWAYS establish the visible, interactive exterior/world loop with spatial navigation, interactive targets/objectives, and clear visual feedback—never a passive background telemetry counter or empty stub.\n"
                f"   Each track must define its responsible agent(s), target code/asset modules, and explicit, verifiable AcceptanceCriterion items.\n"
                f"   Target modules MUST be Godot 4 GDScript modules chosen from ['main.gd', 'mechanics.gd', 'entities.gd', 'scenes.gd', 'ui_hud.gd'], NEVER Python (.py) files.\n"
                f"5. ZERO BLIND VIEWPORTS OR INSTANT LOCKOUTS: Never specify opaque full-screen overlays (>40% startup occlusion) that blind the player at boot, and never specify unrecoverable failure/game-over lockouts within the first 30 seconds of play; hazards must be recoverable via player actions.\n"
                f"6. VERIFIABLE ACCEPTANCE CRITERIA: Each criterion must declare a concrete verification_method ('ast_inspection', 'runtime_telemetry', 'headless_playtest', 'multi_modal', or 'design_critic') and explicit expected behavior.\n"
                f"7. DYNAMIC MULTIMODAL ASSET MANIFEST: Define exact sprite, tile, portrait, and audio assets required in asset_manifest as AssetRequirement items, avoiding hardcoded legacy arena asset names.\n"
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GameRequirementsSpec,
                    temperature=1.0,
                ),
            )
            if not response.text:
                raise StudioAgentError("Empty response returned from Gemini requirements authoring call")
            data = json.loads(response.text)
            spec = GameRequirementsSpec(**data)
            if game_bible is not None and getattr(game_bible, "game_title", None) and game_bible.game_title != "Untitled Concept":
                spec.game_title = game_bible.game_title
            # Post-validation to enforce GDScript 4 target_modules (Milestone 7 Godot 4 Transition)
            for req in spec.requirements:
                sanitized_mods = []
                for mod in getattr(req, "target_modules", []):
                    mod_gd = mod.replace(".py", ".gd")
                    if not mod_gd.endswith(".gd"):
                        mod_gd = f"{mod_gd}.gd"
                    sanitized_mods.append(mod_gd)
                req.target_modules = sanitized_mods or ["mechanics.gd"]
            return spec
        except Exception as exc:
            logger.error("Gemini requirements authoring failed: %s", exc)
            raise StudioAgentError(f"GameConceptDirectorAgent failed to author requirements spec: {exc}") from exc


CreativeDirectorWorldbuilderAgent = GameConceptDirectorAgent
