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

    def __init__(self, client: Optional[Any] = None, model_name: str = "gemini-3.8-flash"):
        self.client = client
        self.model_name = model_name

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
                + f"Conceive an original, deeply engaging game world, setting, overarching lore, conflict narrative, "
                f"and core mechanics guidelines tailored specifically to what this player genuinely loves playing. "
                f"The world can span any genre or paradigm (action, puzzle, strategy, simulation, automation) without hardcoded arena constraints. "
                f"You MUST provide reasoning/deliberation FIRST before any other field."
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GenesisWorldSpec,
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
            hyp_block = "\n".join(hypothesis_lines) if hypothesis_lines else "None provided."

            prompt = (
                f"You are the Game Concept Director and Lead Systems Architect AI.\n"
                f"Your task is to deliberate on the player's profile and author a comprehensive, rigorous Game Requirements & Feature Specification (`GameRequirementsSpec`).\n"
                f"\n--- PLAYER STEAM DNA ---\n{dna_block}\n"
                f"\n--- CONCEIVED WORLD BLUEPRINT ---\n{world_block}\n"
                f"\n--- EVOLUTION HYPOTHESES ---\n{hyp_block}\n"
                f"\n--- ARCHITECTURAL & AGENTIC DIRECTIVES ---\n"
                f"1. RATIONALE-FIRST MANDATE (Rule 4): You MUST provide 'deliberation' FIRST before any decision tokens, spec details, or requirement lists.\n"
                f"2. PURE GENERALIZABILITY MANDATE (Rule 1): Zero hardcoded genre, mechanic, character count, or arena prescriptions. "
                f"The game can be an abstract puzzle, simulation, automation, turn-based tactics, bullet heaven, or narrative adventure depending on player preferences. "
                f"If the game concept is character-less, it requires 0 characters.\n"
                f"3. REQUIREMENTS DECOMPOSITION: Decompose the game into ordered, discrete RequirementTrack items ('R1', 'R2', ...). "
                f"Each track must define its responsible agent(s), target code/asset modules, and explicit, verifiable AcceptanceCriterion items.\n"
                f"Target modules MUST be Godot 4 GDScript modules chosen from ['main.gd', 'mechanics.gd', 'entities.gd', 'scenes.gd', 'ui_hud.gd'], NEVER Python (.py) files.\n"
                f"4. VERIFIABLE ACCEPTANCE CRITERIA: Each criterion must declare a concrete verification_method ('ast_inspection', 'runtime_telemetry', 'headless_playtest', 'multi_modal', or 'design_critic') and explicit expected behavior.\n"
                f"5. DYNAMIC MULTIMODAL ASSET MANIFEST: Define exact sprite, tile, portrait, and audio assets required in asset_manifest as AssetRequirement items, avoiding hardcoded legacy arena asset names.\n"
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GameRequirementsSpec,
                ),
            )
            if not response.text:
                raise StudioAgentError("Empty response returned from Gemini requirements authoring call")
            data = json.loads(response.text)
            spec = GameRequirementsSpec(**data)
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
