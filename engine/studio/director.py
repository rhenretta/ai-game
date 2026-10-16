"""engine/studio/director.py
Creative Director & Worldbuilder Agent — Feature 17.
Conceives original game world lore, premise, aesthetic direction, and pacing grounded in SteamPlayerDNA.
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
from engine.studio.models import GenesisWorldSpec, UpdateWorldSpec, StudioAgentError

logger = logging.getLogger("CreativeDirectorWorldbuilderAgent")


class CreativeDirectorWorldbuilderAgent:
    """Specialist Agent that conceives world lore, premise, aesthetic direction, and pacing."""
    deliberation: str = "Cognitive worldbuilding agent driven by live Gemini reasoning."

    def __init__(self, client: Optional[Any] = None, model_name: str = "gemini-3.8-flash"):
        self.client = client
        self.model_name = model_name

    def generate_genesis_world(self, dna: SteamPlayerDNA) -> GenesisWorldSpec:
        """Derives an original GenesisWorldSpec from the user's SteamPlayerDNA."""
        if self.client is None:
            raise StudioAgentError("CreativeDirectorWorldbuilderAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)")

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
            raise StudioAgentError("CreativeDirectorWorldbuilderAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)")

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
