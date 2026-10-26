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
    bounded_thinking_config,
    call_gemini_with_transient_retry,
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
        creative_temperature: float = 0.7,
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
            response = call_gemini_with_transient_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GenesisWorldSpec,
                        temperature=self.creative_temperature,
                        thinking_config=bounded_thinking_config(2048),
                    ),
                ),
                label="GameConceptDirectorAgent.generate_genesis_world",
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
            response = call_gemini_with_transient_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=UpdateWorldSpec,
                        temperature=self.creative_temperature,
                        thinking_config=bounded_thinking_config(2048),
                    ),
                ),
                label="GameConceptDirectorAgent.generate_episodic_update_world",
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
        candidate_dir: Optional[Any] = None,
    ) -> GameRequirementsSpec:
        """Authors a comprehensive GameRequirementsSpec decomposing the game concept into
        discrete, ordered implementation tracks, verifiable acceptance criteria, and a
        concept-grounded Cartridge Asset Library manifest.
        Enforces Rule 4 (deliberation first) and Rule 1 (zero hardcoded genre/character assumptions).
        """
        if self.client is None:
            raise StudioAgentError("GameConceptDirectorAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)")

        try:
            from pathlib import Path
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
            mockup_image_parts: List[Any] = []
            if game_bible is not None:
                storyboard_str = "\n".join(
                    f"  * Beat {b.beat_index} ({b.phase_title}): Emotion={b.player_emotional_state} | Staging={b.sensory_and_camera_staging} | Diegetic Teaching={b.diegetic_teaching_moment} | Story={b.narrative_integration}"
                    for b in (getattr(game_bible, "experience_storyboard", []) or [])
                )
                mockup_notes = [
                    f"{m.title} ({getattr(m, 'camera_and_spatial_paradigm', '')}): {m.scene_description} | Callouts: {m.ui_callout_notes}"
                    for m in (getattr(game_bible, "visual_mockups", []) or [])
                ]
                eng_blueprint_str = "\n".join(
                    f"  * [{e.mechanic_id}] {e.title} (modules={e.target_gd_modules}): "
                    f"Nodes/Colliders={e.godot_node_and_collider_architecture} | "
                    f"State Machine={e.state_machine_and_input_contract} | "
                    f"Required Assets={e.required_sprite_and_tile_assets} | "
                    f"Test Invariants={e.falsifiable_test_invariants}"
                    for e in (getattr(game_bible, "technical_architecture_blueprint", []) or [])
                )
                bible_context_lines.extend([
                    f"- Working Title: {getattr(game_bible, 'game_title', '')}",
                    f"- Elevator Pitch: {getattr(game_bible, 'elevator_pitch', '')}",
                    f"- Core Player Fantasy: {getattr(game_bible, 'core_fantasy', '')}",
                    f"- First 60 Seconds Hook: {getattr(game_bible, 'first_60_seconds_hook', '')}",
                    f"- Camera & Spatial Paradigm: {getattr(game_bible, 'camera_and_spatial_paradigm', '')}",
                    f"- Color Palette Hex: {getattr(game_bible, 'color_palette_hex', [])}",
                    f"- UI, HUD & Onboarding Philosophy: {getattr(game_bible, 'diegetic_ui_philosophy', '')}",
                    f"- Gameplay Pillars: {getattr(game_bible, 'gameplay_pillars', [])}",
                    f"- Visual Style Guide: {getattr(game_bible, 'visual_style_guide', '')}",
                    f"- Audio Direction Guide: {getattr(game_bible, 'audio_direction_guide', '')}",
                    f"- Visual Mockup Callouts: {mockup_notes}",
                ])
                if eng_blueprint_str:
                    bible_context_lines.append(
                        f"- Technical Architecture Blueprint (from Writers' Room GameplaySystemsEngineer — honor these exact node/collider strategies, state machines, required assets, and invariants):\n{eng_blueprint_str}"
                    )
                if storyboard_str:
                    bible_context_lines.append(f"- Experience Storyboard:\n{storyboard_str}")

                # Attach 16:9 mockup image bytes multimodally when running with a live genai client
                mod_name = getattr(type(self.client), "__module__", "") or ""
                if mod_name.startswith("google.genai") and candidate_dir is not None:
                    for m in (getattr(game_bible, "visual_mockups", []) or [])[:1]:
                        img_rel = getattr(m, "image_path", "") or ""
                        if img_rel:
                            mp = Path(candidate_dir) / img_rel
                            if mp.is_file():
                                try:
                                    mockup_image_parts.append(
                                        types.Part.from_bytes(data=mp.read_bytes(), mime_type="image/png")
                                    )
                                except Exception:
                                    pass

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
                f"Your task is to deliberate on the player's profile and the Writers' Room Living Game Bible (and 16:9 concept mockup) to author a comprehensive, rigorous Game Requirements & Feature Specification (`GameRequirementsSpec`).\n"
                f"\n--- PLAYER STEAM DNA ---\n{dna_block}\n"
                f"\n--- CONCEIVED WORLD BLUEPRINT ---\n{world_block}\n"
                f"\n--- LIVING GAME BIBLE (WRITERS' ROOM CONSENSUS) ---\n{bible_block}\n"
                f"\n--- EVOLUTION HYPOTHESES ---\n{hyp_block}\n"
                f"\n--- ARCHITECTURAL & AGENTIC DIRECTIVES ---\n"
                f"1. RATIONALE-FIRST MANDATE (Rule 4): You MUST provide 'deliberation' FIRST before any decision tokens, spec details, or requirement lists.\n"
                f"2. PURE GENERALIZABILITY MANDATE (Rule 1): Zero hardcoded genre, mechanic, character count, or arena prescriptions. "
                f"The game can be any genre or camera perspective conceived by the Writers' Room.\n"
                f"3. EXPERIENCE & MOCKUP-TO-SCENE FIDELITY: Your `high_concept_summary` and `RequirementTrack` items MUST explicitly lock the camera perspective and visual scene staging from the 16:9 Writers' Room concept mockup (layered sky/backdrop, parallax midground, procedural or textured terrain/platforms, animated characters/entities, interactive props, atmospheric particles/shaders, and UI/HUD).\n"
                f"4. PLAYABLE-WORLD-FIRST DECOMPOSITION: Decompose the game into ordered, discrete RequirementTrack items ('R1', 'R2', ...). "
                f"Track 'R1' MUST ALWAYS establish the multi-layered visual environment (`scenes.gd`), animated player/entity actors (`entities.gd`), and core movement/traversal loop (`main.gd`, `mechanics.gd`) matching the 16:9 mockup.\n"
                f"   Target modules MUST be Godot 4 GDScript modules chosen from ['main.gd', 'mechanics.gd', 'entities.gd', 'scenes.gd', 'ui_hud.gd'], NEVER Python (.py) files.\n"
                f"5. ZERO BLIND VIEWPORTS OR INSTANT LOCKOUTS: Never specify opaque full-screen overlays (>40% startup occlusion) that blind the player at boot, and never specify unrecoverable failure/game-over lockouts within the first 30 seconds of play.\n"
                f"6. VERIFIABLE ACCEPTANCE CRITERIA: Each criterion must declare a concrete verification_method ('ast_inspection', 'runtime_telemetry', 'headless_playtest', 'multi_modal', or 'design_critic') and explicit expected behavior.\n"
                f"7. DECOMPOSE CONCEPT ART INTO COMPOSABLE WORLD ELEMENTS (`asset_manifest`) & PROCEDURAL RENDERING: NEVER slap a static full-screen background painting behind the game. Combine Godot 4 procedural rendering (`Polygon2D`, `Line2D`, custom `CanvasItem` shaders / `ShaderMaterial`, `GPUParticles2D`, `PointLight2D`) for dynamic fluids, energy beams, conduits, and sculpted terrain with discrete visual assets in `asset_manifest` where appropriate:\n"
                f"   - Use `fabrication_mode='static_sprite'` for isolated alpha-transparent world props, machines, platforms with silhouettes, and horizon/midground set-pieces.\n"
                f"   - Use `fabrication_mode='articulated_parts'` when an entity, character, or machine needs independently rotating/moving parts.\n"
                f"   - Use `fabrication_mode='sprite_sheet'` (with `animation_states` such as 'idle', 'move', 'jump', 'action') for frame-animated in-world characters, companions, and reactive props in `assets/sprites/`.\n"
                f"   - Use `fabrication_mode='tile_surface'` ONLY for uniform, borderless interior fill materials—never for fluids, beams, channels, or objects with silhouettes/edges.\n"
                f"   - Use `fabrication_mode='portrait'` STRICTLY for UI dialogue busts in `assets/portraits/` (NEVER use portraits as in-world sprites).\n"
                f"8. 1:1 VISUAL-TO-PHYSICAL COLLIDER CONTAINMENT: Track 'R1' MUST mandate that physical `StaticBody2D` / `CollisionPolygon2D` boundaries align 1:1 with the visual walkable terrain edges so the player character can NEVER walk into sky backdrops, across unbridged chasms, or off walkable shelves.\n"
                f"9. ABSOLUTE BAN ON PRE-SOLVED t=0 LEVELS & FAKE SIMULATIONS: Every mechanic and stage MUST initialize in an UNSOLVED state at t=0 (zero pre-completed paths or self-winning demo loops). Simulation propagation (e.g. fluid, energy, signals, logistics) MUST require true graph/source connectivity from a source to a target, and levels must feature genuine multi-stage obstacles and progression depth.\n"
                f"10. ATOMIC TDD & MULTI-TIER VISUAL RENDER VERIFICATION: Structure each track's acceptance criteria so they can be decomposed into atomic work units tested for (a) unsolved passive baseline, (b) negative counterfactual & spatial boundary containment, (c) positive causal state transition, and (d) rendered visual, animation strip, and collision-overlay alignment.\n"
            )

            contents_payload: Any = (mockup_image_parts + [prompt]) if mockup_image_parts else prompt
            response = call_gemini_with_transient_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents_payload,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GameRequirementsSpec,
                        temperature=0.4,
                        thinking_config=bounded_thinking_config(2048),
                    ),
                ),
                label="GameConceptDirectorAgent.generate_requirements_spec",
            )
            if not response.text:
                raise StudioAgentError("Empty response returned from Gemini requirements authoring call")
            data = json.loads(response.text)
            spec = GameRequirementsSpec(**data)
            if game_bible is not None and getattr(game_bible, "game_title", None) and game_bible.game_title != "Untitled Concept":
                spec.game_title = game_bible.game_title
            # Post-validation to enforce GDScript 4 target_modules (Milestone 7 Godot 4 Transition)
            for idx_r, req in enumerate(spec.requirements):
                sanitized_mods = []
                for mod in getattr(req, "target_modules", []):
                    mod_gd = mod.replace(".py", ".gd")
                    if not mod_gd.endswith(".gd"):
                        mod_gd = f"{mod_gd}.gd"
                    sanitized_mods.append(mod_gd)
                if idx_r == 0:
                    for core_m in ("scenes.gd", "entities.gd", "mechanics.gd", "main.gd"):
                        if core_m not in sanitized_mods:
                            sanitized_mods.append(core_m)
                req.target_modules = sanitized_mods or ["mechanics.gd"]

            # Ensure baseline concept-derived assets from GameBibleSpec are always included in asset_manifest
            if game_bible is not None:
                try:
                    from engine.studio.asset_library import CartridgeAssetLibrary
                    temp_lib = CartridgeAssetLibrary(cartridge_dir=candidate_dir or Path("/tmp/studio_asset_plan"))
                    baseline_reqs = temp_lib.derive_baseline_assets_from_bible(game_bible)
                    existing_ids = {a.asset_id for a in spec.asset_manifest}
                    for breq in baseline_reqs:
                        if breq.asset_id not in existing_ids:
                            spec.asset_manifest.append(breq)
                            existing_ids.add(breq.asset_id)
                except Exception as exc:
                    logger.debug("Baseline asset manifest derivation warning: %s", exc)
            return spec
        except Exception as exc:
            logger.error("Gemini requirements authoring failed: %s", exc)
            raise StudioAgentError(f"GameConceptDirectorAgent failed to author requirements spec: {exc}") from exc


CreativeDirectorWorldbuilderAgent = GameConceptDirectorAgent
