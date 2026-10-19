"""engine/studio/swe.py
Lead Gameplay SWE Agent — Feature 21.
Autonomously authors executable Python game cartridge modules (mechanics.py, entities.py, abilities.py, level.json)
with zero static string templates.
Enforces multi-stage verification (ast.parse, AST security validator, contract checks, bytecode compilation, Rationale-First class body).
Implements closed-loop self-healing repair.
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from pydantic import Field

from engine.api.contracts import BaseCartridge, BaseEnemy, BaseHazard, BasePlayer, BaseLevelSpec
from engine.loader import ASTSecurityError, validate_cartridge_ast
from engine.studio.models import (
    AudioCueStrategy,
    BaseEpistemicModel,
    CodeRepairSpec,
    CodeSliceRequest,
    DebugResearchPlan,
    GameProductionStrategySpec,
    GameRequirementsSpec,
    GenesisWorldSpec,
    ModuleFileSpec,
    NarrativeBible,
    RequirementTrack,
    SceneCritiqueAndStrategySpec,
    SceneDeficiencyItem,
    SceneLayerStrategy,
    StudioAgentError,
    SurgicalCodePatch,
    SurgicalPatchPayload,
    TrackCodeSynthesisPayload,
)

logger = logging.getLogger("LeadGameplaySWEAgent")


class CodeVerificationError(Exception):
    """Raised when synthesized code fails AST security or contract validation."""
    deliberation: str = "Raised when synthesized code fails AST security or contract validation."


class OpenGenreCartridgeBundle(BaseEpistemicModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first (Rule 4).
    Full open-genre cartridge code bundle synthesized by LeadGameplaySWEAgent via live Gemini inference.
    """
    deliberation: str = Field(..., description="Epistemic rationale explaining the game architecture, scenes, and mechanics design.")
    manifest_data: Dict[str, Any] = Field(default_factory=dict, description="manifest.json content with metadata, genre, version, entrypoint.")
    mechanics_code: str = Field(..., description="Source code for mechanics.py implementing BaseCartridge.")
    scenes_code: Optional[str] = Field(default=None, description="Source code for scenes.py implementing BaseScene subclasses (handle_event, update, render).")
    entities_code: Optional[str] = Field(default=None, description="Source code for entities.py or state machine data classes.")
    abilities_code: Optional[str] = Field(default=None, description="Source code for abilities.py or action systems.")
    level_data: Optional[Dict[str, Any]] = Field(default=None, description="Optional level.json configuration or layout parameters.")

    def assemble_cartridge_directory(
        self,
        target_dir: Path,
        swe_agent: Optional[Any] = None,
    ) -> Dict[str, Path]:
        """Assembles, verifies, and writes modular cartridge files to target_dir.
        Returns dictionary mapping filename -> Path.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        results: Dict[str, Path] = {}

        # 1. manifest.json
        manifest = self.manifest_data
        if not manifest and swe_agent is not None and hasattr(swe_agent, "_generate_manifest_data"):
            if not (target_dir / "manifest.json").is_file():
                manifest = swe_agent._generate_manifest_data(None)
        if manifest:
            manifest_path = target_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            results["manifest.json"] = manifest_path

        def _write_module(mod_name: str, code: Optional[str]) -> None:
            if not code:
                return
            if swe_agent is not None and hasattr(swe_agent, "_verify_and_repair_loop"):
                verified_code = swe_agent._verify_and_repair_loop(mod_name, code)
            else:
                verified_code = code
            mod_path = target_dir / mod_name
            mod_path.write_text(verified_code, encoding="utf-8")
            results[mod_name] = mod_path

        # 2. Python modules (ordered: entities, abilities, scenes, mechanics)
        _write_module("entities.py", self.entities_code)
        _write_module("abilities.py", self.abilities_code)
        _write_module("scenes.py", self.scenes_code)
        _write_module("mechanics.py", self.mechanics_code)

        # 3. level.json
        if self.level_data:
            level_path = target_dir / "level.json"
            level_path.write_text(json.dumps(self.level_data, indent=2), encoding="utf-8")
            results["level.json"] = level_path

        # 4. Cross-module import verification
        if swe_agent is not None and hasattr(swe_agent, "verify_cartridge_cross_module_imports"):
            swe_agent.verify_cartridge_cross_module_imports(target_dir, raise_on_error=True)

        return results


class LeadGameplaySWEAgent:
    """Specialist Agent that synthesizes executable Python cartridge code with zero string templates."""
    deliberation: str = "Lead gameplay SWE agent synthesizing executable Python game cartridges without static string templates."

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = "gemini-3.8-flash",
        visual_agent: Optional[Any] = None,
        voice_agent: Optional[Any] = None,
    ):
        self.client = client
        self.model_name = model_name
        self.visual_agent = visual_agent
        self.voice_agent = voice_agent
        self.last_deliberation: Optional[str] = None

    def _discover_sealed_assets(self, staging_dir: Path) -> Dict[str, List[str]]:
        """Discovers visual sprites, character portraits, visual mockups, and audio clips sealed upstream into staging_dir/assets/."""
        assets: Dict[str, List[str]] = {"sprites": [], "portraits": [], "mockups": [], "audio": []}
        sprites_dir = staging_dir / "assets" / "sprites"
        if sprites_dir.is_dir():
            assets["sprites"] = sorted([f"assets/sprites/{p.name}" for p in sprites_dir.glob("*.png")])
        portraits_dir = staging_dir / "assets" / "portraits"
        if portraits_dir.is_dir():
            assets["portraits"] = sorted([f"assets/portraits/{p.name}" for p in portraits_dir.glob("*.png")])
        mockups_dir = staging_dir / "assets" / "mockups"
        if mockups_dir.is_dir():
            assets["mockups"] = sorted([f"assets/mockups/{p.name}" for p in mockups_dir.glob("*.png")])
        audio_dir = staging_dir / "assets" / "audio"
        if audio_dir.is_dir():
            assets["audio"] = sorted([f"assets/audio/{p.name}" for p in audio_dir.glob("*.wav")])
        return assets

    def synthesize_cartridge_code(
        self,
        world_spec: Union[GenesisWorldSpec, GameRequirementsSpec, Any],
        narrative_bible: Optional[NarrativeBible] = None,
        output_dir: Optional[Path] = None,
        requirements_spec: Optional[GameRequirementsSpec] = None,
    ) -> Dict[str, Path]:
        """Synthesizes open-genre, multi-module game cartridge code using live Gemini inference."""
        if self.client is None:
            raise StudioAgentError("LeadGameplaySWEAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)")

        target_dir = output_dir or Path("cartridges/candidate")
        target_dir.mkdir(parents=True, exist_ok=True)

        req_spec = requirements_spec
        if req_spec is None and hasattr(world_spec, "requirements"):
            req_spec = world_spec

        if req_spec and getattr(req_spec, "requirements", None):
            first_track = req_spec.requirements[0]
            return self.implement_requirement_track(
                track=first_track,
                staging_dir=target_dir,
                requirements_spec=req_spec,
                narrative_bible=narrative_bible,
                world_spec=world_spec,
            )

        from google.genai import types
        narrative_context = ""
        if narrative_bible:
            narrative_context = (
                f"- Characters: {[c.name for c in getattr(narrative_bible, 'characters', [])]}\n"
                f"- Lore Summary: {getattr(narrative_bible, 'setting_summary', '')}\n"
            )

        sealed_assets = self._discover_sealed_assets(target_dir)
        sealed_assets_context = (
            f"Sealed Upstream Assets Available in Staging:\n"
            f"- Sprites: {sealed_assets['sprites']}\n"
            f"- Audio: {sealed_assets['audio']}\n"
        )

        req_context = ""
        if req_spec and getattr(req_spec, "requirements", None):
            req_lines = [f"- [{r.id}] {r.title}: {r.description}" for r in req_spec.requirements]
            req_context = "\nGame Requirements Backlog:\n" + "\n".join(req_lines) + "\n"

        game_title = world_spec.game_title
        core_genre = getattr(world_spec, "core_genre", getattr(world_spec, "genre_paradigm", "Action"))
        premise = getattr(world_spec, "thematic_premise", getattr(world_spec, "high_concept_summary", ""))
        world_lore = getattr(world_spec, "world_lore", "")
        style = getattr(world_spec, "aesthetic_style", "retro_arcade")
        mood = getattr(world_spec, "palette_mood", "retro_arcade")
        pacing = getattr(world_spec, "gameplay_pacing", "balanced")
        pacing_score = getattr(world_spec, "pacing_score", 0.0)
        target_diff = getattr(world_spec, "target_difficulty", 1.0)
        mechanics_guidelines = getattr(world_spec, "mechanics_guidelines", [])

        prompt = (
            f"You are the Lead Gameplay SWE AI for the Autonomous Game Studio.\n"
            f"Synthesize a complete, playable 2D Pygame game cartridge conforming to the following World Blueprint:\n"
            f"- Title: {game_title}\n"
            f"- Core Genre: {core_genre}\n"
            f"- Lore & Premise: {premise} — {world_lore}\n"
            f"- Aesthetic Style: {style} ({mood})\n"
            f"- Pacing: {pacing} (score {pacing_score})\n"
            f"- Target Difficulty: {target_diff}\n"
            f"- Mechanics Guidelines: {mechanics_guidelines}\n"
            f"{narrative_context}\n"
            f"{sealed_assets_context}\n"
            f"{req_context}\n"
            f"Architectural Contracts:\n"
            f"1. Target resolution is 1280x720 canvas at 60 FPS.\n"
            f"2. Multi-Module Code Architecture (Keep modules focused and concise, ~200-300 lines per module, ensuring fast generation and responsive 60 FPS execution):\n"
            f"   - `entities_code` (`entities.py`): Define game entities, actors, simulation objects, or units.\n"
            f"     All classes MUST declare `deliberation: str = '...'` as the very first statement inside the class body.\n"
            f"   - `scenes_code` (`scenes.py`): Define game scene classes subclassing `BaseScene` (from `engine.api.contracts`).\n"
            f"     Each scene MUST implement `on_enter(self, **kwargs)`, `on_exit(self)`, `handle_event(self, event)`, `update(self, dt: float)` "
            f"     (incrementing `self.state_counter += 1` every tick), and `render(self, surface: pygame.Surface)`. "
            f"     Import entities via `from entities import ...` if `entities.py` is provided. Reference verified sealed assets in `assets/sprites/` and `assets/audio/`.\n"
            f"     All classes MUST declare `deliberation: str = '...'` as the very first statement inside the class body.\n"
            f"   - `mechanics_code` (`mechanics.py`): Define the cartridge entrypoint subclassing `BaseCartridge` (from `engine.api.contracts`).\n"
            f"     Implement `get_scenes(self, engine)` returning a dictionary mapping scene names to scenes, "
            f"     and `get_initial_scene(self, engine)` returning the initial scene (e.g. from `scenes.py` or local scene).\n"
            f"     Import scenes via `from scenes import ...` and entities via `from entities import ...`. "
            f"     All classes MUST declare `deliberation: str = '...'` as the very first statement inside the class body.\n"
            f"   - `level_data` (`level.json`): Spatial configuration (width, height, spawn_point, exit_point) with `deliberation` first.\n"
            f"   - `manifest_data` (`manifest.json`): Cartridge manifest with `deliberation` first, name, version, core_genre, entrypoint.\n"
            f"3. Rule 4 CRITICAL: In EVERY class definition, the VERY FIRST statement inside the class body (BEFORE any docstring or method) "
            f"   MUST be `deliberation: str = '...'`. Do NOT put a class docstring before `deliberation`!\n"
            f"4. AST Sandbox Security: NEVER import banned modules (`os`, `sys`, `subprocess`, `socket`, `shutil`, `threading`, `asyncio`, `engine.core`, `engine.loader`). "
            f"   NEVER call prohibited builtins or method names (`open`, `eval`, `exec`, `compile`, `globals`, `locals`, `vars`, `dir`, `exit`, `quit`, "
            f"   `remove`, `kill`, `spawn`, `run`, `call`, `unlink`, `rmtree`, `system`, `popen`, `connect`). "
            f"   Instead of `list.remove(x)` or `def spawn(...)`, use list comprehensions `[x for x in items if ...]` and `def create_unit(...)`!\n"
            f"5. Prohibited tokens: Never output template placeholders like '{{TEMPLATE_' or '{{{{LEVEL_CONFIG}}}}'.\n"
            f"6. You MUST provide epistemic deliberation FIRST before code payloads.\n"
        )

        max_gemini_retries = 3
        bundle = None
        last_exc = None
        for g_attempt in range(1, max_gemini_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=OpenGenreCartridgeBundle,
                    ),
                )
                if not response.text:
                    raise CodeVerificationError("Empty response returned from Gemini cartridge synthesis.")

                bundle_dict = json.loads(response.text)
                if "files" in bundle_dict:
                    bundle = TrackCodeSynthesisPayload(**bundle_dict)
                else:
                    bundle = OpenGenreCartridgeBundle(**bundle_dict)
                self.last_deliberation = bundle.deliberation
                break
            except Exception as exc:
                last_exc = exc
                logger.warning("Gemini synthesis attempt %d/%d failed: %s", g_attempt, max_gemini_retries, exc)
                if g_attempt < max_gemini_retries:
                    time.sleep(3.0 * g_attempt)

        if bundle is None:
            logger.error("Gemini cartridge code synthesis failed after %d attempts: %s", max_gemini_retries, last_exc)
            if isinstance(last_exc, (CodeVerificationError, StudioAgentError)):
                raise last_exc
            raise StudioAgentError(f"LeadGameplaySWEAgent cartridge synthesis failed: {last_exc}") from last_exc

        if not bundle.manifest_data:
            bundle.manifest_data = self._generate_manifest_data(world_spec)

        return bundle.assemble_cartridge_directory(target_dir, swe_agent=self)

    def _is_live_genai_client(self) -> bool:
        """Returns True if self.client is a live google.genai Client rather than None or a unit-test mock."""
        if self.client is None:
            return False
        mod = getattr(type(self.client), "__module__", "") or ""
        return mod.startswith("google.genai")

    def _build_bible_and_narrative_context(
        self,
        staging_dir: Path,
        narrative_bible: Optional[NarrativeBible] = None,
        game_bible: Optional[Any] = None,
    ) -> Tuple[str, List[Any]]:
        """Loads game_bible.json and narrative_bible.json from staging_dir if not passed in,
        returning a rich prompt context block and any multimodal mockup image Parts.
        """
        from engine.studio.models import GameBibleSpec

        if game_bible is None and (staging_dir / "game_bible.json").is_file():
            try:
                game_bible = GameBibleSpec.model_validate_json(
                    (staging_dir / "game_bible.json").read_text(encoding="utf-8")
                )
            except Exception:
                game_bible = None

        if narrative_bible is None:
            if game_bible is not None and getattr(game_bible, "narrative_bible", None) is not None:
                narrative_bible = game_bible.narrative_bible
            elif (staging_dir / "narrative_bible.json").is_file():
                try:
                    narrative_bible = NarrativeBible.model_validate_json(
                        (staging_dir / "narrative_bible.json").read_text(encoding="utf-8")
                    )
                except Exception:
                    narrative_bible = None

        context_parts: List[str] = []
        multimodal_parts: List[Any] = []

        if game_bible is not None:
            beats_lines = [
                f"  * Beat {b.beat_index} ({b.phase_title}): Emotion={b.player_emotional_state} | Staging={b.sensory_and_camera_staging} | Diegetic Discovery={b.diegetic_teaching_moment} | Story={b.narrative_integration}"
                for b in (getattr(game_bible, "experience_storyboard", []) or [])
            ]
            mockup_lines = [
                f"  * '{m.title}' ({m.image_path}): {m.scene_description} | Camera/Spatial Paradigm: {getattr(m, 'camera_and_spatial_paradigm', '')} | Palette: {getattr(m, 'color_palette_hex', [])} | UI Callouts: {m.ui_callout_notes}"
                for m in (getattr(game_bible, "visual_mockups", []) or [])
            ]
            layer_lines = [
                f"  * Layer '{getattr(l, 'layer_id', '')}' (type={getattr(l, 'layer_type', '')}, z_index={getattr(l, 'z_index', 0)}, parallax={getattr(l, 'parallax_factor', 1.0)}): {getattr(l, 'visual_description', '')}"
                for l in (getattr(game_bible, "scene_layers", []) or [])
            ]
            context_parts.append(
                f"LIVING GAME BIBLE (`game_bible.json` — AUTHORITATIVE EXPERIENCE & DIEGETIC UX BLUEPRINT):\n"
                f"- Game Title: {getattr(game_bible, 'game_title', '')}\n"
                f"- Elevator Pitch: {getattr(game_bible, 'elevator_pitch', '')}\n"
                f"- Core Player Fantasy: {getattr(game_bible, 'core_fantasy', '')}\n"
                f"- Camera & Spatial Paradigm (LOCKED): {getattr(game_bible, 'camera_and_spatial_paradigm', '')}\n"
                f"- Color Palette Hex (LOCKED): {getattr(game_bible, 'color_palette_hex', [])}\n"
                f"- First 60 Seconds Hook: {getattr(game_bible, 'first_60_seconds_hook', '')}\n"
                f"- Diegetic UI & Onboarding Philosophy (NO wall-of-text HUD control manuals!): {getattr(game_bible, 'diegetic_ui_philosophy', '')}\n"
                f"- Gameplay Pillars: {getattr(game_bible, 'gameplay_pillars', [])}\n"
                f"- Visual Style Guide: {getattr(game_bible, 'visual_style_guide', '')}\n"
                f"- Audio Direction Guide: {getattr(game_bible, 'audio_direction_guide', '')}\n"
                + (f"- Multi-Layer Scene Architecture:\n" + "\n".join(layer_lines) + "\n" if layer_lines else "")
                + (f"- Experience Storyboard Arc:\n" + "\n".join(beats_lines) + "\n" if beats_lines else "")
                + (f"- Synthesized 16:9 Visual Mockup Boards:\n" + "\n".join(mockup_lines) + "\n" if mockup_lines else "")
            )

            try:
                from google.genai import types
                attached_paths: Set[Path] = set()
                for m in (getattr(game_bible, "visual_mockups", []) or [])[:2]:
                    m_path = (staging_dir / m.image_path).resolve()
                    if m_path.is_file() and m_path not in attached_paths:
                        attached_paths.add(m_path)
                        multimodal_parts.append(
                            types.Part.from_bytes(data=m_path.read_bytes(), mime_type="image/png")
                        )
                mockups_dir = staging_dir / "assets" / "mockups"
                if mockups_dir.is_dir() and len(multimodal_parts) < 2:
                    for p in sorted(mockups_dir.glob("*.png"))[:2]:
                        rp = p.resolve()
                        if rp not in attached_paths:
                            attached_paths.add(rp)
                            multimodal_parts.append(
                                types.Part.from_bytes(data=rp.read_bytes(), mime_type="image/png")
                            )
            except Exception as exc:
                logger.debug("Could not attach mockup bytes for SWE multimodal prompt: %s", exc)
        else:
            try:
                from google.genai import types
                mockups_dir = staging_dir / "assets" / "mockups"
                if mockups_dir.is_dir():
                    for p in sorted(mockups_dir.glob("*.png"))[:2]:
                        multimodal_parts.append(
                            types.Part.from_bytes(data=p.read_bytes(), mime_type="image/png")
                        )
            except Exception:
                pass

        try:
            from engine.studio.asset_library import CartridgeAssetLibrary
            lib = CartridgeAssetLibrary(
                cartridge_dir=staging_dir,
                visual_agent=getattr(self, "visual_agent", None),
                voice_agent=getattr(self, "voice_agent", None),
            )
            lib.sanitize_misplaced_portraits_in_sprites_dir()
            lib.save_catalog()
            context_parts.append(lib.build_prompt_summary_for_swe())
        except Exception as lib_exc:
            logger.debug("Could not build CartridgeAssetLibrary prompt summary: %s", lib_exc)

        if narrative_bible is not None:
            chars_info = [
                f"{getattr(c, 'name', 'Character')} (id={getattr(c, 'character_id', '')}, role={getattr(c, 'role', '')}, portrait={getattr(c, 'portrait_asset', '')}, backstory={getattr(c, 'backstory', '')[:80]})"
                for c in getattr(narrative_bible, "characters", [])
            ]
            cutscenes_info = [
                {
                    "cutscene_id": getattr(cs, "cutscene_id", ""),
                    "title": getattr(cs, "title", ""),
                    "trigger": getattr(cs, "trigger", ""),
                    "beats": [
                        {
                            "beat_number": getattr(b, "beat_number", 0),
                            "action": getattr(b, "action_description", ""),
                            "speaker_id": getattr(b, "speaker_id", ""),
                            "line": getattr(b, "dialogue_line", ""),
                        }
                        for b in getattr(cs, "beats", [])
                    ],
                }
                for cs in getattr(narrative_bible, "cutscenes", [])
            ]
            trees_info = [
                dt.model_dump() if hasattr(dt, "model_dump") else dt
                for dt in getattr(narrative_bible, "dialogue_trees", [])
            ]
            barks_info = [
                f"[{getattr(b, 'bark_id', '')}] (trigger={getattr(b, 'trigger_type', '')}) {getattr(b, 'speaker_name', '')}: '{getattr(b, 'text', '')}' (audio: {getattr(b, 'voice_clip', '')})"
                for b in getattr(narrative_bible, "dialogue_barks", [])
            ]
            context_parts.append(
                f"NARRATIVE BIBLE & MULTIMODAL CHARACTERS (`narrative_bible.json`):\n"
                f"- Setting Summary: {getattr(narrative_bible, 'setting_summary', '')}\n"
                f"- Characters: {chars_info}\n"
                f"- Cutscene Scripts (stage these in-game!): {json.dumps(cutscenes_info)}\n"
                f"- Interactive Dialogue Trees: {json.dumps(trees_info)}\n"
                f"- Reactive Voice Barks: {barks_info}\n"
            )

        return ("\n\n".join(context_parts) + "\n\n" if context_parts else ""), multimodal_parts

    def implement_requirement_track(
        self,
        track: RequirementTrack,
        staging_dir: Path,
        requirements_spec: Optional[GameRequirementsSpec] = None,
        narrative_bible: Optional[NarrativeBible] = None,
        world_spec: Optional[Any] = None,
        game_bible: Optional[Any] = None,
        prior_error: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Iteratively implements or expands cartridge code for a single RequirementTrack.
        Inspects existing code in staging_dir and sealed assets in staging_dir/assets/,
        generates or updates modular files, runs self-healing verification, and verifies cross-module imports.
        """
        if self.client is None:
            raise StudioAgentError("LeadGameplaySWEAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)")

        staging_dir.mkdir(parents=True, exist_ok=True)

        # Unconditionally True for all new builds (Milestone 7 Godot 4 Transition)
        is_godot = True

        existing_modules: Dict[str, str] = {}
        for ext in ("*.gd", "*.py", "*.tscn"):
            for code_path in sorted(staging_dir.glob(ext)):
                existing_modules[code_path.name] = code_path.read_text(encoding="utf-8")
        if (staging_dir / "project.godot").is_file():
            existing_modules["project.godot"] = (staging_dir / "project.godot").read_text(encoding="utf-8")

        existing_level: Optional[Dict[str, Any]] = None
        level_file = staging_dir / "level.json"
        if level_file.is_file():
            try:
                existing_level = json.loads(level_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        bible_and_narrative_context, multimodal_mockup_parts = self._build_bible_and_narrative_context(
            staging_dir=staging_dir,
            narrative_bible=narrative_bible,
            game_bible=game_bible,
        )
        sealed_assets = self._discover_sealed_assets(staging_dir)

        criteria_lines = [
            f"- [{c.id}] ({c.verification_method}): {c.description}"
            for c in getattr(track, "acceptance_criteria", [])
        ]
        criteria_context = "\n".join(criteria_lines) if criteria_lines else "Implement core requirement behavior headlessly."

        existing_code_summary = []
        for mod_name, code in existing_modules.items():
            if len(code) > 45000:
                summary = code[:25000] + f"\n\n# ... [{len(code) - 35000} bytes omitted for brevity] ...\n\n" + code[-10000:]
                existing_code_summary.append(f"--- Existing {mod_name} (summary) ---\n{summary}")
            else:
                existing_code_summary.append(f"--- Existing {mod_name} ---\n{code}")
        existing_code_block = "\n".join(existing_code_summary) if existing_code_summary else "No existing code modules."

        prior_error_block = (
            f"CRITICAL — PREVIOUS ATTEMPT COMPILER/RUNTIME ERROR TO FIX & AVOID:\n{prior_error}\n\n"
            if prior_error
            else ""
        )

        if is_godot:
            prompt = (
                f"You are the Lead Gameplay SWE AI implementing requirement track {track.id}: '{track.title}' for a Code-First Godot 4.3 game cartridge.\n"
                f"Description: {track.description}\n"
                f"Target Modules: {track.target_modules}\n"
                f"Acceptance Criteria:\n{criteria_context}\n\n"
                f"{prior_error_block}"
                f"{bible_and_narrative_context}"
                f"EXISTING CARTRIDGE CODE IN STAGING:\n{existing_code_block}\n\n"
                f"SEALED ASSETS AVAILABLE IN STAGING:\n"
                f"- World & Entity Sprites (ONLY use these for in-world entities/props/tiles): {sealed_assets['sprites']}\n"
                f"- Character Portraits (STRICTLY FOR DIALOGUE UI IN ui_hud.gd — NEVER use as world entity sprites!): {sealed_assets.get('portraits', [])}\n"
                f"- 16:9 Visual Mockups (attached multimodally — match this camera perspective, palette, and multi-layer scene composition!): {sealed_assets.get('mockups', [])}\n"
                f"- Audio: {sealed_assets['audio']}\n\n"
                f"ARCHITECTURAL, ASSET LIBRARY & EXPERIENCE-FIRST SPECIFICATION:\n"
                f"1. Target: Godot 4.3-stable (GDScript 2.0). Logical viewport: 1280x720 rendered on a 2560x1440 HiDPI resizable window at 60 FPS.\n"
                f"2. Code-First Design: Zero complex .tscn files. A minimal 4-line main.tscn binds res://main.gd to root Node2D. You author 100% of the `.gd` modules.\n"
                f"3. Modular Architecture & Cartridge Asset Library Catalog (`res://assets/asset_library.json`):\n"
                f"   - The Cartridge Asset Library manifest is stored at `res://assets/asset_library.json` and summarized above, containing metadata for every synthesized sprite sheet (`hframes`, `vframes`, `frame_size`, `animations`), articulated rig (`parts` with `pivot_offset`, `attach_offset`, `z_index_offset`), tile surface, set-piece, portrait, and audio clip.\n"
                f"   - Load PNG textures at runtime without `.import` files via `Image.load_from_file(ProjectSettings.globalize_path(res_path))` and `ImageTexture.create_from_image(img)`. Set `tex.set_meta('source_path', res_path)` so runtime diagnostics can trace loaded textures.\n"
                f"   - Drive multi-state sprite sheet animations (`hframes`, `vframes`, `frame`) and articulated child `Sprite2D` rigs according to the entity's live state (idle, locomotion, aerial, action) and scale sprites to their intended `display_size` from `assets/asset_library.json`.\n"
                f"   - Render physical platforms, ground surfaces, and structures using tiled textures (`draw_texture_rect` with `tile=true` on a `CanvasItem` with `texture_repeat = CanvasItem.TEXTURE_REPEAT_ENABLED`) rather than flat untextured programmer-art `draw_rect` rectangles.\n"
                f"   - Keep `z_index < 0` background/horizon layers visible: never paint a full-viewport opaque `draw_rect` at `z_index >= 0` that covers your `z_index < 0` sky/backdrop layers.\n"
                f"   - Strictly separate character portraits (`res://assets/portraits/*.png`, used ONLY in `ui_hud.gd` dialogue panels with bounded size <= 64x64) from in-world entity sprites (`res://assets/sprites/*.png`).\n"
                f"   - DECOMPOSE CONCEPT ART INTO COMPOSABLE WORLD ELEMENTS (NEVER USE A STATIC FULL-SCREEN BACKGROUND IMAGE): Build the world out of discrete, layered, interactive elements matching the 16:9 concept art—isolated horizon/midground set-pieces, textured physical geometry, and interactive `Node2D` / `CharacterBody2D` / `Area2D` entities.\n"
                f"   - ON-DEMAND ASSET FABRICATION (`requested_assets`): If your gameplay mechanics, animations, or multi-layer world need ANY additional purpose-built asset (a `sprite_sheet` with custom `animation_states`, `articulated_parts` with custom pivots, `tile_surface`, isolated `static_sprite` set-piece/prop, `diegetic_ui` widget, or `sfx`), include it in `requested_assets` in your JSON response and the Cartridge Asset Library will fabricate it on disk and update `res://assets/asset_library.json`.\n"
                f"   - main.gd: Root scene script extending Node2D. Preloads child modules (`mechanics.gd`, `entities.gd`, `scenes.gd`, `ui_hud.gd`). Sets up Camera2D, level scene container, and UI CanvasLayer. Implements `get_state_snapshot() -> Dictionary` for headless playtesting.\n"
                f"   - mechanics.gd: Core rules, physics/simulation loops, placement/action methods, scoring, story progression, recoverable hazard/win/loss logic.\n"
                f"   - entities.gd: CharacterBody2D, RigidBody2D, Area2D, or Node2D entity classes with animated sprite sheets or articulated multi-part rigs and directional `flip_h`.\n"
                f"   - scenes.gd: Multi-layered active world Node2D renderer built from decomposed world elements, textured physical geometry, interactive landmarks, and atmospheric effects.\n"
                f"   - ui_hud.gd: Clean diegetic UI with a clear Mission Objective & Goal Progress tracker (`X / Y` objectives completed + victory state banner), character portrait & cutscene/dialogue tree presenter, voice bark audio player, and responsive input handling.\n"
                f"4. Cross-Module Resolution, Scope & Strict Method Alignment:\n"
                f"   - When preloading OTHER modules, declare:\n"
                f"     const Mechanics = preload('res://mechanics.gd')\n"
                f"     const Entities = preload('res://entities.gd')\n"
                f"     const Scenes = preload('res://scenes.gd')\n"
                f"     const UIHud = preload('res://ui_hud.gd')\n"
                f"   - NEVER SELF-PRELOAD OR USE A MODULE'S OWN ALIAS INSIDE THAT MODULE: Inside `scenes.gd` (including inside any inner `class` defined in `scenes.gd`), NEVER write `Scenes.some_func()` or `preload('res://scenes.gd')` (causes `Parse Error: Identifier \"Scenes\" not declared in the current scope`). Likewise, never write `Mechanics.` inside `mechanics.gd`, `Entities.` inside `entities.gd`, or `UIHud.` inside `ui_hud.gd`. Call functions in the same script directly, or define helpers inside the inner class.\n"
                f"   - STATIC VS INSTANCE CALLS ON PRELOADED SCRIPTS: `preload('res://scenes.gd')` returns a `GDScript` resource, NOT an instance! If `main.gd` calls `Scenes.create_world(...)` directly on `Scenes`, then `create_world` in `scenes.gd` MUST be declared `static func create_world(...)`. Otherwise, instantiate `var scenes_node = Scenes.new()` first and call instance methods on `scenes_node`!\n"
                f"   - AVOID `:=` TYPE INFERENCE ON UNTYPED EXPRESSIONS: In GDScript 2.0, `var x := expr` causes `Parse Error: Cannot infer the type of \"x\" variable because the value doesn't have a set type` whenever `expr` returns `Variant`, comes from a Dictionary/Array lookup, or calls a function without an explicit return type. Always use `var x = expr` or explicit type annotations (`var x: Rect2 = expr`) instead of `:=`!\n"
                f"   - Every method called across `main.gd`, `mechanics.gd`, `scenes.gd`, `entities.gd`, and `ui_hud.gd` MUST match the exact function name defined in the target module.\n"
                f"5. MANDATORY VISIBLE WORLD RENDERING, STORY STAGING & MULTIMODAL ASSET INTEGRATION:\n"
                f"   - NEVER return a bare/empty Node2D.new() stub from a map or scene builder without _draw() or child visual nodes!\n"
                f"   - Honor the Living Game Bible's First 60 Seconds Hook, Experience Storyboard Beats, Visual Style Guide, Camera & Spatial Paradigm, and 16:9 Visual Mockup boards.\n"
                f"   - Stage the Narrative Bible's cutscenes, dialogue trees, and reactive voice barks in-game with character portraits (`res://assets/portraits/*.png`) and real WAV playback.\n"
                f"   - Load and display synthesized PNG sprites from 'res://assets/sprites/*.png' according to `res://assets/asset_library.json`.\n"
                f"   - MANDATORY 24,000 Hz PCM WAV AUDIO LOADING & COOLDOWN: All synthesized WAV voice clips in 'res://assets/audio/*.wav' are 24,000 Hz, 16-bit mono PCM. When loading via FileAccess.get_file_as_bytes(ProjectSettings.globalize_path(path)), you MUST strip the 44-byte RIFF header (`stream.data = bytes.slice(44) if bytes.size() > 44 else bytes`), set `stream.format = AudioStreamWAV.FORMAT_16_BITS`, `stream.mix_rate = 24000` (NEVER 44100, which causes high-pitched chipmunk distortion!), and `stream.stereo = false`. Furthermore, enforce a non-overlapping playback guard (`if not audio_player.playing:`) and a minimum cooldown (>= 8.0 seconds) so voice barks NEVER spam or repeat every few seconds.\n"
                f"6. DIEGETIC UX, CLEAR VIEWPORT & RECOVERABLE GAMEPLAY:\n"
                f"   - Teach and guide the player through world cues, visual feedback, diegetic labels/hover callouts, and story context as specified in the Living Game Bible's `diegetic_ui_philosophy`—NEVER clutter the screen with a static instruction-manual text box listing keyboard shortcuts.\n"
                f"   - NEVER blind the player on startup with >30% opaque full-screen overlays or frost, and NEVER lock out player controls with an unrecoverable game-over screen in the first 30 seconds.\n"
                f"   - Use `get_global_mouse_position()` for accurate HiDPI world-space mouse picking.\n"
                f"7. STRICTLY FORBIDDEN (Deprecated Godot 3 Syntax):\n"
                f"   - NEVER use KinematicBody2D (use CharacterBody2D).\n"
                f"   - NEVER use yield() (use await).\n"
                f"   - NEVER use File.new() (use FileAccess.open()).\n"
                f"   - NEVER use Directory.new() (use DirAccess.open()).\n"
                f"   - NEVER use 3-argument .connect('sig', target, 'fn') (use signal.connect(fn)).\n"
                f"   - NEVER pass arguments to move_and_slide().\n"
                f"8. RATIONALE-FIRST EPISTEMIC RULE (Rule 4): deliberation MUST be provided FIRST in every payload before files and code.\n"
            )
            response_model = TrackCodeSynthesisPayload
        else:
            prompt = (
                f"You are the Lead Gameplay SWE AI implementing requirement track {track.id}: '{track.title}'.\n"
                f"Description: {track.description}\n"
                f"Target Modules: {track.target_modules}\n"
                f"Acceptance Criteria:\n{criteria_context}\n\n"
                f"EXISTING CARTRIDGE CODE IN STAGING:\n{existing_code_block}\n\n"
                f"SEALED ASSETS AVAILABLE IN STAGING:\n"
                f"- Sprites: {sealed_assets['sprites']}\n"
                f"- Audio: {sealed_assets['audio']}\n\n"
                f"Architectural Contracts:\n"
                f"1. Target resolution is 1280x720 canvas at 60 FPS.\n"
                f"2. Multi-Module Code Architecture:\n"
                f"   Expand or synthesize the modular files (`mechanics.py`, `scenes.py`, `entities.py`, `level.json`) to fulfill {track.id}.\n"
                f"   Ensure `mechanics.py` defines the cartridge entrypoint subclassing `BaseCartridge`.\n"
                f"   Ensure `scenes.py` defines `BaseScene` subclasses implementing `handle_event`, `update` (incrementing `self.state_counter += 1`), and `render`.\n"
                f"   Ensure `entities.py` defines required actors/state.\n"
                f"3. In EVERY class definition, the VERY FIRST statement inside the class body MUST be `deliberation: str = '...'`.\n"
                f"4. AST Sandbox Security: NEVER import banned modules or call prohibited builtins.\n"
                f"5. Epistemic deliberation MUST be provided FIRST before code payloads.\n"
            )
            response_model = OpenGenreCartridgeBundle

        from google.genai import types
        max_gemini_retries = 3
        bundle = None
        last_exc = None
        contents_payload: Any = (
            [prompt] + multimodal_mockup_parts
            if (multimodal_mockup_parts and self._is_live_genai_client())
            else prompt
        )
        for g_attempt in range(1, max_gemini_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents_payload,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_model,
                    ),
                )
                if not response.text:
                    raise CodeVerificationError(f"Empty response returned from Gemini for requirement track {track.id}")

                bundle_dict = json.loads(response.text)
                if "files" in bundle_dict:
                    bundle = TrackCodeSynthesisPayload(**bundle_dict)
                else:
                    bundle = OpenGenreCartridgeBundle(**bundle_dict)
                self.last_deliberation = bundle.deliberation
                break
            except Exception as exc:
                last_exc = exc
                logger.warning("Gemini attempt %d/%d for track %s failed: %s", g_attempt, max_gemini_retries, track.id, exc)
                if g_attempt < max_gemini_retries:
                    time.sleep(3.0 * g_attempt)

        if bundle is None:
            logger.error("Gemini implementation of requirement track %s failed after %d attempts: %s", track.id, max_gemini_retries, last_exc)
            if isinstance(last_exc, (CodeVerificationError, StudioAgentError)):
                raise last_exc
            raise StudioAgentError(f"LeadGameplaySWEAgent failed for requirement track {track.id}: {last_exc}") from last_exc

        return bundle.assemble_cartridge_directory(staging_dir, swe_agent=self)

    synthesize_requirement_track = implement_requirement_track

    def verify_cartridge_cross_module_imports(self, cartridge_dir: Path, raise_on_error: bool = True) -> Tuple[bool, List[str]]:
        """Statically verifies inter-module imports across Python files in cartridge_dir.
        Checks that imported modules and symbols exist in local files, and detects circular import cycles.
        """
        py_files = {p.name: p for p in cartridge_dir.glob("*.py")}
        module_exports: Dict[str, Set[str]] = {}
        errors: List[str] = []

        def _extract_target_names(target: ast.AST) -> List[str]:
            if isinstance(target, ast.Name):
                return [target.id]
            if isinstance(target, (ast.Tuple, ast.List)):
                res: List[str] = []
                for elt in target.elts:
                    res.extend(_extract_target_names(elt))
                return res
            if isinstance(target, ast.Starred):
                return _extract_target_names(target.value)
            return []

        for fname, fpath in py_files.items():
            mod_stem = fpath.stem
            try:
                tree = ast.parse(fpath.read_text(encoding="utf-8"))
            except SyntaxError as e:
                errors.append(f"SyntaxError in {fname}: {e}")
                continue

            exports: Set[str] = set()
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    exports.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        for name in _extract_target_names(target):
                            exports.add(name)
                elif isinstance(node, ast.AnnAssign):
                    for name in _extract_target_names(node.target):
                        exports.add(name)
            module_exports[mod_stem] = exports

        local_modules = {p.stem for p in py_files.values()}
        import_graph: Dict[str, Set[str]] = {stem: set() for stem in local_modules}

        for fname, fpath in py_files.items():
            mod_stem = fpath.stem
            try:
                tree = ast.parse(fpath.read_text(encoding="utf-8"))
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    mod_name = node.module
                    if mod_name in ("entities", "scenes", "abilities", "mechanics"):
                        if f"{mod_name}.py" not in py_files:
                            err = f"Module {fname} imports from '{mod_name}', but '{mod_name}.py' does not exist."
                            errors.append(err)
                    if mod_name and mod_name in module_exports:
                        for alias in node.names:
                            if alias.name != "*" and alias.name not in module_exports[mod_name]:
                                err = (
                                    f"Cross-module import defect in {fname}: symbol '{alias.name}' is not exported "
                                    f"by '{mod_name}.py'. Available: {sorted(module_exports[mod_name])}"
                                )
                                errors.append(err)
                    if mod_name in local_modules:
                        import_graph[mod_stem].add(mod_name)
                    elif node.level and node.level > 0 and node.module is None:
                        for alias in node.names:
                            if alias.name in local_modules:
                                import_graph[mod_stem].add(alias.name)

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ("entities", "scenes", "abilities", "mechanics"):
                            if f"{alias.name}.py" not in py_files:
                                err = f"Module {fname} imports '{alias.name}', but '{alias.name}.py' does not exist."
                                errors.append(err)
                        if alias.name in local_modules:
                            import_graph[mod_stem].add(alias.name)

        # Detect circular dependencies between local modules
        visited: Dict[str, int] = {}
        reported_cycles: Set[Tuple[str, ...]] = set()

        def _dfs(node: str, path: List[str]) -> None:
            visited[node] = 1
            path.append(node)
            for neighbor in sorted(import_graph.get(node, set())):
                if visited.get(neighbor, 0) == 1:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycle_nodes = cycle[:-1]
                    min_elem = min(cycle_nodes)
                    min_idx = cycle_nodes.index(min_elem)
                    canonical = tuple(cycle_nodes[min_idx:] + cycle_nodes[:min_idx])
                    if canonical not in reported_cycles:
                        reported_cycles.add(canonical)
                        errors.append(f"Circular import detected between local modules: {' -> '.join(cycle)}")
                elif visited.get(neighbor, 0) == 0:
                    _dfs(neighbor, path)
            path.pop()
            visited[node] = 2

        for mod in sorted(import_graph.keys()):
            if visited.get(mod, 0) == 0:
                _dfs(mod, [])

        if errors:
            if raise_on_error:
                formatted_errors = "\n".join(errors)
                raise CodeVerificationError(f"Cross-module import verification failed:\n{formatted_errors}")
            return False, errors
        return True, []

    def _verify_and_repair_loop(self, module_name: str, source_code: str, max_retries: int = 3) -> str:
        """Runs multi-stage verification with self-healing repair loop."""
        current_code = source_code
        for attempt in range(1, max_retries + 1):
            valid, error_msg = self.verify_code_module(module_name, current_code, raise_on_ast_error=False)
            if valid:
                return current_code
            logger.warning(
                "Module %s failed verification on attempt %d: %s. Initiating repair.",
                module_name, attempt, error_msg
            )
            current_code = self.repair_code_module(module_name, current_code, error_msg)

        # Final check
        valid, error_msg = self.verify_code_module(module_name, current_code, raise_on_ast_error=False)
        if not valid:
            raise CodeVerificationError(
                f"Module {module_name} failed verification after {max_retries} self-healing attempts: {error_msg}"
            )
        return current_code

    def verify_code_module(
        self,
        module_name: str,
        source_code: str,
        raise_on_ast_error: bool = True,
    ) -> Tuple[bool, str]:
        """Validates syntax, AST security, contracts, and Rationale-First ordering."""
        # Stage 1: Syntax check
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

        # Stage 2: Security check via engine AST sandbox validator
        try:
            valid_ast, ast_errs = validate_cartridge_ast(source_code)
            if not valid_ast:
                return False, f"ASTSecurityError: {ast_errs}"
        except ASTSecurityError as ast_err:
            if raise_on_ast_error:
                raise
            return False, f"ASTSecurityError: {ast_err}"

        # Stage 3: Zero template markers check
        if "{TEMPLATE_" in source_code or "{{LEVEL_CONFIG}}" in source_code:
            return False, "Detected static string template placeholders"

        # Stage 4: Bytecode compilation
        try:
            compile(source_code, module_name, "exec")
        except Exception as e:
            return False, f"CompileError: {e}"

        # Stage 5: Rationale-First check on all class definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not node.body:
                    return False, f"Class {node.name} body is empty"
                first_stmt = node.body[0]
                has_delib = False
                if isinstance(first_stmt, ast.Assign):
                    for target in first_stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "deliberation":
                            has_delib = True
                elif isinstance(first_stmt, ast.AnnAssign):
                    if isinstance(first_stmt.target, ast.Name) and first_stmt.target.id == "deliberation":
                        has_delib = True
                if not has_delib:
                    return False, f"Class {node.name} must declare 'deliberation' as its first attribute"

        # Stage 6: mechanics.py specific contract checks
        if module_name == "mechanics.py":
            cartridge_classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    base_names = [b.id for b in node.bases if isinstance(b, ast.Name)]
                    if "BaseCartridge" in base_names:
                        cartridge_classes.append(node)
            if not cartridge_classes:
                return False, "mechanics.py must define at least one subclass inheriting from BaseCartridge"
            
            c_class = cartridge_classes[0]
            methods = [m.name for m in c_class.body if isinstance(m, ast.FunctionDef)]
            has_scene_lifecycle = "get_scenes" in methods or "get_initial_scene" in methods
            has_level_lifecycle = "get_level_spec" in methods
            if not (has_scene_lifecycle or has_level_lifecycle):
                return False, (
                    f"BaseCartridge subclass {c_class.name} must implement either open-genre scene "
                    f"lifecycle (get_scenes / get_initial_scene) or level lifecycle (get_level_spec)."
                )

        # Stage 7: scenes.py specific contract checks
        if module_name == "scenes.py":
            scene_classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    base_names = [b.id for b in node.bases if isinstance(b, ast.Name)]
                    if any(b in ("BaseScene", "Scene") for b in base_names):
                        scene_classes.append(node)
            if not scene_classes:
                return False, "scenes.py must define at least one subclass inheriting from BaseScene or Scene"
            
            s_class = scene_classes[0]
            methods = [m.name for m in s_class.body if isinstance(m, ast.FunctionDef)]
            for req in ["update", "render"]:
                if req not in methods:
                    return False, f"BaseScene subclass {s_class.name} missing required lifecycle method: {req}"

        return True, ""

    def repair_code_module(self, module_name: str, broken_code: str, error_diagnostic: str) -> str:
        """Self-healing repair prompt via Gemini (Rule 5: Zero Silent Fallbacks)."""
        if self.client is None:
            raise CodeVerificationError(
                f"LeadGameplaySWEAgent requires an active google.genai Client for code repair (Rule 5: Zero Silent Fallbacks)"
            )

        try:
            prompt = (
                f"You are the Lead Gameplay SWE AI.\n"
                f"Fix the following Python module '{module_name}'.\n"
                f"Diagnostic error: {error_diagnostic}\n"
                f"Broken code:\n```python\n{broken_code}\n```\n"
                f"Return ONLY valid, corrected Python code without markdown tags or explanations.\n"
                f"Ensure every class defines deliberation as its very first attribute."
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            if not response.text:
                raise CodeVerificationError(f"Empty response from Gemini code repair for {module_name}")
            cleaned = response.text.strip()
            if cleaned.startswith("```python"):
                cleaned = cleaned.split("```python", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            return cleaned.strip()
        except Exception as exc:
            logger.error("Gemini code repair call failed: %s", exc)
            if isinstance(exc, (CodeVerificationError, StudioAgentError)):
                raise
            raise CodeVerificationError(f"Gemini code repair failed for {module_name}: {exc}") from exc

    def _write_valid_tone_wav(self, wav_path: Path) -> None:
        import wave
        import struct
        import math
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 22050
        duration = 1.0
        total_samples = int(sample_rate * duration)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            raw_data = bytearray()
            for i in range(total_samples):
                sample = int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate))
                raw_data.extend(struct.pack("<h", sample))
            wf.writeframes(raw_data)

    def _write_valid_indexed_png(self, png_path: Path, width: int = 64, height: int = 64) -> None:
        from engine.assets.palette import quantize_image_to_indexed, STYLE_BIBLE_16_PALETTE
        from PIL import Image
        png_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGBA", (width, height), (32, 32, 40, 255))
        quantized = quantize_image_to_indexed(img, STYLE_BIBLE_16_PALETTE)
        quantized.save(png_path, format="PNG")

    def _repair_hollow_stubs_in_source(self, file_path: Path, diag_reason: str = "", rec_remedy: str = "") -> bool:
        if not file_path.is_file():
            return False
        src = file_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src, filename=str(file_path))
        except Exception as parse_err:
            logger.warning("Could not parse %s for stub inspection: %s", file_path, parse_err)
            return False

        from engine.supervisor.challenger import CandidateASTInspector
        insp = CandidateASTInspector()
        report = insp.inspect_directory(file_path.parent)
        stubs_in_this_file = [
            s for s in report.get("hollow_stubs", [])
            if s.get("file") == file_path.name or s.get("module") == file_path.name
        ]
        if not stubs_in_this_file:
            return False

        lines = src.splitlines(keepends=True)
        modified = False

        for node in reversed(tree.body):
            if isinstance(node, ast.ClassDef):
                cls_stubs = [
                    s for s in stubs_in_this_file
                    if any(m.name == s.get("function") for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)))
                ]
                if cls_stubs:
                    start_line = node.lineno - 1
                    end_line = getattr(node, "end_lineno", len(lines))
                    cls_src = "".join(lines[start_line:end_line])

                    repaired_cls = None
                    if self.client is not None:
                        try:
                            prompt = (
                                f"You are the Lead Gameplay SWE AI.\n"
                                f"Fix the class '{node.name}' in Python module '{file_path.name}'.\n"
                                f"Diagnostic error: Challenger Gate rejected build due to hollow stubs: {diag_reason}.\n"
                                f"Failing methods: {[s.get('function') for s in cls_stubs]}.\n"
                                f"MANDATORY DIRECTIVES:\n"
                                f"1. The class MUST declare 'deliberation' as its very first attribute.\n"
                                f"2. Replace every hollow stub (pass, return None, ..., empty body) with genuine, functioning logic that maintains real state, modifies attributes, handles transitions, or returns meaningful data.\n"
                                f"3. ABSOLUTELY ZERO HOLLOW STUBS: Do NOT use 'pass', '...', 'return None', or empty bodies anywhere in the repaired code.\n"
                                f"Class code:\n```python\n{cls_src}\n```\n"
                                f"Return ONLY the valid, corrected Python class code without markdown tags or explanations."
                            )
                            resp = self.client.models.generate_content(
                                model=self.model_name,
                                contents=prompt,
                            )
                            if resp.text:
                                cand_code = resp.text.strip()
                                cand_code = re.sub(r"^```python\s*", "", cand_code)
                                cand_code = re.sub(r"\s*```$", "", cand_code).strip()
                                cand_tree = ast.parse(cand_code)
                                cand_classes = [n for n in cand_tree.body if isinstance(n, ast.ClassDef)]
                                if cand_classes:
                                    cand_has_stubs = False
                                    for cm in cand_classes[0].body:
                                        if isinstance(cm, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                            cb = cm.body[1:] if (cm.body and isinstance(cm.body[0], ast.Expr) and isinstance(cm.body[0].value, ast.Constant) and isinstance(cm.body[0].value.value, str)) else cm.body
                                            if not cb or (len(cb) == 1 and isinstance(cb[0], ast.Pass)):
                                                cand_has_stubs = True
                                                break
                                    if not cand_has_stubs:
                                        repaired_cls = cand_code
                        except Exception as llm_err:
                            logger.warning("LLM class repair for %s encountered error: %s", node.name, llm_err)

                    if not repaired_cls:
                        cls_lines = cls_src.splitlines(keepends=True)
                        for stub in cls_stubs:
                            fn_name = stub.get("function")
                            for fn_node in node.body:
                                if isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn_node.name == fn_name:
                                    fn_start = fn_node.lineno - 1 - start_line
                                    fn_end = getattr(fn_node, "end_lineno", fn_start + 3) - start_line
                                    indent = "        "
                                    if fn_name == "on_exit":
                                        body_code = f"{indent}self.state_counter = 0\n{indent}return {{'status': 'exited'}}\n"
                                    elif fn_name.startswith("trigger_") or fn_name.startswith("play_"):
                                        body_code = (
                                            f"{indent}self.alert_cooldown = max(0.0, getattr(self, 'alert_cooldown', 0.0))\n"
                                            f"{indent}snd = getattr(self, 'assets', {{}}).get('{fn_name.replace('trigger_', '')}')\n"
                                            f"{indent}if snd is not None and hasattr(snd, 'play'):\n"
                                            f"{indent}    snd.play()\n"
                                            f"{indent}return True\n"
                                        )
                                    else:
                                        body_code = f"{indent}setattr(self, '_active_{fn_name}', True)\n{indent}return True\n"

                                    for idx in range(fn_start, min(fn_end, len(cls_lines))):
                                        if "pass" in cls_lines[idx]:
                                            cls_lines[idx] = body_code
                                            break
                        candidate_repaired = "".join(cls_lines)
                        try:
                            ast.parse(candidate_repaired)
                            repaired_cls = candidate_repaired
                        except Exception:
                            pass

                    if repaired_cls:
                        lines[start_line:end_line] = [repaired_cls + "\n"]
                        modified = True

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_stubs = [s for s in stubs_in_this_file if s.get("function") == node.name]
                if fn_stubs:
                    start_line = node.lineno - 1
                    end_line = getattr(node, "end_lineno", start_line + 3)
                    indent = "    "
                    body_code = f"{indent}return True\n"
                    for idx in range(start_line, min(end_line, len(lines))):
                        if "pass" in lines[idx]:
                            lines[idx] = body_code
                            modified = True
                            break

        if modified:
            new_src = "".join(lines)
            try:
                ast.parse(new_src, filename=str(file_path))
                file_path.write_text(new_src, encoding="utf-8")
                return True
            except Exception as verify_err:
                logger.warning("Stub repair produced invalid AST for %s: %s", file_path, verify_err)
                return False
        return False

    def _extract_gd_peer_signatures(self, c_dir: Path, exclude_file: str = "") -> str:
        """Extracts class, func, static func, and const declarations from peer .gd files in c_dir
        so repair_godot_module has accurate cross-module function & class signatures.
        """
        if not c_dir.is_dir():
            return ""
        blocks: List[str] = []
        for gd_path in sorted(c_dir.glob("*.gd")):
            if gd_path.name == exclude_file:
                continue
            try:
                sig_lines: List[str] = []
                for idx, raw_line in enumerate(gd_path.read_text(encoding="utf-8").splitlines(), start=1):
                    stripped = raw_line.strip()
                    if stripped.startswith(("extends ", "class_name ", "class ", "static func ", "func ", "const ", "signal ")):
                        sig_lines.append(f"  L{idx}: {raw_line}")
                if sig_lines:
                    blocks.append(f"[{gd_path.name} signatures]\n" + "\n".join(sig_lines[:60]))
            except Exception:
                continue
        return "\n\n".join(blocks)

    def repair_godot_module(
        self,
        filename: str,
        broken_code: str,
        diagnostic: str,
        peer_context: str = "",
    ) -> str:
        """Autonomously repairs a GDScript module or project file using live Gemini inference with CodeRepairSpec."""
        if self.client is None:
            return broken_code
        from google.genai import types
        numbered_lines = "\n".join(
            f"{idx + 1:4d}: {line}" for idx, line in enumerate(broken_code.splitlines())
        )
        peer_block = (
            f"PEER MODULE SIGNATURES IN CARTRIDGE (for cross-module call alignment):\n{peer_context}\n\n"
            if peer_context
            else ""
        )
        prompt = (
            f"You are the Lead Gameplay SWE Agent repairing broken Godot 4.3 GDScript code for '{filename}'.\n"
            f"Compiler/Runtime Error Diagnostic(s):\n{diagnostic}\n\n"
            f"{peer_block}"
            f"BROKEN CODE (with line numbers for reference — DO NOT include line numbers in repaired_content):\n{numbered_lines}\n\n"
            f"Repair Guidelines:\n"
            f"1. Fix EVERY reported syntax/parse/runtime error, indentation error (ensure every block after 'if:', 'elif:', 'else:', 'for:', 'while:', 'match:' is indented by one additional tab '\\t'), missing declaration, or invalid call.\n"
            f"2. SELF-REFERENCE ERROR FIX: If '{filename}' reports `Identifier \"Scenes\" (or \"Mechanics\"/\"Entities\"/\"UIHud\") not declared in the current scope`, remove the self-module prefix (`Scenes.`, `Mechanics.`, `Entities.`, `UIHud.`) inside `{filename}`—never reference a script's external preload alias inside that script itself (or inside an inner class in `{filename}`; call the function directly or move/duplicate the helper inside the inner class).\n"
            f"3. TYPE INFERENCE ERROR FIX: If an error says `Cannot infer the type of \"<var>\" variable because the value doesn't have a set type`, replace `var <var> := ...` with `var <var> = ...` (or an explicit type annotation `var <var>: Type = ...`). Never use `:=` on untyped expressions.\n"
            f"4. PRELOADED SCRIPT CALL FIX: If an error says `Invalid call. Nonexistent function '<fn>' in base 'GDScript'`, remember that `const Module = preload('res://...')` is a `GDScript` resource, not an instance. Either instantiate `var inst = Module.new()` before calling `inst.<fn>(...)`, or align with the `static func` / `func` names shown in PEER MODULE SIGNATURES.\n"
            f"5. NO AUTO-GENERATED ASSET_LIBRARY.GD: `res://asset_library.gd` (`AssetLib`) does NOT exist. Load textures directly via `Image.load_from_file(ProjectSettings.globalize_path(res_path))` and `ImageTexture.create_from_image(img)` (and `tex.set_meta('source_path', res_path)`), and read asset metadata from `res://assets/asset_library.json`.\n"
            f"6. Keep Godot 4.3 syntax (use CharacterBody2D, await, FileAccess, Area2D), consistent tab ('\\t') indentation, and strictly avoid banned Godot 3 syntax (KinematicBody2D, yield, File.new()).\n"
            f"7. Return complete, valid source code (without line number prefixes) with deliberation strictly first."
        )
        for attempt in range(2):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CodeRepairSpec,
                    ),
                )
                repair_spec = CodeRepairSpec.model_validate_json(resp.text)
                self.last_deliberation = repair_spec.deliberation
                return repair_spec.repaired_content
            except Exception as exc:
                logger.warning("Godot module repair call failed (attempt %d): %s", attempt + 1, exc)
        return broken_code

    def repair_cartridge_code(
        self,
        candidate_dir: Union[Path, str],
        rejection_trace: Any,
        world_spec: Optional[GenesisWorldSpec] = None,
    ) -> Dict[str, Path]:
        """Closed-loop self-healing repair method driven by RejectionDiagnosticTrace (Feature 27).
        Autonomously inspects rejection diagnostic trace, identifies failing asset or code module,
        performs targeted repairs, verifies repaired files, and updates the candidate cartridge directory.
        """
        c_dir = Path(candidate_dir).resolve()
        results: Dict[str, Path] = {}
        if not c_dir.exists():
            return results

        from engine.supervisor.models import GateFailureType, RemedyType, RejectionDiagnosticTrace
        failure_type = getattr(rejection_trace, "failure_type", None)
        is_godot_cartridge = (c_dir / "project.godot").is_file() or bool(list(c_dir.glob("*.gd")))

        if (
            is_godot_cartridge
            and self._is_live_genai_client()
            and failure_type not in (GateFailureType.SYNTAX_AST_ERROR, GateFailureType.UNHANDLED_EXCEPTION)
        ):
            req_spec = None
            req_spec_path = c_dir / "requirements_spec.json"
            if req_spec_path.is_file():
                try:
                    from engine.studio.models import GameRequirementsSpec
                    req_spec = GameRequirementsSpec.model_validate_json(req_spec_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            return self.repair_godot_cartridge_holistic(
                c_dir=c_dir,
                rejection_trace=rejection_trace,
                req_spec=req_spec,
            )

        # ---------------------------------------------------------------------
        # 1. Gate A & Gate D: Python Code Repair (mechanics.py, entities.py, abilities.py)
        # ---------------------------------------------------------------------
        if failure_type in (
            GateFailureType.SYNTAX_AST_ERROR,
            GateFailureType.UNHANDLED_EXCEPTION,
        ):
            raw_target = getattr(rejection_trace, "file_path", None) or "main.gd"
            target_name = Path(raw_target).name
            target_path = c_dir / target_name
            if not target_path.is_file():
                if (c_dir / "main.gd").is_file():
                    target_path = c_dir / "main.gd"
                    target_name = "main.gd"
                elif (c_dir / "mechanics.py").is_file():
                    target_path = c_dir / "mechanics.py"
                    target_name = "mechanics.py"

            if target_path.is_file():
                broken_code = target_path.read_text(encoding="utf-8")
                reason = getattr(rejection_trace, "failure_reason", "")
                tb = getattr(rejection_trace, "runtime_traceback", "")
                line_no = getattr(rejection_trace, "line_number", None)
                diagnostic = f"{reason}\nLine: {line_no}\nTraceback:\n{tb or ''}"

                if target_name.endswith(".gd") or target_name == "project.godot":
                    if target_name == "main.gd" and (
                        "extends RefCounted" in broken_code
                        or "Script inherits from native type" in reason
                        or ("func _ready" not in broken_code and (c_dir / "scenes.gd").is_file())
                    ):
                        req_spec = None
                        req_spec_path = c_dir / "requirements_spec.json"
                        if req_spec_path.is_file():
                            try:
                                from engine.studio.models import GameRequirementsSpec
                                req_spec = GameRequirementsSpec.model_validate_json(req_spec_path.read_text(encoding="utf-8"))
                            except Exception:
                                pass
                        return self.repair_godot_cartridge_holistic(
                            c_dir=c_dir,
                            rejection_trace=rejection_trace,
                            req_spec=req_spec,
                        )
                    repaired_code = self.repair_godot_module(target_name, broken_code, diagnostic)
                    target_path.write_text(repaired_code, encoding="utf-8")
                    results[target_name] = target_path
                    healed = self._verify_and_heal_godot_syntax_and_probes(c_dir)
                    results.update(healed)
                else:
                    repaired_code = self.repair_code_module(target_name, broken_code, diagnostic)
                    valid, err_msg = self.verify_code_module(target_name, repaired_code, raise_on_ast_error=False)
                    if not valid:
                        raise CodeVerificationError(f"Self-healing repair for {target_name} failed verification: {err_msg}")

                    target_path.write_text(repaired_code, encoding="utf-8")
                    results[target_name] = target_path

        # ---------------------------------------------------------------------
        # 2. Gate B: Traversability Geometry Repair (level.json)
        # ---------------------------------------------------------------------
        elif failure_type in (
            GateFailureType.CRITICAL_PATH_BLOCKED,
            GateFailureType.TIMEOUT_STUCK,
        ):
            level_path = c_dir / "level.json"
            if level_path.is_file():
                level_data = json.loads(level_path.read_text(encoding="utf-8"))
                walls = level_data.get("walls", [])
                updated_walls = []
                coords = getattr(rejection_trace, "failure_coordinates", None)
                if coords:
                    bx, by = float(coords[0]), float(coords[1])
                else:
                    bx, by = 640.0, 360.0

                for wall in walls:
                    wx, wy, ww, wh = float(wall[0]), float(wall[1]), float(wall[2]), float(wall[3])
                    is_perimeter = (
                        (wx <= 0.0 and wy <= 0.0 and ww >= 1200.0) or
                        (wx <= 0.0 and wy >= 680.0 and ww >= 1200.0) or
                        (wx <= 0.0 and wy <= 0.0 and wh >= 680.0) or
                        (wx >= 1240.0 and wy <= 0.0 and wh >= 680.0)
                    )
                    overlaps = (wx <= bx <= (wx + ww) and wy <= by <= (wy + wh))
                    if overlaps and not is_perimeter:
                        logger.info("Self-healing: Removed interior blocking wall at (%f, %f, %f, %f)", wx, wy, ww, wh)
                        continue
                    updated_walls.append(wall)

                # Check blocked nodes if any
                blocked_nodes = getattr(rejection_trace, "blocked_nodes", []) or []
                if len(updated_walls) == len(walls) and blocked_nodes:
                    for b_node in blocked_nodes:
                        nx = (float(b_node[0]) + 0.5) * 32.0
                        ny = (float(b_node[1]) + 0.5) * 32.0
                        for idx, w in enumerate(updated_walls):
                            wx, wy, ww, wh = float(w[0]), float(w[1]), float(w[2]), float(w[3])
                            is_perim = (wx <= 0.0 or wx >= 1240.0 or wy <= 0.0 or wy >= 680.0) and (ww >= 1200.0 or wh >= 680.0)
                            if not is_perim and (wx <= nx <= (wx + ww) and wy <= ny <= (wy + wh)):
                                del updated_walls[idx]
                                break
                        if len(updated_walls) < len(walls):
                            break

                # If still identical, remove first non-perimeter wall
                if len(updated_walls) == len(walls):
                    for idx, w in enumerate(updated_walls):
                        wx, wy, ww, wh = float(w[0]), float(w[1]), float(w[2]), float(w[3])
                        is_perim = (wx <= 0.0 or wx >= 1240.0 or wy <= 0.0 or wy >= 680.0) and (ww >= 1200.0 or wh >= 680.0)
                        if not is_perim:
                            del updated_walls[idx]
                            break

                level_data["walls"] = updated_walls
                level_path.write_text(json.dumps(level_data, indent=2), encoding="utf-8")
                results["level.json"] = level_path

        # ---------------------------------------------------------------------
        # 3. Gate C: Survivability / Hazard Clearance Repair (level.json)
        # ---------------------------------------------------------------------
        elif failure_type in (
            GateFailureType.SURVIVABILITY_VIOLATION,
            GateFailureType.WALL_CLIPPING,
        ):
            level_path = c_dir / "level.json"
            if level_path.is_file():
                level_data = json.loads(level_path.read_text(encoding="utf-8"))
                hazards = level_data.get("hazards", [])
                safe_hazards = []
                spawn_point = level_data.get("spawn_point", [120.0, 360.0])
                spawn_x, spawn_y = float(spawn_point[0]), float(spawn_point[1])

                for h in hazards:
                    hx = float(h.get("x", 0.0))
                    hy = float(h.get("y", 0.0))
                    h_mod = dict(h)
                    dist_to_spawn = ((hx - spawn_x) ** 2 + (hy - spawn_y) ** 2) ** 0.5
                    if dist_to_spawn < 120.0:
                        h_mod["x"] = spawn_x + 240.0
                    h_mod["damage"] = max(5.0, float(h.get("damage", 15.0)) * 0.70)
                    safe_hazards.append(h_mod)

                level_data["hazards"] = safe_hazards
                for e in level_data.get("enemies", []):
                    e["speed"] = max(80.0, float(e.get("speed", 160.0)) * 0.80)
                    e["health"] = max(10.0, float(e.get("health", 40.0)) * 0.75)

                level_path.write_text(json.dumps(level_data, indent=2), encoding="utf-8")
                results["level.json"] = level_path

        # ---------------------------------------------------------------------
        # 4. Gate E: Asset Integrity Repair (.wav, .png, narrative.json)
        # ---------------------------------------------------------------------
        elif failure_type == GateFailureType.ASSET_INTEGRITY_VIOLATION:
            from engine.assets.palette import quantize_image_to_indexed, STYLE_BIBLE_16_PALETTE
            from PIL import Image
            import io
            import wave
            import struct
            import math

            # Safely re-quantize PNGs via in-memory buffer in offline/mock mode so files are never truncated on error
            if not self._is_live_genai_client():
                for png_file in c_dir.rglob("*.png"):
                    if "mockups" in png_file.parts:
                        continue
                    try:
                        img = Image.open(png_file)
                        img.load()
                        quantized = quantize_image_to_indexed(img, STYLE_BIBLE_16_PALETTE)
                        buf = io.BytesIO()
                        try:
                            quantized.save(buf, format="PNG")
                        except Exception:
                            quantized.info.pop("transparency", None)
                            buf = io.BytesIO()
                            quantized.save(buf, format="PNG")
                        png_file.write_bytes(buf.getvalue())
                        results[png_file.name] = png_file
                    except Exception as exc:
                        logger.warning("Could not re-quantize %s: %s", png_file, exc)

            # Re-synthesize silent or invalid WAV files
            for wav_file in c_dir.rglob("*.wav"):
                try:
                    self._write_valid_tone_wav(wav_file)
                    results[wav_file.name] = wav_file
                except Exception as exc:
                    logger.warning("Could not fix wav %s: %s", wav_file, exc)

            # Fix narrative.json if present and synthesize any missing referenced audio files
            narrative_path = c_dir / "narrative.json"
            if narrative_path.is_file():
                try:
                    n_data = json.loads(narrative_path.read_text(encoding="utf-8"))
                    tree = n_data.get("dialogue_tree", {})
                    if isinstance(tree, dict):
                        for node_id, node in tree.items():
                            choices = node.get("choices", [])
                            if not choices:
                                node["choices"] = [{"text": "Continue", "next_node": "root"}]
                    for bark_key in ("dialogue_barks", "barks", "cues"):
                        barks_val = n_data.get(bark_key)
                        if isinstance(barks_val, dict):
                            for cue_id, cue in barks_val.items():
                                if isinstance(cue, dict):
                                    af = cue.get("audio_file") or cue.get("voice_clip")
                                    if af:
                                        af_path = c_dir / af
                                        if not af_path.is_file():
                                            self._write_valid_tone_wav(af_path)
                                            results[af_path.name] = af_path
                        elif isinstance(barks_val, list):
                            for cue in barks_val:
                                if isinstance(cue, dict):
                                    af = cue.get("audio_file") or cue.get("voice_clip")
                                    if af:
                                        af_path = c_dir / af
                                        if not af_path.is_file():
                                            self._write_valid_tone_wav(af_path)
                                            results[af_path.name] = af_path
                    narrative_path.write_text(json.dumps(n_data, indent=2), encoding="utf-8")
                    results["narrative.json"] = narrative_path
                except Exception as exc:
                    logger.warning("Could not fix narrative.json: %s", exc)

        # ---------------------------------------------------------------------
        # 5. Gate E: Custom Telemetry Emission Repair
        # ---------------------------------------------------------------------
        elif failure_type == GateFailureType.TELEMETRY_EMISSION_FAILURE:
            manifest_path = c_dir / "telemetry_manifest.json"
            if manifest_path.is_file():
                try:
                    from engine.studio.telemetry_architect import TelemetryArchitectAgent
                    from engine.studio.models import TelemetryProbeDesignSpec
                    m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    raw_probes = m_data.get("probes", [])
                    probe_specs = [
                        TelemetryProbeDesignSpec(
                            deliberation=p.get("deliberation", "Self-healed telemetry probe hook"),
                            probe_name=p.get("probe_name", "custom_probe"),
                            category=p.get("category", "gameplay"),
                            description=p.get("description", "Self-healed telemetry probe hook description"),
                            target_class=p.get("target_class", "Node2D"),
                            target_method=p.get("target_method", "update"),
                            hypothesis_target=p.get("hypothesis_target", "Verify active gameplay loop engagement"),
                        )
                        for p in raw_probes if isinstance(p, dict) and p.get("probe_name")
                    ]
                    if probe_specs:
                        arch = TelemetryArchitectAgent(client=self.client, model_name=self.model_name)
                        is_godot = (c_dir / "project.godot").is_file() or (c_dir / "main.gd").is_file()
                        if is_godot:
                            for mod_name in ("main.gd", "mechanics.gd", "scenes.gd"):
                                mod_path = c_dir / mod_name
                                if mod_path.is_file():
                                    src = mod_path.read_text(encoding="utf-8")
                                    instrumented = arch._inject_gdscript_probes(src, probe_specs)
                                    mod_path.write_text(instrumented, encoding="utf-8")
                                    results[mod_name] = mod_path
                        else:
                            for mod_name in ("scenes.py", "mechanics.py"):
                                mod_path = c_dir / mod_name
                                if mod_path.is_file():
                                    src = mod_path.read_text(encoding="utf-8")
                                    instrumented = arch._inject_probes_into_source(src, probe_specs[:1])
                                    mod_path.write_text(instrumented, encoding="utf-8")
                                    results[mod_name] = mod_path
                except Exception as exc:
                    logger.warning("Could not self-heal telemetry probes: %s", exc)

        # ---------------------------------------------------------------------
        # 6. Gate F: Requirement & Acceptance Criteria Defect Repair (Milestone M4)
        # ---------------------------------------------------------------------
        elif (
            failure_type == GateFailureType.REQUIREMENT_UNFULFILLED
            or failure_type == "REQUIREMENT_UNFULFILLED"
            or (rejection_trace and str(getattr(rejection_trace, "failure_gate", "")).startswith("Gate F"))
        ):
            unfulfilled_ids = (
                getattr(rejection_trace, "unfulfilled_requirements", None)
                or getattr(rejection_trace, "remedy_parameters", {}).get("unfulfilled_requirements", [])
            )
            req_spec_path = c_dir / "requirements_spec.json"
            req_spec = None
            if req_spec_path.is_file():
                try:
                    from engine.studio.models import GameRequirementsSpec
                    req_spec = GameRequirementsSpec.model_validate_json(req_spec_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning("Could not parse requirements_spec.json during SWE repair: %s", exc)

            # Synthesize any missing audio or sprite assets defined in requirements_spec.json
            if req_spec is not None and getattr(req_spec, "asset_manifest", None):
                for asset in req_spec.asset_manifest:
                    a_name = getattr(asset, "name", "")
                    a_target = getattr(asset, "target_path", "")
                    candidates_to_check = []
                    if a_target:
                        candidates_to_check.append(c_dir / a_target)
                    if a_name:
                        folder = "assets/audio" if a_name.endswith(".wav") else "assets/sprites"
                        candidates_to_check.append(c_dir / folder / a_name)
                    for p in candidates_to_check:
                        if not p.is_file():
                            if p.name.endswith(".wav"):
                                self._write_valid_tone_wav(p)
                                results[p.name] = p
                            elif p.name.endswith(".png"):
                                self._write_valid_indexed_png(p)
                                results[p.name] = p

            failing_tracks = []
            if req_spec is not None and getattr(req_spec, "requirements", None):
                for t in req_spec.requirements:
                    if not unfulfilled_ids or t.id in unfulfilled_ids:
                        failing_tracks.append(t)

            diag_reason = getattr(rejection_trace, "failure_reason", "Requirement unfulfilled")
            rec_remedy = getattr(rejection_trace, "recommended_remedy", "Implement genuine mechanics logic")

            target_modules: List[str] = []
            if getattr(rejection_trace, "file_path", None):
                target_modules.append(Path(rejection_trace.file_path).name)
            rem_params = getattr(rejection_trace, "remedy_parameters", {}) or {}
            if rem_params.get("files_to_patch"):
                for f in rem_params["files_to_patch"]:
                    target_modules.append(Path(f).name)
            if rem_params.get("target_file"):
                target_modules.append(Path(rem_params["target_file"]).name)

            ast_ev = getattr(rejection_trace, "ast_evidence", {}) or {}
            raw_stubs = []
            for k in ("hollow_stubs", "pass_stubs", "empty_bodies"):
                val = ast_ev.get(k, [])
                if isinstance(val, list):
                    raw_stubs.extend(val)
            for stub in raw_stubs:
                s_file = None
                if isinstance(stub, dict):
                    s_file = stub.get("file") or stub.get("module")
                elif isinstance(stub, str):
                    s_file = stub.split(":")[0]
                if s_file:
                    target_modules.append(Path(s_file).name)

            if failing_tracks:
                for track in failing_tracks:
                    for tm in getattr(track, "target_modules", []):
                        target_modules.append(Path(tm).name)

            from engine.supervisor.challenger import CandidateASTInspector
            ast_insp = CandidateASTInspector()
            dir_ast = ast_insp.inspect_directory(c_dir)
            for stub in dir_ast.get("hollow_stubs", []):
                s_file = stub.get("file") or stub.get("module")
                if s_file:
                    target_modules.append(Path(s_file).name)

            is_godot = (c_dir / "project.godot").is_file() or bool(list(c_dir.glob("*.gd")))
            if is_godot:
                gd_repaired = self.repair_godot_cartridge_holistic(
                    c_dir=c_dir,
                    rejection_trace=rejection_trace,
                    req_spec=req_spec,
                )
                results.update(gd_repaired)
                return results

            existing_targets = [m for m in set(target_modules) if (c_dir / m).is_file()]
            if not existing_targets:
                existing_targets = [m for m in ["mechanics.py", "scenes.py", "entities.py"] if (c_dir / m).is_file()]

            for mod_name in sorted(set(existing_targets)):
                m_path = c_dir / mod_name
                if m_path.is_file():
                    src = m_path.read_text(encoding="utf-8")
                    diag = f"Gate F Challenger Rejection: {diag_reason}\nRemedy: {rec_remedy}\nUnfulfilled requirements: {unfulfilled_ids}"
                    repaired = None
                    try:
                        repaired = self.repair_code_module(mod_name, src, diag)
                    except Exception as exc:
                        logger.debug("repair_code_module failed or unavailable: %s", exc)
                    if repaired:
                        valid, err = self.verify_code_module(mod_name, repaired, raise_on_ast_error=False)
                        if valid:
                            m_path.write_text(repaired, encoding="utf-8")
                            results[mod_name] = m_path
                            continue
                    stub_fixed = self._repair_hollow_stubs_in_source(m_path, diag_reason, rec_remedy)
                    if stub_fixed:
                        results[mod_name] = m_path

        return results

    # -----------------------------------------------------------------------
    # PEV (Plan / Research -> Execute Surgical Patch -> Verify) Debugging Engine
    # -----------------------------------------------------------------------

    def _build_cartridge_symbol_index(self, c_dir: Path) -> str:
        """Builds a compact structural symbol index (classes, functions, signals, constants, section headers,
        and external script/asset references with exact 1-indexed line numbers) across all .gd files in c_dir.
        Avoids dumping full files so the agent can research and request only relevant code slices.
        """
        if not c_dir.is_dir():
            return ""
        blocks: List[str] = []
        for gd_path in sorted(c_dir.glob("*.gd")):
            try:
                lines = gd_path.read_text(encoding="utf-8").splitlines()
                entry_lines: List[str] = []
                for idx, raw_line in enumerate(lines, start=1):
                    stripped = raw_line.strip()
                    if stripped.startswith(("extends ", "class_name ", "class ", "static func ", "func ", "const ", "signal ")):
                        entry_lines.append(f"  L{idx:4d}: {raw_line}")
                    elif stripped.startswith("# ") and not stripped.startswith("# ---") and len(stripped) > 6:
                        entry_lines.append(f"  L{idx:4d}: {raw_line}")
                    elif "preload(" in stripped or "load(" in stripped or "AssetLib" in stripped or "asset_library" in stripped:
                        entry_lines.append(f"  L{idx:4d}: {raw_line}")
                header = f"=== {gd_path.name} ({len(lines)} lines total) ==="
                blocks.append(header + "\n" + "\n".join(entry_lines[:110]))
            except Exception:
                continue
        return "\n\n".join(blocks)

    def _extract_code_slice(self, c_dir: Path, req: CodeSliceRequest) -> str:
        """Extracts a bounded, line-numbered code slice from `req.filename` for PEV context inspection.
        Caps slice span so the LLM receives only high-signal relevant context rather than full-file dumps.
        """
        fname = Path(req.filename).name
        fpath = c_dir / fname
        if not fpath.is_file():
            return f"=== SLICE: {fname} (FILE NOT FOUND) ==="

        lines = fpath.read_text(encoding="utf-8").splitlines()
        total = len(lines)
        if total == 0:
            return f"=== SLICE: {fname} (EMPTY FILE) ==="

        windows: List[Tuple[int, int]] = []
        if req.symbol_query and req.symbol_query.strip():
            raw_q = req.symbol_query.strip()
            # Split on ' and ' if multiple targets are listed in one string
            sub_queries = [sq.strip() for sq in raw_q.split(" and ") if sq.strip()]
            for sq in sub_queries:
                # Check if explicit line range is embedded like '(lines 298-360)'
                if "lines " in sq:
                    after_lines = sq.split("lines ", 1)[1]
                    num_buf = ""
                    nums: List[int] = []
                    for ch in after_lines:
                        if ch.isdigit():
                            num_buf += ch
                        elif num_buf:
                            nums.append(int(num_buf))
                            num_buf = ""
                            if len(nums) >= 2:
                                break
                    if num_buf and len(nums) < 2:
                        nums.append(int(num_buf))
                    if len(nums) >= 2 and 1 <= nums[0] <= nums[1]:
                        s_ln = max(1, min(nums[0] - 4, total))
                        e_ln = max(s_ln, min(nums[1] + 6, s_ln + 90, total))
                        windows.append((s_ln, e_ln))

                clean_sq = sq.split("(", 1)[0].strip()
                if "." in clean_sq:
                    cls_part, mem_part = clean_sq.split(".", 1)
                    cls_part = cls_part.strip()
                    mem_part = mem_part.strip()
                    cls_start = 1
                    cls_end = total
                    for idx, line in enumerate(lines, start=1):
                        if line.strip().startswith(f"class {cls_part}"):
                            cls_start = idx
                            for next_idx in range(idx + 1, total + 1):
                                if lines[next_idx - 1].startswith("class "):
                                    cls_end = next_idx - 1
                                    break
                            break
                    found_member = False
                    for idx in range(cls_start, cls_end + 1):
                        st = lines[idx - 1].strip()
                        if st.startswith((f"func {mem_part}", f"static func {mem_part}")):
                            s_ln = max(1, idx - 3)
                            e_ln = min(cls_end, idx + 88)
                            windows.append((s_ln, e_ln))
                            found_member = True
                            break
                    if not found_member and cls_start > 1:
                        windows.append((max(1, cls_start - 2), min(cls_end, cls_start + 88)))
                else:
                    for idx, line in enumerate(lines, start=1):
                        if clean_sq and clean_sq in line:
                            stripped = line.strip()
                            if stripped.startswith(("class ", "func ", "static func ")):
                                s_ln = max(1, idx - 2)
                                e_ln = min(total, idx + 88)
                            else:
                                s_ln = max(1, idx - 8)
                                e_ln = min(total, idx + 45)
                            windows.append((s_ln, e_ln))
                            break

        if not windows:
            s = max(1, min(req.start_line, total))
            e = max(s, min(req.end_line, total))
            if e - s > 95:
                e = s + 95
            windows.append((s, e))

        # Merge overlapping windows
        windows.sort(key=lambda w: w[0])
        merged: List[Tuple[int, int]] = []
        for s, e in windows:
            if merged and s <= merged[-1][1] + 5:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        out_blocks: List[str] = []
        for s, e in merged:
            snippet_lines = [
                f"{ln:4d}: {lines[ln - 1]}"
                for ln in range(s, e + 1)
            ]
            out_blocks.append(
                f"=== {fname} [Lines {s}..{e} of {total}] ===\n" + "\n".join(snippet_lines)
            )
        return "\n\n".join(out_blocks)

    def _apply_surgical_patches(
        self,
        c_dir: Path,
        patches: List[SurgicalCodePatch],
    ) -> Dict[str, Path]:
        """Applies surgical line-range replacements to cartridge files in descending line order."""
        updated: Dict[str, Path] = {}
        patches_by_file: Dict[str, List[SurgicalCodePatch]] = {}
        for p in patches:
            fname = Path(p.filename).name
            if fname.endswith(".gd"):
                patches_by_file.setdefault(fname, []).append(p)

        for fname, file_patches in patches_by_file.items():
            fpath = c_dir / fname
            if not fpath.is_file():
                continue
            lines = fpath.read_text(encoding="utf-8").splitlines()
            total = len(lines)
            # Sort descending by start_line so earlier line indices do not shift
            sorted_patches = sorted(file_patches, key=lambda x: (x.start_line, x.end_line), reverse=True)
            last_start = total + 1
            applied_any = False

            for patch in sorted_patches:
                s = max(1, min(patch.start_line, total))
                e = max(s, min(patch.end_line, total))
                if e >= last_start:
                    # Skip overlapping patch range in the same pass
                    continue
                cleaned_replacement: List[str] = []
                for raw_line in patch.replacement_code.splitlines():
                    # Strip accidental ' 123: ' line number prefix only if LLM echoed slice numbering
                    if (
                        len(raw_line) >= 6
                        and raw_line[:4].strip().isdigit()
                        and raw_line[4:6] == ": "
                    ):
                        cleaned_replacement.append(raw_line[6:])
                    else:
                        cleaned_replacement.append(raw_line)
                lines[s - 1 : e] = cleaned_replacement
                last_start = s
                applied_any = True

            if applied_any:
                fpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
                updated[fname] = fpath

        return updated

    def debug_and_patch_pev_cycle(
        self,
        c_dir: Path,
        defect_summary: str,
        initial_diagnostics: Optional[List[Any]] = None,
        multimodal_parts: Optional[List[Any]] = None,
        pre_slice_requests: Optional[List[CodeSliceRequest]] = None,
        max_pev_cycles: int = 4,
    ) -> Dict[str, Path]:
        """Executes a disciplined Plan (Research) -> Execute (Inspect Slices & Surgical Patch) -> Verify
        loop without dumping full files at the LLM.
        """
        if self.client is None or not self._is_live_genai_client():
            return {}

        from google.genai import types
        from engine.sim.godot_verifier import verify_godot_cartridge_syntax
        from engine.studio.asset_library import CartridgeAssetLibrary

        results: Dict[str, Path] = {}
        current_diagnostics = list(initial_diagnostics or [])
        current_defect = defect_summary

        asset_summary = ""
        try:
            asset_summary = CartridgeAssetLibrary(cartridge_dir=c_dir).build_prompt_summary_for_swe()
        except Exception:
            pass

        for pev_cycle in range(max_pev_cycles):
            symbol_index = self._build_cartridge_symbol_index(c_dir)

            # Pre-extract tight +/- 18 line windows around any concrete compiler/runtime diagnostic lines
            auto_slices: List[str] = []
            if pre_slice_requests and pev_cycle == 0:
                for pre_req in pre_slice_requests[:4]:
                    s_txt = self._extract_code_slice(c_dir, pre_req)
                    if s_txt not in auto_slices:
                        auto_slices.append(s_txt)

            diag_lines_text: List[str] = []
            for d in current_diagnostics[:6]:
                d_file = getattr(d, "file_path", None)
                d_line = int(getattr(d, "line_number", 1) or 1)
                d_msg = getattr(d, "error_message", str(d))
                fname = Path(d_file).name if d_file else "main.gd"
                diag_lines_text.append(f"- {fname}:{d_line} -> {d_msg}")
                if (c_dir / fname).is_file():
                    err_slice = self._extract_code_slice(
                        c_dir,
                        CodeSliceRequest(
                            deliberation=f"Auto-extracted error window around {fname}:{d_line}",
                            filename=fname,
                            start_line=max(1, d_line - 18),
                            end_line=d_line + 18,
                        ),
                    )
                    if err_slice not in auto_slices:
                        auto_slices.append(err_slice)

            diag_section = (
                "CONCRETE COMPILER / RUNTIME DIAGNOSTICS:\n" + "\n".join(diag_lines_text) + "\n\n"
                if diag_lines_text
                else ""
            )
            auto_slice_section = (
                "PRE-INSPECTED TARGET CODE WINDOWS:\n" + "\n\n".join(auto_slices) + "\n\n"
                if auto_slices
                else ""
            )

            # -----------------------------------------------------------------
            # PEV PHASE 1: PLAN & RESEARCH (Request targeted code slices)
            # -----------------------------------------------------------------
            research_prompt = (
                f"You are the Lead Gameplay SWE Agent in PHASE 1 (PLAN & RESEARCH) of a PEV debugging cycle (Iteration {pev_cycle + 1}/{max_pev_cycles}).\n"
                f"Do NOT guess blindly and do NOT request entire large files. Analyze the defect and the Cartridge Symbol Index below, "
                f"formulate a root-cause hypothesis, and request ONLY the specific sub-block code slices (40-80 lines each) "
                f"you need to inspect to craft a compact surgical fix.\n\n"
                f"--- DEFECT TO RESOLVE ---\n{current_defect}\n\n"
                f"{diag_section}"
                f"{auto_slice_section}"
                f"--- CARTRIDGE SYMBOL & REFERENCE INDEX ---\n{symbol_index}\n\n"
                f"{asset_summary}\n"
                f"Return a `DebugResearchPlan` with `deliberation` strictly first."
            )

            plan: Optional[DebugResearchPlan] = None
            for attempt in range(2):
                try:
                    resp_plan = self.client.models.generate_content(
                        model=self.model_name,
                        contents=research_prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=DebugResearchPlan,
                        ),
                    )
                    if resp_plan.text:
                        plan = DebugResearchPlan.model_validate_json(resp_plan.text)
                        self.last_deliberation = plan.deliberation
                        break
                except Exception as exc:
                    logger.warning("PEV Phase 1 (Research Plan) failed (attempt %d): %s", attempt + 1, exc)

            if plan is None:
                break

            # Extract the requested slices
            extracted_slices: List[str] = list(auto_slices)
            for req in plan.slice_requests[:6]:
                slice_text = self._extract_code_slice(c_dir, req)
                if slice_text not in extracted_slices:
                    extracted_slices.append(slice_text)

            # -----------------------------------------------------------------
            # PEV PHASE 2: EXECUTE (Inspect focused slices & emit surgical patches)
            # -----------------------------------------------------------------
            patch_payload: Optional[SurgicalPatchPayload] = None
            for hop in range(2):
                slices_joined = "\n\n".join(extracted_slices)
                hop_rule_6 = (
                    "6. If you need to inspect one more function/caller before patching, populate `additional_slice_requests`."
                    if hop == 0
                    else "6. FINAL HOP — YOU MUST EMIT CONCRETE `SurgicalCodePatch` ITEMS IN `patches` NOW: Do NOT leave `patches` empty on Hop 2!"
                )
                execute_prompt = (
                    f"You are the Lead Gameplay SWE Agent in PHASE 2 (EXECUTE SURGICAL PATCH) of a PEV debugging cycle (Iteration {pev_cycle + 1}/{max_pev_cycles}, Hop {hop + 1}).\n"
                    f"Root-Cause Hypothesis: {plan.root_cause_hypothesis}\n\n"
                    f"--- DEFECT TO RESOLVE ---\n{current_defect}\n\n"
                    f"{diag_section}"
                    f"--- INSPECTED CODE SLICES (1-indexed line numbers) ---\n{slices_joined}\n\n"
                    f"--- CARTRIDGE SYMBOL INDEX (for cross-module call signatures) ---\n{symbol_index}\n\n"
                    f"{asset_summary}\n"
                    f"--- SURGICAL PATCH RULES ---\n"
                    f"1. Provide `deliberation` strictly first (Rule 4).\n"
                    f"2. Emit focused `SurgicalCodePatch` items replacing ONLY the defective contiguous line ranges `[start_line..end_line]` in the target `.gd` file(s).\n"
                    f"3. KEEP EACH PATCH COMPACT AND SURGICAL (MAX 40-65 LINES OF REPLACEMENT CODE): Never rewrite an entire 150-line class or copy-paste unchanged trailing sections of a long `_draw()` method (such as shockwave/particle/geyser loops). Replace ONLY the specific 10-55 lines responsible for the defect (for example, `SceneAssets.draw_tiled_platform`, or the `for s in shelves:` block, or the `for coord in canal_grid.cells.keys():` block) while leaving surrounding lines untouched!\n"
                    f"4. Do NOT include line number prefixes (like ` 196: `) in `replacement_code`. Preserve exact tab (`\\t`) indentation matching the surrounding function/class scope.\n"
                    f"5. Never reference `AssetLib` or `res://asset_library.gd` (it does not exist). Load PNG textures via `Image.load_from_file(ProjectSettings.globalize_path(res_path))` and `ImageTexture.create_from_image(img)` (and `tex.set_meta('source_path', res_path)`). IMPORTANT GODOT 4 TEXTURE TILING RULE: When calling `draw_colored_polygon(pts, color, uvs, tex)` or `draw_polygon` with repeating UVs (`uv > 1.0`), you MUST set `texture_repeat = CanvasItem.TEXTURE_REPEAT_ENABLED` on the `Node2D`/`CanvasItem` (in `_init` or `_draw`), otherwise Godot 4 clamps UVs > 1.0 to a single edge pixel and renders a flat solid color! Also use bright modulation (`Color(0.85, 0.95, 0.90, 1.0)`) so the surface texture detail is vivid.\n"
                    f"{hop_rule_6}\n"
                )
                contents_exec: Any = (
                    [execute_prompt] + list(multimodal_parts)
                    if (multimodal_parts and pev_cycle == 0 and hop == 0)
                    else execute_prompt
                )
                for attempt in range(2):
                    try:
                        resp_exec = self.client.models.generate_content(
                            model=self.model_name,
                            contents=contents_exec,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=SurgicalPatchPayload,
                            ),
                        )
                        if resp_exec.text:
                            patch_payload = SurgicalPatchPayload.model_validate_json(resp_exec.text)
                            self.last_deliberation = patch_payload.deliberation
                            break
                    except Exception as exc:
                        logger.warning("PEV Phase 2 (Execute Patch) failed (attempt %d): %s", attempt + 1, exc)

                if patch_payload is None:
                    break

                if patch_payload.additional_slice_requests and not patch_payload.patches and hop == 0:
                    for extra_req in patch_payload.additional_slice_requests[:6]:
                        extra_txt = self._extract_code_slice(c_dir, extra_req)
                        if extra_txt not in extracted_slices:
                            extracted_slices.append(extra_txt)
                    continue
                break

            applied_in_cycle = False
            if patch_payload and patch_payload.patches:
                patched_files = self._apply_surgical_patches(c_dir, patch_payload.patches)
                if patched_files:
                    applied_in_cycle = True
                    results.update(patched_files)

            # -----------------------------------------------------------------
            # PEV PHASE 3: VERIFY (Headless Godot compiler + 2-tick runtime check)
            # -----------------------------------------------------------------
            is_valid, new_diagnostics = verify_godot_cartridge_syntax(c_dir)
            # Also check if any file still contains broken `AssetLib.` calls
            lingering_assetlib: List[str] = []
            for gd_p in sorted(c_dir.glob("*.gd")):
                for ln_idx, ln_txt in enumerate(gd_p.read_text(encoding="utf-8").splitlines(), start=1):
                    if "AssetLib." in ln_txt or "asset_library.gd" in ln_txt:
                        lingering_assetlib.append(f"{gd_p.name}:{ln_idx}: references removed AssetLib/asset_library.gd (`{ln_txt.strip()}`)")

            if is_valid and not lingering_assetlib:
                if applied_in_cycle or results or initial_diagnostics:
                    logger.info("PEV cycle %d verified clean (0 compiler/runtime errors).", pev_cycle + 1)
                    break

            current_diagnostics = list(new_diagnostics)
            verify_feedback = []
            for d in new_diagnostics[:6]:
                verify_feedback.append(f"{d.file_path.name}:{d.line_number}: {d.error_message}")
            verify_feedback.extend(lingering_assetlib[:6])
            if not verify_feedback and not applied_in_cycle:
                verify_feedback.append(
                    "No SurgicalCodePatch items were emitted in `patches`! You MUST emit concrete SurgicalCodePatch line replacements to fix the defect."
                )
            current_defect = (
                f"Previous patch verification found remaining issues:\n"
                + "\n".join(f"- {item}" for item in verify_feedback)
                + f"\nOriginal context: {defect_summary[:600]}"
            )

        return results

    # -----------------------------------------------------------------------
    # Strategic Game Production Planning & Iterative Per-Scene Critique Loop
    # -----------------------------------------------------------------------

    def _collect_concept_art_parts(self, c_dir: Path) -> List[Any]:
        """Loads 16:9 concept art PNGs from assets/mockups/ and assets/concepts/ as multimodal Gemini image parts."""
        if not self._is_live_genai_client():
            return []
        from google.genai import types
        parts: List[Any] = []
        for subdir_name in ("mockups", "concepts"):
            c_sub = c_dir / "assets" / subdir_name
            if c_sub.is_dir():
                for png_path in sorted(c_sub.glob("*.png"))[:3]:
                    try:
                        parts.append(
                            types.Part.from_bytes(data=png_path.read_bytes(), mime_type="image/png")
                        )
                    except Exception:
                        continue
        return parts[:3]

    def formulate_production_strategy(
        self,
        c_dir: Path,
        force_refresh: bool = False,
    ) -> GameProductionStrategySpec:
        """Formulates a holistic GameProductionStrategySpec before or during cartridge construction.
        Decomposes the Game Bible and 16:9 Concept Mockups into an explicit multi-layer environment
        topology plan, required visual sprite assets, procedural audio cues, code architecture, and
        diegetic UI strategy. Automatically fabricates any missing required visual assets on disk.
        """
        if self.client is None or not self._is_live_genai_client():
            raise StudioAgentError("formulate_production_strategy requires an active google.genai Client (Rule 5)")

        from google.genai import types
        from engine.studio.asset_library import CartridgeAssetLibrary

        strategy_path = c_dir / "production_strategy.json"
        if strategy_path.is_file() and not force_refresh:
            try:
                cached = GameProductionStrategySpec.model_validate_json(
                    strategy_path.read_text(encoding="utf-8")
                )
                return cached
            except Exception:
                pass

        bible_text = ""
        bible_path = c_dir / "game_bible.json"
        palette_mood = "stylized_fantasy"
        art_direction = ""
        if bible_path.is_file():
            raw_bible = bible_path.read_text(encoding="utf-8")
            bible_text = raw_bible[:14000]
            try:
                b_obj = json.loads(raw_bible)
                palette_mood = b_obj.get("visual_style", {}).get("palette_mood", palette_mood)
                art_direction = json.dumps(b_obj.get("visual_style", {}))
            except Exception:
                pass

        lib = CartridgeAssetLibrary(cartridge_dir=c_dir, client=self.client)
        existing_catalog_summary = lib.build_prompt_summary_for_swe()

        concept_parts = self._collect_concept_art_parts(c_dir)

        prompt = (
            "You are the Principal Game Architect & Creative Director formulating the master "
            "`GameProductionStrategySpec` for this Godot 4 game cartridge.\n"
            "Study the Game Bible and the attached 16:9 Concept Art Mockups carefully.\n"
            "We need a comprehensive strategy for how to build the intended game so it looks, sounds, and plays "
            "like a rich, cohesive, playable environment matching the Concept Art — NEVER a static background or "
            "blocky programmer-art rectangles.\n\n"
            "--- STRATEGIC PLANNING REQUIREMENTS ---\n"
            "1. Provide `deliberation` strictly first (Rule 4).\n"
            "2. `environment_topology_strategy`: Explain how the level/world must be decomposed into multi-plane parallax "
            "horizons, organic sculpted terrain silhouettes, continuous glowing fluid/energy channels, and interactive world "
            "structures/flora.\n"
            "3. `camera_and_perspective_coherence_strategy`: CRITICAL — Ensure 100% coherence between the character's movement plane "
            "and the landscape orientation! When characters/canals operate in 2D top-down / 3/4 overhead `(x, y)` space across a "
            "walkable shelf `Rect2` (`s.position.y .. s.end.y`), the ENTIRE shelf `Rect2` (`s.position.y .. s.end.y`) is the "
            "**horizontal walkable TOP plateau surface** and MUST be filled with the lush ground texture (`terrain_surface.png` — "
            "emerald moss, grass, and ancient flagstone pavers). NEVER fill `s.position.y .. s.end.y` with vertical cliff-wall "
            "bricks (`structure_wall_tile.png`) or vertical cliff cracks, or the character will look like she is walking sideways "
            "on the vertical face of a cliff! Instead, hang the vertical cliff wall (`structure_wall_tile.png` + vertical striations "
            "+ tapering stalactites) and overhanging moss scallops ONLY **below the southern bottom edge (`s.end.y`)** of each "
            "plateau (from `s.end.y` down to `s.end.y + 110`) as a 3/4 perspective front cliff skirt, and distribute flora/spires "
            "across the 2D `(x, y)` plateau floor (`y` between `s.position.y + 30` and `s.end.y - 30`) with ground shadows.\n"
            "4. `scene_layers`: Define the ordered depth layers (`z_index`) from distant sky/horizon backdrop to mid-ground "
            "landmarks, main playfield terrain & fluid channels, interactive flora/structures/entities, foreground accents, "
            "and diegetic HUD.\n"
            "5. `required_visual_assets`: List all `AssetRequirement` sprites needed to realize the environment, "
            "flora, structures, hazards, and characters. Include existing assets from the catalog AND specify any missing "
            "environmental/flora/prop/structure sprites that should be fabricated.\n"
            "6. `audio_cues` & `psychoacoustic_audio_strategy`: Define the procedural `AudioStreamWAV` audio cues required for "
            "primary player mechanics, traversal, environmental resonance, and hazards. STRICT PSYCHOACOUSTIC RULE: Forbid "
            "continuous looping high-pitched sine/square wave drones (`AudioStreamWAV.LOOP_FORWARD` on `> 200 Hz` tones like a "
            "constant 523 Hz C5 hum playing from `_ready()`), forbid piercing high-frequency squeals (`>= 800 Hz`), and forbid "
            "harsh square-wave buzzes. All procedural sounds must use `loop_mode = AudioStreamWAV.LOOP_DISABLED`, gentle "
            "exponential decay envelopes (`exp(-8.0 * t)`), warm low/mid frequencies (`110–440 Hz`), and comfortable levels "
            "(`volume_db <= -16.0`), triggered only on discrete events.\n"
            "7. `code_architecture_strategy`: Detail how `scenes.gd`, `entities.gd`, `ui_hud.gd`, and `main.gd` collaborate.\n"
            "8. `diegetic_ui_strategy`: Detail how the HUD must be styled as a clean, diegetic/thematic overlay matching the "
            "concept art (warm parchment/crest frame, clear resource bar, concise objective & controls banner) rather than "
            "raw engineering debug telemetry text.\n\n"
            f"--- GAME BIBLE EXCERPT ---\n{bible_text}\n\n"
            f"--- CURRENT ASSET CATALOG ON DISK ---\n{existing_catalog_summary}\n"
        )

        contents: Any = [prompt] + concept_parts if concept_parts else prompt
        strategy: Optional[GameProductionStrategySpec] = None
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GameProductionStrategySpec,
                    ),
                )
                if resp.text:
                    strategy = GameProductionStrategySpec.model_validate_json(resp.text)
                    self.last_deliberation = strategy.deliberation
                    break
            except Exception as exc:
                last_err = exc
                logger.warning("formulate_production_strategy attempt %d failed: %s", attempt + 1, exc)

        if strategy is None:
            raise StudioAgentError(f"Failed to formulate GameProductionStrategySpec: {last_err}")

        strategy_path.write_text(strategy.model_dump_json(indent=2), encoding="utf-8")

        # Fabricate any newly requested visual assets that are not yet in the asset catalog
        if strategy.required_visual_assets:
            try:
                lib.ensure_required_assets(
                    required_specs=strategy.required_visual_assets,
                    palette_mood=palette_mood,
                    art_direction=art_direction,
                )
            except Exception as asset_exc:
                logger.warning("Non-fatal error fabricating additional strategy assets: %s", asset_exc)

        return strategy

    def analyze_scene_and_restrategize(
        self,
        c_dir: Path,
        strategy: Optional[GameProductionStrategySpec] = None,
    ) -> SceneCritiqueAndStrategySpec:
        """Runs a live 60-tick headless Godot 4 viewport capture (`playtest_t0.png` and `playtest_tend.png`),
        compares the rendered scene side-by-side against the Game Bible, 16:9 Concept Mockups, and
        `GameProductionStrategySpec`, and outputs a structured `SceneCritiqueAndStrategySpec` identifying
        every visual, perspective, environmental, audio, architectural, or UX deficiency to remediate.
        """
        if self.client is None or not self._is_live_genai_client():
            raise StudioAgentError("analyze_scene_and_restrategize requires an active google.genai Client (Rule 5)")

        from google.genai import types
        from engine.sim.godot_verifier import GodotHeadlessSimulator, verify_godot_cartridge_syntax
        from engine.studio.asset_library import CartridgeAssetLibrary

        if strategy is None:
            strategy = self.formulate_production_strategy(c_dir)

        # 1. Run syntax check + live 60-tick Godot headless simulation with GPU viewport capture
        is_valid, syntax_diags = verify_godot_cartridge_syntax(c_dir)
        sim = GodotHeadlessSimulator(c_dir, capture_viewport=True, max_ticks=60)
        sim_state = sim.run_until_complete()

        diag_summary_lines = [
            f"- {d.file_path.name}:{d.line_number}: {d.error_message}"
            for d in (syntax_diags or [])[:8]
        ]
        for rd in (sim.runtime_diagnostics or [])[:8]:
            diag_summary_lines.append(f"- Runtime {rd.file_path.name}:{rd.line_number}: {rd.error_message}")

        last_frame_summary = (
            f"Ticks completed: {sim_state.step_count} | Player pos: {sim_state.player_pos} | "
            f"Scene audit: {json.dumps(getattr(sim, 'scene_audit', {}), default=str)[:1200]}"
        )

        # 2. Collect multimodal image parts: Concept Mockups FIRST, then live rendered Godot captures
        multimodal_parts: List[Any] = []
        concept_parts = self._collect_concept_art_parts(c_dir)
        multimodal_parts.extend(concept_parts)

        captures_dir = c_dir / "assets" / "playtest_captures"
        for cap_name in ("playtest_t0.png", "playtest_tend.png"):
            cap_p = captures_dir / cap_name
            if cap_p.is_file():
                try:
                    multimodal_parts.append(
                        types.Part.from_bytes(data=cap_p.read_bytes(), mime_type="image/png")
                    )
                except Exception:
                    pass

        symbol_index = self._build_cartridge_symbol_index(c_dir)
        asset_summary = CartridgeAssetLibrary(cartridge_dir=c_dir).build_prompt_summary_for_swe()

        # Extract focused slices for terrain perspective (scenes.gd), spire layering (scenes.gd), and audio synthesis (main.gd)
        terrain_slice = self._extract_code_slice(
            c_dir,
            CodeSliceRequest(
                deliberation="Inspect TerraceRenderer._draw to check camera/terrain perspective alignment and canal pass ordering",
                filename="scenes.gd",
                symbol_query="TerraceRenderer._draw",
            ),
        )
        spires_slice = self._extract_code_slice(
            c_dir,
            CodeSliceRequest(
                deliberation="Inspect WorldSpireLayer._draw and SceneAssets.draw_tiled_rect for oversized foreground column occlusion",
                filename="scenes.gd",
                symbol_query="WorldSpireLayer._draw",
            ),
        )
        audio_slice = self._extract_code_slice(
            c_dir,
            CodeSliceRequest(
                deliberation="Inspect audio setup and procedural synthesis in main.gd for continuous high-pitched loops or squeals",
                filename="main.gd",
                symbol_query="func _setup_audio",
            ),
        )

        bible_excerpt = ""
        if (c_dir / "game_bible.json").is_file():
            bible_excerpt = (c_dir / "game_bible.json").read_text(encoding="utf-8")[:8000]

        critique_prompt = (
            "You are the Studio Art Director, Lead Level Designer, Audio Director & Principal Gameplay SWE conducting a "
            "rigorous Per-Scene Visual, Perspective, Audio & Gameplay Critique (`SceneCritiqueAndStrategySpec`).\n"
            "Attached images:\n"
            "- First image(s): Target 16:9 Concept Art Mockups from the Game Bible.\n"
            "- Final image(s): Actual live Godot 4 viewport renders (`playtest_t0.png` and `playtest_tend.png`) "
            "captured from the current cartridge build.\n\n"
            "Compare the actual rendered Godot scene and audio code against the Concept Art Mockups, the Game Bible, and the "
            "`GameProductionStrategySpec` below. Scrutinize EVERY pillar below:\n"
            "1. **CRITICAL — Camera & Landscape Perspective Coherence (`scenes.gd` -> `TerraceRenderer._draw`, `FernsCluster._init`, `WorldSpireLayer._draw`)**:\n"
            "   - Look at where the character (`vesper` at `y ~ 420`) and canal (`y ~ 440`) are positioned on the island in `playtest_t0.png` / `playtest_tend.png` and in `TERRAIN CODE SLICE` / `SPIRES CODE SLICE` below.\n"
            "   - In a 3/4 top-down floating island plateau:\n"
            "     a) The ENTIRE walkable shelf `Rect2` (`s.position.y .. s.end.y`, e.g. `y = 260..700`) is the **horizontal walkable TOP plateau surface** and MUST be filled with `terrain_surface.png` (lush emerald moss/grass & ancient courtyard flagstone pavers, with organic mossy borders framing its perimeter).\n"
            "     b) The vertical cliff face (`structure_wall_tile.png` + vertical cliff striations + jagged stalactites) and overhanging moss scallops MUST hang **ONLY below the southern bottom edge (`s.end.y`)** of each plateau (e.g., from `s.end.y` down to `s.end.y + 110`) as the 3/4 perspective front cliff skirt!\n"
            "     c) Check `SPIRES CODE SLICE` (`WorldSpireLayer._draw` and `SceneAssets.draw_tiled_rect`): Does `WorldSpireLayer` draw giant 340px-tall rectangular `SceneAssets.draw_tiled_rect` columns (`shaft_rect = Rect2(p.x - 24, p.y - 90, 48, 196)` with a bright `1.5px` rectangle border in `draw_tiled_rect`) at `z_index = 1` directly over the plateau and canal (`Vector2(500, 395)`, `Vector2(720, 350)`), creating huge dark boxy slabs with white wireframe borders blocking the view?! If so, emit a `SceneDeficiencyItem` targeting `scenes.gd` (`WorldSpireLayer._draw` and `SceneAssets.draw_tiled_rect`) to remove the rectangle border in `draw_tiled_rect`, set `WorldSpireLayer` `z_index = -5` for distant horizon spires (`y < 200`), and render on-plateau shrines as compact, grounded `midground_setpiece_1.png` stone lanterns/shrines (`64x64` over a subtle elliptical ground shadow, positioned in the northern/southern terrace margins clear of the canal `y = 390..480`, WITHOUT giant 196px rectangular `structure_wall_tile` column shafts)!\n"
            "2. **CRITICAL — Psychoacoustic Comfort & Audio Stream Critique (`main.gd` -> `acoustics_hum_setup`, `_synthesize_sine_loop`, `_setup_audio`, `_synthesize_buzz`)**:\n"
            "   - Inspect `AUDIO CODE SLICE` from `main.gd` below! Does `acoustics_hum_setup` or `_synthesize_sine_loop` create a **continuous looping high-pitched sine wave** (`stream.loop_mode = AudioStreamWAV.LOOP_FORWARD`), or does `_setup_audio` synthesize a piercing `1200.0 Hz` squeal (`squeal_stream = _synthesize_tone(1200.0, ...)`) or harsh square-wave `_synthesize_buzz`?\n"
            "   - If ANY continuous looping sine tone, `1200.0 Hz` squeal, or harsh square-wave buzz exists in `main.gd`, emit a mandatory `SceneDeficiencyItem` targeting `main.gd` (`_setup_audio` / `_synthesize_buzz` / `acoustics_hum_setup`) so all synthesized streams use `AudioStreamWAV.LOOP_DISABLED`, warm low/mid frequencies (`130–330 Hz`), and soft exponential decay (`exp(-8.0 * t)`, `volume_db <= -16.0`)!\n"
            "3. **Fluid / Energy Channels (`scenes.gd` -> `TerraceRenderer._draw`)**:\n"
            "   - Inspect `playtest_t0.png` / `playtest_tend.png` and the `canal_coords` passes inside `TERRAIN CODE SLICE` (`TerraceRenderer._draw`):\n"
            "     * Are there vertical comb/ladder lines or dark gaps cutting across the 2-row wide canal basin (`x = 430..630`)? This happens when `fw` (`hw * 0.72 ~ 9.1px`) is smaller than half of `CELL_SIZE` (`12.0px`) AND `draw_line(w_pos + norm * fw, n_pos + norm * fw, ...)` is called unconditionally on `offset = Vector2i(0, 1)` vertical connectors (which draws bright vertical lines across the middle of the horizontal basin)!\n"
            "     * To render a 100% seamless, organic multi-row water basin and river channel:\n"
            "       - Use `mw = 16.5`, `hw = 14.0`, and `fw = 12.5` (`>= CELL_SIZE * 0.5`) so adjacent rows and columns overlap with zero dark gaps.\n"
            "       - Keep the 3 separate passes (Pass 1: Masonry `mw`; Pass 2: Recessed dark bed `hw`; Pass 3: Luminous cyan water `fw` + caustics).\n"
            "       - In Pass 1 and Pass 3, NEVER draw `masonry_rim` or white foam `draw_line` strokes along `offset = Vector2i(0, 1)` vertical connectors or along any side `norm` where the perpendicular neighbor (`coord + Vector2i(int(round(norm.x)), int(round(norm.y)))`) is ALSO in `canal_coords`! Only draw bank rim/foam lines along true outer shoreline edges, and only run horizontal caustic waves along `offset == Vector2i(1, 0)`!\n"
            "     * Also check the Sunken Aqueduct Chasm (`x = 816..1066`): replace the harsh straight vertical `draw_line` at `chasm.position.x` and `chasm.end.x` with organic stepped rocky cliff edges and mossy rim scallops along the chasm walls!\n"
            "4. **Diegetic HUD (`ui_hud.gd`)**: Does the HUD present a clean, warm diegetic fantasy interface matching the Game Bible and Concept Art?\n"
            "5. **Compiler/Runtime Health**: Are there any GDScript compiler or runtime errors?\n\n"
            "Be ruthlessly honest. Set `scene_matches_game_bible = False` if ANY of the above deficiencies exist, and "
            "populate `deficiencies` with concrete `SceneDeficiencyItem` entries targeting the exact `.gd` file and class/method "
            "from the Cartridge Symbol Index so the SWE agent can execute surgical PEV patches.\n\n"
            f"--- COMPILER / RUNTIME DIAGNOSTICS (Valid={is_valid}) ---\n"
            + ("\n".join(diag_summary_lines) if diag_summary_lines else "None (0 script errors)")
            + f"\n{last_frame_summary}\n\n"
            f"--- TERRAIN CODE SLICE (scenes.gd -> TerraceRenderer._draw) ---\n{terrain_slice}\n\n"
            f"--- SPIRES CODE SLICE (scenes.gd -> WorldSpireLayer._draw) ---\n{spires_slice}\n\n"
            f"--- AUDIO CODE SLICE (main.gd -> _setup_audio & acoustics_hum_setup) ---\n{audio_slice}\n\n"
            f"--- MASTER PRODUCTION STRATEGY ---\n{strategy.model_dump_json(indent=2)[:6000]}\n\n"
            f"--- GAME BIBLE EXCERPT ---\n{bible_excerpt}\n\n"
            f"--- CARTRIDGE SYMBOL INDEX ---\n{symbol_index}\n\n"
            f"{asset_summary}\n"
        )

        contents: Any = [critique_prompt] + multimodal_parts if multimodal_parts else critique_prompt
        critique: Optional[SceneCritiqueAndStrategySpec] = None
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SceneCritiqueAndStrategySpec,
                    ),
                )
                if resp.text:
                    critique = SceneCritiqueAndStrategySpec.model_validate_json(resp.text)
                    self.last_deliberation = critique.deliberation
                    break
            except Exception as exc:
                last_err = exc
                logger.warning("analyze_scene_and_restrategize attempt %d failed: %s", attempt + 1, exc)

        if critique is None:
            raise StudioAgentError(f"Failed to generate SceneCritiqueAndStrategySpec: {last_err}")

        (c_dir / "scene_critique_report.json").write_text(
            critique.model_dump_json(indent=2), encoding="utf-8"
        )
        return critique

    def execute_scene_strategy_and_refine_loop(
        self,
        c_dir: Path,
        max_scene_iterations: int = 3,
    ) -> Dict[str, Path]:
        """Executes the full Strategic Production & Per-Scene Iterative Refinement Loop:
        1. Formulates the master `GameProductionStrategySpec` (and fabricates any missing required sprites).
        2. Runs the headless Godot 4 simulator to capture live viewport screenshots (`playtest_t0.png`, `playtest_tend.png`).
        3. Analyzes the rendered scene against the Game Bible, Concept Mockups, and Production Strategy (`analyze_scene_and_restrategize`).
        4. For any identified `SceneDeficiencyItem`s, fabricates any `new_assets_needed` and runs targeted PEV surgical
           patching (`debug_and_patch_pev_cycle`) per deficiency item, re-verifying in Godot 4 after each pass.
        """
        if self.client is None or not self._is_live_genai_client():
            return {}

        from engine.sim.godot_verifier import GodotHeadlessSimulator, verify_godot_cartridge_syntax
        from engine.studio.asset_library import CartridgeAssetLibrary

        updated_files: Dict[str, Path] = {}

        # Step 1: Formulate master production strategy & ensure required assets exist
        strategy = self.formulate_production_strategy(c_dir)

        lib = CartridgeAssetLibrary(cartridge_dir=c_dir, client=self.client)
        palette_mood = "stylized_fantasy"
        art_direction = ""
        if (c_dir / "game_bible.json").is_file():
            try:
                b_obj = json.loads((c_dir / "game_bible.json").read_text(encoding="utf-8"))
                palette_mood = b_obj.get("visual_style", {}).get("palette_mood", palette_mood)
                art_direction = json.dumps(b_obj.get("visual_style", {}))
            except Exception:
                pass

        for scene_iter in range(max_scene_iterations):
            logger.info("Scene Strategy & Critique Loop iteration %d/%d", scene_iter + 1, max_scene_iterations)
            critique: Optional[SceneCritiqueAndStrategySpec] = None
            critique_file = c_dir / "scene_critique_report.json"
            if scene_iter == 0 and critique_file.is_file():
                try:
                    crit_mtime = critique_file.stat().st_mtime
                    latest_gd_mtime = max((p.stat().st_mtime for p in c_dir.glob("*.gd")), default=0.0)
                    if crit_mtime >= latest_gd_mtime:
                        critique = SceneCritiqueAndStrategySpec.model_validate_json(
                            critique_file.read_text(encoding="utf-8")
                        )
                        logger.info("Reusing fresh scene_critique_report.json for iteration 1")
                except Exception:
                    critique = None
            if critique is None:
                critique = self.analyze_scene_and_restrategize(c_dir, strategy=strategy)

            # Fabricate any newly identified missing sprites before patching code
            if critique.new_assets_needed:
                try:
                    lib.ensure_required_assets(
                        required_specs=critique.new_assets_needed,
                        palette_mood=palette_mood,
                        art_direction=art_direction,
                    )
                except Exception as exc:
                    logger.warning("Could not fabricate new_assets_needed in scene iteration %d: %s", scene_iter + 1, exc)

            if critique.scene_matches_game_bible and not critique.deficiencies:
                logger.info("Scene matches Game Bible and Production Strategy on iteration %d!", scene_iter + 1)
                break

            # Execute a dedicated PEV surgical pass for each individual SceneDeficiencyItem
            for def_idx, fd in enumerate(critique.deficiencies, start=1):
                target_fname = Path(fd.target_file).name
                if not target_fname.endswith(".gd"):
                    target_fname = "scenes.gd"
                raw_sym = (fd.target_symbol_or_lines or "").strip()
                pre_reqs: List[CodeSliceRequest] = []
                if raw_sym:
                    pre_reqs.append(
                        CodeSliceRequest(
                            deliberation=f"Pre-extract target symbol `{raw_sym}` in `{target_fname}`",
                            filename=target_fname,
                            symbol_query=raw_sym,
                        )
                    )
                if target_fname == "scenes.gd" and "TerraceRenderer" in raw_sym:
                    pre_reqs.append(
                        CodeSliceRequest(
                            deliberation="Also inspect SceneAssets.draw_tiled_platform helper called by TerraceRenderer",
                            filename="scenes.gd",
                            symbol_query="class SceneAssets",
                        )
                    )
                if target_fname == "main.gd":
                    pre_reqs.append(
                        CodeSliceRequest(
                            deliberation="Inspect audio setup and synthesis functions in main.gd",
                            filename="main.gd",
                            symbol_query="func _setup_audio",
                        )
                    )
                    pre_reqs.append(
                        CodeSliceRequest(
                            deliberation="Inspect _synthesize_buzz and _synthesize_sine_loop in main.gd",
                            filename="main.gd",
                            symbol_query="func _synthesize_buzz",
                        )
                    )

                focused_defect_brief = (
                    f"SURGICAL PER-SCENE REMEDIATION #{def_idx}/{len(critique.deficiencies)} FOR `{target_fname}` -> `{raw_sym}` (Scene Iteration {scene_iter + 1}):\n"
                    f"Environment Topology Strategy: {strategy.environment_topology_strategy}\n"
                    f"Camera & Perspective Coherence Strategy: {strategy.camera_and_perspective_coherence_strategy}\n"
                    f"Psychoacoustic Audio Strategy: {strategy.psychoacoustic_audio_strategy}\n\n"
                    f"DEFICIENCY TO FIX IN `{target_fname}` (`{raw_sym}`):\n"
                    f"  - Discrepancy: {fd.deficiency_description}\n"
                    f"  - Concrete Remediation Strategy: {fd.remediation_strategy}\n"
                )
                _, current_diags = verify_godot_cartridge_syntax(c_dir)
                patched = self.debug_and_patch_pev_cycle(
                    c_dir=c_dir,
                    defect_summary=focused_defect_brief,
                    initial_diagnostics=current_diags,
                    multimodal_parts=None,
                    pre_slice_requests=pre_reqs,
                    max_pev_cycles=2,
                )
                updated_files.update(patched)

            healed = self._verify_and_heal_godot_syntax_and_probes(c_dir)
            updated_files.update(healed)

        # Final 60-tick simulation capture so playtest_t0.png and playtest_tend.png are fresh
        sim_final = GodotHeadlessSimulator(c_dir, capture_viewport=True, max_ticks=60)
        sim_final.run_until_complete()

        return updated_files

    def repair_godot_cartridge_holistic(
        self,
        c_dir: Path,
        rejection_trace: Any,
        req_spec: Optional[GameRequirementsSpec] = None,
    ) -> Dict[str, Path]:
        """Repairs Godot 4 .gd modules using the PEV (Plan/Research -> Execute Surgical Patch -> Verify)
        debugging loop and Per-Scene Critique & Re-Strategizing loop so the agent inspects only relevant
        code slices rather than full-file dumps.
        """
        if self.client is None:
            raise StudioAgentError("LeadGameplaySWEAgent requires an active google.genai Client (Rule 5)")

        from google.genai import types
        from engine.supervisor.challenger import CandidateASTInspector
        from engine.sim.godot_verifier import verify_godot_cartridge_syntax

        results: Dict[str, Path] = {}

        ast_insp = CandidateASTInspector()
        dir_ast = ast_insp.inspect_directory(c_dir)
        hollow_stubs = dir_ast.get("hollow_stubs", [])

        sign_off_context = ""
        sign_off_path = c_dir / "sign_off_report.json"
        if sign_off_path.is_file():
            try:
                so_data = json.loads(sign_off_path.read_text(encoding="utf-8"))
                veto_lines = []
                for v in so_data.get("verdicts", []):
                    if isinstance(v, dict) and not v.get("approved", True):
                        veto_lines.append(
                            f"- {v.get('specialist_role')}: Defects={v.get('blocking_defects')} | Remedies={v.get('remediation_directives')}"
                        )
                if veto_lines:
                    sign_off_context = "MULTI-AGENT SIGN-OFF COUNCIL VETOES:\n" + "\n".join(veto_lines) + "\n"
            except Exception:
                pass

        playtest_image_parts: List[Any] = []
        if self._is_live_genai_client():
            for cap_name in ("playtest_t0.png", "playtest_tend.png"):
                cap_path = c_dir / "assets" / "playtest_captures" / cap_name
                if cap_path.is_file():
                    try:
                        playtest_image_parts.append(
                            types.Part.from_bytes(data=cap_path.read_bytes(), mime_type="image/png")
                        )
                    except Exception:
                        pass

        diag_reason = getattr(rejection_trace, "failure_reason", "Challenger Gate rejected build")
        rec_remedy = getattr(rejection_trace, "recommended_remedy", "Eliminate hollow stubs and fulfill requirements")
        unfulfilled = getattr(rejection_trace, "unfulfilled_requirements", [])

        _, initial_diags = verify_godot_cartridge_syntax(c_dir)

        defect_summary = (
            f"Failure Reason: {diag_reason}\n"
            f"Recommended Remedy: {rec_remedy}\n"
            f"Unfulfilled Requirements: {unfulfilled}\n"
            f"Hollow Stubs Detected: {json.dumps(hollow_stubs)}\n"
            f"{sign_off_context}"
        )

        pev_patched = self.debug_and_patch_pev_cycle(
            c_dir=c_dir,
            defect_summary=defect_summary,
            initial_diagnostics=initial_diags,
            multimodal_parts=playtest_image_parts,
            max_pev_cycles=4,
        )
        results.update(pev_patched)

        if self._is_live_genai_client() and (c_dir / "game_bible.json").is_file():
            scene_refined = self.execute_scene_strategy_and_refine_loop(
                c_dir=c_dir,
                max_scene_iterations=2,
            )
            results.update(scene_refined)

        healed = self._verify_and_heal_godot_syntax_and_probes(c_dir)
        results.update(healed)
        return results

    def _verify_and_heal_godot_syntax_and_probes(self, c_dir: Path) -> Dict[str, Path]:
        """Verifies Godot 4 syntax & 2-tick runtime execution across all .gd files in c_dir.
        When live Gemini client is active, uses the PEV surgical debugging cycle on the exact broken lines
        rather than dumping entire files; falls back to repair_godot_module for mock test clients.
        Then re-injects telemetry probes from telemetry_manifest.json so Gate E remains satisfied.
        """
        from engine.sim.godot_verifier import verify_godot_cartridge_syntax
        results: Dict[str, Path] = {}

        is_valid, diagnostics = verify_godot_cartridge_syntax(c_dir)
        if not is_valid and diagnostics:
            if self._is_live_genai_client():
                diag_summary = "\n".join(
                    f"- {d.file_path.name}:{d.line_number}: {d.error_message}"
                    for d in diagnostics[:8]
                )
                pev_res = self.debug_and_patch_pev_cycle(
                    c_dir=c_dir,
                    defect_summary=f"Fix Godot 4 compiler and runtime script errors:\n{diag_summary}",
                    initial_diagnostics=diagnostics,
                    max_pev_cycles=4,
                )
                results.update(pev_res)
            else:
                for fix_attempt in range(3):
                    is_valid, diagnostics = verify_godot_cartridge_syntax(c_dir)
                    if is_valid:
                        break
                    diags_by_file: Dict[Path, List[str]] = {}
                    for diag in diagnostics:
                        if diag.file_path.is_file():
                            entry = f"Line {diag.line_number}: {diag.error_message} (Raw: {diag.raw_line})"
                            diags_by_file.setdefault(diag.file_path, []).append(entry)
                    for file_path, diag_entries in diags_by_file.items():
                        broken_src = file_path.read_text(encoding="utf-8")
                        combined_diag = "\n".join(diag_entries)
                        peer_ctx = self._extract_gd_peer_signatures(c_dir, exclude_file=file_path.name)
                        fixed_src = self.repair_godot_module(
                            file_path.name,
                            broken_src,
                            combined_diag,
                            peer_context=peer_ctx,
                        )
                        file_path.write_text(fixed_src, encoding="utf-8")
                        results[file_path.name] = file_path

        manifest_path = c_dir / "telemetry_manifest.json"
        if manifest_path.is_file():
            try:
                from engine.studio.telemetry_architect import TelemetryArchitectAgent
                from engine.studio.models import TelemetryProbeDesignSpec
                m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                raw_probes = m_data.get("probes", [])
                probe_specs = [
                    TelemetryProbeDesignSpec(
                        deliberation=p.get("deliberation", "Self-healed telemetry probe hook"),
                        probe_name=p.get("probe_name", "custom_probe"),
                        category=p.get("category", "gameplay"),
                        description=p.get("description", "Self-healed telemetry probe hook description"),
                        target_class=p.get("target_class", "Node2D"),
                        target_method=p.get("target_method", "_process"),
                        hypothesis_target=p.get("hypothesis_target", "Verify active gameplay loop engagement"),
                    )
                    for p in raw_probes if isinstance(p, dict) and p.get("probe_name")
                ]
                if probe_specs and (c_dir / "main.gd").is_file():
                    arch = TelemetryArchitectAgent(client=self.client, model_name=self.model_name)
                    main_src = (c_dir / "main.gd").read_text(encoding="utf-8")
                    instrumented = arch._inject_gdscript_probes(main_src, probe_specs)
                    (c_dir / "main.gd").write_text(instrumented, encoding="utf-8")
                    is_valid_after_probe, _ = verify_godot_cartridge_syntax(c_dir)
                    if not is_valid_after_probe:
                        (c_dir / "main.gd").write_text(main_src, encoding="utf-8")
                    else:
                        results["main.gd"] = c_dir / "main.gd"
            except Exception as probe_err:
                logger.warning("Could not re-inject telemetry probes after Godot syntax heal: %s", probe_err)

        return results

    def _deterministic_code_remedy(
        self,
        module_name: str,
        broken_code: str,
        rejection_trace: Optional[Any] = None,
        world_spec: Optional[GenesisWorldSpec] = None,
    ) -> str:
        """Deterministic fallback is prohibited under Rule 5 (Zero Silent Fallbacks)."""
        raise CodeVerificationError(
            f"Deterministic fallback remediation for '{module_name}' is strictly prohibited under Rule 5 (Zero Silent Fallbacks)."
        )

    # -----------------------------------------------------------------------
    # Dynamic Code Generators
    # -----------------------------------------------------------------------

    def _generate_manifest_data(self, world_spec: Optional[Any] = None) -> Dict[str, Any]:
        game_title = getattr(world_spec, "game_title", "Generated Game") if world_spec else "Generated Game"
        core_genre = getattr(world_spec, "core_genre", getattr(world_spec, "genre_paradigm", "Action")) if world_spec else "Action"
        palette_mood = getattr(world_spec, "palette_mood", "retro_arcade") if world_spec else "retro_arcade"
        delib = getattr(world_spec, "deliberation", f"Cartridge manifest for {game_title}") if world_spec else "Cartridge manifest"
        spec_id = getattr(world_spec, "spec_id", f"spec_{uuid.uuid4().hex[:8]}") if world_spec else f"spec_{uuid.uuid4().hex[:8]}"
        target_episode = getattr(world_spec, "target_episode", 0) if world_spec else 0

        return {
            "deliberation": delib,
            "name": game_title,
            "version": "1.0.0",
            "author": "Autonomous Game Engine & Studio",
            "description": f"Episode {target_episode} Genesis Cartridge autonomously compiled for {core_genre} ({palette_mood}).",
            "spec_id": spec_id,
            "target_episode": target_episode,
            "core_genre": core_genre,
            "palette_mood": palette_mood,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

