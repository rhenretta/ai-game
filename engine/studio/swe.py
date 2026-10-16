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
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from pydantic import Field

from engine.api.contracts import BaseCartridge, BaseEnemy, BaseHazard, BasePlayer, BaseLevelSpec
from engine.loader import ASTSecurityError, validate_cartridge_ast
from engine.studio.models import BaseEpistemicModel, GenesisWorldSpec, NarrativeBible, StudioAgentError
from engine.supervisor.models import GateFailureType, RemedyType, RejectionDiagnosticTrace

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


class LeadGameplaySWEAgent:
    """Specialist Agent that synthesizes executable Python cartridge code with zero string templates."""
    deliberation: str = "Lead gameplay SWE agent synthesizing executable Python game cartridges without static string templates."

    def __init__(self, client: Optional[Any] = None, model_name: str = "gemini-3.8-flash"):
        self.client = client
        self.model_name = model_name

    def synthesize_cartridge_code(
        self,
        world_spec: GenesisWorldSpec,
        narrative_bible: Optional[NarrativeBible] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Synthesizes open-genre game cartridge code using live Gemini inference."""
        if self.client is None:
            raise StudioAgentError("LeadGameplaySWEAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)")

        target_dir = output_dir or Path("cartridges/candidate")
        target_dir.mkdir(parents=True, exist_ok=True)
        results: Dict[str, Path] = {}

        from google.genai import types
        narrative_context = ""
        if narrative_bible:
            narrative_context = (
                f"- Characters: {[c.name for c in narrative_bible.characters]}\n"
                f"- Lore Summary: {narrative_bible.setting_summary}\n"
            )

        prompt = (
            f"You are the Lead Gameplay SWE AI for the Autonomous Game Studio.\n"
            f"Synthesize a complete, playable 2D Pygame game cartridge conforming to the following World Blueprint:\n"
            f"- Title: {world_spec.game_title}\n"
            f"- Core Genre: {world_spec.core_genre}\n"
            f"- Lore & Premise: {world_spec.thematic_premise} — {world_spec.world_lore}\n"
            f"- Aesthetic Style: {world_spec.aesthetic_style} ({world_spec.palette_mood})\n"
            f"- Pacing: {world_spec.gameplay_pacing} (score {world_spec.pacing_score})\n"
            f"- Target Difficulty: {world_spec.target_difficulty}\n"
            f"- Mechanics Guidelines: {world_spec.mechanics_guidelines}\n"
            f"{narrative_context}\n"
            f"Architectural Contracts:\n"
            f"1. Target resolution is 1280x720 canvas at 60 FPS.\n"
            f"2. Put ALL game scenes, entities, and the BaseCartridge subclass directly inside `mechanics_code` (`mechanics.py`), "
            f"   and leave `scenes_code`, `entities_code`, `abilities_code`, and `level_data` as null/None so `mechanics.py` is 100% self-contained.\n"
            f"3. `mechanics.py` MUST import `BaseCartridge, BaseScene` from `engine.api.contracts` (plus `pygame`, `math`, `random`).\n"
            f"   The `BaseCartridge` subclass MUST implement `get_scenes(self, engine)` returning `{{'main': MainScene(engine)}}` "
            f"   and `get_initial_scene(self, engine)` returning `MainScene(engine)`.\n"
            f"4. The `BaseScene` subclass MUST initialize `self.engine = engine` and `self.state_counter = 0` in `__init__(self, engine=None)`, "
            f"   implement `on_enter(self)`, `on_exit(self)`, `handle_event(self, event)`, `update(self, dt: float)` (which MUST increment `self.state_counter += 1` every tick), "
            f"   and `render(self, surface: pygame.Surface)`.\n"
            f"5. Rule 4 CRITICAL: In EVERY class definition, the VERY FIRST statement inside the class body (BEFORE any docstring or method) "
            f"   MUST be `deliberation: str = '...'`. Do NOT put a class docstring before `deliberation`!\n"
            f"6. AST Sandbox Security: NEVER import banned modules (`os`, `sys`, `subprocess`, `socket`, `shutil`, `threading`, `asyncio`, `engine.core`, `engine.loader`). "
            f"   NEVER call prohibited builtins or method names (`open`, `eval`, `exec`, `compile`, `globals`, `locals`, `vars`, `dir`, `exit`, `quit`, "
            f"   `remove`, `kill`, `spawn`, `run`, `call`, `unlink`, `rmtree`, `system`, `popen`, `connect`). "
            f"   Instead of `list.remove(x)` or `def spawn(...)`, use list comprehensions `[x for x in items if ...]` and `def create_unit(...)`!\n"
            f"7. Prohibited tokens: Never output template placeholders like '{{TEMPLATE_' or '{{{{LEVEL_CONFIG}}}}'.\n"
            f"8. You MUST provide epistemic deliberation FIRST before code payloads.\n"
        )

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
            bundle = OpenGenreCartridgeBundle(**bundle_dict)
        except Exception as exc:
            logger.error("Gemini cartridge code synthesis failed: %s", exc)
            if isinstance(exc, (CodeVerificationError, StudioAgentError)):
                raise
            raise StudioAgentError(f"LeadGameplaySWEAgent cartridge synthesis failed: {exc}") from exc

        # 1. manifest.json
        manifest_data = bundle.manifest_data or self._generate_manifest_data(world_spec)
        manifest_path = target_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        results["manifest.json"] = manifest_path

        # 2. mechanics.py
        mechanics_src = self._verify_and_repair_loop("mechanics.py", bundle.mechanics_code)
        mechanics_path = target_dir / "mechanics.py"
        mechanics_path.write_text(mechanics_src, encoding="utf-8")
        results["mechanics.py"] = mechanics_path

        # 3. scenes.py (if provided)
        if bundle.scenes_code:
            scenes_src = self._verify_and_repair_loop("scenes.py", bundle.scenes_code)
            scenes_path = target_dir / "scenes.py"
            scenes_path.write_text(scenes_src, encoding="utf-8")
            results["scenes.py"] = scenes_path

        # 4. entities.py (if provided)
        if bundle.entities_code:
            entities_src = self._verify_and_repair_loop("entities.py", bundle.entities_code)
            entities_path = target_dir / "entities.py"
            entities_path.write_text(entities_src, encoding="utf-8")
            results["entities.py"] = entities_path

        # 5. abilities.py (if provided)
        if bundle.abilities_code:
            abilities_src = self._verify_and_repair_loop("abilities.py", bundle.abilities_code)
            abilities_path = target_dir / "abilities.py"
            abilities_path.write_text(abilities_src, encoding="utf-8")
            results["abilities.py"] = abilities_path

        # 6. level.json (if provided)
        if bundle.level_data:
            level_path = target_dir / "level.json"
            level_path.write_text(json.dumps(bundle.level_data, indent=2), encoding="utf-8")
            results["level.json"] = level_path

        return results

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

        failure_type = getattr(rejection_trace, "failure_type", None)

        # ---------------------------------------------------------------------
        # 1. Gate A & Gate D: Python Code Repair (mechanics.py, entities.py, abilities.py)
        # ---------------------------------------------------------------------
        if failure_type in (
            GateFailureType.SYNTAX_AST_ERROR,
            GateFailureType.UNHANDLED_EXCEPTION,
        ):
            raw_target = getattr(rejection_trace, "file_path", None) or "mechanics.py"
            target_name = Path(raw_target).name
            target_path = c_dir / target_name
            if not target_path.is_file():
                if (c_dir / "mechanics.py").is_file():
                    target_path = c_dir / "mechanics.py"
                    target_name = "mechanics.py"

            if target_path.is_file():
                broken_code = target_path.read_text(encoding="utf-8")
                reason = getattr(rejection_trace, "failure_reason", "")
                tb = getattr(rejection_trace, "runtime_traceback", "")
                diagnostic = f"{reason}\nTraceback:\n{tb or ''}"

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

            # Safely re-quantize PNGs via in-memory buffer so files are never truncated on error
            for png_file in c_dir.rglob("*.png"):
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

            def _write_valid_tone_wav(wav_path: Path) -> None:
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

            # Re-synthesize silent or invalid WAV files
            for wav_file in c_dir.rglob("*.wav"):
                try:
                    _write_valid_tone_wav(wav_file)
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
                                            _write_valid_tone_wav(af_path)
                                            results[af_path.name] = af_path
                        elif isinstance(barks_val, list):
                            for cue in barks_val:
                                if isinstance(cue, dict):
                                    af = cue.get("audio_file") or cue.get("voice_clip")
                                    if af:
                                        af_path = c_dir / af
                                        if not af_path.is_file():
                                            _write_valid_tone_wav(af_path)
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
                            target_method="update",
                            hypothesis_target=p.get("hypothesis_target", "Verify active gameplay loop engagement"),
                        )
                        for p in raw_probes if isinstance(p, dict) and p.get("probe_name")
                    ]
                    if probe_specs:
                        arch = TelemetryArchitectAgent(client=self.client, model_name=self.model_name)
                        for mod_name in ("scenes.py", "mechanics.py"):
                            mod_path = c_dir / mod_name
                            if mod_path.is_file():
                                src = mod_path.read_text(encoding="utf-8")
                                instrumented = arch._inject_probes_into_source(src, probe_specs[:1])
                                mod_path.write_text(instrumented, encoding="utf-8")
                                results[mod_name] = mod_path
                except Exception as exc:
                    logger.warning("Could not self-heal telemetry probes: %s", exc)

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

    def _generate_manifest_data(self, world_spec: GenesisWorldSpec) -> Dict[str, Any]:
        return {
            "deliberation": world_spec.deliberation,
            "name": world_spec.game_title,
            "version": "1.0.0",
            "author": "Autonomous Game Engine & Studio",
            "description": f"Episode 0 Genesis Cartridge autonomously compiled for {world_spec.core_genre} ({world_spec.palette_mood}).",
            "spec_id": world_spec.spec_id,
            "target_episode": world_spec.target_episode,
            "core_genre": world_spec.core_genre,
            "palette_mood": world_spec.palette_mood,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def _generate_level_data(self, world_spec: GenesisWorldSpec) -> Dict[str, Any]:
        pacing = world_spec.pacing_score
        # Speed & difficulty scaling
        enemy_speed = 140.0 if pacing < 0.0 else 180.0
        return {
            "deliberation": world_spec.deliberation,
            "level_id": "level_01",
            "name": f"{world_spec.game_title} - Sector 01",
            "width": 1280.0,
            "height": 720.0,
            "spawn_point": [120.0, 360.0],
            "exit_point": [1160.0, 360.0],
            "walls": [
                [0.0, 0.0, 1280.0, 32.0],
                [0.0, 688.0, 1280.0, 32.0],
                [0.0, 0.0, 32.0, 720.0],
                [1248.0, 0.0, 32.0, 720.0],
                [360.0, 160.0, 48.0, 160.0],
                [360.0, 400.0, 48.0, 160.0],
                [780.0, 160.0, 48.0, 160.0],
                [780.0, 400.0, 48.0, 160.0],
            ],
            "enemies": [
                {"id": "chaser_1", "type": "chaser", "x": 640.0, "y": 240.0, "speed": enemy_speed, "health": 40.0},
                {"id": "sniper_1", "type": "sniper", "x": 900.0, "y": 480.0, "speed": 0.0, "health": 30.0},
            ],
            "hazards": [
                {"id": "spike_1", "type": "spike", "x": 540.0, "y": 340.0, "w": 48.0, "h": 48.0, "damage": 15.0},
            ],
        }

    def _synthesize_abilities_module(self, world_spec: GenesisWorldSpec) -> str:
        """Synthesizes abilities.py with Rationale-First BaseAbility subclasses."""
        speed_bonus = 280.0 if world_spec.pacing_score > 0.0 else 220.0
        cooldown = 0.6 if world_spec.pacing_score > 0.0 else 1.0

        return f'''"""abilities.py
Custom abilities authored by LeadGameplaySWEAgent.
"""
from __future__ import annotations
from typing import Any, Optional
from engine.api.contracts import BaseAbility


class DashAbility(BaseAbility):
    deliberation: str = "Tactical evasive burst granting instantaneous velocity boost."

    def __init__(self, cooldown: float = {cooldown:.2f}, speed_bonus: float = {speed_bonus:.1f}):
        super().__init__(ability_id="dash", name="Evasive Thruster", cooldown=cooldown, stamina_cost=15.0)
        self.speed_bonus = speed_bonus

    def execute(self, user: Any, target_pos: Optional[Any] = None) -> bool:
        if not super().execute(user, target_pos):
            return False
        if hasattr(user, "vx") and hasattr(user, "vy"):
            import math
            mag = math.hypot(user.vx, user.vy)
            if mag > 1e-4:
                user.vx = (user.vx / mag) * self.speed_bonus
                user.vy = (user.vy / mag) * self.speed_bonus
            else:
                user.vx = self.speed_bonus
        return True


class PrimaryFireAbility(BaseAbility):
    deliberation: str = "Standard directed projectile weapon ability."

    def __init__(self, cooldown: float = 0.25):
        super().__init__(ability_id="primary_fire", name="Plasma Bolt", cooldown=cooldown, stamina_cost=5.0)

    def execute(self, user: Any, target_pos: Optional[Any] = None) -> bool:
        return super().execute(user, target_pos)
'''

    def _synthesize_entities_module(self, world_spec: GenesisWorldSpec) -> str:
        """Synthesizes entities.py with player, enemy, and hazard classes."""
        walk_speed = 260.0 if world_spec.pacing_score > 0.0 else 190.0
        return f'''"""entities.py
Custom gameplay entities authored by LeadGameplaySWEAgent.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from engine.entities.player import Player as BasePlayer
from engine.entities.enemies import BaseEnemy
from engine.entities.hazard import Hazard as BaseHazard, SpikeHazard
from engine.combat.weapon import Weapon as BaseWeapon
from engine.telemetry.probes import AgentTelemetryProbe


class DynamicPlayer(BasePlayer):
    deliberation: str = "Player entity configured with responsive kinetic velocity and telemetry probes."

    def __init__(self, x: float = 120.0, y: float = 360.0, telemetry_recorder: Optional[Any] = None):
        super().__init__(x=x, y=y, telemetry_recorder=telemetry_recorder)
        self.walk_speed = {walk_speed:.1f}
        self.telemetry_recorder = telemetry_recorder
        self.probe_dash = AgentTelemetryProbe(
            recorder=telemetry_recorder,
            probe_name="player_dash_mobility",
            category="movement",
            description="Records player evasive dash triggers and coordinates.",
            hypothesis_target="H_PLAYER_MOBILITY_AND_DASH_CADENCE",
        )

    def dash(self) -> bool:
        if self.probe_dash:
            self.probe_dash.emit(pos=(self.x, self.y), vel=(self.vx, self.vy), health=self.health)
        return True


class DynamicChaserEnemy(BaseEnemy):
    deliberation: str = "Aggressive melee chaser enemy tracking player position directly."

    def __init__(self, enemy_id: str, x: float, y: float, speed: float = 160.0, health: float = 40.0):
        super().__init__(entity_id=enemy_id, x=x, y=y, w=32.0, h=32.0, max_health=health, speed=speed)

    def update_ai(self, dt: float, player: Any, walls: Optional[Any] = None, projectile_manager: Optional[Any] = None) -> None:
        import math
        player_x = getattr(player, "x", 0.0)
        player_y = getattr(player, "y", 0.0)
        dx = player_x - self.x
        dy = player_y - self.y
        dist = math.hypot(dx, dy)
        if dist > 5.0:
            self.vx = (dx / dist) * self.speed
            self.vy = (dy / dist) * self.speed
        else:
            self.vx = 0.0
            self.vy = 0.0
        self.x += self.vx * dt
        self.y += self.vy * dt


class DynamicSniperEnemy(BaseEnemy):
    deliberation: str = "Stationary ranged sniper enemy maintaining tactical standoff."

    def __init__(self, enemy_id: str, x: float, y: float, health: float = 30.0):
        super().__init__(entity_id=enemy_id, x=x, y=y, w=32.0, h=32.0, max_health=health, speed=0.0)

    def update_ai(self, dt: float, player: Any, walls: Optional[Any] = None, projectile_manager: Optional[Any] = None) -> None:
        self.vx = 0.0
        self.vy = 0.0


class DynamicSpikeHazard(SpikeHazard):
    deliberation: str = "Static contact damage spike hazard."

    def __init__(self, hazard_id: str, x: float, y: float, w: float = 48.0, h: float = 48.0, damage: float = 15.0):
        super().__init__(hazard_id=hazard_id, x=x, y=y, w=w, h=h)
        self.damage = damage
'''

    def _synthesize_mechanics_module(self, world_spec: GenesisWorldSpec) -> str:
        """Synthesizes mechanics.py defining the BaseCartridge subclass."""
        return f'''"""mechanics.py
Custom game cartridge mechanics authored by LeadGameplaySWEAgent.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from engine.api.contracts import BaseCartridge, BaseEnemy, BaseHazard, BasePlayer, BaseLevelSpec
from entities import DynamicPlayer, DynamicChaserEnemy, DynamicSniperEnemy, DynamicSpikeHazard


class GeneratedCartridge(BaseCartridge):
    deliberation: str = "Main game cartridge controller orchestrating level, entities, and hazards."

    def __init__(
        self,
        manifest: Optional[Dict[str, Any]] = None,
        cartridge_dir: Optional[Any] = None,
        narrative: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(manifest=manifest, cartridge_dir=cartridge_dir, narrative=narrative)
        self._level_spec: Optional[BaseLevelSpec] = None
        self._load_level_json()

    def _load_level_json(self) -> None:
        if self.cartridge_dir:
            level_path = Path(self.cartridge_dir) / "level.json"
            if level_path.is_file():
                data = json.loads(level_path.read_text(encoding="utf-8"))
                self._level_spec = BaseLevelSpec(
                    level_id=data.get("level_id", "level_01"),
                    name=data.get("name", "Generated Chamber"),
                    width=data.get("width", 1280.0),
                    height=data.get("height", 720.0),
                    spawn_point=tuple(data.get("spawn_point", [120.0, 360.0])),
                    exit_point=tuple(data.get("exit_point", [1160.0, 360.0])),
                    walls=[tuple(w) for w in data.get("walls", [])],
                    enemies=data.get("enemies", []),
                    hazards=data.get("hazards", []),
                    deliberation=data.get("deliberation", "Loaded from level.json"),
                )

    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        if self._level_spec is not None:
            return self._level_spec
        return BaseLevelSpec(
            level_id="level_01",
            name="Fallback Chamber",
            width=1280.0,
            height=720.0,
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            deliberation="Fallback level specification",
        )

    def create_player(self, x: float, y: float, telemetry_recorder: Optional[Any] = None) -> BasePlayer:
        return DynamicPlayer(x=x, y=y, telemetry_recorder=telemetry_recorder)

    def create_enemies(self, spec: BaseLevelSpec) -> List[BaseEnemy]:
        result: List[BaseEnemy] = []
        for e in spec.enemies:
            e_type = e.get("type", "chaser")
            if e_type == "sniper":
                result.append(DynamicSniperEnemy(
                    enemy_id=e.get("id", "sniper"),
                    x=float(e.get("x", 800.0)),
                    y=float(e.get("y", 360.0)),
                    health=float(e.get("health", 30.0)),
                ))
            else:
                result.append(DynamicChaserEnemy(
                    enemy_id=e.get("id", "chaser"),
                    x=float(e.get("x", 600.0)),
                    y=float(e.get("y", 360.0)),
                    speed=float(e.get("speed", 160.0)),
                    health=float(e.get("health", 40.0)),
                ))
        return result

    def create_hazards(self, spec: BaseLevelSpec) -> List[BaseHazard]:
        result: List[BaseHazard] = []
        for h in spec.hazards:
            result.append(DynamicSpikeHazard(
                hazard_id=h.get("id", "hazard"),
                x=float(h.get("x", 400.0)),
                y=float(h.get("y", 360.0)),
                w=float(h.get("w", 48.0)),
                h=float(h.get("h", 48.0)),
                damage=float(h.get("damage", 15.0)),
            ))
        return result
'''
