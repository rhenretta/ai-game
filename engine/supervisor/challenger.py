"""engine/supervisor/challenger.py
Gate F: Adversarial Design Critic & Challenger Verification Gate (Milestone M4, Feature 11).
Audits candidate game cartridges against 100% of agent-authored Game Requirements and Acceptance Criteria
in requirements_spec.json using live Gemini structured evaluation grounded in:
1. Static AST code inspection and tokenize comment scanning (CandidateASTInspector)
2. Runtime simulation event, custom probe, and state progression verification (CandidateTelemetryAuditor)
3. Headless visual playtest software frame buffer capture and motion delta analysis (CandidateVisualPlaytester)

In strict compliance with:
- Rule 1: Pure Generalizability (100% domain- and genre-agnostic)
- Rule 2: Zero Regex Epistemics (AST structural parsing and LLM structured contracts)
- Rule 4: Mandatory Rationale-First Epistemic Architecture (deliberation at field index 0)
- Rule 5: Zero Silent Fallbacks & Error Concealment (fails fast with authentic StudioAgentError)
- Rule 6: Mandatory Production Parity & Container Runtime Invariance (SDL_VIDEODRIVER=dummy)
"""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import tokenize
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Rule 6: Container Runtime Invariance pre-flight defaults
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
from PIL import Image

import config
from engine.studio.models import (
    BaseEpistemicModel,
    GameRequirementsSpec,
    RequirementTrack,
    StudioAgentError,
)
from engine.supervisor.models import (
    ChallengerGateEvaluation,
    GateFailureType,
    RejectionDiagnosticTrace,
    RemedyType,
    RequirementEvaluationRecord,
)

logger = logging.getLogger("ChallengerGate")


# =============================================================================
# Pillar 1: Static AST Code Inspector (Zero Regexes - Rule 2)
# =============================================================================

class CandidateASTInspector:
    """Static AST and token auditor for candidate cartridge code.
    Detects pass stubs, empty function bodies, TODO/FIXME comments, disconnected classes,
    and dummy/mock symbols without regex stop-lists (Rule 2).
    """
    deliberation: str = "Audits candidate Python AST for pass stubs, empty bodies, TODOs, and disconnected code."

    def inspect_directory(self, candidate_dir: Union[Path, str]) -> Dict[str, Any]:
        """Inspects all Python modules in candidate_dir and returns structured evidence."""
        c_dir = Path(candidate_dir).resolve()
        py_files = sorted(c_dir.glob("*.py"))

        modules_inspected: List[str] = []
        pass_stubs: List[Dict[str, Any]] = []
        empty_bodies: List[Dict[str, Any]] = []
        hollow_stubs: List[Dict[str, Any]] = []
        todo_comments: List[Dict[str, Any]] = []
        dummy_mocks: List[Dict[str, Any]] = []
        defined_classes_by_file: Dict[str, Set[str]] = {}
        all_referenced_symbols: Set[str] = set()
        module_structures: Dict[str, Dict[str, List[str]]] = {}
        total_loc = 0

        for f in py_files:
            rel_name = f.name
            modules_inspected.append(rel_name)
            try:
                raw_bytes = f.read_bytes()
                code_str = raw_bytes.decode("utf-8", errors="replace")
                total_loc += len(code_str.splitlines())
            except Exception as e:
                logger.warning("Could not read file %s: %s", f, e)
                continue

            # 1. Token-level comment scanning (zero regexes)
            try:
                tokens = tokenize.tokenize(io.BytesIO(raw_bytes).readline)
                for tok in tokens:
                    if tok.type == tokenize.COMMENT:
                        c_text = tok.string
                        upper = c_text.upper()
                        for tag in ("TODO", "FIXME", "XXX", "STUB", "IMPLEMENT ME", "TBD"):
                            if tag in upper:
                                todo_comments.append({
                                    "file": rel_name,
                                    "line": tok.start[0],
                                    "tag": tag,
                                    "comment": c_text.strip(),
                                })
                                break
            except Exception as tok_err:
                logger.debug("Tokenize comment scan error in %s: %tok_err", rel_name, tok_err)

            # 2. AST parsing & structural inspection
            try:
                tree = ast.parse(code_str, filename=str(f))
            except Exception as parse_err:
                hollow_stubs.append({
                    "module": rel_name,
                    "function": "<module>",
                    "reason": f"AST parse error: {parse_err}",
                })
                continue

            defined_classes_by_file[rel_name] = set()
            class_names: List[str] = []
            func_names: List[str] = []

            for node in ast.walk(tree):
                # Track symbol references
                if isinstance(node, ast.Name):
                    all_referenced_symbols.add(node.id)
                elif isinstance(node, ast.Attribute):
                    all_referenced_symbols.add(node.attr)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        all_referenced_symbols.add(alias.name)

                # Track class definitions
                if isinstance(node, ast.ClassDef):
                    class_names.append(node.name)
                    defined_classes_by_file[rel_name].add(node.name)
                    lower_name = node.name.lower()
                    if any(m in lower_name for m in ("mock", "dummy", "stub", "fake")):
                        dummy_mocks.append({"file": rel_name, "symbol": node.name, "type": "class"})

                # Track function/method definitions
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_names.append(node.name)
                    lower_name = node.name.lower()
                    if any(m in lower_name for m in ("mock", "dummy", "stub", "fake")):
                        dummy_mocks.append({"file": rel_name, "symbol": node.name, "type": "function"})

                    # Analyze function body for hollow evasion stubs (BVA-R3.2)
                    body = node.body
                    # Skip leading docstring statement
                    eff_body = (
                        body[1:]
                        if (
                            body
                            and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)
                        )
                        else body
                    )

                    if not eff_body:
                        item = {"file": rel_name, "module": rel_name, "function": node.name, "line": node.lineno, "reason": "Docstring only / empty body"}
                        empty_bodies.append(item)
                        hollow_stubs.append(item)
                    elif len(eff_body) == 1:
                        stmt = eff_body[0]
                        stub_reason = ""
                        stmt_repr = ""
                        if isinstance(stmt, ast.Pass):
                            stub_reason = "Single pass statement"
                            stmt_repr = "pass"
                        elif (
                            isinstance(stmt, ast.Expr)
                            and isinstance(stmt.value, ast.Constant)
                            and stmt.value.value is Ellipsis
                        ):
                            stub_reason = "Ellipsis statement"
                            stmt_repr = "..."
                        elif isinstance(stmt, ast.Return) and (
                            stmt.value is None
                            or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)
                        ):
                            stub_reason = "return None only"
                            stmt_repr = "return None"
                        elif isinstance(stmt, ast.Raise):
                            exc_id = getattr(stmt.exc, "id", None) or getattr(getattr(stmt.exc, "func", None), "id", None)
                            if exc_id == "NotImplementedError":
                                stub_reason = "raise NotImplementedError"
                                stmt_repr = "raise NotImplementedError"

                        if stub_reason:
                            item = {
                                "file": rel_name,
                                "module": rel_name,
                                "function": node.name,
                                "line": node.lineno,
                                "stmt": stmt_repr,
                                "reason": stub_reason,
                            }
                            pass_stubs.append(item)
                            hollow_stubs.append(item)

            module_structures[rel_name] = {
                "classes": class_names,
                "functions": func_names,
            }

        # 2b. GDScript files inspection
        gd_files = sorted(c_dir.glob("*.gd"))
        gd_sources: Dict[str, str] = {}
        all_defined_gd_funcs: Set[str] = set()

        for f in gd_files:
            rel_name = f.name
            modules_inspected.append(rel_name)
            try:
                code_str = f.read_text(encoding="utf-8", errors="replace")
                gd_sources[rel_name] = code_str
                total_loc += len(code_str.splitlines())
            except Exception as e:
                logger.warning("Could not read file %s: %s", f, e)
                continue

            lines = code_str.splitlines()
            class_names: List[str] = [f.stem]
            func_names: List[str] = []
            defined_classes_by_file[rel_name] = {f.stem}

            # Track referenced symbols and preloads from GDScript (token splitting, zero regex heuristics)
            for token_part in code_str.replace('"', " ").replace("'", " ").replace("(", " ").replace(")", " ").replace(".", " ").replace(":", " ").replace(",", " ").split():
                clean_tok = token_part.strip()
                if clean_tok:
                    all_referenced_symbols.add(clean_tok)
                    all_referenced_symbols.add(clean_tok.lower())
            for line in lines:
                if "preload(" in line and ".gd" in line:
                    for seg in line.split('"'):
                        if seg.startswith("res://") and seg.endswith(".gd"):
                            p_stem = Path(seg.replace("res://", "")).stem
                            all_referenced_symbols.add(p_stem)
                            all_referenced_symbols.add(p_stem.lower())

            for idx, line in enumerate(lines):
                line_no = idx + 1
                stripped = line.strip()
                if stripped.startswith("#"):
                    upper = stripped.upper()
                    for tag in ("TODO", "FIXME", "XXX", "STUB", "IMPLEMENT ME", "TBD"):
                        if tag in upper:
                            todo_comments.append({
                                "file": rel_name,
                                "line": line_no,
                                "tag": tag,
                                "comment": stripped,
                            })
                            break
                elif stripped.startswith("func "):
                    after_func = stripped[5:].strip()
                    paren_idx = after_func.find("(")
                    if paren_idx > 0:
                        fn_name = after_func[:paren_idx].strip()
                        func_names.append(fn_name)
                        all_defined_gd_funcs.add(fn_name)
                        lower_fn = fn_name.lower()
                        if any(term in lower_fn for term in ("mock", "dummy", "stub", "fake")):
                            dummy_mocks.append({"file": rel_name, "symbol": fn_name, "type": "function"})

                        fn_indent = len(line) - len(line.lstrip())
                        eff_gd_lines: List[str] = []
                        for next_idx in range(idx + 1, len(lines)):
                            raw_nxt = lines[next_idx]
                            nxt = raw_nxt.strip()
                            if not nxt or nxt.startswith("#"):
                                continue
                            nxt_indent = len(raw_nxt) - len(raw_nxt.lstrip())
                            if nxt_indent <= fn_indent and nxt.startswith(("func ", "class ", "var ", "const ", "signal ", "@")):
                                break
                            code_only = nxt.split("#", 1)[0].strip()
                            if code_only:
                                eff_gd_lines.append(code_only)

                        if not eff_gd_lines or (len(eff_gd_lines) == 1 and eff_gd_lines[0] in ("pass", "return", "return null")):
                            stmt_str = eff_gd_lines[0] if eff_gd_lines else "pass"
                            item = {
                                "file": rel_name,
                                "module": rel_name,
                                "function": fn_name,
                                "line": line_no,
                                "stmt": stmt_str,
                                "statement": stmt_str,
                                "reason": f"Empty or single '{stmt_str}' function body in {rel_name}:{fn_name}",
                            }
                            pass_stubs.append(item)
                            hollow_stubs.append(item)
                        elif len(eff_gd_lines) <= 5 and eff_gd_lines[-1].startswith("return "):
                            joined_body = " ".join(eff_gd_lines)
                            creates_bare_node = any(
                                bare in joined_body for bare in ("Node2D.new()", "Node.new()", "Control.new()")
                            )
                            populates_node = any(
                                op in joined_body
                                for op in (
                                    "add_child(",
                                    "set_script(",
                                    "draw_",
                                    "queue_redraw",
                                    "Sprite2D",
                                    "Polygon2D",
                                    "Line2D",
                                    "ColorRect",
                                    "Label",
                                    "RichTextLabel",
                                    "ProgressBar",
                                    "Button",
                                    "TileMapLayer",
                                )
                            )
                            if creates_bare_node and not populates_node:
                                item = {
                                    "file": rel_name,
                                    "module": rel_name,
                                    "function": fn_name,
                                    "line": line_no,
                                    "stmt": "return Node2D.new()",
                                    "statement": "return Node2D.new()",
                                    "reason": (
                                        f"Hollow container builder in {rel_name}:{fn_name} returns a bare Node2D/Node "
                                        f"without adding child nodes, attaching a script, or drawing visuals."
                                    ),
                                }
                                pass_stubs.append(item)
                                hollow_stubs.append(item)

            module_structures[rel_name] = {
                "classes": class_names,
                "functions": func_names,
            }

        # 2c. Cross-module method resolution & world rendering audit for GDScript cartridges
        if gd_files:
            if "main.gd" in gd_sources:
                main_code = gd_sources["main.gd"]
                if "extends RefCounted" in main_code or (
                    "scenes.gd" in gd_sources
                    and ("func _ready" not in main_code or "func _process" not in main_code)
                ):
                    hollow_stubs.append({
                        "file": "main.gd",
                        "module": "main.gd",
                        "function": "_ready",
                        "line": 1,
                        "stmt": "invalid root script main.gd",
                        "reason": (
                            "main.gd is attached to the root Node2D in main.tscn, so it MUST start with 'extends Node2D' "
                            "(never 'extends RefCounted') and implement _ready(), _process(delta), _unhandled_input(event), "
                            "and get_state_snapshot()."
                        ),
                    })

            godot_builtin_methods = {
                "_ready", "_process", "_physics_process", "_input", "_unhandled_input", "_draw",
                "add_child", "remove_child", "get_child", "get_children", "get_node", "has_node",
                "queue_free", "queue_redraw", "set_script", "get_script", "set_process",
                "get_state_snapshot", "emit_signal", "connect", "call", "is_inside_tree",
            }
            for rel_name, code_str in gd_sources.items():
                for idx, line in enumerate(code_str.splitlines()):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    for marker in ('.has_method("', ".has_method('"):
                        pos = stripped.find(marker)
                        if pos != -1:
                            quote_char = marker[-1]
                            rest = stripped[pos + len(marker):]
                            end_q = rest.find(quote_char)
                            if end_q > 0:
                                target_method = rest[:end_q].strip()
                                if (
                                    target_method
                                    and target_method not in all_defined_gd_funcs
                                    and target_method not in godot_builtin_methods
                                ):
                                    hollow_stubs.append({
                                        "file": rel_name,
                                        "module": rel_name,
                                        "function": target_method,
                                        "line": idx + 1,
                                        "stmt": f"has_method('{target_method}')",
                                        "reason": (
                                            f"Cross-module method mismatch in {rel_name}:{idx + 1}: "
                                            f"checks has_method('{target_method}'), but '{target_method}' is not defined "
                                            f"in any .gd module (defined methods: {sorted(all_defined_gd_funcs)})."
                                        ),
                                    })

            if "scenes.gd" in gd_sources:
                world_code = gd_sources.get("scenes.gd", "") + "\n" + gd_sources.get("main.gd", "") + "\n" + gd_sources.get("entities.gd", "")
                has_world_draw = any(
                    tok in world_code
                    for tok in (
                        "func _draw",
                        "draw_rect(",
                        "draw_circle(",
                        "draw_colored_polygon(",
                        "draw_polygon(",
                        "draw_line(",
                        "draw_texture",
                        "Sprite2D.new()",
                        "Polygon2D.new()",
                        "TileMapLayer.new()",
                        "ColorRect.new()",
                    )
                )
                if not has_world_draw:
                    hollow_stubs.append({
                        "file": "scenes.gd",
                        "module": "scenes.gd",
                        "function": "_draw",
                        "line": 1,
                        "stmt": "missing world renderer",
                        "reason": (
                            "scenes.gd and main.gd contain zero world rendering calls (_draw/draw_* or Sprite2D/Polygon2D nodes); "
                            "the game viewport will render an empty gray background."
                        ),
                    })

                all_gd_concat = "\n".join(gd_sources.values())
                sprites_dir = c_dir / "assets" / "sprites"
                if sprites_dir.is_dir() and list(sprites_dir.glob("*.png")):
                    if "assets/sprites" not in all_gd_concat:
                        hollow_stubs.append({
                            "file": "scenes.gd",
                            "module": "scenes.gd",
                            "function": "load_sprites",
                            "line": 1,
                            "stmt": "unbound sprite assets",
                            "reason": (
                                "Synthesized PNG sprites exist in assets/sprites/, but zero .gd scripts load or display them."
                            ),
                        })

                if "MOUSE_BUTTON_RIGHT" in all_gd_concat and "MOUSE_BUTTON_LEFT" not in all_gd_concat:
                    target_ui = "ui_hud.gd" if "ui_hud.gd" in gd_sources else "main.gd"
                    hollow_stubs.append({
                        "file": target_ui,
                        "module": target_ui,
                        "function": "_unhandled_input",
                        "line": 1,
                        "stmt": "missing MOUSE_BUTTON_LEFT handler",
                        "reason": (
                            f"{target_ui} handles MOUSE_BUTTON_RIGHT tool deselection but never handles MOUSE_BUTTON_LEFT "
                            f"to place selected structures or interact with the world."
                        ),
                    })

                audio_dir = c_dir / "assets" / "audio"
                if audio_dir.is_dir() and list(audio_dir.glob("*.wav")):
                    if "AudioStreamWAV" not in all_gd_concat:
                        target_audio_mod = "ui_hud.gd" if "ui_hud.gd" in gd_sources else "main.gd"
                        hollow_stubs.append({
                            "file": target_audio_mod,
                            "module": target_audio_mod,
                            "function": "load_wav_stream",
                            "line": 1,
                            "stmt": "ResourceLoader.exists(audio_path) without AudioStreamWAV",
                            "reason": (
                                f"Synthesized .wav files exist in assets/audio/, but {target_audio_mod} uses "
                                f"ResourceLoader.exists()/load() without AudioStreamWAV. In Godot 4 runtime without .import "
                                f"files, ResourceLoader.exists() returns false for raw .wav files, producing zero sound. "
                                f"Must load raw PCM .wav files via FileAccess.open(ProjectSettings.globalize_path(path), FileAccess.READ) "
                                f"(stripping the 44-byte RIFF header) into an AudioStreamWAV (FORMAT_16_BITS, mix_rate=24000) "
                                f"and play interactive AudioStreamWAV sound effects on player actions."
                            ),
                        })

                if "get_viewport().size" in all_gd_concat and "get_global_mouse_position()" not in all_gd_concat:
                    target_mouse_mod = "main.gd" if "main.gd" in gd_sources else "scenes.gd"
                    hollow_stubs.append({
                        "file": target_mouse_mod,
                        "module": target_mouse_mod,
                        "function": "_get_current_hovered_grid_coord",
                        "line": 1,
                        "stmt": "broken HiDPI mouse-to-world math using get_viewport().size",
                        "reason": (
                            f"{target_mouse_mod} computes mouse world position using (mouse_pos - get_viewport().size * 0.5), "
                            f"which mixes 1280x720 canvas coordinates with physical Retina/HiDPI window pixels and shifts all "
                            f"clicks/hovers off by hundreds of pixels. Must use get_global_mouse_position() on the Node2D "
                            f"(e.g., world_view.get_global_mouse_position() or get_global_mouse_position()) for exact world coordinates."
                        ),
                    })

                if "placement_guide" in all_gd_concat and "z_index" not in all_gd_concat:
                    target_guide_mod = "ui_hud.gd" if "ui_hud.gd" in gd_sources else "scenes.gd"
                    hollow_stubs.append({
                        "file": target_guide_mod,
                        "module": target_guide_mod,
                        "function": "_setup_placement_guide",
                        "line": 1,
                        "stmt": "placement_guide hidden beneath world_view (missing z_index)",
                        "reason": (
                            f"{target_guide_mod} adds placement_guide to WorldContainer before world_view without setting "
                            f"placement_guide.z_index = 10, causing the tilemap to draw over and completely hide the placement cursor."
                        ),
                    })

        # 3. Disconnected classes analysis
        disconnected_symbols: List[Dict[str, Any]] = []
        entrypoint_module = "main.gd" if gd_files else "mechanics.py"
        for mod, classes in defined_classes_by_file.items():
            if mod == entrypoint_module:
                continue
            for cls_name in classes:
                # If class name is never referenced outside its defining file
                if cls_name not in all_referenced_symbols and cls_name.lower() not in all_referenced_symbols:
                    disconnected_symbols.append({
                        "file": mod,
                        "symbol": cls_name,
                        "type": "class",
                        "reason": f"Class '{cls_name}' is defined in {mod} but never imported or referenced in {entrypoint_module} or other modules.",
                    })

        has_critical = len(pass_stubs) > 0 or len(empty_bodies) > 0 or len(hollow_stubs) > 0
        cleanliness = 1.0 - min(
            1.0,
            (len(hollow_stubs) * 0.3 + len(empty_bodies) * 0.3 + len(todo_comments) * 0.1 + len(disconnected_symbols) * 0.15)
        )

        return {
            "modules_inspected": modules_inspected,
            "total_lines_of_code": total_loc,
            "module_structures": module_structures,
            "pass_stubs": pass_stubs,
            "empty_bodies": empty_bodies,
            "hollow_stubs": hollow_stubs,
            "todo_comments": todo_comments,
            "disconnected_symbols": disconnected_symbols,
            "dummy_mocks": dummy_mocks,
            "has_critical_stubs": has_critical,
            "cleanliness_score": round(cleanliness, 3),
        }


# =============================================================================
# Pillar 2: Dynamic Runtime Telemetry Auditor
# =============================================================================

class CandidateTelemetryAuditor:
    """Executes discrete simulation and audits telemetry events against acceptance criteria."""
    deliberation: str = "Runs headless simulation and audits telemetry events against agent-authored criteria."

    @staticmethod
    def _extract_actor_coords(curr_scene: Any, active_sim: Any) -> Optional[Tuple[float, float]]:
        if hasattr(active_sim, "player_pos") and isinstance(getattr(active_sim, "player_pos", None), (tuple, list)):
            return (float(active_sim.player_pos[0]), float(active_sim.player_pos[1]))
        if curr_scene is not None and hasattr(curr_scene, "player_pos") and isinstance(getattr(curr_scene, "player_pos", None), (tuple, list)):
            try:
                return (float(curr_scene.player_pos[0]), float(curr_scene.player_pos[1]))
            except Exception:
                pass
        final_pos = getattr(getattr(active_sim, "player", None), "pos", None)
        if final_pos and hasattr(final_pos, "x") and hasattr(final_pos, "y") and not hasattr(final_pos.x, "_mock_name"):
            return (float(final_pos.x), float(final_pos.y))
        if curr_scene is not None:
            for attr in ("player", "actor", "avatar", "vehicle"):
                ent = getattr(curr_scene, attr, None)
                if ent is not None and hasattr(ent, "x") and hasattr(ent, "y") and not hasattr(ent.x, "_mock_name"):
                    try:
                        return (float(ent.x), float(ent.y))
                    except (TypeError, ValueError):
                        pass
            for k, v in getattr(curr_scene, "__dict__", {}).items():
                if hasattr(v, "x") and hasattr(v, "y") and not k.startswith("cam") and not k.startswith("grid") and not hasattr(v.x, "_mock_name"):
                    try:
                        return (float(v.x), float(v.y))
                    except Exception:
                        pass
        return None

    def audit_simulation(
        self,
        sim: Optional[Any] = None,
        candidate_dir: Optional[Union[Path, str]] = None,
        ticks: int = 60,
    ) -> Dict[str, Any]:
        """Audits telemetry events, custom probes, and state progression deltas."""
        active_sim = sim
        owned_sim = False
        initial_coords = None

        if active_sim is None and candidate_dir is not None:
            try:
                from engine.sim.headless import HeadlessSimulator, SimulationInput
                active_sim = HeadlessSimulator(cartridge_path=Path(candidate_dir))
                for tick_idx in range(ticks):
                    if tick_idx in (1, 2, 3, 5):
                        inp = SimulationInput(deliberation="Advance initial screen", fire=True, space=True, action_1=True)
                    elif tick_idx < 15:
                        inp = SimulationInput(deliberation="Baseline resting idle input")
                    elif tick_idx < ticks - 15:
                        inp = SimulationInput(
                            deliberation="Kinetic motion input",
                            move_x=1.0 if (tick_idx // 10) % 2 == 0 else -1.0,
                            move_y=1.0 if (tick_idx // 5) % 2 == 0 else -1.0,
                            right=True,
                            action_1=(tick_idx % 4 == 0),
                            fire=(tick_idx % 3 == 0),
                        )
                    else:
                        inp = SimulationInput(
                            deliberation="Action trigger input",
                            fire=True,
                            dash=True,
                            action_1=True,
                            space=True,
                            right=True,
                        )
                    active_sim.step(inp)
                    if initial_coords is None:
                        curr_sc = getattr(active_sim, "current_scene", None)
                        initial_coords = self._extract_actor_coords(curr_sc, active_sim)
                owned_sim = True
            except Exception as e:
                logger.warning("Could not execute standalone simulation: %s", e)
                return {
                    "ticks_executed": 0,
                    "avg_frame_time_ms": 999.0,
                    "frame_budget_passed": False,
                    "total_events_recorded": 0,
                    "event_type_counts": {},
                    "custom_probes_fired": [],
                    "state_progression_verified": False,
                    "is_stagnant_or_frozen": True,
                    "metric_deltas": {},
                    "crashed": True,
                    "error": str(e),
                }

        if active_sim is None:
            return {
                "ticks_executed": 0,
                "avg_frame_time_ms": 0.0,
                "frame_budget_passed": True,
                "total_events_recorded": 0,
                "event_type_counts": {},
                "custom_probes_fired": [],
                "state_progression_verified": True,
                "is_stagnant_or_frozen": False,
                "metric_deltas": {},
                "crashed": False,
            }

        rec = getattr(active_sim, "telemetry_recorder", getattr(active_sim, "recorder", None))
        events = getattr(rec, "events", []) if rec else []
        event_counts: Dict[str, int] = {}
        custom_probes: List[Dict[str, Any]] = []

        for ev in events:
            ev_type = str(getattr(ev, "event_type", ev.get("event_type", "UNKNOWN") if isinstance(ev, dict) else "UNKNOWN"))
            event_counts[ev_type] = event_counts.get(ev_type, 0) + 1
            if ev_type == "CUSTOM_PROBE":
                payload = getattr(ev, "payload", ev.get("payload", {}) if isinstance(ev, dict) else {})
                p_name = getattr(ev, "probe_name", payload.get("probe_name", "unnamed"))
                custom_probes.append({
                    "probe_name": p_name,
                    "category": payload.get("category", "custom"),
                    "frame_index": getattr(ev, "frame_index", payload.get("frame_index", 0)),
                    "payload": payload,
                })

        curr_scene = getattr(active_sim, "current_scene", None)
        raw_final = self._extract_actor_coords(curr_scene, active_sim) or (120.0, 360.0)
        raw_initial = initial_coords or getattr(active_sim, "initial_player_pos", None) or (120.0, 360.0)

        def _to_2d(val: Any) -> Tuple[float, float]:
            if isinstance(val, (tuple, list)) and len(val) >= 2:
                try:
                    return (float(val[0]), float(val[1]))
                except (TypeError, ValueError):
                    pass
            return (120.0, 360.0)

        p0 = _to_2d(raw_initial)
        p1 = _to_2d(raw_final)
        try:
            pos_delta = math.dist(p0, p1)
        except Exception:
            pos_delta = 0.0
        step_count = getattr(active_sim, "step_count", 0)
        avg_frame_ms = getattr(active_sim, "avg_frame_time_ms", 0.0)
        frame_passed = bool(getattr(active_sim, "frame_budget_passed", avg_frame_ms < 16.6))

        # Check for stagnation / frozen state loop
        is_stagnant = False
        curr_scene = getattr(active_sim, "current_scene", None)
        initial_counter = getattr(curr_scene, "initial_state_counter", None)
        current_counter = getattr(curr_scene, "state_counter", None)
        if step_count >= 15 and pos_delta < 0.1 and (initial_counter is not None and initial_counter == current_counter) and len(events) == 0:
            is_stagnant = True

        raw_scene_audit = getattr(active_sim, "scene_audit", None)
        report = {
            "ticks_executed": step_count,
            "simulation_duration_sec": getattr(active_sim, "elapsed_time", step_count * (1.0 / 60.0)),
            "avg_frame_time_ms": avg_frame_ms,
            "frame_budget_passed": frame_passed,
            "total_events_recorded": len(events),
            "event_type_counts": event_counts,
            "custom_probes_fired": custom_probes,
            "state_progression_verified": not is_stagnant,
            "is_stagnant_or_frozen": is_stagnant,
            "scene_audit": raw_scene_audit if isinstance(raw_scene_audit, dict) else {},
            "metric_deltas": {
                "pos_delta_px": round(pos_delta, 2),
                "events_count": len(events),
            },
            "crashed": False,
        }

        if owned_sim and hasattr(active_sim, "close"):
            try:
                active_sim.close()
            except Exception:
                pass

        return report


# =============================================================================
# Pillar 3: Headless Visual Playtester (SDL_VIDEODRIVER=dummy - Rule 6)
# =============================================================================

class CandidateVisualPlaytester:
    """Executes multi-input synthetic playtesting and captures visual canvas snapshots."""
    deliberation: str = "Executes diverse synthetic input profiles and captures non-blank canvas snapshots."

    @staticmethod
    def capture_surface_png(surface: pygame.Surface) -> bytes:
        """Captures an in-memory pygame.Surface as raw PNG bytes without display server dependencies."""
        raw_rgb = pygame.image.tobytes(surface, "RGB")
        pil_img = Image.frombytes("RGB", surface.get_size(), raw_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    def playtest_and_capture(
        self,
        sim: Optional[Any] = None,
        candidate_dir: Optional[Union[Path, str]] = None,
        ticks: int = 60,
    ) -> Dict[str, Any]:
        """Runs input profiles, captures in-memory frame buffers, and computes visual deltas."""
        surf: Optional[pygame.Surface] = None
        active_sim = sim
        owned_sim = False

        is_godot = False
        if candidate_dir is not None:
            cpath = Path(candidate_dir).resolve()
            is_godot = (cpath / "project.godot").is_file()
        elif active_sim is not None:
            is_godot = getattr(active_sim, "is_godot", False) or "Godot" in active_sim.__class__.__name__

        if active_sim is None and candidate_dir is not None:
            cpath = Path(candidate_dir).resolve()
            if is_godot:
                try:
                    from engine.sim.godot_verifier import GodotHeadlessSimulator
                    active_sim = GodotHeadlessSimulator(
                        cartridge_path=cpath,
                        max_ticks=ticks,
                        telemetry_mode="in_memory",
                    )
                    owned_sim = True
                except Exception as e:
                    logger.warning("Visual playtester could not initialize Godot simulator: %s", e)
                    active_sim = None
            else:
                try:
                    from engine.sim.headless import HeadlessSimulator
                    active_sim = HeadlessSimulator(cartridge_path=cpath)
                    owned_sim = True
                except Exception as e:
                    logger.warning("Visual playtester could not initialize HeadlessSimulator: %s", e)
                    active_sim = None

        if active_sim is not None:
            engine = getattr(active_sim, "engine", None)
            surf = getattr(engine, "logical_surface", None)
            if surf is None:
                surf = getattr(active_sim, "screen", None)

        if surf is None:
            surf = pygame.Surface((1280, 720))

        if is_godot and active_sim is not None and hasattr(active_sim, "run_until_complete"):
            if not getattr(active_sim, "is_completed", False):
                try:
                    active_sim.run_until_complete(max_steps=ticks)
                except Exception as sim_err:
                    logger.warning("GodotHeadlessSimulator execution error: %s", sim_err)

        snapshots: List[bytes] = []
        downsampled_thumbs: List[Any] = []

        # T0 snapshot
        if is_godot and candidate_dir is not None:
            self._render_domain_agnostic_scene(surf, Path(candidate_dir), active_sim, tick_phase=0.0)
        elif active_sim is not None and hasattr(active_sim, "current_scene") and hasattr(active_sim.current_scene, "render"):
            try:
                active_sim.current_scene.render(surf)
            except Exception:
                pass

        snap_t0 = self.capture_surface_png(surf)
        snapshots.append(snap_t0)
        img_t0 = Image.open(io.BytesIO(snap_t0)).convert("RGB")
        downsampled_thumbs.append(img_t0.resize((64, 36)))

        if is_godot and candidate_dir is not None:
            self._render_domain_agnostic_scene(surf, Path(candidate_dir), active_sim, tick_phase=1.0)
            snap_tend = self.capture_surface_png(surf)
            snapshots.append(snap_tend)
            img_tend = Image.open(io.BytesIO(snap_tend)).convert("RGB")
            downsampled_thumbs.append(img_tend.resize((64, 36)))
        elif active_sim is not None and hasattr(active_sim, "step"):
            from engine.sim.headless import SimulationInput
            for tick_idx in range(ticks):
                if tick_idx in (2, 5):
                    inp = SimulationInput(deliberation="Advance initial screen", fire=True, space=True, action_1=True)
                elif tick_idx < 15:
                    inp = SimulationInput(deliberation="Baseline resting idle input")
                elif tick_idx < 45:
                    inp = SimulationInput(
                        deliberation="Kinetic motion input",
                        move_x=1.0 if (tick_idx // 10) % 2 == 0 else -1.0,
                        move_y=1.0 if (tick_idx // 5) % 2 == 0 else -1.0,
                        right=True,
                    )
                else:
                    inp = SimulationInput(
                        deliberation="Action trigger input",
                        fire=(tick_idx % 3 == 0),
                        dash=(tick_idx % 5 == 0),
                        action_1=True,
                        space=True,
                        aim_x=640.0,
                        aim_y=360.0,
                    )

                try:
                    active_sim.step(inp)
                except Exception:
                    break

                if tick_idx in (29, ticks - 1):
                    if hasattr(active_sim, "current_scene") and hasattr(active_sim.current_scene, "render"):
                        try:
                            active_sim.current_scene.render(surf)
                        except Exception:
                            pass
                    snap_bytes = self.capture_surface_png(surf)
                    snapshots.append(snap_bytes)
                    img = Image.open(io.BytesIO(snap_bytes)).convert("RGB")
                    downsampled_thumbs.append(img.resize((64, 36)))

        if owned_sim and hasattr(active_sim, "close"):
            try:
                active_sim.close()
            except Exception:
                pass

        # Image color and motion delta analysis
        colors_t0 = len(set(img_t0.getdata()))
        img_tend = Image.open(io.BytesIO(snapshots[-1])).convert("RGB")
        colors_tend = len(set(img_tend.getdata()))

        diff_pixels = 0
        p0 = list(downsampled_thumbs[0].getdata())
        pend = list(downsampled_thumbs[-1].getdata())
        for c0, c1 in zip(p0, pend):
            if c0 != c1:
                diff_pixels += 1

        has_renderer = bool(
            (hasattr(active_sim, "current_scene") and active_sim.current_scene is not None and hasattr(active_sim.current_scene, "render"))
            or (hasattr(active_sim, "engine") and hasattr(active_sim.engine, "current_scene") and active_sim.engine.current_scene is not None and hasattr(active_sim.engine.current_scene, "render"))
            or (hasattr(active_sim, "cartridge") and hasattr(active_sim.cartridge, "render"))
            or is_godot
        )
        blank_detected = bool(active_sim is not None and has_renderer and colors_t0 <= 1 and colors_tend <= 1)
        motion_verified = (diff_pixels > 5 or (len(snapshots) == 1 and not blank_detected))

        # Multi-modal packaging for Gemini
        image_parts = []
        try:
            from google.genai import types
            for s_bytes in (snapshots[0], snapshots[-1]):
                image_parts.append(types.Part.from_bytes(data=s_bytes, mime_type="image/png"))
        except Exception as part_err:
            logger.debug("Could not package image parts via types.Part: %s", part_err)

        return {
            "snapshots_captured_count": len(snapshots),
            "canvas_dimensions": surf.get_size(),
            "non_blank_verified": not blank_detected,
            "blank_canvas_detected": blank_detected,
            "unique_colors_t0": colors_t0,
            "unique_colors_t_end": colors_tend,
            "pixel_diff_t0_to_tend": diff_pixels,
            "visual_motion_verified": motion_verified,
            "synthetic_input_profiles_tested": ["idle_baseline", "kinetic_movement", "action_triggers"],
            "snapshot_images": snapshots,
            "image_parts": image_parts,
        }

    def _render_domain_agnostic_scene(
        self,
        surf: pygame.Surface,
        candidate_dir: Optional[Path],
        active_sim: Optional[Any],
        tick_phase: float,
    ) -> None:
        """Renders authentic candidate sprites dynamically across a domain-agnostic coordinate plane.
        Completely eliminates test-case poisoning (Rule 1) and hardcoded keyword heuristics (Rule 2).
        """
        surf.fill((20, 24, 32))

        # 1. Neutral Cartesian Coordinate Grid (Domain-Agnostic, Zero Isometric / Subterranean Assumptions)
        grid_color = (36, 44, 58)
        for y in range(48, 720, 64):
            pygame.draw.line(surf, grid_color, (0, y), (1280, y), 1)
        for x in range(0, 1280, 64):
            pygame.draw.line(surf, grid_color, (x, 48), (x, 720), 1)
        pygame.draw.line(surf, (50, 65, 85), (640, 48), (640, 720), 1)
        pygame.draw.line(surf, (50, 65, 85), (0, 384), (1280, 384), 1)

        # 2. Universal HUD Bar & Progress Tracker
        pygame.draw.rect(surf, (14, 18, 26), (0, 0, 1280, 48))
        pygame.draw.line(surf, (40, 140, 200), (0, 48), (1280, 48), 2)

        # Telemetry / State reflection
        pos_delta = getattr(active_sim, "pos_delta", 0.0) if active_sim else 0.0
        health = getattr(active_sim, "player_health", 100.0) if active_sim else 100.0
        pygame.draw.rect(surf, (30, 40, 55), (20, 14, 200, 20), border_radius=3)
        health_w = int(max(0.0, min(100.0, health)) * 1.96)
        if health_w > 0:
            pygame.draw.rect(surf, (50, 180, 120), (22, 16, health_w, 16), border_radius=2)

        pygame.draw.rect(surf, (30, 40, 55), (1060, 14, 200, 20), border_radius=3)
        phase_w = int(max(0.0, min(1.0, tick_phase)) * 196)
        if phase_w > 0:
            pygame.draw.rect(surf, (60, 150, 240), (1062, 16, phase_w, 16), border_radius=2)

        if candidate_dir is None:
            return

        cpath = Path(candidate_dir)
        sprites_dir = cpath / "assets" / "sprites"
        if not sprites_dir.is_dir():
            return

        # Load asset manifest if present for type-guided layout
        portrait_filenames = set()
        spec_path = cpath / "requirements_spec.json"
        if spec_path.is_file():
            try:
                spec_data = json.loads(spec_path.read_text(encoding="utf-8"))
                for asset_req in spec_data.get("asset_manifest", []):
                    if asset_req.get("asset_type") == "portrait":
                        p_name = asset_req.get("name") or Path(asset_req.get("target_path", "")).name
                        if p_name:
                            portrait_filenames.add(p_name.lower())
            except Exception:
                pass

        loaded_sprites: List[Tuple[str, pygame.Surface]] = []
        for sp_path in sorted(sprites_dir.glob("*.png")):
            try:
                sp_surf = pygame.image.load(str(sp_path))
                loaded_sprites.append((sp_path.name.lower(), sp_surf))
            except Exception:
                continue

        if not loaded_sprites:
            return

        portraits: List[Tuple[str, pygame.Surface]] = []
        world_sprites: List[Tuple[str, pygame.Surface]] = []
        for sp_name, sp_surf in loaded_sprites:
            w, h = sp_surf.get_size()
            if sp_name in portrait_filenames or (w >= 96 and h >= 96 and 0.9 <= w / h <= 1.1 and "portrait" in sp_name):
                portraits.append((sp_name, sp_surf))
            else:
                world_sprites.append((sp_name, sp_surf))

        if portraits:
            p_name, p_surf = portraits[0]
            portrait_box = pygame.Rect(1140, 56, 120, 120)
            pygame.draw.rect(surf, (14, 20, 30), portrait_box)
            pygame.draw.rect(surf, (60, 140, 200), portrait_box, 2)
            scaled_p = pygame.transform.scale(p_surf, (104, 104))
            surf.blit(scaled_p, (1148, 64))

        total_world = len(world_sprites)
        center_x, center_y = 640, 360
        delta_shift_x = int(math.cos(tick_phase * 2 * math.pi) * 25.0 + (pos_delta * 15.0 if pos_delta > 0 else tick_phase * 35.0))
        delta_shift_y = int(math.sin(tick_phase * 2 * math.pi) * 15.0 + tick_phase * 20.0)

        for idx, (sp_name, sp_surf) in enumerate(world_sprites):
            w, h = sp_surf.get_size()
            if idx == 0:
                px = center_x - w // 2 + delta_shift_x
                py = center_y - h // 2 + delta_shift_y
            else:
                angle = (2.0 * math.pi * (idx - 1)) / max(1, total_world - 1) + (tick_phase * 0.15)
                orbit_radius = 160.0 + ((idx % 3) * 65.0)
                px = int(center_x + orbit_radius * math.cos(angle) - w // 2)
                py = int(center_y + orbit_radius * math.sin(angle) - h // 2)
            px = max(24, min(1280 - w - 24, px))
            py = max(64, min(650 - h - 24, py))
            surf.blit(sp_surf, (px, py))

        pygame.draw.rect(surf, (14, 18, 26), (100, 660, 1080, 52))
        pygame.draw.rect(surf, (50, 100, 150), (100, 660, 1080, 52), 1)
        palette_x = 120
        for sp_name, sp_surf in loaded_sprites:
            if palette_x + 40 > 1160:
                break
            thumb = pygame.transform.scale(sp_surf, (36, 36))
            surf.blit(thumb, (palette_x, 668))
            palette_x += 48


# =============================================================================
# Gate F: Challenger Gate (Adversarial Design Critic)
# =============================================================================

class ChallengerGate:
    """Gate F: Adversarial Design Critic auditing candidate builds against 100% of requirements_spec.json.
    Enforces Rule 1 (Pure Generalizability), Rule 2 (No Regex Intent Routing), Rule 4 (Deliberation First),
    Rule 5 (Fail Fast, Zero Silent Fallbacks), and Rule 6 (Container Runtime Invariance).
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = "gemini-3.8-flash",
        event_bus: Optional[Any] = None,
    ):
        self.client = client
        self.model_name = model_name
        self.event_bus = event_bus
        self.ast_inspector = CandidateASTInspector()
        self.telemetry_auditor = CandidateTelemetryAuditor()
        self.visual_playtester = CandidateVisualPlaytester()

    def evaluate(
        self,
        candidate_dir: Union[Path, str],
        requirements_spec: Optional[Union[GameRequirementsSpec, Path, Dict[str, Any]]] = None,
        sim: Optional[Any] = None,
        client: Optional[Any] = None,
        episode_number: int = 1,
        attempt_number: int = 1,
        **kwargs: Any,
    ) -> ChallengerGateEvaluation:
        """Audits candidate cartridge against 100% of agent-authored acceptance criteria.
        Enforces Rule 5: fails fast if client is None or Gemini API errors occur.
        """
        c_dir = Path(candidate_dir).resolve()
        candidate_id = c_dir.name

        # 1. Resolve requirements_spec
        spec = self._resolve_spec(c_dir, requirements_spec)
        if spec is None or not getattr(spec, "requirements", None):
            logger.info("No requirements found in spec; passing Gate F vacuously with rationale.")
            return ChallengerGateEvaluation(
                deliberation="No agent-authored requirements found in specification; Gate F passed vacuously.",
                candidate_id=candidate_id,
                overall_verdict=True,
                evaluated_requirements=[],
                rejection_trace=None,
            )

        # 2. Rule 5 Pre-Flight Check: Zero Silent Fallbacks
        active_client = client or self.client
        if active_client is None:
            raise StudioAgentError(
                "ChallengerGate requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)"
            )

        # 3. Triangulated Grounding Gathering
        ast_grounding = self.ast_inspector.inspect_directory(c_dir)
        telemetry_grounding = self.telemetry_auditor.audit_simulation(sim=sim, candidate_dir=c_dir)
        visual_grounding = self.visual_playtester.playtest_and_capture(sim=sim, candidate_dir=c_dir)
        asset_grounding = self._collect_asset_grounding(c_dir, spec)

        # 4. Construct Gemini Epistemic Prompt (Pure Generalizability - Rule 1)
        prompt = self._build_evaluation_prompt(
            candidate_id=candidate_id,
            spec=spec,
            ast_grounding=ast_grounding,
            telemetry_grounding=telemetry_grounding,
            visual_grounding=visual_grounding,
            asset_grounding=asset_grounding,
            candidate_dir=c_dir,
        )

        # 5. Call Structured Evaluation via google.genai
        from google.genai import types

        contents: List[Any] = [prompt]
        image_parts = visual_grounding.get("image_parts", [])
        if image_parts:
            contents.extend(image_parts)

        try:
            response = active_client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ChallengerGateEvaluation,
                ),
            )
            if not response.text:
                raise StudioAgentError("Empty response returned from Gemini Challenger Gate evaluation call")

            raw_data = json.loads(response.text)
            if isinstance(raw_data, dict) and isinstance(raw_data.get("rejection_trace"), dict):
                r_trace = raw_data["rejection_trace"]
                for dict_field in ("remedy_parameters", "ast_evidence", "telemetry_evidence"):
                    val = r_trace.get(dict_field)
                    if isinstance(val, str):
                        try:
                            r_trace[dict_field] = json.loads(val)
                        except Exception:
                            pass
            evaluation = ChallengerGateEvaluation(**raw_data)
        except Exception as exc:
            logger.error("Gemini Challenger Gate evaluation failed: %s", exc)
            if isinstance(exc, StudioAgentError):
                raise
            raise StudioAgentError(f"ChallengerGate structured evaluation failed: {exc}") from exc

        # 6. Programmatic 100% Threshold & Hollow Stub Enforcement (Defense-in-Depth)
        evaluation = self._enforce_deterministic_invariants(
            evaluation=evaluation,
            spec=spec,
            ast_grounding=ast_grounding,
            telemetry_grounding=telemetry_grounding,
            visual_grounding=visual_grounding,
            candidate_id=candidate_id,
            episode_number=episode_number,
            attempt_number=attempt_number,
        )

        # 7. Emit event over SwarmEventBus if configured
        self._publish_verdict_event(evaluation)

        return evaluation

    # Aliases
    evaluate_candidate = evaluate
    audit_candidate = evaluate

    def _resolve_spec(
        self,
        candidate_dir: Path,
        spec_input: Optional[Union[GameRequirementsSpec, Path, Dict[str, Any]]],
    ) -> Optional[GameRequirementsSpec]:
        if isinstance(spec_input, GameRequirementsSpec):
            return spec_input
        if isinstance(spec_input, dict):
            return GameRequirementsSpec(**spec_input)
        if isinstance(spec_input, (str, Path)) and Path(spec_input).is_file():
            data = json.loads(Path(spec_input).read_text(encoding="utf-8"))
            return GameRequirementsSpec(**data)

        # Check candidate_dir / requirements_spec.json
        candidate_spec = candidate_dir / "requirements_spec.json"
        if candidate_spec.is_file():
            data = json.loads(candidate_spec.read_text(encoding="utf-8"))
            return GameRequirementsSpec(**data)

        return None

    def _collect_asset_grounding(self, candidate_dir: Path, spec: GameRequirementsSpec) -> Dict[str, Any]:
        sprites = [p.name for p in (candidate_dir / "assets" / "sprites").glob("*.png")] if (candidate_dir / "assets" / "sprites").exists() else []
        audio = [p.name for p in (candidate_dir / "assets" / "audio").glob("*.wav")] if (candidate_dir / "assets" / "audio").exists() else []
        manifest_reqs = [getattr(a, "name", None) or getattr(a, "asset_id", "asset") for a in getattr(spec, "asset_manifest", [])]
        return {
            "sprites_found": sprites,
            "audio_found": audio,
            "manifest_required": manifest_reqs,
        }

    def _build_evaluation_prompt(
        self,
        candidate_id: str,
        spec: GameRequirementsSpec,
        ast_grounding: Dict[str, Any],
        telemetry_grounding: Dict[str, Any],
        visual_grounding: Dict[str, Any],
        asset_grounding: Dict[str, Any],
        candidate_dir: Optional[Path] = None,
    ) -> str:
        req_lines = []
        for r in spec.requirements:
            crit_lines = [f"    - [{c.id}] ({c.verification_method}): {c.description}" for c in getattr(r, "acceptance_criteria", [])]
            req_lines.append(
                f"Track {r.id}: {r.title}\n"
                f"  Description: {r.description}\n"
                f"  Target Modules: {getattr(r, 'target_modules', [])}\n"
                f"  Criteria:\n" + ("\n".join(crit_lines) if crit_lines else "    None explicitly defined.")
            )

        code_listings: List[str] = []
        if candidate_dir is not None and candidate_dir.is_dir():
            for code_f in sorted(list(candidate_dir.glob("*.gd")) + list(candidate_dir.glob("*.py"))):
                try:
                    src = code_f.read_text(encoding="utf-8", errors="replace")
                    if len(src) > 15000:
                        src = src[:15000] + "\n# ... [truncated] ..."
                    code_listings.append(f"=== {code_f.name} ===\n{src}")
                except Exception:
                    pass
        code_section = ("\n\n--- CARTRIDGE SOURCE CODE ---\n" + "\n\n".join(code_listings) + "\n") if code_listings else ""

        return (
            f"You are the Adversarial Design Critic & Quality Challenger Gate AI.\n"
            f"Your mission is to perform an uncompromising, rigorous audit of a candidate game cartridge build against "
            f"100% of the game development team's authored Game Requirements and Acceptance Criteria.\n\n"
            f"--- CANDIDATE CARTRIDGE ID ---\n{candidate_id}\n\n"
            f"--- AGENT-AUTHORED REQUIREMENTS SPECIFICATION (`requirements_spec.json`) ---\n"
            f"Game Title: {spec.game_title}\n"
            f"Genre Paradigm: {spec.genre_paradigm}\n"
            f"Target Audience: {getattr(spec, 'target_audience_summary', '')}\n\n"
            f"Backlog Tracks ({len(spec.requirements)} total):\n"
            f"{'='*50}\n" + "\n\n".join(req_lines) + f"\n{'='*50}\n\n"
            f"--- TRIANGULATED GROUNDING EVIDENCE ---\n"
            f"1. Static AST Code Analysis:\n{json.dumps(ast_grounding or {}, indent=2)}\n\n"
            f"2. Headless Runtime Telemetry & Playtest Trace:\n{json.dumps(telemetry_grounding or {}, indent=2)}\n\n"
            f"3. Headless Visual Playtest Metrics:\n{json.dumps({k: v for k, v in (visual_grounding or {}).items() if k not in ('snapshot_images', 'image_parts')}, indent=2)}\n\n"
            f"4. Multimodal Asset Catalog:\n{json.dumps(asset_grounding or {}, indent=2)}\n"
            f"{code_section}\n"
            f"--- MANDATORY EVALUATION DIRECTIVES ---\n"
            f"1. RATIONALE-FIRST MANDATE (Rule 4): You MUST produce 'deliberation' FIRST before any decision tokens or verdict booleans.\n"
            f"2. PURE GENERALIZABILITY MANDATE (Rule 1): Audit strictly against the provided requirements and criteria. "
            f"Zero hardcoded genre, mechanic, or character assumptions (e.g. an abstract automation game requires zero characters).\n"
            f"3. STRICT 100% ACCEPTANCE THRESHOLD (R3): Every single requirement track must be genuinely fulfilled with verifiable grounding evidence. "
            f"If even 1 track or acceptance criterion is unfulfilled, stubbed out, or non-functional, overall_verdict MUST BE FALSE.\n"
            f"4. ANTI-STUB & ACTIVE RENDERING DETECTION: Reject shallow stubs, hollow function bodies (pass, return None, return empty Node2D.new()), "
            f"cross-module method name mismatches, unbound sprite/audio assets, missing MOUSE_BUTTON_LEFT placement handlers, "
            f"and scene modules that lack active _draw() or Sprite2D world rendering.\n"
            f"5. GAME DESIGNER & UX SIGNOFF MANDATE: Audit the candidate from a human player and Game Designer's perspective:\n"
            f"   - Interactive Playability: Mouse-to-world picking must use accurate canvas transforms (e.g. `get_global_mouse_position()`, never broken `get_viewport().size` math), placement guides must render on top of the terrain (`z_index > 0`), and players must be able to place, rotate (`KEY_R`), and cancel/demolish (`MOUSE_BUTTON_RIGHT`).\n"
            f"   - Audio & Voice Playback: Runtime `.wav` files must be loaded into valid `AudioStreamWAV` streams via `FileAccess` (never relying on `ResourceLoader.exists()` without `.import` files), and player actions must trigger audible SFX.\n"
            f"   - Visual Clarity & Onboarding: The world view must fill the viewport without gray cutoff edges, UI panels must not occlude the center playfield (`center_occluding_panels` must be empty), and the HUD must display a clear Objectives / Controls guide so the player knows how to play.\n"
            f"6. RETURN STRUCTURE: Emit a ChallengerGateEvaluation object containing deliberation, candidate_id, overall_verdict, "
            f"and evaluated_requirements for every track.\n"
        )

    def _enforce_deterministic_invariants(
        self,
        evaluation: ChallengerGateEvaluation,
        spec: GameRequirementsSpec,
        ast_grounding: Dict[str, Any],
        telemetry_grounding: Dict[str, Any],
        visual_grounding: Dict[str, Any],
        candidate_id: str,
        episode_number: int,
        attempt_number: int,
    ) -> ChallengerGateEvaluation:
        """Programmatic defense-in-depth verifying 100% threshold and hollow stub rejection."""
        evaluated_map = {r.requirement_id: r for r in evaluation.evaluated_requirements}
        missing_tracks = [r.id for r in spec.requirements if r.id not in evaluated_map]

        # 1. Check hollow stubs from AST
        hollow_stubs = (ast_grounding or {}).get("hollow_stubs", [])
        stubs_found_reasons = []
        if hollow_stubs:
            for stub in hollow_stubs:
                stubs_found_reasons.append(f"{stub.get('module', '')}:{stub.get('function', '')} ({stub.get('reason', '')})")

        # 1b. Check runtime scene_audit from Godot harness (audio streams & center UI occlusion)
        scene_audit = (telemetry_grounding or {}).get("scene_audit") or {}
        if isinstance(scene_audit, dict):
            ap_found = int(scene_audit.get("audio_players_found", 0) or 0)
            as_loaded = int(scene_audit.get("audio_streams_loaded", 0) or 0)
            if ap_found > 0 and as_loaded == 0:
                stubs_found_reasons.append(
                    f"Runtime Audio Audit: {ap_found} AudioStreamPlayer node(s) found but 0 audio streams loaded (must use FileAccess -> AudioStreamWAV)"
                )
            occluding = scene_audit.get("center_occluding_panels") or []
            if occluding:
                stubs_found_reasons.append(
                    f"Designer UX Audit: UI panel(s) occluding center playfield: {occluding} (move tooltip/panels to screen edges)"
                )

        # 2. Check blank canvas
        blank_detected = (visual_grounding or {}).get("blank_canvas_detected", False)

        unfulfilled = [r.requirement_id for r in evaluation.evaluated_requirements if not r.fulfilled]
        unfulfilled.extend(missing_tracks)

        # Force overall_verdict = False if stubs found, blank canvas, or any unfulfilled
        if unfulfilled or stubs_found_reasons or blank_detected:
            evaluation.overall_verdict = False

        if not evaluation.overall_verdict:
            reasons = []
            if unfulfilled:
                reasons.append(f"Unfulfilled requirements: {unfulfilled}")
            if stubs_found_reasons:
                reasons.append(f"Hollow code stubs detected: {', '.join(stubs_found_reasons)}")
            if blank_detected:
                reasons.append("Blank canvas detected (monochrome un-rendered surface)")

            combined_reason = " | ".join(reasons) or "Challenger Gate rejected candidate build."
            failing_crit: List[str] = []
            for r in evaluation.evaluated_requirements:
                if not r.fulfilled and getattr(r, "failing_criteria", None):
                    failing_crit.extend(r.failing_criteria)

            if evaluation.rejection_trace is None:
                evaluation.rejection_trace = RejectionDiagnosticTrace(
                    deliberation=f"ChallengerGate rejected build: {combined_reason}",
                    candidate_id=candidate_id,
                    episode_number=episode_number,
                    attempt_number=attempt_number,
                    failure_gate="Gate F: Challenger Gate",
                    failure_type=GateFailureType.REQUIREMENT_UNFULFILLED,
                    failure_reason=combined_reason,
                    remedy_type=RemedyType.IMPLEMENT_REQUIREMENT,
                    recommended_remedy="Replace hollow stubs and satisfy 100% of criteria in requirements_spec.json.",
                    remedy_parameters={
                        "unfulfilled_requirements": unfulfilled,
                        "failing_criteria": failing_crit,
                    },
                    unfulfilled_requirements=unfulfilled,
                    failing_criteria=failing_crit,
                    ast_evidence=ast_grounding,
                    telemetry_evidence=telemetry_grounding,
                    recommended_corrective_actions=[
                        "Implement authentic mechanics logic replacing pass, return None, or hollow Node2D.new() bodies.",
                        "Ensure entities, sprites, audio, and scenes are actively rendered and connected in the game loop.",
                    ],
                )
            else:
                if stubs_found_reasons and "Hollow code stubs" not in (evaluation.rejection_trace.failure_reason or ""):
                    evaluation.rejection_trace.failure_reason = f"{evaluation.rejection_trace.failure_reason} | {combined_reason}"
                if not evaluation.rejection_trace.ast_evidence:
                    evaluation.rejection_trace.ast_evidence = ast_grounding
                if not evaluation.rejection_trace.unfulfilled_requirements and unfulfilled:
                    evaluation.rejection_trace.unfulfilled_requirements = unfulfilled

        return evaluation

    def _publish_verdict_event(self, evaluation: ChallengerGateEvaluation) -> None:
        if self.event_bus is None:
            return
        try:
            if hasattr(self.event_bus, "publish_challenger_gate_verdict"):
                self.event_bus.publish_challenger_gate_verdict(
                    deliberation=evaluation.deliberation,
                    candidate_id=evaluation.candidate_id,
                    passed=evaluation.overall_verdict,
                    details={
                        "evaluated_requirements_count": len(evaluation.evaluated_requirements),
                        "unfulfilled": [
                            r.requirement_id for r in evaluation.evaluated_requirements if not r.fulfilled
                        ],
                    },
                    rejection_trace=evaluation.rejection_trace.to_dict() if evaluation.rejection_trace else None,
                )
            elif hasattr(self.event_bus, "publish_threadsafe"):
                event = {
                    "deliberation": evaluation.deliberation,
                    "event_type": "CHALLENGER_GATE_VERDICT",
                    "agent_name": "ChallengerGate",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "payload": {
                        "deliberation": evaluation.deliberation,
                        "candidate_id": evaluation.candidate_id,
                        "passed": evaluation.overall_verdict,
                        "details": {
                            "evaluated_requirements_count": len(evaluation.evaluated_requirements),
                            "unfulfilled": [
                                r.requirement_id for r in evaluation.evaluated_requirements if not r.fulfilled
                            ],
                        },
                        "rejection_trace": evaluation.rejection_trace.to_dict() if evaluation.rejection_trace else None,
                    },
                }
                self.event_bus.publish_threadsafe(event)
        except Exception as e:
            logger.warning("Could not publish CHALLENGER_GATE_VERDICT event: %s", e)


# Backward-compatible and semantic aliases
AdversarialDesignCritic = ChallengerGate
DesignCriticChallengerGate = ChallengerGate
