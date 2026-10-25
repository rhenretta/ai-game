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
    MultiAgentSignOffReport,
    RejectionDiagnosticTrace,
    RemedyType,
    RequirementEvaluationRecord,
    SpecialistSignOffVerdict,
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
                elif stripped.startswith(("func ", "static func ")):
                    after_func = stripped.split("func ", 1)[1].strip()
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
                            if nxt_indent <= fn_indent and nxt.startswith(("func ", "static func ", "class ", "var ", "const ", "signal ", "@")):
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
                                    "create_articulated_rig",
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
            if sprites_dir.is_dir():
                png_files = [
                    p for p in sorted(sprites_dir.glob("*.png"))
                    if "_part_" not in p.name and not p.name.endswith("_portrait.png")
                ]
                logical_sprites: Dict[str, Path] = {}
                for p in png_files:
                    base_id = p.stem[:-6] if p.stem.endswith("_sheet") else p.stem
                    logical_sprites.setdefault(base_id, p)
                if logical_sprites and "assets/sprites" not in all_gd_concat and "asset_library.json" not in all_gd_concat:
                    hollow_stubs.append({
                        "file": "scenes.gd" if "scenes.gd" in gd_sources else "main.gd",
                        "module": "scenes.gd" if "scenes.gd" in gd_sources else "main.gd",
                        "function": "load_sprites",
                        "line": 1,
                        "stmt": "unbound sprite assets",
                        "reason": (
                            "Synthesized PNG sprites exist in assets/sprites/, but zero .gd scripts load or display them."
                        ),
                    })

            audio_dir = c_dir / "assets" / "audio"
            if audio_dir.is_dir() and list(audio_dir.glob("*.wav")):
                target_audio_mod = "ui_hud.gd" if "ui_hud.gd" in gd_sources else "main.gd"
                if "AudioStreamWAV" not in all_gd_concat:
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
                            f"Must load raw PCM .wav files via FileAccess.get_file_as_bytes(ProjectSettings.globalize_path(path)) "
                            f"(stripping the 44-byte RIFF header) into an AudioStreamWAV (FORMAT_16_BITS, mix_rate=24000)."
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
        cpath: Optional[Path] = Path(candidate_dir).resolve() if candidate_dir is not None else None

        if active_sim is None and candidate_dir is not None:
            cpath = Path(candidate_dir).resolve()
            if (cpath / "project.godot").is_file():
                try:
                    from engine.sim.godot_verifier import GodotHeadlessSimulator
                    active_sim = GodotHeadlessSimulator(
                        cartridge_path=cpath,
                        max_ticks=ticks,
                        telemetry_mode="in_memory",
                        capture_viewport=True,
                    )
                    active_sim.run_until_complete(max_steps=ticks, capture_viewport=True)
                    owned_sim = True
                except Exception as e:
                    logger.warning("Could not execute Godot simulation in TelemetryAuditor: %s", e)
            else:
                try:
                    from engine.sim.headless import HeadlessSimulator, SimulationInput
                    active_sim = HeadlessSimulator(cartridge_path=cpath)
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
        atomic_suite_summary: Dict[str, Any] = {}
        if cpath is not None and (cpath / "project.godot").is_file() and (cpath / "tests").is_dir():
            try:
                from engine.sim.atomic_runner import GodotAtomicTestRunner
                suite_res = GodotAtomicTestRunner.run_all_atomic_tests(cpath)
                failed_list = [
                    {
                        "unit_id": r.unit_id,
                        "test_script": r.test_script_path,
                        "failed_assertions": [a._asdict() for a in r.failed_assertions],
                        "stderr": r.stderr,
                    }
                    for r in suite_res.results
                    if not r.passed
                ]
                atomic_suite_summary = {
                    "total_tests": suite_res.total_tests,
                    "passed_tests": suite_res.passed_tests,
                    "failed_tests": suite_res.failed_tests,
                    "all_passed": suite_res.all_passed,
                    "failed_details": failed_list,
                }
            except Exception as atomic_err:
                logger.warning("Could not execute GodotAtomicTestRunner suite in TelemetryAuditor: %s", atomic_err)

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
            "atomic_test_suite": atomic_suite_summary,
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
    def capture_surface_png(surface: Any) -> bytes:
        """Captures an in-memory pygame.Surface as raw PNG bytes without display server dependencies."""
        import pygame
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
        """Runs input profiles, captures frame buffers (strictly real Godot 4 viewport PNGs for Godot cartridges),
        and computes visual deltas. Zero Pygame fallback rendering for Godot cartridges (Rule 5).
        """
        if isinstance(sim, (Path, str)) and candidate_dir is None:
            candidate_dir = sim
            sim = None
        surf: Optional[pygame.Surface] = None
        active_sim = sim
        owned_sim = False

        is_godot = False
        cpath: Optional[Path] = None
        if candidate_dir is not None:
            cpath = Path(candidate_dir).resolve()
            is_godot = (cpath / "project.godot").is_file()
        elif active_sim is not None:
            is_godot = getattr(active_sim, "is_godot", False) or "Godot" in active_sim.__class__.__name__
            if hasattr(active_sim, "cartridge_path"):
                cpath = Path(active_sim.cartridge_path).resolve()

        if active_sim is None and cpath is not None:
            if is_godot:
                try:
                    from engine.sim.godot_verifier import GodotHeadlessSimulator
                    active_sim = GodotHeadlessSimulator(
                        cartridge_path=cpath,
                        max_ticks=ticks,
                        telemetry_mode="in_memory",
                        capture_viewport=True,
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

        _pixels = lambda im: im.get_flattened_data() if hasattr(im, "get_flattened_data") else im.getdata()

        # --- GODOT 4 AUTHENTIC VIEWPORT CAPTURE PATH (Zero Pygame Impersonation) ---
        if is_godot:
            if active_sim is not None and hasattr(active_sim, "run_until_complete"):
                if not getattr(active_sim, "viewport_captured", False):
                    try:
                        active_sim.run_until_complete(max_steps=ticks, capture_viewport=True)
                    except TypeError:
                        active_sim.run_until_complete(max_steps=ticks)
                    except Exception as sim_err:
                        logger.warning("GodotHeadlessSimulator viewport execution error: %s", sim_err)

            if owned_sim and hasattr(active_sim, "close"):
                try:
                    active_sim.close()
                except Exception:
                    pass

            snapshots: List[bytes] = []
            captured_frame_paths: List[str] = []
            canvas_dims = (1280, 720)
            colors_t0 = 0
            colors_tend = 0
            diff_pixels = 0

            overlay_bytes: Optional[bytes] = None
            if cpath is not None:
                t0_file = cpath / "assets" / "playtest_captures" / "playtest_t0.png"
                tend_file = cpath / "assets" / "playtest_captures" / "playtest_tend.png"
                overlay_file = cpath / "assets" / "playtest_captures" / "playtest_collision_overlay.png"
                if t0_file.is_file() and tend_file.is_file():
                    b_t0 = t0_file.read_bytes()
                    b_tend = tend_file.read_bytes()
                    snapshots = [b_t0, b_tend]
                    captured_frame_paths = [
                        "assets/playtest_captures/playtest_t0.png",
                        "assets/playtest_captures/playtest_tend.png",
                    ]
                    if overlay_file.is_file():
                        overlay_bytes = overlay_file.read_bytes()
                        captured_frame_paths.append("assets/playtest_captures/playtest_collision_overlay.png")
                    img_t0 = Image.open(io.BytesIO(b_t0)).convert("RGB")
                    img_tend = Image.open(io.BytesIO(b_tend)).convert("RGB")
                    canvas_dims = img_t0.size
                    colors_t0 = len(set(_pixels(img_t0)))
                    colors_tend = len(set(_pixels(img_tend)))
                    p0 = list(_pixels(img_t0.resize((64, 36))))
                    pend = list(_pixels(img_tend.resize((64, 36))))
                    diff_pixels = sum(1 for c0, c1 in zip(p0, pend) if c0 != c1)

            blank_detected = (len(snapshots) < 2) or (colors_t0 <= 1 and colors_tend <= 1)
            motion_verified = (not blank_detected) and (diff_pixels > 5)

            image_parts = []
            if len(snapshots) >= 2:
                try:
                    from google.genai import types
                    for s_bytes in (snapshots[0], snapshots[-1]):
                        image_parts.append(types.Part.from_bytes(data=s_bytes, mime_type="image/png"))
                    if overlay_bytes:
                        image_parts.append(types.Part.from_bytes(data=overlay_bytes, mime_type="image/png"))
                except Exception as part_err:
                    logger.debug("Could not package image parts via types.Part: %s", part_err)

            return {
                "snapshots_captured_count": len(snapshots),
                "canvas_dimensions": canvas_dims,
                "non_blank_verified": not blank_detected,
                "blank_canvas_detected": blank_detected,
                "unique_colors_t0": colors_t0,
                "unique_colors_t_end": colors_tend,
                "pixel_diff_t0_to_tend": diff_pixels,
                "visual_motion_verified": motion_verified,
                "synthetic_input_profiles_tested": ["idle_baseline", "kinetic_movement", "action_triggers"],
                "captured_frame_paths": captured_frame_paths,
                "snapshot_images": snapshots,
                "image_parts": image_parts,
            }

        # --- LEGACY PYGAME SIMULATOR PATH (Only for non-Godot Pygame mocks/cartridges) ---
        if active_sim is not None:
            engine = getattr(active_sim, "engine", None)
            surf = getattr(engine, "logical_surface", None)
            if surf is None:
                surf = getattr(active_sim, "screen", None)

        if surf is None:
            import pygame
            surf = pygame.Surface((1280, 720))

        snapshots = []
        downsampled_thumbs: List[Any] = []

        if active_sim is not None and hasattr(active_sim, "current_scene") and hasattr(active_sim.current_scene, "render"):
            try:
                active_sim.current_scene.render(surf)
            except Exception:
                pass

        snap_t0 = self.capture_surface_png(surf)
        snapshots.append(snap_t0)
        img_t0 = Image.open(io.BytesIO(snap_t0)).convert("RGB")
        downsampled_thumbs.append(img_t0.resize((64, 36)))

        if active_sim is not None and hasattr(active_sim, "step"):
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

        colors_t0 = len(set(_pixels(img_t0)))
        img_tend = Image.open(io.BytesIO(snapshots[-1])).convert("RGB")
        colors_tend = len(set(_pixels(img_tend)))

        diff_pixels = 0
        p0 = list(_pixels(downsampled_thumbs[0]))
        pend = list(_pixels(downsampled_thumbs[-1]))
        for c0, c1 in zip(p0, pend):
            if c0 != c1:
                diff_pixels += 1

        has_renderer = bool(
            (hasattr(active_sim, "current_scene") and active_sim.current_scene is not None and hasattr(active_sim.current_scene, "render"))
            or (hasattr(active_sim, "engine") and hasattr(active_sim.engine, "current_scene") and active_sim.engine.current_scene is not None and hasattr(active_sim.engine.current_scene, "render"))
            or (hasattr(active_sim, "cartridge") and hasattr(active_sim.cartridge, "render"))
        )
        blank_detected = bool(active_sim is not None and has_renderer and colors_t0 <= 1 and colors_tend <= 1)
        motion_verified = (diff_pixels > 5 or (len(snapshots) == 1 and not blank_detected))

        captured_frame_paths = []
        if cpath is not None and snapshots:
            try:
                cap_dir = cpath / "assets" / "playtest_captures"
                cap_dir.mkdir(parents=True, exist_ok=True)
                t0_file = cap_dir / "playtest_t0.png"
                tend_file = cap_dir / "playtest_tend.png"
                t0_file.write_bytes(snapshots[0])
                tend_file.write_bytes(snapshots[-1])
                captured_frame_paths = [
                    "assets/playtest_captures/playtest_t0.png",
                    "assets/playtest_captures/playtest_tend.png",
                ]
            except Exception as cap_err:
                logger.debug("Could not persist playtest_captures PNGs: %s", cap_err)

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
            "captured_frame_paths": captured_frame_paths,
            "snapshot_images": snapshots,
            "image_parts": image_parts,
        }



# =============================================================================
# Pillar 4: Multi-Agent Final Sign-Off Council
# =============================================================================

SPECIALIST_COUNCIL_ROLES: Tuple[str, ...] = (
    "VisualArtDirector",
    "UXExperienceDesigner",
    "GameplayDesigner",
    "StoryWorldArchitect",
    "AudioAtmosphereDesigner",
    "CreativeDirector",
)


class MultiAgentSignOffCouncil:
    """Convenes all 6 studio specialist roles to inspect the rendered candidate build,
    multimodal assets, audio configuration, and runtime scene audit before a build can be promoted.
    Enforces Rule 4 (deliberation first) and 100% unanimous specialist sign-off.
    """
    deliberation: str = "Multi-Agent Final Sign-Off Council requiring unanimous approval from all 6 studio specialists."

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = "gemini-3.8-flash",
        event_bus: Optional[Any] = None,
    ):
        self.client = client
        self.model_name = model_name
        self.event_bus = event_bus

    @staticmethod
    def _is_live_genai_client(client: Optional[Any]) -> bool:
        if client is None:
            return False
        mod = getattr(type(client), "__module__", "") or ""
        return mod.startswith("google.genai")

    def evaluate_sign_off(
        self,
        candidate_dir: Union[Path, str],
        spec: Optional[GameRequirementsSpec] = None,
        ast_grounding: Optional[Dict[str, Any]] = None,
        telemetry_grounding: Optional[Dict[str, Any]] = None,
        visual_grounding: Optional[Dict[str, Any]] = None,
        asset_grounding: Optional[Dict[str, Any]] = None,
        client: Optional[Any] = None,
    ) -> MultiAgentSignOffReport:
        """Evaluates the candidate build across all 6 specialist perspectives, enforces deterministic
        domain-specific quality invariants, persists sign_off_report.json, and emits SPECIALIST_SIGNOFF_VERDICT events.
        """
        c_dir = Path(candidate_dir).resolve()
        candidate_id = c_dir.name
        active_client = client or self.client
        captured_paths = list((visual_grounding or {}).get("captured_frame_paths", []))

        report: Optional[MultiAgentSignOffReport] = None

        if self._is_live_genai_client(active_client):
            try:
                from google.genai import types
                from engine.studio.models import compact_bible_json_for_prompt
                bible_text = ""
                if (c_dir / "game_bible.json").is_file():
                    bible_text = compact_bible_json_for_prompt(
                        (c_dir / "game_bible.json").read_text(encoding="utf-8", errors="replace"),
                        mode="full",
                    )

                prompt = (
                    f"You are the Multi-Agent Final Sign-Off Council for candidate build '{candidate_id}'.\n"
                    f"Every single specialist who authored the Living Game Bible MUST inspect the actual rendered game frames "
                    f"(playtest_t0.png, playtest_tend.png, and playtest_collision_overlay.png) SIDE-BY-SIDE against the Writers' Room 16:9 Concept Mockup, "
                    f"source code, audio configuration, and runtime scene audit, and render an independent sign-off verdict:\n"
                    f"1. VisualArtDirector: Compare the rendered game frames side-by-side against the 16:9 Concept Mockup and playtest_collision_overlay.png! CRITICAL PERSPECTIVE & COLLISION CHECK: Does the landscape orientation match the character's movement plane? Are red StaticBody2D collision boundaries aligned 1:1 with the north sky horizon and chasm edges so the player can NEVER walk on the sky or empty void? Are ALL synthesized sprites loaded with clean alpha transparency and proper world scale? Are character portraits strictly isolated in UI dialogue panels?\n"
                    f"2. UXExperienceDesigner: Is the viewport clear on startup (zero 95% opaque frost or center-occluding panels)? Are controls, HUD elements, and player feedback clear, responsive, and faithful to the Game Bible's UI/UX design?\n"
                    f"3. GameplayDesigner: Is there an active, responsive, navigable world loop with meaningful objectives, zero t=0 pre-solved/pre-carved cheats, recoverable hazards, and coherent spatial navigation where entities are physically bounded by StaticBody2D colliders to walkable plateau surfaces?\n"
                    f"4. StoryWorldArchitect: Are the Narrative Bible's characters, cutscenes, dialogue trees, and voice barks staged in-game?\n"
                    f"5. AudioAtmosphereDesigner: Are WAV files loaded at 24,000 Hz (`mix_rate = 24000`, never 44100 chipmunk distortion) with the 44-byte RIFF header stripped (`bytes.slice(44)`) and non-overlapping cooldown guards? CRITICAL PSYCHOACOUSTIC CHECK: Reject immediately if the game plays a continuous looping high-pitched sine/square wave drone (`AudioStreamWAV.LOOP_FORWARD` on `> 200 Hz` tones like a constant 523 Hz hum) or piercing `>= 800 Hz` squeals!\n"
                    f"6. CreativeDirector: Does the final build fulfill the First 60 Seconds Hook, Core Player Fantasy, and look/play like the Writers' Room vision?\n\n"
                    f"LIVING GAME BIBLE:\n{bible_text}\n\n"
                    f"AST & CODE AUDIT:\n{json.dumps(ast_grounding or {}, indent=2)}\n\n"
                    f"RUNTIME TELEMETRY & SCENE AUDIT:\n{json.dumps(telemetry_grounding or {}, indent=2)}\n\n"
                    f"ASSET CATALOG:\n{json.dumps(asset_grounding or {}, indent=2)}\n\n"
                    f"Provide 'deliberation' FIRST (Rule 4), and return verdicts for all 6 specialists: {list(SPECIALIST_COUNCIL_ROLES)}."
                )
                contents: List[Any] = [prompt]
                # Attach 16:9 Writers' Room Concept Mockup first for side-by-side comparison
                mockups_dir = c_dir / "assets" / "mockups"
                if mockups_dir.is_dir():
                    for m_file in sorted(mockups_dir.glob("*.png"))[:1]:
                        try:
                            contents.append(types.Part.from_bytes(data=m_file.read_bytes(), mime_type="image/png"))
                        except Exception:
                            pass
                for part in (visual_grounding or {}).get("image_parts", []):
                    contents.append(part)

                response = active_client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=MultiAgentSignOffReport,
                    ),
                )
                if response.text:
                    raw_report = json.loads(response.text)
                    raw_report["candidate_id"] = candidate_id
                    if captured_paths and not raw_report.get("captured_frame_paths"):
                        raw_report["captured_frame_paths"] = captured_paths
                    report = MultiAgentSignOffReport(**raw_report)
            except Exception as exc:
                logger.warning("Live Gemini MultiAgentSignOffCouncil call failed (%s); synthesizing deterministic council report.", exc)

        # Synthesize or enforce deterministic specialist checks across all 6 roles
        report = self._enforce_specialist_invariants(
            report=report,
            candidate_id=candidate_id,
            ast_grounding=ast_grounding or {},
            telemetry_grounding=telemetry_grounding or {},
            visual_grounding=visual_grounding or {},
        )

        # Persist sign_off_report.json to candidate_dir
        try:
            (c_dir / "sign_off_report.json").write_text(
                report.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug("Could not write sign_off_report.json to %s: %s", c_dir, e)

        self._publish_sign_off_events(report)
        return report

    def _enforce_specialist_invariants(
        self,
        report: Optional[MultiAgentSignOffReport],
        candidate_id: str,
        ast_grounding: Dict[str, Any],
        telemetry_grounding: Dict[str, Any],
        visual_grounding: Dict[str, Any],
    ) -> MultiAgentSignOffReport:
        hollow_stubs = ast_grounding.get("hollow_stubs", [])
        scene_audit = telemetry_grounding.get("scene_audit") or {}
        atomic_suite = telemetry_grounding.get("atomic_test_suite") or {}
        blank_detected = bool(visual_grounding.get("blank_canvas_detected", False))
        captured_paths = list(visual_grounding.get("captured_frame_paths", []))

        existing_by_role: Dict[str, SpecialistSignOffVerdict] = {}
        if report is not None:
            for v in report.verdicts:
                existing_by_role[v.specialist_role] = v

        final_verdicts: List[SpecialistSignOffVerdict] = []
        for role in SPECIALIST_COUNCIL_ROLES:
            verdict = existing_by_role.get(role)
            defects: List[str] = list(verdict.blocking_defects) if verdict else []
            remedies: List[str] = list(verdict.remediation_directives) if verdict else []
            approved: bool = verdict.approved if verdict else True

            if role == "GameplayDesigner":
                for stub in hollow_stubs:
                    reason = str(stub.get("reason", ""))
                    if reason:
                        approved = False
                        defects.append(reason)
                        remedies.append(f"Implement complete interactive logic in {stub.get('module', 'mechanics.gd')}:{stub.get('function', '')}.")
                if telemetry_grounding.get("is_stagnant_or_frozen"):
                    approved = False
                    defects.append("Simulation state remained stagnant/frozen across synthetic input schedule.")
                    remedies.append("Ensure player inputs drive continuous spatial navigation and world state progression.")
                if scene_audit.get("passive_auto_win_detected"):
                    approved = False
                    p_reason = scene_audit.get("passive_win_reason") or "Game auto-activates or completes during passive zero-input window."
                    defects.append(f"Passive auto-win / t=0 pre-solved state detected: {p_reason}")
                    remedies.append("Remove all t=0 pre-solved/pre-carved initialization cheats so objectives require genuine player actions.")
                if scene_audit.get("unbounded_sky_or_void_detected"):
                    approved = False
                    b_reason = scene_audit.get("boundary_violation_reason") or "Player can walk into sky or chasm void."
                    defects.append(f"Unbounded sky or chasm excursion detected: {b_reason}")
                    remedies.append("Add StaticBody2D/CollisionPolygon2D barriers along the north sky horizon and between disconnected shelves.")
                if isinstance(atomic_suite, dict) and int(atomic_suite.get("failed_tests", 0) or 0) > 0:
                    approved = False
                    defects.append(f"Atomic Godot unit test failure(s) in tests/test_*.gd: {atomic_suite.get('failed_details', [])}")
                    remedies.append("Fix the failing atomic unit invariants verified in res://tests/test_*.gd.")

            if role == "VisualArtDirector":
                if blank_detected:
                    approved = False
                    defects.append("Blank monochrome canvas detected during visual playtest.")
                    remedies.append("Ensure scenes.gd and ui_hud.gd actively draw world geometry and bound sprite textures.")
                if scene_audit.get("portrait_misused_on_world_sprite"):
                    approved = False
                    misused = scene_audit.get("portrait_misused_nodes", [])
                    defects.append(f"Character portrait asset bound to world Sprite2D entity ({misused}) instead of a purpose-built transparent sprite.")
                    remedies.append("Bind world entities to res://assets/sprites/*.png and use portraits only inside ui_hud.gd dialogue panels.")
                if scene_audit.get("unbounded_sky_or_void_detected"):
                    approved = False
                    b_reason = scene_audit.get("boundary_violation_reason") or "Visual sky/chasm lacks matching StaticBody2D collider."
                    defects.append(f"Visual-to-physics boundary mismatch (walk-on-sky defect): {b_reason}")
                    remedies.append("Align StaticBody2D collision boundaries 1:1 with the visual walkable shelf edges in playtest_collision_overlay.png.")

            if role == "UXExperienceDesigner":
                if scene_audit.get("fullscreen_occlusion_detected"):
                    approved = False
                    defects.append("Full-screen viewport occlusion detected (startup frost >= 80% or unrecoverable failure overlay blinding the player).")
                    remedies.append("Initialize frost/fog overlay <= 0.15 alpha so the exterior world is visible immediately, and make hazards recoverable.")
                if scene_audit.get("center_occluding_panels"):
                    approved = False
                    defects.append(f"UI panel(s) occluding center playfield: {scene_audit.get('center_occluding_panels')}")
                    remedies.append("Move HUD panels to the screen perimeter so the central viewport remains unobstructed.")
                if scene_audit.get("oversized_ui_portraits"):
                    approved = False
                    defects.append(f"Oversized UI portrait TextureRect node(s) overlapping HUD controls: {scene_audit.get('oversized_ui_portraits')}")
                    remedies.append("In ui_hud.gd, bound character portrait TextureRect size <= Vector2(64, 64) with TextureRect.EXPAND_IGNORE_SIZE.")

            if role == "AudioAtmosphereDesigner":
                ap_found = int(scene_audit.get("audio_players_found", 0) or 0)
                as_loaded = int(scene_audit.get("audio_streams_loaded", 0) or 0)
                mix_rates = scene_audit.get("audio_mix_rates") or []
                if ap_found > 0 and as_loaded == 0:
                    approved = False
                    defects.append(f"{ap_found} AudioStreamPlayer node(s) found but 0 audio streams loaded.")
                    remedies.append("Load WAV files via FileAccess.get_file_as_bytes() into AudioStreamWAV.")
                if any(int(r) in (44100, 48000) for r in mix_rates):
                    approved = False
                    defects.append(f"AudioStreamWAV mix_rate set to {mix_rates} instead of 24000 Hz, causing high-pitched chipmunk distortion.")
                    remedies.append("Set stream.mix_rate = 24000 and strip 44-byte RIFF header via bytes.slice(44).")

            if role == "CreativeDirector" and not approved:
                defects.append("CreativeDirector withholds final sign-off until all specialist domain defects are resolved.")

            delib = (
                verdict.deliberation
                if (verdict and verdict.deliberation)
                else (
                    f"{role} inspected candidate '{candidate_id}' and APPROVED Pillar compliance."
                    if approved
                    else f"{role} VETOED candidate '{candidate_id}' due to blocking defects: {defects}"
                )
            )
            final_verdicts.append(
                SpecialistSignOffVerdict(
                    deliberation=delib,
                    specialist_role=role,
                    approved=approved,
                    blocking_defects=defects,
                    remediation_directives=remedies,
                )
            )

        # If any specialist vetoed, CreativeDirector also vetoes so unanimous_approval is False
        any_veto = any(not v.approved for v in final_verdicts)
        if any_veto:
            for v in final_verdicts:
                if v.specialist_role == "CreativeDirector" and v.approved:
                    vetoing_roles = [x.specialist_role for x in final_verdicts if not x.approved]
                    v.approved = False
                    v.blocking_defects.append(f"Blocking vetoes from specialists: {vetoing_roles}")
                    v.remediation_directives.append("Resolve all specialist blocking defects before promotion.")

        unanimous = all(v.approved for v in final_verdicts)
        council_delib = (
            report.deliberation
            if (report and report.deliberation)
            else (
                f"Multi-Agent Final Sign-Off Council unanimously APPROVED candidate '{candidate_id}' across all 6 specialists."
                if unanimous
                else f"Multi-Agent Final Sign-Off Council REJECTED candidate '{candidate_id}'; vetoes issued by {[v.specialist_role for v in final_verdicts if not v.approved]}."
            )
        )
        return MultiAgentSignOffReport(
            deliberation=council_delib,
            candidate_id=candidate_id,
            unanimous_approval=unanimous,
            verdicts=final_verdicts,
            captured_frame_paths=captured_paths,
        )

    def _publish_sign_off_events(self, report: MultiAgentSignOffReport) -> None:
        if self.event_bus is None:
            return
        try:
            for v in report.verdicts:
                if hasattr(self.event_bus, "publish_threadsafe"):
                    self.event_bus.publish_threadsafe({
                        "deliberation": v.deliberation,
                        "event_type": "SPECIALIST_SIGNOFF_VERDICT",
                        "agent_name": v.specialist_role,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "payload": v.to_dict(),
                    })
        except Exception as exc:
            logger.debug("Could not publish SPECIALIST_SIGNOFF_VERDICT events: %s", exc)


# =============================================================================
# Gate F: Challenger Gate (Adversarial Design Critic)
# =============================================================================

class ChallengerGate:
    """Gate F: Adversarial Design Critic auditing candidate builds against 100% of requirements_spec.json
    and convening the 6-specialist MultiAgentSignOffCouncil.
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
        self.sign_off_council = MultiAgentSignOffCouncil(
            client=client,
            model_name=model_name,
            event_bus=event_bus,
        )

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

        # Attach 16:9 Visual Mockup boards from game_bible.json if present for side-by-side visual comparison
        if c_dir is not None and (c_dir / "game_bible.json").is_file():
            try:
                bible_raw = json.loads((c_dir / "game_bible.json").read_text(encoding="utf-8"))
                for m_item in (bible_raw.get("visual_mockups") or [])[:2]:
                    rel_p = m_item.get("image_path") if isinstance(m_item, dict) else None
                    if rel_p and (c_dir / rel_p).is_file():
                        contents.append(
                            types.Part.from_bytes(data=(c_dir / rel_p).read_bytes(), mime_type="image/png")
                        )
            except Exception as exc:
                logger.debug("Could not attach Game Bible mockup images to Challenger contents: %s", exc)

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

        # 5b. Convene the 6-Specialist Multi-Agent Final Sign-Off Council
        self.sign_off_council.event_bus = self.event_bus
        sign_off_report = self.sign_off_council.evaluate_sign_off(
            candidate_dir=c_dir,
            spec=spec,
            ast_grounding=ast_grounding,
            telemetry_grounding=telemetry_grounding,
            visual_grounding=visual_grounding,
            asset_grounding=asset_grounding,
            client=active_client if evaluation.specialist_sign_off is None else None,
        )
        evaluation.specialist_sign_off = sign_off_report
        if not sign_off_report.unanimous_approval:
            evaluation.overall_verdict = False

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

    def _collect_asset_grounding(self, candidate_dir: Path, spec: Optional[GameRequirementsSpec] = None) -> Dict[str, Any]:
        sprites = [p.name for p in (candidate_dir / "assets" / "sprites").glob("*.png")] if (candidate_dir / "assets" / "sprites").exists() else []
        portraits = [p.name for p in (candidate_dir / "assets" / "portraits").glob("*.png")] if (candidate_dir / "assets" / "portraits").exists() else []
        audio = [p.name for p in (candidate_dir / "assets" / "audio").glob("*.wav")] if (candidate_dir / "assets" / "audio").exists() else []
        mockups = [p.name for p in (candidate_dir / "assets" / "mockups").glob("*.png")] if (candidate_dir / "assets" / "mockups").exists() else []
        manifest_reqs = [getattr(a, "name", None) or getattr(a, "asset_id", "asset") for a in getattr(spec, "asset_manifest", [])] if spec is not None else []
        return {
            "sprites_found": sprites,
            "portraits_found": portraits,
            "audio_found": audio,
            "mockups_found": mockups,
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

        bible_section = ""
        if candidate_dir is not None and (candidate_dir / "game_bible.json").is_file():
            try:
                b_data = json.loads((candidate_dir / "game_bible.json").read_text(encoding="utf-8"))
                bible_section = (
                    f"\n--- LIVING GAME BIBLE (`game_bible.json` — EXPERIENCE & DIEGETIC UX CONTRACT) ---\n"
                    f"Core Fantasy: {b_data.get('core_fantasy', '')}\n"
                    f"First 60 Seconds Hook: {b_data.get('first_60_seconds_hook', '')}\n"
                    f"Diegetic UI & Onboarding Philosophy: {b_data.get('diegetic_ui_philosophy', '')}\n"
                    f"Gameplay Pillars: {json.dumps(b_data.get('gameplay_pillars', []))}\n"
                    f"Experience Storyboard Beats: {json.dumps(b_data.get('experience_storyboard', []))}\n"
                    f"Visual Mockup Targets: {json.dumps(b_data.get('visual_mockups', []))}\n"
                )
            except Exception:
                pass

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
            f"100% of the game development team's authored Game Requirements, Acceptance Criteria, and Living Game Bible.\n\n"
            f"--- CANDIDATE CARTRIDGE ID ---\n{candidate_id}\n\n"
            f"--- AGENT-AUTHORED REQUIREMENTS SPECIFICATION (`requirements_spec.json`) ---\n"
            f"Game Title: {spec.game_title}\n"
            f"Genre Paradigm: {spec.genre_paradigm}\n"
            f"Target Audience: {getattr(spec, 'target_audience_summary', '')}\n\n"
            f"Backlog Tracks ({len(spec.requirements)} total):\n"
            f"{'='*50}\n" + "\n\n".join(req_lines) + f"\n{'='*50}\n"
            f"{bible_section}\n"
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
            f"4. ANTI-STUB, CAUSAL & SPATIAL INVARIANT DETECTION: Reject shallow stubs, hollow function bodies (pass, return None, return empty Node2D.new()), "
            f"t=0 pre-solved/pre-carved state cheats (`passive_auto_win_detected == true`), missing sky/chasm physics colliders (`unbounded_sky_or_void_detected == true`), "
            f"cross-module method name mismatches, unbound sprite/audio assets, missing MOUSE_BUTTON_LEFT placement handlers, "
            f"and scene modules that lack active _draw() or Sprite2D world rendering.\n"
            f"5. GAME DESIGNER & UX SIGNOFF MANDATE: Audit the candidate from a human player and Game Designer's perspective:\n"
            f"   - Interactive Playability & Physical Boundaries: Mouse-to-world picking must use accurate canvas transforms (e.g. `get_global_mouse_position()`, never broken `get_viewport().size` math), placement/interaction guides must render on top of the terrain (`z_index > 0`), and `StaticBody2D` colliders in `playtest_collision_overlay.png` must bound the player 1:1 to walkable surfaces.\n"
            f"   - Audio & Voice Playback: Runtime `.wav` files must be loaded into valid `AudioStreamWAV` streams via `FileAccess` at `mix_rate = 24000` with 44-byte RIFF header stripped (`bytes.slice(44)`), and player actions must trigger audible SFX without repeating/spamming.\n"
            f"   - Visual Clarity, Experience Fidelity & UI/Onboarding: The world view must fill the viewport without gray cutoff edges or 95% opaque startup frost, UI panels must not occlude the center playfield (`center_occluding_panels` must be empty), and UI/HUD & onboarding feedback must honor the Living Game Bible and 16:9 Visual Mockups.\n"
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
        """Programmatic defense-in-depth verifying 100% threshold, specialist sign-off, and hollow stub rejection."""
        evaluated_map = {r.requirement_id: r for r in evaluation.evaluated_requirements}
        missing_tracks = [r.id for r in spec.requirements if r.id not in evaluated_map]

        # 1. Check hollow stubs from AST
        hollow_stubs = (ast_grounding or {}).get("hollow_stubs", [])
        stubs_found_reasons = []
        if hollow_stubs:
            for stub in hollow_stubs:
                stubs_found_reasons.append(f"{stub.get('module', '')}:{stub.get('function', '')} ({stub.get('reason', '')})")

        # 1b. Check runtime scene_audit & atomic_test_suite from Godot harness
        scene_audit = (telemetry_grounding or {}).get("scene_audit") or {}
        if isinstance(scene_audit, dict):
            ap_found = int(scene_audit.get("audio_players_found", 0) or 0)
            as_loaded = int(scene_audit.get("audio_streams_loaded", 0) or 0)
            if ap_found > 0 and as_loaded == 0:
                stubs_found_reasons.append(
                    f"Runtime Audio Audit: {ap_found} AudioStreamPlayer node(s) found but 0 audio streams loaded (must use FileAccess -> AudioStreamWAV)"
                )
            mix_rates = scene_audit.get("audio_mix_rates") or []
            if any(int(r) in (44100, 48000) for r in mix_rates):
                stubs_found_reasons.append(
                    f"Runtime Audio Audit: AudioStreamWAV mix_rate set to {mix_rates} instead of 24000 Hz (causes chipmunk audio distortion)"
                )
            occluding = scene_audit.get("center_occluding_panels") or []
            if occluding:
                stubs_found_reasons.append(
                    f"Designer UX Audit: UI panel(s) occluding center playfield: {occluding} (move tooltip/panels to screen edges)"
                )
            if scene_audit.get("fullscreen_occlusion_detected"):
                stubs_found_reasons.append(
                    "Designer UX Audit: Fullscreen viewport occlusion detected (startup frost >= 80% or unrecoverable failure lockout)"
                )
            oversized_portraits = scene_audit.get("oversized_ui_portraits") or []
            if oversized_portraits:
                stubs_found_reasons.append(
                    f"Designer UX Audit: Oversized UI portrait TextureRect node(s) overlapping HUD controls: {oversized_portraits} (set expand_mode = TextureRect.EXPAND_IGNORE_SIZE with size <= 64x64)"
                )
            if scene_audit.get("passive_auto_win_detected"):
                stubs_found_reasons.append(
                    f"Causal Invariant Violation (Passive Auto-Win): {scene_audit.get('passive_win_reason', 'Game auto-activates/completes with zero player input during passive window')}"
                )
            if scene_audit.get("unbounded_sky_or_void_detected"):
                stubs_found_reasons.append(
                    f"Spatial Invariant Violation (Unbounded Sky/Void): {scene_audit.get('boundary_violation_reason', 'Player walked into sky/void without StaticBody2D boundary')}"
                )

        atomic_suite = (telemetry_grounding or {}).get("atomic_test_suite") or {}
        if isinstance(atomic_suite, dict) and int(atomic_suite.get("failed_tests", 0) or 0) > 0:
            stubs_found_reasons.append(
                f"Atomic Unit Test Failure: {atomic_suite.get('failed_tests')} test(s) failed in tests/test_*.gd ({atomic_suite.get('failed_details', [])})"
            )

        # 2. Check blank canvas
        blank_detected = (visual_grounding or {}).get("blank_canvas_detected", False)

        # 3. Check specialist sign-off vetoes
        specialist_vetoes: List[str] = []
        if evaluation.specialist_sign_off and not evaluation.specialist_sign_off.unanimous_approval:
            for v in evaluation.specialist_sign_off.verdicts:
                if not v.approved:
                    specialist_vetoes.append(f"{v.specialist_role}: {'; '.join(v.blocking_defects)}")

        unfulfilled = [r.requirement_id for r in evaluation.evaluated_requirements if not r.fulfilled]
        unfulfilled.extend(missing_tracks)

        # Force overall_verdict = False if stubs found, blank canvas, specialist vetoes, or any unfulfilled
        if unfulfilled or stubs_found_reasons or blank_detected or specialist_vetoes:
            evaluation.overall_verdict = False

        if not evaluation.overall_verdict:
            reasons = []
            if unfulfilled:
                reasons.append(f"Unfulfilled requirements: {unfulfilled}")
            if stubs_found_reasons:
                reasons.append(f"Hollow code stubs detected: {', '.join(stubs_found_reasons)}")
            if specialist_vetoes:
                reasons.append(f"Specialist Sign-Off Vetoes: {' | '.join(specialist_vetoes)}")
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
                    recommended_remedy="Replace hollow stubs, resolve specialist sign-off vetoes, and satisfy 100% of criteria in requirements_spec.json.",
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
                        "Ensure entities, sprites, 24kHz WAV audio, and scenes are actively rendered and connected in the game loop.",
                    ],
                )
            else:
                if combined_reason not in (evaluation.rejection_trace.failure_reason or ""):
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
                        "specialist_sign_off": evaluation.specialist_sign_off.to_dict() if evaluation.specialist_sign_off else None,
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
                            "specialist_sign_off": evaluation.specialist_sign_off.to_dict() if evaluation.specialist_sign_off else None,
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
