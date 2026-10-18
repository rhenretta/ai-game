"""engine/studio/telemetry_architect.py
Telemetry Architect Agent — Feature 22.
Autonomously inspects synthesized gameplay code AST, designs custom telemetry probes,
registers probe metadata in SQLite WAL (telemetry_probes), and injects instrumentation hooks into cartridge code.
Strictly adheres to Rationale-First Epistemic Architecture (deliberation strictly field 0).
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
import ast
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from config import DB_PATH
from engine.loader import validate_cartridge_ast
from engine.studio.models import (
    InstrumentationResult,
    StudioAgentError,
    TelemetryArchitecturePlan,
    TelemetryProbeDesignSpec,
)
from engine.telemetry.db import TelemetryDB
from engine.telemetry.probes import AgentTelemetryProbe

logger = logging.getLogger("TelemetryArchitectAgent")

CANONICAL_PROBE_CATEGORIES = {
    "combat",
    "movement",
    "dialogue",
    "ability",
    "risk_reward",
    "exploration",
}


class TelemetryArchitectAgent:
    """Agent that inspects gameplay code and designs custom telemetry probes."""
    deliberation: str = "Telemetry Architect analyzing gameplay code AST and designing hypothesis-driven probes."

    def __init__(
        self,
        db_path: Union[str, Path] = DB_PATH,
        gemini_client: Optional[Any] = None,
        model_name: str = "gemini-3.8-flash",
        client: Optional[Any] = None,
    ):
        self.db_path = Path(db_path)
        self.db = TelemetryDB(db_path)
        self.gemini_client = gemini_client or client
        self.model_name = model_name

    def inspect_code_and_plan(
        self,
        code_modules: Dict[str, str],
        world_spec: Optional[Any] = None,
        player_dna: Optional[Any] = None,
        hypotheses_to_test: Optional[List[str]] = None,
    ) -> TelemetryArchitecturePlan:
        """Inspects module ASTs and designs custom probe specifications via live Gemini inference."""
        if self.gemini_client is None:
            raise StudioAgentError(
                "TelemetryArchitectAgent requires an active google.genai Client (Rule 5: Zero Silent Fallbacks)"
            )

        ast_targets = self._extract_ast_targets(code_modules)
        try:
            from google.genai import types
            prompt = (
                f"You are the Telemetry Architect AI.\n"
                f"Analyze these code classes/methods:\n{json.dumps(ast_targets, indent=2)}\n"
                f"Design custom telemetry probes to evaluate player preferences and combat behavior.\n"
                f"Categories must be one of: {list(CANONICAL_PROBE_CATEGORIES)}.\n"
                f"Provide deliberation first."
            )
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TelemetryArchitecturePlan,
                ),
            )
            if not response.text:
                raise StudioAgentError("Empty response from Gemini during telemetry architecture planning")
            data = json.loads(response.text)
            return TelemetryArchitecturePlan(**data)
        except Exception as exc:
            logger.error("Gemini telemetry architecture call failed: %s", exc)
            if isinstance(exc, StudioAgentError):
                raise
            raise StudioAgentError(f"TelemetryArchitectAgent failed to design telemetry plan: {exc}") from exc

    def register_probes(self, plan: TelemetryArchitecturePlan) -> int:
        """Registers all probes in plan into SQLite WAL telemetry_probes table."""
        count = 0
        for probe_spec in plan.probes:
            self.db.register_probe(
                probe_id=probe_spec.probe_id,
                probe_name=probe_spec.probe_name,
                category=probe_spec.category,
                description=probe_spec.description,
                hypothesis_target=probe_spec.hypothesis_target,
                schema_json=probe_spec.payload_schema,
            )
            count += 1
        return count

    def instrument_code(
        self,
        code_modules: Dict[str, str],
        plan: TelemetryArchitecturePlan,
    ) -> InstrumentationResult:
        """Injects probe initialization and emission sites into code modules."""
        instrumented: Dict[str, str] = {}
        errors: List[str] = []

        probes_by_mod: Dict[str, List[TelemetryProbeDesignSpec]] = {}
        for p in plan.probes:
            probes_by_mod.setdefault(p.target_module, []).append(p)

        for mod_name, source in code_modules.items():
            if mod_name.endswith(".gd"):
                mod_probes = probes_by_mod.get(mod_name, [])
                if not mod_probes and plan.probes:
                    mod_probes = [plan.probes[0]]
                if not mod_probes:
                    instrumented[mod_name] = source
                    continue
                modified_source = self._inject_gdscript_probes(source, mod_probes)
                instrumented[mod_name] = modified_source
                continue

            mod_probes = probes_by_mod.get(mod_name, [])
            if not mod_probes and plan.probes and "def update(" in source:
                mod_probes = [plan.probes[0]]
            if not mod_probes:
                instrumented[mod_name] = source
                continue

            modified_source = self._inject_probes_into_source(source, mod_probes)
            valid, ast_errs = validate_cartridge_ast(modified_source)
            if not valid:
                errors.extend([f"[{mod_name}] AST error: {e}" for e in ast_errs])
            instrumented[mod_name] = modified_source

        registered_count = self.register_probes(plan)
        success = (len(errors) == 0)

        return InstrumentationResult(
            deliberation=f"Instrumented {len(plan.probes)} custom probes across {len(code_modules)} modules. AST validated: {success}.",
            success=success,
            instrumented_code=instrumented,
            probes_registered=registered_count,
            ast_security_passed=success,
            validation_errors=errors,
        )

    # -----------------------------------------------------------------------
    # Internal AST Target Analysis & Injection
    # -----------------------------------------------------------------------

    def _extract_gdscript_targets(self, mod_name: str, src: str) -> List[Dict[str, Any]]:
        targets: List[Dict[str, Any]] = []
        cls_name = Path(mod_name).stem
        for line in src.splitlines():
            line_str = line.strip()
            if line_str.startswith("func "):
                m = re.match(r"func\s+([a-zA-Z0-9_]+)\s*\((.*?)\)", line_str)
                if m:
                    fn_name = m.group(1)
                    if not fn_name.startswith("__"):
                        raw_args = m.group(2)
                        args = [a.split(":")[0].strip() for a in raw_args.split(",") if a.strip()]
                        targets.append({
                            "module": mod_name,
                            "class": cls_name,
                            "method": fn_name,
                            "args": args,
                        })
        return targets

    def _get_func_body_indent(self, lines: List[str], func_idx: int, uses_tabs: bool) -> str:
        func_line = lines[func_idx]
        func_lead = func_line[:len(func_line) - len(func_line.lstrip())]
        default_indent = func_lead + ("\t" if uses_tabs else "    ")
        for j in range(func_idx + 1, min(len(lines), func_idx + 15)):
            cand = lines[j]
            stripped = cand.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lead = cand[:len(cand) - len(cand.lstrip())]
            if len(lead) > len(func_lead):
                return lead
            break
        return default_indent

    def _inject_gdscript_probes(
        self,
        source: str,
        probes: List[TelemetryProbeDesignSpec],
    ) -> str:
        lines = source.splitlines()
        uses_tabs = sum(1 for line in lines if line.startswith("\t")) >= sum(1 for line in lines if line.startswith("  "))
        injected_any = False
        injected_process = False

        for probe in probes:
            target_fn = probe.target_method
            target_idx = -1
            indent_str = "\t" if uses_tabs else "    "
            for idx, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(f"func {target_fn}(") or (
                    target_fn in ("update", "_process", "_physics_process") and (stripped.startswith("func _process(") or stripped.startswith("func _physics_process("))
                ):
                    target_idx = idx + 1
                    indent_str = self._get_func_body_indent(lines, idx, uses_tabs)
                    if stripped.startswith("func _process(") or stripped.startswith("func _physics_process("):
                        injected_process = True
                    break
            if target_idx != -1 and target_idx <= len(lines):
                probe_stmt = (
                    f'{indent_str}print("[TELEMETRY_EVENT] " + JSON.stringify({{'
                    f'"probe_name": "{probe.probe_name}", "category": "{probe.category}", '
                    f'"payload": call("get_state_snapshot") if has_method("get_state_snapshot") else {{"probe_name": "{probe.probe_name}", "frame": Engine.get_process_frames()}}'
                    f'}}))'
                )
                if f'"probe_name": "{probe.probe_name}"' not in "\n".join(lines):
                    lines.insert(target_idx, probe_stmt)
                    injected_any = True

        # Ensure active emission during simulation: if the module has _process or _physics_process
        # but no probe was hooked directly to it, hook probes[0] so ticks actively emit telemetry
        if not injected_process and probes:
            for idx, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("func _process(") or stripped.startswith("func _physics_process("):
                    target_idx = idx + 1
                    indent_str = self._get_func_body_indent(lines, idx, uses_tabs)
                    probe_stmt = (
                        f'{indent_str}print("[TELEMETRY_EVENT] " + JSON.stringify({{'
                        f'"probe_name": "{probes[0].probe_name}", "category": "{probes[0].category}", '
                        f'"payload": call("get_state_snapshot") if has_method("get_state_snapshot") else {{"probe_name": "{probes[0].probe_name}", "frame": Engine.get_process_frames()}}'
                        f'}}))'
                    )
                    if not any(f'"probe_name": "{probes[0].probe_name}"' in l for l in lines[target_idx:min(len(lines), target_idx + 5)]):
                        lines.insert(target_idx, probe_stmt)
                        injected_any = True
                        injected_process = True
                    break

        if not injected_any and probes:
            for idx, line in enumerate(lines):
                if line.strip().startswith("func "):
                    indent_str = self._get_func_body_indent(lines, idx, uses_tabs)
                    probe_stmt = (
                        f'{indent_str}print("[TELEMETRY_EVENT] " + JSON.stringify({{'
                        f'"probe_name": "{probes[0].probe_name}", "category": "{probes[0].category}", '
                        f'"payload": call("get_state_snapshot") if has_method("get_state_snapshot") else {{"probe_name": "{probes[0].probe_name}", "frame": Engine.get_process_frames()}}'
                        f'}}))'
                    )
                    if probe_stmt.strip() not in [l.strip() for l in lines]:
                        lines.insert(idx + 1, probe_stmt)
                    break
        return "\n".join(lines)

    def _extract_ast_targets(self, modules: Dict[str, str]) -> List[Dict[str, Any]]:
        targets: List[Dict[str, Any]] = []
        for mod_name, src in modules.items():
            if mod_name.endswith(".gd"):
                targets.extend(self._extract_gdscript_targets(mod_name, src))
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith("__"):
                            targets.append({
                                "module": mod_name,
                                "class": class_name,
                                "method": item.name,
                                "args": [a.arg for a in item.args.args],
                            })
        return targets

    def _generate_ast_driven_plan(
        self,
        targets: List[Dict[str, Any]],
        world_spec: Optional[Any] = None,
        player_dna: Optional[Any] = None,
        hypotheses_to_test: Optional[List[str]] = None,
    ) -> TelemetryArchitecturePlan:
        """AST-driven keyword heuristic fallback is prohibited under Rule 2 and Rule 5."""
        raise StudioAgentError(
            "Deterministic keyword-heuristic telemetry planning is prohibited under Rule 2 & Rule 5 (Zero Silent Fallbacks)."
        )

    def _inject_probes_into_source(
        self,
        source: str,
        probes: List[TelemetryProbeDesignSpec],
    ) -> str:
        lines = source.splitlines()

        import_stmt = "from engine.telemetry.probes import AgentTelemetryProbe"
        if import_stmt not in source:
            insert_idx = 0
            try:
                tree = ast.parse(source)
                for stmt in tree.body:
                    if isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant) and isinstance(stmt.value.value, str):
                        insert_idx = max(insert_idx, getattr(stmt, "end_lineno", stmt.lineno))
                    elif isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
                        insert_idx = max(insert_idx, getattr(stmt, "end_lineno", stmt.lineno))
                    else:
                        break
            except Exception:
                for idx, l_str in enumerate(lines):
                    if l_str.strip().startswith("from __future__"):
                        insert_idx = idx + 1
            lines.insert(insert_idx, import_stmt)

        def _build_emit_block(p: TelemetryProbeDesignSpec, indent: int) -> str:
            pad = " " * indent
            return (
                f"{pad}# Telemetry Architect Probe Emission: {p.probe_name}\n"
                f"{pad}if hasattr(self, '_probe_{p.probe_name}') and getattr(self, '_probe_{p.probe_name}') is not None:\n"
                f"{pad}    getattr(self, '_probe_{p.probe_name}').emit(\n"
                f"{pad}        pos=(getattr(self, 'x', 0.0), getattr(self, 'y', 0.0)),\n"
                f"{pad}        vel=(getattr(self, 'vx', 0.0), getattr(self, 'vy', 0.0)),\n"
                f"{pad}        health=getattr(self, 'health', None),\n"
                f"{pad}        data={{'probe_name': '{p.probe_name}', 'category': '{p.category}'}},\n"
                f"{pad}    )\n"
                f"{pad}elif getattr(getattr(self, 'engine', None), 'telemetry_recorder', None) is not None:\n"
                f"{pad}    self.engine.telemetry_recorder.emit_custom_probe(\n"
                f"{pad}        probe_name='{p.probe_name}',\n"
                f"{pad}        category='{p.category}',\n"
                f"{pad}        data={{'probe_name': '{p.probe_name}', 'category': '{p.category}'}},\n"
                f"{pad}    )\n"
                f"{pad}elif getattr(getattr(self, 'engine', None), 'telemetry', None) is not None:\n"
                f"{pad}    self.engine.telemetry.emit_custom_probe(\n"
                f"{pad}        probe_name='{p.probe_name}',\n"
                f"{pad}        category='{p.category}',\n"
                f"{pad}        data={{'probe_name': '{p.probe_name}', 'category': '{p.category}'}},\n"
                f"{pad}    )"
            )

        def _find_method_insert_pos(fn_node: ast.FunctionDef) -> Tuple[int, int]:
            first_stmt = fn_node.body[0]
            if first_stmt.lineno <= fn_node.lineno:
                return -1, 8
            is_docstring = (
                isinstance(first_stmt, ast.Expr)
                and isinstance(getattr(first_stmt, "value", None), ast.Constant)
                and isinstance(first_stmt.value.value, str)
            )
            if is_docstring and len(fn_node.body) > 1 and fn_node.body[1].lineno > getattr(first_stmt, "end_lineno", first_stmt.lineno):
                return fn_node.body[1].lineno - 1, fn_node.body[1].col_offset
            elif is_docstring:
                return getattr(first_stmt, "end_lineno", first_stmt.lineno), first_stmt.col_offset
            else:
                return first_stmt.lineno - 1, first_stmt.col_offset

        def _is_scene_class(cls_node: ast.ClassDef) -> bool:
            for base in cls_node.bases:
                base_id = getattr(base, "id", getattr(base, "attr", ""))
                if base_id in ("BaseScene", "Scene") or "Scene" in str(base_id):
                    return True
            return "Scene" in cls_node.name

        def _inject_into_method(
            src_lines: List[str],
            method_name: str,
            probe_spec: TelemetryProbeDesignSpec,
            scene_only: bool = False,
        ) -> Tuple[List[str], bool]:
            current_src = "\n".join(src_lines)
            flat_lines = current_src.splitlines()
            targets: List[Tuple[int, int]] = []
            try:
                tree = ast.parse(current_src)
                scene_targets: List[Tuple[int, int]] = []
                other_targets: List[Tuple[int, int]] = []
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        is_scene = _is_scene_class(node)
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name == method_name and item.body:
                                pos, indent = _find_method_insert_pos(item)
                                if pos != -1:
                                    if is_scene:
                                        scene_targets.append((pos, indent))
                                    elif not scene_only:
                                        other_targets.append((pos, indent))
                    elif isinstance(node, ast.FunctionDef) and node.name == method_name and node.body and not scene_only:
                        pos, indent = _find_method_insert_pos(node)
                        if pos != -1:
                            other_targets.append((pos, indent))
                targets = scene_targets if scene_targets else other_targets
            except Exception:
                for idx, line in enumerate(flat_lines):
                    if line.strip().startswith(f"def {method_name}(") and line.rstrip().endswith(":"):
                        targets.append((idx + 1, len(line) - len(line.lstrip()) + 4))

            if not targets:
                return flat_lines, False

            # Inject in reverse line order so earlier line indices remain valid
            candidate_lines = list(flat_lines)
            injected_any = False
            for insert_pos, target_indent in sorted(targets, key=lambda x: x[0], reverse=True):
                emit_lines = _build_emit_block(probe_spec, target_indent).splitlines()
                next_lines = candidate_lines[:insert_pos] + emit_lines + candidate_lines[insert_pos:]
                try:
                    ast.parse("\n".join(next_lines))
                    candidate_lines = next_lines
                    injected_any = True
                except SyntaxError:
                    continue

            return (candidate_lines, True) if injected_any else (flat_lines, False)

        scene_update_injected = False
        for p in probes:
            lines, ok = _inject_into_method(lines, p.target_method, p)
            if ok and p.target_method == "update":
                # Check if BaseScene.update was actually injected
                lines_scene, ok_scene = _inject_into_method(lines, "update", p, scene_only=True)
                if ok_scene:
                    lines = lines_scene
                    scene_update_injected = True

        if probes and not scene_update_injected:
            lines, ok_scene = _inject_into_method(lines, "update", probes[0], scene_only=True)
            if not ok_scene:
                lines, _ = _inject_into_method(lines, "update", probes[0], scene_only=False)

        return "\n".join(lines)
