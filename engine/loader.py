"""engine/loader.py
Dynamic Module & Cartridge Loader for hot-swapping game modules and episodic content.
"""
from __future__ import annotations
import ast as _ast
import importlib.machinery as _importlib_machinery
import importlib.util as _importlib_util
import json as _json
import logging as _logging
from pathlib import Path
import shutil as _shutil
import sys as _sys
import types as _types
from typing import Any, Dict, List, Optional, Tuple, Type, Union

_sys.dont_write_bytecode = True

from engine.api.contracts import (
    BaseAbility,
    BaseCartridge,
    BaseEnemy,
    BaseHazard,
    BaseLevelSpec,
    BasePlayer,
)
from engine.entities.player import Player

logger = _logging.getLogger("CartridgeLoader")


class CartridgeError(Exception):
    """Base exception for cartridge loading and validation failures."""
    pass


class ASTSecurityError(CartridgeError):
    """Raised when dynamically loaded game code violates engine safety invariants."""
    pass


__all__ = [
    "ASTSecurityError",
    "BANNED_MODULES",
    "BaseAbility",
    "BaseCartridge",
    "BaseEnemy",
    "BaseHazard",
    "BaseLevelSpec",
    "BasePlayer",
    "CartridgeError",
    "CartridgeLoader",
    "DANGEROUS_ATTRIBUTES",
    "DANGEROUS_NAMES",
    "GenericCartridge",
    "PROHIBITED_CALLS",
    "SecurityNodeVisitor",
    "validate_cartridge_ast",
]


# Banned module roots and submodules that cartridges are strictly forbidden from importing
BANNED_MODULES = {
    # System and OS access
    "os", "sys", "posix", "nt", "pty", "fcntl", "resource", "platform",
    # Process creation and execution
    "subprocess", "multiprocessing", "threading", "_thread", "asyncio",
    # Networking and remote access
    "socket", "http", "urllib", "urllib3", "requests", "ftplib", "smtplib", "xmlrpc",
    # Low-level system and C interop
    "ctypes", "_ctypes",
    # Dynamic imports and builtins manipulation
    "builtins", "importlib", "imp", "pkgutil",
    # Filesystem manipulation utilities
    "shutil",
    # Code execution / bytecode manipulation
    "code", "codeop", "pickle", "marshal", "shelve",
    # Introspection, reflection, and debugging
    "inspect", "gc", "traceback", "linecache", "types",
    # Internal engine runtime and tooling modules forbidden in cartridge code
    "engine.loader", "engine.compiler", "engine.supervisor", "engine.core",
    "engine.steam", "engine.mcp", "engine.scheduler",
}

# Identifiers forbidden as function calls, method calls, or imported symbols
PROHIBITED_CALLS = {
    "system", "popen", "spawn", "fork", "execv", "kill",
    "run", "call", "check_output", "check_call",
    "socket", "connect",
    "rmtree", "unlink", "remove",
    "eval", "exec", "__import__", "compile", "globals", "locals",
    "import_module", "load_module",
    "vars", "dir", "open", "exit", "quit",
}

# Dangerous dunder and introspection attributes used for Python sandbox escapes
DANGEROUS_ATTRIBUTES = {
    "__subclasses__", "__bases__", "__base__", "__class__", "__mro__",
    "__dict__", "__closure__", "__globals__", "__code__", "__builtins__",
    "__getattribute__", "__setattr__", "__delattr__", "__subclasshook__",
    "f_locals", "f_globals", "f_back",
    "sys", "modules", "loader", "importlib",
}

# Dangerous builtin identifiers forbidden anywhere in cartridge code
DANGEROUS_NAMES = {
    "eval", "exec", "__import__", "compile", "globals", "locals", "__builtins__",
    "vars", "dir", "open", "exit", "quit", "breakpoint",
}

# Dynamic reflection primitives inspected for evasion and dangerous targets
DYNAMIC_REFLECTION_FUNCS = {"getattr", "setattr", "delattr", "hasattr", "__getattribute__"}


class SecurityNodeVisitor(_ast.NodeVisitor):
    """AST NodeVisitor enforcing sandbox security invariants on dynamic game cartridges."""

    def __init__(self, raise_on_violation: bool = False):
        super().__init__()
        self.violations: List[str] = []
        self.raise_on_violation = raise_on_violation
        self._allowed_open_nodes: set = set()

    def _record_violation(self, msg: str) -> None:
        self.violations.append(msg)
        if self.raise_on_violation:
            raise ASTSecurityError(msg)

    @staticmethod
    def _is_banned_module(mod_name: str) -> bool:
        if not mod_name:
            return False
        parts = mod_name.split(".")
        if parts[0] in BANNED_MODULES:
            return True
        for i in range(1, len(parts) + 1):
            if ".".join(parts[:i]) in BANNED_MODULES:
                return True
        return False

    @staticmethod
    def _is_allowed_cartridge_open_call(call_node: _ast.Call) -> bool:
        """Allows legacy read-only open(level_file, "r", ...) exclusively for loading local cartridge level.json."""
        if not call_node.args:
            return False
        first_arg = call_node.args[0]
        is_level_file = False
        if isinstance(first_arg, _ast.Name) and first_arg.id == "level_file":
            is_level_file = True
        elif isinstance(first_arg, _ast.Constant) and isinstance(first_arg.value, str) and first_arg.value.endswith("level.json"):
            is_level_file = True
        if not is_level_file:
            return False

        mode = "r"
        if len(call_node.args) >= 2 and isinstance(call_node.args[1], _ast.Constant):
            mode = call_node.args[1].value
        for kw in call_node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, _ast.Constant):
                mode = kw.value.value
        return mode in ("r", "rb")

    def visit_Import(self, node: _ast.Import) -> None:
        for alias in node.names:
            if self._is_banned_module(alias.name):
                self._record_violation(f"Prohibited module import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: _ast.ImportFrom) -> None:
        if node.module:
            if self._is_banned_module(node.module):
                self._record_violation(f"Prohibited module import: '{node.module}'")
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                if self._is_banned_module(full_name):
                    self._record_violation(f"Prohibited module import: '{full_name}'")
        for alias in node.names:
            if (
                alias.name in BANNED_MODULES
                or alias.name in PROHIBITED_CALLS
                or alias.name in DANGEROUS_NAMES
                or alias.name in DANGEROUS_ATTRIBUTES
            ):
                self._record_violation(f"Prohibited imported symbol: '{alias.name}'")
        self.generic_visit(node)

    def visit_Call(self, node: _ast.Call) -> None:
        func_name = None
        if isinstance(node.func, _ast.Name):
            func_name = node.func.id
            if node.func.id == "open" and self._is_allowed_cartridge_open_call(node):
                self._allowed_open_nodes.add(node.func)
            elif node.func.id in PROHIBITED_CALLS or node.func.id in DANGEROUS_NAMES:
                self._record_violation(f"Prohibited function call: '{node.func.id}'")
        elif isinstance(node.func, _ast.Attribute):
            func_name = node.func.attr
            if node.func.attr in PROHIBITED_CALLS or node.func.attr in DANGEROUS_ATTRIBUTES:
                self._record_violation(f"Prohibited method call: '{node.func.attr}'")

        # Inspect dynamic reflection calls (getattr, setattr, delattr, hasattr, __getattribute__)
        if func_name in DYNAMIC_REFLECTION_FUNCS:
            if len(node.args) < 2:
                self._record_violation(
                    f"Prohibited dynamic attribute access: '{func_name}' requires at least 2 arguments"
                )
            else:
                attr_arg = node.args[1]
                if isinstance(attr_arg, _ast.Constant) and isinstance(attr_arg.value, str):
                    attr_val = attr_arg.value
                    if (
                        attr_val in DANGEROUS_ATTRIBUTES
                        or attr_val in PROHIBITED_CALLS
                        or attr_val in BANNED_MODULES
                        or attr_val in DANGEROUS_NAMES
                        or any(attr_val == m.split(".")[-1] for m in BANNED_MODULES if "." in m)
                    ):
                        self._record_violation(
                            f"Prohibited dynamic attribute access in '{func_name}': '{attr_val}'"
                        )
                else:
                    self._record_violation(
                        f"Prohibited dynamic attribute access in '{func_name}': attribute name must be a static string literal"
                    )

        self.generic_visit(node)

    def visit_Attribute(self, node: _ast.Attribute) -> None:
        if node.attr in DANGEROUS_ATTRIBUTES:
            self._record_violation(f"Prohibited attribute access: '{node.attr}'")
        self.generic_visit(node)

    def visit_Name(self, node: _ast.Name) -> None:
        if node.id in DANGEROUS_NAMES:
            if node.id == "open" and node in self._allowed_open_nodes:
                pass
            else:
                self._record_violation(f"Prohibited identifier reference: '{node.id}'")
        self.generic_visit(node)


def validate_cartridge_ast(source_or_path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """Validates that dynamically compiled code does not execute prohibited system calls or imports."""
    if isinstance(source_or_path, Path):
        with open(source_or_path, "r", encoding="utf-8") as f:
            code = f.read()
    elif isinstance(source_or_path, str):
        if "\n" not in source_or_path and (source_or_path.endswith(".py") or Path(source_or_path).is_file()):
            try:
                p = Path(source_or_path)
                if p.is_file():
                    with open(p, "r", encoding="utf-8") as f:
                        code = f.read()
                else:
                    code = source_or_path
            except Exception:
                code = source_or_path
        else:
            code = source_or_path
    else:
        code = str(source_or_path)

    tree = _ast.parse(code)
    visitor = SecurityNodeVisitor(raise_on_violation=False)
    visitor.visit(tree)
    if visitor.violations:
        raise ASTSecurityError(f"AST Security violation: {visitor.violations[0]}")
    return True, []


def _attach_cartridge_helpers(
    cartridge_instance: BaseCartridge,
    cartridge_dir: Path,
    manifest: Dict[str, Any],
    narrative: Dict[str, Any],
    loaded_modules: Dict[str, Any]
) -> None:
    """Attaches metadata, paths, and multi-module resolution helpers to a cartridge instance."""
    cartridge_instance.cartridge_dir = cartridge_dir
    cartridge_instance.manifest = manifest
    cartridge_instance.narrative = narrative
    cartridge_instance.modules = loaded_modules
    cartridge_instance.assets_dir = cartridge_dir / "assets"

    if not hasattr(cartridge_instance, "get_narrative"):
        setattr(cartridge_instance, "get_narrative", lambda: getattr(cartridge_instance, "narrative", {}))
    if not hasattr(cartridge_instance, "get_characters"):
        setattr(cartridge_instance, "get_characters", lambda: getattr(cartridge_instance, "narrative", {}).get("characters", {}))
    if not hasattr(cartridge_instance, "get_dialogue_barks"):
        def _get_barks(trigger_event: str) -> List[Dict[str, Any]]:
            barks = getattr(cartridge_instance, "narrative", {}).get("dialogue_barks", {})
            if isinstance(barks, dict):
                return barks.get(trigger_event, [])
            elif isinstance(barks, list):
                return [b for b in barks if b.get("trigger") == trigger_event]
            return []
        setattr(cartridge_instance, "get_dialogue_barks", _get_barks)
    if not hasattr(cartridge_instance, "get_abilities"):
        def _get_abilities() -> Dict[str, Any]:
            mods = getattr(cartridge_instance, "modules", {})
            if "abilities" in mods:
                mod = mods["abilities"]
                return {name: getattr(mod, name) for name in dir(mod) if not name.startswith("_")}
            return {}
        setattr(cartridge_instance, "get_abilities", _get_abilities)
    if not hasattr(cartridge_instance, "resolve_asset_path"):
        def _resolve_asset_path(relative_path: str) -> Optional[Path]:
            cdir = getattr(cartridge_instance, "cartridge_dir", None)
            if not cdir:
                return None
            clean_rel = relative_path.lstrip("/")
            candidates = [
                cdir / clean_rel,
                cdir / "assets" / clean_rel,
                cdir / "assets" / "sprites" / clean_rel,
                cdir / "assets" / "audio" / clean_rel,
                cdir / "assets" / "tiles" / clean_rel,
            ]
            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()
            return None
        setattr(cartridge_instance, "resolve_asset_path", _resolve_asset_path)
    if not hasattr(cartridge_instance, "get_character_portrait"):
        def _get_portrait(character_id: str) -> Optional[Path]:
            char_info = cartridge_instance.get_characters().get(character_id, {})
            portrait_rel = char_info.get("portrait") or char_info.get("portrait_file")
            if portrait_rel:
                resolved = cartridge_instance.resolve_asset_path(portrait_rel)
                if resolved:
                    return resolved
            return cartridge_instance.resolve_asset_path(f"portrait_{character_id}.png")
        setattr(cartridge_instance, "get_character_portrait", _get_portrait)
    if not hasattr(cartridge_instance, "get_voice_audio"):
        setattr(cartridge_instance, "get_voice_audio", lambda key: cartridge_instance.resolve_asset_path(key))


class GenericCartridge(BaseCartridge):
    """Fallback cartridge binding JSON level specifications or custom scenes."""

    def __init__(
        self,
        directory: Path,
        manifest: Optional[Dict[str, Any]] = None,
        narrative: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(manifest=manifest, cartridge_dir=directory, narrative=narrative)
        self.directory = Path(directory).resolve()
        self.level_specs: Dict[int, BaseLevelSpec] = {}
        _attach_cartridge_helpers(self, self.directory, manifest or {}, narrative or {}, {})
        self._load_specs()

    def _load_specs(self) -> None:
        level_json = self.directory / "level.json"
        if not level_json.exists():
            level_json = self.directory / "level_spec.json"

        if level_json.exists():
            try:
                with open(level_json, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                spec = BaseLevelSpec.from_dict(data)
                self.level_specs[1] = spec
            except Exception as exc:
                logger.warning("Error parsing %s in %s: %s", level_json.name, self.directory, exc)

    def get_level_spec(self, episode_number: int = 1) -> BaseLevelSpec:
        if episode_number in self.level_specs:
            return self.level_specs[episode_number]
        return BaseLevelSpec(
            level_id=f"level_{episode_number:02d}",
            name=f"Level {episode_number}",
            spawn_point=(120.0, 360.0),
            exit_point=(1160.0, 360.0),
            walls=[],
            enemies=[],
            hazards=[]
        )

    def create_player(self, x: float, y: float, telemetry_recorder: Optional[Any] = None) -> Optional[BasePlayer]:
        return Player(x, y, telemetry_recorder)

    def create_enemies(self, spec: Optional[BaseLevelSpec] = None) -> List[BaseEnemy]:
        if spec and getattr(spec, "enemies", None):
            from engine.entities.enemies import ChaserEnemy, RangedEnemy
            enemies = []
            for idx, e_def in enumerate(spec.enemies):
                e_type = str(e_def.get("type", "chaser")).lower()
                ex = float(e_def.get("x", 400.0))
                ey = float(e_def.get("y", 300.0))
                eid = str(e_def.get("id", f"{e_type}_{idx + 1}"))
                if e_type == "ranged":
                    enemies.append(RangedEnemy(eid, ex, ey))
                else:
                    enemies.append(ChaserEnemy(eid, ex, ey))
            return enemies
        return []

    def create_hazards(self, spec: Optional[BaseLevelSpec] = None) -> List[BaseHazard]:
        if spec and getattr(spec, "hazards", None):
            from engine.entities.hazard import SpikeHazard, FloorAcidHazard, LaserSweepHazard
            hazards = []
            for idx, h_def in enumerate(spec.hazards):
                h_type = str(h_def.get("type", "spike")).lower()
                hx = float(h_def.get("x", 400.0))
                hy = float(h_def.get("y", 300.0))
                hid = str(h_def.get("id", f"{h_type}_{idx + 1}"))
                hw = float(h_def.get("w", 48.0))
                hh = float(h_def.get("h", 48.0))
                if "acid" in h_type:
                    hazards.append(FloorAcidHazard(hid, hx, hy, hw, hh))
                elif "laser" in h_type:
                    length = float(h_def.get("length", 120.0))
                    angle = float(h_def.get("angle", 0.0))
                    hazards.append(LaserSweepHazard(hid, hx, hy, length, angle))
                else:
                    hazards.append(SpikeHazard(hid, hx, hy, hw, hh))
            return hazards
        return []

    def get_scenes(self, engine: Any) -> Dict[str, Any]:
        """Returns custom scenes defined in loaded cartridge modules."""
        scenes: Dict[str, Any] = {}
        modules = getattr(self, "modules", {})
        for mod in modules.values():
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (
                    isinstance(attr, type)
                    and hasattr(attr, "update")
                    and hasattr(attr, "render")
                    and not attr_name.startswith("Base")
                ):
                    try:
                        scenes[attr_name] = attr(engine)
                    except Exception:
                        pass
        return scenes

    def get_initial_scene(self, engine: Any) -> Optional[Any]:
        """Returns initial scene if custom scenes are defined in cartridge modules."""
        scenes = self.get_scenes(engine)
        for key in ["MainScene", "PrimaryScene", "PlayingScene", "GameScene"]:
            if key in scenes:
                return scenes[key]
        if scenes:
            return next(iter(scenes.values()))
        return None



def _purge_pycache(target_dir: Path) -> None:
    """Purges __pycache__ directories within target_dir to prevent stale bytecode execution.
    Executed in dedicated function scope to ensure shutil/sys are never in caller frame locals.
    """
    for pycache_dir in target_dir.rglob("__pycache__"):
        if pycache_dir.is_dir():
            _shutil.rmtree(pycache_dir, ignore_errors=True)


def _load_virtual_package(
    pkg_name: str,
    cartridge_dir: Path,
    py_files: List[Path],
    entrypoint_file: Optional[Path],
) -> Dict[str, Any]:
    """Sets up virtual package namespace and loads cartridge submodules cleanly.
    Encapsulated in separate helper to ensure spec, mod, and temp loop variables
    do not linger in load_cartridge frame locals when cartridge __init__ executes.
    """
    # Clean existing modules for this package to support clean reloading
    existing_subkeys = [k for k in _sys.modules if k == pkg_name or k.startswith(f"{pkg_name}.")]
    for k in existing_subkeys:
        _sys.modules.pop(k, None)

    pkg_module = _types.ModuleType(pkg_name)
    pkg_module.__file__ = str(cartridge_dir / "__init__.py")
    pkg_module.__path__ = [str(cartridge_dir)]
    pkg_module.__package__ = pkg_name
    pkg_spec = _importlib_machinery.ModuleSpec(
        name=pkg_name,
        loader=None,
        is_package=True,
    )
    pkg_spec.submodule_search_locations = [str(cartridge_dir)]
    pkg_module.__spec__ = pkg_spec
    _sys.modules[pkg_name] = pkg_module

    # Determine loading order: top-level dependencies first, entrypoint last
    non_entrypoint_files = [
        f for f in py_files
        if f.parent == cartridge_dir and f != entrypoint_file and f.name != "__init__.py"
    ]

    def _module_priority(f: Path) -> int:
        if f.stem == "entities":
            return 1
        if f.stem == "abilities":
            return 2
        return 3

    non_entrypoint_files.sort(key=_module_priority)
    ordered_files = non_entrypoint_files + ([entrypoint_file] if entrypoint_file else [])

    loaded_modules: Dict[str, Any] = {}
    temp_aliases: Dict[str, Any] = {}

    try:
        for py_file in ordered_files:
            sub_name = py_file.stem
            full_sub_name = f"{pkg_name}.{sub_name}"

            spec = _importlib_util.spec_from_file_location(
                full_sub_name,
                py_file,
            )
            if not spec or not spec.loader:
                continue

            mod = _importlib_util.module_from_spec(spec)
            mod.__package__ = pkg_name
            _sys.modules[full_sub_name] = mod
            setattr(pkg_module, sub_name, mod)

            # Provide temporary bare aliases in sys.modules during execution of cartridge modules
            if sub_name not in _sys.modules:
                _sys.modules[sub_name] = mod
                temp_aliases[sub_name] = None
            elif _sys.modules[sub_name] is not mod:
                temp_aliases[sub_name] = _sys.modules[sub_name]
                _sys.modules[sub_name] = mod

            spec.loader.exec_module(mod)
            loaded_modules[sub_name] = mod
    finally:
        # Clean up temporary bare aliases to prevent module pollution across cartridges
        for alias_name, orig_val in temp_aliases.items():
            if orig_val is None:
                _sys.modules.pop(alias_name, None)
            else:
                _sys.modules[alias_name] = orig_val

    return loaded_modules


class CartridgeLoader:
    """Dynamically loads and hot-swaps multi-module game cartridges and episode packages."""

    @staticmethod
    def load_cartridge(path: str | Path) -> BaseCartridge:
        cartridge_dir = Path(path).resolve()
        if not cartridge_dir.exists() or not cartridge_dir.is_dir():
            raise FileNotFoundError(f"Cartridge directory not found: {cartridge_dir}")

        # 1. Load manifest.json if present (fail fast on malformed JSON, Rule 5)
        manifest: Dict[str, Any] = {}
        manifest_path = cartridge_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = _json.load(f)

        # 2. Load narrative.json if present
        narrative_data: Dict[str, Any] = {}
        narrative_path = cartridge_dir / "narrative.json"
        if narrative_path.exists():
            try:
                with open(narrative_path, "r", encoding="utf-8") as f:
                    narrative_data = _json.load(f)
            except Exception as exc:
                logger.warning("Failed to parse narrative.json in %s: %s", cartridge_dir, exc)

        # Invalidate / remove bytecode cache in cartridge directory to prevent stale .pyc execution
        # Stack hygiene: purge cache via helper so shutil/sys never enter load_cartridge frame locals
        _purge_pycache(cartridge_dir)

        # 3. Security AST Sandbox Verification across ALL Python files in the cartridge directory (recursive)
        py_files = sorted(cartridge_dir.rglob("*.py"))
        for py_path in py_files:
            try:
                with open(py_path, "r", encoding="utf-8") as f:
                    source_code = f.read()
                validate_cartridge_ast(source_code)
                compile(source_code, str(py_path), "exec")
            except ASTSecurityError as e:
                logger.error("AST Security violation in %s: %s", py_path.name, e)
                raise
            except SyntaxError as e:
                logger.error("Syntax error compiling %s: %s", py_path.name, e)
                raise

        # 4. Identify Entrypoint Module (mechanics.py or cartridge.py)
        entrypoint_file: Optional[Path] = None
        for candidate_name in ["mechanics.py", "cartridge.py"]:
            candidate_path = cartridge_dir / candidate_name
            if candidate_path.exists():
                entrypoint_file = candidate_path
                break

        # If no Python files exist, return GenericCartridge directly
        if not py_files:
            return GenericCartridge(cartridge_dir, manifest, narrative=narrative_data)

        # 5. Dynamic Virtual Package Creation & Scoped Submodule Loading
        pkg_name = f"cartridge_{cartridge_dir.name}"
        loaded_modules = _load_virtual_package(pkg_name, cartridge_dir, py_files, entrypoint_file)

        # 6. Locate BaseCartridge subclass
        cartridge_instance: Optional[BaseCartridge] = None
        search_modules = []
        if entrypoint_file and entrypoint_file.stem in loaded_modules:
            search_modules.append(loaded_modules[entrypoint_file.stem])
        search_modules.extend(m for m in loaded_modules.values() if m not in search_modules)

        for mod in search_modules:
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseCartridge)
                    and attr not in (BaseCartridge, GenericCartridge)
                ):
                    # Flexible instantiation signature
                    try:
                        cartridge_instance = attr(manifest=manifest, cartridge_dir=cartridge_dir, narrative=narrative_data)
                    except TypeError:
                        try:
                            cartridge_instance = attr(manifest=manifest, cartridge_dir=cartridge_dir)
                        except TypeError:
                            try:
                                cartridge_instance = attr(manifest=manifest)
                            except TypeError:
                                cartridge_instance = attr()
                    logger.info("Loaded custom cartridge class '%s' from %s", attr_name, getattr(mod, "__file__", ""))
                    break
            if cartridge_instance is not None:
                break

        if cartridge_instance is None:
            cartridge_instance = GenericCartridge(cartridge_dir, manifest, narrative=narrative_data)

        # 7. Inject metadata and multi-module capabilities onto cartridge instance
        _attach_cartridge_helpers(
            cartridge_instance=cartridge_instance,
            cartridge_dir=cartridge_dir,
            manifest=manifest,
            narrative=narrative_data,
            loaded_modules=loaded_modules,
        )

        return cartridge_instance
