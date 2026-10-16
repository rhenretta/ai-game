"""tests/adversarial/test_tier5_ast_evasion_hardening.py
Tier 5 Adversarial Test Suite 2: AST Security Invariants, Dunder Reflection & Builtin Evasion Hardening.
"""
from __future__ import annotations
import pytest

from engine.loader import ASTSecurityError, validate_cartridge_ast


class TestTier5ASTEvasionHardening:
    """Verifies that cartridge code cannot bypass AST sandboxing via introspection, dunders, or builtins."""

    def test_ast_rejection_getattribute_reflection_escape(self):
        """Validates rejection of __getattribute__ calls and attribute access."""
        payloads = [
            "obj = int\nsub = object.__getattribute__(obj, '__subclasses__')",
            "obj = int\nsub = getattr(obj, '__subclasses__')",
            "class Attack:\n    def run(self):\n        return self.__getattribute__('__class__')",
            "x = int.__getattribute__",
        ]

        for payload in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(payload)
            assert "Prohibited" in str(exc_info.value)

    def test_ast_rejection_vars_builtin_evasion(self):
        """Validates rejection of vars() calls and identifier references."""
        payloads = [
            "def exploit():\n    return vars()",
            "class Target:\n    pass\nx = vars(Target)",
            "f = vars\nres = f()",
        ]

        for payload in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(payload)
            assert "Prohibited" in str(exc_info.value)
            assert "vars" in str(exc_info.value)

    def test_ast_rejection_dir_builtin_evasion(self):
        """Validates rejection of dir() calls and identifier references."""
        payloads = [
            "def inspect_scope():\n    return dir()",
            "x = dir(object)",
            "inspector = dir",
        ]

        for payload in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(payload)
            assert "Prohibited" in str(exc_info.value)
            assert "dir" in str(exc_info.value)

    def test_ast_rejection_host_termination_exit_quit(self):
        """Validates rejection of exit(), quit(), and breakpoint() calls designed to kill the host engine."""
        payloads = [
            "def on_init():\n    exit(0)",
            "def on_update():\n    quit(1)",
            "killer = exit\nkiller()",
            "terminator = quit\nterminator()",
            "def debug_hook():\n    breakpoint()",
        ]

        for payload in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(payload)
            assert "Prohibited" in str(exc_info.value)

    def test_ast_rejection_arbitrary_file_open(self):
        """Validates rejection of arbitrary file open() operations."""
        payloads = [
            "with open('/etc/passwd', 'r') as f:\n    data = f.read()",
            "with open('exploit.txt', 'w') as f:\n    f.write('pwned')",
            "opener = open\nh = opener('leak.dat', 'wb')",
            "f = open('/tmp/test', 'r')",
        ]

        for payload in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(payload)
            assert "Prohibited" in str(exc_info.value)
            assert "open" in str(exc_info.value)

    def test_ast_rejection_subclasshook_setattr_delattr_dunders(self):
        """Validates rejection of dangerous dunders __setattr__, __delattr__, and __subclasshook__."""
        payloads = [
            "class Hook:\n    def __init__(self):\n        x = self.__subclasshook__",
            "class Tamper:\n    def patch(self, target):\n        target.__setattr__('flag', True)",
            "class Wipe:\n    def wipe(self, target):\n        target.__delattr__('critical')",
        ]

        for payload in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(payload)
            assert "Prohibited" in str(exc_info.value)

    def test_ast_comprehension_and_lambda_obfuscated_escapes(self):
        """Validates that comprehension expressions and lambdas cannot hide forbidden dunders."""
        payloads = [
            "classes = [c for c in ().__class__.__bases__[0].__subclasses__()]",
            "getter = lambda: getattr(int, '__subclasses__')",
            "gen = (x for x in [].__class__.__base__.__subclasses__())",
            "leak = [f.f_locals for f in []]",
        ]

        for payload in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(payload)
            assert "Prohibited" in str(exc_info.value)

    def test_ast_benign_gameplay_code_acceptance(self):
        """Ensures legitimate game mechanics, classes, math, and structures pass validation."""
        benign_code = """
import math
from typing import Dict, List, Tuple

class PlayerMechanics:
    def __init__(self, x: float = 100.0, y: float = 100.0):
        self.x = x
        self.y = y
        self.hp = 100
        self.score = 0
        self.inventory: List[str] = []

    def move(self, dx: float, dy: float) -> Tuple[float, float]:
        dist = math.hypot(dx, dy)
        if dist > 0.0:
            self.x += (dx / dist) * 5.0
            self.y += (dy / dist) * 5.0
        return (self.x, self.y)

    def take_damage(self, amount: int) -> int:
        self.hp = max(0, self.hp - amount)
        return self.hp
"""
        is_valid, violations = validate_cartridge_ast(benign_code)
        assert is_valid is True
        assert len(violations) == 0
