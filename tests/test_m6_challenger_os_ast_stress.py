"""tests/test_m6_challenger_os_ast_stress.py
Milestone M6 Gate Challenger 1: Adversarial Stress Test Suite.
Exhaustively tests:
1. OS NAME_MAX > 255 & Path Traversal Fuzzing on all UI server endpoints.
2. AST Sandbox Introspection & Reflection Evasion vectors in cartridge compiler.
"""
from __future__ import annotations
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from engine.loader import ASTSecurityError, validate_cartridge_ast
from engine.ui.server import create_app, resolve_asset_safe


@pytest.fixture
def ui_client(tmp_path: Path):
    """Creates a hermetic TestClient for Studio Observability Server."""
    db_file = tmp_path / "telemetry.db"
    cartridges_dir = tmp_path / "cartridges"
    active_dir = cartridges_dir / "active"
    seed_dir = cartridges_dir / "seed"
    (active_dir / "assets" / "audio").mkdir(parents=True, exist_ok=True)
    (active_dir / "assets" / "sprites").mkdir(parents=True, exist_ok=True)
    (seed_dir / "assets" / "audio").mkdir(parents=True, exist_ok=True)

    # Valid dummy assets
    (active_dir / "assets" / "audio" / "valid_tone.wav").write_bytes(b"RIFF" + b"\x00" * 96)
    (active_dir / "assets" / "sprites" / "valid_icon.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    (active_dir / "mechanics.py").write_text("# active mechanics\nclass Mechanics: pass\n", encoding="utf-8")
    (seed_dir / "mechanics.py").write_text("# seed mechanics\nclass Mechanics: pass\n", encoding="utf-8")

    app = create_app(
        db_path=db_file,
        cartridges_root=cartridges_dir,
        active_cartridge_dir=active_dir,
    )
    return TestClient(app, raise_server_exceptions=False)


class TestChallengerOSNameMaxAndTraversal:
    """Adversarial stress-testing of OS parameter overflows and path traversal."""

    @pytest.mark.parametrize("length", [256, 300, 1024])
    def test_name_max_overflow_across_all_asset_endpoints(self, ui_client: TestClient, length: int):
        """Asserts 100% of requests with filenames > 255 chars return HTTP 400 or 404, never 500."""
        oversized = "A" * length

        endpoints = [
            f"/api/assets/audio/{oversized}.wav",
            f"/api/assets/sprites/{oversized}.png",
            f"/api/assets/generic/{oversized}.json",
            f"/api/assets/audio/{oversized}",
            f"/api/assets/sprites/{oversized}",
            f"/api/assets/image/{oversized}.png",
            f"/api/artifacts/audio/{oversized}.wav",
            f"/api/artifacts/sprites/{oversized}.png",
            f"/api/artifacts/custom/{oversized}.bin",
        ]

        for ep in endpoints:
            resp = ui_client.get(ep)
            assert resp.status_code in (400, 404), (
                f"Endpoint {ep} returned unexpected HTTP {resp.status_code} (body: {resp.text})"
            )
            assert resp.status_code != 500, f"Endpoint {ep} crashed with HTTP 500 OSError"

    @pytest.mark.parametrize("length", [256, 300, 1024])
    def test_name_max_overflow_in_cartridge_override_param(self, ui_client: TestClient, length: int):
        """Asserts query parameter cartridge > 255 chars returns HTTP 400 Bad Request, never 500."""
        oversized_cart = "B" * length

        endpoints = [
            f"/api/assets/audio/valid_tone.wav?cartridge={oversized_cart}",
            f"/api/assets/sprites/valid_icon.png?cartridge={oversized_cart}",
            f"/api/assets/generic/valid_tone.wav?cartridge={oversized_cart}",
            f"/api/assets/audio?cartridge={oversized_cart}",
            f"/api/assets/sprites?cartridge={oversized_cart}",
            f"/api/artifacts/audio?cartridge={oversized_cart}",
        ]

        for ep in endpoints:
            resp = ui_client.get(ep)
            assert resp.status_code == 400, (
                f"Endpoint {ep} returned HTTP {resp.status_code} instead of 400"
            )
            assert resp.status_code != 500, f"Endpoint {ep} crashed with HTTP 500"

    @pytest.mark.parametrize("length", [256, 300, 1024])
    def test_name_max_overflow_in_diff_endpoint(self, ui_client: TestClient, length: int):
        """Asserts /api/diff query params > 255 chars return HTTP 400 Bad Request, never 500."""
        oversized = "C" * length

        probes = [
            f"/api/diff?target={oversized}",
            f"/api/diff?candidate={oversized}",
            f"/api/diff?module={oversized}",
            f"/api/diff?target={oversized}&candidate={oversized}&module={oversized}",
        ]

        for ep in probes:
            resp = ui_client.get(ep)
            assert resp.status_code == 400, (
                f"Endpoint {ep} returned HTTP {resp.status_code} instead of 400 (body: {resp.text})"
            )
            assert resp.status_code != 500, f"Endpoint {ep} crashed with HTTP 500"

    def test_path_traversal_fuzz_matrix(self, ui_client: TestClient):
        """Exhaustively fuzzes path traversal vectors across asset routes and asserts zero file disclosure."""
        traversal_vectors = [
            "../etc/passwd",
            "../../etc/passwd",
            "../../../etc/passwd",
            "../../../../etc/passwd",
            "..%2f..%2f..%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "%2e%2e/%2e%2e/etc/passwd",
            "..\\..\\..\\windows\\system32\\cmd.exe",
            "..%5c..%5cwindows",
            "....//....//etc/passwd",
            "/etc/passwd",
            "/etc/hosts",
            "valid_tone%00.wav",
            ".env",
            ".git/config",
            "%2e",
            "%2e%2e",
            "...",
        ]

        base_routes = [
            "/api/assets/audio/",
            "/api/assets/sprites/",
            "/api/assets/generic/",
            "/api/artifacts/audio/",
        ]

        for route in base_routes:
            for vec in traversal_vectors:
                url = f"{route}{vec}"
                resp = ui_client.get(url)
                # Must return 400, 404, or Starlette route 405 — NEVER 200, NEVER 500
                assert resp.status_code in (400, 404, 405), (
                    f"Traversal URL '{url}' returned unexpected status {resp.status_code}"
                )
                assert "root:" not in resp.text, f"Critical security breach: /etc/passwd leaked by {url}"
                assert "PRIVATE KEY" not in resp.text, f"Key leaked by {url}"

    def test_direct_resolve_asset_safe_null_byte_and_traversals(self, tmp_path: Path):
        """Directly exercises resolve_asset_safe against unencoded null bytes and traversal tokens."""
        roots = [tmp_path / "active", tmp_path / "seed"]

        assert resolve_asset_safe("valid_tone\x00.wav", ["audio"], roots) is None
        assert resolve_asset_safe("\x00secret.wav", ["audio"], roots) is None
        assert resolve_asset_safe("audio/\x00", ["audio"], roots) is None
        assert resolve_asset_safe("../../../etc/passwd", ["audio"], roots) is None
        assert resolve_asset_safe("..\\..\\windows", ["audio"], roots) is None
        assert resolve_asset_safe(".hidden_file", ["audio"], roots) is None
        assert resolve_asset_safe("A" * 300, ["audio"], roots) is None

    def test_cartridge_override_traversal_matrix(self, ui_client: TestClient):
        """Fuzzes cartridge query parameter with directory traversal attacks."""
        dangerous_cartridges = [
            "../../",
            "../../../",
            "/etc",
            "/tmp",
            "..%2f..%2f",
            "\\\\127.0.0.1\\c$",
            "active/../../",
            ".",
            "..",
            "...",
        ]

        for cart in dangerous_cartridges:
            resp = ui_client.get(f"/api/assets/audio/valid_tone.wav?cartridge={cart}")
            assert resp.status_code == 400, (
                f"Cartridge traversal '{cart}' returned HTTP {resp.status_code} instead of 400"
            )


class TestChallengerASTSandboxEvasion:
    """Adversarial stress-testing of AST Sandbox Introspection & Reflection Defenses."""

    def test_ast_evasion_object_getattribute(self):
        """Asserts object.__getattribute__ and attribute access trigger ASTSecurityError."""
        payloads = [
            "x = object.__getattribute__(int, '__subclasses__')",
            "def exploit(): return object.__getattribute__(self, '__dict__')",
            "getattr_ref = object.__getattribute__",
            "class Foo:\n    def run(self): return self.__getattribute__('secret')",
            "f = f'{object.__getattribute__}'",
        ]
        for p in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(p)
            assert "Prohibited" in str(exc_info.value)

    def test_ast_evasion_getattr_subclasses(self):
        """Asserts getattr(..., '__subclasses__') and dynamic getattr trigger ASTSecurityError."""
        payloads = [
            "sub = getattr(int, '__subclasses__')",
            "sub = getattr(object, '__subclasses__')",
            "x = getattr(self, '__bases__')",
            "attr = '__subclasses__'\ny = getattr(int, attr)",  # non-literal attribute name
            "z = getattr(int, f'__sub' + 'classes__')",  # computed attribute name
            "g = getattr(int)",  # < 2 arguments
        ]
        for p in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(p)
            assert "Prohibited" in str(exc_info.value)

    def test_ast_evasion_vars_builtin(self):
        """Asserts all forms of vars() trigger ASTSecurityError."""
        payloads = [
            "def run(): return vars()",
            "v = vars(int)",
            "f = vars\nres = f()",
            "data = [vars() for _ in range(5)]",
            "@vars\nclass Decorated: pass",
            "msg = f'Scope: {vars()}'",
            "from builtins import vars",
        ]
        for p in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(p)
            assert "Prohibited" in str(exc_info.value)

    def test_ast_evasion_dir_builtin(self):
        """Asserts all forms of dir() trigger ASTSecurityError."""
        payloads = [
            "def run(): return dir()",
            "d = dir(object)",
            "inspector = dir\nres = inspector()",
            "names = [dir(x) for x in []]",
            "@dir\nclass Inspected: pass",
            "msg = f'Dir: {dir()}'",
            "from builtins import dir",
        ]
        for p in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(p)
            assert "Prohibited" in str(exc_info.value)

    def test_ast_evasion_arbitrary_file_open(self):
        """Asserts arbitrary open() calls trigger ASTSecurityError while preserving level.json."""
        prohibited_open_payloads = [
            "open('/etc/passwd', 'r')",
            "open('secrets.py', 'r')",
            "with open('manifest.json', 'r') as f: data = f.read()",
            "f = open\nh = f('foo.txt', 'r')",
            "open('level.json', 'w')",  # write mode prohibited!
            "open('level.json', mode='w')",  # write mode prohibited!
            "open('level.json', 'a')",  # append mode prohibited!
            "open('level.json', mode='a+')",  # append mode prohibited!
            "from builtins import open",
        ]
        for p in prohibited_open_payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(p)
            assert "Prohibited" in str(exc_info.value)

        # Confirm legitimate cartridge level.json read-only open IS allowed
        allowed_open_payloads = [
            "with open('level.json', 'r') as f:\n    level = f.read()",
            "with open('level.json', mode='r') as f:\n    level = f.read()",
            "f = open('assets/level.json', 'r')",
            "f = open('assets/level.json', 'rb')",
        ]
        for p in allowed_open_payloads:
            is_valid, violations = validate_cartridge_ast(p)
            assert is_valid is True
            assert len(violations) == 0

    def test_ast_evasion_host_termination_exit_quit(self):
        """Asserts exit(), quit(), and breakpoint() trigger ASTSecurityError."""
        payloads = [
            "exit(0)",
            "quit(1)",
            "breakpoint()",
            "terminator = exit\nterminator()",
            "quitter = quit\nquitter()",
            "from builtins import exit",
            "from builtins import quit",
            "from builtins import breakpoint",
            "def crash(): [exit() for _ in range(1)]",
            "def die(): f'{quit()}'",
        ]
        for p in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(p)
            assert "Prohibited" in str(exc_info.value)

    def test_ast_evasion_dunder_chains_and_reflection(self):
        """Asserts dunder chains like ().__class__.__bases__[0].__subclasses__() trigger ASTSecurityError."""
        payloads = [
            "subclasses = ().__class__.__bases__[0].__subclasses__()",
            "mro = int.__mro__",
            "builtins = [].__class__.__base__.__subclasses__()[0].__init__.__globals__['__builtins__']",
            "obj = object()\nobj.__setattr__('evil', True)",
            "obj = object()\nobj.__delattr__('safe')",
            "class Hook:\n    def __init__(self):\n        x = self.__subclasshook__",
        ]
        for p in payloads:
            with pytest.raises(ASTSecurityError) as exc_info:
                validate_cartridge_ast(p)
            assert "Prohibited" in str(exc_info.value)
