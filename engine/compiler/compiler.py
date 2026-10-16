"""engine/compiler/compiler.py
Autonomous SWE Engine Compiler (Milestone 4, Feature 12).
Programmatic SWE worker that compiles GenesisSpec (Episode 0) and EpisodicUpdateSpec (Episode 1+)
into executable Python cartridges, enforces AST security invariants, performs bytecode compilation,
assembles disk packages atomically, verifies cartridge instantiation via CartridgeLoader,
and seals cryptographic artifact envelopes.
"""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

import config
from engine.api.contracts import BaseCartridge
from engine.compiler.codegen import (
    generate_entities_code,
    generate_level_json,
    generate_manifest_json,
    generate_mechanics_code,
)
from engine.director.models import EpisodicUpdateSpec, GenesisSpec
from engine.loader import ASTSecurityError, CartridgeLoader, validate_cartridge_ast
from engine.scheduler.artifact import ArtifactMetadataEnvelope, ArtifactType

logger = logging.getLogger("EngineCompiler")


class CompilerError(Exception):
    """Base exception for compilation failures."""
    deliberation: str = "Base exception for compilation failures."


class SpecValidationError(CompilerError):
    """Raised when an incoming specification violates schema or rationale requirements."""
    deliberation: str = "Raised when an incoming specification violates schema or rationale requirements."


class BytecodeCompilationError(CompilerError):
    """Raised when generated source code fails bytecode compilation."""
    deliberation: str = "Raised when generated source code fails bytecode compilation."


class CartridgeVerificationError(CompilerError):
    """Raised when newly compiled cartridge fails instantiation or contract verification."""
    deliberation: str = "Raised when newly compiled cartridge fails instantiation or contract verification."


class CompiledFile(BaseModel):
    """Encapsulates a single synthesized cartridge file with cryptographic hash."""
    deliberation: str = Field(default="", description="Epistemic rationale explaining file role.")
    relative_path: str = Field(..., description="Relative file path within cartridge (e.g. 'mechanics.py').")
    content: str = Field(..., description="Full text/source content of the file.")
    sha256_hash: str = Field(..., description="SHA-256 hex digest of file content bytes.")
    file_type: str = Field(..., description="'python' | 'json'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "relative_path": self.relative_path,
            "sha256_hash": self.sha256_hash,
            "file_type": self.file_type,
            "size_bytes": len(self.content.encode("utf-8")),
        }


class CompilationResult(BaseModel):
    """MANDATORY RATIONALE-FIRST MODEL: deliberation declared strictly first.
    Complete execution summary emitted by EngineCompiler.
    """
    deliberation: str = Field(default="", description="Epistemic rationale explaining compilation results.")
    spec_id: str = Field(..., description="Identifier of the input specification.")
    episode_number: int = Field(..., description="Target episode counter (0 for Genesis, 1+ for episodic).")
    target_dir: str = Field(..., description="Filesystem directory where cartridge was assembled.")
    files: Dict[str, CompiledFile] = Field(default_factory=dict, description="Synthesized cartridge files.")
    cartridge_class_name: str = Field(..., description="Name of the verified cartridge class.")
    ast_validated: bool = Field(default=False, description="True if AST security checks passed on all Python files.")
    bytecode_compiled: bool = Field(default=False, description="True if compile() verified all bytecode.")
    cartridge_loaded: bool = Field(default=False, description="True if CartridgeLoader instantiated the cartridge cleanly.")
    envelope: Optional[ArtifactMetadataEnvelope] = Field(default=None, description="Cryptographically sealed envelope.")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation": self.deliberation,
            "spec_id": self.spec_id,
            "episode_number": self.episode_number,
            "target_dir": self.target_dir,
            "cartridge_class_name": self.cartridge_class_name,
            "ast_validated": self.ast_validated,
            "bytecode_compiled": self.bytecode_compiled,
            "cartridge_loaded": self.cartridge_loaded,
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "envelope_id": self.envelope.artifact_id if self.envelope else None,
            "created_at_utc": self.created_at_utc,
        }


class EngineCompiler:
    """Autonomous SWE Engine Compiler transforming cognitive specifications into live cartridges."""
    deliberation: str = "Autonomous SWE Engine Compiler transforming cognitive specifications into live cartridges."

    def __init__(
        self,
        cartridges_root: Optional[Union[Path, str]] = None,
        active_cartridge_dir: Optional[Union[Path, str]] = None,
        specs_dir: Optional[Union[Path, str]] = None,
    ):
        self.cartridges_root = Path(cartridges_root or (config.PROJECT_ROOT / "cartridges"))
        self.active_cartridge_dir = Path(active_cartridge_dir or (self.cartridges_root / "active"))
        self.specs_dir = Path(specs_dir or config.SPECS_DIR)
        self.cartridges_root.mkdir(parents=True, exist_ok=True)
        self.specs_dir.mkdir(parents=True, exist_ok=True)

    def compile(
        self,
        spec: Union[GenesisSpec, EpisodicUpdateSpec],
        target_dir: Optional[Union[Path, str]] = None,
        promote_to_active: bool = False,
    ) -> CompilationResult:
        """Main compilation entry point executing the full 6-stage pipeline."""
        # 1. Spec Validation
        if not getattr(spec, "deliberation", ""):
            raise SpecValidationError("Input specification missing mandatory 'deliberation' rationale.")

        is_genesis = isinstance(spec, GenesisSpec)
        episode_number = 0 if is_genesis else spec.target_episode_number
        spec_id = spec.spec_id

        # Determine target assembly directory
        if target_dir is not None:
            cartridge_dir = Path(target_dir).resolve()
        elif is_genesis:
            cartridge_dir = (self.cartridges_root / "genesis").resolve()
        else:
            cartridge_dir = (self.cartridges_root / f"episode_{episode_number:02d}").resolve()

        logger.info("Beginning compilation for spec %s (Episode %d) -> %s", spec_id, episode_number, cartridge_dir)

        # Stage 1: Programmatic Code Generation
        manifest_text = generate_manifest_json(spec)
        level_text = generate_level_json(spec)
        entities_code = generate_entities_code(spec)
        mechanics_code = generate_mechanics_code(spec)

        raw_files = {
            "manifest.json": (manifest_text, "json"),
            "level.json": (level_text, "json"),
            "entities.py": (entities_code, "python"),
            "mechanics.py": (mechanics_code, "python"),
        }

        # Stage 2: AST Security & Invariant Validation
        for rel_name, (content, ftype) in raw_files.items():
            if ftype == "python":
                try:
                    validate_cartridge_ast(content)
                except ASTSecurityError as e:
                    logger.error("AST Security violation in %s: %s", rel_name, e)
                    raise

        # Stage 3: Bytecode Compilation
        for rel_name, (content, ftype) in raw_files.items():
            if ftype == "python":
                try:
                    compile(content, rel_name, "exec")
                except SyntaxError as e:
                    logger.error("Bytecode compilation syntax error in %s: %s", rel_name, e)
                    raise BytecodeCompilationError(f"Syntax error compiling {rel_name}: {e}") from e

        # Stage 4: Atomic Disk Output & Assembly
        cartridge_dir.mkdir(parents=True, exist_ok=True)
        compiled_files: Dict[str, CompiledFile] = {}

        for rel_name, (content, ftype) in raw_files.items():
            target_path = cartridge_dir / rel_name
            temp_path = cartridge_dir / f".{rel_name}.tmp"
            content_bytes = content.encode("utf-8")
            h = hashlib.sha256(content_bytes).hexdigest()

            with open(temp_path, "wb") as f:
                f.write(content_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, target_path)

            compiled_files[rel_name] = CompiledFile(
                deliberation=f"Compiled {rel_name} for Episode {episode_number}",
                relative_path=rel_name,
                content=content,
                sha256_hash=h,
                file_type=ftype,
            )

        # Stage 5: Cartridge Verification
        dir_str = str(cartridge_dir)
        sys_path_added = False
        if dir_str not in sys.path:
            sys.path.insert(0, dir_str)
            sys_path_added = True

        cartridge_class_name = ""
        try:
            cartridge = CartridgeLoader.load_cartridge(cartridge_dir)
            if not isinstance(cartridge, BaseCartridge):
                raise CartridgeVerificationError(f"Loaded object is not an instance of BaseCartridge: {type(cartridge)}")
            cartridge_class_name = type(cartridge).__name__

            # Verify contract methods
            level_spec = cartridge.get_level_spec(episode_number if episode_number > 0 else 1)
            if not level_spec or not level_spec.level_id:
                raise CartridgeVerificationError("Cartridge get_level_spec() returned empty or invalid spec.")

            player = cartridge.create_player(100.0, 100.0)
            if player is None or not hasattr(player, "health"):
                raise CartridgeVerificationError("Cartridge create_player() failed.")

            enemies = cartridge.create_enemies(level_spec)
            hazards = cartridge.create_hazards(level_spec)
            logger.info("Verified cartridge %s: %d enemies, %d hazards", cartridge_class_name, len(enemies), len(hazards))
        except Exception as e:
            if not isinstance(e, CartridgeVerificationError):
                raise CartridgeVerificationError(f"Cartridge loading verification failed: {e}") from e
            raise
        finally:
            # We keep cartridge_dir in sys.path for entities import resolution
            pass

        # Optional Promotion to Active
        if promote_to_active:
            self.active_cartridge_dir.mkdir(parents=True, exist_ok=True)
            for fname in raw_files.keys():
                shutil.copy2(cartridge_dir / fname, self.active_cartridge_dir / fname)
            logger.info("Promoted compiled cartridge to active: %s", self.active_cartridge_dir)

        # Stage 6: Cryptographic Sealing & Artifact Handoff
        file_hashes = {name: cf.sha256_hash for name, cf in compiled_files.items()}
        envelope_payload = {
            "deliberation": f"Cryptographically sealed cartridge package for Episode {episode_number}",
            "spec_id": spec_id,
            "episode_number": episode_number,
            "cartridge_class": cartridge_class_name,
            "cartridge_dir": str(cartridge_dir),
            "file_hashes": file_hashes,
        }

        envelope = ArtifactMetadataEnvelope.seal(
            deliberation=f"Autonomous SWE Compiler verified and sealed cartridge for Episode {episode_number}",
            artifact_type=ArtifactType.GAME_MODULE,
            payload=envelope_payload,
            source_agent="SWECompiler",
            target_agent="BuildSupervisor",
            metadata={
                "file_hashes": file_hashes,
                "target_dir": str(cartridge_dir),
                "episode_number": episode_number,
            }
        )

        # Save envelope to cartridge directory
        envelope.save_to_disk(cartridge_dir, filename="envelope.json")

        result = CompilationResult(
            deliberation=f"Successfully synthesized and validated {cartridge_class_name} with {len(compiled_files)} files.",
            spec_id=spec_id,
            episode_number=episode_number,
            target_dir=str(cartridge_dir),
            files=compiled_files,
            cartridge_class_name=cartridge_class_name,
            ast_validated=True,
            bytecode_compiled=True,
            cartridge_loaded=True,
            envelope=envelope,
        )
        return result
