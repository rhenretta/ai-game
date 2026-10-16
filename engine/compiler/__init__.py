"""engine/compiler package - Autonomous SWE Engine Compiler."""
from engine.compiler.codegen import (
    generate_entities_code,
    generate_level_json,
    generate_manifest_json,
    generate_mechanics_code,
)
from engine.compiler.compiler import (
    BytecodeCompilationError,
    CartridgeVerificationError,
    CompilationResult,
    CompiledFile,
    CompilerError,
    EngineCompiler,
    SpecValidationError,
)

__all__ = [
    "EngineCompiler",
    "CompilationResult",
    "CompiledFile",
    "CompilerError",
    "SpecValidationError",
    "BytecodeCompilationError",
    "CartridgeVerificationError",
    "generate_manifest_json",
    "generate_level_json",
    "generate_entities_code",
    "generate_mechanics_code",
]
