"""engine/scheduler/artifact.py
Typed ArtifactMetadataEnvelope with Rationale-First ordering and cryptographic SHA-256 validation.
Enforces Rationale-First Epistemic Architecture (deliberation declared first in all classes).
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid
from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    """Authoritative Artifact Type Registry for Autonomous Game Engine & Showrunner."""
    STEAM_DNA = "STEAM_DNA"
    TELEMETRY_BATCH = "TELEMETRY_BATCH"
    BEHAVIORAL_PROFILE = "BEHAVIORAL_PROFILE"
    GENESIS_SPEC = "GENESIS_SPEC"
    UPDATE_SPEC = "UPDATE_SPEC"
    GAME_MODULE = "GAME_MODULE"
    ASSET_BUNDLE = "ASSET_BUNDLE"
    QA_REPORT = "QA_REPORT"
    PROMOTION_MANIFEST = "PROMOTION_MANIFEST"


class ArtifactIntegrityError(Exception):
    """Raised when cryptographic verification of an artifact payload fails."""
    deliberation: str = "Raised when cryptographic verification of an artifact payload fails."


class ArtifactNotFoundError(Exception):
    """Raised when an external artifact payload file cannot be found."""
    deliberation: str = "Raised when an external artifact payload file cannot be found."


class ArtifactMetadataEnvelope(BaseModel):
    """Authoritative Typed Artifact Envelope with Rationale-First Schema Invariants.
    
    INVARIANT 1: 'deliberation' is the very first declared attribute to satisfy AST checks.
    INVARIANT 2: 'deliberation' is the very first key in dictionary/JSON serialization.
    INVARIANT 3: Cryptographic SHA256 integrity validation for memory and disk payloads.
    INVARIANT 4: Context collapse prevention: allows referencing external payload paths.
    """
    deliberation: str = Field(
        default="",
        description="MANDATORY FIRST PROPERTY: Epistemic justification explaining why this artifact was produced."
    )
    artifact_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUIDv4 identifier for the artifact envelope."
    )
    artifact_type: Union[ArtifactType, str] = Field(
        default=ArtifactType.GENESIS_SPEC,
        description="Categorical type tag for routing and schema validation."
    )
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC creation timestamp."
    )
    source_agent: str = Field(
        default="",
        description="Identifier of the producing agent."
    )
    target_agent: str = Field(
        default="",
        description="Identifier of the intended consumer agent."
    )
    producer_agent: Optional[str] = Field(
        default=None,
        description="Compatibility alias for source_agent."
    )
    consumer_agent: Optional[str] = Field(
        default=None,
        description="Compatibility alias for target_agent."
    )
    sha256_hash: str = Field(
        default="",
        description="Cryptographic SHA256 hex digest of the payload."
    )
    payload: Any = Field(
        default=None,
        description="In-memory structured payload (dictionary, list, or primitive)."
    )
    payload_path: Optional[str] = Field(
        default=None,
        description="Path to external disk file holding raw payload to prevent context collapse."
    )
    input_sha256: Optional[str] = Field(
        default=None,
        description="Provenance tracking: SHA256 of the predecessor artifact input."
    )
    output_sha256: Optional[str] = Field(
        default=None,
        description="Provenance tracking: SHA256 of the output artifact."
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Correlation ID grouping all artifacts from a single genesis/update run."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary supplementary key-value metadata."
    )

    @classmethod
    def compute_sha256_bytes(cls, data: bytes) -> str:
        """Computes SHA-256 hex digest of raw bytes."""
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def canonicalize_payload_bytes(cls, payload: Any) -> bytes:
        """Deterministically canonicalizes an in-memory payload to authentic raw bytes.
        
        - None: returns empty bytes b""
        - bytes / bytearray: returns raw bytes without modification
        - str: returns UTF-8 encoded bytes
        - dict / list / primitives: returns compact canonical JSON bytes (sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        """
        if payload is None:
            return b""
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        if isinstance(payload, str):
            return payload.encode("utf-8")
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        elif hasattr(payload, "to_dict") and callable(payload.to_dict):
            payload = payload.to_dict()
        canonical_json = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        return canonical_json.encode("utf-8")

    @classmethod
    def compute_payload_sha256(cls, payload: Any) -> str:
        """Deterministically canonicalizes and hashes an in-memory payload."""
        return cls.compute_sha256_bytes(cls.canonicalize_payload_bytes(payload))

    @classmethod
    def compute_file_sha256(cls, file_path: Union[str, Path], chunk_size: int = 65536) -> str:
        """Computes SHA-256 hex digest of a file on disk via streaming reads."""
        path = Path(file_path)
        if not path.is_file():
            raise ArtifactNotFoundError(f"Artifact payload file does not exist: {path}")
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def seal(
        cls,
        deliberation: str,
        artifact_type: Union[ArtifactType, str],
        payload: Any = None,
        payload_path: Optional[Union[str, Path]] = None,
        source_agent: str = "",
        target_agent: str = "",
        input_sha256: Optional[str] = None,
        output_sha256: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactMetadataEnvelope:
        """Factory method that calculates cryptographic hashes and returns a sealed envelope."""
        resolved_path_str: Optional[str] = str(payload_path) if payload_path else None
        
        if resolved_path_str:
            calculated_hash = cls.compute_file_sha256(resolved_path_str)
        else:
            calculated_hash = cls.compute_payload_sha256(payload)

        final_output_sha256 = output_sha256 or calculated_hash

        return cls(
            deliberation=deliberation,
            artifact_type=artifact_type,
            source_agent=source_agent,
            target_agent=target_agent,
            producer_agent=source_agent or None,
            consumer_agent=target_agent or None,
            sha256_hash=calculated_hash,
            payload=payload,
            payload_path=resolved_path_str,
            input_sha256=input_sha256,
            output_sha256=final_output_sha256,
            run_id=run_id,
            metadata=metadata or {},
        )

    def verify_integrity(self, raise_on_error: bool = False) -> bool:
        """Verifies that current payload or disk file matches the recorded sha256_hash."""
        if not self.sha256_hash:
            if raise_on_error:
                raise ArtifactIntegrityError(f"Artifact {self.artifact_id} has no recorded sha256_hash")
            return False

        if self.payload_path:
            try:
                actual_hash = self.compute_file_sha256(self.payload_path)
            except (ArtifactNotFoundError, FileNotFoundError):
                if raise_on_error:
                    raise
                return False
        else:
            actual_hash = self.compute_payload_sha256(self.payload)

        is_valid = (actual_hash == self.sha256_hash)
        if not is_valid and raise_on_error:
            raise ArtifactIntegrityError(
                f"Artifact {self.artifact_id} integrity verification failed: "
                f"expected {self.sha256_hash}, calculated {actual_hash}"
            )
        return is_valid

    def to_dict(self) -> Dict[str, Any]:
        """Serializes envelope to a Python dict ensuring 'deliberation' is the first key."""
        raw_dict = self.model_dump()
        ordered_dict: Dict[str, Any] = {"deliberation": self.deliberation}
        for k, v in raw_dict.items():
            if k != "deliberation":
                ordered_dict[k] = v
        return ordered_dict

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serializes envelope to a formatted JSON string preserving key order."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_to_disk(
        self,
        directory: Union[str, Path],
        filename: Optional[str] = None,
        separate_payload_file: bool = False,
    ) -> Path:
        """Atomically saves the envelope and optional separate payload to disk.
        
        Guarantees:
        1. Atomic persistence: Writes to temporary files and atomically renames via os.replace.
        2. Hash invariance: Canonical payload bytes written to disk match sealed sha256_hash.
        3. Binary payload fidelity: Binary data (bytes) written in binary mode without corruption.
        4. Sealed hash protection: Does not mutate sealed sha256_hash.
        """
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        if separate_payload_file and self.payload is not None:
            raw_bytes = self.canonicalize_payload_bytes(self.payload)
            if isinstance(self.payload, (bytes, bytearray)):
                payload_filename = f"payload_{self.artifact_id}.bin"
            elif isinstance(self.payload, str):
                payload_filename = f"payload_{self.artifact_id}.txt"
            else:
                payload_filename = f"payload_{self.artifact_id}.json"

            final_payload_path = target_dir / payload_filename
            temp_payload_path = target_dir / f".{payload_filename}.tmp_{uuid.uuid4().hex}"

            try:
                with open(temp_payload_path, "wb") as pf:
                    pf.write(raw_bytes)
                    pf.flush()
                    os.fsync(pf.fileno())
                os.replace(temp_payload_path, final_payload_path)
            except Exception:
                if temp_payload_path.exists():
                    temp_payload_path.unlink()
                raise

            # Verify written file hash matches sealed hash
            written_hash = self.compute_file_sha256(final_payload_path)
            if self.sha256_hash:
                if written_hash != self.sha256_hash:
                    raise ArtifactIntegrityError(
                        f"Cryptographic hash mismatch writing separate payload file: "
                        f"sealed hash {self.sha256_hash} != written file hash {written_hash}"
                    )
            else:
                self.sha256_hash = written_hash
                if not self.output_sha256:
                    self.output_sha256 = self.sha256_hash

            self.payload_path = str(final_payload_path)
            self.payload = None

        envelope_name = filename or f"artifact_{self.artifact_id}.json"
        final_envelope_path = target_dir / envelope_name
        temp_envelope_path = target_dir / f".{envelope_name}.tmp_{uuid.uuid4().hex}"

        envelope_bytes = self.to_json().encode("utf-8")
        try:
            with open(temp_envelope_path, "wb") as ef:
                ef.write(envelope_bytes)
                ef.flush()
                os.fsync(ef.fileno())
            os.replace(temp_envelope_path, final_envelope_path)
        except Exception:
            if temp_envelope_path.exists():
                temp_envelope_path.unlink()
            raise

        return final_envelope_path

    @classmethod
    def load_from_disk(
        cls,
        envelope_path: Union[str, Path],
        verify: bool = True,
    ) -> ArtifactMetadataEnvelope:
        """Loads an artifact envelope from disk and automatically verifies cryptographic hash."""
        path = Path(envelope_path)
        if not path.is_file():
            raise ArtifactNotFoundError(f"Artifact envelope file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        envelope = cls(**data)
        if verify:
            envelope.verify_integrity(raise_on_error=True)
        return envelope
