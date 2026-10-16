"""engine/studio/audio.py
Voice Acting & Audio Producer Agent — Feature 19.
Synthesizes authentic spoken dialogue audio files (.wav) via Gemini TTS.
Generates and validates 44-byte standard RIFF/WAVE headers, enforces duration bounds [0.1s, 30.0s],
sanitizes SSML characters, and adheres strictly to Rule 5: Zero Silent Degradation.
Zero occurrences of legacy branding tokens.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import hashlib
import logging
import math
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

from engine.studio.models import AudioSynthesisResult, CharacterSheet, DialogueBarkSpec, StudioAgentError

logger = logging.getLogger("VoiceActingAudioProducerAgent")


class VoiceSynthesisError(Exception):
    """Raised when voice line synthesis fails (Rule 5: Zero Silent Degradation)."""
    deliberation: str = "Raised when voice line synthesis fails or violates duration/format invariants."


def escape_ssml(text: str) -> str:
    """Escapes XML/SSML sensitive characters in raw dialogue lines."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def create_wav_header(
    pcm_bytes_len: int,
    sample_rate: int = 44100,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Generates standard 44-byte RIFF/WAVE header."""
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    riff_chunk_size = 36 + pcm_bytes_len
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_chunk_size,
        b"WAVE",
        b"fmt ",
        16,              # Subchunk1Size for PCM
        1,               # AudioFormat 1 = PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        pcm_bytes_len,
    )


def validate_wav_file(path: Path) -> Tuple[bool, str, float]:
    """Validates RIFF header, format markers, sample rate, and calculates duration in seconds."""
    if not path.is_file():
        return False, f"File does not exist: {path}", 0.0

    data = path.read_bytes()
    if len(data) < 44:
        return False, f"File size {len(data)} bytes is smaller than 44-byte RIFF header", 0.0

    if data[:4] != b"RIFF":
        return False, f"Invalid RIFF magic bytes: {data[:4]!r}", 0.0
    if data[8:12] != b"WAVE":
        return False, f"Invalid WAVE format marker: {data[8:12]!r}", 0.0
    if data[12:16] != b"fmt ":
        return False, f"Invalid fmt subchunk marker: {data[12:16]!r}", 0.0

    try:
        channels = struct.unpack("<H", data[22:24])[0]
        sample_rate = struct.unpack("<I", data[24:28])[0]
        bits_per_sample = struct.unpack("<H", data[34:36])[0]
        data_size = struct.unpack("<I", data[40:44])[0]
    except Exception as exc:
        return False, f"Corrupted RIFF header unpack: {exc}", 0.0

    bytes_per_sec = sample_rate * channels * (bits_per_sample // 8)
    duration = data_size / max(1, bytes_per_sec)
    return True, "", duration


class AudioSynthesisBackend(ABC):
    """Abstract base class for audio speech synthesis engines."""

    @abstractmethod
    def synthesize_speech(
        self,
        text: str,
        voice_name: str,
        output_path: Path,
        sample_rate: int = 44100,
        bark_id: str = "bark",
    ) -> AudioSynthesisResult:
        """Synthesizes speech audio and writes to output_path."""
        pass


class GeminiAudioBackend(AudioSynthesisBackend):
    """Live Google Gemini TTS speech synthesis backend."""

    def __init__(self, client: Any, model_name: str = "gemini-3.8-flash-tts"):
        self.client = client
        self.model_name = model_name

    def synthesize_speech(
        self,
        text: str,
        voice_name: str,
        output_path: Path,
        sample_rate: int = 44100,
        bark_id: str = "bark",
    ) -> AudioSynthesisResult:
        escaped_text = escape_ssml(text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=escaped_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                        )
                    ),
                ),
            )
            # Find audio part
            audio_bytes = None
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "inline_data") and part.inline_data:
                                audio_bytes = part.inline_data.data
                                break

            if not audio_bytes:
                raise VoiceSynthesisError(f"No audio bytes returned from Gemini TTS for bark '{bark_id}'")

            # Check if raw PCM or contains WAV header
            if audio_bytes[:4] == b"RIFF":
                full_wav = audio_bytes
            else:
                header = create_wav_header(len(audio_bytes), sample_rate=sample_rate, channels=1, bits_per_sample=16)
                full_wav = header + audio_bytes

            output_path.write_bytes(full_wav)
            valid, err, duration = validate_wav_file(output_path)
            if not valid:
                raise VoiceSynthesisError(f"Synthesized WAV validation failed: {err}")

            sha256 = hashlib.sha256(full_wav).hexdigest()
            return AudioSynthesisResult(
                deliberation=f"Synthesized speech via Gemini TTS '{self.model_name}' for voice '{voice_name}'.",
                bark_id=bark_id,
                file_path=str(output_path),
                sample_rate=sample_rate,
                duration_seconds=duration,
                file_size_bytes=len(full_wav),
                sha256_hash=sha256,
            )
        except Exception as exc:
            if isinstance(exc, VoiceSynthesisError):
                raise
            raise VoiceSynthesisError(f"Gemini TTS generation failed for bark '{bark_id}': {exc}") from exc


class OfflineMockAudioBackend(AudioSynthesisBackend):
    """Deterministic offline audio synthesis backend for unit tests."""

    def synthesize_speech(
        self,
        text: str,
        voice_name: str,
        output_path: Path,
        sample_rate: int = 44100,
        bark_id: str = "bark",
    ) -> AudioSynthesisResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Approximate duration based on word count (150 words per minute ~ 2.5 words/sec)
        words = max(1, len(text.split()))
        duration = min(30.0, max(0.5, words * 0.4))

        # Generate smooth synthesized voice harmonics
        num_samples = int(duration * sample_rate)
        # Base pitch based on voice
        base_freq = 140.0 if voice_name in ["Fenrir", "Charon"] else (220.0 if voice_name in ["Aoede", "Kore"] else 180.0)

        samples: List[int] = []
        for i in range(num_samples):
            t = i / sample_rate
            # Modulated speech formant envelope
            envelope = math.sin(math.pi * (i / num_samples))
            val = math.sin(2.0 * math.pi * base_freq * t) * 0.6 + math.sin(4.0 * math.pi * base_freq * t) * 0.3
            sample_val = int(val * envelope * 12000.0)
            sample_val = max(-32768, min(32767, sample_val))
            samples.append(sample_val)

        pcm_data = struct.pack(f"<{len(samples)}h", *samples)
        header = create_wav_header(len(pcm_data), sample_rate=sample_rate, channels=1, bits_per_sample=16)
        full_wav = header + pcm_data

        output_path.write_bytes(full_wav)
        valid, err, calc_duration = validate_wav_file(output_path)
        if not valid:
            raise VoiceSynthesisError(f"Offline WAV validation failed: {err}")

        sha256 = hashlib.sha256(full_wav).hexdigest()
        return AudioSynthesisResult(
            deliberation=f"Deterministic offline synthesized audio for voice '{voice_name}' duration {calc_duration:.2f}s.",
            bark_id=bark_id,
            file_path=str(output_path),
            sample_rate=sample_rate,
            duration_seconds=calc_duration,
            file_size_bytes=len(full_wav),
            sha256_hash=sha256,
        )


class VoiceActingAudioProducerAgent:
    """Specialist Agent producing spoken dialogue voice acting lines (.wav)."""
    deliberation: str = "Voice acting audio producer synthesizing speech lines via Gemini TTS."

    def __init__(
        self,
        client: Optional[Any] = None,
        backend: Optional[AudioSynthesisBackend] = None,
        model_name: str = "gemini-3.8-flash-tts",
    ):
        self.model_name = model_name
        if backend is not None:
            self.backend = backend
        elif client is not None:
            self.backend = GeminiAudioBackend(client, model_name=model_name)
        else:
            raise StudioAgentError(
                "VoiceActingAudioProducerAgent requires an active google.genai Client or explicit backend (Rule 5: Zero Silent Fallbacks)"
            )

    def synthesize_voice_line(
        self,
        bark: DialogueBarkSpec,
        output_path: Path,
        character: Optional[CharacterSheet] = None,
        sample_rate: int = 44100,
    ) -> AudioSynthesisResult:
        """Synthesizes a voice line bark and writes .wav to output_path."""
        if not bark.text or not bark.text.strip():
            raise ValueError("TTS text cannot be empty")

        # Select character voice dynamically
        if character is not None and getattr(character, "gemini_voice_name", None):
            voice_name = character.gemini_voice_name
        else:
            voice_map = {
                "hero": "Aoede",
                "boss": "Fenrir",
                "npc": "Puck",
                "companion": "Puck",
                "rival": "Charon",
            }
            voice_name = voice_map.get(bark.speaker_id, "Aoede")

        result = self.backend.synthesize_speech(
            text=bark.text,
            voice_name=voice_name,
            output_path=output_path,
            sample_rate=sample_rate,
            bark_id=bark.bark_id,
        )

        # Enforce duration bounds [0.1s, 30.0s]
        if result.duration_seconds < 0.1 or result.duration_seconds > 30.0:
            raise VoiceSynthesisError(
                f"Voice line duration {result.duration_seconds:.2f}s out of acceptable bounds [0.1s, 30.0s] for '{bark.bark_id}'"
            )

        return result
