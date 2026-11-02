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
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from engine.studio.models import (
    AssetManifestSpec,
    AssetRequirement,
    AudioSynthesisResult,
    CharacterSheet,
    DialogueBarkSpec,
    GameRequirementsSpec,
    StudioAgentError,
)

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
    if data_size <= 0 or len(data) <= 44 or duration < 0.05:
        return (
            False,
            f"WAV file has zero or insufficient PCM audio data (data_size={data_size}, file_bytes={len(data)}, duration={duration:.3f}s)",
            duration,
        )
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
        max_retries = 3
        audio_bytes = None
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                from google.genai import types
                from engine.studio.models import TOKEN_USAGE_TRACKER, _emit_gateway_llm_activity
                tts_label = f"GeminiAudioBackend.tts[{bark_id}]"
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
                tts_stats = TOKEN_USAGE_TRACKER.record_response(response, label=tts_label)
                if tts_stats is not None:
                    _emit_gateway_llm_activity(tts_label, tts_stats)
                if response.candidates:
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, "inline_data") and part.inline_data:
                                    data = getattr(part.inline_data, "data", None)
                                    if isinstance(data, (bytes, bytearray)):
                                        audio_bytes = bytes(data)
                                        break
                if audio_bytes and len(audio_bytes) > 0:
                    break
                logger.warning("Attempt %d: No audio bytes returned for bark '%s', retrying...", attempt, bark_id)
                time.sleep(1.0)
            except Exception as exc:
                last_exc = exc
                logger.warning("Attempt %d failed for audio bark '%s': %s", attempt, bark_id, exc)
                time.sleep(1.0)

        if audio_bytes is None or len(audio_bytes) == 0:
            if last_exc:
                raise VoiceSynthesisError(f"Gemini TTS failed for bark '{bark_id}' after {max_retries} attempts: {last_exc}") from last_exc
            raise VoiceSynthesisError(f"No audio bytes returned from Gemini TTS for bark '{bark_id}' after {max_retries} attempts")

        # Check if raw PCM or contains WAV header
        if audio_bytes[:4] == b"RIFF":
            full_wav = audio_bytes
        else:
            header = create_wav_header(len(audio_bytes), sample_rate=sample_rate, channels=1, bits_per_sample=16)
            full_wav = header + audio_bytes

        if not isinstance(full_wav, (bytes, bytearray)):
            raise VoiceSynthesisError(
                f"Synthesized audio must be bytes or bytearray, got {type(full_wav).__name__} for bark '{bark_id}'"
            )

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


GEMINI_TTS_VOICES: Tuple[str, ...] = ("Aoede", "Charon", "Fenrir", "Kore", "Puck")


class VoiceActingAudioProducerAgent:
    """Specialist Agent producing spoken dialogue voice acting lines (.wav)."""
    deliberation: str = "Voice acting audio producer synthesizing speech lines via Gemini TTS."

    def __init__(
        self,
        client: Optional[Any] = None,
        backend: Optional[AudioSynthesisBackend] = None,
        model_name: str = "gemini-3.8-flash-tts",
        voice_map: Optional[Dict[str, str]] = None,
    ):
        self.model_name = model_name
        self.client = client
        self.character_voices: Dict[str, str] = dict(voice_map or {})
        if backend is not None:
            self.backend = backend
        elif client is not None:
            self.backend = GeminiAudioBackend(client, model_name=model_name)
        else:
            raise StudioAgentError(
                "VoiceActingAudioProducerAgent requires an active google.genai Client or explicit backend (Rule 5: Zero Silent Fallbacks)"
            )

    def register_character_voice(self, speaker_id: str, voice_name: str) -> None:
        """Dynamically registers or overrides voice selection for a speaker."""
        self.character_voices[speaker_id] = voice_name
        self.character_voices[speaker_id.lower()] = voice_name

    def resolve_voice_name(
        self,
        speaker_id: str,
        character: Optional[CharacterSheet] = None,
        bark: Optional[Any] = None,
    ) -> str:
        """Dynamically resolves prebuilt TTS voice name for any speaker,
        supporting custom character names from narrative bible or asset manifest without
        hardcoding to fixed archetypes.
        """
        def _match_timbre(timbre_str: str) -> Optional[str]:
            t = timbre_str.lower().strip()
            if any(k in t for k in ("deep", "heavy", "dark", "resonant", "bass", "gruff", "low")):
                return "Fenrir"
            if any(k in t for k in ("grave", "menacing", "grim", "sinister", "whisper", "raspy")):
                return "Charon"
            if any(k in t for k in ("bright", "light", "cheerful", "playful", "high", "airy")):
                return "Kore"
            if any(k in t for k in ("focused", "sharp", "swift", "clear", "crisp")):
                return "Aoede"
            return None

        # 1. Direct character specification
        if character is not None:
            c_voice = getattr(character, "gemini_voice_name", None)
            c_timbre = getattr(character, "voice_timbre", None)
            if hasattr(character, "model_fields_set"):
                fields_set = character.model_fields_set
            elif hasattr(character, "__fields_set__"):
                fields_set = character.__fields_set__
            else:
                fields_set = set()

            if c_voice and str(c_voice).strip():
                voice_str = str(c_voice).strip()
                # If explicit non-default voice was configured, honor it immediately
                if voice_str != "Aoede":
                    return voice_str

                # When gemini_voice_name is the default 'Aoede' and voice_timbre is provided,
                # evaluate voice_timbre keywords first before falling back to the 'Aoede' default
                timbre_provided = ("voice_timbre" in fields_set) or (c_timbre and c_timbre != "focused")
                if timbre_provided and c_timbre:
                    matched = _match_timbre(str(c_timbre))
                    if matched:
                        return matched

                # If gemini_voice_name was explicitly set by caller, return 'Aoede'
                if "gemini_voice_name" in fields_set:
                    return "Aoede"

        # 2. Registered dynamic character voices
        if speaker_id in self.character_voices:
            return self.character_voices[speaker_id]
        if speaker_id.lower() in self.character_voices:
            return self.character_voices[speaker_id.lower()]

        if character is not None:
            c_name = getattr(character, "name", "")
            if c_name in self.character_voices:
                return self.character_voices[c_name]
            if c_name.lower() in self.character_voices:
                return self.character_voices[c_name.lower()]

        # 3. Known archetype defaults for backward compatibility
        norm_speaker = speaker_id.lower().strip()
        legacy_archetypes = {
            "hero": "Aoede",
            "protagonist": "Aoede",
            "boss": "Fenrir",
            "antagonist": "Fenrir",
            "rival": "Charon",
            "npc": "Puck",
            "companion": "Puck",
            "drone": "Puck",
            "narrator": "Kore",
            "system": "Kore",
        }
        if norm_speaker in legacy_archetypes:
            return legacy_archetypes[norm_speaker]

        # 4. Character personality or timbre hints
        if character is not None:
            timbre = getattr(character, "voice_timbre", "")
            if timbre and isinstance(timbre, str):
                matched = _match_timbre(timbre)
                if matched:
                    return matched
            # Fallback to default 'Aoede' if character specified it
            c_voice = getattr(character, "gemini_voice_name", None)
            if c_voice and str(c_voice).strip() == "Aoede":
                return "Aoede"

        # 5. Dynamic deterministic assignment from speaker_id hash
        digest = hashlib.sha256(speaker_id.encode("utf-8")).hexdigest()
        voice_idx = int(digest[:6], 16) % len(GEMINI_TTS_VOICES)
        return GEMINI_TTS_VOICES[voice_idx]

    def synthesize_voice_line(
        self,
        bark: Union[DialogueBarkSpec, AssetRequirement, Dict[str, Any]],
        output_path: Union[Path, str],
        character: Optional[CharacterSheet] = None,
        voice_name: Optional[str] = None,
        sample_rate: int = 44100,
    ) -> AudioSynthesisResult:
        """Synthesizes a voice line bark and writes .wav to output_path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize bark input
        if isinstance(bark, dict):
            bark_id = bark.get("bark_id") or bark.get("asset_id") or "bark"
            text = bark.get("text") or bark.get("visual_prompt_or_text") or bark.get("description", "")
            speaker_id = bark.get("speaker_id") or bark.get("name") or "speaker"
        elif hasattr(bark, "visual_prompt_or_text") and not getattr(bark, "text", None):
            bark_id = getattr(bark, "asset_id", "bark") or Path(getattr(bark, "target_path", "bark.wav")).stem
            text = getattr(bark, "visual_prompt_or_text", "") or getattr(bark, "description", "")
            speaker_id = getattr(bark, "asset_id", "speaker")
        else:
            bark_id = getattr(bark, "bark_id", "bark")
            text = getattr(bark, "text", "")
            speaker_id = getattr(bark, "speaker_id", "speaker")

        if not text or not str(text).strip():
            raise ValueError("TTS text cannot be empty")

        # Select character voice dynamically
        if not voice_name:
            voice_name = self.resolve_voice_name(speaker_id=str(speaker_id), character=character, bark=bark)

        result = self.backend.synthesize_speech(
            text=str(text).strip(),
            voice_name=voice_name,
            output_path=output_path,
            sample_rate=sample_rate,
            bark_id=str(bark_id),
        )

        # Enforce duration bounds [0.1s, 30.0s]
        if result.duration_seconds < 0.1 or result.duration_seconds > 30.0:
            raise VoiceSynthesisError(
                f"Voice line duration {result.duration_seconds:.2f}s out of acceptable bounds [0.1s, 30.0s] for '{bark_id}'"
            )

        return result

    def generate_sfx(
        self,
        prompt: str,
        output_path: Union[Path, str],
        kind: str = "sfx",
        purpose: str = "",
        loop: bool = False,
        design_model_name: str = "gemini-3.8-flash",
    ) -> AudioSynthesisResult:
        """Designs (LLM recipe) and renders (deterministic DSP) a sound effect or ambient loop to a .wav."""
        from engine.studio.sfx_synth import SFX_SAMPLE_RATE, SfxSynthesisDesigner, render_recipe, write_wav

        if self.client is None:
            raise StudioAgentError(
                f"generate_sfx requires a google.genai Client to design '{kind}' audio; "
                "this agent was built with an explicit backend only (Rule 5)."
            )
        output_path = Path(output_path)
        recipe = SfxSynthesisDesigner(self.client, model_name=design_model_name).design(
            prompt=prompt, kind=kind, purpose=purpose, loop=loop
        )
        write_wav(render_recipe(recipe), output_path)
        valid, err, duration = validate_wav_file(output_path)
        if not valid:
            raise VoiceSynthesisError(f"Rendered SFX WAV validation failed for '{output_path.name}': {err}")
        data = output_path.read_bytes()
        return AudioSynthesisResult(
            deliberation=recipe.deliberation,
            bark_id=output_path.stem,
            file_path=str(output_path),
            sample_rate=SFX_SAMPLE_RATE,
            duration_seconds=duration,
            file_size_bytes=len(data),
            sha256_hash=hashlib.sha256(data).hexdigest(),
        )

    def generate_character_barks(
        self,
        barks: Optional[List[DialogueBarkSpec]] = None,
        output_dir: Optional[Union[Path, str]] = None,
        characters: Optional[List[CharacterSheet]] = None,
        manifest: Optional[Union[List[Any], AssetManifestSpec]] = None,
        requirements_spec: Optional[Any] = None,
        sample_rate: int = 44100,
    ) -> List[AudioSynthesisResult]:
        """Synthesizes dialogue barks and audio assets from list, manifest, or requirements_spec.

        Dynamically derives speaker names, audio cues, and voice mappings with zero hardcoded constraints.
        For games with 0 dialogue characters (e.g. abstract/automation sims), returns an empty list cleanly.
        """
        results: List[AudioSynthesisResult] = []
        out_d = Path(output_dir) if output_dir else None
        if out_d:
            out_d.mkdir(parents=True, exist_ok=True)

        char_map: Dict[str, CharacterSheet] = {}
        if characters:
            for c in characters:
                if getattr(c, "character_id", None):
                    char_map[c.character_id] = c
                if getattr(c, "name", None):
                    char_map[c.name] = c

        # 1. Synthesize explicit barks if provided
        if barks:
            for bark in barks:
                wav_name = Path(bark.voice_clip).name if getattr(bark, "voice_clip", None) else f"{bark.bark_id}.wav"
                if not wav_name.endswith(".wav"):
                    wav_name = f"{wav_name}.wav"
                target_path = (out_d / wav_name) if out_d else Path(bark.voice_clip)
                char_obj = char_map.get(getattr(bark, "speaker_id", "")) or char_map.get(getattr(bark, "speaker_name", ""))
                res = self.synthesize_voice_line(bark=bark, output_path=target_path, character=char_obj, sample_rate=sample_rate)
                results.append(res)

        # 2. Synthesize audio assets declared in manifest or requirements_spec
        manifest_items = None
        if manifest is not None:
            if hasattr(manifest, "assets"):
                manifest_items = manifest.assets
            elif isinstance(manifest, list):
                manifest_items = manifest
        elif requirements_spec is not None:
            if hasattr(requirements_spec, "asset_manifest"):
                manifest_items = requirements_spec.asset_manifest
            elif isinstance(requirements_spec, dict) and "asset_manifest" in requirements_spec:
                manifest_items = requirements_spec["asset_manifest"]

        if manifest_items:
            for item in manifest_items:
                if isinstance(item, dict):
                    atype = item.get("asset_type", "")
                    aid = item.get("asset_id", "")
                    tpath = item.get("target_path", "")
                    text = item.get("visual_prompt_or_text") or item.get("description", "")
                    name = item.get("name") or aid
                else:
                    atype = getattr(item, "asset_type", "")
                    aid = getattr(item, "asset_id", "")
                    tpath = getattr(item, "target_path", "")
                    text = getattr(item, "visual_prompt_or_text", "") or getattr(item, "description", "")
                    name = getattr(item, "name", "") or aid

                # Filter audio assets
                if atype not in ("audio", "audio_bark", "sfx", "voice") and not str(tpath).endswith(".wav"):
                    continue

                wav_name = Path(tpath).name if tpath else f"{aid}.wav"
                if not wav_name.endswith(".wav"):
                    wav_name = f"{wav_name}.wav"
                target_path = (out_d / wav_name) if out_d else (Path(tpath) if tpath else Path(wav_name))

                bark_spec = DialogueBarkSpec(
                    deliberation=f"Dynamic manifest audio asset '{aid}'",
                    bark_id=aid or wav_name.replace(".wav", ""),
                    speaker_id=aid or "narrator",
                    speaker_name=name or "Speaker",
                    trigger_type="manifest_event",
                    text=text or f"Audio alert {aid}",
                    voice_clip=str(target_path),
                )
                char_obj = char_map.get(aid) or char_map.get(name)
                res = self.synthesize_voice_line(bark=bark_spec, output_path=target_path, character=char_obj, sample_rate=sample_rate)
                results.append(res)

        return results

