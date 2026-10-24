"""engine/studio/sfx_synth.py
Model-designed, deterministically rendered sound effects and ambient loops.

The LLM authors a structured synthesis recipe (rationale first, Rule 4); a pure-numpy DSP renderer turns that
recipe into a 16-bit mono PCM .wav. Domain-agnostic: no sound is hardcoded here, and the renderer runs
identically in headless Linux containers (numpy only, no host audio tools). Invalid recipes fail loudly
(Rule 5) instead of being repaired or replaced.
"""
from __future__ import annotations

import json
import logging
import wave
from pathlib import Path
from typing import Any, List, Literal, Optional

import numpy as np
from pydantic import Field

from engine.studio.models import BaseEpistemicModel, StudioAgentError

logger = logging.getLogger("SfxSynth")

SFX_SAMPLE_RATE = 24000
MAX_RECIPE_SECONDS = 30.0
MIN_FREQ_HZ = 20.0

Waveform = Literal["sine", "triangle", "square", "saw", "noise_white", "noise_pink"]


class SfxLayerSpec(BaseEpistemicModel):
    """One synthesized voice. `deliberation` (why this layer exists) precedes all parameters (Rule 4)."""
    source: Waveform = Field(..., description="Base signal generator for this layer.")
    start_s: float = Field(default=0.0, description="Layer onset in seconds from recipe start.")
    duration_s: float = Field(..., description="Layer length in seconds.")
    freq_start_hz: float = Field(default=220.0, description="Fundamental at layer start (tonal sources).")
    freq_end_hz: float = Field(default=220.0, description="Fundamental at layer end; exponential sweep from start.")
    harmonic_amplitudes: List[float] = Field(
        default_factory=list,
        description="Optional relative amplitudes for harmonics 1..N of the fundamental (tonal sources only). "
                    "Empty means use the base waveform directly.",
    )
    band_low_hz: Optional[float] = Field(default=None, description="Optional band-pass lower edge in Hz.")
    band_high_hz: Optional[float] = Field(default=None, description="Optional band-pass upper edge in Hz.")
    tremolo_hz: float = Field(default=0.0, description="Amplitude modulation rate in Hz (0 = none).")
    tremolo_depth: float = Field(default=0.0, description="Amplitude modulation depth 0..1.")
    attack_s: float = Field(default=0.01, description="Linear fade-in seconds.")
    release_s: float = Field(default=0.1, description="Linear fade-out seconds.")
    gain: float = Field(default=0.5, description="Relative layer loudness 0..1.")


class SfxRecipe(BaseEpistemicModel):
    """Full sound design. `deliberation` precedes every decision field (Rule 4)."""
    total_duration_s: float = Field(..., description="Total length in seconds (0.1 to 30).")
    loop_seamlessly: bool = Field(default=False, description="Crossfade the tail into the head for looping playback.")
    loop_crossfade_s: float = Field(default=0.5, description="Crossfade length when loop_seamlessly is true.")
    master_gain: float = Field(default=0.8, description="Final peak amplitude 0..1 after normalization.")
    layers: List[SfxLayerSpec] = Field(..., description="Synthesis layers mixed together.")


def validate_recipe(recipe: SfxRecipe, sample_rate: int = SFX_SAMPLE_RATE) -> None:
    nyq = sample_rate / 2.0
    problems: List[str] = []
    if not (0.1 <= recipe.total_duration_s <= MAX_RECIPE_SECONDS):
        problems.append(f"total_duration_s {recipe.total_duration_s} outside [0.1, {MAX_RECIPE_SECONDS}]")
    if not (0.0 < recipe.master_gain <= 1.0):
        problems.append(f"master_gain {recipe.master_gain} outside (0, 1]")
    if recipe.loop_seamlessly and not (0.0 < recipe.loop_crossfade_s <= recipe.total_duration_s / 2.0):
        problems.append(f"loop_crossfade_s {recipe.loop_crossfade_s} must be in (0, total_duration_s/2]")
    if not recipe.layers:
        problems.append("recipe has no layers")
    for i, ly in enumerate(recipe.layers):
        tag = f"layer[{i}]"
        if ly.duration_s <= 0 or ly.start_s < 0 or ly.start_s + ly.duration_s > recipe.total_duration_s + 1e-6:
            problems.append(f"{tag} window [{ly.start_s}, {ly.start_s + ly.duration_s}] outside recipe length")
        if ly.source.startswith(("sine", "triangle", "square", "saw")):
            for name, f in (("freq_start_hz", ly.freq_start_hz), ("freq_end_hz", ly.freq_end_hz)):
                if not (MIN_FREQ_HZ <= f < nyq):
                    problems.append(f"{tag} {name} {f} outside [{MIN_FREQ_HZ}, {nyq})")
        lo, hi = ly.band_low_hz, ly.band_high_hz
        if (lo is None) != (hi is None):
            problems.append(f"{tag} band_low_hz/band_high_hz must be given together")
        elif lo is not None and not (0 <= lo < hi <= nyq):
            problems.append(f"{tag} band [{lo}, {hi}] invalid for Nyquist {nyq}")
        if not (0.0 <= ly.tremolo_depth <= 1.0) or ly.tremolo_hz < 0:
            problems.append(f"{tag} invalid tremolo (hz={ly.tremolo_hz}, depth={ly.tremolo_depth})")
        if not (0.0 < ly.gain <= 1.0):
            problems.append(f"{tag} gain {ly.gain} outside (0, 1]")
        if ly.attack_s < 0 or ly.release_s < 0 or ly.attack_s + ly.release_s > ly.duration_s + 1e-6:
            problems.append(f"{tag} attack+release exceed duration")
    if problems:
        raise StudioAgentError("Invalid SFX recipe: " + "; ".join(problems))


def _tonal(kind: str, phase: np.ndarray) -> np.ndarray:
    if kind == "sine":
        return np.sin(phase)
    frac = (phase / (2.0 * np.pi)) % 1.0
    if kind == "saw":
        return 2.0 * frac - 1.0
    if kind == "square":
        return np.where(frac < 0.5, 1.0, -1.0)
    return 2.0 * np.abs(2.0 * frac - 1.0) - 1.0  # triangle


def _noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    if kind == "noise_white":
        return white
    spec = np.fft.rfft(white)
    scale = np.ones(len(spec))
    scale[1:] = 1.0 / np.sqrt(np.arange(1, len(spec)))
    pink = np.fft.irfft(spec * scale, n)
    return pink / (np.max(np.abs(pink)) + 1e-12)


def _bandpass(x: np.ndarray, lo: float, hi: float, sr: int) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    edge = max(1.0, 0.05 * (hi - lo))
    mask = np.clip((freqs - (lo - edge)) / (2 * edge), 0, 1) * np.clip(((hi + edge) - freqs) / (2 * edge), 0, 1)
    return np.fft.irfft(spec * mask, len(x))


def _render_layer(ly: SfxLayerSpec, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = max(1, int(round(ly.duration_s * sr)))
    t = np.arange(n) / sr
    if ly.source.startswith("noise"):
        sig = _noise(ly.source, n, rng)
    else:
        ratio = ly.freq_end_hz / ly.freq_start_hz
        freq = ly.freq_start_hz * np.power(ratio, t / max(ly.duration_s, 1e-9))
        phase = 2.0 * np.pi * np.cumsum(freq) / sr
        if ly.harmonic_amplitudes:
            sig = np.zeros(n)
            for k, amp in enumerate(ly.harmonic_amplitudes, start=1):
                h = np.sin(k * phase)
                h[(freq * k) >= sr / 2.0] = 0.0  # no aliasing above Nyquist
                sig += amp * h
        else:
            sig = _tonal(ly.source, phase)
    if ly.band_low_hz is not None and ly.band_high_hz is not None:
        sig = _bandpass(sig, ly.band_low_hz, ly.band_high_hz, sr)
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak
    if ly.tremolo_hz > 0 and ly.tremolo_depth > 0:
        sig = sig * (1.0 - ly.tremolo_depth * (0.5 + 0.5 * np.sin(2.0 * np.pi * ly.tremolo_hz * t)))
    env = np.ones(n)
    a, r = int(ly.attack_s * sr), int(ly.release_s * sr)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    if r > 0:
        env[n - r:] = np.minimum(env[n - r:], np.linspace(1.0, 0.0, r))
    return sig * env * ly.gain


def render_recipe(recipe: SfxRecipe, sample_rate: int = SFX_SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    """Renders a validated recipe to float32 mono samples in [-1, 1]. Deterministic for a given seed."""
    validate_recipe(recipe, sample_rate)
    rng = np.random.default_rng(seed)
    total = int(round(recipe.total_duration_s * sample_rate))
    mix = np.zeros(total)
    for ly in recipe.layers:
        data = _render_layer(ly, sample_rate, rng)
        s = int(round(ly.start_s * sample_rate))
        e = min(total, s + len(data))
        mix[s:e] += data[: e - s]
    if recipe.loop_seamlessly:
        x = int(recipe.loop_crossfade_s * sample_rate)
        fade = np.linspace(0.0, np.pi / 2.0, x)
        head, tail = mix[:x].copy(), mix[-x:].copy()
        mix = mix[:-x]
        mix[:x] = head * np.sin(fade) + tail * np.cos(fade)
    peak = np.max(np.abs(mix))
    if peak <= 1e-9:
        raise StudioAgentError("SFX recipe rendered silence (all layers cancelled or zero-gain).")
    return (mix / peak * recipe.master_gain).astype(np.float32)


def write_wav(samples: np.ndarray, path: Path, sample_rate: int = SFX_SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


class SfxSynthesisDesigner:
    """Asks the model for a synthesis recipe and renders it. No keyword routing: the model decides the design."""

    def __init__(self, client: Any, model_name: str = "gemini-3.8-flash", temperature: float = 0.7):
        if client is None:
            raise StudioAgentError("SfxSynthesisDesigner requires an active google.genai Client (Rule 5).")
        self.client = client
        self.model_name = model_name
        self.temperature = temperature

    def design(self, prompt: str, kind: str, purpose: str = "", loop: bool = False) -> SfxRecipe:
        from google.genai import types

        instruction = (
            "You are a sound designer authoring a synthesis recipe that a deterministic DSP renderer will play.\n"
            f"Asset kind: {kind}. Seamless loop required: {loop}.\n"
            f"Gameplay purpose: {purpose or '(unspecified)'}\n"
            f"Sound brief:\n{prompt}\n\n"
            "Available generators: sine, triangle, square, saw, noise_white, noise_pink, with optional harmonic "
            "amplitudes, exponential frequency sweeps, band-pass limits, tremolo, attack/release envelopes. "
            f"Sample rate {SFX_SAMPLE_RATE} Hz, so all frequencies must be below {SFX_SAMPLE_RATE // 2} Hz. "
            f"Total length must be 0.1-{MAX_RECIPE_SECONDS} seconds and every layer must fit inside it. "
            "Honor any explicit frequency ranges, timings or layer counts in the brief. "
            "Provide `deliberation` first, in the recipe and in every layer, before any parameter (Rule 4)."
        )
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=instruction,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SfxRecipe,
                    temperature=self.temperature,
                ),
            )
            if not response.text:
                raise StudioAgentError("Empty response from SFX design call.")
            recipe = SfxRecipe(**json.loads(response.text))
        except Exception as exc:
            raise StudioAgentError(f"SFX recipe design failed for kind '{kind}': {exc}") from exc
        logger.info("reasoning_rationale[sfx %s]: %s", kind, recipe.deliberation)
        return recipe
