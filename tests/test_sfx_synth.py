import json
import types

import numpy as np
import pytest

from engine.studio.audio import VoiceActingAudioProducerAgent, OfflineMockAudioBackend, validate_wav_file
from engine.studio.models import StudioAgentError
from engine.studio.sfx_synth import SFX_SAMPLE_RATE, SfxRecipe, render_recipe


def _recipe(**over):
    base = {
        "deliberation": "Two layers: a tone sweep and a noise band.",
        "total_duration_s": 1.0,
        "loop_seamlessly": False,
        "master_gain": 0.8,
        "layers": [
            {"deliberation": "tone", "source": "sine", "duration_s": 1.0, "freq_start_hz": 200.0,
             "freq_end_hz": 800.0, "gain": 0.6},
            {"deliberation": "hiss", "source": "noise_white", "start_s": 0.2, "duration_s": 0.6,
             "band_low_hz": 2000.0, "band_high_hz": 4000.0, "gain": 0.4},
        ],
    }
    base.update(over)
    return SfxRecipe(**base)


class _FakeClient:
    def __init__(self, payload):
        self.models = types.SimpleNamespace(
            generate_content=lambda **kw: types.SimpleNamespace(text=json.dumps(payload))
        )


def test_render_is_deterministic_and_bounded():
    a, b = render_recipe(_recipe()), render_recipe(_recipe())
    assert np.array_equal(a, b)
    assert len(a) == SFX_SAMPLE_RATE
    assert 0.79 <= float(np.max(np.abs(a))) <= 0.81


def test_band_limited_noise_energy_stays_in_band():
    r = _recipe(layers=[{"deliberation": "x", "source": "noise_white", "duration_s": 1.0,
                         "band_low_hz": 2000.0, "band_high_hz": 4000.0, "gain": 1.0, "attack_s": 0, "release_s": 0}])
    spec = np.abs(np.fft.rfft(render_recipe(r))) ** 2
    freqs = np.fft.rfftfreq(SFX_SAMPLE_RATE, 1.0 / SFX_SAMPLE_RATE)
    assert spec[(freqs >= 1800) & (freqs <= 4200)].sum() / spec.sum() > 0.95


def test_invalid_recipe_fails_loudly():
    bad = _recipe(layers=[{"deliberation": "x", "source": "sine", "duration_s": 1.0,
                           "freq_start_hz": 50000.0, "freq_end_hz": 100.0}])
    with pytest.raises(StudioAgentError, match="freq_start_hz"):
        render_recipe(bad)


def test_agent_generate_sfx_writes_valid_wav(tmp_path):
    payload = _recipe().model_dump()
    agent = VoiceActingAudioProducerAgent(client=_FakeClient(payload))
    out = tmp_path / "sfx" / "cue.wav"
    result = agent.generate_sfx(prompt="brief", output_path=out, kind="sfx")
    ok, err, dur = validate_wav_file(out)
    assert ok, err
    assert abs(dur - 1.0) < 0.01 and result.sample_rate == SFX_SAMPLE_RATE


def test_generate_sfx_without_client_raises(tmp_path):
    agent = VoiceActingAudioProducerAgent(backend=OfflineMockAudioBackend())
    with pytest.raises(StudioAgentError, match="requires a google.genai Client"):
        agent.generate_sfx(prompt="x", output_path=tmp_path / "x.wav")
