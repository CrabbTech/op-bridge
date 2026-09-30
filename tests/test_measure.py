"""measure_take, the log-frequency spectrogram and STOI on synthetic and spoken audio, without the device."""
import os
import shutil

import numpy as np
import pytest
import soundfile as sf

from op_bridge import analysis as an
from op_bridge import server as srv
from op_bridge import session
from op_bridge.session import Config, session_dir, write_json


@pytest.fixture
def home(tmp_path, monkeypatch):
    d = tmp_path / "home"; d.mkdir()
    monkeypatch.setenv("OP_BRIDGE_HOME", str(d))
    monkeypatch.setattr(session, "HOME", str(d))
    monkeypatch.setattr(srv, "HOME", str(d))
    return str(d)


def _vibrato_tone(seconds=4.0, sr=44100, hz=220.0, rate=5.5, depth_cents=30.0, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    inst = hz * 2 ** (depth_cents / 1200 * np.sin(2 * np.pi * rate * t))
    ph = 2 * np.pi * np.cumsum(inst) / sr
    x = sum((1.0 / k) * np.sin(k * ph) for k in range(1, 8))
    return (amp * x / np.abs(x).max()).astype(np.float32), sr


def _take(home, tid, x, sr, extra=None):
    d = session_dir(Config.load().current_session)
    wav = os.path.join(d, "takes", tid + ".wav")
    sf.write(wav, np.stack([x, x], axis=1), sr, subtype="FLOAT")
    write_json(os.path.join(d, "takes", tid + ".json"), {"id": tid, "score": {}, "analysis": {}, "wav": wav, "kind": "take", **(extra or {})})
    return wav


def test_measure_take_reads_pitch_bands_vibrato_and_draws_a_log_spectrogram(home):
    x, sr = _vibrato_tone()
    _take(home, "vib", x, sr)
    m = srv.measure_take("vib")
    assert m["pitch"]["note"] == "A3" and abs(m["pitch"]["f0_hz"] - 220) < 3 and m["pitch"]["voiced_fraction"] > 0.9
    assert 5.0 <= m["pitch"]["vibrato"]["rate_hz"] <= 6.0 and 40 <= m["pitch"]["vibrato"]["depth_cents"] <= 80
    b = m["bands_dbfs"]
    assert b["bass"] > b["mid"] > b["air"] and abs(b["bass"] - m["rms_db"]) < 3   # the fundamental carries the level
    assert m["harmonics_db_rel_h1"][0] == 0.0 and m["harmonics_db_rel_h1"][1] < 0
    assert m["seconds"] == 4.0 and os.path.exists(os.path.join(os.path.dirname(m["wav"]), "vib-log8000.png"))
    img = srv.view_spectrogram("vib", log_frequency=True)
    assert img is not None
    w = srv.measure_take("vib", start_s=1.0, end_s=2.0)
    assert w["window"] == [1.0, 2.0] and w["seconds"] == 1.0


def test_band_levels_sum_to_the_signal_level():
    sr = 44100; t = np.arange(2 * sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 100 * t) + 0.1 * np.sin(2 * np.pi * 3000 * t)).astype(np.float64)
    b = an.band_levels(x, sr)
    total = 10 * np.log10(sum(10 ** (v / 10) for v in b.values()))
    assert abs(total - an.rms_db(x)) < 0.5, (total, an.rms_db(x))
    assert abs(b["bass"] - 20 * np.log10(0.3 / np.sqrt(2))) < 1.0 and abs(b["presence"] - 20 * np.log10(0.1 / np.sqrt(2))) < 1.0


@pytest.mark.skipif(shutil.which("say") is None, reason="macOS say not available")
def test_stoi_of_speech_against_itself_and_against_noise(home, tmp_path):
    from op_bridge import speech as SP
    ref = str(tmp_path / "ref.wav")
    SP.synthesize("the quick brown fox jumps over the lazy dog, twice for good measure", ref, rate=170)
    x, sr = sf.read(ref, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    late = np.concatenate([np.zeros(int(0.25 * sr), np.float32), x * 0.5]).astype(np.float32)    # the same speech, quieter, 250 ms late
    _take(home, "spoken", late, sr, {"speech_wav": ref, "kind": "vocoder"})
    r = srv.speech_intelligibility("spoken")
    assert r["stoi"] > 0.95 and abs(r["lag_ms"] - 250) <= 20, r
    noise = (np.random.default_rng(1).standard_normal(len(x)) * 0.1).astype(np.float32)
    _take(home, "noise", noise, sr, {"speech_wav": ref})
    n = srv.speech_intelligibility("noise")
    assert n["stoi"] < 0.5 and n["stoi"] < r["stoi"] - 0.4, (n, r)   # STOI floors around 0.3-0.45 for unrelated signals
