"""Sound profiles, classification and ranking on real synthesized audio and real Field preset files."""
import os

import numpy as np
import pytest

from op_bridge import sounds as SO, judge as JU

BACKUP = os.path.expanduser("~/Music/op-bridge/field-backup/2026-09-26-disk/synth/user")
SR = 44100


def _note(freq, seconds, attack_s=0.005, decay_to=1.0, bright=1.0, noise=0.0):
    t = np.arange(int(seconds * SR)) / SR
    x = np.sin(2 * np.pi * freq * t) + bright * 0.5 * np.sin(2 * np.pi * 2 * freq * t) + bright * 0.25 * np.sin(2 * np.pi * 3 * freq * t)
    env = np.minimum(1.0, t / attack_s) * (decay_to + (1 - decay_to) * np.exp(-t * 4))
    x = 0.3 * x * env + noise * np.random.default_rng(1).normal(0, 0.3, len(t)) * env
    return x.astype(np.float32)


def _take(x, tail=1.0):
    pre = np.zeros(int(0.3 * SR), np.float32)
    post = np.zeros(int(tail * SR), np.float32)
    mono = np.concatenate([pre, x, post])
    return np.stack([mono, mono], axis=1), len(pre), len(pre) + len(x)


def test_sustained_dark_note_tags_as_bass_or_pad():
    audio, on, off = _take(_note(55.0, 1.2, bright=0.3))
    f = SO.measure(audio, SR, on, off, 33)
    tags = SO.tags_for(f, 33)
    assert "sustained" in tags["tags"] and ("dark" in tags["tags"] or "sub-heavy" in tags["tags"])
    assert "bass" in tags["roles"]
    assert f.harmonicity_db > 8 and abs(f.pitch_error_cents or 0) < 20


def test_plucky_bright_note_tags_as_keys_or_pluck():
    audio, on, off = _take(_note(440.0, 1.2, attack_s=0.003, decay_to=0.05, bright=1.5))
    f = SO.measure(audio, SR, on, off, 69)
    tags = SO.tags_for(f, 69)
    assert f.attack_ms is not None and f.attack_ms < 25 and f.sustain_ratio_db < -6
    assert "keys" in tags["roles"] or "pluck" in tags["roles"] or "percussive" in tags["roles"]


def test_noise_tags_as_fx():
    audio, on, off = _take(_note(200.0, 1.0, noise=1.0) * 0.2)
    f = SO.measure(audio, SR, on, off, 55)
    assert "noisy" in SO.tags_for(f, 55)["tags"]


def test_local_ranking_prefers_matching_tags():
    cands = [{"id": "a", "tags": ["dark", "sustained", "harmonic"], "roles": ["bass", "pad"], "engine": "dsynth", "name": "back bass"},
             {"id": "b", "tags": ["very bright", "short", "harmonic"], "roles": ["pluck"], "engine": "cluster", "name": "sparkle"}]
    r = JU.rank(cands, "dark sustained bass", top=2)
    assert r["judge"] in ("local", "jev") and r["ranked"][0][1]["id"] == "a"


@pytest.mark.skipif(not os.path.isdir(BACKUP), reason="no disk backup of the Field on this machine")
def test_profiles_from_field_files_and_range_guard():
    p3 = SO.profile_from_file(os.path.join(BACKUP, "3.aif"), "synth", 3)   # suitcase, a sampler rooted at C5
    assert p3.engine == "sampler" and p3.root_midi == 72 and p3.playable == (60, 84)
    shift, notes = SO.transpose_into_range([48, 52, 55, 59], p3)
    assert shift == 12 and notes
    p2 = SO.profile_from_file(os.path.join(BACKUP, "2.aif"), "synth", 2)   # a synthesis engine
    assert p2.root_midi is None and p2.playable == (SO.AUDIBLE_LOW, SO.AUDIBLE_HIGH)
    assert SO.transpose_into_range([48, 52, 55], p2)[0] == 0
    assert SO.notes_out_of_range([12, 60, 110], p2) == [12, 110]


def _saw(freq, seconds):
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * freq * h * t) / h for h in range(1, 13))
    return (0.2 * x).astype(np.float32)


def _square(freq, seconds):
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * freq * h * t) / h for h in range(1, 13, 2))
    return (0.2 * x).astype(np.float32)


def test_hollow_versus_rich_and_pulsing_are_described():
    audio, on, off = _take(_square(220.0, 1.2))
    d = SO.describe(SO.measure_deep(audio, SR, on, off, 57))
    assert "hollow" in d, d
    audio, on, off = _take(_saw(220.0, 1.2))
    deep = SO.measure_deep(audio, SR, on, off, 57)
    assert deep["harmonics_above_minus40db"] >= 8 and "rich" in SO.describe(deep), deep
    t = np.arange(int(1.2 * SR)) / SR
    trem = (_saw(220.0, 1.2) * (0.6 + 0.4 * np.sin(2 * np.pi * 5.0 * t))).astype(np.float32)
    audio, on, off = _take(trem)
    deep = SO.measure_deep(audio, SR, on, off, 57)
    assert deep["level_mod_hz"] and 4.0 <= deep["level_mod_hz"] <= 6.0, deep
    assert any(s.startswith("pulsing") for s in SO.describe(deep))


def test_bell_like_inharmonic_partials_are_described():
    t = np.arange(int(1.2 * SR)) / SR
    bell = (0.2 * (np.sin(2 * np.pi * 440 * t) + 0.7 * np.sin(2 * np.pi * 440 * 2.76 * t) + 0.5 * np.sin(2 * np.pi * 440 * 5.4 * t))).astype(np.float32)
    audio, on, off = _take(bell)
    deep = SO.measure_deep(audio, SR, on, off, 69)
    d = SO.describe(deep)
    assert deep["inharmonic_share"] > 0.18 and ("bell-like" in d or "metallic" in d), (deep["inharmonic_share"], d)


def test_request_implication_and_composite_weights():
    assert JU.implied("a dark sustained bass") == {"role": "bass", "character": "dark", "envelope": "sustained"}
    tax = {"role": {"probabilities": {"bass": 0.9, "pad": 0.1}}, "character": {"probabilities": {"dark": 0.8}}, "envelope": {"probabilities": {"sustained": 0.7}}}
    assert abs(JU.composite(1.0, tax, JU.implied("a dark sustained bass")) - (0.55 * 1 + 0.2 * 0.9 + 0.15 * 0.8 + 0.1 * 0.7)) < 1e-9
    assert JU.composite(0.5, None, {}) == 0.5
