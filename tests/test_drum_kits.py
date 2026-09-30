"""Drum kit files: the bridge's drum sampler specification against the kits the Field wrote.

Every test reads real files. The Field-written kits live in the disk backup; the tests that need them
skip when the backup is not on this machine. No test touches the device."""
import glob
import os
import struct

import numpy as np
import pytest

from op_bridge import presets as P

BACKUP = os.path.expanduser("~/Music/op-bridge/field-backup/2026-09-26-disk")
KITS = sorted(glob.glob(os.path.join(BACKUP, "drum/user/*.aif")))
needs_backup = pytest.mark.skipif(not KITS, reason="no disk backup of the Field on this machine")


def _pcm(preset: P.PresetFile) -> np.ndarray:
    ch = dict(preset.chunks)
    return np.frombuffer(ch[b"SSND"][8:], dtype="<i2").reshape(-1, 2).astype(np.float32) / 32768


def _frames(preset: P.PresetFile) -> int:
    return struct.unpack(">hIh", dict(preset.chunks)[b"COMM"][:8])[1]


def _sampler_kits():
    return [P.read_preset(p) for p in KITS if P.read_preset(p).engine == "drum"]


# --------------------------------------------------------------------------- the specification itself

def test_position_formulas_are_exact_and_invertible():
    assert P.DRUM_BANK_FRAMES == 882000 and P.DRUM_POSITION_SPAN == 2 ** 31
    assert abs(P.DRUM_POSITION_UNIT - 2434.79) < 0.01
    # a whole bank is the top of the int32 range; 1 s is 2^31 / 20
    assert P.drum_frame_to_position(882000) == 2 ** 31
    assert P.drum_frame_to_position(44100) == 107374182 == (2 ** 31) // 20
    for frame in (0, 1, 4410, 44084, 661531, 881999):
        v = P.drum_frame_to_position(frame)
        assert frame - 1 < P.drum_position_to_frame(v) <= frame
    assert P.drum_key_note(0) == 53 and P.drum_key_name(0) == "F3" and P.drum_key_name(23) == "E5"


@needs_backup
def test_field_kits_are_stereo_16_bit_44k_banks_under_20_seconds():
    seen = set()
    for path in KITS:
        k = P.read_preset(path)
        comm = dict(k.chunks)[b"COMM"]
        ch, frames, bits = struct.unpack(">hIh", comm[:8])
        assert k.kind == "drum" and bits == 16 and comm[18:22] == b"sowt" and len(comm) == 64
        assert k.meta["drum_version"] == 2 and len(k.meta["dyna_env"]) == 8
        seen.add(k.engine)
        if k.engine == "drum":
            assert ch == 2 and frames <= P.DRUM_BANK_FRAMES
            for arr in P.DRUM_KEY_ARRAYS:
                assert len(k.meta[arr]) == 24, arr
            assert k.meta["dyna_env"] == P.DRUM_ENVELOPE_DEFAULT
        else:
            assert k.engine == "dbox" and ch == 1
            assert "start" not in k.meta and len(k.meta["dbox_data"]) == 24 and all(len(r) == 8 for r in k.meta["dbox_data"])
        assert P.check_drum_kit(k) == [], path
    assert seen == {"drum", "dbox"}


@needs_backup
def test_positions_are_floor_of_frame_times_unit_in_a_20_second_bank():
    off_grid = []
    for k in _sampler_kits():
        m, frames = k.meta, _frames(k)
        for name in ("start", "end"):
            for i, v in enumerate(m[name]):
                if P.drum_frame_to_position(round(P.drum_position_to_frame(v))) != v:
                    off_grid.append((m["name"], name, i, v))
        # the last end sits 18 frames before the end of the bank, the file's own length under the 20 s unit
        assert 17 < frames - P.drum_position_to_frame(max(m["end"])) < 19, m["name"]
        # regions are separated by exactly 16 frames
        gaps = [P.drum_position_to_frame(m["start"][i + 1]) - P.drum_position_to_frame(m["end"][i]) for i in range(23)]
        assert sum(1 for g in gaps if abs(g - 16) < 0.01) >= 22, (m["name"], gaps)
        assert all(0 <= s <= e < 2 ** 31 for s, e in zip(m["start"], m["end"]))
    # one factory in point was nudged by hand on the device and does not sit on a frame; everything else does
    assert len(off_grid) <= 1 and all(o[0] == "aeroplane" for o in off_grid), off_grid


@needs_backup
def test_region_boundaries_fall_in_silence_only_with_the_20_second_unit():
    for k in _sampler_kits():
        pcm, m, frames = _pcm(k), k.meta, _frames(k)
        for unit, expect_silent in ((P.DRUM_POSITION_UNIT, True), ((2 ** 31) / frames, False)):
            gap_rms, onset_rms = [], []
            for i in range(23):
                a, b = int(m["end"][i] / unit), int(m["start"][i + 1] / unit)
                gap_rms.append(float(np.sqrt(np.mean(pcm[a:b] ** 2))) if b > a else 0.0)
                onset_rms.append(float(np.sqrt(np.mean(pcm[b:b + 256] ** 2))))
            if expect_silent:
                assert np.mean(gap_rms) < 0.001 and np.mean(onset_rms) > 0.1, m["name"]
            else:
                assert np.mean(gap_rms) > 0.002, m["name"]


@needs_backup
def test_discrete_values_match_the_spec():
    modes, reverse, pans_plain, pans_ab, vols = set(), set(), set(), set(), []
    for k in _sampler_kits():
        m = k.meta
        modes |= set(m["playmode"]); reverse |= set(m["reverse"]); vols += m["volume"]
        (pans_ab if all(m["pan_ab"]) else pans_plain).update(m["pan"])
        assert all(a == 0 for a in m["attack"]) and all(f == 0 for f in m["fademode"])
        assert set(m["pan_ab"]) <= {True, False}
        # the hi-hat keys C#4 and D#4 carry the 20480 mode in every factory kit
        assert m["playmode"][8] == m["playmode"][10] == P.DRUM_PLAYMODE_HIHAT, m["name"]
        assert m["playmode"].count(P.DRUM_PLAYMODE_DEFAULT) >= 16, m["name"]
    assert modes == set(P.DRUM_PLAYMODE_VALUES)
    assert all(v % 8192 == 4096 for v in modes), "play modes are bin centres of an 8192-wide selector"
    assert P.DRUM_DIRECTION_FORWARD in reverse and max(reverse) < 16384
    assert P.DRUM_PAN_CENTRE in pans_plain and all(p % 1024 == 0 for p in pans_plain) and min(pans_plain) >= 12288 and max(pans_plain) <= 20480
    assert {0, P.DRUM_PAN_MAX} <= pans_ab
    assert P.DRUM_GAIN_UNITY == max(set(vols), key=vols.count) and min(vols) >= 3564 and max(vols) <= 15077


@needs_backup
def test_stacked_ab_kit_holds_two_sounds_per_key():
    """pan_ab true: left and right are independent mono sounds (A and B); false: a stereo image of one sound."""
    for k in _sampler_kits():
        pcm, m = _pcm(k), k.meta
        corr = []
        for i in range(24):
            a, b = int(m["start"][i] / P.DRUM_POSITION_UNIT), int(m["end"][i] / P.DRUM_POSITION_UNIT)
            l, r = pcm[a:b, 0], pcm[a:b, 1]
            corr.append(float(np.corrcoef(l, r)[0, 1]))
        if all(m["pan_ab"]):
            assert max(abs(c) for c in corr) < 0.5, m["name"]
        else:
            assert np.mean(corr) > 0.5, m["name"]


@needs_backup
def test_dbox_data_layout():
    k = next(P.read_preset(p) for p in KITS if P.read_preset(p).engine == "dbox")
    d = k.meta["dbox_data"]
    cols = list(zip(*d))
    assert min(cols[0]) < 0 < max(cols[0]), "column 0 is bipolar: oscillator pitch"
    # columns 1 (waveform), 2 and 6 (the two envelopes) sit on 1 % steps of the encoder (328 = 32767 / 100):
    # 24, 23 and 20 of 24 values; the other columns are continuous
    def on_grid(c):
        return sum(1 for v in cols[c] if v == 0 or abs(v / 328 - round(v / 328)) < 0.02)
    assert on_grid(1) == 24 and cols[1].count(0) >= 12
    assert on_grid(2) >= 22 and on_grid(6) >= 20 and max(cols[2]) < 16384 and max(cols[6]) < 16384
    assert all(on_grid(c) <= 6 for c in (0, 3, 4, 5, 7))
    # the two hi-hat keys are the same voice with a different envelope
    assert d[8][:2] == d[10][:2] and d[8][3:] == d[10][3:] and d[8][2] != d[10][2]
    assert len(P.DBOX_PARAMS) == 8 and k.meta["dyna_env"][:4] == [25599, 6656, 11263, 2816]
