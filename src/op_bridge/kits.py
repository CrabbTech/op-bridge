"""Drum kit maps: which of a Field drum kit's 24 keys is the kick, the snare, the hats, and so on.

Everything here is file work; nothing opens a MIDI or audio device. Two sources of evidence feed a KitMap:

  kit_map_from_file(path)   the kit file the Field wrote (drum/user/<n>.aif): the 24 regions are decoded with
                            op_bridge.presets.drum_kit_regions (start/end in 2^31 = 20 s units), each key's audio is
                            measured and classified, and the per-key settings (playmode, pan, pan_ab, volume) are attached
  kit_map_from_clips(dir)   live auditions saved as <midi>_<name>.wav, one hit per file

and the owner's labels by ear always win over the classifier when maps are merged (merge_kit_maps, save_kit_map).

Key naming: region index 0 is MIDI 53. The Field's screen calls that key F2, one octave below the MIDI convention that
calls 53 F3; field_key_name() gives the Field's name and midi_key_name() the MIDI one. Everything is keyed by MIDI number.

The feature rules and the factory layout come from the kit study of 2026-09-26 (~/Music/op-bridge/probe/kit-study): the
seven Field-written sampler kits, the live capture of kit 1 and the owner's nine labels. Labels: kick, snare, clap,
closed hat, open hat, tom, cymbal, perc (nothing named fits) and silent; "unsure" when the audio does not decide.
"""
from __future__ import annotations

import copy
import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from op_bridge import presets as P

SR = 44100
KIT_FIRST_MIDI = P.DRUM_FIRST_NOTE          # 53
KIT_KEYS = P.DRUM_KEYS                      # 24
KIT_LAST_MIDI = KIT_FIRST_MIDI + KIT_KEYS - 1   # 76
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

LABELS = ("kick", "snare", "clap", "closed hat", "open hat", "tom", "cymbal", "perc", "silent")
NAMED_CLASSES = ("kick", "snare", "clap", "closed hat", "open hat", "tom", "cymbal")
UNSURE = "unsure"
ROLES = ("BD", "SN", "CH", "OH", "CL", "TOM_LOW", "TOM_MID", "TOM_HIGH", "CY", "PERC")
EXTRA_ROLES = ("BD2", "SN2", "CH_CHOKE")    # written when they exist, as in the owner's map of kit 1

CONFIDENCE_HIGH = 0.7        # the label is trusted on its own
CONFIDENCE_LABEL = 0.5       # below this the label is "unsure"
PERC_BELOW = 0.35            # best named score below this: "perc"
SILENT_PEAK = 10 ** (-70 / 20)   # peak under -70 dBFS: nothing there

PLAYMODE_CHOKE = P.DRUM_PLAYMODE_HIHAT       # 20480: on C#3 and D#3 of every factory kit
PLAYMODE_NOTES = {
    P.DRUM_PLAYMODE_HIHAT: "playmode 20480, the hi-hat/choke marker TE puts on C#3 and D#3 of every kit",
    P.DRUM_PLAYMODE_DEFAULT: "playmode 12288, the factory default",
    4096: "playmode 4096, seen on one factory key only (apes are us C3), meaning unverified",
}
BAND_NAMES = ("sub", "low", "mid", "himid", "high")
BANDS = [("sub", 0.0, 120.0), ("low", 120.0, 400.0), ("mid", 400.0, 2000.0), ("himid", 2000.0, 6000.0), ("high", 6000.0, None)]
HP_DECAY_HZ = 2000.0
BURST_HP_HZ = 1000.0

# Level law fitted on kit 1 (24 keys, r = 0.97, residual sd 0.44 dB): the live peak of a key on the day's main volume
# was file_peak + 2.09 * 20 log10(volume / 8192) - 7.7 dB, i.e. the key `volume` behaves like 40 log10(v / 8192) dB.
LEVEL_LAW_SLOPE = 2.09
LEVEL_LAW_OFFSET_DB = -7.7

# The factory layout: what TE put on each key across the six distinct Field-written kits (kit study, position tally).
# `class` is the expected class, `prior` whether it is strong enough to break ties, `role` the pattern role it serves.
FACTORY_LAYOUT: dict[int, dict[str, Any]] = {
    53: {"class": "kick", "role": "BD", "prior": True, "kits": "6/6", "note": "main kick"},
    54: {"class": "kick", "role": "BD2", "prior": True, "kits": "6/6", "note": "second kick, usually longer or 808-style"},
    55: {"class": "snare", "role": "SN", "prior": True, "kits": "4/6 strict, 6/6 snare family", "note": "main snare"},
    56: {"class": "snare", "role": "SN2", "prior": True, "kits": "5/6", "note": "second snare: snares off, rim or noisier"},
    58: {"class": "clap", "role": "CL", "prior": False, "kits": "2/6 clear, 2 clap-like", "note": "clap or clap-like hit, weak convention"},
    60: {"class": "closed hat", "role": "CH", "prior": True, "kits": "3/6", "note": "closed hat, weak: toms and tonal hits sit here too"},
    61: {"class": "closed hat", "role": "CH_CHOKE", "prior": True, "kits": "5/6 by audio, playmode 20480 in 6/6",
         "note": "closed hat in the choke group", "choke": "chokes the open hat on D#3 (63)"},
    62: {"class": "closed hat", "role": "CH", "prior": True, "kits": "5/6", "note": "closed hat"},
    63: {"class": "open hat", "role": "OH", "prior": True, "kits": "3/6 genuine open hats, playmode 20480 in 6/6",
         "note": "open hat in the choke group, the hat that rings longer than C#3", "choke": "choked by the closed hat on C#3 (61)"},
    64: {"class": "closed hat", "role": "CH", "prior": True, "kits": "2/6 strict, 4/6 hat family", "note": "closed hat or other hat-family hit, weak"},
    68: {"class": "cymbal", "role": "CY", "prior": True, "kits": "4/6", "note": "crash or ride cymbal, the longest region of the kit"},
}


# --------------------------------------------------------------------------- key names

def field_key_name(midi: int) -> str:
    """The name the Field shows for a drum key: one octave below the MIDI convention (53 -> F2, 63 -> D#3, 76 -> E4)."""
    midi = int(midi)
    return NOTE_NAMES[midi % 12] + str(midi // 12 - 2)


def midi_key_name(midi: int) -> str:
    """The MIDI-convention name of a note (53 -> F3), the one the live clip files use."""
    midi = int(midi)
    return NOTE_NAMES[midi % 12] + str(midi // 12 - 1)


def field_name_to_midi(name: str) -> int:
    """Field key name -> MIDI note ("F2" -> 53, "D#3" -> 63). Raises ValueError for anything outside the 24 kit keys."""
    m = re.fullmatch(r"\s*([A-Ga-g])([#b]?)(-?\d)\s*", str(name))
    if not m:
        raise ValueError(f"{name!r} is not a key name like F2 or D#3 (the Field's names for the drum keys run F2 .. E4)")
    pc = NOTE_NAMES.index(m.group(1).upper()) + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)
    midi = (int(m.group(3)) + 2) * 12 + pc
    if not KIT_FIRST_MIDI <= midi <= KIT_LAST_MIDI:
        raise ValueError(f"{name} is MIDI {midi}, outside the kit keys F2 .. E4 (MIDI {KIT_FIRST_MIDI} .. {KIT_LAST_MIDI})")
    return midi


def key_index(midi: int) -> int:
    """Region index 0 .. 23 of a MIDI note; ValueError outside 53 .. 76."""
    midi = int(midi)
    if not KIT_FIRST_MIDI <= midi <= KIT_LAST_MIDI:
        raise ValueError(f"MIDI {midi} is not a drum kit key: kits hold 24 keys, MIDI {KIT_FIRST_MIDI} ({field_key_name(KIT_FIRST_MIDI)} on the Field) "
                         f"to {KIT_LAST_MIDI} ({field_key_name(KIT_LAST_MIDI)})")
    return midi - KIT_FIRST_MIDI


def predict_live_peak_db(file_peak_db: float, volume: int) -> float:
    """Expected played peak (dBFS on the USB pair) of a key from its file-region peak and its `volume` value, by the level
    law fitted on kit 1 at one main-volume setting: file_peak + 2.09 * 20 log10(volume / 8192) - 7.7 dB (sd 0.44 dB)."""
    v = max(int(volume), 1)
    return float(file_peak_db + LEVEL_LAW_SLOPE * 20.0 * np.log10(v / P.DRUM_GAIN_UNITY) + LEVEL_LAW_OFFSET_DB)


# --------------------------------------------------------------------------- features

def _db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def _mono(audio: Any) -> np.ndarray:
    a = np.asarray(audio)
    if a.dtype.kind in "iu":
        a = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    a = a.astype(np.float32)
    if a.ndim == 0:
        return np.zeros(0, dtype=np.float32)
    if a.ndim == 2:
        if a.shape[1] > a.shape[0] and a.shape[0] <= 8:     # (channels, frames)
            a = a.T
        a = a.mean(axis=1)
    elif a.ndim > 2:
        raise ValueError(f"audio must be 1-D mono or 2-D (frames, channels), got shape {a.shape}")
    return np.nan_to_num(a.astype(np.float32))


def _envelope_db(x: np.ndarray, sr: int, win_ms: float = 5.0, hop_ms: float = 1.0) -> np.ndarray:
    """RMS envelope in dBFS on a 1 ms grid (5 ms windows)."""
    win = max(int(sr * win_ms / 1000.0), 1)
    hop = max(int(sr * hop_ms / 1000.0), 1)
    if x.shape[0] < win:
        x = np.pad(x, (0, win - x.shape[0]))
    n = 1 + (x.shape[0] - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx].astype(np.float64) ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-9))


def _find_onset(env: np.ndarray, gate_db: float = 20.0) -> int:
    """Index (1 ms grid) of the onset: the last frame before the envelope peak that sits more than gate_db below it,
    minus 2 ms. Robust to a previous hit's tail leaking into the start of a clip."""
    pk = int(np.argmax(env))
    below = np.nonzero(env[:pk + 1] < env[pk] - gate_db)[0]
    start = int(below[-1]) if below.size else 0
    return max(start - 2, 0)


def _highpass(x: np.ndarray, sr: int, hz: float) -> np.ndarray:
    if x.shape[0] < 32 or hz >= sr / 2:
        return x.astype(np.float64)
    sos = signal.butter(4, hz / (sr / 2), btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, x.astype(np.float64))


def _decay_ms(env: np.ndarray, drop: float, hard_end: bool) -> tuple[float | None, bool]:
    """(ms from the envelope peak to the last frame above peak - drop dB, truncated). A hit still above the threshold at
    the end of a cut clip (hard_end) reports None and truncated = True."""
    pk = int(np.argmax(env))
    above = np.nonzero(env > env[pk] - drop)[0]
    last = int(above[-1])
    truncated = bool(hard_end and last >= env.shape[0] - 3)
    return (None if truncated else float(last - pk)), truncated


def _band_ratios(x: np.ndarray, sr: int) -> dict[str, float]:
    if x.shape[0] < 32 or not np.any(x):
        return {name: 0.0 for name in BAND_NAMES}
    spec = np.abs(np.fft.rfft(x.astype(np.float64) * np.hanning(x.shape[0]))) ** 2
    f = np.fft.rfftfreq(x.shape[0], 1.0 / sr)
    total = float(spec.sum()) or 1e-20
    out = {}
    for name, lo, hi in BANDS:
        sel = (f >= lo) if hi is None else ((f >= lo) & (f < hi))
        out[name] = float(spec[sel].sum() / total)
    return out


def _centroid_hz(x: np.ndarray, sr: int) -> float:
    if x.shape[0] < 32 or not np.any(x):
        return 0.0
    spec = np.abs(np.fft.rfft(x.astype(np.float64) * np.hanning(x.shape[0]))) ** 2
    f = np.fft.rfftfreq(x.shape[0], 1.0 / sr)
    return float((f * spec).sum() / (spec.sum() or 1e-20))


def _flatness(x: np.ndarray, sr: int, lo: float = 100.0, hi: float = 16000.0) -> float:
    """Spectral flatness (geometric / arithmetic mean of the Welch PSD between lo and hi Hz): 0 = one tone, 1 = white noise."""
    if x.shape[0] < 64 or not np.any(x):
        return 0.0
    nper = min(2048, max(256, x.shape[0]))
    f, psd = signal.welch(x.astype(np.float64), fs=sr, nperseg=nper)
    sel = (f >= lo) & (f <= min(hi, sr / 2))
    p = psd[sel] + 1e-20
    if p.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def _zero_crossing_rate(x: np.ndarray, sr: int) -> float:
    if x.shape[0] < 2:
        return 0.0
    return float(np.count_nonzero(np.diff(np.signbit(x))) * sr / x.shape[0])


def _nccf_pitch(x: np.ndarray, sr: int, skip_ms: float = 5.0, frame_ms: float = 100.0, f_lo: float = 30.0, f_hi: float = 1000.0,
                dip: float = 0.3) -> tuple[float, float]:
    """Pitchedness of the body of a hit: the peak of the normalised cross-correlation (RAPT style) over lags between
    1/f_hi and 1/f_lo after the main lobe (the first lag whose NCCF drops below `dip`). Returns (peak 0..1, f0 Hz);
    (0, 0) when nothing periodic is in range. Toms, 808 kicks and cowbells sit near 0.7 or above; hats and claps below 0.35."""
    a = int(sr * skip_ms / 1000.0)
    W = int(sr * frame_ms / 1000.0)
    L = int(sr / f_lo)
    lo = max(int(sr / f_hi), 2)
    y = x[a:a + W + L].astype(np.float64)
    if y.shape[0] < W // 2 or not np.any(y):
        return 0.0, 0.0
    y = np.pad(y, (0, W + L - y.shape[0]))
    w = y[:W]
    e0 = float(np.dot(w, w))
    if e0 <= 0:
        return 0.0, 0.0
    num = signal.correlate(y, w, mode="valid")
    sq = np.concatenate([[0.0], np.cumsum(y * y)])
    ks = np.arange(L + 1)
    ek = sq[ks + W] - sq[ks]
    nccf = num / np.sqrt(e0 * np.maximum(ek, 1e-20))
    below = np.nonzero(nccf[1:] < dip)[0]
    if below.size == 0:
        return 0.0, 0.0
    start = max(int(below[0]) + 1, lo)
    if start >= L:
        return 0.0, 0.0
    k = start + int(np.argmax(nccf[start:L + 1]))
    return float(np.clip(nccf[k], 0.0, 1.0)), float(sr / k)


def _burst_count(x: np.ndarray, sr: int, span_ms: int = 50, prominence_db: float = 4.0) -> int:
    """Separate bursts in the first span_ms of the > 1 kHz part of the hit (2 ms windows, peaks at least 5 ms apart,
    prominence_db proud, within 20 dB of the loudest). A clap's flam gives 2 to 4; single hits give 1."""
    hp = _highpass(x[:int(sr * (span_ms + 10) / 1000.0)], sr, BURST_HP_HZ)
    if hp.shape[0] < 64 or not np.any(hp):
        return 1
    env = _envelope_db(hp.astype(np.float32), sr, win_ms=2.0, hop_ms=1.0)
    seg = env[:span_ms]
    if seg.size < 3:
        return 1
    peaks, _ = signal.find_peaks(seg, prominence=prominence_db, distance=5, height=seg.max() - 20.0)
    return max(int(peaks.size), 1)


def _silent_features(x: np.ndarray, sr: int) -> dict[str, Any]:
    peak = float(np.abs(x).max()) if x.shape[0] else 0.0
    return {"silent": True, "length_ms": round(x.shape[0] * 1000.0 / sr, 1), "clip_ms": round(x.shape[0] * 1000.0 / sr, 1),
            "peak_db": round(_db(peak), 1), "rms_db": -120.0, "hp_peak_db": -120.0,
            "decay_20_ms": None, "decay_40_ms": None, "hp_decay_20_ms": None, "hp_decay_40_ms": None,
            "decay_20_truncated": False, "decay_40_truncated": False, "hp_decay_20_truncated": False, "hp_decay_40_truncated": False,
            "truncated": False, "abrupt_end": False, "attack_ms": 0.0, "attack_db_per_ms": 0.0, "crest_db": 0.0, "bursts": 0,
            "centroid_hz": 0.0, "centroid_30ms_hz": 0.0, "centroid_100ms_hz": 0.0,
            "bands": {k: 0.0 for k in BAND_NAMES}, "bands_30ms": {k: 0.0 for k in BAND_NAMES}, "bands_100ms": {k: 0.0 for k in BAND_NAMES},
            "flatness": 0.0, "flatness_100ms": 0.0, "zcr_hz": 0.0, "pitchedness": 0.0, "f0_hz": 0.0, "pre_onset_db": None}


def key_features(audio: np.ndarray, sr: int = SR, clip: bool = False) -> dict[str, Any]:
    """Numeric features of one drum hit (mono or stereo samples, any float or int dtype), trimmed to its onset.

    clip=True marks audio that was cut (a live capture): a decay still above its threshold at the end is reported as
    None with the matching *_truncated flag instead of a value, and `truncated` summarises them.

    Keys: silent, length_ms (after onset trimming), clip_ms, peak_db, rms_db, hp_peak_db (> 2 kHz part), decay_20_ms /
    decay_40_ms (ms from the envelope peak to the last frame above peak - 20 / 40 dB), hp_decay_20_ms / hp_decay_40_ms
    (the same on the > 2 kHz part), abrupt_end (cut while still loud), attack_ms (-20 dB crossing to the peak),
    attack_db_per_ms, crest_db, bursts (> 1 kHz bursts in the first 50 ms), centroid_hz (whole hit), centroid_30ms_hz,
    centroid_100ms_hz, bands / bands_30ms / bands_100ms (energy fractions: sub < 120 Hz, low 120-400, mid 400-2000,
    himid 2000-6000, high > 6000), flatness / flatness_100ms (0 = one tone, 1 = white noise), zcr_hz (first 100 ms),
    pitchedness (normalised cross-correlation peak, 1 = periodic) and f0_hz, pre_onset_db (level of the 50 ms before the
    onset relative to the peak; leakage of a previous hit). Short (< 10 ms) or silent (< -70 dBFS) audio gives
    silent = True with neutral values, never an error."""
    if int(sr) <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")
    sr = int(sr)
    x = _mono(audio)
    if x.shape[0] < sr // 100 or not np.any(x) or float(np.abs(x).max()) < SILENT_PEAK:
        return _silent_features(x, sr)
    clip_ms = round(x.shape[0] * 1000.0 / sr, 1)
    env = _envelope_db(x, sr)
    onset = _find_onset(env)
    pre = x[max(int((onset - 50) * sr / 1000.0), 0):int(onset * sr / 1000.0)]
    x = x[int(onset * sr / 1000.0):]
    env = _envelope_db(x, sr)
    pk = int(np.argmax(env))
    peak_env = float(env[pk])
    out: dict[str, Any] = {"silent": False, "length_ms": round(x.shape[0] * 1000.0 / sr, 1), "clip_ms": clip_ms}
    out["peak_db"] = round(_db(np.abs(x).max()), 1)
    out["rms_db"] = round(_db(np.sqrt(np.mean(x.astype(np.float64) ** 2))), 1)
    out["pre_onset_db"] = round(_db(np.sqrt(np.mean(pre.astype(np.float64) ** 2))) - out["peak_db"], 1) if pre.shape[0] > 64 else None
    for drop in (20, 40):
        ms, trunc = _decay_ms(env, drop, clip)
        out[f"decay_{drop}_ms"] = ms
        out[f"decay_{drop}_truncated"] = trunc
    d20, d40 = out["decay_20_ms"], out["decay_40_ms"]
    out["truncated"] = bool(out["decay_20_truncated"] or out["decay_40_truncated"])
    out["abrupt_end"] = bool(d20 is not None and d40 is not None and d40 >= 100 and (d40 - d20) < 15)
    hp = _highpass(x, sr, HP_DECAY_HZ)
    if np.any(hp) and np.abs(hp).max() > 1e-4:
        henv = _envelope_db(hp.astype(np.float32), sr)
        for drop in (20, 40):
            ms, trunc = _decay_ms(henv, drop, clip)
            out[f"hp_decay_{drop}_ms"] = ms
            out[f"hp_decay_{drop}_truncated"] = trunc
        out["hp_peak_db"] = round(_db(np.abs(hp).max()), 1)
    else:
        out["hp_decay_20_ms"] = out["hp_decay_40_ms"] = 0.0
        out["hp_decay_20_truncated"] = out["hp_decay_40_truncated"] = False
        out["hp_peak_db"] = -120.0
    rise = np.nonzero(env[:pk + 1] > peak_env - 20.0)[0]
    out["attack_ms"] = round(float(pk - int(rise[0])), 1) if rise.size else 0.0
    slope = np.diff(env[:pk + 1]) if pk > 0 else np.array([0.0])
    out["attack_db_per_ms"] = round(float(slope.max()), 1) if slope.size else 0.0
    head = x[:int(0.05 * sr)]
    out["crest_db"] = round(_db(np.abs(head).max()) - _db(np.sqrt(np.mean(head.astype(np.float64) ** 2))), 1)
    out["bursts"] = _burst_count(x, sr)
    first = x[:int(0.1 * sr)]
    trans = x[:int(0.03 * sr)]
    out["centroid_30ms_hz"] = round(_centroid_hz(trans, sr))
    out["centroid_100ms_hz"] = round(_centroid_hz(first, sr))
    out["centroid_hz"] = round(_centroid_hz(x, sr))
    out["bands"] = {k: round(v, 3) for k, v in _band_ratios(x, sr).items()}
    out["bands_100ms"] = {k: round(v, 3) for k, v in _band_ratios(first, sr).items()}
    out["bands_30ms"] = {k: round(v, 3) for k, v in _band_ratios(trans, sr).items()}
    out["flatness"] = round(_flatness(x, sr), 3)
    out["flatness_100ms"] = round(_flatness(first, sr), 3)
    out["zcr_hz"] = round(_zero_crossing_rate(first, sr))
    p, f0 = _nccf_pitch(x, sr)
    out["pitchedness"] = round(p, 3)
    out["f0_hz"] = round(f0, 1)
    return out


# --------------------------------------------------------------------------- rules

def ramp(v: float | None, lo: float, hi: float) -> float:
    """0 at v <= lo, 1 at v >= hi, linear between (0 for a missing value)."""
    if v is None:
        return 0.0
    if hi <= lo:
        return 1.0 if v >= hi else 0.0
    return float(min(max((float(v) - lo) / (hi - lo), 0.0), 1.0))


def fall(v: float | None, lo: float, hi: float) -> float:
    """1 at v <= lo, 0 at v >= hi."""
    return 1.0 - ramp(v, lo, hi)


def _decay(f: dict[str, Any], key: str) -> float:
    """A decay value; a truncated clip's decay counts as the hit's length (it is at least that)."""
    v = f.get(key)
    if v is None:
        return float(f.get("length_ms") or 600.0)
    return float(v)


def class_scores(f: dict[str, Any]) -> dict[str, float]:
    """Soft membership (0..1) of one hit in each named class plus "perc" = 1 - max(named). Each class is the minimum of a
    few ramps, so a score reads as the weakest of its conditions. Thresholds were set on the Field's factory kits and the
    owner's labels of kit 1 (kit study, 2026-09-26)."""
    if f.get("silent"):
        return {**{k: 0.0 for k in NAMED_CLASSES}, "perc": 0.0}
    b, b30 = f["bands"], f["bands_30ms"]
    sub, low, mid, himid, high = (float(b[k]) for k in BAND_NAMES)
    hh30 = float(b30["himid"]) + float(b30["high"])          # brightness of the transient
    body30 = float(b30["low"]) + float(b30["mid"])            # drum body in the transient (120 Hz .. 2 kHz)
    c100, cf = float(f["centroid_100ms_hz"]), float(f["centroid_hz"])
    c30 = float(f.get("centroid_30ms_hz", c100))
    cmax = max(c100, cf)
    d20, d40 = _decay(f, "decay_20_ms"), _decay(f, "decay_40_ms")
    hd20, hd40 = _decay(f, "hp_decay_20_ms"), _decay(f, "hp_decay_40_ms")
    truncated = bool(f.get("decay_40_truncated"))
    abrupt = bool(f.get("abrupt_end"))
    flat = float(f["flatness"])
    pitch, f0 = float(f["pitchedness"]), float(f["f0_hz"])
    rise = float(f.get("attack_ms", 0.0))
    # bursts only mean something when the > 1 kHz part is a real component of the hit
    bursts = int(f.get("bursts", 1)) if float(f.get("hp_peak_db", -120.0)) >= float(f["peak_db"]) - 20.0 else 1
    tonal_ring = ramp(pitch, 0.5, 0.7) * ramp(d20, 150, 250)      # a pitched body that keeps ringing: tom, not snare
    tom_pitch = ramp(pitch, 0.6, 0.8) * ramp(f0, 65, 90)          # a clearly pitched body above ~80 Hz: tom, not kick
    kick = min(fall(c100, 250, 400), ramp(sub + low, 0.70, 0.85), ramp(sub, 0.15, 0.30), 1.0 - tom_pitch,
               ramp(d20, 40, 80), fall(rise, 60, 120))
    tom = min(ramp(pitch, 0.5, 0.7), ramp(f0, 60, 80) * fall(f0, 300, 450), fall(cf, 400, 900),
              ramp(sub + low + mid, 0.85, 0.95), ramp(d20, 100, 160), fall(flat, 0.03, 0.08), fall(rise, 60, 120))
    clap_shape = min(ramp(cmax, 1200, 2000) * fall(cmax, 5000, 7000), ramp(mid + himid, 0.50, 0.70),
                     fall(sub, 0.35, 0.55), fall(d20, 150, 250), ramp(flat, 0.04, 0.10), fall(rise, 40, 80))
    clap = min(ramp(bursts, 1, 2), clap_shape)                    # a clap is a clap-shaped hit with a flam
    snare = min(ramp(max(c30, c100), 250, 350) * fall(c30, 3000, 4000), ramp(body30, 0.45, 0.65), fall(sub, 0.30, 0.50),
                fall(d20, 250, 400) * ramp(d40, 60, 120), ramp(flat, 0.004, 0.009), fall(pitch, 0.86, 0.94),
                fall(rise, 40, 80), 1.0 - tonal_ring, 1.0 - 0.5 * clap)
    bright = min(ramp(c100, 3000, 4500), ramp(hh30, 0.30, 0.50), fall(sub, 0.10, 0.25), fall(rise, 70, 120))
    closed_hat = min(bright, max(fall(hd40, 200, 350), 0.8 * fall(hd20, 60, 120)), fall(d40, 350, 500))
    ring = max(hd40, d40) if hh30 >= 0.6 else hd40
    open_hat = min(bright, ramp(hd20, 100, 160), ramp(ring, 120, 180) if abrupt else ramp(ring, 280, 400), fall(d40, 900, 1400))
    cymbal = min(ramp(cmax, 2500, 3500), ramp(himid + high, 0.55, 0.70), ramp(d40, 700, 1000), ramp(d20, 250, 400), fall(sub, 0.10, 0.30))
    if truncated or abrupt:
        # the capture or the sample ends while still loud: open hat and cymbal cannot be told apart by length
        cymbal = max(cymbal, min(open_hat, ramp(d20, 150, 250)))
    scores = {"kick": kick, "snare": snare, "clap": clap, "closed hat": closed_hat, "open hat": open_hat, "tom": tom, "cymbal": cymbal}
    scores["perc"] = 1.0 - max(scores.values())
    return {k: round(float(v), 3) for k, v in scores.items()}


def feature_hints(f: dict[str, Any]) -> list[str]:
    """Short descriptors of a hit (low, mid, bright, tonal about N Hz, noisy, long, very short, slow attack) so a hit the
    rules do not place can still be talked about."""
    if f.get("silent"):
        return ["silent"]
    b = f["bands"]
    out = []
    if b["sub"] + b["low"] >= 0.8:
        out.append("low")
    if f["centroid_hz"] >= 3000:
        out.append("bright")
    elif f["centroid_hz"] >= 800:
        out.append("mid")
    if f["pitchedness"] >= 0.7 and f["flatness"] < 0.05:
        out.append(f"tonal about {f['f0_hz']:.0f} Hz")
    elif f["flatness"] >= 0.2:
        out.append("noisy")
    d40 = f.get("decay_40_ms")
    if d40 is None or d40 >= 600:
        out.append("long")
    elif d40 <= 100:
        out.append("very short")
    if f.get("attack_ms", 0) >= 100:
        out.append("slow attack")
    return out


def feature_flags(f: dict[str, Any]) -> list[str]:
    """Capture problems the caller should know about."""
    flags = []
    if f.get("truncated"):
        flags.append("capture cut before the hit decayed: decays are lower bounds")
    if f.get("abrupt_end"):
        flags.append("sample ends abruptly while still loud: decays are lower bounds")
    if f.get("pre_onset_db") is not None and f["pre_onset_db"] > -40.0:
        flags.append(f"signal before the onset ({f['pre_onset_db']} dB re peak): the previous hit leaks into this capture")
    return flags


def _ms(v: float | None) -> str:
    return "the clip's end" if v is None else f"{v:.0f} ms"


def _reason(label: str, f: dict[str, Any], s: dict[str, float], ranked: list[tuple[str, float]]) -> str:
    b = f["bands"]
    c100, cf = float(f["centroid_100ms_hz"]), float(f["centroid_hz"])
    if label == "kick":
        return (f"low hit: {100 * (b['sub'] + b['low']):.0f}% of the energy below 400 Hz, centroid {c100:.0f} Hz, "
                f"{_ms(f['decay_20_ms'])} to -20 dB")
    if label == "snare":
        return f"drum body with noise: centroid {c100:.0f} Hz, flatness {f['flatness']:.2f}, down 40 dB in {_ms(f['decay_40_ms'])}"
    if label == "clap":
        return f"{f['bursts']} bursts in the first 50 ms on a mid-bright noise body (centroid {max(c100, cf):.0f} Hz)"
    if label == "closed hat":
        return f"bright hit (centroid {c100:.0f} Hz) whose part above 2 kHz is gone within {_ms(f['hp_decay_40_ms'])}"
    if label == "open hat":
        return f"bright hit (centroid {c100:.0f} Hz) that rings {_ms(f['decay_40_ms'])} to -40 dB"
    if label == "tom":
        return f"pitched body about {f['f0_hz']:.0f} Hz (periodicity {f['pitchedness']:.2f}) ringing {_ms(f['decay_20_ms'])} to -20 dB"
    if label == "cymbal":
        return f"bright (centroid {max(c100, cf):.0f} Hz) and long: {_ms(f['decay_40_ms'])} to -40 dB"
    if label == "perc":
        hints = ", ".join(feature_hints(f)) or "no strong trait"
        best = ranked[0]
        nearest = f"nearest {best[0]} at {best[1]:.2f}" if best[1] > 0.0 else "no named class scores at all"
        return f"no named class fits ({nearest}): {hints}"
    cands = ", ".join(f"{k} {v:.2f}" for k, v in ranked[:3] if v > 0)
    return f"between {cands}; " + (", ".join(feature_hints(f)) or "no strong trait")


def classify_key_detail(features: dict[str, Any]) -> dict[str, Any]:
    """Everything classify_key knows: label, confidence (0..1), tier (high / medium / low), reason, best, scores,
    candidates (up to three named classes with score > 0, best first), margin, hints, flags."""
    f = features
    if f.get("silent"):
        return {"label": "silent", "confidence": 1.0, "tier": "high", "best": "silent", "score": 1.0, "margin": None,
                "reason": f"nothing there: peak {f.get('peak_db', -120.0)} dBFS over {f.get('clip_ms', 0)} ms",
                "scores": class_scores(f), "candidates": [], "hints": ["silent"], "flags": []}
    s = class_scores(f)
    named = {k: v for k, v in s.items() if k in NAMED_CLASSES}
    ranked = sorted(named.items(), key=lambda kv: (-kv[1], NAMED_CLASSES.index(kv[0])))
    top, second = ranked[0], ranked[1]
    candidates = [k for k, v in ranked[:3] if v > 0.0]
    base = {"scores": s, "candidates": candidates, "hints": feature_hints(f), "flags": feature_flags(f)}
    if top[1] < PERC_BELOW:
        conf = round(1.0 - top[1], 2)
        return {"label": "perc", "confidence": conf, "tier": "high" if conf >= CONFIDENCE_HIGH else "medium", "best": "perc",
                "score": s["perc"], "margin": None, "reason": _reason("perc", f, s, ranked), **base}
    margin = round(top[1] - second[1], 3)
    # the study's tiers: high = best >= 0.7 leading by >= 0.3, medium = best >= 0.5 leading by >= 0.15, else unsure;
    # the number is min(best, 0.4 + margin), which lands at or above 0.7 / 0.5 exactly when those hold
    if top[1] >= CONFIDENCE_HIGH and margin >= 0.3:
        tier = "high"
    elif top[1] >= CONFIDENCE_LABEL and margin >= 0.15:
        tier = "medium"
    else:
        tier = "low"
    conf = min(top[1], 0.4 + margin)
    conf = round(min(conf, CONFIDENCE_LABEL - 0.01) if tier == "low" else conf, 2)
    label = top[0] if tier != "low" else UNSURE
    reason = _reason(label, f, s, ranked)
    if label != UNSURE and tier == "medium" and second[1] > 0.0:
        reason += f" (runner-up {second[0]} {second[1]:.2f})"
    if base["flags"]:
        reason += "; " + base["flags"][0].split(":")[0]
    return {"label": label, "confidence": conf, "tier": tier, "best": top[0], "score": top[1], "margin": margin, "reason": reason, **base}


def classify_key(features: dict[str, Any]) -> tuple[str, float, str]:
    """(label, confidence 0..1, reason) for one hit from its key_features(). Labels: kick, snare, clap, closed hat,
    open hat, tom, cymbal, perc, silent, or "unsure" when the confidence is below 0.5. The reason is one short sentence
    with the numbers that decided; classify_key_detail() has the full score table."""
    d = classify_key_detail(features)
    return d["label"], d["confidence"], d["reason"]


# --------------------------------------------------------------------------- layout prior

def combine_with_prior(midi: int, label: str, confidence: float, reason: str = "", scores: dict[str, float] | None = None,
                       playmode: int | None = None) -> tuple[str, float, str]:
    """Audio verdict plus the factory layout for that key -> (label, confidence, reason).

    A high-confidence audio verdict (>= 0.7) is never overridden across families: a tom on a hat key stays a tom and the
    reason notes the departure. Within the hat family the choke pair decides which hat is which: a closed-hat verdict on
    D#3 (63) with the choke marker becomes the kit's open hat at medium confidence, and vice versa on C#3 (61). Below
    high confidence the layout breaks ties: when the class the layout expects scores at least 0.35 and within 0.3 of the
    best (or the audio was plainly unsure and no scores were given), that class is taken at 0.6 and the reason says the
    layout decided. Keys without a strong convention (57, 58, 59, 65 .. 67, 69 .. 76) come back unchanged."""
    midi = int(midi)
    entry = FACTORY_LAYOUT.get(midi)
    prior = entry["class"] if entry and entry.get("prior") else None
    name = field_key_name(midi)
    if label == "silent" or prior is None:
        return label, confidence, reason
    choke = entry.get("choke") if playmode in (None, PLAYMODE_CHOKE) else None
    sep = "; " if reason else ""
    if confidence >= CONFIDENCE_HIGH:
        if label == prior:
            return label, confidence, reason + (f"{sep}{choke}" if choke else "")
        if choke and label in ("closed hat", "open hat"):
            kind = "closed" if prior == "closed hat" else "open"
            return label, confidence, (f"{reason}{sep}{name} is the factory {kind}-hat key of the choke group ({choke}), "
                                       f"but the audio says {label}; the roles follow the audio")
        return label, confidence, reason + f"{sep}audio says {label} where the factory layout expects {prior} on {name}"
    if scores is None:
        if label == UNSURE:
            return prior, 0.6, reason + f"{sep}the factory layout for {name} ({prior}) decided" + (f"; {choke}" if choke else "")
        if label == prior:
            return label, confidence, reason + f"{sep}agrees with the factory layout" + (f"; {choke}" if choke else "")
        return label, confidence, reason + f"{sep}the factory layout expects {prior} on {name}"
    ps = float(scores.get(prior, 0.0))
    best = max((float(v) for k, v in scores.items() if k in NAMED_CLASSES), default=0.0)
    if ps >= PERC_BELOW and ps >= best - 0.3:
        if label == prior:
            return label, max(confidence, 0.6), reason + f"{sep}agrees with the factory layout" + (f"; {choke}" if choke else "")
        was = f"audio was {label}" + (f" (best {max(scores, key=scores.get)} {best:.2f})" if label == UNSURE else f" at {confidence:.2f}")
        return prior, 0.6, f"{was}; the factory layout for {name} ({prior}, {ps:.2f} here) decided" + (f"; {choke}" if choke else "")
    if label == prior:
        return label, confidence, reason
    return label, confidence, reason + f"{sep}the factory layout expects {prior} on {name} but it scores only {ps:.2f} here"


# --------------------------------------------------------------------------- kit maps

FEATURE_SUMMARY = ("peak_db", "decay_20_ms", "decay_40_ms", "hp_decay_40_ms", "attack_ms", "centroid_hz", "centroid_100ms_hz",
                   "flatness", "pitchedness", "f0_hz", "bursts", "zcr_hz")


def _feature_summary(f: dict[str, Any]) -> dict[str, Any]:
    out = {k: f.get(k) for k in FEATURE_SUMMARY if k in f}
    if "bands" in f:
        out.update({k: f["bands"][k] for k in BAND_NAMES})
    return out


@dataclass
class KitKey:
    """One key of a kit map. `label` is the verdict (human label, or classifier plus layout), `source` says which."""
    midi: int
    field_name: str = ""
    label: str = UNSURE
    confidence: float | None = None
    notes: str = ""                 # free text; the owner's words
    reason: str = ""                # the classifier's one-sentence justification
    source: str = "audio"           # human | audio | audio+layout | layout
    audio_label: str | None = None  # the verdict before the layout prior or a human label replaced it
    candidates: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)      # playmode, pan_ab, volume, region, choke facts
    flags: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    playmode: int | None = None
    stacked_ab: bool | None = None
    pan: int | None = None
    volume: int | None = None
    region_ms: float | None = None
    channel_a: str | None = None
    channel_b: str | None = None
    features: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.midi = int(self.midi)
        if not self.field_name:
            self.field_name = field_key_name(self.midi)

    @property
    def midi_name(self) -> str:
        return midi_key_name(self.midi)

    @property
    def human(self) -> bool:
        return self.source == "human"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"field_name": self.field_name, "midi_name": self.midi_name, "label": self.label}
        if self.confidence is not None:
            d["confidence"] = round(float(self.confidence), 2)
        d["source"] = self.source
        for k in ("notes", "reason", "audio_label", "candidates", "facts", "flags", "hints", "playmode", "stacked_ab", "pan",
                  "volume", "region_ms", "channel_a", "channel_b", "features"):
            v = getattr(self, k)
            if v not in (None, "", [], {}):
                d[k] = v
        for k, v in self.extra.items():
            d.setdefault(k, v)
        return d

    @classmethod
    def from_dict(cls, midi: int, d: dict[str, Any]) -> "KitKey":
        d = dict(d)
        known = {"field_name", "midi_name", "label", "confidence", "notes", "reason", "source", "audio_label", "candidates", "facts",
                 "flags", "hints", "playmode", "stacked_ab", "pan", "volume", "region_ms", "channel_a", "channel_b", "features"}
        source = d.get("source")
        if source is None:
            source = "human" if "confidence" not in d else "audio"     # the owner's file carries no confidence: labels by ear
        conf = d.get("confidence")
        if conf is None and source == "human":
            conf = 1.0
        key = cls(midi=int(midi), field_name=str(d.get("field_name") or ""), label=str(d.get("label") or UNSURE),
                  confidence=None if conf is None else float(conf), notes=str(d.get("notes") or ""), reason=str(d.get("reason") or ""),
                  source=str(source), audio_label=d.get("audio_label"), candidates=list(d.get("candidates") or []),
                  facts=list(d.get("facts") or []), flags=list(d.get("flags") or []), hints=list(d.get("hints") or []),
                  playmode=d.get("playmode"), stacked_ab=d.get("stacked_ab"), pan=d.get("pan"), volume=d.get("volume"),
                  region_ms=d.get("region_ms"), channel_a=d.get("channel_a"), channel_b=d.get("channel_b"),
                  features=dict(d.get("features") or {}), extra={k: v for k, v in d.items() if k not in known})
        return key

    def describe(self) -> str:
        conf = "" if self.confidence is None else f" {self.confidence:.2f}"
        by = " by ear" if self.human else ""
        text = f"{self.midi} {self.field_name:>4}: {self.label}{conf}{by}"
        if self.notes:
            text += f" ({self.notes})"
        elif self.reason:
            text += f": {self.reason}"
        return text


def _slot_number(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


@dataclass
class KitMap:
    """A drum kit's 24 keys with labels, plus roles (BD, SN, CH, OH, CL, TOM_LOW/MID/HIGH, CY, PERC, and BD2/SN2/CH_CHOKE
    when present) -> MIDI note. JSON-compatible with the owner's labelled file (kit_name, slot, keys{midi: {field_name,
    label, notes, confidence}}, roles{ROLE: midi}); unknown fields are kept in `extra`."""
    kit_name: str = ""
    slot: int | None = None
    keys: dict[int, KitKey] = field(default_factory=dict)
    roles: dict[str, int] = field(default_factory=dict)
    source: str = ""
    file: str | None = None
    clips_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    path: str | None = None         # where the map was loaded from; not written into the JSON

    def key(self, midi: int) -> KitKey | None:
        return self.keys.get(int(midi))

    def label_of(self, midi: int) -> str:
        k = self.keys.get(int(midi))
        return k.label if k else UNSURE

    def keys_labelled(self, label: str) -> list[int]:
        return sorted(m for m, k in self.keys.items() if k.label == label)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kit_name": self.kit_name, "slot": self.slot}
        if self.source:
            d["source"] = self.source
        if self.file:
            d["file"] = self.file
        if self.clips_dir:
            d["clips_dir"] = self.clips_dir
        d["keys"] = {str(m): self.keys[m].to_dict() for m in sorted(self.keys)}
        d["roles"] = dict(self.roles)
        for k, v in self.extra.items():
            d.setdefault(k, v)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1) + "\n"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KitMap":
        if not isinstance(d, dict):
            raise ValueError(f"a kit map is a JSON object with kit_name, slot, keys and roles, got {type(d).__name__}")
        keys: dict[int, KitKey] = {}
        for m, kd in (d.get("keys") or {}).items():
            try:
                midi = int(m)
            except (TypeError, ValueError):
                raise ValueError(f"kit map key {m!r} is not a MIDI note number (keys run {KIT_FIRST_MIDI} .. {KIT_LAST_MIDI})")
            if isinstance(kd, str):
                kd = {"label": kd}
            keys[midi] = KitKey.from_dict(midi, kd or {})
        roles: dict[str, int] = {}
        for r, m in (d.get("roles") or {}).items():
            try:
                roles[str(r)] = int(m)
            except (TypeError, ValueError):
                raise ValueError(f"role {r!r} must map to a MIDI note number, got {m!r}")
        known = {"kit_name", "slot", "keys", "roles", "source", "file", "clips_dir"}
        return cls(kit_name=str(d.get("kit_name") or ""), slot=_slot_number(d.get("slot")), keys=keys, roles=roles,
                   source=str(d.get("source") or ""), file=d.get("file"), clips_dir=d.get("clips_dir"),
                   extra={k: v for k, v in d.items() if k not in known})

    @classmethod
    def from_json(cls, text: str | bytes | dict[str, Any]) -> "KitMap":
        if isinstance(text, dict):
            return cls.from_dict(text)
        try:
            d = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"kit map is not valid JSON: {e}")
        return cls.from_dict(d)

    def merge(self, human: "KitMap") -> "KitMap":
        return merge_kit_maps(self, human)

    def describe(self) -> str:
        head = f"{self.kit_name or 'kit'}" + (f" (drum slot {self.slot})" if self.slot else "")
        lines = [head] + [self.keys[m].describe() for m in sorted(self.keys)]
        if self.roles:
            lines.append("roles: " + ", ".join(f"{r} = {m} {field_key_name(m)}" for r, m in self.roles.items()))
        return "\n".join(lines)


def assign_roles(keys: dict[int, KitKey]) -> dict[str, int]:
    """Pick the best key for each role from the labels: the factory position when it holds that class, otherwise the most
    confident key of the class, otherwise an unsure key whose top two candidates include it. Toms are ranked by pitch
    (f0, or the centroid when unpitched): one tom serves all three tom roles, two serve LOW (also MID) and HIGH."""
    def pick(cls: str, prefer: tuple[int, ...], exclude: set[int] = frozenset()) -> int | None:
        sure = {m: k for m, k in keys.items() if k.label == cls and m not in exclude}
        for m in prefer:
            if m in sure:
                return m
        if sure:
            return max(sure, key=lambda m: (sure[m].confidence or 0.0, -m))
        maybe = {m: k for m, k in keys.items() if k.label == UNSURE and cls in (k.candidates or [])[:2] and m not in exclude}
        for m in prefer:
            if m in maybe:
                return m
        if maybe:
            return max(maybe, key=lambda m: (maybe[m].confidence or 0.0, -m))
        return None

    roles: dict[str, int] = {}
    bd = pick("kick", (53, 54))
    if bd is not None:
        roles["BD"] = bd
        bd2 = pick("kick", (54, 53), {bd})
        if bd2 is not None:
            roles["BD2"] = bd2
    sn = pick("snare", (55, 56))
    if sn is not None:
        roles["SN"] = sn
        sn2 = pick("snare", (56, 55), {sn})
        if sn2 is not None:
            roles["SN2"] = sn2
    ch = pick("closed hat", (60, 62, 64, 61))
    if ch is not None:
        roles["CH"] = ch
    oh = pick("open hat", (63,))
    if oh is None:
        # no genuine open hat: the longest-ringing hat serves, preferring the choke group's D#3 (63)
        hats = {m: kk for m, kk in keys.items() if kk.label in ("closed hat", "open hat", "cymbal") and m != roles.get("CH")}
        if hats:
            def ring(m: int) -> float:
                f = hats[m].features or {}
                return float(f.get("decay_40_ms") or f.get("decay_20_ms") or 0.0) + (0.5 if m == 63 else 0.0)
            oh = max(hats, key=ring)
    if oh is not None:
        roles["OH"] = oh
    k61 = keys.get(61)
    if k61 is not None and k61.label == "closed hat" and oh == 63:
        roles["CH_CHOKE"] = 61
    cl = pick("clap", (58,))
    if cl is not None:
        roles["CL"] = cl
    toms = [(m, k) for m, k in keys.items() if k.label == "tom"]
    if toms:
        def pitch(k: KitKey) -> float:
            f = k.features or {}
            f0 = f.get("f0_hz") or 0.0
            return float(f0) if f0 and (f.get("pitchedness") or 0.0) >= 0.5 else float(f.get("centroid_hz") or 0.0)
        ordered = [m for m, _ in sorted(toms, key=lambda mk: (pitch(mk[1]), mk[0]))]
        if len(ordered) == 1:
            roles["TOM_LOW"] = roles["TOM_MID"] = roles["TOM_HIGH"] = ordered[0]
        elif len(ordered) == 2:
            roles["TOM_LOW"] = roles["TOM_MID"] = ordered[0]
            roles["TOM_HIGH"] = ordered[1]
        else:
            roles["TOM_LOW"], roles["TOM_MID"], roles["TOM_HIGH"] = ordered[0], ordered[len(ordered) // 2], ordered[-1]
    cy = pick("cymbal", (68,))
    if cy is not None:
        roles["CY"] = cy
    perc = pick("perc", ())
    if perc is not None:
        roles["PERC"] = perc
    return roles


def merge_kit_maps(base: KitMap, human: KitMap) -> KitMap:
    """`base` (usually classifier output) with every human label from `human` written over it: a key the owner labelled
    keeps the owner's label, notes and full confidence, and the classifier's verdict moves to audio_label with its reason
    kept for reference. Roles are recomputed from the merged labels, then the owner's roles override them."""
    out = copy.deepcopy(base)
    for midi, hk in human.keys.items():
        if not hk.human:
            continue
        bk = out.keys.get(midi)
        if bk is None:
            out.keys[midi] = copy.deepcopy(hk)
            continue
        if bk.label != hk.label or not bk.human:
            bk.audio_label = bk.audio_label if bk.human else bk.label
        bk.label = hk.label
        bk.confidence = hk.confidence if hk.confidence is not None else 1.0
        bk.source = "human"
        if hk.notes:
            bk.notes = hk.notes
        for k, v in hk.extra.items():
            bk.extra.setdefault(k, v)
    if human.kit_name:
        out.kit_name = human.kit_name
    if human.slot is not None:
        out.slot = human.slot
    for k, v in human.extra.items():
        out.extra.setdefault(k, v)
    out.roles = assign_roles(out.keys)
    out.roles.update(human.roles)
    return out


# --------------------------------------------------------------------------- building maps

def _key_facts(region: dict[str, Any] | None) -> list[str]:
    if region is None:
        return []
    facts = [PLAYMODE_NOTES.get(int(region["playmode"]), f"playmode {region['playmode']}, a value no factory kit uses")]
    if region.get("stacked_ab"):
        b = float(region["pan"]) / P.DRUM_PAN_MAX
        which = "channel A only" if region["pan"] == 0 else "channel B only" if region["pan"] >= P.DRUM_PAN_MAX else f"{100 * (1 - b):.0f}% A + {100 * b:.0f}% B"
        facts.append(f"stacked A/B key (pan_ab): the left and right channels hold two sounds, pan {region['pan']} plays {which}")
    elif region.get("pan") not in (None, P.DRUM_PAN_CENTRE):
        facts.append(f"pan {region['pan']} ({'left' if region['pan'] < P.DRUM_PAN_CENTRE else 'right'} of centre 16384)")
    vol = region.get("volume")
    if vol is not None:
        gain = 40.0 * np.log10(max(int(vol), 1) / P.DRUM_GAIN_UNITY)
        facts.append(f"volume {vol}" + (" (factory default)" if vol == P.DRUM_GAIN_UNITY else f" (about {gain:+.1f} dB against the default)"))
    if region.get("pitch"):
        facts.append(f"pitch {region['pitch']} (0 is untuned)")
    if region.get("reverse") is not None and region["reverse"] >= 16384:
        facts.append(f"reverse {region['reverse']}: may play backwards (encoding unverified)")
    facts.append(f"region {region['seconds'] * 1000:.0f} ms")
    return facts


def _classified_key(midi: int, f: dict[str, Any], region: dict[str, Any] | None = None, channel_a: dict[str, Any] | None = None,
                    channel_b: dict[str, Any] | None = None) -> KitKey:
    d = classify_key_detail(f)
    playmode = int(region["playmode"]) if region else None
    label, conf, reason = combine_with_prior(midi, d["label"], d["confidence"], d["reason"], d["scores"], playmode)
    key = KitKey(midi=midi, label=label, confidence=conf, reason=reason, source="audio" if label == d["label"] else "audio+layout",
                 audio_label=None if label == d["label"] else d["label"], candidates=d["candidates"], facts=_key_facts(region),
                 flags=list(d["flags"]), hints=list(d["hints"]), features=_feature_summary(f))
    if region is not None:
        key.playmode = playmode
        key.stacked_ab = bool(region.get("stacked_ab"))
        key.pan = int(region["pan"])
        key.volume = int(region["volume"])
        key.region_ms = round(float(region["seconds"]) * 1000.0, 1)
        if not f.get("silent"):
            key.features["expected_live_peak_db"] = round(predict_live_peak_db(float(f["peak_db"]), key.volume), 1)
    if channel_a is not None and channel_b is not None:
        ca, cb = classify_key_detail(channel_a), classify_key_detail(channel_b)
        key.channel_a, key.channel_b = ca["label"], cb["label"]
        key.flags.append(f"stacked A/B key: channel A alone is {ca['label']}, channel B alone is {cb['label']}")
        if label in (UNSURE, "perc") and any(c in NAMED_CLASSES for c in (ca["label"], cb["label"])):
            key.reason += f"; channel A alone is {ca['label']}, channel B alone is {cb['label']}"
    return key


def kit_pcm(preset: P.PresetFile) -> np.ndarray:
    """SSND payload of a sampler kit as float32 (frames, 2), from the 16-bit little-endian data."""
    ch = dict(preset.chunks)
    if b"SSND" not in ch:
        raise ValueError(f"{preset.name!r}: no SSND chunk, the file carries no sample data")
    body = ch[b"SSND"][8:]
    channels, _, _ = P._comm_fields(preset)
    channels = channels if channels in (1, 2) else 2
    n = (len(body) // (2 * channels)) * channels
    pcm = np.frombuffer(body[:n * 2], dtype="<i2").reshape(-1, channels).astype(np.float32) / 32768.0
    if channels == 1:
        pcm = np.repeat(pcm, 2, axis=1)
    return pcm


def region_audio(pcm: np.ndarray, region: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mono, left, right) of one key's region. A stacked A/B key's mono is the two channels cross-faded by its pan
    (0 = A = left, 32766 = B = right); a plain key's mono is the mean of both channels."""
    a, b = max(int(region["start_frame"]), 0), min(int(region["end_frame"]), pcm.shape[0])
    seg = pcm[a:b] if b > a else np.zeros((0, 2), dtype=np.float32)
    left, right = seg[:, 0].astype(np.float32), seg[:, 1].astype(np.float32)
    if left.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), left, right
    if region.get("stacked_ab"):
        w = float(region["pan"]) / P.DRUM_PAN_MAX
        return ((1.0 - w) * left + w * right).astype(np.float32), left, right
    return ((left + right) * 0.5).astype(np.float32), left, right


def _dbox_map(preset: P.PresetFile, path: str, slot: int | None) -> KitMap:
    keys: dict[int, KitKey] = {}
    data = preset.meta.get("dbox_data") or []
    for i in range(KIT_KEYS):
        midi = KIT_FIRST_MIDI + i
        entry = FACTORY_LAYOUT.get(midi)
        row = data[i] if i < len(data) and isinstance(data[i], list) else []
        facts = [f"dbox voice: {', '.join(f'{n} {v}' for n, v in zip(P.DBOX_PARAMS, row))}"] if row else []
        if entry and entry.get("prior"):
            keys[midi] = KitKey(midi=midi, label=entry["class"], confidence=0.5, source="layout",
                                reason=f"dbox kit, no sample audio in the file: the factory layout expects {entry['class']} on {field_key_name(midi)}"
                                       + (f"; {entry['choke']}" if entry.get("choke") else ""), facts=facts)
        else:
            keys[midi] = KitKey(midi=midi, label=UNSURE, confidence=0.0, source="layout",
                                reason="dbox kit, no sample audio in the file and no factory convention for this key", facts=facts)
    return KitMap(kit_name=preset.name, slot=slot, keys=keys, roles=assign_roles(keys), file=path,
                  source="factory layout only: a dbox kit carries no sample audio; audition it and use kit_map_from_clips for real labels")


def kit_map_from_file(path: str | P.PresetFile, slot: int | None = None) -> KitMap:
    """Decode a Field-written drum kit file and classify its 24 keys.

    Each region (op_bridge.presets.drum_kit_regions, positions in 2^31 = 20 s units) is measured with key_features(),
    classified, combined with the factory layout, and annotated with its playmode, pan / pan_ab, volume and length. Stacked
    A/B keys are cross-faded by their pan and also classified per channel. The slot number comes from a file name like
    drum/user/3.aif unless given. A dbox kit has no sample audio: its map carries the factory layout only, at 0.5."""
    if isinstance(path, P.PresetFile):
        preset, path_str = path, ""
    else:
        path_str = os.path.expanduser(str(path))
        if not os.path.isfile(path_str):
            raise ValueError(f"{path_str}: no such file; Field-written kits live at <disk backup>/drum/user/1-8.aif")
        try:
            preset = P.read_preset(path_str)
        except ValueError as e:
            raise ValueError(f"{path_str} is not a Field preset: {e}")
    if slot is None and path_str:
        m = re.fullmatch(r"(\d+)\.aif{1,2}", os.path.basename(path_str), re.IGNORECASE)
        slot = int(m.group(1)) if m else None
    if preset.kind != "drum":
        raise ValueError(f"{path_str or preset.name!r} is a {preset.engine} synth preset, not a drum kit; drum kits have type 'drum' or 'dbox'")
    if preset.engine == "dbox":
        return _dbox_map(preset, path_str or None, slot)
    if "start" not in preset.meta or "end" not in preset.meta:
        raise ValueError(f"{path_str or preset.name!r}: drum kit without start/end arrays; the file is not a Field sampler kit")
    pcm = kit_pcm(preset)
    keys: dict[int, KitKey] = {}
    for region in P.drum_kit_regions(preset):
        midi = int(region["note"])
        mono, left, right = region_audio(pcm, region)
        f = key_features(mono, P.DRUM_SAMPLERATE)
        if region.get("stacked_ab") and mono.shape[0]:
            keys[midi] = _classified_key(midi, f, region, key_features(left, P.DRUM_SAMPLERATE), key_features(right, P.DRUM_SAMPLERATE))
        else:
            keys[midi] = _classified_key(midi, f, region)
    fx = preset.meta.get("fx_type"), bool(preset.meta.get("fx_active"))
    lfo = preset.meta.get("lfo_type"), bool(preset.meta.get("lfo_active"))
    source = (f"classified from the kit file's 24 regions ({pcm.shape[0] / P.DRUM_SAMPLERATE:.2f} s bank); fx {fx[0]} "
              f"{'on' if fx[1] else 'off'}, lfo {lfo[0]} {'on' if lfo[1] else 'off'}")
    if fx[1] or lfo[1]:
        source += "; active fx/lfo change the played sound but not these file regions"
    return KitMap(kit_name=preset.name, slot=slot, keys=keys, roles=assign_roles(keys), file=path_str or None, source=source)


def kit_map_from_clips(clip_dir: str, kit_name: str | None = None, slot: int | None = None) -> KitMap:
    """Classify live auditions saved as <midi>_<name>.wav (one hit per file, e.g. 63_D#4.wav) in `clip_dir`.

    Clips are treated as cut captures: a hit still ringing at the end of its file is flagged and its decays are lower
    bounds, and a tail from the previous key leaking into a clip is flagged too. Keys without a clip are absent from the
    map. Playmode and pan facts are not known from clips; only the layout convention is applied."""
    d = os.path.expanduser(str(clip_dir))
    if not os.path.isdir(d):
        raise ValueError(f"{d}: not a directory; expected live clips named <midi>_<name>.wav, e.g. 53_F3.wav")
    paths = sorted(glob.glob(os.path.join(d, "*.wav")))
    keys: dict[int, KitKey] = {}
    skipped: list[str] = []
    for path in paths:
        base = os.path.basename(path)
        m = re.match(r"(\d+)_", base)
        if not m:
            skipped.append(base)
            continue
        midi = int(m.group(1))
        if not KIT_FIRST_MIDI <= midi <= KIT_LAST_MIDI:
            skipped.append(base)
            continue
        try:
            x, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:  # soundfile raises RuntimeError / sf.LibsndfileError on bad files
            raise ValueError(f"{path}: cannot be read as audio ({e})")
        mono = x.mean(axis=1)
        if sr != SR:
            mono = signal.resample_poly(mono, SR, int(sr)).astype(np.float32)
        f = key_features(mono, SR, clip=True)
        key = _classified_key(midi, f)
        key.extra["clip"] = base
        keys[midi] = key
    if not keys:
        raise ValueError(f"{d}: no clips named <midi>_<name>.wav with MIDI {KIT_FIRST_MIDI} .. {KIT_LAST_MIDI} "
                         f"({len(paths)} wav files, none usable)")
    source = f"classified from {len(keys)} live clips in {d}"
    if skipped:
        source += f"; skipped {len(skipped)} file(s) not named <midi>_<name>.wav: {', '.join(skipped[:5])}"
    return KitMap(kit_name=kit_name or os.path.basename(os.path.normpath(d)), slot=slot, keys=keys, roles=assign_roles(keys),
                  clips_dir=d, source=source)


# --------------------------------------------------------------------------- storage

def kits_dir(home: str) -> str:
    return os.path.join(os.path.expanduser(str(home)), "kits")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def _check_slot(slot: Any) -> int:
    n = _slot_number(slot)
    if n is None or n < 1:
        raise ValueError(f"slot must be a drum slot number 1 .. 8, got {slot!r}")
    return n


def kit_map_path(home: str, slot: int, kit_name: str = "") -> str:
    slot = _check_slot(slot)
    slug = _slug(kit_name)
    return os.path.join(kits_dir(home), f"drum-slot-{slot}-{slug}.json" if slug else f"drum-slot-{slot}.json")


def _slot_files(home: str, slot: int) -> list[str]:
    d = kits_dir(home)
    files = glob.glob(os.path.join(d, f"drum-slot-{slot}-*.json")) + glob.glob(os.path.join(d, f"drum-slot-{slot}.json"))
    return sorted(set(files), key=lambda p: (os.path.getmtime(p), p), reverse=True)


def load_kit_map_file(path: str) -> KitMap:
    path = os.path.expanduser(str(path))
    if not os.path.isfile(path):
        raise ValueError(f"{path}: no such kit map file")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        kitmap = KitMap.from_json(text)
    except ValueError as e:
        raise ValueError(f"{path}: {e}")
    kitmap.path = path
    return kitmap


def load_kit_map(home: str, slot: int) -> KitMap | None:
    """The newest saved map for a drum slot under <home>/kits (drum-slot-<n>-<kit>.json), or None when there is none."""
    slot = _check_slot(slot)
    files = _slot_files(home, slot)
    if not files:
        return None
    kitmap = load_kit_map_file(files[0])
    if kitmap.slot is None:
        kitmap.slot = slot
    return kitmap


def save_kit_map(home: str, slot: int, kitmap: KitMap, keep_human_labels: bool = True) -> str:
    """Write a kit map to <home>/kits/drum-slot-<slot>-<kit name>.json and return the path. When a map for that slot is
    already saved for the same kit (or the saved map names no kit), its human labels, notes and roles are merged in first
    so that labels by ear always survive a re-classification; keep_human_labels=False writes the map as given."""
    slot = _check_slot(slot)
    if not isinstance(kitmap, KitMap):
        raise ValueError("save_kit_map needs a KitMap (from kit_map_from_file, kit_map_from_clips or KitMap.from_json)")
    if keep_human_labels:
        target = kit_map_path(home, slot, kitmap.kit_name)
        existing = load_kit_map_file(target) if os.path.isfile(target) else load_kit_map(home, slot)
        if existing is not None and (os.path.isfile(target) or not existing.kit_name or not kitmap.kit_name or existing.kit_name == kitmap.kit_name):
            human_only = copy.deepcopy(existing)
            human_only.keys = {m: k for m, k in existing.keys.items() if k.human}
            kitmap = merge_kit_maps(kitmap, human_only)
    kitmap = copy.deepcopy(kitmap)
    kitmap.slot = slot
    path = kit_map_path(home, slot, kitmap.kit_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(kitmap.to_json())
    os.replace(tmp, path)
    kitmap.path = path
    return path
