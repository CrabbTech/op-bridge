"""Synth sampler presets for the OP-1 field: author them from audio on this Mac, decode the ones the Field wrote.

A synth sampler preset is an ordinary synth preset file (see presets.py) whose type is "sampler" and whose
SSND chunk is the sample itself: AIFF-C, 16-bit little-endian ("sowt") at 44.1 kHz, mono or stereo. The
JSON adds three keys to the synth block: base_freq (the frequency of the MIDI note that plays the sample
at its recorded speed), stereo (mirrors the COMM channel count) and fade (meaning unknown; the Field's own
import writer omits the key or writes 0, and factory files carry large values such as 134217728).

Everything here was learned from the sampler files the Field wrote (field-backup/*/synth, 2026-09-26): five
factory presets (pipe dream, suitcase, baby string, voices, metal mat) and four files the Field produced by
itself when it turned arbitrary audio into a sampler slot ("imports"). No value in this module was made up:
defaults are the Field's import defaults, and the region unit was fitted to every file.

The four T1 knobs (start, loop in, loop out, end) are positions on a fixed 6.000 s timeline, not fractions of
the file: 32767 = 264,600 frames at 44.1 kHz, so knob = frame * 32767 / 264600. The Field's end marker is the
index of the last frame (frames - 1) rounded up: ceil((frames - 1) * 32767 / 264600) reproduces every whole-file
end in the backups, pipe dream 177166 frames -> 21939.40 -> 21940; voices 244178 -> 30237.90 -> 30238 (rounding
the frame count itself, 30238.02, would give 30239); baby string 252757 -> 31300.29 -> 31301; suitcase 264604 ->
32767.37 -> 32767 (clamped); the Field's own imports 44100 -> 5461.04 -> 5462 and 44160 -> 5468.47 -> 5469. The
alternative "32767 = the file's own length" is falsified by the same files: every whole-file end would then be
32767, and pipe dream's end would fall at 2.69 s with audio still after it. Drum kits use the same fixed-span
idea with 2^31 over a 20 s bank (presets.DRUM_POSITION_UNIT), which corroborates the reading. The longest
factory samples are 264,598 and 264,604 frames, so 6 s is the usable maximum.
"""
from __future__ import annotations

import math
import os
import re
import struct
import time
from typing import Any

from . import presets as P
from .presets import PresetFile, RAW_MAX

# --------------------------------------------------------------------------- specification

SAMPLER_SAMPLERATE = 44100
SAMPLER_MAX_SECONDS = 6.0
SAMPLER_SPAN_FRAMES = int(SAMPLER_MAX_SECONDS * SAMPLER_SAMPLERATE)   # 264600: the frame that knob value 32767 addresses
SAMPLER_POSITION_MAX = RAW_MAX                                         # 32767
SAMPLER_UNIT_FRAMES = SAMPLER_SPAN_FRAMES / SAMPLER_POSITION_MAX       # 8.0752 frames per knob unit
SAMPLER_UNIT_SECONDS = SAMPLER_MAX_SECONDS / SAMPLER_POSITION_MAX      # 0.1831 ms per knob unit
SAMPLER_KNOBS = ["start", "loop in", "loop out", "end"]                # T1 page, knobs[0..3]
SAMPLER_KNOBS_SHIFT = ["direction", "fine tune", "loop fade", "gain"]  # shift page, knobs[4..7] (manual page 20)

# Shift page conventions, from the Field's import writer and the factory files.
SAMPLER_DIRECTION_FORWARD = 12000            # the Field's import default; 8192 (pipe dream, suitcase) is forward too
SAMPLER_DIRECTION_FORWARD_VALUES = (12000, 8192)
SAMPLER_DIRECTION_REVERSE = 24576            # UNVERIFIED: no Field-written file plays reversed. TE stores N-way switches as bin centres
                                             # (drum playmode 4096/12288/20480), so reverse is expected in the upper half, 24576 as its centre.
SAMPLER_FINE_TUNE_CENTRE = 0                 # import default; bipolar like other TE detune knobs; cents per unit unknown
SAMPLER_LOOP_FADE_OFF = 0                    # import default (no crossfade); factory values up to 27328
SAMPLER_GAIN_UNITY = 8192                    # import default, the same convention as drum volume; factory make-up gains 9173..17138
SAMPLER_GAIN_MAX = RAW_MAX / SAMPLER_GAIN_UNITY   # 4.0x if the curve is linear (hypothesis; only unity is certain)
SAMPLER_FADE_DEFAULT = 0                     # the JSON "fade" key, not the loop-fade knob: the Field's import writer omits it or writes 0
                                             # (0 in 130339-replaced/synth/8.aif, absent from the three other imports); factory files hold
                                             # large values of unknown meaning (134217728, 210526481, 213208224)
SAMPLER_NAME_MAX = P.PRESET_NAME_MAX         # 11: the longest name in any Field-written file ("baby string", "china whist", "2014-268#01")
SAMPLER_QUIET_DBFS = -6.0                    # below this peak the sample is reported as quiet (docs/device.md: a -22 dBFS cut played 16 dB low)

# What the Field itself writes when it turns arbitrary audio into a sampler slot (three such files in the backups:
# 2026-09-26-121922-replaced/synth/7.aif, 2026-09-26-123417-replaced/synth/7.aif and 8.aif, 2026-09-26-130339-replaced/synth/8.aif).
# The region knobs [0, 0, E, E] with E = the whole sample, base_freq 440.0 (A4, no key held) and the name are per file.
SAMPLER_IMPORT_ADSR = [64, 10746, 32767, 10000, 4000, 64, 4000, 18021]
SAMPLER_IMPORT_META: dict[str, Any] = {
    "adsr": list(SAMPLER_IMPORT_ADSR),
    "base_freq": 440.0,
    "fade": SAMPLER_FADE_DEFAULT,
    "fx_active": False,
    "fx_params": [8000] * 8,
    "fx_type": "delay",
    "knobs": [0, 0, SAMPLER_POSITION_MAX, SAMPLER_POSITION_MAX, SAMPLER_DIRECTION_FORWARD, SAMPLER_FINE_TUNE_CENTRE, SAMPLER_LOOP_FADE_OFF, SAMPLER_GAIN_UNITY],
    "lfo_active": False,
    "lfo_params": [16000, 0, 0, 16000, 0, 0, 0, 0],
    "lfo_type": "tremolo",
    "name": "sample",
    "octave": 0,
    "stereo": False,
    "synth_version": 3,
    "type": "sampler",
}

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)(-?\d+)\s*$")


def _default_home(home: str | None) -> str:
    return home or os.environ.get("OP_BRIDGE_HOME", os.path.expanduser("~/Music/op-bridge"))


# --------------------------------------------------------------------------- position math

def sampler_frame_to_knob(frame: float, end: bool = False, samplerate: int = SAMPLER_SAMPLERATE) -> int:
    """Sample frame -> T1 knob value on the fixed 6 s timeline: knob = frame * 32767 / (6 s * samplerate).

    Rounds to nearest. With `end=True` the value is an end marker for a region of `frame` frames: the Field
    writes the index of the last frame (frame - 1) rounded up, which reproduces all six whole-file ends in
    its files (44100 frames -> 5461.04 -> 5462; 44160 -> 5468.47 -> 5469; 244178 -> 30237.90 -> 30238, where
    rounding the count itself would give 30239). Clamped to 0..32767; frames past 264,600 (6 s at 44.1 kHz)
    cannot be addressed."""
    if end:
        exact = (float(frame) - 1.0) * SAMPLER_POSITION_MAX / (SAMPLER_MAX_SECONDS * samplerate)
        v = math.ceil(exact - 1e-9)
    else:
        exact = float(frame) * SAMPLER_POSITION_MAX / (SAMPLER_MAX_SECONDS * samplerate)
        v = round(exact)
    return int(max(0, min(SAMPLER_POSITION_MAX, v)))


def sampler_knob_to_frame(knob: int, samplerate: int = SAMPLER_SAMPLERATE) -> float:
    """T1 knob value -> sample frame (fractional): frame = knob / 32767 * 6 s * samplerate."""
    return knob / SAMPLER_POSITION_MAX * SAMPLER_MAX_SECONDS * samplerate


def sampler_seconds_to_knob(seconds: float, end: bool = False) -> int:
    """Seconds -> T1 knob value: knob = seconds / 6.0 * 32767 (one unit = 0.1831 ms). See sampler_frame_to_knob."""
    return sampler_frame_to_knob(seconds * SAMPLER_SAMPLERATE, end=end)


def sampler_knob_to_seconds(knob: int) -> float:
    """T1 knob value -> seconds: seconds = knob / 32767 * 6.0."""
    return knob / SAMPLER_POSITION_MAX * SAMPLER_MAX_SECONDS


def sampler_positions(start_s: float = 0.0, loop_in_s: float | None = None, loop_out_s: float | None = None, end_s: float | None = None,
                      frames: int | None = None, samplerate: int = SAMPLER_SAMPLERATE) -> dict[str, int]:
    """The four T1 knob values (start, loop_in, loop_out, end) for a region given in seconds.

    knob = seconds / 6.0 * 32767, rounded to nearest (the end marker is the last frame's index rounded up, like the
    Field's own writer). Evidence for
    the 6 s span is in the module docstring: every Field-written end marker lands on the file's last frame under
    this unit, within five frames. Omitted loop points fall on start and end (the import default: the whole
    sample loops); an omitted end is the whole sample when `frames` is given, else 6 s. The region must satisfy
    0 <= start <= loop in <= loop out <= end, as every Field-written file does, and stay inside the sample."""
    if frames is not None and frames > SAMPLER_SPAN_FRAMES:
        raise ValueError(f"{frames} frames is {frames / samplerate:.3f} s; the synth sampler addresses at most {SAMPLER_MAX_SECONDS:.0f} s ({SAMPLER_SPAN_FRAMES} frames at 44.1 kHz)")
    sample_end = sampler_frame_to_knob(frames, end=True, samplerate=samplerate) if frames is not None else SAMPLER_POSITION_MAX
    sample_seconds = frames / samplerate if frames is not None else SAMPLER_MAX_SECONDS
    if end_s is None:
        end = sample_end
    else:
        if end_s > sample_seconds + SAMPLER_UNIT_SECONDS:
            raise ValueError(f"end {end_s:.4f} s is past the end of the sample ({sample_seconds:.4f} s)")
        end = min(sample_end, sampler_frame_to_knob(end_s * samplerate, end=True, samplerate=samplerate))
    start = sampler_frame_to_knob((start_s or 0.0) * samplerate, samplerate=samplerate)
    loop_in = start if loop_in_s is None else sampler_frame_to_knob(loop_in_s * samplerate, samplerate=samplerate)
    loop_out = end if loop_out_s is None else sampler_frame_to_knob(loop_out_s * samplerate, samplerate=samplerate)
    if (start_s or 0.0) < 0:
        raise ValueError("start must be at or after 0 s")
    if not (0 <= start <= loop_in <= loop_out <= end <= SAMPLER_POSITION_MAX):
        raise ValueError(f"region must satisfy 0 <= start <= loop in <= loop out <= end; got start {sampler_knob_to_seconds(start):.4f} s, "
                         f"loop in {sampler_knob_to_seconds(loop_in):.4f} s, loop out {sampler_knob_to_seconds(loop_out):.4f} s, end {sampler_knob_to_seconds(end):.4f} s")
    return {"start": start, "loop_in": loop_in, "loop_out": loop_out, "end": end}


# --------------------------------------------------------------------------- pitch

def note_number(note: int | str) -> int:
    """MIDI note number from a name like C4, F#3, Bb2 (C4 = 60, A4 = 69) or an integer passed through. A MIDI number
    sent as a string ("60", as remote clients often do) is read as the number."""
    if isinstance(note, bool):
        raise ValueError("note must be a name or a MIDI number")
    if isinstance(note, int):
        if not 0 <= note <= 127:
            raise ValueError(f"MIDI note {note} outside 0..127")
        return note
    text = str(note).strip()
    if text.lstrip("-").isdigit():
        n = int(text)
        if not 0 <= n <= 127:
            raise ValueError(f"MIDI note {n} outside 0..127")
        return n
    m = _NOTE_RE.match(text)
    if not m:
        raise ValueError(f"cannot read note {note!r}; use a name like C4, F#3 or Bb2, or a MIDI number")
    letter, accidental, octave = m.groups()
    n = (int(octave) + 1) * 12 + _NOTE_NAMES.index(letter.upper()) + {"": 0, "#": 1, "b": -1}[accidental]
    if not 0 <= n <= 127:
        raise ValueError(f"note {note!r} is MIDI {n}, outside 0..127")
    return n


def note_name(midi: int) -> str:
    return _NOTE_NAMES[int(midi) % 12] + str(int(midi) // 12 - 1)


def base_freq_from_note(note: int | str) -> float:
    """base_freq for a root note: the equal-tempered frequency 440 * 2^((n - 69) / 12) of the MIDI note that plays
    the sample at its recorded speed. Every Field-written value is such a frequency: 261.6256 (C4) and 523.2511 (C5)
    in the factory presets, 440.0 (A4) when the Field imports audio without a key held. Other keys play chromatically
    at rate f(note) / base_freq."""
    return 440.0 * 2.0 ** ((note_number(note) - 69) / 12.0)


def note_from_base_freq(base_freq: float) -> float:
    """MIDI note (fractional) whose equal-tempered frequency is base_freq: 69 + 12 * log2(f / 440)."""
    if base_freq <= 0:
        raise ValueError("base_freq must be positive")
    return 69.0 + 12.0 * math.log2(base_freq / 440.0)


# --------------------------------------------------------------------------- reading what the Field wrote

def _pcm(preset: PresetFile):
    """The SSND audio as float32 (frames, channels) in -1..1, and the sample rate from COMM."""
    import numpy as np
    ch, frames, _ = P._comm_fields(preset)
    body = dict(preset.chunks).get(b"SSND", b"")
    data = np.frombuffer(body[8:], dtype="<i2")
    if ch > 0:
        data = data[: (len(data) // ch) * ch].reshape(-1, ch)
    return data.astype(np.float32) / 32768.0


def _peak_dbfs(x) -> float:
    import numpy as np
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return round(20.0 * math.log10(peak), 2) if peak > 0 else -120.0


def sampler_region(preset: PresetFile) -> dict[str, Any]:
    """Decode a sampler preset: the four markers in seconds and frames, the shift page, the root note and the audio."""
    m = preset.meta
    if m.get("type") != "sampler":
        raise ValueError(f"not a sampler preset (type {m.get('type')!r})")
    ch, frames, bits = P._comm_fields(preset)
    knobs = list(m.get("knobs", [0] * 8)) + [0] * 8
    start, loop_in, loop_out, end = knobs[:4]
    direction, fine_tune, loop_fade, gain = knobs[4:8]
    bf = float(m.get("base_freq", 440.0))
    midi = note_from_base_freq(bf)
    out: dict[str, Any] = {
        "name": m.get("name"), "channels": ch, "stereo": bool(m.get("stereo", ch == 2)), "frames": frames,
        "seconds": round(frames / SAMPLER_SAMPLERATE, 4), "bits": bits,
        "start_s": round(sampler_knob_to_seconds(start), 4), "loop_in_s": round(sampler_knob_to_seconds(loop_in), 4),
        "loop_out_s": round(sampler_knob_to_seconds(loop_out), 4), "end_s": round(sampler_knob_to_seconds(end), 4),
        "start_frame": round(sampler_knob_to_frame(start)), "loop_in_frame": round(sampler_knob_to_frame(loop_in)),
        "loop_out_frame": round(sampler_knob_to_frame(loop_out)), "end_frame": round(sampler_knob_to_frame(end)),
        "knobs": knobs[:4], "knobs_shift": knobs[4:8],
        "loops": loop_out > loop_in,
        "direction": "forward" if direction < 16384 else "reverse (inferred, unverified)",
        "fine_tune": round(fine_tune / RAW_MAX, 4), "loop_fade": round(loop_fade / RAW_MAX, 4),
        "gain": round(gain / SAMPLER_GAIN_UNITY, 4),
        "base_freq": bf, "root_midi": round(midi, 3), "root_note": note_name(round(midi)),
        "octave": m.get("octave", 0), "fade": m.get("fade"),
    }
    if frames and b"SSND" in dict(preset.chunks):
        out["peak_dbfs"] = _peak_dbfs(_pcm(preset))
    return out


def check_sampler_preset(preset: PresetFile) -> list[str]:
    """Problems that would make a sampler preset unlike anything the Field writes; empty when it matches the spec.

    The Field validates values per engine and demotes a preset to a plain sample when something is out of range
    (docs/device.md round five), so this is the guard to run before a file is staged."""
    m = preset.meta
    problems: list[str] = []
    if m.get("type") != "sampler":
        return [f"type {m.get('type')!r} is not sampler"]
    ch, frames, bits = P._comm_fields(preset)
    if bits != 16:
        problems.append(f"COMM bits {bits}, the Field writes 16")
    if ch not in (1, 2):
        problems.append(f"COMM channels {ch}, the Field writes 1 or 2")
    if frames == 0:
        problems.append("no sample frames")
    if frames > SAMPLER_SPAN_FRAMES + 8:
        problems.append(f"{frames} frames ({frames / SAMPLER_SAMPLERATE:.2f} s); the region knobs address at most {SAMPLER_SPAN_FRAMES} (6 s)")
    for cid, body in preset.chunks:
        if cid == b"COMM" and (len(body) != 64 or body[18:22] != b"sowt"):
            problems.append("COMM must be the Field's 64-byte 'sowt' layout (presets._comm)")
    if m.get("synth_version") != 3:
        problems.append(f"synth_version {m.get('synth_version')!r}, the Field writes 3")
    if not isinstance(m.get("stereo"), bool) or (ch in (1, 2) and m.get("stereo") != (ch == 2)):
        problems.append(f"stereo must be a boolean equal to (channels == 2); channels {ch}, stereo {m.get('stereo')!r}")
    bf = m.get("base_freq")
    if not isinstance(bf, (int, float)) or isinstance(bf, bool) or not 8.0 <= float(bf) <= 13000.0:
        problems.append(f"base_freq {bf!r} must be a frequency in Hz (the Field writes 261.6256, 440.0, 523.2511)")
    if "fade" in m and not (isinstance(m["fade"], int) and m["fade"] >= 0):
        problems.append(f"fade {m['fade']!r} must be a non-negative integer (0 is what the Field's import writer puts)")
    if not (isinstance(m.get("octave"), int) and -4 <= m["octave"] <= 4):
        problems.append(f"octave {m.get('octave')!r} outside -4..4")
    name = m.get("name")
    if not isinstance(name, str) or not name:
        problems.append("name must be a non-empty string")
    elif len(name) > SAMPLER_NAME_MAX:
        problems.append(f"name {name!r} is longer than {SAMPLER_NAME_MAX} characters, the longest the Field has written")
    elif P._NAME_DISALLOWED.search(name):
        problems.append(f"name {name!r} uses characters the Field has never written (only lowercase letters, digits, space, '-' and '#')")
    knobs = m.get("knobs")
    if not (isinstance(knobs, list) and len(knobs) == 8 and all(isinstance(v, int) and not isinstance(v, bool) for v in knobs)):
        problems.append("knobs must be 8 integers")
    else:
        s, li, lo, e = knobs[:4]
        if not (0 <= s <= li <= lo <= e <= SAMPLER_POSITION_MAX):
            problems.append(f"region {knobs[:4]} must satisfy 0 <= start <= loop in <= loop out <= end <= 32767")
        elif frames and e > sampler_frame_to_knob(frames, end=True) + 1:
            problems.append(f"end {e} is {sampler_knob_to_seconds(e):.4f} s but the sample is {frames / SAMPLER_SAMPLERATE:.4f} s long")
        if not 0 <= knobs[4] <= RAW_MAX:
            problems.append(f"direction {knobs[4]} outside 0..32767")
        if not -RAW_MAX <= knobs[5] <= RAW_MAX:
            problems.append(f"fine tune {knobs[5]} outside -32767..32767")
        if not 0 <= knobs[6] <= RAW_MAX:
            problems.append(f"loop fade {knobs[6]} outside 0..32767")
        if not 0 <= knobs[7] <= RAW_MAX:
            problems.append(f"gain {knobs[7]} outside 0..32767")
    adsr = m.get("adsr")
    if not (isinstance(adsr, list) and len(adsr) == 8 and all(isinstance(v, int) and 0 <= v <= RAW_MAX for v in adsr)):
        problems.append("adsr must be 8 integers 0..32767")
    for block, types in (("fx", P.FX_TYPES), ("lfo", P.LFO_TYPES)):
        if not (isinstance(m.get(f"{block}_params"), list) and len(m[f"{block}_params"]) == 8 and all(isinstance(v, int) for v in m[f"{block}_params"])):
            problems.append(f"{block}_params must hold 8 integers")
        if m.get(f"{block}_type") not in types:
            problems.append(f"{block}_type {m.get(f'{block}_type')!r} unknown; known: {sorted(types)}")
        if m.get(f"{block}_active") not in (True, False, 0, 1):
            problems.append(f"{block}_active must be a boolean")
    return problems


# --------------------------------------------------------------------------- templates

def _is_field_import(meta: dict[str, Any]) -> bool:
    """True for a file the Field's own import writer produced (audio dropped into a slot, no key held)."""
    k = meta.get("knobs") or []
    return (meta.get("type") == "sampler" and len(k) == 8 and k[4:8] == [SAMPLER_DIRECTION_FORWARD, SAMPLER_FINE_TUNE_CENTRE, SAMPLER_LOOP_FADE_OFF, SAMPLER_GAIN_UNITY]
            and k[0] == k[1] == 0 and k[2] == k[3] and meta.get("adsr") == SAMPLER_IMPORT_ADSR and meta.get("fx_type") == "delay"
            and meta.get("lfo_type") == "tremolo" and not meta.get("fx_active") and not meta.get("lfo_active"))


def sampler_template(home: str | None = None) -> tuple[dict[str, Any], str]:
    """Metadata of a sampler preset the Field wrote, and where it came from, to start authoring from.

    Prefers a file the Field's own import writer produced (shift page [12000, 0, 0, 8192], delay off, tremolo off,
    adsr [64, 10746, 32767, 10000, ...]): that is what the Field itself writes when it turns audio into a sampler
    slot, so it is the neutral starting point. Otherwise any Field-written sampler preset (presets.factory_example),
    whose shift page, FX and LFO then carry that preset's own settings. Otherwise the same import values as a literal
    (SAMPLER_IMPORT_META), which the tests hold equal to a Field-written file."""
    home = _default_home(home)
    root = os.path.join(home, "field-backup")
    if os.path.isdir(root):
        for d in sorted(os.listdir(root), reverse=True):
            for sub in ("synth/user", "synth/snapshot", "synth"):
                dd = os.path.join(root, d, sub)
                if not os.path.isdir(dd):
                    continue
                for f in sorted(os.listdir(dd)):
                    if not f.endswith(".aif") or f.startswith("."):
                        continue
                    try:
                        pf = P.read_preset(os.path.join(dd, f))
                    except Exception:
                        continue
                    if _is_field_import(pf.meta):
                        return dict(pf.meta), os.path.join(dd, f)
        src = P.factory_example(home, "sampler")
        if src:
            return dict(P.read_preset(src).meta), src
    return {k: (list(v) if isinstance(v, list) else v) for k, v in SAMPLER_IMPORT_META.items()}, "SAMPLER_IMPORT_META (the Field's import defaults; no Field-written sampler file in the backups)"


# --------------------------------------------------------------------------- audio

def load_sample_audio(path: str, samplerate: int = SAMPLER_SAMPLERATE, max_frames: int | None = None) -> tuple[Any, dict[str, Any]]:
    """Read an audio file as float32 (frames, channels) at `samplerate`, resampling when needed; mono stays mono,
    stereo stays stereo, more channels keep the first two. With `max_frames`, reads only what fits."""
    import numpy as np
    import soundfile as sf
    from math import gcd
    info = sf.info(path)
    sr = int(info.samplerate)
    want = None
    if max_frames is not None:
        # a little extra so the polyphase filter's edge does not shorten the cut
        want = int(math.ceil(max_frames * sr / samplerate)) + 64
    x, sr = sf.read(path, dtype="float32", always_2d=True, frames=-1 if want is None else want)
    notes: dict[str, Any] = {"source": path, "source_samplerate": sr, "source_channels": int(x.shape[1]), "source_frames": int(info.frames),
                             "source_seconds": round(info.frames / sr, 4)}
    if x.shape[1] > 2:
        x = x[:, :2]
        notes["channels_kept"] = "first two of %d" % info.channels
    if sr != samplerate:
        from scipy.signal import resample_poly
        g = gcd(sr, samplerate)
        x = resample_poly(x, samplerate // g, sr // g, axis=0).astype(np.float32)
        notes["resampled"] = f"{sr} -> {samplerate} Hz"
    if max_frames is not None and len(x) > max_frames:
        x = x[:max_frames]
    return np.ascontiguousarray(x, dtype=np.float32), notes


def _aifc_chunks(pcm, samplerate: int = SAMPLER_SAMPLERATE) -> list[tuple[bytes, bytes]]:
    """FVER, COMM and SSND exactly as the Field lays them out (presets.silent_template does the same for synths)."""
    import numpy as np
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")   # 'sowt' = little-endian 16-bit
    fver = struct.pack(">I", 0xA2805140)
    comm = P._comm(int(pcm16.shape[1]), int(len(pcm16)), samplerate)
    ssnd = struct.pack(">II", 0, 0) + pcm16.tobytes()
    return [(b"FVER", fver), (b"COMM", comm), (b"SSND", ssnd)]


# --------------------------------------------------------------------------- authoring

def make_sampler_preset(wav_path: str, name: str, root_note: int | str = "C4", start_s: float | None = None, loop_in_s: float | None = None,
                        loop_out_s: float | None = None, end_s: float | None = None, loop: bool = True, direction: str = "forward",
                        gain: float | None = None, fine_tune: float = 0.0, loop_fade: float | None = None, octave: int = 0,
                        fx: str | None = None, lfo: str | None = None, fx_active: bool | None = None, lfo_active: bool | None = None,
                        adsr: list[float] | None = None, root_hz: float | None = None, truncate: bool = False,
                        catalog_path: str | None = None, home: str | None = None) -> tuple[PresetFile, dict[str, Any]]:
    """Build a synth sampler preset from an audio file on this computer. Returns (preset, what was derived).

    The audio is resampled to 44.1 kHz and stored as 16-bit; mono stays mono, stereo stays stereo. The synth
    sampler holds 6 s at most (264,600 frames): longer audio raises ValueError unless `truncate` is set (cut
    the take first with the sample_from_take tool, trim_take here). Every value that is not the region, the name
    or the root comes from a file the Field wrote (sampler_template), because the Field demotes presets with
    out-of-range values to samples. The Field plays the sample at its recorded level: a quiet cut (peak below
    about -6 dBFS) is reported under notes["level"], since an authored preset once came out 16 dB quiet that way.

    root_note: the key (name like C4, or MIDI number) that plays the sample at its recorded speed; base_freq is
      that note's equal-tempered frequency. `root_hz` overrides it with a measured pitch so that key sounds in
      tune (unverified: every Field-written base_freq is a note frequency).
    start_s, loop_in_s, loop_out_s, end_s: region in seconds on the sample; defaults are the whole sample with
      the whole sample looping (the Field's import default). loop=False puts loop in and loop out on the end
      marker (the file format shows no loop switch; unverified).
    direction: "forward" (12000) or "reverse" (24576, unverified).  fine_tune: -1..1 of the encoder (0 = centre;
      scale unknown).  loop_fade: 0..1 of the encoder (None keeps the template's, 0 for the import default).
    gain: linear multiplier, 1.0 = unity = 8192 (None keeps the template's; curve above unity unverified).
    fx / lfo: a type from the catalog; its parameters are the last Field-written example. fx_active / lfo_active
      default to on when a type is requested, otherwise the template's flag (off for the import default).
    adsr: attack, decay, sustain, release as 0..1 fractions within the ranges the Field has written.
    """
    import numpy as np
    home = _default_home(home)
    catalog_path = catalog_path or os.path.join(home, "engine-catalog.json")
    if direction not in ("forward", "reverse"):
        raise ValueError(f"direction must be 'forward' or 'reverse', not {direction!r}")
    if fx is not None and fx not in P.FX_TYPES:
        raise ValueError(f"unknown fx {fx!r}; known: {sorted(P.FX_TYPES)}")
    if lfo is not None and lfo not in P.LFO_TYPES:
        raise ValueError(f"unknown lfo {lfo!r}; known: {sorted(P.LFO_TYPES)}")
    if not loop and (loop_in_s is not None or loop_out_s is not None):
        raise ValueError("loop=False and loop points given; drop one or the other")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    notes: dict[str, Any] = {}
    unverified: list[str] = []

    # audio
    import soundfile as sf
    info = sf.info(wav_path)
    src_seconds = info.frames / info.samplerate
    est_frames = info.frames * SAMPLER_SAMPLERATE / info.samplerate
    # a few frames over 6 s are cut silently (the factory sample "suitcase" is 264604 frames, its end clamped at 32767);
    # anything longer than about one knob unit over is refused unless the caller asks for the cut
    if est_frames > SAMPLER_SPAN_FRAMES + SAMPLER_UNIT_FRAMES and not truncate:
        raise ValueError(f"{os.path.basename(wav_path)} is {src_seconds:.2f} s; the synth sampler holds {SAMPLER_MAX_SECONDS:.0f} s "
                         f"({SAMPLER_SPAN_FRAMES} frames at 44.1 kHz). Cut it with sample_from_take or pass truncate=True")
    pcm, audio_notes = load_sample_audio(wav_path, SAMPLER_SAMPLERATE, max_frames=SAMPLER_SPAN_FRAMES)
    frames, channels = int(len(pcm)), int(pcm.shape[1])
    if frames == 0:
        raise ValueError(f"{wav_path} holds no audio")
    audio_notes.update(frames=frames, seconds=round(frames / SAMPLER_SAMPLERATE, 4), channels=channels, peak_dbfs=_peak_dbfs(pcm),
                       truncated=bool(est_frames > SAMPLER_SPAN_FRAMES + 0.5))
    if not np.any(pcm):
        audio_notes["warning"] = "the sample is silent"
    notes["audio"] = audio_notes
    if np.any(pcm) and audio_notes["peak_dbfs"] < SAMPLER_QUIET_DBFS and (gain is None or float(gain) <= 1.0):
        notes["level"] = (f"the sample peaks at {audio_notes['peak_dbfs']} dBFS and the Field plays it at that level (an authored "
                          f"preset recorded at -22 dBFS came out about 16 dB quiet): normalize the cut (sample_from_take normalize_db=-1.0) "
                          f"or pass gain above 1.0")

    # template: a file the Field wrote
    meta, template_from = sampler_template(home)
    notes["template_from"] = template_from
    meta.pop("original_folder", None)
    meta["type"] = "sampler"
    meta["synth_version"] = 3
    name, name_note = P.normalize_preset_name(name)
    if name_note:
        notes["name_truncated"] = name_note
    meta["name"] = name
    meta["octave"] = int(max(-4, min(4, octave)))
    meta["stereo"] = channels == 2
    meta["fade"] = SAMPLER_FADE_DEFAULT

    # root
    midi = note_number(root_note)
    if root_hz is not None:
        if root_hz <= 0:
            raise ValueError("root_hz must be positive")
        base_freq = float(root_hz)
        unverified.append(f"base_freq {root_hz} is a measured pitch, not a note frequency; every Field-written value is a note frequency")
    else:
        base_freq = base_freq_from_note(midi)
    meta["base_freq"] = round(base_freq, 6)
    notes["root"] = {"note": note_name(midi), "midi": midi, "base_freq": meta["base_freq"]}

    # region
    if loop:
        region = sampler_positions(start_s or 0.0, loop_in_s, loop_out_s, end_s, frames=frames)
    else:
        region = sampler_positions(start_s or 0.0, None, None, end_s, frames=frames)
        region["loop_in"] = region["loop_out"] = region["end"]
        unverified.append("loop=False is written as loop in = loop out = end; the files show no loop switch")
    knobs = list(meta.get("knobs") or SAMPLER_IMPORT_META["knobs"])[:8]
    knobs += [0] * (8 - len(knobs))
    knobs[0], knobs[1], knobs[2], knobs[3] = region["start"], region["loop_in"], region["loop_out"], region["end"]
    notes["region"] = {k: {"knob": v, "seconds": round(sampler_knob_to_seconds(v), 4), "frame": round(sampler_knob_to_frame(v))} for k, v in region.items()}
    notes["region"]["unit"] = f"knob = seconds / {SAMPLER_MAX_SECONDS:.0f} * {SAMPLER_POSITION_MAX} ({SAMPLER_UNIT_SECONDS * 1000:.4f} ms per unit), fixed 6 s span"

    # shift page
    if direction == "forward":
        knobs[4] = knobs[4] if knobs[4] in SAMPLER_DIRECTION_FORWARD_VALUES else SAMPLER_DIRECTION_FORWARD
    else:
        knobs[4] = SAMPLER_DIRECTION_REVERSE
        unverified.append(f"reverse direction written as {SAMPLER_DIRECTION_REVERSE}; no Field-written file plays reversed")
    knobs[5] = P.pct_to_raw(float(fine_tune), bipolar=True)
    if knobs[5] != 0:
        unverified.append("fine tune scale (cents per unit) is unknown; 0 is the Field's default")
    if loop_fade is not None:
        knobs[6] = P.pct_to_raw(float(loop_fade))
    if gain is not None:
        if gain < 0:
            raise ValueError("gain must be a non-negative multiplier (1.0 = unity)")
        knobs[7] = int(max(0, min(RAW_MAX, round(float(gain) * SAMPLER_GAIN_UNITY))))
        if knobs[7] != SAMPLER_GAIN_UNITY:
            unverified.append("gain curve is unverified beyond unity (8192); linear v/8192 is the hypothesis")
    meta["knobs"] = [int(v) for v in knobs]
    notes["shift"] = {"direction": knobs[4], "fine_tune": knobs[5], "loop_fade": knobs[6], "gain": knobs[7],
                      "gain_linear": round(knobs[7] / SAMPLER_GAIN_UNITY, 3)}

    # envelope
    adsr_raw = list(meta.get("adsr") or SAMPLER_IMPORT_ADSR)[:8]
    adsr_raw += [0] * (8 - len(adsr_raw))
    if adsr is not None:
        rng = P.adsr_ranges(catalog_path) or [(64, 16320), (64, 16320), (0, 32767), (64, 16320)]
        for i, frac in enumerate(list(adsr)[:4]):
            lo, hi = rng[i]
            adsr_raw[i] = int(round(lo + (hi - lo) * max(0.0, min(1.0, float(frac)))))
        notes["adsr_ranges_used"] = [list(r) for r in rng[:4]]
    meta["adsr"] = [int(v) for v in adsr_raw]

    # fx and lfo blocks from Field-written examples
    if fx is not None and fx != meta.get("fx_type"):
        ex = P.catalog_examples(catalog_path, "fx", fx)
        if not ex:
            raise ValueError(f"no Field-written example of FX {fx!r}; known: {P.known_engine_ids(catalog_path)['fx']}")
        meta["fx_type"] = fx
        meta["fx_params"] = [int(v) for v in ex[-1]["params"]]
        notes["fx_from"] = ex[-1]["from"]
    meta["fx_active"] = bool(fx_active) if fx_active is not None else (True if fx is not None else bool(meta.get("fx_active", False)))
    if lfo is not None and lfo != meta.get("lfo_type"):
        ex = P.catalog_examples(catalog_path, "lfo", lfo)
        if not ex:
            raise ValueError(f"no Field-written example of LFO {lfo!r}; known: {P.known_engine_ids(catalog_path)['lfo']}")
        meta["lfo_type"] = lfo
        meta["lfo_params"] = [int(v) for v in ex[-1]["params"]]
        notes["lfo_from"] = ex[-1]["from"]
    meta["lfo_active"] = bool(lfo_active) if lfo_active is not None else (True if lfo is not None else bool(meta.get("lfo_active", False)))
    meta["fx_params"] = [int(v) for v in (list(meta.get("fx_params") or [8000] * 8) + [0] * 8)[:8]]
    meta["lfo_params"] = [int(v) for v in (list(meta.get("lfo_params") or [0] * 8) + [0] * 8)[:8]]
    meta["mtime"] = float(int(time.time()))
    if unverified:
        notes["unverified"] = unverified

    preset = PresetFile(form_type=b"AIFC", chunks=_aifc_chunks(pcm), meta=meta)
    problems = check_sampler_preset(preset)
    if problems:
        raise ValueError("the composed preset does not match the Field's sampler files: " + "; ".join(problems))
    return preset, notes


# --------------------------------------------------------------------------- cutting a region out of a take

def trim_take(wav_in: str, out_path: str, start_s: float, end_s: float | None = None, fade_ms: float = 5.0, normalize_db: float | None = None,
              subtype: str = "PCM_24") -> dict[str, Any]:
    """Cut [start_s, end_s) out of a recorded take, fade both ends, optionally peak-normalize, and write a WAV.

    The take is what the bridge records from the Field (stereo float WAV); any WAV soundfile reads works. The cut
    keeps the take's sample rate and channels; make_sampler_preset resamples later. Fades are linear ramps of
    `fade_ms` at each end (shortened when the region is shorter than two fades), so the sample starts and ends at
    zero and loops without a click at its boundaries. normalize_db is a peak target in dBFS (e.g. -1.0); None
    keeps the level. subtype is a soundfile WAV subtype: PCM_24 (default), FLOAT or PCM_16. Returns what was
    written: frames, seconds, peak before and after."""
    import numpy as np
    import soundfile as sf
    if subtype not in ("PCM_24", "FLOAT", "PCM_16"):
        raise ValueError("subtype must be PCM_24, FLOAT or PCM_16")
    info = sf.info(wav_in)
    sr = int(info.samplerate)
    total = int(info.frames)
    if start_s < 0:
        raise ValueError("start must be at or after 0 s")
    start_f = int(round(start_s * sr))
    if start_f >= total:
        raise ValueError(f"start {start_s:.3f} s is past the end of the take ({total / sr:.3f} s)")
    if end_s is None:
        end_f = total
    else:
        if end_s <= start_s:
            raise ValueError("end must be after start")
        end_f = min(total, int(round(end_s * sr)))
    x, _ = sf.read(wav_in, dtype="float32", always_2d=True, start=start_f, stop=end_f)
    n = int(len(x))
    if n == 0:
        raise ValueError("the region holds no audio")
    peak_before = _peak_dbfs(x)
    fade_frames = min(int(round(fade_ms / 1000.0 * sr)), n // 2)
    if fade_frames > 0:
        ramp = np.linspace(0.0, 1.0, fade_frames, endpoint=False, dtype=np.float32)[:, None]
        x[:fade_frames] *= ramp
        x[n - fade_frames:] *= ramp[::-1]
    if normalize_db is not None:
        peak = float(np.max(np.abs(x)))
        if peak > 0:
            x *= (10.0 ** (normalize_db / 20.0)) / peak
    if subtype != "FLOAT":
        x = np.clip(x, -1.0, 1.0)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    sf.write(out_path, x, sr, subtype=subtype)
    return {"path": out_path, "frames": n, "seconds": round(n / sr, 4), "samplerate": sr, "channels": int(x.shape[1]),
            "start_s": round(start_f / sr, 4), "end_s": round(end_f / sr, 4), "fade_ms": round(fade_frames / sr * 1000.0, 3),
            "peak_dbfs_before": peak_before, "peak_dbfs": _peak_dbfs(x), "normalized_to_db": normalize_db, "subtype": subtype,
            "fits_sampler": n / sr <= SAMPLER_MAX_SECONDS}
