"""Read and write OP-1 field preset files.

A preset is an AIFF-C file with an `APPL` chunk whose payload is the four bytes `op-1`
followed by JSON. The JSON holds the engine type, knob values, envelope, FX and LFO. The
sound data (`SSND`) is a short sample for synth engines and the sample bank for sampler and
drum kits. Values in the JSON are integers, mostly 0 to 32767, some bipolar.

Slots are files: in disk mode the Field mounts as a volume with `synth/user/1.aif` to
`8.aif` and `drum/user/1.aif` to `8.aif`. Writing a file there sets that slot.
"""
from __future__ import annotations

import json
import os
import re
import struct
import time
from dataclasses import dataclass, field, asdict
from typing import Any

RAW_MAX = 32767

# Preset names as the Field writes them: the 27 distinct names in the backups are at most 11 characters and use
# only lowercase letters, digits, space, '-' and '#' ("baby string", "china whist", "2014-268#01", "de4d-mall").
PRESET_NAME_MAX = 11
_NAME_DISALLOWED = re.compile(r"[^a-z0-9 \-#]+")


def normalize_preset_name(name: str) -> tuple[str, str | None]:
    """A preset name shaped like the ones the Field writes: lowercase, only [a-z0-9 -#], single spaces, at most
    PRESET_NAME_MAX characters. Returns (name, note); the note says what changed, None when nothing did.
    Raises ValueError when nothing usable is left."""
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    cleaned = " ".join(_NAME_DISALLOWED.sub(" ", name.lower()).split())
    cleaned = cleaned[:PRESET_NAME_MAX].rstrip()
    if not cleaned:
        raise ValueError(f"name {name!r} has no usable characters; the Field writes lowercase letters, digits, space, '-' and '#'")
    note = None if cleaned == name else (f"{name!r} -> {cleaned!r} (the Field writes at most {PRESET_NAME_MAX} characters: "
                                         "lowercase letters, digits, space, '-' and '#')")
    return cleaned, note

# Engines and what their four encoders do, from TE's user guide (main page, then shifted page).
SYNTH_ENGINES: dict[str, dict[str, list[str]]] = {
    "cluster": {"knobs": ["octave", "detune and ring modulation", "digitalness", "wave number"], "shift": ["wave envelope", "spread", "unitor", ""]},
    "digital": {"knobs": ["wave shaper", "filter", "wave number", "wave modifier"], "shift": ["noise", "", "", ""]},
    "dimension": {"knobs": ["waveform", "modulation", "filter cutoff frequency", "filter resonance"], "shift": ["", "", "", ""]},
    "dna": {"knobs": ["dna filter", "wave number", "wave modifier", "noise"], "shift": ["", "", "", ""]},
    "drwave": {"knobs": ["wave type and length", "filter", "phase", "chorus"], "shift": ["env crossfader", "", "", ""]},
    "dsynth": {"knobs": ["waveform", "envelope", "cross modulation", "frequency"], "shift": ["waveform", "envelope", "filter cutoff frequency", ""]},
    "fm": {"knobs": ["fm amount", "frequency", "topology", "detune"], "shift": ["", "", "", ""]},
    "phase": {"knobs": ["phase shift", "distortion amount", "phase filter", "phase tilt"], "shift": ["", "", "", ""]},
    "pulse": {"knobs": ["filter", "amplitude", "second pulse", "modulation"], "shift": ["", "", "", ""]},
    "string": {"knobs": ["tension", "decay", "detune", "impulse"], "shift": ["", "", "", ""]},
    "voltage": {"knobs": ["modulation", "ground noise", "phase filter", "detune"], "shift": ["", "", "", ""]},
    "vocoder": {"knobs": ["waveform", "formant", "bands", "mix"], "shift": ["", "", "", ""]},
    "amp": {"knobs": ["volume", "compressor", "tone", "overdrive"], "shift": ["", "", "", ""]},
    "sampler": {"knobs": ["start", "loop in", "loop out", "end"], "shift": ["direction", "fine tune", "loop fade", "gain"]},
}

FX_TYPES: dict[str, list[str]] = {
    "cwo": ["frequency", "delay", "feedback", "sideband"],
    "delay": ["range", "speed", "feedback", "level"],
    "fazer": ["rate", "depth", "feedback", "mix"],
    "grid": ["delay x size", "y size", "z feedback", "mix"],
    "mother": ["distance", "gate", "color", "mix"],
    "nitro": ["frequency", "filter follow", "feedback", "frequency"],
    "phone": ["tone", "gsm", "baud", "telemetry"],
    "punch": ["frequency", "rounds", "power", "punch"],
    "spring": ["tone", "turns", "damping", "mix"],
    "terminal": ["rate", "bits", "model", "mix"],
}

LFO_TYPES: dict[str, list[str]] = {
    "random": ["speed", "amount", "destination", "envelope"],
    "element": ["source", "amount", "destination", "parameter"],
    "midi": ["destination 1", "destination 2", "destination 3", "destination 4"],
    "tremolo": ["speed", "pitch amount", "volume level", "pitch envelope"],
    "value": ["speed", "amount", "destination", "parameter"],
    "velocity": ["destination amount", "volume amount", "destination", "parameter"],
}

ENVELOPE = ["attack", "decay", "sustain", "release"]
ENVELOPE_SHIFT = ["play mode", "portamento", "bend range", "volume"]


def pct_to_raw(x: float, bipolar: bool = False) -> int:
    """0..1 (or -1..1 when bipolar) -> integer the Field stores."""
    if bipolar:
        return int(round(max(-1.0, min(1.0, x)) * RAW_MAX))
    return int(round(max(0.0, min(1.0, x)) * RAW_MAX))


def raw_to_pct(v: int) -> float:
    return v / RAW_MAX


def cc_to_raw(cc: int) -> int:
    return int(round(max(0, min(127, cc)) / 127.0 * RAW_MAX))


# --------------------------------------------------------------------------- file format

def _read_chunks(data: bytes) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    if data[:4] != b"FORM":
        raise ValueError("not an AIFF file")
    form_type = data[8:12]
    pos = 12
    chunks: list[tuple[bytes, bytes]] = []
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        chunks.append((cid, data[pos + 8:pos + 8 + size]))
        pos += 8 + size + (size & 1)
    return form_type, chunks


def _write_chunks(form_type: bytes, chunks: list[tuple[bytes, bytes]]) -> bytes:
    body = bytearray(form_type)
    for cid, payload in chunks:
        body += cid + struct.pack(">I", len(payload)) + payload
        if len(payload) & 1:
            body += b"\x00"
    return b"FORM" + struct.pack(">I", len(body)) + bytes(body)


@dataclass
class PresetFile:
    form_type: bytes
    chunks: list[tuple[bytes, bytes]]
    meta: dict[str, Any]

    @property
    def kind(self) -> str:
        return "drum" if self.meta.get("type") in ("drum", "dbox") else "synth"

    @property
    def engine(self) -> str:
        return str(self.meta.get("type"))

    @property
    def name(self) -> str:
        return str(self.meta.get("name", ""))

    def summary(self) -> dict[str, Any]:
        m = self.meta
        out: dict[str, Any] = {"kind": self.kind, "engine": m.get("type"), "name": m.get("name"), "octave": m.get("octave"),
                               "fx": {"type": m.get("fx_type"), "active": m.get("fx_active"), "params": m.get("fx_params", [])[:4]},
                               "lfo": {"type": m.get("lfo_type"), "active": m.get("lfo_active"), "params": m.get("lfo_params", [])[:4]}}
        if self.kind == "synth":
            out["knobs"] = m.get("knobs", [])[:4]
            out["knobs_shift"] = m.get("knobs", [])[4:8]
            out["adsr"] = m.get("adsr", [])[:4]
            out["adsr_shift"] = m.get("adsr", [])[4:8]
        else:
            out["drum_envelope"] = m.get("dyna_env", [])[:4]
            out["keys"] = 24 if "dbox_data" in m or "start" in m else None
            out["first_key"] = f"{drum_key_name(0)} (MIDI {DRUM_FIRST_NOTE})"
            if m.get("type") == "drum" and "start" in m and "end" in m:
                ch, frames, _ = _comm_fields(self)
                out["bank_seconds"] = round(frames / DRUM_SAMPLERATE, 3) if frames else None
                out["keys_used"] = sum(1 for s, e in zip(m["start"], m["end"]) if e > s)
                out["play_modes"] = sorted(set(m.get("playmode", [])))
                out["stacked_ab"] = any(m.get("pan_ab", []))
        return out


def read_preset(path: str) -> PresetFile:
    with open(path, "rb") as f:
        data = f.read()
    form_type, chunks = _read_chunks(data)
    meta = None
    for cid, payload in chunks:
        if cid == b"APPL" and payload[:4] == b"op-1":
            meta = json.loads(payload[4:].decode("utf-8").rstrip("\x00"))
    if meta is None:
        raise ValueError(f"{path}: no op-1 metadata chunk")
    return PresetFile(form_type=form_type, chunks=chunks, meta=meta)


def write_preset(path: str, preset: PresetFile) -> str:
    payload = b"op-1" + json.dumps(preset.meta, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
    if len(payload) & 1:
        payload += b" "
    new_chunks = []
    replaced = False
    for cid, body in preset.chunks:
        if cid == b"APPL" and body[:4] == b"op-1":
            new_chunks.append((cid, payload)); replaced = True
        else:
            new_chunks.append((cid, body))
    if not replaced:
        # put it before SSND, like the Field does
        idx = next((i for i, (cid, _) in enumerate(new_chunks) if cid == b"SSND"), len(new_chunks))
        new_chunks.insert(idx, (b"APPL", payload))
    data = _write_chunks(preset.form_type, new_chunks)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- authoring

def make_synth_preset(template: PresetFile, engine: str, name: str, knobs: list[float] | None = None, knobs_shift: list[float] | None = None,
                      adsr: list[float] | None = None, fx: str | None = None, fx_params: list[float] | None = None, fx_active: bool = True,
                      lfo: str | None = None, lfo_params: list[float] | None = None, lfo_active: bool = True, octave: int = 0) -> PresetFile:
    """Build a synth preset from a template file's audio, replacing the metadata.

    All parameter lists are 0..1 fractions of the encoder range (bipolar ones may be -1..1).
    Missing lists keep the template's values.
    """
    if engine not in SYNTH_ENGINES:
        raise ValueError(f"unknown engine {engine!r}; known: {sorted(SYNTH_ENGINES)}")
    if fx is not None and fx not in FX_TYPES:
        raise ValueError(f"unknown fx {fx!r}; known: {sorted(FX_TYPES)}")
    if lfo is not None and lfo not in LFO_TYPES:
        raise ValueError(f"unknown lfo {lfo!r}; known: {sorted(LFO_TYPES)}")
    meta = dict(template.meta)
    for k in ("base_freq", "fade", "stereo", "original_folder"):
        meta.pop(k, None)
    meta["type"] = engine
    meta["name"], _ = normalize_preset_name(name)
    meta["octave"] = int(max(-4, min(4, octave)))
    meta["synth_version"] = 3
    meta["mtime"] = float(int(time.time()))
    knob_raw = list(meta.get("knobs", [0] * 8))[:8] + [0] * (8 - len(meta.get("knobs", [])))
    if knobs is not None:
        for i, v in enumerate(knobs[:4]):
            knob_raw[i] = pct_to_raw(v, bipolar=v < 0)
    if knobs_shift is not None:
        for i, v in enumerate(knobs_shift[:4]):
            knob_raw[4 + i] = pct_to_raw(v, bipolar=v < 0)
    meta["knobs"] = knob_raw
    adsr_raw = list(meta.get("adsr", [0] * 8))[:8] + [0] * (8 - len(meta.get("adsr", [])))
    if adsr is not None:
        for i, v in enumerate(adsr[:4]):
            adsr_raw[i] = pct_to_raw(v)
    meta["adsr"] = adsr_raw
    if fx is not None:
        meta["fx_type"] = fx
    meta["fx_active"] = bool(fx_active)
    fx_raw = list(meta.get("fx_params", [0] * 8))[:8] + [0] * (8 - len(meta.get("fx_params", [])))
    if fx_params is not None:
        for i, v in enumerate(fx_params[:4]):
            fx_raw[i] = pct_to_raw(v, bipolar=v < 0)
    meta["fx_params"] = fx_raw
    if lfo is not None:
        meta["lfo_type"] = lfo
    meta["lfo_active"] = bool(lfo_active)
    lfo_raw = list(meta.get("lfo_params", [0] * 8))[:8] + [0] * (8 - len(meta.get("lfo_params", [])))
    if lfo_params is not None:
        for i, v in enumerate(lfo_params[:4]):
            lfo_raw[i] = pct_to_raw(v, bipolar=v < 0)
    meta["lfo_params"] = lfo_raw
    return PresetFile(form_type=template.form_type, chunks=list(template.chunks), meta=meta)


def find_field_volume() -> str | None:
    """The mounted OP-1 disk, when the Field is in disk mode."""
    for name in ("OP-1", "OP-1 field", "OP1"):
        p = os.path.join("/Volumes", name)
        if os.path.isdir(os.path.join(p, "synth")) and os.path.isdir(os.path.join(p, "drum")):
            return p
    return None


def slot_path(volume: str, kind: str, slot: int) -> str:
    if kind not in ("synth", "drum") or not 1 <= slot <= 8:
        raise ValueError("kind must be synth or drum and slot 1-8")
    return os.path.join(volume, kind, "user", f"{slot}.aif")


# --------------------------------------------------------------------------- templates

COMPRESSION_NAME = b"Signed integer (little-endian) linear PCM"


def _comm(channels: int, frames: int, samplerate: int) -> bytes:
    """AIFC COMM chunk byte for byte like the Field writes it (64 bytes for the name TE uses)."""
    return struct.pack(">hIh", channels, frames, 16) + _ieee754_extended(samplerate) + b"sowt" + bytes([len(COMPRESSION_NAME)]) + COMPRESSION_NAME


def silent_template(seconds: float = 1.0, samplerate: int = 44100) -> PresetFile:
    """A synth preset skeleton with a short quiet mono sample, mirroring the chunk layout the Field writes."""
    import numpy as np
    frames = 44160 if seconds == 1.0 and samplerate == 44100 else int(seconds * samplerate)
    fver = struct.pack(">I", 0xA2805140)
    comm = _comm(1, frames, samplerate)
    t = np.arange(frames) / samplerate
    wave = (0.2 * np.sin(2 * np.pi * 220.0 * t) * np.exp(-3.0 * t)).astype(np.float32)
    ssnd = struct.pack(">II", 0, 0) + (wave * 32767).astype("<i2").tobytes()
    meta = {"adsr": [64, 10746, 32767, 4000, 4000, 64, 4000, 18021], "fx_active": False, "fx_params": [8000] * 4 + [0] * 4, "fx_type": "delay",
            "knobs": [16384] * 4 + [0] * 4, "lfo_active": False, "lfo_params": [8000] * 4 + [0] * 4, "lfo_type": "tremolo",
            "name": "template", "octave": 0, "synth_version": 3, "type": "cluster"}
    return PresetFile(form_type=b"AIFC", chunks=[(b"FVER", fver), (b"COMM", comm), (b"SSND", ssnd)], meta=meta)


def _ieee754_extended(x: float) -> bytes:
    """80-bit IEEE 754 extended float, as AIFF sample rates are stored."""
    import math
    if x == 0:
        return b"\x00" * 10
    sign = 0
    if x < 0:
        sign = 0x8000; x = -x
    m, e = math.frexp(x)
    exp = e + 16382
    mant = int(m * (1 << 64))
    return struct.pack(">H", sign | exp) + struct.pack(">Q", mant)


# --------------------------------------------------------------------------- drum sampler specification
#
# Learned from the seven `drum` kits and the one `dbox` kit the Field wrote (field-backup/*/drum/user,
# 2026-09-26). Every drum kit is a stereo 16-bit 44.1 kHz bank (the longest factory kit is 19.66 s) with
# 24 per-key arrays, one entry per key from F3 (MIDI 53) upward: start, end, pitch, playmode, reverse,
# volume, pan, pan_ab, attack, fademode. dyna_env[8] is the drum envelope (attack, gain, release, timing,
# then four unused zeros); fx and lfo blocks are the same as for synths. dbox kits replace the per-key
# arrays with dbox_data[24][8] and carry a 22.05 kHz mono placeholder sample.
#
# Positions: start and end count units where 2^31 is the 20 second bank, so unit = 2^31 / 882000 per
# frame. TE's own values are exactly floor(frame * 2^31 / 882000) for integer frames (335 of 336 values
# in the backups; the one exception is a hand-nudged in point). Region boundaries checked against the
# audio fall in the silent 16-frame gaps between hits, and each bank ends 18 frames after the last end.
DRUM_BANK_SECONDS = 20.0
DRUM_SAMPLERATE = 44100
DRUM_BANK_FRAMES = int(DRUM_BANK_SECONDS * DRUM_SAMPLERATE)   # 882000
DRUM_POSITION_SPAN = 1 << 31                                   # positions are 0 .. 2^31-1 = 0 .. 20 s
DRUM_POSITION_UNIT = DRUM_POSITION_SPAN / DRUM_BANK_FRAMES     # about 2434.79 units per frame
DRUM_KEYS = 24
DRUM_FIRST_NOTE = 53                                            # F3; the device answered drum slot 1 on notes 53 to 82
DRUM_KEY_ARRAYS = ("start", "end", "pitch", "playmode", "reverse", "volume", "pan", "pan_ab", "attack", "fademode")
# play mode (orange encoder): a selector stored as the centre of an 8192-wide bin. TE's kits use 12288 on
# nearly every key and 20480 on the C#4/D#4 hi-hat keys of every kit; 4096 appears once. Which mode each
# value plays (one shot, while held, loop) is not visible in the files: verify on the device.
DRUM_PLAYMODE_VALUES = (4096, 12288, 20480)
DRUM_PLAYMODE_DEFAULT = 12288
DRUM_PLAYMODE_HIHAT = 20480
DRUM_DIRECTION_FORWARD = 8192      # `reverse` array; every factory key is forward (one key holds 15353, still below 16384)
DRUM_PAN_CENTRE = 16384            # `pan`: 0 = left (or channel A when stacked) .. 32766 = right (channel B), 1024 per encoder step
DRUM_PAN_MAX = 32766
DRUM_GAIN_UNITY = 8192             # `volume` and dyna_env[1]: 8192 is the factory default; observed 3564 .. 15077 on keys
DRUM_GAIN_MAX_OBSERVED = 15077     # the loudest `volume` in any Field-written kit (about 1.84x unity); louder is untested
DRUM_ENVELOPE = ["attack", "gain", "release", "timing"]
DRUM_ENVELOPE_DEFAULT = [0, 8192, 0, 0, 0, 0, 0, 0]
DBOX_PARAMS = ["pitch", "waveform", "envelope", "cross modulation", "pitch 2", "waveform 2", "envelope 2", "filter cutoff"]


def drum_frame_to_position(frame: int, samplerate: int = DRUM_SAMPLERATE) -> int:
    """Sample frame -> start/end value, the exact integer arithmetic that reproduces TE's files."""
    return (int(frame) * DRUM_POSITION_SPAN) // int(DRUM_BANK_SECONDS * samplerate)


def drum_position_to_frame(value: int, samplerate: int = DRUM_SAMPLERATE) -> float:
    """start/end value -> sample frame (fractional; TE's values sit just below an integer frame)."""
    return value * DRUM_BANK_SECONDS * samplerate / DRUM_POSITION_SPAN


def drum_key_note(index: int) -> int:
    return DRUM_FIRST_NOTE + index


def drum_key_name(index: int) -> str:
    n = drum_key_note(index)
    return ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][n % 12] + str(n // 12 - 1)


def _comm_fields(preset: PresetFile) -> tuple[int, int, int]:
    """(channels, frames, bits) from the COMM chunk; zeros when there is none."""
    for cid, body in preset.chunks:
        if cid == b"COMM" and len(body) >= 8:
            ch, frames, bits = struct.unpack(">hIh", body[:8])
            return ch, frames, bits
    return 0, 0, 0


def drum_kit_regions(preset: PresetFile) -> list[dict[str, Any]]:
    """Decode a `drum` kit's 24 keys into frames and seconds plus the per-key settings, key 0 = F3."""
    m = preset.meta
    if m.get("type") != "drum":
        raise ValueError(f"not a drum sampler kit (type {m.get('type')!r})")
    out = []
    for i in range(DRUM_KEYS):
        s, e = m["start"][i], m["end"][i]
        sf_, ef_ = drum_position_to_frame(s), drum_position_to_frame(e)
        out.append({
            "key": i, "note": drum_key_note(i), "name": drum_key_name(i),
            "start": s, "end": e, "start_frame": round(sf_), "end_frame": round(ef_),
            "seconds": round((ef_ - sf_) / DRUM_SAMPLERATE, 4),
            "playmode": m["playmode"][i], "reverse": m["reverse"][i], "pitch": m["pitch"][i],
            "volume": m["volume"][i], "pan": m["pan"][i], "stacked_ab": bool(m["pan_ab"][i]),
            "attack": m["attack"][i], "fademode": m["fademode"][i],
        })
    return out


def check_drum_kit(preset: PresetFile) -> list[str]:
    """Problems that would make a drum kit unlike anything the Field writes; empty when it matches the spec."""
    m = preset.meta
    problems: list[str] = []
    kind = m.get("type")
    if kind not in ("drum", "dbox"):
        return [f"type {kind!r} is not drum or dbox"]
    ch, frames, bits = _comm_fields(preset)
    if bits != 16:
        problems.append(f"COMM bits {bits}, the Field writes 16")
    if m.get("drum_version") != 2:
        problems.append(f"drum_version {m.get('drum_version')!r}, the Field writes 2")
    env = m.get("dyna_env")
    if not (isinstance(env, list) and len(env) == 8 and all(isinstance(v, int) and 0 <= v <= RAW_MAX for v in env)):
        problems.append("dyna_env must be 8 integers 0..32767 (attack, gain, release, timing, 0, 0, 0, 0)")
    for block in ("fx_params", "lfo_params"):
        if not (isinstance(m.get(block), list) and len(m[block]) == 8):
            problems.append(f"{block} must hold 8 integers")
    if kind == "dbox":
        d = m.get("dbox_data")
        if not (isinstance(d, list) and len(d) == DRUM_KEYS and all(isinstance(r, list) and len(r) == 8 for r in d)):
            problems.append("dbox_data must be 24 keys of 8 integers")
        return problems
    if ch != 2:
        problems.append(f"COMM channels {ch}, drum kits are stereo")
    for arr in DRUM_KEY_ARRAYS:
        if not (isinstance(m.get(arr), list) and len(m[arr]) == DRUM_KEYS):
            problems.append(f"{arr} must have {DRUM_KEYS} entries")
    if problems:
        return problems
    for i in range(DRUM_KEYS):
        s, e = m["start"][i], m["end"][i]
        if not (0 <= s <= e < DRUM_POSITION_SPAN):
            problems.append(f"key {i} ({drum_key_name(i)}): start {s} and end {e} must satisfy 0 <= start <= end < 2^31")
        elif frames and drum_position_to_frame(e) > frames + 1:
            problems.append(f"key {i} ({drum_key_name(i)}): end is frame {drum_position_to_frame(e):.0f} but the bank holds {frames} frames")
        if m["playmode"][i] not in DRUM_PLAYMODE_VALUES:
            problems.append(f"key {i}: playmode {m['playmode'][i]} not one of {DRUM_PLAYMODE_VALUES}")
        if not 0 <= m["reverse"][i] <= RAW_MAX:
            problems.append(f"key {i}: reverse {m['reverse'][i]} outside 0..32767")
        if not 0 <= m["pan"][i] <= RAW_MAX:
            problems.append(f"key {i}: pan {m['pan'][i]} outside 0..32767")
        if not 0 <= m["volume"][i] <= RAW_MAX:
            problems.append(f"key {i}: volume {m['volume'][i]} outside 0..32767")
        if not isinstance(m["pan_ab"][i], bool):
            problems.append(f"key {i}: pan_ab must be a boolean")
        if not -RAW_MAX <= m["pitch"][i] <= RAW_MAX:
            problems.append(f"key {i}: pitch {m['pitch'][i]} outside -32767..32767")
    if frames > DRUM_BANK_FRAMES:
        problems.append(f"bank holds {frames} frames, more than the 20 s ({DRUM_BANK_FRAMES}) a position can address")
    return problems





def check_preset(preset: PresetFile) -> list[str]:
    """Problems for a preset the bridge is about to write: drum kits through check_drum_kit, sampler presets through
    sampler.check_sampler_preset. Other synth engines have no validator (their values are copied from Field-written
    files by compose_synth_preset), so they return no problems."""
    if preset.kind == "drum":
        return check_drum_kit(preset)
    if preset.engine == "sampler":
        from .sampler import check_sampler_preset
        return check_sampler_preset(preset)
    return []


# --------------------------------------------------------------------------- staging and install

def staging_dir(home: str) -> str:
    d = os.path.join(home, "staging")
    for k in ("synth", "drum"):
        os.makedirs(os.path.join(d, k), exist_ok=True)
    return d


def stage_preset(home: str, kind: str, slot: int, preset: PresetFile) -> str:
    if kind not in ("synth", "drum") or not 1 <= slot <= 8:
        raise ValueError("kind must be synth or drum and slot 1-8")
    return write_preset(os.path.join(staging_dir(home), kind, f"{slot}.aif"), preset)


def staged_presets(home: str) -> list[dict[str, Any]]:
    out = []
    d = staging_dir(home)
    for kind in ("synth", "drum"):
        for f in sorted(os.listdir(os.path.join(d, kind))):
            if f.endswith(".aif"):
                p = read_preset(os.path.join(d, kind, f))
                out.append({"kind": kind, "slot": int(f.split(".")[0]), "engine": p.engine, "name": p.name, "path": os.path.join(d, kind, f)})
    return out


def check_staged(home: str) -> list[dict[str, Any]]:
    """Staged files that fail their validator (check_preset), as {kind, slot, path, problems}; empty when all pass."""
    bad = []
    for entry in staged_presets(home):
        problems = check_preset(read_preset(entry["path"]))
        if problems:
            bad.append({**entry, "problems": problems})
    return bad


def install_staged(home: str, volume: str, backup_dir: str | None = None, record_dir: str | None = None) -> dict[str, Any]:
    """Copy staged presets onto the mounted Field disk, backing up what they replace.

    Each file is written under a temporary name, renamed into place and read back before the
    staged copy is removed, so an eject in the middle of the copy never leaves a half-written
    slot or loses the staged file. Failures are reported per slot, not raised. With record_dir, a copy of
    each installed file is kept at record_dir/<kind>/user/<slot>.aif, the layout of a disk copy, so the
    newest local record of a slot is the file now on the Field rather than the one it replaced."""
    import shutil
    installed, backed_up, failed = [], [], []
    d = staging_dir(home)
    for kind in ("synth", "drum"):
        for f in sorted(os.listdir(os.path.join(d, kind))):
            if not f.endswith(".aif"):
                continue
            src = os.path.join(d, kind, f)
            slot = int(f.split(".")[0])
            dst = slot_path(volume, kind, slot)
            tmp = dst + ".part"
            try:
                if backup_dir and os.path.exists(dst):
                    bdir = os.path.join(backup_dir, kind)
                    os.makedirs(bdir, exist_ok=True)
                    shutil.copyfile(dst, os.path.join(bdir, f)); backed_up.append(dst)
                with open(src, "rb") as fi, open(tmp, "wb") as fo:
                    shutil.copyfileobj(fi, fo, 1 << 20)
                    fo.flush(); os.fsync(fo.fileno())
                os.replace(tmp, dst)
                back = read_preset(dst)
                want = read_preset(src)
                if back.meta != want.meta:
                    raise IOError("read-back does not match the staged file")
                for junk in (os.path.join(os.path.dirname(dst), "._" + f), os.path.join(os.path.dirname(dst), "._" + f + ".part")):
                    if os.path.exists(junk):
                        os.remove(junk)
                installed.append({"kind": kind, "slot": slot, "path": dst, "engine": back.engine, "name": back.name})
                if record_dir:
                    rdir = os.path.join(record_dir, kind, "user")
                    os.makedirs(rdir, exist_ok=True)
                    shutil.copyfile(src, os.path.join(rdir, f))
                os.remove(src)
            except Exception as e:  # keep the staged file for a retry
                failed.append({"kind": kind, "slot": slot, "error": f"{type(e).__name__}: {e}"})
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
    return {"installed": installed, "backed_up": backed_up, "failed": failed}


def eject_volume(volume: str) -> bool:
    import subprocess
    r = subprocess.run(["diskutil", "eject", volume], capture_output=True, text=True)
    return r.returncode == 0


# --------------------------------------------------------------------------- learning from the device

def latest_backup_dir(home: str) -> str | None:
    root = os.path.join(home, "field-backup")
    if not os.path.isdir(root):
        return None
    dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    return os.path.join(root, dirs[-1]) if dirs else None


def learn_engines(source: str, catalog_path: str) -> dict[str, Any]:
    """Read every preset under `source` (a mounted Field disk or a backup of one) and record, per
    engine identifier, an example of its parameter values. Merges into the JSON catalog at
    `catalog_path`, which is how the bridge learns the Field's real type strings."""
    catalog: dict[str, Any] = {"synth": {}, "drum": {}, "fx": {}, "lfo": {}}
    if os.path.exists(catalog_path):
        catalog.update(json.load(open(catalog_path)))
    seen = 0
    for sub in ("synth/user", "synth/snapshot", "drum/user"):
        d = os.path.join(source, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith((".aif", ".aiff", ".aifc")) or f.startswith("."):
                continue
            try:
                p = read_preset(os.path.join(d, f))
            except Exception:
                continue
            seen += 1
            m = p.meta
            kind = "drum" if p.kind == "drum" else "synth"
            entry = catalog[kind].setdefault(p.engine, {"examples": []})
            ex = {"name": p.name, "file": f"{sub}/{f}"}
            if kind == "synth":
                ex.update(knobs=m.get("knobs"), adsr=m.get("adsr"), octave=m.get("octave"))
            else:
                ex.update(dyna_env=m.get("dyna_env"))
            if ex not in entry["examples"]:
                entry["examples"] = (entry["examples"] + [ex])[-3:]
            if m.get("fx_type"):
                fx = catalog["fx"].setdefault(m["fx_type"], {"examples": []})
                fx["examples"] = (fx["examples"] + [{"params": m.get("fx_params"), "from": p.name}])[-3:]
            if m.get("lfo_type"):
                lf = catalog["lfo"].setdefault(m["lfo_type"], {"examples": []})
                lf["examples"] = (lf["examples"] + [{"params": m.get("lfo_params"), "from": p.name}])[-3:]
    os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
    json.dump(catalog, open(catalog_path, "w"), indent=1)
    return {"presets_read": seen, "synth_engines": sorted(catalog["synth"]), "drum_engines": sorted(catalog["drum"]), "fx": sorted(catalog["fx"]), "lfo": sorted(catalog["lfo"]), "catalog": catalog_path}


def known_engine_ids(catalog_path: str) -> dict[str, list[str]]:
    if not os.path.exists(catalog_path):
        return {"synth": [], "drum": [], "fx": [], "lfo": []}
    c = json.load(open(catalog_path))
    return {k: sorted(c.get(k, {})) for k in ("synth", "drum", "fx", "lfo")}


def copy_volume(src: str, dst: str) -> int:
    """Copy every preset file from a mounted Field disk (FAT, with flags copytree cannot set). Returns the file count."""
    import shutil
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        rel = os.path.relpath(root, src)
        out = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(out, exist_ok=True)
        for f in files:
            if f.startswith("."):
                continue
            shutil.copyfile(os.path.join(root, f), os.path.join(out, f)); n += 1
    return n


def factory_example(home: str, engine: str) -> str | None:
    """A file the Field itself wrote for this engine, from any backup, to use as the authoring template."""
    root = os.path.join(home, "field-backup")
    if not os.path.isdir(root):
        return None
    for d in sorted(os.listdir(root), reverse=True):
        for sub in ("synth/user", "synth/snapshot", "synth"):
            dd = os.path.join(root, d, sub)
            if not os.path.isdir(dd):
                continue
            for f in sorted(os.listdir(dd)):
                if not f.endswith(".aif"):
                    continue
                try:
                    pf = read_preset(os.path.join(dd, f))
                except Exception:
                    continue
                if pf.kind == "synth" and pf.engine == engine:
                    return os.path.join(dd, f)
    return None


# --------------------------------------------------------------------------- authoring from Field-written values

def catalog_examples(catalog_path: str, section: str, key: str) -> list[dict[str, Any]]:
    if not os.path.exists(catalog_path):
        return []
    c = json.load(open(catalog_path))
    return c.get(section, {}).get(key, {}).get("examples", [])


def knob_ranges(catalog_path: str, engine: str) -> list[tuple[int, int]] | None:
    """Observed [min, max] per knob for an engine, from every example the catalog holds. None when unknown."""
    exs = [e["knobs"] for e in catalog_examples(catalog_path, "synth", engine) if e.get("knobs")]
    if not exs:
        return None
    return [(min(k[i] for k in exs), max(k[i] for k in exs)) for i in range(8)]


def compose_synth_preset(home: str, catalog_path: str, engine: str, name: str, fx: str | None = None, lfo: str | None = None,
                         fx_active: bool = True, lfo_active: bool = True, octave: int | None = None,
                         knobs: list[float] | None = None, adsr: list[float] | None = None) -> tuple[PresetFile, dict[str, Any]]:
    """Build a preset whose every value comes from files the Field wrote.

    The engine block (knobs, envelope, sample) comes from a factory file of `engine`; the FX and LFO
    blocks come from catalog examples of those types. Knob fractions are only applied within the
    range the catalog has observed for that engine; with a single example the knob keeps that
    example's value and the model shapes it live over CC instead."""
    src = factory_example(home, engine)
    if not src:
        raise ValueError(f"no file written by the Field for engine {engine!r} in the backups; load a factory preset of it into a slot, enter disk mode and run learn_engines")
    pf = read_preset(src)
    meta = dict(pf.meta)
    # the Field writes original_folder only on factory and browser items (its own snapshots and imports lack it)
    meta.pop("original_folder", None)
    notes: dict[str, Any] = {"engine_from": os.path.basename(src)}
    meta["name"], name_note = normalize_preset_name(name)
    if name_note:
        notes["name_truncated"] = name_note
    if octave is not None:
        meta["octave"] = int(max(-4, min(4, octave)))
    if fx is not None and fx != meta.get("fx_type"):
        ex = catalog_examples(catalog_path, "fx", fx)
        if not ex:
            raise ValueError(f"no Field-written example of FX {fx!r}; known: {sorted(json.load(open(catalog_path))['fx']) if os.path.exists(catalog_path) else []}")
        meta["fx_type"] = fx; meta["fx_params"] = list(ex[-1]["params"]); notes["fx_from"] = ex[-1]["from"]
    meta["fx_active"] = bool(fx_active)
    if lfo is not None and lfo != meta.get("lfo_type"):
        ex = catalog_examples(catalog_path, "lfo", lfo)
        if not ex:
            raise ValueError(f"no Field-written example of LFO {lfo!r}")
        meta["lfo_type"] = lfo; meta["lfo_params"] = list(ex[-1]["params"]); notes["lfo_from"] = ex[-1]["from"]
    meta["lfo_active"] = bool(lfo_active)
    if knobs is not None and meta.get("type") == "sampler":
        # the sampler's four knobs are start, loop in, loop out and end on this file's own audio: positions, not
        # independent parameters, so ranges from other sample files would write regions the Field never does
        notes["knobs_kept_from_example"] = [1, 2, 3, 4]
        notes["hint"] = ("a sampler's engine knobs are the region on the example's audio, kept as the Field wrote it; "
                         "shape them live with set_parameters (engine1-4), or author a new sample with author_sampler_preset")
    elif knobs is not None:
        rng = knob_ranges(catalog_path, engine) or []
        applied, kept = [], []
        kv = list(meta["knobs"])
        for i, frac in enumerate(knobs[:4]):
            lo, hi = rng[i] if i < len(rng) else (kv[i], kv[i])
            if hi > lo:
                kv[i] = int(round(lo + (hi - lo) * max(0.0, min(1.0, frac)))); applied.append(i + 1)
            else:
                kept.append(i + 1)
        meta["knobs"] = kv
        notes["knobs_applied_within_observed_range"] = applied
        if kept:
            notes["knobs_kept_from_example"] = kept
            notes["hint"] = "only one Field-written example of this engine is known, so those knobs keep its values; shape them live with set_parameters (engine1-4)"
    if adsr is not None:
        # envelope units are shared by every engine; stay inside what the Field has written
        rng = adsr_ranges(catalog_path) or [(64, 16320), (64, 16320), (0, 32767), (64, 16320)]
        av = list(meta["adsr"])
        for i, frac in enumerate(adsr[:4]):
            lo, hi = rng[i]
            av[i] = int(round(lo + (hi - lo) * max(0.0, min(1.0, frac))))
        meta["adsr"] = av
        notes["adsr_ranges_used"] = rng[:4]
    meta["mtime"] = float(int(time.time()))
    out = PresetFile(form_type=pf.form_type, chunks=list(pf.chunks), meta=meta)
    problems = check_preset(out)
    if problems:
        raise ValueError("the composed preset does not match the Field's files: " + "; ".join(problems))
    return out, notes


def adsr_ranges(catalog_path: str) -> list[tuple[int, int]] | None:
    """Observed [min, max] per envelope value across every synth example the catalog holds (the envelope is engine-independent)."""
    if not os.path.exists(catalog_path):
        return None
    c = json.load(open(catalog_path))
    exs = [e["adsr"] for eng in c.get("synth", {}).values() for e in eng.get("examples", []) if e.get("adsr")]
    if not exs:
        return None
    return [(min(a[i] for a in exs), max(a[i] for a in exs)) for i in range(8)]
