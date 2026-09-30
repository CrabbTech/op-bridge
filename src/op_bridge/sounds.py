"""Sound profiles, classification and a persistent sound index.

A profile says what a slot can play (engine, root pitch for samplers, a playable and an ideal MIDI range).
A classification turns a short audition into features and tags a model can reason about without listening.
The index remembers every sound audited so nothing is auditioned twice and searches return the few best picks.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from . import analysis as an
from . import presets as P

AUDIBLE_LOW, AUDIBLE_HIGH = 24, 96          # C1 (33 Hz) to C7 (2.1 kHz) fundamentals: the useful musical range
SAMPLER_SPAN = 12                            # a sample keeps its character within an octave of its root
SAMPLER_IDEAL = 7


def midi_from_hz(hz: float) -> int:
    return int(round(69 + 12 * math.log2(hz / 440.0)))


def note_name(m: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"


@dataclass
class SlotProfile:
    kind: str                      # synth or drum
    slot: int
    engine: str = ""
    name: str = ""
    root_midi: int | None = None   # samplers: the note that plays the sample at its recorded speed
    octave: int = 0
    playable: tuple[int, int] = (AUDIBLE_LOW, AUDIBLE_HIGH)
    ideal: tuple[int, int] = (36, 84)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["playable_names"] = [note_name(self.playable[0]), note_name(self.playable[1])]
        d["ideal_names"] = [note_name(self.ideal[0]), note_name(self.ideal[1])]
        if self.root_midi is not None:
            d["root"] = note_name(self.root_midi)
        return d


def profile_from_file(path: str, kind: str, slot: int) -> SlotProfile:
    pf = P.read_preset(path)
    m = pf.meta
    prof = SlotProfile(kind=kind, slot=slot, engine=pf.engine, name=pf.name, octave=int(m.get("octave", 0) or 0), source=f"disk backup file {os.path.basename(path)}; the human may have changed the slot since")
    if kind == "drum":
        prof.playable = (P.DRUM_FIRST_NOTE, P.DRUM_FIRST_NOTE + 23)
        prof.ideal = prof.playable
        return prof
    if pf.engine == "sampler" and m.get("base_freq"):
        root = midi_from_hz(float(m["base_freq"]))
        prof.root_midi = root
        prof.playable = (max(AUDIBLE_LOW, root - SAMPLER_SPAN), min(AUDIBLE_HIGH, root + SAMPLER_SPAN))
        prof.ideal = (max(AUDIBLE_LOW, root - SAMPLER_IDEAL), min(AUDIBLE_HIGH, root + SAMPLER_IDEAL))
        return prof
    # synthesis engines pitch every MIDI note literally; the preset's octave setting hints at its intended register
    lo, hi = AUDIBLE_LOW, AUDIBLE_HIGH
    if prof.octave <= -2:
        prof.ideal = (28, 60)
    elif prof.octave == -1:
        prof.ideal = (36, 72)
    elif prof.octave >= 2:
        prof.ideal = (60, 96)
    elif prof.octave == 1:
        prof.ideal = (48, 88)
    else:
        prof.ideal = (40, 84)
    prof.playable = (lo, hi)
    return prof


def default_profile(kind: str, slot: int) -> SlotProfile:
    if kind == "drum":
        return SlotProfile(kind="drum", slot=slot, engine="drum", playable=(P.DRUM_FIRST_NOTE, P.DRUM_FIRST_NOTE + 23), ideal=(P.DRUM_FIRST_NOTE, P.DRUM_FIRST_NOTE + 23), source="default")
    return SlotProfile(kind="synth", slot=slot, source="default")


# --------------------------------------------------------------------------- range guarding

def notes_out_of_range(midis: list[int], prof: SlotProfile) -> list[int]:
    lo, hi = prof.playable
    return sorted({m for m in midis if m < lo or m > hi})


def transpose_into_range(midis: list[int], prof: SlotProfile) -> tuple[int, list[str]]:
    """The octave shift (in semitones, multiple of 12) that moves the part into the profile's playable range,
    preferring the ideal range and keeping the part's shape. Returns (shift, notes)."""
    if not midis:
        return 0, []
    lo, hi = prof.playable
    ilo, ihi = prof.ideal
    best, best_cost = 0, None
    for shift in (0, 12, -12, 24, -24, 36, -36):
        moved = [m + shift for m in midis]
        outside = sum(1 for m in moved if m < lo or m > hi)
        off_ideal = sum(1 for m in moved if m < ilo or m > ihi)
        cost = outside * 100 + abs(shift) / 12 * 2.0 + off_ideal * 0.5   # smallest shift that fits; the ideal window only breaks ties
        if best_cost is None or cost < best_cost:
            best, best_cost = shift, cost
    notes = []
    if best:
        notes.append(f"transposed by {best:+d} semitones to fit {prof.kind} {prof.slot} ({prof.engine or 'unknown engine'}, playable {note_name(lo)}-{note_name(hi)}" + (f", root {note_name(prof.root_midi)}" if prof.root_midi is not None else "") + ")")
    still = [m + best for m in midis if not lo <= m + best <= hi]
    if still:
        notes.append(f"{len(still)} notes still outside the playable range after transposing")
    return best, notes


# --------------------------------------------------------------------------- classification

@dataclass
class SoundFeatures:
    peak_db: float
    attack_ms: float | None
    sustain_ratio_db: float           # level at the end of the hold minus the peak (0 = full sustain, very negative = decayed)
    release_ms: float | None
    centroid_hz: float
    centroid_motion: float            # relative spread of the centroid during the hold (movement, LFO, FX)
    level_motion_db: float            # peak-to-peak wobble of the level during the sustained part
    bands: dict[str, float]           # sub, low, mid, high, air energy fractions
    flatness: float
    harmonicity_db: float             # prominence of the played fundamental
    width: float                      # 1 - |L/R correlation|
    pitch_error_cents: float | None   # measured fundamental versus the played note


def measure(audio: np.ndarray, sr: int, on: int, off: int, midi: int) -> SoundFeatures:
    """Features of one held note: `audio` is frames x 2, `on`/`off` the note-on and note-off sample indices."""
    x = audio[:, :2].mean(axis=1) if audio.ndim == 2 else audio
    hold = x[on: off]
    tail = x[off: off + int(3.0 * sr)]
    t, lev = an.envelope_db(hold, sr, 5.0)
    if len(lev) == 0:
        return SoundFeatures(-120, None, 0, None, 0, 0, 0, {}, 0, -60, 0, None)
    pk = float(lev.max()); ipk = int(np.argmax(lev))
    atk = None
    hits = np.nonzero(lev >= pk - 3.0)[0]
    if len(hits):
        atk = round(float(t[hits[0]] * 1000.0), 1)
    late = float(lev[int(len(lev) * 0.8):].mean())
    sustain = round(late - pk, 1)
    rel = None
    tt, tl = an.envelope_db(tail, sr, 10.0)
    if len(tl):
        base = float(tl[0])
        below = np.nonzero(tl < base - 40.0)[0]
        rel = round(float(tt[below[0]] * 1000.0), 0) if len(below) else None
    body = hold[int(0.08 * sr):] if len(hold) > int(0.2 * sr) else hold
    ct, cent = an.spectral_centroid_series(body, sr) if len(body) > 4096 else (None, None)
    centroid = float(np.median(cent)) if cent is not None else 0.0
    motion = float((np.percentile(cent, 90) - np.percentile(cent, 10)) / max(1.0, centroid)) if cent is not None else 0.0
    st, sl = an.envelope_db(body, sr, 50.0)
    level_motion = float(np.percentile(sl, 95) - np.percentile(sl, 5)) if len(sl) > 4 else 0.0
    spec = np.abs(np.fft.rfft(body * np.hanning(len(body)))) ** 2 if len(body) > 1024 else np.zeros(2)
    freqs = np.fft.rfftfreq(len(body), 1 / sr) if len(body) > 1024 else np.zeros(2)
    tot = float(spec.sum()) + 1e-12
    bands = {n: round(float(spec[(freqs >= lo) & (freqs < hi)].sum() / tot), 3) for n, lo, hi in (("sub", 0, 100), ("low", 100, 400), ("mid", 400, 2000), ("high", 2000, 8000), ("air", 8000, 22050))}
    sel = (freqs >= 100) & (freqs <= 12000)
    flat = float(np.exp(np.mean(np.log(spec[sel] + 1e-12))) / (spec[sel].mean() + 1e-12)) if sel.any() else 0.0
    harm = float(an.note_prominence(body, sr, [midi], harmonics=1)[midi]) if len(body) > 2048 else -60.0
    if audio.ndim == 2 and audio.shape[1] >= 2:
        L, R = audio[on: off, 0], audio[on: off, 1]
        corr = float(np.corrcoef(L, R)[0, 1]) if L.std() > 1e-6 and R.std() > 1e-6 else 1.0
        width = round(1.0 - abs(corr), 3)
    else:
        width = 0.0
    err = None
    if harm > 6:
        expected = 440.0 * 2 ** ((midi - 69) / 12)
        band = (freqs > expected * 0.94) & (freqs < expected * 1.06)
        if band.any():
            f0 = float(freqs[band][int(np.argmax(spec[band]))])
            err = round(1200 * math.log2(f0 / expected), 0)
    return SoundFeatures(round(pk, 1), atk, sustain, rel, round(centroid), round(motion, 3), round(level_motion, 1), bands, round(flat, 3), round(harm, 1), width, err)


def tags_for(f: SoundFeatures, midi: int | None = None) -> dict[str, Any]:
    """Plain-language descriptors a model can use, from one note's features."""
    tags: list[str] = []
    b = f.bands or {}
    # register / timbre
    if f.centroid_hz < 300 or b.get("sub", 0) + b.get("low", 0) > 0.7:
        tags.append("dark")
    elif f.centroid_hz < 1200:
        tags.append("warm")
    elif f.centroid_hz < 3500:
        tags.append("bright")
    else:
        tags.append("very bright")
    if b.get("sub", 0) > 0.3:
        tags.append("sub-heavy")
    if b.get("air", 0) > 0.15:
        tags.append("airy")
    # envelope
    if f.attack_ms is not None and f.attack_ms > 120:
        tags.append("slow attack")
    elif f.attack_ms is not None and f.attack_ms < 15:
        tags.append("instant attack")
    if f.sustain_ratio_db > -6:
        tags.append("sustained")
    elif f.sustain_ratio_db > -20:
        tags.append("decaying")
    else:
        tags.append("short")
    if f.release_ms is not None:
        tags.append("long release" if f.release_ms > 900 else "medium release" if f.release_ms > 250 else "short release")
    # texture
    if f.flatness > 0.08 or (f.harmonicity_db < 4 and f.flatness > 0.03):
        tags.append("noisy")
    elif f.harmonicity_db < 8:
        tags.append("inharmonic")
    else:
        tags.append("harmonic")
    if f.centroid_motion > 0.35 or f.level_motion_db > 6:
        tags.append("moving")
    else:
        tags.append("static")
    if f.width > 0.4:
        tags.append("wide stereo")
    if f.pitch_error_cents is not None and abs(f.pitch_error_cents) > 40:
        tags.append(f"detuned {int(f.pitch_error_cents):+d} cents")
    # roles
    roles: list[str] = []
    sustained = f.sustain_ratio_db > -8
    if "sub-heavy" in tags or (f.centroid_hz < 350 and f.harmonicity_db > 8):
        roles.append("bass")
    if ("slow attack" in tags or sustained) and "noisy" not in tags and f.centroid_hz < 3500:
        roles.append("pad")
    if f.attack_ms is not None and f.attack_ms < 25 and -25 < f.sustain_ratio_db < -6 and "harmonic" in tags:
        roles += ["keys", "pluck"]
    if sustained and f.centroid_hz > 800 and "harmonic" in tags:
        roles.append("lead")
    if "moving" in tags and sustained:
        roles.append("drone")
    if "noisy" in tags or "inharmonic" in tags:
        roles.append("fx")
    if f.sustain_ratio_db < -25 and (f.release_ms or 0) < 300:
        roles.append("percussive")
    return {"tags": tags, "roles": sorted(set(roles)) or ["unclassified"]}


# --------------------------------------------------------------------------- the index

def index_path(home: str) -> str:
    return os.path.join(home, "sounds", "index.json")


def load_index(home: str) -> dict[str, Any]:
    p = index_path(home)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {"sounds": {}}


def save_index(home: str, idx: dict[str, Any]) -> str:
    p = index_path(home)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(idx, open(tmp, "w"), indent=1)
    os.replace(tmp, p)
    return p


def record_sound(home: str, sound_id: str, entry: dict[str, Any]) -> str:
    idx = load_index(home)
    entry = dict(entry); entry["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    idx["sounds"][sound_id] = entry
    return save_index(home, idx)


def local_rank(candidates: list[dict[str, Any]], query: str) -> list[tuple[float, dict[str, Any]]]:
    """Rule-based ranking of index entries against a free-text request: term overlap with tags, roles,
    engine and name, with a few synonyms. The optional Jev judge replaces this when configured."""
    q = query.lower()
    syn = {"bass": ["bass", "sub", "sub-heavy", "dark"], "pad": ["pad", "sustained", "slow attack", "wide stereo"], "keys": ["keys", "pluck", "instant attack", "decaying", "harmonic"],
           "rhodes": ["keys", "warm", "decaying"], "lead": ["lead", "bright", "sustained"], "pluck": ["pluck", "short", "instant attack"], "dark": ["dark", "warm"], "bright": ["bright", "very bright", "airy"],
           "warm": ["warm", "dark"], "moving": ["moving", "drone"], "static": ["static"], "noisy": ["noisy", "fx"], "texture": ["drone", "moving", "noisy", "fx"], "drone": ["drone", "sustained", "moving"],
           "percussive": ["percussive", "short"], "sustained": ["sustained", "pad"], "lofi": ["warm", "dark", "keys"], "lo-fi": ["warm", "dark", "keys"], "chill": ["warm", "pad", "keys"]}
    wanted: set[str] = set()
    for w in q.replace(",", " ").split():
        wanted.add(w)
        wanted.update(syn.get(w, []))
    out = []
    for c in candidates:
        words = set(t.lower() for t in c.get("tags", [])) | set(r.lower() for r in c.get("roles", [])) | {str(c.get("engine", "")).lower()} | set(str(c.get("name", "")).lower().split())
        score = sum(1.0 for w in wanted if w in words)
        # penalties for direct contradictions
        if "dark" in wanted and ("very bright" in words or "bright" in words):
            score -= 1.0
        if "bright" in wanted and "dark" in words:
            score -= 1.0
        out.append((score, c))
    out.sort(key=lambda x: -x[0])
    return out


# --------------------------------------------------------------------------- deeper timbre measurements

def harmonic_profile(x: np.ndarray, sr: int, midi: int, max_h: int = 12) -> dict[str, Any]:
    """Harmonic structure at the played pitch: odd/even balance, richness, tilt, inharmonicity."""
    if len(x) < 4096:
        return {}
    n = 1 << 16
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w, n=n)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    levels = []
    harm_energy = 0.0
    for h in range(1, max_h + 1):
        f = f0 * h
        if f >= sr / 2:
            break
        band = (freqs >= f * 0.985) & (freqs <= f * 1.015)
        if not band.any():
            break
        e = float(spec[band].max())
        levels.append(e); harm_energy += float(spec[band].sum())
    if not levels or levels[0] <= 0:
        return {}
    ref = max(levels)
    db = [10 * math.log10(l / ref + 1e-12) for l in levels]
    odd = sum(levels[i] for i in range(0, len(levels), 2))
    even = sum(levels[i] for i in range(1, len(levels), 2))
    odd_even_db = round(10 * math.log10((odd + 1e-12) / (even + 1e-12)), 1)
    rich = sum(1 for d in db if d > -40)
    # tilt: dB per octave of the harmonic peaks (linear fit against log2 of harmonic number)
    xs = np.log2(np.arange(1, len(db) + 1)); ys = np.array(db)
    tilt = float(np.polyfit(xs, ys, 1)[0]) if len(db) > 2 else 0.0
    # inharmonicity: energy between harmonics, within the band up to the last harmonic, relative to harmonic energy
    top = f0 * (len(levels) + 0.5)
    sel = (freqs > f0 * 0.7) & (freqs < top)
    total = float(spec[sel].sum()) + 1e-12
    inharm = round(max(0.0, 1.0 - harm_energy / total), 3)
    return {"odd_even_db": odd_even_db, "harmonics_above_minus40db": rich, "tilt_db_per_octave": round(tilt, 1), "inharmonic_share": inharm}


def modulation_rate(series: np.ndarray, hop_s: float, lo: float = 0.4, hi: float = 14.0) -> tuple[float | None, float]:
    """Dominant periodicity of a level or brightness series, in Hz, and its strength (0-1)."""
    if len(series) < 16:
        return None, 0.0
    # remove the slow trend of the note itself (decay, swell) so only real modulation remains
    n = len(series); xs = np.arange(n)
    trend = np.polyval(np.polyfit(xs, series, 2), xs)
    s = series - trend
    if np.std(s) < 1e-9:
        return None, 0.0
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    freqs = np.fft.rfftfreq(len(s), hop_s)
    lo = max(lo, 2.0 / (n * hop_s))            # at least two full cycles inside the window
    sel = (freqs >= lo) & (freqs <= hi)
    if not sel.any():
        return None, 0.0
    peak = float(spec[sel].max()); i = int(np.argmax(spec[sel]))
    strength = peak / (float(spec[1:].sum()) + 1e-12)
    return round(float(freqs[sel][i]), 2), round(min(1.0, strength * 4), 2)


def measure_deep(audio: np.ndarray, sr: int, on: int, off: int, midi: int) -> dict[str, Any]:
    """The basic features plus harmonic structure, transient, brightness change and modulation. One held note."""
    base = measure(audio, sr, on, off, midi)
    x = audio[:, :2].mean(axis=1) if audio.ndim == 2 else audio
    hold = x[on: off]
    out = dict(base.__dict__)
    if len(hold) < int(0.3 * sr):
        return out
    sustain_part = hold[int(0.25 * sr):] if len(hold) > int(0.5 * sr) else hold[len(hold) // 2:]
    out.update(harmonic_profile(sustain_part, sr, midi))
    # transient: peak level within the first 20 ms versus the peak of the whole hold
    t, lev = an.envelope_db(hold, sr, 2.0)
    if len(lev) > 12:
        out["transient_db"] = round(float(lev[:10].max() - lev.max()), 1)
    # brightness change: first 80 ms versus the sustained part
    early = hold[: int(0.08 * sr)]
    if len(early) > 2048 and len(sustain_part) > 4096:
        _, c1 = an.spectral_centroid_series(early, sr, nperseg=1024)
        _, c2 = an.spectral_centroid_series(sustain_part, sr)
        if len(c1) and len(c2):
            out["brightness_change"] = round(float(np.median(c2) / max(1.0, np.median(c1))), 2)   # >1 opens, <1 closes
    # modulation of level and brightness during the sustain
    st, sl = an.envelope_db(sustain_part, sr, 20.0)
    rate, strength = modulation_rate(np.array(sl), 0.02)
    ct, cent = an.spectral_centroid_series(sustain_part, sr)
    hop = float(ct[1] - ct[0]) if len(ct) > 1 else 0.023
    crate, cstrength = modulation_rate(np.array(cent), hop)
    out["level_mod_hz"] = rate; out["level_mod_strength"] = strength
    out["brightness_mod_hz"] = crate; out["brightness_mod_strength"] = cstrength
    return out


def describe(f: dict[str, Any]) -> list[str]:
    """Musician vocabulary from the deep features. Each word has a stated rule."""
    d: list[str] = []
    cent = f.get("centroid_hz", 0) or 0
    atk = f.get("attack_ms"); sus = f.get("sustain_ratio_db", 0) or 0; rel = f.get("release_ms")
    oe = f.get("odd_even_db"); rich = f.get("harmonics_above_minus40db"); inh = f.get("inharmonic_share", 0) or 0
    flat = f.get("flatness", 0) or 0; bc = f.get("brightness_change"); tr = f.get("transient_db")
    lm, ls = f.get("level_mod_hz"), f.get("level_mod_strength", 0) or 0
    bm, bs = f.get("brightness_mod_hz"), f.get("brightness_mod_strength", 0) or 0
    # envelope
    if atk is not None and atk > 200: d.append("swelling")
    elif atk is not None and atk > 60: d.append("soft attack")
    elif tr is not None and tr > -3 and atk is not None and atk < 15: d.append("plucky")
    if sus > -4: d.append("sustaining")
    elif sus > -18: d.append("decaying")
    else: d.append("short")
    if rel is not None and rel > 1200: d.append("long tail")
    # character
    if flat > 0.08 or (f.get("harmonicity_db", 0) < 4 and flat > 0.03): d.append("breathy" if cent > 2500 else "noisy")
    elif inh > 0.35: d.append("metallic")
    elif inh > 0.18: d.append("bell-like")
    if oe is not None and oe > 6: d.append("hollow")
    elif oe is not None and oe < -3: d.append("even-rich")
    if rich is not None:
        if rich <= 2: d.append("pure")
        elif rich >= 8: d.append("rich")
    if cent < 350: d.append("dark")
    elif cent < 1100: d.append("warm")
    elif cent < 3000: d.append("bright")
    else: d.append("glassy" if (rich or 0) <= 3 else "harsh")
    if bc is not None:
        if bc < 0.6: d.append("filter closes")
        elif bc > 1.6: d.append("filter opens")
    if lm and ls > 0.45: d.append(f"pulsing {lm:g} Hz")
    if bm and bs > 0.45: d.append(f"sweeping {bm:g} Hz")
    if (f.get("width") or 0) > 0.4: d.append("wide")
    if f.get("pitch_error_cents") is not None and abs(f["pitch_error_cents"]) > 40: d.append("detuned")
    return d
