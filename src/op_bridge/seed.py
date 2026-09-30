"""Capture a starting point from the player: notes, chords, knob moves and the sound itself."""
from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any

import mido
import numpy as np

from . import analysis as an
from .device import Field, Recorder
from .score import midi_to_name
from .session import SCALES

CHORD_TEMPLATES = {
    "": {0, 4, 7}, "m": {0, 3, 7}, "dim": {0, 3, 6}, "aug": {0, 4, 8}, "sus2": {0, 2, 7}, "sus4": {0, 5, 7},
    "7": {0, 4, 7, 10}, "maj7": {0, 4, 7, 11}, "m7": {0, 3, 7, 10}, "m7b5": {0, 3, 6, 10}, "dim7": {0, 3, 6, 9},
    "6": {0, 4, 7, 9}, "m6": {0, 3, 7, 9}, "add9": {0, 2, 4, 7}, "madd9": {0, 2, 3, 7}, "5": {0, 7},
}
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def name_chord(pitches: list[int]) -> str:
    pcs = sorted({p % 12 for p in pitches})
    if len(pcs) == 1:
        return NAMES[pcs[0]]
    best = None
    for root in pcs:
        rel = {(p - root) % 12 for p in pcs}
        for suffix, tmpl in CHORD_TEMPLATES.items():
            if rel == tmpl:
                score = (len(tmpl) == len(rel), root == min(pitches) % 12)
                if best is None or score > best[0]:
                    best = (score, f"{NAMES[root]}{suffix}")
    if best:
        bass = min(pitches) % 12
        name = best[1]
        if not name.startswith(NAMES[bass]) and len(pcs) >= 3:
            name += f"/{NAMES[bass]}"
        return name
    return "+".join(NAMES[p] for p in pcs)


def guess_key(pitches: list[int]) -> list[dict[str, Any]]:
    """Rank keys by how many of the played pitch classes fit, weighted by occurrence."""
    hist = Counter(p % 12 for p in pitches)
    total = sum(hist.values()) or 1
    ranked = []
    for scale in ("major", "minor", "dorian", "mixolydian", "pentatonic_minor", "pentatonic_major"):
        for root in range(12):
            members = {(root + i) % 12 for i in SCALES[scale]}
            fit = sum(c for pc, c in hist.items() if pc in members) / total
            ranked.append((fit, -len(members), f"{NAMES[root]} {scale}"))
    ranked.sort(reverse=True)
    return [{"key": k, "fit": round(f, 2)} for f, _, k in ranked[:5]]


def guess_tempo(onsets: list[float]) -> dict[str, Any] | None:
    if len(onsets) < 4:
        return None
    iois = np.diff(sorted(onsets))
    iois = iois[(iois > 0.08) & (iois < 2.5)]
    if len(iois) < 3:
        return None
    # fold every interval into the 60-180 bpm range and take the densest cluster
    cands = []
    for i in iois:
        b = 60.0 / i
        while b < 60: b *= 2
        while b > 180: b /= 2
        cands.append(b)
    cands = np.array(cands)
    hist, edges = np.histogram(cands, bins=np.arange(60, 181, 4))
    k = int(np.argmax(hist))
    sel = cands[(cands >= edges[k]) & (cands < edges[k + 1])]
    return {"bpm": round(float(np.median(sel)), 1), "confidence": round(float(hist[k] / len(cands)), 2)}


def capture_seed(field: Field, wait_seconds: float = 60.0, max_seconds: float = 40.0, silence_seconds: float = 4.0, record_audio: bool = True, on_status=None, cancel=None) -> dict[str, Any]:
    """Listen to what the player does on the Field. Returns notes, chords, knob moves, key and tempo guesses, and audio.
    A set `cancel` event ends the listening early (what was heard so far is returned; nothing heard counts as not captured)."""
    if field.midi_in_name is None:
        raise RuntimeError("no OP-1 field MIDI input")
    events: list[dict[str, Any]] = []
    rec: Recorder | None = None
    with mido.open_input(field.midi_in_name) as inp:
        for _ in inp.iter_pending():
            pass
        if record_audio and field.audio is not None:
            rec = field.recorder().start()
        t_start = time.perf_counter()
        first: float | None = None
        last_activity = time.perf_counter()
        while True:
            now = time.perf_counter()
            for m in inp.iter_pending():
                if m.type in ("note_on", "note_off", "control_change", "pitchwheel", "program_change"):
                    if first is None:
                        first = now
                        if rec: rec.mark("first_event")
                        if on_status: on_status("player started")
                    last_activity = now
                    d = {"t": round(now - first, 4), "type": m.type}
                    if m.type in ("note_on", "note_off"):
                        d.update(note=m.note, velocity=m.velocity, channel=m.channel + 1)
                        if m.type == "note_on" and m.velocity == 0:
                            d["type"] = "note_off"
                    elif m.type == "control_change":
                        d.update(cc=m.control, value=m.value, channel=m.channel + 1)
                    elif m.type == "pitchwheel":
                        d.update(bend=m.pitch, channel=m.channel + 1)
                    elif m.type == "program_change":
                        d.update(program=m.program, channel=m.channel + 1)
                    events.append(d)
            if first is None and now - t_start > wait_seconds:
                break
            if first is not None and (now - first > max_seconds or now - last_activity > silence_seconds):
                break
            if cancel is not None and cancel.is_set():
                break
            time.sleep(0.005)
        if rec:
            time.sleep(0.3)
    r = rec.stop() if rec else None
    if first is None:
        return {"captured": False, "reason": f"nothing played within {wait_seconds:.0f} s"}
    # notes with durations
    notes: list[dict[str, Any]] = []
    open_notes: dict[int, dict[str, Any]] = {}
    for e in events:
        if e["type"] == "note_on":
            open_notes[e["note"]] = {"t": e["t"], "pitch": e["note"], "name": midi_to_name(e["note"]), "velocity": e["velocity"]}
        elif e["type"] == "note_off" and e["note"] in open_notes:
            n = open_notes.pop(e["note"]); n["duration"] = round(e["t"] - n["t"], 3); notes.append(n)
    for n in open_notes.values():
        n["duration"] = None; notes.append(n)
    notes.sort(key=lambda n: n["t"])
    # chords: note-ons within 80 ms of each other
    chords: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = []
    for n in notes:
        if group and n["t"] - group[0]["t"] > 0.08:
            chords.append(group); group = []
        group.append(n)
    if group:
        chords.append(group)
    chord_list = [{"t": g[0]["t"], "pitches": [x["pitch"] for x in g], "names": [x["name"] for x in g], "chord": name_chord([x["pitch"] for x in g]) if len(g) > 1 else g[0]["name"], "duration": max((x["duration"] or 0) for x in g)} for g in chords]
    knob_moves = [e for e in events if e["type"] == "control_change"]
    out: dict[str, Any] = {
        "captured": True,
        "seconds": round(events[-1]["t"], 2) if events else 0,
        "note_count": len(notes),
        "notes": notes,
        "chords": chord_list,
        "progression": [c["chord"] for c in chord_list],
        "key_guesses": guess_key([n["pitch"] for n in notes]) if notes else [],
        "tempo_guess": guess_tempo([c["t"] for c in chord_list]),
        "knob_moves": knob_moves,
        "raw_event_count": len(events),
    }
    if r is not None:
        lv = [an.rms_db(r.audio[:, c]) for c in range(r.audio.shape[1])]
        pairs = [(lv[i] + lv[i + 1], i) for i in range(0, r.audio.shape[1] - 1, 2)]
        best = max(pairs)
        if best[0] > -200:
            pair = (best[1] + 1, best[1] + 2)
            x = r.audio[:, pair[0] - 1: pair[1]]
            t, cent = an.spectral_centroid_series(x.mean(axis=1), r.samplerate)
            out["audio"] = {"live_usb_channels": list(pair), "peak_db": round(an.peak_db(x), 1), "rms_db": round(an.rms_db(x), 1), "spectral_centroid_hz": round(float(np.median(cent))) if len(cent) else None}
            out["_recording"] = r
            out["_pair"] = pair
    return out
