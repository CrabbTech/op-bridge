"""Empirical characterization of a connected OP-1 field. Run: uv run python -m op_bridge.probe <step>"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from . import analysis as an
from .device import Field, Recording

OUT_DIR = os.environ.get("OP_BRIDGE_PROBE_DIR", os.path.expanduser("~/Music/op-bridge/probe"))


def _out(name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def _save_wav(rec: Recording, name: str) -> str:
    import soundfile as sf

    path = _out(name)
    sf.write(path, rec.audio, rec.samplerate, subtype="FLOAT")
    return path


def live_channels(rec: Recording, start: int, seconds: float = 0.8, min_db: float = -50.0) -> list[int]:
    seg = rec.audio[start: start + int(seconds * rec.samplerate)]
    levels = an.channel_levels(seg)
    return [c for c, lv in enumerate(levels) if lv > min_db]


# ----------------------------------------------------------------------------- steps

def step_channels(field: Field) -> dict:
    """Play one note; report the level on every USB channel and the latency from note-on to onset."""
    with field.recorder() as rec:
        time.sleep(0.4)
        rec.mark("floor")
        time.sleep(0.3)
        rec.mark("note_on")
        field.note_on(60, 110)
        time.sleep(1.0)
        rec.mark("note_off")
        field.note_off(60)
        time.sleep(1.0)
    r = rec.stop()
    on = r.mark_sample("note_on")
    result = {"samplerate": r.samplerate, "channels": []}
    for c in range(r.audio.shape[1]):
        x = r.audio[:, c]
        floor = an.rms_db(x[r.mark_sample("floor"): on])
        level = an.rms_db(x[on + int(0.1 * r.samplerate): on + int(0.9 * r.samplerate)])
        onset = an.onset_sample(x, r.samplerate, start=on - int(0.05 * r.samplerate))
        latency_ms = None if onset is None else (onset - on) / r.samplerate * 1000.0
        result["channels"].append({"channel": c + 1, "floor_db": round(floor, 1), "note_db": round(level, 1), "latency_ms": None if latency_ms is None else round(latency_ms, 1)})
    result["wav"] = _save_wav(r, "channels.wav")
    top = max(c["note_db"] for c in result["channels"])
    lc = [c["channel"] for c in result["channels"] if c["note_db"] > -80 and c["note_db"] > top - 12]
    result["live_channels"] = lc
    if lc:
        c0 = lc[0] - 1
        result["spectrogram"] = an.spectrogram_png(r.audio[:, c0], r.samplerate, _out("channels.png"), marks=r.marks)
    return result


def step_velocity(field: Field, channel: int) -> dict:
    vels = [8, 16, 32, 48, 64, 80, 96, 112, 127]
    with field.recorder() as rec:
        time.sleep(0.3)
        for v in vels:
            rec.mark(f"v{v}")
            field.note_on(60, v)
            time.sleep(0.5)
            field.note_off(60)
            time.sleep(0.5)
        time.sleep(0.3)
    r = rec.stop()
    x = r.audio[:, channel - 1]
    rows = []
    for v in vels:
        s = r.mark_sample(f"v{v}")
        seg = x[s + int(0.05 * r.samplerate): s + int(0.45 * r.samplerate)]
        rows.append({"velocity": v, "peak_db": round(an.peak_db(seg), 1), "rms_db": round(an.rms_db(seg), 1)})
    return {"rows": rows, "wav": _save_wav(r, "velocity.wav")}


def step_polyphony(field: Field, channel: int) -> dict:
    """Play growing chords; count how many of the notes are actually sounding."""
    base = [48, 55, 60, 64, 67, 71, 74, 79, 84, 88]  # spread pitches, distinct pitch classes where possible
    results = []
    with field.recorder() as rec:
        time.sleep(0.3)
        for n in (3, 4, 5, 6, 7, 8, 10):
            notes = base[:n]
            rec.mark(f"chord{n}")
            for m in notes:
                field.note_on(m, 100)
                time.sleep(0.004)
            time.sleep(1.4)
            for m in notes:
                field.note_off(m)
            time.sleep(0.8)
        time.sleep(0.3)
    r = rec.stop()
    x = r.audio[:, channel - 1]
    for n in (3, 4, 5, 6, 7, 8, 10):
        notes = base[:n]
        s = r.mark_sample(f"chord{n}")
        seg = x[s + int(0.4 * r.samplerate): s + int(1.3 * r.samplerate)]
        prom = an.note_prominence(seg, r.samplerate, notes)
        sounding = [m for m in notes if prom[m] >= 10.0]
        results.append({"asked": n, "sounding": len(sounding), "missing": [m for m in notes if m not in sounding], "prominence_db": {str(m): round(p, 1) for m, p in prom.items()}})
    return {"rows": results, "wav": _save_wav(r, "polyphony.wav")}


def step_cc(field: Field, channel: int, control: str = "engine1") -> dict:
    """Hold a note while sweeping one CC; report how the spectral centroid moves. Restores the patch by reloading slot 1."""
    with field.recorder() as rec:
        time.sleep(0.3)
        field.cc(control, 0)
        time.sleep(0.2)
        rec.mark("note_on")
        field.note_on(60, 100)
        time.sleep(0.6)
        rec.mark("sweep_up")
        for v in range(0, 128, 2):
            field.cc(control, v)
            time.sleep(0.02)
        rec.mark("sweep_down")
        for v in range(127, -1, -2):
            field.cc(control, v)
            time.sleep(0.02)
        rec.mark("sweep_end")
        time.sleep(0.5)
        field.note_off(60)
        time.sleep(0.6)
    r = rec.stop()
    field.select_synth_slot(1)  # reload the saved preset so the sweep leaves no trace
    x = r.audio[:, channel - 1]
    t, cent = an.spectral_centroid_series(x, r.samplerate)
    def cent_at(label):
        s = r.mark_sample(label) / r.samplerate
        i = int(np.argmin(np.abs(t - s)))
        return float(np.mean(cent[i: i + 6]))
    def cent_between(a, b):
        sa, sb = r.mark_sample(a) / r.samplerate, r.mark_sample(b) / r.samplerate
        sel = (t >= sa) & (t <= sb)
        return float(cent[sel].min()), float(cent[sel].max())
    lo, hi = cent_between("sweep_up", "sweep_end")
    return {
        "control": control,
        "centroid_hz_before_sweep": round(cent_at("note_on"), 1),
        "centroid_hz_min_during_sweep": round(lo, 1),
        "centroid_hz_max_during_sweep": round(hi, 1),
        "rms_db_during_sweep": round(an.rms_db(x[r.mark_sample("sweep_up"): r.mark_sample("sweep_end")]), 1),
        "wav": _save_wav(r, f"cc_{control}.wav"),
        "spectrogram": an.spectrogram_png(x, r.samplerate, _out(f"cc_{control}.png"), marks=r.marks),
    }


def step_slots(field: Field, channel: int) -> dict:
    """Load synth slots 1-8 by program change, play the same note on each, fingerprint them."""
    rows = []
    with field.recorder() as rec:
        time.sleep(0.3)
        for slot in range(1, 9):
            field.select_synth_slot(slot)
            time.sleep(0.35)
            rec.mark(f"slot{slot}")
            field.note_on(60, 100)
            time.sleep(0.8)
            field.note_off(60)
            time.sleep(0.7)
        time.sleep(0.3)
    r = rec.stop()
    x = r.audio[:, channel - 1]
    for slot in range(1, 9):
        s = r.mark_sample(f"slot{slot}")
        seg = x[s: s + int(1.4 * r.samplerate)]
        onset = an.onset_sample(x, r.samplerate, start=s - int(0.05 * r.samplerate))
        t, cent = an.spectral_centroid_series(seg, r.samplerate)
        body = seg[int(0.1 * r.samplerate): int(0.7 * r.samplerate)]
        tail = seg[int(0.9 * r.samplerate): int(1.3 * r.samplerate)]
        rows.append({
            "slot": slot,
            "peak_db": round(an.peak_db(seg), 1),
            "body_rms_db": round(an.rms_db(body), 1),
            "release_tail_db": round(an.rms_db(tail), 1),
            "attack_ms": None if onset is None else round((onset - s) / r.samplerate * 1000.0, 1),
            "centroid_hz": round(float(np.median(cent)), 0),
        })
    field.select_synth_slot(1)
    return {"rows": rows, "wav": _save_wav(r, "slots.wav"), "spectrogram": an.spectrogram_png(x, r.samplerate, _out("slots.png"), marks=r.marks)}


def step_drums(field: Field, channel: int, slot: int = 1, lo: int = 36, hi: int = 96) -> dict:
    """Switch to drum slot `slot`, tap every MIDI note in [lo, hi], report which ones make a sound."""
    notes = list(range(lo, hi + 1))
    with field.recorder() as rec:
        time.sleep(0.3)
        field.select_drum_slot(slot)
        time.sleep(0.5)
        for m in notes:
            rec.mark(f"n{m}")
            field.note_on(m, 110)
            time.sleep(0.12)
            field.note_off(m)
            time.sleep(0.10)
        time.sleep(0.5)
    r = rec.stop()
    field.select_synth_slot(1)  # back to synth mode, slot 1
    x = r.audio[:, channel - 1]
    rows = {}
    for m in notes:
        s = r.mark_sample(f"n{m}")
        seg = x[s: s + int(0.2 * r.samplerate)]
        rows[m] = round(an.peak_db(seg), 1)
    floor = float(np.percentile(list(rows.values()), 10))
    sounding = [m for m, p in rows.items() if p > floor + 15]
    return {"drum_slot": slot, "sounding_notes": sounding, "lowest": min(sounding) if sounding else None, "highest": max(sounding) if sounding else None, "peaks_db": rows, "wav": _save_wav(r, "drums.wav")}


STEPS = {
    "channels": step_channels,
    "velocity": step_velocity,
    "polyphony": step_polyphony,
    "cc": step_cc,
    "slots": step_slots,
    "drums": step_drums,
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in STEPS:
        print("usage: python -m op_bridge.probe <" + "|".join(STEPS) + "> [channel] [extra]")
        return 2
    step = argv[0]
    with Field() as field:
        field.set_mode("synth")
        time.sleep(0.2)
        if step == "channels":
            res = step_channels(field)
        else:
            channel = int(argv[1]) if len(argv) > 1 else 1
            extra = argv[2:]
            if step == "cc" and extra:
                res = step_cc(field, channel, extra[0])
            elif step == "drums" and extra:
                res = step_drums(field, channel, int(extra[0]))
            else:
                res = STEPS[step](field, channel)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
