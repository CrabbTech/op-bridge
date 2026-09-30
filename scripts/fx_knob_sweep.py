"""Measure what an effect's four encoders do, for a page without labels (the fazer) or a knob you want numbers for.

Loads a synth slot, holds one note per take with all four FX knobs set explicitly (a program change does not
reload a patch and CC 63 does nothing), sweeps one knob at a time from an all-64 baseline and prints, per take,
the periodic modulation rate and prominence from harmonic tracking, the median harmonic swing, and the mean
levels of the fundamental, harmonics 2-4, 5-12 and 13-24. Use a static engine (voltage, string, digital); a
cluster engine moves on its own and swamps the effect.

    uv run python scripts/fx_knob_sweep.py --slot 3 --note 60 --hold 5 --values 0,42,64,85,106,127 --out takes/
"""
import argparse
import os
import time

import numpy as np
import soundfile as sf

from op_bridge.analysis import harmonic_levels, modulation_rate
from op_bridge.device import Field
from op_bridge.player import detect_live_pair
from op_bridge.session import Config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slot", type=int, default=3)
    ap.add_argument("--note", type=int, default=60, help="MIDI note to hold (60 = C4)")
    ap.add_argument("--hold", type=float, default=5.0, help="seconds to hold; 14 resolves rates below 0.4 Hz")
    ap.add_argument("--values", default="0,42,64,85,106,127")
    ap.add_argument("--knobs", default="fx1,fx2,fx3,fx4", help="which encoders to sweep (fx1-4 patch, master_fx1-4 master)")
    ap.add_argument("--base", type=int, default=64, help="value the other knobs sit at during a sweep")
    ap.add_argument("--out", default=None, help="folder for the takes as WAV")
    a = ap.parse_args()
    cfg = Config.load()
    f0 = 440.0 * 2 ** ((a.note - 69) / 12)
    values = [int(v) for v in a.values.split(",")]
    knobs = a.knobs.split(",")
    base = {k: a.base for k in knobs}
    lo_hz = max(0.08, 2.0 / a.hold)
    if a.out:
        os.makedirs(a.out, exist_ok=True)

    def take(field: Field, label: str, setting: dict[str, int]) -> str:
        for k, v in setting.items():
            field.cc(k, v)
        time.sleep(0.4)
        with field.recorder() as rec:
            time.sleep(0.25); field.note_on(a.note, 100); time.sleep(a.hold); field.note_off(a.note); time.sleep(0.5)
        r = rec.stop()
        pair = detect_live_pair(r.audio)
        if not pair:
            return f"{label:12s} no live audio"
        x = r.audio[:, pair[0] - 1: pair[1]].astype(np.float64).mean(axis=1)
        if a.out:
            sf.write(os.path.join(a.out, label + ".wav"), (x / 32768.0 if np.abs(x).max() > 1.5 else x).astype(np.float32), r.samplerate)
        seg = x[int(0.9 * r.samplerate): len(x) - int(0.6 * r.samplerate)]
        m = modulation_rate(seg, r.samplerate, f0, lo_hz=lo_hz)
        H, _ = harmonic_levels(seg, r.samplerate, f0)
        lv = H.mean(axis=1)
        rate = f"{m['rate_hz']:5.2f} Hz" if m["rate_hz"] else "   none "
        return (f"{label:12s} rate {rate} (prom {m['prominence']:6.1f}) swing {m['swing_db']:4.1f} dB | "
                f"h1 {lv[0]:6.1f} h2-4 {lv[1:4].mean():6.1f} h5-12 {lv[4:12].mean():6.1f} h13-24 {lv[12:].mean():6.1f} dB")

    with Field(channel=cfg.midi_channel - 1) as field:
        field.select_synth_slot(a.slot); time.sleep(0.4)
        print(take(field, "baseline", base), flush=True)
        for knob in knobs:
            for v in values:
                s = dict(base); s[knob] = v
                print(take(field, f"{knob}={v}", s), flush=True)
        for k, v in base.items():
            field.cc(k, v)


if __name__ == "__main__":
    main()
