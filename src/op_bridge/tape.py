"""Tape, mixer and master bus over MIDI, plus backups of what is on the tape."""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from . import analysis as an
from .device import Field, StreamRecorder, usb_layout

TAPE_ACTIONS = {
    "play": "tape_play", "stop": "tape_stop", "start": "tape_start", "end": "tape_end",
    "prev_bar": "tape_prev_bar", "next_bar": "tape_next_bar",
    "loop_in": "tape_loop_in", "loop_out": "tape_loop_out", "loop_toggle": "tape_loop_toggle",
}


def transport(field: Field, action: str) -> str:
    if action in TAPE_ACTIONS:
        field.cc(TAPE_ACTIONS[action], 127)
        return f"tape {action}"
    if action == "midi_start":
        field.start(); return "MIDI start sent (tape playback starts)"
    if action == "midi_stop":
        field.stop(); return "MIDI stop sent"
    if action == "midi_continue":
        field.continue_(); return "MIDI continue sent"
    raise ValueError(f"unknown tape action {action!r}; known: {sorted(TAPE_ACTIONS)} plus midi_start, midi_stop, midi_continue")


def tempo_cc_value(bpm: float) -> int:
    """CC 80 mapping from TE's reference: 0-5 -> 40-50, 6-120 -> 52-166, 121-127 -> 168-180."""
    bpm = max(40.0, min(180.0, bpm))
    if bpm <= 50:
        return int(round((bpm - 40) / 10 * 5))
    if bpm <= 166:
        return int(round(6 + (bpm - 52) / (166 - 52) * (120 - 6)))
    return int(round(121 + (bpm - 168) / (180 - 168) * 6))


def cc_value_to_tempo(v: int) -> float:
    """Inverse of tempo_cc_value, from TE's table."""
    v = max(0, min(127, int(v)))
    if v <= 5:
        return 40.0 + v * 2.0
    if v <= 120:
        return 52.0 + (v - 6) * (166 - 52) / (120 - 6)
    return 168.0 + (v - 121) * 2.0


def set_mixer(field: Field, track: int, volume: int | None = None, pan: int | None = None, mute: bool | None = None) -> dict[str, Any]:
    if not 1 <= track <= 4:
        raise ValueError("track must be 1-4")
    ch = track - 1
    out: dict[str, Any] = {"track": track}
    if volume is not None:
        field.cc("mixer_volume", volume, channel=ch); out["volume"] = volume
    if pan is not None:
        field.cc("mixer_pan", pan, channel=ch); out["pan"] = pan
    if mute is not None:
        field.cc("mixer_mute", 127 if mute else 0, channel=ch); out["mute"] = mute
    return out


MASTER_PARAMS = {
    "fx1": "master_fx1", "fx2": "master_fx2", "fx3": "master_fx3", "fx4": "master_fx4",
    "left": "master1", "right": "master2", "drive": "master3", "release": "master4",
    "eq_low": "eq_low", "eq_mid": "eq_mid", "eq_high": "eq_high",
}


def set_master(field: Field, values: dict[str, int]) -> dict[str, int]:
    done = {}
    for k, v in values.items():
        if k not in MASTER_PARAMS:
            raise ValueError(f"unknown master parameter {k!r}; known: {sorted(MASTER_PARAMS)}")
        field.cc(MASTER_PARAMS[k], int(v)); done[k] = int(v)
    return done


def capture_tape(field: Field, path: str, seconds: float | None = None, from_start: bool = True, silence_stop: float = 4.0, max_seconds: float = 400.0, cancel=None) -> dict[str, Any]:
    """Play the tape and record every USB channel to `path`. Stops after `seconds`, or after
    `silence_stop` seconds of digital silence once something has been heard, or at `max_seconds`;
    a set `cancel` event stops the tape and raises jobs.Cancelled."""
    if field.audio is None:
        raise RuntimeError("no OP-1 field audio input")
    nch = field.audio["inputs"]
    rec = StreamRecorder(field.audio["index"], nch, path, int(field.audio["samplerate"]))
    rec.start()
    t0 = time.time()
    heard = False
    try:
        time.sleep(0.3)
        if from_start:
            field.cc("tape_start", 127); time.sleep(0.2)
        rec.mark("play")
        field.cc("tape_play", 127)
        while True:
            time.sleep(0.25)
            if cancel is not None and cancel.is_set():
                field.cc("tape_stop", 127)
                from .jobs import Cancelled
                raise Cancelled()
            el = time.time() - t0
            pk = rec.recent_peak(0.5)
            if pk > 1e-4:
                heard = True
                last_sound = time.time()
            if seconds is not None and el >= seconds + 0.5:
                break
            if seconds is None and heard and time.time() - last_sound > silence_stop:
                break
            if el > max_seconds:
                break
        field.cc("tape_stop", 127)
        time.sleep(0.3)
    finally:
        path, frames = rec.stop()
    layout = usb_layout(nch)
    return {"path": path, "seconds": round(frames / rec.samplerate, 2), "channels": nch, "layout": {k: list(v) for k, v in layout.items()}, "heard_audio": heard}


def split_stems(path: str, out_dir: str, base: str) -> dict[str, str]:
    """Split a multichannel capture into stereo files per USB pair, named by the layout."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    layout = usb_layout(audio.shape[1])
    os.makedirs(out_dir, exist_ok=True)
    out: dict[str, str] = {}
    for name, (a, b) in layout.items():
        seg = audio[:, a - 1: b]
        if an.rms_db(seg) < -150:
            continue  # empty track
        p = os.path.join(out_dir, f"{base}-{name}.wav")
        sf.write(p, seg, sr, subtype="FLOAT")
        out[name] = p
    return out
