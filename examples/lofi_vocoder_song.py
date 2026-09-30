"""A lo-fi hip hop piece with vocoder vocals, played on the OP-1 field through op-bridge.

Usage (Field connected, normal mode):
    uv run python examples/lofi_vocoder_song.py rehearse drums|keys|bass|vocals [bars]
    uv run python examples/lofi_vocoder_song.py record   drums|keys|bass|vocals      # the human has armed a tape track
    uv run python examples/lofi_vocoder_song.py bounce                              # play the tape and capture stems + main mix

Every part starts on beat one so the Field's armed recording starts each track at the same place.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import soundfile as sf

from op_bridge import analysis as an, drums as DR, speech as SP, tape as T
from op_bridge.device import Field
from op_bridge.player import play_score, save_take
from op_bridge.score import Score, pitch_to_midi
from op_bridge.session import Config, session_dir, new_id, write_json

TEMPO = 78.0
BARS = 24
SPB = 60.0 / TEMPO
SLOTS = {"drums": ("drum", 1), "keys": ("synth", 2), "bass": ("synth", 5), "vocals": ("synth", 8)}   # keys: the piano-like dimension engine on key 2; samplers only pitch near their root
OUT = session_dir("lofi-vocoder")

# ----------------------------------------------------------------------------- harmony
A_SECTION = [["F3", "A3", "C4", "E4"], ["E3", "G3", "B3", "D4"], ["D3", "F3", "A3", "C4"], ["C3", "E3", "G3", "B3"]]     # Fmaj7 Em7 Dm7 Cmaj7
B_SECTION = [["A2", "C3", "E3", "G3", "B3"], ["D3", "F3", "A3", "C4"], ["G2", "B2", "D3", "F3"], ["C3", "E3", "G3", "B3"]]  # Am9 Dm7 G7 Cmaj7
BASS_A = ["F2", "E2", "D2", "C2"]
BASS_B = ["A1", "D2", "G1", "C2"]


def chord_for_bar(bar: int) -> list[str]:
    """bar is 1-based. 1-12 and 21-24: A section; 13-20: B section."""
    if 13 <= bar <= 20:
        return B_SECTION[(bar - 13) % 4]
    return A_SECTION[(bar - 1) % 4]


def bass_for_bar(bar: int) -> str:
    if 13 <= bar <= 20:
        return BASS_B[(bar - 13) % 4]
    return BASS_A[(bar - 1) % 4]


def fifth_above(root: str) -> int:
    return pitch_to_midi(root) + 7


# ----------------------------------------------------------------------------- parts
def drums_doc() -> dict:
    intro = {"CH": "o...o...o...o..."}                                                   # a soft tick keeps time and starts the tape
    groove = {"BD": "x......x..x.....", "SN": "....x.......x...", "SN2": "......o....o..o.",
              "CH": "X.x.x.x.X.x.x.x.", "OH": ".......x........", "CHK": "........x......."}
    groove_b = {"BD": "x.....x...x...x.", "SN": "....x.......x...", "SN2": "......o.......o.", "CL": "............x...",
                "CH": "X.x.x.x.X.x.x.x.", "OH": "...............x", "CHK": "x..............."}
    fill = {"BD": "x......x..x.....", "SN": "....x.......x.xx", "SN2": "......o....o....", "CH": "X.x.x.x.X.x.x...", "PERC": "..............x."}
    outro_1 = {"BD": "x.........x.....", "SN": "....x...........", "CH": "x...x...x...x..."}
    outro_2 = {"CY": "o..............."}
    return {
        "tempo": TEMPO, "steps_per_bar": 16, "swing": 0.55, "humanize_ms": 9, "humanize_velocity": 8, "seed": 11,
        "velocities": {"hit": 82, "accent": 100, "ghost": 46},
        "kit_map": {"BD": 53, "SN": 55, "SN2": 56, "CH": 60, "OH": 63, "CHK": 61, "CL": 58, "PERC": 66, "CY": 68},
        "patterns": {"I": intro, "A": groove, "F": fill, "B": groove_b, "O1": outro_1, "O2": outro_2},
        "arrangement": ["I", "I", "I", "I", "A", "A", "A", "F", "A", "A", "A", "F", "B", "B", "B", "F", "B", "B", "B", "F", "A", "A", "O1", "O2"],
    }


def keys_score() -> Score:
    notes = []
    for bar in range(1, BARS + 1):
        base = (bar - 1) * 4
        chord = chord_for_bar(bar)
        top_up = [pitch_to_midi(p) for p in chord]           # a synthesis engine pitches every note exactly
        if 13 <= bar <= 20:      # B section: sparser stabs, room for the voice
            hits = [(0.0, 1.5, 74), (2.5, 0.9, 62)]
        elif bar >= 23:          # outro: one long chord per bar
            hits = [(0.0, 3.8, 70)]
        else:                    # comping: one, the and of two, and four
            hits = [(0.0, 1.4, 78), (1.5, 0.6, 60), (3.0, 0.8, 68)]
        for off, dur, vel in hits:
            for i, p in enumerate(top_up):
                notes.append({"start": base + off, "duration": dur, "pitch": p, "velocity": max(1, vel - (i == 0) * 6)})
        if bar in (12, 20):      # a small pickup melody into the next section
            for k, (p, off) in enumerate(zip(["E5", "D5", "C5"], [2.5, 3.0, 3.5])):
                notes.append({"start": base + off, "duration": 0.45, "pitch": p, "velocity": 64})
    # key 2's preset carries the terminal FX: a slow sweep of its third knob through the B section, back by the end
    cc = [{"param": "fx3", "points": [[0, 40], [48, 40], [64, 88], [80, 70], [92, 40]]}]
    return Score(tempo=TEMPO, notes=notes, cc=cc, humanize_ms=14, humanize_velocity=6, seed=5, tail_seconds=1.5, send_clock=True)


def bass_score() -> Score:
    notes = []
    for bar in range(1, BARS + 1):
        base = (bar - 1) * 4
        root = bass_for_bar(bar)
        r = pitch_to_midi(root)
        if bar <= 4:             # intro: one note per bar, quiet, to start the tape and set the floor
            notes.append({"start": base, "duration": 3.5, "pitch": r, "velocity": 58})
            continue
        if bar >= 23:
            notes.append({"start": base, "duration": 3.6, "pitch": r, "velocity": 70})
            continue
        notes.append({"start": base + 0.0, "duration": 1.3, "pitch": r, "velocity": 96})
        notes.append({"start": base + 1.5, "duration": 0.6, "pitch": r, "velocity": 80})
        notes.append({"start": base + 2.5, "duration": 0.9, "pitch": r if bar % 2 else r + 7, "velocity": 86})
        if bar % 4 == 0:
            notes.append({"start": base + 3.5, "duration": 0.45, "pitch": r + 12, "velocity": 72})
    return Score(tempo=TEMPO, notes=notes, humanize_ms=6, humanize_velocity=5, seed=9, tail_seconds=1.5, send_clock=True)


LINES = [
    (13, "Late night. Low light."),
    (15, "The city hums, in D minor."),
    (17, "Tape hiss, and a slow heartbeat."),
    (19, "We are the ghost, in the machine."),
    (22, "Keep the chords. Lose the clock."),
]
HUMS = [(14.0, "Mmmmmmmm.", 2.4), (16.0, "Aaaaaaah.", 2.4), (18.0, "Mmmmmmmm.", 2.4), (20.0, "Oooooooh.", 2.2), (23.0, "Mmmmmmmm.", 5.0)]   # (bar, sound, seconds)


def sustain(x: np.ndarray, sr: int, seconds: float, fade_ms: float = 40.0) -> np.ndarray:
    """Loop the voiced middle of a short clip, with crossfades, into a steady hum of `seconds`."""
    n = len(x)
    core = x[int(n * 0.25): int(n * 0.75)]
    if len(core) < int(0.05 * sr):
        core = x
    cf = int(fade_ms / 1000.0 * sr)
    out = core.copy()
    while len(out) < int(seconds * sr):
        a, b = out[:-cf], out[-cf:]
        ramp = np.linspace(0, 1, cf, dtype=np.float32)[:, None]
        joined = np.concatenate([a, b * (1 - ramp) + core[:cf] * ramp, core[cf:]], axis=0)
        out = joined
    out = out[: int(seconds * sr)]
    env = np.ones(len(out), np.float32)
    k = int(0.25 * sr)
    env[:k] = np.linspace(0, 1, k); env[-k:] = np.linspace(1, 0, k)
    return out * env[:, None]


def vocals(voice: str = "Daniel", rate: int = 132, gain_db: float = -2.0, hum_db: float = -8.0) -> tuple[Score, np.ndarray]:
    """Carrier chords for every bar (silent until the voice arrives), slower spoken lines on their bars, hums between them."""
    notes = []
    for bar in range(1, BARS + 1):
        base = (bar - 1) * 4
        chord = chord_for_bar(bar)
        shell = [pitch_to_midi(chord[0]) + 12, pitch_to_midi(chord[1]) + 12, pitch_to_midi(chord[-1]) + 12]   # root, third, top: a leaner carrier
        for m in shell:
            notes.append({"start": base, "duration": 3.95, "pitch": m, "velocity": 90})
    # vocoder set for intelligibility at the start of the take: more bands, mix toward the vocoded signal
    cc = [{"param": "engine3", "points": [[0, 104]], "step": True}, {"param": "engine4", "points": [[0, 100]], "step": True}]
    score = Score(tempo=TEMPO, notes=notes, cc=cc, tail_seconds=1.5, send_clock=True)
    sr = 44100
    total = int((BARS * 4 * SPB + 2.0 + 0.5) * sr)
    buf = np.zeros((total, 2), np.float32)
    os.makedirs(os.path.join(OUT, "speech"), exist_ok=True)
    g = 10 ** (gain_db / 20.0)
    def place(x: np.ndarray, bar: float, gain: float) -> None:
        at = int((0.5 + (bar - 1) * 4 * SPB) * sr)
        end = min(total, at + len(x))
        buf[at:end] += x[: end - at] * gain
    for bar, text in LINES:
        wav = os.path.join(OUT, "speech", f"line-{bar:02d}.wav")
        SP.synthesize(text, wav, voice=voice, rate=rate)
        x, xsr = sf.read(wav, dtype="float32", always_2d=True)
        place(np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x, bar, g)
    for bar, sound, seconds in HUMS:
        wav = os.path.join(OUT, "speech", f"hum-{bar:04.1f}.wav")
        SP.synthesize(sound, wav, voice=voice, rate=rate)
        x, xsr = sf.read(wav, dtype="float32", always_2d=True)
        x = np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x
        place(sustain(x, xsr, seconds), bar, 10 ** (hum_db / 20.0))
    return score, np.clip(buf, -1, 1)


def part(name: str, bars: int | None = None):
    if name == "drums":
        doc = drums_doc()
        if bars:
            doc["arrangement"] = doc["arrangement"][:bars]
        return DR.compile_drums(doc), None
    if name == "keys":
        s = keys_score()
    elif name == "bass":
        s = bass_score()
    elif name == "vocals":
        s, buf = vocals()
        if bars:
            s = Score(**{**s.model_dump(), "notes": [n for n in s.model_dump()["notes"] if n["start"] < bars * 4], "length_beats": bars * 4})
            buf = buf[: int((0.5 + bars * 4 * SPB + 2.0) * 44100)]
        return s, buf
    else:
        raise SystemExit(f"unknown part {name}")
    if bars:
        s = Score(**{**s.model_dump(), "notes": [n for n in s.model_dump()["notes"] if n["start"] < bars * 4], "length_beats": bars * 4})
    return s, None


def play_part(name: str, bars: int | None, to_tape: bool) -> dict:
    cfg = Config.load()
    score, playback = part(name, bars)
    kind, slot = SLOTS[name]
    with Field(channel=cfg.midi_channel - 1) as f:
        (f.select_synth_slot if kind == "synth" else f.select_drum_slot)(slot); time.sleep(0.5)
        # never touch the transport after the human has armed: a jump or a MIDI start cancels the arm.
        # The first incoming note starts the armed recording (verified on the device); the tape must
        # already sit at the start, which rewind() does before the human arms.
        res = play_score(f, score, record=True, latency_ms=cfg.latency_ms, playback=playback, set_tempo=not to_tape)
        if to_tape:
            f.cc("tape_stop", 127)
    tid = new_id(f"{'take' if to_tape else 'rehearsal'}-{name}")
    wav = save_take(res, os.path.join(OUT, "takes", tid + ".wav"))
    a = res.analysis
    if name == "drums":
        for n in a.get("notes", []):
            n["heard"] = n.get("onset_offset_ms") is not None
        a["hits_heard"] = sum(1 for n in a.get("notes", []) if n["heard"]); a["hits_checked"] = len(a.get("notes", []))
    write_json(os.path.join(OUT, "takes", tid + ".json"), {"id": tid, "part": name, "bars": bars or BARS, "to_tape": to_tape, "analysis": a, "wav": wav})
    keep = {k: a.get(k) for k in ("live_usb_channels", "peak_db", "rms_db", "main_mix", "notes_heard", "notes_checked", "hits_heard", "hits_checked", "spectral_centroid_hz")}
    return {"part": name, "bars": bars or BARS, "seconds": round(score.seconds(), 1), "take": tid, **keep}


def rewind() -> None:
    cfg = Config.load()
    with Field(channel=cfg.midi_channel - 1) as f:
        f.cc("tape_stop", 127); time.sleep(0.2); f.cc("tape_start", 127); time.sleep(0.2)
        f.cc("tempo", T.tempo_cc_value(TEMPO))      # tempo before arming; a tempo change after arming may cancel it


def bounce() -> dict:
    cfg = Config.load()
    base = new_id("lofi-vocoder")
    with Field(channel=cfg.midi_channel - 1) as f:
        res = T.capture_tape(f, os.path.join(OUT, "backups", base + "-all.wav"), seconds=BARS * 4 * SPB + 3.0)
    stems = T.split_stems(res["path"], os.path.join(OUT, "backups"), base)
    res["stems"] = stems
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd in ("rehearse", "record"):
        name = sys.argv[2]
        bars = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
        for arg in sys.argv[3:]:
            if arg.startswith("slot="):                      # e.g. record bass slot=5
                SLOTS[name] = (SLOTS[name][0], int(arg.split("=")[1]))
        print(json.dumps(play_part(name, bars, to_tape=(cmd == "record")), indent=1))
    elif cmd == "rewind":
        rewind(); print("tape stopped and at the start; now arm the track")
    elif cmd == "bounce":
        print(json.dumps(bounce(), indent=1))
    elif cmd == "grid":
        print(DR.render_grid(drums_doc()))
    else:
        print(__doc__)
