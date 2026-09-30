"""The score: what the model asks the Field to play. Validation and compilation to timed MIDI events."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .device import CC
from .session import Config, allowed_pitch_classes

NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)$")
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def pitch_to_midi(p: int | str) -> int:
    """MIDI number from an int, a note name (C4, F#3, Bb2) or a MIDI number sent as a string ("60", as remote clients often do)."""
    if isinstance(p, int):
        return p
    text = str(p).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    m = NOTE_RE.match(text)
    if not m:
        raise ValueError(f"bad pitch {p!r}; use MIDI numbers or names like C4, F#3, Bb2")
    n = PC[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)]
    return (int(m.group(3)) + 1) * 12 + n


def midi_to_name(n: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[n % 12]}{n // 12 - 1}"


class Note(BaseModel):
    start: float = Field(ge=0, description="beat position, 0 = first beat")
    duration: float = Field(gt=0, description="length in beats")
    pitch: int | str = Field(description="MIDI number or note name like C4, F#3, Bb2")
    velocity: int = Field(default=96, ge=1, le=127)

    @property
    def midi(self) -> int:
        return pitch_to_midi(self.pitch)


class Automation(BaseModel):
    param: str = Field(description="a named parameter: engine1-4, attack, decay, sustain, release, fx1-4, lfo1-4, master_fx1-4, master1-4, eq_low/mid/high, tape_rec_level, metronome, or a raw CC number as a string")
    points: list[list[float]] = Field(description="[[beat, value 0-127], ...]; values are interpolated linearly between points")
    step: bool = Field(default=False, description="true = jump to each value instead of ramping")
    channel: int | None = Field(default=None, description="MIDI channel 1-16 override (mixer CCs use 1-4 for tracks)")

    def cc_number(self) -> int:
        if self.param.isdigit():
            return int(self.param)
        if self.param not in CC:
            raise ValueError(f"unknown parameter {self.param!r}; known: {sorted(CC)}")
        return CC[self.param]


class Bend(BaseModel):
    points: list[list[float]] = Field(description="[[beat, bend -8192..8191], ...], interpolated")


class Sustain(BaseModel):
    start: float
    end: float


class Score(BaseModel):
    tempo: float = Field(default=100.0, ge=20, le=300)
    beats_per_bar: int = Field(default=4, ge=1, le=16)
    swing: float = Field(default=0.0, ge=0, le=1, description="delays every second 8th note; 0.5 is a classic triplet swing")
    humanize_ms: float = Field(default=0.0, ge=0, le=60, description="random timing jitter, standard deviation")
    humanize_velocity: int = Field(default=0, ge=0, le=40)
    seed: int | None = Field(default=None, description="random seed for humanize, for repeatable takes")
    notes: list[Note] = Field(default_factory=list)
    cc: list[Automation] = Field(default_factory=list)
    bends: list[Bend] = Field(default_factory=list)
    sustain: list[Sustain] = Field(default_factory=list)
    length_beats: float | None = Field(default=None, description="total length; default is the last event plus a tail")
    tail_seconds: float = Field(default=1.5, ge=0, le=20, description="recording continues this long after the last note-off")
    send_clock: bool = Field(default=True, description="send MIDI clock at the score tempo while playing")

    @property
    def spb(self) -> float:
        return 60.0 / self.tempo

    def end_beat(self) -> float:
        last = 0.0
        for n in self.notes:
            last = max(last, n.start + n.duration)
        for a in self.cc:
            for b, _ in a.points:
                last = max(last, b)
        for bd in self.bends:
            for b, _ in bd.points:
                last = max(last, b)
        if self.length_beats is not None:
            last = max(last, self.length_beats)
        return last

    def seconds(self) -> float:
        return self.end_beat() * self.spb + self.tail_seconds


@dataclass
class Event:
    t: float            # seconds from score start
    kind: str           # note_on, note_off, cc, bend, clock
    a: int = 0          # note / cc number / bend value
    b: int = 0          # velocity / cc value
    channel: int | None = None
    order: int = 0      # tie-breaker: note_off before note_on at the same time


def compile_events(score: Score, humanize: bool = True) -> list[Event]:
    rng = random.Random(score.seed)
    spb = score.spb
    ev: list[Event] = []

    def swung(beat: float) -> float:
        if score.swing <= 0:
            return beat
        # every second 8th note is delayed by up to a 16th (swing 1.0)
        eighth = beat * 2
        frac = eighth - int(eighth)
        if int(eighth) % 2 == 1 and abs(frac) < 1e-6:
            return beat + score.swing * 0.25
        return beat

    # notes; cut a note short if the same pitch restarts before it ends
    on_times: dict[int, list[tuple[float, float, int]]] = {}
    for n in score.notes:
        m = n.midi
        if not 0 <= m <= 127:
            raise ValueError(f"pitch {n.pitch} out of MIDI range")
        s = swung(n.start) * spb
        e = swung(n.start + n.duration) * spb
        vel = n.velocity
        if humanize:
            if score.humanize_ms:
                s = max(0.0, s + rng.gauss(0, score.humanize_ms / 1000.0))
                e = max(s + 0.01, e + rng.gauss(0, score.humanize_ms / 1000.0))
            if score.humanize_velocity:
                vel = max(1, min(127, int(round(vel + rng.gauss(0, score.humanize_velocity)))))
        on_times.setdefault(m, []).append((s, e, vel))
    for m, lst in on_times.items():
        lst.sort()
        for i, (s, e, vel) in enumerate(lst):
            if i + 1 < len(lst):
                e = min(e, lst[i + 1][0] - 0.002)
            ev.append(Event(s, "note_on", m, vel, order=1))
            ev.append(Event(e, "note_off", m, 0, order=0))

    # automation, 8 ms resolution when ramping
    for a in score.cc:
        num = a.cc_number()
        pts = sorted((float(b), float(v)) for b, v in a.points)
        ch = None if a.channel is None else a.channel - 1
        if not pts:
            continue
        ev.append(Event(pts[0][0] * spb, "cc", num, int(round(pts[0][1])), ch, order=-1))
        for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
            t0, t1 = b0 * spb, b1 * spb
            if a.step or t1 <= t0:
                ev.append(Event(t1, "cc", num, int(round(v1)), ch, order=-1))
                continue
            steps = max(1, int((t1 - t0) / 0.008))
            last = None
            for k in range(1, steps + 1):
                t = t0 + (t1 - t0) * k / steps
                v = int(round(v0 + (v1 - v0) * k / steps))
                if v != last:
                    ev.append(Event(t, "cc", num, max(0, min(127, v)), ch, order=-1)); last = v
    for bd in score.bends:
        pts = sorted((float(b), float(v)) for b, v in bd.points)
        if not pts:
            continue
        ev.append(Event(pts[0][0] * spb, "bend", int(pts[0][1]), 0, order=-1))
        for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
            t0, t1 = b0 * spb, b1 * spb
            steps = max(1, int((t1 - t0) / 0.008))
            for k in range(1, steps + 1):
                ev.append(Event(t0 + (t1 - t0) * k / steps, "bend", int(round(v0 + (v1 - v0) * k / steps)), 0, order=-1))
    for s in score.sustain:
        ev.append(Event(s.start * spb, "cc", 64, 127, order=-1))
        ev.append(Event(s.end * spb, "cc", 64, 0, order=-1))
    if score.send_clock:
        total = score.end_beat() * spb + score.tail_seconds
        tick = spb / 24.0
        n = int(total / tick) + 1
        ev.extend(Event(i * tick, "clock", order=-2) for i in range(n))
    ev.sort(key=lambda e: (e.t, e.order))
    return ev


def max_polyphony(score: Score) -> int:
    """Largest number of notes sounding at once, before humanize."""
    edges: list[tuple[float, int]] = []
    for n in score.notes:
        edges.append((n.start, 1)); edges.append((n.start + n.duration, -1))
    edges.sort(key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, d in edges:
        cur += d; best = max(best, cur)
    return best


def validate(score: Score, cfg: Config) -> list[str]:
    """Return a list of problems (empty = ok). Enforces device limits always and constraints in guided mode."""
    problems: list[str] = []
    cons = cfg.constraints
    poly = max_polyphony(score)
    limit = cons.max_polyphony if cfg.guided() else 8
    if poly > limit:
        problems.append(f"{poly} simultaneous notes; the Field steals voices above {limit}")
    if score.seconds() > (cons.max_take_seconds if cfg.guided() else 600):
        problems.append(f"take is {score.seconds():.0f} s, longer than allowed")
    for n in score.notes:
        try:
            m = n.midi
        except ValueError as e:
            problems.append(str(e)); continue
        if not 24 <= m <= 108:
            problems.append(f"pitch {n.pitch} is outside the useful range C1-C8")
    if cfg.guided():
        if cons.tempo is not None and abs(score.tempo - cons.tempo) > 0.01:
            problems.append(f"tempo must be {cons.tempo} in this session")
        if cons.tempo_range is not None and not (cons.tempo_range[0] <= score.tempo <= cons.tempo_range[1]):
            problems.append(f"tempo must be within {cons.tempo_range}")
        # send_clock makes the player send CC 80 (the Field's BPM) and MIDI clock: transport, and a tempo change
        # unless the score sits on the session's fixed tempo
        if score.send_clock and not cons.allow_transport:
            problems.append("send_clock sends MIDI clock and CC 80 (tempo), which are transport, reserved for the human; set send_clock=false")
        elif score.send_clock and not cons.allow_tempo_change and (cons.tempo is None or abs(score.tempo - cons.tempo) > 0.01):
            problems.append("send_clock sends CC 80 to set the Field's BPM to the score tempo; tempo change is reserved for the human: "
                            "set send_clock=false, or use the session's fixed tempo")
        if cons.key:
            allowed = allowed_pitch_classes(cons.key) | set(cons.extra_pitch_classes)
            bad = sorted({midi_to_name(n.midi) for n in score.notes if n.midi % 12 not in allowed})
            if bad:
                problems.append(f"notes outside {cons.key}: {', '.join(bad)}")
        for a in score.cc:
            group = _cc_group(a.param)
            if group == "sound_design" and not cons.allow_sound_design:
                problems.append(f"automation of {a.param} is sound design, reserved for the human")
            if group == "master" and not cons.allow_master:
                problems.append(f"automation of {a.param} touches the master bus, reserved for the human")
            if group == "mixer" and not cons.allow_mixer:
                problems.append(f"automation of {a.param} touches the mixer, reserved for the human")
            if group == "transport" and not cons.allow_transport:
                problems.append(f"automation of {a.param} touches transport, reserved for the human")
            if cons.allowed_cc_params is not None and a.param not in cons.allowed_cc_params:
                problems.append(f"automation of {a.param} is not in the allowed list {cons.allowed_cc_params}")
    return problems


def _cc_group(param: str) -> str:
    if param.isdigit():
        n = int(param)
        if 46 <= n <= 63: return "sound_design"
        if 70 <= n <= 77 or 90 <= n <= 92: return "master"
        if n in (7, 9, 10): return "mixer"
        if 78 <= n <= 88 or n in (104, 105): return "transport"
        return "other"
    if param.startswith(("engine", "attack", "decay", "sustain", "release", "fx", "lfo", "randomize", "reset")): return "sound_design"
    if param.startswith(("master", "eq_")): return "master"
    if param.startswith("mixer_"): return "mixer"
    if param.startswith("tape_") or param in ("tempo", "metronome", "octave"): return "transport"
    return "other"


SCORE_GUIDE = """Score format (JSON). Times are in beats; beat 0 is the first beat.
{
  "tempo": 96, "beats_per_bar": 4, "swing": 0.0, "humanize_ms": 6, "humanize_velocity": 8,
  "notes": [{"start": 0, "duration": 1, "pitch": "C4", "velocity": 96}, {"start": 1, "duration": 0.5, "pitch": 67, "velocity": 70}],
  "cc": [{"param": "engine1", "points": [[0, 20], [8, 110], [16, 20]]}, {"param": "fx4", "points": [[0, 0], [16, 90]], "step": false}],
  "bends": [{"points": [[4, 0], [5, 4096], [6, 0]]}],
  "sustain": [{"start": 0, "end": 4}],
  "length_beats": 16, "tail_seconds": 1.5, "send_clock": true
}
Pitches are MIDI numbers or names (C4 = 60). Velocity 1-127 is honored by the Field (about 20 dB of range).
Chords are just notes with the same start. Keep simultaneous notes at or below the polyphony limit.
Parameters for "cc": engine1-4 (the four encoders of the engine page), attack, decay, sustain, release,
fx1-4, lfo1-4, master_fx1-4, master1-4 (master left, right, drive, release), eq_low, eq_mid, eq_high,
tape_rec_level, metronome, mixer_volume / mixer_pan / mixer_mute (with "channel" 1-4 = tape track).
Values are 0-127. A note that restarts before it ends cuts the earlier one short.
"""
