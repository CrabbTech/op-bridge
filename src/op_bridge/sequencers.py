"""The Field's seven sequencers: what they are, what the bridge can drive over MIDI, and helpers that build the messages.

Sources: the OP-1 field user guide for firmware 1.7 (pages 47 to 53), TE's MIDI reference for firmware 1.7.0, the
firmware changelog and docs/device.md. Nothing in this module opens a device. The integrator wires SEQUENCER_GUIDE into
get_guide("sequencers"), HUMAN_STEPS into human_steps, and the helpers into tools.
"""
from __future__ import annotations

import copy
from typing import Any, Sequence

import mido

from .score import Event, Note, Score, compile_events
from .tape import tempo_cc_value

CLOCK_TICKS_PER_BEAT = 24     # MIDI clock: pulses per quarter note
TICKS_PER_SIXTEENTH = 6       # one song position unit (a "MIDI beat") is a 16th note, six clock ticks
MAX_SONG_POSITION = 16383     # 14 bits: LSB then MSB after the F2 status byte
MAX_ENDLESS_STEPS = 999       # raised from 128 in firmware 1.2.5
SAFE_POLYPHONY = 6            # held chord notes or tombola balls; matches Constraints.max_polyphony

# Browser order on the device.
SEQUENCER_NAMES: tuple[str, ...] = ("arpeggio", "endless", "finger", "hold", "pattern", "sketch", "tombola")

_ENC = ("blue", "ochre", "gray", "orange")


def _enc(*labels: str | None) -> dict[str, str | None]:
    return dict(zip(_ENC, labels))


# The catalog. "encoders" is the main page, "shift" the page reached by holding shift (None when there is none).
# "midi_driveable" says whether a MIDI note-on from the bridge makes the sequencer produce sound; "midi_evidence"
# says how well that is established. Content entry (steps, grid, curve, finger patterns) is always human-only.
_SEQUENCERS: list[dict[str, Any]] = [
    {
        "name": "arpeggio", "manual_page": 48,
        "screen": "note value (1/8), trigger mode ALL, a rhythm of trigger dots, HOLD",
        "drives_by": "held keys: the chord you hold is arpeggiated at the note value",
        "content": "none stored; the held notes are the material",
        "encoders": _enc("note value", "trigger mode", "trigger pattern", "hold"),
        "shift": _enc("note length", "type", "pause / skip", "swing"),
        "midi_driveable": True,
        "midi_evidence": "changelog 1.2.5 'velocity support in arpeggio and hold sequencer' (before the keyboard had velocity) and 1.2.7 'support full external midi range in arpeggio sequencer'",
        "notes": "a chord held N beats at note value 1/n yields N*n/4 arpeggio notes; trigger modes seen: ALL and EACH; the type encoder sets the arpeggio order",
    },
    {
        "name": "endless", "manual_page": 48,
        "screen": "the step count (128 in the manual), note value (1/4), OFF/HOLD",
        "drives_by": "stored steps entered from the keys, or from the bridge's notes, while stopped; a key press or note plays them transposed to that key",
        "content": "up to 999 steps, entered on the device or sent as short notes while the sequencer is stopped",
        "encoders": _enc("note value", "swing", "trigger pattern", "hold"),
        "shift": _enc("manual mode", None, "rotate pattern", "direction"),
        "midi_driveable": True,
        "midi_evidence": "verified 2026-09-26 in drum mode (docs/device.md): eight short incoming notes entered steps while the sequencer was stopped and one held note played the sequence back transposed, 18 hits in 4 s on a steady grid; synth mode by analogy",
        "notes": "one step is 4/n beats at note value 1/n; a full pass over S steps takes S*4/n beats",
    },
    {
        "name": "finger", "manual_page": 49,
        "screen": "two pattern players A and B, a JOIN control, -1 +1, a keyboard graphic (same for finger drum)",
        "drives_by": "held keys: each key triggers a stored pattern in that key",
        "content": "patterns edited on the device with the cursor and erase; velocity support since 1.3.2",
        "encoders": _enc("move cursor", "swing", "pattern length", "hold"),
        "shift": _enc("erase notes", None, None, "play mode"),
        "midi_driveable": True,
        "midi_evidence": "by analogy with a key press (unverified); changelog 1.3.2 'finger sequencer velocity support'",
        "notes": "finger exists for synth and for drum with the same encoders",
    },
    {
        "name": "hold", "manual_page": 50,
        "screen": "the current note (D3), POLY, two keyboard rows with a break-point marker",
        "drives_by": "held keys or chords, which the sequencer sustains",
        "content": "none stored; the held notes are the material",
        "encoders": _enc("break point", "mono / poly", "transpose", "hold"),
        "shift": None,
        "midi_driveable": True,
        "midi_evidence": "changelog 1.2.5 'velocity support in arpeggio and hold sequencer'",
        "notes": "clicking the orange encoder clears; whether a chord outlives the bridge's note-off until the next chord is unverified",
    },
    {
        "name": "pattern", "manual_page": 50,
        "screen": "a 16-column step grid with a playhead, swing 48",
        "drives_by": "placed grid steps, played by keys",
        "content": "notes placed on the grid on the device",
        "encoders": _enc("move cursor", "swing", "pattern length", "hold"),
        "shift": _enc("erase notes", "offset notes", "move section", "play mode"),
        "midi_driveable": True,
        "midi_evidence": "by analogy with a key press (unverified)",
        "notes": "a drawn step sequencer; play mode is on the shift page",
    },
    {
        "name": "sketch", "manual_page": 51,
        "screen": "a curve drawn on a dotted grid",
        "drives_by": "a drawn curve, traced while a key is held",
        "content": "the curve, drawn on the device with the four encoders",
        "encoders": _enc("draw x", "draw y", "move x", "move y"),
        "shift": _enc("erase", "use divider", "use grid", "hold"),
        "midi_driveable": True,
        "midi_evidence": "by analogy with a key press (unverified)",
        "notes": "hold is on the shift page",
    },
    {
        "name": "tombola", "manual_page": 51,
        "screen": "balls bouncing inside a rotating polygon (a hexagon), G-FORCE, BOUNCE",
        "drives_by": "held keys: each key drops a ball into the polygon and every bounce retriggers the note",
        "content": "none stored; the balls are the material",
        "encoders": _enc("rotation speed", "heaviness (g-force)", "shape (polygon sides)", "bounciness"),
        "shift": _enc("manual mode", None, None, None),
        "midi_driveable": True,
        "midi_evidence": "by analogy with a key press (unverified); whether balls outlive the note-off is open",
        "notes": "each note-on is one ball; keep balls at or below six so the engine does not steal voices",
    },
]


def sequencer_catalog() -> list[dict[str, Any]]:
    """The seven sequencers as data, in browser order: name, screen, drives_by, content, encoders (blue, ochre, gray,
    orange), shift page or None, midi_driveable, midi_evidence, notes, manual_page. Returns copies."""
    return copy.deepcopy(_SEQUENCERS)


def sequencer_info(name: str) -> dict[str, Any]:
    key = name.lower().strip()
    for s in _SEQUENCERS:
        if s["name"] == key:
            return copy.deepcopy(s)
    raise ValueError(f"unknown sequencer {name!r}; the Field has {', '.join(SEQUENCER_NAMES)}")


# ------------------------------------------------------------------ human-only steps (exact key presses from the manual)

HUMAN_STEPS: dict[str, str] = {
    "select_sequencer": "In synth or drum mode hold shift and press the sequencer key: the sequencer browser opens. Turn the blue encoder to ARPEGGIO, ENDLESS, FINGER, HOLD, PATTERN, SKETCH or TOMBOLA. Synth mode and drum mode each remember their own choice.",
    "enable_sequencer": "With the browser open (shift + sequencer key) and the type chosen with the blue encoder, tap the blue encoder or press the sequencer key again. The sequencer screen replaces the sound screen and it is now on: hold a key to hear it run.",
    "disable_sequencer": "Press the sequencer key on its own; each press toggles the selected sequencer off or on. Turn it off before a part that should record the bridge's notes exactly as sent.",
    "sequencer_parameters": "On the sequencer screen turn blue, ochre, gray and orange for the main page, and hold shift while turning for the shift page. No MIDI CC reaches these encoders (CC 46 to 61 stay on the sound pages), so set the values the model asks for by hand and tell it what the screen shows.",
    "arpeggio_setup": "Select and enable ARPEGGIO (shift + sequencer key, blue encoder, tap blue). Blue = note value (1/8, 1/16 ...), ochre = trigger mode (ALL or EACH), gray = trigger pattern, orange = hold. Hold shift: blue = note length, ochre = type (the arpeggio order), gray = pause / skip, orange = swing. Hold a chord to hear it.",
    "endless_setup": "Select and enable ENDLESS. With the sequencer stopped, play keys to enter steps (up to 999; the step count is on the screen), or let the model send them: its short notes enter steps the same way while the sequencer is stopped (verified in drum mode). Blue = note value, ochre = swing, gray = trigger pattern, orange = hold (OFF or HOLD). Hold shift: blue = manual mode, gray = rotate pattern, orange = direction. Press a key (or let the model hold a note) to play the steps transposed to it.",
    "finger_setup": "Select and enable FINGER (synth or drum). Blue = move cursor, ochre = swing, gray = pattern length, orange = hold. Hold shift: blue = erase notes, orange = play mode. The screen shows pattern players A and B and a JOIN control; each key plays a stored pattern in that key.",
    "hold_setup": "Select and enable HOLD. Blue = break point (the split marker on the keyboard graphic), ochre = mono / poly, gray = transpose, orange = hold; click the orange encoder to clear what is held. There is no shift page. Play a chord and it keeps sounding.",
    "pattern_setup": "Select and enable PATTERN. Blue = move cursor on the 16-column grid, ochre = swing, gray = pattern length, orange = hold. Hold shift: blue = erase notes, ochre = offset notes, gray = move section, orange = play mode. Place the notes on the grid, then play a key to run it.",
    "sketch_setup": "Select and enable SKETCH. Draw the curve on the grid with blue = draw x, ochre = draw y, gray = move x, orange = move y. Hold shift: blue = erase, ochre = use divider, gray = use grid, orange = hold. Hold a key and the curve is traced.",
    "tombola_setup": "Select and enable TOMBOLA. Blue = rotation speed, ochre = heaviness (g-force), gray = shape (the number of polygon sides), orange = bounciness. Hold shift: blue = manual mode; its effect on MIDI notes is unverified, so start with it off. Each key you hold drops a ball.",
    "clear_sequencer": "HOLD: click the orange encoder (clear). PATTERN and FINGER: hold shift and use the blue encoder (erase notes). SKETCH: hold shift and use the blue encoder (erase). ENDLESS: the manual lists no erase control; look for one on the device and tell the model what you find.",
    "sync_mode": "Open the tempo screen (the tempo key, marked with a metronome symbol). Blue sets the song BPM (tap tempo works too), ochre sets the sync mode: free, beat match, midi sync, PO sync; hold shift and turn ochre for 1/16 sync. Gray is tape speed, orange the metronome. For the bridge's clock choose midi sync and keep the BPM equal to the score tempo; choose free or beat match to ignore incoming clock.",
}


# ------------------------------------------------------------------ message helpers (no device access)

def bar_beat_to_beat(bar: int, beat: int = 1, beats_per_bar: int = 4) -> int:
    """1-based bar and beat, as a musician counts, to the 0-based beat index a Score uses: bar 2 beat 1 in 4/4 is beat 4."""
    if bar < 1 or beat < 1 or beats_per_bar < 1 or beat > beats_per_bar:
        raise ValueError(f"bar and beat are 1-based and beat must be within beats_per_bar (got bar {bar}, beat {beat}, {beats_per_bar} per bar)")
    return (bar - 1) * beats_per_bar + (beat - 1)


def song_position(beat: float, ticks_per_beat: int = CLOCK_TICKS_PER_BEAT) -> int:
    """MIDI song position for a beat: counts 16th notes (six clock ticks each) from the start. Beat 4 at 24 ticks per beat is 16."""
    if ticks_per_beat <= 0 or ticks_per_beat % TICKS_PER_SIXTEENTH:
        raise ValueError("ticks_per_beat must be a positive multiple of 6 (a 16th note is six MIDI clock ticks)")
    if beat < 0:
        raise ValueError("song position cannot be before the start")
    exact = beat * ticks_per_beat / TICKS_PER_SIXTEENTH
    pos = int(round(exact))
    if abs(exact - pos) > 1e-6:
        raise ValueError(f"beat {beat} is not on the 16th-note grid the song position pointer uses")
    if pos > MAX_SONG_POSITION:
        raise ValueError(f"song position {pos} exceeds the 14-bit maximum {MAX_SONG_POSITION} ({MAX_SONG_POSITION / 4:.0f} beats)")
    return pos


def song_position_message(beat: float, ticks_per_beat: int = CLOCK_TICKS_PER_BEAT) -> mido.Message:
    """The song position pointer (TE: 'set sequencer position') for a beat, as a mido message: bytes F2, LSB, MSB.
    Send it before the first note to start an endless or pattern sequence from that step (open question until tried)."""
    return mido.Message("songpos", pos=song_position(beat, ticks_per_beat))


def field_bpm_for_cc(value: int) -> int:
    """The song BPM the Field adopts for a CC 80 value (inverse of tape.tempo_cc_value): 0-5 = 40-50 in 2s, 6-120 = 52-166, 121-127 = 168-180 in 2s."""
    if not 0 <= value <= 127:
        raise ValueError("CC values are 0-127")
    if value <= 5:
        return 40 + 2 * value
    if value <= 120:
        return 46 + value
    return 168 + 2 * (value - 121)


def clock_plan(tempo: float, beats: float, ticks_per_beat: int = CLOCK_TICKS_PER_BEAT) -> dict[str, Any]:
    """How the bridge's clock lines up with the Field for a span of beats: tick timing, the CC 80 value the score sends first,
    the BPM the Field lands on, and the tape speed that mismatch would produce in midi sync mode."""
    if tempo <= 0 or beats < 0:
        raise ValueError("tempo must be positive and beats non-negative")
    tick = 60.0 / (tempo * ticks_per_beat)
    seconds = beats * 60.0 / tempo
    cc80 = tempo_cc_value(tempo)
    field_bpm = field_bpm_for_cc(cc80)
    speed = 100.0 * tempo / field_bpm
    plan: dict[str, Any] = {
        "tempo": tempo, "beats": beats, "seconds": seconds,
        "ticks_per_beat": ticks_per_beat, "tick_seconds": tick,
        "ticks": int(round(beats * ticks_per_beat)),
        "ticks_sent": int(seconds / tick) + 1,                   # what compile_events emits: t = 0 through t = end
        "sixteenths": beats * ticks_per_beat / TICKS_PER_SIXTEENTH,
        "song_position_at_end": int(round(beats * ticks_per_beat / TICKS_PER_SIXTEENTH)),
        "cc80": cc80, "field_bpm": field_bpm, "tape_speed_percent": round(speed, 2),
        "warning": "",
    }
    if abs(speed - 100.0) > 0.01:
        plan["warning"] = (f"CC 80 puts the Field at {field_bpm} BPM, not {tempo}; in midi sync mode the tape would run at "
                           f"{speed:.1f} percent and the take would play back off pitch. Use a tempo the Field offers: "
                           "40-50 in steps of 2, 52-166 in steps of 1, 168-180 in steps of 2.")
    return plan


def held_chords_score(chords: Sequence[Sequence[int | str]], beats: float | Sequence[float] = 4.0, tempo: float = 100.0,
                      velocity: int = 96, gap_beats: float = 0.0, tail_seconds: float = 1.5, send_clock: bool = True,
                      beats_per_bar: int = 4) -> Score:
    """A score that holds one chord after another, each for its beats: the material for the arpeggio, hold, finger and
    sketch sequencers. gap_beats shortens each held chord so the sequencer sees a release before the next one."""
    if not chords:
        raise ValueError("give at least one chord")
    durations = [float(beats)] * len(chords) if isinstance(beats, (int, float)) else [float(b) for b in beats]
    if len(durations) != len(chords):
        raise ValueError("one duration per chord, or a single number for all")
    if any(d <= 0 for d in durations) or gap_beats < 0:
        raise ValueError("durations must be positive and the gap non-negative")
    notes: list[Note] = []
    start = 0.0
    for chord, dur in zip(chords, durations):
        if not chord:
            raise ValueError("a chord needs at least one pitch")
        held = max(0.05, dur - gap_beats)
        notes.extend(Note(start=start, duration=held, pitch=p, velocity=velocity) for p in chord)
        start += dur
    return Score(tempo=tempo, beats_per_bar=beats_per_bar, notes=notes, tail_seconds=tail_seconds, send_clock=send_clock, length_beats=start)


def hold_chord_events(notes: Sequence[int | str], beats: float, tempo: float, velocity: int = 96, send_clock: bool = True) -> list[Event]:
    """Compiled events that hold one chord for N beats: note-ons at 0 s, note-offs at N * 60 / tempo s, and MIDI clock at 24
    ticks per beat from 0 s through the end when send_clock is on. Built through Score and compile_events, no humanize."""
    if not notes:
        raise ValueError("give at least one pitch")
    score = held_chords_score([list(notes)], beats, tempo, velocity=velocity, tail_seconds=0.0, send_clock=send_clock)
    return compile_events(score, humanize=False)


def ball_drop_score(pitches: Sequence[int | str], tempo: float = 100.0, beats: float = 16.0, stagger_beats: float = 1.0,
                    velocity: int = 96, tail_seconds: float = 4.0, send_clock: bool = True) -> Score:
    """A score for the tombola: each pitch is one ball, dropped stagger_beats after the previous one and held to the end
    of the span. At most SAFE_POLYPHONY balls, since every ball is a voice."""
    if not pitches:
        raise ValueError("give at least one pitch; each is one ball")
    if len(pitches) > SAFE_POLYPHONY:
        raise ValueError(f"{len(pitches)} balls would exceed the safe voice budget of {SAFE_POLYPHONY}")
    if beats <= 0 or stagger_beats < 0:
        raise ValueError("beats must be positive and the stagger non-negative")
    notes: list[Note] = []
    for i, p in enumerate(pitches):
        start = i * stagger_beats
        if start >= beats:
            raise ValueError(f"ball {i + 1} would drop at beat {start}, after the span of {beats} beats")
        notes.append(Note(start=start, duration=beats - start, pitch=p, velocity=velocity))
    return Score(tempo=tempo, notes=notes, tail_seconds=tail_seconds, send_clock=send_clock, length_beats=beats)


def arpeggio_notes(note_value: int, beats: float) -> float:
    """How many notes the arpeggiator plays from a chord held for `beats` at note value 1/note_value in 4/4."""
    if note_value <= 0 or beats < 0:
        raise ValueError("note_value must be positive (8 for 1/8) and beats non-negative")
    return beats * note_value / 4


def endless_pass_beats(steps: int, note_value: int) -> float:
    """Beats for one pass over an endless sequence of `steps` at note value 1/note_value."""
    if not 1 <= steps <= MAX_ENDLESS_STEPS:
        raise ValueError(f"endless holds 1 to {MAX_ENDLESS_STEPS} steps")
    if note_value <= 0:
        raise ValueError("note_value must be positive (16 for 1/16)")
    return steps * 4 / note_value


def endless_pass_seconds(steps: int, note_value: int, tempo: float) -> float:
    if tempo <= 0:
        raise ValueError("tempo must be positive")
    return endless_pass_beats(steps, note_value) * 60.0 / tempo


# ------------------------------------------------------------------ the guide, served through get_guide("sequencers")

SEQUENCER_GUIDE = """The OP-1 field's sequencers and how to use them through this bridge.

The Field has seven: ARPEGGIO, ENDLESS, FINGER, HOLD, PATTERN, SKETCH and TOMBOLA. They store note data, not
audio, so the sound can change while the notes run. Each mode has its own sequencer memory; one plays at a
time. The song BPM drives every sequencer, the tempo-synced LFOs and the tape.

Encoders in blue, ochre, gray, orange order, the shift page after "shift:":
- ARPEGGIO: note value, trigger mode (ALL, EACH), trigger pattern, hold; shift: note length, type, pause-skip,
  swing; a chord held N beats at 1/n yields N*n/4 notes.
- ENDLESS: note value, swing, trigger pattern, hold; shift: manual mode, -, rotate pattern, direction; up to 999
  steps, played transposed by a key, 4/n beats each at 1/n.
- FINGER (synth and drum): move cursor, swing, pattern length, hold; shift: erase notes, -, -, play mode; each
  held key plays a stored pattern, transposed.
- HOLD: break point, mono-poly, transpose, hold; clicking orange clears; no shift page; sustains the chord you play.
- PATTERN: move cursor, swing, pattern length, hold; shift: erase notes, offset notes, move section, play mode; a
  16-column grid the human fills, played by keys.
- SKETCH: draw x, draw y, move x, move y; shift: erase, use divider, use grid, hold; a drawn curve traced while a
  key is held.
- TOMBOLA: rotation speed, heaviness, shape (polygon sides), bounciness; shift: manual mode; each held key drops
  a ball, every bounce retriggers the note.

What the bridge drives over MIDI (evidence per sequencer in list_sequencers):
- Notes on the Field's channel. ENDLESS is verified (2026-09-26, drum mode): short incoming notes entered steps
  while stopped and a held note played them back transposed. Arpeggio and hold accept external notes with
  velocity (changelog 1.2.5, 1.2.7); finger, pattern, sketch and tombola follow by analogy: confirm with one
  held note first.
- Tempo: set_tempo sends CC 80 (40-180 BPM). A score with send_clock sends CC 80 for its tempo, then MIDI clock
  at 24 ticks per beat (play, record_to_tape and hold_chord; the latter returns the clock plan). In midi sync
  mode the Field scales the tape speed to the clock relative to its song BPM: use a BPM the Field offers (40-50
  in 2s, 52-166, 168-180 in 2s); free and beat match ignore clock. Guided mode: send_clock needs
  allow_transport, and allow_tempo_change unless the score is on the fixed tempo.
- Transport: MIDI start, continue and stop are tape transport (tape midi_start, midi_stop, midi_continue). The
  song position pointer (TE: "set sequencer position") counts 16th notes, beat 4 in 4/4 being position 16; the
  song_position tool sends it.
- Tape: an armed take starts on the first note; record_to_tape stops the tape. With a sequencer running
  notes_heard means nothing (the audible notes are not the sent ones): judge it by level, clipping, spectral
  movement and spectrogram.

Only the human can (ask with human_steps): select and enable a sequencer (shift + sequencer key, turn the
blue encoder, tap blue or press the sequencer key again), toggle it off (the sequencer key alone), turn its
encoders and shift page, place pattern notes, draw the sketch curve, edit finger patterns, clear hold (click
orange), choose the sync mode (ochre on the tempo screen), select the tape track and arm recording (shift +
record). Endless steps can also be the bridge's short notes while it is stopped. Sequencer content is not a
file: disk mode exposes synth/ and drum/ only.

Recipes, each ending with a take on tape (validate_score first; at most six held notes; the human selects a track
and arms recording before each). Try a chord with hold_chord (it returns its score); commit with record_to_tape:
1. Arpeggiated pad. Human: enable ARPEGGIO, note value 1/16, trigger mode ALL, hold off. Model: record_to_tape
   held chords, each chord's notes sharing start and duration (C4, E4, G4 at 0; A3, C4, E4 at 4; duration 3.8 so
   the sequencer sees a release; "length_beats" 8).
2. Hold chord bed. Human: enable HOLD, poly, break point at the bottom. Model: short chords and a long tail; the
   sequencer should sustain each until the next (verify on the first take).
3. Tombola texture. Human: enable TOMBOLA, manual mode off, set speed, heaviness, shape, bounciness. Model:
   record_to_tape up to six long staggered notes, one ball each, held to the end (C3 at 0 for 16 beats, G3 at 1
   for 15, C4 at 2 for 14; "length_beats" 16, "tail_seconds" 4), with automation.
4. Endless or pattern loop. Human: enable ENDLESS or PATTERN, enter the steps or grid, set note value and
   direction, leave it stopped (for ENDLESS the model can send the steps as short notes while stopped, hold off:
   verified in drum mode). Model: one long root note; the sequence plays transposed while it lasts (hold latches
   it), a pass of S steps at 1/n being S*4/n beats.
5. Finger or sketch phrase. Human: enable FINGER (edit patterns) or SKETCH (draw the curve). Model: one note per
   phrase, as long as the phrase.
6. Capture a human's sequence: the human plays with a sequencer on; capture_seed keeps audio and spectrogram,
   note data only if the Field transmits sequencer notes (unverified).

Caveats: a trigger is expected to restart the sequencer's phase (unverified; the endless test used one held note):
start phrases on the beat and check the first take's onsets. Disable the sequencer before recording a plain part.
"""

__all__ = [
    "CLOCK_TICKS_PER_BEAT", "TICKS_PER_SIXTEENTH", "MAX_SONG_POSITION", "MAX_ENDLESS_STEPS", "SAFE_POLYPHONY",
    "SEQUENCER_NAMES", "SEQUENCER_GUIDE", "HUMAN_STEPS",
    "sequencer_catalog", "sequencer_info",
    "bar_beat_to_beat", "song_position", "song_position_message", "field_bpm_for_cc", "clock_plan",
    "held_chords_score", "hold_chord_events", "ball_drop_score",
    "arpeggio_notes", "endless_pass_beats", "endless_pass_seconds",
]
