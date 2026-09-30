"""The sequencer module: guide text, human steps, catalog and the message helpers. Real code paths, no device."""
import json

import mido
import pytest

from op_bridge import sequencers as SQ
from op_bridge.score import Event, Note, Score, compile_events, max_polyphony, validate
from op_bridge.session import Config
from op_bridge.tape import tempo_cc_value

SEVEN = ("arpeggio", "endless", "finger", "hold", "pattern", "sketch", "tombola")


def test_guide_covers_every_sequencer_and_the_human_only_steps():
    g = SQ.SEQUENCER_GUIDE.lower()
    for name in SEVEN:
        assert name in g, name
    words = len(SQ.SEQUENCER_GUIDE.split())
    assert 500 <= words <= 900, words
    # the human-only gestures with their key presses, and the pointer to human_steps
    for phrase in ("shift", "sequencer key", "blue encoder", "human_steps", "arm", "sync mode", "disk mode"):
        assert phrase in g, phrase
    # what the bridge drives, and the caveat about take analysis
    for phrase in ("cc 80", "24 ticks", "song position", "16th notes", "midi_start", "record_to_tape", "notes_heard", "capture_seed"):
        assert phrase in g, phrase
    assert "recipes" in g and g.count("tape") >= 6


def test_human_steps_give_exact_key_presses():
    required = {"select_sequencer", "enable_sequencer", "disable_sequencer", "sequencer_parameters", "arpeggio_setup",
                "endless_setup", "finger_setup", "hold_setup", "pattern_setup", "sketch_setup", "tombola_setup", "sync_mode"}
    assert required <= set(SQ.HUMAN_STEPS)
    for k, v in SQ.HUMAN_STEPS.items():
        assert isinstance(v, str) and len(v) > 60, k
        assert "%" not in v, f"{k}: human_steps in server.py formats entries with %d, a stray % would break it"
    assert "shift" in SQ.HUMAN_STEPS["select_sequencer"] and "sequencer key" in SQ.HUMAN_STEPS["select_sequencer"]
    assert "blue encoder" in SQ.HUMAN_STEPS["select_sequencer"] and "tap the blue encoder" in SQ.HUMAN_STEPS["enable_sequencer"]
    assert "toggles" in SQ.HUMAN_STEPS["disable_sequencer"]
    assert "orange" in SQ.HUMAN_STEPS["hold_setup"] and "clear" in SQ.HUMAN_STEPS["hold_setup"]
    assert "ochre" in SQ.HUMAN_STEPS["sync_mode"] and "midi sync" in SQ.HUMAN_STEPS["sync_mode"]
    # every setup entry names the sequencer in capitals as the browser shows it and all four encoder colours
    for name in SEVEN:
        step = SQ.HUMAN_STEPS[f"{name}_setup"]
        assert name.upper() in step
        for colour in ("blue", "ochre", "gray", "orange"):
            assert colour in step.lower(), (name, colour)


def test_catalog_is_structured_and_serialisable():
    cat = SQ.sequencer_catalog()
    assert [s["name"] for s in cat] == list(SEVEN)
    for s in cat:
        assert list(s["encoders"]) == ["blue", "ochre", "gray", "orange"], s["name"]
        assert all(s["encoders"].values()), s["name"]
        assert s["shift"] is None or list(s["shift"]) == ["blue", "ochre", "gray", "orange"]
        assert isinstance(s["midi_driveable"], bool) and s["drives_by"] and s["midi_evidence"] and s["manual_page"] in range(47, 52)
    by = {s["name"]: s for s in cat}
    assert by["hold"]["shift"] is None and by["hold"]["encoders"]["blue"] == "break point"
    assert by["arpeggio"]["encoders"] == {"blue": "note value", "ochre": "trigger mode", "gray": "trigger pattern", "orange": "hold"}
    assert by["arpeggio"]["shift"]["orange"] == "swing" and by["endless"]["shift"]["ochre"] is None
    assert by["tombola"]["encoders"]["ochre"].startswith("heaviness") and by["tombola"]["shift"]["blue"] == "manual mode"
    assert by["sketch"]["shift"]["orange"] == "hold" and by["pattern"]["shift"]["ochre"] == "offset notes"
    assert "changelog" in by["arpeggio"]["midi_evidence"] and "unverified" in by["tombola"]["midi_evidence"]
    json.dumps(cat)
    # copies: a caller cannot corrupt the module data
    cat[0]["encoders"]["blue"] = "x"
    assert SQ.sequencer_catalog()[0]["encoders"]["blue"] == "note value"
    assert SQ.sequencer_info("Hold")["name"] == "hold"
    with pytest.raises(ValueError):
        SQ.sequencer_info("grid")


def test_song_position_counts_sixteenths_from_the_start():
    m = SQ.song_position_message(4)
    assert isinstance(m, mido.Message) and m.type == "songpos" and m.pos == 16
    assert m.bytes() == [0xF2, 16, 0]
    assert SQ.song_position_message(0).pos == 0
    assert SQ.song_position_message(0.25).pos == 1          # one 16th note
    assert SQ.song_position_message(2, ticks_per_beat=48).pos == 16
    assert SQ.song_position_message(SQ.bar_beat_to_beat(2, 1, 4)).pos == 16
    assert SQ.song_position_message(SQ.bar_beat_to_beat(3, 2, 3)).pos == (6 + 1) * 4
    big = SQ.song_position_message(1000)                    # position 4000 needs the LSB/MSB split
    assert big.pos == 4000 and big.bytes() == [0xF2, 4000 & 0x7F, 4000 >> 7]
    assert SQ.song_position_message(SQ.MAX_SONG_POSITION / 4).pos == 16383
    for bad in (-1, 0.1, 5000):
        with pytest.raises(ValueError):
            SQ.song_position_message(bad)
    with pytest.raises(ValueError):
        SQ.song_position_message(1, ticks_per_beat=20)
    with pytest.raises(ValueError):
        SQ.bar_beat_to_beat(0, 1)
    with pytest.raises(ValueError):
        SQ.bar_beat_to_beat(1, 5, 4)


def test_clock_plan_matches_the_score_compiler():
    plan = SQ.clock_plan(120, 4)
    assert plan["ticks_per_beat"] == 24 and plan["ticks"] == 96 and plan["seconds"] == 2.0
    assert abs(plan["tick_seconds"] - 60 / (120 * 24)) < 1e-12
    assert plan["cc80"] == tempo_cc_value(120) == 74 and plan["field_bpm"] == 120 and plan["tape_speed_percent"] == 100
    assert plan["sixteenths"] == 16 and plan["song_position_at_end"] == 16 and plan["warning"] == ""
    # what compile_events sends over the same span
    s = Score(tempo=120, notes=[Note(start=0, duration=4, pitch=60)], tail_seconds=0)
    clocks = [e for e in compile_events(s) if e.kind == "clock"]
    assert len(clocks) == plan["ticks_sent"] == 97
    # a tempo the Field cannot hit exactly is flagged, with the speed the tape would run at
    odd = SQ.clock_plan(97.5, 4)
    assert odd["field_bpm"] == 98 and odd["tape_speed_percent"] == round(100 * 97.5 / 98, 2) and "98 BPM" in odd["warning"]
    assert SQ.clock_plan(30, 4)["field_bpm"] == 40 and SQ.clock_plan(30, 4)["warning"]
    with pytest.raises(ValueError):
        SQ.clock_plan(0, 4)


def test_cc80_round_trips_every_bpm_the_field_offers():
    offered = list(range(40, 51, 2)) + list(range(52, 167)) + list(range(168, 181, 2))
    for bpm in offered:
        assert SQ.field_bpm_for_cc(tempo_cc_value(bpm)) == bpm, bpm
    assert SQ.field_bpm_for_cc(0) == 40 and SQ.field_bpm_for_cc(127) == 180
    with pytest.raises(ValueError):
        SQ.field_bpm_for_cc(128)


def test_hold_chord_events_hold_the_chord_for_n_beats_with_clock():
    ev = SQ.hold_chord_events(["C4", "E4", "G4"], beats=4, tempo=120)
    assert ev and all(isinstance(e, Event) for e in ev)
    ons = [e for e in ev if e.kind == "note_on"]
    offs = [e for e in ev if e.kind == "note_off"]
    assert [e.a for e in ons] == [60, 64, 67] and all(e.t == 0 and e.b == 96 for e in ons)
    assert [e.a for e in offs] == [60, 64, 67] and all(abs(e.t - 2.0) < 1e-9 for e in offs)
    clocks = [e for e in ev if e.kind == "clock"]
    assert len(clocks) == 4 * 24 + 1 and clocks[0].t == 0 and abs(clocks[-1].t - 2.0) < 1e-9
    gaps = {round(b.t - a.t, 9) for a, b in zip(clocks, clocks[1:])}
    assert gaps == {round(60 / (120 * 24), 9)}
    assert {e.kind for e in ev} == {"note_on", "note_off", "clock"}
    assert [e.t for e in ev] == sorted(e.t for e in ev)
    # velocity and clock are options; an empty chord is refused
    quiet = SQ.hold_chord_events([60], 1, 120, velocity=40, send_clock=False)
    assert [(e.kind, e.a, e.b, e.t) for e in quiet] == [("note_on", 60, 40, 0.0), ("note_off", 60, 0, 0.5)]
    with pytest.raises(ValueError):
        SQ.hold_chord_events([], 4, 120)


def test_held_chords_score_is_a_valid_score_for_record_to_tape():
    s = SQ.held_chords_score([["C4", "E4", "G4"], ["A3", "C4", "E4"]], beats=4, tempo=96)
    assert isinstance(s, Score) and s.tempo == 96 and s.send_clock
    assert [n.start for n in s.notes] == [0, 0, 0, 4, 4, 4] and all(n.duration == 4 for n in s.notes)
    assert max_polyphony(s) == 3 and validate(s, Config()) == []
    assert s.end_beat() == 8 and abs(s.seconds() - (8 * 60 / 96 + s.tail_seconds)) < 1e-9
    js = json.loads(s.model_dump_json())      # the shape a tool hands back to the model
    assert js["notes"][3]["pitch"] == "A3" and js["length_beats"] == 8
    # per-chord durations and a gap so the sequencer sees a release between chords
    s2 = SQ.held_chords_score([[60], [62]], beats=[2, 6], tempo=120, gap_beats=0.5)
    assert [(n.start, n.duration) for n in s2.notes] == [(0, 1.5), (2, 5.5)]
    for bad in ({"chords": []}, {"chords": [[60]], "beats": [1, 2]}, {"chords": [[]]}, {"chords": [[60]], "beats": 0}):
        with pytest.raises(ValueError):
            SQ.held_chords_score(**bad)


def test_ball_drop_score_staggers_balls_and_holds_them_to_the_end():
    s = SQ.ball_drop_score(["C3", "G3", "C4", "E4"], tempo=120, beats=8, stagger_beats=1)
    assert [n.start for n in s.notes] == [0, 1, 2, 3] and [n.start + n.duration for n in s.notes] == [8] * 4
    assert max_polyphony(s) == 4 and validate(s, Config()) == [] and s.length_beats == 8
    ev = compile_events(s, humanize=False)
    assert [e.t for e in ev if e.kind == "note_on"] == [0.0, 0.5, 1.0, 1.5]
    with pytest.raises(ValueError):
        SQ.ball_drop_score([48, 50, 52, 53, 55, 57, 59], tempo=120, beats=8)      # seven balls, seven voices
    with pytest.raises(ValueError):
        SQ.ball_drop_score([48, 50], tempo=120, beats=1, stagger_beats=2)          # second ball after the span


def test_arpeggio_and_endless_arithmetic():
    assert SQ.arpeggio_notes(16, beats=4) == 16 and SQ.arpeggio_notes(8, beats=2) == 4
    assert SQ.endless_pass_beats(128, 16) == 32
    assert SQ.endless_pass_seconds(128, 16, tempo=120) == 16
    assert SQ.endless_pass_seconds(SQ.MAX_ENDLESS_STEPS, 4, tempo=60) == 999
    for bad in ((0, 16), (1000, 16), (8, 0)):
        with pytest.raises(ValueError):
            SQ.endless_pass_beats(*bad)
