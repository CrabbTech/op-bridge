"""audition_slots and the upgraded vocoder tool on the connected Field."""
import os
import time

import pytest

from op_bridge import server as srv
from op_bridge.session import Config, read_state
from tests.conftest import needs_field


def _restore(before):
    try:
        if before.get("mode") == "synth" and before.get("synth_slot"):
            srv.select_sound("synth", int(before["synth_slot"]))
        elif before.get("mode") == "drum" and before.get("drum_slot"):
            srv.select_sound("drum", int(before["drum_slot"]))
    except Exception:
        pass


@needs_field
def test_audition_slots_hears_two_slots_in_one_call():
    before = read_state()
    try:
        out = srv.audition_slots(["synth 1", "synth 3"], pitches=["C4"], seconds=1.0, wait=True)
    finally:
        _restore(before)
    rows = out["auditioned"]
    assert [r["slot"] for r in rows] == ["synth 1", "synth 3"]
    heard = [r for r in rows if r["heard"]]
    assert len(heard) >= 1, rows                      # a slot may be a silent or one-shot sound, but not both
    for r in heard:
        assert r["id"] and r["peak_db"] is not None and "notes_heard" in r
    print(rows)


@needs_field
def test_audition_slots_long_list_is_a_job():
    before = read_state()
    out = srv.audition_slots(["synth 1", "synth 2", "synth 3"], seconds=1.0, wait=False)
    assert out["status"] == "running" and out["job_id"]
    t0 = time.time()
    while time.time() - t0 < 60:
        st = srv.job_status(out["job_id"])
        if st["status"] != "running":
            break
        time.sleep(1.0)
    _restore(before)
    assert st["status"] == "done", st
    assert len(st["result"]["auditioned"]) == 3


@needs_field
def test_vocoder_lines_are_placed_and_chords_held():
    """Runs the upgraded tool on the loaded sound; skips when nothing vocoded comes back (input not on usb audio,
    input off, or no vocoder preset loaded), so it never fails on a setup the human has not made."""
    cfg = Config.load()
    st = read_state()
    slot = int(st.get("synth_slot") or 0) or None
    before = dict(st)
    try:
        out = srv.speak_through_vocoder(lines=[{"beat": 0, "text": "one two"}, {"beat": 2, "text": "three four"}], chords=[["C3", "G3", "C4"], ["C3", "G3", "C4"], ["A2", "E3", "A3"]],
                                        beats_per_chord=1.0, tempo=120.0, min_beats=4.0, hold_chords=True, record=True, name="voc-lines-test", wait=True)
    finally:
        _restore(before)
    assert out["spoken"] is True
    assert [l["beat"] for l in out["lines"]] == [0.0, 2.0]
    assert out["chord_beats"][0] == [0.0, 2.0], out["chord_beats"]      # two identical chords held as one
    assert sum(d for _, d in out["chord_beats"]) >= 4.0
    assert os.path.exists(out["speech_wav"]) and out["take_id"]
    a = out["analysis"]
    if a.get("main_mix", {}).get("peak_db", a.get("peak_db", -120)) < -50:
        pytest.skip("nothing came back from the Field: the vocoder path is not set up (input usb audio, input on, vocoder preset)")
    print(out.get("intelligibility"), a)
    assert "intelligibility" in out


@needs_field
def test_stream_to_tape_streams_and_reports_alignment():
    """start='now' needs no armed tape: the Mac's audio is streamed into the USB input and whatever the Field
    returns is kept. Skips when the input is off or not on usb audio (nothing comes back)."""
    out = srv.stream_to_tape(text="testing the input path, one two three", start="now", record=True, name="input-test", wait=True)
    assert out["streamed"] is True and out["take_id"] and os.path.exists(out["wav"]) and out["count_in_seconds"] == 0
    a = out["analysis"]
    if not a.get("live_usb_channels") or (a.get("peak_db") or -120) < -50:
        pytest.skip("the Field returned nothing: input off or not on usb audio")
    print(a)
    assert a["alignment"]["stoi"] is not None and abs(a["alignment"]["lag_ms"]) < 400, a["alignment"]
