"""Plays real notes on the connected OP-1 field and checks the recording."""
import os
import time

import pytest

from op_bridge.device import Field
from op_bridge.player import play_score, save_take
from op_bridge.score import Score
from tests.conftest import needs_field
from op_bridge.session import read_state


def restore_field(f):
    """Put the Field back to what the bridge last knew before the test changed it."""
    st = read_state()
    if st.get("mode") == "drum" and st.get("drum_slot"):
        f.select_drum_slot(int(st["drum_slot"]))
    elif st.get("mode") == "synth" and st.get("synth_slot"):
        f.select_synth_slot(int(st["synth_slot"]))
    elif st.get("mode") in ("drum", "synth"):
        f.set_mode(st["mode"])


@needs_field
def test_phrase_is_heard_in_time(tmp_path):
    score = Score(tempo=120, notes=[
        {"start": 0, "duration": 0.9, "pitch": "C4", "velocity": 100},
        {"start": 1, "duration": 0.9, "pitch": "E4", "velocity": 90},
        {"start": 2, "duration": 0.9, "pitch": "G4", "velocity": 80},
        {"start": 3, "duration": 0.9, "pitch": "C5", "velocity": 110},
    ], tail_seconds=1.0)
    before = read_state()
    with Field() as f:
        f.set_mode("synth"); time.sleep(0.1)
        f.select_synth_slot(1); time.sleep(0.3)
        res = play_score(f, score, record=True)
        from op_bridge.session import update_state
        update_state(**{k: v for k, v in before.items() if k in ("mode", "synth_slot", "drum_slot")})
        restore_field(f)
    a = res.analysis
    print(a)
    assert res.live_pair is not None, a.get("hint")
    assert a["notes_heard"] >= 3, a
    offsets = [n["onset_offset_ms"] for n in a["notes"] if n["onset_offset_ms"] is not None]
    assert offsets, "no onsets found"
    assert all(abs(o) < 40 for o in offsets), offsets
    wav = save_take(res, str(tmp_path / "take.wav"))
    assert os.path.getsize(wav) > 100000


@needs_field
def test_cc_automation_moves_the_sound():
    score = Score(tempo=120, notes=[{"start": 0, "duration": 4, "pitch": "A3", "velocity": 100}],
                  cc=[{"param": "engine1", "points": [[0, 0], [2, 127], [4, 0]]}], tail_seconds=0.5)
    before = read_state()
    with Field() as f:
        f.set_mode("synth"); time.sleep(0.1)
        f.select_synth_slot(1); time.sleep(0.3)
        res = play_score(f, score, record=True)
        f.select_synth_slot(1)  # reload the saved preset, undoing the sweep
        from op_bridge.session import update_state
        update_state(**{k: v for k, v in before.items() if k in ("mode", "synth_slot", "drum_slot")})
        restore_field(f)
    c = res.analysis["spectral_centroid_hz"]
    print(res.analysis["spectral_centroid_hz"], res.analysis["peak_db"])
    assert c["max"] is not None and c["min"] is not None
    assert c["max"] - c["min"] > 50, c
