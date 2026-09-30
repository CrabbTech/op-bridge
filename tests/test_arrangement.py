"""The node-based arrangement compiles to one score per part with sections in path order."""
import pytest

from op_bridge.arrangement import compile_arrangement
from op_bridge.score import compile_events

DOC = {
    "tempo": 120, "beats_per_bar": 4,
    "nodes": {
        "intro": {"type": "section", "bars": 1},
        "A": {"type": "section", "bars": 2},
        "B": {"type": "section", "bars": 1},
        "keys": {"type": "part", "sound": {"kind": "synth", "slot": 2}, "track": 2,
                 "clips": {"intro": {"chords": [{"notes": ["C3", "E3", "G3"], "beats": 4}]},
                           "A": {"chords": [{"notes": ["F3", "A3"], "beats": 2, "rhythm": "x.x."}, {"notes": ["G3", "B3"], "beats": 2}]},
                           "B": {"as": "intro", "transpose": 12}}},
        "drums": {"type": "part", "sound": {"kind": "drum", "slot": 1}, "track": 1,
                  "clips": {"A": {"pattern": {"kit_map": {"BD": 53, "CH": 60}, "patterns": {"a": {"BD": "x.......", "CH": "x.x.x.x."}}, "steps_per_bar": 8}}}},
        "sweep": {"type": "automation", "part": "keys", "param": "fx3", "sections": {"A": [[0, 20], [8, 100]]}},
    },
    "edges": [["intro", "A"], ["A", "B"], ["B", "A"]],
}


def test_edges_walk_and_sections_place_clips_in_order():
    out = compile_arrangement(DOC)
    assert out["order"] == ["intro", "A", "B", "A"] and out["total_bars"] == 6 and out["seconds"] == 12.0
    keys = out["parts"]["keys"].score
    starts = sorted({n.start for n in keys.notes})
    assert starts[0] == 0.0                                   # intro chord at the top
    assert any(abs(s - 4.0) < 1e-9 for s in starts)           # A starts at bar 2
    b_notes = [n for n in keys.notes if 12.0 <= n.start < 16.0]
    assert b_notes and {n.midi for n in b_notes} == {60, 64, 67}   # B reuses the intro an octave up
    assert keys.cc and keys.cc[0].param == "fx3" and keys.cc[0].points[0][0] == 4.0 and keys.cc[0].points[-1][0] == 24.0
    drums = out["parts"]["drums"].score
    assert drums.notes and all(4.0 <= n.start < 24.0 for n in drums.notes)   # only during the two A passes
    assert len([n for n in drums.notes if n.midi == 53]) == 4                # one bar clip looped over 2 bars, twice
    assert "keys" in out["timeline"] and "drums" in out["timeline"]
    assert len(compile_events(keys)) > 0


def test_clip_longer_than_its_section_is_an_error():
    bad = dict(DOC, nodes=dict(DOC["nodes"], intro={"type": "section", "bars": 1}))
    bad["nodes"]["keys"] = dict(DOC["nodes"]["keys"], clips=dict(DOC["nodes"]["keys"]["clips"], intro={"chords": [{"notes": ["C3"], "beats": 8}]}))
    with pytest.raises(ValueError, match="section 'intro'"):
        compile_arrangement(bad)


def test_unknown_section_in_order_is_a_helpful_error():
    with pytest.raises(ValueError, match="not section nodes"):
        compile_arrangement(dict(DOC, edges=None, order=["intro", "Z"]))
