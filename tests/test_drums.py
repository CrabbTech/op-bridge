"""The drum grid notation: parse, compile to a Score, render back, summarize. Real code paths and real kit files, no device."""
import json
import os

import pytest

from op_bridge import drums as D
from op_bridge.score import Score, compile_events, validate
from op_bridge.session import Config

KIT_JSON = os.path.join(os.path.dirname(__file__), "fixtures", "kit1-owner-labels.json")
FIELD_KITS = os.path.expanduser("~/Music/op-bridge/field-backup/2026-09-26-disk/drum/user")

# the introductory example of the drum pattern books: kick on 1 and 3, snare on 2 and 4, closed hats on every 8th
BOOK_INTRO = {
    "tempo": 100,
    "kit_map": {"BD": 53, "SN": 55, "CH": 60},
    "patterns": {"A": {"BD": "x.......x.......", "SN": "....x.......x...", "CH": "x.x.x.x.x.x.x.x."}},
}


def note_ons(score: Score) -> list[tuple[float, int, int]]:
    """(seconds, note, velocity) of every note_on in compiled order."""
    return [(round(e.t, 6), e.a, e.b) for e in compile_events(score, humanize=False) if e.kind == "note_on"]


def starts(score: Score, note: int) -> list[float]:
    return [n.start for n in score.notes if n.midi == note]


# ---------------------------------------------------------------------------------------------------------------- the example

def test_book_intro_compiles_to_the_expected_beats_and_events():
    score = D.compile_drums(BOOK_INTRO)
    assert score.tempo == 100 and score.beats_per_bar == 4 and score.send_clock is True
    assert score.length_beats == 4.0
    assert starts(score, 53) == [0.0, 2.0]
    assert starts(score, 55) == [1.0, 3.0]
    assert starts(score, 60) == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    for n in score.notes:
        assert n.duration == 0.125 and n.velocity == 100          # half a 16th step at the default hit velocity

    spb = 60 / 100
    expected = []
    for step in range(16):
        beat = step / 4
        t = round(beat * spb, 6)
        if step in (0, 8):
            expected.append((t, 53, 100))
        if step in (4, 12):
            expected.append((t, 55, 100))
        if step % 2 == 0:
            expected.append((t, 60, 100))
    ons = note_ons(score)
    assert len(ons) == len(expected) == 12
    assert [t for t, _, _ in ons] == sorted(t for t, _, _ in ons)                  # in time order
    assert sorted(ons) == sorted(expected)                                        # exactly those note_ons
    # simultaneous hits share a time to the microsecond; nothing else is a note_on
    assert {t for t, _, _ in ons} == {round(k * 0.5 * spb, 6) for k in range(8)}
    # and the score passes the device validator in a free session
    assert validate(score, Config()) == []


def test_parse_gives_a_validated_document():
    d = D.parse_pattern(BOOK_INTRO)
    assert isinstance(d, D.DrumDocument)
    assert d.steps_per_bar == 16 and d.beats_per_bar == 4 and d.step_beats == 0.25
    assert d.arrangement == ["A"] and d.total_bars == 1 and d.total_steps == 16
    a = d.patterns["A"]
    assert a.bars == 1 and a.steps == 16
    assert a.rows["BD"].note == 53 and a.rows["BD"].cells == "x.......x......." and a.rows["BD"].hits == 2
    assert a.rows["CH"].hits == 8
    assert d.velocities == {"hit": 100, "accent": 120, "ghost": 60}


# ------------------------------------------------------------------------------------------------------------- hit types

def test_accents_raise_and_ghosts_lower_velocity():
    doc = {"kit_map": {"SN": 55}, "patterns": {"A": {"SN": "x.X.o.^.........",}}}
    score = D.compile_drums(doc)
    assert [(n.start, n.velocity) for n in score.notes] == [(0.0, 100), (0.5, 120), (1.0, 60), (1.5, 120)]
    custom = dict(doc, velocities={"hit": 90, "accent": 127, "ghost": 40})
    assert [n.velocity for n in D.compile_drums(custom).notes] == [90, 127, 40, 127]


def test_ratchet_step_yields_ratchet_hits_note_ons_inside_the_step():
    doc = {"tempo": 120, "kit_map": {"SN": 55}, "patterns": {"A": {"SN": "....r..........."}}, "ratchet_hits": 3}
    score = D.compile_drums(doc)
    assert len(score.notes) == 3
    step = 0.25
    assert [round(n.start, 6) for n in score.notes] == [1.0, round(1 + step / 3, 6), round(1 + 2 * step / 3, 6)]
    assert all(n.start < 1.25 for n in score.notes)
    assert all(n.duration <= step / 3 + 1e-9 for n in score.notes)           # sub-hits never overlap
    ons = note_ons(score)
    assert len(ons) == 3 and all(a == 55 and v == 100 for _, a, v in ons)
    assert ons[0][0] < ons[1][0] < ons[2][0] < round(1.25 * 0.5, 6)
    # the default is two sub-hits, and R is an accented ratchet
    two = D.compile_drums({"kit_map": {"SN": 55}, "patterns": {"A": {"SN": "R..............."}}})
    assert [(n.start, n.velocity) for n in two.notes] == [(0.0, 120), (0.125, 120)]
    # list form: "5r" and "5^r"
    lst = D.compile_drums({"kit_map": {"SN": 55}, "patterns": {"A": {"SN": ["1r", "3^r"]}}, "ratchet_hits": 2})
    assert [(n.start, n.velocity) for n in lst.notes] == [(0.0, 100), (0.125, 100), (0.5, 120), (0.625, 120)]


def test_ties_are_rests_and_spaces_and_bars_are_ignored():
    doc = {"kit_map": {"BD": 53}, "patterns": {"A": {"BD": "x--- | x... | x-.- | x..."}}}
    assert starts(D.compile_drums(doc), 53) == [0.0, 1.0, 2.0, 3.0]


# ------------------------------------------------------------------------------------------------------ arrangement, forms

def test_arrangement_places_b_after_a_in_every_form():
    doc = {
        "kit_map": {"BD": 53, "SN": 55},
        "patterns": {"A": {"BD": "x.......x......."}, "B": {"SN": "....x.......x..."}},
        "arrangement": "AB",
    }
    score = D.compile_drums(doc)
    assert starts(score, 53) == [0.0, 2.0]
    assert starts(score, 55) == [5.0, 7.0]
    assert score.length_beats == 8.0
    ons = note_ons(score)
    assert [a for _, a, _ in ons] == [53, 53, 55, 55]
    assert D.compile_drums(dict(doc, arrangement=["A", "B"])).notes == score.notes
    assert D.compile_drums(dict(doc, arrangement="A, B")).notes == score.notes
    assert D.compile_drums(dict(doc, arrangement="A B")).notes == score.notes
    # default: each pattern once in order; repeats with "A*2"
    assert D.compile_drums({k: v for k, v in doc.items() if k != "arrangement"}).notes == score.notes
    rep = D.compile_drums(dict(doc, arrangement="A*2 B"))
    assert starts(rep, 53) == [0.0, 2.0, 4.0, 6.0] and starts(rep, 55) == [9.0, 11.0] and rep.length_beats == 12.0
    assert D.compile_drums(dict(doc, arrangement=["A*2", "B"])).notes == rep.notes
    # a list-form-only pattern in the arrangement counts its own bars
    ba = D.compile_drums(dict(doc, arrangement="BA"))
    assert starts(ba, 55) == [1.0, 3.0] and starts(ba, 53) == [4.0, 6.0]


def test_list_rows_equal_string_rows():
    strings = {"kit_map": {"BD": 53, "SN": 55, "OH": 63},
               "patterns": {"B": {"BD": "x.......x.x.....", "SN": "....X.......x...", "OH": "...............x"}}}
    lists = {"kit_map": {"BD": 53, "SN": 55, "OH": 63},
             "patterns": {"B": {"BD": [1, 9, 11], "SN": ["5^", 13], "OH": [16]}}}
    assert D.compile_drums(lists).notes == D.compile_drums(strings).notes
    assert D.compile_drums(lists).length_beats == D.compile_drums(strings).length_beats == 4.0
    # mixed: a list row inside a two-bar string pattern is stretched to the pattern's length
    mixed = {"kit_map": {"BD": 53, "SN": 55}, "patterns": {"A": {"BD": "x...x...x...x...x...x...x...x...", "SN": [5, 13, 21, 29]}}}
    assert starts(D.compile_drums(mixed), 55) == [1.0, 3.0, 5.0, 7.0]
    # a list of bar strings is one long row
    bars = {"kit_map": {"BD": 53}, "patterns": {"A": {"BD": ["x...x...x...x...", "x...x...x..xx..."]}}}
    assert starts(D.compile_drums(bars), 53) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.75, 7.0]
    # {"bars": n, "rows": {...}} sizes a list-only pattern
    sized = {"kit_map": {"BD": 53}, "patterns": {"A": {"bars": 2, "rows": {"BD": [1]}}}}
    assert D.compile_drums(sized).length_beats == 8.0


def test_multi_bar_rows_and_other_grids():
    two_bars = {"kit_map": {"BD": 53}, "patterns": {"A": {"BD": "x...x...x...x...x...x...x...x.x."}}}
    score = D.compile_drums(two_bars)
    assert D.parse_pattern(two_bars).patterns["A"].bars == 2
    assert starts(score, 53) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5]
    # a 12-step shuffle grid: 8th-note triplets
    shuffle = {"steps_per_bar": 12, "kit_map": {"CH": 60}, "patterns": {"A": {"CH": "x.xx.xx.xx.x"}}}
    s = D.compile_drums(shuffle)
    assert [round(x, 6) for x in starts(s, 60)] == [round(k / 3, 6) for k in (0, 2, 3, 5, 6, 8, 9, 11)]
    # 3/4 with 12 steps: 16ths
    waltz = {"steps_per_bar": 12, "beats_per_bar": 3, "kit_map": {"BD": 53, "SN": 55}, "patterns": {"A": {"BD": "x...........", "SN": "....x...x..."}}}
    w = D.compile_drums(waltz)
    assert w.beats_per_bar == 3 and starts(w, 55) == [1.0, 2.0] and w.length_beats == 3.0


def test_numbers_sent_as_strings_are_accepted():
    doc = dict(BOOK_INTRO, tempo="92", steps_per_bar="16", swing="0.5", seed="3", velocities={"hit": "90"})
    score = D.compile_drums(doc)
    assert (score.tempo, score.swing, score.seed) == (92.0, 0.0, 3) and score.notes[0].velocity == 90   # swing is applied on the grid, not carried by the Score
    with pytest.raises(ValueError, match="tempo must be a number"):
        D.parse_pattern(dict(BOOK_INTRO, tempo="fast"))


def test_tempo_swing_humanize_and_seed_are_copied():
    doc = dict(BOOK_INTRO, tempo=92, swing=0.5, humanize_ms=6, humanize_velocity=8, seed=7, hit_length_steps=1.0)
    score = D.compile_drums(doc)
    assert (score.tempo, score.swing, score.humanize_ms, score.humanize_velocity, score.seed) == (92, 0.0, 6, 8, 7)
    assert all(n.duration == 0.25 for n in score.notes)
    # swing acts on the grid: only every second 16th step moves, so 8th-note hats on odd (1-based) steps stay straight
    ons = note_ons(score)
    spb = 60 / 92
    ch = [t for t, a, _ in ons if a == 60]
    assert abs(ch[0] - 0.0) < 1e-6 and abs(ch[1] - 0.5 * spb) < 1e-6 and abs(ch[2] - 1.0 * spb) < 1e-6
    sixteenths = D.compile_drums(dict(BOOK_INTRO, tempo=92, swing=0.5, patterns={"A": {"CH": "xxxxxxxxxxxxxxxx"}}))
    hh = [t for t, a, _ in note_ons(sixteenths) if a == 60]
    assert abs(hh[1] - (0.25 + 0.0625) * spb) < 1e-6 and abs(hh[2] - 0.5 * spb) < 1e-6


# -------------------------------------------------------------------------------------------------------------- errors

def test_unknown_instrument_is_a_helpful_error():
    with pytest.raises(ValueError) as e:
        D.parse_pattern({"kit_map": {"BD": 53, "SN": 55}, "patterns": {"A": {"BD": "x...x...x...x...", "TAMBO": "..x."}}})
    msg = str(e.value)
    # the unknown name comes before any complaint about the short row
    assert "unknown instrument 'TAMBO'" in msg and "pattern 'A'" in msg and "BD=53" in msg and "SN=55" in msg
    # a known drum book role that the kit_map lacks says what to add
    with pytest.raises(ValueError) as e:
        D.parse_pattern({"kit_map": {"BD": 53}, "patterns": {"A": {"BD": [1], "OH": [3]}}})
    assert "OH" in str(e.value) and "add" in str(e.value) and "kit_map" in str(e.value)


def test_wrong_length_rows_are_a_helpful_error():
    with pytest.raises(ValueError) as e:
        D.parse_pattern({"kit_map": {"BD": 53}, "patterns": {"A": {"BD": "x...x...x...x.."}}})
    msg = str(e.value)
    assert "row 'BD'" in msg and "15 steps" in msg and "steps_per_bar = 16" in msg
    with pytest.raises(ValueError) as e:
        D.parse_pattern({"kit_map": {"BD": 53, "SN": 55}, "patterns": {"A": {"BD": "x...x...x...x...", "SN": "....x..."}}})
    msg = str(e.value)
    assert "different lengths" in msg and "SN: 8" in msg and "BD: 16" in msg
    with pytest.raises(ValueError) as e:
        D.parse_pattern({"kit_map": {"BD": 53, "SN": 55}, "patterns": {"A": {"BD": "x...x...x...x...", "SN": [17]}}})
    assert "step 17 is beyond the pattern's 16 steps" in str(e.value)
    with pytest.raises(ValueError, match="not a positive multiple"):
        D.parse_pattern({"kit_map": {"BD": 53}, "patterns": {"A": {"BD": ""}}})
    # a list-only pattern grows to whole bars around its furthest step instead
    assert D.parse_pattern({"kit_map": {"BD": 53}, "patterns": {"A": {"BD": [17]}}}).patterns["A"].bars == 2


def test_other_errors_name_the_problem():
    with pytest.raises(ValueError, match="bad character 'q' at position 5"):
        D.parse_pattern({"patterns": {"A": {"BD": "x...q...x...x..."}}})
    with pytest.raises(ValueError, match="arrangement refers to pattern 'C'"):
        D.parse_pattern({"patterns": {"A": {"BD": [1]}, "B": {"BD": [1]}}, "arrangement": "AC"})
    with pytest.raises(ValueError, match="outside the Field's drum keys"):
        D.parse_pattern({"kit_map": {"BD": 36}, "patterns": {"A": {"BD": [1]}}})
    with pytest.raises(ValueError, match="both mean"):
        D.parse_pattern({"kit_map": {"BD": 53}, "patterns": {"A": {"BD": [1], "kick": [3]}}})
    with pytest.raises(ValueError, match="listed twice"):
        D.parse_pattern({"patterns": {"A": {"BD": [1, 1]}}})
    with pytest.raises(ValueError, match="bad step"):
        D.parse_pattern({"patterns": {"A": {"BD": ["one"]}}})
    with pytest.raises(ValueError, match="unknown key"):
        D.parse_pattern({"pattern": {"A": {"BD": [1]}}})
    with pytest.raises(ValueError, match="needs \"patterns\""):
        D.parse_pattern({"tempo": 100})
    with pytest.raises(ValueError, match="tempo = 400 is out of range"):
        D.parse_pattern({"tempo": 400, "patterns": {"A": {"BD": [1]}}})
    with pytest.raises(ValueError, match="unknown velocity kind"):
        D.parse_pattern({"velocities": {"loud": 127}, "patterns": {"A": {"BD": [1]}}})
    with pytest.raises(ValueError, match="choke"):
        D.parse_pattern({"patterns": {"A": {"BD": [1]}}, "choke": {"OH": "NOPE"}})


# ------------------------------------------------------------------------------------------------ names, kit map, roles

def test_default_kit_roles_and_aliases():
    for name in ("BD", "BD2", "SN", "SN2", "CH", "CH2", "CH3", "CH_CHOKE", "OH"):
        assert not D.DEFAULT_KIT_ROLES[name]["kit_dependent"], name
    assert {n: D.DEFAULT_KIT_ROLES[n]["note"] for n in ("BD", "BD2", "SN", "SN2", "CH", "CH2", "CH3", "CH_CHOKE", "OH")} == \
           {"BD": 53, "BD2": 54, "SN": 55, "SN2": 56, "CH": 60, "CH2": 62, "CH3": 64, "CH_CHOKE": 61, "OH": 63}
    notes = [r["note"] for r in D.DEFAULT_KIT_ROLES.values()]
    assert sorted(notes) == list(range(53, 77)) and len(set(notes)) == 24          # every one of the 24 keys has a role
    for name, r in D.DEFAULT_KIT_ROLES.items():
        if r["note"] in range(57, 60) or r["note"] in range(65, 77):
            assert r["kit_dependent"], name
        assert r["key"] == D.midi_to_name(r["note"]) and r["field_shows"] == D.midi_to_name(r["note"] - 12)
    assert D.DEFAULT_KIT_ROLES["BD"]["key"] == "F3" and D.DEFAULT_KIT_ROLES["BD"]["field_shows"] == "F2"
    assert D.DEFAULT_KIT_MAP["OH"] == 63
    # the drum book abbreviations are all accepted and resolve to a default role
    for abbr in "BD SN LT MT HT CL SH RS CB CY OH CH".split():
        assert D.canonical_role(abbr) in D.DEFAULT_KIT_ROLES, abbr
    assert D.canonical_role("SD") == "SN" and D.canonical_role("closed hat") == "CH" and D.canonical_role("Open Hi-Hat") == "OH"
    assert D.canonical_role("kick") == "BD" and D.canonical_role("CP") == "CL" and D.canonical_role("hh") == "CH"
    assert D.canonical_role("nothing like this") is None
    # rows may use aliases, case, MIDI numbers and note names; the kit map may use note names and described entries
    doc = {"kit_map": {"kick": "F3", "Snare": {"note": 55, "description": "snares on"}, "CH": 60},
           "patterns": {"A": {"BD": [1], "SD": [5], "hh": [1, 3], "63": [7], "C#4": [9]}}}
    d = D.parse_pattern(doc)
    rows = d.patterns["A"].rows
    assert (rows["BD"].note, rows["SD"].note, rows["hh"].note, rows["63"].note, rows["C#4"].note) == (53, 55, 60, 63, 61)
    assert rows["BD"].kit_name == "kick" and rows["SD"].kit_name == "Snare"
    with pytest.raises(ValueError, match="both mean"):
        D.parse_pattern(dict(doc, patterns={"A": {"63": [7], "D#4": [9]}}))
    # no kit_map: the factory layout
    assert D.parse_pattern({"patterns": {"A": {"OH": [1], "CH_CHOKE": [3]}}}).kit_map == D.DEFAULT_KIT_MAP


def test_guide_documents_the_notation():
    g = D.DRUM_GUIDE
    for phrase in ("patterns", "kit_map", "arrangement", "ratchet", "accent", "ghost", "BD SN LT MT HT CL SH RS CB CY OH CH",
                   "53", "76", "one octave lower", "render_grid", "summarize", "choke", "kit dependent"):
        assert phrase in g, phrase
    assert "CH_CHOKE  61" in g and "OH        63" in g


# ------------------------------------------------------------------------------------------------ render and summarize

def test_render_grid_round_trips_the_example():
    text = D.render_grid(BOOK_INTRO)
    lines = text.splitlines()
    assert lines[0].strip().startswith("|1...2...3...4...|")
    rows = {}
    for line in lines[1:]:
        label, cells = line.split("|", 1)
        rows[label.strip()] = cells.replace("|", "")
    assert rows == BOOK_INTRO["patterns"]["A"]
    # and parsing the rendered rows back gives the same score
    again = dict(BOOK_INTRO, patterns={"A": rows})
    assert D.compile_drums(again).notes == D.compile_drums(BOOK_INTRO).notes
    assert D.grid_rows(BOOK_INTRO)[0] == rows


def test_render_grid_shows_accents_ghosts_ratchets_and_arrangement():
    doc = {"kit_map": {"BD": 53, "SN": 55, "OH": 63}, "ratchet_hits": 3,
           "patterns": {"A": {"BD": "x...x...x...x...", "SN": "....X..o....r..."}, "B": {"OH": [16]}}, "arrangement": "AB"}
    rows, spb, bpb = D.grid_rows(doc)
    assert (spb, bpb) == (16, 4)
    assert rows == {"BD": "x...x...x...x..." + "." * 16, "SN": "....X..o....r..." + "." * 16, "OH": "." * 31 + "x"}
    text = D.render_grid(doc)
    assert "|1...2...3...4...|1...2...3...4...|" in text.splitlines()[0]
    assert "BD |x...x...x...x...|................|" in text
    assert "SN |....X..o....r...|................|" in text
    # a Score with its own kit map: rows in kit_map order, unmapped notes labelled by number
    score = D.compile_drums(doc)
    rows2, _, _ = D.grid_rows(score, {"SN": 55, "BD": 53}, steps_per_bar=16)
    assert list(rows2) == ["SN", "BD", "63 (D#4)"] and rows2["SN"] == rows["SN"]


def test_summarize_reports_bars_steps_hits_polyphony_and_seconds():
    doc = {"tempo": 120, "kit_map": {"BD": 53, "SN": 55, "CH": 60, "OH": 63, "CH_CHOKE": 61}, "ratchet_hits": 2,
           "patterns": {"A": {"BD": "x...x...x...x...", "SN": "....x.......x...", "CH": "x.x.x.x.x.x.x.r."},
                        "B": {"BD": [1, 9, 11], "SN": ["5^", 13], "OH": [16]}},
           "arrangement": "AAAB", "choke": {"OH": "CH_CHOKE"}}
    s = D.summarize(doc)
    assert s["bars"] == 4 and s["steps"] == 64 and s["beats"] == 16.0 and s["seconds"] == 8.0
    assert s["arrangement"] == ["A", "A", "A", "B"]
    assert s["patterns"]["A"] == {"bars": 1, "steps": 16, "instruments": ["BD", "SN", "CH"]}
    assert s["hits"] == {"BD": 3 * 4 + 3, "SN": 3 * 2 + 2, "CH": 3 * 9, "OH": 1}          # the ratchet counts its sub-hits
    assert s["notes"] == sum(s["hits"].values())
    # BD, SN and CH all land on steps 5 and 13 of A; B has no three-way hit
    assert s["polyphony"]["peak"] == 3 and s["polyphony"]["at_steps"] == [5, 13, 21, 29, 37, 45]
    assert s["polyphony"]["overlapping"] == 3 and s["polyphony"]["field_voices"] == 8
    assert s["take_seconds"] == round(D.compile_drums(doc).seconds(), 3) == 9.5
    assert s["step_ms"] == 125.0 and s["choke"] == {"OH": "CH_CHOKE"} and s["warnings"] == []
    # warnings: the open hat and its choke hat together, and too many overlapping voices
    clash = D.summarize(dict(doc, patterns={"A": {"OH": [1], "CH_CHOKE": [1]}}, arrangement="A"))
    assert any("OH and CH_CHOKE" in w and "step(s) [1]" in w for w in clash["warnings"])
    thick = D.summarize({"kit_map": {str(n): n for n in range(53, 63)}, "hit_length_steps": 4,
                         "patterns": {"A": {str(n): [1] for n in range(53, 63)}}})
    assert thick["polyphony"]["peak"] == 10 and any("eight" in w or "8 voices" in w for w in thick["warnings"])


# ------------------------------------------------------------------------------------------ the real kit on this machine

@pytest.mark.skipif(not os.path.exists(KIT_JSON), reason="no drum slot 1 kit description on this machine")
def test_ground_truth_kit_map_compiles_the_intro_and_agrees_with_the_default_roles():
    kit_map = D.kit_map_from_file(KIT_JSON)
    assert kit_map["BD"] == 53 and kit_map["SN"] == 55 and kit_map["CH"] == 60 and kit_map["OH"] == 63 and kit_map["CH_CHOKE"] == 61
    for role, note in kit_map.items():
        assert D.DEFAULT_KIT_ROLES[role]["note"] == note, role
    with open(KIT_JSON) as f:
        labels = json.load(f)["keys"]
    assert labels["61"]["label"] == "closed hat" and labels["63"]["label"] == "open hat"
    score = D.compile_drums(dict(BOOK_INTRO, kit_map=kit_map))
    assert sorted({n.midi for n in score.notes}) == [53, 55, 60]
    # the 'keys' labels alone also make a kit map (duplicate labels get numbered)
    with open(KIT_JSON) as f:
        keys_only = {"keys": json.load(f)["keys"]}
    p = os.path.join(os.environ.get("TMPDIR", "/tmp"), "op-bridge-test-kit-keys.json")
    with open(p, "w") as f:
        json.dump(keys_only, f)
    km = D.kit_map_from_file(p)
    assert km["kick"] == 53 and km["kick 2"] == 54 and km["open hat"] == 63
    assert D.parse_pattern({"kit_map": km, "patterns": {"A": {"BD": [1], "open hat": [3]}}}).patterns["A"].rows["BD"].note == 53


@pytest.mark.skipif(not os.path.isdir(FIELD_KITS), reason="no Field disk backup on this machine")
def test_default_roles_match_the_field_written_kits():
    """The choke-group hats of DEFAULT_KIT_ROLES sit on keys the Field wrote with playmode 20480 in every sampler kit."""
    from op_bridge import presets as P
    checked = 0
    for name in sorted(os.listdir(FIELD_KITS)):
        preset = P.read_preset(os.path.join(FIELD_KITS, name))
        if preset.meta.get("type") != "drum":
            continue
        regions = {r["note"]: r for r in P.drum_kit_regions(preset)}
        assert set(regions) == set(range(53, 77))
        for role in ("CH_CHOKE", "OH"):
            assert regions[D.DEFAULT_KIT_ROLES[role]["note"]]["playmode"] == 20480, (name, role)
        for role, info in D.DEFAULT_KIT_ROLES.items():
            assert regions[info["note"]]["end"] >= regions[info["note"]]["start"], (name, role)
        checked += 1
    assert checked >= 1


def test_swing_keeps_hit_length_and_ratchet_order():
    from op_bridge.drums import compile_drums
    from op_bridge.score import compile_events
    doc = {"tempo": 120, "kit_map": {"SN": 55, "CH": 60}, "swing": 0.5,
           "patterns": {"A": {"SN": ".x..............", "CH": "xxxxxxxxxxxxxxxx"}}}
    sc = compile_drums(doc)
    ev = [e for e in compile_events(sc) if e.kind in ("note_on", "note_off")]
    for i, e in enumerate(ev):
        if e.kind == "note_off":
            on = [x for x in ev[:i] if x.kind == "note_on" and x.a == e.a]
            assert on and e.t > on[-1].t, "every note_off must follow its note_on"
    ch_on = sorted(e.t for e in ev if e.kind == "note_on" and e.a == 60)
    gaps = [round(b - a, 4) for a, b in zip(ch_on, ch_on[1:])]
    assert gaps and max(gaps) > min(gaps), "swing must alternate long and short gaps on an 8th grid"
    doc["patterns"]["A"]["SN"] = ".r.............."
    ev2 = [e for e in compile_events(compile_drums(doc)) if e.kind == "note_on" and e.a == 55]
    assert len(ev2) == 2 and ev2[1].t > ev2[0].t


def test_arrangement_repeat_shorthand_and_newlines():
    from op_bridge.drums import compile_drums, summarize
    doc = {"tempo": 100, "kit_map": {"BD": 53}, "patterns": {"A": {"BD": "x...x...x...x...\nx...x...x...x..."}, "B": {"BD": "x...x...x...x..."}},
           "arrangement": "A*2B"}
    s = summarize(doc)
    assert s["steps"] == 80, s
    assert len(compile_drums(doc).notes) == 2 * 8 + 4
