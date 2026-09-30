"""Drum grid notation: the step grid a model writes after reading any drum pattern diagram, compiled to a Score.

A pattern document is JSON. Only "patterns" is required; everything else has a default:

{
  "tempo": 92, "steps_per_bar": 16, "beats_per_bar": 4, "swing": 0.0, "humanize_ms": 0, "humanize_velocity": 0, "seed": null,
  "kit_map": {"BD": 53, "SN": 55, "CH": 60, "OH": 63},        # instrument name -> MIDI note (or note name F3, or {"note": 53, ...})
  "velocities": {"hit": 100, "accent": 120, "ghost": 60},
  "hit_length_steps": 0.5,                                    # note length as a fraction of one step
  "ratchet_hits": 2,                                          # sub-hits inside a ratchet step
  "patterns": {"A": {"BD": "x...x...x...x...", "SN": "....x.......x...", "CH": "x.x.x.x.x.x.x.x."},
               "B": {"BD": [1, 9, 11], "SN": ["5^", 13], "OH": [16]}},
  "arrangement": "AAAB",                                      # or ["A", "A", "A", "B"]; default: each pattern once, in order
  "choke": {"OH": "CH_CHOKE"}                                 # documentation of which instrument cuts which; no events are generated
}

Row strings: x hit, X or ^ accent, o ghost, . rest, r ratchet (R accented ratchet), - tie (a rest for drums); spaces and | are
ignored, so bars can be written "x...x...|x...x..." for the eye. A row may span several bars; its length must be a multiple of
steps_per_bar and every string row of a pattern must have the same length. Rows may also be 1-based step lists: 5 hit,
"5^" accent, "5o" ghost, "5r" ratchet, "5^r" accented ratchet. Step k starts at beat (k - 1) * beats_per_bar / steps_per_bar.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .score import Note, Score, max_polyphony, midi_to_name, pitch_to_midi

# ---------------------------------------------------------------------------------------------------------------------------
# The Field's drum keys and the role names the parser understands
# ---------------------------------------------------------------------------------------------------------------------------

DRUM_FIRST_NOTE = 53        # key 0 of a kit is MIDI 53 (F3); the Field's display calls it F2, one octave below the MIDI convention
DRUM_LAST_NOTE = 76         # key 23 is MIDI 76 (E5); the device also answered drum slot 1 on 77 to 82 (which region plays is unverified)
DRUM_LAST_ANSWERED_NOTE = 82
FIELD_VOICES = 8            # eight voices with oldest-note stealing on the engine tested (docs/device.md section 9)


def _role(note: int, meaning: str, kit_dependent: bool = False, source: str = "") -> dict[str, Any]:
    return {"note": note, "key": midi_to_name(note), "field_shows": midi_to_name(note - 12), "meaning": meaning,
            "kit_dependent": kit_dependent,
            "source": source or ("suggestion: listen to the key first (audition), kits differ" if kit_dependent
                                 else "drum slot 1 'hard spunch', labelled by ear on 2026-09-26")}


# Canonical role names with their default MIDI notes for the factory layout. The core roles come from drum slot 1
# ("hard spunch": kick 53, kick 2 54, snare 55, snare 2 56, closed hats 60/62/64, choke hat 61, open hat 63); the choke-group
# hats on the C# and D# keys (61, 63; playmode 20480) hold on every factory kit the Field wrote. Everything else is a
# suggestion for keys 57-59 and 65-76 and depends on the kit: audition the key before relying on it.
DEFAULT_KIT_ROLES: dict[str, dict[str, Any]] = {
    "BD": _role(53, "bass drum (kick)"),
    "BD2": _role(54, "second bass drum (kick with a little cymbal on slot 1)"),
    "SN": _role(55, "snare drum (snares on)"),
    "SN2": _role(56, "second snare (snares off on slot 1; a rim or clap on other kits)"),
    "LT": _role(57, "low tom", True),
    "MT": _role(58, "mid tom", True),
    "HT": _role(59, "high tom", True),
    "CH": _role(60, "closed hi-hat"),
    "CH_CHOKE": _role(61, "closed hi-hat that chokes the open hat on 63 (choke group, playmode 20480 on every factory kit)",
                      source="choke-group hats on C# and D# in all seven factory kits (file analysis) and slot 1 by ear"),
    "CH2": _role(62, "second closed hi-hat"),
    "OH": _role(63, "open hi-hat, rings until choked by CH_CHOKE on 61 (playmode 20480 on every factory kit)",
                source="choke-group hats on C# and D# in all seven factory kits (file analysis) and slot 1 by ear"),
    "CH3": _role(64, "third closed hi-hat"),
    "CL": _role(65, "clap (some drum books use CL for claves and CP for clap)", True),
    "RS": _role(66, "rimshot", True),
    "CB": _role(67, "cowbell", True),
    "SH": _role(68, "shaker", True),
    "CY": _role(69, "crash cymbal", True),
    "RD": _role(70, "ride cymbal", True),
    "PC1": _role(71, "percussion or effect 1", True),
    "PC2": _role(72, "percussion or effect 2", True),
    "PC3": _role(73, "percussion or effect 3", True),
    "PC4": _role(74, "percussion or effect 4", True),
    "PC5": _role(75, "percussion or effect 5", True),
    "PC6": _role(76, "percussion or effect 6", True),
}

DEFAULT_KIT_MAP: dict[str, int] = {name: info["note"] for name, info in DEFAULT_KIT_ROLES.items()}

# Drum book abbreviations (Bardet's "200 Drum Machine Patterns" legend and its descendants: BD SD LT MT HT CH OH CY RS CP CB
# CL SH ...) and everyday names, folded to the canonical role names above. Matching ignores case, spaces, hyphens and
# underscores, so "closed hat", "Closed-Hat" and "CLOSEDHAT" all reach CH.
INSTRUMENT_ALIASES: dict[str, str] = {
    "BD": "BD", "KICK": "BD", "K": "BD", "KD": "BD", "BASS": "BD", "BASSDRUM": "BD", "KICKDRUM": "BD",
    "BD2": "BD2", "KICK2": "BD2", "BASSDRUM2": "BD2",
    "SN": "SN", "SD": "SN", "S": "SN", "SNARE": "SN", "SNAREDRUM": "SN",
    "SN2": "SN2", "SD2": "SN2", "SNARE2": "SN2",
    "LT": "LT", "LOWTOM": "LT", "FLOORTOM": "LT", "TOMLOW": "LT",
    "MT": "MT", "MIDTOM": "MT", "MIDDLETOM": "MT", "TOMMID": "MT",
    "HT": "HT", "HIGHTOM": "HT", "HITOM": "HT", "TOMHIGH": "HT",
    "CH": "CH", "HH": "CH", "HC": "CH", "HHC": "CH", "CHH": "CH", "HAT": "CH", "HATS": "CH", "HIHAT": "CH",
    "CLOSEDHAT": "CH", "CLOSEDHIHAT": "CH", "HATCLOSED": "CH", "HIHATCLOSED": "CH",
    "CH2": "CH2", "CLOSEDHAT2": "CH2", "CH3": "CH3", "CLOSEDHAT3": "CH3",
    "CHCHOKE": "CH_CHOKE", "CHOKE": "CH_CHOKE", "CHOKEHAT": "CH_CHOKE", "HATCHOKE": "CH_CHOKE", "PEDALHAT": "CH_CHOKE",
    "PH": "CH_CHOKE", "HP": "CH_CHOKE",
    "OH": "OH", "HO": "OH", "HHO": "OH", "OHH": "OH", "OPENHAT": "OH", "OPENHIHAT": "OH", "HATOPEN": "OH", "HIHATOPEN": "OH",
    "CL": "CL", "CP": "CL", "CPS": "CL", "CLAP": "CL", "CLAPS": "CL", "HANDCLAP": "CL", "CLAVES": "CL", "CLAVE": "CL",
    "RS": "RS", "RIM": "RS", "RIMSHOT": "RS", "SIDESTICK": "RS",
    "CB": "CB", "COWBELL": "CB", "BELL": "CB",
    "CY": "CY", "CR": "CY", "CRASH": "CY", "CYMBAL": "CY", "CRASHCYMBAL": "CY",
    "RD": "RD", "RIDE": "RD", "RIDECYMBAL": "RD",
    "SH": "SH", "SHAKER": "SH", "MA": "SH", "MARACAS": "SH", "TAMB": "SH", "TAMBOURINE": "SH", "CABASA": "SH",
    "PC1": "PC1", "PC2": "PC2", "PC3": "PC3", "PC4": "PC4", "PC5": "PC5", "PC6": "PC6",
    "PERC1": "PC1", "PERC2": "PC2", "PERC3": "PC3", "PERC4": "PC4", "PERC5": "PC5", "PERC6": "PC6",
}

DEFAULT_VELOCITIES: dict[str, int] = {"hit": 100, "accent": 120, "ghost": 60}
DEFAULT_STEPS_PER_BAR = 16
DEFAULT_BEATS_PER_BAR = 4
DEFAULT_HIT_LENGTH_STEPS = 0.5
DEFAULT_RATCHET_HITS = 2

# Canonical cell characters after parsing. '-' (tie) and whitespace become '.'.
CELL_HIT, CELL_ACCENT, CELL_GHOST, CELL_REST, CELL_RATCHET, CELL_ACCENT_RATCHET = "x", "X", "o", ".", "r", "R"
_CHAR_TO_CELL = {"x": CELL_HIT, "X": CELL_ACCENT, "^": CELL_ACCENT, "o": CELL_GHOST, ".": CELL_REST, "-": CELL_REST,
                 "r": CELL_RATCHET, "R": CELL_ACCENT_RATCHET}
_ROW_ALPHABET = "x hit, X or ^ accent, o ghost, . rest, r ratchet, R accented ratchet, - tie (a rest for drums); spaces, newlines and | are ignored"
_STEP_TOKEN = re.compile(r"^\s*(\d+)\s*([\^xXorR]*)\s*$")
_DOC_KEYS = {"tempo", "steps_per_bar", "beats_per_bar", "swing", "humanize_ms", "humanize_velocity", "seed", "kit_map",
             "velocities", "hit_length_steps", "ratchet_hits", "patterns", "arrangement", "choke"}
_DOC_META_KEYS = {"name", "title", "comment", "description", "source", "book", "style"}


# ---------------------------------------------------------------------------------------------------------------------------
# Parsed document
# ---------------------------------------------------------------------------------------------------------------------------

@dataclass
class DrumRow:
    instrument: str          # the name as written in the pattern
    kit_name: str            # the kit_map entry it resolved to (or the note itself when written as a number / note name)
    note: int                # MIDI note
    cells: str               # one canonical character per step: x X o . r R

    @property
    def hits(self) -> int:
        return sum(1 for c in self.cells if c != CELL_REST)


@dataclass
class DrumPattern:
    name: str
    steps: int
    bars: int
    rows: dict[str, DrumRow] = field(default_factory=dict)      # keyed by the instrument name as written


@dataclass
class DrumDocument:
    tempo: float
    steps_per_bar: int
    beats_per_bar: int
    swing: float
    humanize_ms: float
    humanize_velocity: int
    seed: int | None
    kit_map: dict[str, int]
    velocities: dict[str, int]
    hit_length_steps: float
    ratchet_hits: int
    patterns: dict[str, DrumPattern]
    arrangement: list[str]
    choke: dict[str, str]

    @property
    def step_beats(self) -> float:
        return self.beats_per_bar / self.steps_per_bar

    @property
    def total_steps(self) -> int:
        return sum(self.patterns[n].steps for n in self.arrangement)

    @property
    def total_bars(self) -> int:
        return self.total_steps // self.steps_per_bar

    @property
    def total_beats(self) -> float:
        return self.total_steps * self.step_beats


# ---------------------------------------------------------------------------------------------------------------------------
# Instrument names
# ---------------------------------------------------------------------------------------------------------------------------

def _fold(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def canonical_role(name: str) -> str | None:
    """The canonical role (BD, SN, CH ...) for a drum book abbreviation or everyday name, or None if it is not one."""
    return INSTRUMENT_ALIASES.get(_fold(name))


def _note_from_value(name: str, value: Any) -> int:
    if isinstance(value, dict):
        if "note" not in value:
            raise ValueError(f"kit_map entry {name!r} is an object without a \"note\" field; write {{\"note\": 53, \"description\": \"...\"}}")
        value = value["note"]
    if isinstance(value, bool) or isinstance(value, float) and not value.is_integer():
        raise ValueError(f"kit_map entry {name!r} must be a MIDI note number or a note name, not {value!r}")
    try:
        note = pitch_to_midi(int(value) if isinstance(value, float) else value)
    except ValueError as e:
        raise ValueError(f"kit_map entry {name!r}: {e}") from None
    if not DRUM_FIRST_NOTE <= note <= DRUM_LAST_ANSWERED_NOTE:
        raise ValueError(f"kit_map entry {name!r} = {note} ({midi_to_name(note)}) is outside the Field's drum keys: a kit has 24 keys "
                         f"from MIDI {DRUM_FIRST_NOTE} ({midi_to_name(DRUM_FIRST_NOTE)}, shown as {midi_to_name(DRUM_FIRST_NOTE - 12)} on the "
                         f"device) to {DRUM_LAST_NOTE} ({midi_to_name(DRUM_LAST_NOTE)}); notes up to {DRUM_LAST_ANSWERED_NOTE} also sound. "
                         f"General MIDI drum numbers (36 = kick) do not apply")
    return note


def parse_kit_map(kit_map: Any) -> dict[str, int]:
    """Validate a kit_map (name -> MIDI note / note name / {"note": ...}); None gives the factory layout DEFAULT_KIT_MAP."""
    if kit_map is None:
        return dict(DEFAULT_KIT_MAP)
    if not isinstance(kit_map, dict) or not kit_map:
        raise ValueError("kit_map must be a non-empty object of instrument name -> MIDI note, like {\"BD\": 53, \"SN\": 55, \"CH\": 60}")
    out: dict[str, int] = {}
    for name, value in kit_map.items():
        if not str(name).strip():
            raise ValueError("kit_map has an empty instrument name")
        out[str(name)] = _note_from_value(str(name), value)
    return out


def kit_map_from_file(path: str) -> dict[str, int]:
    """Read a kit description JSON (like kits/drum-slot-1-hard-spunch.json): its "roles" object, else its "keys" labels."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data.get("roles"), dict) and data["roles"]:
        return parse_kit_map(data["roles"])
    keys = data.get("keys")
    if isinstance(keys, dict) and keys:
        out: dict[str, int] = {}
        for note, info in keys.items():
            label = info.get("label") if isinstance(info, dict) else str(info)
            name = label or midi_to_name(int(note))
            k, i = name, 2
            while k in out:
                k, i = f"{name} {i}", i + 1
            out[k] = int(note)
        return parse_kit_map(out)
    raise ValueError(f"{path} has neither a \"roles\" object (name -> MIDI note) nor a \"keys\" object (MIDI note -> label)")


class _Resolver:
    """Finds a kit_map entry for an instrument name: exact, case-insensitive, or through the drum book aliases."""

    def __init__(self, kit_map: dict[str, int]):
        self.kit_map = kit_map
        self.by_upper = {k.upper(): k for k in kit_map}
        self.by_role: dict[str, str] = {}
        for k in kit_map:
            role = canonical_role(k)
            if role is not None:
                self.by_role.setdefault(role, k)

    def resolve(self, name: str, where: str) -> tuple[str, int]:
        text = str(name).strip()
        if text in self.kit_map:
            return text, self.kit_map[text]
        if text.upper() in self.by_upper:
            k = self.by_upper[text.upper()]
            return k, self.kit_map[k]
        role = canonical_role(text)
        if role is not None and role in self.by_role:
            k = self.by_role[role]
            return k, self.kit_map[k]
        if text.lstrip("-").isdigit() or re.match(r"^[A-Ga-g][#b]?-?\d$", text):
            note = pitch_to_midi(text)
            if not DRUM_FIRST_NOTE <= note <= DRUM_LAST_ANSWERED_NOTE:
                raise ValueError(f"{where}: row {text!r} is a note outside the Field's drum keys {DRUM_FIRST_NOTE}-{DRUM_LAST_NOTE} "
                                 f"({midi_to_name(DRUM_FIRST_NOTE)}-{midi_to_name(DRUM_LAST_NOTE)}, MIDI convention)")
            return f"{note}", note
        known = ", ".join(f"{k}={v}" for k, v in self.kit_map.items())
        hint = ""
        if role is not None:
            hint = f" ({text!r} is the drum book role {role}, but the kit_map has no {role} entry: add \"{role}\": <MIDI note> to kit_map)"
        raise ValueError(f"{where}: unknown instrument {text!r}{hint}; the kit_map knows {known}. Row names must be kit_map names, "
                         f"drum book abbreviations that resolve to them (BD SN LT MT HT CL SH RS CB CY OH CH, plus KICK, SNARE, HH, "
                         f"CLAP ...), a MIDI note number ({DRUM_FIRST_NOTE}-{DRUM_LAST_NOTE}) or a note name (F3-E5)")


# ---------------------------------------------------------------------------------------------------------------------------
# Rows and patterns
# ---------------------------------------------------------------------------------------------------------------------------

_IGNORED = " \t\r\n|"


def _is_bar_string(text: str) -> bool:
    """A string made only of row characters (and the ignored spaces and bars), so it cannot be a step token like "5^"."""
    body = [c for c in text if c not in _IGNORED]
    return bool(body) and all(c in _CHAR_TO_CELL for c in body)


def _parse_row_string(text: str, where: str) -> str:
    cells = []
    for pos, ch in enumerate(text):
        if ch in _IGNORED:
            continue
        if ch not in _CHAR_TO_CELL:
            raise ValueError(f"{where}: bad character {ch!r} at position {pos + 1} of {text!r}; the row alphabet is {_ROW_ALPHABET}")
        cells.append(_CHAR_TO_CELL[ch])
    return "".join(cells)


def _cell_from_suffix(suffix: str, where: str, token: str) -> str:
    accent = ratchet = ghost = False
    for ch in suffix:
        if ch in "^X":
            accent = True
        elif ch == "r":
            ratchet = True
        elif ch == "R":
            ratchet = accent = True
        elif ch == "o":
            ghost = True
        # 'x' is a plain hit
    if ghost and (accent or ratchet):
        raise ValueError(f"{where}: step {token!r} mixes ghost with accent/ratchet; use one of 5, 5^, 5o, 5r, 5^r")
    if ratchet:
        return CELL_ACCENT_RATCHET if accent else CELL_RATCHET
    if accent:
        return CELL_ACCENT
    if ghost:
        return CELL_GHOST
    return CELL_HIT


def _parse_step_list(items: list[Any], where: str) -> dict[int, str]:
    """1-based step tokens (5, "5^", "5o", "5r") -> {step: cell}."""
    out: dict[int, str] = {}
    for item in items:
        if isinstance(item, bool):
            raise ValueError(f"{where}: {item!r} is not a step; list rows hold 1-based step numbers like 1, 9 or \"5^\"")
        if isinstance(item, (int, float)):
            if isinstance(item, float) and not item.is_integer():
                raise ValueError(f"{where}: step {item!r} is not a whole number; steps are 1-based integers")
            step, cell = int(item), CELL_HIT
        elif isinstance(item, str):
            m = _STEP_TOKEN.match(item)
            if not m:
                raise ValueError(f"{where}: bad step {item!r}; list rows hold 1-based step numbers with an optional suffix: "
                                 f"5 hit, \"5^\" accent, \"5o\" ghost, \"5r\" ratchet, \"5^r\" accented ratchet")
            step, cell = int(m.group(1)), _cell_from_suffix(m.group(2), where, item)
        else:
            raise ValueError(f"{where}: {item!r} is not a step; list rows hold 1-based step numbers like 1, 9 or \"5^\"")
        if step < 1:
            raise ValueError(f"{where}: step {step} is out of range; steps are 1-based (the first step of the pattern is 1)")
        if step in out:
            raise ValueError(f"{where}: step {step} is listed twice")
        out[step] = cell
    return out


def _parse_pattern(name: str, body: Any, doc: DrumDocument, resolver: _Resolver) -> DrumPattern:
    where = f"pattern {name!r}"
    bars_hint: int | None = None
    if isinstance(body, dict) and isinstance(body.get("rows"), dict):
        bars_hint = body.get("bars")
        if bars_hint is not None and (isinstance(bars_hint, bool) or not isinstance(bars_hint, int) or bars_hint < 1):
            raise ValueError(f"{where}: \"bars\" must be a positive integer")
        body = body["rows"]
    if not isinstance(body, dict) or not body:
        raise ValueError(f"{where} must be an object of instrument -> row, like {{\"BD\": \"x...x...x...x...\", \"SN\": [5, 13]}}")
    spb = doc.steps_per_bar

    # instruments first, so an unknown name is reported before anything about row lengths
    resolved: dict[str, tuple[str, int]] = {}
    seen: dict[str, str] = {}
    for inst in body:
        kit_name, note = resolver.resolve(inst, f"{where}, row {inst!r}")
        if kit_name in seen and seen[kit_name] != inst:
            raise ValueError(f"{where}: rows {seen[kit_name]!r} and {inst!r} both mean {kit_name} (MIDI {note}); keep one row per instrument")
        seen[kit_name] = inst
        resolved[inst] = (kit_name, note)

    strings: dict[str, str] = {}
    lists: dict[str, dict[int, str]] = {}
    for inst, row in body.items():
        rw = f"{where}, row {inst!r}"
        if isinstance(row, str):
            strings[inst] = _parse_row_string(row, rw)
        elif isinstance(row, list):
            if row and all(isinstance(x, str) and _is_bar_string(x) for x in row):
                strings[inst] = _parse_row_string("".join(row), rw)       # a list of bar strings
            else:
                lists[inst] = _parse_step_list(row, rw)
        else:
            raise ValueError(f"{rw} must be a string like \"x...x...\" or a list of 1-based steps like [1, 9, \"5^\"], not {type(row).__name__}")

    # pattern length: from the string rows, else the bars hint, else the furthest listed step rounded up to whole bars
    lengths = {len(c) for c in strings.values()}
    if len(lengths) > 1:
        detail = ", ".join(f"{k}: {len(v)}" for k, v in strings.items())
        raise ValueError(f"{where}: rows have different lengths ({detail}); every string row of a pattern must have the same number "
                         f"of steps, a multiple of steps_per_bar = {spb}. Pad short rows with '.' or repeat them to the full length")
    if lengths:
        steps = lengths.pop()
        if steps == 0 or steps % spb:
            bad = next(k for k, v in strings.items() if len(v) == steps)
            want = f"{spb} for one bar, {2 * spb} for two"
            raise ValueError(f"{where}, row {bad!r} has {steps} steps, which is not a positive multiple of steps_per_bar = {spb} "
                             f"(want {want}). Check for a missing or extra character, or set steps_per_bar to match the grid")
    elif bars_hint is not None:
        steps = bars_hint * spb
    else:
        furthest = max((max(s) for s in lists.values() if s), default=0)
        steps = max(1, math.ceil(furthest / spb)) * spb
    if bars_hint is not None and steps != bars_hint * spb:
        raise ValueError(f"{where}: \"bars\": {bars_hint} disagrees with the row length {steps} steps ({steps // spb} bars)")

    pat = DrumPattern(name=name, steps=steps, bars=steps // spb)
    for inst, (kit_name, note) in resolved.items():
        rw = f"{where}, row {inst!r}"
        if inst in strings:
            cells = strings[inst]
        else:
            listed = lists[inst]
            beyond = [s for s in listed if s > steps]
            if beyond:
                raise ValueError(f"{rw}: step {beyond[0]} is beyond the pattern's {steps} steps ({steps // spb} bar(s) of {spb}); "
                                 f"give the pattern a longer string row, or {{\"bars\": n, \"rows\": {{...}}}}")
            cells = "".join(listed.get(k, CELL_REST) for k in range(1, steps + 1))
        pat.rows[inst] = DrumRow(instrument=inst, kit_name=kit_name, note=note, cells=cells)
    return pat


def _parse_arrangement(value: Any, patterns: dict[str, DrumPattern]) -> list[str]:
    names = list(patterns)
    if value is None:
        return names
    tokens: list[str]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("arrangement is an empty string; leave it out to play each pattern once, in order")
        if re.search(r"[\s,]", text):
            tokens = [t for t in re.split(r"[\s,]+", text) if t]
        elif text in patterns:
            tokens = [text]
        elif all(len(n) == 1 for n in names):
            tokens = [m.group(0).replace(" ", "") for m in re.finditer(r"\S\s*[*x]\s*\d+|\S", text)]
        else:
            raise ValueError(f"arrangement {value!r} cannot be split into pattern names {names}; write a list like {names[:2]} "
                             f"or separate names with spaces or commas")
    elif isinstance(value, list):
        tokens = []
        for t in value:
            if isinstance(t, bool) or not isinstance(t, (str, int)):
                raise ValueError(f"arrangement entries must be pattern names, not {t!r}")
            tokens.append(str(t))
    else:
        raise ValueError("arrangement must be a string like \"AAAB\" or a list like [\"A\", \"A\", \"A\", \"B\"]")

    out: list[str] = []
    for tok in tokens:
        m = re.match(r"^(.+?)\s*[*x]\s*(\d+)$", tok)
        if m and m.group(1) in patterns and tok not in patterns:
            if int(m.group(2)) == 0:
                raise ValueError(f"arrangement {tok!r} repeats pattern {m.group(1)!r} zero times; use a count of 1 or more")
            out.extend([m.group(1)] * int(m.group(2)))
            continue
        if tok not in patterns:
            raise ValueError(f"arrangement refers to pattern {tok!r}, which is not defined; patterns: {', '.join(names)}")
        out.append(tok)
    if not out:
        raise ValueError("arrangement is empty; leave it out to play each pattern once, in order")
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------------------------------------

def _number(doc: dict, key: str, default: float, lo: float, hi: float, integer: bool = False) -> float:
    v = doc.get(key, default)
    if v is None:
        v = default
    if isinstance(v, str) and re.match(r"^\s*-?\d+(\.\d+)?\s*$", v):      # remote clients often send numbers as strings
        v = float(v) if "." in v else int(v)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{key} must be a number, not {v!r}")
    if integer and float(v) != int(v):
        raise ValueError(f"{key} must be a whole number, not {v!r}")
    if not lo <= v <= hi:
        raise ValueError(f"{key} = {v} is out of range; allowed {lo} to {hi}")
    return int(v) if integer else float(v)


def parse_pattern(doc: dict) -> DrumDocument:
    """Validate a pattern document (see the module docstring) and resolve every row to a MIDI note. Raises ValueError."""
    if not isinstance(doc, dict):
        raise ValueError("a pattern document is a JSON object with at least \"patterns\"")
    unknown = sorted(set(doc) - _DOC_KEYS - _DOC_META_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in the pattern document; known keys: {sorted(_DOC_KEYS)}")
    if "patterns" not in doc:
        raise ValueError("the pattern document needs \"patterns\": {\"A\": {\"BD\": \"x...x...x...x...\", ...}}")
    if not isinstance(doc["patterns"], dict) or not doc["patterns"]:
        raise ValueError("\"patterns\" must be a non-empty object of pattern name -> rows, like {\"A\": {\"BD\": \"x...x...x...x...\"}}")

    tempo = _number(doc, "tempo", 100.0, 20, 300)
    steps_per_bar = int(_number(doc, "steps_per_bar", DEFAULT_STEPS_PER_BAR, 1, 64, integer=True))
    beats_per_bar = int(_number(doc, "beats_per_bar", DEFAULT_BEATS_PER_BAR, 1, 16, integer=True))
    swing = _number(doc, "swing", 0.0, 0, 1)
    humanize_ms = _number(doc, "humanize_ms", 0.0, 0, 60)
    humanize_velocity = int(_number(doc, "humanize_velocity", 0, 0, 40, integer=True))
    seed = doc.get("seed")
    if isinstance(seed, str) and seed.strip().lstrip("-").isdigit():
        seed = int(seed)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError(f"seed must be an integer or null, not {seed!r}")
    hit_length_steps = _number(doc, "hit_length_steps", DEFAULT_HIT_LENGTH_STEPS, 0.01, 64)
    ratchet_hits = int(_number(doc, "ratchet_hits", DEFAULT_RATCHET_HITS, 2, 16, integer=True))

    velocities = dict(DEFAULT_VELOCITIES)
    vel = doc.get("velocities") or {}
    if not isinstance(vel, dict):
        raise ValueError("velocities must be an object like {\"hit\": 100, \"accent\": 120, \"ghost\": 60}")
    for k, v in vel.items():
        if k not in DEFAULT_VELOCITIES:
            raise ValueError(f"unknown velocity kind {k!r}; use hit, accent, ghost")
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v <= 127:
            raise ValueError(f"velocities.{k} must be 1 to 127, not {v!r}")
        velocities[k] = int(v)
    if not velocities["ghost"] < velocities["hit"] < velocities["accent"]:
        raise ValueError(f"velocities must satisfy ghost < hit < accent (got ghost {velocities['ghost']}, hit {velocities['hit']}, "
                         f"accent {velocities['accent']}); render_grid labels hits by these thresholds")

    kit_map = parse_kit_map(doc.get("kit_map"))
    resolver = _Resolver(kit_map)

    partial = DrumDocument(tempo=tempo, steps_per_bar=steps_per_bar, beats_per_bar=beats_per_bar, swing=swing, humanize_ms=humanize_ms,
                           humanize_velocity=humanize_velocity, seed=seed, kit_map=kit_map, velocities=velocities,
                           hit_length_steps=hit_length_steps, ratchet_hits=ratchet_hits, patterns={}, arrangement=[], choke={})
    for name, body in doc["patterns"].items():
        partial.patterns[str(name)] = _parse_pattern(str(name), body, partial, resolver)
    partial.arrangement = _parse_arrangement(doc.get("arrangement"), partial.patterns)

    choke = doc.get("choke") or {}
    if not isinstance(choke, dict):
        raise ValueError("choke must be an object of instrument -> the instrument that cuts it, like {\"OH\": \"CH_CHOKE\"}")
    for cut, cutter in choke.items():
        a, _ = resolver.resolve(str(cut), "choke")
        b, _ = resolver.resolve(str(cutter), f"choke[{cut!r}]")
        partial.choke[a] = b
    return partial


def compile_drums(doc: dict | DrumDocument) -> Score:
    """Compile a pattern document to a Score: one Note per hit, ratchets as evenly spaced sub-hits inside their step,
    patterns concatenated by the arrangement, tempo / swing / humanize copied, send_clock on."""
    d = doc if isinstance(doc, DrumDocument) else parse_pattern(doc)
    vel = d.velocities
    notes: list[Note] = []
    offset_steps = 0
    for name in d.arrangement:
        pat = d.patterns[name]
        for row in pat.rows.values():
            for i, cell in enumerate(row.cells):
                if cell == CELL_REST:
                    continue
                step_index = offset_steps + i
                for start, dur, v in _cell_notes(cell, step_index, d):
                    notes.append(Note(start=start, duration=dur, pitch=row.note, velocity=v))
        offset_steps += pat.steps
    notes.sort(key=lambda n: (n.start, n.midi))
    return Score(tempo=d.tempo, beats_per_bar=d.beats_per_bar, swing=0.0, humanize_ms=d.humanize_ms,
                 humanize_velocity=d.humanize_velocity, seed=d.seed, notes=notes,
                 length_beats=_beats(offset_steps, d), send_clock=True)


def _beats(step_index: float, d: DrumDocument) -> float:
    return round(step_index * d.beats_per_bar / d.steps_per_bar, 9)


def swing_applies(d: DrumDocument) -> bool:
    """Swing moves every second step of an 8th or 16th grid; triplet and finer grids are left straight."""
    return abs(d.step_beats - 0.25) < 1e-9 or abs(d.step_beats - 0.5) < 1e-9


def swing_offset(step_index: int, d: DrumDocument) -> float:
    """Delay, in beats, of a step: odd steps of an 8th or 16th grid move late by swing x half a step
    (swing 1.0 on 8ths is a full triplet; about 0.667 gives a 16th-note triplet feel)."""
    if d.swing <= 0 or not swing_applies(d) or step_index % 2 == 0:
        return 0.0
    return round(d.swing * d.step_beats / 2.0, 9)


def _cell_notes(cell: str, step_index: int, d: DrumDocument) -> list[tuple[float, float, int]]:
    """(start beat, duration beats, velocity) for every note a cell produces. Swing is applied here, to the whole hit,
    so a swung hit keeps its length and ratchet sub-hits stay in order; durations never cross the next step."""
    step_beats = d.step_beats
    hit_beats = round(d.hit_length_steps * step_beats, 9)
    vel = d.velocities
    late = swing_offset(step_index, d)
    room = round(step_beats - late, 9)            # the next step is never swung later than this one, so stay inside it
    if cell in (CELL_RATCHET, CELL_ACCENT_RATCHET):
        v = vel["accent"] if cell == CELL_ACCENT_RATCHET else vel["hit"]
        sub = room / d.ratchet_hits
        dur = round(max(0.01, min(hit_beats, sub)), 9)      # touching, never overlapping: note_off sorts before note_on
        return [(round(_beats(step_index, d) + late + k * sub, 9), dur, v) for k in range(d.ratchet_hits)]
    v = {CELL_HIT: vel["hit"], CELL_ACCENT: vel["accent"], CELL_GHOST: vel["ghost"]}[cell]
    return [(round(_beats(step_index, d) + late, 9), round(max(0.01, min(hit_beats, room)), 9), v)]


# ---------------------------------------------------------------------------------------------------------------------------
# Back to a grid
# ---------------------------------------------------------------------------------------------------------------------------

def grid_rows(doc_or_score: dict | DrumDocument | Score, kit_map: dict[str, int] | None = None, *,
              steps_per_bar: int | None = None, beats_per_bar: int | None = None,
              velocities: dict[str, int] | None = None) -> tuple[dict[str, str], int, int]:
    """Quantize a score (or a document, compiled first) onto the grid: ({instrument: cells}, steps_per_bar, beats_per_bar).
    Cells: X at or above the accent velocity, o at or below the ghost velocity, x otherwise, r (R) where several hits share a
    step. Instruments come in kit_map order; a note with no kit_map name is labelled by its MIDI number and note name."""
    if isinstance(doc_or_score, (dict, DrumDocument)):
        d = doc_or_score if isinstance(doc_or_score, DrumDocument) else parse_pattern(doc_or_score)
        score = compile_drums(d)
        kit_map = parse_kit_map(kit_map) if kit_map is not None else d.kit_map
        steps_per_bar = steps_per_bar or d.steps_per_bar
        beats_per_bar = beats_per_bar or d.beats_per_bar
        velocities = {**d.velocities, **(velocities or {})}
    elif isinstance(doc_or_score, Score):
        score = doc_or_score
        kit_map = parse_kit_map(kit_map)
        steps_per_bar = steps_per_bar or DEFAULT_STEPS_PER_BAR
        beats_per_bar = beats_per_bar or score.beats_per_bar
        velocities = {**DEFAULT_VELOCITIES, **(velocities or {})}
    else:
        raise ValueError("render_grid takes a pattern document (dict) or a Score")

    step_beats = beats_per_bar / steps_per_bar
    unknown = set(velocities) - set(DEFAULT_VELOCITIES)
    if unknown:
        raise ValueError(f"unknown velocity kind {sorted(unknown)}; use hit, accent, ghost")
    last_note_step = max([int(math.floor(n.start / step_beats + 1e-6)) for n in score.notes] + [-1])
    steps_from_length = math.ceil((score.length_beats or 0.0) / step_beats - 1e-6)
    needed = max(last_note_step + 1, steps_from_length, 1)
    total_steps = max(steps_per_bar, math.ceil(needed / steps_per_bar) * steps_per_bar)

    by_note: dict[int, dict[int, list[int]]] = {}
    for n in score.notes:
        step = int(math.floor(n.start / step_beats + 1e-6))     # floor, so ratchet sub-hits stay inside their step
        step = min(max(step, 0), total_steps - 1)
        by_note.setdefault(n.midi, {}).setdefault(step, []).append(n.velocity)

    def cell(vels: list[int]) -> str:
        if len(vels) > 1:
            return CELL_ACCENT_RATCHET if max(vels) >= velocities["accent"] else CELL_RATCHET
        v = vels[0]
        if v >= velocities["accent"]:
            return CELL_ACCENT
        if v <= velocities["ghost"]:
            return CELL_GHOST
        return CELL_HIT

    names_by_note: dict[int, str] = {}
    for name, note in kit_map.items():
        names_by_note.setdefault(note, name)
    rows: dict[str, str] = {}
    ordered = [note for note in kit_map.values() if note in by_note]
    ordered += sorted(note for note in by_note if note not in names_by_note)
    for note in dict.fromkeys(ordered):
        label = names_by_note.get(note, f"{note} ({midi_to_name(note)})")
        hits = by_note[note]
        rows[label] = "".join(cell(hits[k]) if k in hits else CELL_REST for k in range(total_steps))
    return rows, steps_per_bar, beats_per_bar


def render_grid(doc_or_score: dict | DrumDocument | Score, kit_map: dict[str, int] | None = None, *,
                steps_per_bar: int | None = None, beats_per_bar: int | None = None,
                velocities: dict[str, int] | None = None) -> str:
    """An ASCII grid of a document or score for the model to double check: a beat ruler, then one row per instrument, one
    character per step, bars separated by |. Accents show as X, ghosts as o, ratchets as r."""
    rows, spb, bpb = grid_rows(doc_or_score, kit_map, steps_per_bar=steps_per_bar, beats_per_bar=beats_per_bar, velocities=velocities)
    if not rows:
        return "(no hits)"
    total = len(next(iter(rows.values())))
    width = max(len(k) for k in rows)
    ruler = "".join(_ruler_char(k, spb, bpb) for k in range(spb))
    bars = total // spb
    lines = [" " * width + " |" + "|".join([ruler] * bars) + "|"]
    for label, cells in rows.items():
        chunks = [cells[b * spb:(b + 1) * spb] for b in range(bars)]
        lines.append(f"{label:<{width}} |" + "|".join(chunks) + "|")
    return "\n".join(lines)


def _ruler_char(k: int, spb: int, bpb: int) -> str:
    pos = k * bpb / spb
    beat = int(round(pos))
    if abs(pos - beat) < 1e-9:
        return str(beat + 1)[-1]
    return "."


# ---------------------------------------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------------------------------------

def summarize(doc: dict | DrumDocument) -> dict[str, Any]:
    """Bars, steps, hits per instrument, polyphony peaks and seconds of a pattern document, plus warnings the Field cares about."""
    d = doc if isinstance(doc, DrumDocument) else parse_pattern(doc)
    score = compile_drums(d)
    spb_seconds = 60.0 / d.tempo

    hits: dict[str, int] = {}
    starts: dict[int, set[int]] = {}         # global step index -> notes starting there
    offset = 0
    for name in d.arrangement:
        pat = d.patterns[name]
        for row in pat.rows.values():
            label = row.kit_name
            for i, c in enumerate(row.cells):
                if c == CELL_REST:
                    continue
                n = d.ratchet_hits if c in (CELL_RATCHET, CELL_ACCENT_RATCHET) else 1
                hits[label] = hits.get(label, 0) + n
                starts.setdefault(offset + i, set()).add(row.note)
        offset += pat.steps
    peak = max((len(s) for s in starts.values()), default=0)
    peak_steps = sorted(k + 1 for k, s in starts.items() if len(s) == peak) if peak else []

    warnings: list[str] = []
    overlap = max_polyphony(score)
    if overlap > FIELD_VOICES:
        warnings.append(f"{overlap} notes sound at once (hit_length_steps = {d.hit_length_steps}); the Field has {FIELD_VOICES} voices "
                        f"and steals the oldest: shorten hit_length_steps or thin the grid")
    high = sorted({n.midi for n in score.notes if n.midi > DRUM_LAST_NOTE})
    if high:
        warnings.append(f"notes {high} are above the kit's last key {DRUM_LAST_NOTE} ({midi_to_name(DRUM_LAST_NOTE)}); the device answers "
                        f"them but which region plays is unverified")
    for cut, cutter in d.choke.items():
        a, b = d.kit_map.get(cut), d.kit_map.get(cutter)
        together = [k + 1 for k, s in starts.items() if a in s and b in s]
        if together:
            warnings.append(f"{cut} and {cutter} (which chokes it) hit together at step(s) {together[:8]}; the open sound is cut "
                            f"at once, which is usually not what a grid means")

    return {
        "tempo": d.tempo, "steps_per_bar": d.steps_per_bar, "beats_per_bar": d.beats_per_bar,
        "step_beats": d.step_beats, "step_ms": round(d.step_beats * spb_seconds * 1000, 2),
        "arrangement": list(d.arrangement),
        "patterns": {n: {"bars": p.bars, "steps": p.steps, "instruments": list(p.rows)} for n, p in d.patterns.items()},
        "bars": d.total_bars, "steps": d.total_steps, "beats": d.total_beats,
        "seconds": round(d.total_beats * spb_seconds, 3), "take_seconds": round(score.seconds(), 3),
        "notes": len(score.notes), "hits": hits,
        "polyphony": {"peak": peak, "at_steps": peak_steps[:16], "overlapping": overlap, "field_voices": FIELD_VOICES},
        "kit_map": dict(d.kit_map), "choke": dict(d.choke), "velocities": dict(d.velocities),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------------------------------------------------------
# Guide text for the model
# ---------------------------------------------------------------------------------------------------------------------------

def kit_roles_table() -> str:
    lines = ["role      note  key   Field shows  meaning"]
    for name, r in DEFAULT_KIT_ROLES.items():
        flag = " (kit dependent)" if r["kit_dependent"] else ""
        lines.append(f"{name:<9} {r['note']:<5} {r['key']:<5} {r['field_shows']:<12} {r['meaning']}{flag}")
    return "\n".join(lines)


DRUM_GUIDE = f"""Drum grid notation (JSON). Write it after reading any step-grid diagram; compile_drums turns it into a score.
{{
  "tempo": 92, "steps_per_bar": 16, "beats_per_bar": 4, "swing": 0.0, "humanize_ms": 0, "humanize_velocity": 0, "seed": null,
  "kit_map": {{"BD": 53, "SN": 55, "CH": 60, "OH": 63}},
  "velocities": {{"hit": 100, "accent": 120, "ghost": 60}}, "hit_length_steps": 0.5, "ratchet_hits": 2,
  "patterns": {{"A": {{"BD": "x...x...x...x...", "SN": "....x.......x...", "CH": "x.x.x.x.x.x.x.x."}},
               "B": {{"BD": [1, 9, 11], "SN": ["5^", 13], "OH": [16]}}}},
  "arrangement": "AAAB", "choke": {{"OH": "CH_CHOKE"}}
}}
Only "patterns" is required. Rows are strings, one character per step ({_ROW_ALPHABET}), or lists of 1-based
steps (5 hit, "5^" accent, "5o" ghost, "5r" ratchet, "5^r" accented ratchet). Step k starts at beat (k-1) * beats_per_bar /
steps_per_bar, so a 16-step bar in 4/4 is 16th notes and a 12-step bar is 8th-note triplets (shuffle). A row may span
several bars; its length must be a multiple of steps_per_bar and all string rows of a pattern must be the same length.
A pattern may also be {{"bars": 2, "rows": {{...}}}} when every row is a step list. Ratchets are ratchet_hits evenly spaced
sub-hits inside the step. The arrangement is a string of single-letter pattern names, a list of names, or "A*4" repeats;
patterns play back to back. Swing and humanize are copied to the score: swing delays every second step of an 8th or 16th grid by swing x half a step, keeping each hit's length (0.667 is a triplet feel, 1.0 on an 8th grid is a full triplet); it has no effect on triplet grids such as 12 or 24 steps per bar.
triplet feel), humanize_ms jitters timing and humanize_velocity jitters loudness; seed makes a take repeatable.
Instrument names must be kit_map names; the parser also accepts the drum book abbreviations
BD SN LT MT HT CL SH RS CB CY OH CH (and SD, HH, KICK, SNARE, CLAP, OPEN HAT ... folded to them, case and spaces ignored),
a MIDI note number or a note name as a row name. The Field's kits have 24 keys, MIDI 53 (F3) to 76 (E5); the device's own
display names them one octave lower (F2 to E4). Factory kits put the choke-group hats on C# and D# (61 closed, 63 open).
Default roles:
{kit_roles_table()}
"choke" only documents which instrument cuts another (the kit does the choking); summarize warns when both hit together.
Check the result with render_grid (the grid back from the score: X accent, o ghost, r ratchet) and summarize (bars, steps,
hits per instrument, polyphony peak, seconds, warnings). Keep hits at or below the Field's eight voices.
"""
