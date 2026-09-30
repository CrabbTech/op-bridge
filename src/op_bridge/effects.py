"""The Field's effects: what each one does, its four encoder names, and how the bridge reaches them.

A patch (synth or drum) carries one effect on T3; the master bus carries one more, chosen by hand in the mixer.
Over MIDI the bridge turns the four encoders (CC 54-57 for the patch, CC 70-73 for the master) but cannot choose the
type; types are chosen in authored presets for any effect the Field has written a file with, or by the human.
Knob names are the firmware 1.7 manual's (pages 23-28). The fazer's device page shows pictures instead of labels, so
its four were also measured on the device and agree with the manual (docs/device.md, "Fazer measured"). The one-line
characters are the bridge's own."""
from __future__ import annotations

from typing import Any

EFFECTS: dict[str, dict[str, Any]] = {
    "cwo":      {"knobs": ["frequency", "delay", "feedback", "sideband"], "character": "a modulated delay with frequency shifting: chorus and flanger textures at low settings, metallic sideband smears at high ones", "good_for": ["width on pads", "detuned keys", "sci-fi movement"]},
    "delay":    {"knobs": ["range", "speed", "feedback", "level"], "character": "a tape-style delay; speed sets the echo time within the range, feedback the repeats", "good_for": ["dub echoes", "rhythmic slap on plucks", "space without reverb"]},
    "fazer":    {"knobs": ["frequency", "feedback", "speed", "depth"], "character": "a phaser. The manual names the encoders (its device page shows pictures instead) and the bridge measured them: frequency places the notch region (low values thin the bass, high values keep it and sweep the upper harmonics), feedback deepens the notches into resonance, speed runs from a 7 s cycle to audio-rate flutter (FAZER_RATE_TABLE; set_effect takes rate_hz), depth 0 is dry", "good_for": ["movement on pads and keys", "70s electric piano", "slow washes at speed 60-75", "ray-gun flutter above speed 110"]},
    "grid":     {"knobs": ["x size", "y size", "z feedback", "mix"], "character": "a multi-tap delay laid out on a grid: dense rhythmic patterns and clouds of echoes", "good_for": ["arpeggio patterns", "percussive textures", "ambient washes"]},
    "mother":   {"knobs": ["distance", "gate", "color", "mix"], "character": "the big reverb: distance sets the space, gate cuts the tail, color darkens or brightens it", "good_for": ["pads", "vocoder voices", "drum room"]},
    "nitro":    {"knobs": ["frequency", "filter follow", "feedback", "frequency"], "character": "a resonant filter with drive; the screen labels the first frequency LOWS and the last HIGHS (aliases lows, highs), and feedback pushes it toward squelch and self-oscillation", "good_for": ["acid basslines", "filter sweeps", "lo-fi tone shaping"]},
    "phone":    {"knobs": ["tone", "gsm", "baud", "telemetry"], "character": "a telephone and radio codec degrader: band-limited, gargled, crackly", "good_for": ["lo-fi vocals", "distant drums", "transitions"]},
    "punch":    {"knobs": ["frequency", "punch", "rounds", "power"], "character": "a transient and drive shaper: adds attack and weight, rounds the peaks", "good_for": ["drum kits", "bass", "gluing a sound"]},
    "spring":   {"knobs": ["tone", "turns", "damping", "mix"], "character": "a spring reverb: boingy, short, characterful", "good_for": ["guitars and keys", "dub drums", "surf and lo-fi"]},
    "terminal": {"knobs": ["rate", "bits", "model", "mix"], "character": "a digital degrader: sample rate and bit reduction with selectable converter models", "good_for": ["lo-fi crunch", "chiptune", "8-bit drums"]},
}

# CC names in device.CC for the four encoders
PATCH_KNOBS = ["fx1", "fx2", "fx3", "fx4"]
MASTER_KNOBS = ["master_fx1", "master_fx2", "master_fx3", "master_fx4"]


# fazer rate: CC value -> sweep rate in Hz, measured 2026-09-27 (flat below the middle, exponential above)
FAZER_RATE_TABLE: list[tuple[int, float]] = [(0, 0.15), (42, 0.18), (64, 0.65), (85, 2.0), (106, 7.8), (117, 20.0), (127, 69.0)]


def fazer_rate_cc(hz: float) -> int:
    """CC value (0-127) whose measured fazer sweep rate is closest to hz, interpolated on a log scale."""
    import math
    hz = max(FAZER_RATE_TABLE[0][1], min(FAZER_RATE_TABLE[-1][1], float(hz)))
    for (c0, h0), (c1, h1) in zip(FAZER_RATE_TABLE, FAZER_RATE_TABLE[1:]):
        if h0 <= hz <= h1:
            if h1 == h0:
                return c0
            t = (math.log(hz) - math.log(h0)) / (math.log(h1) - math.log(h0))
            return int(round(c0 + t * (c1 - c0)))
    return FAZER_RATE_TABLE[-1][0]


def fazer_rate_hz(cc: int) -> float:
    """Measured sweep rate in Hz for a fazer rate CC value, interpolated on a log scale."""
    import math
    cc = max(0, min(127, int(cc)))
    for (c0, h0), (c1, h1) in zip(FAZER_RATE_TABLE, FAZER_RATE_TABLE[1:]):
        if c0 <= cc <= c1:
            t = (cc - c0) / (c1 - c0) if c1 != c0 else 0
            return round(math.exp(math.log(h0) + t * (math.log(h1) - math.log(h0))), 2)
    return FAZER_RATE_TABLE[-1][1]


KNOB_ALIASES: dict[str, dict[str, int]] = {
    "fazer": {"rate": 2, "mix": 3, "amount": 3},
    "nitro": {"lows": 0, "low": 0, "highs": 3, "high": 3, "frequency 2": 3, "frequency2": 3},
    "delay": {"time": 0, "mix": 3},
    "terminal": {"level": 3},
}


def knob_index(fx_type: str, name: str) -> int:
    """0-3 for a knob given by name, position ("1".."4") or alias, on an effect type."""
    fx = EFFECTS.get(fx_type)
    if fx is None:
        raise ValueError(f"unknown effect {fx_type!r}; known: {sorted(EFFECTS)}")
    n = name.strip().lower()
    if n in ("1", "2", "3", "4"):
        return int(n) - 1
    if n in KNOB_ALIASES.get(fx_type, {}):
        return KNOB_ALIASES[fx_type][n]
    for i, k in enumerate(fx["knobs"]):
        if n == k.lower() or n == k.lower().split()[0] or n == f"knob{i + 1}" or n == f"fx{i + 1}":
            return i
    # the nitro effect has two knobs called frequency: accept "frequency 2" / "frequency2"
    if n.replace(" ", "") in ("frequency2",) and fx_type == "nitro":
        return 3
    raise ValueError(f"{fx_type} has knobs {fx['knobs']}; {name!r} is none of them")


def guide() -> str:
    lines = ["Effects on the OP-1 field. One effect per patch (T3) and one on the master bus (mixer, T3); the type is",
             "chosen on the device or in an authored preset, the four encoders are set over MIDI. Ten effects:", ""]
    for name, fx in EFFECTS.items():
        lines.append(f"- {name}: {fx['character']}. Knobs: {', '.join(fx['knobs'])}. Good for: {', '.join(fx['good_for'])}.")
        if name == "fazer":
            lines.append("  fazer speed CC -> Hz: " + ", ".join(f"{c} -> {h:g}" for c, h in FAZER_RATE_TABLE) + "; set_effect accepts rate_hz for it.")
    lines += ["", "Tools: slot_effects(kind, slot) tells which effect a slot's saved file carries and its knob values;",
              "set_effect({knob: value}) turns the loaded patch's effect by knob name (say the type if the file is stale);",
              "set_master_effect({knob: value}) does the same for the master bus once set_device_state(master_fx=...) has",
              "recorded which effect the human chose; author_synth_preset(fx=...) chooses a type for a new preset.",
              "All ten effects have Field-written examples in the catalog, so any of them can be chosen in an authored",
              "preset. Every effect responds to MIDI clock for its tempo-synced times when the bridge sends clock.",
              "The manual (page 22): the master effect is not recorded to tape, only into an output mixdown, so a",
              "master effect the human set is absent from tape captures and stems; patch effects are on the tape.",
              "", "Two facts from the device: the four FX encoders keep the last value sent until the human reverts the",
              "preset (a program change does not reload it and CC 63 does nothing), so set every knob you rely on;",
              "and a patch's LFO can have an effect parameter as its destination, in which case the effect moves on its",
              "own (audition shows the modulation rate). The bridge cannot switch an LFO off; set_parameters lfo2 (the",
              "amount knob of random, element and value LFOs) to 0 quiets it, or ask the human to turn it off."]
    return "\n".join(lines)
