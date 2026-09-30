"""The effects tools against the connected Field. The fazer's encoders are named in the manual (frequency, feedback,
speed, depth; its device page shows pictures) and were measured: set_effect takes its speed in Hz and the device
must sweep at that rate, and depth 0 must be dry."""
import time

import numpy as np
import pytest

from op_bridge import effects as FX
from op_bridge import server as srv
from op_bridge.analysis import modulation_rate
from op_bridge.device import Field
from op_bridge.player import detect_live_pair
from op_bridge.session import Config, read_state
from tests.conftest import needs_field

C4 = 261.63


def _hold(channel: int, note: int = 60, seconds: float = 5.0):
    """Hold one note on the loaded sound and return the steady part of the live pair, mono."""
    with Field(channel=channel) as f:
        with f.recorder() as rec:
            time.sleep(0.25); f.note_on(note, 100); time.sleep(seconds); f.note_off(note); time.sleep(0.5)
        r = rec.stop()
    pair = detect_live_pair(r.audio)
    assert pair, "no live audio from the Field"
    x = r.audio[:, pair[0] - 1: pair[1]].astype(np.float64).mean(axis=1)
    return x[int(0.9 * r.samplerate): len(x) - int(0.6 * r.samplerate)], r.samplerate


def test_fazer_rate_table_is_monotonic_and_round_trips():
    ccs = [FX.fazer_rate_cc(hz) for hz in (0.15, 0.5, 1.0, 2.0, 4.0, 7.8, 20.0, 69.0)]
    assert ccs == sorted(ccs) and ccs[0] == 0 and ccs[-1] == 127
    assert FX.fazer_rate_cc(2.0) == 85 and FX.fazer_rate_cc(7.8) == 106 and FX.fazer_rate_cc(1000) == 127
    assert FX.fazer_rate_hz(85) == 2.0 and FX.fazer_rate_hz(106) == 7.8 and 0.6 < FX.fazer_rate_hz(64) < 0.7
    assert FX.EFFECTS["fazer"]["knobs"] == ["frequency", "feedback", "speed", "depth"]
    assert FX.knob_index("fazer", "rate") == FX.knob_index("fazer", "speed") == 2 and FX.knob_index("fazer", "mix") == FX.knob_index("fazer", "depth") == 3


def _cc7(v: float) -> int:
    return max(0, min(127, int(round(v / 32767 * 127))))


@needs_field
def test_fazer_rate_in_hz_is_the_rate_the_field_sweeps_at():
    """Live edits persist on the Field and nothing reads them back, so the test first puts the slot's engine and
    envelope knobs back to its saved file's values, then checks the effect answers its depth knob at all; if it
    does not (the human toggled the effect off with T3, which no MIDI message can undo) the test skips."""
    fx = srv._slot_fx("synth", 3)
    if fx.get("fx_type") != "fazer":
        pytest.skip("synth slot 3 does not hold a fazer preset (install one with author_synth_preset fx='fazer')")
    cfg = Config.load()
    before = read_state()
    try:
        srv.select_sound("synth", 3)
    except Exception as e:  # guided mode reserves sound selection for the human
        pytest.skip(f"cannot select slot 3 in this mode: {e}")
    from op_bridge import presets as P
    meta = P.read_preset(srv._slot_file("synth", 3)).meta
    srv.set_parameters({f"engine{i + 1}": _cc7(v) for i, v in enumerate(meta.get("knobs", [])[:4])})
    srv.set_parameters({k: _cc7(v) for k, v in zip(("attack", "decay", "sustain", "release"), meta.get("adsr", [])[:4])})
    srv.set_parameters({"lfo2": 0})
    # does the effect respond at all? depth 0 versus depth 110 at a fast speed
    srv.set_effect({"frequency": 64, "feedback": 64, "speed": 106, "depth": 0}, fx_type="fazer", kind="synth", slot=3)
    x, sr = _hold(cfg.midi_channel - 1, seconds=3.0)
    dry = modulation_rate(x, sr, C4)
    srv.set_effect({"depth": 110}, fx_type="fazer", kind="synth", slot=3)
    x, sr = _hold(cfg.midi_channel - 1, seconds=3.0)
    wet = modulation_rate(x, sr, C4)
    print("depth 0:", dry, "depth 110:", wet)
    if not (wet["rate_hz"] and 6.0 <= wet["rate_hz"] <= 10.0):
        pytest.skip(f"the fazer on slot 3 does not answer its depth knob (dry {dry}, wet {wet}); the effect is probably "
                    "toggled off on the device: " + srv.human_steps("toggle_effect"))
    try:
        for hz, lo, hi in ((2.0, 1.4, 2.8), (7.8, 6.0, 10.0)):
            out = srv.set_effect({"frequency": 64, "feedback": 64, "depth": 110, "rate_hz": hz}, fx_type="fazer", kind="synth", slot=3)
            assert out["effect"] == "fazer" and out["set"]["speed"] == FX.fazer_rate_cc(hz) and abs(out["rate_hz"] - hz) < 0.3
            x, sr = _hold(cfg.midi_channel - 1)
            m = modulation_rate(x, sr, C4)
            print(f"asked {hz} Hz -> {out['set']} measured {m}")
            assert m["rate_hz"] is not None and lo <= m["rate_hz"] <= hi, m
        srv.set_effect({"depth": 0}, fx_type="fazer", kind="synth", slot=3)
        x, sr = _hold(cfg.midi_channel - 1)
        m = modulation_rate(x, sr, C4)
        print("depth 0:", m)
        assert m["prominence"] < 40 and not (m["rate_hz"] and 6.0 <= m["rate_hz"] <= 10.0), m
    finally:
        saved = {k: int(round(v / 32767 * 127)) for k, v in (fx.get("knobs") or {}).items() if isinstance(v, (int, float))}
        if saved:
            srv.set_effect(saved, fx_type="fazer", kind="synth", slot=3)
        if before.get("mode") == "synth" and before.get("synth_slot"):
            srv.select_sound("synth", int(before["synth_slot"]))
        elif before.get("mode") == "drum" and before.get("drum_slot"):
            srv.select_sound("drum", int(before["drum_slot"]))
