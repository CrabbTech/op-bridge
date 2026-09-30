"""The sampler and sequencer tools of the MCP server, called in-process on a temporary OP_BRIDGE_HOME.

Every test runs the real tool functions (the wrapped ones the server registers) against real files. No test opens
the MIDI or audio device: the tools that would reach the Field are exercised only on the paths that refuse or return
before the device is touched (guided-mode refusals, score validation)."""
import json
import os

import numpy as np
import pytest
import soundfile as sf

from mcp.server.mcpserver.exceptions import ToolError

from op_bridge import presets as P
from op_bridge import sampler as S
from op_bridge import sequencers as SQ
from op_bridge import server as srv
from op_bridge import session
from op_bridge.session import Config


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point the server, the session module and the sampler template lookup at an empty home."""
    d = tmp_path / "home"
    d.mkdir()
    monkeypatch.setenv("OP_BRIDGE_HOME", str(d))
    monkeypatch.setattr(session, "HOME", str(d))
    monkeypatch.setattr(srv, "HOME", str(d))
    monkeypatch.setattr(srv, "CATALOG", str(d / "engine-catalog.json"))
    return str(d)


def _wav(path, seconds, samplerate=48000, hz=220.0, channels=2):
    t = np.arange(int(seconds * samplerate)) / samplerate
    x = (0.5 * np.sin(2 * np.pi * hz * t) * np.exp(-t / max(seconds, 0.1))).astype(np.float32)
    if channels == 2:
        x = np.stack([x, 0.7 * x], axis=1)
    sf.write(path, x, samplerate, subtype="FLOAT")
    return str(path)


def test_author_sampler_preset_stages_a_field_shaped_file(home, tmp_path):
    wav = _wav(tmp_path / "pluck.wav", 1.5, channels=1)
    out = srv.author_sampler_preset(8, wav, "opb pluck long name", root_note="A3", start_s=0.0, loop_in_s=0.2, loop_out_s=1.0, end_s=1.5)
    assert out["staged"] == os.path.join(home, "staging", "synth", "8.aif") and os.path.exists(out["staged"])
    assert out["slot"] == 8 and out["preset"]["engine"] == "sampler" and out["preset"]["kind"] == "synth"
    assert out["preset"]["name"] == "opb pluck l" and "name_truncated" in out["derived"]
    assert out["next"].startswith("call install_presets")
    r = out["region"]
    assert r["root_note"] == "A3" and abs(r["base_freq"] - 220.0) < 1e-3 and r["channels"] == 1 and r["seconds"] == 1.5
    assert abs(r["loop_in_s"] - 0.2) < 0.001 and abs(r["loop_out_s"] - 1.0) < 0.001 and abs(r["end_s"] - 1.5) < 0.001 and r["loops"]
    # the staged file reads back as what the Field writes and passes the sampler validator
    pre = P.read_preset(out["staged"])
    assert S.check_sampler_preset(pre) == [] and pre.meta["knobs"][4:8] == [12000, 0, 0, 8192]
    assert P.staged_presets(home) == [{"kind": "synth", "slot": 8, "engine": "sampler", "name": "opb pluck l", "path": out["staged"]}]
    # the derived block surfaces what the file relied on that the device has not confirmed
    rev = srv.author_sampler_preset(7, wav, "rev", direction="reverse", gain=1.5, loop=False)
    assert any("reverse" in u for u in rev["derived"]["unverified"]) and any("gain" in u for u in rev["derived"]["unverified"])
    assert rev["region"]["knobs"][1] == rev["region"]["knobs"][2] == rev["region"]["knobs"][3]
    json.dumps(out); json.dumps(rev)


def test_author_sampler_preset_refusals_reach_the_model_as_tool_errors(home, tmp_path):
    long = _wav(tmp_path / "long.wav", 7.0, samplerate=44100)
    with pytest.raises(ToolError, match="6 s"):
        srv.author_sampler_preset(1, long, "too long")
    cut = srv.author_sampler_preset(1, long, "cut", truncate=True)
    assert cut["region"]["frames"] == S.SAMPLER_SPAN_FRAMES and cut["derived"]["audio"]["truncated"] is True
    with pytest.raises(ToolError, match="FileNotFoundError"):
        srv.author_sampler_preset(1, str(tmp_path / "missing.wav"), "x")
    with pytest.raises(ToolError, match="slot 1-8"):
        srv.author_sampler_preset(9, long, "x", truncate=True)
    with pytest.raises(ToolError, match="direction"):
        srv.author_sampler_preset(1, long, "x", direction="backwards", truncate=True)
    # guided mode: sound design reserved, then a slot outside the allowed list
    cfg = Config.load(); cfg.mode = "guided"; cfg.constraints.allow_sound_design = False; cfg.save()
    with pytest.raises(ToolError, match="human"):
        srv.author_sampler_preset(1, long, "x", truncate=True)
    cfg.constraints.allow_sound_design = True; cfg.constraints.allowed_slots = ["synth 2"]; cfg.save()
    with pytest.raises(ToolError, match="allowed_slots"):
        srv.author_sampler_preset(1, long, "x", truncate=True)
    assert srv.author_sampler_preset(2, long, "ok", truncate=True)["slot"] == 2


def test_sample_from_take_cuts_into_the_session_samples_folder(home, tmp_path):
    cfg = Config.load(); cfg.current_session = "resample"; cfg.save()
    d = session.session_dir("resample")
    take = _wav(os.path.join(d, "takes", "take-20260926-120000.wav"), 4.0, samplerate=44100)
    res = srv.sample_from_take("take-20260926-120000", 0.5, 2.0, name="bell hit")
    assert res["sample"] == os.path.join(d, "samples", "bell-hit.wav") and os.path.exists(res["sample"])
    assert res["source"] == take and res["seconds"] == 1.5 and res["samplerate"] == 44100 and res["channels"] == 2
    assert res["fits_sampler"] is True and "author_sampler_preset" in res["next"]
    x, sr = sf.read(res["sample"], dtype="float32", always_2d=True)
    assert len(x) == 66150 and abs(x[0]).max() == 0.0 and abs(x[-1]).max() < 1e-3
    # a second cut with the same name does not overwrite the first
    again = srv.sample_from_take("take-20260926-120000", 0.5, 2.0, name="bell hit", normalize_db=-1.0)
    assert again["sample"] == os.path.join(d, "samples", "bell-hit-2.wav") and abs(again["peak_dbfs"] + 1.0) < 0.05
    # to the end of the take, by WAV path, and too long for the synth sampler
    tail = srv.sample_from_take(take, 1.0)
    assert tail["seconds"] == 3.0 and tail["fits_sampler"] is True
    whole = srv.sample_from_take("take-20260926-120000", 0.0, name="whole")
    assert whole["seconds"] == 4.0
    # the cut goes straight into a sampler preset
    out = srv.author_sampler_preset(3, res["sample"], "bell", root_note=60)
    assert out["region"]["seconds"] == 1.5 and out["region"]["channels"] == 2 and out["preset"]["name"] == "bell"
    # auditions live in takes/ and seeds in seeds/; unknown ids are readable errors
    _wav(os.path.join(d, "seeds", "seed-1.wav"), 1.0, samplerate=44100)
    assert srv.sample_from_take("seed-1", 0.0, 0.5)["seconds"] == 0.5
    with pytest.raises(ToolError, match="no WAV"):
        srv.sample_from_take("take-nope", 0.0, 1.0)
    with pytest.raises(ToolError, match="past the end"):
        srv.sample_from_take("take-20260926-120000", 9.0, 10.0)


def test_sequencer_tools_without_the_device(home):
    cat = srv.list_sequencers()
    assert [c["name"] for c in cat] == list(SQ.SEQUENCER_NAMES) and cat == SQ.sequencer_catalog()
    # hold_chord validates before it touches the device: nine notes exceed the Field's voices
    out = srv.hold_chord(["C3", "E3", "G3", "B3", "D4", "F4", "A4", "C5", "E5"], beats=4, tempo=120)
    assert out["played"] is False and any("simultaneous" in p for p in out["problems"])
    with pytest.raises(ToolError, match="at least one pitch"):
        srv.hold_chord([], beats=4, tempo=120)
    cfg = Config.load(); cfg.mode = "guided"; cfg.constraints.key = "D minor"; cfg.constraints.allow_transport = False; cfg.save()
    out = srv.hold_chord(["C#4", "F4"], beats=2, tempo=100)
    assert out["played"] is False and any("D minor" in p for p in out["problems"])
    # clock and CC 80 are transport: with allow_transport off the chord is refused unless send_clock is off
    assert any("send_clock" in p and "transport" in p for p in out["problems"])
    out = srv.hold_chord(["C#4", "F4"], beats=2, tempo=100, send_clock=False)
    assert out["played"] is False and [p for p in out["problems"] if "send_clock" in p] == []
    # song position: the transport gate refuses before any port is opened, and the pointer math is the module's
    with pytest.raises(ToolError, match="human"):
        srv.song_position(4)
    assert SQ.song_position_message(4).pos == 16


def test_guides_and_human_steps_cover_sampling_and_sequencers(home):
    sampler = srv.get_guide("sampler")
    words = len(sampler.split())
    assert 300 <= words <= 600, words
    for phrase in ("6 s", "32767", "264,600", "0.1831 ms", "2^31", "2434.79", "base_freq", "sample_from_take", "author_sampler_preset",
                   "install_presets", "human_steps", "microphone", "ear", "11 characters", "unverified"):
        assert phrase in sampler, phrase
    assert srv.get_guide("sequencers") == SQ.SEQUENCER_GUIDE and srv.get_guide("sequencer") == SQ.SEQUENCER_GUIDE
    assert srv.get_guide("sampling") == sampler
    assert "sampler" in srv.get_guide("nonsense") and "sequencers" in srv.get_guide("nonsense")
    assert 'get_guide("sampler")' in srv.get_guide("overview") and 'get_guide("sequencers")' in srv.get_guide("overview")
    assert "sample_from_take" in srv.get_guide("workflow") and "hold_chord" in srv.get_guide("workflow")
    # human steps: the sequencer entries and the sampling entries are served, the old ones still format the channel
    for key in SQ.HUMAN_STEPS:
        assert srv.human_steps(key) == SQ.HUMAN_STEPS[key], key
    for key in ("sample_from_input", "lift_tape_to_sampler", "drop_sample_to_tape"):
        text = srv.human_steps(key)
        assert len(text) > 100 and "%" not in text, key
    s = srv.human_steps("sample_from_input")
    assert "input key" in s and "shift and press input" in s and "blue encoder" in s and "ear" in s
    assert "lift key" in srv.human_steps("lift_tape_to_sampler") and "drop" in srv.human_steps("lift_tape_to_sampler")
    assert "shift and press record" in srv.human_steps("drop_sample_to_tape") and "drop" in srv.human_steps("drop_sample_to_tape")
    assert "channel 1" in srv.human_steps("midi_settings")
    unknown = srv.human_steps("nope")
    assert "sample_from_input" in unknown and "tombola_setup" in unknown and "arm_recording" in unknown
