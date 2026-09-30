"""Fixes from the 2026-09-26 review, checked on real files and the real tool functions. No test opens the device."""
import glob
import os
import struct

import numpy as np
import pytest
import soundfile as sf

from mcp.server.mcpserver.exceptions import ToolError

from op_bridge import knowledge, presets as P, sampler as S, sequencers as SQ, server as srv, session
from op_bridge.score import Note, Score, pitch_to_midi, validate
from op_bridge.session import Config

REAL_HOME = os.path.expanduser("~/Music/op-bridge")
BACKUP_ROOT = os.path.join(REAL_HOME, "field-backup")
CATALOG = os.path.join(REAL_HOME, "engine-catalog.json")
needs_backup = pytest.mark.skipif(not (os.path.isdir(BACKUP_ROOT) and os.path.isfile(CATALOG)), reason="no Field backups on this machine")


@pytest.fixture
def home(tmp_path, monkeypatch):
    d = tmp_path / "home"
    d.mkdir()
    monkeypatch.setenv("OP_BRIDGE_HOME", str(d))
    monkeypatch.setattr(session, "HOME", str(d))
    monkeypatch.setattr(srv, "HOME", str(d))
    monkeypatch.setattr(srv, "CATALOG", str(d / "engine-catalog.json"))
    return str(d)





def _frames(preset):
    return struct.unpack(">hIh", dict(preset.chunks)[b"COMM"][:8])[1]


# --------------------------------------------------------------------------- names, one rule for every path

def test_one_name_rule_for_synth_sampler_and_drum_presets(tmp_path):
    assert P.PRESET_NAME_MAX == S.SAMPLER_NAME_MAX == 11
    assert P.normalize_preset_name("Big Loud PAD Name!!") == ("big loud pa", P.normalize_preset_name("Big Loud PAD Name!!")[1])
    assert P.normalize_preset_name("baby string") == ("baby string", None)
    assert P.normalize_preset_name("2014-268#01")[0] == "2014-268#01" and P.normalize_preset_name("de4d-mall")[1] is None
    assert P.normalize_preset_name("  opb_pad  ")[0] == "opb pad"
    for bad in ("", "!!!", "   "):
        with pytest.raises(ValueError):
            P.normalize_preset_name(bad)
    assert P.make_synth_preset(P.silent_template(), "cluster", "A Sixteen Char Nm").name == "a sixteen c"
    wav = str(tmp_path / "s.wav"); sf.write(wav, np.full(4410, 0.5, np.float32), 44100)
    pre, notes = S.make_sampler_preset(wav, "My Sample!", "C4", catalog_path=str(tmp_path / "none.json"))
    assert pre.meta["name"] == "my sample" and "name_truncated" in notes and S.check_sampler_preset(pre) == []
    # the sampler validator now flags characters the Field never writes, not only the length
    bad = P.PresetFile(pre.form_type, list(pre.chunks), dict(pre.meta)); bad.meta["name"] = "Loud"
    assert any("characters" in p for p in S.check_sampler_preset(bad))


@needs_backup
def test_every_field_written_name_passes_the_rule_unchanged():
    names = set()
    for p in glob.glob(os.path.join(BACKUP_ROOT, "*", "**", "*.aif"), recursive=True):
        try:
            names.add(P.read_preset(p).name)
        except Exception:
            pass
    assert len(names) >= 20
    for n in names:
        assert P.normalize_preset_name(n) == (n, None), n


# --------------------------------------------------------------------------- compose_synth_preset

@needs_backup
def test_composed_synth_presets_drop_the_factory_folder_marker_like_the_fields_own_snapshots():
    # the Field writes original_folder on factory and browser items only; its own snapshots and imports lack it
    lacking = [p for p in glob.glob(os.path.join(BACKUP_ROOT, "*", "synth", "snapshot", "*.aif")) if "original_folder" not in P.read_preset(p).meta]
    assert lacking, "expected Field-written snapshots without original_folder"
    engines = [e for e in P.known_engine_ids(CATALOG)["synth"] if e != "sampler"]
    assert engines
    for engine in engines[:3]:
        pf, notes = P.compose_synth_preset(REAL_HOME, CATALOG, engine, "opb pad")
        assert "original_folder" not in pf.meta and pf.name == "opb pad" and pf.engine == engine


@needs_backup
def test_composing_a_sampler_keeps_the_examples_region_and_passes_the_sampler_validator():
    pf, notes = P.compose_synth_preset(REAL_HOME, CATALOG, "sampler", "x", knobs=[0, 0, 1, 0])
    src = P.read_preset(P.factory_example(REAL_HOME, "sampler"))
    assert pf.meta["knobs"] == src.meta["knobs"], "the four knobs are positions on this file's audio, not ranges from other files"
    assert notes["knobs_kept_from_example"] == [1, 2, 3, 4] and "author_sampler_preset" in notes["hint"]
    assert S.check_sampler_preset(pf) == [] and "original_folder" not in pf.meta
    pf2, _ = P.compose_synth_preset(REAL_HOME, CATALOG, "sampler", "x", knobs=[0, 0, 0, 1], adsr=[0.1, 0.2, 1.0, 0.3])
    assert pf2.meta["knobs"][:4] == src.meta["knobs"][:4] and S.check_sampler_preset(pf2) == []


# --------------------------------------------------------------------------- drum kits


def test_install_presets_refuses_staged_files_the_field_would_demote(home, tmp_path):
    wav = str(tmp_path / "s.wav"); sf.write(wav, np.full(4410, 0.5, np.float32), 44100)
    pre, _ = S.make_sampler_preset(wav, "bad", "C4", catalog_path=str(tmp_path / "none.json"))
    pre.meta["knobs"][3] = 32767
    assert any("end" in p for p in P.check_preset(pre))
    P.stage_preset(home, "synth", 1, pre)
    assert P.check_staged(home) and P.check_staged(home)[0]["slot"] == 1
    res = srv.install_presets(wait_seconds=0)
    assert res["installed"] == [] and res["failed"][0]["kind"] == "synth" and "demoted" in res["message"]
    assert os.path.exists(os.path.join(home, "staging", "synth", "1.aif")), "the staged file is kept for the model to fix or discard"
    srv.discard_staged("synth", 1)
    good, _ = S.make_sampler_preset(wav, "ok", "C4", catalog_path=str(tmp_path / "none.json"))
    P.stage_preset(home, "synth", 1, good)
    assert P.check_staged(home) == []
    assert P.check_preset(good) == [] and P.check_preset(P.silent_template()) == []


# --------------------------------------------------------------------------- sampler level and the end marker

@needs_backup
def test_the_end_marker_rule_reproduces_every_field_written_whole_file_end():
    whole = 0
    for p in glob.glob(os.path.join(BACKUP_ROOT, "*", "synth", "**", "*.aif"), recursive=True):
        pf = P.read_preset(p)
        if pf.engine != "sampler":
            continue
        frames = _frames(pf); e = pf.meta["knobs"][3]
        if pf.meta["knobs"][2] == e or pf.name in ("pipe dream", "voices", "baby string", "suitcase", "7", "8"):
            assert e == S.sampler_frame_to_knob(frames, end=True), (p, frames, e)
            whole += 1
        assert "fade" not in pf.meta or (isinstance(pf.meta["fade"], int) and pf.meta["fade"] >= 0)
    assert whole >= 6


def test_quiet_cuts_are_reported_by_the_builder_and_by_sample_from_take(home, tmp_path):
    wav = str(tmp_path / "quiet.wav")
    t = np.arange(44100) / 44100
    sf.write(wav, (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 44100)        # -26 dBFS
    pre, notes = S.make_sampler_preset(wav, "quiet", "A3", catalog_path=str(tmp_path / "none.json"))
    assert notes["audio"]["peak_dbfs"] < -20 and "normalize_db=-1.0" in notes["level"] and "gain" in notes["level"]
    _, loud_notes = S.make_sampler_preset(wav, "quiet", "A3", gain=4.0, catalog_path=str(tmp_path / "none.json"))
    assert "level" not in loud_notes
    cfg = Config.load(); cfg.current_session = "lvl"; cfg.save()
    d = session.session_dir("lvl")
    assert os.path.isdir(os.path.join(d, "samples")), "session_dir creates the samples folder"
    take = os.path.join(d, "takes", "take-1.wav"); sf.write(take, np.stack([0.05 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32), 44100, subtype="FLOAT")
    res = srv.sample_from_take("take-1", 0.0, 0.5, name="soft")
    assert res["peak_dbfs"] < -20 and res["next"].startswith("the cut peaks at") and "normalize_db=-1.0" in res["next"]
    res2 = srv.sample_from_take("take-1", 0.0, 0.5, name="loud", normalize_db=-1.0)
    assert res2["next"].startswith("author_sampler_preset")
    assert "normalize_db=-1.0" in srv.sample_from_take.__doc__
    samples = srv._list_samples(d)
    assert [x["file"] for x in samples] == ["loud.wav", "soft.wav"] and all(abs(x["seconds"] - 0.5) < 1e-3 for x in samples)
    assert "samples" in session.__doc__ and "samples" in srv.list_sessions.__doc__


# --------------------------------------------------------------------------- pitches as strings, clock gate

def test_midi_numbers_sent_as_strings_are_accepted_everywhere():
    assert pitch_to_midi("60") == 60 and pitch_to_midi(" 72 ") == 72 and pitch_to_midi("C4") == 60
    s = Score(tempo=100, notes=[Note(start=0, duration=1, pitch="60"), Note(start=0, duration=1, pitch=64)])
    assert validate(s, Config()) == [] and [n.midi for n in s.notes] == [60, 64]
    with pytest.raises(ValueError, match="bad pitch"):
        pitch_to_midi("sixty")
    # nine string pitches: refused for polyphony, not for their spelling, before the device is touched
    out = srv.hold_chord([str(n) for n in (48, 52, 55, 59, 62, 65, 69, 72, 76)], beats=1, tempo=120)
    assert out["played"] is False and all("bad pitch" not in p for p in out["problems"]) and any("simultaneous" in p for p in out["problems"])


def test_guided_mode_gates_clock_and_cc80_like_set_tempo():
    s = Score(tempo=100, notes=[Note(start=0, duration=1, pitch="D4")])
    cfg = Config(); cfg.mode = "guided"
    assert validate(s, cfg) == []
    cfg.constraints.allow_transport = False
    assert any("transport" in p and "send_clock" in p for p in validate(s, cfg))
    assert validate(s.model_copy(update={"send_clock": False}), cfg) == []
    cfg.constraints.allow_transport = True; cfg.constraints.allow_tempo_change = False
    assert any("tempo change" in p for p in validate(s, cfg))
    cfg.constraints.tempo = 100.0
    assert validate(s, cfg) == [], "on the session's fixed tempo CC 80 changes nothing"
    assert any("tempo must be" in p for p in validate(s.model_copy(update={"tempo": 96.0}), cfg))
    assert "send_clock" in srv.hold_chord.__doc__


# --------------------------------------------------------------------------- guides and human steps

def test_guides_state_what_was_verified_and_name_only_real_tools():
    ov = knowledge.guide("overview")
    assert "all verified" not in ov and "marked *" in ov and "endless *" in ov
    assert "by analogy" in ov and "author_drum_kit" not in ov
    sg = SQ.SEQUENCER_GUIDE
    for helper in ("sequencer_catalog", "clock_plan", "song_position_message", "held_chords_score", "ball_drop_score"):
        assert helper not in sg, helper
    for tool in ("list_sequencers", "hold_chord", "song_position", "record_to_tape", '"length_beats"', '"tail_seconds"'):
        assert tool in sg, tool
    assert "expected to restart" in sg and "unverified" in sg.split("Caveats")[1]
    assert "Whether MIDI note-ons can enter endless steps is unverified" not in sg
    endless = SQ.sequencer_info("endless")
    assert "verified 2026-09-26" in endless["midi_evidence"] and "unverified" not in endless["midi_evidence"]
    assert "let the model send them" in SQ.HUMAN_STEPS["endless_setup"]
    sam = knowledge.guide("sampler")
    assert "normalize_db=-1.0" in sam and "original_folder" in sam and "lowercase" in sam
    assert "sample_from_take" in S.make_sampler_preset.__doc__ and "trim_take)" not in S.make_sampler_preset.__doc__


def test_human_steps_follow_the_manual_and_claim_nothing_unverified():
    assert "two seconds" in srv.human_steps("save_preset") and "three seconds" not in srv.human_steps("save_preset")
    assert "snapshot" in srv.human_steps("save_preset")
    disk = srv.human_steps("disk_mode")
    assert "MIDI and audio are off" not in disk and "unverified" in disk
    step = srv.human_steps("sample_from_input")
    assert "that key receives the sample" not in step and "does not say where" in step
    readme = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")).read()
    assert "hide MIDI and audio" not in readme and "samples" in readme.split("## Files")[1]
