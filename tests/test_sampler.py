"""Synth sampler presets: what the bridge writes must be what the Field's own sampler files look like.

Every test reads and writes real files. The tests that need the Field's own sampler files skip when the disk
backup is not on this machine. No test touches the device."""
import glob
import json
import os
import re
import struct
import warnings

import numpy as np
import pytest
import soundfile as sf

from op_bridge import presets as P
from op_bridge import sampler as S

HOME = os.path.expanduser("~/Music/op-bridge")
BACKUP_ROOT = os.path.join(HOME, "field-backup")
BACKUP = os.path.join(BACKUP_ROOT, "2026-09-26-disk")


def _sampler_files():
    out = []
    for p in sorted(glob.glob(os.path.join(BACKUP_ROOT, "*", "synth", "**", "*.aif"), recursive=True)):
        try:
            if P.read_preset(p).engine == "sampler":
                out.append(p)
        except Exception:
            pass
    return out


SAMPLER_FILES = _sampler_files() if os.path.isdir(BACKUP_ROOT) else []
needs_backup = pytest.mark.skipif(not SAMPLER_FILES, reason="no Field-written sampler file under ~/Music/op-bridge/field-backup")


def _chunks(path):
    data = open(path, "rb").read()
    out = {}
    order = []
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]; size = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        out[cid] = data[pos + 8:pos + 8 + size]; order.append(cid)
        pos += 8 + size + (size & 1)
    return data[8:12], out, order


def _aifc_open(path):
    aifc = pytest.importorskip("aifc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return aifc.open(path)


def _sine_wav(path, seconds, samplerate, hz=261.6256, amp=0.5, channels=1):
    t = np.arange(int(seconds * samplerate)) / samplerate
    x = (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    if channels == 2:
        x = np.stack([x, 0.5 * x], axis=1)
    sf.write(path, x, samplerate)
    return path


# --------------------------------------------------------------------------- the unit and the pitch formula

def test_unit_math_matches_the_fields_import_writer():
    assert S.SAMPLER_SPAN_FRAMES == 264600 and S.SAMPLER_POSITION_MAX == 32767
    assert abs(S.SAMPLER_UNIT_FRAMES - 8.0752) < 0.001
    # the end marker is the last frame's index (frames - 1) rounded up, which reproduces every Field-written whole-file end:
    # the Field's own import writer: 44100 frames -> 5462 (exact 5461.04), 44160 -> 5469 (exact 5468.47)
    assert S.sampler_frame_to_knob(44100, end=True) == 5462
    assert S.sampler_frame_to_knob(44160, end=True) == 5469
    # factory whole-file ends: pipe dream 177166 -> 21940, baby string 252757 -> 31301, suitcase clamps at 32767,
    # and voices 244178 -> 30238 (rounding the frame count itself, 30238.02, would give 30239, which no file holds)
    assert S.sampler_frame_to_knob(177166, end=True) == 21940
    assert S.sampler_frame_to_knob(252757, end=True) == 31301
    assert S.sampler_frame_to_knob(264604, end=True) == 32767
    assert S.sampler_frame_to_knob(244178, end=True) == 30238
    assert S.sampler_frame_to_knob(264600, end=True) == 32767 and S.sampler_frame_to_knob(0) == 0
    # seconds: 6 s is the top of the range, 3 s the middle
    assert S.sampler_seconds_to_knob(6.0) == 32767 and S.sampler_seconds_to_knob(3.0) == round(32767 / 2)
    assert abs(S.sampler_knob_to_seconds(32767) - 6.0) < 1e-9
    for k in (0, 1, 546, 5462, 21940, 32767):
        assert S.sampler_seconds_to_knob(S.sampler_knob_to_seconds(k)) == k
        assert abs(S.sampler_knob_to_frame(k) - k * 264600 / 32767) < 1e-6
    r = S.sampler_positions(0.0, frames=88200)
    assert r == {"start": 0, "loop_in": 0, "loop_out": 10923, "end": 10923}
    r = S.sampler_positions(0.1, 0.5, 1.5, 2.0, frames=88200)
    assert r == {"start": 546, "loop_in": 2731, "loop_out": 8192, "end": 10923}
    with pytest.raises(ValueError):
        S.sampler_positions(1.0, 0.5, 1.5, 2.0, frames=88200)       # loop in before start
    with pytest.raises(ValueError):
        S.sampler_positions(0.0, 0.5, 1.5, 3.0, frames=88200)       # end past the sample
    with pytest.raises(ValueError):
        S.sampler_positions(0.0, frames=300000)                     # more than 6 s cannot be addressed


def test_note_names_and_base_freq():
    assert S.note_number("C4") == 60 and S.note_number("A4") == 69 and S.note_number("Db4") == 61 and S.note_number("F#3") == 54
    assert S.note_number("C-1") == 0 and S.note_number(72) == 72
    # a MIDI number sent as a string (remote clients do this) is read as the number
    assert S.note_number("60") == 60 and S.note_number(" 69 ") == 69 and S.base_freq_from_note("69") == 440.0
    for bad in ("128", "-1"):
        with pytest.raises(ValueError):
            S.note_number(bad)
    assert S.note_name(60) == "C4" and S.note_name(53) == "F3"
    assert S.base_freq_from_note("A4") == 440.0
    assert abs(S.base_freq_from_note("C4") - 261.6256) < 1e-4
    assert abs(S.base_freq_from_note(72) - 523.2511) < 1e-4
    # the values the Field wrote map back to the notes the research identified
    for f, n in ((261.62555, 60), (261.625375, 60), (523.2511, 72), (523.25115, 72), (440.0, 69)):
        assert abs(S.note_from_base_freq(f) - n) < 0.01
    for bad in ("H4", "C", "", "C#", 128):
        with pytest.raises(ValueError):
            S.note_number(bad)


# --------------------------------------------------------------------------- authoring

def test_build_a_sampler_preset_from_a_sine_wav(tmp_path):
    wav = _sine_wav(str(tmp_path / "sine.wav"), 2.0, 48000)   # 48 kHz mono, so the bridge must resample
    pre, notes = S.make_sampler_preset(wav, "opb sine", "C4", start_s=0.1, loop_in_s=0.5, loop_out_s=1.5, end_s=2.0,
                                       catalog_path=str(tmp_path / "no-catalog.json"))
    assert S.check_sampler_preset(pre) == []
    out = str(tmp_path / "8.aif")
    P.write_preset(out, pre)
    back = P.read_preset(out)
    m = back.meta
    assert back.kind == "synth" and back.engine == "sampler" and m["type"] == "sampler" and m["synth_version"] == 3
    assert m["name"] == "opb sine" and m["octave"] == 0 and m["stereo"] is False and m["fade"] == 0
    assert abs(m["base_freq"] - 261.6256) < 1e-3
    # region knobs within one unit of seconds / 6 * 32767
    for i, s in enumerate((0.1, 0.5, 1.5, 2.0)):
        assert abs(m["knobs"][i] - s / 6.0 * 32767) <= 1, (i, s, m["knobs"][i])
    assert m["knobs"][0] <= m["knobs"][1] <= m["knobs"][2] <= m["knobs"][3]
    assert m["knobs"][3] == S.sampler_frame_to_knob(88200, end=True) == 10923
    # shift page is the Field's import default, the rest of the block too
    assert m["knobs"][4:] == [12000, 0, 0, 8192]
    assert m["adsr"] == [64, 10746, 32767, 10000, 4000, 64, 4000, 18021]
    assert m["fx_type"] == "delay" and m["fx_active"] is False and m["fx_params"] == [8000] * 8
    assert m["lfo_type"] == "tremolo" and m["lfo_active"] is False and m["lfo_params"] == [16000, 0, 0, 16000, 0, 0, 0, 0]
    assert "original_folder" not in m
    assert set(m) == {"adsr", "base_freq", "fade", "fx_active", "fx_params", "fx_type", "knobs", "lfo_active", "lfo_params", "lfo_type",
                      "mtime", "name", "octave", "stereo", "synth_version", "type"}
    # container: the Field's layout, chunk order FVER COMM APPL SSND
    form, ch, order = _chunks(out)
    assert form == b"AIFC" and order == [b"FVER", b"COMM", b"APPL", b"SSND"]
    assert ch[b"FVER"] == bytes.fromhex("a2805140")
    comm = ch[b"COMM"]
    channels, frames, bits = struct.unpack(">hIh", comm[:8])
    assert (channels, frames, bits) == (1, 88200, 16)
    assert len(comm) == 64 and comm[18:22] == b"sowt" and comm[23:] == b"Signed integer (little-endian) linear PCM"
    assert comm[8:18] == bytes.fromhex("400eac44000000000000")   # 44100 Hz as an 80-bit float
    appl = ch[b"APPL"]
    assert appl.startswith(b'op-1{"adsr":') and appl.rstrip(b" ").endswith(b'"type":"sampler"}\n') and len(appl) % 2 == 0
    assert len(ch[b"SSND"]) == 8 + frames * channels * 2
    # the standard aifc module reads it (it hands 'sowt' data back byte-swapped to big-endian)
    a = _aifc_open(out)
    assert (a.getnchannels(), a.getsampwidth(), a.getframerate(), a.getnframes(), a.getcomptype()) == (1, 2, 44100, 88200, b"sowt")
    pcm = np.frombuffer(a.readframes(88200), dtype=">i2").astype(np.float32) / 32767
    a.close()
    # and the audio is the sine, resampled, at its level
    t = np.arange(88200) / 44100
    ref = 0.5 * np.sin(2 * np.pi * 261.6256 * t)
    seg = slice(200, 88000)
    corr = float(np.dot(pcm[seg], ref[seg]) / (np.linalg.norm(pcm[seg]) * np.linalg.norm(ref[seg])))
    assert corr > 0.999 and abs(np.max(np.abs(pcm)) - 0.5) < 0.01
    # what the bridge reports about it
    assert notes["audio"]["resampled"] == "48000 -> 44100 Hz" and notes["audio"]["frames"] == 88200 and notes["audio"]["channels"] == 1
    assert notes["root"] == {"note": "C4", "midi": 60, "base_freq": m["base_freq"]}
    assert notes["region"]["end"]["knob"] == 10923 and "unverified" not in notes
    r = S.sampler_region(back)
    assert r["root_note"] == "C4" and r["loops"] and r["direction"] == "forward" and r["gain"] == 1.0
    assert abs(r["start_s"] - 0.1) < 0.001 and abs(r["end_s"] - 2.0) < 0.001 and r["frames"] == 88200 and r["channels"] == 1
    assert back.summary()["engine"] == "sampler" and back.summary()["knobs"] == m["knobs"][:4]


def test_stereo_source_and_whole_sample_default_match_the_fields_own_import(tmp_path):
    # a 1.000 s stereo file at 44.1 kHz: the Field wrote knobs [0, 0, 5462, 5462] when it imported one of 44100 frames
    wav = _sine_wav(str(tmp_path / "st.wav"), 1.0, 44100, channels=2)
    pre, notes = S.make_sampler_preset(wav, "opb stereo", "A4", catalog_path=str(tmp_path / "no-catalog.json"))
    out = str(tmp_path / "7.aif"); P.write_preset(out, pre)
    back = P.read_preset(out)
    assert back.meta["knobs"] == [0, 0, 5462, 5462, 12000, 0, 0, 8192]
    assert back.meta["base_freq"] == 440.0 and back.meta["stereo"] is True
    channels, frames, bits = struct.unpack(">hIh", dict(back.chunks)[b"COMM"][:8])
    assert (channels, frames, bits) == (2, 44100, 16)
    a = _aifc_open(out)
    assert a.getnchannels() == 2 and a.getnframes() == 44100
    pcm = np.frombuffer(a.readframes(44100), dtype=">i2").reshape(-1, 2); a.close()
    # channel two was written at half level, so it must still be the second channel after interleaving
    assert abs(np.max(np.abs(pcm[:, 1])) / np.max(np.abs(pcm[:, 0])) - 0.5) < 0.01
    assert S.check_sampler_preset(back) == [] and "resampled" not in notes["audio"]


def test_options_limits_and_the_catalog(tmp_path):
    cat = str(tmp_path / "engine-catalog.json")
    if os.path.isdir(BACKUP):
        P.learn_engines(BACKUP, cat)
    else:
        json.dump({"synth": {}, "drum": {}, "fx": {"mother": {"examples": [{"params": [22115, 0, 8152, 11263, 0, 0, 0, 0], "from": "voices"}]}},
                   "lfo": {"random": {"examples": [{"params": [14324, 24304, 7168, 24000, 0, 0, 0, 13797], "from": "voices"}]}}}, open(cat, "w"))
    wav = _sine_wav(str(tmp_path / "s.wav"), 1.5, 44100)
    # shift page, loop switch, envelope, fx and lfo
    pre, notes = S.make_sampler_preset(wav, "a name that is far too long", 60, loop=False, direction="reverse", gain=2.0, fine_tune=-0.5,
                                       loop_fade=0.25, octave=1, fx="mother", lfo="random", adsr=[0.0, 0.5, 1.0, 0.5], catalog_path=cat)
    m = pre.meta
    assert m["name"] == "a name that" and "name_truncated" in notes
    assert m["knobs"][1] == m["knobs"][2] == m["knobs"][3] and m["knobs"][0] == 0
    assert m["knobs"][4] == S.SAMPLER_DIRECTION_REVERSE == 24576
    assert m["knobs"][5] == P.pct_to_raw(-0.5, bipolar=True) < 0
    assert m["knobs"][6] == P.pct_to_raw(0.25) == 8192
    assert m["knobs"][7] == 16384 and notes["shift"]["gain_linear"] == 2.0
    assert m["octave"] == 1 and m["fx_type"] == "mother" and m["fx_active"] is True and m["lfo_type"] == "random" and m["lfo_active"] is True
    fx_ex = P.catalog_examples(cat, "fx", "mother")[-1]
    assert m["fx_params"] == fx_ex["params"] and notes["fx_from"] == fx_ex["from"]
    assert m["lfo_params"] == P.catalog_examples(cat, "lfo", "random")[-1]["params"]
    assert m["adsr"][0] == 64 and m["adsr"][2] == 32767 and 0 < m["adsr"][1] < 32767
    assert any("reverse" in u for u in notes["unverified"]) and any("loop=False" in u for u in notes["unverified"])
    assert S.check_sampler_preset(pre) == []
    # explicit flags win, and an fx type the Field has not written is refused
    pre2, _ = S.make_sampler_preset(wav, "quiet", "C4", fx="mother", fx_active=False, catalog_path=cat)
    assert pre2.meta["fx_type"] == "mother" and pre2.meta["fx_active"] is False
    with pytest.raises(ValueError):
        S.make_sampler_preset(wav, "x", "C4", fx="spring", catalog_path=cat)
    with pytest.raises(ValueError):
        S.make_sampler_preset(wav, "x", "C4", fx="nope", catalog_path=cat)
    with pytest.raises(ValueError):
        S.make_sampler_preset(wav, "x", "C4", direction="backwards", catalog_path=cat)
    with pytest.raises(ValueError):
        S.make_sampler_preset(wav, "x", "C4", loop=False, loop_in_s=0.2, catalog_path=cat)
    with pytest.raises(ValueError):
        S.make_sampler_preset(wav, "x", "C4", start_s=1.0, loop_in_s=0.5, catalog_path=cat)
    with pytest.raises(ValueError):
        S.make_sampler_preset(wav, "x", "C4", end_s=2.0, catalog_path=cat)
    with pytest.raises(ValueError):
        S.make_sampler_preset(wav, "", "C4", catalog_path=cat)
    # a measured root pitch goes straight into base_freq, flagged
    pre3, notes3 = S.make_sampler_preset(wav, "tuned", "C4", root_hz=261.1, catalog_path=cat)
    assert pre3.meta["base_freq"] == 261.1 and any("root_hz" in u or "measured" in u for u in notes3["unverified"])
    # more than 6 s is refused unless truncated; truncated audio fills the whole 6 s span
    long_wav = _sine_wav(str(tmp_path / "long.wav"), 7.0, 44100)
    with pytest.raises(ValueError, match="sample_from_take"):
        S.make_sampler_preset(long_wav, "long", "C4", catalog_path=cat)
    pre4, notes4 = S.make_sampler_preset(long_wav, "long", "C4", truncate=True, catalog_path=cat)
    _, frames, _ = P._comm_fields(pre4)
    assert frames == 264600 and pre4.meta["knobs"][:4] == [0, 0, 32767, 32767] and notes4["audio"]["truncated"] is True
    assert S.check_sampler_preset(pre4) == []
    # exactly 6 s at 48 kHz resamples to the full span, no error
    six = _sine_wav(str(tmp_path / "six.wav"), 6.0, 48000)
    pre5, _ = S.make_sampler_preset(six, "six", "C4", catalog_path=cat)
    assert P._comm_fields(pre5)[1] == 264600
    # a few frames over, like the factory sample "suitcase" (264604 frames), is cut to the span rather than refused
    over = str(tmp_path / "over.wav"); sf.write(over, np.full(264604, 0.1, np.float32), 44100)
    pre6, notes6 = S.make_sampler_preset(over, "over", "C4", catalog_path=cat)
    assert P._comm_fields(pre6)[1] == 264600 and pre6.meta["knobs"][3] == 32767 and notes6["audio"]["truncated"] is True
    with pytest.raises(ValueError):
        sf.write(over, np.full(264700, 0.1, np.float32), 44100)
        S.make_sampler_preset(over, "over", "C4", catalog_path=cat)


def test_the_validator_catches_what_the_field_would_demote(tmp_path):
    wav = _sine_wav(str(tmp_path / "s.wav"), 0.5, 44100)
    pre, _ = S.make_sampler_preset(wav, "ok", "C4", catalog_path=str(tmp_path / "none.json"))
    assert S.check_sampler_preset(pre) == []
    bad = P.PresetFile(pre.form_type, list(pre.chunks), dict(pre.meta))
    bad.meta["knobs"] = [5000, 100, 2000, 2731, 12000, 0, 0, 8192]
    assert any("start <= loop in" in p for p in S.check_sampler_preset(bad))
    bad.meta["knobs"] = [0, 0, 32767, 32767, 12000, 0, 0, 8192]     # end past a 0.5 s sample
    assert any("end" in p for p in S.check_sampler_preset(bad))
    bad.meta["knobs"] = list(pre.meta["knobs"]); bad.meta["stereo"] = True
    assert any("stereo" in p for p in S.check_sampler_preset(bad))
    bad.meta["stereo"] = False; bad.meta["name"] = "twelve chars"
    assert any("name" in p for p in S.check_sampler_preset(bad))
    bad.meta["name"] = "ok"; bad.meta["fx_type"] = "reverb"
    assert any("fx_type" in p for p in S.check_sampler_preset(bad))
    assert S.check_sampler_preset(P.silent_template()) == ["type 'cluster' is not sampler"]


# --------------------------------------------------------------------------- against the Field's own files

@needs_backup
def test_field_written_sampler_files_fit_the_six_second_span():
    whole_file, names = 0, set()
    for path in SAMPLER_FILES:
        pf = P.read_preset(path)
        ch, frames, bits = struct.unpack(">hIh", dict(pf.chunks)[b"COMM"][:8])
        m = pf.meta
        assert bits == 16 and ch in (1, 2) and m["stereo"] == (ch == 2) and frames <= S.SAMPLER_SPAN_FRAMES + 8, path
        assert S.check_sampler_preset(pf) == [], path
        s, li, lo, e = m["knobs"][:4]
        assert 0 <= s <= li <= lo <= e <= 32767, path
        predicted = S.sampler_frame_to_knob(frames, end=True)
        assert e <= predicted + 1, f"{path}: end {e} is past the sample under the 6 s unit ({predicted})"
        if abs(e - predicted) <= 1:
            whole_file += 1; names.add(m["name"])
        r = S.sampler_region(pf)
        assert r["root_note"] in ("C4", "C5", "A4") and abs(r["root_midi"] - round(r["root_midi"])) < 0.01, path
        assert abs(r["end_s"] - S.sampler_knob_to_seconds(e)) < 1e-3
    # pipe dream, suitcase, baby string, voices and the Field's imports all end on the last frame under this unit
    assert whole_file >= 4 and {"pipe dream", "7"} & names, names


@needs_backup
def test_the_file_length_alternative_is_falsified_by_the_audio():
    """Under "32767 = the file's own length" pipe dream's end marker would sit at 2.69 s with the sample still sounding after it."""
    paths = [p for p in SAMPLER_FILES if P.read_preset(p).name == "pipe dream"]
    if not paths:
        pytest.skip("the factory preset 'pipe dream' is not in the backups")
    pf = P.read_preset(paths[0])
    ch, frames, _ = P._comm_fields(pf)
    e = pf.meta["knobs"][3]
    pcm = S._pcm(pf)
    alt_end = int(e / 32767 * frames)
    after_alt = float(np.max(np.abs(pcm[alt_end:])))
    assert alt_end < frames * 0.8 and 20 * np.log10(after_alt) > -20, "audio after the alternative end marker should be loud"
    six_s_end = round(S.sampler_knob_to_frame(e))
    assert abs(six_s_end - frames) <= 5, "under the 6 s unit the end marker is the last frame"


@needs_backup
def test_rewriting_field_sampler_files(tmp_path):
    """The Field's own import writer is matched byte for byte; TE's factory files differ only in how they print floats."""
    def normalise(appl: bytes) -> bytes:
        # TE's factory writer prints floats with trailing zeros ("523.25110", "7019846.50"); json.dumps does not
        text = re.sub(rb"(\d+\.\d*?[1-9])0+(?=[,}])", rb"\1", appl.rstrip(b" "))
        return re.sub(rb"(\d+\.)0+(?=[,}])", rb"\g<1>0", text)
    imports, factory = 0, 0
    for path in SAMPLER_FILES:
        pf = P.read_preset(path)
        out = str(tmp_path / os.path.basename(path))
        P.write_preset(out, pf)
        f1, c1, o1 = _chunks(path); f2, c2, o2 = _chunks(out)
        assert f1 == f2 == b"AIFC" and o1 == o2 == [b"FVER", b"COMM", b"APPL", b"SSND"]
        assert c1[b"FVER"] == c2[b"FVER"] and c1[b"COMM"] == c2[b"COMM"] and c1[b"SSND"] == c2[b"SSND"]
        assert P.read_preset(out).meta == pf.meta
        if S._is_field_import(pf.meta):
            assert open(out, "rb").read() == open(path, "rb").read(), f"{path} must round-trip byte for byte"
            imports += 1
        else:
            assert normalise(c1[b"APPL"]) == normalise(c2[b"APPL"]), path
            factory += 1
    assert imports >= 1


@needs_backup
def test_import_defaults_are_what_the_field_wrote():
    meta, src = S.sampler_template(HOME)
    assert os.path.isfile(src) and S._is_field_import(meta), src
    for k, v in S.SAMPLER_IMPORT_META.items():
        if k in ("knobs", "name", "base_freq"):
            continue
        assert meta[k] == v, k
    assert meta["knobs"][4:] == S.SAMPLER_IMPORT_META["knobs"][4:] and meta["base_freq"] == 440.0
    assert meta["knobs"][2] == meta["knobs"][3] == S.sampler_frame_to_knob(P._comm_fields(P.read_preset(src))[1], end=True)
    # with no backups the literal stands in, unchanged
    lit, where = S.sampler_template("/nonexistent/op-bridge-home")
    assert lit == S.SAMPLER_IMPORT_META and "SAMPLER_IMPORT_META" in where


# --------------------------------------------------------------------------- trimming a take

def test_trim_take_cuts_fades_and_normalises(tmp_path):
    sr = 48000
    t = np.arange(3 * sr) / sr
    take = np.stack([0.25 * np.sin(2 * np.pi * 220 * t), 0.25 * np.sin(2 * np.pi * 330 * t)], axis=1).astype(np.float32)
    src = str(tmp_path / "take.wav"); sf.write(src, take, sr, subtype="FLOAT")
    out = str(tmp_path / "cut.wav")
    r = S.trim_take(src, out, 1.0, 2.0, fade_ms=10, normalize_db=-1.0)
    assert r["frames"] == sr and r["seconds"] == 1.0 and r["channels"] == 2 and r["samplerate"] == sr and r["subtype"] == "PCM_24"
    assert r["fade_ms"] == 10.0 and r["fits_sampler"] is True and abs(r["peak_dbfs_before"] + 12.04) < 0.1
    info = sf.info(out)
    assert info.frames == sr and info.channels == 2 and info.subtype == "PCM_24"
    x, _ = sf.read(out, dtype="float32", always_2d=True)
    peak = float(np.max(np.abs(x)))
    assert abs(20 * np.log10(peak) + 1.0) < 0.05 and abs(r["peak_dbfs"] + 1.0) < 0.05
    fade = int(0.010 * sr)
    assert np.all(x[0] == 0) and np.all(np.abs(x[-1]) < 1e-3)
    assert np.max(np.abs(x[:fade // 4])) < 0.3 * peak and np.max(np.abs(x[-fade // 4:])) < 0.3 * peak
    assert np.max(np.abs(x[fade:-fade])) > 0.95 * peak
    # the cut is the right piece of the take: it starts where second 1.0 of the take starts
    ref = take[sr:2 * sr, 0]
    mid = slice(fade, sr - fade)
    corr = float(np.dot(x[mid, 0], ref[mid]) / (np.linalg.norm(x[mid, 0]) * np.linalg.norm(ref[mid])))
    assert corr > 0.9999
    # a 1.000 s cut becomes exactly the sampler preset the Field itself writes for a 1 s import: end knob 5462
    pre, notes = S.make_sampler_preset(out, "cut", "A4", catalog_path=str(tmp_path / "none.json"))
    assert pre.meta["knobs"] == [0, 0, 5462, 5462, 12000, 0, 0, 8192] and pre.meta["stereo"] is True
    assert P._comm_fields(pre) == (2, 44100, 16) and notes["audio"]["resampled"] == "48000 -> 44100 Hz"
    # to the end of the take, as float, without normalising
    r2 = S.trim_take(src, str(tmp_path / "tail.wav"), 2.5, subtype="FLOAT")
    assert r2["frames"] == sr // 2 and r2["end_s"] == 3.0 and sf.info(str(tmp_path / "tail.wav")).subtype == "FLOAT"
    assert abs(r2["peak_dbfs"] - r2["peak_dbfs_before"]) < 0.5 and r2["normalized_to_db"] is None
    # a region shorter than two fades still fades from and to zero
    r3 = S.trim_take(src, str(tmp_path / "tiny.wav"), 1.0, 1.0 + 8 / sr, fade_ms=10)
    y, _ = sf.read(str(tmp_path / "tiny.wav"), always_2d=True)
    assert r3["frames"] == 8 and y[0, 0] == 0 and abs(y[-1, 0]) < 1e-3
    for bad in ((3.5, None), (1.0, 0.5), (-1.0, 1.0)):
        with pytest.raises(ValueError):
            S.trim_take(src, str(tmp_path / "bad.wav"), *bad)
    with pytest.raises(ValueError):
        S.trim_take(src, str(tmp_path / "bad.wav"), 0.0, 1.0, subtype="PCM_32")
