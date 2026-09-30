"""Preset files: what the bridge writes must be what the Field reads. Uses real files from the disk backup when present."""
import os
import struct

import pytest

from op_bridge import presets as P

BACKUP = os.path.expanduser("~/Music/op-bridge/field-backup/2026-09-26-disk")


def _chunks(path):
    data = open(path, "rb").read()
    out = {}
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]; size = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        out[cid] = data[pos + 8:pos + 8 + size]
        pos += 8 + size + (size & 1)
    return data[8:12], out


def test_template_matches_the_fields_container(tmp_path):
    path = str(tmp_path / "7.aif")
    P.write_preset(path, P.make_synth_preset(P.silent_template(), "cluster", "opb test", knobs=[0.5, 0.3, 0.6, 0.2], fx="delay", lfo="tremolo"))
    form, ch = _chunks(path)
    assert form == b"AIFC"
    assert ch[b"FVER"] == bytes.fromhex("a2805140")
    comm = ch[b"COMM"]
    assert len(comm) == 64 and comm[18:22] == b"sowt" and comm[23:] == b"Signed integer (little-endian) linear PCM"
    appl = ch[b"APPL"]
    assert appl.startswith(b'op-1{"adsr":') and appl.rstrip(b" ").endswith(b'"type":"cluster"}\n') and len(appl) % 2 == 0
    assert any(ch[b"SSND"][8:]), "the sample must not be silent or the Field treats the file as a plain sample"
    back = P.read_preset(path)
    assert back.engine == "cluster" and back.name == "opb test" and back.meta["knobs"][:4] == [16384, 9830, 19660, 6553]


@pytest.mark.skipif(not os.path.isdir(BACKUP), reason="no disk backup of the Field on this machine")
def test_rewriting_a_field_file_keeps_its_layout(tmp_path):
    src = os.path.join(BACKUP, "synth/user/6.aif")
    original = P.read_preset(src)
    out = str(tmp_path / "6.aif")
    P.write_preset(out, P.make_synth_preset(original, "voltage", "opb volt", knobs=[0.5, 0.1, 0.5, 0.2]))
    f1, c1 = _chunks(src); f2, c2 = _chunks(out)
    assert f1 == f2 and c1[b"COMM"] == c2[b"COMM"] and c1[b"SSND"] == c2[b"SSND"]
    assert P.read_preset(out).engine == "voltage"


@pytest.mark.skipif(P.find_field_volume() is None, reason="Field not in disk mode")
def test_disk_mode_volume_has_the_slots():
    vol = P.find_field_volume()
    for kind in ("synth", "drum"):
        for slot in range(1, 9):
            assert os.path.exists(P.slot_path(vol, kind, slot))


@pytest.mark.skipif(not os.path.isdir(BACKUP), reason="no disk backup of the Field on this machine")
def test_learning_engine_identifiers_from_field_files(tmp_path):
    cat = str(tmp_path / "engine-catalog.json")
    r = P.learn_engines(BACKUP, cat)
    assert r["presets_read"] >= 16
    assert {"dimension", "digital", "string", "sampler"} <= set(r["synth_engines"])
    assert {"drum", "dbox"} <= set(r["drum_engines"])
    ids = P.known_engine_ids(cat)
    assert "cluster" not in ids["synth"], "cluster has not been seen in a Field-written file yet"
