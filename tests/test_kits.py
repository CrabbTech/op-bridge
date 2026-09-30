"""Kit maps: the drum key classifier and kit maps against the Field-written kits, the live capture of kit 1 and the
owner's labels by ear. Every test reads real files; nothing here opens a MIDI or audio device. Tests that need the disk
backup, the live clips or the labelled map skip when those are not on this machine."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

from op_bridge import kits as K
from op_bridge import presets as P

BACKUP = os.path.expanduser("~/Music/op-bridge/field-backup/2026-09-26-disk")
KIT_FILES = sorted(glob.glob(os.path.join(BACKUP, "drum/user/*.aif")))
SYNTH_FILES = sorted(glob.glob(os.path.join(BACKUP, "synth/user/*.aif")))
CLIPS = os.path.expanduser("~/Music/op-bridge/probe/kit1")
LABELS = os.path.join(os.path.dirname(__file__), "fixtures", "kit1-owner-labels.json")

needs_backup = pytest.mark.skipif(not KIT_FILES, reason="no disk backup of the Field on this machine")
needs_clips = pytest.mark.skipif(not glob.glob(os.path.join(CLIPS, "*.wav")), reason="no live clips of kit 1 on this machine")
needs_labels = pytest.mark.skipif(not os.path.isfile(LABELS), reason="no labelled kit map on this machine")


@pytest.fixture(scope="module")
def file_maps() -> dict[str, K.KitMap]:
    """kit_map_from_file on every sampler kit of the backup, keyed by file name (the dbox kit is left out)."""
    out = {}
    for path in KIT_FILES:
        if P.read_preset(path).engine == "drum":
            out[os.path.basename(path)] = K.kit_map_from_file(path)
    return out


@pytest.fixture(scope="module")
def clip_map() -> K.KitMap:
    return K.kit_map_from_clips(CLIPS)


@pytest.fixture(scope="module")
def owner() -> K.KitMap:
    return K.load_kit_map_file(LABELS)


def _labelled(owner: K.KitMap) -> dict[int, str]:
    return {m: k.label for m, k in owner.keys.items() if k.human}


# --------------------------------------------------------------------------- names and features

def test_key_names_follow_the_field_display():
    assert K.field_key_name(53) == "F2" and K.midi_key_name(53) == "F3"
    assert K.field_key_name(61) == "C#3" and K.field_key_name(63) == "D#3" and K.field_key_name(76) == "E4"
    assert K.midi_key_name(63) == "D#4" and K.midi_key_name(76) == "E5"
    for midi in range(53, 77):
        assert K.field_name_to_midi(K.field_key_name(midi)) == midi
        assert K.key_index(midi) == midi - 53
        assert K.field_key_name(midi) == P.drum_key_name(midi - 53)[:-1] + str(int(P.drum_key_name(midi - 53)[-1]) - 1)
    assert K.field_name_to_midi("Eb3") == 63
    with pytest.raises(ValueError, match="F2"):
        K.field_name_to_midi("C2")
    with pytest.raises(ValueError, match="24 keys"):
        K.key_index(80)


def test_key_features_and_classifier_survive_short_and_silent_audio():
    for audio in (np.zeros(44100, dtype=np.float32), np.zeros(5, dtype=np.float32), np.zeros((300, 2)), np.zeros(0),
                  np.full(44100, 1e-5, dtype=np.float32)):
        f = K.key_features(audio, 44100)
        assert f["silent"] is True and f["decay_40_ms"] is None and set(f["bands"]) == set(K.BAND_NAMES)
        label, conf, reason = K.classify_key(f)
        assert label == "silent" and conf == 1.0 and reason
        assert K.combine_with_prior(53, label, conf, reason) == (label, conf, reason)
    with pytest.raises(ValueError, match="sample rate"):
        K.key_features(np.zeros(100), 0)


def _decaying(sr: int, seconds: float, tau: float, rng: np.random.Generator, tone_hz: float | None) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    body = np.sin(2 * np.pi * tone_hz * t) if tone_hz else rng.standard_normal(t.size)
    return (0.8 * body * np.exp(-t / tau)).astype(np.float32)


def test_synthetic_hits_land_in_the_expected_classes():
    """A decaying 55 Hz sine is a kick, a 30 ms white-noise burst a closed hat, a 500 ms one an open hat; the same
    features come from stereo and int16 input."""
    sr, rng = 44100, np.random.default_rng(7)
    kick = _decaying(sr, 0.6, 0.08, rng, 55.0)
    fk = K.key_features(kick, sr)
    assert fk["silent"] is False and fk["bands"]["sub"] > 0.9 and fk["f0_hz"] < 65
    label, conf, reason = K.classify_key(fk)
    assert label == "kick" and conf >= 0.7 and "below 400 Hz" in reason
    ch = _decaying(sr, 0.4, 0.012, rng, None)
    assert K.classify_key(K.key_features(ch, sr))[0] == "closed hat"
    oh = _decaying(sr, 1.2, 0.09, rng, None)
    foh = K.key_features(oh, sr)
    assert K.classify_key(foh)[0] == "open hat", K.class_scores(foh)
    stereo = np.stack([kick, kick], axis=1)
    assert K.key_features(stereo, sr)["centroid_hz"] == fk["centroid_hz"]
    as_int = (kick * 32767).astype(np.int16)
    assert abs(K.key_features(as_int, sr)["peak_db"] - fk["peak_db"]) < 0.1
    # a cut clip reports the decay it could not see as None and flags it
    cut = K.key_features(oh[: int(0.1 * sr)], sr, clip=True)
    assert cut["truncated"] and cut["decay_40_ms"] is None
    assert any("cut" in fl for fl in K.classify_key_detail(cut)["flags"])


# --------------------------------------------------------------------------- kit 1: clips, file and the owner's labels

@needs_clips
@needs_labels
def test_kit1_clips_match_the_owner_labels(clip_map, owner):
    labelled = _labelled(owner)
    assert len(labelled) == 9 and len(clip_map.keys) == 24
    wrong = {m: (exp, clip_map.keys[m].label) for m, exp in labelled.items() if clip_map.keys[m].label != exp}
    assert len(wrong) <= 1, wrong
    for m in labelled:
        key = clip_map.keys[m]
        assert key.confidence is not None and key.confidence >= 0.5, (m, key.label, key.confidence, key.reason)
        assert key.reason and key.field_name == K.field_key_name(m)
    roles = clip_map.roles
    assert roles["BD"] == 53 and roles["SN"] == 55 and roles["CH"] in (60, 61, 62, 64) and roles["OH"] == 63, roles
    assert roles.get("CH_CHOKE") == 61 and "chokes" in clip_map.keys[61].reason
    # the 600 ms clips cut five long keys (the crash on G#3 among them): each is flagged and none is guessed as a kick
    for m in (68, 71, 72, 74, 76):
        key = clip_map.keys[m]
        assert any("cut" in fl for fl in key.flags) and key.features["decay_40_ms"] is None, (m, key.flags)
    assert clip_map.keys[68].label in ("cymbal", "open hat", "unsure")
    # a tail leaking into the next clip is flagged when it is audible before the onset
    assert any("leak" in fl for fl in clip_map.keys[57].flags)
    assert clip_map.clips_dir == CLIPS and clip_map.keys[68].extra["clip"].startswith("68_")


@needs_backup
@needs_clips
@needs_labels
def test_kit1_file_agrees_with_the_clips_on_labelled_keys(file_maps, clip_map, owner):
    km = file_maps["1.aif"]
    assert km.kit_name == "hard spunch" and km.slot == 1 and len(km.keys) == 24
    labelled = _labelled(owner)
    disagree = {m: (km.keys[m].label, clip_map.keys[m].label) for m in labelled if km.keys[m].label != clip_map.keys[m].label}
    assert not disagree, disagree
    wrong = {m: (exp, km.keys[m].label) for m, exp in labelled.items() if km.keys[m].label != exp}
    assert len(wrong) <= 1, wrong
    # per-key facts from the file: the choke marker on C#3 and D#3, region lengths, the volume law
    for m in (61, 63):
        assert km.keys[m].playmode == P.DRUM_PLAYMODE_HIHAT and any("choke" in fa for fa in km.keys[m].facts)
    assert km.keys[63].region_ms > km.keys[61].region_ms
    assert km.keys[68].label == "cymbal" and km.keys[68].features["decay_40_ms"] > 1000
    for m, key in km.keys.items():
        assert key.features["expected_live_peak_db"] == pytest.approx(K.predict_live_peak_db(key.features["peak_db"], key.volume), abs=0.06)
    # the live clips sit where the level law says, within about a decibel, on the keys the clips did not cut
    for m in labelled:
        assert abs(clip_map.keys[m].features["peak_db"] - km.keys[m].features["expected_live_peak_db"]) < 1.5, m


# --------------------------------------------------------------------------- every kit in the backup

@needs_backup
def test_every_sampler_kit_maps_with_kick_on_f2_and_open_hat_on_dsharp3(file_maps):
    assert len(file_maps) == 7, sorted(file_maps)
    kicks = [name for name, km in file_maps.items() if km.keys[53].label == "kick"]
    assert len(kicks) >= 5, kicks
    for name, km in file_maps.items():
        assert len(km.keys) == 24 and set(km.keys) == set(range(53, 77)), name
        assert km.keys[63].label in ("open hat", "closed hat", "unsure"), (name, km.keys[63].label, km.keys[63].reason)  # short hats sit on the open-hat key in some kits
        assert km.keys[63].label != "kick"
        for m, key in km.keys.items():
            assert key.label in K.LABELS or key.label == K.UNSURE, (name, m, key.label)
            assert key.reason and key.facts and key.playmode in P.DRUM_PLAYMODE_VALUES, (name, m)
            assert key.source in ("audio", "audio+layout")
            if key.label == K.UNSURE:
                assert key.candidates, (name, m)
        assert km.roles["BD"] == 53 and km.keys[km.roles["OH"]].label in ("open hat", "closed hat", "cymbal"), (name, km.roles)
        if km.keys[63].label == "open hat":
            assert km.roles["OH"] == 63, (name, km.roles)
        assert km.keys[61].playmode == P.DRUM_PLAYMODE_HIHAT and km.keys[63].playmode == P.DRUM_PLAYMODE_HIHAT
        assert km.source.startswith("classified from the kit file")
    exoex = file_maps["5.aif"]
    assert exoex.kit_name == "exoex" and all(k.stacked_ab and k.channel_a and k.channel_b for k in exoex.keys.values())
    assert exoex.keys[61].channel_b == "closed hat" and "channel B" in exoex.keys[61].reason
    # departures from the convention are reported as such, never hidden: the 808 tom on workshop's C3 stays a tom
    workshop = file_maps["7.aif"]
    assert workshop.keys[60].label == "tom" and "factory layout expects closed hat" in workshop.keys[60].reason
    # a high-confidence short hat on the open-hat key is reported as the kit's open hat with the audio verdict kept
    apes = file_maps["4.aif"]
    assert apes.keys[63].label == "closed hat" and "open-hat key" in apes.keys[63].reason


@needs_backup
def test_dbox_kit_gets_a_layout_only_map():
    path = os.path.join(BACKUP, "drum/user/3.aif")
    if P.read_preset(path).engine != "dbox":
        pytest.skip("slot 3 of this backup is not a dbox kit")
    km = K.kit_map_from_file(path)
    assert km.kit_name == "whathump" and km.slot == 3 and len(km.keys) == 24
    assert "dbox" in km.source and all(k.source == "layout" for k in km.keys.values())
    assert km.keys[53].label == "kick" and km.keys[53].confidence == 0.5 and "factory layout" in km.keys[53].reason
    assert km.keys[57].label == K.UNSURE and any(fa.startswith("dbox voice") for fa in km.keys[57].facts)
    assert km.roles["BD"] == 53 and km.roles["OH"] == 63


@needs_backup
def test_errors_name_the_problem(tmp_path):
    with pytest.raises(ValueError, match="no such file"):
        K.kit_map_from_file(str(tmp_path / "missing.aif"))
    if SYNTH_FILES:
        with pytest.raises(ValueError, match="not a drum kit"):
            K.kit_map_from_file(SYNTH_FILES[0])
    with pytest.raises(ValueError, match="not a directory"):
        K.kit_map_from_clips(str(tmp_path / "nowhere"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="<midi>_<name>.wav"):
        K.kit_map_from_clips(str(empty))
    with pytest.raises(ValueError, match="slot"):
        K.load_kit_map(str(tmp_path), 0)
    with pytest.raises(ValueError, match="not valid JSON"):
        K.KitMap.from_json("{not json")


# --------------------------------------------------------------------------- JSON, merging, storage

@needs_labels
def test_json_round_trip_matches_the_owner_file(owner):
    raw = json.load(open(LABELS))
    assert owner.kit_name == "hard spunch" and owner.slot == 1
    assert set(owner.keys) == {int(m) for m in raw["keys"]} and owner.roles == raw["roles"]
    for m, kd in raw["keys"].items():
        key = owner.keys[int(m)]
        assert key.human and key.confidence == 1.0 and key.label == kd["label"] and key.field_name == kd["field_name"]
        assert key.notes == kd.get("notes", "")
    assert owner.extra["features_file"] == raw["features_file"] and owner.clips_dir == raw["clips_dir"]
    assert owner.path == LABELS and "path" not in owner.to_dict()
    again = K.KitMap.from_json(owner.to_json())
    assert again.to_dict() == owner.to_dict()
    d = again.to_dict()
    assert d["keys"]["61"]["notes"] == "chokes the open hat on D#3" and d["roles"]["CH_CHOKE"] == 61
    assert K.KitMap.from_json(json.dumps(raw)).to_dict() == owner.to_dict()


@needs_clips
@needs_labels
def test_merge_keeps_human_labels_over_the_classifier(clip_map, owner):
    merged = clip_map.merge(owner)
    assert merged.kit_name == "hard spunch" and merged.slot == 1
    for m, key in owner.keys.items():
        mk = merged.keys[m]
        assert mk.human and mk.label == key.label and mk.notes == key.notes and mk.confidence == 1.0
        assert mk.reason == clip_map.keys[m].reason           # the classifier's evidence stays readable
    assert sum(1 for k in merged.keys.values() if not k.human) == 15
    assert merged.roles["CH"] == 60 and merged.roles["CH_CHOKE"] == 61 and merged.roles["OH"] == 63
    # a human label that contradicts the audio wins and the audio verdict moves aside
    human = K.KitMap(kit_name="hard spunch", keys={60: K.KitKey(midi=60, label="shaker", notes="by ear", source="human", confidence=1.0),
                                                  70: K.KitKey(midi=70, label="rim", source="human")}, roles={"CH": 62})
    m2 = K.merge_kit_maps(clip_map, human)
    assert m2.keys[60].label == "shaker" and m2.keys[60].audio_label == "closed hat" and m2.keys[60].notes == "by ear"
    assert m2.keys[70].label == "rim" and m2.keys[70].human
    assert m2.roles["CH"] == 62 and m2.roles["BD"] == 53
    assert clip_map.keys[60].label == "closed hat"           # the input map is untouched
    # a machine-labelled key in the other map does not override anything
    machine = K.KitMap(keys={53: K.KitKey(midi=53, label="tom", source="audio", confidence=0.9)})
    assert K.merge_kit_maps(clip_map, machine).keys[53].label == "kick"


@needs_backup
@needs_labels
def test_save_and_load_kit_maps_under_home(tmp_path, file_maps, owner):
    home = str(tmp_path / "home")
    assert K.load_kit_map(home, 1) is None
    path = K.save_kit_map(home, 1, owner)
    assert path == os.path.join(home, "kits", "drum-slot-1-hard-spunch.json") and os.path.isfile(path)
    back = K.load_kit_map(home, 1)
    assert back is not None and back.to_dict()["keys"] == owner.to_dict()["keys"] and back.roles == owner.roles
    # saving the classifier's map for the same slot keeps every label by ear and adds the other 15 keys
    path2 = K.save_kit_map(home, 1, file_maps["1.aif"])
    assert path2 == path
    saved = K.load_kit_map(home, 1)
    assert len(saved.keys) == 24
    for m, key in owner.keys.items():
        assert saved.keys[m].human and saved.keys[m].label == key.label and saved.keys[m].notes == key.notes
    assert saved.keys[68].label == "cymbal" and not saved.keys[68].human
    assert saved.roles["CH"] == 60 and saved.roles["CH_CHOKE"] == 61 and saved.roles["CY"] == 68
    assert json.load(open(path))["keys"]["61"]["notes"] == "chokes the open hat on D#3"
    # a different kit in that slot is saved beside it and becomes the newest map for the slot
    other = file_maps["8.aif"]
    path3 = K.save_kit_map(home, 1, other)
    assert path3.endswith("drum-slot-1-cherry.json") and path3 != path
    os.utime(path3, None)
    assert K.load_kit_map(home, 1).kit_name == "cherry"
    # keep_human_labels=False writes the map exactly as given
    K.save_kit_map(home, 4, file_maps["1.aif"], keep_human_labels=False)
    assert not any(k.human for k in K.load_kit_map(home, 4).keys.values())
    assert K.load_kit_map(home, 4).slot == 4


# --------------------------------------------------------------------------- the layout prior on its own

def test_combine_with_prior_refines_hats_and_breaks_ties_only():
    label, conf, reason = K.combine_with_prior(61, "closed hat", 1.0, "bright and short")
    assert (label, conf) == ("closed hat", 1.0) and "chokes" in reason
    label, conf, reason = K.combine_with_prior(63, "closed hat", 1.0, "bright and short", playmode=P.DRUM_PLAYMODE_HIHAT)
    assert label == "closed hat" and conf == 1.0 and "open-hat key" in reason and "roles follow the audio" in reason
    label, conf, reason = K.combine_with_prior(63, "closed hat", 1.0, "bright and short", playmode=P.DRUM_PLAYMODE_DEFAULT)
    assert label == "closed hat" and conf == 1.0 and "expects open hat" in reason
    label, conf, reason = K.combine_with_prior(60, "tom", 1.0, "pitched")
    assert (label, conf) == ("tom", 1.0) and "expects closed hat" in reason
    assert K.combine_with_prior(63, "kick", 1.0, "low")[0] == "kick"
    scores = {"kick": 0.0, "snare": 0.0, "clap": 0.0, "closed hat": 0.2, "open hat": 0.88, "tom": 0.0, "cymbal": 0.8, "perc": 0.12}
    label, conf, reason = K.combine_with_prior(63, "unsure", 0.45, "between open hat and cymbal", scores)
    assert label == "open hat" and conf == 0.6 and "layout" in reason
    far = {"kick": 1.0, "snare": 0.0, "clap": 0.0, "closed hat": 0.0, "open hat": 0.3, "tom": 0.6, "cymbal": 0.0, "perc": 0.0}
    label, conf, reason = K.combine_with_prior(63, "unsure", 0.45, "between kick and tom", far)
    assert label == "unsure" and conf == 0.45 and "only 0.30" in reason
    assert K.combine_with_prior(70, "unsure", 0.4, "x") == ("unsure", 0.4, "x")
    assert K.combine_with_prior(58, "unsure", 0.4, "x")[0] == "unsure"      # clap on A#2 is too weak a convention
    label, conf, reason = K.combine_with_prior(55, "unsure", 0.4, "x")
    assert label == "snare" and conf == 0.6 and "layout" in reason
    assert set(K.FACTORY_LAYOUT) == {53, 54, 55, 56, 58, 60, 61, 62, 63, 64, 68}
