"""Audio routing and subprocess lifetime tests; no device, GPU or model download."""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from op_bridge import listener as L
from op_bridge.jobs import Cancelled


def write_audio(path, channels=2, seconds=3):
    sr = 48000
    t = np.arange(seconds * sr) / sr
    audio = np.zeros((len(t), channels), dtype=np.float32)
    audio[:, :min(2, channels)] = (.2 * np.sin(2*np.pi*440*t))[:, None]
    if channels == 10:
        audio[:, 8:10] = (.7 * np.sin(2*np.pi*1200*t))[:, None]
    sf.write(path, audio, sr, subtype="FLOAT")
    return audio


def test_multichannel_excerpt_routes_main_mix_and_preserves_source(tmp_path):
    source, dest = tmp_path / "source.wav", tmp_path / "excerpt.wav"
    write_audio(source, channels=10)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence = L.prepare_excerpt(source, dest, .5, 1.5)
    audio, sr = sf.read(dest)
    assert sr == 16000 and len(audio) == 24000 and audio.ndim == 1
    assert evidence["channels"] == [1, 2] and evidence["start_s"] == .5 and evidence["end_s"] == 2
    spectrum = np.abs(np.fft.rfft(audio))
    assert np.fft.rfftfreq(len(audio), 1/sr)[spectrum.argmax()] == 440
    explicit = L.prepare_excerpt(source, dest, 1, 1, [9, 10])
    assert explicit["measured"]["rms_dbfs"] > evidence["measured"]["rms_dbfs"] + 10
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


def test_antiphase_is_not_misreported_as_silence(tmp_path):
    source, dest = tmp_path / "source.wav", tmp_path / "excerpt.wav"
    audio = write_audio(source)
    audio[:, 1] *= -1
    sf.write(source, audio, 48000, subtype="FLOAT")
    result = L.prepare_excerpt(source, dest, 0, 2)
    assert not result["silent"] and "cancellation" in result["notes"][0]
    assert np.max(np.abs(sf.read(dest)[0])) > .19


@pytest.mark.parametrize("channels,start", [([0], 0), ([11], 0), ([1, 1], 0), ([], 0), (None, 3), (None, 2.5)])
def test_bad_channel_or_short_window_is_rejected(tmp_path, channels, start):
    source = tmp_path / "source.wav"
    write_audio(source, channels=10)
    with pytest.raises(ValueError):
        L.prepare_excerpt(source, tmp_path / "excerpt.wav", start, 1, channels)


def fake_runtime(tmp_path, body):
    root = tmp_path / "runtime"
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(sys.executable)
    (root / "adapter.py").write_text("import json,os,sys,time\nfrom pathlib import Path\nr=json.load(sys.stdin)\n" + body)
    return root


def protocol_result():
    return {"type": "result", "observations": "A repeated percussive pulse.",
            "model_id": L.MODEL_ID, "model_revision": L.MODEL_REVISION}


def test_protocol_progress_and_result(tmp_path):
    root = fake_runtime(tmp_path, "print(json.dumps({'type':'progress','message':'testing'}),flush=True)\nprint(" + repr(json.dumps(protocol_result())) + ",flush=True)\n")
    stages = []
    result = L.run_adapter(root, tmp_path / "excerpt.wav", "What sounds?", 128, tmp_path / "log", None, time.monotonic()+5, stages.append)
    assert stages == ["testing"] and result["observations"] == "A repeated percussive pulse."


@pytest.mark.parametrize("output", ["not json", "[]", json.dumps({**protocol_result(), "model_revision": "wrong"}),
                                   json.dumps(protocol_result()) + "\n" + json.dumps(protocol_result()), "x" * 70000])
def test_bad_protocol_is_not_trusted(tmp_path, output):
    root = fake_runtime(tmp_path, "print(" + repr(output) + ",flush=True)\n")
    with pytest.raises(RuntimeError):
        L.run_adapter(root, tmp_path / "excerpt.wav", "What sounds?", 128, tmp_path / "log", None, time.monotonic()+5, lambda _: None)


@pytest.mark.parametrize("cancelled", [False, True])
def test_timeout_and_cancellation_reap_the_process(tmp_path, cancelled):
    root = fake_runtime(tmp_path, "Path('child.pid').write_text(str(os.getpid()))\ntime.sleep(30)\n")
    event = threading.Event()
    timer = threading.Timer(.3, event.set) if cancelled else None
    if timer: timer.start()
    try:
        with pytest.raises(Cancelled if cancelled else RuntimeError):
            L.run_adapter(root, tmp_path / "excerpt.wav", "What sounds?", 128, tmp_path / "log", event,
                          time.monotonic() + (5 if cancelled else .4), lambda _: None)
    finally:
        if timer: timer.join()
    pid = int((tmp_path / "child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_missing_or_malformed_runtime_is_available_false(tmp_path, monkeypatch):
    monkeypatch.setenv("OP_BRIDGE_LISTENER_HOME", str(tmp_path))
    assert not L.status(tmp_path)["available"]
    (tmp_path / "runtime.json").write_text("[]")
    assert not L.status(tmp_path)["available"]


def test_silence_skips_inference_and_saves_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "status", lambda _: {"available": True})
    monkeypatch.setattr(L, "run_adapter", lambda *a: pytest.fail("silence should skip the model"))
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros(16000), 16000)
    result = L.listen(tmp_path, source, tmp_path / "reviews", seconds=1)
    assert result["model_invoked"] is False and result["evidence"]["silent"]
    assert Path(result["report"]).is_file()


def test_server_tool_resolves_take_and_runs_as_job(tmp_path, monkeypatch):
    from op_bridge import server as srv, session
    monkeypatch.setattr(session, "HOME", str(tmp_path))
    monkeypatch.setattr(srv, "HOME", str(tmp_path))
    cfg = session.Config.load()
    take = Path(session.session_dir(cfg.current_session)) / "takes/test.wav"
    write_audio(take)
    seen = []
    def listen(home, source, report_dir, **kwargs):
        seen.append((home, source, report_dir, kwargs))
        return {"model_invoked": False, "observations": "test"}
    monkeypatch.setattr(L, "listen", listen)
    out = srv.listen_to_take("test", seconds=1)
    job = srv.JOBS.get(out["job_id"])
    deadline = time.monotonic()+2
    while not job.done and time.monotonic() < deadline: time.sleep(.01)
    assert job.status == "done" and seen[0][1] == str(take)
    assert seen[0][3]["cancel"] is job.cancel
    assert "16 kHz mono" in srv.get_guide("listening")
