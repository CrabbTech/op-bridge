"""Background jobs on the connected Field: a long play runs as a job while short calls are refused as busy,
its result is the same record a direct call returns, and cancelling releases the notes."""
import os
import time

import numpy as np
import pytest

from mcp.server.mcpserver.exceptions import ToolError

from op_bridge import server as srv
from op_bridge.device import Field
from op_bridge.session import Config
from tests.conftest import needs_field


def _score(seconds: float, tempo: float = 120.0, pitch: str = "C4"):
    beats = seconds * tempo / 60.0
    return {"tempo": tempo, "notes": [{"start": 0, "duration": beats, "pitch": pitch, "velocity": 90}], "tail_seconds": 0.2}


def _wait(job_id: str, limit: float):
    t0 = time.time()
    while time.time() - t0 < limit:
        st = srv.job_status(job_id)
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} still {st['status']} after {limit} s: {st}")


@needs_field
def test_short_play_is_direct_and_long_play_is_a_job():
    direct = srv.play(_score(1.0), record=True, name="job-test-short")
    assert direct.get("played") is True and "job_id" not in direct and direct.get("take_id")
    out = srv.play(_score(6.0), record=True, name="job-test-long", wait=False)
    assert out["status"] == "running" and out["job_id"] and out["expected_seconds"] > 6
    time.sleep(1.0)
    with pytest.raises(ToolError) as e:   # the device is held by the job; a short call fails fast instead of hanging
        srv.audition(["C4"], seconds=0.5)
    assert "busy" in str(e.value) and out["job_id"] in str(e.value)
    running = srv.job_status(out["job_id"])
    assert running["status"] == "running" and running["stage"] in ("playing", "analysing", "starting") and 0 < running["progress"] < 1
    assert any(j["job_id"] == out["job_id"] for j in srv.get_status()["jobs_running"])
    st = _wait(out["job_id"], 30)
    assert st["status"] == "done", st
    res = st["result"]
    assert res["played"] is True and res["take_id"] and os.path.exists(res["wav"]) and res["analysis"]["notes_heard"] >= 1
    assert st["progress"] == 1.0 and st["remaining_seconds"] == 0
    assert any(j["job_id"] == out["job_id"] and j["status"] == "done" for j in srv.list_jobs())


def _loudest_db(seconds: float) -> float:
    """Loudest USB channel over a fresh capture (the job records nothing itself, so the audio device is free)."""
    cfg = Config.load()
    with Field(channel=cfg.midi_channel - 1) as f:
        with f.recorder() as rec:
            time.sleep(seconds)
        r = rec.stop()
    x = r.audio.astype(np.float64) / (32768.0 if r.audio.dtype.kind == "i" else 1.0)
    return float(20 * np.log10(np.sqrt((x ** 2).mean(axis=0)).max() + 1e-9))


@needs_field
def test_cancel_releases_the_notes():
    out = srv.play(_score(40.0, pitch="E4"), record=False, name="job-test-cancel", wait=False)
    assert out["status"] == "running"
    time.sleep(1.5)
    during = _loudest_db(1.0)
    st = srv.cancel_job(out["job_id"])
    if st["status"] != "cancelled":
        st = _wait(out["job_id"], 5)
    assert st["status"] == "cancelled" and st["error"] == "cancelled", st
    time.sleep(2.5)                      # let the sound's release tail die
    after = _loudest_db(1.0)
    print(f"loudest channel while held {during:.1f} dB, 2.5 s after cancel {after:.1f} dB")
    assert during > -45, during          # the note was sounding
    assert after < during - 25 and after < -45, (during, after)


def test_job_status_unknown_id_is_a_tool_error():
    with pytest.raises(ToolError):
        srv.job_status("nope")
