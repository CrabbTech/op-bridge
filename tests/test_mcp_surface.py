"""Drives the real MCP server over stdio, the way Claude Desktop does. Device tools run only when the Field is connected."""
import json
import os
import sys

import pytest

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from op_bridge import knowledge
from tests.conftest import field_present


def _params(home: str) -> StdioServerParameters:
    env = dict(os.environ); env["OP_BRIDGE_HOME"] = home
    return StdioServerParameters(command=sys.executable, args=["-m", "op_bridge.cli", "serve"], env=env)


def _text(result):
    parts = []
    for c in result.content:
        if getattr(c, "type", "") == "text":
            parts.append(c.text)
    return "\n".join(parts)


@pytest.mark.anyio
async def test_tools_and_readonly_calls(home):
    async with stdio_client(_params(home)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            names = {t.name for t in tools.tools}
            for expected in ("get_status", "get_guide", "play", "record_to_tape", "author_synth_preset", "install_presets", "capture_seed", "set_constraints", "backup_tape",
                             "author_sampler_preset", "sample_from_take", "song_position", "hold_chord", "list_sequencers",
                             "job_status", "list_jobs", "cancel_job", "measure_take", "speech_intelligibility", "audition_slots"):
                assert expected in names, expected
            st = await s.call_tool("get_status", {})
            data = st.structured_content or json.loads(_text(st))
            assert data["mode"] in ("freeform", "guided")
            assert "constraints" in data
            g = await s.call_tool("get_guide", {"topic": "score"})
            assert "tempo" in _text(g)
            m = await s.call_tool("search_manual", {"query": "usb audio modes"})
            # the user guide is Teenage Engineering's and is added locally (docs/reference/README.md); without it the tool says how
            assert ("10 channel" if os.path.exists(knowledge.manual_path()) else "extract-manual.py") in _text(m)
            v = await s.call_tool("validate_score", {"score": {"tempo": 100, "notes": [{"start": 0, "duration": 1, "pitch": "C4"}, {"start": 0, "duration": 1, "pitch": "E4"}]}})
            vd = v.structured_content or json.loads(_text(v))
            assert vd["ok"] is True and vd["max_polyphony"] == 2


@pytest.mark.anyio
async def test_sampler_and_sequencer_tools_over_stdio(home, tmp_path):
    """The resampling loop at file level: a take in the session, cut with sample_from_take, staged with author_sampler_preset."""
    import numpy as np, soundfile as sf
    from op_bridge.session import Config, session_dir
    cfg = Config.load(); cfg.current_session = "stdio"; cfg.save()
    d = session_dir("stdio")
    t = np.arange(3 * 44100) / 44100
    sf.write(os.path.join(d, "takes", "take-stdio.wav"), np.stack([np.sin(2 * np.pi * 330 * t)] * 2, axis=1).astype("float32") * 0.4, 44100, subtype="FLOAT")
    async with stdio_client(_params(home)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            c = await s.call_tool("sample_from_take", {"take_id": "take-stdio", "start_s": 0.25, "end_s": 1.25, "name": "stdio cut"})
            cd = c.structured_content or json.loads(_text(c))
            assert cd["seconds"] == 1.0 and cd["fits_sampler"] is True and cd["sample"].endswith(os.path.join("samples", "stdio-cut.wav"))
            a = await s.call_tool("author_sampler_preset", {"slot": 5, "wav_path": cd["sample"], "name": "stdio", "root_note": "E4", "loop_in_s": 0.1, "loop_out_s": 0.9})
            ad = a.structured_content or json.loads(_text(a))
            assert ad["staged"] == os.path.join(home, "staging", "synth", "5.aif") and ad["preset"]["engine"] == "sampler"
            assert ad["region"]["root_note"] == "E4" and abs(ad["region"]["loop_out_s"] - 0.9) < 0.001 and ad["region"]["seconds"] == 1.0
            st = await s.call_tool("list_staged_presets", {})
            assert any(x["slot"] == 5 and x["engine"] == "sampler" for x in (st.structured_content or json.loads(_text(st)))["result"])
            bad = await s.call_tool("author_sampler_preset", {"slot": 5, "wav_path": os.path.join(d, "takes", "take-stdio.wav"), "name": "x", "direction": "sideways"})
            assert bad.is_error and "direction" in _text(bad)
            q = await s.call_tool("list_sequencers", {})
            seq = (q.structured_content or json.loads(_text(q)))["result"]
            assert [x["name"] for x in seq] == ["arpeggio", "endless", "finger", "hold", "pattern", "sketch", "tombola"]
            g = await s.call_tool("get_guide", {"topic": "sampler"})
            assert "32767" in _text(g) and "sample_from_take" in _text(g)
            g2 = await s.call_tool("get_guide", {"topic": "sequencers"})
            assert "TOMBOLA" in _text(g2)
            h = await s.call_tool("human_steps", {"topic": "arpeggio_setup"})
            assert "ARPEGGIO" in _text(h)
            h2 = await s.call_tool("human_steps", {"topic": "sample_from_input"})
            assert "input key" in _text(h2)
            # hold_chord refuses an impossible chord before it reaches the device
            hc = await s.call_tool("hold_chord", {"pitches": ["C3", "E3", "G3", "B3", "D4", "F4", "A4", "C5", "E5"], "beats": 2, "tempo": 120})
            hd = hc.structured_content or json.loads(_text(hc))
            assert hd["played"] is False and hd["problems"]


@pytest.mark.anyio
async def test_guided_mode_blocks_reserved_tools(home):
    from op_bridge.session import Config
    cfg = Config.load(); cfg.mode = "guided"; cfg.constraints.allow_master = False; cfg.constraints.allow_transport = False; cfg.constraints.key = "D minor"; cfg.save()
    try:
        async with stdio_client(_params(home)) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool("set_master", {"values": {"drive": 100}})
                assert res.is_error, "master bus should be reserved for the human in guided mode"
                assert "human" in _text(res).lower()
                v = await s.call_tool("validate_score", {"score": {"tempo": 100, "notes": [{"start": 0, "duration": 1, "pitch": "C#4"}]}})
                vd = v.structured_content or json.loads(_text(v))
                assert vd["ok"] is False and any("D minor" in p for p in vd["problems"])
                sp = await s.call_tool("song_position", {"beat": 4})
                assert sp.is_error and "human" in _text(sp).lower(), "the song position pointer is transport, reserved for the human"
                t = await s.call_tool("set_constraints", {"changes": {"allow_master": True, "max_polyphony": 4}})
                td = t.structured_content or json.loads(_text(t))
                assert td["constraints"]["allow_master"] is False, "the model must not be able to loosen constraints"
                assert td["constraints"]["max_polyphony"] == 4
    finally:
        cfg = Config.load(); cfg.mode = "freeform"; cfg.constraints.__init__(); cfg.save()


@pytest.mark.anyio
@pytest.mark.skipif(not field_present(), reason="OP-1 field not connected")
async def test_audition_and_play_through_mcp(home):
    async with stdio_client(_params(home)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            a = await s.call_tool("audition", {"pitches": ["C4", "G4"], "seconds": 1.0, "kind": "synth", "slot": 1})
            ad = a.structured_content or json.loads(_text(a))
            assert ad["heard"] is True, ad
            assert ad["peak_db"] > -60
            img = await s.call_tool("view_spectrogram", {"id": ad["id"]})
            assert any(getattr(c, "type", "") == "image" for c in img.content)
            p = await s.call_tool("play", {"score": {"tempo": 130, "notes": [{"start": i, "duration": 0.8, "pitch": n} for i, n in enumerate(["A3", "C4", "E4", "A4"])]}, "name": "mcp-test"})
            pd = p.structured_content or json.loads(_text(p))
            assert pd["played"] is True and pd["analysis"]["notes_heard"] >= 3, pd


@pytest.mark.anyio
@pytest.mark.skipif(not field_present(), reason="OP-1 field not connected")
async def test_job_round_trip_over_stdio(home):
    """A play longer than the client's patience comes back as a job over the wire, and job_status delivers the take."""
    import asyncio
    score = {"tempo": 120, "notes": [{"start": 0, "duration": 12, "pitch": "C4", "velocity": 80}], "tail_seconds": 0.2}
    async with stdio_client(_params(home)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool("play", {"score": score, "record": True, "name": "mcp-job", "wait": False})
            data = res.structured_content or json.loads(_text(res))
            assert data["status"] == "running" and data["job_id"], data
            busy = await s.call_tool("audition", {"pitches": ["C4"], "seconds": 0.5})
            assert busy.is_error and "busy" in _text(busy)
            for _ in range(60):
                st = await s.call_tool("job_status", {"job_id": data["job_id"]})
                sd = st.structured_content or json.loads(_text(st))
                if sd["status"] != "running":
                    break
                await asyncio.sleep(0.5)
            assert sd["status"] == "done", sd
            assert sd["result"]["take_id"] and sd["result"]["analysis"]["notes_heard"] >= 1
            listed = await s.call_tool("list_jobs", {})
            assert data["job_id"] in _text(listed)
