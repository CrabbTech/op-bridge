"""The op-bridge MCP server: lets a model wield a connected OP-1 field.

Runs over stdio (Claude Desktop, Claude Code, Cursor) or Streamable HTTP (claude.ai and ChatGPT
custom connectors, through a public HTTPS tunnel). Every tool talks to the real device.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

from mcp.server.mcpserver import MCPServer, Image
from mcp.server.mcpserver.exceptions import ToolError
import contextlib
import functools
import numpy as np
from mcp.types import ToolAnnotations

from . import analysis as an
from . import listener as LI
from . import knowledge, presets as P, sampler as SM, sequencers as SQ, tape as T
from .device import Field, find_midi_output, find_audio_input, midi_output_names, audio_devices, usb_layout
from .jobs import Jobs, Job, Cancelled
from .player import play_score, save_take
from .score import Score, compile_events, max_polyphony, validate, midi_to_name, pitch_to_midi
from .seed import capture_seed as _capture_seed
from .session import HOME, Config, Constraints, session_dir, list_sessions as _list_sessions, new_id, write_json, read_json, read_state, update_state

INSTRUCTIONS = """op-bridge controls a Teenage Engineering OP-1 field that is plugged into this computer.
Start with get_guide("quickstart") (five moves, a minute to read), then get_status. Scores follow
get_guide("score"); the other guides (workflow, drums, arrangement, sounds, effects, sampler, sequencers,
device, manual) are there for specific questions. Two modes: freeform (every tool) and guided (the human
owns settings and sets constraints; read them in get_status). Some steps need the human at the device
(arm recording, select a tape track, enter disk mode, enable a sequencer, toggle an effect): ask with the
exact key presses from human_steps. Anything longer than about 25 s comes back at once as a job: poll
job_status(job_id) for the same result a direct call gives, and send no other device calls meanwhile.
For actual recorded-audio descriptions, use listener_status then listen_to_take; see get_guide("listening").
Measurements and the text-only Jev judge are not listening. Audio-model descriptions can also be wrong;
compare stems and variants before changing an arrangement. Words are optional; nothing here assumes a vocal."""

_server = MCPServer("op-bridge", instructions=INSTRUCTIONS, version="0.1.0")


class _Registrar:
    """Wraps every tool so refusals and device errors reach the model as readable tool errors."""

    def __init__(self, server):
        self._s = server

    def __getattr__(self, name):
        return getattr(self._s, name)

    def tool(self, **kw):
        def deco(fn):
            @functools.wraps(fn)
            def wrapped(*a, **k):
                try:
                    return fn(*a, **k)
                except ToolError:
                    raise
                except (PermissionError, ValueError, RuntimeError, FileNotFoundError, KeyError) as e:
                    raise ToolError(f"{type(e).__name__}: {e}") from e
            return self._s.tool(**kw)(wrapped)
        return deco


mcp = _Registrar(_server)
_device_lock = threading.Lock()
RO = ToolAnnotations(read_only_hint=True, open_world_hint=False)
RW = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)


def _cfg() -> Config:
    return Config.load()


def _field(cfg: Config) -> Field:
    f = Field(channel=cfg.midi_channel - 1)
    if not f.connected:
        raise RuntimeError("OP-1 field not found over USB MIDI. Connect it with USB-C and leave it in normal mode (not disk or MTP mode).")
    return f


JOBS = Jobs()
SYNC_LIMIT = float(os.environ.get("OP_BRIDGE_SYNC_LIMIT", "25"))   # seconds a tool call may take before the work becomes a job


@contextlib.contextmanager
def _device(cfg: Config, timeout: float = 3.0):
    """The connected Field, held exclusively. A call that cannot get it within `timeout` fails with a message
    naming the job that holds it, instead of hanging until the client gives up."""
    if not _device_lock.acquire(timeout=timeout):
        running = JOBS.running()
        if running:
            j = running[0].snapshot()
            raise RuntimeError(f"the Field is busy with job {j['job_id']} ({j['kind']} '{j['name']}', {j['stage']}, about {j['remaining_seconds']} s left): poll job_status, or cancel_job, then retry")
        raise RuntimeError("the Field is busy with another call; retry in a moment")
    try:
        with _field(cfg) as f:
            yield f
    finally:
        _device_lock.release()


def _stage(job: Job | None, text: str) -> None:
    if job is not None:
        job.stage = text


def _cancel(job: Job | None):
    return job.cancel if job is not None else None


def _run_or_job(kind: str, name: str, expected_seconds: float, wait: bool | None, work) -> dict[str, Any]:
    """Run `work(job)` inline when it fits in a tool call, else on a background job. wait=True forces inline
    (the caller accepts the client's timeout), wait=False forces a job, None decides by expected length."""
    if wait is True or (wait is None and expected_seconds <= SYNC_LIMIT):
        return work(None)
    job = JOBS.start(new_id(f"job-{name}"), kind, name, expected_seconds, work)
    return {"job_id": job.id, "status": "running", "kind": kind, "expected_seconds": round(expected_seconds, 1),
            "note": f"about {expected_seconds:.0f} s of work, longer than a chat tool call allows, so it runs in the background: poll job_status(\"{job.id}\") every few seconds until status is done; its result is exactly what a direct call returns. cancel_job stops it and releases the notes."}


def _expected(s: Score, extra: float = 2.5) -> float:
    return float(s.seconds()) + 0.5 + extra


def _sess_dir(cfg: Config) -> str:
    return session_dir(cfg.current_session)


def _score(score: dict[str, Any] | str) -> Score:
    if isinstance(score, str):
        score = json.loads(score)
    return Score.model_validate(score)


def _human(cfg: Config) -> list[str]:
    notes = []
    a = find_audio_input()
    if a is not None and a["inputs"] < 10:
        notes.append("USB audio mode is %d channel; set it to 10 channel in COM > T1 > system > usb audio to capture the main mix as well as the tracks." % a["inputs"])
    return notes


# ------------------------------------------------------------------ status, knowledge, session

@mcp.tool(annotations=RO)
def get_status() -> dict[str, Any]:
    """Connection state of the OP-1 field, bridge mode and constraints, current session, staged presets and palette notes. Call this first."""
    cfg = _cfg()
    midi = find_midi_output(); audio = find_audio_input()
    d = _sess_dir(cfg)
    takes = sorted(f[:-5] for f in os.listdir(os.path.join(d, "takes")) if f.endswith(".json"))
    seeds = sorted(f[:-5] for f in os.listdir(os.path.join(d, "seeds")) if f.endswith(".json"))
    return {
        "field": {"midi_port": midi, "audio_device": audio and audio["name"], "usb_audio_channels": audio and audio["inputs"],
                  "usb_layout": usb_layout(audio["inputs"]) if audio else None, "midi_channel": cfg.midi_channel,
                  "connected": bool(midi and audio), "hint": None if (midi and audio) else "no OP-1 MIDI port and audio device on this computer: connect the Field over USB-C in normal mode (the bridge only uses disk mode for install_presets)"},
        "mode": cfg.mode,
        "field_state_last_set_by_bridge": read_state(),
        "field_state_note": "the Field cannot be queried; this is what the bridge last set (mode, slot, tempo). Changes made by hand on the device are unknown until the bridge sets them again.",
        "constraints": cfg.constraints.__dict__,
        "jobs_running": [j.snapshot() for j in JOBS.running()],
        "session": cfg.current_session, "takes": takes[-20:], "seeds": seeds[-20:],
        "samples": _list_samples(d)[-20:],
        "staged_presets": P.staged_presets(HOME),
        "palette": cfg.palette,
        "latency_ms": cfg.latency_ms,
        "notes_for_human": _human(cfg),
        "judge": {"mode": cfg.judge, "active": JU.active_judge(), "note": "local rules always work; Jev ranks by cached taxonomy plus a rubric match when enabled (op-bridge judge auto|local|jev)"},
        "listener": LI.status(HOME),
        "home": HOME,
    }


def _list_samples(session_d: str) -> list[dict[str, Any]]:
    """The cuts sample_from_take made in a session: WAV basenames with their length, oldest first."""
    d = os.path.join(session_d, "samples")
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(".wav"):
            continue
        entry: dict[str, Any] = {"file": f}
        try:
            import soundfile as sf
            info = sf.info(os.path.join(d, f))
            entry["seconds"] = round(info.frames / info.samplerate, 3)
        except Exception:
            pass
        out.append(entry)
    return out


@mcp.tool(annotations=RO)
def get_guide(topic: str = "overview") -> str:
    """Reference for wielding the Field. Topics: overview, workflow, listening (local audio model), score, sampler (synth and drum samplers, resampling takes), sequencers (the seven sequencers and what MIDI drives), engines, midi, device, manual."""
    return knowledge.guide(topic)


@mcp.tool(annotations=RO)
def search_manual(query: str) -> str:
    """Full-text search in Teenage Engineering's OP-1 field user guide (firmware 1.7). Returns matching passages with page numbers."""
    return knowledge.manual_search(query)


# Sampling on the device, from the firmware 1.7 manual (pages 12, 21, 40, 41 and 54). No entry may contain a percent sign:
# human_steps formats entries that hold "%d" with the MIDI channel.
SAMPLING_STEPS: dict[str, str] = {
    "sample_from_input": "Load a sampler sound first: in synth mode hold shift and press a key 1-8 and choose the sampler engine with the blue encoder, or in drum mode load a drum kit. Press the input key (the top right key with the mic symbol) to prepare sampling; hold shift and press input to open the input screen and turn the blue encoder to the source: built-in microphone or line in, FM radio, usb audio (the computer's audio output), or ear (the Field's own output, for resampling what it plays). Ochre sets stereo or mono for line in, gray the recording threshold, orange the input gain; press input again to confirm. Hold any key while the sound plays (in drum mode the manual does not say where the recording lands; report which key plays it), release it, then play the keyboard and trim the start and end points with the encoders. Tell the model which slot holds the new sample so it can audition it, and save it by holding the sound key for two seconds.",
    "lift_tape_to_sampler": "In tape mode select the track with T1-T4 and press the lift key (lift / erase): the take under the playhead is lifted into memory; with a loop active (loop in and loop out on keys 1 and 2, loop on with key 3) shift + lift (lift all) lifts every track inside the loop. Then switch to synth mode with a sampler sound loaded, or to drum mode with a drum kit, and press drop. The firmware 1.7 manual documents lift and drop for tape takes and for preset files only, so tell the model whether the waveform appeared in the sampler. The bridge's own route needs no lift: play or record_to_tape with the USB capture, then sample_from_take and author_sampler_preset.",
    "drop_sample_to_tape": "Documented route: press tape, choose the track with T1-T4, move the playhead with the arrow keys, hold shift and press record to arm, then play the key that holds the sample (or let the model play it with record_to_tape); the take lands on the tape and the model stops it. Lifted-take route: with a take in memory from the lift key, press drop in tape mode to place it at the playhead; shift + drop (merge drop) merges lifted tracks onto a single track. Whether the sound of a sampler slot can be lifted from the synth or drum screen is not in the firmware 1.7 manual: try the lift key there and report what happens.",
}


@mcp.tool(annotations=RO)
def human_steps(topic: str) -> str:
    """Exact key presses for things only the human can do on the Field: arm_recording, select_track, disk_mode, exit_disk_mode, usb_audio_mode, midi_settings, ctrl_mode, load_preset, save_preset, tape_style, mixdown, randomize; sampling: sample_from_input, lift_tape_to_sampler, drop_sample_to_tape; sequencers: select_sequencer, enable_sequencer, disable_sequencer, sequencer_parameters, arpeggio_setup, endless_setup, finger_setup, hold_setup, pattern_setup, sketch_setup, tombola_setup, clear_sequencer, sync_mode."""
    steps = {
        "arm_recording": "In tape mode, choose the track with T1-T4, move to where the take should start, then hold shift and press record to arm. The first incoming note starts recording (a count-in plays if enabled). The bridge stops the tape when the take ends.",
        "select_track": "Press tape, then T1, T2, T3 or T4. The live synth or drum is heard and recorded on that track.",
        "disk_mode": "Hold shift and press COM (the output key), keep holding shift and press T4. The Field mounts as a disk named OP-1 on the computer. The bridge sends no MIDI and captures no audio while the disk is mounted (whether the Field keeps them alive in disk mode is unverified).",
        "exit_disk_mode": "After the computer ejects the disk, hold shift and press COM to return to normal mode.",
        "usb_audio_mode": "Hold shift and press COM, press T1 for system settings, select system with the blue encoder, find usb audio and choose 10 channel (main stereo plus tape tracks 1-4).",
        "midi_settings": "Hold shift and press COM, press T1, select midi. Set the MIDI channel (the bridge uses channel %d), clock to in or both, notes to both, other to both. The monitor entry shows incoming MIDI.",
        "ctrl_mode": "Hold shift and press COM, press T2. The Field becomes a MIDI controller: keys send notes, encoders send CCs (relative or absolute). Press shift and COM again to leave.",
        "load_preset": "In synth or drum mode hold shift and press a key 1-8, pick the engine with the blue encoder and the preset with ochre, press the key again to confirm.",
        "save_preset": "Hold the sound key 1-8 for two seconds to save a snapshot of the current sound. The file lands in the snapshot folder, named by the Field's internal date (manual page 10); shift + 1-8 opens the browser to find it.",
        "tape_style": "Press shift and tape to open the tape browser; choose a tape and its style (studio, vintage, porta, disk mini).",
        "mixdown": "Press the output key, then T4 to start a mixdown recording of everything you play or play back; T2 stops it. Mixdowns are files reachable over MTP.",
        "randomize": "With a sound loaded, hold shift and press drop to randomize it.",
        "vocoder_setup": "In synth mode hold shift and press a sound key, choose vocoder with the blue encoder, pick a preset and confirm. Then hold shift and press the input key (mic symbol), turn the blue encoder to usb audio, and press the input key so the input is on; set the input gain to the middle. The Mac's speech then arrives as the vocoder's modulator.",
        "enable_endless": "Press drum (or synth), choose the sound, then hold shift and press the sequencer key, pick endless with the blue encoder and confirm. The sequencer key toggles it on and off. Set the note value with the blue encoder on the endless screen before entering steps.",
        "disable_sequencer": "Press the sequencer key once; the screen shows the sequencer off.",
        "input_to_tape": "Verified 2026-09-28. Hold shift and press the input (mic) key and choose usb audio with the blue encoder; press tape, pick the track with T1-T4 and scrub to where the recording should start; press the input key while in tape mode so the external audio is on (in tape mode the key routes external audio to the mix and tape; from the input screen or synth mode it only feeds the vocoder and sampler); set the gain with orange; then hold record and press play to start recording, and press stop at the end. Press the input key again afterwards to switch the external audio off. The bridge streams the audio with stream_to_tape(start='now') once the tape is rolling; play over MIDI does not start an armed take.",
        "toggle_effect": "With the sound loaded, press T3 to show its effect page; pressing T3 again toggles the effect on or off (the page dims when off). No MIDI message does this, and the bridge cannot tell whether an effect is on: if the FX encoders change nothing, ask for T3.",
        "revert_preset": "With the changed sound loaded, hold shift and press the synth key (the waveform icon); for a drum sound hold shift and press the drum key. The Field discards every change to the active sound and reloads its saved preset (no MIDI message does this). It only works for a sound that came from the preset library: a slot file the bridge installed is not in the library, so for those the human reloads the slot another way (pick it again from the library, or a disk-mode round with install_presets).",
    }
    steps.update(SAMPLING_STEPS)
    steps.update(SQ.HUMAN_STEPS)
    cfg = _cfg()
    if topic not in steps:
        return "unknown topic; known: " + ", ".join(sorted(steps))
    return steps[topic] % cfg.midi_channel if "%d" in steps[topic] else steps[topic]


@mcp.tool(annotations=RO)
def list_engines() -> dict[str, Any]:
    """Synth engines, FX and LFOs with what each of the four encoders does."""
    return knowledge.engines_catalog()


@mcp.tool(annotations=RO)
def list_sessions() -> list[str]:
    """Sessions on disk. A session holds takes, seeds, samples (cuts from sample_from_take) and backups."""
    return _list_sessions()


@mcp.tool(annotations=RW)
def use_session(name: str) -> dict[str, Any]:
    """Switch to (or create) a session by name."""
    cfg = _cfg(); cfg.current_session = name; cfg.save()
    return {"session": name, "dir": _sess_dir(cfg)}


@mcp.tool(annotations=RW)
def set_constraints(changes: dict[str, Any]) -> dict[str, Any]:
    """Tighten the constraints (never loosen them; the human loosens via the CLI). Keys: key ("D minor"), tempo, tempo_range [lo, hi], max_polyphony, max_take_seconds, allowed_slots ["synth 1", "drum 2"], allow_sound_select, allow_sound_design, allow_master, allow_mixer, allow_transport, allow_tempo_change, allowed_cc_params, notes."""
    cfg = _cfg()
    changed = cfg.constraints.tighten(changes)
    cfg.save()
    return {"changed": changed, "constraints": cfg.constraints.__dict__, "mode": cfg.mode}


@mcp.tool(annotations=RW)
def describe_slot(kind: str, slot: int, description: str) -> dict[str, Any]:
    """Remember what a sound slot holds (engine, character, what it is good for). Shown in get_status as the palette."""
    cfg = _cfg()
    cfg.palette[f"{kind} {slot}"] = {"description": description, "updated": time.strftime("%Y-%m-%d %H:%M")}
    cfg.save()
    return {"palette": cfg.palette}


# ------------------------------------------------------------------ sounds

@mcp.tool(annotations=RW)
def select_sound(kind: str, slot: int) -> dict[str, Any]:
    """Load a sound slot: kind synth or drum, slot 1-8 (program change). Live edits made over MIDI stay on a slot (a program change does not reload the saved preset); only the human's revert shortcut restores it, see human_steps("revert_preset")."""
    cfg = _cfg(); cfg.check("sound_select")
    if not cfg.slot_allowed(kind, slot):
        raise PermissionError(f"{kind} {slot} is not in allowed_slots {cfg.constraints.allowed_slots}")
    with _device(cfg) as f:
        if kind == "synth":
            f.select_synth_slot(slot)
        elif kind == "drum":
            f.select_drum_slot(slot)
        else:
            raise ValueError("kind must be synth or drum")
    return {"loaded": f"{kind} {slot}", "palette_note": cfg.palette.get(f"{kind} {slot}")}


@mcp.tool(annotations=RW)
def set_parameters(values: dict[str, int]) -> dict[str, Any]:
    """Turn encoders on the loaded sound or the master bus by name, value 0-127. Names: engine1-4, attack, decay, sustain, release, fx1-4, lfo1-4, master_fx1-4, master1-4 (left, right, drive, release), eq_low, eq_mid, eq_high, tape_rec_level, metronome, octave. The Field keeps these until the human reverts the preset (a program change does not reload it)."""
    cfg = _cfg()
    from .score import _cc_group
    for k in values:
        g = _cc_group(k)
        if g == "sound_design": cfg.check("sound_design")
        elif g == "master": cfg.check("master")
        elif g == "transport": cfg.check("transport")
        elif g == "mixer": raise ValueError("use set_mixer for track volume, pan and mute")
    with _device(cfg) as f:
        for k, v in values.items():
            f.cc(k, int(v))
    return {"set": values}


@mcp.tool(annotations=RW)
def randomize_sound() -> str:
    """Randomize the loaded patch (CC 62), like shift + drop on the device. Follow with audition to hear what you got. Not undoable over MIDI: randomize also changes the effect type, LFO and shift-layer values that no CC reaches, so only the human's revert shortcut (human_steps("revert_preset")) brings the saved preset back. Do not randomize a sound the human set up unless they asked for it."""
    cfg = _cfg(); cfg.check("sound_design")
    with _device(cfg) as f:
        f.cc("randomize", 127)
    return "randomized the active patch; the saved preset comes back only when the human reverts it on the device: " + human_steps("revert_preset")


@mcp.tool(annotations=RW)
def revert_sound() -> str:
    """Send TE's reset-active-patch CC (63). On firmware 1.7 the Field showed no response to it in five trials (edited knob, randomized patch, values 64 and 127, note held, other channel), so this returns the human step that does revert the preset; ask the human when the saved sound is needed."""
    cfg = _cfg(); cfg.check("sound_design")
    with _device(cfg) as f:
        f.cc("reset", 127)
    return "sent CC 63; the Field has shown no response to it, so if the sound is still changed ask the human to revert it: " + human_steps("revert_preset")


@mcp.tool(annotations=RW)
def audition(pitches: list[str] = ["C4"], seconds: float = 1.5, velocity: int = 100, kind: str | None = None, slot: int | None = None) -> dict[str, Any]:
    """Play a note or chord on the loaded sound (or on kind/slot) while recording, and return level, brightness, attack and release measurements plus a spectrogram id for view_spectrogram."""
    cfg = _cfg()
    if kind and slot:
        cfg.check("sound_select")
        if not cfg.slot_allowed(kind, slot):
            raise PermissionError(f"{kind} {slot} is not allowed")
    notes = [pitch_to_midi(p) for p in pitches]
    with _device(cfg) as f:
        if kind and slot:
            (f.select_synth_slot if kind == "synth" else f.select_drum_slot)(slot); time.sleep(0.4)
        rec = f.recorder().start()
        time.sleep(0.3)
        s0 = rec.mark("on")
        for n in notes: f.note_on(n, velocity)
        time.sleep(seconds)
        rec.mark("off")
        for n in notes: f.note_off(n)
        time.sleep(min(3.0, max(1.0, seconds)))
        r = rec.stop()
    pair = _live_pair(r.audio)
    if pair is None:
        return {"heard": False, "hint": "no audio came back; is a tape track selected and the Field in synth or drum mode?"}
    x = r.audio[:, pair[0] - 1: pair[1]].mean(axis=1); sr = r.samplerate
    on, off = r.mark_sample("on"), r.mark_sample("off")
    onset = an.onset_sample(x, sr, start=max(0, on - int(0.02 * sr)))
    body = x[on: off]
    tail = x[off + int(0.05 * sr):]
    t, cent = an.spectral_centroid_series(body, sr) if len(body) > 4096 else (None, None)
    tl_t, tl = an.envelope_db(tail, sr, 20.0)
    peak_tail = float(tl.max()) if len(tl) else -120.0
    rel_ms = None
    if len(tl):
        below = [i for i, v in enumerate(tl) if v < peak_tail - 40]
        rel_ms = round(float(tl_t[below[0]] * 1000.0), 0) if below else None
    # attack: time from onset to 90 percent of peak level
    e_t, e_l = an.envelope_db(x[on: on + int(1.0 * sr)], sr, 2.0)
    atk_ms = None
    if len(e_l):
        pk = float(e_l.max()); idx = [i for i, v in enumerate(e_l) if v >= pk - 1.0]
        atk_ms = round(float(e_t[idx[0]] * 1000.0), 1) if idx else None
    tid = new_id("aud")
    d = _sess_dir(cfg)
    an.spectrogram_png(x, sr, os.path.join(d, "takes", tid + ".png"), marks=r.marks)
    save_path = os.path.join(d, "takes", tid + ".wav")
    import soundfile as sf
    sf.write(save_path, r.audio[:, pair[0] - 1: pair[1]], sr, subtype="FLOAT")
    prom = an.note_prominence(body[int(0.1 * sr):], sr, notes) if len(body) > 4096 else {}
    return {
        "heard": True, "id": tid, "live_usb_channels": list(pair),
        "peak_db": round(an.peak_db(body), 1), "rms_db": round(an.rms_db(body), 1),
        "brightness_hz": {"median": round(float(cent.mean())) if cent is not None else None, "start": round(float(cent[:4].mean())) if cent is not None else None, "end": round(float(cent[-4:].mean())) if cent is not None else None},
        "attack_to_peak_ms": atk_ms, "release_to_minus40db_ms": rel_ms, "latency_ms": None if onset is None else round((onset - on) / sr * 1000.0, 1),
        "notes_heard": {midi_to_name(n): round(p, 1) for n, p in prom.items()},
        "wav": save_path,
    }


def _live_pair(audio):
    from .player import detect_live_pair
    return detect_live_pair(audio)


@mcp.tool(annotations=RW)
def audition_slots(slots: list[str], pitches: list[str] = ["C4"], seconds: float = 1.2, velocity: int = 100, wait: bool | None = None) -> dict[str, Any]:
    """Audition several slots in one call, e.g. ["synth 1", "synth 3", "drum 2"]: each is loaded and played with the pitches while recording, and one compact line per slot comes back (peak, brightness, attack, release, notes heard, take id, plus the index's tags when the slot was audited before). Use it instead of one audition call per slot when choosing sounds; it runs as a job when the list is long."""
    cfg = _cfg(); cfg.check("sound_select")
    parsed: list[tuple[str, int]] = []
    for item in slots:
        m = re.match(r"^\s*(synth|drum)\s*[- ]?\s*([1-8])\s*$", str(item).lower())
        if not m:
            raise ValueError(f"slot {item!r} is not of the form 'synth 3' or 'drum 2'")
        kind, slot = m.group(1), int(m.group(2))
        if not cfg.slot_allowed(kind, slot):
            raise PermissionError(f"{kind} {slot} is not in allowed_slots")
        parsed.append((kind, slot))
    if not parsed:
        raise ValueError("no slots given")
    index = SO.load_index(HOME) if hasattr(SO, "load_index") else {}
    per_slot = seconds + min(3.0, max(1.0, seconds)) + 1.4

    def work(job):
        rows = []
        for i, (kind, slot) in enumerate(parsed):
            _stage(job, f"auditioning {kind} {slot} ({i + 1}/{len(parsed)})")
            r = audition(pitches, seconds, velocity, kind, slot)
            row: dict[str, Any] = {"slot": f"{kind} {slot}", "heard": r.get("heard", False)}
            if r.get("heard"):
                row.update({"id": r.get("id"), "peak_db": r.get("peak_db"), "brightness_hz": (r.get("brightness_hz") or {}).get("median"),
                            "attack_ms": r.get("attack_to_peak_ms"), "release_ms": r.get("release_to_minus40db_ms"), "notes_heard": r.get("notes_heard")})
            else:
                row["hint"] = r.get("hint")
            entry = index.get(f"{kind} {slot}") or index.get(f"{kind}-{slot}") or {}
            for k in ("tags", "descriptors", "engine", "name", "role"):
                if entry.get(k):
                    row[k] = entry[k]
            note = cfg.palette.get(f"{kind} {slot}")
            if note:
                row["palette_note"] = note.get("description") if isinstance(note, dict) else note
            rows.append(row)
        return {"auditioned": rows, "pitches": pitches, "note": "brightness is the median spectral centroid in Hz; view_spectrogram(id) or measure_take(id) for more on any of them"}
    return _run_or_job("audition_slots", "auditions", per_slot * len(parsed), wait, work)


# ------------------------------------------------------------------ presets (engine, fx and lfo choice)

@mcp.tool(annotations=RW)
def author_synth_preset(slot: int, engine: str, name: str, fx: str | None = None, lfo: str | None = None, fx_active: bool = True, lfo_active: bool = True,
                        octave: int | None = None, knobs: list[float] | None = None, adsr: list[float] | None = None) -> dict[str, Any]:
    """Design a synth preset for a slot: engine type (see list_engines and get_status for the identifiers this Field has written), FX type, LFO type, octave, envelope (fractions 0..1: attack, decay, sustain, release) and engine knobs (fractions, applied only within ranges observed in Field-written files; otherwise shape them live with set_parameters after loading). Every value in the file comes from presets the Field itself wrote, because the Field rejects presets with out-of-range values and turns them into samples. Engine "sampler" reuses a factory sample's audio and region (its knobs are positions on that audio); for your own audio use author_sampler_preset. Names are written as the Field writes them (lowercase, digits, space, '-', '#', 11 characters; sources.name_truncated says when they were changed). The file is staged; install_presets puts it on the Field in disk mode."""
    cfg = _cfg(); cfg.check("sound_design")
    if not cfg.slot_allowed("synth", slot):
        raise PermissionError(f"synth {slot} is not in allowed_slots")
    known = P.known_engine_ids(CATALOG)["synth"]
    if engine not in known:
        raise ValueError(f"engine {engine!r} has not been seen in a file from this Field; known: {known}. Ask the human to load a factory preset of it into a slot, enter disk mode, and call learn_engines.")
    pre, notes = P.compose_synth_preset(HOME, CATALOG, engine, name, fx, lfo, fx_active, lfo_active, octave, knobs, adsr)
    path = P.stage_preset(HOME, "synth", slot, pre)
    return {"staged": path, "slot": slot, "preset": pre.summary(), "sources": notes, "next": "call install_presets when the human has put the Field in disk mode (human_steps disk_mode); then select_sound and shape it with set_parameters"}





_PRESET_NEXT = "call install_presets when the human has put the Field in disk mode (human_steps disk_mode); then select_sound and shape it with set_parameters"


@mcp.tool(annotations=RW)
def author_sampler_preset(slot: int, wav_path: str, name: str, root_note: str | int = "C4", start_s: float | None = None, loop_in_s: float | None = None,
                          loop_out_s: float | None = None, end_s: float | None = None, loop: bool = True, direction: str = "forward",
                          gain: float | None = None, fine_tune: float = 0.0, octave: int = 0, fx: str | None = None, lfo: str | None = None,
                          loop_fade: float | None = None, fx_active: bool | None = None, lfo_active: bool | None = None,
                          adsr: list[float] | None = None, truncate: bool = False) -> dict[str, Any]:
    """Turn a WAV on this computer (a sample_from_take cut, a backup stem, any audio file) into a synth sampler preset for a synth slot: the Field's chromatic stereo sampler, 6 s at most (longer files are refused; cut them with sample_from_take or pass truncate=true). root_note (name like C4 or MIDI number) is the key that plays the sample at its recorded speed. Region in seconds on the sample: start_s, loop_in_s, loop_out_s, end_s default to the whole sample looping; loop=false puts the loop points on the end (unverified encoding). direction forward or reverse (reverse unverified), gain as a linear multiplier (1.0 = unity), fine_tune -1..1 of the encoder (0 = centre, scale unknown), loop_fade 0..1, octave -4..4, fx and lfo types from list_engines (their parameters come from Field-written examples; fx_active and lfo_active default to on when a type is given, otherwise the Field's import default, off), adsr as 0..1 fractions within observed ranges. Every other value is what the Field's own sampler writer puts, because the Field demotes presets with out-of-range values. Names are cut to 11 characters. Returns the staged path, the decoded region (knob = seconds / 6 * 32767) and, under derived.unverified, anything the file relies on that has not been heard on the device. install_presets copies it onto the Field in disk mode."""
    cfg = _cfg(); cfg.check("sound_design")
    if not cfg.slot_allowed("synth", slot):
        raise PermissionError(f"synth {slot} is not in allowed_slots")
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"no audio file at {wav_path}")
    pre, notes = SM.make_sampler_preset(wav_path, name, root_note, start_s, loop_in_s, loop_out_s, end_s, loop=loop, direction=direction,
                                        gain=gain, fine_tune=fine_tune, loop_fade=loop_fade, octave=octave, fx=fx, lfo=lfo,
                                        fx_active=fx_active, lfo_active=lfo_active, adsr=adsr, truncate=truncate, catalog_path=CATALOG, home=HOME)
    path = P.stage_preset(HOME, "synth", slot, pre)
    return {"staged": path, "slot": slot, "preset": pre.summary(), "region": SM.sampler_region(pre), "derived": notes, "next": _PRESET_NEXT}


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "-", name).strip("-") or "sample"


def _session_wav(d: str, take_id: str) -> str:
    """The WAV behind a take, audition or seed id in session dir d; a path to a WAV also works."""
    if take_id.lower().endswith(".wav") and os.path.isfile(take_id):
        return take_id
    tid = take_id[:-4] if take_id.lower().endswith(".wav") else take_id
    for sub in ("takes", "seeds", "samples", "backups"):
        p = os.path.join(d, sub, tid + ".wav")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"no WAV for {take_id!r} in this session's takes, seeds, samples or backups (list_takes, list_seeds)")


@mcp.tool(annotations=RW)
def sample_from_take(take_id: str, start_s: float, end_s: float | None = None, name: str | None = None, fade_ms: float = 5.0, normalize_db: float | None = None) -> dict[str, Any]:
    """Cut a region out of a recorded take, audition or seed of the current session (its id, e.g. take-20260926-121031 or aud-20260926-121031; a WAV path also works) into sessions/<session>/samples/<name>.wav (get_status lists them): [start_s, end_s) in seconds of the take (end_s omitted = to the end), faded over fade_ms at both ends so it starts and ends at zero and loops without a click, optionally peak-normalized to normalize_db dBFS. Pass normalize_db=-1.0 unless you want the take's own level: the Field plays a sample at its recorded level, and a cut left at -22 dBFS came out about 16 dB quiet on the device. The cut keeps the take's sample rate and channels. Feed the path to author_sampler_preset (a synth sampler slot, 6 s at most; fits_sampler says whether it fits). This is the resampling loop: play a sound, record it, cut it, author it, install it."""
    cfg = _cfg(); d = _sess_dir(cfg)
    src = _session_wav(d, take_id)
    out_dir = os.path.join(d, "samples")
    os.makedirs(out_dir, exist_ok=True)
    base = _safe_name(name or f"{os.path.splitext(os.path.basename(src))[0]}-{start_s:.2f}s")
    out = os.path.join(out_dir, base + ".wav")
    n = 1
    while os.path.exists(out):
        n += 1
        out = os.path.join(out_dir, f"{base}-{n}.wav")
    res = SM.trim_take(src, out, start_s, end_s, fade_ms=fade_ms, normalize_db=normalize_db)
    nxt = ("author_sampler_preset(slot, wav_path=sample, name, root_note) for a synth sampler slot"
           if res["fits_sampler"] else
           f"{res['seconds']:.2f} s is longer than the synth sampler's 6 s: cut a shorter region for author_sampler_preset")
    if res["peak_dbfs"] < SM.SAMPLER_QUIET_DBFS:
        nxt = (f"the cut peaks at {res['peak_dbfs']} dBFS and the Field will play it that quiet: cut again with normalize_db=-1.0, "
               f"or pass gain above 1.0 when authoring. Then " + nxt)
    res.update({"source": src, "sample": out, "next": nxt})
    return res


@mcp.tool(annotations=RO)
def list_staged_presets() -> list[dict[str, Any]]:
    """Presets authored but not yet installed on the Field."""
    return P.staged_presets(HOME)


@mcp.tool(annotations=RW)
def discard_staged(kind: str = "all", slot: int | None = None) -> dict[str, Any]:
    """Throw away staged presets (kind synth/drum with a slot, or all)."""
    d = P.staging_dir(HOME); removed = []
    for k in ("synth", "drum"):
        if kind != "all" and kind != k:
            continue
        for f in os.listdir(os.path.join(d, k)):
            if f.endswith(".aif") and (slot is None or f == f"{slot}.aif"):
                os.remove(os.path.join(d, k, f)); removed.append(f"{k} {f}")
    return {"removed": removed}


@mcp.tool(annotations=RW)
def install_presets(wait_seconds: float = 180.0) -> dict[str, Any]:
    """Copy staged presets into the Field's slots. Every staged file is checked against the Field's own files first (drum kits and sampler presets have validators) and nothing is copied when one fails. Needs the Field in disk mode (human_steps disk_mode); waits up to wait_seconds for the disk, backs up the replaced files, installs, ejects, and waits for the Field to return."""
    cfg = _cfg(); cfg.check("sound_design")
    staged = P.staged_presets(HOME)
    if not staged:
        return {"installed": [], "message": "nothing staged"}
    bad = P.check_staged(HOME)
    if bad:
        return {"installed": [], "failed": bad,
                "message": "nothing was copied: these staged files are unlike anything the Field writes and would be demoted to samples; author them again or discard_staged them, then call install_presets"}
    t0 = time.time(); vol = P.find_field_volume()
    while vol is None and time.time() - t0 < wait_seconds:
        time.sleep(2); vol = P.find_field_volume()
    if vol is None:
        return {"installed": [], "message": f"the OP-1 disk did not appear within {wait_seconds:.0f} s. Ask the human: " + human_steps("disk_mode")}
    stamp = time.strftime("%Y-%m-%d-%H%M%S")
    backup = os.path.join(HOME, "field-backup", stamp + "-replaced")
    result = P.install_staged(HOME, vol, backup_dir=backup, record_dir=os.path.join(HOME, "field-backup", stamp + "-installed"))
    if hasattr(os, "sync"):
        os.sync()
    ejected = P.eject_volume(vol) if not result["failed"] else False
    if result["failed"]:
        result["message"] = "some slots failed and the disk was left mounted; the staged files are kept. Do not eject from the Field while install_presets runs."
    back = False
    t1 = time.time()
    while time.time() - t1 < 60:
        if find_midi_output() and find_audio_input():
            back = True; break
        time.sleep(2)
    result.update({"ejected": ejected, "field_back_in_normal_mode": back, "backup_of_replaced": backup if result["backed_up"] else None,
                   "next": "select the slot with select_sound and audition it" if back else "ask the human to leave disk mode: " + human_steps("exit_disk_mode")})
    return result


def _slot_file(kind: str, slot: int) -> str | None:
    vol = P.find_field_volume()
    if vol:
        p = P.slot_path(vol, kind, slot)
        return p if os.path.exists(p) else None
    root = os.path.join(HOME, "field-backup")
    if not os.path.isdir(root):
        return None
    for d in sorted(os.listdir(root), reverse=True):
        p = os.path.join(root, d, kind, "user", f"{slot}.aif")
        if os.path.exists(p):
            return p
    return None


CATALOG = os.path.join(HOME, "engine-catalog.json")


@mcp.tool(annotations=RW)
def restore_slot(kind: str, slot: int, backup: str | None = None) -> dict[str, Any]:
    """Stage the copy of a slot from a disk backup (latest by default, or a backup folder name) so install_presets puts the original sound back."""
    cfg = _cfg(); cfg.check("sound_design")
    root = os.path.join(HOME, "field-backup")
    src_dir = os.path.join(root, backup) if backup else P.latest_backup_dir(HOME)
    # prefer the full-disk backup that holds this slot
    candidates = [src_dir] + [os.path.join(root, d) for d in sorted(os.listdir(root), reverse=True)] if os.path.isdir(root) else []
    for d in candidates:
        if not d:
            continue
        for rel in (os.path.join(kind, "user", f"{slot}.aif"), os.path.join(kind, f"{slot}.aif")):
            src = os.path.join(d, rel)
            if os.path.exists(src):
                import shutil
                dst = os.path.join(P.staging_dir(HOME), kind, f"{slot}.aif")
                shutil.copy2(src, dst)
                return {"staged": dst, "from": src, "preset": P.read_preset(src).summary(), "next": "install_presets in disk mode"}
    raise FileNotFoundError(f"no backup of {kind} {slot}")


@mcp.tool(annotations=RW)
def learn_engines() -> dict[str, Any]:
    """Read every preset on the mounted Field disk (or the latest backup) and record the Field's own engine, FX and LFO identifier strings with example values. Run after the human loads unfamiliar engines into slots and enters disk mode."""
    src = P.find_field_volume() or P.latest_backup_dir(HOME)
    if not src:
        raise FileNotFoundError("no mounted Field disk and no backup to learn from")
    if P.find_field_volume():
        dst = os.path.join(HOME, "field-backup", time.strftime("%Y-%m-%d-%H%M%S") + "-disk")
        P.copy_volume(src, dst)
        src = dst
    return P.learn_engines(src, CATALOG)


@mcp.tool(annotations=RO)
def read_slots() -> dict[str, Any]:
    """What the 16 slots contain (engine, name, FX, LFO, knob values), read from the mounted disk or the latest disk backup."""
    out: dict[str, Any] = {"source": "mounted disk" if P.find_field_volume() else "latest backup", "synth": {}, "drum": {}}
    for kind in ("synth", "drum"):
        for slot in range(1, 9):
            p = _slot_file(kind, slot)
            out[kind][slot] = P.read_preset(p).summary() if p else None
    return out


@mcp.tool(annotations=RW)
def backup_disk() -> dict[str, Any]:
    """With the Field in disk mode, copy all of its preset files to a dated backup folder."""
    vol = P.find_field_volume()
    if not vol:
        return {"backed_up": False, "message": "the OP-1 disk is not mounted; " + human_steps("disk_mode")}
    dst = os.path.join(HOME, "field-backup", time.strftime("%Y-%m-%d-%H%M%S") + "-disk")
    n = P.copy_volume(vol, dst)
    return {"backed_up": True, "path": dst, "files": n}


# ------------------------------------------------------------------ playing

@mcp.tool(annotations=RO)
def validate_score(score: dict[str, Any]) -> dict[str, Any]:
    """Check a score against the Field's limits and the session constraints without playing it."""
    cfg = _cfg(); s = _score(score)
    problems = validate(s, cfg)
    return {"ok": not problems, "problems": problems, "seconds": round(s.seconds(), 2), "max_polyphony": max_polyphony(s), "events": len(compile_events(s))}


@mcp.tool(annotations=RW)
def play(score: dict[str, Any], record: bool = True, name: str | None = None, wait: bool | None = None) -> dict[str, Any]:
    """Play a score on the loaded sound. With record=true the Field's USB audio is captured and analysed (notes heard, timing, level, spectral movement) and saved as a take. A score longer than about 25 s runs as a background job (the reply carries a job_id to poll with job_status; wait=false forces a job, wait=true forces waiting)."""
    cfg = _cfg(); s = _score(score)
    problems = validate(s, cfg)
    if problems:
        return {"played": False, "problems": problems}
    return _run_or_job("play", name or "take", _expected(s), wait, lambda job: _play_and_keep(cfg, s, record, name or "take", job=job))


def _play_and_keep(cfg: Config, s: Score, record: bool, take_name: str, kind: str = "take", job: Job | None = None) -> dict[str, Any]:
    """Play a validated score on the device and, when recording, save and analyse the take the way play() does."""
    with _device(cfg, timeout=30.0 if job else 3.0) as f:
        _stage(job, "playing")
        res = play_score(f, s, record=record, latency_ms=cfg.latency_ms, cancel=_cancel(job))
    _stage(job, "analysing")
    out: dict[str, Any] = {"played": True, "seconds": round(res.seconds, 2), "events": res.events}
    if record and res.recording is not None:
        tid = new_id(take_name)
        d = _sess_dir(cfg)
        wav = save_take(res, os.path.join(d, "takes", tid + ".wav"))
        if res.live_pair:
            x = res.recording.audio[:, res.live_pair[0] - 1: res.live_pair[1]].mean(axis=1)
            an.spectrogram_png(x, res.recording.samplerate, os.path.join(d, "takes", tid + ".png"), marks=res.recording.marks)
        write_json(os.path.join(d, "takes", tid + ".json"), {"id": tid, "score": s.model_dump(), "analysis": res.analysis, "wav": wav, "kind": kind})
        out.update({"take_id": tid, "wav": wav, "analysis": res.analysis, "spectrogram": "view_spectrogram(\"%s\")" % tid})
    return out


@mcp.tool(annotations=RW)
def hold_chord(pitches: list[str | int], beats: float = 4.0, tempo: float = 100.0, velocity: int = 100, record: bool = True, tail_seconds: float = 1.5, name: str | None = None, send_clock: bool = True, wait: bool | None = None) -> dict[str, Any]:
    """Hold one chord for N beats with MIDI clock at the tempo (CC 80 sets the Field's BPM first, so the tape runs at 100 percent in midi sync mode; in guided mode that needs allow_transport, and allow_tempo_change unless tempo is the session's fixed tempo, or send_clock=false): the material for the arpeggio, hold, finger, sketch and tombola sequencers, which run while the notes are held (endless is verified over MIDI; the others follow by analogy). The human must have selected and enabled the sequencer first (human_steps arpeggio_setup, hold_setup, tombola_setup ...); with none enabled this is a plain sustained chord. Keep it to six pitches: each is a voice, or a tombola ball. With record=true the take is captured, saved and analysed like play, but while a sequencer runs the audible notes are not the sent ones, so ignore notes_heard and judge the take by level, clipping, spectral movement and view_spectrogram. Returns the score it played, ready for record_to_tape once the sound is right."""
    cfg = _cfg()
    if not pitches:
        raise ValueError("give at least one pitch")
    s = SQ.held_chords_score([list(pitches)], beats, tempo, velocity=velocity, tail_seconds=tail_seconds, send_clock=send_clock)
    problems = validate(s, cfg)
    if problems:
        return {"played": False, "problems": problems}
    def work(job):
        out = _play_and_keep(cfg, s, record, name or "hold", kind="hold", job=job)
        out["score"] = s.model_dump()
        out["clock"] = SQ.clock_plan(tempo, beats) if send_clock else {"sent": False, "note": "send_clock was false: no CC 80 and no MIDI clock went to the Field"}
        out["note"] = "with a sequencer enabled, notes_heard compares the audible notes with the held chord and means nothing; judge the take by level, spectral movement and the spectrogram"
        return out
    return _run_or_job("hold", name or "hold", _expected(s), wait, work)


@mcp.tool(annotations=RW)
def record_to_tape(score: dict[str, Any], name: str | None = None, wait: bool | None = None) -> dict[str, Any]:
    """Commit a part to the Field's tape: the human must first select the track and arm recording (human_steps arm_recording). Playing the score starts the recording; the bridge stops the tape at the end and keeps a USB capture of the take. Parts longer than about 25 s run as a background job (job_status has the result; the tape is stopped if the job is cancelled)."""
    cfg = _cfg(); s = _score(score)
    problems = validate(s, cfg)
    if problems:
        return {"recorded": False, "problems": problems}

    def work(job):
        with _device(cfg, timeout=30.0 if job else 3.0) as f:
            _stage(job, "recording to tape")
            try:
                res = play_score(f, s, record=True, latency_ms=cfg.latency_ms, cancel=_cancel(job))
            finally:
                f.cc("tape_stop", 127)
        _stage(job, "saving")
        tid = new_id(name or "tape-take")
        d = _sess_dir(cfg)
        wav = save_take(res, os.path.join(d, "takes", tid + ".wav")) if res.recording is not None else None
        write_json(os.path.join(d, "takes", tid + ".json"), {"id": tid, "score": s.model_dump(), "analysis": res.analysis, "wav": wav, "kind": "tape"})
        return {"recorded": True, "take_id": tid, "analysis": res.analysis, "note": "tape stopped; the take is on the armed track if recording was armed. Rewind with tape('start') and tape('play') to hear it."}
    return _run_or_job("record_to_tape", name or "tape-take", _expected(s), wait, work)


@mcp.tool(annotations=RO)
def list_takes() -> list[dict[str, Any]]:
    """Takes and auditions in the current session."""
    cfg = _cfg(); d = os.path.join(_sess_dir(cfg), "takes")
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            j = read_json(os.path.join(d, f)); a = j.get("analysis", {})
            out.append({"id": j["id"], "kind": j.get("kind"), "seconds": j.get("score", {}).get("length_beats"), "notes_heard": a.get("notes_heard"), "peak_db": a.get("peak_db")})
    return out


@mcp.tool(annotations=RO)
def get_take(take_id: str) -> dict[str, Any]:
    """Full record of a take: score and analysis."""
    cfg = _cfg(); p = os.path.join(_sess_dir(cfg), "takes", take_id + ".json")
    return read_json(p)


def _take_wav(cfg: Config, id: str) -> str:
    """The WAV of a take, audition or seed in the current session."""
    d = _sess_dir(cfg)
    for sub in ("takes", "seeds"):
        j = os.path.join(d, sub, id + ".json")
        if os.path.exists(j):
            w = read_json(j).get("wav")
            if w and os.path.exists(w):
                return w
        w = os.path.join(d, sub, id + ".wav")
        if os.path.exists(w):
            return w
    raise FileNotFoundError(f"no audio for {id!r} in session {cfg.current_session}; list_takes and list_seeds show the ids")


def _take_json(cfg: Config, id: str) -> dict[str, Any]:
    d = _sess_dir(cfg)
    for sub in ("takes", "seeds"):
        j = os.path.join(d, sub, id + ".json")
        if os.path.exists(j):
            return read_json(j)
    return {}


@mcp.tool(annotations=RO)
def listener_status(backend: str = "local") -> dict[str, Any]:
    """Check local model installation or Gemini key configuration. No network or Field access.
    backend='gemini' reports configured models; configuration does not prove authentication.
    """
    return LI.status(HOME, backend)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True))
def listen_to_take(id: str, start_s: float = 0.0, seconds: float = 15.0,
                   question: str = LI.DEFAULT_QUESTION, channels: list[int] | None = None,
                   max_tokens: int | None = None, timeout_seconds: float = 300.0,
                   wait: bool | None = False, backend: str = "local",
                   model: str | None = None) -> dict[str, Any]:
    """Describe a saved take, seed, audition or absolute WAV path with an audio model.
    Default backend='local' uses offline Qwen2-Audio. Explicit backend='gemini' uploads the
    excerpt and question to Google, using the saved GEMINI_API_KEY; API charges may apply.
    Gemini model choices are in listener_status(backend='gemini'). No automatic retries.
    max_tokens defaults to 256 locally or 2048 for Gemini (including thinking; maximum 4096).
    Select 1–30 seconds; channels are 1-based (default: first stereo pair, the Field main mix).
    Returns a background job by default; poll job_status and use cancel_job to stop inference.
    Cancelling closes the local connection; Google may still bill a request already received.
    Saves an excerpt and review with source timestamps, measured levels and fallible observations.
    Gemini reviews include usage, estimated cost and usable=false for blocked/truncated answers.
    Does not play, record, alter the source audio or access the Field. It cannot verify exact chords,
    notes, BPM or aesthetic quality. Compare stems/variants and combine observations with measurements.
    """
    from pathlib import Path
    from uuid import uuid4
    LI.validate_options(start_s, seconds, question, max_tokens, timeout_seconds, backend, model)
    cfg = _cfg()
    path = Path(id).expanduser()
    source = str(path.resolve(strict=True)) if path.is_absolute() else _take_wav(cfg, id)
    report_dir = Path(_sess_dir(cfg)) / "reviews"
    def work(job):
        return LI.listen(HOME, source, report_dir, start_s=start_s, seconds=seconds,
                         question=question, channels=channels, max_tokens=max_tokens,
                         timeout_seconds=timeout_seconds, cancel=_cancel(job),
                         progress=lambda text: _stage(job, text), backend=backend, model=model)
    return _run_or_job("listen_to_take", "listen-" + uuid4().hex[:6], 60.0, wait, work)


@mcp.tool(annotations=RO)
def view_spectrogram(id: str, log_frequency: bool = False, max_hz: float = 8000.0) -> Image:
    """The spectrogram image of a take, audition or seed (time left to right, frequency bottom to top, red lines are score marks). log_frequency=true redraws it with a logarithmic axis from 30 Hz, which shows bass and low-mid detail the linear view hides."""
    cfg = _cfg(); d = _sess_dir(cfg)
    if log_frequency or max_hz != 8000.0:
        import soundfile as sf
        wav = _take_wav(cfg, id)
        x, sr = sf.read(wav, dtype="float32", always_2d=True)
        out = os.path.join(os.path.dirname(wav), f"{id}-{'log' if log_frequency else 'lin'}{int(max_hz)}.png")
        an.spectrogram_png(x.mean(axis=1), sr, out, max_hz=max_hz, log_frequency=log_frequency)
        return Image(path=out)
    for sub in ("takes", "seeds"):
        p = os.path.join(d, sub, id + ".png")
        if os.path.exists(p):
            return Image(path=p)
    raise FileNotFoundError(f"no spectrogram for {id}")


@mcp.tool(annotations=RO)
def measure_take(id: str, f0_hz: float | None = None, start_s: float = 0.0, end_s: float | None = None) -> dict[str, Any]:
    """Deeper numbers for a take, audition or seed (optionally a time window): level and peak, six band levels (sub, bass, low_mid, mid, presence, air, dBFS), a pitch track summary (median f0, note, cents, stability, voiced fraction), the levels of the first twelve harmonics against f0 (given, or the pitch track's), periodic movement (rate in Hz, prominence, swing in dB: a phaser, LFO or tremolo shows here), onset count, and a log-frequency spectrogram to view with view_spectrogram(id, log_frequency=true)."""
    import soundfile as sf
    cfg = _cfg()
    wav = _take_wav(cfg, id)
    audio, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = audio.mean(axis=1).astype(np.float64)
    a, b = int(max(0.0, start_s) * sr), int(end_s * sr) if end_s else len(x)
    x = x[a:b]
    if len(x) < sr // 4:
        raise ValueError("less than a quarter second of audio in the window")
    out: dict[str, Any] = {"id": id, "wav": wav, "seconds": round(len(x) / sr, 2), "window": [round(a / sr, 2), round(b / sr, 2)],
                           "rms_db": an.rms_db(x), "peak_db": an.peak_db(x), "bands_dbfs": an.band_levels(x, sr)}
    ts, f0, conf = an.pitch_track(x, sr)
    ps = an.pitch_summary(f0, conf)
    ps["vibrato"] = an.pitch_modulation(ts, f0)
    out["pitch"] = ps
    step = max(1, len(f0) // 24)
    out["pitch_track_hz"] = [round(float(v), 1) for v in f0[::step]]
    ref = f0_hz or ps.get("f0_hz")
    if ref:
        H, hop = an.harmonic_levels(x, sr, float(ref), n_harm=12)
        mean = H.mean(axis=1)
        out["harmonics_db_rel_h1"] = [round(float(v - mean[0]), 1) for v in mean]
        out["modulation"] = an.modulation_rate(x, sr, float(ref)) if len(x) > 2 * sr else {"rate_hz": None, "prominence": 0.0, "swing_db": 0.0, "note": "under two seconds"}
    ot, ostr = an.onset_strength(x, sr)
    if len(ostr):
        thr = float(np.percentile(ostr, 90)) * 0.6 + float(np.median(ostr)) * 0.4
        peaks = [i for i in range(1, len(ostr) - 1) if ostr[i] > thr and ostr[i] >= ostr[i - 1] and ostr[i] >= ostr[i + 1]]
        out["onsets"] = {"count": len(peaks), "first_s": [round(float(ot[i]), 2) for i in peaks[:12]]}
    png = os.path.join(os.path.dirname(wav), f"{id}-log8000.png")
    an.spectrogram_png(x, sr, png, log_frequency=True)
    out["log_spectrogram"] = f'view_spectrogram("{id}", log_frequency=true)'
    return out


@mcp.tool(annotations=RO)
def speech_intelligibility(id: str, speech_wav: str | None = None) -> dict[str, Any]:
    """STOI score (0-1) of a vocoder or vocal take against the speech that was sent (the take's own speech file by default, or any WAV): the measure that ranks vocoder settings reliably. About 0.75 is clearly intelligible on the Field's vocoder, below 0.5 is mush."""
    import soundfile as sf
    cfg = _cfg()
    j = _take_json(cfg, id)
    ref = speech_wav or j.get("speech_wav")
    if not ref or not os.path.exists(ref):
        raise ValueError("no speech file for this take; pass speech_wav")
    rec, rsr = sf.read(_take_wav(cfg, id), dtype="float32", always_2d=True)
    spk, ssr = sf.read(ref, dtype="float32", always_2d=True)
    out = an.speech_intelligibility(spk, ssr, rec, rsr)
    out.update({"id": id, "speech_wav": ref})
    return out


# ------------------------------------------------------------------ tape, mixer, master

@mcp.tool(annotations=RW)
def tape(action: str) -> str:
    """Tape transport: play, stop, start (rewind to the beginning), end, prev_bar, next_bar, loop_in, loop_out, loop_toggle, midi_start, midi_stop, midi_continue."""
    cfg = _cfg(); cfg.check("transport")
    with _device(cfg) as f:
        return T.transport(f, action)


@mcp.tool(annotations=RW)
def set_tempo(bpm: float) -> dict[str, Any]:
    """Set the Field's tempo (CC 80, 40-180 BPM in the device's steps). Scores also carry their own tempo for the MIDI clock they send."""
    cfg = _cfg(); cfg.check("tempo_change"); cfg.check("transport")
    if cfg.guided() and cfg.constraints.tempo is not None and abs(bpm - cfg.constraints.tempo) > 0.01:
        raise PermissionError(f"tempo is fixed at {cfg.constraints.tempo}")
    v = T.tempo_cc_value(bpm)
    with _device(cfg) as f:
        f.cc("tempo", v)
    return {"bpm_requested": bpm, "cc80": v}


@mcp.tool(annotations=RW)
def set_mixer(track: int, volume: int | None = None, pan: int | None = None, mute: bool | None = None) -> dict[str, Any]:
    """Tape track 1-4 mixer: volume 0-127, pan 0-127 (64 centre), mute."""
    cfg = _cfg(); cfg.check("mixer")
    with _device(cfg) as f:
        return T.set_mixer(f, track, volume, pan, mute)


@mcp.tool(annotations=RW)
def set_master(values: dict[str, int]) -> dict[str, Any]:
    """Master bus, values 0-127: fx1-fx4 (master FX encoders), left, right, drive, release, eq_low, eq_mid, eq_high."""
    cfg = _cfg(); cfg.check("master")
    with _device(cfg) as f:
        return T.set_master(f, values)


@mcp.tool(annotations=RW)
def backup_tape(name: str = "tape", seconds: float | None = None, wait: bool | None = None) -> dict[str, Any]:
    """Play the tape from the start and record every USB channel to the session's backups folder, then split into stems per track (and the main mix in 10-channel mode). Stops after `seconds`, or when the tape goes silent."""
    cfg = _cfg()
    d = os.path.join(_sess_dir(cfg), "backups")
    base = new_id(name)

    def work(job):
        with _device(cfg, timeout=30.0 if job else 3.0) as f:
            _stage(job, "playing the tape into the Mac")
            res = T.capture_tape(f, os.path.join(d, base + "-all.wav"), seconds=seconds, cancel=_cancel(job))
        _stage(job, "splitting stems")
        res["stems"] = T.split_stems(res["path"], d, base)
        return res
    return _run_or_job("backup_tape", name, (seconds + 3.0) if seconds else 400.0, wait, work)


@mcp.tool(annotations=RW)
def panic() -> str:
    """All notes off on every channel."""
    cfg = _cfg()
    with _device(cfg) as f:
        f.all_notes_off()
    return "all notes off"


# ------------------------------------------------------------------ sequencers

@mcp.tool(annotations=RO)
def list_sequencers() -> list[dict[str, Any]]:
    """The Field's seven sequencers (arpeggio, endless, finger, hold, pattern, sketch, tombola): what each does, what drives it, its encoders and shift page, and whether MIDI notes from the bridge drive it (with the evidence). Selecting, enabling and setting one up is the human's work: human_steps <name>_setup. get_guide("sequencers") has the recipes."""
    return SQ.sequencer_catalog()


@mcp.tool(annotations=RW)
def song_position(beat: float = 0.0) -> dict[str, Any]:
    """Send the MIDI song position pointer (TE's "set sequencer position") for a beat: 0-based like a score's beats (bar 2 beat 1 in 4/4 is beat 4), on the 16th-note grid, at most 4095 beats. The pointer counts 16th notes, so beat 4 is position 16. What it moves on the Field is untested: TE's table says the sequencer, while the Field's SPP ORIGIN setting was seen as tape; send it before the first note (or before tape midi_continue) and listen."""
    cfg = _cfg(); cfg.check("transport")
    msg = SQ.song_position_message(beat)
    with _device(cfg) as f:
        f.song_position(msg.pos)
    return {"beat": beat, "position_16ths": msg.pos, "bytes": list(msg.bytes()),
            "note": "the Field's SPP ORIGIN setting decides what moved (tape was the value seen); tell the human what you expect and ask what the screen shows"}


# ------------------------------------------------------------------ background jobs

@mcp.tool(annotations=RO)
def job_status(job_id: str) -> dict[str, Any]:
    """Progress of a background job (a long play, tape recording, arrangement part, drum pattern, vocoder line, tape backup or seed capture): status running, done, failed or cancelled, the stage, seconds elapsed and remaining, and when done the full result the direct call would have returned. Poll every few seconds; do not send other device calls while it runs, they are refused as busy."""
    job = JOBS.get(job_id)
    if job is None:
        raise KeyError(f"no job {job_id!r}; list_jobs shows the recent ones")
    return job.snapshot()


@mcp.tool(annotations=RO)
def list_jobs() -> list[dict[str, Any]]:
    """Background jobs, running and recently finished, oldest first (without their results; job_status has those)."""
    return [{k: v for k, v in j.snapshot().items() if k not in ("result",)} for j in JOBS.all()]


@mcp.tool(annotations=RW)
def cancel_job(job_id: str) -> dict[str, Any]:
    """Stop a running background job between events: held notes are released, a tape recording is stopped, a seed capture keeps what it heard. Returns the job's status; poll job_status until it reads cancelled."""
    job = JOBS.cancel(job_id)
    for _ in range(40):
        if job.done:
            break
        time.sleep(0.1)
    return job.snapshot()


# ------------------------------------------------------------------ seeds from the player

@mcp.tool(annotations=RW)
def capture_seed(wait_seconds: float = 60.0, max_seconds: float = 40.0, silence_seconds: float = 4.0, name: str | None = None, wait: bool | None = None) -> dict[str, Any]:
    """Listen while the human plays on the Field: returns the notes, detected chords and progression, key and tempo guesses, encoder moves (if transmitted), and the audio of what they played. Ask the human to play first, then call this. With the default waits it runs as a background job: poll job_status while the human plays (cancel_job ends the listening early and keeps what was heard)."""
    cfg = _cfg()

    def work(job):
        with _device(cfg, timeout=30.0 if job else 3.0) as f:
            _stage(job, "listening for the player")
            res = _capture_seed(f, wait_seconds, max_seconds, silence_seconds, cancel=_cancel(job))
        if not res.get("captured"):
            return res
        _stage(job, "analysing")
        sid = new_id(name or "seed"); d = _sess_dir(cfg)
        r = res.pop("_recording", None); pair = res.pop("_pair", None)
        if r is not None and pair is not None:
            import soundfile as sf
            sf.write(os.path.join(d, "seeds", sid + ".wav"), r.audio[:, pair[0] - 1: pair[1]], r.samplerate, subtype="FLOAT")
            an.spectrogram_png(r.audio[:, pair[0] - 1: pair[1]].mean(axis=1), r.samplerate, os.path.join(d, "seeds", sid + ".png"), marks=r.marks)
            res["wav"] = os.path.join(d, "seeds", sid + ".wav")
        res["id"] = sid
        write_json(os.path.join(d, "seeds", sid + ".json"), res)
        return res
    return _run_or_job("capture_seed", name or "seed", wait_seconds + max_seconds + 2.0, wait, work)


@mcp.tool(annotations=RO)
def list_seeds() -> list[dict[str, Any]]:
    """Seeds captured in this session."""
    cfg = _cfg(); d = os.path.join(_sess_dir(cfg), "seeds")
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            j = read_json(os.path.join(d, f))
            out.append({"id": j.get("id"), "progression": j.get("progression"), "key_guesses": (j.get("key_guesses") or [])[:2], "tempo_guess": j.get("tempo_guess")})
    return out


@mcp.tool(annotations=RO)
def get_seed(seed_id: str) -> dict[str, Any]:
    """Everything captured for a seed."""
    cfg = _cfg(); return read_json(os.path.join(_sess_dir(cfg), "seeds", seed_id + ".json"))




# ------------------------------------------------------------------ drums: patterns, kits, endless

from . import drums as DR, kits as KT


def _drum_analysis(res, score) -> dict[str, Any]:
    """Percussion has no pitch to check, so a hit counts as heard when an onset lands near its beat."""
    a = dict(res.analysis or {})
    notes = a.get("notes") or []
    for n in notes:
        n["heard"] = n.get("onset_offset_ms") is not None
        n.pop("prominence_db", None)
    a["hits_checked"] = len(notes)
    a["hits_heard"] = sum(1 for n in notes if n["heard"])
    a.pop("notes_heard", None); a.pop("notes_checked", None)
    offs = [n["onset_offset_ms"] for n in notes if n.get("onset_offset_ms") is not None]
    a["timing_ms"] = {"median": sorted(offs)[len(offs) // 2] if offs else None, "max_abs": max(abs(o) for o in offs) if offs else None}
    return a


def _kit_map_for_patterns(km: "KT.KitMap") -> dict[str, int]:
    """Every role plus every key by its Field name (F2, F#2 ...) and by MIDI number, so all 24 keys are usable in patterns."""
    out: dict[str, int] = dict(km.roles)
    for midi, key in sorted(km.keys.items()):
        out.setdefault(key.field_name or KT.field_key_name(midi), midi)
        out.setdefault(str(midi), midi)
    return out


@mcp.tool(annotations=RO)
def compile_drum_pattern(pattern: dict[str, Any]) -> dict[str, Any]:
    """Validate a drum grid document (see get_guide("drums")) and return its summary, an ASCII grid to double check, any problems against the session constraints, and the compiled score to pass to play or record_to_tape."""
    cfg = _cfg()
    d = DR.parse_pattern(pattern)
    score = DR.compile_drums(d)
    return {"summary": DR.summarize(d), "grid": DR.render_grid(d), "problems": validate(score, cfg), "score": score.model_dump()}


@mcp.tool(annotations=RW)
def play_drums(pattern: dict[str, Any], kit_slot: int | None = None, record: bool = True, to_tape: bool = False, name: str | None = None, wait: bool | None = None) -> dict[str, Any]:
    """Play a drum grid document on the loaded drum kit (or on drum slot kit_slot). Hits are checked by onset, not pitch. With to_tape=true the human must have selected a track and armed recording; the first hit starts the take and the tape is stopped at the end."""
    cfg = _cfg()
    d = DR.parse_pattern(pattern)
    score = DR.compile_drums(d)
    problems = validate(score, cfg)
    if problems:
        return {"played": False, "problems": problems, "grid": DR.render_grid(d)}
    if kit_slot is not None:
        cfg.check("sound_select")
        if not cfg.slot_allowed("drum", kit_slot):
            raise PermissionError(f"drum {kit_slot} is not in allowed_slots")
    def work(job):
        with _device(cfg, timeout=30.0 if job else 3.0) as f:
            if kit_slot is not None:
                f.select_drum_slot(kit_slot); time.sleep(0.4)
            else:
                f.set_mode("drum"); time.sleep(0.1)
            _stage(job, "recording to tape" if to_tape else "playing")
            try:
                res = play_score(f, score, record=record, latency_ms=cfg.latency_ms, cancel=_cancel(job))
            finally:
                if to_tape:
                    f.cc("tape_stop", 127)
        _stage(job, "analysing")
        out: dict[str, Any] = {"played": True, "seconds": round(res.seconds, 2), "hits": len(score.notes), "grid": DR.render_grid(d), "on_tape": to_tape}
        if record and res.recording is not None:
            tid = new_id(name or ("drums-tape" if to_tape else "drums"))
            dd = _sess_dir(cfg)
            wav = save_take(res, os.path.join(dd, "takes", tid + ".wav"))
            analysis = _drum_analysis(res, score)
            if res.live_pair:
                x = res.recording.audio[:, res.live_pair[0] - 1: res.live_pair[1]].mean(axis=1)
                an.spectrogram_png(x, res.recording.samplerate, os.path.join(dd, "takes", tid + ".png"), marks=res.recording.marks)
            write_json(os.path.join(dd, "takes", tid + ".json"), {"id": tid, "score": score.model_dump(), "pattern": pattern, "analysis": analysis, "wav": wav, "kind": "tape" if to_tape else "take"})
            out.update({"take_id": tid, "wav": wav, "analysis": analysis, "spectrogram": 'view_spectrogram("%s")' % tid})
        return out
    return _run_or_job("play_drums", name or "drums", _expected(score), wait, work)
    return out


@mcp.tool(annotations=RW)
def kit_map(slot: int, audition: bool = True, kit_name: str | None = None) -> dict[str, Any]:
    """Learn what every one of the 24 keys of drum slot `slot` holds. With audition=true the bridge plays each key on the Field, records it, classifies it (kick, snare, clap, closed hat, open hat, tom, cymbal, perc, or unsure with candidates) and saves the map; with audition=false it classifies from the kit's file in the latest disk backup. Human labels already saved for the slot always survive. Returns the per-key verdicts, the roles, and a kit_map ready to paste into patterns that exposes all 24 keys by role, Field key name and MIDI number."""
    cfg = _cfg()
    if audition:
        cfg.check("sound_select")
        if not cfg.slot_allowed("drum", slot):
            raise PermissionError(f"drum {slot} is not in allowed_slots")
        clip_dir = os.path.join(_sess_dir(cfg), "kits", f"drum-{slot}")
        os.makedirs(clip_dir, exist_ok=True)
        notes = list(range(DR.DRUM_FIRST_NOTE, DR.DRUM_LAST_NOTE + 1))
        with _device(cfg) as f:
            f.select_drum_slot(slot); time.sleep(0.5)
            rec = f.recorder().start()
            time.sleep(0.3)
            for n in notes:
                rec.mark(f"n{n}"); f.note_on(n, 110); time.sleep(0.12); f.note_off(n); time.sleep(0.6)
            time.sleep(0.6)
            r = rec.stop()
        pair = _live_pair(r.audio)
        if pair is None:
            return {"mapped": False, "hint": "no audio came back; is a tape track selected?"}
        import soundfile as sf
        for n in notes:
            s = r.mark_sample(f"n{n}")
            seg = r.audio[s: s + int(0.7 * r.samplerate), pair[0] - 1: pair[1]]
            sf.write(os.path.join(clip_dir, f"{n}_{midi_to_name(n)}.wav"), seg, r.samplerate, subtype="FLOAT")
        km = KT.kit_map_from_clips(clip_dir, kit_name=kit_name, slot=slot)
    else:
        src = _slot_file("drum", slot)
        if not src:
            raise FileNotFoundError(f"no backup file for drum slot {slot}; run with audition=true or enter disk mode and backup_disk")
        km = KT.kit_map_from_file(src, slot=slot)
        if kit_name:
            km.kit_name = kit_name
    path = KT.save_kit_map(HOME, slot, km)
    km = KT.load_kit_map_file(path)
    return {"mapped": True, "path": path, "kit_name": km.kit_name, "description": km.describe(), "roles": km.roles,
            "keys": {str(m): {"field_name": k.field_name, "label": k.label, "confidence": k.confidence, "notes": k.notes, "reason": k.reason} for m, k in sorted(km.keys.items())},
            "kit_map_for_patterns": _kit_map_for_patterns(km)}


@mcp.tool(annotations=RO)
def get_kit_map(slot: int) -> dict[str, Any]:
    """The saved map of a drum slot: every key's label and description, the roles, and a kit_map for patterns."""
    km = KT.load_kit_map(HOME, slot)
    if km is None:
        raise FileNotFoundError(f"no kit map saved for drum slot {slot}; call kit_map({slot})")
    return {"kit_name": km.kit_name, "description": km.describe(), "roles": km.roles,
            "keys": {str(m): {"field_name": k.field_name, "label": k.label, "notes": k.notes, "source": k.source} for m, k in sorted(km.keys.items())},
            "kit_map_for_patterns": _kit_map_for_patterns(km)}


@mcp.tool(annotations=RW)
def describe_kit_key(slot: int, key: str, label: str | None = None, description: str | None = None, by_human: bool = False, role: str | None = None) -> dict[str, Any]:
    """Annotate one key of a drum slot's map: key is a MIDI number (53-76) or the Field's key name (F2 ... E4); label is free text (shaker, brush, fx, rim, tom ...), description your words about it, role optionally binds a role name (BD, SN, CH, OH, SH, CL, RS, CB, CY, LT, MT, HT or your own) to this key. Set by_human=true when the human gave the label so it outranks the classifier."""
    km = KT.load_kit_map(HOME, slot)
    if km is None:
        raise FileNotFoundError(f"no kit map saved for drum slot {slot}; call kit_map({slot}) first")
    midi = int(key) if str(key).isdigit() else KT.field_name_to_midi(str(key))
    if not DR.DRUM_FIRST_NOTE <= midi <= DR.DRUM_LAST_NOTE:
        raise ValueError(f"key must be MIDI {DR.DRUM_FIRST_NOTE}-{DR.DRUM_LAST_NOTE} (Field F2 to E4)")
    kk = km.keys.get(midi) or KT.KitKey(midi=midi, field_name=KT.field_key_name(midi))
    if label:
        kk.label = label
        kk.confidence = 1.0 if by_human else kk.confidence
        kk.source = "human" if by_human else "model"
    if description:
        kk.notes = description
    km.keys[midi] = kk
    if role:
        km.roles[role.upper()] = midi
    path = KT.save_kit_map(HOME, slot, km, keep_human_labels=not by_human)
    return {"saved": path, "key": {"midi": midi, "field_name": kk.field_name, "label": kk.label, "notes": kk.notes, "source": kk.source}, "roles": km.roles}


@mcp.tool(annotations=RW)
def endless_program(steps: list[Any], kit_slot: int | None = None, gap_ms: int = 250, hold_beats: float = 0.0, tempo: float | None = None) -> dict[str, Any]:
    """Enter steps into the Field's endless sequencer over MIDI: each entry is one step (a MIDI note, a Field key name like F2, or a role from the slot's kit map), sent as a short tap. The human must first enable endless on the device (human_steps enable_endless); endless is one note per step with no rests, so give it one instrument row at a time or a melodic line, and set the note value on the device. With hold_beats > 0 the first step is then held that long so the sequence plays back and is recorded."""
    cfg = _cfg(); cfg.check("transport")
    km = KT.load_kit_map(HOME, kit_slot) if kit_slot else None
    notes: list[int] = []
    for s in steps:
        if isinstance(s, int) or (isinstance(s, str) and s.isdigit()):
            notes.append(int(s))
        elif km is not None and str(s).upper() in km.roles:
            notes.append(km.roles[str(s).upper()])
        else:
            try:
                notes.append(KT.field_name_to_midi(str(s)))
            except Exception:
                notes.append(pitch_to_midi(str(s)))
    if not notes:
        raise ValueError("give at least one step")
    with _device(cfg) as f:
        if kit_slot is not None:
            cfg.check("sound_select"); f.select_drum_slot(kit_slot); time.sleep(0.4)
        for n in notes:
            f.note_on(n, 110); time.sleep(0.06); f.note_off(n); time.sleep(max(0.05, gap_ms / 1000.0))
        out: dict[str, Any] = {"entered_steps": len(notes), "notes": notes}
        if hold_beats > 0:
            bpm = tempo or 120.0
            rec = f.recorder().start(); time.sleep(0.3)
            rec.mark("hold"); f.note_on(notes[0], 110); time.sleep(hold_beats * 60.0 / bpm); f.note_off(notes[0]); rec.mark("release"); time.sleep(0.6)
            r = rec.stop()
            pair = _live_pair(r.audio)
            if pair:
                x = r.audio[:, pair[0] - 1: pair[1]].mean(axis=1); sr = r.samplerate
                t, flux = an.onset_strength(x, sr)
                a, b = r.mark_sample("hold") / sr, r.mark_sample("release") / sr
                sel = (t >= a) & (t <= b)
                ff = flux[sel]; tt = t[sel]
                thr = max(float(np.percentile(ff, 90)) * 1.5, float(ff.max()) * 0.25) if len(ff) else 0
                ons = []
                for i in range(1, len(ff) - 1):
                    if ff[i] > thr and ff[i] >= ff[i - 1] and ff[i] >= ff[i + 1] and (not ons or tt[i] - ons[-1] > 0.08):
                        ons.append(float(tt[i]))
                tid = new_id("endless"); dd = _sess_dir(cfg)
                import soundfile as sf
                sf.write(os.path.join(dd, "takes", tid + ".wav"), r.audio[:, pair[0] - 1: pair[1]], sr, subtype="FLOAT")
                out.update({"playback_hits": len(ons), "playback_seconds": round(b - a, 2), "take_id": tid,
                            "note": "many hits during the hold means the sequence played; one hit means endless is not enabled"})
    return out




# ------------------------------------------------------------------ what the Field is set to

@mcp.tool(annotations=RW)
def set_device_state(mode: str | None = None, synth_slot: int | None = None, drum_slot: int | None = None, tempo_bpm: float | None = None, note: str | None = None, master_fx: str | None = None) -> dict[str, Any]:
    """Tell the bridge what the human changed by hand on the Field (mode synth/drum, the sound key in use, the tempo, the master effect type). The Field cannot be queried and transmits nothing when these change, so this is how hand changes become known."""
    fields: dict[str, Any] = {"source": "human"}
    if master_fx is not None:
        from . import effects as FXm
        if master_fx not in FXm.EFFECTS:
            raise ValueError(f"master_fx must be one of {sorted(FXm.EFFECTS)}")
        fields["master_fx"] = master_fx
    if mode is not None:
        if mode not in ("synth", "drum"):
            raise ValueError("mode must be synth or drum")
        fields["mode"] = mode
    if synth_slot is not None:
        if not 1 <= int(synth_slot) <= 8:
            raise ValueError("synth_slot must be 1-8")
        fields["synth_slot"] = int(synth_slot)
    if drum_slot is not None:
        if not 1 <= int(drum_slot) <= 8:
            raise ValueError("drum_slot must be 1-8")
        fields["drum_slot"] = int(drum_slot)
    if tempo_bpm is not None:
        fields["tempo_bpm"] = float(tempo_bpm)
    if note:
        fields["note"] = note
    return update_state(**fields)


@mcp.tool(annotations=RW)
def detect_mode() -> dict[str, Any]:
    """Guess by ear whether the Field is in synth or drum mode: plays two adjacent keys briefly and compares them. A synth answers with the same timbre a semitone apart and sustains while held; a drum kit answers with unrelated sounds that decay on their own. Audible, about two seconds. The guess is recorded in the state as source detected."""
    cfg = _cfg()
    with _device(cfg) as f:
        rec = f.recorder().start(); time.sleep(0.3)
        marks = []
        for n in (60, 61):
            marks.append(rec.mark(f"n{n}")); f.note_on(n, 100); time.sleep(0.5); f.note_off(n); time.sleep(0.5)
        r = rec.stop()
    pair = _live_pair(r.audio)
    if pair is None:
        return {"mode_guess": None, "confidence": 0.0, "evidence": "no audio came back; is a tape track selected?"}
    sr = r.samplerate
    x = r.audio[:, pair[0] - 1: pair[1]].mean(axis=1)
    feats = []
    for n, s in zip((60, 61), marks):
        seg = x[s: s + int(0.5 * sr)]
        t, lev = an.envelope_db(seg, sr, 10.0)
        pk = float(lev.max()) if len(lev) else -120.0
        late = float(lev[int(len(lev) * 0.7):].mean()) if len(lev) > 10 else -120.0
        body = seg[int(0.08 * sr): int(0.45 * sr)]
        prom = an.note_prominence(body, sr, [n], harmonics=2)[n] if len(body) > 2048 else -60.0
        tt, cent = an.spectral_centroid_series(body, sr) if len(body) > 4096 else (None, None)
        feats.append({"note": n, "peak_db": round(pk, 1), "sustain_drop_db": round(pk - late, 1), "pitch_prominence_db": round(prom, 1), "centroid_hz": round(float(np.median(cent))) if cent is not None else None})
    a, b = feats
    pitched = a["pitch_prominence_db"] >= 12 and b["pitch_prominence_db"] >= 12
    sustains = a["sustain_drop_db"] < 12 and b["sustain_drop_db"] < 12
    similar = a["centroid_hz"] and b["centroid_hz"] and 0.6 < a["centroid_hz"] / max(1, b["centroid_hz"]) < 1.7
    score = (1 if pitched else 0) + (1 if sustains else 0) + (1 if similar else 0)
    guess = "synth" if score >= 2 else "drum"
    conf = round({3: 0.9, 2: 0.7, 1: 0.7, 0: 0.9}[score], 2)
    update_state(mode=guess, source="detected")
    return {"mode_guess": guess, "confidence": conf, "evidence": {"both_keys_pitched_at_their_note": pitched, "both_sustain_while_held": sustains, "similar_timbre": similar, "keys": feats},
            "note": "an audible guess, not a reading; the Field has no state read-back"}




# ------------------------------------------------------------------ speech into the vocoder

from . import speech as SP


@mcp.tool(annotations=RO)
def list_voices() -> list[dict[str, str]]:
    """Voices the Mac's built-in speech engine offers for speak_through_vocoder (offline)."""
    return SP.available_voices()


@mcp.tool(annotations=RW)
def speak_through_vocoder(text: str | None = None, lines: list[dict[str, Any]] | None = None, chords: list[list[str]] | None = None, beats_per_chord: float = 2.0, tempo: float = 90.0,
                          voice: str | None = None, rate: int | None = None, slot: int | None = None, speech_gain_db: float = 0.0, velocity: int = 100, record: bool = True,
                          name: str | None = None, hold_chords: bool = True, send_clock: bool = False, to_tape: bool = False, min_beats: float | None = None, input_latency_ms: float = 100.0,
                          wait: bool | None = None) -> dict[str, Any]:
    """Speak words through the Field's vocoder: the Mac synthesises them offline and streams them into the Field's USB input as the modulator while the bridge holds carrier chords over MIDI, and records the result. The human must first load a vocoder preset (slot, or already loaded), set the input source to usb audio and switch the input on (human_steps vocoder_setup). Words: `text` (one line at beat 0) or `lines`, a list of {"beat": b, "text": "..."} (or {"seconds": s, ...}) placed on the score's beat grid, with [[slnc 400]] for a pause inside a line. Chords: lists of note names cycled every beats_per_chord over the whole length (the last word's end, or min_beats if longer); with hold_chords a repeated chord is held rather than re-struck and the last chord holds to the end. send_clock sends CC 80 tempo and MIDI clock like play; to_tape records onto the armed track (nothing but notes and clock is sent, the tape is stopped at the end). Recipe measured on this Field: input at middle gain; waveform 0, formant 64, bands 120, mix 0 (mix only blends in the dry voice); a 3-4 note carrier with a low note around Bb2-F3, velocity about 90. The result carries an STOI intelligibility score (0.75 is clear, below 0.5 is mush) and on_beat_error_ms, how late the words arrived against their beats after the Mac-to-Field latency (input_latency_ms; measured 100 ms through the vocoder, 200 ms on the dry input path) was compensated. Longer speech runs as a job."""
    import soundfile as sf
    cfg = _cfg()
    if slot is not None:
        cfg.check("sound_select")
        if not cfg.slot_allowed("synth", slot):
            raise PermissionError(f"synth {slot} is not in allowed_slots")
    if not lines:
        if not text or not str(text).strip():
            raise ValueError("give text or lines")
        lines = [{"beat": 0.0, "text": text}]
    d = _sess_dir(cfg); os.makedirs(os.path.join(d, "speech"), exist_ok=True)
    sid = new_id(name or "speech")
    rendered: list[dict[str, Any]] = []
    pieces: list[tuple[float, np.ndarray]] = []
    ssr: int | None = None
    for i, ln in enumerate(lines):
        t = str(ln.get("text", "")).strip()
        if not t:
            continue
        path = os.path.join(d, "speech", f"{sid}-{i}.wav")
        info = SP.synthesize(t, path, voice=voice, rate=rate)
        a, sr_i = sf.read(path, dtype="float32", always_2d=True)
        if ssr is None:
            ssr = int(sr_i)
        elif int(sr_i) != ssr:
            raise RuntimeError(f"speech sample rates differ ({sr_i} vs {ssr})")
        start_s = float(ln["seconds"]) if ln.get("seconds") is not None else float(ln.get("beat", 0.0)) * 60.0 / tempo
        rendered.append({"line": i, "text": t, "beat": round(start_s * tempo / 60.0, 3), "start_s": round(start_s, 3), "seconds": info["seconds"], "voice": info["voice"], "path": path})
        pieces.append((start_s, a.mean(axis=1)))
    if not pieces or ssr is None:
        raise ValueError("no words to speak")
    speech_end = max(r["start_s"] + r["seconds"] for r in rendered)
    total_beats = max((speech_end + 0.4) * tempo / 60.0, float(min_beats or 0.0))
    total_s = total_beats * 60.0 / tempo
    buf = np.zeros(int((total_s + 1.0) * ssr), np.float32)
    for start_s, a in pieces:
        k = int(start_s * ssr); seg = a[: max(0, len(buf) - k)]
        buf[k: k + len(seg)] += seg
    peak = float(np.abs(buf).max())
    if peak > 0.98:
        buf *= 0.98 / peak
    wav = os.path.join(d, "speech", sid + ".wav")
    sf.write(wav, buf, ssr, subtype="FLOAT")
    chords = chords or [["C3", "G3", "C4", "E4"], ["A2", "E3", "A3", "C4"]]
    segs: list[list[Any]] = []   # [start_beat, duration, chord]
    beat = 0.0; i = 0
    while beat < total_beats:
        dur = min(beats_per_chord, total_beats - beat)
        chord = list(chords[i % len(chords)])
        if hold_chords and segs and segs[-1][2] == chord:
            segs[-1][1] += dur
        else:
            segs.append([beat, dur, chord])
        beat += beats_per_chord; i += 1
    notes = [{"start": round(s0, 4), "duration": round(d0, 4), "pitch": p, "velocity": velocity} for s0, d0, c0 in segs for p in c0]
    score = Score(tempo=tempo, notes=notes, tail_seconds=1.0, send_clock=send_clock)
    problems = validate(score, cfg)
    if problems:
        return {"spoken": False, "problems": problems}
    pre_roll_s = 0.5                               # play_score's pre-roll before beat 0
    lead_s = max(0.0, pre_roll_s - float(input_latency_ms) / 1000.0)   # send the speech early by the Mac-to-Field latency
    lead = np.zeros(int(lead_s * ssr), np.float32)
    playback = np.concatenate([lead, buf])[:, None]
    gain = 10 ** (speech_gain_db / 20.0)
    full_text = " / ".join(r["text"] for r in rendered)

    def work(job):
        with _device(cfg, timeout=30.0 if job else 3.0) as f:
            if slot is not None:
                f.select_synth_slot(slot); time.sleep(0.4)
            _stage(job, "recording to tape" if to_tape else "speaking")
            try:
                res = play_score(f, score, record=record, latency_ms=cfg.latency_ms, playback=playback, playback_gain=gain, set_tempo=not to_tape, cancel=_cancel(job))
            finally:
                if to_tape:
                    f.cc("tape_stop", 127)
        _stage(job, "analysing")
        out: dict[str, Any] = {"spoken": True, "lines": [{k: v for k, v in r.items() if k != "path"} for r in rendered], "speech_wav": wav, "chords": [c for _, _, c in segs],
                               "chord_beats": [[round(s0, 3), round(d0, 3)] for s0, d0, _ in segs], "seconds": round(res.seconds, 2), "on_tape": to_tape, "clock": send_clock}
        if record and res.recording is not None:
            tid = new_id(name or ("vocoder-tape" if to_tape else "vocoder"))
            take = save_take(res, os.path.join(d, "takes", tid + ".wav"))
            if res.live_pair:
                x = res.recording.audio[:, res.live_pair[0] - 1: res.live_pair[1]].mean(axis=1)
                an.spectrogram_png(x, res.recording.samplerate, os.path.join(d, "takes", tid + ".png"), marks=res.recording.marks)
            write_json(os.path.join(d, "takes", tid + ".json"), {"id": tid, "score": score.model_dump(), "text": full_text, "lines": out["lines"], "speech_wav": wav, "analysis": res.analysis, "wav": take, "kind": "tape" if to_tape else "vocoder"})
            out.update({"take_id": tid, "wav": take, "analysis": {k: v for k, v in res.analysis.items() if k != "notes"}, "spectrogram": 'view_spectrogram("%s")' % tid,
                        "hint": "silence means the Field's input is not on usb audio or the input is off; a dry voice with no chords means no vocoder preset is loaded"})
            try:
                if res.live_pair:
                    rec_pair = res.recording.audio[:, res.live_pair[0] - 1: res.live_pair[1]].astype(np.float32) / (32768.0 if res.recording.audio.dtype.kind == "i" else 1.0)
                    out["intelligibility"] = an.speech_intelligibility(buf, ssr, rec_pair, res.recording.samplerate)
                    lag = out["intelligibility"].get("lag_ms")
                    if lag is not None:
                        # the recording starts pre_roll_s before beat 0; the speech buffer starts at beat 0
                        out["on_beat_error_ms"] = round(float(lag) - (res.t_score_start_sample or 0) / res.recording.samplerate * 1000.0, 0)
            except Exception as e:  # a missing score never hides the take
                out["intelligibility"] = {"stoi": None, "error": f"{type(e).__name__}: {e}"}
        return out
    return _run_or_job("speak_through_vocoder", name or "vocoder", _expected(score, extra=3.0), wait, work)


@mcp.tool(annotations=RW)
def stream_to_tape(wav_path: str | None = None, text: str | None = None, voice: str | None = None, rate: int | None = None, gain_db: float = 0.0, start: str = "play",
                   count_in_beats: float = 4.0, tempo: float | None = None, record: bool = True, name: str | None = None, wait: bool | None = None) -> dict[str, Any]:
    """Put audio from the Mac onto the Field's tape dry, with no vocoder: a WAV (any vocal, a rendered voice, a sample) or `text` the Mac speaks is streamed into the Field's USB input while the tape records; no notes are played. The human first sets the input to usb audio and switches it on, then selects the track and arms recording (human_steps input_to_tape). start="play" sends tape play, which on an armed tape gives the Field's count-in (count_in_beats at `tempo`, or the tempo the bridge last set; 0 when the Field's count-in is off) before recording starts, and the audio is streamed right after it; start="now" streams at once for a recording the human already started. The tape is stopped at the end and the USB return is kept as a take with the level heard and the lag between sent and heard audio (speech_intelligibility on it gives STOI for spoken material). Verified 2026-09-28 with TE's procedure (human_steps input_to_tape): the external audio is toggled with the input key while in tape mode and recording is started by holding record and pressing play, so use start="now" once the human says the tape is rolling; the line then sits on the track at the same clarity as the monitored return (STOI 0.94) about 200 ms after it left the Mac. Switching the input on from the input screen or synth mode feeds only the vocoder and sampler, and play over MIDI does not start an armed take. Another dry route: a sampler preset authored from the WAV (6 s), installed and played to the armed track."""
    import soundfile as sf
    cfg = _cfg()
    if start not in ("play", "now"):
        raise ValueError("start must be 'play' or 'now'")
    d = _sess_dir(cfg); os.makedirs(os.path.join(d, "speech"), exist_ok=True)
    src = wav_path
    spoken = None
    if text and str(text).strip():
        sid = new_id(name or "speech")
        src = os.path.join(d, "speech", sid + ".wav")
        spoken = SP.synthesize(str(text), src, voice=voice, rate=rate)
    if not src or not os.path.exists(src):
        raise FileNotFoundError("give wav_path (an existing WAV) or text")
    audio, ssr = sf.read(src, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    bpm = float(tempo or (read_state().get("tempo_bpm") or 120.0))
    count_in_s = float(count_in_beats) * 60.0 / bpm if start == "play" else 0.0
    lead_s = 0.3 + count_in_s + (0.05 if start == "play" else 0.0)
    lead = np.zeros(int(lead_s * ssr), np.float32)
    pb = np.concatenate([lead, mono, np.zeros(int(0.3 * ssr), np.float32)])
    playback = np.stack([pb, pb], axis=1)
    gain = 10 ** (gain_db / 20.0)
    total_s = len(pb) / ssr

    def work(job):
        with _device(cfg, timeout=30.0 if job else 3.0) as f:
            if f.audio is None or int(f.audio["samplerate"]) != int(ssr):
                raise RuntimeError(f"the WAV must be at the Field's rate ({f.audio and f.audio['samplerate']} Hz); resample it first")
            rec = f.duplex_recorder(playback, gain=gain)
            rec.start()
            t0 = time.perf_counter()
            try:
                time.sleep(0.3)
                if start == "play":
                    rec.mark("play")
                    f.cc("tape_play", 127)
                _stage(job, "streaming into the Field's input")
                while time.perf_counter() - t0 < total_s + 0.2:
                    time.sleep(0.2)
                    if job is not None and job.cancel.is_set():
                        raise Cancelled()
            finally:
                f.cc("tape_stop", 127)
                r = rec.stop()
        _stage(job, "analysing")
        started = rec.playback_started_at_sample or 0
        audio_start = started + len(lead)
        r.marks.append(("audio_start", int(audio_start)))
        pair = _live_pair(r.audio)
        out: dict[str, Any] = {"streamed": True, "source_wav": src, "seconds": round(len(mono) / ssr, 2), "count_in_seconds": round(count_in_s, 2), "tempo_bpm": bpm, "start": start,
                               "audio_started_at_s": round(audio_start / r.samplerate, 3)}
        if spoken:
            out["speech"] = spoken
        if record:
            tid = new_id(name or "input-tape")
            wav = os.path.join(d, "takes", tid + ".wav")
            sf.write(wav, r.audio[:, pair[0] - 1: pair[1]] if pair else r.audio[:, :2], r.samplerate, subtype="FLOAT")
            analysis: dict[str, Any] = {"live_usb_channels": list(pair) if pair else None}
            if pair:
                x = r.audio[:, pair[0] - 1: pair[1]].astype(np.float32) / (32768.0 if r.audio.dtype.kind == "i" else 1.0)
                seg = x[int(audio_start):]
                analysis.update({"peak_db": an.peak_db(seg.mean(axis=1)) if len(seg) else None, "rms_db": an.rms_db(seg.mean(axis=1)) if len(seg) else None})
                an.spectrogram_png(x.mean(axis=1), r.samplerate, os.path.join(d, "takes", tid + ".png"), marks=r.marks)
                try:
                    al = an.speech_intelligibility(mono, ssr, seg, r.samplerate)
                    analysis["alignment"] = al
                except Exception as e:
                    analysis["alignment"] = {"error": f"{type(e).__name__}: {e}"}
            else:
                analysis["hint"] = "nothing came back: the input is off, not on usb audio, or no track is selected"
            write_json(os.path.join(d, "takes", tid + ".json"), {"id": tid, "score": {}, "kind": "input", "source_wav": src, "speech_wav": src, "analysis": analysis, "wav": wav})
            out.update({"take_id": tid, "wav": wav, "analysis": analysis, "spectrogram": f'view_spectrogram("{tid}")',
                        "note": "the tape was stopped; if recording was armed the audio is on that track (tape('start') then tape('play') to hear it, backup_tape to capture it)"})
        return out
    return _run_or_job("stream_to_tape", name or "input-tape", total_s + 3.0, wait, work)


@mcp.tool(annotations=RW)
def probe_usb_input(seconds: float = 1.5, gain_db: float = -12.0) -> dict[str, Any]:
    """Send a test tone from the Mac into the Field's USB input and listen for it coming back, to check the input routing before speak_through_vocoder. Nothing plays over MIDI."""
    cfg = _cfg()
    d = _sess_dir(cfg); os.makedirs(os.path.join(d, "speech"), exist_ok=True)
    path = os.path.join(d, "speech", "probe-tone.wav")
    sr = 44100; t = np.arange(int(seconds * sr)) / sr
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t) * np.minimum(1.0, t * 20) * np.minimum(1.0, (seconds - t) * 20)).astype(np.float32)
    import soundfile as sf
    sf.write(path, tone, sr)
    lead = np.zeros((int(0.3 * sr),), np.float32)
    with _device(cfg) as f:
        rec = f.duplex_recorder(np.concatenate([lead, tone]), gain=10 ** (gain_db / 20.0)).start()
        time.sleep(0.3); rec.mark("tone"); time.sleep(seconds + 0.6)
        r = rec.stop()
    s = r.mark_sample("tone")
    seg = r.audio[s: s + int(seconds * r.samplerate)]
    levels = [round(an.rms_db(seg[:, c]), 1) for c in range(seg.shape[1])]
    heard = [c + 1 for c, lv in enumerate(levels) if lv > -60]
    prom = an.note_prominence(seg[:, heard[0] - 1] if heard else seg[:, 0], r.samplerate, [69], harmonics=1)[69] if len(seg) > 4096 else -60
    return {"tone_heard_on_usb_channels": heard, "levels_db": levels, "a4_prominence_db": round(float(prom), 1),
            "verdict": "the Field passes usb audio input through" if heard and prom > 15 else "nothing came back: on the Field hold shift + input, choose usb audio, press input to enable it"}




# ------------------------------------------------------------------ sound profiles, range guarding, the sound index, search, arrangement

from . import sounds as SO, judge as JU, arrangement as AR


def _profile(kind: str, slot: int) -> SO.SlotProfile:
    src = _slot_file(kind, slot)
    try:
        return SO.profile_from_file(src, kind, slot) if src else SO.default_profile(kind, slot)
    except Exception:
        return SO.default_profile(kind, slot)


def _guard_score(score: Score, sound: dict[str, Any] | None, auto_transpose: bool) -> tuple[Score, dict[str, Any]]:
    """Check a score against the profile of the sound it targets; transpose by octaves when allowed."""
    if not sound or not sound.get("slot"):
        return score, {}
    kind = sound.get("kind", "synth"); slot = int(sound["slot"])
    prof = _profile(kind, slot)
    midis = [n.midi for n in score.notes]
    out = SO.notes_out_of_range(midis, prof)
    info: dict[str, Any] = {"profile": prof.to_dict()}
    if not out:
        return score, info
    if not auto_transpose:
        info["out_of_range"] = [midi_to_name(m) for m in out]
        return score, info
    shift, notes = SO.transpose_into_range(midis, prof)
    if shift:
        d = score.model_dump()
        d["notes"] = [{**n, "pitch": pitch_to_midi(n["pitch"]) + shift} for n in d["notes"]]
        score = Score.model_validate(d)
    info["transposed_semitones"] = shift
    info["notes"] = notes
    return score, info


@mcp.tool(annotations=RO)
def slot_profile(kind: str, slot: int) -> dict[str, Any]:
    """What a sound slot can play: engine, name, a sampler's root note, and the playable and ideal MIDI ranges, read from the slot's file in the latest disk backup. Samplers keep their character only within an octave of the root; synthesis engines pitch every note literally."""
    return _profile(kind, slot).to_dict()


@mcp.tool(annotations=RW)
def audit_sound(kind: str = "synth", slot: int = 1, notes: list[str] | None = None, seconds: float = 1.2, velocity: int = 100, name: str | None = None) -> dict[str, Any]:
    """Learn a sound: plays a low, middle and high note (or the notes given) on the slot while recording, measures each (attack, sustain, release, brightness, bands, noisiness, harmonicity, movement, width, pitch error), derives tags and role suggestions, and stores the result in the sound index so it is never auditioned again. Returns the compact record; view_spectrogram shows the picture."""
    cfg = _cfg(); cfg.check("sound_select")
    if not cfg.slot_allowed(kind, slot):
        raise PermissionError(f"{kind} {slot} is not in allowed_slots")
    prof = _profile(kind, slot)
    lo, hi = prof.ideal
    pitches = [pitch_to_midi(p) for p in notes] if notes else ([lo, (lo + hi) // 2, hi] if kind == "synth" else [53, 55, 60])
    with _device(cfg) as f:
        (f.select_synth_slot if kind == "synth" else f.select_drum_slot)(slot); time.sleep(0.4)
        rec = f.recorder().start(); time.sleep(0.3)
        marks = []
        for m in pitches:
            on = rec.mark(f"on{m}"); f.note_on(m, velocity); time.sleep(seconds); off = rec.mark(f"off{m}"); f.note_off(m); time.sleep(min(2.5, max(1.0, seconds)))
            marks.append((m, on, off))
        r = rec.stop()
    pair = _live_pair(r.audio)
    if pair is None:
        return {"audited": False, "hint": "no audio came back; is a tape track selected and the Field in normal mode?"}
    audio = r.audio[:, pair[0] - 1: pair[1]]
    per_note = {}
    for m, on, off in marks:
        deep = SO.measure_deep(audio, r.samplerate, on, off, m)
        feats = SO.SoundFeatures(**{k: deep[k] for k in SO.SoundFeatures.__dataclass_fields__})
        per_note[midi_to_name(m)] = {"features": deep, "descriptors": SO.describe(deep), **SO.tags_for(feats, m)}
    mid = per_note[midi_to_name(pitches[len(pitches) // 2])]
    all_tags = sorted({t for v in per_note.values() for t in v["tags"]})
    roles = sorted({rr for v in per_note.values() for rr in v["roles"] if rr != "unclassified"}) or ["unclassified"]
    descriptors = sorted({dd for v in per_note.values() for dd in v["descriptors"]})
    regs = ["low", "mid", "high"] if len(pitches) == 3 else [midi_to_name(m) for m in pitches]
    per_register = {reg: per_note[midi_to_name(m)]["descriptors"] for reg, m in zip(regs, pitches)}
    sid = name or f"{kind}-{slot}-{prof.name or 'slot'}".replace(" ", "-")
    d = _sess_dir(cfg)
    png = os.path.join(d, "takes", sid + ".png")
    an.spectrogram_png(audio.mean(axis=1), r.samplerate, png, marks=r.marks)
    mf = mid["features"]
    measured = {k: mf.get(k) for k in ("attack_ms", "sustain_ratio_db", "release_ms", "centroid_hz", "odd_even_db", "harmonics_above_minus40db", "tilt_db_per_octave", "inharmonic_share", "flatness", "brightness_change", "level_mod_hz", "brightness_mod_hz", "width", "bands", "peak_db")}
    entry = {"kind": kind, "slot": slot, "engine": prof.engine, "name": prof.name, "root": SO.note_name(prof.root_midi) if prof.root_midi is not None else None,
             "playable": [SO.note_name(prof.playable[0]), SO.note_name(prof.playable[1])], "tags": all_tags, "roles": roles, "descriptors": descriptors,
             "summary": {"brightness_hz": mf["centroid_hz"], "attack_ms": mf["attack_ms"], "sustain_db": mf["sustain_ratio_db"], "release_ms": mf["release_ms"], "peak_db": mf["peak_db"]},
             "measured": measured, "per_register": per_register, "per_note": per_note, "spectrogram": png}
    note = cfg.palette.get(f"{kind} {slot}", {}).get("description")
    if note:
        entry["human_note"] = note
    entry["metadata_source"] = prof.source
    entry["effect"] = _slot_fx(kind, slot).get("fx_type")
    tax = JU.jev_taxonomy(entry) if JU.jev_available() else None
    if tax:
        entry["taxonomy"] = tax
    SO.record_sound(HOME, sid, entry)
    out = {"audited": True, "id": sid, "engine": prof.engine, "name": prof.name, "descriptors": descriptors, "roles": roles, "per_register": per_register, "summary": entry["summary"], "playable": entry["playable"], "root": entry["root"]}
    if tax:
        out["jev"] = {q: (a["choice"], round(float(a["confidence"] or 0), 2)) for q, a in tax.items()}
    return out


@mcp.tool(annotations=RO)
def find_sounds(query: str, top: int = 5, kind: str | None = None) -> dict[str, Any]:
    """Search the sound index with a plain request such as "dark sustained bass" or "bright plucky keys". Uses the Jev judge when TYPESAFE_API_KEY is set, otherwise the local rule-based ranking. Returns the best few with their tags and where they live."""
    idx = SO.load_index(HOME)
    cands = [{"id": sid, **{k: v for k, v in e.items() if k not in ("per_note", "spectrogram")}} for sid, e in idx["sounds"].items() if not kind or e.get("kind") == kind]
    if not cands:
        return {"judge": None, "results": [], "hint": "the sound index is empty; run audit_sound on slots (or kit_map for drums) first"}
    r = JU.rank(cands, query, top=top)
    return {"judge": r["judge"], "request_implies": r["wants"], "results": [{"score": round(float(s), 3), **{k: v for k, v in c.items() if k not in ("measured", "per_register", "taxonomy", "summary")}} for s, c in r["ranked"]], "indexed": len(cands)}


@mcp.tool(annotations=RW)
def sound_search(slot: int, want: str, iterations: int = 6, note: str | None = None, seconds: float = 1.0, keep: str = "best") -> dict[str, Any]:
    """Explore a synth slot by randomizing it (CC 62) and auditioning each result, scored against `want` (for example "dark sustained bass" or "bright short pluck"). Ends on the best candidate found when keep="best" (it re-rolls until that candidate's seed... no: the Field cannot recall a roll, so the final state is the best of the LAST roll and the bridge reports every roll's tags; ask the human to hold the sound key two seconds to keep what is loaded). keep="last" leaves the last roll. The saved preset is untouched until the human saves."""
    cfg = _cfg(); cfg.check("sound_design")
    if not cfg.slot_allowed("synth", slot):
        raise PermissionError(f"synth {slot} is not in allowed_slots")
    prof = _profile("synth", slot)
    m = pitch_to_midi(note) if note else (prof.ideal[0] + prof.ideal[1]) // 2
    rolls = []
    with _device(cfg) as f:
        f.select_synth_slot(slot); time.sleep(0.4)
        for i in range(max(1, int(iterations))):
            f.cc("randomize", 127); time.sleep(0.25)
            rec = f.recorder().start(); time.sleep(0.25)
            on = rec.mark("on"); f.note_on(m, 100); time.sleep(seconds); off = rec.mark("off"); f.note_off(m); time.sleep(1.2)
            r = rec.stop()
            pair = _live_pair(r.audio)
            if pair is None:
                rolls.append({"roll": i + 1, "silent": True}); continue
            deep = SO.measure_deep(r.audio[:, pair[0] - 1: pair[1]], r.samplerate, on, off, m)
            feats = SO.SoundFeatures(**{k: deep[k] for k in SO.SoundFeatures.__dataclass_fields__})
            t = SO.tags_for(feats, m)
            rolls.append({"roll": i + 1, "descriptors": SO.describe(deep), "tags": t["tags"], "roles": t["roles"], "measured": {k: deep.get(k) for k in ("attack_ms", "sustain_ratio_db", "centroid_hz", "odd_even_db", "inharmonic_share", "brightness_change", "level_mod_hz")}, "brightness_hz": feats.centroid_hz, "attack_ms": feats.attack_ms, "sustain_db": feats.sustain_ratio_db, "peak_db": feats.peak_db})
    ranked = JU.rank([r for r in rolls if not r.get("silent")], want, top=len(rolls))
    best = ranked["ranked"][0] if ranked["ranked"] else None
    return {"slot": slot, "want": want, "judge": ranked["judge"], "rolls": rolls,
            "best_roll": best[1]["roll"] if best else None, "best_score": round(float(best[0]), 3) if best else None,
            "loaded_now": "the LAST roll" if keep != "best" else "the last roll; the Field cannot recall an earlier roll, so if the best roll was earlier, roll again toward it or accept the last one",
            "to_keep": "ask the human to hold the sound key for two seconds, which saves the loaded sound as a snapshot; a program change reloads the saved preset and discards the roll"}


@mcp.tool(annotations=RO)
def compile_arrangement(arrangement: dict[str, Any]) -> dict[str, Any]:
    """Compile a node-based arrangement (see get_guide("arrangement")) into one score per part: sections with bar counts, parts with a clip per section (score, drum pattern, chord list, reuse of another section, or rest), automation nodes, and edges or an order. Returns the timeline, each part's length and warnings; scores are cached by part name for play_arrangement_part."""
    out = AR.compile_arrangement(arrangement)
    parts = {}
    for name, cp in out["parts"].items():
        parts[name] = {"sound": cp.sound, "track": cp.track, "notes": len(cp.score.notes), "automations": len(cp.score.cc), "sections_with_clips": cp.clips_used, "warnings": cp.warnings, "problems": validate(cp.score, _cfg())}
    return {"order": out["order"], "total_bars": out["total_bars"], "seconds": out["seconds"], "timeline": out["timeline"], "parts": parts}


@mcp.tool(annotations=RW)
def play_arrangement_part(arrangement: dict[str, Any], part: str, record: bool = True, to_tape: bool = False, auto_transpose: bool = True, name: str | None = None, wait: bool | None = None) -> dict[str, Any]:
    """Play one part of an arrangement on its sound (selected first), guarding its notes against the slot's playable range. With to_tape=true the human must have selected the part's track and armed recording; nothing but clock and notes is sent, and the tape is stopped at the end. Parts longer than about 25 s run as a background job (poll job_status for the same result)."""
    cfg = _cfg()
    out = AR.compile_arrangement(arrangement)
    if part not in out["parts"]:
        raise ValueError(f"no part {part!r}; parts: {list(out['parts'])}")
    cp = out["parts"][part]
    sound = cp.sound or {}
    score, guard = _guard_score(cp.score, sound, auto_transpose)
    problems = validate(score, cfg)
    if problems:
        return {"played": False, "problems": problems, "guard": guard}
    if sound.get("slot"):
        cfg.check("sound_select")

    def work(job):
        with _device(cfg, timeout=30.0 if job else 3.0) as f:
            if sound.get("slot"):
                (f.select_synth_slot if sound.get("kind", "synth") == "synth" else f.select_drum_slot)(int(sound["slot"])); time.sleep(0.4)
            _stage(job, f"{'recording' if to_tape else 'playing'} part {part}")
            try:
                res = play_score(f, score, record=record, latency_ms=cfg.latency_ms, set_tempo=not to_tape, cancel=_cancel(job))
            finally:
                if to_tape:
                    f.cc("tape_stop", 127)
        _stage(job, "analysing")
        result: dict[str, Any] = {"played": True, "part": part, "seconds": round(res.seconds, 2), "guard": guard, "on_tape": to_tape}
        if record and res.recording is not None:
            tid = new_id(name or f"arr-{part}")
            d = _sess_dir(cfg)
            wav = save_take(res, os.path.join(d, "takes", tid + ".wav"))
            a = res.analysis
            if sound.get("kind") == "drum":
                a = _drum_analysis(res, score)
            write_json(os.path.join(d, "takes", tid + ".json"), {"id": tid, "part": part, "score": score.model_dump(), "analysis": a, "wav": wav, "kind": "tape" if to_tape else "take"})
            result.update({"take_id": tid, "wav": wav, "analysis": {k: v for k, v in a.items() if k != "notes"}})
        return result
    return _run_or_job("play_arrangement_part", name or f"arr-{part}", _expected(score), wait, work)




# ------------------------------------------------------------------ effects

from . import effects as FX


def _slot_fx(kind: str, slot: int) -> dict[str, Any]:
    src = _slot_file(kind, slot)
    if not src:
        return {"fx_type": None, "source": "no file for this slot in the backups"}
    pf = P.read_preset(src)
    m = pf.meta
    t = m.get("fx_type")
    info = FX.EFFECTS.get(t, {})
    vals = (m.get("fx_params") or [])[:4]
    return {"fx_type": t, "active": bool(m.get("fx_active")), "knobs": {k: v for k, v in zip(info.get("knobs", ["1", "2", "3", "4"]), vals)},
            "character": info.get("character"), "source": f"saved file {os.path.basename(src)}; live edits and hand changes are not reflected"}


@mcp.tool(annotations=RO)
def list_effects() -> dict[str, Any]:
    """The Field's ten effects with what they do, their four knob names, and what they suit."""
    return FX.EFFECTS


@mcp.tool(annotations=RO)
def slot_effects(kind: str, slot: int) -> dict[str, Any]:
    """Which effect a slot's saved preset carries, whether it is active, and its knob values (from the latest disk backup)."""
    return _slot_fx(kind, slot)


@mcp.tool(annotations=RW)
def set_effect(knobs: dict[str, int], fx_type: str | None = None, kind: str = "synth", slot: int | None = None) -> dict[str, Any]:
    """Turn the loaded patch's effect by knob name, values 0-127: e.g. {"feedback": 90, "mix": 64} on a delay, {"bits": 30, "rate": 40} on terminal. On the fazer, "rate_hz" takes the speed as a sweep rate in Hz (0.15 to 69) instead of a CC value. Give fx_type when the slot file is stale or unknown; otherwise it is read from the slot's saved file (slot defaults to the last selected). Knob positions "1"-"4" always work. Values stay on the patch until the human reverts it."""
    cfg = _cfg(); cfg.check("sound_design")
    st = read_state()
    if slot is None:
        slot = st.get("synth_slot") if kind == "synth" else st.get("drum_slot")
    t = fx_type or (_slot_fx(kind, int(slot))["fx_type"] if slot else None)
    if not t:
        raise ValueError("cannot tell which effect is loaded; pass fx_type (one of " + ", ".join(sorted(FX.EFFECTS)) + ") or use knob positions with fx_type set")
    sends = {}
    for name, value in knobs.items():
        if t == "fazer" and str(name).strip().lower() in ("rate_hz", "speed_hz", "hz"):
            sends[FX.PATCH_KNOBS[FX.knob_index(t, "speed")]] = FX.fazer_rate_cc(float(value))
            continue
        i = FX.knob_index(t, str(name))
        sends[FX.PATCH_KNOBS[i]] = int(max(0, min(127, int(value))))
    with _device(cfg) as f:
        for cc_name, v in sends.items():
            f.cc(cc_name, v)
    out = {"effect": t, "set": {FX.EFFECTS[t]["knobs"][int(k[-1]) - 1]: v for k, v in sends.items()}, "note": "live edit on the loaded patch; it stays until the human reverts the preset (a program change does not reload it)"}
    if t == "fazer" and "fx3" in sends:
        out["rate_hz"] = FX.fazer_rate_hz(sends["fx3"])
    return out


@mcp.tool(annotations=RW)
def set_master_effect(knobs: dict[str, int], fx_type: str | None = None) -> dict[str, Any]:
    """Turn the master bus effect by knob name (values 0-127). The master effect type is chosen by hand in the mixer; record it with set_device_state(master_fx=...) or pass fx_type here."""
    cfg = _cfg(); cfg.check("master")
    t = fx_type or read_state().get("master_fx")
    if not t:
        raise ValueError("the master effect type is unknown; ask the human which effect the mixer's T3 page shows, then set_device_state(master_fx=that) or pass fx_type")
    sends = {}
    for name, value in knobs.items():
        i = FX.knob_index(t, str(name))
        sends[FX.MASTER_KNOBS[i]] = int(max(0, min(127, int(value))))
    with _device(cfg) as f:
        for cc_name, v in sends.items():
            f.cc(cc_name, v)
    return {"master_effect": t, "set": {FX.EFFECTS[t]["knobs"][int(k[-1]) - 1]: v for k, v in sends.items()}}


# ------------------------------------------------------------------ resources and prompts

@mcp.resource("opbridge://guide/{topic}")
def guide_resource(topic: str) -> str:
    """The same reference text as get_guide, as a resource."""
    return knowledge.guide(topic)


@mcp.prompt()
def make_a_track(brief: str = "a short piece") -> str:
    """Starting prompt for composing on the Field through this bridge."""
    return (f"Make {brief} on the OP-1 field using the op-bridge tools. Call get_status, then get_guide('workflow'). "
            "Use a seed if one exists. Choose or author sounds, audition them, write one score per part, listen to each take's "
            "analysis, and commit parts to tape with the human's help. Finish with backup_tape.")


def run(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765, path_secret: str | None = None, public: bool = False) -> None:
    if transport == "stdio":
        _server.run(transport="stdio")
        return
    from mcp.server.transport_security import TransportSecuritySettings
    path = f"/{path_secret}/mcp" if path_secret else "/mcp"
    sec = TransportSecuritySettings(enable_dns_rebinding_protection=not public)
    _server.run(transport="streamable-http", host=host, port=port, streamable_http_path=path, transport_security=sec)
