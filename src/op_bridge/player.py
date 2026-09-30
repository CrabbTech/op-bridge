"""Plays compiled events on the Field with tight timing, optionally recording a take."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import analysis as an
from .device import Field, Recorder, Recording
from .jobs import check as _check_cancel
from .score import Event, Score, compile_events, midi_to_name


def _sleep_until(t_target: float, cancel=None) -> None:
    """Sleep until a perf_counter time, spinning the last two milliseconds; a cancel flag is polled every 0.2 s."""
    while True:
        now = time.perf_counter()
        d = t_target - now
        if d <= 0:
            return
        if d > 0.25:
            time.sleep(0.2)
            _check_cancel(cancel)
            continue
        if d > 0.003:
            time.sleep(d - 0.002)
        # spin the last two milliseconds


def play_events(field: Field, events: list[Event], t0: float | None = None, on_progress=None, cancel=None) -> float:
    """Send events relative to t0 (perf_counter). Returns the perf_counter time of the last event.
    With a cancel event, stops between events by raising jobs.Cancelled."""
    if t0 is None:
        t0 = time.perf_counter() + 0.05
    last = t0
    n = len(events)
    for i, e in enumerate(events):
        _sleep_until(t0 + e.t, cancel)
        _check_cancel(cancel)
        if e.kind == "note_on":
            field.note_on(e.a, e.b, e.channel)
        elif e.kind == "note_off":
            field.note_off(e.a, e.channel)
        elif e.kind == "cc":
            field.cc(e.a, e.b, e.channel)
        elif e.kind == "bend":
            field.pitch_bend(e.a, e.channel)
        elif e.kind == "clock":
            field.clock()
        last = t0 + e.t
        if on_progress and i % 200 == 0:
            on_progress(i, n)
    return last


@dataclass
class TakeResult:
    recording: Recording | None
    live_pair: tuple[int, int] | None      # 1-based USB channels
    t_score_start_sample: int | None       # sample index of beat 0 in the recording
    events: int
    seconds: float
    analysis: dict[str, Any] = field(default_factory=dict)


def play_score(field: Field, score: Score, record: bool = True, latency_ms: float = 8.0, pre_roll: float = 0.5, playback: np.ndarray | None = None, playback_gain: float = 1.0, set_tempo: bool = True,
               cancel=None) -> TakeResult:
    """Play a score (and stream `playback` into the Field's USB input when given), recording the take.
    `cancel` is a threading.Event from a background job: when set, playback stops between events with
    jobs.Cancelled and the recorder is closed; the caller releases any held notes."""
    events = compile_events(score)
    total = score.seconds()
    if score.send_clock and set_tempo:
        # The Field keeps its own song BPM and follows external clock by scaling the tape speed.
        # Set its BPM to the score tempo first so the tape runs at 100 percent while clock is sent.
        from .tape import tempo_cc_value
        field.cc("tempo", tempo_cc_value(score.tempo))
        time.sleep(0.05)
    if not record:
        t0 = time.perf_counter() + 0.05
        play_events(field, events, t0, cancel=cancel)
        _sleep_until(t0 + total, cancel)
        return TakeResult(None, None, None, len(events), total)
    rec = field.duplex_recorder(playback, gain=playback_gain) if playback is not None else field.recorder()
    rec.start()
    try:
        time.sleep(pre_roll)
        start_mark = rec.mark("score_start")
        t0 = time.perf_counter() + 0.02
        # the mark and t0 differ by ~20 ms; fold that into the mark
        start_mark += int(0.02 * rec.samplerate)
        play_events(field, events, t0, cancel=cancel)
        _sleep_until(t0 + total, cancel)
    finally:
        r = rec.stop()
    pair = detect_live_pair(r.audio)
    res = TakeResult(r, pair, start_mark, len(events), total)
    res.analysis = analyze_take(r, start_mark, score, pair, latency_ms)
    return res


def detect_live_pair(audio: np.ndarray, prefer_track: bool = True) -> tuple[int, int] | None:
    """The stereo pair carrying the live sound. In 10-channel mode the main mix sits on 1-2 and the
    tape tracks follow; with prefer_track the dry track pair wins over the main mix when it has signal."""
    if audio.size == 0:
        return None
    lv = [an.rms_db(audio[:, c]) for c in range(audio.shape[1])]
    pairs = [(lv[i] + lv[i + 1], i) for i in range(0, audio.shape[1] - 1, 2)]
    if prefer_track and audio.shape[1] >= 10:
        tracks = [p for p in pairs if p[1] >= 2]
        best = max(tracks)
        if best[0] > -200:
            return (best[1] + 1, best[1] + 2)
    best = max(pairs)
    if best[0] < -200:
        return None
    return (best[1] + 1, best[1] + 2)


def analyze_take(r: Recording, start: int, score: Score, pair: tuple[int, int] | None, latency_ms: float) -> dict[str, Any]:
    sr = r.samplerate
    if pair is None:
        return {"silent": True, "hint": "no channel carried audio; is a tape track selected and the Field in synth or drum mode?"}
    x = r.audio[:, pair[0] - 1: pair[1]].mean(axis=1)
    lat = int(latency_ms / 1000.0 * sr)
    spb = score.spb
    notes_report = []
    hits = 0
    for n in sorted(score.notes, key=lambda n: n.start)[:64]:
        s = start + int(n.start * spb * sr) + lat
        e = s + int(min(n.duration * spb, 1.5) * sr)
        seg = x[s: e]
        if len(seg) < 1024:
            continue
        prom = an.note_prominence(seg, sr, [n.midi], harmonics=2)[n.midi]
        offset_ms = an.onset_offset_ms(x, sr, s)
        ok = prom >= 8.0
        hits += int(ok)
        notes_report.append({"beat": n.start, "pitch": midi_to_name(n.midi), "heard": ok, "prominence_db": round(prom, 1), "onset_offset_ms": offset_ms})
    body = x[start: start + int(score.seconds() * sr)]
    t, cent = an.spectral_centroid_series(body, sr) if len(body) > 4096 else (np.zeros(0), np.zeros(0))
    clip = float(np.mean(np.abs(body) >= 0.999)) if len(body) else 0.0
    main = None
    if r.audio.shape[1] >= 10:
        m = r.audio[start: start + int(score.seconds() * sr), 0:2]
        main = {"peak_db": round(an.peak_db(m), 1), "rms_db": round(an.rms_db(m), 1), "clipping_fraction": round(float(np.mean(np.abs(m) >= 0.999)), 5) if len(m) else 0.0}
    return {
        "live_usb_channels": list(pair),
        "main_mix": main,
        "peak_db": round(an.peak_db(body), 1),
        "rms_db": round(an.rms_db(body), 1),
        "clipping_fraction": round(clip, 5),
        "spectral_centroid_hz": {"median": round(float(np.median(cent))) if len(cent) else None, "min": round(float(cent.min())) if len(cent) else None, "max": round(float(cent.max())) if len(cent) else None},
        "notes_checked": len(notes_report),
        "notes_heard": hits,
        "notes": notes_report,
    }


def save_take(res: TakeResult, path_wav: str, stereo_only: bool = True) -> str:
    import soundfile as sf
    r = res.recording
    assert r is not None
    if stereo_only and res.live_pair:
        data = r.audio[:, res.live_pair[0] - 1: res.live_pair[1]]
    else:
        data = r.audio
    sf.write(path_wav, data, r.samplerate, subtype="FLOAT")
    return path_wav
