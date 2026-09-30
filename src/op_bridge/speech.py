"""Local speech for the Field's vocoder: macOS `say` synthesises text offline, the bridge streams the audio into the
Field over USB (the Field's input source set to usb audio) while playing carrier notes over MIDI."""
from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf

from .device import find_audio_input, refresh_audio_devices


def available_voices() -> list[dict[str, str]]:
    """Voices `say` offers, from `say -v ?`."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=20).stdout
    except Exception as e:
        raise RuntimeError(f"the macOS say command is not available: {e}")
    voices = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s{2,}([a-z]{2}_[A-Z]{2})\s+#\s*(.*)$", line.strip())
        if m:
            voices.append({"name": m.group(1).strip(), "language": m.group(2), "sample": m.group(3)})
    return voices


def synthesize(text: str, out_path: str, voice: str | None = None, rate: int | None = None) -> dict[str, Any]:
    """Text -> 16-bit 44.1 kHz mono WAV with the built-in speech engine. Returns duration and peak."""
    if not text or not text.strip():
        raise ValueError("give some text to speak")
    cmd = ["say", "-o", out_path, "--file-format=WAVE", "--data-format=LEI16@44100"]
    if voice:
        cmd += ["-v", voice]
    if rate:
        cmd += ["-r", str(int(rate))]
    cmd.append(text)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"say failed: {r.stderr.strip() or r.stdout.strip() or 'no output'}; voices: use available_voices()")
    x, sr = sf.read(out_path, dtype="float32", always_2d=True)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    return {"path": out_path, "seconds": round(len(x) / sr, 2), "samplerate": sr, "peak_dbfs": round(20 * np.log10(peak + 1e-9), 1), "voice": voice or "default"}


def find_field_output() -> dict | None:
    """The Field as an audio output device (the computer -> Field direction, 2 channels)."""
    refresh_audio_devices()
    for i, d in enumerate(sd.query_devices()):
        if "OP-1" in d["name"] and d["max_output_channels"] >= 2:
            return {"index": i, "name": d["name"], "outputs": d["max_output_channels"], "samplerate": d["default_samplerate"]}
    return None
