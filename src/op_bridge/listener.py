"""Audio-language observations. Offline by default; explicit Gemini requests use Google."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .jobs import check
from . import gemini

MODEL_ID = "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
MODEL_REVISION = "c65570002626f41b4dc08b7b54f42f99f3e82e7f"
DEFAULT_QUESTION = "Describe the sounds in this audio."
MAX_LINE = 65536


def root(home):
    return Path(os.environ.get("OP_BRIDGE_LISTENER_HOME", str(Path(home) / "listener"))).expanduser().resolve()


def status(home, backend="local"):
    if backend == "gemini":
        return gemini.status(home)
    if backend != "local":
        raise ValueError("backend must be 'local' or 'gemini'")
    d = root(home)
    try:
        manifest = json.loads((d / "runtime.json").read_text())
        if manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
            raise ValueError("installation pins do not match")
        if not (d / ".venv/bin/python").is_file() or not (d / "adapter.py").is_file() or not (d / "frontend.py").is_file():
            raise ValueError("isolated Python, adapter or audio frontend is missing")
        files = manifest.get("model_files", [])
        if not files or not any(x.get("name", "").endswith(".safetensors") for x in files):
            raise ValueError("model inventory is missing")
        for entry in files:
            p = d / "model" / entry["name"]
            if Path(entry["name"]).name != entry["name"] or not p.is_file() or p.stat().st_size != entry["size_bytes"]:
                raise ValueError("a model file is missing or incomplete")
        return {"available": True, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "local_only": True, "max_excerpt_seconds": 30, "runtime_root": str(d),
                "note": "Coarse descriptions only. Calibration found invented instruments in layered audio; not validated for harmony or mix quality."}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return {"available": False, "local_only": True, "runtime_root": str(d), "reason": str(error),
                "setup": "Run python3 adapters/qwen_audio/setup.py explicitly; inference never downloads models."}


def validate_options(start_s, seconds, question, max_tokens, timeout_seconds, backend="local", model=None):
    if backend not in ("local", "gemini"):
        raise ValueError("backend must be 'local' or 'gemini'")
    if backend == "local" and model is not None:
        raise ValueError("model is only selectable with backend='gemini'")
    if backend == "gemini" and model is not None and model not in gemini.MODELS:
        raise ValueError("unsupported Gemini model; see listener_status(backend='gemini')")
    low, high = (64, 768) if backend == "local" else (256, 4096)
    if max_tokens is None:
        max_tokens = 256 if backend == "local" else 2048
    if not math.isfinite(start_s) or start_s < 0:
        raise ValueError("start_s must be finite and nonnegative")
    if not math.isfinite(seconds) or not 1 <= seconds <= 30:
        raise ValueError("seconds must be between 1 and 30")
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1500:
        raise ValueError("question must contain 1–1500 characters")
    if type(max_tokens) is not int or not low <= max_tokens <= high:
        raise ValueError(f"max_tokens must be an integer between {low} and {high} for {backend}")
    if not math.isfinite(timeout_seconds) or not 10 <= timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be between 10 and 600")
    return max_tokens, (model or gemini.DEFAULT_MODEL) if backend == "gemini" else None


def prepare_excerpt(source, destination, start_s, seconds, channels=None, *, pcm16=False):
    source = Path(source).expanduser().resolve(strict=True)
    with sf.SoundFile(source) as stream:
        sr = stream.samplerate
        if not 8000 <= sr <= 192000 or not 1 <= stream.channels <= 32:
            raise ValueError("unsupported source sample rate or channel count")
        selected = channels if channels is not None else list(range(1, min(2, stream.channels) + 1))
        if not selected or len(selected) > 2 or len(set(selected)) != len(selected) or any(type(c) is not int or not 1 <= c <= stream.channels for c in selected):
            raise ValueError("channels must select one or two distinct 1-based source channels")
        begin = round(start_s * sr)
        if begin >= len(stream):
            raise ValueError("start_s is past the end of the recording")
        stream.seek(begin)
        samples = stream.read(min(round(seconds * sr), len(stream) - begin), dtype="float32", always_2d=True)
    if len(samples) < sr:
        raise ValueError("at least one second of source audio is required")
    samples = samples[:, [c - 1 for c in selected]]
    if not np.isfinite(samples).all():
        raise ValueError("source contains non-finite samples")
    mono = samples.mean(axis=1)
    notes = []
    if samples.shape[1] == 2 and np.mean(mono ** 2) < 0.05 * np.mean(samples ** 2):
        mono = samples[:, int(np.argmax(np.mean(samples ** 2, axis=0)))]
        notes.append("Stereo cancellation detected; the louder selected channel was used for model input.")
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples ** 2)))
    factor = math.gcd(sr, 16000)
    out = resample_poly(mono, 16000 // factor, sr // factor).astype(np.float32)
    gain = 1.0
    if pcm16 and np.max(np.abs(out)) > 1 - 1/32768:
        gain = float((1 - 1/32768) / np.max(np.abs(out)))
        out *= gain
        notes.append("Model input was attenuated to avoid PCM clipping; source measurements are unchanged.")
    sf.write(destination, out, 16000, subtype="PCM_16" if pcm16 else "FLOAT")
    return {"source": str(source), "start_s": begin / sr, "end_s": (begin + len(samples)) / sr,
            "seconds": len(samples) / sr, "source_sample_rate": sr, "channels": selected,
            "model_input_sample_rate": 16000, "model_input_channels": 1,
            "model_input_subtype": "PCM_16" if pcm16 else "FLOAT", "model_input_gain": gain,
            "excerpt_sha256": hashlib.sha256(Path(destination).read_bytes()).hexdigest(),
            "measured": {"sample_peak_dbfs": round(20 * math.log10(max(peak, 1e-6)), 2),
                         "rms_dbfs": round(20 * math.log10(max(rms, 1e-6)), 2),
                         "clipped_sample_fraction": float(np.mean(np.abs(samples) >= 1.0))},
            "silent": bool(np.max(np.abs(out)) < 1e-6), "notes": notes}


@contextlib.contextmanager
def inference_lock(d, cancel, deadline):
    import fcntl
    with (d / "inference.lock").open("a") as lock:
        while True:
            check(cancel)
            if time.monotonic() >= deadline:
                raise RuntimeError("listener timed out waiting for another analysis")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_adapter(d, excerpt, question, max_tokens, log, cancel, deadline, progress):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", OP_BRIDGE_LISTENER_PARENT_PID=str(os.getpid()))
    command = [str(d / ".venv/bin/python"), str(d / "adapter.py"), "--runtime-root", str(d)]
    result = run_process(command, {"audio_path": str(excerpt), "question": question, "max_tokens": max_tokens},
                         excerpt.parent, log, cancel, deadline, progress, env)
    if result.get("model_id") != MODEL_ID or result.get("model_revision") != MODEL_REVISION:
        raise RuntimeError("listener returned a different model than the installed one")
    if not isinstance(result.get("observations"), str) or not result["observations"].strip():
        raise RuntimeError("listener returned no observations")
    return result


def run_process(command, request, cwd, log, cancel, deadline, progress, env):
    check(cancel)
    messages = queue.Queue(maxsize=32)
    overflow = threading.Event()
    with log.open("wb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                   env=env, cwd=cwd, start_new_session=True)
        def drain():
            try:
                while line := process.stdout.readline(MAX_LINE + 1):
                    if len(line) > MAX_LINE:
                        overflow.set(); break
                    try:
                        messages.put_nowait(line)
                    except queue.Full:
                        overflow.set(); break
            finally:
                process.stdout.close()
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            process.stdin.write(json.dumps(request).encode())
            process.stdin.close()
            result = None
            while process.poll() is None or reader.is_alive() or not messages.empty():
                check(cancel)
                if time.monotonic() >= deadline:
                    raise RuntimeError("audio analysis timed out")
                if overflow.is_set():
                    raise RuntimeError("listener exceeded its response limit")
                try:
                    line = messages.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError) as error:
                    raise RuntimeError("invalid listener response") from error
                if not isinstance(event, dict):
                    raise RuntimeError("invalid listener response")
                if event.get("type") == "progress":
                    progress(str(event.get("message", "Listening"))[:200])
                elif event.get("type") == "error":
                    raise RuntimeError(str(event.get("message", "listener failed"))[:1000])
                elif event.get("type") == "result" and result is None:
                    result = {k: v for k, v in event.items() if k != "type"}
                else:
                    raise RuntimeError("unexpected or duplicate listener response")
            if overflow.is_set() or process.returncode != 0 or not result:
                raise RuntimeError(f"listener failed (exit {process.returncode}); see {log}")
            return result
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=3)
            reader.join(timeout=1)


def listen(home, source, report_dir, *, start_s=0.0, seconds=15.0, question=DEFAULT_QUESTION,
           channels=None, max_tokens=None, timeout_seconds=300.0, cancel=None, progress=lambda s: None,
           backend="local", model=None):
    max_tokens, model = validate_options(start_s, seconds, question, max_tokens, timeout_seconds, backend, model)
    check(cancel)
    if backend == "local":
        found = status(home)
        if not found["available"]:
            raise RuntimeError(f"local listener is unavailable: {found['reason']}. {found['setup']}")
    else:
        gemini.api_key(home)
    directory = Path(report_dir) / ("listen-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    excerpt = directory / "excerpt.wav"
    evidence = prepare_excerpt(source, excerpt, start_s, seconds, channels, pcm16=backend == "gemini")
    check(cancel)
    started = time.monotonic()
    deadline = started + timeout_seconds
    if evidence["silent"]:
        result = {"observations": "The selected excerpt is digitally silent or below -120 dBFS.",
                  "assessment": "Measured silence; the audio model was not invoked.", "model_invoked": False,
                  "usable": True}
    elif backend == "local":
        d = root(home)
        progress("Waiting for the local audio listener")
        with inference_lock(d, cancel, deadline):
            result = run_adapter(d, excerpt, question.strip(), max_tokens, directory / "runtime.log", cancel, deadline, progress)
        result["model_invoked"] = True
    else:
        progress(f"Sending the selected excerpt to Google ({model})")
        request = {"home": str(home), "audio_path": str(excerpt), "question": question.strip(),
                   "model": model, "max_tokens": max_tokens, "timeout_seconds": timeout_seconds}
        result = run_process([sys.executable, str(Path(gemini.__file__).resolve())], request,
                             directory, directory / "runtime.log", cancel, deadline, progress,
                             dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        if result.get("model_id") != model or not isinstance(result.get("observations"), str):
            raise RuntimeError("Gemini adapter returned an invalid result")
        result.update(model_invoked=True, cloud_upload=True)
    check(cancel)
    result.update(backend=backend, question=question.strip(), max_tokens=max_tokens,
                  cloud_upload=bool(result.get("cloud_upload", False)),
                  evidence=evidence, elapsed_seconds=round(time.monotonic() - started, 3))
    report = directory / "review.json"
    result["report"] = str(report)
    report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
