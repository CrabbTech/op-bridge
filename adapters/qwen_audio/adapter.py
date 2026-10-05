#!/usr/bin/env python3
"""One offline Qwen2-Audio request; the bridge owns excerpt selection and cancellation."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

MODEL_ID = "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
MODEL_REVISION = "c65570002626f41b4dc08b7b54f42f99f3e82e7f"
RUNTIME_REVISION = "ee0c65d12d0be748aed5fb8e1587dc7cc8f9e644"
PROMPT_VERSION = "op-listen-neutral-v2"
PROTOCOL_STREAM = sys.stdout


def emit(kind, **values):
    print(json.dumps({"type": kind, **values}, allow_nan=False), file=PROTOCOL_STREAM, flush=True)


def offline():
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    def denied(*args, **kwargs):
        raise RuntimeError("The audio listener is offline; network access is disabled.")
    socket.create_connection = denied
    socket.getaddrinfo = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied


def watch_parent():
    parent = int(os.environ.get("OP_BRIDGE_LISTENER_PARENT_PID", os.getppid()))
    if parent <= 1 or os.getppid() != parent:
        raise RuntimeError("The listener's parent process is no longer available.")
    def watch():
        while os.getppid() == parent:
            time.sleep(0.25)
        os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {"audio_path", "question", "max_tokens"}:
        raise ValueError("Expected audio_path, question and max_tokens.")
    path = Path(request["audio_path"])
    if not path.is_absolute() or not path.is_file() or path.suffix.lower() != ".wav":
        raise ValueError("audio_path must be an existing absolute WAV path.")
    question = request["question"]
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1500:
        raise ValueError("question must contain 1–1500 characters.")
    budget = request["max_tokens"]
    if type(budget) is not int or not 64 <= budget <= 768:
        raise ValueError("max_tokens must be an integer between 64 and 768.")
    return path, question.strip(), budget


def infer(root, request):
    import numpy as np
    import soundfile as sf
    path, question, budget = validate_request(request)
    info = sf.info(path)
    if info.samplerate != 16000 or info.channels != 1 or not 1 <= info.duration <= 30:
        raise ValueError("The listener expects 1–30 seconds of 16 kHz mono WAV.")
    samples, _ = sf.read(path, dtype="float32")
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-6:
        raise ValueError("The listener needs finite, non-silent audio.")
    manifest = json.loads((root / "runtime.json").read_text())
    if manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("Listener installation pins do not match the adapter.")
    for entry in manifest["model_files"]:
        p = root / "model" / entry["name"]
        if Path(entry["name"]).name != entry["name"] or not p.is_file() or p.stat().st_size != entry["size_bytes"]:
            raise ValueError("A local model file is missing or incomplete.")
    for package, expected in manifest["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Listener runtime changed: {package}.")
    import mlx.core as mx
    from mlx_audio.stt.utils import load_model
    from mlx_audio.stt.models.qwen2_audio.qwen2_audio import Model
    from transformers import AutoTokenizer
    from frontend import install_frontend, FRONTEND_VERSION

    # The installed Qwen implementation only uses processor.tokenizer. Avoid its
    # AutoProcessor trust_remote_code=True hook and load the built-in tokenizer.
    def local_hook(cls, model, model_path):
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False, local_files_only=True)
        model._processor = SimpleNamespace(tokenizer=tokenizer)
        enc = model.audio_tower
        enc._embed_positions = enc._embed_positions.astype(enc.conv1.weight.dtype)
        mx.eval(enc._embed_positions)
        return model

    emit("progress", message="Loading local Qwen2-Audio")
    started = time.monotonic()
    with patch.object(Model, "post_load_hook", classmethod(local_hook)):
        model = load_model(root / "model", strict=True)
    install_frontend(model, root / "model")
    load_seconds = time.monotonic() - started
    mx.random.seed(0)
    emit("progress", message="Analyzing the recorded audio")
    # A long rubric or a menu of possible sounds primed false instrument claims
    # in the controls. Pass a short neutral question without additional examples.
    result = model.generate(samples, prompt=question, max_tokens=budget, temperature=0.0, verbose=False)
    text = result.text.strip()
    if not text or len(text) > 12000:
        raise RuntimeError("The audio model returned no usable observation.")
    if result.generation_tokens >= budget:
        raise RuntimeError("The audio model reached its token limit; retry with a shorter question or a larger max_tokens.")
    return {"observations": text, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "runtime_revision": RUNTIME_REVISION, "frontend_version": FRONTEND_VERSION,
            "prompt_version": PROMPT_VERSION,
            "question": question, "temperature": 0.0, "seed": 0,
            "generation_tokens": result.generation_tokens, "load_seconds": round(load_seconds, 3),
            "inference_seconds": round(result.total_time, 3), "peak_memory_bytes": int(mx.get_peak_memory()),
            "assessment": "Model observations, not verified musical facts."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    offline()
    watch_parent()
    try:
        data = sys.stdin.buffer.read(65537)
        if len(data) > 65536:
            raise ValueError("Listener request is too large.")
        request = json.loads(data)
        # ML libraries must not corrupt the NDJSON response stream.
        with contextlib.redirect_stdout(sys.stderr):
            result = infer(args.runtime_root.resolve(), request)
        emit("result", **result)
    except Exception as error:
        emit("error", message=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
