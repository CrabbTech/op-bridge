#!/usr/bin/env python3
"""Explicit installation only. Inference never installs or downloads anything."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

MODEL_ID = "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
MODEL_REVISION = "c65570002626f41b4dc08b7b54f42f99f3e82e7f"
RUNTIME_REVISION = "ee0c65d12d0be748aed5fb8e1587dc7cc8f9e644"
MODEL_FILES = ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt", "weights.safetensors", "README.md"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    home = Path(os.environ.get("OP_BRIDGE_HOME", "~/Music/op-bridge")).expanduser()
    parser.add_argument("--runtime-root", type=Path, default=Path(os.environ.get("OP_BRIDGE_LISTENER_HOME", home / "listener")))
    parser.add_argument("--python", default="3.11", help="Python 3.11 interpreter or uv version selector")
    parser.add_argument("--uv", default=shutil.which("uv"))
    parser.add_argument("--hf", default=shutil.which("hf"))
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("This pinned MLX runtime requires Apple Silicon macOS.")
    if not args.uv or not args.hf:
        parser.error("Install uv and the Hugging Face hf CLI first, or pass --uv and --hf.")
    root = args.runtime_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous = root / "runtime.json"
    if previous.exists():
        manifest = json.loads(previous.read_text())
        if manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
            parser.error("This directory belongs to a different listener. Choose a new --runtime-root.")
    source = Path(__file__).resolve().parent
    python = root / ".venv/bin/python"
    if not python.exists():
        subprocess.run([args.uv, "venv", "--python", args.python, str(root / ".venv")], check=True)
    subprocess.run([args.uv, "pip", "sync", "--python", str(python), str(source / "requirements-macos.lock.txt")], check=True)
    subprocess.run([args.hf, "download", MODEL_ID, *MODEL_FILES, "--revision", MODEL_REVISION,
                    "--local-dir", str(root / "model"), "--max-workers", "2"], check=True)
    print("Checking the downloaded files and recording runtime provenance...", flush=True)
    inventory = [{"name": name, "size_bytes": (root / "model" / name).stat().st_size,
                  "sha256": digest(root / "model" / name)} for name in MODEL_FILES]
    packages = json.loads(subprocess.check_output([str(python), "-c",
        "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"], text=True))
    shutil.copyfile(source / "adapter.py", root / "adapter.py")
    shutil.copyfile(source / "frontend.py", root / "frontend.py")
    shutil.copyfile(source / "requirements-macos.lock.txt", root / "requirements-macos.lock.txt")
    manifest = {"schema_version": 1, "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "runtime_revision": RUNTIME_REVISION,
                "model_files": inventory, "packages": packages,
                "adapter_sha256": digest(root / "adapter.py"),
                "frontend_sha256": digest(root / "frontend.py"),
                "lock_sha256": digest(root / "requirements-macos.lock.txt")}
    temporary = root / "runtime.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(previous)
    print(json.dumps({"installed": True, "runtime_root": str(root), "model_id": MODEL_ID,
                      "model_bytes": sum(f["size_bytes"] for f in inventory)}))


if __name__ == "__main__":
    main()
