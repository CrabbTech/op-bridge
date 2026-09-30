"""Configuration, modes, constraints and on-disk sessions.

Everything lives under OP_BRIDGE_HOME (default ~/Music/op-bridge):

    config.json                 mode, constraints, midi channel, latency, palette
    sessions/<name>/takes/      recorded takes: <id>.wav (live pair), <id>.json (score + analysis)
    sessions/<name>/seeds/      captured seeds: <id>.json, <id>.wav
    sessions/<name>/samples/    regions cut from takes by sample_from_take: <name>.wav (listed in get_status)
    sessions/<name>/backups/    tape stem backups
    staging/synth/<n>.aif       presets authored by the model, waiting for disk mode
    staging/drum/<n>.aif
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any

HOME = os.environ.get("OP_BRIDGE_HOME", os.path.expanduser("~/Music/op-bridge"))

MODES = ("freeform", "guided")

SCALES: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "chromatic": list(range(12)),
}
PITCH_CLASSES = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}


def parse_key(key: str) -> tuple[int, str]:
    """'D minor', 'F#m', 'Bb major', 'A dorian' -> (root pitch class, scale name)"""
    s = key.strip()
    m = re.match(r"^([A-Ga-g])([#b]?)\s*(.*)$", s)
    if not m:
        raise ValueError(f"cannot parse key {key!r}")
    root = PITCH_CLASSES[(m.group(1) + m.group(2)).upper()]
    rest = m.group(3).strip().lower().replace(" ", "_")
    if rest in ("", "maj", "major"):
        scale = "major"
    elif rest in ("m", "min", "minor"):
        scale = "minor"
    elif rest in SCALES:
        scale = rest
    else:
        raise ValueError(f"unknown scale {rest!r}; known: {sorted(SCALES)}")
    return root, scale


def allowed_pitch_classes(key: str) -> set[int]:
    root, scale = parse_key(key)
    return {(root + i) % 12 for i in SCALES[scale]}


@dataclass
class Constraints:
    """What the model may do. In guided mode every field is enforced. Absent means unrestricted."""
    key: str | None = None                      # e.g. "D minor": notes must belong to it
    extra_pitch_classes: list[int] = field(default_factory=list)  # allowed on top of the key (e.g. seed notes)
    tempo: float | None = None                  # fixed tempo
    tempo_range: list[float] | None = None      # [min, max]
    max_polyphony: int = 6
    max_take_seconds: float = 240.0
    allowed_slots: list[str] | None = None      # e.g. ["synth 1", "synth 3", "drum 2"]
    allow_sound_select: bool = True             # program change
    allow_sound_design: bool = True             # engine, envelope, FX, LFO CCs and preset authoring
    allow_master: bool = True                   # master FX, EQ, drive and compressor
    allow_mixer: bool = True                    # track level, pan, mute
    allow_transport: bool = True                # tape play/stop/jump/loop, tempo CC, metronome
    allow_tempo_change: bool = True
    allowed_cc_params: list[str] | None = None  # names from device.CC the model may automate in scores
    notes: str = ""                             # free-text instructions from the human

    def tighten(self, other: dict[str, Any]) -> list[str]:
        """Apply changes that only make the constraints stricter. Returns a list of what changed."""
        changed: list[str] = []
        for k, v in other.items():
            if k not in self.__dataclass_fields__:
                raise ValueError(f"unknown constraint {k!r}")
            cur = getattr(self, k)
            if k.startswith("allow_"):
                if cur and v is False:
                    setattr(self, k, False); changed.append(k)
            elif k == "max_polyphony":
                if v is not None and int(v) < cur:
                    setattr(self, k, int(v)); changed.append(k)
            elif k == "max_take_seconds":
                if v is not None and float(v) < cur:
                    setattr(self, k, float(v)); changed.append(k)
            elif k in ("key", "tempo"):
                if cur is None and v is not None:
                    setattr(self, k, v); changed.append(k)
            elif k == "tempo_range":
                if v is not None:
                    lo, hi = float(v[0]), float(v[1])
                    if cur is None:
                        self.tempo_range = [lo, hi]; changed.append(k)
                    else:
                        nl, nh = max(lo, cur[0]), min(hi, cur[1])
                        if [nl, nh] != cur:
                            self.tempo_range = [nl, nh]; changed.append(k)
            elif k in ("allowed_slots", "allowed_cc_params"):
                if v is not None:
                    new = list(v) if cur is None else [x for x in cur if x in v]
                    if new != cur:
                        setattr(self, k, new); changed.append(k)
            elif k == "extra_pitch_classes":
                pass  # widening; ignore in tighten
            elif k == "notes":
                if v:
                    self.notes = (self.notes + "\n" + str(v)).strip(); changed.append(k)
        return changed


@dataclass
class Config:
    mode: str = "freeform"
    constraints: Constraints = field(default_factory=Constraints)
    midi_channel: int = 1               # the Field's configured MIDI channel (device setting)
    latency_ms: float = 8.0             # measured note-on to audio onset
    default_polyphony: int = 6
    palette: dict[str, dict[str, Any]] = field(default_factory=dict)  # "synth 1": {"name":..., "engine":..., "notes":...}
    current_session: str = "default"
    judge: str = "auto"                 # auto: Jev when TYPESAFE_API_KEY is set, else local rules; local: never call out; jev: require Jev

    @classmethod
    def load(cls) -> "Config":
        path = os.path.join(HOME, "config.json")
        cfg = cls()
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            cons = raw.pop("constraints", {}) or {}
            for k, v in raw.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            cfg.constraints = Constraints(**{k: v for k, v in cons.items() if k in Constraints.__dataclass_fields__})
        if cfg.mode not in MODES:
            cfg.mode = "freeform"
        return cfg

    def save(self) -> str:
        os.makedirs(HOME, exist_ok=True)
        path = os.path.join(HOME, "config.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)
        return path

    def guided(self) -> bool:
        return self.mode == "guided"

    def check(self, capability: str) -> None:
        """Raise PermissionError when guided mode forbids a capability."""
        if not self.guided():
            return
        if not getattr(self.constraints, f"allow_{capability}", True):
            raise PermissionError(f"guided mode: {capability.replace('_', ' ')} is reserved for the human (constraint allow_{capability}=false)")

    def slot_allowed(self, kind: str, slot: int) -> bool:
        if not self.guided() or self.constraints.allowed_slots is None:
            return True
        return f"{kind} {slot}" in self.constraints.allowed_slots


# --------------------------------------------------------------------------- sessions

def session_dir(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-]+", "-", name).strip("-") or "default"
    d = os.path.join(HOME, "sessions", safe)
    for sub in ("takes", "seeds", "samples", "backups", "mixes"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d


def list_sessions() -> list[str]:
    root = os.path.join(HOME, "sessions")
    if not os.path.isdir(root):
        return []
    return sorted(os.listdir(root))


def new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def write_json(path: str, data: Any) -> str:
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=str)
    return path


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- what the Field is set to

STATE_PATH = os.path.join(HOME, "device-state.json")


def read_state() -> dict[str, Any]:
    """What the bridge last set on the Field (mode, slots, tempo) with timestamps. The Field cannot be asked, so
    anything the human changed by hand is unknown until the bridge sets it again or hears it on MIDI input."""
    if os.path.exists(STATE_PATH):
        try:
            return json.load(open(STATE_PATH))
        except Exception:
            return {}
    return {}


def update_state(**fields: Any) -> dict[str, Any]:
    st = read_state()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for k, v in fields.items():
        st[k] = v
        st[k + "_at"] = now
    os.makedirs(HOME, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE_PATH)
    return st


# --------------------------------------------------------------------------- secrets

SECRETS_PATH = os.path.join(HOME, "secrets.env")


def load_secrets() -> list[str]:
    """Load KEY=VALUE lines from the owner-only secrets file into the environment (never logged). Returns the key names."""
    names: list[str] = []
    if not os.path.exists(SECRETS_PATH):
        return names
    try:
        with open(SECRETS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip(); v = v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
                    names.append(k)
    except Exception:
        pass
    return names


load_secrets()
