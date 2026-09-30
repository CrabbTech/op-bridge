"""Low-level access to the OP-1 field over USB MIDI and USB audio.

Everything here talks to the real device. There are no mocks: the functional tests
open the same ports this module opens.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import mido
import numpy as np
import sounddevice as sd

DEFAULT_NAME = "OP-1"


# --------------------------------------------------------------------------- discovery

def midi_output_names() -> list[str]:
    return list(mido.get_output_names())


def midi_input_names() -> list[str]:
    return list(mido.get_input_names())


def find_midi_output(hint: str = DEFAULT_NAME) -> str | None:
    for name in midi_output_names():
        if hint.lower() in name.lower():
            return name
    return None


def find_midi_input(hint: str = DEFAULT_NAME) -> str | None:
    for name in midi_input_names():
        if hint.lower() in name.lower():
            return name
    return None


_open_streams = 0


def refresh_audio_devices() -> None:
    """Re-scan CoreAudio. Needed after the Field re-enumerates (disk mode, USB mode change, replug);
    PortAudio otherwise keeps a stale device list and stream opening fails with an internal error."""
    if _open_streams == 0:
        try:
            sd._terminate(); sd._initialize()
        except Exception:
            pass


def audio_devices() -> list[dict]:
    refresh_audio_devices()
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append(
            {
                "index": i,
                "name": d["name"],
                "inputs": d["max_input_channels"],
                "outputs": d["max_output_channels"],
                "samplerate": d["default_samplerate"],
            }
        )
    return out


def find_audio_input(hint: str = DEFAULT_NAME) -> dict | None:
    for d in audio_devices():
        if hint.lower() in d["name"].lower() and d["inputs"] > 0:
            return d
    return None


# --------------------------------------------------------------------------- recorder

@dataclass
class Recording:
    audio: np.ndarray  # frames x channels, float32 in [-1, 1]
    samplerate: int
    marks: list[tuple[str, int]] = field(default_factory=list)  # (label, sample index)

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.samplerate

    def mark_sample(self, label: str) -> int | None:
        for name, idx in self.marks:
            if name == label:
                return idx
        return None


class Recorder:
    """Records all channels of an input device into memory.

    While recording, `now()` returns the stream clock and `mark(label)` stores the
    sample index that corresponds to this instant, so MIDI events sent by the caller
    can be lined up with the audio to within one block.
    """

    def __init__(self, device: int, channels: int, samplerate: int = 44100, blocksize: int = 256):
        self.device = device
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._blocks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._t0_adc: float | None = None
        self._t0_perf: float | None = None
        self._use_perf = False
        self.marks: list[tuple[str, int]] = []

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass  # over/underruns are reported by sounddevice; keep recording
        with self._lock:
            if self._t0_adc is None:
                adc = float(time_info.inputBufferAdcTime) if time_info is not None else 0.0
                self._t0_perf = time.perf_counter()
                if adc > 0:
                    self._t0_adc = adc
                else:
                    self._use_perf = True
                    self._t0_adc = self._t0_perf
            self._blocks.append(indata.copy())

    def start(self) -> "Recorder":
        global _open_streams
        try:
            self._stream = sd.InputStream(device=self.device, channels=self.channels, samplerate=self.samplerate, blocksize=self.blocksize, dtype="float32", callback=self._callback)
        except sd.PortAudioError:
            # stale device list after a re-enumeration: refresh and look the device up again by name
            refresh_audio_devices()
            dev = find_audio_input()
            if dev is None:
                raise
            self.device = dev["index"]
            self._stream = sd.InputStream(device=self.device, channels=self.channels, samplerate=self.samplerate, blocksize=self.blocksize, dtype="float32", callback=self._callback)
        self._stream.start()
        _open_streams += 1
        # wait for the first block so the clock origin exists
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            with self._lock:
                if self._t0_adc is not None:
                    break
            time.sleep(0.002)
        if self._t0_adc is None:
            raise RuntimeError("audio input never delivered a block; is the OP-1 field in normal mode?")
        return self

    def now(self) -> float:
        if self._use_perf or self._stream is None:
            return time.perf_counter()
        return float(self._stream.time)

    def sample_at(self, t: float) -> int:
        assert self._t0_adc is not None
        return int(round((t - self._t0_adc) * self.samplerate))

    def mark(self, label: str) -> int:
        idx = self.sample_at(self.now())
        self.marks.append((label, idx))
        return idx

    def stop(self) -> Recording:
        global _open_streams
        assert self._stream is not None
        self._stream.stop()
        self._stream.close()
        _open_streams = max(0, _open_streams - 1)
        with self._lock:
            audio = np.concatenate(self._blocks, axis=0) if self._blocks else np.zeros((0, self.channels), np.float32)
        return Recording(audio=audio, samplerate=self.samplerate, marks=list(self.marks))

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        if self._stream is not None and self._stream.active:
            self._stream.stop()
            self._stream.close()


# --------------------------------------------------------------------------- the field

# Named parameters -> CC numbers, from TE's firmware 1.7 MIDI reference.
CC = {
    "engine1": 46, "engine2": 47, "engine3": 48, "engine4": 49,
    "attack": 50, "decay": 51, "sustain": 52, "release": 53,
    "fx1": 54, "fx2": 55, "fx3": 56, "fx4": 57,
    "lfo1": 58, "lfo2": 59, "lfo3": 60, "lfo4": 61,
    "randomize": 62, "reset": 63, "sustain_pedal": 64,
    "master_fx1": 70, "master_fx2": 71, "master_fx3": 72, "master_fx4": 73,
    "master1": 74, "master2": 75, "master3": 76, "master4": 77,
    "tape_rec_level": 78, "octave": 79, "tempo": 80, "metronome": 81,
    "tape_prev_bar": 82, "tape_next_bar": 83, "tape_start": 84, "tape_end": 85,
    "tape_loop_in": 86, "tape_loop_out": 87, "tape_loop_toggle": 88,
    "eq_low": 90, "eq_mid": 91, "eq_high": 92,
    "mode": 93, "slot_select": 102, "tape_stop": 104, "tape_play": 105,
    "all_notes_off": 123,
    "mixer_volume": 7, "mixer_mute": 9, "mixer_pan": 10,
    "midi_lfo1": 1, "midi_lfo2": 2, "midi_lfo3": 3, "midi_lfo4": 4,
}


class Field:
    """One connected OP-1 field: a MIDI output, an optional MIDI input and its USB audio input."""

    def __init__(self, midi_out: str | None = None, midi_in: str | None = None, audio: dict | None = None, channel: int = 0):
        self.midi_out_name = midi_out or find_midi_output()
        self.midi_in_name = midi_in or find_midi_input()
        self.audio = audio or find_audio_input()
        self.channel = channel  # mido channels are 0-based; this is MIDI channel 1
        self._out: mido.ports.BaseOutput | None = None
        self._open_notes: set[tuple[int, int]] = set()   # (channel, note) started by this connection and not yet released

    # ---- lifecycle
    @property
    def connected(self) -> bool:
        return self.midi_out_name is not None

    def open(self) -> "Field":
        if self.midi_out_name is None:
            raise RuntimeError("no OP-1 field MIDI output found; connect it over USB-C and leave it in normal mode")
        self._out = mido.open_output(self.midi_out_name)
        return self

    def close(self) -> None:
        """Release only the notes this connection started; other connections (a tool called mid-performance) keep theirs."""
        if self._out is not None:
            try:
                for ch, note in list(self._open_notes):
                    self.send(mido.Message("note_off", note=note, velocity=0, channel=ch))
                self._open_notes.clear()
            finally:
                self._out.close()
                self._out = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ---- messages
    def send(self, msg: mido.Message) -> None:
        assert self._out is not None, "Field is not open"
        self._out.send(msg)

    def note_on(self, note: int, velocity: int = 100, channel: int | None = None) -> None:
        ch = self.channel if channel is None else channel
        self.send(mido.Message("note_on", note=note, velocity=velocity, channel=ch))
        self._open_notes.add((ch, note))

    def note_off(self, note: int, channel: int | None = None) -> None:
        ch = self.channel if channel is None else channel
        self.send(mido.Message("note_off", note=note, velocity=0, channel=ch))
        self._open_notes.discard((ch, note))

    def cc(self, control: int | str, value: int, channel: int | None = None) -> None:
        if isinstance(control, str):
            control = CC[control]
        self.send(mido.Message("control_change", control=control, value=int(value), channel=self.channel if channel is None else channel))
        if control == 80:
            try:
                from .session import update_state
                from .tape import cc_value_to_tempo
                update_state(tempo_bpm=cc_value_to_tempo(int(value)), source="bridge")
            except Exception:
                pass

    def program_change(self, program: int, channel: int | None = None) -> None:
        self.send(mido.Message("program_change", program=program, channel=self.channel if channel is None else channel))
        try:
            from .session import update_state
            if 0 <= program <= 7:
                update_state(mode="synth", synth_slot=program + 1, source="bridge")
            elif 8 <= program <= 15:
                update_state(mode="drum", drum_slot=program - 7, source="bridge")
        except Exception:
            pass

    def pitch_bend(self, value: int, channel: int | None = None) -> None:
        # mido: -8192 .. 8191
        self.send(mido.Message("pitchwheel", pitch=int(value), channel=self.channel if channel is None else channel))

    def select_synth_slot(self, slot: int) -> None:
        """slot 1-8 -> program change 0-7"""
        self.program_change(slot - 1)

    def select_drum_slot(self, slot: int) -> None:
        """slot 1-8 -> program change 8-15"""
        self.program_change(8 + slot - 1)

    def set_mode(self, mode: str) -> None:
        self.cc("mode", 0 if mode == "synth" else 127)
        try:
            from .session import update_state
            update_state(mode="synth" if mode == "synth" else "drum", source="bridge")
        except Exception:
            pass

    def all_notes_off(self) -> None:
        for ch in range(16):
            self.send(mido.Message("control_change", control=123, value=0, channel=ch))
        self._open_notes.clear()

    def clock(self) -> None:
        self.send(mido.Message("clock"))

    def start(self) -> None:
        self.send(mido.Message("start"))

    def stop(self) -> None:
        self.send(mido.Message("stop"))

    def continue_(self) -> None:
        self.send(mido.Message("continue"))

    def song_position(self, pos: int) -> None:
        """Song position pointer (F2, LSB, MSB): pos counts 16th notes from the start, 0..16383. TE lists it as 'set sequencer position'."""
        self.send(mido.Message("songpos", pos=int(pos)))

    # ---- recording
    def duplex_recorder(self, playback: np.ndarray, gain: float = 1.0, blocksize: int = 256) -> "DuplexRecorder":
        """Record the Field while playing `playback` (frames x 2 at the device rate) into its USB input."""
        if self.audio is None:
            raise RuntimeError("no OP-1 field audio input found")
        return DuplexRecorder(device=self.audio["index"], channels=self.audio["inputs"], playback=playback, samplerate=int(self.audio["samplerate"]), blocksize=blocksize, gain=gain)

    def recorder(self, channels: int | None = None, samplerate: int | None = None, blocksize: int = 256) -> Recorder:
        if self.audio is None:
            raise RuntimeError("no OP-1 field audio input found")
        return Recorder(
            device=self.audio["index"],
            channels=channels or self.audio["inputs"],
            samplerate=int(samplerate or self.audio["samplerate"]),
            blocksize=blocksize,
        )


# --------------------------------------------------------------------------- long captures

class StreamRecorder:
    """Records an input device straight to a WAV file, for captures too long to hold in memory."""

    def __init__(self, device: int, channels: int, path: str, samplerate: int = 44100, blocksize: int = 1024):
        import queue
        self.device, self.channels, self.path, self.samplerate, self.blocksize = device, channels, path, samplerate, blocksize
        self._q: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._writer: threading.Thread | None = None
        self.frames = 0
        self.marks: list[tuple[str, int]] = []
        self.peak_history: list[float] = []  # peak per block, for silence detection
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        self._q.put(indata.copy())
        with self._lock:
            self.frames += frames
            self.peak_history.append(float(np.abs(indata).max()))

    def _write_loop(self):
        import soundfile as sf
        with sf.SoundFile(self.path, mode="w", samplerate=self.samplerate, channels=self.channels, subtype="FLOAT") as f:
            while True:
                block = self._q.get()
                if block is None:
                    break
                f.write(block)

    def start(self) -> "StreamRecorder":
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        global _open_streams
        try:
            self._stream = sd.InputStream(device=self.device, channels=self.channels, samplerate=self.samplerate, blocksize=self.blocksize, dtype="float32", callback=self._callback)
        except sd.PortAudioError:
            refresh_audio_devices()
            dev = find_audio_input()
            if dev is None:
                raise
            self.device = dev["index"]
            self._stream = sd.InputStream(device=self.device, channels=self.channels, samplerate=self.samplerate, blocksize=self.blocksize, dtype="float32", callback=self._callback)
        self._stream.start()
        _open_streams += 1
        return self

    def mark(self, label: str) -> int:
        with self._lock:
            idx = self.frames
        self.marks.append((label, idx))
        return idx

    def recent_peak(self, seconds: float) -> float:
        n = max(1, int(seconds * self.samplerate / self.blocksize))
        with self._lock:
            hist = self.peak_history[-n:]
        return max(hist) if hist else 0.0

    def stop(self) -> tuple[str, int]:
        global _open_streams
        assert self._stream is not None
        self._stream.stop(); self._stream.close()
        _open_streams = max(0, _open_streams - 1)
        self._q.put(None)
        if self._writer is not None:
            self._writer.join(timeout=30)
        return self.path, self.frames


def usb_layout(channels: int) -> dict[str, tuple[int, int]]:
    """Which USB input channels carry what, by the Field's usb audio mode (1-based pairs)."""
    if channels >= 10:
        return {"main": (1, 2), "track1": (3, 4), "track2": (5, 6), "track3": (7, 8), "track4": (9, 10)}
    if channels >= 8:
        return {"track1": (1, 2), "track2": (3, 4), "track3": (5, 6), "track4": (7, 8)}
    return {"main": (1, 2)}


class DuplexRecorder(Recorder):
    """Records the Field's USB input while playing audio into its USB output through one full-duplex stream.
    Two separate streams on the same device stall each other on macOS; one stream does both and keeps
    the playback and the recording sample-aligned. `playback` is a frames x 2 float32 array."""

    def __init__(self, device: int, channels: int, playback: np.ndarray, samplerate: int = 44100, blocksize: int = 256, gain: float = 1.0):
        super().__init__(device, channels, samplerate, blocksize)
        pb = np.asarray(playback, dtype=np.float32)
        if pb.ndim == 1:
            pb = pb[:, None]
        if pb.shape[1] == 1:
            pb = np.repeat(pb, 2, axis=1)
        self._playback = np.clip(pb[:, :2] * gain, -1.0, 1.0)
        self._pos = 0
        self.playback_started_at_sample: int | None = None

    def _duplex_callback(self, indata, outdata, frames, time_info, status):
        with self._lock:
            if self._t0_adc is None:
                adc = float(time_info.inputBufferAdcTime) if time_info is not None else 0.0
                self._t0_perf = time.perf_counter()
                if adc > 0:
                    self._t0_adc = adc
                else:
                    self._use_perf = True
                    self._t0_adc = self._t0_perf
            self._blocks.append(indata.copy())
            if self.playback_started_at_sample is None:
                self.playback_started_at_sample = sum(len(b) for b in self._blocks) - frames
        end = self._pos + frames
        chunk = self._playback[self._pos:end]
        outdata[: len(chunk)] = chunk
        if len(chunk) < frames:
            outdata[len(chunk):] = 0.0
        self._pos = end

    def start(self) -> "DuplexRecorder":
        global _open_streams
        self._stream = sd.Stream(device=(self.device, self.device), channels=(self.channels, 2), samplerate=self.samplerate, blocksize=self.blocksize, dtype="float32", callback=self._duplex_callback)
        self._stream.start()
        _open_streams += 1
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            with self._lock:
                if self._t0_adc is not None:
                    break
            time.sleep(0.002)
        if self._t0_adc is None:
            raise RuntimeError("the duplex audio stream never delivered a block")
        return self

    def start_playback(self) -> None:
        """Playback begins with the stream; call this to (re)start the buffer from the top at the current position."""
        self._pos = 0

    @property
    def playback_seconds(self) -> float:
        return len(self._playback) / self.samplerate
