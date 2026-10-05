# Listening: measurements, spectrograms and audio models

The model cannot hear. op-bridge gives it three kinds of evidence about what the Field actually
played, and keeps them apart:

1. **Measurements** computed from the recorded audio, on every take. Numbers, not opinions.
2. **Spectrograms**, the same picture a person would look at, returned as an image.
3. **Audio-model descriptions** (optional): a second model listens to an excerpt and describes it.
   These are labelled as fallible observations and never replace the measurements.

- [Measurements on every take](#measurements-on-every-take)
- [Spectrograms](#spectrograms)
- [Optional local listener](#optional-local-listener)
- [Optional Google audio listener](#optional-google-audio-listener)
- [What the listeners are not](#what-the-listeners-are-not)

## Measurements on every take

Every `play`, `record_to_tape`, `hold_chord`, `play_arrangement_part` and vocoder take is captured
from the Field's USB audio while it plays, saved as a WAV with a JSON sidecar, and analysed in
[`src/op_bridge/player.py`](../src/op_bridge/player.py) (`analyze_take`):

- **which USB pair carried the live sound** (it follows the selected tape track) and, in
  10-channel mode, the main mix level separately;
- **peak, RMS and clipping fraction**;
- **spectral centroid** over the take (median, min, max), a brightness measure;
- **notes heard**: for each of the first 64 notes in the score, the prominence of that pitch and
  its first two harmonics in the note's window; a note counts as heard at 8 dB or more. Each note
  also gets its onset offset in milliseconds.

Drum takes are judged differently, because percussion has no pitch to check: a hit counts as heard
when an onset lands near its beat, and the result reports `hits_heard` and timing in milliseconds
(`_drum_analysis` in [`server.py`](../src/op_bridge/server.py)). `hold_chord` tells the model to
ignore `notes_heard` when a sequencer is running, since the audible notes are then not the sent ones.

On demand, `measure_take(id)` goes further: six band levels (sub, bass, low_mid, mid, presence,
air), a pitch track with vibrato rate and depth, the first twelve harmonic levels, periodic
movement (an LFO, phaser or tremolo shows here), and onsets. `speech_intelligibility(id)` scores a
vocoder or vocal take with STOI against the speech that was sent. The analysis code is
[`src/op_bridge/analysis.py`](../src/op_bridge/analysis.py).

## Spectrograms

Takes from `play`, `play_drums`, `hold_chord`, the vocoder and USB-input tools, auditions and seeds
are saved with a spectrogram PNG (1200 × 400, time left to right, 0 to 8 kHz bottom to top, red
lines at score marks). `view_spectrogram(id)` returns it to the model as an image;
`view_spectrogram(id, log_frequency=true)` draws one from the WAV of any take on a logarithmic axis
from 30 Hz with gridlines, which shows bass detail the linear view hides.

<p align="center">
  <img src="assets/spectrogram-voices.png" width="100%" alt="Spectrogram of a four-bar rehearsal of the voices part: stacked horizontal harmonic lines that change with each chord, with two upward bends.">
</p>

<sub>Rehearsal of the "voices" part at 80 BPM, recorded from the Field's USB audio on 2026-09-30.
The pitch-bend rises written into the score show near 2 s and 8 s. The take's analysis heard 16 of
16 notes.</sub>

## Optional local listener

`listen_to_take` asks Qwen2-Audio about a saved excerpt. It returns observations alongside measured
levels and the exact source window. `listener_status` checks the installation;
`get_guide("listening")` explains its limitations to the model.

With `uv` and the Hugging Face `hf` CLI installed, on Apple Silicon, run this explicitly from the
checkout:

```sh
python3 adapters/qwen_audio/setup.py
```

This installs the pinned 4-bit `mlx-community/Qwen2-Audio-7B-Instruct-4bit` checkpoint and an isolated
Python 3.11/MLX runtime under `~/Music/op-bridge/listener` (or `OP_BRIDGE_LISTENER_HOME`). The model
files occupy about 6.6 GB. The bridge's Python dependencies do not change. Setup records model
revision, package versions and file hashes in `runtime.json`; it can be rerun at the same pins.
Reconnect the MCP server after adding these tools to an already running checkout.

Call `listen_to_take(id, start_s=0, seconds=15, question="Describe the audible rhythm and texture.")`.
`id` may be a take/seed/audition ID or an absolute WAV path. A job is returned immediately by default;
use `job_status` for the result and `cancel_job` to stop it. Excerpts and reviews are saved under
the current session's `reviews/`. No playback or Field access is involved.

Inference runs offline, never downloads a model, runs one local model process at a time and
releases model memory after each request. Inputs are limited to 1-30 seconds and converted to
16 kHz mono. Default channels are the first stereo pair; select other USB channels explicitly with
`channels=[3,4]`. Stereo placement and frequencies above 8 kHz are outside the model input. A
source timestamp identifies the analyzed window, not an event localized by the model. Exact note,
chord and tempo claims need independent verification. Compare isolated parts and altered mixes
before relying on a description for musical decisions.

The adapter includes a scoped correction for its pinned MLX-Audio revision: reference Whisper/Slaney
features and valid audio lengths replace the port's HTK features and fixed padded-token count.
[`adapters/qwen_audio/test_frontend.py`](../adapters/qwen_audio/test_frontend.py) checks the features
and attention equivalence in the isolated runtime. Real OP-1 controls distinguished coarse timbres
with a neutral question, but still produced invented instruments in layered audio. Long rubrics and
lists of possible sounds made this worse. This installation is not validated for harmonic accuracy
or deciding whether a mix sounds good.

## Optional Google audio listener

Save `GEMINI_API_KEY` in `~/Music/op-bridge/secrets.env` (under `OP_BRIDGE_HOME` if set), for
example with `bash scripts/set-secret.sh GEMINI_API_KEY` ([setup.md](setup.md#secrets)). Keep this
file private. The listener re-reads it for each request, including after key rotation. No Google
SDK or local model installation is needed. `listener_status(backend="gemini")` checks
configuration without contacting Google; it does not claim the key is authenticated.

```python
listen_to_take(id, backend="gemini", model="gemini-3.1-pro-preview", seconds=15)
# Alternative: model="gemini-3.8-flash"
```

Only explicit `backend="gemini"` requests send audio to Google and incur potential API charges.
Default calls remain local. Use cloud requests only for audio authorized for sharing with Google.
The chosen 1-30 second excerpt becomes 16 kHz mono PCM WAV and is sent with the question only; no
filenames, scores or session metadata are sent. Source measurements remain separate from model
descriptions. The output budget defaults to 2048 tokens including thinking (range 256-4096), with
no automatic retries. Each review records model version, token usage, a published-rate cost
estimate and `usable`. `usable` means complete, not verified accurate; truncated or blocked answers
are marked unusable. Cost estimates include thinking tokens and are not an account bill.
`cancel_job` terminates the worker and connection promptly, but Google may still bill a request it
has received. No startup or status call uploads audio or spends credits.

Descriptions are still fallible; compare stems and controlled variants before trusting claims
about musical parts, balance or harmony. Both backends share the mono and 8 kHz bandwidth
limits above. Reconnect an already running MCP server to load new tool arguments.

## What the listeners are not

The server's own instructions say it to every model that connects: measurements and the text-only
Jev judge are not listening, and audio-model descriptions can be wrong, so compare stems and
variants before changing an arrangement ([`server.py`](../src/op_bridge/server.py), `INSTRUCTIONS`).
The listening guide adds that a fluent description is not a good track, and that the listener is a
perceptual aid, not direct hearing by the chat model or an authoritative critic.
