# op-bridge

An MCP server that lets an AI model **wield a Teenage Engineering OP-1 field**: choose and design
sounds, play parts with expression, run the tape and mixer, listen back through the Field's USB
audio, and keep local backups. The Field is the instrument and the studio; the computer only
sends MIDI and listens.

Works from Claude Desktop, Claude Code and any stdio MCP client, and from claude.ai or ChatGPT
through a public HTTPS tunnel (Streamable HTTP).

## What the model can do

- **Sounds**: load any of the 16 slots, turn every encoder on the engine, envelope, FX and LFO
  pages, randomize or revert a patch, and *author presets* (engine type, FX type, LFO type, octave,
  envelope) from Field-written values, installed when the Field is in disk mode. The Field
  validates preset values per engine, so authored files are composed from values the Field itself
  wrote (learned with `learn_engines` from factory presets) and knobs are then shaped live over CC.
- **Playing**: scores with velocity, timing feel, CC automation, pitch bend and sustain, sent with
  MIDI clock. Every take is recorded from the Field's USB audio and analysed: which notes were
  heard, timing, level, clipping, spectral movement, plus a spectrogram image the model can view.
  `measure_take` goes deeper on any take (band levels from sub to air, a pitch track with vibrato,
  harmonic levels, periodic movement, onsets) and `view_spectrogram(id, log_frequency=true)` shows
  the low end; `audition_slots` hears a list of slots in one call.
- **Tape and mix**: transport, loop points, per-track level/pan/mute, master EQ, master FX, drive.
  Recording to tape needs the human to select the track and arm record; the model's first note
  starts the take.
- **Sampler**: turn any WAV, or a region cut from a recorded take (`sample_from_take`), into a
  synth sampler preset (`author_sampler_preset`, 6 s, positions on the Field's fixed 6 s timeline)
  installed in the same disk-mode round: play a sound, record it, cut it, author it.
  Sampling from the Field's inputs stays a human action with exact key presses in `human_steps`.
- **Sequencers**: `list_sequencers` and `get_guide("sequencers")` describe the seven sequencers;
  `hold_chord` holds a chord with clock for the arpeggio, hold and tombola sequencers once the human
  has enabled one, and `song_position` sends the song position pointer.
- **Drums**: a grid notation the model writes from any step diagram (accents, ghosts, ratchets, swing,
  chained sections), kit maps that label all 24 keys of a kit from their audio with human corrections,
  beats played on the Field or recorded to tape, and the endless sequencer programmed over MIDI.
- **Sound index and search**: `audit_sound` measures a slot at three pitches and tags it (dark/warm/bright,
  sustained/decaying/short, harmonic/noisy, moving) with role suggestions; `find_sounds` ranks the index
  against a plain request; `sound_search` randomizes a slot and auditions each roll against a request.
  `slot_profile` knows a sampler's root and every slot's playable range, and playing tools transpose
  out-of-range parts by octaves. Ranking uses the local rules, or TypeSafe's Jev when
  `TYPESAFE_API_KEY` is set in `~/Music/op-bridge/secrets.env` (owner-only file, loaded at start) and the
  judge mode allows it: `uv run op-bridge judge auto|local|jev`. Everything works without Jev.
- **Arrangement**: a node graph of sections, parts (a clip per section: score, drum pattern, chords,
  reuse, rest), automation and edges; compiled to one score per part and recorded track by track.
- **Voices, when wanted**: `speak_through_vocoder` places spoken lines on the beat grid, holds carrier
  chords, can send clock and record straight to tape, and scores the result with STOI
  (`speech_intelligibility` does the same for any take against its speech). Nothing assumes a vocal:
  `stream_to_tape` puts any WAV or spoken text on a track dry through the USB input (the human toggles the
  input key in tape mode and presses record + play, as TE's guide says), a sung or spoken WAV can become a
  sampler preset, or vocals can be mixed with the stems outside the Field.
- **Seeds**: capture what the player plays on the Field (chords, progression, key and tempo
  guesses, encoder moves, the audio) and build on it.
- **Backups**: play the tape and capture all USB channels as stems; in the Field's 10-channel
  USB mode the main mix with the master bus is captured too.
- **Long operations are jobs**: a take, tape recording, arrangement part, drum pattern, vocoder line,
  tape backup or seed capture longer than about 25 s returns a `job_id` at once (chat clients cut a
  tool call off near a minute); `job_status` reports progress and then the same result a direct call
  gives, `cancel_job` stops it and releases the notes, and other device calls are refused as busy
  meanwhile. `wait=false` forces a job, `wait=true` insists on waiting.
- **Two modes**: `freeform` (every tool) and `guided` (the human owns settings and sets
  constraints: key, tempo, polyphony, allowed slots, which parameter groups the model may touch).
  The model can only tighten constraints; the human loosens them with the CLI.

The model's reference material travels with the server: `get_guide`, `search_manual` (TE's
firmware 1.7 manual), `list_engines`, `human_steps` (exact key presses for the human).

## What the Field cannot tell you

The Field has no state read-back: in normal mode it transmits only the keys you play, nothing for
mode, sound or transport changes. The bridge therefore remembers what it last set (shown in
`get_status`), accepts hand changes through `set_device_state`, can guess synth versus drum by ear
with `detect_mode`, and every playing tool selects the sound it needs first.

## Setup

```bash
git clone <this repo> && cd op-bridge
uv sync
uv run op-bridge status
```

On the Field (hold shift + COM, then T1 for system settings):

- midi: channel 1 (or tell the bridge with `uv run op-bridge channel N`), clock **both**,
  notes **both**, other **both**
- system > USB MODE: **10CH** (the manual calls it usb audio modes: main stereo + tape tracks) so the main mix is captured
- keep the Field in normal mode for playing and capture; the bridge only uses disk mode for preset
  installs (whether MIDI and USB audio survive MTP or disk mode is unverified, see `docs/device.md`)

The first audio capture on macOS asks for microphone permission for the process that runs the
server; grant it.

### Claude Desktop

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "op-bridge": {
      "command": "uv",
      "args": ["--directory", "/ABSOLUTE/PATH/op-bridge", "run", "op-bridge", "serve"]
    }
  }
}
```

On macOS the desktop app keeps this file in memory and writes its copy back on quit, so an entry
added while the app is running is lost. Either edit with the app closed, or run this from any
terminal (a detached worker quits Claude, merges the entry with a backup, and relaunches it):

```bash
python3 scripts/register-claude-desktop.py
```

### Claude Code

```bash
claude mcp add op-bridge -- uv --directory /ABSOLUTE/PATH/op-bridge run op-bridge serve
```

### Codex CLI

`~/.codex/config.toml`:

```toml
[mcp_servers.op-bridge]
command = "uv"
args = ["--directory", "/ABSOLUTE/PATH/op-bridge", "run", "op-bridge", "serve"]
```

(or `codex mcp add op-bridge -- uv --directory /ABSOLUTE/PATH/op-bridge run op-bridge serve` on
versions that have `codex mcp`). Any other stdio MCP client takes the same command and arguments.

Every client sees the same 64 tools and the same guides (`get_guide("overview")` first, then
`get_status`); the server carries its own instructions, so a new session needs no preamble beyond
what you want made.

### claude.ai and ChatGPT (remote connector)

Both need a public HTTPS URL. Run the server over HTTP with a secret path, then expose it with a
tunnel:

```bash
uv run op-bridge serve --http --port 8765 --path-secret auto --public
```

```bash
cloudflared tunnel --url http://localhost:8765
```

The server prints its local URL, for example `http://127.0.0.1:8765/<secret>/mcp`; the connector
URL is the tunnel host plus the same path. In claude.ai: Settings > Connectors > Add custom
connector, paste the URL, no OAuth. In ChatGPT: Settings > Connectors > Advanced > Developer mode,
add the URL. The secret path is the only access control, so treat the URL like a password and
restart with a new one when done. Anyone with the URL can play your Field and read the session
files, nothing more.

## Modes and constraints (human side)

```bash
uv run op-bridge mode guided
uv run op-bridge constraints set key="D minor" tempo=96 allow_master=false allowed_slots="synth 1,synth 3,drum 2"
uv run op-bridge constraints clear
uv run op-bridge mode freeform
```

## Files

Everything the bridge records lives under `~/Music/op-bridge` (override with `OP_BRIDGE_HOME`):
`config.json`, `sessions/<name>/{takes,seeds,samples,backups}` (`samples/` holds the cuts
`sample_from_take` makes, listed by `get_status`), `staging/` (presets waiting for disk mode),
`field-backup/` (copies of the Field's disk and of any slot file the bridge replaces).

## Tests

All tests are functional. Those that need the Field skip when it is not connected.

```bash
uv run pytest -q
```

## Reference

`docs/device.md` is the device brief including what was measured on a real Field.
`docs/reference/` holds TE's user guide (firmware 1.7), the MIDI reference and the firmware
changelog.
