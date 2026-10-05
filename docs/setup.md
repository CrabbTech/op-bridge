# Setup

- [Requirements](#requirements)
- [Install](#install)
- [Settings on the Field](#settings-on-the-field)
- [Connect a client](#connect-a-client): [Claude Desktop](#claude-desktop) ·
  [Claude Code](#claude-code) · [Codex CLI](#codex-cli) · [claude.ai and ChatGPT](#claudeai-and-chatgpt-remote-connector)
- [Modes and constraints](#modes-and-constraints)
- [Command line](#command-line)
- [Secrets](#secrets)
- [Files](#files)
- [Environment variables](#environment-variables)

## Requirements

- An OP-1 field connected over USB-C, in normal mode.
- macOS. Disk-mode installs look for the Field under `/Volumes` and eject with `diskutil`, and the
  speech tools use the Mac's built-in `say`; the project has only been run on a Mac.
- Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/).

## Install

```bash
git clone https://github.com/CrabbTech/op-bridge.git && cd op-bridge
uv sync
uv run op-bridge status
```

`status` prints the configuration and whether the Field's MIDI port and audio device are visible.

The first audio capture on macOS asks for microphone permission for the process that runs the
server; grant it. (On the device, the very first capture stalled until the prompt was answered:
[device.md §9](device.md#9-verified-on-the-device-2026-09-26).)

## Settings on the Field

On the Field (hold shift + COM, then T1 for system settings):

- **midi**: channel 1 (or tell the bridge with `uv run op-bridge channel N`), clock **in** or
  **both**, notes **both**, other **both**. The Field ignores notes on any other channel.
- **system > usb audio**: **10 channel** (labelled 10CH; main stereo plus tape tracks 1-4) so the
  main mix with the master bus is captured as well as the tracks. In 8-channel mode `get_status`
  says so in `notes_for_human`.
- Keep the Field in normal mode for playing and capture; the bridge only uses disk mode for preset
  installs and disk backups. Whether MIDI and USB audio survive MTP or disk mode is unverified
  (see [device.md](device.md)).

The model can ask for any of these with `human_steps("midi_settings")` or
`human_steps("usb_audio_mode")`, which return the exact key presses.

## Connect a client

Every client sees the same 72 tools and the same guides; the server carries its own instructions
(`get_guide("quickstart")`, then `get_status`), so a new session needs no preamble beyond what you
want made.

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
python3 scripts/register-claude-desktop.py              # quit Claude, write, relaunch
python3 scripts/register-claude-desktop.py --no-restart # only write; refuses while Claude runs
```

The script writes `/opt/homebrew/bin/uv` as the command (Homebrew's location on Apple Silicon);
edit the entry if `uv` lives elsewhere. It logs to `~/Music/op-bridge/register-claude-desktop.log`.

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

### claude.ai and ChatGPT (remote connector)

Both need a public HTTPS URL. Run the server over Streamable HTTP with a secret path, then expose it
with a tunnel:

```bash
uv run op-bridge serve --http --port 8765 --path-secret auto --public
```

```bash
cloudflared tunnel --url http://localhost:8765
```

The server prints its local URL, for example `http://127.0.0.1:8765/<secret>/mcp`; the connector
URL is the tunnel host plus the same path. In claude.ai: Settings > Connectors > Add custom
connector, paste the URL, no OAuth. In ChatGPT: Settings > Connectors > Advanced > Developer mode,
add the URL.

The secret path is the only access control, so treat the URL like a password and restart with a
new one when done. Anyone with the URL can play your Field and read the session files, nothing
more. `--public` makes the server accept any Host header (it turns off the MCP SDK's DNS-rebinding
protection), which a tunnel such as cloudflared or ngrok needs; leave it off for local use.

## Modes and constraints

Two modes:

- **freeform**: every tool is available.
- **guided**: the human owns the settings and sets constraints, which every tool enforces: key
  (notes must belong to it), tempo or tempo range, maximum polyphony, maximum take length, allowed
  slots, and which groups the model may touch (sound select, sound design, master, mixer,
  transport, tempo change, CC parameters in scores), plus free-text notes.

The model can only tighten constraints (`set_constraints`); the human loosens them from the CLI:

```bash
uv run op-bridge mode guided
uv run op-bridge constraints set key="D minor" tempo=96 allow_master=false allowed_slots="synth 1,synth 3,drum 2"
uv run op-bridge constraints show
uv run op-bridge constraints clear
uv run op-bridge mode freeform
```

A refused call comes back to the model as a readable tool error naming the constraint, for example
`PermissionError: guided mode: master is reserved for the human (constraint allow_master=false)`.

## Command line

| Command | What it does |
|---|---|
| `op-bridge serve [--http --host --port --path-secret --public]` | Run the MCP server (stdio by default). |
| `op-bridge status` | Configuration and whether the Field is visible. |
| `op-bridge mode freeform\|guided` | Set the mode. |
| `op-bridge constraints set\|clear\|show [key=value ...]` | Human-side constraints. |
| `op-bridge channel N` | Tell the bridge which MIDI channel the Field is set to. |
| `op-bridge judge auto\|local\|jev` | Ranking judge for sound search (Jev only when its key is set). |
| `op-bridge probe` | Characterize the connected Field. |
| `op-bridge sessions` / `op-bridge use NAME` | List sessions; switch session. |

Run them with `uv run` from the checkout.

## Secrets

Keys never live in the repository. They go in `~/Music/op-bridge/secrets.env`, an owner-only file:

```bash
bash scripts/set-secret.sh TYPESAFE_API_KEY   # optional: Jev ranking for sound search
bash scripts/set-secret.sh GEMINI_API_KEY     # optional: Google audio listener
```

[`scripts/set-secret.sh`](../scripts/set-secret.sh) reads the value with hidden input, so it never
appears in a chat or in shell history, and writes the file with mode 600 in a directory with mode
700 (the script always writes to `~/Music/op-bridge`, even when `OP_BRIDGE_HOME` points
elsewhere). The server loads the file at start; the Gemini listener re-reads it for every request, so a
rotated key needs no restart. `.gitignore` also excludes `secrets.env` and `*.env`.

Neither key is required. Without them, ranking uses local rules and listening uses the local model
(or nothing, if it is not installed).

## Files

Everything the bridge records lives under `~/Music/op-bridge` (override with `OP_BRIDGE_HOME`):

| Path | Contents |
|---|---|
| `config.json` | Mode, constraints, MIDI channel, latency, palette, current session, judge mode. |
| `device-state.json` | What the bridge last set on the Field (mode, slots, tempo, master effect), with timestamps. |
| `sessions/<name>/takes/` | Every take and audition: `<id>.wav`, `<id>.json` (score and analysis), `<id>.png` (spectrogram). |
| `sessions/<name>/seeds/` | Captured seeds. |
| `sessions/<name>/samples/` | Cuts made by `sample_from_take` (listed by `get_status`). |
| `sessions/<name>/backups/` | Tape stem backups from `backup_tape`. |
| `sessions/<name>/reviews/` | Audio-model reviews and the exact excerpts they heard. |
| `staging/` | Presets waiting for disk mode. |
| `field-backup/` | Copies of the Field's disk and of every slot file the bridge replaced. |
| `secrets.env` | Optional API keys (above). |

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `OP_BRIDGE_HOME` | `~/Music/op-bridge` | Where config, sessions and backups live. Tests point it at a temp dir. |
| `OP_BRIDGE_SYNC_LIMIT` | `25` | Seconds a tool call may take before the work becomes a background job. |
| `OP_BRIDGE_LISTENER_HOME` | `$OP_BRIDGE_HOME/listener` | Where the optional local audio model is installed. |
