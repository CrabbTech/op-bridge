# Tools and capabilities

op-bridge exposes **72 MCP tools**, one resource (`opbridge://guide/{topic}`, the same text as
`get_guide`) and one prompt (`make_a_track`). Every client sees the same surface. The server
carries its own instructions, so a new session needs no preamble: the model starts with
`get_guide("quickstart")` and `get_status`.

The authoritative descriptions are the docstrings in [`src/op_bridge/server.py`](../src/op_bridge/server.py);
this page summarises them.

- [What the model can do](#what-the-model-can-do)
- [Tool reference](#tool-reference)
- [Guides the model reads](#guides-the-model-reads)

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
- **Audio descriptions** (optional): `listen_to_take` asks an audio model (local Qwen2-Audio on
  Apple Silicon, or Google Gemini when explicitly requested) about a saved excerpt. It returns
  observations alongside measured levels and the exact source window. See [listening.md](listening.md).
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
  `slot_profile` knows a sampler's root and every slot's playable range, and arrangement parts are
  transposed by octaves when they fall out of range. Ranking uses the local rules, or TypeSafe's Jev when
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
  meanwhile. `wait=false` forces a job, `wait=true` insists on waiting. The threshold is
  `OP_BRIDGE_SYNC_LIMIT` (seconds, default 25).
- **Two modes**: `freeform` (every tool) and `guided` (the human owns settings and sets
  constraints: key, tempo, polyphony, allowed slots, which parameter groups the model may touch).
  The model can only tighten constraints; the human loosens them with the CLI ([setup.md](setup.md#modes-and-constraints)).

## Tool reference

"Device" means the call talks to the Field over USB MIDI and/or USB audio. "Job" means the call
returns a `job_id` when the work is expected to take longer than the sync limit.

<details open>
<summary><b>Status, knowledge and session</b> (11)</summary>

| Tool | What it does |
|---|---|
| `get_status` | Connection state, USB audio layout, mode and constraints, what the bridge last set, running jobs, session contents, staged presets, palette notes, judge and listener status. Called first. |
| `get_guide` | Reference text by topic (see [below](#guides-the-model-reads)). |
| `search_manual` | Full-text search of TE's user guide with page numbers, if you added a local copy ([docs/reference](reference/README.md)); otherwise says how to add it. |
| `human_steps` | Exact key presses for what only the human can do: arm recording, select a track, disk mode, USB audio mode, MIDI settings, sampling, sequencer setup, effect toggle, revert a preset, and more. |
| `list_engines` | Synth engines, FX and LFOs with what each of the four encoders does. |
| `list_sessions` / `use_session` | List sessions on disk; switch to or create one. |
| `set_constraints` | Tighten the guided-mode constraints. Never loosens them. |
| `describe_slot` | Remember what a slot holds; shown in `get_status` as the palette. |
| `set_device_state` | Tell the bridge what the human changed by hand (mode, slot, tempo, master effect type). |
| `detect_mode` | Device. Guess synth vs drum mode by ear from two adjacent keys; reported as a guess with its evidence. |

</details>

<details>
<summary><b>Sounds and effects</b> (14)</summary>

| Tool | What it does |
|---|---|
| `select_sound` | Device. Load synth or drum slot 1-8 (program change). |
| `set_parameters` | Device. Turn encoders by name (engine, envelope, FX, LFO, master, EQ, octave), 0-127. |
| `randomize_sound` | Device. CC 62, like shift + drop; returns the human step that restores the saved preset. |
| `revert_sound` | Device. Sends TE's reset CC 63, which showed no response in five trials on firmware 1.7, so it returns the human step that does revert. |
| `audition` | Device. Play a note or chord while recording; returns level, brightness, attack, release, latency, notes heard and a spectrogram id. |
| `audition_slots` | Device, job. Audition a list of slots in one call, one compact line per slot. |
| `slot_profile` | Engine, name, sampler root and playable range of a slot, from its file in the latest disk backup. |
| `audit_sound` | Device. Measure a slot at three pitches, derive tags and roles, store it in the sound index. |
| `find_sounds` | Rank the sound index against a plain request ("dark sustained bass"). |
| `sound_search` | Device. Randomize a slot repeatedly and score each roll against a request. |
| `list_effects` | The ten effects with their knob names, character and uses. |
| `slot_effects` | Which effect a slot's saved preset carries and its knob values. |
| `set_effect` | Device. Turn the loaded patch's effect by knob name; the fazer also takes a rate in Hz. |
| `set_master_effect` | Device. Turn the master effect by knob name once its type is known. |

</details>

<details>
<summary><b>Presets, samples and the Field's disk</b> (10)</summary>

| Tool | What it does |
|---|---|
| `author_synth_preset` | Stage a synth preset (engine, FX, LFO, octave, envelope) composed only from values the Field has written. |
| `author_sampler_preset` | Stage a synth sampler preset from any WAV of up to 6 s, with root note and region. |
| `sample_from_take` | Cut a faded, optionally normalised region of a take into the session's `samples/`. |
| `list_staged_presets` / `discard_staged` | See or drop presets waiting for disk mode. |
| `install_presets` | Disk mode. Check every staged file (drum kits and sampler presets have validators; nothing is copied if one fails), back up the slots being replaced, write and read back each file, eject, wait for the Field to return. |
| `restore_slot` | Stage a slot's file from a disk backup so the next install puts the original back. |
| `learn_engines` | Read every preset on the mounted disk (or the latest backup) and record the Field's own engine, FX and LFO identifiers. |
| `read_slots` | What the 16 slots contain, from the mounted disk or the latest backup. |
| `backup_disk` | Copy all of the Field's preset files to a dated backup folder. |

</details>

<details>
<summary><b>Playing and recording</b> (5)</summary>

| Tool | What it does |
|---|---|
| `validate_score` | Check a score against the Field's limits and the session constraints without playing it. |
| `play` | Device, job. Play a score on the loaded sound; with `record=true` capture, analyse and save the take. |
| `hold_chord` | Device, job. Hold a chord with MIDI clock for the arpeggio, hold, finger, sketch and tombola sequencers. |
| `record_to_tape` | Device, job. Commit a part to the armed tape track; the bridge stops the tape at the end. |
| `panic` | Device. All notes off on every channel. |

</details>

<details>
<summary><b>Takes and analysis</b> (7)</summary>

| Tool | What it does |
|---|---|
| `list_takes` / `get_take` | Takes and auditions in the session; the full record (score and analysis) of one. |
| `view_spectrogram` | The spectrogram image of a take, audition or seed; `log_frequency=true` redraws it from 30 Hz. |
| `measure_take` | Band levels, pitch track with vibrato, harmonic levels, periodic movement, onsets. |
| `speech_intelligibility` | STOI (0-1) of a vocoder or vocal take against the speech that was sent. |
| `listener_status` | Is the local audio model installed, or the Gemini key configured. No network, no device. |
| `listen_to_take` | Job. Ask an audio model about 1-30 s of a saved take. Local by default; Google only with `backend="gemini"`. |

</details>

<details>
<summary><b>Tape, mixer and master</b> (5)</summary>

| Tool | What it does |
|---|---|
| `tape` | Device. Transport: play, stop, start, end, bars, loop points, MIDI start/stop/continue. |
| `set_tempo` | Device. Set the Field's tempo (CC 80, 40-180 BPM in the device's steps). |
| `set_mixer` | Device. Tape track 1-4 volume, pan, mute. |
| `set_master` | Device. Master FX encoders, left, right, drive, release, EQ. |
| `backup_tape` | Device, job. Play the tape and record every USB channel, then split into stems. |

</details>

<details>
<summary><b>Drums</b> (6)</summary>

| Tool | What it does |
|---|---|
| `compile_drum_pattern` | Validate a drum grid and return an ASCII grid, problems, and the compiled score. |
| `play_drums` | Device, job. Play a grid on a kit (or to tape); hits are checked by onset, not pitch. |
| `kit_map` | Device. Audition all 24 keys of a kit, classify each, save the map. Human labels always survive. |
| `get_kit_map` | The saved map of a drum slot. |
| `describe_kit_key` | Label one key; `by_human=true` outranks the classifier. |
| `endless_program` | Device. Enter steps into the endless sequencer over MIDI and optionally record a playback. |

</details>

<details>
<summary><b>Sequencers, seeds and arrangement</b> (7)</summary>

| Tool | What it does |
|---|---|
| `list_sequencers` | The seven sequencers, what drives each, and whether MIDI from the bridge drives it (with evidence). |
| `song_position` | Device. Send the MIDI song position pointer; what it moves is reported as untested. |
| `capture_seed` | Device, job. Listen while the human plays: notes, chords, progression, key and tempo guesses, audio. |
| `list_seeds` / `get_seed` | Seeds in the session; everything captured for one. |
| `compile_arrangement` | Compile a node graph of sections, parts and automation into one score per part. |
| `play_arrangement_part` | Device, job. Play or record one part, guarding its notes against the slot's playable range. |

</details>

<details>
<summary><b>Voice and USB audio input</b> (4)</summary>

| Tool | What it does |
|---|---|
| `list_voices` | Voices the Mac's built-in speech engine offers (offline). |
| `speak_through_vocoder` | Device, job. Stream synthesised speech into the Field's USB input while holding carrier chords; reports STOI and on-beat error. |
| `stream_to_tape` | Device, job. Stream a WAV or spoken text into the Field's USB input while the tape records, dry. |
| `probe_usb_input` | Device. Send a test tone into the USB input and check it comes back. |

</details>

<details>
<summary><b>Background jobs</b> (3)</summary>

| Tool | What it does |
|---|---|
| `job_status` | Status, stage, elapsed and remaining seconds, and when done the result a direct call would have returned. |
| `list_jobs` | Running and recently finished jobs. |
| `cancel_job` | Stop a job between events; held notes are released, a tape recording is stopped. |

</details>

## Guides the model reads

`get_guide(topic)` serves text from [`src/op_bridge/knowledge.py`](../src/op_bridge/knowledge.py)
and the modules it imports:

| Topic | Contents |
|---|---|
| `quickstart` | A piece in five moves; the entry point named in the server instructions. |
| `overview` | The Field in one page, what MIDI can and cannot do, with items verified on the device marked `*`. |
| `workflow` | Suggested order of work for a piece. |
| `score` | The score JSON format. |
| `drums` | Drum grid notation and the drum workflow. |
| `arrangement` | The arrangement graph format and workflow. |
| `sounds` | Profiles, audit, index and search. |
| `effects` | The ten effects with knob names. |
| `sampler` | Both samplers, the resampling loop, units and open questions. |
| `sequencers` | The seven sequencers and what MIDI drives. |
| `listening` | The audio-model listeners and their limits. |
| `engines` | The engine, FX and LFO catalogue as JSON. |
| `device` | [`docs/device.md`](device.md), the device brief with what was verified on the real Field. |
| `manual`, `midi` | TE's user guide and MIDI tables, if you added local copies ([docs/reference](reference/README.md)). |
