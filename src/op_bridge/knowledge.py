"""What the model needs to know to wield the OP-1 field. Served through the MCP server so it travels with the bridge."""
from __future__ import annotations

import os

from .presets import SYNTH_ENGINES, FX_TYPES, LFO_TYPES
from .score import SCORE_GUIDE
from .sequencers import SEQUENCER_GUIDE

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "reference")

OVERVIEW = """OP-1 field in one page.

The Field is a portable synthesizer, sampler, drum machine, four-track tape recorder and mixer.
Four modes: synth, drum, tape, mixer. A sound (synth preset or drum kit) is four modules on the
track keys T1 to T4: engine, envelope, FX, LFO. Eight synth slots and eight drum slots sit on keys
1 to 8. Encoders are blue, ochre, gray, orange; every page has four of them.

How a track gets made on the Field: pick or design a sound, play it, record it to one of the four
stereo tape tracks, overdub the other tracks, then balance levels, pan, EQ, master FX and drive in
the mixer. The tape is six minutes long; up to eight tapes with four tape styles.

What this bridge lets you do over USB MIDI (TE's firmware 1.7 MIDI map; items marked * were verified
on Tyler's Field, see get_guide("device"); the rest is TE's documentation, not yet tried here):
- play notes with velocity *, only on the Field's configured MIDI channel *; pitch bend; hold notes
  with the sustain pedal CC 64
- load any synth or drum slot (program change *), switch synth/drum mode (CC 93 *)
- shape the loaded sound live: the four engine encoders (CC 46-49; 46 *), envelope ADSR (50-53),
  FX encoders (54-57), LFO encoders (58-61), randomize (62) the patch; reset (63) is in TE's map but
  did nothing on the device *, and a program change does not reload a patch *: live edits persist until
  the human reverts the preset on the device, so set every knob you rely on explicitly
- master bus: master FX encoders (70-73), master out (74-77, TE's "master compressor parameters";
  left/right/drive/release is the bridge's reading of that page), EQ (90-92)
- mixer: per-track volume, pan, mute (CC 7/10/9 on MIDI channels 1-4 = tracks 1-4; mute *, and the
  USB pairs are post-mixer *)
- tape transport: play, stop, jump to start (CC 105/104/84 *), jump to end, previous/next bar, loop
  in/out/toggle. Clock: in midi sync mode the Field follows the bridge's MIDI clock by scaling the
  tape speed relative to its song BPM *, so the bridge sends CC 80 (song BPM, 40-180 in TE's steps)
  with the score tempo before clock; a mismatch records the take at the wrong speed and pitch *
- record to tape: an armed take starts on the first incoming note * and the bridge stops the tape *
- voices and words, only when a piece calls for them (most pieces do not, and nothing here assumes a
  vocal): speak_through_vocoder streams words the Mac synthesises into the Field's USB input as the
  vocoder's modulator while holding carrier chords (the human loads a vocoder preset and sets the
  input to usb audio; formant near 64 and a low waveform keep words intelligible, mix only blends in
  the dry voice); a spoken or sung WAV can become a sampler preset (author_sampler_preset) played as an
  instrument; stream_to_tape streams any WAV or spoken text into the Field's USB input for TE's external-audio
  recording procedure (input key toggled in tape mode, record + play; human_steps input_to_tape) or for the
  Field's own sampler; and vocals can be added outside the Field, mixed with the stem backups
- author presets as files (engine type, FX type, LFO type, octave, envelope) and install them into
  slots when the human puts the Field in disk mode * (a composed cluster preset loaded and played *).
  The Field rejects files whose values are out of range for the engine and turns them into samples *,
  so the bridge composes files only from values the Field itself has written; after loading, set the
  engine knobs live with set_parameters
- sample: turn any WAV, or a region cut from a recorded take, into a synth sampler preset * (a cut
  from a take played at its root, chromatically and through its loop) and install it the same way
  (get_guide("sampler") has the resampling loop and the limits)
- drive the sequencers: endless * (in drum mode, short incoming notes entered steps while it was
  stopped and one held note played them back); a held chord with clock is expected to feed the
  arpeggio, hold, finger, sketch and tombola sequencers by analogy with a key press once the human
  has enabled one, so confirm with one held note (get_guide("sequencers"), list_sequencers, hold_chord)

Long operations are jobs: play, record_to_tape, play_drums, play_arrangement_part, hold_chord,
speak_through_vocoder, backup_tape and capture_seed return at once with a job_id when the work would take
longer than about 25 s (chat clients cut a tool call off near a minute). Poll job_status(job_id) every few
seconds; when status is done, its result is exactly what the direct call returns (take_id, analysis, wav).
While a job runs, other device calls are refused as busy; cancel_job stops one and releases its notes. Pass
wait=false to make any of them a job, wait=true to insist on waiting.

Analysis beyond the tools: every take and audition is a WAV (get_take gives its path) and the bridge's own
Python environment already has numpy, scipy, soundfile and its analysis module (op_bridge.analysis:
harmonic tracking, modulation rate, onsets, spectrograms): run `uv run --directory {REPO} python`
instead of building a separate environment.

What MIDI cannot do (the human does it on the device when you ask): choose engine/FX/LFO types
of the loaded sound (use preset files instead), select the tape track to record on, arm record,
lift/drop/split, pick tape style, change sync mode, start a mixdown, sample from an input, select
and enable a sequencer or turn its encoders.

Recording to tape: the human selects a track and arms recording (shift + record). The Field
starts recording on the first incoming note, so playing a score starts the take; the bridge
sends tape stop at the end. Count-in applies if enabled.

Listening: the Field's USB audio carries the tape tracks as stereo pairs (8-channel mode) or the
main mix plus the tracks (10-channel mode, set in system settings). The bridge records these while
playing and analyses them: which notes were heard, timing, level, clipping, spectral movement,
and a spectrogram image. The live synth appears on the currently selected track's pair.

The Field cannot be queried. It transmits only the keys the human plays; mode switches, sound key presses,
encoder turns and tape transport send nothing (verified). get_status shows what the bridge last set; hand
changes are unknown until set_device_state records them or detect_mode guesses by ear. Select the sound
you need (select_sound, or kit_slot in play_drums) before playing rather than assuming the mode.

Effects, ten of them with named knobs: get_guide("effects"). Finding and choosing sounds: get_guide("sounds") (profiles, audit, index, search). Building a piece from
sections and parts: get_guide("arrangement"). Drums: get_guide("drums"). Sampler and sequencers have their
own topics too.

Device limits measured: about 8 ms latency, velocity gives about 20 dB of range, eight voices with
oldest-note stealing on the engine tested (keep chords to six unless told otherwise), drum kits
answer on MIDI notes 53 (F3) and up, notes are accepted only on the Field's configured MIDI channel.
"""

WORKFLOW = """Suggested workflow for a piece of music.

1. get_status: confirm the Field is connected, which mode the bridge is in, and the constraints.
2. If the player gave a seed (capture_seed), read its chords, key and tempo and build on them.
3. Choose sounds: audition the slots (audition tool) or author presets for what you need
   (author_synth_preset: engine, FX, LFO, octave, envelope), then ask the human to enter disk mode
   and call install_presets. Plan the palette for a session in one disk-mode round: sixteen slots
   is plenty, and switching between them afterwards is a program change.
4. Design each sound live: set_parameters for engine, envelope, FX and LFO encoders, then
   audition again and look at the spectrogram. A sound worth keeping as material can be resampled:
   cut a region of its take with sample_from_take and author_sampler_preset it for the next
   disk-mode round (get_guide("sampler")).
5. Write a score per part (validate_score, then play with record=true). Read the analysis: were
   all notes heard, is the level healthy, did the automation move the sound.
6. To commit a part to tape: ask the human to select the track and arm recording, then
   record_to_tape. One part per track, four tracks per tape. Bounce tracks if you need more.
   For an arpeggiated, held or tombola part ask the human to enable the sequencer first and play
   held chords (hold_chord to try it, record_to_tape with a held-chord score to commit; see
   get_guide("sequencers")).
7. Mix on the device: mixer levels, pans, mutes; master EQ, master FX, drive.
8. backup_tape to keep stems on the Mac; with the Field in 10-channel USB mode the main mix
   with the master bus is captured too.
"""


SAMPLER = """Sampling on the OP-1 field, and how the bridge does it.

The Field has two samplers. The synth sampler is chromatic and stereo: one sound of up to 6 seconds,
played from start, looped between loop in and loop out while a key is held, then played to the end.
T1 holds start, loop in, loop out, end; the shift page direction, fine tune, loop fade, gain; T2 to
T4 envelope, FX, LFO. The drum sampler holds one stereo bank of up to 20 seconds cut into 24 key
regions, F3 (MIDI 53) to E5 (76), each with tuning, in/out points, play mode, direction, pan and
gain. Both are AIFF-C preset files (16-bit 44.1 kHz PCM plus TE's JSON) the bridge writes.

Authoring from the computer:
- author_sampler_preset(slot, wav_path, name, root_note, region) turns any WAV on this Mac into a
  synth sampler preset. It stages a file; install_presets copies it onto the Field in disk mode;
  select_sound loads it and set_parameters shapes it live (engine1-4: start, loop in, loop out, end
  on the sampler; pitch, in, out, play mode of the active drum key).
- The resampling loop: play a sound (play, audition or hold_chord with record=true) or
  record_to_tape a part; the take is the Field's USB audio; sample_from_take cuts a faded region of
  it into the session's samples folder; author_sampler_preset turns the cut into a preset; install_presets puts it on the Field. Level: the Field plays a sample at its recorded
  level (a -22 dBFS cut came out 16 dB quiet): cut with normalize_db=-1.0 or raise the gain; the
  tools warn below -6 dBFS.
- The Field's own sampling (input key, then shift + input to choose microphone, line in, FM radio,
  usb audio or ear, its own output) is a human action: human_steps sample_from_input,
  lift_tape_to_sampler, drop_sample_to_tape.

Units and limits, from the sampler files the Field wrote (docs/device.md):
- Synth sampler positions sit on a fixed 6.000 s timeline, not the file's length: knob = seconds /
  6 * 32767, so 32767 is frame 264,600 at 44.1 kHz and one unit is 0.1831 ms. Sources over 6 s are
  refused unless truncated (cut them with sample_from_take). The end marker is the last frame's
  index rounded up, like the Field's; the loop defaults to the whole sample.
- Drum positions use 2^31 for a 20 s bank, one frame being 2434.79 units; TE leaves 16 frames of
  silence between regions.
- root_note sets base_freq, the frequency of the key that plays the sample at recorded speed (C4 =
  261.6256 Hz; imports use A4 = 440 Hz); other keys play by frequency ratio.
- Shift page defaults are the Field's import defaults: direction 12000 (forward), fine tune 0, loop
  fade 0, gain 8192 (unity). Reverse (24576), the fine tune scale, gain above unity, loop=false and
  root_hz are unverified; the tool lists them under derived.unverified.
- Names follow the Field's own (lowercase letters, digits, space, '-', '#', at most 11 characters;
  tools report changes). Every other value comes from a Field-written file, because the Field
  demotes a preset with out-of-range values to a plain sample.

Open questions: the human reports preset names invisible on every authored slot, and a slot that looks
like a sample file while shift + its key opens the browser at the intended engine. Authored synth
presets kept the factory original_folder marker (written by the Field only on browser items);
author_synth_preset now drops it, to confirm on the next install. Confirm a slot with read_slots and
audition, not the display.
"""

def engines_catalog() -> dict:
    return {"synth_engines": SYNTH_ENGINES, "fx": FX_TYPES, "lfo": LFO_TYPES,
            "drum_engines": {"drum": "stereo drum sampler, 24 keys from MIDI note 53 (F3) to 76 (E5), one 20 s stereo bank at 44.1 kHz: per-key sample region (start/end, 2^31 = 20 s), pitch, play mode (encoder values 4096/12288/20480; 12288 is TE's default, 20480 sits on the hi-hat keys), direction, pan (16384 centre; stacked A+B kits crossfade two mono sounds), gain (8192 unity)",
                             "dbox": "dual oscillator drum synth: per key pitch, waveform, envelope, cross modulation; shift page: pitch 2, waveform 2, envelope 2, filter cutoff (dbox_data[24][8])"},
            "envelope": ["attack", "decay", "sustain", "release"], "envelope_shift": ["play mode", "portamento", "bend range", "volume"],
            "drum_envelope": ["attack", "gain", "release", "timing"]}


DRUMS_WORKFLOW = """

How to work with drums on the Field through this bridge.
1. kit_map(slot) auditions all 24 keys of a drum kit, labels them by their audio and saves the map; read the description,
   listen through view_spectrogram if needed, and fix or enrich labels with describe_kit_key. Kits hold far more than kick,
   snare and hats: shakers, brushes, toms, effects, vocal snips. Every key is usable in a pattern by role, Field key name
   (F2 ... E4) or MIDI number (53 ... 76); the map's kit_map_for_patterns lists them all.
2. Write a grid document (above), compile_drum_pattern to check the grid and constraints, then play_drums to hear it and
   read the onset analysis (hits_heard, timing). Accents, ghosts and ratchets carry the feel; humanize a little.
3. Accompaniment: read the seed or take you are playing to (tempo, feel, section lengths), match its tempo, and write the
   drum part to sit under it; record it to tape with play_drums(to_tape=true) after the human arms a track.
4. The endless sequencer is an alternative destination: it takes one note per step over MIDI, no rests, so send one
   instrument row per pass (endless_program) after the human enables endless on the drum kit and sets the note value.
   The bridge remains the main drum machine; the sequencers are for when the Field should keep playing on its own.
"""


ARRANGEMENT_WORKFLOW = """

Workflow: compile_arrangement to see the timeline and each part's length and problems; play_arrangement_part
for a rehearsal with record=true and read its analysis; then, one part at a time, ask the human to select
the part's track and arm recording (human_steps arm_recording) and call play_arrangement_part with
to_tape=true. Rewind and set the tempo BEFORE the human arms (tape("start"), set_tempo); after arming send
nothing but the part. Every part should start on beat one, since the armed recording starts on the first
note. Keep clips in beats relative to their section; reuse with {"as": "A"}; loop short clips.
"""

SOUNDS_GUIDE = """Sounds: what a slot can play, what it sounds like, and how to find one.

- slot_profile(kind, slot): engine, name and the playable range read from the slot's file. Samplers carry a root
  note and keep their character within an octave of it (a Rhodes sample rooted at C5 played at C3 is a slowed,
  muddy drone); synthesis engines pitch every MIDI note literally. play(...) with sound={"kind","slot"} guards
  the range and transposes by octaves when needed; keep fundamentals between C1 and C7.
- audit_sound(kind, slot): auditions a low, middle and high note, measures attack, sustain, release, brightness,
  band energies, noisiness, harmonicity, movement and stereo width, derives tags (dark/warm/bright, sustained/
  decaying/short, harmonic/noisy, static/moving) and role suggestions (bass, pad, keys, pluck, lead, drone, fx,
  percussive), and stores everything in the sound index. Audit each slot once; then reason from the index.
- find_sounds(query): ranks the index against a plain request. With TYPESAFE_API_KEY the Jev judge scores
  candidates; otherwise a local rule-based ranking does. Either way the model reads only the top picks.
- sound_search(slot, want): randomizes the loaded patch and auditions each roll, scoring rolls against the
  request. The Field cannot recall an earlier roll and a program change discards the current one, so when a
  roll is right, ask the human to hold the sound key for two seconds to snapshot it, then read the snapshot
  file in the next disk-mode round to learn its parameters.
- Drums: kit_map and describe_kit_key do the same job for the 24 keys of a kit.
"""


QUICKSTART = """Quickstart: a piece on the Field in five moves (about a minute of reading; the other guides are on demand).

1. get_status: is the Field connected, which mode (freeform or guided) and constraints, what the bridge last
   set (slots, tempo), any running job. Read get_guide("score") before writing notes. Ask the human for nothing yet.
2. Choose sounds from what is already known before playing anything: the palette notes in get_status,
   describe_slot and slot_profile (engine, playable range, a sampler's root), find_sounds over the index.
   Audition the candidates in one call with audition_slots rather than one slot per call. Author a preset
   only for a sound that does not exist yet: it costs the human a disk-mode round.
3. Write each part as a score, drums as a grid (get_guide("drums")), the song as an arrangement when it has
   sections (get_guide("arrangement")). Rehearse with play or play_arrangement_part (record=true) and read the
   analysis: notes_heard, timing, peak and clipping, spectral movement; measure_take for bass, pitch or
   movement questions; view_spectrogram(id, log_frequency=true) to see the low end.
4. Record track by track: ask for human_steps("arm_recording") per track, then record_to_tape or
   play_arrangement_part(to_tape=true) or play_drums(to_tape=true). After arming send nothing but notes and
   clock. Anything over 25 s comes back as a job: poll job_status until done.
5. Balance with set_mixer and set_master, keep stems with backup_tape, and stop. Words are optional: a
   piece needs no vocal, and when it wants one the overview lists the routes (vocoder is one of them).

Rules of thumb: one device call at a time (a busy device says so; wait or cancel_job); the Field cannot be
read back, so set every knob you rely on and record hand changes with set_device_state; live edits persist on
a slot until the human reverts it; the device and manual guides are long, open them for a specific question.
"""


LISTENING_GUIDE = """AUDIO LISTENING
Offline is the default. Explicit backend='gemini' sends just the chosen excerpt and question
to Google's Gemini API. Set GEMINI_API_KEY in ~/Music/op-bridge/secrets.env (or OP_BRIDGE_HOME).
The saved file is re-read for each request, so rotating the key needs no MCP restart.
listener_status(backend='gemini') checks configuration without making a network request.
listen_to_take(id, backend='gemini', model='gemini-3.1-pro-preview', seconds=15) uses Pro;
model='gemini-3.8-flash' selects Flash. Use cloud analysis only for audio the user has authorized
sharing with Google. The local model and its files are not required for Gemini requests.
Gemini's output budget defaults to 2048 tokens, including thinking; allowed range 256–4096.
There are no automatic retries or uploads on startup/status checks. Reviews include provider,
model version, usage and an estimated USD cost at published standard-tier rates, not actual billing.
usable=true means a complete answer, not verified accuracy. Blocked/truncated answers preserve usage
but are marked unusable. Cancelling stops the local worker and closes its connection; Google
may still bill work already received. The maximum excerpt duration remains 30 seconds.

LOCAL BACKEND
listener_status reports whether the optional Qwen2-Audio installation is ready. Explicit setup:
python3 adapters/qwen_audio/setup.py (Apple Silicon, uv and hf required; about 6.6 GB of model files).
Inference uses a separate Python runtime, runs offline, and releases model memory after each request.

listen_to_take(id, start_s=0, seconds=15, question=..., channels=None) reads a saved take, audition,
seed, or absolute WAV path. It does not access the Field. Choose 1–30 seconds and ask one concrete
question about audible instruments, rhythm, texture or balance. Start with the default neutral question;
menus of possible instruments and long rubrics caused false positives in calibration. Channels default to 1–2 (main mix on
10-channel Field USB). For another track, supply its one or two 1-based channel numbers explicitly.
Input becomes 16 kHz mono, so the model cannot judge stereo placement or treble above 8 kHz.

By default the call returns a job immediately. Poll job_status; cancel_job stops the subprocess.
The local timeout includes waiting for any other local listener job. Only one local model process runs at a time.
Reviews and the exact analyzed excerpts are saved under the current session's reviews directory.
Returned source start/end times are file positions, not events inferred by the model.

Treat observations as fallible evidence. Models can invent instruments, tempo, chords and praise.
The local controls found broad timbre differences, but also invented drums in a mix without a drum stem;
layered instrument identification, harmonic accuracy and mix-quality judgment remain unvalidated.
Use measure_take for measured properties; use known scores plus pitch analysis for harmonic checks.
Do not equate a fluent description with a good track. Compare the same window in the full mix and
isolated stems, and check deliberately altered variants when establishing whether a claim is useful.
This listener is a perceptual aid, not direct hearing by the chat model or an authoritative critic.
"""


def repo_dir() -> str:
    """The checkout this package runs from (src/op_bridge/knowledge.py -> repo root)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def guide(topic: str = "overview") -> str:
    topic = topic.lower().strip()
    if topic in ("overview", ""):
        return OVERVIEW.replace("{REPO}", repo_dir())
    if topic in ("quickstart", "start", "quick"):
        return QUICKSTART
    if topic == "workflow":
        return WORKFLOW
    if topic in ("listening", "listener", "listen"):
        return LISTENING_GUIDE
    if topic in ("score", "scores"):
        return SCORE_GUIDE
    if topic in ("sampler", "sampling", "samples", "resampling"):
        return SAMPLER
    if topic in ("sequencers", "sequencer", "arpeggio", "tombola"):
        return SEQUENCER_GUIDE
    if topic in ("arrangement", "arrangements", "song", "structure"):
        from .arrangement import __doc__ as ARR
        return ARR + ARRANGEMENT_WORKFLOW
    if topic in ("effects", "fx", "effect"):
        from .effects import guide as fx_guide
        return fx_guide()
    if topic in ("sounds", "sound", "search", "timbre", "palette"):
        return SOUNDS_GUIDE
    if topic in ("drums", "drum", "beats", "kits"):
        from .drums import DRUM_GUIDE
        return DRUM_GUIDE + DRUMS_WORKFLOW
    if topic in ("engines", "sounds", "fx", "lfo"):
        import json
        return json.dumps(engines_catalog(), indent=1)
    if topic in ("midi", "cc"):
        p = os.path.join(REFERENCE_DIR, "midi-reference-fw1.7.0.txt")
        return open(p).read() if os.path.exists(p) else "MIDI reference not found"
    if topic in ("manual", "guide"):
        p = os.path.join(REFERENCE_DIR, "op-1-field-user-guide-fw1.7.txt")
        return open(p).read() if os.path.exists(p) else "manual not found"
    if topic == "device":
        p = os.path.join(os.path.dirname(REFERENCE_DIR), "device.md")
        return open(p).read() if os.path.exists(p) else "device brief not found"
    return f"unknown topic {topic!r}; try overview, workflow, listening, score, sampler, sequencers, engines, midi, manual, device"


def manual_search(query: str, context: int = 400) -> str:
    p = os.path.join(REFERENCE_DIR, "op-1-field-user-guide-fw1.7.txt")
    if not os.path.exists(p):
        return "manual not found"
    text = open(p).read()
    low = text.lower(); q = query.lower()
    hits = []
    start = 0
    while len(hits) < 6:
        i = low.find(q, start)
        if i < 0:
            break
        a, b = max(0, i - context // 2), min(len(text), i + context)
        page = text.rfind("===== PAGE", 0, i)
        pno = text[page + 11: text.find("=", page + 11)].strip() if page >= 0 else "?"
        hits.append(f"[page {pno}] ..." + text[a:b].replace("\n", " ") + "...")
        start = i + len(q)
    return "\n\n".join(hits) if hits else f"no match for {query!r} in the manual"
