# OP-1 field: what the device offers a bridge

Compiled 2026-09-26 from Teenage Engineering's OP-1 field user guide (PDF, firmware 1.4.2 era),
the web guide's MIDI reference (firmware 1.7.0), the firmware changelog (1.1.2 through 1.7.3),
and what this Mac sees with the Field plugged in. Items marked **verify** have not been tested
on the device yet.

## 1. Ways to connect

| Path | Direction | What travels | Notes |
|---|---|---|---|
| USB-C, Field as **device** (normal mode) | both | USB MIDI in/out, USB audio 8 in / 2 out at 44.1 kHz (seen from the Mac), charging | Class-compliant. This is what the Mac sees today: MIDI port "OP-1", audio device "OP-1". Multichannel audio arrived in firmware 1.6.0. Channel layout: **verify**. |
| USB-C, Field as **host** | both | MIDI and audio from a controller or USB audio device plugged into the Field | For OP-Z, K.O. II, keyboards, class 2.0 audio devices. A Mac cannot be a USB device, so this path is not for a laptop. |
| Bluetooth LE MIDI | both | MIDI only (notes, CC, clock out since 1.6.0, transport in improved in 1.7.0) | COM screen, blue encoder = advertise. macOS pairs it in Audio MIDI Setup. No audio. |
| MTP (COM > T4, default) | files | Tapes as four WAV tracks, mixdowns, presets (AIFF), samples | Primary file access. macOS needs an MTP client; not native in Finder. Whether MIDI stays alive in MTP mode: **verify**. |
| Disk mode (COM > shift+T4) | files | Patch data only, as a plain USB disk | Mounts in Finder. Standard mass storage. |
| 3.5 mm line out / line in | analog audio | Stereo | Headset mic supported on the output jack. |
| PO sync / 1/16 sync (3.5 mm) | out | Sync pulses | Left channel sync, right channel audio mix. |
| FM transmit / receive | radio | Audio | Novelty for this project. |

Input sources the Field can sample or feed to the element LFO: microphone, line in, FM radio,
USB audio from the computer, and "ear" (its own output).

## 2. The COM screen (hold shift + output key)

- **T1 midi**: MIDI channel in/out (blue), how to handle MIDI clock (ochre), MIDI notes (gray) and
  "other" messages such as mod wheel and CC (orange). Firmware 1.7.0 moved these into system
  settings under COM > T1, added a **MIDI monitor** that shows incoming messages, and an
  **SPP origin** setting. The per-group filters mean the Field only obeys clock, notes or CCs
  from the Mac if those groups are enabled on the device. The 1.6.9 notes mention transport
  messages being filtered under the "other" group.
- **T2 ctrl**: controller mode. The Field becomes a MIDI keyboard controller for the laptop.
  Settings: MIDI channel 1 to 16, encoders relative or absolute, octave. Whether the internal
  engines still sound in this mode: **verify**. Useful for capturing a player's chords and
  encoder moves cleanly.
- **T3 list**: lists and controls connected USB and BLE MIDI devices when the Field is host,
  with per-device MIDI settings.
- **T4 mtp / disk**: file access modes above.
- Blue: advertise BLE MIDI. Orange: toggle USB charging (removes USB noise).

The Field also sends MIDI in normal mode: the changelog mentions outgoing notes to other TE
devices, MIDI start when recording begins by holding rec and pressing a note, song position
when jumping with shift+arrows, and clock out over BLE. Exactly what the keys, sequencers and
encoders send in normal mode: **verify**.

## 3. What the Field obeys over MIDI (firmware 1.7.0 reference)

Channel messages, any channel 1 to 16 unless noted:

- Note on with velocity 1 to 127, note off. Both synth and drum respond to velocity (keyboard
  velocity sensitivity arrived in 1.3.2).
- Pitch bend, on synth.
- Program change: values 0 to 7 load synth slots 1 to 8, values 8 to 15 load drum slots 1 to 8.
- Clock (sync input), start (starts tape playback), continue, stop (stops tape), song position
  (sets sequencer position).

Control changes:

| CC | Function | Value |
|---|---|---|
| 7, 9, 10 on channels 1 to 4 | mixer volume, mute, pan for tape tracks 1 to 4 | 0 to 127; mute at 64 and above |
| 46 to 49 | synth: engine parameters 1 to 4 (the T1 page's blue, ochre, gray, orange encoders). drum: active key pitch, loop in, loop out, play mode | 0 to 127 |
| 50 to 53 | synth envelope attack, decay, sustain, release. drum envelope attack, gain, release, smooth | 0 to 127 |
| 54 to 57 | patch FX parameters 1 to 4 (T3 page encoders) | 0 to 127 |
| 58 to 61 | patch LFO parameters 1 to 4 (T4 page encoders) | 0 to 127 |
| 62 | randomize active patch | 64 and above |
| 63 | reset active patch | 64 and above |
| 64 | sustain pedal | 64 and above = down |
| 70 to 73 | master FX parameters 1 to 4 | 0 to 127 |
| 74 to 77 | master out page: TE calls these "master compressor parameters 1 to 4". The page holds master left, master right, drive, release | 0 to 127 |
| 78 | tape record level | 0 to 127 |
| 79 | octave down (below 64) or up | |
| 80 | tempo: 0 to 5 is 40 to 50 BPM, 6 to 120 is 52 to 166 BPM, 121 to 127 is 168 to 180 BPM | |
| 81 | metronome level | 0 to 127 |
| 82 to 85 | tape: previous bar, next bar, jump to start, jump to end | 64 and above |
| 86 to 88 | tape: set loop in, set loop out, toggle loop | 64 and above |
| 90 to 92 | master EQ low, mid, high | 0 to 127 |
| 93 | mode select: below 64 synth, 64 and above drum | |
| 102 on channels 1 to 8 | sound slot select, slot = channel | 64 and above |
| 104, 105 | tape stop, tape play | 64 and above |
| 123 | all notes off | |
| 1 to 4 | MIDI LFO inputs, when the patch's LFO is set to "midi": each mapped to a chosen destination parameter | 0 to 127 |

Two more behaviours matter for recording:

- Since 1.4.2, an **armed** tape recording starts on incoming external MIDI. The human arms
  (shift + record) and the bridge's first note starts the take. **verify** on 1.7.
- Count-in recording exists since 1.6.0 and can be disabled in system settings.

## 4. What MIDI cannot do

Choosing an engine, an FX type or an LFO type (only via preset slots, randomize, or files).
Selecting the tape track to record on (T1 to T4). Arming record. Lift, drop, split, join.
Tape style or tape selection. Sync mode. Input source. Starting a mixdown. Choosing a sequencer.
Sampling. Saving snapshots. All of these remain human or file-side actions.

## 5. Sound architecture the bridge should know

Every synth or drum preset is four modules on T1 to T4: engine, envelope, FX, LFO. Eight slots
each for synth and drum on keys 1 to 8. Encoders are blue, ochre, gray, orange, and CC 46 to 61
address exactly those four encoders on each of the four pages.

Synth engines and their four encoder parameters, from the manual:

- cluster: octave, detune and ring modulation, digitalness, wave number (shift: wave envelope, spread, unitor)
- digital: wave shaper page; dna: filter, wave number, wave modifier, noise
- dimension: waveform, modulation, filter cutoff frequency, filter resonance
- dr wave: wave type and length, filter, phase, chorus; env crossfader
- dsynth: waveform, envelope, cross modulation, frequency (shift: waveform, envelope, filter cutoff)
- fm: fm amount, frequency, topology, detune
- phase: phase shift, distortion amount, phase filter, phase tilt
- pulse: filter, amplitude, second pulse, modulation
- sampler (synth sampler): start, loop in, loop out, end (shift: direction, fine tune, loop fade, gain)
- string: tension, decay, detune, impulse
- voltage: modulation, ground noise, phase filter, detune
- vocoder: waveform, formant, bands, mix (uses the selected input as modulator)
- amp with tuner: added in 1.6.0, parameters not in the PDF

Envelope (T2): attack, decay, sustain, release. Shift page: play mode, portamento, bend range,
volume.

FX (T3), one active at a time, same list for patches and the master bus:
cwo (frequency, delay, feedback, sideband), delay (range, speed, feedback, level),
grid (delay x size, y size, z feedback, mix), nitro (frequency, filter follow, feedback,
frequency), mother (distance, gate, color, mix), phone (tone, gsm, baud, telemetry),
punch (frequency, rounds, power, punch), spring (tone, turns, damping, mix),
terminal (added in 1.5.0).

LFO (T4): random (speed, amount, destination, envelope), element (source, amount, destination,
parameter; sources include g-force, external input, envelope, sum), midi (CC 1 to 4 to four
destinations), tremolo (speed, pitch amount, volume level, pitch envelope; shapes sine, saw,
exp, square, blip), value (speed, amount, destination, parameter; shapes square, ramp, saw,
sine), velocity (destination amount, volume amount, destination, parameter).

Drum mode: drum sampler (per key: tuning, in point, out point, play mode; shift: direction,
panning, attack, gain) and dbox (dual oscillator drum synth: pitch, waveform, envelope, cross
modulation; shift: second oscillator and filter). Drum envelope is a transient shaper: attack,
gain, release, timing.

Tape: four stereo 32-bit tracks, six minutes each, up to eight tapes, four styles (studio,
vintage, porta, disk mini). Tape tricks on keys 1 to 8: loop in, loop out, loop on/off, break,
reverse, chop, memo 1, memo 2. Lift, drop, split, join, tape undo (1.7.0).

Mixer: track levels, pans, mutes; three-band EQ; master FX; master out with left, right, drive
and release. Master FX is not recorded to tape but is included in a mixdown.

Sequencers (note data, not audio): endless, pattern, finger, hold, arpeggio, sketch, tombola.
Tempo: free, beat match (sends clock), midi sync (follows external clock; tape speed follows
clock too), PO sync.

Polyphony is commonly reported as six voices; TE's docs do not state it. **verify**.

## 6. Audio back to the computer

- Live: USB audio, eight channels in. Most likely four stereo pairs. Which pair carries the live
  synth and which carry tape tracks 1 to 4: **verify** by recording all eight while playing a
  note with the tape stopped, then with the tape running.
- After the fact: mixdown files (output screen, T4 start, T2 stop) and tape tracks as WAV, both
  via MTP. Tape WAVs are trimmed to the recorded length since 1.7.0.
- Analog: line out into any interface.

## 7. Ways a bridge could work

- **A. Field as instrument, Mac as player and listener.** Normal mode over USB. Mac sends notes,
  CC, program change, clock and transport. The Field's engines, tape and mixer make the music.
  Mac records the eight USB channels as the AI's ears and as stems. Human does what MIDI
  cannot: picks engine types, arms record, selects the track, sets tape style. Fits "the AI
  wields the OP-1".
- **B. Controller mode for seed capture.** Human switches to COM > ctrl, plays chords and turns
  encoders; the Mac receives notes and encoder CCs. Then back to normal mode. Whether normal
  mode already sends enough for this: **verify**.
- **C. BLE MIDI as the control path.** Same messages without the cable. Audio still needs USB or
  line out. A fallback if USB MIDI and file modes conflict.
- **D. Patch authoring over disk mode.** OP-1 presets are AIFF files carrying the engine type and
  all module parameters. Writing them from the Mac and loading them by program change would let
  the AI choose engines and FX types, which MIDI cannot. Requires switching the Field into disk
  mode and back, so it is a between-sessions operation. Format for the Field: **verify**.
- **E. Field as host.** Not applicable to a Mac.

## 8. Things to verify on the device before locking the architecture

1. USB audio channel layout, and MIDI-to-audio latency.
2. Which device MIDI settings must be on for clock, notes and CC from the Mac, using the MIDI
   monitor.
3. Velocity response and polyphony per engine.
4. CC 46 to 49 act on the engine page as documented; CC 93, program change and CC 62 behave.
5. Armed recording starts on the first incoming note; MIDI start, stop and clock behaviour in
   midi sync mode.
6. What the Field sends in normal mode and in ctrl mode (keys, encoders, sequencers, clock).
7. Whether MIDI and USB audio keep working while MTP is active.
8. Preset file format on the Field, if option D is wanted.

## 9. Verified on the device, 2026-09-26

Tyler's Field, firmware with the 1.7 MIDI map, USB-C to a MacBook Pro. Device MIDI settings at
the time: channel 8 (then changed to 1), clock off, notes both, other both, SPP origin tape.

- **Notes are only accepted on the Field's configured MIDI channel.** With the device on
  channel 8, notes on channel 1 were ignored and notes on channel 8 played. TE's table says
  "1 to 16"; in practice the channel setting filters. The bridge must send on that channel.
- **The Field transmits its keyboard over USB MIDI** on its configured channel, at a fixed
  velocity of 127, note on and note off. Nothing else was seen while keys were played.
- **USB audio in is the four tape tracks as stereo pairs**: channels 1 and 2 are track 1,
  3 and 4 track 2, 5 and 6 track 3, 7 and 8 track 4. The live synth or drum appears on the
  pair of the currently selected track. Silence is exact digital zero. The main mix with EQ,
  master FX and drive is not among the eight channels.
- **Latency** from note-on to audio onset: about 8 ms at 44.1 kHz with 256-sample blocks.
- **Velocity** is honored: about 20 dB between velocity 8 and 127 on slot 1's engine, monotonic.
- **Polyphony**: chromatic clusters of 4, 6, 7 and 8 all sounded. At 10 and 12 the earliest
  notes faded, so the engine on slot 1 has eight voices with oldest-note stealing. Treat six as
  the safe default and eight as the ceiling; engines may differ.
- **Program change** 0 to 7 loads synth slots 1 to 8, each a distinct sound. Program change 8
  loads drum slot 1. Loading a slot does **not** discard live edits: encoder values changed over MIDI (and
  a randomize) stayed in place after switching to another slot and back, and after a program change to
  the same slot (verified 2026-09-27). Only the device's revert-preset shortcut, or a fresh load of the slot
  file after a disk-mode round, restores the saved preset.
- **CC 46** (engine parameter 1) changes the sound while a note holds, on channel 1 and on
  channel 8. CC 93 (mode) and CC 84, 104, 105 (tape start, stop, play) work.
- **Drum slot 1** responds to MIDI notes 53 through 82; nothing below F3 (53).
- **macOS side**: capture works through sounddevice and ffmpeg once the process has microphone
  permission. The very first capture attempt stalled until the permission prompt was answered,
  so the bridge should not block forever on a stream that never delivers audio.
- **10-channel USB MODE verified**: a note on the selected track arrives on channels 1 and 2 (main mix,
  with the master bus) and on channels 3 and 4 (track 1's pair); channels 5 to 10 stay silent.
- **The USB pairs are post-mixer.** Muting track 1 with CC 9 on channel 1 during tape playback
  silenced USB channels 1 and 2. Track levels, pans and mutes set over MIDI are what the Mac
  hears, so a stem backup of the tape is a playback capture of all eight channels. The master
  bus (EQ, master FX, drive) still is not in the USB stream.

## 10. From the firmware 1.7 manual (docs/reference/op-1-field-user-guide-fw1.7.pdf)

- **USB audio modes** (system settings, labelled USB MODE on the device: 2CH, 8CH, 10CH): 2 channel = stereo only;
  8 channel = tape tracks 1 to 4; 10 channel = main stereo plus tape tracks 1 to 4. Ten-channel mode is how the Mac gets the
  master bus with EQ, master FX and drive, alongside the stems.
- **COM > T1 is system settings**: keyboard (velocity, detuning), system, midi (outgoing channel,
  filters for clock, notes and other messages, song position pointer origin), clock, battery,
  and monitor (a MIDI monitor).
- **Randomize a loaded preset**: hold shift and press drop.
- **Revert a preset** (page 41, shift functions): hold shift and press the synth key for a synth sound, or
  shift and the drum key for a drum sound. This discards any changes to the active sound and reloads its
  saved preset. It is the only way back: see section 17. Tyler found it does nothing for a slot whose
  sound is not in the preset library (a file the bridge installed into `synth/user/N.aif`), so those
  are restored by choosing a library preset again or by a disk-mode round.
- **Amp engine** (new in 1.6.0): volume, compressor, tone, overdrive; T2 on the amp screen is a tuner.
- **Terminal FX** (new in 1.5.0): rate, bits, model, mix.
- **Count-in**: pressing play while recording is armed gives a count-in at the current tempo.
- **Tape undo and redo** exist since 1.7.0.

## 11. Preset files, verified 2026-09-26

- In disk mode the volume `OP-1` holds `synth/user/1-8.aif`, `synth/snapshot/*.aif` and `drum/user/1-8.aif`.
  The slot files are the slots. Writing `synth/user/8.aif` and ejecting changed what key 8 plays.
- Format: AIFF-C, chunks FVER, COMM (16-bit `sowt`), APPL (`op-1` + JSON), SSND. Synth JSON keys:
  type (engine), name, octave, knobs[8] (four encoders, then the shift page), adsr[8], fx_type,
  fx_active, fx_params[8], lfo_type, lfo_active, lfo_params[8], synth_version 3, mtime; sampler
  presets add base_freq, fade, stereo. Values are integers, mostly 0 to 32767, some bipolar.
  Drum JSON: per-key arrays of 24 (start, end, pitch, playmode, reverse, volume, pan, pan_ab,
  attack, fademode), dyna_env[8], fx and lfo; dbox kits carry dbox_data[24][8]. Drum sample
  positions use 2^31 for the 20-second bank, so one frame at 44.1 kHz is about 2435 units.
- Engine strings seen in files: sampler, dimension, digital, string, phase; FX: mother, terminal,
  nitro, phone, punch, delay, cwo; LFO: tremolo, element, random, value. Written and confirmed
  audible on the device: cluster and voltage. Remaining names (dna, drwave, dsynth, fm, pulse,
  vocoder, amp) are the original OP-1 identifiers and are expected to work the same way.
- **The container must match what the Field writes.** A file with a shorter COMM compression name
  and an all-zero sample was not recognised as a preset: the Field turned slot 7 into a sampler
  preset named after the file, which played silence. Matching TE's COMM byte layout (64 bytes with
  the name "Signed integer (little-endian) linear PCM"), key-sorted JSON ending in a newline padded
  with a space, and a real sample made the same preset load and play as a cluster engine.
- macOS creates `._name.aif` companions on the disk; the bridge deletes them before ejecting.
- The Field sometimes returns to normal mode by itself after the Mac ejects the disk, and sometimes
  waits for shift + COM.
- Seed capture: the Field transmits played keys on its MIDI channel; a single B4 was captured with
  its audio on the selected track's pair.
- **Recording to tape over MIDI, verified**: with track 2 selected and record armed (shift + record)
  the first incoming note started the recording, CC 104 stopped the tape at the end, and playback
  shows the take on track 2's USB pair. The live take during recording arrived on the same pair.
- **Engine choice through files is still open**: files written or modified from the Mac loaded as
  sampler presets playing their embedded audio, even when the JSON said cluster or voltage.
  The tests in progress separate container, serialization, type string and file identity.
- **Round four, 2026-09-26**: a byte-identical copy of a TE-written dimension preset placed in another
  slot loaded as that engine; the same file re-serialized by the bridge is byte-identical and loaded
  the same; a TE sampler preset copied to another slot loaded as itself; the same dimension file
  with only `type` changed to `cluster` loaded as a sampler. Conclusion: writing slot files works,
  and the Field falls back to a sampler when it does not recognise the type string. The
  identifiers seen so far are dimension, digital, string, phase and sampler; the rest must be
  learned from files the Field writes (load a factory preset per engine, read it in disk mode).
- **Clock sync, verified**: in midi sync mode the Field keeps its song BPM and follows incoming clock
  by changing the tape speed (the tempo screen shows the speed moving, the BPM stays). A take
  recorded with a 96 BPM clock while the Field sat at 120 BPM went to tape at 80 percent speed and
  played back a major third high. The bridge now sends CC 80 with the score tempo before clock.
- **Round five verdict, 2026-09-26**: the factory cluster preset with only its name changed loaded as
  a cluster; the same preset with only its knob values changed to values from another engine loaded
  as a sampler. The Field validates parameter values per engine and falls back to a sample when a
  value is out of range. Authoring therefore composes files from values the Field wrote: engine
  block from a factory file of that engine, FX and LFO blocks from files using those types,
  envelope inside observed ranges; knob positions are then set live over CC, which is always valid.
- **Authoring from scratch, verified 2026-09-26 (round seven)**: a preset composed by the bridge
  (cluster engine block from the factory file "caterpillar", delay FX parameters and tremolo LFO
  parameters from a Field-written pulse preset, envelope values inside the observed ranges, a new
  name) installed to key 8 and loaded as a cluster with an audible delay. The install now writes
  through a temporary name, reads the slot back and ejects only after verification; ejecting from
  the Field before that finishes aborts the copy.

- Manual page 22 names the shifted envelope page, which is what `adsr[4:8]` holds in a preset file: play mode
  (poly, mono...), portamento, bend range, volume. Page 22 also states that the master effect is not recorded
  to tape, only into an output mixdown.

## 12. Drum kit files, from the seven `drum` kits and one `dbox` kit the Field wrote (file analysis, 2026-09-26)

Analysed with tests/test_drum_kits.py; nothing in this section was played on the device yet.

- **Container**: every sampler kit is AIFF-C, stereo, 16-bit `sowt`, 44.1 kHz, chunks FVER, COMM (64 bytes),
  APPL, SSND. Frame counts 802366 to 867111 (18.19 to 19.66 s). The dbox kit carries a 22.05 kHz mono
  placeholder of 28896 frames (1.31 s), the same sample the Field writes into synth snapshots.
- **JSON**: type drum, name, octave 0, drum_version 2, stereo true, ten per-key arrays of 24 (start, end,
  pitch, playmode, reverse, volume, pan, pan_ab, attack, fademode), dyna_env[8], fx_type/fx_active/
  fx_params[8], lfo_type/lfo_active/lfo_params[8], optional mtime and original_folder (factory marker,
  absent from user snapshots). dbox kits have dbox_data[24][8] instead of the per-key arrays.
- **Positions are exact**: start and end are `floor(frame * 2^31 / 882000)`, so 2^31 is a 20 s bank
  at 44.1 kHz and one frame is 2434.79 units. 335 of the 336 values in the kits are reproduced by that
  integer formula (the one exception is an in point nudged by hand on the device, 898 frames off the grid).
  Checked against the audio: with this unit every region boundary lands in a silent gap (RMS 0.00001 to
  0.00013) and onsets follow the start points (RMS 0.16 to 0.32); scaling 2^31 to the file length instead
  puts the boundaries in the middle of hits. TE's kits leave 16 frames of silence between regions and
  18 frames after the last end. The "exoex" kit is laid out on a 0.5 s grid (start values 2^31 x 0.025 x n).
- **Maximum bank**: a position is below 2^31, so the addressable bank is 20.000 s = 882000 frames; the
  longest factory bank is 19.66 s. What the Field does with a longer file is unverified.
- **Keys**: 24 regions, key 0 = F3 (MIDI 53) up to E5 (76). The device answered drum slot 1 on notes
  53 to 82; which region notes 77 to 82 play is unverified.
- **playmode** (orange encoder): 4096, 12288 or 20480, the centres of 8192-wide selector bins. 12288 is
  on 120 of 144 factory keys, 20480 on the C#4 and D#4 hi-hat keys of every kit (plus a few others),
  4096 once. Every region decays to silence before its end, so the files do not show which value is
  one shot, play while held or loop: verify by sending a 50 ms note to a 12288 key and a 20480 key.
- **reverse** (shift blue, direction): 8192 on every factory key but one (15353, still under 16384).
  The encoding of reverse playback is unobserved.
- **pan** (shift ochre): 16384 centre, encoder steps of 1024, factory range 12288 to 20480 on plain
  kits. **pan_ab** true means stacking A+B: the left and right channels hold two different mono sounds
  (L/R correlation about 0 on every key of "exoex", 0.5 to 1.0 on the other kits) and pan crossfades
  between them, 0 = A only, 32766 = B only.
- **volume** (shift orange, gain): 8192 is the factory default and the mode of all values; observed
  3564 to 15077. dyna_env[1] (envelope gain) uses the same 8192 default.
- **pitch** (blue, tuning): 0 on every key but one (-223 on "exoex" G4). Semitone scale unknown.
- **attack** and **fademode** (zoomed "attack / release" and shift gray "fade amount"): 0 everywhere.
- **dyna_env** = [attack, gain, release, timing, 0, 0, 0, 0] (manual page 38; CC 50 to 53 call the
  fourth "smooth"). Sampler kits: [0, 8192, 0, 0]; the dbox kit: [25599, 6656, 11263, 2816].
- **dbox_data[key]** = [pitch, waveform, envelope, cross modulation, pitch 2, waveform 2, envelope 2,
  filter cutoff], the main page then the shift page like knobs[8]. Evidence: column 0 is bipolar
  (-12098 to 23862, lowest on the kick keys); column 1 is a stepped selector (every value on a
  328-unit, 1 percent grid, 0 on 17 keys) as a waveform choice would be; columns 2 and 6 hold short
  values, 328 to 4592, almost all on the same grid; columns 3, 4, 5 and 7 are continuous;
  keys C#4 and D#4 are identical except column 2 (984 vs 4592): closed and open hi-hat as the same
  voice with a longer envelope.
- **Authoring kits from the computer was removed from the project on 2026-09-29** (Tyler: maybe far down the
  line); the file analysis above still serves kit maps and validation of Field-written kits.
- **Endless sequencer over MIDI, verified 2026-09-26**: with drum mode, kit 1 and the endless
  sequencer enabled on the device, eight short incoming notes entered steps (hits were audible
  during entry) and a single held note then played a sequence back for as long as it was held,
  18 hits in 4 seconds on a steady grid with a swung alternation. Step entry and playback are
  therefore MIDI-drivable; selecting, enabling and clearing the sequencer stay human actions.
- **Composed beat on kit 1, 2026-09-26**: a two-bar 96 BPM pattern written from Tyler's key labels
  (kick 53, snare 55, closed hat 60, open hat 63, choke hat 61) played on drum slot 1 with 30 of 33
  hits detected by onset within 16 ms. The take's pitch-based "notes heard" check does not apply to
  drums; use the onset count. Main mix peaked at -0.3 dBFS, so drum takes should be played a little
  softer or the kit's gain trimmed before recording to tape.
- **Sampler preset from a recorded take, verified 2026-09-26**: a 1.5 s region trimmed from an
  audition WAV, authored as a sampler preset (root C4, loop 0.3 to 1.2 s on the 6 s knob scale)
  and installed on key 8, played C4 at its root, G4 chromatically, and sustained a 3.5 s hold
  through the loop. It came out about 16 dB quiet because the clip was recorded at -22 dBFS and
  gain stayed at unity: normalize when trimming, or raise the gain knob.
- Drum design note from Tyler: kits hold effects, shakers, brushes and toms as well as the core
  kick, snare and hats, so every one of the 24 keys is addressable in patterns by name or MIDI
  note, and kit maps carry free-form descriptions per key.

## 13. Drums, 2026-09-26

- Tyler's labels for drum slot 1 ("hard spunch", Field key names one octave below MIDI): F2 (53) kick,
  F#2 (54) kick with a little cymbal, G2 (55) snare with snares on, G#2 (56) snare with snares off,
  C3/D3/E3 (60, 62, 64) closed hats, C#3 (61) closed hat that chokes the open hat, D#3 (63) open hat.
- The kit classifier (src/op_bridge/kits.py), built from the seven Field-written sampler kits and the
  live capture of kit 1, agrees with all nine labels from the kit's file and labels the remaining
  fifteen keys (claps, cymbal, toms, percussion, a second snare family). Roles follow the audio; the
  factory layout only breaks ties. Kits hold much more than kick, snare and hats, so patterns can
  address every key by role, Field key name or MIDI number.
- Grid notation (src/op_bridge/drums.py): steps per bar, instrument rows, x/X/o/./r cells, list form
  with 1-based steps and ^ accents, chained sections ("A*2B"), swing applied on the grid to every
  second 8th or 16th step with hit lengths preserved.
- A composed two-bar beat played on kit 1 landed 30 of 33 hits within 16 ms; drum takes are judged
  by onsets, not pitch. The endless sequencer accepts steps over MIDI and plays them back from a held
  note; it is monophonic, so it takes one instrument row per pass.
- **Live kit map and a beat through the tools, 2026-09-26**: kit_map(1) auditioned all 24 keys and
  agreed with Tyler's nine labels 9/9; play_drums on kit 1 landed 34 of 36 hits by onset within 16 ms
  and Tyler judged the loop good. The device tests leave the Field in synth mode, which surprised
  Tyler: the Field cannot be queried, so the bridge now records what it last set (mode, slot, tempo)
  and reports it in get_status, and a MIDI monitor was run to learn what the Field transmits when the
  human changes mode, sound or transport by hand.
- **No state read-back, verified 2026-09-26**: a 90 s MIDI monitor while Tyler switched synth/drum,
  pressed sound keys, turned an encoder and used tape play/stop captured only the played keys as
  notes on the configured channel. The Field transmits nothing for mode, sound or transport changes
  in normal mode (with sync mode set to midi sync). The bridge records what it sets, accepts hand
  changes through set_device_state, and detect_mode guesses synth versus drum by ear (correct in
  both directions on this Field). Tools select the sound they need before playing.
- **Endless programmed through the tool, 2026-09-26**: endless_program entered eight steps (closed
  hat, open hat, choke hat by role from the kit map) into the enabled endless sequencer on kit 1 and a
  4-beat hold at 96 BPM played back 4 hits, a quarter-note grid, so the note value on the device was
  still 1/4; the entry and playback path works. The bridge stays the main drum machine.

## 14. Speech through the vocoder, 2026-09-26

- The Mac's built-in speech engine (`say`, offline, 74 voices) writes 44.1 kHz WAV; the bridge streams
  it into the Field's USB input while holding carrier chords over MIDI and records the result.
- Field setup by hand: a vocoder preset loaded, input source usb audio (shift + input, blue encoder),
  input switched on with the input key, input gain adjusted so the meter moves.
- Two separate audio streams on the same USB device stall each other in the audio library: the first
  takes were 0.07 s long. One full-duplex stream (DuplexRecorder) plays and records together and
  keeps them sample-aligned; with it the vocoded speech came back at -6 dB with no clipping.
- A vocoder with no modulator is silent, so a carrier-free tone probe cannot test the routing; the
  probe passes when the input is monitored, the real test is speech plus chords.
- **Arming is fragile, 2026-09-26 evening**: sending tape jump-to-start (CC 84) or MIDI start after
  the human armed a track cancelled the arm; three takes played without recording. Rewind before
  the human arms, then send nothing but the notes: the first incoming note starts the recording.

### Vocoder learnings from the second piece ("By Ear", an Opus session on 2026-09-28)

Measured with STOI (pystoi, the recorded track against the speech sent; repeatable to about 0.005):
- Intelligibility lives in **formant** (encoder 2) and **waveform** (encoder 1), not in mix. Formant peaks
  at neutral (56-68 → STOI 0.65-0.67; 0 → 0.34; 127 → 0.41). Waveform 0 gives the clearest words (0.74),
  127 the loudest. Bands (encoder 3) 120-127 (0.74). Mix (encoder 4) is a dry/wet blend of the raw voice:
  keep it at 0 unless a person talking over the synth is wanted; the dry voice goes to tape.
- The vocoder is silent without a modulator, so carrier chords can be held through a whole song and only
  sound under words; its output level barely follows carrier velocity or speech gain, the voice matters
  (Samantha clipped the main mix, Daniel sat 5 dB lower).
- macOS `say`: `[[slnc ms]]` works and the rendered file is sample-exact, so line timing can be designed
  offline; no sustained vowels (TUNE/PHON are read aloud, longest vowel 0.55 s); rates are rounded per
  voice (Daniel 120 = 140, 160 = 180); a 6 dB shelf above 1.8 kHz helps the ear but not STOI.
- Whisper cannot compare settings (it hallucinates); STOI can. Dry voice shows as gliding harmonics in
  a spectrogram, the vocoder's as flat lines on the carrier notes.
- Limits of speak_through_vocoder as it stands: it sends neither tempo nor clock, re-strikes repeated
  chords instead of holding them, trims the last chord to the speech, and a take longer than about 50 s
  exceeds a chat client's tool timeout, so that session streamed long takes from a script instead.
- Recipe: input usb audio, gain middle; waveform 0, formant 64, bands 120, mix 0; Daniel at rate 140;
  3-4 note carriers with a low note around Bb2-F3, velocity about 90; lines about 15 ms early.

## 15. First piece on tape, 2026-09-26

A 24-bar lo-fi hip hop piece at 78 BPM with vocoder vocals, all four tape tracks played by the
bridge (examples/lofi_vocoder_song.py). Lessons from the session:
- Sampler presets pitch correctly only near their root (base_freq): "suitcase" rooted at C5 played
  two octaves low as a slowed sample. Voice samplers near their root or use synthesis engines.
- After the human arms a track, send nothing but clock and notes: a tape jump, a MIDI start or a
  tempo change cancels the arm. Rewind and set the tempo before the human arms.
- Live pair follows the selected track: after recording track 3 the live synth sits on channels 7-8.
- Vocoder intelligibility: slower speech (rate about 130), fewer carrier notes, more bands, and
  hums looped from a short "mmm" between lines keep the pad audible; the vocoder passes a little
  carrier even without a modulator.
- Mix set over CC: track levels 100/110/104/120, keys panned left, vocals right, master EQ
  low 72 mid 60 high 46, drive 38 release 64. Bounced through the 10-channel USB capture.

## 16. Sounds, search and arrangement, 2026-09-27

- Playable ranges come from the slot files: samplers carry base_freq (C4 or C5 in the presets seen),
  synthesis engines pitch MIDI notes literally, the preset's octave setting only hints at register.
- Auditing every synth slot at three pitches and tagging by rules matched the ear: the piano-like
  dimension engine came out keys/pluck, the string engine bass-capable, baby string a pad/drone.
- Randomize (CC 62) produces widely different sounds on the same slot, roles included; the Field
  cannot recall a roll, so keeping one is a human snapshot (hold the sound key two seconds).
- Jev (TypeSafe.ai) is text-only and hosted: useful as a ranker over tags and features, never as a
  listener. It sits behind the judge interface with the local rules as the default.
- Jev (jev-1.13) as the judge: on the audited palette it ranked the string engine first for a dark sustained bass (0.85) and baby string first for a slow-moving pad (0.87), with 0.95 versus 0.01 on a clear positive and negative. Rankings cost input tokens only; the key lives in ~/Music/op-bridge/secrets.env (owner-only) written by scripts/set-secret.sh.

## 17. Effects and the judge, 2026-09-27

- Effects layer: ten effect types with knob names from the manual (fazer's knobs inferred), one-line
  characters, named-knob setting for the loaded patch (CC 54-57) and the master bus (CC 70-73 once the
  human names the type), and the effect type read from slot files into the sound index. Named edits
  verified on the terminal effect of key 2: extreme settings erase the sound, moderate ones colour it.
- Judge modes: auto (Jev when a key is stored), local (never calls out), jev (requires the key); Jev
  answers a cached taxonomy (role, character, envelope, register) once per sound plus a rubric match per
  request, combined by weights in judge.py. Deep measurements add odd/even balance, richness, tilt,
  inharmonicity, transient, brightness change and detrended modulation rates.
- 2026-09-27: spring and fazer learned from Tyler's slots 1 and 3 (all ten effects now authorable); the vocoder engine identifier is `vocoder`, learned from slot 8.
- **Fazer's page carries no text**: it is a pictographic bonus screen (a figure, a ray gun, a UFO and a
  panel of question marks), so its knob names cannot be read from the device; the manual has them (page 24). Tyler describes it as a
  fairly traditional phaser. The bridge names its knobs from measurement (below) rather than from labels.
- **Live edits persist across program changes.** With key 3 loaded, an FX knob moved over CC stayed moved
  after a program change to the same slot and after switching to key 4 and back. A randomize (CC 62)
  likewise stayed. So a program change selects a slot's *current* state, not its saved file; the bridge
  must set every knob it relies on explicitly and cannot assume a fresh patch.
- **CC 63 (reset active patch) did nothing** in five trials on firmware 1.7: after an FX knob edit and after a
  randomize, with values 64 and 127, with a note held, and on another channel; the sound stayed changed each
  time. Randomize (CC 62) works. The only revert is the shift + synth (or drum) shortcut on the device.
- **Randomize reaches past MIDI**: after a randomize, restoring the engine, envelope and FX knobs over CC
  (46-57) did not bring the saved sound back (level and a 1 Hz sweep remained), so randomize also changes
  state no CC addresses (effect or LFO type and activity, the shift-layer knob values). Treat it as
  destructive to the live slot until the human reverts.

### Fazer measured, 2026-09-27

The fazer's device page has no labels; the manual does name them on page 24 (frequency, feedback, speed,
depth), which the first pass at the effects catalogue missed because the text extraction interleaves the
pages' columns. The four encoders were measured anyway, and the measurement agrees with the manual's order. Test bed: an authored
`voltage` preset with the fazer active and no LFO installed in slot 3 (Tyler's cluster preset backed up to
`field-backup/2026-09-27-132257-replaced`; a cluster engine moves on its own at 0.6 to 2 Hz, which had
swamped the first attempt). Each take set all four knobs explicitly (nothing reloads a patch), held C4 for
5 s (14 s for the slow rates) and tracked the level of the first 24 harmonics over time; a phaser shows as
periodic dips walking across harmonics, and the dips' rate, depth, spread and the mean levels were read
per knob. The method lives in `analysis.modulation_rate` and `scripts/fx_knob_sweep.py`, which sweeps any
effect knob the same way; `tests/test_effects.py` re-checks the rate table on the device.

- **CC 54, blue = frequency.** Moves the notch region. At 0 the notches sweep over the fundamental (C4's
  fundamental swings 11 dB and sits 13 dB lower on average); at 127 the fundamental and second harmonic
  are steady and loud while the harmonics above still sweep. Low values thin the bass, high values keep it.
- **CC 55, green = feedback.** At 0 the notches are shallow (median harmonic swing 5.5 dB, fundamental 5 dB
  quieter); at 127 they are deep and resonant (swing 18 dB across all 24 harmonics, highs 2 dB louder).
  With the rate fast, feedback 0 still leaves a faint sweep, unlike mix 0.
- **CC 56, white = speed** (the bridge also accepts `rate`). Measured sweep rates: 0 to 42 → 0.14 to 0.18 Hz (a 6 to 7 s cycle), 64 → 0.65 Hz,
  85 → 2.0 Hz, 106 → 7.8 Hz, 117 → 20 Hz, 127 → 69 Hz (audio-rate flutter, the ray gun). Exponential above
  the middle, flat below it.
- **CC 57, orange = depth** (also `mix`). At 0 no periodic modulation remains at any rate and the level is unchanged (dry);
  the sweep depth grows steadily to 127 (fundamental swing 3.5 → 13 dB) with no level change.
- Musical reading: speed 64-80 with depth 60-90 and feedback 40-70 is the classic slow phaser; frequency
  below 40 makes bass and keys hollow, above 90 keeps low end solid; speed above 110 turns it into a tremolo
  or ring-mod texture.
- The same page read also fixed two catalogue slips: punch's order is frequency, punch, rounds, power, and
  grid's first knob is x size. Nitro's two frequency knobs are LOWS and HIGHS on its screen.

## 18. Background jobs, 2026-09-28

Chat clients cut a tool call off after roughly a minute, which the "By Ear" session hit with long takes and
worked around with its own scripts. The server now runs anything expected to exceed 25 s (env
`OP_BRIDGE_SYNC_LIMIT`) on a daemon thread and answers with a job id; `job_status` returns progress and, once
done, the identical result of a direct call; `cancel_job` stops playback between events (the player polls
the flag every 0.2 s inside long sleeps), the device context releases the notes it holds, a tape recording is
stopped and a seed capture keeps what it heard. One lock guards the device: a call that cannot take it within
3 s fails at once naming the job that holds it. Verified on the Field with `tests/test_jobs.py`.

## 19. Measurement tools, batch audition and the vocoder tool, 2026-09-28

- `measure_take(id)` returns level, six band levels (sub, bass, low_mid, mid, presence, air, calibrated so
  they sum to the signal power), a pitch track summary with vibrato rate and depth, the first twelve harmonic
  levels, periodic movement by harmonic tracking, onsets, and writes a log-frequency spectrogram
  (`view_spectrogram(id, log_frequency=true)`, 30 Hz to 8 kHz with gridlines). These replace the private
  scripts the "By Ear" session had to write.
- `speech_intelligibility(id)` is STOI (pystoi) of a take against the speech that was sent, aligned by level
  envelopes (a 250 ms late copy scores above 0.95 at the right lag; unrelated noise floors near 0.4). The
  vocoder tool now reports it with every recorded take.
- `speak_through_vocoder` takes `lines` placed at beats or seconds, holds repeated chords, keeps the last
  chord to the end (or to `min_beats`), can send tempo and clock, and records to the armed track with
  `to_tape`; speech longer than the job limit runs as a job. Verified on the Field: two lines at beats 0 and 2
  over held chords, with the combined speech file kept beside the take.
- `audition_slots(["synth 1", "synth 3"])` auditions a list in one call and returns one line per slot with the
  palette note and index tags; long lists run as a job.
- The overview no longer presents the vocoder as the way to words; get_guide("quickstart") is the short entry.
- `stream_to_tape(wav_path | text, start="play"|"now")` streams audio from the Mac into the Field's USB input
  while the tape records, playing no notes: with the track armed, tape play (CC 105) starts the Field's
  count-in and the audio follows it; the USB return is kept with the lag between sent and heard audio. The
  streaming path works (the tool ran with start="now"; the Field returned nothing because the input was off).
  **First test, 2026-09-28: nothing recorded, but the documented procedure was not followed.** With the input
  switched on from the synth side, a new tape, track 4 armed with shift + record and play pressed by hand
  (count-in off), a streamed spoken line left the track at -73 dB and the USB return at the noise floor; the
  same stream drove the vocoder on key 8 to STOI 0.69, so the stream and the input are fine. TE's guide for
  recording external sources (written for the original OP-1, usb audio included) says: select usb as input
  (shift + mic), press tape, scrub to the spot, **press mic in tape mode to toggle the external audio on**,
  then **hold record and press play**; in tape mode the mic key toggles external audio rather than sampling.
  **Retested that way the same day: it works.** With the input key pressed in tape mode, the streamed line came
  back through the main mix at -4 dB, STOI 0.99, 200 ms after leaving the Mac (the Mac → Field → USB round
  trip; the vocoder shows the same ~200 ms) and Tyler heard it from the Field; with record + play pressed by
  hand the line landed on track 4 at -25 dB with STOI 0.94 (it sat 18.5 s into the tape because that is where
  the head was; a backup must be long enough to reach it). So the field keeps the OP-1 rule: the input key in
  tape mode routes external audio to the mix and the tape, and recording it is record + play.
- Latency of the Mac → Field path, measured the same day: about 200 ms on the dry input (monitored return
  against the sent file) and about 100 ms through the vocoder (vocoded words against their beats). The
  vocoder tool now sends speech `input_latency_ms` early (default 100) and reports `on_beat_error_ms`: with
  the default it is +6 to +9 ms, with no compensation +97 ms, with 200 ms the words arrived 101 ms early and
  lost their first syllable before the carrier started. Lesson recorded: read
  the manual and TE's online guides before an exploratory device test. Play over MIDI (CC 105) does not
  start an armed take.
