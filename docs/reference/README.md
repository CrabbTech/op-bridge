# Reference material

The bridge can answer questions about the device from Teenage Engineering's own documentation
(`search_manual`, `get_guide("manual")`, `get_guide("midi")`). That documentation belongs to
Teenage Engineering and is not distributed with op-bridge: add your own copies here and the tools
pick them up. Without them the tools say where to get them, and everything else works as usual.

| File | What it is | How to add it |
|---|---|---|
| `op-1-field-user-guide-fw1.7.txt` | The official user guide, firmware 1.7 edition, as text with a `===== PAGE n =====` line before each page | Download the PDF ("download as PDF" on the web guide), then `uv run scripts/extract-manual.py path/to/guide.pdf` |
| `midi-reference-fw1.7.0.txt` | The incoming MIDI message and CC tables from the web guide, firmware 1.7.0 | Copy the tables from the web guide's MIDI section into this file as plain text |
| `firmware-changelog.txt` | Release notes from the downloads page. Useful for knowing which feature arrived when (multichannel USB audio in 1.6.0, the CC map in 1.7.0) | Optional; copy from the downloads page |

Source pages: https://teenage.engineering/guides/op-1 and
https://teenage.engineering/downloads/op-1/field

These files are listed in `.gitignore` so a local copy never ends up in a commit.

`../device.md` is our own brief, including what was verified on the real device. It is part of
the repository and is what `get_guide("device")` returns.
