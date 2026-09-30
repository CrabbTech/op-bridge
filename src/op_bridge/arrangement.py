"""A basic node-based arrangement layer.

An arrangement is a graph. Section nodes carry a length in bars. Part nodes bind a sound (kind and slot) and a tape
track to one clip per section. Automation nodes add CC curves to a part over chosen sections. Edges (or an explicit
order) give the path through the sections, and a section may appear more than once. Compiling walks the path and
produces one full-length Score per part, ready for play, record_to_tape or the drum tools, plus a timeline.

    {
      "tempo": 78, "beats_per_bar": 4,
      "nodes": {
        "intro": {"type": "section", "bars": 4},
        "A":     {"type": "section", "bars": 8},
        "B":     {"type": "section", "bars": 8},
        "keys":  {"type": "part", "sound": {"kind": "synth", "slot": 2}, "track": 2,
                  "clips": {"intro": {"score": {...}}, "A": {"chords": [...]}, "B": {"as": "A", "transpose": 0}}},
        "drums": {"type": "part", "sound": {"kind": "drum", "slot": 1}, "track": 1,
                  "clips": {"A": {"pattern": {...}}, "B": {"pattern": {...}}}},
        "sweep": {"type": "automation", "part": "keys", "param": "fx3", "sections": {"B": [[0, 40], [32, 90]]}}
      },
      "order": ["intro", "A", "B", "A", "intro"]        // or "edges": [["intro","A"],["A","B"],["B","A"]] walked from "intro"
    }

Clip forms: {"score": <score document with beats relative to the section start>}, {"pattern": <drum grid document>},
{"chords": [{"notes": ["C3","E3","G3"], "beats": 4, "velocity": 80, "rhythm": "x...x.x."}, ...]},
{"as": "<other section>", "transpose": <semitones>} to reuse a clip, or {"rest": true}. A clip shorter than its
section repeats to fill it unless "loop": false; a longer clip is an error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .score import Score, Note, Automation, pitch_to_midi, midi_to_name
from . import drums as DR


@dataclass
class Section:
    name: str
    bars: int
    start_beat: float = 0.0
    beats: float = 0.0


@dataclass
class CompiledPart:
    name: str
    sound: dict[str, Any]
    track: int | None
    score: Score
    clips_used: list[str]
    warnings: list[str] = field(default_factory=list)


def _order(doc: dict[str, Any], sections: dict[str, dict[str, Any]]) -> list[str]:
    if doc.get("order"):
        order = [str(s) for s in doc["order"]]
    elif doc.get("edges"):
        nxt: dict[str, list[str]] = {}
        for a, b in doc["edges"]:
            nxt.setdefault(str(a), []).append(str(b))
        start = str(doc.get("start") or doc["edges"][0][0])
        order = [start]; seen_edges: set[tuple[str, str]] = set()
        cur = start
        while True:
            outs = [b for b in nxt.get(cur, []) if (cur, b) not in seen_edges]
            if not outs:
                break
            seen_edges.add((cur, outs[0])); cur = outs[0]; order.append(cur)
            if len(order) > 256:
                raise ValueError("the edge walk did not terminate; use an explicit order list")
    else:
        order = list(sections)
    unknown = [s for s in order if s not in sections]
    if unknown:
        raise ValueError(f"order refers to sections that are not section nodes: {unknown}; sections: {list(sections)}")
    return order


def _chords_to_notes(spec: list[dict[str, Any]], beats_per_bar: int) -> tuple[list[dict[str, Any]], float]:
    notes: list[dict[str, Any]] = []
    beat = 0.0
    for ch in spec:
        pitches = ch.get("notes") or ch.get("chord") or []
        length = float(ch.get("beats", beats_per_bar))
        vel = int(ch.get("velocity", 84))
        rhythm = ch.get("rhythm")
        if rhythm:
            cells = [c for c in str(rhythm) if c in "x.X"]
            step = length / max(1, len(cells))
            for i, c in enumerate(cells):
                if c == ".":
                    continue
                for p in pitches:
                    notes.append({"start": beat + i * step, "duration": max(0.1, step * 0.9), "pitch": p, "velocity": min(127, vel + (16 if c == "X" else 0))})
        else:
            for p in pitches:
                notes.append({"start": beat, "duration": max(0.1, length - 0.05), "pitch": p, "velocity": vel})
        beat += length
    return notes, beat


def _clip_score(clip: dict[str, Any], section: Section, doc: dict[str, Any], part_clips: dict[str, Any], depth: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """(notes, cc automations, warnings) for one clip, beats relative to the section start, filling the section."""
    warnings: list[str] = []
    tempo = float(doc.get("tempo", 100)); bpb = int(doc.get("beats_per_bar", 4))
    if clip.get("rest"):
        return [], [], warnings
    if "as" in clip:
        if depth > 4:
            raise ValueError("clip reuse chain too deep")
        src = part_clips.get(clip["as"])
        if src is None:
            raise ValueError(f"clip refers to section {clip['as']!r} which this part has no clip for")
        notes, ccs, w = _clip_score(src, section, doc, part_clips, depth + 1)
        t = int(clip.get("transpose", 0))
        if t:
            notes = [{**n, "pitch": pitch_to_midi(n["pitch"]) + t} for n in notes]
        return notes, ccs, warnings + w
    if "pattern" in clip:
        pat = dict(clip["pattern"]); pat.setdefault("tempo", tempo); pat.setdefault("beats_per_bar", bpb)
        sc = DR.compile_drums(pat)
        notes = [n.model_dump() for n in sc.notes]
        length = float(sc.length_beats or (max((n.start + n.duration) for n in sc.notes) if sc.notes else 0))
        ccs: list[dict[str, Any]] = []
    elif "chords" in clip:
        notes, length = _chords_to_notes(clip["chords"], bpb)
        ccs = []
    elif "score" in clip:
        sc = Score.model_validate(dict(clip["score"], tempo=tempo))
        notes = [n.model_dump() for n in sc.notes]
        ccs = [a.model_dump() for a in sc.cc]
        length = float(sc.length_beats or sc.end_beat())
    else:
        raise ValueError(f"clip for section {section.name!r} needs one of score, pattern, chords, as or rest")
    if length <= 0:
        return [], [], warnings
    if length > section.beats + 1e-6:
        raise ValueError(f"clip for section {section.name!r} is {length:.2f} beats but the section is {section.beats:.2f}; shorten the clip or lengthen the section")
    if clip.get("loop", True) and length < section.beats - 1e-6:
        reps = int(section.beats // length)
        out = []
        for k in range(reps):
            out += [{**n, "start": float(n["start"]) + k * length} for n in notes if float(n["start"]) + k * length < section.beats]
        if reps * length < section.beats - 1e-6:
            warnings.append(f"section {section.name!r}: clip of {length:g} beats loops {reps} times and leaves {section.beats - reps * length:g} beats empty")
        notes = out
        ccs = [{**a, "points": [[float(b) + k * length, v] for k in range(reps) for b, v in a["points"] if float(b) + k * length < section.beats]} for a in ccs]
    return notes, ccs, warnings


def compile_arrangement(doc: dict[str, Any]) -> dict[str, Any]:
    nodes = doc.get("nodes") or {}
    if not nodes:
        raise ValueError("an arrangement needs nodes")
    tempo = float(doc.get("tempo", 100)); bpb = int(doc.get("beats_per_bar", 4))
    sections = {n: v for n, v in nodes.items() if v.get("type") == "section"}
    parts = {n: v for n, v in nodes.items() if v.get("type") == "part"}
    autos = {n: v for n, v in nodes.items() if v.get("type") == "automation"}
    other = [n for n, v in nodes.items() if v.get("type") not in ("section", "part", "automation")]
    if other:
        raise ValueError(f"unknown node types on {other}; use section, part or automation")
    if not sections or not parts:
        raise ValueError("an arrangement needs at least one section node and one part node")
    order = _order(doc, sections)
    timeline: list[Section] = []
    beat = 0.0
    for name in order:
        bars = int(sections[name].get("bars", 4))
        s = Section(name=name, bars=bars, start_beat=beat, beats=bars * bpb)
        timeline.append(s); beat += s.beats
    total_beats = beat
    compiled: dict[str, CompiledPart] = {}
    for pname, part in parts.items():
        clips = part.get("clips") or {}
        notes: list[dict[str, Any]] = []
        ccs: list[dict[str, Any]] = []
        used: list[str] = []
        warnings: list[str] = []
        for s in timeline:
            clip = clips.get(s.name)
            if clip is None:
                continue
            n, c, w = _clip_score(clip, s, doc, clips)
            notes += [{**x, "start": float(x["start"]) + s.start_beat} for x in n]
            ccs += [{**a, "points": [[float(b) + s.start_beat, v] for b, v in a["points"]]} for a in c]
            used.append(s.name); warnings += w
        for aname, auto in autos.items():
            if auto.get("part") != pname:
                continue
            pts: list[list[float]] = []
            for s in timeline:
                seg = (auto.get("sections") or {}).get(s.name)
                if seg:
                    pts += [[float(b) + s.start_beat, float(v)] for b, v in seg]
            if pts:
                ccs.append({"param": auto["param"], "points": sorted(pts), "step": bool(auto.get("step", False))})
        feel = part.get("feel") or {}
        score = Score(tempo=tempo, beats_per_bar=bpb, notes=notes, cc=ccs, swing=float(feel.get("swing", 0)), humanize_ms=float(feel.get("humanize_ms", 0)),
                      humanize_velocity=int(feel.get("humanize_velocity", 0)), seed=feel.get("seed"), length_beats=total_beats, tail_seconds=float(part.get("tail_seconds", 1.5)), send_clock=True)
        compiled[pname] = CompiledPart(name=pname, sound=part.get("sound") or {}, track=part.get("track"), score=score, clips_used=used, warnings=warnings)
    lines = [f"{'section':10s} {'bars':>4s} {'start':>7s}  parts"]
    for s in timeline:
        lines.append(f"{s.name:10s} {s.bars:4d} {s.start_beat / bpb + 1:7.1f}  " + ", ".join(p for p, c in compiled.items() if s.name in (parts[p].get('clips') or {})))
    return {"tempo": tempo, "beats_per_bar": bpb, "order": order, "total_bars": int(total_beats // bpb), "seconds": round(total_beats * 60 / tempo, 1),
            "timeline": "\n".join(lines), "parts": compiled}
