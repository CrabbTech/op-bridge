"""Ranking judges, composed in code.

The local judge is rule based and always available. The Jev judge (TypeSafe.ai, text only, hosted) answers typed
questions about a sound's descriptors and measurements. Two kinds of questions are used:

* taxonomy questions that do not depend on the request (role, character, envelope class, register): asked once per
  sound and cached in the sound index;
* one match question that embeds the request, scored on a concrete rubric.

The ranking is a weighted composite of the match score and the taxonomy probabilities the request implies. The
weights, the rubric and the option descriptions are all here, where they can be read and tuned."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from .sounds import local_rank

JEV_URL = "https://api.typesafe.ai/v1/systemone"

LEGEND = {
    "attack_ms": "time from note-on to the level peak; under 15 is instant, over 150 is a swell",
    "sustain_ratio_db": "level at the end of a 1 second hold minus the peak; near 0 sustains, below -20 has died away",
    "release_ms": "time to fall 40 dB after note-off",
    "centroid_hz": "spectral centre of mass: under 400 dark, 400-1200 warm, 1200-3000 bright, above harsh or glassy",
    "odd_even_db": "odd minus even harmonic energy: above +6 hollow like a square or clarinet, below -3 even-rich",
    "harmonics_above_minus40db": "how many harmonics are audible: 1-2 pure, 8+ rich",
    "inharmonic_share": "energy between harmonics: above 0.2 bell-like, above 0.35 metallic or FM",
    "flatness": "noise-likeness of the spectrum: above 0.08 noisy",
    "brightness_change": "sustain brightness over attack brightness: below 0.6 a filter closes, above 1.6 it opens",
    "level_mod_hz": "rate of any tremolo or pulsing", "brightness_mod_hz": "rate of any wobble or filter sweep",
    "width": "stereo width 0-1", "bands": "energy shares: sub <100 Hz, low 100-400, mid 400-2k, high 2k-8k, air >8k",
}

TAXONOMY = {
    "role": {"type": "choice", "instructions": "What musical role does this sound serve best?",
             "criteria": {"bass": "low register, strong fundamental, holds a bassline", "keys": "chordal comping, plucked or struck, decays",
                          "pad": "sustained, soft attack, sits behind a mix", "lead": "sustained and forward, carries a melody",
                          "pluck": "very short bright hit, arpeggios", "drone": "sustained with slow movement, texture",
                          "fx": "noisy, inharmonic or unpitched, sound design", "percussive": "short hit with no sustain"}},
    "character": {"type": "choice", "instructions": "How would a musician describe its tone?",
                  "criteria": {"dark": "dull, little high end", "warm": "rounded, mellow, present mids", "bright": "clear highs, open",
                               "harsh": "aggressive, buzzy or grating highs", "glassy": "pure, bell-like clarity", "gritty": "noisy, distorted or lo-fi"}},
    "envelope": {"type": "choice", "instructions": "How does its loudness evolve while a key is held?",
                 "criteria": {"plucked": "instant attack, quick decay", "struck": "instant attack, medium decay", "sustained": "holds steady",
                              "swelling": "fades in slowly", "pulsing": "throbs or wobbles"}},
    "register": {"type": "choice", "instructions": "Where does it sit best?",
                 "criteria": {"low": "bass register", "mid": "middle register", "high": "upper register", "wide": "works across registers"}},
}

MATCH_RUBRIC = ["does not fit the request at all", "shares one trait with the request but the overall character is wrong",
                "partly fits: right role or right tone, not both", "fits well with a small mismatch", "exactly what the request describes"]

WEIGHTS = {"match": 0.55, "role": 0.2, "character": 0.15, "envelope": 0.1}

REQUEST_HINTS = {
    "role": {"bass": ["bass", "sub", "808", "low end"], "keys": ["keys", "rhodes", "piano", "comp", "chords", "organ"], "pad": ["pad", "strings", "ambient", "bed"],
             "lead": ["lead", "melody", "solo"], "pluck": ["pluck", "arp", "arpeggio", "stab"], "drone": ["drone", "texture"], "fx": ["fx", "noise", "riser", "sound design"],
             "percussive": ["hit", "percussive", "perc"]},
    "character": {"dark": ["dark", "dull", "muffled"], "warm": ["warm", "mellow", "soft", "lo-fi", "lofi", "vintage"], "bright": ["bright", "clear", "shiny"],
                  "harsh": ["harsh", "aggressive", "buzzy", "acid"], "glassy": ["glassy", "bell", "crystal"], "gritty": ["gritty", "dirty", "distorted", "crunchy"]},
    "envelope": {"plucked": ["pluck", "plucky", "short"], "struck": ["struck", "piano", "keys"], "sustained": ["sustained", "held", "long", "pad", "drone"],
                 "swelling": ["swell", "slow attack", "fade in"], "pulsing": ["pulsing", "wobble", "tremolo", "throb"]},
}


def judge_mode() -> str:
    try:
        from .session import Config
        mode = Config.load().judge
    except Exception:
        mode = "auto"
    return mode if mode in ("auto", "local", "jev") else "auto"


def jev_available() -> bool:
    """True when Jev may be called: a key is present and the configured mode allows it."""
    mode = judge_mode()
    if mode == "local":
        return False
    has_key = bool(os.environ.get("TYPESAFE_API_KEY"))
    if mode == "jev" and not has_key:
        raise RuntimeError("judge mode is 'jev' but TYPESAFE_API_KEY is not set; store it with scripts/set-secret.sh or switch the mode with: op-bridge judge auto")
    return has_key


def active_judge() -> str:
    try:
        return "jev" if jev_available() else "local"
    except RuntimeError:
        return "jev (key missing)"


def _state(candidate: dict[str, Any]) -> dict[str, Any]:
    keep = {k: v for k, v in candidate.items() if k in ("engine", "name", "root", "playable", "tags", "roles", "descriptors", "summary", "measured", "per_register", "human_note", "metadata_source")}
    keep["legend"] = LEGEND
    return keep


def jev_ask(candidate: dict[str, Any], questions: dict[str, Any], timeout: float = 15.0) -> dict[str, Any] | None:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        return None
    body = {"state": _state(candidate), "model": os.environ.get("TYPESAFE_MODEL", "jev-latest"), "questions": questions}
    req = urllib.request.Request(JEV_URL, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r).get("answers")
    except Exception:
        return None


def jev_taxonomy(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Request-independent classification, meant to be cached in the index: {question: {choice, confidence, probabilities}}."""
    ans = jev_ask(candidate, TAXONOMY)
    if not ans:
        return None
    return {q: {"choice": a.get("choice"), "confidence": a.get("confidence"), "probabilities": a.get("probabilities")} for q, a in ans.items()}


def jev_match(candidate: dict[str, Any], request: str) -> tuple[float, float] | None:
    """(score 0-1, confidence) for how well the candidate fits the request."""
    q = {"match": {"type": "score", "instructions": f"How well does this sound fit the request: \"{request}\"? Judge from its descriptors, measurements, engine and name.", "criteria": MATCH_RUBRIC}}
    ans = jev_ask(candidate, q)
    if not ans or "match" not in ans:
        return None
    a = ans["match"]
    return float(a.get("score", 0.0)) / (len(MATCH_RUBRIC) - 1), float(a.get("confidence", 0.0))


def implied(request: str) -> dict[str, str]:
    """Which taxonomy options a plain request asks for, by keyword; missing dimensions carry no weight."""
    q = request.lower()
    out: dict[str, str] = {}
    for dim, opts in REQUEST_HINTS.items():
        for opt, words in opts.items():
            if any(w in q for w in words):
                out[dim] = opt; break
    return out


def composite(match: float | None, taxonomy: dict[str, Any] | None, wants: dict[str, str]) -> float:
    """Weighted combination; weights of missing pieces are redistributed to what is available."""
    parts: dict[str, float] = {}
    if match is not None:
        parts["match"] = match
    if taxonomy:
        for dim, opt in wants.items():
            probs = (taxonomy.get(dim) or {}).get("probabilities") or {}
            if probs:
                parts[dim] = float(probs.get(opt, 0.0))
    if not parts:
        return 0.0
    total_w = sum(WEIGHTS[k] for k in parts)
    return sum(WEIGHTS[k] * v for k, v in parts.items()) / total_w


def jev_score(candidate: dict[str, Any], criteria: str) -> float | None:
    """One-number fit used by tests and simple callers: the composite of a fresh match and any cached taxonomy."""
    m = jev_match(candidate, criteria)
    if m is None:
        return None
    return composite(m[0], candidate.get("taxonomy"), implied(criteria))


def rank(candidates: list[dict[str, Any]], criteria: str, top: int = 5) -> dict[str, Any]:
    """Best candidates for a request: {"judge": "jev"|"local", "wants": {...}, "ranked": [(score, candidate)...]}."""
    wants = implied(criteria)
    if jev_available() and candidates:
        scored = []
        ok = True
        for c in candidates:
            m = jev_match(c, criteria)
            if m is None:
                ok = False; break
            scored.append((composite(m[0], c.get("taxonomy"), wants), c))
        if ok:
            scored.sort(key=lambda x: -x[0])
            return {"judge": "jev", "wants": wants, "ranked": scored[:top]}
    ranked = local_rank(candidates, criteria)
    return {"judge": "local", "wants": wants, "ranked": ranked[:top]}
