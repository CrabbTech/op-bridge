"""Bounded Google audio requests. Standard library only; also runs as a child process."""
from __future__ import annotations

import base64
import datetime
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

MODELS = ("gemini-3.8-flash", "gemini-3.1-pro-preview")
DEFAULT_MODEL = "gemini-3.1-pro-preview"
PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
MAX_RESPONSE_BYTES = 262144


def api_key(home):
    """Re-read the saved key each time, including after rotation in a live MCP server."""
    path = Path(home) / "secrets.env"
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        lines = []
    except (OSError, UnicodeError):
        raise RuntimeError("Cannot read op-bridge secrets.env") from None
    saved = None
    for line in lines:
        key, separator, value = line.strip().partition("=")
        if separator and key.strip() == "GEMINI_API_KEY":
            saved = value.strip().strip('"').strip("'")
    # The file takes precedence over session.load_secrets' possibly stale environment.
    key = saved if saved is not None else os.environ.get("GEMINI_API_KEY", "")
    if not key or any(c.isspace() for c in key):
        raise RuntimeError("Save GEMINI_API_KEY in op-bridge secrets.env or its environment")
    return key


def status(home):
    try:
        api_key(home)
        configured, reason = True, None
    except RuntimeError as error:
        configured, reason = False, str(error)
    return {"backend": "gemini", "configured": configured, "authenticated": None,
            "local_only": False, "provider": "Google Gemini API", "default_model": DEFAULT_MODEL,
            "models": list(MODELS), "max_excerpt_seconds": 30, "reason": reason,
            "note": "No network request made. Explicit backend='gemini' uploads the excerpt to Google and may incur charges."}


def estimate_cost(model, usage, today=None):
    """Published standard-tier rates, not an account bill; no caching or tools are used."""
    today = today or datetime.date.today()
    rates = (2.0, 12.0) if model == "gemini-3.1-pro-preview" else (
        (0.75, 3.75) if today < datetime.date(2027, 1, 1) else (1.5, 7.5))
    prompt = usage.get("promptTokenCount")
    output = usage.get("candidatesTokenCount", 0)
    thinking = usage.get("thoughtsTokenCount", 0)
    total = usage.get("totalTokenCount")
    if any(type(v) is not int or v < 0 for v in (prompt, output, thinking, total)):
        return None
    # Include hidden thinking or other output tokens even if their breakdown is absent.
    billable_output = max(output + thinking, total - prompt)
    return {"usd": round((prompt * rates[0] + billable_output * rates[1]) / 1_000_000, 8),
            "input_tokens": prompt, "output_including_thinking_tokens": billable_output,
            "input_usd_per_million": rates[0], "output_usd_per_million": rates[1],
            "pricing_checked": "2026-09-29", "pricing_url": PRICING_URL,
            "note": "Estimate from published standard-tier rates; excludes tax and account-specific credits."}


def parse_response(payload, model):
    if not isinstance(payload, dict) or not isinstance(payload.get("usageMetadata", {}), dict):
        raise RuntimeError("Google returned an invalid audio response")
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or len(candidates) > 1:
        raise RuntimeError("Google returned an unexpected number of candidates")
    candidate = candidates[0] if candidates else {}
    if not isinstance(candidate, dict):
        raise RuntimeError("Google returned an invalid audio candidate")
    content = candidate.get("content", {})
    if not isinstance(content, dict) or not isinstance(content.get("parts", []), list):
        raise RuntimeError("Google returned invalid audio content")
    parts = content.get("parts", [])
    if any(not isinstance(p, dict) for p in parts):
        raise RuntimeError("Google returned invalid audio parts")
    observations = "\n".join(p["text"] for p in parts
                             if not p.get("thought") and isinstance(p.get("text"), str)).strip()
    finish = candidate.get("finishReason", "NO_CANDIDATE")
    usable = finish == "STOP" and bool(observations)
    usage = payload.get("usageMetadata", {})
    return {"backend": "gemini", "provider": "Google Gemini API", "model_id": model,
            "model_version": payload.get("modelVersion"), "observations": observations,
            "usable": usable, "finish_reason": finish, "usage": usage,
            "estimated_cost": estimate_cost(model, usage),
            "assessment": ("Fallible audio-model observations; verify consequential musical claims."
                           if usable else "No complete description. Partial output is not a reliable review; no automatic retry was made.")}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def generate(home, excerpt, question, model, max_tokens, timeout_seconds):
    if model not in MODELS or type(max_tokens) is not int or not 256 <= max_tokens <= 4096:
        raise ValueError("Unsupported Gemini model or output budget")
    audio = Path(excerpt).read_bytes()
    if len(audio) > 2_000_000:
        raise ValueError("Gemini excerpt exceeds the upload limit")
    payload = {"contents": [{"role": "user", "parts": [
        {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(audio).decode("ascii")}},
        {"text": question}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "candidateCount": 1,
                             "thinkingConfig": {"thinkingLevel": "LOW"}}}
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent",
        data=json.dumps(payload).encode(),
        headers={"x-goog-api-key": api_key(home), "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout_seconds) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        hints = {400: "Check key validity and model/request support.",
                 401: "Check the saved API key.", 403: "Check the key's project permissions.",
                 404: "The selected model is unavailable.",
                 429: "Check the project's billing, quota or rate limit."}
        raise RuntimeError(f"Google audio request failed (HTTP {error.code}). "
                           + hints.get(error.code, "No automatic retry was made.")) from None
    except (urllib.error.URLError, OSError, ValueError):
        # Never echo request objects, authentication headers or arbitrary provider error bodies.
        raise RuntimeError("Google audio network request failed or timed out; no automatic retry was made") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Google audio response exceeded the size limit")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError):
        raise RuntimeError("Google returned invalid JSON") from None
    return parse_response(payload, model)


def main():
    try:
        request = json.load(sys.stdin)
        result = generate(request["home"], request["audio_path"], request["question"],
                          request["model"], request["max_tokens"], request["timeout_seconds"])
        print(json.dumps({"type": "result", **result}, allow_nan=False), flush=True)
    except (RuntimeError, ValueError) as error:
        print(json.dumps({"type": "error", "message": str(error)}), flush=True)
    except Exception:
        print(json.dumps({"type": "error", "message": "Google audio adapter failed"}), flush=True)


if __name__ == "__main__":
    main()
