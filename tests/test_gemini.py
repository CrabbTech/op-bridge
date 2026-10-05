"""Google listener tests. All requests are mocked; no network or audio device."""
import datetime
import io
import json
from pathlib import Path
import urllib.error

import numpy as np
import pytest
import soundfile as sf

from op_bridge import gemini as G, listener as L


def payload(finish="STOP", text="A soft repeated pulse."):
    return {"candidates": [{"content": {"parts": [
        {"text": "private reasoning", "thought": True}, {"text": text}]}, "finishReason": finish}],
        "modelVersion": "test-version", "usageMetadata": {
            "promptTokenCount": 400, "candidatesTokenCount": 100,
            "thoughtsTokenCount": 500, "totalTokenCount": 1000}}


def test_rotated_file_key_overrides_stale_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "stale-test-key")
    assert G.api_key(tmp_path) == "stale-test-key"
    secret = tmp_path / "secrets.env"
    secret.write_text('# credentials\nOTHER=value\nGEMINI_API_KEY="first-test-key"\n')
    assert G.api_key(tmp_path) == "first-test-key"
    secret.write_text("GEMINI_API_KEY=second-test-key\n")
    assert G.api_key(tmp_path) == "second-test-key"
    secret.write_text("GEMINI_API_KEY=\n")
    with pytest.raises(RuntimeError):
        G.api_key(tmp_path)


def test_status_never_calls_network_or_exposes_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-marker")
    monkeypatch.setattr(G, "generate", lambda *a: pytest.fail("status must not contact Google"))
    status = L.status(tmp_path, backend="gemini")
    assert status["configured"] and status["authenticated"] is None
    assert "secret-test-marker" not in json.dumps(status)


def test_parse_excludes_thoughts_and_counts_their_cost():
    result = G.parse_response(payload(), "gemini-3.1-pro-preview")
    assert result["observations"] == "A soft repeated pulse."
    assert result["usable"] and result["model_version"] == "test-version"
    assert result["estimated_cost"]["usd"] == pytest.approx(.008)
    assert result["estimated_cost"]["output_including_thinking_tokens"] == 600


@pytest.mark.parametrize("body", [payload("MAX_TOKENS"), payload("SAFETY", ""),
                                  {"promptFeedback": {"blockReason": "SAFETY"}}])
def test_incomplete_response_is_not_a_usable_review(body):
    result = G.parse_response(body, "gemini-3.8-flash")
    assert not result["usable"]
    assert "automatic retry" in result["assessment"]


@pytest.mark.parametrize("body", [[], {"candidates": [[], {}]}, {"candidates": [None]},
                                  {"candidates": [{"content": {"parts": [None]}}]}])
def test_malformed_response_rejected(body):
    with pytest.raises(RuntimeError):
        G.parse_response(body, "gemini-3.8-flash")


def test_cost_uses_total_if_thinking_breakdown_missing():
    usage = payload()["usageMetadata"]
    del usage["thoughtsTokenCount"]
    old = G.estimate_cost("gemini-3.8-flash", usage, datetime.date(2026, 9, 29))
    new = G.estimate_cost("gemini-3.8-flash", usage, datetime.date(2027, 1, 1))
    assert old["output_including_thinking_tokens"] == 600
    assert new["usd"] == 2 * old["usd"]
    assert G.estimate_cost("gemini-3.8-flash", {}) is None


def test_google_transport_is_bounded_and_has_no_metadata_or_url_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-test-credential")
    wav = tmp_path / "private-song-name.wav"
    wav.write_bytes(b"RIFF-test")
    seen = []
    class Opener:
        def open(self, request, timeout):
            body = json.loads(request.data)
            assert "dummy-test-credential" not in request.full_url
            assert request.get_header("X-goog-api-key") == "dummy-test-credential"
            assert request.full_url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
            assert "private-song-name" not in json.dumps(body)
            assert body["generationConfig"]["maxOutputTokens"] == 2048
            assert len(body["contents"][0]["parts"]) == 2
            seen.append(timeout)
            return io.BytesIO(json.dumps(payload()).encode())
    monkeypatch.setattr(G.urllib.request, "build_opener", lambda *a: Opener())
    result = G.generate(tmp_path, wav, "Describe the clip.", "gemini-3.8-flash", 2048, 25)
    assert result["usable"] and seen == [25]
    assert G._NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example") is None


@pytest.mark.parametrize("status", [400, 403, 429, 500])
def test_provider_errors_redact_headers_and_body_without_retry(tmp_path, monkeypatch, status):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-marker")
    wav = tmp_path / "excerpt.wav"
    wav.write_bytes(b"RIFF-test")
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append(1)
            raise urllib.error.HTTPError(request.full_url, status, "secret-marker", {}, io.BytesIO(b"secret-marker"))
    monkeypatch.setattr(G.urllib.request, "build_opener", lambda *a: Opener())
    with pytest.raises(RuntimeError) as error:
        G.generate(tmp_path, wav, "Describe", "gemini-3.8-flash", 2048, 25)
    assert str(status) in str(error.value) and "secret-marker" not in str(error.value)
    assert len(calls) == 1


def test_cloud_silence_skips_upload_without_local_install(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(L, "status", lambda *a: pytest.fail("cloud must not require local installation"))
    monkeypatch.setattr(L, "run_process", lambda *a: pytest.fail("silence must not upload"))
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros(16000), 16000)
    result = L.listen(tmp_path, source, tmp_path / "reviews", seconds=1, backend="gemini")
    assert not result["model_invoked"] and not result["cloud_upload"]


def test_cloud_dispatch_and_evidence_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = []
    def run(command, request, *args):
        assert "test-key" not in json.dumps([command, request])
        seen.append(request)
        return G.parse_response(payload(), request["model"])
    monkeypatch.setattr(L, "run_process", run)
    source = tmp_path / "source.wav"
    sf.write(source, .2*np.sin(np.arange(16000)*.1), 16000)
    result = L.listen(tmp_path, source, tmp_path / "reviews", seconds=1, backend="gemini", model="gemini-3.1-pro-preview")
    assert result["usable"] and result["cloud_upload"] and result["model_invoked"]
    assert seen[0]["max_tokens"] == 2048
    assert result["evidence"]["model_input_subtype"] == "PCM_16"
    assert json.loads(Path(result["report"]).read_text())["estimated_cost"]


def test_pcm_conversion_preserves_source_measurements_and_avoids_clipping(tmp_path):
    source, dest = tmp_path / "source.wav", tmp_path / "excerpt.wav"
    sf.write(source, 1.5*np.sin(np.arange(16000)*.1), 16000, subtype="FLOAT")
    result = L.prepare_excerpt(source, dest, 0, 1, pcm16=True)
    assert result["measured"]["sample_peak_dbfs"] > 3
    assert result["measured"]["clipped_sample_fraction"] > 0
    assert result["model_input_gain"] < 1
    assert np.max(np.abs(sf.read(dest)[0])) < 1


def test_defaults_never_upload_and_invalid_routing_is_rejected():
    assert L.validate_options(0, 15, "Describe", None, 60) == (256, None)
    with pytest.raises(ValueError):
        L.validate_options(0, 15, "Describe", None, 60, model="gemini-3.8-flash")
    with pytest.raises(ValueError):
        L.validate_options(0, 15, "Describe", None, 60, "gemini", "arbitrary-url")
    with pytest.raises(ValueError):
        L.validate_options(0, 15, "Describe", 5000, 60, "gemini")
