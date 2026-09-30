"""Audio analysis with numpy and scipy only. These are the AI's ears."""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import stft

EPS = 1e-12


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), EPS))


def rms_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    return db(math.sqrt(float(np.mean(np.square(x, dtype=np.float64)))))


def peak_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    return db(float(np.max(np.abs(x))))


def channel_levels(audio: np.ndarray) -> list[float]:
    """RMS in dBFS per channel of a frames x channels array."""
    return [rms_db(audio[:, c]) for c in range(audio.shape[1])]


def envelope_db(x: np.ndarray, sr: int, win_ms: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Short-window RMS envelope. Returns (times in seconds, level in dBFS)."""
    win = max(1, int(sr * win_ms / 1000.0))
    n = len(x) // win
    if n == 0:
        return np.zeros(0), np.zeros(0)
    frames = x[: n * win].reshape(n, win).astype(np.float64)
    lev = 20.0 * np.log10(np.sqrt(np.mean(frames * frames, axis=1)) + EPS)
    t = (np.arange(n) * win + win / 2) / sr
    return t, lev


def onset_sample(x: np.ndarray, sr: int, start: int = 0, rise_db: float = 12.0, floor_window_ms: float = 60.0, win_ms: float = 2.0) -> int | None:
    """First sample at or after `start` where the level rises `rise_db` above the floor measured just before `start`."""
    win = max(1, int(sr * win_ms / 1000.0))
    fw = int(sr * floor_window_ms / 1000.0)
    pre = x[max(0, start - fw): start]
    floor = rms_db(pre) if len(pre) > 0 else -90.0
    floor = max(floor, -90.0)
    thresh = floor + rise_db
    seg = x[start:]
    n = len(seg) // win
    if n == 0:
        return None
    frames = seg[: n * win].reshape(n, win).astype(np.float64)
    lev = 20.0 * np.log10(np.sqrt(np.mean(frames * frames, axis=1)) + EPS)
    hits = np.nonzero(lev > thresh)[0]
    if len(hits) == 0:
        return None
    return start + int(hits[0]) * win


def spectral_centroid_series(x: np.ndarray, sr: int, nperseg: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    f, t, Z = stft(x.astype(np.float64), fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    mag = np.abs(Z)
    denom = mag.sum(axis=0) + EPS
    cent = (f[:, None] * mag).sum(axis=0) / denom
    return t, cent


def note_hz(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def note_prominence(x: np.ndarray, sr: int, notes: list[int], harmonics: int = 3) -> dict[int, float]:
    """How strongly each note's fundamental (or its harmonics) stands out, in dB above the
    surrounding spectrum. Values above ~10 dB mean the note is sounding."""
    n = len(x)
    if n < 2048:
        return {m: -60.0 for m in notes}
    nfft = 1
    while nfft < n:
        nfft *= 2
    nfft = max(nfft, 1 << 15)
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(x.astype(np.float64) * w, n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    spec_db = 20.0 * np.log10(spec + EPS)
    out: dict[int, float] = {}
    for m in notes:
        best = -60.0
        for h in range(1, harmonics + 1):
            f0 = note_hz(m) * h
            if f0 >= sr / 2:
                break
            lo, hi = f0 * (1 - 0.012), f0 * (1 + 0.012)  # about +/- 20 cents
            band = (freqs >= lo) & (freqs <= hi)
            if not band.any():
                continue
            peak = spec_db[band].max()
            # local floor: one third-octave around, excluding the note band
            wlo, whi = f0 / 1.26, f0 * 1.26
            around = (freqs >= wlo) & (freqs <= whi) & ~band
            if around.sum() < 8:
                continue
            floor = np.median(spec_db[around])
            best = max(best, peak - floor)
        out[m] = float(best)
    return out


def spectrogram_png(x: np.ndarray, sr: int, path: str, max_hz: float = 8000.0, width: int = 1200, height: int = 400, marks: list[tuple[str, int]] | None = None,
                    log_frequency: bool = False, min_hz: float = 30.0) -> str:
    """Write a log-magnitude spectrogram as a PNG (with Pillow) and return the path. With log_frequency the
    vertical axis is logarithmic between min_hz and max_hz (a finer 4096-point window shows the bass) and
    frequency gridlines are drawn; otherwise the axis is linear from 0 to max_hz."""
    from PIL import Image, ImageDraw

    nperseg = 4096 if log_frequency else 1024
    f, t, Z = stft(x.astype(np.float64), fs=sr, nperseg=nperseg, noverlap=nperseg - (512 if log_frequency else 256))
    mag = 20.0 * np.log10(np.abs(Z) + EPS)
    if log_frequency:
        grid = np.geomspace(max(min_hz, f[1]), min(max_hz, f[-1]), height)
        idx = np.clip(np.searchsorted(f, grid), 0, len(f) - 1)
        mag = mag[idx][::-1]
    else:
        keep = f <= max_hz
        mag = mag[keep][::-1]  # low freqs at the bottom
    top = np.percentile(mag, 99.5)
    img = np.clip((mag - (top - 70.0)) / 70.0, 0.0, 1.0)
    img8 = (img * 255).astype(np.uint8)
    pil = Image.fromarray(img8, mode="L").resize((width, height), Image.BILINEAR)
    pil = pil.convert("RGB")
    d = ImageDraw.Draw(pil)
    total = len(x) / sr
    if total > 0:
        for s in range(int(total) + 1):
            xx = int(s / total * width)
            d.line([(xx, 0), (xx, 8)], fill=(255, 255, 255))
            d.text((xx + 2, 8), f"{s}s", fill=(255, 255, 255))
        for label, idx in marks or []:
            xx = int(idx / len(x) * width)
            d.line([(xx, 0), (xx, height)], fill=(255, 80, 80))
            d.text((xx + 2, height - 12), label, fill=(255, 80, 80))
    if log_frequency:
        lo, hi = max(min_hz, f[1]), min(max_hz, f[-1])
        for hz in (50, 100, 200, 500, 1000, 2000, 5000):
            if lo < hz < hi:
                yy = int(height - (np.log(hz) - np.log(lo)) / (np.log(hi) - np.log(lo)) * height)
                d.line([(0, yy), (width, yy)], fill=(90, 90, 160))
                d.text((width - 44, yy - 11), f"{hz if hz < 1000 else str(hz // 1000) + 'k'} Hz", fill=(150, 150, 255))
    pil.save(path)
    return path


BANDS: list[tuple[str, float, float]] = [("sub", 20, 60), ("bass", 60, 250), ("low_mid", 250, 500), ("mid", 500, 2000), ("presence", 2000, 5000), ("air", 5000, 20000)]


def band_levels(x: np.ndarray, sr: int) -> dict[str, float]:
    """Mean level in dBFS of six bands (sub 20-60, bass 60-250, low_mid 250-500, mid 500-2k, presence 2-5k, air 5k+)."""
    if len(x) < 4096:
        return {name: -120.0 for name, _, _ in BANDS}
    f, _, Z = stft(x.astype(np.float64), fs=sr, nperseg=4096, noverlap=4096 - 1024)
    P = (np.abs(Z) ** 2).mean(axis=1)
    P = P * (float(np.mean(x.astype(np.float64) ** 2)) / (float(P.sum()) + 1e-20))   # so the bands sum to the signal's mean-square power
    out = {}
    for name, lo, hi in BANDS:
        sel = (f >= lo) & (f < hi)
        out[name] = round(float(10 * np.log10(P[sel].sum() + 1e-12)), 1) if sel.any() else -120.0
    return out


def pitch_track(x: np.ndarray, sr: int, hop_s: float = 0.02, win_s: float = 0.05, fmin: float = 40.0, fmax: float = 2000.0, min_conf: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fundamental frequency per frame by autocorrelation. Returns (times, f0 in Hz with 0 where unvoiced, confidence 0-1)."""
    n = int(win_s * sr); h = max(1, int(hop_s * sr))
    lo = max(2, int(sr / fmax)); hi = min(n - 2, int(sr / fmin))
    ts, f0s, cs = [], [], []
    xx = x.astype(np.float64)
    for start in range(0, max(0, len(xx) - n), h):
        fr = xx[start: start + n]
        fr = fr - fr.mean()
        e = float(np.sqrt((fr ** 2).mean()))
        t = (start + n / 2) / sr
        if e < 1e-4:
            ts.append(t); f0s.append(0.0); cs.append(0.0); continue
        ac = np.correlate(fr, fr, "full")[n - 1:]
        ac = ac / (ac[0] + 1e-12)
        i = lo + int(np.argmax(ac[lo:hi]))
        conf = float(ac[i])
        if 0 < i < len(ac) - 1:   # parabolic refinement
            a, b, c = ac[i - 1], ac[i], ac[i + 1]
            den = a - 2 * b + c
            i = i + (0.5 * (a - c) / den if den != 0 else 0.0)
        f0 = sr / i if conf >= min_conf else 0.0
        ts.append(t); f0s.append(float(f0)); cs.append(conf)
    return np.array(ts), np.array(f0s), np.array(cs)


def pitch_summary(f0: np.ndarray, conf: np.ndarray) -> dict[str, Any]:
    """Median pitch of the voiced frames as Hz, note name and cents, with how steady it is."""
    voiced = f0[(f0 > 0) & (conf > 0)]
    if len(voiced) == 0:
        return {"voiced_fraction": 0.0, "f0_hz": None, "note": None, "cents_off": None, "stability_cents": None}
    med = float(np.median(voiced))
    midi = 69 + 12 * np.log2(med / 440.0)
    m = int(round(midi)); cents = round(float((midi - m) * 100), 0)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    name = f"{names[m % 12]}{m // 12 - 1}"
    spread = float(np.percentile(1200 * np.log2(voiced / med), 90) - np.percentile(1200 * np.log2(voiced / med), 10))
    return {"voiced_fraction": round(float(len(voiced) / max(1, len(f0))), 2), "f0_hz": round(med, 1), "note": name, "cents_off": cents,
            "stability_cents": round(spread, 0), "min_hz": round(float(voiced.min()), 1), "max_hz": round(float(voiced.max()), 1)}


def pitch_modulation(times: np.ndarray, f0: np.ndarray, lo_hz: float = 0.5, hi_hz: float = 15.0) -> dict[str, Any]:
    """Vibrato or pitch wobble: the strongest periodic movement of the voiced pitch track, as rate in Hz and
    peak-to-peak depth in cents (None when the track is too short or has no clear period)."""
    v = f0 > 0
    if v.sum() < 20 or len(times) < 2:
        return {"rate_hz": None, "depth_cents": None}
    hop = float(np.median(np.diff(times)))
    cents = 1200 * np.log2(f0[v] / np.median(f0[v]))
    cents = cents - np.polyval(np.polyfit(np.arange(len(cents)), cents, 2), np.arange(len(cents)))
    n = 1 << ((len(cents) - 1).bit_length() + 2)
    w = np.hanning(len(cents))
    spec = np.abs(np.fft.rfft(cents * w, n)) * 2.0 / w.sum()      # amplitude of each sinusoidal component, in cents
    fr = np.fft.rfftfreq(n, hop)
    band = (fr >= lo_hz) & (fr <= hi_hz)
    if not band.any():
        return {"rate_hz": None, "depth_cents": None}
    i = int(np.argmax(spec[band]))
    prom = float(spec[band][i] / (np.median(spec[band]) + 1e-9))
    depth = float(2 * spec[band][i])
    if prom < 6 or depth < 5:
        return {"rate_hz": None, "depth_cents": round(depth, 0), "note": "no clear periodic pitch movement"}
    return {"rate_hz": round(float(fr[band][i]), 2), "depth_cents": round(depth, 0), "prominence": round(prom, 1)}


def speech_intelligibility(reference: np.ndarray, ref_sr: int, recorded: np.ndarray, rec_sr: int, max_lag_s: float = 1.5) -> dict[str, Any]:
    """STOI (0-1) of a recording against the speech that was sent, after aligning the two by their level
    envelopes. Compares the overlap; repeatable to about 0.005 on the Field. STOI ignores content above
    about 5 kHz, so it undervalues consonants; it ranks vocoder settings well, Whisper does not."""
    from scipy.signal import resample_poly
    from pystoi import stoi

    def mono16(x, sr):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if sr != 16000:
            g = np.gcd(int(sr), 16000)
            x = resample_poly(x, 16000 // g, int(sr) // g)
        return x
    ref = mono16(reference, ref_sr); rec = mono16(recorded, rec_sr)
    def env(x):
        n = 160
        m = len(x) // n
        return np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(axis=1))
    er, ec = env(ref), env(rec)
    er = er - er.mean(); ec = ec - ec.mean()
    max_lag = int(max_lag_s * 100)
    best, best_lag = -1e9, 0
    for lag in range(-max_lag, max_lag + 1):   # lag > 0: the recording is late
        if lag >= 0:
            a, b = er[: len(er) - lag], ec[lag: lag + len(er)]
        else:
            a, b = er[-lag:], ec[: len(er) + lag]
        n = min(len(a), len(b))
        if n < 20:
            continue
        v = float((a[:n] * b[:n]).sum() / (np.sqrt((a[:n] ** 2).sum() * (b[:n] ** 2).sum()) + 1e-12))
        if v > best:
            best, best_lag = v, lag
    shift = best_lag * 160
    if shift >= 0:
        rec_al = rec[shift:]
    else:
        rec_al = np.concatenate([np.zeros(-shift), rec])
    n = min(len(ref), len(rec_al))
    if n < 16000:
        return {"stoi": None, "lag_ms": None, "reason": "less than a second of overlap"}
    value = float(stoi(ref[:n], rec_al[:n], 16000, extended=False))
    return {"stoi": round(value, 3), "lag_ms": round(best_lag * 10.0, 0), "envelope_match": round(best, 2), "seconds_compared": round(n / 16000.0, 1)}


def onset_strength(x: np.ndarray, sr: int, hop: int = 256, nfft: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Spectral flux: how much new energy appears from one frame to the next. Peaks mark note starts,
    even inside another note's tail. Returns (times, strength)."""
    from scipy.signal import stft as _stft
    f, t, Z = _stft(x.astype(np.float64), fs=sr, nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False)
    mag = 20.0 * np.log10(np.abs(Z) + 1e-6)
    diff = np.diff(mag, axis=1)
    flux = np.sum(np.maximum(diff, 0.0), axis=0)
    return t[1:], flux


def onset_offset_ms(x: np.ndarray, sr: int, expected: int, window_ms: float = 80.0) -> float | None:
    """Offset in ms between the strongest onset near `expected` (a sample index) and `expected` itself."""
    w = int(window_ms / 1000.0 * sr)
    a, b = max(0, expected - w - 2048), min(len(x), expected + w + 2048)
    seg = x[a:b]
    if len(seg) < 4096:
        return None
    t, flux = onset_strength(seg, sr)
    if len(flux) == 0:
        return None
    centre = (expected - a) / sr
    sel = np.abs(t - centre) <= window_ms / 1000.0
    if not sel.any():
        return None
    idx = np.nonzero(sel)[0]
    peak = idx[int(np.argmax(flux[idx]))]
    # require the peak to stand out from the median flux in the window
    if flux[peak] < 3.0 * (np.median(flux[idx]) + 1e-9):
        return None
    return round(float((t[peak] - centre) * 1000.0), 1)


def harmonic_levels(x: np.ndarray, sr: int, f0: float, n_harm: int = 24, nperseg: int = 4096, hop: int = 512) -> tuple[np.ndarray, float]:
    """Level in dB of the first n_harm harmonics of f0 over time (rows = harmonics), and the hop in seconds."""
    from scipy.signal import stft

    f, _, Z = stft(x, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)
    S = 20 * np.log10(np.abs(Z) + 1e-9)
    rows = []
    for k in range(1, n_harm + 1):
        i = int(np.argmin(np.abs(f - f0 * k)))
        rows.append(S[max(0, i - 2): i + 3].max(axis=0))
    return np.array(rows), hop / sr


def modulation_rate(x: np.ndarray, sr: int, f0: float, lo_hz: float = 0.4, hi_hz: float = 40.0, n_harm: int = 24) -> dict[str, float | None]:
    """Rate of periodic spectral movement on a held note of pitch f0, from the harmonics' level series
    (a phaser, a filter LFO or a tremolo all show as periodic dips walking across harmonics).
    Returns the rate in Hz with its prominence over the modulation spectrum's median (None when nothing
    periodic stands out), and the median peak-to-peak swing of the harmonics in dB."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    H, hop = harmonic_levels(x.astype(np.float64), sr, f0, n_harm=n_harm)
    if H.shape[1] < 16:
        return {"rate_hz": None, "prominence": 0.0, "swing_db": 0.0}
    H = H[:, 2:-2]
    n_ma = max(3, int(1.0 / hop) | 1)
    combined = None
    fr = None
    for k in range(H.shape[0]):
        s = H[k] - np.convolve(H[k], np.ones(n_ma) / n_ma, mode="same")
        s = s - s.mean()
        n = 1 << ((len(s) - 1).bit_length() + 2)
        spec = np.abs(np.fft.rfft(s * np.hanning(len(s)), n))
        fr = np.fft.rfftfreq(n, hop)
        band = (fr >= lo_hz) & (fr <= hi_hz)
        spec = spec / (np.median(spec[band]) + 1e-12)
        combined = spec if combined is None else combined + spec
    band = (fr >= lo_hz) & (fr <= hi_hz)
    i = int(np.argmax(combined[band]))
    prom = float(combined[band][i] / (np.median(combined[band]) + 1e-12))
    swing = float(np.median(np.percentile(H, 95, axis=1) - np.percentile(H, 5, axis=1)))
    return {"rate_hz": round(float(fr[band][i]), 2) if prom > 8 else None, "prominence": round(prom, 1), "swing_db": round(swing, 1)}
