"""Clinically interpretable EEG parameters, with the formulas that define them.

Every quantity here is computed from the sampled voltage series. None of it
can be recovered from a rendered picture of a trace without first digitising
that picture back to a signal and establishing a timebase -- the band edges
below are in Hz, and Hz is meaningless without one. eeg/digitize.py enforces
that; this module assumes the signal and sampling rate it is handed are real.

Formulas
--------
Power spectral density, Welch's estimate:

    P(f) = (1 / (K * Fs * U)) * sum_k | FFT( w[n] * x_k[n] ) |^2

for K overlapping segments and window normalisation U = (1/L) sum w[n]^2.

Relative spectral power in a band [f1, f2]:

    RSP_band = integral_{f1}^{f2} P(f) df  /  integral_{f_lo}^{f_hi} P(f) df

evaluated by trapezoidal quadrature over the discrete PSD, with the
denominator taken over the full analysis band (default 0.5-45 Hz). The
reported bands tile that range exactly, so the values sum to 1.

Spectral entropy, Shannon entropy of the normalised power spectrum:

    p_k     = P(f_k) / sum_j P(f_j)           (a probability distribution)
    H_spec  = - sum_k p_k * log2(p_k)
    H_norm  = H_spec / log2(N)                (0 = pure tone, 1 = flat/chaotic)

Spike density, paroxysmal discharges per window:

    density = count(peaks with |x - median| > T * MAD_sigma) / duration_s

where MAD_sigma = 1.4826 * median(|x - median(x)|), a robust standard
deviation that a seizure's own high-amplitude content cannot inflate the way
it inflates an ordinary standard deviation.
"""
import numpy as np
from scipy.signal import find_peaks, welch

# Clinical band edges (Hz). Delta slowing and the theta/alpha ratio are the
# classical markers of cortical slowing in dementia; beta carries drug and
# muscle effects.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    # Not among the four classical clinical bands, but included so the
    # reported distribution covers the whole analysis band and therefore sums
    # to 1. It doubles as an EMG-contamination indicator: scalp muscle
    # activity is broadband and lands here first.
    "gamma": (30.0, 45.0),
}
ANALYSIS_BAND = (0.5, 45.0)
MAD_TO_SIGMA = 1.4826
SPIKE_THRESHOLD_SIGMA = 5.0
SPIKE_MIN_SEPARATION_S = 0.02      # 20 ms; shorter is one discharge, not two


def power_spectrum(x, fs, nperseg=None):
    """Welch PSD restricted to the analysis band. Returns (freqs, power)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros(0), np.zeros(0)
    if nperseg is None:
        nperseg = min(len(x), int(round(fs)))     # ~1 s segments -> ~1 Hz bins
    nperseg = max(8, min(nperseg, len(x)))
    freqs, power = welch(x, fs=fs, nperseg=nperseg)
    keep = (freqs >= ANALYSIS_BAND[0]) & (freqs <= ANALYSIS_BAND[1])
    return freqs[keep], power[keep]


def _band_power(freqs, power, lo, hi):
    """Integrate the PSD over [lo, hi], interpolating at the exact edges.

    Integrating only over the bins strictly inside a band loses the slice
    between the last bin of one band and the first of the next, so the bands
    no longer tile the analysis range and the shares sum to less than 1. The
    interpolated endpoints close those gaps exactly.
    """
    lo = max(lo, freqs[0])
    hi = min(hi, freqs[-1])
    if hi <= lo:
        return 0.0
    inner = freqs[(freqs > lo) & (freqs < hi)]
    grid = np.concatenate(([lo], inner, [hi]))
    return float(np.trapezoid(np.interp(grid, freqs, power), grid))


def relative_spectral_power(x, fs):
    """RSP per band. Returns a dict summing to 1.0, or None if unmeasurable.

    None rather than zeros: a flat or empty segment has no power distribution,
    and reporting 0.25 across four bands would look like a measured finding.
    """
    freqs, power = power_spectrum(x, fs)
    if freqs.size < 2:
        return None
    total = _band_power(freqs, power, *ANALYSIS_BAND)
    if not np.isfinite(total) or total <= 0:
        return None

    return {name: _band_power(freqs, power, lo, hi) / total
            for name, (lo, hi) in BANDS.items()}


def spectral_entropy(x, fs, normalise=True):
    """Shannon entropy of the normalised power spectrum, or None.

    Normalised to [0, 1] by log2(N) so windows of different length stay
    comparable -- without that, entropy rises with resolution alone and a
    longer window looks more chaotic than a shorter one of identical signal.
    """
    freqs, power = power_spectrum(x, fs)
    if freqs.size < 2:
        return None
    total = power.sum()
    if not np.isfinite(total) or total <= 0:
        return None

    p = power / total
    p = p[p > 0]                                   # 0*log0 := 0
    if p.size < 2:
        return 0.0
    h = float(-np.sum(p * np.log2(p)))
    return h / np.log2(p.size) if normalise else h


def spike_metrics(x, fs, threshold_sigma=SPIKE_THRESHOLD_SIGMA):
    """Paroxysmal discharge count, rate and mean amplitude for one window.

    Amplitude is judged against a median-absolute-deviation sigma. An ordinary
    standard deviation is inflated by the very spikes being detected, so a
    dense burst raises its own threshold and hides itself.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    duration = len(x) / fs if fs > 0 else 0.0
    empty = {"count": 0, "rate_per_s": 0.0, "mean_amplitude": 0.0,
             "threshold": 0.0}
    if len(x) < 3 or duration <= 0:
        return empty

    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    sigma = MAD_TO_SIGMA * mad
    if sigma <= 0:
        return empty

    threshold = threshold_sigma * sigma
    distance = max(1, int(round(SPIKE_MIN_SEPARATION_S * fs)))
    peaks, props = find_peaks(np.abs(x - median), height=threshold,
                              distance=distance)
    if peaks.size == 0:
        return {**empty, "threshold": threshold}

    return {
        "count": int(peaks.size),
        "rate_per_s": float(peaks.size / duration),
        "mean_amplitude": float(np.mean(props["peak_heights"])),
        "threshold": threshold,
    }


def anomaly_episodes(flags, times, min_duration_s=0.0):
    """Merge per-window anomaly flags into episodes with onset and duration.

    Windows overlap, so consecutive flagged windows describe one clinical
    event, not several. An episode runs from the start of its first flagged
    window to the end of its last -- the earliest and latest instants the
    evidence actually covers, never a midpoint that would imply precision the
    windowing does not provide.
    """
    flags = list(flags)
    if len(flags) != len(times):
        raise ValueError(
            f"{len(flags)} flags but {len(times)} windows; refusing to pair a "
            "detection with another window's timestamp"
        )

    episodes, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, len(flags) - 1))

    out = []
    for a, b in episodes:
        onset, offset = times[a][0], times[b][1]
        duration = offset - onset
        if duration < min_duration_s:
            continue
        out.append({"onset_s": float(onset), "offset_s": float(offset),
                    "duration_s": float(duration),
                    "n_windows": b - a + 1,
                    "first_window": a, "last_window": b})
    return out


def window_report(x, fs):
    """Every clinical parameter for one window, in one dict."""
    rsp = relative_spectral_power(x, fs)
    return {
        "rsp": rsp,
        "theta_alpha_ratio": (rsp["theta"] / rsp["alpha"]
                              if rsp and rsp["alpha"] > 0 else None),
        "spectral_entropy": spectral_entropy(x, fs),
        "spikes": spike_metrics(x, fs),
    }
