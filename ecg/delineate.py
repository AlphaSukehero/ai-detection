"""Locate ECG fiducial points (P, QRS, T boundaries) on a 1-D signal."""
import numpy as np
from scipy.signal import find_peaks


def _normalise(signal):
    sig = np.asarray(signal, dtype=float).flatten()
    sig = sig - np.mean(sig)
    std = float(np.std(sig))
    return sig / std if std > 1e-9 else sig


def detect_r_peaks(signal, fs=360.0):
    """Return sample indices of R-peaks.

    Refractory distance of 250 ms reflects the shortest physiologically
    plausible RR interval; prominence rejects T waves and baseline wander.
    """
    sig = _normalise(signal)
    if np.max(np.abs(sig)) < 1e-9:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(sig, distance=int(0.25 * fs), prominence=0.5)
    return peaks.astype(int)


QRS_SEARCH_S = 0.12          # widest half-window we will walk from R


def qrs_bounds(signal, peak, fs=360.0):
    """Walk outward from an R-peak until the trace returns to baseline.

    The threshold is a fraction of the local peak height, so it adapts to
    beats of differing amplitude instead of using one global cut-off.
    """
    sig = _normalise(signal)
    span = int(QRS_SEARCH_S * fs)
    lo = max(0, peak - span)
    hi = min(len(sig) - 1, peak + span)
    thresh = 0.15 * abs(sig[peak])

    onset = peak
    while onset > lo and abs(sig[onset]) > thresh:
        onset -= 1
    offset = peak
    while offset < hi and abs(sig[offset]) > thresh:
        offset += 1
    return int(onset), int(offset)
