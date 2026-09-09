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


P_SEARCH_EARLY_S = 0.30      # look this far back from QRS onset
P_SEARCH_LATE_S = 0.08       # ...but not closer than this
P_MIN_PROMINENCE = 0.08      # relative to normalised signal


def p_onset(signal, qrs_onset, fs=360.0):
    """Find the P-wave start before a QRS, or None when no P wave is present.

    Atrial fibrillation genuinely has no P wave, so None is a real clinical
    answer here rather than a detection failure.
    """
    sig = _normalise(signal)
    lo = max(0, qrs_onset - int(P_SEARCH_EARLY_S * fs))
    hi = max(lo + 1, qrs_onset - int(P_SEARCH_LATE_S * fs))
    window = sig[lo:hi]
    if len(window) < 3:
        return None
    peaks, props = find_peaks(window, prominence=P_MIN_PROMINENCE)
    if len(peaks) == 0:
        return None
    best = peaks[int(np.argmax(props["prominences"]))]
    # Walk back to where the P wave leaves baseline.
    idx = best
    thresh = 0.3 * window[best]
    while idx > 0 and window[idx] > thresh:
        idx -= 1
    return int(lo + idx)
