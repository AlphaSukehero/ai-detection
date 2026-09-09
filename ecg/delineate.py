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
