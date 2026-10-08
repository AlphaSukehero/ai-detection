"""Locate ECG fiducial points (P, QRS, T boundaries) on a 1-D signal.

R-peak detection is measured against MIT-BIH reference annotations (150 ms
tolerance): sensitivity 0.996 and precision 0.992 over the first five
minutes of all 44 DS1/DS2 records. See tests/test_ecg_delineate_mitbih.py
for the gates on both.
"""
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


def _normalise(signal):
    sig = np.asarray(signal, dtype=float).flatten()
    sig = sig - np.mean(sig)
    std = float(np.std(sig))
    return sig / std if std > 1e-9 else sig


QRS_BAND_HZ = (5.0, 18.0)     # where QRS energy sits; P and T lie below it
INTEGRATION_S = 0.12          # roughly one QRS width
REFRACTORY_S = 0.25           # shortest physiologically plausible RR
LOCAL_WINDOW_S = 4.0          # the threshold follows the beats either side
ENERGY_FRACTION = 0.15        # of the local 90th-percentile beat energy
REFINE_S = 0.08               # the R peak lies this close to the energy peak


def _prominence_peaks(signal, fs):
    sig = _normalise(signal)
    peaks, _ = find_peaks(sig, distance=max(1, int(REFRACTORY_S * fs)),
                          prominence=0.5)
    return peaks.astype(int)


def detect_r_peaks(signal, fs=360.0):
    """Return sample indices of R-peaks.

    QRS complexes are found by slope energy in the 5-18 Hz band (the
    Pan-Tompkins idea), not by height. Thresholding the raw trace on
    prominence could not tell a T wave from an R wave: over the first five
    minutes of all 44 MIT-BIH records it scored sensitivity 0.976 but
    precision 0.647, so a third of its "beats" were not beats and every
    heart rate built on them read high. This detector scores 0.996 / 0.992
    on the same data. The threshold is local, so a run of small beats is
    judged against its neighbours rather than against the tallest beat in
    the recording.
    """
    sig = np.asarray(signal, dtype=float).flatten()
    n = len(sig)
    if n == 0 or float(np.ptp(sig)) < 1e-9:
        return np.array([], dtype=int)
    high = min(QRS_BAND_HZ[1], 0.45 * fs)
    if n < int(0.6 * fs) or high <= QRS_BAND_HZ[0]:
        # Too short or too coarsely sampled to band-pass.
        return _prominence_peaks(sig, fs)
    try:
        sos = butter(2, [QRS_BAND_HZ[0], high], btype="band", fs=fs, output="sos")
        slope = np.gradient(sosfiltfilt(sos, sig)) ** 2
    except ValueError:
        return _prominence_peaks(sig, fs)
    width = max(1, int(round(INTEGRATION_S * fs)))
    energy = np.convolve(slope, np.ones(width) / width, mode="same")
    candidates, _ = find_peaks(energy, distance=max(1, int(REFRACTORY_S * fs)))
    if len(candidates) == 0:
        return np.array([], dtype=int)

    heights = energy[candidates]
    half = LOCAL_WINDOW_S * fs
    reach = max(1, int(round(REFINE_S * fs)))
    base = int(0.3 * fs)
    peaks = []
    for c, height in zip(candidates, heights, strict=True):
        local = heights[(candidates >= c - half) & (candidates <= c + half)]
        if height < ENERGY_FRACTION * np.percentile(local, 90):
            continue
        lo, hi = max(0, c - reach), min(n, c + reach + 1)
        baseline = np.median(sig[max(0, c - base):min(n, c + base)])
        peaks.append(lo + int(np.argmax(sig[lo:hi] - baseline)))

    # Two energy peaks can refine onto the same complex; keep the taller.
    kept = []
    for p in sorted(set(peaks)):
        if kept and p - kept[-1] < 0.2 * fs:
            if sig[p] > sig[kept[-1]]:
                kept[-1] = p
        else:
            kept.append(p)
    return np.array(kept, dtype=int)


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
    # Absolute value: an inverted P wave has a negative apex, and a negative
    # threshold would never stop the backward walk before the window start.
    thresh = 0.3 * abs(window[best])
    while idx > 0 and abs(window[idx]) > thresh:
        idx -= 1
    return int(lo + idx)


def t_end(signal, qrs_offset, rr_s, fs=360.0):
    """End of the T wave by the tangent method.

    The search window scales with RR because the T wave moves closer to the
    QRS as heart rate rises.
    """
    sig = _normalise(signal)
    start = qrs_offset + int(0.04 * fs)
    stop = min(len(sig) - 1, qrs_offset + int(min(0.60, 0.6 * rr_s) * fs))
    if stop - start < 5:
        return None
    window = sig[start:stop]
    apex = int(np.argmax(np.abs(window)))
    if apex >= len(window) - 2:
        return None
    # Steepest descent after the apex, extrapolated to baseline.
    slopes = np.diff(window[apex:])
    if len(slopes) == 0:
        return None
    steep = int(np.argmin(slopes))
    slope = slopes[steep]
    if slope >= -1e-6:
        return None
    idx = apex + steep
    offset = int(window[idx] / (-slope))
    end = idx + max(0, offset)
    if end > len(window) - 1:
        # A near-flat T wave makes the tangent extrapolation run past the
        # search window. Clamping it to the window edge would return a
        # fabricated T-end that often lands inside the QT plausibility band
        # and is then reported as a measured, OK-quality QT. Reject instead.
        return None
    return int(start + end)
