"""Turn fiducial points into the parameters shown on the ECG page."""
import numpy as np

from ecg.quality import Measurement, OK, LOW, UNAVAILABLE

MIN_PEAKS_FOR_RATE = 3
MIN_PEAKS_FOR_RHYTHM = 5
REGULAR_CV_THRESHOLD = 0.10


def _rr_seconds(peaks, fs):
    return np.diff(np.asarray(peaks, dtype=float)) / fs


def heart_rate(peaks, fs=360.0):
    if len(peaks) < MIN_PEAKS_FOR_RATE:
        return Measurement(None, UNAVAILABLE, "Fewer than 3 R-peaks detected")
    rr = _rr_seconds(peaks, fs)
    mean_rr = float(np.mean(rr))
    if mean_rr <= 0:
        return Measurement(None, UNAVAILABLE, "Invalid RR interval")
    return Measurement(60.0 / mean_rr, OK)


def rhythm(peaks, fs=360.0):
    if len(peaks) < MIN_PEAKS_FOR_RHYTHM:
        return Measurement(None, UNAVAILABLE, "Fewer than 5 beats")
    rr = _rr_seconds(peaks, fs)
    mean_rr = float(np.mean(rr))
    if mean_rr <= 0:
        return Measurement(None, UNAVAILABLE, "Invalid RR interval")
    cv = float(np.std(rr) / mean_rr)
    label = "Regular" if cv < REGULAR_CV_THRESHOLD else "Irregular"
    return Measurement(cv, OK, label)


from ecg.delineate import detect_r_peaks, qrs_bounds, p_onset

PR_PLAUSIBLE_S = (0.08, 0.30)


def pr_interval(signal, peaks=None, fs=360.0):
    """Median P-onset to QRS-onset across beats."""
    if peaks is None:
        peaks = detect_r_peaks(signal, fs)
    if len(peaks) == 0:
        return Measurement(None, UNAVAILABLE, "No R-peaks detected")

    values = []
    for peak in peaks:
        onset, _ = qrs_bounds(signal, int(peak), fs)
        p = p_onset(signal, onset, fs)
        if p is None:
            continue
        pr = (onset - p) / fs
        if PR_PLAUSIBLE_S[0] <= pr <= PR_PLAUSIBLE_S[1]:
            values.append(pr)

    if not values:
        return Measurement(None, UNAVAILABLE, "P wave not detectable")
    quality = OK if len(values) >= max(1, len(peaks) // 2) else LOW
    return Measurement(float(np.median(values)), quality)
