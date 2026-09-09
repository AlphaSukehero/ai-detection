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
