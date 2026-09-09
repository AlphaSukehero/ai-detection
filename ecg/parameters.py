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


from ecg.delineate import t_end

QRS_PLAUSIBLE_S = (0.03, 0.20)
QT_PLAUSIBLE_S = (0.20, 0.65)


def qrs_duration(signal, peaks=None, fs=360.0):
    if peaks is None:
        peaks = detect_r_peaks(signal, fs)
    values = []
    for peak in peaks:
        onset, offset = qrs_bounds(signal, int(peak), fs)
        width = (offset - onset) / fs
        if QRS_PLAUSIBLE_S[0] <= width <= QRS_PLAUSIBLE_S[1]:
            values.append(width)
    if not values:
        return Measurement(None, UNAVAILABLE, "QRS boundaries not resolvable")
    return Measurement(float(np.median(values)), OK)


def qtc_fridericia(qt_s, rr_s):
    """QT / cube-root(RR). Stable at rates where Bazett over-corrects."""
    if rr_s <= 0:
        raise ValueError("rr_s must be positive")
    return qt_s / (rr_s ** (1.0 / 3.0))


def qt_interval(signal, peaks=None, fs=360.0):
    if peaks is None:
        peaks = detect_r_peaks(signal, fs)
    if len(peaks) < 2:
        return Measurement(None, UNAVAILABLE, "Fewer than 2 R-peaks detected")
    rr = _rr_seconds(peaks, fs)
    mean_rr = float(np.mean(rr))
    values = []
    for peak in peaks:
        onset, offset = qrs_bounds(signal, int(peak), fs)
        end = t_end(signal, offset, mean_rr, fs)
        if end is None:
            continue
        qt = (end - onset) / fs
        if QT_PLAUSIBLE_S[0] <= qt <= QT_PLAUSIBLE_S[1]:
            values.append(qt)
    if not values:
        return Measurement(None, UNAVAILABLE, "T wave end not resolvable")
    quality = OK if len(values) >= max(1, len(peaks) // 2) else LOW
    return Measurement(float(np.median(values)), quality)


ST_OFFSET_S = 0.06           # J+60 ms, the conventional measurement point
ST_THRESHOLD_MM = 1.0


def st_deviation(signal, peaks=None, fs=360.0, mm_per_mv=10.0):
    """Deviation at J+60 ms relative to the PR-segment isoelectric baseline."""
    sig = np.asarray(signal, dtype=float).flatten()
    if peaks is None:
        peaks = detect_r_peaks(sig, fs)
    if len(peaks) == 0:
        return Measurement(None, UNAVAILABLE, "No R-peaks detected")

    deviations = []
    for peak in peaks:
        onset, offset = qrs_bounds(sig, int(peak), fs)
        p = p_onset(sig, onset, fs)
        # PR segment is the flat stretch between P end and QRS onset; fall
        # back to just before QRS onset when no P wave is present.
        base_lo = p if p is not None else max(0, onset - int(0.04 * fs))
        baseline = float(np.median(sig[base_lo:onset])) if onset > base_lo else 0.0
        j = offset + int(ST_OFFSET_S * fs)
        if j >= len(sig):
            continue
        # mm = mV x (mm per mV). Standard ECG gain is 10 mm/mV.
        deviations.append((sig[j] - baseline) * mm_per_mv)

    if not deviations:
        return Measurement(None, UNAVAILABLE, "ST point beyond signal end")

    dev_mm = float(np.median(deviations))
    if dev_mm > ST_THRESHOLD_MM:
        label = "Elevated"
    elif dev_mm < -ST_THRESHOLD_MM:
        label = "Depressed"
    else:
        label = "Normal"
    return Measurement(dev_mm, OK, label)
