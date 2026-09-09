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


def _net_qrs_area(signal, fs):
    """Signed area under the QRS complexes: the lead's net deflection."""
    sig = np.asarray(signal, dtype=float).flatten()
    peaks = detect_r_peaks(sig, fs)
    if len(peaks) == 0:
        return 0.0
    total = 0.0
    for peak in peaks:
        onset, offset = qrs_bounds(sig, int(peak), fs)
        total += float(np.sum(sig[onset:offset + 1]))
    return total / len(peaks)


def qrs_axis(leads, fs=360.0):
    """Frontal-plane QRS axis from the net deflections of leads I and aVF.

    One lead is a single projection of the electrical vector, so a 2-D angle
    cannot be recovered from it. Single-lead input therefore reports
    "Requires 12-lead" rather than a guess.
    """
    if not isinstance(leads, dict) or "I" not in leads or "aVF" not in leads:
        return Measurement(None, UNAVAILABLE, "Requires 12-lead")
    # A blank panel digitizes to None (see ecg.digitize._trace_row_band), so a
    # sheet can carry the lead names without carrying usable traces.
    if leads["I"] is None or leads["aVF"] is None:
        return Measurement(None, UNAVAILABLE, "Lead I or aVF has no trace")

    net_i = _net_qrs_area(leads["I"], fs)
    net_avf = _net_qrs_area(leads["aVF"], fs)
    if abs(net_i) < 1e-9 and abs(net_avf) < 1e-9:
        return Measurement(None, UNAVAILABLE, "No measurable QRS deflection")

    degrees = float(np.degrees(np.arctan2(net_avf, net_i)))
    if -30.0 <= degrees <= 90.0:
        label = "Normal axis"
    elif -90.0 <= degrees < -30.0:
        label = "Left axis deviation"
    elif 90.0 < degrees <= 180.0:
        label = "Right axis deviation"
    else:
        label = "Extreme axis"
    return Measurement(degrees, OK, label)


def _all_unavailable(reason):
    """A complete report in which nothing could be measured.

    Returned rather than raising, so a blank or unreadable image degrades to
    honest dashes instead of a 500.
    """
    keys = ["heart_rate", "rhythm", "pr_interval", "qrs_duration",
            "qt_interval", "qtc", "st_segment", "axis", "rr_interval",
            "sdnn", "rmssd"]
    report = {k: Measurement(None, UNAVAILABLE, reason) for k in keys}
    report["display"] = {k: "—" for k in keys}
    return report


def analyse(signal_or_leads, fs=360.0, px_per_mm=None, from_image=False):
    """Measure every displayed parameter, flagging what could not be measured."""
    if isinstance(signal_or_leads, dict):
        leads = signal_or_leads
        # A blank panel digitizes to None, and dict.get() only substitutes its
        # default when the KEY is missing - not when the value is None. Pick the
        # first lead that actually carries a trace, preferring II.
        primary = leads.get("II")
        if primary is None:
            primary = next((v for v in leads.values() if v is not None), None)
        if primary is None:
            return _all_unavailable("No lead carried a usable trace")
    else:
        leads = {"II": np.asarray(signal_or_leads, dtype=float).flatten()}
        primary = leads["II"]

    peaks = detect_r_peaks(primary, fs)
    hr = heart_rate(peaks, fs)
    rhy = rhythm(peaks, fs)
    pr = pr_interval(primary, peaks, fs)
    qrs = qrs_duration(primary, peaks, fs)
    qt = qt_interval(primary, peaks, fs)
    axis = qrs_axis(leads, fs)

    # ST is the one parameter expressed in millimetres, so it needs the paper
    # scale. from_image=True with no detected grid means we cannot honestly
    # convert amplitude to mm.
    if from_image and px_per_mm is None:
        st = Measurement(None, UNAVAILABLE, "ECG grid not detected")
    else:
        st = st_deviation(primary, peaks, fs)

    if qt.value is not None and len(peaks) >= 2:
        rr_s = _rr_seconds(peaks, fs)
        mean_rr = float(np.mean(rr_s))
        qtc = Measurement(qtc_fridericia(qt.value, mean_rr), qt.quality)
        rr = Measurement(mean_rr, OK)
        # Stored in seconds like every other duration here; the display layer
        # is the only place that converts to milliseconds.
        sdnn = Measurement(float(np.std(rr_s, ddof=1)), OK) \
            if len(rr_s) >= 2 else Measurement(None, UNAVAILABLE, "Too few beats")
        rmssd = Measurement(float(np.sqrt(np.mean(np.diff(rr_s) ** 2))), OK) \
            if len(rr_s) >= 3 else Measurement(None, UNAVAILABLE, "Too few beats")
    else:
        qtc = Measurement(None, UNAVAILABLE, "QT not measurable")
        rr = Measurement(None, UNAVAILABLE, "Fewer than 2 R-peaks detected")
        sdnn = Measurement(None, UNAVAILABLE, "Too few beats")
        rmssd = Measurement(None, UNAVAILABLE, "Too few beats")

    report = {
        "heart_rate": hr, "rhythm": rhy, "pr_interval": pr,
        "qrs_duration": qrs, "qt_interval": qt, "qtc": qtc,
        "st_segment": st, "axis": axis, "rr_interval": rr,
        "sdnn": sdnn, "rmssd": rmssd,
    }
    report["display"] = {
        "heart_rate": hr.format("bpm"),
        "rhythm": rhy.reason if rhy.value is not None else "—",
        "pr_interval": Measurement(
            pr.value * 1000 if pr.value is not None else None, pr.quality).format("ms", 0),
        "qrs_duration": Measurement(
            qrs.value * 1000 if qrs.value is not None else None, qrs.quality).format("ms", 0),
        "qt_interval": Measurement(
            qt.value * 1000 if qt.value is not None else None, qt.quality).format("ms", 0),
        "qtc": Measurement(
            qtc.value * 1000 if qtc.value is not None else None, qtc.quality).format("ms", 0)
            + (" (Fridericia)" if qtc.value is not None else ""),
        "st_segment": (f"{st.reason} ({st.value:+.1f} mm)"
                       if st.value is not None else "—"),
        "axis": (f"{axis.value:+.0f}° ({axis.reason})"
                 if axis.value is not None else "—"),
        "rr_interval": rr.format("s", 3),
        "sdnn": Measurement(
            sdnn.value * 1000 if sdnn.value is not None else None,
            sdnn.quality).format("ms"),
        "rmssd": Measurement(
            rmssd.value * 1000 if rmssd.value is not None else None,
            rmssd.quality).format("ms"),
    }
    return report
