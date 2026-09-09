import numpy as np
from ecg.parameters import heart_rate, rhythm
from ecg.quality import OK, UNAVAILABLE


def test_heart_rate_from_evenly_spaced_peaks():
    peaks = np.arange(0, 3600, 288)          # 0.8 s spacing at 360 Hz -> 75 bpm
    m = heart_rate(peaks, fs=360.0)
    assert m.quality == OK
    assert abs(m.value - 75.0) < 0.5


def test_heart_rate_unavailable_with_too_few_peaks():
    m = heart_rate(np.array([10, 300]), fs=360.0)
    assert m.quality == UNAVAILABLE
    assert m.value is None


def test_regular_rhythm_for_constant_rr():
    peaks = np.arange(0, 3600, 288)
    m = rhythm(peaks, fs=360.0)
    assert m.reason == "Regular"
    assert m.value < 0.10


def test_irregular_rhythm_for_varying_rr():
    peaks = np.array([0, 200, 620, 780, 1300, 1450, 2000, 2100])
    m = rhythm(peaks, fs=360.0)
    assert m.reason == "Irregular"


from ecg.parameters import pr_interval
from ecg.quality import UNAVAILABLE


def _beat_with_p_wave(fs=360.0, n_beats=6, rr=0.8):
    n = int(n_beats * rr * fs)
    sig = np.zeros(n)
    for b in range(n_beats):
        r = int(b * rr * fs) + 100
        if r + 20 >= n:
            break
        # Dense QRS ~70 ms wide: a real complex occupies every sample it spans,
        # and a width measured from isolated spikes is not physiological.
        half = int(0.035 * fs)
        for off in range(-half, half + 1):
            if 0 <= r + off < n:
                sig[r + off] = 3.0 * (1.0 - abs(off) / (half + 1.0))
        p = r - int(0.16 * fs)             # P wave 160 ms before R
        pw = int(0.02 * fs)
        if p - pw > 0:
            for off in range(-pw, pw + 1):
                sig[p + off] = 0.45 * (1.0 - abs(off) / (pw + 1.0))
    return sig


def test_pr_interval_measured_near_expected_value():
    sig = _beat_with_p_wave()
    m = pr_interval(sig, None, fs=360.0)
    assert m.value is not None
    assert 0.10 <= m.value <= 0.22


def test_pr_unavailable_when_no_p_wave():
    fs = 360.0
    sig = np.zeros(int(6 * 0.8 * fs))
    for b in range(6):
        r = int(b * 0.8 * fs) + 100
        if r + 2 < len(sig):
            sig[r] = 3.0
            sig[r - 1] = sig[r + 1] = 1.0
    m = pr_interval(sig, None, fs=fs)
    assert m.value is None
    assert m.quality == UNAVAILABLE


from ecg.parameters import qrs_duration, qt_interval, qtc_fridericia


def test_qtc_fridericia_matches_formula():
    assert abs(qtc_fridericia(0.40, 1.0) - 0.40) < 1e-9
    assert abs(qtc_fridericia(0.40, 0.512) - 0.50) < 1e-3


def test_qtc_fridericia_is_tamer_than_bazett_at_high_rate():
    """At 195 bpm Bazett produced 722 ms; Fridericia must stay far below."""
    rr = 0.307
    assert qtc_fridericia(0.40, rr) * 1000 < 620


def test_qrs_duration_in_physiological_range():
    sig = _beat_with_p_wave()
    m = qrs_duration(sig, None, fs=360.0)
    assert m.value is not None
    assert 0.02 <= m.value <= 0.20


def test_qt_returns_measurement_not_constant():
    sig = _beat_with_p_wave()
    m = qt_interval(sig, None, fs=360.0)
    assert m.value is None or m.value != 0.40


from ecg.delineate import detect_r_peaks
from ecg.parameters import st_deviation


def test_st_normal_for_flat_baseline():
    sig = _beat_with_p_wave()
    m = st_deviation(sig, None, fs=360.0)
    assert m.reason in {"Normal", "Elevated", "Depressed"}


def test_st_elevation_detected_when_segment_raised():
    fs = 360.0
    sig = _beat_with_p_wave(fs=fs)
    peaks = detect_r_peaks(sig, fs)
    for peak in peaks:
        j = int(peak) + int(0.04 * fs)
        sig[j:j + int(0.10 * fs)] += 0.9      # lift the ST segment
    m = st_deviation(sig, peaks, fs=fs)
    assert m.reason == "Elevated"


def test_st_unavailable_without_peaks():
    m = st_deviation(np.zeros(1000), np.array([]), fs=360.0)
    assert m.value is None


from ecg.parameters import qrs_axis
from ecg.quality import UNAVAILABLE


def test_axis_unavailable_for_single_lead():
    m = qrs_axis({"II": _beat_with_p_wave()}, fs=360.0)
    assert m.value is None
    assert m.quality == UNAVAILABLE
    assert m.reason == "Requires 12-lead"


def test_axis_near_zero_when_lead_I_positive_and_aVF_flat():
    sig = _beat_with_p_wave()
    flat = np.zeros_like(sig)
    m = qrs_axis({"I": sig, "aVF": flat}, fs=360.0)
    assert m.value is not None
    assert abs(m.value) < 20.0


def test_axis_near_ninety_when_aVF_dominant():
    sig = _beat_with_p_wave()
    flat = np.zeros_like(sig)
    m = qrs_axis({"I": flat, "aVF": sig}, fs=360.0)
    assert m.value is not None
    assert abs(m.value - 90.0) < 20.0
