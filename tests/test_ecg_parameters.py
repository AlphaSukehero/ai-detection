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
        sig[r] = 3.0                       # R peak
        sig[r - 1] = sig[r + 1] = 1.0
        p = r - int(0.16 * fs)             # P wave 160 ms before R
        if p > 2:
            sig[p] = 0.45
            sig[p - 1] = sig[p + 1] = 0.25
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
