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
