import numpy as np
from ecg.delineate import detect_r_peaks


def _synth_ecg(n_beats=10, fs=360.0, rr=0.8):
    """Build a signal with sharp R spikes at exactly known positions."""
    n = int(n_beats * rr * fs)
    sig = np.zeros(n)
    idx = (np.arange(n_beats) * rr * fs).astype(int)
    idx = idx[idx < n]
    for i in idx:
        sig[i] = 3.0
        if i > 0:
            sig[i - 1] = 1.0
        if i + 1 < n:
            sig[i + 1] = 1.0
    return sig, idx


def test_detects_all_r_peaks_at_known_positions():
    sig, truth = _synth_ecg()
    peaks = detect_r_peaks(sig, fs=360.0)
    assert len(peaks) == len(truth)
    assert np.all(np.abs(peaks - truth) <= 2)


def test_returns_empty_array_for_flat_signal():
    peaks = detect_r_peaks(np.zeros(1000), fs=360.0)
    assert len(peaks) == 0
