import numpy as np
from ecg.delineate import detect_r_peaks, qrs_bounds


def _synth_ecg(n_beats=10, fs=360.0, rr=0.8):
    """Build a signal with sharp R spikes at exactly known positions."""
    # Offset the first beat off sample 0: find_peaks needs neighbours on both
    # sides, so a peak at index 0 is undetectable by construction.
    lead_in = 100
    n = int(n_beats * rr * fs) + lead_in
    sig = np.zeros(n)
    idx = (np.arange(n_beats) * rr * fs).astype(int) + lead_in
    idx = idx[idx < n - 1]
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


def test_qrs_bounds_bracket_the_r_peak():
    fs = 360.0
    sig = np.zeros(720)
    # A dense triangular QRS ~80 ms wide centred at index 360. Every sample in
    # the complex is filled: a real trace is contiguous, and a boundary walk
    # must not be able to halt on a gap between spikes.
    half = int(0.04 * fs)                      # 40 ms each side
    for off in range(-half, half + 1):
        sig[360 + off] = 3.0 * (1.0 - abs(off) / (half + 1.0))
    onset, offset = qrs_bounds(sig, peak=360, fs=fs)
    assert onset < 360 < offset
    width_s = (offset - onset) / fs
    assert 0.03 <= width_s <= 0.20
