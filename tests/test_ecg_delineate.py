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


from ecg.delineate import t_end


def test_t_end_rejects_near_flat_t_wave():
    """A near-zero descent slope must yield None, not a window-edge clamp.

    Extrapolating a tangent with a tiny slope puts T-end far past the search
    window. Clamping to the window edge returns a fabricated index that, at
    360 Hz, lands squarely inside the 0.20-0.65 s QT plausibility band and is
    then reported as a measured, OK-quality QT.
    """
    fs = 360.0
    n = 720
    sig = np.zeros(n)
    qrs_offset = 100
    # A very broad, very shallow bump: apex early, descent almost flat.
    for i in range(qrs_offset + 20, n):
        sig[i] = 0.02 * np.exp(-((i - (qrs_offset + 40)) ** 2) / (2 * 400.0 ** 2))
    assert t_end(sig, qrs_offset, rr_s=1.0, fs=fs) is None
