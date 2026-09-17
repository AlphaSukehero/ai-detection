"""RR context must be patient-normalised and aligned 1:1 with the beats.

The point of these features is that an S beat is an *early* beat. A test
suite that only checked shapes would let a sign error or an off-by-one in the
pre/post assignment through, and that error would be invisible downstream --
it would just look like a model that never learned S.
"""
import numpy as np
import pytest

from ecg.beats import (rr_features, extract_beats, segment_signal,
                       BEAT_LEN, N_RR_FEATURES, BEFORE)


def _regular_peaks(n=40, rr=360):
    return np.arange(1, n + 1) * rr


# ------------------------------------------------------------- rr_features

def test_shape_matches_the_number_of_beats():
    f = rr_features(_regular_peaks(20))
    assert f.shape == (20, N_RR_FEATURES)


def test_a_perfectly_regular_rhythm_gives_neutral_ratios():
    f = rr_features(_regular_peaks(30))
    assert np.allclose(f, 1.0, atol=1e-5)


def test_a_premature_beat_has_a_short_pre_interval():
    """The defining property of an S beat, and the reason these features
    exist. Column 0 is pre_RR/median."""
    peaks = list(_regular_peaks(30))
    peaks[15] = peaks[14] + 180          # half the usual interval
    f = rr_features(np.array(peaks))
    assert f[15, 0] < 0.6, "premature beat must show a short preceding interval"
    assert f[14, 0] == pytest.approx(1.0, abs=0.05), "its predecessor is normal"


def test_a_premature_beat_is_followed_by_a_compensatory_pause():
    peaks = list(_regular_peaks(30))
    peaks[15] = peaks[14] + 180
    f = rr_features(np.array(peaks))
    assert f[15, 1] > 1.3, "the interval after a premature beat is long"
    assert f[15, 3] < 1.0, "pre/post ratio must be below 1 for a premature beat"


def test_features_are_ratios_so_two_patients_at_different_rates_match():
    """A 50 bpm record and a 100 bpm record with identical rhythm structure
    must produce identical features; that is what transfers across patients."""
    slow = rr_features(_regular_peaks(30, rr=720))
    fast = rr_features(_regular_peaks(30, rr=360))
    assert np.allclose(slow, fast, atol=1e-5)


def test_extreme_intervals_are_clipped_not_propagated():
    peaks = np.array([0, 360, 720, 100000, 100360])
    f = rr_features(peaks)
    assert np.isfinite(f).all()
    assert f.max() <= 4.0


@pytest.mark.parametrize("peaks,expected_rows", [(np.array([]), 0),
                                                 (np.array([500]), 1)])
def test_degenerate_inputs_do_not_raise(peaks, expected_rows):
    f = rr_features(peaks)
    assert f.shape == (expected_rows, N_RR_FEATURES)
    assert np.isfinite(f).all()


# ------------------------------------------------------------ extract_beats

def _synthetic_signal(peaks, length=None, fs=360.0):
    n = length or int(peaks[-1] + 1000)
    sig = np.random.default_rng(0).normal(0, 0.05, n)
    for p in peaks:
        sig[int(p)] += 5.0                      # a sharp R spike
    return sig


def test_beats_are_centred_on_the_peak():
    peaks = _regular_peaks(10)
    sig = _synthetic_signal(peaks)
    beats, feats, kept = extract_beats(sig, peaks)
    assert beats.shape[1] == BEAT_LEN
    # The R spike must land at BEFORE in every extracted window.
    assert all(int(np.argmax(b)) == BEFORE for b in beats)


def test_beats_and_features_stay_aligned_when_edge_beats_are_dropped():
    """A beat too close to the signal edge is dropped; its features must go
    with it, or every downstream label is off by one."""
    peaks = np.array([10, 400, 800, 1200])      # first has no room before it
    sig = _synthetic_signal(peaks, length=1400)
    beats, feats, kept = extract_beats(sig, peaks)
    assert len(beats) == len(feats) == len(kept)
    assert 0 not in kept, "the edge beat should have been dropped"


def test_dropped_edge_beats_still_inform_their_neighbours_intervals():
    """RR context is computed before dropping, so the surviving first beat
    knows the interval that preceded it."""
    peaks = np.array([10, 400, 800, 1200])
    sig = _synthetic_signal(peaks, length=1400)
    _, feats, kept = extract_beats(sig, peaks)
    full = rr_features(peaks)
    assert np.allclose(feats, full[kept])


def test_each_beat_is_z_scored():
    peaks = _regular_peaks(8)
    beats, _, _ = extract_beats(_synthetic_signal(peaks), peaks)
    assert np.allclose(beats.mean(axis=1), 0.0, atol=1e-4)
    assert np.allclose(beats.std(axis=1), 1.0, atol=1e-3)


def test_a_flat_segment_does_not_divide_by_zero():
    beats, _, _ = extract_beats(np.zeros(2000), np.array([500, 1000]))
    assert np.isfinite(beats).all()


def test_no_detectable_beats_returns_empty_arrays_not_an_error():
    beats, feats, kept = extract_beats(np.zeros(100), np.array([500]))
    assert beats.shape == (0, BEAT_LEN)
    assert feats.shape == (0, N_RR_FEATURES)


# ----------------------------------------------------------- segment_signal

def test_segment_signal_finds_beats_and_returns_matching_features():
    peaks = _regular_peaks(12)
    beats, feats = segment_signal(_synthetic_signal(peaks))
    assert len(beats) == len(feats) > 0
    assert beats.shape[1] == BEAT_LEN
    assert feats.shape[1] == N_RR_FEATURES
