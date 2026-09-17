"""Clinical parameters must track the physiology they claim to measure.

Shape assertions would pass on a sign error or a swapped band edge, and that
error would surface as a confident Delta/Theta reading rather than as a
failure. Each test here drives the metric with a signal whose correct answer
is known from construction.
"""
import numpy as np
import pytest

from eeg.metrics import (relative_spectral_power, spectral_entropy,
                         spike_metrics, anomaly_episodes, window_report, BANDS)

FS = 256.0


def _sine(freq, seconds=4.0, fs=FS, amp=1.0):
    t = np.arange(0, seconds, 1 / fs)
    return amp * np.sin(2 * np.pi * freq * t)


# ------------------------------------------------- relative spectral power

@pytest.mark.parametrize("band,freq", [("delta", 2.0), ("theta", 6.0),
                                       ("alpha", 10.0), ("beta", 20.0),
                                       ("gamma", 38.0)])
def test_a_pure_tone_lands_in_its_own_band(band, freq):
    rsp = relative_spectral_power(_sine(freq), FS)
    assert rsp is not None
    assert rsp[band] == max(rsp.values()), f"{freq} Hz did not dominate {band}"
    assert rsp[band] > 0.5


def test_band_powers_sum_to_one():
    """They are shares of the analysis band, and the bands tile it exactly.
    A sum below 1 means a range of the denominator belongs to no band."""
    mixed = _sine(2.0) + _sine(10.0) + _sine(20.0) + _sine(38.0)
    rsp = relative_spectral_power(mixed, FS)
    assert sum(rsp.values()) == pytest.approx(1.0, abs=0.02)


def test_cortical_slowing_shows_as_a_delta_theta_shift():
    """The Alzheimer's biomarker: power migrating from alpha to delta/theta."""
    healthy = relative_spectral_power(_sine(10.0) * 3 + _sine(2.0), FS)
    slowed = relative_spectral_power(_sine(2.0) * 3 + _sine(10.0), FS)
    assert slowed["delta"] > healthy["delta"]
    assert slowed["alpha"] < healthy["alpha"]


def test_rsp_is_amplitude_invariant():
    """It is a ratio; doubling the gain must not change the distribution."""
    a = relative_spectral_power(_sine(10.0, amp=1.0), FS)
    b = relative_spectral_power(_sine(10.0, amp=50.0), FS)
    for band in BANDS:
        assert a[band] == pytest.approx(b[band], abs=1e-6)


def test_a_flat_signal_is_unmeasurable_not_uniform():
    """Returning 0.25 per band would look like a finding. None says nothing
    was measurable."""
    assert relative_spectral_power(np.zeros(1024), FS) is None


# ----------------------------------------------------- spectral entropy

def test_a_pure_tone_has_low_entropy():
    assert spectral_entropy(_sine(10.0), FS) < 0.35


def test_broadband_noise_has_high_entropy():
    noise = np.random.default_rng(0).normal(0, 1, int(FS * 4))
    assert spectral_entropy(noise, FS) > 0.8


def test_entropy_orders_tone_below_noise():
    """The property the dashboard relies on: a rhythmic seizure discharge is
    more ordered than resting background."""
    noise = np.random.default_rng(1).normal(0, 1, int(FS * 4))
    assert spectral_entropy(_sine(10.0), FS) < spectral_entropy(noise, FS)


def test_entropy_is_normalised_so_window_lengths_compare():
    short = spectral_entropy(_sine(10.0, seconds=2.0), FS)
    long = spectral_entropy(_sine(10.0, seconds=8.0), FS)
    assert short == pytest.approx(long, abs=0.15)
    assert 0.0 <= short <= 1.0


def test_entropy_of_a_flat_signal_is_unmeasurable():
    assert spectral_entropy(np.zeros(1024), FS) is None


# ---------------------------------------------------------- spike metrics

def test_no_spikes_in_smooth_background():
    assert spike_metrics(_sine(10.0), FS)["count"] == 0


def test_injected_spikes_are_counted():
    sig = _sine(10.0, seconds=4.0)
    positions = [200, 500, 800]
    for p in positions:
        sig[p] += 40.0
    assert spike_metrics(sig, FS)["count"] == len(positions)


def test_rate_is_per_second_not_per_window():
    sig = np.zeros(int(FS * 4))
    sig += np.random.default_rng(2).normal(0, 0.01, sig.size)
    for p in range(100, 1000, 300):
        sig[p] += 10.0
    m = spike_metrics(sig, FS)
    assert m["rate_per_s"] == pytest.approx(m["count"] / 4.0, abs=1e-6)


def test_a_dense_burst_does_not_raise_its_own_threshold():
    """With a plain standard deviation the spikes inflate sigma and hide
    themselves. The MAD-based sigma is why this works."""
    sig = np.random.default_rng(3).normal(0, 0.1, int(FS * 4))
    for p in range(100, 900, 25):
        sig[p] += 8.0
    assert spike_metrics(sig, FS)["count"] >= 20


def test_a_flat_signal_yields_no_spikes_and_does_not_divide_by_zero():
    m = spike_metrics(np.zeros(1024), FS)
    assert m["count"] == 0 and np.isfinite(m["rate_per_s"])


# --------------------------------------------------------- onset/duration

def _times(n, window_s=2.0, overlap=0.5):
    step = window_s * (1 - overlap)
    return [(i * step, i * step + window_s) for i in range(n)]


def test_consecutive_flags_merge_into_one_episode():
    """Overlapping windows describe one event, not several."""
    flags = [0, 0, 1, 1, 1, 0, 0]
    eps = anomaly_episodes(flags, _times(7))
    assert len(eps) == 1
    assert eps[0]["n_windows"] == 3


def test_onset_and_offset_span_the_flagged_windows():
    flags = [0, 1, 1, 0]
    t = _times(4)
    ep = anomaly_episodes(flags, t)[0]
    assert ep["onset_s"] == t[1][0]
    assert ep["offset_s"] == t[2][1]
    assert ep["duration_s"] == pytest.approx(t[2][1] - t[1][0])


def test_separate_bursts_stay_separate():
    eps = anomaly_episodes([1, 0, 0, 1, 1], _times(5))
    assert len(eps) == 2


def test_an_episode_running_to_the_end_is_closed():
    eps = anomaly_episodes([0, 1, 1], _times(3))
    assert len(eps) == 1 and eps[0]["last_window"] == 2


def test_short_episodes_can_be_filtered():
    assert anomaly_episodes([1, 0, 0], _times(3), min_duration_s=5.0) == []


def test_mismatched_flags_and_times_are_refused():
    """Pairing a detection with another window's timestamp would put the
    anomaly at the wrong moment in the recording."""
    with pytest.raises(ValueError, match="refusing"):
        anomaly_episodes([1, 0, 1], _times(2))


# --------------------------------------------------------------- combined

def test_window_report_carries_every_parameter():
    r = window_report(_sine(10.0), FS)
    assert set(r) == {"rsp", "theta_alpha_ratio", "spectral_entropy", "spikes"}
    assert r["rsp"] is not None and r["spectral_entropy"] is not None


def test_window_report_on_a_flat_window_reports_absence():
    r = window_report(np.zeros(1024), FS)
    assert r["rsp"] is None
    assert r["spectral_entropy"] is None
    assert r["theta_alpha_ratio"] is None
