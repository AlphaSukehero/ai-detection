"""An image without a timebase must be refused, not guessed at.

The dangerous failure here is silent: assume a sampling rate and every band
edge scales by a constant, so a 3 Hz spike-wave discharge is reported as
whatever the wrong scale makes it, and the output looks entirely plausible.
"""
import numpy as np
import pytest

from eeg.digitize import (digitize, extract_trace, load_grayscale,
                          CalibrationError)


def _plot_image(freq=5.0, duration=10.0, width=1000, height=200, fs_render=None):
    """Render a sine as a white image with a black trace, like a plot export."""
    img = np.ones((height, width))
    t = np.linspace(0, duration, width)
    y = np.sin(2 * np.pi * freq * t)
    rows = ((1 - y) / 2 * (height - 20) + 10).astype(int)
    for c, r in enumerate(rows):
        img[max(0, r - 1):r + 2, c] = 0.0
    return img


# ----------------------------------------------------------- refusal path

def test_an_image_with_no_timebase_is_refused():
    with pytest.raises(CalibrationError, match="no timebase"):
        digitize(_plot_image())


def test_the_refusal_explains_why_and_what_to_do():
    with pytest.raises(CalibrationError) as e:
        digitize(_plot_image())
    msg = str(e.value)
    assert "hertz" in msg
    assert "EDF" in msg, "must point the user at the format that works"


def test_a_non_plot_image_is_refused():
    noise = np.ones((100, 100))
    with pytest.raises(CalibrationError, match="No continuous trace"):
        digitize(noise, duration_s=10.0)


def test_a_nonpositive_duration_is_refused():
    with pytest.raises(CalibrationError, match="must be positive"):
        digitize(_plot_image(), duration_s=0)


def test_an_implausible_derived_rate_is_refused():
    with pytest.raises(CalibrationError, match="implausible"):
        digitize(_plot_image(width=1000), duration_s=100000.0)


# ------------------------------------------------------- stated duration

def test_a_stated_duration_sets_the_sampling_rate():
    sig, fs, source = digitize(_plot_image(duration=10.0, width=1000),
                               duration_s=10.0)
    assert fs == pytest.approx(100.0)
    assert source == "stated duration"
    assert sig.size == 1000


def test_the_recovered_signal_has_the_right_frequency():
    """The whole point: a 5 Hz trace must measure as 5 Hz after digitising."""
    from eeg.metrics import power_spectrum
    sig, fs, _ = digitize(_plot_image(freq=5.0, duration=10.0, width=2000),
                          duration_s=10.0)
    freqs, power = power_spectrum(sig, fs)
    assert freqs[int(np.argmax(power))] == pytest.approx(5.0, abs=1.0)


def test_a_different_duration_rescales_the_measured_frequency():
    """Confirms the timebase actually drives the result -- which is exactly
    why it must never be assumed."""
    from eeg.metrics import power_spectrum
    img = _plot_image(freq=5.0, duration=10.0, width=2000)
    peaks = []
    for duration in (10.0, 20.0):
        sig, fs, _ = digitize(img, duration_s=duration)
        f, p = power_spectrum(sig, fs)
        peaks.append(f[int(np.argmax(p))])
    assert peaks[0] > peaks[1] * 1.5


def test_resampling_preserves_the_time_span():
    sig, fs, _ = digitize(_plot_image(duration=10.0, width=1000),
                          duration_s=10.0, target_fs=256.0)
    assert fs == 256.0
    assert sig.size / fs == pytest.approx(10.0, abs=0.05)


def test_resampling_preserves_the_measured_frequency():
    from eeg.metrics import power_spectrum
    img = _plot_image(freq=8.0, duration=10.0, width=3000)
    a_sig, a_fs, _ = digitize(img, duration_s=10.0)
    b_sig, b_fs, _ = digitize(img, duration_s=10.0, target_fs=128.0)
    fa, pa = power_spectrum(a_sig, a_fs)
    fb, pb = power_spectrum(b_sig, b_fs)
    assert fa[int(np.argmax(pa))] == pytest.approx(fb[int(np.argmax(pb))], abs=1.5)


# -------------------------------------------------------- trace extraction

def test_the_trace_is_not_vertically_inverted():
    """Image rows grow downward. Getting this wrong flips every waveform
    while leaving the power spectrum identical -- invisible in the metrics."""
    img = np.ones((100, 50))
    img[80:83, :25] = 0.0          # low on screen = negative voltage
    img[10:13, 25:] = 0.0          # high on screen = positive voltage
    trace = extract_trace(img)
    assert trace[:25].mean() < trace[25:].mean()


def test_columns_without_ink_are_interpolated_not_dropped():
    """A non-uniformly sampled series silently invalidates every spectral
    estimate downstream."""
    img = _plot_image(width=600)
    img[:, 200:230] = 1.0          # erase a slice of the trace
    trace = extract_trace(img)
    assert trace.size == 600
    assert np.all(np.isfinite(trace))


def test_grayscale_loading_accepts_colour_arrays():
    rgb = np.ones((20, 20, 3)) * 255
    assert load_grayscale(rgb).max() <= 1.0
