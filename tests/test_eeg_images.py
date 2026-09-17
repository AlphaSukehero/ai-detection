"""Time-frequency images must place energy where the signal actually is.

A shape-only test passes on a transposed axis or an inverted frequency grid,
and the model would still train -- on a systematically wrong picture.
"""
import numpy as np
import pytest

from eeg.images import scalogram, spectrogram, render, render_batch, IMAGE_SIZE

FS = 256.0


def _sine(freq, seconds=2.0, fs=FS):
    t = np.arange(0, seconds, 1 / fs)
    return np.sin(2 * np.pi * freq * t)


@pytest.mark.parametrize("fn", [scalogram, spectrogram])
def test_output_has_the_requested_shape_and_range(fn):
    img = fn(_sine(10.0), FS)
    assert img.shape == IMAGE_SIZE
    assert img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0


@pytest.mark.parametrize("fn", [scalogram, spectrogram])
def test_a_low_tone_and_a_high_tone_energise_different_rows(fn):
    """If the frequency axis were collapsed or inverted these would match."""
    low = fn(_sine(2.0), FS).mean(axis=1)
    high = fn(_sine(30.0), FS).mean(axis=1)
    assert int(np.argmax(low)) != int(np.argmax(high))


def test_scalogram_frequency_axis_is_ordered_low_to_high():
    """Row index must increase with frequency; an inverted axis would make
    every delta finding a gamma finding."""
    rows = [int(np.argmax(scalogram(_sine(f), FS).mean(axis=1)))
            for f in (2.0, 8.0, 30.0)]
    assert rows == sorted(rows), f"frequency axis not monotonic: {rows}"


def test_a_transient_is_localised_in_time():
    """A spike late in the window must not light up the early columns."""
    x = np.zeros(int(FS * 2))
    x += np.random.default_rng(0).normal(0, 0.01, x.size)
    x[int(FS * 1.6)] += 20.0
    cols = scalogram(x, FS).mean(axis=0)
    assert int(np.argmax(cols)) > len(cols) // 2


def test_normalisation_is_per_image_so_gain_does_not_matter():
    """Electrode impedance changes amplitude, not physiology."""
    a = scalogram(_sine(10.0), FS)
    b = scalogram(_sine(10.0) * 100.0, FS)
    assert np.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("fn", [scalogram, spectrogram])
def test_a_flat_window_renders_blank_rather_than_raising(fn):
    img = fn(np.zeros(int(FS * 2)), FS)
    assert img.shape == IMAGE_SIZE and np.all(img == 0)


@pytest.mark.parametrize("fn", [scalogram, spectrogram])
def test_a_too_short_window_renders_blank(fn):
    assert fn(np.zeros(3), FS).shape == IMAGE_SIZE


def test_batch_rendering_adds_the_channel_axis():
    windows = np.stack([_sine(10.0), _sine(3.0)])[:, np.newaxis, :]
    batch = render_batch(windows, FS)
    assert batch.shape == (2,) + IMAGE_SIZE + (1,)


def test_batch_averages_multi_channel_windows():
    w = np.stack([np.stack([_sine(10.0), _sine(10.0)])])
    assert render_batch(w, FS).shape == (1,) + IMAGE_SIZE + (1,)


def test_empty_batch_keeps_the_model_input_shape():
    assert render_batch([], FS).shape == (0,) + IMAGE_SIZE + (1,)


def test_unknown_representation_is_refused():
    with pytest.raises(ValueError, match="unknown representation"):
        render(_sine(10.0), FS, kind="mel")
