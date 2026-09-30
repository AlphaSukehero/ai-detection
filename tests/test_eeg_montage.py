"""A montage screenshot holds many channels, a grid and labels.

Treating the whole image as one trace averaged the ink of every channel,
every grid line and every label into one centroid per column -- a signal that
matched no channel and that every downstream metric was then computed on.
"""
import numpy as np
import pytest

from eeg.digitize import (CalibrationError, digitize, digitize_channels,
                          find_channel_bands, remove_grid)

FREQS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def _montage(n=10, width=1200, band_h=50, duration=10.0, grid=True, labels=True):
    """Stacked sine channels on white, with a dotted grid and label boxes."""
    h = n * band_h
    img = np.ones((h, width))
    t = np.linspace(0, duration, width)
    truth = []
    for ch in range(n):
        y = np.sin(2 * np.pi * FREQS[ch % len(FREQS)] * t)
        truth.append(y)
        centre = ch * band_h + band_h // 2
        rows = (centre - y * band_h * 0.3).astype(int)
        for c, r in enumerate(rows):
            img[r - 1:r + 1, c] = 0.0
        if labels:
            img[centre - 4:centre + 4, 2:18] = 0.0   # channel label glyph
    if grid:
        for c in range(0, width, 120):                # dotted vertical lines
            img[::3, c] = 0.3
    return img, truth


def test_grid_lines_are_removed():
    img, _ = _montage()
    cleaned = remove_grid(img)
    assert np.all(cleaned[:, ::120][::3] > 0.9)


def test_each_channel_band_is_found():
    img, _ = _montage(n=10)
    assert len(find_channel_bands(remove_grid(img))) == 10


def test_each_channel_is_recovered_separately():
    img, truth = _montage(n=10)
    signals, fs, _source, n_ch = digitize_channels(img, duration_s=10.0)
    assert n_ch == 10
    for rec, true in zip(signals, truth, strict=True):
        # Label boxes occupy the first columns; compare the trace region.
        r = np.corrcoef(rec[40:], true[40:])[0, 1]
        assert r > 0.9


def test_a_single_trace_image_still_works():
    img, truth = _montage(n=1, band_h=200, grid=False, labels=False)
    signals, _fs, _s, n_ch = digitize_channels(img, duration_s=10.0)
    assert n_ch == 1
    assert np.corrcoef(signals[0], truth[0])[0, 1] > 0.9


def test_digitize_returns_the_channel_mean_for_a_montage():
    img, _ = _montage(n=4)
    sig, fs, _ = digitize(img, duration_s=10.0)
    signals, fs2, _s, _n = digitize_channels(img, duration_s=10.0)
    assert fs == fs2
    assert np.allclose(sig, signals.mean(axis=0))


def test_a_blank_image_is_refused():
    with pytest.raises(CalibrationError):
        digitize_channels(np.ones((200, 400)), duration_s=10.0)
