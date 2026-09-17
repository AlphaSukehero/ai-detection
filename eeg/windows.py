"""Sliding-window segmentation, shared by training and inference.

One implementation for both paths. The ECG side of this repository learned
the cost of two: training centred every beat on its R peak while inference
chopped the trace into arbitrary blocks, and nothing raised -- the model just
quietly meant less than it appeared to. Windowing is the equivalent seam
here, so it lives in one place.
"""
import numpy as np

DEFAULT_WINDOW_S = 2.0
DEFAULT_OVERLAP = 0.5


def window_bounds(n_samples, fs, window_s=DEFAULT_WINDOW_S,
                  overlap=DEFAULT_OVERLAP):
    """Start/stop sample indices of each window, as a list of (start, stop).

    Only complete windows are returned. A trailing partial window is dropped
    rather than zero-padded: padding invents signal, and a spectral estimate
    over invented signal is indistinguishable from a real one downstream.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    width = int(round(window_s * fs))
    if width <= 0:
        raise ValueError(f"window_s * fs must be at least 1 sample, got {width}")
    step = max(1, int(round(width * (1.0 - overlap))))

    return [(s, s + width) for s in range(0, max(0, n_samples - width + 1), step)]


def window_times(n_samples, fs, window_s=DEFAULT_WINDOW_S,
                 overlap=DEFAULT_OVERLAP):
    """(start_seconds, stop_seconds) per window -- the clinical timeline.

    Every anomaly timestamp the dashboard reports is traced back through
    these, so they are derived from the same bounds the model actually saw.
    """
    return [(a / fs, b / fs)
            for a, b in window_bounds(n_samples, fs, window_s, overlap)]


def segment(signal, fs, window_s=DEFAULT_WINDOW_S, overlap=DEFAULT_OVERLAP):
    """Split a 1-D or (channels, samples) signal into windows.

    Returns (windows, times) where windows has shape
    (n_windows, channels, width) -- channels first and always present, so
    downstream code never has to branch on single- versus multi-channel.
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[np.newaxis, :]
    if sig.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D signal, got shape {sig.shape}")

    bounds = window_bounds(sig.shape[1], fs, window_s, overlap)
    if not bounds:
        return (np.zeros((0, sig.shape[0], 0)), [])

    windows = np.stack([sig[:, a:b] for a, b in bounds])
    times = [(a / fs, b / fs) for a, b in bounds]
    return windows, times
