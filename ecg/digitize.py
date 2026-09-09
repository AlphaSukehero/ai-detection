"""Convert ECG images into calibrated signals."""
import numpy as np

MIN_PX_PER_MM = 2.0
MAX_PX_PER_MM = 40.0


def detect_grid_scale(gray):
    """Pixels per millimetre, from the dominant period of the printed grid.

    The grid is strongly periodic, so its spacing shows up as a clear peak in
    the FFT of the column-ink profile. Returns None when no such peak stands
    out, which forces every time-based parameter to report "—" rather than
    silently assuming a scale.
    """
    arr = np.asarray(gray, dtype=float)
    ink = 255.0 - arr
    profile = ink.mean(axis=0)
    profile = profile - profile.mean()
    if np.std(profile) < 1e-6:
        return None

    spectrum = np.abs(np.fft.rfft(profile))
    freqs = np.fft.rfftfreq(len(profile), d=1.0)
    # Only consider periods that could plausibly be 1 mm rulings.
    valid = (freqs > 1.0 / MAX_PX_PER_MM) & (freqs < 1.0 / MIN_PX_PER_MM)
    if not np.any(valid):
        return None

    band = spectrum[valid]
    peak = float(np.max(band))
    if peak < 4.0 * float(np.median(band)):
        return None
    return float(1.0 / freqs[valid][int(np.argmax(band))])
