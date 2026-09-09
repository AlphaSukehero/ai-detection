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


from PIL import Image
from scipy.signal import find_peaks as _find_peaks

LEAD_NAMES_12 = ["I", "II", "III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]

# Physical panel positions on a standard 3x4 printed sheet. The columns are
# limb / augmented / precordial groups, so reading the sheet row-major gives
# I, aVR, V1, V4 across the top row — NOT the clinical listing order above.
# Task 10 reads leads I and aVF from here, so a transposed mapping would
# silently yield a wrong axis in degrees with no visible error.
LEAD_PANELS_3X4 = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


def _trace_row_band(ink_band):
    """Column-wise centre of ink mass within one horizontal band."""
    h = ink_band.shape[0]
    rows = np.arange(h, dtype=float)
    out = np.full(ink_band.shape[1], h / 2.0)
    thr = np.percentile(ink_band, 88)
    found_any = False
    for x in range(ink_band.shape[1]):
        col = ink_band[:, x]
        m = col >= max(thr, 1.0)
        if m.sum() >= 1 and col[m].sum() > 1e-6:
            out[x] = float(np.average(rows[m], weights=col[m]))
            found_any = True
    if not found_any:
        # No ink anywhere in this band: report nothing rather than a flat
        # zero trace, which would read downstream as a real isoelectric lead.
        return None
    sig = (h - 1.0) - out
    sig = sig - np.mean(sig)
    std = float(np.std(sig))
    return sig / std if std > 1e-9 else sig


def detect_layout(gray):
    """Distinguish a rhythm strip from a 3x4 twelve-lead sheet.

    A twelve-lead sheet has several well-separated horizontal trace bands; a
    rhythm strip has one. Counting ink-density bands is more robust than
    trying to find panel borders, which many printouts omit.
    """
    arr = np.asarray(gray, dtype=float)
    ink = 255.0 - arr
    rows = ink.mean(axis=1)
    if np.std(rows) < 1e-6:
        return "single"
    rows = rows - rows.mean()
    bands, _ = _find_peaks(rows, distance=max(1, arr.shape[0] // 12),
                           prominence=float(np.std(rows)))
    return "twelve_lead" if len(bands) >= 3 else "single"


def extract_leads(image_path):
    """Digitize an ECG image into one signal per lead."""
    img = Image.open(image_path).convert("L")
    gray = np.asarray(img, dtype=float)
    px_per_mm = detect_grid_scale(gray)
    layout = detect_layout(gray)
    ink = 255.0 - gray

    if layout == "single":
        return {"leads": {"II": _trace_row_band(ink)},
                "layout": layout, "px_per_mm": px_per_mm}

    h, w = ink.shape
    leads = {}
    for r in range(3):
        band = ink[r * h // 3:(r + 1) * h // 3, :]
        for c in range(4):
            col = band[:, c * w // 4:(c + 1) * w // 4]
            leads[LEAD_PANELS_3X4[r][c]] = _trace_row_band(col)
    return {"leads": leads, "layout": layout, "px_per_mm": px_per_mm}
