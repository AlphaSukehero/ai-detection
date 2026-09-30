"""Recover a signal from a picture of an EEG trace, or refuse.

Every parameter this system reports -- relative spectral power, spectral
entropy, spike rate -- is defined over frequency, and frequency is cycles per
second. A PNG contains pixels, not seconds. So an uploaded plot can only
yield those numbers if the pixel-to-second scale is known, and this module's
main job is to establish that scale honestly or decline.

Three ways the timebase can be established, in descending order of trust:

  1. The caller states the recording duration (the clinician knows it, and it
     is printed on most clinical exports). Exact, and always preferred.
  2. A calibration grid is detected in the image and the paper speed is
     known, giving px/mm and therefore px/s.
  3. Nothing is available -- in which case this module raises. It does not
     assume a default, because a wrong assumption does not look wrong: every
     interval and every band edge scales by a constant and the output stays
     entirely plausible.

The ECG side of this repository reaches the same conclusion for the same
reason; see ecg/digitize.py and the calibration note in ecg/parameters.py.
"""
import numpy as np

from PIL import Image


class CalibrationError(RuntimeError):
    """The image carries no usable timebase, so no frequency is recoverable."""


MIN_TRACE_COLUMNS = 0.5      # share of columns that must contain ink
GRID_PEAK_RATIO = 3.0        # grid line prominence over the column median


def load_grayscale(path_or_array):
    """Read an image to a float array in [0, 1], 0 = ink."""
    if isinstance(path_or_array, np.ndarray):
        arr = path_or_array
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        arr = arr.astype(np.float64)
        return arr / 255.0 if arr.max() > 1.0 else arr
    img = Image.open(path_or_array).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0


def detect_grid_spacing(gray):
    """Median spacing in pixels between vertical grid lines, or None.

    Grid detection is a convenience, not a guarantee. Screenshots of digital
    EEG viewers usually have no printed grid at all, which is exactly why the
    caller-supplied duration path exists and is preferred.
    """
    ink = 1.0 - gray
    column_strength = ink.sum(axis=0)
    if column_strength.size < 16:
        return None
    median = float(np.median(column_strength))
    if median <= 0:
        return None
    peaks = np.flatnonzero(column_strength > GRID_PEAK_RATIO * median)
    if peaks.size < 4:
        return None
    gaps = np.diff(peaks)
    gaps = gaps[gaps > 1]
    if gaps.size < 3:
        return None
    spacing = float(np.median(gaps))
    return spacing if spacing >= 2.0 else None


def extract_trace(gray):
    """One amplitude per pixel column, in pixel units, centred on zero.

    The darkest pixel in each column is taken as the trace. Columns with no
    ink are interpolated from their neighbours rather than dropped, so the
    returned series stays uniformly sampled in time -- a non-uniform series
    would silently invalidate every spectral estimate downstream.
    """
    ink = 1.0 - gray
    threshold = max(0.25, float(np.percentile(ink, 99)) * 0.4)
    rows = []
    for col in range(ink.shape[1]):
        column = ink[:, col]
        if column.max() < threshold:
            rows.append(np.nan)
        else:
            rows.append(float(np.average(np.flatnonzero(column >= threshold))))

    trace = np.asarray(rows, dtype=np.float64)
    present = np.isfinite(trace)
    if present.mean() < MIN_TRACE_COLUMNS:
        raise CalibrationError(
            "No continuous trace was found in this image. It may not be an "
            "EEG plot, or the trace may be too faint to follow."
        )
    idx = np.arange(trace.size)
    trace = np.interp(idx, idx[present], trace[present])
    # Image rows increase downward; invert so a rising trace is a rising
    # voltage. Getting this backwards flips every waveform vertically while
    # leaving the power spectrum identical -- invisible in the metrics.
    return -(trace - float(np.mean(trace)))


GRID_BLOCKS = 20             # a grid column has ink in nearly every block
GRID_BLOCK_SHARE = 0.8
GRID_ROW_SHARE = 0.7         # a horizontal rule spans most of the width
BAND_ROW_SHARE = 0.02        # a row is "active" above this share of peak ink
BAND_MIN_ROWS = 5
BAND_MERGE_GAP = 2


def remove_grid(gray):
    """Blank out grid lines so they are not read as signal.

    A vertical grid line is inked in almost every horizontal block of the
    image, including the blank gaps between channels; a trace column is only
    inked where a channel is. Horizontal rules span most of the width. Both
    are painted white; the trace gaps this leaves are interpolated later,
    exactly like any other ink-free column.
    """
    ink = 1.0 - gray
    inked = ink > 0.25
    h, w = inked.shape
    blocks = np.array_split(np.arange(h), min(GRID_BLOCKS, h))
    coverage = np.mean([inked[b].any(axis=0) for b in blocks], axis=0)
    out = gray.copy()
    out[:, coverage >= GRID_BLOCK_SHARE] = 1.0
    out[inked.mean(axis=1) >= GRID_ROW_SHARE, :] = 1.0
    return out


def find_channel_bands(gray):
    """Row ranges (start, stop) of each channel in a stacked montage.

    Channels are separated by rows with (almost) no ink. A run that does not
    carry ink across most of the width -- a row of timestamps, a caption --
    is not a channel and is dropped.
    """
    inked = (1.0 - gray) > 0.25
    profile = inked.sum(axis=1).astype(float)
    if profile.max() <= 0:
        return []
    active = profile > BAND_ROW_SHARE * profile.max()

    runs, start = [], None
    for r, on in enumerate(active):
        if on and start is None:
            start = r
        elif not on and start is not None:
            runs.append([start, r])
            start = None
    if start is not None:
        runs.append([start, active.size])

    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= BAND_MERGE_GAP:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    bands = []
    for r0, r1 in merged:
        if r1 - r0 < BAND_MIN_ROWS:
            continue
        if inked[r0:r1].any(axis=0).mean() < MIN_TRACE_COLUMNS:
            continue
        bands.append((r0, r1))
    return bands


def _timebase(gray, n, duration_s, px_per_mm, paper_speed_mm_s):
    """(fs, source) for a trace of n columns, or raise CalibrationError."""
    if duration_s is not None:
        if duration_s <= 0:
            raise CalibrationError("Recording duration must be positive.")
        return n / float(duration_s), "stated duration"
    if px_per_mm is None:
        px_per_mm = detect_grid_spacing(gray)
    if px_per_mm is None:
        raise CalibrationError(
            "This image has no detectable calibration grid and no "
            "recording duration was supplied, so there is no timebase. "
            "Every frequency-domain measurement (relative spectral "
            "power, spectral entropy, spike rate) is defined in hertz "
            "and cannot be computed without one. Re-upload with the "
            "recording duration, or upload the EDF/BDF file instead."
        )
    return (float(px_per_mm) * float(paper_speed_mm_s),
            f"grid ({px_per_mm:.1f} px/mm at {paper_speed_mm_s} mm/s)")


def _resample(trace, fs, target_fs):
    if target_fs and target_fs > 0 and abs(fs - target_fs) > 1e-9:
        # Time span is preserved, so every timestamp the dashboard reports
        # still refers to the real recording.
        n = trace.shape[-1]
        new_n = max(2, int(round(n / fs * target_fs)))
        grid = np.linspace(0, n - 1, new_n)
        idx = np.arange(n)
        if trace.ndim == 1:
            return np.interp(grid, idx, trace), float(target_fs)
        return (np.stack([np.interp(grid, idx, t) for t in trace]),
                float(target_fs))
    return trace, fs


def digitize_channels(path_or_array, duration_s=None, px_per_mm=None,
                      paper_speed_mm_s=30.0, target_fs=None):
    """Image -> (signals[n_channels, n], fs, source, n_channels).

    Grid lines are removed, the image is split into channel bands, and each
    band is traced on its own. Averaging ink across bands is what produced
    a signal matching no channel; tracing per band is the fix.
    """
    gray = load_grayscale(path_or_array)
    cleaned = remove_grid(gray)
    traces = []
    for r0, r1 in find_channel_bands(cleaned):
        try:
            traces.append(extract_trace(cleaned[r0:r1]))
        except CalibrationError:
            continue
    if not traces:
        raise CalibrationError(
            "No continuous trace was found in this image. It may not be an "
            "EEG plot, or the trace may be too faint to follow."
        )
    signals = np.stack(traces)
    fs, source = _timebase(gray, signals.shape[1], duration_s, px_per_mm,
                           paper_speed_mm_s)
    if fs <= 1.0:
        raise CalibrationError(
            f"Derived sampling rate {fs:.3f} Hz is implausible; the timebase "
            "for this image cannot be trusted."
        )
    signals, fs = _resample(signals, fs, target_fs)
    return signals, fs, source, len(traces)


def digitize(path_or_array, duration_s=None, px_per_mm=None,
             paper_speed_mm_s=30.0, target_fs=None):
    """Image -> (signal, fs, source): the mean of every detected channel.

    EEG paper speed is conventionally 30 mm/s, unlike the 25 mm/s of ECG.
    Channels are averaged, as the EDF path does: this pipeline localises in
    time, not space.
    """
    signals, fs, source, _n = digitize_channels(
        path_or_array, duration_s=duration_s, px_per_mm=px_per_mm,
        paper_speed_mm_s=paper_speed_mm_s, target_fs=target_fs)
    return signals.mean(axis=0), fs, source

