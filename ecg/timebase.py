"""Where an ECG recording's timebase comes from.

Every interval and the heart rate are samples divided by a sampling rate, so
a wrong rate is a wrong answer with nothing on the page to show for it. The
rate is taken from the strongest evidence available and the source is named
in the result.
"""
import numpy as np

DEFAULT_FS = 360.0          # MIT-BIH, the rate the classifier was trained at
PLAUSIBLE_FS = (50.0, 5000.0)
# A stated duration within this share of what the printed grid measures is
# the same recording described more exactly; beyond it, one of them is wrong.
GRID_AGREEMENT = 0.20


def _plausible(fs):
    return PLAUSIBLE_FS[0] <= fs <= PLAUSIBLE_FS[1]


def _time_axis_fs(column):
    """Sampling rate a monotonic first column implies, or None for an index."""
    step = float(np.median(np.diff(column)))
    if step == 1.0:
        # 0, 1, 2, ... counts samples. It could be milliseconds at 1 kHz, but
        # nothing in the file says so, and a guess here is a silent 3x error.
        return None, None
    for unit, per_second in (("s", 1.0), ("ms", 1000.0)):
        fs = per_second / step
        if _plausible(fs):
            return fs, unit
    return None, None


def table_signal(values, duration_s):
    """(signal, fs, source) for the numeric table read from an ECG data file.

    A first column that only ever increases is a time axis or a sample index,
    not a lead: read as the signal it is a ramp with no heartbeat in it. The
    first remaining column is the lead analysed, as the classifier is trained
    on one.
    """
    table = np.asarray(values, dtype=float)
    if table.ndim == 1:
        table = table.reshape(-1, 1)
    if table.shape[0] < table.shape[1]:
        table = table.T                         # one row per lead -> columns
    axis_fs = unit = None
    if table.shape[1] > 1 and table.shape[0] > 2 and np.all(np.diff(table[:, 0]) > 0):
        axis_fs, unit = _time_axis_fs(table[:, 0])
        table = table[:, 1:]
    signal = table[:, 0]

    if duration_s:
        fs = len(signal) / duration_s
        if not _plausible(fs):
            raise ValueError(
                f"The recording duration does not fit this file: {len(signal)} "
                f"samples in {duration_s:g} s is {fs:.4g} Hz, outside the "
                f"{PLAUSIBLE_FS[0]:g}-{PLAUSIBLE_FS[1]:g} Hz an ECG is recorded "
                "at. Check the duration entered with the patient details.")
        return signal, fs, f"{fs:.4g} Hz from the stated duration {duration_s:g} s"
    if axis_fs:
        return signal, axis_fs, f"{axis_fs:.4g} Hz from the file's time column ({unit})"
    return signal, DEFAULT_FS, (
        f"{DEFAULT_FS:g} Hz assumed; enter the recording duration if the file "
        "was sampled at another rate")


def image_timebase(width_px, grid_px_per_mm, duration_s, paper_speed):
    """(px_per_mm, description, warning) for a digitised ECG image.

    The printed grid is measured from the page; a duration is typed in. When
    both are present and agree, the duration is the more exact of the two.
    When they disagree the grid is kept and the disagreement is reported:
    believing the typed figure turned a 72 bpm strip into 29 bpm.
    """
    if not duration_s:
        if grid_px_per_mm is None:
            return None, "none", None
        return grid_px_per_mm, f"detected grid at {paper_speed:g} mm/s", None
    stated = width_px / duration_s / paper_speed
    if grid_px_per_mm is None:
        return stated, f"stated duration {duration_s:g} s at {paper_speed:g} mm/s", None
    grid_s = width_px / grid_px_per_mm / paper_speed
    if abs(duration_s - grid_s) <= GRID_AGREEMENT * grid_s:
        return stated, f"stated duration {duration_s:g} s at {paper_speed:g} mm/s", None
    return (grid_px_per_mm, f"detected grid at {paper_speed:g} mm/s",
            f"The recording duration entered ({duration_s:g} s) does not match "
            f"the printed grid, which measures {grid_s:.1f} s across the trace "
            f"at {paper_speed:g} mm/s. The grid was used. If the grid is not a "
            "standard 1 mm ECG grid, check the paper speed.")
