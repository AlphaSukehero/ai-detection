# ECG Parameter Extraction (Sub-project A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ECG page's fabricated parameters with measured ones, adding Rhythm, ST and Axis, each carrying an honest quality flag.

**Architecture:** A new `ecg/` package separates delineation from classification. `delineate.py` finds fiducial points on a signal, `parameters.py` turns fiducials into the seven displayed values, `digitize.py` converts an image into calibrated signals. `app.py` only calls them and renders the result.

**Tech Stack:** Python 3.12, NumPy, SciPy (`find_peaks`), OpenCV, Pillow, Flask, pytest. All already installed in `.venv`.

## Global Constraints

- Sampling rate default is **360 Hz** (MIT-BIH), passed explicitly, never assumed inside helpers.
- A parameter that cannot be measured is `None` with a quality flag — **never** a hardcoded or guessed number.
- Quality flag values are exactly `"ok"`, `"low_confidence"`, `"unavailable"`.
- QTc uses **Fridericia** (`QT / RR**(1/3)`), and the displayed string names the formula.
- ST: Elevated if `> +1.0 mm`, Depressed if `< -1.0 mm`, else Normal.
- Rhythm: Regular if RR coefficient of variation `< 0.10`, else Irregular.
- Axis requires 12-lead; single-lead yields `None` with reason `"Requires 12-lead"`.
- Run tests with `.venv/bin/python -m pytest`. Never bare `pytest`, and never
  `.venv/bin/pytest` — that console script has a stale shebang pointing at a
  previous checkout path and cannot execute.
- Every measurement function takes a signal in **millivolt-normalised units** and returns times in **seconds**; only the formatting layer converts to ms.

---

### Task 1: Package scaffold and the quality-flag type

**Files:**
- Create: `ecg/__init__.py`
- Create: `ecg/quality.py`
- Test: `tests/test_ecg_quality.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Measurement(value: float | None, quality: str, reason: str = "")` dataclass with method `format(unit: str, decimals: int = 1) -> str` returning `"—"` when value is None, else `f"{value:.{decimals}f} {unit}"`. Constants `OK = "ok"`, `LOW = "low_confidence"`, `UNAVAILABLE = "unavailable"`.

- [ ] **Step 1: Write the failing test**

```python
from ecg.quality import Measurement, OK, UNAVAILABLE


def test_measurement_formats_value_with_unit():
    m = Measurement(value=0.156, quality=OK)
    assert m.format("s", decimals=3) == "0.156 s"


def test_unavailable_measurement_renders_dash():
    m = Measurement(value=None, quality=UNAVAILABLE, reason="P wave absent")
    assert m.format("ms") == "—"
    assert m.reason == "P wave absent"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ecg_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecg'`

- [ ] **Step 3: Write minimal implementation**

Create empty `ecg/__init__.py`, then `ecg/quality.py`:

```python
"""Measured values that can honestly report their own reliability."""
from dataclasses import dataclass

OK = "ok"
LOW = "low_confidence"
UNAVAILABLE = "unavailable"


@dataclass
class Measurement:
    value: float | None
    quality: str
    reason: str = ""

    def format(self, unit: str, decimals: int = 1) -> str:
        if self.value is None:
            return "—"
        return f"{self.value:.{decimals}f} {unit}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_quality.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add ecg/__init__.py ecg/quality.py tests/test_ecg_quality.py
git commit -m "feat: add Measurement type carrying quality flags for ECG parameters"
```

---

### Task 2: R-peak detection validated against MIT-BIH annotations

**Files:**
- Create: `ecg/delineate.py`
- Test: `tests/test_ecg_delineate.py`

**Interfaces:**
- Consumes: nothing
- Produces: `detect_r_peaks(signal: np.ndarray, fs: float = 360.0) -> np.ndarray` returning integer sample indices of R-peaks.

- [ ] **Step 1: Write the failing test**

Uses a synthetic signal with known peak spacing, so the test needs no data files.

```python
import numpy as np
from ecg.delineate import detect_r_peaks


def _synth_ecg(n_beats=10, fs=360.0, rr=0.8):
    """Build a signal with sharp R spikes at exactly known positions."""
    # Offset the first beat off sample 0: find_peaks needs neighbours on both
    # sides, so a peak at index 0 is undetectable by construction.
    lead_in = 100
    n = int(n_beats * rr * fs) + lead_in
    sig = np.zeros(n)
    idx = (np.arange(n_beats) * rr * fs).astype(int) + lead_in
    idx = idx[idx < n - 1]
    for i in idx:
        sig[i] = 3.0
        if i > 0:
            sig[i - 1] = 1.0
        if i + 1 < n:
            sig[i + 1] = 1.0
    return sig, idx


def test_detects_all_r_peaks_at_known_positions():
    sig, truth = _synth_ecg()
    peaks = detect_r_peaks(sig, fs=360.0)
    assert len(peaks) == len(truth)
    assert np.all(np.abs(peaks - truth) <= 2)


def test_returns_empty_array_for_flat_signal():
    peaks = detect_r_peaks(np.zeros(1000), fs=360.0)
    assert len(peaks) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ecg_delineate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecg.delineate'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Locate ECG fiducial points (P, QRS, T boundaries) on a 1-D signal."""
import numpy as np
from scipy.signal import find_peaks


def _normalise(signal):
    sig = np.asarray(signal, dtype=float).flatten()
    sig = sig - np.mean(sig)
    std = float(np.std(sig))
    return sig / std if std > 1e-9 else sig


def detect_r_peaks(signal, fs=360.0):
    """Return sample indices of R-peaks.

    Refractory distance of 250 ms reflects the shortest physiologically
    plausible RR interval; prominence rejects T waves and baseline wander.
    """
    sig = _normalise(signal)
    if np.max(np.abs(sig)) < 1e-9:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(sig, distance=int(0.25 * fs), prominence=0.5)
    return peaks.astype(int)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_delineate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add ecg/delineate.py tests/test_ecg_delineate.py
git commit -m "feat: add R-peak detection with refractory and prominence gating"
```

---

### Task 3: Heart rate and rhythm regularity

**Files:**
- Create: `ecg/parameters.py`
- Test: `tests/test_ecg_parameters.py`

**Interfaces:**
- Consumes: `detect_r_peaks` (Task 2), `Measurement`/`OK`/`UNAVAILABLE` (Task 1)
- Produces: `heart_rate(peaks: np.ndarray, fs: float) -> Measurement` (bpm); `rhythm(peaks: np.ndarray, fs: float) -> Measurement` where `.value` is the RR coefficient of variation and `.reason` is `"Regular"` or `"Irregular"`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from ecg.parameters import heart_rate, rhythm
from ecg.quality import OK, UNAVAILABLE


def test_heart_rate_from_evenly_spaced_peaks():
    peaks = np.arange(0, 3600, 288)          # 0.8 s spacing at 360 Hz -> 75 bpm
    m = heart_rate(peaks, fs=360.0)
    assert m.quality == OK
    assert abs(m.value - 75.0) < 0.5


def test_heart_rate_unavailable_with_too_few_peaks():
    m = heart_rate(np.array([10, 300]), fs=360.0)
    assert m.quality == UNAVAILABLE
    assert m.value is None


def test_regular_rhythm_for_constant_rr():
    peaks = np.arange(0, 3600, 288)
    m = rhythm(peaks, fs=360.0)
    assert m.reason == "Regular"
    assert m.value < 0.10


def test_irregular_rhythm_for_varying_rr():
    peaks = np.array([0, 200, 620, 780, 1300, 1450, 2000, 2100])
    m = rhythm(peaks, fs=360.0)
    assert m.reason == "Irregular"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ecg_parameters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecg.parameters'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Turn fiducial points into the parameters shown on the ECG page."""
import numpy as np

from ecg.quality import Measurement, OK, LOW, UNAVAILABLE

MIN_PEAKS_FOR_RATE = 3
MIN_PEAKS_FOR_RHYTHM = 5
REGULAR_CV_THRESHOLD = 0.10


def _rr_seconds(peaks, fs):
    return np.diff(np.asarray(peaks, dtype=float)) / fs


def heart_rate(peaks, fs=360.0):
    if len(peaks) < MIN_PEAKS_FOR_RATE:
        return Measurement(None, UNAVAILABLE, "Fewer than 3 R-peaks detected")
    rr = _rr_seconds(peaks, fs)
    mean_rr = float(np.mean(rr))
    if mean_rr <= 0:
        return Measurement(None, UNAVAILABLE, "Invalid RR interval")
    return Measurement(60.0 / mean_rr, OK)


def rhythm(peaks, fs=360.0):
    if len(peaks) < MIN_PEAKS_FOR_RHYTHM:
        return Measurement(None, UNAVAILABLE, "Fewer than 5 beats")
    rr = _rr_seconds(peaks, fs)
    mean_rr = float(np.mean(rr))
    if mean_rr <= 0:
        return Measurement(None, UNAVAILABLE, "Invalid RR interval")
    cv = float(np.std(rr) / mean_rr)
    label = "Regular" if cv < REGULAR_CV_THRESHOLD else "Irregular"
    return Measurement(cv, OK, label)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_parameters.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add ecg/parameters.py tests/test_ecg_parameters.py
git commit -m "feat: measure heart rate and rhythm regularity from R-peaks"
```

---

### Task 4: QRS onset and offset

**Files:**
- Modify: `ecg/delineate.py`
- Test: `tests/test_ecg_delineate.py`

**Interfaces:**
- Consumes: `detect_r_peaks` (Task 2)
- Produces: `qrs_bounds(signal: np.ndarray, peak: int, fs: float) -> tuple[int, int]` returning `(onset_idx, offset_idx)` sample indices around one R-peak.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ecg_delineate.py`:

```python
from ecg.delineate import qrs_bounds


def test_qrs_bounds_bracket_the_r_peak():
    fs = 360.0
    sig = np.zeros(720)
    # A dense triangular QRS ~80 ms wide centred at index 360. Every sample in
    # the complex is filled: a real trace is contiguous, and a boundary walk
    # must not be able to halt on a gap between spikes.
    half = int(0.04 * fs)                      # 40 ms each side
    for off in range(-half, half + 1):
        sig[360 + off] = 3.0 * (1.0 - abs(off) / (half + 1.0))
    onset, offset = qrs_bounds(sig, peak=360, fs=fs)
    assert onset < 360 < offset
    width_s = (offset - onset) / fs
    assert 0.03 <= width_s <= 0.20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ecg_delineate.py::test_qrs_bounds_bracket_the_r_peak -v`
Expected: FAIL with `ImportError: cannot import name 'qrs_bounds'`

- [ ] **Step 3: Write minimal implementation**

Append to `ecg/delineate.py`:

```python
QRS_SEARCH_S = 0.12          # widest half-window we will walk from R


def qrs_bounds(signal, peak, fs=360.0):
    """Walk outward from an R-peak until the trace returns to baseline.

    The threshold is a fraction of the local peak height, so it adapts to
    beats of differing amplitude instead of using one global cut-off.
    """
    sig = _normalise(signal)
    span = int(QRS_SEARCH_S * fs)
    lo = max(0, peak - span)
    hi = min(len(sig) - 1, peak + span)
    thresh = 0.15 * abs(sig[peak])

    onset = peak
    while onset > lo and abs(sig[onset]) > thresh:
        onset -= 1
    offset = peak
    while offset < hi and abs(sig[offset]) > thresh:
        offset += 1
    return int(onset), int(offset)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_delineate.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add ecg/delineate.py tests/test_ecg_delineate.py
git commit -m "feat: locate QRS onset and offset by adaptive baseline walk"
```

---

### Task 5: P wave detection and a real PR interval

Replaces the hardcoded `"145.0 ms"` at `app.py:560`.

**Files:**
- Modify: `ecg/delineate.py`
- Modify: `ecg/parameters.py`
- Test: `tests/test_ecg_parameters.py`

**Interfaces:**
- Consumes: `qrs_bounds` (Task 4)
- Produces: `p_onset(signal, qrs_onset, fs) -> int | None`; `pr_interval(signal, peaks, fs) -> Measurement` (seconds).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ecg_parameters.py`:

```python
from ecg.parameters import pr_interval
from ecg.quality import UNAVAILABLE


def _beat_with_p_wave(fs=360.0, n_beats=6, rr=0.8):
    n = int(n_beats * rr * fs)
    sig = np.zeros(n)
    for b in range(n_beats):
        r = int(b * rr * fs) + 100
        if r + 20 >= n:
            break
        # Dense QRS ~70 ms wide: a real complex occupies every sample it spans,
        # and a width measured from isolated spikes is not physiological.
        half = int(0.035 * fs)
        for off in range(-half, half + 1):
            if 0 <= r + off < n:
                sig[r + off] = 3.0 * (1.0 - abs(off) / (half + 1.0))
        p = r - int(0.16 * fs)             # P wave 160 ms before R
        pw = int(0.02 * fs)
        if p - pw > 0:
            for off in range(-pw, pw + 1):
                sig[p + off] = 0.45 * (1.0 - abs(off) / (pw + 1.0))
    return sig


def test_pr_interval_measured_near_expected_value():
    sig = _beat_with_p_wave()
    m = pr_interval(sig, None, fs=360.0)
    assert m.value is not None
    assert 0.10 <= m.value <= 0.22


def test_pr_unavailable_when_no_p_wave():
    fs = 360.0
    sig = np.zeros(int(6 * 0.8 * fs))
    for b in range(6):
        r = int(b * 0.8 * fs) + 100
        if r + 2 < len(sig):
            sig[r] = 3.0
            sig[r - 1] = sig[r + 1] = 1.0
    m = pr_interval(sig, None, fs=fs)
    assert m.value is None
    assert m.quality == UNAVAILABLE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ecg_parameters.py -v`
Expected: FAIL with `ImportError: cannot import name 'pr_interval'`

- [ ] **Step 3: Write minimal implementation**

Append to `ecg/delineate.py`:

```python
P_SEARCH_EARLY_S = 0.30      # look this far back from QRS onset
P_SEARCH_LATE_S = 0.08       # ...but not closer than this
P_MIN_PROMINENCE = 0.08      # relative to normalised signal


def p_onset(signal, qrs_onset, fs=360.0):
    """Find the P-wave start before a QRS, or None when no P wave is present.

    Atrial fibrillation genuinely has no P wave, so None is a real clinical
    answer here rather than a detection failure.
    """
    sig = _normalise(signal)
    lo = max(0, qrs_onset - int(P_SEARCH_EARLY_S * fs))
    hi = max(lo + 1, qrs_onset - int(P_SEARCH_LATE_S * fs))
    window = sig[lo:hi]
    if len(window) < 3:
        return None
    peaks, props = find_peaks(window, prominence=P_MIN_PROMINENCE)
    if len(peaks) == 0:
        return None
    best = peaks[int(np.argmax(props["prominences"]))]
    # Walk back to where the P wave leaves baseline.
    idx = best
    thresh = 0.3 * window[best]
    while idx > 0 and window[idx] > thresh:
        idx -= 1
    return int(lo + idx)
```

Append to `ecg/parameters.py`:

```python
from ecg.delineate import detect_r_peaks, qrs_bounds, p_onset

PR_PLAUSIBLE_S = (0.08, 0.30)


def pr_interval(signal, peaks=None, fs=360.0):
    """Median P-onset to QRS-onset across beats."""
    if peaks is None:
        peaks = detect_r_peaks(signal, fs)
    if len(peaks) == 0:
        return Measurement(None, UNAVAILABLE, "No R-peaks detected")

    values = []
    for peak in peaks:
        onset, _ = qrs_bounds(signal, int(peak), fs)
        p = p_onset(signal, onset, fs)
        if p is None:
            continue
        pr = (onset - p) / fs
        if PR_PLAUSIBLE_S[0] <= pr <= PR_PLAUSIBLE_S[1]:
            values.append(pr)

    if not values:
        return Measurement(None, UNAVAILABLE, "P wave not detectable")
    quality = OK if len(values) >= max(1, len(peaks) // 2) else LOW
    return Measurement(float(np.median(values)), quality)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_parameters.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add ecg/delineate.py ecg/parameters.py tests/test_ecg_parameters.py
git commit -m "feat: measure PR interval from detected P waves

Replaces a hardcoded 145.0 ms that was reported for every patient."
```

---

### Task 6: QRS duration, T-wave end, QT and Fridericia QTc

Replaces the hardcoded `qt = 0.40` at `app.py:610`.

**Files:**
- Modify: `ecg/delineate.py`
- Modify: `ecg/parameters.py`
- Test: `tests/test_ecg_parameters.py`

**Interfaces:**
- Consumes: `qrs_bounds` (Task 4)
- Produces: `t_end(signal, qrs_offset, rr_s, fs) -> int | None`; `qrs_duration(signal, peaks, fs) -> Measurement`; `qt_interval(signal, peaks, fs) -> Measurement`; `qtc_fridericia(qt_s: float, rr_s: float) -> float`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ecg_parameters.py`:

```python
from ecg.parameters import qrs_duration, qt_interval, qtc_fridericia


def test_qtc_fridericia_matches_formula():
    assert abs(qtc_fridericia(0.40, 1.0) - 0.40) < 1e-9
    assert abs(qtc_fridericia(0.40, 0.512) - 0.50) < 1e-3


def test_qtc_fridericia_is_tamer_than_bazett_at_high_rate():
    """At 195 bpm Bazett produced 722 ms; Fridericia must stay far below."""
    rr = 0.307
    assert qtc_fridericia(0.40, rr) * 1000 < 620


def test_qrs_duration_in_physiological_range():
    sig = _beat_with_p_wave()
    m = qrs_duration(sig, None, fs=360.0)
    assert m.value is not None
    assert 0.02 <= m.value <= 0.20


def test_qt_returns_measurement_not_constant():
    sig = _beat_with_p_wave()
    m = qt_interval(sig, None, fs=360.0)
    assert m.value is None or m.value != 0.40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ecg_parameters.py -v`
Expected: FAIL with `ImportError: cannot import name 'qrs_duration'`

- [ ] **Step 3: Write minimal implementation**

Append to `ecg/delineate.py`:

```python
def t_end(signal, qrs_offset, rr_s, fs=360.0):
    """End of the T wave by the tangent method.

    The search window scales with RR because the T wave moves closer to the
    QRS as heart rate rises.
    """
    sig = _normalise(signal)
    start = qrs_offset + int(0.04 * fs)
    stop = min(len(sig) - 1, qrs_offset + int(min(0.60, 0.6 * rr_s) * fs))
    if stop - start < 5:
        return None
    window = sig[start:stop]
    apex = int(np.argmax(np.abs(window)))
    if apex >= len(window) - 2:
        return None
    # Steepest descent after the apex, extrapolated to baseline.
    slopes = np.diff(window[apex:])
    if len(slopes) == 0:
        return None
    steep = int(np.argmin(slopes))
    slope = slopes[steep]
    if slope >= -1e-6:
        return None
    idx = apex + steep
    offset = int(window[idx] / (-slope))
    return int(start + min(len(window) - 1, idx + max(0, offset)))
```

Append to `ecg/parameters.py`:

```python
from ecg.delineate import t_end

QRS_PLAUSIBLE_S = (0.03, 0.20)
QT_PLAUSIBLE_S = (0.20, 0.65)


def qrs_duration(signal, peaks=None, fs=360.0):
    if peaks is None:
        peaks = detect_r_peaks(signal, fs)
    values = []
    for peak in peaks:
        onset, offset = qrs_bounds(signal, int(peak), fs)
        width = (offset - onset) / fs
        if QRS_PLAUSIBLE_S[0] <= width <= QRS_PLAUSIBLE_S[1]:
            values.append(width)
    if not values:
        return Measurement(None, UNAVAILABLE, "QRS boundaries not resolvable")
    return Measurement(float(np.median(values)), OK)


def qtc_fridericia(qt_s, rr_s):
    """QT / cube-root(RR). Stable at rates where Bazett over-corrects."""
    if rr_s <= 0:
        raise ValueError("rr_s must be positive")
    return qt_s / (rr_s ** (1.0 / 3.0))


def qt_interval(signal, peaks=None, fs=360.0):
    if peaks is None:
        peaks = detect_r_peaks(signal, fs)
    if len(peaks) < 2:
        return Measurement(None, UNAVAILABLE, "Fewer than 2 R-peaks detected")
    rr = _rr_seconds(peaks, fs)
    mean_rr = float(np.mean(rr))
    values = []
    for peak in peaks:
        onset, offset = qrs_bounds(signal, int(peak), fs)
        end = t_end(signal, offset, mean_rr, fs)
        if end is None:
            continue
        qt = (end - onset) / fs
        if QT_PLAUSIBLE_S[0] <= qt <= QT_PLAUSIBLE_S[1]:
            values.append(qt)
    if not values:
        return Measurement(None, UNAVAILABLE, "T wave end not resolvable")
    quality = OK if len(values) >= max(1, len(peaks) // 2) else LOW
    return Measurement(float(np.median(values)), quality)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_parameters.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add ecg/delineate.py ecg/parameters.py tests/test_ecg_parameters.py
git commit -m "feat: measure QRS duration and QT, correct QTc with Fridericia

Replaces a hardcoded qt = 0.40 that made every QTc a formula artifact."
```

---

### Task 7: ST-segment deviation

**Files:**
- Modify: `ecg/parameters.py`
- Test: `tests/test_ecg_parameters.py`

**Interfaces:**
- Consumes: `qrs_bounds` (Task 4), `p_onset` (Task 5)
- Produces: `st_deviation(signal, peaks, fs, mm_per_mv=10.0) -> Measurement` where `.value` is deviation in **mm** and `.reason` is `"Normal"`, `"Elevated"` or `"Depressed"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ecg_parameters.py`:

```python
from ecg.parameters import st_deviation


def test_st_normal_for_flat_baseline():
    sig = _beat_with_p_wave()
    m = st_deviation(sig, None, fs=360.0)
    assert m.reason in {"Normal", "Elevated", "Depressed"}


def test_st_elevation_detected_when_segment_raised():
    fs = 360.0
    sig = _beat_with_p_wave(fs=fs)
    peaks = detect_r_peaks(sig, fs)
    for peak in peaks:
        j = int(peak) + int(0.04 * fs)
        sig[j:j + int(0.10 * fs)] += 0.9      # lift the ST segment
    m = st_deviation(sig, peaks, fs=fs)
    assert m.reason == "Elevated"


def test_st_unavailable_without_peaks():
    m = st_deviation(np.zeros(1000), np.array([]), fs=360.0)
    assert m.value is None
```

Add the missing import at the top of the test file:

```python
from ecg.delineate import detect_r_peaks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ecg_parameters.py -v`
Expected: FAIL with `ImportError: cannot import name 'st_deviation'`

- [ ] **Step 3: Write minimal implementation**

Append to `ecg/parameters.py`:

```python
ST_OFFSET_S = 0.06           # J+60 ms, the conventional measurement point
ST_THRESHOLD_MM = 1.0


def st_deviation(signal, peaks=None, fs=360.0, mm_per_mv=10.0):
    """Deviation at J+60 ms relative to the PR-segment isoelectric baseline."""
    sig = np.asarray(signal, dtype=float).flatten()
    if peaks is None:
        peaks = detect_r_peaks(sig, fs)
    if len(peaks) == 0:
        return Measurement(None, UNAVAILABLE, "No R-peaks detected")

    deviations = []
    for peak in peaks:
        onset, offset = qrs_bounds(sig, int(peak), fs)
        p = p_onset(sig, onset, fs)
        # PR segment is the flat stretch between P end and QRS onset; fall
        # back to just before QRS onset when no P wave is present.
        base_lo = p if p is not None else max(0, onset - int(0.04 * fs))
        baseline = float(np.median(sig[base_lo:onset])) if onset > base_lo else 0.0
        j = offset + int(ST_OFFSET_S * fs)
        if j >= len(sig):
            continue
        # mm = mV x (mm per mV). Standard ECG gain is 10 mm/mV.
        deviations.append((sig[j] - baseline) * mm_per_mv)

    if not deviations:
        return Measurement(None, UNAVAILABLE, "ST point beyond signal end")

    dev_mm = float(np.median(deviations))
    if dev_mm > ST_THRESHOLD_MM:
        label = "Elevated"
    elif dev_mm < -ST_THRESHOLD_MM:
        label = "Depressed"
    else:
        label = "Normal"
    return Measurement(dev_mm, OK, label)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_parameters.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add ecg/parameters.py tests/test_ecg_parameters.py
git commit -m "feat: measure ST deviation at J+60ms against PR-segment baseline"
```

---

### Task 8: Grid calibration from the ECG paper

**Files:**
- Create: `ecg/digitize.py`
- Test: `tests/test_ecg_digitize.py`

**Interfaces:**
- Consumes: nothing
- Produces: `detect_grid_scale(gray: np.ndarray) -> float | None` returning **pixels per millimetre**, or None when the grid is not detectable.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from ecg.digitize import detect_grid_scale


def _grid_image(px_per_mm=5, w=600, h=400):
    """White page with dark rulings every px_per_mm pixels."""
    img = np.full((h, w), 255.0)
    img[:, ::px_per_mm] = 120.0
    img[::px_per_mm, :] = 120.0
    return img


def test_detects_known_grid_spacing():
    scale = detect_grid_scale(_grid_image(px_per_mm=5))
    assert scale is not None
    assert abs(scale - 5.0) < 0.6


def test_returns_none_for_blank_page():
    assert detect_grid_scale(np.full((400, 600), 255.0)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ecg_digitize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecg.digitize'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_digitize.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add ecg/digitize.py tests/test_ecg_digitize.py
git commit -m "feat: derive pixels-per-mm from ECG grid via FFT of ink profile"
```

---

### Task 9: Layout detection and per-lead extraction

**Files:**
- Modify: `ecg/digitize.py`
- Test: `tests/test_ecg_digitize.py`

**Interfaces:**
- Consumes: `detect_grid_scale` (Task 8)
- Produces: `detect_layout(gray) -> str` returning `"single"` or `"twelve_lead"`; `extract_leads(image_path) -> dict` with keys `leads` (`dict[str, np.ndarray]`), `layout` (str), `px_per_mm` (float | None). Lead names for 12-lead are `["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]` in 3-row x 4-column reading order; single-lead uses key `"II"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ecg_digitize.py`:

```python
from ecg.digitize import detect_layout, extract_leads, LEAD_NAMES_12
from PIL import Image
import os


def test_single_strip_detected_as_single():
    img = np.full((120, 1200), 255.0)
    img[60, :] = 0.0
    assert detect_layout(img) == "single"


def test_wide_grid_of_panels_detected_as_twelve_lead():
    """Three rows of traces across a roughly page-shaped image."""
    img = np.full((900, 1200), 255.0)
    for row in range(3):
        y = 150 + row * 300
        img[y, :] = 0.0
    assert detect_layout(img) == "twelve_lead"


def test_panel_positions_map_to_correct_lead_names():
    """Pin the physical sheet layout: a set-equality check cannot catch a
    transposition, and Task 10 reads aVF from a specific panel."""
    from ecg.digitize import LEAD_PANELS_3X4
    assert LEAD_PANELS_3X4[0] == ["I", "aVR", "V1", "V4"]
    assert LEAD_PANELS_3X4[1] == ["II", "aVL", "V2", "V5"]
    assert LEAD_PANELS_3X4[2] == ["III", "aVF", "V3", "V6"]
    # aVF drives the axis calculation; assert its position explicitly.
    assert LEAD_PANELS_3X4[2][1] == "aVF"
    assert LEAD_PANELS_3X4[0][0] == "I"


def test_blank_panel_yields_no_signal():
    """A traceless panel must report nothing, not a fabricated flat line."""
    from ecg.digitize import _trace_row_band
    assert _trace_row_band(np.zeros((100, 300))) is None


def test_extract_leads_returns_named_signals(tmp_path):
    img = np.full((900, 1200), 255) .astype(np.uint8)
    for row in range(3):
        img[150 + row * 300, :] = 0
    path = os.path.join(tmp_path, "ecg.png")
    Image.fromarray(img).save(path)
    out = extract_leads(path)
    assert out["layout"] in {"single", "twelve_lead"}
    if out["layout"] == "twelve_lead":
        assert set(out["leads"]) == set(LEAD_NAMES_12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ecg_digitize.py -v`
Expected: FAIL with `ImportError: cannot import name 'detect_layout'`

- [ ] **Step 3: Write minimal implementation**

Append to `ecg/digitize.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_digitize.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add ecg/digitize.py tests/test_ecg_digitize.py
git commit -m "feat: detect strip vs 12-lead layout and extract per-lead signals"
```

---

### Task 10: QRS axis from leads I and aVF

**Files:**
- Modify: `ecg/parameters.py`
- Test: `tests/test_ecg_parameters.py`

**Interfaces:**
- Consumes: `extract_leads` output (Task 9), `qrs_bounds` (Task 4)
- Produces: `qrs_axis(leads: dict, fs: float) -> Measurement` where `.value` is degrees and `.reason` is `"Normal axis"`, `"Left axis deviation"`, `"Right axis deviation"`, `"Extreme axis"` or `"Requires 12-lead"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ecg_parameters.py`:

```python
from ecg.parameters import qrs_axis
from ecg.quality import UNAVAILABLE


def test_axis_unavailable_for_single_lead():
    m = qrs_axis({"II": _beat_with_p_wave()}, fs=360.0)
    assert m.value is None
    assert m.quality == UNAVAILABLE
    assert m.reason == "Requires 12-lead"


def test_axis_near_zero_when_lead_I_positive_and_aVF_flat():
    sig = _beat_with_p_wave()
    flat = np.zeros_like(sig)
    m = qrs_axis({"I": sig, "aVF": flat}, fs=360.0)
    assert m.value is not None
    assert abs(m.value) < 20.0


def test_axis_near_ninety_when_aVF_dominant():
    sig = _beat_with_p_wave()
    flat = np.zeros_like(sig)
    m = qrs_axis({"I": flat, "aVF": sig}, fs=360.0)
    assert m.value is not None
    assert abs(m.value - 90.0) < 20.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ecg_parameters.py -v`
Expected: FAIL with `ImportError: cannot import name 'qrs_axis'`

- [ ] **Step 3: Write minimal implementation**

Append to `ecg/parameters.py`:

```python
def _net_qrs_area(signal, fs):
    """Signed area under the QRS complexes: the lead's net deflection."""
    sig = np.asarray(signal, dtype=float).flatten()
    peaks = detect_r_peaks(sig, fs)
    if len(peaks) == 0:
        return 0.0
    total = 0.0
    for peak in peaks:
        onset, offset = qrs_bounds(sig, int(peak), fs)
        total += float(np.sum(sig[onset:offset + 1]))
    return total / len(peaks)


def qrs_axis(leads, fs=360.0):
    """Frontal-plane QRS axis from the net deflections of leads I and aVF.

    One lead is a single projection of the electrical vector, so a 2-D angle
    cannot be recovered from it. Single-lead input therefore reports
    "Requires 12-lead" rather than a guess.
    """
    if not isinstance(leads, dict) or "I" not in leads or "aVF" not in leads:
        return Measurement(None, UNAVAILABLE, "Requires 12-lead")
    # A blank panel digitizes to None (see ecg.digitize._trace_row_band), so a
    # sheet can carry the lead names without carrying usable traces.
    if leads["I"] is None or leads["aVF"] is None:
        return Measurement(None, UNAVAILABLE, "Lead I or aVF has no trace")

    net_i = _net_qrs_area(leads["I"], fs)
    net_avf = _net_qrs_area(leads["aVF"], fs)
    if abs(net_i) < 1e-9 and abs(net_avf) < 1e-9:
        return Measurement(None, UNAVAILABLE, "No measurable QRS deflection")

    degrees = float(np.degrees(np.arctan2(net_avf, net_i)))
    if -30.0 <= degrees <= 90.0:
        label = "Normal axis"
    elif -90.0 <= degrees < -30.0:
        label = "Left axis deviation"
    elif 90.0 < degrees <= 180.0:
        label = "Right axis deviation"
    else:
        label = "Extreme axis"
    return Measurement(degrees, OK, label)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_parameters.py -v`
Expected: PASS (16 passed)

- [ ] **Step 5: Commit**

```bash
git add ecg/parameters.py tests/test_ecg_parameters.py
git commit -m "feat: compute frontal-plane QRS axis from leads I and aVF"
```

---

### Task 11: Assemble the full parameter report

**Files:**
- Modify: `ecg/parameters.py`
- Test: `tests/test_ecg_parameters.py`

**Interfaces:**
- Consumes: every measurement function above
- Produces: `analyse(signal_or_leads, fs=360.0, px_per_mm=None, from_image=False) -> dict` mapping each of `heart_rate`, `rhythm`, `pr_interval`, `qrs_duration`, `qt_interval`, `qtc`, `st_segment`, `axis`, `rr_interval`, `sdnn`, `rmssd` to a `Measurement`, plus key `display` holding preformatted strings for the template.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ecg_parameters.py`:

```python
from ecg.parameters import analyse


def test_analyse_returns_every_expected_parameter():
    report = analyse(_beat_with_p_wave(), fs=360.0)
    for key in ["heart_rate", "rhythm", "pr_interval", "qrs_duration",
                "qt_interval", "qtc", "st_segment", "axis"]:
        assert key in report, key


def test_analyse_never_emits_the_old_hardcoded_values():
    """Guards the two fabricated constants this work removes."""
    report = analyse(_beat_with_p_wave(), fs=360.0)
    pr = report["pr_interval"]
    qt = report["qt_interval"]
    assert not (pr.value is not None and abs(pr.value - 0.145) < 1e-9)
    assert not (qt.value is not None and abs(qt.value - 0.400) < 1e-9)


def test_analyse_marks_axis_unavailable_for_single_lead():
    report = analyse(_beat_with_p_wave(), fs=360.0)
    assert report["axis"].reason == "Requires 12-lead"


def test_analyse_accepts_a_dict_of_leads():
    sig = _beat_with_p_wave()
    report = analyse({"I": sig, "aVF": np.zeros_like(sig), "II": sig}, fs=360.0)
    assert report["heart_rate"].value is not None


def test_analyse_skips_blank_leads_when_choosing_primary():
    """A None-valued lead II must not crash or become the primary signal."""
    sig = _beat_with_p_wave()
    report = analyse({"II": None, "I": sig}, fs=360.0)
    assert report["heart_rate"].value is not None


def test_analyse_reports_all_unavailable_when_every_lead_is_blank():
    report = analyse({"I": None, "II": None}, fs=360.0)
    assert report["heart_rate"].value is None
    assert report["display"]["heart_rate"] == "—"


def test_st_unavailable_from_image_without_grid_scale():
    """ST is in millimetres, so no paper scale means no honest ST value."""
    report = analyse(_beat_with_p_wave(), fs=360.0,
                     px_per_mm=None, from_image=True)
    assert report["st_segment"].value is None
    assert report["st_segment"].reason == "ECG grid not detected"


def test_hrv_measurements_are_stored_in_seconds():
    """Every Measurement in the report uses seconds; only display converts."""
    report = analyse(_beat_with_p_wave(), fs=360.0)
    if report["sdnn"].value is not None:
        # ~0.0 s for a metronomic synthetic signal, but certainly sub-second.
        assert report["sdnn"].value < 1.0
        assert report["display"]["sdnn"].endswith("ms")


def test_display_strings_use_dash_for_unavailable():
    report = analyse(np.zeros(500), fs=360.0)
    assert report["display"]["heart_rate"] == "—"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ecg_parameters.py -v`
Expected: FAIL with `ImportError: cannot import name 'analyse'`

- [ ] **Step 3: Write minimal implementation**

Append to `ecg/parameters.py`:

```python
def _all_unavailable(reason):
    """A complete report in which nothing could be measured.

    Returned rather than raising, so a blank or unreadable image degrades to
    honest dashes instead of a 500.
    """
    keys = ["heart_rate", "rhythm", "pr_interval", "qrs_duration",
            "qt_interval", "qtc", "st_segment", "axis", "rr_interval",
            "sdnn", "rmssd"]
    report = {k: Measurement(None, UNAVAILABLE, reason) for k in keys}
    report["display"] = {k: "—" for k in keys}
    return report


def analyse(signal_or_leads, fs=360.0, px_per_mm=None, from_image=False):
    """Measure every displayed parameter, flagging what could not be measured."""
    if isinstance(signal_or_leads, dict):
        leads = signal_or_leads
        # A blank panel digitizes to None, and dict.get() only substitutes its
        # default when the KEY is missing - not when the value is None. Pick the
        # first lead that actually carries a trace, preferring II.
        primary = leads.get("II")
        if primary is None:
            primary = next((v for v in leads.values() if v is not None), None)
        if primary is None:
            return _all_unavailable("No lead carried a usable trace")
    else:
        leads = {"II": np.asarray(signal_or_leads, dtype=float).flatten()}
        primary = leads["II"]

    peaks = detect_r_peaks(primary, fs)
    hr = heart_rate(peaks, fs)
    rhy = rhythm(peaks, fs)
    pr = pr_interval(primary, peaks, fs)
    qrs = qrs_duration(primary, peaks, fs)
    qt = qt_interval(primary, peaks, fs)
    axis = qrs_axis(leads, fs)

    # ST is the one parameter expressed in millimetres, so it needs the paper
    # scale. from_image=True with no detected grid means we cannot honestly
    # convert amplitude to mm.
    if from_image and px_per_mm is None:
        st = Measurement(None, UNAVAILABLE, "ECG grid not detected")
    else:
        st = st_deviation(primary, peaks, fs)

    if qt.value is not None and len(peaks) >= 2:
        rr_s = _rr_seconds(peaks, fs)
        mean_rr = float(np.mean(rr_s))
        qtc = Measurement(qtc_fridericia(qt.value, mean_rr), qt.quality)
        rr = Measurement(mean_rr, OK)
        # Stored in seconds like every other duration here; the display layer
        # is the only place that converts to milliseconds.
        sdnn = Measurement(float(np.std(rr_s, ddof=1)), OK) \
            if len(rr_s) >= 2 else Measurement(None, UNAVAILABLE, "Too few beats")
        rmssd = Measurement(float(np.sqrt(np.mean(np.diff(rr_s) ** 2))), OK) \
            if len(rr_s) >= 3 else Measurement(None, UNAVAILABLE, "Too few beats")
    else:
        qtc = Measurement(None, UNAVAILABLE, "QT not measurable")
        rr = Measurement(None, UNAVAILABLE, "Fewer than 2 R-peaks detected")
        sdnn = Measurement(None, UNAVAILABLE, "Too few beats")
        rmssd = Measurement(None, UNAVAILABLE, "Too few beats")

    report = {
        "heart_rate": hr, "rhythm": rhy, "pr_interval": pr,
        "qrs_duration": qrs, "qt_interval": qt, "qtc": qtc,
        "st_segment": st, "axis": axis, "rr_interval": rr,
        "sdnn": sdnn, "rmssd": rmssd,
    }
    report["display"] = {
        "heart_rate": hr.format("bpm"),
        "rhythm": rhy.reason if rhy.value is not None else "—",
        "pr_interval": Measurement(
            pr.value * 1000 if pr.value is not None else None, pr.quality).format("ms", 0),
        "qrs_duration": Measurement(
            qrs.value * 1000 if qrs.value is not None else None, qrs.quality).format("ms", 0),
        "qt_interval": Measurement(
            qt.value * 1000 if qt.value is not None else None, qt.quality).format("ms", 0),
        "qtc": Measurement(
            qtc.value * 1000 if qtc.value is not None else None, qtc.quality).format("ms", 0)
            + (" (Fridericia)" if qtc.value is not None else ""),
        "st_segment": (f"{st.reason} ({st.value:+.1f} mm)"
                       if st.value is not None else "—"),
        "axis": (f"{axis.value:+.0f}° ({axis.reason})"
                 if axis.value is not None else "—"),
        "rr_interval": rr.format("s", 3),
        "sdnn": Measurement(
            sdnn.value * 1000 if sdnn.value is not None else None,
            sdnn.quality).format("ms"),
        "rmssd": Measurement(
            rmssd.value * 1000 if rmssd.value is not None else None,
            rmssd.quality).format("ms"),
    }
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_parameters.py -v`
Expected: PASS (20 passed)

- [ ] **Step 5: Commit**

```bash
git add ecg/parameters.py tests/test_ecg_parameters.py
git commit -m "feat: assemble full ECG parameter report with display strings"
```

---

### Task 12: Round-trip digitization test

**Files:**
- Modify: `tests/test_ecg_digitize.py`

**Interfaces:**
- Consumes: `extract_leads` (Task 9)
- Produces: nothing — this task only adds the correctness gate the spec requires.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ecg_digitize.py`:

```python
def test_render_then_digitize_preserves_signal_shape(tmp_path):
    """Spec gate: round-trip correlation must exceed 0.95."""
    fs, n = 360.0, 1440
    t = np.arange(n) / fs
    truth = np.sin(2 * np.pi * 1.2 * t)

    h, w = 300, n
    img = np.full((h, w), 255.0)
    mid, amp = h // 2, h // 3
    for x in range(w):
        y = int(mid - truth[x] * amp)
        img[max(0, y - 1):min(h, y + 2), x] = 0.0

    path = os.path.join(tmp_path, "sine.png")
    Image.fromarray(img.astype(np.uint8)).save(path)

    got = extract_leads(path)["leads"]["II"]
    m = min(len(got), len(truth))
    r = np.corrcoef(got[:m], truth[:m])[0, 1]
    assert r > 0.95, f"round-trip correlation {r:.3f} below 0.95"
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_digitize.py::test_render_then_digitize_preserves_signal_shape -v`
Expected: This validates Task 9's tracer. If it FAILS, the tracer is wrong — fix `_trace_row_band` in `ecg/digitize.py` until it passes. Do not weaken the 0.95 threshold.

- [ ] **Step 3: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests pass, including the pre-existing ones.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ecg_digitize.py
git commit -m "test: assert digitization round-trip correlation above 0.95"
```

---

### Task 13: Wire the new parameters into the Flask app

**Files:**
- Modify: `app.py:554-617` (delete `calculate_ecg_parameters`), `app.py:1289-1385` (`analyze_ecg` route)
- Test: `tests/test_ecg_route.py`

**Interfaces:**
- Consumes: `analyse` (Task 11), `extract_leads` (Task 9)
- Produces: `result` dict keys `rhythm`, `st_segment`, `axis` added alongside existing ones; all parameter values become the preformatted strings from `report["display"]`.

- [ ] **Step 1: Write the failing test**

```python
import app as flask_app


def _post(client, path, extra):
    data = {"patient_name": "T", "patient_id": "1", "age": "40",
            "gender": "Male", "contact": "1234567890",
            "referring_physician": "D", "study_date": "2026-09-09",
            "clinical_history": "test"}
    data.update(extra)
    return client.post(path, data=data, content_type="multipart/form-data")


def test_ecg_page_reports_new_parameters():
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as client:
        resp = _post(client, "/analyze_ecg", {"sample_type": "normal"})
        html = resp.get_data(as_text=True)
        assert resp.status_code == 200
        for label in ["Rhythm", "ST Segment", "QRS Axis"]:
            assert label in html, label


def test_pr_is_no_longer_the_hardcoded_constant():
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as client:
        resp = _post(client, "/analyze_ecg", {"sample_type": "normal"})
        html = resp.get_data(as_text=True)
        assert "145.0 ms" not in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ecg_route.py -v`
Expected: FAIL — `"Rhythm"` absent and `"145.0 ms"` still present.

- [ ] **Step 3: Write the implementation**

Delete `calculate_ecg_parameters` (`app.py:554-617`) entirely. Add near the other imports in `app.py`:

```python
from ecg.parameters import analyse as analyse_ecg_parameters
from ecg.digitize import extract_leads
```

In `analyze_ecg`, replace the `is_image` branch and the `parameters = calculate_ecg_parameters(ecg_values)` call with:

```python
        if is_image:
            digitized = extract_leads(filepath)
            leads = digitized["leads"]
            ecg_values = leads.get("II", next(iter(leads.values())))
            X = prepare_qcnn_input(ecg_values)
            report = analyse_ecg_parameters(leads, fs=360.0,
                                            px_per_mm=digitized["px_per_mm"],
                                            from_image=True)
            input_source = f"ECG image ({digitized['layout'].replace('_', ' ')})"
        else:
            ecg_values = load_ecg_file(filepath)
            X = prepare_qcnn_input(ecg_values)
            report = analyse_ecg_parameters(ecg_values, fs=360.0)
            input_source = "ECG data file"
```

Then replace the parameter entries in the `result` dict with:

```python
            "heart_rate": report["display"]["heart_rate"],
            "rr_interval": report["display"]["rr_interval"],
            "qrs_duration": report["display"]["qrs_duration"],
            "pr_interval": report["display"]["pr_interval"],
            "qt_interval": report["display"]["qt_interval"],
            "qtc": report["display"]["qtc"],
            "sdnn": report["display"]["sdnn"],
            "rmssd": report["display"]["rmssd"],
            "rhythm": report["display"]["rhythm"],
            "st_segment": report["display"]["st_segment"],
            "axis": report["display"]["axis"],
            "signal_quality": "Good" if report["heart_rate"].value else "Insufficient data",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ecg_route.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_ecg_route.py
git commit -m "feat: serve measured ECG parameters from the ecg package

Removes calculate_ecg_parameters and its hardcoded PR and QT constants."
```

---

### Task 14: Show Rhythm, ST and Axis in the template and PDF

**Files:**
- Modify: `templates/ecg.html:176-225`
- Modify: `app.py:1738-1790` (`download_ecg_report`)

**Interfaces:**
- Consumes: `result` keys from Task 13
- Produces: nothing downstream.

- [ ] **Step 1: Add the three metric cards**

In `templates/ecg.html`, inside the "ECG Waveform Parameters" grid, after the QTc card:

```html
                    <div class="metric-card">
                        <div class="metric-label">Rhythm</div>
                        <div class="metric-value" style="font-size: 16px; color: {% if result.rhythm == 'Regular' %}var(--success){% elif result.rhythm == '—' %}var(--text-muted){% else %}var(--warning){% endif %};">{{ result.rhythm }}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">ST Segment</div>
                        <div class="metric-value" style="font-size: 16px; color: {% if 'Normal' in result.st_segment %}var(--success){% elif result.st_segment == '—' %}var(--text-muted){% else %}var(--danger){% endif %};">{{ result.st_segment }}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">QRS Axis</div>
                        <div class="metric-value" style="font-size: 16px; color: #a78bfa;">{{ result.axis }}</div>
                    </div>
```

- [ ] **Step 2: Carry them into the PDF**

In `app.py`, in the `download_ecg_report` parameter rows (near `app.py:1758`), after the QTc row:

```python
            ("Rhythm", _pdf_value(form.get("rhythm"))),
            ("ST Segment", _pdf_value(form.get("st_segment"))),
            ("QRS Axis", _pdf_value(form.get("axis"))),
```

Add this helper above `download_ecg_report` in `app.py`, so an unmeasurable
parameter is stated as such in the report rather than shown as a bare dash:

```python
def _pdf_value(raw):
    """Render an unmeasured parameter explicitly in the PDF.

    A dash in a clinical report is ambiguous - it could mean zero, or missing.
    "Not measurable" says which.
    """
    if raw is None or raw.strip() in {"", "\u2014", "-"}:
        return "Not measurable"
    return raw
```

And add the matching hidden inputs in `templates/ecg.html` inside the report form:

```html
                    <input type="hidden" name="rhythm" value="{{ result.rhythm }}">
                    <input type="hidden" name="st_segment" value="{{ result.st_segment }}">
                    <input type="hidden" name="axis" value="{{ result.axis }}">
```

- [ ] **Step 3: Verify end to end**

```bash
.venv/bin/python app.py > /tmp/ecg_app.log 2>&1 &
sleep 25
curl -s -X POST http://127.0.0.1:5050/analyze_ecg \
  -F patient_name=T -F patient_id=1 -F age=40 -F gender=Male \
  -F contact=1234567890 -F referring_physician=D -F study_date=2026-09-09 \
  -F clinical_history=test -F sample_type=abnormal \
  | sed 's/<[^>]*>//g' | grep -A1 -E "Rhythm|ST Segment|QRS Axis|PR Interval"
```

Expected: Rhythm, ST Segment and QRS Axis all render. PR shows a measured value or `—`, never `145.0 ms`.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add templates/ecg.html app.py
git commit -m "feat: display Rhythm, ST segment and QRS axis on ECG page and report"
```

---

### Task 15: Validate R-peak detection against real MIT-BIH annotations

The spec requires detection sensitivity to be **measured, not asserted**. Tasks
2-14 use synthetic signals only; this task closes that gap using the real
records already in `data/raw/mitdb` and the `wfdb` package (4.3.1, installed).

**Files:**
- Create: `tests/test_ecg_delineate_mitbih.py`

**Interfaces:**
- Consumes: `detect_r_peaks` (Task 2)
- Produces: nothing — this is a correctness gate.

- [ ] **Step 1: Write the failing test**

```python
"""Measure R-peak detection against MIT-BIH reference annotations.

Sensitivity is computed, not assumed. A detected peak counts as a true
positive when it lands within 150 ms of an annotated beat, the tolerance
used by the standard EC57 evaluation.
"""
import os
import numpy as np
import pytest

from ecg.delineate import detect_r_peaks

wfdb = pytest.importorskip("wfdb")

MITDB = "data/raw/mitdb"
RECORDS = ["100", "101", "103", "115", "123"]
TOLERANCE_S = 0.15
MIN_SENSITIVITY = 0.90

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MITDB), reason="MIT-BIH records not present")


def _sensitivity(record_name):
    rec = wfdb.rdrecord(os.path.join(MITDB, record_name))
    ann = wfdb.rdann(os.path.join(MITDB, record_name), "atr")
    fs = float(rec.fs)

    # First 60 seconds of the first channel keeps the test fast.
    n = int(60 * fs)
    signal = rec.p_signal[:n, 0]
    truth = np.array([s for s in ann.sample if s < n])
    if len(truth) == 0:
        pytest.skip(f"no annotations in first 60s of {record_name}")

    detected = detect_r_peaks(signal, fs=fs)
    tol = TOLERANCE_S * fs
    hits = sum(1 for t in truth
               if len(detected) and np.min(np.abs(detected - t)) <= tol)
    return hits / len(truth)


@pytest.mark.parametrize("record", RECORDS)
def test_r_peak_sensitivity_per_record(record):
    se = _sensitivity(record)
    assert se >= MIN_SENSITIVITY, f"{record}: sensitivity {se:.3f} below {MIN_SENSITIVITY}"


def test_mean_sensitivity_across_records():
    scores = [_sensitivity(r) for r in RECORDS]
    mean = float(np.mean(scores))
    assert mean >= 0.95, f"mean sensitivity {mean:.3f} below 0.95"
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/test_ecg_delineate_mitbih.py -v`
Expected: This measures the Task 2 detector on real data. If sensitivity falls
below threshold, fix `detect_r_peaks` — typically by band-pass filtering
5-15 Hz before peak-finding to suppress baseline wander and T waves:

```python
from scipy.signal import butter, filtfilt

def _bandpass(sig, fs, lo=5.0, hi=15.0):
    nyq = 0.5 * fs
    b, a = butter(2, [lo / nyq, min(0.99, hi / nyq)], btype="band")
    return filtfilt(b, a, sig)
```

Do not lower `MIN_SENSITIVITY` to make the test pass.

- [ ] **Step 3: Record the measured numbers**

Add the achieved per-record sensitivities to the docstring at the top of
`ecg/delineate.py`, so the file states its own measured performance:

```python
"""Locate ECG fiducial points (P, QRS, T boundaries) on a 1-D signal.

R-peak sensitivity measured against MIT-BIH reference annotations
(150 ms tolerance, first 60 s per record): see
tests/test_ecg_delineate_mitbih.py for the gate.
"""
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ecg_delineate_mitbih.py ecg/delineate.py
git commit -m "test: measure R-peak sensitivity against MIT-BIH annotations"
```
