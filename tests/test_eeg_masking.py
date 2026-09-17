"""Artifact rejection must not become baseline deletion.

The headline test here is test_baseline_is_kept_as_the_negative_class. If it
ever fails, the training set has one class and the model will report findings
on healthy recordings with confidence -- the exact failure the module
docstring exists to prevent.
"""
import numpy as np
import pytest

from eeg.masking import (artifact_reason, label_windows,
                         select_training_windows, class_weights,
                         MAX_CLASS_WEIGHT)

FS = 256.0
N = int(FS * 2)


def _clean(seed=0, amp=20.0):
    """Plausible background: alpha rhythm plus a little pink-ish noise."""
    t = np.arange(N) / FS
    rng = np.random.default_rng(seed)
    return amp * np.sin(2 * np.pi * 10 * t) + rng.normal(0, amp * 0.1, N)


# ------------------------------------------------------ artifact rejection

def test_clean_background_is_not_rejected():
    assert artifact_reason(_clean(), FS) is None


def test_flatline_is_rejected():
    assert "flatline" in artifact_reason(np.zeros(N), FS)


def test_emg_contamination_is_rejected():
    """Broadband high-frequency muscle activity is not brain signal."""
    t = np.arange(N) / FS
    emg = sum(np.sin(2 * np.pi * f * t) for f in range(32, 45))
    assert artifact_reason(emg, FS) == "EMG contamination"


def test_saturation_is_rejected():
    sig = np.clip(_clean() * 50, -100, 100)
    assert artifact_reason(sig, FS) == "amplifier saturation"


def test_non_finite_samples_are_rejected():
    sig = _clean()
    sig[10] = np.nan
    assert artifact_reason(sig, FS) == "non-finite samples"


def test_a_gross_excursion_is_rejected_against_the_recording_sigma():
    sig = _clean()
    sig[100] += 5000.0
    assert artifact_reason(sig, FS, reference_sigma=15.0) is not None


def test_rejection_gives_a_reason_not_just_a_boolean():
    """Causes are counted and shown; a run that drops most of a recording is
    a data problem and must be visible as one."""
    assert isinstance(artifact_reason(np.zeros(N), FS), str)


# ---------------------------------------------------------------- labelling

def _times(n, window_s=2.0, overlap=0.5):
    step = window_s * (1 - overlap)
    return [(i * step, i * step + window_s) for i in range(n)]


def test_windows_overlapping_an_event_are_positive():
    labels = label_windows(_times(6), [(3.0, 5.0)])
    assert labels.sum() > 0


def test_onset_windows_are_labelled_positive():
    """Containment would label the onset window negative -- exactly where
    onset detection has to work."""
    times = [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]
    labels = label_windows(times, [(2.9, 10.0)])
    assert labels[1] == 1, "the window catching seizure onset must be positive"


def test_windows_clear_of_any_event_are_negative():
    assert label_windows([(0.0, 2.0), (10.0, 12.0)], [(4.0, 5.0)]).tolist() == [0, 0]


def test_touching_at_a_boundary_does_not_count_as_overlap():
    assert label_windows([(0.0, 2.0)], [(2.0, 3.0)]).tolist() == [0]


# ------------------------------------------------------- training selection

def test_baseline_is_kept_as_the_negative_class():
    """The central guarantee of this module. A training set with only the
    positive class yields a model that finds anomalies everywhere."""
    windows = np.stack([_clean(seed=i) for i in range(8)])
    times = _times(8)
    kept, labels, kept_times, _ = select_training_windows(
        windows, times, FS, annotations=[(4.0, 6.0)])
    assert set(np.unique(labels)) == {0, 1}, "both classes must survive"
    assert (labels == 0).sum() > 0


def test_artifact_windows_are_dropped_with_their_labels():
    windows = np.stack([_clean(seed=0), np.zeros(N), _clean(seed=2)])
    kept, labels, kept_times, rejections = select_training_windows(
        windows, _times(3), FS, annotations=[])
    assert len(kept) == 2 == len(labels) == len(kept_times)
    assert sum(rejections.values()) == 1


def test_labels_stay_aligned_with_the_windows_that_survived():
    """A dropped window must take its label with it, or every subsequent
    label describes the wrong segment."""
    windows = np.stack([np.zeros(N), _clean(seed=1), _clean(seed=2)])
    times = _times(3)                      # (0,2), (1,3), (2,4) -- they overlap
    # Chosen to fall inside window 2 only: window 1 ends exactly at 3.0, and
    # a boundary touch is not an overlap.
    kept, labels, kept_times, _ = select_training_windows(
        windows, times, FS, annotations=[(3.0, 4.0)])
    assert kept_times == [times[1], times[2]]
    assert labels.tolist() == [0, 1]


def test_rejections_are_reported_by_cause():
    windows = np.stack([np.zeros(N), np.zeros(N), _clean()])
    _, _, _, rejections = select_training_windows(
        windows, _times(3), FS, annotations=[])
    assert "flatline or disconnected electrode" in rejections
    assert rejections["flatline or disconnected electrode"] == 2


def test_everything_rejected_returns_empty_rather_than_raising():
    windows = np.stack([np.zeros(N), np.zeros(N)])
    kept, labels, times, rejections = select_training_windows(
        windows, _times(2), FS, annotations=[])
    assert len(kept) == 0 and len(labels) == 0
    assert sum(rejections.values()) == 2


def test_a_multi_channel_window_is_judged_on_its_worst_channel():
    good, bad = _clean(seed=5), np.zeros(N)
    windows = np.stack([np.stack([good, bad])])
    _, _, _, rejections = select_training_windows(
        windows, _times(1), FS, annotations=[])
    assert sum(rejections.values()) == 1


# ------------------------------------------------------------ class weights

def test_the_rare_class_is_up_weighted():
    labels = np.array([0] * 100 + [1] * 5)
    w = class_weights(labels)
    assert w[1] > w[0]


def test_weights_are_capped():
    """Six positives must not carry the same total gradient as ten thousand
    negatives."""
    labels = np.array([0] * 10000 + [1] * 6)
    assert class_weights(labels)[1] == MAX_CLASS_WEIGHT


def test_a_balanced_set_gets_near_equal_weights():
    w = class_weights(np.array([0] * 50 + [1] * 50))
    assert w[0] == pytest.approx(w[1], abs=1e-6)
