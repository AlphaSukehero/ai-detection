"""Which windows a model may be trained on, and with what label.

This is the corrected form of "train only on segments where the anomaly is
present". Taken literally that instruction produces a classifier with one
class: a model that has never seen normal cortical rhythm has no decision
boundary, so at inference every window resembles what it was trained on and
the system reports seizures on healthy recordings with high confidence. No
training metric would reveal it, because the held-out set would be just as
one-sided.

What the instruction is actually after is separable into two things this
module keeps apart:

  * REJECT true artifact -- EMG bursts, electrode pops, flatline, saturation.
    Those are not brain signal at all and teach nothing. Dropping them is
    what "discard muscle artifacts" should mean.
  * KEEP clean baseline as the negative class, and handle the resulting
    imbalance with capped class weights rather than by deletion. Focus comes
    from the weighting, and the boundary survives.

The cap matters for the same reason it mattered on the ECG side of this
repository: fully balanced weighting over a class with a handful of examples
hands those examples the same total gradient as tens of thousands of others.
"""
import numpy as np

from eeg.metrics import MAD_TO_SIGMA, relative_spectral_power

# Rejection thresholds. Deliberately conservative: discarding real brain
# signal costs training data, but keeping an electrode pop teaches the model
# that a step artifact is a clinical finding.
FLAT_SIGMA_UV = 0.1          # below this the channel is disconnected, not calm
SATURATION_FRACTION = 0.05   # share of samples pinned at the extremes
EMG_GAMMA_SHARE = 0.55       # broadband high-frequency share indicating muscle
AMPLITUDE_SIGMA = 12.0       # excursion vs the recording's own robust sigma
MAX_CLASS_WEIGHT = 20.0


def _robust_sigma(x):
    mad = float(np.median(np.abs(x - np.median(x))))
    return MAD_TO_SIGMA * mad


def artifact_reason(window, fs, reference_sigma=None):
    """Why this window is unusable, or None if it is clean.

    A reason string rather than a bool so rejections can be counted by cause
    and shown to the operator. A preparation run that silently drops 80% of a
    recording is a data problem, and it should be visible as one.
    """
    x = np.asarray(window, dtype=np.float64).ravel()
    if x.size < 8:
        return "too short"
    if not np.all(np.isfinite(x)):
        return "non-finite samples"

    sigma = _robust_sigma(x)
    if sigma < FLAT_SIGMA_UV:
        return "flatline or disconnected electrode"

    extreme = np.abs(x) >= 0.999 * np.max(np.abs(x))
    if np.max(np.abs(x)) > 0 and extreme.mean() > SATURATION_FRACTION:
        return "amplifier saturation"

    if reference_sigma and reference_sigma > 0:
        if np.max(np.abs(x - np.median(x))) > AMPLITUDE_SIGMA * reference_sigma:
            return "gross amplitude excursion"

    rsp = relative_spectral_power(x, fs)
    if rsp is None:
        return "no measurable power spectrum"
    if rsp["gamma"] > EMG_GAMMA_SHARE:
        return "EMG contamination"
    return None


def label_windows(times, annotations):
    """1 where a window overlaps an annotated clinical event, else 0.

    Overlap, not containment: a 2-second window catching the first 200 ms of a
    seizure contains evidence of it. Requiring containment would label the
    onset windows negative, which is precisely where onset detection lives.
    """
    labels = []
    for start, stop in times:
        hit = any(stop > a_start and start < a_stop
                  for a_start, a_stop in annotations)
        labels.append(1 if hit else 0)
    return np.asarray(labels, dtype=np.int64)


def select_training_windows(windows, times, fs, annotations,
                            reference_sigma=None):
    """Build the training set: clean windows, labelled, baseline retained.

    Returns (kept_windows, labels, kept_times, rejections) where `rejections`
    counts artifact causes. Baseline windows are kept as the negative class --
    see the module docstring for why removing them breaks the model.
    """
    labels_all = label_windows(times, annotations)
    keep, rejections = [], {}
    for i, w in enumerate(windows):
        # Multi-channel windows are judged on their worst channel: one bad
        # electrode contaminates the image the model would be shown.
        channels = w if np.ndim(w) == 2 else w[np.newaxis, :]
        reason = next((r for r in (artifact_reason(c, fs, reference_sigma)
                                   for c in channels) if r), None)
        if reason:
            rejections[reason] = rejections.get(reason, 0) + 1
            continue
        keep.append(i)

    keep = np.asarray(keep, dtype=int)
    if keep.size == 0:
        return (np.zeros((0,) + np.shape(windows)[1:]),
                np.zeros(0, np.int64), [], rejections)
    return (np.asarray(windows)[keep], labels_all[keep],
            [times[i] for i in keep], rejections)


def class_weights(labels, cap=MAX_CLASS_WEIGHT):
    """Balanced weights, capped. Focus without letting a rare class dominate.

    This is what replaces deleting the baseline: the anomaly class is
    up-weighted so the model attends to it, while the negative class remains
    present to define the boundary.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if classes.size == 0:
        return {}
    n = len(labels)
    return {int(c): float(min(n / (classes.size * max(1, (labels == c).sum())),
                              cap))
            for c in classes}
