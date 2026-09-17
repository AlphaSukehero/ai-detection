"""Inference must feed the model what the model was trained on.

The defect this file guards against is invisible at runtime: the app used to
chop a trace into consecutive 280-sample blocks while training centred every
window on the R peak at index 100. Nothing raised, nothing logged, the
numbers just meant less than they appeared to. Both sides now share
ecg.beats, and these tests pin that.
"""
import numpy as np
import pytest

import app as flask_app
from ecg.beats import BEAT_LEN, N_RR_FEATURES, BEFORE


def _signal_with_peaks(n_beats=12, rr=360, fs=360.0):
    """A smooth synthetic trace with realistic QRS morphology.

    Deliberately not white noise plus a one-sample spike: real ECG is
    band-limited, and sample-to-sample noise produces prominences that R-peak
    detection rightly treats as peaks. An unrealistic fixture here would test
    the detector against a signal it will never see.
    """
    length = (n_beats + 2) * rr
    t = np.arange(length, dtype=float)
    sig = np.zeros(length)
    for i in range(1, n_beats + 1):
        centre = i * rr
        # Narrow Gaussian for the R wave, a broader low one for the T wave.
        sig += 5.0 * np.exp(-0.5 * ((t - centre) / 4.0) ** 2)
        sig += 1.0 * np.exp(-0.5 * ((t - centre - 70) / 18.0) ** 2)
    # Smooth, low-amplitude baseline wander rather than white noise.
    sig += 0.05 * np.sin(2 * np.pi * t / (3.0 * rr))
    return sig


def test_prepare_beats_returns_aligned_beats_and_rr_context():
    beats, rr = flask_app.prepare_beats(_signal_with_peaks())
    assert beats.shape[1] == BEAT_LEN
    assert rr.shape == (len(beats), N_RR_FEATURES)


def test_prepare_beats_centres_on_the_r_peak_like_training_does():
    beats, _ = flask_app.prepare_beats(_signal_with_peaks())
    assert all(int(np.argmax(b)) == BEFORE for b in beats)


def test_prepare_beats_refuses_a_trace_with_no_detectable_beats():
    """Silence is the wrong answer here: zero beats must not become zero
    findings."""
    with pytest.raises(ValueError, match="No heartbeats"):
        flask_app.prepare_beats(np.zeros(5000))


def test_prepare_beats_accepts_a_multi_lead_array():
    leads = np.stack([_signal_with_peaks(), _signal_with_peaks()])
    beats, rr = flask_app.prepare_beats(leads)
    assert len(beats) == len(rr) > 0


class _TwoInputModel:
    """Records what it was called with, so the contract can be asserted."""

    def __init__(self):
        self.seen = None

    def predict(self, x, verbose=0):
        self.seen = x
        n = len(x["beat"]) if isinstance(x, dict) else len(x)
        return np.tile(np.array([0.9, 0.04, 0.03, 0.02, 0.01]), (n, 1))


CARD_WITH_RR = {
    "classes": ["N", "S", "V", "F", "Q"],
    "aux_inputs": {"rr": N_RR_FEATURES},
    "metrics": {"per_class_f1": {"N": 0.9, "S": 0.6, "V": 0.7,
                                 "F": 0.01, "Q": 0.0}},
}
CARD_WITHOUT_RR = {k: v for k, v in CARD_WITH_RR.items() if k != "aux_inputs"}


def test_a_card_declaring_rr_inputs_gets_both_tensors(monkeypatch):
    model = _TwoInputModel()
    monkeypatch.setattr(flask_app, "get_ecg_model", lambda: (model, CARD_WITH_RR))
    beats, rr = flask_app.prepare_beats(_signal_with_peaks())
    flask_app.predict_ecg_cnn((beats, rr))
    assert isinstance(model.seen, dict)
    assert set(model.seen) == {"beat", "rr"}
    assert model.seen["rr"].shape[1] == N_RR_FEATURES


def test_an_older_morphology_only_card_still_gets_one_tensor(monkeypatch):
    """The card decides, not the calling code -- an older checkpoint must
    keep working rather than being handed an input it cannot accept."""
    model = _TwoInputModel()
    monkeypatch.setattr(flask_app, "get_ecg_model",
                        lambda: (model, CARD_WITHOUT_RR))
    beats, rr = flask_app.prepare_beats(_signal_with_peaks())
    flask_app.predict_ecg_cnn((beats, rr))
    assert not isinstance(model.seen, dict)
    assert model.seen.shape[1:] == (BEAT_LEN, 1)


def test_mismatched_beat_and_rr_lengths_are_refused(monkeypatch):
    """Pairing a beat with another beat's rhythm would be silently wrong."""
    monkeypatch.setattr(flask_app, "get_ecg_model",
                        lambda: (_TwoInputModel(), CARD_WITH_RR))
    beats = np.zeros((5, BEAT_LEN), np.float32)
    rr = np.ones((3, N_RR_FEATURES), np.float32)
    with pytest.raises(flask_app.NoECGModelError, match="disagree in length"):
        flask_app.predict_ecg_cnn((beats, rr))
