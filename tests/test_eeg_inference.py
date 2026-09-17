"""Whole-recording analysis: timeline, episodes, and what happens with no model."""
import numpy as np
import pytest

from eeg.inference import analyse_signal, summarise

FS = 128.0


def _recording(seconds=30.0, fs=FS, seizure=None, seed=0):
    """Background alpha, optionally with a high-amplitude 3 Hz burst."""
    t = np.arange(0, seconds, 1 / fs)
    rng = np.random.default_rng(seed)
    sig = 20 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 2, t.size)
    if seizure:
        a, b = seizure
        m = (t >= a) & (t < b)
        sig[m] += 120 * np.sin(2 * np.pi * 3 * t[m])
    return sig


class _Model:
    """Flags windows whose image has high low-frequency energy."""

    def __init__(self, scores=None):
        self.scores = scores
        self.calls = 0

    def predict(self, batch, verbose=0):
        self.calls += 1
        n = len(batch)
        if self.scores is not None:
            out = np.asarray(self.scores[:n], dtype=float)
            self.scores = self.scores[n:]
            return out.reshape(-1, 1)
        return np.zeros((n, 1))


def test_without_a_model_metrics_are_reported_and_nothing_is_flagged():
    """Silence must not read as 'no seizure found'."""
    r = analyse_signal(_recording(), FS, model=None)
    assert r["model_used"] is False
    assert r["n_windows"] > 0
    assert all(w["score"] is None and not w["flagged"] for w in r["windows"])
    assert r["episodes"] == []
    assert "no window was classified" in r["note"]


def test_every_window_carries_its_clinical_parameters():
    r = analyse_signal(_recording(), FS, model=None)
    w = r["windows"][0]
    for key in ("rsp", "spectral_entropy", "spikes", "start_s", "stop_s"):
        assert key in w


def test_a_recording_shorter_than_one_window_is_refused():
    r = analyse_signal(np.zeros(int(FS)), FS, model=None, window_s=2.0)
    assert r["n_windows"] == 0 and "shorter than one" in r["error"]


def test_timeline_covers_the_recording():
    r = analyse_signal(_recording(seconds=20.0), FS, model=None)
    assert r["windows"][0]["start_s"] == 0.0
    assert r["duration_s"] == pytest.approx(20.0, abs=2.0)


def test_consecutive_flagged_windows_become_one_episode():
    r0 = analyse_signal(_recording(), FS, model=None)
    n = r0["n_windows"]
    scores = [0.0] * n
    for i in range(5, 10):
        scores[i] = 0.9
    r = analyse_signal(_recording(), FS, model=_Model(scores))
    assert r["model_used"] is True
    assert len(r["episodes"]) == 1
    ep = r["episodes"][0]
    assert ep["onset_s"] == r["windows"][5]["start_s"]
    assert ep["peak_score"] == pytest.approx(0.9)


def test_artifact_windows_are_never_flagged():
    """A clenched jaw is not a seizure. If EMG can raise an alarm the system
    loses trust before it earns any."""
    sig = _recording(seconds=20.0)
    fs = FS
    t = np.arange(sig.size) / fs
    m = (t >= 6) & (t < 10)
    sig[m] = sum(np.sin(2 * np.pi * f * t[m]) for f in range(32, 45)) * 60
    n = analyse_signal(sig, fs, model=None)["n_windows"]
    r = analyse_signal(sig, fs, model=_Model([1.0] * n))
    flagged_artifacts = [w for w in r["windows"] if w["artifact"] and w["flagged"]]
    assert flagged_artifacts == []


def test_batching_covers_every_window():
    r0 = analyse_signal(_recording(seconds=120.0), FS, model=None)
    n = r0["n_windows"]
    assert n > 64, "test needs more than one batch to be meaningful"
    m = _Model([0.0] * n)
    r = analyse_signal(_recording(seconds=120.0), FS, model=m)
    assert len(r["windows"]) == n
    assert all(w["score"] is not None for w in r["windows"])
    assert m.calls > 1


def test_summary_reports_onset_of_the_earliest_episode():
    r0 = analyse_signal(_recording(), FS, model=None)
    scores = [0.0] * r0["n_windows"]
    scores[4] = scores[5] = 0.95
    s = summarise(analyse_signal(_recording(), FS, model=_Model(scores)))
    assert s["episodes"] == 1
    assert "onset at" in s["headline"]
    assert s["burden_pct"] > 0


def test_summary_says_so_when_nothing_was_detected():
    r0 = analyse_signal(_recording(), FS, model=None)
    s = summarise(analyse_signal(_recording(), FS,
                                 model=_Model([0.0] * r0["n_windows"])))
    assert s["episodes"] == 0
    assert "No anomalous episode" in s["headline"]


def test_summary_without_a_model_does_not_claim_absence():
    s = summarise(analyse_signal(_recording(), FS, model=None))
    assert "No anomalous episode" not in s["headline"]
