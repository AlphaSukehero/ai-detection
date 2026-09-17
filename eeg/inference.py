"""Full-recording inference: window, render, batch-predict, aggregate.

The requirement behind this module was "the uploaded image is one long
recording; slice it into the windows the model was trained on". The slicing
happens in SIGNAL space, not pixel space, and that is a deliberate
difference from the obvious reading.

Cutting a long plot into pixel strips and feeding those to the model would
mean the model sees a picture of a picture: a rasterised trace, re-rendered,
at whatever resolution the crop happened to land on. The model was trained on
scalograms computed from sampled voltages. So an uploaded image is first
digitised back to a signal (eeg.digitize), and from there it is the same code
path as an EDF upload -- one windowing implementation, one renderer, one
model input contract. Anything that cannot be digitised with a real timebase
is refused rather than approximated, because every frequency in this pipeline
is in Hz and Hz requires a timebase.
"""
import numpy as np

from eeg.images import render_batch
from eeg.masking import artifact_reason
from eeg.metrics import anomaly_episodes, window_report
from eeg.windows import DEFAULT_OVERLAP, DEFAULT_WINDOW_S, segment

DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 64


def analyse_signal(signal, fs, model=None, card=None,
                   window_s=DEFAULT_WINDOW_S, overlap=DEFAULT_OVERLAP,
                   threshold=DEFAULT_THRESHOLD, kind="scalogram"):
    """Analyse a whole recording and return a per-window timeline.

    `model` may be None: the clinical parameters are measured from the signal
    and do not depend on it. In that case every window reports its metrics
    and no window is flagged, which is an honest "not assessed" rather than a
    silent zero-risk verdict -- the caller is told via `model_used`.
    """
    windows, times = segment(signal, fs, window_s, overlap)
    if len(windows) == 0:
        return {
            "model_used": False, "n_windows": 0, "windows": [],
            "episodes": [], "duration_s": float(np.size(signal) / fs),
            "error": (f"Recording is shorter than one {window_s:.0f}s "
                      "window; nothing to analyse."),
        }

    # Robust sigma over the whole recording, so a window is judged against
    # this patient's own background rather than an absolute microvolt figure
    # that varies with montage and electrode impedance.
    flat = np.asarray(signal, dtype=np.float64).ravel()
    med = float(np.median(flat))
    reference_sigma = 1.4826 * float(np.median(np.abs(flat - med)))

    per_window = []
    for i, (w, (t0, t1)) in enumerate(zip(windows, times, strict=True)):
        sig = np.mean(w, axis=0) if np.ndim(w) == 2 else w
        per_window.append({
            "index": i, "start_s": float(t0), "stop_s": float(t1),
            "artifact": artifact_reason(sig, fs, reference_sigma),
            **window_report(sig, fs),
        })

    scores = _score(windows, fs, model, card, kind)
    if scores is None:
        for w in per_window:
            w["score"] = None
            w["flagged"] = False
        return {
            "model_used": False, "n_windows": len(per_window),
            "windows": per_window, "episodes": [],
            "duration_s": float(times[-1][1]),
            "note": ("No validated model was used, so no window was "
                     "classified. The measurements above are computed from "
                     "the signal and stand on their own."),
        }

    # An artifact window is never flagged: a clenched jaw is not a seizure,
    # and letting EMG raise an alarm is how these systems lose trust.
    for w, s in zip(per_window, scores, strict=True):
        w["score"] = float(s)
        w["flagged"] = bool(s >= threshold and w["artifact"] is None)

    episodes = anomaly_episodes([w["flagged"] for w in per_window], times)
    for ep in episodes:
        span = per_window[ep["first_window"]:ep["last_window"] + 1]
        ep["peak_score"] = max(w["score"] for w in span)
        ep["mean_score"] = float(np.mean([w["score"] for w in span]))
        ep["spike_rate_per_s"] = float(np.mean(
            [w["spikes"]["rate_per_s"] for w in span]))

    return {
        "model_used": True, "n_windows": len(per_window),
        "windows": per_window, "episodes": episodes,
        "duration_s": float(times[-1][1]),
        "threshold": threshold,
        "artifact_windows": sum(1 for w in per_window if w["artifact"]),
    }


def _score(windows, fs, model, card, kind):
    """Per-window anomaly probability, or None when no model can be used."""
    if model is None:
        return None
    images = render_batch(windows, fs, kind=kind)
    if len(images) == 0:
        return None

    out = []
    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start:start + BATCH_SIZE]
        preds = model.predict(batch, verbose=0)
        preds = np.asarray(preds)
        if preds.ndim == 2 and preds.shape[1] == 1:
            out.append(preds[:, 0])
        elif preds.ndim == 2:
            # Multi-class head: the anomaly probability is everything that is
            # not the negative class, read by the card's class order rather
            # than by assuming index 0.
            classes = (card or {}).get("classes")
            neg = classes.index("normal") if classes and "normal" in classes else 0
            out.append(1.0 - preds[:, neg])
        else:
            out.append(preds.ravel())
    return np.concatenate(out)


def summarise(result):
    """One-line clinical summary plus the headline numbers for the UI."""
    if result["n_windows"] == 0:
        return {"headline": result.get("error", "Nothing to analyse."),
                "episodes": 0, "burden_pct": 0.0}
    if not result["model_used"]:
        return {"headline": result.get("note", "No model was used."),
                "episodes": 0, "burden_pct": 0.0}

    episodes = result["episodes"]
    burden = sum(e["duration_s"] for e in episodes)
    pct = 100.0 * burden / result["duration_s"] if result["duration_s"] else 0.0
    if not episodes:
        headline = ("No anomalous episode was detected in "
                    f"{result['duration_s']:.0f}s of recording.")
    else:
        first = episodes[0]
        headline = (f"{len(episodes)} episode(s) detected; earliest onset at "
                    f"{first['onset_s']:.1f}s lasting "
                    f"{first['duration_s']:.1f}s.")
    return {"headline": headline, "episodes": len(episodes),
            "burden_pct": pct, "total_anomaly_s": burden,
            "artifact_windows": result.get("artifact_windows", 0)}
