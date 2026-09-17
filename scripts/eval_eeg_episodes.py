"""Event-level evaluation of the seizure head on whole held-out recordings.

    python scripts/eval_eeg_episodes.py

The prepared test.npz cannot answer event-level questions: negatives were
subsampled, so its windows are not contiguous in time. This runs every
held-out recording end to end through eeg.inference -- the same path as the
dashboard -- scores each window once, and compares per-window thresholding
with temporal smoothing on those identical scores.

Smoothing parameters are the module defaults, fixed from clinical convention
before this script was run. Nothing here tunes them on the test recordings.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import numpy as np
import tensorflow as tf

from eeg.inference import (DEFAULT_MIN_EPISODE_S, DEFAULT_SMOOTH_WINDOWS,
                           DEFAULT_THRESHOLD, analyse_signal, smooth_scores)
from eeg.loaders import load_chb_record, parse_chb_summary
from eeg.metrics import anomaly_episodes
from prepare_eeg import CHB_DIR, CHB_TEST_FILES, _resample

CONFIGS = {"per-window": (1, 0.0),
           "smoothed": (DEFAULT_SMOOTH_WINDOWS, DEFAULT_MIN_EPISODE_S)}


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def evaluate(scores, artifact, times, seizures, k, min_s):
    smoothed = smooth_scores(scores, artifact, k)
    flags = (smoothed >= DEFAULT_THRESHOLD) & ~artifact
    eps = anomaly_episodes(flags, times, min_duration_s=min_s)
    spans = [(e["onset_s"], e["onset_s"] + e["duration_s"]) for e in eps]
    detected, latency = 0, []
    for sz in seizures:
        hits = [s for s in spans if _overlaps(s, sz)]
        if hits:
            detected += 1
            latency.append(max(0.0, min(h[0] for h in hits) - sz[0]))
    false_eps = [s for s in spans if not any(_overlaps(s, sz) for sz in seizures)]
    return {"seizures": len(seizures), "detected": detected,
            "latency_s": latency, "episodes": len(spans),
            "false_alarms": len(false_eps),
            "false_alarm_s": float(sum(b - a for a, b in false_eps))}


def main():
    model = tf.keras.models.load_model("model/eeg_seizure.keras", compile=False)
    card = json.load(open("model/eeg_seizure.json"))
    summary = parse_chb_summary(f"{CHB_DIR}/chb01-summary.txt")
    totals = {name: {"seizures": 0, "detected": 0, "false_alarms": 0,
                     "false_alarm_s": 0.0, "latency_s": []} for name in CONFIGS}
    hours = 0.0
    for name in CHB_TEST_FILES:
        signal, fs, seizures = load_chb_record(f"{CHB_DIR}/{name}", summary)
        signal, fs = _resample(signal, fs)
        r = analyse_signal(signal, fs, model=model, card=card,
                           smooth_windows=1, min_episode_s=0.0)
        scores = np.array([w["score"] for w in r["windows"]])
        artifact = np.array([w["artifact"] is not None for w in r["windows"]])
        times = [(w["start_s"], w["stop_s"]) for w in r["windows"]]
        hours += r["duration_s"] / 3600
        for cfg, (k, min_s) in CONFIGS.items():
            e = evaluate(scores, artifact, times, seizures, k, min_s)
            print(f"{name} [{cfg}]: seizures {e['detected']}/{e['seizures']}, "
                  f"false-alarm episodes {e['false_alarms']} "
                  f"({e['false_alarm_s']:.0f}s), latency {e['latency_s']}",
                  flush=True)
            for key in ("seizures", "detected", "false_alarms", "false_alarm_s"):
                totals[cfg][key] += e[key]
            totals[cfg]["latency_s"] += e["latency_s"]

    print(f"\n{hours:.2f} h of held-out recording")
    for cfg, t in totals.items():
        lat = t["latency_s"]
        print(f"{cfg}: seizures {t['detected']}/{t['seizures']}, "
              f"false alarms {t['false_alarms']} "
              f"({t['false_alarms'] / hours:.1f}/h, "
              f"{t['false_alarm_s']:.0f}s flagged), "
              f"median latency {np.median(lat) if lat else float('nan'):.0f}s")


if __name__ == "__main__":
    main()
