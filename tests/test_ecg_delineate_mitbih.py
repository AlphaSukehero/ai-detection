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
