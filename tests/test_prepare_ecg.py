import os
import numpy as np
import pytest

OUT = "data/processed/ecg"
pytestmark = pytest.mark.skipif(not os.path.exists(f"{OUT}/train.npz"),
                                reason="run scripts/prepare_ecg.py first")


def _records(split):
    return set(np.load(f"{OUT}/{split}.npz", allow_pickle=True)["records"].tolist())


def test_no_patient_appears_in_two_splits():
    train, val, test = _records("train"), _records("val"), _records("test")
    assert train & test == set(), f"record leak train/test: {train & test}"
    assert train & val == set(), f"record leak train/val: {train & val}"
    assert val & test == set(), f"record leak val/test: {val & test}"


def test_paced_records_excluded():
    everything = _records("train") | _records("val") | _records("test")
    assert everything.isdisjoint({"102", "104", "107", "217"})


def test_beat_window_is_280_samples():
    assert np.load(f"{OUT}/train.npz")["X"].shape[1] == 280


def test_labels_within_five_aami_classes():
    y = np.load(f"{OUT}/train.npz")["y"]
    assert y.min() >= 0 and y.max() <= 4
