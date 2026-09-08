import os
import numpy as np
import pytest

OUT = "data/processed/ecg"
# Guard on every split, not just train: an interrupted preparation run leaves
# zero-byte .npz files behind, which read back as a bare EOFError.
_SPLITS = ("train", "val", "test")
_ready = all(os.path.getsize(f"{OUT}/{s}.npz") > 0
             for s in _SPLITS if os.path.exists(f"{OUT}/{s}.npz")) and \
         all(os.path.exists(f"{OUT}/{s}.npz") for s in _SPLITS)
pytestmark = pytest.mark.skipif(
    not _ready, reason="run scripts/prepare_ecg.py first (splits missing or empty)")


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
