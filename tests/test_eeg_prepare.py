import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import prepare_eeg  # noqa: E402


def _part(label, source, n=4):
    return (np.zeros((n, 2, 2, 1)), np.full(n, label), np.full(n, source))


def test_save_refuses_a_split_holding_one_class(tmp_path, monkeypatch):
    # A test split of only positives makes precision trivially 1.0 and AUC
    # undefined -- it reads as a result while measuring nothing.
    monkeypatch.setattr(prepare_eeg, "OUT", str(tmp_path))
    buckets = {"train": [_part(1, "a"), _part(0, "b")],
               "test": [_part(1, "c")]}
    with pytest.raises(SystemExit, match="test.*one class"):
        prepare_eeg._save("t", buckets, {}, {})


def test_save_accepts_splits_with_both_classes(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_eeg, "OUT", str(tmp_path))
    buckets = {"train": [_part(1, "a"), _part(0, "b")],
               "test": [_part(1, "c"), _part(0, "d")]}
    prepare_eeg._save("t", buckets, {}, {})
    assert (tmp_path / "t" / "manifest.json").exists()


def test_download_subject_groups_match_participants_file():
    # The control list once named frontotemporal (F) subjects; prep then
    # silently dropped them all and left no negative class.
    import download_eeg
    from eeg.loaders import parse_participants
    tsv = "data/raw/ds004504/participants.tsv"
    if not os.path.exists(tsv):
        pytest.skip("participants.tsv not downloaded")
    groups = parse_participants(tsv)
    for group, subjects in download_eeg.ADF_SUBJECTS.items():
        for sub in subjects:
            assert groups[sub] == group, sub
    assert set(prepare_eeg.ADF_TEST_SUBJECTS) <= {
        s for subs in download_eeg.ADF_SUBJECTS.values() for s in subs}
