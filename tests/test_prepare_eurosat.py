import os
import pytest
from mlkit.manifest import read_manifest, assert_no_split_overlap

MANIFEST = "data/manifests/eurosat.csv"
pytestmark = pytest.mark.skipif(not os.path.exists(MANIFEST),
                                reason="run scripts/prepare_eurosat.py first")


def test_no_leakage():
    assert_no_split_overlap(read_manifest(MANIFEST), "sha256")


def test_ten_classes_in_every_split():
    rows = read_manifest(MANIFEST)
    for split in ["train", "val", "test"]:
        labels = {r["label"] for r in rows if r["split"] == split}
        assert len(labels) == 10, f"{split} has {len(labels)} classes, expected 10"


def test_expected_dataset_size():
    assert len(read_manifest(MANIFEST)) == 27000
