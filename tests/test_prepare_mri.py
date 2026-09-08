import os
import pytest
from mlkit.manifest import read_manifest, assert_no_split_overlap, sha256

MANIFEST = "data/manifests/mri.csv"
pytestmark = pytest.mark.skipif(not os.path.exists(MANIFEST),
                                reason="run scripts/prepare_mri.py first")


def test_no_leakage_between_splits():
    assert_no_split_overlap(read_manifest(MANIFEST), "sha256")


def test_all_splits_present_and_every_class_represented():
    rows = read_manifest(MANIFEST)
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    for split in ["train", "val", "test"]:
        labels = {r["label"] for r in rows if r["split"] == split}
        assert labels == {"glioma", "meningioma", "notumor", "pituitary"}


def test_checksums_match_files_on_disk():
    for row in read_manifest(MANIFEST)[:50]:
        assert os.path.exists(row["path"])
        assert sha256(row["path"]) == row["sha256"]


def test_no_duplicate_hashes_anywhere():
    rows = read_manifest(MANIFEST)
    digests = [r["sha256"] for r in rows]
    assert len(digests) == len(set(digests)), "duplicate images survived deduplication"
