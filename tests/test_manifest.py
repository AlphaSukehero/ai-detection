import pytest
from mlkit.manifest import sha256, write_manifest, read_manifest, assert_no_split_overlap, LeakageError


def test_sha256_stable(tmp_path):
    f = tmp_path / "a.txt"; f.write_bytes(b"hello")
    assert sha256(str(f)) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_manifest_roundtrip(tmp_path):
    rows = [{"path": "a.png", "label": "x", "split": "train", "sha256": "d", "source": "s"}]
    out = tmp_path / "m.csv"
    write_manifest(rows, str(out))
    assert read_manifest(str(out)) == rows


def test_overlap_detected():
    rows = [{"sha256": "d", "split": "train"}, {"sha256": "d", "split": "test"}]
    with pytest.raises(LeakageError):
        assert_no_split_overlap(rows, "sha256")


def test_no_overlap_passes():
    rows = [{"sha256": "a", "split": "train"}, {"sha256": "b", "split": "test"}]
    assert_no_split_overlap(rows, "sha256")
