"""Dataset manifests: every processed file's checksum, label and split.

The manifest is what makes a split auditable -- a reviewer can confirm that no
item crossed between train and test without re-running the preparation.
"""
import csv
import hashlib
from collections import defaultdict

FIELDS = ["path", "label", "split", "sha256", "source"]


class LeakageError(RuntimeError):
    """The same item appears in more than one split."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(rows, out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def assert_no_split_overlap(rows, key):
    """Raise LeakageError if any value of `key` appears under two splits."""
    seen = defaultdict(set)
    for row in rows:
        seen[row[key]].add(row["split"])
    bad = {k: v for k, v in seen.items() if len(v) > 1}
    if bad:
        sample = list(bad.items())[:5]
        raise LeakageError(f"{len(bad)} item(s) appear in multiple splits, e.g. {sample}")
