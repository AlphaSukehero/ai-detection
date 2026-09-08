"""Build data/processed/mri from the on-disk dataset/ tree.

dataset/Testing is kept whole as the held-out test set. A stratified 15%
validation split is carved from dataset/Training with a fixed seed.

The source tree contains 187 byte-identical duplicate images. They are
deduplicated by SHA-256 before splitting: a duplicate landing in both train and
validation would leak, making validation scores optimistic. Deduplication
happens per class, and the first filename in sorted order is kept so the choice
is deterministic.
"""
import os
import random
import shutil
import sys
from collections import OrderedDict

sys.path.insert(0, os.getcwd())
from mlkit.manifest import sha256, write_manifest, assert_no_split_overlap

SEED = 42
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SRC, OUT = "dataset", "data/processed/mri"
VAL_FRACTION = 0.15


def unique_files(split_src, cls):
    """Sorted (filename, sha256) pairs with byte-identical duplicates removed."""
    directory = os.path.join(SRC, split_src, cls)
    by_hash = OrderedDict()
    dropped = 0
    for fname in sorted(os.listdir(directory)):
        digest = sha256(os.path.join(directory, fname))
        if digest in by_hash:
            dropped += 1
            continue
        by_hash[digest] = fname
    return [(f, d) for d, f in by_hash.items()], dropped


def main():
    random.seed(SEED)
    rows, total_dropped = [], 0

    for split_src, split_out in [("Training", None), ("Testing", "test")]:
        for cls in CLASSES:
            files, dropped = unique_files(split_src, cls)
            total_dropped += dropped

            if split_out is None:
                random.shuffle(files)
                n_val = int(len(files) * VAL_FRACTION)
                assignment = ([("val", f) for f in files[:n_val]]
                              + [("train", f) for f in files[n_val:]])
            else:
                assignment = [(split_out, f) for f in files]

            for split, (fname, digest) in assignment:
                dst_dir = os.path.join(OUT, split, cls)
                os.makedirs(dst_dir, exist_ok=True)
                src_p = os.path.join(SRC, split_src, cls, fname)
                dst_p = os.path.join(dst_dir, fname)
                if not os.path.exists(dst_p):
                    shutil.copy2(src_p, dst_p)
                rows.append({"path": dst_p, "label": cls, "split": split,
                             "sha256": digest,
                             "source": f"{SRC}/{split_src}/{cls}/{fname}"})

    assert_no_split_overlap(rows, "sha256")
    os.makedirs("data/manifests", exist_ok=True)
    write_manifest(rows, "data/manifests/mri.csv")

    counts = {}
    for r in rows:
        counts.setdefault(r["split"], {}).setdefault(r["label"], 0)
        counts[r["split"]][r["label"]] += 1
    print(f"  deduplicated: {total_dropped} byte-identical images dropped")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {sum(counts[split].values())} {counts[split]}")


if __name__ == "__main__":
    main()
