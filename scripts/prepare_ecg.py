"""Segment MIT-BIH into AAMI-labelled beats using an inter-patient split.

Splitting by beat rather than by record leaks patient identity across splits and
inflates reported accuracy to roughly 99%. This uses the de Chazal DS1/DS2
record split, so no patient contributes beats to more than one split.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import wfdb
from mlkit.manifest import write_manifest

from ecg.beats import extract_beats

RAW, OUT = "data/raw/mitdb", "data/processed/ecg"
SEED = 42
CLASSES = ["N", "S", "V", "F", "Q"]

# AAMI groupings of the MIT-BIH beat annotation symbols.
AAMI = {}
AAMI.update({s: "N" for s in ["N", "L", "R", "e", "j"]})
AAMI.update({s: "S" for s in ["A", "a", "J", "S"]})
AAMI.update({s: "V" for s in ["V", "E"]})
AAMI.update({"F": "F"})
AAMI.update({s: "Q" for s in ["/", "f", "Q"]})


def beats_for(record):
    """Extract z-scored 280-sample windows plus RR context for each beat.

    Windowing and RR features come from ecg.beats, the same code the app uses
    at inference. Reference annotations supply the peak positions here; the
    app detects them. Everything after that point is shared, so the two
    cannot drift apart.
    """
    base = os.path.join(RAW, str(record))
    rec = wfdb.rdrecord(base)
    ann = wfdb.rdann(base, "atr")
    lead = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
    signal = rec.p_signal[:, lead].astype(np.float32)
    fs = float(rec.fs)

    # Keep only true beat annotations, and keep peaks and labels together:
    # RR context must be computed over the beat sequence as recorded, since a
    # dropped beat would fabricate a long interval across the gap.
    peaks, labels = [], []
    for sample, symbol in zip(ann.sample, ann.symbol, strict=True):
        cls = AAMI.get(symbol)
        if cls is None:
            continue                      # rhythm/quality annotations, not beats
        peaks.append(int(sample))
        labels.append(CLASSES.index(cls))

    if not peaks:
        empty = np.zeros((0, 280), np.float32)
        return empty, np.zeros((0, 4), np.float32), np.zeros(0, np.int64)

    X, R, kept = extract_beats(signal, np.asarray(peaks), fs)
    y = np.asarray(labels, np.int64)[kept]
    return X, R, y


def main():
    splits = json.load(open(f"{RAW}/splits.json"))
    rng = np.random.default_rng(SEED)
    ds1 = list(splits["DS1"])
    rng.shuffle(ds1)
    n_val = max(1, len(ds1) // 5)

    assign = {r: "val" for r in ds1[:n_val]}
    assign.update({r: "train" for r in ds1[n_val:]})
    assign.update({r: "test" for r in splits["DS2"]})

    os.makedirs(OUT, exist_ok=True)
    buckets = {"train": [], "val": [], "test": []}
    rows = []
    for record, split in sorted(assign.items()):
        if not os.path.exists(os.path.join(RAW, f"{record}.dat")):
            print(f"  record {record}: missing, skipped", flush=True)
            continue
        X, R, y = beats_for(record)
        if len(X) == 0:
            continue
        buckets[split].append((X, R, y, np.full(len(X), str(record))))
        rows.append({"path": f"{OUT}/{split}.npz", "label": "mixed", "split": split,
                     "sha256": "", "source": f"mitdb/{record}"})
        print(f"  record {record} -> {split}: {len(X)} beats", flush=True)

    for split, parts in buckets.items():
        if not parts:
            raise SystemExit(f"no records for split {split}")
        X = np.concatenate([p[0] for p in parts])
        R = np.concatenate([p[1] for p in parts])
        y = np.concatenate([p[2] for p in parts])
        records = np.concatenate([p[3] for p in parts])
        np.savez_compressed(f"{OUT}/{split}.npz", X=X, R=R, y=y, records=records)
        dist = {CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))}
        print(f"{split}: {len(X)} beats from {len(set(records.tolist()))} records {dist}")

    os.makedirs("data/manifests", exist_ok=True)
    write_manifest(rows, "data/manifests/ecg.csv")


if __name__ == "__main__":
    main()
