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

RAW, OUT = "data/raw/mitdb", "data/processed/ecg"
SEED, BEFORE, AFTER = 42, 100, 180          # 280 samples at 360 Hz
CLASSES = ["N", "S", "V", "F", "Q"]

# AAMI groupings of the MIT-BIH beat annotation symbols.
AAMI = {}
AAMI.update({s: "N" for s in ["N", "L", "R", "e", "j"]})
AAMI.update({s: "S" for s in ["A", "a", "J", "S"]})
AAMI.update({s: "V" for s in ["V", "E"]})
AAMI.update({"F": "F"})
AAMI.update({s: "Q" for s in ["/", "f", "Q"]})


def beats_for(record):
    """Extract z-scored 280-sample windows centred on each annotated beat."""
    base = os.path.join(RAW, str(record))
    rec = wfdb.rdrecord(base)
    ann = wfdb.rdann(base, "atr")
    lead = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
    signal = rec.p_signal[:, lead].astype(np.float32)

    X, y = [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        cls = AAMI.get(symbol)
        if cls is None:
            continue                      # rhythm/quality annotations, not beats
        start, end = sample - BEFORE, sample + AFTER
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end]
        std = float(beat.std())
        beat = (beat - beat.mean()) / (std if std > 1e-6 else 1.0)
        X.append(beat)
        y.append(CLASSES.index(cls))
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


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
        X, y = beats_for(record)
        if len(X) == 0:
            continue
        buckets[split].append((X, y, np.full(len(X), str(record))))
        rows.append({"path": f"{OUT}/{split}.npz", "label": "mixed", "split": split,
                     "sha256": "", "source": f"mitdb/{record}"})
        print(f"  record {record} -> {split}: {len(X)} beats", flush=True)

    for split, parts in buckets.items():
        if not parts:
            raise SystemExit(f"no records for split {split}")
        X = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        records = np.concatenate([p[2] for p in parts])
        np.savez_compressed(f"{OUT}/{split}.npz", X=X, y=y, records=records)
        dist = {CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))}
        print(f"{split}: {len(X)} beats from {len(set(records.tolist()))} records {dist}")

    os.makedirs("data/manifests", exist_ok=True)
    write_manifest(rows, "data/manifests/ecg.csv")


if __name__ == "__main__":
    main()
