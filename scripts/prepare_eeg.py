"""Turn raw recordings into labelled, artifact-screened window images.

Two tasks, prepared separately because their labels mean different things.

seizure (CHB-MIT)
    Per-event onset/offset annotations, so each window is labelled by what it
    actually overlaps. Split is by RECORDING: whole files go to train or
    test, never windows from the same file to both. Windows overlap by 50%,
    so a window-level split would put two views of the same seconds on both
    sides and report a score that is mostly memorisation.

alzheimer (ds004504)
    One diagnosis per subject and no indication of which moments carry the
    biomarker, so every window inherits its subject's label. That is weak
    labelling and it is the honest ceiling on this head: the model is shown
    windows labelled "Alzheimer's" that may contain nothing distinctive.
    Split is by SUBJECT for the same reason as above, more strongly -- two
    windows from one person are not independent samples.

Both splits are recorded in the manifest so the claim can be checked rather
than trusted.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np

from eeg.images import render_batch
from eeg.loaders import (RecordingError, load_chb_record, parse_chb_summary,
                         parse_participants, read_recording)
from eeg.masking import select_training_windows
from eeg.windows import DEFAULT_OVERLAP, DEFAULT_WINDOW_S, segment

OUT = "data/processed/eeg"
CHB_DIR = "data/raw/chbmit/chb01"
ADF_DIR = "data/raw/ds004504"
TARGET_FS = 128.0          # every head is trained at one rate
SEED = 42
# Negative windows kept per positive window, per recording. This bounds the
# dataset without removing the negative class: every positive window is kept,
# and the negatives are a uniform random sample of the same recording, so the
# boundary the model learns is still between real seizure and real background
# from the same patient, electrodes and session.
#
# It exists because the full set is ~40k windows of which ~1% are positive,
# and rendering a Morlet scalogram for each takes hours -- the low-frequency
# scales have very wide kernels. A recording with no positive windows at all
# keeps NEG_FLOOR of them, so seizure-free files still contribute background.
NEG_PER_POS = 20
NEG_FLOOR = 400

# Held-out recordings/subjects. Chosen before looking at any result, and
# written down here rather than derived at runtime so the split cannot drift
# between runs.
CHB_TEST_FILES = ["chb01_18.edf", "chb01_21.edf", "chb01_26.edf",
                  "chb01_06.edf"]
ADF_TEST_SUBJECTS = ["sub-005", "sub-006", "sub-041", "sub-042"]


def _resample(signal, fs, target=TARGET_FS):
    """Linear resample to the common rate, preserving the time span."""
    if abs(fs - target) < 1e-9:
        return signal, fs
    n_out = max(2, int(round(signal.shape[-1] * target / fs)))
    idx = np.linspace(0, signal.shape[-1] - 1, n_out)
    src = np.arange(signal.shape[-1])
    out = np.stack([np.interp(idx, src, ch) for ch in np.atleast_2d(signal)])
    return out, float(target)


def _subsample_negatives(labels, rng):
    """Indices keeping every positive and a bounded sample of negatives."""
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    budget = max(NEG_FLOOR, NEG_PER_POS * len(pos))
    if len(neg) > budget:
        neg = rng.choice(neg, size=budget, replace=False)
    return np.sort(np.concatenate([pos, neg]))


def _windows_for(signal, fs, annotations, rng):
    signal, fs = _resample(signal, fs)
    windows, times = segment(signal, fs, DEFAULT_WINDOW_S, DEFAULT_OVERLAP)
    if len(windows) == 0:
        return None
    kept, labels, kept_times, rejections = select_training_windows(
        windows, times, fs, annotations)
    if len(kept) == 0:
        return None
    pick = _subsample_negatives(labels, rng)
    return (kept[pick], labels[pick], [kept_times[i] for i in pick],
            rejections, fs)


def prepare_seizure():
    summary = parse_chb_summary(f"{CHB_DIR}/chb01-summary.txt")
    files = sorted(f for f in os.listdir(CHB_DIR) if f.endswith(".edf"))
    if not files:
        raise SystemExit(f"no EDF files in {CHB_DIR}; run scripts/download_eeg.py")

    buckets = {"train": [], "test": []}
    rejected_total = {}
    rng = np.random.default_rng(SEED)
    for name in files:
        split = "test" if name in CHB_TEST_FILES else "train"
        try:
            signal, fs, ann = load_chb_record(f"{CHB_DIR}/{name}", summary)
        except RecordingError as e:
            print(f"  {name}: skipped ({e})")
            continue
        got = _windows_for(signal, fs, ann, rng)
        if got is None:
            print(f"  {name}: no usable windows")
            continue
        kept, labels, times, rejections, out_fs = got
        images = render_batch(kept, out_fs)
        buckets[split].append((images, labels, np.full(len(labels), name)))
        for k, v in rejections.items():
            rejected_total[k] = rejected_total.get(k, 0) + v
        print(f"  {name} -> {split}: {len(labels)} windows, "
              f"{int(labels.sum())} seizure", flush=True)

    _save("seizure", buckets, rejected_total,
          {"split_by": "recording", "test_recordings": CHB_TEST_FILES})


def prepare_alzheimer():
    groups = parse_participants(f"{ADF_DIR}/participants.tsv")
    subjects = sorted(d for d in os.listdir(ADF_DIR) if d.startswith("sub-"))
    if not subjects:
        raise SystemExit(f"no subjects in {ADF_DIR}; run scripts/download_eeg.py")

    buckets = {"train": [], "test": []}
    rejected_total = {}
    rng = np.random.default_rng(SEED)
    for sub in subjects:
        group = groups.get(sub)
        if group not in ("A", "C"):
            continue                       # F (frontotemporal) kept out: binary task
        path = f"{ADF_DIR}/{sub}/eeg/{sub}_task-eyesclosed_eeg.set"
        if not os.path.exists(path):
            continue
        split = "test" if sub in ADF_TEST_SUBJECTS else "train"
        try:
            signal, fs, _ = read_recording(path)
        except RecordingError as e:
            print(f"  {sub}: skipped ({e})")
            continue
        signal, fs = _resample(signal, fs)
        windows, times = segment(signal, fs, DEFAULT_WINDOW_S, DEFAULT_OVERLAP)
        # The whole recording carries the subject's diagnosis; see the module
        # docstring on why that is weak labelling.
        kept, _, kept_times, rejections = select_training_windows(
            windows, times, fs, annotations=[])
        if len(kept) == 0:
            continue
        labels = np.full(len(kept), 1 if group == "A" else 0, dtype=np.int64)
        # Every window carries the subject label, so a long recording would
        # otherwise let one person dominate the split by sheer length.
        if len(kept) > NEG_FLOOR:
            pick = np.sort(rng.choice(len(kept), size=NEG_FLOOR, replace=False))
            kept, labels = kept[pick], labels[pick]
            kept_times = [kept_times[i] for i in pick]
        images = render_batch(kept, fs)
        buckets[split].append((images, labels, np.full(len(labels), sub)))
        for k, v in rejections.items():
            rejected_total[k] = rejected_total.get(k, 0) + v
        print(f"  {sub} ({group}) -> {split}: {len(labels)} windows", flush=True)

    _save("alzheimer", buckets, rejected_total,
          {"split_by": "subject", "test_subjects": ADF_TEST_SUBJECTS})


def _save(task, buckets, rejections, split_info):
    os.makedirs(f"{OUT}/{task}", exist_ok=True)
    manifest = {"task": task, "neg_per_pos": NEG_PER_POS,
                "neg_floor": NEG_FLOOR, "window_s": DEFAULT_WINDOW_S,
                "overlap": DEFAULT_OVERLAP, "fs": TARGET_FS,
                "artifact_rejections": rejections, **split_info, "splits": {}}
    for split, parts in buckets.items():
        if not parts:
            raise SystemExit(f"{task}: split {split!r} is empty")
        classes = np.unique(np.concatenate([p[1] for p in parts]))
        if len(classes) < 2:
            # Precision is trivially perfect and AUC undefined on one class:
            # a number that looks like a result and measures nothing.
            raise SystemExit(f"{task}: split {split!r} holds one class only "
                             f"({classes.tolist()})")
        X = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        src = np.concatenate([p[2] for p in parts])
        np.savez_compressed(f"{OUT}/{task}/{split}.npz", X=X, y=y, source=src)
        manifest["splits"][split] = {
            "windows": int(len(y)), "positive": int(y.sum()),
            "sources": sorted(set(src.tolist())),
        }
        print(f"{task}/{split}: {len(y)} windows, {int(y.sum())} positive, "
              f"{len(set(src.tolist()))} sources")

    # Overlapping windows make a shared source across splits a leak, so this
    # is asserted rather than assumed.
    train_src = set(manifest["splits"]["train"]["sources"])
    test_src = set(manifest["splits"]["test"]["sources"])
    overlap = train_src & test_src
    if overlap:
        raise SystemExit(f"{task}: sources in both splits: {sorted(overlap)}")

    with open(f"{OUT}/{task}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"rejected windows: {rejections}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "seizure"):
        prepare_seizure()
    if which in ("all", "alzheimer"):
        prepare_alzheimer()


if __name__ == "__main__":
    main()
