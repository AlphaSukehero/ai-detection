# Dataset Card

Provenance, splits and caveats for every dataset used by this project. All
raw downloads are reproducible via `scripts/download_raw.py`; all processed
trees are rebuilt by the per-domain `scripts/prepare_*.py`. Every processed
file is recorded with its SHA-256 in `data/manifests/<domain>.csv`.

Global rules: `SEED = 42` everywhere; splits are deterministic; the class
order recorded in a model's JSON sidecar is authoritative and is never
re-derived from a directory listing at inference time.

---

## 1. Brain MRI (4-class tumour classification)

- **Source:** the on-disk `dataset/` tree (Kaggle "Brain Tumor MRI Dataset",
  Masoud Nickparvar), redistributed under **CC0 1.0** (public domain
  dedication). Verify the licence at the source before redistributing.
- **Prepare:** `.venv/bin/python scripts/prepare_mri.py`
- **Classes (authoritative order):** `glioma`, `meningioma`, `notumor`,
  `pituitary`
- **Manifest:** `data/manifests/mri.csv`

### Split policy

`dataset/Testing` is kept whole as the held-out test set — it is never
touched during training or model selection. A stratified 15% validation
split is carved out of `dataset/Training` with the fixed seed.

**Deduplication:** the source tree contains 187 byte-identical duplicate
images. They are removed by SHA-256 *before* splitting, per class, keeping
the first filename in sorted order. Without this, the same image can land in
both train and validation, making validation scores optimistic.

### Counts

| Class | Train | Val | Test |
|---|---|---|---|
| glioma | 1190 | 210 | 386 |
| meningioma | 1179 | 207 | 398 |
| notumor | 1089 | 192 | 400 |
| pituitary | 1158 | 204 | 400 |
| **Total** | **4616** | **813** | **1584** |

### Caveats

- Images vary in acquisition plane, sequence and resolution; there is no
  patient identifier in the source, so a *patient-level* split is not
  possible. Slices from one patient may therefore appear in both train and
  test. Treat absolute accuracy as an upper bound on real-world performance.
- Deduplication is byte-level only. Near-duplicates (re-encoded or resized
  copies of the same slice) are not detected.

---

## 2. MIT-BIH Arrhythmia (5-class AAMI beat classification)

- **Source:** the MIT-BIH Arrhythmia Database on PhysioNet, fetched with
  `wfdb.dl_database`. Published under the **ODC-By 1.0** licence; PhysioNet's
  terms require citing Moody & Mark (2001) and Goldberger et al. (2000).
- **Download:** `.venv/bin/python scripts/download_raw.py mitdb`
- **Prepare:** `.venv/bin/python scripts/prepare_ecg.py`
- **Classes (authoritative order):** `N`, `S`, `V`, `F`, `Q` (AAMI EC57
  super-classes)
- **Manifest:** `data/manifests/ecg.csv`
- **Output:** `data/processed/ecg/{train,val,test}.npz`, each holding `X`
  `(n, 280)` float32, `y` `(n,)` int64, `records` `(n,)` string.

### Split policy — inter-patient, and why it matters

Splits follow the **de Chazal DS1/DS2 inter-patient** convention: DS1 records
are used for training (a fifth of them held out as validation, chosen with
the fixed seed), DS2 records form the test set. **No record ever appears in
two splits**, enforced by `tests/test_prepare_ecg.py`.

This is the single most important decision in this dataset. Splitting by
*beat* rather than by *patient* lets beats from the same recording — same
electrode placement, same morphology, same noise profile — appear on both
sides of the split. Models then partly memorise patient identity, and
reported accuracy inflates to roughly 99%. Many published MIT-BIH results
use that beat-level split and are not comparable to the numbers here.
**Expect materially lower metrics from this pipeline, especially on the
minority S and F classes; that is correct behaviour, not a regression.**

Paced records **102, 104, 107 and 217** are excluded by AAMI convention —
paced beats are a different signal generation process and are conventionally
not scored.

### Beat extraction

Each annotated beat is windowed 100 samples before and 180 samples after the
R peak (280 samples at 360 Hz, ~0.78 s), taken from the MLII lead where
present and lead 0 otherwise, then z-score normalised per beat. Beats whose
window would run off either end of the record are dropped. Annotation symbols
map to AAMI super-classes; symbols outside that mapping are discarded.

### Counts

Roughly 50,000 train and 50,000 test beats. See `scripts/prepare_ecg.py`
output for exact per-split, per-class distributions.

### Caveats

- **Severe class imbalance.** `N` dominates at roughly 90% of beats; `F` and
  `Q` are rare. Accuracy is a near-useless metric here — read per-class
  recall and macro-F1 instead.
- Beat segmentation uses the database's *reference* R-peak annotations, not a
  detector. A deployed system would have to run its own QRS detection, and
  detector error is not represented in these metrics.
- Per-beat z-scoring removes amplitude information that may carry clinical
  signal.
- The DS1/DS2 split is fixed by convention, so there is only one test set —
  repeated evaluation against it will eventually overfit it.

---

## Reproducing everything

```bash
.venv/bin/python scripts/download_raw.py all
.venv/bin/python scripts/prepare_mri.py
.venv/bin/python scripts/prepare_ecg.py
.venv/bin/python -m pytest tests/ -v
```

Every prepared file's SHA-256 is written to `data/manifests/`, so a rebuild
can be diffed against a previous one to prove the data did not change.
