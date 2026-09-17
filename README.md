# AI Detection

A Flask application that runs three image/signal analysers behind a web UI and
generates a PDF report for each:

| Page | Input | Model | Measured performance |
|---|---|---|---|
| `/ecg` | MIT-BIH CSV/TXT, or a photo of an ECG strip | 1D CNN, AAMI 5-class beat classifier | accuracy 0.614, **macro-F1 0.285** |
| `/brain-tumor` | MRI slice (JPEG/PNG/DICOM) | VGG16 transfer, 4-class | accuracy 0.927, macro-F1 0.925 |
| `/satellite` | Satellite scene (JPEG/PNG/TIFF) | EuroSAT CNN, 10-class → 4 display groups | accuracy 0.957, macro-F1 0.957 |

Those numbers come from each model's JSON card in `model/`, written at
training time. They are the numbers the app is entitled to claim.

**This is a research prototype, not a medical device.** See
[Honesty guarantees](#honesty-guarantees) for what that means in the code.

## Running it

Model checkpoints are not in git (they total ~350 MB). Train them with
`scripts/train_all.sh`, or drop existing `.keras` files plus their `.json`
cards into `model/`.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py          # http://127.0.0.1:5050
```

`PORT` overrides the port. `UPLOAD_TTL_SECONDS` (default 86400) sets how long
generated files survive in `uploads/` before the sweep removes them.

Without a model installed, each page degrades rather than guessing — see
below.

## Training

```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python scripts/download_raw.py     # raw datasets -> data/raw/
.venv/bin/python scripts/prepare_ecg.py      # ... and prepare_mri, prepare_eurosat
scripts/train_all.sh                         # or: scripts/train_all.sh ecg mri
EPOCHS=1 scripts/train_all.sh                # smoke test the whole pipeline
```

Provenance, split policy and caveats for every dataset are in
[`data/DATASET.md`](data/DATASET.md). `SEED = 42` throughout; splits are
deterministic.

Each training script writes a **model card** (`model/<name>.json`) recording
the task, class order, input shape, preprocessing and measured metrics. The
app refuses to load any checkpoint whose card does not match what the calling
code expects — see `mlkit/registry.py`. Two defects that shipped here before
the registry existed (a VGG16 model fed `/255` inputs; a brain-tumor model
serving land cover) are now refused loads rather than confident nonsense.

## Honesty guarantees

The central risk in this codebase is not a crash, it is a plausible-looking
number with nothing behind it. Several such paths existed and were removed;
these are enforced by tests in `tests/test_no_fabricated_output.py`:

- **No fallback classifiers.** `predict_ecg()` and `predict_brain_tumor()`
  raise when no validated model is installed. They previously fell back to a
  quantum circuit whose preprocessing was a cross-version unpickled
  scikit-learn estimator, to `sigmoid(std(signal))` presented as a
  confidence, and to hardcoded probability tables (`78.4% Glioma`) chosen by
  a brightness threshold.
- **No unmeasured class names.** A beat class whose card-recorded F1 is below
  `MIN_REPORTABLE_F1` (0.30) is never named in the UI or the report. With the
  current ECG model that suppresses S, F and Q — their F1s are 0.097, 0.008
  and 0.003. Such beats still count toward "abnormal"; they are reported as
  an unnamed group.
- **Reports state absence.** With no classification, the PDF prints
  "Not performed" and the reason, not em-dashes that read like a missing
  measurement.
- **Measurements are measured.** ECG intervals are derived from the signal
  with an explicit timebase, and report "unavailable" when the timebase is
  unknown (a photo with no detectable grid). They are not constants.

### Known limitation: the ECG model

Macro-F1 0.285 is an honest **inter-patient** number (de Chazal DS1/DS2 split,
balanced class weights) — not a bug, and not comparable to the ~99% figures
that beat-wise splits produce. But it means the model reliably distinguishes
only N and V.

The most promising fix is architectural rather than a hyperparameter:
the model sees a z-scored 280-sample window centred on each beat, which
discards RR-interval context. Supraventricular ectopic beats are *defined* by
prematurity, so S is close to undetectable from that window alone. Adding
pre-RR / post-RR / local-RR-ratio features is the standard remedy and the
single biggest lever available here.

### Known limitation: tumour morphometry

"Estimated Area", "Estimated Severity" and "Estimated Spread" on the MRI page
are computed from a Grad-CAM heatmap thresholded at the 90th percentile.
Grad-CAM shows where a classifier looked; it is not a segmentation, and these
figures inherit that. They are labelled "Estimated" for that reason. A small
U-Net trained on BraTS would make them real measurements.

## Layout

```
app.py              Flask routes, model loading, per-domain prediction logic
ecg/                Signal processing: digitize, delineate, parameters, quality
vision/gradcam.py   Saliency maps and the tumour geometry derived from them
reporting/pdf.py    ReportLab document assembly
webapp/metadata.py  Patient / survey metadata shared by pages and reports
mlkit/              Model cards (registry) and dataset manifests
scripts/            download_raw, prepare_*, train_*
tests/              137 tests; model-dependent ones skip when no checkpoint
```

## Development

```bash
.venv/bin/python -m pytest -q
python3 -m ruff check .
```

Both run in CI (`.github/workflows/ci.yml`). Tests that need a model checkpoint
skip themselves, so everything encoding a correctness guarantee — the ECG
measurement maths, the registry, the upload name policy, and the
fabricated-output regressions — runs without the ~350 MB of weights.
