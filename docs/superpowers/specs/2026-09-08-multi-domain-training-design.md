# Multi-Domain Model Training & Dataset Professionalization

**Date:** 2026-09-08
**Status:** Approved

## Problem

The application serves three domains (brain MRI, ECG, satellite land cover) but only
the MRI dataset exists on disk. Verification of the shipped models found two severe
defects:

1. The MRI classifier used `/255` scaling at inference while `train_vgg16.py` trained
   with `vgg16.preprocess_input`. Every scan returned "No Tumor" at 30-48% confidence.
2. `model/efficientnet_best.keras` was trained by `train_efficientnet.py` on
   `dataset/Training` -- the MRI set -- yet was loaded as the satellite land cover
   classifier. It returned class 0 at 99.8% for every scene, relabelled
   "Forest / Vegetation".

Both defects share a root cause: nothing recorded what a model was trained on or how
its inputs were prepared, so nothing could detect the mismatch. `train_qcnn.py` is
also misnamed -- it trains on the MRI images via `ImageFolder`, not on ECG data, and
imports torch, which is not installed.

## Goals

- Train models for all three domains from real, documented data.
- Both a scratch CNN and a VGG16 transfer model for MRI, on identical splits.
- A dataset layout that a reviewer can verify: provenance, checksums, deterministic
  splits, documented caveats.
- A model registry that makes the two defects above structurally impossible.

## Non-Goals

- GPU training. The target machine is CPU-only (4 cores, ~3 GB free RAM).
- Beating published benchmarks. Correctness and honesty over headline numbers.
- Replacing the QCNN. It is retained as an ECG fallback.

## Data Architecture

```
data/
  raw/                      # immutable, exactly as downloaded
    mri/  eurosat/  mitdb/
  processed/                # model-ready, reproducible from raw
    mri/{train,val,test}/<class>/
    eurosat/{train,val,test}/<class>/
    ecg/{train,val,test}.npz
  manifests/<domain>.csv    # sha256, class, split, source record
  DATASET.md                # provenance, licence, splits, caveats
```

Splits are deterministic (fixed seed), stratified, and recorded in manifests with
per-file SHA-256 so split membership is auditable.

### Sources

| Domain | Source | Size | Licence |
|---|---|---|---|
| MRI | already on disk (`dataset/`) | 7,200 imgs | as-received |
| Satellite | HuggingFace `blanchon/EuroSAT_RGB` (parquet) | 27,000 imgs, 64x64 | MIT / EuroSAT terms |
| ECG | PhysioNet `mitdb` via `wfdb` | 48 records, 360 Hz | ODC-By 1.0 |

### Split policy

- **MRI:** existing `Testing/` is the held-out test set. 15% stratified validation is
  carved from `Training/` with a fixed seed. No file appears in two splits (verified
  by SHA-256).
- **EuroSAT:** the upstream train/validation/test splits are used as published and
  recorded in the manifest.
- **ECG:** split by **record (patient)**, never by beat. Uses the inter-patient
  DS1/DS2 convention (de Chazal); paced records 102, 104, 107, 217 are excluded.
  Validation is carved from DS1 by whole record.

Beat-level splitting leaks patient identity between train and test and inflates
accuracy to roughly 99%. Inter-patient splitting is the reason this spec's ECG target
is far below commonly published figures.

## Models

| Domain | Architecture | Input | Preprocessing |
|---|---|---|---|
| MRI | scratch CNN, 4 conv blocks (32/64/128/128) + BN, GAP, dropout | 128x128x3 | rescale 1/255 |
| MRI | VGG16 frozen base + dense head, cached bottleneck features | 224x224x3 | `vgg16.preprocess_input` |
| Satellite | scratch CNN, 4 conv blocks (32/64/128/256), GAP | 64x64x3 | rescale 1/255 |
| ECG | 1D CNN, Conv1D 32/64/128, GAP, dense | 280x1 | per-beat z-score |

CPU strategy: VGG16 bottleneck features are computed once and cached, after which the
head trains in seconds per epoch. EuroSAT trains at native 64x64. The ECG 1D CNN is
inexpensive. Class weights address the ECG imbalance (N is ~90% of beats).

## Model Registry

Every trained model is written with a JSON sidecar:

```json
{"task": "mri",
 "classes": ["glioma", "meningioma", "notumor", "pituitary"],
 "input_shape": [224, 224, 3],
 "preprocessing": "vgg16_preprocess_input",
 "metrics": {"test_accuracy": 0.97, "macro_f1": 0.97},
 "trained": "2026-09-08"}
```

The application loads models only through the registry, which validates the sidecar
against what the calling code expects: class count and order, input shape, and
preprocessing identifier. A mismatch refuses the load rather than serving wrong
predictions. This is the structural fix for both defects in Problem.

## Success Criteria

| Model | Metric | Floor |
|---|---|---|
| MRI VGG16 | test accuracy | 0.93 |
| MRI CNN | test accuracy | 0.88 |
| EuroSAT CNN | test accuracy | 0.88 |
| ECG 1D CNN | macro F1 (inter-patient) | 0.60 |

The ECG floor is deliberately modest. Inter-patient S-class recall is poor throughout
the literature; a higher number would indicate a leaking split, not a better model.

## Testing

- **Known-label assertions:** each model predicts the correct class on held-out
  samples. Smoke tests alone cannot detect a confidently wrong model.
- **Sidecar contract:** class count, input shape and preprocessing name match the code.
- **Leakage:** zero record overlap across ECG splits; zero SHA-256 overlap across
  image splits.
- **Manifest integrity:** every processed file's checksum matches its manifest row.
- **Regression:** the existing 66-check application suite stays green.

## Integration

- Satellite page moves to 10 EuroSAT classes, displaying the specific class and its
  4-way group (Forest/Vegetation, Urban, Water, Agricultural) so the existing
  land-cover breakdown continues to render.
- ECG reports the AAMI class plus a normal/abnormal verdict; the QCNN is retained as
  a fallback when the trained model is absent.
- Patient and survey capture is unchanged. The PDF gains the model version and its
  test metrics, so a report states which model produced it.

## Risks

| Risk | Mitigation |
|---|---|
| CPU training too slow | Cached bottleneck features; native-resolution EuroSAT; early stopping |
| Download unavailable | Sources verified reachable; raw data cached under `data/raw/` |
| Memory exhaustion (3 GB free) | Batch generators, no full-dataset arrays in memory |
| ECG metrics disappoint | Floor set to an honest inter-patient value; per-class F1 reported |
