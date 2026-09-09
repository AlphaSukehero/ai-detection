# ECG Image Analysis: Parameter Extraction and Image-Trained Classifier

Date: 2026-09-09
Status: Approved design, not yet implemented

## Problem

The ECG page reports clinical parameters that are partly fabricated and a
classification that is confidently wrong.

Verified against the running app on 2026-09-09:

- `pr_interval` is hardcoded to `"145.0 ms"` (`app.py:560`) and never computed.
  Every patient receives the same PR value.
- Rhythm, ST segment and QRS axis are not computed at all.
- The serving classifier (`model/ecg_cnn.keras`, macro F1 0.285) classified the
  abnormal demo sample as "Unclassifiable beat" at 100% confidence. That class
  has a per-class F1 of 0.003.
- QTc uses Bazett, which returned 722 ms at 195 bpm — a formula artifact.

ECG image upload already exists (`extract_ecg_signal_from_image`, `app.py:692`)
and digitizes a strip by column-wise centre-of-mass tracing.

## Goals

1. Compute seven ECG parameters honestly, with per-parameter quality flags.
2. Accept both single-lead strips and 12-lead printouts, auto-detected.
3. Train an ECG-image classifier, gated so a weak model refuses to serve.

## Non-goals

- Clinical validation or regulatory clearance. All output stays clearly marked
  as model-derived estimates.
- Retraining the MRI or satellite models.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Input layout | Both, auto-detected | Users have both; grid detection distinguishes them |
| Unreliable values | Render `—` + quality flag | Never print a fabricated clinical number |
| Classifier | VGG16 trained on synthetic ECG images | User preference, recorded below with its cost |
| Class balance | In scope | Current macro F1 of 0.285 makes the model unusable |
| QTc formula | Fridericia, formula name displayed | Bazett breaks above ~120 bpm |

### Recorded cost of the image-model decision

No ECG-image dataset exists on this machine. The image classifier therefore
requires a synthetic dataset generated from MIT-BIH, and a model trained on
synthetic renders will score well on held-out synthetic images while performing
materially worse on real photographs. The augmentation pipeline is what narrows
that gap. This was raised before the decision and is documented so the
limitation is not rediscovered during evaluation.

## Architecture

Delineation is separated from classification. `app.py` is 1897 lines and mixes
routing, PDF generation and three model paths; new code goes in its own module.

```
ecg/
  digitize.py    image  -> signal(s)   layout detect, per-lead trace, calibration
  delineate.py   signal -> fiducial points (P/QRS/T on- and offsets)
  parameters.py  fiducials -> seven parameters + quality flags
```

Data flow — two parallel branches off the digitized signal, not a chain:

```
ECG image --> digitize --+--> delineate --> parameters --> values + quality
                         +--> classifier --> class + confidence
```

The classifier never feeds the parameters. A CNN emits a class label; it cannot
emit intervals.

### Calibration

Millisecond intervals require paper speed (25 mm/s) and pixels-per-mm. The grid
is strongly periodic, so px/mm is derived from the dominant frequency of an FFT
over row and column projections. If grid detection fails, every time-based
parameter degrades to `—`. This is the highest-risk component in the design.

## Parameters

| Parameter | Derivation | Unavailable when |
|---|---|---|
| Heart Rate | 60 / mean(RR) | fewer than 3 R-peaks |
| Rhythm | RR coefficient of variation; Regular if CV < 0.10 | fewer than 5 beats |
| PR | P-onset to QRS-onset; P is the dominant positive deflection 80–300 ms before QRS-onset | P absent (e.g. atrial fibrillation) |
| QRS | Q-onset to S-offset by slope-threshold walk outward from R | noisy baseline |
| QT / QTc | Q-onset to T-end (tangent method on T downslope); Fridericia correction | T-end ambiguous |
| ST | Deviation at J+60 ms vs the PR-segment isoelectric baseline. Elevated if > +1.0 mm, Depressed if < -1.0 mm, else Normal; reported with the mm value | no calibration |
| Axis | `degrees(atan2(net_aVF, net_I))`, net = QRS area | single-lead input |

Axis is 12-lead-only by mathematics: one lead is a single projection of the
electrical vector, and a 2-D angle cannot be recovered from one projection. On
single-lead input Axis renders "Requires 12-lead", never a guessed number.

PR of `—` is frequently the correct clinical answer rather than a failure. The
current hardcoded 145 ms hides that.

### Quality flags

Every parameter carries `ok`, `low_confidence` or `unavailable`. Anything other
than `ok` renders as `—` with a reason, and is written to the PDF as
"Not measurable" rather than omitted, so a report never implies a measurement
that was not made.

## Sub-projects

Built in this order; each gets its own plan.

| | Sub-project | Depends on |
|---|---|---|
| A | Image to parameters | nothing |
| B | Synthetic ECG-image dataset generator | nothing |
| C | VGG16 image classifier plus validation gate | B |

A ships first: it is what makes the page trustworthy and is fully achievable
with data already on disk. A and B may proceed in parallel.

### B: dataset generation

Render MIT-BIH signals onto standard grid paper at 25 mm/s and 10 mm/mV, in
both single-lead and 3x4 twelve-lead layouts. Labels come from the `.atr`
annotation files, which is what makes the approach tractable. Augment for the
photo domain: perspective warp, shadow gradients, JPEG artifacts, blur,
rotation, moire, exposure variation.

### C: classifier and gate

VGG16 with ImageNet weights, frozen convolutional base and retrained head, then
the last block unfrozen at low learning rate — the pattern already used by
`train_vgg16.py`, which reaches 92.7% on MRI. Training applies class weights to
address the imbalance that produced the 0.285 macro F1.

The existing `_load_validated` gate (`app.py:339`) is extended to require a
minimum macro F1 from the model card. A model below threshold does not serve,
which prevents a repeat of the current "Unclassifiable beat at 100% confidence"
behaviour.

## Testing

- `delineate.py` is validated against MIT-BIH `.atr` annotations, which carry
  true R-peak positions, so detection sensitivity is measured rather than
  asserted.
- Digitization is tested round-trip: render a known signal to an image,
  digitize it back, assert correlation above 0.95.
- Parameters are range-checked against normal bounds (PR 120–200 ms,
  QRS 60–100 ms).
- Quality flags are tested by feeding deliberately degraded input and asserting
  that values become `—` rather than numbers.
