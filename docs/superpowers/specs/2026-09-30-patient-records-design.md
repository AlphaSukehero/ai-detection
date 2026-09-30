# Patient Records, History and Diagnostic Bug Fixes

Date: 2026-09-30
Status: Approved design, not yet implemented

## Problem

Two things, verified against the running app on 2026-09-30.

**1. Three analysis pages give wrong or unusable results.**

EEG (image upload):

- `eeg/digitize.py:extract_trace` treats the whole image as one trace. Per
  pixel column it averages the row positions of *all* ink — every channel of a
  montage, dotted grid lines, channel labels, timestamps. On a 10-channel
  screenshot the recovered "signal" is a centroid smear that matches no
  channel. Every spectral metric and model score is computed on that smear.

ECG (image upload):

- The timebase comes only from grid detection (`app.py:941-947`). With no grid,
  beat segmentation silently runs at 360 Hz and every interval is reported
  "unavailable". The form has no way to supply duration or paper speed.

Brain tumour MRI:

- **Wrong diagnosis on 16-bit input.** `load_image_rgb` calls
  `Image.convert("RGB")`, which clips 16-bit values to 255. A 16-bit copy of
  each tumour sample (glioma, meningioma, pituitary) returns
  "No Tumor 100.00%".
- **No input validation.** Random-noise images return "No Tumor 100.00%";
  20×20 images return confident, often wrong, classes.
- **Fabricated lesion metrics.** Area is the share of pixels above the 90th
  percentile of the Grad-CAM map, which is ~10% for every scan by
  construction (glioma 9.99%, meningioma 9.97%), so severity/spread always
  read "Moderate". A degenerate heatmap gives 100% and "High / Extensive"
  (pituitary sample). This repeats the class of defect removed in `2037e7c`.
- **Silent Grad-CAM failure.** Exceptions are printed; the page and PDF show
  zeros and "Not applicable".
- **PDF.** No scan or heatmap images; model hard-coded as "VGG16" whichever
  card was selected; report rebuilt from client-side hidden fields.
- A reported crash was not reproduced across 10 input formats; the input
  validation above converts unknown load failures into explicit refusals.

**2. Nothing is remembered.** Each analysis is standalone; the patient form
states "Nothing is stored on the server" and uploads are swept after a TTL. A
returning patient has no history, no comparison with previous studies, and no
consolidated report.

## Goals

1. Fix the EEG, ECG and MRI defects above, each with a regression test.
2. A single portal (no logins, no roles) where a patient is registered once,
   receives a portal-issued ID, and every later ECG/EEG/MRI study is stored
   under that ID.
3. Per patient: visit timeline, trend charts, side-by-side comparison of any
   two studies of the same modality, a "since last visit" panel on each new
   result, per-study PDFs, and one complete professional history PDF.

## Non-goals

- Authentication, user accounts, roles, audit of who viewed what.
- Patient portal / patient self-access.
- Multi-site or multi-server deployment (SQLite is single-host).
- A real tumour segmentation model. Lesion area/size is removed, not replaced.
- Editing or deleting stored studies from the UI.

## Decisions

| Question | Decision |
|---|---|
| Users | One shared portal, no login. Server binds `127.0.0.1` by default; `PORTAL_HOST` env var opens it to the network deliberately. |
| Patient identity | Portal issues sequential IDs `PT-000001`. Returning patients are found by search (ID, name, phone), never by retyping an ID. |
| What is stored | Everything: original upload, derived images, full result, measurements, generated PDF. |
| History features | Timeline, trends, compare-any-two, since-last-visit, cumulative history PDF. |
| Storage | SQLite (stdlib `sqlite3`, no ORM) + file store on disk. |

## Architecture

```
Browser ──► Flask app (app.py)
             ├─ patients blueprint (new)  register · search · patient page ·
             │                            timeline · trends · compare ·
             │                            history PDF
             ├─ /analyze_{ecg,eeg,brain_tumor}  accept optional patient_id
             │         │
             │         ▼
             │   records/ package (new)
             │     db.py        connection, schema, numbered migrations
             │     patients.py  register, get, search, update
             │     studies.py   save_study, list_studies, get_study,
             │                  previous_study(patient, modality, before)
             │     compare.py   diff two studies → rows with Δ and direction
             │     store.py     file store under records/<patient>/<study>/
             └─ reporting/
                   pdf.py       existing builder, gains image sections
                   history.py   cumulative history PDF (new)
```

New code lives in `records/` and a blueprint module, not in `app.py`
(currently 1,789 lines).

### Unit boundaries

- `records.db` knows SQL and nothing about modalities.
- `records.patients` / `records.studies` expose plain dicts; callers never
  see cursors or SQL.
- `records.compare` is pure: `(study_a, study_b) -> list[row]`. No I/O.
- `records.store` is the only module that touches `records/` on disk.
- Analysis routes produce a result dict exactly as today, then call one
  function, `studies.save_study(...)`, when a patient is bound.

## Data model

```
schema_version(version INTEGER)

patients
  id          TEXT PRIMARY KEY          -- "PT-000123"
  name        TEXT NOT NULL
  dob         TEXT                      -- ISO date; age computed per study
  sex         TEXT
  phone       TEXT
  address     TEXT
  notes       TEXT                      -- allergies, chronic conditions
  created_at  TEXT NOT NULL

studies
  id                   TEXT PRIMARY KEY -- "ST-20260930-0007"
  patient_id           TEXT NOT NULL REFERENCES patients(id)
  modality             TEXT NOT NULL    -- ecg | eeg | mri
  study_date           TEXT NOT NULL    -- entered; defaults to today
  created_at           TEXT NOT NULL
  referring_physician  TEXT
  clinical_history     TEXT
  verdict              TEXT             -- NORMAL | ABNORMAL | NOT ASSESSED
  headline             TEXT
  model_name           TEXT
  model_version        TEXT             -- card "trained" date
  result_json          TEXT NOT NULL    -- full result dict the page rendered
  reanalysis_of        TEXT REFERENCES studies(id)

measurements
  study_id  TEXT NOT NULL REFERENCES studies(id)
  key       TEXT NOT NULL
  value     REAL                        -- NULL when not measurable
  unit      TEXT
  PRIMARY KEY (study_id, key)

study_files
  study_id  TEXT NOT NULL REFERENCES studies(id)
  kind      TEXT NOT NULL               -- original | waveform | heatmap |
                                        -- highlight | timeline | pdf
  path      TEXT NOT NULL               -- relative to records root
```

- IDs are allocated inside the insert transaction, so concurrent
  registrations cannot collide.
- Studies are append-only. Re-analysis creates a new study with
  `reanalysis_of` set.
- Measurement keys per modality are a fixed list in `records/studies.py`
  (e.g. ECG: `heart_rate, pr_interval, qrs_duration, qt_interval, qtc`;
  EEG: `burden_pct, episodes, mean_spike_rate, rsp_delta … rsp_gamma`;
  MRI: `mri_confidence`, `p_glioma`, `p_meningioma`, `p_notumor`,
  `p_pituitary`). Measurements are numeric only; the MRI class is text and
  lives in `verdict`/`headline`/`result_json`. An unmeasurable value is
  stored as NULL, never 0.
- Migrations are numbered functions applied in order at startup.

## Screens and flows

- **Home → Patients**: search box (ID, name, phone); **Register patient**.
- **Register**: name (required), DOB, sex, phone, address, notes → issues
  ID, redirects to the patient page.
- **Patient page** header: ID, name, age, sex, notes. Tabs:
  - **Timeline**: studies newest first — date, modality, verdict badge,
    headline, *Open*, *PDF*.
  - **Trends**: one chart per key measurement over time, reference range
    shaded; MRI shown as class-over-time strip plus confidence.
  - **Compare**: choose two studies of one modality; images side by side
    and a table with a Δ column and direction (e.g. "QTc 430 → 482 ms ▲,
    now prolonged"; "MRI: No Tumor → Glioma").
  - **New study**: ECG / EEG / MRI buttons opening the existing pages with
    `?patient=PT-…`.
- **Analysis pages with a patient bound**: patient block is read-only; only
  per-visit fields (study date, referring physician, clinical history) are
  entered. The result adds a **Since last visit** panel when a previous
  study of that modality exists. The study is saved automatically.
- **Analysis pages without a patient**: unchanged quick-analysis mode;
  nothing is stored.
- **Open study**: re-renders the stored result through the same template
  from `result_json` and stored files.
- **PDFs**:
  - Per-study PDF is built from the stored study (not hidden form fields),
    includes the source image and derived images, and names the model from
    the stored card.
  - **Complete Patient History PDF**: cover page, patient details, table of
    contents, visit timeline table, trend charts, per-modality comparison
    table (first vs latest and latest vs previous), then every study report
    in full; page numbers and a consistent header/footer.

## Bug fixes

### EEG image digitisation

- Detect channel bands from the row ink profile after removing grid lines
  (columns/rows whose ink spans most of the image) and label margins.
- Extract one trace per band; analyse each channel; average only for the
  single timeline summary, as the EDF path already does.
- If no clean band is found, or bands overlap, raise
  `CalibrationError` with a specific reason. Single-trace images keep working.

### ECG image timebase

- Add *Recording duration (s)* and *Paper speed (25 / 50 mm/s)* to the ECG
  form, shown for image uploads.
- Timebase precedence: stated duration → detected grid × paper speed →
  refuse interval measurement. Remove the silent 360 Hz fallback for images.

### MRI

- Load 16-bit and float images by percentile windowing (as the DICOM path
  already does), not by `convert("RGB")`.
- Input gate before classification: minimum 64×64; reject strongly colour
  images (mean channel spread above a threshold); reject noise-like images
  (high-frequency energy share above a threshold). Refusal names the reason.
- Remove area / width / height / severity / spread. Report Grad-CAM
  activation location only, labelled as an attention map, not a lesion
  measurement.
- Grad-CAM failure is surfaced on the page ("attention map unavailable:
  <reason>").
- Model name in page and PDF comes from the selected card.

## Error handling

- `save_study` runs in one transaction. Files are written to a temp folder,
  then moved into `records/<patient>/<study>/` only after the DB commit
  succeeds; on failure the temp folder is removed and the page shows
  "Result shown but NOT saved: <reason>".
- The upload sweeper continues to clean `uploads/` only; `records/` is never
  swept.
- An unknown patient ID in a URL returns 404 with a link back to search.
- A missing stored file renders the study with a "file missing" placeholder
  rather than failing the page.

## Testing

- Regression tests:
  - A 16-bit glioma sample classifies as Glioma.
  - A noise image and a 20×20 image are refused.
  - The lesion-area fields are absent from the page and the PDF.
  - A synthetic 10-channel montage digitises to 10 traces.
  - An ECG image with a stated duration yields measured intervals.
- `records` unit tests against a temp DB:
  - ID allocation.
  - Search.
  - `save_study` atomicity (injected failure leaves no rows or files).
  - `previous_study`.
  - `compare` diffs.
- Route tests: register → ECG study → second ECG study → since-last-visit
  panel → compare → history PDF opens and contains both studies.
- The existing suite stays green; CI unchanged apart from new tests.

## Delivery order

1. EEG, ECG and MRI bug fixes (separate commits, independent of records).
2. `records` package + migrations + unit tests.
3. Patients blueprint: register, search, patient page, timeline.
4. Bind analysis routes to patients; save studies; per-study PDFs from
   stored data.
5. Since-last-visit, compare, trends.
6. Complete history PDF.
