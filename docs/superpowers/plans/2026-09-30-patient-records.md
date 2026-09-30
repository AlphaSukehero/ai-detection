# Patient Records + Diagnostic Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the EEG/ECG/MRI defects in the spec and add a single-portal patient records system with timeline, trends, comparison and a complete history PDF.

**Architecture:** New `records/` package (stdlib sqlite3 + on-disk file store) behind plain-dict functions; a `patients` Flask blueprint for registry pages; the three analysis routes gain an optional `patient_id` and call `records.studies.save_study`. PDFs are rebuilt from stored studies.

**Tech Stack:** Flask, sqlite3, numpy, Pillow, matplotlib, ReportLab, pytest.

Spec: `docs/superpowers/specs/2026-09-30-patient-records-design.md`

## Global Constraints

- No authentication. Server binds `127.0.0.1` unless `PORTAL_HOST` is set.
- Patient IDs `PT-000001` (6 digits, sequential). Study IDs `ST-YYYYMMDD-NNNN`.
- Studies are append-only. Unmeasurable values are NULL, never 0.
- `records/` data root is `RECORDS_ROOT` env var, default `records_data/` (gitignored). Tests use a tmp dir.
- Analysis pages without a patient keep working and store nothing.
- No fabricated numbers: removed MRI area/size/severity/spread must not reappear in page or PDF.
- Existing suite (323 tests) stays green; `ruff check .` clean.

## File Structure

| File | Responsibility |
|---|---|
| `eeg/digitize.py` (modify) | Grid removal, channel-band detection, per-channel extraction |
| `app.py` `_eeg_to_signal` (modify) | Average per-channel image traces like the EDF path |
| `ecg/digitize.py`, `app.py:analyze_ecg`, `templates/ecg.html` (modify) | Stated duration / paper speed timebase |
| `vision/imageio.py` (create) | `load_mri_rgb` (16-bit aware) + `check_mri_input` gate |
| `app.py` MRI route, `templates/brain_tumor.html`, `vision/interpretation.py` (modify) | Remove fabricated metrics; surface Grad-CAM errors; model name from card |
| `records/db.py` | Connection, migrations |
| `records/patients.py` | register / get / search |
| `records/studies.py` | save_study / list / get / previous / measurements extraction |
| `records/store.py` | File store |
| `records/compare.py` | Pure diff of two studies |
| `webapp/patients.py` | Blueprint: register, search, patient page, study view, compare, trends image, history PDF |
| `reporting/pdf.py` (modify) | `image` section kind; build returns story helpers |
| `reporting/history.py` | Complete patient-history PDF |
| `templates/patients/*.html` | search, register, patient, compare |

## Tasks

### Task 1: EEG multi-channel image digitisation
- Test (`tests/test_eeg_digitize.py`): synthetic 10-band montage with dotted vertical grid lines and label text boxes → `digitize_channels(img, duration_s=10)` returns 10 traces; each trace correlates > 0.9 with its source sine; single-trace image still returns 1 channel; blank image raises `CalibrationError`.
- Implement `remove_grid(gray)`, `find_channel_bands(gray) -> list[(r0, r1)]`, `digitize_channels(path, duration_s=None, px_per_mm=None, paper_speed_mm_s=30.0, target_fs=None) -> (signals[n_ch, n], fs, source, n_channels)`. `digitize` keeps its signature and returns the channel mean.
- `_eeg_to_signal` uses `digitize_channels`; provenance states channel count.
- Commit: `fix(eeg): digitise each channel of a montage image separately`.

### Task 2: ECG stated timebase
- Test (`tests/test_ecg_route.py`): gridless synthetic ECG image + `ecg_duration=10` → page shows a measured heart rate (not "unavailable"); gridless without duration → page states the timebase is missing (no silent 360 Hz); form contains `name="ecg_duration"` and `name="paper_speed"`.
- Implement: `analyze_ecg` reads `ecg_duration` (s) and `paper_speed` (25/50); precedence stated duration → `px_per_mm × paper_speed` → none. With none, `prepare_beats` is skipped (no classification) and parameters report unavailable. `ecg.parameters.analyse` gains `fs_override`.
- Commit: `fix(ecg): accept recording duration and paper speed for image uploads`.

### Task 3: MRI loading, input gate, honest metrics
- Tests (`tests/test_mri_input.py`): 16-bit copy of glioma sample → `load_mri_rgb` output correlates > 0.95 with 8-bit version; `check_mri_input` refuses noise, 20×20, and a saturated colour image, accepts the four samples. Route: 16-bit glioma → "Glioma" (skip without model); noise → refusal text; page and PDF contain no "Area of Activation", "Severity", "Spread".
- Implement `vision/imageio.py`; route uses it; remove area/width/height/severity/spread from route, template, PDF and `mri_clinical_notes`; Grad-CAM exception → `result["attention_error"]`; `result["model_name"]` from card path; PDF uses it. Update `tests/test_mri_interpretation.py`, keep `tests/test_tumor_geometry.py` (helpers still exist in `vision.gradcam`).
- Commit: `fix(mri): load 16-bit scans, refuse non-MRI input, drop fabricated lesion metrics`.

### Task 4: records package
- Tests (`tests/test_records.py`, tmp `RECORDS_ROOT`): register issues `PT-000001`, `PT-000002`; search by id/name substring/phone; `save_study` stores row, measurements, files; injected failure (bad file path) leaves no study row and no folder; `previous_study` returns latest earlier same-modality study; `list_studies` newest first; `compare` returns rows with delta and direction and flags class change.
- Implement `records/db.py` (`connect(root)`, `migrate(conn)`), `records/patients.py` (`register(conn, **fields) -> dict`, `get(conn, pid)`, `search(conn, q)`), `records/store.py` (`study_dir(root, pid, sid)`), `records/studies.py` (`MEASUREMENT_KEYS`, `extract_measurements(modality, report) -> list[(key, value, unit)]`, `save_study(root, patient_id, modality, visit, result, report, files) -> dict`, `list_studies`, `get_study`, `previous_study`, `measurement_series(conn, pid, modality)`), `records/compare.py` (`compare(a, b) -> list[dict]`).
- Commit: `feat(records): patient registry and study store on SQLite`.

### Task 5: patients blueprint + UI
- Route tests (`tests/test_patients_routes.py`): register form → redirect to `/patients/PT-000001`; search finds; unknown id → 404; patient page lists studies; nav contains `/patients`.
- Implement `webapp/patients.py` blueprint and templates; register in `app.py`; `PORTAL_HOST` binding.
- Commit: `feat(patients): registry pages — register, search, patient timeline`.

### Task 6: bind analyses to patients
- Tests: ECG sample with `patient_id` saves a study and shows "Saved to record"; second ECG shows "Since last visit"; per-study PDF from `/patients/<pid>/studies/<sid>/pdf` is a PDF containing the patient name; analysis without patient saves nothing.
- Implement: each analyze route reads `patient_id`; when present, fills patient fields from the registry, saves the study (original + images copied), renders since-last-visit via `compare`. Per-study PDF built from stored `report` via the existing per-modality section builders refactored to take a dict.
- Commit: `feat(records): save ECG, EEG and MRI studies to the patient record`.

### Task 7: compare, trends, history PDF
- Tests: compare page for two ECG studies shows Δ; trends PNG endpoint returns image/png; history PDF has ≥ N pages and contains each study ID.
- Implement compare view, trends via matplotlib, `reporting/history.py`.
- Commit: `feat(records): study comparison, trend charts and complete history PDF`.

### Task 8: verification
- Full `pytest -q`, `ruff check .`, run app, click through with `/browse`.
