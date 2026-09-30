"""Per-study PDF content for ECG, EEG and MRI.

Each `*_report_spec(data)` takes the flat report dict a result page carries
(patient metadata + result fields) and returns everything build_pdf_report
needs. The same dict is what a saved study stores as its `report`, so the
download button, the stored per-study PDF and the complete history PDF are
built by one function per modality and cannot disagree.
"""
import json
import uuid
from datetime import datetime

from ecg.clinical import doctor_notes as ecg_doctor_notes
from eeg.interpretation import clinical_notes as eeg_clinical_notes
from reporting.pdf import build_pdf_report, doctor_sections, verdict_section
from vision.interpretation import clinical_notes as mri_clinical_notes, mri_verdict
from webapp.metadata import NOT_PROVIDED, PATIENT_FIELDS, _clean_text

EEG_TASKS = {"seizure": "Epileptiform / seizure activity",
             "alzheimer": "Cortical slowing (dementia screen)"}


def form_meta(form, fields=PATIENT_FIELDS):
    """Rebuild a metadata dict from the fields posted by a result page."""
    meta = {key: (_clean_text(form.get(key), 1000) or NOT_PROVIDED) for key, _ in fields}
    meta["report_id"] = _clean_text(form.get("report_id")) or ("RPT-" + uuid.uuid4().hex[:10].upper())
    meta["generated_at"] = _clean_text(form.get("generated_at")) or datetime.now().strftime("%d %b %Y, %H:%M:%S")
    return meta


def slug(value, fallback):
    """Filename-safe token derived from a metadata value."""
    if not value or value == NOT_PROVIDED:
        return fallback
    cleaned = "".join(c if c.isalnum() else "_" for c in value).strip("_")
    return cleaned[:40] or fallback


def pdf_value(raw):
    """Render an unmeasured parameter explicitly in the PDF.

    A dash in a clinical report is ambiguous - it could mean zero, or missing.
    "Not measurable" says which.
    """
    if raw is None or str(raw).strip() in {"", "—", "-"}:
        return "Not measurable"
    return raw


def _percent(raw):
    """'12.5%' for a value; 'Not measurable' for a blank, never a bare '%'."""
    value = pdf_value(raw)
    return value if value == "Not measurable" else f"{value}%"


def clinical_from_form(form):
    """Rebuild the structured reading from the measurements posted back.

    Returns None when the page posted no measurement payload -- an older
    result page, or a direct post -- so the report simply omits the section
    rather than inventing one.
    """
    raw = form.get("clinical_json")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "parameters" not in data:
        return None
    return data


def ecg_classification_rows(form):
    """Rows for the classification table, or an explicit statement of absence.

    A report must never imply a classification happened when it did not, so
    when no classifier ran the table carries the reason instead of a row of
    em-dashes that reads like a missing measurement.
    """
    prediction = (form.get("prediction") or "").strip()
    if not prediction:
        return [
            ("Parameter", "Result"),
            ("Overall Rhythm Classification", "Not performed"),
            ("Reason", form.get("classifier_error")
                       or "No validated ECG classifier was installed."),
            ("Signal Quality", form.get("signal_quality", NOT_PROVIDED)),
            ("Input Source", form.get("input_source", NOT_PROVIDED)),
        ]
    return [
        ("Parameter", "Result"),
        ("Overall Rhythm Classification", prediction),
        ("Abnormality Class", form.get("abnormal_type", NOT_PROVIDED)),
        ("Classifier", form.get("classifier", NOT_PROVIDED)),
        ("Classifier Confidence", _percent(form.get("confidence"))),
        ("Supraventricular (S) Beat Probability", _percent(form.get("atrial_probability"))),
        ("Signal Quality", form.get("signal_quality", NOT_PROVIDED)),
        ("Input Source", form.get("input_source", NOT_PROVIDED)),
    ]


def eeg_verdict(model_used, episodes, reason=None):
    """Overall NORMAL / ABNORMAL call for an EEG study.

    Without a model nothing was classified, so the answer is NOT ASSESSED:
    measurements alone never make a recording "normal".
    """
    if not model_used:
        return {"label": "NOT ASSESSED", "tone": "warn",
                "detail": reason or "No validated model installed; no window was classified."}
    if episodes:
        return {"label": "ABNORMAL", "tone": "bad",
                "detail": f"{episodes} anomalous episode(s) detected."}
    return {"label": "NORMAL", "tone": "good",
            "detail": "No episode crossed the detection threshold."}


# ------------------------------------------------------------------ ECG

def ecg_report_spec(form):
    # The structured reading is carried as JSON. It is display data echoed
    # back, not a re-measurement: the report can only be as trustworthy as
    # the page that produced it. It is escaped like every other field on the
    # way into the PDF, and a malformed or absent payload omits the section
    # rather than substituting anything.
    clinical = clinical_from_form(form)
    notes = ecg_doctor_notes(clinical, form.get("prediction"),
                             form.get("interpretation"),
                             form.get("recommendation"))
    sections = [
        verdict_section("ECG Classification", notes["verdict"]),
        ("text", "Clinical Impression", notes["impression"]),
        ("table", "1. Beat Classification Result", ecg_classification_rows(form)),
        ("table", "2. ECG Waveform Parameters", [
            ("Measurement", "Value"),
            ("Heart Rate", pdf_value(form.get("heart_rate"))),
            ("RR Interval", pdf_value(form.get("rr_interval"))),
            ("QRS Duration", pdf_value(form.get("qrs_duration"))),
            ("PR Interval", pdf_value(form.get("pr_interval"))),
            ("QT Interval", pdf_value(form.get("qt_interval"))),
            ("QTc (Corrected)", pdf_value(form.get("qtc"))),
            ("HRV SDNN", pdf_value(form.get("sdnn"))),
            ("HRV RMSSD", pdf_value(form.get("rmssd"))),
            ("Rhythm", pdf_value(form.get("rhythm"))),
            ("ST Segment", pdf_value(form.get("st_segment"))),
            ("QRS Axis", pdf_value(form.get("axis"))),
        ]),
    ]
    if clinical:
        sections.append(("table", "3. Parameters, Formulas & Reference Ranges",
                         [("Parameter", "Value / Range / Verdict")] +
                         [(r["label"],
                           f"{r['value']}  |  normal {r['range']}  |  "
                           f"{r['verdict'] or (r['reason'] or 'not measurable')}")
                          for r in clinical["parameters"]]))
        if clinical["status"]["findings"]:
            sections.append(("table", "4. Supporting Findings",
                             [("#", "Finding")] +
                             [(str(i + 1), f) for i, f in
                              enumerate(clinical["status"]["findings"])]))
    return {
        "title": "ECG ANALYSIS REPORT",
        "subtitle": "AI Diagnostic Summary — AAMI 5-class beat classifier + signal measurements",
        "accent": "#4f46e5",
        "sections": sections,
        "closing": doctor_sections(notes),
        "verdict": notes["verdict"],
        "disclaimer": (
            "Disclaimer: This report is generated by a research prototype and is not a "
            "medical device. All findings, measurements and recommendations are "
            "model-derived estimates and must be verified by a qualified cardiologist "
            "before any clinical decision is made."),
        "footer_text": "Unified AI Diagnostic Platform — ECG Analysis (research use only)",
        "file_prefix": "ECG_Report",
    }


# ------------------------------------------------------------------ MRI

def mri_report_spec(form):
    prediction = form.get("prediction", NOT_PROVIDED)
    notes = mri_clinical_notes(prediction, form)
    verdict = mri_verdict(prediction)
    sections = [
        verdict_section("MRI Classification", verdict),
        ("text", "Clinical Impression", notes["impression"]),
        ("table", "1. Classification Result", [
            ("Diagnostic Attribute", "AI System Evaluation"),
            ("Primary Tumor Classification", prediction),
            ("Model Confidence Score", _percent(form.get("confidence"))),
            ("Model", form.get("model_name") or NOT_PROVIDED),
        ]),
    ]
    if prediction and prediction != "No Tumor":
        sections.append(("table", "2. Model Attention (Grad-CAM)", [
            ("Attribute", "Value"),
            ("Peak Attention Location", form.get("location", NOT_PROVIDED)),
            ("Lesion Size / Severity",
             "Not measured — a classifier cannot measure lesion size; "
             "segmentation or radiologist measurement is required."),
        ]))
    sections.append(("text", "3. Radiological Findings Summary",
                     form.get("findings", NOT_PROVIDED)))
    return {
        "title": "BRAIN TUMOR MRI DIAGNOSTIC REPORT",
        "subtitle": "AI Multi-Class MRI Classification with Grad-CAM Localization",
        "accent": "#7c3aed",
        "sections": sections,
        "closing": doctor_sections(notes),
        "verdict": verdict,
        "disclaimer": (
            "Disclaimer: This report is generated by a research prototype and is not a "
            "medical device. The attention location is derived from model activation "
            "maps, not from radiological measurement, and must be validated by a "
            "board-certified radiologist."),
        "footer_text": "Unified AI Diagnostic Platform — Brain Tumor MRI Analysis (research use only)",
        "file_prefix": "MRI_Report",
    }


# ------------------------------------------------------------------ EEG

def eeg_report_spec(form):
    try:
        r = json.loads(form.get("eeg_json") or "")
        if not isinstance(r, dict):
            raise ValueError
    except ValueError:
        r = {}

    def v(key, fmt="{}"):
        return fmt.format(r[key]) if r.get(key) is not None else NOT_PROVIDED

    assessed = bool(r.get("model_used"))
    verdict = eeg_verdict(assessed, r.get("episodes_n") or 0, r.get("not_assessed_reason"))
    task_key = r.get("task_key") if r.get("task_key") in EEG_TASKS else "seizure"
    notes = eeg_clinical_notes(task_key, verdict["label"], r)
    sections = [
        verdict_section("EEG Classification", verdict),
        ("text", "Clinical Impression", notes["impression"]),
        ("table", "1. Recording & Analysis", [
            ("Attribute", "Value"),
            ("Clinical Question", v("task")),
            ("Input Source", v("source")),
            ("Representation", v("representation")),
            ("Duration", v("duration_s", "{} s")),
            ("Windows Analysed", v("n_windows")),
            ("Artifact Windows", v("artifact_windows")),
            ("Model Assessment", "Performed" if assessed else "Not assessed (no validated model)"),
        ]),
        ("table", "2. Findings", [
            ("Measure", "Value"),
            ("Detected Episodes", v("episodes_n") if assessed else "—"),
            ("Anomaly Burden", v("burden_pct", "{}%") if assessed else "—"),
            ("Mean Spike Rate", v("mean_spikes", "{} /s")),
            ("Highest Window Score",
             f"{r['peak_score']:.3f} at {r['peak_start_s']:.1f}–{r['peak_stop_s']:.1f} s"
             if assessed and r.get("peak_score") is not None else "—"),
        ]),
        ("text", "3. Automated Summary",
         " ".join(x for x in (r.get("headline"), r.get("peak_note")) if x) or NOT_PROVIDED),
    ]
    if r.get("band_means"):
        sections.append(("table", "4. Mean Relative Spectral Power",
                         [("Band", "Share of 0.5–45 Hz power")]
                         + [(str(b), str(x)) for b, x in r["band_means"].items()]))
    if r.get("episodes"):
        sections.append(("table", "5. Detected Episodes",
                         [("Onset / Offset (s)", "Duration (s) · Peak · Spikes/s")]
                         + [(f"{e['onset_s']:.1f} – {e['offset_s']:.1f}",
                             f"{e['duration_s']:.1f} · {e['peak_score']:.3f} · "
                             f"{e['spike_rate_per_s']:.2f}")
                            for e in r["episodes"]]))
    return {
        "title": "EEG ANALYSIS REPORT",
        "subtitle": "Windowed Spectral, Spike & Episode Analysis",
        "accent": "#0d9488",
        "sections": sections,
        "closing": doctor_sections(notes),
        "verdict": verdict,
        "disclaimer": (
            "Disclaimer: Research prototype, not a medical device. All values are "
            "model- or signal-derived. Channels are averaged, so findings are "
            "localised in time, not in space. \"Not assessed\" does not mean normal. "
            "Clinical review by a qualified neurologist is required."),
        "footer_text": "Unified AI Diagnostic Platform — EEG Analysis",
        "file_prefix": "EEG_Report",
    }


REPORT_SPECS = {"ecg": ecg_report_spec, "eeg": eeg_report_spec, "mri": mri_report_spec}


def render_study_pdf(modality, data, images=()):
    """(BytesIO, filename) for one study. images: [(caption, path)]."""
    spec = REPORT_SPECS[modality](data)
    meta = form_meta(data)
    image_sections = [("image", caption, path) for caption, path in images]
    buffer = build_pdf_report(
        title=spec["title"], subtitle=spec["subtitle"], accent=spec["accent"],
        meta=meta, meta_fields=PATIENT_FIELDS,
        meta_heading="Patient & Study Details",
        sections=spec["sections"] + image_sections + spec["closing"],
        disclaimer=spec["disclaimer"], footer_text=spec["footer_text"])
    name = slug(meta.get("patient_id"), slug(meta.get("patient_name"), "unidentified"))
    return buffer, f"{spec['file_prefix']}_{name}.pdf"
