"""Bind an ECG / EEG / MRI analysis to a registered patient and save it.

The analysis routes stay as they were; with a patient bound they:
  1. take identity (name, ID, age, sex, phone) from the registry, not the
     form -- only per-visit fields are entered;
  2. after rendering the result, save it as a study, store its PDF, and
     compare it with the patient's previous study of the same modality.
A save failure never hides the result: the page says it was NOT saved.
"""
import json
import os

from flask import current_app, request

from records import patients as registry
from records import studies
from records.compare import compare
from records.store import absolute
from reporting.study_reports import render_study_pdf
from webapp.metadata import GENDER_OPTIONS
from webapp.patients import get_conn

VISIT_FIELDS = ("study_date", "referring_physician", "clinical_history")

# Images each modality embeds in its PDF, in order: (file kind, caption).
PDF_IMAGES = {
    "ecg": [("waveform", "ECG Signal Waveform")],
    "eeg": [("timeline", "Spectral Power & Anomaly Timeline")],
    "mri": [("scan", "MRI Scan"), ("heatmap", "Grad-CAM Attention Map"),
            ("highlight", "Highest-Attention Region")],
}


def _requested_id(source):
    # `record_id` (form) / `patient` (URL), never `patient_id`: that name is
    # the free-text MRN of a quick analysis and must not trigger a lookup.
    return " ".join(str(source.get("record_id") or source.get("patient") or "").split())


def resolve(source):
    """(patient or None, error or None) for the patient a request names."""
    pid = _requested_id(source)
    if not pid:
        return None, None
    p = registry.get(get_conn(), pid)
    if p is None:
        return None, f"No patient with ID {pid}. Search the registry and start the study from the patient's record."
    return p, None


def bound_context():
    """Template context: the patient this page is bound to, if any."""
    if request.endpoint not in ("ecg", "eeg", "brain_tumor", "analyze_ecg",
                                "analyze_eeg", "analyze_brain_tumor"):
        return {}
    source = request.form if request.method == "POST" else request.args
    p, _err = resolve(source)
    return {"bound": p}


def patient_form(form):
    """(form dict with identity from the registry, error or None)."""
    p, err = resolve(form)
    data = {k: form.get(k) for k in form.keys()}
    if err:
        return data, err
    if p is None:
        return data, None
    study_date = form.get("study_date") or None
    age = registry.age_on(p.get("dob"), study_date) if study_date else p.get("age")
    data.update({
        "patient_name": p["name"],
        "patient_id": p["id"],
        "age": "" if age is None else str(age),
        "gender": p["sex"] if p.get("sex") in GENDER_OPTIONS else "",
        "contact": p.get("phone") or "",
    })
    return data, None


# ------------------------------------------------------------ measurements

def _ms(m):
    return None if m is None or m.value is None else m.value * 1000.0


def _num(text):
    try:
        return float(str(text).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None


def ecg_measurements(report, beat_result):
    def val(key):
        m = report.get(key)
        return None if m is None else m.value
    return [
        ("heart_rate", val("heart_rate"), "bpm"),
        ("rr_interval", _ms(report.get("rr_interval")), "ms"),
        ("pr_interval", _ms(report.get("pr_interval")), "ms"),
        ("qrs_duration", _ms(report.get("qrs_duration")), "ms"),
        ("qt_interval", _ms(report.get("qt_interval")), "ms"),
        ("qtc", _ms(report.get("qtc")), "ms"),
        ("axis", val("axis"), "°"),
        ("sdnn", _ms(report.get("sdnn")), "ms"),
        ("rmssd", _ms(report.get("rmssd")), "ms"),
        ("classifier_confidence",
         _num(beat_result["confidence"]) if beat_result else None, "%"),
    ]


def eeg_measurements(report):
    bands = report.get("band_means") or {}
    rows = [
        ("burden_pct", report.get("burden_pct") if report.get("model_used") else None, "%"),
        ("episodes", report.get("episodes_n") if report.get("model_used") else None, ""),
        ("mean_spike_rate", report.get("mean_spikes"), "/s"),
        ("peak_score", report.get("peak_score") if report.get("model_used") else None, ""),
        ("duration_s", report.get("duration_s"), "s"),
    ]
    rows += [(f"rsp_{b}", bands.get(b), "") for b in
             ("delta", "theta", "alpha", "beta", "gamma")]
    return rows


MRI_PROB_KEYS = {"Glioma": "p_glioma", "Meningioma": "p_meningioma",
                 "No Tumor": "p_notumor", "Pituitary": "p_pituitary"}


def mri_measurements(result):
    rows = [("mri_confidence", _num(result["confidence"]), "%")]
    rows += [(key, result["raw_probabilities"].get(cls), "%")
             for cls, key in MRI_PROB_KEYS.items()]
    return rows


# --------------------------------------------------------------- saving

def report_fields(meta, fields):
    """The flat dict a result page would post to its PDF download."""
    out = {k: v for k, v in meta.items()}
    out.update({k: ("" if v is None else v) for k, v in fields.items()})
    return out


def save_study_for(modality, patient, form, *, result, report, verdict,
                   headline, model_name, model_version, measurements, files):
    """Save, attach the PDF, compare with the previous study.

    Returns {"study", "previous", "rows"} or {"error": reason}.
    """
    root = current_app.config["RECORDS_ROOT"]
    conn = get_conn()
    try:
        study = studies.save_study(
            conn, root, patient["id"], modality,
            visit={k: form.get(k) for k in VISIT_FIELDS},
            result=result, report=report, verdict=verdict, headline=headline,
            model_name=model_name, model_version=model_version,
            measurements=measurements,
            files={k: v for k, v in files.items() if v})
        pdf, _name = render_study_pdf(modality, study["report"],
                                      study_images(root, study))
        studies.attach_file(conn, root, study, "pdf", pdf.getvalue(), ".pdf")
        study = studies.get_study(conn, study["id"])
    except Exception as e:  # the result is still shown; say it was not saved
        current_app.logger.exception("study save failed")
        return {"error": f"{type(e).__name__}: {e}"}
    previous = studies.previous_study(conn, study)
    return {"study": study, "previous": previous,
            "rows": compare(previous, study) if previous else []}


def study_images(root, study):
    out = []
    for kind, caption in PDF_IMAGES[study["modality"]]:
        rel = study["files"].get(kind)
        if rel:
            path = absolute(root, rel)
            if os.path.isfile(path):
                out.append((caption, path))
    return out


def dumps(obj):
    return json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o))
