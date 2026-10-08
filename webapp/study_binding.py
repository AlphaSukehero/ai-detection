"""Bind an ECG / EEG / MRI analysis to a patient's record and save it.

The record is the one the study was started from, or the one the Patient ID
typed with the details names (opened on first use).

The analysis routes stay as they were; with a patient bound they:
  1. take identity (name, ID, age, sex, phone) from the registry, not the
     form -- only per-visit fields are entered;
  2. after rendering the result, save it as a study, store its PDF, and
     compare it with the patient's previous study of the same modality.
A save failure never hides the result: the page says it was NOT saved.
"""
import json
import os
from datetime import date

from flask import current_app, request

from records import patients as registry
from records import studies
from records.compare import compare
from records.store import absolute
from reporting.followup import previous_details, previous_sections
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
    # A study with no date is filed under today; say so on the page and the
    # report too, rather than "Not provided" beside a record dated today.
    if not (data.get("study_date") or "").strip():
        data["study_date"] = date.today().isoformat()
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
        previous = studies.previous_study(conn, study)
        pdf, _name = render_stored_pdf(root, study, previous)
        studies.attach_file(conn, root, study, "pdf", pdf.getvalue(), ".pdf")
        study = studies.get_study(conn, study["id"])
    except Exception as e:  # the result is still shown; say it was not saved
        current_app.logger.exception("study save failed")
        return {"error": f"{type(e).__name__}: {e}"}
    return {"study": study, "previous": previous,
            "previous_report": previous_view(previous),
            "rows": compare(previous, study) if previous else []}


NO_PATIENT_ID = ("No Patient ID was entered, so this report is not kept. Enter "
                 "the same Patient ID on every visit and each report is saved "
                 "to that patient's record, with the previous one included.")


def record_for(source, form):
    """(patient or None, note) -- the record this request's result belongs to.

    A study started from a patient's record names it outright. Otherwise the
    Patient ID typed with the details decides: a known ID is that patient, a
    new one opens a record. With no ID there is nothing to file it under.
    """
    patient, _err = resolve(source)
    if patient:
        return patient, None
    typed = registry.clean_id(form.get("patient_id"))
    if not typed:
        return None, NO_PATIENT_ID
    sex = form.get("gender")
    patient, created = registry.find_or_create(
        get_conn(), typed, name=form.get("patient_name"),
        sex=sex if sex in GENDER_OPTIONS else None, phone=form.get("contact"))
    name = " ".join(str(form.get("patient_name") or "").split())
    note = None
    if not created and name and name.casefold() != patient["name"].casefold():
        note = (f"Patient ID {typed} is already on record as {patient['name']}. "
                "This study was added to that record; if this is a different "
                "person, use a different Patient ID.")
    return patient, note


def save_for_request(modality, source, form, **study):
    """Save this request's result to the record its patient details name.

    Returns what save_study_for returns, plus "note"; or {"unsaved": why}.
    """
    try:
        patient, note = record_for(source, form)
    except Exception as e:
        current_app.logger.exception("patient record lookup failed")
        return {"error": f"{type(e).__name__}: {e}"}
    if patient is None:
        return {"unsaved": note}
    saved = save_study_for(modality, patient, form, **study)
    saved["note"] = note
    return saved


def render_stored_pdf(root, study, previous):
    """(BytesIO, filename) of a stored study's report, previous one included."""
    extra = ()
    if previous:
        extra = previous_sections(previous, study, study_images(root, previous))
    return render_study_pdf(study["modality"], study["report"],
                            study_images(root, study), extra)


def previous_view(previous):
    """What a result page shows of the previous report, or None."""
    if not previous:
        return None
    base = f"/patients/{previous['patient_id']}/studies/{previous['id']}"
    return {"study_id": previous["id"], "study_date": previous["study_date"],
            "url": base, "details": previous_details(previous),
            "images": [{"caption": caption, "url": f"{base}/files/{kind}"}
                       for kind, caption in PDF_IMAGES[previous["modality"]]
                       if kind in previous["files"]]}


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


def stored_report(modality, patient_id, study_id):
    """(BytesIO, filename) of a saved study's report, or None if not saved.

    A result page that was saved downloads this rather than a report rebuilt
    from its form fields: only the stored study knows its previous report.
    """
    if not patient_id or not study_id:
        return None
    conn = get_conn()
    study = studies.get_study(conn, study_id)
    if (study is None or study["patient_id"] != patient_id
            or study["modality"] != modality):
        return None
    return render_stored_pdf(current_app.config["RECORDS_ROOT"], study,
                             studies.previous_study(conn, study))
