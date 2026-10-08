"""Analyses done for a registered patient are saved to that patient's record."""
import io
import re

import numpy as np
from PIL import Image

import app as flask_app
from tests.conftest_records import register, rclient  # noqa: F401
from tests.test_pdf_reports import _pdf_text
from tests.test_routes import requires_mri_model

VISIT = {"study_date": "2026-09-01", "referring_physician": "Dr Mehta",
         "clinical_history": "Palpitations"}


def _ecg(client, pid, date="2026-09-01", sample="normal"):
    data = dict(VISIT, record_id=pid, sample_type=sample, study_date=date)
    return client.post("/analyze_ecg", data=data,
                       content_type="multipart/form-data").get_data(as_text=True)


def _study_ids(html):
    return re.findall(r"ST-\d{8}-\d{4}", html)


def test_bound_page_shows_the_patient_instead_of_the_identity_form(rclient):
    pid = register(rclient)
    html = rclient.get(f"/ecg?patient={pid}").get_data(as_text=True)
    assert "Asha Rao" in html and pid in html
    assert f'name="record_id" value="{pid}"' in html
    assert 'id="patient_name"' not in html
    assert 'name="referring_physician"' in html


def test_unknown_patient_is_refused(rclient):
    html = _ecg(rclient, "PT-999999")
    assert "No patient with ID PT-999999" in html
    assert "ECG Waveform Parameters" not in html


def test_ecg_for_a_patient_is_saved_with_measurements(rclient):
    pid = register(rclient)
    html = _ecg(rclient, pid)
    assert "Saved to record" in html
    sid = _study_ids(html)[0]
    timeline = rclient.get(f"/patients/{pid}").get_data(as_text=True)
    assert sid in timeline and "No studies yet" not in timeline
    from records import db, studies
    conn = db.connect(flask_app.app.config["RECORDS_ROOT"])
    s = studies.get_study(conn, sid)
    assert s["modality"] == "ecg" and s["study_date"] == "2026-09-01"
    assert s["referring_physician"] == "Dr Mehta"
    hr = s["measurements"]["heart_rate"]["value"]
    assert hr is None or 20 < hr < 250
    assert {"original", "waveform", "pdf"} <= set(s["files"])


def test_patient_identity_comes_from_the_registry(rclient):
    pid = register(rclient)
    html = _ecg(rclient, pid)
    assert "Asha Rao" in html and pid in html


def test_second_study_shows_since_last_visit(rclient):
    pid = register(rclient)
    first = _study_ids(_ecg(rclient, pid, "2026-08-01"))[0]
    html = _ecg(rclient, pid, "2026-09-01", sample="abnormal")
    assert "Since last visit" in html and first in html
    assert "Heart rate" in html


def test_first_study_has_no_since_last_visit(rclient):
    pid = register(rclient)
    assert "Since last visit" not in _ecg(rclient, pid)


def test_stored_study_pdf_is_the_patients_report(rclient):
    pid = register(rclient)
    sid = _study_ids(_ecg(rclient, pid))[0]
    resp = rclient.get(f"/patients/{pid}/studies/{sid}/pdf")
    assert resp.status_code == 200 and resp.mimetype == "application/pdf"
    text = _pdf_text(resp.data)
    assert b"ECG ANALYSIS REPORT" in text and b"Asha Rao" in text


def test_study_page_shows_the_stored_result(rclient):
    pid = register(rclient)
    sid = _study_ids(_ecg(rclient, pid))[0]
    html = rclient.get(f"/patients/{pid}/studies/{sid}").get_data(as_text=True)
    assert sid in html and "Heart rate" in html and "Dr Mehta" in html
    assert f"/patients/{pid}/studies/{sid}/files/waveform" in html
    assert rclient.get(f"/patients/{pid}/studies/{sid}/files/waveform").status_code == 200


def test_study_of_another_patient_is_404(rclient):
    a = register(rclient)
    b = register(rclient, name="Ben")
    sid = _study_ids(_ecg(rclient, a))[0]
    assert rclient.get(f"/patients/{b}/studies/{sid}").status_code == 404


def test_analysis_without_a_patient_saves_nothing(rclient):
    data = dict(VISIT, sample_type="normal", patient_name="X")
    html = rclient.post("/analyze_ecg", data=data,
                        content_type="multipart/form-data").get_data(as_text=True)
    assert "Saved to record" not in html
    assert "No patients registered yet" in rclient.get("/patients").get_data(as_text=True)


def test_eeg_for_a_patient_is_saved(rclient):
    from tests.test_eeg_route import _trace_png
    pid = register(rclient)
    data = dict(VISIT, record_id=pid, duration="20", task="seizure",
                eeg_file=(_trace_png(), "trace.png"))
    html = rclient.post("/analyze_eeg", data=data,
                        content_type="multipart/form-data").get_data(as_text=True)
    assert "Saved to record" in html, html[html.find("Error"):][:300]
    sid = _study_ids(html)[0]
    pdf = rclient.get(f"/patients/{pid}/studies/{sid}/pdf")
    assert b"EEG ANALYSIS REPORT" in _pdf_text(pdf.data)


@requires_mri_model
def test_mri_for_a_patient_is_saved_with_images_in_the_pdf(rclient):
    pid = register(rclient)
    data = dict(VISIT, record_id=pid,
                mri_file=(open("static/samples/mri_glioma.jpg", "rb"), "g.jpg"))
    html = rclient.post("/analyze_brain_tumor", data=data,
                        content_type="multipart/form-data").get_data(as_text=True)
    assert "Saved to record" in html
    sid = _study_ids(html)[0]
    from records import db, studies
    s = studies.get_study(db.connect(flask_app.app.config["RECORDS_ROOT"]), sid)
    assert s["headline"].startswith("Glioma")
    assert s["measurements"]["p_glioma"]["value"] > 50
    assert {"original", "scan", "heatmap"} <= set(s["files"])
    pdf = rclient.get(f"/patients/{pid}/studies/{sid}/pdf").data
    assert b"/Subtype /Image" in pdf


def test_a_typed_mrn_in_quick_analysis_is_never_refused_as_unknown(rclient):
    """`record_id` must exist; a typed Patient ID opens a record instead."""
    data = dict(VISIT, sample_type="normal", patient_name="X", patient_id="MRN-8821")
    html = rclient.post("/analyze_ecg", data=data,
                        content_type="multipart/form-data").get_data(as_text=True)
    assert "No patient with ID" not in html
    assert "ECG Waveform Parameters" in html and "MRN-8821" in html
    assert "Saved to record" in html
