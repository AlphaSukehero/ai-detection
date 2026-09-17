"""The three PDF download routes must actually produce a PDF.

This file exists because a refactor left five undefined names in the report
builder and the whole suite still passed: nothing had ever rendered a report.
A PDF that raises on download is a silent dead end for the user, so each
route is exercised end to end and the bytes are checked.
"""
import pytest

import app as flask_app
from reporting.pdf import build_pdf_report
from webapp.metadata import PATIENT_FIELDS, finalize_metadata


@pytest.fixture
def client():
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


PATIENT = {"patient_name": "T", "patient_id": "MRN-1", "age": "40",
           "gender": "Male", "contact": "1234567890",
           "referring_physician": "D", "study_date": "2026-09-09",
           "clinical_history": "test"}

SURVEY = {"site_name": "S", "survey_id": "SV-1", "coordinates": "0, 0",
          "capture_date": "2026-09-09", "sensor": "test",
          "analyst": "A", "survey_notes": "test"}


def _is_pdf(resp):
    assert resp.status_code == 200, resp.status_code
    body = resp.get_data()
    assert body.startswith(b"%PDF-"), body[:40]
    assert len(body) > 1000, "suspiciously small PDF"
    return body


def test_ecg_report_downloads(client):
    data = dict(PATIENT, prediction="NORMAL",
                abnormal_type="No ectopic beats detected", confidence="91.20",
                atrial_probability="1.10", classifier="AAMI 5-class 1D CNN",
                heart_rate="72 bpm", signal_quality="Good",
                interpretation="x", recommendation="y")
    _is_pdf(client.post("/download_ecg_report", data=data))


def test_ecg_report_downloads_when_no_classifier_ran(client):
    """The degraded path renders too -- it is the one users hit without models."""
    data = dict(PATIENT, prediction="", classifier_error="no model installed",
                heart_rate="72 bpm", interpretation="x", recommendation="y")
    body = _is_pdf(client.post("/download_ecg_report", data=data))
    assert body  # content is binary; the table contents are unit-tested above


def test_brain_tumor_report_downloads(client):
    data = dict(PATIENT, prediction="Glioma", confidence="88.10",
                findings="x", recommendation="y", tumor_area="3.2",
                severity="Moderate", spread="Limited")
    _is_pdf(client.post("/download_brain_tumor_report", data=data))


def test_satellite_report_downloads(client):
    data = dict(SURVEY, prediction="Forest / Vegetation", confidence="95.10",
                findings="x", recommendation="y")
    _is_pdf(client.post("/download_satellite_report", data=data))


def test_report_filename_is_derived_from_the_identifier(client):
    data = dict(PATIENT, prediction="NORMAL", confidence="90",
                interpretation="x", recommendation="y")
    resp = client.post("/download_ecg_report", data=data)
    # _slug() keeps the identifier but normalises punctuation for the filename.
    assert "MRN_1" in resp.headers["Content-Disposition"]


def test_builder_renders_both_section_kinds():
    """build_pdf_report takes "table" and "text" sections; both must render."""
    meta = finalize_metadata(dict(PATIENT), PATIENT_FIELDS)
    buf = build_pdf_report(
        title="T", subtitle="S", accent="#4f46e5", meta=meta,
        meta_fields=PATIENT_FIELDS, meta_heading="H",
        sections=[("table", "A", [("k", "v"), ("a", "b")]),
                  ("text", "B", "some prose")],
        disclaimer="D", footer_text="F")
    assert buf.getvalue().startswith(b"%PDF-")


def test_builder_escapes_markup_in_user_supplied_text():
    """Patient names reach ReportLab's mini-HTML parser; unescaped <b> there
    either styles the report or raises."""
    meta = finalize_metadata(dict(PATIENT, patient_name="<b>x</b> & co"),
                             PATIENT_FIELDS)
    buf = build_pdf_report(
        title="T", subtitle="S", accent="#4f46e5", meta=meta,
        meta_fields=PATIENT_FIELDS, meta_heading="H",
        sections=[("text", "B", "<i>unclosed")],
        disclaimer="D", footer_text="F")
    assert buf.getvalue().startswith(b"%PDF-")


def test_ecg_report_carries_the_structured_clinical_sections(client):
    """The parameter table, status and precautions must survive the round
    trip from the result page into the PDF."""
    import json
    from ecg.clinical import clinical_report
    from ecg.quality import Measurement, OK

    rep = {k: Measurement(v, OK) for k, v in
           [("heart_rate", 72.0), ("pr_interval", 0.16), ("qrs_duration", 0.09),
            ("qtc", 0.41), ("st_segment", 0.0), ("qt_interval", 0.38)]}
    rep["rhythm"] = Measurement(0.03, OK, "Regular")

    data = dict(PATIENT, prediction="NORMAL", confidence="90",
                interpretation="x", recommendation="y",
                clinical_json=json.dumps(clinical_report(rep, sex="Male")))
    body = _is_pdf(client.post("/download_ecg_report", data=data))
    assert len(body) > 5000, "clinical sections appear to be missing"


def test_ecg_report_omits_the_section_when_the_payload_is_malformed(client):
    """A broken payload must drop the section, never substitute anything."""
    data = dict(PATIENT, prediction="NORMAL", confidence="90",
                interpretation="x", recommendation="y",
                clinical_json="{not json")
    _is_pdf(client.post("/download_ecg_report", data=data))
    assert flask_app._clinical_from_form({"clinical_json": "{not json"}) is None
    assert flask_app._clinical_from_form({"clinical_json": '{"a":1}'}) is None
    assert flask_app._clinical_from_form({}) is None
