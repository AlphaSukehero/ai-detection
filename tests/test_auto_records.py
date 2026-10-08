"""A quick analysis with a Patient ID is saved to that patient's record, and
a follow-up report carries the previous report of the same kind: its image,
its details and what changed."""
import re

import pytest

import app as flask_app
from tests.conftest_records import register, rclient  # noqa: F401
from tests.test_eeg_csv import _csv_upload
from tests.test_pdf_reports import _pdf_text

WHO = {"patient_name": "Ravi Kumar", "patient_id": "MRN-8821", "age": "52",
       "gender": "Male", "contact": "98450 00000",
       "referring_physician": "Dr Mehta", "clinical_history": "Palpitations"}


def _ecg(client, date, sample="normal", **who):
    data = dict(WHO, sample_type=sample, study_date=date)
    data.update(who)
    return client.post("/analyze_ecg", data=data,
                       content_type="multipart/form-data").get_data(as_text=True)


def _saved(html):
    """(patient id, study id) of the study a result page says it saved."""
    m = re.search(r"Saved to record</strong> as <a href=\"/patients/([^/]+)/studies/(ST-\d{8}-\d{4})\"", html)
    assert m, "the page does not say the study was saved"
    return m.group(1), m.group(2)


def _pdf(client, pid, sid):
    resp = client.get(f"/patients/{pid}/studies/{sid}/pdf")
    assert resp.status_code == 200 and resp.mimetype == "application/pdf"
    return resp.data


def _n_images(pdf):
    return pdf.count(b"/Subtype /Image")


def test_quick_analysis_with_a_patient_id_is_saved(rclient):
    pid, _sid = _saved(_ecg(rclient, "2026-08-01"))
    listing = rclient.get("/patients").get_data(as_text=True)
    assert "Ravi Kumar" in listing and "MRN-8821" in listing
    record = rclient.get(f"/patients/{pid}").get_data(as_text=True)
    assert "Ravi Kumar" in record and "MRN-8821" in record


def test_without_a_patient_id_nothing_is_stored_and_the_page_says_so(rclient):
    html = _ecg(rclient, "2026-08-01", patient_id="")
    assert "Saved to record" not in html
    assert "Not stored" in html and "Patient ID" in html
    assert "No patients registered yet" in rclient.get("/patients").get_data(as_text=True)


def test_the_same_id_again_is_the_same_patient(rclient):
    first, _ = _saved(_ecg(rclient, "2026-08-01"))
    second, _ = _saved(_ecg(rclient, "2026-09-01", patient_id="  mrn-8821 "))
    assert first == second
    assert rclient.get("/patients").get_data(as_text=True).count("Ravi Kumar") == 1


def test_a_different_id_is_a_different_patient(rclient):
    first, _ = _saved(_ecg(rclient, "2026-08-01"))
    other, _ = _saved(_ecg(rclient, "2026-08-01", patient_id="MRN-9000",
                           patient_name="Sita Devi"))
    assert first != other


def test_a_registered_patients_id_typed_in_reaches_their_record(rclient):
    pid = register(rclient)
    saved_to, _ = _saved(_ecg(rclient, "2026-08-01", patient_id=pid,
                              patient_name="Asha Rao"))
    assert saved_to == pid


def test_a_known_id_under_another_name_is_flagged(rclient):
    _ecg(rclient, "2026-08-01")
    html = _ecg(rclient, "2026-09-01", patient_name="Someone Else")
    _saved(html)
    assert "already on record as Ravi Kumar" in html


def test_first_report_has_no_previous_report(rclient):
    html = _ecg(rclient, "2026-08-01")
    pid, sid = _saved(html)
    assert "Previous ECG report" not in html
    assert b"Previous ECG Report" not in _pdf_text(_pdf(rclient, pid, sid))


def test_second_report_page_shows_the_first_reports_image_and_details(rclient):
    pid, first = _saved(_ecg(rclient, "2026-08-01"))
    html = _ecg(rclient, "2026-09-01", sample="abnormal")
    assert "Previous ECG report" in html
    assert f"/patients/{pid}/studies/{first}/files/waveform" in html
    assert "2026-08-01" in html and first in html
    assert "Since last visit" in html


def test_second_report_pdf_includes_the_first_reports_image_and_details(rclient):
    pid, first = _saved(_ecg(rclient, "2026-08-01"))
    _pid, second = _saved(_ecg(rclient, "2026-09-01", sample="abnormal"))
    pdf = _pdf(rclient, pid, second)
    text = _pdf_text(pdf)
    assert b"Previous ECG Report" in text
    assert first.encode() in text and b"2026-08-01" in text
    assert b"Change Since Previous Report" in text
    # One picture more than the first report: the first report's waveform.
    assert _n_images(pdf) == _n_images(_pdf(rclient, pid, first)) + 1


def test_third_report_includes_the_second_not_the_first(rclient):
    pid, first = _saved(_ecg(rclient, "2026-07-01"))
    _p, second = _saved(_ecg(rclient, "2026-08-01"))
    _p, third = _saved(_ecg(rclient, "2026-09-01"))
    text = _pdf_text(_pdf(rclient, pid, third))
    assert second.encode() in text
    assert first.encode() not in text


def test_download_button_on_a_saved_result_gives_the_stored_report(rclient):
    pid, _first = _saved(_ecg(rclient, "2026-08-01"))
    html = _ecg(rclient, "2026-09-01")
    _pid, second = _saved(html)
    form = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', html))
    assert form.get("saved_study") == second
    resp = rclient.post("/download_ecg_report", data=form)
    assert resp.status_code == 200 and resp.mimetype == "application/pdf"
    assert b"Previous ECG Report" in _pdf_text(resp.data)


def test_eeg_reports_are_kept_apart_from_ecg_reports(rclient):
    pid, _sid = _saved(_ecg(rclient, "2026-08-01"))
    data = dict(WHO, sampling_rate="128", task="seizure", study_date="2026-09-01",
                eeg_file=(_csv_upload(), "rec.csv"))
    html = rclient.post("/analyze_eeg", data=data,
                        content_type="multipart/form-data").get_data(as_text=True)
    same, _eeg_sid = _saved(html)
    assert same == pid
    assert "Previous EEG report" not in html and "Previous ECG report" not in html


def test_eeg_api_returns_the_previous_report(rclient):
    def post(date):
        data = dict(WHO, sampling_rate="128", task="seizure", study_date=date,
                    eeg_file=(_csv_upload(), "rec.csv"))
        return rclient.post("/api/eeg/analyze", data=data,
                            content_type="multipart/form-data").get_json()
    first = post("2026-08-01")["saved"]
    second = post("2026-09-01")
    prev = second["saved"]["previous_report"]
    assert prev["study_id"] == first["study_id"] and prev["study_date"] == "2026-08-01"
    assert prev["images"] and prev["images"][0]["url"].endswith("/files/timeline")
    assert second["report_form"]["saved_study"] == second["saved"]["study_id"]


def test_eeg_api_says_when_nothing_was_stored(rclient):
    data = dict(WHO, patient_id="", sampling_rate="128", task="seizure",
                eeg_file=(_csv_upload(), "rec.csv"))
    body = rclient.post("/api/eeg/analyze", data=data,
                        content_type="multipart/form-data").get_json()
    assert "Patient ID" in body["saved"]["unsaved"]
